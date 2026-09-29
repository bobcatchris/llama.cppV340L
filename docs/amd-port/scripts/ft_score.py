#!/usr/bin/env python3
"""Fixed-text acceptance scorer (W34 instrument).

Reads the per-request JSONL emitted by the FT-CASCADE / FT-GEN passes of
run_fixedtext_acceptance.sh and reports:

  summarize <cell.jsonl>            per-cell acceptance table
  compare   <ctrl.jsonl> <arm.jsonl>  cross-arm fixed-text comparison

Cascade records carry round1_accepted / round1_proposed (parsed from the
per-round server log lines "accepted k/n draft tokens", LLAMA_TRACE=1).
p_i = P(round1_accepted >= i) over cascade requests: the probability the
draft-mtp head's top-i prefix survives greedy verify at a TRUE frozen
history point. p1 is the pure head-quality hit rate; p2/p3 add the
draft-conditioned chain terms.

Laws folded in:
  L3 / E-139 cont: acceptance deltas across arms are population draws
    UNLESS measured on identical text. Cascade prompts are frozen token
    ids (prompt side), so no trajectory divergence can enter.
  T-10: per-text acceptance varies enormously by text kind -> report
    per-kind and require cross-item sign consistency for a compare
    verdict, not just a pooled delta.
"""
import argparse
import json
import math
import sys


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def load(path):
    recs = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def cascade(recs):
    return [r for r in recs if r.get("mode") == "cascade" and r.get("round1_proposed")]


def gen(recs):
    return [r for r in recs if r.get("mode") == "gen"]


def pos_table(recs, max_pos=4, key=None):
    """Return {pos: (k, n, p, lo, hi)} for round1 accepted-prefix counts."""
    out = {}
    sub = recs if key is None else [r for r in recs if key(r)]
    for i in range(1, max_pos + 1):
        sel = [r for r in sub if r["round1_proposed"] >= i]
        k = sum(1 for r in sel if r["round1_accepted"] >= i)
        n = len(sel)
        p, lo, hi = wilson(k, n)
        out[i] = (k, n, p, lo, hi)
    return out


def fmt_pos(tbl):
    rows = ["  pos   hits /    n   p_hat   wilson95"]
    for i in sorted(tbl):
        k, n, p, lo, hi = tbl[i]
        if n == 0:
            continue
        rows.append("  p%d    %4d / %4d   %.3f   [%.3f, %.3f]" % (i, k, n, p, lo, hi))
    return "\n".join(rows)


def summarize(path):
    recs = load(path)
    cas, gn = cascade(recs), gen(recs)
    if not recs:
        print("no records in %s" % path)
        return 1

    if cas:
        print("FT-CASCADE (fixed-text, round-1 off true history):")
        print(fmt_pos(pos_table(cas)))
        kinds = sorted({r.get("kind", "?") for r in cas})
        for kd in kinds:
            sub = [r for r in cas if r.get("kind") == kd]
            t = pos_table(sub)
            parts = []
            for i in sorted(t):
                k, n, p, lo, hi = t[i]
                if n:
                    parts.append("p%d=%d/%d=%.3f" % (i, k, n, p))
            print("  kind %-11s n=%3d  %s" % (kd, len(sub), "  ".join(parts) if parts else "-"))
        # per-item p1 (T-10 spread witness)
        items = sorted({r.get("item", "?") for r in cas})
        spread = []
        for it in items:
            sub = [r for r in cas if r.get("item") == it]
            k = sum(1 for r in sub if r["round1_accepted"] >= 1)
            spread.append("%s=%d/%d" % (it, k, len(sub)))
        print("  per-item p1 spread (T-10 witness): " + "  ".join(spread))
        print()

    if gn:
        print("FT-GEN (served acceptance + frozen-text identity):")
        for r in gn:
            dn, da = r.get("draft_n", 0), r.get("draft_n_accepted", 0)
            ar = (da / dn) if dn else 0.0
            print("  %-10s kind=%-11s accept=%.4f (%d/%d) predicted_n=%s sha=%s"
                  % (r.get("item"), r.get("kind"), ar, da, dn,
                     r.get("predicted_n"), str(r.get("text_sha256"))[:16]))
    return 0


def compare(ctrl_path, arm_path):
    ctrl, arm = load(ctrl_path), load(arm_path)
    cc, ca = cascade(ctrl), cascade(arm)
    if not cc or not ca:
        print("COMPARE-FAIL: cascade records missing (ctrl=%d arm=%d)" % (len(cc), len(ca)))
        return 2

    print("FIXED-TEXT ACCEPTANCE COMPARE (identical frozen prefixes both arms)")
    tc, ta = pos_table(cc), pos_table(ca)
    print("  pos   ctrl (k/n)           arm (k/n)           delta    CI-separate?")
    # union of positions: a chain-4 arm proposes p4 that a chain-3 control
    # cannot - that new position is the marginal-token datum, so it must
    # print (ctrl side "n/a") rather than be dropped by an intersection.
    for i in sorted(set(tc) | set(ta)):
        kc, nc, pc, loc, hic = tc.get(i, (0, 0, 0.0, 0.0, 1.0))
        ka, na, pa, loa, hia = ta.get(i, (0, 0, 0.0, 0.0, 1.0))
        if nc == 0 and na == 0:
            continue
        cs = ("%3d/%-4d %.3f [%.2f,%.2f]" % (kc, nc, pc, loc, hic)) if nc else "      n/a            "
        as_ = ("%3d/%-4d %.3f [%.2f,%.2f]" % (ka, na, pa, loa, hia)) if na else "      n/a            "
        sep = (nc and na) and ((loa > hic) or (loc > hia))
        print("  p%d    %s  %s  %+0.3f   %s"
              % (i, cs, as_, (pa - pc) if (nc and na) else 0.0,
                 "YES" if sep else "no"))

    # per-item p1 sign consistency (T-10: items are the independent units)
    items = sorted({r.get("item") for r in cc} & {r.get("item") for r in ca})
    per_item = []
    n_up = n_down = n_flat = 0
    for it in items:
        sc = [r for r in cc if r.get("item") == it]
        sa = [r for r in ca if r.get("item") == it]
        pc1 = wilson(sum(1 for r in sc if r["round1_accepted"] >= 1), len(sc))[0]
        pa1 = wilson(sum(1 for r in sa if r["round1_accepted"] >= 1), len(sa))[0]
        d = pa1 - pc1
        if abs(d) < 1e-9:
            n_flat += 1
        elif d > 0:
            n_up += 1
        else:
            n_down += 1
        per_item.append("%s:%+.3f" % (it, d))
    print("  per-item p1 deltas: " + "  ".join(per_item))
    print("  item sign consistency: up=%d down=%d flat=%d (of %d items)"
          % (n_up, n_down, n_flat, len(items)))

    # served-text identity (gen records): same text across arms?
    gc, ga = gen(ctrl), gen(arm)
    if gc and ga:
        mc = {r.get("item"): r.get("text_sha256") for r in gc}
        ma = {r.get("item"): r.get("text_sha256") for r in ga}
        mism = [k for k in mc if k in ma and mc[k] != ma[k]]
        print("  served-text sha: %d items ctrl, %d arm, mismatches=%d %s"
              % (len(mc), len(ma), len(mism), ("(IDENTICAL)" if not mism else str(mism))))
        if mism:
            print("  NOTE: served text diverged across arms (L3 in the wild). The cascade")
            print("        table above is still valid: its text is frozen on the prompt side.")

    nc_, na_ = tc.get(1, (0, 0))[1], ta.get(1, (0, 0))[1]
    nmax_ = max(nc_, na_, 1)
    print()
    print("VERDICT GUIDE: pooled p1 delta beyond the Wilson bands AND consistent")
    print("per-item sign (%d items) = head-quality signal; inside the bands = NULL" % len(items))
    print("(with n~%d/%d per arm the p1 95%% band is roughly +/-%.2f; see receipt s6"
          % (nc_, na_, max(0.0, 0.5 * 1.96 / math.sqrt(nmax_))))
    return 0


def main():
    ap = argparse.ArgumentParser(description="fixed-text acceptance scorer (W34)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("summarize", help="per-cell acceptance table")
    s1.add_argument("cell_jsonl")
    s2 = sub.add_parser("compare", help="ctrl vs arm fixed-text comparison")
    s2.add_argument("ctrl_jsonl")
    s2.add_argument("arm_jsonl")
    args = ap.parse_args()
    if args.cmd == "summarize":
        return summarize(args.cell_jsonl)
    return compare(args.ctrl_jsonl, args.arm_jsonl)


if __name__ == "__main__":
    sys.exit(main())
