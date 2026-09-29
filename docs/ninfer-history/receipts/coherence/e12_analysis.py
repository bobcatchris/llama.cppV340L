#!/usr/bin/env python3
"""E-11/E-12 analyzer (BUGTRACK NVFP4@TP4 incoherence) — zero-GPU, stdlib only.

Part A (COH-ALLOC tables): splits each alloc_<base>.log into request epochs at
  MARK REQ base=0 T=<plen> lines; for the T=66 (PASS) vs T=67 (FAIL) epochs of the
  same arena it (a) diffs the (bytes,align) allocation SEQUENCE (first divergence
  names the T-conditional allocation), (b) lists REUSE events (POP restored=1 with a
  later alloc landing inside the restored span while the request epoch is live).

Part B (COH-OPS dumps): for each stage, per-token blocks are compared between legs.
  X0 (mixer entry) defines the CONTENT-aligned prefix K (rows 0..K are identical
  tokens+embeddings across legs — E-5: rows 0..58 for this body). Within 0..K the
  FIRST stage whose bytes differ convicts its organ:
    X0 -> (sanity, always identical within K)
    HN -> input rmsnorm | G0/B0 -> control proj | QK -> input projection GEMV
    QV/ZB -> tp_unpack_gdn_qkvz | QC/KC/VC -> causal conv | GB/BB -> gbeta unpack
    O0 -> gated_delta_net | ON -> gated rmsnorm | PA -> out_proj GEMV
    XA -> allreduce fold | XL -> mlp_tail

Usage: e12_analysis.py <coh_dir>   # /tmp/coh_e12 (alloc_*.log + dumps/ + resp_*.json)
"""
import glob
import json
import os
import re
import struct
import sys

# units = 16-bit words per token block as they land in the MBP1 payload:
# bf16 tensors dump 2 bytes/elem; FP32 tensors (g/beta) dump 4 bytes/elem = 2 words.
# world=4 shard dims (MEASURED from the E12b dump n values, not from comments):
# k_rows=512, v_rows=1536, qkv_rows=2560, qkvz_rows=4096, hidden=5120, v_loc=12.
ROWS = {"X0": 5120, "HN": 5120, "PA": 5120, "XA": 5120, "XL": 5120,
        "QK": 4096, "QV": 2560, "ZB": 1536, "VC": 1536, "O0": 1536, "ON": 1536,
        "QC": 512, "KC": 512,
        "G0": 96, "B0": 96, "GB": 24, "BB": 24}
# dump-n T vocabulary (PASS T), prompt-plen legs: 64=T@plen66 PASS, 65=T@plen67 FAIL,
# 96=T@plen98 FAIL, 51=warmup PASS glass.
T_PASS, T_FAIL, T_FAIL2, T_WARM = 64, 65, 96, 51
PIPELINE = ["X0", "HN", "G0", "B0", "QK", "QV", "ZB", "QC", "KC", "VC",
            "GB", "BB", "O0", "ON", "PA", "XA", "XL"]


def part_a(coh):
    print("=" * 72)
    print("PART A — arena allocation tables (owner-rank filtered epochs)")
    for path in sorted(glob.glob(os.path.join(coh, "alloc_*.log"))):
        with open(path) as f:
            lines = f.read().splitlines()
        if not lines:
            continue
        name = os.path.basename(path)
        m = re.match(r"OPEN base=0x([0-9a-f]+) cap=(\d+) owns=(\d+) pid=(\d+)", lines[0])
        base = int(m.group(1), 16) if m else None
        # owner rank: the OWNER mark whose work base equals this arena's base
        owner = None
        for ln in lines:
            mo = re.match(r"MARK OWNER R=(-?\d+) work=0x([0-9a-f]+)$", ln)
            if mo and base is not None and int(mo.group(2), 16) == base:
                owner = int(mo.group(1))
                break
        if owner is None:
            print(f"\n-- {name} ({lines[0]})\n   no OWNER handshake match — not a work arena; skipped")
            continue
        # epochs = owner's REQ marks; non-owner marks dropped (4 rank threads interleave)
        epochs, cur = [], None
        for ln in lines:
            mr = re.match(r"MARK REQ R=(-?\d+) base=\d+ T=(\d+)$", ln)
            if mr:
                if int(mr.group(1)) == owner:
                    cur = {"T": int(mr.group(2)), "lines": []}
                    epochs.append(cur)
                continue
            if ln.startswith("MARK"):
                continue
            if cur is not None:
                cur["lines"].append(ln)
        by_t = {}
        for e in epochs:
            by_t.setdefault(e["T"], []).append(e)
        print(f"\n-- {name} ({lines[0]}) owner=R{owner}")
        print(f"   epochs: {sorted((t, len(v)) for t, v in by_t.items())}")
        # allocation sequence diff T=66 vs T=67 (first occurrence of each).
        # T-proportional buffers change by ~67/66 — EXPECTED. Anomalies: sequence-length
        # mismatch, non-proportional size changes, alignment changes.
        if 66 in by_t and 67 in by_t:
            def seq(e):
                out = []
                for ln in e["lines"]:
                    ma = re.match(r"A seq=\d+ off=0x[0-9a-f]+ bytes=(\d+) align=(\d+) ", ln)
                    if ma:
                        out.append((int(ma.group(1)), int(ma.group(2))))
                return out
            s66, s67 = seq(by_t[66][0]), seq(by_t[67][0])
            print(f"   alloc-seq T66 n={len(s66)} vs T67 n={len(s67)}")
            if len(s66) != len(s67):
                n = min(len(s66), len(s67))
                div = next((i for i in range(n) if s66[i] != s67[i]), n)
                print(f"   *** SEQUENCE-LENGTH MISMATCH — first structural divergence at idx {div}")
                for i in range(max(0, div - 2), min(max(div + 2, n), max(len(s66), len(s67)))):
                    a = s66[i] if i < len(s66) else None
                    b = s67[i] if i < len(s67) else None
                    print(f"     [{i}] T66={a}  T67={b}")
            else:
                ratio = 67 / 66
                anom = []
                for i, ((b66, a66), (b67, a67)) in enumerate(zip(s66, s67)):
                    if a66 != a67:
                        anom.append((i, "ALIGN", b66, b67))
                    elif b66 == b67:
                        continue
                    else:
                        r = b67 / b66
                        if abs(r - ratio) > 0.02:
                            anom.append((i, f"RATIO {r:.4f} != {ratio:.4f}", b66, b67))
                print(f"   anomalies (non-proportional / align): {len(anom)}")
                for i, kind, b66, b67 in anom[:10]:
                    print(f"     [{i}] {kind}: T66 bytes={b66}  T67 bytes={b67}")
        # reuse scan per owner epoch; RESET invalidates everything — clears.
        for t, eps in sorted(by_t.items()):
            for ei, e in enumerate(eps):
                stack, reuse = [], []
                for ln in e["lines"]:
                    if ln.startswith("RESET"):
                        stack.clear()
                        continue
                    mp = re.match(r"POP seq=\d+ saved=0x([0-9a-f]+) before=0x([0-9a-f]+) restored=1", ln)
                    if mp:
                        stack.append((int(mp.group(1), 16), int(mp.group(2), 16)))
                        continue
                    ma = re.match(r"A seq=\d+ off=0x([0-9a-f]+) bytes=(\d+) ", ln)
                    if ma and stack:
                        off, by = int(ma.group(1), 16), int(ma.group(2))
                        hit = None
                        for span in stack:
                            lo, hi = span
                            if lo <= off and off + by <= hi:
                                hit = span
                                break
                        if hit is not None:
                            stack.remove(hit)
                            reuse.append((off, by, hit[0], hit[1]))
                if reuse:
                    print(f"   REUSE events T={t} epoch#{ei}: {len(reuse)}"
                          f" (first: alloc off=0x{reuse[0][0]:x} bytes={reuse[0][1]}"
                          f" inside restored [0x{reuse[0][2]:x},0x{reuse[0][3]:x}))")


def read_mbp1(path):
    with open(path, "rb") as f:
        hdr = f.read(20)
        if len(hdr) < 20:
            return None
        magic, ver, sha, cnt = struct.unpack("<IIQI", hdr)
        if magic != 0x3150424D:
            return None
        payload = f.read(cnt * 4)
    return sha, payload


def aligned_prefix(b66, b67, r):
    """Count leading per-token blocks that are byte-identical between legs."""
    nb66, nb67 = len(b66) // (r * 2), len(b67) // (r * 2)
    k = 0
    while k < min(nb66, nb67) and b66[k * r * 2:(k + 1) * r * 2] == b67[k * r * 2:(k + 1) * r * 2]:
        k += 1
    return k, nb66, nb67


def first_div_in_prefix(b66, b67, r, kmax):
    """First token block index < kmax whose bytes differ, else None."""
    for i in range(kmax):
        if b66[i * r * 2:(i + 1) * r * 2] != b67[i * r * 2:(i + 1) * r * 2]:
            return i
    return None


def part_b(coh):
    print("\n" + "=" * 72)
    print("PART B — per-op bisect (layer-0 gidx0 prefill dumps)")
    files = glob.glob(os.path.join(coh, "dumps", "*.bin"))
    if not files:
        print("   no dump files found")
        return
    legs = {}  # (phase, T, rank) -> list of (pass_no, path)
    for p in files:
        base = os.path.basename(p)[:-4]
        m = re.match(r"([A-Z0-9]+)_r(\d+)_p(\d+)_l(\d+)_n(\d+)_s([0-9a-f]+)$", base)
        if not m:
            continue
        ph, rr, ps, rank, n, _sha = m.groups()
        rows = ROWS.get(ph)
        if rows is None or int(n) % rows:
            continue
        legs.setdefault((ph, int(n) // rows, int(rank)), []).append((int(ps), p))
    ranks = sorted({k[2] for k in legs})
    print(f"   ranks seen: {ranks}")
    for rank in ranks:
        # X0 defines the CONTENT-aligned prefix (identical tokens+embeddings): E-5
        # measured rows 0..58 identical for this body, first tokenization diff at 59.
        x66 = sorted(legs.get(("X0", T_PASS, rank), []))
        x67 = sorted(legs.get(("X0", T_FAIL, rank), []))
        x96 = sorted(legs.get(("X0", T_FAIL2, rank), []))
        if not x66 or not x67:
            print(f"\n-- rank {rank}: X0 missing ({bool(x66)},{bool(x67)}) — cannot align")
            continue
        d66, d67 = read_mbp1(x66[0][1]), read_mbp1(x67[0][1])
        K6667, nb66, nb67 = aligned_prefix(d66[1], d67[1], ROWS["X0"])
        print(f"\n-- rank {rank}: content-aligned prefix K(66 vs 67) = {K6667} "
              f"(blocks {nb66}/{nb67}; E-5 expects ~59)")
        print("   stage   blocks66/67   first_div(0..K)  verdict")
        for ph in PIPELINE:
            r = ROWS[ph]
            l66 = sorted(legs.get((ph, T_PASS, rank), []))
            l67 = sorted(legs.get((ph, T_FAIL, rank), []))
            if not l66 or not l67:
                print(f"   {ph:6}  MISSING leg (64:{bool(l66)} 65:{bool(l67)})")
                continue
            a, b = read_mbp1(l66[0][1]), read_mbp1(l67[0][1])
            if not a or not b:
                print(f"   {ph:6}  UNREADABLE")
                continue
            if ph == "X0":
                print(f"   {ph:6}  {nb66}/{nb67}     (alignment reference)")
                continue
            div = first_div_in_prefix(a[1], b[1], r, min(K6667, nb66, nb67))
            verd = "CONVICTS THIS ORGAN" if div is not None else "clean within prefix"
            print(f"   {ph:6}  {len(a[1])//(r*2)}/{len(b[1])//(r*2)}      div={div}   {verd}")
        # E-8 cross-check: T=67 vs T=96 — 'ALL T>=65 share ONE wrong computation' says
        # early rows should be BIT-IDENTICAL between these two FAIL legs.
        if x96:
            d96 = read_mbp1(x96[0][1])
            if d96:
                K6796, n67b, n96b = aligned_prefix(d67[1], d96[1], ROWS["X0"])
                print(f"   cross-check X0: aligned prefix K(67 vs 96) = {K6796} "
                      f"(E-8 law: early rows bit-identical across T>=65 ⇒ large K)")
                for ph in ("XA", "O0", "QK"):
                    r = ROWS[ph]
                    l67 = sorted(legs.get((ph, T_FAIL, rank), []))
                    l96 = sorted(legs.get((ph, T_FAIL2, rank), []))
                    if l67 and l96:
                        a, b = read_mbp1(l67[0][1]), read_mbp1(l96[0][1])
                        if a and b:
                            div = first_div_in_prefix(a[1], b[1], r,
                                                      min(K6796, len(a[1]) // (r * 2),
                                                          len(b[1]) // (r * 2)))
                            print(f"   cross-check {ph}: first_div(67 vs 96 within K) = {div}")


def verdicts(coh):
    print("\n" + "=" * 72)
    print("LEG VERDICTS (from resp_*.json)")
    for p in sorted(glob.glob(os.path.join(coh, "resp_*.json"))):
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"   {os.path.basename(p)}: PARSE_FAIL {e}")
            continue
        c = (d.get("choices") or [{}])[0]
        m = m0 = c.get("message") or {}
        u = d.get("usage") or {}
        print(f"   {os.path.basename(p)}: prompt={u.get('prompt_tokens')} gen={u.get('completion_tokens')}"
              f" content[:32]={(m.get('content') or '')[:32]!r}")


if __name__ == "__main__":
    coh = sys.argv[1] if len(sys.argv) > 1 else "/tmp/coh_e12"
    part_a(coh)
    part_b(coh)
    verdicts(coh)
