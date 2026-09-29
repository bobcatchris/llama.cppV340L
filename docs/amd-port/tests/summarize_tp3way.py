#!/usr/bin/env python3
"""summarize_tp3way.py - aggregate TP3 three-way arm reps into the savings/perf table.

Scans docs/amd-port/results/ for tp3way_<arm><rep>_<stamp>_* artifact sets:
  _phases.txt, _{preboot,boot_ready,postprobe,postteardown}.json, _battery.jsonl,
  _server.log, _vram.log.

Prints a markdown table per phase and the derived B-vs-A / C-vs-A deltas, plus a
per-rep contamination audit (foreign prompts, retries, foreign VRAM on card3).
"""

import glob
import json
import os
import re
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MIB = 1024.0 * 1024.0
CARDS = ("card0", "card1", "card2", "card3")


def load_snap(path):
    d = json.load(open(path))
    out = {}
    for c in CARDS:
        cd = d.get(c, {})
        t = cd.get("VRAM Total Memory (B)")
        u = cd.get("VRAM Total Used Memory (B)")
        if t and u:
            out[c] = (int(t) / MIB, int(u) / MIB)
    return out


def parse_phases(path):
    boot_at = None
    lock_note = ""
    for line in open(path):
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] BOOT_READY", line)
        if m:
            boot_at = m.group(1)
        if "LOCK" in line:
            lock_note = line.strip()
    return boot_at, lock_note


def parse_battery(path):
    receipt = json.loads(open(path).read().strip().splitlines()[-1])
    rows = {r.get("cell"): r for r in receipt.get("results", [])}
    return receipt, rows


def parse_server_log(path):
    accept_lines = []
    n_prompts = 0
    n_exceeded = 0
    n_retry = 0
    if not os.path.isfile(path):
        return accept_lines, n_prompts, n_exceeded, n_retry
    for line in open(path, errors="replace"):
        if "new prompt" in line:
            n_prompts += 1
        if "Context size has been exceeded" in line:
            n_exceeded += 1
        if "retrying with smaller batch size" in line:
            n_retry += 1
        m = re.search(
            r"(\d\d:\d\d:\d\d).*draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)",
            line,
        )
        if m:
            accept_lines.append((m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))))
    return accept_lines, n_prompts, n_exceeded, n_retry


def vram_foreign(path, arm):
    """Max card3 used MiB and the max serving-die sample outside this rep's own
    server residency are the foreign-VRAM signals; simple card3 check for A/B."""
    mx3 = 0.0
    if not os.path.isfile(path):
        return mx3
    for line in open(path):
        if line.startswith("#") or line.startswith("ts_iso"):
            continue
        parts = line.strip().split(",")
        if len(parts) >= 5 and parts[2] == "card3":
            try:
                mx3 = max(mx3, float(parts[3]))
            except ValueError:
                pass
    return mx3


def fmt(v, nd=1):
    return f"{v:.{nd}f}" if v is not None else "-"


def main():
    sets = {}
    for phases in sorted(glob.glob(os.path.join(RESULTS, "tp3way_*_phases.txt"))):
        base = phases[: -len("_phases.txt")]
        m = re.match(r".*tp3way_(off|noflag|iso)(\d+)_(\d{8}_\d{6})$", base)
        if not m:
            continue
        arm, rep, stamp = m.group(1), m.group(2), m.group(3)
        sets[(arm, rep, stamp)] = base

    reps = []
    for (arm, rep, stamp), base in sorted(sets.items()):
        snap = {}
        for ph in ("preboot", "boot_ready", "postprobe", "postteardown"):
            p = f"{base}_{ph}.json"
            snap[ph] = load_snap(p) if os.path.isfile(p) else {}
        boot_at, lock_note = parse_phases(f"{base}_phases.txt")
        receipt, rows = (parse_battery(f"{base}_battery.jsonl")
                         if os.path.isfile(f"{base}_battery.jsonl") else (None, {}))
        acc, n_prompts, n_exceeded, n_retry = parse_server_log(f"{base}_server.log")
        mx3 = vram_foreign(f"{base}_vram.log", arm)
        reps.append(dict(arm=arm, rep=rep, stamp=stamp, snap=snap, rows=rows,
                         receipt=receipt, boot_at=boot_at, acc=acc, n_prompts=n_prompts,
                         n_exceeded=n_exceeded, n_retry=n_retry, mx3=mx3, base=base,
                         lock_note=lock_note))

    if not reps:
        print("no tp3way artifact sets found")
        return 1

    print("## Boot-ready per-die VRAM (used / free MiB)\n")
    print("| arm/rep | boot ready at | die 0 | die 1 | die 2 | die 3 |")
    print("|---|---|---|---|---|---|")
    for r in reps:
        s = r["snap"]["boot_ready"]
        cells = []
        for c in CARDS:
            if c in s:
                t, u = s[c]
                cells.append(f"{u:.1f} / {t-u:.1f}")
            else:
                cells.append("-")
        print(f"| {r['arm']}{r['rep']} | {r['boot_at'] or '-'} | " + " | ".join(cells) + " |")

    print("\n## Post-probe per-die VRAM (used / free MiB)\n")
    print("| arm/rep | die 0 | die 1 | die 2 | die 3 |")
    print("|---|---|---|---|---|")
    for r in reps:
        s = r["snap"]["postprobe"]
        cells = []
        for c in CARDS:
            if c in s:
                t, u = s[c]
                cells.append(f"{u:.1f} / {t-u:.1f}")
            else:
                cells.append("-")
        print(f"| {r['arm']}{r['rep']} | " + " | ".join(cells) + " |")

    # derived savings: mean per serving die vs arm A mean
    def mean_delta(phase, ref_arm, tgt_arm):
        out = {}
        for i, c in enumerate(CARDS[:3]):
            ref = [r["snap"][phase][c][1] for r in reps
                   if r["arm"] == ref_arm and c in r["snap"][phase]]
            tgt = [r["snap"][phase][c][1] for r in reps
                   if r["arm"] == tgt_arm and c in r["snap"][phase]]
            if ref and tgt:
                d = sum(tgt) / len(tgt) - sum(ref) / len(ref)
                out[c] = d
        return out

    for phase in ("boot_ready", "postprobe"):
        print(f"\n## Derived delta vs arm A mean ({phase})\n")
        for tgt, name in (("noflag", "B-vs-A (in-split)"), ("iso", "C-vs-A (dedicated die)")):
            d = mean_delta(phase, "off", tgt)
            print(f"- {name}: " + ", ".join(f"{c} {v:+.1f} MiB" for c, v in d.items()))

    # arm C draft-die residency
    for r in reps:
        if r["arm"] != "iso":
            continue
        pre = r["snap"]["preboot"].get("card3")
        for ph in ("boot_ready", "postprobe"):
            s = r["snap"][ph].get("card3")
            if pre and s:
                print(f"\n- iso{r['rep']} card3 draft residency ({ph}): {s[1] - pre[1]:+.1f} MiB over idle")

    print("\n## Served battery per rep\n")
    print("| arm/rep | pp t/s | decode t/s | accept | mean acc len | determinism | needle | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for r in reps:
        rows, rec = r["rows"], r["receipt"]
        if not rows:
            print(f"| {r['arm']}{r['rep']} | (no battery receipt) | - | - | - | - | - | - |")
            continue
        pp = rows.get("prefill_2k", {}).get("measured_tps")
        dec = rows.get("decode_8k_10k", {})
        det = rows.get("determinism_greedy", {})
        ned = rows.get("needle_recall_8k", {})
        print(f"| {r['arm']}{r['rep']} | {fmt(pp,2)} | {fmt(dec.get('measured_tps'),2)} | "
              f"{dec.get('accept_ratio') if dec.get('accept_ratio') is not None else '-'} | "
              f"{dec.get('mean_len') if dec.get('mean_len') is not None else '-'} | "
              f"{(det.get('sha256_1') or '-') + (' (identical)' if det.get('identical') else ' (DIVERGED)')} | "
              f"{ned.get('recalled_count', '-')}/{ned.get('total_depths', '-')} | {rec.get('verdict')} |")

    print("\n## Contamination audit per rep\n")
    for r in reps:
        acc_last = r["acc"][-1] if r["acc"] else None
        print(f"- {r['arm']}{r['rep']}: server prompts={r['n_prompts']} (expect 8), "
              f"ctx-exceeded={r['n_exceeded']} (expect 0), retries={r['n_retry']} (expect 0), "
              f"max card3 VRAM sample={r['mx3']:.1f} MiB"
              + (f", last draft acceptance={acc_last}" if acc_last and r["arm"] != "off" else ""))


if __name__ == "__main__":
    sys.exit(main())
