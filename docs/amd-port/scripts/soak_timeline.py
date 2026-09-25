#!/usr/bin/env python3
# W18 SOAK-MECHANISM DESK P1: decay timeline from the guard battery's thermal
# sideband logs (5 s cadence per-die edge/junction/mem/sclk, rocm-smi sourced;
# c0=05:00 c1=08:00 c2=0D:00, die 4 10:00 is NOT sampled - sampler gap).
# Decode window per cell = [boot ramp ~10 s .. +prefill+decode+canary slack];
# the 200k needle_recall guard runs AFTER decode, so cells with wall > 120 s
# use [10..95 s] (battery shape) and short 10k cells use their last 25 s.
# Cell tps values are the ledger-banked E-125/E-126/E-128 numbers.
import csv, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(HERE, "../results/W18_soak_timeline_2026-09-24.txt")

# (label, thermal log, decode tps, prefill wall s, decode wall s)
CELLS = [
    ("hb_p0a  ub512 pos1", "/home/chris/hb_p0a_thermal.log", 23.33, 15.312, 47.828),
    ("hb_p1a  ub1024 pos2", "/home/chris/hb_p1a_thermal.log", 19.85, 14.637, 49.899),
    ("hb_p0b  ub512 pos3", "/home/chris/hb_p0b_thermal.log", 17.35, 15.502, 56.933),
    ("hb_p1b  ub1024 pos4", "/home/chris/hb_p1b_thermal.log", 17.71, 14.721, 52.979),
    ("cw_regress_200k pos1", "/home/chris/combowin_regress_200k_thermal.log", 22.86, None, None),
    ("cw_iq4xs_200k pos2", "/home/chris/combowin_iq4xs_200k_thermal.log", 20.19, None, None),
    ("cw_s2r5x_200k pos3", "/home/chris/combowin_s2r5x_200k_thermal.log", 19.57, None, None),
    ("cw_chan4_200k pos4", "/home/chris/combowin_chan4_200k_thermal.log", 20.29, None, None),
    ("cw_regress_10k", "/home/chris/combowin_regress_10k_thermal.log", 22.46, None, None),
    ("cw_iq4xs_10k", "/home/chris/combowin_iq4xs_10k_thermal.log", 22.52, None, None),
    ("cw_s2r5x_10k", "/home/chris/combowin_s2r5x_10k_thermal.log", 22.51, None, None),
    ("cw_chan4_10k", "/home/chris/combowin_chan4_10k_thermal.log", 21.80, None, None),
    ("cw_s0a_10k", "/home/chris/combowin_s0a_10k_thermal.log", 22.98, None, None),
    ("cw_s0b_10k", "/home/chris/combowin_s0b_10k_thermal.log", 23.51, None, None),
    ("cw_dc0_10k", "/home/chris/combowin_dc0_10k_thermal.log", 23.64, None, None),
    ("cw_dc1_10k", "/home/chris/combowin_dc1_10k_thermal.log", 23.25, None, None),
    ("cw_dc0b_10k", "/home/chris/combowin_dc0b_10k_thermal.log", 23.32, None, None),
    ("cw_dc1b_10k", "/home/chris/combowin_dc1b_10k_thermal.log", 23.23, None, None),
]

def window(dur, prew, decw):
    if prew:
        return 10.0, 10.0 + prew + decw + 12.0
    if dur > 120:
        return 10.0, 95.0
    return dur - 25.0, dur + 1.0

rows_out = []
rows_out.append(f"{'cell':22} {'tps':>6} {'win_s':>10} | {'sclk act-mean c0/c1/c2':>24} | {'c2<=991':>7} | {'edge c2':>7} | {'junc c0/c2':>10} | {'mem c0/c2':>9} | {'mem max c0/c1/c2':>15}")
for label, path, tps, prew, decw in CELLS:
    if not os.path.exists(path):
        rows_out.append(f"{label:22} {tps:6.2f}  MISSING {path}")
        continue
    rows = list(csv.DictReader(open(path)))
    dur = float(rows[-1]["elapsed_s"])
    lo, hi = window(dur, prew, decw)
    w = [r for r in rows if lo <= float(r["elapsed_s"]) <= hi]
    sc = [[], [], []]; ju = [[], [], []]; me = [[], [], []]; ed = [[], [], []]
    for r in w:
        for d in range(3):
            s = float(r[f"c{d}_sclk"])
            if s > 500:
                sc[d].append(s)
            ju[d].append(float(r[f"c{d}_junc"]))
            me[d].append(float(r[f"c{d}_mem"]))
            ed[d].append(float(r[f"c{d}_edge"]))
    c2low = sum(1 for s in sc[2] if s <= 991) / max(1, len(sc[2]))
    mean = lambda x: statistics.mean(x) if x else 0.0
    rows_out.append(
        f"{label:22} {tps:6.2f} {lo:4.0f}-{hi:<5.0f} | "
        f"{mean(sc[0]):6.0f}/{mean(sc[1]):6.0f}/{mean(sc[2]):6.0f}      | "
        f"{c2low:6.0%} | {mean(ed[2]):6.0f}-{max(ed[2]):2.0f} | "
        f"{mean(ju[0]):4.0f}/{mean(ju[2]):4.0f} | {mean(me[0]):4.0f}/{mean(me[2]):4.0f} | "
        f"{max(me[0]):3.0f}/{max(me[1]):3.0f}/{max(me[2]):3.0f}")

hdr = [
    "W18 SOAK-MECHANISM DESK: decode-window thermal vs banked tps (P1 log mining)",
    "",
    "tp/s vs die-3 (0D:00) clock state during the decode window:",
    "  tps >= 23.2  <->  c2 act-mean >= ~1190, c2 low-band (<=991 MHz) duty ~0-20%",
    "  tps 19.5-20.3 <-> c2 act-mean 1142-1211, c2 low-band duty 18-41%",
    "  tps 17.4-17.7 <-> c2 act-mean 1068-1135, c2 low-band duty 44-53% (775/560 pins)",
    "  hot-but-unthrottled 10k cells (c2 duty 0%) decode 21.8-22.5 even at mem 83-88 C",
    "",
]
open(OUT, "w").write("\n".join(hdr + rows_out) + "\n")
print("\n".join(hdr + rows_out))
