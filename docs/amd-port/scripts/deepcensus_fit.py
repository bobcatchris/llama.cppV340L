#!/usr/bin/env python3
# W15 DEEP-CENSUS DESK A2: attention/KV ms-per-round vs KV depth, 200k
# projection, 5 ms/round threshold depth. Reads the phase-split dump for the
# measured per-launch medians at the two window depths, fits linear models
# us(d) = a + b*d (physical: parallel-block tile waves + full-pool dequant),
# and projects the served geometry (16 verify + 1 catch-up + 3 draft tile
# launches, 20.6 combines, 33 pool dequants per round per die - W11 P0).
import sys, re, os

DUMP = sys.argv[1] if len(sys.argv) > 1 else \
    "/media/chris/ssd128/llamacpp/wt-deep-census/docs/amd-port/results/W15_deepcensus_phase_split_2026-09-23.txt"

# banked of-record at ne11=7168 (W11_attn_p0 census): per-launch medians
D0 = 7168
BANKED = {"verify": 722.4, "catchup": 722.0, "draft": 61.8, "combine": 21.3, "deq40": 24.9}
# per-round launch counts per die (W11 P0 table)
CNT = {"verify": 16.0, "catchup": 1.0, "draft": 3.0, "combine": 20.6, "deq40": 33.0}

txt = open(DUMP).read()

meas = {}   # (win, name) -> median us
for m in re.finditer(r"win ([AB]) (verify|catchup|draft|combine|deq40)\s+med\s+([\d.]+) us", txt):
    meas[(m.group(1), m.group(2))] = float(m.group(3))
for m in re.finditer(r"round period (DECODE_[AB]): med ([\d.]+) ms", txt):
    print(f"round period {m.group(1)}: {m.group(2)} ms (direct wall witness)")

# window depths: from the server usage lines of the run (set at analysis time)
DA = float(os.environ.get("W15_DEPTH_A", 64 * 1024))
DB = float(os.environ.get("W15_DEPTH_B", 120 * 1024))
DT = float(os.environ.get("W15_DEPTH_T", 200 * 1024))

def fit(pts):  # least squares us = a + b*d
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return a, b

print("\n== measured per-launch medians (us) ==")
print(f"  banked @ {D0:.0f}: {BANKED}")
for w, d in (("A", DA), ("B", DB)):
    row = {k: meas.get((w, k)) for k in ("verify", "catchup", "draft", "combine", "deq40")}
    print(f"  win {w} @ {d:.0f}: {row}")

print("\n== linear fits us(d) = a + b*d ==")
fits = {}
for k in ("verify", "catchup", "draft", "combine", "deq40"):
    pts = [(D0, BANKED[k])]
    if meas.get(("A", k)): pts.append((DA, meas[("A", k)]))
    if meas.get(("B", k)): pts.append((DB, meas[("B", k)]))
    if len(pts) >= 2:
        fits[k] = fit(pts)
        a, b = fits[k]
        print(f"  {k:8s}: a {a:9.1f}  b {b * 1e6:9.4f} us per 1k KV   "
              f"pts {[(round(p[0]), round(p[1], 1)) for p in pts]}")

def us_at(k, d):
    if k in fits:
        a, b = fits[k]
        return max(a + b * d, 0.0)
    return BANKED.get(k, 0.0)

print("\n== attention/KV ms per decode round (per die) vs depth ==")
print("  depth       verify  catchup   draft  combine   deq40   SUM(ms/round/die)")
for d in (D0, 32 * 1024, DA, DB, 160 * 1024, DT):
    vals = {k: us_at(k, d) for k in CNT}
    tot = sum(CNT[k] * vals[k] for k in vals) / 1e3
    print(f"  {d:8.0f}  {CNT['verify'] * vals['verify'] / 1e3:8.2f} {CNT['catchup'] * vals['catchup'] / 1e3:7.2f} "
          f"{CNT['draft'] * vals['draft'] / 1e3:7.2f} {CNT['combine'] * vals['combine'] / 1e3:8.2f} "
          f"{CNT['deq40'] * vals['deq40'] / 1e3:7.2f} {tot:9.2f}")

base = sum(CNT[k] * us_at(k, D0) for k in CNT) / 1e3
lo, hi = D0, 512 * 1024
for _ in range(60):
    mid = (lo + hi) / 2
    if sum(CNT[k] * us_at(k, mid) for k in CNT) / 1e3 - base < 5.0:
        lo = mid
    else:
        hi = mid
print(f"\nattention/KV class @8k banked: {base:.2f} ms/round/die")
print(f"depth where the class DELTA vs 8k exceeds 5 ms/round: ~{lo:.0f} tokens")

deqT = CNT["deq40"] * us_at("deq40", DT) / 1e3
tileT = (CNT["verify"] * us_at("verify", DT) + CNT["catchup"] * us_at("catchup", DT)) / 1e3
print(f"\n200k projection per round/die: verify+catchup tile {tileT:.1f} ms, deq40 {deqT:.1f} ms")
print(f"q4_0-direct tile lever @200k: +{deqT:.1f} ms (dequant deleted) "
      f"+ tile KV traffic cut x{(2 * 3) / (18 / 32):.2f} if BW-bound")
