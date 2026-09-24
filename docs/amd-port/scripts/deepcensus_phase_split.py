#!/usr/bin/env python3
# W15 DEEP-CENSUS DESK A1: phase-split of the deep-context kernel trace.
# Trace: W15_deepcensus_2026-09-23_kernel_trace.csv (rocprofv3 kernel trace,
# probe A = ~64k prefill + 128 gen, probe B = +~55k (total ~120k KV) + 128 gen).
# Phases: the two big Cijk bursts = the two prefills; decode windows = the gaps
# after each burst. Decode rounds clustered by decode-class gated_delta_net
# launches inside each window (W11 P0 method). Per round the 20 tile-class
# launches per die are split by duration rank: 16 verify (largest), 1
# catch-up, 3 draft (smallest). Outputs the W15 phase-split dump.
import csv, sys, collections, statistics

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "/media/chris/ssd128/llamacpp/wt-deep-census/docs/amd-port/results/W15_deepcensus_2026-09-23_kernel_trace.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/media/chris/ssd128/llamacpp/wt-deep-census/docs/amd-port/results/W15_deepcensus_phase_split_2026-09-23.txt"

FAM = {"cijk": 1, "tile": 2, "vec": 3, "combine": 4, "deq40": 5,
       "setrows_q": 6, "gdn": 7, "nccl": 8, "mmvq": 9, "setrows_f16": 10}
FNAME = {v: k for k, v in FAM.items()}

def fam(name):
    if name.startswith("Cijk"): return FAM["cijk"]
    if "flash_attn_tile" in name: return FAM["tile"]
    if "flash_attn_ext_vec" in name: return FAM["vec"]
    if "flash_attn_combine" in name: return FAM["combine"]
    if "dequantize_block_q4_0" in name: return FAM["deq40"]
    if "k_set_rows_quant" in name: return FAM["setrows_q"]
    if name.startswith("void k_set_rows"): return FAM["setrows_f16"]
    if "gated_delta_net" in name: return FAM["gdn"]
    if "ncclDevKernel" in name: return FAM["nccl"]
    if "mul_mat_vec_q" in name: return FAM["mmvq"]
    return 0

rows = []  # (agent, fam, start, dur)
with open(CSV) as f:
    r = csv.reader(f); next(r)
    for line in r:
        if not line or line[0] != "KERNEL_DISPATCH":
            continue
        s = int(line[6]); e = int(line[7])
        if e <= s or (e - s) > 1_000_000_000:  # corrupt/garbage
            continue
        rows.append((int(line[1]), fam(line[4]), s, e - s))
rows.sort(key=lambda x: x[2])
t0 = rows[0][2]; t1 = max(r[2] + r[3] for r in rows)
print(f"trace span {(t1-t0)/1e9:.3f} s, clean dispatches {len(rows)}")

# --- prefill bursts from the Cijk timeline ---
cijk = sorted((s, s + d) for a, f, s, d in rows if f == FAM["cijk"])
bursts = []
for s, e in cijk:
    if bursts and s - bursts[-1][1] < 1_000_000_000:
        bursts[-1][1] = max(bursts[-1][1], e)
    else:
        bursts.append([s, e])
big = [b for b in bursts if sum(1 for s, e in cijk if b[0] <= s < b[1]) > 2000]
print("Cijk bursts:", [(round((b[0]-t0)/1e9, 2), round((b[1]-t0)/1e9, 2)) for b in bursts])
print("PREFILL bursts:", [(round((b[0]-t0)/1e9, 2), round((b[1]-t0)/1e9, 2)) for b in big])
if len(big) < 2:
    print("FATAL: expected two prefill bursts"); sys.exit(1)
PA, PB = big[0], big[1]
preA_end = PA[1]; preB_start = PB[0]; preB_end = PB[1]

W = [  # (label, lo, hi) exclusive windows between phases
    ("A", preA_end, preB_start),
    ("B", preB_end, t1 + 1),
]

# --- decode rounds per window (decode-class GDN clusters, >10 ms gap) ---
dec = {}
for label, lo, hi in W:
    gs = [s for a, f, s, d in rows if f == FAM["gdn"] and d < 100_000 and lo <= s < hi]
    clusters = []
    for s in gs:
        if clusters and s - clusters[-1][-1] > 10_000_000:
            clusters.append([s])
        else:
            if clusters: clusters[-1].append(s)
            else: clusters.append([s])
    clusters = [c for c in clusters if len(c) >= 10]
    dec[label] = clusters
    print(f"decode window {label}: {len(clusters)} rounds")

out = open(OUT, "w")
def P(*a): print(*a); print(*a, file=out)

P(f"# W15 deep-census phase split - trace {CSV.split('/')[-1]}")
P(f"# span {(t1-t0)/1e9:.3f} s, dispatches {len(rows)}")
P(f"# PREFILL_A rel {(PA[0]-t0)/1e9:.3f}..{(PA[1]-t0)/1e9:.3f} s ({(PA[1]-PA[0])/1e9:.3f} s)")
P(f"# PREFILL_B rel {(PB[0]-t0)/1e9:.3f}..{(PB[1]-t0)/1e9:.3f} s ({(PB[1]-PB[0])/1e9:.3f} s)")

# --- per-window family aggregation ---
res = {}
for label, lo, hi in W:
    lo_ms, hi_ms = (lo - t0) / 1e6, (hi - t0) / 1e6
    cnt = collections.Counter(); us = collections.Counter()
    for a, f, s, d in rows:
        if lo_ms <= (s - t0) / 1e6 < hi_ms:
            cnt[f] += 1; us[f] += d / 1e3
    tot = sum(us.values()) / 1e3
    nr = max(len(dec[label]), 1)
    P(f"\n== window {label}: {sum(cnt.values())} launches, {tot:.1f} ms, {nr} rounds ==")
    for f in sorted(us, key=lambda x: -us[x]):
        med = 0.0
        vals = [d / 1e3 for a, ff, s, d in rows if ff == f and lo_ms <= (s - t0) / 1e6 < hi_ms]
        if vals: med = statistics.median(vals)
        P(f"  fam {FNAME.get(f, f'f{f}'):10s}: {us[f]/1e3:9.2f} ms  n/die/round {cnt[f]/4/nr:6.1f}  med/launch {med:8.1f} us")
    res[label] = (cnt, us)

# --- round periods (direct wall witness per window) ---
for label, lo, hi in W:
    cs = dec[label]
    if len(cs) >= 3:
        per = [cs[i + 1][0] - cs[i][0] for i in range(len(cs) - 1)]
        P(f"\nround period {label}: med {statistics.median(per)/1e6:.2f} ms  "
          f"p25 {sorted(per)[len(per)//4]/1e6:.2f}  p75 {sorted(per)[3*len(per)//4]/1e6:.2f}  n {len(per)}")

# --- tile-class split per round by duration rank (verify/catchup/draft) ---
P("\n== tile per round by rank (per die): 16 verify / 1 catch-up / 3 draft ==")
for w_i, (label, lo, hi) in enumerate(W):
    nr = len(dec[label])
    starts = [c[0] - 2_000_000 for c in dec[label]]  # round start boundaries
    per_rd = collections.defaultdict(list)
    for a, f, s, d in rows:
        if f != FAM["tile"] or not (lo <= s < hi): continue
        # round index: last boundary <= s
        import bisect
        i = bisect.bisect_right(starts, s) - 1
        if i < 0: continue
        per_rd[(i, a)].append(d / 1e3)
    for name, pick in (("verify", slice(4, 20)), ("catchup", slice(3, 4)), ("draft", slice(0, 3))):
        vals = []
        for k, ds in per_rd.items():
            ds = sorted(ds)
            if len(ds) >= 20:
                vals.extend(ds[pick])
        if vals:
            P(f"  win {label} {name:8s} med {statistics.median(vals):8.1f} us  "
              f"p10 {sorted(vals)[len(vals)//10]:8.1f}  p90 {sorted(vals)[9*len(vals)//10]:8.1f}  "
              f"n {len(vals)}  rounds/die {len(per_rd)}")

# --- attention/KV per-round totals (per die) ---
P("\n== attention/KV class per decode round (per die, ms) ==")
for label, lo, hi in W:
    cnt, us = res[label]
    nr = max(len(dec[label]), 1)
    tot = (us[FAM["tile"]] + us[FAM["vec"]] + us[FAM["combine"]]
           + us[FAM["deq40"]] + us[FAM["setrows_q"]] + us[FAM["gdn"]]) / 1e3
    P(f"  win {label}: tile {(us[FAM['tile']]/1e3/4/nr):6.3f}  vec {(us[FAM['vec']]/1e3/4/nr):6.3f}  "
      f"combine {(us[FAM['combine']]/1e3/4/nr):6.3f}  deq40 {(us[FAM['deq40']]/1e3/4/nr):6.3f}  "
      f"setrows {(us[FAM['setrows_q']]/1e3/4/nr):5.3f}  gdn {(us[FAM['gdn']]/1e3/4/nr):5.3f}  "
      f"SUM {tot/4/nr:6.2f} ms/round/die")

# --- per-launch medians (aggregate per window) ---
P("\n== per-launch medians by window (us) ==")
for label, lo, hi in W:
    lo_ms, hi_ms = (lo - t0) / 1e6, (hi - t0) / 1e6
    for fname in ("tile", "vec", "combine", "deq40", "setrows_q"):
        f = FAM[fname]
        vals = [d / 1e3 for a, ff, s, d in rows if ff == f and lo_ms <= (s - t0) / 1e6 < hi_ms]
        if vals:
            P(f"  win {label} {fname:10s} med {statistics.median(vals):8.1f} us  n {len(vals)}")

# --- per-die medians (per-die spread) ---
P("\n== per-die medians (per-die spread) ==")
for label, lo, hi in W:
    lo_ms, hi_ms = (lo - t0) / 1e6, (hi - t0) / 1e6
    for fname in ("tile", "deq40", "combine", "nccl"):
        f = FAM[fname]
        bydie = collections.defaultdict(list)
        for a, ff, s, d in rows:
            if ff == f and lo_ms <= (s - t0) / 1e6 < hi_ms:
                bydie[a].append(d / 1e3)
        if not bydie: continue
        meds = {a: statistics.median(v) for a, v in sorted(bydie.items())}
        spread = max(meds.values()) / min(meds.values())
        P(f"  win {label} {fname:10s} per-die med us: " +
          " ".join(f"{a}:{m:.1f}" for a, m in meds.items()) +
          f"  max/min {spread:.3f}")

# --- PREFILL_B flash depth ramp (linearity witness across the depth range) ---
P("\n== PREFILL_B flash_attn_tile + dequant depth ramp (per ~1/12 of the burst) ==")
tileB = sorted((s, d) for a, f, s, d in rows if f == FAM["tile"] and PB[0] <= s < PB[1])
deqB = sorted((s, d) for a, f, s, d in rows if f == FAM["deq40"] and PB[0] <= s < PB[1])
NB = 12
for i in range(NB):
    lo = PB[0] + (PB[1] - PB[0]) * i // NB
    hi = PB[0] + (PB[1] - PB[0]) * (i + 1) // NB
    ds = [d / 1e3 for s, d in tileB if lo <= s < hi]
    dq = [d / 1e3 for s, d in deqB if lo <= s < hi]
    if ds:
        P(f"  ramp {i:2d}: tile med {statistics.median(ds):8.1f} us  n {len(ds):5d}   "
          f"deq40 med {statistics.median(dq) if dq else 0:8.1f} us  n {len(dq):5d}")
out.close()
print("wrote", OUT)
