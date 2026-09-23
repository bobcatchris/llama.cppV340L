#!/usr/bin/env python3
# PREFILL DESK P0: phase-split of the TP4 kernel census
# (TP4_kernel_census_2026-09-23_kernel_trace.csv, 589k dispatches, ns timestamps,
# one probe = 6964-token prefill + 98 decode tokens = the round-map census).
# Windows: PREFILL = Cijk-burst region (first Cijk after the load/warmup quiet
# gap .. last Cijk), DECODE = last-Cijk-end .. trace end (the 98-token battery).
import csv, sys, collections

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "/media/chris/ssd128/llamacpp/llama.cpp/docs/amd-port/results/TP4_kernel_census_2026-09-23_kernel_trace.csv"

rows = []
with open(CSV) as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        if not line or line[0] != "KERNEL_DISPATCH":
            continue
        s = int(line[6]); e = int(line[7])
        if e <= s or (e - s) > 1_000_000_000:  # corrupt/garbage timestamps
            continue
        rows.append((int(line[1]), line[4], s, e - s))

t0 = min(r[2] for r in rows); t1 = max(r[2] + r[3] for r in rows)
print(f"trace span {(t1-t0)/1e9:.3f} s, dispatches (clean) {len(rows)}")

def fam(name):
    if name.startswith("Cijk"): return "Cijk"
    if "mul_mat_vec_q" in name: return "mmvq"
    if "nccl" in name.lower(): return "nccl"
    if "flash_attn" in name: return "flash"
    return "other"

cijk = sorted((s, e) for a, n, s, d in rows if fam(n) == "Cijk")
# main prefill burst = after the last >2 s quiet gap in the Cijk timeline
gap_break = 0
for i in range(1, len(cijk)):
    if cijk[i][0] - cijk[i-1][1] > 2_000_000_000:
        gap_break = i
        print(f"quiet gap {(cijk[i][0]-cijk[i-1][1])/1e9:.3f} s before burst idx {i}")
prefill_start = cijk[gap_break][0]; prefill_end = cijk[-1][1]
print(f"PREFILL [{(prefill_start-t0)/1e9:.3f} .. {(prefill_end-t0)/1e9:.3f}] s = {(prefill_end-prefill_start)/1e9:.3f} s")
print(f"DECODE  [{(prefill_end-t0)/1e9:.3f} .. {(t1-t0)/1e9:.3f}] s = {(t1-prefill_end)/1e9:.3f} s")

def agg(lo_ms, hi_ms, label):
    cnt = collections.Counter(); us = collections.Counter()
    for a, n, s, d in rows:
        if lo_ms <= (s - t0)/1e6 < hi_ms:
            cnt[n] += 1; us[n] += d/1e3
    tot = sum(us.values())/1e3
    print(f"\n== {label}: {sum(cnt.values())} launches, {tot:.1f} ms kernel time ==")
    fam_tot = collections.Counter()
    for n in us: fam_tot[fam(n)] += us[n]/1e3
    for k, v in fam_tot.most_common():
        print(f"  fam {k:6s}: {v:10.1f} ms ({100*v/tot:5.1f}%)")
    return cnt, us

cnt_p, us_p = agg((prefill_start-t0)/1e6, (prefill_end-t0)/1e6, "PREFILL")
cnt_d, us_d = agg((prefill_end-t0)/1e6, (t1-t0)/1e6, "DECODE")

print("\n== PREFILL top 30 kernels by time ==")
for n, u in us_p.most_common(30):
    print(f"  {u/1e3:9.1f} ms {cnt_p[n]:7d} launches {u/cnt_p[n]:9.1f} us/launch  {n[:90]}")

print("\n== PREFILL Cijk variants ==")
cij = collections.Counter()
for n, u in us_p.items():
    if n.startswith("Cijk"):
        cij[n] += u/1e3
tot_c = sum(cij.values())
for n, u in cij.most_common():
    print(f"  {u:9.1f} ms ({100*u/tot_c:5.1f}% of Cijk) {n}")
print(f"  TOTAL Cijk {tot_c:.1f} ms")

print("\n== PREFILL convert/dequant/staging kernels ==")
for pat in ("convert", "dequant", "quantize", "cpy", "copy", "cpy_scalar", "concat"):
    for n, u in sorted(us_p.items(), key=lambda kv: -kv[1]):
        if pat in n.lower():
            print(f"  {u/1e3:9.1f} ms {cnt_p[n]:7d} launches  {n[:90]}")

# per-agent family totals + Cijk grid shapes (grid dims from the trace)
agents = sorted(set(a for a, _, _, _ in rows))
fam_agent = collections.defaultdict(collections.Counter)
for a, n, s, d in rows:
    if prefill_start <= s < prefill_end:
        fam_agent[a][fam(n)] += d/1e6
print("\n== PREFILL per-die family seconds ==")
for a in agents:
    t = sum(fam_agent[a].values())
    print(f"  die(agent) {a}: busy {t:7.2f} s | " +
          " ".join(f"{k}={v:6.2f}s({100*v/t:4.1f}%)" for k, v in fam_agent[a].most_common()))

grids = collections.defaultdict(collections.Counter)
with open(CSV) as f:
    r = csv.reader(f); next(r)
    for line in r:
        if not line or line[0] != "KERNEL_DISPATCH": continue
        s = int(line[6])
        if not (prefill_start <= s < prefill_end): continue
        n = line[4]
        if n.startswith("Cijk"):
            grids[n.split("_MT")[1].split("_")[0] + "|WG" + n.split("_WG")[1].split("_")[0]][(line[13], line[14], line[15])] += 1
print("\n== Cijk variant grids (launch counts, prefill) ==")
for v, g in sorted(grids.items(), key=lambda kv: -sum(kv[1].values())):
    tot = sum(g.values())
    tops = ", ".join(f"grid {k[0]}x{k[1]}x{k[2]}: {c}" for k, c in g.most_common(6))
    print(f"  MT{v}: {tot} launches | {tops}")
