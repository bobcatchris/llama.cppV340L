#!/usr/bin/env python3
# W15 A1b: per-launch depth extraction from grid geometry (no time alignment).
# flash_attn_tile: Grid_Size_Y = parallel_blocks (WG 1) -> depth ~= pb*192
#   (7168 -> 37 blocks, census-exact). decode-class tile: Grid_Size_X/WG = 1.
# dequantize_block_q4_0: Grid_Size_X = ne11*256/32 -> depth = GridX/32.
# Output: binned medians (depth, us) per die for tile-prefill, tile-decode,
# deq40 + text scatter summary for the fit.
import csv, sys, collections, statistics

CSV = sys.argv[1]
OUT = sys.argv[2]
rows = []  # (agent, kind, depth, dur_us)
with open(CSV) as f:
    r = csv.reader(f); next(r)
    for line in r:
        if not line or line[0] != "KERNEL_DISPATCH":
            continue
        name = line[4]
        s = int(line[6]); e = int(line[7])
        dur = e - s
        if dur <= 0 or dur > 1_000_000_000:
            continue
        gx, gy = int(line[13]), int(line[14])
        a = int(line[1])
        if "flash_attn_tile" in name:
            if gx // 256 == 1 and gy > 0:      # decode-class tile (ntiles_x 1)
                rows.append((a, "tile_dec", gy * 192, dur / 1e3))
            else:                               # prefill-class tile (ntiles_x 32)
                rows.append((a, "tile_pre", gy * 192, dur / 1e3))
        elif "dequantize_block_q4_0" in name:
            rows.append((a, "deq40", gx // 32, dur / 1e3))
print("launches kept:", len(rows))
kinds = collections.Counter(k for a, k, d, u in rows)
print(kinds)

out = open(OUT, "w")
def P(*x): print(*x); print(*x, file=out)

# bin by depth (2k bins to 70k), per kind, per die
bins = collections.defaultdict(lambda: collections.defaultdict(list))
for a, k, d, u in rows:
    b = min(int(d // 2000), 34)
    bins[(k, a)][b].append(u)

for k in ("tile_pre", "tile_dec", "deq40"):
    for a in sorted(set(a for a2, kk, d, u in rows if kk == k)):
        series = bins.get((k, a))
        if not series: continue
        pts = []
        for b in sorted(series):
            dep = (b + 0.5) * 2000
            pts.append((dep, round(statistics.median(series[b]), 1)))
        P(f"== {k} die{a} (depth, med us) ==")
        P("  " + " ".join(f"{d//1000}:{u:.0f}" for d, u in pts))
out.close()
print("wrote", OUT)
