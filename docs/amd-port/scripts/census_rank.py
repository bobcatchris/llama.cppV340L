#!/usr/bin/env python3
# W0 census analysis: aggregate rocprof v1 kernel CSV, rank by total seconds,
# attach byte-floor math for matmul kernels (weight bytes / 183.8 GB/s).
import csv, sys, collections, re

prefix = sys.argv[1]
path = prefix + "_kernel.csv"
if len(sys.argv) > 2:
    path = sys.argv[2]

rows = []
with open(path, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

def dur_ns(row):
    for k, v in row.items():
        if k and "Duration" in k:
            try:
                return float(v)
            except ValueError:
                pass
    return 0.0

def name(row):
    for k, v in row.items():
        if k and "Kernel Name" in k:
            return v
    return "?"

tot = sum(dur_ns(r) for r in rows)
agg = collections.Counter()
cnt = collections.Counter()
for r in rows:
    agg[name(r)] += dur_ns(r) / 1e9
    cnt[name(r)] += 1

BW = 183.8e9  # measured per-die D2D ceiling, GB/s

# crude per-kernel weight-bytes estimate for matmul-vec kernels
def weight_bytes(kname, tokens=1):
    # rows = output features * K-dim * bytes/weight, from kernel name hints
    m = re.search(r"mul_mat_vec_q.*", kname)
    return None  # filled manually per cell in the report

print(f"total kernel time: {tot:.2f} s over {len(rows)} dispatches\n")
print(f"{'rank':>4} {'total_s':>9} {'share':>6} {'calls':>7}  kernel")
for i, (k, s) in enumerate(agg.most_common(30)):
    print(f"{i:>4} {s:>9.3f} {100*s/tot:>5.1f}% {cnt[k]:>7}  {k[:110]}")
