#!/usr/bin/env python3
# Parse a rocprofv3 --kernel-trace CSV of bench_tile_resident.cpp runs into
# per-shape GEMM-only medians + class totals. Shapes run sequentially in the
# bench: per shape, 3 warmup computes then reps x iters timed computes, so the
# tile_fp16_gemm rows form reps*iters+warmup-sized blocks in shape order.
import csv, sys, statistics

csv_path = sys.argv[1]
m = int(sys.argv[2]) if len(sys.argv) > 2 else 128
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 20
warmup = 3

# census shapes in bench order: (name, K, N_d, count)
SHAPES = [
    ("ffn_gate",   5120, 5760, 64),
    ("ffn_up",     5120, 5760, 64),
    ("ffn_down",  17408, 1664, 64),
    ("gdn_qkv",    5120, 3328, 48),
    ("ssm_out",    6144, 1664, 48),
    ("gdn_gate",   5120, 2048, 48),
    ("attn_q+gate",5120, 4096, 16),
    ("attn_out",   6144, 1664, 16),
    ("attn_k",     5120,  256, 16),
    ("attn_v",     5120,  256, 16),
]

gemm_rows = []   # duration ns in time order
other = {}       # kernel name -> [count, total ns]
with open(csv_path) as f:
    for row in csv.DictReader(f):
        if row.get("Kind") != "KERNEL_DISPATCH":
            continue
        name = row["Kernel_Name"]
        dur = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
        if "tile_fp16_gemm" in name:
            gemm_rows.append(dur)
        else:
            other.setdefault(name, [0, 0])
            other[name][0] += 1
            other[name][1] += dur

per_shape = reps * iters + warmup
nblocks = len(gemm_rows) // per_shape
if nblocks != len(SHAPES):
    print(f"WARNING: {len(gemm_rows)} gemm rows -> {nblocks} blocks, expected {len(SHAPES)}")

print(f"{'shape':<12} {'K':>6} {'N_d':>5} {'gemm_us':>10} {'TF/s':>7}")
gemm_us = []
for i, (name, k, nd, cnt) in enumerate(SHAPES):
    block = gemm_rows[i*per_shape:(i+1)*per_shape][warmup:]
    med_ns = statistics.median(block)
    us = med_ns / 1000.0
    gemm_us.append((name, k, nd, cnt, us))
    tfs = 2.0 * nd * m * k / (med_ns / 1e9) / 1e12
    print(f"{name:<12} {k:>6} {nd:>5} {us:>10.2f} {tfs:>7.2f}")

gf = sum(c * 2.0 * nd * m * k for (_, k, nd, c, _) in gemm_us) / 1e9
ms = sum(c * us for (_, _, _, c, us) in gemm_us) / 1e3
print(f"call-weighted GEMM-only die agg: {gf:.1f} GF/ubatch, {ms:.2f} ms/ubatch -> {gf/(ms/1e3)/1e3:.2f} TF/s/die")

print("\nnon-GEMM kernel totals:")
for name, (cnt, total) in sorted(other.items(), key=lambda kv: -kv[1][1]):
    short = name.split("(")[0].split("<")[0]
    print(f"  {short:<40} n={cnt:<7} total={total/1e6:.2f} ms")
