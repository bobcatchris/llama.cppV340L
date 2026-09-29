#!/usr/bin/env python3
# Parse the two rocprofv3 --kernel-trace CSVs of bench_tile_nspan.cpp runs
# (unchunked and N-span arms) into per-shape kernel-only medians, us per
# compute, and TF/s. Per shape the bench runs 3 warmup computes then
# reps x iters timed computes (63 per shape at the default 3x20); kernels are
# attributed in time order by the launch model. At the default window budget
# every census shape is a single span, so BOTH arms run per compute:
#   1 gemm + 1 reduce + 1 weight dequant + 1 src1 convert
# (nspan only renames the reduce: tile_fp16_reduce_span).
import csv, sys, statistics

m     = int(sys.argv[2]) if len(sys.argv) > 2 else 128
reps  = int(sys.argv[3]) if len(sys.argv) > 3 else 3
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 20
warm  = 3

SHAPES = [  # (name, K, N_d, ks)
    ("ffn_gate",   5120, 5760, 4),
    ("ffn_up",     5120, 5760, 4),
    ("ffn_down",  17408, 1664, 8),
    ("gdn_qkv",    5120, 3328, 8),
    ("ssm_out",    6144, 1664, 8),
    ("gdn_gate",   5120, 2048, 8),
    ("attn_q+gate",5120, 4096, 4),
    ("attn_out",   6144, 1664, 8),
    ("attn_k",     5120,  256, 8),
    ("attn_v",     5120,  256, 8),
]

def per_compute(stream, per_shape):
    # stream: (name, dur) in time order; per_shape: launches per compute per shape
    # returns shape -> list of per-compute duration sums (ns)
    out = {si: [] for si in range(len(SHAPES))}
    si, need, acc, cur = 0, per_shape[0], 0, 0
    for name, dur in stream:
        if si >= len(SHAPES):
            break
        acc += dur
        cur += 1
        if cur == need:
            out[si].append(acc)
            acc, cur = 0, 0
            if len(out[si]) >= warm + reps * iters:
                si += 1
                if si < len(SHAPES):
                    need = per_shape[si]
    return out

def load(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("Kind") != "KERNEL_DISPATCH":
                continue
            rows.append((row["Kernel_Name"],
                         int(row["End_Timestamp"]) - int(row["Start_Timestamp"])))
    return rows

for arm, path in (("UNCHUNK", sys.argv[1] + "unchunk_kernel_trace.csv"),
                  ("NSPAN",   sys.argv[1] + "nspan_kernel_trace.csv")):
    rows = load(path)
    streams = {
        "gemm":  ([(n, d) for n, d in rows if "tile_fp16_gemm" in n], 1),
        "route": ([(n, d) for n, d in rows
                   if "tile_fp16_gemm" in n or "tile_fp16_reduce" in n
                   or "dequantize" in n or "convert_unary" in n], 4),
        "dequant": ([(n, d) for n, d in rows if "dequantize" in n], 1),
    }
    print(f"== {arm} kernel-only (median us/compute over last {reps*iters} computes)")
    res = {}
    for label, (stream, mult) in streams.items():
        res[label] = per_compute(stream, [mult] * len(SHAPES))
    print(f"{'shape':<14}{'ks':>3} {'gemm us':>10} {'gemm TF/s':>10} {'route us':>10} {'route TF/s':>10}")
    gf_w = 0.0
    ms_w = 0.0
    for si, (name, K, N, ks) in enumerate(SHAPES):
        flops = 2.0 * N * m * K
        g = statistics.median(res["gemm"][si][warm:]) / 1e3
        r = statistics.median(res["route"][si][warm:]) / 1e3
        tfg = flops / (g * 1e-6) / 1e12
        tfr = flops / (r * 1e-6) / 1e12
        count = [64, 64, 64, 48, 48, 48, 16, 16, 16, 16][si]
        gf_w += count * flops / 1e9
        ms_w += count * r / 1e3
        print(f"{name:<14}{ks:>3} {g:>10.1f} {tfg:>10.2f} {r:>10.1f} {tfr:>10.2f}")
    print(f"   route call-weighted: {gf_w / ms_w:.2f} TF/s\n")
