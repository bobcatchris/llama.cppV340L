# Loader dequant-elimination desk - load-time f16 weight residency for the tile, validated die 3

Date: 2026-09-21. Worktree wt-loader-dequant, branch amd/loader-dequant (tile-integ
lineage). GPU: die 3 only (HIP_VISIBLE_DEVICES=3; dies 0-2 untouched). Scope: make
the a8 tile's f16-weight residency real by deleting the retained per-call f32/f16
dequant behind an env switch, with a hard VRAM budget; bank the TP3 fit verdict
with the arithmetic.

## 1. What landed

- ggml/src/ggml-cuda/common.cuh - `ggml_cuda_tile_fp16_residency`: persistent
  (tensor, device) cache of f16 weight copies for the tile route. Keyed by
  {device, weight-slice device pointer, K, row_diff, type} - weights are immutable
  after load, so pointer+shape keying is stable for context lifetime; entries are
  raw cudaMalloc (NOT graph-pool memory, which is recycled within a graph build)
  and freed in the context dtor via clear() with each entry's device current
  (T4 teardown lesson applied). Hit/miss/refusal counters; refused keys are final
  for the process (no per-call churn).
- ggml/src/ggml-cuda/tile-gemm.cu - `tile_fp16_resident_weight()`: fetch-or-produce.
  On first use the SAME dequant kernel the per-call route runs
  (`to_fp16_cuda(src0_dd_i, ...)` -> identical f16 values either way) fills the
  resident buffer; subsequent calls pass the stable resident pointer straight to
  tile_fp16_gemm. Reached only through tile_fp16_should_use, so the residency
  whitelist is exactly the tile census whitelist (quantized src0, K in
  {5120, 6144, 17408}, M<=512 %128, N_d %64, gfx900). src1 f32->f16 convert stays
  per-call (activations change).
- Budget (both gates checked once, at first use, when free VRAM is observable):
  1. cumulative per-device residency <= GGML_CUDA_TILE_FP16_RESIDENT_MIB
     (default 4096);
  2. live free VRAM (cudaMemGetInfo, device current) - 256 MiB margin >= the slice.
  Refusal -> per-call dequant fallback, one WARN per tensor, byte-identical to the
  per-call route (proven below).
- ggml/src/ggml-cuda/ggml-cuda.cu - dtor calls tile_fp16_residency.clear() after
  q81_act_cache.clear().

Env composition: GGML_CUDA_TILE_FP16=1 GGML_CUDA_TILE_FP16_RESIDENT=1. RESIDENT
without TILE_FP16 is provably inert (residency code sits behind the tile branch;
verified: zero route/residency logs, shipped route). TILE_FP16 without RESIDENT is
the E-024 v1 behavior. All 4 env combinations exercised this session.

Build: same flags as W5 (cmake -S . -B build-loader-dequant -DCMAKE_BUILD_TYPE=Release
-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON
-DGGML_HIP_MMQ_MFMA=ON -DLLAMA_BUILD_{EXAMPLES,TOOLS,TESTS}=OFF -DLLAMA_CURL=OFF,
cmake at /home/chris/opt/cmake/bin; target `ggml` only - the new app/ dir fails
without tools headers, pre-existing, unrelated). ggml + tile TUs compile clean,
zero warnings. Bench: docs/amd-port/tests/bench_tile_resident.cpp, g++ -O2
-std=c++17 -I ggml/include ... -L build-loader-dequant/bin -lggml -lggml-base
-lggml-hip (build line in file header).

## 2. Oracle - bit identity, both directions

docs/amd-port/tests/bench_tile_resident.cpp --dump writes the f32 dst per shape
(per-shape re-seeded inputs, deterministic kernel, same protocol as the E-024
env-unset proof):

- resident vs per-call dequant, 10/10 census shapes byte-identical
  (cmp of dumps, M=128). Same weights + same dequant kernel -> same f16 values
  either way -> identical tile math. This is the E-033 gate: tile output on
  resident-f16 weights is BIT-IDENTICAL to the per-call-dequant output.
- cap-refused arm (RESIDENT_MIB=16: 8 refusals + 2 small residents) vs per-call
  arm: 10/10 byte-identical - the fallback path IS the per-call path.
- env fully unset byte-identity was E-024; this change does not touch that path.

## 3. Die-3 bench, M=128, census shapes (bench_gemm_gfx900.cu table, per-die shares)

Wall op-level via host timers, 3 reps x 20 iters median, 4 sessions per arm
(2 interleaved pairs + dump run + final): per-call agg 4.28-4.34, resident
5.41-5.52; medians 4.32 / 5.46. Final clean pair per-shape:

```
shape        K     N_d  | per-call wall  TF/s | resident wall TF/s | ratio
ffn_gate   5120  5760   |   1.736 ms      4.35 |   1.388 ms   5.44  | 1.25x
ffn_up     5120  5760   |   1.736         4.35 |   1.400         5.39 | 1.24x
ffn_down  17408  1664   |   1.646         4.51 |   1.288         5.76 | 1.28x
gdn_qkv    5120  3328   |   1.001         4.36 |   0.838         5.21 | 1.20x
ssm_out    6144  1664   |   0.693         3.78 |   0.502         5.22 | 1.38x
gdn_gate   5120  2048   |   0.649         4.14 |   0.499         5.38 | 1.30x
attn_q+gate 5120 4096   |   1.356         3.96 |   0.998         5.38 | 1.36x
attn_out   6144  1664   |   0.666         3.93 |   0.503         5.20 | 1.32x
attn_k     5120   256   |   0.159         2.11 |   0.141         2.39 | 1.13x
attn_v     5120   256   |   0.158         2.12 |   0.143         2.34 | 1.10x
call-weighted die agg (E-015 counts) |   4.28 TF/s/die  |   5.41 TF/s/die   | +26.4%
```

GEMM-only kernel time (rocprofv3 --kernel-trace on the same binary, same
3x20 schedule, tile_fp16_gemm medians, parse script in tests/): per-call 6.32
TF/s/die call-weighted -> resident 6.13 (-3.0%). The E-024 L2-credit prediction is
confirmed and is SMALL: with the per-call dequant gone, the GEMM no longer reads
weights that were just written (16 MB L2), costing ~3% aggregate (ffn cells -4-5%,
small/medium cells flat to +5%; attn k/v identical at 3.18). Post-loader reference
is 6.13 integrated (E-024's conservative 5.53 standalone-harness anchor is beaten).

Wall decomposition per 63-compute pass (trace totals): per-call = 323.4 ms GEMM +
141.3 ms dequantize_block_iq3_s + 23.6 reduce + 16.0 src1 convert; resident =
333.5 GEMM + 25.2 reduce + 15.0 convert + 3.4 ms one-time residency production
(10 launches total). The deleted tax is real: ~141 ms of dequant per pass, which
is exactly the wall gap (4.28 -> 5.41).

VRAM delta: resident raw footprint 304.2 MiB for the 10-shape bench set (matches
analytic 304.25; per-tensor log lines in the run captures). End-state free VRAM:
per-call 7786 MiB vs resident 7532 MiB = 254 MiB delta = 304 MiB residency minus
the ~56 MiB pool slice the per-call arm keeps warm for its biggest dequant.
Teardown: VRAM used before/after a resident run identical (12,550,144 B,
rocm-smi) - dtor clear() returns every buffer, no T4-style leak.

## 4. TP3 VRAM budget - the verdict is NEGATIVE for served residency

Exact arithmetic, die shares = equal thirds (E-012 census), 64 layers = 48 GDN +
16 full-attn, f16 = 2 B/elem:

```
class          per-slice          x instances          per die
ffn_gate  5120x5760x2 = 56.25 MiB x 64 = 3600.0 MiB
ffn_up    5120x5760x2 = 56.25 MiB x 64 = 3600.0 MiB
ffn_down 17408x1664x2 = 55.25 MiB x 64 = 3536.0 MiB   -> FFN 64L  = 10,736 MiB/die
gdn_qkv   5120x3328x2 = 32.50 MiB x 48 = 1560.0 MiB
gdn_gate  5120x2048x2 = 20.00 MiB x 48 =  960.0 MiB
ssm_out   6144x1664x2 = 19.50 MiB x 48 =  936.0 MiB   -> GDN 48L  =  3,456 MiB/die
attn_q+gate    40.00 MiB x 16 =  640.0 | attn_out 312.0 | k 40.0 | v 40.0
                                                            -> attn 16L =  1,032 MiB/die
FULL WHITELIST = 15,224 MiB/die = 14.87 GiB/die   (45,672 MiB = 44.6 GiB across TP3)
```

- Die is 8573 MiB total. Full whitelist = 1.78x the ENTIRE die. FFN-only
  (10,736 MiB) = 1.25x the die - does not fit even on an empty die with nothing
  else resident. Largest single class (ffn_down, 3536 MiB) also exceeds any
  realistic headroom; f16/quantized ratio for the whitelist = 2.61x (5837 MiB
  quantized per die).
- At the 200k boot of record (8419 of 8573 MiB used, 154 MiB free/die) the
  free-VRAM gate refuses EVERY slice including the smallest FFN one
  (55.25 + 256 margin = 311 MiB > 154 MiB). Empirically exercised: with free VRAM
  artificially held at 234 MiB (7900 MiB hog), 10/10 tensors refused, 0 residents,
  per-call fallback, output byte-identical, clean exit - exactly the served-200k
  behavior. At 10k-class boots (763-991 MiB free) the only fittable classes are
  perf-irrelevant (attn k+v = 80 MiB/die; gdn_qkv 1560 MiB does not fit).
- Therefore at TP3 the switch stays OFF in production: with it off, behavior and
  numerics are today's tile arm unchanged. The +26% wall win is banked for
  environments with headroom (single-GPU short-context serving, >=16 GiB dies,
  or a future weights/KV format change).

Edge gates validated: CAP_MIB=16 -> one WARN per refused tensor (8), the two
sub-cap slices (attn k/v, 2.5 MiB each) resident, 10/10 byte-identity vs per-call,
no per-call log churn (refusal is final). Low-free-VRAM -> same refusal path with
the free/margin numbers in the WARN.

## 5. Honest couplings

1. The -3% GEMM-only aggregate (6.32 -> 6.13) is the price of losing the dequant's
   L2 write-to-read credit; it is already included in every wall number above.
   Net wall is +26% BECAUSE the deleted dequant (141 ms/pass) dwarfs it.
2. rocprofv3 overhead inflates absolute trace times vs the E-024 standalone
   numbers; per-arm comparisons within this session are same-binary same-schedule.
3. In-process dual-arm timing is impossible (env is static-once), so A/B is
   cross-process; drift guard = 4 sessions per arm, agg bands 4.28-4.34 and
   5.41-5.52, disjoint.
4. Pointer-keying assumes immutable weights for context lifetime - true for model
   weights (the only quantized src0 class the census whitelist admits); the bench
   keeps all shape buffers alive for the same reason.

## 6. Run inventory

- Die-3 windows used: smoke+edge (~4 min), oracle dumps (~2 min), 4x2 wall
  sessions (~3 min), 2 traces (~4 min), hog/teardown checks (~3 min). Dies 0-2
  untouched.
- Raw captures: results/loader_dequant_{percall,resident}_run_2026-09-21.txt,
  traces results/loader_dequant_trace_{percall,resident}_2026-09-21.csv.
- Files: ggml/src/ggml-cuda/{common.cuh,tile-gemm.cu,ggml-cuda.cu},
  docs/amd-port/tests/{bench_tile_resident.cpp,parse_tile_resident_trace.py}.
- Ledger: E-033 (implementation + validation), E-034 (TP3 budget verdict).

Next link: this desk's mechanism is served-validated only at the harness level; a
future promotion path at TP3 requires either (a) a weights/KV format change that
opens >= 15.3 GiB/die (none on the roadmap), or (b) partial residency of a
high-call small class - the only candidate under 154 MiB is attn k/v (80 MiB/die,
+10-13% wall on those two cells, ~0.7% call-weighted) which is not worth a served
A/B. The tile wall at TP3 remains 4.33 TF/s/die-class with the per-call dequant;
the next honest lever for the wall is the src1 f32->f16 convert (16 ms/pass here)
and the prefill engine's remaining gap to 6.13.
