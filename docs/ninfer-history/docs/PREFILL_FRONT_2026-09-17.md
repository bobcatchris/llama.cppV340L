# PREFILL FRONT — which kernel serves NVFP4 linear at TP4 prefill (no-GPU desk, 2026-09-17, amd-tp4-cure)

## 1. Bottom line (the honest answer)

**Every NVFP4 linear at TP4 prefill (M = 128-token chunks) on the HIP lane runs the P3 SIMT
small-T kernel, `nvfp4_small_t_hip_kernel`, driven in T=32 slices — a decode-class kernel whose
own header names itself "The PRE-FILL wall", "PERF NAMED-IMMATURE" (nvfp4_small_t_hip.cu:2-10).
There is no large-M GEMM on this lane.** Weights are re-read per token: one warp per (row,token)
pair (nvfp4_small_t_hip.cu:7-9, :43-49) — arithmetic intensity pinned at the M=1 ratio.

## 2. The full dispatch chain (file:line)

- Policy bind: `NINFER_NVFP4_SIMT_LANE` ⇒ `kNvfp4SimtLaneTextPolicy = A16Only`
  (variant_kernels.cpp:63-64; rationale :55-62; route fn :67-87).
- Route table: A16Only ⇒ A16 for every problem (nvfp4_config.h:202); AllowA4 thresholds
  (:210/:218/:225) are dead code on this lane, kept byte-identical as the CUDA contract (:183-194).
- TP4 serving path: fused QKV/GDN projections go through `multi_gpu::tp_gemv`
  (variant_kernels.cpp:229-230, :240-242, :287), which under the SIMT-lane define sends NVFP4
  **at every t** to `ops::detail::nvfp4_dispatch(..., A16Only, ...)` (tp_kernel.cu:63-77) — the
  CUDA-lane t>8 W4A4 prefill arm (:79-90) never compiles in. Plain linears: same dispatch.
- A16 arm: `launch_a16` slices M into `kNvfp4LastSmallT`=32-token views, T=1→GEMV arm,
  2..32→`launch_nvfp4_small_t` (src/ops/linear/nvfp4/nvfp4_dispatch.cpp:26-42). M=128 = 4
  launches/chunk-view sweep.
- LIVE nvfp4 device TUs = exactly two + one snapshot kernel (src/HipSources.cmake:281,283,256):
  nvfp4_gemv_hip.cu (P2), nvfp4_small_t_hip.cu (P3). EVERY W4A4/fused arm is a link stub that
  THROWS (apps/serve/hip_link_stub_arms.cpp:35-40, :65-82) — belt, not route.
- Why no mma/TMA arm can save prefill: gfx900 has no matrix instructions (doc 01 §2
  docs/01_v100_skinny_findings_for_v340l.md:24-29 "Dies"; nvfp4_config.h:186-188 — ROCm 6.2 fp4
  surface measured empty, agent3).

## 3. Measured context (PERF_LOG + coherence rows)

- Prefill: 1029.8 s @10k = 9.7 tok/s mean, 14.5 early → ~9 deep (PLOG-006); cool-start COOL10K
  = 11.30 mean: plateau 24.5 → knee → deep ~9.x (PLOG-020); knee+slow regime is THERMAL
  (PLOG-017/027 — 1.75x degradation locks in after ~45-60 s load). MTP3: 10.65 (PLOG-010).
  q3-parity target: 23-25 (PLOG-003).
- Die ceilings (ROOFLINE_row.txt, measured): copy 369.5 GB/s, pure-read **377.9 GB/s**.
- Decode GEMV after tuning: 211.8-295.1 GB/s = 55-77% of read ceiling, BIT-EQUAL (PLOG-030,
  GEMV_TUNING_row.txt; techniques in EXTRACTING_NUMBERS_and_50TPS_budget.md §B.1).

## 4. Theory: M=128 is PAST the wall — prefill is COMPUTE-bound

- FLOP per weight byte = 2M / 0.5625 B/weight = 3.56·M. Machine FLOP/byte at the MEASURED read
  ceiling: fp32 10.75e12 / 377.9e9 ≈ 28 → crossover **M\* ≈ 8**; fp16 2:1 (21.5e12 [inference,
  doc 01 §3 arithmetic]) → M\* ≈ 16. Doc 01 §3 (nominal 483.8 GB/s) computed M ≈ 12, practical
  6-8 with dequant on the critical path (docs/01:36-44). **M=128 is 8-16x past the crossover.**
- Weight planes per layer per rank at TP4 (W4 shards, codes K/2 + scales K/16 B/row,
  nvfp4_config.h:136-140): Σ = **64.2 MB** (codes 57.1 + scales 7.1); x4 ranks =
  **257 MB/layer all-ranks** — the decode roofline anchor, unchanged by M.
- Bandwidth floor per 128-token step: 64.2 MB x 48 layers / 377.9 GB/s ≈ 8.2 ms/step
  (~15.6k tok/s) — far above the compute wall, so the binding limit is FLOPs: ΣN(W4) = 26,624
  rows ⇒ 2·26624·5120·128 = 34.97 GFLOP/layer/die ⇒ 3.25 ms/layer at nominal 10.75 TFLOP/s
  fp32 ⇒ **156 ms/step ≈ 820 tok/s nominal-fp32 ideal (~1640 at fp16 2:1)**. Measured plateau
  24.5 / mean 11.3 ⇒ the tiled-GEMM prize is nominally 34-145x; even 10% of the fp32 ideal
  beats the plateau 3.4x. [Planning-grade: 1.5 GHz + 2:1 are doc-01 inferences; measured load
  clock SCLK 1269-1350, PLOG-031.]
- Sanity check that today's kernel is ISSUE-bound, not bandwidth-bound: 3.41M (row,token)
  pairs/step/die x ~160 groups/lane x >100 VALU slots/group (the e2m1 GLOBAL table load per
  nibble — small_t calls the raw `e2m1_bits` static table, small_t_hip.cu:67 — exactly the
  pathology GEMV-TUNE removed from the GEMV, B.1 item 1) ⇒ ≥60 G slots/step vs 284 G slot/s
  (56 CU x 4 SIMD x 1.269 GHz) ⇒ ~40 tok/s ceiling — same order as the 24.5 plateau. Consistent.

## 5. Gap class + design sketch

**Gap class: missing large-M tiled LDS dequant GEMM** (doc 01 §2 "Net simplification": the M>8
band "becomes SIMT (issue-bound) or a plain tiled LDS GEMM" — docs/01:30-31; PLAN_50TPS "Part A's
second half — never built"). Design sketch (doc 01 §1-2 + B.1 wins, gfx900-legal): dequant
in-kernel → x-tile in LDS (XOR swizzle survives the 32x4B banks), K-chunk 1024→512 for the
64 KB LDS; e2m1 PAIR table (2 KB) + e4m3 scale table (1 KB) in LDS — B.1 items 1-2, measured
40%→70%+ alone on the GEMV; u64 code loads, 2x uint4 x loads; fp32 accumulators (fp16 2:1 math
with fp32 flush windows as the later option). CONTRACT NOTE: any order change moves the bit
digest and must be declared at its gate (nvfp4_amd_codec.h:131-138) — the tiled kernel is a NEW
route/arm, not a silent small_t edit.

## 6. Bench deliverable + expected read

results/amd/coherence/nvfp4_prefill_bench.cu (NEW, committed): copy/read ceilings + 5 s mclk
hammer (roofline-bench discipline), then PRODUCTION arm (launch_nvfp4_small_t via launch_a16's
exact T=32 chunking) vs an inline ~50-line REFERENCE tiled dequant GEMM (32-row blocks, K-slab
dequant-once into LDS fp32, weight byte serves all M tokens, fp32 accum — same amd codec +
nvfp4_scale_offset planes) at M ∈ {32,128,256} x the four big geometries. Expected read: PROD
GB/s flat-to-falling in M (M=1 intensity + issue-bound); REF GFLOP/s RISING toward a few
TFLOP/s with GB/s falling — the compute wall made visible. Decision rule: REF/PROD GFLOP/s
ratio at M=128 sizes the tiled-GEMM build (predicted ≥3-5x). Compile receipt: exact -O3 law
command (file header), RC=0, 0 errors, 33 warnings all pre-existing nodiscard class; binary
/tmp/nvfp4_prefill_bench NOT run — device time belongs to the next window owner.

## 7. Not claimed here
No GPU measurement was taken; §4 derives from banked rows + doc-01 [inference] peaks. The bench
decides PROD-vs-REF; the thermal knee (PLOG-017/027) stays a separate, additive front.
