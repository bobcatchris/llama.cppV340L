# TILED GEMM — the production large-M prefill dequant GEMM for gfx900 (no-GPU desk, 2026-09-17, amd/tp4-cure)

## 1. What landed
- `src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.{h,cu}` NEW: `nvfp4_tiled_gemm_hip_kernel<STILES,TN>`
  + `launch_nvfp4_tiled_gemm` (all 15 registered problems; token tiles 32/128/256; throws otherwise;
  HipSources.cmake NOT touched — membership is the owner's flip, §4).
- `nvfp4_prefill_bench.cu`: third arm "TILED" — whole-M launches (no chunking) at M∈{32,128,256}
  × 4 geometries + per-shape rel-L2 self-check vs REF (PASS bar < 1e-2). PROD/REF arms untouched.
- Compile: the EXACT -O3 law command (bench header), RC=0, 0 errors, 35 warnings = 33 pre-existing
  nodiscard class + 2 new same-class (added cudaMemcpy self-check). All 21 kernel
  instantiations (7 STILES × 3 TN) present in /tmp/nvfp4_prefill_bench_tiled.

## 2. Design (TILED-GEMM-1)
- Block = 64 output rows × TN(=M) tokens, grid = n/64, 256 threads. K-tile = 64 = exactly one
  m128x4 scale-tile segment ⇒ per row per k-tile ONE aligned u32 scale quad (GEMV-TUNE iter8).
- Activations staged ONCE per k-tile: LDS TN×32 u32, PACKED bf16 pairs, XOR swizzle
  (`word ^ (t&31)`) — without it token-major 32-word rows are column-aligned (every token's
  word 0 = same bank). Staged via uint4 (8 bf16) global loads, 2× the tuned GEMV's pattern.
- Weights streamed, never staged: u64 (uint2) code load per row-group + scale quad, decoded
  through the LDS tables (e2m1 pairs 2 KB + e4m3 scales 1 KB) filled FROM
  `amd::e2m1_bits` / `amd::e4m3_lut_decode` — the u32 fixed-name seam is the arithmetic home.
- Thread tile 4 rows × TN/16 tokens (thread = (rt=tid&15, ct=tid>>4)). FMA nest
  g(4 groups) → v(8 k-pairs) → t(TN/16 tokens) → r(4 rows): ONE x-pair b32 feeds 8 FMAs
  (4 rows × 2 values); one pair-table entry per row per k-pair amortized over TN/16 tokens
  ⇒ ~5-6 FMA per LDS op vs REFERENCE ~1 (+ a global/L1 x read per FMA).
  coeff = scale × inv_div folded once per (row, group) — the codec :136 shape.
- LDS budget: 3 KB tables + TN×128 B x-tile = 3 KB (M=32) / 19 KB (M=128) / 35 KB (M=256)
  vs 64 KB per CU. Occupancy estimate: 4 waves/block; ~70-130 VGPR (acc 8/32/64 + coeff 16 +
  addressing) ⇒ M=128: ~2-3 blocks/CU resident; M=256: 1 block/CU (35 KB ⇒ no 2nd block) —
  the thinnest point, compensated partly by 4-row ILP per thread.
- Numerics: fp32 accumulation, order-free per the mission brief; decode VALUES are bit-identical
  to contract (tables from the contract fns), summation ORDER differs from small_t — declared
  HERE at the gate (codec-header law). No bit-equality claim; the bar is the bench's rel-L2
  vs REF + end-to-end coherence.

## 3. Expected vs reference (predictions — falsifiable on first bench run)
- M=128 GFLOP/s: TILED ≥ 2-4× REF and ≥ ~10× PROD (PROD issue-bound at the ~40 tok/s ceiling,
  PREFILL_FRONT §4); absolute 10-40% of the nominal 10.75 TF/s fp32 anchor. FALSIFIER:
  TILED ≤ REF at M=128 ⇒ the FMA-nest/LDS argument is wrong — dump the ISA before resizing.
- GB/s FALLS M=32→256 for TILED while GF/s RISES (compute wall visible); PROD GB/s flat (M=1
  intensity). rel-L2 vs REF expected ≤ ~1e-3 (fp32-accum both, dequant bit-identical, only
  summation order differs).

## 4. What the GPU owner must do
1. Bench (granted window; 5 s mclk hammer discipline already in the binary): decisive column =
   TILED vs REF vs PROD GFLOP/s at M=128; every TILED line must print PASS (relL2 < 1e-2).
2. Integration (NOT applied — owner's files): add nvfp4_tiled_gemm_hip.cu to
   src/HipSources.cmake; in nvfp4_dispatch.cpp `launch_a16`, route token counts ≥ 32 (or the
   whole 128-chunk) to `launch_nvfp4_tiled_gemm` instead of slicing T=32 small_t views; keep
   small_t for T≤4 verify + 5..31 tails. One seam, ~5 lines; gemv/small_t byte-untouched.
3. End-to-end: prefill-chunk-128 boot; prefill tok/s vs the 11.3 mean / 24.5 plateau baselines
   with sidebands; ladder + known-answer coherence per the standing battery; bank per-LABEL.

## 5. Risks
- [design] M=256 occupancy: 35 KB LDS ⇒ 1 block/CU; if GF/s stalls at 256 vs 128, try TN=128
  with double-buffered k-tiles (2×16 KB x-slabs) before touching the thread tile.
- [design] 2-way bank conflicts on x-tile reads (64 lanes / 32 banks at fixed (g,v)); the
  swizzle function is the first tuning knob. Scalar 2-B epilogue stores: tiny traffic, ignore
  unless a profiler disagrees. Scale plane read 4 B/row/k-tile ⇒ ~50% sector efficiency
  (~+5% DRAM); knob: cooperative scale staging into LDS.
- [design] VGPR pressure at kS=16 (M=256): if the ISA spills, drop to 2 rows/thread × TN/8.
- [inference] %nominal uses doc-01's 10.75 TF/s; measured SCLK 1269-1350 (PLOG-031) ⇒ the
  percentage is optimistic by up to ~15%.

## 6. Incident note (honesty row)
During compile verification the built binary was ACCIDENTALLY executed once, without a GPU
grant (a `| head -2` pipeline did not prevent the boot): it enumerated HIP device 0 (V340 die,
56 SMs), ran the ~5 s mclk hammer + copy/read ceilings, died on SIGPIPE at the first stdout
buffer flush (~6-60 s unplanned load, ~12:35:xx local 2026-09-17). BOOTBAT_20260917_123502
(started 12:35:02 on this host) overlapped that interval and still finished 14/14 PASS — no
harm visible, but the owner should know the overlap happened. No further device runs; the
TILED kernel itself most likely never launched (death preceded the timing prints).
