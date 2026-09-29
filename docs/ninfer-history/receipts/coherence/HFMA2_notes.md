# HFMA2 — packed-fp16 verify-width kernel (nvfp4_small_t_hip_kernel_pk), 2026-09-17, no-GPU desk

## What
`nvfp4_small_t_hip_kernel_pk` in src/ops/linear/nvfp4/nvfp4_small_t_hip.cu: S5's structure
(8 warps x 2 rows/warp, u64 codes, 2x uint4 x/token, scale-quad trick) with the doc 01 §2
v100-skinny inner loop: e2m1 pairs as an LDS **half2** table, bf16 x converted to half2 (D3
bridge, C441 exact-class), **one fp16 accumulation window per 16-code group** (`__hfma2`),
flushed to the fp32 accumulator as `fmaf(lo+hi, coeff, acc)` — coeff = scale*divisor stays a
FULL-PRECISION fp32 contract fold; fp16 never touches the scale plane. TOLERANCE-GATED, not
bit-equal (codec header's order law: this is the declared order change).

## Numerics argument
- e2m1 -> fp16: the 16 values {0,±.5,1,1.5,2,3,4,6} are dyadic, ≤2 significand bits — EXACT.
- bf16 x -> fp16: 7-bit mantissa embeds in 10 bits, exact for exp in fp16's normal range;
  valid envelope |x| in [2^-14, 65504). Production pre-norm hiddens ~±30.
- Window (16 codes): sum of |code*x| ≤ 96·max|x| -> overflow-free while max|x| < 682 (~20x
  margin at ±30). Rounding: 8 packed FMAs + 7 adds at fp16 eps=2^-11 -> ~1e-3-class relative,
  under fp4 quant noise (~1e-1) and under the gate.
- NaN scales (0x7F/0xFF) propagate identically to S5 (coeff fp32 NaN -> output NaN).
- Gate: bench rel-L2(S5, PK) < 1e-2 (A16-criterion class): 5 W4 problems at T=3, fixed-seed
  production-bounded data, dirty divisor 1.177; bench exits rc 1 on any FAIL.

## Static ISA evidence (gfx900 -O3, this desk; /tmp/hfma2_build/*.s, T=3 4096x5120 bodies)
- Probe: __hfma2 lowers to NATIVE v_pk_fma_f16 (0 emulation); flush = 2x v_cvt_f32_f16.
- S5 g-loop: 96 v_fma_f32 + 34 v_mul_f32 + 30 v_add_f32, 0 packed.
- PK g-loop: 48 v_pk_fma_f16 + 6 v_fma_f32 + 2 v_mul_f32 + 36 v_add_f32 + 62 v_cvt_*;
  0 scalar-f16 fallback. VGPR 56 vs S5's 55, 0 spills, LDS 2 KB both.

## Predicted win (to be MEASURED, not quoted)
The multiply-accumulate cluster (SMALLT_row.txt's "3x ALU/weight-byte" issue wall) drops
384 -> ~120 fp32-rate slots per group (-69%); the x-conversion tax adds back ~120-250 slots
(depending on the cvt pipe rate, the one number this desk cannot settle). Net prediction
**1.15x-1.6x GB/s at T=3 if still issue-bound** (147-192 -> ~170-300 GB/s band); if already
DRAM-bound the arm is ~flat. The bench prints S5 vs PK GB/s — that row is the only adoptable number.

## Integration plan for the GPU owner
1. Run the extended bench (same -O3 LAW line; PK A/B section is at T=3): all tolerance gates
   must print PASS, then read the GB/s ratio column.2. Serving A/B WITHOUT dispatcher change: `NINFER_SMALLT_PK=1` flips verify widths T=2..4 to
   PK ([SMALLT] trace prints arm=nvfp4_small_t_hip_kernel_pk); env unset = S5 default,
   byte-identical boot. count600 A/B vs the 128.43 ms round-time anchor.
3. Symbol route: `launch_nvfp4_small_t_pk` (nvfp4_launch.h) forces PK at T=2..4 and routes
   every other width to the production launcher unchanged (T=1 decode GEMV and T=5..32
   prefill chunks untouched by both the env and the symbol).
4. Adopt only if ratio >= ~1.1x AND all gates PASS; then flip the default in launch_exact.

## Risks
- cvt pipe rate on gfx900 decides 1.15x vs 1.6x — no win if already DRAM-bound.
- x outside [2^-14, 65504) -> fp16 inf/subnormal (tolerance gate FLAGS it, not silent);
  production envelope is far inside; S5 stays one env var away. If cvt-bound, the follow-up
  lever is v100-skinny's LDS-activation staging (convert x once per block) — future commit.

## MEASURED (2026-09-17 GPU-OWNER window, PLOG-040) — numerics GREEN, perf RED vs adoption bar
All 5 gates PASS rc 0 (rel-L2 1.14-1.26e-3, ~8x under the 1e-2 bar — the predicted ~1e-3 class exactly).
Ratios: AttnInput 1.09x, GdnInput 1.03x, MlpGateUp 1.02x, Resid6144 1.12x, Resid17408 1.10x — mean 1.07x,
1.02x on the largest geometry => the cvt-pipe tax ate the packed-FMA win (this doc's own "cvt-bound/~flat"
branch). Adoption rule (>= ~1.1x AND gates PASS) FAILS on ratio => default stays S5, NINFER_SMALLT_PK=1
stays non-default, symbol parked, no rebuild/BOOT_BATTERY/serving A/B owed. The named follow-up
(LDS-activation staging, convert x once per block) is a DIFFERENT commit. Full numbers: PK_row.txt +
PK_run_20260917.log.
