# TILED SWEEP — tile-shape variants for the production prefill GEMM (no-GPU desk, 2026-09-17, amd/tp4-cure)

## 1. Variants (`launch_nvfp4_tiled_gemm_variant(x, w, out, stream, id)`, same TU; V0 = shipped path untouched)
| id | block tile | thread tile | K-tile | point of the shape |
|----|-----------|-------------|--------|--------------------|
| V0 | 64r x TN  | 4r x TN/16t | 64  | shipped (75.7 tok/s serving) — id 0 routes to it byte-identically |
| V1 | 32r x TN  | 4r x TN/32t | 64  | 2x blocks, halved x-tile LDS/block, smaller acc file |
| V2 | 128r x TN | 4r x TN/8t  | 64  | half the blocks, 64-float acc/thread, 16-tok pair-table amortization |
| V3 | 64r x TN  | 4r x TN/16t | 128 | two m128x4 scale segments/k-step (quads 512 B apart), half the k-steps |
| V4 | 64r x TN  | 4r x TN/16t | 64  | x-tile double buffer one k-step ahead: ONE barrier per k-step, staging overlaps FMA |
All ids keep the k-ascending fp32 accumulation order => outputs are BIT-IDENTICAL across ids;
the relL2-vs-REF column must print the SAME value V0..V4 (consistency check, same 1e-2 bar).
Dropped at compile time by the LDS/accumulator law, documented not shipped: TN=256 for V2 (acc 128/thr),
V3 (x-tile 64 KB), V4 (2x x-tile >64 KB). The sweep targets M=128 only.

## 2. Predicted winner (falsifiable; the wall at M=128 is issue/latency, not DRAM)
V4 > V3 > V0 > V1 > V2. V4: only variant that both halves the barrier rate (1 vs 2 per 64-K) AND
hides the global x-load latency behind the previous k-step's FMA nest (cost: 35 KB LDS @128 => 1 block/CU).
V3: same barrier RATE as V4 (2 per 128-K) + longer unrolled nest, staging latency still exposed.
V1: V0's occupancy already fine (19 KB, ~3 CTA/CU) => halved pair-table amortization (4 vs 8 tokens/strip).
V2: 64-float acc + addressing => VGPR pressure, likely 1 CTA/CU => last.
FALSIFIER (a): V3/V4 <= V0 by >5% => 35 KB LDS occupancy beats the barrier/overlap win; winner stays V0,
next knob is a SMALLER x-tile footprint, not more shapes. FALSIFIER (b): all within noise of V0 =>
limiter is the FMA/pair-table issue stream itself; stop sweeping shapes, cut decode ops per FMA instead.

## 3. Owner device-test plan (ONE bench session)
1. Granted window: `nvfp4_prefill_bench_sweep --sweep` (M=128 x 4 big geometries x V0-V4, GF/s +
   relL2 PASS bar; ceilings + 5 s mclk hammer included, ~1 min total). NOT yet run — zero-GPU desk.
2. Winner = max GF/s among PASS rows. Promote into `launch_nvfp4_tiled_gemm` default (ids 1-4 are
   one config-type change in nvfp4_tiled_gemm_hip.cu; dispatch seam untouched), rebuild,
   BANK-BEFORE-RELINK per docs/amd/BOOT_LAUNCH_RUNBOOK.md §4.
3. Boot TP4 from the banked artifact: prefill tok/s vs the 75.7 baseline + ladder battery +
   NINFER_TILED_VERIFY=1 arm once on the winner; bank per-LABEL. Order unchanged => relL2-class
   check suffices (no new bit-cell owed).

## 4. Compile status (zero-GPU, -O3 LAW command: ONESHOT_AR_notes §5 form, same include set, nothing added)
RC=0, 0 errors; 143 warnings = 126 nodiscard class (HEAD baseline 70, same double-per-line pattern;
the +56 are run_sweep's own cudaMalloc/Memset/Memcpy/Free/EventCreate ignores — same documented class)
+ 15 pre-existing -Wpass-failed in nvfp4_small_t_hip.cu (count unchanged). Object census: 21 shipped
+ 21 V1 + 14 V2 + 14 V3 + 14 V4 = 84 tiled kernels. Binary /tmp/nvfp4_prefill_bench_sweep — NOT executed
(no grant; TILED_GEMM_notes §6 incident precedent).

## 5. MEASURED (2026-09-17 GPU-OWNER window, PLOG-039) — prediction FALSIFIED, falsifier (a) FIRED
V0 wins ALL FOUR geometries by 15-41% (V0 3083/2756/3103/2366 GF/s vs V4 2064/1908/2008/1491,
V3 1921/1786/1876/1391 — V4/V3 33-41% BELOW V0, not above). Actual order: V0 > V2 >= V1 > V4 > V3.
The 35 KB-LDS occupancy cost beats every barrier/overlap win, AND the falsifier-(a) follow-up knob
(SMALLER x-tile = V1) also lost 16-26% — tile-shape family CLOSED, V0 ships, no promotion, no rebuild.
relL2 0.00e+00 on all 20 rows (bit-identical as designed). Full numbers: SWEEP_row.txt + SWEEP_run_20260917.log.
