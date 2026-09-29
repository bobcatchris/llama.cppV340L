# W37 VERIFY-PATH ROOFLINE ACCOUNTING - the per-kernel budget of the decode cycle - 2026-09-24

Desk roofline (worktree wt-roofline, branch amd/roofline off amd/v340-port-v2
@ 4bf0d6476). ZERO-GPU desk: receipts + code + arithmetic only; no card work,
no kernel builds. Question of record: within the ~80 ms MMVQ/verify term of
the ~124 ms speculative cycle, exactly which kernels burn how much, what
bandwidth do they achieve vs HBM peak, and what are the top-3 kernel
opportunities ranked by expected ms/cycle.

Inputs of record read: W23_mmvq_bw2, W30_bench_output, W26_u1_depth_table,
W32_battery_results, W33_w23env_divergence, TP4_roundmap_2026-09-23
(E-115a), W5_mmvq_bw receipt, W0_census_decode_receipt, W17_fa40codegen,
W15_deepcensus_receipt, dossier v2 (E-141 cycle budget, section 1.3),
ledger E-006/E-104/E-106/E-115a/E-139/E-145/E-148,
src/models/qwen35.cpp, ggml/src/ggml-cuda/mmvq.cu.

## 0. MODEL CORRECTION: THIS IS A DENSE HYBRID, NOT MoE

The desk brief asked for the active-expert fraction (n_expert_active/n_expert).
There is none: the served GGUF is arch qwen35 -> llama_model_qwen35
(src/models/qwen35.cpp), the DENSE hybrid. 64 blocks (n_embd 5120, n_ff
17408) + 1 nextn MTP block; 16 full-attention layers + 48 gated-delta-net
recurrent layers (full_attn_interval 4, qwen35.cpp:31-38); no
FFN_*_EXPS tensor, no routing/permute ops anywhere in the kernel censuses.
The active-expert fraction is therefore 1.0: EVERY weight byte streams once
per target pass. The "MoE expert routing/permute" slot in the cycle
accounting is occupied (at ~1.6 ms/cycle, no lever) by the GDN kernel class
(gated_delta_net_cuda + ssm_conv + fwht + l2_norm).

## 1. THE BYTES/CYCLE FLOOR TABLE (per die, TP4)

Model constants (TP4_roundmap_2026-09-23 section 0): GGUF
12,252,882,720 B = 3.0632 GB/die; 65 blocks; vocab 129272; output head
q6_K 5120x129272 = 135.7 MB/die.

A verify pass at T=4 computes all N output rows for all 4 tokens, so the
full weight set streams exactly once. The draft passes re-read only the
nextn block + (row-sampled) head:

| stream per cycle                          | GB/die  | provenance (measured)                    |
|-------------------------------------------|---------|------------------------------------------|
| target verify pass T=4 (all weights)      | 3.063   | GGUF/4, of-record E-115a                 |
| draft loop: nextn block x 3 steps, T=1    | ~0.19   | census draft MMVQ 3.4 ms/round at 54.7 GB/s |
| catch-up pass (draft ctx, T=4)            | ~0.07   | census 1.2 ms/round                      |
| draft head re-reads x 3 (row-sampled)     | ~0.11   | 3 x 695 us/launch census (T=4 head full = 1379.5 us at 98 GB/s; the 695 us draft launches read ~1/3-1/4 of rows) |
| x-side traffic (q8_1 of activations)      | ~0      | bytes negligible; cost is launch-bound (quantize_q8_1 2.4 ms/cycle, 358k calls/38 s census) |
| **MUST-STREAM total**                     | **~3.42** |                                        |

Floor at each bandwidth anchor (per cycle, per die):

| anchor                        | GB/s | floor ms/cycle | provenance                |
|-------------------------------|------|----------------|---------------------------|
| HBM2 pin-rate upper bound     | ~966 | 3.5            | desk brief "~1 TB/s" (unverified online; search rate-limited. Class spec quoted for Vega 64/MI25 is ~484; no conclusion here depends on which is right) |
| vendor class spec             | ~484 | 7.1            | AMD quoted spec, same hedge |
| measured coalesced consume    | 327  | 10.5           | E-006 pure-consume ablation, this stack |
| W0 measured ceiling           | 183.8| 18.6           | W0_census_decode_receipt (TP3 era) |
| zero-compute schedule band    | 70-120 | 28.5-48.9    | W5 A4 cfull ablation, real kernel |
| **today's MMVQ wall**         | **54.8** | **60.6**   | census (measured)         |

Headline answer: the MMVQ pool runs at **54.7 GB/s effective on the verify
pass (3.063 GB / 56.0 ms; 54.8 GB/s over all 3.42 GB / 60.6 ms)**. That is
**6-18% of any plausible HBM peak and 6.0x off the measured 327 GB/s
consume wall**. MMVQ is in the "5-10% of peak" case: this is a
launch/occupancy/schedule problem, NOT a memory-optimal term. Kernel-level
decode is NOT done. But the win is bounded: the same schedule with all
compute deleted (cfull) still only reaches 70-120 GB/s, so inside THIS
schedule family the pool caps at 25.5-43.8 ms (-12.2 to -30.5 ms vs today).
Getting past 120 GB/s needs a different schedule (the banked W5 LDS-y
staging design) or a format change (owner-guarded).

Whole-cycle view: 3.063 GB in 123.1 ms = 24.9 GB/s per-die average (dies
87% busy; ~half the busy time is MMVQ).

## 2. ACHIEVED vs PEAK PER MAJOR KERNEL (current fa40+ub1024 config)

Cycle of record: 123.1 ms = 3.04 tokens / 24.69 t/s (121.5 ms at the
battery cells' mean_len 3.00). MMVQ is untouched by fa40/ub1024; attention
and small-kernel rows are post-fa40 estimates on the census base.

Per-type MMVQ (verify pool, W23 table = census E-115a; small types from
roundmap all-pass ms which are the plausible allocation):

| type    | GB/die | ms/round (verify) | GB/s   | cfull ceiling GB/s (W5 A4) | % of cfull |
|---------|--------|--------------------|--------|----------------------------|------------|
| IQ3_S   | 0.890  | 17.6               | 50.6   | 79                         | 64%        |
| IQ4_XS  | 0.818  | 10.9               | 75.0   | 98                         | 77%        |
| IQ3_XXS | 0.615  | 14.8               | 41.6   | 76                         | 55%        |
| Q4_K    | 0.415  | 8.5                | 48.8   | 99                         | 49%        |
| Q6_K    | 0.145  | 3.7 (incl heads)   | 39.2   | 120                        | 33% (worst ratio) |
| Q3_K    | 0.095  | ~2.1 all-pass      | ~45    | 71                         | 63%        |
| Q5_K    | 0.052  | ~1.8 all-pass      | ~29    | 109                        | 27%        |
| rest    | 0.03   | ~0.7               | ~43    | -                          | -          |

(W23's ms for Q3_K/Q5_K - 0.5/0.3 - imply 190/173 GB/s, physically
excluded; the roundmap census allocations t11 2.1 / t13 1.8 ms are used
instead. The 5 big types carry 94% of bytes and 55.5 of 57 ms, so the tail
does not move anything.)

Bench-cell cross-check (W30 fresh medians + W7/W5, K=5120 N=17408 T=4,
real kernel): iq4_xs base 47.3 MB / 610.4 us = 77.6 GB/s; iq3_s base 38.3
MB / 1755.6 us = 21.8 GB/s (share arm 687.6 us = 55.7 GB/s); q4_K base
50.1 MB / 1079.1 us = 46.5 GB/s. Bench and served agree - no
bench-to-serving multiplier (W23 section 2 law holds).

Occupancy of the served templates (W30 OCC dump, gfx900): iq4_xs 84 regs /
6 CTAs/CU, iq3_s 77 / 6, q4_K 112 / 4 (of 40 waves: 15-30% occupancy),
__launch_bounds__(..., 1) at mmvq.cu:477. lb2 is a codegen no-op; lb4
forces 8 CTAs but q4_K spills 256 B local. Mechanism of record (W23 M1/M2,
unchanged): 1.25 kbx iterations of runway per lane at K=5120 x
register-capped occupancy = latency starvation, not DRAM saturation.

Full-cycle accounting (~123.1 ms, non-overlapping rows; MEASURED = census/
guard, ESTIMATE = derived, arithmetic shown):

| row | component | ms/cycle | share | status |
|-----|-----------|----------|-------|--------|
| 1 | mul_mat_vec_q, ALL passes (verify 56.0 + draft 3.4 + catch-up 1.2) | 60.6 | 49.2% | MEASURED (census E-115a) |
| 2 | RCCL allreduce wall (136 boundaries) | 22.7-25.6 (E-141 of-record wall 28.7-31.2) | 18-25% | MEASURED census / of-record probe - NOT THIS DESK |
| 3 | attention: fattn-tile fa40 q4_0-direct verify 16 x ~475 us + catch-up 0.45 + draft 3 x 60-67 us + combine ~0.4 | ~10.7 | 8.7% | ESTIMATE: 16.1 census pre-fa40 minus 37.5% on the 12.2 ms verify tile (W17 A4) minus 0.27 catch-up minus ~0.8 dequant launches now deleted (fa40 need_f16=false); band 9.5-11.5 |
| 4 | small kernels + copies (quantize_q8_1 2.4, rms_norm 1.8, k_bin_bcast 1.2, cpy_scalar 1.1, copyBufferRect 1.1, get/set_rows 1.4, concat 0.7, GDN 1.6, rope 0.4, tail) | ~10.5 | 8.5% | ESTIMATE: 11.3 census minus ~0.8 fa40-deleted dequant |
| 5 | host slice A (post-verify drain/sample/accept/build/issue, all 4 dies idle) | 6.5 | 5.3% | MEASURED (census; E-078 corroboration) |
| 6 | draft-loop host/latency slice (8.9 wall minus ~4.7 device already counted in rows 1/3) | 4.2 | 3.4% | MEASURED/derived (census 2.98 ms step period, ~1.4 ms host/step) |
| 7 | catch-up inter-pass gap | 1.2 | 1.0% | MEASURED (census) |
| 8 | microgaps/launch gaps in-verify | ~3 | 2.4% | ESTIMATE (residual) |
|   | TOTAL | ~119-127 | 100% | sums to the 123.1 ms cycle within the 22.7-vs-28.7 allreduce question and ~2 ms census rounding |

Reconciliation notes: (a) the dossier's "~80-85 ms MMVQ" (65%) is the
BROAD bucket - row 1 + row 4 + the in-verify slices of rows 5/8 (~80 ms of
verify-wall compute); the strict mul_mat_vec_q pool is 60.6 ms. The
roofline verdicts are identical on either reading. (b) The draft model
total = 8.9 ms wall (rows 1+3 device inside it + row 6 host); its two
biggest single cells are the row-sampled head (695 us x 3) and nextn-block
MMVQ (~0.43 ms/step). (c) No MoE routing/permute term exists (section 0).

## 3. RANKED TOP-3 KERNEL OPPORTUNITIES (each with arithmetic chain + falsifiable prediction)

Calibration law first: the only served measurement of a pool-arithmetic
prediction is w23env (E-145/W32/W33): predicted -3.4 ms/round, measured
+0.08 t/s (+0.32% ~ -0.4 ms) - bench-CELL gate wins transferred at ~1/8 to
the served POOL. Whole-schedule changes (share at TP3, E-104) transferred
fully. Predictions below are banded accordingly.

### Opportunity 1: whole-pool MMVQ schedule/occupancy lift into the cfull band

- What: attack all 60.6 ms, not per-type gates. First rung (bench-cheap,
  W23 B2 recipe written): the C3 wide-s2r y-preload arm + the M2
  launch_bounds/occupancy probe. Main rung: the banked W5 LDS-y staging
  design (52 B pad, stride 13, occupancy budget 7 CTAs/CU) or any register
  diet that holds 8 CTAs/CU without the lb4 spill.
- Arithmetic chain: pool 60.6 ms at 54.8 GB/s; cfull proves 70-120 GB/s is
  reachable on this schedule with zero compute; entry at 70-80 GB/s puts
  the verify pool at 43.8-38.3 ms (-12.2 to -17.7) and all-pass at ~47-43
  ms. Served haircut band: full transfer (-10 to -14 ms) to the w23env
  1/8-class partial (-3 to -6 ms).
- Expected: -3 to -14 ms/cycle -> 25.4-27.4 t/s (+3% to +11%).
- Class: REAL KERNEL WORK (M/L); the first rung is a bench arm + env gate.
- Falsifiable prediction: if the iq3_s/iq3_xxs classes are latency-starved
  at 42-51 GB/s because of runway x occupancy (M1+M2), then an arm that
  both widens y-preloads and holds >= 8 CTAs/CU (no spill) moves the
  served pool 60.6 -> <= 50 ms with byte-identical output, cycle
  123.1 -> <= 112.7, t/s >= 27.0 (+9.4%). If the served delta is < +2%,
  M1+M2 are the wrong model for served shapes (the pool is already in the
  cfull-attributable band) and the whole occupancy family closes - one
  paired window + one bench session decides.

### Opportunity 2: small-kernel + copies class (quantize_q8_1 dedup, copy/convert elimination)

- What: (a) quantize_q8_1 re-quantizes x ~3x/layer/die (W0 T4 finding);
  cache/fuse to once per graph section: 2.4 -> ~0.8 ms. (b) The copies
  class cpy_scalar 1.1 + copyBufferRect 1.1 + get/set_rows 1.4 + concat
  0.7 = 4.3 ms; roundmap lever 5 prices 2-3 ms recoverable by deleting
  redundant convert/copy nodes in the TP4 graph.
- Arithmetic chain: -1.6 (quantize) + -2.0..-3.0 (copies) = -3.6 to -4.6
  raw; haircut for graph-capture risk (E-090 lesson) -> -2.5 to -3.5 ms.
- Expected: -2.5 to -3.5 ms/cycle -> 25.4-25.5 t/s (+2.0-2.9%).
- Class: graph/scheduler-level edits, NOT env-gated; moderate risk
  (capture/replay), no numerics change.
- Falsifiable prediction: if x re-quantization is truly 3x redundant and
  the four copy nodes are graph artifacts, a one-boot -lv 4 timeline shows
  >= 5.5 ms/cycle in these launches, and the fused/cached build lands
  >= -2.5 ms served (>= +2.0% t/s) with identical logits sha. If the
  timeline shows < 3.5 ms in the class, the lever is dead - measured
  before any kernel work.

### Opportunity 3: attention tile concurrency at serve depth (post-fa40 ~10.7 ms)

- What: W17 proved the q4_0-direct tile is LATENCY/serial-bound, not
  DRAM-bound: its KV stream runs at 1.87-2.35 GB/s effective; the modeled
  15.2x byte reduction landed as only -35..-40% wall. A concurrency arm
  (more in-flight KV blocks per launch: V-tile split across more CTAs,
  double-buffered K/V stages, or wider T per CTA) attacks the remaining
  stall, keeping the v11 scalar-half2 shared-store pattern (the
  ggml_cuda_memcpy_1<8> miscompile wall, W17).
- Arithmetic chain: verify tile ~16 x 475 us = 7.6 ms at serve depth
  (7857-prompt class); -30..-50% per launch if stall-bound = -2.3 to
  -3.8 ms; + catch-up/draft ~0.6 ms untouched.
- Expected: -2 to -4 ms/cycle -> 25.4-26.0 t/s (+1.6-3.3%). Grows with
  depth (unbounded term at 200k: ~240 ms/cycle post-fa40, W15/W17).
- Class: REAL KERNEL WORK (M) under the W17 oracle discipline (staged
  KT/VT/KQ dumps + dst-is-truth; every arm oracled in-session).
- Falsifiable prediction: if the tile is latency-bound, doubling in-flight
  KV blocks shows >= -25% per launch in bench_attn_real at d=7168
  (552.4 -> <= 414 us) AND >= -2 ms/cycle served; if rocprof shows
  issue_active > 80% (issue-bound) or the served delta is < +0.5%, the
  lever is dead - one bench session + one paired window decides.

Not top-3, with reasons: C1+C2 env gates (iq4_xs share + s2r) are ALREADY
MEASURED served: +0.32% (+0.08 t/s), promotion blocked on in-window sha,
"noise-level" per W33 - keep the gates set only if a future full-guard
window proves them free. GLU fusion at T=4 and lb2 are DEAD (W30: fused
iq4_xs oracle-FAIL, fused iq3_s +21.8%, lb2 no-op) - do not re-bench.
q6_K/head cells (39.2 GB/s, worst ratio to cfull) ride inside
Opportunity 1; a dedicated head-split kernel prices at < 1 ms/cycle.
Allreduce (22.7-31.2 ms/cycle) is separately owned - excluded here.
Host slices (rows 5+6, 10.7 ms) are not kernel work but are the largest
non-MMVQ non-comm pool; a T3-style host desk prices -3 to -5 ms.

## 4. STRATEGY VERDICT

The cycle is compute-schedule-bound, not memory-bound: MMVQ at 54.8 GB/s
has 6.0x headroom to the measured consume wall but only 1.3-2.2x to the
schedule-family ceiling - so the decisive kernel work is a schedule
replacement (Opportunity 1 main rung), while Opportunities 2 and 3 are the
affordable complements. Realistic all-kernel stack: -8 to -21 ms/cycle ->
26.0-30.0 t/s (+5% to +21%), consistent with the roundmap's independent
129 -> 100-108 ms projection. The owner's ~30 t/s is reachable only if
Opportunity 1 lands near the top of its band AND the allreduce desk (other
owner) compresses the wall; kernel work alone at the cfull-band floor ends
at ~27-28 t/s.

## 5. PREDICTION GATES FOR FUTURE PAIRED WINDOWS (summary)

| # | if measured ... | then cycle drops to | t/s becomes | falsified if |
|---|-----------------|---------------------|-------------|--------------|
| 1 | MMVQ pool >= 70 GB/s (60.6 -> <= 50 ms) | <= 112.7 ms | >= 27.0 | served +< 2% |
| 2 | quantize+copies class -2.5..-3.5 ms | 119.6-120.6 | 25.4-25.5 | class < 3.5 ms in timeline boot |
| 3 | tile launches -25..-45% at d=7168 | 119.1-121.1 | 25.2-25.6 | issue_active > 80% or served +< 0.5% |
