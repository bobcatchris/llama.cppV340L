# W26: U1 MEASURED DECODE-AT-DEPTH TABLE (fa40 var-11 of-record, live-mined)

Status: IN PROGRESS - mined live during the U1 depth window (cells land every
40-90 min; this file is updated as each cell lands). Final at U1-DONE.

## Provenance (per-cell stamps)

- window: U1 depth cells, one server boot (pid 56419, ctx 200000, port 8081),
  780 s soak settles between cells, E-134e pre-boot gates.
- config of record: NEW of-record arm = GGML_CUDA_FATTN_TILE_Q40_DIRECT=1
  (fa40 var-11 q4_0-direct decode tile) + TP4 (tensor split, 4 dies) + RCCL
  allreduce + MMVQ share envs + draft-MTP, b512/ub512, -ctk q4_0 -ctv q4_0
  -fa on, canonical-200k-inline. BASEENV line stamped in every cell log.
- binary: 049075824e23da47, constant across all cells landed so far. Cell
  provenance commit drifts (2e27d757b633 at 10k, 4bf711608759 at 50k) but the
  drift is tree/docs-only - same binary hash both cells.
- recipe: cache_prompt=false n_predict=128 temperature=0 per cell (decode_guard
  at depth). decode tps, ms_per_round and draft accept are the guard's own
  fields; decode tps = mean_len x 1000 / ms_per_round (identity holds at 10k:
  3.04 x 1000 / 127.8 = 23.78).
- law: E-124 deep-census law on the OLD (pre-fa40) config: +1.95 ms/1k beyond
  ~9.4k, 130.6 ms/round at 10.235k, ~503 ms/round at 200k. law_check delta_pct
  = (measured - law)/law; negative = under the law = the fa40 signature.
- adjudication: void_gate.py E-133 die-3 sclk soak rule, default die-3 group =
  LAST column group of the 18-col sideband (verified via --help). This desk
  independently re-ran the adjudicator per cell; it reproduces the in-log
  verdicts exactly at the in-log windows (10k [55.1,70.5] PASS act-mean 1204
  duty 0%; 50k [1736,1746] VOID act-mean 991 duty 100%).

## Depth table (decode 128 tok, temp 0, TP4+RCCL, fa40 var-11)

Cell status marks: MEASURED = decode leg valid; PREFILL-VALID+DECODE-EOS-VOID =
prefill tps honest, decode leg killed by a deterministic EOS stop (see anomaly
section).

| depth    | prefill tps | decode tps | ms/round | law delta | void-gate | draft accept | status |
|----------|-------------|------------|----------|-----------|-----------|--------------|--------|
| 10.2k    | 170.35      | 23.78      | 127.8    | -2.2%     | PASS      | 0.68 (len 3.04) | MEASURED |
| 51.2k    | 29.41       | EOS-VOID   | EOS-VOID | EOS-VOID  | VOID*     | EOS-VOID (n=0)  | PREFILL-VALID+DECODE-EOS-VOID |
| 102.4k   | 15.63       | EOS-VOID   | EOS-VOID | EOS-VOID  | VOID*     | EOS-VOID (n=0)  | PREFILL-VALID+DECODE-EOS-VOID |
| 153.6k   | pending     | pending    | pending  | pending   | pending   | pending      | running (EOS-risk) |
| 199k     | pending     | pending    | pending  | pending   | pending   | pending      | pending (EOS-risk) |

(*u1_50k void verdict adjudicates only the prefill tail - there is no decode
window to adjudicate. See anomaly section.)

## u1_50k + u1_100k anomaly: DECODE-EOS-VOID (logged 12:12-14:20, 2026-09-24)

Both deep cells landed with decode_guard tps=1000000.00 / ms_per_round=0.0 /
draft_accept 0.0000 draft_n=0, VOID-GATE VOID. Root cause (this desk's
server-log read, coordinator-confirmed 12:40): the model (temperature 0,
deterministic) emitted EOS as the FIRST token after the deep filler prefill -
server log shows `eval time = 0.00 ms / 1 tokens (1000000.00 tokens per
second)` then `stop processing`, n_tokens = prompt length, truncated = 0. The
cell request lacks ignore_eos, so the 128-token decode never ran. Not a
server fault, not a thermal kill.

- u1_50k: window [1736,1746] holds only prefill tail (die 3 pinned 991 MHz
  after 29 min prefill; max junc 96C, max mem 98C).
- u1_100k: window [6547.1,6557.2], same shape (die-3 group act-mean 775 MHz,
  duty 100%; c2 also 991/100%; max junc 92C, max mem 98C). This desk's
  adjudicator re-run reproduces the VOID at the in-log window.
- Both prefill legs ARE valid: 29.41 tps @ 51203 (1741 s, 2.29x the W24 fresh
  model) and 15.63 tps @ 102395 (6552 s, 2.48x the W24 fresh model 2642 s -
  slightly ABOVE their x2-2.3 served band; handed to the W24 prefill desk).
- Pattern status: 2/2 deep decodes EOS-killed, 10k decoded fully -> the
  stop looks systematic (depth-dependent greedy EOS after deep filler
  prefill), not per-cell chance. 150k/199k expected to die the same way.
- E-140 recommendation TRIGGERED: re-run the deep decode legs with the
  coordinator's /home/chris/run_u1_window_v2.sh (adds ignore_eos to the
  /completion request; bash -n clean; no --cells argument - rerun mechanics
  are the coordinator's). Until then the decode table has exactly one valid
  depth point (10k) and deep decode-at-depth stays UNMEASURED on this
  config. Do not interpolate over the holes silently.

## Registered prediction: draft acceptance vs depth (added 13:00, pre-unblind)

Cross-project evidence (ninfer 103-row acceptance study, law L4): MTP
acceptance RISES with context - 54.7 @10k -> 63.8 @40k -> 73.3 @59k -> 80.5
@72k (+25 pts), reproduced across their artifacts (longer context ->
more constrained continuations -> easier proposals). Nothing else moved
acceptance this far this reliably on their stack.

REGISTERED PREDICTION (written before the deep cells unblind): if L4
transfers to llama.cpp draft-mtp, U1 draft_accept climbs ABOVE the 0.66-0.68
short-context band as depth grows - e.g. 0.70+ @50k, 0.75+ @100k on a strong
transfer. Falsifiable tonight from the per-cell verdict blocks. Extract
draft_accept + mean_len for every cell and read the trend against depth.

- Caveats: EOS-killed decodes carry draft_accept 0.0 (u1_50k) and are EXCLUDED
  from the trend, not points on it. The 10k anchor is 0.68. Resolution is
  coarse: +/-2-3 pts per cell at draft_n ~125.
- If the trend HOLDS: deep decode is better than the E-124 law assumes. The
  law models ms/round only; rising acceptance raises tokens/round (mean_len)
  on top, compounding with the fa40 ms/round cut - the deep decode outlook
  improves beyond whatever the law_check deltas show.
- If the trend does NOT hold: that is a real stack difference, also worth
  recording - ninfer's draft head is a separate sliced table, ours ships in
  the GGUF; acceptance depth-dependence does not automatically cross that
  architecture gap.

Acceptance ledger (filled per cell as cells land):

| depth  | draft_accept | mean_len | draft_n | counts? |
|--------|--------------|----------|---------|---------|
| 10.2k  | 0.68         | 3.04     | 125     | yes     |
| 51.2k  | (0.00)       | (1.00)   | 0       | NO - EOS-voided, excluded |
| 102.4k | (0.00)       | (1.00)   | 0       | NO - EOS-voided, excluded |
| 153.6k | pending      | pending  | pending | -       |
| 199k   | pending      | pending  | pending | -       |

EOS watch tally (deep decodes): 50k DEAD, 100k DEAD - both stopped at exactly
1 token (eval time 0.00 ms, sentinel tps=1e6, draft_n=0). 10k decoded fully.
The "independent coin flip" reading is weakening: 2/2 deep decodes hit
deterministic greedy EOS right after a deep filler prefill, which looks
systematic (depth-dependent), so 150k/199k are expected to die the same way
unless depth flips the pattern. The registered L4 acceptance test therefore
has no deep points unless the v2 (ignore_eos) rerun runs - the prediction
stays registered and untested in-window.

## fa40 signature analysis (model vs measured; updated per cell)

W17 oracle-gated the direct arm at -37.5%/launch at serve depth 7168
(884.0 -> 552.4 us), -35..-39% at 200k. The round-level law delta this should
produce, scaled by the tile kernel's depth-dependent share of the round
(E-124: 17 launches/die x 722 us x d/7168; round per law):

| depth | tile ms/round (model) | law ms/round | expected delta @ -37.5% | measured |
|-------|-----------------------|--------------|-------------------------|----------|
| 10.2k | 17.5                  | 130.6        | -5.0%                   | -2.2% (LAW-OK, noise band) |
| 51.2k | 87.6                  | 210.5        | -15.6%                  | EOS-VOID (no decode) |
| 102.4k| 175.2                 | 310.3        | -21.2%                  | EOS-VOID (no decode) |
| 153.6k| 262.8                 | 410.2        | -24.0%                  | pending  |
| 199k  | 341.2                 | 498.7        | -25.7%                  | pending  |

Reading guide: at 10k the tile is ~13% of the round so fa40 buys ~4-5% and the
-2.2% measurement is noise-level; the signature only separates from noise when
the tile share grows. A deep cell at -20..-30% vs law is the fa40 win measured
in the wild; a deep cell at or ABOVE the law is fa40 not helping (or a soak
regression - cross-check the void-gate before believing any above-law cell,
E-133: throttled die 3 alone costs ~26%).

Small additive term not in the model: the direct arm also deletes the f16-pool
dequant launches (E-124: 3.98 us/1k/launch, 7% of the deep delta), so the true
expected delta sits slightly above (-37.5% x tile share) at every depth.

## Disclosure

All numbers in this file are the NEW of-record config (fa40 var-11
q4_0-direct decode arm, TP4 tensor split, RCCL allreduce, MMVQ share envs,
draft-MTP) at b512/ub512, canonical-200k-inline, measured live in the U1 depth
window on one persistent server boot (ctx 200000, port 8081, 780 s inter-cell
soaks). The E-124 law they are differenced against was anchored on the OLD
pre-fa40 config; delta_pct therefore reads the config delta at depth, not
absolute law compliance. Depth cells are informational (no same-config
baseline exists at depth); decision gates still route through the paired-window
protocol elsewhere in the campaign.

## COORDINATOR FINALIZATION (2026-09-24 15:50 - window closed partial by coordinator stop)

| cell | depth | prefill t/s | decode t/s | ms/round | law delta | void | accept |
|---|---|---|---|---|---|---|---|
| u1_10k | 10235 | 170.35 | **23.78** | 127.8 | -2.2% LAW-OK | PASS | 0.680 |
| u1_50k | 51203 | 29.41 | EOS-VOID | - | - | VOID (soak) | - |
| u1_100k | 102395 | 15.63 | EOS-VOID | - | - | VOID (soak, die3 775 MHz 100%) | - |
| u1_150k | - | NOT RUN (oomd killed the chain 17 min into prefill) | | | | | |
| u1_199k | - | NOT RUN | | | | | |

STATUS: the window closed PARTIAL. Deep decodes were killed by SYSTEMATIC
depth-dependent greedy EOS (2/2, eval=1 token) - run_u1_window_v2.sh
(ignore_eos) is REQUIRED and scheduled after the lever battery. The
acceptance-vs-depth prediction (L4) and the fa40 deep signature both remain
UNTESTED until that rerun. The prefill points at 50k/100k are valid and feed
the W24 curve (both sit near the fresh-thermal model once the soak multiplier
is removed: 50k measured 29.41 vs model ~29 fresh, i.e. that cell was NOT
heavily soaked; 100k measured 15.63 vs model ~15.7 - also close).
