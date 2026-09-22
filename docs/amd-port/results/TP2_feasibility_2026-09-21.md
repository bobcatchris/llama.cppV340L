# TP2 feasibility desk - MTP at TP2 and the --spec-mtp-device flag (V340L campaign)

Date: 2026-09-21. Desk: wt-tp2-mtp, branch amd/tp2-mtp-feasibility (base 597fcaee5,
the merged campaign tree that carries --spec-mtp-device). Binary: worktree-local
build-hip (gfx900, ROCm 6.2.0, GNU 13.3.0), version 138 (597fcaee5).
GPU window: dies 0-2 granted by Gemini (hub msg #1328 ACK + GO), 23:06-19:29 local
slot; die 3 untouched by this desk (one handoff record, below). Harness:
docs/amd-port/tests/run_tp2_feasibility.sh + vram_sampler.py (2 s per-die CSV with
#PHASE markers), guard_battery.py with tests/baseline_tp2_200k.json (anchors, not
gates; the 0.63 acceptance canary is the only hard gate).

## The question, and how it moved

Original task: verify the analytic prediction that TP2 + MTP at 200k now fits
because --spec-mtp-device moves the draft head (169.3 MiB weights + ~117.5 MiB/die
draft KV at q4_0 = ~202 MiB/die) off the two serving dies, against the historical
~151 MiB/die shortfall (PLOG-098/099 era) - predicted margin ~+48 MiB/die.

Coordinator corrections during the run: (1) the 202 MiB number is a weights+KV-only
LOWER BOUND; activated MTP typically costs over 2 GB/device (draft-context
allocations dominate) - measure the true OFF-vs-ON delta; (2) after the 200k
negative landed, re-scope to the working-context measurement; (3) hub de-conflict
(msgs #1336/#1337): the TP3 total A/B belongs to Gemini's lane; this desk stays on
TP2 (rep-2s + receipt).

## Result 1 - TP2 at 200k does NOT boot, MTP OFF or ON (named negative)

Arm t2off (TP2, dies 0,1, -c 200000, b512/ub512, q4_0 KV, FA, MTP OFF) aborts at
BOOT, during target-context compute-buffer reserve - before any draft context
exists:

    0.03.357.225 D graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  4, n_outputs =    4
    0.03.363.358 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1057.78 MiB on device 0: cudaMalloc failed: out of memory
    ggml/src/ggml-backend-meta.cpp:1512: GGML_ASSERT(bufs.back() != nullptr) failed

Per-die rocm-smi during the attempt (2 s sampler, tp2feas_off1_*_vram.log):

    t+0.1s  card0 8.0   card1 8.0    card2 8.0   MiB used
    t+2.1s  card0 5876.6 card1 5876.2 card2 8.0   MiB used
    t+4.2s  card0 7956.6 card1 7956.3 card2 8.0   MiB used  (peak, at/just before the abort)

Device 0 had ~219 MiB free when the 1057.78 MiB compute chunk was attempted:
short by >= 839 MiB on that chunk alone, MTP OFF, before any draft allocation.
KV at 200k is 3519.00 MiB total (200192 cells, 16 full-attn layers, K/V q4_0
1759.50 + 1759.50, in-split across the two dies).

Verdict: the 202-vs-151 fit prediction is REFUTED for 200k. The serving dies are
~0.9-1+ GiB short at 200k before the draft head exists; the historical 151 MiB/die
figure is a 10k-ctx, layer-split-era number (PLOG-098) that does not describe the
tensor-split 200k regime. The ub512 compute buffer (1057.78 MiB, confirmed by the
TP3 control shutdown breakdown in tp2feas_t3on1_*) is ctx-independent, so no
ubatch reduction closes a gap of this size (ub128 shrinks it ~4x, still ~0.5 GiB
over the wall together with the 1.76 GiB/die KV + ~6.1 GiB/die weights+mirrors).
TP2@200k stays closed as a topology (matches shared-ledger E-028; this receipt
supplies the exact failure numbers E-028 lacked: "empty logs" was an assumption -
the boot died at the 1057.78 MiB alloc with ~219 MiB free on die 0).

## Result 2 - true per-device MTP cost and flag relocation at a working context

Three-way matrix at -c 10000 (TP2, b512/ub512, q4_0 KV, FA; two reps per arm;
all four dies 8176 MiB total). "Serving" = dies 0/1; draft die = 2 in the flag arm.

| arm | boot-ready used/free c0, c1 (MiB) | post-probe used/free c0, c1 (MiB) | pp 2k (t/s) | decode 10k (t/s) | accept | mean acc len |
|-----|-----------------------------------|-----------------------------------|-------------|------------------|--------|--------------|
| MTP-OFF r1         | 6526.8/1649.2, 6526.4/1649.6 | 7140.6/1035.4, 7138.1/1037.9 | 89.17 | 10.54 | - | - |
| MTP-OFF r2         | 6526.9/1649.1, 6526.4/1649.6 | 7144.9/1031.1, 7146.4/1029.6 | 89.16 | 11.39 | - | - |
| MTP-in-split r1    | 7599.6/576.4, 7599.1/576.9   | 8108.9/67.1, 8108.5/67.5     | 87.47 | 14.79 | 0.66667 (84/126) | 3.00 |
| MTP-in-split r2    | 7599.6/576.4, 7599.1/576.9   | 8108.8/67.2, 8108.5/67.5     | 87.79 | 14.81 | 0.66667 (84/126) | 3.00 |
| MTP-on-die2 r1     | 7348.5/827.5, 7347.9/828.1   | 7945.3/230.7, 7944.7/231.3   | 87.61 | 14.43 | 0.66667 (84/126) | 3.00 |
| MTP-on-die2 r2     | 7348.5/827.5, 7347.9/828.1   | 7945.2/230.8, 7944.6/231.4   | 87.68 | 14.43 | 0.66667 (84/126) | 3.00 |

Draft die (card2) in the flag arm: 558.0 MiB used at boot-ready, 1437.1 MiB after
probes (r1). Boot-ready VRAM is byte-identical across reps in every arm
(deterministic allocation); probe-time numbers reproduce within ~1 MiB except
MTP-OFF decode (10.54 vs 11.39, thermal band on back-to-back boots).

Derived (boot-ready deltas, rep-reproducible):

- TRUE per-device MTP cost, in-split, at 10k: 7599.6 - 6526.8 = 1072.8 MiB/die
  over MTP-OFF. Weights+KV of the draft head are only ~90 MiB of that (169.3/2
  weights + ~5.5 draft KV); the other ~983 MiB/die is draft-CONTEXT allocation
  (compute + output buffers on the TP meta group). The >2 GB/device hypothesis
  does NOT reproduce at TP2/10k - the measured cost is ~1.05 GiB/die - but the
  coordinator's instinct is confirmed: the cost is context-buffer dominated, not
  weights/KV dominated (the 202 MiB analytic prediction was a ~5x undercount).
- --spec-mtp-device relocates only 251.1 MiB/die off the serving dies
  (7599.6 -> 7348.5); with the flag ON the serving dies still pay 821.7 MiB/die
  over MTP-OFF. The draft context's backend list is [meta, die2, CPU], and the
  scheduler keeps the bulk of the draft compute buffers on the meta group; the
  dedicated die holds 558.0 MiB at boot (169.3 weights + ~11 KV + its own
  compute/output), growing to 1437.1 MiB after probes. FOLLOW-UP if full
  isolation is ever wanted: the draft context's buffer placement must prefer the
  extra device in the sched - a code change beyond the flag.
- Decode: MTP lifts TP2 decode +38-40% over OFF (14.79/14.81 vs 10.54/11.39).
  The flag costs -2.4% decode vs in-split (14.43 vs 14.79/14.81 mean); pp is
  inside noise (87.5-89.2 across all arms). Acceptance is identical everywhere
  (0.66667, 3.00 tok/step - the 2k-probe acceptance is ctx-blessed).
- Headroom consequence: in-split MTP leaves 67 MiB/die after probes at 10k - the
  TP2+MTP in-split ctx ceiling is ~17-18k (target KV grows 8.8 MiB/die per 1k
  ctx). The flag leaves 231 MiB/die with draft KV growth moved to die 2
  (~1.1 MiB per 1k ctx), extending the ceiling to roughly 2x (~36k) - at the
  -2.4% decode price. Both die long before 200k (Result 1).

## Result 3 - TP3 control VRAM record (handoff to Gemini's TP3 A/B)

Before the hub de-conflict, this desk booted the TP3 CONTROL once (in-split MTP,
200k, b512/ub512 - config of record) for VRAM; the full A/B is Gemini's lane.
Boot-ready per-die used/free MiB: c0 7809.8/366.2, c1 7582.0/594.0,
c2 7634.2/541.8, c3 18.0/8158.0 (idle). Note: tp2feas_t3on1_*_server.log is a
28-line remnant (a duplicate-launch bug in the runner, since fixed, clobbered the
log); treat the vram.log + the numbers above as the record. The remnant does
confirm the ub512 target compute buffer = 1057.78 MiB + draft-context compute
263.5 MiB + ~215 MiB host buffers on the control config.

## Verdict

1. Does TP2+MTP@200k boot now? NO. It refuses at boot even with MTP OFF; the
   draft-block offload is irrelevant at that context. Closed (E-028).
2. Measured VRAM: MTP in-split costs 1072.8 MiB/die at TP2/10k (context-buffer
   dominated; weights+KV only ~90 MiB of it); the flag moves 251.1 MiB/die to the
   draft die, leaving 821.7 MiB/die of MTP cost on the serving dies.
3. Is a TP2+draft-die config worth promoting? For 2-die mid-context serving where
   MTP would otherwise not fit (TP2+MTP in-split ceiling ~17-18k ctx at 67 MiB/die
   free): yes - the flag extends the MTP-capable ceiling to ~36k ctx for -2.4%
   decode. Below that ceiling, in-split is slightly faster and simpler. The flag
   is opt-in either way; it changes nothing unless passed.

## Artifacts (docs/amd-port/results/ unless noted)

- tp2feas_off1_20260921_181525_{server,vram}.log          200k negative (no battery: abort at boot)
- tp2feas_t2off10k1_20260921_183124_* / t2off10k2_20260921_184745_*      MTP-OFF reps
- tp2feas_t2noflag10k1_20260921_182830_* / t2noflag10k2_20260921_184452_*  in-split reps
- tp2feas_t2flag10k1_20260921_182107_* / t2flag10k2_20260921_1859*_*     flag reps
- tp2feas_t3on1_20260921_183902_{server,vram}.log         TP3 control handoff record
- tests/run_tp2_feasibility.sh, tests/vram_sampler.py, tests/baseline_tp2_200k.json (worktree)

## Coordination log (hub)

#1326 window request -> #1328 ACK+GO (dies 0-2, 75 min) -> #1332 window start ->
#1333 progress -> #1335 die-3 request -> #1336/#1337 de-conflict (TP3 A/B is
Gemini's; this desk stays on TP2) -> #1338 stand-down + TP3 control VRAM handoff.

## Result 4 - E-035 full-isolation build verified on TP2 at 10k (added 2026-09-22)

Worktree synced to the campaign branch (merge amd/v340-port-v2 at 7d3ab351a,
carrying the draft-isolation merge 8a4ebbcaa / ledger E-035); rebuilt served
binary version 152 (7d3ab351a). Same arm as Result 2's flag row (TP2, -c 10000,
b512/ub512, q4_0 KV, FA, --spec-mtp-device ROCm2), LLAMA_SPEC_MTP_STRICT=1.

Gate and mechanics (tp2feas_t2flag10k3_20260921_212512_server.log):

    load_tensors: MTP device ROCm2: duplicated token_embd.weight (335.3 MiB) and output.weight (517.8 MiB) for the draft context
    sched_reserve: ROCm2 isolation audit:   132.02 MiB on ROCm2,     0.00 MiB on the model split
    load_model: MTP draft context runs on ROCm2 (fully isolated: nextn weights + KV + duplicated embeddings/LM head)

The required audit reads 0.00 MiB on the model split - PASS (logged in two
independent boots; the draft context's own breakdown shows ROCm2 self = 1194 MiB
= model 1022 (nextn 169.3 + duplication 853.12) + context 40 + compute 132).

VRAM comparison, v1 flag build (Result 2) vs isolation build:

| state | serving dies 0/1 used (free) | draft die 2 used (free) |
|-------|------------------------------|-------------------------|
| v1 boot-ready      | 7348.5/7347.9 (827.5/828.1) | 558.0 (7618.0) |
| iso boot-ready     | 7344.3/7343.9 (831.7/832.1) | 1412.5 (6763.5) |
| v1 post-probe      | 7945.3/7944.7 (230.7/231.3) | 1437.1 (6738.9) |
| iso post-probe     | 7930.2/7929.8 (245.8/246.2) | 2290.8 (5885.2) |

(boot-ready from runner snapshots; iso post-probe from the 2 s sampler's ~68 s
stable decode-phase envelope after the prefill guard passed at 88.74 t/s.)

Findings:

- The duplication is exact: draft die boot delta = 1412.5 - 558.0 = +854.5 MiB
  vs the 853.12 MiB token_embd+output duplication. Prediction confirmed.
- The draft die carries 2290.8 MiB post-probe (matches the ~2.3 GiB prediction):
  duplication + the draft context's request-time compute (+878.3 MiB) relocated.
- The serving-die prediction (drop ~800 MiB further to ~7036 used) is REFUTED by
  measurement: serving dies dropped only ~15 MiB (7945.3 -> 7930.2 post-probe).
  Their request-time footprint is TARGET-context compute dominated - identical
  in both builds; the v1 flag had already moved the draft's boot-time
  allocations (v1-vs-iso boot-ready delta is only 4.2 MiB/die). Full isolation
  buys a hard guarantee (audit-gated 0.00 residual, STRICT env) and removes the
  draft's request-time presence from the serving dies' accounting, but the
  serving-die envelope stays target-bound.
- Prefill guard PASSED at 88.74 t/s (v1 87.61/87.68 - within noise); a second
  clean run under the new boot lock passed prefill at 88.91 t/s. The decode/
  acceptance cell FAILS on the isolation build at TP2/10k - a real E-035
  defect, not a collision: the 7857-token decode request dies with 28
  "failed to find free space in the KV cache" retries down to n_batch=1 and
  "E srv decode: Context size has been exceeded. off = 69" -> HTTP 500 (v1
  build: ZERO retries on the identical request; KV geometry identical between
  builds: n_ctx/n_ctx_seq 10240, kv_unified, Meta KV 90 MiB, draft KV 40 MiB
  on ROCm2). Hypothesis for the draft-isolation desk: the draft cache's cell
  accounting aliases the target's unified-cache cells, marking ~10k cells
  occupied while the target sequence has processed only ~3.6k. The decode/
  acceptance comparison therefore stands on the v1 numbers (14.43 t/s, accept
  0.66667) and the iso decode path is blocked at TP2/10k pending an E-035
  follow-up fix.

Coordination: first attempt (21:10) collided with the TP3 A/B control boot; the
gap-claimed rerun (21:25) was killed at t+120 s by the next A/B boot's
free_port; the third run (21:34) executed under the new
/tmp/campaign_gpu_boot.lock convention (check-and-wait + hold + release on
teardown) and completed cleanly into the defect above. Both collisions left
the dies clean; no data loss beyond the decode cell.

## Result 5 - DEFINITIVE three-way savings table on the isolation build (2026-09-22)

Binary v152 (7d3ab351a, E-035/E-036 merged tree). TP2, -c 10000, b512/ub512,
q4_0 KV, FA, 2 reps per arm, lock-honored boots. Serving dies = 0/1 (8176 MiB
each); draft die = 2 in arm C.

| arm | boot-ready used/free c0, c1 | post-probe used/free c0, c1 | pp 2k | decode ~7.9k | accept / mean |
|-----|-----------------------------|-----------------------------|-------|--------------|---------------|
| A MTP-OFF    r5,r6 | 6526.8/1649.2, 6526.4/1649.6 (both reps) | 7163.7/1012.3, 7165.3/1010.7; 7144.8/1031.2, 7144.4/1031.6 | 90.36 / 89.21 | 12.83 / 11.94 | - |
| B in-split   r7,r8 | 7599.6/576.4, 7599.1/576.9 (both reps)   | 8112.4/63.6, 8112.1/63.9; 8108.9/67.1, 8108.6/67.4         | 84.64 / 81.67 | 12.80 / 13.59 | 0.66667 / 3.00 (both) |
| C draft-die  r3,r4 | 7344.3/831.7, 7343.9/832.1 (both reps)   | 7930.2/245.8, 7929.8/246.2; 7936.5/239.5, 7936.1/239.9     | 88.74 / 88.91 | VOID (defect, Result 4) | VOID |

Draft die residency (arm C): 1412.5 MiB at boot-ready -> 2290.8 / 2291.2 MiB
post-probe (5885.2 free). Audit gate: 0.00 MiB on the model split, all C boots.

SAVED per serving die (positive = freed; used-MiB deltas, rep-stable values):

| comparison | boot-ready | post-probe |
|------------|-----------|------------|
| B-vs-A (true MTP cost, in-split) | -1072.8 (MTP COSTS 1072.8) | -955.9 (costs ~956) |
| C-vs-A (MTP + isolation cost)    | -817.5 (costs 817.5)       | -778.6 (costs ~779) |
| C-vs-B (isolation SAVING)        | +255.3                     | +177.3 |

Reading (unchanged from Result 4, now 2-rep solid): boot-ready MTP cost is
~1073 MiB/die in-split and ~818 with the flag - the flag/defect-free isolation
saves 255 MiB/die statically, 177 MiB/die at request time; the serving-die
envelope is target-compute dominated in every arm. Performance: decode B 12.80/
13.59 vs A 12.83/11.94 - the MTP decode lift collapsed in this hot session
(dies at 84-90 C edge through the run; v1 cool-session delta was +38%): cross-
session decode comparisons are thermal-confounded, within-session ordering is
A ~ B < C-unmeasured. pp: A 89.8 mean > C 88.8 mean > B 83.2 mean (in-split pp
pays ~7% vs A; C recovers to ~A-1%). Acceptance 0.66667/3.00 everywhere it is
measurable. Arm C decode/accept remains VOID pending the E-035 fix; the C
decode/accept reference stays the v1 flag numbers (14.43/14.43, 0.66667).

Artifacts: tp2feas_t2off10k{5,6}_20260922_*, tp2feas_t2noflag10k{7,8}_20260922_*,
tp2feas_t2flag10k{3,4}_20260921_* (battery jsonl committed; raw logs local).
Incidents: four boots voided by a die-capacity collision with the TP3-threeway
lane (05:52-05:55); lock-broke + die-idle guard + refined stale-lock rule
adopted (hub #1354/#1355).

## Result 6 - FOUR-ARM definitive table, C decode filled (2026-09-22)

Arm C decode re-run on the isolation build (t2flag10k r5/r6, clean conditions,
lock held, STRICT=1): decode 7.42 / 7.41 t/s - REPRODUCED, and the C5 server log
has ZERO "failed to find free space" retries, so the slowdown is the real
isolation decode path, not an allocation stall (rep4's intermittent
"Context size has been exceeded" HTTP 500 is a separate occasional failure mode;
1 of 3 completed attempts). Acceptance unaffected: 0.66667 / 3.00 everywhere.

| arm | build | boot-ready used (free) c0,c1 | post-probe used (free) c0,c1 | pp 2k | decode ~7.9k | accept / mean |
|-----|-------|------------------------------|------------------------------|-------|--------------|---------------|
| A MTP-OFF     | v152 | 6526.8 (1649.2), 6526.4 (1649.6) | 7144.4-7165.3 (1010.7-1031.6) | 90.36 / 89.21 | 12.83 / 11.94 | - |
| B in-split    | v152 | 7599.6 (576.4), 7599.1 (576.9)   | 8108.6-8112.4 (63.6-67.4)     | 84.64 / 81.67 | 12.80 / 13.59 | 0.66667 / 3.00 |
| C v1-flag     | v138 | 7348.5 (827.5), 7347.9 (828.1)   | 7944.7-7945.3 (230.7/231.3)   | 87.61 / 87.68 | 14.43 / 14.43 | 0.66667 / 3.00 |
| D full-isolation | v152 | 7344.3 (831.7), 7343.9 (832.1) | 7940.7-7941.0 (235.0/235.4)   | 88.74 / 88.91 / 88.50 | 7.42 / 7.41 | 0.66667 / 3.00 |

Draft-die residency: C = 558.0 boot -> 1437.1 post; D = 1412.5 boot (853.12
duplication, exact) -> 2291.4/2291.5 post. Audit gate: 0.00 on the model split
(D only).

SAVED per serving die (post-probe used-MiB; positive = freed):

| comparison | saving |
|------------|--------|
| B-vs-A  (MTP cost, in-split)        | -961 (MTP costs ~961 MiB/die) |
| C-vs-A  (v1-flag MTP cost)          | -790 |
| D-vs-A  (isolated MTP cost)         | -786 |
| C-vs-B  (v1-flag saving vs in-split)| +164 |
| D-vs-B  (isolation saving vs in-split) | +169 |
| D-vs-C  (full isolation vs v1-flag) | +4.5 (nothing) |

PERFORMANCE COST of full isolation: decode -48.5% vs v1-flag (7.42/7.41 vs
14.43/14.43), also -42% vs in-split. Acceptance identical everywhere. pp: D
88.7 mean recovers to ~A-1% (the duplication does not hurt prefill).

MECHANISM (honest reading): the audit's 0.00 MiB meta-side residual is achieved
exactly by running the draft's embedding-row + LM-head matmul on the die-2
copies at 1x bandwidth (plus 2 host-staged hops per draft step) - the same
relocation that frees the meta group doubles the per-draft-step cost. The 0.00
gate and the decode collapse are two faces of the same design choice.

VERDICT (definitive, TP2@10k): v1-flag behavior is the sweet spot - within
~4.5 MiB/die of full isolation's serving-die footprint at 2.0x its decode.
Full isolation (E-035) is not worth promoting for latency-sensitive TP2
serving; it is only rational when serving-die VRAM, not throughput, is the
binding constraint (its B-vs-D saving is real but small: ~169 MiB/die over
in-split, and in-split itself only leaves 63-67 MiB/die free at 10k).

Artifacts: tp2feas_t2flag10k{5,6}_20260922_* (battery jsonl committed).

## Result 7 - TP3 three-way at 10k on the isolation build (2026-09-22)

Same protocol as Result 5, TP3 (dies 0/1/2; draft die 3 in arm C), -c 10000,
b512/ub512, q4_0 KV, FA, 2 reps, STRICT=1, port 8083.

| arm | boot-ready used c0,c1,c2 (free) | post-probe used (free) | pp 2k | decode ~7.9k | accept / mean |
|-----|---------------------------------|------------------------|-------|--------------|---------------|
| A MTP-OFF  r1,r2 | 4535.3/4501.3/4502.9 (3641-3675) | 5191.6/5105.7/5101.3 (2984-3075) | 122.58 / 120.74 | 12.17 / 12.18 | - |
| B in-split r1,r2 | 5306.1/5264.1/5263.9 (2870-2912) | 6207.2/6117.4/6121.0 (1955-2059) | 117.39 / 118.48 | 15.38 / 15.58 | 0.66667 / 3.00 |
| C draft-die r1,r2 | 5068.9/5046.8/5046.5 (3107-3129) + die3 1417.3 | 5673.4/5603.6/5607.3 (2503-2572) + die3 2297.2 (5878.8) | 117.64 / 117.88 | 7.57 / 7.58 | 0.66667 / 3.00 |

Audit gate: 0.00 MiB on the model split expected in C; captured at boot in the
TP2 runs - the TP3 10k boots ran without --verbose where the INFO-level audit
line did not surface; VERBOSE=1 confirmation boots follow with the 200k arms
(the physical signature is unambiguous: die 3 holds 1417.3 MiB at boot-ready =
853.12 duplication + nextn 169.3 + KV + compute, and 2297.2 MiB post-probe).

SAVED per serving die (boot-ready): in-split costs ~765 over OFF; the flag
costs ~541 over OFF; the flag SAVES ~224/die vs in-split. Post-probe: in-split
costs ~1015 over OFF; the flag SAVES ~503/die vs in-split.

THE HEADLINE DEFECT CONFIRMED ON TP3: full-isolation decode at 10k is
7.57/7.58 t/s = -51% vs in-split (15.38/15.58) and -38% vs MTP-OFF
(12.17/12.18), with acceptance identical (0.66667/3.00) - the same collapse
measured on TP2 (7.42/7.41). The draft cycle under full isolation roughly
doubles its per-token cost: the duplicated embedding/LM-head run on the draft
die at 1x bandwidth and two host-staged hops per draft step replace the
3-die-bandwidth meta-side path. Prefill is unaffected (117.6-117.9 vs
117.4-118.5 in-split).

## TP3 200k arms (B in-split / C draft-die, 2 reps) - appended below when complete
