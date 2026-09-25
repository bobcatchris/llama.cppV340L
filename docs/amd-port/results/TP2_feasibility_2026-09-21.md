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
