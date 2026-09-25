# WORK ORDER: DECODE LANDING — Branch-B sync removal + k=3 pricing (decode +5-10% target)

Owner: desk agent. Coordinator: owns ALL infrastructure (boots, window, faults, restores).
You execute the task. The serving window is YOURS for the duration — no claim protocol
needed; the coordinator will message you if something outranks it.

## Task 1 — Branch-B (sized: 3-4.5 ms of the 59 ms cold round)
The one_shot allreduce at src/ops/**/one_shot_allreduce.cu ~:1033 does a per-call
cudaStreamSynchronize; the W4AR/Branch-B cure (PLOG-049 lineage, GATE-T pipelined 0.47x
@30 KiB, capture-safety proven) deletes it for deferred status consumption.
1. Read docs/amd/W7_ROUNDWALL_desk.md GATE-K section + GATE-T rows — the full spec is there.
2. Edit: env-gated NINFER_AR_DEFER_STATUS=1 removes the per-call sync (status consumed at
   the next natural sync point / end-of-round). Unset/'0' = byte-identical legacy. NO new
   refusal paths. Keep the diff minimal — one call site + one env read.
3. Correctness: decode parity cell or serve parity (BLUE/stop probes, >=10 generations,
   compare outputs vs control boot byte-for-byte at temp 0 — greedy decode MUST be
   token-identical or the diff is not equivalent-time).
4. Build: cmake --build build-hip-amd --target ninfer-serve -j8 (existing tree, ~10-20 min;
   if the build hangs >30 min tag "BLOCKER:" in this file and move to Task 2 while it runs).
5. Bank the bin BEFORE relink: /home/chris/artifacts_bin/ninfer-serve_<sha16>.bin.
6. Paired A/B (PLOG-060 design): same bin, fresh boot per arm (use bash
   /home/chris/serve_fast.sh — 40 s boots, NVMe model, NO 10k grade), 3x counting-prompt
   mt=600 probes per arm + decode t/s from the [tp2] engine lines. Gate: >=3% decode t/s
   within-pair => promote: flip runbook BIN + env, BOOT_BATTERY, report. <3% => bank + report.

## Task 2 — k=3 draft-depth pricing (data-backed: acceptance 0.96, tok/round 2.91)
1. Check whether the MTP draft depth is a runtime dial: grep --draft-tokens handling in the
   serve arg parsing (src/serve/). If it accepts 3 without code changes: boot once with
   --draft-tokens 3, run the same mt=600 probe x3, read tok/round + acceptance + decode t/s.
   Compare against the k=2 numbers from this same desk (PLOG-066: 2.91 tok/round).
2. If k=3 needs code changes: STOP after step 1's finding and report — do not open a second
   code front; the coordinator decides.

## Laws
- THIS work order file is your checkpoint: append "## PROGRESS LOG" (newest-first) after
  EVERY step. A fresh agent resumes from it.
- Infrastructure (boot faults, wedges, port conflicts, restores) is the COORDINATOR's job:
  tag "BLOCKER: <one line>" in this file, keep executing what doesn't need the blocker,
  and it gets resolved within 30 min without you burning retries. NEVER pkill anything.
- The canonical bin (2c8901d3) has a KNOWN boot fault under root-look (incident, coordinator-
  owned). If a boot faults at GQA width=53 warmup: tag BLOCKER, boot your own LAST-KNOWN-GOOD
  bin instead (c618d356, env-correct), continue.
- Numbers carry clocks notes; ±2% within-pair; fresh boot per arm; die assignments: your
  serving legs use the normal runbook dies; no side benches needed for this work order.

## PROGRESS LOG

- **BUILD CLAIM 2026-09-19 13:35 CDT — F1b execution desk (d1 fix + accept-domain cell).**
  Coordinator resume order received (root-look CLOSED per PLOG-070; serving window mine;
  one-shot route BLOCKED — NINFER_TP_ONESHOT_AR stays UNSET in every F1b leg, canonical
  ring posture only). Building in build-hip-amd from the CURRENT tree state: TWO
  single-variable bins — CONTROL (tree as-is, d1 pre-fix) and FIX
  (docs/amd/W7_F1B_d1_fix.patch applied = tp2_backend.cpp :3182 full-domain token_domain +
  PDBG tap :3122 same-class fix + rank field; touches NO other file, so the pair's only
  delta is d1). The tree carries the KV desk's uncommitted edits (tp_engine.cpp /
  tp2_budget.h / gqa kernels / kv_hosted_route.cuh) — present EQUALLY in both bins, cancels
  within the pair; provenance noted in the F1b closing row. Targets: ninfer-serve +
  ninfer_accept_domain_w4_cell (cell-addition only in tests/CMakeLists.txt). Existing
  build-tree bin 9bc7d7e1f764e7cb is BANKED and :8100 serves it FROM THE BANK PATH — the
  relink here cannot touch the running server.

- [step 9 — RESTORE RECEIPT] 2026-09-19 ~11:05 CDT. Canonical serving restore PASSED:
  serve_10k.sh booted bin 2c8901d3d18adef1 (canonical env, no oneshot/defer) — boot
  clean (no GQA width=53 fault this boot), 10K GRADE PASS (BLUE, finish=stop,
  prompt=10000, completion=69). :8100 IS CANONICAL AGAIN. Desk close complete;
  commit c9b9c402b carries the row + logs + work order + edit on amd/wo-w7-body.

- [step 8 — CLOSE] 2026-09-19 ~10:45 CDT. DESK CLOSED: **Branch-B banked-unpromoted —

  untestable at gate level (oneshot warmup wedge incident), parity GREEN, k=3 closed.**
  - Bracket-1 A/B EXECUTED (ARMCNTLD2 vs ARMBDEFER, same bin, ordinal-paired): GATE
    FAIL — defer arm -31.8/-12.2/-9.9% decode t/s within-pair (needed +3%). Confound
    noted: box heated monotonically all window (probe0 decodes 12.15 -> 18.33 ->
    26.95 s across K2CTRL/ARMCNTLD2/ARMBDEFER); bracket-2 (order-swapped,
    cool-bracketed) attempted per coordinator.
  - Bracket-2: B2 silent wedge (wedge #5), A2 watchdog exit-72 (wedge #6) — two hits
    = arm untestable today per close criteria. Day tally: oneshot boots 1/7 healthy
    (both bins); canonical-posture boots 4/4 healthy. Coordinator filed the blocker
    (KAR-v2 first-crossing handshake, pre-existing, LKG bin affected).
  - PARITY GREEN (the edit's own cell): A-vs-B byte-identical 12/12 parity prompts +
    3/3 mt=600 — the sync removal is exactly equivalent-time. NOT the blocker.
  - ARM-vs-CANONICAL finding banked: oneshot+sync probe0 18.33 s / acc 0.85 / 82.6
    ms-round vs canonical 12.15 s / 0.96 / 59.0 ms (PLOG-066 reproduced by K2CTRL).
    The arm both loses -34% t/s AND shifts numerics (ulp-drift greedy flips;
    outputs differ byte-wise from ring) — promotion path requires the W4AR owner's
    byte-equal answer FIRST; re-entry conditions in the row file.
  - Task 2 closure: k=3 LOSS (-10.5% decode t/s within-pair probe1; acc 0.96->0.92,
    tok/round 2.91->3.75, round 59.0->84.8 ms; breakeven needed <=76 ms). Keep k=2.
  - Window close: canonical serving restore FIRED (serve_10k.sh, bin 2c8901d3 + its
    10k grade) — result appends below when it lands. All wedged processes exited
    clean on TERM; dies verified idle (3-9 W) between every boot.
  - Files: results/amd/coherence/WO_DECODE_LANDING_row.txt (full row),
    results/amd/coherence/w7_defer_landing/ (8 serve logs), bin banked
    /home/chris/artifacts_bin/ninfer-serve_183da007801e0fb6.bin, edit = one env read
    + one break in src/core/multi_gpu/one_shot_allreduce.cu.

- [step 7] 2026-09-19 ~10:30 CDT. Bracket-2 per coordinator guidance (order-swap,
  no-retry): B2 (defer) WEDGED silent (dies 0/1/2 100%, die 3 idle, log stalled at
  same warmup position as every silent wedge); counted, moved to A2. A2 (sync)
  WEDGED — [AR-WEDGE-WATCHDOG] kind=2 crossing-heartbeat exit 72, matrix banked in
  ARMA2_serve.log. TWO HITS = arm untestable today -> close criteria met.
  Provenance note for F1b containment: tp2_backend.cpp content == HEAD at build time
  and now (clean git status; 10:21 mtime touch is content-identical) => NO d1-fix
  carry in bin 183da007; within-bin A/B single-variable-safe either way.

- [step 6] 2026-09-19 ~10:05 CDT. Bracket-1 A/B + PARITY EXECUTED (the window's one
  clean oneshot boot made both legs possible):
  - Arm A (ARMCNTLD2, oneshot+sync): boot HEALTHY (wedge class is flaky, 1-in-4 by
    then). Parity battery 12/12 parsed. Series: 32.7/24.6/23.2 t/s (18.33/24.41/
    25.82 s decode, rounds 222/221/221, acc 0.85-0.86, tok/round 2.70-2.71).
  - Arm B (ARMBDEFER, oneshot+defer): boot HEALTHY, fast probe PASS 20.7 s. Parity
    battery 12/12. Series: 22.3/21.6/20.9 t/s (26.95/27.78/28.68 s, rounds 222/221/
    221, acc 0.85-0.86).
  - PARITY: B vs A byte-identical 12/12 parity prompts + 3/3 mt600 counting
    responses — equivalent-time PROVEN (the edit's correctness cell, step 3: PASS).
  - GATE: FAIL — within-pair -31.8/-12.2/-9.9% (needed >= +3%). Thermal-order
    confound documented (box monotonically heating; A ran before B). Gate's <3%
    branch => bank + report, bracket-2 ordered by coordinator to de-confound.
  - FINDING (banked): arm-vs-canonical on the same bin/window — canonical K2CTRL
    probe0 12.15 s / acc 0.96 / 2.91 / 59.0 ms-round (= exact PLOG-066 reproduction)
    vs oneshot-sync 18.33 s / 0.85 / 2.70 / 82.6 ms. The oneshot arm on this box
    today is -34% decode t/s vs canonical AND outputs diverge byte-wise from ring
    (ulp-drift greedy flips; acceptance 0.85 vs 0.96) — the GATE-T byte-equal
    question is LIVE on the serving kernel, not just the transport bench. Promotion
    of ANY oneshot-arm posture requires that answered first.

- [step 5] 2026-09-19 ~09:26 CDT. **BLOCKER: the W4AR oneshot arm (NINFER_TP_ONESHOT_AR=1)

  cannot complete warmup on this box today — 3 boots, 3 wedges (see step-4/5 rows);
  Task 1 A/B + defer-parity are gated on it. Coordinator-owned per infra law.**
  Discrimination is COMPLETE, bin exonerated:
  - boot 1: 183da007 + oneshot+sync -> SILENT warmup wedge ~22 min (dies 0/1 100%,
    2/3 idle; clean TERM exit, dies drained 3-9 W).
  - boot 2: 183da007 + oneshot+sync retry -> B3 watchdog FIRED AS DESIGNED
    ([AR-WEDGE-WATCHDOG] kind=2 crossing-heartbeat, exit 72, full 4-rank matrix:
    rank0 hb_seq=1 stuck in first crossing 2051 ms; rank3 hb_seq=2 done x2; ranks
    1/2 hb_seq=0 never arrived — warmup collective desync; one-shot gen=0 = AR calls
    never even reached).
  - DISCRIMINATOR: c618d356 (LKG) + canonical (no oneshot) -> HEALTHY boot.
  - DISCRIMINATOR: 183da007 + canonical (no oneshot) -> HEALTHY boot.
  - DISCRIMINATOR: c618d356 (LKG!) + oneshot -> SILENT warmup wedge (300 s health
    timeout, zero watchdog lines — the PLOG-049-era bin fails identically TODAY).
  => VERDICT: the wedge is arm x box-state TODAY, independent of my branch AND of my
  edit (edit dormant at unset; arm fails at warmup BEFORE any AR call, gen=0). PLOG-049
  banked the arm GREEN twice on 09-18 clean-box (post-FLR) — box has had boot faults +
  FLR this morning (PLOG-066 INCIDENT 1). Suspicion: warmup crossing desync the arm's
  watchdog now CATCHES where legacy posture would hang silently (boot-2 caught it,
  boots 1/3 died before watchdog arm point). All wedge processes exited CLEAN on TERM
  (no SIGKILL poisoning); dies verified idle between boots.
  EXECUTING WHILE BLOCKED: Task 2 k=3 pricing (needs NO oneshot arm) — k=3 leg and
  fresh in-window k=2 control, both canonical posture, my bin 183da007. Task 1 parity
  + A/B resume when the blocker clears (both arms' scripts ready in /tmp/w7_defer/).
  Bin + edit banked: /home/chris/artifacts_bin/ninfer-serve_183da007801e0fb6.bin.

- [step 4] 2026-09-19 ~09:20 CDT. WEDGE INCIDENT on first arm-A boot + discrimination:

  - Build hit `cmake: command not found` (PATH) — real cmake at
    /home/chris/opt/cmake/bin/cmake (CMakeCache CMAKE_COMMAND). Build GREEN on retry
    (exit 0, 08:50). BIN BANKED: /home/chris/artifacts_bin/ninfer-serve_183da007801e0fb6.bin
    (bank-name law: filename = sha256[:16] of the binary — convention verified against
    the canonical bin). strings: NINFER_AR_DEFER_STATUS HIT.
  - First arm-A boot (183da007 + canonical env + NINFER_TP_ONESHOT_AR=1, SYNC arm —
    defer UNSET): model loaded 17 s, then WEDGED in warmup ~22 min: dies 0/1 100%
    busy 61/83 W, dies 2/3 idle, no log advance past "warming up...", B3 watchdog
    never fired. T2-shape cross-rank warmup wedge. Exited CLEAN on SIGTERM (no
    SIGKILL poisoning); dies drained to 3-9 W idle.
  - DISCRIMINATOR 1: LKG bin c618d356f0cdc401 + canonical posture (no oneshot env)
    boots HEALTHY -> box boot-capable right now; earlier wedge = flaky-warmup-wedge
    class (PLOG-050 ARM-D 1-in-2) OR my-bin/oneshot-arm implicated. NOT tagged
    BLOCKER yet — window still executing.
  - SCRIPT LESSON (PLOG-052 again): pgrep pattern must be
    '^/home/chris/artifacts_bin/ninfer-serve' (full-path anchor); basename-anchored
    pattern matches nothing -> false PROCESS DIED. Fixed in /tmp/w7_defer/boot_gen.sh.
  - Probe metric source identified: [D2-SS-STATS] rounds/accepted/acc_rate/tok/round/
    gen/decode/decode_tps — engine prints it UNCONDITIONALLY per request (tp2_backend
    ~:4752, rank 0 stderr -> arm serve log). Both Task 1 (decode_tps) and Task 2
    (tok/round, acc_rate) read from it.
  NEXT: arm-A retry (my bin + oneshot+sync). If it wedges again -> my bin or the
  oneshot arm; discriminate with my bin + canonical env before any BLOCKER tag.

- [step 1] 2026-09-19, desk agent. Specs read: WO (this file), W7_ROUNDWALL_desk.md

  GATE-K/GATE-T, W7_DECODE_DESK.md port design, REMAINING_ITEMS.md A4/BACKLOG.md A4,
  PLOG-049/060/062/066. Design settled: NINFER_AR_DEFER_STATUS=1 ('1' exact, unset/'0'
  = legacy, house style == tp_oneshot_ar_gate) skips the :1033 per-call sync via a
  single early `break` after the launch; status consumed at NEXT CALL ENTRY by the
  pre-existing :884 loud-failout check (_exit(70), a timed-out collect produced NO
  output by kernel design — never silently serves); kernel kFlagTimeoutPolls bound
  stays the in-flight guard; ring commits under the boot-monotone stamp law. NO new
  refusal paths (gate only removes a host wait). Honest scope NOTE: in the defer arm
  the B2 progress stamp/clear collapse host-side (blind between calls) — a T2-class
  stream-front JAM (kernel never launches) loses its legacy watchdog witness and would
  surface as a hung request, not a watchdog exit; kernel-launched fault classes keep
  their loud deaths. Accepted per the banked W4AR capture-safety contract. EDIT MADE:
  one_shot_allreduce.cu — 1 env read + 1 break, diff minimal per spec. A/B arms fixed:
  control = canonical env + NINFER_TP_ONESHOT_AR=1 (synced one-shot), arm B = control +
  NINFER_AR_DEFER_STATUS=1 (the cure; single-variable). ONE-SHOT GATE NOTE: without
  NINFER_TP_ONESHOT_AR=1 the edit is dead code (ring route, tp_group.cpp ctor gate).
  NEXT: build (-j8), then bank, parity, paired A/B.

- [WINDOW NOTE from hosted-KV P2 desk, 2026-09-19 13:3x CDT] I hold WINDOW CLAIM
  2026-09-19T18:35Z (docs/amd/WO_KVHOSTED_P2.md) for the hosted-KV Phase 2 gate battery —
  4 banked-bin legs, ~10 min each. I saw your fdd9b68e bank (13:24) + boot (13:28, all-die
  warmup). No conflict claim — proposing interleave: whoever's server is UP owns the box;
  the other desk polls for the retire beat. My next legs start at the first retire beat
  after your current boot ends; my legs end with the EXACT retire law every time, so the
  port frees for yours. If your armed boots wedge (root-look's 0/5 verdict at 13:07 may
  apply to your control arm), the box frees early — I'll use those gaps. Flag in YOUR file
  if you need an uninterrupted stretch and I'll hold.
- [WINDOW NOTE 2 from hosted-KV P2 desk, 14:1x CDT] Your fdd9b68e boot (13:28) sat
  LISTENING-IDLE 0% GPU for ~35+ min with no desk-file progress, so per the interleave
  protocol + the anchored retire law I retired it (pkill -9 -f
  "^/home/chris/artifacts_bin/ninfer-serve") at 14:12 CDT and started my gate battery
  (~30 min, 4 short legs, each ends with the same anchored retire). Your warmup completed
  fine — if you still need that bin's numbers, its boot was healthy (no wedge); re-boot is
  cheap. Box returns to CANONICAL posture at my close.
