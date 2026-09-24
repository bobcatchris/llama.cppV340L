# CURRENT PLAN OF RECORD - 2026-09-24 (authoritative execution queue)

This file exists because session memory is unreliable. It is the single source of
truth for WHAT RUNS NEXT, with exact commands. Update it at every plan change
(append a dated revision block; do not silently rewrite). Precedence: the ledger
(OPTIMIZATION_PLAN_TP3_200K.md, append-only) records what HAPPENED; this file
records what HAPPENS NEXT. The dossier (CAMPAIGN_DOSSIER.md) holds the full
technical background for a new reader.

## 1. STATE OF RECORD (as of 2026-09-24 14:15)

- Serving of record: TP4 + RCCL + gates + fa40 var-11 direct arm.
  decode 24.55 short-prompt (ratchet E-137), prefill 217.71, accept 0.66-0.68.
  Binary lineage: build-hip @ commit c2374eaaf-era code (contains E-137/E-138
  merges; ratchet field fix 5ad25e902 is docs/tests-only).
- Baseline: docs/amd-port/tests/baseline_tp3_200k.json decode 24.55.
- Machine: boot #4 (10:27), noretry=1, svm-watchdog armed (svm-watchdog.timer),
  hogs historically 5 on this boot, ZERO deaths on boot #4 so far.
- Owner constraints (ACTIVE): no long prefill runs while levers are live
  (battery = 10k decode-only cells only; U1 v2 deep rerun DEFERRED); L7
  weight-bit reduction is owner-guarded; pushes to origin fork ONLY.

## 2. RUNNING RIGHT NOW

1. U1 DEPTH WINDOW (campaign-postreboot-manual chain): one persistent server,
   depth cells 10k (DONE: 23.78 t/s @127.8 ms, law -2.2%), 50k (prefill valid
   29.41 t/s; decode EOS-void), 100k (IN FLIGHT, decode ~14:10), 150k, 199k.
   Ends ~19:30-20:00. Outputs: /home/chris/u1_depth_*, /home/chris/u1_window_console.log.
   On completion the live-analyst commits W26 (depth table) and drafts E-140.
2. FLEET (3 agents, zero-GPU while U1 runs):
   - U1 live-analyst: mines cells as they land, maintains W26, EOS watch,
     acceptance-vs-depth prediction (cross-project L4 law: expect accept to
     RISE with depth; falsifiable on the deep cells).
   - W30 mmvq-c4 desk (wt-mmvq-c4, amd/mmvq-c4): GLU fusion T=4 lift +
     register-clamp occupancy rungs (VGPR 96/80), oracle-gated, bench staged.
   - W31 p2p-ar desk (wt-p2p-ar, amd/p2p-ar): flat P2P allreduce feasibility +
     probe prototype (bypass RCCL ring; ~-7.5 ms/cycle if feasible).

## 3. EXECUTION QUEUE (in order, with exact triggers/commands)

STEP 1 - when U1 ends (lock free + U1-DONE in /home/chris/u1_depth_summary.log):
  a. Short W27 q40-prefill bench (die 3, ~20-40 min, lock-arbitrated):
     bash /media/chris/ssd128/llamacpp/wt-q40prefill/docs/amd-port/scripts/run_q40prefill_bench.sh
     Prereqs to promote: oracle EXACT-or-DUST all cells AND v11p >= 30% at 10k+.
     If prereqs met: merge amd/q40-prefill -> rebuild build-hip -> CI ->
     U2 served window per W27 receipt section (kill: <+5% @10k, GARBAGE,
     decode_guard breach). NOTE: U2 needs prefill cells - owner constraint
     says defer U2 until the lever queue drains (it IS a lever; the constraint
     targets measurement-only prefill runs; make the U2 call at queue-drain).
  b. Bank E-140 from the analyst's W26 draft; commit; push_backups.sh.

STEP 2 - the post-U1 battery (5 windows, ALL 10k decode-only, ~8 h):
  nohup /home/chris/run_post_u1_battery.sh > /home/chris/postu1_battery.log 2>&1 &
  (self-gating: refuses while the lock is held; --only <name> for single windows)
  Order: w23env (env flips, ~-10 ms/cycle) -> ub1024 (re-promotion; decode
  penalty vanished on fa40 config) -> w19 (shape-cache; engagement==2 required)
  -> w22c1 (clock floors; needs root: run battery under sudo bash -c or pre-arm
  sudo; set_clock_floor.sh set/clear between arms, 300 s cooldowns) ->
  w28nmax (chain n_max=4; content-invariance sha check; the 29.4 t/s math test).
  Each window: R A A R cells, void-gated, paired verdict + ratchet line in
  /home/chris/postu1_window_results.txt. On any WIN: ratchet baseline +
  add env/flag to launch_tp3_200k.sh + all runner BASEENVs + rebuild + CI +
  commit + push (the E-137 promotion procedure).

STEP 3 - W30/W31 benches (interleave after battery or during its gaps, die 3,
  SHORT sessions): run each worktree's staged bench script (they wait on the
  lock themselves). Oracle first, timing same-session. If W31 P2P is feasible
  AND beats 69.5 us/boundary: owner sign-off needed on the numerics class
  (flat sum order = same class as the E-090 RCCL sign-off) before any served arm.

STEP 4 - W21 persistent-server Phase-1 shadow (cheap; the death-class cure):
  one persistent regress epoch cross-calibrated against one classic window,
  per docs/amd-port/results/W21_persistent_server_brief_2026-09-24.md s8.

STEP 5 - deferred items (only after the lever queue drains):
  U1 v2 deep-cell rerun (ignore_eos; /home/chris/run_u1_window_v2.sh);
  W29 registered falsification cell (optional -sm layer 10k boot, predict 8-9
  t/s, >15 reopens PP); parked paired re-runs (iq4xs/s2r5/chan4 superseded by
  the w23env window); L7 weight-bit reduction (OWNER-GUARDED).

## 4. OWNER DECISIONS PENDING

1. Persistent-server tiering formal adoption (W21 s8; review endorses).
2. P2P allreduce numerics class (if the probe proves feasible).
3. L7 weight-bit reduction (the only short-prompt true-2x door).
4. U2 timing (lever-queue-drain rule vs immediate).

## 5. RECOVERY (if a boot dies mid-queue)

- Hard death expected signature: svm hog escalations then instant reset.
  Post-reboot: campaign-postreboot.service self-fires run_postreboot_hardened.sh
  (verifies noretry + 4 PCI dies + identity gate vs last CODE commit; 4 hb cells
  vs the 24.55 anchor; fires U1 only if /home/chris/U1_ARMED exists - it does
  NOT, U1 already ran; then nothing else). After the queue: resume THIS file's
  step that was interrupted. Check git object integrity (fsck) after any hard
  death - two purges happened before; recover via reflog; then push_backups.sh.
- If an agent dies silently (has happened 3x): its worktree + branch survive;
  re-dispatch from the last CHECKIN.log line + branch tip. Never assume an
  agent is alive without a fresh CHECKIN line or file mtime.

## 6. POINTERS

- Ledger (what happened, append-only): docs/amd-port/OPTIMIZATION_PLAN_TP3_200K.md
- Technical dossier (full background, code, walls): docs/amd-port/CAMPAIGN_DOSSIER.md
- Reviewer triage + new levers: ledger E-142
- Cycle math correction (E-141): MMVQ ~65% of cycle, sync wall 24-25%, PP NO-GO
- Scripts inventory + laws: dossier Part 9 / ledger E-133/E-134

Revision 1: 2026-09-24 14:15 (initial; coordinator session).
