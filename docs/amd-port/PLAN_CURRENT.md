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

## REVISION 2 - 2026-09-24 15:20 (U1 stopped by coordinator; benches before battery)

- U1 was discovered ALIVE but degraded at 15:03: a second instance (51383) had
  restarted from cell 10k after the 14:26 oomd kill of the first instance's
  server (systemd-oomd socket-activated at 14:26:55 and SIGKILLed the U1
  cgroup; it also killed the user terminal - oomd is now DISABLED system-wide,
  kernel OOM-killer remains as backstop). The second instance ran the OLD
  script: its 150k/199k decodes would hit the systematic depth-dependent EOS
  wall (2/2 deep decodes EOS-killed: 50k, 100k - both prefill-valid).
- COORDINATOR DECISION (owner constraint: no long prefill runs while levers
  are live): U1 stopped by PID (script 51383, server 57098, queue wrapper
  4830); U1_ARMED removed. U1 ends PARTIAL: 10k full (23.78 @127.8 ms, law
  -2.2%), 50k/100k prefill-valid (29.41/15.63 t/s) with decodes EOS-void,
  150k/199k not run.
- NEW ORDER: (1) W27 q40-prefill bench RUNNING NOW (first cell result: v11p
  dst BIT-EXACT 0/786432 at prefill M=512/d=7168; staged KQ-X is a DBG
  capture-region artifact, stop rule refined to dst-is-truth); (2) bench
  finisher agent runs W30 (C4+occupancy) and W31 (P2P probe) benches;
  (3) then the 5-window battery; (4) then the U1 v2 deep rerun WITH
  ignore_eos - it is now REQUIRED (EOS is systematic, not coin flips) and
  will regenerate the full depth table including 150k/199k + deep acceptance
  points; (5) W21 persistent shadow; (6) W29 falsification cell optional.
- systemd-oomd is DISABLED system-wide (it killed the U1 cgroup and the user
  terminal under memory pressure; kernel OOM-killer remains the backstop).
  Watch RAM: 62 GB total; the server peaks ~18 GB + builds.

## REVISION 3 - 2026-09-24 15:25 (REBOOT-PROOFING: agent re-dispatch manifest)

Agents are children of the ZCode session and die on any reboot/crash. Their
WORK survives (WIP-commit law). On the FIRST post-reboot standing check, the
woken session MUST re-dispatch the fleet from this manifest (each desk resumes
from its branch tip + CHECKIN.log tail):

| desk | worktree | branch | resume from | re-dispatch action |
|---|---|---|---|---|
| W30 mmvq-c4 | wt-mmvq-c4 | amd/mmvq-c4 | 36ac2f47c + uncommitted bench wiring | finisher: commit wiring, build bench, run SHORT oracle+timing (die 3 = HIP idx per PCI 0d:00 runtime resolve), adjudicate |
| W31 p2p-ar | wt-p2p-ar | amd/p2p-ar | a3216df8b + 993dfb0b9 (probe staged; verdict biased-negative pre-GPU) | finisher: commit strays, run staged probe (4 dies, all-die brief window), record empirical P2P verdict |
| W27 q40-prefill | wt-q40prefill | amd/q40-prefill | 9af2edc3a; bench binaries /tmp/fa40_bench{,_dbg} (REBUILD after reboot: hipcc -O3 --offload-arch=gfx900 -I ggml/include -I ggml/src -I ggml/src/ggml-cuda bench_attn_real.cu vs build-bench libs, rpath; see section 3 of W27 receipt) | run staged bench, adjudicate (dst-is-truth stop rule) |
| U1 v2 deep rerun | n/a (script /home/chris/run_u1_window_v2.sh) | n/a | REQUIRED (EOS systematic 2/2); after lever queue drains | fire v2; it regenerates 150k/199k + deep decodes + acceptance points |
| battery | n/a | n/a | /home/chris/run_post_u1_battery.sh (5 windows) | fire when benches done + machine free |

RE-DISPATCH RULE: first post-reboot cycle = read PLAN_CURRENT + ledger tail,
verify noretry/hogs/dies, run push_backups.sh, re-dispatch per this table
(zero-GPU desks first, GPU benches second, battery last), then normal loop.
Liveness law: every desk CHECKIN.log must advance <=45 min or it is a zombie ->
re-dispatch from its branch tip immediately (3 silent deaths today, zero work
lost - the law held).

## REVISION 4 - 2026-09-24 16:10 (battery live)

- Binary REBUILT (16:06) to include the W30 arms (mmvq-c4 merge touched mmvq.cu);
  identity gate correctly forced it - the gate did its job.
- BATTERY LIVE: campaign-postu1-battery.service (system scope), all 5 windows
  armed, w23r1 booting 16:06. Results: /home/chris/postu1_window_results.txt +
  /home/chris/postu1_w*_10k logs. ETA ~8 h (~midnight).
- On completion: adjudicate per-window verdicts -> ratchet any WIN (E-137
  procedure) -> then U2 (q40-prefill served window) -> then the U1 v2 deep
  rerun (REQUIRED, ignore_eos) -> W21 persistent shadow.
- Next agent wave (post-battery analysis) dispatches when results land.

## REVISION 5 - 2026-09-24 16:30 (PROACTIVE REBOOT for a clean battery run)

The battery hit hog counter 7 (death zone) within 2 cells - 20 server cycles
per battery churns the SVM class fast on a worn boot. DECISION: clean reboot
NOW (no crash risk), postreboot queue DISABLED for the next boot only (its hb
cells are the soak source and would degrade the fresh boot before the
battery), then: re-enable campaign-postreboot.service, relaunch the battery:
  echo <pw> | sudo -S systemd-run --collect --unit=campaign-postu1-battery \
    -p User=chris -p Environment=HOME=/home/chris -p WorkingDirectory=/home/chris \
    /home/chris/run_post_u1_battery.sh
(dies will be cool; position-1 cells = clean conditions; counter = 0).
After the battery completes: re-enable campaign-postreboot.service.
State at reboot: w23env window in flight (no completed cells - restarts clean);
all other state banked through E-144/rev4. oomd disabled (stays disabled).
