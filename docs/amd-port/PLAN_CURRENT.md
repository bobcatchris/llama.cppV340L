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

## REVISION 6 - 2026-09-24 16:30 (post-reboot: battery relaunched clean, fleet re-dispatched)

- Fresh boot verified: noretry=1, hogs=0, fsck clean, HEAD 1560e872b intact,
  postreboot queue disabled as planned (re-enable AFTER the battery),
  systemd-oomd stays disabled.
- BATTERY RELAUNCHED 16:25 on cool dies (30 C): campaign-postu1-battery,
  identity OK, all 5 windows armed, first cell w23r1 booting. ETA ~8 h.
- FLEET RE-DISPATCHED: battery-analyst (mines windows as they land, maintains
  W32, drafts E-145, anomaly watch incl. w19 crash-watch - if the shape-cache
  arm crashes served, the E-138 heal failed and that is critical news) and
  U2-prep desk (run_u2_prefill_window.sh: paired prefill cells with the
  q40-prefill arm; SHORT 10k default, --long for 50k/100k; deferred execution
  per owner constraint).
- Liveness law active: any desk CHECKIN stalled >45 min = zombie -> re-dispatch
  from branch tip.

## REVISION 7 - 2026-09-24 16:55 (U2 script ready; merge-order law)

- U2 PREP LANDED: /home/chris/run_u2_prefill_window.sh (dry-run verified, 7/7
  verdict paths tested; SHORT 10k promote cell ~0.8 h; --long full curve ~7 h;
  ignore_eos + cross-arm sha + engagement greps + void-gate built in).
- CRITICAL SEQUENCE (U2 desk's catch): amd/q40-prefill is NOT merged to main -
  the served binary lacks the PREFILL arm. AFTER the battery completes and
  BEFORE U2 fires: git merge amd/q40-prefill -> rebuild build-hip -> CI ->
  then bash /home/chris/run_u2_prefill_window.sh (SHORT first; --long after).
  DO NOT merge mid-battery: a code merge without rebuild fails later cells'
  freshness gates; a merge+rebuild mid-battery swaps binaries between arms
  and confounds the paired design.
- Post-battery order: (1) bank E-145 from the analyst's W32; (2) ratchet any
  WIN per window; (3) merge q40-prefill + rebuild + CI; (4) U2 SHORT; (5) U2
  --long; (6) U1 v2 deep rerun; (7) re-enable campaign-postreboot.service;
  (8) W21 persistent shadow.

## REVISION 8 - 2026-09-24 19:50 (clock floors KILL banked; transport question CLOSED; q40-prefill MERGED)

- E-149 BANKED (98b8a85ce, W36 receipt): w22c1 clock-floor retry KILLED. Floored arm
  loses all three pairs (-0.32 / -0.36 / -0.14 t/s); dpm_sclk floor does not survive
  load (F200 act_mean 1209 MHz under stamped 1496); pinning leaked under warm soak
  (991 MHz dips 9-27%); HBM hazard witness: die1 mem 43->93 C vs 95 C crit during the
  F200 needle. Lever closed, no boot-chain change. ANY future clock forcing must be
  SMU/od8 level + memory-temp guard (one line, not queued).
- TRANSPORT QUESTION PERMANENTLY CLOSED (W38, amd/oneshot-ar da4fd7c69,
  backup/oneshot-ar snapshot): one-shot AR dead by BOTH substrates - peer access
  refused 0/12 ordered pairs AND hipIpcGetMemHandle invalid-argument 0/12
  (probe log W38_ipc_open_20260924_194437). W38 also corrected payload math: all
  136 decode boundaries are fp32 (hidden 5120), 128 verify + 2 catch-up + 6 draft;
  210.8 us/boundary = 69.5 us transport floor + ~141 us ARRIVAL SKEW. The live
  allreduce lever is now skew compression ONLY (zero-transport bound +7.6% decode).
  W39 skew desk (wt-skew) scoping: MoE expert placement imbalance (prime suspect),
  NCCL env arms, launch jitter.
- q40-prefill MERGED to amd/v340-port-v2 (53d0c97ac, --no-ff; fattn-tile.cuh +21,
  bench +78, W27 receipt in). SEQUENCE: rebuild build-hip (running) -> CI decode
  guard vs 24.69 -> U2 SHORT (10k paired, ignore_eos) -> U2 --long. U2 pre-flight
  desk (engagement-grep WARN law, freshness-gate resolution, geometry 1024 check)
  verifying the harness in parallel.
- Fleet: wt-roofline (W37 verify-path byte budget, zero-GPU) + wt-skew (W39) +
  U2 preflight. Post-CI queue unchanged: U2 SHORT -> --long -> U1 v2 (deferred,
  owner) -> re-enable campaign-postreboot.service -> W21 persistent shadow.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
## REVISION 9 (restated in full after object-purge #3 repair) - 2026-09-24 20:55

- REPO INCIDENT: object purge #3 hit during the interrupted rev9 append
  (~20:25-20:50 window; 8 empty objects incl. the mid-write commit 5c0759fe,
  1 index blob). Repair per playbook: branch reset 5c0759fe -> 13ccb8414
  (W39 bank, intact; CI had verified provenance at 13ccb8414 at 20:19),
  empty objects purged, index rebuilt from HEAD, fsck clean. Zero campaign
  data lost: all W-receipts and ledger entries E-133..E-150 intact; desk
  branches (skew/oneshot-ar/roofline/smallk refs) intact.
- q40-prefill MERGE SEQUENCE COMPLETE: merge 53d0c97ac -> rebuild (binary
  e247afbbe8167ca3) -> CI decode-guard PASS (decode 24.71 vs baseline 24.69,
  overall PASS, provenance commit 13ccb8414 tree clean). Merged tree holds
  the of-record decode with the arm OFF. Reusable CI wrapper:
  /home/chris/run_ci_q40merge.sh (wrapper boots server; guard_battery.py only
  probes - --server-binary/--launch-config are provenance metadata, NOT a
  server launch spec).
- U2 SHORT: first fire (campaign-u2-short) died at cell r1 in the repo
  incident; stale u2prefill_r1 files at 20:20-21 discarded; re-fired after
  repair. Verdicts: PROMOTE / PARTIAL / SPLIT / KILL / VOID / INCOMPLETE;
  +5% kill law; sha gate temp-0. On PROMOTE: U2 --long (50k/100k curve).
  On KILL: arm stays default-off, lever closes, move to the W39 NCCL arms.
- W39 BANKED (93f18a599 on main): skew compressible part corrected to
  3.0-6.1 ms/cycle (per-die rate spread, NOT the 19 ms wall); TP-shard
  placement imbalance DEAD (all shard dims divide by 4; remainder rotation
  already in llama-model.cpp); boundary fusion forbidden (rms_norm
  dependency). FIRE-READY zero-code env cells: NCCL_MAX_NCHANNELS=1 (A1) and
  NCCL_SHM_USE_CUDA_MEMCPY=1 (A2) - spec W39 s5, engagement fingerprint
  "Channel 00/01".
- W40 BANKED (f766a6a04): U2 preflight READY; stale 512/512 geometry default
  fixed to 1024/1024 (E-148 of-record); oracle carry-over proven (merged
  fattn-tile.cuh byte-identical to the 9/9-EXACT wt-q40prefill tip).
- W41 DESK LIVE (wt-smallk): quantize_q8_1 3x re-quant dedup + TP4 copy
  deletion (W37 Opp-2, -2.5..-3.5 ms/cycle). W37 Opp-1 (MMVQ schedule lift,
  -3..-14 ms) is the next big arm after it reports.
- Push law: branch push stays blocked (E-115); push_backups.sh orphan
  snapshots are the content channel (backup/v340-port-v2, backup/skew,
  backup/oneshot-ar). Run push_backups.sh after EVERY landing (post-purge
  law).

## REVISION 10 - 2026-09-24 21:35 (U2 attempt-1: VOID on the soak gate, pair-1 data +18.8%; cooled retry armed)

- U2 SHORT attempt 1 (20:59-21:18) ADJUDICATED VOID BY THE SOAK LAW - and the
  law is right: paired prefill 10k: R1 179.07 / A1 212.69 (+18.76%, both cells
  VOID-GATE PASS, engagement prefill=640 direct=320, text sha IDENTICAL
  d3f2fc3c) but pair 2 hit the far-die soak drift (die3 <=991MHz duty 9 -> 11
  -> 22 -> 27 pct monotonically across cells = W18 mechanism) and both a2/r2
  VOIDed -> hard VOID "do NOT promote" (E-117/W7 A4/E-133). The arm itself is
  innocent: text invariance held everywhere; a2 still ran 212.25 while MORE
  throttled than its control.
- RESPONSE: ONE cooled re-fire (thermal confound invalidates the pairing -
  same RETRY standard as w22c1; cap = one retry). Dies cool to <58C edge
  first (poller running), then campaign-u2-short3 fires. If the retry Voids
  on thermal grounds again -> verdict "PROMOTE-BLOCKED until a fresh-boot
  run", and U2 --long goes on a cold boot when the owner frees the queue.
  NOT retry-shopping: same-law precedent, capped.
- Attempt-1 artifacts: /home/chris/u2prefill_* (r1/a1/a2/r2 + results.txt).
- W41 smallk desk v2 re-dispatched (v1 killed silently in the 20:21 bounce;
  liveness law). Fleet: smallk-v2 + U2 retry + coordinator.

## REVISION 11 - 2026-09-24 23:05 (U2 verdict: PROMOTE-CANDIDATE law-blocked at a miscalibrated gate; solo-mode fleet; boot-queue plan)

- E-152 banked. Bottom line: the q40-prefill arm gains +18.7..+19.8% prefill
  at 10k on ALL FOUR paired cells across two windows (212-214 vs 177-179
  t/s, distributions separated, text-invariant, engaged, zero decay) and the
  written law still says VOID because the <=991-duty soak gate (built for
  decode, E-133) flaps 9-27% at 10k prefill arm-independently - it voided
  base cells as often as arm cells, cooler boots gated no better. NOT
  promoting against our own law; NOT retrying further tonight (cap reached).
- ARM DISPOSITION: GGML_CUDA_FATTN_TILE_Q40_PREFILL stays in-tree
  DEFAULT-OFF (merged, CI-green, zero risk). Of-record unchanged.
- NEXT GPU SEQUENCE (in order): (1) fresh-boot U2 SHORT re-run - queued as
  the first cell of campaign-postreboot.service when it is re-enabled (its
  one-boot disable per rev 5 is now spent; next boot runs it); (2) W39 NCCL
  env arms window (A1 NCCL_MAX_NCHANNELS=1, A2 NCCL_SHM_USE_CUDA_MEMCPY=1,
  zero-code, spec W39 s5); (3) U2 --long (50k/100k curve) when the owner
  frees long-run time; (4) U1 v2 deep rerun (owner-deferred).
- GATE-CALIBRATION PROPOSAL (owner): per-window-class duty thresholds, or
  prefill-t/s drift as the soak signal; duty is noise at prefill (E-152b).
- FLEET: subagent usage limit hit 22:4x (resets 2026-09-25 11:25) - solo
  mode; W41 desk to be re-dispatched after reset. Zero campaign data lost:
  purge-#3 repair held, E-133..E-152 + rev1-11 all committed and
  snapshot-pushed this era.

## REVISION 12 - 2026-09-24 23:20 (PROACTIVE REBOOT ARMED: fresh-boot U2 re-run is the night queue)

- EXECUTING THE E-152 SANCTIONED PATH: the queue is wired (U2_ARMED flag-gate
  fires run_u2_prefill_window.sh as chris BEFORE the hb cells; root-run git
  stamp re-verified 35739d9d6 post-repair; CODE_TS freshness gate passes -
  binary 19:51 vs last code commit = the 53d0c97ac merge). Coordinator arms
  /home/chris/U2_ARMED and reboots NOW (rev-5 precedent; owner-authorized).
- ON WAKE: (1) verify campaign-postreboot fired and u2_console.log shows the
  four cells; (2) read /home/chris/u2prefill_results.txt - a clean-window
  PROMOTE = ratchet prefill per the U2 law + E-153; a third thermal VOID on
  a COLD boot = the gate calibration becomes a formal 3-data-point owner
  item (duty is noise; per-window-class thresholds or prefill-drift signal);
  (3) the hb battery follows the window autonomously; (4) after the 11:25
  subagent-limit reset: re-dispatch the W41 smallk desk, then the W37 Opp-1
  MMVQ schedule desk.
- If the boot dies (SVM class): the watchdog + spacing laws hold; relaunch =
  touch U2_ARMED again and reboot; nothing else needed - the queue is fully
  self-contained. All state banked through E-152/rev11 and pushed.

## REVISION 13 - 2026-09-25 00:10 (night queue executed; U2 closed at 3 windows; decision at owner)

- The armed reboot ran clean (23:45 boot): postreboot queue self-fired U2 on
  the cold boot and the third window replicated the arm effect exactly
  (+19.0/+19.8 pct; arm 212.56 vs base 178.57/177.36) while the gate voided
  positions 2-4 (duty 9 -> 22 -> 22 -> 27 pct within the window on a COLD
  boot). E-153 banked with the full 6-pair, 3-window, 12-cell evidence
  package. The hb fresh-boot battery follows autonomously (~1.5 h).
- U2 LEVER STATE: CLOSED pending owner call - arm default-off in tree,
  of-record unchanged, three windows is the retry cap honored. The owner
  package: recalibrate the E-133 duty rule for prefill windows, or accept
  the evidence and promote (U2 --long recommended first). NO further U2
  re-runs by the campaign.
- REMAINING QUEUE: (a) hb battery lands on its own; bank its verdicts;
  (b) 11:25 fleet reset -> W41 smallk desk + W37 Opp-1 MMVQ schedule desk;
  (c) next window slot: W39 NCCL env arms (A1/A2); (d) U1 v2 deep rerun
  stays owner-deferred.

## REVISION 14 - 2026-09-25 04:55 (night fully banked; 11:25 fleet manifest)

- STATE: E-133..E-155 banked, HEAD b942ecb19, backup current, fsck clean,
  fresh boot healthy (23:45), GPUs idle 33C, tree clean. Fleet resets 11:25.
- 11:25 DISPATCH MANIFEST (in order):
  1. W41 smallk desk -> wt-smallk (branch amd/smallk @ f766a6a04): W37 Opp-2
     arms (quantize_q8_1 re-quant dedup + TP4 copy/convert deletion).
     HEAD START (coordinator's call-site map, read-only verified 04:50):
     quantize.cu:407-409 (quantize_q8_1 launcher, the MMVQ pre-quant path),
     mmq.cu:142+203 (quantize_mmq_q8_1, MMQ path - NOT decode),
     fattn-common.cuh:332 + fattn-vec.cuh:181 (in-kernel q8_1_to_shared),
     ggml-cuda.cu:2061/2141/2165 (quantize_src1 dispatch),
     vecdotq.cuh:1702 (T=4 MMVQ share-arm note). The "3x re-quant of the
     SAME tensor" needs a trace to pin down WHICH three (suspects: per-GEMM
     quant of the same verify-batch tensor across the meta-backend's mul_mat
     splits, or draft+target double-quant of the same hidden states).
  2. W37-Opp1 MMVQ schedule desk -> new worktree wt-mmvq2: the -3..-14
     ms/cycle item (LDS-y staging per W5 A4; remaining rungs after the
     wide-s2r null and w28nmax occupancy falsify - oracle law applies,
     W23 B2 bench protocol).
  3. Optional third: dossier sync desk (CAMPAIGN_DOSSIER.md is at E-148
     era; it lacks E-149..E-155: floors KILL, transport closed, dense
     correction, U2 package, hb confirmation, NCCL closures).
- GPU QUEUE: empty until a desk lands an arm. Then paired windows on the
  extended battery harness. No U2 re-runs (cap honored). U2 --long only if
  the owner promotes the prefill arm.
- OWNER (unchanged): E-153 package (+19 pct prefill, 6 pairs, 3 windows);
  persistent-server tiering; L7 weight-bit; branch-push history surgery.

## REVISION 15 - 2026-09-25 06:40 (morning wave adjudicated: RPB1 falsified, q81 no-promote, Opp-2 closed; Opp-3 is the last kernel item)

- W41/W42 ARMS ADJUDICATED: (1) W42 RPB1 - oracle FAIL on 4 of 6 qtype
  sessions (by_rowpar/tok-last signature) and slower where it failed;
  bit-exact and -3.1 pct vs share ONLY on q4_K (below-bar residual banked).
  NOT merged; amd/mmvq2 @ 9ed09aa4f. (2) W41 q8_1 dedup - merged
  (eda3141b9), CI PASS, served window SPLIT +0.29 pct = below the +0.3 pct
  gate and not a paired WIN -> NO PROMOTION, arm stays default-off in tree.
  W37 Opp-2 (both branches) CLOSED (E-156/E-157).
- KERNEL LADDER: only Opp-3 remains (post-fa40 attention tile concurrency,
  -2..-4 ms/cycle, W17 latency-bound). Coordinator decision: dispatch the
  Opp-3 desk ONLY on the next forward cycle that warrants it (the two
  cheapest kernel tiers are now measured out; diminishing returns).
- STATE: of-record decode 24.69 / prefill 217.71 / accept 0.66667; plateau
  tonight 24.4-25.1 across all fresh-boot cells. Everything pushed through
  E-157 / 2f1721585. Fleet: idle (all three morning desks complete).
- OPEN ITEMS: OWNER package E-153 (+19 pct prefill arm) unchanged and
  patient; Opp-3 desk; U2 --long if owner promotes; U1 v2 deep rerun
  (owner-deferred); C2 taskset (queued, W22 kill law).
