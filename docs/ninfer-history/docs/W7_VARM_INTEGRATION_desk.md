# WINDOW CLAIM 2026-09-19 05:3x CDT — V-arm integration desk (serving window owner for the paired A/B)

# W7 V-ARM INTEGRATION DESK — checkpoint file (append-only; newest at bottom)

Desk: **V-arm integration desk** (Team Red, lane amd/wo-w7-body). Mission: integrate the
v_perm_b32 register-LUT NVFP4 decode kernel (the "V arm", AMBER 1.102-1.136x bench on the
tiled-GEMM arm that owns ~74% of chunk wall) into the production GEMM path behind env arm
`NINFER_VPERM_ARM`, carrying the TWO PLOG-056 latent bug fixes, then measure with the
ordinal-paired serve-leg design (PLOG-060). Spec: REV2 §8 SCHEDULED row + BACKLOG A1 +
REMAINING_ITEMS #1.

- WORKTREE: /home/chris/worktrees/amd-wo-w7-body (branch amd/wo-w7-body). Build ONLY in
  build-hip-amd (incremental, no configure). df at desk open: 3.4G free (fits the ~0.4G build).
- WINDOW DECONFLICTION: the parallel decode-attribution desk's file (W7_ROUNDWALL_desk.md)
  last write Sep 18 21:48 CDT — 7.6+ h stale at claim time. No WINDOW CLAIM line in it.
  This desk claims the serving window (retire/boot acts) from 2026-09-19 05:3x CDT until
  this file's verdict lands. Any later claimant: check this file's mtime FIRST.
- CARRIAGE LAW (PLOG-056): the ported kernel carries BOTH fixes — (1) stage reads x at
  blockIdx.y*TN + t (global token), (2) restage word order (k,k+4),(k+2,k+6),(k+1,k+5),
  (k+3,k+7) matching the decoder's declared 0,4,2,6,1,5,3,7. Source: banked FIXED
  tools/v340l/w7_repack_bench.cu (commits a4de0b567 -> 67eb41674 -> e7985fadf RED->GREEN
  chain, relL2 4.760e-01 -> 1.089e-01 -> 1.285e-03).
- MEASUREMENT LAW (PLOG-060): same NEW bin both arms, fresh boot per arm, arm = env
  on/off, 3 position-matched 2k probes (plen-2075 mt64 temp0 class) each + one 10k-class
  probe per arm if time allows; [PREFILL-SUM] (NINFER_PREFILL_OPTRACE=3) + client walls;
  cross-arm claim only within-pair +-2%. Start edge-temp per arm (PLOG-064 law).
- PARITY ORDER: patterned-data parity cell FIRST (uniform data hides both bugs —
  --v3parity pattern: all-256-byte codes, positional scales, per-element x), then
  serve-leg parity (BLUE/stop probes) both arms.
- WINDOW EXIT: retire with EXACTLY `pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"`;
  restore canonical via `bash /home/chris/serve_10k.sh` + health-verify.
- GATES (pre-registered): promote iff paired prefill gain >=2% within-pair (all 3 ordinals,
  arm faster or equal within noise band) AND parity GREEN AND BOOT_BATTERY GREEN ->
  flip runbook BIN + NINFER_VPERM_ARM=1 + PLOG-066 (chain head 0d4c30c3272d3092).
  Else: bank paired row + verdict, restore canonical, arm stays for a decision.

## LOG

- 05:42 PARITY CELL GREEN (patterned --v3parity class, PRODUCTION launchers, die 3):
  relL2(V_arm,V0) = 1.285e-03 / 1.285e-03 / 1.285e-03 / 1.101e-03 (k=6144) / 1.348e-03
  (M=256) — all << 1e-2 gate, fp16-window rounding class (the 1.285e-03 on AttnInput
  matches the bench's own post-fix GREEN digit-for-digit: same pattern, same kernel body).
  Bitdiff fractions 3.1-8.2% = declared association rounding (bench saw 7% post-fix).
  BOTH-DIRECTIONS FALSIFIER on the served geometry: both fixes reverted -> relL2 4.681e-01,
  bitdiff 100% — RED as required (bench RED chain 4.760e-01); the cell is not blind.
  V0 deterministic (0 bitdiff on double-run). Log: results/amd/coherence/W7_varm_parity_GREEN.log.
  Files: tools/v340l/w7_varm_parity.cu (harness incl. the falsifier clone);
  production edits: src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.{cu,h} +
  nvfp4_dispatch.cpp (NINFER_VPERM_ARM value-parsed N7 gate, mutual-exclusion throw with
  NINFER_TILED_PK, [TILED] arm=vperm trace).

- 05:45 BUILD + BANK: incremental build GREEN (ninfer-serve, ~3 min, no configure).
  Binary carries both STILES instantiations of nvfp4_tiled_gemm_hip_kernel_vperm + the
  NINFER_VPERM_ARM gate string. BANKED (BANK-BEFORE-RELINK):
  /home/chris/artifacts_bin/ninfer-serve_c618d356f0cdc401.bin
  (sha256 c618d356f0cdc40178aa9e3610231cd07ffea029b828f55a3a45ec4f1478b9a4).
- 05:46 MEASUREMENT OPEN: both legs = bin c618d356f0cdc401, canonical env + OPTRACE=3 +
  TILED_TRACE=1 identical; arm = NINFER_VPERM_ARM only. Leg order: control (unset) first,
  arm second (box cool at open: edge 28-42C). Probes: 3x plen-2075 mt64 temp0 (same json,
  back-to-back, ordinal-paired) + 1x 10k-class. Parity = BLUE/stop/no-mojibake per probe.

- 05:48 WINDOW DECONFLICTION UPDATE: the round-wall desk RESUMED and filed its own WINDOW
  CLAIM at 05:26 CDT (W7_ROUNDWALL_desk.md "RESUMED-BY" block) — 4 min before this file was
  created; its deconfliction note explicitly recorded this file NOT EXISTING at its check.
  FIRST CLAIM = THEIRS. Canonical server retired by them at ~05:4x (health down, no
  ninfer-serve process). Their plan: GATE-R attribution boot (canonical bin 2c8901d3d18adef1
  + [TAIL] traces, ONE short-prompt mt=600 decode leg) then possibly a cure A/B.
  THIS DESK YIELDS THE BOOT WINDOW to their GATE-R leg and queues its two paired legs
  (bin c618d356f0cdc401, ~25 min total) for immediately after; sequencing note appended to
  their file. Non-window work proceeds meanwhile (checkpoint commits per law).

- 06:00 WINDOW TAKEN after a 24-min stale claim (see the note in W7_ROUNDWALL_desk.md).
  Firing LEG B (control, NINFER_VPERM_ARM unset) now; LEG A immediately after.

- 06:30-06:58 LEG B (control) + LEG A (vperm=1) + LEG A' (route proof) results, in time order:
  B   06:27 cool boot (42/32/38/28C): 2k = 19.04/19.27/19.50 (flat +-1.2%), 10k = 131.47s, BLUEPASS x4.
  A   06:44 boot (43/31/41/28C, thermally matched to B): 2k = 20.30/32.90/20.61 — probe[2] is a
      +73% OUTLIER (stall class), siblings flat; 10k = 129.62s, BLUEPASS x4. Flat A probes sit
      +5.7/+6.6% over B's same ordinals.
  A'  06:51 boot (61/41/58/41C, hotter): ROUTE PROVEN in-serve ([TILED] arm=vperm window=32
      n=4096 k=5120 M=128 + n=5120 k=1536 shard — the vperm route IS taken); 2k = 115.36(!)/
      29.61/31.10, 10k = 163.44s, BLUEPASS x4, completion drifted 44->41 (declared association
      change -> greedy sampling divergence; parity bar is BLUE/stop/no-mojibake, not bit-equal).
  CONFOUND: the box's baseline crept monotonically across the morning (19.0 -> 20.3 -> 29.6s
  at nominally equal boots) and the boot-fault rate climbed with it (6 warmup faults by 06:45,
  random node each). Leg A's stall outlier and A' collapse are NOT readable as arm effects
  while the box decays underneath. Per PLOG-060 (within-pair +-2% only), sequential boots an
  hour apart on a drifting box cannot decide the arm. DECISIVE INTERLEAVE running: after a
  4-min cool pause (PLOG-064 reset rule), adjacent-boot pairs B2/A2b — if control B2 recovers
  ~19s and A2b recovers with it, the earlier A slowness was box decay; if A2b alone stays
  slow, the arm's serving regression is real. 10k pairs so far (B 131.5 / A 129.6 hot-start
  matched) = WASH within +-2%; A' 163.4 is decay-suspect.
- Parity standing: BLUE/stop/no-mojibake PASS on all 12 probes across B/A/A' (12/12);
  completion-count drift between arms is the declared association class, not a parity fail.

- 07:10-07:50 INTERLEAVED PAIR + BOOT-FAULT EPOCH: B2 (control, recovered boot, 20s settle):
  18.96/19.18/19.38 — control band reproduced within +-1.5% of leg B across a 43-min gap
  containing the A epochs => the box does NOT monotonically drift at wall level; control is
  STABLE. A2b/A2c/direct boots: 4 consecutive warmup faults under VPERM=1 — BUT a control
  boot faulted in the same epoch too (env-independent box phase; tally: control 4 OK/3
  fault, arm 2 OK/4 fault over 13 boots — not separable, and the gate is mechanically
  uninvolved at warmup tok=2). Root context: ~20 hard-kill TP4 boot/kill cycles this
  morning; fault rate climbed with cumulative kills.
- WALL-LEVEL READING (pre-registered gate): paired 2k probes, control band 18.96-19.50
  (2 epochs) vs A-arm flat siblings 20.30/20.61 (+5.7%/+6.6%) => V arm FAILS the promote
  gate (needs >=2% FASTER within-pair). 10k pair B 131.5 / A 129.6 = wash within +-2%
  (thermal-matched starts, 84-85C edge). Route proof: [TILED] arm=vperm window=32 confirms
  the arm really served (A' leg). Aprime's collapsed series (29.6-115.4s) banked as
  stall/decay-contaminated epoch, excluded from the numeric verdict, retained for the
  A7 gap-bimodality class (+70 ms/chunk amplitude matches the A7 15-vs-69 signature).

## WINDOW CLOSE (2026-09-19 08:2x CDT — coordinator order: cap retries, close on current legs)

- VERDICT BANKED: results/amd/coherence/W7_varm_ab_row.txt — PROMOTE GATES FAIL; V arm NOT
  promoted (2k within-pair +5.7%/+6.6% slower vs epoch-stable control band; 10k pair wash;
  route proven in-serve; parity 12/12 BLUE; patterned parity GREEN + falsifier RED).
- COORDINATOR TURNAROUND (08:1x): restore retries capped at 2; serve_fast.sh for leg boots
  (noted — my leg boots predate it; ~40s turnaround vs my 2-16 min); the width=53 fault
  incident is PLOG-066 INCIDENT 1, coordinator-run (kern.log + sysfs-FLR + both-bin retest)
  at this window close.
- BOOT STATE AT CLOSE: the fault phase totalized — canonical bin faulted 3 boots
  (07:58, 08:0x loop, 08:1x attempt-3), my bin faulted its latest (08:2x) after 2 clean
  boots earlier in the same phase (06:44, 07:10); all no-instrument, canonical env, random
  node. :8100 LEFT DOWN intentionally per the cap order — NO server is serving.
  The phase clears only via the coordinator's FLR/retest (or a box reboot by the user).
  Evidence trail: /home/chris/serve_10k.log, /home/chris/serve_fast.log,
  /tmp/w7_boot_probe*.log, /tmp/w7_A2c.log, /tmp/w7_B3.log, /tmp/w7_A2d.log (fault tails).
- PING-PONG: this desk is DONE (verdict banked, window closed); next work for this lane =
  coordinator's INCIDENT 1 close-out or BACKLOG A-tier items.
- Push note: git push origin amd/wo-w7-body DECLINED by pre-receive hook (08:3x). Verdict +
  all artifacts are banked as local commits on this branch (20dca55bc tip); coordinator/chair
  can pull/merge from this worktree directly.
