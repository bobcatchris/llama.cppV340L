# A2 SESSION NOTE — 2026-09-11 ~04:1x EDT (m3_boot1 stall -> increment-4 + P-A shipped -> boot-ready)

> UPDATE (same session, later): everything between here and the close-out is superseded by
docs/A2_GREENSUMMARY_m3_boot3.md (+ the LATER arc below) — P-scorecard 4/4, G1 wall = engine fall-through (fixed,
9a76439e), decode route COMPLETE, first clean acc read = 0.000/31 steps -> quality A/B plan
is the live work. Read THAT doc first if you are resuming.

Resume-point doc (supplements, does not replace, docs/A2_DEBRIEF_2026-09-11_0615Z.md, now
committed as-filed). Read COORDINATOR.md seq 104-118 for the full ping-pong; this is the lane view.

## What happened this session (all CORD-sequenced, zero protocol violations recorded)
1. Resumed on C441 GO (static GREEN on 67b18499). Verified BIN 2f0d14a14b39 = ninja-no-work at
   tip, fresh guard, lane0/1 reverts (NOTE: 1.lane0 was STAGED-modified; `git checkout -- file`
   does NOT unstage — had to redo with `git checkout HEAD -- file`; add to any future SOP).
2. Boot m3_boot1 = the deciding run. Result: FOURTH SHAPE (CORD amended pre-registration to
   include (iv) early-barrier-spin): loop advanced normally (both ranks byte-identical drafts
   through step=29), then FROZE at step ~30: one thread pure-userspace spin (1000 ticks/10 s,
   wchan=0), all others futex, GPU0 100%/GPU1 0%. p1r1's full 58-round request took 12.7 s —
   13+ min frozen = hang, not slow. Teardown PID-scoped, evidence in results/bcode_m2/m3_boot1/
   (classification doc: docs/A2_STALL_m3_boot1_classification.md).
3. A1's static read (accepted): arena-cap theory DEAD (chain scratch rewinds per round; 45 MiB
   headroom at rd 29). Stall = rank-1 control-flow divergence (silent exit or bare park) with
   rank-0 spin-waiting a collective. He also found a REAL surviving grower (below).
4. THREE commits shipped onto the boot tree (src frozen now at fdd2327c):
   - d58b20ba: rank-1 IMMEDIATE exception capture. KEY INSIGHT: the old RANK1-ERR prints sat
     AFTER th1.join() — unreachable exactly in the m3 shape (rank-0 never returns from the
     spin, so the join never happens). Print moved INTO th1's catch (split exception/unknown),
     fflush'd. Also p1.sh card body now derived from argv (mislabel class).
   - 6030296d: INCREMENT-4. (a) GROWER HOIST: accept-tier mb_* + hook-tier d2h_* (60.2 KiB/rd,
     kills the arena at ~795 rounds = farm7 landmine) moved pre-loop, constant shapes, in-loop
     = memcpy only; inner mb_scfg shadow deleted (the pre-loop one at the seed-tier block is
     the live handle — its per-round SamplingConfig re-upload verified unchanged). (b) SYNC
     PRINTS: NINFER_D2_SYNC_TRACE-gated (static bool, zero cost unset, rank+step+fflush) before
     allreduce_argmax(verify), round_end_bar, and the chain fn's vocab-allgather. A missing
     r=1 line beside a present r=0 line at the same step IS the divergence site.
   - fdd2327c: P-A v4 (G2): verdict now (win_a, win_b, lattice_equal, chains_equal) per A1
     29f06e57 — census-shape FP class closed (unit anti-ratchet row proves it: equal windows +
     DIFFERENT lattice + equal chains -> LegitimateConvergence, no throw). Wiring INSIDE
     run_dflash2_chain verbose block, d2_N>=2 + lattice_witnessed guards (INERT on twin:
     seeding there is N=1; round loops verbose=false), lattice witness REUSED from 2b-ii
     lane_hash (two bools bubbled out of the block), [2B-IV-V4] L/W/C line pre-dispatch.
     Unit 7 checks ALL PASS. NOTE: `ctest` in PATH is a broken python shim (ModuleNotFoundError
     'cmake'); /usr/bin/ctest works — flag to CI lanes.
5. Scripts: p1bt.sh = re-boot wrapper with TEETH — freeze detector (10 s sampling, 90 s static)
   -> EXEC gdb -batch 'thread apply all bt' into bt_all.txt BEFORE any kill. WHY exec: yama
   ptrace_scope=1 FORBIDS sibling attach (proven 3x; this is why m3 forensics had no stack);
   the server's parent shell replacing itself with gdb is the only legal tracer (verified
   end-to-end on a dummy, full frames). On bt: wrapper pid is gone BY DESIGN; server+curl left
   alive; kill_list.txt has the PIDs; NEVER pkill system-wide. CORD decision (a): boot tree =
   HEAD fdd2327c, BIN e74560a03a65 — cards honest: cbada7c2/54f8a135 are SCRIPT-ONLY (preflight
   into wrapper; max_tokens truth-read from body JSON — m2b bodies have NO max_tokens, '128'
   was template fiction. RETRACTED INFERENCE (CORD seq-16): server default is 8192 and
   bodies carry NO cap fields — p1r1 gen=30 was a NATURAL STOP (EOS), not a default-cap;
   that re-read puts the m3 step-29/30 hang AT a stop/teardown boundary (seq-16
   hypothesis; readings amended in p1bt.sh card line 5);
   s3 shape_class field is a copy-downstream cosmetic, filed not papered).
6. Era line formalized: 2f0d14a14b39 = stalled m3_boot1 / a4c9ac9c992e = capture-era /
   68f000ab0b56 = increment-4-only / **e74560a03a65 = BOOT BIN** (fire-check v3.1 GREEN on it).
7. Pre-registration for the re-boot (CORD seq-10 merge + A1 + mine): divergent-round(race,
   A1's EXPECTED default — arena headroom says nothing pins round 29) / exact-29-repeat
   (deterministic step-29 event — FLAG LOUDLY) / green-loop(flake) / capture-names-throw
   (+my sub-reading: same-step ±1 WITH a named [D2-SS-RANK1-ERR][immediate] line = collective
   deadlock with dead peer, escalation datum, NOT a fifth boot).

## WHERE THINGS STAND (verify at resume — do not trust this)
- Awaiting: A1 cumulative static on 6030296d+fdd2327c (watch-points: hoist completeness, print
  placement, capture non-lossy, twin-inertness PROOF incl. throw-path-at-N=1) -> CORD GO stamp
  (serial GPU, cards confirmed idle since 03:00, agent3 acked revoke) -> ONE COMMAND:
  `cd /home/intel/ninfer/worktrees/wo-dflash2-w5-habitat && nohup bash results/bcode_m2/p1bt.sh
  m3_boot2 > results/bcode_m2/m3_boot2_wrapper.log 2>&1 &` (nohup+redirect, NOT the bare-bg-then-
  same-call-tail pattern — the file isn't visible to the tool shell until the call returns; the
  boot WAS live the whole time at m3, only the log tail raced).
- Per-step pings to CORD. Halt-on-first-divergence. If (iv) REPEATS with capture naming the
  throw -> STOP, report escalation datum (no freeform 5th boot, CORD seq-7 rule 4). If 6th
  distinct surface appears -> CORD fire-with-label decision point (09:4x rule).
- Post-green-loop path: S3-kill (p2_s3kill.sh; bars 21.7/31.5 @ 256-tok; CONDITIONAL mean
  acc>=~0.2) -> then quality datum decides: (i)-clean => phases 2-5 + S3-kill; (ii) => A1's
  .cu GDN write-target read is A2's in sequence. P-A commit then goes to A1 REVIEW (already
  asked), OPTION_B_LANDED + GUARD_V4 flips, farm7 package (build/tests prune for 25 G floor;
  df now 23 G, CORD says REQUIRED nvfp4 18 G in incoming/ stays — prune habitat build/tests
  only), user CI order.
- Disk: 23 G — boot-safe (MiB-scale logs), farm7 floor needs the prune call at package time.
- Tree: branch 167-piecewise-fix @ 54f8a135 (scripts) / src @ fdd2327c; clean vs HEAD except
  the usual untracked CI-dump strays (deliberate era-bundle material at close; commit or
  .gitignore in the drain, do not leave for the next session to guess).


## LATER SAME SESSION (post-boot3) — quality mechanism FOUND and FIXED; AB3 = verification boot, one stamp away
- AB1 dissolved the era-regression question (vpos arithmetic: one real accept EVER, at
  p1r1 step ~17; acceptance ~zero since the twin loop first ran).
- AB2 (bdccb60c) cleared the verify-side state layer on four static sub-checks.
- A1's parallel PRIME read then NAMED IT (c7114a7a): the twin hook fed context_append
  an UNWRITTEN d2h_fused — f89f5db1's original fusion-feed fix had been silently
  dropped by the arena refactor's sweep. Coord cold-verified (0 fuse calls in twin
  range). AB3 was re-sequenced from diagnostic to FIX-VERIFICATION.
- IMPLEMENTED 4045bfd4: batched-exact fuse mirror + FIX-D null parity (shape audit +
  provenance rows in msg; three self-caught cite-drifts disclosed post-push). A1
  one-word GREEN on the real diff (his seq-17). A1's GO word is NOT the boot stamp:
  CORD's written GO outstanding. When it lands: `bash results/bcode_m2/p1ab3.sh
  ab3_cellrun` (dual-cell, success bar = ANY nonzero twin acceptance; P5/P6/P7 +
  NULL + per-cell canaries on the card; join via ab3_join.py at close). Then:
  drain+era bundle, A1 reviews P-A already landed, and the farm7 package assembles
  (prune decision + arena disposition one-liner + OPTION_B_LANDED/GUARD_V4 flips).
- Era BIN chain tonight: 2f0d(stalled)->a4c9(capture)->68f0(inc4)->e745(P-A)->cf0b
  (counter)->3871(boot3-era, fall-through fixed)->4009(instrument delta)->e143(TODAY's
  fix BIN, AB3 boots on it).

## POST-BATTERY STATE (~07:1x EDT, tip 9dbf8df0) — frozen-lane, awaiting user + CORD
Sequence since the boot-ready note: m6_boot1 P8 (first complete twin request: 200,
stop, 24/.369/2.85 — banked history) -> ab3b cell-T P10 (byte-IDENTICAL to m6
through step 13 but FROZE -> real race, determinism refuted by my own comparison;
C4-as-cause refuted; S1 'fix' shown by my own withdrawal + A1's gate to be
rotation-innocent HARDENING, kept labeled not-the-fix) -> battery bb_fixbin
5 PASS / 1 HANG (run3 = same wall, argmax K-trace CLEAN on both outcomes: 272
lines zero passes -> argmax door EMPIRICALLY CLOSED) -> KAR instrument for the
allreduce family STAGED (9f30dab7ee3b, never booted): that family polls against a
flag that is 1-from-first-call between 128-wraps with a CHANGE-GATED dev_epoch
upload (:215-217) — a host-side TOCTOU the argmax never had; my prime suspect
now, A1's gate ask live (incl. whether KAR alone can see a torn-payload read or
the hunk needs a payload gen-stamp before boot-7).
IF RESUMING: read docs/A2_AB3B_P10_VERDICT.md (incl. the WITHDRAWAL section) +
battery summary; GPU belongs to A3; NO BOOT without CORD's written stamp (user
eyes on the same-wall rule); boot-7 = P1_BIN/staged-BIN battery-shaped N-run with
KAR armed, verdict map per CORD seq-64 amended (B-B' = rate datum only).
Farm-merge prep items live: G7 re-measure + prune, 169 package (four-costumes
exhibit, arena+reset+S1 keep-as-hygiene rows, sibling one_shot_allreduce filing
with the HOT-collective exposure correction, 'completes at parity 5/6 / 1/6
hang' honest sentence for fire-with-label IF the user chooses).

- KAR VERSIONS: v2 (gen-stamp, REJECT-candidate) = staged BIN a804d598e817, current boot-7 candidate; v1 (9f30dab7ee3b) superseded-not-booted; A1 re-gate on v2 outstanding.

## CLOSE-OUT (~08:5x EDT) — one-shot family BURIED both routes; residual = accept-kernel per-device divergence; cards to baseline
- boot-7 b8_hunt run1: DECISIVE CONTROL-HANG — armed KAR, hang at the same
  step-13 terminal, ZERO REJECT lines -> door (iv) closed ON THE TWIN.
- b9/b9b cell-B pair: unarmed run (user's multibatch question: PASSES, gen
  24+31 both stop) + armed re-run (cross-route audit: 0 REJECTs across ~2500
  batched AR calls) -> (iv) BURIED BOTH ROUTES; the unarmed-vs-armed delivery
  gap caught and filed BEFORE the zero was read as a conclusion.
- (iii) static (docs/A2_III_STATIC_VERDICT.md): tail-input table complete;
  every input to the rank-1-only exit is replicated or complementary-consistent
  EXCEPT the accept kernel's own per-device OUTPUT. Residuals (A) uninit-
  envelope read (alloc-history fits the script correlation better than timing),
  (B) device nondeterminism. Instruments proposed, NOT built: D2SS-TAIL
  per-rank a+lic print, or the cheaper 0x5A-poison rate probe first.
- Era chain FINAL: e24951a1800b (m6/ab3b) -> d94965428744 (battery, argmax
  cleared) -> [9f30dab7ee3b v1 staged-not-booted] -> a804d598e817 (KAR-v2,
  boot-7+b9b armed BIN, the LAST booted era). Tally across the wall's life:
  8 PASS / 3 HANG post-everything.
- NEXT SITS WITH: A1 (compare-before-concluding on the input table; his
  surfaces subsumed by rows 4-6 unless he finds a misclassification), CORD
  (boot-8 shape decision: tail-print vs poison-probe vs convene), USER
  (170 pack is his index; multibatch answer is armed-live).
