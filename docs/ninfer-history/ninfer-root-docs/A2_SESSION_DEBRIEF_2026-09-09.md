# A2 session debrief — 2026-09-08/09 (NVFP4 decisive cell + CI honesty + W5 re-arm) — handoff document

Purpose: the next session cold-starts from this file without re-deriving anything.
Companion read order (in order): THIS FILE → `results/157_MISSION_STATE_and_handoff.md`
(main, §1-4/§6/§9) → `docs/157_P4_requant_placement_evidence.md` → `docs/157_P4_nvfp4_qtype_tag_trace.md`
→ `results/157_ci_rerun_20260910_report.md` (ci/full-20260910 branch) → this file's §5 for W5.
All SHAs below verified to exist at write time (see §6 self-check).

## 1. Tasks owned this session — status

### 1.1 NVFP4 placement lane (docs/157 mission §8 item 1) — COMPLETE, BANKED
- **BRANCH NOTE (read first): ALL of §1.1's commits + the three fixture tests live on
  `wo/bf16-w8-requant` (tip `42a04bce` — A1's merge-payload/Path-W line that also carries
  his #91 fix + D0.5). They are NOT on wo/radiance-launch-gap. The ci/full-20260910 farm
  run executed them because that branch MERGES the two lines (rehearsal content).**
- **BF16→W8G32 load-time requant** in materialize_tp (pass-1 sizing + pass-2 writer):
  exporter-faithful quantizer (scale = f32(f64(amax)/127) → f16 RNE, subnormal floor 2^-24,
  reciprocal f32(1/f64(scale)), codes nearbyint clamp [-127,127]); overflow GUARDS match the
  exporter's ValueError (host __float2half SATURATES, so magnitude is tested directly —
  conservative in (65504,65520], unreachable, fail-closed). Commit `5563465b` + docs `6fdffaaa`.
- **NVFP4 blockscale TP-shard placement** (wall-3, coordinator-approved): codes row-major [N,K/2]
  sliced per role, scales unswizzle(FULL)→slice→swizzle(LOCAL) per layouts.py
  (n=m*128+a*32+b, j=kt*4+c — FOUR 16-groups per 64-wide K-tile; a first impl using kt*16 was
  caught by fixture byte 16900), weight divisor 4B verbatim; fail-closed N%128/K%64/col0.
  Commit `5563465b`; fixture test `ninfer_nvfp4_shard_host_test` (9 placement + 4 contract).
- **Provenance correction (in 5563465b message)**: STATUS doc attributed pass-1 death to BF16
  attention; truth = NVFP4 obj#13 precedes BF16 obj#56 (NVFP4-first). Fixed at merge time.
- **Batteries (all CPU-only, registered in tests/CMakeLists.txt)**:
  bf16 15/15 vs numpy re-impl of tools/convert/common/quantize.py; nvfp4 shard 11/11 vs the
  REAL exporter (tools/artifact/layouts.py encode_nvfp4 imported by
  tests/multi_gpu/gen_nvfp4_shard_fixture.py) + 4 contract rejections + gqkv4-pair
  disjoint-assembly audit. Loader-adjacent suites 10/10. **These are the sweep's regression
  gate — keep them green under any change; targeted relink only; PRUNE build/tests after
  (disk standing rule).**
- **DECISIVE RESULT (banked, A1's window + report `2e85c460`)**: NVFP4 weights TP2 =
  2,013.6 t/s decisive prefill, +94.6% vs int-class; 40.5k bit-identical; cell-4 kernel grep
  proved nvfp4_w4a4_tma/nvfp4_gemv hot. THE MISSION ANSWER: the sglang gap was weight format.
- Decoded caveat for citations: decode 87.0 vs 80.2 baseline is k4v4-vs-k4v2 KV config delta —
  "record, do not claim as a weights win" (report §2 finding row has it).

### 1.2 CI honesty lane — COMPLETE (4 commits on main, all pushed)
- `a44db174` vacuous-farm class: forced -DBUILD_TESTING=ON reconfigure + cache VERIFY;
  ctest_ran_real helper (>0 tests AND 0 failed; proven vs TONIGHT'S REAL vacuous log);
  CTEST_EXIT CLOBBER fix ([2b] farm result was overwritten by [3/3] before verdict — farm got
  CTEST_FARM_EXIT); Build-OK empty-stat hard fail; commit_data.sh d41d8cd9 tell.
- `78196ea8` EXTRA_OK=0 init moved above all setters (nccl P2P regression printed-then-swallowed;
  phase gates were double-carried via PHASE_GATE_EXIT, only nccl was a real victim).
- `50dd70d6` verify-battery renderer coercion (str baseline crashed whole table → NO DATA row).
- Named CI cell `requant_placement_gate` + G4 wiring live on wo/sweep-merge-prep @ `68157fc0`
  (merge via rehearsal; NOT yet on main).
- **First all-real farm run** (ci/full-20260910 @ `22e7db03` + completion report):
  144 tests, TWO reds — #91 linear_nvfp4_a4 (REGRESSION-candidate, fixed by A1 `9dd68fad`
  "restore Residual17408-full return"; provenance: never executed pre-tonight) and
  #137 tp2_interleaved_prefill (KNOWN-FLAKY: 3 of 4 recorded runs abort; host-RAM class;
  skip-guard now implemented, see §3). Reality columns are permanent banner features.

### 1.3 W5 re-arm — S1+I4 IMPLEMENTED, verdict = run-2 (fresh grant required)
- **S1** (`0091c7bb`): mid-chain syncs/observes deleted (both chains); round-end bulk arm
  from host-mapped slots via new accessors (OneShotArgmax::conf_from_payloads_at(rank,step),
  rank_step_now; TpGroup forwards). OneShotArgmax::conf_from_payloads now delegates to
  _at(step-1) — bit-identical. R2 veto honored (kernel untouched).
- **Run-1 battery** (runner `44c51706`): 466 arms across gates, ZERO hangs/desyncs — the
  36-hour hang class is DEAD. But G2/G3 RED: armed stream ≠ OFF at 2k (deterministic).
- **Runner self-catches (7 total vacuous-greens project-wide today)**: v1 rung() empty-string
  short-circuit skipped ALL gate bodies → "6/6 GREEN in 25s" — audited, never reported;
  byteid() unbound $3 (set -u); NO stop-on-red in v1; byteid diagnostics polluted command
  substitution (failure strings as shas). All fixed through `34b66a8d` + `d0dca6c7`
  (stderr isolation, nonempty() on ALL parity comparisons, G0-halt, stop-on-red x6,
  REQ_TMO/LISTEN_TRIES/LISTEN_SLEEP knobs for fast self-tests).
- **Discriminant** (`904e6e37`, tools/bench/w5_discriminant.sh + results):
  C1 fixed-k == C2 adaptive at 4 ctxs × 2 passes, deterministic — shipped adaptive is
  LOSSLESS; S1 arm-path is the sole culprit.
- **UNIFIED ROOT CAUSE → INVARIANT I4**: adaptive changes depth BEFORE emission
  (step-6), so emitted==verified always; S1's delayed arm verified a PREFIX of what was
  emitted — first mechanism ever to break that. Explains BOTH: width-dependent commits
  (Problem 1, 2k divergence `1afe030d` reproduced 2×) AND mtp-pool holes (Problem 2:
  appends at nominal width, rewind at reduced — `committed=52 slot=62` signature; the
  fail-closed hole detector REFUSED, zero corruption). >=3258 ctx in C3 = hole refusals,
  NOT divergence (battery verdict lines mislabel ERR as diverges — serve logs are truth).
- **I4 FIX** (`22f3c8c5`): arm clamps EMISSION at post-step-6 (after adaptive n_cur update;
  neither clobbers), clamped width PUBLISHED via pending_depth = next verify's ONLY input
  (adaptive drift between rounds cannot reopen verify>emitted); pending_arm = same-iteration
  carrier. Round 0 + post-reset carries: full width (sentinel/staleness guard).
- **INVARIANT WRITTEN**: `docs/WO_w5_break_deadlock.md` §18.5b (I4 named, binding, evidence).
- **Gate-2 amendment**: §18.5a — value 11782 within rounds 16-18 = green (S1 consumes one
  round late); exact round = bonus; BOTH fail = real red. NOTE: A1's provenance answer still
  open — 11782 was derived from WHICH body? (runner drives the lighthouse unit; A1's trace
  body may differ; green-by-convergence fallback covers the wrong-body case).

### 1.4 W5 battery run-1 artifacts — evidence, some lines VOID
- results/w5_battery_20260909_065347/ (serve logs + summary; G4/G5 lines VOID — runner bugs;
  G1/G3 lines STAND; G2 = hole-refusals). results/w5_discriminant_20260909_073554/ (clean).
- logs/w5_battery_VOID_run1.log = the caught fake-green; logs/w5_i4_smoke_mixed.log also
  committed on branch as results/w5_i4_smoke_mixed.log = REJECTED evidence (3 interleaved
  request histories in one serve instance — the reason the runner uses fresh serve per gate;
  I violated my own design for a 7-min smoke and voided it; do NOT cite).
- G5 cell spec-broken: 200k int8 + MTP slots = 16,895 MiB > 16,310 preflight REFUSE. Align:
  ctx 131072 (launchable) or NINFER_PREFLIGHT_DISABLE=1 documented in-cell (historic cell
  used the old override world).

## 2. Environment / protocol (binding; NEW traps from this session marked ★)
- Mesh: intercom `coordinator`, `agent1`, `agent2` (self). agent_comm for gemini (down days,
  owner-absent rule: you found it in shared tools/ops → you fix it, coordinator ratifies).
- Cards need the coordinator's WRITTEN grant BY NAME; guard 15 MiB/0%/0-apps before every
  serve; PID/pgrep -x kills only (never pkill -f; setsid detaches — $! is the wrapper).
- ★ GRANT SHA RULE: verify which SHA your BINARY carries and STATE it in every report. A
  grant named 42a04bce as "branch moved, docs-only" — it was actually wo/bf16-w8-requant
  (different lineage, PRE-S1); building there would have tested unfixed W5 and faked a
  verdict. Caught by content-grep (armed_conf_base count) before launch.
- ★ BRANCH IDENTITY before EVERY commit in shared worktrees (A1 checked out
  ci/full-20260908 inside MY worktree mid-session; my merge landed on the evidence branch —
  zero harm, but repair needed update-ref + reflog reading).
- ★ command-substitution hygiene: functions that both report and return data must print
  diagnostics to STDERR (say()→stdout pollutes captured shas; stub proof caught it).
- ★ ctest-with-no-tests exits 0; empty-vs-empty equality passes; "no tests found" text is
  the vacuity signature. Every runner gate needs nonempty() + count guards.
- ★ A stale CMakeCache (BUILD_TESTING=OFF from a prune) makes run_ci skip the ON-configure —
  run_ci now asserts (a44db174), but fresh worktrees need explicit
  `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DBUILD_TESTING=ON`.
- ★ Generator/worktree generators (gen_*_fixture.py) write into tests/ relative to CWD —
  run from worktree root.
- wo-radiance build dir = MAKE generator (cmake --build); wo-nvfp4-prefill = ninja. Don't mix.
- /tmp is reboot-volatile-ish (bodies survived today's reboot but /tmp/p2p_probe did not):
  regenerate pattern in results/157_SESSION_DEBRIEF_20260908.md §3 (10k row token math there
  is UNVERIFIED ~6% — flagged).
- Worktrees: wo-radiance (W5, branch wo/radiance-launch-gap), wo-nvfp4-prefill (rehearsal/
  ci branches), wo-nvfp4-sweep (A1's). /home/intel/ninfer/repo = shared main checkout —
  branch-identity check before commit (★ above).
- Disk: 19-20G free at close; farm relink ~16-21G — PRUNE build/tests after campaigns
  (df gate before >5G builds).

## 3. OPEN ITEMS (queue, in order)
1. **W5 battery run-2** — fresh grant (~90 min, single-owner cards), one paste:
   `BINARY=<fresh build of wo/radiance-launch-gap @ eaa92607> bash tools/bench/run_w5_battery.sh`
   (--gate N to resume). Expectations: G1 green (arms=0), G2 per §18.5a (value 11782 in
   rounds 16-18 OR convergence fallback), G3/G4 green vs fresh OFF (I4 makes emission ==
   verify — run-1's divergence should be GONE; if 2k still diverges with emit-clamp lines
   present, I4 is INSUFFICIENT — escalate, don't iterate blind), G5 aligned per §1.4,
   G6 all-CLEAN+arms>0+sha==D. W5 default STAYS OFF regardless; flip = user decision incl.
   the k5v4-tier open item (k5v4 rounds take the UNIFIED route, not packed — decide at flip).
2. **Cherry duty**: 9d23f2e6 (rc-masker + skip-guard fixes, on ci/full-20260910) → main at
   the coordinator's nod (it's already in the merge-payload path via rehearsal — confirm at
   merge time).
3. **Merge chain** (coordinator-sequenced, pending user nod): wo/bf16-w8-requant (A1's #91
   fix 9dd68fad + D0.5 38c05d2b + Path-W plan 42a04bce + dispatch-table test b0f5ba73)
   ⊕ rehearsal wo/sweep-merge-prep @ 95a6d530 (zero-conflict rehearsal proven, needs one
   final sync) → main → WO-G5 post-merge sweep (spec docs/157_WO_postmerge_regression.md,
   owner A1-exec/agent2-backup/coord-interprets, PORT 8099 locked, /tmp-body regen
   precondition). Merge bar: C1+C2 green.
4. **Path-W weight pre-transpose** (A1 will send a one-line layout spec): tp_load pass-2 =
   my lane; precedent + fixture pattern ready (blockscale placement is the template).
5. **A1 async review** of I4 diff (22f3c8c5) — he knows step-6 mechanics; review context =
   discriminant results + §18.5b.
6. **W5 flip decision items for the user** (after run-2 green): k5v4 open item + MTP_ACCEPT
   baseline re-pin per KV dtype (0df9e7fd landmine) + default flip.

## 4. Key SHAs (map, all verified)
- main: `50dd70d6` (renderer fix; carries a44db174 vacuity, 78196ea8 EXTRA_OK, 1ff3bac5
  harness, docs trail) — pushed.
- wo/bf16-w8-requant: `b0f5ba73` (dispatch-table test) ← A1's merge-payload line incl.
  `9dd68fad` (#91 fix), `38c05d2b` (D0.5 flatten, 2,556.3 t/s), `42a04bce` (Path-W plan).
- wo/radiance-launch-gap: `8e9a6028` tip = S1 `0091c7bb` → §18.5a `dc4c5d4a` → runner
  `44c51706` → I4 `22f3c8c5` → §18.5b `eaa92607` → smoke-rejection `8e9a6028`.
- wo/sweep-merge-prep: `95a6d530` (rehearsal; +79af5fbf local content = main∪rehearsal,
  zero-conflict proven; final sync before merge).
- ci/full-20260910: `22e7db03` (gate run + report) + `9d23f2e6` (masker/skip fixes).
- ci/full-20260908: `dc3372be` (FROZEN vacuous evidence — never reuse the name).

## 5. W5 S1/I4 mechanics (what the next session must not re-derive)
- Round flow (post-I4): top → two-barrier pending consume (1947-1959 region) → verify at
  published width → accept sync → **CONSUME** bulk conf (armed_conf_base/len staged by
  PREVIOUS iteration's chain; staleness guard vs rank_step_now catches reset_step rewinds)
  → pending_arm = n_draft → propose d0 (capture armed_conf_base) → step-6 adaptive n_cur
  → **EMIT-CLAMP** rk=min(rk,pending_arm), publish pending_depth=rk, extents write →
  chain emits rk-1 → stage armed_conf_len=rk → pack.
- Chain ALWAYS emits full nominal for round 0; rounds ≥1 emit the clamped width; verify
  ALWAYS reads the published width. I4 holds by construction.tau=OFF ⇒ conf_gate false ⇒
  every W5 block inert (byte-identical, gate 1 proves).
- Line numbers in §18.4/§18.5 prose are STALE (tree drifted) — site NAMES are the truth
  (d0 observe block / loop-top should_break / per-step observe; all now deleted).
- Battery runner v2 knobs: REQ_TMO, LISTEN_TRIES, LISTEN_SLEEP (fast self-tests);
  BINARY/ARTIFACT required; --gate N; --echo-plan; --contract.

## 6. Self-check (this doc was re-read against the repo — all verified)
- Every SHA in §4 resolves (git cat-file) ✓; every referenced file exists on its stated
  branch ✓; every referenced test/binary name matches tests/CMakeLists.txt ✓;
  results dirs (w5_battery_20260909_065347, w5_discriminant_20260909_073554) exist on
  wo/radiance-launch-gap ✓; results/157_ci_rerun_20260910_report.md on ci/full-20260910 ✓;
  docs/157_P4_* on wo/bf16-w8-requant ✓. Post-write corrections applied during re-read: (1) §1.1 branch note added (placement/requant
work is on wo/bf16-w8-requant, NOT radiance); (2) §1.3 now records A1's ASYNC REVIEW of
22f3c8c5: APPROVED, zero blockers — traced every pending_arm leak path (3 break sites),
confirmed the skip-branch (arm fires but adaptive rk < armed_conf_len → pending_depth 0 →
verify NOMINAL >= emitted) is the safe shipped regime, and corrected the impossible-direction
wording: verify>emitted REMAINS POSSIBLE and is safe; what is now impossible is verify<emitted.
Nits applied: §18.5b wording fix + clamp-block skip-branch comment. One path A1 could not
verify statically (armed_conf_len derivation for the round AFTER a clamped round) = run-2's job.
Gaps found and fixed while writing: §1.2 originally
  claimed stop-on-red existed in v1 (it did not — corrected); §1.4 originally omitted the
  G5 preflight detail and the open A1 provenance question (added); §3 originally omitted
  the 2b-cell merge note for requant_placement_gate (added).

## 7. Run-2 scheduling proposal
Proposed slot: ~13:00-14:00 EDT (4-5 h sleep from close; sharp > fast). Fresh 90-min grant;
single-owner cards; runner honors per-gate fresh serve so coordinator can slot new-A1's
packer confirm around it; if the merge nod lands mid-battery, pause at the NEXT GATE
BOUNDARY (never mid-gate). STOP-on-red is wired; expect G1/G3/G4 green under I4, G2 green
per §18.5a (value-window or convergence), G5 green in aligned shape, G6 all-CLEAN — 6/6 =
report to user for flip decision (k5v4 + baseline re-pin riders attached).

---

## ADDENDUM (same session, 2026-09-09 15:25Z) — run-2 ran, RED; fix landed; run-3 armed

§1.3/§3-item-1 above are SUPERSEDED. Actual outcome:

- **run-2** (granted 14:26Z): G0 ok / G1 GREEN / **G2 RED-BY-DEADLOCK at 8 min** — hang at the
  first cross-rank clamped verify. Evidence + root-cause: `results/w5_run2_halt_analysis.md`
  + `results/w5_battery_20260909_102659/` (single-request clean log). The VOID-smoke
  (8e9a6028) retrodicted as an early sighting of the same bug.
- **Root cause ≠ the I4 mechanism this doc feared**: three STACKED defects in the arm path —
  (1) the conf carry is shared by-reference between the two rank worker threads and the
  consume was single-shot (first thread read AND zeroed it; second never evaluated —
  explains r1's silence); (2) shared `pending_arm` read/written cross-thread unordered;
  (3) the post-propose per-request reset zeroed the counter under the staged round-0 carry
  (the STALE rank-flip tell). I4 itself was sound — §18.5b still stands.
- **Fix `33ef6b3d`** (A1 re-review: ACK, all four asks confirmed): double-barrier consume,
  worker-local `my_arm` (shared pending_arm deleted), post-propose reset deleted
  (branch-entry resets suffice). New **§18.6 INVARIANT I5**: ranks must AGREE by
  construction; shared mutable arm state BANNED. Nits closed `21acb9bc`; G7 cell `a3a4bc17`.
- **run-3**: GRANTED-SEQUENCED behind A1's farm (coord ping ~16:15-16:45Z). BATTERY IS NOW 7
  GATES: G7 = 80k tau-ON flip-decision cell (user question "is W5 working at 80k?"): strict
  byteid==same-session-OFF@80k + zero lockstep violations + accept-sum datum (no pinned
  expectation). BINARY after fix = sha256 66d372aa8406 at tip 0ea17863 (provenance chain: bd999b5a = run-2's
death binary; 89c65cec = I4-R fix as A1 reviewed; 0940841b = +comment-nit; 66d372aa = +main's
70522b46 budget-backport [tp2_budget.h only — 17/16 preflight double-count, lane-sync miss,
coordinator transit-trust ruling]). Arm-path bits byte-unchanged since A1's ACK. Any later
commit restating 'the binary' MUST re-run sha256sum, not copy this line.
  Paste unchanged. Halt clock: 8 min. W5 default STAYS OFF; flip memo must carry the G7 answer.
- Battery-adjacent queue unchanged (cherry duty, merge chain, Path-W pass-2) + new: run-3
  verdict report per-gate, then A1 ping + flip-memo data handoff.

---

## ADDENDUM 2 (session end, 2026-09-09 17:32Z) — runs 3+4, two more classes closed, divergence LOCATED

Post-15:25Z state (supersedes addendum 1 where they conflict):

- run-3: G1 RED (arms=6 @ tau=0.01) -> class: no-data conf (prod=0.000000) read as confidence-zero.
  Fix I4-R2 d61b25a8 (A1 ACK 1ae9140d: floor proof conf>=~6.8e-6; window-skip over neutral-1.0;
  nit f2c76bbc). run-4: G0+G1+G2 GREEN (lockstep PROVEN again 62/62 pairs; lossless under
  21/53 clamps @ tau=0.7) -> G3 RED 8k@tau=0.80: FIRST GENUINE LOSSLESSNESS FAILURE, and
  cell-1 (FORCE_A0@0.80, af3db8a6 + results/w5_cell1_forcea0_20260909/): output == OFF
  BIT-EXACT with 62 clamp-pairs firing => locus = DRAFT-COMMIT COUPLING (licensing/mtp-pool
  ar_hidden/rewind-re-append family), verify-forward-width EXCLUDED.
- BINARY lineage final: 50738ac97e07 (f2c76bbc content) = tonight's tested good; earlier hashes
  superseded. Budget backport af3db8a6..0ea17863 in lineage (main 70522b46 semantics).
- G5 GREEN standalone; runner bug filed: --gate N prints full GREEN banner (fix before trusting
  any partial-run verdict line).
- CI side: #26 exit fixed+verified green (8e134dee, live exit 0); #137 parser-was-always-broken
  saga CLOSED honest (cb9f5d82 chain; guard now derived-demand, fail-closed, renders Skipped).
- TOMORROW (fresh): (1) hunt draft-commit coupling (start: rewind-re-append vs mtp-pool phase +
  lic[] read-bounds — NO new theories without a dying cell); (2) gate-spec amendment commit
  (completes+hangs+holes+divergence-datum; A1 eyes first); (3) G6/G7 (ROWP DONE 17:58Z @8c0100f1: REGRESSION -7.7% @10k / -11.1% @40k at 71-76% clamp rate, mechanism-consistent at 70% accept — see addendum-2/relay line); (4) emit-symmetry
  follow-on (:1560 conf_gate) + G1-premise re-read; (5) runner banner fix; (6) memo assembly —
  contract options with tonight's LOCATED facts, incl. invariant-tiers code-facts from
  gqa_attention_kv_quant.cuh:27. W5 OFF throughout; user decision stands on the three options.
