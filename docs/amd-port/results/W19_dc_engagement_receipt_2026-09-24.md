# W19 DC-ENGAGEMENT DESK - P1 ROOT CAUSE + A2 FIX + A3 HOST-VERIFY + CORRECTED SERVED SPEC

Date: 2026-09-24. Desk: DC-ENGAGEMENT (wt-dc-eng, amd/dc-engagement, base
e116dbb1d). Zero GPU. Question: why was the engagement line "draft shape cache
enabled (2 slots)" ABSENT from the dc1 arm's server log (E-126), leaving the dc
window unable to distinguish "engaged with no effect" from "inert env"
(-0.39/-0.09 uninterpretable)? E-117 law: engagement REQUIRED before any verdict.

## 1. P1 - ROOT CAUSE: TWO STACKED DEFECTS, BOTH PROVEN

### 1a. LAYER 1 (decisive for the dc window): STALE SERVED BINARY

- The dc window launches `build-hip/bin/llama-server` from the coordinator tree
  (/home/chris/run_combined_window.sh:16, BIN=); the dc window driver matched it
  verbatim (arm identity blocks combowin_dc{0,1,0b,1b}: "== ARM IDENTITY ==",
  BASEENV line, "bin: ... (built from: ...)").
- The served binary + every lib in that bin dir are dated 2026-09-23 15:24-15:25
  (llama-server 15:25:48). The draft-cache merge b72cd1732 landed 2026-09-23
  20:36:01 -0500. The window ran 2026-09-24 00:58-01:06.
- `strings` scan over llama-server AND every lib*.so* in the served bin dir:
  ZERO occurrences of "draft shape cache". The gate code (src/llama-context.cpp
  env gate) was NOT in the served process. dc1's env WAS delivered
  (combowin_dc1_10k line 2: "arm: dc1  extra_env: LLAMA_DRAFT_SHAPE_CACHE=1")
  - env present, code absent: the arm was inert BY CONSTRUCTION. The -0.39/-0.09
  paired deltas compared IDENTICAL binaries (dc0 vs dc1 differ only by an env
  var the binary never reads); "within-noise" is the only possible reading.
- The arm-identity law's own witness was BLIND: the identity block's build stamp
  `bin: ... (built from: $(git -C ... log --oneline -1))` is EMPTY in all 4 dc
  cells (the window ran as root - the dc cell files are root-owned - and git
  refused the non-owned tree; the error is swallowed by the $( ) capture). A
  non-empty stamp would have shown a pre-merge commit and stopped the window.
- Provenance stamp corroborates: "commit=unknown ... binary=938a6999f1d8beb1"
  (guard_battery hashes the binary but does not tie it to a commit).

### 1b. LAYER 2 (would still have bitten after a rebuild): LOG-VISIBILITY FILTER

The engagement print path, traced end to end in the merged tree:

1. env: `getenv("LLAMA_DRAFT_SHAPE_CACHE")` - src/llama-context.cpp:230
   (gate :228-239 in llama_context::llama_context, :34).
2. SERVED PATH REACHES THE GATE: draft-mtp builds the draft context via
   llama_init_from_model (tools/server/server-context.cpp:1262 and :1286,
   `ctx_dft.reset(...)`) -> the same llama_context constructor; the target ctx
   passes through it too, so a set env prints once per context construction
   (target + draft = 2 lines expected).
3. print: `LLAMA_LOG_INFO("%s: draft shape cache enabled (%d slots)\n", ...)`
   - src/llama-context.cpp:237 (pre-fix).
4. routing: LLAMA_LOG_INFO -> llama_log_internal (src/llama-impl.h:28 ->
   src/llama-impl.cpp:55) -> g_logger_state.log_callback.
5. server wiring: the server installs `common_log_default_callback` as the
   llama log sink (common/common.cpp:373, inside common_init(), which the
   server runs before common_params_print_info prints the verbosity line).
6. THE FILTER: common_log_default_callback (common/log.cpp:454-459) maps
   GGML_LOG_LEVEL_INFO to LOG_LEVEL_TRACE = 4 (common/log.cpp:444,
   common_get_verbosity) and prints only if `verbosity <=
   common_log_verbosity_thold`. The served launch passes no -lv: thold = 3
   (dc1 log line 1: "verbosity = 3 (adjust with the `-lv N` CLI arg)").
   4 <= 3 is false -> the engagement line is DROPPED even on a fresh binary.
- Proof from the served log itself: combowin_dc1_10k.server contains ZERO
  library INFO lines (no "llama_context:" lines at all) while srv/slot/cmn
  lines print - those go through LOG_INF/COM_INF macros directly
  (LOG_LEVEL_INFO = 3 <= 3 passes); every src/llama-* INFO line is filtered.
- Same filter hides the W13 A4 verdict witnesses: "[decode-timeline] n_tokens
  = ..., reused = ..." (src/llama-context.cpp:1730, LLAMA_LOG_INFO behind
  LLAMA_DECODE_TIMELINE) and "[launch-timeline] meta rebuild:"/
  "[launch-timeline] meta nodes = ..." (ggml/src/ggml-backend-meta.cpp:1877,
  :2374, GGML_LOG_INFO behind LLAMA_LAUNCH_TIMELINE) - 0 occurrences in every
  served window log banked so far. The E-117-class exposure is protocol-wide:
  an operator following the W13 A4 spec verbatim could NEVER have grepped the
  required evidence from a served log.

## 2. A2 - FIX (byte-exact host logic; env-gated default OFF unchanged; commit d7d2ce398)

- src/llama-context.cpp: the engagement line INFO -> WARN (survives served
  thold 3; the sibling env-gate notice "graph reuse disabled" one block above
  is WARN for the same visibility reason). Message text UNCHANGED.
- src/llama-context.cpp: the 4 LLAMA_DECODE_TIMELINE-gated [decode-timeline]
  lines INFO -> WARN (same text).
- ggml/src/ggml-backend-meta.cpp: the 2 LLAMA_LAUNCH_TIMELINE-gated
  [launch-timeline] lines INFO -> WARN (same text).
- All three are env-gated diagnostics: default-OFF paths print NOTHING new;
  compute paths untouched (log level token only). Offline parsers
  (profile_hostslice_rounds.py, profile_verify_round.py, profile_launch_timeline.py)
  match on message text, not the level marker - verified compatible.
- Layer 1 (stale binary) is a PROVENANCE defect, not a code defect: fixed
  mechanically by the A3 freshness guard + the corrected served spec (section 4).

## 3. A3 - HOST-VERIFY (commit dcbc873ef; build + CI PASS)

1. Mirror suite extended (docs/amd-port/tests/test_draft_shape_cache_host.cpp,
   section 6): source-pins the real call site - gate env name
   LLAMA_DRAFT_SHAPE_CACHE, level WARN, canonical text "draft shape cache
   enabled (%d slots)", and the reused= witness line at WARN. hipcc
   COMPILE-EXIT:0, RUN-EXIT:0, ALL PASS.
2. NEW real-chain test (docs/amd-port/tests/test_engagement_routing_host.cpp),
   linked against the built tree libs, drives the REAL
   llama_log_internal -> common_log_default_callback chain at served
   conditions:
   - FRESHNESS: the build-hip libllama under test must contain the engagement
     text - a STALE build fails here. NEGATIVE CONTROL PROVEN: run against the
     coordinator's actual served bin dir (the E-126 stack) -> FAIL (exit 1),
     "binary_contains ... draft shape cache enabled" - the guard would have
     caught the dc window defect pre-merge.
   - WIRING: llama_log_get reports common_log_default_callback installed.
   - FILTER thold=3: the WARN engagement line ARRIVES, an INFO probe does NOT
     (the E-126 visibility defect reproduced + fix proven).
   - FILTER thold=4: the INFO probe arrives (direction sanity).
   Build note: the ROCm 6.2 hip-link driver mis-resolves a SECOND -l flag
   ("unable to find library"), so the libs are passed as -Wl inputs (verbatim
   in the CI section).
3. Canonical full build in this worktree: configure
   (/home/chris/opt/cmake/bin/cmake, GGML_HIP=ON, Release, GGML_NATIVE=ON,
   CMAKE_HIP_ARCHITECTURES=gfx900, GGML_HIP_RCCL=ON, LLAMA_CURL=OFF)
   CONFIGURE-EXIT:0; full build BUILD-EXIT:0, zero compiler warnings (only
   deprecation-warning.cpp source filenames match "warning"); fresh
   libllama.so carries the engagement text (strings count 2).
4. CI: test_engagement_routing_host wired as section 2b of
   /home/chris/run_premerge_ci.sh (CI_TREE-based, zero-GPU safe, runs with or
   without CI_SKIP_GPU). Verdict on this tree @ dcbc873ef:
   CI-VERDICT: PASS - tree clean, 8/8 host suites (incl.
   test_draft_shape_cache_host + test_engagement_routing_host), gate-wiring
   section CI-SKIP per the zero-GPU law (merge gate re-runs it). Log:
   results/W19_premerge_ci_2026-09-24.txt.

## 4. CORRECTED SERVED SPEC for the next dc window (the E-126 re-run law)

1. FRESHNESS GATE (new, hard): rebuild the served tree AFTER merging
   amd/dc-engagement, then BEFORE boot verify from the launcher:
      strings <BIN_DIR>/libllama.so | grep -q "draft shape cache enabled"
   and fix the identity stamp so "built from:" is non-empty (the dc cells ran
   as root: run the window as chris, or add the tree to git safe.directory) -
   a silent-empty stamp is a blind arm, E-126 lesson. The merge gate CI now
   fails on a stale build-hip (section 3.2), but the launcher must self-check.
2. Arm: of-record launch script + ONE env line LLAMA_DRAFT_SHAPE_CACHE=1
   (everything else identical to dc0; interleaved A/B per the E-119 soak law).
3. Engagement proof (NOW greppable at the served verbosity 3, no -lv change):
      grep -c "draft shape cache enabled (2 slots)" <log>.server   # expect 2 (target + draft ctx)
   Optional deeper gates, still zero-new-tooling:
      LLAMA_DECODE_TIMELINE=1 -> grep "\[decode-timeline\] n_tokens = 4" reused=1
      AND step 1 reused=1 in steady rounds, issue ~1.1 ms (vs 2.593);
      LLAMA_LAUNCH_TIMELINE=1 -> "[launch-timeline] meta rebuild:" ~2x/round
      BY DESIGN (uid alternation re-derives, memo replays; the ggml-cuda
      recapture tripwire stays SILENT after boot).
4. Battery: unchanged (guard_battery.py 5 cells + interleaved pairs). Expected
   delta still up to ~3 ms/round (~+2-3%) with the honest bound that the 2
   shape re-entries still re-derive the 49-node meta subgraphs.
5. Rollback: unset the env - default OFF path untouched.

## 5. Defects of record

1. E-126's dc window served a pre-merge binary: all 4 cells (dc0/dc1/dc0b/dc1b)
   measured identical code; the verdict "no promotable served win" is VOID as a
   statement about the shape cache (it remains valid as "no effect measurable
   under that protocol"). Engagement-unproven was the correct call.
2. The identity-stamp hole (root-run git -> empty "built from:") is open in
   run_combined_window.sh - out of desk scope (file outside the repo), named
   in the spec above for the coordinator.
3. The engagement text prints once per llama_context construction: the count
   is 2 for the draft-mtp server (target + draft); any other serving shape
   (e.g. no spec) still prints 1 for the target ctx - grep accordingly.
4. ROCm 6.2 hip-link driver: a second -l library flag fails to resolve
   (reproduced minimal case); worked around with -Wl inputs in CI section 2b.
