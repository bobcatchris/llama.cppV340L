# A2 → A1(fresh) — MERGE PACKAGE STATE for run_ci --full orchestration (C441, 2026-09-11 ~19:30Z)

Companion to drafts/172 (coord's orchestration brief). This doc = the merge-package
inventory + the run-1/run-2 flag coord told me to settle with you. All facts verified
this session unless marked [ASSUME]. Read the cited commits before trusting my prose
(rule-8 applies to handoffs too).

## 1. WHAT SHIPS FROM HABITAT (`167-piecewise-fix`, tip 28513333, 447 ahead of main 149c92d9)

**Everything src-side.** The dflash2 single-seq (twin) Option-B line end-to-end:

| area | commits | what |
|---|---|---|
| G1 fix stack | `e481da74` route fix v2 · `557df6f0` httplib ROOT patch · `28513333` 9th/10th-int P6 (sstop/creq/leg/tdec) · `b0f66851` v1 (superseded in place by v2) | cancel-path exit symmetricized (bar-synchronized single post-bar break) + vendored `is_socket_alive` EAGAIN misfire patch. finish_reason leg-aware. |
| PG taps | `41085447` (+`c414f06d` 9th-int, `28513333` leg split) | P0..P7 cross-rank FNV rows, env `NINFER_D2_PHASEGATE`, zero-cost-unset, step<=16 on big D2Hs. Format PARSE-VERIFIED against gemini's parser pre-boot. |
| five-layer chain (earlier tonight) | `9a76439e` fall-through return · `dc92bddf` rank-0-only stats · `4045bfd4` fusion feed second landing · `fcd045ce` one-shot reset parity | the G1-closure precursors; all in tip. |
| harness | `results/bcode_m2/p1n1.sh` (`a45bf032`+`7dc07862`+`4c79ce15`) | N-in-1, ARM=ON/OFF env -u semantics, arms-proof abort both directions, freeze-SOP. |
| evidence | `results/bcode_m2/p1n1_on/` + `p1n1_verify/` era bundles | hang capture + VERDICT + rider + methodology bullets; verify boot 6/6 (acc 0.400, 11.15s walls, BIN 5167a63e). |
| debriefs | docs/A2_DEBRIEF_2026-09-11_1636Z.md + this era's commits | lane history. |

**Anti-resurrection (AGENTS law) — re-verify yourself:** `git diff main -- src/runtime/tp2/tp2_budget.h` = **0 lines**; `git diff main -- src/runtime/tp2/tp_engine.cpp` = +106/−2, ALL dflash routing + §22.54 label fix + the leg-probe wiring — ZERO budget-region lines, no estimate-refusal anywhere. (My e481da74 message said "untouched" — wrong wording, corrected in-flight; the region check is what the law names.)

## 2. WHAT'S IN GEMINI (`wo/dflash2-phase-gate` @ cf1dd723, based on main)

**Pure tooling — zero src changes** (verified: `git diff 149c92d9..cf1dd723 -- src/` empty; +1012 lines/9 files, only the `include/ninfer/runtime/phasegate.h` header + `tests/CMakeLists.txt` touch outside tools/docs):
- docs/171 spec · `tools/ops/phasegate_report.py` (BG_DIV filter for P5 = A1-N1 wired) · `tools/ops/p1n1_oracle.py` (warmup skip cf1dd723, SERVE_LAYER_CANCEL predicate, under-instrumentation rule) · `tools/ops/p1n1_oracle` + report = THE verdict tools
- `tools/bench/dflash2_single_seq_ci.sh` — the G1 regression fence: boots `--spec dflash2`, N=5 m2b_10k_a, bars 200/finish=stop/zero-hang/acc∈[0.20,0.50], PG cross-rank check when armed
- `tests/ops/test_accept_greedy_determinism.cpp` — the battery that (co-)killed class (B); runs against main's ops too (op predates twin)
- wiring: run_ci.sh +6 (inserts their cell after the decode_guard block), contract_lint.sh +8

## 3. THE run-1-vs-run-2 FLAG (coord told us to settle; my recommendation)

**Their twin-route cell REQUIRES the habitat merge first** — the batch-of-one twin route does not exist on main; their cell would boot the §22.6 forbidden fallback. So ordering is forced: habitat rides, then gemini tooling.

**RUN-1 (my recommendation): include their branch, AFTER habitat.** Rationale: (a) their cell IS the G1 fence farm7 was missing; (b) their oracle+parser are what makes the merged PG rows machine-readable; (c) the merge surface is two small conflict sites (§4); (d) BONUS that decides it for me: **run-1 on the merged tree boots the twin with the 10th-int BIN → the leg= arbiter data (natural-stop leg=2 vs residual-misfire leg=1) lands FOR FREE in the CI evidence**, closing the p1n1_on era wording question without a dedicated boot. If you cold-read the conflicts and they fight back, the fallback is honest: `--full` runs on habitat's own fences, gemini's cell+tooling ride run-2 (coord's brief already blesses this).

**Consult gemini (not assign — user order):** they know whether their cell assumes any of their own unmerged state. My read says no (it boots habitat's twin + reads habitat's PG format), but the cell's `P1N1_SCRIPT` lookup falls back to `tools/bench/p1n1.sh` which does NOT exist in either branch yet — the cell must run with `P1N1_SCRIPT` pointing at habitat's `results/bcode_m2/p1n1.sh` or we add the copy as a merge-session item. [verify with them]

## 3b. CRITICAL COLD-READ FINDING (post-handoff, 19:4xZ) — gemini's CI cell is VACUOUS AS COMMITTED
`tools/bench/dflash2_single_seq_ci.sh` @ cf1dd723: BIN is checked but never launched; REPEATS only echoed; `P1N1_SCRIPT` resolved (:45-49) but NEVER EXECUTED; the oracle runs against a `SERVE_LOG` (:35) that NOTHING creates ⇒ the `if [ -f "$SERVE_LOG" ]` is false ⇒ falls to :66-68 'Dry-run/contract pass' **exit 0**. THE CELL PASSES UNCONDITIONALLY TONIGHT — the vacuous-contract class CORD's ledger names. run-1 inclusion verdict AMENDED: their TOOLING (oracle/parser/tests/spec) yes as argued; the CELL may only ride run-1 AFTER gemini completes it (3-line shape: `ARM=ON P1N1_N=$REPEATS P1N1_PORT=$PORT P1N1_OUT=<tmpdir> bash "$P1N1_SCRIPT" ci_$TS` then point SERVE_LOG at `<tmpdir>/ci_$TS/serve.log`). A1 ruled the harness copy as tools/bench/p1n1.sh (my default accepted); CANONICAL COPY NOW LANDED habitat-side: `tools/bench/p1n1.sh` @ this commit, provenance header + CI env interface (P1N1_WT/P1N1_OUT/P1N1_N/ARM/P1N1_PORT/P1_BIN). Their fallback :49 resolves AS-IS on the merged tree. A permanent-green cell stays NOT-ACCEPTED.

## 4. CONFLICT SITES (mechanical, expect ~10 min)
- `tools/ops/run_ci.sh`: habitat +190/−40 vs main (capbuffer #6/#6b, decode_guard v3.1, step-0 parity, guard-v4 rows) vs gemini +6 inserting INTO the same region. Resolve: habitat's version as base; re-drop their 6 lines after the decode_guard block; keep their `vram_fence "dflash2 single-seq cell"` call (it's the law-compliant fence).
- `tools/ops/contract_lint.sh`: gemini +8 cell registration vs habitat drift [check both tips; resolve same way].
- `tests/CMakeLists.txt`: gemini +3 only — clean.

## 5. PRE-FLIGHT FACTS FOR THE 90-MIN WINDOW

- **Disk NOW: 25G** (A3 reclaim mid-flight; coord target ≥35G BEFORE the merge build). Watch the row; farm7 floor 25G, merge build needs headroom.
- Cooperative tax: cold tests-relink ≈ **4 min measured** (this session re-measured) — budget it into the window, once.
- Staged era BINs (disk-only, 236MB each — DO NOT git-add): `results/staged_bins/{5167a63e…,c31ae347…}_serve` are the keep-two (G1-verify BIN + 10th-int head). Older ones (`ec1b3224…`, `5f2d8e91…`, `3cc5bdb3…`, `a804d598…`, `509300c7…`) are md5-cited in commits and safe to prune if disk bites — ask coord, don't scatter-delete.
- VRAM LAW: any CI cell that refuses a launch on a BUDGET CONSTANT is a protocol violation — report measured, reject as bug. Synthetic near-capacity cells must LAUNCH.
- Merge pattern: night-merge NOT-FF, tp2_backend.cpp is BOTH-SIDES-MODIFIED vs main (per-164: no verbatim checkouts), execution = A2 drives locally WITH coord (your 172 §3: orchestrate the window, don't merge alone).
- GPU: both cards, 0 apps now; coord issues the written window grant; guard foreign CUDA contexts not ports; gpu_guard at absolute path.

## 6. WHAT "GREEN" MUST SHOW (era-correct bars, per coord rulings)

1. step-0 parity cells loud-first (budget-header 0-diff, region-scoped).
2. Their dflash2 cell (if run-1): 5/5 finish=stop, zero hangs, acc∈band (merged-tree behavior measured 6/6 acc .400 — band passes).
3. PG leg= rows from the merged BIN: the arbiter. leg=2 everywhere ⇒ trigger = natural policy stop, httplib patch fully vindicated, era closes. ANY leg=1 mid-decode ⇒ residual transport misfire — still G1-safe (exit is symmetric) but the WO list grows one live item.
4. finish_reason label-leak WO rides the 169 §D follow-ups (serve-layer accuracy, not a blocker).

A2 availability: I'm the driver for the local merge execution when you call it, and
anything habitat-side is ask-me-not-rederive. — A2
