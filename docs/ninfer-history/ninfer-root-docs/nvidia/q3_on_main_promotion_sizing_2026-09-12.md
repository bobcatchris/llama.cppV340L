# Q3-on-main full promotion — paper sizing (agent3, 2026-09-12, CPU-only, coord seq-16 ask)

**Trigger:** the seq-11 smoke proved main cannot load Q3 artifacts (`unknown tensor format:
Q3G64_F16S`, reader.cpp:100 — grep: ZERO `Q3G64` refs in the promote tree). The merged attr-raise
pair (e08d6b83) was launcher-only. This doc sizes the FULL loader promotion. **Nothing here was
built or run on a card — every figure is git-measured or cites a named precedent.**

## 0. Headline numbers (measured against `github/main@e08d6b83`, lane tip a80a2725)
- **46 files, +3,779 / −129** (`git diff --shortstat github/main HEAD -- src tools tests` after
  excluding the two attr-pair files, which are already merged = 0-diff).
- **30 NEW files, 16 MODIFIED** — of the modified ones, exactly **2 conflict-surface files**:
  `src/CMakeLists.txt` and `tests/CMakeLists.txt` (both-sides-modified: main moved them during the
  dflash2/phase-gate merges — the ONLY expected manual resolutions).
- 18 lane code commits underlie it, interleaved with superseded intermediate states (the
  a74003c4 chunk-plane fix corrects 98d62327/27f66fef; the 62db738d→75152c45 Stages saga is ALREADY
  in main). **Recommendation: do NOT cherry-pick 18 commits; land 5 dependency-ordered squashed
  commits** (below). The card-proven tree to promote = lane tip blobs; the per-commit history is
  not the contract.

## 1. Dependency-ordered cherry-pick set (5 commits, each independently compile-green-able)
1. **+core/artifact (5 files):** `src/core/tensor.h` (Q3G64_F16S/Q2G64_F16S QType entries),
   `src/artifact/reader.{h,cpp}`, `storage_layouts.cpp`, `typed_binding.cpp` (NumericFormat→QType
   dispatch lanes 23–24/50–53). Pure additive enum + switch arms; zero behavior change to existing
   formats. Throw sites stay loud (unknown-format throw is what caught the smoke — keep it).
2. **+ops kernels (13 files):** `src/ops/linear/{q3,q2}/` (5 headers: storage, gemv ×2, gemm_simt),
   `src/ops/linear_add/{q3,q2}/` (8 files: gemv .cu + plan .cpp/.h + kernels.h). Header-mostly —
   only **4 new nvcc units** join `ninfer_ops` (q3/q2 linear_add .cu + plan .cpp ×2).
   `q3_decode_one` single-source-of-truth pattern preserved (host+device, CPU tests share it).
3. **+dispatch wiring (6 files):** `src/ops/linear/linear.cpp` (+11: Q3/Q2 cases + workspace
   returns 0), `src/ops/wrapper/{linear_add,embedding}.cpp` (T>1 routing + Q3 gather arm),
   `src/ops/kernel/embed_gather.cuh`, `src/ops/launcher/embed_gather.{cu,h}` (Q3 arm, dense
   24 B/group no-high — the file Team Red's blocker-list was missing).
4. **+target/loader profile (1 file):** `src/targets/qwen3_6_27b/impl/package.cpp` (87cf236e:
   format-flexible Qwen38GroupwiseInt arm accepting groupwise-q3 identity). Last code commit —
   until this lands, nothing can NAME the artifact, so 1–3 are inert for existing models.
5. **+registration/tests/tools (21 files):** `src/CMakeLists.txt` (+4 sources — **CONFLICT SITE A**,
   strict-union with main's dflash2-era hunks near :278/:344), `tests/CMakeLists.txt` (**CONFLICT
   SITE B**, +12 targets + `ninfer_add_linear_test` pattern), `tests/ops/linear/` {q3,q2} a16 +
   cpu twins + measure, `tests/ops/quant*` (3), `linear_test_common.{h,cpp}`, `quantized_weight.h`,
   `tools/artifact/{layouts,numeric}.py`, `tools/convert/qwen3_8_27b/` (5 py) + `roundtrip_q3q2.py`.

## 2. Build-time estimate (precedent-cited, target-only law honored)
- `--target ninfer-serve` on merged tree: measured 07:31→07:38 **full cold config+build = ~7 min**
  this morning on the promote tree (296 edges, bin 361 MB). +4 library units ⇒ **+2–4 min**
  (nvcc unit ~30–60 s class from lane build logs). Expect **~10 min**.
- `ninfer_linear_q3_a16_test`/`q2_a16_test` relink-only: ~4 min precedent (runbook: "relink ~4 min").
- **Whole-tests build NOT during any GPU window** (build-size law): ~92+12 binaries × ~230 MB ≈
  **~24 G class** (precedents: 21 G measured 92-bin; step-2a self-abort at 27 G; 15 G/54-bin era
  count in docs/145). Farm/release-window only.

## 3. Disk math vs 27 G free (measured `df` 07:44: 27 G avail, /dev/sda2)
- Commits 1–4 target-only dev loop: serve build tree ≈ 2.2 G (lane build measured) + ~0.3 G delta
  ⇒ **safe at any time**.
- Conflicts: `build/tests` reclaim precedent exists (coord-stamped once); **do not run whole-tests
  while df < 25 G** (G7 rule, habitat row: "23G<25G" blocked a gate). 27 G today = tests build
  possible ONLY in released-window time, re-measure df at fire.
- Artifact: 14.39 G exists on /media/intel (210 G free, outside /) + 14.39 G local worktree copy —
  no / pressure from data.

## 4. CI cells the promotion MUST carry (each named, none invented)
1. **Zero-GPU battery:** `ninfer_linear_q3_a16_cpu_test`, `q2_a16_cpu_test` (shared `q3_decode_one`
   oracle — runs anywhere), `ninfer_quant_recipe_test`, `iq3_recipe_map.py` unit. These are the
   farm-grade numeric gate; device twins below must NOT be the only proof.
2. **ctest device class:** `ninfer_linear_q3_a16_test` / `q2_a16_test` need artifact + card ⇒
   **CTEST_EXCLUDE 9→11** in `run_ci.sh` — the exact merge-exposure class that burned D5 run 1
   (2a batched test registered, excluded-list not unioned). Register in the exclude list IN THE
   SAME commit as the CMakeLists registration, verified by behaviour (ctest counts must match
   prediction, not by eyeball).
3. **step-0 anti-resurrection:** `git diff main -- src/runtime/tp2/tp_engine.cpp
   src/runtime/tp2/tp2_budget.h` 0-diff gate for the merged tree — promotion touches none of those
   files, so this should pass trivially; it must still RUN (law).
4. **contract_lint:** must pass with the 12 new test registrations (cells satisfiable, exit
   contracts present) — cheap, CPU, run pre-merge.
5. **Serve smoke (needs grant, 1 card, ~5 min):** THIS is the cell that becomes possible only
   AFTER this promotion — Q3 artifact on the merged main tree, Tokyo/1161 rows. Until it lands,
   any "Q3 on main" claim is load-path-unverified. Recommend the promotion NOT be declared done
   without one such row (geometry = lane GATE RESOLVED row, 09-12 07:02, as the parity target).

## 5. Risks & non-goals
- **Behavior change to existing models: none intended.** All changes are new-enum/new-arm additive;
  the two switch-defaults (reader throw, tp_gemv throw) must stay loud. Reviewer test: a Q4/Q5/W8
  artifact must load + serve byte-identically post-promotion (reference TP2 parity row exists as
  the check; can run in any future window).
- **VRAM-LAW compliance:** the loader adds NO budget constants (preflight untouched) — verified by
  the file list (no runtime/tp2 paths).
- **NOT in this promotion (separate agenda):** converter imatrix/v2 quality (plan §v2 levers),
  Q2 codebook adoption (RTN Q2 = 0.76–0.78 cos, would FAIL the ≥0.999 gate — the format ships as
  kernels+tests, tier map keeps ZERO Q2 — say so in the commit msg to prevent the tier-map mistake
  Team Red made, in reverse), fused-family Q3 arms (+1–2 d/family), TP2 Q3 dispatch arms
  (WO Phase 2, in flight on this lane).
- **Effort total on paper:** cherry-pick squash + 2 CMake unions + lint/exclude wiring ≈ **0.5 day
  CPU**, + the 5-min serve smoke in a granted window. The long pole is window latency, not work.

— agent3, sized 2026-09-12 post-smoke; all measurements re-derivable via the exact commands cited

## RE-MEASUREMENT at EXECUTION (guardrail-5 compliance, 2026-09-12 ~12:4x)
- FINAL anchor: main@2eec7b52 (moved twice during execution: be970ead -> d5868323 -> 2eec7b52, each
  re-verified; ladder rebased onto each). Original sizing vs e08d6b83 = 46 files +3,779/-129.
- EXECUTED set vs 2eec7b52: commits 1-4 (src) = 27 files (13 pure-add new + 14 modified — all 14
  MAIN-UNMOVED since 149c92d9, blob-hash class in each commit msg, A3 J1 cold-verified UNMOVED
  claims; count corrected from an earlier draft's 24=17+7 per A3 seq-11 cold recount — A3's
  13A+14M reproduces from git diff --name-status 2eec7b52..c4); commit 5 (tests/tools/docs)
  = 20 files +2,380/-12. NOTE 2: this anchor line predates the merge-of-main step — 2eec7b52 is on
  gemini's wo/q3-ci-hardening (NOT on main@d5868323 — sibling chains off be970ead); the ladder
  carries BOTH sides and merge-of-main resolves CROSSLANE keep-both + contract_lint per coord
  standing note; re-state final counts at merge tip. Conflict sites actual: contract_lint.sh ONLY (q3 cells RENUMBERED 30/31->31/32
  per A3 row-2 collision find; gemini's own renumber 2eec7b52 absorbed, main-wins); CROSSLANE.md
  (trivial union); the sized '2 CMakeLists conflicts' resolved CLEAN (--3way, both pure-additive vs
  actual main state — sizing's conflict prediction was over-pessimistic, honest row).
- Corrections absorbed mid-flight: CTEST_EXCLUDE 9->12 (coord seq-28 ruling, gemini count supersedes
  sizing's 9->11); D1 vacuous test_multitoken REWRITTEN as SIMT host-twin (non-vacuity PROVEN by
  stride-mutation negative: 3.75e-03 RED at 5120x6144); D5 measure-test bar 0.90 -> 0.99 with cited
  anchors; D2 goldens registered IN commit 5 (were dead bytes after cherry-pick — A3 J2 catch);
  sizing doc's own 'commit-2 registration' placement FIXED per A3 BLOCKING row (src/CMakeLists +4
  moved from commit 5 into commit 2 via autosquash fixup — same-commit rule applies to ladder too).

## LANDING ROW (2026-09-12 13:0xZ): MERGED. github/main = ebf08aaa (ls-remote verified).
Only open item in the plan: Stage-2 post-land serve smoke (1-card, ~8 min, binary md5 7af40c22…
= tip-faithful, zero src delta between link-proof commit and tip) — pending coord grant.
