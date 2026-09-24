# 77 — KVarN branch QA review, merge gates, and feedback (2026-08-26)

**Status:** REVIEW COMPLETE — main-side quality inspection of the KVarN work,
evidence-only (no server runs; all findings cross-checked against committed
results, live logs, and code). **Audience:** the agent implementing
docs/74 (KVarN attention work order, branch-local docs/72) — same agent that
did all the KVarN work — plus the integrator who lands `wo/kvarn-pp`.

**Scope reviewed:** `wo/kvarn-pp` @ 52af56e5 (31+ commits past main's
ee8f56ff base) + `wo/kvarn-layout-verify` @ fab93b97 (4 commits, main-based).
Docs reviewed: branch 66–72 (now docs/70–74 on main), layout 73–74 (now
docs/75–76), results/ evidence, verify battery report 20260826_083050,
decode-guard JSONs, serve provenance logs.

**Verdict: the work is high quality and merge-worthy once G-1…G-3 clear.**
The B2 regression incident and its handling are a model of the house's
measurement discipline. Findings below are ordered by merge-blockingness.

---

## 1. Numbering map (why the numbers moved)

Doc numbers are assigned on main at landing time. The branch allocated
69–72 locally while main had independently allocated 69 (mtp_adaptive) and 71
(branch review) — a collision, exactly the schema problem this doc also
addresses (§6). Landing map:

| branch number (file) | main number | landing status |
|---|---|---|
| wo/kvarn-pp 69 (kvarn_full_matrix_work_order) | **docs/70** | CLOSED |
| wo/kvarn-pp 70 (kvarn_prefix_reuse_work_order) | **docs/72** | CLOSED |
| wo/kvarn-pp 71 (direct_read_tc_prefill) | **docs/73** | CLOSED |
| wo/kvarn-pp 72 (kvarn_attention_optimization_work_order) | **docs/74** | LIVE (step 1 done, step 2 in flight) |
| wo/kvarn-layout-verify 73 (staged_layout_zero_copy) | **docs/75** | VERIFIED |
| wo/kvarn-layout-verify 74 (attention_path_work_scope) | **docs/76** | SCOPING |
| main 72 (project_magic_dictionary_planning, untracked) | **docs/98** | PLANNING (renumbered per user instruction to avoid the collision) |

Each landed doc carries a traceability banner. In-text references inside the
landed docs still use BRANCH numbers — do not "fix" them; the banners carry
the map.

## 2. What was verified as solid (do not re-litigate)

1. **D-19 wall fix (acd6f791)** — T≤6 over-wall routing to TC split-K.
   step0_report.md attribution is rigorous (pass-2 = 97.8% of attention;
   latency-bound, not bandwidth-bound; cliff scan root-caused the decode
   collapse). Live 39.4→62.8 / 35.1→60.6 t/s consistent with docs.
2. **B2 incident + fix (c289dd06)** — 18-point A/B decode matrix,
   root cause (semantically-neutral kernel-body edits flip MTP acceptance on
   restored requests), byte-identical legacy kernel separation for the
   default tier, provenance tooling (build_and_serve.sh) added as the
   process response. Review-verified: launcher dispatch (legacy_tier),
   decode-split revert, and hazard notes are all in place.
3. **docs/70 m2 GDN checkpoint** — code-reviewed: raw cudaMalloc outside the
   persistent arena (justified by the B2 arena-perturbation hazard), Tensor
   is a non-owning view (manual free correct, no double free), destructor
   leak fixed (b965b44a), restore edge cases traced (ckpt=0 sentinel,
   short prompts, per-slot staleness). Live: turn-2 ttft 47.2s→1.9s @32k,
   greedy bit-identical vs --no-prefix-reuse; 6-turn watch-item stable.
4. **docs/71 direct-read** — CLOSED with a design-space proof (register/smem
   traffic trilemma), kernel kept as gated capability. Verified: zero
   launcher references — NOT routed, as documented.
5. **Battery (20260826_083050, commit c289dd06)** — PASS: determinism ok,
   A2 identity ok, MTP 79.75 t/s @68.9%, pp 254.6 t/s @225 tok. Matches the
   commit claim exactly. Baseline re-saved (legitimate per tool design).
   NOTE: this battery does NOT include mtp_long — see G-1.
6. **docs/72 step 1 (52af56e5)** — correctly falsified the work order's own
   hypothesis (pass-1 is 2–6% at prefill shapes, ~40% of MTP-verify rounds
   at 40k) with an env-parameterized harness. This is the right loop:
   measure → refine the plan → proceed.
7. **layout-verify branch** — CPU-only proof tests; ran
   `test_kvarn_layout` standalone at QA: PASS.
8. **Process discipline** — per-step commits with live numbers + binary
   hashes, results committed alongside (measurement data is project data),
   honest CLOSEDs with rejected alternatives, hazard notes at the code.

## 3. Findings (merge-blocking first)

### G-1 (BLOCKING) — no full CI after the B2 fix
The last `run_ci.sh --full` run with evidence is 20260825_142740 — BEFORE
c289dd06. Since then the decode path (decode-split revert, materialize
legacy/_w split, launcher routing) changed. Battery (4k ctx) + 40k decode
guard pass, but the merge gates — **mtp_long_ok** (long-ctx MTP==plain
identity) and **T19 pp floor (≥450 t/s @60k–88k)** — have not been
re-measured on the post-B2 binary. Also: mtp_long was FAILing on main
already (pre-existing, G1 in docs/71's predecessor) — the post-merge full CI
must separate "branch regression" from "pre-existing main issue" explicitly
in the results.
**Action:** after the wall-raise commit lands (docs/74 step 2), run
`run_ci.sh --full` from the worktree, commit the report. Do not merge before
this is green (or explicitly triaged as pre-existing with main-side evidence).

### G-2 (BLOCKING) — tool-script merge conflicts; branch scripts are stale
Conflicting files at merge: `tools/ops/run_ci.sh`, `run_serve_tests.sh`,
`run_verify_tests.sh`, `tools/smoke/serve_correctness_ci.sh`,
`results/latest.json`. The branch versions predate main's provenance
hardening (8f828ef1/c30c99e7): they LACK the caller-tree FATAL guard
(run_ci.sh) and the exe==BIN server-identity check (serve_correctness_ci.sh)
and add env-var overrides (BUILDDIR/REPO/BASELINE/REPO_ROOT/BIN) that main
lacks. **Resolution recipe: take main's scripts as the base and graft the
branch's env-var overrides onto them** (keep the guards, add the
`${VAR:-default}` indirection + `export REPO_ROOT BUILDDIR` in
run_verify_tests.sh). `results/latest.json`/`latest_report.log`: discard the
stale branch copy (20260825_142740, pre-B2, FAIL); regenerate from the
post-merge full CI.

### G-3 (BLOCKING) — docs/50 ledger never updated on the branch
REPO.md rule: defects are ledgered in the same PR. The branch touched
docs/50 zero times. Required entries at merge:
- **D-18 → FIXED** (it is still OPEN on main): beyond-wall collapse fixed by
  D-19 wall fix (acd6f791) + split-K; verify numbers 39.4→62.8 / 35.1→60.6
  t/s, battery + 40k guard post-B2 (cite the G-1 run when it exists).
- **New line (closed): B2 hazard class** — semantically-neutral edits to
  legacy kernel bodies flip MTP acceptance on restored requests; fixed via
  byte-identical kernel separation (c289dd06); hazard note lives in the
  launcher. This class also covers the m2 arena-perturbation incident —
  record both, one line, with the build_and_serve.sh process response.
- **New line (OPEN): VRAM pre-check underestimates prefix-capacity cost** —
  98k passes pre-check then cudaMalloc OOMs (prefix_reuse_report.md);
  "sizing bug to fix before raising defaults." 48k is the working max today.

### G-4 (GUARD THE FLAG) — wide tiers accepted, decode path not width-generic
`--cache-type-k/var kvarn2..8` is parsed and prefill is width-generic
(materialize `_w`), but the decode path (T≤16: decode-split kernel) is
hardcoded 4/2-bit unpacking (the B2 fix deliberately reverted the
width-generic decode to restore byte-identity, but left the flag). A
non-default tier therefore produces SILENTLY WRONG tokens after the first
decode step — no crash, no log, no startup guard. This is the house's
favorite failure mode (silent corruption). **Action (choose, before merge):**
(a) startup FATAL/refusal for non-4/2 until decode is width-generic
(recommended), or (b) implement width-generic decode-split, or (c) mark the
flags experimental + loud warning at startup + docs/serving.md. Also update
`docs/serving.md` (G-5). The DType enum still reports KVARN_K4V2 for all
tiers (cosmetic; capacity accounting uses bits — verify the log line says
the actual bits).

### G-5 (docs gap) — serving.md not updated; status headers stale
docs/69 deliverable "updated docs/serving.md" never happened on the branch:
no `--cache-type-k/var` flag reference, no width ladder, no staged-wall
docs, no prefix-reuse window + VRAM cost (48k max, ~20 KB/token/rank,
73.4 MiB/rank GDN checkpoint buffer). Write it at merge.
Also: branch work orders that are done still say `Status: CURRENT`
(matrix, prefix-reuse; direct-read's header lags its own CLOSED body).
Mark them CLOSED when the branch merges (the landed copies on main carry the
landing status in their banners).

### G-6 (measurement caveat, non-blocking) — width ladder is a floor check, not KLD
The mission asked for KLD/perplexity per tier; the server exposes no
logprobs, so the ladder (docs/70 step E) uses greedy agreement vs BF16 —
disclosed in the doc, correctly framed as a floor. Consequence: all ten tiers
score identically (14/15 byte-identical) — the ladder proves no observable
regression, not quality differentiation. Fine to merge; do not cite the
ladder as "quality validated per tier" in the merge report.

### G-7 (in-flight, expected) — wall raise uncommitted at review time
`kKvarnStagedBudgetBytes` 1024→1344 MiB (docs/74 step 2) is a working-tree
edit (uncommitted) while the live server runs it. The in-capacity conversion
logic (attention_path_baseline.md §C) is sound; on commit: re-run startup
VRAM preflight at 250k (expect +320 MiB/rank; check the k8v8 96k fit note),
then it's covered by the G-1 full CI. Also re-read the header comment it
replaces — "DO NOT raise this without measuring" — the new comment cites the
measurement; good, keep it that way.

### G-8 (cosmetic) — results/ hygiene
18 one-off B2 A/B decode JSONs sit flat in results/. Measurement data is
project data — keep them, but consider a `results/b2_ab/` subfolder at merge
to keep the top level readable. `results/latest.json` is stale (see G-2).

### G-9 (context) — main-side open items the merge does NOT fix
mtp_long_ok (pre-existing), docs/69 mtp_adaptive ngram-mod port (unstarted,
main's own 69), docs/98 magic dictionary (phases 1–2 ready to spawn work
orders). The 74-agent's next phase (docs/74 steps 2–6) depends on the merge
gates, not on those.

## 4. Merge recipe (for the integrator) — EXECUTED 2026-08-26

Merged at **main `fe3d8857`** (2026-08-26). Actual sequence: G-1 full CI
landed on the branch first (20260826_095551 @ 98b5395e, green except
mtp_long) with the mtp_long triage committed (6f9396f0); wall raise was
already committed (98b5395e); then the merge ran with conflicts resolved:

- `run_verify_tests.sh`: **branch side won.** Main's c30c99e7 tree-relative
  refactor is BUGGY in this file: `REPO=$(cd "$(dirname ...)" && pwd -P)`
  resolves to `tools/ops`, not the repo root — build/baseline paths broken,
  which is why main's CI has been red. Branch's `SCRIPTS_DIR/../..` + env
  overrides (+ pp_warmup) is correct.
- `run_ci.sh`, `serve_correctness_ci.sh`: main side (caller-tree guard,
  exe==BIN provenance, correct derivation).
- `run_serve_tests.sh`: main side + branch's `${BIN:-}` override.
- `results/latest.json`: branch (generated; post-merge CI regenerates).
- Branch docs 69–72 dropped as content-identical duplicates of the
  already-landed 70/72/73/74 (banners carry the map).

G-3 (ledger), G-4 (flag guard), G-5 (serving.md) were NOT done before merge
(user decision: merge the verified improvement now, follow up under hold).
They are the hold list below.

## 5. Feedback to the docs/74 agent (you)

- The B2 handling is the best incident work in this repo's history: matrix
  of A/B builds, root cause, minimal semantic fix, process tooling as the
  response, and the hazard written where the code lives. Keep doing exactly
  that.
- Step 1 (attribution) did the right thing: it falsified the work order's
  assumption with numbers before any code. The step-2 implication (wall
  raise → O(delta) rounds, zero kernel changes) follows correctly from it.
  Ship it as a commit with the preflight line, then the full CI.
- The docs/76 ordering (C1→C2→C3→C5→C6→C4, kernel rewrite last) is correct;
  don't start C4 before C5 is measured.
- Process asks, in priority order: (1) G-1 full CI; (2) G-3 ledger lines;
  (3) G-4 flag guard — a silently-wrong-output flag is the one thing this
  house cannot ship; (4) G-5 serving.md; (5) status headers.
- One style note: commit messages are excellent (numbered, measured,
  root-caused). Keep them. The only recurring miss is the ledger + serving
  docs lagging the code by a whole branch — house rule says same PR.

## 6. The numbering schema (fix for next time)

The collision happened because "next available number" was evaluated on a
diverged branch. Proposed schema (needs user sign-off before REPO.md edit):

1. **Numbers are assigned on main, at merge/landing time** — by the
   integrator, not the branch author. Branch authors still pick a number
   locally (it's a working number), but it is PROVISIONAL.
2. **Landing rules:** on merge, the integrator renumbers any incoming doc
   that collides with main's allocation, updates the title + adds the
   traceability banner (original file/branch/commit), and does NOT rewrite
   in-text references (the banner carries the map; docs stay immutable
   snapshots).
3. **Collision-zone convention:** branch authors start new docs at
   `main's max + 2` and leave a gap; anything in the gap is marked
   `PROVISIONAL` in the status line. (In this case: main was at 71, the
   branch started at 66 because it branched earlier — the rule only helps
   going forward, which is its whole job.)
4. **Renumbered-landing example (this event):** 69→70, 70→72, 71→73, 72→74,
   73→75, 74→76, 72(magic-dict)→98 — all with banners; zero in-text edits.
5. Alternative considered and rejected: per-branch number namespaces
   (`pp-69`) — breaks the existing doc corpus and every tool that greps
   `docs/\d+`.

## 7. Review method (for the record)

No server runs, no builds beyond a dry `make -n` (the live build dir belongs
to the running server; the layout-verify CPU test was compiled to /tmp and
ran standalone). All claims cross-checked against: committed results/ JSONs
+ reports, /home/intel/verify_logs/20260826_083050 (the battery the agent
ran), /home/intel/logs/decode_guard_* + serve_provenance.log (live), git
diffs main..wo/kvarn-pp, and merge-tree conflict simulation.

## 8. MERGE HOLD — remaining items (added 2026-08-26 after merge fe3d8857)

**HOLD IN EFFECT: no further merges into main until every item below is
complete.** REPO.md §4a carries the same marker. Update this list as items
clear (strike through + date + commit); when empty, delete REPO.md §4a.

- [x] **H1 (docs/50 ledger)** — add: D-18 → FIXED (D-19 wall fix, acd6f791,
  verified 62.8/60.6 t/s + full CI 20260826_095551); new closed line: B2
  hazard class (semantically-neutral kernel-body edits flip MTP acceptance
  on restored requests; c289dd06; build_and_serve.sh process response);
  new OPEN line: VRAM pre-check underestimates prefix-capacity cost
  (98k OOM after pre-check pass; 48k working max). ~~**DONE 2026-08-26, commit `a3e53f07`**~~
- [x] **H2 (wide-tier flag guard)** — `--cache-type-k/var kvarn2..8` accepted
  but decode path (T≤16) is hardcoded 4/2-bit: non-default tiers emit
  silently wrong tokens. Startup FATAL (recommended) or width-generic
  decode, + docs/serving.md note. G-4. ~~**DONE 2026-08-26, commit `a3e53f07` — startup FATAL for kb!=4||vb!=2; test + smoke verified**~~
- [x] **H3 (docs/serving.md)** — new flags, width ladder, staged wall
  (1344 MiB, 40.4k-token wall), prefix-reuse window + VRAM cost (48k max,
  73.4 MiB/rank GDN buffers). ~~**DONE 2026-08-26, commit `a3e53f07`**~~
- [ ] **H0' (USER DIRECTIVE 2026-08-26 — wall removal via docs/78, NOT doc 74
  step 3)** — **Do NOT start doc 74 step 3 / C5 (persistent BF16 overflow).
  It is DELETED.** The plan is now **docs/78**: remove the staged shadow
  (the "wall") entirely, KV residency = packed pool only, and build the
  packed-only decode kernel (C4) as the critical path. Rationale: step 3
  converts KVarN into a fill-up Q5 (BF16) cache (the opposite of the value
  prop; ~22.5 GB/rank @250k, doesn't fit 16 GB). docs/78 §2 proves the
  decode side is the only make-or-break (packed-only decode is DRAM-cheaper
  than int8; the 4.0 t/s beyond-wall is the materialize path, not the floor)
  and the prefill side is already correct (materialize-once, beyond-wall
  484 tok/s). Execute docs/78 steps 0–4 in order; G1 (≥150 GB/s effective
  packed-code decode @40k/250k) is the make-or-break gate — miss it and
  follow docs/78 Rollback, do not ship step 3. Owner: agent. **PROGRESS
  (2026-08-26): docs/78 step 0 baseline DONE** (commits `82009788` + `ad0d5fba`
  — `tests/bench_kvarn_attention.cu`, `results/doc78_step0_baseline.md`, live
  in-wall decode_guard ~71 t/s @40k; measured ~8.5 GB/s effective packed-code
  read → **G1 gap ~17.7×**). Next: docs/78 step 1 (packed-only decode kernel,
  G1 ≥150 GB/s).
- [x] **H4 (post-merge full CI)** — `bash tools/ops/run_ci.sh --full` from
  the MERGED tree (validates the resolved scripts too — esp. run_verify_tests.sh
  taken from branch). Commit the report. Expect: green except mtp_long
  (pre-existing). ~~**DONE 2026-08-26 — GREEN except mtp_long → see H5/D-21. Evidence `results/20260826_110519_ci.json`, commit `a91801ee`**~~

  **Run 20260826_110519 (merged tree, a3e53f07 build):** battery PASS —
  determinism/A2-identity/sampling det+div/int8-KV/prefix identity+skip ALL
  true; serve S1–S5 pass; correctness fast gate + int8 full (T1–T12) + KVarN
  @250k (T1–T19 incl T16/T17/T18, T19 ≥450) + KVarN T14 tolerance all pass.
  pp 260.3 t/s, plain 35.4, MTP 80.71 @82.0%. Sole red = `verify.mtp_long_ok`
  (→ H5: real bug, docs/50 D-21). Top-level `_fails=[]`; serve+correctness
  all_pass True.
- [x] **H5 (mtp_long re-test)** — re-run long-prefill MTP-vs-plain on a
  NON-degenerate (non-repetitive, clean-boundary) >512-token prompt. If
  identical → gate prompt is the problem (relax/replace the gate input).
  If it diverges → real numerical divergence in the MTP chunked-prefill
  path → new defect line in docs/50 (would change the "pre-existing
  flaky gate" triage). ~~**DONE 2026-08-26 — DIVERGED → real bug, docs/50 D-21**~~

  **Result: real numerical divergence, NOT the degenerate-gate artifact.**
  Clean bracketing (tools/ops/h5_retest_mtp_long.sh, clean space-joined
  non-degenerate prompts): 250/509/655/751/844/937/987/1005/1015 tokens ALL
  MATCH (MTP==Plain exact); **only 1024 tokens DIVERGES** at the first
  generated token (word 814: plain `astronomer`, MTP `farmer`). Both paths
  individually deterministic (plain A/B exact, MTP A/B exact). 1024 = 2 full
  512-token chunks = 16×64 pages — an exact chunk/page-aligned boundary.
  **Supersedes the earlier "degenerate near-tie flaky gate" triage** (that
  was the 1793-tok repetitive prompt); the real signature is first-token
  divergence at the exact 1024 boundary. → **docs/50 D-21** (new OPEN line).
- [x] **H6 (k8v8 fit) — RE-SCOPED (2026-08-26)** — original: re-check k8v8
  @96k preflight after the wall raise (was 15969 MiB; +320 MiB ≈ 16289 vs
  16310 usable — ~21 MiB headroom, never measured in CI).
  ~~**DONE 2026-08-26 — re-scoped; real k4v2 250k config FITS (H4 measured)**~~

  **Why re-scoped:** two merged/committed facts make the original premise
  obsolete. (1) H2 startup-FATALs any non-4/2 width for SERVING (decode path
  is 4/2-bit specific), so k8v8 is no longer a servable config — the server
  correctly rejects it, so there is no k8v8 @96k server to measure. (2) docs/78
  (wall removal) deletes the BF16 mirror, which *relaxes* VRAM — so the
  '+320 MiB wall-raise headroom' concern dissolves in the direction of the
  goal. **Do not re-litigate k8v8 serving; it is correctly rejected by H2.**
  **Re-scoped + RESULT:** confirm the REAL served KVarN config (k4v2, 250k/250k,
  MTP k3) still passes startup after the +320 MiB wall-raise. **Measured (H4
  run 20260826_110519): the server BOOTED on both ranks and ran the full
  battery — 16 passed / 0 failed / 2 skipped — with 'materialized: 9059 MB
  (capacity)' per rank (vs 16310 MiB usable).** The real config fits with huge
  headroom. docs/78 step 3 (delete the wall) additionally frees 1344 MiB/rank,
  so k8v8 @96k resolves to comfortable — re-check and record as part of docs/78.
- [x] **H7 (layout-verify branch)** — merge `wo/kvarn-layout-verify`
  (tests/test_kvarn_layout.cpp + budget/codec tests; backs docs/75–76).
  Its 4 commits are main-based; should merge cleanly. ~~**DONE 2026-08-26, commit `b2a18c39` — test payload only**~~

  **Note (landed as test payload, not the branch):** recon showed the
  branch's 4 commits are 2 doc commits (docs/73/74) whose content ALREADY
  landed on main as renumbered docs/75/76 (banners + content identical)
  plus 3 test commits (test_kvarn_layout/budget/codec_edge + CMakeLists).
  Merging the whole branch would collide with main's 75/76, so only the
  3 CPU-only tests + CMakeLists wiring were landed (all pass, exit 0).
  Doc content is fully preserved as main's 75/76; nothing lost.
- [x] **H8 (status headers)** — landed work orders now on main: mark
  70/72/73 status lines CLOSED (they still say CURRENT; banners carry the
  landing status). 74 stays CURRENT (work continues). ~~**DONE 2026-08-26, commit `a3e53f07`**~~
- [x] **H9 (housekeeping)** — prune stale worktrees wo-kvarn, wo-kvarn-d18,
  wo-kvarn-live (after their branches are merged/confirmed dead); consider
  `results/b2_ab/` subfolder for the 18 one-off decode JSONs.
  ~~**DONE 2026-08-26 — 3 worktrees pruned + 3 branches deleted; b2_ab absent**~~

  **Result:** `wo-kvarn` (03861e10), `wo-kvarn-d18` (879bc70d), `wo-kvarn-live`
  (6b535c76) were all confirmed merged into main (`git branch --merged main`;
  879bc70d is an ancestor of main; d18 has 0 commits not-in-main) and git had
  marked them `prunable`. Removed all 3 worktrees + deleted all 3 local branches.
  `results/b2_ab/` no longer exists (already cleaned; only `results/reports/`
  subfolder remains) — nothing to do there. Remaining worktrees kept:
  `wo-kvarn-hold` (this), `wo-kvarn-pp`, `wo-kvarn-attn-verify`,
  `wo-kvarn-layout-verify` (used for H7 recon; not requested in H9).

Ordering suggestion: **H0' (docs/78) is the main code track** — start docs/78
step 0 (baseline); it supersedes C5/step 3 and is the next code work after
H1/H2/H3. Then H4 (validates the merge), H1/H2/H3 (ledger + guard + docs),
H5 with H4's server, H6 after docs/78 step 3 (deleting the shadow frees
1344 MiB/rank, so k8v8 @96k fit resolves automatically — re-check and record),
H7/H8/H9 anytime.

---

> **H10 note (superseded 2026-08-26):** an earlier inline "wall-removal plan"
> was drafted as H10 before docs/78 existed. It is now SUPERSEDED by the
> canonical **docs/78** work order + this H0' item, which carry the full plan
> (gates G1–G4, Rollback, constraints, def-of-done). H10's prose is dropped
> to avoid duplication — see docs/78 for the authoritative plan, and docs/77
> H0' for the directive marker.
