# 117 — Pipeline board (2026-08-29, schedule-slip recovery)

**Status:** LIVE — plan owner maintains. One row per work item; the GPU column
is the single serial resource (queue order below). Read this first each morning.

## 1. Work items

| Item | Doc | Branch / owner | State (13:15) |
|---|---|---|---|
| GDN T=1 perf fix | 105, 112, 103, **119** | wo-kvarn-hold / Agent 1 | **DONE (ca3fe4aa + d03b78d3 + d029a07a).** Option B' (GEMV unification T≤8, shared per-column code): D-21 identity ALL PASS, plain 35.43 restored, verify 35.71 / mtp 78.77 = D-21-safe ceiling (split-K structurally forbidden by bit-identity). **Re-baseline v2 APPROVED** (v1 pre-dated the D-21 fix and contained the bug): targets plain ≥35.0 / verify ≤35.8 / mtp ≥78.5. pp + prefix-WARN exonerated (outlier/variance, results/103 §9-10). |
| Baseline v1 + guard tooling | 104 | wo-kvarn-hold / plan owner | DONE (819c3fd2, 1527ff88, cf3ea4ca, 4498f0b1, b60ce3d1) |
| Phase 2 kvarn decode gap | 115 | TBD / free agent? | READY (5bfd7071) — needs owner + GPU slot (tomorrow) |
| KV uniformity Phase 1 (launch config table) | 106, 114 | wo-kv-uniform / plan owner | Landed + reviewed (PASS) + CPU sprint DONE (rebase clean, build green, host test rc=0) — GPU window pending |
| Phase 3 prefill fix (direct read, kill materialize temp) | 113 | wo-kv-uniform / plan owner | DRAFT (6a9c1581, self-reviewed bd2f4144) — plan owner will execute in Phase 3 slot |
| Phase 4 prefix-restore unification | 116 | wo-kv-uniform / plan owner | DRAFT (1e1122ba) — after Phase 3 |
| mtp-adaptive land & verify | 107, 80 | wo-mtp-adaptive / **plan owner (agent absent)** | REBASED onto 02929a90 (13 conflicts resolved, D-21 anchor verbatim, 0 markers) — **BUILD ✓ (CUDA 13.1), HOST TESTS ✓** (ninfer_mtp_adaptive_test all OK, ninfer_serve_options_test ok) → GPU slot queued behind Agent 1 Phase 3 → re-rebase onto final hold head before merge |
| magic-dict land & verify | 108, 100 | wo-magic-dict / **plan owner (agent absent)** | REBASED onto 02929a90 (clean) — building (CUDA 13.1, after stale-cache wipe) → host tests (test_vocab_output_counter) → GPU slot behind mtp-adaptive |
| Auto-reviewer (headless pi) | — | plan owner | **FIXED** (pinned to desktop-27b; b.ai default hangs) — fires on next agent commit |
| Disk space | — | plan owner | 21G freed (repo/build) → 28G free (88%) |

## 2. GPU queue (single resource; updated 16:40 — HANDOFF: Agent 1 = GPU lane,
##    plan owner = all CPU per user decision)

1. **15:15-15:25** — plan owner spot check: **PASS** — kvarn @40k 66.6 (vs
   baseline v1 64.0, +4.1%; the old ≥69 target was a wall-era number, NOT a
   valid ref — user-confirmed; B' recovered kvarn_k4v2 to int8 parity 66.4),
   @250k 44.8 (gate ≥30, above baseline 43.5).
   results/spotcheck_wip_greedy_decode_20260829_151526.json.
2. **15:59→~17:20** — **Agent 1**: ratchet v2 FULL guard (server live kvarn
   250k, port 8091; setsid'd chain pid 114169; logs /tmp/guard_p{1,2}.log).
   **PASS1 DONE: all 9 cases healthy** — @40k 66.6 / @250k 44.7-44.8 /
   @10k 73.0 (deterministic, matches spot check; vs v1 64.0/43.5). PASS2
   running; teardown + results + report on completion.
3. **~17:25-17:45** — **Agent 1**: ctest GPU-verify of test fixes (commits
   444887c3 spec top_k 1→2, 2e34ddc1 gdn k35Routes→MmaUnsplit; binaries built
   in wo-kvarn-hold/build/tests/). Both must go green.
4. **~18:30-20:00** — **Agent 1**: mtp-adaptive docs/107 step 1 — (a) live
   byte-identity vs hold head (3 prompts × 10k ctx × 192 tok, temp 0, no
   seed + pool-active-no-seed), (b) seeded delta (--ngram-mod-seed, 6-prompt
   battery, 2 runs/config determinism), (c) **D-21 depth canary: max verify
   T ≤ 8 across the battery** (new, per docs/119 §11).
   **FIXED (2 fix commits, binary v3 @ 18:41, host-green)**: branch had never
   been live-run with --mtp-adaptive → first-contact bugs, all now fixed:
   (i) decode loop passed full k_buf-sized buffers to round ops whose contract
   is the round width {rk+1} (strict shapes throw; self-describing ops would
   verify garbage) + T∈[2,6] cap + kMaximumMtpDraftTokens=5 → per-round views
   + caps T≤8/rk≤7 (9c7faa3f); (ii) ar_* views were {ar_steps,1} but buffers
   are STEP-major → {1,ar_steps}; D-22 allgather loop j<=k → j<=rk (cols
   k+1..rk unallgathered = garbage accept data); verify-depth canary print
   (NINFER_VERIFY_DEPTH_LOG, both depth sources) (7301a8fc). Agent 1's
   ref_p1/2/3 baselines valid + reused (fixed-k path byte-identical).
   Reference: wo-kvarn-hold build/apps/ninfer-serve.
   **STEP 1 = GREEN (23:05)**: byte-identity PASS in 6 configs (a1/a2/seeded/
   forced-depth-T8/generic-code/repo-code), greedy determinism ×2 PASS,
   canary max T = 4 (default) / 8 (forced cap). Findings doc:
   wo-mtp-adaptive results/mtp_adaptive_step1_findings.md (ef0cb491). ACCEPTANCE
   finding (honest limit): pool n=24 EXACT 24-gram match vs repo source —
   generative LLM rewrites never hit → pool no-op on code-gen (62.4-77.4% =
   MTP-head baseline in all runs; pool load verified 3.1M tok/2M n-grams).
   Value = workload-bound (agent-style in-pool continuation) or smaller-n
   design → Phase 2 (docs/98). Not a defect (mechanism verified correct).
5. **~23:00-00:30** — **Agent 1**: magic-dict docs/108 — step 1 DONE GREEN
   (23:03: counter on==off byte-identity ×3, count-sum 236==236 ✓ (5-tok diff
   vs completion_tokens = round-0 anchors, documented), id-range ✓, merge
   exact ✓, sampling smoke ✓). Next: rebase onto wo/kvarn-hold eb7ba785 (picks
   up ctest CI gate + gdn re-measure; zero src delta → zero conflicts)
   → gdn re-run (expect PASS @ 3.0e-6/1.0e-5) → run_ci.sh --full (first live
   run of the ctest step 2b gate) → report + docs/100 status.
6. **plan owner (CPU, in parallel)**: board + triage + runbook upkeep,
   merge prep (baseline v2 sign-off, /home/intel/verify_baseline.json holds
   pre-final B' numbers — re-baseline at merge only).
7. **tomorrow AM** — plan owner: merge (runbook) → official ratchet v2 full
   guard re-run on post-merge main → baseline v2. Then Phase 2 (docs/115) +
   Phase 3 (docs/113) + kv-uniform Phase 3/4 slots.

## 3. Merge (plan owner, TOMORROW AM — was today ~14:30)

`results/105_merge_runbook.md` — preconditions now include: probes done,
MmaUnsplitT1 perf gate PASS (plain ≥35.0 / verify ≤35.6 / mtp ≥79.0),
byte-identity re-verified, full CI 0 fails, spot check PASS. 77 commits,
--no-ff convention (fe3d8857). Clears docs/77 §8 H0' + REPO.md §4a.

## 4. Known issues

- **Pre-existing ctest reds (CI never runs ctest units) — TRIAGED:**
  results/pre_existing_ctest_reds_triage.md. (a) speculative_round:
  68e2c6bc top_k=1 argmax branch vs Aug-10 seeded-sampling oracle → fix:
  top_k 1→2 (test-only, GPU-verify). (b) gdn_gating_proj: k35Routes 35-model
  cols 1-8 → SmallTSplit10 broken since 07-17 (candidate_is_legal rejects it)
  → fix: k35Routes = single MmaUnsplit (mirrors 1d3a0533). Both land post-gate
  (clean tree), separate commits, GPU-verified.
- **/home/intel/verify_baseline.json overwritten** by verify_battery run path
  (holds pre-final B' numbers 35.44/35.74). NO re-baseline until merge
  (plan-owner sign-off + written justification). Baseline comparisons in the
  tuning window use explicit thresholds.
- **File anomaly (my session only):** two pre-fix binaries + one stale
  ninner-serve inode unreadable from the plan-owner shell (ENOENT despite
  ls/stat normal; files created after ~Aug 29 09:46 fine). Agent sessions
  unaffected (Agent 1 execs them fine). Mitigation: plan owner re-copies the
  binaries from a clean shell if spot check needs them; new builds are new
  inodes = readable.
- Auto-reviewer state file: pre-fix commits were never reviewed (it hung) —
  backfill optional; new commits auto-review from now on.
- docs/105 harness `model` field bug fixed (25ed7aa9) — temp0 battery now valid.
- Agents 2/3/4 absent — plan owner takes over all their CPU work + their GPU
  slots (mtp-adaptive, magic-dict, kv-uniform Phase 3/4).
