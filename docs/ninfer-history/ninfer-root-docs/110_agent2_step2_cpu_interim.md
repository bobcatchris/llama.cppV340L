# 110 — Agent 2: step 2 is CPU-only, start now (interim)

**Status:** INTERIM — clarification for docs/106 while the GPU is held by the
official baseline run (ends ~11:25).
**Mission:** Execute docs/106 **step 2** (both audits) immediately — it needs
no GPU. Your step-3 GPU slot opens when Agent 1's full CI commits (expected
~16:00-17:00; plan owner pings), earlier than the ~19:00 in docs/106 because
the GDN fix (your A/B baseline) lands sooner than scheduled.

---

## No-GPU clarification (read first)

docs/106 step 2's "measure its share of prefill at 80k/160k" does **not**
require a new GPU run:
- The official baseline run (ends ~11:25) writes per-cell JSON to
  `~/ninfer/worktrees/wo-kvarn-hold/results/official_sampling_decode_*.json`
  and `official_greedy_decode_*.json` (also logged in
  `~/ninfer/logs/official_baseline2_20260829_090807.log`). From each cell:
  `decode_time ≈ completion_tokens / server_decode_tps`,
  `prefill_time ≈ wall_s - decode_time`, `prefill_tps ≈ prompt_tokens /
  prefill_time`. That gives prefill t/s + prefill share per variant per
  context — enough for the Phase 3 audit.
- The older 08-29 06:08/06:24/06:28 matrix (same files, tags
  `sampling`/`greedy`) is supplementary.
- If you conclude the data is insufficient for a specific claim, **write the
  audit with what exists + a "targeted measurement needed" section** (name
  the run: which variant, which contexts, which metric) — do NOT launch a
  server or block. I will queue it tomorrow.

## Deliverables (docs/106 §6 step 2, unchanged)

1. `results/106_prefill_materialize_audit.md` (Phase 3): locate the O(n²) KV
   materialize pass in the KVarN prefill path (docs/104 R4), quantify its
   share from the data above (kvarn prefill ~13% slower than bf16/int8 at
   10k — confirm/quantify across 80k/160k), propose the fix design (direct
   read / single pass) + the gate it must pass.
2. `results/106_prefix_reuse_audit.md` (Phase 4): the prefix-restore mechanics
   per variant (docs/104 R5 — incl. the 48k-max prefix-reuse window and the
   73.4 MiB/rank GDN buffers from the docs/77 H3 item), where variants
   differ, unification proposal + gate.
3. Both committed: `results(106): prefill materialize audit (docs/104 P3)` and
   `results(106): prefix reuse audit (docs/104 P4)`.

## Step 3 timing (for your planning)

Your branch currently sits on `93456b5f` (pre-GDN-fix). When Agent 1's full
CI commits to `wo/kvarn-hold`, **rebase `wo/kv-uniform` onto its new head**
(your step-3 byte-identity baseline must be the post-fix build — your commits
touch disjoint files, so this should be trivial; if not, stop and report).
Then step 3 (A/B + guard + fast CI) in your GPU slot.

## Constraints

- Worktree `wo-kv-uniform` only; no merges; no GPU before the ping; no
  touching docs/105 scope (GDN gating plan/kernels) or the guard baseline
  files. Audit docs must cite their data sources (file paths + run stamps).
  Commit messages name docs/106.
