# 116 — Phase 4: prefix-reuse unification (all variants snapshot): work order DRAFT

**Status:** DRAFT (plan-owner review pending) — drafted 2026-08-29 per the
Phase 4 audit (`results/106_prefix_reuse_audit.md`, commit 08eea70b). Scope:
docs/104 Phase 4. Sequencing: AFTER docs/113 (Phase 3) lands — same code
neighborhood (prefix-restore dispatch in tp2_backend.cpp + kvarn_workspace.cpp),
one GPU owner at a time.

**Mission:** prefix-restore becomes variant-independent — every variant
captures the bf16 open-page tails + MTP seed at cache-save and restores them
on a hit (Option (a) of the audit, the recommended lower-risk path) — with
restore identity, t/s, and VRAM budget gates passing. Done = gate §5 passed,
closeout doc committed.

Read this document fully before touching code.

---

## 1. Context (60-second version)

Prefix reuse is variant-specific today: BF16/INT8 reuse their paged cache in
place (exact — nothing to restore); KVarN must rewind its in-flight tiles and
D2D-restore a bf16 open-page snapshot + MTP seed, because its open-tile commit
at the 64-token boundary is lossy (docs/102 §9, D-16 snapshot dance). The GDN
state restore (48 layers) is already common to all three.

docs/104: variants should differ only by storage loss — not by restore
*mechanism*. Option (a) unifies on "all variants snapshot"; option (b) (make
KVarN commit exact-enough to drop the snapshot) is a numerics change (risk
class per docs/50 B2 hazard: semantically-neutral edits shift MTP acceptance)
and is NOT the first move.

**Must NOT change:** KVarN commit numerics, GDN restore, prefix-cache capacity
(D-20, 48k working max), decode/verify paths, D-16/D-21 closeouts.

**Already built and verified (do not redo):**
- KVarN snapshot capture/restore (kvarn_workspace.cpp:523/555), QA-verified.
- Phase 4 audit + recommended option + gate (results/106_prefix_reuse_audit.md).
- Restore dispatch: tp2_backend.cpp — `:923` (snap_tc truncation), `:979-982`
  (restore branch, gated `token_count == prefix_len`).

## 2. Environment & build/test

Same as docs/113 §2: worktree wo/kv-uniform (or a fresh worktree if Phase 3
landed on it and the plan owner says so), `/usr/bin/ctest`,
`bash tools/ops/run_ci.sh` from worktree root, GPU slot assigned by plan owner,
`pkill -x ninfer-serve` only, model meta sha `45912dd83c71a1b3`.

## 3. Steps

**Step 0 — measure first (GPU, ~1 h).** The audit's data-driven decision
(docs/104 §4 Phase 4) needs the restore-cost numbers that no committed run has:
cache-hit restore battery per variant at ~45k (just under the 48k window):
turn-2 TTFT / restore t/s, cold-vs-hit delta (docs/50 D-15 contrast: cold
prefill 683-728 tok/s vs cache-hit restore 23.5-24.2k tok/s). Also measure the
save-time capture cost per variant today (KVarN pays it; bf16/int8 pay zero).
Deliverable: table in closeout doc. (If data shows option (b) is cheap AND the
numerics re-validation is green, re-propose before Step 1 — plan owner
decides.)

**Step 1 — extend snapshot capture to BF16/INT8 (CPU+GPU, ~1 d).**
At cache-save, bf16/int8 capture the bf16 open-page tails + MTP seed
(ar_hidden/d0) into the same snapshot structure KVarN uses (generalize
`kvarn_prefix_snap` to a variant-agnostic `prefix_snap` in tp2_backend state).
Restore dispatch: the `is_kvarn → rewind+snapshot` branch becomes
`all variants → snapshot restore` (KVarN keeps its rewind; bf16/int8 gain the
D2D restore). No numerics changes anywhere. Gate: fast CI 0 fails +
byte-identity of the restore path per variant (Step 3's battery at Step-1
granularity).

**Step 2 — correctness (GPU, ~half day).**
- Prefix-restore identity per variant: restored decode byte-identical to
  non-restored decode from the same cache point (existing A2/A4 batteries +
  prefix-identity battery, per variant).
- Multi-turn unaligned-restore battery T17 unchanged at all contexts.
- KVarN 250k must-pass battery unchanged.
- Determinism 2× per variant (snapshot D2D must be exact).

**Step 3 — performance + VRAM gate (GPU, ~half day).**
- Restore t/s per variant within the Phase-0 envelope: bf16/int8 ~unchanged
  (capture is save-time, off the hot restore path — MEASURE, don't assume),
  KVarN no regression.
- Working set: unified restore stays within docs/77 G-5 budget (~20
  KB/token/rank arena + 73.4 MiB/rank GDN buffer) at the 48k window; validate
  against ACTUAL free VRAM at startup (D-20: the 98k OOM passed the pre-check
  — the gate is a startup check against real free VRAM, not the pre-check).

**Step 4 — closeout (CPU+GPU).**
`run_ci.sh --full` 0 fails (known reds only). Closeout doc
`results/116_prefix_restore_unification_closeout.md`: Step 0 tables, gate
numbers, option (a)-vs-(b) decision with data, remaining risk.

## 4. The gate (all must pass)

1. Prefix-restore identity per variant (byte-identical vs own non-restored
   decode from same cache point).
2. Restore t/s per variant within Phase-0 envelope (bf16/int8 ~unchanged,
   KVarN no regression).
3. T17 multi-turn unaligned battery unchanged; 250k KVarN must-pass battery
   unchanged.
4. Working set within G-5 budget at 48k window; startup VRAM check against
   actual free VRAM (D-20 pattern).
5. No decode-side regression (packed decode byte-diff battery identical vs
   pre-Step-1 build); guard no cell regresses >2%.
6. Fast gate 0 fails per step; full CI 0 fails at closeout.

## 5. Non-goals

- Option (b) (KVarN commit exactness) — numerics change; only if Step 0 data
  says so and plan owner re-approves.
- Raising `--prefix-cache-capacity` (D-20 follow-up, docs/72).
- GDN restore (already common).
- Phase 3 (docs/113) — must land first.
- MTP-seed capture semantics changes (reuse existing ar_hidden/d0 capture).

## 6. Risks

- Option (a) adds a D2D capture at every cache-save for bf16/int8 — the
  "small" cost must be measured, not assumed (Step 3). If it shows up in
  turn-2 TTFT, the fallback is a lazy capture (only when a prefix-cache entry
  is being written) — design in Step 1, measure in Step 3.
- Snapshot structure generalization touches tp2_backend state layout — the
  merge window with main (post-wo/kvarn-hold merge) may land during this
  work; rebase discipline per docs/99.
- B2 hazard class (docs/50): any "neutral" edit near the restore path can
  shift MTP acceptance on restored requests — the identity battery exists
  exactly for this; any acceptance drift = STOP.

## 7. References

- results/106_prefix_reuse_audit.md (08eea70b) — mechanics survey + gate source.
- docs/104 (R5, §3, §4 Phase 4), docs/102 §5.1/§8/§9, docs/54 §9 (250k battery),
  docs/50 D-16/D-15/D-20/B2, docs/77 G-5/G-3b, docs/serving.md (restore
  window/VRAM), tp2_backend.cpp restore dispatch, kvarn_workspace.cpp.
- docs/113 (Phase 3, lands first), docs/99 (template).
