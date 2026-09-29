# 134 — Work Order: M1 — doc-128 CPU references D4–D7 + D3 (the long pole)

**Owner:** agent 2 (second worker) — spawn pending; deliverable on spawn.
**Coordinator:** 01a05fb6. **GPU:** none until final chain test (coordinator
schedules the slot). **WBS ref:** docs/132 §11.3 item 3b. **Read first:**
docs/128 in full (§6 M1, §7 risk (a) — accumulation order is THE risk,
§8.4 tolerance calibration, §8.7 worktree), docs/83 (the dormant QK ref),
docs/131 (test-suite patterns), docs/132 §11.3.

## 1. Context

The unified-kernel certification pipeline (docs/128) needs CPU references for
the decode chain phases D4–D7 so that any future kernel bug localizes in one
CI run. This work order builds those references. **This is the long pole of
the whole closeout (2–3 days CPU) — it starts day 1, in parallel with the
GPU-heavy Stage 0, so it is done before it is needed.** M0 (docs/133, gemini)
builds the chain skeleton + dump hooks in `wo/phase-gate`; your references
land in `tests/` only and merge into `wo/phase-gate` when M0's skeleton is
ready (or stand alone until then).

## 2. Environment & test entry points

- **Worktree:** NEW, based on `wo/kv-uniform` HEAD at start: `cd
  /home/intel/ninfer/repo && git worktree add
  /home/intel/ninfer/worktrees/wo-phase-gate-m1 -b wo/phase-gate-m1
  wo/kv-uniform`. Own `build/` dir — never build into another agent's
  build. (You work in tests/ only; no `src/` edits — that keeps the later
  merge into `wo/phase-gate` conflict-free with gemini's M0 hooks.)
- **Test entry:** the reference binaries themselves (see §3); they must
  compile and self-check against known-good artifacts (from M0's dump hooks
  when available, otherwise against CPU-side recomputation).
- **GPU:** not needed until the final chain test — the coordinator schedules
  that slot once M0 is ready. Your daily loop is pure CPU.
- **Test gates:** if a reference disagrees with a known-good artifact,
  the first suspect is accumulation order (docs/128 risk a) — check your
  reduction/accumulation order against the kernel's BEFORE touching
  tolerances. Tolerances are reviewed like code (docs/128 §8.4).

## 3. Scope (and NOT)

IN, in this order:
1. **D4 (QK):** wire the dormant `tests/kvarn_codespace_qk_cpu_ref.cpp`
   (docs/83) into the chain artifact format (fp32 scores; small ctx where
   the CPU ref is exact). Int8/s8 siblings for reference:
   `kvarn_int8_qk_cpu_ref.cpp`, `slice3_i8_s8qk_cpu_ref.cpp`.
2. **D5 (softmax + rescale):** fp32 probs CPU ref.
3. **D6 (PV + per-tile scales + fp32 accumulation):** accumulated outputs
   CPU ref — **same-input fp32 accumulation order for byte-identity**
   (this is the hard one, risk a).
4. **D7 (reduce: split-K / multi-CTA shared reduce):** per-head finals CPU
   ref.
5. **D3 (draft, MTP draft head W8G32 → 3 drafts):** reference draft on the
   pinned input, pattern `tests/targets/qwen3_6_27b/test_draft_head.py`.
6. **Tolerance table draft** (docs/128 §8.4): per phase, byte-identity where
   order matches, FP64 + tolerance where tile order can't be matched —
   documented, reviewed like code.

NOT: M0 (gemini's, docs/133), M2 CI wiring, any `src/` kernel changes,
prefill (M3, later — that's also your likely next job, but not this WO).

## 4. Design

- Byte-identity is the goal (docs/128 §7 risk a): the CPU ref must
  accumulate in the SAME ORDER as the kernel where the tile order is
  reproducible. Where it isn't (split-K order is hardware/scheduler
  dependent), pin an FP64 oracle + per-phase tolerance in the table — and
  record WHY byte-identity is impossible for that phase.
- Same pinned prompt as M0 (docs/128 §8.2) so artifacts are chain-compatible.
- References must be self-contained CPU code (no CUDA in the ref itself;
  a small GPU driver to produce "known-good" inputs for self-check is fine
  and is what M0's hooks provide).

## 5. Design decisions (FINAL — do not re-litigate)

1. Worktree = `wo/phase-gate-m1` based on `wo/kv-uniform`; tests/ only.
   REJECTED: working in gemini's `wo-phase-gate` worktree (two writers,
   conflict risk) — revisit never.
2. Byte-identity where order matches; FP64 + tolerance only where proven
   impossible, with the proof documented. REJECTED: blanket tolerances —
   a tolerance without a reason is a bug.
3. Reuse the dormant QK ref as the D4 base. REJECTED: rewriting it —
   revisit only if it's wrong.
4. Same pinned prompt as M0. REJECTED: per-ref prompts — revisit never
   (chain compatibility breaks).

## 6. Execution order (commit + test each step before the next)

1. Worktree + build; baseline compiles.
2. D4: port dormant ref to chain artifact format; self-check.
3. D5: softmax+rescale ref; self-check.
4. D6: PV+scales ref with matching accumulation order; self-check (this is
   the 1–2 day part — budget it).
5. D7: reduce ref; self-check.
6. D3: draft-head reference pattern.
7. Tolerance table draft (documented per phase).
8. (When M0 is ready + GPU slot is scheduled by coordinator) chain
   integration test: your refs vs M0's known-good artifacts → all phases
   agree (byte or within documented tolerance).
9. Write the debrief (§9): per-phase order analysis, tolerance rationale,
   what M1-integration (gemini) needs from you.

## 7. Pitfalls

- **Accumulation order is the whole game** (docs/128 risk a). If D6/D7
  "almost match", it is an order problem, not a tolerance problem — find
  the order mismatch before widening anything.
- bf16 rounding semantics: document every assumption about how the kernel
  rounds (FMA vs separate mul/add, upcast points) and match it exactly.
- Pinned prompt immutable: committed + hashed (docs/128 risk c).
- No GPU in your daily loop — if you catch yourself waiting on the GPU,
  you're doing M1's integration step early; go back to CPU work.
- Don't touch `src/` — if you think a KERNEL bug is blocking your ref,
  stop and report to the coordinator (that's the whole point of the
  pipeline: the chain should find that, and your ref must stay the
  oracle, not the patch).

## 8. Definition of done

1. D4–D7 + D3 references compile and self-check clean (evidence in
   debrief; chain test vs M0 artifacts once the GPU slot runs).
2. Tolerance table drafted, per-phase, with byte-identity-or-FP64
   rationale for every phase.
3. All work committed on `wo/phase-gate-m1`; worktree leaves clean.
4. Debrief written (§9).

## 9. Debrief (owner fills before stopping)

- (to be written)

---

## 10. ADDENDUM — scope superseded (2026-09-02, coordinator)

Status of this WO when A2 (01a0632c) started: G (gemini) had ALREADY
committed packed-route M1 refs D1–D7 + D3 on `wo-phase-gate`
(`5da04594` → `6ae2581e`, verified by coordinator). Those refs exercise
the **packed-route CLONE**, not the production kernel — review blocker
§2.1 (wo-phase-gate docs/132_phase_gate_review_feedback.md) is open:
"a regression in the unified kernel or the kvarn launcher passes this
gate green".

**Superseded lane (A2, implementer of record for production-route refs):**
- Target the **PRODUCTION slice4 route** (`gqa_decode_slice4_kvarn_kernel`
  + unified/slice4 launcher) instead of duplicating the packed refs.
- Pin the exact accumulation order per phase: Bc=32 MMA, cp.async
  staging, rotated-domain dequant prologue, online softmax/rescale,
  split-K merge (docs/128 risk a = the byte-identity-or-FP64 call).
- The packed-route refs (G's commits) REMAIN as the secondary oracle
  (logic-level catch). No duplication from A2.
- A2's refs CONSUME production-kernel dumps. Dump-arena hooks for the
  production kernel = G's lane (src/), pattern = `write_phase()` /
  `dump_dir/phase_*.bin` (dtype + prompt_sha256_hi) in
  wo-phase-gate `tests/phase_gate.cu`. Interface to be defined when the
  hooks land.
- **Pinned prompt resolved**: not a text file — `PinnedInput::make(seed,
  ntiles_k, ntiles_v)` in wo-phase-gate `tests/phase_gate.cu`
  (deterministic LLCG + channel outliers, bf16 round-trip, default
  seed=1, sha over k_bf‖v_bf). A2 reuses it verbatim for
  byte-identity; do NOT change its output (sha ratchets break).
- §1–§9 above still apply except where they assumed packed-route
  targets; worktree/branch/tests-only rules unchanged.
