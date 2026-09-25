# 108 — Magic dictionary (counter + hook): land & verify (evening work order)

**Status:** CURRENT — for one implementer agent, tonight (GPU slot after the
`wo/mtp-adaptive` agent finishes, ~23:00). Short order: the branch is
implemented and CPU-proven; this is a rebase + live-proof + land job.
NOTE (10:45, schedule slip): rebase onto the CURRENT wo/kvarn-hold head
(9a31a070) instead of waiting for the ~14:00 merge — same conflict profile,
saves 2.5h. Plan owner merges wo/kvarn-hold → main before your GPU slot.

**Mission:** Take `wo/magic-dict` from "Phase 1 done + Phase 2-prep committed"
to landed on main: rebase onto post-merge main, rebuild, prove the output
counter + hook + `--merge-counts` live (observation-only: byte-identical
behavior, sane counts), full CI, report. Plan owner merges.

---

## 1. Context (60-second version)

The branch (2 commits from `c63b2443`) carries:
- **Phase 1 (docs/100, DONE + verified, offline only):** corpus collector
  (`tools/collect/export_tokenizer.py`, `project_vocab.py`), pinned tokenizer
  (SHA256 matches the official pin), BPE differential engine-vs-HF
  **0 mismatches across 256,916 lines**, draft vocab
  `qwen38_draft_vocab_ids.json` (40,960 ids, domain 248,077).
- **Phase 2-prep (docs/98 §B):** vocab output counter + serve hook +
  `--merge-counts`. This is the live part this order verifies.

The magic dictionary's *value* (acceptance lift from a project draft vocab)
is Phase 2 proper — NOT in scope tonight. Tonight = prove the counting
machinery works live and is observation-only (zero behavior change).

## 2. Environment & build/test

- Worktree: `~/ninfer/worktrees/wo-magic-dict` (branch `wo/magic-dict`) —
  exists; **build dir was deleted, rebuild from scratch.**
- Work protocol: work ONLY in that worktree; never the main tree; commit per
  step; push. **No merges** — plan owner merges.
- Rebuild: `cmake -S . -B build && cmake --build build -j 16` (CUDA `sm_120a`).
- Unit tests: **`/usr/bin/ctest`** from `build/` (PATH `ctest` is broken).
  Must-pass includes `test_vocab_output_counter` (branch's own) + the full
  default set.
- **GPU QUEUE (tonight):** your slot opens ~23:00 (after the mtp-adaptive
  agent's CI completes). CPU step 0 runs BEFORE the slot, starting ~17:00
  after the kvarn-hold merge. One live server at a time; `pkill -x
  ninfer-serve` (NEVER `pkill -f`); `pgrep -x ninfer-serve` + `nvidia-smi`
  before every launch; leave the server restored.
- Server: `bash tools/ops/build_and_serve.sh` conventions; model
  `/home/intel/models/qwen3_8_27b.ninfer`.
- Server tests: `bash tools/ops/run_ci.sh` from YOUR worktree root (relative
  path); `--full` at closeout only.

## 3. Execution order (commit + test each step)

### Step 0 — Rebase + rebuild + ctest (CPU; start ~17:00)
Rebase `wo/magic-dict` onto the NEW main (post-kvarn-merge). Expected
conflicts: `src/runtime/tp2/tp2_backend.cpp` / `.h` (kvarn D-21/22/23 fixes vs
your hook — keep BOTH; the D-21 anchor fix survives verbatim). **Semantic
conflict = STOP and report.** Rebuild, `/usr/bin/ctest` green. Commit
`rebase(wo/magic-dict): onto post-kvarn-merge main (docs/108)`.
**Tests:** build green; ctest green; main's merge head is ancestor.

### Step 1 — Live verification (GPU slot, ~23:00)
On your rebuilt binary:
- **(a) Observation-only proof (the gate):** greedy decode (temp 0), 3
  prompts × 10k ctx, 192 tokens, counter ENABLED → outputs byte-identical to
  the same runs with the counter DISABLED (and to new main, if a main build
  is handy). A counter that changes bytes is a bug, not a feature.
- **(b) Count sanity:** after (a)'s runs: counter file(s) written;
  `sum(counts) == total generated tokens` (± documented hook semantics);
  all ids < 248,077; no negatives/duplicates-of-different-tokens anomalies.
  Then `--merge-counts` on the produced files → merged file = exact
  elementwise sum (verify with a small python check; commit the check output).
- **(c) Sampling smoke:** one temp>0 run with the counter on → coherent
  output, no crash, counts still sane.
Save everything under `results/108_live/`.
Commit results + `live(magic-dict): counter/hook/merge-counts verified (docs/108)`.

### Step 2 — Full CI
`bash tools/ops/run_ci.sh --full` from your worktree (counter off = default
CI path). Commit report with build commit + model identity.

### Step 3 — Report + status
Update the docs/100 status block (Phase 2-prep landed, evidence) +
`results/108_landing_report.md`. Commit `docs(100,108): magic-dict prep landed`.

## 4. Constraints (non-negotiable)

- Worktree only; no merges; commit per step; keep the tree buildable.
- GPU queue rule is absolute — no GPU before your slot; no killing servers
  that aren't yours.
- Do not touch: KVarN path, sampler, the D-21 fix (verbatim), docs/105/107
  scope, the draft-vocab floor rule (docs/100 step 4 — top-20,480 of the
  baseline list).
- No Phase 2 work (acceptance-lift mechanism) — that is a future work order.
- Commit messages name docs/108.

## 5. Definition of done

1. Steps 0-3 committed to `wo/magic-dict` with green tests at each.
2. **Live proof:** byte-identity (counter on==off==main), count-sum equality,
   id-range check, merge-counts exactness, sampling smoke — all outputs
   committed.
3. Full CI 0 fails; report committed with model identity + build commit.
4. docs/100 status updated.
