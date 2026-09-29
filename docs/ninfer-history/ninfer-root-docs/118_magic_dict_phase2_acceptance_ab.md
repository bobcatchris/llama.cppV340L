# docs/118 — magic-dict Phase 2: acceptance A/B + output-trained ngram pool (work order)

Date: 2026-08-29 (plan owner, CPU lane)
Status: DRAFT — starts after (1) wo/magic-dict full-CI green (Agent 1 queue),
(2) merge (tomorrow AM), (3) step 1 CPU work below lands.
Preceding: docs/98 §6 ("acceptance benefit measured on agent-style runs,
Phase 2"), docs/100 (Phase 1), docs/108 (landing), ef0cb491 (step-1 findings:
ngram pool n=24 never fires on generative code — phrasing mismatch).

## Question to answer
Does project specialization (draft-vocab re-ranking, and/or an ngram pool
built from the model's OWN accepted outputs) measurably raise MTP
acceptance / decode t/s on agent-style workloads in this repo — vs the
shipped generic 40,960 vocab? Gate: lift ≥ 3 acceptance points or ≥ 2% t/s,
otherwise document negative result (magic-dict = smaller-head only).

## Facts settled (no re-investigation needed)
- Draft head needs NO training: at startup it is row-copied from the full
  `text/output_head` (248,320-row q4) — one row per selected ID, dequant
  32-block → requant 64-block (tp2_backend.cpp create()). Re-ranking the
  vocab = change the ID JSON; "head generation" = a few seconds of row-copy
  at next startup.
- Vocab file is loaded from a fixed path (tp_engine.cpp:352,
  tests/multi_gpu/data/qwen38_draft_vocab_ids.json). A/B = file swap (or add
  a --draft-vocab serve option — preferred, ~15 min).
- Counter (NINFER_VOCAB_COUNT_DIR) emits a token HISTOGRAM (NINFCOUT:
  per-ID u32 counts) — enough for vocab re-ranking, NOT enough for ngram
  construction (needs ordered sequences).
- ngram pool = offline artifact (project_pool_stream.bin, raw token stream)
  replayed at startup (104 ms measured). Pool built from repo SOURCE tokens;
  drafts are matched against MODEL OUTPUT tokens → phrasing mismatch = the
  step-1 "never fires" finding.
- MTP greedy (temp 0) is byte-identical to plain greedy (D-21) — holds for
  ANY vocab/pool config (verified 6 configs, ef0cb491). Correctness gate is
  trivially maintainable; the metric is acceptance, not identity.

## Step 1 — instrumentation (CPU, plan owner, ~1-2 h)
1. Add serve option `--draft-vocab <path>` (default = current hardcoded
   path). Unit: options test.
2. Counter extension: env `NINFER_VOCAB_STREAM_DIR=<dir>` → alongside the
   histogram, append accepted token IDs as raw uint32-LE to
   `stream_YYYYMMDD_HHMMSS.bin` per flush (same hook, same lic vector;
   ~4 B/accepted token — negligible). Keep histogram behavior unchanged.
   Unit test: stream bytes == accepted IDs in order across 2 flushes.
3. Offline tool `tools/collect/pool_from_stream.py`: token stream →
   project_pool_stream.bin format (n=24 default; `--n` flag for sweeps).
   Reuse project_vocab.py pool construction. Also `--report` mode: emit
   n-gram coverage stats (distinct n-grams, top-k frequencies) for pool QA.
4. Commit to wo/kvarn-hold (post-merge) — tests only + tools; zero decode
   path changes beyond the option.

## Step 2 — burn-in (GPU, Agent 1, one-time, ~3 h)
Build a task suite: 20-30 scripted agent-style tasks drawn from recent repo
work (continue truncated function, fix small bug, add test, refactor
snippet — each = prompt + expected-ish output; temp 0 greedy). Store as
JSON in results/118_tasks/. Run serve per config, capture:
  - acceptance (tok/round), decode t/s,
  - accepted-token stream (NINFER_VOCAB_STREAM_DIR),
  - correctness: output sha vs plain-greedy (non-MTP) reference (D-21).
Configs: (A) generic vocab, no pool [current default]; (B) project vocab
(qwen38_draft_vocab_project.json), no pool.
Deliverable: results/118_burnin/ per-task + aggregate.

## Step 3 — pool construction (CPU, minutes)
From burn-in config-B accepted streams (and separately from config-A):
- P_out: pool from MODEL OUTPUT stream (the new thing),
- P_src: pool from repo source (existing project_pool_stream.bin).
Report n-gram coverage stats for each (expect P_out to contain the model's
actual phrasings by construction).

## Step 4 — A/B matrix (GPU, Agent 1, ~4-6 h)
Same task suite, fixed-k MTP (k=3, temp 0), measure acceptance + t/s:
  A. generic vocab, no pool                    (baseline)
  B. project vocab, no pool                    (vocab effect)
  D. project vocab + P_out pool                (full loop)
  C. project vocab + P_src pool                (optional, controls phrasing
                                               theory: expect ~A)
Correctness gate per config: byte-identity vs plain-greedy reference (D-21).
Secondary (overfit check): re-rank vocab on first-half tasks, measure on
second-half tasks — does train-half re-ranking generalize?

## Step 5 — report + decision (plan owner)
- results/118_report.md: acceptance/t/s table, D-21 gate, overfit check,
  pool coverage stats.
- Gate: D (or B) ≥ A + 3 acceptance points or +2% t/s → adopt project vocab
  (and pool if it adds) as deployment default + document the loop cadence
  (re-burn-in when project drifts). Otherwise: negative result, keep
  generic, magic-dict value = smaller head only; close docs/98 §6.

## Cost
CPU: step 1 (~1-2 h) + step 3 (minutes). GPU: step 2 (~3 h) + step 4
(~4-6 h) ≈ one GPU day, one-time. Re-runs after drift: step 4 only.

## Sequencing
1. Agent 1 current queue (rebase → gdn re-run → full CI → docs/108 report).
2. Merge (tomorrow AM).
3. Step 1 (plan owner) → step 2-4 (Agent 1) → step 5 (plan owner).
