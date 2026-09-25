# 100 · Magic dictionary Phase 1 — corpus collector (work order + results)

Parent: `docs/98` (§5A spec, §6 draft-vocab construction). This is Phase 1 of
that plan, executed on branch `wo/magic-dict`.

## Scope (Phase 1 = offline only)

No server, no GPU, no model load. CPU-only: extract the pinned tokenizer,
tokenize this branch's tracked sources, emit the three spec artifacts.

## Work order (as executed)

1. **Worktree** `wo/magic-dict` from `main` (`c63b2443`).
2. **Tokenizer extraction** (`tools/collect/export_tokenizer.py`,
   `--out build/collect/` → writes `build/collect/frontend/`, gitignored,
   derivable). `tokenizer.json` SHA256 matches the official pin in
   `tools/convert/qwen3_8_27b/convert.py:35-36`. BPE, base vocab 248,044 +
   33 added tokens = domain **248,077** (= `kTokenizerVocab`,
   `bindings.cpp:429`).
3. **BPE differential (engine vs HF)** — built the engine tokenizer
   (`src/targets/qwen3_6/impl/frontend/tokenizer.cpp` + `src/text/unicode.cpp`
   + `third_party/utf8proc`) as a standalone CLI, tokenized the entire corpus
   with both: **0 mismatches across 256,916 lines**. HF is safe to use as the
   collector's tokenizer (docs/98 §5A's "silent coverage failure" risk is
   ruled out for this corpus).
4. **Baseline verification** — shipped `qwen38_draft_vocab_ids.json` is
   40,960 unique ids, max 248,076 < 248,077 ✓. It is *not* exactly the
   top-40,960 of the ranking fixture row 0 (snapshot drift, a few hundred
   ids) — therefore the floor is "top-20,480 *of the baseline list* by
   baseline freq" per docs/98 §6 (protects the shipped distribution).
5. **Collector** (`tools/collect/project_vocab.py`) — spec-conformant:
   `git ls-files` only (never untracked), ext filter
   `.h .hpp .cpp .cu .cuh .py .sh .cmake .txt .md .jinja`, excludes
   `.git/ build/ models/ artifacts/`, non-UTF8/binary skipped, domain check
   parsed live from `bindings.cpp`, built-in spot check + round-trip.
6. **Run** `--branch wo/magic-dict --out build/collect/`.

## Results (see `results/magic_dict_phase1_report.md` for the full report)

- corpus: **1,158 files / 11.4 MB / 2,988,563 tokens / 25,782 unique ids**
- draft vocab: 20,480 floor + **15,220 corpus** + 5,260 backfill = 40,960
  unique ids, all < 248,077
- **corpus token coverage: project 100.000% vs shipped baseline 91.834%**
  (baseline misses 8.2% of this project's token mass — the docs/48 failure
  mode at scale; its unique-id coverage here is 60.9%)
- 10,086 ids in the project list are new to the shipped baseline
- pool stream: 11.95 MB uint32-LE (2,988,563 ids) → `pool_seed/project_pool_stream.bin`
- backfill rule (new, documented in the script): the corpus has only 25,782
  distinct ids, fewer than the 20,480 non-floor slots; the fixed head
  geometry (20,480 rows/rank) requires exactly 40,960, so the shortfall is
  filled with next-best baseline ids.

## Artifacts

| path | committed | note |
|---|---|---|
| `tools/collect/project_vocab.py` | yes | the collector (re-runnable per branch) |
| `tools/collect/export_tokenizer.py` | yes | pinned tokenizer extraction |
| `tests/multi_gpu/data/qwen38_draft_vocab_project.json` | yes | deliverable, 40,960 ids (same format as baseline) |
| `results/magic_dict_phase1_report.md` | yes | full report + top-50 table |
| `build/collect/{frontend,vocab,pool_seed}/` | no | derivable; gitignored |

## Phase 2-prep (landed with Phase 1, no server needed)

### B: passive output counter (code complete, validation needs server)

- `src/runtime/tp2/vocab_output_counter.h` — header-only
  `VocabOutputCounter`: accepted-token counter, domain 248,320 (output-head
  rows), delta flush files `output_counts_<UTC>_<seq>.bin`
  (`"NINFCOUT"` + u32 version + u32 vocab_size + u64 n_tokens + u32 counts),
  flush every 100k tokens + shutdown, coverage log every 10k.
- Hook: `tp2_backend.cpp` MTP round loop, rank 0, after `acc_accepted += a` —
  `backend.vocab_counter.add(lic, a + 1)` (anchor + accepted drafts only;
  rejected drafts never reach this site). One line.
- Init: `TpBackend::create()`, **opt-in via env var `NINFER_VOCAB_COUNT_DIR`**
  (flush directory). Default unset ⇒ counter disabled ⇒ serve behavior
  bit-identical (house rule). No serve-option changes (doc 69 owns
  `serve_options.cpp`; the env→flag promotion is a one-liner at merge time).
  Env chosen deliberately: `make_rank` loads the draft vocab before the
  backend object exists, so the counter is a `TpBackend` member initialized
  in `create()` after rank assembly; the accepted-token hook lives in
  `run_tp2_request`, which only sees `backend`.
- Shutdown flush in `~TpBackend()`.
- Test: `tests/test_vocab_output_counter.cpp` (ctest `ninfer_vocab_output_counter_test`):
  disabled no-op, counting/coverage, out-of-range drop, delta union == totals
  (bin read-back), empty flush no-op, shutdown delta, double-init, null-vocab.
  `ctest` green; `tp2_backend.cpp` TU compiles clean under CUDA 13.1.

### `--merge-counts` (collector extension, code complete)

`project_vocab.py --merge-counts [--bins-dir DIR]` (default `<out>/vocab`):
parses all `output_counts_*.bin` (delta semantics ⇒ plain sum), weights
output counts **×2** over corpus counts (docs/98 §B), re-ranks with the same
floor/remainder/backfill logic, keeps the domain check (ids ≥ 248,077 from
the 248,320-row head are excluded from ranking), writes
`draft_vocab_project.json` + appends a merge section to `collect_report.md`
with merged-distribution coverage (project vs baseline). Verified end-to-end
with a synthetic bin; on the real corpus the merge path reproduces the
Phase 1 list when the bins dir is empty.

### What still needs the server (Phase 2/3)

- Run serve with `NINFER_VOCAB_COUNT_DIR=<dir> --draft-vocab
  tests/multi_gpu/data/qwen38_draft_vocab_project.json` on an agent-style
  workload; collect bins.
- `--merge-counts` on those bins → promote `draft_vocab_project.json` to
  `tests/multi_gpu/data/` (rebuild the artifact via the standard tooling).
- §6 gate: ≥99.0% coverage on the agent-style run (corpus 100% is a proxy,
  not the gate).
- C: `--ngram-mod-seed` wiring in the docs/69 port (pool stream is ready).
