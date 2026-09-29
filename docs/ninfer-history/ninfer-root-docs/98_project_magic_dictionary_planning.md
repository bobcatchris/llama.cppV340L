# 98 — Project-Aware MTP Drafting ("Magic Dictionary"): Planning

**Status:** PLANNING — design + research findings; not yet a work order.
**Owner:** main side. Feeds docs/69 (ngram-mod + adaptive MTP port) and may
spawn 1–2 small work orders after §9 decisions.

**Nickname:** "magic dictionary" — a per-project set of drafting aids that

> **Landed on main 2026-08-26 as docs/98** (renumbered from untracked branch-local 72 to avoid collision with the KVarN attention work order; see docs/77).
makes MTP speculation match *this* codebase and *this* workload.

---

## 1. Mission

Spec decoding speed is bounded by **draft acceptance**, and acceptance is
bounded by how well the drafter's proposal space matches the tokens the
target actually emits **on this project**. Today our drafter is tuned to a
generic distribution (the 3090 repo's prompt mix) and we have measured the
cost of that mismatch (docs/48: 7.6% coverage gap → 67.7% vs 80.4%
acceptance, hence `--no-draft-vocab` in the battery).

Goal: make the drafter **project-aware** by (a) building drafting data from
the project itself (source, docs, history) with no manual step, and (b)
collecting output-side data passively from live serving — so the "magic
dictionary" builds and updates itself.

Done (phase 1) = a script rebuilds a project-tuned draft vocab + n-gram pool
seed from the repo, the server loads them opt-in, and coverage/acceptance
are tracked per run — with losslessness gates (T10 + battery) green.

## 2. What exists today (do not redo)

| Piece | Where | State |
|---|---|---|
| Draft vocab mechanism (40,960-row head slice, remapped argmax) | `src/runtime/tp2/tp2_backend.cpp:326+`, `src/ops/linear/w8/w8_config.h`, `src/core/multi_gpu/one_shot_argmax.h` | Built; **disabled in battery** (docs/48) because the 3090 list misses 7.6% of our outputs |
| `--lookup` per-request n-gram (2–4 grams) | `tp2_backend.cpp:46` (`find_context_lookup_drafts`), used at 1174/1491 | Built; small, per-request only |
| ngram-mod port + adaptive MTP depth (3→12) | docs/69 work order | **Scoped, not started** (no `wo/mtp-adaptive` branch) |
| Acceptance stats (`acc_accepted/acc_rounds`) | `tp2_backend.cpp:~1365` | Built |
| Tokenizer (Qwen3.8 BPE, merge ranks embedded) | `src/targets/qwen3_6/impl/frontend/tokenizer.h` | Built; artifact domain check at `bindings.cpp:429` |
| Request log + metrics plumbing | `src/serve/request_log.cpp` | Built |
| Headroom: draft head is NOT the current round bottleneck (MTP attention/FFN is; docs/48 §Evidence) | — | The 40k vocab buys ~zero t/s *today* but becomes the cost-saver for long adaptive drafts (see §8) |

## 3. Problem statement

Two distinct data consumers, one data pipeline:

1. **Draft vocab (token-ID list)** — limits what the MTP head can propose.
   Any target token outside the list = guaranteed rejection. Needs: the
   **output token distribution** of the model on this workload.
2. **n-gram pool (context → next token)** — drafts verbatim repeats. Needs:
   the **text corpus** the model is likely to reproduce (repo + its own
   past outputs).

Key finding (validated externally, §4): domain shift *does* drop acceptance
(2503.07807), and retrieval/n-gram drafting from a corpus is a recognized
training-free fix (RASD 2503.03434, llama.cpp ngram-mod, vLLM n-gram
spec). Our current 40k list is exactly the "generic drafter, out-of-domain
project" failure case, measured (docs/48).

## 4. External research (2026-08-25)

### 4.1 llama.cpp ngram-mod (PR #19164, merged — we're porting it in docs/69)
- LCG rolling hash of n-grams (default **n=24**), stores next token; ~16 MB,
  constant memory, **process-shared across server slots** ("different
  requests can benefit from each other").
- Variable draft length (`--draft-min 48 --draft-max 64`); small `n`
  discouraged.
- Upstream application notes: *"iterating over a block of text/code"*,
  *"reasoning models (when they have to repeat their thinking)"*,
  *"summarization"* — i.e., exactly agent traffic.
- Header: `common/ngram-mod.h` — `idx(tokens) / add(tokens) / get(tokens)`,
  `int32_t` entries, `EMPTY = -1`. Port target in docs/69 §6 step 3.

### 4.2 RASD — Retrieval-Augmented Speculative Decoding (arXiv 2503.03434)
- Motivation matches ours verbatim: model-based speculation "frequently
  becomes less effective in out-of-domain scenarios"; RASD drafts by
  retrieving context from a **database** instead of relying on the draft
  model. Predecessor of ngram-mod-style pool drafting.

### 4.3 Training Domain Draft Models (arXiv 2503.07807)
- Confirms the hypothesis: "when adapting speculative decoding to
  domain-specific target models, the acceptance rate of the generic draft
  model drops significantly due to domain shift."
- Their fix = train a domain draft model (heavy). Ours = vocab + n-gram pool
  (light, no training) — acceptable because our drafter is the model's own
  MTP head, not a separate small model.

### 4.4 Oilbird (arXiv 2608.03839, Aug 2026 — newest)
- Training-free exact-suffix lookup drafting, evaluated incl. **tool-calling
  traffic**.
- Key diagnostic for us: lookup misses are often a problem of
  **"addressing rather than coverage"** — on dense tool-calling benchmarks
  ~half of what exact matching misses **is present in the pool** but
  unreachable because one differing value token breaks the exact match.
  They add a semantic second channel.
- Implication: an n-gram pool alone under-serves JSON/tool-call traffic
  (agent traffic is heavy JSON with varying values). Plan for it (§9 Q4);
  don't over-promise on pool hit rates.

### 4.5 ToolSpec (arXiv 2604.13519)
- Schema-aware + retrieval-augmented speculation for tool calls: traces are
  "highly structured, conform to constrained schemas, recurring invocation
  patterns." Future extension: seed/structure pool + vocab from our tool
  schemas (serve is OpenAI/Anthropic-compatible — agents drive it).

### 4.6 vLLM
- First-class **N-Gram Speculation** + Suffix Decoding alongside EAGLE/MTP
  (v0.27 docs). Industry convergence on n-gram/lookup drafting as the
  training-free lever — de-risks docs/69.

**Net:** our direction (project corpus → n-gram pool; output distribution →
draft vocab) is mainstream, validated, and unstarted in our tree. Nothing
found that changes the plan; §4.4 is the one caveat to design around.

## 5. Design: the data pipeline

Three strictly additive pieces. Each is opt-in; default OFF ⇒ bit-identical
server behavior (house rule).

```
A. corpus collector (offline, per branch)
     git ls-files → filter → tokenize (Qwen3.8 BPE) →
       1. vocab/project_token_counts.json        {id: count}
       2. draft_vocab_project.json               40,960 ids (see §6)
       3. pool_seed/project_pool_stream.bin      tokenized corpus stream
B. passive output counter (online, in serve)
     uint32_t[248320] (≈1 MB), incremented per ACCEPTED token
       → periodic flush vocab/output_counts_YYYYMMDD.bin
       → coverage % log line (generated tokens inside active draft vocab)
C. ngram-mod pool (docs/69) — self-filling
     seeded at startup from A.3; fills online from generated tokens
     (process-shared); optional pool export/import across restarts
```

### A. Corpus collector — spec
- Script: `tools/collect/project_vocab.py` (Python; tokenizer via
  `tools/collect/export_tokenizer.cpp` if we need the engine's exact BPE —
  cross-check against HF Qwen3 tokenizer first; **mismatched BPE = silent
  coverage failure**, so include a round-trip + known-string→id spot check
  in the script, and validate against `bindings.cpp:429`'s domain check).
- File selection: `git ls-files` (respects branch), include `.h .cpp .cu .py
  .sh .cmake .txt .md .jinja .hpp .cuh`; exclude `.git/ build/ models/
  artifacts/` + non-UTF8 (binary sniff). **Do not** scan untracked files.
- Rebuild trigger: on demand per branch (`python3 tools/collect/project_vocab.py
  --branch main --out build/collect/`); later a git hook / CI step.
- Per-branch artifacts are natural (an agent on branch X is served branch X's
  corpus). Commit only the script + tiny samples to git; cache built
  artifacts under `build/collect/` (they're derivable).

### B. Passive output counter — spec
- One `uint32_t[248320]` in serve state (≈1 MB, host). Increment for
  **accepted tokens only** (that is the target's actual output; rejected
  drafts must not pollute the distribution).
- Flush: every 100k tokens + on shutdown → append/merge to
  `vocab/output_counts_*.bin` (path configurable; default under
  `~/ninfer/data/<project>/`). Counts only — **no token text stored**
  (privacy + size).
- Metrics: every 10k tokens log `vocab_coverage=99.2%` to serve.log (via
  request_log metrics plumbing); this becomes the tracked dictionary-quality
  number.
- Rebuild merge: `project_vocab.py --merge-counts` unions corpus counts +
  output counts for ranking (output counts weighted ×2 — they are the
  ground-truth distribution).

### C. Pool seeding (rides on docs/69)
- docs/69 step 3 (ngram-mod pool) gains one flag: `--ngram-mod-seed <file>`
  — replay the corpus token stream through `add()` at startup. No other
  change to the port.
- Optional (later): `--ngram-mod-export/import` to persist pool across
  restarts so a project's live dictionary survives.

## 6. Draft vocab construction (the 40,960 budget)

Budget is fixed by the sharded head geometry (20,480 rows/rank, 106 MB;
docs/14). Building the list:

1. `baseline` = current `tests/multi_gpu/data/qwen38_draft_vocab_ids.json`
   (40,960, general distribution).
2. `corpus` = token ids from A.1 (this project).
3. `outputs` = ids from B (this project's actual emissions, when available).
4. Rank: **floor** = top-20,480 of `baseline` by baseline frequency (protects
   prose acceptance — docs/48 shows what happens when the list is off-distribution);
   remaining 20,480 slots = `corpus ∪ outputs` ranked by merged count.
5. Emit `draft_vocab_project.json`. Load path already exists
   (`tp2_backend.cpp:326+`).
6. **Gate before re-enabling:** measured coverage of the list on a real
   agent-style run ≥ **99.0%** (docs/48's 92.4% → 67.7% acceptance is the
   floor to beat; 99%+ expected ≈ 80%+ acceptance with no new cost).

Mix (20k/20k floor) is a tuning knob — A/B it if prose acceptance drops.

## 7. Measurement plan

Probe classes (extend docs/69 step 0's three): **reasoning / prose / code /
tool-call-JSON** (agent-style: fixed OpenAI tool-calling request with
schemas, ≥2k generated tokens, greedy).

Per run, record (serve.log): t/s, acceptance (acc stats),
`vocab_coverage` (B), n-gram hit rate + chain lengths (docs/69 C3),
`nvidia-smi` clocks. Commit to `results/`.

Primary diagnostic = **coverage %** and **rejection breakdown**:
- target token outside active draft vocab → vocab problem (§6)
- in-vocab, MTP wrong but pool had the token → pool/depth problem (docs/69)
- in-vocab, nothing had it → head quality (out of scope here; DFlash2
  territory)

Losslessness gates unchanged: T10 determinism, battery, OFF ⇒ bit-identical.

## 8. Synergy + interaction with docs/69

- docs/69 changes **nothing**: this doc feeds it (seed file = one flag;
  vocab rebuild = one data file) and its step-0 probes gain the
  tool-call-JSON class (§7).
- **The vocab is what makes long drafts affordable.** docs/48: draft head
  GEMV is not the bottleneck at k=3 (MTP attention/FFN is). With adaptive
  depth up to 12, up to 12 MTP steps/round each re-score the head → the
  40k slice (3.28 ms vs 12.46 ms per docs/14) stops being optional.
  So: project vocab (this doc) and adaptive depth (docs/69) need each other;
  the pool (docs/69) makes the long drafts *accurate*.
- Ordering: this doc's pieces A + B can land **before or independently of**
  docs/69. Recommended: A first (offline, zero risk), B second (small
  server hook), then docs/69 with seed flag, then §6 rebuild + re-enable
  draft vocab with the coverage gate.

## 9. Open questions / risks

1. **Tokenizer fidelity** — collector must use the exact Qwen3.8 BPE.
   Mitigation: export from engine `Tokenizer`, round-trip validation,
   spot-check known strings → ids; domain check parity with `bindings.cpp`.
2. **Budget displacement** — corpus tokens evict baseline tail tokens;
   prose acceptance could dip. Mitigation: 20k baseline floor (§6.4) + A/B.
3. **Addressing vs coverage (Oilbird, §4.4)** — exact n-gram matching
   under-hits on JSON/tool-call traffic even when the token is "in the
   pool". Expect pool hit rates on tool-call probes to be lower than on
   prose/code; if the class under-performs, options: (a) schema-aware
   seeding (ToolSpec-style), (b) value-masked n-grams (hash n-gram with
   leaf values masked), (c) defer that class. Decide with data, §7.
4. **Pool persistence** — process RAM only today; agent sessions that
   restart lose the live dictionary. Export/import is a small follow-up.
5. **Multi-project / multi-user serve** — one shared pool + one active
   vocab per server. Fine for our single-project box; if we ever serve
   multiple projects, vocab is per-branch file (already so) but pool
   sharing becomes cross-contamination → revisit then.
6. **Stale corpus** — rebuild after significant changes; git hook is a
   follow-up, not phase 1.

## 10. Phase plan

| Phase | Scope | Effort | Output |
|---|---|---|---|
| 1 | Collector (A): script + tokenizer export + validation + `draft_vocab_project.json` + pool stream on main branch | ~1–2 days | artifacts in `build/collect/`, coverage of main corpus measured |
| 2 | Passive counter (B) + coverage log line in serve; OFF by default, bit-identical gate | ~1 day + battery | `vocab_coverage` in serve.log, counts files accumulating |
| 3 | docs/69 step 3 gains `--ngram-mod-seed`; load corpus stream at startup | rides docs/69 | seed flag + probe hit-rate with/without seed |
| 4 | Rebuild vocab with corpus + output counts; A/B: `--no-draft-vocab` vs project vocab on the 4 probe classes; re-enable if coverage ≥99% and t/s ≥ baseline | ~1 day | t/s + acceptance + coverage table in `results/` |

Each phase = its own small work order when started (template: docs/99),
worktree + branch per house protocol.

## 11. References

- Internal: docs/13 (draft vocab objective), docs/14 (80.55 t/s run),
  docs/48 (coverage root cause — the number that motivated this),
  docs/69 (ngram-mod + adaptive port), docs/12 §3.1/3.6 (catalog),
  docs/99 (work order template).
- External: llama.cpp PR #19164 (ngram-mod) + `common/ngram-mod.h`;
  llama.cpp PR #27210 (adaptive MTP, docs/69); arXiv 2503.03434 (RASD);
  arXiv 2503.07807 (domain draft models); arXiv 2608.03839 (Oilbird);
  arXiv 2604.13519 (ToolSpec); vLLM N-Gram Speculation docs (v0.27).
