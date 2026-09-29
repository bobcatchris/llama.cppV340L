# Document 38: v1 Musts + Post-v1 Pipeline

Date: 2026-08-21. Owner decisions: **quantized KV cache and prefix caching are v1 musts**
(prefix cache **on by default**, matching llama.cpp server behavior). Advanced features
(DFlash, KVarN) enter the post-v1 pipeline. Supersedes the "WI-5 only if >64k ctx"
triage from doc 36/37.

## 1. What "v1" means

A **stable, usable model server** for this one model (Qwen3.8-27B) on 2× 5060 Ti:
multi-request without crashes, full sampling, quantized KV, prefix caching on by default,
64k context, no regressions. After v1, the owner runs daily chat + pi against it.

## 2. v1 musts (work items)

### M1 — Multi-request stability ✅ DONE (verify)
- `f633644b` (WI-1): persistent + staging arena scopes per request; per-token staging
  allocs removed (per-step slices, matching the driver); rope positions passed correctly.
- `4d6d1d08`: per-request rank state reset — `reset_one_shot_step`, `lengths`,
  `accepted`, `anchor` all reset; needed for *correctness* of request N>1, not just
  crash-freeness.
- **Gate**: 5× 2k-token serve requests all 200 + 10-request mixed smoke + battery green.

### M2 — Full per-request sampling (WI-4b) — IN PROGRESS
- All 7 attributes (temperature, top_k, top_p, min_p, presence/frequency penalties, seed)
  flow schema → request → translate → engine → backend `host_cfg` (**chain verified in
  code**, min_p added in `4d6d1d08`).
- `68e2c6bc`: top_k=1 → argmax branch in the sampling finalize kernel (top_k=1 ≡ greedy,
  consistent with the engine defaults temp=0/top_k=1).
- Remaining: TC-S battery case, G1–G8 gates, priority check (per-request overrides must
  beat the hardcoded `sampling_defaults_` — verify in G2), penalty path through the accept
  kernel.
- **Gate**: G1–G8 + TC-S in battery (always-run), greedy determinism untouched.

### M3 — Option plumbing (WI-3)
- draft-tokens flag, max_context preflight (400 with clear message), LOOKUP flag,
  stop tokens from config.

### M4 — Quantized KV cache (I8) — PROMOTED to v1 must
- Owner: "kv cache is very important". Wire pool planes to dtype; I8 prefill kernel
  already exists (`gqa_attention_prefill_i8.cuh`), decode launcher already checks I8.
- Enables 64k–200k ctx (6.4 GB→3.2 GB/rank at 200k) and halves attention KV bandwidth.
- Also the foundation for advanced KV work later (KVarN).
- **Gate**: 64k-ctx serve request with I8 KV, quality spot-check (same prompt, I8 vs BF16
  outputs should be near-identical for reasonable-length generation), battery green.

### M5 — Prefix caching, ON BY DEFAULT — NEW (WI-6)
llama.cpp server: KV persists across requests per slot; longest common prefix matched;
off only via `--no-cache-reuse`. We are FullReset-everything today.

**Design (design doc first, like TP2's):**
1. **KV page retention**: on request completion, retain KV pages on a hit-eligible LRU
   list (cap: N slots × L tokens, VRAM-budgeted; evict LRU on overfill).
2. **Prefix matching**: hash new prompt token IDs in 64-token page units; longest
   matching page prefix against the cache. Qwen chat template is deterministic →
   system+history prefix is byte-identical turn-to-turn (the common case hits).
3. **GDN state restore (the hard part)**: 48 linear-attention layers hold recurrent
   state, not pages. A retained cache slot = {KV pages, per-layer GDN state snapshot at
   boundary L, L}. On hit: slot-copy into the working slot — the MTP slot protocol
   (`copy_slot`/rebase) is the working precedent; extend it from MTP slots to cache
   slots. No GDN replay (defeats the purpose).
4. **Protocol**: single-flight server → cache is an LRU of ≤N slots (default N=4).
   Hit → continue from L; miss → FullReset as today. Flags: `--prefix-cache` (default
   **on**), `--prefix-cache-slots N`.
- **Estimate**: design doc 1 day; KV half easy (pages exist); GDN snapshot/rebase is the
  risk — **3–7 days** total.
- **Gate**: 2-turn serve test — turn 2 shares ≥90% of prompt; turn-2 effective prefill
  wall-time proportionally reduced (2k-token prefill 62 ms → <20 ms); **token-identical
  to FullReset greedy run** (correctness); battery green.

### M6 — Batched/chunked prefill (WI-4) — PROMOTED to v1 must
Prerequisite for 64k context AND makes prefix-miss suffixes cheap. Target pp ≥ 300 t/s
(T=512 chunks, TP batch GDN + batch attention). 1–3 days.

### M7 — Standing gates (after every work item)
Battery (incl. TC-S), greedy determinism, A2 identity, serve smoke: 10 mixed requests
(sizes + streaming + one tools call), cancel-on-disconnect, doc 37 verify items.

## 3. v1 Definition of Done (all green)

1. Battery all PASS incl. TC-S; t/s ≥ baseline (93.51) — no regression
2. 10 consecutive mixed-size serve requests, all 200, no leak (VRAM flat across requests)
3. Sampling G1–G8 green
4. Prefix hit demonstrated (2-turn test, gate above)
5. 64k context request completes end-to-end (I8 KV + batched prefill)
6. Determinism (greedy bit-identical) intact
7. Docs: results doc + this doc updated

## 4. Post-v1 pipeline

### v1.x (small, from doc 37)
| Item | Effort |
|---|---|
| `stop` custom strings (host-side match in decode loop) | ~1 h |
| `logit_bias` (4-row logit adjustment) | ~2 h |
| `/v1/completions` | ~1-2 h |
| `/tokenize` + `/detokenize` | ~1 h |
| `/metrics` (Prometheus) | ~0.5 day |

### v2 — Advanced (owner-requested) — NOW SPECIFIED (2026-08-21)

**A1. DFlash / DFlash 2 — block-diffusion parallel drafter**
- Repo: `https://github.com/z-lab/dflash`. Paper: arXiv 2602.06036. A lightweight
  **block diffusion** model that drafts a whole block in parallel (non-causal,
  bidirectional cross-attention over the cached context) — the block-parallel
  analogue of our MTP chain. Catalog item 3.4 (doc 12) is confirmed.
- **A DFlash 2 checkpoint for our exact target exists: `z-lab/Qwen3.8-27B-DFlash2`.**
  Reference integration runs `num_speculative_tokens: 15`.
- Backends: vLLM, SGLang, **llama.cpp (PR 27342)**, Transformers, MLX — i.e. the
  drafter is a standard small model; the serving side needs the parallel-verify +
  non-causal-cross-attention plumbing. That plumbing is the engine work: block
  proposals verified against the target, non-causal attention over the cached
  context, commit-only-accepted protocol (same invariant our MTP slot protocol
  already enforces).
- The >90 → >120 t/s bet on 5060 Ti; carries to V340L (doc 13). Own design doc +
  drafter artifact pipeline (convert Qwen3.8-27B-DFlash2 to our artifact format,
  sharded like the MTP head).

**A2. KVarN — variance-normalized KV-cache quantization**
- Repo: `https://github.com/huawei-csl/KVarN` (Apache 2.0, vLLM fork). Paper:
  arXiv 2606.03458 ("Mitigates Error Accumulation in Reasoning Tasks").
- Algorithm, per fixed token tile (group 128, 64 supported):
  1. Hadamard **rotation along channels** (orthonormal → attention scores
     preserved; spreads outliers)
  2. **Iterative variance normalization** (Sinkhorn-like: alternate row/column
     std-normalization in log space)
  3. **Asymmetric round-to-nearest quantization: K = 4-bit (per-channel scales),
     V = 2-bit (per-token scales)** — `kvarn_k4v2_g128`
  - Calibration-free, plug-and-play. Claims: **3–5× KV capacity vs FP16,
    throughput ≥ FP16 (up to ~1.3×), FP16-level accuracy** (AIME25 parity on
    Qwen3-32B; GLM-4.7-Flash MLA: 2.77× capacity at accuracy parity).
- **Explicitly supports our exact configuration** (verified in its README):
  - **Hybrid models** (Mamba/linear-attention + full-attention): compresses only
    the full-attention layers — that is our 16-attn/48-GDN topology
  - **MTP speculative decoding** — same commit-only-accepted invariant as ours
  - Weight-quant checkpoints (AWQ INT4) — composes with our Q4_1 weights
- Porting path to our engine: **M4 (I8 KV) is the foundation** — same pool-plane
  plumbing, same dequant-on-read attention pattern. KVarN adds: tile-rotate +
  normalize kernels on the write path, K4/V2 packed storage, dequant in the
  attention kernels (decode + prefill + verify). The Hadamard + Sinkhorn steps
  are per-tile O(C·T) work — batchable, calibration-free, no weight changes.
- Synergy with M5 (prefix caching): cached slots live in the KV pool; K4V2 fits
  **~4× more cached tokens** in the same VRAM — prefix cache depth scales with
  the capacity win.
- Estimate (after v1, design doc first): **3–5 days** (kernels: rotate/normalize/
  quant pack + dequant in 3 attention paths; correctness gate: I8-vs-KVarN-vs-BF16
  quality comparison on fixed prompts + battery).
3. Concurrency / continuous batching (multi-turn) — months-class; only if real contention.
4. `logprobs` surface, structured output / grammar — only if a consumer appears.

### Out of scope (owner-confirmed)
Web UI. llama.cpp slot/websocket protocol. Multimodal, LoRA, model reload, sleep/wake.

## 5. Work order (current)

1. **Finish M2** (WI-4b: TC-S + G1–G8 + priority check) — in progress
2. **M3** (option plumbing, ~1 h)
3. **M4** (I8 KV, 1–2 days)
4. **M6** (batched prefill, 1–3 days) — design-first if chunk scheduling is non-trivial
5. **M5** (prefix caching, 3–7 days) — design doc first
6. v1 DoD run → results doc → v1.x small items in any order

## 6. Code review — latest commits (2026-08-21)

**`f633644b` (WI-1 arena scope)** — ✅ APPROVED.
Scope-based rewind of `persistent` + `staging` at request entry; per-token staging
allocs replaced by per-step slices (matches driver pattern); `rope_positions` now
actually passed (was `pos` reused). Token host-copy happens inside the scope. Residual:
confirm no result buffer outlives the scope (battery will catch this as garbage tokens).

**`68e2c6bc` (top_k=1 argmax branch)** — ✅ APPROVED.
top_k=1 ≡ argmax target ≡ greedy proposal tokens → all-accept path. Consistent with
engine defaults (temp=0, top_k=1). No effect on the greedy fast path.

**`4d6d1d08` (min_p + rank state reset)** — ✅ APPROVED.
min_p chain complete end-to-end (verified: schema → `SamplingParams` → translate →
engine → `host_cfg` line 708). Penalties → `token_counts_buf` wired (lines 712–716).
Per-request reset (`reset_one_shot_step`, `lengths`, `accepted`, `anchor`) is the
multi-request *correctness* fix — good catch by the agent; WI-1 alone would have left
request N>1 producing garbage even without the leak.
**Flag for agent**: verify per-request sampling overrides take priority over the
hardcoded `sampling_defaults_` (tp_engine.cpp:197–202) — the G2 TC-S test (sampled
output differs from greedy same-seed) is the empirical check; make it explicit in the
test notes.

**Process note**: the agent's in-flight WI-4b lines were swept into doc commit
`0b05f921` (assistant `git add -A` error). Nothing lost; subsequent agent commits are
correctly scoped.

## 7. Test plan for the new additions

Two levels: **driver-level** (new `TC-*` cases in `tp2_decode.cpp`, run by
`verify_battery.sh`, GPU-only, no HTTP) and **serve-level** (new `serve_battery.sh`,
HTTP against a live server, mirrors the doc 37 smoke items). Every case: fixed seed,
fixed prompts in a committed fixture file, logs under `/home/intel/verify_logs/`.

### Driver-level (battery `TC-*` cases)

| Case | What | Gate | Notes |
|---|---|---|---|
| `TC-S` (in flight) | sampling vs greedy | doc 36 G1–G8 | always-run |
| `TC-M` multi-request | 5× large prompt, one process | all complete; VRAM flat between requests (pool-usage query, not nvidia-smi); request N output token-identical to its single-run baseline | validates M1 (arena scope + rank reset) at driver level; catches use-after-scope as garbage tokens |
| `TC-KV` I8 KV | greedy decode, I8 vs BF16 pool | (a) I8 run completes, no NaN; (b) MTP acceptance within 3% of BF16 baseline; (c) pool footprint ≈ half of BF16; (d) I8 run **bit-identical to itself** (determinism under I8) | NOT required to match BF16 tokens (quant delta expected); gate (b) is the quality proxy |
| `TC-X` batched prefill | full prompt, T=512 chunks vs T=1 path | (a) chunked path **deterministic** (2 runs bit-identical); (b) **first token identical** to T=1 path; (c) later-token divergence < 0.5% of positions; (d) pp ≥ 300 t/s | (c) is the honesty gate — BF16 reassociation in chunked attention sums means bit-identity ACROSS paths is NOT achievable and must not be gated |
| `TC-P` prefix cache | prompt P (2k), then P+suffix; plus FullReset control of P+suffix | (a) cache-hit run **token-identical to FullReset control** (exact KV/GDN restore ⇒ bit-identity IS required here, unlike TC-X); (b) turn-2 effective prefill wall-time ≤ 25% of turn-1; (c) eviction: fill past cap, re-send P — token-identical via re-prefill, no crash; (d) partial match (common prefix = first 1k only) hits at 1k boundary, output identical to FullReset | the v1 crown-jewel test; (a) correctness invariant, (b) value proof |

### Serve-level (`serve_battery.sh`)

| Case | What | Gate |
|---|---|---|
| `S1` mixed traffic | 10 requests: 3 small, 2× 2k, 1 streaming, 1 tools call, 1 multi-turn short-suffix, 2 small | all 200; streaming yields N chunks; tools call parses; request log JSONL has 10 entries |
| `S2` prefix multi-turn | turn 1 (2k prompt) → turn 2 (same 2k + 200 new) | turn-2 TTF ≤ 25% of turn-1 TTF; turn-2 output identical to a fresh FullReset request resending full history |
| `S3` sampling via API | same seed ×2; different seed ×2; temp=0 vs absent | same-seed byte-identical; diff-seed differ; temp=0 == greedy baseline |
| `S4` cancel | streaming request, kill client at ~50% | server log shows cancellation; next request clean; no GPU hang |
| `S5` VRAM flatness | 10× 2k requests, sample pool usage after each | max−min ≤ 1% of pool (no leak) |
| `S6` 64k context (post M4+M6) | 64k-token prompt + 100-token generation | completes; prefill consistent with pp ≥ 300 t/s; no OOM with I8 KV + prefix cache enabled |

**Ordering:** `TC-M` now (validates the already-landed M1 work). `TC-S` ships with
WI-4b. `TC-KV` + `TC-X` when M4/M6 land. `TC-P` when M5 lands — the M5 design doc
must include the TC-P harness as an acceptance deliverable. Serve: S1/S3/S5 after
WI-4b, S4 after M1, S2 after M5, S6 last.
