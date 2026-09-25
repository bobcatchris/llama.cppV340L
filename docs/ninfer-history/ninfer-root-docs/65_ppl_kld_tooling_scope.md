# 65 — Perplexity / KLD tooling: scope

Status: SCOPE — 2026-08-24, no code changes. Roadmap item #6. This becomes
the permanent quality gate for all-Q4 (#7), KVarN P4, and DFlash diagnostics.

## Why we need it (user-approved use)

1. **KVarN acceptance beyond "it fits"**: llama.cpp issue #24139 (Anbeeld's
   KVarN in BeeLlama.cpp) shows the method holds quality — PPL wikitext-2:
   kvarn4-kvarn4 ≈ f16 (ΔPPL ~0.001–0.009) vs q4_0 (+0.012–0.025) on
   Qwen3.6-27B; KLD shows kvarn8 ≈ q8_0 at the same size. We need OUR numbers
   on OUR model: KVarN k4v2 vs int8 vs bf16 KV, same weights.
2. **All-Q4 validation (#7)**: full-Q4 artifact quality delta is unmeasured
   (docs/33 explains why selective q5 was the default); PPL/KLD vs the Q5
   reference is the gate before we ship it.
3. **DFlash diagnostics**: draft-vs-target divergence per position is a
   KLD-shaped metric; the endpoint makes it measurable.

## Current state (verified)

- `top_logprobs` is parsed but hard-rejected:
  `responses_schema.cpp:648-651` → `bad_request("top_logprobs is not
  supported", ..., "logprobs_not_supported")`. The schema hook exists; no
  backend.
- TP2 logits are sharded: lm_head is `TpRole::ColumnN`, vocab 248320 split
  124160/rank (tp_load.cpp); greedy uses fused `allreduce_argmax` exchanging
  only (argmax_id, score) — zero logit traffic.

## Design

### Endpoint: `top_logprobs` support on /v1/chat/completions + /v1/completions

OpenAI-compatible: request `logprobs: true, top_logprobs: N` → per-token
`{token, logprob}` for the actual token plus `top_logprobs: [{token, logprob}]`
(N candidates). **Distributed top-K (no full-vocab gather):**

1. Per rank, over its 124160 local logits: compute local max and local
   sumexp (bf16→fp32 in the epilogue; one small allreduce of 2 floats/token
   for the global normalization constant).
2. Each rank computes exact global logprobs for any token in its half using
   the shared normalization; selects its local top-N.
3. Exchange N (id, logprob) pairs per rank (tiny), merge → global top-N.

Cost per token: ~2 floats allreduce + 2N float exchange vs ~500 KB full
gather — negligible at decode rates. For PPL-only use (logprob of the actual
next token only) step 3 degenerates to a 1-float exchange.

**Scope cut:** implement on the DECODE path first (PPL/KLD are decode metrics);
prefill logprobs later if ever needed. Non-greedy sampling unchanged (this is
read-only over logits).

### Client: `tools/eval/ppl_kld.py`

- **PPL mode**: feed a reference corpus (wikitext-2 test split, same as
  llama.cpp #24139 for comparability) token-by-token with fixed history;
  geometric-mean exp(−logprob(actual)) over all positions. Report per-context
  bucket (e.g., PPL at 1k/8k/32k/128k positions) — KV quantization error
  grows with context, so the number that matters is long-context PPL.
- **KLD mode**: two endpoints (A = reference, B = candidate): sample/generate
  from A on a fixed prompt set, evaluate B's logprobs at A's tokens;
  KLD = mean over positions of Σ_{t∈top-K(A)} p_A(t)·(log p_A − log p_B) with
  the tail mass folded into one bucket (document the approximation; K=64).
- Output: JSON + markdown table, committed under `results/` (measurement data
  is project data).

### Acceptance anchors (first run, before any feature work uses this)

| Run | Expectation |
|---|---|
| bf16 KV vs int8 KV, same weights, 32k context | ΔPPL < 0.05 (sanity: our I8 is close to lossless at these sizes — verify, don't assume) |
| KVarN k4v2 vs int8 KV, 32k + 128k context | ΔPPL within the #24139 envelope (~0.01–0.03); this is the KVarN P4 quality gate |
| all-Q4 artifact vs Q5 reference | measured; go/no-go for #7 |

## Effort estimate

- Server endpoint (decode path + TP collectives + schema + streaming shape): ~1–2 days.
- Script + corpus setup: ~half day.
- First baseline runs + doc: ~half day.
Total ~2–3 days, single agent, needs the live server (queue behind KVarN).

## Ordering note

Roadmap keeps this at #6 (after single-GPU #5) — but if the KVarN work order
finishes early, pulling this forward is justified: it is the quality gate for
the two biggest pending items (#7 all-Q4, KVarN P4) and costs nothing on the
VRAM side.
