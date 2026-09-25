# Document 37: Feature Parity Matrix — NInfer TP2 Serve vs llama.cpp / vLLM (single model)

**Status:** ARCHIVE

Date: 2026-08-21. Verified against serve code at `e0e0222e` (plus WI-4b as scoped in doc 36).
Scope: serving **this one model** (Qwen3.8-27B). "Stock" = upstream single-GPU serve stack;
"TP2" = our TPEngine path.

Legend: ✅ have & honored · ⚠️ parsed but NOT honored (silent no-op) · ❌ missing · 🚫 out of scope for this use

## 1. Request parameters (chat completions)

| Parameter | Stock wire | TP2 honors | Notes |
|---|---|---|---|
| `model`, `messages` | ✅ | ✅ | chat template from artifact |
| `temperature`, `top_p`, `top_k` | ✅ | ❌ (WI-4b) | sampler clamps top_k to 20 — the owner's value of 20 sits exactly at the supported bound |
| `min_p` | ⚠️ **not parsed per-request** (server preset only) | ❌ | small gap: add to request parse (~10 min, fold into WI-4b) |
| `presence_penalty`, `frequency_penalty` | ✅ | ❌ (WI-4b) | sampler op natively supports via device `token_counts` buffer |
| `seed` | ✅ | ❌ (WI-4b) | |
| `logit_bias` | ⚠️ **documented silent no-op in stock** (`request.h`: "parsed for wire compatibility... does not affect generation") | ❌ | fill-later: 4-row logit adjustment before accept, or keep documented no-op |
| `n > 1` | ✅ rejected with clean 400 | ✅ | honest rejection, not silent |
| `max_tokens` | ✅ | ✅ | server default 512 |
| `stop` (custom strings) | ⚠️ **parsed (`stop_strings`) but consumed nowhere in serve or runtime** | ❌ | Qwen's real stops (im_start/im_end) are template-handled, so the common case works; arbitrary strings do not. Fill-later: host-side string match in the decode loop (~1 h) |
| `logprobs` / `top_logprobs` | ❌ not parsed | ❌ | sampler op doesn't surface logprobs → moderate work. Fill-later (eval tooling only) |
| `response_format` / json_schema / guided decoding | ⚠️ only `{type:"text"}` accepted; json_schema → clean 400 `response_format_not_supported`. **No grammar engine anywhere in runtime** | ❌ | biggest algorithmic gap vs vLLM. pi uses `tools`/`tool_calls` (which work), not json_schema. Out of scope now (weeks-class: xgrammar/outline integration) |
| `tools`, `tool_choice` | ✅ | ✅ | schema + tool_call_parser inherited from stock; round-trip through TPEngine to be smoke-tested |
| `reasoning_effort`, `enable_thinking`, `preserve_thinking` | ✅ | ✅ | |
| `chat_template_kwargs` | ✅ | ✅ | |
| `stream_options.include_usage` | ✅ | ✅ | usage in final stream chunk |

## 2. Response fields

| Field | Status |
|---|---|
| `usage` (prompt/completion tokens) | ✅ non-streaming + streaming final chunk |
| `finish_reason` | ✅ — "stop" fires only on the hardcoded Qwen tokens, not custom `stop` strings |
| `logprobs` | ❌ |
| Per-request timing stats (llama.cpp `eval_time` style) | ❌ (llama.cpp-proprietary, not OpenAI — skip) |

## 3. Endpoints

| Endpoint | Have? | Triage |
|---|---|---|
| `/health` | ✅ | |
| `/v1/models`, `/v1/models/{id}` | ✅ | |
| `/v1/chat/completions` | ✅ | |
| `/v1/messages` (+ `count_tokens`, Anthropic-native) | ✅ | ahead of llama.cpp |
| `/v1/responses` (+ CRUD, `cancel`, `compact`, `input_items`, `input_tokens`) | ✅ | on par with latest OpenAI, ahead of llama.cpp |
| `/v1/completions` (plain-text completions API) | ❌ | fill-later (~1-2 h, shares the chat pipeline); some tooling hits this |
| `/tokenize`, `/detokenize` | ❌ | fill-later (~1 h; tokenizer already in the frontend) |
| `/metrics` (Prometheus) | ❌ | fill-later (~0.5 day; stats infra exists); ops value |
| `/bench`, `/perf` (llama.cpp) | ❌ | skip — the battery is our bench |
| `/slots`, `/props`, websocket `/completion`/`/slot` (llama.cpp) | ❌ | 🚫 llama.cpp-proprietary slot protocol; irrelevant to OpenAI clients |
| `/v1/models/reload`, vLLM sleep/wake | ❌ | 🚫 single-model box; restart the server |
| `/v1/embeddings`, `/v1/rerank`, `/v1/score`, audio | ❌ | 🚫 not in this artifact |

## 4. Behavior / operational

| Capability | Status | Notes |
|---|---|---|
| Concurrency | ❌ single-flight (1 request at a time) | biggest behavioral gap vs llama.cpp `-np` / vLLM batching. Multi-turn prefix caching + continuous batching = months-class. For one human + one agent: not a practical constraint; revisit only if real contention appears |
| Prefix caching / `cache_reuse` | ❌ | FullReset per request → re-prefill full history each turn. llama.cpp server has the same default (no prefix caching); vLLM does. Combined with WI-4 this is acceptable for typical pi turn lengths |
| Cancel on client disconnect | ✅ designed (stock passes `is_connection_alive` into the cancellation view) | **verify empirically in serve smoke** — abort a streaming curl mid-generation, confirm GPU work stops |
| Auth (API key; OpenAI + Anthropic header styles) | ✅ | |
| CORS | ✅ | |
| Request logging (JSONL) | ✅ | richer than llama.cpp |
| Response store (replayable records, size-capped) | ✅ | neither llama.cpp nor vLLM has this |
| Quantized KV (q4/q8) | ❌ → WI-5 (I8) | needed only >64k ctx |
| Long context | ❌ → WI-4 + preflight | 8k now, ~64k, 200k after WI-4+WI-5 |
| Multimodal / LoRA | 🚫 | not in this artifact / single model |
| Web UI | 🚫 (owner: not concerned) | |

## 5. Where we are ahead

- MTP speculative decode served as a first-class feature at ~94 t/s (llama.cpp draft-mtp ~70 t/s on this hardware; vLLM MTP for Qwen on sm_120 is unproven)
- Anthropic-native `/v1/messages` (llama.cpp doesn't have it)
- Full Responses API incl. cancel/compact (on par with latest OpenAI, ahead of llama.cpp)
- Built-in response store
- Deterministic greedy bit-identity as a testable property — nothing else offers it

## 6. Triage summary (single-model context)

**Fold into current batch:**
- `min_p` per-request parse → add to WI-4b (10 min)

**Fill-later list (each small, none block pi or daily chat):**
1. `stop` strings: host-side match in decode loop (~1 h)
2. `logit_bias`: 4-row logit adjustment kernel (~2 h)
3. `/v1/completions` (~1-2 h)
4. `/tokenize` + `/detokenize` (~1 h)
5. `/metrics` Prometheus (~0.5 day)

**Out of scope (large projects, no current consumer):**
- `logprobs` surface (moderate, eval-only consumer)
- Structured output / json_schema / grammar engine (weeks)
- Concurrency / continuous batching / prefix caching (months)
- llama.cpp slot protocol, web UI, reload, sleep/wake

**Verify (no code, smoke-test items):**
- Cancel-on-disconnect actually stops GPU work in the TP2 path
- `tools`/`tool_choice` round-trip end-to-end through TPEngine (parser is stock; engine side is new)
