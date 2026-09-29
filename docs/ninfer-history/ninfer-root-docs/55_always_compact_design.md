# 55 — Always-Compact: context overflow + compaction design

Status: LAYER 1 VERIFIED (T14 PASS 2026-08-23, ce7077c5), LAYER 2 IMPLEMENTED (2026-08-23, pending live T15).
Date: 2026-08-23. Register entry: docs/50 D-13.

## Problem

Agent sessions routinely exceed the context window mid-turn (the model keeps
generating past any compaction trigger; observed at 102–108% of a 200k window,
and our pi session hit 83.7k prompt against an 80k window). When that happens:

- The next request's prompt is larger than the window.
- Pi's auto-compaction sends **the whole old portion as one summarization
  request** — which is itself larger than the window → `400` → "Auto-compaction
  failed" → session stuck and growing.

llama.cpp tolerates slightly-over prompts (generation just stops at the
boundary; sessions go over all the time). Our server hard-rejected with 400,
and worse: `--kv-capacity` did not actually size the KV pool — the ring was
always `max_context + mtp_k + 4`, so an 83.7k prompt physically could not fit
an 80k-configured server at all.

Requirement (user): **compaction must always succeed, no matter how far over
the limit the session is.** "This is normal."

## Layer 1 — Tolerance zone (implemented)

`--kv-capacity C` (≥ `--max-context N`) becomes the **physical** pool size and
the hard per-request limit on prompt+output. `N` remains the soft limit for
normal operation.

Changes:
- `tp2_backend.{h,cpp}`: `TpBackendOptions.kv_capacity`; ring sized to
  `max(max_context, kv_capacity) + mtp_k + 4`; `kv_capacity_tokens()` reports
  the real pool.
- `tp_engine.cpp`: capacity resolved in ctor (explicit flag; auto → max_context);
  VRAM preflight validates the *capacity*, not just max_context; admission in
  `prepare()`/`submit()` rejects only when `prompt > kv_capacity`; per-request
  effective limit = `max_context` normally, `kv_capacity` when the prompt
  already overshoots (decode stop + output room follow it).

Effect: a compaction request with prompt P fits whenever `P ≤ C`. With
`--kv-capacity 100000 --max-context 80000`, an 83.7k compaction prompt is
admitted with ~16k output room — the observed failure mode is fixed with zero
client changes. Cost: +20k tokens ≈ +330 MiB/rank I8 KV (fits; preflight
validates).

Cost/limit tradeoff: the tolerance zone exists exactly as far as VRAM allows.
`C − N` = how far over we tolerate. 200k config (`C=N=200k`) has no zone; a
200k-capacity server with `--max-context 160000` would have a 40k zone.

## Layer 2 — `/v1/compact` chunked summarization (spec)

For conversations **larger than the physical capacity** (e.g. 3× over), no
single request can hold them, so compaction must be recursive. Algorithm:

```
input:  messages[] of total length C > target   (target = N − reserve, e.g. 60k)
chunk  = floor(0.65 × N) tokens                 # bounded by WINDOW, not by C
loop while token_count(convo) > target:
    head  = oldest `chunk` tokens of convo, cut at a message boundary
    rest  = the remainder
    summary = generate(summarize_prompt + head)   # request size ≈ chunk ≪ N ✓
    convo = [summary-as-system-or-assistant-note] + rest
return convo
```

Invariants that make it *always* terminate:
1. Every summarization request holds one chunk + instructions ≤ ~0.7·N < N —
   fits by construction regardless of how large C is (this is why the chunk is
   sized from the window, not a fraction of the total).
2. Each pass removes `chunk·(1 − r)` tokens (r ≈ 0.1–0.3 compression) > 0.
3. Summarize from the **oldest** end: recent working context is preserved
   intact; only stale history condenses.

Edge cases:
- Single message longer than `chunk`: hard-split mid-message at a token
  boundary (summarizer tolerates fragments); never block on it.
- `C` grows between passes is impossible — the endpoint holds the snapshot.
- Idempotent-ish: re-compacting an already-compacted convo just condenses the
  existing summary note further.

Endpoint contract (OpenAI-compatible surface, no auth beyond existing):
```
POST /v1/compact
{ "messages": [...],            # full conversation, any size
  "target_tokens": 60000 }      # optional; default max_context − 16384
→ { "messages": [...],          # compacted transcript ≤ target + summary slack
   "original_tokens": N0, "final_tokens": N1, "passes": k }
```

Client wiring (pi): the ninfer provider extension calls `/v1/compact` instead
of pi's built-in single-shot summarization when `contextTokens > window`.
Fallback chain: (a) conversation ≤ C → normal requests + pi's own compaction
work (layer 1); (b) > C → `/v1/compact`; (c) endpoint unavailable → error with
explicit "server capacity" number, not a bare 400.

## Testing criteria

- T14 (battery, in `test_serve_correctness.py`): overshoot prompt at zone
  midpoint against `--max-context 80000 --kv-capacity 100000` → accepted,
  generates; beyond-capacity prompt → clean 4xx naming kv capacity.
  **PASS 2026-08-23** (90,439-tok prompt served, generated as requested;
  125k-tok rejected).
- T15 (layer 2, battery): synthetic ~30k-token conversation → `POST /v1/compact`
  with `target_tokens=10000` → `after_tokens ≤ 10000`, `passes ≥ 1`, first
  message is the summary note, recent tail verbatim-preserved, no 400s.
  Multi-pass termination (32+ passes) is covered by the unit test
  `ninfer_compaction_test` (fake counter/summarizer).

## Layer 2 implementation notes (2026-08-23)

- Endpoint: `POST /v1/compact` `{messages, target_tokens?}` →
  `{messages, passes, before_tokens, after_tokens}`. `target_tokens`
  defaults to `--max-context`.
- Pure loop in `src/serve/compaction.{h,cpp}` (injected token counter +
  summarizer); `GenerationService::compact()` wires the real
  `count_prompt_tokens` oracle and internal greedy summarization requests
  (system prompt + segment, max 4096 tokens, thinking off).
- Chunk sizing uses a measured chars/token ratio from the whole conversation
  (one count call), so no per-message token counting; each chunk is validated
  against the window with one extra count and throws if it cannot fit.
- The summary note is a `user` message prefixed
  `[Earlier conversation, summarized]`; the tail is never touched.
- Loop throws if it does not converge within 64 passes (never returns an
  over-target result silently).
- Regression: T1–T12 unchanged at `C = N` configs (zone width 0 ⇒ identical
  behavior to today).

## References

- llama.cpp: prompt > n_ctx → error; mid-generation overflow → stop at boundary
  (`ctx_shift` optional). Agent flows survive because compaction fits within
  the window — which requires exactly what layer 1 provides.
- Pi compaction: `reserveTokens 16384`, `keepRecentTokens 20000`, single-shot
  summarization of (total − recent) → fails when that portion > window.
