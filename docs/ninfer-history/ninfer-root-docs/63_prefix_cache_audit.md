# 63 — Prefix-cache correctness audit (llama.cpp #24891 checklist)

Status: DONE 2026-08-24 (static audit, read-only). Roadmap item #2, first of
the three audits. Findings: 5/5 checklist items PASS; one low-priority edge
case registered (§3); battery test T17 added (§4). No code changes required.

Checklist source: docs/58 §2 (llama.cpp PR #24891 — checkpoint invalidation
after tool requests, accepted as audit only). Audit scope: `tp2_backend.cpp`
per-request prefix handling, `PagedKVPool`/`PagedKVAllocation` lifecycle.

## 1. Item-by-item results

| # | Check | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | True token-match prefix never shrunk by a stale checkpoint value | **PASS** | `prefix_len` is re-measured token-by-token every request (`tp2_backend.cpp:757-761`, compares `req.prompt_tokens` vs `st.cached_tokens`). No cached/stale length bounds or overrides it. The only shrink (partial match → 0, L763-769) is a correctness requirement (GDN state cannot be partially restored), not staleness. |
| 2 | Clamp restored values to what actually exists; zero on reset | **PASS** | Full restore implies `prefix_len == st.cached_tokens.size()` (partials are zeroed first) and the match loop guarantees `≤ plen`. GDN: full `copy_slot` or full `zero_slot` + all other slots zeroed (L771-790). KV: **pages are reserved once at startup with full entitlement** (`tp2_backend.cpp:259`) and only re-published per request (`publish_mapping`, L791) — there is no between-request eviction; admission control rejects prompts that don't fit, so a restored prefix's pages are always resident. `t0_token` restored from `st.cached_t0_token` (L872). |
| 3 | Erasure decisions use physical prompt length, not a corrupted position | **PASS** | No partial erasure exists: `st.cached_tokens = req.prompt_tokens` replaces the whole vector with the current request's real tokens (L865); `cached_ph` is updated for `[prefix_len, plen)` from the fresh prefill (L849-856), and `[0, prefix_len)` is retained because those tokens are identical by construction of the match. |
| 4 | Generation-phase checkpoints must not evict input-phase ones | **PASS (note)** | `cached_tokens` is set to the **prompt only** at end-of-prefill — decode output is never appended, so generated text cannot displace or extend the reusable prefix. Cross-turn and tool-round reuse works: the next request's history starts with the previous prompt → full match → restore, re-prefill from `plen`. Note: this is a **single most-recent-prompt cache** (one backend state); concurrent different clients get last-writer-wins — performance loss only, correctness preserved (mismatched tokens just re-prefill). |
| 5 | Sequence removal must fail soft, never crash | **PASS + finding** | No per-session teardown exists to crash: `st` is global backend state; a client disconnect mid-request leaves no dangling handle. **Finding F-63-1 (§3):** an abort mid-decode can leave stale KVarN workspace state that the full-restore path does not reset (re-prefill path does, since 2026-08-24). |

## 2. Why the D-01 family is NOT this failure class

The warm/cold divergence (D-01) reproduces with identical prompts and no
tools — items 1-5 all hold in that shape, so #24891's invalidation class is
excluded as its cause. D-01 remains open on its own investigation (needs the
live server; deferred until KVarN lands).

## 3. Finding F-63-1 — registered as **D-17** in docs/50 (2026-08-24)

**Aborted request + subsequent full-match restore can carry stale KVarN
workspace state.** Sequence: request A decodes mid-page (partial tile
resident), client disconnects; request B's prompt fully matches A's cached
prompt → restore path appends from `plen` without resetting the workspace →
possible "jumped pages" throw or a commit quantizing a tile with foreign
slots. The re-prefill path is safe (reset added 2026-08-24); only the
full-restore path is exposed, and only when KV storage is KVarN. Fix options
when picked up: reset workspace on every request start (cheap; costs one
branch), or track workspace owner sequence id. **Not fixed now** — requires
the live server for a real repro (test-first policy), which the KVarN agent
owns.

## 4. Battery test T17 (added to `tools/smoke/test_serve_correctness.py`)

Per docs/58 §2 action item: "long prompt → tool call round → continuation
reuses prefix (speed check)". Multi-turn + tools is the repro shape for this
failure class. Test runs after KVarN lands (needs a stable server); expected
result: PASS with `prefix hit` log line on the continuation and prefill time
on the tail only.
