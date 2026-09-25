# Document 36: TP2 Serve — Doc 35 Audit, Code Review, and Worklist

Date: 2026-08-21. Reviewer: assistant (independent of the implementing agent).
Scope: commits `d58e1a40` (P0), `0cf4b177` (P1), `09d1ed84` (P2), `e0e0222e` (fix) on
branch `mtp-perf`, plus doc 35's claims, plus a live serve smoke test performed for this review.

---

## 1. Verdict

The serve integration is **structurally sound and correctly architected** (virtualized Engine,
factory routing, single-flight FIFO, FullReset, thin driver wrapper — stock path untouched).
However it is **NOT production-ready for pi-agent wiring yet**. One critical bug (server hard
crash on the second large request, empirically confirmed) and one structural limitation (T=1
prefill loop → 74 s TTFT on a 2.2k-token prompt) block real use. Worklist in §6: WI-1..WI-3
are ~half a day of fixes; after that, pi test-drive is safe for ≤8k ctx, greedy.

## 2. Doc 35 audit (claim → verification)

| Doc 35 claim | Verdict | Evidence |
|---|---|---|
| Battery PASS (0 fail, 0 warn) | **CONFIRMED** | `/home/intel/verify_baseline.json` auto-updated 2026-08-21 14:15 (only updates on clean PASS): 93.51 t/s, 85.6%, determinism PASS, A2 PASS, draft vocab 40960 PASS |
| Standalone 93.51–94.64 t/s | **CONFIRMED** | Same baseline file (93.51 at this commit; 94.64 earlier state) |
| Served HTTP 94.50 t/s, 99.85% parity | **PLAUSIBLE, not independently re-run** | Smoke test confirms served decode works; no HTTP-throughput re-run performed in this review |
| 87.3% acceptance under serve | **PLAUSIBLE** | Above standalone 85.6% is consistent with prompt mix; not independently verified |
| VRAM 9,059 MiB per rank | **PARTIAL** | 9,059 MB is the *materialized capacity* (weights+state). Actual `nvidia-smi` in use: **12,049 MiB/rank** (adds 2 GB work arena, 32 MB staging, pinned). Headroom per rank: ~4.3 GB |
| "Consolidated template instantiations (instantiate.h) to avoid linker duplicate symbol collisions" (§2.4) | **UNSUPPORTED** | None of the four commits touch any `instantiate.h`. Either misattributed or hallucinated. Harmless to behavior; noted for trust accounting |
| Phases P0–P5 complete | **OVERSTATED** | P0–P2 + fix delivered. P4 parity claims rest on the battery (standalone), not a serve-side benchmark; P5 (pi wiring) is what this doc now unblocks |

## 3. Code review findings

### P0 (`d58e1a40`) — CLEAN
All `Engine` public methods virtualized; `make_engine` factory (single-device → stock `Engine`,
2-device → `TPEngine`); `--device`/`--devices` parsed correctly (comma list, validates, sets
`.device` = first). `GenerationService` changed by 3 lines to call the factory. Stock path
bit-identical. No issues.

### P1 (`0cf4b177`) — EXTRACT CLEAN, ONE CRITICAL BUG
Driver core moved to `src/runtime/tp2/` (backend/request/rounds); `tests/multi_gpu/tp2_decode.cpp`
survives as a thin wrapper, so the battery still builds and runs (verified). GDN slot protocol
and round protocol preserved verbatim.

- **R1 (CRITICAL) — per-request arena leak → server hard crash.**
  `run_tp2_request` allocates per request from `st.persistent` (a bump `DeviceArena`):
  `mtp_ph`/`mtp_mh` BF16 [5120, plen] (≈23 MB each at plen=2250), `mtp_pos`/`mtp_ids` I32.
  There is **no `DeviceArena::Scope`, no `reset()`, no rewind anywhere** in tp2_backend.cpp.
  The persistent arena's headroom beyond setup allocations is small (the 160 MB nominal
  headroom is largely consumed by the 106 MB draft output head and other setup allocations).
  Per-request leak ≈ 46 MB/rank at plen=2250.
  **Empirical proof (this review):** serve smoke test, request 1 (2,250-token prompt,
  8 output tokens) completed in 74.07 s; request 2 (same prompt) → worker thread throws
  `std::bad_alloc` → `terminate called without an active exception` → **server process dead**;
  requests 3–5 connection-refused. Log: `/tmp/serve_test.log`, probe req `/tmp/probe_req.json`.
  Any pi session (multi-turn, growing history) kills the server within 1–2 turns.
  **Fix:** wrap the per-request section in `auto scope = st.persistent.scope();`
  (+ `st.staging` — the prefill loop also leaks 2 tiny allocs per token from the 32 MB staging
  arena, slow but same class). `DeviceArena::Scope` (RAII rewind) already exists;
  single-flight mutex makes this trivially safe. Acceptance: 5× consecutive 2k-token serve
  requests all succeed, `nvidia-smi` stable.

- R8 (INFO) — per-request `cudaMemsetAsync` of the full 461 MB state span (FullReset) costs
  ~1–2 ms/rank. Acceptable; note for the prefix-reuse phase (can be scoped to dirty pages later).

### P2 (`09d1ed84`) — WIRING CORRECT, OPTION HONESTY GAPS
`TPEngine` implements the public surface correctly: FIFO via `shared->mutex` held across the
whole generation (true single-flight), metrics atomics, `FullReset` prefix path, cancellation
checked before/after lock and passed into `run_tp2_request`, stop tokens 151643/151645 (Qwen
im_start/im_end) always appended, `GenerationHandle::Impl` plumbing (engine_handle_impl.h) is
sound. `pimpl.reset()` after extracting token ids is correct.

- **R3 (HIGH) — sampling silently ignored.** `submit()` hardcodes `temperature=0, top_k=1,
  top_p=1.0` regardless of request options; the driver sets a single device `SamplingConfig`
  (greedy) once at setup (`tp2_backend.cpp:468-471`). A user requesting
  `temperature=1, top_p=0.95, top_k=20, min_p=0.0` gets **greedy output with no error and no
  log**. Silent semantic mismatch is the worst failure mode for an API server.
  **Why it is greedy:** inherited from the MTP verify protocol the driver was built on —
  temperature=0 makes the whole speculative path bit-identical (our entire test infra:
  battery determinism, A2 identity, bisection — depends on this), and Objective 5 (doc 23)
  closed with "greedy suffices for the agent workload" (pi runs temp 0). It is not a server
  design decision, and it is the wrong default for a general model server.
  **The fix is much cheaper than originally estimated (this review):** the repo already
  contains both halves of the machinery —
  (a) the full `sample()` op (`include/ninfer/ops/sampling.h`): temperature, top-k, top-p,
  min-p, presence/frequency penalties, seeded RNG;
  (b) a **sampling branch inside the speculative accept kernel**
  (`src/ops/kernel/speculative_round.cuh`): temperature > 0 runs the target-distribution
  correction — accept the draft token with probability p_target(draft) under the truncated
  target distribution, resample from the masked residual on reject. Draft proposal stays
  greedy (one-hot), which is the correct standard scheme (vLLM's MTP approach); output
  distribution is *exactly* the target's, MTP stays at full speed, acceptance rate drops
  modestly with temperature.
  So real sampling is **plumbing, not kernels**: copy the request's sampling params into the
  device `SamplingConfig` per request in `run_tp2_request`, and adapt the test battery (A2
  identity and the determinism check are greedy-only by construction — keep them at temp 0,
  add a seeded-sampling smoke test: same seed → identical tokens, different seed → diverge).
  Estimated ~1–2 days incl. tests. Promoted to WI-4b below.
  **Interim (WI-2, keep):** until WI-4b lands, reject non-greedy requests with a clean 4xx
  instead of silently serving greedy.
- **R4 (MED) — `--draft-tokens` ignored.** `b_opts.mtp_k` hardcoded to 3 whenever speculative
  is enabled. Fix: use `options.speculative.draft_tokens` (clamped to [0,4]; k=4 known-worse,
  doc 22 — reject k>3 with an error or warn).
- **R5 (MED) — no max_context vs VRAM preflight.** KV pool is sized at load
  (`pages = 1 + (cap-1)/64`, `cap = max_context + k + 4`). KV cost ≈ **32 KB/token/rank**
  (16 attn layers, 2 local kv heads, 256 dim, BF16) + 2 KB/token/rank (MTP head). At 8k ctx
  that is 0.26 GB (fine); at 200k ctx it is 6.6 GB/rank → 15.3 GB/rank total vs 16.3 GB
  available — cudaMalloc may "succeed" with zero headroom or fail opaquely at load.
  Fix: preflight estimate at startup; refuse with a clear message including the math.
- R6 (LOW) — stop tokens hardcoded; should come from `generation_config_json` (already loaded).
- R7 (LOW) — `workspace_bytes = 2 GB` hardcoded; draft-vocab path fallback chain includes a
  machine-specific `/tmp/ninfer/` prefix (works here; make artifact-relative or a serve flag);
  `LOOKUP` enabled via env var instead of a flag; `memory_summary()` reports capacity≈used
  (cosmetic).

### Fix commit (`e0e0222e`) — CORRECT
Passes full `FrontendResources` (tokenizer + template) from the TP2 backend to the TPEngine
frontend, so chat templating works identically to stock. `PromptInput` carries
`preserve_thinking` (types.h) and serve passes it through (generation_service.cpp:283) —
`--preserve-thinking` flag works end-to-end.

## 4. Serve operation (answers)

### 4.1 Launch command (canonical)

```bash
/tmp/ninfer/build/apps/ninfer-serve /home/intel/models/qwen3_8_27b.ninfer \
  --devices 0,1 --spec mtp --draft-tokens 3 \
  --max-context 8192 \
  --host 0.0.0.0 --port 8091
```

Startup ≈ 7 s to "listening" (12,049 MiB/rank once up). `--max-context` sets the context
length: the KV pool is sized at load, so it is a **server-startup** setting, not per-request;
requests longer than it get a clean 4xx (`ContextLengthExceeded`).

### 4.2 Options vs the owner's llama.cpp reference command

llama.cpp command: `--ctx-size 200111 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
--temp 1 --top-p 0.95 --top-k 20 --min-p 0.0 -b 4096 -ub 512 -sm tensor -ts 1,1
--presence-penalty 0.0 -n -1 -np 1 --spec-type draft-mtp --spec-draft-n-max 3
--chat-template-kwargs '{"preserve_thinking": true}' --chat-template-file q38.jinja`

| llama.cpp | NInfer serve equivalent | Status |
|---|---|---|
| `--host/--port` | `--host 0.0.0.0 --port 8091` | ✅ works |
| `-sm tensor -ts 1,1` | `--devices 0,1` | ✅ TP2 (this work) |
| `--spec-type draft-mtp --spec-draft-n-max 3` | `--spec mtp --draft-tokens 3` | ✅ works, but value ignored → hardcoded 3 (R4) |
| `--ctx-size 200111` | `--max-context N` | ⚠️ works at 8k; 200k blocked by (a) T=1 prefill (R2), (b) BF16 KV VRAM (needs I8 KV, §5). 64k feasible after R5 preflight |
| `--temp/--top-p/--top-k/--min-p/--presence-penalty` | parsed by serve + wire schema | ❌ **silently ignored, greedy only** (R3). Interim: 4xx (WI-2). Real support: WI-4b (kernels already in repo — plumbing only) |
| `-n -1` | `--default-max-tokens` / per-request `max_tokens` | ✅ (default 512 currently) |
| `-np 1` | single-flight design (`--max-concurrency` accepted but TPEngine serializes anyway) | ✅ equivalent |
| `--chat-template-kwargs '{"preserve_thinking": true}'` | `--preserve-thinking` + per-request flag | ✅ plumbed end-to-end |
| `--chat-template-file q38.jinja` | (template comes from artifact) | ❌ custom template file not supported; stock artifact template used. Likely fine (same Qwen template); only needed if owner's q38.jinja differs |
| `--flash-attn on` | n/a — custom fused attention kernels | ✅ inherent |
| `-b 4096 -ub 512` | n/a user-facing | ⚠️ internal: prefill is a T=1 loop (R2), so no batching exists yet |
| `-ngl 999` | n/a — everything is on GPU | ✅ inherent |
| `--cache-type-k/v q8_0`, `-ctkd/-ctvd q8_0` | `--kv-dtype bf16\|int8` (parse-only) | ❌ **BF16 only, see §5** |
| `--api-key` | `--api-key` | ✅ |

Other available serve flags (stock, work with TPEngine): `--model-id`, `--kv-capacity`,
`--max-pending-requests`, `--pending-timeout-ms`, `--prefill-chunk`, `--log-stats-interval-ms`,
`--max-request-mib`, `--request-log-jsonl`, `--response-store-*`, `--no-thinking`, `--greedy`,
`--seed`, `--frequency-penalty`, `--temperature/--top-p/--top-k/--min-p` (server defaults —
currently ignored by TPEngine, see R3).

### 4.3 KV cache quantization (q4/q5/q8)

- **Current state: BF16 only, in both stock NInfer and TP2.** The `--kv-dtype int8` flag
  parses (`KvCacheStorage::Int8Group64`) but **no engine code consumes the option** — it is a
  dead stub (verified: zero references in `src/runtime`, `src/artifact`).
- **However, the kernel layer is already prepared**: `PagedKVLayerView` carries
  `dtype`/`quant_group`/scale planes, and an **I8 prefill attention kernel exists in the repo**
  (`src/ops/kernel/gqa_attention_prefill_i8.cuh`, decode launcher also checks I8). What is
  missing is only the engine-side pool construction (I8 planes + scale factors per layer) and
  threading the option through. So I8 KV is a **plumbing work item (~1–2 days), not a kernel
  project**.
- **Value analysis**: KV quantization buys VRAM, not decode speed (attention is 0.2 ms = 0.5%
  of verify; dequant overhead immaterial). At 8k ctx, BF16 KV costs 0.26 GB/rank — quantization
  is pointless. At 200k ctx: BF16 = 6.6 GB/rank → infeasible (15.3 GB total); **I8 ≈ 3.5 GB/rank
  → feasible (~12.5 GB total)**. So: I8 KV is the prerequisite for >~100k ctx; q4/q5 KV would be
  further VRAM headroom but no kernel exists and no need until >200k ctx.
- **Recommendation**: don't do KV quantization before WI-4 (batched prefill). 64k ctx works
  today's BF16 math; 200k needs WI-4 + I8 KV together.

## 5. The prefill problem (R2, the structural one)

`tp2_backend.cpp:704` — "prompt prefill: decode every prompt token (T=1 path)". Each prompt
token runs the full per-token decode pass (~31 ms) **plus** a per-token D2D copy of the hidden
state for MTP-head prefill. Measured in this review: **2,250-token prompt → 74.07 s** (32 t/s).
Every request is FullReset, so pi's growing conversation history is **re-prefilled every turn**:

| Context | TTFT today |
|---|---|
| 1k tokens | ~31 s |
| 5k tokens | ~2.6 min |
| 20k tokens | ~10.4 min |

This is the dominant blocker for pi-agent usability after the crash bug.

**Fix design (WI-4):** chunked batched prefill — process the prompt in T=512 chunks using the
T>1 batched ops. Components: (a) TP GEMM T>1 — already free (verified earlier); (b) batched
attention prefill with causal masking over local KV heads — stock `gqa_attention_prefill`
exists, needs TP2 port (verify-path attention is already T-general, so the machinery is there);
(c) **batched GDN over a chunk with initial-state carry** — the hard part; GDN TP path is
currently T=1 only, needs a chunked-scan kernel port per rank (each rank owns its v-head
subset; GDN state copied between chunks via the existing slot/copy machinery); (d) MTP head
prefill likewise batched (its T=1 loop is parallel to the main one, so same speedup).
Target: pp ≥ 300 t/s (llama.cpp-class on this hardware), which puts 20k ctx TTFT at ~70 s
and 8k ctx at ~27 s; prefix caching (later phase) removes the per-turn re-prefill entirely.

## 6. Worklist (prioritized, for the implementing agent)

**WI-1 — Arena scope fix (CRITICAL, ~30 min).**
Wrap per-request allocations in `DeviceArena::Scope` on `st.persistent` (and `st.staging`) in
`run_tp2_request`. Do not resize arenas.
Gate: serve 5× consecutive 2k-token requests, all 200, `nvidia-smi` flat; battery still PASS.

**WI-2 — Sampling honesty: ABSORBED into WI-4b.**
While WI-4b is not yet landed, the interim behavior is: validate request sampling; non-greedy
→ `RequestError` 4xx with a clear message (log the request's temp/top_p/top_k). When WI-4b
lands, this becomes parameter *validation* (reject invalid values, accept all valid ones).

**WI-3 — Option plumbing (MED, ~1 h).**
(a) honor `--draft-tokens` (clamp 0–3, reject >3 with error — k=4 is worse, doc 22);
(b) `--max-context` VRAM preflight at startup (estimate §4.3, refuse with the math);
(c) `LOOKUP` env → `--lookup` serve flag; (d) stop tokens from `generation_config_json`.
Gate: each flag behaves; battery PASS.

**WI-4 — Chunked batched prefill (structural, 1–3 days).** Design in §5.
Gate: pp ≥ 300 t/s on 8k-token prompt in battery; determinism + A2 still PASS; serve 8k-ctx
request TTFT ≤ 30 s.

**WI-4b — Sampling support (PROMOTED to must-fix batch, ~1–2 days, full scope below).**

*Requirement (owner): greedy/temp 0 must remain available and the default for pi, but every
standard sampling attribute must be settable per request. No more silent greedy.*

**Per-attribute behavior contract:**

| Attribute | Contract |
|---|---|
| `temperature` | ≤ 0 → greedy: current bit-identical path, zero behavior change. > 0 → accept kernel's sampling branch: target-distribution correction (accept draft with p_target(draft), resample from masked residual) |
| `top_k` | truncates the target distribution used in verify-accept. **Draft proposal stays greedy argmax** (standard vLLM-MTP scheme; sampling proposal = future work, out of scope) |
| `top_p` | same truncation mechanism |
| `min_p` | same |
| `presence_penalty` | logit adjustment on the T=4 verify rows before accept. Implement as a tiny 4×248k adjustment kernel using per-request token counts (see penalty state below) |
| `frequency_penalty` | same kernel, scaled by count |
| `seed` | per-request RNG seed; same seed + params + prompt → identical output (deterministic across runs) |

**Defaulting:** merge request sampling with server flag defaults (`--temperature`,
`--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, `--frequency-penalty`, `--seed`)
using the stock serve's existing merge logic; request value wins when present.

**Implementation steps (ordered):**
1. Add a `SamplingOptions` (temp, top_k, top_p, min_p, pres_pen, freq_pen, seed) to
   `TpRequest`; populate in `TPEngine::submit()` from merged request/server options.
2. Per request in `run_tp2_request`: build host `SamplingConfig`, `cudaMemcpyAsync` into the
   device config (replaces the setup-time greedy copy at `tp2_backend.cpp:468-471`). One
   sub-µs H2D per request — negligible.
3. Penalty state: pinned per-request token-count buffer (248k × I32 = 1 MB), incremented
   lazily for each accepted token (few per round — no per-round full upload). Verify first
   whether the accept kernel's sampling branch already consumes `token_counts`; if it does,
   wire the buffer in and skip step 4. If not, add the 4-row logit-adjustment kernel before
   accept (trivial: only T=4 rows).
4. (Only if needed by step 3) 4×248k penalty kernel, launched on the rank-0 stream before
   the accept op. Must be a no-op when both penalties are 0.
5. `TPEngine::sampling_defaults()` reports actual capability (temp>0 supported).
6. Parameter validation: reject invalid combos (top_k ≤ 0, top_p outside (0,1], min_p ≥ 1,
   temperature < 0) with 4xx + message. Also **add `min_p` to the per-request parse**
   (currently server-preset only — doc 37 §1) and plumb it through like the other params.
7. **Driver + battery exposure (owner-requested):** add sampling flags to the standalone
   driver (`--temp`, `--top-k`, `--top-p`, `--min-p`, `--seed`, `--presence-penalty`,
   `--frequency-penalty` — all optional, defaults = greedy). They map onto the same
   `SamplingOptions` struct as serve, so one code path is tested by both. The battery gains
   a **TC-S (sampling) case**, always run (not opt-in):
   - run 1: `--temp 1 --top-k 20 --top-p 0.95 --seed 42`, 512 tokens → record t/s + acceptance
   - run 2: identical flags → **hard gate: token-identical to run 1** (seed determinism)
   - run 3: `--seed 43` → must diverge from run 1 within 100 tokens
   - Baseline JSON gains: `sampled_mtp_tps`, `sampled_accept_pct`, `sampled_determinism_ok`
   - Thresholds (stochastic metrics get wider bands than greedy): sampled t/s FAIL >10%
     regression / WARN >3%; sampled acceptance FAIL >10 pts / WARN >4 pts below baseline;
     determinism + divergence = hard PASS/FAIL. First battery run after WI-4b records the
     sampling baseline; gating is active from the next run.
   - Greedy remains the primary DoD gate and is untouched (bit-identical).

**Test gates (all must pass, in addition to the standing battery):**
- **G1 invariant:** `temperature: 1, top_k: 1` → output identical to greedy (top_k=1 ⇒ argmax).
- **G2 seed determinism:** same seed → identical token stream (2 runs); different seeds →
  diverge within ~100 tokens (3 runs).
- **G3 acceptance sanity:** temp 1, temp 0 → acceptance drops but stays > 50%.
- **G4 throughput:** temp 1 at ≥ 80% of greedy t/s (sampling branch touches only T=4 rows).
- **G5 serve end-to-end:** curl with temp 1 / top_p 0.95 / top_k 20 / min_p / presence-penalty
  → 200; three runs with distinct seeds give three distinct continuations.
- **G6 battery unchanged:** greedy path bit-identical (determinism + A2 PASS, t/s within 3%).
- **G7 pi compat:** pi requests (temp 0) → greedy path → token-identical to pre-WI-4b runs.
- **G8 battery TC-S:** sampling case passes all sub-gates (determinism, divergence, t/s and
  acceptance within threshold); sampling baseline recorded in `verify_baseline.json`.

**WI-5 — I8 KV cache plumbing (only if >64k ctx wanted, 1–2 days).**
Build I8 pool planes + scales from `options.kv_cache`, thread through TP2 attention (kernels
exist). Gate: 200k-ctx load succeeds at ≤13 GB/rank; battery PASS at 8k unchanged.

**Standing gates after every WI:** `verify_battery.sh` (baseline 93.51 t/s @ `e0e0222e`;
from WI-4b onward the battery also runs the TC-S sampling case — greedy + sampled),
serve smoke (small + 2× large requests + one streaming request), determinism, A2 identity.

**Serve smoke additions (doc 37 §6 verify items):** (a) cancel-on-disconnect — abort a
streaming request mid-generation, confirm GPU work stops; (b) `tools`/`tool_choice`
round-trip end-to-end through TPEngine.

**Feature parity:** the full request-parameter / endpoint / behavior matrix vs
llama.cpp and vLLM, with triage (fill-now / fill-later / out-of-scope), is **doc 37**.
Fill-later items (stop strings, logit_bias, /v1/completions, /tokenize, /metrics) are
deliberately NOT in this batch — none block pi or daily chat.

## 7. Pi test-drive readiness

After WI-1 + WI-2 + WI-3 (~half a day): safe to wire pi at temp 0, ≤8k ctx, single concurrent
request. Expect TTFT ≈ ctx/32 s (250 s at 8k — uncomfortable; WI-4 fixes this). Streaming SSE
already works. No auth by default (`--api-key` optional). Pi itself runs temp 0, so WI-4b
(sampling) does not block the test drive; it only blocks using this server as a general
model server for temp>0 workloads.
