# Doc 34 — TP2 Serve Integration Plan (Work Order)

**Status:** WORK ORDER (authoritative for the implementer agent)
**Reviewer:** assistant (code review at each checkpoint)
**Goal:** Run the 2× RTX 5060 Ti TP2 + MTP engine (94.64 t/s, deterministic) behind the stock NInfer HTTP serve stack, OpenAI-compatible, streaming, so the pi agent can be served by it.

---

## 1. Current state

- The stock serve stack is complete and single-GPU:
  `apps/serve/main.cpp` → `HttpServer` → `GenerationService` → `ninfer::Engine` (pimpl).
  - `include/ninfer/engine.h` (line 57): concrete `Engine` class, pimpl (`shared_ptr<Impl>`).
  - `src/serve/generation_service.cpp` line 245: `engine_ = std::make_unique<ninfer::Engine>(std::move(engine_options));` — the ONLY construction site.
  - `GenerationService` talks to the engine exclusively through the public `Engine` surface (prepare / prepare_tokens / submit / generate / summaries) + `GenerationHandle::wait(OutputSink*, CancellationView)` for streaming.
- Our TP2 + MTP stack exists ONLY in the test driver `tests/multi_gpu/tp2_decode.cpp` plus the TP foundation libraries:
  - `src/core/multi_gpu/tp_group.{h,cpp}` (TpGroup, allreduce, one-shot AR/argmax)
  - `src/core/multi_gpu/weight_shard.{h,cpp}`, `tp_kernel.cu`
  - `src/targets/qwen3_6_27b/impl/load/tp_load.cpp` (sharded loader)
  - Driver: rank-pair construction (`make_rank`, ~line 380), prefill (~line 840), MTP round protocol (~lines 1030–1250), lookup drafting, calibration hook.
- Known performance reference (baseline `verify_baseline.json`): **94.64 t/s MTP k=3, 85.6% acceptance, deterministic, A2 token-identity PASS**.

## 2. Architecture decision (fixed — do not redesign)

**`TPEngine` = subclass of `Engine` + minimal virtualization + factory selection.**

1. Make `Engine`'s public methods `virtual` (engine.h): `prepare`, `prepare_tokens`, `count_tokens`, `prompt_capabilities`, `sampling_defaults`, `submit`, `generate`, `options`, `load_summary`, `memory_summary`, `runtime_stats`, `media_cache_summary`, `reset_memory_peaks`, and the destructor. No behavior change to the stock path.
2. Add a protected no-op constructor to `Engine` (e.g. `protected: Engine() = default;` — verify pimpl `shared_ptr<Impl>` default-constructs; if not, add a `static Impl* empty()` path or make Impl default-constructible). This lets `TPEngine` construct without triggering single-device materialization.
3. `class TPEngine : public Engine` in `src/runtime/tp2/tp_engine.{h,cpp}` overrides every public method. Owns:
   - `TpGroup` (2 devices) — persistent
   - TP-sharded model weights via `tp_load` — persistent
   - persistent arena(s) for per-request state
   - internal request FIFO (single-flight; see §4)
4. Factory: `std::unique_ptr<ninfer::Engine> make_engine(EngineOptions)` (new small file or in runtime). `GenerationService` line 245 becomes `engine_ = ninfer::make_engine(std::move(engine_options));`. Selection: `options.devices.size() == 2` → `TPEngine`, else stock.
5. `EngineOptions` (in `include/ninfer/types.h`): add `std::vector<int> devices = {0};` if absent. `ServeOptions`/`parse_serve_options`: add `--devices a,b` flag.

**Stock-path invariant (hard rule):** with `devices.size() == 1`, the binary must behave bit-identically to today. Every phase is gated by the verification battery (see §7) before moving on.

## 3. Phases

### P0 — Virtualization + factory (no TP2 yet)
- Edit `include/ninfer/engine.h`: virtualize public methods + protected base ctor.
- Add `make_engine` factory returning the stock `Engine` for 1 device.
- Point `GenerationService` at the factory.
- Add `--devices` to serve options (accepted but 2-device rejected with a clean error until P2).
- **Gate:** stock serve app boots, loads model, answers a chat completion identically; battery PASS; zero diff in stock decode path.
- **Review checkpoint R0.**

### P1 — Extract the driver core into `src/runtime/tp2/` (driver stays the test)
- Move (do not copy-then-delete until green) from `tp2_decode.cpp` into library code:
  - `tp2_backend.{h,cpp}` — rank-pair setup: device contexts, `make_rank` equivalent, TP load, draft head, draft vocab, workspace arenas.
  - `tp2_request.{h,cpp}` — per-request state: KV envelope (`ctx` tokens + k + 4), MTP pools, GDN slot protocol (committed slot 0, rebase via `copy_slot`), pinned accept buffer, cursor.
  - `tp2_rounds.{h,cpp}` — prefill + MTP round loop as `run_request(req, token_cb)`, where `token_cb(const TokenId*, int)` is called per accepted token (this is the streaming seam). Lookup drafting optional flag.
- The driver `tp2_decode.cpp` is rewritten to be a thin wrapper over `tp2_backend` + `tp2_request` + `tp2_rounds` (it keeps CLI parsing, timing prints, `--stream`).
- **Gate:** driver output byte-identical to pre-refactor on the battery prompt (A2 + determinism), t/s within −1% of 94.64, all `--lookup`/`--cal-dump`/`--stream` flags still work.
- **Review checkpoint R1.** (This is the highest-risk phase; review the GDN slot protocol and cursor semantics carefully.)

### P2 — `TPEngine` implementation
- `TPEngine::TPEngine(EngineOptions)`: materialize weights (both ranks), report `load_summary`/`memory_summary` from the TP capacity math (per-rank free-after-weights, KV pages).
- `prepare` / `prepare_tokens`: tokenize via stock frontend path (reuse the same tokenizer/chat-template pipeline the stock engine uses — do NOT re-tokenize; TPEngine may delegate preparation to an internal stock single-device Engine constructed with `no_materialize`-ish options IF such a mode exists, otherwise instantiate the target package's frontend directly as the driver does via `Package::resolve_weights` + `Tokenizer`).
- `submit`: enqueue into internal FIFO; return handle.
- `wait(OutputSink*)`: drain the loop; call `OutputSink` per token from `token_cb`; map finish reasons (`OutputLimit`, `Stop`); fill `GenerationResult` usage/metrics incl. the `GenerationMetrics` speculative fields (rounds, accepted, per-position).
- **Prefix path: `PrefixReusePath::FullReset` only.** Fresh per-request KV + GDN state each request. (Prefix reuse with GDN snapshots is P6+ / out of scope.)
- **Sampling: greedy only.** If resolved sampling is not greedy (temperature != 0), return `RequestError` cleanly (serve maps to HTTP 4xx). `sampling_defaults()` reports greedy.
- Concurrency: single-flight FIFO. Multiple `wait()` calls serialize. `runtime_stats()` reports queue depth.
- ctx: default 8192; request prompts longer than ctx → clean `RequestError` (context_length exceeded).
- **Gate:** unit test `tests/multi_gpu/test_tp_engine.cpp`: one prompt through TPEngine → token-identical to driver on same seed (greedy = deterministic); t/s ≥ 94.64 − 2%; metrics populated.
- **Review checkpoint R2.**

### P3 — Serve wiring
- `make_engine` selects `TPEngine` when `devices.size() == 2`.
- `apps/serve/main.cpp` memory summary prints per-rank capacity (extend `MemorySummary` or log via `runtime_stats`). (Serve binary target: `ninfer-serve`, built at `build/apps/ninfer-serve`.)
- Warmup: one 1-token MTP round at boot (mirrors driver warm behavior).
- **Gate:** `./build/apps/serve/ninfer-serve --artifact /home/intel/models/qwen3_8_27b.ninfer --devices 0,1 --port 8091` boots; `curl` chat completions (stream + non-stream) work; SSE chunks arrive per token; usage counts correct.
- **Review checkpoint R3.**

### P4 — Verification + parity
- curl smoke suite script `tests/multi_gpu/serve_smoke.sh`: non-stream, stream, two sequential requests (state reset check — second request must NOT continue the first's KV), error case (temp 0.7 → clean 4xx), context-overflow case.
- **Battery parity:** measure serve t/s over a 512-token generation via the HTTP endpoint (time-to-last-token / tokens). Must be ≥ 94.64 − 2% (i.e. ~92+). If short: profile before optimizing — suspect per-request allocation churn; move per-request state to a reusable arena slab.
- Determinism: two identical requests → identical token streams.
- Full verification battery still PASS (stock path untouched).
- **Review checkpoint R4.**

### P5 — pi agent wiring + results doc
- Document endpoint config for pi agent: base URL `http://<box>:8091/v1`, model id, `temperature: 0` requirement.
- Write doc 35 (results): t/s via serve, acceptance, parity numbers, known limits.
- Push, update doc 13/32 worklist.
- **Review checkpoint R5 (final).**

## 4. Concurrency model (fixed for this project)

- TPEngine is **single-flight**: the rank pair processes one request at a time. The stock `concurrent_executor`/admission policy continue to queue; `wait()` blocks until the request's turn. This is acceptable for a single-user pi agent.
- Do NOT attempt multi-request overlap (would need per-request rank pairs sharing weights) — that is future work, documented in doc 35 if relevant.

## 5. Constraints (hard)

1. Push ONLY to `chrisconcepcion/dual_5060_ti_ninfer` (origin) + `local` bare backup. Never Neroued.
2. Every commit: tree builds clean (`make -j24`), zero new warnings.
3. Battery (`/home/intel/verify_battery.sh`) PASS after each phase before the next starts; baseline must remain 94.64 t/s / 85.6%.
4. Do not touch the MTP round protocol semantics from P1 onward — refactor only. Any behavior change = stop and ask.
5. `--mtp 3` is the production config. k=4 stays opt-in.
6. Time-box P1 aggressively: if extraction keeps failing, the fallback is a *copy*-based `tp2_rounds` (driver keeps its own copy) — parity gate still applies. But deletion of driver duplicates only after green.
7. All long runs under `timeout`; stdout via `stdbuf -o0` when running drivers.

## 6. Risks

| Risk | Mitigation |
|---|---|
| pimpl `Impl` not default-constructible for protected base ctor | Add `Impl::empty()` static or default ctor; P0 gate catches it immediately |
| Per-request KV allocation churn kills t/s parity | Reusable arena slab sized at boot (max ctx); P4 gate |
| Stock frontend prepare path entangled with single-device materialization | Fallback: instantiate target frontend directly (driver pattern) in TPEngine |
| GDN slot protocol mis-extracted → silent nondeterminism | A2 + determinism gate in P1 is byte-strict |
| Serve per-token overhead (HTTP chunk flush) | Negligible at 94 t/s (10.6 ms/token budget); measure in P4 anyway |

## 7. Gate summary

| Phase | Gate | Review |
|---|---|---|
| P0 | stock serve unchanged; battery PASS | R0 |
| P1 | driver byte-identical, t/s ≥ 93.7 (−1%) | R1 |
| P2 | unit: token-identical, t/s ≥ 92.7 (−2%), metrics ok | R2 |
| P3 | serve boots, curl stream works | R3 |
| P4 | serve t/s ≥ 92.7, determinism, smoke suite green, battery PASS | R4 |
| P5 | pi agent served; doc 35 | R5 |

## 8. Reference numbers (for parity checks)

- Baseline: **94.64 t/s**, acceptance 85.6%, 3.58 tok/round, round ~37.8 ms, verify ~34.8 ms.
- Plain decode ~35 t/s; pp ~32 t/s (220-token prompt).
- `verify_baseline.json` at `/home/intel/verify_baseline.json`; battery at `/home/intel/verify_battery.sh`.
