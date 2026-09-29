# 127 — KVarN MultiBatch (Part b): status, what shipped, and the scheduler gap

Date: 2026-09-01. Branch: `wo/kv-uniform`. Builds on the decode-decay fix (docs/122 §22-§23).

## TL;DR

The **kernel + launcher side of multi-sequence batched KVarN decode is DONE and bit-exact**
(Stage A, commits `3c58dfad..+`). The **runtime cannot yet use it**: the only engine that binds
KVarN workspaces (the TP2 backend) serves requests strictly one-at-a-time, and the only engine
that runs batch>1 forwards (the graph/ConcurrentExecutor path) has no KVarN binding at all and
cannot fit this 18 GB model on this box's 2×16 GB cards single-GPU. The handoff's Part-(b)
estimate ("slice4 is ready; wire batch_size>1 + MultiBatch") was ~60% of the story; the last
mile is a TP2 request-lifecycle scheduler, which is an owner-scoped decision, not a drive-by.

## R1 core is now IMPLEMENTED + PROVEN (2026-09-01, commits after c631f7f1 + 1b9c10c5)

The ops-layer core of option R1 (batched decode) is done and unit-verified:

1. **Kernel + launcher** (Stage A): slice4 MultiBatch with per-lane tail state
   (`lane_packed_pages` + `tail_batch_elems`); batched reduce; batched attend is BIT-EXACT vs
   per-lane single-sequence (bench BATCHTEST).
2. **Ops** (1b9c10c5): `kvarn_bind_batched_workspace` (per-lane tiles as slices of a shared
   [batch]-strided buffer, returns the attend stride) + `gqa_kv_append_kvarn_batched` (one call
   appends+commits all lanes: contiguous lane slices, per-lane block-table row view, publishes
   per-lane committed_pages). Validated by `ninfer_kvarn_batched_ops_test`: 2 lanes (100/250
   tokens, 2 layers) -> per-lane tail tiles + committed_pages/tail_count BIT-IDENTICAL to two
   independent single-lane sequences.
3. **Batched attend entry** (326ec81c): public `gqa_attention_cached_batched` (4D q, batched
   cache view, valid_columns/kv_table_rows, batched tail) -> detail launcher
   `gqa_attention_kvarn_cached_batched_launch` (slice4 MultiBatch + batched reduce, decode-only).
   `validate_batch_cache` now accepts KVARN_K4V2 (U8 code planes + kvarn_scale_pages).
   `ninfer_kvarn_batched_ops_test` now drives bind -> append -> attend through the public API and
   compares against per-lane `gqa_attention_cached`: BOTH lanes bit-identical (append + attend).
4. **RULE OF THUMB that made this tractable**: because the batch dimension is the OUTERMOST
   dim of every batched tensor (k/v {D,kvh,width,batch}, positions {width,batch}), the per-lane
   slices are CONTIGUOUS. So per-lane append = the existing single-lane function + a per-lane
   block-table row view. No strided-gather needed.
3. **RULE OF THUMB that made this tractable**: because the batch dimension is the OUTERMOST
   dim of every batched tensor (k/v {D,kvh,width,batch}, positions {width,batch}), the per-lane
   slices are CONTIGUOUS. So per-lane append = the existing single-lane function + a per-lane
   block-table row view. No strided-gather needed.

## What remains (NOT done) — the runtime scheduler + engine contract

Steps 4-5 of R1 are the runtime glue, and are a FUNDAMENTAL redesign of the TP2 engine's request
model, not a drive-by wiring:

- **Step 4** (`TextContext::kvarn_attend_text` batch branch): per-lane workspace array + per-lane
  cache view + batched attend call. The ops building blocks now exist and are proven.
- **Step 5 (the hard part)**: the TP2 engine processes requests strictly one-at-a-time. In
  `tp_engine.cpp`, the ENTIRE `run_tp2_request` runs under `shared->mutex` (line 136/223) and
  stream its output per-request via a `TokenCallback`. `TpRankState` holds single-lane tensors
  (`ids, cpos, rpos, kvr, slt` are all `[1]`), and the decode loop is a two-rank `std::barrier`
  pair. Batching N requests requires: collecting N admissions, driving their decode in lockstep
  (per-lane MTP draft/verify/accept/sampling/stop-state), interleaving N independent output
  streams, per-lane prefix-cache/rewind, and re-syncing the 2-rank barrier with the scheduler
  instead of a single request.

This is a 1-2+ day focused engine rewrite with real regression risk to the shipped single-
sequence path (the hard requirement that both cache lanes stay green), and is NOT something to
rush as an autonomous drive-by. Recommend: owner-scoped effort behind `--max-concurrency N`,
with the single-sequence path (N==1, the guard's config) pre-severned.

**Status as of the R1-core commits:** kernel + launcher + ops (bind/append/attend) + the public
batched entry are DONE and bit-exactly proven against per-lane single-sequence execution.
The remaining scheduler is the only blocker to a served MultiBatch inference loop; the batched
kernel/append/reduce are all correct and ready to be driven.

## What shipped (Stage A)

1. **Kernel** (`gqa_decode_slice4_kvarn.cuh`): two new params — `lane_packed_pages` (device I32
   [batch]) and `tail_batch_elems`. Under `MultiBatch`, each CTA (grid.z = lane) derives its own
   `key_base`/`tail_count` from its lane's committed pages + window (invariant
   `window_b = packed_b*64 + tail_b`, `tail_b ∈ [0,64)`) and offsets the batched tail tiles by
   `batch * tail_batch_elems`. Everything else in the body was ALREADY lane-correct via the
   templated `MultiBatch`/`Masked` path (per-lane `column_base`, positions, block-table row,
   window, active-splits, partial offsets). Single-sequence instantiations are byte-identical
   (`if constexpr` discards the block; the launcher passes nullptr/0).
2. **Launcher** (`gqa_attention_kvarn.cu` + `GqaKvarnTail`): tail gains
   `lane_packed_pages / tail_batch_elems / batch_size / lane_table_rows / lane_valid_columns`
   (defaults = single-sequence). The unified route dispatches `MultiBatch=true/Masked=true` when
   q is 4D `{D,qh,width,batch}`, sizes partials by `width*batch`, and runs the SHARED reduce
   kernel with `MultiBatch/Masked=true` (it already supported both). `full_width` follows the
   decode.cu convention (= per-lane `q.ne[2]`).
3. **Validation** (`ninfer_slice4_kvarn_bench`, now the DEFAULT ctest behavior): batch=2 lanes
   with windows 100/250 (1pg+36tail / 3pg+58tail), DIFFERENT block-table rows, random tails ->
   final reduced outputs **BIT-IDENTICAL** to per-lane single-sequence kernel+reduce (0/6144).
   The timing sweep moved behind ctx args / `SWEEP=1`.

## Why the runtime can't use it yet (the scheduler gap)

Serving topology on this box (`ninfer-serve --devices 0,1`):

* **TP2 backend** (`src/runtime/tp2/`): binds KVarN (`tp2_backend.cpp:437-447`
  `kvarn_bind_sequence_workspace` + `set_kvarn_workspaces`). Request lifecycle = one request in
  flight; `TPEngine::submit` queues (mutex + condvar); the decode loop
  (`tp2_backend.cpp:1470+`) is a per-request, barrier-synced rank-thread pair calling
  `ordinary_decode_batch`/`target_verify_batch` with **batch=1**. The `batch>1` shapes in those
  calls are the MTP verify width (draft tokens), not sequence lanes.
* **Graph engine** (`src/runtime/engine/concurrent_executor.h` + `program_impl.h`): real
  multi-lane scheduling (`decode_batch(lanes)`, per-batch-size CUDA graph capture, lane ->
  block-table-row via `bound_row()`). But it has **zero KVarN references** — KVarN was never
  wired there — and it is the single-GPU path, so the 18 GB artifact doesn't fit on one 16 GB
  card anyway.
* `TextContext::kvarn_attend_text/kvarn_attend_mtp` throw on `active_sequence_batch_ > 1` —
  the correct guard today (the attend is ready; the per-lane APPEND/commit state is not).

## The last mile — two options (owner decision)

**Option R1: batched scheduler inside the TP2 backend.** Admit up to N concurrent requests;
drive one batched forward per step across lanes.
- Per-lane KVarN state: allocate N `KvarnSequenceWorkspace`s whose layer tiles come from ONE
  batched buffer per layer (`[batch][G,D,kvh]` k / `[batch][D,G,kvh]` v), so lane b's
  `k_tile` is a contiguous slice at `b*tail_batch_elems` — the attend kernel needs no change.
  (`kvarn_bind_sequence_workspace` gains a batched variant; ~50 LoC.)
- Per-lane append/commit: loop lanes calling the existing `gqa_kv_append_kvarn_and_commit` with
  a row-view of the lane's block table (`pool.block_table_row(b)` exists). For decode width=1
  the k/v lane-slices are contiguous; for MTP verify (width k+1) the append kernel needs a
  batch-stride param (or a per-lane strided gather — small kernel).
- The invasive part: the request loop itself — ids/cpos/rpos/kvr/slt become [batch] vectors
  (the tensors already exist per lane), MTP drafts/acceptance/streaming/cancellation/stop
  handled per lane, prefix-cache lanes disabled for v1, and the barrier now synchronizes the
  SCHEDULER with both ranks instead of one request. Realistically a 1-2 day focused change in
  `tp2_backend.cpp`/`tp_engine.cpp` with real regression risk to the shipped single-seq path.
- Payoff estimate (measured components): attention per round at 160k ≈ 15 ms of a 59.8 ms round
  (T=4, 16 attn layers × 0.938 ms). Aggregate throughput at lane count N ≈
  `N / (N·A + G)` vs `1 / (A + G)`: **~1.6× @N=2, ~2.3× @N=4, ~2.9× @N=8** (GEMM/GDN/state
  costs amortize; attention does not — it's already DRAM-saturated per lane). The fork's x5.9
  dual does NOT transfer 1:1: their single-seq kernel was latency-starved (few CTAs); ours runs
  340 CTAs / 4.7 waves and is already GPU-saturated.

**Option R2: bind KVarN into the graph engine (ConcurrentExecutor).** The scheduler already
exists (lanes, rows, per-batch graphs). Needs: KVarN workspace array per lane in the program,
the batched append/commit (same pieces as R1), and graph-capture compatibility — the hard part:
KVarN append/commit is HOST-driven (page acquire depends on host positions), while the graph
engine replays captured batched forwards; int8/bf16 survive capture because their append is
FUSED into the attention kernel (device-side). KVarN would need a device-side page-publish path
(allocator in device memory) — deeper than R1, and still single-GPU-only (wrong hardware for
this model). Not recommended here.

**Recommendation:** R1 behind an opt-in (`--max-concurrency N` on the TP2 server), v1 scope =
decode-only lanes (width=1 per lane; MTP-verify lanes = v1.1), prefix reuse disabled for
batch>1 initially, gated by a 2-sequence aggregate-tps guard
(`tools/bench/run_serve_concurrency.py` exists for exactly this).

## Updated decay table (post-fix, greedy, ITERS=1, COMP=192, serve logs cpasync2/final2/stageA_cells)

| ctx  | baseline | fixed  | Δ      | int8 ref | ratio  |
|------|---------|--------|--------|----------|--------|
| 10k  | 69.1    | 74.8   | +8.2%  | 69.6     | 1.07   |
| 25k  | 68.3    | 71.1   | +4.1%  | 70.0     | 1.02   |
| 40k  | 64.0    | 67.7   | +5.8%  | 66.4     | 1.02   |
| 80k  | 58.2    | 61.6   | +5.8%  | 63.4     | 0.97   |
| 160k | 47.1    | 53.7   | +14.0% | 60.0     | 0.90   |
| 250k | 43.5    | 46.4   | +6.7%  | (n/a)    | —      |

MTP accept 67.9-73.7% (baseline band; accept fluctuates ±5pp run-to-run — single-run cells).
Prefill unchanged at every cell (806.7/740.1/583.4/503.3 tok/s at 10k/40k/160k/250k).

## Post-fix phase attribution (T=1 Wc=4 @160k, for whoever continues)

K-deq ~20%, V-deq ~20%, staging ~22% (mostly hidden; residual wait), softmax ~23%, QK ~10%,
PV ~15% (overlapping upper bounds). Remaining structural lever = direct-to-MMA-fragment
dequant (kills the k_s/v_s smem round-trip + one barrier per tile); Bc=64 tiles are
architecturally BLOCKED at 2 blocks/SM (k_s+v_s would need 64 KB > the 50.7 KB budget).
