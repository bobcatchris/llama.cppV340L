# 46 — KV capacity sizing runbook (radiance §3.1/§3.2 port)

**Date:** 2026-09-02 · **Status:** CPU-only deliverable (docs + math; validation needs a GPU slot)
**Source:** doc 44 §3.1/§3.2 (radiance's KV group-size-by-capacity + explicit calibrated KV pin),
NInfer `src/runtime/engine/kv_capacity.{h,cpp}`, `src/targets/qwen3_6/impl/runtime/layouts_impl.h`,
`apps/cli/options.cpp`.

Radiance measured **+20.7% KV tokens** by deriving the KV group size from measured capacity
(§3.1) and another **+5.7%** by replacing the profiler-derived pin with a calibrated explicit
pin that hands the KV everything but ~0.3 GiB (§3.2). This is the NInfer equivalent: the
mechanisms already exist (auto-sizing with a measured stride + explicit pin via
`--kv-capacity`); what was missing is the runbook for when to use which and how to calibrate.

## 1. Model KV geometry (qwen3_6_27b, TP2)

Only the 16 full-attention layers carry KV (the 48 GDN layers are linear-attention with no KV
cache). TP2 shards kv heads across ranks: **2 of 4 kv heads per rank**
(`tp2_backend.cpp` L312 "TP: local kv-head block (4/2)").

Per token, per rank:

| | K (2 heads × 256 dim × 16 layers) | V (same) | total elements |
|---|---|---|---|
| elements/token/rank | 8,192 | 8,192 | 16,384 |

Per-token KV bytes per rank by storage type (`KvCacheStorage`):

| dtype | K | V | per-token/rank | vs bf16 |
|---|---|---|---|---|
| `bf16` | 16 KB | 16 KB | **32 KB** | 1× |
| `int8` (Int8Group64) | 8 KB | 8 KB + group scales | **≈16.2 KB** | ~2× |
| `kvarn` (K4V2) | 8,192 × 0.5 B = 4 KB | 8,192 × 0.25 B = 2 KB + tile scale/zp | **≈6.5 KB** | ~5× |

Kvarn is int4 K / int2 V (asymmetric RTN, quant group 64) — sub-8-bit on *both* legs, i.e.
below radiance's fp8 KV (doc 44 §2.4 triage, resolved).

## 2. Page structure and how capacity resolves

- **Page size is fixed at 64 tokens** (`kPagedKVPageSize`, `src/core/paged_kv_cache.h` L15;
  power-of-two — the address math uses `kPagedKVPageMask`). Radiance's §3.1 group-size lever
  does **not** port 1:1: our "group" is already 64 tokens/page (their "block=64 today"), and
  changing it is a compile-time kernel change, not config. The portable lever is **page count
  = capacity**, which below is sized from measured memory.
- Page count bounds (`layouts_impl.h` L705–714):
  - `minimum_pages = max(max_context, max_concurrency)` — one page per context token, at
    least one page per concurrent request.
  - `maximum_pages = max_concurrency × max_context` pages.
  - `capacity_tokens = pages × 64` (`KvCapacityResolution::resolved_tokens`).
- **Explicit mode** (`--kv-capacity N`): `pages = ceil(N / 64)`. Must fit in
  `available_after_weights` or the engine refuses to start (fail-fast, good).
- **Automatic mode** (`--kv-capacity auto`, the default policy):
  `pages = min + (available_after_weights − headroom − min_reservation) / stride`,
  capped at `maximum_pages` (`kv_capacity.cpp` `resolve_kv_capacity`).
  - `headroom` default **1 GiB** (`kDefaultKvCapacityHeadroomBytes`) — deliberately more
    conservative than radiance's calibrated ~0.3 GiB.
  - `stride = bytes_per_additional_main_page_group` is **measured from the actual layout**
    (adjacent-candidate reservation delta, `layouts_impl.h` L724–731), not a nominal table —
    the NInfer instance of radiance's "derive from measured capacity, not the smallest
    bucket". Includes KV pages + workspace + graph allowance + transient.
- The resolution is fully reported: `MemorySummary` carries `kv_capacity_mode`,
  `kv_capacity_page_groups`, `kv_capacity_max_page_groups`,
  `kv_capacity_increment_bytes`, `kv_capacity_headroom_bytes`, `planned_slack_bytes`
  (`concurrent_executor.h` `memory_summary()`).

## 3. Sizing formula (the runbook)

Given a serving shape (concurrency `C`, max context `T`, card size `B_card`, weights+runtime
per rank `W`):

1. **Start with auto.** `--kv-capacity auto` gives the maximum pages that fit with 1 GiB
   headroom. Read `resolved_tokens` from the startup log / `MemorySummary`. This is the safe
   default; nothing below is required for correctness, only for squeezing tokens.
2. **Measure the real peak.** Run the target shape under realistic load:
   - concurrency `C`, prompts of length ~T, decode to the max output (the results/*.json
     battery shape: `mtp_on_<kvarn|bf16>_<ctx>_r1`),
   - record peak device usage per rank (nvidia-smi / driver query) **and** the engine's
     `planned_slack_bytes` at steady state.
   Peak = KV reservation (fixed at start) + runtime peaks (activations, graph instantiation
   headroom, transient). The reservation is fixed, so the variable part is what headroom must
   cover.
3. **Pin explicitly when the shape is stable.** Set
   `--kv-capacity N` where
   `N = 64 × (min_pages + floor((available_after_weights − peak_runtime − M) / stride))`,
   with **M = 0.3 GiB** (radiance's calibrated margin; their 260k-token sweep survived it).
   Equivalently: keep `planned_slack_bytes ≈ M` after the pin. This recovers the difference
   between 1 GiB default headroom and the measured need — the whole of radiance §3.2's win.
4. **Verify:**
   - `resolved_tokens` ≥ `C × T` (or the intended token budget),
   - the full battery runs no-OOM at the pinned capacity, including the longest-context case,
   - decode t/s unchanged vs auto (capacity must not affect the hot path — if it does, that's
     a finding, not a ship-it).
5. **Re-derive after anything that moves device memory:** weights (quant change, MTP module
   size), cuda-graph on/off, draft window (`--mtp-k` / adaptive max), prefill chunk, new
   ops/buffers, concurrency. Radiance's explicit rule: the pin is a *calibration*, not a
   constant — their `calibrate-kv.sh` re-ran after every such change.

### Worked example (2× 16 GB cards, 4-bit body, TP2, kvarn)

- Weights ~17 GB total (4-bit body + bf16 MTP module + draft head) → ~8.5 GB/rank;
  runtime + graphs + activations peak ≈ 3 GB/rank (measured in step 2, not assumed).
- `available_after_weights` ≈ 8.5 GB; auto: `pages = (8.5G − 1G − min_res) / stride`.
- kvarn per-token/rank ≈ 6.5 KB → ~1.3 GB of KV buys 200k tokens →
  **KV capacity ≈ 500k–800k tokens** at this shape (exact value from step 2's measurement),
  vs ≈ 100k–160k for bf16 KV on the same cards. Sanity anchor: radiance's MXFP4 build held
  940k KV tokens on 2×R9700 (24 GB) — same order.

## 4. What does NOT port

- **§3.1 group-size-by-capacity** — page size is a compile-time 64 (address math). If a
  future layout wants a different page size, the capacity curve already measures the stride
  from the layout, so `resolve_kv_capacity` needs no change; the kernels do.
- **§3.2 "do NOT re-derive from the profiler's fit-into-budget line"** — our auto mode
  computes against actual measured layout reservations (not a profiler's util estimate), so
  the failure mode they patched doesn't exist here; the *calibration* practice (explicit pin
  at M ≈ 0.3 GiB after measurement) still applies.

## 5. Open items (GPU phase)

- [ ] Run step 2 (peak measurement) on the production box for the doc 134 serving shape;
      replace the worked-example estimates with measured `planned_slack_bytes`.
- [ ] Decide the ship default: keep 1 GiB auto headroom (safe) vs document the explicit-pin
      procedure as the serving recipe.
- [ ] Add a results/*.json field for `resolved_tokens` + `planned_slack_bytes` so the battery
      records the KV shape it ran in (currently the JSONs record `vram_mb` only).
