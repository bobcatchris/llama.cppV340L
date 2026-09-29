# A1 — gzenz/ninfer Optimization Backport

**Lane:** A1 (01a077a7) — (vi) fix DONE, now assigned to backport gzenz/ninfer optimizations.
**Source repo:** https://github.com/gzenz/ninfer (fork of upstream ninfer)
**Cloned to:** `/tmp/gzenz_ninfer` (depth 1, tip 4b882d0)
**Priority:** **NVFP4 KV cache is the FIRST thing.**

## Context

The user (14:06 ET) asked: "next thing for a1 is implementation of nvfp4 kv cache but I also
need you to find all optimizations found here including the kv cache for a1 to work on (first
thing)." This doc is the coordinator's survey of the gzenz/ninfer fork's optimizations.

The gzenz fork targets **reliable 555k-context inference with 3 concurrent agentic sessions on a
single RTX 5090 with NVFP4 KV**. It is a clean divergence — upstream has its own host-KV cache
implementation; the fork's is independently developed and battle-tested with real workloads.

## Optimizations (in priority order)

### 1. NVFP4 KV cache (FIRST) — `--kv-dtype nvfp4`

**Doc:** `/tmp/gzenz_ninfer/docs/maintainer/kv-nvfp4-yarn.md`

**What it is:** 4-bit E2M1 codes + E4M3 group-16 scales for KV storage. 144 bytes/token/KV-head
(vs 264 int8, 512 bf16) — **45% reduction vs int8**.

**Storage format:**
| Component | Format | Bytes/token/KV-head |
|-----------|--------|---------------------|
| Codes | E2M1 (4-bit), 2 per byte | 128 |
| Scales | E4M3FN, 1 per 16-element group | 16 |
| **Total** | | **144** |

**Architecture:**
- **QK:** `mma_nvfp4_e4m3` (m16n8k64) with built-in E4M3 block scales. Q and K both quantized to
  NVFP4. **Hadamard rotation applied to K (and Q) for outlier suppression; V is NOT rotated.**
- **PV:** V dequantized to BF16 via `decode_nvfp4_e2m1x2 × E4M3 scale`, then BF16 MMA for PV.
- **Fused append:** decode kernel quantizes current K/V to NVFP4 in-place.
- **Scale layout:** natural (non-swizzled) row-major for KV (not the M128x4 swizzle used by weight MMA).

**Dispatch:** `--kv-dtype nvfp4` selects `KvCacheStorage::Nvfp4Group16` → `DType::U8` with
`quant_group=16`. Code planes use `leading_extent=head_dim/2=128`; scale planes use
`leading_extent=16` with `DType::U8` (E4M3 bytes, not FP16).

**VRAM impact:**
| Config | int8 KV | NVFP4 KV | Reduction |
|--------|---------|----------|-----------|
| 262k, c=4 | 9.96 GiB | 6.0 GiB | 40% |
| 555k, c=3 | — | 12.15 GiB | — |
| 600k, c=1 | — | 11.4 GiB | — |

**Quality (LongBench v2, 20 samples):**
| Config | Score |
|--------|-------|
| Old int8 (262k native) | 9/20 = 45% |
| NVFP4 (262k native) | 6/20 = 30% |
| **NVFP4 + YaRN (600k ctx)** | **9/20 = 45%** |

**AIME 2025:** int8+Hadamard 28/30 = 93.3%; NVFP4 29/30 = 96.7%. **Needle 128k: 5/5 = 100%.**

**Max context (RTX 5090, 32GB):**
- c=3 + vision: 555k max (YaRN factor 2.12).
- c=1, no vision: 600k max (YaRN factor 2.30).
- 96GB GPU projection: c=1 → 7.7M tokens (29.2x), c=4 → 1.9M tokens (7.3x).

**Key files (fork):**
- `src/ops/nvfp4/` (or similar) — NVFP4 MMA kernels.
- `src/core/paged_kv_cache.{h,cpp}` — `KvCacheStorage::Nvfp4Group16`.
- `src/runtime/engine/context_cost_defaults.cpp` — NVFP4 context cost.
- `docs/maintainer/kv-nvfp4-yarn.md` — full spec.

**A1's task:** Implement NVFP4 KV cache in our repo. The fork's implementation is the reference.
Port the storage format, the MMA kernels, the fused append, the scale layout, and the dispatch.
Verify with LongBench + AIME + needle (the fork's quality benchmarks).

### 2. Paged KV cache (P=64)

**Doc:** `/tmp/gzenz_ninfer/docs/maintainer/paged-kv-cache.md` (Chinese)

**What it is:** Growing KV uses a set of startup-fixed homogeneous pools. Each pool stores all
planes with the same frontier and lifetime, uses a fixed token page size (P=64), plane order,
and page-group count. Owns an independent physical page-ID namespace and free capacity. Provides
one logical address space per active sequence. Consumers address via block table.

**Three independent granularities:**
| Granularity | Meaning | Contract |
|-------------|---------|----------|
| Allocation | Pool acquires/releases | P=64 tokens |
| Valid-frontier | Consumer can read up to | 1 token |
| Reusable-state | Frontier with complete continuation | target-defined checkpoint |

**Key concepts:**
- **Typed pool set:** Main Text, MTP, DFlash Full (each with its own pool).
- **Page group:** A pool-local page-group ID selects the payload in all grouped planes.
- **Closed plane orders:** Page-major `[X,P,H,N]` (Main/MTP), head-major `[X,P,N,H]` (DFlash).
- **Logical page identity:** Generation-checked, content-epoch, committed columns, replicas.
- **Device/Host replicas:** D2H/H2D transfer with epoch/coverage verification.
- **KV address space:** Owning membership, three extents (entitlement, membership, committed frontier).
- **Execution rows:** Fixed-address Device block-table matrix `[N_logical, C]`.
- **Sharing, Move, COW:** Private continuation (Move), immutable fork (Fork), non-page-aligned frontier (COW).
- **CUDA Graph & table publication:** Plane bases and block-table matrix base stable across replays.

**Key files (fork):**
- `src/core/paged_kv_cache.{h,cpp}`
- `src/core/host_kv_arena.{h,cpp}`
- `src/targets/qwen3_6/impl/runtime/logical_kv_store.h`
- `src/targets/qwen3_6/impl/runtime/host_kv_extent_store.h`
- `docs/maintainer/paged-kv-cache.md`

**A1's task:** Implement paged KV cache (P=64) in our repo. This is the foundation for the
Host-KV safety net and the NVFP4 KV cache's page-level parking.

### 3. Host-KV safety net — `--host-kv-mib`, `--host-state-slots`

**What it is:** When device KV is full, evicted continuations are spilled to a pinned host arena
(D2H) and restored via H2D on cache reuse. Scatter-gather allocation handles arena fragmentation.
**Smallest-first eviction with pinned-entry protection.**

**Verified:** Across 140+ requests with 3 concurrent 330k–470k sessions.

**KV Cache Stress Test (3 × 200k concurrent):**
| Metric | Value |
|--------|-------|
| Total demand | ~636k tokens vs 537k device capacity |
| host_kv_occupied | 2.81 GB (parked to host) |
| maximal_fallbacks | 0 (graceful pressure relief) |
| owners_evicted | 0 |
| crashes | 0 |

**Key files (fork):**
- `src/targets/qwen3_6/impl/runtime/host_kv_safety_net.h`
- `src/core/host_kv_arena.{h,cpp}`
- `src/serve/stats_json.cpp` — host_kv_occupied counter.

**A1's task:** Implement Host-KV safety net in our repo. Requires paged KV cache (item 2).

### 4. YaRN context extension — `--rope-scaling-factor`, `--rope-scaling-original-context`

**Doc:** `/tmp/gzenz_ninfer/docs/maintainer/kv-nvfp4-yarn.md` (YaRN section)

**What it is:** Linear ramp applied to `rope_positions` only (not `cache_positions`):
```
if position <= original_context:
    scaled = position
else:
    scaled = original_context + (position - original_context) / factor
```
CLI flags: `--rope-scaling-factor F` (1.0–32.0), `--rope-scaling-original-context N` (default 262144).

**Why linear scaling preserves quality (Qwen3.8-27B):**
| Factor | Value | Effect |
|--------|-------|--------|
| rope_theta | 1e7 | Lowest frequency period = 38M tokens |
| rotary_dim | 64/256 (25%) | 75% of dimensions are position-independent |
| softmax layers | 16/64 (25%) | 48 GDN layers have no RoPE |

At 600k tokens, scaled RoPE position is 1.08% of the 38M-token period — no frequency wrapping.
Quality guaranteed up to ~9.5M tokens.

**A1's task:** Implement YaRN context extension in our repo. Small change (linear ramp on
rope_positions). Requires NVFP4 KV cache (item 1) for the full 555k/600k context.

### 5. OOM recovery (`std::bad_alloc` catch)

**What it is:** Materialization reserve and worker loop catch OOM, clear active state while
preserving pending requests, back off admission for 4 iterations, fail all after 8 consecutive
recoveries.

**A1's task:** Implement OOM recovery in our repo.

### 6. Rewrite checkpoint at turn boundary

**What it is:** Checkpoint is captured where the prompt ends (before reasoning begins), not at
the execution frontier. Follow-up prompts with `preserve_thinking=off` match the stored ledger
up to the checkpoint.

**A1's task:** Implement rewrite checkpoint at turn boundary in our repo.

### 7. Token stability with `preserve_thinking=off`

**What it is:** Reasoning is dropped from ALL assistant messages when `preserve_thinking=off`,
keeping prompt tokens stable across turns. Without this, the last assistant message's reasoning
was kept on its turn but dropped on the next, shifting all subsequent tokens and breaking prefix
reuse.

**A1's task:** Implement token stability with `preserve_thinking=off` in our repo.

### 8. Checkpoint lifecycle preservation

**What it is:** Rewrite checkpoint is retained when state slot reservation fails during `finish()`
instead of being silently dropped. The aliased state image is still valid.

**A1's task:** Implement checkpoint lifecycle preservation in our repo.

### 9. Tolerant tool-call recovery — `--tolerant-tool-calls`

**What it is:** Recovers complete Qwen calls with malformed wrappers.

**A1's task:** Implement tolerant tool-call recovery in our repo.

### 10. Reasoning-effort tier mapping

**What it is:** High/Max map to XHigh instead of rejecting.

**A1's task:** Implement reasoning-effort tier mapping in our repo.

### 11. /stats endpoint

**What it is:** Runtime gauges, KV transfer counters, pressure metrics, cache reuse paths.

**A1's task:** Implement /stats endpoint in our repo.

### 12. Monitoring dashboard (`tools/monitor/`)

**What it is:** Live GPU/util/decode/prefill/TTFT graphs, KV cache occupancy, request log,
12VHPWR sensor.

**A1's task:** Implement monitoring dashboard in our repo.

### 13. E2E test suite (`tools/e2e/`)

**What it is:** Multi-phase KV eviction, device-KV pressure, slot pressure, trash mode.

**A1's task:** Implement E2E test suite in our repo.

### 14. Request-log rotation — `--request-log-max-mib`, `--request-log-keep`

**What it is:** Size-based JSONL rotation.

**A1's task:** Implement request-log rotation in our repo.

## Performance benchmarks (fork)

### Concurrent MTP3 decode saturation (Qwen3.6-27B NVFP4):
| C | Steady (s) | Avg batch | Aggregate decode tok/s | MTP acceptance | Speedup vs C1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× |
| 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× |
| 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× |
| 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× |

### MTP3 long-reasoning decode (Qwen3.6-35B-A3B NVFP4):
| Fixture | Decode tok/s | MTP acceptance | MTP tokens/round |
|---------|-------------|---------------|-----------------|
| long_decode_aime26_01 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| long_decode_aime26_15 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| long_decode_aime26_30 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### DFlash block=8 (Qwen3.6-35B-A3B):
| Fixture | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---------|-------------|------------------|---------------------|
| long_decode_aime26_01 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 |
| long_decode_aime26_15 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 |
| long_decode_aime26_30 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 |

## A1's plan (suggested)

1. **NVFP4 KV cache** (FIRST) — port from fork, verify with LongBench + AIME + needle.
2. **Paged KV cache (P=64)** — port from fork, this is the foundation for Host-KV.
3. **Host-KV safety net** — port from fork, requires paged KV.
4. **YaRN context extension** — small change, requires NVFP4 KV for full context.
5. **OOM recovery** — small change.
6. **Rewrite checkpoint at turn boundary** — small change.
7. **Token stability with `preserve_thinking=off`** — small change.
8. **Checkpoint lifecycle preservation** — small change.
9. **Tolerant tool-call recovery** — small change.
10. **Reasoning-effort tier mapping** — small change.
11. **/stats endpoint** — medium change.
12. **Monitoring dashboard** — medium change.
13. **E2E test suite** — medium change.
14. **Request-log rotation** — small change.

## Notes

- The fork is a clean divergence — upstream has its own host-KV cache implementation; the fork's
  is independently developed.
- The fork targets single RTX 5090 (32GB) with 1-8 concurrent requests.
- The fork is compiled for `sm_120a` and tuned/measured on RTX 5090.
- The fork's product contract is: one GPU, one resident model, startup-fixed 1-8 active requests,
  bounded FIFO ingress, no request preemption.

## GPU protocol

- **Claim the cards only after a written grant from the coordinator.**
- **Re-guard at the moment of claim** (nvidia-smi should show 15 MiB / 0% / no apps on both cards).
- **Release the cards immediately after the run** (verify via nvidia-smi).
- **Report one line per commit** (per the cadence rule).

## Handoff

This doc is the coordinator's survey of the gzenz/ninfer fork's optimizations. A1 should read this
doc, then start with the NVFP4 KV cache (item 1). The fork's implementation is the reference.
