# Doc 44: MTP Acceptance Regression Analysis (94 → 80 t/s)

**Date**: 2026-08-22
**Status**: Analysis only — no code changes, no builds, no test runs
**Regression window**: `184e9d15` (93.48 t/s, 85.6% acceptance) → `0a3b50e3` (80.14 t/s, 67.7% acceptance)

---

## Summary

MTP k=3 throughput dropped from **93.48 t/s to 80.30 t/s** between commits `184e9d15` and `0a3b50e3` (TC-P fix). Round time is actually *faster* (37.74ms vs 38.05ms). The entire regression is from **MTP acceptance rate dropping from 85.6% to 67.7%** — the draft head is proposing worse tokens.

This is a **quality regression**, not a performance regression. The verify phase is faster, but fewer draft tokens pass target verification.

---

## Measured Data

| Metric | Before (184e9d15) | After (0a3b50e3) | Delta |
|--------|-------------------|------------------|-------|
| MTP t/s | 93.48 | 80.30 | -14.1% |
| Acceptance rate | 85.6% | 67.7% | -17.9 pp |
| Mean a/round | 2.569 | 2.030 | -0.54 |
| Tokens/round | 3.56 | 3.03 | -0.53 |
| Round time (ms) | 38.05 | 37.74 | -0.8% |
| Verify phase (ms) | 34.85 | 34.59 | -0.7% |
| AR Draft Chain (ms) | 2.05 | 2.05 | 0% |

**Key insight**: Every phase is the same or faster. The only thing worse is acceptance rate.

---

## Changes in Commit 0a3b50e3

The commit touched 7 files. Non-probe code changes:

### 1. `tp_group.cpp` — NCCL residual fold removed (later restored in WO-A)
- Removed `tp_axpy_bf16` after `ncclAllReduce` in `allreduce_local_bf16`
- Added `reset_one_shot_argmax_step` to `reset_one_shot_step`
- **Status**: Residual fold restored in `60eb4180` (WO-A). Not the cause.

### 2. `tp2_backend.cpp` — 6 code changes (excluding probes)

#### Change A: Position indexing (`cur_F` → `cur_F_mtp`)
```cpp
// Old (184e9d15):
int cur_F = plen - 1;  // set after prefill
// MTP loop: const int F = cur_F;  // F = plen - 1

// New (0a3b50e3):
int cur_F_mtp = plen;  // initialized at MTP loop start
// MTP loop: const int F = cur_F_mtp;  // F = plen
```
**Impact**: `F` is the base position for the MTP verify batch. It feeds into `speculative_prepare_verify_inputs` which computes RoPE positions for the verify batch.

#### Change B: GDN slot geometry
```cpp
// Old: cache_slot = k + 2 = 5 (for k=3), slot_count = 6
// New: cache_slot = 2*k + 1 = 7 (for k=3), slot_count = 8
```
**Impact**: More GDN slots allocated. Verify columns still use slots 1..k+1 (1-4). Cache slot moved from 5 to 7.

#### Change C: KV allocation (`publish_mapping` → `materialize_pages`)
```cpp
// Old: state->kv_alloc.publish_mapping(ctx.stream);
// New: state->kv_alloc.materialize_pages(pages, ctx.stream);
```
**Impact**: `materialize_pages` allocates real physical pages and writes them to the block table. `publish_mapping` only copies existing page IDs. This was the block table fix — previously all page IDs were zero, mapping every logical page to physical page 0 (64-token KV window).

#### Change D: GDN zeroing (full memset → per-slot)
```cpp
// Old: cudaMemsetAsync(st.decoder_state_span.data, 0, ...)  // zero everything
// New: zero_slot(0) + loop zero_slot(1..slot_count-1, skipping cache_slot)
```
**Impact**: Old code zeroed the entire state span including cache slot. New code preserves cache slot for prefix reuse.

#### Change E: New persistent allocations
```cpp
st.mtp_ph  = st.persistent.alloc(DType::BF16, {5120, plen});
st.mtp_pos = st.persistent.alloc(DType::I32, {plen});
st.mtp_mh  = st.persistent.alloc(DType::BF16, {5120, plen});
Tensor mtp_ids = st.persistent.alloc(DType::I32, {plen});
```
**Impact**: Additional VRAM usage for MTP head position history. ~4 * 448 bytes for 224-token prompt = negligible.

#### Change F: sync_bar additions (TC-P AR reset fix)
```cpp
CUDA_CHECK(cudaStreamSynchronize(s));
sync_bar.arrive_and_wait();
backend.group().reset_one_shot_step(rank);
CUDA_CHECK(cudaStreamSynchronize(s));
sync_bar.arrive_and_wait();
```
**Impact**: Ensures AR state is clean between requests. Correctness fix, shouldn't affect acceptance.

---

## Ranked Hypotheses

### H1: Position Indexing Bug (F = plen vs plen-1) — **PRIMARY SUSPECT**

**What changed**: `cur_F_mtp = plen` instead of inheriting `cur_F = plen - 1`

**Why it matters**: 
- `F` is passed to `write(st.base_pos, F)` then to `speculative_prepare_verify_inputs`
- The kernel computes: `positions[off] = base_positions[row] + j`
- Old code: positions = [plen-1, plen, plen+1, plen+2] for k=3
- New code: positions = [plen, plen+1, plen+2, plen+3] for k=3

**Mechanism**:
1. The anchor token (t0) was generated at position `plen-1` during prefill
2. The KV cache stores t0 at position `plen-1`
3. The verify batch should verify anchor at position `plen-1` (matching KV cache)
4. New code verifies anchor at position `plen` (off by one)
5. Wrong position → wrong RoPE → wrong attention weights → wrong target predictions
6. Target predictions are degraded → fewer draft tokens match → lower acceptance

**Expected impact**: ~15-20% acceptance drop (matches observed 85.6% → 67.7%)

**How to verify**: 
- Change `cur_F_mtp = plen` to `cur_F_mtp = plen - 1`
- Run battery, check if acceptance returns to ~85%

**Risk**: Low. Single line change. If wrong, acceptance won't improve.

---

### H2: GDN Slot Geometry Change (cache_slot 5 → 7)

**What changed**: `cache_slot = k+2` → `2*k+1`, `slot_count = cache_slot+1`

**Why it might matter**:
- Old geometry: slots 0-5 (0=committed, 1-4=verify, 5=cache)
- New geometry: slots 0-7 (0=committed, 1-4=verify, 5-6=unused, 7=cache)
- Zeroing loop zeros slots 1-6 (was 1-4)

**Mechanism**:
- If the verify batch or rebase logic assumes a specific slot layout, the extra unused slots (5-6) could cause state corruption
- The `cur_slot = a` value (0-3) is used as source slot for next round — still valid in both geometries

**Why it's unlikely**:
- Verify columns use slots 1-4 in both geometries
- `cur_slot` (0-3) is always within valid range
- Zeroing loop correctly skips cache_slot

**How to verify**: Revert to `cache_slot = k+2` while keeping other changes

**Risk**: Medium. Could interact with H1.

---

### H3: KV Block Table Fix (materialize_pages)

**What changed**: `publish_mapping` → `materialize_pages`

**Why it might matter**:
- Old code: block table was all zeros → every logical page mapped to physical page 0
- This meant a **64-token KV window** (only first page)
- New code: real physical page IDs → full KV context

**Mechanism**:
- With 64-token window, the model only saw the last 64 tokens of context
- This could actually *help* acceptance if the relevant context was in the last 64 tokens
- Full context might introduce noise or conflicting signals

**Why it's unlikely**:
- The old behavior was a bug — the model was operating with truncated context
- Full context should produce *better* predictions, not worse
- The acceptance drop is too large to explain by context quality alone

**How to verify**: Compare acceptance with/without block table fix on same position indexing

**Risk**: Low. The block table fix is correct — reverting would be a step backward.

---

### H4: GDN Zeroing Change (memset → per-slot)

**What changed**: Full state span zero → per-slot zero (skipping cache_slot)

**Why it might matter**:
- Old code zeroed everything including unused memory between slots
- New code only zeros specific slots

**Mechanism**:
- If there's uninitialized memory between slots, the GDN kernels could read garbage
- The per-slot zeroing might miss some state

**Why it's unlikely**:
- GDN state is tightly packed (no gaps between slots)
- `zero_slot` is the same kernel used in both paths
- The full memset was overkill — per-slot is more precise

**How to verify**: Add full memset back and compare

**Risk**: Low.

---

### H5: New Persistent Allocations (VRAM pressure)

**What changed**: Added mtp_ph, mtp_pos, mtp_mh, mtp_ids allocations

**Why it might matter**:
- Additional VRAM usage could cause fragmentation or eviction
- Could affect CUDA stream scheduling

**Why it's unlikely**:
- Allocations are small (~4KB for 224-token prompt)
- VRAM is not pressure-constrained (9GB used of 16GB)
- Round time is unchanged, so no scheduling impact

**Risk**: Negligible.

---

## Recommended Investigation Order

1. **H1 first** (position indexing): Single line change, highest confidence, matches observed impact
2. **H2 second** (GDN geometry): If H1 doesn't fully explain, check slot layout
3. **H3-H5**: Only if H1+H2 don't resolve the regression

---

## What This Means for the Agent's M6 Work

The agent is working on M6 (batched/chunked prefill). The regression is in the MTP round loop, which is separate from prefill. The agent's M6 work should not be affected by this regression — it's in the decode path, not the prefill path.

However, if the agent's M6 changes touch `cur_F_mtp` or position indexing, the same bug could appear in batched prefill.

---

## Appendix: Code References

- Position indexing: `src/runtime/tp2/tp2_backend.cpp:L984` (`cur_F_mtp = plen`)
- Verify batch: `src/runtime/tp2/tp2_backend.cpp:L1001-1008` (`speculative_prepare_verify_inputs`)
- Position kernel: `src/ops/kernel/speculative_round.cuh:L18-37` (`speculative_prepare_verify_inputs_kernel`)
- GDN slots: `src/runtime/tp2/tp2_backend.cpp:L184-185` (`cache_slot`, `slot_count`)
- Block table: `src/core/paged_kv_cache.cpp:L347-390` (`materialize_pages`, `publish_mapping`)
