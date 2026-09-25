# 40: Prefix Caching Investigation & Technical Handoff

**Date:** 2026-08-21  
**Status:** In Progress — Root Cause Isolated to Full Attention Layer 0 (`fidx = 0`, Layer 3) KV Cache Attention at Step 225  
**Target File / Component:** `ninfer` runtime (`src/runtime/tp2/tp2_backend.cpp`, `src/targets/qwen3_6/impl/runtime/text_context_impl.h`, `src/ops/kernel/gqa_attention_decode_bf16.cuh`)

---

## 1. Executive Summary & Verification State

### Battery Test Results (`/home/intel/verify_battery.sh`):
Out of all 17 test cases across performance, latency, memory, determinism, and prefix caching, **16 PASS** and **1 FAILS**:

| Metric / Check | Target / Threshold | Measured | Status |
| :--- | :--- | :--- | :--- |
| **Prompt processing (pp)** | $\ge 28$ t/s | **31.90 t/s** | **PASS** |
| **Plain decode** | $\ge 30$ t/s | **35.40 t/s** | **PASS** |
| **Plain decode step** | $\le 33$ ms | **29.00 ms** | **PASS** |
| **MTP $k=3$ throughput** | $\ge 90$ t/s | **93.48 t/s** | **PASS** |
| **MTP acceptance rate** | $\ge 70\%$ | **85.60%** | **PASS** |
| **Mean $a$/round** | $\ge 2.1$ | **2.57** | **PASS** |
| **Round phase total** | $\le 45$ ms | **38.05 ms** | **PASS** |
| **Verify phase ($T=4$)** | $\le 40$ ms | **34.85 ms** | **PASS** |
| **VRAM per rank** | $\le 11500$ MB | **9059 MB** | **PASS** |
| **Determinism (2 identical runs)** | exact match | **yes** | **PASS** |
| **A2 Token Identity ($MTP == plain$)** | exact match | **yes** | **PASS** |
| **Draft vocab active** | $W8G32$ verified | **yes** | **PASS** |
| **Sampling determinism ($s=42$)** | exact match | **yes** | **PASS** |
| **Sampling divergence ($s=43$)** | divergent output | **yes** | **PASS** |
| **I8 KV Gate** | $\ge 80$ t/s | **87.43 t/s** ($77.1\%$ a) | **PASS** |
| **Prefix 2nd prefill speedup** | $\le 1000$ ms vs $\ge 5000$ ms | **230.1 ms vs 6721.3 ms** ($29\times$) | **PASS** |
| **Prefix skip check** | $\le 90\%$ ms | **yes** | **PASS** |
| **Prefix identity TC-P(a)** | `prefix_on == prefix_off` | **FAIL (tokens differ)** | **FAIL** |

---

## 2. The Failing Test Case: TC-P(a)

### Test Flow:
1. **Request 1:** Prompt of 225 tokens. Runs MTP $k=3$ generation for 64 tokens.
2. **Request 2 (`prefix_on`):** Prompt of 232 tokens (shares exact 225-token prefix with Request 1 + 7 new tokens).
   - Skips tokens $0 \dots 224$ via prefix cache hit.
   - Restores GDN linear attention slot 0 from slot 7 (`st.cache_slot = 2k + 1 = 7`).
   - Restores prompt hidden states `st.cached_ph` to `st.mtp_ph`.
   - Prefills tokens $225 \dots 231$ (7 tokens in 235 ms).
   - Decodes 64 tokens.
3. **Request 2 (`prefix_off`):** Same 232-token prompt run with `--no-prefix-cache`.
   - Prefills all 232 tokens sequentially ($0 \dots 231$ in 6.9 s).
   - Decodes 64 tokens.
4. **Failure Condition:** The generated 64 tokens of Request 2 under `prefix_on` do not match the generated 64 tokens under `prefix_off`.

---

## 3. Bugs Discovered & Already Fixed

### Bug 1: `OneShotArgmax` Step Desynchronization
- **Cause:** `TpGroup::reset_one_shot_step(rank)` only called `impl_->one_shot->reset_step(rank)` and omitted `impl_->one_shot_argmax->reset_step(rank)`.
- **Effect:** When Request 2 started, `OneShotArgmax` had residual step counts from Request 1, causing token selection desynchronization.
- **Fix:** Added `if (impl_->one_shot_argmax) { impl_->one_shot_argmax->reset_step(rank); }` in [`src/core/multi_gpu/tp_group.cpp`](file:///tmp/ninfer/src/core/multi_gpu/tp_group.cpp#L285-L292).
- **Verification:** Step 0 of Request 2 in `prefix_off` is now 100% bit-for-bit identical to Step 0 of Request 1 (`hid=[bf2c bf5c 3f25 3f9b]`).

### Bug 2: `st.mtp_ph` Allocation Order in Persistent Scope
- **Cause:** In [`src/runtime/tp2/tp2_backend.cpp`](file:///tmp/ninfer/src/runtime/tp2/tp2_backend.cpp#L690-L735), `st.cached_ph.copy_to(st.mtp_ph)` was executed before `st.mtp_ph` was allocated in the request's persistent arena.
- **Fix:** Moved `st.persistent.alloc(...)` for `st.mtp_ph`, `st.mtp_pos`, `st.mtp_mh`, `mtp_ids` before the prefix restore check.

### Bug 3: Snapshot Cache Slot Collision
- **Cause:** `cache_slot` was originally set to 1, which collided with speculative verify round snapshots (slots $1 \dots k$) during Request 1 decode.
- **Fix:** Set `cache_slot = 2*k + 1 = 7` with `slot_count = 8` in [`src/runtime/tp2/tp2_backend.cpp`](file:///tmp/ninfer/src/runtime/tp2/tp2_backend.cpp#L181-L205).

---

## 4. Root Cause Isolation & Probe Findings

We placed synchronous layer probes across the model to trace where `prefix_on` and `prefix_off` diverge during Request 2:

### Probe Point 1: Layers 0, 1, 2 (GDN Linear Attention Layers)
At Step 225 of Request 2 (the first prefilled token after the prefix hit):
- **Input `x` to Layer 0:** `[bd63 bbce 3c3f 3d05]` in both `prefix_on` and `prefix_off`.
- **Output `x` from Layer 0:** `[bdac bc62 3c94 3cfa]` in both `prefix_on` and `prefix_off`.
- **Output `x` from Layer 1:** `[3ca8 bd07 bc1d 3bf0]` in both `prefix_on` and `prefix_off`.
- **Output `x` from Layer 2:** Identical across both runs.

> [!NOTE]
> **Conclusion on GDN:** GDN Linear Attention state saving (`copy_slot(0, 7)`) and restoration (`copy_slot(7, 0)`) are **100% bit-for-bit exact**. There is zero divergence in the linear attention conv or recurrent states.

---

### Probe Point 2: Layer 3 (`fidx = 0`, First Full Attention Layer)
Layer 3 is the first standard GQA transformer layer in Qwen 3.6 (3 GDN layers followed by 1 Full Attention layer).

At Step 225:
- **`qn_v` entering `ops::gqa_attention`:** `[bef7 c010 c026 bfa7]` (**100% BIT-FOR-BIT IDENTICAL**).
- **`kn_v` entering `ops::gqa_attention`:** `[3f4a bfa5 c020 3dfd]` (**100% BIT-FOR-BIT IDENTICAL**).
- **`v` entering `ops::gqa_attention`:** **100% BIT-FOR-BIT IDENTICAL**.
- **`active_gqa_envelope`:** `Envelope{226, 226}` in both runs.
- **`cache_positions`:** `[225]` in both runs.
- **`kv_table_rows`:** `[0]` in both runs.

#### The Divergence:
Immediately upon returning from `ops::gqa_attention(..., a_v, s)`:
- **`prefix_on`:** `a_v = [bdda 3e3d 3e0c 3e00]` (rank 0), `[3e32 beeb be10 be3f]` (rank 1)
- **`prefix_off`:** `a_v = [3cf9 3d8f 3e99 3f6f]` (rank 0), `[3f04 bf3f bd75 bf20]` (rank 1)

Output `x` leaving Layer 3:
- **`prefix_on`:** `x = [3b00 bd63 3ab0 3c50]`
- **`prefix_off`:** `x = [3c9e bccb bd4c 3c97]`

---

## 5. Mechanism Analysis: Why `gqa_attention` Differs

`ops::gqa_attention` takes only:
1. `qn_v` (identical)
2. `kn_v` (identical)
3. `v` (identical)
4. `batch_text_kv_` (the paged KV cache containing keys $0 \dots 224$ and values $0 \dots 224$)

Since inputs 1, 2, and 3 are identical, **the difference in `a_v` is caused entirely by the KV cache data (`batch_text_kv_`) or its reading in `gqa_attention_small_t_tc_partial_bf16_kernel`**.

### Detailed KV Cache Lifecycle:
1. **Request 1 Prefill:**
   - Sequential prefill steps $0 \dots 224$ write keys and values for positions $0 \dots 224$ into physical pages 0, 1, 2, and 3 (offsets 0..32 of page 3).
2. **Request 1 Generation (64 decode steps):**
   - Speculative verify rounds (`target_verify_batch`) write keys and values for generated positions $225 \dots 288$ into physical page 3 (offsets 33..63) and physical page 4 (offsets 0..32).
3. **Request 2 `prefix_off`:**
   - Runs prefill for all positions $0 \dots 231$.
   - Steps $0 \dots 224$ overwrite physical pages 0, 1, 2, and page 3 (offsets 0..32) with fresh keys/values.
   - At step 225, `gqa_attention` reads keys $0 \dots 224$ and the new key 225.
4. **Request 2 `prefix_on`:**
   - Skips steps $0 \dots 224$.
   - At step 225, `gqa_attention` reads keys $0 \dots 224$ from `batch_text_kv_` (as populated by Request 1) and new key 225.

---

## 8. Progress & Findings from Recent Investigation (Last 30–40 Minutes)

### 8.1 Build & Linker Invariants Validated
- **Link Graph Resolution:** Resolved the cross-library dependency between `ninfer_core` and `ninfer_ops` by eliminating the unneeded `tp_axpy_bf16` symbol call from `tp_group.cpp`.
- **Zero-Warning Clean Build:** Re-verified with `make -j24` that all 100% of targets (unit tests, benchmarks, servers, and CLI) build completely clean with zero compiler/linker warnings.

---

### 8.2 Small-T GQA Kernel (`gqa_attention_decode_bf16.cuh`) Mathematical Trace

We traced the execution of `gqa_attention_small_t_tc_partial_bf16_kernel` specifically for Step 225 of Request 2 (the divergence point):

1. **Parameters at Step 225:**
   - `tokens = 1` ($T=1$), `pos = [225]`, `first_pos = 225`, `last_pos = 225`.
   - `logical_capacity = 226`, `window = 226`.
   - Head geometry for TP2: `QHeads = 12`, `KVHeads = 2`, `GroupSize = 6`, `DecodeSplitScale = 2`.
   - `active_splits = 4` (from `gqa_small_t_active_splits`).
   - `logical_tiles = div_up(226, 32) = 8`.
   - `units_per_split = div_up(8, 4) = 2 tiles = 64 keys`.

2. **Split Key Ranges:**
   - **Split 0:** `split_start = 0`, `split_end = 64` (Tile 0 & 1, Page 0).
   - **Split 1:** `split_start = 64`, `split_end = 128` (Tile 2 & 3, Page 1).
   - **Split 2:** `split_start = 128`, `split_end = 192` (Tile 4 & 5, Page 2).
   - **Split 3:** `split_start = 192`, `split_end = 226` (Tile 6 & 7, Page 3).

3. **Split 3 Tile Loading Execution:**
   - `first_tile = 192`, `key_blocks = 2`.
   - `first_page = 192 >> 6 = 3`.
   - `page_count = ((226 - 1) >> 6) - 3 + 1 = 1` $\implies$ loads `physical_pages_s[0] = block_table[3]`.
   - **Tile 6 (`kb = 0, k0 = 192`):**
     - Keys $192 \dots 223$ load from `cache_k` at Page 3, offsets $0 \dots 31$.
     - `from_new = false` for all keys.
   - **Tile 7 (`kb = 1, k0 = 224`):**
     - Key $224$: `new_token = 224 - 225 = -1` $\implies$ `from_new = false`. Loads from `cache_k` at Page 3, offset 32.
     - Key $225$: `new_token = 225 - 225 = 0` $\implies$ `from_new = true`. Loads directly from `input.k` (token 225).
     - Keys $226 \dots 255$: `key >= split_end (226)` $\implies$ zeroes `k_dst`/`v_dst` via `store_vec(0)`.

4. **Causal Masking:**
   - For row 0 (`token0 = 0`, `qabs0 = pos[0] = 225`):
     - Keys $\le 225$ have valid scores scaled by `kAttnScale`.
     - Keys $> 225$ (keys $226 \dots 255$) are masked with `-CUDART_INF_F`.

> [!IMPORTANT]
> **Kernel Logic Invariant:** The GQA kernel indexing math for `split_start`, `split_end`, `first_tile`, `physical_page`, and `from_new` is mathematically identical between `prefix_on` and `prefix_off`. Therefore, the kernel is reading the exact same physical addresses. The data stored at those physical addresses in `cache_k` / `cache_v` is what differs.

---

### 8.3 Exact Lifecycle Difference in KV Storage

| Phase | `prefix_off` | `prefix_on` |
| :--- | :--- | :--- |
| **Request 1 Prefill ($0 \dots 224$)** | Writes keys $0 \dots 224$ into Pages $0 \dots 3$ | Writes keys $0 \dots 224$ into Pages $0 \dots 3$ |
| **Request 1 MTP Decode (64 steps)** | Writes keys $225 \dots 288+$ into Pages $3 \dots 4+$ | Writes keys $225 \dots 288+$ into Pages $3 \dots 4+$ |
| **Request 2 Start** | Does **not** preserve cache | Restores GDN slot 7 $\to$ 0; skips steps $0 \dots 224$ |
| **Request 2 Prefill ($0 \dots 224$)** | **Overwrites** Pages $0 \dots 3$ with fresh keys $0 \dots 224$ | **Skips** prefill; reads whatever remains in Pages $0 \dots 3$ |
| **Step 225 Attention** | Reads freshly written keys $0 \dots 224$ | Reads Request 1 leftover keys in Pages $0 \dots 3$ |

---

### 8.4 Narrowed Hypotheses for the KV Cache Mismatch

1. **Hypothesis 1: In-Place Mutation during Request 1 Decode**
   During Request 1 MTP rounds, kernels such as `ops::rope`, `ops::gqa_attention`, or MTP alignment passes may have performed an in-place mutation or out-of-bounds write that corrupted positions $0 \dots 224$ in `st.decoder->text_kv`.
2. **Hypothesis 2: `mtp_kv` vs `text_kv` Page Overlap / Table Collisions**
   `mtp_kv_alloc` and `kv_alloc` are allocated from separate pools (`text_kv.pool()` and `mtp_cache()->pool()`), but verify if table row 0 bindings in `st.round->text_kv_table_row` vs `st.round->backend_kv_table_row` share physical page IDs or if MTP decode writes to the text KV cache instead of the MTP cache.
3. **Hypothesis 3: Numerical Sensitivity / Unsynchronized Cache Writes**
   During Request 1 prefill, keys $0 \dots 224$ were written asynchronously with `cp_async` or `store_vec`. Check if stream synchronization before Request 1 generation allowed all prefill KV writes to flush.

---

### 8.5 Verification & Debug Recipe for the Next Agent

To definitively prove if/when Pages $0 \dots 3$ are modified:
1. **Insert KV Cache Checksum Probe:**
   In `src/runtime/tp2/tp2_backend.cpp`, compute a CRC32 or SHA256 of `st.decoder->text_kv` physical pages $0 \dots 3$:
   - Probe A: Immediately after Request 1 Prefill (Step 224).
   - Probe B: Immediately after Request 1 Generation completes (after 64 decode steps).
   - Probe C: At start of Request 2 before Step 225.
   - Probe D: During Request 2 `prefix_off` after Step 224.
2. If `Probe A == Probe D` but `Probe A != Probe B`, then Request 1 generation corrupted the prefix cache.
3. If `Probe A == Probe B == Probe C == Probe D`, then the physical memory is identical, and the difference is in the GPU L2 cache state or block table mapping.

---

## 7. How to Run & Verify

1. **Build:**
   ```bash
   cd /tmp/ninfer/build && make -j24 ninfer_tp2_decode_test
   ```
2. **Reproduce Single Test:**
   ```bash
   LONGPROMPT="The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. "
   P2="$LONGPROMPT And to summarize in one word."

   # Prefix ON
   /tmp/ninfer/build/tests/ninfer_tp2_decode_test --artifact /home/intel/models/qwen3_8_27b.ninfer --mtp 3 --tokens 64 --ctx 4096 --prompt "$LONGPROMPT" --prompt2 "$P2"

   # Prefix OFF
   /tmp/ninfer/build/tests/ninfer_tp2_decode_test --artifact /home/intel/models/qwen3_8_27b.ninfer --mtp 3 --tokens 64 --ctx 4096 --prompt "$LONGPROMPT" --prompt2 "$P2" --no-prefix-cache
   ```
3. **Full Battery Run:**
   ```bash
   /home/intel/verify_battery.sh
   ```

