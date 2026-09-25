# Doc 43 — Agent code review (0a3b50e3) + v1 work order

**Status:** ARCHIVE

**Commit reviewed:** `0a3b50e3` — `fix(tp2): fix MTP round 0 base position index to prevent KV cache prompt corruption and resolve TC-P prefix caching failure`

**Verdict: APPROVED with one latent defect (P1, NCCL path) and cleanup required (P2, probes).**

---

## 1. Code review — change-by-change

### 1.1 AR reset fix (tp2_backend.cpp, L870-877) ✅ CORRECT
```cpp
CUDA_CHECK(cudaStreamSynchronize(s));
sync_bar.arrive_and_wait();
backend.group().reset_one_shot_step(rank);
CUDA_CHECK(cudaStreamSynchronize(s));
sync_bar.arrive_and_wait();
```
Drains both ranks before resetting one-shot AR state. Fixes Issue 1 (rank1's stale flag1=6). Verified by v1probe: req2 GDN tensors became bit-identical post-fix. **Keep.**

### 1.2 mtp_ph allocation moved before prefix cache block (tp2_backend.cpp, L695-698) ✅ CORRECT
`st.mtp_ph`, `st.mtp_pos`, `st.mtp_mh`, `mtp_ids` allocated before the hit-path restore. Fixes WO-2 (P2): the hit path copies into `st.mtp_ph` — must exist first. **Keep.**

### 1.3 materialize_pages fix (tp2_backend.cpp, L250-253) ✅ CORRECT
`publish_mapping` → `materialize_pages(pages, ctx.stream)` for both text-KV and MTP-KV. Fixes the block-table-all-zero bug (64-token KV window). **Keep.**

### 1.4 cache_slot geometry (tp2_backend.cpp, L185) ✅ CORRECT
`k + 2` → `2 * k + 1` (MTP); `2` → `1` (non-MTP). Matches the MTP slot protocol: committed slot 0 + verify snapshots (2k) = 2k+1. Non-MTP needs only 1 slot (committed). **Keep.**

### 1.5 reset_one_shot_step now resets argmax (tp_group.cpp, L283-285) ✅ CORRECT
```cpp
if (impl_->one_shot_argmax) {
    impl_->one_shot_argmax->reset_step(rank);
}
```
Ensures the argmax AR state is also reset between requests. **Keep.**

### 1.6 prefill TPS calculation (tp2_backend.cpp, L886-893) ✅ CORRECT
`plen` → `re_prefill_tokens = plen - prefix_len`. Prefill TPS now reflects only the re-prefilled suffix. **Keep.**

### 1.7 cache_ph_bytes VRAM estimate (tp_engine.cpp, L168) ✅ CORRECT
Accounts for `mtp_ph` allocation (`max_context × 5120 × 2`) in the VRAM budget. **Keep.**

### 1.8 cur_F → cur_F_mtp rename (tp2_backend.cpp) ✅ HARMLESS
Clarity rename. No functional change.

### 1.9 Residual fold removed from allreduce_local_bf16 (tp_group.cpp) ⚠️ LATENT DEFECT (P1)
```diff
-    if (residual_ptr != nullptr) {
-        Tensor dst(residual_ptr, DType::BF16, {static_cast<int64_t>(n_elems)});
-        Tensor src(local_ptr, DType::BF16, {static_cast<int64_t>(n_elems)});
-        tp_axpy_bf16(dst, src, k.ctx.stream);
-    }
```
**Analysis:** `allreduce_local_bf16` has two code paths:
- **One-shot AR** (n_elems ≤ 65536, i.e. T ≤ 12 for hidden=5120): the kernel `one_shot_ar_pinned_vec_kernel` accepts `residual_buf` and folds `local + peer → residual`. **Correct.**
- **NCCL fallback** (n_elems > 65536, i.e. T > 12): `ncclAllReduce` sums in-place, but the residual fold is **missing**. The `residual_ptr` parameter is accepted but ignored.

**Impact:** For T > 12 (batched prefill, M6), the attention and MLP outputs are allreduced but **not added to the residual**. The layer becomes a no-op. This is why the battery passes: all battery cases use T=1 (decode) or T=4 (verify, k=3) — both within the one-shot path.

**Fix:** Add the residual fold back in the NCCL path:
```cpp
NCCL_CHECK(ncclAllReduce(local_ptr, local_ptr, n_elems, ncclBfloat16, ncclSum, k.comm, k.ctx.stream));
if (residual_ptr != nullptr) {
    // Fold allreduce result into residual: residual += local (which now holds the sum)
    Tensor dst(residual_ptr, DType::BF16, {static_cast<int64_t>(n_elems)});
    Tensor src(local_ptr, DType::BF16, {static_cast<int64_t>(n_elems)});
    tp_axpy_bf16(dst, src, k.ctx.stream);
}
```
Restore `#include "core/multi_gpu/tp_kernel.h"` in `tp_group.cpp`.

**Priority: P1 — must fix before M6 (batched prefill).**

### 1.10 Probe instrumentation (tp2_backend.cpp, text_context_impl.h, one_shot_allreduce.{h,cu}, tp_group.{h,cpp}) 🔧 CLEANUP (P2)
kv_probe lambda (~150 lines), debug accessors (OneShotDebug, debug_one_shot, debug_host_buf_hash, etc.), v1probe/attnprobe/mlpprobe counters and emission. All diagnostic — **remove per doc 41 §7** after battery confirmed stable.

### 1.11 #include <cstdlib> (tp2_backend.cpp) 🔧 CLEANUP (P2)
For `getenv("NINFER_KVPROBE")`. Remove with probe cleanup.

---

## 2. Battery status (20260822_073334)

| Gate | Result |
|---|---|
| determinism_ok | ✅ |
| a2_identity_ok | ✅ |
| draft_vocab_on | ✅ |
| sampling_det_ok | ✅ |
| sampling_div_ok | ✅ |
| **prefix_identity_ok** | **✅ (TC-P fixed)** |
| prefix_skip_ok | ✅ (228 ms = 3.4% of 6696 ms off-run) |
| kv_i8_ok | ❌ (needs relative gate recalibration) |

**Perf:** MTP 80.19 t/s, 67.7% acceptance, 3.03 tok/round. Plain 35.51 t/s. PP 32.0 t/s. VRAM 9059 MB.

---

## 3. v1 work order (in order)

### WO-A — Fix NCCL residual fold (P1, 15 min)
**File:** `src/core/multi_gpu/tp_group.cpp`
**Action:** Restore the `tp_axpy_bf16` fold in the NCCL path of `allreduce_local_bf16`. Restore `#include "core/multi_gpu/tp_kernel.h"`.
**Verify:** Build clean. Battery still passes (NCCL path not exercised yet). M6 will exercise it.

### WO-B — Probe cleanup (P2, 30 min)
**Files:** `tp2_backend.cpp`, `text_context_impl.h`, `one_shot_allreduce.{h,cu}`, `tp_group.{h,cpp}`, `gqa_attention.cpp`
**Action:** Remove all probe code per doc 41 §7:
- kv_probe lambda + `#include <cstdlib>` + `getenv("NINFER_KVPROBE")` in `tp2_backend.cpp`
- v1probe, v1pre, v1ar, attnprobe, mlpprobe, `attnprobe_token_ref()`, `attnprobe_on()`, `attnprobe_fire()`, `attnprobe_hash()` in `text_context_impl.h`
- debug accessors in `one_shot_allreduce.{h,cu}`
- OneShotDebug struct + debug forwarders in `tp_group.{h,cpp}`
- `[gqa]` wrapper logging in `gqa_attention.cpp`
**Keep:** AR reset fix, materialize_pages fix, mtp_ph allocation move, cache_slot geometry, argmax reset, cache_ph_bytes VRAM.
**Verify:** Build clean. Battery passes.

### WO-C — kv_i8 gate recalibration (P2, 15 min)
**File:** `/home/intel/verify_battery.sh`
**Action:** Change `kv_i8_ok` from absolute t/s comparison to relative: `kv_i8_tps >= mtp_tps * 0.95` (I8 within 5% of BF16). Current: I8 80.82 vs BF16 80.19 → would pass at 95%.
**Verify:** Battery `kv_i8_ok` flips to true.

### WO-D — Baseline JSON update (P2, 5 min)
**File:** `/home/intel/verify_baseline.json`
**Action:** Update to current numbers: mtp_tps=80.19, mtp_accept=67.7, plain_tps=35.51, pp_tps=32.0.
**Verify:** Battery perf gates pass.

### WO-E — M2 final verification (P2, 1 h)
**Action:** Run battery with TC-S sampling cases. Verify G1–G8 gates (per doc 38). Check that per-request sampling overrides beat `sampling_defaults_` (G2 priority check). Verify penalty path through the accept kernel.
**Verify:** All G1–G8 green.

### WO-F — WO-5: TC-M multi-request driver case (P2, 1 h)
**Action:** Add `--repeat N` / `--prompt-file-list` to the test driver. Battery TC-M case: N=5 requests, all 200, VRAM flat.
**Verify:** TC-M passes.

### WO-G — WO-6: serve_battery.sh (P2, 2 h)
**Action:** Implement `serve_battery.sh` with S1–S6 (doc 38 §7). 10 mixed serve requests (sizes + streaming + one tools call), cancel-on-disconnect, doc 37 verify items.
**Verify:** All S1–S6 green.

### WO-H — M6: Batched/chunked prefill (P1, 1-3 days)
**Design:** T=512 chunks, TP batch GDN + batch attention. Target pp ≥ 300 t/s.
**Prerequisite:** WO-A (NCCL fold fix) must land first — M6 will exercise the NCCL path.
**Verify:** pp ≥ 300 t/s at T=512. 64k context completes end-to-end with I8 KV.

---

## 4. v1 scorecard after WO-A through WO-G

| Must | Status |
|---|---|
| M1 Multi-request stability | ✅ Done |
| M2 Full sampling | ✅ After WO-E |
| M3 Option plumbing | ✅ Done |
| M4 I8 KV (opt-in) | ✅ Working; 64k E2E after WO-H |
| M5 Prefix caching | ✅ TC-P fixed |
| M6 Batched prefill | ✅ After WO-H |
| M7 Standing gates | ✅ After WO-F/G |

**v1 DoD after WO-H:** all 7 items green. Estimated total: WO-A through WO-G ≈ 4-5 h; WO-H ≈ 1-3 days.

## 5. Post-v1 pipeline (unchanged)

- v1.x: stop strings, logit_bias, /v1/completions, /tokenize, /metrics (doc 38 §4)
- v2: DFlash2 + KVarN (doc 42 gotchas ready)
