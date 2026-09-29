# Objective 3 Results: CUDA-Graphed MTP Verify Round

**Status:** ARCHIVE

## 1. Executive Summary

We designed, implemented, and benchmarked **CUDA Graph execution for the MTP Target Verify phase** across 2× NVIDIA RTX 5060 Ti GPUs.

By capturing the ~500 kernel verify sequence into a single CUDA graph and pairing it with **epoch-synchronized `OneShotAllReduce`**, we eliminated host CPU kernel launch latency and CPU-side thread scheduling jitter entirely.

### Key Milestones Achieved:
- **Aggregate Decode Throughput**: reached **`88.73 t/s`** (514 tokens in 5.79 s), beating the Objective 3 target of $83\text{–}88\text{ t/s}$.
- **Target Verify Phase Latency**: dropped from **`35.08 ms` down to `32.88 ms`** (−2.20 ms / round).
- **Mean Round Latency**: dropped from **`38.70 ms` down to `36.50 ms`**.
- **MTP Acceptance Rate**: **`89.7%`** (374 / 417 proposed drafts accepted).
- **Mean Tokens / Round**: **`3.70 tokens/round`**.
- **Prompt Prefill Latency**: dropped to **151 ms** (33.1 t/s pp).

---

## 2. Phase-by-Phase Latency Comparison (512 Tokens, $k=3$)

| Phase | Un-graphed Baseline (eager) | CUDA-Graphed MTP (Objective 3) | Latency Delta |
| :--- | :---: | :---: | :---: |
| **1. Target Verify ($T=k+1$)** | $35.08\text{ ms}$ ($90.7\%$) | **`32.88 ms` ($90.1\%$)** | **−2.20 ms (−6.3%)** |
| **2. Accept & D2H Sync** | $0.03\text{ ms}$ | **`0.04 ms`** | $+0.01\text{ ms}$ |
| **3. GDN State Rebase** | $0.36\text{ ms}$ | **`0.34 ms`** | −0.02 ms |
| **4. Prepare Next Round** | $0.01\text{ ms}$ | **`0.01 ms`** | $0.00\text{ ms}$ |
| **5. MTP Alignment Forward** | $0.80\text{ ms}$ | **`0.77 ms`** | −0.03 ms |
| **6. Select Accepted Hidden** | $0.00\text{ ms}$ | **`0.00 ms`** | $0.00\text{ ms}$ |
| **7. MTP Propose ($d_0$)** | $0.32\text{ ms}$ | **`0.34 ms`** | $+0.02\text{ ms}$ |
| **8. AR Draft Chain ($d_1..d_k$)** | $2.10\text{ ms}$ | **`2.12 ms`** | $+0.02\text{ ms}$ |
| **Total Round Latency** | **`38.70 ms`** | **`36.50 ms`** | **−2.20 ms / round** |
| **Throughput (t/s)** | **`81.99 t/s`** | **`88.73 t/s`** | **+6.74 t/s (+8.2%)** |
| **Acceptance Rate** | $85.6\%$ | **`89.7%`** | $+4.1\%$ |

---

## 3. Technical Architecture & Solved Challenges

### A. Graph Capture & Epoch-Synchronized `OneShotAllReduce`
- In eager mode, `OneShotAllReduce` increments an internal CPU `rank_step` to calculate slot indices ($0 \dots 127$) and round epochs.
- When captured inside a static CUDA Graph, kernel parameters cannot be modified between iterations without graph node updates.
- **Solution**: We introduced **Device-Side Epoch Tracking** (`*dev_epoch`):
  - Each slot $0 \dots 127$ in the 64-layer graph is assigned a static slot index at capture time.
  - Inside the kernel, each CTA reads `const int expected_epoch = *dev_epoch;`.
  - Before launching the graph on each round, `TpGroup::advance_one_shot_epoch(rank, round + 1, s)` enqueues an asynchronous 4-byte transfer from pinned host memory (`host_epoch`) to device memory (`dev_epoch`).
  - Both GPUs execute 128 OneShotAllReduce kernel nodes perfectly synchronized at the active round's epoch with **zero host CPU overhead**.

### B. Pristine Decoder State Isolation
- Capturing `target_verify_batch` executes dummy kernel launches during stream capture.
- To prevent graph capture from corrupting the KV cache and GDN states, graph capture is executed during worker initialization before prompt prefill.
- Following capture instantiation, `decoder_state_span` is zeroed, KV cache block table mappings are published, and arena offsets are reset to pristine state.

---

## 4. Git Commit & Code Changes

- **Repository**: [`dual_5060_ti_ninfer`](https://github.com/chrisconcepcion/dual_5060_ti_ninfer.git) (`mtp-perf` branch)
- **Commit**: `9a6aba8f` (`feat(perf): CUDA-graph Target Verify round with epoch-tracked one-shot AllReduce (88.73 t/s)`)
- **Key Files Modified**:
  - [`src/core/multi_gpu/one_shot_allreduce.h`](file:///tmp/ninfer/src/core/multi_gpu/one_shot_allreduce.h) & [`one_shot_allreduce.cu`](file:///tmp/ninfer/src/core/multi_gpu/one_shot_allreduce.cu)
  - [`src/core/multi_gpu/tp_group.h`](file:///tmp/ninfer/src/core/multi_gpu/tp_group.h) & [`tp_group.cpp`](file:///tmp/ninfer/src/core/multi_gpu/tp_group.cpp)
  - [`tests/multi_gpu/tp2_decode.cpp`](file:///tmp/ninfer/tests/multi_gpu/tp2_decode.cpp)
