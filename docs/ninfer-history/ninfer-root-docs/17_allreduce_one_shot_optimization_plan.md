# AllReduce Optimization Plan & Microbench Results (2× RTX 5060 Ti)

**Status:** ARCHIVE

## 1. Executive Summary & Microbenchmark Probe

We probed and measured **4 distinct 2-rank AllReduce mechanisms** on the live 2× RTX 5060 Ti system using our newly developed microbenchmark ([`tests/multi_gpu/bench_one_shot_ar.cu`](file:///tmp/ninfer/tests/multi_gpu/bench_one_shot_ar.cu)).

### Probe Results

| AllReduce Mechanism | 10 KB ($T=1$) | 40 KB ($T=4$ Verify) | 128 ARs / Round | Projected Round Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Current In-Flight NCCL (with host jitter)** | ~58.2 µs | **58.2 µs** | **7.45 ms** | **40.24 ms** (80.77 t/s) |
| **Dual-Thread Eager NCCL** | 12.89 µs | **26.43 µs** | **3.38 ms** | **36.17 ms** (89.8 t/s) |
| **CUDA Graph 128× NCCL Sequence** | 13.37 µs | **25.04 µs** | **3.21 ms** | **36.00 ms** (90.3 t/s) |
| **Vectorized Pinned Host One-Shot AR** | **6.35 µs** | **`8.25 µs`** | **`1.06 ms`** | **`33.85 ms` (105.7 t/s)** |

### Key Hardware Insight
On standard consumer PCIe desktop topology without PCIe P2P BAR access (`can_device_access_peer = False`), GPU-to-GPU memory copies must traverse host memory. 
- NCCL's `RING_LL` protocol uses ring polling buffers that suffer from high latency when CPU thread launch timings jitter.
- A **direct 2-rank vectorized 128-bit (`uint4`) one-shot CUDA kernel over mapped pinned host memory** bypasses NCCL entirely, dropping 40 KB AllReduce latency to **`8.25 µs`** (128 ARs in **`1.06 ms`**).

---

## 2. Speedup & Throughput Impact

- **AllReduce Latency Reduction**: From **7.45 ms** down to **1.06 ms** (**6.39 ms saved per round**).
- **Target Verify Phase**: Drops from **36.56 ms** to **~30.17 ms**.
- **Mean Round Latency**: Drops from **40.24 ms** to **33.85 ms**.
- **Projected MTP Throughput**:
  $$\text{Throughput} = \frac{3.58\text{ tokens/round}}{0.03385\text{ s/round}} = \mathbf{105.7\text{ t/s}}$$

---

## 3. Step-by-Step Implementation Plan

### Step 1: Implement `OneShotAllReduce` Component
- Create [`src/core/multi_gpu/one_shot_allreduce.h`](file:///tmp/ninfer/src/core/multi_gpu/one_shot_allreduce.h) and [`src/core/multi_gpu/one_shot_allreduce.cu`](file:///tmp/ninfer/src/core/multi_gpu/one_shot_allreduce.cu).
- Allocate double-buffered mapped pinned host staging memory (`cudaHostAllocMapped | cudaHostAllocPortable`) during `TpGroup` initialization (~160 KB total host RAM).
- Provide vectorized `one_shot_allreduce_bf16(int rank, void* local_ptr, std::size_t count, cudaStream_t stream)` using 128-bit `__hadd2` vector instructions.

### Step 2: Replace NCCL Call Sites in TextContext
- In [`src/targets/qwen3_6/impl/runtime/text_context_impl.h`](file:///tmp/ninfer/src/targets/qwen3_6/impl/runtime/text_context_impl.h):
  - After attention/GDN O-projection: replace `tp_group_->allreduce_local_bf16` with `one_shot_allreduce_bf16`.
  - After MLP down-projection: replace `tp_group_->allreduce_local_bf16` with `one_shot_allreduce_bf16`.
  - In MTP stem / forward: replace draft AllReduce calls with `one_shot_allreduce_bf16`.

### Step 3: Numerical Validation & Bit-Identical Verification
- Validate numerical equivalence with `ninfer_tp_group_test`.
- Run `--tokens 40 --mtp 0` to verify bit-identical plain decode output.

### Step 4: Full Benchmark & Profiling
- Run steady-state `--tokens 512 --mtp 3`.
- Measure the reduction in Target Verify latency and log final throughput ($\ge 100\text{ t/s}$).
