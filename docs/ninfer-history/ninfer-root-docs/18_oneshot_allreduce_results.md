# Results: Vectorized 2-Rank OneShotAllReduce (2× RTX 5060 Ti)

**Status:** ARCHIVE

## 1. Executive Summary

We designed, implemented, and validated a custom **Vectorized 2-Rank One-Shot AllReduce (`OneShotAllReduce`)** over mapped pinned host memory for the 2× RTX 5060 Ti TP2 setup.

By eliminating NCCL polling and host launch jitter, per-round Target Verify latency dropped from **`36.56 ms` down to `35.12 ms`** (−1.44 ms/round), pushing overall decode throughput to **`81.92 t/s`** (515 tokens in 6.29 s, 85.6% acceptance rate) while maintaining 100% bit-identical token output.

---

## 2. Benchmark Comparison (512 Tokens on 2× RTX 5060 Ti)

| Metric | Objective 6 Baseline (NCCL AR) | Vectorized `OneShotAllReduce` | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Decode Throughput** | $80.77\text{ t/s}$ | **`81.92 t/s`** | **+1.15 t/s** |
| **Total Generation Time** | $6.34\text{ s}$ | **`6.29 s`** | **−0.05 s** |
| **Mean Round Latency** | $40.26\text{ ms}$ | **`38.74 ms`** | **−1.52 ms/round** |
| **Target Verify Latency** | $36.59\text{ ms}$ | **`35.12 ms`** | **−1.47 ms (−4.0%)** |
| **Plain Decode Latency** | $30.2\text{ ms/step}$ | **`29.4 ms/step`** | **−0.8 ms/step** |
| **Draft Acceptance Rate** | $85.6\%$ ($370/432$) | **$85.6\%$ ($370/432$)** | **Bit-Identical Determinism** |

---

## 3. Microbenchmark Comparison (10 KB & 40 KB Payloads)

| AllReduce Mechanism | 10 KB ($T=1$) | 40 KB ($T=4$ Verify) | 128 ARs / Round |
| :--- | :---: | :---: | :---: |
| **NCCL `RING_LL` (with host jitter)** | ~58.2 µs | **58.2 µs** | **7.45 ms** |
| **Dual-Thread Eager NCCL** | 12.89 µs | **26.43 µs** | **3.38 ms** |
| **CUDA Graph 128× NCCL Sequence** | 13.37 µs | **25.04 µs** | **3.21 ms** |
| **Vectorized Pinned Host One-Shot AR** | **6.35 µs** | **`8.25 µs`** | **`1.06 ms`** |

---

## 4. Key Implementation Details

1. **Host-Mapped Staging Buffers**:
   - In [`OneShotAllReduce`](file:///tmp/ninfer/src/core/multi_gpu/one_shot_allreduce.h), allocated 64 cyclic slots of mapped, portable pinned host memory (`cudaHostAllocMapped | cudaHostAllocPortable`) (~16 MB total host RAM).
2. **128-bit Vectorized CUDA Kernel with Cache Bypassing**:
   - In [`one_shot_allreduce.cu`](file:///tmp/ninfer/src/core/multi_gpu/one_shot_allreduce.cu), used PTX `st.global.wt.v4.u32` (write-through) and `ld.global.cv.v4.u32` (cache-volatile loads) to guarantee immediate cache coherency across the PCIe bus without stale L2 cache hits.
   - Used vectorized `__hadd2` instructions on 1024-thread single-CTA blocks, allowing all threads to participate in the reduction barrier.
3. **Transparent Drop-in Integration**:
   - In [`tp_group.cpp`](file:///tmp/ninfer/src/core/multi_gpu/tp_group.cpp), routed `allreduce_local_bf16` directly to `OneShotAllReduce` when `size() == 2` and `n_elems <= 65536`.

---

## 5. Verification & Git Commit

- **Unit Test**: 1,000 / 1,000 consecutive rounds verified bit-exact via [`ninfer_test_one_shot_correctness`](file:///tmp/ninfer/tests/multi_gpu/test_one_shot_correctness.cu).
- **Git Commit**: `e5d798cd` (`feat(multi_gpu): implement vectorized 128-bit OneShotAllReduce on mapped pinned host memory`) pushed to `origin/mtp-perf`.
