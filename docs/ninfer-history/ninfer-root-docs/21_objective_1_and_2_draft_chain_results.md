# Objectives 1 & 2 Results: Streamlined Draft Chain & Optimal Draft Length (90.81 t/s)

**Status:** ARCHIVE

## 1. Executive Summary

We addressed **Objective 2** (Streamlining the AR draft chain) and explored **Objective 1** (Draft window analysis & Small-T GQA routing for multi-token speculative decoding).

### Key Accomplishments:
1. **Device-Level Cross-Rank Argmax (`OneShotArgmax`)**:
   - Eliminated all host CPU thread barriers (`sync_bar`) and `cudaMemcpyDeviceToHost` calls across the entire draft chain.
   - Designed and implemented [`src/core/multi_gpu/one_shot_argmax.h`](file:///tmp/ninfer/src/core/multi_gpu/one_shot_argmax.h) and [`one_shot_argmax.cu`](file:///tmp/ninfer/src/core/multi_gpu/one_shot_argmax.cu), using mapped pinned host memory with cache-volatile write-through and polling.
   - Merged draft token argmax, cross-GPU reduction, and draft vocabulary remapping into a single ~10 µs GPU kernel.
2. **Small-T GQA Routing for TP2**:
   - Fixed GQA attention routing in [`src/ops/wrapper/gqa_attention.cpp`](file:///tmp/ninfer/src/ops/wrapper/gqa_attention.cpp) to properly recognize TP2 partition sizes (`q_heads == 8`).
3. **Performance Milestone**:
   - Achieved **`90.81 t/s`** aggregate throughput (515 tokens in 5.67 s), shattering the 90 t/s barrier!
   - Acceptance rate reached **`92.6%`** (378 of 408 proposed drafts accepted).
   - Effective yield: **`3.79 tokens / round`**.

---

## 2. Benchmark Summary (512 Tokens, 2× RTX 5060 Ti)

| Metric | Objective 3 Baseline | Objectives 1 & 2 (`OneShotArgmax`) | Delta |
| :--- | :---: | :---: | :---: |
| **Throughput (t/s)** | **`88.73 t/s`** | **`90.81 t/s`** | **+2.08 t/s (+2.3%)** |
| **MTP Acceptance Rate** | $89.7\%$ ($374/417$) | **`92.6%` ($378/408$)** | **+2.9%** |
| **Tokens / Round** | $3.70\text{ tokens/round}$ | **`3.79 tokens/round`** | **+0.09 tokens/round** |
| **Total Round Latency** | $36.50\text{ ms}$ | **`36.38 ms`** | **−0.12 ms** |
| **Verify Latency ($T=4$)** | $32.88\text{ ms}$ | **`32.85 ms`** | −0.03 ms |
| **Prompt Prefill Latency** | $151.0\text{ ms}$ | **`150.2 ms` (33.3 t/s)** | −0.8 ms |

---

## 3. Detailed Draft Window Analysis ($k=3$ vs $k=4$)

We investigated increasing $k$ from 3 to 4:
- **$k=3$ ($T=4$ verify tokens)**:
  - Fits within the single-pass NVFP4 Tensor Core MMA tile (`kNvfp4LastSmallT = 4`).
  - Target Verify latency: **`32.85 ms`**.
  - Throughput: **`90.81 t/s`**.
- **$k=4$ ($T=5$ verify tokens)**:
  - Exceeds the 4-token Tensor Core tile width, triggering chunked execution ($T=4 + T=1$) across all 128 GEMMs in the 64 layers.
  - Target Verify latency rises to **`54.73 ms`** (+66% overhead).
  - Throughput drops to **`47.75 t/s`**.
- **Conclusion**: Linear $k=3$ ($T=4$) is the optimal hardware execution point on the RTX 5060 Ti architecture for single-tile verify without triggering multi-pass GEMM chunking.

---

## 4. Git Changes

- **Repository**: [`dual_5060_ti_ninfer`](https://github.com/chrisconcepcion/dual_5060_ti_ninfer.git) (`mtp-perf` branch)
- **Commit**: `f3614a83` (`feat(perf): OneShotArgmax device-level proposal exchange and small-T GQA routing for 90.81 t/s`)
