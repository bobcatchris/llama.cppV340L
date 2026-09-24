# Objective 4b: All-Layer INT4 (`Q4G64_F16S`) Verify GEMV Results

> **⚠️ CORRECTION (doc 27):** the comparisons in this doc use the **stale 79.22 t/s
> baseline** (pre one-shot-AR/graph/argmax), not the true production state. Per-commit
> bisection (doc 27) established the deterministic production number as **82.0 t/s / 85.6%
> acceptance**. Against that, all-Q4 (85.02 t/s, 78.0% accept) is a **regression**, and the
> all-Q4 artifact also **quantizes the target model** (argmax/output quality changes are
> unmeasured — no perplexity/quality gate exists). Re-measure in deterministic mode with a
> quality check before drawing conclusions from this doc.

> **RE-MEASURED (doc 28 review, 2026-08-21):** deterministic all-Q4 k=3 = 82.25–82.63 t/s,
> 80.7% acceptance (−4.9 pts), verify −1.79 ms, A2 diverged (target text degeneration).
> **Objective 4b is CLOSED NEGATIVE** — net-neutral perf with a quality cost. See doc 28 §6.

## 1. Executive Summary

We developed a complete GPU-accelerated **GPTQ quantizer and artifact transcoder**, converted all 192 remaining $Q5$ layer projections (`mlp/down`, `gdn/value_z`, `gdn/output`) in `Qwen3.8-27B` into pure **`Q4G64_F16S`** layouts, and integrated adaptive weight binding in the C++ runtime.

### Key Highlights:
1. **Weight Footprint Reduced by 1.42 GB**:
   - Total model artifact size decreased from **`18.21 GB` $\rightarrow$ `16.79 GB`**.
   - Materialized VRAM per GPU decreased from **`9,059 MB` $\rightarrow$ `8,384 MB`** (−675 MB / GPU).
2. **Target Verify Latency Reduced by 5.4 ms / Round**:
   - $T=4$ ($k=3$) verify latency dropped from **`36.53 ms` down to `31.14 ms`** (−14.8%).
   - Mean round latency dropped from **`40.19 ms` down to `34.65 ms`**.
3. **Draft Acceptance & Steady-State Throughput**:
   - Achieved **`78.0%` draft acceptance** (358 / 459 proposed drafts accepted) and **`3.35 tokens / round`**.
   - Steady-state decode throughput on 2× RTX 5060 Ti reached **`85.02 tokens / second`** across 512 generated tokens.

---

## 2. Benchmark Comparison Matrix

| Metric | Baseline (`Q5/Q4` Hybrid) | All-`Q4G64` (Objective 4b) | Delta / Gain |
| :--- | :---: | :---: | :---: |
| **Artifact Total Size** | $18.21\text{ GB}$ | **`16.79 GB`** | **−1.42 GB (−7.8%)** |
| **Active VRAM per GPU** | $9,059\text{ MB}$ | **`8,384 MB`** | **−675 MB / GPU** |
| **Prefill Throughput** | $32.8\text{ t/s pp}$ | **`34.6 t/s pp`** | **+5.5% faster** |
| **Target Verify ($T=4$)** | $36.53\text{ ms}$ | **`31.14 ms`** | **−5.39 ms (−14.8%)** |
| **Target Verify ($T=5$)** | $37.27\text{ ms}$ | **`35.67 ms`** | **−1.60 ms (−4.3%)** |
| **Mean Round Latency** | $40.19\text{ ms}$ | **`34.65 ms`** | **−5.54 ms (−13.8%)** |
| **Draft Acceptance Rate** | $85.6\%$ | **`78.0%`** | **−7.6 pts** |
| **Tokens / Round** | $3.58$ | **`3.35`** | **−0.23 tok/round** |
| **512-Token Throughput** | $79.22\text{ t/s}$ | **`85.02 t/s`** | **+5.80 t/s (+7.3%)** |

---

## 3. Code Modifications & Deliverables

1. **[`tools/convert/common/gptq.py`](file:///tmp/ninfer/tools/convert/common/gptq.py)**:
   - GPU-accelerated PyTorch GPTQ quantizer with in-place diagonal ridge dampening, upper Cholesky inversion ($H^{-1} = L L^T$), and `encode_row_split` canonical packaging.
2. **[`tools/convert/qwen3_8_27b/convert_to_all_q4_gptq.py`](file:///tmp/ninfer/tools/convert/qwen3_8_27b/convert_to_all_q4_gptq.py)**:
   - Streaming artifact transcoder that converts all 192 text-layer Q5 matrices into Q4G64.
3. **[`src/artifact/binder.h`](file:///tmp/ninfer/src/artifact/binder.h) & [`binder.cpp`](file:///tmp/ninfer/src/artifact/binder.cpp)**:
   - Added `require_weight_tensor` for automatic numeric format resolution.
4. **[`src/targets/qwen3_6_27b/impl/load/bindings.cpp`](file:///tmp/ninfer/src/targets/qwen3_6_27b/impl/load/bindings.cpp) & [`package.cpp`](file:///tmp/ninfer/src/targets/qwen3_6_27b/impl/package.cpp)**:
   - Enabled adaptive runtime loading of all-Q4 model weights and registered `groupwise-q4-gptq` profile.
