# Objective 4 Results: Draft Head Quantization (Q4G64 vs W8G32)

**Status:** ARCHIVE

## 1. Executive Summary

We implemented and integrated **`Q4G64_F16S` draft head quantization** and dispatch into the 2× RTX 5060 Ti TP2 pipeline, and evaluated it against the reference `W8G32_F16S` implementation per the Objective 4 Quality Gate specifications.

---

## 2. Benchmark Comparison (512 Tokens on 2× RTX 5060 Ti)

| Metric | W8G32 Draft Head (Default) | Q4G64 Draft Head (INT4) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Draft Head VRAM per GPU** | $106\text{ MB}$ | **`53 MB`** | **−53 MB (−50%)** |
| **MTP Propose Latency ($d_0$)** | $0.32\text{ ms}$ | **`0.20 ms`** | **−37.5% faster** |
| **AR Draft Chain Latency ($d_1..d_k$)** | $2.10\text{ ms}$ | **`1.87 ms`** | **−11.0% faster** |
| **Mean Round Latency** | $38.54\text{ ms}$ | **`38.32 ms`** | **−0.22 ms/round** |
| **Draft Acceptance Rate** | **`85.6%` (370/432)** | $73.8\%$ (352/477) | **−11.8% (RTN degradation)** |
| **Mean Tokens / Round** | **`3.58 tokens/round`** | $3.22\text{ tokens/round}$ | **−0.36 tokens/round** |
| **Aggregate Decode Throughput** | **`81.64 t/s`** | $74.27\text{ t/s}$ | **−7.37 t/s** |
| **Output Token Identity** | **100% Bit-Identical Deterministic** | Diverges after step 16 | **Quality Gate Fails on RTN** |

---

## 3. Findings & Quality Gate Decision

### Performance Findings:
- Slicing and quantizing the draft head to **Q4G64** halves draft head memory bandwidth and VRAM from **106 MB to 53 MB** per GPU.
- Individual draft proposal GEMVs run **~38% faster** ($0.32\text{ ms} \rightarrow 0.20\text{ ms}$).

### Quality Gate Check (Objective 4 Spec):
> *"Gate: A2 token-identity at greedy... vs the W8G32 baseline. If quality regresses, keep int8."*

- **Uncalibrated RTN INT4** causes draft proposal error to accumulate across the 3 autoregressive draft steps ($d_1 \dots d_3$), lowering acceptance from **85.6% down to 73.8%**.
- Because fewer tokens are accepted per round, the overall system throughput drops from **81.64 t/s to 74.27 t/s**.
- **Decision**: Per the quality-gate requirement, **`W8G32` is retained as the production default (`--draft-head-quant w8`)**, keeping throughput at **`81.64 t/s`** and acceptance at **`85.6%`** with 100% bit-identical determinism.
- The `Q4G64` dispatch infrastructure is fully implemented and accessible via `--draft-head-quant q4` for when full GPTQ calibration matrices are computed.

---

## 4. Git Commit & Code Changes

- **Files Modified**:
  - [`src/ops/linear/q4/q4_dispatch.cpp`](file:///tmp/ninfer/src/ops/linear/q4/q4_dispatch.cpp): added dispatch for $N=20480$ and $N=40960$ with $K=5120$.
  - [`tests/multi_gpu/tp2_decode.cpp`](file:///tmp/ninfer/tests/multi_gpu/tp2_decode.cpp): added dynamic on-the-fly Q4 quantizer, CWD fallback resolution, and `--draft-head-quant <w8|q4>` parameter.
- **Git Commit**: `4a87951f` (`feat(drafter): implement Q4G64 draft head support with quality-gated selectable quant`) pushed to `origin/mtp-perf`.
