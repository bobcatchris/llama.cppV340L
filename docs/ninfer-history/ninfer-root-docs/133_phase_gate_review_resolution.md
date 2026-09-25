# 133: Phase Gate Code Review Resolution & Clean M1 Baseline

**Status**: Resolved & Verified  
**Supersedes**: [docs/131_phase_gate_m1_shipped.md](file:///home/intel/ninfer/worktrees/wo-phase-gate/docs/131_phase_gate_m1_shipped.md)  
**Reference Review**: [docs/132_phase_gate_review_feedback.md](file:///home/intel/ninfer/worktrees/wo-phase-gate/docs/132_phase_gate_review_feedback.md)  
**Binary Target**: `ninfer_phase_gate` (`tests/phase_gate.cu`, `tests/CMakeLists.txt`)  
**Runner & Baseline**: `tools/bench/phase_gate.sh`, `tools/bench/phase_baseline.json`, `tools/bench/phase_tolerance.json`  

---

## 1. Executive Summary & Review Disposition

Per the code review feedback in [`docs/132_phase_gate_review_feedback.md`](file:///home/intel/ninfer/worktrees/wo-phase-gate/docs/132_phase_gate_review_feedback.md):
1. **D1–D7 + D3 (Accepted)**: Confirmed clean, mathematically rigorous, independent CPU references and zero-copy/real-model execution (`qwen3_8_27b.ninfer`).
2. **D8–D10 Synthetic Stubs (Remediated)**: The synthetic placeholders (`speculative_accept_kernel`, `gdn_recurrent_state_kernel`, and LCG token stream) have been **completely removed**. In `ninfer_phase_gate`, D8, D9, and D10 are now honestly marked as `BLOCKED (requires multi-layer dual-GPU TP2 runtime)`.
3. **CMake Target Link (Fixed)**: Added `ninfer_artifact` to `tests/CMakeLists.txt:446` for `ninfer_phase_gate`. The CMake build target compiles and links cleanly.
4. **Tolerance & Error Guards (Fixed)**: Reconciled `tools/bench/phase_tolerance.json`, unified masked-key sentinels, added `CUDA_CHECK` error macros, and updated negative tests for all active phases (D2–D7).

---

## 2. Phase-by-Phase Verification Matrix

| Phase | Description | Status | sha256(first16) | Detail / Metric | Oracle / Verification Method |
|---|---|---|---|---|---|
| **D1** | K/V Quantize (Sinkhorn + Code Pack) | **PASS** | `4c9821bd85385d12` | `codes byte-diff=1`, `scales max\|D\|=76 ULP` (tol 128) | Independent CPU Sinkhorn (16 iters) + byte diff |
| **D2** | K/V Dequantize (Row+Col scale + zero point) | **PASS** | `d272364e22065333` | `K/V bf16 max\|D\|=24 ULP` (tol 64) | Independent CPU dequantize vs GPU output |
| **D3** | Draft Head Weight Decode & Top GEMV | **PASS** | `a7505509dfe6ef13` | `draft=[20026,35138,40950]`, `top max_rel=0.00e+00` | Real model weights (`/home/intel/models/qwen3_8_27b.ninfer` draft head `W8G32/Q4`) |
| **D4** | Online Attention QK Score Computation | **PASS** | `ac48c9279b56f134` | `scores max_rel=1.51e-07` (tol 1e-4) | Independent CPU single-pass tile reference |
| **D5** | Online Softmax Rescaling & Probability Accum | **PASS** | `e91211dc7ea3bb6f` | `probs max_rel=1.05e-07` (tol 1e-4) | Independent CPU online softmax reference |
| **D6** | PV Value Accumulation & Tile Update | **PASS** | `88d8add0bffbc72f` | `acc max_rel=2.53e-07` (tol 1e-4) | Independent CPU PV tile accumulation reference |
| **D7** | Attention Reduce & BF16 Normalization | **PASS** | `f2461073f8aefaad` | `out bf16 max\|D\|=0 ULP` (tol 8) | Independent CPU reduction & bitcast comparison |
| **D8** | Speculative Verification & Accept | **BLOCKED** | `----------------` | `requires multi-layer dual-GPU TP2 runtime` | Honest status (handled by TP2 decode runtime) |
| **D9** | GDN Recurrent State Update | **BLOCKED** | `----------------` | `requires multi-layer dual-GPU TP2 runtime` | Honest status (handled by TP2 decode runtime) |
| **D10** | End-to-End Stream Token Generation | **BLOCKED** | `----------------` | `requires multi-layer dual-GPU TP2 runtime` | Honest status (handled by TP2 decode runtime) |

---

## 3. Negative Mutation Test Results

The test harness was verified by injecting targeted mutations into each phase (`--mutate-d2` through `--mutate-d7`):

```bash
[phase-gate] running negative tests across phases...
[phase-gate] -> testing --mutate-d2 (expect D1 PASS, D2 FAIL, exit 2)
[phase-gate] -> D2 mutation caught: exit 2
[phase-gate] -> testing --mutate-d3 (expect D1/D2 PASS, D3 FAIL, exit 3)
[phase-gate] -> D3 mutation caught: exit 3
[phase-gate] -> testing --mutate-d4 (expect D1-D3 PASS, D4 FAIL, exit 4)
[phase-gate] -> D4 mutation caught: exit 4
[phase-gate] -> testing --mutate-d5 (expect D1-D4 PASS, D5 FAIL, exit 5)
[phase-gate] -> D5 mutation caught: exit 5
[phase-gate] -> testing --mutate-d6 (expect D1-D5 PASS, D6 FAIL, exit 6)
[phase-gate] -> D6 mutation caught: exit 6
[phase-gate] -> testing --mutate-d7 (expect D1-D6 PASS, D7 FAIL, exit 7)
[phase-gate] -> D7 mutation caught: exit 7
[phase-gate] ALL NEGATIVE TESTS PASSED (mutations localized to correct phases D2-D7)
```

---

## 4. Verification & Baseline Comparison

- **Baseline Generation**: `tools/bench/phase_gate.sh --baseline` -> writes green baseline to `tools/bench/phase_baseline.json` (exit 0).
- **Baseline Comparison**: `tools/bench/phase_gate.sh --compare` -> `MATCH baseline (clean, exit 0)`.
- **Missing Artifact Graceful Handling**: `./build/ninfer_phase_gate --no-artifact` marks D3 as `BLOCKED (model artifact missing)` while executing D1, D2, D4–D7 with exit 0.
