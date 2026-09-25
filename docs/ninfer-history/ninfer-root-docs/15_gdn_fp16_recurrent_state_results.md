# Objective 6 Results: FP16 GDN Recurrent State

**Status:** ARCHIVE

## 1. Executive Summary

We have completed **Objective 6 (FP16 GDN Recurrent State)** from [`13_future_objectives_mtp_70plus.md`](file:///home/intel/comfy_templates/v340l_optimization/13_future_objectives_mtp_70plus.md).

By changing the Linear Attention (GDN / Delta-Net) recurrent state storage across all 48 layers from 32-bit float (`DType::FP32`) to 16-bit half precision (`DType::FP16`), we halved the recurrent state memory footprint and bandwidth consumption while keeping the recurrent register accumulation arithmetic in full FP32.

### Highlights
- **Decoder State VRAM Reduction**: Dropped from **513 MB** down to **325 MB** per rank (saving **188 MB VRAM per GPU**, **376 MB VRAM total**).
- **MTP Decode Throughput**: Reached **`80.77 t/s`** (512 tokens in 6.34 s, 89.2% acceptance rate).
- **GDN State Rebase Latency**: Dropped from **0.44 ms** down to **0.37 ms** (−16%).
- **Verification**: Output token text remains **100% bit-identical and deterministic**.

---

## 2. Benchmark Comparison (512 Tokens on 2× RTX 5060 Ti)

| Metric | Objective 1 (FP32 State) | Objective 6 (FP16 State) | Delta |
| :--- | :---: | :---: | :---: |
| **Decode Throughput** | $80.55\text{ t/s}$ | **`80.77 t/s`** | **+0.22 t/s** |
| **Total Generation Time** | $6.36\text{ s}$ | **`6.34 s`** | **−0.02 s** |
| **Mean Round Latency** | $40.54\text{ ms}$ | **`40.38 ms`** | **−0.16 ms/round** |
| **GDN State Rebase Latency** | $0.44\text{ ms}$ | **`0.37 ms`** | **−16.0%** |
| **Target Verify Latency** | $36.79\text{ ms}$ | **`36.71 ms`** | **−0.08 ms** |
| **Decoder State Memory / GPU** | $513\text{ MB}$ | **`325 MB`** | **−188 MB (−36.6%)** |
| **Draft Acceptance Rate** | $89.2\%$ ($372/417$) | **$89.2\%$ ($372/417$)** | **Bit-Identical** |

---

## 3. B1 Phase Timing Breakdown

Mean latency per round over 138 rounds ($k=3$, batch=1):

```text
=== B1 Phase Breakdown (Mean over 138 rounds, total = 40.38 ms/round) ===
  1. Target Verify (T=k+1):      36.71 ms (90.9%)
  2. Accept & D2H Sync:           0.03 ms ( 0.1%)
  3. GDN State Rebase:            0.37 ms ( 0.9%)   <-- Reduced from 0.44 ms
  4. Prepare Next Round:          0.01 ms ( 0.0%)
  5. MTP Alignment Forward:       0.81 ms ( 2.0%)
  6. Select Accepted Hidden:      0.00 ms ( 0.0%)
  7. MTP Propose (d0):            0.32 ms ( 0.8%)
  8. AR Draft Chain (d1..d_k):    2.13 ms ( 5.3%)
=================================================================================
```

---

## 4. Key Architectural Changes

1. **State Pool Specification**:
   - Added `DType recurrent_dtype = DType::FP32` to [`LinearAttentionStatePoolSpec`](file:///tmp/ninfer/src/core/linear_attention_state.h#L13-L23).
   - In [`linear_attention_state.cpp`](file:///tmp/ninfer/src/core/linear_attention_state.cpp#L90-L145), allocated recurrent state buffers according to `spec.recurrent_dtype` (`FP32`, `FP16`, or `BF16`), dynamically sizing `copy_slot` and `zero_slot` operations.
2. **Templated CUDA Recurrent Kernels**:
   - In [`recurrent.cuh`](file:///tmp/ninfer/src/ops/linear_attention/gated_delta_net/recurrent.cuh#L20-L55), added vectorized 64-bit load/store routines for `__half` using `__half2` pairs (`__floats2half2_rn`, `__low2float`, `__high2float`).
   - Templated `recurrent_bf16_direct_kernel`, `SnapshotAccess`, `recurrent_snapshot_kernel`, `RecordAccess`, `FoldAccess`, and `recurrent_fold_kernel` on `typename StateT`.
3. **Dispatch Layer**:
   - In [`recurrent.cu`](file:///tmp/ninfer/src/ops/linear_attention/gated_delta_net/recurrent.cu#L150-L245), added runtime type dispatch in `launch_recurrent`, `launch_recurrent_inout`, `launch_recurrent_snapshot`, `launch_recurrent_record`, and `launch_replay_fold`.
   - In [`gated_delta_net.cpp`](file:///tmp/ninfer/src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp#L70-L175) and [`replay.cpp`](file:///tmp/ninfer/src/ops/linear_attention/gated_delta_net/replay.cpp#L190-L215), relaxed shape/dtype validation checks to permit `FP16` state tensors.
4. **TP2 Speculative Decoder**:
   - In [`tests/multi_gpu/tp2_decode.cpp`](file:///tmp/ninfer/tests/multi_gpu/tp2_decode.cpp#L256-L268), enabled `.recurrent_dtype = DType::FP16`.

---

## 5. Verification & Git Commit

- **Git Commit**: `dc20c949` (`feat(gdn): implement FP16 GDN recurrent state to halve state memory traffic`) pushed to `origin/mtp-perf`.
- **Bit-Identical Determinism**: Output tokens under plain decode and MTP match previous runs 100%.
