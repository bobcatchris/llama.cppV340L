# Objective 2: Target Verify Analysis & Split-KV Attention Scaling

## 1. Executive Summary

We investigated and profiled **Objective 2 (Split-KV Attention & Target Verify Optimization)** from [`13_future_objectives_mtp_70plus.md`](file:///home/intel/comfy_templates/v340l_optimization/13_future_objectives_mtp_70plus.md).

Through full `nsys` kernel profiling and micro-benchmarking on the 2× RTX 5060 Ti system, we mapped the precise breakdown of the **36.56 ms Target Verify** phase:

```text
=== Target Verify Phase Cost Breakdown (36.56 ms total) ===
  1. Projection & Linear GEMVs (Q4/Q5/W8):     23.5 ms  (64.3%)
  2. NCCL AllReduces (128 calls per round):      7.5 ms  (20.5%)
  3. Host Launch Overhead & Synchronizations:    3.8 ms  (10.4%)
  4. GDN Recurrent + Conv1d Steps:               1.2 ms   (3.3%)
  5. GQA Paged Attention:                        0.2 ms   (0.5%)
```

---

## 2. Kernel-Level Profiling Highlights (`nsys`)

| Top Kernel | Calls / 15 Tok | Total Time | Mean / Call | Role |
| :--- | :---: | :---: | :---: | :--- |
| `q5_rowsplit_gemm_simt_kernel` | 4,800 | 257.31 ms | 53.6 µs | GDN & Full Attention Input Projections |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 2,622 | 152.57 ms | 58.2 µs | Inter-GPU AllReduce (2 per layer × 64 layers) |
| `q4_rowsplit_gemm_simt_kernel` | 2,560 | 248.83 ms | 97.2 µs | MLP `gate_up` and `down` GEMVs |
| `w8_small_t_mma_kernel` | 10 | 16.32 ms | 1.63 ms | Full LM Head projection |
| `recurrent_snapshot_kernel<__half>` | 480 | 3.90 ms | 8.1 µs | GDN FP16 Recurrence ($T=4$) |
| `gqa_attention_small_t_tc_partial_bf16` | 160 | 1.33 ms | 8.3 µs | GQA Attention Partial |
| `gqa_attention_small_t_reduce_output` | 322 | 1.20 ms | 3.7 µs | GQA Attention Split Reducer |

---

## 3. Key Findings & Adjustments

1. **GQA Attention Efficiency**:
   - `gqa_attention_small_t_tc_partial_bf16` is already extremely fast ($8.3\text{ µs/layer} \times 16\text{ layers} = 0.13\text{ ms}$).
   - Updated `Gqa27TpGeometry` in [`gqa_attention_geometry.cuh`](file:///tmp/ninfer/src/ops/kernel/gqa_attention_geometry.cuh#L25-L29) from `DecodeSplitScale = 1` to `DecodeSplitScale = 2`, ensuring sufficient grid blocks to saturate all 36 SMs of the RTX 5060 Ti even at small contexts.
2. **Small-$T$ Dispatch for TP GEMV**:
   - In [`tp_kernel.cu`](file:///tmp/ninfer/src/core/multi_gpu/tp_kernel.cu#L38), updated the small-$T$ dispatch threshold from `t <= 4` to `t <= 7`.
   - This fixed a large regression in $k=4$ ($T=5$) where `r8_c8` was running un-padded 8-wide tiles, reducing $k=4$ round latency from **`68.19 ms`** down to **`63.88 ms`**.
3. **The Real Levers for Breaking 80+ t/s**:
   - **Launch / Host Overhead (~4 ms)**: Objective 3 (CUDA Graph Capture) eliminates CPU launch latency across the ~500 kernels in Target Verify.
   - **AllReduce Latency (~7.5 ms)**: Tuning NCCL transport or fusing AllReduce into GEMV epilogues.
   - **GEMV Quantization**: Objective 4 (INT4 Draft Head) and deeper quantization for heavy projection layers.

---

## 4. Verification & Git Commit

- **Git Commit**: `cdf0f758` (`perf(attn): scale Gqa27TpGeometry decode split capacity and support small-T TP GEMV up to T<=7`) pushed to `origin/mtp-perf`.
- **Determinism**: 100% bit-identical token output verified across plain decode and MTP $k=3$.

---

## 5. Independent Review (chris, 2026-08-20)

**Verdict: profiling accepted; Objective 2 closed as a negative result.**

Verified against the verification battery (`tools/verify_battery.sh`, run 20260820_211740):

1. Commit `cdf0f758` confirmed pushed (`git ls-remote origin mtp-perf` = `cdf0f758`).
2. Breakdown is internally consistent: 23.5+7.5+3.8+1.2+0.2 = 36.2 ms ≈ measured verify 36.55 ms.
   B1 phase sum matching the *simple sum* of parts ⇒ the AR time is **exposed (serialized on the
   stream)**, not hidden under GEMV.
3. AR math checks: 128 calls/round × 58.2 µs = 7.4 ms ✓; nsys window total implies ~20.5 rounds.
4. The 0.5% attention finding is confirmed by a null result: `DecodeSplitScale 1→2` produced no
   measurable verify change (36.74 → 36.55, within noise). T-batched TP GQA at short context is
   already SM-saturated via head-block parallelism.
5. Determinism claim independently re-verified: battery PASS (bit-identical, plain/MTP identity).

**Corrections / open items:**

- **Table header**: "Calls / 15 Tok" does not reconcile with the totals (AR: 152.57 ms ÷ 7.5 ms/round
  ⇒ ~20.5 rounds; q5/q4 totals agree). Re-label the column as rounds.
- **AR lever is under-specified.** P1's "direct non-blocking allreduce" (f184738f) is still
  `ncclAllReduce` — it only removed the host worker-thread barrier. 58 µs/call is the GPU-side RING_LL
  latency for 10 KB over PHB. The repo already has validated 2-rank P2P send/recv (bench_ar.cpp,
  6–9 µs/call). **New Objective 2a (doc 13): microbench a one-shot 2-rank AR at 5120 bf16 elems;
  if ≤ ~30 µs, route the MTP round's 128 ARs through it. Expected −3–6 ms/round ⇒ ~95–105 t/s,
  no quality risk, bigger and easier than CUDA graph.**
- **T=5 anomaly (open):** verify T=5 = 58.66 ms vs T=4 = 36.55 ms (+22.1 ms for one query row).
  GEMV reads weights once; attention is µs-scale. Neither explains 1.6×. Profile the T=5 GEMV
  path (SIMT vs MMA selection, padding) before revisiting k=4 (Objective 7).
- **q4 mean 97.2 µs reconciles** as a mix of gate_up (17408 local rows ≈ 50 MB ≈ 118 µs at 427 GB/s)
  and down (8704 rows ≈ 25 MB ≈ 59 µs). GEMV runs at ~99% of measured peak bandwidth — deeper
  quantization (Objective 4) is the only GEMV-side lever, quality-gated.

**Revised lever order** (round 40.20 ms, 79.20 t/s):

| # | Lever | Saving | → t/s | Risk |
|---|---|---|---|---|
| 1 | 2a: P2P one-shot AR (probe first) | 3–6 ms | ~95–105 | none |
| 2 | Obj 3: CUDA-graph the round | ~4 ms | ~110–125 | low |
| 3 | Obj 4: INT4 verify GEMV | ~11 ms | ~140–160 (BW ceiling) | quality |
| 4 | Obj 7: k=4 after T=5 probe | — | — | — |
