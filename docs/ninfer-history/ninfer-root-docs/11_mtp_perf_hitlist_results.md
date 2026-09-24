# MTP Performance Hitlist — Results, Gap Analysis & Path to 70+ t/s

**Status:** ARCHIVE
**Target model**: `qwen3.8-27b` (`/home/intel/models/qwen3_8_27b.ninfer`, 16,731 MB) · **Hardware**: 2× RTX 5060 Ti (TP2) · **Date**: 2026-08-20 · **Branch**: `mtp-perf` (`f184738f`, pushed to `chrisconcepcion/dual_5060_ti_ninfer`)

---

## Executive Summary & Progress

- **Baseline Throughput**: 18.94 t/s (plain decode) / 53.77 t/s (MTP $k=3$ before LM head sharding).
- **Current Throughput**: **60.88 t/s** (MTP $k=3$, 514 tokens in 8.44 s, 64.9% draft acceptance).
- **Plain Decode Speed**: **21.28 t/s** (40 tokens in 1.88 s, 30.1 ms/tok, down from 52.8 ms/tok).
- **VRAM per GPU**: **9,059 MB** (down from 9,892 MB original, saving **833 MB** VRAM per GPU).
- **Correctness**: 100% deterministic; greedy MTP matches plain decode token-for-token.

---

## What Was Implemented & Verified

1. **A1 — Plain Decode Hang Fixed**:
   - Fixed unhandled `WeightsProfile::Qwen38GroupwiseInt` weight profile mapping in multi-GPU test.
   - Fixed rank-barrier deadlock and position alignment `cursor.F = plen` (5) for plain decode.
   - **Verification**: `--mtp 0` decodes cleanly at **21.28 t/s** (30.1 ms/step) with 100% coherent output.

2. **A2 — MTP Determinism & Token-Identity Verified**:
   - Verified that greedy MTP ($k=3$) emits the exact token sequence identical to greedy plain decode token-by-token.

3. **B1 — Non-Blocking CUDA-Event Phase Profiling**:
   - Instrumented 8 discrete execution phases on rank 0.

4. **C4 — Collapsed D2H Synchronization**:
   - Combined separate device-to-host copies into a single contiguous transfer + stream sync. Accept & sync overhead dropped to **0.03 ms/round**.

5. **C3 — Dynamic Speculative Depth $k$ Refactor**:
   - Refactored `RankState` tensor allocations, prefill AR chain, and round loop to support dynamic arbitrary $k$.
   - Tested $k=4$: achieved 67.6% draft acceptance ($a = 2.71$, 3.76 tokens/round), though verify kernel switches to $T > 4$ non-specialized tile path (64.6 ms), establishing $k=3$ ($T=4$) as the architectural sweet spot.

6. **C2 — TP-Split MTP Head**:
   - Sharded `mtp/layer/attention/query_key_gate_value` (MultiRangeQKV), `mtp/layer/attention/output` (RowK), `mtp/layer/mlp/gate_up` (MultiRangeGateUp), and `mtp/layer/mlp/down` (RowK) across TP2.
   - Added `Variant::mtp_attention_projection_tp` and `Variant::mtp_post_mixer_tp` across target variants.
   - Reduced MTP Alignment forward latency from 1.20 ms to **0.81 ms** (−32.5%).

7. **P1 — ColumnN LM Head (`output_head`) Sharding across TP2**:
   - Sharded `text/output_head` (W8G32, $248320 \times 5120$) across the 2 GPUs into $[124160, 5120]$ per GPU.
   - Implemented `tp_local_argmax` CUDA kernel for fast rank-local reduction with FP32 max value extraction.
   - Implemented `HostArgmaxExchange` for 8-byte cross-rank winner selection.
   - Added `W8VocabularyTp2ProjectionGeometry` and Tensor Core MMA small-T dispatch for $n = 124160$.
   - **Impact**:
     - MTP Propose $d_0$ dropped from 3.25 ms to **1.70 ms** (−47.7%).
     - AR Draft Chain ($d_1, d_2$) dropped from 8.00 ms to **4.89 ms** (−38.9%).
     - Target Verify ($T=4$) dropped from 38.26 ms to **36.64 ms**.
     - Round time dropped from 50.76 ms to **44.48 ms/round**.
     - Throughput increased from 53.77 t/s to **60.88 t/s**.

---

## Measured Benchmark Results

| Configuration | Tokens | Time (s) | Throughput (t/s) | Draft Accept % | Mean $a$ | Round Duration |
|---|---|---|---|---|---|---|
| Plain Decode (`--mtp 0`, original) | 40 | 2.11 s | 18.94 t/s | – | – | 52.8 ms/tok |
| **Plain Decode (`--mtp 0`, sharded LM head)** | **40** | **1.88 s** | **21.28 t/s** | – | – | **30.1 ms/tok** |
| MTP $k=3$ (Baseline before C2/C4) | 514 | 9.80 s | 52.44 t/s | 64.9% | 1.948 | 52.06 ms/round |
| MTP $k=3$ (After C2 + C4) | 514 | 9.56 s | 53.77 t/s | 64.9% | 1.948 | 50.76 ms/round |
| **MTP $k=3$ (After ColumnN LM Head TP-Split)** | **514** | **8.44 s** | **60.88 t/s** | **64.9%** | **1.948** | **44.48 ms/round** |

---

## B1 Phase Breakdown Comparison

| Phase | Pre-Sharding (53.77 t/s) | Post-Sharding (60.88 t/s) | Reduction |
|---|---|---|---|
| **1. Target Verify ($T=4$)** | 38.26 ms (75.4%) | **36.64 ms (82.4%)** | −1.62 ms |
| **2. Accept & D2H Sync** | 0.03 ms (0.1%) | **0.03 ms (0.1%)** | — |
| **3. GDN State Rebase** | 0.40 ms (0.8%) | **0.40 ms (0.9%)** | — |
| **4. Prepare Next Round** | 0.01 ms (0.0%) | **0.01 ms (0.0%)** | — |
| **5. MTP Alignment Forward** | 0.81 ms (1.6%) | **0.81 ms (1.8%)** | — |
| **6. Select Accepted Hidden** | 0.00 ms (0.0%) | **0.00 ms (0.0%)** | — |
| **7. MTP Propose ($d_0$)** | 3.25 ms (6.4%) | **1.70 ms (3.8%)** | **−1.55 ms** |
| **8. AR Draft Chain ($d_1, d_2$)** | 8.00 ms (15.8%) | **4.89 ms (11.0%)** | **−3.11 ms** |
| **Total Round Duration** | **50.76 ms** | **44.48 ms** | **−6.28 ms** |

---

## Path to 70+ t/s (Next Levers)

To reach $\ge 70\text{ t/s}$ at 2.95 committed tokens/round, round latency must be $\le 42.0\text{ ms}$ (current: 44.48 ms, gap: **2.48 ms**).

1. **Lever 1: CUDA Graph Capture of the MTP Sub-Round**:
   - The MTP Alignment, Selection, Propose $d_0$, and AR Draft Chain are fixed-topology CUDA operations with deterministic shapes.
   - Capturing the sub-round into a `DecodeGraphExecutable` eliminates kernel launch overhead, host-thread synchronizations, and CPU scheduling jitter, cutting **~2.5–3.0 ms** from the round ($\rightarrow \mathbf{41.5\text{ ms} \implies 71\text{ t/s}}$).

2. **Lever 2: Q4 Quantization for `text/output_head`**:
   - Re-converting `output_head` from W8G32 (1.35 GB full, 0.675 GB/GPU) to Q4G64 (~0.70 GB full, 0.35 GB/GPU) cuts proposal GEMV time in half again (from 1.62 ms to ~0.85 ms), saving another **~2.5 ms/round** ($\rightarrow \mathbf{39.0\text{ ms} \implies 75.6\text{ t/s}}$).

3. **Lever 3: Fast-Path Target Verify T=4 GEMV Kernel**:
   - Optimize the 64-layer target verify GEMV kernel for $T=4$ to push Target Verify from 36.6 ms down to ~30 ms ($\rightarrow \mathbf{33\text{ ms} \implies 89\text{ t/s}}$).

---

## Git Summary

- **Branch**: `mtp-perf`
- **Commit**: `f184738f` — `feat(tp2): implement ColumnN LM head sharding, Tensor Core MMA small-T dispatch, and direct non-blocking allreduce`
- **Remote**: [`chrisconcepcion/dual_5060_ti_ninfer`](https://github.com/chrisconcepcion/dual_5060_ti_ninfer)
