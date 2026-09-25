# MTP Performance Hitlist — Results, Gap Analysis & Path to 70+ t/s
**Target model**: `qwen3.8-27b` (`/home/intel/models/qwen3_8_27b.ninfer`, 16,731 MB) · **Hardware**: 2× RTX 5060 Ti (TP2) · **Date**: 2026-08-20 · **Branch**: `mtp-perf` (`674d9a7b`, pushed to `chrisconcepcion/dual_5060_ti_ninfer`)

---

## TL;DR — honest status
- **Achieved: 53.77 t/s** (MTP k=3, 514 tok, 64.9% acceptance). **Target: 70+ t/s. DoD NOT met.**
- This is a **real 2.8× over plain** (plain decode 18.94 t/s → MTP 53.77). The MTP mechanism works and is correct/deterministic.
- **The gap to 70 is NOT verify, and NOT the hardware.** Reference: llama.cpp does **70 t/s at ~4k actual context** on this box — that is *harder* than our ~200-token decode test. So the hardware ceiling is proven. NInfer's verify is actually **9.6 ms/token (batched T=4)** — *faster* than llama.cpp's ~14.3 ms/token single decode.
- **The pure waste is the MTP head:** it re-runs the **248,320-vocab lm_head GEMV ~4×/round** (propose d0 + AR d1–d3) = **~11.25 ms/round**. That is the single lever to 70+.
- **Verify-bound ceiling: ~77 t/s** (38.26 ms verify ÷ 2.95 committed/round, MTP head made free). So **70+ is reachable** by making the MTP head cheap. See the plan below.

## What was done (all verified real + reproducible)
1. **A1 — plain decode hang FIXED.** Root cause: `WeightsProfile::Qwen38GroupwiseInt` was unhandled, a rank-barrier deadlock, and the plain path needed `cursor.F = plen`. Re-run confirmed: **19.08 t/s @40 tok, no hang, coherent.**
2. **A2 — determinism + token-identity verified** (greedy MTP == greedy plain, token-for-token).
3. **B1 — per-phase CUDA-event timing** (8 phases, rank 0). See breakdown below.
4. **B2 — steady-state on 3.8 @512 tok** = 53.77 t/s.
5. **C4 — collapsed D2H sync** → accept+sync bucket now **0.03 ms/round** (was ~1–2 ms). Real, small.
6. **C3 — dynamic k** refactor; tested k=4 (below) → **k=3 confirmed optimal** (T=4 is the specialized-kernel boundary).
7. **C2 — TP-split the MTP head** (sharded `mtp/…/query_key_gate_value`, `output`, `mlp/gate_up`, `mlp/down`; added `Variant::mtp_attention_projection_tp` / `mtp_post_mixer_tp`). VRAM −188 MB/rank; align 1.20→0.81 ms, AR 8.83→8.00 ms. **But only ~1.3 ms/round total** — the MTP head was never the dominant cost (see breakdown).

## Benchmark (measured)
| Config | Tok | t/s | Draft accept | Mean a | Round |
|---|---|---|---|---|---|
| Plain (`--mtp 0`, host-sync-serialized) | 40 | 18.94 | – | – | 52.8 ms/tok |
| MTP k=3 (before C2/C4) | 514 | 52.44 | 64.9% | 1.95 | 52.06 ms |
| **MTP k=3 (after C2+C4)** | **514** | **53.77** | **64.9%** | **1.95** | **50.76 ms** |
| MTP k=4 | 64 | 29.87 | 67.6% | 2.71 | 82.85 ms |

## B1 phase breakdown (mean/round, total 50.76 ms)
```
1. Target Verify (T=4)      38.26 ms  75.4%   ← floor (9.6 ms/token, batched)
2. Accept & D2H Sync          0.03 ms   0.1%   (C4 done)
3. GDN State Rebase           0.40 ms   0.8%
4. Prepare Next Round         0.01 ms   0.0%
5. MTP Alignment (T=4)        0.81 ms   1.6%   (C2)
6. Select Accepted Hidden     0.00 ms   0.0%
7. MTP Propose (d0) lm_head   3.25 ms   6.4%   ← lm_head GEMV
8. AR Draft Chain (d1..d3)    8.00 ms  15.8%   ← 3× (MTP fwd + lm_head)
                                        ─────────
                                  50.76 ms
```
**Round commits ~2.95 tok → 50.76/2.95 = 17.2 ms/token → 58 t/s ceiling; measured 53.77** (the ~4 ms gap is launch/D2H not in the phase sum).

## ⚠️ Corrections to the earlier report
- **"2.54 GB at 750 GB/s" is wrong twice.** The 5060 Ti is ~**448 GB/s**, and the lm_head (`text/output_head`, artifact object [776]) is **W8G32_F16S (8-bit), 1.35 GB** — NOT 2.54 GB BF16 (confirmed from the artifact dump). So the 3.25 ms GEMV is **bandwidth-bound at ~415 GB/s on an already-8-bit matrix** — it is NOT cheaply quantizable for a free win. (The MTP head has no lm_head of its own; it reuses `text/output_head`.)
- **C2 was not the lever doc 10 predicted.** It saved ~1.3 ms because the whole MTP head is only ~12 ms, not the dominant cost. The dominant, *removable* cost is the **redundant output_head GEMVs** (items 7+8 = 11.25 ms).

## 🎯 PATH TO 70+ t/s — do these, in order (next agent)
Round math: to hit 70 t/s at 2.95 tok/round we need round ≤ **42 ms**. We're at 50.76. **Cut ~9 ms from the MTP head (items 7+8) → ~42 ms → 70+ t/s.** Verify (38.26 ms) stays as the floor.

**P1 — Cut the MTP-head `output_head` cost (the #1 lever, ~6–9 ms/round).** The MTP head proposes via the shared `text/output_head` (W8G32, 1.35 GB), called ~4×/round (propose d0 + AR d1–d3) = 11.25 ms, each GEMV bandwidth-bound. Levers, best-first:
   1. **Batch the AR `output_head` GEMVs** (biggest, *no quality risk*): d0 + the 3 AR drafts run 4 sequential 1.35-GB GEMVs; if the draft hidden states can be gathered and pushed through **one batched (4-row) `output_head` GEMM**, the weight is read once → ~3.25 ms instead of 11.25 ms. **Verify feasibility first:** the AR chain is sequential (h_{i+1}=MTP_fwd(h_i, d_i), d_i=argmax(output_head(h_i))), so the argmax may sit on the critical path — determine whether we can compute all h_i then batch, or whether d_i must be known before h_{i+1}. If infeasible, this is out.
   2. **CUDA-graph the MTP-head sub-round** (propose + AR + rebase are fixed-shape; only buffer contents change). `DecodeGraphExecutable::update` supports it. Kills the ~4 ms launch/D2H tail → 70→~75.
   3. **Q4 the `output_head`** (1.35 GB W8G32 → ~0.7 GB Q4G64 — re-convert that one artifact object only). Each GEMV ~2× faster → saves ~6 ms/round → crosses 70. **⚠️ Quality check mandatory** (top-1 accuracy + A2 token-identity must still pass).

**P2 — (Beyond 77 t/s) Speed up the verify T=4 kernel.** 38.26 ms is the floor; if P1 lands ~75 and we still want more, optimize the 3.8 groupwise T=4 verify GEMV kernel toward llama.cpp's per-token speed. This is kernel engineering, separate from MTP.

**Acceptance criteria (updated DoD):**
- MTP k=3 steady (`--tokens 512`) **≥ 70 t/s**, acceptance ≥ 55%.
- **A2 still passes** (token-identical to plain) — mandatory after any lm_head precision change.
- Plain decode still works (A1 regression stays fixed).
- Build clean; commit to `mtp-perf`; push to `chrisconcepcion/dual_5060_ti_ninfer`.

## Baseline clarifications (do not chase ghosts)
- **Plain 18.94 t/s vs the earlier 32.5**: not a regression. Plain decode is **host-sync-serialized** (a D2H read per token); the old 32.5 was a different "warm" measurement. MTP exists precisely to amortize that.
- **Our test is at ~200 actual context** (5-token prompt + ~200 decode), *lower* than llama.cpp's 70 t/s @~4k context — so the 70 t/s hardware reference is at an *easier-or-equal* KV regime. The gap is compute (MTP head), not KV bandwidth.
- **k=4 is correctly rejected**: T>4 drops to the unspecialized 8-column kernel (64.6 ms verify) — k=3 (T=4) is the sweet spot.

## Commands
```
cd /tmp/ninfer && git checkout mtp-perf && cd build && make -j24 ninfer_tp2_decode_test
ART=/home/intel/models/qwen3_8_27b.ninfer
cd build
# steady-state (the DoD number)
timeout 300 stdbuf -o0 ./tests/ninfer_tp2_decode_test --artifact $ART --tokens 512 --ctx 4096 --mtp 3 --prompt "The capital of France is"
# plain baseline (A1 regression check)
timeout 120 stdbuf -o0 ./tests/ninfer_tp2_decode_test --artifact $ART --tokens 40 --ctx 4096 --mtp 0 --prompt "The capital of France is"
```

## Git
- Branch `mtp-perf`, commit `674d9a7b`, pushed to `chrisconcepcion/dual_5060_ti_ninfer`.
- Modified: `tests/multi_gpu/tp2_decode.cpp`, `src/targets/qwen3_6/impl/runtime/text_context_impl.h`, `src/targets/qwen3_6_27b/impl/load/{tp_load.cpp,bindings.cpp}`, `src/targets/qwen3_6_27b/impl/{variant.h,variant_kernels.cpp}`, `src/targets/qwen3_6_35b_a3b/impl/{variant.h,variant.cpp}`.
