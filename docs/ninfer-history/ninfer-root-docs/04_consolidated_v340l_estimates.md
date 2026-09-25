# 04 · Consolidated 2× V340L performance estimates (Qwen3.6-27B)

**Status:** ARCHIVE

Supersedes the "tuned" column of `02_v340l_projections.md`. This is the single reference for what **2× V340L** should deliver under three engine tiers: stock, ninfer-only, and combined (ninfer + v100-skinny).

## 1. Hardware & topology baseline (fix the ambiguity here)

| Unit | Count | Per-unit | Total |
|---|---|---|---|
| V340L **card** | **2** | 2 gfx900 dies, 16 GB | 32 GB |
| gfx900 **die** (the device) | **4** | 8 GB HBM2 @ **483.8 GB/s**, 21.5 TFLOPS fp16 (2:1) | **1935 GB/s**, **86 TFLOPS** fp16 |

- "4-die" in every doc = **2× V340L**. 1 die = 1 CUDA/HIP device = 8 GB.
- Model: Qwen3.6-27B Q4_1 GGUF = **17.54 GB** → fits on 2 cards (32 GB), **not** on 1 (16 GB).
- **Only full-model topology on 2× V340L = 4-die (2-card) tensor split** — exactly the stock `-sm tensor` run.
- The 2-die single-card run is **not** a 27B serving config (17.54 > 16 GB); it is only a *sync-speed probe* with a smaller model, to measure how much cross-card PCIe cost is removable.
- No cross-die fabric; all inter-die traffic is PCIe 3.0 (x8/die, x4 on x99-class boards).

Stock baseline (measured, MTP on, ROCm, 4-die tensor): **tg 19.15 t/s**, **pp512 144.66 / pp1024 140.14 / pp2048 138.09 t/s**. Implied MTP-off ≈ 15–17 t/s.

## 2. The three engine tiers

| Workload | Stock (meas.) | **ninfer-only** | **Combined** (ninfer + v100-skinny) | Combined vs stock |
|---|---|---|---|---|
| tg plain (M=1) | ~15–17 | ~33–48 (c. 40) | **~50–62 (c. 57)** | ~3.5× |
| tg MTP k=4 math | 19.15 | ~125–140 | **~150–190 (c. 170)** | ~9× |
| tg MTP k=4 structured | 19.15 | ~140–155 | **~170–210 (c. 190)** | ~10× |
| tg MTP k=4 code | 19.15 | ~95–115 | **~120–150 (c. 135)** | ~7× |
| tg MTP k=4 prose | 19.15 | ~60–80 | **~80–105 (c. 90)** | ~4.7× |
| pp512 | 144.66 | ~350–650 | **~450–700 (c. 570)** | ~4× |
| pp2048 | 138.09 | ~300–550 | **~400–600 (c. 480)** | ~3.5× |

**What each tier adds over the one before:**
- **ninfer-only** = graph-captured decode, per-shape specialization, offline prepack, MTP (`lm-head-draft` + small-T), paged KV + INT8 group-64, chunked prefill — with a **standard** allreduce and **separate** argmax, and a *conservative* 55–65% GEMV.
- **Combined** = ninfer-only **plus** the v100-skinny kernel/system wins: GEMV tuned to 65–75%, **custom one-shot allreduce + AR batching (every 2–4 layers)**, **fused argmax lm_head**, NUMA/clock pinning, GDN fast-metadata.

## 3. Combined per-token cost model (4-die tensor split, 4.385 GB/die)

| Component | Cost | Source |
|---|---|---|
| GEMV (weight read) | 12–14 ms | 4.385 GB/die @ 65–75% of 483.8 GB/s (v100-skinny tuning × ninfer prepack) |
| Allreduce (custom one-shot, batched /2–4) | 1–1.5 ms | v100-skinny AR, PCIe latency floor |
| KV + GDN state | 0.5–1.5 ms | ninfer INT8 KV + v100-skinny fast metadata |
| GDN recurrent | 0.5–1 ms | tuned |
| Launch (graph) | 0.5–1 ms | ninfer graph |
| Argmax (fused) | ~0.05 ms | v100-skinny |
| **Plain decode** | **~16–19 ms** | **~52–62 t/s, central ~57** |

MTP k=4 round ≈ **24 ms** (verify@M=5 ~21 + 4 drafts ~2.3 + misc ~0.5); tok/round = 1 + 4×acceptance.

## 4. Why combined > sum of the parts

The two sets are **synergistic, not additive**:
1. **Prepack (ninfer) is a prerequisite for the tuned GEMV (v100-skinny)** — the 72–78% number only exists because of the offline-interleaved layout. They multiply.
2. **Graph capture (ninfer) wraps the custom AR (v100-skinny)** — the collective launches are graph-captured, so zero launch overhead on ARs.
3. **INT8 KV (ninfer) + fast GDN metadata (v100-skinny)** both cut the same non-GEMV line.
4. **AR batching** (accumulate 2–4 layers' partials, reduce once) cuts the AR count ~128 → ~32–64. This is the single biggest combined-only win and exists in neither stock nor the ninfer-only case.
5. **MTP:** ninfer's implementation + v100-skinny's acceptance knowledge + byte-diff validation.

That's why plain decode jumps ~40 (ninfer-only) → ~57 (combined): you get *both* the fast GEMV *and* the fast AR.

## 5. Key uncertainties (verify cheapest-first)

1. **★ GEMV hits 65–75% on gfx900** (v100-skinny measured 72–78% on V100; 64-wide wavefronts + 64 KB LDS may shave a few pts). Single-die microbench settles it; if 55–65%, everything drops ~15%.
2. **★ AR batching is numerically clean** (fp16 accumulation over 2–4 layers). Byte-diff vs unbatched; if it drifts, fall back to per-layer AR (−1–2 ms/token).
3. **PCIe per-die link width** (x8 vs x4). x4 halves AR bandwidth headroom — check `rocm-smi`/`lspci` first.
4. **Prefill GEMM tiling (30–45% of fp16 peak)** — the main pp unknown; a better large-M tile pushes pp512 toward 700+.

## 6. Context & ceiling

- 4× V100 (v100-skinny, byte-verified) = 86.6–91.2 t/s plain. 2× V340L at ~57 plain / ~190 MTP-structured ≈ **63% of that** on a fraction of the cost — a strong $/t/s story, gated on uncertainties #1–#2.
- Decode roofline (4-die, 100% BW) = 110 t/s; the combined ~57 is ~52% of roofline, consistent with v100-skinny's measured ~37–54% whole-token efficiency.
