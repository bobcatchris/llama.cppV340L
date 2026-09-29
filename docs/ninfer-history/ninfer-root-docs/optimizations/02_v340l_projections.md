# Projections: 27B-class decode & prefill on 2× V340L (4× gfx900)

Date: 2026-08-19. Companion to `01_v100_skinny_findings_for_v340l.md`.
Model: Qwen3.6-27B (27.32B params, hidden 5120, vocab 248,320, 64 layers = 16 full-attention + 48 GDN linear-attention).
Weights: Q4_1 GGUF = 16.33 GiB = **17.54 GB** (measured artifact size).
Hardware: 2× V340L = 4× gfx900 dies, 8 GB HBM2 each @ **483.8 GB/s** (2048-bit, 1890 Mbps eff), fp16 2:1 = 21.5 TFLOPS/die (86 TFLOPS aggregate), 64 KB LDS/CU, 4 MB L2/die, PCIe 3.0 x8 per die (x4 on x99-class boards), **no cross-die fabric**.

Stock baseline (published, MTP on, ROCm, tensor split, 4 dies): **tg 19.15 t/s**, **pp512 144.66 / pp1024 140.14 / pp2048 138.09 t/s**.

All "tuned" numbers below assume the v100-skinny recipe: custom dequant GEMV/GEMM, graph-captured decode round, custom one-shot allreduce, MTP k=3–4 greedy drafter, fused argmax head. Confidence: central estimate ± range; re-verify the two starred assumptions first.

---

## 1. Decode rooflines

| Path | Math | Ceiling |
|---|---|---|
| Tensor split, 4 dies (parallel weight read) | 17.54/4 = 4.385 GB/die @ 483.8 GB/s = 9.05 ms | **~110 t/s** |
| Layer split, 4 dies (serial) | 17.54 GB @ 483.8 GB/s = 36.2 ms | **~27.6 t/s** |
| 2-way TP within one card (2 dies, x8 each, on-card bridge) | 8.77 GB/die @ 483.8 = 18.1 ms + small AR | **~50–55 t/s ideal** |

The stock 19.15 t/s = **17% of the tensor-split ceiling** → ~43 ms/token of the 52.2 ms measured is sync/launch/orchestration overhead, not weight traffic. That overhead is the prize.

Calibration anchor: v100-skinny's *measured* in-server whole-token efficiency was ~37% of aggregate bandwidth (15.2 GB / 11.5 ms / 3.6 TB/s) with GEMM alone at 66–72%.

## 2. Decode projections (tg)

### Plain decode (MTP off), 4-die tensor split, tuned

| Aggregate efficiency | ms/token | t/s |
|---|---|---|
| 37% (v100-skinny measured in-server) | 24.5 | 41 |
| 50% | 18.4 | 54 |
| 60% (optimistic) | 15.3 | 65 |

**Central: ~40–60 t/s, planning number 50.**

### With MTP k=3–4 (greedy drafter, `--lm-head-draft`-style shared head)

Round cost ≈ 1.25× M1 (verify at M=4–5 is still bandwidth-bound on 4-die fp16 — see §5) + k drafter passes (~1.6 ms; MTP module ≈ 0.6 GB Q4, TP4) + AR (~2 ms) + misc (~1 ms) ≈ **~30 ms/round** at planning efficiency.

Acceptance (τ) curves from v100-skinny measured data (model+quant property, hardware-invariant — verified there across kernel swaps):

| Domain | τ @ k=4 | tokens/round | **t/s (tuned)** | vs stock 19.15 |
|---|---|---|---|---|
| Math (thinking) | ~3.1 | 4.1 | **110–140** | ~6–7× |
| Extraction | ~3.9 | 4.9 | **130–165** | ~7–8× |
| Code (no-think) | ~2.0 | 3.0 | **85–110** | ~4.5–5.5× |
| Prose (chat, no-think) | ~1.1 | 2.1 | **50–60** | ~2.5–3× |

Caveats: (a) τ was measured on NVFP4; Q4_1 numerics may shift acceptance ±3–5 pts (v100-skinny documented quant-dependent acceptance, AWQ beating NVFP4 on code). (b) Prose is the weak case *even on V100* — at ~30% acceptance, k=4 barely beats plain; don't sell chat speed, sell math/structured.

### 2-way within-card TP (the untested high-upside path)

Ideal: 18.1 ms weight read + ~1–2 ms AR (128 × ~10 µs one-shot over the on-card bridge) ≈ **45–55 t/s ideal**; at 75% per-die efficiency + realistic AR: **~35–45 t/s**. This would beat 4-die cross-card tensor split *and* layer split. **Experiment to run first (stock software):** `HIP_VISIBLE_DEVICES=0,1 llama-bench -m … -sm tensor -n 128`. If stock 2-way gives ≥28 t/s, within-card sync is cheap and the 4-die targets hold; if it also sits at ~19, per-die GEMV efficiency (or within-card sync) is the wall and everything shifts down ~30%.

### Batching (concurrent streams), 4-die tensor split

Weight bytes read once per step regardless of batch; M=4 GEMV ≈ 1.2–1.3× M=1 cost (M-wall §5): **~3–3.5× aggregate per 4 streams** → ~150–200 t/s aggregate at 4 streams (32 GB holds 27B Q4 + 4× ~1 GB KV). Diminishing after M≈6–8.

## 3. Prefill (prompt processing) projections

Stock pp512 = 144.66 t/s implies 54.6 GFLOP/token ÷ 144.66 = **378 GFLOPS ≈ 0.4% of the 86 TFLOPS fp16 peak**. Prefill at M=512 is ~1800 FLOP/byte per die — deeply compute-bound — so the stock number is sync/kernel-bound, not compute-bound. Huge headroom.

Tuned dequant GEMM (Marlin-class, A16 W4) at 25–40% of fp16 peak (discounted: 48/64 layers are GDN chunked-scan, not pure GEMM; no bf16/TC):

| | Stock (measured) | Tuned estimate | Gain |
|---|---|---|---|
| pp512 | 144.66 | **450–650** (central ~550) | ~3.5–4.5× |
| pp2048 | 138.09 | **400–550** | ~3–4× |
| pp8k+ | — | flat to ~8k, then −5%/×2 ctx (16 attn layers O(n²); GDN linear) | — |

Reference point: ninfer (RTX 5090, native FP4 W4A4 tensor cores) does 11,191 t/s at 7,680 ctx on the same model — the 5090's native-FP4 prefill is ~20× the gfx900 estimate. That gap is hardware (FP4 TC + 1.8 TB/s), not software; do not chase it.

## 4. Summary table

| Workload | Stock (MTP on, measured) | Tuned estimate | Improvement |
|---|---|---|---|
| tg plain | ~15–17 (implied) | **40–60** | ~2.5–3× |
| tg math/structured, k=3–4 | 19.15 | **110–165** | **~6–8×** |
| tg code, k=3–4 | 19.15 | **85–110** | ~5× |
| tg prose, k=3–4 | 19.15 | **50–60** | ~2.5–3× |
| tg 4-stream batch | — | **150–200 agg** | — |
| pp512 | 144.66 | **450–650** | **~3.5–4.5×** |

Context check: 4× V100 (v100-skinny, byte-verified) does 86.6–91.2 t/s plain on this model. Two $50 V340Ls tuned ≈ 25% of that — consistent with the aggregate bandwidth ratio (1935 vs 3600 GB/s) plus NVLink-vs-PCIe. As a $100 machine it's a strong $/t/s story, not a V100 replacement.

## 5. Key assumptions & their failure modes

1. **★ Sync overhead is removable** (graph capture + one-shot AR cuts ~25–40 ms of the 52 ms stock round). Failure mode: if PCIe x8 latency makes per-layer lockstep intrinsically ~20 ms, tuned tg drops to ~25–30 and the 2-way within-card path becomes the mainline. Test: §2 experiment.
2. **★ Per-die GEMV reaches 70–80% of 483.8 GB/s** (v100-skinny hit 72–78% on V100). Failure mode: GCN5 64-wide wavefronts + 64 KB LDS cap the achievable to ~55–65% → everything ×0.75. Test: single-die microbench of the ported GEMV before any multi-die work.
3. **M-wall at M≈6–8** (fp16 ALU pool 86 TFLOPS ÷ 483.8 GB/s ÷ 3.56 FLOP/B-per-M, dequant-discounted). Failure mode: wall at M≈4 → batching and k=4 verify both degrade; drop to k=2–3 and 2–3 streams.
4. **τ curves transfer from NVFP4 to Q4_1** (±3–5 pts). Failure mode: Q4_1 acceptance materially lower on code → code column drops ~20%.
5. **GDN chunked-scan prefill runs at ~60–80% of GEMM efficiency.** Failure mode: if GDN prefill is half-efficiency, pp central drops to ~400–450.
6. **Card health:** V340L units vary (power caps, limp-mode power delivery reported by owners). Re-measure the memcpy ceiling per card; a 30% power-capped die drags the whole tensor-split round (slowest-die gating, same as v100-skinny's per-rank asymmetry finding).

## 6. Suggested measurement order (cheapest signal first)

1. `rocm-smi` memcpy ceiling per die (10 min) → sets the roofline.
2. Stock 2-way within-card TP (10 min) → disambiguates assumption 1.
3. Stock layer split + per-die util during tensor run (10 min) → calibrates stock GEMV efficiency.
4. Single-die GEMV microbench of the ported kernel (1–2 days) → assumption 2.
5. 4-die tensor split with graph-captured round + one-shot AR (1–2 weeks) → assumptions 1+3.
6. MTP k=2–4 with byte-diff validation (1 week) → assumption 4.
7. Prefill GEMM (A16 W4, large M) → assumption 5.
