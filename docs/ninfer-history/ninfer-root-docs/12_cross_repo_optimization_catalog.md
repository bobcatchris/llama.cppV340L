# Cross-Repo Optimization Catalog — Qwen-27B class models, many GPUs

**Status:** ARCHIVE
**Purpose:** a reusable reference of every optimization technique we have mined from the repos we've studied, organized **by technique** (not by repo), each tagged with **source repo**, **measured win**, and **which GPU arches it helps**. We will run this same analysis again for many different GPUs — this is the master list to check against each time.

**Date:** 2026-08-20 · **Target model context:** Qwen3.8-27B (groupwise-int / NVFP4 class), 64 layers (48 GDN + 16 attn), hidden 5120, MTP-1 speculative head.

---

## 0. Repo inventory (what we studied)
| Repo | Arch / HW | Stack | Role / headline |
|---|---|---|---|
| **`v100-skinny`** (`/tmp/v100-skinny`) | sm_75 Turing, 4× V100-SXM2-16GB (NVFP4, TP4) | vLLM + custom skinny CUDA kernels | Low-bandwidth "skinny kernel" decode + chain-MTP; single-round CUDA graph; FP4 |
| **`NInfer`** (`/tmp/ninfer`) | sm_120a Blackwell, 2× RTX 5060 Ti (TP2) | Custom C++/CUDA engine | **Our base.** TP2 + MTP k=3 built. 53.77 t/s on 3.8 |
| **`qwen38-27b-rtx3090`** (`/tmp/qwen38-3090`) | sm_86 Ampere, 1× RTX 3090 24GB (250 W) | vLLM + Triton patches | Single-GPU MTP + draft-vocab + GPTQ + split-KV attn → **114–118 t/s** vs 46 stock |

**Key cross-check:** the 3090 has *more* bandwidth (936 GB/s) than our 5060 Ti (448 GB/s), yet hits 118 t/s where we sit at 53.77. The gap is **not hardware** — it is the MTP-side techniques below (draft vocab, int4 drafter, split-KV verify) that the 3090 repo has and NInfer does not yet.

---

## 1. MEMORY-BANDWIDTH / SKINNY KERNELS
The whole game on decode is reading weights once, fast, at T=1..k. "Skinny" = tall-thin (rows≫cols) GEMV/GEMM tuned to saturate HBM/GDDR bandwidth, not FLOPs.

| # | Technique | Source | Win | GPU-arch applicability |
|---|---|---|---|---|
| 1.1 | **Skinny GEMV/GEMM kernels** tuned per arch (tall-thin, split-K, vectorized loads) | v100-skinny (`skinny_kernels.cu`) | NVFP4+skinny wins 8/10 domains (+8..+34%) | **Every** bandwidth-bound decode arch. Port = retune block/warp/unroll + vector width to the arch's LSU/L2 |
| 1.2 | **Tuned SIMT r8_c4 / r8_c8 skinny kernels** (NInfer's decode GEMV) | NInfer (`tp_kernel.cu`) | 414–427 GB/s on 5060 Ti (~92% roofline) | Our arch baseline. Same family as 1.1 |
| 1.3 | **Volta-native tensor-core path** — `mma.sync.m8n8k4` for skinny W4A16 GEMM (quadpairs split on N) | v100-skinny (`qpn_race_notes.md`) | Beats WMMA on sm_70 | **Older arches without modern TC** (Volta/Turing). On Ampere+ use WGMMA/`mma.sync.m16n8k16`+ instead |
| 1.4 | **Batch-band register-fragment vs WMMA** (M=16–64): WMMA stands | v100-skinny (`twin_race_notes.md`) | register-fragment challengers lose | Decode with small M; pick the arch's native fragment |
| 1.5 | **Don't chase the GDN/delta-net decode kernel** — already ~85% of BW; the *state dtype* is the lever (see 5.1) | qwen38-3090 (gotcha #11) | every kernel variant within 3% | Universal lesson: when a kernel is BW-bound at ~85%, change the data it reads, not the kernel |

**Porting rule:** for a new GPU, microbench HBM BW + the skinny GEMV kernel first (our Window #1), then compute the roofline before touching MTP.

---

## 2. QUANTIZE THE BIG READS (lm_head, embeddings, drafter)
The largest single tensors are the vocab matrices (248,320 rows). They are read on **every** decode step (lm_head) — so their precision is the highest-leverage quant target.

| # | Technique | Source | Win | Notes / porting |
|---|---|---|---|---|
| 2.1 | **Requantize lm_head + embed_tokens to int8 group-128** (public W4A16 quants leave them bf16 2×2.5GB) | qwen38-3090 (`quant_lm_head.py`, `quant_embed.py`) | ~2.6 GB VRAM back; ~0.6% round-trip err, no quality loss | Our `output_head` is **already W8G32 1.35GB** — so int8 is a no-op for us; the lever is **int4** (2.2) or the **draft vocab** (3.1) |
| 2.2 | **GPTQ int4 lm_head + drafter** (calibrated, not naive) — "fast variant" | qwen38-3090 (`quant_mtp.py`, `drafter/gptq_lm_head.py`, `fetch_fast_variant.py`) | ~15% single-user; int4 costs ~5% acceptance at default sampling, **0 at greedy** | **This is our P1.3 done right.** GPTQ (calibrated) de-risks the quality hit. Port = GPTQ-quantize our `output_head` (+MTP module) to int4, gate on A2 token-identity |
| 2.3 | **W4A8 int8 tensor-core GEMMs** (weights int4, activations int8 per-token, int8 MMA) | qwen38-3090 (`marlin-int8-*.patch`) | ~+20% e2e at high concurrency (1222 vs 1094 tok/s) | **High-concurrency** lever. ⚠️ Negative-group-scale bug (AutoRound ~50% neg scales read unsigned → garbage). We are T≤4 single-stream → lower priority now |
| 2.4 | **lm_head provenance / numerical-equivalence ledger** — track which layers are bit-native vs requantized | v100-skinny (`lmhead_provenance.md`, `terminology_audit.md`) | correctness discipline | Do this whenever you requantize: keep a ledger of which objects changed precision |

---

## 3. SPECULATIVE DECODING (MTP / draft head)
The MTP head's cost is dominated by **re-running the vocab lm_head per draft**. Two independent attacks: shrink the vocab the *drafter* uses, and shrink the drafter's precision.

| # | Technique | Source | Win | Notes / porting |
|---|---|---|---|---|
| 3.1 | **Draft vocabulary** — drafter scores a **40,960-row slice** of lm_head instead of 248k. Misses = guaranteed rejection (exact, target sample used). | qwen38-3090 (`build_draft_vocab.py`, `draft_vocab_ids.json`, `qwen3_5-mtp-draft-vocab.patch`) | draft ~3ms → ~0.5–1ms; **92%→97.5% coverage = 98→109 tok/s** | **THE #1 lever for us.** Count ids over the model's *own* outputs (not web text). Coverage saturates ~40k (model emits ~54k distinct). ⚠️ **Their `draft_vocab_ids.json` (40,960 ids, max 248076) is for the SAME model — likely directly reusable.** Target quality unaffected (only proposals shrink) |
| 3.2 | **Cheap drafts enable k=4+** — with a draft vocab + int4, each extra draft is ~1ms, so 4 drafts pay off (opposite of NInfer where k=4 was *worse*) | qwen38-3090 (optimizations #5) | MTP-4 faster than MTP-2 | Our k=4 regressed (29.87) because our drafts are full-248k + T>4 non-specialized kernel. **Draft vocab (3.1) flips this** — revisit k=4 after 3.1 |
| 3.3 | **Draft `sample_method = greedy`** — a config flag that closes a 10–25 pt acceptance gap | v100-skinny (`acceptance_gap_notes.md`) | +10–25% acceptance | Cheap config win. Check NInfer's draft sampling mode |
| 3.4 | **Block drafter (DFlash2)** — 5 Qwen3 layers predict a whole 7-token block **non-autoregressively** from target's layer 5/19/33/47/61 hidden states + a path selector over 16 candidates/slot | qwen38-3090 (`dflash2-*.patch`, `drafter/quant_dflash2.py`) | 4.80 vs 4.28 tok/step; 122–132 t/s | **Bigger architectural bet** (separate 1.92B drafter, GPTQ W4A16 1.19GB). The "one lever left after acceptance." Not for the immediate 70 goal; candidate for >90 |
| 3.5 | **Chain-MTP depth cap** — verify qlen > 16 corrupts output; k ≤ 15 is the serving cap | v100-skinny (`mtp_width_findings.md`) | correctness boundary | Universal: find the arch/model's verify-length cap before pushing k |
| 3.6 | **Lookup drafting** — draft from the request's *own context* (verbatim reproduction), point-mass distribution, verify a longer block than the drafter | qwen38-3090 (`dflash2-lookup-drafting.patch`) | 159→381 tok/s reproducing a 25k doc | **Long-context/RAG only.** Not relevant to our low-context agent prompts. Skip for now |
| 3.7 | **Native round = single CUDA graph** — one complete k-round (drafter×k → verify → commit) captured as one graph | v100-skinny (`native_round_design.md`) | removes per-op launch overhead | Our **C1 / P1.2** (CUDA-graph the MTP head). Fixed-shape round → graphable. Directly applicable |

---

## 4. ATTENTION (the verify step)
The verify step has **k+1 query rows per request** — most attention backends under-utilize SMs on multi-query decode.

| # | Technique | Source | Win | Notes / porting |
|---|---|---|---|---|
| 4.1 | **Split-KV attention for multi-query verify** — FA2 only splits KV when q==1; a Triton split-KV kernel gives every (req, kv-head, query-tile) NUM_SEGMENTS blocks + online-softmax combine | qwen38-3090 (`spec-decode-attn.patch`) | 57µs→23µs per layer @1.5k ctx; 1.3ms→120µs @16k | **Directly attacks our 38.26ms verify floor.** On our T=4 TP2 verify, attention may be SM-idle too. Port = write a split-KV GQA decode kernel in NInfer (we already have GQA kernels; add the multi-query KV split + query-row tiling). ⚠️ partial buffers must be fixed-size (captured in the CUDA graph) |
| 4.2 | **Query-row tiling** so the verify block can exceed the drafter's block | qwen38-3090 (spec-decode-attn, gotcha #13) | enables long verify blocks | Pair with 4.1 |

---

## 5. STATE / KV DTYPE
| # | Technique | Source | Win | Notes / porting |
|---|---|---|---|---|
| 5.1 | **16-bit (fp16) recurrent state** — GDN/Mamba state fp32→fp16 (not bf16; 10 vs 7 mantissa bits) | qwen38-3090 (optimizations #3, gotcha #11) | halves state footprint+traffic; all 64 seqs run (was 37); perplexity unchanged | **Port to NInfer:** if our GDN state is fp32, making it fp16 halves the per-step read/write + our rebase (0.40ms) traffic. The kernel is already ~85% BW — the dtype is the lever |
| 5.2 | **int8 KV cache** | qwen38-3090 (`spec-decode-int8-kv.patch`) | halves KV read | Higher-context lever. Our test is ~200 ctx → low impact now; matters at 4k+ (the user's "low context" regime) |

---

## 6. SAMPLING
| # | Technique | Source | Win | Notes / porting |
|---|---|---|---|---|
| 6.1 | **Sort-free small-topk + multi-block softmax** — with top-k ≤ 64 known on host, mask = one `topk`, softmax multi-block (vs sorting the whole 248k vocab per row) | qwen38-3090 (`sampler-small-topk-fast-softmax.patch`) | +4% at default sampling | Port = fast argmax/top-k in NInfer's sampler + draft sampling. Our argmax is 8.2µs (Window #1) — likely already fine at T=1; check the verify path |

---

## 7. TENSOR / PIPELINE PARALLELISM
| # | Technique | Source | Win | Notes / porting |
|---|---|---|---|---|
| 7.1 | **TP2 tensor-parallel decode** (weight sharding, allreduce post-MLP) | NInfer (our TP2 work) | our 32.5→53.77 path | For arches where the model doesn't fit on one GPU |
| 7.2 | **TP4 for single-stream latency** on 4× V100 | v100-skinny (`hardware_audit.md`, `DEPLOYMENT.md`) | the V100 box's decode topology | When N GPUs are available and model fits split |
| 7.3 | **Hybrid PP2→TP2** for models that don't fit even split | NInfer (our PP2 work) | 27B on 2× under-capacity GPUs | Fallback topology |
| 7.4 | **Weight sharding: pointer-based, geometry recomputed inline, planar RowSplitK128V1** (low/high/scale planes) | NInfer (`weight_shard.*`) | exact split | Port = same sharding math for any arch (layout-driven, arch-agnostic) |

---

## 8. SCHEDULING / OPS (correctness + no-silent-loss)
| # | Technique | Source | Win | Notes / porting |
|---|---|---|---|---|
| 8.1 | **Quality battery as a gate** — perplexity + GSM8K against the live server *before* trusting any tok/s | qwen38-3090 (gotcha #1, `quality_battery.py`) | int8 path served garbage an hour before a perplexity check caught it | **Always.** A tok/s number without a quality check is worthless |
| 8.2 | **Prefix caching for hybrid (GDN) models** — resume recurrent state from last cached block boundary | qwen38-3090 (optimizations #9) | 23s→1s follow-up turns; 222s→17s shared system prompt | Multi-turn/RAG. Not single-shot decode |
| 8.3 | **Dirty-GPU silent 25% loss** — memory profiler runs once at startup; a still-releasing prior process shrinks the pool | qwen38-3090 (gotcha #2) | avoid silent 25% | Ops: gate start on GPU actually free |
| 8.4 | **Cold-start pool / restart-once** — cold torch.compile profile inflates activation peak → smaller pool | qwen38-3090 (gotcha #16) | 196k→224k KV tokens | Ops: benchmark twice, first run reads 30–50% low (gotcha #8) |
| 8.5 | **Random-token benchmarks flatter spec decode** — use real prompts | qwen38-3090 (gotcha #6) | same server 35/83/151 t/s on random | Always benchmark real prompts |
| 8.6 | **Decode residual ledger** — nsys per-step GPU cost breakdown methodology | v100-skinny (`decode_residual_ledger.md`) | the profiling method behind our B1 breakdown | Use this method for every new GPU |

---

## 9. PRIORITY MAP — our current target (2× 5060 Ti → 70+ t/s, Qwen3.8-27B)
Ranked by (expected win × effort) for *our* engine (see doc 11 for the round math: 50.76ms/round, verify 38.26ms floor, MTP head 12.46ms):

1. **3.1 Draft vocabulary (40k)** — biggest, no target-quality risk. Cuts the MTP-head output_head cost ~3-4×. **Try reusing their `draft_vocab_ids.json` (same model).** → likely crosses 70 by itself.
2. **4.1 Split-KV verify attention** — attacks the 38.26ms verify floor (our remaining hard wall). Needed to go past ~77 and to make k=4 viable.
3. **3.7 / P1.2 CUDA-graph the MTP round** — kills the ~4ms launch/D2H tail (fixed shape).
4. **2.2 GPTQ int4 output_head + drafter** — quality-aware; de-risks, cuts further. Gate on A2.
5. **3.3 Draft sample_method=greedy** — check the config; possible free 10–25% acceptance.
6. **5.1 fp16 GDN state** — halves GDN state/rebase traffic (our 0.40ms rebase + GDN verify).
7. **k=4 revisit** — *after* 3.1+4.1 (cheap drafts + fast verify flip our k=4 regression).

**Not now:** 3.4 DFlash2 (architectural, for >90), 3.6 lookup drafting (long-context), 2.3 W4A8 (high-concurrency), 5.2 int8 KV (higher-context), 8.2 prefix cache (multi-turn).

---

## 10. REUSABLE RECIPE — running this analysis for the next GPU
When we take on a new GPU, run in this order (each step is where the previous repos earned their keep):

1. **Hardware audit** (`nvidia-smi`/`rocminfo` + CUDA device props → SM count, HBM/GDDR BW, L2, TC gen, P2P topology). *(v100-skinny `hardware_audit.md`)*
2. **Window #1 microbench** → measure real BW + skinny GEMV kernel → compute roofline. *(NInfer Window #1; 1.5)*
3. **Artifact dump** → list the big objects (lm_head/embed/drafter) + their dtype + bytes. Decide the quant targets. *(our `dump_artifact`; 2.x)*
4. **Baseline + residual ledger** (nsys per-step cost) → find the verify vs MTP-head split. *(3.x, 8.6)*
5. **Apply the catalog** in §9 priority order, **gating every change on the quality battery** (8.1) and token-identity (A2).
6. **Record** the arch-specific findings back into this catalog (the "porting" column is the living part).

> **Rule of thumb across arches:** decode is bandwidth-bound → (a) make the big reads cheaper (quant + draft vocab), (b) make the verify's multi-query attention use the SMs (split-KV), (c) graph the fixed-shape round. The specific numbers change; the levers don't.
