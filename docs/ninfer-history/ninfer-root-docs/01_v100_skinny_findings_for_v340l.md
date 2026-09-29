# v100-skinny findings applicable to the 2× V340L (gfx900) scenario

**Status:** ARCHIVE

Source repo: https://github.com/Alfinaa9442/v100-skinny (cloned & audited 2026-08-19)
Target scenario: Qwen3.6-27B-class models on 2× AMD Radeon Pro V340L (4× gfx900 / Vega 20, 56 CU, 8 GB HBM2 @ 483.8 GB/s per die, fp16 2:1, no tensor cores, 64-wide wavefronts, 64 KB LDS/CU, PCIe 3.0 x8 per die, no cross-die fabric).

Confidence tags: **[measured]** = number measured in the repo's results/ or docs; **[design]** = documented design, not yet shipped; **[inference]** = our extrapolation to gfx900 (loose evidence, flag for re-verification).

---

## 1. The core transferable idea: W4A16 in-kernel dequant on pre-FP4 silicon

**[measured]** v100-skinny serves NVFP4 weights (e2m1 nibbles + fp8-e4m3 group-16 scales + global float scale, 0.5625 B/weight) on Volta by decoding to fp16 *inside* the GEMM kernel and doing fp16 math with fp32 accumulation. Plain decode: 86.6–91.2 tok/s on 4× V100 (TP4) vs 70.7 for the incumbent AWQ stack (1.22×) and vs 86.4 on a single RTX 5090.

**[inference]** This is *more* central on gfx900 than on V100: Vega 20 has no FP4, no FP8, no BF16 hardware at all. There is no native low-precision path to lean on — the entire low-bit story is "dequant in-kernel, run the fp16 2:1 ALUs." The e2m1 bit-twiddle decoder (TurboMind-derived `dequant8_tm`: shift sign/EM bits into fp16 positions, fold the 2^14 re-bias into the group scale) and the `fp8e4m3_to_half2` re-bias trick (`((b&0x80)<<8)|((b&0x7f)<<7)` as half = value×2⁻⁸) are pure ALU and port verbatim to GCN5.

Key enabler (why it's nearly free): **decode GEMMs are bandwidth-bound**, so each weight byte is dequantized exactly once as it streams. v100-skinny's M=1 SIMT kernel hit 648 GB/s = 78.5% of the measured 825 GB/s memcpy ceiling; in-server GEMM ran 596 GB/s (72%). **[inference]** Target the same 70–80% of 483.8 GB/s (≈340–385 GB/s effective) per die on gfx900.

## 2. Kernel family — what ports, what dies

v100-skinny dispatches by M (batch/verify width): SIMT M≤3 / QPN (m8n8k4) M 4–16 / WMMA (m16n16k16) M 17–64 / Marlin M>64.

| v100-skinny asset | gfx900 applicability |
|---|---|
| **SIMT dequant GEMV** (warp-per-output-row, HFMA2 inner loop, smem activation staging, XOR bank swizzle, fp16 accum windows flushed to fp32 every 16 codes) | **[inference] Ports directly.** wavefront-per-row; HFMA2 → `v_pk_fma_f16`; smem → LDS. Caveats: 64-wide wavefront = 2× register pressure per logical thread; 64 KB LDS (vs 128 KB) → halve the K-chunk (KC 1024→512); LDS is also 32×4B banks so the XOR swizzle math survives. |
| **QPN m8n8k4 tensor-core path** (quadpairs split N, prepacked B-fragments, no main-loop smem) | **Dies.** No matrix instructions on gfx900 (MFMA starts at CDNA1/gfx908). The whole M 4–16 tensor band is gone. |
| **WMMA m16n16k16 batch band** (software-pipelined dequant→smem→fragment) | **Dies** as written; the M>8 band becomes SIMT (issue-bound) or a plain tiled LDS GEMM. |
| **DP4A int8 path** (e2m1×2 is integer-exact → lossless int8 weights, `dp4a` 4 MACs/slot) | **Rejected on V100** (dp4a MACs share the INT pipe with the nibble unpack → issue contention). **[inference]** Same prediction on gfx900: INT8 MAD is on the same ALU pipe as the unpack, no separate INT tensor pipe. Skip unless a prototype says otherwise. |
| **Fused argmax lm_head epilogue** (M=1; logits never touch HBM; strict-`>` tie-break matches argmax-over-half semantics) | **[inference] Ports directly; high value** on the 248,320-vocab head (1.27B params ≈ 0.7 GB at 4-bit). Saves a full logits HBM round-trip per token. |
| **Split-K for N-starved shapes** | **[inference] Ports** (atomicAdd fp32 partials); keep as a tuning option. |
| **Short-K two-rows-per-warp trick** (K≤2048 rows ran at 66% of flagship BW with one row/warp; two rows restore latency hiding) | **[inference] Ports** — same latency-hiding logic; re-sweep per-die. |

**Net simplification:** on gfx900 the kernel family collapses to *one* SIMT dequant GEMV (M≤~8) + a large-M dequant GEMM for prefill. No M-band dispatch between SIMT/TC. That's less work than the V100 project — but see §5 for the M-wall consequence.

## 3. The M-wall: where verify cost stops being free

**[measured]** On V100, SIMT went compute-bound by M=5 (which is why QPN/WMMA existed); with the TC band, verify GEMM cost was **flat from M=16 to M=64** — that flatness is what made k=15 speculative depth economic (verify width "nearly cost-free").

**[inference]** On gfx900 there is no TC escape. Compute catches up when 3.56 FLOP/byte-per-M × M ≈ (fp16 peak/BW) per die = 21.5e12/483.8e9 ≈ 44.4 → **M ≈ 12 per die**. But that's the pure-GEMM crossover; with dequant on the critical path the practical wall is earlier, **M ≈ 6–8**. Consequences:

- Speculative verify (M = k+1) stays cheap only for **k ≤ 4–7**.
- The V100 k=15 economics (extraction 3.48×, "saturated at exactly k+1") **do not transfer**; plan k=2–4.
- Batched decode (M = concurrency) hits the same wall: 4 concurrent streams (M=4) ≈ ~1.2–1.3× the M=1 GEMV cost, so aggregate scaling is ~3–3.5× per 4 streams, not 4×.

## 4. Speculative decoding findings (chain-MTP) that transfer

All of these are **model+quantization properties, not hardware properties** — v100-skinny explicitly verified acceptance is invariant across kernel swaps (QPN⇄WMMA, two decimals) and across stacks.

- **[measured] Acceptance is a domain curve, not a scalar:** extraction 97–100% (flat to the tail), math 79–82%, code 53–64%, prose ~31% (collapses by position 4). Thinking mode lifts prose +7–10 pts.
- **[measured] Greedy drafter is a free 10–25 acceptance points** on sampled serving (`draft_sample_method=greedy`; stochastic proposals get rejected far more). Zero runtime cost.
- **[measured] Non-thinking prose never beat plain decode at any k** (peaks k=3, 90.0 vs 93.2 plain). Don't count on spec for chat.
- **[measured] lm_head provenance matters:** serving the *original* 4-bit lm_head codes (no re-quantization) removed most of the 4-bit head penalty (92% worst-case top-1 → τ-identical to BF16 on stable domains) and cut 2.54 GB → 179 MB/rank. For the V340L: keep the GGUF's native head quant, don't round-trip through bf16.
- **[measured] Deep-k corruption wall** (V100 flash-decode 16-query tile overruns at qlen 17): hardware-specific, **does not transfer** — but the *validation rule* does:
- **[measured] Validation rule (adopt verbatim):** *never quote a spec-decode throughput without a byte-level output diff against plain decode at matched config.* Corrupted verification inflates both acceptance and tok/s (the drafter trivially predicts the degenerate text the bug produces). They retracted a τ=10.59 ngram result on this basis.
- **[measured] Seeded triplicates** for any stochastic acceptance claim (run-to-run noise ±3–5 pts); greedy pairs for zero-variance checks.

## 5. Engine-level findings

- **[measured] Round anatomy (k=7, 4×V100):** 29 ms = verify GPU 13.5 + allreduce 1.7 (custom one-shot, 11–15 µs/op) + drafter+host 14 (of which **9–10 ms was Python orchestration**, GPU only 4.5). The host path is the largest single line.
- **[design] Whole-round CUDA graph** (drafter×7 + verify + rejection in one `cudaGraphLaunch`): built, byte-validated, but **inert in their env** — the captured graph failed to persist the drafter's recurrent state across rounds (τ→0, all drafts rejected). *Lesson for gfx900:* if you graph-capture a round on HIP, the drafter's GDN conv/SSM state + KV write-slot indices must be in the graph's owned/refreshed set; prove parity with a state-norm probe after N rounds before trusting it.
- **[measured] Custom one-shot allreduce at 11–15 µs/op** cut their per-layer collective to 1.7 ms/round on NVLink. **[inference]** On V340L the equivalent is a one-shot AR of the ~10 KB hidden vector over the on-card PCIe bridge / x8 links: bandwidth is trivial (µs-scale), **latency is the cost** (~2–3× NVLink). Budget ~2–5 ms/round for ~128 ARs; batch ARs every 2–4 layers if latency dominates.
- **[measured] NUMA/system audit worth 3–4%** on their box (workers pinned to the GPU's NUMA node, `numa_balancing=0`, app clocks pinned 1312→1530 MHz, persistence mode, C-states). **[inference]** Port the *checklist* to the V340L host: CPU affinity to the socket owning the PCIe root, clock/power pinning (Vega powerplay tables are soft-tunable on Linux — community-confirmed), persistence of settings.
- **[measured] GDN chain-spec fast metadata build: −1.4 ms/step**, byte-identical — device-side/pre-computed metadata instead of host rebuild. **[inference]** Same win available on any stack that rebuilds per-step metadata on the host.
- **[measured] Concurrency:** their M≤64 WMMA flat zone gave 90 → 1,301 tok/s across 1→64 streams (11.6× at 32 streams, 2.7× step cost). **[inference]** The gfx900 analogue is smaller (M-wall at ~6–8): expect ~3–3.5× aggregate at 4 streams.
- **[measured] Capture-sizes-are-tokens** (vLLM footgun: `cudagraph_capture_sizes` entries are token counts; under-provisioned lists silently run eager and masquerade as a throughput collapse). Keep the lesson: *verify the served config from the boot log, not the intended one.*

## 6. Things that do NOT transfer (avoid wasting time)

1. QPN/WMMA/DP4A kernels — no tensor cores, no dp4a-on-separate-pipe on gfx900.
2. The 16-query flash-decode tile bug — V100-kernel-specific.
3. NVLink-based TP economics — V340L dies have **no cross-die fabric**; TP over PCIe 3.0 x8/x4 is latency-bound (community-measured: 4-die cross-card tensor split at ~30–65% GPU idle; 8-die x4 collapses to 35%).
4. Their 825 GB/s roofline — re-measure the per-die memcpy ceiling on the actual card first (V340L units vary: power caps, limp-mode units reported by owners).
5. The vLLM-fork integration layer — on gfx900 the realistic hosts are llama.cpp (ROCm/Vulkan) or a from-scratch engine (see ninfer analysis); the vLLM gfx900 path is deprecated and painful.

## 7. Methodology to adopt wholesale

1. **Roofline-first:** measure memcpy ceiling, target a % of it, sweep tiles/occupancy against it.
2. **Per-op microbench + full-pipeline A/B** in the same repo (their benchmarks/ + results/ CSVs with losslessness verdicts printed alongside throughput).
3. **Byte-diff validation** for every spec-decode number (rule in §4).
4. **Losslessness-laddered adoption:** each k-depth row validated against plain decode before being quoted.
5. **Retraction discipline:** they published a retraction (ngram τ=10.59) with root cause. Keep a "retracted" section, not a silent edit.
6. **Terminology audit:** they banned bare "head" (lm_head vs mtp_head vs backbone) after it caused wrong causal stories. In the V340L project: "die" vs "card" vs "device" must be explicit (1 card = 2 dies = 2 devices).

## 8. Bottom line for the V340L project

The transferable core is: **(a)** one well-tuned SIMT dequant GEMV targeting 70–80% of per-die bandwidth + fused argmax head; **(b)** small-k (2–4) chain-MTP with greedy drafter, domain-aware (math/structured only); **(c)** graph-captured round + custom one-shot AR (with the drafter-state-persistence hazard on the watchlist); **(d)** the validation discipline. The non-transferable core is everything tensor-core and everything NVLink.
