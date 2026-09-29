# Radiance MXFP4 Optimization Survey — R9700 RDNA4 stack, ported ideas for NInfer
**Purpose:** full survey of `radiance-vllm-mxfp4` (the 280 tok/s Qwen3.8-27B / 2×R9700 Reddit build),
organized by technique, each tagged with **where the code lives in radiance**, **where the NInfer
counterpart is**, **their measured win**, and **effort/risk for us**. These are the next adventures;
low-effort drop-ins are flagged for the post-multi-batch sprint.

**Date:** 2026-09-02 · **Source repo:** `/home/intel/Downloads/radiance-vllm-mxfp4`
(codeberg `ggz14/radiance-vllm-mxfp4`, local clone) · **Format:** same as `12_cross_repo_optimization_catalog.md`.

**Headline numbers (their box, 2×R9700 32GB TP2, MXFP4 body + fp8 DFlash2 drafter SPEC=7):**
decode 224.3 t/s combined (22.8 ms/step, 5.5–5.9 tok/update), 573 t/s aggregate @ conc-8,
prefill 4300–4800 t/s to 64k, 943,581 KV tokens, weights 9.4 GiB/GPU.
Quality: WikiText-2 ppl 8.3708, GSM8K 97.8% — held across the whole optimization stack.

**Architecture match:** same model family (Qwen3.8-27B GDN hybrid: 48 GDN layers + 16 attn),
TP2, MTP/spec-decode. Their kernels are HIP/RDNA4 (gfx1201 WMMA) — **not portable** — but every
idea below is architecture-level and portable. Their repo is unusually honest: every landed change
carries its measurement, its gate, and its verdict (including negative results).

---

## 0. Repo inventory (what lives where in radiance)
| Area | Files |
|---|---|
| Kernels (MXFP4 W4A8 GEMM) | `radiance_mxfp4_fp8.hip` (60k, the one HIP file in-repo), `radiance_mxfp4.py`, `mxfp4-configs/` |
| GDN path | `radiance_gdn.py`, `patch_gdn_metadata.py`, `patch_gdn_shared_build.py`, `patch_gdn_wmma.py`, `patch_conv1d_blockn.py`, `radiance_gdnmerge.py` + `patch_gdn_merge_inproj.py`, `patch_gdn_wmma.py` |
| Spec decode | `radiance_draft.py` + `radiance_draft_gpu.py` + `patch_mtp_loopbreak.py` (dynamic MTP depth), `patch_dynwidth.py` (dynamic verify width), `dflash2/` + `patch_dflash2*.py` + `radiance_drafthead.py` + `radiance_verifyhead.py` (DFlash2 block-diffusion drafter + int2 verify head) |
| Attention | `radiance_r4d_attn.py` (R4D backend), `patch_unified_attention_lds.py`, `patch_unpad.py` (libr4d kernels live in external `StillDeadcode/libr4d`) |
| All-reduce | `radiance_allreduce.py`, `radiance_arnq.py`, `radiance_aroverlap.py` (negative result), `patch_ar_geometry.py`, `patch_ar_maxbytes.py` |
| GEMM dispatch / skinny | `radiance_gemm.py`, `patch_skinny_gemm.py` |
| KV | `patch_kv_group_size.py`, `calibrate-kv.sh`, `kv-profiles.tsv` |
| Fusions | `patch_rmsquant_fusion.py`, `radiance_rmsquant.py`, `patch_radiance_fusion.py` |
| Checkpoints | `fp8_mtp.py` (fp8 MTP head requant), `quantize_dflash_mxfp4.py`, `paroquant/` |
| Measurements | `README.md` ("Where the speed came from" table), `MXFP4-NOTES.md` |

---

## 1. Decode launch overhead & fusion  ← highest relevance to NInfer's MTP step

| # | Technique | Radiance code | NInfer counterpart | Their win | Effort/risk |
|---|---|---|---|---|---|
| 1.1 | **GDN `in_proj` single-GEMM merge** — `in_proj_qkvz` + `in_proj_ba` are two ColumnParallel linears over the *same* activation; concat weights (row-local in N → bit-identical) and run one GEMM. Removes 96 of ~1904 launches + 48 activation quants per forward (48 GDN layers). | `radiance_gdnmerge.py` (merge at load, after `process_weights_after_loading`, before graph capture), `patch_gdn_merge_inproj.py` (call-site install) | `text_context_impl.h` `gdn_input_projection_tp_verify` (~L1788, feeds q4/k4/v4/g4/b4 ~L1831-1836). **VERIFY FIRST:** NInfer's naming (`gdn_input_proj…` producing all of q/k/v/g/b in one op) suggests this may *already be merged*. If yes → done, no action. | -2.9% ms/step (26.25→25.50), -6.5% stacked with WPERM | **LOW (verify, ~10 min code read)** if unmerged → MEDIUM (weight concat + one kernel) |
| 1.2 | **Skinny split-K GEMM for small projections** — vendor BLAS lays tiny weights out as a handful of workgroups → underfilled grid; custom split-K kernel takes M∈[6,64]. Flagged shape: GDN `in_proj_ba` (480 KiB, **48×/step, 28.5µs → 3.6µs**). | `patch_skinny_gemm.py` (hook), `radiance_gemm.py` (measured (N,K)→config table; declines everything else) | NInfer decode GEMV already ~92% BW roofline (`tp_kernel.cu` simt r8_c4/r8_c8, doc 12 §1.2) — but **audit the small GDN gate/ba projections specifically** at decode M. If they ride the generic GEMV, a split-K variant fills the SMs. | 8× on the one flagged shape; +3.5% tokens/s stacked | **LOW-MED** (audit = launch/shape dump; kernel = small) |
| 1.3 | **Fused split-K reduction** — KS blocks racing an atomic counter; last arrival reduces in place, **no second launch**. | in `radiance_mxfp4_fp8.hip` (decode tiling) | Applies to any split-K kernel we add (1.2) or future skinny work | -3.9% of step time, bit-identical | LOW (pattern, not a project) |
| 1.4 | **GDN decode conv+recurrent fused into one launch** | `radiance_gdn.py` + `patch_gdn_*.py` (fused decode conv+rec path) | `src/ops/kernel/causal_conv1d.cuh` + `gated_delta_net/recurrent.cuh` (`apply_gdn_transition` ~L660). **CHECK whether NInfer's decode conv and rec transition are separate kernels.** | -0.4%, **bit-identical** | **LOW** (check first; if already one kernel → done) |
| 1.5 | **Launch-gap stack** — traced quant (no redundant quant passes), fp8 residual stream, fused AR epilogue. 1477 → ~1080 launches/step. | `patch_radiance_fusion.py`, `patch_rmsquant_fusion.py`, AR epilogue in `radiance_allreduce.py` | Count NInfer's launches/MTP-step (nsys or launch counter) and hunt the same three classes: redundant quant, non-fused residual ops, allreduce as standalone op. Our lpp per-layer H2D was this class. | **25.4 → 22.66 ms/step — their single biggest win** (+14% single-stream, +24% conc-8) | MED (audit LOW; each fusion MEDIUM) |
| 1.6 | **Epilogue store width** (T=512 at prefill M) — wider stores, 405→429 GB/s | `radiance_mxfp4_fp8.hip` | prefill attention/GEMM epilogues | 726→685 µs @ M≥2048 | LOW-MED, prefill-only |

**Method note (theirs):** "decode launch-gap stack" was gated with paired GSM8K sign test
(p=0.219) + bit-identical epilogue kernels — i.e., a fusion is only "free" if the reference
reduction order is preserved or the quality gate passes.

---

## 2. Quantization (the big adventures)

| # | Technique | Radiance code | NInfer counterpart | Their win | Effort/risk |
|---|---|---|---|---|---|
| 2.1 | **MXFP4 body (W4A4 native) + W4A8 fp8-WMMA GEMM** — hand-written kernel reaches the fp8 matrix instruction Triton won't emit; 1.47–2.26× vs tuned aiter, *and* 4.2× more accurate (fp8 activations beat mxfp4 activation quant). Weights 9.24 GiB/GPU vs ~12.6 FP8 → headroom goes to KV. | `radiance_mxfp4_fp8.hip` (two tilings: prefill BM=256; decode TM=ceil(M/16) + split-K, `RADIANCE_MXFP4_DECODE_MAX_M`), `patch_quark_mxfp4.py` | Full 4-bit body on sm_120a. NInfer's decode GEMV is already ~92% roofline, so the win is *weight traffic* (27B bf16→4bit ≈ 2.6× less) + KV headroom, not kernel efficiency. Largest single adventure on this list. | decode GEMM 1.47–2.26×, +28.5% agg @4-conc (decode tile) | **HIGH** (weight pipeline + kernels + quality gates) |
| 2.2 | **fp8 drafter/MTP weights** — 4-bit drafter *cost* acceptance (2.5→2.21; AWQ didn't rescue: mxfp4's per-32 e8m0 already does the scaling); fp8 e4m3 per-channel holds 2.60–2.80. | `fp8_mtp.py` (checkpoint requant), `MXFP4-NOTES.md` | NInfer MTP module weights → fp8. Doc 12 §2.2 already tracks **GPTQ int4 lm_head + drafter** as "P1.3 done right" (3090 repo) — this radiance result *confirms fp8/int4 drafter is the safe point; 4-bit drafter is not*. | -17% decode weight traffic, acceptance held | **MEDIUM** (quant pipeline + A2 identity gate) |
| 2.3 | **int2 target verify head with exact bf16 rerank** — lm_head at 2 bits, exact rerank restores the distribution; 24/24 seeded completions byte-identical. | `radiance_verifyhead.py`, `patch_verify_head.py`, `patch_mtp_mm_mask.py` | NInfer `output_head` is already W8G32 (doc 12 §2.1). int4 is the known P1.3; int2-with-rerank is the next step on that ladder. Bold but lossless-by-construction. | +2.9% combined decode | MED (rerank kernel + equivalence proof) |
| 2.4 | **fp8 KV cache** | vLLM flag (`--kv-cache-dtype fp8`) + `kv-profiles.tsv` | **CHECK current NInfer KV dtype** (kvarn; doc 117 explored int4 KV cache types). If bf16 → fp8 = ~2× capacity + halved attention BW. Quality: their ppl 8.3708→8.3707, GSM8K p=1.000 — "error is free" *on their end-task gates*. | 940k KV tokens @ MXFP4; capacity lever | MED (dtype through append/attend/rewind + quality gate) |
| 2.5 | **fp8 QK + PV legs in prefill attention** | R4D backend (`radiance_r4d_attn.py`, libr4d) | NInfer prefill attention. Upstream vLLM deleted these legs on *kernel* accuracy; end-task gates say the error is free — same gate must be run for us. | +10.8% prefill @106k (3448→3831 t/s) | MED |

---

## 3. KV capacity management

| # | Technique | Radiance code | NInfer counterpart | Their win | Effort/risk |
|---|---|---|---|---|---|
| 3.1 | **KV group size by capacity, not smallest bucket** — allocator-derived group size | `patch_kv_group_size.py` | kvarn paged layout (block=64 tokens/page today). Serving shape (doc 134) should derive block/group size from measured capacity | +20.7% KV tokens (739k→892k), 2.82→3.41× concurrency | LOW (config-level) |
| 3.2 | **Explicit calibrated KV pin** — measure peak under real 8-way 8192-chunk load, hand KV everything but ~0.3 GiB; do NOT re-derive from the profiler's "fit into budget" line (it computes against GPU_UTIL, not the card) | `calibrate-kv.sh`, `kv-profiles.tsv` | NInfer serving allocator (doc 134 tp2 serve integration). Re-derive after anything moves weights/graph sizes | +5.7% KV (892k→943k), survives 260k sweep no-OOM | LOW (runbook) |

---

## 4. Speculative-decode structure (lossless by construction)

| # | Technique | Radiance code | NInfer counterpart | Their win | Effort/risk |
|---|---|---|---|---|---|
| 4.1 | **Dynamic verify width** — per-request EMA of accepted counts caps verify rows; content-dependent spread is 2× (code/json 4.7–6.0 tok/update want depth 7+, prose ~2.8). Pure scheduler-side CPU; the cap engages only at ≥3 running requests (below that M sits in the weight-bound flat zone). | `patch_dynwidth.py` (EMA α=0.35, margin 2, floor 2) | NInfer batched MTP: per-lane verify width when acceptance histories diverge (e.g., B=1 phase verifying 4 rows for 1 lane). Complication: width change moves M → decode-tile/split-K choice → graph change. | +11–13% @conc-8 (52→46 ms steps); single-stream a wash | MED (scheduler plumbing + graph variants) |
| 4.2 | **Dynamic draft depth (MTP)** — per-slot confidence-product gate (keep drafting while ∏conf ≥ τ=0.35) + verbatim n-gram tail (free when it equals the drafter's own top guess) + batch-size schedule (1:8, 2:7, 4:6, 8:5). All hot-path on-device; one tiny D2H per slot short-circuits the serial loop. | `radiance_draft.py`, `radiance_draft_gpu.py`, `patch_mtp_loopbreak.py` | NInfer already has skip-draft-continue when `!any_draft` (93fda35e) — this is the *partial* extension: break the k=3 chain per-lane at low confidence. Lossless: every token still verifies through the unchanged sampler. | +5.3% (policy-tuned), mostly at concurrency | LOW-MED (plumbing exists; add per-lane conf check + loop break) |
| 4.3 | **SPEC depth is content-dependent** — a sweep on non-repetitive prose flipped default to 5; the weighted mix says 7 (184.3 vs 159.4 t/s). Tuning on one content class picks the wrong default. | `MXFP4-NOTES.md`, README "two results worth carrying" | Applies to any NInfer depth/width tuning we do: benchmark on a *mix*, and prefer the dynamic knob (4.1/4.2) that dissolves the trade | lesson | — |
| 4.4 | **DFlash2 block-diffusion drafter** — emits all SPEC=7 positions in ONE graphed pass (no serial draft forwards); 5.5–5.9 tok/update. | `dflash2/`, `patch_dflash2*.py` (selector topk, fused kv fp8, mxfp4 kv), `radiance_drafthead.py` | Alternative drafter architecture vs NInfer's MTP chain. Doc 118 (`dflash_implementation_plan.md`) already exists in NInfer — this is the reference implementation + measured target. | the 280 tok/s headline | **HIGH** (new drafter + calibration) |
| 4.5 | **Prefix caching on GDN hybrid (`mamba-cache-mode=align`)** — snapshot/restore GDN recurrent state at block boundaries; verified bit-identical to full recompute **including under MTP**. | vLLM flag + compose (`--enable-prefix-caching --mamba-cache-mode=align`) | NInfer already has a **prefix GDN checkpoint buffer** (73.4 MiB, see warmup log) + doc 116 (phase4 prefix restore unification). This is the external validation that the approach is correct; compare their boundary-snapshot semantics against ours. | large TTFT on shared prefixes | LOW (verify parity with existing work) |

---

## 5. Communication (TP2)

| # | Technique | Radiance code | NInfer counterpart | Their win | Effort/risk |
|---|---|---|---|---|---|
| 5.1 | **TP=2 P2P one-shot all-reduce** with optional **wht6 6-bit compressed payload** | `radiance_allreduce.py`, libr4d `ar_oneshot_2rank_exact`, `patch_ar_geometry.py`, `patch_ar_maxbytes.py` | NInfer TP2 uses NCCL. One-shot P2P over PCIe (5060 Ti has no NVLink) is the same class of idea; the 6-bit payload compresses the AR message itself. | AR = 1.32 ms/call @80 MiB = 15.5% of prefill GPU time (their prefill problem) | MED (custom comm path vs NCCL) |
| 5.2 | **⚠️ NEGATIVE: AR/GEMM overlap** — spin-wait AR kernel occupies CUs *while waiting over PCIe* → contention, not hiding. Measured -3.5 to -5.6%, ships OFF. | `radiance_aroverlap.py` (docstring is the full post-mortem) | **Do not** attempt to overlap NCCL/custom AR with compute on the same device expecting a win; if wanted, the transfer must be on a true async engine (SDMA/`hipMemcpyPeerAsync` analogue) at 2.7× bytes | lesson | — |

---

## 6. GDN numerics lesson (audit, not port)

Their libr4d v0.4.0 shipped NaN on this model family (ppl 653586). Three unguarded `__expf`
overflow sites, all the same shape — `0 * INF = NaN`:
1. **kkt_solve padding rows**: padding rows force `gi=0` but keep real `gb` cumsum → `d = -gb` large
   POSITIVE, violating the "never positive" invariant that holds only for *live* rows; NaNs in a
   padding row merge into live rows through the WMMA tile.
2. **chunk_scan split-form halves**: `e^{g_i-c}·e^{c-g_j}` with cref at chunk midpoint; a span past
   ~176 sends one half to +INF.
3. **chunk_scan V' staged in bf16** (the dominant one): needs fp32 headroom under the 3.4e38 ceiling.

**Action for NInfer:** 10-minute code-read of our GDN chunk/scan + kkt paths for the same patterns
(NInfer's rec path is pure fp32 → likely clean, but the *padding-row invariant* is the classic one —
anywhere we compute a live-only invariant for all rows of a padded tile is the bug).
**Effort: LOW, no GPU, zero risk.**

---

## 7. Negative results & traps (don't walk these paths)
- **AR/GEMM overlap** — see 5.2.
- **Microbench tilings** — an epilogue prefetch justified on an idle-GPU M=2048 microbench was 14%
  *worse* at the real M=8192 shape (+40 VGPRs/thread cost occupancy exactly when 8192 workgroups
  compete). **Fill the GPU before judging a tiling.**
- **4-bit drafter** — acceptance cost (2.5→2.21) > bandwidth saved. fp8 is the floor for drafters.
- **BK=128** — wins at decode (1.87×) but -34% at prefill; the loss at prefill was purely an LDS
  occupancy cliff. Tile choices are M-regime-specific.
- **Compile-cache flags must key on the feature flags** — a warm cache from a different flag
  *silently replays the old graph*. (Same trap class as our build-cache discipline.)
- **vLLM dual model runners** — patching one of two `load_model` bodies is a silent no-op.
  (Generic lesson: verify the patch actually applied; our `str.replace` no-op trap, same class.)

---

## 8. Methodology worth adopting
1. Report **ms/step**, not tok/s — acceptance luck swings tok/s ~14% at fixed config.
   `ms/step = 1000 × (accepted/draft + 1) / tok_s` divides it out.
2. Every change lands with its own gate: ppl + GSM8K paired sign test, or bit-identical greedy
   completions where the change is exact.
3. **Land one at a time** — "a combined A/B cannot attribute a regression."
4. Kernels named for their exact geometry; refuse on mismatch; visible in startup log.
5. Byte-reproducible pinned builds (sha256-verified .so); cache dirs keyed on config.
6. "Nothing in the way was a compiler limitation" — their 6.1× MXFP4 win was three soft allowlist
   gates, not a new kernel. *Check the flags before writing the kernel.*

---

## 9. LOW-EFFORT DROP-IN SHORTLIST (user ask: low risk first)
Ordered by effort, all post-multi-batch-complete:
1. **§6 GDN NaN-pattern audit** — code read only, no GPU, ~10 min. (do this first, even now if idle)
2. **§1.1 verify GDN in_proj already merged** — code read ~10 min; if unmerged it becomes a MED item.
3. **§1.4 check decode conv+rec is one kernel** — code read ~10 min; if two kernels, -0.4% bit-identical fusion.
4. **§1.2/§1.5 launch + small-projection audit** — count launches/MTP-step + dump GDN gate/ba
   GEMV shapes at decode M. Measurement only; decides whether 1.2/1.5 are worth doing.
5. **§2.4 check current KV dtype** — if bf16, fp8 KV is the biggest single capacity+BW win (MED effort).
6. **§4.2 dynamic draft depth (per-lane early chain break)** — plumbing partially exists
   (skip-draft-continue), lossless by construction.
7. **§3.1/§3.2 KV capacity sizing runbook** — config-level, for serving (doc 134).

Then MEDIUM: §2.2 fp8 drafter (confirms doc 12 §2.2 P1.3 direction), §4.1 dynamic verify width,
§5.1 one-shot AR, §2.5 fp8 QK/PV.
Then HIGH (the real adventures): §2.1 4-bit body, §4.4 DFlash2 drafter, §2.3 int2 verify head.

---

## 10. Phase-gate interaction (does doc-128 M0/B-chain help these?)
**Yes — for exactly the right reason, with one caveat.**

1. **Validation speed & trust (the big one).** Every drop-in above, especially the numeric ones
   (fp8 KV, fp8 drafter, GEMM tiling, AR compression), must be proven "no regression." Today that
   proof is: full 64/128/256 + B=1 matrix in fresh processes + token diff + (for numeric changes)
   a quality bench. Gate v2 (pair on (F, anchor, gen), token-prefix compare, converged-state
   match, terminal=done) automates exactly that, and *pinpoints the first phase where data diverges*
   — which is the failure mode of a bad optimization (wrong GDN fusion, wrong KV dtype at attend,
   wrong draft width). For **bit-identical** changes (fusions 1.1–1.5, §4.2 early break) the gate
   gives a cheap STATE-MATCH certificate — most of the risk of a low-effort drop-in is "silent
   numeric drift," and the gate is the detector.
2. **Regression net for stream/concurrency changes.** The T=256 terminal race was found *because*
   we had a deterministic oracle. Several ideas here (AR overlap — even though we won't do it,
   dynamic width, per-lane draft depth) touch stream/scheduling structure and can introduce
   latent races. Gate + flake-harness (×10 runs) is the detection net for each new drop-in.
   This is the radiance "land one at a time, each with its own gate" discipline, mechanized.
3. **Attribution.** Their methodology: each change measured against the build immediately before.
   Phase dumps make that mechanical — dump before, dump after, gate diff → per-phase attribution,
   no combined A/B.
**Caveat:** the gate shortens the *verification loop*, not the implementation. Kernel adventures
(§2.1 4-bit, §1.2 split-K tuning) are kernel work either way; the gate is insurance + the merge
criterion, not the accelerator. For the low-effort shortlist (§9) the gate pays off almost
immediately because verification IS most of the work.

**Conclusion:** complete multi-batch → formalize the B-chain milestone (bf16 phases, test binary,
CI, doc-128 section) → *then* run §9 shortlist, each item gated. The order the user chose is the
correct order.
