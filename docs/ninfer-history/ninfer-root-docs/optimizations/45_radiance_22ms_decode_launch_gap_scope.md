# Scope: port the radiance "decode launch-gap stack" (the 22 ms win)
**Purpose:** scope doc 44 §1.5 — the single biggest measured win in the radiance project
(`db9dba6`: decode 25.4 → **22.66 ms/step**, launches 1477 → ~1080/step, weighted single-stream
+14%, conc-8 aggregate +24%) — for NInfer. This doc records (a) what that change actually was on
the radiance side, (b) what NInfer already has per component (verified in code, not assumed), and
(c) the remaining work items.

**Date:** 2026-09-02 · **Source:** doc 44, radiance repo `/home/intel/Downloads/radiance-vllm-mxfp4`
(README "Where the speed came from", `patch_rmsquant_fusion.py`, `patch_radiance_fusion.py`,
`radiance_rmsquant.py`, `radiance_allreduce.py`).

**Bottom line up front:** of the three components of the radiance win, NInfer **already implements
two by design** (fused one-shot AR + residual; no per-layer activation quants at all). The
portable remainder is launch-count reduction in the data-movement and small-op classes, anchored
by a measured baseline. The 22.66 ms number itself is *their* box/stack (2×R9700, MXFP4 body,
DFlash2 drafter) — do not treat it as our target absolute.

---

## 1. What the radiance change actually was

README row (their honest per-change table): *"decode launch-gap stack -- traced quant, fp8
residual stream, fused AR epilogue (`db9dba6`) | 25.4 -> 22.66 ms/step; decode launches 1477 ->
~1080/step; weighted single-stream +14%, conc-8 aggregate +24% | GSM8K 500q 97.8, paired sign
test p=0.219; epilogue kernels bit-identical to the traced reference"*.

Three components, in radiance's vLLM/Inductor world:

| # | Component | Radiance mechanism | Code |
|---|---|---|---|
| A | **Traced quant** | Hoist per-linear activation quants out of the opaque MXFP4 op so Inductor sees them in the graph, then fuse `rms_norm + per-token-fp8-quant` into one kernel (and `fused_add_rms_norm + quant`). Their measurement: the per-linear activation quant is **14,288 launches at decode (~4.0% of wall)**, each ~2.2 µs of work against ~4.7 µs of dispatch — i.e. pure launch overhead. Bonus: their merged GDN `in_proj` (two ColumnParallel linears over the same normed activation) means the fused norm collapses two identical quants into one. | `patch_rmsquant_fusion.py` (swaps the pattern's *replacement* op; patterns untouched), `patch_radiance_fusion.py` (registers the native-quant matcher variant), `radiance_rmsquant.py` (the replacement, **deliberately plain torch** — see caveat below) |
| B | **fp8 residual stream** | residual stream carried in fp8 instead of bf16, so the add+norm+quant chain folds and residual traffic halves | in-graph (vLLM `--kv-cache-dtype`-class flags + the fusion above) |
| C | **Fused AR epilogue** | the 2-rank all-reduce runs in the GEMM epilogue / fused into the producing kernel instead of a standalone AR launch | libr4d (external), wired via `radiance_allreduce.py` |

**Caveat worth stealing (from `radiance_rmsquant.py` docstring):** they first wrote the fused
norm+quant replacement as a `torch.library.custom_op`. It is opaque to Inductor, so it ran as an
eager chain of ~8 elementwise kernels — launches per forward went 1904 → 2009 and step time got
*worse*. The replacement had to be plain torch so Inductor traces and fuses the whole thing into
one Triton kernel. Generalization: **a fusion that hides work from the graph tracer is not a
fusion.** For NInfer (explicit C++ launches) this means: fuse at the kernel level, and re-count
launches after, because a "fused" op that internally launches N kernels is N launches.

**Methodology attached to the win (doc 44 §8, adopt all):** report ms/step (not tok/s); each
change measured against the build immediately before it; land one at a time (a combined A/B
cannot attribute a regression); bit-identical greedy completions where the change is exact,
ppl+GSM8K sign test where numeric.

## 2. NInfer current state, per component (verified)

Production decode path (TP2, `src/targets/qwen3_6/impl/runtime/text_context_impl.h`):
`run_layers` → per layer `gdn_mix_tp` (48 GDN) / `attn_mix_tp` (16 full-attn) + `mlp_tail`;
then the MTP module (`mtp_forward_stem`/`_tail`).

### 2.1 Component C (fused AR epilogue) — ✅ ALREADY DONE
`src/core/multi_gpu/one_shot_allreduce.{h,cu}`: `one_shot_ar_pinned_vec_kernel` is a one-shot
P2P 2-rank all-reduce (write-through PCIe host staging, epoch flags, CUDA-graph-safe via
`advance_epoch`) with the **residual add fused in the same kernel**
(`residual_buf[i] = bf16(bf16(residual_buf[i]) + sum)`, `one_shot_allreduce.cu` ~L98).
Capacity: 128 slots × 65,536 elems (128 KB) — covers hidden=5120 at T ≤ 12.
`TpGroup::allreduce_local_bf16` (`tp_group.cpp` L266) dispatches to it for decode-size messages,
NCCL fallback only above. Every layer in the decode path uses it with the residual folded in:
GDN `out_proj` (text_context_impl.h ~L1345), attn `o_proj` (~L1352), MLP down (~L1648).

### 2.2 Component A (traced quant) — ✅ ALREADY N/A
NInfer decode GEMVs (`multi_gpu::tp_gemv`, `tp_kernel.cu`) consume **pre-quantized weights**
(Q4G64/Q5G64/Q6G64/W8G32, fp16 scales) with bf16 activations. There is **no activation
quantization kernel anywhere in the decode path** — radiance's 14,288-launch quant class simply
does not exist here. Norm+control-projection is also already one op
(`Variant::gdn_norm_control_projection` → `ops::gdn_norm_gating_proj`,
`variant_kernels.cpp` L362), and the GDN input projection writes q/k/v/z directly into one
combined buffer from the GEMVs (`gdn_input_projection_tp`) — the NInfer analogue of their
`in_proj` merge (doc 44 §1.1 "verify first" item resolves as: **already merged, done**).

### 2.3 Component B (fp8 residual stream) — ✅ N/A at decode M
Residual is already folded into the AR kernel (2.1). Decode residual traffic is 5120×2 B =
10 KB per op; halving it is bandwidth noise at M=1. Only reconsider if a future
high-concurrency decode (M≫12, NCCL fallback regime) makes it material.

### 2.4 What is left: launch count in the data-movement / small-op classes
Per-layer decode launch census (TP2, T=1), counted from the code:

| Stage (per layer) | Launches | Notes |
|---|---|---|
| GDN control (norm+GEMV) | 1 | already fused (`gdn_norm_gating_proj`) |
| GDN input projection (qkv+z) | 2 | two GEMVs into one buffer |
| GDN conv+silu | 1 | |
| **GDN q/k/v column extraction** | **3** | `extract_bf16_columns` ×3 — pure data movement, splits the conv output buffer |
| GDN delta-net recurrence | 1 | |
| GDN gated rmsnorm | 1 | |
| GDN out GEMV | 1 | |
| GDN AR+residual | 1 | fused one-shot (2.1) |
| attn: rmsnorm, QKV GEMV, q/k rmsnorm ×2, rope, gqa, sigmoid | 7 | |
| attn: o GEMV + AR+residual | 2 | |
| MLP: rmsnorm, gate_up GEMV, silu_mul, down GEMV, AR+residual | 5 | |

≈ **~970 launches/step** estimated (48 GDN + 16 attn + 64 mlp + MTP module ~20 + final).
Radiance's win took 1477 → ~1080 in a stack where ~14k/step quants dominated; NInfer is already
in the same ~1k regime with the two big classes gone. The concrete remaining launch-gap items:

1. **`extract_bf16_columns` ×3 per GDN layer = 144 launches/step** (48 layers) of pure
   buffer-splitting between conv and delta-net. This is the cleanest NInfer analogue of their
   "kill standalone launches" win, and it is bit-identical by construction.
2. **MTP module tail (TP path)**: `allreduce_local_bf16` is called *without* the residual, then
   `tp_axpy_bf16` is launched separately (×2 per step: `o_proj` and post-mixer,
   text_context_impl.h `mtp_forward_tail` TP branch ~L570/L585). Passing `x.data` as the
   residual arg folds 2 launches away, bit-identical. Also in scope: stem's two rmsnorms +
   `mtp_pack_fc_input` (3 small launches that could be one).
3. **Verify-phase (T>1) unpacks**: `tp_unpack_qkv_strided`, `tp_unpack_gbeta_strided`,
   `tp_unpack_split_2` add launches at MTP verify width. Lower priority (same class, more
   shapes); revisit after the T=1 items if W0 says the verify phase is launch-gap-bound.
4. **Skinny GEMV shapes** (doc 44 §1.2, separate item): the GDN control projection and other
   small (N,K) GEMVs at decode M may run underfilled. Audit, don't assume — radiance flagged
   one shape at 28.5 µs → 3.6 µs with split-K, and the audit (launch/shape dump) is the
   deciding measurement.

## 3. Work items (ordered)

| ID | Item | Effort | Risk | Gate |
|---|---|---|---|---|
| **W0** | **Baseline**: count actual launches/decode-step + ms/step (nsys, or a CUPTI/launch counter), per-class breakdown (GEMV / conv / delta / norm / extract / AR / MTP). Replaces the ~970 estimate in 2.4 with a number. | LOW | none | recorded in results/ |
| **W1** | **Fuse the extract trio**: make `causal_conv1d_silu` write q/k/v directly into their destination buffers (epilogue scatter), or fold the split into the `gated_delta_net` prologue. Both GDN paths (decode T=1 and verify snapshot) + the non-TP prefill path if trivial. −144 launches/step. | MED (kernel + both GDN variants + tests) | none (data movement only) | bit-identical greedy completions (phase-gate STATE-MATCH per doc 44 §10); ms/step Δ vs W0 build |
| **W2** | **MTP-module small-op fold**: (a) pass residual into the 2 MTP-tail ARs (drop the 2 axpys); (b) stem: fold 2×rmsnorm+`mtp_pack_fc_input` toward one kernel if W0 shows it worth it. | LOW-MED | none (a is exact) | bit-identical; ms/step Δ |
| **W3** | **Skinny GEMV audit** (doc 44 §1.2): dump (N,K) of every GEMV at decode M; flag shapes under SM fill; split-K only where the dump justifies (radiance's kernel *declines* everything else). | LOW (audit) / MED (kernel) | low | bench per flagged shape, then ms/step |
| W4 | *(contingent on W0)* extend one-shot AR coverage / verify-T unpack folds if the verify phase shows launch-gap. | TBD | low | same |
| **W5** | **Dynamic draft depth: per-step confidence-product early chain break** (doc 44 §4.2). See §6. | MED | low (lossless by construction) | off-flag byte-identity; on: greedy byte-identical to full-rk, sampling token-diff + acceptance stats |

Explicitly **out of scope** (and why):
- Quant fusion / fp8 residual — components A/B don't apply (2.2/2.3).
- AR/GEMM overlap — radiance's documented **negative result** (doc 44 §5.2): spin-wait AR on
  the same device contended and lost −3.5…−5.6%. Do not attempt.
- Absolute 22 ms as a target — their hardware + MXFP4 body + DFlash2 drafter; not comparable.

## 4. Verification & sequencing (radiance discipline, per doc 44 §8/§10)
1. W0 first — no change lands without a launches + ms/step baseline to attribute against.
2. Land W1, W2, W3 one at a time; each measured against the build immediately before.
3. Exact fusions (W1/W2a) get **bit-identical greedy completions** — the phase-gate B-chain
   (doc 44 §10) gives the STATE-MATCH certificate and pinpoints first-divergent phase if a
   "bit-identical" change isn't. Run the flake harness (×10) on anything touching
   stream/scheduling structure.
4. Numerical item (W3 split-K) gets ppl + GSM8K paired sign test on top of the token diff.
5. Report ms/step, not tok/s (acceptance luck swings tok/s ~14% at fixed config).

**Honest expectation:** with components A/C already present, the launch-gap class is a smaller
share of our step than it was of theirs (their +14% was dominated by the 14k quant launches we
don't have; their other line items in the same table ran −0.4%…−2.9%). W1 is the biggest
concrete piece (144/step ≈ 15% of estimated launch count). W0 decides whether W2/W3/W4 are
worth it at all.

## 5. Easy wins: doc 44 §9 low-effort shortlist — triage

Doc 44 §9 lists seven low-effort drop-ins. Four are code-read-only and were triaged here on
2026-09-02; the rest map to items already scoped above.

| Doc 44 item | Effort claimed | Triage result | Status |
|---|---|---|---|
| §6 GDN NaN-pattern audit (10-min code read, no GPU) | LOW | **Done — clean**, see 5.1 | ✅ resolved |
| §1.1 verify GDN `in_proj` already merged (10-min code read) | LOW | **Done — already merged**: `Variant::gdn_input_projection_tp` writes q/k/v/z directly into one combined buffer from the GEMVs (text_context_impl.h `gdn_mix_tp` step 2); control projection has a `FusedGdnControlProjectionPayload` (variant_kernels.cpp L372). No action. | ✅ resolved |
| §1.4 check decode conv+rec is one kernel (10-min code read; −0.4% bit-identical if two) | LOW | **Done — two kernels**: `ops::causal_conv1d_silu` then `ops::gated_delta_net`, with three `extract_bf16_columns` between them (text_context_impl.h `gdn_mix_tp` steps 3–5). The fusion is real but subsumed by W1 (killing the extracts already removes the seam; fusing conv+rec on top is the natural follow-on). | → **folded into W1** |
| §2.4 check current KV dtype (if bf16 → fp8 is the biggest capacity+BW win) | LOW | **Done — already better**: production KV is `KvCacheStorage::KvarnK4V2` (int4 K / int2 V, kvarn packed layout) — sub-8-bit, below their fp8 KV. No action. | ✅ resolved |
| §1.2/§1.5 launch + small-projection audit (measurement only) | LOW | = **W0 + W3** in §3. | open (scoped) |
| §4.2 dynamic draft depth — per-lane early chain break at low confidence | LOW-MED | **Scoped as W5 (§6)**: the adaptive-depth controller (docs/69) + variable per-round `rk` plumbing already exist; the missing piece is the in-chain confidence gate. | open (**W5**) |
| §3.1/§3.2 KV capacity sizing runbook (config-level) | LOW | Serving-shape item (doc 134 tp2 serve integration), not a decode-path win. | open (serving) |

### 5.1 §6 audit: the three radiance `0 * INF = NaN` patterns, checked against our chunked GDN

Radiance's libr4d v0.4.0 shipped NaN on this model family (ppl 653586) from three unguarded
`__expf` overflow sites. Checked each against `src/ops/linear_attention/gated_delta_net/`:

1. **kkt/WY strict-upper-triangle padding rows** (`0 * INF` in the masked triangle, NaN merging
   into live rows through the tile): **guarded**. `prepare_wy_wu.cuh` L570–595 carries an explicit
   "Footgun" comment — the masked products use a **conditional select** (`cond ? a*expf(...) :
   0.0f`) instead of a float mask multiply, precisely because `inf * 0 = NaN`; same pattern in
   `stage_chunk_output.cu` Phase C.
2. **chunk-scan split-form halves** (`e^{g_i-c}·e^{c-g_j}`, cref at midpoint → one half
   +INF past a ~176 span): **not present**. `state_passing.cu/.cuh` has no split-form exps at
   all; the only decays are `exp2_approx((g_C - g_row) * log2e)` with `g_C = g_smem[BT-1]` — the
   chunk's **last** cumsum value, i.e. the reference is the chunk tail, so `g_C - g_row ≤ 0`
   for every row (cumsum monotone decreasing) and the decay is structurally bounded in [0,1].
3. **V' staged in bf16** (needs fp32 headroom): **safe by construction**. `v_new` is bf16 storage
   but holds the **undecayed** value (input-data range, no exp amplification — "STG vnew
   (UNDECAYED)", state_passing.cuh ~L424); the decayed copy is staged in **fp32**
   (`make_float2(v0 * dec_top, ...)` into the float `vd_view`).

Conclusion: our chunked path already knows this bug class and is clean. The §6 item costs no
further time. (The *live-only invariant on padded tiles* remains the one to keep an eye on in any
future GDN kernel work — the same invariant our conditional selects depend on.)

## 6. W5 — dynamic draft depth: in-chain confidence-product early break (doc 44 §4.2)

**Radiance's change** (`radiance_draft.py`, `radiance_draft_gpu.py`, `patch_mtp_loopbreak.py`):
per-slot confidence-product gate — keep drafting while ∏conf ≥ τ (τ = 0.35, policy-tuned) —
plus a verbatim n-gram tail (free when it equals the drafter's own top guess) and a batch-size
draft schedule. All hot-path on-device; **one tiny D2H per slot short-circuits the serial loop**.
Measured: **+5.3% (policy-tuned), mostly at concurrency**. Lossless by construction: every
token still verifies through the unchanged sampler.

**Why it's a partial extension, not a new feature, in NInfer** (verified 2026-09-02):
- The *coarse* form already ships: `MtpAdaptiveController` (`src/runtime/tp2/mtp_adaptive.h`,
  ported from llama.cpp PR #27210, docs/69) adjusts the per-round draft depth `rk` from
  acceptance history (climbs after consecutive full-accepts, drops on accumulated misses),
  active in `tp2_backend.cpp` (L964–975). What it lacks is the *in-chain* signal: a round at
  rk=7 still drafts all 7 tokens even when the chain is obviously dead at token 2.
- The *variable-width* plumbing already exists and is battle-tested: `rk` varies per round
  today (adaptive + ngram-mod both change it), `extents` is written per round (L1705), verify
  inputs use per-round self-describing views `verify_ids_r`/`verify_pos_r`/`verify_hidden_r`
  (L1724–1726), and the D-21 canary (docs/119 §11) bounds rk ≤ 7. A shortened round is
  structurally identical to an ngram-shortened round.
- The serial draft chain is at `tp2_backend.cpp` L1468–1498: `mtp_forward_decode_batch` →
  `launch_draft_head` → `allreduce_argmax` per step, fully device-resident (no D2H between
  steps — the loop is `rk-1` max steps, so the break must come from a per-step decision).

**Design (follows radiance's mechanism):**
1. **Confidence source.** After each `launch_draft_head`, the draft-head logits
   (`st.proposal_logits`, draft-vocab space) are on-device. Extract the draft token's
   confidence: extend the argmax pass or add one tiny kernel to emit the top-1 logit (or
   top-1/top-2 margin) into a device scalar per step. (Exact formula to be pinned by the
   sweep in step 4 — radiance's "confidence" is the drafter's top-1 softmax probability.)
2. **Gate + break.** Device accumulator `prod_conf *= conf` per step; host breaks the loop
   when `prod_conf < τ` via the one small D2H per step (radiance's accepted cost — a saved
   `mtp_forward_decode_batch` step is far larger than the D2H). Remaining draft slots are
   zeroed, `extents` is written with the actual count, verify round proceeds at the shorter
   width through the existing per-round views. Steps already launched cannot be un-launched,
   so the break saves *future* steps in the chain (chain length ≤ 6 at rk=7), not the current
   one.
3. **Controller interaction.** Feed the adaptive controller `update(n_draft_actual,
   n_accepted, ...)` with the *shortened* n_draft so climb/drop pressure sees the true
   proposal length. Ngram-mod path (`ngram_used`) bypasses the serial chain — the gate
   applies only to chain-drafted rounds. The batch-size schedule (their 1:8/2:7/4:6/8:5) is
   N/A for single-request serve; revisit for the lanes path.
4. **τ sweep.** Radiance's τ=0.35 is *policy-tuned on their weighted mix* — their explicit
   lesson (doc 44 §4.3): tune on a mix, one content class picks the wrong default. Sweep
   τ ∈ {0.2, 0.3, 0.35, 0.5} on the standard mix workload; default ships at the mix-optimum
   and the feature is **off by default** until then (env/flag, byte-identical when off).
5. **Lanes (batched serve) — phase 2.** The multi-lane path (`tp2_batched_decode`, lanes with
   independent chains; the skip-draft-continue `!any_draft` termination from `93fda35e` lives
   here) is where radiance's win was "mostly" (concurrency). Per-lane `prod_conf`/break, same
   mechanism. Do single-request serve first: it's the simpler gate (byte-identity infra is
   there) and the lanes path carries the known 256-token shrink-phase regression (docs/107) —
   don't stack a new knob on an open regression.

**Losslessness argument (the gate's backbone):** a shortened proposal is a *prefix* of a
full proposal. The target verifies causally — for greedy, every verified token and the bonus
token depend only on the causal context, which is identical; GDN/KV state advances by the
accepted count, which is identical. So **greedy token stream is byte-identical to the
full-rk run** (the cheap, strong gate). For sampling, standard spec-decode correction
preserves the target distribution (token-diff + acceptance statistics, per doc 44 §8).

**Effort:** MED — one small kernel (confidence), ~1 day of plumbing in the serve loop
(loop break + extents + controller feed + flag), sweep + gates. The hard parts
(variable verify width, D-21 bounds, per-round views, graph capture at variable T) are all
already done by docs/69 + docs/119.
