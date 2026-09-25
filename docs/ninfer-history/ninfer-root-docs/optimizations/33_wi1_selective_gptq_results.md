# 33 — WI-1 Selective GPTQ INT4: negative result, infrastructure delivered

**Date:** 2026-08-21
**Executed by:** assistant (took over WI-1 from agent, per user instruction)
**Model:** qwen3_8_27b (3.8, target) · 2× RTX 5060 Ti · TP2 + MTP k=3
**Verdict:** value_z scope is **net negative** (−4.1% t/s). Closed per doc-32's
2-attempt budget. All calibration/quantization infrastructure now works and is
committed for future scopes.

## 1. Scope and method (per doc 32 WI-1)

- Quant `text/layers/{4..59}/gdn/value_z` (48 GDN layers) Q5G64 → Q4G64.
- **GPTQ with real Hessian**: the GEMV input is `h = RMSNorm(x)` (confirmed in
  `gdn_norm_control_projection` → `gdn_norm_gating_proj`), not raw `x`.
- Calibration: 8 diverse prompts × ~251 tokens, 1024 tokens/layer (avg),
  dumped from a `--cal-dump` engine hook. `h` is [5120, T] bf16.

## 2. Two quantizer bugs found and fixed (pre-existing, never exercised)

All prior "GPTQ" artifacts (doc 19, doc 25) called `quantize()` **without
`add_batch`** → silent RTN fallback. This run was the first to exercise the
Hessian path, which had two bugs:

1. **Wrong Cholesky target.** The update coefficients must equal the
   Schur-complement rows `H_active[j, j+1:]/H_active[j, j]`. Identity
   (verified to 4.5e-8): `R[j, j+1:]/R[j, j]` with `R = cholesky(H, upper=True)`.
   The old code used `chol(H⁻¹)` (wrong coefficients; verified on 2×2: `b/c`
   instead of `b/a`). Fix: one direct `torch.linalg.cholesky(h, upper=True)`.
2. **Codes vs weights in the tail.** The Hessian loop stored integer codes in
   `q`, but the canonical tail computes group scales from `|q|` and re-rounds
   `q/scale` — with codes, scales are derived from |code|≤7 and the final
   magnitudes are wrong by a random 1–4×. Model output was garbage from token 1.
   Fix: store dequantized values (`q_col * raw_scale`).

**Verification:** dense brute-force Schur reference GPTQ (implemented
independently) vs NInfer quantizer: output SNR 13.45 vs 13.36 dB — match.

## 3. Results (battery, deterministic)

| Metric | Baseline (94.64) | GPTQ value_z | RTN value_z (control) |
|---|---|---|---|
| MTP k=3 t/s | 94.64 | 90.78 (−4.1%) | 90.74 (−4.1%) |
| Acceptance | 85.6% | 79.9% (−5.7 pts) | 79.9% (−5.7 pts) |
| Verify T=4 | 34.65 ms | 34.35 (−0.9%) | 34.36 (−0.8%) |
| Round | 37.81 ms | 37.51 (−0.8%) | 37.53 (−0.7%) |
| Determinism / A2 / draft vocab | PASS | PASS | PASS |

- **GPTQ ≈ RTN exactly** → the Hessian (1024 tokens, rank ≤ 1024 < 5120 cols)
  buys nothing over RTN at this calibration volume for this layer set.
- **Acceptance cost is the quantization itself, not the method**: Q4 value_z
  perturbs draft quality ~5.7 pts. Mechanism: value_z feeds every GDN layer's
  recurrent state; small per-layer perturbation accumulates over 64 layers and
  shifts the target-vs-draft distribution mismatch.
- Net: −5.0% tok/round vs −0.8% round → **−4.1% t/s**. Net loss.
- Knowledge probe on the GPTQ artifact: fully coherent (27×43=1161 ✓, Frank
  Herbert ✓) — the model is not broken, it is just draft-deteriorated.

## 4. Consistency with doc 25

Doc 25 (all-layers, actually-RTN): −11.8 pts acceptance, +13.2% verify, net
≈ break-even. This run: 15% of the GEMV bytes quantized → −5.7 pts (≈ 48% of
the full-scope cost) for −0.8% verify (≈ 6% of the full-scope gain). The
acceptance/gain ratio is *worse* for partial scope — selective quantization
concentrates the draft-quality cost without capturing the full bandwidth
gain. **Do not retry partial scopes.**

## 5. What this delivers (usable infrastructure)

- `src/core/multi_gpu/cal_dump.h` + engine hook (gated, zero-cost when null)
- Driver `--cal-dump DIR` (rank-0-guarded D2H, no protocol impact)
- `tools/convert/qwen3_8_27b/convert_selective_gptq.py` (strict: hard-exits
  without calibration, no silent RTN fallback)
- **Fixed** `tools/convert/common/gptq.py` Hessian path (now genuinely GPTQ)
- Battery fix: baseline auto-update now requires PASS (was fails==0, so WARN
  overwrote good baselines — happened once this session)
- `/home/intel/knowledge_probe.sh` (4 factual coherence probes)
- Artifacts: `/home/intel/models/qwen3_8_27b_selq4_gptq.ninfer` (kept for
  future full-scope experiments)

## 6. If INT4 is ever revisited

1. Full scope (mlp gate_up + value_z) — the only scope where the verify gain
   (−25% GEMV) can beat the acceptance cost (−12 pts per doc 25).
2. Much more calibration: ≥ 8k–64k tokens (GPTQ paper uses ~10⁵; Hessian rank
   needs to cover 5120 cols). `--cal-dump` supports arbitrary prompt counts.
3. Gated by: acceptance ≥ 85.0%, t/s ≥ baseline+5%, verify ≤ 28 ms.

## 7. Next

WI-1 closed negative. Remaining work per doc 32: WI-4 (pp tiling) and WI-5
(V340L prep). Baseline restored to 94.64 t/s.
