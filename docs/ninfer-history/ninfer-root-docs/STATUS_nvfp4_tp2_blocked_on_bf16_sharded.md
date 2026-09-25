# wo-nvfp4-prefill — STATUS: the decisive cell is blocked on BF16-sharded weight support in materialize_tp

**Date:** 2026-09-08 · owner: A1 lane (reviving the parked subagent work) · branch `wo/nvfp4-prefill`

## Where this stands

1. Artifact verified: `~/ninfer/incoming/nvfp4/qwen3_8_27b_nvfp4.ninfer` (18,324,067,840 B, SHA OK, HF Ostfralla/Qwen3.8-27B-NVFP4-NInfer). Identity `qwen3.8-27b / nvfp4`. The bundled HF patch is identity-gate only (already merged).
2. Load guard opened (NINFER_ALLOW_NVFP4_TP2=1, fail-closed, default-off) — tp2_backend.cpp in this branch.
3. Three loader fixes applied to `bindings.cpp` `bind_qwen38_nvfp4_text_layers` (ported from gzenz/ninfer's proven binder):
   - GDN control projection: fused **or split** via `binder.has_tensor` (artifact ships split a/b [48,5120] BF16)
   - GDN `query_key_value_z` + GDN `output`: `bind_nvfp4_weight` + their `input/output_scale_divisor` sidecars
   - MLP gate_up/down: unconditional `bind_nvfp4_weight` + divisors (the old `layer < 56` FP8 tail doesn't match this export — all 64 layers are NVFP4)
   - Attention `query_key_gate_value` + `output`: **per-layer mixed** — BF16 on layers {3,7,11,19,23}, NVFP4+divisor on the rest (verified per-layer from the artifact manifest)
4. Binder now completes: `1307 objects, 16840 MB device total`.

## THE BLOCKER (precise)

`materialize_tp` (`tp_load.cpp`) sizes/materializes every **sharded** tensor through
`row_split_geometry(t->format, t->shape)` → `quant_geometry(format)`, which supports ONLY
grouped-quant formats (Q4G64/Q5G64/Q6G64/W8G32). The NVFP4 artifact's **attention linears are BF16
on layers {3,7,11,19,23}** (verified per-layer from the manifest) and are TpRole-sharded →
`std::runtime_error("row-split-k128-v1 requires a grouped quantized format")` during pass-1 sizing,
before any H2D.

gzenz upstream never hits this: their NVFP4 path is single-GPU (no `tp_world`, plain contiguous
materialization — BF16 tensors just copy).

## What unblocks it (design, ~a lane of work)

**Load-time requant (recommended):** in `materialize_tp`, for sharded tensors whose source format is
BF16, dequantize-copy → requantize into **W8G32** (group scales; quantizer kernels already exist in
`src/ops`) per rank shard. Runtime sees a grouped weight it already consumes — **no runtime GEMM
dispatch changes needed** for the BF16-attention layers. W8 error is negligible for a handful of
attention linears. Alternative (requiring runtime work): true BF16 Weight + a BF16 GEMM path in the
TP2 attention runtime — larger, and pointless if W8 requant is numerically fine.

Pass-1 sizing change is mechanical: sharded && BF16 → `bytes = loc.rows * loc.cols * 2`.

## Verification after unblock

1. Materialization completes both ranks; serve listens on 8192-ctx int8-KV (the original granted cell).
2. Decode-guard cells 10k / 40k / 80k, kvarn_k4v4 KV, 1 iter (`DG_SPECS` per docs/157) — compare vs
   the groupwise-int rows in results/157_p2 + the kvarn decode guard (80.2/56.3 t/s @10k/160k).
3. **The decisive cell** (docs/157 §13.2.2): 25k prefill, same config as the groupwise baseline
   (1035 t/s). If NVFP4 weights move 25k prefill to ~1,400+ t/s, the sglang gap is the weight format
   (ship FP4 weights); if not, it's bounded GEMM work (1.2–1.5×, docs/157 §13.2.4).
