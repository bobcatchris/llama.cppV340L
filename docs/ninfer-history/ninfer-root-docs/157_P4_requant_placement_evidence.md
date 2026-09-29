# docs/157 P4 — BF16->W8G32 requant + NVFP4 TP-shard placement: evidence & semantics

**Author:** agent2 · **Date:** 2026-09-08 · **Branch:** `wo/bf16-w8-requant` (base `9b9efefb` + A1 tags
`39bfa3c6` + A1 runtime `02089ed4` + placement `5563465b`)
**Coordinators:** design §6 `results/157_MISSION_STATE_and_handoff.md`; ruling chain 19:19-20:18Z.

## 1. Why W8G32 requant (not BF16 passthrough)
`src/ops/linear/bf16/bf16_dispatch.cpp:12-13` hard-admits EXACTLY two problems:
`{n=14336,k=5120}` and `{n=5120,k=6144}` — FULL shapes. The TP2 shard shapes
(7168x5120 attn-qkv local, 5120x3072 attn-out local, 5120x8704 mlp-down local) all throw
`unsupported shape`. A1 confirmed by runtime census and closed the gap by registering the five
LOCAL problems for NVFP4 (02089ed4) — but for the BF16-sourced linears there is no bf16 local
registration, and §6 rejected adding one: W8G32 planes ride the PROVEN w8 rowsplit kernels
(all M classes, w8_dispatch) with zero runtime changes.

## 2. Exporter semantics pinned (condition (b) finding)
Source of truth: `tools/convert/common/quantize.py::_canonical_scale_words + quantize_matrix`:
- scale: `f16(f32(f64(amax_f32)/127.0))` — RNE at every step; `amax` computed on f32-decoded bf16
- underflow: f16 scale == 0 with amax > 0 -> bit pattern 0x0001 (= 2^-24), applied BEFORE the guard
- guard: after floor, non-finite (f16 inf) scale with amax > 0 -> **ValueError** (i.e. raises)
- reciprocal: `f32(1/f64(scale_f16))`; codes: `RNE(x_f32 * recip)` clamped [-127,127], int8
  two's complement; W8G32: group 32, symmetric, no high plane.

**FINDING (logged as accepted):** host `__float2half` **saturates** at 0x7BFF for overflow
input instead of producing f16 inf, so the bit-pattern guard never fires. The writer tests the
f32 magnitude directly (`raw_scale > 65504`), which is CONSERVATIVE vs numpy's exact RNE
boundary (65520): in the band (65504, 65520] the exporter would store 65504 and we throw.
Unreachable for real weights (needs group amax > 8.3e6; attention linears max ~10^1);
fail-closed = refuse service, never store garbage. Test row `overflow_throws` pins the behavior.

## 3. Wall-3 attribution correction (coordinator-ordered)
The STATUS doc (`docs/STATUS_nvfp4_tp2_blocked_on_bf16_sharded.md`) attributed the pass-1 death
to BF16 attention tensors. Manifest order says **NVFP4 objects precede them** (obj#13
`gdn/query_key_value_z`, #19 `mlp/gate_up`, #21 `mlp/down` on layer 0 vs BF16 qkv at obj#56),
and the generic sharded branch calls `row_split_geometry -> quant_geometry`, which throws for
NVFP4 too. The observed first death was NVFP4, not BF16. Fixed here by placing sharded NVFP4
via `nvfp4_shard_image` (blockscale planes at local dims, divisor verbatim) — same branch also
keeps BF16 on the W8 route. Fix at merge: correct the STATUS doc's §"THE BLOCKER" wording.

## 4. Placement contract vs A1's registered problems — FIT 5/5
| A1 problem (02089ed4) | dims | my pass-1 local dims (role) | fit |
|---|---|---|---|
| AttnInputL   | 7168x5120  | MultiRangeQKV 14336/2    | ✓ |
| GdnInputL    | 8192x5120  | MultiRangeGQKV 4-range [q_l;k_l;v_l;z_l] | ✓ (A1 review fix 20:25Z: contiguous half WRONG — q/k would all land on rank0; ranges q=[r*1024,1024) k=[2048+r*1024,1024) v=[4096+r*3072,3072) z=[10240+r*3072,3072); pair-audited: disjoint + full coverage + byte-exact assembly vs fused source) |
| MlpGateUpL   | 17408x5120 | MultiRangeGateUp ranges  | ✓ |
| Residual6144L| 5120x3072  | attn-out AND gdn-out RowK| ✓ |
| Residual17408L| 5120x8704 | mlp-down RowK            | ✓ |
All local N multiples of 128, local K multiples of 64; weight_divisor 4 B copied verbatim to the
LOCAL `weight_divisor_offset`; scales re-swizzled per `layouts.py:swizzle_nvfp4_scales`
(`n = m*128 + a*32 + b`, `j = kt*4 + c` — four 16-groups per 64-wide K-tile; a first
implementation using `j = kt*16 + c` was caught by the fixture on byte 16900).

## 5. Test evidence (all CPU-only, zero card contact)
- `ninfer_bf16_requant_host_test`: 15 rows byte-exact vs numpy re-impl of quantize.py
  (zeros, ±clamp both ends, two RNE-tie rows, subnormal-floor pair, overflow-throw, 6 random). PASS.
- `ninfer_nvfp4_shard_host_test`: 11 placement cases byte-exact vs **the real exporter**
  (`tools/artifact/layouts.py::encode_nvfp4` imported by the generator; expected images are
  exporter outputs on natural slices => rank-sum reassembly == single-GPU image holds by
  construction) + 4 fail-closed contract rejections (rows%128, col0%64, cols>full, col0+cols) +
  explicit divisor-verbatim check + gqkv4-pair assembly audit (every fused source row owned
  exactly once across ranks, codes+unswizzled scales byte-equal to source). PASS.
- Regression subset (loader-adjacent host suites): weight_shard, artifact_reader,
  artifact_materialization, qwen3_6_27b_load_plan (device-skipped as designed), tp2_budget,
  serve_options, host_kv_arena, public_api — **10/10 PASS** on the merged tree
  (02089ed4+5563465b).
- Groupwise byte-for-byte: structural — both new pass-1/pass-2 branches are format-gated
  (BF16/NVFP4 sharded); the groupwise artifact contains no such tensors (A1 census: 0 BF16
  sharded, 0 NVFP4). GdnConv stays on its own raw branch.

## 6. Ops notes
- ctest full-tree build pruned post-run (16G of 230 MB static binaries) per the disk standing
  rule; subset battery is the repeatable form for this branch.
- GPU gates that remain genuinely open (cannot be closed on CPU): real 1307-object NVFP4 load,
  decode guards 10k/40k/80k, decisive 25k prefill vs 1,035 t/s, ln_mma/ln_tma routing grep
  (WO-G3 cells 1-4, Gemini harness `run_wo_g3.sh`, BINARY = this worktree's serve build).
