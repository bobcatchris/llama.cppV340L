# W5 prefill desk - shape census + rocBLAS route survey (zero-GPU phase)

Date: 2026-09-21. Branch amd/w5-prefill-gemm (worktree wt-prefill-gemm). No GPU touched:
dies 0-2 are Gemini's served battery, die 3 is held by the W2 decode-atom desk. Harness
authored and compile-validated only; first run deferred until W2 releases the cell
(ledger E-012).

## 1. Geometry (read from the GGUF header + src/models/qwen35.cpp, not assumed)

Qwen3.8-27B-ASCII-P1M.gguf: n_embd 5120, n_ff 17408, 65 blocks = 64 trunk + 1 nextn
draft; 48 GDN layers + 16 full-attention (full_attention_interval 4, blocks 3,7,...,63);
attention 24 q heads x 256 + fused q-gate (attn_q is 5120x12288 = q 6144 + gate 6144 in
one tensor, src/models/qwen35.cpp:267-289), 4 kv heads x 256 (k/v 5120x1024 each), wo
6144x5120; GDN attn_qkv 5120x10240 (key_dim 2048 x2 + value_dim 6144), attn_gate
5120x6144, ssm_out 6144x5120; FFN gate/up 5120x17408 + down 17408x5120 every layer;
vocab 129272 (output.weight 5120x129272, q6_K, last-token only -> MMVQ not GEMM).

## 2. Shape census, trunk, ubatch M=128, TP3 row-split

Split law: weights split along ne[1] (output rows); boundaries at nrows*tensor_split[i]
rounded DOWN to 128-row multiples - rounding = get_mmq_y_host(gfx900) = 128
(ggml/src/ggml-cuda/mmq.cuh:142-144), applied at ggml/src/ggml-cuda/ggml-cuda.cu:1898-1914
and 872-885. Canonical equal-thirds shares below; runtime shares wobble by one 128-row
block per boundary because tensor_split comes from free-memory fractions at load.

  shape        K     N      calls/pass  per-die N (die0/1/2)  agg GFLOP  share
  ffn_gate     5120  17408  64          5760/5760/5888        1460.3     23.4%
  ffn_up       5120  17408  64          5760/5760/5888        1460.3     23.4%
  ffn_down     17408 5120   64          1664/1664/1792        1460.3     23.4%
  gdn_qkv      5120  10240  48          3328/3456/3456         644.2     10.3%
  ssm_out      6144  5120   48          1664/1664/1792         386.5      6.2%
  gdn_gate     5120  6144   48          2048/2048/2048         386.5      6.2%
  attn_q+gate  5120  12288  16          4096/4096/4096         257.7      4.1%
  attn_out     6144  5120   16          1664/1664/1792         128.8      2.1%
  attn_k       5120  1024   16          256/384/384             21.5      0.3%
  attn_v       5120  1024   16          256/384/384             21.5      0.3%
  ssm_a/beta   5120  48     48          0/0/48 (last die only)   3.0     0.05%

Trunk total 6.23 TFLOP/ubatch = 48.7 GFLOP/token over 3 dies, ~2.04 TFLOP/die. FFN trio
= 70.3% of GEMM FLOPs; top 5 = ffn_gate, ffn_up, ffn_down, gdn_qkv, ssm_out/gdn_gate.
At the pp 93.87 t/s baseline (16 ubatches at 2k, 1.365 s/ubatch, 67.4% GEMM share) the
implied rocBLAS rate is ~2.0-2.2 TF/s/die vs the 10.75-12.5 fp32-class and ~21.5
packed-fp16 ceilings. Per-call weight dequant q->f16 writes 2*N_d*K bytes (~16 GB/die/
ubatch of f16 traffic across all shapes) plus a quantized read of every weight - the
route tax a quantized-input custom tile removes. Attention/GDN flash+scan kernels are
NOT in this table (24.4% flash-attn share stays W5b).

## 3. Shipped dispatch today (file:line, what the W5 kernel must slot behind)

- ggml_cuda_mul_mat (ggml-cuda.cu:2541): split-src0 disables the batched-cublas fast
  path (2608-2621 all gated on !split); quantized weights at M=128 > MMVQ_MAX_BATCH_SIZE
  = 8 (mmvq.cuh:3) kill mul_mat_vec_q (2554-2556); MMQ is disabled for gfx900 dense:
  ggml_cuda_should_use_mmq (mmq.cu:267) falls through to the explicit Vega rule
  "gfx900 lacks native dp4a, loses to dequant + hipBLAS for dense matrices; keep MMQ
  only for MoE" -> return n_experts > 0 (mmq.cu:371-376). Net: every dense prefill GEMM
  takes the final else (ggml-cuda.cu:2629) -> ggml_cuda_op_mul_mat(.., cublas, nullptr).
- Per call per die (ggml-cuda.cu): f32 activations are peer-copied to each die
  (2050-2051), converted f32->f16 (1703-1711), and the ENTIRE quantized weight slice is
  dequantized to f16 (1694-1699) before gemm; on gfx900 use_fp16 holds (fast_fp16:
  common.cuh:310; GGML_PREC_DEFAULT is the mul_mat default, ggml.h:437) and the branch
  taken is the f16-accumulate/f16-out hipblasGemmEx + f16->f32 convert (1732-1748);
  GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1 flips it to f32-acc/f32-out (1717-1731) - both
  variants benched by the harness.
- Env/compile switches that exist today: GGML_CUDA_FORCE_CUBLAS / FORCE_MMQ are
  COMPILE-time defines (mmq.cu:268-270, 315-317; ggml-hip/CMakeLists.txt:97-103) - not
  runtime env; runtime env exists only for compute type (ggml-cuda.cu:1600-1626,
  static-once getenv lambda = the pattern a W5 route switch should copy).
- Custom-tile insertion points (pick one): (a) a new runtime-env-gated branch inside
  ggml_cuda_mul_mat just before ggml-cuda.cu:2629's cublas fallback (split path already
  hands each die its row range dev[id].row_low/high, 1898-1914, and dst scatter at
  2081-2091 needs no change - row-split outputs are DISJOINT slices, no partial sum);
  (b) flip the mmq.cu:371-376 gfx900 rule behind the same env to reuse the MMQ plumbing
  (heavier; gfx900 MMQ kernels are dp4a-class, likely wrong class for fp16x2 tiles).
  Option (a) is the W5 plan: kernel reads quantized weights + q8/f16 activations
  directly, v_pk_fma_f16 math, per GEMM_CONSTRAINED_TILES_2026-09-18 + PREFILL_1K_PLAN
  G1 constraint set - note that doc's own caveat: the fp32@M=128 kills do NOT transfer
  to packed-fp16 MAC density blindly (re-derive at M=128/256/512).

## 4. Harness

docs/amd-port/tests/bench_gemm_gfx900.cu - standalone hipcc/rocblas microbench of the
census shapes at M=128/256/512, per-die row shares, both shipped compute modes (h16 =
shipped, f32 = FORCE_CUBLAS_COMPUTE_32F analog), TF/s vs 12.5/21.5 ceilings and the
5.4 TF/s W5 gate, call-weighted per-die aggregate. Build line validated against
ggml/src/ggml-hip/CMakeLists.txt (hip/hipblas/rocblas from /opt/rocm-6.2.0; headers in
include/rocblas/, include/hipblas/; libs lib/librocblas.so, lib/libhipblas.so) and
compile-checked clean for gfx900 with the pinned toolchain (HIP 6.2.41133, amdclang 18).
NOT RUN - no GPU. Run line: HIP_VISIBLE_DEVICES=3 /tmp/bench_gemm [m] [reps].

## 5. What changes in the W5 plan

1. The bench matrix is the FFN trio, not "K=5120/17408 x N=per-die share" in general:
   17408-wide gate/up split 5760/5760/5888 and ffn_down/ssm_out/attn_out share the
   17408x5120 / 6144x5120 classes; a custom tile tuned for N_d in {1664..1792} x
   K in {5120, 6144, 17408} + N_d ~5.8k x K=5120 covers 76.5% of GEMM FLOPs.
2. ssm_a/beta (5120x48) land entirely on the last die at N<128 rounding - harmless
   (0.05%) but a reminder the rounding law creates lopsided tiny GEMMs.
3. Both rocBLAS compute modes are live dispatch branches; the f16-out shipped mode adds
   a whole-output f16->f32 convert pass per GEMM on top of the GEMM itself - the f32-out
   mode may win even on rocBLAS alone (harness measures it for free).
4. The dequant-per-call tax (~16 GB/die/ubatch f16 writes) is charged to the GEMM share
   in-server; a rocBLAS-only harness will therefore overstate the in-server GEMM rate -
   compare harness TF/s against the implied ~2.0-2.2 TF/s/die with that in mind.
5. MMQ is compile-blocked, not env-blocked, on gfx900 - any W5 A/B arm must be a
   runtime-env switch on a fresh branch point (option (a) above), matching the
   static-once getenv pattern at ggml-cuda.cu:1600-1626.
