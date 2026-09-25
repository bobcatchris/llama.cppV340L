#pragma once

#include "common.cuh"

// Packed-fp16 GEMM tile, arm a8 from docs/amd-port/tests/bench_tile_gfx900.cu
// (oracle-gated, W5 gate met: 5.53 TF/s/die call-weighted vs 4.62 rb16, E-016/E-022).
// Ported verbatim from the harness with one change: the dst row stride (ldc) is
// decoupled from the weight N, since the split-mul_mat main device writes into a
// strided slice of the full dst.
//
// C[m][n] = sum_k X[m][k] * W[n][k], f16 in, half2 k-pair accumulation, split-K in z
// with a fixed-order f32 slice reduce. Weights are the f16 buffer the shipped cublas
// route produces by dequant per call; f32 out, no trailing convert pass.
// Numerics are NOT the shipped h16/f32 class: the arm is numerics-changing and must
// pass the served byte-difference protocol (greedy determinism within the arm).
//
// Kernels are defined in tile-gemm.cu (the non-template reduce has strong linkage,
// so this header stays device-code free for the TUs that only call the glue).

// a8 geometry
#define TILE_FP16_MT 128
#define TILE_FP16_NT 64
#define TILE_FP16_TY 4
#define TILE_FP16_TX 8
#define TILE_FP16_KC 32

// tile route glue: eligibility whitelist + launch, defined in tile-gemm.cu.
// Returns false (no side effects) when the env is unset or the shape is off the
// census whitelist; the caller then runs the shipped branches unchanged.
bool ggml_cuda_tile_fp16_mul_mat(
        ggml_backend_cuda_context & ctx, int id,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
        const char * src0_dd_i, const float * src1_ddf_i, float * dst_dd_i,
        int64_t row_diff, int64_t src1_ncols, int64_t ldc, cudaStream_t stream);
