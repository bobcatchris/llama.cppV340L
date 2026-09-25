// W2 decode-atom cell v2 (oracle-gated): iq3_s MMVQ decode-atom variants for
// gfx900 (Vega 10, wave64, emulated dp4a), die 3 dev cell.
//
// Attacks the per-lane dependent decode chain named by the pure-consume
// ablation (E-006): same-geometry loads 245 GB/s -> +8 constant LUT lookups
// 178 -> full decode 105. Arms, one per rung, all in ONE binary (same-BIN):
//   base   : verbatim vec_dot_iq3_s_q8_1 clone, schedule-cloned from
//            mul_mat_vec_q (GCN, T=1, NW2, kqs=2*(tid&7), 16 kbx slots).
//            v2 fixes two clone bugs found vs mmvq.cu/vecdotq.cuh:
//            u1 index l0+1 (was l0+4) and y offset kby = kbx*(qk/QK8_1).
//   gmem   : same math, grid table in __device__ global memory instead of
//            __constant__ (divergent constant-bank access class test)
//   pip    : base math, intra-call restructure: weight loads, all 8 x ints and
//            all 8 LUT reads issued before sign/dp4a ALU; 2 int accumulators
//            (bit-exact: integer associativity, final int sum unchanged)
//   sgn    : sign-folded combined LUT. Index = qs | qh<<8 | s4<<9 (13 bits,
//            8192 entries, 32 KB), entry bytes pre-negated per the 4 sign bits
//            of each 4-weight code group. Deletes the whole __vcmpne4/
//            __vsub4 sign chain. Bit-exact integer math.
//   sgnpip : sgn table + pip restructure.
//   sexp   : 256-entry (2 KB) pre-expanded sign-mask table: one lookup per
//            sign byte returns both 0x00/0xFF byte masks; vcmpne4 chains go.
//            Bit-exact.
//   dp4a   : hand-rolled dp4a (v_bfe_i32 sign-extending byte extracts +
//            v_mad_i32_i24, 12 vops vs ~16 of the library emulation).
//            Bit-exact.
//   sexpdp : sexp + dp4a combined.
//
// ORACLE: host C++ mirror of the shipped vec_dot_iq3_s_q8_1 over the same real
// weight bytes + synthetic q8_1. Every variant must match on rows 0..63
// (rel err < 1e-4) BEFORE its timing counts; failing variants are refused.
//
// Table fact (documented per task): iq3s_grid has 512 entries (ggml-common.h
// GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)); the kernel indexes with a 9th
// qh bit: idx9 = qs_byte | ((qh >> l) & 1) << 8.
//
// build:
//   /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP \
//     -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//     docs/amd-port/tests/bench_mmvq_gfx900.cu -o /tmp/bench_mmvq \
//     -L /media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin -lggml-base
// run (die 3 only):
//   HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin \
//     /tmp/bench_mmvq <gguf> [niter=200] [reps=3]
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <hip/hip_runtime.h>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------- tables ---

// global-memory copy of the 512-entry grid (gmem arm)
static __device__ uint32_t g_grid32[512];

// sign-folded combined table: idx13 = qs(8) | qh<<8 | s4<<9; entry bytes are
// the grid quad pre-negated per the 4 per-weight sign bits of the code group
static __device__ uint32_t g_sgn_tab[8192];

// pre-expanded sign masks: entry.x = 0x00/0xFF byte mask for the even code,
// entry.y = same for the odd code of one sign byte (sexp arm, 2 KB table)
static __device__ uint2 g_sexp[256];

static void build_sexp(std::vector<uint2> & out) {
    out.resize(256);
    for (int s8 = 0; s8 < 256; ++s8) {
        uint32_t me = 0, mo = 0;
        for (int j = 0; j < 4; ++j) {
            if ((s8 >> j) & 1) me |= 0xFFu << (8*j);        // bits 0..3 -> even-code bytes
            if ((s8 >> (4 + j)) & 1) mo |= 0xFFu << (8*j);  // bits 4..7 -> odd-code bytes
        }
        out[s8] = make_uint2(me, mo);
    }
}

// hand-rolled dp4a: sign-extending byte extract + 24-bit mads (12 vops vs the
// ~16 of the library emulation on gfx900); bit-exact vs ggml_cuda_dp4a.
// Non-volatile asm: lets LLVM hoist the extracts and schedule the mad chain.
static inline __device__ int dp4a_bfe(int a, int b, int c) {
    int a0, a1, a2, a3, b0, b1, b2, b3;
    __asm__("v_bfe_i32 %0, %1, 0, 8"  : "=v"(a0) : "v"(a));
    __asm__("v_bfe_i32 %0, %1, 8, 8"  : "=v"(a1) : "v"(a));
    __asm__("v_bfe_i32 %0, %1, 16, 8" : "=v"(a2) : "v"(a));
    __asm__("v_bfe_i32 %0, %1, 24, 8" : "=v"(a3) : "v"(a));
    __asm__("v_bfe_i32 %0, %1, 0, 8"  : "=v"(b0) : "v"(b));
    __asm__("v_bfe_i32 %0, %1, 8, 8"  : "=v"(b1) : "v"(b));
    __asm__("v_bfe_i32 %0, %1, 16, 8" : "=v"(b2) : "v"(b));
    __asm__("v_bfe_i32 %0, %1, 24, 8" : "=v"(b3) : "v"(b));
    int s0, s1, s2, s3;
    __asm__("v_mad_i32_i24 %0, %1, %2, %3" : "=v"(s0) : "v"(a0), "v"(b0), "v"(c));
    __asm__("v_mad_i32_i24 %0, %1, %2, %3" : "=v"(s1) : "v"(a1), "v"(b1), "v"(s0));
    __asm__("v_mad_i32_i24 %0, %1, %2, %3" : "=v"(s2) : "v"(a2), "v"(b2), "v"(s1));
    __asm__("v_mad_i32_i24 %0, %1, %2, %3" : "=v"(s3) : "v"(a3), "v"(b3), "v"(s2));
    return s3;
}

// pull the __device__ grid table to host memory (HIP cannot register the
// internal static symbol for hipMemcpyFromSymbol)
__global__ static void pull_grid(uint32_t * out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 512) out[i] = iq3s_grid[i];
}

static void build_sgn_tab(const std::vector<uint32_t> & h_grid, std::vector<uint32_t> & out) {
    out.resize(8192);
    for (int idx9 = 0; idx9 < 512; ++idx9) {
        const uint32_t g = h_grid[idx9];
        for (int s4 = 0; s4 < 16; ++s4) {
            uint32_t v = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t b = (g >> (8*j)) & 0xFF;         // odd, 1..15
                if ((s4 >> j) & 1) b = (uint8_t)(-(int8_t)b); // two's-complement negate
                v |= (uint32_t)b << (8*j);
            }
            out[idx9 | (s4 << 9)] = v;
        }
    }
}

// ------------------------------------------------------- device vec_dots ---
// All five atoms consume exactly the same loads; only the decode chain
// differs. iqs = kqs in {0,2,...,14}; bq8_1 already offset by kby = kbx*8.

// base: verbatim vec_dot_iq3_s_q8_1 (vecdotq.cuh, u1 = l0+1)
static __device__ __forceinline__ float vd_base(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// gmem: identical math, grid in __device__ global memory
static __device__ __forceinline__ float vd_gmem(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            g_grid32[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            g_grid32[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// pip: base math, restructured. Load everything (weights, 8 x ints), issue all
// 8 LUT reads, then the sign ALU, then a 2-accumulator dp4a chain. Bit-exact
// vs base: integer associativity only; per-sub-block sumi is the same integer.
static __device__ __forceinline__ float vd_pip(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh  = bq3->qh[iqs/2];
    const int sw  = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * s8 = (const uint8_t *) &sw;
    const uint8_t * ys  = (const uint8_t *) bq8_1[iqs/2].qs;
    int u[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) u[j] = get_int_b4(ys, j);
    int g[8];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int l0 = 2*p;
        g[2*p + 0] = iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        g[2*p + 1] = iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
    }
    int sg[4][2];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        sg[p][0] = __vcmpne4(((s8[p] & 0x03) << 7) | ((s8[p] & 0x0C) << 21), 0x00000000);
        sg[p][1] = __vcmpne4(((s8[p] & 0x30) << 3) | ((s8[p] & 0xC0) << 17), 0x00000000);
    }
    int sa = 0, sb = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        sa = ggml_cuda_dp4a(__vsub4(g[2*p + 0] ^ sg[p][0], sg[p][0]), u[2*p + 0], sa);
        sb = ggml_cuda_dp4a(__vsub4(g[2*p + 1] ^ sg[p][1], sg[p][1]), u[2*p + 1], sb);
    }
    int sumi = sa + sb;
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// sgn: sign-folded 32 KB combined LUT, verbatim schedule otherwise
static __device__ __forceinline__ float vd_sgn(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int sw = get_int_b2(bq3->signs, iqs/2);
    int sumi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int l0 = 2*p;
        const int sa = ((sw >> (8*p))      & 0x0F) << 9;
        const int sb = ((sw >> (8*p + 4))  & 0x0F) << 9;
        const int gl = g_sgn_tab[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100) | sa];
        const int gh = g_sgn_tab[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100) | sb];
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(gl, u0, sumi);
        sumi = ggml_cuda_dp4a(gh, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// sgnpip: sign-folded table + pip restructure (all 8 table reads + 8 x ints in
// flight before the dp4a pair-chain)
static __device__ __forceinline__ float vd_sgnpip(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int sw = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * ys = (const uint8_t *) bq8_1[iqs/2].qs;
    int u[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) u[j] = get_int_b4(ys, j);
    int g[8];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int l0 = 2*p;
        const int sa = ((sw >> (8*p))     & 0x0F) << 9;
        const int sb = ((sw >> (8*p + 4)) & 0x0F) << 9;
        g[2*p + 0] = g_sgn_tab[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100) | sa];
        g[2*p + 1] = g_sgn_tab[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100) | sb];
    }
    int sa = 0, sb = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        sa = ggml_cuda_dp4a(g[2*p + 0], u[2*p + 0], sa);
        sb = ggml_cuda_dp4a(g[2*p + 1], u[2*p + 1], sb);
    }
    int sumi = sa + sb;
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// sexp: pre-expanded sign-mask table (2 KB). Per sign byte ONE lookup returns
// both 0x00/0xFF byte masks; the per-code vcmpne4 shift/or chains disappear.
// Bit-exact: same xor/vsub4 sign application, masks precomputed.
static __device__ __forceinline__ float vd_sexp(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int sw = get_int_b2(bq3->signs, iqs/2);
    int sumi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int l0 = 2*p;
        const uint2 me = g_sexp[(sw >> (8*p)) & 0xFF];
        const int gl = g_grid32[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const int gh = g_grid32[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = dp4a_bfe(__vsub4(gl ^ me.x, me.x), u0, sumi);
        sumi = dp4a_bfe(__vsub4(gh ^ me.y, me.y), u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// dp4a: base sign chain, hand-rolled byte-extract+mads dp4a
static __device__ __forceinline__ float vd_dp4a(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            g_grid32[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            g_grid32[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = dp4a_bfe(grid_l, u0, sumi);
        sumi = dp4a_bfe(grid_h, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// sexpdp: sexp masks + hand dp4a combined
static __device__ __forceinline__ float vd_sexpdp(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int sw = get_int_b2(bq3->signs, iqs/2);
    int sumi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int l0 = 2*p;
        const uint2 me = g_sexp[(sw >> (8*p)) & 0xFF];
        const int gl = g_grid32[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const int gh = g_grid32[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = dp4a_bfe(__vsub4(gl ^ me.x, me.x), u0, sumi);
        sumi = dp4a_bfe(__vsub4(gh ^ me.y, me.y), u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// -------------------------------------------------------------- kernels ----

template <int ARM>  // 0 base, 1 gmem, 2 pip, 3 sgn, 4 sgnpip, 5 sexp, 6 dp4a, 7 sexpdp
__global__ __launch_bounds__(128, 1) static void gemv_iq3s(
        const block_iq3_s * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        float * __restrict__ dst, const int blocks_per_row) {
    const int row = blockIdx.x;
    const int kqs  = 2*(threadIdx.x & 7);              // iqs in {0,2,...,14}
    const int kbx0 = 8*threadIdx.y + (threadIdx.x >> 3); // 16 kbx slots, NW2
    const int kstep = 16;
    const block_iq3_s * h = vx + (size_t)row * blocks_per_row;
    float tmp = 0.0f;
    for (int kbx = kbx0; kbx < blocks_per_row; kbx += kstep) {
        const block_iq3_s * b = h + kbx;
        const block_q8_1 * y = vy + kbx*8;             // kby = kbx*(qk/QK8_1)
        if (ARM == 0)      tmp += vd_base(b, y, kqs);
        else if (ARM == 1) tmp += vd_gmem(b, y, kqs);
        else if (ARM == 2) tmp += vd_pip(b, y, kqs);
        else if (ARM == 3) tmp += vd_sgn(b, y, kqs);
        else if (ARM == 4) tmp += vd_sgnpip(b, y, kqs);
        else if (ARM == 5) tmp += vd_sexp(b, y, kqs);
        else if (ARM == 6) tmp += vd_dp4a(b, y, kqs);
        else               tmp += vd_sexpdp(b, y, kqs);
    }
    __shared__ float red[1][64];
    if (threadIdx.y > 0) red[0][threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int l = 0; l < 1; ++l) tmp += red[l][threadIdx.x];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) tmp += __shfl_down(tmp, off, 64);
    if (threadIdx.x == 0) dst[row] = tmp;
}

// -------------------------------------------------------- host oracle ------
// Plain-C++ mirror of the SHIPPED vec_dot_iq3_s_q8_1 (the reference every arm
// must reproduce). Little-endian host: get_int_b2/b4 are direct reads.

static inline int h_vcmpne4(int a, int b) {
    int r = 0;
    for (int j = 0; j < 4; ++j) {
        const uint8_t ab = (a >> (8*j)) & 0xFF, bb = (b >> (8*j)) & 0xFF;
        r |= (ab != bb ? 0xFF : 0x00) << (8*j);
    }
    return r;
}
static inline int h_vsub4(int a, int b) {
    int r = 0;
    for (int j = 0; j < 4; ++j) {
        r |= ((((a >> (8*j)) & 0xFF) - ((b >> (8*j)) & 0xFF)) & 0xFF) << (8*j);
    }
    return r;
}
static inline int h_dp4a(int a, int b, int c) {
    int r = c;
    for (int j = 0; j < 4; ++j) {
        r += (int)((int8_t)((a >> (8*j)) & 0xFF)) * (int)((int8_t)((b >> (8*j)) & 0xFF));
    }
    return r;
}
static inline int h_rd32(const uint8_t * p) { int v; memcpy(&v, p, 4); return v; }

// bit-exact fp16 -> float, no library/__half involvement (HIP host-side
// ggml_half fields go through __half and VALUE-convert on assignment, which
// turned a naive ds.x = ggml_fp32_to_fp16(1.0f) fill into bits 0x7380)
static inline float h_fp16_to_fp32(uint16_t h) {
    const uint32_t s = h & 0x8000u, e = (h >> 10) & 0x1Fu, m = h & 0x03FFu;
    if (e == 0) return (s ? -1.0f : 1.0f) * (float)m * 5.9604644775390625e-08f; // m * 2^-24
    if (e == 31) return (m ? NAN : ((s ? -1.0f : 1.0f) * INFINITY));
    const uint32_t bits = (s << 16) | ((e + 112u) << 23) | (m << 13);
    float out; memcpy(&out, &bits, 4); return out;
}

// row dot over RAW BYTES: y layout = ggml block_q8_1 as compiled here
// (ds.x@0, ds.y@2, qs@4, stride 36); weight row = blocks_per_row blocks of
// 110 B (d@0, qs@2, qh@66, signs@74, scales@106). No ggml typed access on the
// host side: every field is read as explicit little-endian bytes.
static double host_row_dot(const uint8_t * h_grid, const uint8_t * wrow,
                           const uint8_t * y, int blocks_per_row) {
    double total = 0.0;
    for (int kbx = 0; kbx < blocks_per_row; ++kbx) {
        const uint8_t * b = wrow + (size_t)kbx * 110;
        for (int islot = 0; islot < 8; ++islot) {
            const int iqs = 2*islot;
            const uint8_t * yb = y + 36*(kbx*8 + islot);
            uint8_t qs8[8];
            memcpy(qs8, b + 2 + 4*iqs, 8);
            const int qh = b[66 + iqs/2];
            const int sw  = h_rd32(b + 74 + 4*(iqs/2));
            const uint8_t * s8 = (const uint8_t *) &sw;
            int sumi = 0;
            for (int l0 = 0; l0 < 8; l0 += 2) {
                const int g0 = (int) ((const uint32_t *)h_grid)[qs8[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
                const int g1 = (int) ((const uint32_t *)h_grid)[qs8[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
                const int signs0 = h_vcmpne4(((s8[l0/2] & 0x03) << 7) | ((s8[l0/2] & 0x0C) << 21), 0);
                const int signs1 = h_vcmpne4(((s8[l0/2] & 0x30) << 3) | ((s8[l0/2] & 0xC0) << 17), 0);
                const int gl = h_vsub4(g0 ^ signs0, signs0);
                const int gh = h_vsub4(g1 ^ signs1, signs1);
                const int u0 = h_rd32(yb + 4 + 4*(l0 + 0));
                const int u1 = h_rd32(yb + 4 + 4*(l0 + 1));
                sumi = h_dp4a(gl, u0, sumi);
                sumi = h_dp4a(gh, u1, sumi);
            }
            sumi *= 1 + 2*((b[106 + iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
            const float d  = h_fp16_to_fp32(*(const uint16_t *)(b + 0));
            const float ds = h_fp16_to_fp32(*(const uint16_t *)(yb + 0));
            total += (double)d * (double)ds * (double)sumi;
        }
    }
    return total;
}

// ------------------------------------------------------------------ main ---

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <gguf> [niter=200] [reps=3]\n", argv[0]); return 1; }
    const int niter = argc > 2 ? atoi(argv[2]) : 200;
    const int reps  = argc > 3 ? atoi(argv[3]) : 3;

    // blk.21.ffn_up.weight [5120,17408] iq3_s, offset 4301102016,
    // len = next_offset - offset = 38,297,600 B (offset-delta law)
    const long off = 4301102016L, len = 38297600L;
    const int K = 5120, N = 17408;
    const int blocks_per_row = K / 256;     // 20
    const int nq8 = K / 32;                 // 160
    if ((long)blocks_per_row * sizeof(block_iq3_s) * N != len) {
        printf("TUNE MISMATCH: %ld vs %ld - refusing (offset-delta law)\n",
               (long)blocks_per_row * sizeof(block_iq3_s) * N, len);
        return 1;
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { printf("open gguf failed\n"); return 1; }
    void * mm = mmap(nullptr, len + off, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm == MAP_FAILED) { printf("mmap failed\n"); return 1; }
    const uint8_t * wsrc = (const uint8_t *) mm + off;

    // device buffers
    block_iq3_s * dw; block_q8_1 * dy; float * dd;
    CHECK(hipMalloc(&dw, len));
    CHECK(hipMalloc(&dy, nq8 * sizeof(block_q8_1)));
    CHECK(hipMalloc(&dd, N * sizeof(float)));
    CHECK(hipMemcpy(dw, wsrc, len, hipMemcpyHostToDevice));
    // synthetic q8_1 as RAW BYTES matching the compiled block_q8_1 layout
    // (ds.x@0, ds.y@2, qs@4, stride 36). Never assign through ggml_half fields
    // on the host: HIP __half semantics VALUE-convert instead of bit-store.
    std::vector<uint8_t> hq(nq8 * 36, 0);
    for (int i = 0; i < nq8; ++i) {
        uint8_t * b = hq.data() + 36*i;
        for (int j = 0; j < 32; ++j) b[4 + j] = (uint8_t)(int8_t)((i*37 + j*11) % 255 - 127);
        const uint16_t dx = 0x3C00, dyv = 0x5400;          // fp16(1.0), fp16(64.0)
        memcpy(b + 0, &dx, 2);
        memcpy(b + 2, &dyv, 2);
    }
    CHECK(hipMemcpy(dy, hq.data(), nq8 * 36, hipMemcpyHostToDevice));
    // raw-byte echo check: device must see exactly what the host built
    {
        std::vector<uint8_t> echo(72, 0xAA);
        CHECK(hipMemcpy(echo.data(), dy, 72, hipMemcpyDeviceToHost));
        if (memcmp(echo.data(), hq.data(), 72) != 0) {
            printf("Y UPLOAD MISMATCH - refusing\n");
            return 1;
        }
    }

    // tables. GGML_COMMON_IMPL_HIP makes iq3s_grid a __device__-only symbol:
    // pull it to host once (single source of truth) and derive all copies.
    std::vector<uint32_t> h_grid(512);
    uint32_t * dgrid = nullptr;
    CHECK(hipMalloc(&dgrid, 512 * sizeof(uint32_t)));
    pull_grid<<<1, 512>>>(dgrid);
    CHECK(hipMemcpy(h_grid.data(), dgrid, 512 * sizeof(uint32_t), hipMemcpyDeviceToHost));
    CHECK(hipFree(dgrid));
    // pinned values straight from ggml-common.h (lines 1043, 1083): catches a
    // broken table pull before any arm is judged
    if (h_grid[0] != 0x01010101u || h_grid[322] != 0x07070b0bu) {
        printf("GRID PULL MISMATCH: [0]=%08x [322]=%08x - refusing\n", h_grid[0], h_grid[322]);
        return 1;
    }
    std::vector<uint32_t> sgn_host;
    build_sgn_tab(h_grid, sgn_host);
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_grid32), h_grid.data(), 512 * sizeof(uint32_t)));
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_sgn_tab), sgn_host.data(), 8192 * sizeof(uint32_t)));
    // verify device table copies landed (both eyes on the upload path)
    std::vector<uint2> sexp_host;
    build_sexp(sexp_host);
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_sexp), sexp_host.data(), 256 * sizeof(uint2)));
    std::vector<uint32_t> tchk(512 + 8192, 0xdeadbeef);
    CHECK(hipMemcpyFromSymbol(tchk.data(), HIP_SYMBOL(g_grid32), 512 * sizeof(uint32_t)));
    CHECK(hipMemcpyFromSymbol(tchk.data() + 512, HIP_SYMBOL(g_sgn_tab), 8192 * sizeof(uint32_t)));
    if (memcmp(tchk.data(), h_grid.data(), 512 * 4) != 0 ||
        memcmp(tchk.data() + 512, sgn_host.data(), 8192 * 4) != 0) {
        printf("TABLE UPLOAD MISMATCH - refusing\n");
        return 1;
    }
    // spot-check sexp semantics against the atom's vcmpne4 chain on host
    for (int s8 = 0; s8 < 256; ++s8) {
        const int m0 = h_vcmpne4(((s8 & 0x03) << 7) | ((s8 & 0x0C) << 21), 0);
        const int m1 = h_vcmpne4(((s8 & 0x30) << 3) | ((s8 & 0xC0) << 17), 0);
        if ((uint32_t)m0 != sexp_host[s8].x || (uint32_t)m1 != sexp_host[s8].y) {
            printf("SEXP BUILD MISMATCH at %d - refusing\n", s8);
            return 1;
        }
    }

    // host oracle reference on rows 0..63
    const int ORACLE_ROWS = 64;
    std::vector<double> ref(ORACLE_ROWS);
    for (int r = 0; r < ORACLE_ROWS; ++r) {
        ref[r] = host_row_dot((const uint8_t *)h_grid.data(),
                              wsrc + (size_t)r * blocks_per_row * 110,
                              hq.data(), blocks_per_row);
    }

    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));

    const double bytes = (double)len + (double)nq8 * sizeof(block_q8_1) + (double)N * 4;
    const double floor_us = bytes / 183.8e9 * 1e6;

    printf("shape: N=%d K=%d iq3_s, %.1f MB weights | byte floor %.1f us at 183.8 GB/s\n",
           N, K, len/1e6, floor_us);
    printf("oracle: %d rows, rel err gate 1e-4 | sgn table 8192 x 4 B = 32 KB\n", ORACLE_ROWS);
    printf("%-10s %10s %10s %10s %9s   %s\n", "variant", "us/call", "GB/s", "x floor", "vs base", "oracle / per-rep us");

    struct V { const char * name; int arm; };
    V vars[] = {
        {"base", 0}, {"gmem", 1}, {"pip", 2}, {"sgn", 3}, {"sgnpip", 4},
        {"sexp", 5}, {"dp4a", 6}, {"sexpdp", 7},
    };
    const int nv = (int)(sizeof(vars)/sizeof(vars[0]));
    dim3 grid(N), block(64, 2);
    double med_us[8] = {0};

    for (int iv = 0; iv < nv; ++iv) {
        auto launch = [&]() {
            switch (vars[iv].arm) {
                case 0: gemv_iq3s<0><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
                case 1: gemv_iq3s<1><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
                case 2: gemv_iq3s<2><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
                case 3: gemv_iq3s<3><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
                case 4: gemv_iq3s<4><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
                case 5: gemv_iq3s<5><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
                case 6: gemv_iq3s<6><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
                case 7: gemv_iq3s<7><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
            }
        };

        // warmup + ORACLE GATE
        launch(); CHECK(hipDeviceSynchronize());
        std::vector<float> dev(ORACLE_ROWS);
        CHECK(hipMemcpy(dev.data(), dd, ORACLE_ROWS * sizeof(float), hipMemcpyDeviceToHost));
        double max_rel = 0.0; int worst = -1; int nfail = 0;
        for (int r = 0; r < ORACLE_ROWS; ++r) {
            const double rel = fabs((double)dev[r] - ref[r]) / (fabs(ref[r]) + 1e-30);
            if (rel > max_rel) { max_rel = rel; worst = r; }
            if (rel >= 1e-4) ++nfail;
        }
        if (max_rel >= 1e-4) {
            printf("%-10s ORACLE FAIL: %d/%d rows over 1e-4; worst row %d dev=%.9g ref=%.9g rel=%.3e - refused\n",
                   vars[iv].name, nfail, ORACLE_ROWS, worst, (double)dev[worst], ref[worst], max_rel);
            continue;
        }

        // 3 interleaved reps of niter timed iterations each
        double us[8];
        for (int rep = 0; rep < reps; ++rep) {
            CHECK(hipEventRecord(e0));
            for (int i = 0; i < niter; ++i) launch();
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, e0, e1));
            us[rep] = 1000.0 * ms / niter;
        }
        std::vector<double> usv(us, us + reps);
        std::sort(usv.begin(), usv.end());
        med_us[iv] = usv[reps/2];
        const double gbs = bytes / (med_us[iv] * 1e-6) / 1e9;
        printf("%-10s %10.1f %10.1f %10.2f %8.2fx   PASS (%.1e) [", vars[iv].name, med_us[iv], gbs,
               med_us[iv] / floor_us, med_us[iv] / med_us[0], max_rel);
        for (int rep = 0; rep < reps; ++rep) printf("%.1f%s", us[rep], rep + 1 < reps ? " " : "");
        printf("]\n");
    }

    printf("\ncontrol drift check: base reps above; >1%% spread between reps = void session\n");
    return 0;
}
