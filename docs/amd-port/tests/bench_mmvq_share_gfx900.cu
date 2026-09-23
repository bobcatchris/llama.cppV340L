// MMVQ share-extension cell v1 (memcmp-gated): decode-once share for the
// uncovered served types at T=2..4 for gfx900 (Vega 10, wave64), die 3 dev
// cell. Extends the banked iq3_s T-band receipt (W2_mmvq2_tband) to the
// served T=4 mix named by the first kernel profile (E-102 cont):
//   type18 iq3_xxs, type12 q4_K, type23 iq4_xs, type13 q5_K, type14 q6_K,
//   type11 q3_K.
//
// KEY SCHEDULE FACT vs the iq3_s harness: the REAL mul_mat_vec_q on the GCN
// table runs ncols_dst 2..4 with rows_per_cuda_block = 2 (calc_rows_per_block)
// and nwarps = 2 (block 64x2, 128 threads). This cell clones THAT schedule:
//   row0 = 2*blockIdx.x, kbx = tid/(qi/vdr) (+ vdr*nwarps*64/qi stride),
//   kqs = vdr*(tid % (qi/vdr)), y block for token j = j*stride_col_y + kby,
//   x block for row i = (row0+i)*blocks_per_row + kbx.
// The decode-once share amortizes the y-independent decode per (row, kbx,
// lane) across the T tokens; rows share nothing with each other.
//
// Arms per (type, T):
//   base  : the EXACT shipped vec_dot_<t>_q8_1 from vecdotq.cuh, once per
//           (token j, row i) - the unshared path of record.
//   share : the EXACT new vec_dot_<t>_q8_1_decode/apply split pair from
//           vecdotq.cuh (the functions the env gates wire into mmvq.cu).
//   aln   : share over the 48-byte aligned q8_1 layout (block_q8_1_aln,
//           ds@0 qs@16, uint4 operand loads) - the EXACT *_apply_aln twins
//           from vecdotq.cuh (the functions the LLAMA_MMVQ_ALN gate wires
//           into mmvq.cu for the T=4 share arms). Legacy T=1-3 arms keep
//           the 36-byte layout; aln arms at T=1-3 measure the regression
//           class only (the served gate never engages them).
//
//   wide  : share with the x-operand fetched through the widest LEGAL load
//           per type (uint2 for the iq4_xs 8B-aligned windows, overlapping-
//           dword funnel shifts for the 2B-aligned iq3_s/iq3_xxs windows) -
//           the A2 rung. Bit-exact: only the load instructions change, the
//           loaded bytes and all arithmetic are identical.
//   cfull/cx/cy : pure-consume ablations (the Phase-1 instrument): the exact
//           same global loads as share, xor-consumed instead of decode+dp4a.
//           cfull = x+y loads, cx = x only, cy = y only. Timing-only arms
//           (dst holds xor garbage by design, no oracle); cfull vs its dup
//           control carries a determinism memcmp on the garbage dst.
//
// ORACLE: bit-exact f32 output vs the unshared path - device memcmp of the
// full dst (all N rows x T tokens), per (type, T), before any timing counts.
// In-run controls: t4b2/t4s2/t4a2 duplicate the T=4 base/share/aln configs;
// spread gate 1% per config (session law).
//
// build:
//   /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP \
//     -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//     docs/amd-port/tests/bench_mmvq_share_gfx900.cu -o /tmp/bench_mmvq_share \
//     -L /media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin -lggml-base
// run (die 3 only, under the campaign lock):
//   HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin \
//     /tmp/bench_mmvq_share <gguf> [niter=200] [reps=3]
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

static_assert(sizeof(block_iq3_xxs) == 98,  "iq3_xxs block size");
static_assert(sizeof(block_q3_K)   == 110, "q3_K block size");
static_assert(sizeof(block_q4_K)   == 144, "q4_K block size");
static_assert(sizeof(block_q5_K)   == 176, "q5_K block size");
static_assert(sizeof(block_q6_K)   == 210, "q6_K block size");
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block size");

// --- A2 wide-load helpers (bit-exact byte sources, wider legal loads) ------
// 8-byte window from a 2B-aligned stream: three overlapping aligned dwords +
// funnel shifts. Same bytes as two get_int_b2 words, 3 loads instead of 4.
// Worst-case read touches <= 8 bytes past the window, still inside the block
// for every window used here.
static __device__ __forceinline__ uint2 ldw8_b2(const uint8_t * p) {
    const uint32_t s = ((uint32_t)(uintptr_t)p & 3u) * 8u;   // 0 or 16
    const uint32_t * p4 = (const uint32_t *)((uintptr_t)p & ~(uintptr_t)3u);
    const uint32_t d0 = p4[0], d1 = p4[1], d2 = p4[2];
    return make_uint2(__funnelshift_r(d0, d1, s), __funnelshift_r(d1, d2, s));
}
// 4-byte window from a 2B-aligned stream: two overlapping dwords + funnel.
static __device__ __forceinline__ uint32_t ldw4_b2(const uint8_t * p) {
    const uint32_t s = ((uint32_t)(uintptr_t)p & 3u) * 8u;   // 0 or 16
    const uint32_t * p4 = (const uint32_t *)((uintptr_t)p & ~(uintptr_t)3u);
    return __funnelshift_r(p4[0], p4[1], s);
}

// per-type schedule constants (GCN table, GGML common defs) + the shipped
// vec_dot / split pair bindings
template <ggml_type GT> struct share_ops;

// --- A2 wide twins: same loaded bytes and identical arithmetic, only the
// x-operand load instructions change (bit-exact by construction).

static __device__ __forceinline__ void vec_dot_iq3_s_q8_1_decode_wide(
    const void * __restrict__ vbq, const int & kbx, const int & iqs, int * dec, int & scale, float & d) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const uint2 qs_packed = ldw8_b2(bq3->qs + 4*iqs);
    const uint8_t * qs = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int signs_packed_32 = (int) ldw4_b2(bq3->signs + 4*(iqs/2));
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;

#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int g0 = iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const int g1 = iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        dec[l0 + 0] = __vsub4(g0 ^ signs0, signs0);
        dec[l0 + 1] = __vsub4(g1 ^ signs1, signs1);
    }

    scale = 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    d     = __half2float(bq3->d);
}

static __device__ __forceinline__ void vec_dot_iq3_xxs_q8_1_decode_wide(
    const void * __restrict__ vbq, const int & kbx, const int & iqs, int * dec, int & ls, float & d) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const uint2 q3_packed = ldw8_b2(bq3->qs + 4*iqs);
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = ldw4_b2(bq3->qs + 4*(QK_K/16 + iqs/2));

#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int g0 = iq3xxs_grid[q3[l0 + 0]];
        const int g1 = iq3xxs_grid[q3[l0 + 1]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);

        dec[l0 + 0] = __vsub4(g0 ^ signs0, signs0);
        dec[l0 + 1] = __vsub4(g1 ^ signs1, signs1);
    }

    ls = aux32 >> 28;
    d  = __half2float(bq3->d);
}

// iq4_xs: block stride 136 = 17x8 keeps every lane's 16-byte qs window
// 8B aligned, so two uint2 loads legally replace the four scalar dwords.
static __device__ __forceinline__ void vec_dot_iq4_xs_q8_1_decode_u2(
    const void * __restrict__ vbq, const int & kbx, const int & iqs, int * dec, int & scale, float & d) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

#pragma unroll
    for (int j = 0; j < 4; j += 2) {
        const uint2 q4v = *reinterpret_cast<const uint2 *>(bq4->qs + 4*(iqs + j));
        const int2 v0 = get_int_from_table_16((int)q4v.x, kvalues_iq4nl);
        const int2 v1 = get_int_from_table_16((int)q4v.y, kvalues_iq4nl);
        dec[j + 0] = v0.x;
        dec[j + 4] = v0.y;
        dec[j + 1] = v1.x;
        dec[j + 5] = v1.y;
    }

    scale = (((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4)) - 32;
    d     = __half2float(bq4->d);
}

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1_u2(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
        const uint2 q4v = *reinterpret_cast<const uint2 *>(bq4->qs + 4*(iqs + j));
        const int2 v0 = get_int_from_table_16((int)q4v.x, kvalues_iq4nl);
        const int2 v1 = get_int_from_table_16((int)q4v.y, kvalues_iq4nl);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);
        const int u2 = get_int_b4(bq8_1[iqs/4].qs, j + 1);
        const int u3 = get_int_b4(bq8_1[iqs/4].qs, j + 5);

        sumi = ggml_cuda_dp4a(v0.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v0.y, u1, sumi);
        sumi = ggml_cuda_dp4a(v1.x, u2, sumi);
        sumi = ggml_cuda_dp4a(v1.y, u3, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}

// --- A3 lever (a): row-shared y loads. The rpb=2 schedule makes both rows'
// apply() read the SAME y words per (token, kbx, lane); the compiler does not
// CSE them across the row loop. apply_u() takes the y words preloaded once
// per (token, lane) and replays the identical arithmetic per row.

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_apply_u(
    const int * __restrict__ u, const int ds_w, const int * dec, const int scale, const float d) {
    int sumi = 0;
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        sumi = ggml_cuda_dp4a(dec[m], u[m], sumi);
    }
    sumi *= scale;
    const float ds = __low2float(*reinterpret_cast<const ggml_half2 *>(&ds_w));
    return d * ds * sumi;
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_apply_u(
    const int * __restrict__ u, const int ds_w, const int * dec, const int ls, const float d) {
    int sumi = 0;
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        sumi = ggml_cuda_dp4a(dec[m], u[m], sumi);
    }
    sumi = (ls*sumi + sumi/2)/2;
    const float ds = __low2float(*reinterpret_cast<const ggml_half2 *>(&ds_w));
    return d * ds * sumi;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_apply_u(
    const int (* __restrict__ u)[2], const int * __restrict__ ds_w,
    const int * dec, const int * sc, const int * m, const ggml_half2 dm) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const float d8 = __low2float(*reinterpret_cast<const ggml_half2 *>(&ds_w[i]));
        const int dot1 = ggml_cuda_dp4a(dec[2*i+1], u[i][1], ggml_cuda_dp4a(dec[2*i+0], u[i][0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[i][1], ggml_cuda_dp4a(0x01010101, u[i][0], 0));
        sumf_d += d8 * (dot1 * sc[i]);
        sumf_m += d8 * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_apply_u(
    const int (* __restrict__ u)[2], const int * __restrict__ ds_w,
    const int * dec, const int * sc, const int * m, const ggml_half2 dm) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const float d8 = __low2float(*reinterpret_cast<const ggml_half2 *>(&ds_w[i]));
        const int dot1 = ggml_cuda_dp4a(dec[2*i+0], u[i][0], ggml_cuda_dp4a(dec[2*i+1], u[i][1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[i][0], ggml_cuda_dp4a(0x01010101, u[i][1], 0));
        sumf_d += d8 * (dot1 * sc[i]);
        sumf_m += d8 * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm);
    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_apply_u(
    const int * __restrict__ u, const int * __restrict__ ds_w,
    const int * dec, const int * sc, const float d) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const float d8 = __low2float(*reinterpret_cast<const ggml_half2 *>(&ds_w[i]));
        sumf += d8 * (ggml_cuda_dp4a(dec[i], u[i], 0) * sc[i]);
    }
    return d*sumf;
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_apply_u(
    const int * __restrict__ u, const int * __restrict__ ds_w,
    const int * dec, const int * sc, const float d) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const float d8 = __low2float(*reinterpret_cast<const ggml_half2 *>(&ds_w[i]));
        sumf += d8 * (ggml_cuda_dp4a(dec[i], u[i], 0) * sc[i]); // SIMD dot product
    }
    return d * sumf;
}


static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1_apply_u(
    const int * __restrict__ u, const int ds_w, const int * dec, const int scale, const float d) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sumi = ggml_cuda_dp4a(dec[j + 0], u[j + 0], sumi);
        sumi = ggml_cuda_dp4a(dec[j + 4], u[j + 4], sumi);
    }
    sumi *= scale;
    const float ds = __low2float(*reinterpret_cast<const ggml_half2 *>(&ds_w));
    return d * ds * sumi;
}

template <> struct share_ops<GGML_TYPE_IQ3_S> {
    static constexpr bool has_aln = true;
    static constexpr bool has_wide = true;
    static constexpr bool wide_is_base = false;
    static constexpr int qi = QI3_S, vdr = VDR_IQ3_S_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq3_s_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[8]; int scale; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_s_q8_1_decode(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ void decode_wide(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_s_q8_1_decode_wide(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_iq3_s_q8_1_apply(y, kqs, s.dec, s.scale, s.d);
    }    static __device__ float apply_aln(const block_q8_1_aln * y, int kqs, const state & s) {
        return vec_dot_iq3_s_q8_1_apply_aln(y, kqs, s.dec, s.scale, s.d);
    }
    // pure-consume twins: the exact global loads of decode/apply, xor only
    static __device__ uint32_t consume_x(const void * vx, int kbx, int kqs) {
        const block_iq3_s * bq3 = (const block_iq3_s *) vx + kbx;
        uint32_t a = (uint32_t) get_int_b2(bq3->qs, kqs);
        a ^= (uint32_t) get_int_b2(bq3->qs, kqs + 1);
        a ^= bq3->qh[kqs/2];
        a ^= (uint32_t) get_int_b2(bq3->signs, kqs/2);
        a ^= bq3->scales[kqs/4];
        a ^= (uint32_t) *(const uint16_t *) &bq3->d;
        return a;
    }
    static __device__ uint32_t consume_y(const block_q8_1 * y, int kqs) {
        const block_q8_1 * b = y + kqs/2;
        uint32_t a = 0;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            a ^= (uint32_t) get_int_b4(b->qs, m);
        }
        a ^= (uint32_t) ((const int *) b)[0];
        return a;
    }
    struct yw { int u[8]; int ds; };
    static __device__ yw preload(const block_q8_1 * y, int kqs) {
        const block_q8_1 * b = y + kqs/2;
        yw w;
#pragma unroll
        for (int m = 0; m < 8; ++m) w.u[m] = get_int_b4(b->qs, m);
        w.ds = ((const int *) b)[0];
        return w;
    }
    static __device__ float apply_u(const yw & w, int kqs, const state & s) {
        return vec_dot_iq3_s_q8_1_apply_u(w.u, w.ds, s.dec, s.scale, s.d);
    }
};

template <> struct share_ops<GGML_TYPE_IQ3_XXS> {
    static constexpr bool has_aln = true;
    static constexpr bool has_wide = true;
    static constexpr bool wide_is_base = false;
    static constexpr int qi = QI3_XXS, vdr = VDR_IQ3_XXS_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq3_xxs_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[8]; int ls; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_xxs_q8_1_decode(vx, kbx, kqs, s.dec, s.ls, s.d);
    }
    static __device__ void decode_wide(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_xxs_q8_1_decode_wide(vx, kbx, kqs, s.dec, s.ls, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_iq3_xxs_q8_1_apply(y, kqs, s.dec, s.ls, s.d);
    }    static __device__ float apply_aln(const block_q8_1_aln * y, int kqs, const state & s) {
        return vec_dot_iq3_xxs_q8_1_apply_aln(y, kqs, s.dec, s.ls, s.d);
    }
    static __device__ uint32_t consume_x(const void * vx, int kbx, int kqs) {
        const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vx + kbx;
        uint32_t a = (uint32_t) get_int_b2(bq3->qs, kqs);
        a ^= (uint32_t) get_int_b2(bq3->qs, kqs + 1);
        a ^= (uint32_t) get_int_b2(bq3->qs, QK_K/16 + kqs/2);
        a ^= (uint32_t) *(const uint16_t *) &bq3->d;
        return a;
    }
    static __device__ uint32_t consume_y(const block_q8_1 * y, int kqs) {
        const block_q8_1 * b = y + kqs/2;
        uint32_t a = 0;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            a ^= (uint32_t) get_int_b4(b->qs, m);
        }
        a ^= (uint32_t) ((const int *) b)[0];
        return a;
    }
    struct yw { int u[8]; int ds; };
    static __device__ yw preload(const block_q8_1 * y, int kqs) {
        const block_q8_1 * b = y + kqs/2;
        yw w;
#pragma unroll
        for (int m = 0; m < 8; ++m) w.u[m] = get_int_b4(b->qs, m);
        w.ds = ((const int *) b)[0];
        return w;
    }
    static __device__ float apply_u(const yw & w, int kqs, const state & s) {
        return vec_dot_iq3_xxs_q8_1_apply_u(w.u, w.ds, s.dec, s.ls, s.d);
    }
};

template <> struct share_ops<GGML_TYPE_Q4_K> {
    static constexpr bool has_aln = true;
    static constexpr bool has_wide = false;
    static constexpr bool wide_is_base = false;
    static constexpr int qi = QI4_K, vdr = VDR_Q4_K_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_q4_K_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[2*QR4_K]; int sc[QR4_K]; int m[QR4_K]; ggml_half2 dm; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_q4_K_q8_1_decode(vx, kbx, kqs, s.dec, s.sc, s.m, s.dm);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_q4_K_q8_1_apply(y, kqs, s.dec, s.sc, s.m, s.dm);
    }    static __device__ float apply_aln(const block_q8_1_aln * y, int kqs, const state & s) {
        return vec_dot_q4_K_q8_1_apply_aln(y, kqs, s.dec, s.sc, s.m, s.dm);
    }
    static __device__ uint32_t consume_x(const void * vx, int kbx, int kqs) {
        const block_q4_K * bq4 = (const block_q4_K *) vx + kbx;
        const int bq8_offset = QR4_K * ((kqs/2) / (QI8_1/2));
        const int * q4 = (const int *)(bq4->qs + 16 * bq8_offset + 4 * ((kqs/2)%4));
        uint32_t a = (uint32_t) q4[0];
        a ^= (uint32_t) q4[4];
        const uint16_t * scales = (const uint16_t *)bq4->scales;
        const int j = bq8_offset/2;
        if (j < 2) {
            a ^= scales[j+0];
            a ^= scales[j+2];
        } else {
            a ^= scales[j+2];
            a ^= scales[j-2];
            a ^= scales[j];
        }
        a ^= (uint32_t) ((const int *) bq4)[0];   // dm word
        return a;
    }
    static __device__ uint32_t consume_y(const block_q8_1 * y, int kqs) {
        const int bq8_offset = QR4_K * ((kqs/2) / (QI8_1/2));
        uint32_t a = 0;
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1 * bq8i = y + bq8_offset + i;
            const int * q8 = (const int *)bq8i->qs + ((kqs/2)%4);
            a ^= (uint32_t) q8[0];
            a ^= (uint32_t) q8[4];
            a ^= (uint32_t) ((const int *) bq8i)[0];
        }
        return a;
    }

    struct yw { int u[QR4_K][2]; int ds[QR4_K]; };
    static __device__ yw preload(const block_q8_1 * y, int kqs) {
        const int bq8_offset = QR4_K * ((kqs/2) / (QI8_1/2));
        yw w;
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1 * bq8i = y + bq8_offset + i;
            const int * q8 = (const int *)bq8i->qs + ((kqs/2)%4);
            w.u[i][0] = q8[0];
            w.u[i][1] = q8[4];
            w.ds[i]   = ((const int *) bq8i)[0];
        }
        return w;
    }
    static __device__ float apply_u(const yw & w, int kqs, const state & s) {
        return vec_dot_q4_K_q8_1_apply_u(w.u, w.ds, s.dec, s.sc, s.m, s.dm);
    }
};

template <> struct share_ops<GGML_TYPE_IQ4_XS> {
    static constexpr bool has_aln = false;
    static constexpr bool has_wide = true;
    static constexpr bool wide_is_base = true;   // served path is base: the wide twin replaces the scalar loads there
    static constexpr int qi = QI4_XS, vdr = VDR_IQ4_XS_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq4_xs_q8_1(vx, y, kbx, kqs);
    }
    static __device__ float base_wide(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq4_xs_q8_1_u2(vx, y, kbx, kqs);
    }
    struct state { int dec[8]; int scale; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq4_xs_q8_1_decode(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ void decode_wide(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq4_xs_q8_1_decode_u2(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_iq4_xs_q8_1_apply(y, kqs, s.dec, s.scale, s.d);
    }
    static __device__ uint32_t consume_x(const void * vx, int kbx, int kqs) {
        const block_iq4_xs * bq4 = (const block_iq4_xs *) vx + kbx;
        uint32_t a = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            a ^= (uint32_t) get_int_b4(bq4->qs, kqs + j);
        }
        a ^= bq4->scales_l[kqs/8];
        a ^= (uint32_t) bq4->scales_h;
        a ^= (uint32_t) *(const uint16_t *) &bq4->d;
        return a;
    }
    static __device__ uint32_t consume_y(const block_q8_1 * y, int kqs) {
        const block_q8_1 * b = y + kqs/4;
        uint32_t a = 0;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            a ^= (uint32_t) get_int_b4(b->qs, m);
        }
        a ^= (uint32_t) ((const int *) b)[0];
        return a;
    }

    struct yw { int u[8]; int ds; };
    static __device__ yw preload(const block_q8_1 * y, int kqs) {
        const block_q8_1 * b = y + kqs/4;
        yw w;
#pragma unroll
        for (int m = 0; m < 8; ++m) w.u[m] = get_int_b4(b->qs, m);
        w.ds = ((const int *) b)[0];
        return w;
    }
    static __device__ float apply_u(const yw & w, int kqs, const state & s) {
        return vec_dot_iq4_xs_q8_1_apply_u(w.u, w.ds, s.dec, s.scale, s.d);
    }
};

template <> struct share_ops<GGML_TYPE_Q5_K> {
    static constexpr bool has_aln = true;
    static constexpr bool has_wide = false;
    static constexpr bool wide_is_base = false;
    static constexpr int qi = QI5_K, vdr = VDR_Q5_K_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_q5_K_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[2*QR5_K]; int sc[QR5_K]; int m[QR5_K]; ggml_half2 dm; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_q5_K_q8_1_decode(vx, kbx, kqs, s.dec, s.sc, s.m, s.dm);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_q5_K_q8_1_apply(y, kqs, s.dec, s.sc, s.m, s.dm);
    }    static __device__ float apply_aln(const block_q8_1_aln * y, int kqs, const state & s) {
        return vec_dot_q5_K_q8_1_apply_aln(y, kqs, s.dec, s.sc, s.m, s.dm);
    }
    static __device__ uint32_t consume_x(const void * vx, int kbx, int kqs) {
        const block_q5_K * bq5 = (const block_q5_K *) vx + kbx;
        const int bq8_offset = QR5_K * ((kqs/2) / (QI8_1/2));
        const int * ql = (const int *)(bq5->qs + 16 * bq8_offset + 4 * ((kqs/2)%4));
        const int * qh = (const int *)(bq5->qh + 4 * ((kqs/2)%4));
        uint32_t a = (uint32_t) ql[0];
        a ^= (uint32_t) ql[4];
        a ^= (uint32_t) qh[0];
        a ^= (uint32_t) qh[4];
        const uint16_t * scales = (const uint16_t *)bq5->scales;
        const int j = bq8_offset/2;
        if (j < 2) {
            a ^= scales[j+0];
            a ^= scales[j+2];
        } else {
            a ^= scales[j+2];
            a ^= scales[j-2];
            a ^= scales[j];
        }
        a ^= (uint32_t) ((const int *) bq5)[0];
        return a;
    }
    static __device__ uint32_t consume_y(const block_q8_1 * y, int kqs) {
        const int bq8_offset = QR5_K * ((kqs/2) / (QI8_1/2));
        uint32_t a = 0;
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1 * bq8i = y + bq8_offset + i;
            const int * q8 = (const int *)bq8i->qs + ((kqs/2)%4);
            a ^= (uint32_t) q8[0];
            a ^= (uint32_t) q8[4];
            a ^= (uint32_t) ((const int *) bq8i)[0];
        }
        return a;
    }

    struct yw { int u[QR5_K][2]; int ds[QR5_K]; };
    static __device__ yw preload(const block_q8_1 * y, int kqs) {
        const int bq8_offset = QR5_K * ((kqs/2) / (QI8_1/2));
        yw w;
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1 * bq8i = y + bq8_offset + i;
            const int * q8 = (const int *)bq8i->qs + ((kqs/2)%4);
            w.u[i][0] = q8[0];
            w.u[i][1] = q8[4];
            w.ds[i]   = ((const int *) bq8i)[0];
        }
        return w;
    }
    static __device__ float apply_u(const yw & w, int kqs, const state & s) {
        return vec_dot_q5_K_q8_1_apply_u(w.u, w.ds, s.dec, s.sc, s.m, s.dm);
    }
};

template <> struct share_ops<GGML_TYPE_Q6_K> {
    static constexpr bool has_aln = true;
    static constexpr bool has_wide = false;
    static constexpr bool wide_is_base = false;
    static constexpr int qi = QI6_K, vdr = VDR_Q6_K_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_q6_K_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[QR6_K]; int sc[QR6_K]; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_q6_K_q8_1_decode(vx, kbx, kqs, s.dec, s.sc, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_q6_K_q8_1_apply(y, kqs, s.dec, s.sc, s.d);
    }    static __device__ float apply_aln(const block_q8_1_aln * y, int kqs, const state & s) {
        return vec_dot_q6_K_q8_1_apply_aln(y, kqs, s.dec, s.sc, s.d);
    }
    static __device__ uint32_t consume_x(const void * vx, int kbx, int kqs) {
        const block_q6_K * bq6 = (const block_q6_K *) vx + kbx;
        const int scale_offset = (QI6_K/4) * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/8);
        uint32_t a = (uint32_t) get_int_b2(bq6->ql, kqs);
        a ^= (uint32_t) get_int_b2(bq6->qh, (QI6_K/4) * (kqs / (QI6_K/2)) + kqs % (QI6_K/4));
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            a ^= (uint32_t) bq6->scales[scale_offset + 4*i];
        }
        a ^= (uint32_t) *(const uint16_t *) &bq6->d;
        return a;
    }
    static __device__ uint32_t consume_y(const block_q8_1 * y, int kqs) {
        const int bq8_offset = 2 * QR6_K * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/4);
        uint32_t a = 0;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            const block_q8_1 * bq8i = y + bq8_offset + 2*i;
            a ^= (uint32_t) get_int_b4(bq8i->qs, kqs % QI8_1);
            a ^= (uint32_t) ((const int *) bq8i)[0];
        }
        return a;
    }

    struct yw { int u[QR6_K]; int ds[QR6_K]; };
    static __device__ yw preload(const block_q8_1 * y, int kqs) {
        const int bq8_offset = 2 * QR6_K * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/4);
        yw w;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            const block_q8_1 * bq8i = y + bq8_offset + 2*i;
            w.u[i]  = get_int_b4(bq8i->qs, kqs % QI8_1);
            w.ds[i] = ((const int *) bq8i)[0];
        }
        return w;
    }
    static __device__ float apply_u(const yw & w, int kqs, const state & s) {
        return vec_dot_q6_K_q8_1_apply_u(w.u, w.ds, s.dec, s.sc, s.d);
    }
};

template <> struct share_ops<GGML_TYPE_Q3_K> {
    static constexpr bool has_aln = true;
    static constexpr bool has_wide = false;
    static constexpr bool wide_is_base = false;
    static constexpr int qi = QI3_K, vdr = VDR_Q3_K_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_q3_K_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[QR3_K]; int sc[QR3_K]; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_q3_K_q8_1_decode(vx, kbx, kqs, s.dec, s.sc, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_q3_K_q8_1_apply(y, kqs, s.dec, s.sc, s.d);
    }    static __device__ float apply_aln(const block_q8_1_aln * y, int kqs, const state & s) {
        return vec_dot_q3_K_q8_1_apply_aln(y, kqs, s.dec, s.sc, s.d);
    }
    static __device__ uint32_t consume_x(const void * vx, int kbx, int kqs) {
        const block_q3_K * bq3 = (const block_q3_K *) vx + kbx;
        const int bq8_offset = QR3_K * (kqs / (QI3_K/2));
        const int scale_offset = kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1/2);
        uint32_t a = (uint32_t) get_int_b2(bq3->qs, kqs);
        a ^= (uint32_t) (~get_int_b2(bq3->hmask, kqs % (QI3_K/2)) >> bq8_offset);
#pragma unroll
        for (int i = 0; i < QR3_K; ++i) {
            const int isc = scale_offset + 2*i;
            a ^= bq3->scales[isc % (QK_K/32)];
            a ^= bq3->scales[(QK_K/32) + isc % (QK_K/64)];
        }
        a ^= (uint32_t) *(const uint16_t *) &bq3->d;
        return a;
    }
    static __device__ uint32_t consume_y(const block_q8_1 * y, int kqs) {
        const int bq8_offset = QR3_K * (kqs / (QI3_K/2));
        uint32_t a = 0;
#pragma unroll
        for (int i = 0; i < QR3_K; ++i) {
            const block_q8_1 * bq8i = y + bq8_offset + i;
            a ^= (uint32_t) get_int_b4(bq8i->qs, kqs % QI8_1);
            a ^= (uint32_t) ((const int *) bq8i)[0];
        }
        return a;
    }

    struct yw { int u[QR3_K]; int ds[QR3_K]; };
    static __device__ yw preload(const block_q8_1 * y, int kqs) {
        const int bq8_offset = QR3_K * (kqs / (QI3_K/2));
        yw w;
#pragma unroll
        for (int i = 0; i < QR3_K; ++i) {
            const block_q8_1 * bq8i = y + bq8_offset + i;
            w.u[i]  = get_int_b4(bq8i->qs, kqs % QI8_1);
            w.ds[i] = ((const int *) bq8i)[0];
        }
        return w;
    }
    static __device__ float apply_u(const yw & w, int kqs, const state & s) {
        return vec_dot_q3_K_q8_1_apply_u(w.u, w.ds, s.dec, s.sc, s.d);
    }
};

// GCN schedule clone: nwarps=2, rows_per_cuda_block=2 (ncols_dst 2..4),
// block 64x2, kbx/kqs/stride exactly as mul_mat_vec_q computes them.
// ARM 0 = base (shipped per-token vec_dot), 1 = share (shipped split pair),
// 2 = aln (shipped *_apply_aln twins over the 48-byte layout),
// 3 = cfull (consume x+y), 4 = cx (consume x), 5 = cy (consume y),
// 6 = wide (A2 x-operand wide loads; share-wide, or base-wide when
//     ops::wide_is_base - the iq4_xs served path).
template <ggml_type GT, int T, int ARM>
__global__ __launch_bounds__(128, 1) static void gemv_share_t(
        const void * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const int blocks_per_row, const int stride_col_y) {
    using ops = share_ops<GT>;
    constexpr int nwarps = 2;
    constexpr int rpb    = 2;
    constexpr int warp_size = 64;
    constexpr int qi  = ops::qi;
    constexpr int vdr = ops::vdr;
    constexpr int qk  = ops::qk;
    constexpr int kstep = vdr*nwarps*warp_size/qi;

    const int tid  = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = rpb*blockIdx.x;
    const int kbx0 = tid / (qi/vdr);
    const int kqs  = vdr * (tid % (qi/vdr));

    const block_q8_1     * yb = (const block_q8_1     *) vy;
    const block_q8_1_aln * ya = (const block_q8_1_aln *) vy;

    float tmp[T][rpb];
#pragma unroll
    for (int j = 0; j < T; ++j) {
#pragma unroll
        for (int i = 0; i < rpb; ++i) tmp[j][i] = 0.0f;
    }

    for (int kbx = kbx0; kbx < blocks_per_row; kbx += kstep) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx
        if constexpr (ARM == 0) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    tmp[j][i] += ops::base(vx, yb + j*stride_col_y + kby,
                        row0*blocks_per_row + i*blocks_per_row + kbx, kqs);
                }
            }
        } else if constexpr (ARM == 1) {
            typename ops::state s[rpb];
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                ops::decode(vx, (row0 + i)*blocks_per_row + kbx, kqs, s[i]);
            }
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    tmp[j][i] += ops::apply(yb + j*stride_col_y + kby, kqs, s[i]);
                }
            }
        } else if constexpr (ARM == 2) {
            static_assert(ops::has_aln, "aln arm needs aln twins");
            typename ops::state s[rpb];
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                ops::decode(vx, (row0 + i)*blocks_per_row + kbx, kqs, s[i]);
            }
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    tmp[j][i] += ops::apply_aln(ya + j*stride_col_y + kby, kqs, s[i]);
                }
            }
        } else if constexpr (ARM == 3) {
            // cfull: the share arm's exact x and y loads, xor instead of decode+dp4a
            uint32_t cxv[rpb];
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                cxv[i] = ops::consume_x(vx, (row0 + i)*blocks_per_row + kbx, kqs);
            }
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    tmp[j][i] += __int_as_float((int)(cxv[i] ^ ops::consume_y(yb + j*stride_col_y + kby, kqs)));
                }
            }
        } else if constexpr (ARM == 4) {
            // cx: x-operand loads only
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                tmp[0][i] += __int_as_float((int)ops::consume_x(vx, (row0 + i)*blocks_per_row + kbx, kqs));
            }
        } else if constexpr (ARM == 5) {
            // cy: y-operand loads only
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    tmp[j][i] += __int_as_float((int)ops::consume_y(yb + j*stride_col_y + kby, kqs));
                }
            }
        } else if constexpr (ARM == 6) {
            if constexpr (ops::wide_is_base) {
#pragma unroll
                for (int j = 0; j < T; ++j) {
#pragma unroll
                    for (int i = 0; i < rpb; ++i) {
                        tmp[j][i] += ops::base_wide(vx, yb + j*stride_col_y + kby,
                            row0*blocks_per_row + i*blocks_per_row + kbx, kqs);
                    }
                }
            } else {
                static_assert(ops::has_wide, "wide arm needs wide twins");
                typename ops::state s[rpb];
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    ops::decode_wide(vx, (row0 + i)*blocks_per_row + kbx, kqs, s[i]);
                }
#pragma unroll
                for (int j = 0; j < T; ++j) {
#pragma unroll
                    for (int i = 0; i < rpb; ++i) {
                        tmp[j][i] += ops::apply(yb + j*stride_col_y + kby, kqs, s[i]);
                    }
                }
            }
        } else if constexpr (ARM == 7) {
            // share + row-shared y loads (A3 lever a): preload each token's y
            // words once, apply per row from registers (bit-exact: identical
            // values and per-row accumulation order)
            typename ops::state s[rpb];
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                ops::decode(vx, (row0 + i)*blocks_per_row + kbx, kqs, s[i]);
            }
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const auto w = ops::preload(yb + j*stride_col_y + kby, kqs);
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    tmp[j][i] += ops::apply_u(w, kqs, s[i]);
                }
            }
        }
    }

    __shared__ float red[nwarps-1][T][rpb][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < T; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) red[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < T; ++j) {
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
            for (int l = 0; l < nwarps-1; ++l) tmp[j][i] += red[l][j][i][threadIdx.x];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) tmp[j][i] += __shfl_down(tmp[j][i], off, 64);
        }
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
        if (threadIdx.x < rpb) dst[(row0 + threadIdx.x)*T + j] = tmp[j][threadIdx.x];
    }
}

// -------------------------------------------------------- producer bench ---
// verbatim clone of the quantize_q8_1 kernel body from ggml-cuda/quantize.cu
// in both store layouts (legacy 36 B block_q8_1, aln 48 B block_q8_1_aln) on
// the served T=4 ffn activation shape (K=5120, MATRIX_ROW_PADDING-padded).
// VALUE memcmp legacy-vs-aln + per-arm timing: the producer cost delta of the
// aln emit (same stores, different addresses).
// MODE 0 = legacy 36 B only, 1 = aln 48 B only, 2 = DUAL emit (the shipped
// GGML_CUDA_MMVQ_ALN producer: legacy region at +0, aln region at +aln_off)
template <int MODE>
__global__ __launch_bounds__(256, 1) static void quantize_q8_1_clone(
        const float * x_ptr, void * vy_ptr,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2, const int64_t aln_off = 0) {
    const float * GGML_CUDA_RESTRICT x  = x_ptr;
    void        * GGML_CUDA_RESTRICT vy = vy_ptr;
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i3*s03 + i2*s02 + i1*s01 + i0] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    if (MODE == 2) {
        block_q8_1     * y  = (block_q8_1 *) vy;
        block_q8_1_aln * ya = (block_q8_1_aln *) ((char *) vy + aln_off);
        y[ib].qs[iqs]  = q;
        ya[ib].qs[iqs] = q;
        if (iqs > 0) {
            return;
        }
        y[ib].ds  = make_half2(d, sum);
        ya[ib].ds = make_half2(d, sum);
    } else if (MODE == 1) {
        block_q8_1_aln * y = (block_q8_1_aln *) vy;
        y[ib].qs[iqs] = q;
        if (iqs > 0) {
            return;
        }
        y[ib].ds = make_half2(d, sum);
    } else {
        block_q8_1 * y = (block_q8_1 *) vy;
        y[ib].qs[iqs] = q;
        if (iqs > 0) {
            return;
        }
        y[ib].ds = make_half2(d, sum);
    }
}

static void run_producer_bench(int niter, int reps) {
    // served ffn_up/gate src1 shape: K = 5120 x T = 4 tokens
    constexpr int K = 5120, T = 4;
    const int64_t ne0 = GGML_PAD(K, MATRIX_ROW_PADDING);
    const int64_t nblocks = T * ne0 / QK8_1;

    float * dx; void * dy36; void * dydual;
    CHECK(hipMalloc(&dx, (size_t)K*T*sizeof(float)));
    CHECK(hipMalloc(&dy36, (size_t)nblocks*36));
    CHECK(hipMalloc(&dydual, (size_t)nblocks*(36+48)));
    const int64_t aln_off = (int64_t)nblocks*36;
    {
        std::vector<float> hx((size_t)K*T);
        for (size_t i = 0; i < hx.size(); ++i) {
            hx[i] = ((float)((i*2654435761u) % 2001) - 1000.0f) / 512.0f;
        }
        CHECK(hipMemcpy(dx, hx.data(), hx.size()*sizeof(float), hipMemcpyHostToDevice));
    }

    const int64_t block_num_x = (ne0 + 256 - 1) / 256;
    dim3 grid((unsigned)block_num_x, T, 1);
    dim3 blk(256, 1, 1);

    // value gate: the dual-emit aln region must hold identical qs/ds bytes
    // per block as the legacy region of the same buffer
    CHECK(hipMemset(dydual, 0xAA, nblocks*(36+48)));
    quantize_q8_1_clone<0><<<grid, blk>>>(dx, dy36, K, K, K, K, ne0, T, make_uint3(1,1,1));
    quantize_q8_1_clone<2><<<grid, blk>>>(dx, dydual, K, K, K, K, ne0, T, make_uint3(1,1,1), aln_off);
    CHECK(hipDeviceSynchronize());
    {
        std::vector<uint8_t> a(nblocks*48), b(nblocks*36);
        CHECK(hipMemcpy(a.data(), (char *) dydual + aln_off, a.size(), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(), dydual, b.size(), hipMemcpyDeviceToHost));
        size_t bad = 0;
        for (int64_t i = 0; i < nblocks; ++i) {
            // compare ds(4 B at +0) and qs(32 B at +16); pad stays unwritten
            const uint8_t * pa = a.data() + i*48;
            const uint8_t * pb = b.data() + i*36;
            if (memcmp(pa, pb, 4) != 0 || memcmp(pa + 16, pb + 4, 32) != 0) {
                ++bad;
            }
        }
        printf("== producer emit (quantize_q8_1 clone, K=%d T=%d): %lld blocks, dual-region value memcmp %s\n",
               K, T, (long long)nblocks, bad == 0 ? "IDENTICAL" : "MISMATCH - REFUSED");
        if (bad != 0) {
            printf("producer value gate FAILED\n");
            exit(1);
        }
    }

    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    const char * armname[3] = { "legacy", "aln-only", "dual" };
    double med[3] = {0, 0, 0};
    for (int arm = 0; arm < 3; ++arm) {
        for (int w = 0; w < 200; ++w) {
            if (arm == 0)      quantize_q8_1_clone<0><<<grid, blk>>>(dx, dy36, K, K, K, K, ne0, T, make_uint3(1,1,1));
            else if (arm == 1) quantize_q8_1_clone<1><<<grid, blk>>>(dx, (char *) dydual + aln_off, K, K, K, K, ne0, T, make_uint3(1,1,1));
            else               quantize_q8_1_clone<2><<<grid, blk>>>(dx, dydual, K, K, K, K, ne0, T, make_uint3(1,1,1), aln_off);
        }
        CHECK(hipDeviceSynchronize());
        std::vector<double> us;
        for (int r = 0; r < reps; ++r) {
            CHECK(hipEventRecord(e0));
            for (int i = 0; i < niter; ++i) {
                if (arm == 0)      quantize_q8_1_clone<0><<<grid, blk>>>(dx, dy36, K, K, K, K, ne0, T, make_uint3(1,1,1));
                else if (arm == 1) quantize_q8_1_clone<1><<<grid, blk>>>(dx, (char *) dydual + aln_off, K, K, K, K, ne0, T, make_uint3(1,1,1));
                else               quantize_q8_1_clone<2><<<grid, blk>>>(dx, dydual, K, K, K, K, ne0, T, make_uint3(1,1,1), aln_off);
            }
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, e0, e1));
            us.push_back(1000.0*ms/niter);
        }
        std::sort(us.begin(), us.end());
        med[arm] = us[us.size()/2];
    }
    printf("   legacy emit %.2f us | aln-only emit %.2f us (%+.1f%%) | dual emit %.2f us (%+.1f%%)\n",
           med[0], med[1], (med[1]/med[0] - 1.0)*100.0, med[2], (med[2]/med[0] - 1.0)*100.0);

    CHECK(hipFree(dx)); CHECK(hipFree(dy36)); CHECK(hipFree(dydual));
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
}

// ------------------------------------------------------------------ main ---

struct TypeDef {
    const char * name;
    int          id;      // share_ops dispatch tag
    long         off;     // gguf tensor byte offset (offset-delta law)
    long         len;     // tensor byte len = next offset - offset
    int          K, N, bs;
};

struct Cfg { const char * name; int T; int arm; bool dup; };

template <ggml_type GT>
static int run_type(const TypeDef & td, const uint8_t * wsrc, int niter, int reps) {
    using ops = share_ops<GT>;
    const int K = td.K, N = td.N;
    const int blocks_per_row = K / ops::qk;
    const int nq8 = K / 32;
    const int STRIDE_COL_Y = nq8;
    if ((long)blocks_per_row * td.bs * N != td.len) {
        printf("  %s TUNE MISMATCH: %ld vs %ld - refusing (offset-delta law)\n",
               td.name, (long)blocks_per_row * td.bs * N, td.len);
        return 1;
    }

    const int TMAX = 4;
    void * dw; block_q8_1 * dy; block_q8_1_aln * dya; float * dd; float * dc;
    CHECK(hipMalloc(&dw, td.len + 16));   // +16: wide twins may read up to 8B past a block window
    CHECK(hipMalloc(&dy, TMAX * nq8 * sizeof(block_q8_1)));
    CHECK(hipMalloc(&dya, TMAX * nq8 * sizeof(block_q8_1_aln)));
    CHECK(hipMalloc(&dd, (size_t)N * TMAX * sizeof(float)));
    CHECK(hipMalloc(&dc, (size_t)N * TMAX * sizeof(float)));   // consume-arm dst (garbage)
    CHECK(hipMemcpy(dw, wsrc, td.len, hipMemcpyHostToDevice));

    // synthetic valid q8_1 as RAW BYTES, per-token distinct, standard 36 B
    // layout (ds.x@0, ds.y@2, qs@4)
    std::vector<uint8_t> hq(TMAX * nq8 * 36, 0);
    // 48-byte aligned twin with IDENTICAL values (ds@0, pad zeroed, qs@16)
    std::vector<uint8_t> hqa(TMAX * nq8 * 48, 0);
    for (int j = 0; j < TMAX; ++j) {
        for (int i = 0; i < nq8; ++i) {
            uint8_t * b = hq.data() + ((size_t)j*nq8 + i)*36;
            const uint16_t dx = 0x3C00, dyv = 0x5400;      // fp16(1.0), fp16(64.0)
            memcpy(b + 0, &dx, 2); memcpy(b + 2, &dyv, 2);
            for (int t = 0; t < 32; ++t) {
                b[4 + t] = (uint8_t)(int8_t)((j*1009 + i*37 + t*11) % 255 - 127);
            }
            uint8_t * a = hqa.data() + ((size_t)j*nq8 + i)*48;
            memcpy(a + 0, b + 0, 4);                        // ds
            memcpy(a + 16, b + 4, 32);                      // qs
        }
    }
    CHECK(hipMemcpy(dy, hq.data(), hq.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dya, hqa.data(), hqa.size(), hipMemcpyHostToDevice));

    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));

    Cfg cfgs[] = {
        {"t1_base", 1, 0, false}, {"t1_aln",  1, 2, false},
        {"t2_base", 2, 0, false}, {"t3_base", 3, 0, false},
        {"t4_base", 4, 0, false}, {"t4_base2", 4, 0, true},
        {"t2_share", 2, 1, false}, {"t3_share", 3, 1, false},
        {"t4_share", 4, 1, false}, {"t4_share2", 4, 1, true},
        {"t2_aln", 2, 2, false}, {"t3_aln", 3, 2, false},
        {"t4_aln", 4, 2, false}, {"t4_aln2", 4, 2, true},
        {"t4_cfull", 4, 3, false}, {"t4_cfull2", 4, 3, true},
        {"t4_cx", 4, 4, false}, {"t4_cy", 4, 5, false},
        {"t2_wide", 2, 6, false}, {"t3_wide", 3, 6, false},
        {"t4_wide", 4, 6, false}, {"t4_wide2", 4, 6, true},
        {"t2_s2r", 2, 7, false}, {"t3_s2r", 3, 7, false},
        {"t4_s2r", 4, 7, false}, {"t4_s2r2", 4, 7, true},
    };
    const int NC = (int)(sizeof(cfgs)/sizeof(cfgs[0]));
    dim3 grid((N + 1) / 2), block(64, 2);
    double rep_us[NC][8] = {{0}};
    int fail = 0;
    std::vector<float> dump[TMAX+1];      // base-arm dst copy per T (memcmp ref)
    std::vector<float> cfdump;            // cfull dst garbage (determinism ref)
    double t4_base = 0, t4_share = 0, t4_aln = 0, t4_wide = 0;
    double t4_cfull = 0, t4_cx = 0, t4_cy = 0, t4_s2r = 0;
    double worst_spread = 0.0;

    // clock soak: land timing at the sustained-load steady state (the served
    // condition); kills the rep0-fast drift class seen in session 1
    {
        const int soak = 500;
        for (int w = 0; w < soak; ++w) {
            gemv_share_t<GT,4,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
        }
        CHECK(hipDeviceSynchronize());
    }

    for (int rep = 0; rep < reps; ++rep) {
        for (int ic = 0; ic < NC; ++ic) {
            if (cfgs[ic].arm == 2 && !ops::has_aln) continue; // kept-off type: no aln twins
            if (cfgs[ic].arm == 6 && !ops::has_wide) continue; // no wide twins for this type
            const int T = cfgs[ic].T, ARM = cfgs[ic].arm;
            const bool consume = ARM == 3 || ARM == 4 || ARM == 5;
            float * dout = consume ? dc : dd;
            auto launch = [&]() {
                if (consume) {
                    if (ARM == 3)      gemv_share_t<GT,4,3><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                    else if (ARM == 4) gemv_share_t<GT,4,4><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                    else               gemv_share_t<GT,4,5><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                    return;
                }
                if (ARM == 6) {
                    if constexpr (ops::has_wide) {
                        if (T == 1)      gemv_share_t<GT,1,6><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                        else if (T == 2) gemv_share_t<GT,2,6><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                        else if (T == 3) gemv_share_t<GT,3,6><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                        else             gemv_share_t<GT,4,6><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                    }
                    return;
                }
                if (ARM == 7) {
                    if (T == 2)      gemv_share_t<GT,2,7><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                    else if (T == 3) gemv_share_t<GT,3,7><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                    else             gemv_share_t<GT,4,7><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                    return;
                }
                if constexpr (ops::has_aln) {
                    if (ARM == 2) {
                        if (T == 1) gemv_share_t<GT,1,2><<<grid, block>>>(dw, dya, dout, blocks_per_row, STRIDE_COL_Y);
                        else if (T == 2) gemv_share_t<GT,2,2><<<grid, block>>>(dw, dya, dout, blocks_per_row, STRIDE_COL_Y);
                        else if (T == 3) gemv_share_t<GT,3,2><<<grid, block>>>(dw, dya, dout, blocks_per_row, STRIDE_COL_Y);
                        else             gemv_share_t<GT,4,2><<<grid, block>>>(dw, dya, dout, blocks_per_row, STRIDE_COL_Y);
                        return;
                    }
                }
                if (T == 1) { if (ARM == 0) gemv_share_t<GT,1,0><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y); }
                else if (T == 2) { if (ARM == 0) gemv_share_t<GT,2,0><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                              else          gemv_share_t<GT,2,1><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y); }
                else if (T == 3) { if (ARM == 0) gemv_share_t<GT,3,0><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                                   else          gemv_share_t<GT,3,1><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y); }
                else { if (ARM == 0) gemv_share_t<GT,4,0><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y);
                       else          gemv_share_t<GT,4,1><<<grid, block>>>(dw, dy, dout, blocks_per_row, STRIDE_COL_Y); }
            };

            if (rep == 0) {   // warmup + gates on first pass
                launch(); CHECK(hipDeviceSynchronize());
                std::vector<float> dev((size_t)N * T);
                CHECK(hipMemcpy(dev.data(), dout, dev.size()*sizeof(float), hipMemcpyDeviceToHost));
                if (consume) {
                    // no oracle (dst is xor garbage); cfull dup = determinism witness
                    if (ARM == 3 && cfgs[ic].dup) {
                        if (cfdump.size() == dev.size() &&
                            memcmp(cfdump.data(), dev.data(), dev.size()*sizeof(float)) == 0) {
                            printf("  %-10s in-run control IDENTICAL (determinism memcmp)\n", cfgs[ic].name);
                        } else {
                            printf("  %-10s determinism control MISMATCH\n", cfgs[ic].name);
                            fail = 1;
                        }
                    } else if (ARM == 3) {
                        cfdump = dev;
                    }
                } else if (ARM == 0) {
                    const std::vector<float> & ref = dump[T];
                    if (cfgs[ic].dup) {
                        if (ref.size() == dev.size() &&
                            memcmp(ref.data(), dev.data(), dev.size()*sizeof(float)) == 0) {
                            printf("  %-10s in-run control IDENTICAL (memcmp)\n", cfgs[ic].name);
                        } else {
                            printf("  %-10s in-run control MISMATCH - REFUSED\n", cfgs[ic].name);
                            fail = 1;
                            continue;
                        }
                    } else if (ref.empty()) {
                        dump[T] = dev;
                    }
                } else {
                    // share/aln/wide must be bit-identical to the base arm at the same T
                    const std::vector<float> & ref = dump[T];
                    if (ref.size() == dev.size() &&
                        memcmp(ref.data(), dev.data(), dev.size()*sizeof(float)) == 0) {
                        printf("  %-10s BITEXACT vs %s t%d_base (memcmp, %d rows x %d tokens)\n",
                               cfgs[ic].name, td.name, T, N, T);
                    } else {
                        size_t pos = 0;
                        for (; pos + 4 <= ref.size(); ++pos) {
                            if (memcmp(ref.data() + pos, dev.data() + pos, sizeof(float)) != 0) break;
                        }
                        float fr = 0.0f, fd = 0.0f;
                        if (pos < ref.size()) { memcpy(&fr, ref.data() + pos, 4); memcpy(&fd, dev.data() + pos, 4); }
                        printf("  %-10s BIT MISMATCH vs %s t%d_base - REFUSED (first diff at flat %zu/%zu: ref %.9g dev %.9g)\n",
                               cfgs[ic].name, td.name, T, pos, ref.size(), fr, fd);
                        fail = 1;
                        continue;
                    }
                }
            }
            CHECK(hipEventRecord(e0));
            for (int i = 0; i < niter; ++i) launch();
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, e0, e1));
            rep_us[ic][rep] = 1000.0 * ms / niter;
        }
    }

    // report: medians, spread gate, share-vs-base verdicts
    printf("  %-10s %4s %10s %10s %9s   %s\n", "config", "T", "us/call", "GB/s", "spread", "per-rep us");
    for (int ic = 0; ic < NC; ++ic) {
        if (cfgs[ic].arm == 2 && !ops::has_aln) continue;
        std::vector<double> v(rep_us[ic], rep_us[ic] + reps);
        std::sort(v.begin(), v.end());
        const double med = v[reps/2];
        const double spread = reps > 1 ? (v[reps-1] - v[0]) / med * 100.0 : 0.0;
        worst_spread = std::max(worst_spread, spread);
        if (strcmp(cfgs[ic].name, "t4_base") == 0)  t4_base  = med;
        if (strcmp(cfgs[ic].name, "t4_share") == 0) t4_share = med;
        if (strcmp(cfgs[ic].name, "t4_aln") == 0)   t4_aln   = med;
        if (strcmp(cfgs[ic].name, "t4_wide") == 0)  t4_wide  = med;
        if (strcmp(cfgs[ic].name, "t4_cfull") == 0) t4_cfull = med;
        if (strcmp(cfgs[ic].name, "t4_cx") == 0)    t4_cx    = med;
        if (strcmp(cfgs[ic].name, "t4_cy") == 0)    t4_cy    = med;
        if (strcmp(cfgs[ic].name, "t4_s2r") == 0)    t4_s2r   = med;
        const double bytes = (double)td.len + (double)cfgs[ic].T*nq8*(cfgs[ic].arm == 2 ? 48 : 36)
                           + (double)N*cfgs[ic].T*4;
        printf("  %-10s %4d %10.1f %10.1f %8.2f%%   [",
               cfgs[ic].name, cfgs[ic].T, med, bytes / (med * 1e-6) / 1e9, spread);
        for (int rep = 0; rep < reps; ++rep) printf("%.1f%s", rep_us[ic][rep], rep + 1 < reps ? " " : "");
        printf("]\n");
    }
    if (t4_base > 0 && t4_share > 0) {
        printf("  %s T=4 share verdict: %+.1f%% vs base\n", td.name, (t4_share/t4_base - 1.0) * 100.0);
    }
    if (t4_base > 0 && t4_aln > 0) {
        printf("  %s T=4 aln verdict: %+.1f%% vs base, %+.1f%% vs share (the served increment)\n",
               td.name, (t4_aln/t4_base - 1.0) * 100.0, (t4_aln/t4_share - 1.0) * 100.0);
    }
    if (t4_wide > 0) {
        if (ops::wide_is_base) {
            printf("  %s T=4 wide(base u2 x-loads) verdict: %+.1f%% vs base\n",
                   td.name, (t4_wide/t4_base - 1.0) * 100.0);
        } else {
            printf("  %s T=4 wide verdict: %+.1f%% vs base, %+.1f%% vs share (the served increment)\n",
                   td.name, (t4_wide/t4_base - 1.0) * 100.0, (t4_wide/t4_share - 1.0) * 100.0);
        }
    }
    if (t4_cfull > 0) {
        const double bytes4 = (double)td.len + 4.0*nq8*36 + (double)N*4*4;
        printf("  %s T=4 consume ceilings: cfull %.1f us (%.1f GB/s) | cx %.1f us (%.1f GB/s) | cy %.1f us (%.1f GB/s)\n",
               td.name, t4_cfull, bytes4/(t4_cfull*1e-6)/1e9,
               t4_cx, bytes4/(t4_cx*1e-6)/1e9, t4_cy, bytes4/(t4_cy*1e-6)/1e9);
        printf("  %s T=4 decode+dp4a tax: %+.1f%% (share vs cfull); x-only / y-only load time ratio: %.2f\n",
               td.name, (t4_share/t4_cfull - 1.0) * 100.0, t4_cx/t4_cy);
    }
    if (t4_s2r > 0 && t4_share > 0) {
        printf("  %s T=4 s2r verdict: %+.1f%% vs base, %+.1f%% vs share (the served increment)\n",
               td.name, (t4_s2r/t4_base - 1.0) * 100.0, (t4_s2r/t4_share - 1.0) * 100.0);
    }
    printf("  %s session law: worst per-config spread %.2f%% %s (gate 1%%)\n",
           td.name, worst_spread, worst_spread > 1.0 ? "VOID" : "PASS");

    CHECK(hipFree(dw)); CHECK(hipFree(dy)); CHECK(hipFree(dya)); CHECK(hipFree(dd)); CHECK(hipFree(dc));
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    return fail;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <gguf> [niter=200] [reps=3] [type-filter]\n", argv[0]); return 1; }
    const int niter = argc > 2 ? atoi(argv[2]) : 200;
    const int reps  = argc > 3 ? atoi(argv[3]) : 3;
    const char * filter = argc > 4 ? argv[4] : nullptr;

    // served T=4 mix representatives (offset-delta law, ASCII-P1M census):
    //   iq3_xxs blk.0.ffn_gate [5120,17408] 98 B/block
    //   q4_K    blk.63.ffn_up  [5120,17408] 144 B/block
    //   iq4_xs  blk.4.ffn_gate [5120,17408] 136 B/block
    //   q5_K    blk.27.attn_q  [5120,12288] 176 B/block
    //   q6_K    output.weight  [5120,129272] 210 B/block (served T=4 lm_head)
    //   q3_K    blk.54.ffn_gate [5120,17408] 110 B/block
    const TypeDef types[] = {
        {"iq3_xxs", 0, 969405216L,   34119680L, 5120, 17408, 98},
        {"q4_K",    1, 12025224992L, 50135040L, 5120, 17408, 144},
        {"iq4_xs",  2, 1595404832L,  47349760L, 5120, 17408, 136},
        {"q5_K",    3, 5220080544L,  43253760L, 5120, 12288, 176},
        {"q6_K",    4, 4637536L,    542942400L, 5120, 129272, 210},
        {"q3_K",    5, 10469424416L, 38297600L, 5120, 17408, 110},
        // the shipped covered type, measured on the REAL rpb=2 schedule
        {"iq3_s",   6, 4301102016L,  38297600L, 5120, 17408, 110},
    };

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { printf("open gguf failed\n"); return 1; }
    const long maxend = 12025224992L + 50135040L;
    void * mm = mmap(nullptr, maxend, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm == MAP_FAILED) { printf("mmap failed\n"); return 1; }

    int fail = 0;
    run_producer_bench(niter, reps);
    for (const auto & td : types) {
        if (filter && strcmp(td.name, filter) != 0) continue;
        printf("== %s [K=%d N=%d %d B/block] off=%ld len=%ld\n",
               td.name, td.K, td.N, td.bs, td.off, td.len);
        const uint8_t * wsrc = (const uint8_t *) mm + td.off;
        switch (td.id) {
            case 0: fail |= run_type<GGML_TYPE_IQ3_XXS>(td, wsrc, niter, reps); break;
            case 1: fail |= run_type<GGML_TYPE_Q4_K>(td, wsrc, niter, reps); break;
            case 2: fail |= run_type<GGML_TYPE_IQ4_XS>(td, wsrc, niter, reps); break;
            case 3: fail |= run_type<GGML_TYPE_Q5_K>(td, wsrc, niter, reps); break;
            case 4: fail |= run_type<GGML_TYPE_Q6_K>(td, wsrc, niter, reps); break;
            case 5: fail |= run_type<GGML_TYPE_Q3_K>(td, wsrc, niter, reps); break;
            case 6: fail |= run_type<GGML_TYPE_IQ3_S>(td, wsrc, niter, reps); break;
        }
    }
    printf("\ncontrol: t4_base2/t4_share2 are the in-run controls (bit + spread);\n");
    printf("base-vs-share deltas at same T are the A/B verdicts; session law 1%%.\n");
    return fail;
}
