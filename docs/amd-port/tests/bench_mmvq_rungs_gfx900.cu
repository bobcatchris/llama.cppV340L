// MMVQ rung cell v1 (memcmp-gated): the W1 rung ladder re-derived for the
// four dominant served MMVQ types at the T=4 verify shape on gfx900.
//   type21 iq3_s, type18 iq3_xxs, type12 q4_K, type23 iq4_xs
// Instrument: docs/amd-port/tests/bench_mmvq_share_gfx900.cu v1 (schedule,
// oracle, session law) with rung arms added; build/run contract identical.
//
// LADDER RE-DERIVATION (served schedule: GCN nwarps=2, rpb=2, block 64x2,
// K=5120 -> blocks_per_row = 20, one kbx iteration per thread):
//   C1/C2 K-split: DEAD by geometry. There is no serial K-chain left (1
//     iteration for every type at K=5120) and the grid is many-wave
//     (N/2 = 8704 CTAs on 64 CUs), the class W1 measured DEAD for K-split.
//   A1 staging: mostly ILLEGAL by alignment law. Block strides 110/98/136 B
//     and the 36 B y stride keep the contiguous dword pairs misaligned for
//     dwordx2/x4 on half or all blocks; the y-side relayout is the W4 aln
//     NEGATIVE. Only iq4_xs qs (8 B aligned on every block) merges 4 -> 2.
//   B1 decode atom: SURVIVES for the LUT+sign types (iq3_s, iq3_xxs). The
//     shipped sign chain (__vcmpne4 + __vsub4 = byte-loop + sub_sat emu)
//     compiles to 28-30 VALU ops per sign pair on gfx900; the v_perm_b32
//     MSB-replication atom (probe_sign exhaustive ALL EXACT, probe_isa
//     census) builds the FF/00 masks in 2-3 ops and applies signs in 3:
//       s0 = perm(m | (m << 7), same, 0xBA98)   [iq3_s masks: bits 7,8,23,24]
//       s0 = perm(m * 0xF0, ...) / perm(m | m * 0x0E, ...)  [iq3_xxs even/odd]
//       dec = (g ^ s0) + (s0 & 0x01010101)      [grid bytes never 0, no carry]
//   q4_K: no arm - decode is nibble shift/mask (cheap), apply is 8 emulated
//     dp4a (6 VALU each, gfx900 ISA wall, 09-21 probe); residual = traffic
//     (W3 ceiling statement). Reproduce base/share only.
//
// ARMS per (type, T=4):
//   base  : EXACT shipped vec_dot per token (the unshared path of record)
//   share : EXACT shipped decode-once split pair (the served increment, W3)
//   perm  : share with the decode() replaced by the perm-atom twin (bit-exact
//           by construction: identical dec[]/scale/d, apply unchanged)
//   wide  : share with iq4_xs qs loads as 2x dwordx2 (same 4 dwords)
//   wshare: base + wide loads, no share (iq4_xs: share measured negative)
//
// ORACLE: bit-exact f32 output vs the base arm - host memcmp of the full dst
// (N rows x T tokens), per config, before any timing counts. In-run controls:
// t4_base2/t4_share2/t4_perm2 duplicate configs; 1% spread gate per config.
//
// build:
//   /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP \
//     -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//     docs/amd-port/tests/bench_mmvq_rungs_gfx900.cu -o /tmp/bench_mmvq_rungs \
//     -L /media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin -lggml-base
// run (die 3 only, under the campaign lock):
//   HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin \
//     /tmp/bench_mmvq_rungs <gguf> [niter=200] [reps=3] [type-filter]
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
static_assert(sizeof(block_q4_K)   == 144, "q4_K block size");
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block size");
static_assert(sizeof(block_iq3_s)  == 110, "iq3_s block size");

#if defined(GGML_USE_HIP)
// v_perm_b32 byte-select with the sign-extend (MSB replicate) mode bit set in
// every selector nibble: 0xBA98 = byte i selects byte i with mode.
static __device__ __forceinline__ int perm_msb_replicate(uint32_t f) {
    return (int)__builtin_amdgcn_perm(f, f, 0xBA98u);
}
#endif

// ------------------------------------------------------- perm-atom twins ---
// Bit-exact replacements for the decode() halves: same loads, same dec[]
// values (probe_sign exhaustive over all 256 grid entries x 256 sign bytes),
// so the shipped apply() runs unchanged.

static __device__ __forceinline__ void vec_dot_iq3_s_q8_1_decode_perm(
    const void * __restrict__ vbq, const int & kbx, const int & iqs, int * dec, int & scale, float & d) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int g0 = iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const int g1 = iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];

        const uint8_t sp8 = signs_packed_8[l0/2];
        // sign bits sit at positions 7,8,23,24 of the masked word (both mask
        // idioms); one shift+or puts all four on byte MSBs for perm mode-bit
        const uint32_t m0 = ((sp8 & 0x03) << 7) | ((sp8 & 0x0C) << 21);
        const uint32_t m1 = ((sp8 & 0x30) << 3) | ((sp8 & 0xC0) << 17);
        const int signs0 = perm_msb_replicate(m0 | (m0 << 7));
        const int signs1 = perm_msb_replicate(m1 | (m1 << 7));

        // per-byte -g where signed: (255-g)+1, grid bytes are never 0x00 so
        // the +1 never carries across bytes (exhaustive host proof)
        dec[l0 + 0] = (g0 ^ signs0) + (signs0 & 0x01010101);
        dec[l0 + 1] = (g1 ^ signs1) + (signs1 & 0x01010101);
    }

    scale = 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    d     = __half2float(bq3->d);
}

static __device__ __forceinline__ void vec_dot_iq3_xxs_q8_1_decode_perm(
    const void * __restrict__ vbq, const int & kbx, const int & iqs, int * dec, int & ls, float & d) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int g0 = iq3xxs_grid[q3[l0 + 0]];
        const int g1 = iq3xxs_grid[q3[l0 + 1]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const uint32_t me = signs & 0x08040201;   // isolated bits 0,9,18,27
        const uint32_t mo = signs & 0x80402010;   // isolated bits 4,13,22,31
        const int signs0 = perm_msb_replicate(me * 0xF0);
        const int signs1 = perm_msb_replicate(mo | (mo * 0x0E));

        dec[l0 + 0] = (g0 ^ signs0) + (signs0 & 0x01010101);
        dec[l0 + 1] = (g1 ^ signs1) + (signs1 & 0x01010101);
    }

    ls = aux32 >> 28;
    d  = __half2float(bq3->d);
}

// iq4_xs wide: the 4 contiguous qs dwords as two 8-byte loads (every block's
// qs is 8-byte aligned; 16-byte is not, so dwordx4 stays illegal)
static __device__ __forceinline__ void vec_dot_iq4_xs_q8_1_decode_wide(
    const void * __restrict__ vbq, const int & kbx, const int & iqs, int * dec, int & scale, float & d) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    const int2 q01 = *((const int2 *)((const uint8_t *)bq4->qs + 4*iqs));      // ints iqs+0, iqs+1
    const int2 q23 = *((const int2 *)((const uint8_t *)bq4->qs + 4*iqs) + 1);  // ints iqs+2, iqs+3

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = j < 2 ? (j == 0 ? q01.x : q01.y) : (j == 2 ? q23.x : q23.y);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        dec[j + 0] = v.x;
        dec[j + 4] = v.y;
    }

    scale = (((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4)) - 32;
    d     = __half2float(bq4->d);
}

// ------------------------------------------------------------ rung ops ----
template <ggml_type GT> struct rung_ops;

template <> struct rung_ops<GGML_TYPE_IQ3_S> {
    static constexpr int qi = QI3_S, vdr = VDR_IQ3_S_Q8_1_MMVQ, qk = QK_K;
    static constexpr bool has_perm = true, has_wide = false;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq3_s_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[8]; int scale; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_s_q8_1_decode(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ void decode_rung(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_s_q8_1_decode_perm(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_iq3_s_q8_1_apply(y, kqs, s.dec, s.scale, s.d);
    }
};

template <> struct rung_ops<GGML_TYPE_IQ3_XXS> {
    static constexpr int qi = QI3_XXS, vdr = VDR_IQ3_XXS_Q8_1_MMVQ, qk = QK_K;
    static constexpr bool has_perm = true, has_wide = false;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq3_xxs_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[8]; int ls; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_xxs_q8_1_decode(vx, kbx, kqs, s.dec, s.ls, s.d);
    }
    static __device__ void decode_rung(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_xxs_q8_1_decode_perm(vx, kbx, kqs, s.dec, s.ls, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_iq3_xxs_q8_1_apply(y, kqs, s.dec, s.ls, s.d);
    }
};

template <> struct rung_ops<GGML_TYPE_Q4_K> {
    static constexpr int qi = QI4_K, vdr = VDR_Q4_K_Q8_1_MMVQ, qk = QK_K;
    static constexpr bool has_perm = false, has_wide = false;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_q4_K_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[2*QR4_K]; int sc[QR4_K]; int m[QR4_K]; ggml_half2 dm; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_q4_K_q8_1_decode(vx, kbx, kqs, s.dec, s.sc, s.m, s.dm);
    }
    static __device__ void decode_rung(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_q4_K_q8_1_decode(vx, kbx, kqs, s.dec, s.sc, s.m, s.dm); // no rung arm; kept for template instantiation
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_q4_K_q8_1_apply(y, kqs, s.dec, s.sc, s.m, s.dm);
    }
};

template <> struct rung_ops<GGML_TYPE_IQ4_XS> {
    static constexpr int qi = QI4_XS, vdr = VDR_IQ4_XS_Q8_1_MMVQ, qk = QK_K;
    static constexpr bool has_perm = false, has_wide = true;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq4_xs_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[8]; int scale; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq4_xs_q8_1_decode(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ void decode_rung(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq4_xs_q8_1_decode_wide(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_iq4_xs_q8_1_apply(y, kqs, s.dec, s.scale, s.d);
    }
};

// GCN schedule clone: nwarps=2, rows_per_cuda_block=2, block 64x2.
// ARM: 0 = base, 1 = share (shipped decode), 2 = rung (perm/wide decode twin).
template <ggml_type GT, int T, int ARM>
__global__ __launch_bounds__(128, 1) static void gemv_rung_t(
        const void * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const int blocks_per_row, const int stride_col_y) {
    using ops = rung_ops<GT>;
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

    const block_q8_1 * yb = (const block_q8_1 *) vy;

    float tmp[T][rpb];
#pragma unroll
    for (int j = 0; j < T; ++j) {
#pragma unroll
        for (int i = 0; i < rpb; ++i) tmp[j][i] = 0.0f;
    }

    for (int kbx = kbx0; kbx < blocks_per_row; kbx += kstep) {
        const int kby = kbx * (qk/QK8_1);
        if constexpr (ARM == 0) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    tmp[j][i] += ops::base(vx, yb + j*stride_col_y + kby,
                        row0*blocks_per_row + i*blocks_per_row + kbx, kqs);
                }
            }
        } else {
            typename ops::state s[rpb];
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                if constexpr (ARM == 2) {
                    ops::decode_rung(vx, (row0 + i)*blocks_per_row + kbx, kqs, s[i]);
                } else {
                    ops::decode(vx, (row0 + i)*blocks_per_row + kbx, kqs, s[i]);
                }
            }
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    tmp[j][i] += ops::apply(yb + j*stride_col_y + kby, kqs, s[i]);
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

// ------------------------------------------------------------------ main ---

struct TypeDef {
    const char * name;
    int          id;
    long         off;
    long         len;
    int          K, N, bs;
};

struct Cfg { const char * name; int T; int arm; bool dup; bool on; };

template <ggml_type GT>
static int run_type(const TypeDef & td, const uint8_t * wsrc, int niter, int reps) {
    using ops = rung_ops<GT>;
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
    void * dw; block_q8_1 * dy; float * dd;
    CHECK(hipMalloc(&dw, td.len));
    CHECK(hipMalloc(&dy, TMAX * nq8 * sizeof(block_q8_1)));
    CHECK(hipMalloc(&dd, (size_t)N * TMAX * sizeof(float)));
    CHECK(hipMemcpy(dw, wsrc, td.len, hipMemcpyHostToDevice));

    std::vector<uint8_t> hq(TMAX * nq8 * 36, 0);
    for (int j = 0; j < TMAX; ++j) {
        for (int i = 0; i < nq8; ++i) {
            uint8_t * b = hq.data() + ((size_t)j*nq8 + i)*36;
            const uint16_t dx = 0x3C00, dyv = 0x5400;
            memcpy(b + 0, &dx, 2); memcpy(b + 2, &dyv, 2);
            for (int t = 0; t < 32; ++t) {
                b[4 + t] = (uint8_t)(int8_t)((j*1009 + i*37 + t*11) % 255 - 127);
            }
        }
    }
    CHECK(hipMemcpy(dy, hq.data(), hq.size(), hipMemcpyHostToDevice));

    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));

    Cfg cfgs[] = {
        {"t1_base",   1, 0, false, true },
        {"t4_base",   4, 0, false, true },
        {"t4_base2",  4, 0, true,  true },
        {"t4_share",  4, 1, false, true },
        {"t4_share2", 4, 1, true,  true },
        {"t4_rung",   4, 2, false, ops::has_perm || ops::has_wide},
        {"t4_rung2",  4, 2, true,  ops::has_perm || ops::has_wide},
    };
    const int NC = (int)(sizeof(cfgs)/sizeof(cfgs[0]));
    dim3 grid((N + 1) / 2), block(64, 2);
    double rep_us[NC][8] = {{0}};
    int fail = 0;
    std::vector<float> dump[TMAX+1];
    double t4_base = 0, t4_share = 0, t4_rung = 0;
    double worst_spread = 0.0;

    {
        const int soak = 500;
        for (int w = 0; w < soak; ++w) {
            gemv_rung_t<GT,4,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
        }
        CHECK(hipDeviceSynchronize());
    }

    for (int rep = 0; rep < reps; ++rep) {
        for (int ic = 0; ic < NC; ++ic) {
            if (!cfgs[ic].on) continue;
            const int T = cfgs[ic].T, ARM = cfgs[ic].arm;
            auto launch = [&]() {
                if (T == 1) {
                    if (ARM == 0)      gemv_rung_t<GT,1,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    else if (ARM == 1) gemv_rung_t<GT,1,1><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    else               gemv_rung_t<GT,1,2><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                } else {
                    if (ARM == 0)      gemv_rung_t<GT,4,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    else if (ARM == 1) gemv_rung_t<GT,4,1><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    else               gemv_rung_t<GT,4,2><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                }
            };

            if (rep == 0) {
                launch(); CHECK(hipDeviceSynchronize());
                std::vector<float> dev((size_t)N * T);
                CHECK(hipMemcpy(dev.data(), dd, dev.size()*sizeof(float), hipMemcpyDeviceToHost));
                if (ARM == 0) {
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
                    const std::vector<float> & ref = dump[T];
                    if (ref.size() == dev.size() &&
                        memcmp(ref.data(), dev.data(), dev.size()*sizeof(float)) == 0) {
                        printf("  %-10s BITEXACT vs t%d_base (memcmp, %d rows x %d tokens)\n",
                               cfgs[ic].name, T, N, T);
                    } else {
                        printf("  %-10s BIT MISMATCH vs t%d_base - REFUSED\n", cfgs[ic].name, T);
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

    printf("  %-10s %4s %10s %10s %9s   %s\n", "config", "T", "us/call", "GB/s", "spread", "per-rep us");
    for (int ic = 0; ic < NC; ++ic) {
        if (!cfgs[ic].on || rep_us[ic][0] == 0) continue;
        std::vector<double> v(rep_us[ic], rep_us[ic] + reps);
        std::sort(v.begin(), v.end());
        const double med = v[reps/2];
        const double spread = reps > 1 ? (v[reps-1] - v[0]) / med * 100.0 : 0.0;
        worst_spread = std::max(worst_spread, spread);
        if (strcmp(cfgs[ic].name, "t4_base")   == 0) t4_base  = med;
        if (strcmp(cfgs[ic].name, "t4_share")  == 0) t4_share = med;
        if (strcmp(cfgs[ic].name, "t4_rung")   == 0) t4_rung  = med;
        const double bytes = (double)td.len + (double)cfgs[ic].T*nq8*36
                           + (double)N*cfgs[ic].T*4;
        printf("  %-10s %4d %10.1f %10.1f %8.2f%%   [",
               cfgs[ic].name, cfgs[ic].T, med, bytes / (med * 1e-6) / 1e9, spread);
        for (int rep = 0; rep < reps; ++rep) printf("%.1f%s", rep_us[ic][rep], rep + 1 < reps ? " " : "");
        printf("]\n");
    }
    if (t4_base > 0 && t4_share > 0) {
        printf("  %s T=4 share verdict (W3 reproduction): %+.1f%% vs base\n", td.name, (t4_share/t4_base - 1.0) * 100.0);
    }
    if (t4_share > 0 && t4_rung > 0) {
        printf("  %s T=4 RUNG verdict (perm/wide on share): %+.1f%% vs share, %+.1f%% vs base\n",
               td.name, (t4_rung/t4_share - 1.0) * 100.0, (t4_rung/t4_base - 1.0) * 100.0);
    }
    printf("  %s session law: worst per-config spread %.2f%% %s (gate 1%%)\n",
           td.name, worst_spread, worst_spread > 1.0 ? "VOID" : "PASS");

    CHECK(hipFree(dw)); CHECK(hipFree(dy)); CHECK(hipFree(dd));
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    return fail;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <gguf> [niter=200] [reps=3] [type-filter]\n", argv[0]); return 1; }
    const int niter = argc > 2 ? atoi(argv[2]) : 200;
    const int reps  = argc > 3 ? atoi(argv[3]) : 3;
    const char * filter = argc > 4 ? argv[4] : nullptr;

    const TypeDef types[] = {
        {"iq3_xxs", 0, 969405216L,   34119680L, 5120, 17408, 98},
        {"q4_K",    1, 12025224992L, 50135040L, 5120, 17408, 144},
        {"iq4_xs",  2, 1595404832L,  47349760L, 5120, 17408, 136},
        {"iq3_s",   3, 4301102016L,  38297600L, 5120, 17408, 110},
    };

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { printf("open gguf failed\n"); return 1; }
    const long maxend = 12025224992L + 50135040L;
    void * mm = mmap(nullptr, maxend, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm == MAP_FAILED) { printf("mmap failed\n"); return 1; }

    int fail = 0;
    for (const auto & td : types) {
        if (filter && strcmp(td.name, filter) != 0) continue;
        printf("== %s [K=%d N=%d %d B/block] off=%ld len=%ld\n",
               td.name, td.K, td.N, td.bs, td.off, td.len);
        const uint8_t * wsrc = (const uint8_t *) mm + td.off;
        switch (td.id) {
            case 0: fail |= run_type<GGML_TYPE_IQ3_XXS>(td, wsrc, niter, reps); break;
            case 1: fail |= run_type<GGML_TYPE_Q4_K>(td, wsrc, niter, reps); break;
            case 2: fail |= run_type<GGML_TYPE_IQ4_XS>(td, wsrc, niter, reps); break;
            case 3: fail |= run_type<GGML_TYPE_IQ3_S>(td, wsrc, niter, reps); break;
        }
    }
    printf("\ncontrol: t4_base2/t4_share2/t4_rung2 are the in-run controls (bit + spread);\n");
    printf("same-T deltas vs base/share are the A/B verdicts; session law 1%%.\n");
    return fail;
}
