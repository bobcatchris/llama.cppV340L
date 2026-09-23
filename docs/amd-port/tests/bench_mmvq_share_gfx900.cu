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
//
// ORACLE: bit-exact f32 output vs the unshared path - device memcmp of the
// full dst (all N rows x T tokens), per (type, T), before any timing counts.
// In-run controls: t4b2/t4s2 duplicate the T=4 base/share configs; spread
// gate 1% per config (session law).
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

// per-type schedule constants (GCN table, GGML common defs) + the shipped
// vec_dot / split pair bindings
template <ggml_type GT> struct share_ops;

template <> struct share_ops<GGML_TYPE_IQ3_S> {
    static constexpr int qi = QI3_S, vdr = VDR_IQ3_S_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq3_s_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[8]; int scale; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_s_q8_1_decode(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_iq3_s_q8_1_apply(y, kqs, s.dec, s.scale, s.d);
    }
};

template <> struct share_ops<GGML_TYPE_IQ3_XXS> {
    static constexpr int qi = QI3_XXS, vdr = VDR_IQ3_XXS_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq3_xxs_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[8]; int ls; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq3_xxs_q8_1_decode(vx, kbx, kqs, s.dec, s.ls, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_iq3_xxs_q8_1_apply(y, kqs, s.dec, s.ls, s.d);
    }
};

template <> struct share_ops<GGML_TYPE_Q4_K> {
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
    }
};

template <> struct share_ops<GGML_TYPE_IQ4_XS> {
    static constexpr int qi = QI4_XS, vdr = VDR_IQ4_XS_Q8_1_MMVQ, qk = QK_K;
    static __device__ float base(const void * vx, const block_q8_1 * y, int kbx, int kqs) {
        return vec_dot_iq4_xs_q8_1(vx, y, kbx, kqs);
    }
    struct state { int dec[8]; int scale; float d; };
    static __device__ void decode(const void * vx, int kbx, int kqs, state & s) {
        vec_dot_iq4_xs_q8_1_decode(vx, kbx, kqs, s.dec, s.scale, s.d);
    }
    static __device__ float apply(const block_q8_1 * y, int kqs, const state & s) {
        return vec_dot_iq4_xs_q8_1_apply(y, kqs, s.dec, s.scale, s.d);
    }
};

template <> struct share_ops<GGML_TYPE_Q5_K> {
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
    }
};

template <> struct share_ops<GGML_TYPE_Q6_K> {
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
    }
};

template <> struct share_ops<GGML_TYPE_Q3_K> {
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
    }
};

// GCN schedule clone: nwarps=2, rows_per_cuda_block=2 (ncols_dst 2..4),
// block 64x2, kbx/kqs/stride exactly as mul_mat_vec_q computes them.
// ARM 0 = base (shipped per-token vec_dot), 1 = share (shipped split pair).
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

    const block_q8_1 * yb = (const block_q8_1 *) vy;

    float tmp[T][rpb];
#pragma unroll
    for (int j = 0; j < T; ++j) {
#pragma unroll
        for (int i = 0; i < rpb; ++i) tmp[j][i] = 0.0f;
    }

    for (int kbx = kbx0; kbx < blocks_per_row; kbx += kstep) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx
        if (ARM == 0) {
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
                ops::decode(vx, (row0 + i)*blocks_per_row + kbx, kqs, s[i]);
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
    void * dw; block_q8_1 * dy; float * dd;
    CHECK(hipMalloc(&dw, td.len));
    CHECK(hipMalloc(&dy, TMAX * nq8 * sizeof(block_q8_1)));
    CHECK(hipMalloc(&dd, (size_t)N * TMAX * sizeof(float)));
    CHECK(hipMemcpy(dw, wsrc, td.len, hipMemcpyHostToDevice));

    // synthetic valid q8_1 as RAW BYTES, per-token distinct, standard 36 B
    // layout (ds.x@0, ds.y@2, qs@4)
    std::vector<uint8_t> hq(TMAX * nq8 * 36, 0);
    for (int j = 0; j < TMAX; ++j) {
        for (int i = 0; i < nq8; ++i) {
            uint8_t * b = hq.data() + ((size_t)j*nq8 + i)*36;
            const uint16_t dx = 0x3C00, dyv = 0x5400;      // fp16(1.0), fp16(64.0)
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
        {"t2_base", 2, 0, false}, {"t3_base", 3, 0, false},
        {"t4_base", 4, 0, false}, {"t4_base2", 4, 0, true},
        {"t2_share", 2, 1, false}, {"t3_share", 3, 1, false},
        {"t4_share", 4, 1, false}, {"t4_share2", 4, 1, true},
    };
    const int NC = (int)(sizeof(cfgs)/sizeof(cfgs[0]));
    dim3 grid((N + 1) / 2), block(64, 2);
    double rep_us[NC][8] = {{0}};
    int fail = 0;
    std::vector<float> dump[TMAX+1];      // base-arm dst copy per T (memcmp ref)
    double t4_base = 0, t4_share = 0;
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
            const int T = cfgs[ic].T, ARM = cfgs[ic].arm;
            auto launch = [&]() {
                if (T == 2) { if (ARM == 0) gemv_share_t<GT,2,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                              else          gemv_share_t<GT,2,1><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y); }
                else if (T == 3) { if (ARM == 0) gemv_share_t<GT,3,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                                   else          gemv_share_t<GT,3,1><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y); }
                else { if (ARM == 0) gemv_share_t<GT,4,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                       else          gemv_share_t<GT,4,1><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y); }
            };

            if (rep == 0) {   // warmup + BIT-EXACT GATE on first pass
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
                    // share must be bit-identical to the base arm at the same T
                    const std::vector<float> & ref = dump[T];
                    if (ref.size() == dev.size() &&
                        memcmp(ref.data(), dev.data(), dev.size()*sizeof(float)) == 0) {
                        printf("  %-10s BITEXACT vs %s t%d_base (memcmp, %d rows x %d tokens)\n",
                               cfgs[ic].name, td.name, T, N, T);
                    } else {
                        printf("  %-10s BIT MISMATCH vs %s t%d_base - REFUSED\n",
                               cfgs[ic].name, td.name, T);
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
        std::vector<double> v(rep_us[ic], rep_us[ic] + reps);
        std::sort(v.begin(), v.end());
        const double med = v[reps/2];
        const double spread = reps > 1 ? (v[reps-1] - v[0]) / med * 100.0 : 0.0;
        worst_spread = std::max(worst_spread, spread);
        if (strcmp(cfgs[ic].name, "t4_base") == 0)  t4_base  = med;
        if (strcmp(cfgs[ic].name, "t4_share") == 0) t4_share = med;
        const double bytes = (double)td.len + (double)cfgs[ic].T*nq8*36 + (double)N*cfgs[ic].T*4;
        printf("  %-10s %4d %10.1f %10.1f %8.2f%%   [",
               cfgs[ic].name, cfgs[ic].T, med, bytes / (med * 1e-6) / 1e9, spread);
        for (int rep = 0; rep < reps; ++rep) printf("%.1f%s", rep_us[ic][rep], rep + 1 < reps ? " " : "");
        printf("]\n");
    }
    if (t4_base > 0 && t4_share > 0) {
        printf("  %s T=4 share verdict: %+.1f%% vs base\n", td.name, (t4_share/t4_base - 1.0) * 100.0);
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
