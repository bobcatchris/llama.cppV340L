// T-band cell v1 (oracle-gated): mul_mat_vec_q iq3_s T=1..4 for gfx900 (Vega
// 10, wave64), die 3 dev cell. The 09-21 receipts closed the T=1 atom and
// schedule classes; this bench opens the never-benched T>1 band (MTP verify
// runs T=2..4 in served decode graphs).
//
// Schedule clone (GCN table, iq3_s traits: qk=256, qi=16, vdr=2):
//   nwarps=2 (block 64x2), rows_per_block=1, kbx = tid/(qi/vdr) = tid/8,
//   kqs = vdr*(tid % (qi/vdr)) = 2*(tid&7), blocks_per_iter = 16,
//   kby = kbx*8, y block for token j = j*stride_col_y + kby, stride_col_y=160.
//   dst[j*stride_col_dst + row], stride_col_dst = N (column-major T-column).
//
// Arms per T (all bit-exact; same per-j integer sequence as the shipped math):
//   base  : shipped structure - vec_dot (8 LUT lookups + sign chain + 8 dp4a)
//           re-executed for every token j.
//   share : decode-once/dot-per-j. The grid lookups and sign application do
//           not depend on y; compute the 8 signed quads once per (kbx, lane),
//           then per j only the 8 x-int loads + 8 dp4a in the same order.
//   aln   : share + 48-byte aligned q8_1 relayout (ds@0, pad@4..16, qs@16,
//           uint4 operand loads instead of 8 scalar 4-byte reads). Arm-d
//           probe: operand stream alignment. Values identical -> oracle-equal.
//
// ORACLE: host C++ mirror of the shipped vec_dot_iq3_s_q8_1 per (row, j) over
// the same real weight bytes; every config must pass rows 0..63 x all T
// (rel err < 1e-4) BEFORE its timing counts.
//
// build:
//   /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP \
//     -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//     docs/amd-port/tests/bench_mmvq_t_gfx900.cu -o /tmp/bench_mmvq_t \
//     -L /media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin -lggml-base
// run (die 3 only, under the campaign lock):
//   HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin \
//     /tmp/bench_mmvq_t <gguf> [niter=200] [reps=3]
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

// aligned-relayout q8_1: 48 B/block, qs 16B-aligned for uint4 loads
struct q8_1_al48 {
    ggml_half2 ds;      // @0
    uint32_t   pad[3];  // @4..16
    int32_t    qs[8];   // @16..48
};

static inline __device__ int get_int_al48(const q8_1_al48 * b, int i) {
    const uint4 * p = (const uint4 *) b->qs;
    if (i < 4) {
        const uint4 v = p[0];
        return (i == 0) ? (int) v.x : (i == 1) ? (int) v.y : (i == 2) ? (int) v.z : (int) v.w;
    }
    const uint4 v = p[1];
    return (i == 4) ? (int) v.x : (i == 5) ? (int) v.y : (i == 6) ? (int) v.z : (int) v.w;
}

// pull the __device__ grid table to host memory (single source of truth)
__global__ static void pull_grid(uint32_t * out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 512) out[i] = iq3s_grid[i];
}

// ------------------------------------------- device vec_dot (shipped atom) --
// verbatim vec_dot_iq3_s_q8_1 clone; bq8_1 already offset to (j, kby) slot.
static __device__ __forceinline__ float vd_shipped(
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

// kernels --------------------------------------------------------------------
// ARM 0 = base (shipped j-loop), 1 = share (decode-once, harness copy),
// 2 = aln (share + 48B-aligned y), 3 = shipshare (the EXACT vecdotq.cuh
// vec_dot_iq3_s_q8_1_decode/apply pair wired into mmvq.cu by the env switch).
// T = 1..4. GCN schedule: NW2, kqs = 2*(tid&7), 16 kbx slots.
template <int T, int ARM>
__global__ __launch_bounds__(128, 1) static void gemv_iq3s_t(
        const block_iq3_s * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const int blocks_per_row, const int stride_col_y) {
    constexpr int nwarps = 2;
    const int warp_size = 64;
    const int tid  = warp_size*threadIdx.y + threadIdx.x;
    const int kbx0 = tid / 8;                 // qi/vdr = 8
    const int kqs  = 2*(tid & 7);             // vdr*(tid % 8)
    const int kstep = 16;                     // vdr*nwarps*warp_size/qi
    const int row = blockIdx.x;
    const block_iq3_s * h = vx + (size_t)row * blocks_per_row;
    const block_q8_1  * yb = (const block_q8_1 *) vy;
    const q8_1_al48   * ya = (const q8_1_al48  *) vy;

    float tmp[T];
#pragma unroll
    for (int j = 0; j < T; ++j) tmp[j] = 0.0f;

    for (int kbx = kbx0; kbx < blocks_per_row; kbx += kstep) {
        const block_iq3_s * b = h + kbx;
        const int kby = kbx*8;                // qk/QK8_1
        if (ARM == 3) {
            // exact shipped split functions (the mmvq.cu env-gated path)
            int dec[8]; int scale; float d;
            vec_dot_iq3_s_q8_1_decode((const void *)((const block_iq3_s *) vx + (size_t)row * blocks_per_row),
                                      kbx, kqs, dec, scale, d);
#pragma unroll
            for (int j = 0; j < T; ++j) {
                tmp[j] += vec_dot_iq3_s_q8_1_apply(yb + j*stride_col_y + kby, kqs, dec, scale, d);
            }
            continue;
        }
        if (ARM == 0) {
            for (int j = 0; j < T; ++j) {
                tmp[j] += vd_shipped(b, yb + j*stride_col_y + kby, kqs);
            }
        } else {
            // decode once: 8 signed quads, identical math/order to the atom
            const int2      qs_packed = make_int2(get_int_b2(b->qs, kqs + 0), get_int_b2(b->qs, kqs + 1));
            const uint8_t * qs        = (const uint8_t *) &qs_packed;
            const int qh = b->qh[kqs/2];
            const int       signs_packed_32 = get_int_b2(b->signs, kqs/2);
            const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;
            int dec[8];
#pragma unroll
            for (int l0 = 0; l0 < 8; l0 += 2) {
                const int g0 = iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
                const int g1 = iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
                const int s0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
                const int s1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
                dec[l0 + 0] = __vsub4(g0 ^ s0, s0);
                dec[l0 + 1] = __vsub4(g1 ^ s1, s1);
            }
            const int scale = 1 + 2*((b->scales[kqs/4] >> ((kqs << 1) & 0x04)) & 0x0F);
            const float d = __half2float(b->d);
            for (int j = 0; j < T; ++j) {
                int sumi = 0;
                float dsf;
                if (ARM == 1) {
                    const block_q8_1 * y = yb + j*stride_col_y + kby;
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        sumi = ggml_cuda_dp4a(dec[m], get_int_b4(y[kqs/2].qs, m), sumi);
                    }
                    dsf = __low2float(y[kqs/2].ds);
                } else {
                    const q8_1_al48 * y = ya + (size_t)j*stride_col_y + kby;
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        sumi = ggml_cuda_dp4a(dec[m], get_int_al48(y + kqs/2, m), sumi);
                    }
                    dsf = __low2float(y[kqs/2].ds);   // same 48B view as the operand reads
                }
                sumi *= scale;
                tmp[j] += d * dsf * sumi;
            }
        }
    }

    __shared__ float red[nwarps-1][T][64];
    if (threadIdx.y > 0) {
        for (int j = 0; j < T; ++j) red[threadIdx.y-1][j][threadIdx.x] = tmp[j];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    for (int l = 0; l < nwarps-1; ++l) {
        for (int j = 0; j < T; ++j) tmp[j] += red[l][j][threadIdx.x];
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
        for (int off = 32; off > 0; off >>= 1) tmp[j] += __shfl_down(tmp[j], off, 64);
        if (threadIdx.x == 0) dst[j*gridDim.x + row] = tmp[j];
    }
}

// -------------------------------------------------------- host oracle ------

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

static inline float h_fp16_to_fp32(uint16_t h) {
    const uint32_t s = h & 0x8000u, e = (h >> 10) & 0x1Fu, m = h & 0x03FFu;
    if (e == 0) return (s ? -1.0f : 1.0f) * (float)m * 5.9604644775390625e-08f;
    if (e == 31) return (m ? NAN : ((s ? -1.0f : 1.0f) * INFINITY));
    const uint32_t bits = (s << 16) | ((e + 112u) << 23) | (m << 13);
    float out; memcpy(&out, &bits, 4); return out;
}

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

struct Cfg { const char * name; int T; int arm; };

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <gguf> [niter=200] [reps=3]\n", argv[0]); return 1; }
    const int niter = argc > 2 ? atoi(argv[2]) : 200;
    const int reps  = argc > 3 ? atoi(argv[3]) : 3;

    // blk.21.ffn_up.weight [5120,17408] iq3_s, offset 4301102016,
    // len = next_offset - offset = 38,297,600 B (offset-delta law)
    const long off = 4301102016L, len = 38297600L;
    const int K = 5120, N = 17408;
    const int blocks_per_row = K / 256;     // 20
    const int nq8 = K / 32;                 // 160 blocks per token
    const int STRIDE_COL_Y = nq8;
    if ((long)blocks_per_row * 110 * N != len) {
        printf("TUNE MISMATCH: %ld vs %ld - refusing (offset-delta law)\n",
               (long)blocks_per_row * 110 * N, len);
        return 1;
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { printf("open gguf failed\n"); return 1; }
    void * mm = mmap(nullptr, len + off, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm == MAP_FAILED) { printf("mmap failed\n"); return 1; }
    const uint8_t * wsrc = (const uint8_t *) mm + off;

    const int TMAX = 4;
    block_iq3_s * dw; block_q8_1 * dy; q8_1_al48 * dya; float * dd;
    CHECK(hipMalloc(&dw, len));
    CHECK(hipMalloc(&dy, TMAX * nq8 * sizeof(block_q8_1)));
    CHECK(hipMalloc(&dya, TMAX * nq8 * sizeof(q8_1_al48)));
    CHECK(hipMalloc(&dd, TMAX * N * sizeof(float)));
    CHECK(hipMemcpy(dw, wsrc, len, hipMemcpyHostToDevice));

    // synthetic valid q8_1 as RAW BYTES, per-token distinct, standard 36B
    // layout (ds.x@0, ds.y@2, qs@4) + 48B aligned twin with identical values
    std::vector<uint8_t> hq(TMAX * nq8 * 36, 0);
    std::vector<uint8_t> hqa(TMAX * nq8 * 48, 0);
    for (int j = 0; j < TMAX; ++j) {
        for (int i = 0; i < nq8; ++i) {
            uint8_t * b = hq.data() + (j*nq8 + i)*36;
            uint8_t * a = hqa.data() + (j*nq8 + i)*48;
            const uint16_t dx = 0x3C00, dyv = 0x5400;      // fp16(1.0), fp16(64.0)
            memcpy(b + 0, &dx, 2); memcpy(b + 2, &dyv, 2);
            memcpy(a + 0, &dx, 2); memcpy(a + 2, &dyv, 2);
            for (int t = 0; t < 32; ++t) {
                const uint8_t v = (uint8_t)(int8_t)((j*1009 + i*37 + t*11) % 255 - 127);
                b[4 + t] = v;
                a[16 + t] = v;
            }
        }
    }
    CHECK(hipMemcpy(dy, hq.data(), hq.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dya, hqa.data(), hqa.size(), hipMemcpyHostToDevice));
    {   // raw-byte echo check on both layouts
        std::vector<uint8_t> echo(48, 0xAA);
        CHECK(hipMemcpy(echo.data(), dya, 48, hipMemcpyDeviceToHost));
        if (memcmp(echo.data(), hqa.data(), 48) != 0) {
            printf("YA UPLOAD MISMATCH - refusing\n");
            return 1;
        }
    }

    std::vector<uint32_t> h_grid(512);
    {
        uint32_t * dgrid = nullptr;
        CHECK(hipMalloc(&dgrid, 512 * sizeof(uint32_t)));
        pull_grid<<<1, 512>>>(dgrid);
        CHECK(hipMemcpy(h_grid.data(), dgrid, 512 * sizeof(uint32_t), hipMemcpyDeviceToHost));
        CHECK(hipFree(dgrid));
    }
    if (h_grid[0] != 0x01010101u || h_grid[322] != 0x07070b0bu) {
        printf("GRID PULL MISMATCH: [0]=%08x [322]=%08x - refusing\n", h_grid[0], h_grid[322]);
        return 1;
    }

    // host oracle: rows 0..63 x tokens 0..TMAX-1
    const int ORACLE_ROWS = 64;
    std::vector<double> ref(ORACLE_ROWS * TMAX);
    for (int j = 0; j < TMAX; ++j) {
        for (int r = 0; r < ORACLE_ROWS; ++r) {
            ref[j*ORACLE_ROWS + r] = host_row_dot((const uint8_t *)h_grid.data(),
                                                  wsrc + (size_t)r * blocks_per_row * 110,
                                                  hq.data() + (size_t)j*nq8*36, blocks_per_row);
        }
    }

    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));

    Cfg cfgs[] = {
        {"t1_base", 1, 0},
        {"t2_base", 2, 0}, {"t3_base", 3, 0}, {"t4_base", 4, 0},
        {"t2_share", 2, 1}, {"t3_share", 3, 1}, {"t4_share", 4, 1},
        {"t2_aln", 2, 2}, {"t3_aln", 3, 2}, {"t4_aln", 4, 2},
        {"t2_ship", 2, 3}, {"t3_ship", 3, 3}, {"t4_ship", 4, 3},
    };
    const int NC = (int)(sizeof(cfgs)/sizeof(cfgs[0]));
    dim3 grid(N), block(64, 2);
    double med_us[NC] = {0};
    std::vector<float> base_dump[TMAX+1];   // bit-identity reference per T

    // per-rep interleaving: rep-outer, config-inner (order kills era-drift)
    const bool diag = getenv("MMVQ_T_DIAG") != nullptr;   // oracle-only mode
    double rep_us[NC][8] = {{0}};
    for (int rep = 0; rep < (diag ? 1 : reps); ++rep) {
        for (int ic = 0; ic < NC; ++ic) {
            const int T = cfgs[ic].T, ARM = cfgs[ic].arm;
            auto launch = [&]() {
                if (T == 1) { if (ARM == 0) gemv_iq3s_t<1,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y); }
                else if (T == 2) {
                    if (ARM == 0) gemv_iq3s_t<2,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    if (ARM == 1) gemv_iq3s_t<2,1><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    if (ARM == 2) gemv_iq3s_t<2,2><<<grid, block>>>(dw, dya, dd, blocks_per_row, STRIDE_COL_Y);
                    if (ARM == 3) gemv_iq3s_t<2,3><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                } else if (T == 3) {
                    if (ARM == 0) gemv_iq3s_t<3,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    if (ARM == 1) gemv_iq3s_t<3,1><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    if (ARM == 2) gemv_iq3s_t<3,2><<<grid, block>>>(dw, dya, dd, blocks_per_row, STRIDE_COL_Y);
                    if (ARM == 3) gemv_iq3s_t<3,3><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                } else {
                    if (ARM == 0) gemv_iq3s_t<4,0><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    if (ARM == 1) gemv_iq3s_t<4,1><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                    if (ARM == 2) gemv_iq3s_t<4,2><<<grid, block>>>(dw, dya, dd, blocks_per_row, STRIDE_COL_Y);
                    if (ARM == 3) gemv_iq3s_t<4,3><<<grid, block>>>(dw, dy, dd, blocks_per_row, STRIDE_COL_Y);
                }
            };

            if (rep == 0) {   // warmup + ORACLE GATE on first pass
                launch(); CHECK(hipDeviceSynchronize());
                // dst is T columns x N rows; token j plane starts at j*N
                std::vector<float> dev((size_t)(T-1) * N + ORACLE_ROWS);
                CHECK(hipMemcpy(dev.data(), dd, dev.size() * sizeof(float), hipMemcpyDeviceToHost));
                double max_rel = 0.0; int worst_j = -1, worst_r = -1, nfail = 0;
                for (int j = 0; j < T; ++j) {
                    double jmax = 0.0; int jr = -1;
                    for (int r = 0; r < ORACLE_ROWS; ++r) {
                        const double got = dev[j*N + r];
                        const double want = ref[j*ORACLE_ROWS + r];
                        const double rel = fabs(got - want) / (fabs(want) + 1e-30);
                        if (!(jmax >= rel)) { jmax = rel; jr = r; }      // NaN-aware
                        if (!(rel < 1e-4)) ++nfail;                      // NaN fails the gate
                        if (!(max_rel >= rel)) { max_rel = rel; worst_j = j; worst_r = r; }
                    }
                    if (getenv("MMVQ_T_DIAG")) {
                        printf("  %s token %d: max rel %.3e at row %d (dev=%.9g ref=%.9g)\n",
                               cfgs[ic].name, j, jmax, jr, (double)dev[j*N + jr], ref[j*ORACLE_ROWS + jr]);
                    }
                }
                if (max_rel >= 1e-4) {
                    printf("%-10s ORACLE FAIL: %d/%d cells over 1e-4; worst (j=%d row=%d) dev=%.9g ref=%.9g rel=%.3e - REFUSED\n",
                           cfgs[ic].name, nfail, T*ORACLE_ROWS, worst_j, worst_r,
                           (double)dev[worst_j*N + worst_r], ref[worst_j*ORACLE_ROWS + worst_r], max_rel);
                    med_us[ic] = -1.0;
                    continue;
                }
                if (ARM == 0) {
                    base_dump[T] = dev;
                }
                if (ARM == 3) {   // shipped split pair must be BIT-identical to base
                    const std::vector<float> & bd = base_dump[T];
                    if (bd.size() == dev.size() && memcmp(bd.data(), dev.data(), dev.size()*sizeof(float)) == 0) {
                        printf("%-10s BITEXACT vs t%d_base (memcmp, %d rows x %d tokens)\n",
                               cfgs[ic].name, T, ORACLE_ROWS, T);
                    } else {
                        printf("%-10s BIT MISMATCH vs t%d_base - REFUSED\n", cfgs[ic].name, T);
                        med_us[ic] = -1.0;
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

    const double bytes[TMAX+1] = {0,
        (double)len + 1.0*nq8*36 + 1.0*N*4,
        (double)len + 2.0*nq8*36 + 2.0*N*4,
        (double)len + 3.0*nq8*36 + 3.0*N*4,
        (double)len + 4.0*nq8*36 + 4.0*N*4};
    const double floor_us = bytes[1] / 183.8e9 * 1e6;   // weight-dominated all T

    printf("shape: N=%d K=%d iq3_s, %.1f MB weights | T floor ~%.1f us at 183.8 GB/s\n",
           N, K, len/1e6, floor_us);
    printf("oracle: %d rows x T tokens, rel err gate 1e-4 | interleaved %d reps x %d iters\n",
           ORACLE_ROWS, reps, niter);
    printf("%-10s %4s %10s %10s %9s %9s   %s\n", "config", "T", "us/call", "GB/s", "x floor", "vs t1b", "oracle / per-rep us");
    double t1_us = 0;
    for (int ic = 0; ic < NC; ++ic) {
        if (cfgs[ic].T == 1 && cfgs[ic].arm == 0) {
            std::vector<double> v(rep_us[ic], rep_us[ic] + reps);
            std::sort(v.begin(), v.end());
            t1_us = v[reps/2];
        }
    }
    for (int ic = 0; ic < NC; ++ic) {
        if (med_us[ic] < 0) { printf("%-10s REFUSED (oracle)\n", cfgs[ic].name); continue; }
        std::vector<double> v(rep_us[ic], rep_us[ic] + reps);
        std::sort(v.begin(), v.end());
        med_us[ic] = v[reps/2];
        const double gbs = bytes[cfgs[ic].T] / (med_us[ic] * 1e-6) / 1e9;
        printf("%-10s %4d %10.1f %10.1f %9.2f %8.2fx   PASS [",
               cfgs[ic].name, cfgs[ic].T, med_us[ic], gbs, med_us[ic] / floor_us, med_us[ic] / t1_us);
        for (int rep = 0; rep < reps; ++rep) printf("%.1f%s", rep_us[ic][rep], rep + 1 < reps ? " " : "");
        printf("]\n");
    }
    printf("\ncontrol: t1_base is the in-run control vs 421.1 us banked (+-2%% gate);\n");
    printf("base-vs-share/aln deltas at same T are the A/B verdicts (>2%% = movement).\n");
    return 0;
}
