// W2 device cell: MMVQ GEMV variants for gfx900 (iq3_s class), die 3 dev cell.
// Mirrors mul_mat_vec_q structure for T=1 (wave64, rows_per_block=1, GCN table
// nwarps=2) on REAL GGUF weight bytes. Variants:
//   NW2 constant-LUT (as shipped) | NW2 LDS-LUT | NW4 constant | NW8 constant
// The kernel math is a verbatim copy of vec_dot_iq3_s_q8_1 (vecdotq.cuh) so a
// result delta is a schedule delta, not a math delta.
//
// build:
//   /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP \
//     -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//     docs/amd-port/tests/bench_mmvq_gfx900.cu -o /tmp/bench_mmvq
// run (die 3 only):
//   HIP_VISIBLE_DEVICES=3 /tmp/bench_mmvq <gguf> [niter]
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <hip/hip_runtime.h>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

// verbatim math from vecdotq.cuh, iq3_s
static __device__ __forceinline__ float vd_iq3s_const(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1,
        const int kbx, const int iqs) {
    bq3 += kbx;
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
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 4);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// LDS variant: identical math, grid read from __shared__ copy
static __device__ __forceinline__ float vd_iq3s_lds(
        const block_iq3_s * __restrict__ bq3, const block_q8_1 * __restrict__ bq8_1,
        const int kbx, const int iqs, const uint32_t * __restrict__ grid) {
    bq3 += kbx;
    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 4);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

template <int NWARPS, bool LDS, bool HAVE_LDS, bool BALANCED>
__global__ __launch_bounds__(64*NWARPS, 1) static void gemv_iq3s(
        const block_iq3_s * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        float * __restrict__ dst, const int blocks_per_row) {
    const int tid = 64*threadIdx.y + threadIdx.x;
    __shared__ uint32_t lut[256];
    if (HAVE_LDS) {
        for (int i = tid; i < 256; i += NWARPS*64) lut[i] = ((const uint32_t *) iq3s_grid)[i];
        __syncthreads();
    }
    float tmp = 0.0f;
    const int kqs = 2*(tid & 7);            // vdr=2, qi=16 -> 8 lanes-slots
    const int kbx0 = tid >> 3;
    const int kper_iter = 8*NWARPS;         // 8 kbx-slots per warp
    if (BALANCED) {
        // static balanced kbx assignment: slot s takes ceil-distributed blocks,
        // removing the partially-idle tail iteration of the strided loop
        const int nslot = kper_iter;
        const int base = kbx0;
        const int cnt = blocks_per_row / nslot + (base < (blocks_per_row % nslot) ? 1 : 0);
        int kbx = base;
        for (int i = 0; i < cnt; ++i, kbx += nslot) {
            tmp += LDS ? vd_iq3s_lds(vx, vy, kbx, kqs, lut) : vd_iq3s_const(vx, vy, kbx, kqs);
        }
    } else {
        for (int kbx = kbx0; kbx < blocks_per_row; kbx += kper_iter) {
            tmp += LDS ? vd_iq3s_lds(vx, vy, kbx, kqs, lut) : vd_iq3s_const(vx, vy, kbx, kqs);
        }
    }
    __shared__ float red[NWARPS > 1 ? NWARPS-1 : 1][64];
    if (threadIdx.y > 0) red[threadIdx.y-1][threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int l = 0; l < NWARPS-1; ++l) tmp += red[l][threadIdx.x];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) tmp += __shfl_down(tmp, off, 64);
    if (threadIdx.x == 0) dst[blockIdx.x] = tmp;
}

// oracle: host dot of one row slice vs device result (bit-different accumulation
// order is allowed within float tolerance; check rel err < 1e-4)
static double host_row_sum(const block_iq3_s * h, const block_q8_1 * y, int blocks_per_row, int row) {
    (void)h; (void)y; (void)blocks_per_row; (void)row;
    return 0.0;  // full host oracle needs iq3 decode in C++ - added when the kernel cell goes GREEN
}

// multi-row variant: each lane processes ROWS rows' worth of independent
// chains (ROWS x iterations), multiplying memory-level parallelism
template <int NWARPS, int ROWS>
__global__ __launch_bounds__(64*NWARPS, 1) static void gemv_iq3s_rows(
        const block_iq3_s * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        float * __restrict__ dst, const int blocks_per_row, const int nrows) {
    const int tid = 64*threadIdx.y + threadIdx.x;
    const int kqs = 2*(tid & 7);
    const int kbx0 = tid >> 3;
    const int nslot = 8*NWARPS;
    const int row0 = blockIdx.x * ROWS;
    float tmp[ROWS] = {0.0f};
    const int cnt = blocks_per_row / nslot + (kbx0 < (blocks_per_row % nslot) ? 1 : 0);
    for (int r = 0; r < ROWS; ++r) {
        const int row = row0 + r;
        if (row >= nrows) break;
        const block_iq3_s * h = vx + (size_t)row * blocks_per_row;
        int kbx = kbx0;
        for (int i = 0; i < cnt; ++i, kbx += nslot) tmp[r] += vd_iq3s_const(h, vy, kbx, kqs);
    }
    __shared__ float red[NWARPS > 1 ? NWARPS-1 : 1][ROWS][64];
    if (threadIdx.y > 0) {
        for (int r = 0; r < ROWS; ++r) red[threadIdx.y-1][r][threadIdx.x] = tmp[r];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int l = 0; l < NWARPS-1; ++l) tmp[r] += red[l][r][threadIdx.x];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) tmp[r] += __shfl_down(tmp[r], off, 64);
        if (threadIdx.x == 0 && row0 + r < nrows) dst[row0 + r] = tmp[r];
    }
}

// ILP variant: manual 2-way unroll - two independent kbx chains in flight
// (the runtime trip count prevents compiler unrolling, so each iteration
// otherwise serializes: load -> LUT -> decode -> dp4a)
template <int NWARPS>
__global__ __launch_bounds__(64*NWARPS, 1) static void gemv_iq3s_ilp2(
        const block_iq3_s * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        float * __restrict__ dst, const int blocks_per_row) {
    const int tid = 64*threadIdx.y + threadIdx.x;
    const int kqs = 2*(tid & 7);
    const int kbx0 = tid >> 3;
    const int nslot = 8*NWARPS;
    float ta = 0.0f, tb = 0.0f;
    int kbx = kbx0;
    for (; kbx + nslot < blocks_per_row; kbx += 2*nslot) {
        ta += vd_iq3s_const(vx, vy, kbx, kqs);
        tb += vd_iq3s_const(vx, vy, kbx + nslot, kqs);
    }
    if (kbx < blocks_per_row) ta += vd_iq3s_const(vx, vy, kbx, kqs);
    float tmp = ta + tb;
    __shared__ float red[NWARPS > 1 ? NWARPS-1 : 1][64];
    if (threadIdx.y > 0) red[threadIdx.y-1][threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int l = 0; l < NWARPS-1; ++l) tmp += red[l][threadIdx.x];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) tmp += __shfl_down(tmp, off, 64);
    if (threadIdx.x == 0) dst[blockIdx.x] = tmp;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <gguf> [niter]\n", argv[0]); return 1; }
    const int niter = argc > 2 ? atoi(argv[2]) : 100;

    // blk.21.ffn_up.weight [5120,17408] iq3_s, offset 4301102016, 28200960 B
    const long off = 4301102016L, len = 38297600L;
    const int K = 5120, N = 17408;
    const int blocks_per_row = K / 256;     // 20
    const int nq8 = K / 32;                 // 160

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { printf("open gguf failed\n"); return 1; }
    void * mm = mmap(nullptr, len + off, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm == MAP_FAILED) { printf("mmap failed\n"); return 1; }
    const char * wsrc = (const char *) mm + off;

    // device buffers
    block_iq3_s * dw; block_q8_1 * dy; float * dd;
    CHECK(hipMalloc(&dw, len));
    CHECK(hipMalloc(&dy, nq8 * sizeof(block_q8_1)));
    CHECK(hipMalloc(&dd, N * sizeof(float)));
    CHECK(hipMemcpy(dw, wsrc, len, hipMemcpyHostToDevice));
    std::vector<block_q8_1> hq(nq8);
    for (int i = 0; i < nq8; ++i) {
        for (int j = 0; j < 32; ++j) hq[i].qs[j] = (int8_t)((i*37 + j*11) % 255 - 127);
        hq[i].ds.x = ggml_fp32_to_fp16(1.0f);
        hq[i].ds.y = ggml_fp32_to_fp16(64.0f);
    }
    CHECK(hipMemcpy(dy, hq.data(), nq8 * sizeof(block_q8_1), hipMemcpyHostToDevice));

    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));

    // warm the weight bytes into HBM clocks state with one launch per variant
    float ms = 0.0f;
    const double bytes = (double)len + (double)nq8 * sizeof(block_q8_1) + (double)N * 4;
    const double floor_us = bytes / 183.8e9 * 1e6;

    printf("shape: N=%d K=%d iq3_s, %.1f MB weights | byte floor %.1f us at 183.8 GB/s\n",
           N, K, len/1e6, floor_us);
    printf("%-28s %10s %10s %10s\n", "variant", "us/call", "GB/s", "x floor");

    struct V { const char * name; int nw; bool lds; bool bal; int rows; };
    V vars[] = {{"NW2 const (shipped)", 2, false, false, 1}, {"NW2 r2", 2, false, false, 2},
                {"NW2 r4", 2, false, false, 4}, {"NW2 r8", 2, false, false, 8},
                {"NW1 r4", 1, false, false, 4}, {"NW2 ilp2", 2, false, false, -2}, {"NW4 ilp2", 4, false, false, -4}};

    for (const V & v : vars) {
        dim3 grid((N + (v.rows > 0 ? v.rows : 1) - 1) / (v.rows > 0 ? v.rows : 1)), block(64, v.nw);
        auto launch = [&]() {
            switch (v.nw * 100 + v.rows) {
                case 201: gemv_iq3s<2, false, false, false><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
                case 202: gemv_iq3s_rows<2, 2><<<grid, block>>>(dw, dy, dd, blocks_per_row, N); break;
                case 204: gemv_iq3s_rows<2, 4><<<grid, block>>>(dw, dy, dd, blocks_per_row, N); break;
                case 208: gemv_iq3s_rows<2, 8><<<grid, block>>>(dw, dy, dd, blocks_per_row, N); break;
                case 104: gemv_iq3s_rows<1, 4><<<grid, block>>>(dw, dy, dd, blocks_per_row, N); break;
                case 108: gemv_iq3s_rows<1, 8><<<grid, block>>>(dw, dy, dd, blocks_per_row, N); break;
                case 198: gemv_iq3s_ilp2<2><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
                case 396: gemv_iq3s_ilp2<4><<<grid, block>>>(dw, dy, dd, blocks_per_row); break;
            }
        };
        launch(); CHECK(hipDeviceSynchronize());
        // sanity: dst not all-zero
        float probe = 0.0f;
        CHECK(hipMemcpy(&probe, dd + N/2, 4, hipMemcpyDeviceToHost));
        if (probe == 0.0f) { printf("%s: BLANK OUTPUT - cell refused\n", v.name); continue; }

        CHECK(hipEventRecord(e0));
        for (int i = 0; i < niter; ++i) launch();
        CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        const double us = 1000.0 * ms / niter;
        printf("%-28s %10.1f %10.1f %10.2f\n", v.name, us, bytes / (us * 1e-6) / 1e9, us / floor_us);
    }
    (void)host_row_sum;
    return 0;
}
