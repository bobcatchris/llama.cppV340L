// W2 pure-consume ablation: names the access-pattern ceiling for MMVQ-style
// scattered 1,416-B block reads on gfx900 (die 3 dev cell).
//   consume_seq     : coalesced uint4 grid-stride read of the whole buffer
//                     (the sequential-copy-class ceiling)
//   consume_pattern : EXACT same load pattern as gemv_iq3s (block 64x2, 16
//                     kbx-slots, same qs/qh/signs/scales/x offsets) minus the
//                     LUT lookups and dp4a - pure memory pattern, trivial ALU
// Usage: HIP_VISIBLE_DEVICES=3 ./bench_consume <gguf>
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <vector>
#include <hip/hip_runtime.h>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

__global__ __launch_bounds__(256, 1) static void consume_seq(
        const uint4 * __restrict__ src, uint4 * __restrict__ sink, const long n_vec) {
    const long stride = (long)gridDim.x * blockDim.x;
    uint4 acc = {0, 0, 0, 0};
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
        const uint4 v = src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xdeadbeefu && acc.y == 1u) sink[threadIdx.x] = acc;  // never true; defeats DCE
}

// pattern clone: same shape/geometry as gemv_iq3s, loads only, no LUT, no dp4a
__global__ __launch_bounds__(128, 1) static void consume_pattern(
        const block_iq3_s * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        float * __restrict__ dst, const int blocks_per_row) {
    const int tid = 64*threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;
    const int kqs = 2*(tid & 7);
    const int kbx0 = tid >> 3;
    const int kper_iter = 8*2;
    int acc = 0;
    const block_iq3_s * h = vx + (size_t)row * blocks_per_row;
    const block_q8_1 * y0 = vy;
    for (int kbx = kbx0; kbx < blocks_per_row; kbx += kper_iter) {
        const block_iq3_s * b = h + kbx;
        acc ^= get_int_b2(b->qs, kqs + 0);
        acc ^= get_int_b2(b->qs, kqs + 1);
        acc ^= b->qh[kqs/2];
        acc ^= get_int_b2(b->signs, kqs/2);
        acc ^= b->scales[kqs/4];
        const block_q8_1 * y = y0 + kbx*8 + kqs/2;
        acc ^= get_int_b4(y->qs, 0);
        acc ^= get_int_b4(y->qs, 4);
    }
    __shared__ float red[1][64];
    if (threadIdx.y > 0) red[0][threadIdx.x] = (float)acc;
    __syncthreads();
    if (threadIdx.y > 0) return;
    float v = (float)acc + red[0][threadIdx.x];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off, 64);
    if (threadIdx.x == 0) dst[row] = v;
}

// pattern + LUT variant: adds ONLY the 8 constant-memory iq3s_grid lookups
// (dependent on qs/qh loads) on top of consume_pattern - no decode ALU
__global__ __launch_bounds__(128, 1) static void consume_pattern_lut(
        const block_iq3_s * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        float * __restrict__ dst, const int blocks_per_row) {
    const int tid = 64*threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;
    const int kqs = 2*(tid & 7);
    const int kbx0 = tid >> 3;
    int acc = 0;
    const block_iq3_s * h = vx + (size_t)row * blocks_per_row;
    const block_q8_1 * y0 = vy;
    for (int kbx = kbx0; kbx < blocks_per_row; kbx += 16) {
        const block_iq3_s * b = h + kbx;
        const int2 qs_packed = make_int2(get_int_b2(b->qs, kqs + 0), get_int_b2(b->qs, kqs + 1));
        const uint8_t * qs = (const uint8_t *) &qs_packed;
        const int qh = b->qh[kqs/2];
        int lut = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            lut ^= iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
            lut ^= iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        }
        acc ^= lut ^ qs[0] ^ qh;
    }
    __shared__ float red[1][64];
    if (threadIdx.y > 0) red[0][threadIdx.x] = (float)acc;
    __syncthreads();
    if (threadIdx.y > 0) return;
    float v = (float)acc + red[0][threadIdx.x];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off, 64);
    if (threadIdx.x == 0) dst[row] = v;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <gguf> [niter]\n", argv[0]); return 1; }
    const int niter = argc > 2 ? atoi(argv[2]) : 200;
    const long off = 4301102016L, len = 38297600L;
    const int K = 5120, N = 17408, blocks_per_row = K / 256, nq8 = K / 32;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { printf("open gguf failed\n"); return 1; }
    void * mm = mmap(nullptr, len + off, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm == MAP_FAILED) { printf("mmap failed\n"); return 1; }

    char * dw; block_q8_1 * dy; float * dd; uint4 * sink;
    CHECK(hipMalloc((void**)&dw, len));
    CHECK(hipMalloc(&dy, nq8 * sizeof(block_q8_1)));
    CHECK(hipMalloc(&dd, N * sizeof(float)));
    CHECK(hipMalloc(&sink, 256 * sizeof(uint4)));
    CHECK(hipMemcpy(dw, (const char *) mm + off, len, hipMemcpyHostToDevice));
    std::vector<block_q8_1> hq(nq8);
    for (int i = 0; i < nq8; ++i) {
        for (int j = 0; j < 32; ++j) hq[i].qs[j] = (int8_t)((i*37 + j*11) % 255 - 127);
        hq[i].ds.x = ggml_fp32_to_fp16(1.0f);
        hq[i].ds.y = ggml_fp32_to_fp16(64.0f);
    }
    CHECK(hipMemcpy(dy, hq.data(), nq8 * sizeof(block_q8_1), hipMemcpyHostToDevice));

    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    float ms = 0.0f;
    const double bytes = (double)len + (double)nq8 * sizeof(block_q8_1) + (double)N * 4;

    printf("pure-consume ablation on %.1f MB iq3_s tensor | seq-copy floor 208.8 us (183.8 GB/s)\n", len/1e6);
    printf("%-24s %10s %10s\n", "kernel", "us/call", "GB/s");

    // sequential coalesced ceiling
    {
        const long n_vec = len / 16;
        dim3 grid(1024), block(256);
        consume_seq<<<grid, block>>>((const uint4 *) (void*) dw, sink, n_vec);
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(e0));
        for (int i = 0; i < niter; ++i) consume_seq<<<grid, block>>>((const uint4 *) (void*) dw, sink, n_vec);
        CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        printf("%-24s %10.1f %10.1f\n", "consume_seq", 1000.0*ms/niter, len/(1000.0*ms/niter*1e-6)/1e9);
    }

    // MMVQ-shaped pattern, loads only
    {
        dim3 grid(N), block(64, 2);
        consume_pattern<<<grid, block>>>((const block_iq3_s *) dw, dy, dd, blocks_per_row);
        CHECK(hipDeviceSynchronize());
        float probe = 0; CHECK(hipMemcpy(&probe, dd + N/2, 4, hipMemcpyDeviceToHost));
        if (probe == 0.0f && dd[0] == 0.0f) printf("WARNING: blank output\n");
        CHECK(hipEventRecord(e0));
        for (int i = 0; i < niter; ++i) consume_pattern<<<grid, block>>>((const block_iq3_s *) dw, dy, dd, blocks_per_row);
        CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        printf("%-24s %10.1f %10.1f\n", "consume_pattern", 1000.0*ms/niter, bytes/(1000.0*ms/niter*1e-6)/1e9);
    }

    // pattern + constant-LUT lookups
    {
        dim3 grid(N), block(64, 2);
        consume_pattern_lut<<<grid, block>>>((const block_iq3_s *) dw, dy, dd, blocks_per_row);
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(e0));
        for (int i = 0; i < niter; ++i) consume_pattern_lut<<<grid, block>>>((const block_iq3_s *) dw, dy, dd, blocks_per_row);
        CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        printf("%-24s %10.1f %10.1f\n", "consume_pattern_lut", 1000.0*ms/niter, bytes/(1000.0*ms/niter*1e-6)/1e9);
    }
    printf("(gemv shipped reference: 364.7 us, 105.2 GB/s)\n");
    return 0;
}
