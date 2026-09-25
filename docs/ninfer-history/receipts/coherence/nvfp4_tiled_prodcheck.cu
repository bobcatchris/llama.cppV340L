// nvfp4_tiled_prodcheck.cu — TILEDV-zeros hunt, production-flag harness (2026-09-17).
// Reproduces the serving [TILEDV] cell OUTSIDE serving: GdnInputW4 (4096x5120) at M=128,
// codec-layout scales, random codes/x — tiled vs legacy small_t chunks on the SAME buffers.
// Compiled with the EXACT HipSources production flags (see companion row).
#include "ops/linear/nvfp4/nvfp4_small_t_hip.cu"
#include "ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>

using namespace ninfer::ops;
using ninfer::Weight;
using ninfer::Tensor;
using ninfer::DType;

namespace {
#include <cstdlib>
static int N = 4096;              // geometry from argv (default GdnInputW4)
static int K = 5120;
static int M = 128;
static int STILES = K / 64;

std::uint32_t lcg(std::uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(s >> 33);
}
} // namespace

int main(int argc, char** argv) {
    if (argc >= 3) { N = atoi(argv[1]); K = atoi(argv[2]); }
    if (argc >= 4) { M = atoi(argv[3]); }
    STILES = K / 64;
    if (cudaFree(nullptr) != cudaSuccess) { std::printf("SKIP no device\n"); return 77; }

    // host setup, production layout
    const std::size_t x_elems = static_cast<std::size_t>(M) * K;
    const std::size_t out_elems = static_cast<std::size_t>(M) * N;
    auto* h_x = static_cast<std::uint16_t*>(malloc(x_elems * 2));
    auto* h_codes = static_cast<std::uint8_t*>(malloc(static_cast<std::size_t>(N) * (K / 2)));
    auto* h_scales = static_cast<std::uint8_t*>(malloc(static_cast<std::size_t>(N) * STILES * 4));
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
    const bool benchdata = (argc >= 5 && argv[4][0] == 'b');
    if (benchdata) { std::memset(h_x, 0x3C, x_elems * 2); }
    for (std::size_t i = 0; i < x_elems; ++i) {
        if (benchdata) { break; }
        const float v = 1.7f * std::cos(static_cast<float>(lcg(seed) % 4096) * 0.00153f);
        const std::uint32_t bits = *reinterpret_cast<const std::uint32_t*>(&v);
        h_x[i] = static_cast<std::uint16_t>(bits >> 16);
    }
    if (benchdata) { std::memset(h_codes, 0x11, static_cast<std::size_t>(N) * (K / 2)); }
    for (std::size_t i = 0; i < static_cast<std::size_t>(N) * (K / 2); ++i) {
        if (benchdata) { break; }
        h_codes[i] = static_cast<std::uint8_t>(lcg(seed) & 0xFF);
    }
    // scales: the codec plane IS the single home — fill per (row, group) at the codec offset
    for (int row = 0; row < N; ++row) {
        for (int g = 0; g < K / 16; ++g) {
            // nvfp4_scale_offset<STILES>(row, g) inlined (STILES is runtime here)
            const int m_tile = row / 128, ri = row - m_tile * 128;
            const std::int64_t off = (static_cast<std::int64_t>(m_tile) * STILES + g / 4) * 512 +
                                     (ri & 31) * 16 + ((ri >> 5) & 3) * 4 + (g & 3);
            h_scales[off] = benchdata ? 0x38 : static_cast<std::uint8_t>(0x30 + (lcg(seed) & 0x0F));
        }
    }

    std::uint16_t *d_x, *d_out1, *d_out2;
    std::uint8_t *d_codes, *d_scales;
    cudaMalloc(&d_x, x_elems * 2);
    cudaMalloc(&d_out1, out_elems * 2);
    cudaMalloc(&d_out2, out_elems * 2);
    cudaMalloc(&d_codes, static_cast<std::size_t>(N) * (K / 2));
    cudaMalloc(&d_scales, static_cast<std::size_t>(N) * STILES * 4);
    cudaMemset(d_out1, 0xEE, out_elems * 2); // sentinel: proves the kernel WROTE
    cudaMemset(d_out2, 0xDD, out_elems * 2);
    cudaMemcpy(d_x, h_x, x_elems * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_codes, h_codes, static_cast<std::size_t>(N) * (K / 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales, static_cast<std::size_t>(N) * STILES * 4, cudaMemcpyHostToDevice);

    Weight w{};
    w.n = N; w.k = K;
    w.qdata = d_codes;
    w.scales = d_scales;
    w.weight_scale_divisor = benchdata ? 1.0f : 116.0f;

    Tensor xin; xin.data = d_x; xin.ne[0] = K; xin.ne[1] = M; xin.ne[2] = 1; xin.ne[3] = 1;
    xin.dtype = DType::BF16;
    Tensor o1; o1.data = d_out1; o1.ne[0] = N; o1.ne[1] = M; o1.ne[2] = 1; o1.ne[3] = 1;
    o1.dtype = DType::BF16;
    Tensor o2; o2.data = d_out2; o2.ne[0] = N; o2.ne[1] = M; o2.ne[2] = 1; o2.ne[3] = 1;
    o2.dtype = DType::BF16;

    // warm the mclk (5s-class hammer, short: this cell is correctness, not perf)
    { cudaEvent_t w0, w1; cudaEventCreate(&w0); cudaEventCreate(&w1); cudaEventRecord(w0);
      std::uint32_t* sink; cudaMalloc(&sink, 4); float secs = 0.f;
      do { for (int i = 0; i < 40; ++i) { cudaMemcpyAsync(d_out2, d_x, x_elems * 2, cudaMemcpyDeviceToDevice); }
           cudaEventRecord(w1); cudaEventSynchronize(w1); cudaEventElapsedTime(&secs, w0, w1); } while (secs < 3000.f);
      cudaEventDestroy(w0); cudaEventDestroy(w1); }

    { // tiny throwaway kernel FIRST (first-launch hypothesis): a memcpyAsync kernel
      cudaStreamSynchronize(0);
      cudaMemcpyAsync(d_out2, d_x, 16, cudaMemcpyDeviceToDevice); // not a kernel; use a real one:
    }
    // run small_t FIRST so tiled is not the process's first kernel launch
    for (int tb = 0; tb < M; tb += 32) {
        const int act = (M - tb) < 32 ? (M - tb) : 32;
        Tensor xc(reinterpret_cast<std::uint8_t*>(d_x) + static_cast<std::int64_t>(tb) * K * 2,
                  DType::BF16, {K, act});
        Tensor oc(reinterpret_cast<std::uint8_t*>(d_out2) + static_cast<std::int64_t>(tb) * N * 2,
                  DType::BF16, {N, act});
        ninfer::ops::detail::launch_nvfp4_small_t(xc, w, oc, 0);
    }
    cudaDeviceSynchronize();

    ninfer::ops::detail::launch_nvfp4_tiled_gemm(xin, w, o1, 0);
    const auto e1 = cudaGetLastError();
    std::printf("tiled launch err=%s\n", cudaGetErrorString(e1));

    for (int tb = 0; tb < M; tb += 32) {
        const int act = (M - tb) < 32 ? (M - tb) : 32;
        Tensor xc(reinterpret_cast<std::uint8_t*>(d_x) + static_cast<std::int64_t>(tb) * K * 2,
                  DType::BF16, {K, act});
        Tensor oc(reinterpret_cast<std::uint8_t*>(d_out2) + static_cast<std::int64_t>(tb) * N * 2,
                  DType::BF16, {N, act});
        ninfer::ops::detail::launch_nvfp4_small_t(xc, w, oc, 0);
    }
    const auto e2 = cudaGetLastError();
    std::printf("smallt launch err=%s\n", cudaGetErrorString(e2));
    cudaDeviceSynchronize();

    auto* h_o1 = static_cast<std::uint16_t*>(malloc(out_elems * 2));
    auto* h_o2 = static_cast<std::uint16_t*>(malloc(out_elems * 2));
    cudaMemcpy(h_o1, d_out1, out_elems * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_o2, d_out2, out_elems * 2, cudaMemcpyDeviceToHost);

    std::size_t zeros1 = 0, mism = 0, first = out_elems;
    double rel2 = 0.0, ref2 = 0.0;
    for (std::size_t i = 0; i < out_elems; ++i) {
        if (h_o1[i] == 0x0000) { ++zeros1; }
        if (h_o1[i] != h_o2[i]) { ++mism; if (i < first) { first = i; } }
        const float a = *reinterpret_cast<const __nv_bfloat16*>(&h_o1[i]);
        const float b = *reinterpret_cast<const __nv_bfloat16*>(&h_o2[i]);
        rel2 += (a - b) * (a - b);
        ref2 += b * b;
    }
    std::printf("RAW16:");
    for (int i = 0; i < 16; ++i) { std::printf(" %04x", h_o1[i]); }
    std::printf("\n");
    std::printf("HOST x0=%04x codes01=%02x%02x scales512=%02x%02x scales01=%02x%02x\n",
                h_x[0], h_codes[1], h_codes[0], h_scales[513], h_scales[512], h_scales[1], h_scales[0]);
    cudaMemcpy(h_x, d_x, x_elems * 2, cudaMemcpyDeviceToHost);
    std::printf("LDS t38=%04x%04x p11x=%04x%04x p11y=%04x%04x prx=%04x%04x\n",
                h_x[1025], h_x[1024], h_x[1027], h_x[1026], h_x[1029], h_x[1028], h_x[1031], h_x[1030]);
    std::printf("ACC a00=%04x%04x a01=%04x%04x a10=%04x%04x mk=%04x%04x\n",
                h_x[1025], h_x[1024], h_x[1027], h_x[1026], h_x[1029], h_x[1028], h_x[1031], h_x[1030]);
    { std::size_t sent = 0; for (std::size_t i = 0; i < out_elems; ++i) { if (h_o1[i] == 0xEEEE) { ++sent; } }
      std::printf("sentinel_EEEE=%zu/%zu first8:", sent, out_elems);
      for (int i = 0; i < 8; ++i) { std::printf(" %04x", h_o1[i]); }
      std::printf("\n"); }
    std::printf("zeros_tiled=%zu/%zu mism=%zu/%zu relL2=%.3e first=%zu tiled=%04x smallt=%04x\n",
                zeros1, out_elems, mism, out_elems, rel2 / (ref2 + 1e-30), first,
                first < out_elems ? h_o1[first] : 0, first < out_elems ? h_o2[first] : 0);
    // structure: zeros per 64-row block (first 8) and per token (first 8)
    std::printf("zeros_by_rowblk[0..7]:");
    for (int rb = 0; rb < 8; ++rb) {
        std::size_t z = 0;
        for (int r = 0; r < 64; ++r) { for (int t = 0; t < M; ++t) { if (h_o1[static_cast<std::size_t>(t) * N + rb * 64 + r] == 0) { ++z; } } }
        std::printf(" %zu/%d", z, 64 * M);
    }
    std::printf("\nzeros_by_tok[0..7]:");
    for (int t = 0; t < 8; ++t) {
        std::size_t z = 0;
        for (int r = 0; r < N; ++r) { if (h_o1[static_cast<std::size_t>(t) * N + r] == 0) { ++z; } }
        std::printf(" %zu/%d", z, N);
    }
    std::printf("\n");
    return 0;
}
