// NVFP4 GEMV ROOFLINE BENCH (doc 01 §7 rule 1: roofline-first; ROADMAP_STATUS next-build #1).
//
// Measures, on ONE gfx900 die:
//   1. D2D copy ceiling  (GB/s, 2x convention)  — the "memcpy ceiling" analogue.
//   2. Read ceiling      (GB/s)                 — the honest ceiling for a read-dominated GEMV.
//   3. The PRODUCTION nvfp4_gemv_hip decode kernel (launched via launch_nvfp4_decode, the
//      exact arm serve uses) on the big registered geometries: weight bytes streamed / second
//      and % of both ceilings.
//
// Weight bytes per call = rows*(K/2 codes + K/16 e4m3 scales) + K*2 x-vector + rows*2 out.
// v100-skinny reference points: kernel 648 GB/s = 78.5% of their 825 GB/s memcpy ceiling;
// in-server GEMM 596 GB/s (72%). Correctness is NOT re-proven here (duel-validated already);
// this bench prices the tuning gap named in nvfp4_gemv_hip.cu's header.

#include "ops/linear/nvfp4/nvfp4_gemv_hip.cu"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

namespace {

double time_ms(const std::function<void()>& fn, int warmup, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < warmup; ++i) { fn(); }
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) { fn(); }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return static_cast<double>(ms) / iters;
}

__global__ void copy_kernel(const std::uint32_t* __restrict__ src, std::uint32_t* __restrict__ dst,
                            std::size_t n4) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t k = i; k < n4; k += stride) { dst[k] = src[k]; }
}

__global__ void read_kernel(const std::uint32_t* __restrict__ src, std::uint32_t* __restrict__ sink,
                            std::size_t n4) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::uint32_t acc = 0;
    for (std::size_t k = i; k < n4; k += stride) { acc ^= src[k]; }
    if (acc == 0xDEADBEEFu) { sink[0] = acc; } // never true for our data; defeats DCE
}

constexpr double kMiB = 1024.0 * 1024.0;

} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("SKIP: no device\n");
        return 77;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("device %s SMs=%d\n", prop.name, prop.multiProcessorCount);

    // ---- 1. copy + read ceilings on a 256 MiB working set -------------------------------
    const std::size_t bytes = static_cast<std::size_t>(256) * 1024 * 1024;
    const std::size_t n4 = bytes / 4;
    std::uint32_t *src, *dst, *sink;
    cudaMalloc(&src, bytes);
    cudaMalloc(&dst, bytes);
    cudaMalloc(&sink, 4);
    cudaMemset(src, 0x5A, bytes);
    const int grid = prop.multiProcessorCount * 8;
    // CLOCK WARM-UP: mclk idles at level 0 (167 MHz measured via rocm-smi) and short bursts
    // never generate enough demand to ramp it — at idle mclk the ceiling is trivially met so
    // the governor never escalates (measured 2026-09-17: copy at ~28 GB/s 2x cold). Hammer
    // continuously for ~5 s wall-clock to force load state, then measure.
    {
        cudaEvent_t w0, w1;
        cudaEventCreate(&w0);
        cudaEventCreate(&w1);
        cudaEventRecord(w0);
        float secs = 0.f;
        do {
            for (int i = 0; i < 50; ++i) { copy_kernel<<<grid, 256>>>(src, dst, n4); }
            cudaEventRecord(w1);
            cudaEventSynchronize(w1);
            cudaEventElapsedTime(&secs, w0, w1);
        } while (secs < 5000.f);
        cudaEventDestroy(w0);
        cudaEventDestroy(w1);
    }
    const double copy_ms = time_ms([&] { copy_kernel<<<grid, 256>>>(src, dst, n4); }, 3, 30);
    const double read_ms = time_ms([&] { read_kernel<<<grid, 256>>>(src, sink, n4); }, 3, 30);
    const double copy_gbs = 2.0 * bytes / 1e6 / copy_ms; // 2x convention
    const double read_gbs = bytes / 1e6 / read_ms;
    std::printf("CEILING copy  %8.2f ms  %8.1f GB/s (2x conv)\n", copy_ms, copy_gbs);
    std::printf("CEILING read  %8.2f ms  %8.1f GB/s (pure read)\n", read_ms, read_gbs);

    // ---- 2. production NVFP4 GEMV on the big registered geometries ----------------------
    struct Prob { const char* name; int rows, k; };
    const Prob probs[] = {
        {"AttnInput  ", 14336, 5120},
        {"GdnInput   ", 16384, 5120},
        {"MlpGateUp  ", 34816, 5120},
        {"Residual6144", 5120, 6144},
    };
    std::printf("%-14s %10s %9s %9s %7s %7s %7s\n", "problem", "ms/call", "GB/s", "%read", "%copy", "rows", "K");
    double worst = 1e9;
    for (const Prob& p : probs) {
        const std::size_t code_bytes = static_cast<std::size_t>(p.rows) * (p.k / 2);
        const std::size_t scale_bytes = static_cast<std::size_t>(p.rows) * (p.k / 16);
        const std::size_t x_bytes = static_cast<std::size_t>(p.k) * 2;
        const std::size_t out_bytes = static_cast<std::size_t>(p.rows) * 2;
        const std::size_t total = code_bytes + scale_bytes + x_bytes + out_bytes;

        void *x, *codes, *scales, *out;
        cudaMalloc(&x, x_bytes);
        cudaMalloc(&codes, code_bytes);
        cudaMalloc(&scales, scale_bytes);
        cudaMalloc(&out, out_bytes);
        cudaMemset(codes, 0x11, code_bytes);   // valid e2m1 pairs, nonzero
        cudaMemset(scales, 0x38, scale_bytes); // ~1.0-ish e4m3
        cudaMemset(x, 0, x_bytes);

        ninfer::Tensor tx;
        tx.data = x;
        tx.ne[0] = p.k;
        tx.ne[1] = 1;
        tx.ne[2] = 1;
        tx.ne[3] = 1;
        tx.dtype = ninfer::DType::BF16;
        ninfer::Weight tw;
        tw.qdata = codes;
        tw.scales = scales;
        tw.n = p.rows;
        tw.k = p.k;
        tw.weight_scale_divisor = 1.0f;
        ninfer::Tensor tout;
        tout.data = out;
        tout.ne[0] = p.rows;
        tout.ne[1] = 1;
        tout.ne[2] = 1;
        tout.ne[3] = 1;
        tout.dtype = ninfer::DType::BF16;

        const double ms = time_ms(
            [&] { ninfer::ops::detail::launch_nvfp4_decode(tx, tw, tout, 0); }, 5, 100);
        const double gbs = total / 1e6 / ms;
        std::printf("%-14s %10.4f %9.1f %6.1f%% %6.1f%% %7d %7d\n", p.name, ms, gbs,
                    100.0 * gbs / read_gbs, 100.0 * gbs / copy_gbs, p.rows, p.k);
        worst = gbs < worst ? gbs : worst;
        cudaFree(x);
        cudaFree(codes);
        cudaFree(scales);
        cudaFree(out);
    }
    std::printf("WORST-CASE GEMV BW: %.1f GB/s (%.1f%% of read ceiling) — v100-skinny ref 648 GB/s = 78.5%%\n",
                worst, 100.0 * worst / read_gbs);
    return 0;
}
