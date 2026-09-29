// nvfp4_smallt_roofline_bench.cu — SMALLT-TUNE companion of nvfp4_roofline_bench.cu.
//
// Drives the PRODUCTION small_t arm launch_nvfp4_small_t (the exact arm serve uses at verify
// widths T=2..4 and prefill chunks, proven by SMALLT_row.txt's [SMALLT] serving trace) at the
// five TP4 W4 geometries, T=3 primary (+ T=2/4 context), same discipline as the GEMV bench:
//   - production -O3 compile line (EXTRACTING_NUMBERS Part A trap #1, the -O3 LAW)
//   - 5 s continuous mclk hammer BEFORE timing (Part A trap #2)
//   - hipEvent timing on the legacy stream (cross-checked vs wall in the GEMV triage)
// Bytes convention (same as the GEMV bench): useful bytes per call =
//   rows*(K/2 codes + K/16 scales) + K*T*2 x + rows*T*2 out — graded vs the pure-read ceiling.
// The BASE kernel re-reads weights per token internally (DRAM demand up to T x useful weight
// bytes); the reported number is the USEFUL stream rate — the same convention the GEMV row used.
//
// Compile EXACTLY like the GEMV bench (the -O3 is LAW):
//   clang++ -O3 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 -D__HIP_PLATFORM_AMD__=1
//     -D__HIP_ROCclr__=1 -I$W/src/common/hip_shim -I$W/include -I$W/src -I$W/third_party
//     -DNDEBUG -std=gnu++20 --offload-arch=gfx900 -x hip <this file> $W/src/core/device.cu
//     -o nvfp4_smallt_roofline_bench
//
// SMALLT-PK section (HFMA2 work order, 2026-09-17): at T=3, drives the PRODUCTION S5 arm
// (launch_nvfp4_small_t, env unset) against the PACKED-FP16 arm (launch_nvfp4_small_t_pk,
// nvfp4_small_t_hip_kernel_pk) on the same buffers — prints GB/s for both plus the
// A16-criterion-class tolerance gate rel-L2(S5, PK) < 1e-2. The PK numerics are
// tolerance-gated, NOT bit-equal (packed-fp16 accumulation windows — see HFMA2_notes.md), so
// the check is statistical: fixed-seed splitmix64 data, scales NaN-masked (fin-style), x
// bounded to the production pre-norm envelope |x| in [2^-9, ~32] (exp field 118..131) since
// the fp16 window requires max|x| < 682. Exit rc 1 if ANY tolerance check fails.

#include "ops/linear/nvfp4/nvfp4_small_t_hip.cu"

#include <cstdio>
#include <cmath>
#include <functional>
#include <vector>

namespace {

// Fixed-seed generator (bitcheck pattern) for the PK tolerance arm.
std::uint64_t splitmix64(std::uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

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
    if (acc == 0xDEADBEEFu) { sink[0] = acc; }
}

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

    const std::size_t bytes = static_cast<std::size_t>(256) * 1024 * 1024;
    const std::size_t n4 = bytes / 4;
    std::uint32_t *src, *dst, *sink;
    cudaMalloc(&src, bytes);
    cudaMalloc(&dst, bytes);
    cudaMalloc(&sink, 4);
    cudaMemset(src, 0x5A, bytes);
    const int grid = prop.multiProcessorCount * 8;
    {   // CLOCK WARM-UP: ~5 s continuous hammer (Part A trap #2)
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
    const double copy_gbs = 2.0 * bytes / 1e6 / copy_ms;
    const double read_gbs = bytes / 1e6 / read_ms;
    std::printf("CEILING copy  %8.2f ms  %8.1f GB/s (2x conv)\n", copy_ms, copy_gbs);
    std::printf("CEILING read  %8.2f ms  %8.1f GB/s (pure read)\n", read_ms, read_gbs);

    struct Prob { const char* name; int rows, k; };
    const Prob probs[] = {
        {"AttnInputW4", 3584, 5120},
        {"GdnInputW4", 4096, 5120},
        {"MlpGateUpW4", 8704, 5120},
        {"Resid6144W4", 5120, 1536},
        {"Resid17408W4", 5120, 4352},
    };
    const int ts[] = {3, 2, 4};  // T=3 primary (verify width at k=2), then context
    for (int ti = 0; ti < 3; ++ti) {
        const int T = ts[ti];
        std::printf("== T=%d ==\n", T);
        std::printf("%-14s %10s %9s %9s %7s %7s\n", "problem", "ms/call", "GB/s", "%read", "rows", "K");
        for (const Prob& p : probs) {
            const std::size_t code_bytes = static_cast<std::size_t>(p.rows) * (p.k / 2);
            const std::size_t scale_bytes = static_cast<std::size_t>(p.rows) * (p.k / 16);
            const std::size_t x_bytes = static_cast<std::size_t>(p.k) * 2 * T;
            const std::size_t out_bytes = static_cast<std::size_t>(p.rows) * 2 * T;
            const std::size_t total = code_bytes + scale_bytes + x_bytes + out_bytes;

            void *x, *codes, *scales, *out;
            cudaMalloc(&x, x_bytes);
            cudaMalloc(&codes, code_bytes);
            cudaMalloc(&scales, scale_bytes);
            cudaMalloc(&out, out_bytes);
            cudaMemset(codes, 0x11, code_bytes);
            cudaMemset(scales, 0x38, scale_bytes);
            cudaMemset(x, 0, x_bytes);

            ninfer::Tensor tx;
            tx.data = x;
            tx.ne[0] = p.k;
            tx.ne[1] = T;
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
            tout.ne[1] = T;
            tout.ne[2] = 1;
            tout.ne[3] = 1;
            tout.dtype = ninfer::DType::BF16;

            const double ms = time_ms(
                [&] { ninfer::ops::detail::launch_nvfp4_small_t(tx, tw, tout, 0); }, 5, 100);
            const double gbs = total / 1e6 / ms;
            std::printf("%-14s %10.4f %9.1f %6.1f%% %7d %7d\n", p.name, ms, gbs,
                        100.0 * gbs / read_gbs, p.rows, p.k);
            cudaFree(x);
            cudaFree(codes);
            cudaFree(scales);
            cudaFree(out);
        }
    }

    // ---- SMALLT-PK A/B (T=3 verify width): production S5 arm vs packed-fp16 PK arm on the
    // SAME data; prints both GB/s and the rel-L2 tolerance gate. rc 1 on any gate failure.
    constexpr int kPkT = 3;
    const double kRelL2Bar = 1e-2;  // A16-criterion class
    std::printf("== SMALLT-PK A/B, T=%d (S5 = launch_nvfp4_small_t default, PK = "
                "launch_nvfp4_small_t_pk), rel-L2 bar %.0e ==\n",
                kPkT, kRelL2Bar);
    std::printf("%-14s %9s %9s %7s %10s %6s\n", "problem", "S5 GB/s", "PK GB/s", "ratio",
                "rel-L2", "check");
    int pk_bad = 0;
    for (const Prob& p : probs) {
        const std::size_t code_bytes  = static_cast<std::size_t>(p.rows) * (p.k / 2);
        const std::size_t scale_bytes = static_cast<std::size_t>(p.rows) * (p.k / 16);
        const std::size_t x_bytes     = static_cast<std::size_t>(p.k) * 2 * kPkT;
        const std::size_t out_bytes   = static_cast<std::size_t>(p.rows) * 2 * kPkT;
        const std::size_t total       = code_bytes + scale_bytes + x_bytes + out_bytes;

        std::vector<unsigned char> h_codes(code_bytes), h_scales(scale_bytes);
        std::vector<std::uint16_t> h_x(static_cast<std::size_t>(p.k) * kPkT);
        std::uint64_t seed = 0x5EED'0000'0000'0001ull ^
                             (static_cast<std::uint64_t>(p.rows) << 24) ^
                             static_cast<std::uint64_t>(p.k);
        for (std::size_t i = 0; i < code_bytes; ++i) {
            h_codes[i] = static_cast<unsigned char>(splitmix64(seed) >> 33);
        }
        for (std::size_t i = 0; i < scale_bytes; ++i) {
            unsigned char b = static_cast<unsigned char>(splitmix64(seed) >> 33);
            if ((b & 0x7Fu) == 0x7Fu) { b &= 0x7Eu; }  // fin-style: no NaN scales
            h_scales[i] = b;
        }
        for (std::size_t i = 0; i < h_x.size(); ++i) {
            // Production pre-norm envelope: exp field 118..131 => |x| in [2^-9, ~32]; the
            // fp16 window overflows only above max|x| = 682, so this data has >=20x margin.
            const std::uint16_t r = static_cast<std::uint16_t>(splitmix64(seed) >> 48);
            h_x[i] = static_cast<std::uint16_t>((r & 0x807Fu) |
                                                (static_cast<std::uint16_t>(118 + (r % 14)) << 7));
        }

        void *x, *codes, *scales, *out5, *outpk;
        cudaMalloc(&x, x_bytes);
        cudaMalloc(&codes, code_bytes);
        cudaMalloc(&scales, scale_bytes);
        cudaMalloc(&out5, out_bytes);
        cudaMalloc(&outpk, out_bytes);
        cudaMemcpy(x, h_x.data(), x_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(codes, h_codes.data(), code_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(scales, h_scales.data(), scale_bytes, cudaMemcpyHostToDevice);

        ninfer::Tensor tx;
        tx.data = x; tx.ne[0] = p.k; tx.ne[1] = kPkT; tx.ne[2] = 1; tx.ne[3] = 1;
        tx.dtype = ninfer::DType::BF16;
        ninfer::Weight tw;
        tw.qdata = codes; tw.scales = scales;
        tw.n = p.rows; tw.k = p.k;
        tw.weight_scale_divisor = 1.177f;  // dirty divisor: exercises the fp32 coeff fold
        ninfer::Tensor t5, tpk;
        t5.data = out5;  t5.ne[0] = p.rows;  t5.ne[1] = kPkT;  t5.ne[2] = 1; t5.ne[3] = 1;
        t5.dtype = ninfer::DType::BF16;
        tpk.data = outpk; tpk.ne[0] = p.rows; tpk.ne[1] = kPkT; tpk.ne[2] = 1; tpk.ne[3] = 1;
        tpk.dtype = ninfer::DType::BF16;

        const double ms5 = time_ms(
            [&] { ninfer::ops::detail::launch_nvfp4_small_t(tx, tw, t5, 0); }, 5, 100);
        const double mspk = time_ms(
            [&] { ninfer::ops::detail::launch_nvfp4_small_t_pk(tx, tw, tpk, 0); }, 5, 100);
        cudaDeviceSynchronize();

        std::vector<std::uint16_t> h5(p.rows * kPkT), hpk(p.rows * kPkT);
        cudaMemcpy(h5.data(), out5, out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hpk.data(), outpk, out_bytes, cudaMemcpyDeviceToHost);
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < h5.size(); ++i) {
            // bf16 bits -> f32 via the same <<16 embed the kernels decode with (host-side
            // union bit-cast, the codec's own type-punning pattern; no device-only helper).
            union { std::uint32_t u; float f; } a5, apk;
            a5.u  = static_cast<unsigned>(h5[i]) << 16;
            apk.u = static_cast<unsigned>(hpk[i]) << 16;
            const double a = static_cast<double>(a5.f);
            const double b = static_cast<double>(apk.f);
            num += (a - b) * (a - b);
            den += a * a;
        }
        const double rel_l2 = (den > 0.0) ? std::sqrt(num / den) : 0.0;
        const bool ok = rel_l2 < kRelL2Bar;
        if (!ok) { ++pk_bad; }
        std::printf("%-14s %9.1f %9.1f %6.2fx %10.2e %6s\n", p.name, total / 1e6 / ms5,
                    total / 1e6 / mspk, mspk / ms5, rel_l2, ok ? "PASS" : "FAIL");
        cudaFree(x);
        cudaFree(codes);
        cudaFree(scales);
        cudaFree(out5);
        cudaFree(outpk);
    }
    if (pk_bad != 0) {
        std::printf("SMALLT-PK: %d/%zu tolerance gate(s) FAILED (bar %.0e) — PK NOT clear to "
                    "integrate\n",
                    pk_bad, sizeof(probs) / sizeof(probs[0]), kRelL2Bar);
        return 1;
    }
    std::printf("SMALLT-PK: all tolerance gates PASS (bar %.0e)\n", kRelL2Bar);
    return 0;
}
