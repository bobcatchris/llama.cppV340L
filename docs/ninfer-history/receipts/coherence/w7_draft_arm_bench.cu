// w7_draft_arm_bench.cu — F4a draft-arm retune bench (decode retune desk, lane amd/wo-w7-body).
//
// Prices the SIX W8G32 GEMVs of ONE MTP draft forward at their EXACT TP4 per-rank serving
// geometries (T=1, the decode-chain width), all of which the serving stack routes to
// launch_w8_simt_r8_c4 today (the five tp_gemv riders via tp_kernel.cu:139-145 "W8G32 t<=4 ->
// simt_r8_c4"; fc + proposal head via the select_w8_a16_launch generic tail):
//   1 fc-stem       [5120,10240]  full-weight ops::linear   (LMHEAD-gate cell, flipped 2026-09-17)
//   2 attn-qkv-pack [3584, 5120]  n-sharded packed QKV (full 14336 = 4x3584)
//   3 o-proj        [5120, 1536]  k-sharded o_proj     (full k 6144 = 4x1536)
//   4 gate_up       [8704, 5120]  n-sharded MLP gate|up (full 34816 = 4x8704)
//   5 down          [5120, 4352]  k-sharded MLP down    (full k 17408 = 4x4352)
//   6 proposal-head [10240,5120]  draft-vocab head slice ("[rank r] draft output head ready
//                                  (W8G32, 10240 rows, 53 MB)")
// plus CONTEXT rows at the UNSHARDED class geometries the W7_DECODE_DESK names
// ([14336,5120], [34816,5120]) so the "~54-56 GB/s serving class" claim is tested at both.
//
// PRE-REGISTERED ACCEPTANCE (written before any timing run, decode-desk discipline):
//   A1 reproduce-first: the bench is RUN TWICE; a shape/arm row is judged only if the two runs
//      agree within 5% (the banked W8_LMHEAD precedent used <=2%).
//   A2 slow-shape definition (retune candidate): current serving arm (simt_r8_c4) < 60% of the
//      137 GB/s proven-achievable floor of this exact format/arm family (W8_LMHEAD_row band
//      137-233 GB/s; the class ceiling reads 340-373 GB/s pure-read in-process).
//   A3 retune gate: a candidate arm/flip must show >=1.5x over simt_r8_c4 on the SAME shape in
//      the SAME process (same-band A/B) AND rel-L2 vs the simt_r8_c4 output <= 1e-2 (house
//      arm-agreement bar; serving parity stays the owner's count600 fingerprint leg).
//   A4 census: any kernel EDIT ships VGPR/scratch counts vs baseline + the wave64 checklist
//      (2-rows-per-wave mapping kept, DPP 32-lane reductions, uniform control flow across the
//      co-scheduled workers). Launch-shape-only changes are census-exempt but checklist-gated.
//   A5 honest re-pricing: if ALL serving shapes measure >= the A2 floor isolated, the master
//      plan's "~10 ms/round draft-arm" residual is NOT kernel-isolatable — the deliverable is
//      the verified per-shape roofline + the re-priced row (serving stalls/AR class), and no
//      kernel change ships.
//
// Arms are called DIRECTLY (independent of any gate env), same discipline as w8_roofline_bench.
// Held-out arms reproduce the serve build's die-loud stub discipline.
//
// Compile EXACTLY like the banked W8 bench (the -O3 is LAW), from the worktree root $W:
//   clang++ -O3 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 -D__HIP_PLATFORM_AMD__=1 \
//     -D__HIP_ROCclr__=1 -I$W/src/common/hip_shim -I$W/include -I$W/src -I$W/third_party \
//     -DNDEBUG -std=gnu++20 --offload-arch=gfx900 -x hip \
//     $W/results/amd/coherence/w7_draft_arm_bench.cu $W/src/core/device.cu \
//     $W/src/core/tensor.cpp $W/src/core/dtype.cpp -o w7_draft_arm_bench
//
// Run (standalone; claims no port; ~700 MB peak on ONE die — pin away from foreign benches):
//   HIP_VISIBLE_DEVICES=2 ./w7_draft_arm_bench

#include "ops/linear/w8/w8_rowsplit_gemm_mma.cu"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cu"
#include "ops/linear/w8/w8_small_t.cu"
#include "ops/linear/w8/w8_dispatch.cpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

// ---- link closure (same discipline as w8_roofline_bench: die loud, never silently time
// a held arm) — the mma/splitk TUs are not compiled here and no route below reaches them.
namespace ninfer::ops::detail {
[[noreturn]] static void bench_stub_die(const char* arm) {
    std::fprintf(stderr, "w7_draft_arm_bench: route reached held arm '%s' — FAIL LOUD\n", arm);
    std::abort();
}
void launch_w8_decode_r4(const Tensor&, const Weight&, Tensor&, cudaStream_t) { bench_stub_die("launch_w8_decode_r4"); }
void launch_w8_exact_t_splitk(const Tensor&, const Weight&, Tensor&, cudaStream_t) { bench_stub_die("launch_w8_exact_t_splitk"); }
void launch_w8_exact_t_composite(const Tensor&, const Weight&, Tensor&, cudaStream_t) { bench_stub_die("launch_w8_exact_t_composite"); }
void launch_w8_dflash_medium(const Tensor&, const Weight&, Tensor&, cudaStream_t) { bench_stub_die("launch_w8_dflash_medium"); }
void launch_w8_medium_splitk_c144(const Tensor&, const Weight&, Tensor&, cudaStream_t) { bench_stub_die("launch_w8_medium_splitk_c144"); }
} // namespace ninfer::ops::detail

namespace {

std::uint64_t splitmix64(std::uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double time_ms_auto(const std::function<void()>& fn, int warmup, int& iters_out) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < warmup; ++i) { fn(); }
    cudaEventRecord(a);
    for (int i = 0; i < 3; ++i) { fn(); }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float probe = 0.f;
    cudaEventElapsedTime(&probe, a, b);
    const double ms_probe = static_cast<double>(probe) / 3.0;
    int iters = static_cast<int>(1200.0 / (ms_probe > 0.05 ? ms_probe : 0.05));
    if (iters < 10) { iters = 10; }
    if (iters > 300) { iters = 300; }
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) { fn(); }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    iters_out = iters;
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

void print_mclk_sysfs() {
    for (int card = 0; card <= 7; ++card) {
        char path[160];
        std::snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/pp_dpm_mclk", card);
        std::FILE* f = std::fopen(path, "r");
        if (f == nullptr) { continue; }
        char buf[1024] = {0};
        const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        if (n > 0) {
            std::printf("SYSFS %s:\n%s", path, buf);
            return;
        }
    }
    std::printf("SYSFS pp_dpm_mclk: not found (sampler law stays with the GPU owner)\n");
}

double rel_l2_bf16(const std::uint16_t* a, const std::uint16_t* b, std::size_t count) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        union { std::uint32_t u; float f; } fa, fb;
        fa.u = static_cast<unsigned>(a[i]) << 16;
        fb.u = static_cast<unsigned>(b[i]) << 16;
        const double da = static_cast<double>(fa.f);
        const double db = static_cast<double>(fb.f);
        num += (da - db) * (da - db);
        den += da * da;
    }
    return (den > 0.0) ? std::sqrt(num / den) : 0.0;
}

struct Shape {
    const char* name;
    int n;
    int k;
    bool serving;  // true = exact per-rank serving geometry (counts toward the forward sums)
};

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
    std::printf("device %d (%s) SMs=%d — pinned by HIP_VISIBLE_DEVICES; serving die must differ "
                "from foreign benches\n",
                dev, prop.name, prop.multiProcessorCount);

    const std::size_t cbytes = static_cast<std::size_t>(256) * 1024 * 1024;
    const std::size_t n4 = cbytes / 4;
    std::uint32_t *src, *dst, *sink;
    cudaMalloc(&src, cbytes);
    cudaMalloc(&dst, cbytes);
    cudaMalloc(&sink, 4);
    cudaMemset(src, 0x5A, cbytes);
    const int grid = prop.multiProcessorCount * 8;
    {   // ~5 s continuous mclk hammer (Part A trap #2, house law)
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
    int unused_iters = 0;
    const double copy_ms = time_ms_auto([&] { copy_kernel<<<grid, 256>>>(src, dst, n4); }, 3, unused_iters);
    const double read_ms = time_ms_auto([&] { read_kernel<<<grid, 256>>>(src, sink, n4); }, 3, unused_iters);
    const double copy_gbs = 2.0 * cbytes / 1e6 / copy_ms;
    const double read_gbs = cbytes / 1e6 / read_ms;
    std::printf("CEILING copy  %8.2f ms  %8.1f GB/s (2x conv)\n", copy_ms, copy_gbs);
    std::printf("CEILING read  %8.2f ms  %8.1f GB/s (pure read)\n", read_ms, read_gbs);
    print_mclk_sysfs();
    cudaFree(src);
    cudaFree(dst);
    cudaFree(sink);

    // The six serving GEMVs + two unsharded context rows. T=1 (decode-chain width).
    const Shape shapes[] = {
        {"fc-stem",            5120, 10240, true},
        {"attn-qkv-pack",      3584,  5120, true},
        {"o-proj",             5120,  1536, true},
        {"gate_up",            8704,  5120, true},
        {"down",               5120,  4352, true},
        {"proposal-head",     10240,  5120, true},
        {"CTX attn-full",     14336,  5120, false},
        {"CTX gate_up-full",  34816,  5120, false},
    };
    constexpr int kNumShapes = static_cast<int>(sizeof(shapes) / sizeof(shapes[0]));

    double serving_simt_ms = 0.0;   // per-draft-forward isolated kernel sum, serving arm
    double serving_best_ms = 0.0;   // per-draft-forward best-arm sum (retune ceiling)
    int bad = 0;

    for (int si = 0; si < kNumShapes; ++si) {
        const Shape& sh = shapes[si];
        const int N = sh.n, K = sh.k, T = 1;
        const int groups = K / 32;  // W8G32_F16S: 1 B code + 2 B f16 scale per group
        const std::size_t cb = static_cast<std::size_t>(N) * K;
        const std::size_t sb = static_cast<std::size_t>(N) * groups * 2;
        const std::size_t xb = static_cast<std::size_t>(K) * 2 * T;
        const std::size_t ob = static_cast<std::size_t>(N) * 2 * T;
        const std::size_t total = cb + sb + xb + ob;

        std::vector<unsigned char> h_codes(cb);
        std::vector<std::uint16_t> h_scales(sb / 2);
        std::uint64_t seed = 0x5EED'0000'0000'07A7ull ^ (static_cast<std::uint64_t>(N) << 24) ^
                             static_cast<std::uint64_t>(K);
        for (std::size_t i = 0; i < cb; ++i) { h_codes[i] = static_cast<unsigned char>(splitmix64(seed) >> 33); }
        for (std::size_t i = 0; i < h_scales.size(); ++i) {
            unsigned char b0 = static_cast<unsigned char>(splitmix64(seed) >> 33);
            unsigned char b1 = static_cast<unsigned char>(splitmix64(seed) >> 33);
            if ((b1 & 0x7Cu) == 0x7Cu) { b1 &= 0x78u; }  // fin-style: no NaN/inf exponent pattern
            h_scales[i] = static_cast<std::uint16_t>((static_cast<std::uint16_t>(b1) << 8) | b0);
        }
        std::vector<std::uint16_t> h_x(static_cast<std::size_t>(K) * T);
        for (std::size_t i = 0; i < h_x.size(); ++i) {
            const std::uint16_t r = static_cast<std::uint16_t>(splitmix64(seed) >> 48);
            h_x[i] = static_cast<std::uint16_t>((r & 0x807Fu) |
                                                (static_cast<std::uint16_t>(118 + (r % 14)) << 7));
        }

        void *x, *codes, *scales, *outC4, *outST, *outC8;
        cudaMalloc(&x, xb);
        cudaMalloc(&codes, cb);
        cudaMalloc(&scales, sb);
        cudaMalloc(&outC4, ob);
        cudaMalloc(&outST, ob);
        cudaMalloc(&outC8, ob);
        cudaMemcpy(x, h_x.data(), xb, cudaMemcpyHostToDevice);
        cudaMemcpy(codes, h_codes.data(), cb, cudaMemcpyHostToDevice);
        cudaMemcpy(scales, h_scales.data(), sb, cudaMemcpyHostToDevice);

        ninfer::Tensor tx;
        tx.data = x; tx.ne[0] = K; tx.ne[1] = T; tx.ne[2] = 1; tx.ne[3] = 1;
        tx.dtype = ninfer::DType::BF16;
        ninfer::Weight tw;
        tw.qdata = codes; tw.scales = scales;
        tw.n = N; tw.k = K;
        tw.qtype = ninfer::QType::W8G32_F16S;
        tw.group_size = 32;
        tw.shape[0] = N; tw.shape[1] = K;
        tw.padded_shape[0] = N; tw.padded_shape[1] = K;
        tw.ndim = 2;
        tw.weight_scale_divisor = 1.0f;
        auto mk_out = [&](void* p) {
            ninfer::Tensor t;
            t.data = p; t.ne[0] = N; t.ne[1] = T; t.ne[2] = 1; t.ne[3] = 1;
            t.dtype = ninfer::DType::BF16;
            return t;
        };
        ninfer::Tensor tC4 = mk_out(outC4), tST = mk_out(outST), tC8 = mk_out(outC8);
        namespace d = ninfer::ops::detail;

        std::printf("== %s [%d,%d] T=%d ==  useful %.1f MB/call%s\n", sh.name, N, K, T, total / 1e6,
                    sh.serving ? "  (SERVING shape)" : "  (context)");
        std::printf("%-24s %10s %9s %8s %7s\n", "arm", "ms/call", "GB/s", "%read", "iters");
        int i1 = 0, i2 = 0, i3 = 0;
        const double msC4 = time_ms_auto([&] { d::launch_w8_simt_r8_c4(tx, tw, tC4, 0); }, 3, i1);
        // small_t is REGISTERED only for specific geometries (it throws "unsupported exact
        // problem" elsewhere — the serve never routes it at t<=4 on this seam anyway); a
        // throw here is recorded as n/a, not a failure.
        double msST = -1.0, msC8 = -1.0;
        try {
            msST = time_ms_auto([&] { d::launch_w8_small_t(tx, tw, tST, 0); }, 3, i2);
        } catch (const std::exception& e) {
            std::printf("   small_t: n/a (%s)\n", e.what());
        }
        try {
            msC8 = time_ms_auto([&] { d::launch_w8_simt_r8_c8(tx, tw, tC8, 0); }, 3, i3);
        } catch (const std::exception& e) {
            std::printf("   simt_r8_c8: n/a (%s)\n", e.what());
        }
        std::printf("%-24s %10.4f %9.1f %7.1f%% %7d\n", "simt_r8_c4 (serving)", msC4,
                    total / 1e6 / msC4, 100.0 * (total / 1e6 / msC4) / read_gbs, i1);
        if (msST > 0) {
            std::printf("%-24s %10.4f %9.1f %7.1f%% %7d\n", "small_t", msST,
                        total / 1e6 / msST, 100.0 * (total / 1e6 / msST) / read_gbs, i2);
        }
        if (msC8 > 0) {
            std::printf("%-24s %10.4f %9.1f %7.1f%% %7d\n", "simt_r8_c8", msC8,
                        total / 1e6 / msC8, 100.0 * (total / 1e6 / msC8) / read_gbs, i3);
        }
        double best = msC4;
        if (msST > 0 && msST < best) { best = msST; }
        if (msC8 > 0 && msC8 < best) { best = msC8; }
        if (msST > 0) {
            std::printf("   c4 vs small_t %.2fx", msST / msC4);
        } else {
            std::printf("   c4 vs small_t n/a");
        }
        if (msC8 > 0) {
            std::printf("  c4 vs c8 %.2fx", msC8 / msC4);
        } else {
            std::printf("  c4 vs c8 n/a");
        }
        std::printf("   (137 GB/s-floor bar: %.2f ms)\n", total / 1e6 / 137.0);
        if (sh.serving) {
            serving_simt_ms += msC4;
            serving_best_ms += best;
        }

        cudaDeviceSynchronize();
        std::vector<std::uint16_t> hC4(N * T);
        cudaMemcpy(hC4.data(), outC4, ob, cudaMemcpyDeviceToHost);
        if (msST > 0) {
            std::vector<std::uint16_t> hST(N * T);
            cudaMemcpy(hST.data(), outST, ob, cudaMemcpyDeviceToHost);
            const double l2st = rel_l2_bf16(hST.data(), hC4.data(), hC4.size());
            const bool okst = l2st < 1e-2;
            if (!okst) { ++bad; }
            std::printf("   rel-L2(small_t,c4)=%.3e %s", l2st, okst ? "PASS" : "FAIL");
        } else {
            std::printf("   rel-L2(small_t,c4)=n/a      ");
        }
        if (msC8 > 0) {
            std::vector<std::uint16_t> hC8(N * T);
            cudaMemcpy(hC8.data(), outC8, ob, cudaMemcpyDeviceToHost);
            const double l2c8 = rel_l2_bf16(hC8.data(), hC4.data(), hC4.size());
            const bool okc8 = l2c8 < 1e-2;
            if (!okc8) { ++bad; }
            std::printf("   rel-L2(c8,c4)=%.3e %s\n", l2c8, okc8 ? "PASS" : "FAIL");
        } else {
            std::printf("   rel-L2(c8,c4)=n/a\n");
        }

        cudaFree(x);
        cudaFree(codes);
        cudaFree(scales);
        cudaFree(outC4);
        cudaFree(outST);
        cudaFree(outC8);
    }

    std::printf("\n== DRAFT-FORWARD SUM (serving shapes, T=1, isolated kernels) ==\n");
    std::printf("  serving arm (simt_r8_c4): %.3f ms per draft forward  ->  %.3f ms per round (2 forwards)\n",
                serving_simt_ms, 2.0 * serving_simt_ms);
    std::printf("  best-arm sum:             %.3f ms per draft forward  ->  %.3f ms per round\n",
                serving_best_ms, 2.0 * serving_best_ms);
    std::printf("  serving-side banked residual for the same two forwards: align 2.156 + chain_fwd 1.393"
                " = 3.549 ms (LMHEAD_flip_row)\n");
    std::printf("  weight bytes/forward/rank: 210.5 MB; 137 GB/s floor = 1.54 ms/forward; pure-read"
                " ceiling floor = %.2f ms/forward\n", 210.5 / read_gbs);

    if (bad != 0) {
        std::printf("W7-DRAFT-ARM bench: %d rel-L2 gate(s) FAILED (bar 1e-2)\n", bad);
        return 1;
    }
    std::printf("W7-DRAFT-ARM bench: all rel-L2 gates PASS (bar 1e-2)\n");
    return 0;
}
