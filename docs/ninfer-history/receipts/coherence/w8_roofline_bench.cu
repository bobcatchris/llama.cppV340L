// w8_roofline_bench.cu — W8 sibling of nvfp4_roofline_bench.cu / nvfp4_smallt_roofline_bench.cu.
//
// LMHEAD work order (TAIL_LMHEAD_2026-09-17 §4.1, DECISIVE CHECK 1): prices the verify lm_head
// GEMM at the EXACT TP4 shard geometry n=62080, k=5120 (full-vocab ColumnN shard, W8G32_F16S,
// 337,715,200 B/rank) for T in {1, 3, 4} (T=3 = MTP k=2 verify width, PRIMARY), across the
// three arms the route table names for this problem:
//   A = launch_w8_small_t                (current serving arm; family-10 cure route)
//   B = launch_w8_simt_r8_c4             (proposed flip arm; ~174 GB/s serving-side anchor on
//                                         the same format at [10240,5120], propose field)
//   C = launch_w8_mma_r64x16_c48_k128_a1 (registered mma arm, legal t<=48; gfx900 executes it
//                                         as emulated-SIMT mma_bf16 — priced for completeness,
//                                         expected not to win; the SERVE build keeps this
//                                         symbol on the die-loud HIP link stub, the bench
//                                         links the REAL TU to price the schedule itself)
// Arms are called DIRECTLY — the timing legs are independent of NINFER_LMHEAD_ARM (which only
// the SELECTOR leg at the bottom exercises, through the real w8_dispatch.cpp select table).
//
// Same discipline as the SMALLT bench:
//   - production -O3 compile line (the -O3 LAW)
//   - 5 s continuous mclk hammer BEFORE timing + best-effort sysfs pp_dpm_mclk print (verify
//     the level-3 945 MHz park did not happen; external sampler confirmation stays GPU-owner law)
//   - hipEvent timing on the legacy stream
// Bytes convention (same as the NVFP4 benches): useful bytes per call =
//   rows*(K codes + K/32*2 scale bytes) + K*T*2 x + rows*T*2 out — graded vs the pure-read
//   ceiling measured in the same process.
// Parity legs: rel-L2(B,A) and rel-L2(C,A) on fixed-seed splitmix64 data (fin-style scales,
// production-envelope x) with the A16-criterion-class bar 1e-2 — an ARM-AGREEMENT signal only;
// the serving parity gate stays the count600 fingerprint (204 rounds / acc 0.97 / t-r 2.94).
// Exit rc 1 if any rel-L2 gate fails.
//
// Compile EXACTLY like the NVFP4 benches (the -O3 is LAW), from the worktree root $W
// (device/tensor/dtype close the core symbols the arm TUs' launch_route token-slicing needs):
//   clang++ -O3 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 -D__HIP_PLATFORM_AMD__=1 \
//     -D__HIP_ROCclr__=1 -I$W/src/common/hip_shim -I$W/include -I$W/src -I$W/third_party \
//     -DNDEBUG -std=gnu++20 --offload-arch=gfx900 -x hip \
//     $W/results/amd/coherence/w8_roofline_bench.cu $W/src/core/device.cu \
//     $W/src/core/tensor.cpp $W/src/core/dtype.cpp -o w8_roofline_bench
//
// Run (GPU window only; server keeps its grant — this bench claims no port, allocates ~700 MB):
//   ./w8_roofline_bench                      # arm A/B/C timing + parity + selector (default gate)
//   NINFER_LMHEAD_ARM=simt ./w8_roofline_bench   # selector leg under the route flip

#include "ops/linear/w8/w8_rowsplit_gemm_mma.cu"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cu"
#include "ops/linear/w8/w8_small_t.cu"
#include "ops/linear/w8/w8_dispatch.cpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <vector>

// ---- link closure for select_w8_a16_launch: arms held out of the HIP whitelist whose TUs this
// bench does NOT compile (their schedules are not on any (62080,5120,T<=4) route). Definitions
// reproduce the serve build's own die-loud stub discipline: if a route change ever reaches one
// from this bench, it dies naming the arm instead of silently timing the wrong kernel.
namespace ninfer::ops::detail {
[[noreturn]] static void bench_stub_die(const char* arm) {
    std::fprintf(stderr, "w8_roofline_bench: route reached held arm '%s' — FAIL LOUD\n", arm);
    std::abort();
}
void launch_w8_decode_r4(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    bench_stub_die("launch_w8_decode_r4");
}
void launch_w8_exact_t_splitk(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    bench_stub_die("launch_w8_exact_t_splitk");
}
void launch_w8_exact_t_composite(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    bench_stub_die("launch_w8_exact_t_composite");
}
void launch_w8_dflash_medium(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    bench_stub_die("launch_w8_dflash_medium");
}
void launch_w8_medium_splitk_c144(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    bench_stub_die("launch_w8_medium_splitk_c144");
}
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
    // keep each arm's timed section ~1.5 s (small_t at 62 ms/call => ~25 iters; fast arms => 200)
    int iters = static_cast<int>(1500.0 / (ms_probe > 0.05 ? ms_probe : 0.05));
    if (iters < 10) { iters = 10; }
    if (iters > 200) { iters = 200; }
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

const char* arm_name(const void* fp) {
    if (fp == reinterpret_cast<const void*>(&ninfer::ops::detail::launch_w8_small_t)) {
        return "launch_w8_small_t";
    }
    if (fp == reinterpret_cast<const void*>(&ninfer::ops::detail::launch_w8_simt_r8_c4)) {
        return "launch_w8_simt_r8_c4";
    }
    if (fp ==
        reinterpret_cast<const void*>(&ninfer::ops::detail::launch_w8_mma_r64x16_c48_k128_a1)) {
        return "launch_w8_mma_r64x16_c48_k128_a1";
    }
    if (fp == reinterpret_cast<const void*>(&ninfer::ops::detail::launch_w8_simt_r8_c8)) {
        return "launch_w8_simt_r8_c8";
    }
    return "OTHER";
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
    std::printf("device %s SMs=%d  NINFER_LMHEAD_ARM=%s\n", prop.name, prop.multiProcessorCount,
                std::getenv("NINFER_LMHEAD_ARM") != nullptr ? std::getenv("NINFER_LMHEAD_ARM")
                                                            : "<unset=smallt>");

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
    int ceil_iters_unused_a = 0, ceil_iters_unused_b = 0;
    const double copy_ms =
        time_ms_auto([&] { copy_kernel<<<grid, 256>>>(src, dst, n4); }, 3, ceil_iters_unused_a);
    const double read_ms =
        time_ms_auto([&] { read_kernel<<<grid, 256>>>(src, sink, n4); }, 3, ceil_iters_unused_b);    const double copy_gbs = 2.0 * bytes / 1e6 / copy_ms;
    const double read_gbs = bytes / 1e6 / read_ms;
    std::printf("CEILING copy  %8.2f ms  %8.1f GB/s (2x conv)\n", copy_ms, copy_gbs);
    std::printf("CEILING read  %8.2f ms  %8.1f GB/s (pure read)\n", read_ms, read_gbs);
    print_mclk_sysfs();

    // ---- the LMHEAD TP4 shard problem, EXACTLY as serving binds it (k=5120, n=248320/4)
    constexpr int kN = 62080;
    constexpr int kK = 5120;
    constexpr int kGroups = kK / 32;  // 160 groups/row, W8G32_F16S: 1 B code + 2 B f16 scale
    const std::size_t code_bytes = static_cast<std::size_t>(kN) * kK;             // 317,849,600
    const std::size_t scale_bytes = static_cast<std::size_t>(kN) * kGroups * 2;   //  19,865,600
    const std::size_t weight_bytes = code_bytes + scale_bytes;                    // 337,715,200
    std::printf("LMHEAD shard [%d,%d] W8G32_F16S: codes %.1f MB + scales %.1f MB = %.1f MB/rank\n",
                kN, kK, code_bytes / 1e6, scale_bytes / 1e6, weight_bytes / 1e6);

    // fixed-seed host fill: s8 codes full-range, fin-style f16 scales, production-envelope x
    std::vector<unsigned char> h_codes(code_bytes);
    std::vector<std::uint16_t> h_scales(scale_bytes / 2);
    std::uint64_t seed = 0x5EED'0000'0000'0620ull ^ (static_cast<std::uint64_t>(kN) << 24) ^ kK;
    for (std::size_t i = 0; i < code_bytes; ++i) {
        h_codes[i] = static_cast<unsigned char>(splitmix64(seed) >> 33);
    }
    for (std::size_t i = 0; i < h_scales.size(); ++i) {
        unsigned char b0 = static_cast<unsigned char>(splitmix64(seed) >> 33);
        unsigned char b1 = static_cast<unsigned char>(splitmix64(seed) >> 33);
        if ((b1 & 0x7Cu) == 0x7Cu) { b1 &= 0x78u; }  // fin-style: no NaN/inf exponent pattern
        h_scales[i] = static_cast<std::uint16_t>((static_cast<std::uint16_t>(b1) << 8) | b0);
    }

    const int ts[] = {3, 1, 4};  // T=3 primary (MTP k=2 verify width), then context
    int bad = 0;
    for (int ti = 0; ti < 3; ++ti) {
        const int T = ts[ti];
        const std::size_t x_bytes = static_cast<std::size_t>(kK) * 2 * T;
        const std::size_t out_bytes = static_cast<std::size_t>(kN) * 2 * T;
        const std::size_t total = weight_bytes + x_bytes + out_bytes;

        std::vector<std::uint16_t> h_x(static_cast<std::size_t>(kK) * T);
        for (std::size_t i = 0; i < h_x.size(); ++i) {
            // production pre-norm envelope (PK-bench convention): exp field 118..131
            const std::uint16_t r = static_cast<std::uint16_t>(splitmix64(seed) >> 48);
            h_x[i] = static_cast<std::uint16_t>((r & 0x807Fu) |
                                                (static_cast<std::uint16_t>(118 + (r % 14)) << 7));
        }

        void *x, *codes, *scales, *outA, *outB, *outC;
        cudaMalloc(&x, x_bytes);
        cudaMalloc(&codes, code_bytes);
        cudaMalloc(&scales, scale_bytes);
        cudaMalloc(&outA, out_bytes);
        cudaMalloc(&outB, out_bytes);
        cudaMalloc(&outC, out_bytes);
        cudaMemcpy(x, h_x.data(), x_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(codes, h_codes.data(), code_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(scales, h_scales.data(), scale_bytes, cudaMemcpyHostToDevice);

        ninfer::Tensor tx;
        tx.data = x;
        tx.ne[0] = kK;
        tx.ne[1] = T;
        tx.ne[2] = 1;
        tx.ne[3] = 1;
        tx.dtype = ninfer::DType::BF16;
        ninfer::Weight tw;
        tw.qdata = codes;
        tw.scales = scales;
        tw.n = kN;
        tw.k = kK;
        tw.qtype = ninfer::QType::W8G32_F16S;
        tw.group_size = 32;
        tw.shape[0] = kN;
        tw.shape[1] = kK;
        tw.padded_shape[0] = kN;
        tw.padded_shape[1] = kK;
        tw.ndim = 2;
        tw.weight_scale_divisor = 1.0f;
        auto mk_out = [&](void* p) {
            ninfer::Tensor t;
            t.data = p;
            t.ne[0] = kN;
            t.ne[1] = T;
            t.ne[2] = 1;
            t.ne[3] = 1;
            t.dtype = ninfer::DType::BF16;
            return t;
        };
        ninfer::Tensor tA = mk_out(outA), tB = mk_out(outB), tC = mk_out(outC);
        namespace d = ninfer::ops::detail;

        std::printf("== T=%d ==  useful %.1f MB/call\n", T, total / 1e6);
        std::printf("%-36s %10s %9s %8s %7s\n", "arm", "ms/call", "GB/s", "%read", "iters");
        int itersA = 0, itersB = 0, itersC = 0;
        const double msA = time_ms_auto([&] { d::launch_w8_small_t(tx, tw, tA, 0); }, 3, itersA);
        const double msB = time_ms_auto([&] { d::launch_w8_simt_r8_c4(tx, tw, tB, 0); }, 3, itersB);
        const double msC =
            time_ms_auto([&] { d::launch_w8_mma_r64x16_c48_k128_a1(tx, tw, tC, 0); }, 3, itersC);
        std::printf("%-36s %10.4f %9.1f %7.1f%% %7d\n", "A launch_w8_small_t", msA,
                    total / 1e6 / msA, 100.0 * (total / 1e6 / msA) / read_gbs, itersA);
        std::printf("%-36s %10.4f %9.1f %7.1f%% %7d\n", "B launch_w8_simt_r8_c4", msB,
                    total / 1e6 / msB, 100.0 * (total / 1e6 / msB) / read_gbs, itersB);
        std::printf("%-36s %10.4f %9.1f %7.1f%% %7d\n", "C launch_w8_mma_r64x16_c48_k128_a1", msC,
                    total / 1e6 / msC, 100.0 * (total / 1e6 / msC) / read_gbs, itersC);
        std::printf("   B/A %.2fx  C/A %.2fx  (B's measured ms IS the predicted serving "
                    "lm_head at this geometry: %.2f ms)\n",
                    msA / msB, msA / msC, msB);

        cudaDeviceSynchronize();
        // arm-agreement legs (statistical; NOT the serving parity gate)
        std::vector<std::uint16_t> hA(kN * T), hB(kN * T), hC(kN * T);
        cudaMemcpy(hA.data(), outA, out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hB.data(), outB, out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hC.data(), outC, out_bytes, cudaMemcpyDeviceToHost);
        const double l2ba = rel_l2_bf16(hB.data(), hA.data(), hA.size());
        const double l2ca = rel_l2_bf16(hC.data(), hA.data(), hA.size());
        const bool okB = l2ba < 1e-2, okC = l2ca < 1e-2;
        if (!okB) { ++bad; }
        if (!okC) { ++bad; }
        std::printf("   rel-L2(B,A)=%.3e %s   rel-L2(C,A)=%.3e %s  (bar 1e-2, arm-agreement only)\n",
                    l2ba, okB ? "PASS" : "FAIL", l2ca, okC ? "PASS" : "FAIL");

        cudaFree(x);
        cudaFree(codes);
        cudaFree(scales);
        cudaFree(outA);
        cudaFree(outB);
        cudaFree(outC);
    }

    // ---- serving-side cross-anchor: the draft-slice geometry the ~174 GB/s propose number
    // rode ([10240,5120] via simt_r8_c4) — same format, 1/6.06 the rows, T=3.
    {
        constexpr int kNAnchor = 10240;
        const int T = 3;
        const std::size_t cb = static_cast<std::size_t>(kNAnchor) * kK;
        const std::size_t sb = static_cast<std::size_t>(kNAnchor) * kGroups * 2;
        const std::size_t xb = static_cast<std::size_t>(kK) * 2 * T;
        const std::size_t ob = static_cast<std::size_t>(kNAnchor) * 2 * T;
        const std::size_t total = cb + sb + xb + ob;
        void *x, *codes, *scales, *out;
        cudaMalloc(&x, xb);
        cudaMalloc(&codes, cb);
        cudaMalloc(&scales, sb);
        cudaMalloc(&out, ob);
        cudaMemset(codes, 0x11, cb);
        cudaMemset(scales, 0x38, sb);
        cudaMemset(x, 0, xb);
        ninfer::Tensor tx;
        tx.data = x; tx.ne[0] = kK; tx.ne[1] = T; tx.ne[2] = 1; tx.ne[3] = 1;
        tx.dtype = ninfer::DType::BF16;
        ninfer::Weight tw;
        tw.qdata = codes; tw.scales = scales;
        tw.n = kNAnchor; tw.k = kK;
        tw.qtype = ninfer::QType::W8G32_F16S;
        tw.group_size = 32;
        tw.shape[0] = kNAnchor; tw.shape[1] = kK;
        tw.padded_shape[0] = kNAnchor; tw.padded_shape[1] = kK;
        tw.ndim = 2;
        tw.weight_scale_divisor = 1.0f;
        ninfer::Tensor tout;
        tout.data = out; tout.ne[0] = kNAnchor; tout.ne[1] = T; tout.ne[2] = 1; tout.ne[3] = 1;
        tout.dtype = ninfer::DType::BF16;
        int iters = 0;
        const double ms = time_ms_auto(
            [&] { ninfer::ops::detail::launch_w8_simt_r8_c4(tx, tw, tout, 0); }, 3, iters);
        std::printf("ANCHOR draft-slice [%d,%d] T=%d simt_r8_c4: %8.4f ms  %8.1f GB/s "
                    "(serving propose field was ~174 GB/s incl. argmax+remap)\n",
                    kNAnchor, kK, T, ms, total / 1e6 / ms);
        cudaFree(x);
        cudaFree(codes);
        cudaFree(scales);
        cudaFree(out);
    }

    // ---- SELECTOR leg: the REAL w8_dispatch.cpp route table under this process's gate env.
    // The gate is a process-lifetime magic static, so the flip is proven by RE-RUNNING the
    // binary with NINFER_LMHEAD_ARM=simt (see header), not by mutating env mid-process.
    std::printf("SELECTOR select_w8_a16_launch under NINFER_LMHEAD_ARM=%s:\n",
                std::getenv("NINFER_LMHEAD_ARM") != nullptr ? std::getenv("NINFER_LMHEAD_ARM")
                                                            : "<unset=smallt>");
    namespace d = ninfer::ops::detail;
    const int sel_ts[] = {1, 3, 4, 33, 40};
    for (int t : sel_ts) {
        const d::W8Launch fp = d::select_w8_a16_launch(kN, kK, t);
        std::printf("  lm_head (62080,5120) T=%-3d -> %s\n", t,
                    arm_name(reinterpret_cast<const void*>(fp)));
    }
    for (int t : {1, 3, 4}) {
        const d::W8Launch fp = d::select_w8_a16_launch(5120, 10240, t);
        std::printf("  draft-fc (5120,10240) T=%-3d -> %s\n", t,
                    arm_name(reinterpret_cast<const void*>(fp)));
    }

    // ---- DRAFT-FC leg (coordinator scope extension, DRAIN_FIX_NOTES D2 commit 981386566):
    // the stem input projection [5120,10240] is the ONLY draft T=1 forward on the
    // select_w8_a16_launch seam (the other four draft linears ride multi_gpu::tp_gemv, which
    // already routes W8 t<=4 to simt_r8_c4). A = small_t (current T=1 route), B = the flip arm.
    {
        constexpr int kNFc = 5120;
        constexpr int kKFc = 10240;
        constexpr int kGroupsFc = kKFc / 32;
        const int T = 1;
        const std::size_t cb = static_cast<std::size_t>(kNFc) * kKFc;
        const std::size_t sb = static_cast<std::size_t>(kNFc) * kGroupsFc * 2;
        const std::size_t xb = static_cast<std::size_t>(kKFc) * 2 * T;
        const std::size_t ob = static_cast<std::size_t>(kNFc) * 2 * T;
        const std::size_t total = cb + sb + xb + ob;
        std::vector<unsigned char> fc_codes(cb);
        std::vector<std::uint16_t> fc_scales(sb / 2);
        std::uint64_t fseed = 0x5EED'0000'0000'0FC1ull ^ (static_cast<std::uint64_t>(kNFc) << 24);
        for (std::size_t i = 0; i < cb; ++i) { fc_codes[i] = static_cast<unsigned char>(splitmix64(fseed) >> 33); }
        for (std::size_t i = 0; i < fc_scales.size(); ++i) {
            unsigned char b0 = static_cast<unsigned char>(splitmix64(fseed) >> 33);
            unsigned char b1 = static_cast<unsigned char>(splitmix64(fseed) >> 33);
            if ((b1 & 0x7Cu) == 0x7Cu) { b1 &= 0x78u; }
            fc_scales[i] = static_cast<std::uint16_t>((static_cast<std::uint16_t>(b1) << 8) | b0);
        }
        std::vector<std::uint16_t> fc_x(static_cast<std::size_t>(kKFc) * T);
        for (std::size_t i = 0; i < fc_x.size(); ++i) {
            const std::uint16_t r = static_cast<std::uint16_t>(splitmix64(fseed) >> 48);
            fc_x[i] = static_cast<std::uint16_t>((r & 0x807Fu) |
                                                 (static_cast<std::uint16_t>(118 + (r % 14)) << 7));
        }
        void *x, *codes, *scales, *outA, *outB;
        cudaMalloc(&x, xb);
        cudaMalloc(&codes, cb);
        cudaMalloc(&scales, sb);
        cudaMalloc(&outA, ob);
        cudaMalloc(&outB, ob);
        cudaMemcpy(x, fc_x.data(), xb, cudaMemcpyHostToDevice);
        cudaMemcpy(codes, fc_codes.data(), cb, cudaMemcpyHostToDevice);
        cudaMemcpy(scales, fc_scales.data(), sb, cudaMemcpyHostToDevice);
        ninfer::Tensor tx;
        tx.data = x; tx.ne[0] = kKFc; tx.ne[1] = T; tx.ne[2] = 1; tx.ne[3] = 1;
        tx.dtype = ninfer::DType::BF16;
        ninfer::Weight tw;
        tw.qdata = codes; tw.scales = scales;
        tw.n = kNFc; tw.k = kKFc;
        tw.qtype = ninfer::QType::W8G32_F16S;
        tw.group_size = 32;
        tw.shape[0] = kNFc; tw.shape[1] = kKFc;
        tw.padded_shape[0] = kNFc; tw.padded_shape[1] = kKFc;
        tw.ndim = 2;
        tw.weight_scale_divisor = 1.0f;
        ninfer::Tensor tA, tB;
        tA.data = outA; tA.ne[0] = kNFc; tA.ne[1] = T; tA.ne[2] = 1; tA.ne[3] = 1;
        tA.dtype = ninfer::DType::BF16;
        tB.data = outB; tB.ne[0] = kNFc; tB.ne[1] = T; tB.ne[2] = 1; tB.ne[3] = 1;
        tB.dtype = ninfer::DType::BF16;
        std::printf("== DRAFT-FC [%d,%d] T=%d ==  useful %.1f MB/call\n", kNFc, kKFc, T,
                    total / 1e6);
        std::printf("%-36s %10s %9s %8s %7s\n", "arm", "ms/call", "GB/s", "%read", "iters");
        int itersA = 0, itersB = 0;
        const double msA = time_ms_auto(
            [&] { ninfer::ops::detail::launch_w8_small_t(tx, tw, tA, 0); }, 3, itersA);
        const double msB = time_ms_auto(
            [&] { ninfer::ops::detail::launch_w8_simt_r8_c4(tx, tw, tB, 0); }, 3, itersB);
        std::printf("%-36s %10.4f %9.1f %7.1f%% %7d\n", "A launch_w8_small_t", msA,
                    total / 1e6 / msA, 100.0 * (total / 1e6 / msA) / read_gbs, itersA);
        std::printf("%-36s %10.4f %9.1f %7.1f%% %7d\n", "B launch_w8_simt_r8_c4", msB,
                    total / 1e6 / msB, 100.0 * (total / 1e6 / msB) / read_gbs, itersB);
        std::printf("   B/A %.2fx  (align/chain_fwd prediction: saves (msA-msB) per draft "
                    "forward ON THIS SEAM only)\n", msA / msB);
        cudaDeviceSynchronize();
        std::vector<std::uint16_t> hA(kNFc * T), hB(kNFc * T);
        cudaMemcpy(hA.data(), outA, ob, cudaMemcpyDeviceToHost);
        cudaMemcpy(hB.data(), outB, ob, cudaMemcpyDeviceToHost);
        const double l2ba = rel_l2_bf16(hB.data(), hA.data(), hA.size());
        const bool ok = l2ba < 1e-2;
        if (!ok) { ++bad; }
        std::printf("   rel-L2(B,A)=%.3e %s  (bar 1e-2, arm-agreement only)\n", l2ba,
                    ok ? "PASS" : "FAIL");
        cudaFree(x);
        cudaFree(codes);
        cudaFree(scales);
        cudaFree(outA);
        cudaFree(outB);
    }

    if (bad != 0) {
        std::printf("W8-LMHEAD bench: %d rel-L2 gate(s) FAILED (bar 1e-2)\n", bad);
        return 1;
    }
    std::printf("W8-LMHEAD bench: all rel-L2 gates PASS (bar 1e-2)\n");
    return 0;
}
