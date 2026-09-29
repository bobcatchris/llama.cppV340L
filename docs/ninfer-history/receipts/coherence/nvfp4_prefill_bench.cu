// nvfp4_prefill_bench.cu — PREFILL-FRONT roofline extension (2026-09-17, no-GPU desk).
//
// Which kernel serves NVFP4 linear at TP4 prefill (M = chunk = 128 tokens/forward) on the
// HIP lane? Dispatch truth (so the arm benched below IS production prefill):
//   variant_kernels.cpp:63-87 binds A16Only under NINFER_NVFP4_SIMT_LANE
//   -> resolve_nvfp4_route (nvfp4_config.h:202) returns A16 for every problem
//   -> launch_a16 (nvfp4_dispatch.cpp:26-42) splits M into <=32-token chunk views
//   -> launch_nvfp4_small_t -> nvfp4_small_t_hip.cu (P3 SIMT port: one warp per (row,token),
//      weights RE-READ per token; its header names itself "The PRE-FILL wall", "PERF
//      NAMED-IMMATURE"). Every W4A4/fused arm is a link stub that dies
//      (apps/serve/hip_link_stub_arms.cpp:65-82); mma/tma arms cannot exist on gfx900
//      (no matrix instructions — docs/01 §2). Full chain: PREFILL_FRONT_2026-09-17.md.
//
// Measures, on ONE gfx900 die (discipline copied from nvfp4_roofline_bench.cu — ROOFLINE_row:
// copy 369.5, read 377.9 GB/s, 256 MiB cells, 5 s mclk-ramp hammer, event timing):
//   1. copy + read ceilings.
//   2. PRODUCTION prefill arm: launch_nvfp4_small_t driven exactly like launch_a16's T=32
//      chunking, M in {32,128,256} on the four big registered geometries.
//   3. REFERENCE tiled dequant GEMM, inline, LABELED REFERENCE: NOT production, NOT
//      order-contract-clean, ungraded. Standard tiled kernel: per 32-row block each K-slab
//      of weights is dequanted ONCE into LDS fp32 (same amd codec + nvfp4_scale_offset
//      planes as production), fp32 accum, every weight byte then serves all M tokens. This
//      prices the ceiling envelope doc 01 §2 sketches for the M>8 band ("the M>8 band
//      becomes ... a plain tiled LDS GEMM" — never built on this lane).
//   4. TILED arm (TILED-GEMM-1, 2026-09-17): the production candidate from
//      src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu (64-row x M-token blocks, activations
//      staged in LDS packed bf16 with XOR swizzle, weights streamed via the LDS decode
//      tables) driven WHOLE-M (no chunking), plus a per-shape rel-L2 self-check against the
//      REFERENCE arm (PASS bar < 1e-2).
//   5. TILED-PK mode (--pk, 2026-09-17): V0 vs V0-PK (HFMA2 tiled arm, PREFILL_GEMM_SCALE
//      2026-09-17 §2/§5, spec commit 9a38c0c50) at M=128 on the FIVE W4 SHARD geometries,
//      W in {16,32,64}; rel-L2(PK, V0) gate 1e-2 every row, xV0 ratio vs the pre-registered
//      floors (>= 1.8x pass / all < 1.3x dead), max|window| census, mclk hammer + sysfs
//      print + event timing. NO FNV column (bit-identity is false BY DESIGN under PK).
// Past the M-wall (doc 01 §3: M* ~ 8-16 here) prefill is COMPUTE-bound: GB/s should FALL
// with M while GFLOP/s rises. Decisive column: REF vs PROD GFLOP/s at M=128.
//
// -O3 LAW (ROOFLINE_row VOID note): compile with EXACTLY the production -O3 flag set; an
// -O0 build already produced garbage ceilings once. Do not run without a GPU grant.

#include "ops/linear/nvfp4/nvfp4_small_t_hip.cu"
// TILED arm (TILED-GEMM-1): the production large-M tiled dequant GEMM — activations in LDS,
// weights streamed through the LDS decode tables. launcher: launch_nvfp4_tiled_gemm.
#include "ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
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
    if (acc == 0xDEADBEEFu) { sink[0] = acc; } // never true; defeats DCE
}

constexpr int kBM = 32;      // REFERENCE: output rows per block tile
constexpr int kBK = 128;     // K-slab per LDS stage
constexpr int kThreads = 256; // 32 rows x 8 token-stripes

// REFERENCE tiled dequant GEMM. STILES = K/64 (scale-plane tiles), M_ = tokens, TPT = M_/8
// fp32 accumulators/thread. Block tile kBM x M_; K loop inside; weights dequant once/block.
template <int STILES, int M_, int TPT>
__global__ void __launch_bounds__(kThreads) ref_tiled_dequant_gemm(
    const std::uint16_t* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inv_div, std::uint16_t* __restrict__ out,
    int n_rows, int k_dim) {
    __shared__ float wtile[kBM][kBK];
    namespace amd = ninfer::ops::detail::amd;
    const int tid  = threadIdx.x;
    const int row  = tid >> 3;                // row within block 0..31
    const int lane = tid & 7;                 // token stripe
    const int grow = blockIdx.x * kBM + row;  // global output row
    float acc[TPT];
#pragma unroll
    for (int i = 0; i < TPT; ++i) { acc[i] = 0.f; }
    const std::uint8_t* row_codes = codes + static_cast<std::int64_t>(grow) * (k_dim / 2);
    for (int slab = 0; slab < k_dim / kBK; ++slab) {
        { // Phase A: one 16-value group per thread -> dequant ONCE into LDS (amd codec planes).
            const int g = slab * (kBK / 16) + lane;
            const float coeff = amd::bits_to_f32(amd::e4m3_lut_decode(
                scales[amd::nvfp4_scale_offset<STILES>(grow, g)])) * inv_div;
            const std::uint64_t c8 =
                *reinterpret_cast<const std::uint64_t*>(row_codes + static_cast<std::int64_t>(g) * 8);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const std::uint32_t p = static_cast<std::uint32_t>(c8 >> (8 * v)) & 0xFFu;
                wtile[row][lane * 16 + 2 * v] = amd::bits_to_f32(amd::e2m1_bits(p & 0xFu)) * coeff;
                wtile[row][lane * 16 + 2 * v + 1] = amd::bits_to_f32(amd::e2m1_bits(p >> 4)) * coeff;
            }
        }
        __syncthreads();
        for (int kk = 0; kk < kBK; ++kk) { // Phase C: each weight byte serves ALL M_ tokens.
            const float wn = wtile[row][kk];
            const std::uint16_t* xk = x + static_cast<std::int64_t>(lane) * k_dim + slab * kBK + kk;
#pragma unroll
            for (int i = 0; i < TPT; ++i) {
                const float xv = amd::bits_to_f32(
                    static_cast<std::uint32_t>(xk[static_cast<std::int64_t>(8 * i) * k_dim]) << 16);
                acc[i] = fmaf(wn, xv, acc[i]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TPT; ++i) {
        // BIT-CAST fix (2026-09-17): the old static_cast<uint16_t>(__nv_bfloat16) truncated
        // through the float conversion — REF agreed with TILED's truncated garbage and the
        // relL2 gate validated garbage-vs-garbage (incident in TILED_row.txt).
        out[static_cast<std::int64_t>(lane + 8 * i) * n_rows + grow] =
            __bfloat16_as_ushort(__float2bfloat16_rn(acc[i]));
    }
}

template <int STILES, int M_>
void run_ref(const void* x, const void* codes, const void* scales, float inv_div, void* out,
             int n_rows, int k_dim, cudaStream_t stream) {
    ref_tiled_dequant_gemm<STILES, M_, M_ / 8><<<n_rows / kBM, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(x), static_cast<const std::uint8_t*>(codes),
        static_cast<const std::uint8_t*>(scales), inv_div, static_cast<std::uint16_t*>(out),
        n_rows, k_dim);
}

// Production prefill arm: EXACTLY launch_a16's chunking (nvfp4_dispatch.cpp:26-42).
void run_prod(void* x, const ninfer::Weight& tw, void* out, int m, cudaStream_t stream) {
    constexpr std::int32_t kChunk = ninfer::ops::detail::kNvfp4LastSmallT; // 32
    for (std::int32_t begin = 0; begin < m; begin += kChunk) {
        ninfer::Tensor xc;
        xc.data = static_cast<std::uint8_t*>(x) + static_cast<std::int64_t>(begin) * tw.k * 2;
        xc.ne[0] = tw.k; xc.ne[1] = kChunk; xc.ne[2] = 1; xc.ne[3] = 1;
        xc.dtype = ninfer::DType::BF16;
        ninfer::Tensor xo;
        xo.data = static_cast<std::uint8_t*>(out) + static_cast<std::int64_t>(begin) * tw.n * 2;
        xo.ne[0] = tw.n; xo.ne[1] = kChunk; xo.ne[2] = 1; xo.ne[3] = 1;
        xo.dtype = ninfer::DType::BF16;
        ninfer::ops::detail::launch_nvfp4_small_t(xc, tw, xo, stream); // throws on mismatch
    }
}

// TILED arm (TILED-GEMM-1): the WHOLE M in one launch (no chunking — that is the point).
// Layout matches the REFERENCE arm / production chunk views: x[t*k + k], out[t*n + row].
void run_tiled(void* x, const ninfer::Weight& tw, void* out, int m, cudaStream_t stream) {
    ninfer::Tensor xc;
    xc.data = x;
    xc.ne[0] = tw.k; xc.ne[1] = m; xc.ne[2] = 1; xc.ne[3] = 1;
    xc.dtype = ninfer::DType::BF16;
    ninfer::Tensor xo;
    xo.data = out;
    xo.ne[0] = tw.n; xo.ne[1] = m; xo.ne[2] = 1; xo.ne[3] = 1;
    xo.dtype = ninfer::DType::BF16;
    ninfer::ops::detail::launch_nvfp4_tiled_gemm(xc, tw, xo, stream); // throws on mismatch
}

// TILED-SWEEP arm (2026-09-17): run_tiled with a runtime-selected tile shape (same layout
// contract; ids 0..4 per nvfp4_tiled_gemm_hip.h — id 0 IS the shipped path).
void run_tiled_variant(void* x, const ninfer::Weight& tw, void* out, int m, cudaStream_t stream,
                       int variant_id) {
    ninfer::Tensor xc;
    xc.data = x;
    xc.ne[0] = tw.k; xc.ne[1] = m; xc.ne[2] = 1; xc.ne[3] = 1;
    xc.dtype = ninfer::DType::BF16;
    ninfer::Tensor xo;
    xo.data = out;
    xo.ne[0] = tw.n; xo.ne[1] = m; xo.ne[2] = 1; xo.ne[3] = 1;
    xo.dtype = ninfer::DType::BF16;
    ninfer::ops::detail::launch_nvfp4_tiled_gemm_variant(xc, tw, xo, stream, variant_id);
}

// TILED-PK arm ("V0-PK", 2026-09-17): run_tiled with the packed-fp16 windowed arithmetic at
// V0's exact tile shape (window_w in {16,32,64}; 32 = the design's recommended).
// docs/amd/PREFILL_GEMM_SCALE_2026-09-17.md §2.
void run_tiled_pk(void* x, const ninfer::Weight& tw, void* out, int m, cudaStream_t stream,
                  int window_w) {
    ninfer::Tensor xc;
    xc.data = x;
    xc.ne[0] = tw.k; xc.ne[1] = m; xc.ne[2] = 1; xc.ne[3] = 1;
    xc.dtype = ninfer::DType::BF16;
    ninfer::Tensor xo;
    xo.data = out;
    xo.ne[0] = tw.n; xo.ne[1] = m; xo.ne[2] = 1; xo.ne[3] = 1;
    xo.dtype = ninfer::DType::BF16;
    ninfer::ops::detail::launch_nvfp4_tiled_gemm_pk(xc, tw, xo, stream, window_w);
}

// Best-effort sysfs mclk print (house discipline, w8_roofline_bench.cu form): verifies the
// level-3 park did not happen for THIS process's window; external sampler confirmation stays
// GPU-owner law.
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

constexpr double kNominalF32Peak = 10.75e12; // [inference] doc 01 §3 arithmetic @1.5 GHz, 56 CU

// TILED-SWEEP mode (--sweep argv flag): V0..V4 back-to-back at M=128 on the four big
// geometries, each timed against the REFERENCE arm and rel-L2-checked (same 1e-2 bar as
// the default mode's TILED arm). Ceilings + mclk-ramp hammer: the SAME discipline as the
// default mode so GF/s numbers are comparable across modes. Default mode untouched.
int run_sweep(const cudaDeviceProp& prop) {
    struct Prob { const char* name; int rows, k, stiles; };
    const Prob probs[] = {
        {"AttnInput  ", 14336, 5120, 80},
        {"GdnInput   ", 16384, 5120, 80},
        {"MlpGateUp  ", 34816, 5120, 80},
        {"Residual6144", 5120, 6144, 96},
    };
    struct VarDesc { int id; const char* tag; };
    const VarDesc vars[] = {
        {0, "V0 shipped  64r xTN  4rxTN/16t BK64 "},
        {1, "V1 skinny  32r xTN  4rxTN/32t BK64 "},
        {2, "V2 tall   128r xTN  4rxTN/8t  BK64 "},
        {3, "V3 wideK   64r xTN  4rxTN/16t BK128"},
        {4, "V4 dblbuf  64r xTN  4rxTN/16t BK64 "},
    };
    constexpr int m = 128; // the production chunk; the whole sweep lives past the M-wall
    std::printf("SWEEP TILED tile-shape variants at M=%d (GF/s; relL2 vs REF bar 1e-2; all "
                "ids share the k-ascending fp32 order = bit-identical outputs)\n", m);

    // ---- ceilings + 5 s mclk-ramp hammer (identical cells to the default mode) ----------
    const std::size_t bytes = static_cast<std::size_t>(256) * 1024 * 1024;
    const std::size_t n4 = bytes / 4;
    std::uint32_t *src, *dst, *sink;
    cudaMalloc(&src, bytes);
    cudaMalloc(&dst, bytes);
    cudaMalloc(&sink, 4);
    cudaMemset(src, 0x5A, bytes);
    const int grid = prop.multiProcessorCount * 8;
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
        cudaEventDestroy(w0); cudaEventDestroy(w1);
    }
    const double copy_gbs = 2.0 * bytes / 1e6 / time_ms([&] { copy_kernel<<<grid, 256>>>(src, dst, n4); }, 3, 30);
    const double read_gbs = bytes / 1e6 / time_ms([&] { read_kernel<<<grid, 256>>>(src, sink, n4); }, 3, 30);
    std::printf("CEILING copy %8.1f GB/s | read %8.1f GB/s\n", copy_gbs, read_gbs);
    cudaFree(src); cudaFree(dst); cudaFree(sink);

    for (const Prob& p : probs) {
        const std::size_t code_bytes  = static_cast<std::size_t>(p.rows) * (p.k / 2);
        const std::size_t scale_bytes = static_cast<std::size_t>(p.rows) * (p.k / 16);
        void *x, *codes, *scales, *out;
        cudaMalloc(&x, static_cast<std::size_t>(p.k) * 256 * 2);
        cudaMalloc(&codes, code_bytes);
        cudaMalloc(&scales, scale_bytes);
        cudaMalloc(&out, static_cast<std::size_t>(p.rows) * 256 * 2);
        cudaMemset(codes, 0x11, code_bytes);   // valid e2m1 pairs, nonzero
        cudaMemset(scales, 0x38, scale_bytes); // ~1.0-ish e4m3
        cudaMemset(x, 0x3C, static_cast<std::size_t>(p.k) * 256 * 2); // small nonzero bf16
        ninfer::Weight tw;
        tw.qdata = codes;
        tw.scales = scales;
        tw.n = p.rows;
        tw.k = p.k;
        tw.weight_scale_divisor = 1.0f;
        const std::size_t nel   = static_cast<std::size_t>(p.rows) * m;
        const double gflops     = 2.0 * p.rows * p.k * m / 1e9; // per call
        std::vector<std::uint16_t> href(nel);
        std::vector<std::uint16_t> hvar(nel);

        // REFERENCE arm: timing anchor + rel-L2 semantic anchor for every variant row.
        auto fire_ref = [&] {
            if (p.stiles == 96) { run_ref<96, 128>(x, codes, scales, 1.0f, out, p.rows, p.k, 0); }
            else { run_ref<80, 128>(x, codes, scales, 1.0f, out, p.rows, p.k, 0); }
        };
        fire_ref(); // shape self-check
        const double rms = time_ms(fire_ref, 3, 20);
        std::printf("%-13s n=%5d k=%5d M=%3d | REF %7.3fms %6.1f GF/s\n", p.name, p.rows, p.k,
                    m, rms, gflops / rms * 1e3);
        fire_ref();
        cudaMemcpy(href.data(), out, nel * 2, cudaMemcpyDeviceToHost);

        for (const VarDesc& v : vars) {
            auto fire_var = [&] { run_tiled_variant(x, tw, out, m, 0, v.id); };
            fire_var(); // shape self-check: exact launcher throws on mismatch
            const double vms = time_ms(fire_var, 3, 20);
            fire_var();
            cudaMemcpy(hvar.data(), out, nel * 2, cudaMemcpyDeviceToHost);
            double num = 0.0, den = 0.0;
            for (std::size_t i = 0; i < nel; ++i) {
                const double r =
                    ninfer::ops::detail::amd::bits_to_f32(static_cast<std::uint32_t>(href[i]) << 16);
                const double t =
                    ninfer::ops::detail::amd::bits_to_f32(static_cast<std::uint32_t>(hvar[i]) << 16);
                num += (t - r) * (t - r);
                den += r * r;
            }
            const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
            std::printf("  %s | %7.3fms %6.1f GF/s %5.1f%%nom | relL2 %.2e %s\n", v.tag, vms,
                        gflops / vms * 1e3, 100.0 * gflops * 1e3 / vms / (kNominalF32Peak / 1e9),
                        rel, rel < 1e-2 ? "PASS" : "FAIL");
        }
        cudaFree(x); cudaFree(codes); cudaFree(scales); cudaFree(out); // nodiscard-ignore class
    }
    std::printf("READ: winner = max GF/s among PASS rows at M=128; promote that shape into\n"
                "launch_nvfp4_tiled_gemm default (plan + predictions: TILED_SWEEP_notes.md).\n");
    return 0;
}

// TILED-PK mode (--pk argv flag, 2026-09-17): V0 vs V0-PK at M=128 on the FIVE W4 SHARD
// geometries — the production TP4 problems (nvfp4_config.h AttnInputW4..Residual17408W4);
// the prior sweeps ran the four FULL geometries, so the shards AND any V-PK arm have NEVER
// been measured. Spec + pre-registered floors: docs/amd/PREFILL_GEMM_SCALE_2026-09-17.md §5
// (commit 9a38c0c50). Discipline = this file's law: production -O3 line, 5 s mclk-ramp
// hammer + best-effort sysfs pp_dpm_mclk print, hipEvent timing (3 warmup + 20 iters).
//   columns: GF/s, %fp32-nominal, rel-L2(PK, V0) — gate 1e-2 every row, the DECLARED
//   tolerance gate for the association change (nvfp4_amd_codec.h :131-138). NO
//   FNV/bit-equality column may gate this arm (design §2 gate (c)); xV0 ratio vs the
//   pre-registered floors; max|window| census (envelope guard (b)) MEASURED from the actual
//   staged buffers — reported, never a refusal (VRAM-law spirit).
int run_pk(const cudaDeviceProp& prop) {
    struct Prob { const char* name; int rows, k, stiles; };
    const Prob probs[] = {
        {"AttnInputW4 ", 3584, 5120, 80},
        {"GdnInputW4  ", 4096, 5120, 80},
        {"MlpGateUpW4 ", 8704, 5120, 80},
        {"Resid6144W4 ", 5120, 1536, 24},
        {"Resid17408W4", 5120, 4352, 68},
    };
    struct WDesc { int w; const char* tag; };
    const WDesc wins[] = {{16, "PK-W16"}, {32, "PK-W32"}, {64, "PK-W64"}};
    constexpr int m = 128; // the production chunk (design §5 primary)
    std::printf("PK BENCH — V0 vs V0-PK (HFMA2 tiled arm) at M=%d on the FIVE W4 shard "
                "geometries, W in {16,32,64}\n", m);
    std::printf("PRE-REGISTERED FLOOR (design §5): PK(W=32) >= 1.8x V0 GF/s, else the "
                "flush/cvt/int bill dominates the model — dump the ISA before ANY further "
                "tuning; ALL PK arms < 1.3x => the arm is DEAD.\n");
    std::printf("GATE: rel-L2(PK, V0) < 1e-2 every row (association change DECLARED "
                "tolerance-gated per nvfp4_amd_codec.h :131-138; NO FNV column).\n\n");

    // ---- ceilings + 5 s mclk-ramp hammer (identical cells to the default/sweep modes) ----
    const std::size_t bytes = static_cast<std::size_t>(256) * 1024 * 1024;
    const std::size_t n4 = bytes / 4;
    std::uint32_t *src, *dst, *sink;
    cudaMalloc(&src, bytes);
    cudaMalloc(&dst, bytes);
    cudaMalloc(&sink, 4);
    cudaMemset(src, 0x5A, bytes);
    const int grid = prop.multiProcessorCount * 8;
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
        cudaEventDestroy(w0); cudaEventDestroy(w1);
    }
    const double copy_gbs = 2.0 * bytes / 1e6 / time_ms([&] { copy_kernel<<<grid, 256>>>(src, dst, n4); }, 3, 30);
    const double read_gbs = bytes / 1e6 / time_ms([&] { read_kernel<<<grid, 256>>>(src, sink, n4); }, 3, 30);
    std::printf("CEILING copy %8.1f GB/s | read %8.1f GB/s\n", copy_gbs, read_gbs);
    print_mclk_sysfs();
    std::printf("\n");
    cudaFree(src); cudaFree(dst); cudaFree(sink);

    namespace amd = ninfer::ops::detail::amd;
    int rel_fails = 0;
    for (const Prob& p : probs) {
        const std::size_t code_bytes  = static_cast<std::size_t>(p.rows) * (p.k / 2);
        const std::size_t scale_bytes = static_cast<std::size_t>(p.rows) * (p.k / 16);
        void *x, *codes, *scales, *out;
        cudaMalloc(&x, static_cast<std::size_t>(p.k) * 256 * 2);
        cudaMalloc(&codes, code_bytes);
        cudaMalloc(&scales, scale_bytes);
        cudaMalloc(&out, static_cast<std::size_t>(p.rows) * 256 * 2);
        cudaMemset(codes, 0x11, code_bytes);   // valid e2m1 pairs, nonzero
        cudaMemset(scales, 0x38, scale_bytes); // ~1.0-ish e4m3
        cudaMemset(x, 0x3C, static_cast<std::size_t>(p.k) * 256 * 2); // small nonzero bf16
        ninfer::Weight tw;
        tw.qdata = codes;
        tw.scales = scales;
        tw.n = p.rows;
        tw.k = p.k;
        tw.weight_scale_divisor = 1.0f;
        const std::size_t nel   = static_cast<std::size_t>(p.rows) * m;
        const double gflops     = 2.0 * p.rows * p.k * m / 1e9; // per call
        std::vector<std::uint16_t> href(nel);
        std::vector<std::uint16_t> hvar(nel);

        // ---- envelope census from the ACTUAL staged buffers (design §2 guards a+b) ------
        // The kernel consumes the first m token rows of x and the whole scale plane; both
        // are copied back and measured (max|x| over bf16 bits, max|coeff| over e4m3 bytes x
        // divisor). The printed window bound is computed from THESE numbers — measured, not
        // assumed. Reported only: no code path may refuse on it.
        {
            std::vector<std::uint16_t> hx(static_cast<std::size_t>(p.k) * m);
            cudaMemcpy(hx.data(), x, hx.size() * 2, cudaMemcpyDeviceToHost);
            std::vector<std::uint8_t> hsc(scale_bytes);
            cudaMemcpy(hsc.data(), scales, scale_bytes, cudaMemcpyDeviceToHost);
            double max_x = 0.0, min_x = 1e30;
            for (const std::uint16_t b : hx) {
                const double v = std::fabs(static_cast<double>(
                    amd::bits_to_f32(static_cast<std::uint32_t>(b) << 16)));
                if (v > max_x) { max_x = v; }
                if (v > 0.0 && v < min_x) { min_x = v; }
            }
            double max_coeff = 0.0;
            for (const std::uint8_t b : hsc) {
                const double v = std::fabs(static_cast<double>(
                    amd::bits_to_f32(amd::e4m3_lut_decode(b)) * tw.weight_scale_divisor));
                if (v > max_coeff) { max_coeff = v; }
            }
            const bool env_ok = min_x >= 6.103515625e-05 && max_x <= 65504.0; // fp16-normal law
            std::printf("ENVELOPE %-13s max|x|=%.6g min|x|>0=%.6g max|coeff|=%.6g | "
                        "fp16-normal %s\n",
                        p.name, max_x, min_x, max_coeff, env_ok ? "PASS" : "FAIL");
            for (const WDesc& wd : wins) {
                // guard (b): W=16 keeps coeff OUT of the window (fp32 at flush); W=32/64 fold
                // coeff2 into the pair, so the window bound carries max|coeff|.
                const double bound = static_cast<double>(wd.w) * 6.0 * max_x *
                                     (wd.w == 16 ? 1.0 : max_coeff);
                std::printf("WINDOWCENSUS %-13s W=%2d bound=%.1f vs 65504 %s\n", p.name, wd.w,
                            bound, bound < 65504.0 ? "HEADROOM" : "OVERFLOW-RISK");
            }
        }

        // ---- V0 anchor: timing + semantic reference for every PK row --------------------
        auto fire_v0 = [&] { run_tiled(x, tw, out, m, 0); };
        fire_v0(); // shape self-check: exact launcher throws on mismatch
        const double v0ms = time_ms(fire_v0, 3, 20);
        fire_v0();
        cudaMemcpy(href.data(), out, nel * 2, cudaMemcpyDeviceToHost);
        std::printf("%-13s n=%5d k=%5d M=%3d | V0 %7.3fms %6.1f GF/s %5.1f%%nom\n", p.name,
                    p.rows, p.k, m, v0ms, gflops / v0ms * 1e3,
                    100.0 * gflops * 1e3 / v0ms / (kNominalF32Peak / 1e9));

        for (const WDesc& wd : wins) {
            auto fire_pk = [&] { run_tiled_pk(x, tw, out, m, 0, wd.w); };
            fire_pk(); // shape self-check: exact launcher throws on mismatch
            const double pkms = time_ms(fire_pk, 3, 20);
            fire_pk();
            cudaMemcpy(hvar.data(), out, nel * 2, cudaMemcpyDeviceToHost);
            double num = 0.0, den = 0.0;
            for (std::size_t i = 0; i < nel; ++i) {
                const double r =
                    amd::bits_to_f32(static_cast<std::uint32_t>(href[i]) << 16);
                const double t =
                    amd::bits_to_f32(static_cast<std::uint32_t>(hvar[i]) << 16);
                num += (t - r) * (t - r);
                den += r * r;
            }
            const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
            const double xV0 = v0ms / pkms; // same FLOPs: time ratio = GF/s ratio
            if (!(rel < 1e-2)) { ++rel_fails; }
            const char* rel_verdict = rel < 1e-2 ? "PASS" : "FAIL";
            const char* floor_verdict = xV0 >= 1.8 ? "FLOOR-PASS"
                                        : xV0 >= 1.3 ? "FLOOR-MISS" : "DEAD";
            std::printf("%-13s %4d | %-6s %7.3fms %6.1f GF/s %5.1f%%nom | xV0 %.2fx %s | "
                        "relL2(vsV0) %.2e %s\n",
                        p.name, m, wd.tag, pkms, gflops / pkms * 1e3,
                        100.0 * gflops * 1e3 / pkms / (kNominalF32Peak / 1e9), xV0,
                        floor_verdict, rel, rel_verdict);
        }
        cudaFree(x); cudaFree(codes); cudaFree(scales); cudaFree(out); // nodiscard-ignore class
    }
    std::printf("\nREAD: decisive row = PK-W32 xV0 at M=128. >= 1.8x => serving leg "
                "(NINFER_TILED_PK=1, PREFILL-SUM gemm < 798.8 ms/chunk). All arms < 1.3x => "
                "arm DEAD, ISA post-mortem, reopen nothing (design §5).\n");
    std::printf("READ: relL2 FAIL on any row => the fp16 window/flush design is falsified at "
                "that geometry — do NOT promote, dump the census first.\n");
    return rel_fails == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (cudaFree(nullptr) != cudaSuccess) { std::printf("SKIP: no device\n"); return 77; }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("device %s SMs=%d\n", prop.name, prop.multiProcessorCount);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sweep") == 0) { return run_sweep(prop); }
        if (std::strcmp(argv[i], "--pk") == 0) { return run_pk(prop); }
    }

    // ---- 1. ceilings: 256 MiB cells + 5 s mclk-ramp hammer (roofline-bench discipline) ----
    const std::size_t bytes = static_cast<std::size_t>(256) * 1024 * 1024;
    const std::size_t n4 = bytes / 4;
    std::uint32_t *src, *dst, *sink;
    cudaMalloc(&src, bytes);
    cudaMalloc(&dst, bytes);
    cudaMalloc(&sink, 4);
    cudaMemset(src, 0x5A, bytes);
    const int grid = prop.multiProcessorCount * 8;
    { // CLOCK WARM-UP: mclk idles at level 0 (167 MHz); short bursts never ramp it.
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
        cudaEventDestroy(w0); cudaEventDestroy(w1);
    }
    const double copy_gbs = 2.0 * bytes / 1e6 / time_ms([&] { copy_kernel<<<grid, 256>>>(src, dst, n4); }, 3, 30);
    const double read_gbs = bytes / 1e6 / time_ms([&] { read_kernel<<<grid, 256>>>(src, sink, n4); }, 3, 30);
    std::printf("CEILING copy %8.1f GB/s (2x conv) | read %8.1f GB/s (pure read)\n", copy_gbs, read_gbs);

    // ---- 2+3. production prefill arm vs REFERENCE tiled dequant GEMM ---------------------
    struct Prob { const char* name; int rows, k, stiles; };
    const Prob probs[] = {
        {"AttnInput  ", 14336, 5120, 80},
        {"GdnInput   ", 16384, 5120, 80},
        {"MlpGateUp  ", 34816, 5120, 80},
        {"Residual6144", 5120, 6144, 96},
    };
    const int m_list[] = {32, 128, 256};
    std::printf("%-13s %4s | %17s | %17s | GF/s REF vs PROD (%%fp32nom)\n", "problem", "M",
                "PROD ms   wGB/s", "REF ms    wGB/s");
    for (const Prob& p : probs) {
        const std::size_t code_bytes  = static_cast<std::size_t>(p.rows) * (p.k / 2);
        const std::size_t scale_bytes = static_cast<std::size_t>(p.rows) * (p.k / 16);
        void *x, *codes, *scales, *out;
        cudaMalloc(&x, static_cast<std::size_t>(p.k) * 256 * 2);
        cudaMalloc(&codes, code_bytes);
        cudaMalloc(&scales, scale_bytes);
        cudaMalloc(&out, static_cast<std::size_t>(p.rows) * 256 * 2);
        cudaMemset(codes, 0x11, code_bytes);   // valid e2m1 pairs, nonzero
        cudaMemset(scales, 0x38, scale_bytes); // ~1.0-ish e4m3
        cudaMemset(x, 0x3C, static_cast<std::size_t>(p.k) * 256 * 2); // small nonzero bf16
        ninfer::Weight tw;
        tw.qdata = codes;
        tw.scales = scales;
        tw.n = p.rows;
        tw.k = p.k;
        tw.weight_scale_divisor = 1.0f;
        // TILED-arm self-check buffers (host): REF output vs TILED output, rel L2 per shape.
        std::vector<std::uint16_t> href(static_cast<std::size_t>(p.rows) * 256);
        std::vector<std::uint16_t> htil(static_cast<std::size_t>(p.rows) * 256);
        for (int m : m_list) {
            const double wbytes = static_cast<double>(code_bytes + scale_bytes) +
                                  static_cast<double>(p.k + p.rows) * m * 2;
            const double gflops = 2.0 * p.rows * p.k * m / 1e9; // per call
            run_prod(x, tw, out, m, 0); // shape self-check: exact launcher throws on mismatch
            const double pms = time_ms([&] { run_prod(x, tw, out, m, 0); }, 3, 20);
            auto fire = [&] {
                if (p.stiles == 96) {
                    if (m == 32) { run_ref<96, 32>(x, codes, scales, 1.0f, out, p.rows, p.k, 0); }
                    else if (m == 128) { run_ref<96, 128>(x, codes, scales, 1.0f, out, p.rows, p.k, 0); }
                    else { run_ref<96, 256>(x, codes, scales, 1.0f, out, p.rows, p.k, 0); }
                } else {
                    if (m == 32) { run_ref<80, 32>(x, codes, scales, 1.0f, out, p.rows, p.k, 0); }
                    else if (m == 128) { run_ref<80, 128>(x, codes, scales, 1.0f, out, p.rows, p.k, 0); }
                    else { run_ref<80, 256>(x, codes, scales, 1.0f, out, p.rows, p.k, 0); }
                }
            };
            fire();
            const double rms = time_ms(fire, 3, 20);
            std::printf("%-13s %4d | %7.3fms %7.1f | %7.3fms %7.1f | %6.1f vs %6.1f (%.0f%%/%.0f%%)\n",
                        p.name, m, pms, wbytes / 1e6 / pms, rms, wbytes / 1e6 / rms,
                        gflops / rms * 1e3, gflops / pms * 1e3,
                        100.0 * gflops * 1e3 / rms / (kNominalF32Peak / 1e9),
                        100.0 * gflops * 1e3 / pms / (kNominalF32Peak / 1e9));
            // ---- TILED arm (TILED-GEMM-1) + tolerance self-check vs REF (semantic anchor) ----
            auto fire_tiled = [&] { run_tiled(x, tw, out, m, 0); };
            fire_tiled(); // shape self-check: exact launcher throws on mismatch
            const double tms = time_ms(fire_tiled, 3, 20);
            const std::size_t nel = static_cast<std::size_t>(p.rows) * m;
            fire(); // REF -> host reference
            cudaMemcpy(href.data(), out, nel * 2, cudaMemcpyDeviceToHost);
            fire_tiled(); // TILED -> host candidate
            cudaMemcpy(htil.data(), out, nel * 2, cudaMemcpyDeviceToHost);
            double num = 0.0, den = 0.0;
            for (std::size_t i = 0; i < nel; ++i) {
                const double r =
                    ninfer::ops::detail::amd::bits_to_f32(static_cast<std::uint32_t>(href[i]) << 16);
                const double t =
                    ninfer::ops::detail::amd::bits_to_f32(static_cast<std::uint32_t>(htil[i]) << 16);
                num += (t - r) * (t - r);
                den += r * r;
            }
            const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
            { std::size_t z = 0; for (std::size_t i = 0; i < nel; ++i) { if (htil[i] == 0x0000) { ++z; } }
              std::printf("TILEDVAL htil[0..3]=%04x %04x %04x %04x href[0]=%04x zeros=%zu/%zu\n",
                          htil[0], htil[1], htil[2], htil[3], href[0], z, nel); }
            std::printf("%-13s %4d | TILED %7.3fms %7.1f | GF/s %6.1f (%.0f%%nom) | relL2(vsREF) "
                        "%.2e %s\n",
                        p.name, m, tms, wbytes / 1e6 / tms, gflops / tms * 1e3,
                        100.0 * gflops * 1e3 / tms / (kNominalF32Peak / 1e9), rel,
                        rel < 1e-2 ? "PASS" : "FAIL");
        }
        cudaFree(x); cudaFree(codes); cudaFree(scales); cudaFree(out); // nodiscard-ignore class
    }
    std::printf("READ: PROD = the arm serve runs at prefill today (T=32 chunks, weights re-read\n"
                "per token). REF = tiled LDS envelope (dequant once per 32-row block, reused M\n"
                "times). Past the M-wall GB/s falls with M while GF/s rises; the build decision\n"
                "is REF-vs-PROD GF/s at M=128 (doc 01 §2: M>8 band = 'a plain tiled LDS GEMM').\n");
    return 0;
}
