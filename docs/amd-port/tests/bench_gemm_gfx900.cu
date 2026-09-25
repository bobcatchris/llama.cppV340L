// W5 device cell: rocBLAS fp16 GEMM on the exact TP3 prefill shapes, gfx900, die 3 dev cell.
//
// Measures the shipped dense-prefill route (dequant-to-f16 + hipBLAS/rocBLAS) per GEMM
// shape and reports TF/s/die against the campaign ceilings:
//   fp32-class realistic top  10.75-12.5 TF/s/die (docs/ninfer-history GEMM_CONSTRAINED_TILES
//   2026-09-18: 32 flop/cyc/SIMD x 224 SIMD; PREFILL_1K_PLAN 2026-09-18)
//   packed-fp16 peak          ~21.5 TF/s/die (VALU fp16x2, no MFMA on gfx900)
// W5 gate: if rocBLAS < ~50% of achievable (about 5.4 TF/s/die at M=128), the custom
// packed-fp16 tile route opens (plan W5, OPTIMIZATION_PLAN_TP3_200K.md).
//
// SHAPE CENSUS (trunk, ubatch M=128, weights row-split by ne[1] across 3 dies; row
// rounding = 128 = get_mmq_y_host(gfx900), ggml/src/ggml-cuda/mmq.cuh:142; boundaries
// ggml/src/ggml-cuda/ggml-cuda.cu:1898-1914; canonical equal-thirds split):
//   shape        K     N      calls/pass  per-die N shares (die0/1/2)  agg GFLOP  share
//   ffn_gate     5120  17408  64          5760/5760/5888               1460.3     23.4%
//   ffn_up       5120  17408  64          5760/5760/5888               1460.3     23.4%
//   ffn_down     17408 5120   64          1664/1664/1792               1460.3     23.4%
//   gdn_qkv      5120  10240  48          3328/3456/3456                644.2     10.3%
//   ssm_out      6144  5120   48          1664/1664/1792                386.5      6.2%
//   gdn_gate     5120  6144   48          2048/2048/2048                386.5      6.2%
//   attn_q+gate  5120  12288  16          4096/4096/4096                257.7      4.1%
//   attn_out     6144  5120   16          1664/1664/1792                128.8      2.1%
//   attn_k       5120  1024   16          256/384/384                    21.5      0.3%
//   attn_v       5120  1024   16          256/384/384                    21.5      0.3%
//   ssm_a/beta   5120  48     48          0/0/48 (last die only)          3.0      0.05%
//   trunk total 6.23 TFLOP/ubatch = 48.7 GFLOP/token over 3 dies, about 2.04 TFLOP/die.
//   FFN trio alone is 70.3% of GEMM FLOPs. Implied served rate: at pp 93.87 t/s the
//   67.4% GEMM share works out to ~2.0-2.2 TF/s/die vs the ~10.75 fp32-class top.
//   Per-call weight dequant q->f16 writes 2*N_d*K bytes (about 16 GB/die/ubatch of f16
//   traffic across all shapes) - the route tax the custom tile would remove.
// NOTE: actual runtime shares wobble by 128-row blocks (tensor_split comes from free
// memory fractions at load, not exact thirds); re-stamp with the boot banner if the
// split banner deviates.
//
// Build (validated: ggml/src/ggml-hip/CMakeLists.txt:46-48,157 uses hip, hipblas,
// rocblas from /opt/rocm; headers live in include/rocblas/, include/hipblas/):
//   /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 \
//     -I/opt/rocm-6.2.0/include/rocblas \
//     docs/amd-port/tests/bench_gemm_gfx900.cu \
//     -o /tmp/bench_gemm -L/opt/rocm-6.2.0/lib -lrocblas -Wl,-rpath,/opt/rocm-6.2.0/lib
// Run (DEFERRED - do not run now): die 3 is held by the W2 decode-atom desk. First run
// happens after W2 integration is recorded in the campaign ledger (E-012 tracks this).
//   HIP_VISIBLE_DEVICES=3 /tmp/bench_gemm [m] [reps]
//
// Modes benched (both are live ggml branches on gfx900):
//   mode h16 (default): rocblas_hgemm, f16 in/out, f16 accumulate - the shipped branch
//     (ggml_cuda_op_mul_mat_cublas else-arm, ggml-cuda.cu:1732-1748, CUBLAS_COMPUTE_16F
//     + f16 out + f16->f32 convert; use_fp16 holds because GGML_PREC_DEFAULT is the
//     mul_mat default, ggml/include/ggml.h:437). The trailing f16->f32 convert kernel is
//     NOT counted here; in-server it adds one more pass over the output.
//   mode f32: rocblas_gemm_ex, f16 in, f32 accumulate/out - the GGML_CUDA_FORCE_CUBLAS_
//     COMPUTE_32F analog (ggml-cuda.cu:1717-1731).
// Layout mirrors ggml: weights (N_d x K) column-major lda=K, activations (K x M)
// ldb=K, gemm(transA=T, transB=N, m=N_d, n=M, k=K, ldc=N_d).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
#define RB_CHECK(x) do { rocblas_status s_ = (x); if (s_ != rocblas_status_success) { \
    printf("rocBLAS error %d at %d\n", (int) s_, __LINE__); exit(1); } } while (0)

// fp32-class and packed-fp16 ceilings, TF/s per die (see header)
static const float CEIL_FP32 = 12.5f;
static const float CEIL_PK16 = 21.5f;
// W5 gate: 50% of fp32-class achievable
static const float GATE_TFS  = 0.5f * 10.75f;

struct shape {
    const char * name;
    int64_t k;       // K = src0->ne[0]
    int64_t n;       // N = src0->ne[1] (row-split dim)
    int     calls;   // GEMM calls per trunk pass (per layer count)
    int64_t nd[3];   // per-die row shares, canonical equal thirds
};

static const shape SHAPES[] = {
    { "ffn_gate   ",  5120, 17408, 64, { 5760, 5760, 5888 } },
    { "ffn_up     ",  5120, 17408, 64, { 5760, 5760, 5888 } },
    { "ffn_down   ", 17408,  5120, 64, { 1664, 1664, 1792 } },
    { "gdn_qkv    ",  5120, 10240, 48, { 3328, 3456, 3456 } },
    { "ssm_out    ",  6144,  5120, 48, { 1664, 1664, 1792 } },
    { "gdn_gate   ",  5120,  6144, 48, { 2048, 2048, 2048 } },
    { "attn_q+gate",  5120, 12288, 16, { 4096, 4096, 4096 } },
    { "attn_out   ",  6144,  5120, 16, { 1664, 1664, 1792 } },
    { "attn_k     ",  5120,  1024, 16, {  256,  384,  384 } },
    { "attn_v     ",  5120,  1024, 16, {  256,  384,  384 } },
};

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

static double tflops(int64_t m, int64_t n, int64_t k, double ms) {
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12;
}

int main(int argc, char ** argv) {
    const int64_t m    = argc > 1 ? atoll(argv[1]) : 128;   // ubatch tokens
    const int     reps = argc > 2 ? atoi(argv[2])  : 3;     // timing reps (median)
    const int     iters = 20;

    CHECK(hipInit(0));
    hipDeviceProp_t prop;
    int dev = 0;
    CHECK(hipGetDevice(&dev));
    CHECK(hipGetDeviceProperties(&prop, dev));
    printf("device %d: %s CUs %d clk %d MHz\n", dev, prop.name,
            prop.multiProcessorCount, prop.clockRate/1000);

    rocblas_handle handle;
    RB_CHECK(rocblas_create_handle(&handle));

    // scratch pools sized for the largest config
    size_t a_max = 0, b_max = 0, c_max = 0;
    for (const shape & s : SHAPES) {
        for (int d = 0; d < 3; d++) {
            a_max = std::max(a_max, (size_t) s.nd[d]*s.k);
            b_max = std::max(b_max, (size_t) s.k*m);
            c_max = std::max(c_max, (size_t) s.nd[d]*m);
        }
    }
    rocblas_half * a  = nullptr;   // weights, N_d x K col-major
    rocblas_half * b  = nullptr;   // activations, K x M col-major
    rocblas_half * ch = nullptr;   // f16 out (mode h16)
    float * cf        = nullptr;   // f32 out (mode f32)
    CHECK(hipMalloc((void **) &a,  a_max*2));
    CHECK(hipMalloc((void **) &b,  b_max*2));
    CHECK(hipMalloc((void **) &ch, c_max*2));
    CHECK(hipMalloc((void **) &cf, c_max*4));

    // deterministic fill in [-0.5, 0.5); checksums are sanity, not a correctness oracle
    uint32_t rng = 0x12345678u;
    auto frnd = [&]() { rng = rng*1664525u + 1013904223u; return ((rng >> 8) & 0xFFFF)
        / 65535.0f - 0.5f; };
    std::vector<rocblas_half> ha(a_max), hb(b_max);
    // rocblas_half is uint16_t bits host-side; _Float16 conversion is host-supported
    auto mkh = [&](float f) { rocblas_half h; _Float16 v = (_Float16) f;
        memcpy(&h.data, &v, 2); return h; };
    for (size_t i = 0; i < a_max; i++) ha[i] = mkh(frnd());
    for (size_t i = 0; i < b_max; i++) hb[i] = mkh(frnd());
    CHECK(hipMemcpy(a, ha.data(), a_max*2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(b, hb.data(), b_max*2, hipMemcpyHostToDevice));

    hipEvent_t ev0, ev1;
    CHECK(hipEventCreate(&ev0));
    CHECK(hipEventCreate(&ev1));

    const rocblas_half one_h = mkh(1.0f), zero_h = mkh(0.0f);
    const float one_f = 1.0f, zero_f = 0.0f;

    printf("\nm=%lld reps=%d x %d iters, per-die GEMM (mode h16 = shipped, f32 = acc/out f32)\n\n",
            (long long) m, reps, iters);
    printf("%-13s %6s %6s %11s %9s %7s %7s\n", "shape", "K", "N_d", "mode", "ms", "TF/s", "gate%");
    printf("%-13s %6s %6s %11s %9s %7s %7s\n", "-", "-", "-", "-", "-", "(vs 12.5/21.5)", "(5.4 TF/s)");

    struct agg { double fl = 0.0, ms = 0.0; };
    agg sum[2] = {};

    for (const shape & s : SHAPES) {
        for (int d = 0; d < 3; d++) {
            const int64_t nd = s.nd[d];
            if (nd == 0) continue; // ssm_a/beta class: empty slice on dies 0/1
            const double fl = 2.0 * nd * m * s.k;
            for (int md = 0; md < 2; md++) {
                // warmup (first call selects/Tensile-caches the kernel)
                for (int w = 0; w < 5; w++) {
                    if (md == 0) {
                        RB_CHECK(rocblas_hgemm(handle, rocblas_operation_transpose,
                            rocblas_operation_none, (int) nd, (int) m, (int) s.k,
                            &one_h, a, (int) s.k, b, (int) s.k, &zero_h, ch, (int) nd));
                    } else {
                        RB_CHECK(rocblas_gemm_ex(handle, rocblas_operation_transpose,
                            rocblas_operation_none, (int) nd, (int) m, (int) s.k,
                            &one_f, a, rocblas_datatype_f16_r, (int) s.k,
                                    b, rocblas_datatype_f16_r, (int) s.k,
                            &zero_f, cf, rocblas_datatype_f32_r, (int) nd,
                                     cf, rocblas_datatype_f32_r, (int) nd,
                            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
                    }
                }
                std::vector<float> tms;
                for (int r = 0; r < reps; r++) {
                    CHECK(hipEventRecord(ev0, nullptr));
                    for (int it = 0; it < iters; it++) {
                        if (md == 0) {
                            RB_CHECK(rocblas_hgemm(handle, rocblas_operation_transpose,
                                rocblas_operation_none, (int) nd, (int) m, (int) s.k,
                                &one_h, a, (int) s.k, b, (int) s.k, &zero_h, ch, (int) nd));
                        } else {
                            RB_CHECK(rocblas_gemm_ex(handle, rocblas_operation_transpose,
                                rocblas_operation_none, (int) nd, (int) m, (int) s.k,
                                &one_f, a, rocblas_datatype_f16_r, (int) s.k,
                                        b, rocblas_datatype_f16_r, (int) s.k,
                                &zero_f, cf, rocblas_datatype_f32_r, (int) nd,
                                         cf, rocblas_datatype_f32_r, (int) nd,
                                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
                        }
                    }
                    CHECK(hipEventRecord(ev1, nullptr));
                    CHECK(hipEventSynchronize(ev1));
                    float ms = 0;
                    CHECK(hipEventElapsedTime(&ms, ev0, ev1));
                    tms.push_back(ms/iters);
                }
                const float ms = median(tms);
                const double tfs = tflops(m, nd, s.k, ms);
                if (d == 0) { sum[md].fl += fl * s.calls; sum[md].ms += ms * s.calls; }
                printf("%-13s %6lld %6lld %11s %9.4f %7.2f %7.1f  die%d\n",
                        s.name, (long long) s.k, (long long) nd, md == 0 ? "h16" : "f32",
                        ms, tfs, 100.0f * tfs / GATE_TFS, d);
            }
        }
    }

    // checksum sanity (finite, non-degenerate)
    std::vector<float> probe(64);
    CHECK(hipMemcpy(probe.data(), cf, 64*4, hipMemcpyDeviceToHost));
    double csum = 0; for (int i = 0; i < 64; i++) csum += probe[i];
    printf("f32-out checksum(64) %.3e %s\n", csum, csum > 0 && std::isfinite(csum) ? "ok" : "BAD");

    printf("\nper-die aggregate (die0 shares, call-weighted):\n");
    for (int md = 0; md < 2; md++) {
        const double tfs = sum[md].fl / (sum[md].ms * 1e-3) / 1e12;
        printf("  mode %s: %.1f GF/ubatch, %.2f ms/ubatch -> %.2f TF/s/die"
               " = %.0f%% of fp32-class 12.5, %.0f%% of packed 21.5, W5 gate %s\n",
               md == 0 ? "h16" : "f32", sum[md].fl/1e9, sum[md].ms, tfs,
               100.0*tfs/CEIL_FP32, 100.0*tfs/CEIL_PK16,
               tfs >= GATE_TFS ? "MET (custom tiles stay closed)" : "MISSED (custom-tile route opens)");
    }

    RB_CHECK(rocblas_destroy_handle(handle));
    CHECK(hipFree(a));
    CHECK(hipFree(b));
    CHECK(hipFree(cf));
    CHECK(hipFree(ch));
    return 0;
}
