// Device oracle for the INTEGRATED tile kernel (tile-gemm.cu), die 3.
// Port-gate: the kernel as merged into ggml must reproduce the tile spec of
// docs/amd-port/tests/bench_tile_gfx900.cu (E-016/E-022) exactly:
//   1. host mirror of the per-output accumulation (chunk schedule, per-slice
//      half2 state via f64 fused-half emulation, fixed-order f32 slice sum):
//      device mismatch < 0.1%, max rel < 1e-3
//   2. fp64 reference over the same f16 inputs: L2 rel err reported
//      (banked band for a8: 2.3-5.4e-3)
// Run: HIP_VISIBLE_DEVICES=3 /tmp/test_tile_integ_oracle
//
// Build (flags mirrored from build-hip compile_commands, see T4 receipt; links the
// integrated build's libggml since the test TU includes tile-gemm.cu with its glue):
//   /opt/rocm-6.2.0/bin/hipcc -O2 -std=gnu++17 -DGGML_USE_HIP -DGGML_BACKEND_BUILD \
//     -DGGML_SHARED -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -D__HIP_PLATFORM_AMD__=1 \
//     -D__HIP_ROCclr__=1 --offload-arch=gfx900 -O3 \
//     -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//     docs/amd-port/tests/test_tile_integ_oracle.cu -o /tmp/test_tile_integ_oracle \
//     -L build-tile-integ/bin -lggml-hip -lggml-base -lamdhip64 \
//     -Wl,-rpath,<abs>/build-tile-integ/bin -Wl,-rpath,/opt/rocm-6.2.0/lib

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <thread>

#include "ggml-cuda/tile-gemm.cu"

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

typedef uint16_t f16bits;

static inline _Float16 bits2h(uint16_t b) { _Float16 v; memcpy(&v, &b, 2); return v; }
static inline uint16_t h2bits(_Float16 v) { uint16_t b; memcpy(&b, &v, 2); return b; }

// fused-half FMA emulation matching v_pk_fma_f16 (single rounding, exact in f64)
static inline _Float16 hfma_emu(_Float16 a, _Float16 b, _Float16 c) {
    return _Float16((double) a * (double) b + (double) c);
}

struct shape {
    const char * name;
    int64_t k;
    int64_t nd;
};

// census shapes, die0 per-die shares (die1/2 shares differ by one 128-row block
// at most; the kernel math is share-independent, gate on the die0 set)
static const shape SHAPES[] = {
    { "ffn_gate   ",  5120, 5760 },
    { "ffn_up     ",  5120, 5760 },
    { "ffn_down   ", 17408, 1664 },
    { "gdn_qkv    ",  5120, 3328 },
    { "ssm_out    ",  6144, 1664 },
    { "gdn_gate   ",  5120, 2048 },
    { "attn_q+gate",  5120, 4096 },
    { "attn_out   ",  6144, 1664 },
    { "attn_k     ",  5120,  256 },
    { "attn_v     ",  5120,  256 },
};
static const int NSHAPES = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));

static void host_mirror(const std::vector<f16bits> & Wh, const std::vector<f16bits> & Xh,
                        std::vector<float> & C, int M, int N, int K, int ks) {
    const int kc = TILE_FP16_KC;
    const int nkp = kc / 2;
    C.assign((size_t) M * N, 0.0f);
    auto rows = [&](int m0, int m1) {
        for (int m = m0; m < m1; m++) {
            const f16bits * x = &Xh[(size_t) m * K];
            float * cr = &C[(size_t) m * N];
            for (int n = 0; n < N; n++) {
                const f16bits * w = &Wh[(size_t) n * K];
                float o = 0.0f;
                for (int z = 0; z < ks; z++) {
                    const int kb0 = z * (K / ks);
                    _Float16 cl = bits2h(0), ch = bits2h(0);
                    const int nch = (K / ks) / kc;
                    for (int c = 0; c < nch; c++) {
                        const int kb = kb0 + c * kc;
                        for (int kp = 0; kp < nkp; kp++) {
                            const int k = kb + kp * 2;
                            cl = hfma_emu(bits2h(x[k]),     bits2h(w[k]),     cl);
                            ch = hfma_emu(bits2h(x[k + 1]), bits2h(w[k + 1]), ch);
                        }
                    }
                    o += float(cl) + float(ch);
                }
                cr[n] = o;
            }
        }
    };
    int nth = std::min(8u, std::thread::hardware_concurrency());
    if (nth < 1) nth = 1;
    std::vector<std::thread> th;
    for (int i = 0; i < nth; i++) th.emplace_back(rows, M * i / nth, M * (i + 1) / nth);
    for (auto & t : th) t.join();
}

static void host_ref64(const std::vector<f16bits> & Wh, const std::vector<f16bits> & Xh,
                       std::vector<double> & C, int M, int N, int K) {
    C.assign((size_t) M * N, 0.0);
    auto rows = [&](int m0, int m1) {
        for (int m = m0; m < m1; m++) {
            double * cr = &C[(size_t) m * N];
            for (int n = 0; n < N; n++) {
                const f16bits * w = &Wh[(size_t) n * K];
                const f16bits * x = &Xh[(size_t) m * K];
                double s0 = 0.0, s1 = 0.0;
                for (int k = 0; k + 1 < K; k += 2) {
                    s0 += (double) bits2h(x[k])     * (double) bits2h(w[k]);
                    s1 += (double) bits2h(x[k + 1]) * (double) bits2h(w[k + 1]);
                }
                cr[n] = s0 + s1;
            }
        }
    };
    int nth = std::min(8u, std::thread::hardware_concurrency());
    if (nth < 1) nth = 1;
    std::vector<std::thread> th;
    for (int i = 0; i < nth; i++) th.emplace_back(rows, M * i / nth, M * (i + 1) / nth);
    for (auto & t : th) t.join();
}

static double l2_rel(const std::vector<float> & a, const std::vector<double> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = a[i] - b[i];
        num += d * d; den += b[i] * b[i];
    }
    return sqrt(num / den);
}

int main() {
    CHECK(hipInit(0));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    printf("device 0: %s CUs %d | integrated-kernel oracle (tile-gemm.cu)\n",
           prop.name, prop.multiProcessorCount);

    constexpr int mt = TILE_FP16_MT, nt = TILE_FP16_NT, ty = TILE_FP16_TY, tx = TILE_FP16_TX;
    constexpr int kc = TILE_FP16_KC;
    constexpr int M = 128;
    const dim3 block(mt * nt / (ty * tx));

    int failures = 0;
    for (const shape & s : SHAPES) {
        const int64_t N = s.nd;
        const int64_t K = s.k;
        const int nw = (int)(N * K);
        const int ks = tile_fp16_pick_ks(N, K);

        // deterministic fill, same LCG as bench_tile_gfx900.cu
        uint32_t rng = 0x12345678u;
        auto frnd = [&]() { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) & 0xFFFF) / 65535.0f - 0.5f; };
        std::vector<f16bits> hW(nw), hX((size_t) M * K);
        for (int i = 0; i < nw; i++) hW[i] = h2bits((_Float16) frnd());
        for (size_t i = 0; i < hX.size(); i++) hX[i] = h2bits((_Float16) frnd());

        std::vector<double> ref;
        host_ref64(hW, hX, ref, M, (int) N, (int) K);

        std::vector<float> mir;
        host_mirror(hW, hX, mir, M, (int) N, (int) K, ks);

        __half * Wd; __half * Xd; float * Cd; float * Pd;
        CHECK(hipMalloc((void **) &Wd, nw * 2));
        CHECK(hipMalloc((void **) &Xd, hX.size() * 2));
        CHECK(hipMalloc((void **) &Cd, (size_t) M * N * 4));
        CHECK(hipMalloc((void **) &Pd, (size_t) ks * M * N * 4));
        CHECK(hipMemcpy(Wd, hW.data(), nw * 2, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(Xd, hX.data(), hX.size() * 2, hipMemcpyHostToDevice));

        const dim3 grid((unsigned)(N / nt), (unsigned)(M / mt), (unsigned) ks);
        // gate at the launch-site geometry: ldc == N when not behind the split path
        tile_fp16_gemm<mt, nt, ty, tx, kc, false><<<grid, block, 0, nullptr>>>(
            Wd, Xd, Cd, Pd, (int) N, (int) K, (int) (K / ks), N, (int) K);
        if (ks > 1) {
            const int mn = (int)(M * N);
            tile_fp16_reduce<<<(mn/4 + 255) / 256, 256, 0, nullptr>>>(Pd, Cd, mn, ks);
        }
        CHECK(hipDeviceSynchronize());

        std::vector<float> out((size_t) M * N);
        CHECK(hipMemcpy(out.data(), Cd, out.size() * 4, hipMemcpyDeviceToHost));

        // chunked arm: one launch per split-K slice, the slice gathered into a
        // contiguous N x ksl window first (mirrors the glue's quant-block gather;
        // hipMemcpy2D is the same operation on already-f16 data), ACC into C in
        // slice order; output must be bit-identical to the unchunked P + reduce
        float * Cd2;
        __half * Wc;
        const int ksl = (int) (K / ks);
        CHECK(hipMalloc((void **) &Cd2, (size_t) M * N * 4));
        CHECK(hipMalloc((void **) &Wc, (size_t) N * ksl * 2));
        const dim3 grid1((unsigned)(N / nt), (unsigned)(M / mt), 1);
        for (int z = 0; z < ks; z++) {
            CHECK(hipMemcpy2DAsync(Wc, ksl * 2, Wd + z*ksl, K * 2, ksl * 2, N,
                                   hipMemcpyDeviceToDevice, nullptr));
            if (z == 0) {
                tile_fp16_gemm<mt, nt, ty, tx, kc, false><<<grid1, block, 0, nullptr>>>(
                    Wc, Xd + z*ksl, Cd2, nullptr, (int) N, ksl, ksl, N, (int) K);
            } else {
                tile_fp16_gemm<mt, nt, ty, tx, kc, true><<<grid1, block, 0, nullptr>>>(
                    Wc, Xd + z*ksl, Cd2, nullptr, (int) N, ksl, ksl, N, (int) K);
            }
        }
        CHECK(hipDeviceSynchronize());
        std::vector<float> out2((size_t) M * N);
        CHECK(hipMemcpy(out2.data(), Cd2, out2.size() * 4, hipMemcpyDeviceToHost));
        size_t bit_diff = 0;
        for (size_t i = 0; i < out.size(); i++) {
            if (memcmp(&out[i], &out2[i], 4) != 0) bit_diff++;
        }
        CHECK(hipFree(Cd2));
        CHECK(hipFree(Wc));

        size_t bad = 0;
        double mx = 0;
        for (size_t i = 0; i < out.size(); i++) {
            const double df = fabs((double) out[i] - (double) mir[i]);
            const double rel = df / (fabs((double) mir[i]) + 1e-3);
            if (rel > 1e-3) { bad++; mx = std::max(mx, rel); }
        }
        const double l2 = l2_rel(out, ref);
        const double mismatch_pct = 100.0 * (double) bad / (double) out.size();
        const bool pass = mismatch_pct < 0.1 && mx < 1e-3 && bit_diff == 0;
        if (!pass) failures++;
        printf("  %-13s K=%5lld N_d=%4lld ks=%d: mismatch %6.3f%% max rel %.2e | L2 f64 %.3e | chunked bit-diff %zu/%zu -> %s\n",
               s.name, (long long) K, (long long) N, ks, mismatch_pct, mx, l2,
               bit_diff, out.size(), pass ? "PASS" : "FAIL");
        fflush(stdout);

        CHECK(hipFree(Wd)); CHECK(hipFree(Xd)); CHECK(hipFree(Cd)); CHECK(hipFree(Pd));
    }

    printf("\n%s (%d failure%s)\n", failures == 0 ? "ORACLE PASS" : "ORACLE FAIL",
           failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
