// W5 custom-tile cell (oracle-gated): packed-fp16 GEMM tile for M=128 prefill,
// gfx900 (Vega 10, 56 CU, wave64, no MFMA, v_pk_fma_f16 packed math), die 3 dev cell.
//
// Attacks the W5 gap named by E-015: shipped route (per-call dequant + rocBLAS h16)
// = 4.52 TF/s/die die0-aggregate / 4.62 all-die call-weighted. The tile keeps weights
// as resident f16 in HBM (load-time dequant assumed free at steady state - the thing
// deleted is the shipped route's per-call dequant of the whole weight slice) and does
// the GEMM with half2 packed FMAs, fp32 accumulation either pure-f16-class (matches
// shipped CUBLAS_COMPUTE_16F numerics class) or with periodic f32 flush (template).
//
// Geometry (ggml view, same as baseline harness): weights W (N_d x K, k contiguous),
// activations X (M x K, k contiguous), out C (M x N_d, n contiguous, f32).
//   C[m][n] = sum_k X[m][k] * W[n][k]
//
// Tile: workgroup computes an MT x NT output tile, one K-slice (split-K in z).
// Per thread: TY x TX half2 accumulators (packed along k pairs), inner loop over
// KC/2 k-pairs reading frags from LDS double buffers. LDS layouts:
//   as[m][kp]   (m-major, A staged as b128, frag reads are broadcast - lanes of a
//                wave phase share m_base)
//   ws[kp][n]   (kp-major, stride NT+4 to spread lanes' b128 frags across banks)
// Split-K: gridDim.z slices, partials in P, fixed-order f32 reduce kernel (the
// reduce is IN the timed region; ks=1 writes C directly, no reduce).
//
// ORACLE (every arm, every shape, die0 share, BEFORE its timing counts):
//   1. host mirror of the exact per-output accumulation order (chunk-local flush
//      schedule, per-slice f16 state, fixed-order f32 slice sum) with fused-half
//      emulation: device output must match mirror (mismatch < 0.1%, max rel < 1e-3).
//   2. fp64 reference over the same f16 inputs: L2 rel err of the arm must be
//      <= 2x the rocBLAS h16 (shipped) L2 rel err on the same data.
// Failing arms are refused and never timed.
//
// ARMS (all template instantiations of one kernel, runtime-selected):
//   rb16  : rocBLAS h16 control (= shipped route, E-015 binary class)
//   a1    : MT128 NT32  TY4 TX4 KC32 flush 0   (16 acc, VGPR-light)
//   a2    : MT128 NT32  TY4 TX4 KC32 flush 64  (a1 + f32 flush every 64 k)
//   a3    : MT128 NT64  TY8 TX4 KC32 flush 0   (48 acc, NT64)
//   a4    : MT128 NT64  TY8 TX4 KC16 flush 0   (a3 with smaller LDS chunk)
//   a5    : MT128 NT128 TY8 TX8 KC16 flush 0   (64 acc, max math density)
//   a6    : MT128 NT128 TY8 TX8 KC32 flush 0   (a5 with KC32, LDS 33 KB)
//   a7    : MT64  NT128 TY8 TX8 KC32 flush 0   (a6 split in M for tail balance)
//   a8    : MT128 NT64  TY4 TX8 KC32 flush 0   (32 acc, wide-N frag)
//   a9    : MT128 NT64  TY8 TX8 KC16 flush 0   (64 acc, NT64, THT 128)
// Split-K: auto per shape: ks doubled while grid*ks < 224 and (K/ks) % KC == 0,
// cap 8; ks=1 skips partials+reduce.
//
// Build:
//   /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 \
//     docs/amd-port/tests/bench_tile_gfx900.cu -o /tmp/bench_tile
// Run (die 3 only):
//   HIP_VISIBLE_DEVICES=3 /tmp/bench_tile [reps=3] [iters=100] [smoke_shapes]
//
// Protocol: 3 interleaved reps x 100 iters per arm per shape per die share (median of
// reps); rb16 control in every session; call-weighted aggregates reuse the E-015
// weights so numbers are directly comparable (die0 agg ~ 4.52, all-die agg ~ 4.62).
// W5 promotion gate: all-die call-weighted > 4.62 TF/s/die.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <thread>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
#define RB_CHECK(x) do { rocblas_status s_ = (x); if (s_ != rocblas_status_success) { \
    printf("rocBLAS error %d at %d\n", (int) s_, __LINE__); exit(1); } } while (0)

typedef uint16_t f16bits;

static inline _Float16 bits2h(uint16_t b) { _Float16 v; memcpy(&v, &b, 2); return v; }
static inline uint16_t h2bits(_Float16 v) { uint16_t b; memcpy(&b, &v, 2); return b; }

// fused-half FMA emulation matching v_pk_fma_f16 (single rounding): the product of
// two f16 and the add are exact in f64 (53-bit mantissa covers the 22-bit product
// plus worst-case alignment), so one double-precision expression reproduces the
// GPU's fused f16 FMA bit-exactly
static inline _Float16 hfma_emu(_Float16 a, _Float16 b, _Float16 c) {
    return _Float16((double) a * (double) b + (double) c);
}

// ------------------------------------------------------------------ shapes ---

struct shape {
    const char * name;
    int64_t k;
    int64_t n;       // full N; per-die share = nd[d]
    int     calls;
    int64_t nd[3];
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
static const int NSHAPES = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));

// ------------------------------------------------------------------ kernel ---

template <int MT, int NT, int TY, int TX, int KC, int FLUSH>
__global__ void __launch_bounds__(MT * NT / (TY * TX))
tile_gemm(const __half * __restrict__ Wp,   // N x K, k contiguous
          const __half * __restrict__ X,    // M x K, k contiguous
          float * __restrict__ C,           // M x N, n contiguous (used when gridDim.z==1)
          float * __restrict__ P,           // partials (gridDim.z>1)
          int N, int K) {
    constexpr int THT = MT * NT / (TY * TX);
    constexpr int AST = KC / 2;        // as[m][kp] stride in half2
    constexpr int WST = NT + 4;        // ws[kp][n] stride in half2 (bank pad)
    constexpr int AIT = (MT * (KC / 8) + THT - 1) / THT;
    constexpr int WIT = (NT * (KC / 8) + THT - 1) / THT;
    constexpr int NKP = KC / 2;

    // flat LDS, 32-bit indexing throughout: the max element offset here is
    // N_d*K = 5888*17408 < 2^31, so 64-bit index arithmetic (a v_mad_u64_u32
    // chain per address) is pure waste on gfx900
    __shared__ half2 asx[2 * MT * AST];
    __shared__ half2 wsx[2 * NKP * WST];

    const int tid = threadIdx.x;
    const int mt  = blockIdx.y;        // M tile index (M = gridDim.y * MT)
    const int mb  = mt * MT;
    const int n0  = blockIdx.x * NT;
    const int ks  = gridDim.z;
    const int z   = blockIdx.z;
    const int ksl = K / ks;            // slice length, multiple of KC
    const int kb0 = z * ksl;
    const int nch = ksl / KC;

    // output mapping: NC = NT/TX column groups, tid%NC = column group (interleaved
    // ownership: thread owns columns cg + t*NC), tid/NC = row group. Within a wave
    // phase the lanes share the row group (A frag reads are broadcasts) and their
    // column-group indices are consecutive (W frag reads hit distinct banks)
    constexpr int NC = NT / TX;
    const int cg  = tid % NC;
    const int mg  = tid / NC;
    const int m0t = mg * TY;

    half2 acc[TY][TX];
    float sm[TY][TX][2];
#pragma unroll
    for (int r = 0; r < TY; r++) {
#pragma unroll
        for (int t = 0; t < TX; t++) {
            acc[r][t] = __float2half2_rn(0.0f);
            if (FLUSH > 0) { sm[r][t][0] = 0.0f; sm[r][t][1] = 0.0f; }
        }
    }

    auto stage = [&](int buf, int c) {
        const int kb = kb0 + c * KC;
        const int abase = buf * MT * AST;
        const int wbase = buf * NKP * WST;
#pragma unroll
        for (int i = 0; i < AIT; i++) {
            const int slot = tid + i * THT;
            if (slot < MT * (KC / 8)) {
                const int m   = slot / (KC / 8);
                const int sub = slot % (KC / 8);
                const uint4 v = *reinterpret_cast<const uint4 *>(
                    X + (mb + m) * K + kb + sub * 8);
                *reinterpret_cast<uint4 *>(&asx[abase + m * AST + sub * 4]) = v;
            }
        }
#pragma unroll
        for (int i = 0; i < WIT; i++) {
            const int slot = tid + i * THT;
            if (slot < NT * (KC / 8)) {
                const int n   = slot / (KC / 8);
                const int sub = slot % (KC / 8);
                const uint4 v = *reinterpret_cast<const uint4 *>(
                    Wp + (n0 + n) * K + kb + sub * 8);
                half2 * d = &wsx[wbase + sub * 4 * WST + n];
                const half2 * sv = reinterpret_cast<const half2 *>(&v);
#pragma unroll
                for (int j = 0; j < 4; j++) d[j * WST] = sv[j];
            }
        }
    };

    stage(0, 0);
    __syncthreads();
    for (int c = 0; c < nch; c++) {
        if (c + 1 < nch) stage((c + 1) & 1, c + 1);
        __syncthreads();
        const int abuf = (c & 1) * MT * AST + m0t * AST;
        const int wbuf = (c & 1) * NKP * WST + cg;
#pragma unroll
        for (int kp = 0; kp < NKP; kp++) {
            half2 fr[TY], fw[TX];
#pragma unroll
            for (int r = 0; r < TY; r++) fr[r] = asx[abuf + r * AST + kp];
#pragma unroll
            for (int t = 0; t < TX; t++) fw[t] = wsx[wbuf + kp * WST + t * NC];
#pragma unroll
            for (int r = 0; r < TY; r++) {
#pragma unroll
                for (int t = 0; t < TX; t++) acc[r][t] = __hfma2(fr[r], fw[t], acc[r][t]);
            }
            if (FLUSH > 0) {
                if ((kp + 1) % FLUSH == 0) {
#pragma unroll
                    for (int r = 0; r < TY; r++) {
#pragma unroll
                        for (int t = 0; t < TX; t++) {
                            sm[r][t][0] += __low2float(acc[r][t]);
                            sm[r][t][1] += __high2float(acc[r][t]);
                            acc[r][t] = __float2half2_rn(0.0f);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    const bool direct = (ks == 1);
    const int crow = mb + m0t;
#pragma unroll
    for (int r = 0; r < TY; r++) {
#pragma unroll
        for (int t = 0; t < TX; t++) {
            float o = __low2float(acc[r][t]) + __high2float(acc[r][t]);
            if (FLUSH > 0) o += sm[r][t][0] + sm[r][t][1];
            const int col = n0 + cg + t * NC;
            if (direct) C[(size_t)(crow + r) * N + col] = o;
            else        P[((size_t) z * gridDim.y * MT + crow + r) * N + col] = o;
        }
    }
}

__global__ void reduce_pk(const float * __restrict__ P, float * __restrict__ C,
                          int mn, int ks) {
    const int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i4 + 3 < mn) {
        float4 s = *reinterpret_cast<const float4 *>(P + i4);
        for (int z = 1; z < ks; z++) {
            const float4 v = *reinterpret_cast<const float4 *>(P + (size_t) z * mn + i4);
            s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
        *reinterpret_cast<float4 *>(C + i4) = s;
    }
}

// ------------------------------------------------------------ dispatch table ---

template <int MT, int NT, int TY, int TX, int KC, int FLUSH>
static void launch_tile(const __half * Wp, const __half * X, float * C, float * P,
                        int N, int K, dim3 g, dim3 b, hipStream_t s) {
    hipLaunchKernelGGL((tile_gemm<MT, NT, TY, TX, KC, FLUSH>), g, b, 0, s, Wp, X, C, P, N, K);
}

struct arm_def {
    const char * name;
    int mt, nt, ty, tx, kc, flush;
    void (*launch)(const __half *, const __half *, float *, float *, int, int, dim3, dim3, hipStream_t);
};

static const arm_def ARMS[] = {
    { "a1", 128,  32, 4, 4, 32,  0, launch_tile<128,  32, 4, 4, 32,  0> },
    { "a2", 128,  32, 4, 4, 32, 64, launch_tile<128,  32, 4, 4, 32, 64> },
    { "a3", 128,  64, 8, 4, 32,  0, launch_tile<128,  64, 8, 4, 32,  0> },
    { "a4", 128,  64, 8, 4, 16,  0, launch_tile<128,  64, 8, 4, 16,  0> },
    { "a5", 128, 128, 8, 8, 16,  0, launch_tile<128, 128, 8, 8, 16,  0> },
    { "a6", 128, 128, 8, 8, 32,  0, launch_tile<128, 128, 8, 8, 32,  0> },
    { "a7",  64, 128, 8, 8, 32,  0, launch_tile< 64, 128, 8, 8, 32,  0> },
    { "a8", 128,  64, 4, 8, 32,  0, launch_tile<128,  64, 4, 8, 32,  0> },
    { "a9", 128,  64, 8, 8, 16,  0, launch_tile<128,  64, 8, 8, 16,  0> },
};
static const int NARMS = (int)(sizeof(ARMS) / sizeof(ARMS[0]));

// split-K choice: fill the 56-CU device (224 SIMD target) without breaking the
// K step law (slice must stay a multiple of KC)
static int pick_ks(int64_t N, int64_t K, int nt, int kc) {
    const int64_t grid = (N + nt - 1) / nt;
    int ks = 1;
    while (grid * ks < 224 && ks < 8) {
        const int cand = ks * 2;
        if (K % cand != 0) break;
        if ((K / cand) % kc != 0) break;
        ks = cand;
    }
    return ks;
}

// ------------------------------------------------------------- host oracle ---

// exact mirror of tile_gemm per-output math: chunk-local flush schedule, per-slice
// f16 state, fixed-order f32 slice sum. Threaded over M.
static void host_mirror(const std::vector<f16bits> & Wh, const std::vector<f16bits> & Xh,
                        std::vector<float> & C, int M, int N, int K,
                        int kc, int flush, int ks) {
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
                    float sl = 0.0f, sh = 0.0f;
                    const int nch = (K / ks) / kc;
                    for (int c = 0; c < nch; c++) {
                        const int kb = kb0 + c * kc;
                        for (int kp = 0; kp < nkp; kp++) {
                            const int k = kb + kp * 2;
                            cl = hfma_emu(bits2h(x[k]),     bits2h(w[k]),     cl);
                            ch = hfma_emu(bits2h(x[k + 1]), bits2h(w[k + 1]), ch);
                            if (flush > 0 && (kp + 1) % flush == 0) {
                                sl += float(cl); sh += float(ch);
                                cl = bits2h(0); ch = bits2h(0);
                            }
                        }
                    }
                    o += sl + sh + float(cl) + float(ch);
                }
                cr[n] = o;
            }
        }
    };
    int nth = std::min(8, (int) std::thread::hardware_concurrency());
    if (nth < 1) nth = 1;
    std::vector<std::thread> th;
    for (int i = 0; i < nth; i++) th.emplace_back(rows, M * i / nth, M * (i + 1) / nth);
    for (auto & t : th) t.join();
}

// fp64 reference over the f16 inputs
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
    int nth = std::min(8, (int) std::thread::hardware_concurrency());
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

struct mirror_gate {
    double mismatch_pct;   // device vs mirror
    double max_rel;        // device vs mirror max rel diff
    double l2_dev;         // device vs fp64
    double l2_rb;          // rocBLAS h16 vs fp64 (numerics class anchor)
    bool   pass;
};

// ------------------------------------------------------------------- bench ---

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char ** argv) {
    const int reps  = argc > 1 ? atoi(argv[1]) : 3;
    const int iters = argc > 2 ? atoi(argv[2]) : 100;
    const int smoke = argc > 3 ? atoi(argv[3]) : 0;   // run only first N shapes

    CHECK(hipInit(0));
    hipDeviceProp_t prop;
    int dev = 0;
    CHECK(hipGetDevice(&dev));
    CHECK(hipGetDeviceProperties(&prop, dev));
    printf("device %d: %s CUs %d clk %d MHz | reps %d x %d iters (interleaved)\n",
            dev, prop.name, prop.multiProcessorCount, prop.clockRate / 1000, reps, iters);

    rocblas_handle rb;
    RB_CHECK(rocblas_create_handle(&rb));

    const int64_t M = 128;
    size_t w_max = 0, x_max = 0;
    for (const shape & s : SHAPES) {
        for (int d = 0; d < 3; d++) w_max = std::max(w_max, (size_t)(s.nd[d] * s.k));
        x_max = std::max(x_max, (size_t)(s.k * M));
    }
    __half * Wd; __half * Xd; float * Cd; float * Pd; rocblas_half * Crb;
    CHECK(hipMalloc((void **) &Wd, w_max * 2));
    CHECK(hipMalloc((void **) &Xd, x_max * 2));
    CHECK(hipMalloc((void **) &Cd, (size_t) M * 5888 * 4));
    CHECK(hipMalloc((void **) &Pd, (size_t) 8 * M * 5888 * 4));
    CHECK(hipMalloc((void **) &Crb, (size_t) M * 5888 * 2));

    // deterministic fill, same LCG as the baseline harness
    uint32_t rng = 0x12345678u;
    auto frnd = [&]() { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) & 0xFFFF) / 65535.0f - 0.5f; };
    std::vector<f16bits> hW(w_max), hX(x_max);
    for (size_t i = 0; i < w_max; i++) hW[i] = h2bits((_Float16) frnd());
    for (size_t i = 0; i < x_max; i++) hX[i] = h2bits((_Float16) frnd());
    CHECK(hipMemcpy(Wd, hW.data(), w_max * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(Xd, hX.data(), x_max * 2, hipMemcpyHostToDevice));

    hipEvent_t ev0, ev1;
    CHECK(hipEventCreate(&ev0));
    CHECK(hipEventCreate(&ev1));

    // resource census
    printf("\nresource census (hipFuncGetAttributes):\n");
    for (const arm_def & a : ARMS) {
        hipFuncAttributes at;
        #define QATTR(MT, NT, TY, TX, KC, FL) \
            if (a.mt == MT && a.nt == NT && a.ty == TY && a.tx == TX && a.kc == KC && a.flush == FL) { \
                CHECK(hipFuncGetAttributes(&at, (const void *) tile_gemm<MT, NT, TY, TX, KC, FL>)); }
        QATTR(128,  32, 4, 4, 32,  0) else QATTR(128,  32, 4, 4, 32, 64)
        else QATTR(128,  64, 8, 4, 32,  0) else QATTR(128,  64, 8, 4, 16,  0)
        else QATTR(128, 128, 8, 8, 16,  0) else QATTR(128, 128, 8, 8, 32,  0)
        else QATTR( 64, 128, 8, 8, 32,  0) else QATTR(128,  64, 4, 8, 32,  0)
        else QATTR(128,  64, 8, 8, 16,  0)
        #undef QATTR
        const int tht = a.mt * a.nt / (a.ty * a.tx);
        const int lds = 2 * (a.kc / 2) * (a.mt + a.nt + 4) * 4;
        const int vper = ((at.numRegs + 7) / 8) * 8;
        const int cap_v = vper > 0 ? std::min(10, 65536 / (64 * vper)) : 0;
        const int cap_l = std::min(16, 65536 / lds);
        printf("  %s MT%d NT%d TY%d TX%d KC%d fl%d: THT %3d VGPR %3d lds %5d B "
               "(wave cap: VGPR %d/SIMD, LDS %d WG/CU, reg-limited threads %d)\n",
                a.name, a.mt, a.nt, a.ty, a.tx, a.kc, a.flush, tht, at.numRegs, lds,
                cap_v, cap_l, at.maxThreadsPerBlock);
    }

    const int nsh = smoke > 0 ? std::min(smoke, NSHAPES) : NSHAPES;

    // phase 1: oracle, every arm x shape, die0 share
    printf("\nphase 1: oracle (die0 share) + fp64/rocBLAS error anchors\n");
    std::vector<std::vector<mirror_gate>> gate(NSHAPES, std::vector<mirror_gate>(NARMS));
    std::vector<std::vector<int>> gks(NSHAPES, std::vector<int>(NARMS, 1));
    for (int si = 0; si < nsh; si++) {
        const shape & s = SHAPES[si];
        const int64_t nd = s.nd[0];
        const int nw = (int)(nd * s.k);
        std::vector<f16bits> wS(hW.begin(), hW.begin() + nw);

        std::vector<double> ref;
        host_ref64(wS, hX, ref, (int) M, (int) nd, (int) s.k);

        // rocBLAS h16 on the same data: control + numerics anchor
        const rocblas_half one = { h2bits((_Float16) 1.0f) }, zero = { h2bits((_Float16) 0.0f) };
        RB_CHECK(rocblas_hgemm(rb, rocblas_operation_transpose, rocblas_operation_none,
            (int) nd, (int) M, (int) s.k, &one, (const rocblas_half *) Wd, (int) s.k,
            (const rocblas_half *) Xd, (int) s.k, &zero, (rocblas_half *) Crb, (int) nd));
        std::vector<float> rbf((size_t) M * nd);
        {
            std::vector<f16bits> rbb(rbf.size());
            CHECK(hipMemcpy(rbb.data(), Crb, rbb.size() * 2, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < rbb.size(); i++) rbf[i] = (float) bits2h(rbb[i]);
        }
        const double l2_rb = l2_rel(rbf, ref);

        // mirrors are shared by arms with identical (kc, flush, ks) - the per-output
        // math does not depend on tile geometry
        struct mirent { int kc, fl, ks; std::vector<float> c; };
        std::vector<mirent> mirs;
        for (int ai = 0; ai < NARMS; ai++) {
            const arm_def & a = ARMS[ai];
            const int ks = pick_ks(nd, s.k, a.nt, a.kc);
            bool have = false;
            for (mirent & m : mirs) have |= (m.kc == a.kc && m.fl == a.flush && m.ks == ks);
            if (!have) mirs.push_back({ a.kc, a.flush, ks, {} });
        }
        for (mirent & m : mirs) {
            host_mirror(wS, hX, m.c, (int) M, (int) nd, (int) s.k, m.kc, m.fl, m.ks);
        }

        for (int ai = 0; ai < NARMS; ai++) {
            const arm_def & a = ARMS[ai];
            const int ks = pick_ks(nd, s.k, a.nt, a.kc);
            gks[si][ai] = ks;
            dim3 b(a.mt * a.nt / (a.ty * a.tx));
            dim3 g2((unsigned)((nd + a.nt - 1) / a.nt), (unsigned)(M / a.mt), (unsigned) ks);
            a.launch(Wd, Xd, Cd, Pd, (int) nd, (int) s.k, g2, b, nullptr);
            if (ks > 1) {
                const int mn = (int)(M * nd);
                hipLaunchKernelGGL(reduce_pk, dim3((mn / 4 + 255) / 256), dim3(256), 0, nullptr,
                                   (const float *) Pd, Cd, mn, ks);
            }
            std::vector<float> out((size_t) M * nd);
            CHECK(hipMemcpy(out.data(), Cd, out.size() * 4, hipMemcpyDeviceToHost));
            CHECK(hipDeviceSynchronize());

            const std::vector<float> * mirp = nullptr;
            for (mirent & m : mirs) if (m.kc == a.kc && m.fl == a.flush && m.ks == ks) mirp = &m.c;

            size_t bad = 0;
            double mx = 0;
            const bool dbg = getenv("W5_DEBUG") != nullptr;
            int ndbg = 0;
            for (size_t i = 0; i < out.size(); i++) {
                const double df = fabs((double) out[i] - (double) (*mirp)[i]);
                const double den = fabs((double) (*mirp)[i]) + 1e-3;
                const double rel = df / den;
                if (rel > 1e-3) {
                    if (dbg && ndbg < 12) {
                        printf("    DBG m=%zu n=%zu dev=%.6f mir=%.6f ref=%.6f rb=%.6f\n",
                                i / (size_t) nd, i % (size_t) nd, out[i], (*mirp)[i],
                                ref[i], rbf[i]);
                        ndbg++;
                    }
                    bad++; mx = std::max(mx, rel);
                }
            }
            mirror_gate gg = {};
            gg.mismatch_pct = 100.0 * (double) bad / (double) out.size();
            gg.max_rel = mx;
            gg.l2_dev = l2_rel(out, ref);
            gg.l2_rb = l2_rb;
            gg.pass = (gg.mismatch_pct < 0.1 && gg.max_rel < 1e-3 && gg.l2_dev <= 2.0 * l2_rb + 1e-9);
            gate[si][ai] = gg;
            printf("  %-13s %s MT%d NT%d TY%d TX%d KC%d fl%d ks%d: mismatch %6.3f%% max rel %.2e"
                   " | L2 f64 %.3e (rb16 %.3e) -> %s\n",
                    s.name, a.name, a.mt, a.nt, a.ty, a.tx, a.kc, a.flush, ks,
                    gg.mismatch_pct, gg.max_rel, gg.l2_dev, l2_rb, gg.pass ? "PASS" : "REFUSED");
            fflush(stdout);
        }
    }

    // phase 2: interleaved timing
    // rall[(r*NSHAPES + si)*3 + di)*(NARMS+1) + ai]: ms per rep; slot NARMS = rb16
    std::vector<float> rall((size_t) reps * NSHAPES * 3 * (NARMS + 1), -1.0f);
    auto ridx = [&](int r, int si, int di, int ai) {
        return ((size_t) r * NSHAPES + si) * 3 * (NARMS + 1) + di * (NARMS + 1) + ai;
    };
    auto get_ms = [&](int si, int di, int ai) {
        std::vector<float> v;
        for (int r = 0; r < reps; r++) if (rall[ridx(r, si, di, ai)] > 0) v.push_back(rall[ridx(r, si, di, ai)]);
        if (v.empty()) return -1.0f;
        return median(v);
    };

    auto run_once = [&](int si, int di, int ai) {
        const shape & s = SHAPES[si];
        const int64_t nd = s.nd[di];
        const rocblas_half one = { h2bits((_Float16) 1.0f) }, zero = { h2bits((_Float16) 0.0f) };
        CHECK(hipEventRecord(ev0, nullptr));
        if (ai == NARMS) {
            for (int it = 0; it < iters; it++) {
                RB_CHECK(rocblas_hgemm(rb, rocblas_operation_transpose, rocblas_operation_none,
                    (int) nd, (int) M, (int) s.k, &one, (const rocblas_half *) Wd, (int) s.k,
                    (const rocblas_half *) Xd, (int) s.k, &zero, Crb, (int) nd));
            }
        } else {
            const arm_def & a = ARMS[ai];
            const int ks = gks[si][ai];
            dim3 b(a.mt * a.nt / (a.ty * a.tx));
            dim3 g2((unsigned)((nd + a.nt - 1) / a.nt), (unsigned)(M / a.mt), (unsigned) ks);
            if (ks > 1) {
                const int mn = (int)(M * nd);
                for (int it = 0; it < iters; it++) {
                    a.launch(Wd, Xd, Cd, Pd, (int) nd, (int) s.k, g2, b, nullptr);
                    hipLaunchKernelGGL(reduce_pk, dim3((mn / 4 + 255) / 256), dim3(256), 0, nullptr,
                                       (const float *) Pd, Cd, mn, ks);
                }
            } else {
                for (int it = 0; it < iters; it++) {
                    a.launch(Wd, Xd, Cd, Pd, (int) nd, (int) s.k, g2, b, nullptr);
                }
            }
        }
        CHECK(hipEventRecord(ev1, nullptr));
        CHECK(hipEventSynchronize(ev1));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        return ms / iters;
    };

    printf("\nphase 2: %d interleaved reps x %d iters\n\n", reps, iters);
    printf("%-13s %4s %5s %6s %5s %10s %8s %8s\n", "shape", "die", "arm", "ks", "THT", "ms", "TF/s", "gate%");
    for (int si = 0; si < nsh; si++) {
        const shape & s = SHAPES[si];
        for (int d = 0; d < 3; d++) {
            const int64_t nd = s.nd[d];
            if (nd == 0) continue;
            const double fl = 2.0 * nd * M * s.k;
            // arm list: rb16 + passing tiles
            std::vector<int> list;
            list.push_back(NARMS);
            for (int ai = 0; ai < NARMS; ai++) if (gate[si][ai].pass) list.push_back(ai);
            // warmup each (Tensile kernel-select cache for rb16, icache for tiles)
            for (int ai : list) run_once(si, d, ai);
            for (int r = 0; r < reps; r++) {
                for (int ai : list) {
                    rall[ridx(r, si, d, ai)] = run_once(si, d, ai);
                }
            }
            for (int ai : list) {
                const float ms = get_ms(si, d, ai);
                const double tfs = fl / (ms * 1e-3) / 1e12;
                const char * nm = (ai == NARMS) ? "rb16" : ARMS[ai].name;
                const int ks = (ai == NARMS) ? 1 : gks[si][ai];
                const int tht = (ai == NARMS) ? 0 : ARMS[ai].mt * ARMS[ai].nt / (ARMS[ai].ty * ARMS[ai].tx);
                printf("%-13s %4d %5s %6d %5d %10.5f %8.2f %8.1f\n",
                        s.name, d, nm, ks, tht, ms, tfs, 100.0 * tfs / 4.62);
            }
        }
    }

    // checksum sanity + aggregates, same weights as the E-015 baseline run
    std::vector<float> probe(64);
    CHECK(hipMemcpy(probe.data(), Cd, 64 * 4, hipMemcpyDeviceToHost));
    double csum = 0;
    for (int i = 0; i < 64; i++) csum += probe[i];
    printf("\nf32-out checksum(64) %.3e %s\n", csum, csum > 0 && std::isfinite(csum) ? "ok" : "BAD");

    printf("\ncall-weighted aggregates (same weights as E-015 run):\n");
    for (int ai = 0; ai <= NARMS; ai++) {
        // die0 aggregate (baseline txt: rb16 = 4.52) and all-die aggregate (ledger: 4.62)
        const char * nm = (ai == NARMS) ? "rb16" : ARMS[ai].name;
        bool any = (ai == NARMS);
        for (int si = 0; si < nsh && !any; si++) any = gate[si][ai].pass;
        if (!any) continue;
        for (int si = 0; si < nsh && any; si++) {
            for (int d = 0; d < 3 && any; d++) {
                if (SHAPES[si].nd[d] != 0 && get_ms(si, d, ai) < 0) any = false;
            }
        }
        if (!any) continue;
        double fl0 = 0, ms0 = 0, flA = 0, msA = 0;
        for (int si = 0; si < nsh; si++) {
            const shape & s = SHAPES[si];
            fl0 += 2.0 * s.nd[0] * M * s.k * s.calls;
            ms0 += get_ms(si, 0, ai) * s.calls;
            for (int d = 0; d < 3; d++) {
                if (s.nd[d] == 0) continue;
                flA += 2.0 * s.nd[d] * M * s.k * s.calls;
                msA += get_ms(si, d, ai) * s.calls;
            }
        }
        printf("  %s: die0 agg %.2f TF/s/die (vs rb16-txt 4.52) | all-die agg %.2f TF/s/die"
               " (W5 gate 4.62: %s)\n",
                nm, fl0 / (ms0 * 1e-3) / 1e12, flA / (msA * 1e-3) / 1e12,
                flA / (msA * 1e-3) / 1e12 > 4.62 ? "MET" : "missed");
    }

    RB_CHECK(rocblas_destroy_handle(rb));
    CHECK(hipFree(Wd)); CHECK(hipFree(Xd)); CHECK(hipFree(Cd)); CHECK(hipFree(Pd)); CHECK(hipFree(Crb));
    return 0;
}
