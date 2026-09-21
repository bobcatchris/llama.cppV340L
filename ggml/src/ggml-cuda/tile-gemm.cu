#include "tile-gemm.cuh"
#include "convert.cuh"

// split-K choice: fill the 56-CU device (224 SIMD target) without breaking the
// K step law (slice must stay a multiple of KC)
inline int tile_fp16_pick_ks(int64_t n, int64_t k) {
    constexpr int kc = TILE_FP16_KC;
    constexpr int nt = TILE_FP16_NT;
    const int64_t grid = (n + nt - 1) / nt;
    int ks = 1;
    while (grid*ks < 224 && ks < 8) {
        const int cand = ks*2;
        if (k % cand != 0) break;
        if ((k / cand) % kc != 0) break;
        ks = cand;
    }
    return ks;
}

template <int MT, int NT, int TY, int TX, int KC>
__global__ void __launch_bounds__(MT * NT / (TY * TX))
tile_fp16_gemm(const __half * __restrict__ Wp,   // N x K, k contiguous
               const __half * __restrict__ X,    // M x K, k contiguous
               float * __restrict__ C,           // M x N_d, ldc row stride (used when gridDim.z==1)
               float * __restrict__ P,           // partials (gridDim.z>1)
               int N, int K, int64_t ldc) {
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
    // ownership: thread owns columns cg + t*NC), tid/NC = row group
    constexpr int NC = NT / TX;
    const int cg  = tid % NC;
    const int mg  = tid / NC;
    const int m0t = mg * TY;

    half2 acc[TY][TX];
#pragma unroll
    for (int r = 0; r < TY; r++) {
#pragma unroll
        for (int t = 0; t < TX; t++) {
            acc[r][t] = __float2half2_rn(0.0f);
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
        }
        __syncthreads();
    }

    const bool direct = (ks == 1);
    const int crow = mb + m0t;
#pragma unroll
    for (int r = 0; r < TY; r++) {
#pragma unroll
        for (int t = 0; t < TX; t++) {
            const float o = __low2float(acc[r][t]) + __high2float(acc[r][t]);
            const int col = n0 + cg + t * NC;
            if (direct) C[(size_t)(crow + r) * ldc + col] = o;
            else        P[((size_t) z * gridDim.y * MT + crow + r) * N + col] = o;
        }
    }
}

// fixed-order f32 slice sum, in the timed region when ks > 1
__global__ void tile_fp16_reduce(const float * __restrict__ P, float * __restrict__ C,
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

// W5 tile route (E-022): env-gated dispatch of the a8 packed-fp16 GEMM tile behind
// the dense-prefill cublas fallback. GGML_CUDA_TILE_FP16=1 routes census prefill
// shapes (M <= 512, trunk K set, per-die N share 64-aligned) to the tile; everything
// else falls back to the shipped branches. Env unset = byte-identical shipped path.

static bool tile_fp16_env_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TILE_FP16");
        const bool on = env != nullptr && env[0] == '1';
        if (on) {
            GGML_LOG_INFO("ggml-cuda: GGML_CUDA_TILE_FP16=1, packed-fp16 tile GEMM route enabled\n");
        }
        return on;
    }();
    return enabled;
}

// census trunk whitelist; device guards keep the kernel's 32-bit index math and
// b128/float4 accesses valid. Any miss = shipped route (logged once).
static bool tile_fp16_should_use(int cc, const ggml_tensor * src0, const ggml_tensor * src1,
                                 const ggml_tensor * dst, const void * w, const void * x,
                                 const float * c, int64_t ne00, int64_t row_diff, int64_t src1_ncols,
                                 int * ks_out) {
    if (!tile_fp16_env_enabled()) {
        return false;
    }

    const bool ok = cc == GGML_CUDA_CC_VEGA &&                    // gfx900, the benched platform
        ggml_is_quantized(src0->type) &&                          // f16 weights come from the dequant
        src1->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) &&
        dst->op_params[0] == GGML_PREC_DEFAULT &&
        ne00 == src1->ne[0] &&
        src1_ncols <= 512 && src1_ncols % TILE_FP16_MT == 0 &&
        (ne00 == 5120 || ne00 == 6144 || ne00 == 17408) &&        // trunk K set
        row_diff >= TILE_FP16_NT && row_diff % TILE_FP16_NT == 0 &&
        row_diff*ne00 + ne00 <= INT32_MAX &&                      // 32-bit W/X indexing in the kernel
        src1_ncols*ne00 + ne00 <= INT32_MAX &&
        ((uintptr_t) w | (uintptr_t) x | (uintptr_t) c) % 16 == 0;

    if (!ok) {
        static std::once_flag warn_flag;
        std::call_once(warn_flag, []() {
            GGML_LOG_WARN("%s: shape off the census whitelist, shipped route in use\n", __func__);
        });
        return false;
    }

    const int ks = tile_fp16_pick_ks(row_diff, ne00);
    if (ks > 1 && (int64_t) ks*src1_ncols*row_diff > INT32_MAX) {  // 32-bit P indexing
        return false;
    }
    *ks_out = ks;
    return true;
}

bool ggml_cuda_tile_fp16_mul_mat(
        ggml_backend_cuda_context & ctx, int id,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
        const char * src0_dd_i, const float * src1_ddf_i, float * dst_dd_i,
        int64_t row_diff, int64_t src1_ncols, int64_t ldc, cudaStream_t stream) {
    const int64_t ne00 = src0->ne[0];

    int ks = 1;
    if (!tile_fp16_should_use(ggml_cuda_info().devices[id].cc, src0, src1, dst,
                              src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, src1_ncols, &ks)) {
        return false;
    }

    // same f16 weight buffer the shipped cublas route produces by per-call dequant;
    // load-time f16 steady state (deleting this tax) is a later step
    const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
    GGML_ASSERT(to_fp16_cuda != nullptr);
    ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id), row_diff*ne00);
    to_fp16_cuda(src0_dd_i, src0_as_f16.get(), row_diff*ne00, stream);

    const to_fp16_cuda_t to_fp16_cuda_src1 = ggml_get_to_fp16_cuda(src1->type);
    GGML_ASSERT(to_fp16_cuda_src1 != nullptr);
    ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id), src1_ncols*ne00);
    to_fp16_cuda_src1(src1_ddf_i, src1_as_f16.get(), src1_ncols*ne00, stream);

    ggml_cuda_pool_alloc<float> partials(ctx.pool(id));
    float * p = dst_dd_i;
    if (ks > 1) {
        partials.alloc(ctx.pool(id), ks*src1_ncols*row_diff);
        p = partials.get();
    }

    constexpr int mt = TILE_FP16_MT, nt = TILE_FP16_NT, ty = TILE_FP16_TY, tx = TILE_FP16_TX;
    const dim3 block(mt * nt / (ty * tx));
    const dim3 grid((unsigned) (row_diff / nt), (unsigned) (src1_ncols / mt), (unsigned) ks);

    tile_fp16_gemm<mt, nt, ty, tx, TILE_FP16_KC><<<grid, block, 0, stream>>>(
        src0_as_f16.get(), src1_as_f16.get(), dst_dd_i, p, (int) row_diff, (int) ne00, ldc);

    if (ks > 1) {
        const int64_t mn = src1_ncols*row_diff;
        tile_fp16_reduce<<<(unsigned) ((mn/4 + 255) / 256), 256, 0, stream>>>(p, dst_dd_i, (int) mn, ks);
    }
    CUDA_CHECK(cudaGetLastError());

    static std::once_flag once_flag;
    std::call_once(once_flag, [&]() {
        GGML_LOG_INFO("%s: tile route engaged (K=%lld N_d=%lld M=%lld ks=%d ldc=%lld)\n",
                      __func__, (long long) ne00, (long long) row_diff,
                      (long long) src1_ncols, ks, (long long) ldc);
    });
    return true;
}
