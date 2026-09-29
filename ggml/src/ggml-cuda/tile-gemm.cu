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

// ACC: accumulate into C instead of overwriting (chunked arm, slices z > 0; the
// f32 add order then equals the unchunked fixed-order slice reduce)
template <int MT, int NT, int TY, int TX, int KC, bool ACC>
__global__ void __launch_bounds__(MT * NT / (TY * TX))
tile_fp16_gemm(const __half * __restrict__ Wp,   // N x K, k contiguous (K = row stride)
               const __half * __restrict__ X,    // M x Kfull, xs row stride
               float * __restrict__ C,           // M x N_d, ldc row stride (used when gridDim.z==1)
               float * __restrict__ P,           // partials (gridDim.z>1)
               int N, int K, int ksl, int64_t ldc, int xs) {
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
    const int kb0 = z * ksl;           // slice start within the X row
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
                    X + (mb + m) * xs + kb + sub * 8);
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
            float o = __low2float(acc[r][t]) + __high2float(acc[r][t]);
            const int col = n0 + cg + t * NC;
            if (direct) {
                if (ACC) o += C[(size_t)(crow + r) * ldc + col];
                C[(size_t)(crow + r) * ldc + col] = o;
            }
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

// N-span arm: the same fixed-order f32 slice sum for one span's partials (ks x M x
// nsp, row-major); the span's dst columns sit at row stride ldc, column offset coff
// (a multiple of NT, like nsp, so the float4 mapping stays 16 B aligned when ldc is)
__global__ void tile_fp16_reduce_span(const float * __restrict__ P, float * __restrict__ C,
                                      int mn, int ks, int ncols, int64_t ldc, int64_t coff) {
    const int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i4 + 3 < mn) {
        float4 s = *reinterpret_cast<const float4 *>(P + i4);
        for (int z = 1; z < ks; z++) {
            const float4 v = *reinterpret_cast<const float4 *>(P + (size_t) z * mn + i4);
            s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
        const int row = i4 / ncols;
        *reinterpret_cast<float4 *>(C + (size_t) row*ldc + coff + i4 - (size_t) row*ncols) = s;
    }
}

// chunked arm: copy one row's k-slice worth of quantized blocks (enclosing blocks
// when the slice start/end is not block-aligned) into the contiguous gather window,
// so the vetted per-type to_fp16 dequant kernels run unchanged on it and produce
// bit-identical f16 values
__global__ void tile_chunk_gather_q(const char * __restrict__ src, char * __restrict__ dst,
                                    int64_t row_bytes, int64_t seg_off, int64_t seg_bytes) {
    const int64_t n = blockIdx.x;
    const char * s = src + n*row_bytes + seg_off;
    char * d = dst + n*seg_bytes;
    const bool a16 = (((uintptr_t) s | (uintptr_t) d) & 15) == 0;
    const bool a4  = (((uintptr_t) s | (uintptr_t) d) &  3) == 0;
    if (a16) {
        const int64_t body = seg_bytes & ~15;
        for (int64_t i = threadIdx.x*16; i < body; i += blockDim.x*16) {
            *reinterpret_cast<uint4 *>(d + i) = *reinterpret_cast<const uint4 *>(s + i);
        }
        for (int64_t i = body + threadIdx.x; i < seg_bytes; i += blockDim.x) d[i] = s[i];
    } else if (a4) {
        const int64_t body = seg_bytes & ~3;
        for (int64_t i = threadIdx.x*4; i < body; i += blockDim.x*4) {
            *reinterpret_cast<uint32_t *>(d + i) = *reinterpret_cast<const uint32_t *>(s + i);
        }
        for (int64_t i = body + threadIdx.x; i < seg_bytes; i += blockDim.x) d[i] = s[i];
    } else {
        for (int64_t i = threadIdx.x; i < seg_bytes; i += blockDim.x) d[i] = s[i];
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

// loader step: steady-state f16 weights. GGML_CUDA_TILE_FP16_RESIDENT=1 keeps one
// f16 copy per (weight slice, device), produced at first use by the same dequant
// kernel the per-call route runs; the tile then reads the stable resident buffer
// and the per-call dequant disappears. Budget: cumulative per-device cap
// GGML_CUDA_TILE_FP16_RESIDENT_MIB (default 4096) AND live free VRAM at decision
// time minus a 256 MiB margin; refusal is final for the tensor and falls back to
// the per-call dequant. Only reached through tile_fp16_should_use, so the
// residency whitelist is exactly the tile census whitelist (trunk K set).

static bool tile_fp16_resident_env_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TILE_FP16_RESIDENT");
        const bool on = env != nullptr && env[0] == '1';
        if (on) {
            GGML_LOG_INFO("ggml-cuda: GGML_CUDA_TILE_FP16_RESIDENT=1, load-time f16 weight residency enabled\n");
        }
        return on;
    }();
    return enabled;
}

static size_t tile_fp16_resident_cap_bytes() {
    static const size_t cap = [] {
        const char * env = getenv("GGML_CUDA_TILE_FP16_RESIDENT_MIB");
        const size_t mib = env != nullptr ? (size_t) atoll(env) : 4096;
        return mib * 1024 * 1024;
    }();
    return cap;
}

// returns the resident f16 weights, or nullptr to dequantize per call
static const half * tile_fp16_resident_weight(
        ggml_backend_cuda_context & ctx, int id,
        const ggml_tensor * src0, const char * src0_dd_i, int64_t row_diff, int64_t ne00,
        const to_fp16_cuda_t to_fp16_cuda, cudaStream_t stream) {
    if (!tile_fp16_resident_env_enabled()) {
        return nullptr;
    }

    auto & cache = ctx.tile_fp16_residency;
    if (!cache.active) {
        cache.active   = true;
        cache.cap_bytes = tile_fp16_resident_cap_bytes();
    }

    const ggml_cuda_tile_fp16_residency::key key { id, src0_dd_i, ne00, row_diff, (int64_t) src0->type };

    const auto hit = cache.entries.find(key);
    if (hit != cache.entries.end()) {
        cache.n_hits++;
        return (const half *) hit->second.buf;
    }
    cache.n_misses++;

    if (cache.refused.count(key) > 0) {
        return nullptr;
    }

    // first use for this slice: the budget decision is made once, here, when free
    // VRAM is observable; a later retry would churn per call and the answer does
    // not change (weights grow, free VRAM only shrinks from here)
    const size_t bytes = row_diff*ne00*sizeof(half);
    if (cache.total_bytes + bytes > cache.cap_bytes) {
        cache.refused.insert(key);
        cache.n_refused++;
        GGML_LOG_WARN("%s: residency cap %.0f MiB full (%.1f MiB used), per-call dequant stays for K=%lld rows=%lld\n",
                      __func__, (double) cache.cap_bytes / (1024.0*1024.0), (double) cache.total_bytes / (1024.0*1024.0),
                      (long long) ne00, (long long) row_diff);
        return nullptr;
    }

    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));   // device id is current in the op
    constexpr size_t margin = 256ull*1024*1024;      // graph/compute headroom beyond the cap
    if (bytes + margin > free_b) {
        cache.refused.insert(key);
        cache.n_refused++;
        GGML_LOG_WARN("%s: %.1f MiB free < %.1f MiB needed (+%zu MiB margin), per-call dequant stays for K=%lld rows=%lld\n",
                      __func__, (double) free_b / (1024.0*1024.0), (double) bytes / (1024.0*1024.0), (size_t) (margin / (1024*1024)),
                      (long long) ne00, (long long) row_diff);
        return nullptr;
    }

    void * buf = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, bytes));
    to_fp16_cuda(src0_dd_i, (half *) buf, row_diff*ne00, stream);

    cache.entries.emplace(key, ggml_cuda_tile_fp16_residency::entry { buf, bytes });
    cache.total_bytes += bytes;
    GGML_LOG_INFO("%s: resident f16 K=%lld rows=%lld (%.1f MiB, %.1f MiB resident total, %.1f MiB free)\n",
                  __func__, (long long) ne00, (long long) row_diff,
                  (double) bytes / (1024.0*1024.0), (double) cache.total_bytes / (1024.0*1024.0),
                  (double) (free_b - bytes) / (1024.0*1024.0));
    return (const half *) buf;
}

// chunked-dequant arm: env + window budget. GGML_CUDA_TILE_FP16_CHUNKED=1 on top
// of GGML_CUDA_TILE_FP16=1 dequantizes one split-K slice at a time into a small
// reusable f16 window and accumulates the slice GEMMs into dst in the fixed slice
// order, so the route never holds a whole f16 weight copy (the per-call pool alloc
// that OOMs at 200k, E-044). Slice boundaries are the unchunked kernel's z
// partition and each slice launch runs the same half2 schedule, so the f32
// accumulation order equals the unchunked reduce order and the output is
// bit-identical to it. A slice can start mid quant block, so the window covers the
// enclosing blocks and the GEMM reads it at offset r = kb0 - jb0*bs with row
// stride l. Window: N_d x (ksl + bs) x 2 B plus the gathered quantized bytes
// (~18 MiB for the ffn gate/up slice vs the 59 MiB per-call dequant of that
// weight). Refusal is clean: shipped route, no abort.
static bool tile_fp16_chunked_env_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TILE_FP16_CHUNKED");
        const bool on = env != nullptr && env[0] == '1';
        if (on) {
            GGML_LOG_INFO("ggml-cuda: GGML_CUDA_TILE_FP16_CHUNKED=1, chunked weight dequant into a reusable slice window\n");
        }
        return on;
    }();
    return enabled;
}

static size_t tile_fp16_chunked_cap_bytes() {
    static const size_t cap = [] {
        const char * env = getenv("GGML_CUDA_TILE_FP16_CHUNKED_MIB");
        const size_t mib = env != nullptr ? (size_t) atoll(env) : 64;
        return mib * 1024 * 1024;
    }();
    return cap;
}

static bool tile_fp16_chunked_run(
        ggml_backend_cuda_context & ctx, int id, cudaStream_t stream,
        const ggml_tensor * src0, const ggml_tensor * src1,
        const char * src0_dd_i, const float * src1_ddf_i, float * dst_dd_i,
        int64_t row_diff, int64_t src1_ncols, int64_t ne00, int64_t ldc, int ks,
        const to_fp16_cuda_t to_fp16_cuda) {
    const int  bs  = ggml_blck_size(src0->type);
    const int  tsz = ggml_type_size(src0->type);
    const int  ksl = ne00 / ks;

    // window size: the max padded slice length (slices can start mid quant block)
    int64_t lmax = 0;
    for (int z = 0; z < ks; z++) {
        const int64_t r = ((int64_t) z*ksl) % bs;
        const int64_t l = ((r + ksl + bs - 1) / bs) * bs;
        if (l > lmax) lmax = l;
    }
    if ((int64_t) row_diff*lmax + lmax > INT32_MAX) {   // 32-bit W indexing in the kernel
        return false;
    }

    const size_t f16_bytes = row_diff*lmax*sizeof(half);
    const size_t q_bytes   = row_diff*(lmax/bs)*tsz;

    auto & cache = ctx.tile_fp16_chunk;
    if (cache.cap_bytes == 0) {
        cache.cap_bytes = tile_fp16_chunked_cap_bytes();
    }
    ggml_cuda_tile_fp16_chunk::entry * e = cache.find(stream);
    if (e == nullptr || e->f16_bytes < f16_bytes || e->q_bytes < q_bytes) {
        const size_t old = e ? e->f16_bytes + e->q_bytes : 0;
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        constexpr size_t margin = 64ull*1024*1024;   // compute headroom beyond the window
        if (cache.total_bytes - old + f16_bytes + q_bytes > cache.cap_bytes ||
            f16_bytes + q_bytes + margin > free_b) {
            static std::once_flag warn_flag;
            std::call_once(warn_flag, []() {
                GGML_LOG_WARN("%s: slice window over budget or free VRAM, shipped route in use\n", __func__);
            });
            return false;
        }
        void * nf = nullptr;
        void * nq = nullptr;
        if (cudaMalloc(&nf, f16_bytes) != cudaSuccess ||
            (q_bytes > 0 && cudaMalloc(&nq, q_bytes) != cudaSuccess)) {
            (void) cudaGetLastError();
            if (nf) CUDA_CHECK(cudaFree(nf));
            static std::once_flag warn_flag;
            std::call_once(warn_flag, []() {
                GGML_LOG_WARN("%s: slice window alloc failed, shipped route in use\n", __func__);
            });
            return false;
        }
        if (e) {
            CUDA_CHECK(cudaStreamSynchronize(stream));   // the replaced window may be in flight
            CUDA_CHECK(cudaFree(e->f16));
            if (e->q) CUDA_CHECK(cudaFree(e->q));
            cache.total_bytes -= old;
            e->f16 = nf; e->q = nq;
            e->f16_bytes = f16_bytes; e->q_bytes = q_bytes;
        } else {
            cache.entries.push_back(ggml_cuda_tile_fp16_chunk::entry { stream, nf, nq, f16_bytes, q_bytes });
            e = &cache.entries.back();
        }
        cache.total_bytes += f16_bytes + q_bytes;
        GGML_LOG_INFO("%s: slice window %.1f MiB (f16 %zu B + q %zu B), %.1f MiB free\n",
                      __func__, (double) (f16_bytes + q_bytes) / (1024.0*1024.0),
                      f16_bytes, q_bytes, (double) (free_b - f16_bytes - q_bytes) / (1024.0*1024.0));
    }

    const to_fp16_cuda_t to_fp16_cuda_src1 = ggml_get_to_fp16_cuda(src1->type);
    GGML_ASSERT(to_fp16_cuda_src1 != nullptr);
    ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id), src1_ncols*ne00);
    to_fp16_cuda_src1(src1_ddf_i, src1_as_f16.get(), src1_ncols*ne00, stream);

    constexpr int mt = TILE_FP16_MT, nt = TILE_FP16_NT, ty = TILE_FP16_TY, tx = TILE_FP16_TX;
    const dim3 block(mt * nt / (ty * tx));
    const dim3 grid((unsigned) (row_diff / nt), (unsigned) (src1_ncols / mt), 1);
    const int64_t row_q_bytes = (int64_t) (ne00/bs)*tsz;

    for (int z = 0; z < ks; z++) {
        const int64_t kb0 = (int64_t) z*ksl;
        const int64_t jb0 = kb0 / bs;
        const int64_t r   = kb0 - jb0*bs;
        const int64_t l   = ((r + ksl + bs - 1) / bs) * bs;
        tile_chunk_gather_q<<<(unsigned) row_diff, 256, 0, stream>>>(
            src0_dd_i, (char *) e->q, row_q_bytes, jb0*tsz, (l/bs)*tsz);
        to_fp16_cuda(e->q, (half *) e->f16, row_diff*l, stream);
        if (z == 0) {
            tile_fp16_gemm<mt, nt, ty, tx, TILE_FP16_KC, false><<<grid, block, 0, stream>>>(
                (const half *) e->f16 + r, src1_as_f16.get() + kb0, dst_dd_i, nullptr,
                (int) row_diff, (int) l, ksl, ldc, (int) ne00);
        } else {
            tile_fp16_gemm<mt, nt, ty, tx, TILE_FP16_KC, true><<<grid, block, 0, stream>>>(
                (const half *) e->f16 + r, src1_as_f16.get() + kb0, dst_dd_i, nullptr,
                (int) row_diff, (int) l, ksl, ldc, (int) ne00);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    static std::once_flag once_flag;
    std::call_once(once_flag, [&]() {
        GGML_LOG_INFO("%s: tile route engaged, chunked dequant (K=%lld N_d=%lld M=%lld ks=%d window=%.1f MiB ldc=%lld)\n",
                      __func__, (long long) ne00, (long long) row_diff,
                      (long long) src1_ncols, ks, (double) (f16_bytes + q_bytes) / (1024.0*1024.0),
                      (long long) ldc);
    });
    return true;
}

// N-span full-K arm: GGML_CUDA_TILE_FP16_NSPAN=1 on top of GGML_CUDA_TILE_FP16=1
// processes the weight rows in contiguous N-spans with the FULL K dimension, so the
// dequant runs on the quant tensor directly (contiguous row slabs, no gather) and
// every span launch keeps the unchunked kernel's gridDim.z = ks partition plus its
// P + fixed-order reduce. Output columns are disjoint across spans and each
// element's slice schedule and f32 sum order are the unchunked ones, so the result
// is bit-identical to the unchunked arm while the route never holds a whole f16
// weight copy in the per-call pool (the E-044 OOM class). Window: one f16 slab
// (span x K) plus one span partials slab (ks x M x span), per stream, grow-only,
// freed at context teardown. Budget: GGML_CUDA_TILE_FP16_NSPAN_MIB (default 128)
// total per stream AND free VRAM minus a 64 MiB margin at decision time; refusal
// falls back to the chunked / shipped routes with one WARN - never aborts.
static bool tile_fp16_nspan_env_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TILE_FP16_NSPAN");
        const bool on = env != nullptr && env[0] == '1';
        if (on) {
            GGML_LOG_INFO("ggml-cuda: GGML_CUDA_TILE_FP16_NSPAN=1, N-span full-K weight dequant into a reusable window\n");
        }
        return on;
    }();
    return enabled;
}

static size_t tile_fp16_nspan_cap_bytes() {
    static const size_t cap = [] {
        const char * env = getenv("GGML_CUDA_TILE_FP16_NSPAN_MIB");
        const size_t mib = env != nullptr ? (size_t) atoll(env) : 128;
        return mib * 1024 * 1024;
    }();
    return cap;
}

static bool tile_fp16_nspan_run(
        ggml_backend_cuda_context & ctx, int id, cudaStream_t stream,
        const ggml_tensor * src0, const ggml_tensor * src1,
        const char * src0_dd_i, const float * src1_ddf_i, float * dst_dd_i,
        int64_t row_diff, int64_t src1_ncols, int64_t ne00, int64_t ldc, int ks,
        const to_fp16_cuda_t to_fp16_cuda) {
    // the span reduce maps float4 lanes into strided dst rows; spans and column
    // offsets are NT-aligned, the dst row stride must keep the writes 16 B aligned
    if (ks > 1 && ldc % 4 != 0) {
        return false;
    }

    // span size: the largest NT-aligned row count whose full-K f16 slab fits the
    // window budget (a single span when the whole slice fits)
    int64_t nsp = row_diff;
    if (row_diff*ne00*sizeof(half) > tile_fp16_nspan_cap_bytes()) {
        nsp = (int64_t) (tile_fp16_nspan_cap_bytes() / (ne00*sizeof(half))) / TILE_FP16_NT * TILE_FP16_NT;
        if (nsp < TILE_FP16_NT) {
            return false;
        }
    }

    const int  bs  = ggml_blck_size(src0->type);
    const int  tsz = ggml_type_size(src0->type);

    const size_t f16_bytes = nsp*ne00*sizeof(half);
    const size_t p_bytes   = (size_t) ks*src1_ncols*nsp*sizeof(float);

    auto & cache = ctx.tile_fp16_nspan;
    if (cache.cap_bytes == 0) {
        cache.cap_bytes = tile_fp16_nspan_cap_bytes();
    }
    ggml_cuda_tile_fp16_nspan::entry * e = cache.find(stream);
    if (e == nullptr || e->f16_bytes < f16_bytes || e->p_bytes < p_bytes) {
        const size_t old = e ? e->f16_bytes + e->p_bytes : 0;
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        constexpr size_t margin = 64ull*1024*1024;   // compute headroom beyond the windows
        if (cache.total_bytes - old + f16_bytes + p_bytes > cache.cap_bytes ||
            f16_bytes + p_bytes + margin > free_b) {
            static std::once_flag warn_flag;
            std::call_once(warn_flag, []() {
                GGML_LOG_WARN("%s: span window over budget or free VRAM, shipped route in use\n", __func__);
            });
            return false;
        }
        void * nf = nullptr;
        void * np = nullptr;
        if (cudaMalloc(&nf, f16_bytes) != cudaSuccess ||
            (p_bytes > 0 && cudaMalloc(&np, p_bytes) != cudaSuccess)) {
            (void) cudaGetLastError();
            if (nf) CUDA_CHECK(cudaFree(nf));
            static std::once_flag warn_flag;
            std::call_once(warn_flag, []() {
                GGML_LOG_WARN("%s: span window alloc failed, shipped route in use\n", __func__);
            });
            return false;
        }
        if (e) {
            CUDA_CHECK(cudaStreamSynchronize(stream));   // the replaced window may be in flight
            CUDA_CHECK(cudaFree(e->f16));
            if (e->p) CUDA_CHECK(cudaFree(e->p));
            cache.total_bytes -= old;
            e->f16 = nf; e->p = np;
            e->f16_bytes = f16_bytes; e->p_bytes = p_bytes;
        } else {
            cache.entries.push_back(ggml_cuda_tile_fp16_nspan::entry { stream, nf, np, f16_bytes, p_bytes });
            e = &cache.entries.back();
        }
        cache.total_bytes += f16_bytes + p_bytes;
        GGML_LOG_INFO("%s: span window %.1f MiB (f16 %zu B + p %zu B), %.1f MiB free\n",
                      __func__, (double) (f16_bytes + p_bytes) / (1024.0*1024.0),
                      f16_bytes, p_bytes, (double) (free_b - f16_bytes - p_bytes) / (1024.0*1024.0));
    }

    const to_fp16_cuda_t to_fp16_cuda_src1 = ggml_get_to_fp16_cuda(src1->type);
    GGML_ASSERT(to_fp16_cuda_src1 != nullptr);
    ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id), src1_ncols*ne00);
    to_fp16_cuda_src1(src1_ddf_i, src1_as_f16.get(), src1_ncols*ne00, stream);

    constexpr int mt = TILE_FP16_MT, nt = TILE_FP16_NT, ty = TILE_FP16_TY, tx = TILE_FP16_TX;
    const dim3 block(mt * nt / (ty * tx));
    const int  ksl = (int) (ne00 / ks);
    const int64_t row_q_bytes = (int64_t) (ne00/bs)*tsz;

    // even NT-aligned spans, the last takes the remainder; the GEMM reads the
    // span's f16 rows at window row stride K and writes span-local partials
    // (N param = ns), or dst columns directly when ks == 1 (C base offset r0)
    int nspans = 1;
    if (nsp < row_diff) {
        nspans = (int) ((row_diff + nsp - 1) / nsp);
        nsp    = ((row_diff + nspans - 1) / nspans + nt - 1) / nt * nt;
    }

    for (int s = 0; s < nspans; s++) {
        const int64_t r0 = (int64_t) s*nsp;
        const int64_t ns = std::min(nsp, row_diff - r0);
        to_fp16_cuda(src0_dd_i + r0*row_q_bytes, (half *) e->f16, ns*ne00, stream);
        const dim3 grid((unsigned) (ns / nt), (unsigned) (src1_ncols / mt), (unsigned) ks);
        tile_fp16_gemm<mt, nt, ty, tx, TILE_FP16_KC, false><<<grid, block, 0, stream>>>(
            (const half *) e->f16, src1_as_f16.get(), dst_dd_i + r0, (float *) e->p,
            (int) ns, (int) ne00, ksl, ldc, (int) ne00);
        if (ks > 1) {
            const int mn = (int) (src1_ncols*ns);
            tile_fp16_reduce_span<<<(unsigned) ((mn/4 + 255) / 256), 256, 0, stream>>>(
                (const float *) e->p, dst_dd_i, mn, ks, (int) ns, ldc, r0);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    static std::once_flag once_flag;
    std::call_once(once_flag, [&]() {
        GGML_LOG_INFO("%s: tile route engaged, N-span full-K dequant (K=%lld N_d=%lld M=%lld ks=%d spans=%d span=%lld window=%.1f MiB ldc=%lld)\n",
                      __func__, (long long) ne00, (long long) row_diff,
                      (long long) src1_ncols, ks, nspans, (long long) nsp,
                      (double) (f16_bytes + p_bytes) / (1024.0*1024.0), (long long) ldc);
    });
    return true;
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

    // f16 weights: the resident steady-state copy when the budget admits the
    // slice, else N-span full-K dequant (GGML_CUDA_TILE_FP16_NSPAN=1), else
    // chunked dequant (GGML_CUDA_TILE_FP16_CHUNKED=1), else the per-call dequant
    // the shipped cublas route produces
    const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
    GGML_ASSERT(to_fp16_cuda != nullptr);
    const half * w_f16 = tile_fp16_resident_weight(ctx, id, src0, src0_dd_i, row_diff, ne00, to_fp16_cuda, stream);
    if (w_f16 == nullptr && tile_fp16_nspan_env_enabled() &&
        tile_fp16_nspan_run(ctx, id, stream, src0, src1, src0_dd_i, src1_ddf_i, dst_dd_i,
                            row_diff, src1_ncols, ne00, ldc, ks, to_fp16_cuda)) {
        return true;
    }
    if (w_f16 == nullptr && tile_fp16_chunked_env_enabled() &&
        tile_fp16_chunked_run(ctx, id, stream, src0, src1, src0_dd_i, src1_ddf_i, dst_dd_i,
                              row_diff, src1_ncols, ne00, ldc, ks, to_fp16_cuda)) {
        return true;
    }
    ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
    if (w_f16 == nullptr) {
        src0_as_f16.alloc(row_diff*ne00);
        to_fp16_cuda(src0_dd_i, src0_as_f16.get(), row_diff*ne00, stream);
        w_f16 = src0_as_f16.get();
    }

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

    tile_fp16_gemm<mt, nt, ty, tx, TILE_FP16_KC, false><<<grid, block, 0, stream>>>(
        w_f16, src1_as_f16.get(), dst_dd_i, p, (int) row_diff, (int) ne00, (int) (ne00/ks), ldc, (int) ne00);

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
