// REAL-kernel MMVQ instrument (W7 LDS-y desk): instantiates the actual
// mul_mat_vec_q template from ggml/src/ggml-cuda/mmvq.cu (this tree, included
// verbatim below) and drives it with the exact served GCN launch geometry
// (rpb = 2, nwarps = 2, block 64x2) on REAL GGUF tensor bytes (offset-delta
// law, ASCII-P1M census). This is the post-E-113 lesson instrument: no
// replica schedules; the timed kernels ARE the served template instances.
//
// Arms (per type/T, one process, oracle-gated):
//   base      share=false, the shipped per-token decode path (default variant)
//   share     decode-once share apply (the *_SHARE=1 gated path)
//   s2r       share + row-shared y preload (the *_S2R=1 gated path; requires
//             the s2r switch_type wiring fix this desk landed)
//   ws2r      s2r + wide-s2r y-preload hoist: all T y-preloads issued before
//             any apply (the GGML_CUDA_MMVQ_WIDE_S2R=1 gated path, W25/C3)
//
// ORACLE: the first arm in the list must be base; every later arm's full dst
// (N rows x T tokens) is device-copied and memcmp'd against it BEFORE any
// timing counts. A base dup at the end of the list doubles as the in-run
// determinism + drift control. Timing: hip events around niter back-to-back
// launches, reps times; spread law 1.05% per config.
//
// build (flags mirrored from build-hip):
//   /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP \
//     -DGGML_BACKEND_BUILD -DGGML_SHARED -I ggml/include -I ggml/src \
//     -I ggml/src/ggml-cuda docs/amd-port/tests/bench_mmvq_real.cu \
//     -o /tmp/bench_mmvq_real -L <build>/bin -lggml-hip -lggml-base -lamdhip64 \
//     -Wl,-rpath,<abs-build>/bin -Wl,-rpath,/opt/rocm-6.2.0/lib
// run (die 3 only, after check-and-hold of /tmp/campaign_gpu_boot.lock):
//   HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=<abs-build>/bin /tmp/bench_mmvq_real \
//     <gguf> <type-name> <T> <arm[,arm...]> [niter=30] [reps=4] [--occupancy]
#include "ggml.h"
#include "ggml-cuda/mmvq.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct TypeInfo {
    const char * name;
    ggml_type    type;
    long         off;   // gguf tensor byte offset (offset-delta law)
    long         len;
    int          K, N;
    int          bs;    // bytes per block
};

// served T=4 mix representatives (offset-delta law, ASCII-P1M census), same
// table as bench_mmvq_share_gfx900.cu
static const TypeInfo g_types[] = {
    {"iq3_xxs", GGML_TYPE_IQ3_XXS, 969405216L,   34119680L, 5120, 17408,  98},
    {"q4_K",    GGML_TYPE_Q4_K,    12025224992L, 50135040L, 5120, 17408, 144},
    {"iq4_xs",  GGML_TYPE_IQ4_XS,  1595404832L,  47349760L, 5120, 17408, 136},
    {"q5_K",    GGML_TYPE_Q5_K,    5220080544L,  43253760L, 5120, 12288, 176},
    {"q6_K",    GGML_TYPE_Q6_K,    4637536L,    542942400L, 5120, 129272, 210},
    {"q3_K",    GGML_TYPE_Q3_K,    10469424416L, 38297600L, 5120, 17408, 110},
    {"iq3_s",   GGML_TYPE_IQ3_S,   4301102016L,  38297600L, 5120, 17408, 110},
};

struct ArmSpec {
    std::string name;   // base | share | s2r
};

enum ArmKind { ARM_BASE, ARM_SHARE, ARM_S2R, ARM_WS2R };

static ArmKind arm_kind(const ArmSpec & a) {
    if (a.name == "base")     return ARM_BASE;
    if (a.name == "share")    return ARM_SHARE;
    if (a.name == "ws2r")     return ARM_WS2R;
    return ARM_S2R;
}

// exact launch arguments of ggml_cuda_mul_mat_vec_q for the 2D no-ids case
// (GCN table: T 2..4 -> nwarps 2, rows_per_cuda_block 2, grid (N/2, 1, 1))
template <ggml_type type, int T, bool S2R, bool WS2R = false>
static void launch_k(const void * vx, const block_q8_1 * vy, float * dst,
                     const int K, const int stride_row_x, const int stride_col_y,
                     const uint32_t stride_col_dst, const uint32_t stride_channel_y,
                     const uint32_t stride_sample_y, const uint32_t stride_channel_dst,
                     const uint32_t stride_sample_dst, dim3 grid, dim3 block,
                     hipStream_t stream, bool share) {
    const ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q<type, T, false, false, false, S2R, WS2R>
        <<<grid, block, 0, stream>>>(
            vx, vy, vy, nullptr, fusion, dst,
            (const uint32_t) K, make_uint3(0, 0, 0), (const uint32_t) stride_row_x,
            (const uint32_t) stride_col_y, stride_col_dst,
            init_fastdiv_values(1), 1, stride_channel_y, stride_channel_dst,
            init_fastdiv_values(1), 1, stride_sample_y, stride_sample_dst,
            0, share);
    HIP_CHECK(hipGetLastError());
}

template <ggml_type type, int T, bool S2R, bool WS2R = false>
static void occ_k(const char * name) {
    hipFuncAttributes attr{};
    const void * kp = (const void *) &mul_mat_vec_q<type, T, false, false, false, S2R, WS2R>;
    HIP_CHECK(hipFuncGetAttributes(&attr, kp));
    int blocks = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kp, 128, 0));
    printf("OCC %s: numRegs=%d shared=%zu local=%zu maxThreadsPerBlock=%d ctas_per_cu=%d\n",
           name, attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes, attr.maxThreadsPerBlock, blocks);
}

template <ggml_type type, int T>
static void dispatch_arm(const ArmSpec & a, const void * vx, const block_q8_1 * vy,
                         float * dst, const int K, const int stride_row_x, const int stride_col_y,
                         const uint32_t stride_col_dst, const uint32_t stride_channel_y,
                         const uint32_t stride_sample_y, const uint32_t stride_channel_dst,
                         const uint32_t stride_sample_dst, dim3 grid, dim3 block,
                         hipStream_t stream, bool occupancy) {
    const ArmKind k = arm_kind(a);
    char nm[96];
    snprintf(nm, sizeof(nm), "%s/%s/T%d", ggml_type_name(type), a.name.c_str(), T);
#define LK(S2R_, SHARE_) launch_k<type, T, S2R_>(vx, vy, dst, K, \
        stride_row_x, stride_col_y, stride_col_dst, stride_channel_y, stride_sample_y, \
        stride_channel_dst, stride_sample_dst, grid, block, stream, SHARE_)
#define LKW(SHARE_) launch_k<type, T, true, true>(vx, vy, dst, K, \
        stride_row_x, stride_col_y, stride_col_dst, stride_channel_y, stride_sample_y, \
        stride_channel_dst, stride_sample_dst, grid, block, stream, SHARE_)
#define OK(S2R_) occ_k<type, T, S2R_>(nm)
#define OKW() occ_k<type, T, true, true>(nm)
    switch (k) {
        case ARM_BASE:
            if (occupancy) { OK(false); return; }
            LK(false, false);
            break;
        case ARM_SHARE:
            if (occupancy) { OK(false); return; }
            LK(false, true);
            break;
        case ARM_S2R:
            if (occupancy) { OK(true); return; }
            LK(true, true);
            break;
        case ARM_WS2R:
            if (occupancy) { OKW(); return; }
            LKW(true);
            break;
    }
#undef LK
#undef LKW
#undef OK
#undef OKW
}

template <ggml_type type>
static void dispatch_T(int T, const ArmSpec & a, const void * vx, const block_q8_1 * vy,
                       float * dst, const int K, const int stride_row_x, const int stride_col_y,
                       const uint32_t stride_col_dst, const uint32_t stride_channel_y,
                       const uint32_t stride_sample_y, const uint32_t stride_channel_dst,
                       const uint32_t stride_sample_dst, dim3 grid, dim3 block,
                       hipStream_t stream, bool occupancy) {
    if (T == 2) {
        dispatch_arm<type, 2>(a, vx, vy, dst, K, stride_row_x, stride_col_y,
            stride_col_dst, stride_channel_y, stride_sample_y, stride_channel_dst,
            stride_sample_dst, grid, block, stream, occupancy);
    } else if (T == 4) {
        dispatch_arm<type, 4>(a, vx, vy, dst, K, stride_row_x, stride_col_y,
            stride_col_dst, stride_channel_y, stride_sample_y, stride_channel_dst,
            stride_sample_dst, grid, block, stream, occupancy);
    } else {
        printf("T=%d not instantiated (use 2 or 4)\n", T);
        exit(1);
    }
}

static void dispatch_type(ggml_type type, int T, const ArmSpec & a, const void * vx,
                          const block_q8_1 * vy, float * dst, const int K, const int stride_row_x,
                          const int stride_col_y, const uint32_t stride_col_dst,
                          const uint32_t stride_channel_y, const uint32_t stride_sample_y,
                          const uint32_t stride_channel_dst, const uint32_t stride_sample_dst,
                          dim3 grid, dim3 block, hipStream_t stream, bool occupancy) {
    switch (type) {
        case GGML_TYPE_IQ3_XXS: dispatch_T<GGML_TYPE_IQ3_XXS>(T, a, vx, vy, dst, K, stride_row_x, stride_col_y, stride_col_dst, stride_channel_y, stride_sample_y, stride_channel_dst, stride_sample_dst, grid, block, stream, occupancy); break;
        case GGML_TYPE_Q4_K:    dispatch_T<GGML_TYPE_Q4_K>   (T, a, vx, vy, dst, K, stride_row_x, stride_col_y, stride_col_dst, stride_channel_y, stride_sample_y, stride_channel_dst, stride_sample_dst, grid, block, stream, occupancy); break;
        case GGML_TYPE_IQ4_XS:  dispatch_T<GGML_TYPE_IQ4_XS> (T, a, vx, vy, dst, K, stride_row_x, stride_col_y, stride_col_dst, stride_channel_y, stride_sample_y, stride_channel_dst, stride_sample_dst, grid, block, stream, occupancy); break;
        case GGML_TYPE_Q5_K:    dispatch_T<GGML_TYPE_Q5_K>   (T, a, vx, vy, dst, K, stride_row_x, stride_col_y, stride_col_dst, stride_channel_y, stride_sample_y, stride_channel_dst, stride_sample_dst, grid, block, stream, occupancy); break;
        case GGML_TYPE_Q6_K:    dispatch_T<GGML_TYPE_Q6_K>   (T, a, vx, vy, dst, K, stride_row_x, stride_col_y, stride_col_dst, stride_channel_y, stride_sample_y, stride_channel_dst, stride_sample_dst, grid, block, stream, occupancy); break;
        case GGML_TYPE_Q3_K:    dispatch_T<GGML_TYPE_Q3_K>   (T, a, vx, vy, dst, K, stride_row_x, stride_col_y, stride_col_dst, stride_channel_y, stride_sample_y, stride_channel_dst, stride_sample_dst, grid, block, stream, occupancy); break;
        case GGML_TYPE_IQ3_S:   dispatch_T<GGML_TYPE_IQ3_S>  (T, a, vx, vy, dst, K, stride_row_x, stride_col_y, stride_col_dst, stride_channel_y, stride_sample_y, stride_channel_dst, stride_sample_dst, grid, block, stream, occupancy); break;
        default: printf("type not supported\n"); exit(1);
    }
}

static std::vector<float> build_src1(int K, int T) {
    std::vector<float> x((size_t) K*T);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            x[(size_t) t*K + k] = ((float)(((size_t) k*2654435761u + 97u*(unsigned) t) % 2001u) - 1000.0f) / 256.0f;
        }
    }
    return x;
}

// reference q8_1 quantization, ds-first block layout (matches quantize_row_q8_1_ref:
// d = amax/127, qs = round(x/d), s = sum(qs) scaled like the ggml reference)
static void quantize_q8_1_row(const float * x, int K, int nblocks, uint8_t * out) {
    for (int b = 0; b < nblocks; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < QK8_1; ++i) amax = std::max(amax, fabsf(x[(size_t) b*QK8_1 + i]));
        const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
        const float inv = amax == 0.0f ? 0.0f : 127.0f / amax;
        int sum = 0;
        for (int i = 0; i < QK8_1; ++i) {
            const int v = (int) roundf(x[(size_t) b*QK8_1 + i] * inv);
            out[(size_t) b*36 + 4 + i] = (uint8_t) v;
            sum += v;
        }
        const ggml_half2 ds = { ggml_fp32_to_fp16(d), ggml_fp32_to_fp16((float) sum) };
        memcpy(out + (size_t) b*36, &ds, 4);
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        printf("usage: %s <gguf> <type-name> <T> <arm[,arm...]> [niter=30] [reps=4] [--occupancy]\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    const char * type_name = argv[2];
    const int    T         = atoi(argv[3]);
    int    niter     = argc > 5 ? atoi(argv[5]) : 30;
    int    reps      = argc > 6 ? atoi(argv[6]) : 4;
    bool   occupancy = false;
    for (int i = 5; i < argc; ++i) {
        if (strcmp(argv[i], "--occupancy") == 0) occupancy = true;
    }
    if (occupancy) { niter = 1; reps = 1; }

    const TypeInfo * ti = nullptr;
    for (const auto & t : g_types) {
        if (strcmp(t.name, type_name) == 0) { ti = &t; break; }
    }
    if (!ti) { printf("unknown type %s\n", type_name); return 1; }
    if (T != 2 && T != 4) { printf("T must be 2 or 4\n"); return 1; }

    std::vector<ArmSpec> arms;
    {
        char * dup = strdup(argv[4]);
        for (char * tok = strtok(dup, ","); tok; tok = strtok(nullptr, ",")) {
            ArmSpec a;
            a.name = std::string(tok);
            arms.push_back(a);
        }
        free(dup);
    }
    if (arms.empty() || arms[0].name != "base") {
        printf("first arm must be base (oracle reference)\n");
        return 1;
    }

    HIP_CHECK(hipSetDevice(0));
    hipStream_t stream = nullptr;

    // real weight bytes (offset-delta law)
    const int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { printf("open gguf failed\n"); return 1; }
    const long maxend = 12025224992L + 50135040L;
    void * mm = mmap(nullptr, maxend, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm == MAP_FAILED) { printf("mmap failed\n"); return 1; }

    const int K = ti->K;
    const int N = ti->N;
    const int stride_row_x = K / QK_K;               // x blocks per row (20 at K = 5120)
    const int ne10_padded  = (int) GGML_PAD(K, MATRIX_ROW_PADDING);
    const int stride_col_y = ne10_padded / QK8_1;    // 160 at K = 5120

    void * vx = nullptr;
    HIP_CHECK(hipMalloc(&vx, ti->len));
    HIP_CHECK(hipMemcpy(vx, (const uint8_t *) mm + ti->off, ti->len, hipMemcpyHostToDevice));

    std::vector<float> src1 = build_src1(K, T);
    std::vector<uint8_t> vy_host((size_t) T*stride_col_y*36, 0);
    for (int t = 0; t < T; ++t) {
        quantize_q8_1_row(src1.data() + (size_t) t*K, K, stride_col_y,
                          vy_host.data() + (size_t) t*stride_col_y*36);
    }
    block_q8_1 * vy = nullptr;
    HIP_CHECK(hipMalloc((void **) &vy, vy_host.size()));
    HIP_CHECK(hipMemcpy(vy, vy_host.data(), vy_host.size(), hipMemcpyHostToDevice));

    float * dst = nullptr;
    HIP_CHECK(hipMalloc((void **) &dst, (size_t) N*T*sizeof(float)));

    dim3 grid((unsigned)((N + 1) / 2), 1, 1);
    dim3 block(64, 2, 1);

    printf("bench_mmvq_real: %s K=%d N=%d T=%d niter=%d reps=%d arms=%s\n",
           ti->name, K, N, T, niter, reps, argv[4]);

    if (occupancy) {
        for (const auto & a : arms) {
            dispatch_type(ti->type, T, a, vx, vy, dst, K, stride_row_x, stride_col_y,
                          N, (uint32_t) T*stride_col_y, (uint32_t) T*stride_col_y,
                          (uint32_t) N*T, (uint32_t) N*T, grid, block, stream, true);
        }
        printf("occupancy dump done\n");
        return 0;
    }

    // oracle reference from the first (base) arm
    std::vector<float> base_ref((size_t) N*T);
    std::vector<float> arm_out((size_t) N*T);
    {
        const ArmSpec & a0 = arms[0];
        dispatch_type(ti->type, T, a0, vx, vy, dst, K, stride_row_x, stride_col_y,
                      N, (uint32_t) T*stride_col_y, (uint32_t) T*stride_col_y,
                      (uint32_t) N*T, (uint32_t) N*T, grid, block, stream, false);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipMemcpy(base_ref.data(), dst, base_ref.size()*sizeof(float), hipMemcpyDeviceToHost));
    }

    // process-level clock warmup: kill the power-state ramp transient
    for (int w = 0; w < 120; ++w) {
        dispatch_type(ti->type, T, arms[0], vx, vy, dst, K, stride_row_x, stride_col_y,
                      N, (uint32_t) T*stride_col_y, (uint32_t) T*stride_col_y,
                      (uint32_t) N*T, (uint32_t) N*T, grid, block, stream, false);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));

    bool all_exact = true;
    for (size_t ai = 0; ai < arms.size(); ++ai) {
        const ArmSpec & a = arms[ai];
        const bool is_ref = ai == 0;

        if (!is_ref) {
            // oracle first: one launch, full dst memcmp vs base
            dispatch_type(ti->type, T, a, vx, vy, dst, K, stride_row_x, stride_col_y,
                          N, (uint32_t) T*stride_col_y, (uint32_t) T*stride_col_y,
                          (uint32_t) N*T, (uint32_t) N*T, grid, block, stream, false);
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipMemcpy(arm_out.data(), dst, arm_out.size()*sizeof(float), hipMemcpyDeviceToHost));
            const bool exact = memcmp(base_ref.data(), arm_out.data(), base_ref.size()*sizeof(float)) == 0;
            printf("%-14s ORACLE %s\n", a.name.c_str(), exact ? "BITEXACT" : "FAIL");
            if (!exact) {
                size_t nmis = 0;
                size_t by_tok[8] = {0}, by_par[2] = {0}, by_cta[64] = {0};
                double max_rel = 0.0;
                for (size_t e = 0; e < base_ref.size(); ++e) {
                    if (memcmp(base_ref.data() + e, arm_out.data() + e, sizeof(float)) != 0) {
                        const size_t tok = e / N, row = e % N;
                        by_tok[tok < 8 ? tok : 7]++;
                        by_par[row & 1]++;
                        by_cta[(row/2) & 63]++;
                        const double rel = fabs((base_ref[e] - arm_out[e]) / base_ref[e]);
                        if (rel > max_rel) max_rel = rel;
                        if (nmis < 4) {
                            printf("  mismatch[%zu]: row %zu tok %zu: base %.9g arm %.9g rel %.3g\n",
                                   e, row, tok, base_ref[e], arm_out[e], rel);
                        }
                        ++nmis;
                    }
                }
                printf("  total: %zu/%zu max_rel %.3g | by_tok:", nmis, base_ref.size(), max_rel);
                for (int t = 0; t < (int) T; ++t) printf(" %zu", by_tok[t]);
                printf(" | by_rowpar: %zu %zu | by_cta16:", by_par[0], by_par[1]);
                for (int c = 0; c < 16; ++c) {
                    size_t s2 = 0; for (int c2 = c; c2 < 64; c2 += 16) s2 += by_cta[c2];
                    printf(" %zu", s2);
                }
                printf("\n");
            }
            if (!exact) all_exact = false;
        }

        // timing
        std::vector<float> us;
        for (int r = 0; r < reps; ++r) {
            for (int w = 0; w < 3; ++w) {
                dispatch_type(ti->type, T, a, vx, vy, dst, K, stride_row_x, stride_col_y,
                              N, (uint32_t) T*stride_col_y, (uint32_t) T*stride_col_y,
                              (uint32_t) N*T, (uint32_t) N*T, grid, block, stream, false);
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipEventRecord(ev0, stream));
            for (int i = 0; i < niter; ++i) {
                dispatch_type(ti->type, T, a, vx, vy, dst, K, stride_row_x, stride_col_y,
                              N, (uint32_t) T*stride_col_y, (uint32_t) T*stride_col_y,
                              (uint32_t) N*T, (uint32_t) N*T, grid, block, stream, false);
            }
            HIP_CHECK(hipEventRecord(ev1, stream));
            HIP_CHECK(hipEventSynchronize(ev1));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
            us.push_back(ms * 1000.0f / niter);
        }
        std::vector<float> sorted = us;
        std::sort(sorted.begin(), sorted.end());
        const float med = sorted[sorted.size()/2];
        const float spread = (sorted.back() - sorted.front()) / med * 100.0f;
        printf("%-14s us: ", a.name.c_str());
        for (float v : us) printf("%.1f ", v);
        printf("| med %.1f spread %.2f%%%s\n", med, spread, spread <= 1.05f ? "" : " SPREAD-FAIL");
    }

    printf("oracle: %s\n", all_exact ? "ALL BITEXACT" : "DEFECT - do not use timing");
    return all_exact ? 0 : 1;
}
