// REAL-kernel flash-attn instrument (W11 attention desk): instantiates the
// actual flash_attn_tile template from ggml/src/ggml-cuda/fattn-tile.cu (this
// tree, included verbatim below) and drives it through the REAL launch_fattn
// host chain (F16 KV pool dequant + parallel-block scan + combine) at the
// exact served gfx900 decode geometry from the TP4 kernel census:
//
//   DKQ=DV=256, ncols1=4, ncols2=2 (base), T=4 Q rows, 6 Q heads / 1 KV head
//   per die (gqa_ratio 6), ne11 = 7168 (q4_0 KV, padded), mask on,
//   scale = 1/sqrt(256), no sinks, no KV_max scan (Q->ne[1] < 1024).
//
// KV bytes: real GGUF bytes with the q4_0 block scale sanitized to a fixed
// positive half (the KV cache is runtime q4_0, not GGUF-resident; raw GGUF
// scales can be NaN). Q/mask: deterministic f32/f16 patterns.
//
// Arms (per process, oracle-gated; launch_fattn<DKQ, ncols1, ncols2>):
//   base   ncols2=2  - the shipped instance  flash_attn_tile<256,256,4,2,false>
//   mid3   ncols2=3  - GGML_CUDA_FATTN_TILE_GQA_WIDE arm, 2 z-blocks
//   wide6  ncols2=6  - GGML_CUDA_FATTN_TILE_GQA_WIDE arm, 1 z-block
//   basedup          - base again, in-run determinism control
// Every arm after base is device-copied and memcmp'd against base BEFORE any
// timing counts; mismatches are counted and max ulp-class diff reported.
//
// The served env entry (ggml_cuda_flash_attn_ext_tile -> switch_ncols2 ->
// GQA_WIDE arm) is checked in --served-entry mode: run once with the env set
// (INFO line must appear) and once unset (negative control).
//
// build (flags mirrored from bench_mmvq_real / build-hip):
//   /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP \
//     -DGGML_BACKEND_BUILD -DGGML_SHARED -I ggml/include -I ggml/src \
//     -I ggml/src/ggml-cuda docs/amd-port/tests/bench_attn_real.cu \
//     -o /tmp/bench_attn_real -L <build>/bin -lggml-hip -lggml-base -lamdhip64 \
//     -Wl,-rpath,<abs-build>/bin -Wl,-rpath,/opt/rocm-6.2.0/lib
// run (die 3 only, after check-and-hold of /tmp/campaign_gpu_boot.lock):
//   HIP_VISIBLE_DEVICES=3 /tmp/bench_attn_real <gguf> [niter=30] [reps=5] [--served-entry] [--occupancy]
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-cuda/fattn-tile.cu"

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

static constexpr int   DKQ     = 256;
static constexpr int   DV      = 256;
static constexpr int   T       = 4;      // Q rows per die (decode verify)
static constexpr int   HEADS_Q = 6;      // per die (24/4)
static constexpr int   HEADS_KV= 1;      // per die (4/4)
static constexpr int   NE11    = 7168;   // padded KV length at serve
static constexpr float SCALE   = 0.0625f;

// q4_0 row: 256 el = 8 blocks x 18 B
static constexpr long KV_ROW_BYTES = (DKQ / QK4_0) * sizeof(block_q4_0);

enum ArmKind { ARM_BASE, ARM_MID3, ARM_WIDE6, ARM_BASEDUP };

static const char * arm_name(ArmKind k) {
    switch (k) {
        case ARM_BASE:    return "base";
        case ARM_MID3:    return "mid3";
        case ARM_WIDE6:   return "wide6";
        case ARM_BASEDUP: return "basedup";
    }
    return "?";
}

template <int ncols1, int ncols2>
static void launch_arm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int nthreads  = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2, cc);
    const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2, cc);
    launch_fattn<DV, ncols1, ncols2>(ctx, dst,
        flash_attn_tile<DKQ, DV, ncols1, ncols2, false>,
        nthreads / 32, 0, nbatch_fa, true, true, false, 32);
    HIP_CHECK(hipGetLastError());
}

static void run_arm(ArmKind k, ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (k) {
        case ARM_BASE:    launch_arm<4, 2>(ctx, dst); break;
        case ARM_MID3:    launch_arm<4, 3>(ctx, dst); break;
        case ARM_WIDE6:   launch_arm<4, 6>(ctx, dst); break;
        case ARM_BASEDUP: launch_arm<4, 2>(ctx, dst); break;
    }
}

static void occ_arm(ArmKind k) {
    const void * kp = nullptr;
    int nthreads = 0; size_t shm = 0; const char * nm = arm_name(k);
    if (k == ARM_BASE || k == ARM_BASEDUP) {
        kp = (const void *) &flash_attn_tile<DKQ, DV, 4, 2, false>; nthreads = 256;
        shm = (size_t) (8*512 + 64*(64+4)*4 + 8*64*2);
    } else if (k == ARM_MID3) {
        kp = (const void *) &flash_attn_tile<DKQ, DV, 4, 3, false>; nthreads = 192;
        shm = (size_t) (12*512 + 64*(64+4)*4 + 12*64*2);
    } else {
        kp = (const void *) &flash_attn_tile<DKQ, DV, 4, 6, false>; nthreads = 384;
        shm = (size_t) (24*512 + 64*(64+4)*4 + 24*64*2);
    }
    hipFuncAttributes attr{};
    HIP_CHECK(hipFuncGetAttributes(&attr, kp));
    int blocks = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kp, nthreads, shm));
    printf("OCC %s: numRegs=%d static_shared=%zu threads=%d ctas_per_cu=%d\n",
           nm, attr.numRegs, attr.sharedSizeBytes, nthreads, blocks);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s <gguf> [niter=30] [reps=5] [--served-entry] [--occupancy]\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    int  niter     = argc > 2 ? atoi(argv[2]) : 30;
    int  reps      = argc > 3 ? atoi(argv[3]) : 9;
    bool served    = false;
    bool occupancy = false;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--served-entry") == 0) served = true;
        if (strcmp(argv[i], "--occupancy") == 0)    occupancy = true;
    }
    if (occupancy) {
        occ_arm(ARM_BASE); occ_arm(ARM_MID3); occ_arm(ARM_WIDE6);
        return 0;
    }

    // backend + device info init (ggml_cuda_info() used by launch_fattn)
    ggml_backend_t bak = ggml_backend_cuda_init(0);
    if (!bak) { printf("ggml_backend_cuda_init failed\n"); return 1; }
    ggml_backend_cuda_context ctx(0);

    // read KV bytes from the GGUF (token_embd region, offset-delta law) and
    // sanitize the q4_0 block scales to a fixed positive half
    const long kv_bytes = (long) NE11 * HEADS_KV * KV_ROW_BYTES;
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { printf("cannot open gguf\n"); return 1; }
    const long src_off = 4637536L + 100000000L; // inside token_embd q6_K bytes
    std::vector<uint8_t> kv_host((size_t) kv_bytes);
    if (pread(fd, kv_host.data(), kv_bytes, src_off) != (ssize_t) kv_bytes) {
        printf("gguf pread failed\n"); return 1;
    }
    const uint16_t d_san = ggml_fp32_to_fp16(0.03125f);
    for (long b = 0; b < kv_bytes / 18; ++b) {
        memcpy(kv_host.data() + b*18, &d_san, 2);
    }

    void * kv_dev = nullptr;
    HIP_CHECK(hipMalloc(&kv_dev, kv_bytes));
    HIP_CHECK(hipMemcpy(kv_dev, kv_host.data(), kv_bytes, hipMemcpyHostToDevice));
    void * v_dev = nullptr;
    HIP_CHECK(hipMalloc(&v_dev, kv_bytes));
    HIP_CHECK(hipMemcpy(v_dev, kv_host.data(), kv_bytes, hipMemcpyHostToDevice));

    // Q f32 [256 x 4 x 6], mask f16 [7168 x 4] (zeros = full attention)
    std::vector<float> q_host((size_t) DKQ*T*HEADS_Q);
    for (size_t i = 0; i < q_host.size(); ++i) {
        q_host[i] = ((float)((i*2654435761u) % 2003u) - 1001.0f) / 512.0f;
    }
    void * q_dev = nullptr; HIP_CHECK(hipMalloc(&q_dev, q_host.size()*4));
    HIP_CHECK(hipMemcpy(q_dev, q_host.data(), q_host.size()*4, hipMemcpyHostToDevice));
    void * m_dev = nullptr; HIP_CHECK(hipMalloc(&m_dev, (size_t) NE11*T*2));
    HIP_CHECK(hipMemset(m_dev, 0, (size_t) NE11*T*2));

    // dst f32 [256 x 4 x 6] + slack for the F16 KV pools that launch_fattn
    // appends after dst (fattn-common.cuh get_f16_extra_data)
    const size_t dst_bytes  = (size_t) DKQ*T*HEADS_Q*4;
    const size_t slack      = (size_t) 2 * kv_bytes * 4 + (1<<20);
    void * dst_dev = nullptr; HIP_CHECK(hipMalloc(&dst_dev, dst_bytes + slack));

    // tensor plumbing via real ggml helpers (no_alloc: data pointers set by hand)
    ggml_init_params ip = { /*mem_size*/ 4u<<20, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * gctx = ggml_init(ip);
    ggml_tensor * q_t   = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,  DKQ, T, HEADS_Q, 1);
    ggml_tensor * k_t   = ggml_new_tensor_4d(gctx, GGML_TYPE_Q4_0, DKQ, NE11, HEADS_KV, 1);
    ggml_tensor * v_t   = ggml_new_tensor_4d(gctx, GGML_TYPE_Q4_0, DKQ, NE11, HEADS_KV, 1);
    ggml_tensor * m_t   = ggml_new_tensor_4d(gctx, GGML_TYPE_F16,  NE11, T, 1, 1);
    ggml_tensor * dst_t = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,  DKQ, T, HEADS_Q, 1);
    q_t->data = q_dev; k_t->data = kv_dev; v_t->data = v_dev; m_t->data = m_dev;
    dst_t->data = dst_dev;
    memcpy(dst_t->op_params + 0, &SCALE, sizeof(float));
    const float max_bias = 0.0f;      memcpy(dst_t->op_params + 1, &max_bias, sizeof(float));
    const float softcap  = 0.0f;      memcpy(dst_t->op_params + 2, &softcap,  sizeof(float));
    dst_t->src[0] = q_t; dst_t->src[1] = k_t; dst_t->src[2] = v_t; dst_t->src[3] = m_t; dst_t->src[4] = nullptr;
    dst_t->op = GGML_OP_FLASH_ATTN_EXT; // asserted by get_f16_extra_data

    if (served) {
        // full served entry: exercises the fattn.cu TILE selection +
        // launch_fattn_tile_switch_ncols2 (+ GQA_WIDE arm when env is set)
        ggml_cuda_flash_attn_ext_tile(ctx, dst_t);
        HIP_CHECK(hipDeviceSynchronize());
        printf("SERVED-ENTRY done\n");
        ggml_free(gctx); return 0;
    }

    const std::vector<ArmKind> arms = { ARM_BASE, ARM_MID3, ARM_WIDE6, ARM_BASEDUP };

    // warmup + oracle
    std::vector<std::vector<uint8_t>> out(arms.size());
    std::vector<int> mism(arms.size(), 0);
    double max_rel = 0.0;
    for (size_t a = 0; a < arms.size(); ++a) {
        run_arm(arms[a], ctx, dst_t);
        HIP_CHECK(hipDeviceSynchronize());
        out[a].resize(dst_bytes);
        HIP_CHECK(hipMemcpy(out[a].data(), dst_dev, dst_bytes, hipMemcpyDeviceToHost));
        if (a > 0 && (arms[a] != ARM_BASEDUP)) {
            const float * base = (const float *) out[0].data();
            const float * arm  = (const float *) out[a].data();
            for (size_t i = 0; i < dst_bytes/4; ++i) {
                if (base[i] != arm[i]) {
                    mism[a]++;
                    const float rel = fabsf(base[i] - arm[i]) / (fabsf(base[i]) + 1e-30f);
                    if (rel > max_rel) max_rel = rel;
                }
            }
        }
        if (arms[a] == ARM_BASEDUP && out[a] != out[0]) {
            printf("ORACLE FAIL: basedup != base (nondeterministic base)\n");
            return 1;
        }
    }
    for (size_t a = 1; a < arms.size(); ++a) {
        if (arms[a] == ARM_BASEDUP) continue;
        printf("ORACLE %s vs base: %s (%d/%zu els differ, max rel %.3e)\n",
               arm_name(arms[a]), mism[a] == 0 ? "BIT-EXACT" : "DUST",
               mism[a], dst_bytes/4, max_rel);
    }

    // extra warmup (2 full 4-arm passes) so every arm sees ramped clocks
    for (int w = 0; w < 2; ++w) {
        for (size_t a = 0; a < arms.size(); ++a) {
            run_arm(arms[a], ctx, dst_t);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    // timing: reps x (niter back-to-back launches, hip events), arms interleaved;
    // events ride ctx.stream() so they bracket the real launch stream
    std::vector<std::vector<double>> t(arms.size());
    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0)); HIP_CHECK(hipEventCreate(&ev1));
    hipStream_t st = ctx.stream();
    for (int r = 0; r < reps; ++r) {
        for (size_t a = 0; a < arms.size(); ++a) {
            HIP_CHECK(hipEventRecord(ev0, st));
            for (int i = 0; i < niter; ++i) {
                run_arm(arms[a], ctx, dst_t);
            }
            HIP_CHECK(hipEventRecord(ev1, st));
            HIP_CHECK(hipEventSynchronize(ev1));
            float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
            t[a].push_back(ms * 1000.0 / niter);
        }
    }
    printf("\nper-launch us (niter=%d, reps=%d), served geometry T=%d gqa=6 ne11=%d q4_0 KV:\n",
           niter, reps, T, NE11);
    for (size_t a = 0; a < arms.size(); ++a) {
        std::vector<double> s = t[a];
        std::sort(s.begin(), s.end());
        const double med = s[s.size()/2];
        const double spread = (s.back() - s.front()) / med * 100.0;
        printf("  %-8s med=%8.1f us  min=%8.1f  max=%8.1f  spread=%.2f%%  vs base %+7.2f%%%s\n",
               arm_name(arms[a]), med, s.front(), s.back(), spread,
               (med - t[0][t[0].size()/2]) / t[0][t[0].size()/2] * 100.0,
               spread <= 1.05 ? "" : "  SPREAD-FAIL (>1.05%)");
    }
    ggml_free(gctx);
    return 0;
}
