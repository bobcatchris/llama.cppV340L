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
//   direct           - GGML_CUDA_FATTN_TILE_Q40_DIRECT arm: same <4,2>
//                      instance shape but the kernel dequantizes the q4_0 KV
//                      blocks in-kernel (launch_fattn need_f16 = false, false:
//                      no pool, no dequant launches)
//   basedup          - base again, in-run determinism control
// Every arm after base is device-copied and memcmp'd against base BEFORE any
// timing counts; mismatches are counted and max ulp-class diff reported.
//
// W16 depth sweep: --depth N overrides ne11 (default 7168; N must be a
// multiple of 256 so use_gqa_opt and the no-oob scan stay valid).
//
// W27 prefill sweep: --M N overrides T (default 4 decode rows; 512 = prefill
// ubatch, 268 = tail ubatch, 33x8+4) so Q/mask/dst are shaped at the prefill
// instance. --prefill selects the prefill arm set: basep = the f16-pool
// prefill instance flash_attn_tile<256,256,8,2> (served today), v11p =
// <256,256,8,2,.,11> (GGML_CUDA_FATTN_TILE_Q40_PREFILL arm, need_f16=false),
// basedupp = basep again (determinism control).
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
#include <chrono>
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
static long           g_T     = 4;      // Q rows per die (--M overrides; 4 = decode verify)
static constexpr int   HEADS_Q = 6;      // per die (24/4)
static constexpr int   HEADS_KV= 1;      // per die (4/4)
static long           g_ne11  = 7168;   // padded KV length at serve (--depth overrides)
static long           g_used  = 0;      // attended KV count; >0 fills mask cols >= used with -inf (padded n_kv causal boundary)
static constexpr float SCALE   = 0.0625f;

// q4_0 row: 256 el = 8 blocks x 18 B
static constexpr long KV_ROW_BYTES = (DKQ / QK4_0) * sizeof(block_q4_0);

enum ArmKind { ARM_BASE, ARM_MID3, ARM_WIDE6, ARM_DIRECT,
               ARM_V1, ARM_V2, ARM_V3, ARM_V4, ARM_V6, ARM_V7, ARM_V8, ARM_V9,
               ARM_V10, ARM_V11, ARM_BASEDUP,
               ARM_BASEP, ARM_V11P, ARM_BASEDUPP };

static const char * arm_name(ArmKind k) {
    switch (k) {
        case ARM_BASE:    return "base";
        case ARM_MID3:    return "mid3";
        case ARM_WIDE6:   return "wide6";
        case ARM_DIRECT:  return "direct";
        case ARM_V1:      return "v1_ctl";  // q40 as-of-W16 (negative control)
        case ARM_V2:      return "v2_c1c";  // C1c: noinline dequant
        case ARM_V3:      return "v3_c1d";  // C1d: volatile LDS reads in KQ stage
        case ARM_V4:      return "v4_nlm";  // noinline mad chain
        case ARM_V6:      return "v6_hk";   // C3b: half-width K chunks
        case ARM_V7:      return "v7_lb";   // C1a: explicit launch bounds (256,1)
        case ARM_V8:      return "v8_u1";   // unroll-1 dot chain
        case ARM_V9:      return "v9_optno";// optnone diagnostic
        case ARM_V10:     return "v10_st16";// 16B shared-tile stores (f16 granule)
        case ARM_V11:     return "v11_stsc";// scalar half2 shared-tile stores
        case ARM_BASEDUP: return "basedup";
        case ARM_BASEP:   return "basep";   // f16-pool prefill instance <8,2>
        case ARM_V11P:    return "v11p";    // q4_0-direct prefill instance <8,2,11>
        case ARM_BASEDUPP:return "basedupp";
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

// q4_0-direct arm: identical geometry/config lookup as base, but the kernel
// reads the q4_0 KV tensor in place and the pool conversion is skipped.
// VAR selects the codegen-workaround variant (1 = as-of-W16 control).
template <int VAR>
static void launch_arm_q40(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int nthreads  = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, 8, cc);
    const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, 8, cc);
    launch_fattn<DV, 4, 2>(ctx, dst,
        flash_attn_tile<DKQ, DV, 4, 2, false, VAR>,
        nthreads / 32, 0, nbatch_fa, false, false, false, 32);
    HIP_CHECK(hipGetLastError());
}

static void launch_arm_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    launch_arm_q40<1>(ctx, dst);
}

// q4_0-direct prefill arm (W27): the served prefill instance <8,2>
// (cols_per_block=16 config row) with in-kernel q4_0 KV and the pool
// conversion skipped, mirroring the decode direct arm's need_f16 = false.
template <int VAR>
static void launch_arm_q40p(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int nthreads  = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, 16, cc);
    const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, 16, cc);
    launch_fattn<DV, 8, 2>(ctx, dst,
        flash_attn_tile<DKQ, DV, 8, 2, false, VAR>,
        nthreads / 32, 0, nbatch_fa, false, false, false, 32);
    HIP_CHECK(hipGetLastError());
}

static void run_arm(ArmKind k, ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (k) {
        case ARM_BASE:    launch_arm<4, 2>(ctx, dst); break;
        case ARM_MID3:    launch_arm<4, 3>(ctx, dst); break;
        case ARM_WIDE6:   launch_arm<4, 6>(ctx, dst); break;
        case ARM_DIRECT:  launch_arm_direct(ctx, dst); break;
        case ARM_V1:      launch_arm_q40<1>(ctx, dst); break;
        case ARM_V2:      launch_arm_q40<2>(ctx, dst); break;
        case ARM_V3:      launch_arm_q40<3>(ctx, dst); break;
        case ARM_V4:      launch_arm_q40<4>(ctx, dst); break;
        case ARM_V6:      launch_arm_q40<6>(ctx, dst); break;
        case ARM_V7:      launch_arm_q40<7>(ctx, dst); break;
        case ARM_V8:      launch_arm_q40<8>(ctx, dst); break;
        case ARM_V9:      launch_arm_q40<9>(ctx, dst); break;
        case ARM_V10:     launch_arm_q40<10>(ctx, dst); break;
        case ARM_V11:     launch_arm_q40<11>(ctx, dst); break;
        case ARM_BASEDUP: launch_arm<4, 2>(ctx, dst); break;
        case ARM_BASEP:   launch_arm<8, 2>(ctx, dst); break;
        case ARM_V11P:    launch_arm_q40p<11>(ctx, dst); break;
        case ARM_BASEDUPP:launch_arm<8, 2>(ctx, dst); break;
    }
}

// served instance with padded DYNAMIC shared memory: pads the launch's total
// LDS footprint (static + dynamic) to a target without touching the kernel
template <int ncols1, int ncols2>
static void launch_arm_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst, size_t pad_bytes) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int nthreads  = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2, cc);
    const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2, cc);
    launch_fattn<DV, ncols1, ncols2>(ctx, dst,
        flash_attn_tile<DKQ, DV, ncols1, ncols2, false>,
        nthreads / 32, pad_bytes, nbatch_fa, true, true, false, 32);
    HIP_CHECK(hipGetLastError());
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

// W14 wide-launch desk probe kernels: minimal instances that isolate the
// ~210 us wide-launch GPU idle gap from the fattn kernel complexity.
__global__ void widelaunch_noop_kernel(int * p) {
    if (threadIdx.x == 0 && p != nullptr) {
        *p = 1;
    }
}

template <int LDS_BYTES, bool BURN_VGPR = false>
__global__ void widelaunch_lds_kernel(float * out) {
    __shared__ float buf[LDS_BYTES/4];
    const int tid = threadIdx.x;
    for (int i = tid; i < LDS_BYTES/4; i += blockDim.x) {
        buf[i] = 0.0f;
    }
    __syncthreads();
    uint32_t sink = 0;
    if (BURN_VGPR) {
        // force >=128 arch vgprs: wrap-around circular dependencies keep the
        // whole array live; the magic-compare guard defeats constant folding
        float acc[128];
        #pragma unroll
        for (int i = 0; i < 128; ++i) {
            acc[i] = (float) (i ^ tid) + 1.0f;
        }
        #pragma unroll
        for (int r2 = 0; r2 < 2; ++r2) {
            #pragma unroll
            for (int i = 0; i < 128; ++i) {
                acc[i] = fmaf(acc[i], acc[(i + 1 + r2*64) & 127], 1.0f)
                         - acc[(i + 32) & 127] * 0.5f;
            }
        }
        float t = 0.0f;
        #pragma unroll
        for (int i = 0; i < 128; ++i) {
            t += acc[i];
        }
        sink = (uint32_t) t;
    }
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < LDS_BYTES/4; i += 128) {
            s += buf[i];
        }
        out[blockIdx.y] = s;
        if (sink == 0xdeadbeefu) {
            out[blockIdx.y] = 2.0f;
        }
    }
}

// forced scratch (local-memory spill) kernels. LIGHT variant: 256 B/thread
// spill, near-zero traffic, isolates start latency. HEAVY variant: ~1440
// B/thread (FA<4,6> class) - execution-dominated, kept for reference.
__global__ void widelaunch_spill_kernel(float * out) {
    float r[360];
    const int tid = threadIdx.x;
    for (int i = 0; i < 360; ++i) {
        r[i] = (float) (i ^ tid);
    }
    float s = 0.0f;
    for (int i = 0; i < 360; ++i) {
        s += r[(i*7 + tid) % 360];
    }
    if (tid == 0) {
        out[blockIdx.y] = s;
    }
}

__global__ void widelaunch_spill_light_kernel(float * out) {
    float r[64];
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        r[i] = (float) (i ^ tid);
    }
    float s = 0.0f;
    for (int i = 0; i < 64; ++i) {
        s += r[(i*7 + tid) & 63];
    }
    if (tid == 0) {
        out[blockIdx.y] = s;
    }
}

// FA<4,6>-class spill (1456 B/thread local) with CHEAP access: scattered
// init forces the local allocation, linear reads keep execution tiny -
// isolates the scr threshold class (FA<4,2> 1104 clean vs FA<4,6> 1432 pays)
__global__ void widelaunch_spill_lin_kernel(float * out) {
    float r[364];
    const int tid = threadIdx.x;
    for (int i = 0; i < 364; ++i) {
        r[(i*7 + tid) % 364] = (float) (i ^ tid);
    }
    float s = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < 364; ++i) {
        s += r[i];
    }
    if (tid == 0) {
        out[blockIdx.y] = s;
    }
}

// huge-code kernels: unrolled dependent math. heavy = 123 KB code, full
// execution; light = same code volume in guarded 64-iteration chunks, only
// the first chunk runs (isolate start latency from execution).
template <int N>
__global__ void widelaunch_bigcode_kernel(float * out) {
    float a = threadIdx.x * 0.5f + 1.0f;
    float b = 0.5f;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        a = fmaf(a, b, 1.0001f);
        b = fmaf(b, a, 0.9999f);
        a = sqrtf(a) + 1e-9f;
    }
    if (a == 12345.678f) {
        out[threadIdx.x] = a + b;
    }
}

template <int CHUNKS>
__global__ void widelaunch_bigcode_light_kernel(float * out) {
    float a = threadIdx.x * 0.5f + 1.0f;
    float b = 0.5f;
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        if (c == 0 || out[0] < -1.0e30f) {
            #pragma unroll
            for (int i = 0; i < 64; ++i) {
                a = fmaf(a, b, 1.0001f);
                b = fmaf(b, a, 0.9999f);
                a = sqrtf(a) + 1e-9f;
            }
        }
    }
    if (a == 12345.678f) {
        out[threadIdx.x] = a + b;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s <gguf> [niter=30] [reps=5] [--served-entry] [--occupancy] [--probe] [--matrix] [--depth N] [--M N] [--used N] [--prefill] [--allarms]\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    int  niter     = argc > 2 ? atoi(argv[2]) : 30;
    int  reps      = argc > 3 ? atoi(argv[3]) : 9;
    bool served    = false;
    bool occupancy = false;
    bool probe     = false;
    bool matrix    = false;
    bool allarms   = false;
    bool oracle_only = false;
    bool solo        = false; // v11 winner alone (no miscompile-shaped variants in the TU)
    bool prefill     = false; // W27 prefill arm set: basep/v11p/basedupp
    const char * adj_pair = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--served-entry") == 0) served = true;
        if (strcmp(argv[i], "--occupancy") == 0)    occupancy = true;
        if (strcmp(argv[i], "--probe") == 0)        probe = true;
        if (strcmp(argv[i], "--matrix") == 0)       matrix = true;
        if (strcmp(argv[i], "--allarms") == 0)      allarms = true;
        if (strcmp(argv[i], "--oracle") == 0)       oracle_only = true;
        if (strcmp(argv[i], "--solo") == 0)         solo = true;
        if (strcmp(argv[i], "--prefill") == 0)      prefill = true;
        if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) g_ne11 = atol(argv[++i]);
        if (strcmp(argv[i], "--M") == 0 && i + 1 < argc)     g_T    = atol(argv[++i]);
        if (strcmp(argv[i], "--used") == 0 && i + 1 < argc)  g_used = atol(argv[++i]);
        if (strcmp(argv[i], "--adj") == 0 && i + 1 < argc) adj_pair = argv[++i];
    }
    if (g_ne11 % FATTN_KQ_STRIDE != 0 || g_ne11 % 64 != 0) {
        printf("depth %ld must be a multiple of 256\n", g_ne11);
        return 1;
    }
    if (occupancy) {
        occ_arm(ARM_BASE); occ_arm(ARM_MID3); occ_arm(ARM_WIDE6);
        return 0;
    }

    // backend + device info init (ggml_cuda_info() used by launch_fattn)
    ggml_backend_t bak = ggml_backend_cuda_init(0);
    if (!bak) { printf("ggml_backend_cuda_init failed\n"); return 1; }
    // heap leak on purpose: this port's ctx dtor (chunk-cache clears) aborts
    // on hipFree at exit teardown; measurements are complete before we return
    ggml_backend_cuda_context & ctx = *new ggml_backend_cuda_context(0);

    // read KV bytes from the GGUF (token_embd region, offset-delta law) and
    // sanitize the q4_0 block scales to a fixed positive half
    const long kv_bytes = g_ne11 * HEADS_KV * KV_ROW_BYTES;
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

    // Q f32 [256 x M x 6], mask f16 [ne11 x M] (zeros = full attention)
    std::vector<float> q_host((size_t) DKQ*g_T*HEADS_Q);
    for (size_t i = 0; i < q_host.size(); ++i) {
        q_host[i] = ((float)((i*2654435761u) % 2003u) - 1001.0f) / 512.0f;
    }
    void * q_dev = nullptr; HIP_CHECK(hipMalloc(&q_dev, q_host.size()*4));
    HIP_CHECK(hipMemcpy(q_dev, q_host.data(), q_host.size()*4, hipMemcpyHostToDevice));
    void * m_dev = nullptr; HIP_CHECK(hipMalloc(&m_dev, (size_t) g_ne11*g_T*2));
    if (g_used > 0) {
        // padded-n_kv causal boundary (W24 risk 3): attend only to the first
        // g_used KV positions, -inf beyond; both arms must zero the oob tail
        std::vector<uint16_t> m_host((size_t) g_ne11*g_T);
        const uint16_t ninf = 0xFC00; // -inf half
        for (long j = 0; j < g_ne11; ++j) {
            const uint16_t v = j < g_used ? ggml_fp32_to_fp16(0.0f) : ninf;
            for (long r = 0; r < g_T; ++r) {
                m_host[(size_t) r*g_ne11 + j] = v;
            }
        }
        HIP_CHECK(hipMemcpy(m_dev, m_host.data(), m_host.size()*2, hipMemcpyHostToDevice));
    } else {
        HIP_CHECK(hipMemset(m_dev, 0, (size_t) g_ne11*g_T*2));
    }

    // dst f32 [256 x M x 6] + slack for the F16 KV pools that launch_fattn
    // appends after dst (fattn-common.cuh get_f16_extra_data)
    const size_t dst_bytes  = (size_t) DKQ*g_T*HEADS_Q*4;
    const size_t slack      = (size_t) 2 * kv_bytes * 4 + (1<<20);
    void * dst_dev = nullptr; HIP_CHECK(hipMalloc(&dst_dev, dst_bytes + slack));

    // tensor plumbing via real ggml helpers (no_alloc: data pointers set by hand)
    ggml_init_params ip = { /*mem_size*/ 4u<<20, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * gctx = ggml_init(ip);
    ggml_tensor * q_t   = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,  DKQ, g_T, HEADS_Q, 1);
    ggml_tensor * k_t   = ggml_new_tensor_4d(gctx, GGML_TYPE_Q4_0, DKQ, g_ne11, HEADS_KV, 1);
    ggml_tensor * v_t   = ggml_new_tensor_4d(gctx, GGML_TYPE_Q4_0, DKQ, g_ne11, HEADS_KV, 1);
    ggml_tensor * m_t   = ggml_new_tensor_4d(gctx, GGML_TYPE_F16,  g_ne11, g_T, 1, 1);
    ggml_tensor * dst_t = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,  DKQ, g_T, HEADS_Q, 1);
    q_t->data = q_dev; k_t->data = kv_dev; v_t->data = v_dev; m_t->data = m_dev;
    dst_t->data = dst_dev;
    memcpy(dst_t->op_params + 0, &SCALE, sizeof(float));
    const float max_bias = 0.0f;      memcpy(dst_t->op_params + 1, &max_bias, sizeof(float));
    const float softcap  = 0.0f;      memcpy(dst_t->op_params + 2, &softcap,  sizeof(float));
    dst_t->src[0] = q_t; dst_t->src[1] = k_t; dst_t->src[2] = v_t; dst_t->src[3] = m_t; dst_t->src[4] = nullptr;
    dst_t->op = GGML_OP_FLASH_ATTN_EXT; // asserted by get_f16_extra_data

    // staged-oracle dump channel (dbg builds only): the kernel receives it via
    // the sinks arg; the dbg build compiles the sink bias out. Region layout
    // (float words): [0] magic, [1] variant id, KT @1024 (64x128 half2 as u32),
    // VT @12288 (32x128), KQ_acc @16384 (256 threads x 4 slots).
    float * dbg_dev = nullptr;
    bool dbg_mode = false;
#ifdef GGML_FATTN_Q40_DBG
    if (getenv("GGML_FATTN_Q40_DBG")) {
        dbg_mode = true;
        HIP_CHECK(hipMalloc(&dbg_dev, 1 << 20));
        HIP_CHECK(hipMemset(dbg_dev, 0, 1 << 20));
        ggml_tensor * dbg_t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, (1 << 20) / 4);
        dbg_t->data = dbg_dev;
        dst_t->src[4] = dbg_t;
    }
#endif

    if (probe) {
        // per-launch event timing of the same arm back-to-back (no sequence
        // context): separates intrinsic per-launch cost from the inter-
        // kernel gap the full deq,deq,FA,comb sequence shows
        const ArmKind pk = ARM_WIDE6;
        hipEvent_t pe0, pe1;
        HIP_CHECK(hipEventCreate(&pe0)); HIP_CHECK(hipEventCreate(&pe1));
        for (int w = 0; w < 20; ++w) run_arm(pk, ctx, dst_t);
        HIP_CHECK(hipDeviceSynchronize());
        for (int i = 0; i < 20; ++i) {
            HIP_CHECK(hipEventRecord(pe0, ctx.stream()));
            run_arm(pk, ctx, dst_t);
            HIP_CHECK(hipEventRecord(pe1, ctx.stream()));
            HIP_CHECK(hipEventSynchronize(pe1));
            float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, pe0, pe1));
            printf("PROBE %s launch %2d: %.1f us\n", arm_name(pk), i, ms * 1000.0f);
        }
        ggml_free(gctx);
        return 0;
    }

    if (adj_pair != nullptr) {
        // adjacency probe for rocprof decomposition: niter x ([pre step] +
        // event-bracketed fattn launch), stream drained between steps.
        // pair: ww bb wb bw (pre = that arm), nw nb (pre = tiny noop kernel),
        // idleb idlew (pre = 2 ms idle). The event print gives per-launch
        // elapsed; the rocprof trace gives per-kernel-boundary gaps.
        hipEvent_t e0, e1;
        HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
        hipStream_t st = ctx.stream();
        int * noop_dev = nullptr;
        HIP_CHECK(hipMalloc(&noop_dev, 4));
        for (int w = 0; w < 100; ++w) {
            run_arm(ARM_BASE, ctx, dst_t);
            run_arm(ARM_WIDE6, ctx, dst_t);
        }
        HIP_CHECK(hipDeviceSynchronize());
        const bool pre_w = strcmp(adj_pair, "ww") == 0 || strcmp(adj_pair, "wb") == 0;
        const bool pre_b = strcmp(adj_pair, "bw") == 0 || strcmp(adj_pair, "bb") == 0;
        const bool pre_n = strcmp(adj_pair, "nw") == 0 || strcmp(adj_pair, "nb") == 0;
        const bool pre_i = strcmp(adj_pair, "idlew") == 0 || strcmp(adj_pair, "idleb") == 0;
        const bool run_w = strcmp(adj_pair, "ww") == 0 || strcmp(adj_pair, "bw") == 0 ||
                           strcmp(adj_pair, "nw") == 0 || strcmp(adj_pair, "idlew") == 0;
        const ArmKind k  = run_w ? ARM_WIDE6 : ARM_BASE;
        for (int i = 0; i < niter; ++i) {
            if (pre_w) { run_arm(ARM_WIDE6, ctx, dst_t); HIP_CHECK(hipDeviceSynchronize()); }
            if (pre_b) { run_arm(ARM_BASE, ctx, dst_t); HIP_CHECK(hipDeviceSynchronize()); }
            if (pre_n) {
                widelaunch_noop_kernel<<<1, 64, 0, st>>>(noop_dev);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipEventRecord(e0, st));
            } else {
                if (pre_i) { HIP_CHECK(hipDeviceSynchronize()); usleep(2000); }
                HIP_CHECK(hipEventRecord(e0, st));
            }
            run_arm(k, ctx, dst_t);
            HIP_CHECK(hipEventRecord(e1, st));
            HIP_CHECK(hipEventSynchronize(e1));
            float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
            printf("ADJ %s launch %2d: %.1f us\n", adj_pair, i, ms * 1000.0f);
        }
        HIP_CHECK(hipFree(noop_dev));
        ggml_free(gctx);
        return 0;
    }

    if (matrix) {
        // W14 wide-launch desk: probe matrix isolating WHEN the ~210 us
        // pre-fattn GPU idle gap appears. Every scenario times per-launch
        // with device-event brackets on ctx.stream() (probe discipline);
        // kernel sums (rocprof, W11): base 751.5 mid3 727.2 wide6 585.8 us,
        // so elapsed - kernel_sum = launch overhead incl. the gap.
        hipEvent_t e0, e1;
        HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
        hipStream_t st = ctx.stream();

        auto launch_timed = [&](ArmKind k) -> double {
            HIP_CHECK(hipEventRecord(e0, st));
            run_arm(k, ctx, dst_t);
            HIP_CHECK(hipEventRecord(e1, st));
            HIP_CHECK(hipEventSynchronize(e1));
            float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
            return ms * 1000.0;
        };
        auto report = [](const char * name, std::vector<double> v) {
            std::sort(v.begin(), v.end());
            const double med = v[v.size()/2];
            const double spread = (v.back() - v.front()) / med * 100.0;
            printf("MATRIX %-18s n=%2d med=%8.1f us  min=%8.1f  max=%8.1f  spread=%.2f%%%s\n",
                   name, (int) v.size(), med, v.front(), v.back(), spread,
                   spread <= 1.05 ? "" : "  SPREAD>1.05");
        };
        auto run_scenario = [&](const char * name, int n, ArmKind k) {
            std::vector<double> v;
            for (int i = 0; i < n; ++i) v.push_back(launch_timed(k));
            report(name, v);
        };

        // warm all instances into steady state
        for (int w = 0; w < 300; ++w) {
            run_arm(ARM_BASE, ctx, dst_t);
            run_arm(ARM_MID3, ctx, dst_t);
            run_arm(ARM_WIDE6, ctx, dst_t);
        }
        HIP_CHECK(hipDeviceSynchronize());

        // (a) same-kernel back-to-back
        run_scenario("a_solo_base",    niter, ARM_BASE);
        run_scenario("a_solo_mid3",    niter, ARM_MID3);
        run_scenario("a_solo_wide6",   niter, ARM_WIDE6);

        // (b) alternate two kernels, per-launch brackets
        std::vector<double> vb, vw;
        for (int i = 0; i < niter; ++i) {
            vb.push_back(launch_timed(ARM_BASE));
            vw.push_back(launch_timed(ARM_WIDE6));
        }
        report("b_alt_base", vb);
        report("b_alt_wide6", vw);

        // (c) wide-then-served: does a completed wide launch poison the next
        // base launch (immediately after sync)?
        std::vector<double> vpb;
        for (int i = 0; i < niter; ++i) {
            run_arm(ARM_WIDE6, ctx, dst_t);
            HIP_CHECK(hipDeviceSynchronize());
            vpb.push_back(launch_timed(ARM_BASE));
        }
        report("c_after_wide_base", vpb);

        // (d) served-after-a-small-kernel (NCCL-boundary adjacency class):
        // tiny kernel enqueued immediately before the bracketed fattn launch
        int * noop_dev = nullptr;
        HIP_CHECK(hipMalloc(&noop_dev, 4));
        {
            std::vector<double> vwb, vww;
            for (int i = 0; i < niter; ++i) {
                widelaunch_noop_kernel<<<1, 64, 0, st>>>(noop_dev);
                HIP_CHECK(hipGetLastError());
                vwb.push_back(launch_timed(ARM_BASE));
                widelaunch_noop_kernel<<<1, 64, 0, st>>>(noop_dev);
                HIP_CHECK(hipGetLastError());
                vww.push_back(launch_timed(ARM_WIDE6));
            }
            report("d_noop_base", vwb);
            report("d_noop_wide6", vww);
        }

        // sparse launches (2 ms host sleep + full drain before each launch):
        // power-state / clock-ramp discriminator
        {
            std::vector<double> vwb, vww;
            for (int i = 0; i < niter; ++i) {
                HIP_CHECK(hipDeviceSynchronize());
                usleep(2000);
                vwb.push_back(launch_timed(ARM_BASE));
                HIP_CHECK(hipDeviceSynchronize());
                usleep(2000);
                vww.push_back(launch_timed(ARM_WIDE6));
            }
            report("sparse_base", vwb);
            report("sparse_wide6", vww);
        }

        // double launch inside one bracket (second launch overlaps first):
        // fixed per-launch cost vs execution overlap discriminator
        {
            std::vector<double> vd, vd2;
            for (int i = 0; i < niter; ++i) {
                HIP_CHECK(hipEventRecord(e0, st));
                run_arm(ARM_WIDE6, ctx, dst_t);
                run_arm(ARM_WIDE6, ctx, dst_t);
                HIP_CHECK(hipEventRecord(e1, st));
                HIP_CHECK(hipEventSynchronize(e1));
                float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
                vd.push_back(ms * 1000.0 / 2.0);
            }
            for (int i = 0; i < niter; ++i) {
                HIP_CHECK(hipEventRecord(e0, st));
                run_arm(ARM_BASE, ctx, dst_t);
                run_arm(ARM_BASE, ctx, dst_t);
                HIP_CHECK(hipEventRecord(e1, st));
                HIP_CHECK(hipEventSynchronize(e1));
                float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
                vd2.push_back(ms * 1000.0 / 2.0);
            }
            report("double_wide6/2", vd);
            report("double_base/2", vd2);
        }

        // pipelined window (A3 protocol discipline): events only around the
        // whole loop - distinguishes empty-queue vs full-queue gap behavior
        {
            std::vector<double> vpb, vpw;
            for (int w = 0; w < 6; ++w) {
                HIP_CHECK(hipEventRecord(e0, st));
                for (int i = 0; i < niter; ++i) run_arm(ARM_BASE, ctx, dst_t);
                HIP_CHECK(hipEventRecord(e1, st));
                HIP_CHECK(hipEventSynchronize(e1));
                float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
                vpb.push_back(ms * 1000.0 / niter);
                HIP_CHECK(hipEventRecord(e0, st));
                for (int i = 0; i < niter; ++i) run_arm(ARM_WIDE6, ctx, dst_t);
                HIP_CHECK(hipEventRecord(e1, st));
                HIP_CHECK(hipEventSynchronize(e1));
                float ms2 = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms2, e0, e1));
                vpw.push_back(ms2 * 1000.0 / niter);
            }
            report("pipe_base", vpb);
            report("pipe_wide6", vpw);
        }

        // host enqueue rate (no sync inside): bounds the host contribution
        {
            HIP_CHECK(hipDeviceSynchronize());
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 30; ++i) run_arm(ARM_BASE, ctx, dst_t);
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < 30; ++i) run_arm(ARM_WIDE6, ctx, dst_t);
            auto t2 = std::chrono::steady_clock::now();
            HIP_CHECK(hipDeviceSynchronize());
            printf("MATRIX hostrate        base %6.1f us/launch  wide6 %6.1f us/launch (host enqueue only)\n",
                   std::chrono::duration<double, std::micro>(t1 - t0).count() / 30.0,
                   std::chrono::duration<double, std::micro>(t2 - t1).count() / 30.0);
        }

        // LDS x grid discriminator: minimal kernels with the base (22528 B)
        // and wide6 (32768 B) static LDS, near-zero work, in both grid shapes
        {
            // dynamic-LDS padding probe on the REAL served kernel: base
            // (static 22528) padded to mid3/wide6 total LDS footprints
            {
                std::vector<double> v25, v32;
                for (int i = 0; i < niter; ++i) {
                    HIP_CHECK(hipEventRecord(e0, st));
                    launch_arm_pad<4, 2>(ctx, dst_t, 2560);
                    HIP_CHECK(hipEventRecord(e1, st));
                    HIP_CHECK(hipEventSynchronize(e1));
                    float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
                    v25.push_back(ms * 1000.0);
                }
                for (int i = 0; i < niter; ++i) {
                    HIP_CHECK(hipEventRecord(e0, st));
                    launch_arm_pad<4, 2>(ctx, dst_t, 10240);
                    HIP_CHECK(hipEventRecord(e1, st));
                    HIP_CHECK(hipEventSynchronize(e1));
                    float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
                    v32.push_back(ms * 1000.0);
                }
                report("base_pad_25088", v25);
                report("base_pad_32768", v32);
            }

            float * lds_out = nullptr;
            HIP_CHECK(hipMalloc(&lds_out, 4096*4));
            const dim3 g_wide(1, 56, 1), g_base(1, 37, 3);
            const int b_wide = 384, b_base = 256;
            HIP_CHECK(hipDeviceSynchronize());
            auto lds_timed = [&](auto kern, dim3 g, int b) -> double {
                HIP_CHECK(hipEventRecord(e0, st));
                kern<<<g, b, 0, st>>>(lds_out);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipEventRecord(e1, st));
                HIP_CHECK(hipEventSynchronize(e1));
                float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
                return ms * 1000.0;
            };
            for (int w = 0; w < 50; ++w) {
                widelaunch_lds_kernel<22528><<<g_wide, b_wide, 0, st>>>(lds_out);
                widelaunch_lds_kernel<32768><<<g_wide, b_wide, 0, st>>>(lds_out);
                widelaunch_lds_kernel<22528><<<g_base, b_base, 0, st>>>(lds_out);
                widelaunch_lds_kernel<32768><<<g_base, b_base, 0, st>>>(lds_out);
                widelaunch_lds_kernel<32768, true><<<g_wide, b_wide, 0, st>>>(lds_out);
                widelaunch_lds_kernel<22528, true><<<g_wide, b_wide, 0, st>>>(lds_out);
                widelaunch_lds_kernel<16384, true><<<g_wide, b_wide, 0, st>>>(lds_out);
                widelaunch_spill_light_kernel<<<g_wide, b_wide, 0, st>>>(lds_out);
                widelaunch_spill_lin_kernel<<<g_wide, b_wide, 0, st>>>(lds_out);
                widelaunch_bigcode_light_kernel<80><<<g_wide, b_wide, 0, st>>>(lds_out);
            }
            HIP_CHECK(hipDeviceSynchronize());
            auto lds_scenario = [&](const char * name, auto kern, dim3 g, int b) {
                std::vector<double> v;
                for (int i = 0; i < niter; ++i) v.push_back(lds_timed(kern, g, b));
                report(name, v);
            };
            lds_scenario("lds22k_gw384", widelaunch_lds_kernel<22528>, g_wide, b_wide);
            lds_scenario("lds32k_gw384", widelaunch_lds_kernel<32768>, g_wide, b_wide);
            lds_scenario("lds22k_gb256", widelaunch_lds_kernel<22528>, g_base, b_base);
            lds_scenario("lds32k_gb256", widelaunch_lds_kernel<32768>, g_base, b_base);
            lds_scenario("lds32k_vg128", widelaunch_lds_kernel<32768, true>, g_wide, b_wide);
            lds_scenario("lds22k_vg128", widelaunch_lds_kernel<22528, true>, g_wide, b_wide);
            lds_scenario("lds16k_vg128", widelaunch_lds_kernel<16384, true>, g_wide, b_wide);
            // print dummy attributes once (vgpr check for the burn variants)
            {
                hipFuncAttributes attr{};
                HIP_CHECK(hipFuncGetAttributes(&attr, (const void *) widelaunch_lds_kernel<32768, true>));
                printf("MATRIX attr lds32k_vg128: numRegs=%d shared=%zu local=%zu\n",
                       attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes);
                HIP_CHECK(hipFuncGetAttributes(&attr, (const void *) widelaunch_spill_light_kernel));
                printf("MATRIX attr spilllight: numRegs=%d shared=%zu local=%zu\n",
                       attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes);
                HIP_CHECK(hipFuncGetAttributes(&attr, (const void *) widelaunch_spill_lin_kernel));
                printf("MATRIX attr splillin: numRegs=%d shared=%zu local=%zu\n",
                       attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes);
                HIP_CHECK(hipFuncGetAttributes(&attr, (const void *) widelaunch_spill_kernel));
                printf("MATRIX attr spillheavy: numRegs=%d shared=%zu local=%zu\n",
                       attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes);
            }
            // spill and big-code discriminators, wide6-like shape
            float zero = 0.0f;
            HIP_CHECK(hipMemcpy(lds_out, &zero, 4, hipMemcpyHostToDevice));
            for (int w = 0; w < 50; ++w) {
                widelaunch_spill_light_kernel<<<g_wide, b_wide, 0, st>>>(lds_out);
                widelaunch_bigcode_light_kernel<80><<<g_wide, b_wide, 0, st>>>(lds_out);
            }
            HIP_CHECK(hipDeviceSynchronize());
            lds_scenario("spilllight_gw384", widelaunch_spill_light_kernel, g_wide, b_wide);
            lds_scenario("splillin_gw384", widelaunch_spill_lin_kernel, g_wide, b_wide);
            lds_scenario("bigcodelight_gw384", widelaunch_bigcode_light_kernel<80>, g_wide, b_wide);
            // alternate big/small LDS, wide-like shape
            {
                std::vector<double> vs, vl;
                for (int i = 0; i < niter; ++i) {
                    vs.push_back(lds_timed(widelaunch_lds_kernel<22528>, g_wide, b_wide));
                    vl.push_back(lds_timed(widelaunch_lds_kernel<32768>, g_wide, b_wide));
                }
                report("lds_alt_22k", vs);
                report("lds_alt_32k", vl);
            }
            HIP_CHECK(hipFree(lds_out));
        }
        HIP_CHECK(hipFree(noop_dev));

        ggml_free(gctx);
        return 0;
    }

    if (served) {
        // full served entry: exercises the fattn.cu TILE selection +
        // launch_fattn_tile_switch_ncols2 (+ GQA_WIDE arm when env is set)
        ggml_cuda_flash_attn_ext_tile(ctx, dst_t);
        HIP_CHECK(hipDeviceSynchronize());
        printf("SERVED-ENTRY done\n");
        ggml_free(gctx); return 0;
    }

    const std::vector<ArmKind> arms = prefill ? std::vector<ArmKind>{ ARM_BASEP, ARM_V11P, ARM_BASEDUPP }
                                      : allarms ? std::vector<ArmKind>{ ARM_BASE, ARM_MID3, ARM_WIDE6, ARM_DIRECT, ARM_BASEDUP }
                                      : solo   ? std::vector<ArmKind>{ ARM_BASE, ARM_V11, ARM_BASEDUP }
                                      : std::vector<ArmKind>{ ARM_BASE, ARM_V1, ARM_V2, ARM_V3, ARM_V4,
                                                              ARM_V6, ARM_V7, ARM_V8, ARM_V9,
                                                              ARM_V10, ARM_V11, ARM_BASEDUP };

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

    // staged oracle: in-kernel shared K/V tile dumps + KQ_acc dump vs base
    if (dbg_mode) {
        auto staged_run = [&](ArmKind k, std::vector<uint8_t> & snap) {
            run_arm(k, ctx, dst_t);
            HIP_CHECK(hipDeviceSynchronize());
            snap.resize(1 << 20);
            HIP_CHECK(hipMemcpy(snap.data(), dbg_dev, 1 << 20, hipMemcpyDeviceToHost));
        };
        std::vector<uint8_t> d0, d1;
        staged_run(ARM_BASE, d0);
        { FILE * f = fopen("/tmp/fa40_staged_base.bin", "wb"); if (f) { fwrite(d0.data(), 1, d0.size(), f); fclose(f); } }
        for (size_t a = 1; a < arms.size(); ++a) {
            staged_run(arms[a], d1);
            char pth[128]; snprintf(pth, sizeof(pth), "/tmp/fa40_staged_%s.bin", arm_name(arms[a]));
            { FILE * f = fopen(pth, "wb"); if (f) { fwrite(d1.data(), 1, d1.size(), f); fclose(f); } }
            if (arms[a] == ARM_BASEDUP) continue;
            const char * kt = "KT-X", * vt = "VT-X", * kq = "KQ-X";
            do {
                const float * b = (const float *) d0.data();
                const float * t = (const float *) d1.data();
                if (memcmp(b, t, sizeof(float)) != 0) break; // magic word only (variant ids differ by design)
                if (memcmp(b + 1024,  t + 1024,  8192*sizeof(float)) != 0) break;
                kt = "KT=ok";
                if (memcmp(b + 12288, t + 12288, 4096*sizeof(float)) != 0) break;
                vt = "VT=ok";
                if (memcmp(b + 16384, t + 16384, 1024*sizeof(float)) != 0) break;
                kq = "KQ=ok";
            } while (false);
            printf("STAGED %s vs base: %s %s %s\n", arm_name(arms[a]), kt, vt, kq);
        }
    }

    if (oracle_only) {
        ggml_free(gctx);
        return 0;
    }

    // warmup (2500 full 4-arm passes, ~8 s) into the hot-steady DVFS state;
    // windows after this all run at the same clock (per-rep diagnostic showed
    // rep-1 boost-state bias and sleep-driven p-state lottery otherwise);
    // BENCH_WARMUP overrides for kernel-trace runs (rocprof) only
    int n_warm = 2500;
    if (const char * wenv = getenv("BENCH_WARMUP")) n_warm = atoi(wenv);
    for (int w = 0; w < n_warm; ++w) {
        for (size_t a = 0; a < arms.size(); ++a) {
            run_arm(arms[a], ctx, dst_t);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    // timing: reps x (niter back-to-back launches, hip events), arms interleaved;
    // events ride ctx.stream() so they bracket the real launch stream;
    // rep window 0 is a measured DVFS warm-in window and is discarded
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
            if (r > 0) t[a].push_back(ms * 1000.0 / niter);
        }
    }
    printf("\nper-launch us (niter=%d, reps=%d), served geometry M=%ld gqa=6 ne11=%ld q4_0 KV:\n",
           niter, reps, g_T, g_ne11);
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
    for (size_t a = 0; a < arms.size(); ++a) {
        printf("REPS %-8s:", arm_name(arms[a]));
        for (size_t r = 0; r < t[a].size(); ++r) printf(" %.1f", t[a][r]);
        printf("\n");
    }
    ggml_free(gctx);
    return 0;
}
