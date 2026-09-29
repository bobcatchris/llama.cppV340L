// N-span desk: census-shape bench of the tile route unchunked
// (GGML_CUDA_TILE_FP16=1, per-call full-weight f16 dequant - the alloc class that
// OOMs at 200k, E-044) vs N-span full-K (GGML_CUDA_TILE_FP16=1
// GGML_CUDA_TILE_FP16_NSPAN=1, contiguous row spans with the full K dimension
// dequantized straight into a reusable window, launches keeping gridDim.z = ks and
// the P + fixed-order reduce) through the real ggml mul_mat dispatch, single
// device. M=128 census prefill shapes with per-die N shares and layer call counts
// (E-012 census / E-015 weights), IQ3_S weights. Wall op time via host timers
// (dequant + src1 convert + GEMM in both arms). GEMM-only kernel time comes from a
// rocprofv3 kernel trace of this same binary. --dump writes the f32 dst per shape
// for the byte-identity oracle: nspan-vs-unchunked dst must be bit-identical (same
// per-element slice schedule and f32 reduce order; window dequant runs the same
// to_fp16 kernels on the same contiguous quant rows).
//
// VRAM proof: a sampler thread polls free VRAM every ~0.4 ms for the whole run;
// peak delta = free-at-start - min-free, i.e. every transient the route creates
// (pool blocks + span window), against the 200 MiB free envelope at TP3/200k.
//
// Build: g++ -O2 -std=c++17 -I ggml/include bench_tile_nspan.cpp \
//   -L build-tile-nspan/bin -lggml -lggml-base -lggml-hip -Wl,-rpath,<abs bin>
// Run (die 3): HIP_VISIBLE_DEVICES=3 /tmp/bench_tile_nspan [m] [reps] [iters] [--dump]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

struct shape {
    const char * name;
    int64_t k;
    int64_t nd;    // die0 share
    int     count; // instances per model (call weight, E-015)
};

static const shape SHAPES[] = {
    { "ffn_gate   ",  5120, 5760, 64 },
    { "ffn_up     ",  5120, 5760, 64 },
    { "ffn_down   ", 17408, 1664, 64 },
    { "gdn_qkv    ",  5120, 3328, 48 },
    { "ssm_out    ",  6144, 1664, 48 },
    { "gdn_gate   ",  5120, 2048, 48 },
    { "attn_q+gate",  5120, 4096, 16 },
    { "attn_out   ",  6144, 1664, 16 },
    { "attn_k     ",  5120,  256, 16 },
    { "attn_v     ",  5120,  256, 16 },
};
static const int NSHAPES = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));

struct shape_res {
    ggml_context_ptr    ctxp;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * w = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * o = nullptr;
    ggml_cgraph * gf = nullptr;
};

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char ** argv) {
    const int64_t m     = argc > 1 && argv[1][0] != '-' ? atoll(argv[1]) : 128;
    const int     reps  = argc > 2 ? atoi(argv[2]) : 3;
    const int     iters = argc > 3 ? atoi(argv[3]) : 20;
    const bool    dump  = argc > 4 && strcmp(argv[4], "--dump") == 0;

    ggml_time_init();

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed\n");
        return 1;
    }

    size_t mem_free = 0, mem_total = 0;
    ggml_backend_cuda_get_device_memory(0, &mem_free, &mem_total);
    const size_t base_free = mem_free;

    const char * env_tile  = getenv("GGML_CUDA_TILE_FP16");
    const char * env_nspan = getenv("GGML_CUDA_TILE_FP16_NSPAN");
    const char * env_win   = getenv("GGML_CUDA_TILE_FP16_NSPAN_MIB");
    const char * arm = env_nspan && env_nspan[0] == '1' ? "NSPAN" : "UNCHUNK";
    printf("tile-nspan bench [%s]: M=%lld reps=%d x %d iters | TILE_FP16=%s NSPAN=%s WIN_MIB=%s | free %.1f MiB\n",
           arm, (long long) m, reps, iters,
           env_tile ? env_tile : "unset", env_nspan ? env_nspan : "unset",
           env_win ? env_win : "default", base_free / 1048576.0);

    // free-VRAM sampler: catches every transient (pool blocks + span window);
    // per shape, main resets min_free to the post-alloc steady level so weight
    // residency is not counted, then reads back the during-compute dip
    std::atomic<bool> sampling{true};
    std::atomic<size_t> min_free{SIZE_MAX};
    std::thread sampler([&]() {
        size_t f = 0, t = 0;
        while (sampling.load(std::memory_order_relaxed)) {
            ggml_backend_cuda_get_device_memory(0, &f, &t);
            size_t cur = min_free.load(std::memory_order_relaxed);
            while (f < cur && !min_free.compare_exchange_weak(cur, f, std::memory_order_relaxed)) {}
            std::this_thread::sleep_for(std::chrono::microseconds(400));
        }
    });

    printf("%-13s %6s %5s %4s %9s %9s %12s %10s %10s %9s\n",
           "shape", "K", "N_d", "n", "qMiB", "f16MiB", "ms/iter", "TF/s", "freeMiB", "peakMiB");

    std::vector<shape_res> res(NSHAPES);
    std::vector<float>     ms_med(NSHAPES);
    std::vector<double>    fl(NSHAPES);
    size_t                 peak_max_bytes = 0;

    for (int si = 0; si < NSHAPES; si++) {
        const shape & s = SHAPES[si];
        shape_res & r = res[si];

        struct ggml_init_params ip = { /*mem_size*/ 32u*1024*1024, /*buffer*/ NULL, /*no_alloc*/ true };
        r.ctxp.reset(ggml_init(ip));
        ggml_context * ctx = r.ctxp.get();
        if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

        r.w = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ3_S, s.k, s.nd);
        r.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, m);
        r.o = ggml_mul_mat(ctx, r.w, r.x);

        r.buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!r.buf) { fprintf(stderr, "alloc failed\n"); return 1; }

        // weights: quantize sane random f32 through the real quantizer (raw random
        // bytes overflow the h16 accumulate and produce inf/nan); re-seed per shape
        uint32_t rng = 0x87654321u;
        std::vector<float> hw_f32((size_t) s.k * s.nd);
        for (size_t i = 0; i < hw_f32.size(); i++) {
            rng = rng * 1664525u + 1013904223u;
            hw_f32[i] = 0.25f * (((rng >> 8) & 0xFFFF) / 65535.0f - 0.5f);
        }
        std::vector<uint8_t> hw(ggml_nbytes(r.w));
        ggml_quantize_chunk(GGML_TYPE_IQ3_S, hw_f32.data(), hw.data(), 0, s.nd, s.k, nullptr);
        ggml_backend_tensor_set(r.w, hw.data(), 0, hw.size());

        std::vector<float> hx((size_t) s.k * m);
        for (size_t i = 0; i < hx.size(); i++) {
            rng = rng * 1664525u + 1013904223u;
            hx[i] = ((rng >> 8) & 0xFFFF) / 65535.0f - 0.5f;
        }
        ggml_backend_tensor_set(r.x, hx.data(), 0, hx.size() * sizeof(float));

        r.gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(r.gf, r.o);

        ggml_backend_cuda_get_device_memory(0, &mem_free, &mem_total);
        const size_t steady_free = mem_free;
        min_free.store(steady_free);
        fprintf(stderr, "  [s%d %s] post-alloc free %.1f MiB\n", si, s.name, steady_free / 1048576.0);

        // warmup: route selection, span-window alloc, growth
        for (int i = 0; i < 3; i++) {
            if (ggml_backend_graph_compute(backend, r.gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "compute failed\n"); return 1;
            }
        }
        ggml_backend_cuda_get_device_memory(0, &mem_free, &mem_total);
        fprintf(stderr, "  [s%d %s] post-warmup free %.1f MiB\n", si, s.name, mem_free / 1048576.0);

        std::vector<float> ms_rep;
        for (int rp = 0; rp < reps; rp++) {
            const int64_t t0 = ggml_time_us();
            for (int i = 0; i < iters; i++) {
                if (ggml_backend_graph_compute(backend, r.gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "compute failed\n"); return 1;
                }
            }
            const int64_t t1 = ggml_time_us();
            ms_rep.push_back((t1 - t0) / 1000.0f / iters);
        }

        ms_med[si]   = median(ms_rep);
        fl[si]       = 2.0 * s.nd * m * s.k;
        const double tfs = fl[si] / (ms_med[si] * 1e-3) / 1e12;
        const size_t shape_min = min_free.exchange(SIZE_MAX);
        const double peak = (steady_free - std::min(steady_free, shape_min)) / 1048576.0;
        if (steady_free > shape_min && steady_free - shape_min > peak_max_bytes) {
            peak_max_bytes = steady_free - shape_min;
        }
        ggml_backend_cuda_get_device_memory(0, &mem_free, &mem_total);
        printf("%-13s %6lld %5lld %4d %9.2f %9.2f %12.5f %10.2f %10.1f %9.1f\n",
               s.name, (long long) s.k, (long long) s.nd, s.count,
               ggml_nbytes(r.w) / 1048576.0, (double)(s.nd * s.k * 2) / 1048576.0,
               ms_med[si], tfs, mem_free / 1048576.0, peak);

        if (dump) {
            std::vector<float> out(ggml_nelements(r.o));
            ggml_backend_tensor_get(r.o, out.data(), 0, out.size() * sizeof(float));
            char path[256];
            snprintf(path, sizeof(path), "/tmp/tile_nspan_dump_s%d_m%lld_%s.bin", si, (long long) m, arm);
            FILE * f = fopen(path, "wb");
            if (f) { fwrite(out.data(), 4, out.size(), f); fclose(f); }
        }
    }

    sampling.store(false);
    sampler.join();

    // call-weighted die aggregate over the per-model instance counts (E-015)
    double gf_sum = 0, ms_wsum = 0;
    for (int si = 0; si < NSHAPES; si++) {
        gf_sum  += SHAPES[si].count * fl[si];
        ms_wsum += SHAPES[si].count * ms_med[si];
    }
    printf("call-weighted die agg: %.1f GF/ubatch, %.2f ms/ubatch -> %.2f TF/s/die wall\n",
           gf_sum / 1e9, ms_wsum, gf_sum / (ms_wsum * 1e-3) / 1e12);

    printf("VRAM proof [%s]: peak transient delta over shapes %.1f MiB (envelope 200.0 MiB) -> %s\n",
           arm, peak_max_bytes / 1048576.0,
           peak_max_bytes <= 200ull*1024*1024 ? "FIT" : "OVER");

    for (int si = 0; si < NSHAPES; si++) {
        ggml_backend_buffer_free(res[si].buf);
        res[si].ctxp.reset();
    }
    ggml_backend_free(backend);
    return 0;
}
