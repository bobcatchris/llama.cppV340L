// W5 tile-integ desk: integrated per-shape bench of the env-gated tile dispatch
// (GGML_CUDA_TILE_FP16=1) vs the shipped route (env unset) through the real ggml
// mul_mat dispatch, single device. Census prefill shapes, per-die N shares, M=128
// (optional 256/512 via argv), IQ3_S weights (the served weight class, dequant
// included in both arms). Wall op time via events; GEMM-only kernel time via
// rocprofv3 kernel trace on the same binary (tile_fp16_gemm/tile_fp16_reduce vs
// the cublas kernels).
//
// Build: linked against the integrated build in build-tile-integ (see receipt).
// Run (die 3): HIP_VISIBLE_DEVICES=3 /tmp/bench_tile_integ [m] [reps] [iters] [--dump]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

struct shape {
    const char * name;
    int64_t k;
    int64_t nd;   // die0 share
};

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
    printf("tile-integ bench: M=%lld reps=%d x %d iters | route %s | %s\n",
           (long long) m, reps, iters,
           getenv("GGML_CUDA_TILE_FP16") ? getenv("GGML_CUDA_TILE_FP16") : "unset",
           ggml_backend_name(backend));

    printf("%-13s %6s %5s %12s %10s %8s\n", "shape", "K", "N_d", "ms/iter", "TF/s", "gate%");
    for (int si = 0; si < NSHAPES; si++) {
        const shape & s = SHAPES[si];
        struct ggml_init_params ip = { /*mem_size*/ 32u*1024*1024, /*buffer*/ NULL, /*no_alloc*/ true };
        ggml_context_ptr ctxp(ggml_init(ip));
        ggml_context * ctx = ctxp.get();
        if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ3_S, s.k, s.nd);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, m);
        ggml_tensor * o = ggml_mul_mat(ctx, w, x);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

        // weights: quantize sane random f32 through the real quantizer (raw random
        // bytes overflow the h16 accumulate and produce inf/nan)
        uint32_t rng = 0x87654321u;
        std::vector<float> hw_f32((size_t) s.k * s.nd);
        for (size_t i = 0; i < hw_f32.size(); i++) {
            rng = rng * 1664525u + 1013904223u;
            hw_f32[i] = 0.25f * (((rng >> 8) & 0xFFFF) / 65535.0f - 0.5f);
        }
        std::vector<uint8_t> hw(ggml_nbytes(w));
        ggml_quantize_chunk(GGML_TYPE_IQ3_S, hw_f32.data(), hw.data(), 0, s.nd, s.k, nullptr);
        ggml_backend_tensor_set(w, hw.data(), 0, hw.size());

        std::vector<float> hx((size_t) s.k * m);
        for (size_t i = 0; i < hx.size(); i++) {
            rng = rng * 1664525u + 1013904223u;
            hx[i] = ((rng >> 8) & 0xFFFF) / 65535.0f - 0.5f;
        }
        ggml_backend_tensor_set(x, hx.data(), 0, hx.size() * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);

        // warmup (route selection, graph capture, rocblas/tile code paths)
        for (int i = 0; i < 3; i++) {
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "compute failed\n"); return 1;
            }
        }

        std::vector<float> ms_rep;
        for (int r = 0; r < reps; r++) {
            const int64_t t0 = ggml_time_us();
            for (int i = 0; i < iters; i++) {
                if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "compute failed\n"); return 1;
                }
            }
            const int64_t t1 = ggml_time_us();
            ms_rep.push_back((t1 - t0) / 1000.0f / iters);
        }

        const double ms = median(ms_rep);
        const double fl = 2.0 * s.nd * m * s.k;
        const double tfs = fl / (ms * 1e-3) / 1e12;
        printf("%-13s %6lld %5lld %12.5f %10.2f %8.1f\n",
               s.name, (long long) s.k, (long long) s.nd, ms, tfs, 100.0 * tfs / 4.62);

        if (dump) {
            std::vector<float> out(ggml_nelements(o));
            ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
            double sum = 0;
            for (size_t i = 0; i < out.size(); i++) sum += out[i];
            printf("    dump: sum %.6e mean %.6e out[0..3] %.6e %.6e %.6e %.6e\n",
                   sum, sum / out.size(), out[0], out[1], out[2], out[3]);
            char path[256];
            snprintf(path, sizeof(path), "/tmp/tile_integ_dump_s%d_m%lld.bin", si, (long long) m);
            FILE * f = fopen(path, "wb");
            if (f) { fwrite(out.data(), 4, out.size(), f); fclose(f); }
        }

        ggml_backend_buffer_free(buf);
    }
    return 0;
}
