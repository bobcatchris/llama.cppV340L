// Loader dequant-elimination desk: census-shape bench of the tile route with
// per-call f16 weight dequant (GGML_CUDA_TILE_FP16=1) vs load-time resident f16
// weights (GGML_CUDA_TILE_FP16=1 GGML_CUDA_TILE_FP16_RESIDENT=1) through the real
// ggml mul_mat dispatch, single device. M=128 census prefill shapes with per-die
// N shares and layer call counts (E-012 census / E-015 weights), IQ3_S weights.
// Wall op time via host timers (dequant + src1 convert + GEMM in the per-call
// arm; src1 convert + GEMM in the resident arm). GEMM-only kernel time comes from
// a rocprofv3 kernel trace of this same binary (tile_fp16_gemm/tile_fp16_reduce
// vs dequantize_block_* / convert_unary). --dump writes the f32 dst per shape for
// the byte-identity oracle: resident-vs-per-call dst must be bit-identical (same
// f16 values either way), same protocol as the E-024 env-unset proof.
//
// All shape buffers stay alive for the whole run: residency keys on the device
// pointer, so freeing ffn_gate before allocating ffn_up (same shape) could alias
// the two entries. Weights re-seed per shape, so dumps are comparable across runs.
//
// Build: g++ -O2 -std=c++17 -I ggml/include bench_tile_resident.cpp \
//   -L build-loader-dequant/bin -lggml -lggml-base -lggml-hip -Wl,-rpath,<abs bin>
// Run (die 3): HIP_VISIBLE_DEVICES=3 /tmp/bench_tile_resident [m] [reps] [iters] [--dump]

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

    const char * env_tile     = getenv("GGML_CUDA_TILE_FP16");
    const char * env_resident = getenv("GGML_CUDA_TILE_FP16_RESIDENT");
    const char * env_cap      = getenv("GGML_CUDA_TILE_FP16_RESIDENT_MIB");
    printf("tile-resident bench: M=%lld reps=%d x %d iters | TILE_FP16=%s RESIDENT=%s CAP_MIB=%s | free %.1f MiB\n",
           (long long) m, reps, iters,
           env_tile ? env_tile : "unset", env_resident ? env_resident : "unset",
           env_cap ? env_cap : "default", mem_free / 1048576.0);

    printf("%-13s %6s %5s %4s %9s %9s %12s %10s %10s\n",
           "shape", "K", "N_d", "n", "qMiB", "f16MiB", "ms/iter", "TF/s", "freeMiB");

    std::vector<shape_res> res(NSHAPES);
    std::vector<float>     ms_med(NSHAPES);
    std::vector<double>    fl(NSHAPES);

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

        // warmup: route selection, resident-buffer insert, graph capture
        for (int i = 0; i < 3; i++) {
            if (ggml_backend_graph_compute(backend, r.gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "compute failed\n"); return 1;
            }
        }

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
        ggml_backend_cuda_get_device_memory(0, &mem_free, &mem_total);
        printf("%-13s %6lld %5lld %4d %9.2f %9.2f %12.5f %10.2f %10.1f\n",
               s.name, (long long) s.k, (long long) s.nd, s.count,
               ggml_nbytes(r.w) / 1048576.0, (double)(s.nd * s.k * 2) / 1048576.0,
               ms_med[si], tfs, mem_free / 1048576.0);

        if (dump) {
            std::vector<float> out(ggml_nelements(r.o));
            ggml_backend_tensor_get(r.o, out.data(), 0, out.size() * sizeof(float));
            double sum = 0;
            for (size_t i = 0; i < out.size(); i++) sum += out[i];
            printf("    dump: sum %.6e mean %.6e out[0..3] %.6e %.6e %.6e %.6e\n",
                   sum, sum / out.size(), out[0], out[1], out[2], out[3]);
            char path[256];
            snprintf(path, sizeof(path), "/tmp/tile_resident_dump_s%d_m%lld.bin", si, (long long) m);
            FILE * f = fopen(path, "wb");
            if (f) { fwrite(out.data(), 4, out.size(), f); fclose(f); }
        }
    }

    // call-weighted die aggregate over the per-model instance counts (E-015)
    double gf_sum = 0, ms_wsum = 0;
    for (int si = 0; si < NSHAPES; si++) {
        gf_sum  += SHAPES[si].count * fl[si];
        ms_wsum += SHAPES[si].count * ms_med[si];
    }
    printf("call-weighted die agg: %.1f GF/ubatch, %.2f ms/ubatch -> %.2f TF/s/die wall\n",
           gf_sum / 1e9, ms_wsum, gf_sum / (ms_wsum * 1e-3) / 1e12);

    double f16_total = 0;
    for (int si = 0; si < NSHAPES; si++) {
        f16_total += SHAPES[si].nd * SHAPES[si].k * 2.0;
    }
    printf("bench-set f16 residency footprint: %.2f MiB (%.2f MiB/shape-set)\n",
           f16_total / 1048576.0, f16_total / 1048576.0);

    for (int si = 0; si < NSHAPES; si++) {
        ggml_backend_buffer_free(res[si].buf);
        res[si].ctxp.reset();
    }
    ggml_backend_free(backend);
    return 0;
}
