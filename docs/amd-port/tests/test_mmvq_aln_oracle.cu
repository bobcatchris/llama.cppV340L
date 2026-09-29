// Device oracle for the MMVQ aln producer+consumer pair (GGML_CUDA_MMVQ_ALN),
// die 3. Runs the REAL ggml_mul_mat on the CUDA backend over synthesized
// valid quantized weights, T = 1..4, two mul_mat nodes sharing one src1 (the
// q81 act-cache path), and dumps the full f32 dst. Run twice - gate unset vs
// LLAMA_MMVQ_ALN=1 (+ the six share gates, since aln arms build on share) -
// and memcmp the dumps on the host: the aln layout is value-identical, so
// every dump must be byte-identical to the legacy dump.
//
// Build (flags mirrored from build-gfx900):
//   /opt/rocm-6.2.0/bin/hipcc -O2 -std=gnu++17 -DGGML_USE_HIP -DGGML_BACKEND_BUILD \
//     -DGGML_SHARED -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -D__HIP_PLATFORM_AMD__=1 \
//     -D__HIP_ROCclr__=1 --offload-arch=gfx900 \
//     -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//     docs/amd-port/tests/test_mmvq_aln_oracle.cu -o /tmp/test_mmvq_aln_oracle \
//     -L build-gfx900/bin -lggml-hip -lggml-base -lamdhip64 \
//     -Wl,-rpath,<abs>/build-gfx900/bin -Wl,-rpath,/opt/rocm-6.2.0/lib
// Run (die 3 only, under the campaign lock):
//   HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=<abs>/build-gfx900/bin \
//     /tmp/test_mmvq_aln_oracle <out-dump-path>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cassert>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s <out-dump-path>\n", argv[0]);
        return 1;
    }
    const char * out_path = argv[1];

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        printf("ggml_backend_cuda_init failed\n");
        return 1;
    }

    constexpr int64_t K = 5120;   // served ffn src1 width
    constexpr int64_t N = 512;    // rows per weight tensor
    constexpr int64_t TMAX = 4;

    const ggml_type types[] = {
        GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_XS,   // share gate exists but negative: legacy control
    };

    // deterministic activation + weight source data
    std::vector<float> src1((size_t) K*TMAX);
    for (size_t i = 0; i < src1.size(); ++i) {
        src1[i] = ((float)((i*2654435761u) % 2001) - 1000.0f) / 256.0f;
    }

    // dump layout: [type][T][fixed slot of N*TMAX*2 floats]; the first N*T*2
    // floats of each slot are the (o, o2) outputs at that T
    const int NT = (int)(sizeof(types)/sizeof(types[0]));
    const size_t per_slot_floats = (size_t) N*TMAX*2;
    std::vector<uint8_t> dump((size_t) NT*TMAX*per_slot_floats*sizeof(float), 0);

    for (int it = 0; it < NT; ++it) {
        const ggml_type wtype = types[it];
        std::vector<float> wf((size_t) K*N);
        const uint32_t seed = 0x9E3779B9u * (uint32_t)(it + 1);
        for (size_t i = 0; i < wf.size(); ++i) {
            wf[i] = ((float)(((i + seed)*2654435761u) % 4097) - 2048.0f) / 512.0f;
        }
        std::vector<uint8_t> wq(ggml_row_size(wtype, K)*N);
        ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, N, K, nullptr);

        for (int64_t T = 1; T <= TMAX; ++T) {
            ggml_init_params ip = {
                /*mem_size*/ 256*1024*1024,
                /*mem_buffer*/ nullptr,
                /*no_alloc*/ true,
            };
            ggml_context * ctx = ggml_init(ip);

            ggml_tensor * w = ggml_new_tensor(ctx, wtype, 2, std::vector<int64_t>{K, N}.data());
            ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, std::vector<int64_t>{K, T}.data());
            ggml_tensor * o = ggml_mul_mat(ctx, w, x);

            ggml_tensor * o2 = nullptr;
            // second mul_mat sharing the same src1: exercises the q81 act-cache
            std::vector<uint8_t> wq2;
            ggml_tensor * w_b = nullptr;
            {
                std::vector<float> wf2((size_t) K*N);
                const uint32_t seed2 = seed ^ 0x85EBCA6Bu;
                for (size_t i = 0; i < wf2.size(); ++i) {
                    wf2[i] = ((float)(((i + seed2)*2654435761u) % 4097) - 2048.0f) / 512.0f;
                }
                wq2.resize(ggml_row_size(wtype, K)*N);
                ggml_quantize_chunk(wtype, wf2.data(), wq2.data(), 0, N, K, nullptr);
                w_b = ggml_new_tensor(ctx, wtype, 2, std::vector<int64_t>{K, N}.data());
                o2 = ggml_mul_mat(ctx, w_b, x);
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
            ggml_backend_tensor_set(w_b, wq2.data(), 0, wq2.size());
            ggml_backend_tensor_set(x, src1.data(), 0, K*T*sizeof(float));

            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, o);
            ggml_build_forward_expand(gf, o2);
            ggml_backend_graph_compute(backend, gf);

            std::vector<float> out((size_t) N*T*2);
            ggml_backend_tensor_get(o,  out.data(),                 0, (size_t) N*T*sizeof(float));
            ggml_backend_tensor_get(o2, out.data() + (size_t) N*T,  0, (size_t) N*T*sizeof(float));

            assert(out.size() <= per_slot_floats);
            memcpy(dump.data() + (((size_t) it*TMAX + (T-1))*per_slot_floats)*sizeof(float),
                   out.data(), out.size()*sizeof(float));

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        printf("type %d done\n", (int) wtype);
    }

    FILE * f = fopen(out_path, "wb");
    if (f == nullptr) {
        printf("open %s failed\n", out_path);
        return 1;
    }
    fwrite(dump.data(), 1, dump.size(), f);
    fclose(f);
    printf("dump written: %s (%zu bytes, %d types x T=1..4 x 2 outputs)\n",
           out_path, dump.size(), NT);
    return 0;
}
