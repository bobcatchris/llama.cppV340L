// T3-reopen host experiment: does the graph allocator recycle the ssm_beta dst
// range for the ssm_alpha dst in the served GDN-layer shape? (zero GPU)
//
// E-042 root-cause candidate: ggml_cuda_try_group_mmvq drops a candidate member
// whose dst range overlaps the head's dst (ggml_cuda_mmvq_group_overlap), because
// an early grouped write would clobber data an intermediate still reads. If the
// allocator hands alpha the just-freed beta range, every same-src1 tiny pair is
// rejected and the grouping never forms - exactly the E-042 census observation.
//
// RESULT OF RECORD: recycling is ORDER-DEPENDENT. This file's short chain
// (beta -> sigmoid -> alpha, no downstream inplace users) does NOT recycle:
// the sigmoid's inplace-reuse keeps beta's block alive, alpha lands elsewhere.
// The full qwen35 chain (alpha's reshape/ADD/softplus/mul sequence) DOES
// recycle - see test_t3_detect_host.cpp, which allocates the real topology
// and shows beta/alpha MM dsts at the identical address.
//
// Graph mirrors src/models/qwen35.cpp build_layer_attn_linear (T=1x1 decode):
//   x [5120,1] -> beta = w_beta*x [48,1] -> reshape -> sigmoid
//             -> alpha = w_alpha*x [48,1] -> consumer(add)
// weights q4_K-ish quantized so the nodes are the vec-q class; types do not
// matter for allocation, sizes do. Allocation via ggml_gallocr, the same
// allocator the backend scheduler uses for the unified graph compute buffer.
//
// Build/run (host only, no GPU):
//   cc -O2 -I ggml/include -I ggml/src docs/amd-port/tests/test_t3_alias_host.c \
//     -o /tmp/test_t3_alias -L build-hip/bin -lggml-base -lggml-cpu -Wl,-rpath,<abs>/build-hip/bin
//   /tmp/test_t3_alias

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static bool ranges_overlap(const void * a, size_t abytes, const void * b, size_t bbytes) {
    const uintptr_t ab = (uintptr_t) a;
    const uintptr_t bb = (uintptr_t) b;
    return ab < bb + bbytes && bb < ab + abytes;
}

int main() {
    ggml_time_init();

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "cpu backend init failed\n");
        return 1;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    struct ggml_init_params ip = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t ne10 = 5120;
    const int64_t n_out = 48; // ssm_beta / ssm_alpha rows (Qwen3.8: 48)

    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne10, 1);
    ggml_tensor * wb = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ne10, n_out);
    ggml_tensor * wa = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ne10, n_out);

    // qwen35 build_layer_attn_linear node order (T=1x1):
    //   beta = MM(wb, x); reshape(4d, noop); sigmoid; alpha = MM(wa, x); ...
    ggml_tensor * beta  = ggml_mul_mat(ctx, wb, x);
    ggml_tensor * beta_v = ggml_reshape_2d(ctx, beta, n_out, 1);
    ggml_tensor * beta_s = ggml_sigmoid(ctx, beta_v);
    ggml_tensor * alpha = ggml_mul_mat(ctx, wa, x);
    ggml_tensor * sink  = ggml_add(ctx, alpha, beta_s); // gated_delta_net consumes both

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sink);

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    // mark inputs, run once so the graph is a real compute
    float * xdata = (float *) x->data;
    for (int i = 0; i < ne10; i++) {
        xdata[i] = 0.25f * ((i % 17) - 8);
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }

    const size_t bytes = ggml_nbytes(beta);
    bool overlap = ranges_overlap(beta->data, bytes, alpha->data, ggml_nbytes(alpha));

    printf("beta dst  = %p (%zu B)\n", beta->data, bytes);
    printf("alpha dst = %p (%zu B)\n", alpha->data, ggml_nbytes(alpha));
    printf("beta_s dst = %p\n", beta_s->data);
    printf("recycled (alpha reuses beta's range): %s\n", overlap ? "YES" : "NO");

    // values: with alpha's early-write hazard the sigmoid would read alpha's result
    printf("sink[0] = %.6f\n", ((float *) sink->data)[0]);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (overlap) {
        printf("RESULT: recycling confirmed -> try_group_mmvq overlap check drops the pair\n");
        return 0;
    }
    printf("RESULT: no recycling in this short-chain order"
        " (the full qwen35 chain does recycle: test_t3_detect_host.cpp)\n");
    return 2;
}
