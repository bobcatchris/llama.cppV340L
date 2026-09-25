// test_meta_reentry_host.cpp - the served TP3 draft-MTP crash:
//   GGML_ASSERT(bcj.nodes[i]) at ggml/src/ggml-backend-meta.cpp:1929,
//   first draft-MTP decode of the engaged shape-cache arm (4x gfx900).
//
// Mechanism (all exercised here against the REAL meta backend + gallocr):
//   1. The llama-context shape cache re-enters a PERSISTENT built graph
//      (src/llama-context.cpp:1613-1671, the can_reuse && !sched_is_res
//      branch): the graph's tensors keep their allocation, so the sched
//      re-alloc skips them entirely - ggml_gallocr_init_tensor
//      (ggml/src/ggml-alloc.c:969) only initializes tensors with
//      data == NULL, so no init_tensor runs and no per-die simple tensors
//      are (re)created for the re-entered graph.
//   2. Every meta compute whose uid changed cycles the double-buffered
//      simple-tensor containers and CLEARS the next one
//      (ggml-backend-meta.cpp:1905-1913) - which destroyed the
//      registrations the re-entered graph still needed.
//   3. The rebuild mapping loop then gets NULL from
//      ggml_backend_meta_buffer_simple_tensor and the assert fired served.
//      The draft-MTP ctx alternates catchup/step shapes, so with 2 cache
//      slots the re-entry branch fires on every alternation - first hit
//      crashes. The fix: re-register missing simple tensors in the rebuild
//      loop (fail-open); the uid memo still replays the device graphs.
//
// Host rig: meta device over 2x the CPU device (same device object twice;
// the registration lifecycle under test is device-count-independent), an
// all-MIRRORED split policy (no cross-device reduce needed), and a
// single-backend sched (its uid propagates to the split graph,
// ggml/src/ggml-backend.cpp:1559). Graphs A/B stand in for the draft
// catchup/step shape pair; re-entry goes through the same
// sched_reset + sched_alloc_graph sequence as src/llama-context.cpp:1662.
//
// Sections (heal-WARN counts via ggml_log_set capture):
//   1. A fresh build+alloc+compute -> correct result, no heal
//   2. B fresh (clears A's container) -> correct result, no heal
//   3. A RE-ENTRY (persistent) -> pre-fix aborts at the assert;
//      post-fix heals (>= 1 WARN), result identical to section 1
//   4. B re-entry -> heals, result identical to section 2
//   5. A again (steady alternation, the served crash loop) -> heals, correct
//   6. negative control: immediate A reuse (same uid -> no rebuild)
//      -> zero heal WARNs, result still correct
//
// Build (see run_premerge_ci.sh):
//   hipcc -O2 -std=c++17 -x c++ -I include -I ggml/include -I ggml/src \
//     test_meta_reentry_host.cpp -o /tmp/test_meta_reentry_host -pthread \
//     -Wl,<bin>/libggml.so -Wl,<bin>/libggml-cpu.so -Wl,-rpath,<abs bin>
// Run: /tmp/test_meta_reentry_host   (zero GPU: only the CPU backend loads)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h" // struct ggml_cgraph (uid)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// not in the public header; exported by libggml-cpu.so (GGML_BACKEND_DL_IMPL)
extern "C" ggml_backend_reg_t ggml_backend_cpu_reg(void);

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

static constexpr int H = 64;

// split policy: every statically-allocated tensor is replicated (MIRRORED);
// compute tensors derive compatible states from their ops
static ggml_backend_meta_split_state split_cb(const ggml_tensor *, void *) {
    ggml_backend_meta_split_state ret = {};
    ret.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    ret.nr[0] = 1;
    ret.n_segments = 1;
    return ret;
}

// heal-WARN capture (the fix's diagnostics line)
static int n_heal_warn = 0;
static void test_log_callback(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN && strstr(text, "re-registered")) {
        n_heal_warn++;
    }
}

static ggml_context * wctx = nullptr;
static ggml_tensor * w1 = nullptr;
static ggml_tensor * w2 = nullptr;
static std::vector<float> w1h, w2h;

static void fill_pattern(std::vector<float> & v, float base) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = base + (float)(i % 17) * 0.25f;
    }
}

struct test_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * x   = nullptr;
    ggml_tensor  * out = nullptr;
    int n = 0;
};

static test_graph build_graph(int n, uint64_t uid) {
    test_graph g;
    g.n = n;
    const size_t mem = ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false);
    struct ggml_init_params ip = { mem, nullptr, /*no_alloc =*/ true };
    g.ctx = ggml_init(ip);
    g.x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, H, n);
    g.x->flags |= GGML_TENSOR_FLAG_INPUT;
    ggml_tensor * y1 = ggml_mul_mat(g.ctx, w1, g.x);
    ggml_tensor * y2 = ggml_mul_mat(g.ctx, w2, g.x);
    g.out = ggml_add(g.ctx, y1, y2);
    g.gf = ggml_new_graph_custom(g.ctx, 16, false);
    ggml_build_forward_expand(g.gf, g.out);
    g.gf->uid = uid;
    return g;
}

static void set_input(const test_graph & g) {
    std::vector<float> hx(H*g.n);
    fill_pattern(hx, 0.5f);
    ggml_backend_tensor_set(g.x, hx.data(), 0, hx.size()*sizeof(float));
}

static std::vector<float> input_pattern(int n) {
    std::vector<float> hx(H*n);
    fill_pattern(hx, 0.5f);
    return hx;
}

static std::vector<float> cpu_reference(int n) {
    const std::vector<float> hx = input_pattern(n);
    std::vector<float> ref(H*n, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < H; j++) {
            float acc = 0.0f;
            for (int k = 0; k < H; k++) {
                acc += (w1h[k + j*H] + w2h[k + j*H]) * hx[k + i*H];
            }
            ref[j + i*H] = acc;
        }
    }
    return ref;
}

// one decode: the llama-context sequence of the reuse/re-entry branches
// (sched_reset wipes tensor backend pins, so re-pin after each reset:
// the CPU backend is present only for the sched's CPU-last convention and
// must never run - it cannot dereference meta buffer pointers)
static std::vector<float> run_pass(ggml_backend_sched_t sched, const test_graph & g, ggml_backend_t meta_backend) {
    ggml_backend_sched_reset(sched);
    for (int i = 0; i < g.gf->n_nodes; i++) {
        ggml_backend_sched_set_tensor_backend(sched, g.gf->nodes[i], meta_backend);
    }
    ggml_backend_sched_set_tensor_backend(sched, g.x, meta_backend);
    if (!ggml_backend_sched_alloc_graph(sched, g.gf)) {
        fprintf(stderr, "FAIL: sched_alloc_graph\n");
        n_fail++;
        return {};
    }
    if (getenv("METATEST_DEBUG")) {
        for (int i = 0; i < g.gf->n_nodes; i++) {
            fprintf(stderr, "node %d (%s) -> %s\n", i, ggml_get_name(g.gf->nodes[i]),
                ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, g.gf->nodes[i])));
        }
        fprintf(stderr, "x -> %s\n", ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, g.x)));
    }
    set_input(g);
    const enum ggml_status st = ggml_backend_sched_graph_compute_async(sched, g.gf);
    CHECK(st == GGML_STATUS_SUCCESS);
    std::vector<float> out(H*g.n);
    ggml_backend_tensor_get(g.out, out.data(), 0, out.size()*sizeof(float));
    return out;
}

static bool vec_eq(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    ggml_log_set(test_log_callback, nullptr);

    // 1. meta device over 2x CPU device (zero GPU: the CPU reg only)
    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    ggml_backend_dev_t cpu_dev = ggml_backend_reg_dev_get(cpu_reg, 0);
    ggml_backend_dev_t devs[2] = { cpu_dev, cpu_dev };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 2, split_cb, nullptr);
    CHECK(meta_dev != nullptr);
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    CHECK(meta_backend != nullptr);
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);
    // the sched requires a CPU-type backend last (fallback convention)
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu_dev, nullptr);
    CHECK(cpu_backend != nullptr);
    ggml_backend_t sched_backends[2] = { meta_backend, cpu_backend };

    // 2. static weights on the meta buffer type (the stc_static path)
    {
        struct ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, /*no_alloc =*/ true };
        wctx = ggml_init(ip);
        w1 = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, H, H);
        w2 = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, H, H);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, meta_buft);
        CHECK(wbuf != nullptr);
        w1h.resize(H*H);
        w2h.resize(H*H);
        fill_pattern(w1h, 1.0f);
        fill_pattern(w2h, -0.5f);
        ggml_backend_tensor_set(w1, w1h.data(), 0, w1h.size()*sizeof(float));
        ggml_backend_tensor_set(w2, w2h.data(), 0, w2h.size()*sizeof(float));
    }

    // 3. persistent graphs: A = catchup shape (8), B = step shape (4)
    test_graph ga = build_graph(8, 111);
    test_graph gb = build_graph(4, 222);

    ggml_backend_sched_t sched = ggml_backend_sched_new(sched_backends, nullptr, 2, 64, false, false);
    CHECK(sched != nullptr);


    const std::vector<float> ref_a = cpu_reference(8);
    const std::vector<float> ref_b = cpu_reference(4);
    std::vector<float> out;

    // section 1: A fresh (rebuild) - no heal possible
    n_heal_warn = 0;
    out = run_pass(sched, ga, meta_backend);
    CHECK(vec_eq(out, ref_a));
    CHECK(n_heal_warn == 0);
    printf("SEC1 A fresh: correct, heal-warns=%d\n", n_heal_warn);

    // section 2: B fresh (clears A's registrations) - no heal possible
    n_heal_warn = 0;
    out = run_pass(sched, gb, meta_backend);
    CHECK(vec_eq(out, ref_b));
    CHECK(n_heal_warn == 0);
    printf("SEC2 B fresh: correct, heal-warns=%d (A's simple tensors now cleared)\n", n_heal_warn);

    // section 3: A RE-ENTRY - the served crash shape
    // pre-fix this aborted at GGML_ASSERT(bcj.nodes[i]); post-fix it heals
    n_heal_warn = 0;
    out = run_pass(sched, ga, meta_backend);
    CHECK(vec_eq(out, ref_a));
    CHECK(n_heal_warn >= 1);
    printf("SEC3 A re-entry (persistent graph): correct, heal-warns=%d (was the 1929 abort)\n", n_heal_warn);

    // section 4: B re-entry - the heal converges: the registrations healed in
    // section 3 share one container, so B survives the following cycles and no
    // further heal is required; correctness is the invariant under test
    n_heal_warn = 0;
    out = run_pass(sched, gb, meta_backend);
    CHECK(vec_eq(out, ref_b));
    printf("SEC4 B re-entry: correct, heal-warns=%d (converged: no heal needed)\n", n_heal_warn);

    // section 5: A again - the steady catchup/step alternation survives
    n_heal_warn = 0;
    out = run_pass(sched, ga, meta_backend);
    CHECK(vec_eq(out, ref_a));
    printf("SEC5 A alternation: correct, heal-warns=%d\n", n_heal_warn);

    // section 6: negative control - same-shape reuse (same uid) never heals
    n_heal_warn = 0;
    out = run_pass(sched, ga, meta_backend);
    CHECK(vec_eq(out, ref_a));
    CHECK(n_heal_warn == 0);
    printf("SEC6 A immediate reuse (no rebuild): correct, heal-warns=%d\n", n_heal_warn);

    ggml_backend_sched_free(sched);
    ggml_backend_free(meta_backend);
    ggml_backend_free(cpu_backend);
    ggml_free(ga.ctx);
    ggml_free(gb.ctx);
    ggml_free(wctx);

    if (n_fail == 0) {
        printf("ALL PASS: meta re-entry (draft shape-cache crash) suite\n");
    } else {
        printf("FAILURES: %d\n", n_fail);
    }
    return n_fail != 0;
}
