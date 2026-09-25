// T3-reopen host test: grouped-mmvq DETECTION + copy-back logic over a real
// ggml graph.
//
// Mirrors the ggml_cuda_try_group_mmvq scan (ggml/src/ggml-cuda/ggml-cuda.cu,
// GGML_CUDA_MMVQ_GROUP=1) and the exec-side row whitelist, and runs them over a
// ggml_gallocr-allocated qwen35 GDN-layer graph (the E-042 served shape:
// decode T=1x1, per-die axis-1 weight shards). Graph topology mirrors
// src/models/qwen35.cpp build_layer_attn_linear; tensor sizes mirror the
// Qwen3.8-27B census (n_embd 5120, ssm n_v_heads 48).
//
// Fusion spans are stubbed to 0: the census tiny class is has_fusion=false
// (mul_mat_vec_q<...,1,false,false>) and no try_fuse pattern matches the
// MM/view/UNARY sequences around ssm_beta/ssm_alpha (verified by read of
// ggml_cuda_can_fuse; the GLU patterns need GGML_OP_GLU, sigmoid is UNARY).
//
// Build/run (host only, no GPU):
//   g++ -O2 -std=gnu++17 -I ggml/include -I ggml/src \
//     -x c++ docs/amd-port/tests/test_t3_detect_host.cpp \
//     -o /tmp/test_t3_detect -L build-hip/bin -lggml-base -lggml-cpu
//   LD_LIBRARY_PATH=build-hip/bin /tmp/test_t3_detect

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

static int test_failures = 0;
#define REQUIRE(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); test_failures++; } } while (0)

// ---- mirror: GGML_CUDA_MMVQ_GROUP_MAX / WINDOW / default max_rows ----
static constexpr int    GGML_GROUP_MAX    = 8;
static constexpr int    GGML_GROUP_WINDOW = 16;
static constexpr int64_t DEFAULT_MAX_ROWS = 256;

// ---- mirror: ggml_cuda_mmvq_group_node_eligible ----
static bool node_eligible(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT || node->src[2] != nullptr) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1 || src1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_quantized(src0->type)) {
        return false;
    }
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false; // decode T=1 shape whitelist
    }
    if (node->ne[1] != 1 || node->ne[2] != 1 || node->ne[3] != 1) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src1->ne[0] % 32 != 0) { // QK8_1
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }
    // recycled-padding check: buffer usage != COMPUTE on CPU/weights -> passes
    return true;
}

// ---- mirror: ggml_cuda_mmvq_group_overlap (CPU buft alloc size == nbytes) ----
static bool ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a->data || !b->data) {
        return true;
    }
    const uintptr_t ab = (uintptr_t) a->data;
    const uintptr_t ae = ab + ggml_nbytes(a);
    const uintptr_t bb = (uintptr_t) b->data;
    const uintptr_t be = bb + ggml_nbytes(b);
    return ab < be && bb < ae;
}

// ---- mirror: ggml_cuda_mmvq_group_safe_candidate (fusion spans stubbed 0) ----
static bool safe_candidate(const ggml_cgraph * gf, const int i, const int j, const ggml_tensor * src1) {
    const ggml_tensor * dst_j = gf->nodes[j];
    for (int k = i + 1; k < j; ++k) {
        const ggml_tensor * nk = gf->nodes[k];
        const bool computed = !ggml_is_empty(nk) && nk->op != GGML_OP_RESHAPE && nk->op != GGML_OP_TRANSPOSE &&
            nk->op != GGML_OP_VIEW && nk->op != GGML_OP_PERMUTE && nk->op != GGML_OP_NONE;
        if (computed) {
            if (ranges_overlap(nk, dst_j)) {
                return false; // would write the candidate's dst range
            }
            if (nk->src[0] && ranges_overlap(nk->src[0], dst_j)) {
                return false;
            }
            if (ranges_overlap(nk, src1)) {
                return false; // would mutate the shared x in between
            }
        }
        // fusion spans stubbed to 0 (see file header)
    }
    return true;
}

// ---- mirror: the fixed try_group_mmvq scan (detection only, no exec) ----
// aliased members are admitted as copy-back (temp + D2D copy at the member's
// own graph position) instead of dropped; direct members keep the early write
struct scan_result {
    int n = 1;               // members incl. head
    int n_candidates = 0;    // same-src1 eligible nodes in window
    int n_copy_back = 0;     // admitted members needing the temp+copy path
};
static scan_result try_group_scan(const ggml_cgraph * gf, int i, int64_t max_rows_used_by_exec) {
    scan_result out;
    const ggml_tensor * head = gf->nodes[i];
    if (!node_eligible(head)) {
        out.n = 0;
        return out;
    }
    const ggml_tensor * src1 = head->src[1];
    std::vector<const ggml_tensor *> members = { head };
    std::vector<bool> direct = { true };
    const int j_max = std::min(gf->n_nodes, i + GGML_GROUP_WINDOW);
    for (int j = i + 1; j < j_max && (int) members.size() < GGML_GROUP_MAX; ++j) {
        const ggml_tensor * cand = gf->nodes[j];
        if (!node_eligible(cand) || cand->src[1] != src1) {
            continue;
        }
        out.n_candidates++;
        bool cand_direct = safe_candidate(gf, i, j, src1);
        bool overlaps_member = false;
        for (const ggml_tensor * m : members) {
            overlaps_member = overlaps_member || ranges_overlap(cand, m);
        }
        if (overlaps_member) {
            cand_direct = false; // aliased dst: copy-back member
        }
        direct.push_back(cand_direct);
        members.push_back(cand);
    }
    out.n = (int) members.size();
    for (int m = 1; m < (int) members.size(); ++m) {
        out.n_copy_back += direct[m] ? 0 : 1;
    }
    // exec-side row whitelist (non-split): any member over max_rows aborts
    for (const ggml_tensor * m : members) {
        if (m->src[0]->ne[1] > max_rows_used_by_exec) {
            out.n = -1; // group detected but aborted by the whitelist
            break;
        }
    }
    if (out.n >= 0 && out.n < 2) {
        out.n = 0;
    }
    return out;
}

static int find_node(const ggml_cgraph * gf, const ggml_tensor * t) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i] == t) {
            return i;
        }
    }
    return -1;
}

int main() {
    ggml_time_init();

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    struct ggml_init_params ip = { 32*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t ne10  = 5120;
    const int64_t qkv_n = 10240; // attn_qkv width  (2*key_dim + value_dim)
    const int64_t z_n   = 6144;  // attn_gate width (value_dim)
    const int64_t ba_n  = 48;    // ssm_beta / ssm_alpha rows

    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  ne10, 1);
    ggml_tensor * w_qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ne10, qkv_n);
    ggml_tensor * w_z   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, ne10, z_n);
    ggml_tensor * w_b   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, ne10, ba_n);
    ggml_tensor * w_a   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ne10, ba_n);
    ggml_tensor * ssm_dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ba_n);

    // qwen35 build_layer_attn_linear node order (T=1x1 decode)
    ggml_tensor * qkv      = ggml_mul_mat(ctx, w_qkv, x);
    ggml_tensor * qkv_v    = ggml_reshape_3d(ctx, qkv, qkv_n, 1, 1);
    ggml_tensor * z        = ggml_mul_mat(ctx, w_z, x);
    ggml_tensor * beta     = ggml_mul_mat(ctx, w_b, x);
    ggml_tensor * beta_v   = ggml_reshape_4d(ctx, beta, 1, ba_n, 1, 1);
    ggml_tensor * beta_s   = ggml_sigmoid(ctx, beta_v);
    ggml_tensor * alpha    = ggml_mul_mat(ctx, w_a, x);
    ggml_tensor * alpha_v  = ggml_reshape_3d(ctx, alpha, ba_n, 1, 1);
    ggml_tensor * alpha_b  = ggml_add(ctx, alpha_v, ssm_dt);
    ggml_tensor * beta_s2  = ggml_reshape_2d(ctx, beta_s, ba_n, 1); // downstream consumer shape
    ggml_tensor * sink     = ggml_add(ctx, alpha_b, beta_s2);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    // wire every subtree in program order (unreferenced nodes are dropped by
    // the builder, so the big-cell matmuls need their own expands)
    ggml_build_forward_expand(gf, qkv_v);
    ggml_build_forward_expand(gf, z);
    ggml_build_forward_expand(gf, sink);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    REQUIRE(ggml_gallocr_alloc_graph(galloc, gf), "gallocr alloc failed");

    const int i_qkv = find_node(gf, qkv);
    const int i_z   = find_node(gf, z);
    const int i_b   = find_node(gf, beta);
    const int i_a   = find_node(gf, alpha);
    printf("graph nodes=%d: qkv@%d z@%d beta@%d alpha@%d\n", gf->n_nodes, i_qkv, i_z, i_b, i_a);
    REQUIRE(i_qkv >= 0 && i_z >= 0 && i_b >= 0 && i_a >= 0, "nodes missing");
    const int tiny_gap = i_a > i_b ? i_a - i_b : i_b - i_a;
    REQUIRE(tiny_gap > 0 && tiny_gap < 16, "tiny pair outside window");
    for (int i = 0; i < gf->n_nodes; i++) {
        printf("  node[%d] %-8s %s dst=%p nb=%zu\n", i, ggml_op_desc(gf->nodes[i]),
               gf->nodes[i]->name, gf->nodes[i]->data, ggml_nbytes(gf->nodes[i]));
    }
    fflush(stdout);
    REQUIRE(gf->nodes[i_b]->src[1] == gf->nodes[i_a]->src[1], "beta/alpha src1 pointer identity");

    // A. eligibility of the real tiny nodes
    REQUIRE(node_eligible(gf->nodes[i_b]), "beta eligible");
    REQUIRE(node_eligible(gf->nodes[i_a]), "alpha eligible");

    // B. the pair forms at default max_rows: the scan is forward-only, so the
    // first tiny MM in node order is the head; the second (sitting in the
    // recycled range of the first) is admitted as copy-back - the E-042 fix
    const int i_first = i_a < i_b ? i_a : i_b;
    const int i_second = i_a < i_b ? i_b : i_a;
    const scan_result r_first = try_group_scan(gf, i_first, DEFAULT_MAX_ROWS);
    REQUIRE(r_first.n == 2, "first-anchor pair forms (n_candidates %d, n %d)", r_first.n_candidates, r_first.n);
    REQUIRE(r_first.n_copy_back == 1, "the later member classifies copy-back (got %d)", r_first.n_copy_back);
    const scan_result r_second = try_group_scan(gf, i_second, DEFAULT_MAX_ROWS);
    REQUIRE(r_second.n == 0, "second anchor has no forward candidate (n %d)", r_second.n);

    // C. allocator fact (E-042 mechanism): gallocr recycles the first tiny
    // member's dst range for the second - the overlap check that used to drop
    // the candidate fires on the real ranges
    REQUIRE(ranges_overlap(gf->nodes[i_b], gf->nodes[i_a]),
        "beta/alpha dst ranges overlap (the recycled-range mechanism)");

    // D. big-cell heads: detection admits, whitelist aborts at 256
    scan_result r_q = try_group_scan(gf, i_qkv, DEFAULT_MAX_ROWS);
    REQUIRE(r_q.n == -1, "qkv group detected but whitelist-aborted, got n=%d", r_q.n);
    scan_result r_z = try_group_scan(gf, i_z, DEFAULT_MAX_ROWS);
    REQUIRE(r_z.n == -1, "z group detected but whitelist-aborted, got n=%d", r_z.n);

    // E. raised whitelist admits the big same-src1 sets (MAX_ROWS sweep class;
    // in the served graph these widths are per-die shards, here they are full
    // tensors, so the sweep values scale accordingly)
    REQUIRE(try_group_scan(gf, i_z, 6144).n >= 2, "z+... forms at max_rows=6144");
    REQUIRE(try_group_scan(gf, i_qkv, 10240).n >= 2, "qkv+... forms at max_rows=10240");

    // F. prefill-tail x (ne1=8) is outside the T=1 whitelist
    {
        struct ggml_context * ctx2 = ggml_init(ip);
        ggml_tensor * x8  = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32,  ne10, 8);
        ggml_tensor * w   = ggml_new_tensor_2d(ctx2, GGML_TYPE_Q4_K, ne10, ba_n);
        ggml_tensor * m8  = ggml_mul_mat(ctx2, w, x8);
        struct ggml_cgraph * gf2 = ggml_new_graph(ctx2);
        ggml_build_forward_expand(gf2, m8);
        REQUIRE(!node_eligible(gf2->nodes[0]), "T=8 prefill tail ineligible");
        ggml_free(ctx2);
    }

    // G. attn k/v row math under the TP3 meta split (pure numeric mirror):
    // n_embd_q = 6*256 = 1536, granularity_kv = lcm(1536,256)/6 = 512,
    // kv rows 1024 -> per-die [512,512,0] (rotating) -> default whitelist
    // rejects every attn k/v group; 512 admits the two nonzero dies.
    {
        const int64_t kv_rows = 4*256;
        const int64_t gran    = 512;
        int64_t rows[3] = { gran, gran, kv_rows - 2*gran };
        REQUIRE(rows[2] == 0, "third die zero slice");
        REQUIRE(rows[0] > DEFAULT_MAX_ROWS, "k/v rejected at default max_rows");
        REQUIRE(rows[0] <= 512, "k/v admitted at max_rows=512");
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (test_failures == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURES\n", test_failures);
    return 1;
}
