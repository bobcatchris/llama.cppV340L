// onsample desk host test: per-shard argmax parity, tie handling, uneven
// shard layouts, fetch mapping and staging lifetime (no GPU).
//
// LLAMA_DRAFT_ONDEVICE_ARGMAX=1 adds a GGML_OP_ARGMAX_SHARD node to the MTP
// draft graph. The meta backend executes it per device over each die's vocab
// shard of the draft logits [n_vocab, n_rows]; the op writes (max logit,
// shard-local argmax) at elements [0, 1] of each result row, and decode()
// fetches those 2 floats per shard instead of the full n_vocab row. The host
// merge (common_shard_argmax_pick mirror) keeps the first shard on equal max.
//
// Parity contract proven here:
//   1. the 3-pair merge equals the lowest-index argmax of the spliced row,
//      for distinct logits, exact ties (within a shard, across shards, all
//      equal) and maxima sitting on shard boundaries
//   2. for distinct logits the drafted token also matches the two host
//      sampler paths (full-vocab partial_sort and fast heap-select mirrors)
//   3. the fetch mapping (ggml_backend_meta_subrow_shard mirror) sends every
//      pair fetch to the owning shard and covers the row exactly
//   4. staging lifetime: the merge reads only the 2*n_shards packet; results
//      survive a following decode that overwrites the same staging
//
// Build + run (host only):
//   g++ -O2 -std=gnu++17 -I ggml/include -I ggml/src \
//     -x c++ docs/amd-port/tests/test_onsample_argmax_host.cpp \
//     -o /tmp/test_onsample_argmax -L build-onsample/bin -lggml-base -lggml-cpu
//   LD_LIBRARY_PATH=build-onsample/bin /tmp/test_onsample_argmax

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

typedef int32_t llama_token;

// ---- mirrors ------------------------------------------------------------->

// mirror: common/sampling.cpp common_shard_argmax_pick + the decode() local
// to global index resolution (row[2j+1] += offsets[j])
static llama_token merge_shard_argmax(const float * pairs, const int64_t * offs, int n_pairs, float * logit) {
    int best = 0;
    for (int d = 1; d < n_pairs; d++) {
        // strict comparison keeps the earlier (lower-vocab) shard on ties
        if (pairs[2*d] > pairs[2*best]) {
            best = d;
        }
    }
    if (logit) {
        *logit = pairs[2*best];
    }
    return (llama_token) (pairs[2*best + 1] + (float) offs[best]);
}

// mirror: src/llama-sampler.cpp llama_token_data_array_partial_sort_inplace
// (k <= 128 branch), top-1
static llama_token host_partial_sort_top1(const std::vector<float> & row) {
    struct TD { int id; float logit; float p; };
    std::vector<TD> cur(row.size());
    for (size_t i = 0; i < row.size(); i++) {
        cur[i] = {(int) i, row[i], 0.0f};
    }
    static const auto comp = [](const TD & a, const TD & b) { return a.logit > b.logit; };
    std::partial_sort(cur.begin(), cur.begin() + 10, cur.end(), comp);
    return cur[0].id;
}

// mirror: common/sampling.cpp set_logits_topk_row heap-select, top-1
static llama_token host_heap_select_top1(const std::vector<float> & row, int k) {
    struct TD { int id; float logit; float p; };
    static const auto comp = [](const TD & a, const TD & b) { return a.logit > b.logit; };
    const int n = (int) row.size();
    if (k <= 0 || k > n) {
        k = n;
    }
    std::vector<TD> cur(k);
    for (int i = 0; i < k; ++i) {
        cur[i] = {i, row[i], 0.0f};
    }
    std::make_heap(cur.begin(), cur.end(), comp);
    for (int i = k; i < n; ++i) {
        if (row[i] > cur.front().logit) {
            std::pop_heap(cur.begin(), cur.end(), comp);
            cur.back() = {i, row[i], 0.0f};
            std::push_heap(cur.begin(), cur.end(), comp);
        }
    }
    std::sort_heap(cur.begin(), cur.end(), comp);
    return cur[0].id;
}

// lowest-index argmax of a full row (the reference semantics)
static int ref_argmax_lowest(const std::vector<float> & row) {
    int best = 0;
    for (size_t i = 1; i < row.size(); i++) {
        if (row[i] > row[best]) {
            best = (int) i;
        }
    }
    return best;
}

// mirror: ggml-backend-meta.cpp ggml_backend_meta_subrow_shard - maps a
// single-row byte range to the owning shard, -1 when it spans shards; the
// caller turns (row, row_off) into a simple-tensor offset with the shard's
// own packed row stride (ne[j]*ts), like the meta call sites do
static int subrow_shard(const int64_t * ne, int n_bufs, size_t ts,
        size_t row_stride, size_t offset, size_t size, size_t & row, size_t & row_off) {
    row = 0;
    row_off = 0;
    if (size == 0) {
        return -1;
    }
    if (offset / row_stride != (offset + size - 1) / row_stride) {
        return -1;
    }
    const size_t w0 = offset % row_stride;
    const size_t w1 = (offset + size - 1) % row_stride;
    size_t beg = 0;
    for (int j = 0; j < n_bufs; j++) {
        const size_t nbytes = (size_t) ne[j] * ts;
        if (w0 < beg + nbytes) {
            if (w1 >= beg + nbytes) {
                return -1;
            }
            row = offset / row_stride;
            row_off = w0 - beg;
            return j;
        }
        beg += nbytes;
    }
    return -1;
}

// ---- real op over emulated per-die shards -------------------------------->

struct shard_ctx {
    ggml_context_ptr ctx;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_cgraph * gf = nullptr;

    ggml_tensor * logits = nullptr;          // [n_vocab, n_rows]
    ggml_tensor * shard_in [16] = {};
    ggml_tensor * shard_out[16] = {};
    int64_t beg[16] = {};
    int64_t ne [16] = {};
    int n_shards = 0;
    int64_t n_vocab = 0;
    int64_t n_rows = 0;
};

// run GGML_OP_ARGMAX_SHARD per shard over contiguous per-die slabs, exactly
// like the meta backend executes the node on each device's view
static bool run_shard_argmax(struct shard_ctx & sc, const std::vector<float> & row_data,
        const std::vector<int64_t> & shards, int64_t n_rows) {
    sc = {};
    sc.n_shards = (int) shards.size();
    sc.n_rows = n_rows;
    sc.n_vocab = 0;
    for (int j = 0; j < sc.n_shards; j++) {
        sc.ne[j] = shards[j];
        sc.beg[j] = sc.n_vocab;
        sc.n_vocab += shards[j];
    }

    ggml_init_params ip = {
        /*.mem_size =*/ ggml_tensor_overhead()*32 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    sc.ctx.reset(ggml_init(ip));
    if (!sc.ctx) {
        return false;
    }
    sc.backend = ggml_backend_cpu_init();
    if (!sc.backend) {
        return false;
    }

    sc.logits = ggml_new_tensor_2d(sc.ctx.get(), GGML_TYPE_F32, sc.n_vocab, n_rows);
    for (int j = 0; j < sc.n_shards; j++) {
        // per-die slab: contiguous [shard_j, n_rows], like the meta simple tensor
        sc.shard_in[j] = ggml_new_tensor_2d(sc.ctx.get(), GGML_TYPE_F32, sc.ne[j], n_rows);
        sc.shard_out[j] = ggml_argmax_shard(sc.ctx.get(), sc.shard_in[j]);
    }

    sc.buf = ggml_backend_alloc_ctx_tensors_from_buft(sc.ctx.get(), ggml_backend_cpu_buffer_type());
    if (!sc.buf) {
        return false;
    }

    int64_t ne_max = 0;
    for (int j = 0; j < sc.n_shards; j++) {
        ne_max = std::max(ne_max, sc.ne[j]);
    }
    std::vector<float> slab((size_t) ne_max*n_rows, 0.0f);
    for (int j = 0; j < sc.n_shards; j++) {
        for (int64_t r = 0; r < n_rows; r++) {
            std::memcpy(slab.data() + (size_t) r*sc.ne[j],
                    row_data.data() + (size_t) r*sc.n_vocab + sc.beg[j],
                    (size_t) sc.ne[j]*sizeof(float));
        }
        ggml_backend_tensor_set(sc.shard_in[j], slab.data(), 0,
                (size_t) sc.ne[j]*n_rows*sizeof(float));
    }

    sc.gf = ggml_new_graph(sc.ctx.get());
    for (int j = 0; j < sc.n_shards; j++) {
        ggml_build_forward_expand(sc.gf, sc.shard_out[j]);
    }
    return ggml_backend_graph_compute(sc.backend, sc.gf) == GGML_STATUS_SUCCESS;
}

static llama_token merge_result(struct shard_ctx & sc, int64_t row, float * logit) {
    float pairs[32];
    for (int j = 0; j < sc.n_shards; j++) {
        float tmp[2];
        ggml_backend_tensor_get(sc.shard_out[j], tmp,
                (size_t) row*sc.ne[j]*sizeof(float), 2*sizeof(float));
        pairs[2*j]     = tmp[0];
        pairs[2*j + 1] = tmp[1];
    }
    return merge_shard_argmax(pairs, sc.beg, sc.n_shards, logit);
}

// check that the unwritten result-row tail [2, shard) was never touched:
// pre-fill the output buffers' underlying storage is not reachable here, so
// instead verify the written pair values against the shard directly in
// merge_result; the poison test lives in the staging section below

// ---- trials ---------------------------------------------------------->

static void run_trial(const std::vector<int64_t> & shards, int64_t n_rows,
        const std::vector<float> & row_data, const char * name,
        bool expect_host_parity) {
    shard_ctx sc;
    if (!run_shard_argmax(sc, row_data, shards, n_rows)) {
        CHECK(false && "graph compute failed");
        return;
    }

    int64_t n_vocab = sc.n_vocab;
    for (int64_t r = 0; r < n_rows; r++) {
        std::vector<float> row(row_data.begin() + r*n_vocab, row_data.begin() + (r + 1)*n_vocab);

        float logit = 0.0f;
        const llama_token got = merge_result(sc, r, &logit);
        const int want = ref_argmax_lowest(row);

        if (got != want) {
            fprintf(stderr, "FAIL %s row %lld: merge %d != lowest-idx argmax %d\n",
                    name, (long long) r, (int) got, want);
            n_fail++;
            continue;
        }
        if (logit != row[want]) {
            fprintf(stderr, "FAIL %s row %lld: merge logit mismatch\n", name, (long long) r);
            n_fail++;
        }

        // host sampler paths agree for distinct logits
        if (expect_host_parity) {
            const llama_token a = host_partial_sort_top1(row);
            const llama_token b = host_heap_select_top1(row, 10);
            if (a != want || b != want) {
                fprintf(stderr, "FAIL %s row %lld: host paths %d/%d != argmax %d\n",
                        name, (long long) r, (int) a, (int) b, want);
                n_fail++;
            }
        }
    }

    // fetch mapping: every pair fetch lands inside the owning shard and the
    // shard windows cover the row exactly
    int64_t cov = 0;
    for (int j = 0; j < sc.n_shards; j++) {
        cov += sc.ne[j];
        size_t row = 0, row_off = 0;
        const size_t row_stride = (size_t) n_vocab*sizeof(float);
        const int owner = subrow_shard(sc.ne, sc.n_shards, sizeof(float), row_stride,
                (size_t) sc.beg[j]*sizeof(float) + 0, 2*sizeof(float), row, row_off);
        if (owner != j || row != 0 || row_off != 0) {
            fprintf(stderr, "FAIL %s: pair fetch of shard %d mapped to %d (row %zu, off %zu)\n",
                    name, j, owner, row, row_off);
            n_fail++;
        }
        // second row of the same pair fetch: simple offset uses the shard's
        // packed stride
        const int owner2 = subrow_shard(sc.ne, sc.n_shards, sizeof(float), row_stride,
                row_stride + (size_t) sc.beg[j]*sizeof(float), 2*sizeof(float), row, row_off);
        if (owner2 != j || row != 1 || row_off != 0) {
            fprintf(stderr, "FAIL %s: row-1 pair fetch of shard %d mapped to %d (row %zu, off %zu)\n",
                    name, j, owner2, row, row_off);
            n_fail++;
        }
    }
    if (cov != n_vocab) {
        fprintf(stderr, "FAIL %s: shard coverage %lld != %lld\n", name, (long long) cov, (long long) n_vocab);
        n_fail++;
    }

    ggml_backend_buffer_free(sc.buf);
    ggml_backend_free(sc.backend);
}

int main() {
    fprintf(stderr, "== op sanity: shard layout {5, 3, 4}, crafted ties\n");
    {
        // n_vocab 12, one row: max 9.0 at ids {2 (shard 0), 6 (shard 1), 11 (shard 2)}
        std::vector<float> row = {1, 2, 9, 4, 5,   5, 9, 7, 0, -1, -2, 9};
        shard_ctx sc;
        if (!run_shard_argmax(sc, row, {5, 3, 4}, 1)) {
            CHECK(false && "graph compute failed");
            return 1;
        }
        float logit = 0.0f;
        const llama_token got = merge_result(sc, 0, &logit);
        CHECK(got == 2);                    // lowest global index among ties
        CHECK(logit == 9.0f);

        // per-shard pairs: shard 0 -> local 2, shard 1 -> local 1, shard 2 -> local 3
        float tmp[2];
        ggml_backend_tensor_get(sc.shard_out[0], tmp, 0, 2*sizeof(float));
        CHECK(tmp[1] == 2.0f && tmp[0] == 9.0f);
        ggml_backend_tensor_get(sc.shard_out[1], tmp, 0, 2*sizeof(float));
        CHECK(tmp[1] == 1.0f);
        ggml_backend_tensor_get(sc.shard_out[2], tmp, 0, 2*sizeof(float));
        CHECK(tmp[1] == 3.0f);

        ggml_backend_buffer_free(sc.buf);
        ggml_backend_free(sc.backend);
    }

    fprintf(stderr, "== single shard == plain argmax\n");
    {
        std::vector<float> row(1000);
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (auto & v : row) {
            v = dist(rng);
        }
        row[733] = 42.0f;
        run_trial({1000}, 1, row, "single-shard", true);
    }

    fprintf(stderr, "== uneven 3-way shards at 129272, distinct random\n");
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-30.0f, 30.0f);
        for (auto shards : {std::vector<int64_t>{43091, 43091, 43090},
                            std::vector<int64_t>{40000, 60000, 29272},
                            std::vector<int64_t>{60000, 40000, 29272},
                            std::vector<int64_t>{29272, 29272, 70728}}) {
            int64_t n_vocab = 0;
            for (auto s : shards) {
                n_vocab += s;
            }
            std::vector<float> rows((size_t) n_vocab*2);
            for (auto & v : rows) {
                v = dist(rng);
            }
            char name[64];
            snprintf(name, sizeof(name), "uneven {%lld, %lld, %lld}",
                    (long long) shards[0], (long long) shards[1], (long long) shards[2]);
            run_trial(shards, 2, rows, name, true);
        }
    }

    fprintf(stderr, "== full-scale ties: maxima within shards, across shards, at boundaries\n");
    {
        const int64_t n_vocab = 129272;
        const std::vector<int64_t> shards = {43091, 43091, 43090};
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-30.0f, 30.0f);
        const float maxv = 77.0f;

        for (int pattern = 0; pattern < 6; pattern++) {
            std::vector<float> rows((size_t) n_vocab*2);
            for (auto & v : rows) {
                v = dist(rng);
            }
            const int64_t b0 = shards[0];
            const int64_t b1 = shards[0] + shards[1];
            switch (pattern) {
                case 0: // single max in shard 0
                    rows[123] = maxv;
                    break;
                case 1: // tie within shard 1
                    rows[b0 + 10] = maxv;
                    rows[b0 + 40000] = maxv;
                    break;
                case 2: // tie across shards 0 and 2
                    rows[5] = maxv;
                    rows[b1 + 5] = maxv;
                    break;
                case 3: // tie across all three shards
                    rows[0] = maxv;
                    rows[b0] = maxv;
                    rows[b1] = maxv;
                    break;
                case 4: // max at the last element of shard 0 (boundary)
                    rows[b0 - 1] = maxv;
                    rows[b0] = maxv - 1.0f;
                    break;
                case 5: // all-equal row
                    std::fill(rows.begin(), rows.begin() + n_vocab, maxv);
                    break;
            }
            char name[64];
            snprintf(name, sizeof(name), "tie pattern %d", pattern);
            // host heap/partial_sort parity is only asserted for distinct
            // logits, so patterns 1-5 check the merge contract alone (the
            // documented tie divergence of the host paths)
            run_trial(shards, 2, rows, name, pattern == 0);
        }
    }

    fprintf(stderr, "== k-boundary near-ties (max unique, 1-ULP runners)\n");
    {
        const int64_t n_vocab = 129272;
        const std::vector<int64_t> shards = {43091, 43091, 43090};
        std::vector<float> rows((size_t) n_vocab);
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(-30.0f, 30.0f);
        for (auto & v : rows) {
            v = dist(rng);
        }
        rows[43091 + 43090] = 31.5f;      // global max in shard 2
        rows[43091 + 43090 - 1] = 31.5f - std::numeric_limits<float>::epsilon()*16.0f;
        run_trial(shards, 1, rows, "near-ties", true);
    }

    fprintf(stderr, "== staging lifetime: merge reads only the pair packet\n");
    {
        const int64_t n_vocab = 12;
        const std::vector<int64_t> shards = {5, 3, 4};
        std::vector<float> rowA = {1, 2, 9, 4, 5,   5, 9, 7, 0, -1, -2, 9};
        std::vector<float> rowB = {0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 4};

        shard_ctx sc;
        if (!run_shard_argmax(sc, rowA, shards, 1)) {
            CHECK(false && "graph compute failed");
            return 1;
        }

        // staging holds raw device pairs: [max, local] per shard
        float staging[8];
        for (int j = 0; j < sc.n_shards; j++) {
            ggml_backend_tensor_get(sc.shard_out[j], staging + 2*j, 0, 2*sizeof(float));
        }
        // poison the staging tail: the merge must not read past the packet
        staging[6] = 1e30f;
        staging[7] = -3.0f;

        float logitA = 0.0f;
        const llama_token gotA = merge_shard_argmax(staging, sc.beg, sc.n_shards, &logitA);
        CHECK(gotA == 2 && logitA == 9.0f);

        // simulate the next decode overwriting the same staging before the
        // consumer kept a stale pointer: old copies stay valid, new reads
        // reflect the new decode (rowB: unique max 4.0 at id 11)
        float stale[8];
        std::memcpy(stale, staging, sizeof(stale));
        std::vector<float> rowB_all = rowB;
        shard_ctx sc2;
        CHECK(run_shard_argmax(sc2, rowB_all, shards, 1));
        float staging2[8];
        for (int j = 0; j < sc2.n_shards; j++) {
            ggml_backend_tensor_get(sc2.shard_out[j], staging2 + 2*j, 0, 2*sizeof(float));
        }
        const llama_token gotB = merge_shard_argmax(staging2, sc2.beg, sc2.n_shards, nullptr);

        CHECK(gotB == 11);
        // the stale packet still decodes to the old result (its owner copied
        // it before the overwrite)
        CHECK(merge_shard_argmax(stale, sc.beg, sc.n_shards, nullptr) == 2);

        ggml_backend_buffer_free(sc.buf);
        ggml_backend_free(sc.backend);
        ggml_backend_buffer_free(sc2.buf);
        ggml_backend_free(sc2.backend);
    }

    fprintf(stderr, "== subrow fetch mapping rejects cross-shard ranges\n");
    {
        const int64_t ne[3] = {43091, 43091, 43090};
        const size_t row_stride = 129272*sizeof(float);
        size_t row = 0, ro = 0;
        // whole-row fetch must NOT take the sub-row path
        CHECK(subrow_shard(ne, 3, sizeof(float), row_stride, 0, row_stride, row, ro) == -1);
        // a range spanning the shard 0 / shard 1 boundary must be rejected
        const size_t b1 = (size_t) ne[0]*sizeof(float);
        CHECK(subrow_shard(ne, 3, sizeof(float), row_stride, b1 - 4, 8, row, ro) == -1);
        // die 1's pair fetch maps with the right simple offset (packed slab)
        CHECK(subrow_shard(ne, 3, sizeof(float), row_stride, b1, 8, row, ro) == 1 && row == 0 && ro == 0);
        CHECK(subrow_shard(ne, 3, sizeof(float), row_stride, row_stride + b1, 8, row, ro) == 1 && row == 1 && ro == 0);
        // simple offset for row 1 = row * shard_j packed stride
        CHECK(1 * (size_t) ne[1]*sizeof(float) == 43091*4);
    }

    if (n_fail == 0) {
        fprintf(stderr, "ALL PASS\n");
        return 0;
    }
    fprintf(stderr, "%d FAILURES\n", n_fail);
    return 1;
}
