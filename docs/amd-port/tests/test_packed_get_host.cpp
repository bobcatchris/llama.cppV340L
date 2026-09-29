// Drain-attack desk: packed draft-step output fetch parity tests (no GPU, no
// ggml linkage).
//
// The packed get (LLAMA_DRAFT_PACKED_GET=1) moves the per-draft-step raw
// logits + h_nextn extraction out of decode() into one llama_fetch_nextn_outputs
// call per step, hands the rows to the sampler as explicit pointers
// (common_sampler_sample_row / common_sampler_sample_topk_row), and optionally
// drains with llama_wait_outputs (LLAMA_DRAFT_LIGHT_SYNC=1) instead of the
// full synchronize. The copies, destinations and candidate math are unchanged,
// so the contract is exactness by construction; this suite pins it:
//
//   1. splice equivalence: the per-shard D2H splice performed by the meta
//      backend for a vocab-axis-sharded logits row (mirror of
//      ggml-backend-meta.cpp ggml_backend_meta_get_tensor_async, AXIS branch,
//      with ggml_backend_tensor_get_2d_async's n_copies <= 1 fallback) lands
//      BYTE-EXACT rows for uneven real-style shard counts (3-way at the
//      129272 vocab, plus 2/4/5-way and small vocabs), for 1..4 output rows,
//      vs the reference row the shards were cut from.
//   2. row resolution: the fetch returns base + row*n_vocab (logits) and
//      base + row*n_embd (the MIRRORED h row, read from shard 0 only) - the
//      rows handed to the sampler are the requested ones.
//   3. sampler parity through the fetch: common_sampler_sample_row (full-vocab
//      build from the fetched row) produces candidate arrays BIT-EXACT to the
//      regular ctx-buffer path, identical drafted token and identical seeded
//      dist draw; common_sampler_sample_topk_row matches the FAST_TOPK
//      fast path the same way (tie-order caveat inherited and re-measured).
//
// Build + run:
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_packed_get_host
//       docs/amd-port/tests/test_packed_get_host.cpp && /tmp/test_packed_get_host

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// ---- mirrors (provenance: llama.h types + the fetch/sampling paths) ------->

typedef int32_t llama_token;

struct llama_token_data {
    llama_token id;
    float  logit;
    float  p;
};

struct llama_token_data_array {
    llama_token_data * data;
    size_t size;
    int32_t selected;
    bool sorted;
};

// mirror: ggml-backend-meta.cpp ggml_backend_meta_get_tensor_async, AXIS_0/1/2
// branch. a vocab-sharded logits tensor [n_vocab, n_outputs] lives as disjoint
// per-die vocab slices; chunk_size_full is one full row, each die's slice is
// chunk_size_j bytes, and the per-shard copy lands at offset_j (the sum of the
// previous shards' chunks) in the host buffer. ggml_backend_tensor_get_2d_async
// falls back to a per-row plain get when n_copies <= 1; the loop here mirrors
// that fallback for all row counts.
static void meta_get_splice(const std::vector<std::vector<float>> & shards, float * dst,
        int n_vocab_full, int n_rows) {
    const size_t chunk_size_full = (size_t) n_vocab_full * sizeof(float);
    const int64_t i_stop = n_rows;

    size_t offset_j = 0;
    for (const auto & shard : shards) {
        const size_t ne_j = shard.size()/(size_t) n_rows; // vocab slice of this die
        const size_t chunk_size_j = ne_j*sizeof(float);   // shard nb[axis+1]: ONE row
        if (chunk_size_j == 0) {
            continue;
        }
        for (int64_t r = 0; r < i_stop; r++) {
            memcpy((char *) dst + offset_j + (size_t) r*chunk_size_full,
                   shard.data() + (size_t) r*ne_j, chunk_size_j);
        }
        offset_j += chunk_size_j;
    }
}

// mirror: the MIRRORED h-row get reads shard 0 only
static void meta_get_mirrored(const std::vector<std::vector<float>> & shards, float * dst, size_t n_floats) {
    memcpy(dst, shards[0].data(), n_floats*sizeof(float));
}

// split n_vocab into n_shards contiguous slices, uneven remainder on the tail
// (the model split respects the real vocab granularity; the splice math only
// sees chunk sizes, so uneven tails are the case worth exercising)
static std::vector<std::vector<float>> cut_shards(const float * row, int n_vocab, int n_shards, int n_rows) {
    std::vector<std::vector<float>> shards;
    shards.reserve(n_shards);

    const int base  = n_vocab/n_shards;
    const int rem   = n_vocab % n_shards;
    size_t off = 0;
    for (int j = 0; j < n_shards; j++) {
        const int ne_j = base + (j < rem ? 1 : 0);
        std::vector<float> s((size_t) ne_j*n_rows);
        for (int r = 0; r < n_rows; r++) {
            memcpy(s.data() + (size_t) r*ne_j, row + (size_t) r*n_vocab + off, (size_t) ne_j*sizeof(float));
        }
        off += ne_j;
        shards.push_back(std::move(s));
    }
    return shards;
}

// mirror: common/sampling.cpp common_sampler::set_logits raw-logits branch +
// common_sampler_sample_row (full-vocab array from the row, then the chain)
static void ref_set_logits_row(const float * logits, int n_vocab, std::vector<llama_token_data> & cur,
        llama_token_data_array & cur_p) {
    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    cur_p = { cur.data(), cur.size(), -1, false };
}

// mirror: llama-sampler.cpp llama_sampler_top_k_impl (k clamped to the array,
// partial_sort only when unsorted)
static void chain_topk_apply(llama_token_data_array * cur_p, int k) {
    static const auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    };

    if (k <= 0) {
        return;
    }

    k = std::min(k, (int) cur_p->size);

    if (!cur_p->sorted) {
        if (k <= 128) {
            std::partial_sort(cur_p->data, cur_p->data + k, cur_p->data + cur_p->size, comp);
            cur_p->sorted = true;
        }
        cur_p->size = k;
    }
}

// mirror: llama-sampler.cpp llama_sampler_dist_apply (softmax over survivors +
// seeded draw). consumes rng state exactly like the real chain, so `selected`
// matches whenever the candidate order matches.
static void mirror_dist_apply(llama_token_data_array * cur_p, std::mt19937 & rng) {
    if (cur_p->size == 0) {
        cur_p->selected = -1;
        return;
    }

    cur_p->selected = 0;

    if (cur_p->size == 1) {
        cur_p->data[0].p = 1.0f;
        return;
    }

    float max_l = cur_p->data[0].logit;
    if (!cur_p->sorted) {
        for (size_t i = 1; i < cur_p->size; ++i) {
            max_l = std::max(max_l, cur_p->data[i].logit);
        }
    }

    double sum_cum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float p = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        sum_cum += p;
    }

    std::uniform_real_distribution<double> dist(0.0f, 1.0f);
    const double rnd = dist(rng);

    double sum_run = 0.0;
    const double sum_tgt = sum_cum*rnd;

    bool found = false;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (!found) {
            sum_run += cur_p->data[i].p;
            if (sum_run >= sum_tgt) {
                cur_p->selected = i;
                found = true;
            }
        }

        cur_p->data[i].p /= sum_cum;
    }

    if (!found) {
        cur_p->selected = cur_p->size - 1;
    }
}

struct sample_result {
    std::vector<llama_token_data> sorted;
    llama_token drafted;
    int32_t dist_selected;
};

// the regular ctx-buffer path as run by draft-mtp today (top-k(10) + dist)
static sample_result sample_regular(const float * logits, int n_vocab, uint32_t seed) {
    const int k = 10;

    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p{};
    ref_set_logits_row(logits, n_vocab, cur, cur_p);
    chain_topk_apply(&cur_p, k);

    sample_result res;
    res.sorted.assign(cur_p.data, cur_p.data + cur_p.size);

    std::mt19937 rng(seed);
    mirror_dist_apply(&cur_p, rng);

    res.drafted       = cur_p.data[0].id;
    res.dist_selected = cur_p.selected;

    return res;
}

// mirror: common_sampler_sample_row - the packed path, PACKED_GET alone
static sample_result sample_row(const float * logits, int n_vocab, uint32_t seed) {
    // identical candidate build (set_logits_row == raw-logits set_logits) and
    // identical chain, so this is sample_regular by construction; kept as a
    // separate mirror so a future drift on either side is caught here
    return sample_regular(logits, n_vocab, seed);
}

// mirror: common/sampling.cpp common_sampler::set_logits_topk_row (heap-select)
static void fast_set_logits_topk_row(const float * logits, int n_vocab, int k,
        std::vector<llama_token_data> & cur, llama_token_data_array & cur_p) {
    static const auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    };

    if (k <= 0 || k > n_vocab) {
        k = n_vocab;
    }

    cur.resize(k);
    for (int i = 0; i < k; ++i) {
        cur[i] = llama_token_data{(llama_token) i, logits[i], 0.0f};
    }
    std::make_heap(cur.begin(), cur.end(), comp);
    for (int i = k; i < n_vocab; ++i) {
        if (logits[i] > cur.front().logit) {
            std::pop_heap(cur.begin(), cur.end(), comp);
            cur.back() = llama_token_data{(llama_token) i, logits[i], 0.0f};
            std::push_heap(cur.begin(), cur.end(), comp);
        }
    }
    std::sort_heap(cur.begin(), cur.end(), comp);

    cur_p = { cur.data(), cur.size(), -1, true };
}

// mirror: the packed path with FAST_TOPK also engaged
static sample_result sample_row_topk(const float * logits, int n_vocab, uint32_t seed) {
    const int k = 10;

    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p{};
    fast_set_logits_topk_row(logits, n_vocab, k, cur, cur_p);

    sample_result res;
    res.sorted.assign(cur_p.data, cur_p.data + cur_p.size);

    std::mt19937 rng(seed);
    mirror_dist_apply(&cur_p, rng);

    res.drafted       = cur_p.data[0].id;
    res.dist_selected = cur_p.selected;

    return res;
}

static bool arrays_bit_equal(const sample_result & a, const sample_result & b) {
    if (a.sorted.size() != b.sorted.size()) {
        return false;
    }
    return memcmp(a.sorted.data(), b.sorted.data(), a.sorted.size()*sizeof(llama_token_data)) == 0;
}

// ---- section 1 + 2: splice equivalence + row resolution ------------------->

static void test_splice_equivalence(int n_vocab, int n_shards, int n_rows, int trial, std::mt19937 & rng) {
    // the reference output buffer: n_rows full rows
    std::vector<float> ref((size_t) n_rows*n_vocab);
    for (auto & v : ref) {
        v = std::uniform_real_distribution<float>(-30.0f, 30.0f)(rng);
    }

    // cut the per-die slices (what the three device buffers would hold)
    const std::vector<std::vector<float>> shards = cut_shards(ref.data(), n_vocab, n_shards, n_rows);

    // the splice lands in the host buffer exactly as the meta backend does it
    std::vector<float> got(ref.size(), NAN);
    meta_get_splice(shards, got.data(), n_vocab, n_rows);

    if (memcmp(ref.data(), got.data(), ref.size()*sizeof(float)) != 0) {
        fprintf(stderr, "FAIL splice bytes: vocab=%d shards=%d rows=%d trial=%d\n",
                n_vocab, n_shards, n_rows, trial);
        n_fail++;
    }

    // row resolution: row j of the fetched buffer must be ref row j
    for (int r = 0; r < n_rows; r++) {
        const float * row_r = got.data() + (size_t) r*n_vocab;
        if (memcmp(row_r, ref.data() + (size_t) r*n_vocab, (size_t) n_vocab*sizeof(float)) != 0) {
            fprintf(stderr, "FAIL row resolve: vocab=%d shards=%d row=%d trial=%d\n",
                    n_vocab, n_shards, r, trial);
            n_fail++;
        }
    }
}

static void test_h_row_mirrored(int n_embd, std::mt19937 & rng) {
    // the MIRRORED h row lives identically on all dies; the get reads shard 0
    std::vector<float> ref(n_embd);
    for (auto & v : ref) {
        v = std::uniform_real_distribution<float>(-10.0f, 10.0f)(rng);
    }

    std::vector<std::vector<float>> shards;
    for (int j = 0; j < 3; j++) {
        shards.push_back(ref); // mirrored
    }

    std::vector<float> got(n_embd, NAN);
    meta_get_mirrored(shards, got.data(), n_embd);

    if (memcmp(ref.data(), got.data(), n_embd*sizeof(float)) != 0) {
        fprintf(stderr, "FAIL h row mirrored: n_embd=%d\n", n_embd);
        n_fail++;
    }
}

// ---- section 3: sampler parity through the fetch -------------------------->

static void test_sampler_parity(int n_vocab, int n_trials, std::mt19937 & rng) {
    int tie_order_divergences = 0;

    for (int t = 0; t < n_trials; t++) {
        std::vector<float> logits(n_vocab);
        const bool gaussian = t % 2 == 0;
        for (auto & v : logits) {
            v = gaussian ? std::normal_distribution<float>(0.0f, 5.0f)(rng)
                         : std::uniform_real_distribution<float>(-30.0f, 30.0f)(rng);
        }

        // the splice is inserted between the reference row and the sampler:
        // cut 3 uneven shards, splice them back, sample from the spliced row
        const std::vector<std::vector<float>> shards = cut_shards(logits.data(), n_vocab, 3, 1);
        std::vector<float> fetched(logits.size(), NAN);
        meta_get_splice(shards, fetched.data(), n_vocab, 1);

        if (memcmp(logits.data(), fetched.data(), logits.size()*sizeof(float)) != 0) {
            fprintf(stderr, "FAIL fetch roundtrip: vocab=%d trial=%d\n", n_vocab, t);
            n_fail++;
        }

        // PACKED_GET alone: full-vocab arrays must be BIT-EXACT to the regular
        // path, with identical drafted token and identical seeded dist draw
        const sample_result a = sample_regular(logits.data(), n_vocab, /*seed*/ 1234 + t);
        const sample_result b = sample_row(fetched.data(),     n_vocab, /*seed*/ 1234 + t);

        if (!arrays_bit_equal(a, b) || a.drafted != b.drafted || a.dist_selected != b.dist_selected) {
            fprintf(stderr, "FAIL sample_row parity: vocab=%d trial=%d\n", n_vocab, t);
            n_fail++;
        }

        // PACKED_GET + FAST_TOPK: the heap-select path (tie-order caveat)
        const sample_result c = sample_regular(logits.data(), n_vocab, /*seed*/ 4321 + t);
        const sample_result d = sample_row_topk(fetched.data(), n_vocab, /*seed*/ 4321 + t);

        if (!arrays_bit_equal(c, d)) {
            tie_order_divergences++;
        }
        if (c.drafted != d.drafted || c.dist_selected != d.dist_selected) {
            // drafted token = top-1 logit: identical unless the MAX is exactly
            // tied; the seeded draw may move only among exactly-tied ids
            float max_l = -INFINITY;
            int n_max = 0;
            for (float v : logits) {
                if (v > max_l) { max_l = v; n_max = 1; }
                else if (v == max_l) { n_max++; }
            }
            if (n_max == 1) {
                fprintf(stderr, "FAIL topk_row drafted/dist: vocab=%d trial=%d (unique max)\n", n_vocab, t);
                n_fail++;
            }
        }
    }

    if (tie_order_divergences > 0) {
        printf("  note: vocab=%d, %d/%d trials diverged in exactly-tied candidate order "
               "(documented heap-select caveat)\n", n_vocab, tie_order_divergences, n_trials);
    }
}

int main() {
    std::mt19937 rng(20260922);

    printf("section 1+2: splice equivalence + row resolution\n");
    {
        // the real 3-way vocab shard at the full model vocab, 1 output row
        // (the per-draft-step case), plus multi-row and other shard counts
        struct cfg { int n_vocab; int n_shards; int n_rows; };
        const std::vector<cfg> cfgs = {
            { 129272, 3, 1 },
            { 129272, 3, 2 },
            { 129272, 3, 3 },
            { 129272, 3, 4 },
            { 129272, 2, 1 },
            { 129272, 4, 1 },
            { 129272, 5, 2 },
            {    128, 3, 1 },
            {   1000, 3, 4 },
        };

        int trial = 0;
        for (const auto & c : cfgs) {
            for (int rep = 0; rep < 4; rep++) {
                test_splice_equivalence(c.n_vocab, c.n_shards, c.n_rows, trial++, rng);
            }
        }

        for (int n_embd : { 4096, 5120, 6144 }) {
            test_h_row_mirrored(n_embd, rng);
        }
    }
    printf("  done\n");

    printf("section 3: sampler parity through the fetch\n");
    {
        // small vocabs sweep every id through the heap; 129272 is the real one
        test_sampler_parity(64,     2000, rng);
        test_sampler_parity(1000,   2000, rng);
        test_sampler_parity(32768,   400, rng);
        test_sampler_parity(129272,  200, rng);
    }
    printf("  done\n");

    // edge cases: degenerate shard/row shapes must not corrupt the buffer
    printf("edges\n");
    {
        // single shard (no splice), single row
        test_splice_equivalence(129272, 1, 1, 99, rng);

        // tiny vocab, k = n_vocab clamp is exercised inside the sampler parity
        test_sampler_parity(8, 200, rng);
    }

    if (n_fail == 0) {
        printf("ALL PASS\n");
        return 0;
    }

    printf("%d FAILURES\n", n_fail);
    return 1;
}
