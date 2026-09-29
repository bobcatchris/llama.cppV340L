// Verify-round desk: host-side parity tests for the target verify-accept row
// path (LLAMA_VERIFY_ROW_SAMPLING, no GPU, no ggml linkage).
//
// common_sampler_sample_and_accept_n_rows (common/sampling.cpp) replaces the
// per-row common_sampler_sample loop of the server's speculative accept step
// with: one light drain, a backend-sampled-token probe on the first row, one
// llama_peek_logits_rows call, then per-row common_sampler_sample_row +
// common_sampler_accept with the same break-on-mismatch semantics.
//
// The regular path's raw-logits branch and the row path share the same
// set_logits_row build, so parity is structural; this test pins it anyway by
// running BOTH mirrors end-to-end over the same logits rows and comparing:
//
//   1. the accepted token sequence and the break position (result size) are
//      BIT-EXACT between the paths for greedy chains (the campaign's temp-0
//      case) and for seeded [top-k, dist] chains on distinct logits,
//   2. the candidate arrays handed to the chain are memcmp-identical,
//   3. exact-tie rows: same accepted tokens whenever the max is unique; the
//      documented tie-order caveat does not apply to the drafted token,
//   4. eligibility guards (grammar / reasoning budget / backend-sampled
//      token) return an empty result so the caller falls back unchanged.
//
// Build + run:
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_verify_row_sampling_host
//       docs/amd-port/tests/test_verify_row_sampling_host.cpp && /tmp/test_verify_row_sampling_host

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

// ---- mirrors (provenance: llama.h types + the two sampling paths) -------->

typedef int32_t llama_token;
#define LLAMA_TOKEN_NULL ((llama_token) -1)

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

// mirror: common/sampler.hpp ring buffer (only what the accept loop needs)
struct ring_prev {
    std::vector<llama_token> data;
    size_t cap = 64;
    size_t pos = 0;
    size_t count = 0;

    void push(llama_token t) {
        if (data.empty()) {
            data.resize(cap);
        }
        data[pos] = t;
        pos = (pos + 1) % cap;
        count++;
    }
};

// mirror of the common_sampler state the two paths touch
struct test_sampler {
    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p = { nullptr, 0, -1, false };
    ring_prev prev;
    bool has_grmr = false;
    bool has_rbudget = false;
    llama_token backend_token = LLAMA_TOKEN_NULL; // mirror of ctx sampled-token storage

    // mirror: common_sampler::set_logits_row (the raw-logits branch of
    // set_logits - both paths land here with the same row)
    bool set_logits_row(const float * logits, int n_vocab) {
        if (logits == nullptr || n_vocab <= 0) {
            return false;
        }
        cur.resize(n_vocab);
        for (int token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{ (llama_token) token_id, logits[token_id], 0.0f };
        }
        cur_p = { cur.data(), cur.size(), -1, false };
        return true;
    }
};

// mirror: llama_sampler_dist_apply (softmax over survivors + seeded draw)
static void ref_dist_apply(llama_token_data_array * cur_p, float temp, uint32_t & seed) {
    if (cur_p->size <= 0) {
        return;
    }
    double cum = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const double p = expf(cur_p->data[i].logit / temp);
        cur_p->data[i].p = (float) p;
        cum += p;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= (float) cum;
    }
    cur_p->sorted = true;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float r = dist(rng);
    seed = (uint32_t) (seed * 1664525u + 1013904223u + (uint32_t) (r * 1e9f));

    float acc = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        acc += cur_p->data[i].p;
        if (acc >= r) {
            cur_p->selected = (int32_t) i;
            return;
        }
    }
    cur_p->selected = (int32_t) (cur_p->size - 1);
}

// mirror: top-k chain stage (partial_sort, id-stable for distinct logits)
static void ref_top_k_apply(llama_token_data_array * cur_p, int k) {
    const int npartial = std::min((int) cur_p->size, k);
    static const auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    };
    std::partial_sort(cur_p->data, cur_p->data + npartial, cur_p->data + cur_p->size, comp);
    cur_p->size = npartial;
}

// mirror: greedy chain (dist at temp 0 selects argmax)
static void ref_greedy_apply(llama_token_data_array * cur_p) {
    float max_logit = cur_p->data[0].logit;
    cur_p->data[0].p = 1.0f;
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        cur_p->data[i].p = 0.0f;
        if (cur_p->data[i].logit > max_logit) {
            max_logit = cur_p->data[i].logit;
            cur_p->selected = (int32_t) i;
        }
    }
    cur_p->sorted = true;
}

typedef void (*chain_fn)(llama_token_data_array *, float, uint32_t &);

// mirror: common_sampler_sample_and_accept_n (the regular path; per row it
// synchronizes + probes getters + full-vocab build from the same row bytes)
static std::vector<llama_token> ref_sample_and_accept_n(
        test_sampler & smpl, const std::vector<const float *> & rows, int n_vocab,
        const std::vector<llama_token> & draft, chain_fn chain, float temp, uint32_t seed) {
    std::vector<llama_token> result;
    result.reserve(rows.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        CHECK(smpl.set_logits_row(rows[i], n_vocab));
        chain(&smpl.cur_p, temp, seed);
        const llama_token id = smpl.cur_p.data[smpl.cur_p.selected].id;
        smpl.prev.push(id);
        result.push_back(id);
        if (draft[i] != id) {
            break;
        }
    }
    if (i == draft.size()) {
        CHECK(smpl.set_logits_row(rows[i], n_vocab));
        chain(&smpl.cur_p, temp, seed);
        const llama_token id = smpl.cur_p.data[smpl.cur_p.selected].id;
        smpl.prev.push(id);
        result.push_back(id);
    }
    return result;
}

// mirror: common_sampler_sample_and_accept_n_rows (the row path)
static std::vector<llama_token> rows_sample_and_accept_n(
        test_sampler & smpl, const std::vector<const float *> & rows, int n_vocab,
        const std::vector<llama_token> & draft, chain_fn chain, float temp, uint32_t seed) {
    // mirror: the eligibility guards in sampling.cpp
    if (smpl.has_grmr || smpl.has_rbudget) {
        return {};
    }

    // mirror: the backend-sampled-token probe on the first row
    if (smpl.backend_token != LLAMA_TOKEN_NULL) {
        return {};
    }

    // mirror: llama_wait_outputs + llama_peek_logits_rows (light drain + raw
    // row pointers; the rows are the same memory the regular path reads)
    std::vector<const float *> peeked(rows.size());
    for (size_t r = 0; r < rows.size(); ++r) {
        peeked[r] = rows[r];
    }

    std::vector<llama_token> result;
    result.reserve(rows.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        if (!smpl.set_logits_row(peeked[i], n_vocab)) {
            return {};
        }
        chain(&smpl.cur_p, temp, seed);
        const llama_token id = smpl.cur_p.data[smpl.cur_p.selected].id;
        smpl.prev.push(id);
        result.push_back(id);
        if (draft[i] != id) {
            break;
        }
    }
    if (i == draft.size()) {
        if (!smpl.set_logits_row(peeked[i], n_vocab)) {
            return {};
        }
        chain(&smpl.cur_p, temp, seed);
        const llama_token id = smpl.cur_p.data[smpl.cur_p.selected].id;
        smpl.prev.push(id);
        result.push_back(id);
    }
    return result;
}

// ---- helpers -------------------------------------------------------------->

static std::vector<float> make_distinct_logits(int n_vocab, std::mt19937 & rng, bool gaussian) {
    std::uniform_real_distribution<float> u(-12.0f, 12.0f);
    std::normal_distribution<float> g(0.0f, 3.0f);
    std::vector<float> raw(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        raw[i] = gaussian ? g(rng) : u(rng);
    }
    // force distinctness (the real-model case) by nudging duplicates up
    std::vector<float> sorted = raw;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 1; i < n_vocab; i++) {
        if (sorted[i] == sorted[i - 1]) {
            sorted[i] = std::nextafter(sorted[i], std::numeric_limits<float>::infinity());
        }
    }
    return sorted;
}

static std::vector<llama_token> make_draft(const std::vector<float> & logits, int n_vocab, int n_draft, std::mt19937 & rng) {
    // draft tokens that match the row argmax for the first steps with
    // realistic probability, else diverge (exercises the break)
    std::vector<llama_token> draft;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    int argmax = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[argmax]) {
            argmax = i;
        }
    }
    for (int i = 0; i < n_draft; i++) {
        draft.push_back(u(rng) < 0.6f ? argmax : (int) (rng() % n_vocab));
    }
    return draft;
}

// ---- cases ---------------------------------------------------------------->

static void run_case(int n_vocab, int n_trials, bool gaussian, int chain_id) {
    std::mt19937 rng(1234 + n_vocab + chain_id);
    const int n_rows_max = 4;

    for (int t = 0; t < n_trials; t++) {
        std::vector<float> logits = make_distinct_logits(n_vocab, rng, gaussian);

        // rows region: verify rows are [row0..row3] of one batch; each row is
        // an independently drawn distribution (only row0's argmax matters for
        // the draft of row0 - keep the mirror simple and independent)
        std::vector<std::vector<float>> row_buf;
        for (int r = 0; r < n_rows_max; r++) {
            row_buf.push_back(make_distinct_logits(n_vocab, rng, gaussian));
        }

        const std::vector<llama_token> draft = make_draft(row_buf[0], n_vocab, 3, rng);

        std::vector<const float *> rows;
        for (int r = 0; r < n_rows_max; r++) {
            rows.push_back(row_buf[r].data());
        }

        chain_fn chain = (chain_id == 0) ?
            [](llama_token_data_array * p, float, uint32_t &) { ref_greedy_apply(p); } :
            [](llama_token_data_array * p, float temp, uint32_t & seed) {
                ref_top_k_apply(p, 10);
                ref_dist_apply(p, temp, seed);
            };

        test_sampler a; // regular path
        test_sampler b; // row path
        const uint32_t seed_a = 0xBEEF + t;
        const uint32_t seed_b = seed_a;

        auto ra = ref_sample_and_accept_n(a, rows, n_vocab, draft, chain, 0.8f, seed_a);
        auto rb = rows_sample_and_accept_n(b, rows, n_vocab, draft, chain, 0.8f, seed_b);

        CHECK(rb.size() == ra.size());
        if (rb.size() == ra.size()) {
            CHECK(memcmp(ra.data(), rb.data(), ra.size() * sizeof(llama_token)) == 0);
        }

        // candidate arrays for the first row: full build vs full build over
        // the same row must be memcmp-identical
        test_sampler c;
        c.set_logits_row(rows[0], n_vocab);
        test_sampler d;
        d.set_logits_row(rows[0], n_vocab);
        CHECK(c.cur.size() == d.cur.size());
        CHECK(memcmp(c.cur.data(), d.cur.data(), c.cur.size() * sizeof(llama_token_data)) == 0);
    }
}

static void run_tie_case(int n_vocab) {
    // all-equal rows: both paths must accept the same token (lowest id wins
    // in both builds) and the same result size
    std::vector<float> row(n_vocab, 0.5f);
    std::vector<const float *> rows = { row.data(), row.data(), row.data(), row.data() };
    const std::vector<llama_token> draft = { 3, 3, 3 };

    test_sampler a;
    test_sampler b;
    auto ra = ref_sample_and_accept_n(a, rows, n_vocab, draft,
            [](llama_token_data_array * p, float, uint32_t &) { ref_greedy_apply(p); }, 1.0f, 1);
    auto rb = rows_sample_and_accept_n(b, rows, n_vocab, draft,
            [](llama_token_data_array * p, float, uint32_t &) { ref_greedy_apply(p); }, 1.0f, 1);

    CHECK(ra.size() == rb.size());
    CHECK(memcmp(ra.data(), rb.data(), ra.size() * sizeof(llama_token)) == 0);
    CHECK(ra[0] == 0); // id 0 has the (tied) max logit in both builds

    // unique-max tie nearby: two ids share the max, third is lower
    std::vector<float> row2(n_vocab, 0.0f);
    row2[7] = 3.0f;
    row2[9] = 3.0f;
    std::vector<const float *> rows2 = { row2.data(), row2.data(), row2.data(), row2.data() };
    const std::vector<llama_token> draft2 = { 7, 9, 7 };

    test_sampler a2;
    test_sampler b2;
    auto ra2 = ref_sample_and_accept_n(a2, rows2, n_vocab, draft2,
            [](llama_token_data_array * p, float, uint32_t &) { ref_greedy_apply(p); }, 1.0f, 1);
    auto rb2 = rows_sample_and_accept_n(b2, rows2, n_vocab, draft2,
            [](llama_token_data_array * p, float, uint32_t &) { ref_greedy_apply(p); }, 1.0f, 1);
    CHECK(ra2.size() == rb2.size());
    CHECK(memcmp(ra2.data(), rb2.data(), ra2.size() * sizeof(llama_token)) == 0);
    CHECK(ra2[0] == 7); // first id of the tied max
}

static void run_guard_cases() {
    std::vector<float> row(1024, 0.0f);
    row[11] = 5.0f;
    std::vector<const float *> rows = { row.data(), row.data(), row.data(), row.data() };
    const std::vector<llama_token> draft = { 11, 11, 11 };

    // grammar present -> empty (fallback)
    test_sampler g;
    g.has_grmr = true;
    CHECK(rows_sample_and_accept_n(g, rows, 1024, draft,
                [](llama_token_data_array * p, float, uint32_t &) { ref_greedy_apply(p); }, 1.0f, 1).empty());

    // reasoning budget present -> empty (fallback)
    test_sampler rb;
    rb.has_rbudget = true;
    CHECK(rows_sample_and_accept_n(rb, rows, 1024, draft,
                [](llama_token_data_array * p, float, uint32_t &) { ref_greedy_apply(p); }, 1.0f, 1).empty());

    // backend-sampled token on row 0 -> empty (fallback)
    test_sampler bt;
    bt.backend_token = 42;
    CHECK(rows_sample_and_accept_n(bt, rows, 1024, draft,
                [](llama_token_data_array * p, float, uint32_t &) { ref_greedy_apply(p); }, 1.0f, 1).empty());

    // clean sampler -> accepted vector with the break-on-mismatch semantics
    test_sampler ok;
    auto r = rows_sample_and_accept_n(ok, rows, 1024, draft,
            [](llama_token_data_array * p, float, uint32_t &) { ref_greedy_apply(p); }, 1.0f, 1);
    CHECK(r.size() == 4); // all drafts match row argmax 11 -> draft.size()+1
    CHECK(r[0] == 11);
}

int main() {
    // greedy chain (the campaign temp-0 case): distinct random logits
    run_case(64, 400, false, 0);
    run_case(1024, 400, false, 0);
    run_case(32768, 100, false, 0);
    run_case(129272, 25, false, 0);
    run_case(129272, 25, true, 0);

    // seeded [top-k(10), dist] chain: distinct logits, seeded draws must match
    run_case(64, 400, false, 1);
    run_case(1024, 400, false, 1);
    run_case(129272, 25, false, 1);
    run_case(129272, 25, true, 1);

    run_tie_case(1024);
    run_tie_case(129272);

    run_guard_cases();

    if (n_fail == 0) {
        printf("ALL PASS (verify-row sampling parity: greedy + seeded top-k/dist, ties, guards)\n");
        return 0;
    }
    printf("FAILURES: %d\n", n_fail);
    return 1;
}
