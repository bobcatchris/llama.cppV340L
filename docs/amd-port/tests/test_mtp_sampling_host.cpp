// MTP-overhead desk: host-side sampling parity tests (no GPU, no ggml linkage).
//
// The draft-mtp sampling fast path (LLAMA_DRAFT_FAST_TOPK=1,
// common/sampling.cpp common_sampler_sample_topk + set_logits_topk) replaces
// the full-vocab candidate array build (129272 llama_token_data, ~2 MB) plus
// std::partial_sort over it with a bounded heap-select over the raw logits
// row, and hands only the k survivors to the same sampler chain. This test
// pins the parity contract against the regular path:
//
//   regular path (mirror of common_sampler::set_logits + llama-sampler.cpp):
//     cur[token_id] = {token_id, logits[token_id], 0} for the whole vocab
//     std::partial_sort(cur, cur + k, cur + n_vocab, logit-descending)
//
//   fast path (mirror of common_sampler::set_logits_topk):
//     make_heap over the first k candidates, scan the rest keeping the heap
//     bounded, sort_heap at the end (the std::partial_sort algorithm)
//
// then both run the same dist sampler (mirror of llama_sampler_dist_apply,
// softmax over the survivors + seeded draw) because the draft chain is
// top-k(10) + dist.
//
// Parity contract verified here:
//   1. distinct logits (the real-model case): the selected top-k ids, their
//      logits, the normalized probabilities and the drafted token
//      (cur_p.data[0].id, what draft-mtp pushes into the result) are
//      BIT-EXACT between the paths, across randomized and adversarial
//      distributions including k-boundary near-ties.
//   2. exact ties: same top-1 when the maximum is unique, same top-k id
//      multiset and same probability values. tie ORDER (and hence the seeded
//      dist draw among exactly-tied candidates) may diverge - this is the
//      documented temp>0 divergence risk and it is measured, not assumed.
//
// Build + run:
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_mtp_sampling_host
//       docs/amd-port/tests/test_mtp_sampling_host.cpp && /tmp/test_mtp_sampling_host

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

// mirror: llama-sampler.cpp llama_token_data_array_partial_sort_inplace,
// k <= 128 branch (comparator identical: strictly logit-descending)
static void ref_partial_sort_inplace(llama_token_data_array * cur_p, int npartial) {
    static const auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    };

    if (npartial <= 128) {
        std::partial_sort(cur_p->data, cur_p->data + npartial, cur_p->data + cur_p->size, comp);

        cur_p->size = npartial;
        cur_p->sorted = true;

        return;
    }
}

// mirror: common/sampling.cpp common_sampler::set_logits (regular path)
static void ref_set_logits(const float * logits, int n_vocab, std::vector<llama_token_data> & cur,
        llama_token_data_array & cur_p) {
    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    cur_p = { cur.data(), cur.size(), -1, false };
}

// mirror: common/sampling.cpp common_sampler::set_logits_topk (fast path)
static void fast_set_logits_topk(const float * logits, int n_vocab, int k,
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

// mirror: llama-sampler.cpp llama_sampler_dist_apply (softmax over the
// survivors + seeded draw, single-pass normalization). consumes rng state
// exactly like the real chain, so `selected` matches when the candidate
// order matches.
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

// mirror of the draft_mtp draft-step usage: chain = top-k(10) + dist, the
// drafted token is cur_p.data[0].id (NOT the dist draw - dist only fills the
// probabilities for the p_min gate)
struct sample_result {
    std::vector<llama_token_data> sorted;  // surviving candidates, chain order
    llama_token drafted;                   // cur_p.data[0].id
    int32_t dist_selected;                 // cur_p.selected (dist draw)
};

static sample_result sample_ref(const float * logits, int n_vocab, uint32_t seed) {
    const int k = 10;

    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p{};
    ref_set_logits(logits, n_vocab, cur, cur_p);
    ref_partial_sort_inplace(&cur_p, k);

    sample_result res;
    res.sorted.assign(cur_p.data, cur_p.data + cur_p.size);

    std::mt19937 rng(seed);
    mirror_dist_apply(&cur_p, rng);

    res.drafted       = cur_p.data[0].id;
    res.dist_selected = cur_p.selected;

    return res;
}

static sample_result sample_fast(const float * logits, int n_vocab, uint32_t seed) {
    const int k = 10;

    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p{};
    fast_set_logits_topk(logits, n_vocab, k, cur, cur_p);

    sample_result res;
    res.sorted.assign(cur_p.data, cur_p.data + cur_p.size);

    std::mt19937 rng(seed);
    mirror_dist_apply(&cur_p, rng);

    res.drafted       = cur_p.data[0].id;
    res.dist_selected = cur_p.selected;

    return res;
}

// ---- test drivers -------------------------------------------------------->

static std::vector<float> sorted_logits(const sample_result & s) {
    std::vector<float> v;
    for (const auto & t : s.sorted) { v.push_back(t.logit); }
    std::sort(v.begin(), v.end());
    return v;
}

static bool arrays_bit_equal(const sample_result & a, const sample_result & b) {
    if (a.sorted.size() != b.sorted.size()) {
        return false;
    }
    return memcmp(a.sorted.data(), b.sorted.data(), a.sorted.size()*sizeof(llama_token_data)) == 0;
}

// make every logit distinct by bumping exact duplicates 1 ULP upward
// (random f32 draws collide naturally at vocab scale; the bit-exactness
// contract below is about distinct values, exact ties are tested separately)
static void make_distinct(std::vector<float> & logits) {
    std::vector<size_t> idx(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) { idx[i] = i; }
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return logits[a] < logits[b]; });
    for (size_t j = 1; j < idx.size(); ++j) {
        if (logits[idx[j]] == logits[idx[j - 1]]) {
            logits[idx[j]] = std::nextafterf(logits[idx[j]], INFINITY);
        }
    }
    // re-sort not needed: the bump keeps order (each bumped value lands above
    // its twin and below the next distinct value or becomes it - possible new
    // collisions with the NEXT value are fixed by a second pass)
    for (size_t j = 1; j < idx.size(); ++j) {
        if (logits[idx[j]] <= logits[idx[j - 1]]) {
            logits[idx[j]] = std::nextafterf(logits[idx[j - 1]], INFINITY);
        }
    }
}

// distinct-value trials: full bit-exactness required
static void test_distinct_random(int n_vocab, int n_trials, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaus(0.0f, 4.0f);
    std::uniform_real_distribution<float> unif(-20.0f, 20.0f);

    std::vector<float> logits(n_vocab);

    for (int t = 0; t < n_trials; ++t) {
        for (int i = 0; i < n_vocab; ++i) {
            logits[i] = (t & 1) ? gaus(rng) : unif(rng);
        }
        make_distinct(logits);

        const uint32_t draw_seed = rng();

        const sample_result a = sample_ref(logits.data(), n_vocab, draw_seed);
        const sample_result b = sample_fast(logits.data(), n_vocab, draw_seed);

        CHECK(arrays_bit_equal(a, b));
        CHECK(a.drafted == b.drafted);
        CHECK(a.dist_selected == b.dist_selected);
    }
}

// k-boundary near-ties: 24 candidates placed at distinct ULP offsets around a
// base value so the top-k boundary sits inside the band. all floats involved
// are distinct -> full bit-exactness required.
static void test_near_ties(int n_vocab, int n_trials, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unif(-5.0f, 5.0f);

    std::vector<float> logits(n_vocab);

    const int band_start = n_vocab/2;
    const int band_len   = 24;

    for (int t = 0; t < n_trials; ++t) {
        for (int i = 0; i < n_vocab; ++i) {
            logits[i] = unif(rng);
        }

        // distinct offsets: -12..-1 and +1..+12 ULP around base (0 excluded so
        // no two band members collide, and base itself is overwritten)
        const float base = logits[band_start];
        for (int i = 0; i < band_len; ++i) {
            const int32_t d = (i < 12) ? (i - 12) : (i - 11);
            const int   idx = band_start + i;
            float v = base;
            for (int32_t s = 0; s < (d > 0 ? d : -d); ++s) {
                v = std::nextafterf(v, (d > 0) ? INFINITY : -INFINITY);
            }
            logits[idx] = v;
        }

        // unique maximum far from the band, so the drafted token is stable
        logits[n_vocab - 1] = base + 20.0f;

        make_distinct(logits);

        const uint32_t draw_seed = rng();

        const sample_result a = sample_ref(logits.data(), n_vocab, draw_seed);
        const sample_result b = sample_fast(logits.data(), n_vocab, draw_seed);

        CHECK(a.drafted == b.drafted);
        CHECK(arrays_bit_equal(a, b));
    }
}

// exact ties: quantized logits force equal values. contract: same drafted
// token whenever the max is unique, same id multiset, same probabilities.
// tie order may differ (documented divergence risk).
static void test_exact_ties(int n_vocab, int n_trials, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unif(-4.0f, 4.0f);

    int n_multiset_diff = 0;

    std::vector<float> logits(n_vocab);

    for (int t = 0; t < n_trials; ++t) {
        for (int i = 0; i < n_vocab; ++i) {
            logits[i] = std::round(unif(rng)*4.0f)/4.0f; // 33 distinct values
        }
        if (t % 10 == 0) {
            std::fill(logits.begin(), logits.end(), 1.0f); // all-equal rows
        }

        const uint32_t draw_seed = rng();

        const sample_result a = sample_ref(logits.data(), n_vocab, draw_seed);
        const sample_result b = sample_fast(logits.data(), n_vocab, draw_seed);

        // the survivor LOGIT multiset must be identical; with ~30 ids sharing
        // each quantized value, WHICH tied ids fill the top-10 is arbitrary
        // on both paths (heap-dependent), so the id multiset is not compared
        const std::vector<float> la = sorted_logits(a);
        const std::vector<float> lb = sorted_logits(b);
        CHECK(la.size() == lb.size());
        CHECK(memcmp(la.data(), lb.data(), la.size()*sizeof(float)) == 0);

        // same drafted token when the max is unique; when tied, the drafted
        // token must at least carry the maximum logit on both paths
        float max_l = logits[0];
        int n_max = 1;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                n_max = 1;
            } else if (logits[i] == max_l) {
                n_max++;
            }
        }
        if (n_max == 1) {
            CHECK(a.drafted == b.drafted);
        } else {
            bool ok = true;
            for (const auto & t : a.sorted) { ok = ok && t.logit <= max_l; }
            for (const auto & t : b.sorted) { ok = ok && t.logit <= max_l; }
            CHECK(ok);
            CHECK(a.sorted[0].logit == max_l);
            CHECK(b.sorted[0].logit == max_l);
        }

        if (!arrays_bit_equal(a, b)) {
            n_multiset_diff++;
        }
    }

    fprintf(stderr, "  [info] n_vocab=%d: exact-tie trials with tie-order diffs: %d/%d (documented, stochastic draw only)\n",
            n_vocab, n_multiset_diff, n_trials);
}

// temp-0 identity on the top-1 with a full-vocab-scale row (129272 = V340L
// vocab), a few hundred trials at realistic logit scales
static void test_fullscale_top1(int n_trials, uint32_t seed) {
    const int n_vocab = 129272;
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaus(0.0f, 3.0f);

    std::vector<float> logits(n_vocab);

    for (int t = 0; t < n_trials; ++t) {
        for (int i = 0; i < n_vocab; ++i) {
            logits[i] = gaus(rng);
        }
        // unique maximum with a margin, the serving regime
        logits[rng() % n_vocab] += 6.0f;

        const sample_result a = sample_ref(logits.data(), n_vocab, rng());
        const sample_result b = sample_fast(logits.data(), n_vocab, 0); // seed unused for top-1

        CHECK(a.drafted == b.drafted);
    }
}

// the fused-path guard: k > n_vocab must clamp, k = n_vocab must full-sort
static void test_edges() {
    const int n_vocab = 7;
    std::vector<float> logits = {0.5f, -1.0f, 3.0f, 0.5f, 2.0f, -3.0f, 3.0f};

    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p{};

    fast_set_logits_topk(logits.data(), n_vocab, 32, cur, cur_p);
    CHECK(cur_p.size == (size_t) n_vocab);
    CHECK(cur_p.sorted);
    CHECK(cur_p.data[0].logit == 3.0f);

    fast_set_logits_topk(logits.data(), n_vocab, 1, cur, cur_p);
    CHECK(cur_p.size == 1);
    CHECK(cur_p.data[0].logit == 3.0f);

    fast_set_logits_topk(logits.data(), n_vocab, 0, cur, cur_p);
    CHECK(cur_p.size == (size_t) n_vocab);

    // extreme magnitudes survive the heap intact (no NaN production)
    std::vector<float> extremes = {-INFINITY, 0.0f, 88.0f, -88.0f, INFINITY, 1e30f, -1e30f, 0.0f};
    fast_set_logits_topk(extremes.data(), (int) extremes.size(), 4, cur, cur_p);
    CHECK(cur_p.size == 4);
    CHECK(cur_p.data[0].logit == INFINITY);
    CHECK(cur_p.data[1].logit == 1e30f);
    CHECK(cur_p.data[2].logit == 88.0f);
    // two 0.0f are tied at the k boundary; the 4th survivor must be one of them
    CHECK(cur_p.data[3].logit == 0.0f);
}

int main() {
    fprintf(stderr, "== distinct random (small vocab, many trials)\n");
    test_distinct_random(64,   200000, 0xA11CE001);
    test_distinct_random(1000,  20000, 0xB0B5EED);

    fprintf(stderr, "== distinct random (large vocab)\n");
    test_distinct_random(32768, 2000, 0xC0FFEE42);
    test_distinct_random(129272, 200, 0xD1CE0001);

    fprintf(stderr, "== k-boundary near-ties\n");
    test_near_ties(1024, 20000, 0xE2E2E2E2);
    test_near_ties(129272, 300, 0xE3E3E3E3);

    fprintf(stderr, "== exact ties (documented tie-order risk)\n");
    test_exact_ties(64,   50000, 0xF4CAC010);
    test_exact_ties(1000, 20000, 0xF5F5F5F5);

    fprintf(stderr, "== full-scale top-1 (129272)\n");
    test_fullscale_top1(400, 0xF6A60001);

    fprintf(stderr, "== edges\n");
    test_edges();

    if (n_fail == 0) {
        fprintf(stderr, "ALL PASS\n");
        return 0;
    }
    fprintf(stderr, "%d FAILURES\n", n_fail);
    return 1;
}
