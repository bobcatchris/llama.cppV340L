// T4 producer desk: host-side unit test for the q8_1 activation cache
// (ggml/src/ggml-cuda/common.cuh, enabled by GGML_CUDA_Q81_ACT_CACHE=1).
//
// Zero-GPU: exercises the cache logic with a mock pool - key matching (node,
// data pointer, shape, strides, stream, device), disabled no-op path, byte-cap
// behavior, per-compute invalidation and pool bookkeeping - and proves the
// bit-exactness contract host-side: a cache hit must return bytes identical to
// a fresh quantization. For that, the test carries a host mirror of the
// quantize_q8_1 kernel (ggml/src/ggml-cuda/quantize.cu), including the
// warp-reduce butterfly order for amax/sum.
//
// DEFERRED DEVICE VALIDATION (needs a die-3 window): the fp16 store of the
// block d/s pair is assumed round-to-nearest-even (matches __float2half), and
// on-device quantize-twice-vs-cache byte equality is to be confirmed with the
// served guard battery (greedy determinism must stay byte-identical with
// GGML_CUDA_Q81_ACT_CACHE=1).
//
// Build + run (host only, no GPU, no libggml link - stubs at the bottom):
//   cd <repo root>
//   hipcc -O2 -std=gnu++17 -DGGML_USE_HIP -DGGML_BACKEND_BUILD -DGGML_SHARED \
//     -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
//     -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//     docs/amd-port/tests/test_q81_cache_host.cpp -o /tmp/test_q81_cache
//   /tmp/test_q81_cache

#include "common.cuh"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// mock pool: malloc-backed, tracks live allocations so leaks and double-frees
// of cached buffers are caught
struct mock_pool : ggml_cuda_pool {
    std::vector<std::pair<void *, size_t>> live;
    int n_alloc = 0;
    int n_free  = 0;

    void * alloc(size_t size, size_t * actual_size) override {
        void * p = malloc(size);
        *actual_size = size;
        live.push_back({p, size});
        n_alloc++;
        return p;
    }

    void free(void * ptr, size_t size) override {
        for (size_t i = 0; i < live.size(); i++) {
            if (live[i].first == ptr) {
                CHECK(live[i].second == size);
                live[i] = live.back();
                live.pop_back();
                n_free++;
                ::free(ptr);
                return;
            }
        }
        CHECK(!"free of pointer this pool never allocated");
    }

    size_t live_bytes() const {
        size_t b = 0;
        for (const auto & e : live) {
            b += e.second;
        }
        return b;
    }
};

// synthetic src1 tensor: only the fields the cache reads are filled
static ggml_tensor make_src1(const void * data, const int64_t * ne, const int64_t * nb, enum ggml_op op) {
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.data = const_cast<void *>(data);
    t.op   = op;
    for (int i = 0; i < 4; i++) {
        t.ne[i] = ne[i];
        t.nb[i] = nb[i];
    }
    return t;
}

// byte-exact host mirror of quantize_q8_1 (ggml/src/ggml-cuda/quantize.cu):
// same contiguous block layout, same warp-reduce butterfly for amax/sum,
// same zero padding beyond ne00; the fp16 d/s pair uses __float2half
// (round-to-nearest-even), the same conversion the kernel performs
static void quantize_row_q8_1_host(const float * x, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3) {
    block_q8_1 * y = (block_q8_1 *) vy;

    int64_t ib = 0;
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                for (int64_t i0 = 0; i0 < ne0; i0 += QK8_1, ib++) {
                    float vals[QK8_1], amax_t[QK8_1], sum_t[QK8_1];
                    for (int iq = 0; iq < QK8_1; iq++) {
                        const int64_t i00 = i0 + iq;
                        const float xi = i00 < ne00 ? x[i3*s03 + i2*s02 + i1*s01 + i00] : 0.0f;
                        vals[iq]   = xi;
                        amax_t[iq] = fabsf(xi);
                        sum_t[iq]  = xi;
                    }
                    for (int offset = QK8_1/2; offset > 0; offset >>= 1) {
                        float amax_n[QK8_1], sum_n[QK8_1];
                        for (int iq = 0; iq < QK8_1; iq++) {
                            amax_n[iq] = fmaxf(amax_t[iq], amax_t[iq ^ offset]);
                            sum_n[iq]  = sum_t[iq]  + sum_t[iq ^ offset];
                        }
                        memcpy(amax_t, amax_n, sizeof(amax_t));
                        memcpy(sum_t,  sum_n,  sizeof(sum_t));
                    }
                    const float amax = amax_t[0];
                    const float sum  = sum_t[0];
                    const float d    = amax / 127.0f;
                    for (int iq = 0; iq < QK8_1; iq++) {
                        y[ib].qs[iq] = amax == 0.0f ? (int8_t) 0 : (int8_t) roundf(vals[iq] / d);
                    }
                    y[ib].ds.x = __float2half(d);
                    y[ib].ds.y = __float2half(sum);
                }
            }
        }
    }
}

int main() {
    std::mt19937 rng(3407);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);

    const int64_t ne10 = 5120;                       // V340L hidden size
    const int64_t ne10_pad = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    std::vector<float> x(ne10);
    for (auto & v : x) {
        v = dist(rng);
    }
    x[0] = 0.0f;                                     // all-zero block edge case
    x[QK8_1] = -0.0f;
    x[100] = 1e-30f;                                 // tiny magnitudes

    mock_pool pool;
    ggml_cuda_q81_act_cache cache;

    // contiguous src1, strides in elements are what make_key stores
    const int64_t ne[4]  = {ne10, 1, 1, 1};
    const int64_t nb[4]  = {4, ne10*4, ne10*4, ne10*4};
    ggml_tensor s1 = make_src1(x.data(), ne, nb, GGML_OP_RMS_NORM);

    const size_t nbytes = ggml_cuda_q81_act_cache::nbytes(&s1);
    CHECK(nbytes == (size_t) (1*ne10_pad * (int64_t) sizeof(block_q8_1)/QK8_1));

    const cudaStream_t st = (cudaStream_t) 0x1234;   // fake stream handle, never dereferenced
    const auto key = ggml_cuda_q81_act_cache::make_key(&s1, 0, st);

    CHECK(ggml_cuda_q81_act_cache::cacheable(&s1));

    // disabled cache: find/insert are no-ops with no side effects
    cache.begin_compute(false);
    CHECK(!cache.active);
    CHECK(cache.find(key) == nullptr);
    CHECK(cache.insert(key, nbytes, pool) == nullptr);
    CHECK(pool.n_alloc == 0);

    // ---- graph compute 1: miss -> insert -> quantize into cache-owned buffer
    cache.begin_compute(true);
    CHECK(cache.active);

    CHECK(cache.find(key) == nullptr);               // miss (n_misses = 1)
    ggml_cuda_q81_act_cache::entry * slot = cache.insert(key, nbytes, pool);
    CHECK(slot != nullptr && slot->buf != nullptr);
    quantize_row_q8_1_host(x.data(), slot->buf, ne10, ne10, ne10, ne10, ne10_pad, 1, 1, 1);

    // consumer 2 (wqkv_gate): hit, and a fresh quantize of x is byte-equal
    const ggml_cuda_q81_act_cache::entry * hit = cache.find(key);
    CHECK(hit != nullptr && hit->buf == slot->buf);
    CHECK(cache.n_hits == 1 && cache.n_misses == 1);

    std::vector<uint8_t> fresh(nbytes);
    quantize_row_q8_1_host(x.data(), fresh.data(), ne10, ne10, ne10, ne10, ne10_pad, 1, 1, 1);
    CHECK(memcmp(hit->buf, fresh.data(), nbytes) == 0);

    // consumer 3 (ssm_beta): still the same buffer, no new allocations
    const ggml_cuda_q81_act_cache::entry * hit3 = cache.find(key);
    CHECK(hit3 != nullptr && hit3->buf == slot->buf);
    CHECK(pool.n_alloc == 1);

    // key sensitivity: a DIFFERENT producing node at the same recycled address
    // must miss - this is the ggml graph allocator buffer-reuse defense
    ggml_tensor s1_recycled = make_src1(x.data(), ne, nb, GGML_OP_MUL);
    const auto key_recycled = ggml_cuda_q81_act_cache::make_key(&s1_recycled, 0, st);
    CHECK(!(key_recycled == key));
    CHECK(cache.find(key_recycled) == nullptr);

    // shape / stride / stream / device changes must miss
    {
        const int64_t ne_v[4] = {ne10, 2, 1, 1};
        ggml_tensor s = make_src1(x.data(), ne_v, nb, GGML_OP_RMS_NORM);
        CHECK(cache.find(ggml_cuda_q81_act_cache::make_key(&s, 0, st)) == nullptr);
    }
    {
        const int64_t nb_v[4] = {4, (ne10 + 64)*4, ne10*4, ne10*4};  // strided view
        const int64_t ne_v[4] = {ne10, 1, 1, 1};
        ggml_tensor s = make_src1(x.data(), ne_v, nb_v, GGML_OP_RMS_NORM);
        CHECK(cache.find(ggml_cuda_q81_act_cache::make_key(&s, 0, st)) == nullptr);
    }
    CHECK(cache.find(ggml_cuda_q81_act_cache::make_key(&s1, 0, (cudaStream_t) 0x5678)) == nullptr);
    CHECK(cache.find(ggml_cuda_q81_act_cache::make_key(&s1, 1, st)) == nullptr);



    // layout sensitivity (GGML_CUDA_MMVQ_ALN), fresh cache: a dual-region aln
    // buffer must never satisfy a legacy lookup or vice versa - the aln region
    // only exists behind the legacy one, so a cross-layout hit would read out
    // of bounds
    {
        ggml_cuda_q81_act_cache cache_l;
        mock_pool pool_l;
        cache_l.begin_compute(true);
        const auto key0 = ggml_cuda_q81_act_cache::make_key(&s1, 0, st, 0);
        const auto k1 = ggml_cuda_q81_act_cache::make_key(&s1, 0, st, 1);
        CHECK(!(k1 == key0));
        auto * slot0 = cache_l.insert(key0, nbytes, pool_l);
        auto * slot1 = cache_l.insert(k1, nbytes + nbytes/2, pool_l);
        CHECK(slot0 != nullptr && slot1 != nullptr && slot0->buf != slot1->buf);
        CHECK(cache_l.find(key0) == slot0);
        CHECK(cache_l.find(k1) == slot1);
        CHECK(cache_l.find(ggml_cuda_q81_act_cache::make_key(&s1, 0, st)) == slot0);
        CHECK(pool_l.n_alloc == 2);
    }

    // op filter: synthetic tensors (mul_mat_id scratch slices) are not cacheable
    {
        const int64_t ne_v[4] = {ne10, 1, 1, 1};
        ggml_tensor s = make_src1(x.data(), ne_v, nb, GGML_OP_NONE);
        CHECK(!ggml_cuda_q81_act_cache::cacheable(&s));
    }

    // ---- graph compute 2: everything dropped, pool memory fully returned
    const size_t live_after_compute1 = pool.live_bytes();
    CHECK(live_after_compute1 == slot->buf_size);
    cache.begin_compute(true);
    CHECK(pool.live_bytes() == 0);
    CHECK(pool.n_alloc == pool.n_free);
    CHECK(cache.find(key) == nullptr);               // stale generation must not hit

    // x mutated between computes (same node pointer recycled by a rebuilt graph):
    // fresh quantize of the new content, byte-equal to the fresh mirror
    for (auto & v : x) {
        v = dist(rng);
    }
    slot = cache.insert(key, nbytes, pool);
    CHECK(slot != nullptr);
    quantize_row_q8_1_host(x.data(), slot->buf, ne10, ne10, ne10, ne10, ne10_pad, 1, 1, 1);
    quantize_row_q8_1_host(x.data(), fresh.data(), ne10, ne10, ne10, ne10, ne10_pad, 1, 1, 1);
    CHECK(memcmp(slot->buf, fresh.data(), nbytes) == 0);

    // disabled mid-flight: begin_compute(false) frees entries, find/insert no-op
    cache.begin_compute(false);
    CHECK(pool.live_bytes() == 0);
    CHECK(cache.find(key) == nullptr);
    CHECK(cache.insert(key, nbytes, pool) == nullptr);

    // ---- byte cap: inserts stop at the cap, nothing is evicted, pool stays whole
    {
        ggml_cuda_q81_act_cache cap_cache;
        mock_pool cap_pool;
        cap_cache.begin_compute(true);

        const int64_t ne_c[4] = {4096, 1, 1, 1};     // padded 4096, 4096*36/32 bytes/entry
        const int64_t nb_c[4] = {4, 4096*4, 4096*4, 4096*4};
        const size_t entry_bytes = (size_t) (4096 * (int64_t) sizeof(block_q8_1)/QK8_1);

        std::vector<ggml_tensor> tensors;
        std::vector<ggml_cuda_q81_act_cache::key> keys;
        int n_inserted = 0;
        while (true) {
            tensors.push_back(make_src1(nullptr, ne_c, nb_c, GGML_OP_MUL));
            keys.push_back(ggml_cuda_q81_act_cache::make_key(&tensors.back(), 0, st));
            if (cap_cache.insert(keys.back(), entry_bytes, cap_pool) == nullptr) {
                tensors.pop_back();
                keys.pop_back();
                break;
            }
            n_inserted++;
        }
        CHECK(n_inserted > 100);                     // the cap bound is reached
        CHECK(cap_cache.total_bytes <= ggml_cuda_q81_act_cache::max_total_bytes);
        CHECK(cap_pool.live_bytes() == cap_cache.total_bytes);
        CHECK(cap_cache.find(keys[0]) != nullptr);   // no eviction

        // note: keys hold node pointer VALUES only, never dereferenced, so the
        // tensor vector reallocating during the loop is fine for find()
        ggml_tensor extra = make_src1(nullptr, ne_c, nb_c, GGML_OP_MUL);
        CHECK(cap_cache.find(ggml_cuda_q81_act_cache::make_key(&extra, 0, st)) == nullptr);

        cap_cache.begin_compute(false);
        CHECK(cap_pool.live_bytes() == 0);
    }

    // ---- layout parity: nbytes() equals both upstream call-site expressions
    {
        const int64_t shapes[][4] = {
            {5120, 1, 1, 1},
            {5120, 8, 1, 1},
            {4096, 1, 4, 2},
            {5000, 3, 1, 1},                          // padded 5000 -> 5120
        };
        for (const auto & sh : shapes) {
            const int64_t ne_s[4] = {sh[0], sh[1], sh[2], sh[3]};
            const int64_t nb_s[4] = {4, sh[0]*4, sh[0]*sh[1]*4, sh[0]*sh[1]*sh[2]*4};
            ggml_tensor s = make_src1(nullptr, ne_s, nb_s, GGML_OP_RMS_NORM);

            const int64_t padded = GGML_PAD(sh[0], MATRIX_ROW_PADDING);
            const int64_t nrows1 = sh[1]*sh[2]*sh[3];
            const size_t split_expr   = (size_t) (nrows1*padded * (int64_t) sizeof(block_q8_1)/QK8_1);
            const size_t nonsplit_expr = (size_t) ((int64_t) sh[3]*sh[2] * sh[1]*padded * (int64_t) sizeof(block_q8_1)/QK8_1);
            const size_t cache_expr   = ggml_cuda_q81_act_cache::nbytes(&s);
            CHECK(split_expr == cache_expr);
            CHECK(nonsplit_expr == cache_expr);
        }
    }

    // ---- pure-function determinism of the mirror (quantize-twice == quantize-once)
    {
        std::vector<uint8_t> q1(nbytes), q2(nbytes);
        quantize_row_q8_1_host(x.data(), q1.data(), ne10, ne10, ne10, ne10, ne10_pad, 1, 1, 1);
        quantize_row_q8_1_host(x.data(), q2.data(), ne10, ne10, ne10, ne10, ne10_pad, 1, 1, 1);
        CHECK(memcmp(q1.data(), q2.data(), nbytes) == 0);
    }

    if (n_fail == 0) {
        printf("test_q81_cache_host: ALL PASS\n");
        return 0;
    }
    printf("test_q81_cache_host: %d FAILURES\n", n_fail);
    return 1;
}

// ---- stubs for the GGML_API symbols referenced by the included headers.
// the test intentionally does not link libggml.

void ggml_log_internal(enum ggml_log_level level, const char * format, ...) {
    (void) level;
    (void) format;
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "abort: %s:%d: ", file, line);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    abort();
}

int64_t ggml_nrows(const struct ggml_tensor * tensor) {
    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}
