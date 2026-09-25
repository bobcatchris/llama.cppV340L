// Verify-round desk: host-side logic tests for the pinned device-copy staging
// cache (GGML_PINNED_DEV_COPY, no GPU, no ggml linkage).
//
// ggml/src/ggml-backend.cpp ggml_backend_dev_copy_staging replaces the
// per-call malloc/free host buffer of the slow device-to-device copy fallback
// (the path every n=3 butterfly peer copy takes on hardware without peer
// access) with a grow-only pinned buffer from the source device's host
// buffer type. The ggml_backend_tensor_get / ggml_backend_tensor_set
// sequence around the buffer is unchanged, so the copied bytes are identical
// by construction; what this test pins is the cache logic:
//
//   1. reuse: repeated requests within capacity return the same buffer and
//      never reallocate,
//   2. growth: larger requests reallocate geometrically (1 MiB start, x2)
//      and the old buffer is freed,
//   3. byte-exactness: a get-into-staging + set-from-staging round trip
//      preserves the source bytes exactly, at every capacity class,
//   4. fallback: no host buffer type (or alloc failure) returns nullptr and
//      the caller keeps the historical malloc path; the env gate unset also
//      keeps the historical path.
//
// Build + run:
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_pinned_staging_host
//       docs/amd-port/tests/test_pinned_staging_host.cpp && /tmp/test_pinned_staging_host

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// ---- fakes for the ggml buffer layer -------------------------------------->

struct fake_buffer {
    std::vector<uint8_t> mem;
    size_t size = 0;
    bool pinned = true;   // false = the alloc failed path
    bool is_free = false;
};

static int64_t n_allocs = 0;
static int64_t n_frees = 0;

static fake_buffer * fake_alloc(size_t size, bool pinned) {
    if (!pinned) {
        return nullptr;
    }
    fake_buffer * b = new fake_buffer();
    b->mem.resize(size);
    b->size = size;
    n_allocs++;
    return b;
}

static void fake_free(fake_buffer * b) {
    if (b) {
        b->is_free = true;
        n_frees++;
        delete b;
    }
}

// ---- mirror: ggml_backend_dev_copy_staging (ggml-backend.cpp) ------------->

static void * mirror_staging(bool host_buft_available, size_t nbytes) {
    static bool initialized = false;
    static fake_buffer * buf = nullptr;
    static size_t buf_size = 0;

    if (!initialized) {
        initialized = true;
    }

    if (buf && buf_size >= nbytes) {
        return buf->mem.data();
    }

    if (!host_buft_available) {
        return nullptr;
    }

    size_t new_size = buf_size > 0 ? buf_size : (size_t) 1 << 20;
    while (new_size < nbytes) {
        new_size *= 2;
    }

    fake_buffer * new_buf = fake_alloc(new_size, true);
    if (new_buf == nullptr || new_buf->mem.data() == nullptr) {
        if (new_buf != nullptr) {
            fake_free(new_buf);
        }
        return nullptr;
    }

    if (buf) {
        fake_free(buf);
    }
    buf = new_buf;
    buf_size = new_size;

    return buf->mem.data();
}

// mirror: the gated branch of ggml_backend_tensor_copy
static bool mirror_copy(const std::vector<uint8_t> & src, std::vector<uint8_t> & dst,
                        bool env_set, bool host_buft_available) {
    const size_t nbytes = src.size();

    uint8_t * data = nullptr;
    if (env_set) {
        data = (uint8_t *) mirror_staging(host_buft_available, nbytes);
    }
    const bool staged = data != nullptr;
    if (!staged) {
        data = (uint8_t *) malloc(nbytes);
    }
    memcpy(data, src.data(), nbytes);   // mirror: ggml_backend_tensor_get
    memcpy(dst.data(), data, nbytes);   // mirror: ggml_backend_tensor_set
    if (!staged) {
        free(data);
    }
    return staged;
}

// ---- cases ---------------------------------------------------------------->

static void test_reuse_and_growth() {
    // 1 MiB start: a 64 KiB request allocates once, then reuses
    std::vector<uint8_t> src(64 * 1024, 0);
    std::vector<uint8_t> dst(64 * 1024, 0);

    CHECK(mirror_copy(src, dst, true, true) == true);
    CHECK(n_allocs == 1);

    for (int i = 0; i < 100; i++) {
        CHECK(mirror_copy(src, dst, true, true) == true);
    }
    CHECK(n_allocs == 1);
    CHECK(n_frees == 0);

    // growth: 2 MiB request doubles 1 -> 2 MiB, old buffer freed
    src.assign(2u << 20, 7);
    dst.assign(2u << 20, 0);
    CHECK(mirror_copy(src, dst, true, true) == true);
    CHECK(n_allocs == 2);
    CHECK(n_frees == 1);
    CHECK(dst[0] == 7 && dst.back() == 7);

    // within the new capacity: no realloc
    src.assign(1u << 20, 3);
    dst.assign(1u << 20, 0);
    CHECK(mirror_copy(src, dst, true, true) == true);
    CHECK(n_allocs == 2);
}

static void test_byte_exactness() {
    std::mt19937 rng(42);

    // every capacity class: 1 KiB, 64 KiB (the [4, n_embd] f32 verify
    // partial), 1 MiB boundary, 5 MiB (growth), odd size
    const size_t sizes[] = { 1024, 64 * 1024, (1u << 20), (5u << 20), (3u << 20) + 17 };

    for (size_t sz : sizes) {
        std::vector<uint8_t> src(sz);
        for (size_t i = 0; i < sz; i++) {
            src[i] = (uint8_t) (rng() & 0xFF);
        }
        std::vector<uint8_t> dst_staged(sz, 0);
        std::vector<uint8_t> dst_malloc(sz, 0);

        const bool a = mirror_copy(src, dst_staged, true, true);
        const bool b = mirror_copy(src, dst_malloc, false, true);

        CHECK(a == true);
        CHECK(b == false); // env unset keeps the historical path
        CHECK(dst_staged == dst_malloc);
        CHECK(dst_staged == src);
    }
}

static void test_fallbacks() {
    // no host buffer type: nullptr -> the caller keeps malloc
    std::vector<uint8_t> src(4096, 9);
    std::vector<uint8_t> dst(4096, 0);
    const int64_t allocs_before = n_allocs;
    CHECK(mirror_copy(src, dst, true, false) == false);
    CHECK(dst == src);
    CHECK(n_allocs == allocs_before);
}

int main() {
    // fallbacks first: the no-host-buft case is meaningful on an empty cache
    // (host_buft availability is a per-device static property, so once a
    // buffer exists from a capable device the cache serves it)
    test_fallbacks();
    test_reuse_and_growth();
    test_byte_exactness();

    if (n_fail == 0) {
        printf("ALL PASS (pinned staging cache: reuse, growth, byte-exact copies, fallbacks)\n");
        return 0;
    }
    printf("FAILURES: %d\n", n_fail);
    return 1;
}
