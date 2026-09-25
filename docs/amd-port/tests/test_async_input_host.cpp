// Async-input desk: host-side logic tests for the staged async input sets
// (LLAMA_ASYNC_INPUT, no GPU, no ggml linkage).
//
// src/llama-graph.cpp llama_input_tensor_set replaces the per-ubatch blocking
// input sets (ggml_backend_tensor_set = cudaMemcpyAsync + stream synchronize
// per tensor, per die behind a tensor-parallel meta backend) with: memcpy into
// a pinned ring slot, ggml_backend_tensor_set_async on the owning backend's
// stream, and a per-slot gate (backend event when the device supports events,
// otherwise a synchronize of the issuing backend) before the slot is
// overwritten. What this test pins is that control flow, mirrored 1:1:
//
//   1. byte-exactness: the bytes that land in the device tensor are the bytes
//      handed to the setter, at every size class and wrap count,
//   2. ring-overwrite safety: a slot is never overwritten while the copy from
//      it may still be in flight (shadow snapshot at issue == landed bytes at
//      completion, for every issued copy),
//   3. event vs fallback gates: devices with events gate per slot event;
//      devices without events (the tensor-parallel meta device reports
//      caps.events = false) gate by synchronizing the backend the copy was
//      issued on - including when the previous user of the slot was a
//      different backend (target ctx vs draft ctx),
//   4. growth: slot staging grows geometrically and frees the old buffer,
//   5. fallbacks: env unset, missing scheduler, host tensor, missing host
//      buffer type, alloc failure, non-zero offset - all keep the historical
//      blocking set and issue nothing staged.
//
// Build + run:
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_async_input_host
//       docs/amd-port/tests/test_async_input_host.cpp && /tmp/test_async_input_host

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// ---- fakes for the ggml backend layer -------------------------------------->

struct fake_event;
struct fake_backend;

// one staged copy in flight: the DMA reads the LIVE slot bytes when it
// completes, so an overwrite before the gate shows up as landed != shadow.
// the destination tensor is not modeled per transfer: in the real pipeline a
// device input tensor is stream-ordered behind the graph that consumes it and
// may legitimately be superseded by the next ubatch's set
struct fake_transfer {
    uint8_t * slot;             // live staging memory
    std::vector<uint8_t> shadow;// slot bytes snapshotted at issue time
    std::vector<uint8_t> landed;// slot bytes the DMA actually read
    size_t    size;
    bool      complete = false;
};

struct fake_event {
    fake_transfer * bound = nullptr;
};

struct fake_backend {
    bool has_events = true;     // false models a meta device (caps.events false)
    std::vector<fake_transfer *> stream;    // issued, in issue order

    // host buffer type
    bool host_buft_ok  = true;
    bool fail_alloc    = false;
    size_t n_allocs    = 0;
    size_t n_frees     = 0;

    struct host_buf {
        size_t cap = 0;
        std::vector<uint8_t> mem;
    };
    host_buf * buf = nullptr;
};

static std::vector<fake_transfer *> g_transfers;

static fake_transfer * fake_set_async(fake_backend * b, uint8_t * slot, uint8_t * dst, size_t size) {
    (void) dst; // superseded transfers are stream-ordered in the real pipeline
    fake_transfer * t = new fake_transfer();
    t->slot   = slot;
    t->size   = size;
    t->shadow.assign(slot, slot + size);
    b->stream.push_back(t);
    g_transfers.push_back(t);
    return t;
}

// ggml_backend_event_record: binds the event to the backend's last transfer
static void fake_event_record(fake_event * ev, fake_backend * b) {
    ev->bound = b->stream.empty() ? nullptr : b->stream.back();
}

// ggml_backend_event_synchronize: the DMA lands now, reading live slot bytes
static void fake_event_synchronize(fake_event * ev) {
    if (ev->bound && !ev->bound->complete) {
        ev->bound->landed.assign(ev->bound->slot, ev->bound->slot + ev->bound->size);
        ev->bound->complete = true;
    }
}

// ggml_backend_synchronize: drains the whole backend stream in order
static void fake_backend_synchronize(fake_backend * b) {
    for (fake_transfer * t : b->stream) {
        if (!t->complete) {
            t->landed.assign(t->slot, t->slot + t->size);
            t->complete = true;
        }
    }
}

static size_t g_n_gates = 0;    // gate invocations (event or backend sync)

// ---- mirror of the llama-graph.cpp staging (llama_input_tensor_set) ------->

// ring slot, mirroring llama_async_input_slot
struct mirror_slot {
    fake_backend::host_buf * buf          = nullptr;
    size_t                   cap          = 0;
    fake_event *             event        = nullptr;
    fake_backend *           event_dev    = nullptr; // stands in for the event's device
    fake_backend *           gate_backend = nullptr;
};

static constexpr size_t MIRROR_N_SLOTS = 8;

struct mirror_state {
    mirror_slot slots[MIRROR_N_SLOTS];
    size_t      next = 0;
};

// fake tensor: device memory + owning backend (stands in for the sched lookup)
struct mirror_tensor {
    std::vector<uint8_t> mem;
    fake_backend * owner = nullptr; // nullptr = not allocated / no backend
    bool host = false;              // host buffer tensor
};

static bool mirror_staged_enabled = true;   // stands in for getenv("LLAMA_ASYNC_INPUT")
static bool mirror_have_sched     = true;   // stands in for tl_async_input_sched

static void mirror_blocking_set(mirror_tensor & t, const void * data, size_t offset, size_t size) {
    memcpy(t.mem.data() + offset, data, size);
}

static void mirror_input_tensor_set(
        mirror_state & st,
        mirror_tensor & t,
        const void * data,
        size_t offset,
        size_t size) {
    if (!mirror_staged_enabled || !mirror_have_sched || offset != 0 || size == 0) {
        mirror_blocking_set(t, data, offset, size);
        return;
    }

    fake_backend * backend = t.host ? nullptr : t.owner;
    if (!backend || !backend->host_buft_ok) {
        mirror_blocking_set(t, data, offset, size);
        return;
    }

    mirror_slot & slot = st.slots[st.next];
    st.next = (st.next + 1) % MIRROR_N_SLOTS;

    // gate the previous staged copy from this slot before overwriting it
    if (slot.event) {
        g_n_gates++;
        fake_event_synchronize(slot.event);
    } else if (slot.gate_backend) {
        g_n_gates++;
        fake_backend_synchronize(slot.gate_backend);
    }

    if (slot.cap < size) {
        size_t new_cap = slot.cap > 0 ? slot.cap : 1024;
        while (new_cap < size) {
            new_cap *= 2;
        }

        fake_backend::host_buf * new_buf = nullptr;
        if (!backend->fail_alloc) {
            new_buf = new fake_backend::host_buf();
            new_buf->cap = new_cap;
            new_buf->mem.resize(new_cap);
            backend->n_allocs++;
        }
        if (new_buf == nullptr || new_buf->mem.empty()) {
            delete new_buf;
            mirror_blocking_set(t, data, offset, size);
            return;
        }

        if (slot.buf != nullptr) {
            backend->n_frees++;
            delete slot.buf;
        }
        slot.buf = new_buf;
        slot.cap = new_cap;
    }

    uint8_t * stage = slot.buf->mem.data();
    memcpy(stage, data, size);
    fake_set_async(backend, stage, t.mem.data() + offset, size);

    // stamp the gate for the next reuse of this slot
    slot.gate_backend = backend;
    if (slot.event && slot.event_dev != backend) {
        delete slot.event;
        slot.event     = nullptr;
        slot.event_dev = nullptr;
    }
    if (!slot.event) {
        slot.event = backend->has_events ? new fake_event() : nullptr;
        slot.event_dev = slot.event ? backend : nullptr;
    }
    if (slot.event) {
        fake_event_record(slot.event, backend);
    }
}

// complete every outstanding transfer (the output drain of the decode loop)
static void drain_all() {
    for (fake_transfer * t : g_transfers) {
        if (!t->complete) {
            t->landed.assign(t->slot, t->slot + t->size);
            t->complete = true;
        }
    }
}

// every issued transfer landed exactly the bytes the setter handed over
static bool all_landed_exact() {
    for (fake_transfer * t : g_transfers) {
        if (!t->complete) {
            return false;
        }
        if (t->landed != t->shadow) {
            return false;
        }
    }
    return true;
}

static void reset_all() {
    for (fake_transfer * t : g_transfers) {
        delete t;
    }
    g_transfers.clear();
    g_n_gates = 0;
}

// free a mirror_state's slots (staging buffers + events); test-teardown only
static void free_state(mirror_state & st) {
    drain_all();
    for (auto & slot : st.slots) {
        delete slot.buf;
        slot.buf = nullptr;
        slot.cap = 0;
        delete slot.event;
        slot.event     = nullptr;
        slot.event_dev = nullptr;
        slot.gate_backend = nullptr;
    }
    reset_all();
}

// ---- cases ---------------------------------------------------------------->

// the V340L round shape: 5 ubatches (3 draft steps on ctx_dft, verify + the
// 4-token catch-up), tokens/pos/h rows, backends alternating per context,
// random payloads; drain timing varied per round; ring wraps many times
static void test_round_shape_byte_exact_and_safety() {
    std::mt19937 rng(1234);

    fake_backend ctx_tgt;  // has_events = true variant
    fake_backend ctx_dft;

    mirror_tensor t_tokens_tgt; t_tokens_tgt.mem.resize(64);   t_tokens_tgt.owner = &ctx_tgt;
    mirror_tensor t_pos_tgt;    t_pos_tgt.mem.resize(64);      t_pos_tgt.owner    = &ctx_tgt;
    mirror_tensor t_tokens_dft; t_tokens_dft.mem.resize(64);   t_tokens_dft.owner = &ctx_dft;
    mirror_tensor t_h_dft;      t_h_dft.mem.resize(32 * 1024); t_h_dft.owner      = &ctx_dft;

    mirror_state st;

    for (int round = 0; round < 50; round++) {
        // draft step on ctx_dft: tokens + h row
        for (auto * t : { &t_tokens_dft, &t_h_dft }) {
            std::vector<uint8_t> p(t->mem.size());
            for (auto & b : p) { b = (uint8_t)(rng() & 0xFF); }
            mirror_input_tensor_set(st, *t, p.data(), 0, p.size());
        }

        // verify ubatch on ctx_tgt: tokens + pos
        for (auto * t : { &t_tokens_tgt, &t_pos_tgt }) {
            std::vector<uint8_t> p(t->mem.size());
            for (auto & b : p) { b = (uint8_t)(rng() & 0xFF); }
            mirror_input_tensor_set(st, *t, p.data(), 0, p.size());
        }

        // catch-up ubatch on ctx_dft: tokens + h (4 rows) + pos
        for (auto * t : { &t_tokens_dft, &t_h_dft, &t_pos_tgt }) {
            std::vector<uint8_t> p(t->mem.size());
            for (auto & b : p) { b = (uint8_t)(rng() & 0xFF); }
            mirror_input_tensor_set(st, *t, p.data(), 0, p.size());
        }

        // odd rounds: drain between rounds (sampling waits); even rounds: let
        // transfers pile up across rounds before draining - both must be safe
        if (round % 2 == 0) {
            drain_all();
        }
    }
    drain_all();

    CHECK(all_landed_exact());  // no in-flight slot was ever overwritten
    CHECK(g_transfers.size() == 50 * 7);
    free_state(st);
}

// devices without events (meta device, caps.events = false): the gate is a
// backend synchronize; across contexts the gate must hit the backend that
// issued the in-flight copy
static void test_no_event_gate_backend() {
    std::mt19937 rng(77);
    std::vector<uint8_t> src(16 * 1024);

    fake_backend b1;  b1.has_events = false;
    fake_backend b2; b2.has_events = false;

    mirror_tensor t1; t1.mem.resize(16 * 1024); t1.owner = &b1;
    mirror_tensor t2; t2.mem.resize(16 * 1024); t2.owner = &b2;

    mirror_state st;
    reset_all();

    for (size_t i = 0; i < 3 * MIRROR_N_SLOTS + 5; i++) {
        for (auto & b : src) { b = (uint8_t)(rng() & 0xFF); }
        mirror_tensor & t = (i % 2) ? t2 : t1;
        mirror_input_tensor_set(st, t, src.data(), 0, src.size());
        // no explicit drain: the ring must gate by backend sync on its own
    }
    drain_all();

    CHECK(all_landed_exact());
    CHECK(g_n_gates > 0);       // slots were reused and gated

    // every slot ends up gated by a backend (no events anywhere)
    for (size_t i = 0; i < MIRROR_N_SLOTS; i++) {
        CHECK(st.slots[i].gate_backend != nullptr);
        CHECK(st.slots[i].event == nullptr);
    }
    free_state(st);
}

// event path: gate event is created per device and re-created if the device
// behind a slot changes
static void test_event_gate_and_device_change() {
    std::mt19937 rng(99);
    std::vector<uint8_t> src(4096);

    fake_backend dev_a;    // has_events = true (the default)
    fake_backend dev_b;

    mirror_tensor ta; ta.mem.resize(4096); ta.owner = &dev_a;
    mirror_tensor tb; tb.mem.resize(4096); tb.owner = &dev_b;

    mirror_state st;
    reset_all();

    for (size_t i = 0; i < 4 * MIRROR_N_SLOTS + 3; i++) {
        for (auto & b : src) { b = (uint8_t)(rng() & 0xFF); }
        { mirror_tensor & tt = (i % 2) ? tb : ta; mirror_input_tensor_set(st, tt, src.data(), 0, src.size()); }
    }
    drain_all();

    CHECK(all_landed_exact());
    CHECK(g_n_gates > 0);
    for (size_t i = 0; i < MIRROR_N_SLOTS; i++) {
        CHECK(st.slots[i].event != nullptr);    // event-capable devices gate by event
    }
    free_state(st);
}

// geometric growth of a slot's staging and old-buffer free
static void test_growth() {
    fake_backend b;
    mirror_tensor t; t.mem.resize(1u << 21); t.owner = &b;

    mirror_state st;
    reset_all();

    std::vector<uint8_t> p0(1024, 1);
    mirror_input_tensor_set(st, t, p0.data(), 0, p0.size());
    CHECK(st.slots[0].cap == 1024);
    CHECK(b.n_allocs == 1);

    std::vector<uint8_t> p1(2048, 2);
    mirror_input_tensor_set(st, t, p1.data(), 0, p1.size());
    CHECK(st.slots[1].cap == 2048);

    std::vector<uint8_t> p2(3000, 3);   // odd size grows to 4096
    mirror_input_tensor_set(st, t, p2.data(), 0, p2.size());
    CHECK(st.slots[2].cap == 4096);

    // the next set lands in slot 3 (round-robin): fresh slot, grows to 2048
    mirror_input_tensor_set(st, t, p1.data(), 0, p1.size());
    CHECK(st.slots[3].cap == 2048);
    CHECK(b.n_allocs == 4);
    CHECK(b.n_frees == 0);

    // fill slots 4..7, then wrap: slot 0 reuse with a smaller payload must
    // neither realloc nor free (gate + within-capacity reuse)
    std::vector<uint8_t> p3(64, 4);
    for (size_t i = 4; i < MIRROR_N_SLOTS; i++) {
        mirror_input_tensor_set(st, t, p3.data(), 0, p3.size());
    }
    CHECK(b.n_allocs == 8);

    const size_t allocs_before = b.n_allocs;
    const size_t frees_before  = b.n_frees;
    mirror_input_tensor_set(st, t, p3.data(), 0, p3.size());    // slot 0 again
    CHECK(st.slots[0].cap == 1024);
    CHECK(b.n_allocs == allocs_before);
    CHECK(b.n_frees == frees_before);
    drain_all();

    CHECK(all_landed_exact());
    free_state(st);
}

// fallbacks keep the blocking set and issue nothing staged
static void test_fallbacks() {
    std::mt19937 rng(5);
    mirror_state st;
    reset_all();

    std::vector<uint8_t> src(512, 9);

    // env unset
    mirror_tensor t; t.mem.resize(512); t.owner = new fake_backend();
    mirror_staged_enabled = false;
    mirror_input_tensor_set(st, t, src.data(), 0, src.size());
    CHECK(g_transfers.empty());
    CHECK(t.mem == src);
    mirror_staged_enabled = true;

    // no scheduler
    mirror_have_sched = false;
    mirror_input_tensor_set(st, t, src.data(), 0, src.size());
    CHECK(g_transfers.empty());
    mirror_have_sched = true;

    // host tensor
    t.host = true;
    mirror_input_tensor_set(st, t, src.data(), 0, src.size());
    CHECK(g_transfers.empty());
    t.host = false;

    // no host buffer type
    static_cast<fake_backend *>(t.owner)->host_buft_ok = false;
    mirror_input_tensor_set(st, t, src.data(), 0, src.size());
    CHECK(g_transfers.empty());
    static_cast<fake_backend *>(t.owner)->host_buft_ok = true;

    // alloc failure
    static_cast<fake_backend *>(t.owner)->fail_alloc = true;
    mirror_input_tensor_set(st, t, src.data(), 0, src.size());
    CHECK(g_transfers.empty());
    CHECK(t.mem == src);
    static_cast<fake_backend *>(t.owner)->fail_alloc = false;

    // non-zero offset and zero size
    mirror_input_tensor_set(st, t, src.data(), 8, src.size() - 8);
    CHECK(g_transfers.empty());
    mirror_input_tensor_set(st, t, src.data(), 0, 0);
    CHECK(g_transfers.empty());

    delete static_cast<fake_backend *>(t.owner);
    free_state(st);
}

int main() {
    test_fallbacks();
    test_round_shape_byte_exact_and_safety();
    test_no_event_gate_backend();
    test_event_gate_and_device_change();
    test_growth();

    if (n_fail == 0) {
        printf("ALL PASS (async input staging: byte-exact landings, ring-overwrite safety, "
               "event and backend-sync gates, growth, fallbacks)\n");
        return 0;
    }
    printf("FAILURES: %d\n", n_fail);
    return 1;
}
