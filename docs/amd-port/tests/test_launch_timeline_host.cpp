// Launch/replay-path desk: host-side logic tests (no GPU, no ggml linkage).
//
// Covers the two env-gated changes shipped on amd/launchpath:
//
//   1. LLAMA_TARGET_LIGHT_SYNC=1 (ggml-backend.cpp ggml_backend_sched_compute_splits):
//      without scheduler events, one split-backend drain is hoisted in front of
//      the split-input copy loop and the per-input drains are skipped. The test
//      mirrors both control flows and pins: identical copied bytes, the
//      no-overwrite-before-drain guarantee, the sync collapse (n_inputs -> 1),
//      and that the events path is untouched.
//
//   2. GGML_CUDA_COMPAT_CACHE=1 (ggml-cuda.cu ggml_cuda_graph_compat_verdict):
//      the per-call compatibility scan is skipped when the cgraph uid matches
//      the cached verdict. The test mirrors the verdict decision and pins:
//      cached verdict equals the scan verdict for a stable uid, a uid change
//      rescans, an incompatible verdict is cached too, and env unset keeps the
//      historical rescan.
//
// The [launch-timeline] counters are passive host reads; they are pinned by
// compile + the served arm (engagement lines), not here.
//
// Build + run:
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_launch_timeline_host
//       docs/amd-port/tests/test_launch_timeline_host.cpp && /tmp/test_launch_timeline_host

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// ---- shared fake layer ---------------------------------------------------->

struct fake_tensor {
    std::vector<uint8_t> data;
    bool flag_input = false;
};

struct fake_event {
    int waits = 0;
    int syncs = 0;
};

struct fake_backend {
    int n_syncs = 0;
    // models the previous ubatch graph still reading the copy buffers; a
    // synchronize marks the device drained (reads complete)
    bool device_busy = true;
};

static void fake_synchronize(fake_backend & b) {
    b.n_syncs++;
    b.device_busy = false;
}

static void fake_copy(const fake_tensor & src, fake_tensor & dst) {
    dst.data.resize(src.data.size());
    memcpy(dst.data.data(), src.data.data(), src.data.size());
}

// ---- mirror 1: sched_compute_splits input-copy loop ----------------------->

struct split_copy_trace {
    int n_syncs = 0;
    int n_event_syncs = 0;
    int n_event_waits = 0;
    std::vector<std::vector<uint8_t>> copied;   // bytes landed per input
    std::vector<bool> copy_before_drain;        // overwrite-safety violations
};

// events == NULL path, light off: historical behavior (one drain per input)
static split_copy_trace mirror_regular(fake_backend & b, const std::vector<fake_tensor> & inputs) {
    split_copy_trace t;
    for (const auto & input : inputs) {
        fake_synchronize(b);                    // the per-input drain
        if (b.device_busy) {
            t.copy_before_drain.push_back(true);
        } else {
            t.copy_before_drain.push_back(false);
        }
        fake_tensor cpy;
        fake_copy(input, cpy);
        t.copied.push_back(cpy.data);
    }
    t.n_syncs = b.n_syncs;
    return t;
}

// events == NULL path, light on: one hoisted drain, then copies
static split_copy_trace mirror_light(fake_backend & b, const std::vector<fake_tensor> & inputs, bool light_enabled) {
    split_copy_trace t;
    const bool light_sync = light_enabled && inputs.size() > 0;
    if (light_sync) {
        fake_synchronize(b);
    }
    for (const auto & input : inputs) {
        if (!light_sync) {
            fake_synchronize(b);
        }
        if (b.device_busy) {
            t.copy_before_drain.push_back(true);
        } else {
            t.copy_before_drain.push_back(false);
        }
        fake_tensor cpy;
        fake_copy(input, cpy);
        t.copied.push_back(cpy.data);
    }
    t.n_syncs = b.n_syncs;
    return t;
}

// events != NULL path: event_synchronize (input) / event_wait (other), no drains
static split_copy_trace mirror_events(const std::vector<fake_tensor> & inputs, fake_event & ev) {
    split_copy_trace t;
    for (const auto & input : inputs) {
        if (input.flag_input) {
            ev.syncs++;
            t.n_event_syncs++;
        } else {
            ev.waits++;
            t.n_event_waits++;
        }
        fake_tensor cpy;
        fake_copy(input, cpy);
        t.copied.push_back(cpy.data);
    }
    return t;
}

static void test_light_sync() {
    // 0, 1, 4, 9 inputs; mixed flags; odd sizes (byte-exactness)
    for (size_t n_in : { size_t(0), size_t(1), size_t(4), size_t(9) }) {
        std::vector<fake_tensor> inputs;
        for (size_t i = 0; i < n_in; i++) {
            fake_tensor t;
            t.flag_input = (i % 2) == 0;
            t.data.resize(3 + i * 17);
            for (size_t k = 0; k < t.data.size(); k++) {
                t.data[k] = (uint8_t)(i * 31 + k);
            }
            inputs.push_back(t);
        }

        // events == NULL: regular vs light
        fake_backend b_reg;
        b_reg.device_busy = true;
        auto reg = mirror_regular(b_reg, inputs);

        fake_backend b_light;
        b_light.device_busy = true;
        auto light = mirror_light(b_light, inputs, /*light_enabled=*/true);

        CHECK(reg.copied.size() == light.copied.size());
        for (size_t i = 0; i < reg.copied.size(); i++) {
            CHECK(reg.copied[i].size() == light.copied[i].size());
            CHECK(memcmp(reg.copied[i].data(), light.copied[i].data(), reg.copied[i].size()) == 0);
        }
        // sync collapse: n_inputs drains -> exactly 1 (0 inputs -> 0)
        CHECK(reg.n_syncs == (int) n_in);
        CHECK(light.n_syncs == (n_in > 0 ? 1 : 0));
        // overwrite safety: no copy may run while the previous graph is reading
        for (size_t i = 0; i < reg.copy_before_drain.size(); i++) {
            CHECK(!reg.copy_before_drain[i]);
            CHECK(!light.copy_before_drain[i]);
        }

        // env unset: historical path byte-for-byte (drains per input)
        fake_backend b_off;
        b_off.device_busy = true;
        auto off = mirror_light(b_off, inputs, /*light_enabled=*/false);
        CHECK(off.n_syncs == reg.n_syncs);
        for (size_t i = 0; i < off.copied.size(); i++) {
            CHECK(memcmp(off.copied[i].data(), reg.copied[i].data(), off.copied[i].size()) == 0);
        }

        // events != NULL: the light gate never engages (hoist requires NULL events)
        fake_event ev;
        auto ev_trace = mirror_events(inputs, ev);
        CHECK(ev_trace.copied.size() == reg.copied.size());
        for (size_t i = 0; i < ev_trace.copied.size(); i++) {
            CHECK(memcmp(ev_trace.copied[i].data(), reg.copied[i].data(), ev_trace.copied[i].size()) == 0);
        }
        CHECK(ev.syncs + ev.waits == (int) n_in);
    }
}

// ---- mirror 2: ggml_cuda_graph_compat_verdict ----------------------------->

struct fake_cuda_graph {
    uint64_t compat_uid = 0;
    bool compat_verdict = false;
    int n_scans = 0;
};

struct fake_cgraph {
    uint64_t uid = 0;
    bool compatible = true;   // what the scan would return
};

static bool fake_scan(const fake_cgraph & g) {
    return g.compatible;
}

// gate parameter mirrors the GGML_CUDA_COMPAT_CACHE static
static bool mirror_compat_verdict(fake_cuda_graph & graph, const fake_cgraph & cgraph, bool cache_enabled) {
    if (cache_enabled && cgraph.uid != 0 && cgraph.uid == graph.compat_uid) {
        return graph.compat_verdict;
    }
    graph.n_scans++;
    const bool ok = fake_scan(cgraph);
    if (cache_enabled) {
        graph.compat_uid = cgraph.uid;
        graph.compat_verdict = ok;
    }
    return ok;
}

static void test_compat_cache() {
    // stable uid: one scan, cached verdict equals the scan verdict
    {
        fake_cuda_graph g;
        fake_cgraph cg;
        cg.uid = 42;
        cg.compatible = true;
        CHECK(mirror_compat_verdict(g, cg, true) == true);
        for (int i = 0; i < 8; i++) {
            CHECK(mirror_compat_verdict(g, cg, true) == true);
        }
        CHECK(g.n_scans == 1);

        // incompatible verdict is cached too (still one scan)
        fake_cuda_graph g2;
        fake_cgraph cg2;
        cg2.uid = 7;
        cg2.compatible = false;
        CHECK(mirror_compat_verdict(g2, cg2, true) == false);
        for (int i = 0; i < 8; i++) {
            CHECK(mirror_compat_verdict(g2, cg2, true) == false);
        }
        CHECK(g2.n_scans == 1);
    }

    // uid change (graph rebuild): rescan, new verdict cached
    {
        fake_cuda_graph g;
        fake_cgraph cg1, cg2;
        cg1.uid = 1;
        cg1.compatible = true;
        cg2.uid = 2;
        cg2.compatible = false;
        CHECK(mirror_compat_verdict(g, cg1, true) == true);
        CHECK(mirror_compat_verdict(g, cg1, true) == true);
        CHECK(g.n_scans == 1);
        CHECK(mirror_compat_verdict(g, cg2, true) == false);
        CHECK(mirror_compat_verdict(g, cg2, true) == false);
        CHECK(g.n_scans == 2);
        CHECK(g.compat_uid == 2);
    }

    // uid 0 (unsched'd graph): never cached, always scans
    {
        fake_cuda_graph g;
        fake_cgraph cg;
        cg.uid = 0;
        cg.compatible = true;
        for (int i = 0; i < 5; i++) {
            CHECK(mirror_compat_verdict(g, cg, true) == true);
        }
        CHECK(g.n_scans == 5);
        CHECK(g.compat_uid == 0);
    }

    // env unset: historical behavior, rescan every call, same verdicts
    {
        fake_cuda_graph g;
        fake_cgraph cg;
        cg.uid = 9;
        cg.compatible = true;
        for (int i = 0; i < 5; i++) {
            CHECK(mirror_compat_verdict(g, cg, false) == true);
        }
        CHECK(g.n_scans == 5);
        CHECK(g.compat_uid == 0);
    }
}

// ---- mirror 3: the graph_compute mode classification (timeline line) ------>

static const char * mirror_mode(bool use_cuda_graph, bool update_required) {
    return use_cuda_graph ? (update_required ? "capture" : "replay") : "direct";
}

static void test_mode_classification() {
    CHECK(strcmp(mirror_mode(false, false), "direct") == 0);
    CHECK(strcmp(mirror_mode(true, true), "capture") == 0);
    CHECK(strcmp(mirror_mode(true, false), "replay") == 0);
}

int main() {
    test_light_sync();
    test_compat_cache();
    test_mode_classification();
    if (n_fail == 0) {
        fprintf(stderr, "ALL PASS\n");
        return 0;
    }
    fprintf(stderr, "%d FAILURES\n", n_fail);
    return 1;
}
