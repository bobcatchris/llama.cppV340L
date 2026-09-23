// rccl-ext desk host test: the prefill size-class gate (GGML_RCCL_PREFILL).
//
// Mirrors ggml_backend_cuda_comm_allreduce_tensor, _allreduce_nccl and
// _comm_init (ggml/src/ggml-cuda/ggml-cuda.cu):
//
//   dispatch: class != UNSET && ne >= 131072 -> butterfly (return false)
//   when class == BUTTERFLY or the nccl comms are not up (mode not nccl or
//   init failed); otherwise the selected try_allreduce runs
//   nccl class pick: at ne >= 131072 the gate decides (f32 -> fp32 ring,
//   bf16 -> bf16-compress); below it the upstream heuristic (fp32 when
//   ne < thr(n): 32768 / 131072 / 262144 for n <= 2 / == 3 / >= 4)
//   parse: exact "f32" | "bf16" | "butterfly"; anything else warns and
//   behaves as unset
//
// Pins that unset = today's behavior byte-identically, both gate classes at
// the exact 131071/131072 boundary, the fallback to the butterfly when the
// comms are not up, and the served shapes (Qwen3.8-27B-ASCII-P1M,
// n_embd 5120): decode verify T=4 ne = 20480, prefill chunk ne = 2621440.
//
// Build/run (host-only, no GPU, no libggml link):
//   g++ -O2 -std=c++17 -o /tmp/test_rccl_gate_host
//       docs/amd-port/tests/test_rccl_gate_host.cpp && /tmp/test_rccl_gate_host

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

static int test_failures = 0;
#define REQUIRE(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); test_failures++; } } while (0)

// ---- mirrors (keep in sync with ggml-cuda.cu) ----

enum prefill_class { PRE_UNSET, PRE_F32, PRE_BF16, PRE_BUTTERFLY };

static prefill_class gate_env_parse(const char * env) {
    if (env == nullptr) {
        return PRE_UNSET;
    }
    if (strcmp(env, "f32") == 0) {
        return PRE_F32;
    }
    if (strcmp(env, "bf16") == 0) {
        return PRE_BF16;
    }
    if (strcmp(env, "butterfly") == 0) {
        return PRE_BUTTERFLY;
    }
    return PRE_UNSET; // unknown value warns and behaves as unset
}

static constexpr int64_t GATE_NE = 131072;

struct comm_cfg {
    enum mode { MODE_NCCL, MODE_INTERNAL, MODE_NONE } mode; // resolved init chain
    bool          nccl_up; // ncclCommInitAll succeeded
    prefill_class cls;     // GGML_RCCL_PREFILL
};

// what runs at the boundary, for the receipt tables (mirrors dispatch + nccl pick)
static const char * boundary_path(const comm_cfg & c, size_t n_backends, int64_t ne) {
    if (c.cls != PRE_UNSET && ne >= GATE_NE) {
        if (c.cls == PRE_BUTTERFLY || !c.nccl_up) {
            return "butterfly";
        }
    }
    switch (c.mode) {
        case comm_cfg::MODE_NCCL: {
            bool reduce_fp32;
            if (c.cls != PRE_UNSET && ne >= GATE_NE) {
                reduce_fp32 = c.cls == PRE_F32;
            } else {
                reduce_fp32 = (n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) ||
                    (n_backends >= 4 && ne < 262144);
            }
            return reduce_fp32 ? "rccl-fp32" : "rccl-bf16";
        }
        case comm_cfg::MODE_INTERNAL:
            return "internal";
        default:
            return "butterfly";
    }
}

int main() {
    // --- env parse: only the exact class names select a class ---
    REQUIRE(gate_env_parse(nullptr)     == PRE_UNSET,      "unset env must keep the upstream behavior");
    REQUIRE(gate_env_parse("")          == PRE_UNSET,      "empty env must keep the upstream behavior");
    REQUIRE(gate_env_parse("f32")       == PRE_F32,        "f32 must select the fp32 ring class");
    REQUIRE(gate_env_parse("bf16")      == PRE_BF16,       "bf16 must select the bf16-compress class");
    REQUIRE(gate_env_parse("butterfly") == PRE_BUTTERFLY,  "butterfly must select the staging class");
    REQUIRE(gate_env_parse("F32")       == PRE_UNSET,      "class names are case-sensitive, like GGML_CUDA_ALLREDUCE");
    REQUIRE(gate_env_parse("1")         == PRE_UNSET,      "numeric values are not classes");
    REQUIRE(gate_env_parse("fp32")      == PRE_UNSET,      "unknown names behave as unset (with a warn)");

    // --- served shapes, TP3 (n_backends = 3) ---
    const comm_cfg today  { comm_cfg::MODE_NCCL, true,  PRE_UNSET      }; // GGML_CUDA_ALLREDUCE=nccl
    const comm_cfg f32g   { comm_cfg::MODE_NCCL, true,  PRE_F32        }; // + GGML_RCCL_PREFILL=f32
    const comm_cfg bf16g  { comm_cfg::MODE_NCCL, true,  PRE_BF16       }; // + GGML_RCCL_PREFILL=bf16
    const comm_cfg buttg  { comm_cfg::MODE_NCCL, true,  PRE_BUTTERFLY  }; // + GGML_RCCL_PREFILL=butterfly
    const comm_cfg hipdef { comm_cfg::MODE_NONE, false, PRE_UNSET      }; // env unset on HIP
    const comm_cfg hipf32 { comm_cfg::MODE_NONE, false, PRE_F32        }; // gate without the nccl path

    REQUIRE(strcmp(boundary_path(today,  3, 20480),   "rccl-fp32") == 0, "decode verify 4x5120 stays rccl-fp32 (today)");
    REQUIRE(strcmp(boundary_path(today,  3, 2621440), "rccl-bf16") == 0, "prefill 512x5120 stays rccl-bf16 (today's served class)");
    REQUIRE(strcmp(boundary_path(f32g,   3, 20480),   "rccl-fp32") == 0, "decode verify untouched by the gate");
    REQUIRE(strcmp(boundary_path(f32g,   3, 2621440), "rccl-fp32") == 0, "prefill becomes the fp32 ring under GGML_RCCL_PREFILL=f32");
    REQUIRE(strcmp(boundary_path(bf16g,  3, 20480),   "rccl-fp32") == 0, "decode verify untouched under the bf16 class");
    REQUIRE(strcmp(boundary_path(bf16g,  3, 2621440), "rccl-bf16") == 0, "prefill bf16 class is selectable explicitly");
    REQUIRE(strcmp(boundary_path(buttg,  3, 20480),   "rccl-fp32") == 0, "decode keeps rccl under the butterfly class");
    REQUIRE(strcmp(boundary_path(buttg,  3, 2621440), "butterfly") == 0, "prefill runs the butterfly under GGML_RCCL_PREFILL=butterfly");
    REQUIRE(strcmp(boundary_path(hipdef, 3, 20480),   "butterfly") == 0, "env unset on HIP keeps the butterfly (decode)");
    REQUIRE(strcmp(boundary_path(hipdef, 3, 2621440), "butterfly") == 0, "env unset on HIP keeps the butterfly (prefill)");
    REQUIRE(strcmp(boundary_path(hipf32, 3, 2621440), "butterfly") == 0, "gate is inert without the nccl path (warns at init)");

    // --- exact 131071/131072 boundary, both classes (the gate edge) ---
    REQUIRE(strcmp(boundary_path(today, 3, 131071), "rccl-fp32") == 0, "just below the gate edge: upstream fp32 (n=3)");
    REQUIRE(strcmp(boundary_path(today, 3, 131072), "rccl-bf16") == 0, "at the gate edge: upstream bf16 (n=3)");
    REQUIRE(strcmp(boundary_path(f32g,  3, 131071), "rccl-fp32") == 0, "below the edge the heuristic still rules (fp32 here)");
    REQUIRE(strcmp(boundary_path(f32g,  3, 131072), "rccl-fp32") == 0, "the fp32 class starts exactly at ne = 131072");
    REQUIRE(strcmp(boundary_path(bf16g, 3, 131071), "rccl-fp32") == 0, "the bf16 class does not reach below the edge");
    REQUIRE(strcmp(boundary_path(bf16g, 3, 131072), "rccl-bf16") == 0, "the bf16 class holds from ne = 131072");
    REQUIRE(strcmp(boundary_path(buttg, 3, 131071), "rccl-fp32") == 0, "the butterfly class does not reach below the edge");
    REQUIRE(strcmp(boundary_path(buttg, 3, 131072), "butterfly") == 0, "the butterfly class starts exactly at ne = 131072");

    // --- gate scoping vs the per-rank-count heuristic bands ---
    const comm_cfg f32g_nc = f32g;
    REQUIRE(strcmp(boundary_path(today, 2, 32767),  "rccl-fp32") == 0, "n=2 band edge below: fp32");
    REQUIRE(strcmp(boundary_path(today, 2, 32768),  "rccl-bf16") == 0, "n=2 band edge at: bf16");
    REQUIRE(strcmp(boundary_path(f32g_nc, 2, 65536),  "rccl-bf16") == 0, "f32 class does not leak into the n=2 [32768,131071) band");
    REQUIRE(strcmp(boundary_path(f32g_nc, 2, 131072), "rccl-fp32") == 0, "f32 class holds at the edge for n=2 too");
    REQUIRE(strcmp(boundary_path(today, 4, 131072),  "rccl-fp32") == 0, "n=4 heuristic: fp32 up to 262143");
    REQUIRE(strcmp(boundary_path(bf16g, 4, 131072),  "rccl-bf16") == 0, "bf16 class overrides the n=4 fp32 band from the edge");
    REQUIRE(strcmp(boundary_path(bf16g, 4, 262144),  "rccl-bf16") == 0, "bf16 class matches the n=4 heuristic at 262144");

    // --- fallback on init failure ---
    const comm_cfg nccl_fail3 { comm_cfg::MODE_NONE,     false, PRE_F32 }; // n=3: nccl fail -> internal fail (n!=2) -> none
    const comm_cfg nccl_fail2 { comm_cfg::MODE_INTERNAL, false, PRE_F32 }; // n=2: nccl fail -> internal pipeline up
    REQUIRE(strcmp(boundary_path(nccl_fail3, 3, 2621440), "butterfly") == 0, "init failure: prefill falls back to the butterfly");
    REQUIRE(strcmp(boundary_path(nccl_fail3, 3, 20480),   "butterfly") == 0, "init failure chain to none: decode is the butterfly too");
    REQUIRE(strcmp(boundary_path(nccl_fail2, 2, 2621440), "butterfly") == 0, "init failure with internal up: prefill still the butterfly");
    REQUIRE(strcmp(boundary_path(nccl_fail2, 2, 20480),   "internal")  == 0, "init failure leaves decode on the internal pipeline");
    const comm_cfg bf16_fail3 { comm_cfg::MODE_NONE, false, PRE_BF16 };
    REQUIRE(strcmp(boundary_path(bf16_fail3, 3, 2621440), "butterfly") == 0, "the bf16 class needs comms too and falls back");

    // --- full prefill ladder, every chunk 1..512 rows at TP3 ---
    for (int64_t rows = 1; rows <= 512; ++rows) {
        const int64_t ne = rows * 5120;
        const bool small = ne < GATE_NE;
        const char * want_today = small ? "rccl-fp32" : "rccl-bf16";
        REQUIRE(strcmp(boundary_path(today, 3, ne), want_today) == 0,
                "ladder rows %lld: unset must match today's split", (long long) rows);
        REQUIRE(strcmp(boundary_path(f32g,  3, ne), "rccl-fp32") == 0,
                "ladder rows %lld: f32 class everywhere", (long long) rows);
        REQUIRE(strcmp(boundary_path(bf16g, 3, ne), want_today) == 0,
                "ladder rows %lld: bf16 class equals today at n=3", (long long) rows);
        REQUIRE(strcmp(boundary_path(buttg, 3, ne), small ? "rccl-fp32" : "butterfly") == 0,
                "ladder rows %lld: butterfly class only above the edge", (long long) rows);
    }

    if (test_failures == 0) {
        printf("test_rccl_gate_host: ALL PASS\n");
        return 0;
    }
    printf("test_rccl_gate_host: %d FAILURES\n", test_failures);
    return 1;
}
