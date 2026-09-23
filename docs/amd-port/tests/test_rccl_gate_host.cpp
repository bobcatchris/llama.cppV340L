// rccl-ext desk host test: the nccl allreduce size-class gate + the
// GGML_RCCL_FP32=1 extension.
//
// Mirrors ggml_backend_cuda_comm_allreduce_nccl and ggml_backend_cuda_comm_init
// (ggml/src/ggml-cuda/ggml-cuda.cu):
//
//   reduce_fp32 = nccl_fp32 ||
//       (n_backends <= 2 && ne < 32768) ||
//       (n_backends == 3 && ne < 131072) ||
//       (n_backends >= 4 && ne < 262144)
//
//   gate parse: env != nullptr && atoi(env) == 1   (house pattern,
//   GGML_CUDA_MMVQ_GROUP); the flag is inert unless the nccl mode is active
//   (HIP default is the butterfly, which returns false per call and lets the
//   meta allreduce_fallback run).
//
// The table pins the class of the real served boundary shapes
// (Qwen3.8-27B-ASCII-P1M, n_embd 5120):
//   decode verify  T=4        ne =   20480 (80 KB)  -> fp32 in every config
//   prefill ubatch 512        ne = 2621440 (10 MiB) -> bf16-compress by
//   default (today's nccl behavior), fp32 with GGML_RCCL_FP32=1 (the
//   extension: same sum-order dust class as the signed-off decode path),
//   butterfly when the mode is not nccl.
//
// Build/run (host-only, no GPU, no libggml link):
//   g++ -O2 -std=c++17 -o /tmp/test_rccl_gate_host
//       docs/amd-port/tests/test_rccl_gate_host.cpp && /tmp/test_rccl_gate_host

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

static int test_failures = 0;
#define REQUIRE(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); test_failures++; } } while (0)

// ---- mirrors (keep in sync with ggml-cuda.cu) ----

static bool gate_env_parse(const char * env) {
    return env != nullptr && atoi(env) == 1;
}

struct comm_cfg {
    bool nccl_mode; // GGML_CUDA_ALLREDUCE resolved to the nccl path
    bool nccl_fp32; // GGML_RCCL_FP32=1
};

static bool reduce_is_fp32(const comm_cfg & c, size_t n_backends, int64_t ne) {
    if (!c.nccl_mode) {
        return false; // butterfly try_allreduce returns false -> meta fallback
    }
    return c.nccl_fp32 ||
        ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144));
}

// what runs at the boundary, for the receipt tables
static const char * boundary_path(const comm_cfg & c, size_t n_backends, int64_t ne) {
    if (!c.nccl_mode) {
        return "butterfly";
    }
    return reduce_is_fp32(c, n_backends, ne) ? "rccl-fp32" : "rccl-bf16";
}

int main() {
    // --- env parse (house pattern) ---
    REQUIRE(!gate_env_parse(nullptr), "unset env must keep the gate off");
    REQUIRE(!gate_env_parse(""), "empty env must keep the gate off");
    REQUIRE(!gate_env_parse("0"), "GGML_RCCL_FP32=0 must keep the gate off");
    REQUIRE( gate_env_parse("1"), "GGML_RCCL_FP32=1 must enable the gate");
    REQUIRE(!gate_env_parse("2"), "only the exact value 1 enables the gate");

    // --- served shapes, TP3 (n_backends = 3) ---
    const comm_cfg unset { true, false }; // GGML_CUDA_ALLREDUCE=nccl, gate off
    const comm_cfg ext   { true, true  }; // + GGML_RCCL_FP32=1
    const comm_cfg nofrc { false, false }; // env unset on HIP = butterfly

    REQUIRE(boundary_path(unset, 3, 20480)   == std::string("rccl-fp32"),  "decode verify 4x5120 must stay rccl-fp32");
    REQUIRE(boundary_path(unset, 3, 2621440) == std::string("rccl-bf16"),  "prefill 512x5120 stays rccl-bf16 with gate off (today's behavior)");
    REQUIRE(boundary_path(ext,   3, 20480)   == std::string("rccl-fp32"),  "decode verify unchanged by the extension");
    REQUIRE(boundary_path(ext,   3, 2621440) == std::string("rccl-fp32"),  "prefill 512x5120 becomes rccl-fp32 with GGML_RCCL_FP32=1");
    REQUIRE(boundary_path(nofrc, 3, 20480)   == std::string("butterfly"),  "mode unset on HIP keeps the butterfly at every size");
    REQUIRE(boundary_path(nofrc, 3, 2621440) == std::string("butterfly"),  "mode unset on HIP keeps the butterfly at every size");

    // --- exact threshold boundaries, per rank count (n_backends) ---
    struct { size_t n; int64_t below; int64_t at; } thr[3] = {
        { 2, 32767, 32768 },
        { 3, 131071, 131072 },
        { 4, 262143, 262144 },
    };
    for (const auto & t : thr) {
        REQUIRE( reduce_is_fp32(unset, t.n, t.below), "ne just below the n=%zu threshold must be fp32", t.n);
        REQUIRE(!reduce_is_fp32(unset, t.n, t.at),    "ne at the n=%zu threshold must be bf16 (heuristic unchanged)", t.n);
        REQUIRE( reduce_is_fp32(ext,   t.n, t.at),    "the gate must extend fp32 past the n=%zu threshold", t.n);
        REQUIRE( reduce_is_fp32(ext,   t.n, t.below), "the gate must keep fp32 below the n=%zu threshold", t.n);
    }

    // --- prefill ladder at TP3: every chunk size 1..512 rows lands fp32 under the gate ---
    for (int64_t rows : { 1, 2, 4, 128, 256, 511, 512 }) {
        const int64_t ne = rows * 5120;
        if (ne < 131072) {
            REQUIRE(boundary_path(unset, 3, ne) == std::string("rccl-fp32"), "small prefill chunk stays fp32");
        } else {
            REQUIRE(boundary_path(unset, 3, ne) == std::string("rccl-bf16"), "large prefill chunk stays bf16 with gate off");
        }
        REQUIRE(boundary_path(ext, 3, ne) == std::string("rccl-fp32"), "every prefill chunk size is fp32 with the gate on");
    }

    if (test_failures == 0) {
        printf("test_rccl_gate_host: ALL PASS\n");
        return 0;
    }
    printf("test_rccl_gate_host: %d FAILURES\n", test_failures);
    return 1;
}
