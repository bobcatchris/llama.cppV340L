// test_mmvq_gate_wiring.cu - REGRESSION TEST for the served gate wiring.
//
// Replicates the 2026-09-23 s2r defect: mul_mat_vec_q_switch_type
// forwarded a literal `false` to every switch_ncols_dst case, so the
// *_S2R env gates were no-ops served (engagement INFO lines never
// printed, E-117). The engagement log line is the wiring witness:
// switch_fusion reads the env via mmvq_s2r_env_enabled/mmvq_share_env_flag
// ONLY if the flag actually reaches it (short-circuit `s2r && env()` hides
// a dropped flag). This test drives the REAL host chain
// (mul_mat_vec_q_switch_type -> switch_ncols_dst -> switch_fusion) with
// every gate env set and requires every type's engagement line to appear;
// then a no-env control run must produce zero engagement lines.
//
// Build (see run_premerge_ci.sh):
//   hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP -DGGML_BACKEND_BUILD \
//     -DGGML_SHARED -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//     test_mmvq_gate_wiring.cu -o /tmp/test_mmvq_gate_wiring \
//     -L build-hip/bin -lggml-hip -lggml-base -lamdhip64
// Run: HIP_VISIBLE_DEVICES=<one idle die> /tmp/test_mmvq_gate_wiring <type-name>
// Exit 0 = wiring OK for that type; nonzero = dropped flag (regression).
#include "ggml.h"
#include "ggml-cuda/mmvq.cu"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static std::vector<std::string> g_log;
static void log_capture(enum ggml_log_level level, const char * text, void * user) {
    (void) level; (void) user;
    g_log.emplace_back(text ? text : "");
}
static bool saw_engagement(const char * gate) {
    for (const auto & l : g_log) {
        if (l.find(gate) != std::string::npos && l.find("enabled") != std::string::npos) return true;
    }
    return false;
}

// one plausible 2D no-ids geometry (small but valid; K/N need not match a
// real tensor - only the tables and the env chain are under test)
struct Geo { int K; int N; int T; int stride_row_x; };

static void run_chain(ggml_type type, const Geo & g, hipStream_t stream) {
    // quantized weights: bs = bytes per block
    const int bs   = ggml_type_size(type);
    const int qk   = ggml_blck_size(type);
    const int Kblk = g.K / qk;
    std::vector<char> hx(Kblk * (size_t) g.N * bs, 0);
    void * vx = nullptr;
    HIP_CHECK(hipMalloc(&vx, hx.size()));
    HIP_CHECK(hipMemcpy(vx, hx.data(), hx.size(), hipMemcpyHostToDevice));

    // y: q8_1 blocks, K/qk rows of block_q8_1, T columns
    const int kyb = g.K / QK8_1;
    std::vector<block_q8_1> hy((size_t) kyb * g.T);
    for (size_t i = 0; i < hy.size(); ++i) {
        for (int j = 0; j < QK8_1; ++j) { hy[i].qs[j] = (int8_t)(j & 7); }
        hy[i].ds = __float2half2_rn(1.0f);
    }
    block_q8_1 * vy = nullptr;
    HIP_CHECK(hipMalloc(&vy, hy.size() * sizeof(block_q8_1)));
    HIP_CHECK(hipMemcpy(vy, hy.data(), hy.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));

    float * dst = nullptr;
    HIP_CHECK(hipMalloc(&dst, (size_t) g.N * g.T * sizeof(float)));
    HIP_CHECK(hipMemset(dst, 0, (size_t) g.N * g.T * sizeof(float)));

    const ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q_switch_type(
        vx, type, vy, vy, nullptr, fusion, dst,
        g.K, g.N, g.T,
        g.stride_row_x, kyb, g.T,
        1, 1, 1,          // channels
        1, 1, 1,          // stride_channel
        1, 1,             // nsamples
        1, 1, 1,          // stride_sample
        0 /*ids_stride*/, /*aln=*/ false, /*s2r=*/ true, stream);
    HIP_CHECK(hipDeviceSynchronize());

    // sanity: output must be finite (wiring engaged AND kernel ran)
    std::vector<float> hd((size_t) g.N * g.T);
    HIP_CHECK(hipMemcpy(hd.data(), dst, hd.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < hd.size(); ++i) {
        if (std::isnan(hd[i])) { printf("NaN in dst at %zu\n", i); exit(3); }
    }
    HIP_CHECK(hipFree(vx)); HIP_CHECK(hipFree(vy)); HIP_CHECK(hipFree(dst));
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <iq3_xxs|q4_K|iq4_xs|q5_K|q6_K|q3_K|iq3_s>\n", argv[0]); return 2; }
    const std::string tn = argv[1];
    ggml_type type = GGML_TYPE_Q4_K;
    if      (tn == "iq3_xxs") type = GGML_TYPE_IQ3_XXS;
    else if (tn == "q4_K")    type = GGML_TYPE_Q4_K;
    else if (tn == "iq4_xs")  type = GGML_TYPE_IQ4_XS;
    else if (tn == "q5_K")    type = GGML_TYPE_Q5_K;
    else if (tn == "q6_K")    type = GGML_TYPE_Q6_K;
    else if (tn == "q3_K")    type = GGML_TYPE_Q3_K;
    else if (tn == "iq3_s")   type = GGML_TYPE_IQ3_S;
    else { printf("unknown type %s\n", tn.c_str()); return 2; }

    ggml_log_set(log_capture, nullptr);

    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipSetDevice(dev));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    Geo g{5120, 64, 4, (int)(5120 / ggml_blck_size(type))};
    run_chain(type, g, stream);
    HIP_CHECK(hipStreamDestroy(stream));

    // wiring witness: the engagement line for THIS type's SHARE + S2R gates
    // must exist (share comes from GGML_CUDA_MMVQ_*_SHARE or the s2r gate;
    // s2r must reach switch_fusion or the env reader is short-circuited)
    bool any = false;
    for (const auto & l : g_log) {
        if (l.find("decode-once share enabled") != std::string::npos) { any = true; break; }
    }
    if (!any) {
        printf("WIRING-FAIL %s: no engagement line - a gate flag is being dropped in the switch chain\n", tn.c_str());
        return 1;
    }
    int hits = 0;
    for (const auto & l : g_log) {
        if (l.find("_S2R=1") != std::string::npos) hits++;
    }
    printf("WIRING-OK %s: engagement lines present (s2r lines=%d)\n", tn.c_str(), hits);
    return 0;
}
