// ldsy defect probe: iq3_s, K=5120, N=2 (one CTA), T=2, tiny vy; runs the
// share arm and ldsy arms and hex-dumps both dsts for the first mismatches.
#include "ggml.h"
#include "/tmp/mmvq_dbg.cu"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

static std::vector<float> build_src1(int K, int T) {
    std::vector<float> x((size_t) K*T);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k)
            x[(size_t) t*K + k] = ((float)(((size_t) k*2654435761u + 97u*(unsigned) t) % 2001u) - 1000.0f) / 256.0f;
    return x;
}

static void quantize_q8_1_row(const float * x, int K, int nblocks, uint8_t * out) {
    for (int b = 0; b < nblocks; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < QK8_1; ++i) amax = std::max(amax, fabsf(x[(size_t) b*QK8_1 + i]));
        const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
        const float inv = amax == 0.0f ? 0.0f : 127.0f / amax;
        int sum = 0;
        for (int i = 0; i < QK8_1; ++i) {
            const int v = (int) roundf(x[(size_t) b*QK8_1 + i] * inv);
            out[(size_t) b*36 + 4 + i] = (uint8_t) v;
            sum += v;
        }
        const ggml_half2 ds = { ggml_fp32_to_fp16(d), ggml_fp32_to_fp16((float) sum) };
        memcpy(out + (size_t) b*36, &ds, 4);
    }
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    constexpr ggml_type type = GGML_TYPE_IQ3_S;
    constexpr int T = 4;
    const int K = 5120, N = 20;
    const int stride_row_x = K / QK_K;      // 20
    const int stride_col_y = K / QK8_1;     // 160

    // iq3_s weights: REAL gguf bytes (offset-delta law)
    const int Narg = N;
    long woff = 4301102016L, wlen = 38297600L;
    FILE * gf = fopen("/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf", "rb");
    if (!gf) { printf("no gguf\n"); return 1; }
    std::vector<uint8_t> wx_host((size_t) stride_row_x*Narg*110);
    fseek(gf, woff, SEEK_SET);
    if (fread(wx_host.data(), 1, wx_host.size(), gf) != wx_host.size()) { printf("read fail\n"); return 1; }
    fclose(gf); (void) wlen;
    void * vx = nullptr;
    HIP_CHECK(hipMalloc(&vx, wx_host.size()));
    HIP_CHECK(hipMemcpy(vx, wx_host.data(), wx_host.size(), hipMemcpyHostToDevice));

    std::vector<float> src1 = build_src1(K, T);
    std::vector<uint8_t> vy_host((size_t) T*stride_col_y*36, 0);
    for (int t = 0; t < T; ++t)
        quantize_q8_1_row(src1.data() + (size_t) t*K, K, stride_col_y, vy_host.data() + (size_t) t*stride_col_y*36);
    block_q8_1 * vy = nullptr;
    HIP_CHECK(hipMalloc((void **) &vy, vy_host.size()));
    HIP_CHECK(hipMemcpy(vy, vy_host.data(), vy_host.size(), hipMemcpyHostToDevice));

    float * dst = nullptr;
    HIP_CHECK(hipMalloc((void **) &dst, (size_t) N*T*sizeof(float)));
    dim3 grid((unsigned)((N + 1) / 2), 1, 1), block(64, 2, 1);
    const ggml_cuda_mm_fusion_args_device fusion{};

    std::vector<float> out((size_t) N*T), ref;
    const char * names[] = {"share", "ldsy4", "ldsy8", "ldsy16"};
    for (int arm = 0; arm < 4; ++arm) {
        HIP_CHECK(hipMemset(dst, 0, (size_t) N*T*sizeof(float)));
        if (arm == 0) {
            mul_mat_vec_q<GGML_TYPE_IQ3_S, 4, false, false, false, false, false, 4>
                <<<grid, block>>>(vx, vy, vy, nullptr, fusion, dst, K, make_uint3(0,0,0),
                    stride_row_x, stride_col_y, N, init_fastdiv_values(1), 1, T*stride_col_y, N*T,
                    init_fastdiv_values(1), 1, T*stride_col_y, N*T, 0, true);
        } else {
            const int run = arm == 1 ? 4 : arm == 2 ? 8 : 16;
            if (run == 4)
                mul_mat_vec_q<GGML_TYPE_IQ3_S, 4, false, false, false, false, true, 4><<<grid, block>>>(vx, vy, vy, nullptr, fusion, dst, K, make_uint3(0,0,0), stride_row_x, stride_col_y, N, init_fastdiv_values(1), 1, T*stride_col_y, N*T, init_fastdiv_values(1), 1, T*stride_col_y, N*T, 0, true);
            else if (run == 8)
                mul_mat_vec_q<GGML_TYPE_IQ3_S, 4, false, false, false, false, true, 8><<<grid, block>>>(vx, vy, vy, nullptr, fusion, dst, K, make_uint3(0,0,0), stride_row_x, stride_col_y, N, init_fastdiv_values(1), 1, T*stride_col_y, N*T, init_fastdiv_values(1), 1, T*stride_col_y, N*T, 0, true);
            else if (run == 16)
                mul_mat_vec_q<GGML_TYPE_IQ3_S, 4, false, false, false, false, true, 16><<<grid, block>>>(vx, vy, vy, nullptr, fusion, dst, K, make_uint3(0,0,0), stride_row_x, stride_col_y, N, init_fastdiv_values(1), 1, T*stride_col_y, N*T, init_fastdiv_values(1), 1, T*stride_col_y, N*T, 0, true);
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(out.data(), dst, out.size()*sizeof(float), hipMemcpyDeviceToHost));
        printf("%-7s: %d floats\n", names[arm], (int) out.size());
        if (arm == 0) {
            ref = out;
        } else {
            size_t nmis = 0;
            for (size_t e = 0; e < out.size(); ++e)
                if (memcmp(ref.data() + e, out.data() + e, 4) != 0) ++nmis;
            printf("  mismatches vs share: %zu / %zu\n", nmis, out.size());
        }
    }
    return 0;
}
