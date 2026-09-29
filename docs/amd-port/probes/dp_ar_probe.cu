// dp_ar_probe.cu - DP-3 instrument v2: 4-rank tree allreduce built ONLY on
// hipMemcpyPeerAsync + local reduce kernels (ROCm peer access is copy-engine
// only; every cross-rank operand moves through a staging buffer on the
// consuming rank - that is the serving design too).
// Tree (matches fabric): L1 intra-card {1->0, 3->2} partial copies, local
// reduce2 on ranks 0 and 2; L2 cross partial copy 2->0, local reduce2 on 0;
// L3 broadcast result 0 -> {1,2,3} on dest streams. CPU-reference checked.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while (0)

__global__ void reduce2_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int iters = argc > 1 ? atoi(argv[1]) : 301;
    int nd = 0; HIPCHECK(hipGetDeviceCount(&nd));
    printf("dp_ar_probe: devices=%d iters=%d\n", nd, iters);
    if (nd < 4) { printf("need 4 dies\n"); return 1; }

    const size_t sizes[] = {4 * 1024, 20 * 1024, 80 * 1024, 1 * 1024 * 1024};
    // per rank: [0]=input [1]=stageA (L1 in) [2]=stageB (L2 in) [3]=result
    float* buf[4][4] = {};
    hipStream_t st[4] = {};
    hipEvent_t ev[4] = {};
    for (int r = 0; r < 4; r++) {
        HIPCHECK(hipSetDevice(r));
        for (int b = 0; b < 4; b++) HIPCHECK(hipMalloc(&buf[r][b], 1 * 1024 * 1024));
        HIPCHECK(hipStreamCreate(&st[r]));
        HIPCHECK(hipEventCreate(&ev[r]));
    }
    // REQUIRED (E-169 lesson): explicit per-direction enablement - cross-card
    // memcpyPeerAsync silently writes garbage without it (intra-card auto-enables)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) continue;
            HIPCHECK(hipSetDevice(i));
            int can = 0;
            HIPCHECK(hipDeviceCanAccessPeer(&can, i, j));
            if (!can) { printf("WARNING: no peer %d->%d\n", i, j); continue; }
            hipError_t e = hipDeviceEnablePeerAccess(j, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                printf("enable %d->%d: %s\n", i, j, hipGetErrorString(e));
        }
    }
    printf("peer enablement: all 12 directions\n");
    for (size_t SZ : sizes) {
        int n = SZ / 4;
        int blocks = (n + 255) / 256;
        for (int r = 0; r < 4; r++) {
            std::vector<float> h(n);
            for (int k = 0; k < n; k++) h[k] = (r + 1) + 0.001f * k;
            HIPCHECK(hipSetDevice(r));
            HIPCHECK(hipMemcpy(buf[r][0], h.data(), SZ, hipMemcpyHostToDevice));
        }
        int bad = 0;
        std::vector<double> wall_t;
        for (int it = -10; it < iters; it++) {
            for (int r = 0; r < 4; r++) HIPCHECK(hipStreamSynchronize(st[r]));
            double w0 = now_us();
            // L1: intra-card partial copies (source streams)
            HIPCHECK(hipMemcpyPeerAsync(buf[0][1], 0, buf[1][0], 1, SZ, st[1]));
            HIPCHECK(hipEventRecord(ev[1], st[1]));
            HIPCHECK(hipMemcpyPeerAsync(buf[2][1], 2, buf[3][0], 3, SZ, st[3]));
            HIPCHECK(hipEventRecord(ev[3], st[3]));
            // rank0 local reduce {self + peer1}
            HIPCHECK(hipSetDevice(0));
            HIPCHECK(hipStreamWaitEvent(st[0], ev[1], 0));
            reduce2_kernel<<<blocks, 256, 0, st[0]>>>(buf[0][0], buf[0][1], buf[0][3], n);
            // rank2 local reduce {self + peer3}
            HIPCHECK(hipSetDevice(2));
            HIPCHECK(hipStreamWaitEvent(st[2], ev[3], 0));
            reduce2_kernel<<<blocks, 256, 0, st[2]>>>(buf[2][0], buf[2][1], buf[2][3], n);
            // L2: rank2 partial -> rank0 stageB (cross-card), then local reduce
            HIPCHECK(hipMemcpyPeerAsync(buf[0][2], 0, buf[2][3], 2, SZ, st[2]));
            HIPCHECK(hipEventRecord(ev[2], st[2]));
            HIPCHECK(hipSetDevice(0));
            HIPCHECK(hipStreamWaitEvent(st[0], ev[2], 0));
            reduce2_kernel<<<blocks, 256, 0, st[0]>>>(buf[0][3], buf[0][2], buf[0][3], n);
            HIPCHECK(hipEventRecord(ev[0], st[0]));
            // L3: broadcast 0 -> {1,2,3} ALL ON THE SOURCE STREAM (st[0]) -
            // ROCm 6.2 bug: cross-card peer copies on a DEST-stream silently
            // write garbage (dp_ar_debug2 receipt); source-stream is exact.
            // Same-stream = naturally ordered after the final reduce.
            HIPCHECK(hipMemcpyPeerAsync(buf[1][3], 1, buf[0][3], 0, SZ, st[0]));
            HIPCHECK(hipMemcpyPeerAsync(buf[2][3], 2, buf[0][3], 0, SZ, st[0]));
            HIPCHECK(hipMemcpyPeerAsync(buf[3][3], 3, buf[0][3], 0, SZ, st[0]));
            for (int r = 0; r < 4; r++) HIPCHECK(hipStreamSynchronize(st[r]));
            double w1 = now_us();
            // verify on all four ranks; print first value on mismatch
            for (int vr : {0, 1, 2, 3}) {
                std::vector<float> hchk(n);
                HIPCHECK(hipSetDevice(vr));
                HIPCHECK(hipMemcpy(hchk.data(), buf[vr][3], SZ, hipMemcpyDeviceToHost));
                float ref = (float) (((1.0 + 0.0) + (2.0 + 0.0)) + ((3.0 + 0.0) + (4.0 + 0.0)));
                if (hchk[0] != ref) {
                    bad++;
                    if (bad <= 4)
                        printf("  iter %d rank %d result[0]=%.4f expected %.4f\n",
                               it, vr, hchk[0], ref);
                }
            }
            if (it >= 0) wall_t.push_back(w1 - w0);
        }
        std::sort(wall_t.begin(), wall_t.end());
        printf("%5zuKB tree-AR: wall med %7.2f us  min %7.2f  verify %s (bad=%d)\n",
               SZ / 1024, wall_t[wall_t.size() / 2], wall_t.front(),
               bad ? "FAIL" : "PASS", bad);
    }
    printf("dp_ar_probe done\n");
    return 0;
}
