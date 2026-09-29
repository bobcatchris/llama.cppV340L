// W7 SDMA cell: pinned-async copy bandwidth (the SDMA-engine door for host memory in the
// AR data plane). Question: do the copy engines beat the 6.5 GB/s kernel-read path and the
// SHM ring's effective 3.15 GiB/s? Pre-registered: door OPEN if sustained >=10 GB/s either
// direction; CLOSED if <6.5 GB/s (no better than what hosted KV already gets).
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t N = 256u << 20; // 256 MiB
    void *d = nullptr, *h = nullptr;
    hipError_t e1 = hipMalloc(&d, N); printf("hipMalloc: %s\n", hipGetErrorString(e1)); if (e1 != hipSuccess) return 1;
    hipError_t e2 = hipHostMalloc(&h, N, 0); printf("hipHostMalloc: %s\n", hipGetErrorString(e2)); if (e2 != hipSuccess) return 1;
    hipStream_t s; hipStreamCreate(&s);
    const int iters = 8, depth = 4; // 8 GiB moved per direction, 4-deep queue
    for (int pass = 0; pass < 2; ++pass) {
        const char* name = pass == 0 ? "D2H (SDMA read)" : "H2D (SDMA write)";
        hipDeviceSynchronize();
        for (int i = 0; i < depth; ++i) {
            if (pass == 0) hipMemcpyAsync(h, d, N, hipMemcpyDeviceToHost, s);
            else hipMemcpyAsync(d, h, N, hipMemcpyHostToDevice, s);
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters - depth; ++i) {
            if (pass == 0) hipMemcpyAsync(h, d, N, hipMemcpyDeviceToHost, s);
            else hipMemcpyAsync(d, h, N, hipMemcpyHostToDevice, s);
        }
        hipStreamSynchronize(s);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double gbs = (double)(iters - depth) * N / sec / 1e9;
        printf("%s: %.1f GB/s (%.1f MiB in %.3f s)\n", name, gbs, (iters - depth) * N / 1048576.0, sec);
    }
    // small-message latency class (the AR sizes): 1.5 MB chunks, 8-deep
    const size_t M = 1536u << 10;
    hipDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 64; ++i) hipMemcpyAsync(h, d, M, hipMemcpyDeviceToHost, s);
    hipStreamSynchronize(s);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("D2H 1.5MiB x64 queued: %.1f us per copy effective, %.1f GB/s\n", sec / 64 * 1e6, 64.0 * M / sec / 1e9);
    return 0;
}
