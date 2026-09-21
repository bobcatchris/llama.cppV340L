// W7 unbr desk — pageable-vs-pinned H2D stall cell (deliverable 3).
//
// PRE-REGISTERED PASS GATE (written BEFORE any run, 2026-09-18):
//   On a stream pre-queued with ~200 ms of busy kernels, a 512-B hipMemcpyAsync
//   issued from PAGEABLE host memory must block the calling thread for
//   >= 50% of the queued kernel duration (>= 100 ms), while the byte-identical
//   copy issued from PINNED (hipHostMalloc) memory must return in < 1 ms.
//   PASS = both arms on the same device/window; parity: both copies byte-equal.
//   Mechanism under test (W7_unbr_row.txt FINDING 2): pageable staging cannot
//   return until the DMA retires; on an in-order stream queued ~1 chunk deep
//   that costs ~one chunk of host stall per copy. Pinned memory returns after
//   enqueue. This prices the cure (pin/D2D the per-chunk ids copies).
//
// Build (standalone hipcc cell — no cmake, per disk law):
//   /opt/rocm/lib/llvm/bin/clang++ -O3 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
//     -I src/common/hip_shim -std=gnu++20 --offload-arch=gfx900 -x hip \
//     w7_h2d_stall_cell.cu -o w7_h2d_stall_cell
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP ERR %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(1);} } while(0)

__global__ void spin_kernel(unsigned long long cycles, unsigned* out) {
    unsigned long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
    if (threadIdx.x == 0 && blockIdx.x == 0) { out[0] = (unsigned)(clock64() & 0xffffffffu); }
}

static double ms_now(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
    CK(hipSetDevice(0));
    hipStream_t s; CK(hipStreamCreate(&s));
    unsigned* sink; CK(hipMalloc(&sink, 4));

    // Calibrate spin kernel to ~2 ms per launch (measure once).
    spin_kernel<<<1, 64, 0, s>>>(1000000ull, sink);   // warmup
    CK(hipStreamSynchronize(s));
    const unsigned long long cycles_per_ms = [&] {
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        CK(hipEventRecord(e0));
        spin_kernel<<<1, 64, 0, 0>>>(1000000ull, sink);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
        float ms = 0.f; CK(hipEventElapsedTime(&ms, e0, e1));
        return (unsigned long long)(1000000ull / (ms > 0.01f ? ms : 0.01f));
    }();
    const unsigned long long cycles = cycles_per_ms * 2;  // ~2 ms per kernel

    // Device buffers for the copies.
    unsigned char *dst_a = nullptr, *dst_b = nullptr;
    CK(hipMalloc(&dst_a, 512)); CK(hipMalloc(&dst_b, 512));

    // Source 1: PAGEABLE (plain new — the std::vector class in the server).
    std::vector<unsigned char> src_pageable(512, 0xAB);
    // Source 2: PINNED.
    unsigned char* src_pinned = nullptr;
    CK(hipHostMalloc(&src_pinned, 512, hipHostMallocDefault));
    std::memset(src_pinned, 0xAB, 512);

    // Queue ~200 ms of kernels (the "one chunk deep" stream shape), then issue
    // the pageable copy and time the HOST call.
    for (int i = 0; i < 100; ++i) { spin_kernel<<<1, 64, 0, s>>>(cycles, sink); }
    auto t0 = std::chrono::steady_clock::now();
    CK(hipMemcpyAsync(dst_a, src_pageable.data(), 512, hipMemcpyHostToDevice, s));
    auto t1 = std::chrono::steady_clock::now();
    double pageable_ms = ms_now(t0, t1);

    // Drain, then repeat IDENTICALLY with pinned source.
    CK(hipStreamSynchronize(s));
    for (int i = 0; i < 100; ++i) { spin_kernel<<<1, 64, 0, s>>>(cycles, sink); }
    t0 = std::chrono::steady_clock::now();
    CK(hipMemcpyAsync(dst_b, src_pinned, 512, hipMemcpyHostToDevice, s));
    t1 = std::chrono::steady_clock::now();
    double pinned_ms = ms_now(t0, t1);
    CK(hipStreamSynchronize(s));

    // Parity: both destinations byte-equal.
    std::vector<unsigned char> ha(512), hb(512);
    CK(hipMemcpy(ha.data(), dst_a, 512, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hb.data(), dst_b, 512, hipMemcpyDeviceToHost));
    int parity = std::memcmp(ha.data(), hb.data(), 512) == 0;

    std::printf("[W7-H2D-CELL] queued=100x~2ms pageable_host_block=%.2f ms pinned_host_block=%.2f ms parity=%s\n",
                pageable_ms, pinned_ms, parity ? "OK" : "FAIL");
    bool pass = (pageable_ms >= 100.0) && (pinned_ms < 1.0) && parity;
    std::printf("[W7-H2D-CELL] GATE(pageable>=100ms && pinned<1ms && parity): %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 2;
}
