// dp_p2p_probe.cu - DP-2 data-path probe (minimal, fault-tolerant, unbuffered).
// Steps per ordered pair: canAccessPeer -> EnablePeerAccess -> hipMemcpyPeerAsync
// roundtrip (runtime-translated path; NO UVA raw-pointer assumption - the W31
// probe's kstore step faults by design on ROCm) -> verify -> lockstep latency
// at 4KB/20KB/80KB/1MB. Errors are REPORTED, never HIP_CHECK-aborted, so one
// fault cannot kill the matrix. Receipt printer for DP-2 STEP A/D.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

#define SZ (1 << 20)  // 1 MiB buffers

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static const char * estr(hipError_t e) { return hipGetErrorString(e); }

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int nd = 0;
    hipError_t e = hipGetDeviceCount(&nd);
    printf("dp_p2p_probe: devices=%d (%s)\n", nd, estr(e));
    if (e != hipSuccess || nd < 2) return 1;

    // buffers + pattern per die
    float * buf[8] = {}, * pat[8] = {};
    hipStream_t st[8] = {};
    std::vector<float> host(SZ / 4);
    for (int k = 0; k < (int) host.size(); k++) host[k] = 0.25f * k + 0.5f;
    for (int d = 0; d < nd; d++) {
        hipSetDevice(d);
        e = hipMalloc(&buf[d], SZ);
        printf("die %d hipMalloc: %s (%p)\n", d, estr(e), (void *) buf[d]);
        if (e != hipSuccess) return 2;
        hipMalloc(&pat[d], SZ);
        hipMemcpy(pat[d], host.data(), SZ, hipMemcpyHostToDevice);
        hipMemset(buf[d], 0, SZ);
        hipStreamCreate(&st[d]);
    }

    int rc_final = 0;
    for (int i = 0; i < nd; i++) {
        for (int j = 0; j < nd; j++) {
            if (i == j) continue;
            printf("---- pair %d->%d\n", i, j);
            hipSetDevice(i);
            int can = -1;
            e = hipDeviceCanAccessPeer(&can, i, j);
            printf("  canAccessPeer=%d (%s)\n", can, estr(e));
            if (can != 1) { rc_final |= 4; continue; }
            e = hipDeviceEnablePeerAccess(j, 0);
            if (e == hipErrorPeerAccessAlreadyEnabled) e = hipSuccess;
            printf("  enablePeerAccess=%s\n", estr(e));
            if (e != hipSuccess) { rc_final |= 8; continue; }

            // single-shot copy + verify (runtime-translated peer path)
            e = hipMemcpyPeerAsync(buf[j], j, pat[i], i, 80 * 1024, st[i]);
            printf("  memcpyPeerAsync(80KB): %s\n", estr(e));
            if (e != hipSuccess) { rc_final |= 16; continue; }
            e = hipStreamSynchronize(st[i]);
            printf("  sync: %s\n", estr(e));
            if (e != hipSuccess) { rc_final |= 32; continue; }
            std::vector<float> chk(80 * 1024 / 4);
            hipSetDevice(j);
            e = hipMemcpy(chk.data(), buf[j], 80 * 1024, hipMemcpyDeviceToHost);
            bool ok = (e == hipSuccess) && memcmp(chk.data(), host.data(), 80 * 1024) == 0;
            printf("  VERIFY 80KB: %s%s\n", ok ? "PASS" : "FAIL",
                   ok ? "" : estr(e));
            if (!ok) { rc_final |= 64; continue; }

            // lockstep latency: one copy + sync per iteration, median of 201
            const size_t sizes[] = {4 * 1024, 20 * 1024, 80 * 1024, SZ};
            hipSetDevice(i);
            for (size_t s : sizes) {
                std::vector<double> t;
                for (int it = 0; it < 220; it++) {
                    double t0 = now_us();
                    hipError_t e1 = hipMemcpyPeerAsync(buf[j], j, pat[i], i, s, st[i]);
                    hipError_t e2 = hipStreamSynchronize(st[i]);
                    double t1 = now_us();
                    if (it >= 10) t.push_back(t1 - t0);  // warmup skipped
                    if (e1 != hipSuccess || e2 != hipSuccess) {
                        printf("  %zuKB: FAIL it=%d %s/%s\n", s / 1024, it, estr(e1), estr(e2));
                        rc_final |= 128;
                        break;
                    }
                }
                if (t.size() < 100) continue;
                std::sort(t.begin(), t.end());
                double med = t[t.size() / 2];
                printf("  %4zuKB lockstep: med %7.2f us  min %7.2f  eff %6.2f GB/s\n",
                       s / 1024, med, t.front(), s / med / 1e3);
            }
        }
    }

    for (int d = 0; d < nd; d++) {
        hipSetDevice(d);
        if (buf[d]) hipFree(buf[d]);
        if (pat[d]) hipFree(pat[d]);
        if (st[d]) hipStreamDestroy(st[d]);
    }
    printf("dp_p2p_probe done rc=%d\n", rc_final);
    return rc_final;
}
