// TP4 boundary micro-probe for the V340L boundary desk (P0).
//
// Isolates the per-op allreduce cost from cross-die transport, on ONE die
// (HIP_VISIBLE_DEVICES=<die>).
//
// FINDING OF RECORD: RCCL 2.20.5 REFUSES multi-rank-on-one-device
// communicators (ncclCommInitAll -> ncclInvalidUsage, "Duplicate GPU
// detected" class, for every n >= 2). A single-die RCCL collective does
// not exist, so the per-op cost curve is measured as three arms:
//   RCCL-1R  single-rank ncclAllReduce (RCCL elides the kernel entirely)
//            = the API/enqueue floor
//   KLAUNCH  one add kernel over the op's bytes + sync
//            = the single-launch floor at the op's byte class
//   RING1K   ONE cooperative kernel emulating the n-rank ring allreduce
//            in-device (reduce-scatter + all-gather, grid-synced, the
//            RCCL kernel shape, traffic 3(n-1)/n*S element accesses)
//            = the transport-free algorithm floor
// Served model: us(size, n) = launch floor + ring floor + transport(n,
// topology). The transport share is the coordinator's U3 multi-die probe.
// Served boundary tensors (n_embd 5120): T=4 verify/catch-up = ne 20480
// (80 KB fp32), T=1 draft = ne 5120 (20 KB).
//
// Verdict cells report two half-averages; the desk law requires their
// spread <= 3% for a cell to count.
//
// Build: hipcc -O2 -x hip rccl_tp4_boundary_probe.cpp -o rccl_tp4_boundary_probe -lrccl
// Run:   HIP_VISIBLE_DEVICES=3 ./rccl_tp4_boundary_probe [--iters 2000]
//            [--sizes 4096,8192,16384,20480,24576,32768] [--nsweep 4,2]

#include <hip/hip_runtime_api.h>
#include <hip/hip_cooperative_groups.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>

#define HIP_CHECK(cmd)                                                              \
    do {                                                                            \
        hipError_t e_ = (cmd);                                                      \
        if (e_ != hipSuccess) {                                                     \
            fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n", (int) e_,           \
                    hipGetErrorString(e_), __FILE__, __LINE__, #cmd);               \
            exit(2);                                                                \
        }                                                                           \
    } while (0)

#define NCCL_CHECK(cmd)                                                             \
    do {                                                                            \
        ncclResult_t r_ = (cmd);                                                    \
        if (r_ != ncclSuccess) {                                                    \
            fprintf(stderr, "NCCL error %d (%s) at %s:%d: %s\n", (int) r_,          \
                    ncclGetErrorString(r_), __FILE__, __LINE__, #cmd);              \
            exit(2);                                                                \
        }                                                                           \
    } while (0)

namespace cg = cooperative_groups;

struct rank_res {
    hipStream_t stream = nullptr;
    float     * buf    = nullptr;
};

static double now_us() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::micro>>(steady_clock::now().time_since_epoch()).count();
}

__global__ void add_kernel(float * dst, const float * src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] += src[i];
    }
}

// one-block-per-rank ring allreduce, fully in-device, grid-synced:
// reduce-scatter: at step s rank r folds its chunk (r-s)%n with the same
// chunk of rank (r+1)%n; after n-1 steps rank r holds the full sum of
// chunk (r+1)%n. all-gather: n-1 propagation steps. Memory accesses per
// element = 3(n-1)/n (read src + read dst + write dst per fold; copy for
// gather) - the ring's traffic without the transport.
__global__ void ring_kernel(float ** bufs, int ne, int n) {
    cg::grid_group g = cg::this_grid();
    const int rank = blockIdx.x;
    const int chunk = ne / n;
    float * buf = bufs[rank];
    for (int s = 0; s < n - 1; s++) {
        const int c = ((rank - s) % n + n) % n;
        const float * src = bufs[(rank + 1) % n] + (size_t) c * chunk;
        float * dst = buf + (size_t) c * chunk;
        for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
            dst[i] += src[i];
        }
        g.sync();
    }
    const int mine = (rank + 1) % n;
    for (int s = 0; s < n - 1; s++) {
        const int c = ((mine - s) % n + n) % n;
        const float * src = bufs[(rank + 1) % n] + (size_t) c * chunk;
        float * dst = buf + (size_t) c * chunk;
        for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
            dst[i] = src[i];
        }
        g.sync();
    }
}

static void sync_all(std::vector<rank_res> & r) {
    for (auto & rk : r) {
        HIP_CHECK(hipStreamSynchronize(rk.stream));
    }
}

struct bench_stat { double avg, mn, mx, med; };

static bench_stat summarize(std::vector<double> & v) {
    bench_stat s{0.0, 1e30, 0.0, 0.0};
    for (double x : v) {
        s.avg += x;
        s.mn = std::min(s.mn, x);
        s.mx = std::max(s.mx, x);
    }
    s.avg /= (double) v.size();
    std::sort(v.begin(), v.end());
    s.med = v[v.size() / 2];
    return s;
}

static void report(const char * arm, int n, double kb, int iters, std::vector<double> & t) {
    std::vector<double> h1(t.begin(), t.begin() + t.size() / 2);
    std::vector<double> h2(t.begin() + t.size() / 2, t.end());
    bench_stat st = summarize(t), s1 = summarize(h1), s2 = summarize(h2);
    const double spread = 100.0 * fabs(s1.avg - s2.avg) / std::min(s1.avg, s2.avg);
    printf("%-8s n=%d %6.0f KB x%4d: avg %7.2f us  med %7.2f  min %7.2f  halves %7.2f/%7.2f spread %.2f%%%s\n",
           arm, n, kb, iters, st.avg, st.med, st.mn, s1.avg, s2.avg, spread,
           spread <= 3.0 ? "" : "  [SPREAD-FAIL]");
    fflush(stdout);
}

int main(int argc, char ** argv) {
    int    iters   = 2000;
    int    timeout = 300;
    std::vector<int> sizes  = {4096, 8192, 16384, 20480, 24576, 32768}; // 16..128 KB fp32
    std::vector<int> nsweep = {4, 2};

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return argv[++i]; };
        if      (a == "--iters")  { iters = atoi(next()); }
        else if (a == "--sizes")  { sizes.clear(); char * s = strdup(next()); for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) sizes.push_back(atoi(t)); free(s); }
        else if (a == "--nsweep") { nsweep.clear(); char * s = strdup(next()); for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) nsweep.push_back(atoi(t)); free(s); }
        else if (a == "--timeout"){ timeout = atoi(next()); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    signal(SIGALRM, [](int) { fprintf(stderr, "probe timeout\n"); _exit(3); });
    alarm(timeout);

    int ndev = 0;
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipGetDeviceCount(&ndev));
    printf("rccl_tp4_boundary_probe: single-die isolation, visible devices = %d\n", ndev);
    if (ndev < 1) { fprintf(stderr, "no visible devices\n"); return 1; }

    int coop = 0;
    HIP_CHECK(hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, 0));
    printf("cooperative launch support: %d\n", coop);

    const size_t max_ne = *std::max_element(sizes.begin(), sizes.end());

    // ---- RCCL-1R: single-rank ncclAllReduce (enqueue floor) ----
    {
        printf("\n==== RCCL-1R (single-rank ncclAllReduce, kernel elided) ====\n");
        rank_res r;
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipStreamCreate(&r.stream));
        HIP_CHECK(hipMalloc(&r.buf, max_ne * sizeof(float)));
        std::vector<float> h(max_ne, 0.5f);
        HIP_CHECK(hipMemcpy(r.buf, h.data(), max_ne * sizeof(float), hipMemcpyHostToDevice));
        ncclComm_t comm;
        const double t0 = now_us();
        int dev0 = 0;
        ncclResult_t irc = ncclCommInitAll(&comm, 1, &dev0);
        printf("INIT: %s in %.0f ms\n", irc == ncclSuccess ? "OK" : "FAILED", now_us() - t0);
        if (irc == ncclSuccess) {
            for (int ne : sizes) {
                const double kb = ne * sizeof(float) / 1024.0;
                for (int k = 0; k < 200; k++) {
                    NCCL_CHECK(ncclAllReduce(r.buf, r.buf, ne, ncclFloat, ncclSum, comm, r.stream));
                }
                HIP_CHECK(hipStreamSynchronize(r.stream));
                std::vector<double> t; t.reserve(iters);
                for (int k = 0; k < iters; k++) {
                    double s = now_us();
                    NCCL_CHECK(ncclAllReduce(r.buf, r.buf, ne, ncclFloat, ncclSum, comm, r.stream));
                    HIP_CHECK(hipStreamSynchronize(r.stream));
                    t.push_back(now_us() - s);
                }
                report("RCCL-1R", 1, kb, iters, t);
            }
            ncclCommDestroy(comm);
        }
        HIP_CHECK(hipFree(r.buf));
        HIP_CHECK(hipStreamDestroy(r.stream));
    }

    // ---- KLAUNCH + RING1K arms per n ----
    for (int n : nsweep) {
        printf("\n==== n=%d ranks on one die (in-device emulation) ====\n", n);
        std::vector<rank_res> r(n);
        std::vector<float *> bufs(n, nullptr);
        for (int i = 0; i < n; i++) {
            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipStreamCreate(&r[i].stream));
            HIP_CHECK(hipMalloc(&r[i].buf, max_ne * sizeof(float)));
            bufs[i] = r[i].buf;
            std::vector<float> h(max_ne, 0.5f + 0.25f * i);
            HIP_CHECK(hipMemcpy(r[i].buf, h.data(), max_ne * sizeof(float), hipMemcpyHostToDevice));
        }
        float ** dbufs = nullptr;
        HIP_CHECK(hipMalloc(&dbufs, n * sizeof(float *)));
        HIP_CHECK(hipMemcpy(dbufs, bufs.data(), n * sizeof(float *), hipMemcpyHostToDevice));

        for (int ne : sizes) {
            const double kb = ne * sizeof(float) / 1024.0;
            for (int i = 0; i < n; i++) {
                HIP_CHECK(hipSetDevice(0));
                std::vector<float> h(ne, 0.5f + 0.25f * i);
                HIP_CHECK(hipMemcpy(r[i].buf, h.data(), ne * sizeof(float), hipMemcpyHostToDevice));
            }
            const int warm = 300;

            // KLAUNCH: one add kernel over the op's bytes (rank 0's stream)
            for (int k = 0; k < warm; k++) {
                add_kernel<<<(ne + 255) / 256, 256, 0, r[0].stream>>>(r[0].buf, r[0].buf, ne);
            }
            sync_all(r);
            std::vector<double> t; t.reserve(iters);
            for (int k = 0; k < iters; k++) {
                double s = now_us();
                add_kernel<<<(ne + 255) / 256, 256, 0, r[0].stream>>>(r[0].buf, r[0].buf, ne);
                sync_all(r);
                t.push_back(now_us() - s);
            }
            report("KLAUNCH", n, kb, iters, t);

            // RING1K: one cooperative kernel, n blocks, grid-synced ring
            if (!coop) continue;
            int grid = n, block = 256;
            void * args[] = { &dbufs, &ne, &n };
            for (int k = 0; k < warm; k++) {
                HIP_CHECK(hipLaunchCooperativeKernel((void *) ring_kernel, dim3(grid), dim3(block),
                                                     args, 0, r[0].stream));
            }
            sync_all(r);
            t.clear();
            for (int k = 0; k < iters; k++) {
                double s = now_us();
                HIP_CHECK(hipLaunchCooperativeKernel((void *) ring_kernel, dim3(grid), dim3(block),
                                                     args, 0, r[0].stream));
                sync_all(r);
                t.push_back(now_us() - s);
            }
            report("RING1K", n, kb, iters, t);
        }

        HIP_CHECK(hipFree(dbufs));
        for (int i = 0; i < n; i++) {
            HIP_CHECK(hipFree(r[i].buf));
            HIP_CHECK(hipStreamDestroy(r[i].stream));
        }
    }

    printf("\nDONE\n");
    return 0;
}
