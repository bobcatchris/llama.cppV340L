// TP4 boundary micro-probe for the V340L boundary desk (P0).
//
// Isolates the per-op RCCL allreduce cost from cross-die transport: n ranks
// on ONE die (HIP_VISIBLE_DEVICES=<die> makes it the only visible device),
// fp32 in-place allreduce at the served decode boundary sizes. The served
// boundary tensor is the hidden state (n_embd 5120): T=4 verify/catch-up =
// ne 20480 (80 KB fp32), T=1 draft = ne 5120 (20 KB).
//
// Arms, all faithful to the ggml integration shapes:
//   SYNC    one boundary per iteration: ncclGroupStart, one ncclAllReduce per
//           rank, ncclGroupEnd, stream sync (exactly
//           ggml_backend_cuda_comm_allreduce_nccl per call, no host sync inside)
//   PIPE16  16 boundaries back-to-back on the streams, one sync at the end
//           (prices enqueue/sync amortization = the clustering ceiling)
//   GROUP16 one ncclGroup containing 16 x per-rank ncclAllReduce, one sync
//           (what an env-gated GGML_CUDA_RCCL_CLUSTER=1 would enqueue)
//
// Verdict cells report two half-averages; the desk law requires their spread
// <= 3% for a cell to count.
//
// Build: hipcc -O2 -x hip rccl_tp4_boundary_probe.cpp -o rccl_tp4_boundary_probe -lrccl
// Run:   HIP_VISIBLE_DEVICES=3 ./rccl_tp4_boundary_probe [--ranks 4] [--iters 2000]
//            [--sizes 4096,8192,16384,20480,24576,32768] [--nsweep 4,3,2,1] [--pipe 16]

#include <hip/hip_runtime_api.h>
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

struct rank_res {
    hipStream_t stream = nullptr;
    float     * buf    = nullptr;
};

static double now_us() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::micro>>(steady_clock::now().time_since_epoch()).count();
}

static void one_boundary(std::vector<ncclComm_t> & comms, std::vector<rank_res> & r, int ne) {
    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < r.size(); i++) {
        NCCL_CHECK(ncclAllReduce(r[i].buf, r[i].buf, ne, ncclFloat, ncclSum, comms[i], r[i].stream));
    }
    NCCL_CHECK(ncclGroupEnd());
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

int main(int argc, char ** argv) {
    int    nrank   = 4;
    int    iters   = 2000;
    int    pipe_k  = 16;
    int    timeout = 240;
    std::vector<int> sizes  = {4096, 8192, 16384, 20480, 24576, 32768}; // 16..128 KB fp32
    std::vector<int> nsweep = {4};

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return argv[++i]; };
        if      (a == "--ranks")  { nrank = atoi(next()); }
        else if (a == "--iters")  { iters = atoi(next()); }
        else if (a == "--pipe")   { pipe_k = atoi(next()); }
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
    for (int n : nsweep) {
        if (n > ndev) { fprintf(stderr, "nsweep %d > visible devices %d\n", n, ndev); return 1; }
    }

    const size_t max_ne = *std::max_element(sizes.begin(), sizes.end());

    for (int n : nsweep) {
        printf("\n==== n=%d ranks on one die ====\n", n);
        std::vector<rank_res> r(n);
        for (int i = 0; i < n; i++) {
            HIP_CHECK(hipSetDevice(0));
            HIP_CHECK(hipStreamCreate(&r[i].stream));
            HIP_CHECK(hipMalloc(&r[i].buf, max_ne * sizeof(float)));
            std::vector<float> h(max_ne, 0.5f + 0.25f * i);
            HIP_CHECK(hipMemcpy(r[i].buf, h.data(), max_ne * sizeof(float), hipMemcpyHostToDevice));
        }
        std::vector<int> devlist(n, 0); // all ranks on the single visible device

        std::vector<ncclComm_t> comms(n);
        const double t0 = now_us();
        ncclResult_t irc = ncclCommInitAll(comms.data(), n, devlist.data());
        const double init_ms = (now_us() - t0) / 1000.0;
        if (irc != ncclSuccess) {
            // some RCCL builds refuse duplicate GPUs within one communicator;
            // report and move on with the rest of the n-sweep
            printf("INIT: FAILED rc=%d (%s) after %.0f ms (n=%d on one device)\n",
                   (int) irc, ncclGetErrorString(irc), init_ms, n);
            for (int i = 0; i < n; i++) {
                HIP_CHECK(hipFree(r[i].buf));
                HIP_CHECK(hipStreamDestroy(r[i].stream));
            }
            continue;
        }
        printf("INIT: OK in %.0f ms (n=%d, one device)\n", init_ms, n);

        for (int ne : sizes) {
            const double kb = ne * sizeof(float) / 1024.0;
            for (int i = 0; i < n; i++) {
                HIP_CHECK(hipSetDevice(0));
                std::vector<float> h(ne, 0.5f + 0.25f * i);
                HIP_CHECK(hipMemcpy(r[i].buf, h.data(), ne * sizeof(float), hipMemcpyHostToDevice));
            }

            // ---- SYNC arm ----
            const int warm = std::min(400, std::max(50, iters / 4));
            for (int k = 0; k < warm; k++) { one_boundary(comms, r, ne); }
            sync_all(r);
            std::vector<double> t; t.reserve(iters);
            for (int k = 0; k < iters; k++) {
                double s = now_us(); one_boundary(comms, r, ne); sync_all(r); t.push_back(now_us() - s);
            }
            std::vector<double> h1(t.begin(), t.begin() + t.size() / 2);
            std::vector<double> h2(t.begin() + t.size() / 2, t.end());
            bench_stat st = summarize(t), s1 = summarize(h1), s2 = summarize(h2);
            const double spread = 100.0 * fabs(s1.avg - s2.avg) / std::min(s1.avg, s2.avg);
            printf("SYNC    n=%d %6.0f KB x%4d: avg %7.1f us  med %7.1f  min %7.1f  halves %7.1f/%7.1f spread %.2f%%%s\n",
                   n, kb, iters, st.avg, st.med, st.mn, s1.avg, s2.avg, spread,
                   spread <= 3.0 ? "" : "  [SPREAD-FAIL]");
            fflush(stdout);

            // ---- PIPE16 arm (k boundaries, one sync) ----
            if (pipe_k > 1) {
                for (int k = 0; k < warm / 2; k++) { for (int q = 0; q < pipe_k; q++) { one_boundary(comms, r, ne); } }
                sync_all(r);
                t.clear();
                const int reps = std::max(20, iters / pipe_k);
                for (int k = 0; k < reps; k++) {
                    double s = now_us();
                    for (int q = 0; q < pipe_k; q++) { one_boundary(comms, r, ne); }
                    sync_all(r);
                    t.push_back((now_us() - s) / pipe_k);
                }
                std::vector<double> p1(t.begin(), t.begin() + t.size() / 2);
                std::vector<double> p2(t.begin() + t.size() / 2, t.end());
                st = summarize(t); s1 = summarize(p1); s2 = summarize(p2);
                const double spr2 = 100.0 * fabs(s1.avg - s2.avg) / std::min(s1.avg, s2.avg);
                printf("PIPE%-3d n=%d %6.0f KB x%4d: avg %7.1f us/op  med %7.1f  halves %7.1f/%7.1f spread %.2f%%%s\n",
                       pipe_k, n, kb, (int) t.size(), st.avg, st.med, s1.avg, s2.avg, spr2,
                       spr2 <= 3.0 ? "" : "  [SPREAD-FAIL]");
                fflush(stdout);

                // ---- GROUP16 arm (one ncclGroup with k boundaries) ----
                for (int k = 0; k < warm / 2; k++) {
                    NCCL_CHECK(ncclGroupStart());
                    for (int q = 0; q < pipe_k; q++) {
                        for (int i = 0; i < n; i++) {
                            NCCL_CHECK(ncclAllReduce(r[i].buf, r[i].buf, ne, ncclFloat, ncclSum, comms[i], r[i].stream));
                        }
                    }
                    NCCL_CHECK(ncclGroupEnd());
                }
                sync_all(r);
                t.clear();
                for (int k = 0; k < reps; k++) {
                    double s = now_us();
                    NCCL_CHECK(ncclGroupStart());
                    for (int q = 0; q < pipe_k; q++) {
                        for (int i = 0; i < n; i++) {
                            NCCL_CHECK(ncclAllReduce(r[i].buf, r[i].buf, ne, ncclFloat, ncclSum, comms[i], r[i].stream));
                        }
                    }
                    NCCL_CHECK(ncclGroupEnd());
                    sync_all(r);
                    t.push_back((now_us() - s) / pipe_k);
                }
                std::vector<double> g1(t.begin(), t.begin() + t.size() / 2);
                std::vector<double> g2(t.begin() + t.size() / 2, t.end());
                st = summarize(t); s1 = summarize(g1); s2 = summarize(g2);
                const double spr3 = 100.0 * fabs(s1.avg - s2.avg) / std::min(s1.avg, s2.avg);
                printf("GROUP%-2d n=%d %6.0f KB x%4d: avg %7.1f us/op  med %7.1f  halves %7.1f/%7.1f spread %.2f%%%s\n",
                       pipe_k, n, kb, (int) t.size(), st.avg, st.med, s1.avg, s2.avg, spr3,
                       spr3 <= 3.0 ? "" : "  [SPREAD-FAIL]");
                fflush(stdout);
            }
        }

        for (int i = 0; i < n; i++) {
            ncclCommDestroy(comms[i]);
            HIP_CHECK(hipFree(r[i].buf));
            HIP_CHECK(hipStreamDestroy(r[i].stream));
        }
    }

    printf("\nDONE\n");
    return 0;
}
