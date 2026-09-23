// Multi-die RCCL transport probe for the V340L (gfx900) RCCL TRANSPORT DESK.
//
// One rank per die, one OS process per rank: the launcher forks 4 workers and
// each worker sets HIP_VISIBLE_DEVICES=<die> via its own environ BEFORE its
// first HIP call (the parent never initializes HIP, so fork+HIP is safe and
// each process sees exactly one device). The communicator is formed with
// ncclGetUniqueId on rank 0, handed to the workers through pre-fork shared
// memory, then ncclGroupStart/ncclCommInitRank(world=4)/ncclGroupEnd.
//
// Per-rank timing (the whole point): every worker histograms BOTH
//   wall - host enqueue->sync time per iteration (includes launch)
//   dev  - HIP event time around the ncclAllReduce on the stream
//          (comparable to the rocprof census kernel durations)
// in two regimes:
//   lockstep - host barrier before every iteration: isolated per-boundary
//              latency (transport + algo, launch skew minimized)
//   burst    - N allreduces enqueued back-to-back, one barrier: the served
//              stream-pipelined regime, per-op = elapsed / N
// Reports per-rank median/min/p90/avg, two-half spread (3% law), and a
// bucketed dev-time histogram per rank.
//
// Served boundary class: ne 20480 fp32 = 80 KB (verify + catch-up);
// ne 5120 = 20 KB (draft steps).
//
// Build: hipcc -O2 -x hip rccl_multidie_probe.cu -o rccl_multidie_probe -lrccl
// Run:   ./rccl_multidie_probe [--dies 0,1,2,3] [--sizes 20480]
//            [--iters 2000] [--warmup 300] [--burst 256] [--timeout 600]

#include <hip/hip_runtime_api.h>
#include <rccl/rccl.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define HIP_CHECK(cmd)                                                              \
    do {                                                                            \
        hipError_t e_ = (cmd);                                                      \
        if (e_ != hipSuccess) {                                                     \
            fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n", (int) e_,           \
                    hipGetErrorString(e_), __FILE__, __LINE__, #cmd);               \
            _exit(2);                                                               \
        }                                                                           \
    } while (0)

#define NCCL_CHECK(cmd)                                                             \
    do {                                                                            \
        ncclResult_t r_ = (cmd);                                                    \
        if (r_ != ncclSuccess) {                                                    \
            fprintf(stderr, "NCCL error %d (%s) at %s:%d: %s\n", (int) r_,          \
                    ncclGetErrorString(r_), __FILE__, __LINE__, #cmd);              \
            _exit(2);                                                               \
        }                                                                           \
    } while (0)

namespace {

constexpr int MAX_RANKS = 4;

struct rank_summary {
    int    iters = 0;
    double wall_med = 0, wall_p90 = 0, wall_min = 0, wall_avg = 0, wall_spread = 0;
    double dev_med = 0,  dev_p90 = 0,  dev_min = 0,  dev_avg = 0,  dev_spread = 0;
    double burst_wall = 0, burst_dev = 0;
    int    burst_n = 0;
};

// Pre-fork anonymous shared mapping: uid exchange + host barrier + summaries.
struct shared_state {
    std::atomic<uint32_t> bar_count{0};
    std::atomic<uint32_t> bar_sense{0};
    std::atomic<int>      uid_ready{0};
    ncclUniqueId          uid = {};
    rank_summary          ranks[MAX_RANKS] = {};
};

void barrier(shared_state * sh) {
    constexpr uint32_t n = MAX_RANKS;
    const uint32_t sense = sh->bar_sense.load();
    if (sh->bar_count.fetch_add(1) == n - 1) {
        sh->bar_count.store(0);
        sh->bar_sense.store(1 - sense);
    } else {
        while (sh->bar_sense.load() == sense) {
            sched_yield();
        }
    }
}

double now_us() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::micro>>(steady_clock::now().time_since_epoch()).count();
}

struct stats {
    double med = 0, p90 = 0, min = 0, avg = 0, spread = 0;
};

stats summarize(std::vector<double> v) {
    stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.med = v[v.size() / 2];
    s.min = v.front();
    s.p90 = v[(size_t)(v.size() * 0.90)];
    double acc = 0;
    for (double x : v) acc += x;
    s.avg = acc / v.size();
    const size_t h = v.size() / 2;
    double a1 = 0, a2 = 0;
    for (size_t i = 0; i < h; i++) a1 += v[i];
    for (size_t i = h; i < v.size(); i++) a2 += v[i];
    a1 /= h; a2 /= (v.size() - h);
    s.spread = 100.0 * fabs(a1 - a2) / std::min(a1, a2);
    return s;
}

int worker(int rank, int iters, int warmup, int burst_n, int ne, shared_state * sh) {
    HIP_CHECK(hipInit(0));
    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    if (ndev != 1) {
        fprintf(stderr, "rank %d: expected 1 visible device, got %d\n", rank, ndev);
        _exit(2);
    }
    HIP_CHECK(hipSetDevice(0));

    char pci[32] = {};
    HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    float * buf = nullptr;
    HIP_CHECK(hipMalloc(&buf, (size_t) ne * sizeof(float)));
    std::vector<float> h(ne);
    for (int i = 0; i < ne; i++) h[i] = 0.5f + 0.25f * rank;
    HIP_CHECK(hipMemcpy(buf, h.data(), (size_t) ne * sizeof(float), hipMemcpyHostToDevice));

    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));

    // rank 0 publishes the unique id through the pre-fork mapping
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&sh->uid));
        sh->uid_ready.store(1);
    }
    barrier(sh);
    while (!sh->uid_ready.load()) sched_yield();

    ncclComm_t comm = nullptr;
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclCommInitRank(&comm, MAX_RANKS, sh->uid, rank));
    NCCL_CHECK(ncclGroupEnd());

    // one correctness pass: every rank must see the exact sum pattern
    NCCL_CHECK(ncclAllReduce(buf, buf, ne, ncclFloat, ncclSum, comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(h.data(), buf, (size_t) ne * sizeof(float), hipMemcpyDeviceToHost));
    float expect = 0.0f;
    for (int r = 0; r < MAX_RANKS; r++) expect += 0.5f + 0.25f * r;
    for (int i = 0; i < ne; i++) {
        if (h[i] != expect) {
            fprintf(stderr, "rank %d: CORRECTNESS FAIL at %d: %f != %f\n", rank, i, h[i], expect);
            _exit(2);
        }
    }

    // warmup
    for (int k = 0; k < warmup; k++) {
        NCCL_CHECK(ncclAllReduce(buf, buf, ne, ncclFloat, ncclSum, comm, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // lockstep: host barrier, then one timed allreduce per iteration
    std::vector<double> twall, tdev;
    twall.reserve(iters); tdev.reserve(iters);
    for (int k = 0; k < iters; k++) {
        barrier(sh);
        HIP_CHECK(hipEventRecord(ev0, stream));
        const double t0 = now_us();
        NCCL_CHECK(ncclAllReduce(buf, buf, ne, ncclFloat, ncclSum, comm, stream));
        HIP_CHECK(hipEventRecord(ev1, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const double t1 = now_us();
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        twall.push_back(t1 - t0);
        tdev.push_back(ms * 1000.0);
    }

    const stats w = summarize(twall), d = summarize(tdev);
    printf("RANK %d ne=%d kb=%.0f lockstep iters=%d | wall med %.2f min %.2f p90 %.2f avg %.2f spread %.2f%%"
           " | dev med %.2f min %.2f p90 %.2f avg %.2f spread %.2f%% | pci %s\n",
           rank, ne, ne * 4.0 / 1024.0, iters,
           w.med, w.min, w.p90, w.avg, w.spread,
           d.med, d.min, d.p90, d.avg, d.spread, pci);

    // bucketed dev histogram: 2 us buckets over [min, p99.5]
    {
        std::vector<double> srt = tdev;
        std::sort(srt.begin(), srt.end());
        const double hi = srt[(size_t)(srt.size() * 0.995)];
        const double lo = d.min;
        const int nb = (int) ((hi - lo) / 2.0) + 1;
        printf("RANK %d dev histogram (us, 2us buckets):", rank);
        for (int b = 0; b < nb; b++) {
            int c = 0;
            for (double x : tdev) {
                if (x >= lo + 2.0 * b && (b == nb - 1 || x < lo + 2.0 * (b + 1))) c++;
            }
            if (c) printf(" %.0f-%.0f:%d", lo + 2.0 * b, lo + 2.0 * (b + 1), c);
        }
        printf("\n");
    }
    fflush(stdout);

    // burst: one barrier, N back-to-back allreduces (served pipelined regime)
    if (burst_n > 0) {
        for (int k = 0; k < 50; k++) {
            NCCL_CHECK(ncclAllReduce(buf, buf, ne, ncclFloat, ncclSum, comm, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        barrier(sh);
        HIP_CHECK(hipEventRecord(ev0, stream));
        const double t0 = now_us();
        for (int k = 0; k < burst_n; k++) {
            NCCL_CHECK(ncclAllReduce(buf, buf, ne, ncclFloat, ncclSum, comm, stream));
        }
        HIP_CHECK(hipEventRecord(ev1, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const double t1 = now_us();
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        rank_summary & rs = sh->ranks[rank];
        rs.burst_wall = (t1 - t0) / burst_n;
        rs.burst_dev  = ms * 1000.0 / burst_n;
        rs.burst_n    = burst_n;
        printf("RANK %d ne=%d burst n=%d per-op wall %.2f dev %.2f us\n",
               rank, ne, burst_n, rs.burst_wall, rs.burst_dev);
        fflush(stdout);
    }

    rank_summary & rs = sh->ranks[rank];
    rs.iters = iters;
    rs.wall_med = w.med; rs.wall_p90 = w.p90; rs.wall_min = w.min; rs.wall_avg = w.avg; rs.wall_spread = w.spread;
    rs.dev_med = d.med;  rs.dev_p90 = d.p90;  rs.dev_min = d.min;  rs.dev_avg = d.avg;  rs.dev_spread = d.spread;

    barrier(sh);
    ncclCommDestroy(comm);
    HIP_CHECK(hipFree(buf));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}

void on_alrm(int) {
    fprintf(stderr, "probe timeout\n");
    _exit(3);
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<int> dies = {0, 1, 2, 3};
    int iters = 2000, warmup = 300, burst_n = 256, timeout = 600;
    int ne = 20480;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return argv[++i]; };
        if      (a == "--dies")   { dies.clear(); char * s = strdup(next()); for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) dies.push_back(atoi(t)); free(s); }
        else if (a == "--iters")  { iters = atoi(next()); }
        else if (a == "--warmup") { warmup = atoi(next()); }
        else if (a == "--burst")  { burst_n = atoi(next()); }
        else if (a == "--ne")     { ne = atoi(next()); }
        else if (a == "--timeout"){ timeout = atoi(next()); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if ((int) dies.size() != MAX_RANKS) {
        fprintf(stderr, "need exactly %d dies, got %zu\n", MAX_RANKS, dies.size());
        return 1;
    }

    signal(SIGALRM, on_alrm);
    alarm(timeout);

    shared_state * sh = (shared_state *) mmap(nullptr, sizeof(shared_state),
                                              PROT_READ | PROT_WRITE,
                                              MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (sh == MAP_FAILED) { perror("mmap"); return 1; }
    new (sh) shared_state{};

    printf("rccl_multidie_probe: world %d, dies", MAX_RANKS);
    for (int d : dies) printf(" %d", d);
    printf(", ne %d (%.0f KB fp32), iters %d, burst %d, pid %d\n",
           ne, ne * 4.0 / 1024.0, iters, burst_n, getpid());

    std::vector<pid_t> kids(MAX_RANKS);
    for (int r = 0; r < MAX_RANKS; r++) {
        fflush(nullptr);
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 1; }
        if (pid == 0) {
            char die[16];
            snprintf(die, sizeof(die), "%d", dies[r]);
            if (setenv("HIP_VISIBLE_DEVICES", die, 1) != 0) _exit(2);
            _exit(worker(r, iters, warmup, burst_n, ne, sh));
        }
        kids[r] = pid;
    }

    int fail = 0;
    for (pid_t k : kids) {
        int st = 0;
        if (waitpid(k, &st, 0) < 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0) fail = 1;
    }
    if (fail) { fprintf(stderr, "a worker failed\n"); return 2; }

    // cross-rank summary table
    printf("\nSUMMARY ne=%d lockstep per-rank dev med (us):", ne);
    double wmax = 0, wmin = 1e30, dmax = 0, dmin = 1e30;
    for (int r = 0; r < MAX_RANKS; r++) {
        const rank_summary & rs = sh->ranks[r];
        printf(" die%d=%.1f", dies[r], rs.dev_med);
        wmax = std::max(wmax, rs.wall_med); wmin = std::min(wmin, rs.wall_med);
        dmax = std::max(dmax, rs.dev_med);  dmin = std::min(dmin, rs.dev_med);
    }
    printf(" | dev spread x%.2f wall spread x%.2f\n", dmax / dmin, wmax / wmin);
    double worst_half = 0;
    for (int r = 0; r < MAX_RANKS; r++) worst_half = std::max(worst_half, sh->ranks[r].dev_spread);
    printf("SPREAD-LAW worst halves spread: %.2f%% (%s)\n", worst_half, worst_half <= 3.0 ? "PASS" : "FAIL");
    printf("SUMMARY ne=%d burst per-op dev med (us):", ne);
    for (int r = 0; r < MAX_RANKS; r++) printf(" die%d=%.2f", dies[r], sh->ranks[r].burst_dev);
    printf("\n");
    printf("DONE\n");
    return 0;
}
