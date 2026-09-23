// Multi-die RCCL transport probe for the V340L (gfx900) RCCL TRANSPORT DESK.
//
// Two process regimes, both covering the served boundary class:
//   multi-process (launched by run_rccl_multidie.sh) - one OS process per
//       rank, each exec'd CLEANLY from the shell with HIP_VISIBLE_DEVICES=
//       <die>. Communicator: ncclGetUniqueId on rank 0 -> POSIX shm ->
//       ncclCommInitRank(world=4) per worker; a shm host barrier paces the
//       lockstep loop; rank 0 prints the cross-rank summary.
//       NOTE: workers must be launched from a parent that has NOT loaded
//       the HIP runtime - fork/exec from a HIP-linked launcher breaks 3 of
//       4 workers inside RCCL device init on this stack (ROCm 6.2 finding).
//   inproc --inproc - ONE process, all devices visible, ncclCommInitAll
//       over 4 devices and grouped ncclGroupStart + 4x ncclAllReduce +
//       ncclGroupEnd - byte-for-byte the served regime
//       (ggml/src/ggml-cuda/ggml-cuda.cu ncclCommInitAll + grouped call).
//
// Per-rank timing (the whole point): every rank histograms BOTH
//   wall - host time per iteration (enqueue -> sync; includes launch)
//   dev  - HIP event time around that rank's ncclAllReduce on its stream
//          (comparable to the rocprof census kernel durations)
// in two pacing regimes:
//   lockstep - multi-process: shm barrier before every iteration (isolated
//              per-boundary latency, launch skew minimized);
//              inproc mode: one grouped call per iteration
//   burst    - N allreduces enqueued back-to-back after one barrier: the
//              served stream-pipelined regime, per-op = elapsed / N
// Reports per-rank median/min/p90/avg, two-half + even/odd spread (3% law),
// a bucketed dev-time histogram per rank, and the cross-rank spread table.
//
// Served boundary classes: ne 20480 fp32 = 80 KB (verify + catch-up);
// ne 5120 = 20 KB (draft steps).
//
// Build: hipcc -O2 -x hip rccl_multidie_probe.cu -o rccl_multidie_probe -lrccl
// Run:   docs/amd-port/tests/run_rccl_multidie.sh <label> [args...]

#include <hip/hip_runtime_api.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
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

// cross-process shared state (POSIX shm in fork mode, anonymous mmap there too)
struct shared_state {
    std::atomic<uint32_t> bar_count{0};
    std::atomic<uint32_t> bar_sense{0};
    std::atomic<int>      uid_ready{0};
    ncclUniqueId          uid = {};
    rank_summary          ranks[MAX_RANKS] = {};
};

void barrier(shared_state * sh) {
    const uint32_t sense = sh->bar_sense.load();
    if (sh->bar_count.fetch_add(1) == MAX_RANKS - 1) {
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
    double med = 0, p90 = 0, min = 0, avg = 0, spread = 0, eo_spread = 0;
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
    // even/odd halves: immune to monotonic drift (clock ramp etc)
    double e = 0, o = 0;
    for (size_t i = 0; i < v.size(); i += 2) e += v[i];
    for (size_t i = 1; i < v.size(); i += 2) o += v[i];
    e /= (v.size() + 1) / 2; o /= v.size() / 2;
    s.eo_spread = 100.0 * fabs(e - o) / std::min(e, o);
    return s;
}

void print_rank(int die, int ne, int iters, const stats & w, const stats & d, const char * pci) {
    printf("RANK die%d ne=%d kb=%.0f lockstep iters=%d | wall med %.2f min %.2f p90 %.2f avg %.2f halves %.2f%% eo %.2f%%"
           " | dev med %.2f min %.2f p90 %.2f avg %.2f halves %.2f%% eo %.2f%% | pci %s\n",
           die, ne, ne * 4.0 / 1024.0, iters,
           w.med, w.min, w.p90, w.avg, w.spread, w.eo_spread,
           d.med, d.min, d.p90, d.avg, d.spread, d.eo_spread, pci);
}

void print_histogram(int die, const std::vector<double> & tdev, const stats & d) {
    std::vector<double> srt = tdev;
    std::sort(srt.begin(), srt.end());
    const double hi = srt[(size_t)(srt.size() * 0.995)];
    const double lo = d.min;
    const int nb = (int) ((hi - lo) / 2.0) + 1;
    printf("RANK die%d dev histogram (us, 2us buckets):", die);
    for (int b = 0; b < nb; b++) {
        int c = 0;
        for (double x : tdev) {
            if (x >= lo + 2.0 * b && (b == nb - 1 || x < lo + 2.0 * (b + 1))) c++;
        }
        if (c) printf(" %.0f-%.0f:%d", lo + 2.0 * b, lo + 2.0 * (b + 1), c);
    }
    printf("\n");
}

// one lockstep iteration: (wall_us, dev_us) for one comm/stream
std::pair<double, double> timed_op(float * buf, int ne, ncclComm_t comm, hipStream_t stream,
                                   hipEvent_t ev0, hipEvent_t ev1) {
    HIP_CHECK(hipEventRecord(ev0, stream));
    const double t0 = now_us();
    NCCL_CHECK(ncclAllReduce(buf, buf, ne, ncclFloat, ncclSum, comm, stream));
    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const double t1 = now_us();
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    return { t1 - t0, ms * 1000.0 };
}

struct res {
    hipStream_t stream = nullptr;
    float * buf = nullptr;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
};

// caller has already selected the device
void setup_rank(int ne, res & r) {
    HIP_CHECK(hipStreamCreate(&r.stream));
    HIP_CHECK(hipMalloc(&r.buf, (size_t) ne * sizeof(float)));
    HIP_CHECK(hipEventCreate(&r.ev0));
    HIP_CHECK(hipEventCreate(&r.ev1));
}

void fill_pattern(float * buf, int ne, int rank) {
    std::vector<float> h(ne);
    for (int i = 0; i < ne; i++) h[i] = 0.5f + 0.25f * rank;
    HIP_CHECK(hipMemcpy(buf, h.data(), (size_t) ne * sizeof(float), hipMemcpyHostToDevice));
}

// grouped iteration over all ranks (inproc lockstep, the served shape):
// shared wall + per-rank dev
std::pair<double, std::vector<double>> timed_group(std::vector<float *> & bufs, int ne,
                                                   std::vector<ncclComm_t> & comms,
                                                   std::vector<res> & rr) {
    const int n = (int) bufs.size();
    for (int r = 0; r < n; r++) HIP_CHECK(hipEventRecord(rr[r].ev0, rr[r].stream));
    const double t0 = now_us();
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < n; r++) {
        NCCL_CHECK(ncclAllReduce(bufs[r], bufs[r], ne, ncclFloat, ncclSum, comms[r], rr[r].stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    for (int r = 0; r < n; r++) HIP_CHECK(hipEventRecord(rr[r].ev1, rr[r].stream));
    for (int r = 0; r < n; r++) HIP_CHECK(hipStreamSynchronize(rr[r].stream));
    const double t1 = now_us();
    std::vector<double> dev(n);
    for (int r = 0; r < n; r++) {
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, rr[r].ev0, rr[r].ev1));
        dev[r] = ms * 1000.0;
    }
    return { t1 - t0, dev };
}

void grouped_call(std::vector<float *> & bufs, int ne, std::vector<ncclComm_t> & comms,
                  std::vector<res> & rr) {
    const int n = (int) bufs.size();
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < n; r++) {
        NCCL_CHECK(ncclAllReduce(bufs[r], bufs[r], ne, ncclFloat, ncclSum, comms[r], rr[r].stream));
    }
    NCCL_CHECK(ncclGroupEnd());
}

// ---- fork mode worker body: one rank, one process ---------------------------
int worker_body(int rank, int die, int iters, int warmup, int burst_n, int ne,
                int init_delay_s, shared_state * sh) {
    // optional staggered start: serializes multi-process RCCL init (the
    // concurrent-init race on this stack fails 3 of 4 ranks)
    if (init_delay_s > 0) sleep(rank * init_delay_s);
    HIP_CHECK(hipInit(0));
    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    if (ndev != 1) {
        fprintf(stderr, "rank %d: expected 1 visible device, got %d\n", rank, ndev);
        _exit(2);
    }
    HIP_CHECK(hipSetDevice(0));
    res r;
    setup_rank(ne, r);
    fill_pattern(r.buf, ne, rank);
    printf("W%d hip-ok\n", rank); fflush(stdout);

    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&sh->uid));
        sh->uid_ready.store(1, std::memory_order_release);
    }
    barrier(sh);
    while (!sh->uid_ready.load()) sched_yield();

    ncclComm_t comm = nullptr;
    printf("W%d init-start\n", rank); fflush(stdout);
    NCCL_CHECK(ncclCommInitRank(&comm, MAX_RANKS, sh->uid, rank));
    printf("W%d init-done\n", rank); fflush(stdout);

    // correctness: every rank must see the exact sum pattern
    {
        std::vector<float> h(ne);
        NCCL_CHECK(ncclAllReduce(r.buf, r.buf, ne, ncclFloat, ncclSum, comm, r.stream));
        HIP_CHECK(hipStreamSynchronize(r.stream));
        HIP_CHECK(hipMemcpy(h.data(), r.buf, (size_t) ne * sizeof(float), hipMemcpyDeviceToHost));
        float expect = 0.0f;
        for (int q = 0; q < MAX_RANKS; q++) expect += 0.5f + 0.25f * q;
        for (int i = 0; i < ne; i++) {
            if (h[i] != expect) {
                fprintf(stderr, "rank %d: CORRECTNESS FAIL at %d: %f != %f\n", rank, i, h[i], expect);
                _exit(2);
            }
        }
    }

    for (int k = 0; k < warmup; k++) {
        NCCL_CHECK(ncclAllReduce(r.buf, r.buf, ne, ncclFloat, ncclSum, comm, r.stream));
    }
    HIP_CHECK(hipStreamSynchronize(r.stream));

    std::vector<double> twall, tdev;
    twall.reserve(iters); tdev.reserve(iters);
    for (int k = 0; k < iters; k++) {
        barrier(sh);
        const auto t = timed_op(r.buf, ne, comm, r.stream, r.ev0, r.ev1);
        twall.push_back(t.first);
        tdev.push_back(t.second);
    }

    char pci[32] = {};
    HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));
    const stats w = summarize(twall), d = summarize(tdev);
    print_rank(die, ne, iters, w, d, pci);
    print_histogram(die, tdev, d);
    fflush(stdout);

    rank_summary & rs = sh->ranks[rank];
    rs.iters = iters;
    rs.wall_med = w.med; rs.wall_p90 = w.p90; rs.wall_min = w.min; rs.wall_avg = w.avg; rs.wall_spread = w.eo_spread;
    rs.dev_med = d.med;  rs.dev_p90 = d.p90;  rs.dev_min = d.min;  rs.dev_avg = d.avg;  rs.dev_spread = d.eo_spread;

    if (burst_n > 0) {
        for (int k = 0; k < 50; k++) {
            NCCL_CHECK(ncclAllReduce(r.buf, r.buf, ne, ncclFloat, ncclSum, comm, r.stream));
        }
        HIP_CHECK(hipStreamSynchronize(r.stream));
        barrier(sh);
        HIP_CHECK(hipEventRecord(r.ev0, r.stream));
        const double t0 = now_us();
        for (int k = 0; k < burst_n; k++) {
            NCCL_CHECK(ncclAllReduce(r.buf, r.buf, ne, ncclFloat, ncclSum, comm, r.stream));
        }
        HIP_CHECK(hipEventRecord(r.ev1, r.stream));
        HIP_CHECK(hipStreamSynchronize(r.stream));
        const double t1 = now_us();
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, r.ev0, r.ev1));
        rs.burst_wall = (t1 - t0) / burst_n;
        rs.burst_dev  = ms * 1000.0 / burst_n;
        rs.burst_n    = burst_n;
        printf("RANK die%d ne=%d burst n=%d per-op wall %.2f dev %.2f us\n",
               die, ne, burst_n, rs.burst_wall, rs.burst_dev);
        fflush(stdout);
    }

    barrier(sh);
    ncclCommDestroy(comm);
    HIP_CHECK(hipFree(r.buf));
    return 0;
}

void print_summary(shared_state * sh, int ne) {
    printf("\nSUMMARY ne=%d lockstep per-rank dev med (us):", ne);
    double wmax = 0, wmin = 1e30, dmax = 0, dmin = 1e30;
    for (int r = 0; r < MAX_RANKS; r++) {
        const rank_summary & s = sh->ranks[r];
        printf(" die%d=%.1f", r, s.dev_med);
        wmax = std::max(wmax, s.wall_med); wmin = std::min(wmin, s.wall_med);
        dmax = std::max(dmax, s.dev_med);  dmin = std::min(dmin, s.dev_med);
    }
    printf(" | dev spread x%.2f wall spread x%.2f\n", dmax / dmin, wmax / wmin);
    double worst_half = 0;
    for (int r = 0; r < MAX_RANKS; r++) worst_half = std::max(worst_half, sh->ranks[r].dev_spread);
    printf("SPREAD-LAW worst halves spread: %.2f%% (%s)\n", worst_half, worst_half <= 3.0 ? "PASS" : "FAIL");
    printf("SUMMARY ne=%d burst per-op dev (us):", ne);
    for (int r = 0; r < MAX_RANKS; r++) printf(" die%d=%.2f", r, sh->ranks[r].burst_dev);
    printf("\n");
    printf("DONE\n");
}

// ---- inproc mode: all ranks, one process (the served regime) ----------------
int inproc(int iters, int warmup, int burst_n, int ne) {
    HIP_CHECK(hipInit(0));
    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    printf("inproc: visible devices = %d\n", ndev);
    if (ndev < MAX_RANKS) { fprintf(stderr, "need %d devices\n", MAX_RANKS); return 2; }

    std::vector<int> dev_ids(MAX_RANKS);
    for (int r = 0; r < MAX_RANKS; r++) dev_ids[r] = r;

    std::vector<res> rr(MAX_RANKS);
    std::vector<float *> bufs(MAX_RANKS);
    for (int rank = 0; rank < MAX_RANKS; rank++) {
        HIP_CHECK(hipSetDevice(dev_ids[rank]));
        setup_rank(ne, rr[rank]);
        bufs[rank] = rr[rank].buf;
        fill_pattern(bufs[rank], ne, rank);
    }

    std::vector<ncclComm_t> comms(MAX_RANKS);
    NCCL_CHECK(ncclCommInitAll(comms.data(), MAX_RANKS, dev_ids.data()));

    // grouped correctness pass
    {
        grouped_call(bufs, ne, comms, rr);
        for (int rank = 0; rank < MAX_RANKS; rank++) HIP_CHECK(hipStreamSynchronize(rr[rank].stream));
        float expect = 0.0f;
        for (int q = 0; q < MAX_RANKS; q++) expect += 0.5f + 0.25f * q;
        for (int rank = 0; rank < MAX_RANKS; rank++) {
            std::vector<float> h(ne);
            HIP_CHECK(hipSetDevice(dev_ids[rank]));
            HIP_CHECK(hipMemcpy(h.data(), bufs[rank], (size_t) ne * sizeof(float), hipMemcpyDeviceToHost));
            for (int i = 0; i < ne; i++) {
                if (h[i] != expect) {
                    fprintf(stderr, "rank %d: CORRECTNESS FAIL at %d\n", rank, i);
                    return 2;
                }
            }
        }
        printf("correctness PASS (grouped, %d ranks)\n", MAX_RANKS);
    }

    for (int k = 0; k < warmup; k++) grouped_call(bufs, ne, comms, rr);
    for (int rank = 0; rank < MAX_RANKS; rank++) HIP_CHECK(hipStreamSynchronize(rr[rank].stream));

    // lockstep (one grouped call per iteration)
    std::vector<std::vector<double>> tdev(MAX_RANKS);
    std::vector<double> twall;
    twall.reserve(iters);
    for (int rank = 0; rank < MAX_RANKS; rank++) tdev[rank].reserve(iters);
    for (int k = 0; k < iters; k++) {
        const auto t = timed_group(bufs, ne, comms, rr);
        twall.push_back(t.first);
        for (int rank = 0; rank < MAX_RANKS; rank++) tdev[rank].push_back(t.second[rank]);
    }

    const stats wall_all = summarize(twall);
    rank_summary sums[MAX_RANKS];
    for (int rank = 0; rank < MAX_RANKS; rank++) {
        char pci[32] = {};
        HIP_CHECK(hipSetDevice(dev_ids[rank]));
        HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), dev_ids[rank]));
        const stats d = summarize(tdev[rank]);
        print_rank(rank, ne, iters, wall_all, d, pci);
        print_histogram(rank, tdev[rank], d);
        rank_summary & s = sums[rank];
        s.dev_med = d.med; s.dev_p90 = d.p90; s.dev_min = d.min; s.dev_avg = d.avg; s.dev_spread = d.eo_spread;
        s.wall_med = wall_all.med; s.wall_spread = wall_all.eo_spread;
        s.iters = iters;
    }

    // burst: grouped calls enqueued back-to-back
    if (burst_n > 0) {
        for (int k = 0; k < 20; k++) grouped_call(bufs, ne, comms, rr);
        for (int rank = 0; rank < MAX_RANKS; rank++) HIP_CHECK(hipStreamSynchronize(rr[rank].stream));
        for (int rank = 0; rank < MAX_RANKS; rank++) HIP_CHECK(hipEventRecord(rr[rank].ev0, rr[rank].stream));
        const double t0 = now_us();
        for (int k = 0; k < burst_n; k++) grouped_call(bufs, ne, comms, rr);
        for (int rank = 0; rank < MAX_RANKS; rank++) HIP_CHECK(hipEventRecord(rr[rank].ev1, rr[rank].stream));
        for (int rank = 0; rank < MAX_RANKS; rank++) HIP_CHECK(hipStreamSynchronize(rr[rank].stream));
        const double t1 = now_us();
        printf("BURST ne=%d n=%d wall per-op %.2f us |", ne, burst_n, (t1 - t0) / burst_n);
        for (int rank = 0; rank < MAX_RANKS; rank++) {
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, rr[rank].ev0, rr[rank].ev1));
            sums[rank].burst_dev = ms * 1000.0 / burst_n;
            sums[rank].burst_n = burst_n;
            printf(" die%d dev %.2f", rank, sums[rank].burst_dev);
        }
        printf("\n");
    }

    printf("\nSUMMARY ne=%d lockstep per-rank dev med (us):", ne);
    double dmax = 0, dmin = 1e30;
    for (int rank = 0; rank < MAX_RANKS; rank++) {
        printf(" die%d=%.1f", rank, sums[rank].dev_med);
        dmax = std::max(dmax, sums[rank].dev_med);
        dmin = std::min(dmin, sums[rank].dev_med);
    }
    printf(" | dev spread x%.2f\n", dmax / dmin);
    double worst_half = 0;
    for (int rank = 0; rank < MAX_RANKS; rank++) worst_half = std::max(worst_half, sums[rank].dev_spread);
    printf("SPREAD-LAW worst halves spread: %.2f%% (%s)\n", worst_half, worst_half <= 3.0 ? "PASS" : "FAIL");
    printf("DONE\n");
    for (int rank = 0; rank < MAX_RANKS; rank++) ncclCommDestroy(comms[rank]);
    return 0;
}


int worker_main(int argc, char ** argv) {
    int rank = 0, die = 0, iters = 2000, warmup = 300, burst_n = 256, ne = 20480;
    int init_delay_s = 0;
    const char * shm_name = nullptr;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return argv[++i]; };
        if      (a == "--rank")       { rank = atoi(next()); }
        else if (a == "--die")        { die = atoi(next()); }
        else if (a == "--shm")        { shm_name = next(); }
        else if (a == "--iters")      { iters = atoi(next()); }
        else if (a == "--warmup")     { warmup = atoi(next()); }
        else if (a == "--burst")      { burst_n = atoi(next()); }
        else if (a == "--ne")         { ne = atoi(next()); }
        else if (a == "--init-delay") { init_delay_s = atoi(next()); }
        else { continue; } // --worker and any pass-through args
    }
    if (!shm_name) { fprintf(stderr, "worker: no shm\n"); return 1; }
    // O_CREAT: the runner script (bash) launches workers directly; zero-filled
    // shm pages are exactly shared_state's initial value, so no init step.
    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
    if (fd < 0) { perror("worker shm_open"); return 1; }
    if (ftruncate(fd, sizeof(shared_state)) != 0) { perror("worker ftruncate"); return 1; }
    auto * sh = (shared_state *) mmap(nullptr, sizeof(shared_state), PROT_READ | PROT_WRITE,
                                      MAP_SHARED, fd, 0);
    close(fd);
    if (sh == MAP_FAILED) { perror("worker mmap"); return 1; }
    const int rc = worker_body(rank, die, iters, warmup, burst_n, ne, init_delay_s, sh);
    if (rc == 0 && rank == 0) {
        print_summary(sh, ne);
        fflush(stdout);
    }
    munmap(sh, sizeof(shared_state));
    return rc;
}

} // namespace

int main(int argc, char ** argv) {
    bool inproc_mode = false, worker_mode = false;
    int iters = 2000, warmup = 300, burst_n = 256;
    int ne = 20480;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--worker") { worker_mode = true; break; }
        auto next = [&]() -> const char * { return argv[++i]; };
        if      (a == "--inproc")       { inproc_mode = true; }
        else if (a == "--iters")        { iters = atoi(next()); }
        else if (a == "--warmup")       { warmup = atoi(next()); }
        else if (a == "--burst")        { burst_n = atoi(next()); }
        else if (a == "--ne")           { ne = atoi(next()); }
        else if (a == "--timeout")      { (void) next(); } // accepted for runner compatibility
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    if (worker_mode) {
        return worker_main(argc, argv);
    }
    if (inproc_mode) {
        return inproc(iters, warmup, burst_n, ne);
    }
    fprintf(stderr,
            "usage: run one worker per die from the runner script:\n"
            "  HIP_VISIBLE_DEVICES=<die> %s --worker --rank R --die D --shm /name\n"
            "     [--iters N] [--warmup N] [--burst N] [--ne ELEM]\n"
            "  or single-process served-regime mode: %s --inproc [same options]\n"
            "  (multi-process launch is done by run_rccl_multidie.sh)\n",
            argv[0], argv[0]);
    return 1;
}
