// RCCL transport probe for the TP3 V340L verify-transport desk.
//
// Answers:
//   1. does RCCL init succeed across 3 (or 4) gfx900 ranks with no P2P
//      (canAccessPeer=0 all pairs)?
//   2. which transport does RCCL pick (run with RCCL_DEBUG=INFO to see the
//      per-channel "via SHM/P2P/NET" lines)?
//   3. grouped ncclAllReduce wall latency at the real boundary size (32 KB
//      fp32) and at 128 KB, vs a faithful emulation of the current meta
//      backend butterfly (peer copies through host staging + device ADDs)?
//   4. is the RCCL fp32 sum bit-exact vs the butterfly's host ADD order
//      ((r0 + r2) + r1 for n=3), across N random seeds?
//
// The collectives are driven from a single host thread with
// ncclGroupStart / n x ncclAllReduce / ncclGroupEnd on per-rank streams -
// the same shape as ggml_backend_cuda_comm_allreduce_nccl in ggml-cuda.cu.
//
// Build: hipcc -O2 -x hip rccl_probe.cpp -o rccl_probe -lrccl
// Run:   ./rccl_probe [--ranks 3] [--devs 0,1,2] [--iters 2000]
//                    [--seeds 64] [--sizes 8192,32768]

#include <hip/hip_runtime_api.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <random>
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
    int          dev    = -1;
    hipStream_t  stream = nullptr;
    hipEvent_t   ev     = nullptr; // device-scoped, like ggml's per-ctx copy_event
    float      * buf    = nullptr; // in-place allreduce / accumulator
    float      * tmp    = nullptr; // staging for butterfly emulation
    float      * hstage = nullptr; // pinned host stage (butterfly arm)
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

// one butterfly boundary for n ranks, exactly the sequence the meta backend
// runs for n=3: fold src2 into rank0, exchange (0,1), copyback 0->2.
// cross-rank copies use hipMemcpyPeerAsync like ggml_backend_cpy_tensor_async
// (GGML_CUDA_NO_PEER_COPY off => the driver stages through host when peers
// cannot access each other).
static void butterfly_boundary(std::vector<rank_res> & r, int n, int ne) {
    auto copy_peer = [&](int src, int dst, float * dst_ptr) {
        HIP_CHECK(hipMemcpyPeerAsync(dst_ptr, r[dst].dev, r[src].buf, r[src].dev,
                                     ne * sizeof(float), r[src].stream));
        HIP_CHECK(hipEventRecord(r[src].ev, r[src].stream));
        HIP_CHECK(hipStreamWaitEvent(r[dst].stream, r[src].ev, 0));
    };
    auto add = [&](int dst) {
        const int blocks = (ne + 255) / 256;
        add_kernel<<<blocks, 256, 0, r[dst].stream>>>(r[dst].buf, r[dst].tmp, ne);
    };

    if (n == 3) {
        // fold: rank2 -> rank0, add on rank0
        copy_peer(2, 0, r[0].tmp);
        add(0);
        // butterfly stage: 0<->1 exchange, add on both
        copy_peer(0, 1, r[1].tmp);
        copy_peer(1, 0, r[0].tmp);
        add(1);
        add(0);
        // copyback: rank0 -> rank2
        copy_peer(0, 2, r[2].buf);
    } else if (n == 4) {
        // fold: none. butterfly stage offset 2: 0<->2, 1<->3
        copy_peer(0, 2, r[2].tmp); add(2);
        copy_peer(2, 0, r[0].tmp); add(0);
        copy_peer(1, 3, r[3].tmp); add(3);
        copy_peer(3, 1, r[1].tmp); add(1);
        // butterfly stage offset 1: 0<->1, 2<->3
        copy_peer(0, 1, r[1].tmp); add(1);
        copy_peer(1, 0, r[0].tmp); add(0);
        copy_peer(2, 3, r[3].tmp); add(3);
        copy_peer(3, 2, r[2].tmp); add(2);
    } else {
        fprintf(stderr, "butterfly emulation supports 3 or 4 ranks\n");
        exit(1);
    }
}

static void rccl_boundary(std::vector<ncclComm_t> & comms, std::vector<rank_res> & r, int ne) {
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

static uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

struct bench_stat { double avg, mn, mx; };

static bench_stat summarize(const std::vector<double> & v) {
    bench_stat s{0.0, 1e30, 0.0};
    for (double x : v) {
        s.avg += x;
        s.mn = std::min(s.mn, x);
        s.mx = std::max(s.mx, x);
    }
    s.avg /= (double) v.size();
    return s;
}

int main(int argc, char ** argv) {
    int    nrank     = 3;
    int    iters     = 2000;
    int    seeds     = 64;
    int    timeout_s = 110;
    std::vector<int> devs = {0, 1, 2};
    std::vector<int> sizes = {8192, 32768}; // in fp32 elements (32 KB, 128 KB)

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return argv[++i]; };
        if      (a == "--ranks")  { nrank = atoi(next()); devs.resize(nrank); for (int j = 0; j < nrank; j++) devs[j] = j; }
        else if (a == "--devs")   { devs.clear(); char * s = strdup(next()); for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) devs.push_back(atoi(t)); free(s); nrank = (int) devs.size(); }
        else if (a == "--iters")  { iters = atoi(next()); }
        else if (a == "--seeds")  { seeds = atoi(next()); }
        else if (a == "--sizes")  { sizes.clear(); char * s = strdup(next()); for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) sizes.push_back(atoi(t)); free(s); }
        else if (a == "--timeout"){ timeout_s = atoi(next()); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    // hard cap on wall time so the probe can never overstay the GPU lock
    signal(SIGALRM, [](int) { fprintf(stderr, "probe timeout\n"); _exit(3); });
    alarm(timeout_s);

    printf("rccl_probe: ranks=%d devs=", nrank);
    for (int d : devs) printf("%d,", d);
    printf(" iters=%d seeds=%d\n", iters, seeds);

    int ndev = 0;
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipGetDeviceCount(&ndev));
    int ver = 0;
    HIP_CHECK(hipDriverGetVersion(&ver));
    printf("hip: ndev=%d driver=%d\n", ndev, ver);

    for (int d : devs) {
        if (d >= ndev) { fprintf(stderr, "device %d out of range\n", d); return 1; }
    }

    // P2P matrix as RCCL will see it
    for (int a : devs) {
        for (int b : devs) {
            if (a == b) continue;
            int can = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can, a, b));
            printf("canAccessPeer(%d->%d) = %d\n", a, b, can);
        }
    }

    // resources
    const size_t max_ne = *std::max_element(sizes.begin(), sizes.end());
    std::vector<rank_res> r(nrank);
    for (int i = 0; i < nrank; i++) {
        r[i].dev = devs[i];
        HIP_CHECK(hipSetDevice(r[i].dev));
        HIP_CHECK(hipStreamCreate(&r[i].stream));
        HIP_CHECK(hipEventCreateWithFlags(&r[i].ev, hipEventDisableTiming));
        HIP_CHECK(hipMalloc(&r[i].buf, max_ne * sizeof(float)));
        HIP_CHECK(hipMalloc(&r[i].tmp, max_ne * sizeof(float)));
        HIP_CHECK(hipHostMalloc(&r[i].hstage, max_ne * sizeof(float)));
    }

    // init communicators - same call as ggml_backend_cuda_comm_init_nccl
    int rccl_ver = 0;
    NCCL_CHECK(ncclGetVersion(&rccl_ver));
    printf("rccl version: %d\n", rccl_ver);

    std::vector<ncclComm_t> comms(nrank);
    std::vector<int> devlist(devs);
    const double t0 = now_us();
    ncclResult_t irc = ncclCommInitAll(comms.data(), nrank, devlist.data());
    const double init_ms = (now_us() - t0) / 1000.0;
    if (irc != ncclSuccess) {
        printf("INIT: FAILED rc=%d (%s) after %.0f ms\n", (int) irc,
               ncclGetErrorString(irc), init_ms);
        return 1;
    }
    printf("INIT: OK in %.0f ms (run with RCCL_DEBUG=INFO to see transport)\n", init_ms);

    // ---- latency arms ----
    for (int ne : sizes) {
        const double kb = ne * sizeof(float) / 1024.0;

        // warmup + fill
        for (int i = 0; i < nrank; i++) {
            HIP_CHECK(hipSetDevice(r[i].dev));
            std::vector<float> h(ne, 0.5f + 0.25f * i);
            HIP_CHECK(hipMemcpy(r[i].buf, h.data(), ne * sizeof(float), hipMemcpyHostToDevice));
        }
        for (int k = 0; k < 200; k++) {
            rccl_boundary(comms, r, ne);
        }
        sync_all(r);

        std::vector<double> ts;
        ts.reserve(iters);
        for (int k = 0; k < iters; k++) {
            const double s = now_us();
            rccl_boundary(comms, r, ne);
            sync_all(r);
            ts.push_back(now_us() - s);
        }
        bench_stat st = summarize(ts);
        printf("RCCL     %7.0f KB x%d: avg %.1f us  min %.1f us  max %.1f us\n",
               kb, iters, st.avg, st.mn, st.mx);

        // butterfly emulation, same boot
        std::vector<double> tb;
        tb.reserve(iters);
        for (int k = 0; k < iters; k++) {
            const double s = now_us();
            butterfly_boundary(r, nrank, ne);
            sync_all(r);
            tb.push_back(now_us() - s);
        }
        bench_stat sb = summarize(tb);
        printf("BUTTERFLY %6.0f KB x%d: avg %.1f us  min %.1f us  max %.1f us\n",
               kb, iters, sb.avg, sb.mn, sb.mx);
        printf("ratio butterfly/rccl: %.2fx  (delta %.1f us/boundary)\n",
               sb.avg / st.avg, sb.avg - st.avg);
    }

    // ---- bit-exactness: RCCL vs butterfly ADD order ----
    // butterfly for n=3 computes ((r0 + r2) + r1) on every rank
    // (rank2 receives a copy of rank0's result). host reference in float.
    printf("bit-exactness vs butterfly ADD order ((r0+r2)+r1):\n");
    for (int ne : sizes) {
        long seeds_all_match = 0, elems_mismatch = 0, elems_total = 0;
        uint32_t max_ulp = 0;
        long cross_rank_identical = 0;

        std::vector<float> h0(ne), h1(ne), h2(ne), ref(ne);
        std::vector<float> out(nrank * ne);

        for (int sd = 0; sd < seeds; sd++) {
            std::mt19937 rng(sd * 2654435761u + 12345u);
            std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
            for (int i = 0; i < ne; i++) {
                h0[i] = uni(rng);
                h1[i] = uni(rng);
                h2[i] = uni(rng);
                ref[i] = (h0[i] + h2[i]) + h1[i];
            }

            for (int i = 0; i < nrank; i++) {
                const float * src = (i == 0) ? h0.data() : (i == 1) ? h1.data() : h2.data();
                HIP_CHECK(hipSetDevice(r[i].dev));
                HIP_CHECK(hipMemcpy(r[i].buf, src, ne * sizeof(float), hipMemcpyHostToDevice));
            }
            rccl_boundary(comms, r, ne);
            sync_all(r);
            for (int i = 0; i < nrank; i++) {
                HIP_CHECK(hipMemcpy(out.data() + i * ne, r[i].buf, ne * sizeof(float), hipMemcpyDeviceToHost));
            }

            bool all = true;
            for (int i = 0; i < nrank; i++) {
                if (memcmp(out.data() + i * ne, ref.data(), ne * sizeof(float)) != 0) {
                    all = false;
                    for (int j = 0; j < ne; j++) {
                        if (i == 0) {
                            elems_total++;
                            uint32_t a = f2u(out[j]), b = f2u(ref[j]);
                            uint32_t ulp = (a > b) ? a - b : b - a;
                            max_ulp = std::max(max_ulp, ulp);
                            if (a != b) elems_mismatch++;
                        }
                    }
                }
            }
            if (all) seeds_all_match++;
            if (memcmp(out.data(), out.data() + ne, ne * sizeof(float)) == 0 &&
                memcmp(out.data(), out.data() + 2 * ne, ne * sizeof(float)) == 0) {
                cross_rank_identical++;
            }
        }
        const double pct = elems_total ? 100.0 * elems_mismatch / elems_total : 0.0;
        printf("  ne=%d: seeds %ld/%ld bit-exact on ALL ranks; rank0 mismatched elems %ld/%ld (%.3f%%) max ULP %u; cross-rank identical %ld/%ld\n",
               ne, seeds_all_match, (long) seeds, elems_mismatch, elems_total, pct, max_ulp,
               cross_rank_identical, (long) seeds);
    }

    for (ncclComm_t c : comms) {
        ncclCommDestroy(c);
    }
    printf("DONE\n");
    return 0;
}
