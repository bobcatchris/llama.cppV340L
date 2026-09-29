// RCCL prefill-size transport probe for the TP3 V340L rccl-ext desk.
//
// Extends the verify-transport desk probe (rccl_probe.cpp, E-085: 32/128 KB)
// to the PREFILL boundary sizes. The served boundary tensor is hidden-sized
// (Qwen3.8-27B-ASCII-P1M: n_embd 5120), so:
//   decode verify (T=4):   ne =     20480 ( 80 KB fp32) -> today: RCCL fp32
//   prefill ubatch 512:    ne = 2621440 (10 MiB fp32) -> today (nccl mode):
//                              bf16-compress allreduce; unset: butterfly.
//
// Arms, all faithful to the ggml shapes:
//   RCCL-F32   grouped ncclAllReduce fp32 in-place (the small-tensor path,
//              proposed for extension)
//   RCCL-BF16  f32->bf16 convert + grouped bf16 allreduce + bf16->f32
//              (exactly ggml_backend_cuda_comm_allreduce_nccl above the
//              131072 threshold - today's nccl-mode prefill behavior)
//   BUTTERFLY  hipMemcpyPeerAsync (driver-staged through host) + device ADDs
//              in the meta allreduce_fallback order (today's unset path)
//   BUTT-PIN   same butterfly with pinned host staging (the
//              GGML_PINNED_DEV_COPY shape - the butterfly at its best)
//
// Numerics at representative sizes, vs the butterfly fp32 grouping
// ((r0 + r2) + r1) on a host reference:
//   - RCCL-F32: mismatch fraction + max ULP (the signed-off sum-order dust
//     class, expected bounded) and cross-rank identity (must be 64/64 class)
//   - RCCL-BF16: max ULP + max relative error vs the fp32 grouping (this is
//     the LARGER class that serving runs at prefill today)
//
// Build: hipcc -O2 -x hip rccl_prefill_probe.cpp -o rccl_prefill_probe -lrccl
// Run:   ./rccl_prefill_probe [--ranks 3] [--iters 2000] [--seeds 64]

#include <hip/hip_runtime_api.h>
#include <hip/hip_bfloat16.h>
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

using bf16 = hip_bfloat16;

struct rank_res {
    int         dev    = -1;
    hipStream_t stream = nullptr;
    hipEvent_t  ev     = nullptr;
    float     * buf    = nullptr; // in-place allreduce / accumulator
    float     * tmp    = nullptr; // butterfly staging on device
    bf16      * btmp   = nullptr; // RCCL-BF16 compressed buffer
    float     * hstage = nullptr; // pinned host stage (BUTT-PIN arm)
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

__global__ void cvt_f32_bf16_kernel(const float * src, bf16 * dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = bf16(src[i]);
    }
}

__global__ void cvt_bf16_f32_kernel(const bf16 * src, float * dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = float(src[i]);
    }
}

static void launch_add(hipStream_t s, float * dst, const float * src, int ne) {
    const int blocks = (ne + 255) / 256;
    add_kernel<<<blocks, 256, 0, s>>>(dst, src, ne);
}

// meta allreduce_fallback order for n=3: fold 2->0, exchange 0<->1, copyback 0->2.
static void butterfly_boundary(std::vector<rank_res> & r, int n, int ne, bool pinned) {
    auto copy_peer = [&](int src, int dst, float * dst_ptr) {
        if (pinned) {
            // ggml_backend_tensor_copy_async with pinned staging: D2H into the
            // pinned cache, then H2D from it (get/set sequence, bytes identical)
            HIP_CHECK(hipMemcpyAsync(r[src].hstage, r[src].buf, ne * sizeof(float),
                                     hipMemcpyDeviceToHost, r[src].stream));
            HIP_CHECK(hipEventRecord(r[src].ev, r[src].stream));
            HIP_CHECK(hipStreamWaitEvent(r[dst].stream, r[src].ev, 0));
            HIP_CHECK(hipMemcpyAsync(dst_ptr, r[src].hstage, ne * sizeof(float),
                                     hipMemcpyHostToDevice, r[dst].stream));
        } else {
            HIP_CHECK(hipMemcpyPeerAsync(dst_ptr, r[dst].dev, r[src].buf, r[src].dev,
                                         ne * sizeof(float), r[src].stream));
            HIP_CHECK(hipEventRecord(r[src].ev, r[src].stream));
            HIP_CHECK(hipStreamWaitEvent(r[dst].stream, r[src].ev, 0));
        }
    };
    auto add = [&](int dst) {
        launch_add(r[dst].stream, r[dst].buf, r[dst].tmp, ne);
    };

    copy_peer(2, 0, r[0].tmp);
    add(0);
    copy_peer(0, 1, r[1].tmp);
    copy_peer(1, 0, r[0].tmp);
    add(1);
    add(0);
    copy_peer(0, 2, r[2].buf);
}

static void rccl_f32_boundary(std::vector<ncclComm_t> & comms, std::vector<rank_res> & r, int ne) {
    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < r.size(); i++) {
        NCCL_CHECK(ncclAllReduce(r[i].buf, r[i].buf, ne, ncclFloat, ncclSum, comms[i], r[i].stream));
    }
    NCCL_CHECK(ncclGroupEnd());
}

static void rccl_bf16_boundary(std::vector<ncclComm_t> & comms, std::vector<rank_res> & r, int ne,
                               bool compute[3]) {
    for (size_t i = 0; i < r.size(); i++) {
        if (compute[i]) {
            const int blocks = (ne + 255) / 256;
            cvt_f32_bf16_kernel<<<blocks, 256, 0, r[i].stream>>>(r[i].buf, r[i].btmp, ne);
        } else {
            HIP_CHECK(hipMemsetAsync(r[i].btmp, 0, ne * sizeof(bf16), r[i].stream));
        }
    }
    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < r.size(); i++) {
        NCCL_CHECK(ncclAllReduce(r[i].btmp, r[i].btmp, ne, ncclBfloat16, ncclSum, comms[i], r[i].stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    for (size_t i = 0; i < r.size(); i++) {
        const int blocks = (ne + 255) / 256;
        cvt_bf16_f32_kernel<<<blocks, 256, 0, r[i].stream>>>(r[i].btmp, r[i].buf, ne);
    }
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

static int iters_for(int ne, int iters_cap) {
    if (ne <= 65536)   return iters_cap;
    if (ne <= 262144)  return std::min(iters_cap, 800);
    if (ne <= 524288)  return std::min(iters_cap, 400);
    if (ne <= 1048576) return std::min(iters_cap, 150);
    return std::min(iters_cap, 60);
}

int main(int argc, char ** argv) {
    int    nrank     = 3;
    int    iters_cap = 2000;
    int    seeds     = 64;
    int    seeds_big = 16;
    int    timeout_s = 100;
    std::vector<int> devs = {0, 1, 2};
    // decode-verify continuity + the fp32 threshold + the prefill ladder up to
    // the real served size 512 x 5120
    std::vector<int> sizes = {20480, 131072, 262144, 524288, 1048576, 2097152, 2621440};
    std::vector<int> num_sizes = {20480, 262144, 2621440};

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return argv[++i]; };
        if      (a == "--ranks")  { nrank = atoi(next()); devs.resize(nrank); for (int j = 0; j < nrank; j++) devs[j] = j; }
        else if (a == "--iters")  { iters_cap = atoi(next()); }
        else if (a == "--seeds")  { seeds = atoi(next()); }
        else if (a == "--seedsbig") { seeds_big = atoi(next()); }
        else if (a == "--sizes")  { sizes.clear(); char * s = strdup(next()); for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) sizes.push_back(atoi(t)); free(s); }
        else if (a == "--numsizes") { num_sizes.clear(); char * s = strdup(next()); for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) num_sizes.push_back(atoi(t)); free(s); }
        else if (a == "--timeout"){ timeout_s = atoi(next()); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    signal(SIGALRM, [](int) { fprintf(stderr, "probe timeout\n"); _exit(3); });
    alarm(timeout_s);

    printf("rccl_prefill_probe: ranks=%d devs=", nrank);
    for (int d : devs) printf("%d,", d);
    printf(" iters_cap=%d seeds=%d/%d\n", iters_cap, seeds, seeds_big);

    int ndev = 0;
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipGetDeviceCount(&ndev));
    for (int d : devs) {
        if (d >= ndev) { fprintf(stderr, "device %d out of range\n", d); return 1; }
    }

    const size_t max_ne = *std::max_element(sizes.begin(), sizes.end());
    std::vector<rank_res> r(nrank);
    for (int i = 0; i < nrank; i++) {
        r[i].dev = devs[i];
        HIP_CHECK(hipSetDevice(r[i].dev));
        HIP_CHECK(hipStreamCreate(&r[i].stream));
        HIP_CHECK(hipEventCreateWithFlags(&r[i].ev, hipEventDisableTiming));
        HIP_CHECK(hipMalloc(&r[i].buf,  max_ne * sizeof(float)));
        HIP_CHECK(hipMalloc(&r[i].tmp,  max_ne * sizeof(float)));
        HIP_CHECK(hipMalloc(&r[i].btmp, max_ne * sizeof(bf16)));
        HIP_CHECK(hipHostMalloc(&r[i].hstage, max_ne * sizeof(float)));
    }

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
    printf("INIT: OK in %.0f ms\n", init_ms);

    // ---- latency arms (all four on the same boot) ----
    for (int ne : sizes) {
        const int    iters = iters_for(ne, iters_cap);
        const int    warm  = std::min(200, std::max(10, iters / 4));
        const double kb    = ne * sizeof(float) / 1024.0;
        bool compute[3] = {true, true, true};

        for (int i = 0; i < nrank; i++) {
            HIP_CHECK(hipSetDevice(r[i].dev));
            std::vector<float> h(ne, 0.5f + 0.25f * i);
            HIP_CHECK(hipMemcpy(r[i].buf, h.data(), ne * sizeof(float), hipMemcpyHostToDevice));
        }

        // RCCL-F32
        for (int k = 0; k < warm; k++) rccl_f32_boundary(comms, r, ne);
        sync_all(r);
        std::vector<double> t; t.reserve(iters);
        for (int k = 0; k < iters; k++) {
            double s = now_us(); rccl_f32_boundary(comms, r, ne); sync_all(r); t.push_back(now_us() - s);
        }
        bench_stat st = summarize(t);
        printf("RCCL-F32  %8.0f KB it=%4d: avg %9.1f us  min %9.1f us  (%.2f GB/s wire)\n",
               kb, iters, st.avg, st.mn, 2.0 * (nrank - 1) / nrank * ne * 4.0 / (st.avg * 1e3));

        // RCCL-BF16 (today's nccl-mode path above the threshold)
        for (int k = 0; k < warm; k++) rccl_bf16_boundary(comms, r, ne, compute);
        sync_all(r);
        t.clear();
        for (int k = 0; k < iters; k++) {
            double s = now_us(); rccl_bf16_boundary(comms, r, ne, compute); sync_all(r); t.push_back(now_us() - s);
        }
        st = summarize(t);
        printf("RCCL-BF16 %8.0f KB it=%4d: avg %9.1f us  min %9.1f us  (%.2f GB/s wire)\n",
               kb, iters, st.avg, st.mn, (nrank - 1) / nrank * ne * 2.0 / (st.avg * 1e3));

        // BUTTERFLY (hipMemcpyPeerAsync, the env-unset primitive)
        for (int k = 0; k < warm; k++) butterfly_boundary(r, nrank, ne, false);
        sync_all(r);
        t.clear();
        for (int k = 0; k < iters; k++) {
            double s = now_us(); butterfly_boundary(r, nrank, ne, false); sync_all(r); t.push_back(now_us() - s);
        }
        st = summarize(t);
        printf("BUTTERFLY %8.0f KB it=%4d: avg %9.1f us  min %9.1f us\n", kb, iters, st.avg, st.mn);

        // BUTT-PIN (pinned-staging butterfly = GGML_PINNED_DEV_COPY shape)
        for (int k = 0; k < warm; k++) butterfly_boundary(r, nrank, ne, true);
        sync_all(r);
        t.clear();
        for (int k = 0; k < iters; k++) {
            double s = now_us(); butterfly_boundary(r, nrank, ne, true); sync_all(r); t.push_back(now_us() - s);
        }
        st = summarize(t);
        printf("BUTT-PIN  %8.0f KB it=%4d: avg %9.1f us  min %9.1f us\n", kb, iters, st.avg, st.mn);
        fflush(stdout);
    }

    // ---- numerics vs the butterfly fp32 grouping ((r0+r2)+r1) ----
    printf("numerics vs butterfly fp32 grouping ((r0+r2)+r1):\n");
    for (int ne : num_sizes) {
        const int nsd = ne > 524288 ? seeds_big : seeds;
        long rccl_seeds_match = 0, rccl_mm = 0, rccl_tot = 0;
        long bf16_seeds_match = 0, bf16_mm = 0, bf16_tot = 0;
        uint32_t rccl_max_ulp = 0;
        uint32_t bf16_max_ulp = 0;
        double   bf16_max_rel = 0.0;
        long rccl_xrank = 0, bf16_xrank = 0;

        std::vector<float> h0(ne), h1(ne), h2(ne), ref(ne);
        std::vector<float> out(3 * (size_t) ne);

        for (int sd = 0; sd < nsd; sd++) {
            std::mt19937 rng(sd * 2654435761u + 12345u);
            std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
            for (int i = 0; i < ne; i++) {
                h0[i] = uni(rng);
                h1[i] = uni(rng);
                h2[i] = uni(rng);
                ref[i] = (h0[i] + h2[i]) + h1[i];
            }
            const float * src[3] = { h0.data(), h1.data(), h2.data() };

            // RCCL-F32 arm
            for (int i = 0; i < nrank; i++) {
                HIP_CHECK(hipSetDevice(r[i].dev));
                HIP_CHECK(hipMemcpy(r[i].buf, src[i], ne * sizeof(float), hipMemcpyHostToDevice));
            }
            rccl_f32_boundary(comms, r, ne);
            sync_all(r);
            for (int i = 0; i < nrank; i++) {
                HIP_CHECK(hipMemcpy(out.data() + (size_t) i * ne, r[i].buf, ne * sizeof(float), hipMemcpyDeviceToHost));
            }
            if (memcmp(out.data(), out.data() + (size_t) ne, (size_t) ne * 4) == 0 &&
                memcmp(out.data(), out.data() + 2 * (size_t) ne, (size_t) ne * 4) == 0) {
                rccl_xrank++;
            }
            if (memcmp(out.data(), ref.data(), (size_t) ne * 4) != 0) {
                for (int j = 0; j < ne; j++) {
                    rccl_tot++;
                    uint32_t a = f2u(out[j]), b = f2u(ref[j]);
                    if (a != b) { rccl_mm++; rccl_max_ulp = std::max(rccl_max_ulp, a > b ? a - b : b - a); }
                }
            } else {
                rccl_seeds_match++;
            }

            // RCCL-BF16 arm (today's nccl prefill path)
            for (int i = 0; i < nrank; i++) {
                HIP_CHECK(hipSetDevice(r[i].dev));
                HIP_CHECK(hipMemcpy(r[i].buf, src[i], ne * sizeof(float), hipMemcpyHostToDevice));
            }
            bool compute[3] = {true, true, true};
            rccl_bf16_boundary(comms, r, ne, compute);
            sync_all(r);
            for (int i = 0; i < nrank; i++) {
                HIP_CHECK(hipMemcpy(out.data() + (size_t) i * ne, r[i].buf, ne * sizeof(float), hipMemcpyDeviceToHost));
            }
            if (memcmp(out.data(), out.data() + (size_t) ne, (size_t) ne * 4) == 0 &&
                memcmp(out.data(), out.data() + 2 * (size_t) ne, (size_t) ne * 4) == 0) {
                bf16_xrank++;
            }
            if (memcmp(out.data(), ref.data(), (size_t) ne * 4) != 0) {
                for (int j = 0; j < ne; j++) {
                    bf16_tot++;
                    uint32_t a = f2u(out[j]), b = f2u(ref[j]);
                    if (a != b) {
                        bf16_mm++;
                        bf16_max_ulp = std::max(bf16_max_ulp, a > b ? a - b : b - a);
                        const double rel = fabs((out[j] - ref[j]) / (ref[j] != 0.0f ? ref[j] : 1.0f));
                        bf16_max_rel = std::max(bf16_max_rel, rel);
                    }
                }
            } else {
                bf16_seeds_match++;
            }
        }
        const double rp = rccl_tot ? 100.0 * rccl_mm / rccl_tot : 0.0;
        const double bp = bf16_tot ? 100.0 * bf16_mm / bf16_tot : 0.0;
        printf("  ne=%d seeds=%d:\n", ne, nsd);
        printf("    RCCL-F32 : seeds bit-exact %ld/%ld; rank0 mm %ld/%ld (%.3f%%) max ULP %u; cross-rank identical %ld/%ld\n",
               rccl_seeds_match, (long) nsd, rccl_mm, rccl_tot, rp, rccl_max_ulp, rccl_xrank, (long) nsd);
        printf("    RCCL-BF16: seeds bit-exact %ld/%ld; rank0 mm %ld/%ld (%.3f%%) max ULP %u max rel %.3e; cross-rank identical %ld/%ld\n",
               bf16_seeds_match, (long) nsd, bf16_mm, bf16_tot, bp, bf16_max_ulp, bf16_max_rel, bf16_xrank, (long) nsd);
        fflush(stdout);
    }

    for (ncclComm_t c : comms) {
        ncclCommDestroy(c);
    }
    printf("DONE\n");
    return 0;
}
