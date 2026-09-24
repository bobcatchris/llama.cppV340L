// P2P flat-allreduce probe for the P2P-ALLREDUCE desk (W31).
//
// Question 1 (FEAS): does this ROCm 6.2.0 (HIP 6.2.41133) + gfx900 stack
// allow peer access between the Vega 10 dies under the two PM8533
// switches?  RCCL 2.20.5's own init log of record
// (results/rccl_multidie_a3_fp_default_20260923_153543.log) shows
// "Could not enable P2P" for BOTH same-switch pairs (05:00.0<->08:00.0,
// 0d:00.0<->10:00.0) and then connects every ring hop "via SHM" - but
// that is RCCL's enable path, not necessarily the platform's ceiling.
// This probe answers it directly: hipDeviceCanAccessPeer matrix,
// hipDeviceEnablePeerAccess matrix, a kernel peer-store round trip, and
// a hipMemcpyPeerAsync round trip, per ordered pair.
//
// Question 2 (BENCH): a FLAT (one-shot, sliced, PUSH) allreduce for the
// served decode boundary class, replacing the 6-serialized-hop RCCL ring
// (banked floors: 69.5 us transport-free ring / 103.2 us 4-die burst /
// 126.5 us 4-die lockstep at 80 KB, receipts W8/W9) with 2 concurrent
// transport phases:
//   phase A scatter: die d pushes its chunk p (ne/n elements) into a
//                    staging row on die p (P2P store) or into its own
//                    pinned host row (host arm), then arrival token
//   phase B reduce : die d sums its own chunk: local + peers' rows,
//                    fixed order (local first, peers ascending)
//   phase C bcast  : die d pushes the reduced chunk to every peer's
//                    result buffer (P2P arm) or posts it to its own
//                    pinned host slot which peers pull (host arm)
//   phase D        : spin for peers' bcast tokens, done
// PUSH not PULL for data: PCIe peer stores are posted TLPs (no
// completion round trip); peer loads are non-posted (completion per
// request).  At 20 KB latency-bound slices posted stores are the cheaper
// primitive, and single-writer arrival slots need no atomics - the
// in-tree 2-GPU kernel's scheme (ggml-cuda/allreduce.cu:39-107),
// generalized to n dies and to peer-device staging.
//
// Arms:
//   p2p  : staging + flags + result in device memory; cross-die data
//          moves as kernel peer stores (needs FEAS to pass for every
//          ordered pair among the participating dies)
//   host : staging + flags in mapped pinned host memory; data moves as
//          device post-to-host + peer pull-from-host (no P2P needed -
//          the arm that survives even if FEAS kills direct P2P; this is
//          the RCCL SHM transport class, but FLAT: 2 phases vs 6 hops)
//
// Pacings (W9 instrument convention): lockstep = sync + host barrier
// per op (isolated latency, includes launch); burst = 100 back-to-back
// stream-pipelined ops bracketed by HIP events (the served regime;
// per-op = elapsed/100).  Halves + even/odd spread law (3%/2% class).
//
// Served boundary sizes of record (W8 census): verify/catch-up ne 20480
// (80 KB fp32, hidden 5120 x T=4), draft ne 5120 (20 KB, T=1); fp32 band
// gate ggml-cuda.cu:1259 (n>=4, ne<262144).
//
// Correctness: partial[d][e] = deterministic pattern; CPU reference sums
// each chunk in the kernel's exact order (local first, peers ascending)
// -> bit-exact compare, plus cross-die bit-identity of results.  Kernels
// carry a spin bound that writes an error flag and exits, so a wedged
// handshake cannot hang a die (kill-by-PID stays sufficient).
//
// Build: hipcc -O2 -x hip p2p_allreduce_probe.cu -o p2p_allreduce_probe
// Run:   ./p2p_allreduce_probe [--ranks 4] [--iters 2000]
//           [--sizes 20480,5120] [--arms p2p,host]
//           [--pacing lockstep,burst] [--timeout 600]
// 2-die enablement check: --ranks 2 (runs FEAS on all visible pairs
// regardless; bench uses the first --ranks dies).

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define MAXR 4          // participating dies
#define NBLK 8          // blocks per die (in-tree AR kernel block count)
#define BLOCK 256       // threads per block
#define FLAG_STRIDE 16  // ints per (src, block) arrival slot = 64 B line

static double now_us() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::micro>>(steady_clock::now().time_since_epoch()).count();
}

static __device__ __forceinline__ void sig_set(int * p, int token) {
    *(volatile int *) p = token;
}
static __device__ __forceinline__ int sig_get(const int * p) {
    return *(const volatile int *) p;
}

// Spin with bound: returns 0 on success, 1 on escape (peer never
// arrived).  The escape writes the rank into err_flag so the host can
// name the dead pair; the kernel then returns without touching data.
static __device__ __forceinline__ int spin_until(const int * p, int token, int * err_flag, int rank) {
    const long limit = 400000000L;
    long spins = 0;
    while (sig_get(p) != token) {
        if (++spins > limit) {
            if (threadIdx.x == 0) { *err_flag = rank + 1; }
            return 1;
        }
    }
    return 0;
}

// Flag slot layout per die: [slot][src][block*FLAG_STRIDE].
static __device__ __forceinline__ size_t flag_idx(int slot, int src, int blk) {
    return (((size_t) slot * MAXR) + src) * (NBLK * FLAG_STRIDE) + (size_t) blk * FLAG_STRIDE;
}

// Flat PUSH allreduce, P2P arm.  One launch per boundary per die; blocks
// stripe the copies and own independent arrival slots, so no grid sync
// and no cooperative launch (the in-tree chunked-kernel structure).
//
// stage layout on die d:  [slot][src][C]      (peers P2P-store row src)
// flagA/flagB on die d:   [slot][src][blk]    (peers P2P-store tokens)
// result on die d:        [slot][ne]          (peers P2P-store chunk d)
__global__ void flat_ar_p2p_kernel(
        int rank, int n, int ne, int token,
        const float * __restrict__ my_partial,
        float * __restrict__ my_result,   // [2][ne]
        float * __restrict__ my_stage,    // [2][MAXR][C] on this die
        int * __restrict__ my_flagA,      // [2][MAXR][NBLK*STRIDE]
        int * __restrict__ my_flagB,
        float * const * peer_stage,   // [MAXR] peer bases (this die's VA)
        float * const * peer_result,  // [MAXR]
        int * const * peer_flagA,     // [MAXR]
        int * const * peer_flagB,
        int * err_flag) {

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt  = gridDim.x * blockDim.x;
    const int C    = ne / n;
    const int slot = token & 1;

    // phase A: push my chunk p into die p's staging row for me
    for (int p = 0; p < n; p++) {
        if (p == rank) { continue; }
        const float * src = my_partial + (size_t) p * C;
        float * dst = peer_stage[p] + (((size_t) slot * MAXR + rank) * C);
        for (int i = gtid; i < C; i += gnt) { dst[i] = src[i]; }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        for (int p = 0; p < n; p++) {
            if (p == rank) { continue; }
            sig_set(peer_flagA[p] + flag_idx(slot, rank, bid), token);
        }
        __threadfence_system();
    }
    __syncthreads();

    // wait A: all peers' chunk-rank contributions in my staging
    __threadfence_system();
    for (int p = 0; p < n; p++) {
        if (p == rank) { continue; }
        if (spin_until(my_flagA + flag_idx(slot, p, bid), token, err_flag, rank)) { return; }
    }
    __threadfence_system();
    __syncthreads();

    // phase B: reduce my chunk, fixed order (local first, peers ascending)
    {
        const float * loc = my_partial + (size_t) rank * C;
        float * out = my_result + (size_t) slot * ne + (size_t) rank * C;
        for (int i = gtid; i < C; i += gnt) {
            float acc = loc[i];
            for (int p = 0; p < n; p++) {
                if (p == rank) { continue; }
                const float * st = my_stage + (((size_t) slot * MAXR + p) * C);
                acc += st[i];
            }
            out[i] = acc;
        }
    }
    __threadfence_system();
    __syncthreads();

    // phase C: push my reduced chunk into every peer's result buffer
    {
        const float * src = my_result + (size_t) slot * ne + (size_t) rank * C;
        for (int p = 0; p < n; p++) {
            if (p == rank) { continue; }
            float * dst = peer_result[p] + (size_t) slot * ne + (size_t) rank * C;
            for (int i = gtid; i < C; i += gnt) { dst[i] = src[i]; }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        for (int p = 0; p < n; p++) {
            if (p == rank) { continue; }
            sig_set(peer_flagB[p] + flag_idx(slot, rank, bid), token);
        }
        __threadfence_system();
    }
    __syncthreads();

    // wait D: peers' reduced chunks in my result
    __threadfence_system();
    for (int p = 0; p < n; p++) {
        if (p == rank) { continue; }
        if (spin_until(my_flagB + flag_idx(slot, p, bid), token, err_flag, rank)) { return; }
    }
    __threadfence_system();
}

// Flat allreduce, host arm (no P2P dependency).  Data path per phase:
// post to OWN pinned host slot (posted write over PCIe), token in pinned
// host flags; peers PULL from my slot (reads with completions) directly
// into device memory.  Same phases, same tokens, same 2-slot reuse.
//
// host_stage[r] layout: [slot][dst][C]  (die r's chunk-dst partial)
// host_res[r] layout:   [slot][C]       (die r's reduced chunk)
// host_flagA/B[r]:      [slot][src][blk]
__global__ void flat_ar_host_kernel(
        int rank, int n, int ne, int token,
        const float * __restrict__ my_partial,
        float * __restrict__ my_result,     // [2][ne] device
        float * __restrict__ my_dev_stage,  // [2][MAXR][C] device, filled by pull
        float * __restrict__ my_host_stage, // [2][MAXR][C] pinned, mapped
        float * __restrict__ my_host_res,   // [2][C] pinned, mapped
        int * __restrict__ my_flagA,        // pinned, mapped
        int * __restrict__ my_flagB,
        float * const * peer_host_stage, // [MAXR] mapped-for-me
        float * const * peer_host_res,   // [MAXR]
        int * const * peer_flagA,        // [MAXR]
        int * const * peer_flagB,
        int * err_flag) {

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt  = gridDim.x * blockDim.x;
    const int C    = ne / n;
    const int slot = token & 1;

    // phase A1: post my chunk p into MY host row for dst p
    for (int p = 0; p < n; p++) {
        if (p == rank) { continue; }
        const float * src = my_partial + (size_t) p * C;
        float * dst = my_host_stage + (((size_t) slot * MAXR + p) * C);
        for (int i = gtid; i < C; i += gnt) { dst[i] = src[i]; }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        sig_set(my_flagA + flag_idx(slot, rank, bid), token);
        __threadfence_system();
    }
    __syncthreads();

    // wait A: every peer has posted its contribution for my chunk
    __threadfence_system();
    for (int p = 0; p < n; p++) {
        if (p == rank) { continue; }
        if (spin_until(my_flagA + flag_idx(slot, p, bid), token, err_flag, rank)) { return; }
    }
    __threadfence_system();
    __syncthreads();

    // phase A2: pull peers' rows into my device staging
    for (int p = 0; p < n; p++) {
        if (p == rank) { continue; }
        const float * src = peer_host_stage[p] + (((size_t) slot * MAXR + rank) * C);
        float * dst = my_dev_stage + (((size_t) slot * MAXR + p) * C);
        for (int i = gtid; i < C; i += gnt) { dst[i] = src[i]; }
    }
    __threadfence_system();
    __syncthreads();

    // phase B: reduce my chunk (local first, peers ascending)
    {
        const float * loc = my_partial + (size_t) rank * C;
        float * out = my_result + (size_t) slot * ne + (size_t) rank * C;
        for (int i = gtid; i < C; i += gnt) {
            float acc = loc[i];
            for (int p = 0; p < n; p++) {
                if (p == rank) { continue; }
                const float * st = my_dev_stage + (((size_t) slot * MAXR + p) * C);
                acc += st[i];
            }
            out[i] = acc;
        }
    }
    __threadfence_system();
    __syncthreads();

    // phase C1: post my reduced chunk to my host result slot
    {
        const float * src = my_result + (size_t) slot * ne + (size_t) rank * C;
        float * dst = my_host_res + (size_t) slot * C;
        for (int i = gtid; i < C; i += gnt) { dst[i] = src[i]; }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        sig_set(my_flagB + flag_idx(slot, rank, bid), token);
        __threadfence_system();
    }
    __syncthreads();

    // wait D1: every peer has posted its reduced chunk
    __threadfence_system();
    for (int p = 0; p < n; p++) {
        if (p == rank) { continue; }
        if (spin_until(my_flagB + flag_idx(slot, p, bid), token, err_flag, rank)) { return; }
    }
    __threadfence_system();
    __syncthreads();

    // phase D2: pull peers' reduced chunks into my device result
    for (int p = 0; p < n; p++) {
        if (p == rank) { continue; }
        const float * src = peer_host_res[p] + (size_t) slot * C;
        float * dst = my_result + (size_t) slot * ne + (size_t) rank * C;
        for (int i = gtid; i < C; i += gnt) { dst[i] = src[i]; }
    }
    __threadfence_system();
}

// FEAS helpers: pattern peer-store kernel and verify kernel.
__global__ void peer_store_kernel(float * peer_buf, const float * my_val, int n, int token) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { peer_buf[i] = my_val[i] + (float) token; }
    __threadfence_system();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *(volatile int *) (peer_buf + n) = token;
    }
}

__global__ void host_post_kernel(float * host_buf, const float * my_val, int n, int token) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { host_buf[i] = my_val[i] + (float) token; }
    __threadfence_system();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *(volatile int *) (host_buf + n) = token;
    }
}

__global__ void verify_kernel(const float * buf, const float * expect, int n, int token, int * mism) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && buf[i] != expect[i] + (float) token) { atomicAdd(mism, 1); }
}

struct bench_stat { double avg, mn, mx, med; };

static bench_stat summarize(std::vector<double> & v) {
    bench_stat s{0.0, 1e30, 0.0, 0.0};
    for (double x : v) { s.avg += x; s.mn = std::min(s.mn, x); s.mx = std::max(s.mx, x); }
    s.avg /= (double) v.size();
    std::sort(v.begin(), v.end());
    s.med = v[v.size() / 2];
    return s;
}

static void report(const char * arm, const char * pac, int n, double kb, int iters,
                   std::vector<double> & t) {
    std::vector<double> h1(t.begin(), t.begin() + t.size() / 2);
    std::vector<double> h2(t.begin() + t.size() / 2, t.end());
    std::vector<double> e, o;
    for (size_t k = 0; k < t.size(); k++) { (k % 2 == 0 ? e : o).push_back(t[k]); }
    bench_stat st = summarize(t), s1 = summarize(h1), s2 = summarize(h2);
    bench_stat se = summarize(e), so = summarize(o);
    const double spread = 100.0 * fabs(s1.avg - s2.avg) / std::min(s1.avg, s2.avg);
    const double spreo  = 100.0 * fabs(se.avg - so.avg) / std::min(se.avg, so.avg);
    printf("%-5s/%-8s n=%d %6.1f KB x%4d: med %7.2f us  avg %7.2f  min %7.2f  "
           "halves %7.2f/%7.2f %.2f%%  evenodd %.2f%%%s%s\n",
           arm, pac, n, kb, iters, st.med, st.avg, st.mn, s1.avg, s2.avg, spread, spreo,
           spread <= 3.0 ? "" : " [SPREAD-FAIL]",
           spreo <= 2.0 ? "" : " [EVENODD-FAIL]");
    fflush(stdout);
}

struct feas_pair { int can, ena, kstore, mpeer; };

int main(int argc, char ** argv) {
    int    iters = 2000, timeout = 600, nranks = 4;
    std::vector<int> sizes = {20480, 5120};
    std::string arms = "p2p,host", pacing = "lockstep,burst";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return argv[++i]; };
        if      (a == "--iters")   { iters = atoi(next()); }
        else if (a == "--ranks")   { nranks = atoi(next()); }
        else if (a == "--timeout") { timeout = atoi(next()); }
        else if (a == "--arms")    { arms = next(); }
        else if (a == "--pacing")  { pacing = next(); }
        else if (a == "--sizes")   { sizes.clear(); char * s = strdup(next());
                                     for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) sizes.push_back(atoi(t));
                                     free(s); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    signal(SIGALRM, [](int) { fprintf(stderr, "probe timeout\n"); _exit(3); });
    alarm(timeout);

    int ndev = 0;
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipGetDeviceCount(&ndev));
    printf("p2p_allreduce_probe: visible devices = %d\n", ndev);
    if (ndev < 1) { fprintf(stderr, "no visible devices\n"); return 1; }
    if (nranks > ndev) {
        printf("ranks %d > visible %d; clamping to %d\n", nranks, ndev, ndev);
        nranks = ndev;
    }
    if (nranks > MAXR) { nranks = MAXR; }
    const int n = nranks;

    // ================= FEAS: peer access matrix + round trips =================
    printf("\n==== FEAS: hipDeviceCanAccessPeer / EnablePeerAccess / kernel peer store / hipMemcpyPeerAsync ====\n");
    feas_pair fp[MAXR][MAXR] = {};
    int p2p_ok_all = 1;

    // host-mapping feasibility (needed by the host arm on HIP: the in-tree
    // internal AR is CUDA-only over these APIs - allreduce.cu:957)
    {
        float * hbuf = nullptr, * dsrc = nullptr;
        HIP_CHECK(hipSetDevice(0));
        hipError_t rc = hipHostMalloc((void **) &hbuf, (64 + 4) * sizeof(float), hipHostMallocMapped);
        printf("HOSTMAP: hipHostMalloc(Mapped) -> %s", rc == hipSuccess ? "OK" : hipGetErrorString(rc));
        if (rc == hipSuccess) {
            float * dmap = nullptr;
            rc = hipHostGetDevicePointer((void **) &dmap, hbuf, 0);
            printf(", hipHostGetDevicePointer -> %s (h=%p d=%p)\n",
                   rc == hipSuccess ? "OK" : hipGetErrorString(rc), (void *) hbuf, (void *) dmap);
            if (rc == hipSuccess) {
                HIP_CHECK(hipMalloc(&dsrc, 64 * sizeof(float)));
                std::vector<float> h(64, 3.5f);
                HIP_CHECK(hipMemcpy(dsrc, h.data(), 64 * sizeof(float), hipMemcpyHostToDevice));
                int * mism = nullptr;
                HIP_CHECK(hipMalloc((void **) &mism, sizeof(int)));
                HIP_CHECK(hipMemset(mism, 0, sizeof(int)));
                hipStream_t st = nullptr;
                HIP_CHECK(hipStreamCreate(&st));
                host_post_kernel<<<1, 64, 0, st>>>(dmap, dsrc, 64, 7);
                verify_kernel<<<1, 64, 0, st>>>(dmap, dsrc, 64, 7, mism);
                HIP_CHECK(hipStreamSynchronize(st));
                int m = 0;
                HIP_CHECK(hipMemcpy(&m, mism, sizeof(int), hipMemcpyDeviceToHost));
                int tok = 0;
                HIP_CHECK(hipMemcpy(&tok, hbuf + 64, sizeof(int), hipMemcpyHostToHost));
                printf("HOSTMAP: post+verify from device: mism=%d token=%d -> %s\n", m, tok,
                       (m == 0 && tok == 7) ? "OK" : "FAIL");
                HIP_CHECK(hipStreamDestroy(st));
                HIP_CHECK(hipFree(mism));
                HIP_CHECK(hipFree(dsrc));
            }
            HIP_CHECK(hipHostFree(hbuf));
        } else {
            printf("\n");
        }
    }

    // per ordered pair (i -> j): can i reach j's memory?
    for (int i = 0; i < ndev; i++) {
        for (int j = 0; j < ndev; j++) {
            if (i == j) { continue; }
            feas_pair & f = fp[i][j];
            int can = 0;
            HIP_CHECK(hipSetDevice(i));
            hipError_t rc = hipDeviceCanAccessPeer(&can, i, j);
            f.can = (rc == hipSuccess) ? can : -1;
            if (f.can == 1) {
                rc = hipDeviceEnablePeerAccess(j, 0);
                if (rc == hipErrorPeerAccessAlreadyEnabled) { rc = hipSuccess; }
                if (rc != hipSuccess) {
                    (void) hipGetLastError(); // clear sticky
                }
                f.ena = (rc == hipSuccess) ? 1 : 0;
            }
            printf("FEAS %d->%d: canAccess=%d enable=%d\n", i, j, f.can, f.ena);
        }
    }

    // kernel peer-store + hipMemcpyPeerAsync round trips (only for enabled pairs)
    {
        float * dbuf[MAXR] = {}, * dval[MAXR] = {}, * dpat[MAXR] = {};
        int * mism[MAXR] = {};
        hipStream_t st[MAXR] = {};
        for (int d = 0; d < ndev; d++) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipMalloc(&dbuf[d], (2048 + 4) * sizeof(float)));
            HIP_CHECK(hipMalloc(&dval[d], 2048 * sizeof(float)));
            HIP_CHECK(hipMalloc(&dpat[d], 2048 * sizeof(float)));
            HIP_CHECK(hipMemset(dbuf[d], 0, (2048 + 4) * sizeof(float)));
            HIP_CHECK(hipMalloc(&mism[d], sizeof(int)));
            HIP_CHECK(hipStreamCreate(&st[d]));
            std::vector<float> h(2048);
            for (int k = 0; k < 2048; k++) { h[k] = 0.125f * d + 0.001f * k; }
            HIP_CHECK(hipMemcpy(dval[d], h.data(), 2048 * sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dpat[d], h.data(), 2048 * sizeof(float), hipMemcpyHostToDevice));
        }
        const int nprobe = 256;
        for (int i = 0; i < ndev; i++) {
            for (int j = 0; j < ndev; j++) {
                if (i == j || !fp[i][j].ena) { continue; }
                // die i's kernel writes die j's buffer using the raw pointer
                // allocated on die j (the same-VA assumption CUDA semantics
                // give; this test is what proves or kills it on HIP)
                HIP_CHECK(hipSetDevice(i));
                HIP_CHECK(hipMemset(mism[i], 0, sizeof(int)));
                peer_store_kernel<<<(nprobe + 255) / 256, 256, 0, st[i]>>>(dbuf[j], dval[i], nprobe, 42);
                HIP_CHECK(hipStreamSynchronize(st[i]));
                // verify ON die j against a j-local copy of the pattern
                HIP_CHECK(hipSetDevice(j));
                HIP_CHECK(hipMemset(mism[j], 0, sizeof(int)));
                verify_kernel<<<(nprobe + 255) / 256, 256, 0, st[j]>>>(dbuf[j], dpat[i], nprobe, 42, mism[j]);
                HIP_CHECK(hipStreamSynchronize(st[j]));
                int m = 0, tok = 0;
                HIP_CHECK(hipMemcpy(&m, mism[j], sizeof(int), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(&tok, (const void *) (dbuf[j] + nprobe), sizeof(int), hipMemcpyDeviceToHost));
                fp[i][j].kstore = (m == 0 && tok == 42) ? 1 : 0;

                HIP_CHECK(hipMemset(dbuf[j], 0, nprobe * sizeof(float)));
                hipError_t rc = hipMemcpyPeerAsync(dbuf[j], j, dval[i], i, nprobe * sizeof(float), st[i]);
                if (rc != hipSuccess) { (void) hipGetLastError(); }
                HIP_CHECK(hipStreamSynchronize(st[i]));
                if (rc == hipSuccess) {
                    HIP_CHECK(hipSetDevice(j));
                    HIP_CHECK(hipMemset(mism[j], 0, sizeof(int)));
                    verify_kernel<<<(nprobe + 255) / 256, 256, 0, st[j]>>>(dbuf[j], dpat[i], nprobe, 0, mism[j]);
                    HIP_CHECK(hipStreamSynchronize(st[j]));
                    HIP_CHECK(hipMemcpy(&m, mism[j], sizeof(int), hipMemcpyDeviceToHost));
                    fp[i][j].mpeer = (m == 0) ? 1 : 0;
                }
                printf("FEAS %d->%d: kstore=%d memcpyPeer=%d\n", i, j, fp[i][j].kstore, fp[i][j].mpeer);
            }
        }
        for (int d = 0; d < ndev; d++) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipFree(dbuf[d]));
            HIP_CHECK(hipFree(dval[d]));
            HIP_CHECK(hipFree(dpat[d]));
            HIP_CHECK(hipFree(mism[d]));
            HIP_CHECK(hipStreamDestroy(st[d]));
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && !(fp[i][j].ena && fp[i][j].kstore)) { p2p_ok_all = 0; }
        }
    }
    printf("FEAS verdict: %s\n", p2p_ok_all ?
           "P2P available for all participating ordered pairs (p2p arm runs)" :
           "P2P NOT available for all pairs (p2p arm skipped, host arm carries the design)");

    // ================= BENCH =================
    const size_t max_ne = *std::max_element(sizes.begin(), sizes.end());
    for (int ne : sizes) {
        if (ne % n != 0) {
            printf("size %d not divisible by n=%d; skipped\n", ne, n);
            continue;
        }
        const int C = ne / n;
        const double kb = ne * sizeof(float) / 1024.0;

        // per-die resources
        float * partial[MAXR] = {}, * result[MAXR] = {};
        float * stage_dev[MAXR] = {}, * res_host[MAXR] = {};
        float * stage_host[MAXR] = {};
        int * flagA_dev[MAXR] = {}, * flagB_dev[MAXR] = {};
        int * flagA_host[MAXR] = {}, * flagB_host[MAXR] = {};
        int * errflag[MAXR] = {};
        float * tab_stage_dev[MAXR][MAXR] = {}, * tab_result[MAXR][MAXR] = {};
        int * tab_flagA_dev[MAXR][MAXR] = {}, * tab_flagB_dev[MAXR][MAXR] = {};
        float * tab_stage_host[MAXR][MAXR] = {}, * tab_res_host[MAXR][MAXR] = {};
        int * tab_flagA_host[MAXR][MAXR] = {}, * tab_flagB_host[MAXR][MAXR] = {};
        float ** d_tab_p2p[MAXR] = {}, ** d_tab_res_p2p[MAXR] = {};
        int ** d_tabfA_p2p[MAXR] = {}, ** d_tabfB_p2p[MAXR] = {};
        float ** d_tab_stage_host[MAXR] = {}, ** d_tab_res_host[MAXR] = {};
        int ** d_tabfA_host[MAXR] = {}, ** d_tabfB_host[MAXR] = {};
        hipStream_t stream[MAXR] = {};
        hipEvent_t ev0[MAXR] = {}, ev1[MAXR] = {};

        for (int d = 0; d < n; d++) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipStreamCreate(&stream[d]));
            HIP_CHECK(hipEventCreate(&ev0[d]));
            HIP_CHECK(hipEventCreate(&ev1[d]));
            HIP_CHECK(hipMalloc(&partial[d], 2 * max_ne * sizeof(float)));
            HIP_CHECK(hipMalloc(&result[d], 2 * 2 * max_ne * sizeof(float))); // [2][2*max_ne]
            HIP_CHECK(hipMalloc(&stage_dev[d], 2 * MAXR * (max_ne / n) * sizeof(float)));
            HIP_CHECK(hipMalloc(&flagA_dev[d], 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int)));
            HIP_CHECK(hipMalloc(&flagB_dev[d], 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int)));
            HIP_CHECK(hipMalloc(&errflag[d], sizeof(int)));
            HIP_CHECK(hipMemset(flagA_dev[d], 0, 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int)));
            HIP_CHECK(hipMemset(flagB_dev[d], 0, 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int)));
            HIP_CHECK(hipMemset(errflag[d], 0, sizeof(int)));
            // host-arm pinned buffers, mapped per die
            HIP_CHECK(hipHostMalloc((void **) &stage_host[d], 2 * MAXR * (max_ne / n) * sizeof(float), hipHostMallocMapped));
            HIP_CHECK(hipHostMalloc((void **) &res_host[d], 2 * (max_ne / n) * sizeof(float), hipHostMallocMapped));
            HIP_CHECK(hipHostMalloc((void **) &flagA_host[d], 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int), hipHostMallocMapped));
            HIP_CHECK(hipHostMalloc((void **) &flagB_host[d], 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int), hipHostMallocMapped));
            memset(stage_host[d], 0, 2 * MAXR * (max_ne / n) * sizeof(float));
            memset(res_host[d], 0, 2 * (max_ne / n) * sizeof(float));
            memset(flagA_host[d], 0, 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int));
            memset(flagB_host[d], 0, 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int));
            // per-die device mappings of every die's host buffers (host arm)
            for (int e = 0; e < n; e++) {
                HIP_CHECK(hipSetDevice(e));
                float * m1 = nullptr, * m2 = nullptr;
                int * f1 = nullptr, * f2 = nullptr;
                HIP_CHECK(hipHostGetDevicePointer((void **) &m1, stage_host[d], 0));
                HIP_CHECK(hipHostGetDevicePointer((void **) &m2, res_host[d], 0));
                HIP_CHECK(hipHostGetDevicePointer((void **) &f1, flagA_host[d], 0));
                HIP_CHECK(hipHostGetDevicePointer((void **) &f2, flagB_host[d], 0));
                tab_stage_host[e][d] = m1;
                tab_res_host[e][d] = m2;
                tab_flagA_host[e][d] = f1;
                tab_flagB_host[e][d] = f2;
            }
            HIP_CHECK(hipSetDevice(d));
            // p2p tables: peer device pointers (valid on d only if peer access d->p enabled)
            for (int p = 0; p < n; p++) {
                tab_stage_dev[d][p] = stage_dev[p];
                tab_result[d][p] = result[p];
                tab_flagA_dev[d][p] = flagA_dev[p];
                tab_flagB_dev[d][p] = flagB_dev[p];
            }
        }

        // device copies of the pointer tables
        for (int d = 0; d < n; d++) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipMalloc(&d_tab_p2p[d], MAXR * sizeof(float *)));
            HIP_CHECK(hipMalloc(&d_tab_res_p2p[d], MAXR * sizeof(float *)));
            HIP_CHECK(hipMalloc(&d_tabfA_p2p[d], MAXR * sizeof(int *)));
            HIP_CHECK(hipMalloc(&d_tabfB_p2p[d], MAXR * sizeof(int *)));
            HIP_CHECK(hipMalloc(&d_tab_stage_host[d], MAXR * sizeof(float *)));
            HIP_CHECK(hipMalloc(&d_tab_res_host[d], MAXR * sizeof(float *)));
            HIP_CHECK(hipMalloc(&d_tabfA_host[d], MAXR * sizeof(int *)));
            HIP_CHECK(hipMalloc(&d_tabfB_host[d], MAXR * sizeof(int *)));
            HIP_CHECK(hipMemcpy(d_tab_p2p[d], tab_stage_dev[d], MAXR * sizeof(float *), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_tab_res_p2p[d], tab_result[d], MAXR * sizeof(float *), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_tabfA_p2p[d], tab_flagA_dev[d], MAXR * sizeof(int *), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_tabfB_p2p[d], tab_flagB_dev[d], MAXR * sizeof(int *), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_tab_stage_host[d], tab_stage_host[d], MAXR * sizeof(float *), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_tab_res_host[d], tab_res_host[d], MAXR * sizeof(float *), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_tabfA_host[d], tab_flagA_host[d], MAXR * sizeof(int *), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_tabfB_host[d], tab_flagB_host[d], MAXR * sizeof(int *), hipMemcpyHostToDevice));
        }

        // deterministic partials: partial[d][e] = 0.25*d + 1e-3*(e%97);
        // per-chunk CPU reference replicates the kernel's exact add order
        // (local first, peers ascending) so the compare is bit-exact
        std::vector<float> ref_exact(ne, 0.0f);
        {
            // kernel order per chunk c: local(c) first, then peers ascending
            std::vector<float> part[MAXR];
            for (int d = 0; d < n; d++) {
                part[d].resize(ne);
                for (int e = 0; e < ne; e++) { part[d][e] = 0.25f * d + 1e-3f * (e % 97); }
            }
            for (int c = 0; c < n; c++) {
                for (int e = 0; e < C; e++) {
                    float acc = part[c][c * C + e];
                    for (int p = 0; p < n; p++) {
                        if (p == c) { continue; }
                        acc += part[p][c * C + e];
                    }
                    ref_exact[c * C + e] = acc;
                }
            }
            for (int d = 0; d < n; d++) {
                HIP_CHECK(hipSetDevice(d));
                HIP_CHECK(hipMemcpy(partial[d], part[d].data(), ne * sizeof(float), hipMemcpyHostToDevice));
            }
        }

        // ---- correctness pass (one lockstep op per arm), then timing ----
        auto check_err = [&](const char * arm) -> bool {
            for (int d = 0; d < n; d++) {
                int e = 0;
                HIP_CHECK(hipSetDevice(d));
                HIP_CHECK(hipMemcpy(&e, errflag[d], sizeof(int), hipMemcpyDeviceToHost));
                if (e != 0) {
                    printf("ARM %-4s: kernel spin-bound escape on die %d (errflag=%d) - handshake broken, arm aborted\n",
                           arm, d, e);
                    return false;
                }
            }
            return true;
        };

        auto verify_results = [&](const char * arm, int slot) -> bool {
            std::vector<float> first;
            bool ok = true;
            for (int d = 0; d < n; d++) {
                std::vector<float> h(ne);
                HIP_CHECK(hipSetDevice(d));
                HIP_CHECK(hipMemcpy(h.data(), result[d] + (size_t) slot * ne, ne * sizeof(float), hipMemcpyDeviceToHost));
                for (int e = 0; e < ne; e++) {
                    if (h[e] != ref_exact[e]) {
                        printf("ARM %-4s die %d CORRECTNESS FAIL at e=%d: got %.7f want %.7f\n",
                               arm, d, e, (double) h[e], (double) ref_exact[e]);
                        ok = false;
                        break;
                    }
                }
                if (!ok) { break; }
                if (first.empty()) { first = h; }
                else if (memcmp(first.data(), h.data(), ne * sizeof(float)) != 0) {
                    printf("ARM %-4s die %d CROSS-DIE RESULT MISMATCH\n", arm, d);
                    ok = false;
                    break;
                }
            }
            if (ok) { printf("ARM %-4s: correctness OK (bit-exact vs CPU same-order ref, cross-die identical)\n", arm); }
            return ok;
        };

        printf("\n==== BENCH ne=%d (%.0f KB fp32), n=%d, chunk %d elems ====\n", ne, kb, n, C);

        int token = 0;
        const int warm = 100;

        // -------- p2p arm --------
        if (arms.find("p2p") != std::string::npos && p2p_ok_all) {
            for (int d = 0; d < n; d++) {
                HIP_CHECK(hipSetDevice(d));
                HIP_CHECK(hipMemset(result[d], 0, 2 * 2 * max_ne * sizeof(float)));
                HIP_CHECK(hipMemset(errflag[d], 0, sizeof(int)));
            }
            // correctness pass
            ++token;
            for (int d = 0; d < n; d++) {
                HIP_CHECK(hipSetDevice(d));
                flat_ar_p2p_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                    d, n, ne, token, partial[d], result[d], stage_dev[d],
                    flagA_dev[d], flagB_dev[d], d_tab_p2p[d], d_tab_res_p2p[d],
                    d_tabfA_p2p[d], d_tabfB_p2p[d], errflag[d]);
            }
            for (int d = 0; d < n; d++) { HIP_CHECK(hipStreamSynchronize(stream[d])); }
            if (check_err("p2p")) { verify_results("p2p", token & 1); }

            if (pacing.find("lockstep") != std::string::npos) {
                for (int k = 0; k < warm; k++) {
                    ++token;
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        flat_ar_p2p_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                            d, n, ne, token, partial[d], result[d], stage_dev[d],
                            flagA_dev[d], flagB_dev[d], d_tab_p2p[d], d_tab_res_p2p[d],
                            d_tabfA_p2p[d], d_tabfB_p2p[d], errflag[d]);
                    }
                    for (int d = 0; d < n; d++) { HIP_CHECK(hipStreamSynchronize(stream[d])); }
                }
                std::vector<double> t;
                t.reserve(iters);
                for (int k = 0; k < iters; k++) {
                    ++token;
                    const double s = now_us();
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        flat_ar_p2p_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                            d, n, ne, token, partial[d], result[d], stage_dev[d],
                            flagA_dev[d], flagB_dev[d], d_tab_p2p[d], d_tab_res_p2p[d],
                            d_tabfA_p2p[d], d_tabfB_p2p[d], errflag[d]);
                    }
                    for (int d = 0; d < n; d++) { HIP_CHECK(hipStreamSynchronize(stream[d])); }
                    t.push_back(now_us() - s);
                }
                report("p2p", "lockstep", n, kb, iters, t);
            }
            if (pacing.find("burst") != std::string::npos) {
                const int batch = 100;
                for (int k = 0; k < warm; k++) {
                    ++token;
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        flat_ar_p2p_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                            d, n, ne, token, partial[d], result[d], stage_dev[d],
                            flagA_dev[d], flagB_dev[d], d_tab_p2p[d], d_tab_res_p2p[d],
                            d_tabfA_p2p[d], d_tabfB_p2p[d], errflag[d]);
                    }
                }
                for (int d = 0; d < n; d++) { HIP_CHECK(hipStreamSynchronize(stream[d])); }
                std::vector<double> bt;
                bt.reserve(iters / batch + 1);
                int done = 0;
                while (done < iters) {
                    const int nb = std::min(batch, iters - done);
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        HIP_CHECK(hipEventRecord(ev0[d], stream[d]));
                    }
                    for (int k = 0; k < nb; k++) {
                        ++token;
                        for (int d = 0; d < n; d++) {
                            HIP_CHECK(hipSetDevice(d));
                            flat_ar_p2p_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                                d, n, ne, token, partial[d], result[d], stage_dev[d],
                                flagA_dev[d], flagB_dev[d], d_tab_p2p[d], d_tab_res_p2p[d],
                                d_tabfA_p2p[d], d_tabfB_p2p[d], errflag[d]);
                        }
                    }
                    float worst = 0.0f;
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        HIP_CHECK(hipEventRecord(ev1[d], stream[d]));
                        HIP_CHECK(hipStreamSynchronize(stream[d]));
                        float ms = 0.0f;
                        HIP_CHECK(hipEventElapsedTime(&ms, ev0[d], ev1[d]));
                        worst = std::max(worst, ms);
                    }
                    bt.push_back((double) worst * 1000.0 / nb);
                    done += nb;
                }
                report("p2p", "burst", n, kb, (int) bt.size(), bt);
                check_err("p2p");
            }
        } else if (arms.find("p2p") != std::string::npos) {
            printf("p2p arm: SKIPPED (feasibility)\n");
        }

        // -------- host arm --------
        if (arms.find("host") != std::string::npos) {
            for (int d = 0; d < n; d++) {
                HIP_CHECK(hipSetDevice(d));
                HIP_CHECK(hipMemset(result[d], 0, 2 * 2 * max_ne * sizeof(float)));
                HIP_CHECK(hipMemset(errflag[d], 0, sizeof(int)));
                memset(flagA_host[d], 0, 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int));
                memset(flagB_host[d], 0, 2 * MAXR * NBLK * FLAG_STRIDE * sizeof(int));
            }
            ++token;
            for (int d = 0; d < n; d++) {
                HIP_CHECK(hipSetDevice(d));
                float * my_stage_map = tab_stage_host[d][d];
                float * my_res_map = tab_res_host[d][d];
                int * my_fA = tab_flagA_host[d][d];
                int * my_fB = tab_flagB_host[d][d];
                flat_ar_host_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                    d, n, ne, token, partial[d], result[d], stage_dev[d],
                    my_stage_map, my_res_map, my_fA, my_fB,
                    d_tab_stage_host[d], d_tab_res_host[d], d_tabfA_host[d], d_tabfB_host[d],
                    errflag[d]);
            }
            for (int d = 0; d < n; d++) { HIP_CHECK(hipStreamSynchronize(stream[d])); }
            if (check_err("host")) { verify_results("host", token & 1); }

            if (pacing.find("lockstep") != std::string::npos) {
                for (int k = 0; k < warm; k++) {
                    ++token;
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        flat_ar_host_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                            d, n, ne, token, partial[d], result[d], stage_dev[d],
                            tab_stage_host[d][d], tab_res_host[d][d], tab_flagA_host[d][d], tab_flagB_host[d][d],
                            d_tab_stage_host[d], d_tab_res_host[d], d_tabfA_host[d], d_tabfB_host[d],
                            errflag[d]);
                    }
                    for (int d = 0; d < n; d++) { HIP_CHECK(hipStreamSynchronize(stream[d])); }
                }
                std::vector<double> t;
                t.reserve(iters);
                for (int k = 0; k < iters; k++) {
                    ++token;
                    const double s = now_us();
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        flat_ar_host_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                            d, n, ne, token, partial[d], result[d], stage_dev[d],
                            tab_stage_host[d][d], tab_res_host[d][d], tab_flagA_host[d][d], tab_flagB_host[d][d],
                            d_tab_stage_host[d], d_tab_res_host[d], d_tabfA_host[d], d_tabfB_host[d],
                            errflag[d]);
                    }
                    for (int d = 0; d < n; d++) { HIP_CHECK(hipStreamSynchronize(stream[d])); }
                    t.push_back(now_us() - s);
                }
                report("host", "lockstep", n, kb, iters, t);
            }
            if (pacing.find("burst") != std::string::npos) {
                const int batch = 100;
                for (int k = 0; k < warm; k++) {
                    ++token;
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        flat_ar_host_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                            d, n, ne, token, partial[d], result[d], stage_dev[d],
                            tab_stage_host[d][d], tab_res_host[d][d], tab_flagA_host[d][d], tab_flagB_host[d][d],
                            d_tab_stage_host[d], d_tab_res_host[d], d_tabfA_host[d], d_tabfB_host[d],
                            errflag[d]);
                    }
                }
                for (int d = 0; d < n; d++) { HIP_CHECK(hipStreamSynchronize(stream[d])); }
                std::vector<double> bt;
                bt.reserve(iters / batch + 1);
                int done = 0;
                while (done < iters) {
                    const int nb = std::min(batch, iters - done);
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        HIP_CHECK(hipEventRecord(ev0[d], stream[d]));
                    }
                    for (int k = 0; k < nb; k++) {
                        ++token;
                        for (int d = 0; d < n; d++) {
                            HIP_CHECK(hipSetDevice(d));
                            flat_ar_host_kernel<<<NBLK, BLOCK, 0, stream[d]>>>(
                                d, n, ne, token, partial[d], result[d], stage_dev[d],
                                tab_stage_host[d][d], tab_res_host[d][d], tab_flagA_host[d][d], tab_flagB_host[d][d],
                                d_tab_stage_host[d], d_tab_res_host[d], d_tabfA_host[d], d_tabfB_host[d],
                                errflag[d]);
                        }
                    }
                    float worst = 0.0f;
                    for (int d = 0; d < n; d++) {
                        HIP_CHECK(hipSetDevice(d));
                        HIP_CHECK(hipEventRecord(ev1[d], stream[d]));
                        HIP_CHECK(hipStreamSynchronize(stream[d]));
                        float ms = 0.0f;
                        HIP_CHECK(hipEventElapsedTime(&ms, ev0[d], ev1[d]));
                        worst = std::max(worst, ms);
                    }
                    bt.push_back((double) worst * 1000.0 / nb);
                    done += nb;
                }
                report("host", "burst", n, kb, (int) bt.size(), bt);
                check_err("host");
            }
        }

        // teardown
        for (int d = 0; d < n; d++) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipFree(partial[d]));
            HIP_CHECK(hipFree(result[d]));
            HIP_CHECK(hipFree(stage_dev[d]));
            HIP_CHECK(hipFree(flagA_dev[d]));
            HIP_CHECK(hipFree(flagB_dev[d]));
            HIP_CHECK(hipFree(errflag[d]));
            HIP_CHECK(hipFree(d_tab_p2p[d]));
            HIP_CHECK(hipFree(d_tab_res_p2p[d]));
            HIP_CHECK(hipFree(d_tabfA_p2p[d]));
            HIP_CHECK(hipFree(d_tabfB_p2p[d]));
            HIP_CHECK(hipFree(d_tab_stage_host[d]));
            HIP_CHECK(hipFree(d_tab_res_host[d]));
            HIP_CHECK(hipFree(d_tabfA_host[d]));
            HIP_CHECK(hipFree(d_tabfB_host[d]));
            HIP_CHECK(hipHostFree(stage_host[d]));
            HIP_CHECK(hipHostFree(res_host[d]));
            HIP_CHECK(hipHostFree(flagA_host[d]));
            HIP_CHECK(hipHostFree(flagB_host[d]));
            HIP_CHECK(hipEventDestroy(ev0[d]));
            HIP_CHECK(hipEventDestroy(ev1[d]));
            HIP_CHECK(hipStreamDestroy(stream[d]));
        }
    }

    printf("\nDONE\n");
    return 0;
}
