// IPC-OPEN PROBE for the ONE-SHOT-AR desk (W38).  Closes the one
// mechanism W31 never tested: vLLM custom_all_reduce-style IPC handles
// (hipIpcGetMemHandle / hipIpcOpenMemHandle) as the peer-mapping
// substrate for a flat one-shot allreduce kernel.
//
// W31 measured (results/p2p_allreduce_probe_r4_linebuf_20260924_155026.log):
//   FEAS i->j: canAccess=0 enable=0   for all 12 ordered pairs
// i.e. the hipDeviceEnablePeerAccess path is driver-refused on this
// Vega 10 / ROCm 6.2.0-66 stack.  The IPC handle path is a DIFFERENT
// API (BO import via KFD, not the enable-peer gate) and was never run:
// W31's FEAS short-circuited after enable=0, so hipMemcpyPeerAsync was
// also never exercised.  If IPC open + device read works despite
// canAccess=0, a one-shot AR arm revives; if not, the desk stays dead
// with the last question closed.
//
// Per ordered pair (i, j), single process, all visible dies:
//   1. dev j: hipMalloc 4 MiB, fill 0xAB, hipIpcGetMemHandle
//   2. dev i: hipIpcOpenMemHandle(hipIpcMemLazyEnablePeerAccess)
//   3. dev i: kernel reads the opened pointer, host verifies the magic
//      (then a 1 MiB strided read as a BW datapoint, if step 3 passes)
//   4. dev i: hipIpcCloseMemHandle
//   5. independent of 2-4: hipMemcpyPeerAsync dev j -> dev i round trip
//      (closes W31's short-circuited copy-engine question)
// Same-process open is deliberate: llama.cpp TP is inproc (one process,
// n devices).  If HIP rejects same-process IPC open outright, that is
// itself the verdict (IPC path would require multi-process serving).
//
// Risk note: a bad peer read can fault the context of dev i.  Pairs run
// i-major with handle close after each; a wedged run dies on --timeout
// (SIGALRM, PROBE-EXIT:3, kill-by-PID - W31 convention).  Zero-GPU now;
// staged for the next lock window (~30 s).
//
// Build: hipcc -O2 -x hip ipc_open_probe.cu -o ipc_open_probe
// Run:   ./ipc_open_probe [--ranks 4] [--timeout 300]
// Verdict: exit 0 if at least one pair OPEN+READ passes, else 1.

#include <hip/hip_runtime.h>

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

namespace {

volatile std::sig_atomic_t g_alarm = 0;

void on_alarm(int) {
    g_alarm = 1;
}

__global__ void read_magic_kernel(const unsigned int * src, unsigned int * dst) {
    dst[0] = src[0];
}

__global__ void read_sum_kernel(const unsigned int * src, unsigned long long * dst, int n_words) {
    // strided per-thread sum of a 1 MiB block, one block, 256 threads;
    // 64-bit wrap in the reference below matches, so the compare holds
    unsigned long long acc = 0;
    for (int i = threadIdx.x; i < n_words; i += blockDim.x) {
        acc += src[i];
    }
    atomicAdd(dst, acc);
}

} // namespace

int main(int argc, char ** argv) {
    int n_ranks = 4;
    int timeout_s = 300;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--ranks") && i + 1 < argc) {
            n_ranks = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--timeout") && i + 1 < argc) {
            timeout_s = atoi(argv[++i]);
        }
    }

    signal(SIGALRM, on_alarm);
    alarm(timeout_s);

    int n_dev = 0;
    HIP_CHECK(hipGetDeviceCount(&n_dev));
    if (n_dev < 2) {
        printf("IPC probe: need >= 2 devices, found %d\n", n_dev);
        return 1;
    }
    if (n_ranks > n_dev) {
        n_ranks = n_dev;
    }
    printf("==== IPC-OPEN FEAS: hipIpcGetMemHandle/OpenMemHandle + device read + hipMemcpyPeerAsync ====\n");
    printf("devices=%d ranks=%d (single process, inproc TP shape)\n", n_dev, n_ranks);

    const size_t buf_bytes = 4u << 20;
    const size_t sum_bytes = 1u << 20;
    const unsigned int magic = 0xABABABABu;

    int n_pass = 0;
    int n_pairs = 0;
    int memcpy_pass = 0;

    for (int i = 0; i < n_ranks && !g_alarm; ++i) {
        for (int j = 0; j < n_ranks && !g_alarm; ++j) {
            if (i == j) {
                continue;
            }
            ++n_pairs;

            // dev j: allocate + fill + export
            HIP_CHECK(hipSetDevice(j));
            unsigned int * d_src = nullptr;
            HIP_CHECK(hipMalloc(&d_src, buf_bytes));
            HIP_CHECK(hipMemset(d_src, 0xAB, buf_bytes));
            HIP_CHECK(hipDeviceSynchronize());
            hipIpcMemHandle_t handle;
            hipError_t e_get = hipIpcGetMemHandle(&handle, d_src);
            if (e_get != hipSuccess) {
                printf("IPC %d<-%d: GET=FAIL (%s)\n", i, j, hipGetErrorString(e_get));
                HIP_CHECK(hipFree(d_src));
                continue;
            }

            // dev i: open the handle
            HIP_CHECK(hipSetDevice(i));
            unsigned int * d_out = nullptr;
            HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned int)));
            void * d_peer = nullptr;
            hipError_t e_open = hipIpcOpenMemHandle(&d_peer, handle, hipIpcMemLazyEnablePeerAccess);
            if (e_open != hipSuccess) {
                printf("IPC %d<-%d: GET=ok OPEN=FAIL (%s)\n", i, j, hipGetErrorString(e_open));
            } else {
                // device read through the opened pointer
                hipLaunchKernelGGL(read_magic_kernel, dim3(1), dim3(1), 0, 0,
                                   (const unsigned int *) d_peer, d_out);
                hipError_t e_k = hipGetLastError();
                hipError_t e_s = (e_k == hipSuccess) ? hipDeviceSynchronize() : e_k;
                unsigned int got = 0;
                HIP_CHECK(hipMemcpy(&got, d_out, sizeof(unsigned int), hipMemcpyDeviceToHost));
                if (e_k == hipSuccess && e_s == hipSuccess && got == magic) {
                    // 1 MiB strided read datapoint
                    unsigned long long * d_acc = nullptr;
                    HIP_CHECK(hipMalloc(&d_acc, sizeof(unsigned long long)));
                    HIP_CHECK(hipMemset(d_acc, 0, sizeof(unsigned long long)));
                    hipLaunchKernelGGL(read_sum_kernel, dim3(1), dim3(256), 0, 0,
                                       (const unsigned int *) d_peer, d_acc,
                                       (int) (sum_bytes / sizeof(unsigned int)));
                    hipError_t e_k2 = hipGetLastError();
                    hipError_t e_s2 = (e_k2 == hipSuccess) ? hipDeviceSynchronize() : e_k2;
                    unsigned long long acc = 0;
                    HIP_CHECK(hipMemcpy(&acc, d_acc, sizeof(unsigned long long), hipMemcpyDeviceToHost));
                    // 0xAB words -> per-word sum 0x2D2D2D2D5 (11 x 0xAB + carry-free check below)
                    const unsigned long long expect = (unsigned long long) magic * (sum_bytes / 4);
                    if (e_k2 == hipSuccess && e_s2 == hipSuccess && acc == expect) {
                        printf("IPC %d<-%d: GET=ok OPEN=ok READ=ok SUM1M=ok -> P2P-via-IPC WORKS\n", i, j);
                        ++n_pass;
                    } else {
                        printf("IPC %d<-%d: GET=ok OPEN=ok READ=ok SUM1M=FAIL (acc=%llu expect=%llu)\n",
                               i, j, acc, expect);
                    }
                    HIP_CHECK(hipFree(d_acc));
                } else {
                    printf("IPC %d<-%d: GET=ok OPEN=ok READ=FAIL (k=%s s=%s got=0x%x)\n",
                           i, j, hipGetErrorString(e_k), hipGetErrorString(e_s), got);
                }
                hipError_t e_close = hipIpcCloseMemHandle(d_peer);
                if (e_close != hipSuccess) {
                    printf("IPC %d<-%d: CLOSE=FAIL (%s)\n", i, j, hipGetErrorString(e_close));
                }
            }
            HIP_CHECK(hipFree(d_out));

            // copy-engine question, independent of the IPC open result
            unsigned int * d_ce = nullptr;
            HIP_CHECK(hipMalloc(&d_ce, 4096));
            hipError_t e_mc = hipMemcpyPeerAsync(d_ce, i, d_src, j, 4096, 0);
            hipError_t e_mcs = (e_mc == hipSuccess) ? hipDeviceSynchronize() : e_mc;
            unsigned int got_ce = 0;
            if (e_mc == hipSuccess && e_mcs == hipSuccess) {
                HIP_CHECK(hipMemcpy(&got_ce, d_ce, sizeof(unsigned int), hipMemcpyDeviceToHost));
            }
            printf("MEMCPY %d<-%d: %s (byte=0x%x)\n", i, j,
                   (e_mc == hipSuccess && e_mcs == hipSuccess && got_ce == magic) ? "ok" : "FAIL",
                   got_ce);
            if (e_mc == hipSuccess && e_mcs == hipSuccess && got_ce == magic) {
                ++memcpy_pass;
            }
            HIP_CHECK(hipFree(d_ce));
            HIP_CHECK(hipSetDevice(j));
            HIP_CHECK(hipFree(d_src));
        }
    }

    if (g_alarm) {
        printf("PROBE-EXIT:3 (watchdog %d s fired mid-matrix)\n", timeout_s);
        return 3;
    }

    printf("IPC verdict: %d/%d pairs OPEN+READ ok, %d/%d pairs memcpyPeer ok\n",
           n_pass, n_pairs, memcpy_pass, n_pairs);
    if (n_pass > 0) {
        printf("IPC verdict: P2P-via-IPC AVAILABLE -> one-shot AR arm revives (go to bench stage)\n");
        return 0;
    }
    printf("IPC verdict: P2P-via-IPC NOT available -> one-shot AR stays DEAD on this stack\n");
    return 1;
}
