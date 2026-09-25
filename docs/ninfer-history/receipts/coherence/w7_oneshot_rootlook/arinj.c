// arinj.c — WO_ONESHOT_ROOTLOOK LD_PRELOAD probe: interpose the dynamic NCCL/HIP
// host calls the tp2 engine + tp_group make during warmup, logging ENTER/EXIT per
// call with tid so the serve log itself names which call never returns.
// Build: gcc -O2 -shared -fPIC -o arinj.so arinj.c -ldl -lpthread
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>

static pthread_mutex_t L = PTHREAD_MUTEX_INITIALIZER;
static __thread int in_hook = 0;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
#define LOG(...) do { \
    if (in_hook) break; \
    in_hook = 1; \
    pthread_mutex_lock(&L); \
    fprintf(stderr, "[ARINJ] t=%.4f tid=%lu ", now_s(), (unsigned long)pthread_self()); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    fflush(stderr); \
    pthread_mutex_unlock(&L); \
    in_hook = 0; \
} while (0)

#define WRAP(ret_t, name, proto, argnames) \
    static ret_t (*real_##name) proto; \
    ret_t name proto { \
        if (!real_##name) real_##name = dlsym(RTLD_NEXT, #name); \
        LOG("ENTER " #name); \
        ret_t rc = real_##name argnames; \
        LOG("EXIT  " #name " rc=%d", (int)rc); \
        return rc; \
    }

// NCCL host calls (the ring branch + comm init)
WRAP(int, ncclAllReduce, (const void* a, void* b, size_t c, int d, int e, void* f, void* g), (a,b,c,d,e,f,g))
WRAP(int, ncclAllGather, (const void* a, void* b, size_t c, int d, void* e, void* f), (a,b,c,d,e,f))
WRAP(int, ncclReduce, (const void* a, void* b, size_t c, int d, int e, int f, void* g, void* h), (a,b,c,d,e,f,g,h))
WRAP(int, ncclBroadcast, (const void* a, void* b, size_t c, int d, int e, void* f, void* g), (a,b,c,d,e,f,g))
WRAP(int, ncclSend, (const void* a, size_t b, int c, int d, void* e, void* f), (a,b,c,d,e,f))
WRAP(int, ncclRecv, (void* a, size_t b, int c, int d, void* e, void* f), (a,b,c,d,e,f))
WRAP(int, ncclCommInitRank, (void** a, int b, void* c, int d), (a,b,c,d))
WRAP(int, ncclGroupStart, (void), ())
WRAP(int, ncclGroupEnd, (void), ())

// HIP host sync/alloc calls that can legitimately block
WRAP(int, hipStreamSynchronize, (void* a), (a))
WRAP(int, hipDeviceSynchronize, (void), ())
WRAP(int, hipEventSynchronize, (void* a), (a))
WRAP(int, hipStreamWaitEvent, (void* a, void* b, unsigned c), (a,b,c))
WRAP(int, hipMemcpy, (void* a, const void* b, size_t c, int d), (a,b,c,d))
WRAP(int, hipMemcpyAsync, (void* a, const void* b, size_t c, int d, void* e), (a,b,c,d,e))
WRAP(int, hipMalloc, (void** a, size_t b), (a,b))
WRAP(int, hipHostMalloc, (void** a, size_t b, unsigned c), (a,b,c))
WRAP(int, hipHostAlloc, (void** a, size_t b, unsigned c), (a,b,c))
WRAP(int, hipHostGetDevicePointer, (void** a, void* b, unsigned c), (a,b,c))
WRAP(int, hipModuleLoadData, (void** a, const void* b), (a,b))
WRAP(int, hipDevicePrimaryCtxRetain, (void** a, int b), (a,b))
