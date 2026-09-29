// w7_kvhosted_cell.cu — W7 hosted-KV desk (docs/amd/WO_KV_HOSTED_desk.md §3): the mechanism
// decision data. Standalone hipcc cell, ZERO production code touched.
//
// Question: can a kvarn-shaped KV slab live in mapped-pinned HOST memory and be read by an
// attention-shaped kernel with (1) bit-parity vs the same slab in device memory and (2) enough
// sustained PCIe read bandwidth to make prefill-centric hosting viable?
//
// Slab model mirrors the qwen3_6_27b KVarN k4v2 pool EXACTLY in bytes per (layer,head,page)
// tile (decoder_state.cpp plan_cache): K codes U8 64tok x 256ch x 4b = 8,192 B (channel-major),
// V codes U8 64tok x 256ch x 2b = 4,096 B (token-major), kvarn scales fp32 1,152 fields =
// 4,608 B (K s_col256+zp256+s_row64 + V same). Tile = 16,896 B. A full-sweep kernel pass over
// every tile = ONE decode token's KV read = the LAST prefill chunk's cold read = the decode-tax
// unit priced in the design doc §2.
//
// Arms: device slab (hipMalloc) vs mapped-pinned host slab (hipHostMallocMapped +
// hipHostGetDevicePointer). Same deterministic patterned bytes (LCG). Parity = kernel outputs
// word-identical + raw slab memcmp. Perf = full-sweep GB/s at 10/50/100/200 MiB.
// Bonus probe: pinned H2D async copy concurrent with a compute kernel (the overlap premise
// both mechanisms lean on), timed alone and together.
//
// Bar (pre-registered in WO_KV_HOSTED_desk.md §3 BEFORE first run):
//   parity bit-exact at every size (mandatory) AND hosted full-sweep BW >= 6.0 GB/s @200 MiB.
//
// Run: HIP_VISIBLE_DEVICES=2 (microbench law) ./w7_kvhosted_cell
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

namespace {

constexpr std::size_t kKBytes   = 8192;  // K codes U8: 64 tok x 256 ch x 4 bits
constexpr std::size_t kVBytes   = 4096;  // V codes U8: 64 tok x 256 ch x 2 bits
constexpr std::size_t kScales   = 1152;  // fp32 fields per (layer,head,page) tile
constexpr std::size_t kScaleBytes = kScales * sizeof(float);
constexpr std::size_t kTileBytes  = kKBytes + kVBytes + kScaleBytes;  // 16,896
constexpr std::size_t kPerRankBytesPerToken = kTileBytes * 16 / 64;   // 16 layers / 64 tok = 4,224
constexpr int kThreads = 256;

__global__ void kv_sweep_kernel(const std::uint8_t* blob, std::uint32_t* out,
                                unsigned long long tiles) {
    const unsigned long long t = blockIdx.x;
    if (t >= tiles) { return; }
    const std::uint8_t* base = blob + t * kTileBytes;
    std::uint32_t acc = 0x9e3779b9u ^ static_cast<std::uint32_t>(t);
    // K region: warp-coalesced byte stream, position-weighted mix (every byte read).
    for (std::size_t i = threadIdx.x; i < kKBytes; i += blockDim.x) {
        acc = acc * 1664525u + 1013904223u + static_cast<std::uint32_t>(base[i]) *
              static_cast<std::uint32_t>(i + 1u);
    }
    // V region.
    const std::uint8_t* vbase = base + kKBytes;
    for (std::size_t i = threadIdx.x; i < kVBytes; i += blockDim.x) {
        acc ^= (acc >> 15) + static_cast<std::uint32_t>(vbase[i]) *
               static_cast<std::uint32_t>(i + 7u);
    }
    // Scale region: fp32 stream, bit-mixed (dequant-shaped reads).
    const float* s = reinterpret_cast<const float*>(base + kKBytes + kVBytes);
    for (std::size_t i = threadIdx.x; i < kScales; i += blockDim.x) {
        const float f = s[i];
        std::uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        acc = (acc ^ bits) * 2654435761u + static_cast<std::uint32_t>(i);
    }
    // Cross-thread reduce so no lane's stream can be dead-code eliminated.
    __shared__ std::uint32_t scratch[kThreads];
    scratch[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] ^= scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[t] = scratch[0]; }
}

// Compute-shaped dummy for the overlap probe: independent FMA loops across many blocks so the
// compute arm is a real concurrent load (a 1-block latency chain cannot saturate anything and
// proves nothing about copy/compute concurrency).
__global__ void burn_kernel(float* sink, int iters) {
    float a = static_cast<float>(threadIdx.x + blockIdx.x * 37u) * 1.0e-6f + 1.0f;
    float b = 0.999f;
    for (int i = 0; i < iters; ++i) {
        a = fmaf(a, b, 1.0e-7f);
        b = fmaf(b, a, 1.0e-8f);
    }
    if (a == 42.0f) { sink[threadIdx.x] = a; }  // never true; defeats DCE
}

// Vector-load sweep: what real kvarn kernels do (packed 32/128-bit code-word loads, float4
// scale loads), not byte loads. K region = 512 uint4, V region = 256 uint4, scales = 288 float4.
__global__ void kv_sweep_vec_kernel(const std::uint8_t* blob, std::uint32_t* out,
                                    unsigned long long tiles) {
    const unsigned long long t = blockIdx.x;
    if (t >= tiles) { return; }
    const std::uint8_t* base = blob + t * kTileBytes;
    std::uint32_t acc = 0x9e3779b9u ^ static_cast<std::uint32_t>(t);
    const uint4* kvec = reinterpret_cast<const uint4*>(base);
    for (std::size_t i = threadIdx.x; i < kKBytes / 16; i += blockDim.x) {
        const uint4 q = kvec[i];
        acc = acc * 1664525u + 1013904223u + (q.x ^ q.y) + (q.z ^ q.w) *
              static_cast<std::uint32_t>(i + 1u);
    }
    const uint4* vvec = reinterpret_cast<const uint4*>(base + kKBytes);
    for (std::size_t i = threadIdx.x; i < kVBytes / 16; i += blockDim.x) {
        const uint4 q = vvec[i];
        acc ^= (acc >> 15) + (q.x ^ q.y ^ q.z ^ q.w) * static_cast<std::uint32_t>(i + 7u);
    }
    const float4* s = reinterpret_cast<const float4*>(base + kKBytes + kVBytes);
    for (std::size_t i = threadIdx.x; i < kScales / 4; i += blockDim.x) {
        const float4 f = s[i];
        std::uint32_t b0, b1, b2, b3;
        std::memcpy(&b0, &f.x, 4); std::memcpy(&b1, &f.y, 4);
        std::memcpy(&b2, &f.z, 4); std::memcpy(&b3, &f.w, 4);
        acc = (acc ^ (b0 ^ b1 ^ b2 ^ b3)) * 2654435761u + static_cast<std::uint32_t>(i);
    }
    __shared__ std::uint32_t scratch[kThreads];
    scratch[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] ^= scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[t] = scratch[0]; }
}

void fill_pattern(std::uint8_t* host, unsigned long long tiles) {
    for (unsigned long long t = 0; t < tiles; ++t) {
        std::uint8_t* base = host + t * kTileBytes;
        std::uint32_t x = static_cast<std::uint32_t>(t) * 2654435761u + 0x1234567u;
        for (std::size_t i = 0; i < kTileBytes; ++i) {
            x = x * 1664525u + 1013904223u;
            base[i] = static_cast<std::uint8_t>(x >> 24);
        }
    }
}

double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

#define HIP_CHECK(expr)                                                              \
    do {                                                                             \
        const hipError_t _e = (expr);                                                \
        if (_e != hipSuccess) {                                                      \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),    \
                         __FILE__, __LINE__);                                        \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

struct SweepResult {
    double host_gbps  = 0;
    double dev_gbps   = 0;
    double vhost_gbps = 0;  // vector-load arm
    double vdev_gbps  = 0;
    int    parity     = 0;  // 1 = word-identical kernel outputs AND raw slab memcmp
    int    vparity    = 0;  // vector-arm kernel outputs, hosted vs device
    int    memcmp_eq  = 0;
};

SweepResult run_size(std::uint8_t* host_blob, std::uint8_t* host_devptr, unsigned long long tiles,
                     int iters, std::uint8_t* dev_blob, std::uint32_t* out_host,
                     std::uint32_t* out_devres, std::uint32_t* out_hostres) {
    SweepResult r;
    const std::size_t bytes = tiles * kTileBytes;

    // Fill device slab from the patterned host blob (identical bytes both arms).
    HIP_CHECK(hipMemcpy(dev_blob, host_blob, bytes, hipMemcpyHostToDevice));

    const dim3 grid(static_cast<unsigned>(tiles));
    const dim3 block(kThreads);
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    // Warmup both arms (untimed).
    kv_sweep_kernel<<<grid, block>>>(dev_blob, out_devres, tiles);
    kv_sweep_kernel<<<grid, block>>>(host_devptr, out_hostres, tiles);  // host-mapped arm
    HIP_CHECK(hipDeviceSynchronize());

    // PARITY: same kernel, device-slab pointer vs host-mapped pointer, patterned data.
    HIP_CHECK(hipMemset(out_devres, 0, tiles * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemset(out_hostres, 0, tiles * sizeof(std::uint32_t)));
    kv_sweep_kernel<<<grid, block>>>(dev_blob, out_devres, tiles);
    kv_sweep_kernel<<<grid, block>>>(host_devptr, out_hostres, tiles);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(out_host, out_devres, tiles * sizeof(std::uint32_t),
                        hipMemcpyDeviceToHost));
    r.parity = std::memcmp(out_host, out_hostres, tiles * sizeof(std::uint32_t)) == 0 ? 1 : 0;

    // Raw slab memcmp: D2H the device slab, byte-compare vs host pattern.
    std::uint8_t* shadow = static_cast<std::uint8_t*>(std::malloc(bytes));
    HIP_CHECK(hipMemcpy(shadow, dev_blob, bytes, hipMemcpyDeviceToHost));
    r.memcmp_eq = std::memcmp(shadow, host_blob, bytes) == 0 ? 1 : 0;
    std::free(shadow);

    // PERF: device arm, iters full sweeps (byte kernel, then vector kernel).
    float ms = 0;
    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
        kv_sweep_kernel<<<grid, block>>>(dev_blob, out_devres, tiles);
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    r.dev_gbps = static_cast<double>(bytes) * iters / (static_cast<double>(ms) * 1e6);

    // PERF: hosted arm, same iters (byte kernel).
    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
        kv_sweep_kernel<<<grid, block>>>(host_devptr, out_hostres, tiles);
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    r.host_gbps = static_cast<double>(bytes) * iters / (static_cast<double>(ms) * 1e6);

    // Vector arm: parity then perf, hosted vs device.
    HIP_CHECK(hipMemset(out_devres, 0, tiles * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemset(out_hostres, 0, tiles * sizeof(std::uint32_t)));
    kv_sweep_vec_kernel<<<grid, block>>>(dev_blob, out_devres, tiles);
    kv_sweep_vec_kernel<<<grid, block>>>(host_devptr, out_hostres, tiles);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(out_host, out_devres, tiles * sizeof(std::uint32_t),
                        hipMemcpyDeviceToHost));
    r.vparity = std::memcmp(out_host, out_hostres, tiles * sizeof(std::uint32_t)) == 0 ? 1 : 0;

    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
        kv_sweep_vec_kernel<<<grid, block>>>(dev_blob, out_devres, tiles);
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    r.vdev_gbps = static_cast<double>(bytes) * iters / (static_cast<double>(ms) * 1e6);

    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
        kv_sweep_vec_kernel<<<grid, block>>>(host_devptr, out_hostres, tiles);
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    r.vhost_gbps = static_cast<double>(bytes) * iters / (static_cast<double>(ms) * 1e6);

    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
    return r;
}

} // namespace

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s gcn=%s totalMem=%.1f GiB\n", prop.name, prop.gcnArchName,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    std::printf("tile=%zu B (K %zu + V %zu + scales %zu); per-rank bytes/token @TP4 = %zu\n",
                kTileBytes, kKBytes, kVBytes, kScaleBytes, kPerRankBytesPerToken);

    constexpr int kIters = 20;
    const std::size_t kMaxBytes = std::size_t{200} << 20;  // 200 MiB
    const unsigned long long kMaxTiles = kMaxBytes / kTileBytes;

    // Host slab (mapped-pinned) — sized once at max.
    std::uint8_t* host_blob = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_blob), kMaxTiles * kTileBytes,
                            hipHostMallocMapped));
    fill_pattern(host_blob, kMaxTiles);
    std::uint8_t* host_devptr = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&host_devptr), host_blob, 0));
    std::printf("host blob mapped: host=%p devptr=%p (%s)\n",
                static_cast<void*>(host_blob), static_cast<void*>(host_devptr),
                host_devptr == host_blob ? "unified pointer" : "separate devptr");

    std::uint8_t* dev_blob = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_blob), kMaxTiles * kTileBytes));
    std::uint32_t* out_devres = nullptr;
    std::uint32_t* out_hostres = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&out_devres), kMaxTiles * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&out_hostres), kMaxTiles * sizeof(std::uint32_t)));
    std::vector<std::uint32_t> out_host(kMaxTiles);

    std::printf("%-10s %10s %12s %12s %12s %12s %12s %7s %8s\n", "size", "tokens", "dev_GB/s",
                "host_GB/s", "vdev_GB/s", "vhost_GB/s", "tok_equiv", "parity", "memcmp");
    int all_parity = 1;
    double host200 = 0;
    const std::size_t sizes_mib[] = {10, 50, 100, 200};
    for (const std::size_t mib : sizes_mib) {
        const std::size_t bytes = mib << 20;
        const unsigned long long tiles = bytes / kTileBytes;
        const SweepResult r = run_size(host_blob, host_devptr, tiles, kIters, dev_blob,
                                       out_host.data(), out_devres, out_hostres);
        all_parity &= r.parity & r.memcmp_eq & r.vparity;
        if (mib == 200) { host200 = r.vhost_gbps; }
        std::printf("%-10zu %10llu %12.2f %12.2f %12.2f %12.2f %7llu %8d %8d\n", mib,
                    tiles * 64ull, r.dev_gbps, r.host_gbps, r.vdev_gbps, r.vhost_gbps,
                    tiles * kTileBytes / kPerRankBytesPerToken, r.parity & r.vparity,
                    r.memcmp_eq);
        std::fflush(stdout);
    }

    // ---- Overlap probe: pinned H2D async copy concurrent with a compute kernel ----
    const std::size_t kChunkBytes = std::size_t{2048} * kPerRankBytesPerToken;  // 2k tok/rank
    std::uint8_t* stage = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&stage), kChunkBytes,
                            hipHostMallocMapped));
    std::memset(stage, 0xAB, kChunkBytes);
    std::uint8_t* stage_dev = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&stage_dev), kChunkBytes));
    float* sink = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sink), kThreads * sizeof(float)));
    hipStream_t s_copy, s_comp;
    HIP_CHECK(hipStreamCreate(&s_copy));
    HIP_CHECK(hipStreamCreate(&s_comp));
    hipEvent_t c0, c1, k0, k1, b0, b1;
    HIP_CHECK(hipEventCreate(&c0)); HIP_CHECK(hipEventCreate(&c1));
    HIP_CHECK(hipEventCreate(&k0)); HIP_CHECK(hipEventCreate(&k1));
    HIP_CHECK(hipEventCreate(&b0)); HIP_CHECK(hipEventCreate(&b1));

    // Copy alone.
    HIP_CHECK(hipEventRecord(c0, s_copy));
    HIP_CHECK(hipMemcpyAsync(stage_dev, stage, kChunkBytes, hipMemcpyHostToDevice, s_copy));
    HIP_CHECK(hipEventRecord(c1, s_copy));
    HIP_CHECK(hipEventSynchronize(c1));
    float copy_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&copy_ms, c0, c1));

    // Compute alone (iters tuned so 30 launches are a real multi-block compute load).
    constexpr int kBurnIters = 20000;
    constexpr int kBurnLaunches = 30;
    constexpr unsigned kBurnBlocks = 108;  // saturating: gfx900 64 CU x ~2 resident
    HIP_CHECK(hipEventRecord(k0, s_comp));
    for (int i = 0; i < kBurnLaunches; ++i) {
        burn_kernel<<<kBurnBlocks, kThreads>>>(sink, kBurnIters);
    }
    HIP_CHECK(hipEventRecord(k1, s_comp));
    HIP_CHECK(hipEventSynchronize(k1));
    float comp_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&comp_ms, k0, k1));

    // Together: copy stream vs compute stream, concurrent.
    HIP_CHECK(hipEventRecord(b0));
    HIP_CHECK(hipMemcpyAsync(stage_dev, stage, kChunkBytes, hipMemcpyHostToDevice, s_copy));
    for (int i = 0; i < kBurnLaunches; ++i) {
        burn_kernel<<<kBurnBlocks, kThreads>>>(sink, kBurnIters);
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(b1));
    HIP_CHECK(hipEventSynchronize(b1));
    float both_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&both_ms, b0, b1));

    std::printf("overlap: 2k-chunk pinned H2D copy %.3f ms | compute %dx burn %.3f ms | "
                "both concurrent %.3f ms (sum would be %.3f) -> %s\n",
                copy_ms, kBurnLaunches, comp_ms, both_ms, copy_ms + comp_ms,
                both_ms < 0.75 * (copy_ms + comp_ms) ? "OVERLAP CONFIRMED" : "NO CLEAR OVERLAP");
    std::printf("decode-tax unit @ measured host BW: full sweep of 47.3k tok (200 MiB/rank) "
                "costs %.2f ms per generated token\n",
                static_cast<double>(200 << 20) / (host200 * 1e9) * 1e3);

    const int verdict = (all_parity == 1 && host200 >= 6.0) ? 1 : 0;
    std::printf("VERDICT: parity_all=%d host200_GBs=%.2f bar(>=6.0) -> %s\n",
                all_parity, host200, verdict ? "GO" : "NO GO");

    HIP_CHECK(hipHostFree(host_blob));
    HIP_CHECK(hipHostFree(stage));
    HIP_CHECK(hipFree(dev_blob));
    HIP_CHECK(hipFree(stage_dev));
    HIP_CHECK(hipFree(out_devres));
    HIP_CHECK(hipFree(out_hostres));
    HIP_CHECK(hipFree(sink));
    return verdict == 1 ? 0 : 2;
}
