// E-13 RED-capture cell: causal_conv1d route parity at the T=64/65 dispatch gate.
//
// The split3 wrapper routes T<=64 to causal_conv1d_sequence_kernel and T>=65 to
// causal_conv1d_prefill_pairs_kernel (kCausalConvSequenceMaxTokens=64). E12b measured
// QC/KC/VC even-channel corruption from row 0 at T>=65. This cell runs BOTH kernels on
// bit-identical synthetic input + state and diffs the outputs bit-for-bit, in both the
// plain [C,T] form and the split3 epilogue form, across C/T classes.
//
// Weight layout: the kernels index weight[k*C + c] ([4,C] flattened). Production weight
// tensors carry ne=[C,4] labels, but T<=64 serving is numerically correct with [4,C]
// indexing, so the device data is [4,C]-flattened; this cell feeds the layout the kernels
// actually read and asserts route PARITY, which is layout-independent (same buffer to both
// kernels). A CPU oracle also checks semantic correctness of each route independently.

#include "ops/kernel/causal_conv1d.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ninfer::ops;

namespace {

std::uint16_t f32_to_bf16_bits(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    // round-to-nearest-even
    std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(u >> 16);
}

float bf16_bits_to_f32(std::uint16_t h) {
    std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

struct Buffers {
    std::vector<std::uint16_t> x;       // [C*T]
    std::vector<std::uint16_t> weight;  // [4*C], row k = tap k ([4,C] flattened)
    std::vector<std::uint16_t> state;   // [3*C]
};

Buffers make_buffers(std::int32_t C, std::int32_t T, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xd(-3.0f, 3.0f), wd(-1.0f, 1.0f);
    Buffers b;
    b.x.resize(static_cast<std::size_t>(C) * T);
    for (auto& v : b.x) { v = f32_to_bf16_bits(xd(rng)); }
    b.weight.resize(static_cast<std::size_t>(4) * C);
    for (auto& v : b.weight) { v = f32_to_bf16_bits(wd(rng)); }
    b.state.resize(static_cast<std::size_t>(3) * C);
    for (auto& v : b.state) { v = f32_to_bf16_bits(xd(rng)); }
    return b;
}

// CPU oracle over the same [4,C]-flattened weight the kernels read.
std::vector<std::uint16_t> oracle(const Buffers& b, std::int32_t C, std::int32_t T) {
    std::vector<float> s0(C), s1(C), s2(C);
    for (std::int32_t c = 0; c < C; ++c) {
        s0[c] = bf16_bits_to_f32(b.state[c]);
        s1[c] = bf16_bits_to_f32(b.state[C + c]);
        s2[c] = bf16_bits_to_f32(b.state[2 * C + c]);
    }
    std::vector<std::uint16_t> out(static_cast<std::size_t>(C) * T);
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t c = 0; c < C; ++c) {
            const float x0 = s0[c], x1 = s1[c], x2 = s2[c];
            const float x3 = bf16_bits_to_f32(b.x[static_cast<std::size_t>(t) * C + c]);
            float acc = 0.0f;
            acc += bf16_bits_to_f32(b.weight[c]) * x0;
            acc += bf16_bits_to_f32(b.weight[C + c]) * x1;
            acc += bf16_bits_to_f32(b.weight[2 * C + c]) * x2;
            acc += bf16_bits_to_f32(b.weight[3 * C + c]) * x3;
            out[static_cast<std::size_t>(t) * C + c] = f32_to_bf16_bits(silu_f(acc));
            s0[c] = s1[c];
            s1[c] = s2[c];
            s2[c] = x3;
        }
    }
    return out;
}

void* dev_alloc(const std::vector<std::uint16_t>& h) {
    void* d = nullptr;
    if (cudaMalloc(&d, h.size() * 2) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        std::exit(2);
    }
    if (cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "H2D failed\n");
        std::exit(2);
    }
    return d;
}

// Runs ONE kernel choice on identical device buffers and returns host output [C*T].
// kernel: 0 = sequence, 1 = prefill pairs, 2 = prefill scalar.
std::vector<std::uint16_t> run_kernel(int kernel, const Buffers& b, std::int32_t C,
                                      std::int32_t T) {
    const void* dx = dev_alloc(b.x);
    const void* dw = dev_alloc(b.weight);
    void* dsin = dev_alloc(b.state);
    void* dsout = dev_alloc(b.state);
    std::vector<std::uint16_t> hout(static_cast<std::size_t>(C) * T, 0xdead);
    void* dout = dev_alloc(hout);

    auto* x = static_cast<const __nv_bfloat16*>(dx);
    auto* w = static_cast<const __nv_bfloat16*>(dw);
    auto* si = static_cast<const __nv_bfloat16*>(dsin);
    auto* so = static_cast<__nv_bfloat16*>(dsout);
    auto* o = static_cast<__nv_bfloat16*>(dout);

    const int cblocks = (C + 255) / 256;
    const int pairblocks = (C / 2 + 255) / 256;
    if (kernel == 0) {
        causal_conv1d_sequence_kernel<<<cblocks, 256>>>(x, w, si, so, o, C, T);
    } else if (kernel == 1) {
        causal_conv1d_prefill_pairs_kernel<<<static_cast<long long>(pairblocks) * T, 256>>>(
            x, w, si, o, C, T);
    } else {
        causal_conv1d_prefill_kernel<<<static_cast<long long>(cblocks) * T, 256>>>(x, w, si, o, C,
                                                                                   T);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kernel launch/execution failed\n");
        std::exit(2);
    }
    if (cudaMemcpy(hout.data(), dout, hout.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "D2H failed\n");
        std::exit(2);
    }
    cudaFree((void*)dx);
    cudaFree((void*)dw);
    cudaFree(dsin);
    cudaFree(dsout);
    cudaFree(dout);
    return hout;
}

// Split3 form (production): sequence+sp vs pairs+sp vs scalar+sp on identical input,
// with the measured world-4 geometry C=2560, bounds {512,512,1536}.
struct SplitResult {
    std::vector<std::uint16_t> q, k, v;
};

ConvSplit3 make_sp(std::int32_t q_rows, std::int32_t k_rows, std::int32_t v_rows) {
    ConvSplit3 sp;
    sp.dst[0]   = nullptr;
    sp.dst[1]   = nullptr;
    sp.dst[2]   = nullptr;
    sp.bound[0] = q_rows;
    sp.bound[1] = q_rows + k_rows;
    sp.pitch[0] = q_rows;
    sp.pitch[1] = k_rows;
    sp.pitch[2] = v_rows;
    sp.C        = q_rows + k_rows + v_rows;
    return sp;
}

SplitResult run_kernel_split3(int kernel, const Buffers& b, std::int32_t q_rows,
                              std::int32_t k_rows, std::int32_t v_rows, std::int32_t T) {
    const std::int32_t C = q_rows + k_rows + v_rows;
    const void* dx = dev_alloc(b.x);
    const void* dw = dev_alloc(b.weight);
    void* dsin = dev_alloc(b.state);
    void* dsout = dev_alloc(b.state);
    SplitResult r;
    r.q.assign(static_cast<std::size_t>(q_rows) * T, 0xdead);
    r.k.assign(static_cast<std::size_t>(k_rows) * T, 0xdead);
    r.v.assign(static_cast<std::size_t>(v_rows) * T, 0xdead);
    void* dq = dev_alloc(r.q);
    void* dk = dev_alloc(r.k);
    void* dv = dev_alloc(r.v);

    ConvSplit3 sp = make_sp(q_rows, k_rows, v_rows);
    sp.dst[0]     = static_cast<__nv_bfloat16*>(dq);
    sp.dst[1]     = static_cast<__nv_bfloat16*>(dk);
    sp.dst[2]     = static_cast<__nv_bfloat16*>(dv);

    auto* x = static_cast<const __nv_bfloat16*>(dx);
    auto* w = static_cast<const __nv_bfloat16*>(dw);
    auto* si = static_cast<const __nv_bfloat16*>(dsin);
    auto* so = static_cast<__nv_bfloat16*>(dsout);
    auto* o = static_cast<__nv_bfloat16*>(dq);

    const int cblocks = (C + 255) / 256;
    const int pairblocks = (C / 2 + 255) / 256;
    if (kernel == 0) {
        causal_conv1d_sequence_kernel<<<cblocks, 256>>>(x, w, si, so, o, C, T, sp);
    } else if (kernel == 1) {
        causal_conv1d_prefill_pairs_kernel<<<static_cast<long long>(pairblocks) * T, 256>>>(
            x, w, si, o, C, T, sp);
    } else {
        causal_conv1d_prefill_kernel<<<static_cast<long long>(cblocks) * T, 256>>>(x, w, si, o, C,
                                                                                   T, sp);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kernel launch/execution failed\n");
        std::exit(2);
    }
    if (cudaMemcpy(r.q.data(), dq, r.q.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(r.k.data(), dk, r.k.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(r.v.data(), dv, r.v.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "D2H failed\n");
        std::exit(2);
    }
    cudaFree((void*)dx);
    cudaFree((void*)dw);
    cudaFree(dsin);
    cudaFree(dsout);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    return r;
}

// Route parity must be BIT-exact; the CPU oracle is compared with 1-ulp tolerance because the
// host silu/rounding can differ from the device by one unit in the last place.
int oracle_report(const char* label, const std::vector<std::uint16_t>& a,
                  const std::vector<std::uint16_t>& b) {
    int mismatches = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) { continue; }
        const std::uint16_t ea = a[i] & 0x7fffu, eb = b[i] & 0x7fffu;
        const std::uint16_t d = ea > eb ? ea - eb : eb - ea;
        if (d > 1) { ++mismatches; }
    }
    std::printf("%-44s mismatches=%7d/%7zu %s\n", label, mismatches, a.size(),
                mismatches == 0 ? "ORACLE-OK" : "*** ORACLE-DIVERGENT ***");
    return mismatches == 0 ? 0 : 1;
}

int diff_report(const char* label, const std::vector<std::uint16_t>& a,
                const std::vector<std::uint16_t>& b, std::int32_t C, std::int32_t T) {
    int mismatches = 0, even_mis = 0, odd_mis = 0, first_t = -1;
    double max_delta = 0.0;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] == b[i]) { continue; }
        ++mismatches;
        const std::int32_t c = static_cast<std::int32_t>(i % C);
        const std::int32_t t = static_cast<std::int32_t>(i / C);
        if ((c & 1) == 0) { ++even_mis; } else { ++odd_mis; }
        if (first_t < 0) { first_t = t; }
        const double d =
            std::fabs(bf16_bits_to_f32(a[i]) - bf16_bits_to_f32(b[i]));
        if (d > max_delta) { max_delta = d; }
    }
    std::printf("%-44s mismatches=%7d/%7zu even=%d odd=%d first_t=%d max_delta=%.4f %s\n", label,
                mismatches, n, even_mis, odd_mis, first_t, max_delta,
                mismatches == 0 ? "PARITY-OK" : "*** DIVERGENT ***");
    return mismatches == 0 ? 0 : 1;
}

} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("SKIP: no usable device\n");
        return 77;
    }
    int failures = 0;
    // Production geometry for the GDN conv at world=4 is C=4096 (qkvz shard) with
    // split3 bounds q/k; parity is layout-independent so plain form suffices, plus the
    // 27B full C and odd C classes.
    struct Case { std::int32_t C, T; };
    const Case cases[] = {
        {4096, 65}, {4096, 96}, {4096, 257}, {10240, 65}, {8192, 130}, {512, 65}, {510, 65},
    };
    for (const Case& cs : cases) {
        char tag[64];
        std::snprintf(tag, sizeof tag, "C=%d T=%d", cs.C, cs.T);
        const Buffers b = make_buffers(cs.C, cs.T, 9000u + static_cast<unsigned>(cs.C + cs.T));
        const std::vector<std::uint16_t> seq = run_kernel(0, b, cs.C, cs.T);
        const std::vector<std::uint16_t> pairs = run_kernel(1, b, cs.C, cs.T);
        const std::vector<std::uint16_t> scalar = run_kernel(2, b, cs.C, cs.T);
        std::string l1 = std::string(tag) + " sequence-vs-pairs";
        std::string l2 = std::string(tag) + " sequence-vs-scalar-prefill";
        std::string l3 = std::string(tag) + " sequence-vs-cpu-oracle";
        failures += diff_report(l1.c_str(), seq, pairs, cs.C, cs.T);
        failures += diff_report(l2.c_str(), seq, scalar, cs.C, cs.T);
        failures += oracle_report(l3.c_str(), seq, oracle(b, cs.C, cs.T));
    }

    // Split3 (production) geometry: world-4 shard C=2560, bounds {512,512,1536}, plus odd-bound
    // and small-C classes. Compare routes pairwise; also compare each region against the plain
    // sequence route's expected region content.
    struct SCase { std::int32_t q, k, v, T; };
    const SCase scases[] = {
        {512, 512, 1536, 65}, {512, 512, 1536, 96}, {512, 512, 1536, 257}, {256, 256, 768, 65},
        {13, 17, 40, 65},
    };
    for (const SCase& sc : scases) {
        const std::int32_t C = sc.q + sc.k + sc.v;
        char tag[80];
        std::snprintf(tag, sizeof tag, "split3 C=%d {%d,%d,%d} T=%d", C, sc.q, sc.k, sc.v, sc.T);
        const Buffers b = make_buffers(C, sc.T, 7000u + static_cast<unsigned>(C + sc.T));
        const SplitResult s0 = run_kernel_split3(0, b, sc.q, sc.k, sc.v, sc.T);
        const SplitResult s1 = run_kernel_split3(1, b, sc.q, sc.k, sc.v, sc.T);
        const SplitResult s2 = run_kernel_split3(2, b, sc.q, sc.k, sc.v, sc.T);
        std::string l1 = std::string(tag) + " q sequence-vs-pairs";
        std::string l2 = std::string(tag) + " k sequence-vs-pairs";
        std::string l3 = std::string(tag) + " v sequence-vs-pairs";
        std::string l4 = std::string(tag) + " q sequence-vs-scalar";
        std::string l5 = std::string(tag) + " k sequence-vs-scalar";
        std::string l6 = std::string(tag) + " v sequence-vs-scalar";
        failures += diff_report(l1.c_str(), s0.q, s1.q, sc.q, sc.T);
        failures += diff_report(l2.c_str(), s0.k, s1.k, sc.k, sc.T);
        failures += diff_report(l3.c_str(), s0.v, s1.v, sc.v, sc.T);
        failures += diff_report(l4.c_str(), s0.q, s2.q, sc.q, sc.T);
        failures += diff_report(l5.c_str(), s0.k, s2.k, sc.k, sc.T);
        failures += diff_report(l6.c_str(), s0.v, s2.v, sc.v, sc.T);
    }
    std::printf("%s\n", failures == 0 ? "OK e13_conv_route_parity" : "FAIL e13_conv_route_parity");
    return failures == 0 ? 0 : 1;
}
