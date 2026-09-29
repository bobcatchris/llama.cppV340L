// nvfp4_smallt_bitcheck.cu — SMALLT-TUNE bit-equality cell (permanent, RED/GREEN capable).
//
// Pattern: nvfp4_gemv_bitcheck.cu (GEMV-TUNE's cell), re-keyed to the small_t arm at the TP4 W4
// geometries and T=3 (the proven verify width — SMALLT_row.txt). The tuner may only change HOW
// bytes are fetched, never the arithmetic: nvfp4_amd_codec.h's accumulation order contract
// (per (row,token): lane L groups L,L+32,... ascending; values 0..15 ascending; ONE fp32
// accumulator; descending-halves warp fold) must hold so outputs stay BIT-EQUAL to the
// pre-edit kernel. This tool:
//   1. generates FIXED-SEED (splitmix64) inputs for ALL FIVE TP4 W4 problems, TWO passes each
//      ("fin" = accumulation-order bar with finite x/scales; "raw" = full-range decode coverage),
//      x shaped [K, T=3] (three INDEPENDENT random tokens — a changed fold order on any token
//      column moves real bits),
//   2. drives the PRODUCTION arm launch_nvfp4_small_t (the same dispatcher serve calls),
//   3. dumps every output bit to a binary file + prints per-problem FNV-1a,
//   4. BONUS cross-arm leg (the CT-5 pair property): for each problem, drives
//      launch_nvfp4_decode on token column 0 alone and byte-compares against the small_t
//      T=3 output rows of that token — printed per problem, non-fatal but must read YES
//      on any healthy pair (the header contract: same data => bit-equal between arms).
//   5. usage: nvfp4_smallt_bitcheck <out.bin> [ref.bin]
//        rc 0 = bit-equal (GREEN) / written, rc 1 = mismatch (RED), rc 2 = runtime failure.
//
// Compile EXACTLY like the roofline bench (the -O3 is LAW):
//   clang++ -O3 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 -D__HIP_PLATFORM_AMD__=1
//     -D__HIP_ROCclr__=1 -I$W/src/common/hip_shim -I$W/include -I$W/src -I$W/third_party
//     -DNDEBUG -std=gnu++20 --offload-arch=gfx900 -x hip <this file> $W/src/core/device.cu
//     -o nvfp4_smallt_bitcheck

// The decode arm lives in its own TU (nvfp4_gemv_hip.cu — compiled separately and linked;
// textual inclusion would collide with this file's anonymous-namespace launch_exact):
#include "ops/linear/nvfp4/nvfp4_small_t_hip.cu"

namespace ninfer::ops::detail {
void launch_nvfp4_decode(const Tensor&, const Weight&, Tensor&, cudaStream_t);
}

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

std::uint64_t splitmix64(std::uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

std::uint32_t fnv1a(const unsigned char* p, std::size_t n) {
    std::uint32_t h = 0x811C9DC5u;
    for (std::size_t i = 0; i < n; ++i) { h = (h ^ p[i]) * 0x01000193u; }
    return h;
}

struct Prob { const char* name; int rows, k; };

// The five TP4 W4 problems serve serves at T=3 (nvfp4_config.h geometry table).
const Prob kProbs[] = {
    {"AttnInputW4", 3584, 5120},  {"GdnInputW4", 4096, 5120},  {"MlpGateUpW4", 8704, 5120},
    {"Residual6144W4", 5120, 1536}, {"Residual17408W4", 5120, 4352},
};

constexpr int kT = 3;

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <out.bin> [ref.bin]\n", argv[0]);
        return 2;
    }
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("SKIP: no device\n");
        return 2;
    }

    FILE* out = std::fopen(argv[1], "wb");
    if (!out) {
        std::printf("FAIL: cannot open %s\n", argv[1]);
        return 2;
    }
    int cross_arm_bad = 0;

    for (int pass = 0; pass < 2; ++pass) {  // 0 = fin (order bar), 1 = raw (decode coverage)
        const char* tag = pass == 0 ? "fin" : "raw";
        for (const Prob& p : kProbs) {
            const std::size_t code_bytes = static_cast<std::size_t>(p.rows) * (p.k / 2);
            const std::size_t scale_bytes = static_cast<std::size_t>(p.rows) * (p.k / 16);
            const std::size_t x_bytes = static_cast<std::size_t>(p.k) * 2 * kT;
            const std::size_t out_bytes = static_cast<std::size_t>(p.rows) * 2 * kT;

            std::vector<unsigned char> h_codes(code_bytes), h_scales(scale_bytes);
            std::vector<std::uint16_t> h_x(static_cast<std::size_t>(p.k) * kT);
            std::uint64_t seed = 0xC0FFEE'0000'0000ull ^
                                 (static_cast<std::uint64_t>(pass) << 56) ^
                                 (static_cast<std::uint64_t>(p.rows) << 24) ^
                                 static_cast<std::uint64_t>(p.k);
            for (std::size_t i = 0; i < code_bytes; ++i) {
                h_codes[i] = static_cast<unsigned char>(splitmix64(seed) >> 33);
            }
            for (std::size_t i = 0; i < scale_bytes; ++i) {
                unsigned char b = static_cast<unsigned char>(splitmix64(seed) >> 33);
                if (pass == 0) {
                    if ((b & 0x7Fu) == 0x7Fu) { b &= 0x7Eu; }
                }
                h_scales[i] = b;
            }
            for (std::size_t i = 0; i < h_x.size(); ++i) {
                std::uint16_t b = static_cast<std::uint16_t>(splitmix64(seed) >> 48);
                if (pass == 0) {
                    b = static_cast<std::uint16_t>((b & 0x807Fu) |
                                                   (static_cast<std::uint16_t>(100 + (b % 36)) << 7));
                }
                h_x[i] = b;
            }

            void *x, *codes, *scales, *dout, *dout1;
            if (cudaMalloc(&x, x_bytes) != cudaSuccess ||
                cudaMalloc(&codes, code_bytes) != cudaSuccess ||
                cudaMalloc(&scales, scale_bytes) != cudaSuccess ||
                cudaMalloc(&dout, out_bytes) != cudaSuccess ||
                cudaMalloc(&dout1, static_cast<std::size_t>(p.rows) * 2) != cudaSuccess) {
                std::printf("FAIL: cudaMalloc %s\n", p.name);
                return 2;
            }
            cudaMemcpy(x, h_x.data(), x_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(codes, h_codes.data(), code_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(scales, h_scales.data(), scale_bytes, cudaMemcpyHostToDevice);

            ninfer::Tensor tx;
            tx.data = x; tx.ne[0] = p.k; tx.ne[1] = kT; tx.ne[2] = 1; tx.ne[3] = 1;
            tx.dtype = ninfer::DType::BF16;
            ninfer::Weight tw;
            tw.qdata = codes; tw.scales = scales;
            tw.n = p.rows; tw.k = p.k;
            tw.weight_scale_divisor = 1.177f;  // nonzero dirty divisor: exercises the fold order
            ninfer::Tensor tout;
            tout.data = dout; tout.ne[0] = p.rows; tout.ne[1] = kT; tout.ne[2] = 1; tout.ne[3] = 1;
            tout.dtype = ninfer::DType::BF16;

            ninfer::ops::detail::launch_nvfp4_small_t(tx, tw, tout, 0);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                std::printf("FAIL: launch %s\n", p.name);
                return 2;
            }

            std::vector<std::uint16_t> h_out(static_cast<std::size_t>(p.rows) * kT);
            cudaMemcpy(h_out.data(), dout, out_bytes, cudaMemcpyDeviceToHost);
            const std::uint32_t hsh =
                fnv1a(reinterpret_cast<const unsigned char*>(h_out.data()), out_bytes);
            std::printf("%-3s %-18s rows=%-6d k=%-6d T=%d fnv1a=%08x first=%04x %04x %04x last=%04x %04x\n",
                        tag, p.name, p.rows, p.k, kT, hsh, h_out[0], h_out[1], h_out[2],
                        h_out[h_out.size() - 2], h_out[h_out.size() - 1]);
            std::fwrite(h_out.data(), 2, h_out.size(), out);

            // ---- cross-arm leg: decode GEMV (T=1) on token column 0 must byte-match the
            // small_t T=3 rows of token 0 (CT-5 pair property; both drives on the same data).
            ninfer::Tensor tx1;
            tx1.data = x; tx1.ne[0] = p.k; tx1.ne[1] = 1; tx1.ne[2] = 1; tx1.ne[3] = 1;
            tx1.dtype = ninfer::DType::BF16;
            ninfer::Tensor tout1;
            tout1.data = dout1; tout1.ne[0] = p.rows; tout1.ne[1] = 1; tout1.ne[2] = 1; tout1.ne[3] = 1;
            tout1.dtype = ninfer::DType::BF16;
            ninfer::ops::detail::launch_nvfp4_decode(tx1, tw, tout1, 0);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                std::printf("FAIL: decode launch %s\n", p.name);
                return 2;
            }
            std::vector<std::uint16_t> h_out1(p.rows);
            cudaMemcpy(h_out1.data(), dout1, static_cast<std::size_t>(p.rows) * 2,
                       cudaMemcpyDeviceToHost);
            const int mism = std::memcmp(h_out1.data(), h_out.data(),
                                         static_cast<std::size_t>(p.rows) * 2);
            if (mism != 0) { ++cross_arm_bad; }
            std::printf("    cross-arm decode(T=1,col0) vs small_t(T=3,col0): %s\n",
                        mism == 0 ? "BIT-EQUAL YES" : "BIT-EQUAL NO");

            cudaFree(x); cudaFree(codes); cudaFree(scales); cudaFree(dout); cudaFree(dout1);
        }
    } // pass
    std::fclose(out);

    if (argc >= 3) {
        FILE* rf = std::fopen(argv[2], "rb");
        if (!rf) { std::printf("FAIL: cannot open ref %s\n", argv[2]); return 2; }
        FILE* cf = std::fopen(argv[1], "rb");
        if (!cf) { std::printf("FAIL: cannot open cur %s\n", argv[1]); return 2; }
        const std::size_t BUFSZ = 1 << 16;
        static unsigned char rb[BUFSZ], cb[BUFSZ];
        std::size_t total = 0, off = 0;
        int verdict = 0;
        for (;;) {
            std::size_t rn = std::fread(rb, 1, BUFSZ, rf);
            std::size_t cn = std::fread(cb, 1, BUFSZ, cf);
            if (rn != cn) { verdict = 1; total += rn < cn ? rn : cn; off = total; break; }
            if (rn == 0) { break; }
            total += rn;
            for (std::size_t i = 0; i < rn; ++i) {
                if (rb[i] != cb[i]) { verdict = 1; off = total - rn + i; break; }
            }
            if (verdict) { break; }
        }
        std::fclose(rf);
        std::fclose(cf);
        if (verdict) {
            std::printf("BIT-EQUAL NO (differ at byte offset %zu)\n", off);
            return 1;
        }
        std::printf("BIT-EQUAL YES (%zu bytes, 5 W4 problems x 2 passes at T=%d)%s\n", total, kT,
                    cross_arm_bad == 0 ? "" : " [WARNING: cross-arm leg mismatched]");
        return cross_arm_bad == 0 ? 0 : 1;
    }
    std::printf("wrote %s (pass this file as ref.bin on later runs)\n", argv[1]);
    return 0;
}
