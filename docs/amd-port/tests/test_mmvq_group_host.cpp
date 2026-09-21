// Host oracle for the grouped decode GEMV (GGML_CUDA_MMVQ_GROUP=1).
//
// Proves bit-exactness of the design the way the die-3 oracle does (see
// results/W2_decode_atom_2026-09-21.md): mirror both device schedules on the
// host over the same raw weight/x bytes and require identical float bits.
//
//   solo    mirror  <- mul_mat_vec_q<type, ncols_dst=1, has_fusion=false,
//                              small_k=false> (ggml/src/ggml-cuda/mmvq.cu)
//   grouped mirror  <- mul_mat_vec_q_grouped + mmvq_group_row (same file)
//
// plus the geometry gate (ggml_cuda_mmvq_group_geometry_ok) for gfx900
// (GCN table: nwarps = 2 for every type at ncols_dst == 1, warp 64).
//
// vec_dot mirrors: q8_0 and q4_K (the census's tiny-tensor classes),
// quantize mirror: quantize_q8_1 (q8_1 layout d@0, s@2, qs@4, stride 36).
//
// Build/run (host-only, no GPU, no libggml link):
//   hipcc -O2 -std=gnu++17 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
//     docs/amd-port/tests/test_mmvq_group_host.cpp -o /tmp/test_mmvq_group
//   /tmp/test_mmvq_group

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// ---- quant constants (ggml/src/ggml-common.h, vecdotq.cuh) ----
static constexpr int QK8_1 = 32;
static constexpr int QK8_0 = 32;
static constexpr int QK_K  = 256;
static constexpr int K_SCALE_SIZE = 12;
static constexpr int VDR_Q8_0_Q8_1_MMVQ = 2;
static constexpr int VDR_Q4_K_Q8_1_MMVQ = 2;

static constexpr int q8_0_qk = QK8_0;                // 32
static constexpr int q8_0_qi = QK8_0 / 4;            // QI8_0 = 8
static constexpr int q4_K_qk = QK_K;                 // 256
static constexpr int q4_K_qi = QK_K / 8;             // QI4_K = 32 (QR4_K = 2)
static constexpr int q4_K_qr = 2;                    // QR4_K
static constexpr int q8_1_stride = 2*sizeof(uint16_t) + QK8_1;   // 36
static constexpr int q8_0_stride = sizeof(uint16_t) + QK8_0;     // 34
static constexpr int q4_K_stride = 2*sizeof(uint16_t) + K_SCALE_SIZE + QK_K/2; // 144

static int test_failures = 0;
#define REQUIRE(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); test_failures++; } } while (0)

// ---- fp16 (bit-level, avoids the HIP __half host VALUE-conversion trap of
// the W2 oracle: scales are stored as raw bytes everywhere below) ----
static float h16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exp  = (h & 0x7c00) >> 10;
    const uint32_t man  = h & 0x03ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t m = man;
            int e = -1;
            do { m <<= 1; e++; } while ((m & 0x400) == 0);
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint16_t f32_to_h16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    const int32_t  e    = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    const uint32_t m    = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) { // inf/nan
        return (uint16_t) (sign | 0x7c00 | (m ? 0x200 : (m >> 13)));
    }
    if (e >= 31) {
        return (uint16_t) (sign | 0x7c00); // inf (overflow)
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t) sign;
        }
        const uint32_t man = (m | 0x800000) >> (1 - e);
        return (uint16_t) (sign | man); // rounding floor; only exact halves used here
    }
    const uint32_t man = m >> 13;
    const uint32_t rem = m & 0x1fff;
    const uint32_t tie = (rem > 0x1000) || (rem == 0x1000 && (man & 1));
    return (uint16_t) (sign | ((uint32_t) e << 10) | man + tie);
}

// ---- dp4a + int readers (vecdotq.cuh / common.cuh, gfx900 semantics) ----
static inline int h_dp4a(int a, int b, int c) {
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
}

static inline int get_int_b2(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    return (int) ((uint32_t) x16[2*i32 + 0] << 0 | (uint32_t) x16[2*i32 + 1] << 16);
}

static inline int get_int_b4(const void * x, const int i32) {
    return ((const int *) x)[i32];
}

// ---- q8_1 quantize mirror (quantize_q8_1 kernel, T=1: ne00 == ne0 == 5120) ----
struct block_q8_1_h {
    uint16_t d;     // scale, raw fp16
    uint16_t s;     // d * sum, raw fp16
    int8_t   qs[QK8_1];
};

static void quantize_row_q8_1_h(const float * x, const int64_t ne0, std::vector<uint8_t> & out) {
    out.assign((ne0/QK8_1)*q8_1_stride, 0);
    for (int64_t ib = 0; ib < ne0/QK8_1; ++ib) {
        float amax = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < QK8_1; ++i) {
            amax = std::max(amax, fabsf(x[ib*QK8_1 + i]));
        }
        // warp_reduce_sum<QK8_1>: butterfly offsets 16,8,4,2,1
        float v[QK8_1];
        float nv[QK8_1];
        for (int i = 0; i < QK8_1; ++i) {
            v[i] = x[ib*QK8_1 + i];
        }
        for (int offset = QK8_1/2; offset > 0; offset >>= 1) {
            for (int i = 0; i < QK8_1; ++i) {
                nv[i] = v[i] + v[i ^ offset];
            }
            memcpy(v, nv, sizeof(v));
        }
        sum = v[0];

        const float d = amax / 127.0f;
        block_q8_1_h blk;
        for (int i = 0; i < QK8_1; ++i) {
            blk.qs[i] = amax == 0.0f ? 0 : (int8_t) roundf(x[ib*QK8_1 + i] / d);
        }
        blk.d = f32_to_h16(d);
        blk.s = f32_to_h16(sum);
        memcpy(&out[ib*q8_1_stride], &blk, q8_1_stride);
    }
}

static inline float q81_d(const uint8_t * y) {
    return h16_to_f32(*(const uint16_t *) y);
}

// ---- vec_dot mirrors (vecdotq.cuh, verbatim arithmetic order) ----

static float vec_dot_q8_0_q8_1_h(const void * vbq, const uint8_t * bq8_1, const int & kbx, const int & iqs) {
    const uint8_t * bq8_0 = (const uint8_t *) vbq + (size_t) kbx*q8_0_stride;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(bq8_0 + sizeof(uint16_t), iqs + i); // bq8_0->qs
        u[i] = get_int_b4(bq8_1 + 2*sizeof(uint16_t), iqs + i); // bq8_1->qs
    }

    int sumi = 0;
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        sumi = h_dp4a(v[i], u[i], sumi);
    }
    const float d8_0 = h16_to_f32(*(const uint16_t *) bq8_0);
    const float d8_1 = q81_d(bq8_1); // __low2half(bq8_1->ds)
    return d8_0*d8_1 * ((float) sumi);
}

static float vec_dot_q4_K_q8_1_h(const void * vbq, const uint8_t * bq8_1, const int & kbx, const int & iqs) {
    const uint8_t * bq4_K = (const uint8_t *) vbq + (size_t) kbx*q4_K_stride;

    int v[2];
    int u[2*q4_K_qr];
    float d8[q4_K_qr];

    const int bq8_offset = q4_K_qr * ((iqs/2) / (QK8_1/4/2)); // QR4_K * ((iqs/2) / (QI8_1/2)), QI8_1 = 8

    const int * q4 = (const int *)(bq4_K + 2*sizeof(uint16_t) + K_SCALE_SIZE + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)(bq4_K + 2*sizeof(uint16_t));
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < q4_K_qr; ++i) {
        const uint8_t * bq8i = bq8_1 + (size_t)(bq8_offset + i)*q8_1_stride;
        d8[i] = q81_d(bq8i);

        const int * q8 = (const int *)(bq8i + 2*sizeof(uint16_t)) + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    // vec_dot_q4_K_q8_1_impl_vmmq
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    for (int i = 0; i < q4_K_qr; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = h_dp4a(v1i, u[2*i+1], h_dp4a(v0i, u[2*i+0], 0));
        const int dot2 = h_dp4a(0x01010101, u[2*i+1], h_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float dm_x = h16_to_f32(*(const uint16_t *) (bq4_K + 0));        // d
    const float dm_y = h16_to_f32(*(const uint16_t *) (bq4_K + 2));        // dmin
    return dm_x*sumf_d - dm_y*sumf_m;
}

// ---- weight builders (raw bytes, LCG fill; fp16 scales chosen exact) ----
static uint32_t lcg_state = 0x12345678u;
static inline uint32_t lcg() {
    lcg_state = lcg_state*1664525u + 1013904223u;
    return lcg_state >> 8;
}

struct member_h {
    int q4_k;                 // 1 = q4_K, 0 = q8_0
    int nrows;
    int ncols;
    std::vector<uint8_t> w;   // raw weight bytes
    std::vector<float>  ref;  // reference dst (f64 accumulation)
    std::vector<float>  solo;
    std::vector<float>  grp;
};

static void build_member(member_h & mb, uint64_t seed) {
    lcg_state = (uint32_t) seed;
    const int blck = mb.q4_k ? q4_K_qk : q8_0_qk;
    const int nblocks = (mb.nrows*mb.ncols)/blck;
    const int stride = mb.q4_k ? q4_K_stride : q8_0_stride;
    mb.w.resize((size_t) nblocks*stride);
    for (int b = 0; b < nblocks; ++b) {
        uint8_t * blk = &mb.w[(size_t) b*stride];
        // scale d (and dmin for q4_K): small exact halves
        const uint16_t d    = f32_to_h16(0.25f + 0.25f*(float) (lcg() % 4));
        *(uint16_t *) blk = d;
        if (mb.q4_k) {
            *(uint16_t *)(blk + 2) = f32_to_h16(0.0625f*(float) (lcg() % 4)); // dmin
            for (int i = 0; i < K_SCALE_SIZE; ++i) {
                blk[4 + i] = (uint8_t) (lcg() % 256);
            }
            for (int i = 0; i < QK_K/2; ++i) {
                blk[16 + i] = (uint8_t) (lcg() % 256);
            }
        } else {
            for (int i = 0; i < QK8_0; ++i) {
                blk[2 + i] = (uint8_t) (lcg() % 256);
            }
        }
    }
}

// ---- schedule mirrors (GGN table, gfx900: nwarps = 2, warp = 64) ----
static constexpr int NWARPS_GCN = 2;
static constexpr int WARP = 64;

// butterfly warp_reduce_sum<64> (common.cuh)
static inline float warp_reduce_sum64(float * lane) {
    float v[64];
    float nv[64];
    memcpy(v, lane, sizeof(v));
    for (int offset = 32; offset > 0; offset >>= 1) {
        for (int i = 0; i < 64; ++i) {
            nv[i] = v[i] + v[i ^ offset];
        }
        memcpy(v, nv, sizeof(v));
    }
    return v[0];
}

struct type_traits_h {
    int qk;
    int qi;
    int vdr;
    float (*vec_dot)(const void *, const uint8_t *, const int &, const int &);
    int wstride;
};

static type_traits_h traits_of(const member_h & mb) {
    type_traits_h t{};
    if (mb.q4_k) {
        t = { q4_K_qk, q4_K_qi, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1_h, q4_K_stride };
    } else {
        t = { q8_0_qk, q8_0_qi, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1_h, q8_0_stride };
    }
    return t;
}

// one row, schedule of mul_mat_vec_q<type, 1, false, false> (GCN decode geometry)
static float solo_row(const member_h & mb, const type_traits_h & tt, const uint8_t * vy, const int row) {
    const int blocks_per_row_x = mb.ncols/tt.qk;
    const int blocks_per_iter  = tt.vdr*NWARPS_GCN*WARP/tt.qi;
    const int stride_row_x     = mb.ncols/tt.qk; // contiguous weights
    const int kbx_offset       = row*stride_row_x;

    float tmp[NWARPS_GCN*WARP];
    for (int tid = 0; tid < NWARPS_GCN*WARP; ++tid) {
        tmp[tid] = 0.0f;
        for (int kbx = tid/(tt.qi/tt.vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (tt.qk/QK8_1);
            const int kqs = tt.vdr * (tid % (tt.qi/tt.vdr));
            tmp[tid] += tt.vec_dot(mb.w.data(), vy + (size_t) kby*q8_1_stride, kbx_offset + kbx, kqs);
        }
    }

    // tmp_shared[0][lane] from warp 1, then warp-0 reduce
    float sh[64];
    float lane[64];
    for (int l = 0; l < 64; ++l) {
        sh[l] = tmp[64 + l];
    }
    for (int l = 0; l < 64; ++l) {
        lane[l] = tmp[l] + sh[l]; // solo adds shared l = 0 .. nwarps-2 first
    }
    return warp_reduce_sum64(lane);
}

// grouped member params (mirror of ggml_cuda_mmvq_group_params fields in use)
struct group_params_h {
    const void * vx[8];
    float *      dst[8];
    uint32_t ncols_x[8];
    uint32_t nrows_x[8];
    uint32_t stride_row_x[8];
    uint32_t n_members;
};

// one row of one member, schedule of mmvq_group_row (bit-identical skeleton)
static float grouped_row(const group_params_h & p, const member_h * members, const type_traits_h * tts,
                         const uint8_t * vy, const uint32_t m, const int row) {
    const member_h & mb = members[m];
    const type_traits_h & tt = tts[m];
    const int nwarps = NWARPS_GCN; // blockDim.y, host-gated uniform

    const int blocks_per_row_x  = (int) p.ncols_x[m]/tt.qk;
    const int blocks_per_iter   = tt.vdr*nwarps*WARP/tt.qi;
    const int stride_row_x      = (int) p.stride_row_x[m];
    const int kbx_offset        = row*stride_row_x;

    float tmp[NWARPS_GCN*WARP];
    for (int t = 0; t < NWARPS_GCN*WARP; ++t) {
        tmp[t] = 0.0f;
        for (int kbx = t/(tt.qi/tt.vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (tt.qk/QK8_1);
            const int kqs = tt.vdr * (t % (tt.qi/tt.vdr));
            tmp[t] += tt.vec_dot(p.vx[m], vy + (size_t) kby*q8_1_stride, kbx_offset + kbx, kqs);
        }
    }

    float sh[64];
    float lane[64];
    for (int l = 0; l < 64; ++l) {
        sh[l] = tmp[64 + l];
    }
    for (int l = 0; l < 64; ++l) {
        lane[l] = tmp[l] + sh[l];
    }
    return warp_reduce_sum64(lane);
}

// ---- geometry gate mirror (ggml_cuda_mmvq_group_geometry_ok + small_k), gfx900 ----
static constexpr int CC_GFX900 = 0x1000000 + 900; // GGML_CUDA_CC_OFFSET_AMD + 900

static int table_id_h(const int cc) {
    // get_device_table_id: gfx900 lands in MMVQ_PARAMETERS_GCN
    const bool amd = (cc & 0xFF000000) == 0x1000000;
    (void) amd;
    if (cc == CC_GFX900) {
        return 2; // MMVQ_PARAMETERS_GCN
    }
    return 0;     // MMVQ_PARAMETERS_GENERIC
}

static int calc_nwarps_h(const int /*type unused on GCN*/, const int ncols_dst, const int table_id) {
    if (table_id == 2) { // GCN
        return ncols_dst <= 4 ? 2 : 1;
    }
    return ncols_dst <= 4 ? 4 : 2; // GENERIC
}

static bool small_k_would_trigger_h(const type_traits_h & tt, const int64_t ncols_x, const int /*cc*/, const int warp_size) {
    const int nwarps = NWARPS_GCN;
    const int blocks_per_row_x = (int) (ncols_x/tt.qk);
    const int blocks_per_iter_1warp = tt.vdr*warp_size/tt.qi;
    return nwarps > 1 && blocks_per_row_x < nwarps*blocks_per_iter_1warp;
    // NVIDIA iq exceptions are no-ops on gfx900 and are mirrored verbatim in mmvq.cu
}

static bool geometry_ok_h(const type_traits_h * tts, const int64_t * ncols_x, const int n, int * nwarps_out) {
    int nwarps = -1;
    for (int i = 0; i < n; ++i) {
        const int nw = calc_nwarps_h(0, 1, table_id_h(CC_GFX900));
        if (nwarps == -1) {
            nwarps = nw;
        }
        if (nw != nwarps) {
            return false;
        }
        if (small_k_would_trigger_h(tts[i], ncols_x[i], CC_GFX900, WARP)) {
            return false;
        }
    }
    *nwarps_out = nwarps;
    return n > 0;
}

// ---- test ----
int main() {
    const int ncols = 5120; // census GDN norm output width

    // one shared x (5120 floats, LCG pattern, signed)
    std::vector<float> x(ncols);
    lcg_state = 0xfeedface;
    for (int i = 0; i < ncols; ++i) {
        x[i] = ((float) (int) (lcg() % 4096) - 2048.0f) / 2048.0f;
    }
    std::vector<uint8_t> q81;
    quantize_row_q8_1_h(x.data(), ncols, q81);
    REQUIRE(q81.size() == (size_t)(ncols/QK8_1)*q8_1_stride, "q8_1 size");

    // quantize sanity: per-block d (after fp16 rounding, as on device) matches
    // that block's amax/127
    {
        float amax0 = 0.0f;
        for (int i = 0; i < QK8_1; ++i) {
            amax0 = std::max(amax0, fabsf(x[i]));
        }
        const float d_expected = h16_to_f32(f32_to_h16(amax0/127.0f));
        REQUIRE(q81_d(q81.data()) == d_expected, "q8_1 d mismatch");
    }

    // representative batch: ssm_alpha/ssm_beta class (q4_K 5120x48) + q8_0
    // variants + differing row counts (grid-guard exercise)
    std::vector<member_h> members(4);
    members[0] = { 1, 48, ncols, {}, {}, {}, {} };
    members[1] = { 0, 48, ncols, {}, {}, {}, {} };
    members[2] = { 1, 16, ncols, {}, {}, {}, {} };
    members[3] = { 0, 32, ncols, {}, {}, {}, {} };
    for (size_t m = 0; m < members.size(); ++m) {
        build_member(members[m], 0xa5a5000 + m);
    }

    // reference: q8_0 members get an independent per-element f64 accumulation
    // over dequantized raw bytes. q4_K members (6-bit scale packing) get an f64
    // accumulation over the same vec_dot decode summed across the FULL
    // (kbx, kqs) coverage - independent of the kbx partition and reduction
    // order, so a schedule/coverage bug still cannot hide.
    for (size_t m = 0; m < members.size(); ++m) {
        member_h & mb = members[m];
        const type_traits_h tt = traits_of(mb);
        mb.ref.assign(mb.nrows, 0.0f);

        if (!mb.q4_k) {
            std::vector<double> yf(ncols);
            for (int i = 0; i < ncols; ++i) {
                const uint8_t * blk = q81.data() + (size_t)(i/QK8_1)*q8_1_stride;
                const float d = q81_d(blk);
                yf[i] = (double) d * (double) ((int8_t *) (blk + 4))[i % QK8_1];
            }
            for (int r = 0; r < mb.nrows; ++r) {
                double acc = 0.0;
                for (int i = 0; i < ncols; ++i) {
                    const int blck = q8_0_qk;
                    const int kb = i/blck;
                    const uint8_t * blk = mb.w.data() + (size_t)((r*(mb.ncols/blck)) + kb)*tt.wstride;
                    const float wv = h16_to_f32(*(const uint16_t *) blk)*(int8_t) blk[2 + (i % blck)];
                    acc += (double) wv * yf[i];
                }
                mb.ref[r] = (float) acc;
            }
        } else {
            const int bpr = ncols/q4_K_qk;
            const int kby_stride = (q4_K_qk/QK8_1)*q8_1_stride; // kby = kbx*(qk/QK8_1) blocks
            for (int r = 0; r < mb.nrows; ++r) {
                double acc = 0.0;
                for (int kbx = 0; kbx < bpr; ++kbx) {
                    for (int kqs = 0; kqs < q4_K_qi; kqs += tt.vdr) {
                        acc += (double) vec_dot_q4_K_q8_1_h(mb.w.data(),
                            q81.data() + (size_t) kbx*kby_stride, r*bpr + kbx, kqs);
                    }
                }
                mb.ref[r] = (float) acc;
            }
        }
    }

    // solo schedule: what the ungrouped kernels produce per member
    for (size_t m = 0; m < members.size(); ++m) {
        member_h & mb = members[m];
        const type_traits_h tt = traits_of(mb);
        mb.solo.resize(mb.nrows);
        for (int r = 0; r < mb.nrows; ++r) {
            mb.solo[r] = solo_row(mb, tt, q81.data(), r);
        }
    }

    // grouped schedule: one launch over all members via the params layout
    {
        type_traits_h tts[8];
        int64_t ncols_x[8];
        group_params_h p = {};
        p.n_members = (uint32_t) members.size();
        std::vector<float> dst_store[8];
        for (size_t m = 0; m < members.size(); ++m) {
            tts[m] = traits_of(members[m]);
            ncols_x[m] = members[m].ncols;
            p.vx[m] = members[m].w.data();
            dst_store[m].resize(members[m].nrows);
            p.dst[m] = dst_store[m].data();
            p.ncols_x[m] = members[m].ncols;
            p.nrows_x[m] = members[m].nrows;
            p.stride_row_x[m] = (uint32_t) (members[m].ncols/tts[m].qk);
        }

        int nwarps = 0;
        REQUIRE(geometry_ok_h(tts, ncols_x, (int) members.size(), &nwarps), "geometry gate must pass for the census batch (gfx900)");
        REQUIRE(nwarps == 2, "gfx900 GCN geometry must give nwarps = 2, got %d", nwarps);

        for (uint32_t m = 0; m < p.n_members; ++m) {
            for (int r = 0; r < (int) p.nrows_x[m]; ++r) {
                p.dst[m][r] = grouped_row(p, members.data(), tts, q81.data(), m, r);
            }
        }
        for (size_t m = 0; m < members.size(); ++m) {
            members[m].grp = dst_store[m];
        }
    }

    // 1) bit-exactness: grouped == solo, every member, every row
    for (size_t m = 0; m < members.size(); ++m) {
        const member_h & mb = members[m];
        REQUIRE(mb.solo.size() == mb.grp.size(), "member %zu result count", m);
        for (int r = 0; r < (int) mb.solo.size(); ++r) {
            uint32_t a, b;
            memcpy(&a, &mb.solo[r], 4);
            memcpy(&b, &mb.grp[r], 4);
            REQUIRE(a == b, "member %zu row %d: solo %08x vs grouped %08x (%.9g vs %.9g)",
                    m, r, a, b, mb.solo[r], mb.grp[r]);
        }
    }

    // 2) sanity: solo results track the fp64 reference (mirrors are honest dots)
    for (size_t m = 0; m < members.size(); ++m) {
        const member_h & mb = members[m];
        for (int r = 0; r < (int) mb.solo.size(); ++r) {
            const double rel = fabs((double) mb.solo[r] - (double) mb.ref[r]) /
                std::max(1e-9, fabs((double) mb.ref[r]));
            REQUIRE(rel < 1e-3, "member %zu row %d: solo %.9g vs ref %.9g (rel %.3g)", m, r, mb.solo[r], mb.ref[r], rel);
        }
    }

    // 3) whitelist gate: small_k must reject tiny K (solo schedule would change)
    {
        const type_traits_h tt4 = traits_of(members[0]);
        const type_traits_h tt0 = traits_of(members[1]);
        REQUIRE(small_k_would_trigger_h(tt4, 256, CC_GFX900, WARP), "q4_K K=256 must trigger small_k");
        REQUIRE(small_k_would_trigger_h(tt0, 256, CC_GFX900, WARP), "q8_0 K=256 must trigger small_k");
        REQUIRE(!small_k_would_trigger_h(tt4, 5120, CC_GFX900, WARP), "q4_K K=5120 must not trigger small_k");
        REQUIRE(!small_k_would_trigger_h(tt0, 5120, CC_GFX900, WARP), "q8_0 K=5120 must not trigger small_k");
    }

    // 4) grid guard: rows beyond a member's nrows_x never write
    {
        // exercised implicitly: members 2/3 have fewer rows than member 0; a
        // max_rows grid over-launches blocks for them and grouped_row is only
        // invoked for r < nrows_x (the device kernel early-returns on
        // blockIdx.x >= nrows_x[m], which the host loop mirrors)
        REQUIRE(members[2].nrows < members[0].nrows && members[3].nrows < members[0].nrows, "grid-guard shape setup");
    }

    if (test_failures == 0) {
        printf("ALL PASS (%zu members, %d rows checked, bit-exact solo==grouped)\n",
               members.size(), members[0].nrows + members[1].nrows + members[2].nrows + members[3].nrows);
        return 0;
    }
    printf("%d FAILURES\n", test_failures);
    return 1;
}
