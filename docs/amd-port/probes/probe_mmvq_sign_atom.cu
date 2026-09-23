// probe: exhaustive host validation of the iq3 perm sign atom vs the shipped shim semantics (W6 receipt, B1 rung)
// build+run: g++ -O2 -I ggml/src -DGGML_COMMON_IMPL_CPP <this file> (host-only, no GPU)
// NOTE: the candidate atom passes here and FAILS on device - gfx900 v_perm_b32 selector
// semantics do not match the MSB-replication model; see W6_mmvq_rungs_receipt
// B1 gfx900 probe: exhaustive host validation + ISA census of iq3 sign-chain
// candidate atoms vs the shipped shim semantics.
#include "ggml-common.h"
#include <cstdio>
#include <cstring>
#include <cstdint>

// ---- reference semantics (the CUDA shims in vendors/hip.h) ----
static uint32_t ref_vcmpne4(uint32_t a, uint32_t b) {
    uint32_t c = 0;
    for (int i = 0; i < 4; ++i) {
        const uint8_t va = (a >> (8*i)) & 0xFF, vb = (b >> (8*i)) & 0xFF;
        c |= (uint32_t)(va != vb ? 0xFF : 0x00) << (8*i);
    }
    return c;
}
static uint32_t ref_vsub4(uint32_t a, uint32_t b) {
    // __vsub4 -> __vsubss4 (saturating). The idiom never saturates, so plain
    // per-byte two's-complement wrap is the semantics that matters here.
    uint32_t c = 0;
    for (int i = 0; i < 4; ++i) {
        const int8_t va = (a >> (8*i)) & 0xFF, vb = (b >> (8*i)) & 0xFF;
        c |= (uint32_t)(int8_t)(va - vb) << (8*i);
    }
    return c;
}

// ---- candidate atoms ----
static inline uint32_t perm_msb(uint32_t f) {
    // v_perm_b32(f, f, 0xBA98): output byte i = replicate MSB of byte i
    uint32_t c = 0;
    for (int i = 0; i < 4; ++i) {
        c |= ((f >> (8*i + 7)) & 1) ? (uint32_t)0xFF << (8*i) : 0;
    }
    return c;
}
// iq3_xxs: m has isolated bits at 0,9,18,27 (mask 0x08040201) or 4,13,22,31
// (mask 0x80402010). MSB-build: even m*0xF0, odd m | (m*0x0E).
static inline uint32_t cand_s0_xxs_even(uint32_t m) { return perm_msb(m * 0xF0); }
static inline uint32_t cand_s0_xxs_odd (uint32_t m) { return perm_msb(m | (m * 0x0E)); }
// iq3_s: isolated bits at 7,8,23,24 for both masks (0x<<7 | 0x<<21 forms).
static inline uint32_t cand_s0_s(uint32_t m) { return perm_msb(m | (m << 7)); }
// sign apply: plain 32-bit add form (carry when g byte == 0 and signed?)
static inline uint32_t cand_apply_add(uint32_t g, uint32_t s0) {
    return (g ^ s0) + (s0 & 0x01010101);
}
// sign apply: SWAR per-byte add (borrow-free, 7-bit lanes + MSB xor)
#define H8 0x80808080u
static inline uint32_t swar_add(uint32_t u, uint32_t v) {
    return ((u & ~H8) + (v & ~H8)) ^ ((u ^ v) & H8);
}
static inline uint32_t cand_apply_swar(uint32_t g, uint32_t s0) {
    return swar_add(g ^ s0, s0 & 0x01010101);
}

int main() {
    long bad_add = 0, bad_swar = 0, bad_s0e = 0, bad_s0o = 0, bad_s0s = 0;
    long carry_cases = 0;
    for (int gi = 0; gi < 256; ++gi) {
        const uint32_t g = iq3xxs_grid[gi];
        for (int sb = 0; sb < 256; ++sb) {
            const uint32_t s0 = ref_vcmpne4(sb, 0);  // FF/00 per byte
            const uint32_t want = ref_vsub4(g ^ s0, s0);
            if (cand_apply_add(g, s0) != want)  { ++bad_add; }
            if (cand_apply_swar(g, s0) != want) { ++bad_swar; }
            // count the carry-hazard cases (g byte == 0 and its sign set)
            for (int i = 0; i < 4; ++i) {
                if (((g >> (8*i)) & 0xFF) == 0 && ((s0 >> (8*i)) & 0xFF) == 0xFF) ++carry_cases;
            }
        }
        // s0 builders: exhaustive over the isolated-bit input m
        for (uint32_t m = 0; m < 256; ++m) {
            // even mask idiom: m*0x01010101 broadcast then & 0x08040201
            const uint32_t me = (m * 0x01010101u) & 0x08040201u;
            const uint32_t mo = (m * 0x01010101u) & 0x80402010u;
            if (cand_s0_xxs_even(me) != ref_vcmpne4(me, 0)) ++bad_s0e;
            if (cand_s0_xxs_odd(mo)    != ref_vcmpne4(mo, 0)) ++bad_s0o;
            // iq3_s idiom: (sp8 & 0x03)<<7 | (sp8 & 0x0C)<<21
            const uint32_t ms = ((m & 0x03) << 7) | ((m & 0x0C) << 21);
            if (cand_s0_s(ms) != ref_vcmpne4(ms, 0)) ++bad_s0s;
        }
    }
    printf("exhaustive iq3 sign-atom test over 256 grid entries x 256 sign bytes:\n");
    printf("  apply_add  mismatches: %ld (carry-hazard g_b==0&&signed cases seen: %ld)\n", bad_add, carry_cases);
    printf("  apply_swar mismatches: %ld\n", bad_swar);
    printf("  s0_xxs_even mismatches: %ld  s0_xxs_odd: %ld  s0_iq3s: %ld\n", bad_s0e, bad_s0o, bad_s0s);
    printf("VERDICT: %s\n", (bad_add | bad_swar | bad_s0e | bad_s0o | bad_s0s) == 0 ? "ALL EXACT" : "DEFECTS FOUND");
    return (bad_add | bad_swar | bad_s0e | bad_s0o | bad_s0s) != 0;
}
