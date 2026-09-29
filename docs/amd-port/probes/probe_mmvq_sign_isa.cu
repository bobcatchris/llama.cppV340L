// ISA census on gfx900: shipped shim chain vs candidate perm atom
#include "vendors/hip.h"
#include <cstdint>
__global__ void chain_ref_iq3xxs(const uint8_t * in, int * out) {
    // one l0 pair of the shipped iq3_xxs decode: vcmpne4 + xor + vsub4 x2
    const uint32_t aux32 = *(const uint32_t*)in;
    const uint32_t signs = ((aux32 >> 0) * 0x01010101u); // unpack stand-in
    const uint32_t g0 = in[1], g1 = in[2];
    const int signs0 = __vcmpne4(signs & 0x08040201, 0);
    const int signs1 = __vcmpne4(signs & 0x80402010, 0);
    const int gl = __vsub4(g0 ^ signs0, signs0);
    const int gh = __vsub4(g1 ^ signs1, signs1);
    out[threadIdx.x] = gl + gh;
}
__global__ void chain_cand_iq3xxs(const uint8_t * in, int * out) {
    const uint32_t aux32 = *(const uint32_t*)in;
    const uint32_t signs = ((aux32 >> 0) * 0x01010101u);
    const uint32_t g0 = in[1], g1 = in[2];
    const uint32_t me = signs & 0x08040201;
    const uint32_t mo = signs & 0x80402010;
    const int s0 = __builtin_amdgcn_perm(me * 0xF0, me * 0xF0, 0xBA98);
    const int s1 = __builtin_amdgcn_perm(mo | (mo * 0x0E), mo | (mo * 0x0E), 0xBA98);
    const int gl = (g0 ^ s0) + (s0 & 0x01010101);
    const int gh = (g1 ^ s1) + (s1 & 0x01010101);
    out[threadIdx.x] = gl + gh;
}
__global__ void chain_ref_iq3s(const uint8_t * in, int * out) {
    const uint8_t sp8 = in[0];
    const uint32_t g0 = in[1], g1 = in[2];
    const int signs0 = __vcmpne4(((sp8 & 0x03) << 7) | ((sp8 & 0x0C) << 21), 0x00000000);
    const int signs1 = __vcmpne4(((sp8 & 0x30) << 3) | ((sp8 & 0xC0) << 17), 0x00000000);
    const int gl = __vsub4(g0 ^ signs0, signs0);
    const int gh = __vsub4(g1 ^ signs1, signs1);
    out[threadIdx.x] = gl + gh;
}
__global__ void chain_cand_iq3s(const uint8_t * in, int * out) {
    const uint8_t sp8 = in[0];
    const uint32_t g0 = in[1], g1 = in[2];
    const uint32_t m0 = ((sp8 & 0x03) << 7) | ((sp8 & 0x0C) << 21);
    const uint32_t m1 = ((sp8 & 0x30) << 3) | ((sp8 & 0xC0) << 17);
    const int s0 = __builtin_amdgcn_perm(m0 | (m0 << 7), m0 | (m0 << 7), 0xBA98);
    const int s1 = __builtin_amdgcn_perm(m1 | (m1 << 7), m1 | (m1 << 7), 0xBA98);
    const int gl = (g0 ^ s0) + (s0 & 0x01010101);
    const int gh = (g1 ^ s1) + (s1 & 0x01010101);
    out[threadIdx.x] = gl + gh;
}
int main() { return 0; }
