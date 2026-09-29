#include "vendors/hip.h"
#include <cstdio>
__device__ __forceinline__ unsigned int asm_perm(unsigned int lo, unsigned int hi, unsigned int sel) {
    unsigned int d;
    asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(d) : "v"(lo), "v"(hi), "v"(sel));
    return d;
}
__global__ void probe3(const unsigned int * sel, unsigned int * out) {
    // lo bytes 01,02,04,08 (MSB 0); hi bytes 10,20,40,80 (MSB 1)
    const unsigned int lo = 0x08040201u, hi = 0x80402010u;
    for (int i = 0; i < 16; ++i) out[i] = asm_perm(lo, hi, sel[i]);
    out[16] = asm_perm(lo, hi, 0xBA98u);
    out[17] = asm_perm(lo, hi, 0x0000BA98u ^ 0x0000FFFFu); // 0x4567
    // mode-bit on single nibble 0x8 with byte 0 selected: lo byte0 = 01 MSB0, hi byte0 = 10 MSB1
    out[18] = asm_perm(lo, hi, 0x00000008u);
    out[19] = asm_perm(lo, hi, 0x0000000Fu);
}
int main() {
    unsigned int sels[16];
    for (int i = 0; i < 16; ++i) sels[i] = 0x11111111u * i;
    unsigned int * ds, * dout; unsigned int out[32];
    hipMalloc(&ds, 16*4); hipMalloc(&dout, 32*4);
    hipMemcpy(ds, sels, 16*4, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(probe3, dim3(1), dim3(1), 0, 0, ds, dout);
    hipMemcpy(out, dout, 32*4, hipMemcpyDeviceToHost);
    printf("asm v_perm_b32(lo=08040201, hi=80402010, sel):\n");
    const char * hx = "0123456789ABCDEF";
    for (int i = 0; i < 16; ++i) printf("  sel=%c x4: %08x\n", hx[i], out[i]);
    printf("  sel=BA98: %08x\n", out[16]);
    printf("  sel=4567: %08x\n", out[17]);
    printf("  sel=00000008: %08x\n", out[18]);
    printf("  sel=0000000F: %08x\n", out[19]);
    return 0;
}
