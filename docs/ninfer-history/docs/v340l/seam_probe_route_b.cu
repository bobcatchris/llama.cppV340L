#include <hip/hip_runtime.h>
using ldm_addr_t = unsigned long long;          // mma.cuh:36, HIP arm
__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return (unsigned)(unsigned long long)(const char*)p;
}
// the shape of agent3's PROPOSED hardening, minimal reproduction
__device__ void g(ldm_addr_t a) { (void)a; }
__device__ void g(unsigned a) = delete;         // deleted unsigned-first overload
__global__ void route_a(unsigned p[64]) { g(smem_addr(p)); }                     // raw token as ARGUMENT
__global__ void route_b(unsigned p[64]) { ldm_addr_t x = smem_addr(p); g(x); }    // raw token via DECLARATION
