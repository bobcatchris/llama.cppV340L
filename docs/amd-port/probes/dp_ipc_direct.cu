// dp_ipc_direct.cu - discriminating probe for the IPC export gate (6.2 API).
// hipMalloc 8MB on die 0, hsa_amd_pointer_info (base/size/type), then
// hsa_amd_ipc_memory_create direct - exact base+size and raw ptr variants.

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    hipSetDevice(0);
    const size_t SZ = 8u << 20;
    void* p = nullptr;
    hipError_t he = hipMalloc(&p, SZ);
    printf("hipMalloc: %s p=%p\n", hipGetErrorString(he), p);
    if (he != hipSuccess) return 2;

    hsa_status_t s;
    hsa_amd_pointer_info_t info {};
    info.size = sizeof(info);
    s = hsa_amd_pointer_info(p, &info, nullptr, nullptr, nullptr);
    printf("pointer_info: 0x%x type=%d agentBase=%p sizeInBytes=%zu owner=0x%lx\n",
           s, (int) info.type, info.agentBaseAddress, info.sizeInBytes,
           (unsigned long) info.agentOwner.handle);
    if (s != HSA_STATUS_SUCCESS) return 3;
    printf("base-match: %d  size-match: %d\n",
           info.agentBaseAddress == p, info.sizeInBytes >= SZ);

    hsa_amd_ipc_memory_t h1 {};
    s = hsa_amd_ipc_memory_create(info.agentBaseAddress, info.sizeInBytes, &h1);
    printf("ipc_memory_create(exact base+full size): 0x%x\n", s);
    if (s != HSA_STATUS_SUCCESS) {
        hsa_amd_ipc_memory_t h2 {};
        s = hsa_amd_ipc_memory_create(p, SZ, &h2);
        printf("ipc_memory_create(raw ptr+SZ):           0x%x\n", s);
    }
    hipFree(p);
    return 0;
}
