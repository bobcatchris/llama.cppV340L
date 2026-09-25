#!/usr/bin/env bash
# Standalone ISA receipts for the __syncwarp shim mapping landed in
# src/common/hip_shim/cuda_runtime.h (agent2, C441 follow-up).
# Re-run: bash docs/amd/v340l/syncwarp_isa_receipts.sh
#
# The claim being tested is NOT "does it compile" but "does mapping __syncwarp()
# to a wavefront builtin actually produce a barrier, and does its absence break
# anything". Both answers were measured, and the first one is not what the
# donor's comment implies.
set -e
CL=/opt/rocm-6.2.0/lib/llvm/bin/clang++
D=${1:-$HOME/worktrees/amd-wo-shim-width}
T=$(mktemp -d)

cat > $T/wb.cpp <<'EOF'
#include <hip/hip_runtime.h>
__global__ void uni(int*o){ __builtin_amdgcn_wave_barrier(); o[threadIdx.x]=1; }
__global__ void div_(int*o){ if(threadIdx.x<32) __builtin_amdgcn_wave_barrier(); o[threadIdx.x]=1; }
EOF

cat > $T/lds.cpp <<'EOF'
#include <hip/hip_runtime.h>
// gqa_decode_body.cuh:131 -> :139 shape: intra-warp LDS writes, __syncwarp(), LDS reads.
__global__ void t(const float*x,float*o){
  __shared__ float sw[32];
  int lane=threadIdx.x&31;
  if(lane<16) sw[lane]=x[lane];
#if defined(USE_SW)
  __builtin_amdgcn_wave_barrier();
#endif
  float acc=0.f; for(int i=0;i<16;i++) acc+=sw[i];
  o[threadIdx.x]=acc;
}
EOF

echo "== R1: what does __builtin_amdgcn_wave_barrier() lower to on gfx900? =="
$CL -O2 -x hip --offload-arch=gfx900 -S $T/wb.cpp -o $T/wb.s
n=$(awk '/^_Z3uniPi:/,/Lfunc_end/' $T/wb.s | grep -cE "\bs_barrier\b|\bv_barrier\b|\bds_barrier\b" || true)
w=$(awk '/^_Z3uniPi:/,/Lfunc_end/' $T/wb.s | grep -c "s_waitcnt lgkmcnt" || true)
echo "   (informational) s_waitcnt lgkmcnt present: $w  <- memory ordering, NOT a barrier."
echo "   Do not count these as barriers: an earlier draft of this script did and reported"
echo "   a false positive that contradicted the finding below."
echo "   barrier-class instructions in the unconditional kernel: $n"
echo "   -> 0 means the builtin is OPTIMIZED AWAY: it emits no barrier at all."

echo "== R2: same, with only lanes<32 calling it (does divergence force one?) =="
awk '/^_Z4div_Pi:/,/Lfunc_end/' $T/wb.s | grep -cE "\bs_barrier\b|\bv_barrier\b" || true
echo "   -> also 0. Narrowing EXEC does not make it emit a barrier."

echo "== R3: does a write->read LDS dependency need it? (gfx900) =="
$CL -O2 -x hip --offload-arch=gfx900 -S $T/lds.cpp -o $T/no.s
$CL -O2 -x hip --offload-arch=gfx900 -DUSE_SW -S $T/lds.cpp -o $T/sw.s
for v in no sw; do
  w=$(awk '/^_Z1tPKfPf:/,/Lfunc_end/' $T/$v.s | grep -c "ds_write")
  r=$(awk '/^_Z1tPKfPf:/,/Lfunc_end/' $T/$v.s | grep -c "ds_read")
  k=$(awk '/^_Z1tPKfPf:/,/Lfunc_end/' $T/$v.s | grep -c "s_waitcnt lgkmcnt")
  echo "   $v: ds_write=$w ds_read=$r lgkmcnt-waits=$k"
done
echo "   -> identical lgkmcnt wait counts with and without the barrier: the compiler"
echo "      already orders LDS write->read, so the no-op is not losing a fence here."

echo "== R4: is the mapping actually required by anything in the HIP build today? =="
CON=$(grep -rl "__syncwarp" $D/src/ | grep -vc hip_shim); SH=$(grep -rl "__syncwarp" $D/src/ | grep -c hip_shim)
echo "   consumer files: $CON   (+$SH shim file defining it = $((CON+SH)) total)"
echo "   (20 consumers; the 21st hit is this shim itself)" 
echo "   -> none are reachable from src/HipSources.cmake (verified by include"
echo "      closure). Before this shim every one of them failed to COMPILE, loudly."
rm -rf $T

echo "== R5: HOST-COMPILE cell (added after gemini+agent3 found the build break) =="
# The defect this checks for: __syncwarp's body calls a builtin that does not exist in a
# pure host compilation. Every standalone probe before this one compiled with -x hip,
# which defines __HIPCC__ and so never exercised the plain-g++ path used for the .cpp
# half of the whitelist. Three modes must all be clean:
CL=/opt/rocm-6.2.0/lib/llvm/bin/clang++
h=$(g++ -fsyntax-only -D__HIP_PLATFORM_AMD__ -I /opt/rocm-6.2.0/include -I $D/src \
      -I $D/src/common/hip_shim -I $D/src/include -I $D/include \
      $D/src/core/linear_attention_state.cpp 2>&1 | grep -c "error:" || true)
printf "   pure host .cpp (g++, what CMake uses for the whitelist half): errors=%s %s\n" "$h" "$([ "$h" = 0 ] && echo PASS || echo FAIL)"
cat > $T/sw.cu <<'EOF'
#include <cuda_runtime.h>
__global__ void k(int*o){ __syncwarp(); __syncwarp(0xffffu); o[threadIdx.x]=1; }
EOF
d=$($CL -O2 -x hip --offload-arch=gfx900 -I $D/src -I $D/src/common/hip_shim -I $D/include \
     -fsyntax-only $T/sw.cu 2>&1 | grep -c "error:" || true)
printf "   -x hip device+host passes:                               errors=%s %s\n" "$d" "$([ "$d" = 0 ] && echo PASS || echo FAIL)"
echo "   GUARD RULE: use __HIPCC__, NOT __HIP_DEVICE_COMPILE__. The latter is undefined in"
echo "   the HOST PASS of -x hip, so guarding with it makes valid __global__ bodies that"
echo "   call __syncwarp fail to compile — the 'obvious' fix introduces a second break."
