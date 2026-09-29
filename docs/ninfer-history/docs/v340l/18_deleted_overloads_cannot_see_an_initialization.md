# 18 — Deleted overloads cannot see an initialization: the `ldm_addr_t` guard closes one route to the seam

Measured 2026-09-13, CPU-only (`hipcc -c`, no device, no grant). Triggered by agent3's endorsement of the
second-witness residual (guard blind one level up through the `*_swz_addr` helpers) and their offer to
harden it. **Their finding is real; this note is about how far the proposed fix reaches.**

## The claim

The proposed hardening — deleted `unsigned`-first-param overloads on the three `*_swz_addr` helpers,
`__HIP__`-arm-scoped — **closes raw tokens passed as arguments and does not close raw tokens widened at a
declaration.** Therefore it must not be described as sealing the seam.

## Probe

`/tmp/seam_probe.cu`, using the HIP-arm types verbatim from the landed tree (`ldm_addr_t = unsigned long
long`, `mma.cuh:36`) plus exactly the proposed shape:

```cpp
#include <hip/hip_runtime.h>
using ldm_addr_t = unsigned long long;              // mma.cuh:36, HIP arm
__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return (unsigned)(unsigned long long)(const char*)p;
}
__device__ void g(ldm_addr_t a) { (void)a; }
__device__ void g(unsigned a) = delete;             // the PROPOSED hardening, minimal form
__global__ void route_a(unsigned p[64]) { g(smem_addr(p)); }                    // :9   raw token as ARGUMENT
__global__ void route_b(unsigned p[64]) { ldm_addr_t x = smem_addr(p); g(x); }  // :10  raw token via DECLARATION
```

```
hipcc -c seam_probe.cu -o /dev/null
  -> seam_probe.cu:9:43: error: call to deleted function 'g'
  -> 1 error total; line 10 produces NO diagnostic
```

Reproduce: `hipcc -c /tmp/seam_probe.cu -o /dev/null` (or the file is trivially re-typed from above).

## Why this is structural, not a defect in the fix

Deleted overloads act through **overload resolution at a call site**. `ldm_addr_t x = smem_addr(p)` is an
**initialization**: `unsigned -> unsigned long long` is a standard widening conversion with no candidate
function set for the compiler to refuse. There is no name lookup that could select the deleted entity. So
the type system is not being lenient here — the instrument has no handle on that site at all, and no
amount of overload deletion gives it one.

A real seal has to move the refusal to a place that *is* an expression the compiler checks: a **strong
token type** (wrapper struct, no implicit constructor from `unsigned`), which makes the initialization
itself a diagnostic. Cost is real — it touches the CUDA arm, where `ldm_addr_t = unsigned` (`mma.cuh:106`),
and that arm's bytes have been treated as load-bearing all night. Recommendation: **record the residual,
do not chase the seal in this wave.**

## Consequences for the follow-up commit (relayed to author and coordinator)

1. **Wording.** Carry "closes argument-position raw tokens; declaration-site widening remains" rather than
   "seals"/"hardens the seam". The population left open is exactly the `lane_base`/`sbase` assignments
   agent3's own census counts (16 of them).
2. **The negative cell must gain a route_b leg, asserted to stay green today.** If the new helper-routed
   legs are argument-position only, the tripwire prints green while covering half the seam — a vacuous
   coverage claim, the same failure class their Arm B mutation test exists to prevent. A negative cell
   should fail when coverage *shrinks*, and document what was never covered.
3. **Exploitability is a separate question from the class.** `swz_addr(smem_addr` matches **0** in-tree
   (agent3's predicate, my read agrees the direct-argument form has no live instances), and I did **not**
   measure whether any specific in-tree declaration-site widening reaches an `ldmatrix_*` call. Class real,
   mechanism measured, in-tree reachability unmeasured — stated so nobody upgrades it by inference.

## Census corroboration, no conflict

Their landed log `results/amd/T3_sweep_guardcompile_2026-09-13.log` (`edf485f2`, resolves and is an ancestor
of `origin/amd/t3-wip`) line 32: `TIMEOUT CLOSURE (retry, 1500s budget): w8_small_t.cu compiled CLEAN rc=0
at ~19 min`. Independent of my own run at main `60ee38b6`: **rc=0, 1128 s**. Two instruments, ~19 min,
±12 s. Their `34 CLEAN / 16 pre-existing-class / 0 unknown` is a fleet tally; my `3 red / 4 clean` is the
w8 LDS census over seven files — different populations, both correct, per the counting law.
