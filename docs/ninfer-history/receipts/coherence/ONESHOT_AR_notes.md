# ONESHOT-AR: gfx900 port notes — `src/core/multi_gpu/one_shot_allreduce.cu`

Branch `amd/tp4-cure` (no-GPU seat). Scope honored: only `one_shot_allreduce.cu` touched;
`tp_group.cpp`, `src/HipSources.cmake`, sibling `one_shot_argmax.cu` NOT edited (owner's).

## 0. Tree state found at task time (facts for the GPU owner)

- The CUDA→HIP **volatile-access rewrite already existed in-tree** (the `#if defined(__HIP__)`
  branch, "TA1 census fix" vintage): zero PTX executed on the HIP lane even before this commit —
  the `asm volatile("ld.global.cv…"/"st.global.wt…")` bodies were confined to the `#else`
  (CUDA) branch and are still there, untouched, NVIDIA line's property.
- `src/HipSources.cmake` carries BOTH notes: the stale `:50-55` "deliberately NOT whitelisted"
  block AND the later WO-06 S3d `:248-251` "§3-gated device trio" entry —
  `core/multi_gpu/one_shot_allreduce.cu` **is already inside `NINFER_HIP_STEP1_SOURCES`**
  (`:292`), i.e. it already builds into `ninfer_hip_host` on this branch's base. The landing
  decision this commit enables is the DEVICE VALIDATION of the new access mechanism + the
  tp_group gate flip, not a whitelist add. Parity-guard leg check-(h) should be re-run on
  these bytes by the owner before any boot.
- Caller gate (read-only): `tp_group.cpp:137` constructs `OneShotAllReduce` only under
  `I.n == 2`; `:338` routes an AR call to it only when `n_elems <= kMaxElements` (65536).
  The class is 2-rank by contract (`rank != 0 && rank != 1` throws) — a TP4 (world=4)
  flip is a pairing/extension DESIGN decision, not an env guard; see §4(f).

## 1. asm → portable mapping (what the commit ships)

| CUDA original (NVIDIA branch, kept verbatim) | gfx900 HIP form shipped now | ISA emitted (measured this box, ROCm 6.2.0 clang, `-O3 --offload-arch=gfx900`) |
|---|---|---|
| `asm volatile("ld.global.cv.v4.u32 {…}, [%1];" : "l"(ptr))` | `__builtin_nontemporal_load` on native `unsigned int __attribute__((ext_vector_type(4)))`, punned back to `uint4` (union + `static_assert` size/layout) | `global_load_dwordx4 v[..], v[..], off glc slc` — ONE 128-bit transaction |
| `asm volatile("st.global.wt.v4.u32 [%0], {…};")` | `__builtin_nontemporal_store` (same vector retype) | `global_store_dwordx4 v[..], v[..], off glc slc` — ONE 128-bit transaction |
| flag / gen / status word accesses (`volatile int*` derefs) | UNCHANGED (plain volatile dword) | `flat_load_dword … glc` (×6 in-kernel, count unchanged pre/post) — the G-AMD-30a **measured** transport |

Before/after evidence, whole-TU and kernel-body ISA dumps: `slc` count went **0 → 3**; the
3 are exactly the 3 helper call sites (publish store ×1, peer-read load ×2 — one per
residual/no-residual combine arm). Instruction population otherwise byte-equivalent in form
counts; compile warnings identical (24× pre-existing host-side `nodiscard` on
`cudaSetDevice`/`cudaFreeHost` — untouched by this commit).

API adaptations ledger (everything the HIP path needed and uses — all pre-existing shim
surfaces, none added):
- `cudaHostAlloc/cudaHostAllocMapped/Portable` → `hipHostMalloc/Mapped/Portable`
  (shim `cuda_runtime.h:362-365`; addendum `multi_gpu_hip_addendum.h:28-34`).
- `cudaHostGetDevicePointer` → `hipHostGetDevicePointer` (shim `:390`, addendum `:32`).
- `__float2bfloat16_rn` → `__float2bfloat16` (shim `cuda_bf16.h:36`); `__hadd2` bf16 emulation
  (shim-provided, ~6-instruction RNE — no bf16 ALU on gfx900).
- `__nanosleep` (CUDA-only) already branch-separated; HIP side uses `__builtin_amdgcn_s_sleep(1)`
  + bounded poll + soft-fail (line law 14:4xZ / 15:0xZ) — pre-existing, untouched.

## 2. Why the builtin is compile-mandatory shape (two probe findings)

Probe: `/tmp/oneshot/probe2.hip` (ISA dumped; not banked — reproduce with
`/opt/rocm-6.2.0/lib/llvm/bin/clang++ -O3 -std=gnu++20 --offload-arch=gfx900 -x hip -S`):

1. `__builtin_nontemporal_load/store` **rejects HIP's `uint4`** ("address argument to
   nontemporal builtin must be a pointer to integer, float, pointer, or a vector of such
   types") — `uint4` is a struct wrapper. The native `ext_vector_type(4)` vector passes.
   Hence the helpers re-type internally; call sites keep `uint4`/`void*` signatures.
2. A whole-struct volatile deref (`U4 r = *vp;`) is **C++20-ill-formed** (deleted implicit
   volatile copy-constructor). The componentwise/union retype is language-mandated, not style.

Predecessor (volatile-componentwise) lowering, same toolchain: backend merged the 4 dwords,
loads carrying `glc` but split below 128-bit (context-dependent dwordx2 pairing), stores
merged with **NO cache-hint bits**. The shipped form is strictly closer to the CUDA
original on both legs: single 128-bit transaction + `glc slc`.

## 3. Visibility-semantics argument per construct ([design] where marked)

Box shape (measured, COORDINATOR.md): gfx900 ×4 dies, **zero P2P** (G-AMD-5), all cross-die
traffic host-staged over PCIe (~3.13 GiB/s, RTT 98-101 µs flat). The one-shot AR reads the
peer's **host-mapped pinned staging buffer** directly from the kernel; the CUDA `.cv`/`.wt`
semantics exist to (a) not serve stale L1 and (b) push writes out for cross-agent visibility
over the fabric. Equivalence on gfx900:

- **Load leg (`ld.global.cv` → `global_load_dwordx4 glc slc`)**: on GCN5 the vector L1 is
  per-CU and NOT coherent across CUs/SEs; `glc=1` on a VMEM load bypasses L1, satisfying the
  fetch-fresh-from-L2 semantic of `.cv` [design — ISA manual modifier semantics, not
  device-measured here]. `slc=1` marks the line system-level-coherent, the canonical hint for
  PCIe/host-mapped traffic. The poll protocol (volatile flag dword, `glc`, then payload loads)
  gives read ordering: payload reads are data/control-dependent on the flag observation inside
  one thread, and the leader pattern keeps the poll in one thread (`tid==0`).
- **Store leg (`st.global.wt` → `global_store_dwordx4 glc slc`)**: gfx900 L1 is write-through
  for VMEM stores — every store reaches L2 regardless of hints [design]; `glc slc` on top
  removes any doubt and matches the donor's write-through intent. What `.wt` does NOT do by
  itself on EITHER vendor is order — ordering is the choreography's job and is unchanged:
  `payload stores → __threadfence_system → *my_gen → __threadfence_system → *my_flag`
  (KAR-v2 chain, CORD seq-71), with the peer polling flag/gen via the G-AMD-30a volatile
  transport. The fence is the visibility edge; the store hints only choose the path.
- **Flag/gen/status words**: untouched volatile dword — this is the transport that IS
  device-measured on this box (G-AMD-30a row: volatile bits cross host-mapped memory,
  atomic bits do not). No new mechanism was introduced where a measured one existed.

Honest residual ([design], open): whether gfx900 L2 can serve a stale line for a
host-rewritten staging address on the READ leg (both old and new forms share this exposure;
`slc` centers it but nothing here PROVES L2 invalidation behavior for system lines). This is
exactly what the device window in §4 must witness, and it is not new risk added by this
commit — the KAR gen/flag protocol was built (boot-7/8 lineage) because torn/stale reads on
these paths were observed and instrumented.

## 4. Device-test plan for the GPU owner (no-GPU seat wrote this; owner executes)

Gate: all env-gated, default OFF — default binary behavior must be unchanged when envs are
absent (E-17 tap discipline: env-gated arm, both-direction falsifiers).

(a) **Gate flip, owner-owned shape** (`tp_group.cpp:136-138`, not this commit): construct /
route one-shot behind an env (e.g. `NINFER_TP4_ONESHOT=1`) — but note §0's 2-rank contract:
at world=4 the CURRENT class needs a pairing decision (two 2-rank one-shots + cross-pair
combine, or a 4-rank extension) BEFORE any flip. Until then the honest TP4 arm is: keep the
existing world=4 transport, run the one-shot A/B at world=2 to validate the ACCESS MECHANISM,
and extend only on a measured win.
(b) **RED/GREEN capture (closure law)**: RED = pre-commit sha's binary under the SAME device
cells; GREEN = post-commit sha. Both shas named in the closing row; the row joins the boot
battery per-LABEL.
(c) **Numerics bit-check (E-17 bar, re-scoped)**: one-shot arm vs ring/NCCL arm, temp 0,
count600, SAME geometry — token streams byte-identical; any divergence classified per the
E-17 discipline (batched-verify geometry numerics vs coherence) before it may be called a
transport bug. Cross-arm acceptance ≤ target-argmax if an MTP leg rides the same window.
(d) **KAR witnesses**: `NINFER_MB_ARGMAX_TRACE` (first-16 + REJECT-candidate lines) and
`NINFER_MC31` retry census — grep counts pre/post must be zero-retry steady state; any
REJECT-candidate is a torn-read-window event, loud by design.
(e) **count600 A/B + clocks sidebands**: A = current default transport, B = one-shot arm;
`rocm-smi` clock samples banked alongside both runs (sideband discipline); target check: one
collect ≈ RTT-class cost — the round math is 128 collects × ~98-101 µs ≈ 12.5-13 ms naive
vs the ~2-4 ms/round TP4 target; report measured ms/round, never estimated VRAM or ETA.
(f) **Coherence witness for §3's residual**: at minimum one leg where the peer rank's host
rewrites the SAME slot lines back-to-back (slot reuse inside one boot) with payload
verification — this is the L2-stale-line falsifier; if it tears, the loud path is the
existing gen-gate (REJECT-candidate), and the fix conversation is a real cache-maintenance
op (`buffer_wbinvl1_vol` family), not this commit's re-litigation.
(g) **Whitelist/parity**: re-run check-(h) on these bytes; HipSources.cmake untouched here.

## 5. Compile status (this seat, zero-GPU)

```
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -O3 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 \
  -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I src/common/hip_shim -I include -I src -I third_party -DNDEBUG -std=gnu++20 \
  --offload-arch=gfx900 -x hip -c src/core/multi_gpu/one_shot_allreduce.cu -o oneshot_check.o
# RC=0, 0 errors, 24 warnings (all pre-existing host nodiscard class), object 55464 B
```

Not committed to amd/main; branch `amd/tp4-cure` only, commit prefix `ONESHOT-AR:`.
