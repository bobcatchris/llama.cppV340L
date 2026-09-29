# v340l/07 — distance to a servable HIP `ninfer-serve` (gap inventory + 3 shim fixes)

**Author:** Agent-B (compat/runtime lane). 2026-09-12 ~20:2xZ. **Zero GPU, zero launch** — every
claim below is a compile, a link, a grep, or a `du`. Prompted by asking the question nobody had:
P3's goal is *serving*, so can this tree produce a HIP server binary at all?

## Headline
**No — and not because of the kernels.** Two independent blockers exist, and the one that stops
the build today is **build-graph wiring**, not unported code. The serve layer (engine, HTTP,
schemas, product) already compiles HIP-clean; four device files that produce 20 of the missing
symbols compile clean too and are simply not whitelisted.

## Blocker 1 — `NINFER_BUILD_APPS=ON` cannot configure on the HIP lane
Measured: configuring with `-DNINFER_BUILD_APPS=ON -DNINFER_BACKEND=hip` fails at
`CMakeLists.txt:106` → `Package 'libcurl', required by 'virtual:world', not found`, i.e. **before
`apps/` is ever visited**. Chain of causation, read from the file:
- `CMakeLists.txt:79-81` sets `NINFER_BUILD_MEDIA_ACQUIRE ON` whenever `NINFER_BUILD_APPS` is on,
  via plain `set()` — so, exactly as the v340l/00 traps note records, a `-D` override **cannot**
  turn it off.
- `src/product/media_acquire/acquire.cpp` needs `curl/curl.h`; **no curl headers exist anywhere on
  this host** (`find / -name curl.h` → empty; only `libcurl-gnutls.so.3/.4` runtime, no dev
  package; `openssl` pkg-config also absent). No passwordless sudo, so installing is outside our
  authority.

## Blocker 2 — the targets `apps/` links do not exist on the HIP lane
`src/CMakeLists.txt:9-15`: under `NINFER_BACKEND=hip` the file includes `HipSources.cmake` and
then **`return()`s**. Everything below — including `ninfer_serve` and
`ninfer_product_load_progress` — is the CUDA build. Yet `CMakeLists.txt:118-119` still does
`add_subdirectory(apps)` on the HIP branch, and `apps/CMakeLists.txt:19` links precisely those two
absent targets. Proof, not inference:

```
$ cmake --build build-hip-amd --target ninfer_serve
gmake: *** No rule to make target 'ninfer_serve'.  Stop.
$ find /home/chris/worktrees /home/chris/dual_5060_ti_ninfer -name ninfer-serve -type f
(no output — a HIP serve binary has never been linked on this host)
```

**So the board's "serve stamp pending agent4's link" has a structural precondition nobody had
tested.** No amount of kernel porting yields a server until apps/ gets a HIP branch.

## What is already fine (the useful surprise)
Serving code is *not* the problem: 13 `serve/*.cpp` are already whitelisted and the archive
exports **1299 `ninfer::serve` symbols**; `engine.cpp`, `prompt_input.cpp`,
`load_progress.cpp` each compile **0 errors** under `--offload-arch=gfx900`.

## The remaining symbol distance, classified
Linking `apps/serve/main.cpp` + those three + the archive + `-lrccl` yields exactly **20 distinct
undefined symbols**, all device-layer. Grouped by owning file and by defect class (each row is a
measured syntax check, not a guess):

| class | files | state after my fixes | owner |
|---|---|---|---|
| **whitelist-only** (compiles 0-error today) | `core/multi_gpu/tp_kernel.cu`, `one_shot_allreduce.cu`, `one_shot_argmax.cu`, `ops/kvarn/kvarn_workspace.cu` | **proven**: added to the whitelist, archive built **168/168 objects, parity OK** | needs coord ruling + gemini's PG-1 exception list (bounded-spin law already satisfied by agent3's `:80/:95` fix) |
| `mma_bf16` undeclared (PTX `mma.cuh` family) | `linear/bf16/bf16_gemm_mma.cu`, `attn_input_proj/q4_q5/…_gemm_mma.cu`, `linear/q6/q6_rowsplit_gemm_mma.cu`, `linear/w8/w8_splitk`,`w8_small_t` | arithmetic blocker cleared, still RED on mma | agent3 — HipSources already marks this "fresh exception ruling required" |
| PTX `invalid input constraint 'l'` | `linear/bf16/bf16_gemv.cu`, `bf16_small_t.cu` | RED | agent3 — the same `l`-constraint class documented for one_shot |
| `cuda_fp4.h` absent | `linear/nvfp4/nvfp4_small_t.cu`, `nvfp4_w4a4.cu` | RED | **not a shim hole to paper over** — nvfp4 is a CUDA-only tier; agent3's D1 include-trim is the honest route, and I did **not** fabricate a header |
| `HostKVArena::~HostKVArena` | `runtime/tp2/host_kv_parked.cpp` is whitelisted yet the dtor is undefined | OPEN — not chased to ground | flag for agent3/coord |

## My three fixes (all in `hip_shim`, all measured before/after)
1. **`cuda_pipeline.h` did not include hip headers.** It writes `__device__ __forceinline__`, but
   both names come from hip's `amd_detail/host_defines.h`, and the file pulled in only
   `<cstddef>/<cstdint>/<cstring>`. Result: any TU reaching it through
   `ops/common/memory.cuh` failed — and the *only* errors in `w8_rowsplit_gemm_simt.cu` were my
   three lines, i.e. **a defect in red's own shim was blocking a kernel port that had nothing else
   wrong with it**. Now includes `<hip/hip_runtime.h>`: that file went **RED → CLEAN**.
2. **`__hsub2_rn` for bf16 pairs** (`cuda_bf16.h`), mapping to HIP's `__hsub2`. Demand counted, not
   assumed — the tree uses `__half2half2` 11×, `__hsub2` 8×, `__hmul2` 5×, `__hadd2` 4×,
   `__halves2half2` 2×, `__hsub2_rn` 2×, so this one name is the whole `_rn` gap; HIP has zero
   `_rn` *arithmetic* spellings (`__hsub2_rn`/`__hadd2_rn` = 0 hits in `amd_hip_bf16.h`) but does
   provide the plain pair forms, and exposes no alternative rounding mode, so it is the same
   operation rather than a downgrade. **My first attempt was wrong and is recorded:** I grepped the
   bare name, assumed the fp16 family, and wrote the alias into `cuda_fp16.h`; the call sites are
   `__nv_bfloat162`, so that version had **zero** users — exactly the dead prophylactic surface I've
   been arguing against. Removed; correct file, correct type.
3. **`cuda_bf16.h` now includes `<cstdint>`.** Found while *testing* (2): a standalone host
   `#include <cuda_bf16.h>` produced 20 errors, `unknown type name 'uint64_t'` from hip's
   `device_library_decls.h`. I checked whether my edit caused it by re-running against the
   pre-change header — **identical 20 errors**, so it was pre-existing, and my test rather than my
   change was the first suspect. hip relies on a prior include for `uint64_t`; one line makes the
   standalone path supported.

### D3 WARNING attached to fix (2), in-file, because it is the point
Fix (2) **widens the D3 surface**: everything above it in that header is *conversion* helpers
(float→bf16, legal by construction), while `__hsub2_rn` is *arithmetic* — the precise category whose
"compiler-enforced" premise this sprint proved false (measured: bf16 pair ops compile clean on
gfx900 and lower to ~6-instruction RNE emulation). After this alias, a bf16 arithmetic kernel in
`ops/kernel/**` compiles on the HIP lane with **nothing** checking whether that is intended. The
compensating control is gemini's ISA-fingerprint/symbol-use gate (checks (e)+(i), being corrected
for exactly this hole per STATE 16:0xZ), **not** this header, and the numerics of the two w8 call
sites remain agent3's golden question. I am making the spelling resolve; I am not certifying it.

## Proposed minimal path (needs coord rulings — I did not edit shared build files)
1. **media_acquire**: either obtain libcurl dev headers, or gate `NINFER_BUILD_MEDIA_ACQUIRE` off
   for `NINFER_BACKEND=hip` at `CMakeLists.txt:79-81` **and** supply a loud-throwing HIP
   `acquire_bytes`. My `/tmp/fa/media_stub_hip.cpp` proves the rest of the chain links with it
   present; it throws rather than returning empty, so a text-only bring-up cannot silently lose a
   media path. Text-only q3 bring-up does not use it. Root `CMakeLists.txt` is the shared
   CUDA/HIP file (registered handoff exception) → **your edit, not mine**.
2. **apps HIP branch**: link `ninfer-serve` against `ninfer_hip_host` + `-lrccl` instead of the
   CUDA target names; whitelist `engine.cpp`, `prompt_input.cpp`, `load_progress.cpp`.
3. **Whitelist the 4 already-clean device files** (measured 168/168) subject to gemini's PG-1 list.
4. Leave the three genuine porting classes to agent3; specifically **do not** fabricate
   `cuda_fp4.h`.

## Reproduce
```bash
cd <lane worktree at this commit>
cmake -S . -B /tmp/b -DNINFER_BACKEND=hip -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF \
      -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0            # -> libcurl configure failure
cmake --build build-hip-amd --target ninfer_serve    # -> No rule to make target
for f in src/ops/linear/w8/w8_rowsplit_gemm_simt.cu src/ops/linear/bf16/bf16_gemm_mma.cu \
         src/ops/linear/nvfp4/nvfp4_small_t.cu; do
  clang++ -x hip --offload-arch=gfx900 -std=c++20 -fsyntax-only \
    -I src/common/hip_shim -I src -I include -I third_party "$f" 2>&1 | grep -c 'error:'
done
```
Disk note for whoever acts on this: the two verified artifacts are 20.44 GB + 15.45 GB and only
the ref lives on `/` — free space is **28 GB**, so a full CUDA-tree build (~22 GB in this
project's history) plus a local copy of q3 would not both fit. Check `df -h /` before starting.

---
# UPDATE (20:4xZ) — the "OPEN" item is closed, and it found something better

## HostKVArena::~HostKVArena — resolution: not a mystery, a plain omission + a real finding
Defined in **`src/core/host_kv_arena.cpp:59`**, which is simply **not in `HipSources.cmake`**.
That file compiles **0 errors under hipcc *and* 0 errors under plain g++**, so it belongs to the
same *whitelist-only* class as the four multi_gpu/kvarn files — not a porting task.

**How my own method produced the wrong first answer, recorded because the trap is generic:** my
symbol→file mapping grepped `--include=*.cu` only. A `.cpp` defining file was structurally
invisible to it, so I wrote "OPEN — host_kv_parked.cpp is whitelisted yet the dtor is undefined",
having looked in the wrong place and then blamed a guard. The coordinator's hypothesis
("smells like a conditional-definition guard, similar to your param-1 class") was also not the
cause. The fix to my method: map symbols across **all** source extensions, and treat my own
grep scope as a suspect before treating the code as mysterious.

**Second claim I checked and had to discard:** I thought this file corroborated the param-1
mechanism (`cudaHostAlloc(&p, …)` at :26 with a non-`void**` argument). Measured: with the *old*
`hipHostAlloc` mapping the file compiles **0 errors**, so it corroborates nothing. Two reasons,
both read from the source: `p` is declared `void*` (so `&p` *is* `void**`), and the call sits
inside `#ifdef NINFER_HAS_CUDA`. I had inferred "p is std::byte*" from the `static_cast` on the
next line. Not published as evidence — the single-file q3/q2 reproductions remain the proof.

## The finding that fell out of that: `make_pinned()` is inert on the HIP lane
`src/core/host_kv_arena.cpp:24-31`:

```cpp
#ifdef NINFER_HAS_CUDA
    void* p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
    ...
#else
    return make_pageable(bytes);      // <- the HIP lane lands here
#endif
```

`NINFER_HAS_CUDA` appears **nowhere** in the HIP build's `flags.make`, and I found no definition
for it in any CMake file or header reachable from this checkout (and there is no CUDA build tree
on this host to compare against — no nvcc here — so I cannot say what the CUDA lane does;
someone with the NVIDIA tree should confirm it is defined there). What I *can* say from bytes:
**on the HIP configuration as built, `make_pinned()` silently returns a pageable arena** — no
error, no log, no `cudaHostAlloc` call, and `pinned_mem_` is set false.

Why that matters for P3 and not just for tidiness: pinned host buffers are what the staged TP2
transport relies on, and P1's bandwidth numbers were measured through whatever path was actually
live at the time. A silent pinned→pageable substitution is exactly the class this line keeps
hitting — a capability that is quietly not what its name claims, and a measurement whose
provenance can't be trusted until the substitution is excluded. Two asks, neither mine to
execute: (i) agent4/agent3 should confirm whether P1's staged-transport figures were taken with
pinned actually enabled; (ii) the honest fix is either to define the macro for HIP (it uses only
`cudaHostAlloc`/`cudaFreeHost`, both of which my shim now maps correctly) or to make the fallback
**loud** — a `NINFER_HIP_NO_PINNED` notice at construction rather than silence. Choosing is the
coordinator's.

## NEW DEFECT (21:1xZ): mma.cuh's ldmatrix family is OUTSIDE the __HIP__ guard
`src/ops/common/mma.cuh:7-31` declares `ldmatrix_x2/x4/x2t/x4t` as raw PTX
(`asm volatile("ldmatrix.sync.aligned.m8n8…")`) and the `#if !defined(__HIP__)` guard that covers
the `mma.sync` family does not begin until **:33**. So on the HIP lane the ldmatrix four are
compiled unguarded. Measured on a synthetic caller, same TU both ways:
```
-fsyntax-only … gfx900   -> 0 errors
-c            … gfx900   -> src/ops/common/mma.cuh:8:18: error: invalid instruction
```
A PTX mnemonic is an opaque string to the front end; only the AMDGPU back end rejects it, so
**this is invisible to every `-fsyntax-only` gate**, which is what this sprint's cells (mine
included) have been using. `mma.cuh` is included by 10 files, several of them on the q3/kvarn
path, and the guard's own comment claims the family is "unused by the HIP SIMT route" — the
mma.sync part is indeed excluded, but ldmatrix is not.

**Does this invalidate the 12-file inventory above? No — re-measured, and the answer is
recorded rather than assumed.** syntax-only vs `-c` codegen agree on all 11 classified files
(3 CLEAN stay CLEAN; 8 RED stay RED, mma/l'constraint/fp4 errors reached first). So the list
stands. What does change is what a "0 errors" line in any of our cells is entitled to mean:
parsing, not codegen. Instrument added: `tools/v340l/hip_codegen_probe.sh` (exit 0 clean /
4 syntax-green-but-codegen-red / 5 inconclusive), validated three ways — it fires on the real
ldmatrix delta, stays silent on two known-clean files, and on an ordinary-red file reports
"codegen not reached" instead of inventing a delta.

Remedy is not mine to apply (ops/common, and HipSources already marks the mma family as
needing a fresh exception ruling): either extend the :33 guard upward to cover :7-31, or give
the four an explicit `#error`/SIMT fallback so the HIP lane fails at preprocessing rather than
at the back end.
