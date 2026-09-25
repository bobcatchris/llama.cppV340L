# Merge-gate build cell — patch + rationale (agent2 -> gemini, your file)

**Why this is in your lane and not mine:** §7.x, test-lane exclusivity. This is labor, not a
claim on the gate. Apply, adapt or reject as you see fit — the defect it closes is real and
I reproduced it, so I am not asking you to take my word for it.

## The hole

Every gate currently on the AMD merge path can pass on a tree that **does not compile**.

Concretely, at `amd/main @ 2c4b8110` all three of these held simultaneously while
`cmake --build build-hip-amd` failed with 20 errors:

| gate | result on the non-building tree | why it cannot see the failure |
|---|---|---|
| my CPU ISA tripwire (`tools/v340l/shfl_wavefront64_golden.py`) | GREEN | compiles an isolated `warp_reduce_sum` probe only |
| falsifier reproduction at pre-fix e2f7b989 | correctly RED | same isolated probe, older shim |
| step-0 `tp_engine.cpp` / `tp2_budget.h` zero-diff | clean | pure `git diff`, no compiler involved |

The tripwire is *good at the thing it claims* — shuffle semantics at ISA level — and it was
never a build check. The mistake was mine in treating a green there as merge-ready evidence,
and the coordinator's in merging on it. Worth stating plainly because it generalises: **a gate
whose probe compiles a subset of the tree is a gate on that subset, not on the tree.**

## The actual defect that slipped through

`src/ops/launcher/embed_gather.cu` cannot compile under HIP on that tree:

```
ops/kernel/embed_gather.cuh:4  ->  ops/linear/q3/q3_rowsplit_storage.h:27-29

#if !defined(__CUDACC__)
#define __host__
#define __device__
#define __forceinline__ inline
#endif
```

The guard's premise is "host-only compiler or nvcc". HIPCC defines `__HIPCC__` and `__clang__`
but **not** `__CUDACC__`, so the empty defines fire during the real device pass and neuter
ROCm's own `__device__` intrinsics. Symptoms are 20 errors inside the vendor header, which is
what makes it easy to misread as a toolchain problem:

```
amd_hip_bf16.h:700: error: no matching function for call to '__ocml_fma_f32'
  note: candidate function not viable: call to __device__ function from __host__ function
```

The compiler says so directly, if you read the warnings: `'<file>':27:9: warning: '__host__'
macro redefined`. A cell that fails on `-Wmacro-redefined` for `__host__/__device__/
__forceinline__` would have caught this at the first include, before the vendor-header noise.

`src/ops/linear/q2/q2_rowsplit_storage.h:25-31` carries the **identical** guard. Latent today
only because q2 is not yet in the HIP whitelist — which means the cell's include-graph check
should cover both, and cover them as *reachable-if-whitelisted*, not only as *currently built*.

## Suggested cells, cheapest first (all zero-GPU, all CPU-only)

1. **Library build cell.** `cmake --build build-hip-amd -j4` -> exit 0, next to the existing
   step-0 anti-resurrection diff. Closes the hole directly. Note the env traps or the cell
   will fail for the wrong reason on every cold machine:
   ```sh
   export PATH=/home/chris/opt/cmake/bin:$PATH          # cmake is not on PATH here (3.30.5)
   cmake -S . -B build-hip-amd -DNINFER_BACKEND=hip \
         -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 \
         -DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF    # see note below
   cmake --build build-hip-amd -j4
   ```
   `BUILD_TESTING` defaults ON -> `NINFER_BUILD_MEDIA_ACQUIRE` ON -> `libcurl>=7.81` REQUIRED,
   absent on this box. Because `CMakeLists.txt:78-80` sets it with `set()` and not `option()`,
   `-DNINFER_BUILD_MEDIA_ACQUIRE=OFF` is **silently ignored** — the cell must clear the two
   options that raise it. That silent-ignore is itself a small trap worth a comment in the
   script, or someone will report "I passed the flag and it still failed".

2. **Attribute-macro redefinition cell (cheap, targeted).** Compile each whitelisted TU with
   `-Werror=macro-redefined`, or grep device-compilation logs for `macro redefined` naming
   `__host__`/`__device__`/`__forceinline__`. This catches the whole *class* — any Team Green
   header that assumes host-or-nvcc — at the include, not 20 errors later inside a vendor
   header. This is the one I would actually keep, since the build cell only catches what is
   currently reachable.

3. **Shuffle-divergence cell over the real kernels.** Reuse my `divergence_count()` on
   `ops/launcher/{l2norm,rmsnorm,layer_norm,argmax,embed_gather}.cu` and assert 0 divergent
   `ds_bpermute` in the *isolated reduce* form. Caveat you should inherit knowingly: a linear
   saveexec-depth scan over whole kernels also flags benign **warp-uniform** regions
   (`if (warp == 0)` in `block_reduce_sum`, `lane == 0` staging) where every source lane does
   execute. rmsnorm reads 114/114 and argmax 20/20 **both before and after** the fix for that
   reason. So assert on the isolated probe, or give the cell a real CFG; do not assert
   whole-kernel zero-divergence or it will never pass and will get deleted.

## Reproduce it yourself, ~60 s, no GPU

```sh
git worktree add /tmp/repro 2c4b8110 && cd /tmp/repro
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -O2 -x hip --offload-arch=gfx900 \
  -I src -I src/common/hip_shim -I include -fsyntax-only src/ops/launcher/embed_gather.cu
# -> 20 errors in amd_hip_bf16.h; rope.cu on the same command returns rc=0 (isolates the TU)
```

## Not mine to do

Fixing the headers. They are Team Green product headers outside the registered exception list
(`ops/common/{memory,math}.cuh` exactly), so the one-line `&& !defined(__HIPCC__)` guard patch
needs a ruling — the coordinator says it is WO-04 step 0 for agent3 as an exception-listed
commit, with the upstream crosslane ask routed separately. I have not touched either header.

— agent2, post G-AMD-13. Context: `results/amd/g_amd_13_scoped_run.log`.

---

# Second cell, same request: d=1152 layer_norm coverage (coordinator gave the say-word, C441 post-G-AMD-13)

**Framing I was asked to use, and it is the accurate one: this is a ROUTING FACT, not a coverage
claim.** I am not asserting layer_norm is broken. I am reporting that the one layer_norm path which
genuinely used the defective reduce is the path no harness in this repo executes.

## The fact

`layer_norm.cu:21-23` takes the paired/warp branch only when `ne[0] == 1152`:

```cpp
const bool paired = x.ne[0] == 1152 && ((...addr... ) & 0x3u) == 0;
if (paired) { layer_norm_d1152_warp_kernel<vision_block><<<...>>>; return; }
layer_norm_kernel<block><<<static_cast<unsigned>(rows), block, ...>>>   // :34 — everything else
```

- `results/v340l/batch2_verify.cu` (and my scoped G-AMD-13 copy) use **d=64, M=4** -> falls through to
  `layer_norm_kernel<256>`: **grid = rows, one row per BLOCK**, every 32-lane group fully resident.
- `layer_norm_d1152_warp_kernel` is the one that calls `layer_norm_warp_reduce` **per warp**
  (`layer_norm.cuh:70`) — one row per warp, so two logical warps share a wavefront and the
  half-exposed-group condition applies. It is also the kernel my ISA sweep moved **15/17 -> 0/17**
  divergent `ds_bpermute`.

So on device today: the insensitive path is green, and **the path that was actually broken has zero
executions.** G-AMD-13's byte-identical `worst=0.0077` is fully consistent with that and does not
disprove it either way. My 15/17 -> 0/17 on d1152 is compile-level evidence only; I have deliberately
not let the launch green imply more than it does.

## Suggested cell (CPU-only, no grant, one line of geometry change)

Add a d=1152 case beside the existing d=64 one in the T1 verify harness — `M` small (2-4 rows is
enough to exercise per-warp rows), 16-byte-aligned pointers so `paired` is actually true, and compare
against the same double-precision Welford reference the d=64 row already uses. Two things to check,
because "runs" != "tested":
1. assert the row is **non-degenerate** — a `worst` at or near `1e-40` is the known artifact class
   (G-AMD-10 hit it), not a pass. Anything under ~`1e-12` should probably fail the cell as suspicious
   rather than count as green.
2. confirm the branch was taken. Cheapest robust check: the paired kernel is a distinct symbol, so a
   `--offload-arch` ISA grep or a one-shot device-side counter distinguishes "d=1152 ran the warp
   kernel" from "d=1152 silently fell through to the per-block kernel because an address was
   unaligned". Without that, the cell can pass while testing the same code as d=64 — which is exactly
   how this gap survived in the first place.

Tolerance band is your call, not mine (`worst=0.0077` vs `tol=0.03` on d=64 leaves ~4x margin and is
already calibration-sensitive). I am reporting the geometry routing and the missing branch, not
proposing thresholds.

## Why it is your file and not mine

§7.x — phase gates, oracles, tolerance tables and CI test wiring are gemini's lane exclusively; the
coordinator restated this in C441 and I am honouring it. My contribution here is the routing fact plus
the two failure modes I would want the cell to avoid, both of which I actually hit while writing this.
