# 08 — Three-mode real-build cell: SPEC for gemini's lane (WO-07 G3 / S4)

**Author:** agent5 (C441) · **Date:** 2026-09-12 · **Status:** SPEC ONLY — I do not install gates.
**Audience:** gemini (`tools/ops/gate*`, `run_ci_amd.sh`, `tests/**` are your lane per WO-07 §4).
**Route:** delivered via coordinator. Every number here is MEASURED on this host at
`amd/wo-gfx900-perm`, with the command that produces it, and the tree it was measured against.
**Cost to build this spec:** zero device time, no grant.

---

## 1. Why this cell has to exist at all (the hole, not the wish)

> **Naming note, because a whole exchange was wasted on it:** "Cell 1" is hub shorthand used in the
> agent2/agent5 traffic, **not** a label in `MERGE_GATE_BUILD_CELL_proposal_to_gemini.md` — verified:
> the string "cell 1" appears **0 times** in that file, which organizes instead as "The hole" /
> "The actual defect that slipped through" / "Suggested cells, cheapest first". Two lanes argued about
> whether "Cell 1" was green while meaning different sections. If you take this spec forward, cite the
> proposal's own headings or quote the command, never the nickname.

Every gate this line currently owns is either an **isolated probe** or a **git diff**:

| mechanism | what it can see | what it missed |
|---|---|---|
| ISA tripwire on a probe (`syncwarp_isa_receipts.sh`) | one primitive's lowering | the promote-carried `#if !defined(__CUDACC__)` guard |
| falsifier reproduction | a named construct | ditto |
| step-0 zero-diff (`check_anti_resurrection.sh`) | pure git content | ditto |

`amd/main @ 2c4b8110` shipped **20 real compile errors** in `embed_gather.cu` while **three gates
were green**. That is the argument, and it is agent2's, from the source of record
(`MERGE_GATE_BUILD_CELL_proposal_to_gemini.md`, verified present at `c6d2b617` and at tip
`d6584756`). A build cell is the only mechanism that sees the tree the way the *builder* does.

**Correction to that framing which I verified myself and gemini must not trip over:** the cited
defect is **already fixed** on `amd/main`. `f5035616` added `&& !defined(__HIPCC__)` to *both*
`q3_rowsplit_storage.h:24` and `q2_rowsplit_storage.h:26`, and it is a **descendant** of `2c4b8110`
(`git merge-base --is-ancestor 2c4b8110 f5035616` → YES — checked, because the reverse would have
been an anti-resurrection-rule violation rather than a fix). **So a build cell run at current tip
comes up GREEN. That is the fix landing, not a broken cell.** Say it in the cell's header or someone
will debug a passing gate.

> ### ⚠ CORRECTION TO THIS PARAGRAPH (verified 2026-09-12 ~22:2xZ, agent5 re-measured after agent2's hub #212)
> "Already fixed / comes up GREEN" is **only true for the include path the build actually uses.**
> Measured at `origin/amd/main @ cccc0d5e`, standalone `#include` of the header in a
> `-x hip --offload-arch=gfx900` TU:
>
> | include path | q3 | q2 |
> |---|---|---|
> | **bare** `#include "ops/linear/q3/q3_rowsplit_storage.h"` | **2 errors** — `unknown type name '__forceinline__'` at :42, :49 | **1 error**, same text |
> | via the shim / through `q3_rowsplit_gemv.cuh` | 0 errors | 0 errors |
> | **`ops/kernel/embed_gather.cuh` entered bare** (a REAL consumer, cited by agent2's own proposal) | **5 errors** — `__forceinline__` ×2 (q3:42,:49) + `__shfl_sync` undeclared ×3 (`embed_gather.cuh:136,:182,:275`) | — |
> | whitelisted `ops/launcher/embed_gather.cu` | 0 errors — **but by ordering, not merit** (see below) | — |
>
> **This is not a future risk.** Proven by the include tree (`-H`), not by reasoning: the shim's
> `cuda_runtime.h` is reached at tree line **5**, while `embed_gather.cuh` is entered at line **104**
> and `q3_rowsplit_storage.h` at **105**. So the whitelisted launcher is green **because of include
> ORDER** — an accident, not a property of the headers. Reorder those includes, or enter the `.cuh`
> from anything that does not already pull the shim, and the build breaks with a green gate above it.
> Reproduce: `bash tools/v340l/wo07_bare_header_receipts.sh` (has a `LIVE-CONSUMER` row that asserts
> the tree-order relation, so the claim degrades loudly if someone "fixes" the order by accident).
>
> **Two defect classes, one root:** `__forceinline__` (a type macro) and `__shfl_sync` (a shim
> *function*) are both CUDA-provided names used without including anything that declares them. nvcc
> predefines the first and ships the second in its own headers; on the HIP lane both arrive only via
> `hip/amd_detail/host_defines.h` and our shim's `cuda_runtime.h`.
>
> Root cause, confirmed from the file: `q3_rowsplit_storage.h` includes only `<cstddef>`/`<cstdint>`
> yet uses `__host__ __device__ __forceinline__` on its own definitions. Under nvcc those three names
> are predefined; under hipcc they come from `hip/amd_detail/host_defines.h`, which this header never
> reaches — and `f5035616`'s guard now correctly **stands down** for `__HIPCC__`, so nothing
> neutralizes them either. **The fix converted a build break into a latent one** rather than closing
> the class: green today only because every current consumer enters through a `.cuh` that has already
> pulled the shim.
>
> **Consequence for this spec, and it is the strongest argument in the document:** "cell 1 comes up
> green" is uninformative without naming **which include path was compiled**. A cell that includes via
> the `.cuh` passes while the header is broken standalone — and the live build is green for the same
> accidental reason, so a future host-side HIP consumer of the layout constants or the decoder
> inherits a build break with a passing gate above it. So M2 must carry **both** rows: bare-header
> and via-consumer, and **the cell must name its entry point** — a chain-include row can only ever
> prove the chain's order, which is what everyone here mistook for header health. Same principle as
> check (a)/(i): green must mean the code under test was actually reached.
>
> **And a correction to my own advice in §2 below:** `-Wmacro-redefined` (which I endorsed as "the
> primary keeper") **cannot see this class at all** — nothing is redefined here, something is
> *missing*. A missing declaration needs its own probe. Endorsing the macro keeper as coverage for
> this defect would have been a second vacuous green, so the keeper set is now explicitly two
> independent probes: macro-redefinition **and** bare-header absence.
>
> **Fix shape, tested by agent2 and independently re-verified by me on q3+q2, HIP + plain-g++,
> 0 errors in every config:**
> ```cpp
> #if defined(__HIPCC__)
> #include <hip/hip_runtime.h>   // __host__ / __device__ / __forceinline__
> #endif
> ```
> Two retracted/forbidden alternatives, both measured rather than asserted: (1) `#include
> <cuda_runtime.h>` breaks the plain-g++ CPU-test lane (`cuda_runtime.h: No such file or directory`),
> which is *why* the guard exists; (2) re-adding `__HIPCC__` to the :24 guard does **not** fix it —
> measured on the real consumer chain, 20 errors vs 0 today (though the emitted errors were
> `__ocml_*` overload mismatches rather than attribute errors, so the outcome is confirmed and the
> precise mechanism is not settled; either way it is not a fix). Both files are agent3/agent4's to
> edit (Green's text, exception-pair rule) — this spec records the requirement, not the patch.

**CURRENT STATE OF THE CLASS** — revised twice; the first revision was itself wrong in the
reassuring direction (this paragraph is the one to read before trusting any "closed" claim below).

- `f5035616` correctly fixed the **redefinition** half (the `#if !defined(__CUDACC__)` neutralizers no
  longer stand on top of ROCm's own `__device__` under hipcc) and did so for **both** q3 and q2 in the
  same commit — so "q2 is latent" was wrong, and that part of my earlier note stands.
- It did **not** close the class, and the file is **not** green standalone: bare
  `#include "ops/linear/q3/q3_rowsplit_storage.h"` in a `-x hip --offload-arch=gfx900` TU is
  **2 errors** at :42/:49 (`unknown type name '__forceinline__'`), q2 **1 error** — re-measured at
  `origin/amd/main @ cccc0d5e`. See §1's correction block for cause, both-rows requirement, and the
  tested fix.
- **I also propagated a bad provenance claim**: an earlier version of this paragraph asserted the
  coordinator "verified the **bare-include** case green". The measurement is the opposite — bare
  include is RED. Whatever was verified green was a via-consumer path, and I passed the word along
  without re-measuring it. Third-party-positive claims about *another lane's verification* need the
  same re-measurement as my own; a shared wrong conclusion is worse than two independent ones.
- `494c9220`-era (agent2, parked): `cuda_pipeline.h`/`memory.cuh` **self-sufficiency** repair — the
  same class as this one (a header using `__forceinline__` without including its definer), which is
  why the pattern, not the file, is the thing to gate.

So the practical warning for a cell author, sharpened: the historical shas are green **on the include
paths the build uses**, so hunting a red at a `2c4b8110`-class sha finds none and reads as a broken
cell — but "green on the used path" is exactly what hid *this* defect, and the tree-order check shows
the used path is itself an accident. Synthetic injection (§2) plus a **bare-header row** (§3 M2) are
both required; neither alone is a control.

## 2. THE CENTRAL DESIGN CONSTRAINT: negative controls must be SYNTHETIC, never a historical sha

This is the one thing I'd ask gemini not to rediscover. Every real defect the merge-gate story is
told around is now **closed at every tip**, so *pointing a control at a bad commit yields green, and
a green negative control reads as a broken cell*:

| defect people cite | status at tip |
|---|---|
| `__CUDACC__` rowsplit guard | fixed by `f5035616` (q3 **and** q2) |
| `__syncwarp` host break | fixed by agent2's `be4c394d` line |
| PG-1 check-(a) 7+2 files | fixed by gemini's own `2e1a98e7` |

So each mode below specifies a **synthetic injected violation** in a temp tree — and, following the
lesson of my own S3 retraction (§8c of the 07 doc), **a control that cannot fire is not a control**.
The cell must assert *direction*, not merely *value*: mode N must go RED on the injected violation
and GREEN without it, in the same run.

## 3. The three modes (each with its own negative control)

Flags are not hand-written — they are **scraped from `build-hip/compile_commands.json`**, which is
the only way to guarantee the probe sees the include order the build sees. My G1 census learned this
the hard way: a wrapper TU missing `<cuda_runtime.h>` produced **21 phantom REDs** that all blamed
`hip_shim/cuda_pipeline.h` instead of the file under test.

### M1 — pure-host `.cpp` TU (the mode that has caught things nobody's probe could see)
```sh
g++ -fsyntax-only -D__HIP_PLATFORM_AMD__ -I/opt/rocm-6.2.0/include \
    -I$SRC/src/common/hip_shim -I$SRC/src -I$SRC/include <whitelisted .cpp>
```
Why: `HipSources.cmake` whitelists `.cpp` files compiled by plain **g++**, and `hip_runtime.h` is
written to be host-includable — verified: `__device__ int probe(){return 1;}` passes M1 with rc=0,
because ROCm's headers neutralise the attributes on the host path. That is exactly why a shim defect
can be *invisible to `-x hip` and fatal here*.
**Negative control (synthetic):** add `__device__ void bad(){ __builtin_amdgcn_wave_barrier(); }` to
a scratch header reached from a whitelisted `.cpp`; M1 must go RED. (This is the `__syncwarp` class.)

### M2 — `-x hip`, BOTH passes, one driver invocation
```sh
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -O2 -x hip --offload-arch=gfx900 \
    <scraped -D/-I flags> -fsyntax-only <probe.cu>
```
**PASS LABELLING IS MANDATORY AND IS FREE:** the driver stamps each diagnostic
(`… 1 error generated when compiling for host.`). Capture the stamp; never merge host-pass and
device-pass counts into one number — their defect texts differ and the merge is what hid things.
**GUARD RULE (agent2's, inherited verbatim):** distinguish passes with **`__HIPCC__`**, never
`__HIP_DEVICE_COMPILE__` — the latter is *undefined in the host pass of a `-x hip` compile*, so
guarding with it breaks valid device code and manufactures a second break.
**Negative controls (synthetic, two — they are different classes):**
1. attribute-neutering: `#if !defined(__CUDACC__)` + `#define __device__` in a header reached from a
   device TU → must fire, and it must fire **as warnings at the include**, which is why M2 should
   also run `-Werror=macro-redefined` (that is PG-1 check (g) already; keep it, it is the cheap
   keeper for the whole host-or-nvcc-assumption class and it covers q2-style latent cases).
2. split-group shuffle: the `if (lane < 3) x = warp_reduce_sum<8>(x);` shape from §8c of the 07 doc.
   ⚠ **M2 will NOT catch this one** — it compiles clean; only CFG/ISA analysis sees it. Listed so
   nobody believes M2 covers the shuffle-hazard class. Scope honesty beats a confident green.

   **Update from the follow-on work (07 doc sections 12-13):** the ISA analysis turned out to need
   *predicate typing*, not deeper CFG -- a lane group can only be split by a predicate over the
   **lane index**, so row/pointer/float-data guards are out of class entirely. That took the residue
   16 -> 2 at **0 hazards** across 14 whitelisted device TUs. The surviving 2 are compares against a
   LOADED SCALAR with `(tid >> 6)`, i.e. a row index -- **no static immediate exists**, so neither a
   build cell nor any dominator tree can certify them, ever. The durable scoping statement for this
   spec: **a real-build cell cannot promise anything about cross-lane group integrity, at any
   depth**; the only check that can is the source-level `32 % W == 0` assertion with `Width` resolved
   through constexpr. Not a reason to skip M2 -- M2 still catches the §3b class, which is why it
   exists.

### M3 — the full real build (the only mode that sees link and ODR)
```sh
export PATH=/home/chris/opt/cmake/bin:$PATH          # cmake is OFF PATH on this host (3.30.5)
cmake -B build-hip -DNINFER_BACKEND=hip -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 \
      -DBUILD_TESTING=OFF -DNINFER_BUILD_APPS=OFF
cmake --build build-hip -j
```
Measured, not estimated: `build-hip` = **37 MB**, so **M3 is not a >5 GB build** and the disk gate
in AGENTS.md does not bind it. Current `/` free = **27–28 G** (measured 2026-09-12; *below* the
38–44 G WO-07 §4 assumes, because `artifacts/qwen3_8_27b.ninfer` is 20.3 GB and
`~/Desktop/qwen3_8_27b_q3.ninfer` 6.6 GB — cite before any lane adds another artifact).
**Negative control (synthetic):** reference an undefined symbol from a whitelisted `.cu` → M3 must
fail at LINK while M1/M2 both stay green. That is M3's whole reason to exist; if the control doesn't
fire, the mode is decoration.

## 4. Environment traps that make the cell fail for the WRONG reason on a cold box

All measured on this host; bake them in or the first run is a false red:

1. **cmake off PATH** — `/home/chris/opt/cmake/bin/cmake`, version 3.30.5. `which cmake` is empty.
2. **`-DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0` required** (`/opt/rocm` symlinks there; no other version).
3. **`-DNINFER_BUILD_MEDIA_ACQUIRE=OFF` IS SILENTLY IGNORED.** `CMakeLists.txt:78-80` uses `set()`,
   not `option()` (`set(NINFER_BUILD_MEDIA_ACQUIRE OFF)` then a conditional `set(... ON)`), so a
   `-D` cache entry is overwritten, not honoured. To shrink the build clear
   **`-DBUILD_TESTING=OFF -DNINFER_BUILD_APPS=OFF`** instead — the two inputs the `if()` reads.
3b. **MEASURED EXHIBIT for why M1 and M2 are both mandatory (coordinator asked me to cite it):**
   `src/ops/common/memory.cuh` is WHITELISTED and includes `<cuda_pipeline.h>` at :3 before
   `<cuda_runtime.h>` at :4, and `cuda_pipeline.h` uses `__device__ __forceinline__` while including
   only `<cstddef>/<cstdint>/<cstring>` (`__forceinline__` actually arrives via ROCm's
   `hip/amd_detail/host_defines.h:156/176`, reached *through* the shim). A bare
   `#include "ops/common/memory.cuh"` TU is **GREEN under M1 (plain g++) and RED under M2 (`-x hip`,
   4 errors: `unknown type name '__forceinline__'` at cuda_pipeline.h:16,33,34 + a resulting
   no-matching-function at memory.cuh:114)**. It builds today only because every current consumer
   happens to include the shim earlier. So a host-only cell -- the cheap, thorough-sounding one --
   PASSES this, and only the HIP pass sees it. Routed to agent2 for the fix (their file); the
   three-mode design is what makes it visible at all. Reproduce:
   `bash tools/v340l/wo07_g1_fulltu_receipts.sh` (row `nvfp4_hadamard_d256.cuh`) or the minimal TU
   in `docs/amd/v340l/07_...md` §10a.

4. **Link object archives with `-L`/`-l`, never positionally.** hipcc/clang parses a positional
   `.a`/`.o` as a *source file* (agent2's finding; it is a link-stage red that looks like a toolchain
   bug).
5. **`.cu` needs explicit `LANGUAGE HIP`** — `HipSources.cmake:208-215` and its comment: a HIP-only
   project *silently drops* `.cu` sources (their measured case, quoted in the source: "the first two
   'green' HIP builds produced an archive with 16/21 objects and no error"). The parity guard
   (`src/CheckHipArchive.cmake`, wired at `HipSources.cmake:237-243`, 164/164) is the backstop —
   **keep it inside M3, don't duplicate it as a fourth mode.**

## 5. What I'd ask the runner to report (from my S1 structural finding)

`gate_pg1_whitelist.sh` runs checks (a)…(j) sequentially with 19 unconditional `exit 1` paths, so a
red (a) means **(b)…(j) never execute** — measured at `f4cf9a82`: (a) RED 7 + (b) RED 2 hid the
other eight, including (g)/(h)/(i)/(j). The CI summary printed `Failed Stages: 1`, which is an
undercount **by construction**.

So: **print a per-check `reached / not-reached` line**, and have the build cell report the same way.
This generalises past PG-1 — any sequential gate with early exits masks its own coverage. Worth a
standing rule rather than a fix to one file, and I raise it because the merge-gate practice
("every merge re-verifies") is only as strong as the checks that actually *ran*.

## 6. Acceptance test — the cell must pass this before it is trusted, not after

1. Unmodified tree → M1, M2, M3 all GREEN. (Expected today; see §1's "already fixed" note — **and
   its correction: M2 must be green for BOTH the bare-header and via-consumer include paths, or the
   acceptance test is vacuous.**) 
2. Inject each control of §3 **one at a time** → the mode that owns it goes RED, the others stay
   GREEN. A control that reddens everything is not discriminating, it's just broken.
3. Remove the injection → GREEN again (proves no state leaked).
4. **Zero-GPU run must not require a grant.** All three modes are compile/link only. If the cell ever
   wants a card, it has stopped being this cell.
5. The cell must state its **blind spot** in its own header (§3 M2 control 2 is the example): a gate
   that claims to cover a class it cannot see is check (e) again.

## 7. Explicit non-goals / open items (so nobody reads this as more than it is)

- **Not implemented.** §4 of WO-07: gates and CI are gemini's lane. I staged nothing under
  `tools/ops/`, `tests/`, or `src/`.
- M1/M2 flag sets here are indicative; **scrape from `compile_commands.json`** at install time.
- **Not covered by this spec:** the split-group shuffle hazard (needs CFG dominator analysis — see
  07 doc §8d) and D3 bf16-emulation detection (needs an emulation fingerprint; both current D3
  gates are green on a real violation, 07 doc §2). Both are gate-grade work beyond a build cell.
- The three-mode list is agent2's, inherited not invented (WO-07 §5: reuse the receipts pattern).
  My contribution is §2 (synthetic controls, with the reason), §4 (measured traps), §5 (masked-check
  reporting), and the honesty bounds in §3/§6.5.


---

## 8. Cold-reader handoff — WO-07 S1..S4 in one command list

Everything below is zero-GPU. Run from any worktree at `amd/wo-gfx900-perm` ≥ `02ad9ad7`.

```sh
cd /home/chris/worktrees/amd-wo-gfx900-perm

# 0. the authoritative gate (must be EXIT 0 on a good tree; it is today)
bash tools/ops/gate_pg1_whitelist.sh; echo "PG-1 EXIT=$?"

# 1. S1 -- PG-1 check-by-check census + the D3 gate-hole negative control
bash tools/v340l/wo07_pg1_check_census_receipts.sh

# 2. S2/S3 -- site census, constexpr-width discovery, per-file pass-labelled compiles,
#    in-build sub-group ISA cell
bash tools/v340l/wo07_g1_shuffle_census_receipts.sh

# 3. S3 -- validated tripwires (self-test ABORTS with exit 3 if the control stops firing)
bash tools/v340l/wo07_g2_isa_tripwires_receipts.sh

# 4. S3b -- the EXEC-granularity discriminator (the one that survived its own control)
bash tools/v340l/wo07_g2b_exec_granularity_receipts.sh
```

Read order: `07_...census_agent5.md` (§1 gate masking, §2 D3 hole, §3a the constexpr-width
correction, §8c the retracted gate) then `08_...` (this spec). Prior findings live in
`results/amd/wo07_s1/` as raw logs.

**What is NOT done (S5 remaining, honest list):**
- G1 full-TU compiles for the 12 compile-capable unwhitelisted files (I ran wrapper-TU probes;
  full-TU needs the coordinator's answer on cost vs the 27 G disk floor).
- The CFG-dominator cell that would actually gate the split-group hazard (07 doc §8d, part 2).
  Sized beyond a grep; needs a decision on whether it is mine as a probe or gemini's as a gate.
- Classification of the 10 RED-FILE rows into enablement/new-defect — largely done inline in
  07 §4b (all enablement, one trivial `__maxnreg__` route), but not yet signed off by the
  owning lanes.
- Nothing in this lane has touched a device; the permanence claims here are compile/ISA-level
  by construction and are labelled as such.
