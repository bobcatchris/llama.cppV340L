# 07 — Gfx900 permanence census (WO-07 S1+S2): shuffle/wavefront site audit, PG-1 gate structure, and the D3 gate hole

**Author:** agent5 · **Requested by:** C441 (WO-07) · **Date:** 2026-09-12
**Tree:** `amd/wo-gfx900-perm` (base `amd/main`), S1 measured at `f4cf9a82`, re-baselined and S2 measured at `891fd93d`/lane `578958bd`.
**Cost:** ZERO device time, no grant used or requested. All results are `-fsyntax-only` /
`-E -M` / asm-emission outputs on the host toolchain.
**Reproduce:** `bash tools/v340l/wo07_pg1_check_census_receipts.sh` (gate structure + D3 hole) and
`bash tools/v340l/wo07_g1_shuffle_census_receipts.sh` (site census + per-file compiles).
Machine-readable outputs archived in `results/amd/wo07_s1/`.

> Read-the-matched-line law applied to my own numbers throughout: where a figure here disagrees
> with a figure in circulation, BOTH are stated and the disagreement is the finding, not an error
> to smooth over.

---

## 1. S1 — PG-1 gate structure: a red check (a) silently un-ran eight other checks

`tools/ops/gate_pg1_whitelist.sh` runs checks (a)…(j) **sequentially**, and its failure paths are
unconditional `exit 1` (19 such statements in the file). At `amd/main @ f4cf9a82`:

* check (a) → RED, 7 files; check (b) → RED, 2 files.
* **Checks (c)–(j) never executed.** That set includes the four gates WO-07 exists to protect:
  (g) macro-redefined, (h) spin-guard, (i) D3-ISA fingerprint, (j) d=1152 kernel emission.

So CI's summary line (`Failed Stages: 1`) is an **undercount by construction**, and a green
"Stage 0" on a tree where (a) is red certifies nothing about the checks behind it. The whole line
runs on the premise that "every merge re-verifies" (COORDINATOR merge-gate practice); that premise
holds only for checks that actually **ran**.

Demonstrated both directions, same tree, ~40 minutes apart:

| `amd/main` | (a) | (b) | (c)–(j) | real in-tree gate |
|---|---|---|---|---|
| `f4cf9a82` (S1 base) | **RED 7** | **RED 2** | **never executed** | EXIT 1 |
| `e1ea5dbb` (after gemini `2e1a98e7`) | PASS | PASS | all ran | **EXIT 0**, all 10 check headers reached |

The re-baseline was not my edit: gemini's `2e1a98e7` ("expand PG-1 exceptions for one_shot headers
and T3 attention") landed the exact 7+2 files during my census. Merged as `578958bd`; `src/`+`tools/`
now 0-diff vs `amd/main`; anti-resurrection (d) PASS = **no reversion**.

**The 7 (a)-RED files, classified by reading each diff** (not by the count):

| file | character |
|---|---|
| `ops/common/mma.cuh`, `ops/kernel/gqa_attention_{decode,prefill}_bf16.cuh`, `ops/launcher/gqa_attention_{decode,prefill}.cu` | `__HIP__`-guarded T3 adoptions (29 and 25 guard lines in the two launchers) |
| `core/multi_gpu/one_shot_allreduce.h`, `one_shot_argmax.h` | **pure additive API** (`last_call_timed_out` declarations), zero CUDA logic touched, 0 guard lines *because none are needed* |

→ registration gap, not new defect. Independently consistent with the "check-(a) `.h`-pair" item
already open on gemini. **Not mine to fix** (§4): the gate is gemini's lane, and I staged nothing
under `tools/ops/`.

**Suggested shape (ruling belongs to you/gemini, not me):** have the runner print a per-check
`reached / not-reached` line so a masked gate is visible in the summary. A green stage that ran
2 of 10 checks is the same failure class as the isolated-probe gates this WO exists to replace.

---

## 2. The D3 gate hole, independently reproduced — both D3 gates are green on a real violation

agent2's v340l/03 (b) found the D3 "compiler-enforced" premise false. **Confirmed on my own bytes,
and it is worse than "the premise is wrong": two shipped gates actively certify the violation.**

| probe (gfx900, `-O3 -x hip`) | result |
|---|---|
| check (e)'s exact spelling: scalar `__hadd` on `__hip_bfloat162` | rc=1 — `no matching function for call to '__hadd'` |
| `__hadd2` / `__hmul2` (the 2-vec spellings the gate never tries) | **rc=0, compiles CLEAN** |
| emitted asm for those | `__ocml_*` = **0**, native bf16 VALU = **0**, `v_add_f32`/`v_mul_f32` = **4** |

Why this defeats *both* gates:

* **(e)** fails on **overload resolution**, not hardware legality — scalar `__hadd` simply has no
  2-vec overload. Its PASS is a statement about one spelling. The gate's own comment says "Cell goes
  RED if bf16 arithmetic ever succeeds"; the 2-vec forms already succeed.
* **(i)** asserts *zero `__ocml_`* **and** *presence of `v_fma_f32`/`v_add_f32`* as **correctness**.
  The silently-emulated path produces exactly that signature, because bf16→fp32 unpack/fp32-add/
  repack never calls libm. So a D3 violation **satisfies (i)'s assertion** — (i) certifies
  "not `__ocml`", not "not emulated".

This is a **negative control that fires** (§G2's 10/11 discipline): the detector discriminates, so
the two greens are not decoration-by-accident, they are decoration-by-construction.

**Fix shapes** (gemini's files; routed, not implemented): (e) → emit a **spelling matrix** as data
and gate on the 2-vec forms, per your own 16:0xZ note; (i) → add an fp32-unpack/repack *emulation*
fingerprint (bf16 load + fp32 ALU + repack, e.g. `v_bfi`/`v_cndmask` RNE) instead of relying on
`__ocml_*` alone.

---

## 3. S2 — the site census does not reproduce the figures in WO-07's header

Measured on the tree, from bytes, with the in-build set taken as **TU list + transitive
`-E -M` header closure** (480 files, computed from `build-hip/compile_commands.json`, not from a
hand-maintained list):

| quantity | WO-07 header (cached) | **measured here** |
|---|---|---|
| `__shfl_*` call sites, all of `src/` | 27 "in-build" | **54** across **17** files |
| … of which genuinely **in-build** | — | **11** across **4** files |
| … of which unwhitelisted | 12 | **43** sites across **13** files |
| `warp_<op><N<32>` sub-group uses | 28 | **72** across **18** files |

Candidate explanations, **both explicitly unverified**: (1) the cached figures predate T3's
donor-adoption, which added kernel files carrying shuffles (v340l/03 names its own baseline as
`amd/main @ b5b90bcc`; at that sha my same grep yields 65 lines / 50 files, so the discrepancy is
not *simply* drift either); (2) a different unit was counted (grep lines vs call occurrences vs
reduce-helper *callers* — note `in-build callers of the reduce helpers: 12` appears in v340l/03,
which is plausibly where a "12" came from). I could not reproduce 27 or 28 under any counting rule
I tried, including agent2's own narrower file set (which yields 44, not 28).

**Consequence:** the "12 sites / 27 in-build / 28 sub-group" figures should not be used as a
census gate, and no lane should treat reaching them as completion. The **file list** in §4 is the
stable object; the scalar counts are not.

**What survives the disagreement — and the load-bearing part, WITH ONE CORRECTION I MADE HERE:**
> Of the **72 literal-digit** `warp_<op><N<32>` sites, **0** are in the HIP build — agent2's
> observation about those 18 carrier *files* is TRUE (none is whitelisted).
>
> **BUT the stronger invariant "no sub-group reduce is in the build" is FALSE, and my first
> draft of this doc claimed it.** See §3a. The digit-only regex could not see a constexpr width.

### 3a. CORRECTION — sub-group (Width<32) reduces DO reach the HIP build, via a constexpr width

`ops/common/warp.cuh:80`, inside `block_reduce_sum`:

```cpp
constexpr int Warps = BlockSize / kWarpSize;   // :69
...
if (warp == 0) { x = warp_reduce_sum<Warps>(x); }   // :80  <-- Width = Warps, NOT a literal
```

`block_reduce_sum` is called from `ops/kernel/rmsnorm.cuh:153,199` (with `Block`, `kBlock=512`),
and **`rmsnorm.cuh` is in the HIP whitelist** (confirmed in the `-E -M` closure). The launcher
instantiates the CTA reduce **only** at `rmsnorm_cta_bf16x2_kernel<Epilogue,256,6>` and
`<Epilogue,512,8>` (`ops/launcher/rmsnorm.cu:52,59`; grep over all `src/` returns exactly these
two), so the **reachable** sub-group width set is **{8, 16}** — both `< 32`. My `<[0-9]+>` regex structurally could not match `warp_reduce_sum<Warps>`;
the same blind spot catches `warp_reduce_sum<kD1Warps>` (`sparse_moe_decode_kernels.cu:82`) and the
five `warp_sum<kWarpSize>` sites in `recurrent.cuh` (those are full-width, hence fine).

**ISA measurement at gfx900** (`-O2 -x hip`, real `warp.cuh`, per-function EXEC attribution):

| kernel | shape | `ds_bpermute` | inside narrowed EXEC |
|---|---|---|---|
| `block_reduce_sum<8>` via `if (warp==0)` (k256) | sub-group in build | 8 | **3** |
| `block_reduce_sum<512>` + `warp_reduce_sum<16>` (k512) | sub-group in build | 13 | **4** |
| `block_reduce_sum<64>` (Warps=2) | sub-group in build | 6 | **1** |
| `warp_reduce_sum<8>` **unguarded** (all lanes execute) | shim alone | 3 | **0** |

So the narrowing is REAL and reachable today, and the un-narrowed shim is clean — the 4th row is
the control that makes the first three meaningful (it shows the shim is not what narrows EXEC).

**Why this is NOT a live defect, stated as a checkable predicate rather than a reassurance.**
On wave64 the hardware wavefront is 64 lanes, while `warp = threadIdx.x / 32` is a **32-lane**
division, so `if (warp == 0)` narrows EXEC to a 32-lane *slice* of a wavefront. Each width-`W`
group is still wholly inside the executing set **iff `32 % W == 0`** — here W ∈ {2,8,16} all divide
32, so every bpermute's source lane executes. This is exactly agent2's `rmsnorm 114/114 unchanged
post-fix` observation, now with the arithmetic behind it spelled out: it is **alignment luck, not
enforcement**. Nothing in the tree asserts `32 % Warps == 0` at any call site, and a future
`BlockSize` (or a predicate narrowed below a group, e.g. `lane < 3`) breaks it silently on
hardware whose whole failure mode is a non-publishing lane.

**This is the strongest G2 gate candidate found so far**, and it is cheap and CPU-only:
assert that any `warp_reduce_sum<W>`/`warp_max<W>`/`warp_sum<W>` reachable from a whitelisted TU
satisfies `32 % W == 0` **and** that its EXEC-narrowing predicate is group-aligned. It covers the
constexpr-width case my regex missed, which a digit-only census cannot.

---

## 4. G1 per-file HIP-compilability (pass-labelled), 28 unwhitelisted shuffle-bearing files

Every row is one `-fsyntax-only` invocation with the **real build's flags scraped from
`compile_commands.json`** (11 `-I` entries, `hip_shim` first — the script *refuses to run* if the
first include dir is not the shim, because otherwise `<cuda_runtime.h>` resolves to NVIDIA's and
every row is fiction). `.cuh`/`.h` get a wrapper TU; `.cu` compile directly.

**Result: 12 COMPILES · 10 RED-FILE · 6 RED-INCL.**

### 4a. The 12 that already HIP-compile (whitelist candidates on compile grounds alone)
`core/multi_gpu/one_shot_argmax.cu` · `core/multi_gpu/tp_kernel.cu` ·
`ops/dflash2/dflash2_attention.cu` · `ops/dflash2/dflash2_topk.cu` ·
`ops/kernel/bidirectional_gqa_attention.cuh` · `ops/kernel/gqa_attention_decode_bf16.cuh` ·
`ops/kernel/gqa_attention_kvarn.cuh` · `ops/kernel/gqa_attention_prefill_bf16.cuh` ·
`ops/kernel/nvfp4_hadamard_d256.cuh` · `ops/kernel/vision_attention.cuh` ·
`ops/kvarn/kvarn_tile_cuda.cu` · `ops/sparse_moe/sparse_moe_route.cuh`

⚠ Read these as "HIP-parsable in isolation", **not** "whitelist-ready". §3 of WO-07 forbids
re-whitelisting without G1 evidence, and a wrapper TU cannot exercise template instantiation
context or caller-side `Geometry`/`Metadata` parameters. Compilability is necessary, not sufficient.

### 4b. The 16 that don't — three causes, **zero new defects**

| cause | files | classification |
|---|---|---|
| **`mma_*` undeclared** (`mma_bf16` ×4, `mma_s8` ×3, `mma_tf32` ×1) | `gqa_attention_decode_i8.cuh`, `gqa_attention_kvarn_mma.cuh`, `gqa_attention_prefill_i8.cuh`(partial), `gqa_decode_body.cuh`, `gqa_decode_slice3_i8.cuh`, `gqa_decode_slice3_i8_v2.cuh`, `gqa_decode_slice5_i4.cuh`, `gqa_decode_unified_slice2.cuh`, `…/chunked/prepare_wy_wu.cuh` | **enablement gap, already ruled.** `mma.cuh:33,87` are now `#if !defined(__HIP__)` (agent3's full-family guard, endorsed in your 17:0xZ STATE). The `mma_*` family is NVIDIA-tensor-core-only by measurement, so every consumer is unreachable on HIP *by design*. The donor SIMT route is the answer, not a shim. |
| **`cuda_fp4.h` not found** | `gqa_attention_kv_quant_nvfp4.cuh` (+ 3 dependents via RED-INCL) | **enablement gap.** Shimmed-header list deliberately excludes kernel-side NVIDIA headers (nvfp4 trimmed from HIP scope by ruling). |
| **`__maxnreg__(120)` unknown type** (`gqa_attention_prefill_i8.cuh:214-215`) | 1 site, 1 file | **new-but-trivial, and NOT a defect in the file's logic.** `__maxnreg__` is an nvcc-only kernel attribute spelling; clang-HIP parses it as a declaration missing a type specifier, then cascades (`:218 unknown type name 'Metadata'`). Tree-wide it is **exactly one site in one file, out-of-build**. If these files are ever whitelisted it needs `#if !defined(__HIP__)` or an empty shim macro. Flagging for routing only — `ops/kernel/**` is not mine. |

**6 of the 16 are RED-INCL, i.e. my wrapper's first error sits in a *dependency*, not the file
named.** Those 6 (`gqa_attention_kvarn_direct.cuh`, `gqa_attention_prefill_nvfp4.cuh`,
`gqa_decode_slice2_kernel.cuh`, `…_slice4_kvarn.cuh`, `…_slice6_kvarn_k5v4.cuh`,
`…_slice7_nvfp4.cuh`) have **no independent evidence against them** — they inherit a blocker they
don't cause. Classifying them separately is what keeps the count honest: the real per-file blocker
set is 10, not 16.

---

## 5. Two instrument errors I made, kept visible (the pattern this line says is worth more than clean numbers)

1. **My first census run tabulated 21 phantom REDs.** The wrapper TU included only the header under
   test, not `<cuda_runtime.h>` first. `hip_shim/cuda_pipeline.h` uses `__device__ __forceinline__`
   while including just `<cstddef>/<cstdint>/<cstring>`, so a bare TU fails **inside the shim** on
   the host pass (`unknown type name '__forceinline__'`). 21 of 28 files "failed" identically. The
   corrected harness (shim first, as every real whitelisted TU does) gives 12 clean compiles.
   Had I reported the first run, I would have routed ~20 defects to lanes that have none — and this
   is precisely the trap the line already names: *"treat a compile cell's first error as suspect when
   it names a header or missing symbol rather than a construct."* The tell was the **identical**
   error across 21 unrelated files.
   *Side finding from the same event:* `cuda_pipeline.h` is not self-sufficient — it depends on a
   prior include for `__device__`/`__forceinline__`. Harmless in the real build (shim `cuda_runtime.h`
   is always first) but it is what makes naive header probes lie. agent2's file; reported, not edited.
2. **`set -e` made the harness punish success.** Diagnostic-extraction greps return nonzero on a
   *clean* file, so the census aborted immediately after its first COMPILES row and printed a
   table with a header and no rows. Fixed with explicit no-match guards; `bash -n` + full re-run
   both clean.

---

## 6. Residual permanence gaps (what WO-07 cannot yet claim closed)

1. **G2 tripwires do not exist yet** for the 12 compile-capable unwhitelisted files. S3 is the
   per-class `ds_bpermute` + narrowed-EXEC divergence probe with negative controls. Until then,
   "compiles" is a *parse* guarantee with no ISA content — and §2 just showed a parse-level
   guarantee is exactly how D3 came to look compiler-enforced.
2. **No gate enforces the sub-group invariant — and the invariant needed restating (§3a).**
   The literal-digit claim is true (0/72 in build); the *semantic* claim is false, because
   `warp_reduce_sum<Warps>` reaches Width 2/8/16 into the build through `block_reduce_sum` in
   whitelisted `rmsnorm.cuh`. It is currently SAFE by `32 % W == 0` alignment luck, and nothing
   asserts that. This is the rot case WO-07 was staffed for, and it is LIVE rather than future.
3. **`__maxnreg__` has no guard and no gate** — one site today, silent landmine at whitelist time.
4. **Both D3 gates are blind to emulation** (§2) — highest-severity item here, smallest fix.
5. **Gate masking by `exit 1` ordering** (§1) — a meta-finding that applies to every future PG-1
   change, including ones not yet written.
6. **Cached census figures are unreliable** (§3) and should be retired in favour of the file list.

## 7. Status / next

S1 ✅ (measured, re-baselined, receipts committed `514a55b7`). S2 ✅ this doc §3–§4.
S3 (per-class G2 probes + negative controls) is next and is the real work; S4 = the three-mode
build-cell spec for gemini. S5 = gap classification (started in §4b).


---

## 8. S3 — G2 ISA tripwires, and the gate I had to retract before proposing it

Two scripts, both zero-GPU, both control-validated:
`tools/v340l/wo07_g2_isa_tripwires_receipts.sh` (per-class narrowed-EXEC counts) and
`tools/v340l/wo07_g2b_exec_granularity_receipts.sh` (EXEC-mask-constant analysis).

### 8a. Detector validated before use
The tripwire reconstructs the pre-fix early-return butterfly VERBATIM from the shim's own defect
text and runs it against the shipped cndmask shape:

| shape | `ds_bpermute` | narrowed-EXEC |
|---|---|---|
| pre-fix early return (should FIRE) | 5 | **5** |
| shipped cndmask shim (should be clean) | 5 | **0** |

The script **exits 3 and refuses to print greens** if the control stops firing — the failure mode
that left both D3 gates as decoration (§2). Same bpermute total on both rows, so the count of
shuffles is not the signal; their EXEC context is.

### 8b. Per-class results on the four in-build families — SUPERSEDED, see §11

The table below was built with the **count** method that §8c proves cannot discriminate. It is kept
(with its numbers) only because it is the evidence FOR the retraction: `narrowed=3` on a safe shape
is precisely the thing a count gate would have flagged. Do not use it as a verdict. §11 replaces it.

Shim-level defaults confirmed while doing this: `cuda_runtime.h:431-432` supplies the 3-arg
`__shfl_{xor,down}_sync` overloads that forward **width=32**, matching CUDA — so the three in-build
files that call without an explicit width (argmax/layer_norm/sampling) are semantically correct,
not accidentally narrowed.

### 8c. THE RETRACTION: a narrowed-EXEC COUNT IS NOT A GATE — and my own control proved it

I entered S3 intending to hand gemini a "count bpermutes under `s_and_saveexec`" gate, generalising
agent2's method. Cell 2b was built to test that proposal, and **it failed**:

```cpp
if (warp == 0) x = warp_reduce_sum<8>(x);   // SAFE:   32-lane granular, groups intact
if (lane < 3)  x = warp_reduce_sum<8>(x);   // UNSAFE: splits the width-8 group
```

Both compile; both report **`bp=8`, `narrowed=3`, same EXEC depths, byte-identical shuffle
structure**. The only difference in the whole file is the guard's comparison constant
(`v_cmp_gt_u32_e32 vcc, 32, v0` vs `vcc, 3, v5`). **A divergence-count gate passes the hazard.**
Had I shipped it, it would have been check (e) again — confident, green, blind — and I would have
been the one who did it.

What DOES discriminate is the **EXEC-mask constant** bounding each region. The g2b script extracts
it and its control pair separates cleanly: `32 → GRANULARITY-OK`, `3 → SPLIT-GROUP`.

**Honesty constraint applied to my own tool:** the constant is recovered as "nearest preceding
`v_cmp`", which is a heuristic, not dominator analysis. It is sound on the 2-kernel control and
**unsound on the real kernels**, which interleave staging compares (`lane<8`, `blockDim>>5`,
vector widths 16/28). So real-code rows print `INCONCLUSIVE (needs CFG dominator)`, never
`HAZARD`. The first run of g2b labelled `mask-const=16/28` regions in rmsnorm/argmax/sampling as
"SPLIT-GROUP HAZARD" — **phantom defects in whitelisted code that reads clean on hardware**, from
my own pairing bug, and it is the third time today that reading the matched line rather than the
count was the only thing standing between me and a false route to another lane.

### 8d. Revised recommendation for gemini (supersedes my §3a "32 % W == 0" proposal)
A gate-grade cell needs **both** parts; part (a) alone demonstrably does not catch a split group:
1. resolve `Width` through **constexpr** for every `warp_reduce_sum<W>`/`warp_max<W>`/`warp_sum<W>`
   reachable from a whitelisted TU (§3a: `Warps`, `kD1Warps` are invisible to digit grepping);
2. recover the **dominating EXEC-mask constant** of each wavefull instruction by real CFG
   dominator analysis (not nearest-preceding) and assert it is a multiple of the group width.

This is agent2's "build a real CFG or the cell can never pass" warning, now with a concrete
demonstration that the cheap version silently passes the hazard. Part 2 is the work; I can build
it as an isolated probe if you want it in-lane, but it is bigger than a grep and belongs to the
gate lane.


---

## 9. S3 second tranche — the group-integrity probe that DOES discriminate

Two files, probe-only (nothing installed as a gate; §4): `tools/v340l/wo07_exec_granularity.py`
(AMDGCN EXEC-mask granularity analyzer) + `tools/v340l/wo07_g2c_group_integrity_probe.sh` (driver).

### 9a. Method: predicate-register dataflow, not text proximity
Each `s_and_saveexec_b64 s[18:19], P` names the predicate register `P` that produced the mask. The
analyzer tracks the **definition of `P`** (`v_cmp_*`/`v_mov_b32` writing `vcc`/`scc`/`s[N:M]`) and
reads the integer constant from *that* definition. This is what fixes the nearest-preceding bug: the
mask may be written several instructions or a block earlier, and proximity paired the wrong compare.

### 9b. The control that killed two earlier attempts now passes
Same shuffle, same W=8, only the caller predicate differs:

| shape | recovered granularity | verdict | exit |
|---|---|---|---|
| `if (warp == 0)` (safe) | **32** | `32 % 8 == 0` → GROUP-INTACT | 0 |
| `if (lane < 3)` (unsafe) | **3** | `3 < W=8` → **SPLIT-GROUP HAZARD** | 4 |

The driver **aborts (exit 3) if the control stops firing**, so the greens below are conditional on
the detector still being able to fail — the property whose absence made check (e) decoration.

### 9c. Real whitelisted shapes certify clean
`block_reduce_sum<256>` (Width 8) and `<512>` (Width 16) — the §3a finding, now analyzed properly:
both narrow at granularity **32**, both GROUP-INTACT, **0 hazard, 0 unresolved**. So rmsnorm's
sub-group tail is confirmed safe *by the alignment argument*, not by hope. That is the first
positive, gate-grade statement of this WO.

### 9d. And the limit that changes the gate design (a finding, not a failure)
`argmax_block_reduce` / `sampling_block_max_key` narrow with
`v_cmp_gt_u32_e64 s[4:5], s4, v2` where `s4` is a **loaded scalar** — a runtime staging test
(`lane < blockDim.x>>5`). **There is no static granularity to recover.** The analyzer reports
`UNRESOLVED-DATA-DEPENDENT` and exits **5 — deliberately not 0** — so a CI cell cannot mistake
"couldn't see it" for "clean".

Consequence for gemini's gate, and the reason §8d's "part 2 alone" is still not enough:

- **ISA granularity alone cannot certify these sites.** Any cell that prints green for a
  data-dependent mask is lying. So the sound design is two parts, and *the source-level part is the
  one that carries the invariant*:
  - **part 1 (source, no asm):** resolve `Width` through constexpr (`Warps = BlockSize/kWarpSize`)
    and assert `32 % W == 0` at every call reachable from a whitelisted TU.
  - **part 2 (ISA):** assert recovered granularity is a multiple of `W` where statically
    recoverable, and **report-loud** the data-dependent remainder for source review.
- Exit convention **0/4/5 = clean/hazard/inconclusive** exists so "not green" is never silent. This
  line's whole failure catalogue this week is green-on-blindness; a three-state exit is the smallest
  honest interface.

### 9e. Own bugs caught during this tranche (all by running, not by reasoning)
- `unresolved` counted in `analyze()` but printed in `__main__` → NameError.
- Sorting a set mixing `int` and the `'datadep'` sentinel → TypeError; needed an explicit rank fn.
- My *test command* read `EXIT` from `tail`, not the tool — i.e. I nearly recorded "control exit 0"
  for a run that had actually exited 4. Same class as the coordinator's logged `grep -c`/`echo $?`
  artifact: check what your shell is actually measuring before you cite its number.


---

## 10. S5 — full-TU G1 depth (closing the brief's literal G1 wording) and one real find

`tools/v340l/wo07_g1_fulltu_receipts.sh`. Wrapper probes (§4) answered "does the header parse in
isolation"; G1 asks whether the file is HIP-compilable **today**, which is a property of the TU that
instantiates it. Every target was therefore compiled through its **real consumer TU**, in BOTH free
modes — M1 pure-host g++ (the path `-x hip` structurally never exercises) and M2 `-x hip`
(host+device, pass stamp read off the diagnostic).

| target | M1 | M2 | classification |
|---|---|---|---|
| `one_shot_argmax.cu`, `tp_kernel.cu`, `dflash2_attention.cu`, `dflash2_topk.cu`, `kvarn_tile_cuda.cu` | ok | ok | **clean full-TU, both modes** |
| `gqa_attention_decode_bf16.cuh` (via decode launcher), `gqa_attention_prefill_bf16.cuh` (via prefill launcher), `sparse_moe_route.cuh` (via sparse_moe_decode_kernels.cu) | ok | ok | **clean full-TU, both modes** |
| `bidirectional_gqa_attention.cuh` | ok | RED(20) | `mma_bf16` undeclared — **IN TARGET**; enablement gap per the mma ruling |
| `gqa_attention_kvarn.cuh` | ok | RED(13) | `mma_bf16` in **`gqa_attention_kvarn_mma.cuh`** (dependency) — same gap, not this file's |
| `vision_attention.cuh` | ok | RED(9) | `mma_bf16` — **IN TARGET**; same gap |
| `nvfp4_hadamard_d256.cuh` (via `gqa_attention_prefill_nvfp4.cuh`) | ok | RED(5) | `__forceinline__` in `hip_shim/cuda_pipeline.h` (dependency) — **see §10a** |

**8 of 12 now compile at full-TU depth in both modes** — a materially stronger statement than the
wrapper result, and it narrows the whitelistable set to files whose only blocker is the ruled-out
mma family. Still not "whitelist-ready" (§4 caveat holds: no device run, no instantiation evidence).

### 10a. Real finding for agent2's file: `memory.cuh` include order is a latent break
`src/ops/common/memory.cuh` is **in-build** (registered exception) and includes:
```cpp
#include <cuda_pipeline.h>     // :3   <-- uses __device__ __forceinline__
#include <cuda_runtime.h>      // :4   <-- the shim that pulls in hip_runtime.h
```
`cuda_pipeline.h` includes only `<cstddef>/<cstdint>/<cstring>`; `__forceinline__` is supplied by
ROCm's `hip/amd_detail/host_defines.h` (`:156`/`:176`), reached **through** `cuda_runtime.h`. So
`memory.cuh` uses the macro one line before including the header that defines it.

Measured, not inferred: a bare `#include "ops/common/memory.cuh"` TU fails under `-x hip` with
4 errors, all `unknown type name '__forceinline__'` in `cuda_pipeline.h:16,33,34` + the resulting
`no matching function` at `memory.cuh:114`. **It works in the real build only because every current
consumer includes `<cuda_runtime.h>` earlier in the chain.** That is order-dependent fragility in
whitelisted code, one reorder away from a build break — same class as the promote-carried
`__CUDACC__` guard this WO exists because of.

**Not mine to fix** (`src/ops/common/memory.cuh`, `hip_shim/*` are agent2's per WO-07 §4). Cheapest
repair is one `#include <hip/hip_runtime.h>` in `cuda_pipeline.h` making it self-sufficient (also
fixes every naive header probe at once), or swapping memory.cuh's lines 3 and 4. **STOP-ASK routed
to the coordinator for forwarding.** Note the asymmetry that makes this worth a gate, not just a
patch: M1 (plain g++) is GREEN on the same file where M2 goes RED — so a host-only cell, the kind
that sounds thorough, passes it. Only the `-x hip` pass sees it.

### 10b. Harness bugs, disclosed (two, both the same recurring class)
1. **Inverted DEPENDENCY label.** My first pass compared the error's file against the *consumer*
   path only, so an error inside the **target** header printed as "DEPENDENCY, not this file" —
   which would have routed three real mma defects away from the files that own them. Caught by
   reading my own output against the file paths. Fixed by comparing against target AND consumer.
2. **`grep -c` under `set -e` killed the loop after the first CLEAN file** (exit 1 on zero
   matches → empty table). This is the *second* recurrence of the bug I documented an hour ago in
   §5 of this doc; it is now commented at the site, since recurrence is the argument for the rule.
3. Also re-caught, from the coordinator's own logged class: `echo rc=$?` after a pipeline reads
   `head`'s status, not the compiler's — it briefly made a failing probe look like rc=0.


---

## 11. G2 family table (isolated probes) — *see §15b: 'definitive' was premature* — re-derived with the discriminating analyzer (§10 supersedes §8b)

Method: `tools/v340l/wo07_exec_granularity.py` (predicate-register dataflow), NOT the count method
that §8c withdrew. Same nine in-build families from the real headers, one TU, gfx900 `-O2 -S`.
Receipt: `results/amd/wo07_s1/g2_family_table_definitive.txt`; re-run via
`wo07_g2c_group_integrity_probe.sh` (whose control must pass or the whole thing aborts).

| family | wavefull ops | EXEC granularity | verdict |
|---|---|---|---|
| `famA warp_sum<32>` / `warp_max<32>` / `warp_reduce_sum<32>` | 5 each | none (un-narrowed) | **CLEAN — all lanes publish** |
| `famC argmax_warp_reduce` | 10 | none | **CLEAN** |
| `famD layer_norm_warp_reduce` | 15 | none | **CLEAN** |
| `famB block_reduce_sum<256>` (Width 8) | 5 un-narrowed + 3 narrowed | **32** | **GROUP-INTACT** (32 % 8 == 0) |
| `famB block_reduce_sum<512>` (Width 16) | 5 un-narrowed + 4 narrowed | **32** | **GROUP-INTACT** (32 % 16 == 0) |
| `famE sampling_block_max_key` | 10 + 10 narrowed | **32** | **GROUP-INTACT** |
| `famF argmax_block_reduce` | 10 + 10 narrowed | **32** + 2 unresolved | **GROUP-INTACT**, 2 regions unresolved |
| `argmax_kernel` (header's own kernel) | 1 narrowed | **data-dependent** (loaded scalar) | **UNRESOLVED — not a pass, not a hazard** |
| `argmax_tiled_atomic_kernel` | 10 + 10 narrowed | **32** + 2 unresolved | **GROUP-INTACT**, 2 unresolved |

**Tally: 0 HAZARD, 6 UNRESOLVED of 21 regions

> *(SUPERSEDED assumption, §15b: the per-kernel W=32 used to produce this tally is unsound for
> library-internal shuffles. Treat as INCONCLUSIVE, not clean.)* — and the analyzer therefore exits 5 (INCONCLUSIVE),
not 0.** That is the correct answer and the point of the three-state convention: the tree is clean
*as far as static ISA analysis can see*, and the residue is named rather than absorbed into a green.

What the re-derivation adds over §8b's count table, stated plainly:
* The count method said `famB/famE/famF` had "narrowed = 3/4/10/12" — which, with no granularity
  attached, meant nothing and looked like the hazard class. Granularity **32** is the fact that
  makes them provably safe: 32 is a multiple of every W in use {8, 16, 32}.
* The 6 unresolved split into two DIFFERENT kinds, and the distinction matters for anyone who
  builds the gate: `UNRESOLVED-NO-PRED-DEF` (mask register never written by a tracked compare in
  the scan window — a limitation of my linear scan) vs `UNRESOLVED-DATA-DEPENDENT` (the mask comes
  from a loaded scalar, `v_cmp_gt_u32_e64 s[4:5], s4, v2`, so NO static answer exists). Only the
  second is fundamental; the first is mine to fix with real dominator analysis.
* Invariant over all three assumed widths (W=8, 16, 32) the same 32 is recovered — so the safety
  statement does not depend on which Width I assume. Tested, since the assumption is exactly what
  a digit-grep census gets wrong.

**This is the WO-07 G2 deliverable: a per-family ISA statement that is both discriminating and
honest about its residue.** The permanence question the brief asked — can a future change re-break
cross-lane reduce correctness without anything noticing — now has an answer with a mechanism
attached: yes, and the two-part cell in §9d/§10 is the thing that would notice.


---

## 12. Whole-TU application: 14 whitelisted device TUs, and two false hazards I caught before shipping them

§11's table is isolated probes. Extending the analyzer to the real whitelisted device TUs first
produced **"6 HAZARD" in rmsnorm.cu and "1 HAZARD" in l2norm.cu** — claims against code agent2
measured **correct on hardware** (rmsnorm 114/114). A result that contradicts a device measurement
indicts the instrument, so I inspected before reporting. Two independent analyzer defects:

1. **Operator set too wide.** I counted `ds_read_b*`/`ds_write_b*` alongside `ds_bpermute` as
   "wavefull". Ordinary LDS I/O is routinely and correctly narrowed to one lane
   (`if (lane == 0) sums[warp] = x;` is *the* block-reduce staging pattern). Measured on
   `sampling.cu`: ds_bpermute 60 (30 narrowed) vs ds_read 219 (162 narrowed) + ds_write 57 (41) —
   so the same "24 HAZARD" figure was mostly legitimate single-lane staging. Only `ds_bpermute`
   publishes by executing.
2. **Constant leakage + unpaired region openers.** A flagged l2norm region turned out to be opened
   by `s_andn2_saveexec_b64` (which my regex didn't pair, so every later region misaligned) and to
   contain only bf16→fp32 **arithmetic** (`v_cndmask`, `v_perm`) with no shuffle at all; its "const 1"
   leaked from an unrelated `v_bfe_u32 v8, v6, 16, 1`. Compares in the `e64`/`sdwa`/`src_sel` forms
   were being parsed as if they were simple register/immediate pairs.

**Fixes:** `WAVEFULL` narrowed to `ds_bpermute` only; compares restricted to the strict
`v_cmp_*_{u32,i32,u16}_e32 dst, a, b` form with no selector suffixes; `s_andn2_saveexec` paired; and
a new **UNTRACKED** state so an unparsable predicate is reported as *unknown*, never as a hazard.
The control (safe vs `lane<3`) still separates — 1 hazard, exit 4 — which is what licenses the
greens below, per §9b's rule that a detector must be shown able to fail.

### 12a. Corrected whole-TU result (gfx900, isolated per-TU compiles, W=32)

> **⚠ STATUS: INCONCLUSIVE (1 open), per §15b and coordinator ruling.** The `W=32` in this table's
> header is an *assumption*, and §15 shows it is wrong in principle for library-internal (CUB)
> shuffles. It happens not to have fired — the one real disagreement resolved benignly — but
> “0 hazards” here must be read as “0 hazards under a width assumption that this doc now knows is
> unsound”. Do **not** cite this table as a clean bill of health.

| whitelisted device TU | HAZARD | UNRESOLVED | regions |
|---|---|---|---|
| `rmsnorm.cu` | 0 | 6 | 24 |
| `layer_norm.cu` | 0 | 0 | 1 |
| `rope.cu`, `silu_and_mul.cu`, `gdn_gating.cu`, `gdn_projected_conv.cu` | 0 | 0 | 0 |
| `l2norm.cu`, `argmax.cu` | 0 | 1 | 2 each |
| `embed_gather.cu` | 0 | 2 | 3 |
| `sampling.cu` | 0 | 3 | 5 |
| `w8_pair_decode.cu`, `w8_linear_swiglu_decode.cu`, `w8_gdn_input_decode.cu` | 0 | 0 | 2–3 |
| `w8_attn_input_decode.cu` | 0 | 0 | 4 |

**0 hazards across all 14, 16 unresolved — consistent with the hardware record rather than
contradicting it.** Every narrowed region that could be attributed resolved to granularity 32.

### 12b. agent4's serve path — GATE-GRADE PRIORITY rows, per coordinator request
The files the first real q3 serve will run through, as measured above: `sampling.cu` (0 hazard /
3 unresolved), `argmax.cu` (0/1), `gdn_gating.cu` and `gdn_projected_conv.cu` (0/0, no wavefull
exchanges at all), `rmsnorm.cu` (0/6 — the largest residue, and the file with device evidence).

**One scope fact agent4 should know before serve bring-up:** the gqa launcher pair
(`ops/launcher/gqa_attention_{decode,prefill}.cu`) is **NOT in `src/HipSources.cmake`** — grep count
0 — even though both carry `__HIP__` donor branches (`:9`, `:10` include the adopted
`gqa_attention_*_bf16_gfx906.cuh`, which contain 5 and 3 reduce sites). So the donor attention
kernels are not yet in the HIP build, and my permanence evidence for them is necessarily
pre-build: when they are whitelisted, those 8 sites become the first thing this analyzer should run
over. Not a defect — a sequencing fact, flagged because it changes what "attention verified" can
mean for a serve run.

### 12c. Two recurrence signatures worth keeping as standing tells
Both false-hazard causes above, and the earlier 21 phantom REDs, share one shape: **uniform or
contradictory results across unrelated targets.** (1) 21 unrelated files failing with an *identical*
error → the harness, not the files. (2) a "hazard" in code with a clean *device* measurement →
the detector, not the code. Corollary adopted: a device-era claim outranks a static one, always,
and a static result that contradicts it is re-derived rather than routed. The second signature is
the one that would have cost the sprint the most tonight — 24 hazard rows against agent4's live
serve path, all of them my operator set.


---

## 13. (3) Residue → verdicts: 16 UNRESOLVED down to 2, and the 2 are fundamental

The task was dominator work. What the residue actually needed was **predicate typing**, not more
dominator power: the unresolved writers were mostly guards that *cannot* split a lane group.

### 13a. Classification added (single-pass, structure otherwise untouched)
| class | example predicate writer (measured) | can it split a group? |
|---|---|---|
| **lane, with immediate** | `v_cmp_gt_u32_e32 vcc, 32, v0` | **yes** → granularity recovered, verdict issued |
| **lane, e64 destination** | `v_cmp_gt_u32_e64 s[6:7], s6, v2` | yes — was mis-filed as `untracked` purely because I excluded the `_e64` destination form; same *kind* of compare, different write target |
| **non-lane (value/pointer/loop)** | `v_cmp_gt_i64_e32 vcc, s[6:7], v[0:1]` (row bound) · `v_cmp_eq_f32_e64` (data test) · `s_xor_b64 x, exec` (inversion) | **no** — a group splits only on the **lane index**, so these regions are *out of scope*, reported as an explanation rather than counted as residue |
| **no immediate at all** | `v_cmp_gt_i32_e32 vcc, s11, v1`, where `v1` comes from `v_lshrrev_b32 v1, 6, v0` → **(tid >> 6), a row index** | unanswerable statically → `datadep`. My first pass called these `untracked`, i.e. implied **my regex** had a gap when in fact **the question has no static answer** — the more misleading of the two errors |

### 13b. Result
| TU | before | after |
|---|---|---|
| rmsnorm.cu | 6 unresolved | **0** (24 regions) |
| sampling.cu | 3 | **0** (5) |
| argmax.cu, l2norm.cu | 1, 1 | **0, 0** |
| embed_gather.cu | 2 | **2 — both `datadep`** |
| all other 9 TUs | 0 | 0 |

**Residue 16 → 2, hazards 0 throughout, control still fires** (safe vs `lane<3` → exit 4) — so the
reduction is classification gaining power, not the detector going blind. Exit stays 5 on
`embed_gather.cu` because the 2 are genuinely unanswerable at ISA level, and the whole point of the
three-state convention is that I may not convert "statically unanswerable" into "clean" for
convenience. **The floor is 2, not 0.**

### 13c. Why "real dominator analysis" would NOT close these 2
Both guards compare a **loaded scalar** against `(tid >> 6)`. No immediate exists to recover in any
CFG, and the operand is not even a lane index — it is a row/tile index, so the region selects rows,
not lanes, and the `ds_bpermute` inside it runs with every lane of its own group executing. A
dominator tree would confirm the shape; it cannot manufacture a constant. Closing them needs the
**source-level** half (§9d part 1) or a device run — and a device run is outside WO-07's grant-free
scope. This is the concrete boundary of the ISA approach, stated rather than left as a TODO.

### 13d. Own errors during this step (three, all self-caught)
1. **I "fixed" a bug that wasn't there.** I concluded my parser fabricated regions on
   `embed_gather_q3_grouped_kernel` because the range 5125-5163 held no `saveexec`. Wrong
   occurrence: the body is at 1982, and 5125 is a stub — I inspected the copy of the name I found
   second and reasoned from it. The "fix" (occurrence-keyed parse) was therefore justified by a
   misread; reverting it also **silently discarded the good, uncommitted taxonomy work in the same
   file**, which I noticed only by checking what survived. Re-applied the useful half alone.
   Lesson: verify which occurrence/section you are reading before diagnosing another lane's code,
   and prefer a targeted re-apply over `git checkout --` on a file mixing good and bad edits.
2. **My own audit grep counted its own header.** A "0 vs 1 unresolved" discrepancy in sampling.cu
   was my `grep UNRESOLVED` matching the summary line `0 UNRESOLVED`. No tool bug at all.
3. `echo`-with-backticks again (§12 and the host-test script): third recurrence, now a proposed
   lint rule.


---

## 14. (ii) Real dominator pass — the floor-2 claim PROVED, and a fourth analyzer bug found doing it

Goal was narrow: prove or disprove §13c's claim that embed_gather's two irreducible regions are
row-selects, not lane-selects. It held, but only after the tool had to be repaired.

### 14a. Analyzer bug #4: count-paired regions, not register-paired
`s_and_saveexec_b64 s[A:B], P` is closed by `s_or_b64 exec, exec, s[A:B]` — **the same `s[A:B]`**.
My stack popped on *any* `or_b64`, so non-nested regions (an early-exit saveexec whose `or` lands
after a sibling's) made depth drift. Concretely: in `embed_gather_q6_grouped_kernel` the `lane == 0`
region closes at :727 and the `ds_bpermute` is at :750 — **23 lines outside it** — yet the
count-based stack credited the bpermute to that closed region, and the residue I was about to
"explain" had a completely different cause. Fixed by pairing on the saved register. Control re-verified
after the change (safe vs `lane<3` → exit 4). **Residue is unchanged at 2, but it is now a real
residue rather than a stack artifact** — which is the only reason §13c's claim was worth testing.

### 14b. Proof of row-vs-lane, end to end
Governing compare for each residual region:

```
embed_gather_q6_grouped_kernel :697-699   v_lshrrev_b32_e32 v1, 6, v0
                                           v_lshl_or_b32     v1, s1, 1, v1
                                           v_cmp_gt_i32_e32 vcc, s16, v1     <- s16 is SCALAR
embed_gather_q3_grouped_kernel :2026-2028 (identical shape, s0/s11)
```
`v1 = (s << 1) | (v0 >> 6)`; the bound `s16`/`s11` is a **scalar** (no VGPR writer anywhere). So the
predicate is over `(tid >> 6)`, a row/block index, not a lane index.

**The ABI step verified rather than assumed** (the previous failure mode was reasoning from a
register I had not proven): a controlled compile of `threadIdx.x >> 6` vs `threadIdx.x & 31`
(`/tmp/agent5_probe/abi.cu`) emits
`v_lshrrev_b32_e32 v1, 6, v0` then `v_cmp_gt_i32_e32 vcc, s0, v1` for the row case, and
`v_and_b32_e32 v1, 31, v0` for the lane case. The row form is **byte-identical to embed_gather's**;
the lane form is absent. Hence `v0` is the thread index and the regions are **row-selects**.

**Consequence:** a row-select on a 64-lane wavefront with 64 threads per row is **wave-uniform** — it
cannot split a lane group, so the §13c "argued boundary" is now a **checked** one. The 2 residues are
not merely unanswerable-but-benign; they are provably out of the hazard class. **0 hazards across all
14 whitelisted device TUs, with the last 2 regions' status established by def-chain rather than
inference, and the tool that makes that claim repaired in the same pass.**

### 14c. Standing additions from this pass
1. **Pair EXEC regions by saved register, never by count.** Fourth analyzer bug in this tool and the
   first that changed a *verdict's cause* rather than its label.
2. **Prove the register's provenance before classifying what it selects.** `v0`-is-threadIdx is true
   on this ABI; it is a load-bearing premise and now has a receipt (`abi.cu`) instead of my
   assumption. Generalizes to every disasm claim on this board.
3. A controlled compile of the *two competing source shapes* is the cheapest discriminator between
   row-vs-lane readings — far cheaper than dominator analysis, and it settles claims that dominators
   alone would still leave as "no immediate found".


---

## 15. The escalated `sampling.cu` region — resolved to reading (ii), and it invalidates my §14 method

Register-paired re-analysis raised `sampling_partial_topk_kernel`: a region guarded by
`v10 = (tid << 1) | 1` vs a scalar bound, containing **all 20** of the function's `ds_bpermute`.
Escalated rather than explained away (coordinator seq-31: urgency dropped, because the serve rows
tonight are GREEDY — `tp_engine.cpp:265-272` sends temp<=0 down the allreduce_argmax arm — so this
kernel is not on tonight's path; and the greedy branch `return`s early at `sampling.cuh:131`).

### 15a. Resolution
The guard is **not** `v < token_domain` from the item loop. Source identification:
`SamplingPartialSort` = **`cub::BlockMergeSort<unsigned long long, kSamplerBlock,
kSamplerItemsPerThread>`** (`src/ops/kernel/sampling_device.cuh:21-22`), invoked unconditionally at
`sampling.cuh:133`. A CUB merge-sort's element-pair index is exactly `2*tid + 1` — the parity term is
the network's **pair selection**, not a lane-subset guard around a shuffle. Every lane executes the
exchange; the predicate chooses which pair each lane is responsible for. That is reading (ii), which
is what the coordinator's "odd-term smells like a segmented/paired exchange" hypothesis predicted,
and it is now supported by the construct's identity rather than by my inference from operand shapes.

### 15b. The real defect this exposed — in MY classifier, and it is the important part
I asserted **W = 32 for every in-build shuffle site** (§12/§14 methodology). That is false in
principle and near-missed in practice: CUB sort/scan internals use 2-lane pair exchanges and
sub-group widths, so the width a guard must be group-aligned against is **per-site**, not per-kernel.
My "0 hazards across 14 TUs" is therefore **not** a clean bill of health — it is "0 hazards under an
assumption that is wrong for library-internal shuffles and that happened not to fire here only
because the pairing was benign." Consequences, recorded honestly:

* §11/§12/§14's family and TU tables are marked **INCONCLUSIVE (1 open)** per coordinator ruling (c),
  not republished clean.
* Any width from a **third-party header** (cub, or a vendored primitive) is unverifiable by my
  source-level rule: `warp_reduce_sum<W>` is greppable, CUB's internals are not. The gate must
  treat library-internal wavefull ops as **out of scope** explicitly, not silently pass them.

### 15c. NEW CHECK IDEA for gemini (coordinator's suggestion, sharpened)
**Auto-flag class: a wavefull instruction inside a region whose predicate is derived from an odd
function of the thread index (`(tid<<1)|1`, `2*tid+1`, `tid^1`, `tid&1`) *and* whose shuffles come
from the 3-arg width-defaulting overloads (`cuda_runtime.h:431-432`, width=32).** That combination
means the guard's implied group (2) and the shuffle's assumed width (32) **disagree**, which is
exactly where a pair-network can be mistaken for a lane subset — or, if it really is a subset, a
G-AMD-13-class hazard. Cheap to compute (def-chain to the tid derivation + one parity test), CPU-only,
and it reports "width assumption conflicts with predicate granularity — classify by source" instead
of guessing either way. Second instance would have been caught tonight by the first version of it.

### 15d. Process note
This is the second time in this WO that applying my own method to a *new* TU broke an assumption the
method rested on (first: constexpr widths, §3a; now: uniform W=32, §15b). Both were invisible in the
isolated-probe family table and only appeared when I widened scope. For anyone reading a "0 hazards"
line elsewhere in this doc: check which assumption about **width** was in force when it was produced.


### 15e. §15c's premise tested and **refuted** — do not build the cell I sketched
I wrote the probe in `tools/v340l/wo07_parity_width_disagreement_receipts.sh` to demonstrate that a
per-kernel width assertion cannot separate "pair network" from "lane subset". **It showed the
opposite, and the finding is that my §15c check idea is not needed in the form I proposed it.**

Evidence, from the probe's own run:

| synthetic reading | analyzer output at W=32 |
|---|---|
| pair network (`__shfl_xor_sync` partner, guard selects a pair) | 0 hazard, **0 unresolved**, 1 region |
| lane subset (butterfly *inside* a tid guard) | 0 hazard, **1 unresolved**, 1 region |

The two rows **differ**. So the existing width/containment machinery already separates them; the §15b
failure was not "W=32 blinds me to the distinction" but "on the *real* CUB kernel the predicate's
bound is a runtime value, so the region is unresolvable and I guessed". Smaller claim, different fix:
what is missing is not a parity test but the discipline of **reporting tid-derived, runtime-bounded
regions as needing source classification** — which is what the `unresolved` bucket already means.

Two further corrections from building it, both mine:
1. **The odd term is not robust.** My synthetic `(2*tid+1) > bound` was **algebraically simplified by
   the compiler to `v_cmp_le_u32_e32 vcc, s6, (tid<<1)`** — the `|1` vanished. Real CUB asm retains
   `v10 = (v13<<1)|1` because its bound is data-dependent and not foldable. So "look for the parity
   term" is a *fragile* signal, and §15a's narrative ("the odd term is the giveaway") overstates what
   is actually stable. The stable fact is the operand's provenance (tid-derived), not its parity.
2. My probe's parity-detector section printed "no odd term" for both cases and I nearly filed that as
   a tool bug; it was the folding above. Read the matched line, not the expectation.

**Net value of this section:** the escalation resolved to benign (§15a), §15b's self-criticism stands
(per-kernel W is unsound as an *assumption*), but the proposed new check is **withdrawn** — cheaper
than I claimed and already covered. Left in the doc rather than deleted, for the same reason as §8b.


---

## 16. Reversal: my "0ae1f903 is retired" claim was FALSE, and the real cause was a truncated listing

I told the coordinator (seq-35) that `0ae1f903` was no longer in the `amd/main` history, and advised
that docs citing it should stop being reused as a current tip. **The shas are alive.** Verified after
`git fetch`: `git merge-base --is-ancestor 0ae1f903 amd/main` → YES, and likewise 891fd93d,
2ae7631a, 789d5146, bcd25fb2, 077a4e20, acfb19f8, 614ea829 — all ancestors of both refs.
Docs 07/11's "`0ae1f903` = the tip I verified at" statements remain correct as historical records and
**must not be edited** (coordinator's instruction, with which I agree).

**The diagnosis matters more than the retraction, because both the coordinator's and mine were wrong:**
- His: "stale origin-tracking ref; `git fetch` before ancestry claims." `fetch` is genuinely good
  practice, but it is **not** what happened — my local refs were already current at `614ea829` both
  before and after the fetch, and the ancestry check returned YES in *both* states.
- Mine: I had run `git log --oneline -n 16` and read the **bounded window** as a complete history.
  `0ae1f903` sits at topological position **7**; my quoted "chain" was positions 11–14. So I did not
  compare stale against fresh refs — I mistook a truncated listing for an exhaustive one, and stated a
  negative existence claim from a partial sample.

**Rule, replacing the one I over-generalized last hour:** a bounded listing (`log -n`, `head`,
`sed -n '1,20p'`) can never support "X is not in Y". For a membership or ancestry claim, use the
predicate that answers it (`merge-base --is-ancestor`, `cat-file -e`, `log --all --grep`), never the
absence of a line in a window. The fetch-first rule is fine as far as it goes; it just wasn't the bug,
and accepting the wrong diagnosis would have left the actual bug in place — which is the
green-that-did-not-run pattern applied to a *root cause*: a plausible, partly-correct explanation that
stops the investigation.

Self-count, this session: **six** wrong claims caught by checking rather than by being told —
21 phantom REDs (missing shim include), digit-only census (constexpr widths), `set -e` aborting on
success, count-paired EXEC regions (analyzer bug #4), W=32-per-kernel (§15b), and this truncated-
listing existence error. The last one I published to the board before checking, which is the only
kind that cost someone else a correction cycle.


---

## 17. Provenance corrections from the agent2 thread (hub #218/#222), and a misattribution I am disclaiming

**Confirmed in agent2's favour, re-measured at the live tip (`origin/amd/main` = 9bf50106 — my
frozen tree at `f4cf9a82` showed a different line 388, so cite-by-name caught my own stale read):**
the bf16 lint is `__nv_bfloat162.*__hadd2|__hip_bfloat162.*__hadd2`, it matches **exactly one** pair-op
spelling, and `_rn` occurrences in the entire gate file = **0**. Their structural claim — the most-used
pair op in the tree is unpoliced — holds. My independent reproduction of the underlying fact stands
too (§2: `__hadd2`/`__hmul2` compile clean on gfx900 with `__ocml_*`=0, native-bf16-VALU=0,
fp32-ALU=4), so "symbol lints answer what was typed, not what executed" is now measured by two lanes.

**Their counts need a scope rule before reuse.** Mine, both reproducible: `src/` only → hadd2 6,
hsub2 8, hmul2 5, hadd2_rn 0, hsub2_rn 2; whole repo → 27, 9, 6, —, 4. #222 reports 8/10/7/1/6.
The headline survives every scope (hsub2 > hadd2 in both of my measurements, one spelling covered),
so the conclusion is safe while the magnitudes are scope-dependent — the same numbers-without-a-kind
defect this board adopted as a rule, applying to a post that was itself making that point.

**MISATTRIBUTION, DISAVOWED ON THE RECORD.** #222 assigns the G-AMD-16b anomaly and the
"cos-0.99975 / canary" claims to **agent5** ("treat them as agent5's evidence"). They are not mine.
Verified mechanically, not from memory:
```
git log 514a55b7~1..HEAD --format='%s' | grep -icE 'g-?amd-16|cos|canary|0\.99975'   -> 0
ls results/amd/wo07_s1/ | grep -ciE 'g16|cos|canary'                                  -> 0
git log ... | grep -i device      -> 2 hits, both reading "whitelisted DEVICE TUs"
```
i.e. compile-level sweeps, the opposite of device runs. This lane has used **zero device time** across
38 commits and holds no grant. The G-AMD-16 series is agent3's (t3-wip, e.g. 771ae306). Filed because
misattributed device evidence corrupts an audit trail in the direction hardest to detect: nobody
questions the wrong owner, and a compile-only lane ends up vouching for numbers it never produced.

**Registry indeterminacy, unresolved by either published source.** #222 asserts hub `106391` "is
listed as agent3's session" while self-identifying as coordinator `01a09742`. Neither registry I can
read supports a mapping: `comm_agents` exposes hub name/id/status/cwd with **no lane-name field**; the
intercom roster carries lane names (agent3=01a0972f, coordinator=01a09742, agent2=01a09787) with **no
hub numbers**. The namespaces do not join in anything published — so the mismatch is real as a
*provenance gap*, not as an identified speaker, and it applies symmetrically to #222's own claimed
identity. I know my own hub number (1608656) only because my parent PID is in my environment; that is
accident, not infrastructure. Cheap fix, someone else's call: publish the hub-number ↔ lane-name
mapping in one place.


## 18. Cite-by-sha has a failure mode it does not cover: a stale sha reproduces the headline and carries the stale residue

Found in the coordinator's #226, which re-ran my probe at `5c543b12` rather than trusting my prose —
the right instinct, and all three of its results reproduce at tip. But that sha is **31 minutes**
older than the analyzer's last real repair: register-pairing landed at `2eb27e86` (16:36). Verified at
file level, not narrative: pairing markers **0** at `5c543b12`, **3** at tip.

Why that matters non-pedantically: bug #4 (EXEC regions paired by *count* rather than by the saved
`s[A:B]` register) only misfires on **non-nested regions**, i.e. on real kernels. Straight-line
probes — the SAFE/UNSAFE pair and `blk256/blk512` — were already correct, so a re-run at the stale
sha confirms the discriminator and silently inherits the stale residue. Same tool, both numbers:

| run | hazards, 14 whitelisted device TUs | unresolved |
|---|---|---|
| pre-pairing (`5c543b12`) | **24** on sampling.cu alone (all false) | 16 |
| post-pairing (tip) | **0** | 3 |

So the rule for this lane, extending the one adopted tonight: **cite the sha AND the repaired-tool
sha.** A citation that names a real, relevant, *ancestor* commit passes every check a reviewer can
apply mechanically — it exists, it's in the branch, the output reproduces — while carrying the
residue half of the result, which is the half a gate design consumes. Cite-by-sha caught this;
cite-by-branch-name would not have, and `5c543b12` is not a typo, it is a plausible commit.

Incidental, same family: #226 cites `warp.cuh:70`, which is blank (the constexpr is `:69`) — the
third lane's post to carry that pointer, and the exact off-by-one my own ed47a2d5 corrected. It is
detectable mechanically, so it is now checked rather than remembered: `tools/v340l/wo07_verify_citations.py`
reads every `file:line` in a doc against the actual file and flags blank-line/neighbour-content as
off-by-one suspects. Its own self-test must pass or it exits 5, because the two earlier versions of
that tool passed vacuously.


---

## 19. The masking finding demonstrated live on the canonical pair, at a real tip

Measured at `origin/amd/main = c2ae09fa` (refs verified current first — the staleness that caught
#226 was checked, not assumed).

- PG-1 check (a) **FAILS: 13 unregistered modified src files**, and aborts there. The (g)/(h)/(i)/(j)
  headers print **0** times. So at this tip the gate certifies nothing about the macro guard, spin
  guard, D3 fingerprint, d=1152 check, or **check (d) anti-resurrection**.
- Check (d) run standalone: **EXIT 1 — divergence from canonical main** on
  `src/runtime/tp2/tp_engine.cpp` (main's `mb_prefix_cache.h` include and lane-preference block
  absent from the AMD line's copy).
- **Lag, not reversion**, established the same way as §1's fix/descendant question rather than by
  assertion: `merge-base(main, amd/main) = 7a83cae1` (12:18 local); **11** main commits touch that
  file since; main's newest (`0f3cab43`) is **not** an ancestor of `amd/main`. That is the AGENTS.md
  case the law already answers — *lagging branch trips the cell; merge fixes, excusing never* — so a
  step-0 RED at this tip must not be read as a resurrection.

**Why this is the strongest instance of §1:** the anti-resurrection law is the one check on this board
whose failure mode is named in prose as absolute ("fail loud", "zero-diff law"). At the live tip it
was not failing loud; **it was not running.** A gate whose alarm is sequenced behind an earlier
`exit 1` cannot protect the thing it exists to protect, and the exposure was discovered only by
invoking the check directly. Any "step-0 green" cited for merges in this window is coverage that did
not execute.

The 13 surfaces (extracted from `ERROR:` lines only — my first attempt grepped all `src/` mentions and
produced a wrong 20-item list, checked against the gate's own `FAIL: 13` count before being sent):
`artifact/reader.h`, `artifact/storage_layouts.cpp`, `engine/concurrent_executor.h`,
`tp2/{host_kv_parked.cpp, tp2_backend.{cpp,h}, tp_engine.{cpp,h}}`,
`serve/{generation_service.cpp, request_log.cpp, serve_options.{cpp,h}}`, `targets/registry.cpp` —
i.e. the canonical pair **and** `serve/generation_service.cpp`, the file my §12 inventory found has no
dedicated test. Two of this lane's findings intersect there: the registration set and the
test-blind-spot set name the same file.

Three reporting errors of mine in this pass, filed because they are the recurring class:
1. `EXIT=0` read through a pipeline where the script returned **1** (`$?` was `tail`'s) — the ninth
   instance tonight, and the first that would have **understated a real failure** rather than
   overclaiming one.
2. A sloppy 20-path set from conflating ERROR lines with registered-exception lines — the
   numbers-without-kind defect I flagged in #222/#228, reproduced by me. Caught by checking my count
   against the number the gate itself printed.
3. A timestamp "discrepancy" I nearly filed: `9c957317` at 19:28 local vs 00:29Z is the same instant
   (CDT −0500). Checked before reporting; no finding. Worth a line in the provenance doc: **this
   repo's git default formats in local time while the coordinator's board language is UTC**, so every
   cross-check of "when" needs an explicit offset or nobody can compare two lanes' timings.

No action taken: no re-merge, no registration, no source touched. Reproduce:
`bash tools/ops/gate_pg1_whitelist.sh; echo $?` and
`bash tools/ops/check_anti_resurrection.sh; echo $?` at `c2ae09fa`.


### 19a. Re-verified two tips later: still 13, still masked, nothing resolved

Main moved twice after §19 was written (`c2ae09fa` → `698360be`). Recomputing with git plumbing at
the new tip — not by re-reading my own numbers, which is precisely the stale-sha trap §18 names:

- registered array grew **17 → 28** (gemini's 11-file package landed, so it is real and not pending)
- check (a): **FAIL, 13 unregistered modified src files**
- the 13 are **set-identical** to the §19 list — `added since: none`, `resolved since: none`
- real gate at the new tip prints `[Check (g|h|i|j)]` **0** times → still masked, anti-resurrection
  still unexecuted

So the 11-file package registered a *different* population than the 13 that are diverged at tip. The
masking finding stands unchanged across two tips, which also means any "step-0 green" cited for
merges in this whole window remains coverage-by-hindsight.

Incidental confirmation, on agent4's #230 ask: `src/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu`
**is registered at tip** (array line 92), so that ask is already satisfied upstream — but their claim
was true at their own base (`1b991bcf`, array 17, unregistered), so it is a stale-base artifact, not a
wrong report. Their diff on that file I verified exactly as described: **3 added
`#if !defined(__HIP__)` guard lines, 0 removed lines**. The action is re-merge, not new registration.


---

## 20. §19/§19a adjudicated: my "13 + masked" finding does NOT reproduce at tip, and why — plus the enforcement finding, confirmed, with a corrected mechanism

### 20a. My finding is retracted, and the reason is my own §18 lesson applied to me

The coordinator's bare run at tip: gate **EXIT 0**, **0** 'Pre-existing' mentions, checks (d)–(j) all
printed. I reproduced their result from the tip's own copy: **EXIT 0**, `Baseline:
origin/amd/main`, all six later-check headers emitted. So §19's "13 unregistered at tip / (g)–(j)
masked there" is **wrong as a claim about tip**.

Where the 13 came from: I ran the gate from **my lane's older copy**, which predates `d5774709` — the
commit that changed the default baseline from `origin/main` to preferring `origin/amd/main`. My copy
therefore resolved `origin/main`, i.e. the true **cross-line** comparison, and reported 13 divergent
files. That number is *correct for its baseline*; my error was attributing my copy's baseline to the
tip. The tool does print its baseline in the header, so the ambiguity was mine to resolve and I
resolved it by not checking which script I had executed — the stale-tool version of the stale-sha
error I filed at §18, committed against me within the hour.

### 20b. The enforcement finding is real, confirmed from both sides, and its mechanism is different

Their conclusion — canonical anti-resurrection enforcement is unenforced by default — is **correct**.
The mechanism is not "the child script's default changed"; it is **argument forwarding**:

- `gate_pg1_whitelist.sh:324` (tip): `"${REPO_ROOT}/tools/ops/check_anti_resurrection.sh"
  --baseline "${BASELINE}"` — the child inherits the gate's baseline.
- The child's **own** default is still `origin/main` (line 38).

Measured at tip, one invocation each, statuses assigned before being read (no pipes):

| how invoked | EXIT | printed |
|---|---|---|
| child bare (own default = `origin/main`) | **1** | `FAILED: Divergence detected from canonical main` |
| child `--baseline origin/main` | **1** | same |
| child `--baseline origin/amd/main` | **0** | `PASS: TP2 preflight and budget identical to main` |
| gate bare (forwards `origin/amd/main`) | **0** | PASS |

So the line that prints `identical to main` is comparing the AMD line **to itself**. Why the mechanism
matters for the fix: this is *not* a default to change inside the child — the child is right on its
own. Either the gate stops forwarding its own baseline to the canonical check, or the canonical check
pins `origin/main` regardless of caller. A patch aimed at "the default" would leave the bug in place.

### 20c. What survives of §19, and what is withdrawn

- **Survives:** the masking *structure* (check (a) aborts before (d)–(j), so any (a)-red merge
  certifies nothing downstream) — that is unchanged code, demonstrated on real shas in §1, and it is
  why (d) never surfaced during the (a)-red window. **Survives:** the lag-not-reversion classification
  and its evidence (merge-base `7a83cae1`; main's newest tp2-touching commit `0f3cab43` not an
  ancestor), now independently confirmed by the coordinator.
- **Withdrawn:** "13 unregistered **at tip**" and "(g)–(j) masked **at tip**". Both were measurements
  of my lane's pre-`d5774709` gate copy against the cross-line baseline, presented as properties of
  tip. §19a's "two independent tips" framing made the error sound corroborated when it was the same
  stale script run twice — self-consistency is not independent confirmation, and that is worth writing
  down because it is the one failure mode my whole re-verification habit was supposed to prevent.
- Counting-rule note, minor: my "11 main commits touch `tp_engine.cpp` since merge-base" uses
  merges-included (`git rev-list --count`); `--no-merges` gives 8; the coordinator's is 6. Different
  path/range choices, same conclusion. Stated so nobody re-litigates a number whose rule wasn't
  attached in the original post — my own recurring complaint, applied to myself last.

**Net:** the permanence lane's headline tool failed its own default mode, and it took another lane's
re-run to catch it. That is the finding, not the mistake — and the corrected claim
(enforcement-opt-in-only via argument forwarding, measured four ways) is stronger than the one I
withdrew.

---

## 21. Retracting a retraction-detail: the worktree move disclosed in commit 9921797c never happened

Commit 9921797c's message states: *"to run the real gate at the tip I checked out `amd/main` inside my
frozen lane worktree, then restored it."* **False — git forbids it:**

```
$ git checkout amd/main            (in /home/chris/worktrees/amd-wo-gfx900-perm)
fatal: 'amd/main' is already used by worktree at '/home/chris/dual_5060_ti_ninfer'
rc=128
```

The original attempt read `git checkout -q amd/main 2>/dev/null` — **stderr discarded** — so the
refusal was silent, nothing moved, and my later check "HEAD restored, dirty=0" **passed because
nothing had ever changed**. A verification that confirms the intended outcome while the operation
no-ops is the most expensive kind of green, and it bought me a false first-person confession rather
than a false success. The commit is left as-is and corrected forward, per this thread's own
retraction-forward practice.

Two consequences:

1. **A cause boarded in the coordination thread is impossible.** "A moved reference worktree is a
   different machine" cannot explain the §19/§20 divergence: two worktrees cannot hold the same
   branch, so a frozen lane cannot be silently moved to it. The divergence is fully explained by
   **baseline** — `origin/main` cross-line (my lane's pre-`d5774709` copy → 13) vs
   `origin/amd/main` self-line (a separate detached `/tmp` worktree at tip → 0, all six later checks
   printed). The useful correction: **frozen refs are protected by git, not by my care** — which
   makes "lane X's HEAD is citable ground" a structural property. Worth stating precisely, because a
   rule justified by an impossible failure mode invites the next reader to discount its real content.
2. **Replacement rule, sharper and actually true:** never discard a mutation's stderr, and assert the
   state you intended to **cause** rather than the state you remember. `2>/dev/null` is a bug-hiding
   idiom on writes (usually harmless on reads). Read-only checks against another ref still belong in
   plumbing — but for that reason, not the worktree one. This is the same family as the coordinator's
   no-op-sed-with-assert; mine is the worse instance, because mine had no assertion at all.

Numbers reconciled once, all four true under their own rules: modified-vs-`origin/main` = **36**,
modified-vs-merge-base = **24**, registered array = **28**, unregistered cross-line = **13**. Same
repo, four questions; carry the baseline with the number. Fifth appearance of the
numbers-without-kind rule tonight.

---

## 22. §19a overtaken, and hub #231's state overtaken too — re-measured at the live tip

Checked at `origin/amd/main = 0a953192` (after `git fetch`; the "live tip 1b991bcf" named in #231 is
**100 commits behind**, a STATE commit at 16:04 — a plausible, real, wrong sha, i.e. §18's failure
mode reaching the coordination channel rather than only my own work).

**Absence class: CLOSED at tip.** All five headers named across this thread now carry a guarded
include (`q3`/`q2` → `hip/hip_runtime.h`; `embed_gather.cuh`/`memory.cuh` → `cuda_runtime.h`;
`cuda_pipeline.h` → `hip/hip_runtime.h`). Compiled bare at tip, one TU each:

| bare include at tip | errors |
|---|---|
| `embed_gather.cuh` | **0** (was 5 at `1b991bcf`) |
| `q3_rowsplit_storage.h` | **0** (was 2) |
| `memory.cuh` | **0** |

So the repair batch landed, and #231's "declaration-ABSENCE open" plus its derived rules should be
read as *rationale for a tripwire*, not as a live defect report. The design conclusions survive the
closure — they were never about these three files alone: **name each cell's entry path**, and know
that `-Wmacro-redefined` cannot see absence, so a closed class still needs its bare-include row to
stay closed.

**My own §19a is overtaken in the same way, and that is the part worth keeping.** It asserted "still
13 unregistered, (g)–(j) masked, at a second tip, nothing resolved." At `0a953192`: gate **EXIT 0**,
**0** 'Pre-existing' mentions, all **6** later-check headers printed. The 13-file population was
registered between `698360be` and now. So "confirmed at two independent tips" was again the
*same stale condition* observed twice, not two independent confirmations — the lesson I wrote in §20
and then re-learned by committing it a third time within the hour.

**§20b survives intact, and is now the live item.** The child check run bare — its own default,
`origin/main`, the cross-line comparison the law actually means — still exits **1, FAILED: divergence
from canonical main**, while the gate PASSes at the same tip because `gate:324` forwards
`--baseline "${BASELINE}"` (now self-referential `origin/amd/main`). The enforcement asymmetry is
therefore *not* a transient artifact of a divergent tree: with the header class closed and the gate
green, the canonical anti-resurrection check is still unexecuted by default. That is the finding to
act on, and it is about how the check is invoked, not about any file's current state.

Reproduce:
```sh
git fetch origin amd/main
cd "$(git worktree create ...)" # or a detached worktree at origin/amd/main
bash tools/ops/gate_pg1_whitelist.sh; echo $?                    # 0
bash tools/ops/check_anti_resurrection.sh; echo $?               # 1  <-- the asymmetry
grep -n 'check_anti_resurrection.sh' tools/ops/gate_pg1_whitelist.sh   # the --baseline forwarding
```

---

## 23. The §20b arc closes: the asymmetry's root cause is the PREDICATE, and the direction-aware fix is measured-green tonight

Coordinator #366 amended its own ruling against itself on my four-way repro at gemini's tip
9d2bcd26 (the table: child-bare moved rc=1 → rc=0; every default path now self-compares; canonical
is explicit-flag-only — the opt-in-only pattern the board refuses). Merge condition held by the
coord: one of the two wiring fixes lands before absorption (child pins origin/main for canonical
mode OR gate stops forwarding and calls child bare — the minimal-fix shape my predecessor named).

**The root cause beneath both §20b (forwarding) and the #366 default-flip:** the LAW is
direction-worded (AGENTS.md step-0 cell: "no stale-region REGRESSIONS, fail loud with the REVERTED
lines") while `check_anti_resurrection.sh` is divergence-worded (any diff = FAILED). A
divergence-predicate canonical check is PERMANENTLY RED on a fork line — measured tonight at
amd/main@426e2ff2 vs origin/main@e66efb7d: two-dot shows src/runtime/tp2/ at +181/−1158 (canonical
evolved ahead post-fork; tp_engine.cpp alone 132 lines the AMD line has not absorbed) — and
permanent-red tools get defaulted away, which is the incentive that produced the forwarding AND
the default-flip. The direction-aware reading (merge-base 7a83cae1 → HEAD, minus-lines = reverted)
measures **0 added / 0 removed** in BOTH step-0 files tonight: the law's actual predicate is GREEN
at this tip; the tool's predicate never could be. Fix shape (gemini's lane to build, coord's
condition to hold): direction-aware canonical — reverted lines FAIL loud, forward-only divergence
PASSes with counts named — which makes pinning-canonical-as-default safe and restores the defender
without a ritual. Counting-rule, seventh appearance: the two-dot/three-dot distinction IS the
ref-AND-kind law with git syntax as the kind.

Provenance note for the record: the four-way at gemini's tip and the direction measurements above
are MINE (agent5 fresh session 01a09850, detached /tmp worktree, statuses-before-read, worktree
removed); #355/#358/#363/#364/#366 quotes are coordinator hub text verified where they touch
bytes. The sweep-content work that opened this session is agent3-lane material, adjudicated #363,
banked separately in docs/amd/T3_sweep_audit_prep_agent5.md.

### §23 tail addendum — two design laws from the night's close (coordinator #375, agent2 #334 thread; NOT verified by me at bytes — the 6/6 at 35bdf211 is coordinator-measured; banked as routed law with attribution)

1. **DERIVE-DON'T-LIST for applicability predicates**: compute a check's applicability per header
   from the include closure, never from an enumerated hand-list — a list that matches only the
   files that motivated it is ticket-sampling inside the gate (agent2's phrase; coordinator
   routed it to gemini's cell-contract queue as the design rule). Generalizes every list-in-a-check,
   including my own span-scanner's file set (grep-derived closure, not hand-list — compliant, and
   now with a named law behind the practice).
2. **The vacuous-green linker-input trap, implementer's note**: probes compiled as .cu under g++
   print all-zeros/greens (agent4 documented, coordinator reproduced it an hour after READING the
   documentation — the trap's stickiness is the finding). Any closure-walk tool I build (store-width
   analyzer, tier-3 check-(j) reachability) must compile probes as .cpp/-x c++ and take rc BEFORE
   concluding. This is now a build requirement for my queue items, not lore.

3. **Paraphrase-is-not-text** (coordinator #376, closing #334(2) — the 'declares' chain, channel
   quote): "a paraphrase of a claim about text is not the text — quoting a string into a package
   means quoting the file, at a named ref, from the compiler's or grep's mouth, never a witness's."
   The citation-law family is now complete across its failure modes: stale-sha (§18), same-script-
   twice (§20/§22), ref-without-kind (#358), document-family loci (agent2 #374), and now relay
   paraphrase quoted as file text (#376). My audit-prep file's "re-derive at the landed tree"
   instruction inherits the full family.

4. **Ghost-sha relay** (coordinator #378, adjudicating agent2's #338 — corrects this §'s item 3
   "family is complete at five modes"; the family grew within the hour): a one-hex-digit relay typo
   (0568d98**e**→0568d98**f**) produced a "defect" that never existed — the sha resolved nowhere.
   Predicate: `git cat-file -e <sha>` before quoting; general rule adopted board-wide: registration
   inputs must be reachable-by-others, checked, not relayed. Six modes now: stale-sha, same-script-
   twice, ref-without-kind, document-family loci, relay paraphrase, ghost-sha relay. If it grows a
   seventh, this ledger grows a fifth line — the family is complete only until the next instrument
   fails, which is the permanence lesson itself.

### §23 tail addendum (post-board-final, pin-conflict #622): the paraphrase family's seventh mode — command-stripping

Two honest seats held two stable, non-matching measurements of the same "pin"
(134a73588bba/12 vs 194e0450c0b2/10, both verified at my third seat from local objects,
zero fetch). Neither seat was wrong: the committed triple at ff1ca2ab carries a
`grep -E 'ldmatrix\.sync|mma\.sync'` stage that the channel paraphrase of it (#621)
dropped — so the second seat ran the *described* command, not the *committed* one, and
measured a true value for a different predicate. Resolution rule, now the family's
seventh mode: **when two honest measurements disagree, diff the COMMANDS, not the
claims; the channel version of a receipt is not the receipt** — pull the command bytes
from the artifact, never from the prose describing it. Corollary measured same hour: the
guard's byte-motion lies inside the span but off the filtered lines, so the filtered-set
digest is invariant where the raw-span digest moves (160071462c91→194e0450c0b2) — a
pin's stability can live in its predicate, which is why the dropped stage was
load-bearing, not decoration. Seven modes now: stale-sha, same-script-twice,
ref-without-kind, document-family loci, relay paraphrase, ghost-sha relay,
command-stripping.

### §23 tail addendum 2 (#625, coordinator re-derivation): eighth mode — algorithm-stripping

The #622 conflict's root went one level deeper than command-stripping: the ORIGINAL pin
(7e3328700dfd, count 4, awk first-range + grep) was **sha256 truncated to 12** — the gate's
actual in-script assertion (gate_pg1_whitelist.sh:311-318), stable at all four refs, never
broken. agent4's 59ddc39959cf was the AWK-MULTI-RANGE filtered set [20–22]+[102–132] under
**sha1** — a different command AND a different span from the sed family, not "the same span
under sha1" as this line first wrote (my own paraphrase defect, caught against #631's
line-range reconciliation and #630's identical self-correction); the coordinator's reversal
and agent4's "un-runnable" verdict were both false-negatives born from re-deriving a digest
without its algorithm label — each honest, each within one minute of the other. Law: **every
digest citation carries algorithm + truncation, or the reader re-derives into a false
'not reproducible.'** Net state of the mma pin: THREE true invariants, each stable at its own
(command + algorithm + predicate) — 7e3328700dfd/4 (sha256-trunc-12, the gate's own, PIN),
134a73588bba/12 (sha1, filtered set, valid belt-and-braces per #625), 194e0450c0b2/10
(sha1, raw span, agent3's receipt, true but not a pin). Eight modes: stale-sha,
same-script-twice, ref-without-kind, document-family loci, relay paraphrase, ghost-sha
relay, command-stripping, algorithm-stripping.

Precision on the 12 (#628, verified at my seat — the 2 non-asm lines are the agent3
lineage comment and the `#endif` marker whose trailing text contains "mma.sync"):
**12 = 10 PTX sites + 1 prose comment + 1 end-marker — the filtered anchor counts
instruction-STRINGS, not SITES.** Use 10 wherever a site-count is meant; pin both
invariants as labeled LAW-15 triples per #628.

### §23 tail addendum 3 (#629a, agent4's LAW-15 corollary 3): the vacuum digest

A wrong path fed to sha1sum hashes EMPTY INPUT and emits `da39a3ee…` — a perfectly-formed
digest of nothing that slots into a pin list looking sound. Guard (registered at their
52c4a408): fail on empty before hashing, never hash vacuum; tell = `da39a3ee` on sight.
This is "an empty result is the result most likely to be your own tool" at the digest
layer — the digest-shaped version of the empty-result law, banked beside the eight
citation modes. Cross-file note filed same hour: #629(c)'s replace-the-pin instruction
predates the #625 reversal (gate's sha256-trunc-12 7e3328700dfd/4 never broken; anchored
134a73588bba/12 is belt-and-braces) — flagged to the chair so no one codes from the
stale list.

Full four-digest map closed by #631's absolute-line reconciliation (three seats, zero
irreproducible values): **4** = awk+grep ranges [20–22]+[102–132], sha256-12 = 7e3328700dfd
(gate pin, un-swapped) · **10** = sed-span [20–214] raw asm sites, sha1 = 194e0450c0b2 ·
**12** = sed-span + instruction-strings incl. prose (:135 comment, :214 marker), sha1 =
134a73588bba · **59ddc39959cf** = awk+grep bytes under sha1. Missing labels were
algorithm + truncation + span-set — three attributes, not two.

### §23 tail addendum 4 (#738, agent2): the pgrep pair — opposite-direction failures on one invocation

Instrument form #11 (agent4's `pgrep -f` SELF-MATCH — false POSITIVE from the querying
shell's own command line) now has its mirror, and the pair is the lesson: `pgrep -x <script>`
matches the EXECUTABLE name, so a `bash tools/v340l/artifact_watch.sh --daemon` process
INVISIBLY returns 0 — false NEGATIVE by silence. Both directions mis-called in one session,
by the same author, on the same line of code. Robust form (worked here): never trust the
matcher; take the pid and read `/proc/<pid>/etimes` — a lookup that cannot self-match and
cannot name-confuse. Generalized to the class this census tracks: **the silent check fails
louder than the green one, because absence-of-report reads as absence-of-thing; any check
whose failure mode is "print 0" is an unearned-confidence generator and must be replaced
by a positive artifact read (pid exists? file mtime moved? digest changed?) before its
output enters a claim.** This is why the verifier's `--help` → INSTRUMENT-ERROR and the
negative cell's exit-5-on-wrong-root exist; process liveness deserved the same third state
and #738 names it.
