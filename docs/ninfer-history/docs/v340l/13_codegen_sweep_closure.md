# Codegen-vs-parse sweep — closing the §5 item left unfinished by the previous session

**Who:** agent2 (hip_shim lane). **When:** 2026-09-12 ~21:5xZ, new session resumed from the
Agent-B debrief. **Cost:** zero GPU, compile-only. **Tree measured:** `amd/wo-shim-funcattr`
@ `4897d357` (13 debrief-era commits + this session's watcher commit), plus a throwaway
`origin/amd/main` worktree at `f6ee80d5` and agent3's `amd/t3-wip` @ `5e451bd4`.

## 1. The question the debrief handed over

The previous session shipped `tools/v340l/hip_codegen_probe.sh` after proving that
`-fsyntax-only` cannot see an invalid PTX mnemonic, and noted: *"I started exactly this and
ran out of context: re-run every file tonight's receipts called clean through real `-c`
codegen."* The worry was that tonight's greens were produced by a blind instrument, so the
syntax-green/codegen-red class might be much wider than the one mma.cuh instance that gave
us the instrument.

Answer, measured: **it is not wider. Exactly one instance, and it is the one already named.**

## 2. Method — and the first cut was wrong, so here is the correction

Splitting "already codegen-proven" from "receipt-clean but never compiled" is the cheapest
honest filter: a source file that produced a real object in `build-hip-amd/` has by
definition passed `-c`. Two of my own scripts got this wrong before getting it right:

* first tried to match sources to objects with the wrong path prefix, printed the nonsense
  result "`0` of `143` built", and I caught it because the number contradicted a build log
  that had just said `164/164`;
* then matched on `basename(x, .o)` while the sources still carried `.cu`, so the two sets
  could not intersect and it printed `0` again.

Final matching is on the object path relative to
`build-hip-amd/src/CMakeFiles/ninfer_hip_host.dir/`, which reproduces the CMake guard's own
164-name bookkeeping. Both bugs were silent — a wrong `0` looks like a finding, not like a
broken script. Lesson kept: when a census returns a suspiciously clean zero, check the join
before reporting the number.

**Counts: 143 `.cu` in `src/`; 31 carry objects (codegen-proven); 112 do not.** The 112 were
run through BOTH modes with the build's own flags taken from
`build-hip-amd/src/CMakeFiles/ninfer_hip_host.dir/flags.make` (`-O3 -DNDEBUG -std=gnu++20
--offload-arch=gfx900`, the `hip_shim`-first include order, `__HIP_PLATFORM_AMD__`), not with
the probe script's hand-written approximation of them.

## 3. Result

| state | count | meaning |
|---|---|---|
| `CLEAN` (syntax 0 **and** `-c` 0) | 33 | receipt-clean claim survives real codegen |
| `BOTH-RED` (syntax ≥1) | 79 | ordinary port failures; codegen never reached |
| **`DELTA` (syntax-green / codegen-red)** | **0** | the blindness class does NOT generalize |

Receipt: `results/amd/codegen_sweep_2026-09-12.tsv` (one row per file, `SWEEP_DONE` sentinel).

The single known DELTA remains `mma.cuh`'s four `ldmatrix` wrappers, reached through synthetic
callers: this session re-ran it on a pristine `origin/amd/main` worktree and on my branch.

## 4. A refinement that narrows the original claim (mine to correct)

The debrief says raw-PTX sites are "invisible to `-fsyntax-only`". That is true of
**mnemonics** and false of **constraints**, and the difference matters for how much to trust
each mode. Isolated on single-purpose TUs (the first attempt conflated the two by pairing a
bad mnemonic with an `"l"` constraint, which produced a front-end error and hid the point):

| defect | `-fsyntax-only` | `-c` | class |
|---|---|---|---|
| invalid mnemonic, valid `"r"` constraint | **0 errors** | `invalid instruction` | back end — REAL blindness |
| invalid `"l"` input constraint | `invalid input constraint 'l'` | same | front end — both modes see it |

So the 38 `invalid input constraint 'l'` errors in the both-red set (unguarded
`ld.global.cg` sites in `linear/{bf16,fp8,nvfp4}*_gemv.cuh`, `memory.cuh`) are NOT hidden
defects — a parse cell already reports them. The blindness class is narrower than "any
unguarded PTX": it is any unguarded PTX whose operand constraints happen to be legal on
AMDGCN. `ldmatrix` qualifies because it takes `"r"`.

Consequence for v340l/08's fourth mode (agent5's three-mode build-cell spec, routed to
gemini): the codegen cell is worth running for the valid-constraint case only; for the `"l"`
case a syntax cell suffices and is faster.

## 5. The 79 both-red files: what is actually blocking them

Error-signature census over all 79 (dominant terms):

| hits | signature | what it means |
|---|---|---|
| 430 | `use of undeclared identifier 'mma_bf16'` | the `#if !defined(__HIP__)` at mma.cuh:33 **excludes** the whole `mma.sync` family while callers still call it |
| 38 | `invalid input constraint 'l' in asm` | §4 — front-end-visible unguarded PTX |
| 18 | `'cuda_fp4.h' file not found` | shim has no fp4 header at all |
| 12+12+9 | `kWarpSize`, `cudaErrorInvalidConfiguration`, `cudaErrorInvalidValue` | ordinary shim-name gaps |

The 430 are one cause with one fix, and agent3 already wrote it.

## 6. Does agent3's landing fix the whole class? Measured, not assumed

`0176f99f` (on `amd/t3-wip`, **not** an ancestor of `origin/amd/main` — verified with
`merge-base --is-ancestor`) adds HIP emulation arms for **both** families:
`ldmatrix` at :16-99 and `mma_bf16` at :209. Re-running the same 79 files against agent3's
tree: **27 become syntax-green AND codegen-clean, 52 remain syntax-red, 0 new DELTAs.** The
fix is real, additive, and introduces no codegen regressions in the files it touches.

So the debrief's item 1 stands with a sharpened rationale: main today is red at *parse* on
the TUs that would need this, and where parse passes it is red at *codegen*. Both halves of
that were measured on a throwaway `origin/amd/main` worktree this session, not quoted.

## 7. `__hsub2_rn` — the merge gate, with the decision-grade evidence

My merge parks on gemini's bless-or-block for the bf16 **pair** alias (a D3 widening,
self-flagged). Measured this session, all of it first-hand:

1. **ROCm defines no `_rn` pair spelling.** `grep __hsub2_rn|__hadd2_rn|__hmul2_rn` over
   `/opt/rocm-6.2.0/include/hip/*.h` and `hip/amd_detail/*.h` → **0 hits**.
2. **It is the SOLE blocker for two mainline TUs.** On agent3's tree (mma emulation present,
   my alias absent), `w8_gdn_input_gemm_splitk.cu` fails with exactly **2 errors, both
   `undeclared '__hsub2_rn'`**.
3. **My alias is ISA-INERT relative to plain `__hsub2`.** Two kernels, same signature, one
   using `__hsub2_rn` one using `__hsub2`: **43 instructions each, byte-identical bodies**.
   It adds no new arithmetic; it renames an operation the lane can already lower. Emitted
   form: 2× `v_sub_f32_e32`, **0** `__ocml_*`, **0** native bf16 VALU — i.e. the software
   RNE-emulation class agent5 measured for `__hadd2`, which is the actual D3 content of the
   ruling.
4. **Both TUs pass real `-c` codegen** with the alias overlaid (via an out-of-tree include
   overlay; agent3's worktree untouched).

For the record on my own instrument discipline: my first diff attempt matched function
labels `_Z5k_rn`/`_Z7k_plain` — the mangling is `_Z4k_rn` (the `k` counts), so both
extractions returned **0 instructions** and the tool reported `BYTE-IDENTICAL`. That is a
vacuous green of exactly the type this sprint has been cataloguing, and it is why the claim
above is stated with the instruction count visible. A diff of two empty files proves nothing.

## 8. Related numbers worth having (from the §1 gate re-run, all green)

`funcattr_shim_receipts.sh` 12 cells + 4 mutations · census spine **28 covered / 0 MISSING** ·
`shfl_wavefront64_golden.py` GREEN (5 `ds_bpermute`, 0 under narrowed EXEC) · full HIP build
rc=0 · anti-resurrection `git diff origin/main -- tp_engine.cpp tp2_budget.h` = **0 bytes** ·
`tools/ops/gate_*` absent from my three-dot diff.

**Debrief correction:** the pinned parity string `e85d71f0d0` is **not** a library hash — it
is `CheckHipArchive.cmake:58` `string(SHA1 ... "${_expected}")` over the source-name list.
The archive's own sha1 (`f7b32921aa`) is a different quantity and its mismatch was never
drift. Parity re-measured on this branch by deleting the `.a` and relinking: **164/164,
sha1=e85d71f0d0, exact match.** Worth fixing in the prose of any handoff that calls it a
build hash, including mine.

## 9. Artifact-pipeline finding (new this session, owned by me)

q3 now exists at **two** paths on **different devices** — CIFS `/media/chris/desktop_f` and
ext4 USB `/media/chris/EMTEC256` — both exactly 15,446,796,288 B with identical mtime to the
nanosecond (distinct dev/ino, so genuinely two bodies, not a hardlink), and identical at all
5 sampled offsets. The coordinator's serve handover names EMTEC256 as the pin; my watcher
verifies desktop_f. If the file the server opens is not the file the monitor hashes, a green
watcher row says nothing about served bytes. Shipped in `4897d357`: the watcher now emits a
`q3-alt` row per alternate path, with the cross-check labeled `SAMPLED-IDENTICAL` (never
`VERIFIED` — that word is reserved for a full sha of the primary), plus distinct
absent/size-divergent arms and an ABSENT message that prints the path it looked at, since a
stale path constant reads as ABSENT forever and q3's path moved three times today.


---

## 6. CLOSURE — measured on `amd/main` after merge, not by citation

The hazard this instrument was built for (`mma.cuh`'s four `ldmatrix` wrappers exposed outside the
`__HIP__` guard: syntax-clean, codegen-dead) is **CLOSED on `amd/main`**, verified with the instrument
rather than inferred from a merge message:

    synthetic caller of ninfer::ops::ldmatrix_x2, on main:
      -fsyntax-only  -> 0 errors
      -c (gfx900)    -> 0 errors          # was: syntax 0 / codegen 'invalid instruction'
    mma.cuh alone under -x hip             -> 0 errors

Structure, independently re-derived on main and matching agent4's #250 claims for their tree:
**8** `ldmatrix` definitions — 4 shuffle-based HIP emulations containing **zero** `asm`, and 4
byte-intact PTX originals under the CUDA arm; zero `-` code lines in the landing diff.

### Line-number reconciliation, which is the useful part of #250

Every cited set turned out correct **for its own ref**, except one:

| ref | PTX originals at | why |
|---|---|---|
| `0176f99f`, `7856532e` | `:74/:80/:87/:93` | as landed |
| `1726c918` | `:78/:84/:91/:97` | +4: header self-sufficiency include block |
| `origin/amd/main` (merged) | `:95/:101/:108/:114` | +17 further: `__HIP__` fragment emulations etc. |
| #249's `:79/:86/:92/:99` | matches **no** tree | gaps +7,+6,+7 vs every real file's +6,+7,+6; `:79` is an `asm volatile` line on `1726c918` and a definition line on main, so the set is a blend across refs |

So #250's own conclusion ("quoting a line number requires quoting the file, not the witness chain")
is right and I endorse it, with one correction: the `:79/:86/:92/:99` set originated in the
**coordinator's #249**, which I quoted *in order to refute* — reporting that it matched no tree I could
find. Attributing it to my post repeats, at the author level, exactly the propagation defect the thread
is about: the string `:79` is real text in a real mail, and the mail that first contained it is a
different fact. I checked headers rather than memory, which is the only reason I could say so.

The durable rule for gemini's pair-file arm, agreed with agent4 and reached the same way they did:
assert **content hashes of the guarded blocks**, never line numbers or line-count offsets. Main's
mma.cuh has drifted +17 lines from the commit that introduced the emulations while the *code* stayed
byte-identical — so a number-based arm would false-red on a file whose substance nobody changed, and a
hash-based arm passes it for the right reason. Structural anchors survive; coordinates do not.