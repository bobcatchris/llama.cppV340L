# agent2 — WO-02 lane handoff (wavefront-64 shuffle fix)

> **SUPERSEDED for current state.** This is the wavefront-64/WO-02 lane handoff and its technical
> content still stands, but for the LIVE board position read
> **`docs/amd/AGENT2_SESSION_DEBRIEF_2026-09-12-late.md`** first. That file carries two corrections
> to inherited checklist items that will otherwise cost a session its whole budget: the
> anti-resurrection check `git diff origin/main -- tp_engine.cpp tp2_budget.h` is **unsatisfiable**
> for any AMD-line branch (all four refs measure the same 196 lines of legitimate cross-line
> divergence — assert `0` against `origin/amd/main...HEAD` instead), and the parity pin
> `e85d71f0d0` is a **source-name-list** hash from `CheckHipArchive.cmake:58`, not a library digest
> (so `sha1sum libninfer_hip_host.a` disagreeing is a category error, never drift).

**Branch:** `amd/wo-shim-width` @ `468878e0`, pushed to `origin` and verified there.
**Base:** `amd/main` @ `eab05e9b` (merged in, so the step-0 anti-resurrection cell is clean).
**Integration point:** the coordinator's. Do not merge to `amd/main` yourself.

## State: ALL FOUR STEPS DONE. G-AMD-13 consumed, cards released at baseline.

| WO-02 step | Status |
|---|---|
| 1 Fix four primitives | done, committed; merged to `amd/main` at `2c4b8110` by the coordinator after independent golden re-run + falsifier reproduction |
| 2 Golden under `results/amd/` | done — `results/amd/shfl_wavefront64_golden.md` + `tools/v340l/shfl_wavefront64_golden.py` |
| 3 Blast radius column | done — appended to `docs/amd/v340l/02_shfl_call_site_audit.md`, gemini's 159 rows unedited |
| 4 T1 device re-verify | **DONE under the amended grant (option c, scoped)** — `results/amd/g_amd_13_scoped_run.log`. 6/7 rows, BATCH2 VERIFY PASS, ~10 s of 90 s, release row verbatim |

### Device outcome (the headline numbers)
`l2norm` `inv 0.208008 -> 0.143555` vs predicted full-row `0.143789`; `rownorm` FAIL 1.11 -> ok 0.000443;
`argmax` FAIL -> ok. The fix works on hardware.

### Two findings a reader must not lose
1. **`argmax` was a shuffle victim, not a scan-limit bug.** It went FAIL -> ok with no argmax change
   (shim was the only product delta), so G-AMD-10's "vocab scan limit" diagnosis and its queued fix are
   probably moot. Do not re-launch that work without re-deriving it.
2. **`layer_norm` is untested on the path that was broken, not "verified insensitive".** The harness's
   d=64 takes the per-BLOCK kernel (`layer_norm.cu:21-23` pairs only at `ne[0]==1152` **and** 4-byte
   **alignment of all four pointers** — the gate at :20-24 OR-folds `x/weight/bias/out` addresses
   against `0x3u`, and the branch reinterprets all four as `__nv_bfloat162*`; alignment term added
   2026-09-12 after #182 stated it correctly and I re-read the source at `origin/amd/main` @
   0ae1f903. A d=1152 cell that sets only the extent can land unaligned, take the OTHER kernel, and
   report coverage it did not get — so the cell must assert the branch was TAKEN), so its groups
   are always resident — hence byte-identical. The `d1152` WARP path, which really did use the broken
   reduce and which my ISA sweep took 15/17 -> 0/17, has had **zero device execution**. Compile-level
   evidence only.

### Honest ceiling
16 of the 27 in-build shuffle sites still have no device evidence: embed 2 (TU won't compile, WO-04),
rmsnorm 6 (never on device; `math.cuh` `ex2.approx` exception), layer_norm d1152 5, l2norm_generic 2,
w8 7. "PASS" means the paths that reproduce the defect are fixed. It does **not** mean the tree is
verified, and it is not 7/7.

### Ruled-on follow-ups owned by others
- `embed_gather`/`q3` + `q2` header guard patch -> **agent3, WO-04 step 0** (registered exception;
  both headers, and `q2_rowsplit_storage.h:25-31` verified by me to carry the identical defect).
- Crossline upstream fix -> routed by the coordinator.
- Merge-gate `cmake --build` + `-Wmacro-redefined` cells -> **gemini**, patch and rationale in
  `docs/amd/MERGE_GATE_BUILD_CELL_proposal_to_gemini.md`. Their lane per §7.x; my labor only.
- Calibration freeze lifted by the coordinator with three routed obligations (their lane).

## The mechanism, in one paragraph

`src/common/hip_shim/cuda_runtime.h` guarded HIP's shuffles with
`if (out_of_group) return val;`. On GCN these lower to `ds_bpermute`, a **wavefull**
LDS exchange — a lane publishes its slot only by **executing** the instruction. The early
return made lanes 16..31 opt out of the delta=16 butterfly step while lanes 0..15 read
*from* them, so readers got a stale/zero slot: exactly half the row's squares, self-consistent
across the row. Fix: forward the caller's `width`, always execute the intrinsic, select
out-of-group with a ternary (`cndmask`). `warp.cuh`'s `kWarpSize = 32` untouched.

**The trap, and it generalises:** the guard *predicate* was already correct. Any
single-step enumeration — reading the code, or enumerating lane × delta × width — clears
the shim. The defect exists only across multiple steps. This caught the coordinator twice
and caught me once. Gates on this line must test **ISA behaviour**, not call-site semantics.

## Numbers, with their provenance

* lane0 `120 → 496`, lane32 `1520` — Σ of the claimed lane ranges.
* uniform-row l2norm d=128: `inv 0.125000 → 0.088388`. The `0.125000` equals G-AMD-12 E2's
  device-measured value, which is what makes the mechanism more than a story.
* batch2c `0.208008` reproduced as `0.207781`; residual is bf16 rounding of the
  `out[0]/in[0]` ratio probe. That bounds the error budget C441 said was never bounded.
* ISA tripwire: isolated `warp_reduce_sum` **5/5 divergent → 0/5**.
* Blast radius: **27 of 147** user shuffle sites are in the HIP build; 120 sit in 36 files
> **Qualification on that number (added after agent5's #215; I verified it myself at
> `warp.cuh:80`, `rmsnorm.cuh:153,199`, `HipSources.cmake`).** It counts *sites*, and I let it
> carry an inference it never supported: nothing in that census looked at the **width** argument,
> so it must not be read as "every in-build shuffle is full-width 32". It isn't —
> `warp_reduce_sum<Warps>` with `constexpr int Warps = BlockSize / kWarpSize` ships **today**,
> reached through `block_reduce_sum<Block>` from rmsnorm at Block 256/512, i.e. sub-group widths
> 4, 8 and 16 are already in the whitelisted build (per #226 follow-up: reachable W = BlockSize/32 over
> Block {128,256,512}, plus sparse_moe's literal kD1Warps=8 — W=4 comes from rmsnorm.cu's
> kBlock=128, which the original thread had not enumerated). A literal-digit regex (`<[0-9]+>`) cannot see a
> constexpr template argument — exactly how agent5 published and then retracted "0 of 72
> sub-group sites in-build". So a width-integrity gate must resolve the template argument through
> constexpr, not grep it; and (same source) an EXEC-**count** gate cannot separate safe
> `warp == 0` narrowing from a group-splitting `lane < 3`, which emit byte-identical shuffle
> structure and differ only in the predicate constant. This column says nothing either way about
> those, and now says so in-file.
  unreachable from `HipSources.cmake`. This is why the CUDA line never hit the bug.

## Deliberate deviation from the dispatch — do not silently repeat this

WO-02 §1 originally said the bug was passing `warpSize` instead of the caller's `width`.
I compiled that literal instruction (variant C: swap the width, keep the guard): **still 5
`ds_bpermute`, all 5 under narrowed EXEC — still broken.** Reporting that, rather than
implementing it and waiting for a launch to find out, is what got the work order corrected
(`5c4466bc`) and the falsified diagnosis preserved in-file as the lesson. When a dispatch's
prescription is provably insufficient, say so before writing the code.

## Environment traps on this box (each cost me a cycle; all cold lanes hit them)

* **`cmake` is not on `PATH`**: `/home/chris/opt/cmake/bin/cmake` (3.30.5).
* **Configure needs `-DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF`.** `BUILD_TESTING`
  defaults ON → forces `NINFER_BUILD_MEDIA_ACQUIRE` → requires `libcurl>=7.81`, which is
  absent (only `libcurl-gnutls` runtime, no dev package, no passwordless sudo). Because
  `NINFER_BUILD_MEDIA_ACQUIRE` is set by `set()` at `CMakeLists.txt:78-80`, a `-D` override
  cannot turn it off — you must turn off the two options that raise it.
* **Need `-DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0`** or `find_package(hip)` fails.
* **`hipcc` treats a `.a` given as a positional as source.** Link with
  `-L build-hip-amd/src -lninfer_hip_host`, not by passing the archive path.
* **`.isa` via `objdump` doesn't work here**; use `clang++ -S -x hip --offload-arch=gfx900`
  and read the assembly.

## Pre-existing defect I fixed incidentally (flagged for review)

`tp2_backend.cpp:1400` uses `cudaPointerAttributes` / `cudaPointerGetAttributes`, which the
shim never provided — so `ninfer_hip_host` **did not build on `amd/main` at all** before my
commit. Not shuffle-related, not mine. Added as plain aliases (`hipPointerAttribute_t`
carries the same `device` int field, `hip_runtime_api.h:262-270`). If the coordinator wants
it as its own commit, split it out of `468878e0`.

## Honest limits — read before trusting a green

* rmsnorm `114/114` and argmax `20/20` whole-kernel divergent counts are **unchanged** by
  this fix. Expected, not a miss: those narrowed-EXEC regions are caller-side and
  **warp-uniform** (`if (warp == 0)` in `block_reduce_sum`, `lane == 0` staging), so every
  shuffle source is executing. My linear saveexec-depth scan cannot separate those from real
  divergence — which is why the gate is defined on the isolated reduce. A gate that reports
  them as *safe* rather than *uncounted* needs a real CFG, and that is gemini's lane.
* `mask` is still discarded (`(void)mask`) at all 147 sites. Pre-existing and unchanged. No
  compiled site uses a mask narrower than its `width`, so nothing in the build depends on
  it today. That is not a claim that masks are correct in general.
* I am not predicting 7/7 on step 4. `layer_norm` should move from its vacuous `1.57e-40`
  to a real number; that is a finding, not a regression. `rmsnorm` goes from never exercised
  to actually tested.

## Mistakes of mine worth not repeating

1. My first commit deleted the 17-line "Live branch map at handoff" from
   `docs/amd/COORDINATOR.md` — collateral from `git stash`/`stash pop` cycling while A/B
   testing against the unfixed base. Restored byte-exact and amended. **Don't stash-cycle
   on a branch containing other people's authored docs; use a scratch worktree.**
2. I `cp`'d the fixed shim over the base file before checking what the base contained.
   Caught it and restored.
3. My first two simulators were simply wrong — one dropped `x +=` (wrote `x =`), one
   mis-parsed the `+=` butterfly and reported the shim clean. Both errors agreed with the
   *then-current* consensus, which is exactly why the disagreement had to be resolved by
   compiling real ISA, not by more reasoning. **A model that omits the accumulator silently
   answers a different question than the one asked.**

## Step 4, when stamped

```sh
hipcc -O2 --offload-arch=gfx900 -I src -I src/common/hip_shim -I include \
  results/v340l/batch2_verify.cu -o /tmp/batch2_fixed \
  -L build-hip-amd/src -lninfer_hip_host
HIP_VISIBLE_DEVICES=0 timeout 30 /tmp/batch2_fixed
```
Binary sha256 prefix at report time: `634bd560eb81c01d`. Rebuild for freshness-by-construction
and log marker + binary + archive hashes. Release row verbatim (`No KFD PIDs`,
4 × `18,575,360 B`). Its output is line-comparable with `results/v340l/batch2c_run.log`.

---

# APPENDED: lane change + current queue (post-WO-02, 2026-09-12 ~14:2xZ)

**Agent1 abandoned the line. I am CONFIRMED Agent-B** (user ruling via C441):
`src/common/hip_shim/*` is mine; `tp2_budget.h` is **read-only canonical on both lanes**;
`tp_engine.cpp` likewise. Merge direction `main -> amd/main` only, and the integration point is the
coordinator's — I do not merge to `amd/main`.

## Queue state

| item | what | status |
|---|---|---|
| (1) | gfx906 fork ISA audit, `docs/amd/v340l/03_gfx906_fork_isa_audit.md` | **DONE** — load-bearing masks verdict computed |
| (2) | **WO-TB1** shim closure, 6 TP2 surfaces + `.type` check on 6.2.0 | **DONE** @ `f57a9d3d`; positive + 5 negative cells run; full HIP library green |
| (3) | d=1152 cell + merge-gate build cell | **ROUTED to gemini** (their lane, §7.x) — `docs/amd/MERGE_GATE_BUILD_CELL_proposal_to_gemini.md` |
| — | 7th T1 row (`embed.dense-bits`) | no standalone grant; rides agent3's consolidated **P1 probe window** |

## Durable findings from this phase (things I would want on a cold resume)

1. **`__CUDA_ARCH__` is NOT defined under hipcc** on ROCm 6.2.0 (measured with a `#error` probe). Any
   `#if __CUDA_ARCH__ >= N` device guard silently takes the `#else` branch on the HIP path. This is how
   `one_shot_allreduce.cu:80,95` loses its `__nanosleep` the moment that file is whitelisted — leaving an
   **unbounded, yieldless device spin** on hardware where their fork proved the mode reset can fail
   outright. Landmine, not live bug (file deliberately not whitelisted, `HipSources.cmake:40-46`).
2. **Our D3 "bf16 arithmetic is compiler-enforced" premise is FALSE.** `__hadd2` compiles clean on
   three configs and lowers to ~6-instruction RNE emulation. The D3 gate cannot catch a D3 violation.
   Not a shim defect — a shared ROCm property — but the header's safety story is wrong and should be
   replaced with a real gate (ISA fingerprint or lint on bf16 arith in `ops/kernel/**`).
3. **`.type` on ROCm 6.2.0 exists** (`hip_runtime_api.h:263`), and `hipStreamCaptureStatus` ordering
   matches CUDA (`None=0/Active=1/Invalidated=2`), `hipMemoryTypeDevice=2`,
   `hipErrorPeerAccessAlreadyEnabled=704`. **Review rule now in the header: enum names only, never bare
   integers**, because the fork's evidence is 6.4.1 and I did not enumerate its values.
4. **WO-TB1's demand count is currently ZERO** in our tree — no caller references any of the 6 names.
   It is prophylactic enablement by design; do not describe it as "TP2 transport works now." Three of
   the six are P2P-related and **P2P is absent on this box (G-AMD-5)**; the header says so explicitly.
5. **Their fork runs TP2 on AMD silicon today; ours never has.** They are a stage ahead on transport,
   and their wedge record is the cheapest thing they offer us. Design WO-05 against a documented
   failure mode rather than rediscovering it.

## Method rules this phase earned (my own failures, so they don't recur)

- **Validate the detector before believing a green.** My tripwire passed four trees; only the negative
  control (10/11 hazard) made those greens mean anything. Same for bf16: my first test "proved" the
  opposite of the truth because I omitted `cuda_runtime.h`.
- **Read signatures before trusting my own script.** Two false "narrow mask" positives came from
  matching `block_reduce_sum(x, float* sums)` against the masked-helper signature; a "0 sites" result
  came from assuming clang-format output from `grep -rn`.
- **Never `cp` or `stash`-cycle over a tree containing other people's authored content.** I destroyed a
  17-line coordinator block doing exactly that. Use a scratch worktree for A/B against a base.
- **Compile, don't argue.** Three prior "static reads" of this shim were all confident and all wrong;
  the only thing that settled it was emitted ISA.

---

# APPENDED: session end state (context exhausted, clean hand-back)

**Nothing in progress. Nothing blocked on me. TB2 gated on Green's landing sha (not started).**

Tips at end of session (all pushed; `git status` clean in both):
- `amd/wo-shim-width @ 8bdd4f5a` — __syncwarp shim (content-identical to merged 4e2ae4c6; coord took
  the earlier copy, empty diff, nothing lost).
- `amd/tp2-probes @ af0c2348` — 4/4 donor probes compile green; `parity.cpp` dropped from P1 by ruling.

**OPEN DEFECT I FOUND, routed to agent3 (theirs to fix — NOT mine to port):**
`src/compat/gfx906/gfx906_shim_addendum.h:19` defines `__syncwarp(unsigned = 0)` while
`src/common/hip_shim/cuda_runtime.h` now defines the same name with default mask `0xffffffffu`. Both
in force in any TU including the addendum *after* `cuda_runtime.h` → redefinition or silent shadow.
Their compile cells did not catch it because none includes BOTH. Coordinator confirmed it against
bytes and verified the ruling. Fix = delete the TU-local definition now that the durable mapping is on
`amd/main`, and add a **macro-vs-identifier collision cell** for the addendum pattern generally
(any TU-local shim that duplicates a `hip_shim` name). That cell is unclaimed: I declined it for
lane reasons (test-lane authorship is gemini's; the file is agent3's), so if nobody has picked it up
on a cold resume, it is a real and small gap in the AMD CI.

**The one lesson worth carrying forward, which earned its place in line law:** three separate times
tonight a *total* from my own or someone else's probe would have produced a false result, and reading
the matched line was the only thing that caught it — my R1 counting `s_waitcnt lgkmcnt(0)` as a
barrier and contradicting my own finding 30 s pre-commit; the coordinator's `--showtopo` returning an
empty matrix section while my four-device sweep was complete; and the coordinator's compile probe
returning 1 error that was a missing include path, not agent3's code. Generalization adopted verbatim:
*a receipt that merely agrees with you is not evidence — but a receipt that disagrees is a gift, so
read the matched line, not the total.*

**Also durable, from the same session:** the wavefront-64 shuffle defect is invisible to any
single-step analysis — the guard predicate was correct and three independent enumerations cleared it.
Only multi-step propagation exposes it, and only emitted ISA settles it. That is why every gate here
tests ISA (`tools/v340l/shfl_wavefront64_golden.py`, with a validated negative control at 10/11), not
call-site semantics. A model or script that silently answers a different question than the one asked
is the recurring failure mode of this whole sprint — mine dropped `x +=`, another assumed
clang-format output from `grep -rn`.
