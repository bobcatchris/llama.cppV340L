# WO-TB2 — landing-delta map, q3 spine → TP2 bring-up (Agent-B)

**Date:** 2026-09-12 ~17:0xZ. **Zero GPU, zero launch** — a diff study, per the plan's
"zero GPU" label on P2.
**Method:** the donor's TP2-SLICES partition (`docs/gfx906/TP2-SLICES.md`, adopted as our
phase sequence by `docs/amd/FORK_VELOCITY_PLAN.md:50`): changed files → conflict classes →
slice order. Referenced by structure, not by copying their slice contents — theirs partitioned
a *cherry-pick of a foreign lineage*, this partitions *an already-merged upstream sync*, and
the difference is why §1 exists.
**Branch of record:** `amd/main`. All numbers below are `git` output on this host,
reproducible from the commands inlined under each table.
**Base:** re-synced onto `amd/main` @ `8a96dfcd` (5th sync, carries upstream `7a83cae1`) at
17:2xZ per coord base-notice. §1's counts were re-checked post-sync and still hold; §4's
upstream line numbers :171/:173 unchanged (7a83cae1 does not touch `tools/bench/q3_ci.sh`,
verified by `git log -1 --` on the file).

---

## §1 THE BOUNDARY QUESTION, ANSWERED FIRST (this is the part gemini needs, and it changes the task)

**The q3 landing is not pending. It is already inside `amd/main`, completely.**

```bash
git merge-base origin/main amd/main          # -> 4575ac1b25e01b4087fa742213457c008ce874d0
git rev-list --count amd/main..origin/main   # -> 1
git log --oneline amd/main..origin/main      # -> d9acffea (only)
```

A merge-base equal to Green's landing sha *is* the proof of full containment: every commit
they have, we have. Independently checked per-sha, all YES:
`4575ac1b` (q3_ci TP2 tally + p1n1_oracle pre-warm), `695f9d7b` (smoke row green / promotion),
`818baf07`, `f5c6d5b6`, `fc4da3ca`, `48d5efc6`.

So the residual landing delta — what is upstream and NOT yet ours — is exactly one commit:

| sha | files | LOC | surface | conflict class |
|---|---|---|---|---|
| `d9acffea` | `tools/ops/q3_commit_guard.py` | +30/−0 | CI guard only | **EMPTY vs Team Red** — no `src/`, no `tests/`, no build file |

Its content is a new `check_shared_artifact_collision()` (G2.9, A3 overwrite-compare: flags
arm-tagless dump filenames in compare-joining code) plus one line wiring it into the check
list. It cannot break a HIP build; it *can* fail-red a future Red commit whose message/diff
emits untagged dump paths from a file whose path contains `compare`/`cross`/`join`/`b4` —
worth knowing before someone's merge gets bounced by Green's guard on an AMD-side logging
change. Flagged to gemini as a battery-author question, not a blocker.

**Consequence for the plan.** `TP2_AMD_SUBSET_PLAN.md` §P2 gates WO-TB2/WO-TA4 on "on Green's
landing". That event has passed. TB2's value is therefore NOT "what will conflict when it
lands" but "what did we already absorb, is it HIP-clean, and what is the real remaining
gap". §2 answers that. Anyone still treating the landing as pending is scheduling against a
stale board.

---

## §2 WHAT WE ALREADY ABSORBED (the 58-file promote at `f5c6d5b6` + 4-file at `fc4da3ca`)

27 of the 58 landing files are `src/`. Twelve are in the HIP whitelist — i.e. already
compiling on our lane, which is why the promote is not hypothetical for us:

`artifact/{reader.cpp,storage_layouts.cpp,typed_binding.cpp}` · `ops/launcher/embed_gather.cu` ·
`ops/linear/linear.cpp` · `ops/linear_add/q2/{q2_linear_add_gemv.cu,q2_linear_add_plan.cpp}` ·
`ops/linear_add/q3/{q3_linear_add_gemv.cu,q3_linear_add_plan.cpp}` ·
`ops/wrapper/{embedding,linear_add}.cpp` · `targets/qwen3_6_27b/impl/package.cpp`
(listed verbatim from the script-embedded loop's output, not from memory — the shorthand
version of this line originally mis-spelled the q2 pair as `q3_*`, which is how I checked it)

Verified by building, not by reading: full `ninfer_hip_host` at `amd/main` = **164/164
objects, 0 errors** (my run, 2026-09-12 11:44 local, archive sha256 prefix `0f33b8102f95e054`),
so the q3 shard ops are HIP-green today. The 12-file list is reproducible with the loop in
`tools/v340l/shim_surface_census_spine.sh`'s header comment.

### Conflict classes actually encountered (both historical, both resolved)

| class | instance | resolution of record |
|---|---|---|
| shared build file | `src/CMakeLists.txt` (CUDA-only q3 ops lines) | registered handoff exception; auto-merge, zero markers (`f5c6d5b6` msg) — HIP path is insulated by `HipSources.cmake` early-include, CUDA block textually preserved |
| two-sided edit on a ported header | `ops/linear/{q2,q3}/*_rowsplit_storage.h` | Red's `f5035616` guards the attribute-neutralisation behind `__HIPCC__` — the pattern for "CUDA says opt-in, AMD has no such cap" |
| log-file append collision | `docs/CROSSLANE.md` | keep-both rows |
| **latent, unfixed** | `tools/bench/q3_ci_amd.sh:177-193` vs upstream `q3_ci.sh` post-`4575ac1b` | see §4 — gemini's file |

---

## §3 THE REAL REMAINING GAP (Agent-B lane output): shim surface for the spine

`git log` shows the spine's device-side files are exactly the ones deliberately NOT
whitelisted (`one_shot_allreduce.cu`, `one_shot_argmax.cu`, `tp_kernel.cu`, the
`dflash2_*.cu`). Their shim demand was unmeasured until now. Census script:
**`tools/v340l/shim_surface_census_spine.sh`** (zero GPU, re-runnable).

Method note baked into the script, because the naive version is wrong in a way that would
have cost someone a cycle: **do not derive HIP spellings by substituting `cuda`→`hip`.**
Measured counter-examples on this box — `cudaHostAllocMapped`→`hipHostMallocMapped`,
`cudaHostAllocPortable`→`hipHostMallocPortable`, `cudaDevAttrMultiProcessorCount`→
`hipDeviceAttributeMultiprocessorCount` (prefix *and* body differ). A substitution census
invents gaps and, worse, invents coverages.

Result: **26 symbols covered, 2 missing.**

> **Self-correction, kept in place because it is the instructive part.** The first run of
> this census said 3 missing and included `cudaErrorInvalidValue`. That was a FALSE GAP: in
> `gqa_attention_{kvarn,prefill}.cu` the name occurs only in PROSE — comments describing a
> past warmup crash — never in code (verified: `grep -n cudaErrorInvalidValue <files> |
> grep -v '//'` is empty). Cause: symbol extraction over raw file text. Fixed in the script
> by stripping `//` and `/* */` comments before matching, for both the symbol list and the
> USED-IN column. A false gap is the worst output this script could produce — it invites
> aliasing a symbol nobody calls, which is precisely how zero-reference shim surface
> accumulates. Same rule as everywhere else on this line: read the matched line, not the
> total — and it applied to my own instrument, one commit after I wrote that sentence.

| symbol | needed by | ROCm 6.2.0 target (verified) | note |
|---|---|---|---|
| `cudaHostGetDevicePointer` | `one_shot_allreduce.cu:281` | `hipHostGetDevicePointer(void**, void*, unsigned)` — `hip_runtime_api.h:3792`, signature identical | clean name alias |
| `cudaDevAttrMultiProcessorCount` | `tp_kernel.cu:27` | `hipDeviceAttributeMultiprocessorCount` — `:477`, inside `hipDeviceAttribute_t` | name-only alias; **positional enum value differs from CUDA's 16** — legal because both sides stay in HIP space, and exactly why the enum-names-only rule exists |

**Both are already covered by agent3's TU-scoped `src/core/multi_gpu/multi_gpu_hip_addendum.h:24-25`**
(read, not assumed) — so the shim-level landing is a DURABILITY move, not an unblock: agent3's
T3 is not waiting on me. That materially lowers its priority; see the status paragraph below.

**Not landed by me.** Both serve agent3's whitelist moment and their interim addendum
already covers them. Landing symbols with zero in-build references and no driving task is how
prophylactic surface accumulates untested — WO-TB1 was explicitly stamped by the coordinator
before I did that. Say the word and it is 2 lines plus a cell; the census is the gate output
for it. Priority is LOW, and lower than this section's first draft claimed: the whitelist
contract already makes each absence loud at the moment of use, and the interim addendum means
agent3's T3 compiles today.

### The addendum's half-life, and one thing to do when it is dropped (MEASURED)
`amd/t3-wip` carries a second, file-local instance of the same pattern in
`gqa_attention_{prefill,decode}.cu` (prefill :20-33): `#define cudaFuncSetAttribute
hipFuncSetAttribute`, then a `hip_func_set_attribute_impl` template, then `#define
cudaFuncSetAttribute hip_func_set_attribute_impl`. Two properties verified on gfx900, both
matters when that block is deleted in favour of my shim mapping:

1. **It does NOT hard-collide with my template — but not for the reason it looks like.** Its
   `#ifndef cudaFuncSetAttribute` guard tests for a MACRO, and my shim provides a *template*,
   so the guard passes and the macro lands anyway, shadowing the template for the rest of the
   TU. Compilation still succeeds because their own impl is a by-value template that absorbs
   the bare spelling. So the interaction is silent, which is worse than a loud failure — the
   durable mapping appears to be in use and is not.
2. **Their impl is PERMISSIVE where mine is constrained.** Measured:
   `cudaFuncSetAttribute(42, cudaFuncAttributeMaxDynamicSharedMemorySize, 1)` — an integer as
   the KERNEL argument — COMPILES under `hip_func_set_attribute_impl` (rc=0, `reinterpret_cast`
   swallows it) and is REJECTED by my shim wrapper. That is exactly the hazard my
   `n4_forwarding_escape` negative exists to catch, live in a real file. Deleting the interim
   block is therefore not just cosmetic cleanup: while it is in force, that TU has no argument
   checking at all.
3. Attribution, so nobody chases the wrong thing: the 2 `-Wmacro-redefined` warnings this
   block emits are **inherent to its own double `#define`** — reproduced identically against
   plain `hip/hip_runtime.h` with my shim absent. Not caused by, and not fixed by removing,
   my landing. But note gemini's merge-gate build cell proposes a `-Wmacro-redefined` leg
   (`docs/amd/MERGE_GATE_BUILD_CELL_proposal_to_gemini.md`), which would fire on these TUs
   once they are whitelisted — so the drop should land with the block's deletion, not after it.

### Task-1 mapping's place in this picture
`cudaFuncSetAttribute` / `cudaFuncAttributeMaxDynamicSharedMemorySize` were the *only*
missing symbols for the gqa/kvarn launcher file set and are now in (fb43f0b3). The census
lists them as covered, with the caveat that coverage ≠ capability: AMD has no CUDA-style
dynamic-smem opt-in, so `hipFuncSetAttribute` may answer `hipErrorNotSupported` at the
launchers' `CUDA_CHECK`. Relayed into agent3's T3 expectations by the coordinator at 16:45Z —
the donor SIMT bf16 path may never need the call at all, which would make the question
moot for bring-up and leave it quarantined in the kvarn path.

---

## §4 ROUTED TO GEMINI (hub lane): q3_ci TP2 tally boundary — measured divergence, latent

Upstream `4575ac1b` **relaxed** the formats arms-proof for TP2: in `MODE=tp2` the
`load: formats` line is now *optional* ("verified if emitted, or reported N/A per coord
seq-49 tally-gap boundary"); in `MODE=single` it stays a hard FAIL. Three-way diff of the
check block: upstream `tools/bench/q3_ci.sh:171-201` (post-change) vs
`tools/bench/q3_ci_amd.sh:177-193` (ours).

Our AMD copy still carries the **pre-`4575ac1b` unconditional** version — it has no `MODE`
branch at that point, so it hard-fails where upstream now says N/A.

**Exposure is currently zero, verified:** `q3_ci_amd.sh` is referenced by nothing in
`tools/ops/run_ci_amd.sh` or any script (grep across `tools/` returns only its own file), and
no caller sets `MODE=tp2`. So this is not a live false-red — it is a false-red waiting for
the wiring that WO-TG1 promises. Given P1's finding that our TP2 path is staged-transport
and Green's own note says the tally emission "not yet wired on TP2 path", an AMD tp2 run
plausibly emits no `load: formats` line → our copy fails the cell for a reason upstream has
already declared out of scope. Their call whether to port the branch or keep the strict
check deliberately (strict-and-documented is a legitimate choice; strict-by-fork-drift is not).

### CLOSED — decision recorded (agent3, 17:04Z, seq-15; all three claims byte-verified by them)
**RULING: PORT the MODE branch, verbatim upstream semantics** — tp2 = tally-if-emitted +
structural NOTE; single = unchanged strict (the `Q3G64_F16S=129` assertion is kept).
Rationale on record: P3 bring-up runs `MODE=tp2` and Green's tally emission is not wired on
the TP2 path, so strict-by-drift is a guaranteed false-red on exactly the cell the sprint
needs; porting costs nothing because single-mode semantics are untouched (zero coverage loss).
**Ownership: gemini's** (CI file, WO-TB1/TG1 wiring), routed by agent3 to the coordinator with
this §4 as the reference. My row is closed as decided; I do not edit their file. The 177-193
line cites in this section are my measured range for the current unconditional block (agent3
cited :175-195 for the same region — same block, different anchoring, no disagreement).

### WO-TB2 follow-up (coord option (a), 17:25Z): does strict-single-mode stay intact? — PROVEN YES

**The paragraph requested.** Yes, and it is provable rather than eyeballed: extracting the
old unconditional block (`4575ac1b^:tools/bench/q3_ci.sh`, 171-187 region) and the new
`else`-arm (173-201) and comparing line-by-line after stripping leading whitespace gives
**17 lines vs 17 lines, byte-identical**, the only textual addition being the explanatory
`# Single-card mode: strictly require formats tally` comment — so all three single-mode
assertions (the `load: formats` presence FAIL, the `Q3G64_F16S=[1-9][0-9]*` count FAIL, and
the ART-guarded `Q3G64_F16S=129` FAIL) survive unchanged, as does the success echo. The new
branch is also safe under the script's `set -u` because `MODE` is defaulted at :29
(`MODE="${MODE:-single}"`) and an invalid value is FATAL at :44, so at the branch point `MODE`
is always set and the `else` arm is exhaustively single-mode. Conclusion: **porting the branch
loses nothing on the single path.**

**The asymmetry the verbatim port carries, which is worth fixing while touching the file.**
In the tp2 branch, when the tally IS emitted, upstream checks only `Q3G64_F16S=[1-9][0-9]*`
(count > 0) and **drops the ART-guarded `=129` assertion entirely** — so a 3-of-129 partial
format load passes in tp2 and fails in single. I checked whether that relaxation is forced by
sharding, because if it were, restoring it would be a false-red: it is **not** forced. The
tally is manifest-derived (`src/targets/registry.cpp:143-155` counts
`artifact::TensorDescriptor`s from `reader.objects()`, with the in-file comment "Computed
from the MANIFEST the reader actually parsed"), and the TP2 route materialises shards from a
SINGLE shared reader (`tp_load.cpp:411`, `materialize_tp(const Reader& reader, int rank, int
world, …)`) — so the descriptor count is rank-independent and both ranks report the same 129;
sharding halves bytes, not manifest entries. Therefore gemini can restore the
`[[ "$ART" == *qwen3_8_27b_q3* ]]`-guarded `=129` check inside the tp2 *emitted* sub-branch at
zero false-red cost — the guard already self-excludes every other artifact, which is the usual
reason such an assertion gets dropped. Partial-load-detection is precisely what an arms-proof
is FOR, and TP2 is the route where a partial shard set is the plausible failure.

**Port scope, measured so nobody overestimates the edit:** `diff tools/bench/q3_ci.sh
tools/bench/q3_ci_amd.sh` shows exactly four differing regions — three benign host-shape ones
(ART path `/home/intel/...` vs `/home/chris/...`, BIN auto-discovery of `build-hip/`,
LOGDIR) and the tally block itself. Everything else is already in common, so the MODE port is
a contained single-region edit.

**Verified on the 5th-sync base** (`8a96dfcd`, after `7a83cae1` — which touches
`tools/bench/q3_ci.sh`? it does NOT; `git log -1 -- tools/bench/q3_ci.sh` still reports
`4575ac1b`, and the two branch line numbers :171/:173 are unchanged), so this analysis is
current as of this commit and the §4 ranges above still hold.

**One caveat that bounds everything above:** this is an analysis of the *committed* state on
`amd/main`. The coordinator's own `7a83cae1` message records that a bad `git add -A` nearly
swept away "gemini's strict-key/G2.8 work-in-progress" in `tools/bench/q3_ci.sh` and
`q3_commit_guard.py` — i.e. gemini has/had uncommitted edits in exactly this file family. If
their working tree already contains the MODE port (or a deliberate strict keep), this section
is a no-op for them and my §4 row should be closed as already-superseded rather than acted
on. That is why this is written up as analysis and handed over, not implemented: the file is
theirs and their working state is invisible to my census.

---

## §5 SLICE ORDER FOR THE TP2 SPINE (partition method applied)

Inputs: the donor sequence S1 transport → S2 shards → S3 split ops → S4 attention geometry →
S5 runtime → S6 parity → S7 perf; our P1 measurements; the VRAM Law; the spin law.

| step | content | status / gate | lane |
|---|---|---|---|
| S0 absorb | q3 promote merge | **DONE** — 164/164 HIP build green | — |
| S1 transport | `tp_group` staged/event behind the shim | host side whitelisted + green; device side (`one_shot_*`, `tp_kernel`) needs the 2 §3 symbols — **already covered TU-locally by agent3's addendum, so this is not a dependency on me** — plus agent3's ported PTX volatile lanes; **eager only** — cross-device graph exec is DEAD on 6.2.0 (measured, P1), so the donor's graph-transport variants are excluded, not deferred | agent3 (A); Agent-B durable-symbol landing only on request |
| S2 shards | q3/q2 rowsplit storage | **DONE** incl. the `__HIPCC__` guard pattern (`f5035616`) | — |
| S3 split ops | `*_linear_add_gemv.cu` | in build, goldens clean at 5 shapes (WO-04 step 2) | agent3 |
| S4 attention | gqa decode/prefill/kvarn HIP lanes | blocked on T3 kernel work, NOT on the shim (Task 1 removed that excuse); smem-opt-in question per §3 caveat rides here | agent3 |
| S5 runtime | `tp_engine` / `tp2_backend` HIP | already whitelisted + green; **anti-resurrection: `tp_engine.cpp`/`tp2_budget.h` diff vs `origin/main` is clean** — re-check on every merge, main's copy wins by law | agent3 |
| S6 parity | TG1 battery vs our own tp1 control | gemini's lane; parity gate against OUR artifact, not theirs | gemini |
| S7 perf | AR-count budget | AR is latency-dominated at bring-up shapes (P1: 3.13 GiB/s staged, 14.10/14.32 µs small copy) → **AR count per token is the budget**; the 3.13-vs-6.6 GB/s host-staged discrepancy is UNRESOLVED and no AR budget may be computed off either number until it is | agent3 + coord |

Ordering constraint carried from the donor's wedge record, which is the reason this sequence
exists at all: their S9b made flag-sync the TP2 default, wedged an MI50 at request 12, and
`MODE1 reset FAILED (-22)` — their governance sentence, adopted line-wide, is that *"the S9b
CLI gates never exercised the serve path's sampling + `full_reset` re-arm sequence."* Our
equivalent risk is a gate that green-lights the transport in isolation while the serve path
never ran it. Anything S1-S5 that lands must be gated by a cell that goes through
`ninfer-serve`, not only through a probe.

---

## §6 WHAT q3 CHANGES FOR OUR BRING-UP vs THE PROBE PORT (the dispatch's question)

The probe port (`48d5efc6`, 4/4 donor probes green) measured **mechanism**: transport
bandwidth, P2P absence, capture semantics, replay cost. The q3 promote changed **what the
transport has to carry**, in three ways that matter to bring-up:

1. **Shard-side dtype surface grew a proven path.** q3/q2 rowsplit gemv/gemm-simt + the
   linear_add plans are now in the HIP build and served-proven on the CUDA line (smoke row:
   1161 + Tokyo stop_token, 18.3 tok/s anchor parity, KV geometry bit-identical). Our S3/S6
   no longer needs to establish q3 correctness from scratch — it inherits Green's, and the
   remaining question is gfx900 numerics, not API shape.
2. **The artifact/format contract moved under us.** `artifact/reader.cpp`,
   `storage_layouts.cpp`, `typed_binding.cpp` all changed in the promote (plus
   `reader.h`, `ops/kernel/embed_gather.cuh`, `ops/launcher/embed_gather.{cu,h}`,
   `ops/wrapper/{embedding,linear_add}.cpp`). Those are the files that decide whether a Q3
   tensor binds — so any Red result obtained against a pre-`4575ac1b` reader would be
   provenance-stale. **Checked rather than assumed, and the bad case does not obtain:** the
   four live AMD receipts all postdate the promote commit (commit dates, `-0500`): promote
   `f5c6d5b6` 08:17:55, then `g_amd_13_scoped_run` 08:59, `wo_q3hip_step2_cpu_goldens` 09:24
   (its own message says "on merged amd/main base 039ab28b"), `wo_q3hip_step34_hip_build`
   09:32, `ta1_census_tp2` 09:44. So no existing golden needs re-running on this account —
   but the rule stands for anything produced before 08:17, and for future citations check the
   base sha in the receipt header, not the file mtime.
3. **A no-fallback proof key changed.** `q3_ci.sh`'s arms-proof is the gate that asserts Q3
   actually loaded rather than silently falling back; its TP2 branch is now MODE-conditional
   (§4). Our AMD lane's arms-proof must state which boundary it enforces, or it becomes the
   vacuous-green pattern this sprint has now caught four times.

**Unchanged by the promote, so still our problem:** the spine's device-side allreduce/argmax/
tp_kernel files, the no-P2P topology (G-AMD-5), and the dead cross-device graph executor.

---

## Reproduce everything here

```bash
cd <amd/main checkout>
git merge-base origin/main amd/main && git rev-list --count amd/main..origin/main
bash tools/v340l/shim_surface_census_spine.sh          # 26 covered / 2 missing (comment-free counts)
bash tools/v340l/funcattr_shim_receipts.sh             # Task 1 gate: 12 cells + 4 mutations
diff <(sed -n "171,201p" tools/bench/q3_ci.sh) <(sed -n '177,193p' tools/bench/q3_ci_amd.sh)
```
