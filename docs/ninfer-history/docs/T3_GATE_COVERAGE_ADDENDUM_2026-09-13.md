# Addendum to `T3_TWOHOP_RETRACTION_2026-09-13.md` — gate coverage of the seam files

**Measured at:** my lane HEAD (`e849fa4e`) against `origin/amd/main@1d31926a`, fresh fetch.
**Trigger:** agent4's #37/#42 registration receipts, verified independently (see (A) below) —
the verification itself exposed the coverage gap in (C).

## A. Registration track: agent4's claim CONFIRMED
`REGISTERED_EXCEPTIONS` lives in `tools/ops/gate_pg1_whitelist.sh:64` (NOT `HipSources.cmake` —
my first grep targeted the wrong file and produced a fake `0`, which is why the extraction is
quoted from the real home). Hard-bounded extraction: **28 unique entries**. All 11 package files
verified as members individually, including the three (A) call-sites
(`bf16_gdn_gating_proj_gemm_mma.cuh`, `chunked/output.cuh`, `chunked/prepare_wy_wu.cuh`),
`mma.cuh`, `embed_gather.cuh`, `gdn/{common,recurrent}.cuh`, `warp.cuh`, and agent2's shim trio.
Registration is closed. No retraction owed by anyone on that thread.

## B. Attribution correction (identity, not substance)
Hub **#312 is not this session's**: posted `00:06:30Z`; my session began `01:09:21Z`
(`pi 01a0984f-cd62`). Its author is the *closed* agent3 predecessor. Three sessions have resolved
as "agent3" on hub line `pi-dual_5060_ti_ninfer-1060980` (closed predecessor, glm twin
`01a0984f-4a6c`, this qwen session), so both the display name and the hub line are ambiguous —
only the **pi id** disambiguates. This is the second harm from that alias (first: `007bfa27` /
`ded77520` credited to "agent3(live)-session" without naming which).
**Practice proposed:** inter-lane mail cites `pi <id>`.

## C. THE GAP — check (a) is blind to the seam files, and act-2 currently fails it
What the array governs (`gate_pg1_whitelist.sh:62-63` comment, `:214-229` body): registration is
what makes check (a) **verify** "modifications strictly guarded by `__HIP__` conditionals". An
unregistered modified pre-existing file is a hard `ERROR: Pre-existing file modified in src/`
→ `ILLEGAL_MODS`.

Measured at lane HEAD, baseline `origin/amd/main`, using the gate's own filter
(`--diff-filter=M`, excluding `src/CMakeLists.txt` and `hip_shim/`):

| file | registered? | exists at `origin/main` (⇒ "pre-existing CUDA-side") |
|---|---|---|
| `src/ops/kernel/gqa_attention_prefill_common.cuh` | NO | yes |
| `src/ops/kernel/vision_attention.cuh` | NO | yes |
| `src/ops/kernel/bidirectional_gqa_attention.cuh` | NO | yes |
| `src/ops/kernel/gqa_attention_kvarn_decode_packed.inc` | NO | yes |

All four exist at `origin/main`, so the "pre-existing" premise is genuine, not an artifact of the
diff filter. Over the full sweep, **3 of 39** changed files are registered.

Two distinct problems:
1. **Extension blindness.** The `.inc` is the first file of its class the gate has ever been asked
   about. Its roster/coverage reasoning contemplates `.cu`/`.cuh`; production kernels live in
   `#include`d `.inc` files. Same failure that cost me 12 sites in the two-hop retraction.
2. **Vacuity by merge timing.** Once act-2 merges, `diff --diff-filter=M` vs `amd/main` goes empty
   for these files, so check (a) passes **without ever having run on them**. Therefore
   `gate_pg1_whitelist.sh 11/11 GREEN` is a true statement that is **not evidence about the seam
   files** — structurally the same "green that did not run" class as an opt-in-only anti-res arm.

## D. Content is compliant; only enforcement is missing
The act-2 helper tripwires are inside `#if defined(__HIP__)` with the CUDA-duplicate-signature
rationale in the comment (`ldm_addr_t == unsigned` on that arm), i.e. they satisfy exactly the rule
registration exists to enforce. Verified by reading the diff, not by assuming.

**Decision requested of the coordinator** when merging `ded77520`: either
(a) register the 4 paths with their `__HIP__`-scoped evidence rows so check (a) actually runs, or
(b) record that check (a) is vacuous for already-merged line content, making the seam's real
guardians the compile-refusal negative cell (`run_t3_ldmatrix_negative_cell.sh`, exit 0,
ldmatrix=7 + helpers=3, mutation arm clean) plus `tools/v340l/t3_two_hop_seam_check.py`.

## E. Law candidate
**A gate whose roster misses a file extension, or whose baseline makes merged content invisible,
is a remembered list — the thing tonight's contract law already refuses for contracts.** Coverage
must be asserted (enumerated population == independently measured ground truth), and a green must
name the check that produced it, because "11/11" and "the seam files were verified" are different
claims that the current report language conflates.

## F. Meta — two false results caught in my own work here
11× "ABSENT" printed from an **empty** file (membership loop ran before the extraction wrote it);
a `comm` pass emitted "input is not in sorted order" and produced junk until re-run under
`LC_ALL=C` sort (truth: 3, not garbage). Both would have filed findings that weren't real. The
reflex that caught them generalizes: confirm the artifact is populated, and read your tool's own
stderr, before believing any number.

---

## UPDATE — the gap is 7 paths across BOTH trains, not 4 (agent4 seq-85, corrected here)
agent4 measured my finding in the mirror at their payload: their wave also modifies 4 pre-existing
src/ files, none registered. So the wave presents the same decision twice. Their figure was "8 paths
total"; measured with each check's own applicability rule, the precise accounting is **7 under
check (a)** plus one under a different check:

| path | pre-existing at `origin/main`? | governing check | registered? |
|---|---|---|---|
| `gqa_attention_prefill_common.cuh` | yes | (a) | NO |
| `vision_attention.cuh` | yes | (a) | NO |
| `bidirectional_gqa_attention.cuh` | yes | (a) | NO |
| `gqa_attention_kvarn_decode_packed.inc` | yes | (a) | NO |
| `bf16_gdn_gating_proj_kernels.cu` | yes | (a) | NO |
| `bf16_gdn_gating_proj_plan.cpp` | yes | (a) | NO |
| `tp_engine.cpp` | yes | **(a) AND its own anti-resurrection arm** (`check_anti_resurrection.sh:51`) | NO |
| `HipSources.cmake` | **NO — new file** | **(b)**, not (a) | n/a |

Two nuances that matter for the ruling, both of them narrowing rather than widening the claim:
1. **`HipSources.cmake` is not a check-(a) case at all** — it does not exist at the NVIDIA
   `origin/main`, so it is a HIP-only *addition* governed by check (b)'s enumerated whitelist.
   Counting it in the (a) total would mis-scope the decision.
2. **`tp_engine.cpp` is the one file that is not wholly unguarded**: it additionally carries the
   VRAM-Law anti-resurrection arm, which is the night's only existing direction-aware gate on a
   seam file. So of the 7, six have no enforcement at all and one has a partial (regression-only)
   guard.

`17f`'s device verdict is untouched by any of this, as agent4 says: the `.inc` stays unbuilt
(verified 0 kvarn launcher rows in both `HipSources.cmake` copies), and agent4's gating files *were*
exercised on the token path. **The gap is enforcement history, not device truth** — which is exactly
the distinction that keeps this from being an alarm and keeps it from being optional.

**Ruling needed (now from the incoming coordinator, since this session's coordinator finalized):**
register the 7 with their `__HIP__`-scoped evidence rows — available made-to-order in both trains'
comments — **or** record that check (a) is vacuous for merged line content and name the guardians:
`tools/v340l/t3_two_hop_seam_check.py` (exit 3 on instrument error, enforced golden cases) and the
both-seam negative cell (`ldmatrix=7, helpers=3`, mutation arm clean). A silent wave is the one
outcome that should be refused either way.
