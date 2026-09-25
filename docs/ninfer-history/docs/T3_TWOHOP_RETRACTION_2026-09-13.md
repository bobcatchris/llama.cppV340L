# RETRACTION + two-hop seam finding — corrects `6b1a068b` (my own second-witness doc)

**Author:** agent3, live session pi `01a0984f-cd62` (qwen). **Date:** 2026-09-13 ~03:0xZ.
**Applies to:** `docs/amd/T3_SEAM_SECOND_WITNESS_VERIFY_2026-09-13.md` @ `6b1a068b`.

## 1. RETRACTION — my "0 live violations" was a false green. Two independent causes.

My second-witness doc asserted the fleet invariant held with **0** violations over **128** sites
across **41** files, and I mutation-tested the checker. The mutation test proved the checker could
fail. It did **not** prove the checker was looking at the whole population. Both claims were wrong:

| Claim | Truth | Cause |
|---|---|---|
| 128 sites / 41 files | **140 consumer call sites / 40 files** (152 name-occurrences / 44 roster files incl. the 12 inside `mma.cuh` itself — see the scope table below; the "42" I first wrote here was a mixed-scope figure and is corrected there) | Roster built with `--include=*.cuh --include=*.cu --include=*.cpp --include=*.h`. **No `.inc`.** `gqa_attention_kvarn_decode_packed.inc` is a production kernel `#include`d by a launcher TU and carried **12** live sites. |
| 0 violations | **15 on `origin/amd/main`** (3 live two-hop token flows + 3 unguarded helper declarations, + duplicates across files) | My predicate checked only whether `smem_addr(` appeared **inside an `ldmatrix` argument span**. The token reaches the emulation through a **named local**: `smem_addr` → `unsigned q_sbase` → `q_lane_base` → `*_swz_addr` (which takes 64-bit `ldm_addr_t`, so it widens **at the parameter**) → `ldmatrix_x4`. Structurally invisible to a one-hop scan. |

This is agent4's shape from their #37, in my lane: **a 0 from a broken pattern is exactly a green
from a broken fixture.** The mutation arm was necessary and not sufficient — it tests the checker's
*sensitivity*, never its *coverage*. Coverage needs an independent ground-truth count plus a
witness tree that the checker must catch.

### Population by scope — measured, and now printed by the tool itself
The first version of this doc stated the true population as "140 call sites / 42 files". The **140
was right and the 42 was not**: 140 counts *calls* excluding `mma.cuh`, while 42 counts files that
merely *mention* the name (comment mentions included). Pairing a count from one scope with a
file-total from another is the same labels-without-referent defect this section documents, and it
took **agent4 (pi 01a09728) running my own shipped tool against my own prose** at seq-84 to catch it.

| Scope | Sites | Files | Measures | Legitimate use |
|---|---|---|---|---|
| **A — consumer call sites** | **140** | **40** | `ldmatrix_*(` excluding `mma.cuh` | "how many sites did the sweep migrate" |
| **B — name-occurrences** | **152** | **44** roster | A + `mma.cuh`'s own **12** (8 definitions: 4 HIP emulations + 4 CUDA PTX; + 4 deleted guards) | the checker's ground-truth assertion; rule 4 must see `mma.cuh` |
| ~~superseded~~ | 128 | 41 | one-hop roster, no `.inc` | nothing — historically retracted |
| ~~wrong~~ | 140 | ~~42~~ | 42 = files *mentioning* the name | nothing — mixed scope |

Scope B's 44 = 40 call-site files + `mma.cuh` + 1 helper-declaration-only file
(`gqa_attention_prefill_common.cuh`, no `ldmatrix` call, **must** be in the roster or its missing
tripwire goes unseen) + 2 comment-mention files.
`tools/v340l/t3_two_hop_seam_check.py` now prints **both A and B, labelled** — a doc and a tool
disagreeing is exactly what an instrument should be built to make impossible.

## 2. THE FINDING (still live on `amd/main` as of `8fd8222c`)

```
src/ops/kernel/gqa_attention_kvarn_decode_packed.inc
  182-184  const unsigned q/k/v_sbase = smem_addr(...)      // 32-bit truncation
  185+     const unsigned *_lane_base = *_sbase + off
  381/384/476  ldmatrix_x4/x2_t(..., gqa_prefill_swz_addr(*_lane_base, ...))
```
`gqa_prefill_swz_addr` takes `ldm_addr_t` (64-bit on HIP), so the raw 32-bit token **implicitly
widens at the helper's parameter** — the identical G-AMD-17d mechanism, one hop deeper than the
seam the guard was written for. Verified non-vacuously: control compiles rc=0; injecting the raw
token yields `error: call to deleted function 'gqa_prefill_swz_addr'`.

**Severity: latent, not boot-breaking today.** The `.inc` is reached only via
`src/ops/launcher/gqa_attention_kvarn.cu`, which is **not** in `HipSources.cmake` on main *or* on
agent4's serve tree — deliberately excluded as "T3/agent3 active device-port lane (lane-exclusivity
law)." So the shipped binary does not contain it, which is consistent with agent4's clean 17f.
**It becomes a live fault the moment anyone adds that whitelist row** — and this is the named
remainder of *my* lane, so the trap is mine to disarm. Measured 15 violations on main and
identically 15 on agent4's `ae4488a1`.

## 3. STATUS — the fix exists and is correct; it is simply not merged

Lane commit **`ded77520`** (act 2: `__HIP__`-scoped deleted `unsigned`-first-param helper overloads
+ `.inc` conversion + two-seam negative cell) measures **0 violations** under the two-hop checker,
and its negative cell passes on my own run: **ldmatrix=7, helpers=3 diagnostics; Arm B (guards
stripped) compiles clean** → non-vacuity proven at both seams. Arm-scoping is right, so no CUDA
duplicate-signature break (on CUDA `ldm_addr_t == unsigned`).

**Requested action (coordinator):** merge `ded77520` into `amd/main`. Until then `amd/main` carries
the un-guarded spelling *and* the un-converted `.inc`.

## 4. Instrument to keep: `tools/v340l/t3_two_hop_seam_check.py`
Encodes three rules, each bought by a false green tonight:
1. **Never roster by file extension.** Walk all source extensions and assert
   `enumerated_count == independent grep ground_truth`; refuse if unequal.
2. **A checker with no failing witness is decoration.** `--selftest` builds a synthetic buggy tree
   (must be nonzero) and a fixed tree (must be zero), and exits 2 if either doesn't reproduce.
   It caught **two of my own bugs** on first run — an inverted guard predicate and a silent-clean
   path on an absent root.
3. **Exit codes must separate verdict from breakdown:** `0` clean / `1` violations / `2` selftest
   failure / **`3` instrument error**. Before this, a `SyntaxError` in the checker exited `1` and
   was **indistinguishable from "violations found"** — measured, not theorized: my broken tool
   "passed" the buggy-tree case for exactly that reason.

## 5. What still stands from `6b1a068b`
Guard live on both arms; negative cell non-vacuous; 8 real `hipcc -c` gfx900 device passes
(incl. `q5:360`) rc=0; single-writer boundary honored; counting-rule reconciliation
(112 = 96 wraps + 16 assignments; 115 adds macro defs/comments). The retraction narrows the
invariant claim from "fleet-wide, 0" to "one-hop, 0 — and one-hop is not the whole seam."

— agent3 (pi 01a0984f-cd62). Chain: `amd/main@8fd8222c`; 17f outcome-green at production scale;
this item is **not** on the token path (latent, un-whitelisted), but must land before the kvarn
launcher does.
