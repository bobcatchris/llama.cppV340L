# WO-02 — HIP shuffle width semantics: fix, golden, blast radius
**Owner:** agent2 · **Branch/worktree (MANDATORY):** `git worktree add ~/worktrees/amd-wo-shim-width -b amd/wo-shim-width amd/main`
**Base:** `amd/main` @ `7ce534f1` · **Grants needed:** none for steps 1–3; step 4 needs a device stamp (request with exact command)
**Depends on:** nothing. **Blocks:** T2/T3 GEMV + attention ports, all cross-lane tolerance calibration.

## 1. Context
`l2norm` normalized to ~1/√2 of the correct scale. Numbers: buggy lane0 = 120 (Σ0..15), fixed = 496
(Σ0..31), lane32 = 1520 (Σ32..63) — arithmetic exact.

**CORRECTED ROOT CAUSE (agent2, WO-02 §1 as written by me was wrong — accepted 2026-09-12):**
this is **not** a width-argument bug. The guard *predicate* is CUDA-exact — HIP's built-ins already
handle width correctly on wavefront-64, verified exhaustively over lanes 0–63 × every delta/laneMask in
this tree at width 32 and 64: zero mismatches. The defect is that the guard is an **early return**. On
GCN these intrinsics lower to `ds_bpermute`, a **wavefull** LDS exchange: a lane publishes its slot only
by **executing** the instruction. Lanes 16..31 opted out of the `delta=16` step while lanes 0..15 read
from them ⇒ stale/zero slot ⇒ exactly half the row lost. **agent2 compiled my literal instruction**
(pass the caller's width, keep the guard) as variant C: still 5 `ds_bpermute`, still broken under
narrowed execution — so the prescribed fix would not have worked.

**Fix as landed:** remove the early return, forward the caller's `width`, and select out-of-group values
with a **ternary (`cndmask`) so every lane still executes** the instruction.

**Why single-step reasoning kept failing (the durable lesson):** the bug exists only in **multi-step
propagation**. Any enumeration that checks one shuffle step will clear the shim — twice it did, including
mine. Gates here must therefore be **ISA-level tripwires** (`ds_bpermute` under conditional execution),
which is what agent2 built and demonstrated RED (5/5 divergent) against the unfixed tree.

## 2. Steps — commit each before the next
1. **Fix all four primitives** in `src/common/hip_shim/cuda_runtime.h` (`__shfl_down_sync`,
   `__shfl_up_sync`, `__shfl_xor_sync`, `__shfl_sync`) to pass the caller's `width`. Do **not** assume
   a one-word swap works for `xor`/broadcast — the group-base math must hold for `width` = 32 **and** 64.
2. **Golden, committed under `results/amd/`**: `0..31` reduce = **496**, `32..63` = **1520**,
   uniform-row l2norm at d=128 → `inv = 0.088388` (not 0.125). This is the permanent tripwire.
3. **Blast radius:** extend the existing 159-site table
   (`docs/amd/v340l/02_shfl_call_site_audit.md`) with a per-site column *affected / not affected* by the
   old default. That is how we know what else was silently wrong.
4. **Re-verify T1 device greens under the fix** — one stamped launch, dev0, `run_ci_amd.sh` + the T1
   batch checks. Expect l2norm green and **report any previously-green test that changes**, in either
   direction: a green that only passed because it was vacuous is a finding, not noise.
   (`layer_norm`'s `worst=1.57e-40` is already known-degenerate; `position`/`embed`/`rope` use no shuffles.)

## 3. Constraints
`warp.cuh`'s `kWarpSize = 32` **stays** — the emulation must honour the caller's width; that is the
CUDA-faithful behaviour and leaves the NVIDIA line unaffected. `hip_shim/**` is permitted by
construction, so this is **not** a third product-header exception. Freshness by construction (rebuild
from committed HEAD in-message, marker + hashes in-log). Commit before reporting.
**Docs for this line go under `docs/amd/`; integrate to `amd/main`, never to `main`.**

## 4. Base/dispatch correction (coordinator's error, 2026-09-12)
This WO was issued with base `amd/main @ 7ce534f1`, which **did not contain the T1 HIP work at all** —
the consolidation merges (`c32dece6`, `9da48d4c`, `c8d8265c`) landed *after* dispatch, so the prescribed
file had no shuffle emulation to fix and the tree could not build. **Resolved: `ab7ac370` IS an ancestor
of current `amd/main`; `cuda_runtime.h` is 155 lines with 11 shuffle symbols; `cuda_fp8.h`,
`cuda_pipeline.h`, `math_constants.h` all present.** Any lane that branched from the older tip must
`git merge amd/main` before building. **Anti-resurrection adjudication:** `wo/v340l-hip` shows
`+6/-112` on `tp_engine.cpp` against `main` — that is the branch being **behind** main's DFlash2
admission guard, not deleting it; `amd/main` is byte-identical to `origin/main` on both guarded files
(zero diff), so nothing was lost and the merge order resolves the step-0 cell. A branch that lags main
in that region will correctly trip the parity cell until it merges main — that is the gate working, not
a regression.
