# Triage — Gemini's prior-era Tasks 2 + 3 against plan REV 3.20-3.22 (agent5, plan owner; chair 05:1xZ ask)

Zero card, zero build. Every cite below was opened at this seat in the last hour, not recalled.

## Task 3 — "CC-gate pin cell at layouts_impl.h:631" → **PREMISE STRUCK, live half CLAIMED (owner-tagged)**

**The trap the chair flagged is real, and the bytes say more than "a refusal that never fires here":**

| leg | measured fact | cite |
|---|---|---|
| who calls the guard | `validate_target_options` (layouts_impl.h:543) is called from exactly ONE site: `make_sequence_planner_impl` (:722) — the SINGLE-GPU planner route | `git grep -n "validate_target_options(" origin/amd/main -- src` = 2 hits, both in that file |
| does the TP path ever reach it | **No.** `TPEngine` never calls it — zero call sites in `src/runtime/tp2/`; that is exactly why world=2 escaped the first-light refusal | agent1's forensics row `results/amd/p3/G18_firstlight_forensics_row_agent1.txt:4` (@ origin/amd/main) |
| does world=4 reach it today | **On amd/main and agent4's lane: no.** WO-TP4-E rewrote the factory route to `wants_tensor_parallel(asked)` (engine.cpp:323, `>= 2`), so 3/4-rank requests go to `TPEngine` instead of falling through to the generic Engine that hits the throw. The fix comment is explicit that the sm_120 throw was "downstream LUCK in ONE target's planner, not a gate" — the 27b/35b siblings carry no such guard and would have served the downgrade silently | `origin/amd/main:src/runtime/engine/engine.cpp:310-330` (commit 7bfd8b4d) |
| on MY lane | **Still yes.** `amd-wo-nvfp4` predates the merge: `src/runtime/engine/engine.cpp:310` is still `if (options.devices.size() == 2)`, so a 4-device boot on my branch still falls through to the single-GPU planner and still throws at :631 | my worktree, measured 05:5xZ |
| the host datum | gfx900 reports CC-code 90 through the HIP shim (`cudaDeviceProp = hipDeviceProp_t`, `DeviceContext::sm() = props.major*10 + props.minor`), so `sm() != 120` is TRUE on this box — the guard's condition holds here forever; what changed is whether anything reaches it | `src/common/hip_shim/cuda_runtime.h:20`, `src/core/device.cu:118` |

**Verdict.** Pinned as "the NVFP4 TP4 gate", the cell would freeze a refusal that the TP4 route no longer traverses — and it would do so while reading as GREEN on my lane, because my lane is pre-merge. That is a canary-in-the-wrong-room cell (the same shape as this morning's F1: an arm that passes because the path it guards is unreachable), so the premise is struck.

**The live half, claimed:** the routing fact itself wants a pin, and it is a *step-0 predicate* not a boot test — `grep "devices.size() == 2" src/runtime/engine/engine.cpp` ABSENT (agent1's own pre-declared form, `TP4_FIRSTLIGHT_FORENSICS:39`). Owner: this desk, riding the amd/main → `amd-wo-nvfp4` merge that brings WO-TP4-E into my lane (the merge is already owed for TP4 deploy sequencing; the pin cell lands with it, one commit, with the red captured on the pre-merge tree which is today's tip). If the chair wants it earlier it can be a host-only cell against a cited tree-ish — but it should not be written twice.

## Task 2 — "NVFP4 codec fixture goldens" → **ALREADY COVERED on the pack leg; export leg stays BLOCKED-with-owner**

| piece | state | stamp / home |
|---|---|---|
| generator | main-resident, re-runnable | `tools/v340l/nvfp4/gen_nvfp4_export_golden.py` @ 41d5e020 (byte-identical to the desk pin, §8e three-probe) |
| corpus (fixtures) | banked and stamped | `tests/multi_gpu/nvfp4_shard_fixture.h` @ a024627b — on amd/main, torch-free to READ |
| pack-golden consumer | **shipped and passing**, incl. the cross-language convention duel | `tools/v340l/nvfp4/nvfp4_pack_golden_host.cpp` @ 81aced51 — "pack convention agrees across languages at every corpus case", rebuilt from the GIT BLOB at my seat (§8e) |
| export-golden consumer | desk-only, NOT in any branch, and gated by design | `/home/chris/agent3_cells/nvfp4_export_golden_host.cpp` @ 143650e4 — my plan's NOT-hour-1 row keeps it behind §5 item 5 (ONE run on a torch host) |
| host reality | torch AND numpy both absent on this box (re-measured this minute: two `ModuleNotFoundError`s) | so no local run can promote the export leg — the blocker is the host, not the task |

**Verdict.** Struck as new work: the generator, the fixture corpus, and the pack-side golden duel are covered by shipped cells that this desk has already re-verified from landed bytes. Re-opening Task 2 would fork a second goldens chain, which is the single-home rule in reverse. Two residuals, both already on other rows and neither owned by Gemini: (i) the export-golden consumer needs the §5 item 5 torch-host run (chair's ask to the user; the cell exists, the host doesn't); (ii) `143650e4` is a FILESYSTEM-only pin today — agent2's registration-audit row 4 ("either ships or the citation stops") is the right owner for it, alongside `af31aecd`/`ec45cfde`, in one shepherd-in commit by its author.

## What this triage does NOT touch
The deploy-ledger review (the ledger has not landed in any branch I can read — `git for-each-ref` shows nothing matching it on any of the seven live refs) and the confirmation-fire kit formula check (same: not yet delivered where I can cite bytes). Both are queued as reviews, not guesses: I will not bless rows for a document I cannot open, and I will not pre-accuse a kit whose digest grammar I have not read.
