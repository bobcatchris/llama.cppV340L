# Session report — work order execution status (agent1, 2026-09-12)

**For:** coordinator review (C441). **Scope of this report:** the session that
executed `00_scope_and_work_order.md` from scoping through the T2 first contact.
**Bottom line:** Steps 0–2 of §6 are complete/near-complete; the session's
delegated task (scope → land → begin executing → deliver the first-token list →
start porting under grants) is DONE. The work order as a whole is **NOT
complete** — first token (M4-era Step 6) is unreached, and this report says so
plainly rather than claiming a finished WO.

## Step-by-step status vs §6

| Step | State | Evidence |
|---|---|---|
| 0 env + roofline | **DONE** | PG-0/0b + G-AMD-2b gap-closure; `results/v340l/step0_roofline.md` — sustained D2D ~183 GB/s at mclk top level, all 4 devices |
| 1 HIP build lane | **DONE** (product side) | `NINFER_BACKEND` switch, gfx900 gate, `HipSources.cmake` whitelist, shim dir, parity guard `CheckHipArchive.cmake` (falsifier-proven); PG-1 cells themselves = gemini's lane |
| 2 single-device primitives | **GREEN at build; verification partial** | device.cu/arena.cu compile as HIP; **gemini's PG-A passed on device** (first execution + DecodeGraph capture/instantiate/launch verified). Host layer 100%: targets+serve+engine+tp2 = 137/137 before kernels |
| 3 minimal op parity | **IN PROGRESS** | T1 glue 17/17 whitelisted (5 device-verified green: scalars/cast/scatter + position/embed/rope/layer_norm at G-AMD-8); T2 5/7 joined; **1 true red surviving: l2norm** (probe built, next stamp; argmax + sample-consistency reds have a NAMED harness cause fixed in v6 and a marker-proven fresh binary — likely green on re-run but UNPROVEN on device until then) |
| 4 layer forward | NOT STARTED | gated on step 3 completion |
| 5 TP2 fixtures (M2) | BLOCKED-BY-DESIGN, re-scoped | peer probe (G-AMD-5): **no P2P anywhere** → AR trio is a host-staged REDESIGN, not the 1k-loc translation WO §6 assumed. Q3-on-2-dies (user-confirmed milestone) additionally depends on the q3 team's loader/dispatch work |
| 6 TP4 full artifact (M4) | NOT REACHED | 19.03 GiB artifact not on box; not needed for the revised Q3 milestone |
| 7 AMD CI lane | gemini's lane (per §7.x) | PG-A passed (theirs); PG-0/0b/1 gate-wiring awaiting them against the specs in §6 |
| 8 report | THIS FILE | — |

## Grants executed (all closed with verified release rows; no device touched outside a grant)

G-AMD-1 (roofline), G-AMD-2 + 2b (in-load clocks, dev3 gap), G-AMD-4 (T1 batch-1 port+verify), G-AMD-5 (peer probe — the milestone-shaping matrix), G-AMD-6 (over-ran 14 vs 10 min — flagged unprompted), G-AMD-7 (**self-voided**: stale binary; the run reproduced pre-fix numbers; caught after launch, called it, log carries the verbatim note), G-AMD-8 (one launch, freshness proven in-log by marker-string + 3 hashes, answered 4/7→6/7 of the questions with the split named).

## Issues encountered (for review — full detail in PROGRESS.md + logs)

**Toolchain/port discoveries (all reproducible, several with committed repros):**
1. CMake silently DROPS `.cu` sources in HIP-only projects with **EXIT 0** — caught by object-count audit; fixed via `LANGUAGE HIP` + name-set parity guard. This made two earlier "green" claims false; both retracted on the record.
2. `add_custom_command` semicolon-flattens `-DSOURCES=…` — the guard itself saw "1 source vs 27 objects" before the SOURCES_FILE fix.
3. HIP `__bfloat16_as_ushort` is a **numeric cast** where CUDA's is a bit reinterpretation (−1.0f → ffff vs bf80; `amd_hip_bf16.h:537`, contradicts its own docstring). Poisoned two reference paths before named. Conversions (`__float2bfloat16`/`__bfloat162float`) verified EXACT.
4. CUDA's `cuda_bf16.h` transitively includes `cuda_fp16.h`; HIP's does not (caught via `tp2_backend.cpp` `__half`).
5. ROCm 6.2 has **no E4M3FN** fp8 (only FNUZ) — the shim implements the exact NV decode in software; aliasing would have been numerically wrong.
6. Three product PTX headers block the kernel tiers: `memory.cuh` + `math.cuh` (resolved via registered-exception guarded edits, ruling 1) and **`mma.cuh` (OPEN — third exception ruling requested; proposal: exact bf16/f16 fragment emulations + loud-abort stubs)**.
7. `hipDeviceCanAccessPeer` = 0 on ALL 6 pairs → the WO's "port the AR trio" assumption is invalidated at design level; collective COUNT is the new budget (~50 µs/hop host-staged, flat vs size).
8. Warp-shuffle width semantics (32-lane groups on 64-lane wavefronts) needed an emulation layer — still the prime suspect for the surviving l2norm red, though tracing kept consistency and suspicion has partially shifted back to harness row-slicing.

**Process/convention findings (adopted into docs/174 / PROGRESS):** `{d, rows}` ne[0]-is-feature-dim convention (bit twice); `argmax valid_rows` = vocab scan limit; nonblocking-stream + legacy-memcpy races fake kernel failures; content-based artifact provenance (marker strings) beats mtimes; a number that beats the measured ceiling is instrumentation lying (gemini's 103,755 GB/s, mine several nearly-weres); tallies must cite window+filter.

**My errors, named for the record:** the two false "green build" claims (retracted), the G-AMD-6 over-run, the G-AMD-7 stale-binary void, the initial "converters unreliable" overstatement (corrected to the named function with repro), and three separate shape-assumption harness bugs that cost device runs.

## Open items handed forward
1. **v7 re-run (one 60-s stamp)**: confirm argmax+sample greens predicted by the valid_rows fix, and answer l2norm via its half-vs-full discriminator probe.
2. **mma.cuh third-exception ruling** (proposal written).
3. AR redesign (host-staged, batched, pinned-buffer double-buffering) — design item for the Q3/TP2 plan-of-record, table + consequences already published to the q3 branch by C441.
4. gemini: PG-0b/PG-1 cells against the parity guard (reuse, not reinvent — C441 endorsed), plus the bf16-arithmetic compile-probe negative cell from the earlier finding.
5. Q3 artifact manifest read on arrival (11.1 GB arithmetic ≠ 13.67 GiB plan-of-record; nobody computes headroom until the file's own manifest is read).
6. `weights_id`/13 GB + `/reload` remain user-side.

**Definition-of-done audit honesty note:** WO §8 is NOT met (items 2–5 in particular). What IS met: every step claimed above carries a commit SHA + log/parity line, every measurement has its raw log, every retraction is on the record at the file level.

— agent1 (`dual_5060_ti_ninfer` lane session), HEAD `d9e8066f`
