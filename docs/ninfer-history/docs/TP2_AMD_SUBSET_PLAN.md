# TP2 AMD-SUBSET PLAN — Team Red path from "Green lands q3-quant-TP2" to "TP2 serving on 2× gfx900 dies"

**Author:** agent3 (01a095d8), 2026-09-12, per user directive. **Coordinator distributes this
plan** (stamps WOs, assigns lanes, grants GPU). Staffing: **2 executor agents (Agent-A, Agent-B)
+ Gemini (tests, exclusive lane)**.

**Mission (user framing):** Team Green (NVIDIA) is wrapping up q3-quant TP2 upstream. Part is
done; when it lands, Team Red makes TP2 work on this box with a **small subset of AMD-specific
changes**. This plan defines that subset, who does what, in what order, with which gates.

---

## §1 Ground truth — the three codebases and the integration point (verified on bytes)

| Lineage | Where | What it is | Role for us |
|---|---|---|---|
| **Upstream / Green (the spine)** | `origin/main` → `src/runtime/tp2/` (20 files: `tp_engine.cpp`, `tp2_backend.cpp` 1,277 LOC, `tp2_budget.h` **[VRAM canonical]**, dflash2/mtp modules, `host_kv_parked`) + `src/core/multi_gpu/` (`tp_group.h`, `one_shot_allreduce.h`, `weight_shard.h`) + q3 rowsplit shard kernels (`q3_rowsplit_{gemv,gemm_simt,storage}`) | NVIDIA TP2 incl. fused `allreduce_argmax`, MTP/dflash2 at tp2 | **The target.** We port THIS, never replace it |
| **Fork (the donor for AMD mechanics)** | ninfer-gfx906 @ `7a3c18d` | Donor-lineage TP2 (allreduce.cu event transport) — **different architecture from upstream's** — plus HIP compat, gfx900-runnable kernels, probes, stage logs | **Reference + parts bin**, adapted to Green's interfaces. NOT a merge |
| **Ours (amd/main)** | `amd/main` @ fc4da3ca (+ `amd/wo-q3hip`) | HIP shim, RULING-1 idiom, CI lane, step-0 guard fix (`f5035616`) | The integration base |

**The AMD-specific subset (the whole job, bounded):**
1. `src/core/multi_gpu/` transport on HIP — `one_shot_allreduce`/`tp_group` need peer/staging
   primitives; this box has **no P2P (G-AMD-5)** → staged/event transport. The fork's
   `allreduce.cu` is the working ROCm reference for exactly this shape (3-phase event
   choreography, clean no-P2P fallback, proven on a P2P-less pair) — **adapted behind Green's
   `tp_group` interface, not swapped in**.
2. HIP compile of `src/runtime/tp2/` host majority (`tp_engine`, `tp2_backend`, `tp2_request`,
   `host_kv_parked`) + device minority (`dflash2_*.cu`) under RULING-1 lanes.
3. q3/q2 shard ops on HIP (WO-04, in flight — step 0 done; fork has no q3, this is genuinely
   ours).
4. Ops the tp2 path launches: fork's kernel dir already runs on gfx900 via its fallback arms
   (§2b census, `795640cf`) — adopt, don't rewrite.
5. Shim closure: 6 missing TP2 surfaces in `hip_shim/cuda_runtime.h` (measured) + `warp.cuh`
   wave64 discipline (G-AMD-10 reference).
6. Tests: fork's TP2 battery ported + comparative parity gate **against OUR artifact**
   (fork's artifact ≠ ours: 18.21 GB/`eec39564…` vs 20.44 GB/`0634abb0…` — their goldens never
   transfer).

## §2 Assignments

| Role | Owner | Lane | Exclusive files (extends §3 WO-04 rules) |
|---|---|---|---|
| **Agent-A** (me, agent3) | TP2 integration lead | transport adaptation, tp2 HIP compile, q3 shard ops, probes | `src/core/multi_gpu/*` HIP lanes, `src/runtime/tp2/*` HIP lanes, `src/ops/linear/q3/*`, `q2/*`, fork-adoption kernel lanes |
| **Agent-B** (suggest agent2 — owns the shim; coordinator confirms) | compat + runtime sizing | shim closure, budget review, upstream-delta watch | `src/common/hip_shim/*` (already agent2's), `tp2_budget.h` **read-only both lines — canonical** |
| **Gemini** | tests, exclusive (2026-09-03 rule) | port fork TP2 battery, parity gate, CI cells | `tests/**`, `tools/ops/run_ci_amd.sh`, gate wiring |

Standing rules unchanged: GPU only on written coordinator grants, serial card queue; zero-GPU
stages via `run_ci_amd.sh` grant-free; `tp_engine.cpp`/`tp2_budget.h` = canonical (Green's
spine AND the anti-resurrection region — the fork never touches them); VRAM LAW (measured
only, no estimate refusals); merge direction main→amd/main only.

## §3 Phases and work-order drafts (coordinator stamps; template docs/99)

**P0 — zero GPU, all three lanes in parallel, start now**
- **WO-TA1 (Agent-A): tp2 HIP compile census.** Compile matrix of `src/runtime/tp2/` +
  `src/core/multi_gpu/` under hipcc/gfx900 (syntax-only cells, zero GPU): first genuine error
  list = the AMD-subset defect list. Gate: cell log pasted; static scans inadmissible
  (agent1's 4/6 lesson).
- **WO-TA2 (Agent-A): fork kernel adoption.** Port fork gfx906 kernel dir (1,285 LOC) +
  `warp.cuh` discipline into RULING-1 lanes for the op families tp2 launches. Gate: compile
  green + existing CPU goldens unchanged (md5).
- **WO-TB1 (Agent-B): shim closure.** Land the 6 missing surfaces
  (`cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaMemoryTypeDevice`,
  `cudaErrorPeerAccessAlreadyEnabled`, `cudaStreamIsCapturing`, `cudaStreamCaptureStatus*`);
  verify `hipPointerAttribute_t.type` field on ROCm 6.2.0 (fork verified 6.4.1 only).
  Gate: compile cell + gemini's negative test (shim must fail loud on misuse).
- **WO-TG1 (Gemini): test battery + parity spec.** Port fork's 17 TP2 tests
  (allreduce, shard map, split ops ×7, headlocal, kv_capacity_tp2, executor, engine_tp2_real,
  mtp_tp2_real, graph_tp2) as AMD gate drafts; write the comparative parity spec (argmax / KL
  / 1−cos envelope vs OUR tp1 control, OUR artifact); AMD tp2 MODE for `q3_ci.sh`.
  Gates themselves are gemini's deliverable — agents never edit them.

**P1 — one coordinator-stamped GPU session, minutes (blocking gate for everything below)**
- **WO-TA3 (Agent-A, execute; Gemini's harness): port + run the 4 probes** (`p2p`,
  `transport`, `capture`, `replay`) on 2 dies. Output: measured facts table on ROCm 6.2.0 —
  P2P absence behavior, event transport correctness, cross-device capture semantics
  (fork's S8 graph-executor bug is 6.4.1-reported; ours unmeasured), graph device routing.

**P2 — on Green's landing (merge main→amd/main), zero GPU**
- **WO-TB2 (Agent-B): landing delta map.** Diff study of the landing using the fork's
  TP2-SLICES partition method (they already did this exact study for the donor lineage —
  reuse the method): changed files → conflict classes → slice order.
- **WO-TA4 (Agent-A): HIP lanes for the landing.** RULING-1 compat for new/changed shard ops
  + tp2 modules; transport adaptation behind `tp_group` (staged/event, no-P2P per probe facts).
  Gate: full HIP build green + P0 goldens unchanged.

**P3 — GPU, stamped: bring-up on 2 dies (M2, fixture-class: 19.03 GiB artifact > 15.97 GiB —
runs are capacity-capped and MEASURED per VRAM LAW; cudaMalloc is the only refusal)**
- **WO-TA5 (Agent-A) + WO-TG2 (Gemini gates): tp2 eager bring-up.** Order per fork's proven
  sequence: allreduce test → shard-map tests → split-op tests → attention headlocal →
  engine_tp2_real → parity. **Eager only** — graphs/flag-sync rejected until P1 facts + S9c
  wedge class excluded. Gate: gemini's parity PASS within envelope, tp1 md5s unchanged.

**P4 — real serving (M4 = 4 devices / 2 cards): blocked on coordinator R5 ruling** (engine
2-rank limit vs 4-device sharding of the 19.03 GiB artifact — needs either upstream tp4 work
or a ruling that M4 waits on Green).

## §4 Green interface contract (coordinator routes to Team Green)
1. Landing commit sha + advance notice (we merge within hours, not days).
2. The transport primitive list `tp_group`/`one_shot_allreduce` actually requires
   (peer memcpy? event waits? fused argmax semantics?) — so WO-TA3 probes target the real
   surface, and WO-TA4 adapts the right thing.
3. Their TP2 test list + the `q3_ci.sh` tp2 MODE contract (c419b7c0 notes TP2 path not yet
   format-covered — we inherit that cell at bring-up).

## §5 GPU budget
Three stamped sessions total, serial: P1 probes (minutes) → P3 bring-up (hour-class) →
P3 parity (one long run, fork's comparable phase took ~25 min on slower software). Everything
else is zero-GPU and grant-free.

## §6 Risks carried
R1 flag-sync/graphs rejected until probed (S9c wedged a card, MODE1 reset FAILED).
R2 all 6.4.1 behavior claims re-measured on 6.2.0 (P1) before anything depends on them.
R3 perf: expect a fraction of fork t/s (484 GB/s/die, no V_DOT; fallbacks already in code).
R4 VRAM LAW: budget constants in `tp2_budget.h`/`kv_capacity` are planning-only; any adoption
reviewed against the canonical preflight; launch refusal = live cudaMemGetInfo compare only.
R5 M4/2-rank tension (above).
R6 anti-resurrection: cherry-pick/RULING-1 lanes only; `git diff main -- src/runtime/tp2/
tp_engine.cpp src/runtime/tp2/tp2_budget.h` stays empty through every step.

## §7 Coordinator decisions needed
1. Confirm Agent-B = agent2 (shim ownership) or reassign.
2. Stamp P0 WOs now (TA1, TA2, TB1, TG1 — all zero-GPU) and schedule P1's probe session.
3. Route §4 asks to Team Green.
4. Rule R5 (M4 path) before P3 completes, so P4 isn't a cold start.
