# v340l — HIP/ROCm port of ninfer to AMD Radeon Pro V340L

**Status:** CURRENT — scoping pack for the V340L port lane.
**Created:** 2026-09-12 by the scoping session (agent1).
**Folder-local numbering** (00_, 01_, …) is blessed by the coordinator
(C441, 2026-09-12); global `docs/NN_` stays coordinator-claimed — `docs/174`
is the AMD line's single global pointer doc.

## Files in this folder

| File | Purpose |
|---|---|
| `00_scope_and_work_order.md` | The filled agent work order (per `docs/99_agent_work_order_template.md`) — the scoped task |
| `01_amd_phase_gate_test_lane_work_order.md` | Gemini's test-lane work order (phase-gate implementation + `run_ci_amd.sh` wiring — gemini's exclusive lane) |
| `PROGRESS.md` | Running progress notes / phase-gate log (append-only per step) |
| `02_session_report_2026-09-12.md` | Session completion report for coordinator review (honest DO-D audit + issue log) |

## One-paragraph summary

Port the existing TP2 (2× 5060 Ti) serving path to HIP/ROCm so the
**groupwise-int** Qwen3.8-27B artifact runs on AMD V340L cards — first on
2 HIP devices (one card, fixture-based correctness; the 20.4 GB artifact does
not fit on one card), then on 4 HIP devices (2 cards, full-model serving,
tensor split). Scope is deliberately minimal: plain decode only (MTP/DFlash
off), the minimal kernel subgraph on the active serving path, a dual-backend
CMake gate so the CUDA build is never touched, phase-gate tests at every
step, and a dedicated AMD CI lane (`run_ci_amd.sh`).

## Commit convention for the test lane (agreed with coordinator C441, 2026-09-12)

Shim/host additions are expected drift on this lane (per-file kernel join rule,
WO §6 Step 3). To keep gemini's `ALLOWED_SRC_ADDITIONS` gate trustworthy:

1. Any commit that adds a file under `src/common/hip_shim/**` or extends the
   `src/HipSources.cmake` whitelist says so **in the commit subject**
   (`+shim:` / `+whitelist:` tokens), body lists exact paths.
2. Derivation is available without restating: shim paths are this lane's
   namespace (`src/common/hip_shim/**`), and the whitelist is machine-readable
   from `src/HipSources.cmake` — recommended for gemini's cell per C441.
3. A red addition-gate after a `+shim:`/`+whitelist:` commit is EXPECTED
   drift → ping agent1/coordinator before anyone widens a pattern.
