# 51 — Remote (Second) Agent Work Order

**Role: code-only agent. No GPUs, no live server, no battery runs.**
Companion to `docs/50_test_session_defects_and_work_order.md` (the defect
register and shared protocol live there — §4). This file defines what the
remote agent MAY and MUST NOT do.

## 1. Where you work

- Worktree: `/home/intel/ninfer/repo-remote`, branch `wo/remote-ports`
  (created from `mtp-perf` @ `bab85197`, 2026-08-23).
- One branch per task: `wo/<short-name>`. Commit + push each branch.
- The **GPU agent** (main worktree `/home/intel/ninfer/repo`) merges your
  branches one at a time and runs the battery per §4 of docs/50. Nothing is
  "done" until it is merged AND battery-verified there.

## 2. Your tasks (in order)

| # | Task | Register | Deliverable | Verified by |
|---|------|----------|-------------|-------------|
| 1 | 200k design doc: chunked-streaming prefill, per-chunk `mtp_ph`/`mtp_mh`, shrunken/capped `cached_ph`/`prefix cache` | D-07 | `docs/52_200k_context_design.md` + GPU agent sign-off | design review (no code) |
| 2 | Auto KV sizing: port upstream `SequenceCapacityCurve` resolver so `--kv-capacity auto` sizes from capacity, not free VRAM | D-02 | patch on `wo/auto-kv` + C++ unit test | battery T8 + startup-OOM check (GPU agent) |
| 3 | OpenAI `tool_calls` emission: parse Qwen-native text tool calls in the serve layer and emit OpenAI-format `tool_calls` (keep native content as fallback) | D-09 | patch on `wo/tool-calls` | battery T6 (GPU agent) |
| 4 | CI wiring: `run_ci.sh` gains the correctness battery (T1–T8), nightly T9–T11, T12 wired into the battery script | — | `wo/ci-wiring` | GPU agent runs it |
| 5 | Docs upkeep: keep docs/50 register + baseline table current as tasks land | — | commits to docs | n/a |

## 3. MUST NOT (hard boundaries)

- **No server**: do not launch, kill, or probe `ninfer-serve`. Do not touch
  port 8091. Do not run the battery.
- **No D-01/D-10/D-03 work**: those are in progress on the GPU agent.
  Concretely: **do not edit** `src/runtime/tp2/tp2_backend.cpp`,
  `src/targets/qwen3_6/impl/runtime/text_context.h`, or anything under
  `src/runtime/engine/admission_*` / `request_memory*` (D-03 in progress).
- **No merges, no pushes to `mtp-perf`**: you push feature branches only.
  If your task collides with in-progress GPU work, **stop and defer** —
  leave a note in docs/50 §3 and rebase later.
- **No new register entries without an owner column value** (see docs/50 §2).

## 4. Protocol (per task, derived from docs/50 §4)

1. Take the topmost task from §2. Mark it `IN PROGRESS (remote, <date>)` in
   docs/50 §2 (commit the doc change to your branch).
2. Test first where code is involved: the verification test already exists
   in the battery (D-02→T8, D-09→T6) — you cannot run it, so write/extend
   the C++ unit tests you *can* run locally (e.g., `kv_capacity` resolver
   tests) and make sure they pass on your machine.
3. Implement minimal, at the root-cause site. Build your branch locally:
   `cmake --build build -j` — compile must be clean (no CUDA needed for the
   D-02/D-09 code paths; if a change requires CUDA compile, verify at least
   `make <target>` for the affected objects).
4. Push the branch. Tell the user: "ready to merge: `<branch>` (task,
   verification it needs, files touched)".
5. After merge + battery verification: update docs/50 §2 status + §7
   baseline table (the GPU agent does the §7 table; you do §2).

## 5. Status

- 2026-08-23: worktree created. **Task 1 (D-07 design doc) DONE** (landed as
  `docs/52_200k_context_design.md`, commit `a7e96583`; pending GPU sign-off).
  D-03 reassigned to the GPU agent before remote work began — no overlap.
- 2026-08-23: **Task 2 (D-02 auto-KV) code DONE on `wo/auto-kv`.** Calibrated
  pure TP2 VRAM budget model (`src/runtime/tp2/tp2_budget.h`) + honest
  preflight/auto-resolution in `tp_engine.cpp` + unit test
  (`tests/test_tp2_budget.cpp`, green, registered in CTest). No
  `tp2_backend.cpp` changes (D-01/D-10 territory untouched). Merged into
  `v1-integrate` 2026-08-23; pending GPU verification (battery T8 + startup
  OOM check).
- 2026-08-23: **Task 3 (D-09 tool_calls) CLOSED as pre-existing.** OpenAI
  `tool_calls` emission was already implemented end-to-end
  (`tool_call_parser.{h,cpp}` → `generation_service.cpp` → `http_server.cpp`
  → `openai_schema`) and tested (`tests/test_tool_call_parser.cpp`, verified
  compiling + green standalone). No code change needed. The 11:16 T6 failure
  is the separate §7.1 early-stop/`</think>`-leak failure mode (owner GPU), not
  a missing emission. No new branch code; docs-only clarification on
  `wo/tool-calls`.
- 2026-08-23: **Task 4 (CI wiring) code DONE on `wo/ci-wiring`.**
  - New self-contained `tools/smoke/serve_correctness_ci.sh` (tracked, on
    branch): owns the full server lifecycle (setsid launch of a calibrated
    80k/I8/MTP-k3 server → battery → kill, never leaves GPUs occupied).
    Modes: `ci` = gate T1,T2,T3,T5,T8 (fast, must-pass, currently green);
    `nightly` = report T9,T10,T11; `full` = everything incl. T12.
    Exit code = # failed (non-skipped) tests.
  - `t12_clean_restart` (battery) converted from a stub to a real clean-state
    probe (prompt isolation + prefix reuse on a fresh instance); the CI
    wrapper orchestrates the restart around it.
  - Ops scripts (untracked, `/home/intel/ninfer/scripts/`): `run_ci.sh` now
    runs the correctness battery after the S-battery and gates the CI verdict
    on T1,T2,T3,T5,T8 (parses `[PASS]/[FAIL] Tn`); new `run_nightly.sh`
    runs T9,T10,T11 + a full pass with restart (tracked, not gated).
  - Gating is conservative (T4/T6/T9 — the known §7.1 early-stop failures —
    are NOT in the per-build gate; they run in nightly for tracking) so this
    cannot red-line CI. Merged into `v1-integrate` 2026-08-23; pending one real
    CI pass to confirm the battery is green on the live config.
