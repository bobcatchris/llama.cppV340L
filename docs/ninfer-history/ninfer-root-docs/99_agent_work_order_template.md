# NN — <Task name>: Agent Work Order (boilerplate)

> **Template.** Copy this file to `docs/NN_<short_name>_agent_work_order.md`
> (next free number), fill every `<...>` placeholder, delete the guidance
> notes, and hand the agent: "work on this task in this doc". Complete
> filled examples: `docs/57_kvarn_p2_agent_handoff.md` and
> `docs/68_kvarn_pp_speedup_agent_work_order.md` (work order).

**Status:** CURRENT — work order for a single implementer agent.
**Mission:** `<one paragraph: what "done" means, end state the user sees>`
Read this document fully before writing code.

---

## 1. Context (60-second version)

`<Why this task exists, what problem it solves, and what must NOT change as a
result. Name the docs that hold deeper background (e.g., "docs/54 §9 has the
history").>`

**Already built and verified (do not redo):**
- `<bullet list of prior work with commit hashes + test names>`

**What you are doing:** `<steps N..M of §6 below, in one line each.>`

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; remote `github`).
- **Work protocol (MANDATORY):** ALL work happens in a worktree + branch —
  never edit the main tree at `/home/intel/ninfer/repo` directly. Working in
  the main tree is how uncommitted agent files got swept into unrelated
  commits on 2026-08-24 (`git add -A` incident; steps 3 code landed inside
  docs commits `83da326e`/`0b92f431`).
  ```bash
  cd /home/intel/ninfer/repo
  git worktree add ~/ninfer/worktrees/wo-<short-name> -b wo/<short-name>
  cd ~/ninfer/worktrees/wo-<short-name>
  cmake -S . -B build && cmake --build build -j 16   # own build dir, one-time full build
  ```
  Commit per step to `wo/<short-name>` and push. **Merging to main is done by
  the main-side agent/user** — do not merge or push to main yourself.
- **Communication & routing (MANDATORY — read before sending ANY message):**
  - **Two channels, two audiences — do NOT mix them:**
    - **`intercom`** = ALL agents **except gemini** (A1, A2, and any other
      local pi session, including the coordinator). This is the default
      channel.
    - **`agent_comm`** = **gemini ONLY.** Never send to gemini via intercom;
      never use agent_comm for A1/A2 or any non-gemini agent.
  - **Why this matters (identity):** each channel binds to a specific
    identity. Using the wrong channel can make your message appear to come
    from the WRONG agent — e.g., an `agent_comm` message sent from a
    non-gemini session can be delivered under the coordinator's identity
    (2026-09-04: A1's agent_comm handoff to gemini was delivered as from the
    coordinator, so gemini could have acted on it as a ruling). If you are
    not sure which channel to use, ASK the coordinator — do not guess.
  - **How to find who to send to (always verify — names/IDs change):**
    - **intercom** (A1/A2/coordinator/other pi sessions):
      `intercom({action: "list"})` lists active sessions by name + ID.
      Address by name, full session ID, or the short ID in parentheses.
      (`list-cwd` scopes to a worktree directory.)
    - **agent_comm** (gemini): `agent_comm({action: "list_agents"})` lists
      agents on the mesh; gemini is the non-coordinator agent there.
  - **Never impersonate:** send only as your own identity. If a tool appears
    to send as someone else's identity, STOP and report it to the
    coordinator (same class as the 06:25 pkill incident — a tool doing
    something dangerous silently).
- Build (inside your worktree): `cmake --build build -j 16`
  (CUDA arch forced to `sm_120a`; 2× RTX 5060 Ti 16 GB).
- Unit tests: **use `/usr/bin/ctest`** (the `ctest` on PATH is a broken Python
  wrapper). From your `build/`: `/usr/bin/ctest -R "<pattern>"`.
- **Test entry points (standard — verify before use, do not re-derive):**
  - **ALL server tests go through `bash tools/ops/run_ci.sh`, run from your
    worktree's root** (relative path — NOT `~/ninfer/scripts/run_ci.sh`: that
    symlink resolves to the MAIN tree and would silently test main's code,
    not your build). It stops any running server, builds, runs the test suite
    against YOUR build, and restores the live server at exit (even on
    failure). You have full agency over the server — stop it, start it,
    whatever the work needs. A healthy server on port 8091 is not your build
    unless the pipeline started it — never curl-test a server you did not
    start.
    - `bash tools/ops/run_ci.sh` = fast per-build gate (build + verify +
      serve S1–S5 + T1,T2,T3,T5,T8).
    - `bash tools/ops/run_ci.sh --full` = the entire suite (int8 T1–T12 +
      KVarN + T14 zone) — for closeout, not per-step iteration (too slow).
  - Targeted runs: `tools/smoke/serve_correctness_ci.sh` — env-overridable:
    `KV_DTYPE=... KV_CAPACITY=... MAX_CONTEXT=...` (defaults are stock int8),
    `SPEC=... DRAFT_TOKENS=...`; modes `--mode ci` (fast gate), `--mode full`
    (T1–T11 + clean-restart T12), `--only T<n>,...` always wins. Exit code =
    number of failed (non-skipped) tests. It manages its own server lifecycle
    (launch → suite → kill); if a server already holds the port the launch
    fails — do not curl-test an existing server.
  - Must-pass / expected test lists: **docs/50 §7.1 is the source of truth** —
    read it before running the test suite. T19 (beyond-wall prefill)
    threshold (≥450 tok/s avg) is never modified. T14 needs a tolerance zone
    (`kv_capacity > max_context`; the production 250k/250k KVarN config has
    none — `run_ci.sh --full` runs it on 200k/250k automatically).
  - GPU unit/bench binaries: `build/tests/<name>`, run from your worktree
    (build first: `cmake --build build -j 16`).
  - Ad-hoc live server: launch per LAUNCH.md via `setsid` (one server at a
    time — the two GPUs can only hold one); check `pgrep -x ninfer-serve` +
    `nvidia-smi` before every launch (a stale ~14 GB/rank server OOMs
    startup).
  - Measurement methods: prompt sizes from `usage.prompt_tokens`, NEVER char
    estimates (calibration ~5.33 chars/token; the inverted ratio once built a
    12.7M-token prompt). Decode rate from serve.log (`decode=Xtok/s`).
    `nvidia-smi` clocks recorded with every perf number — environmental drift
    of ~8%/hr has been observed; do not chase single-run deltas.
  - Results: every probe/run output committed under `results/` with its
    server config + clocks (measurement data is project data).
  - **Test gates — how to USE the suite when something is broken.** The
    phase gate is a FAST point-of-failure localizer for the KVarN pipeline in
    general; the MTP T0-T3 suite (docs/131) does the same thing for the MTP
    multi-batch pipeline specifically — MTP is the surface that has broken
    this project repeatedly, so when MTP breaks, start here, not with a
    manual hunt:
    - **MTP point-of-failure workflow** (docs/131 = spec, `docs/131_runbook.md`
      = runbook; all scripts run from the worktree root):
      1. `NINFER_MB_INVASERT=1 bash tools/bench/run_t1.sh` → the first
         structured line `INV=<n> round=<r> layer=<l> lane=<b> expected=<e>
         actual=<a>` IS the point of failure. Map it to a component via the
         docs/131 §9 cause map. (INVASERT is zero-cost when unset.)
      2. No invariant fired but tokens diverge (probabilistic/race):
         `NINFER_MB_HASHPT=1` ×16 runs → first divergent (round, point)
         localizes the race; then `bash tools/bench/run_t0.sh` (tile
         round-trip + stride-audit batteries) for content/geometry bugs.
      3. E2E confirmation: `bash tools/bench/run_t2.sh` golden matrix
         (ISOLATED fresh-process single-seq reference — never the in-process
         one; docs/130 §6 contamination).
      4. T0 batteries also run standalone (seconds, no model):
         `bash tools/bench/run_t0.sh`; `tools/bench/run_all.sh` = everything.
    - **KVarN phase gate (pipeline in general):** the standing steps in
      `tools/ops/run_ci.sh` + `--full`, and the M0 multi-batch phase gate
      `tools/bench/phase_gate_mb.sh` (rank-gated dumps). Gate rule (docs/126/
      128): the FIRST failing phase is the point of failure — do not fix
      downstream phases until it is green.
    - **Merge gate (docs/131 T3.1):** any commit touching the MTP/kvarn
      decode path runs T0 + T1-quick + INVASERT + T2(T=64); full before merge.
      A red from the existing battery OR the T0-T3 suite blocks merge.
- `<task-specific extra entry points only: bench/probe scripts this work
  order builds, new T-tests, special configs>`
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.

## 3. Architecture facts (verified — do not re-derive)

`<Bullet list of the load-bearing facts an outsider would otherwise get wrong,
each with a file anchor. Include: which engine path actually runs (TPEngine,
NOT ConcurrentExecutor — REPO.md §2a), model geometry, page/tile sizes,
storage layouts with exact shapes and byte counts, and the APIs to reuse.>`

## 4. Key call sites (anchors — verify line numbers before editing)

- `<file:line — what lives there and why it matters>`
- `<...one per site the implementer must touch or understand>`

## 5. Design decisions (FINAL — do not re-litigate)

1. `<decision, with the one-sentence reason it beats the alternative>`
2. `<...>`

`<For anything rejected, write "REJECTED: <reason>; revisit if <condition>" so
future agents don't re-derive it.>`

## 6. Execution order (commit + test each step before the next)

**Testing standard (applies to every step):** tests must simulate REAL
behavior, not kernel-level interactions in isolation. A step that touches any
server-facing path is NOT done on unit tests alone — the actual call sequence
the server performs (chunked prefill, prefix restore, MTP rounds, request
lifecycle) must be exercised end-to-end: **spin up the server with the new
code and execute the code path and its dependencies** (test suite or live
request). Kernel/unit isolation is necessary, never sufficient. The 2026-08-24
KVarN defect ("append jumped pages") passed every kernel-level test and still
failed on the first live request — that is the failure mode this rule exists
for. For MTP regressions/divergences/flakes, start from the docs/131
point-of-failure workflow (§2: INVASERT → cause map / HASHPT → T0), not from
manual bisection — the suite exists to localize in minutes.

### Step N — <name>
`<what to implement, 2–5 sentences, referencing §3/§4 anchors>`
**Tests (must pass before moving on):**
- `<concrete, runnable test spec — not "test it works">`
- `<live-path test if the step touches a server-facing path: which test /
  which request sequence exercises the real call pattern>`

### Step N+1 — <name>
`<...>`

## 7. Constraints (non-negotiable)

- **Worktree only:** no edits in `/home/intel/ninfer/repo` outside your
  worktree, ever (§2).
- **Live end-to-end before "done":** a step touching a server-facing path is
  not complete until the real server runs it — spin up, execute the code path
  with its dependencies, verify the response. Kernel-level parity alone does
  not count (§6 testing standard).
- **No damage:** never leave uncommitted or untested code; commit per step with
  a message naming the step; keep the tree buildable at every commit.
- **Server agency:** you have full agency over the server — stop it, start
  it, reconfigure it, whatever the work needs. The standard way is
  `bash tools/ops/run_ci.sh` from your worktree root (§2): it stops any
  running server, tests your build, and restores the live server (even on
  failure). `pkill -x ninfer-serve` (exact name; NEVER `pkill -f`); check
  `pgrep -x ninfer-serve` + `nvidia-smi` before every launch. At the end of
  your work, leave the live server restored per LAUNCH.md (or say so in the
  report).
- `<task-specific don'ts: files not to touch, paths not to modify, invariants>`

## 8. Definition of done

1. Steps N..M committed to `wo/<short-name>` with passing tests at each step
   — including the live-path test for every server-facing step.
2. **Live proof:** a fresh server launch with the new code serves a real
   request through the changed path successfully (test suite T16-style or the
   task's own live sequence), and that launch command is recorded in the
   report.
3. `<functional end state, e.g., "flag X launches cleanly, gate Y green">`.
4. `<measurement data committed to results/ — measurement data is project data>`.
5. **Test gates (docs/131 T3.1):** if the work touched the MTP/kvarn decode
   path, `run_t0.sh` + INVASERT stress + T2(T=64) are green on the final
   commit; the merge gate runs via `run_ci.sh`.
6. Report format when done: one paragraph per step + the key numbers table.
