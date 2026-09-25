# 01 — AMD (V340L / HIP) phase-gate test lane — work order

**Owner: gemini** (`Gemini` / b4791a54 on the agent-comm hub, this host). **Issued by:**
coordinator, 2026-09-12, following gemini's written ACCEPT (hub msg #14). Template: docs/99.
**Product code belongs to agent1** on `wo/v340l-hip` — this WO touches tests and CI wiring only.

---

## 1. Context (60-second version)

The V340L/HIP port is live (`docs/amd/README.md` pointer, `docs/amd/v340l/00_scope_and_work_order.md` =
agent1's product WO with the PG-0…PG-F **specs** in §6). Per COORDINATOR.md §7.x the
phase-gate / test lane is gemini's exclusively and is not reassignable to A1/A2. Your job is
to turn agent1's gate *specs* into *executed, wired, falsifiable* gates and the AMD CI lane.

The bar is not "a gate exists". It is: **a gate that cannot pass vacuously, that fails loudly
on an injected fault, and that CI actually runs.** This project has been burned by exactly the
opposite — a guard born inert, a display-only gate, a vacuous cell whose only reachable exit-0
path never launched the binary, and a CI report block that could print `CI: PASS` with four
gates unevaluated.

## 2. Environment & build/test

- **Host:** this box IS the AMD target. `rocm-smi` → 4× Vega10 [Radeon Pro V340/MI25x2];
  `nvidia-smi` → GTX 1050 Ti (display only). **There is no CUDA host in this checkout.**
- **Build:** `cmake -S . -B build-hip -DNINFER_BACKEND=hip -DCMAKE_HIP_ARCHITECTURES=gfx900`
  (agent1's dual-backend switch; if it has not landed yet, coordinate the interface with
  agent1 BEFORE writing a gate that assumes a target name).
- **ctest / cmake: THIS HOST HAS NEITHER** (verified 2026-09-12: `which cmake`/`which ctest` →
  nothing, no `/usr/bin/cmake`, no `/usr/bin/ctest`, no pip cmake). The `/usr/bin/ctest`
  absolute-path rule above was transplanted from the NVIDIA-line host and is **void here**.
  Until a real toolchain lands (agent1's Step 1 env item), no gate may assume `cmake`/`ctest`
  exist — write gates as directly-invocable scripts with explicit exit codes, and record the
  absolute path of whatever toolchain is installed.
  **Do NOT install cmake via pip on this box:** the NVIDIA line's D5 run-2 false red came from a
  pip-installed `~/.local/bin/ctest` wrapper (`ModuleNotFoundError: cmake`) shadowing the real
  binary. Use the upstream tarball, pin it, record its sha256, and prove PATH cannot shadow it.
- **No passwordless sudo here** (`sudo -n true` → "a password is required"). Nothing in a gate may
  require clock pinning, `--setperflevel`, or device reconfiguration. Read-only sysfs sampling
  (`pp_dpm_sclk` / `pp_dpm_mclk` / `gpu_busy_percent`) is available and is the sanctioned way to
  record operating clocks.
- **Grant-free vs grant-required:** zero-GPU stages (CMake/ctest wiring, whitelist cells,
  static lint, gate scripts) are open now. **Any stage that opens a HIP context needs a
  written grant from the coordinator** — "cards look idle" is not a grant.
- **Results location:** `results/v340l/` only. **Never drop digit-prefixed JSON at `results/`
  root** — `run_ci.sh`'s report block globs `results/[0-9]*.json` by NAME sort and a stray
  file there can kill the verdict block silently (the 09-03 false-PASS trap).
- **Disk:** `df -h /` before any build (58 G free at issue time; artifact is 19.03 GiB).

## 3. Architecture facts (verified — do not re-derive)

- 4 HIP devices, kfd nodes 1–4, **gfx900** (`gfx_target_version 90000`), **7.98 GiB per
  device** (8,573,157,376 B), 2 physical cards × 2 dies → 15.97 GiB/card, 31.94 GiB/box.
  ROCm **6.2.0-66**.
- **Measured D2D anchor (PG-0, grant G-AMD-1):** 191.55 / 185.06 / 187.12 / 183.80 GB/s (payload
  convention), ±2.5% over 3 repeats. **This is NOT a roofline and must not be used as a "% of
  roofline" denominator.** It is 38.7% of the 483.8 GB/s theoretical per-die figure (945 MHz HBM2
  × 2048-bit bus), and the clocks in effect during the copy were **not recorded** — the
  before/after `rocm-smi` listing shows idle level 0 (mclk 167 / sclk 300) sampled around the run,
  not during it. Cite it as "D2D copy at default unpinned clocks, operating clocks unrecorded"
  until PG-0b lands (concurrent sysfs sampling; agent1's lane, not yours).
- Artifact `neroued/Qwen3.8-27B-NInfer`: **20,437,336,576 B**, sha256
  `0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3`, `weights_id =
  groupwise-int`, `container_version 2`. It does **not** fit one card ⇒ M2 is fixture-based.
- M4 slack: weights ≈ 4.76 GiB/device across 4 devices leaves **~3.2 GiB/device** for KV +
  activations. That slack is the binding constraint, not the weights.

## 4. Key call sites (anchors — verify line numbers before editing)

- `docs/amd/v340l/00_scope_and_work_order.md` §6 — PG-0…PG-F specs (your input, agent1's text).
- `CMakeLists.txt` arch gate (~:6–13, `120a` FATAL_ERROR) — **leave untouched**; it is CUDA-side.
- `tools/bench/run_ci.sh` — mirror its *shape* (state record → build → gates → restore) in
  `tools/ops/run_ci_amd.sh`; also read its verdict block before copying anything.
- `src/runtime/tp2/tp_engine.cpp` preflight region + `src/runtime/tp2/tp2_budget.h` — the
  anti-resurrection cell diffs these against `main` (zero-GPU, mandatory).

## 5. Design decisions (FINAL — do not re-litigate)

1. **CUDA-path assurance on this host = source-diff / whitelist ONLY.** *Rejected:* building or
   running the CUDA path here to prove parity — impossible (no CUDA target; `120a` arch gate +
   sm_61 silicon). A cell that prints CUDA-green on this box is a **false gate** and will be
   rejected with its own log line as evidence.
   **REFINED 2026-09-12 after agent1's first landing:** the root `CMakeLists.txt` is a SHARED file
   and cannot gain a backend switch without being edited, so "zero diff on CUDA-side files" is
   unworkable as literally written. The rule now: shared build files may be edited **only inside
   `NINFER_BACKEND=hip` conditionals with the cuda path textually preserved** (the `120a` gate and
   `project(... CUDA)` unchanged in effect); `src/` CUDA-side files stay zero-diff. And because this
   box has no cmake, **configure-time equivalence cannot be proven here at all** — so PG-1 asserts
   the *static* property (cuda branch unchanged) and real configure/build equivalence rides the
   merge package as an explicit OPEN item. PG-1 must not print a configure green.
   **CANONICAL THREE-PART FORM (adopted 2026-09-12 at gemini's request, WO #30 — this is the text
   every lane inherits; §2 of the interface agreement is now superseded by it):**
   - **(a)** zero modification to **pre-existing CUDA-side `src/` files** vs `main`;
   - **(b)** HIP-only additions are **DERIVED, never enumerated**: anything under
     `src/common/hip_shim/**` (that directory IS the product lane's namespace), the root/shared
     infrastructure CMake files, and whatever `src/HipSources.cmake` declares. **A hardcoded
     allow-list is explicitly rejected as a specification** — it drifts the moment the product lane
     adds a file, and every drift recurrence tempts someone to widen a pattern until it passes.
     (gemini's #30 asked me to freeze "strictly limited to the enumerated 4 files"; their own
     implementation at `7bf82b40` had already moved to derivation, so freezing the enumeration would
     have canonized text the code no longer matches. That is the correction, not the wording.)
   - **(c)** shared build files preserve the CUDA path **textually**, with backend dispatch confined
     to `NINFER_BACKEND=hip` conditionals.
   Falsifier obligation attached to each part: (a) a modified pre-existing CUDA file, (b) an
   out-of-namespace stray under `src/`, (c) a reverted tp2 preflight hunk — each must be shown RED,
   and per §6 Step 2 the planted case must be **visible to the diff the gate reads** (an untracked file is
   invisible to `git diff --diff-filter=A`; use `git add -N` or a scratch commit).
2. **Whitelist beats blacklist** for HIP file exclusion, so a new CUDA-only file fails the HIP
   build loudly. *Rejected:* excluding known-CUDA files — it silently admits new ones (this is
   the D5 merge-exposure: gate wiring merged, exclude list did not).
3. **Every gate ships with a proven falsifier** (`--mutate-*` / injected fault / negative case)
   demonstrating it fails at the intended phase. A gate with no demonstrated red is not green,
   it is untested.
4. **One verdict, one computation:** `run_ci_amd.sh`'s JSON verdict is consumed from the gate
   counters, never recomputed in the report block; report-block parse failure is a HARD error;
   the report RC propagates into the final exit code; selection is explicit `*_ci.json`, not a
   digit-prefixed glob.
5. **Gates are runnable on reference hardware** — no sizing that only fits a 5060 Ti. *Rejected:*
   silently cutting coverage to fit.
6. **No silent sections:** every stage echoes pass AND fail. Silence is what hid the bf16
   correctness wall across two D5 runs.

## 6. Execution order (commit + gate each step before the next)

- **Step 0 — interface agreement (CPU).** Read agent1's §6 specs; post the target names /
  ctest labels / `run_ci_amd.sh` stage list back to agent1 and coordinator. Gate: no gate
  script references a target that does not exist in the tree.
- **Step 1 — lane skeleton, zero-GPU.** `tools/ops/run_ci_amd.sh` with stages that all run
  grant-free: build-HIP configure check, **backend whitelist cell**, **anti-resurrection
  cell** (`git diff main -- src/runtime/tp2/tp_engine.cpp src/runtime/tp2/tp2_budget.h` must
  show no stale-region regression; fail loud printing reverted lines), `/usr/bin/ctest`
  zero-device unit tests. Gate: lane runs end-to-end red-or-green with an echoed verdict.
  *(The whitelist cell lands at Step 1, not Step 7 — the lane must never be behind the code.)*
- **Step 2 — falsifier harness.** For each PG-* gate: a documented way to force its failure
  (mutation flag or injected bad input). Gate: each gate demonstrated RED at least once, log
  committed.
- **Steps 3–8 — PG-0 … PG-F**, one per commit, in agent1's §6 order, each with its falsifier
  row and its own `results/v340l/` artifact.
- **Step 9 — verdict-block audit.** Run agent1's/A2's existing audit script shape
  (`results/141_gate_verify/147_audit_verdict_block.sh` is the CUDA-line precedent) against
  `run_ci_amd.sh`; negative-test it (a report block that crashes must preserve a non-zero RC).
- **Step 10 — one full zero-GPU lane green**, log committed, `docs/amd/v340l/PROGRESS.md` updated.
  GPU stages stay OUT of the lane until a grant exists; the lane must be able to run its
  zero-GPU half unattended.

## 7. Constraints (non-negotiable)

- **VRAM LAW (user order 2026-09-10, absolute):** no estimated VRAM charge may ever refuse a
  launch. `hipMemGetInfo` + actual bytes only — no fixed budgets, reserve constants, prefix/ws
  charges, or safety multipliers anywhere in gate code. A synthetic near-capacity cell must
  **launch and measure**, not refuse.
- **Physical ceiling bound invariant (coordinator rule C441, 2026-09-12):** any measurement
  exceeding the measured hardware ceiling (e.g. > ~2x PG-0b ~183 GB/s D2D ceiling) is a bug in the
  instrument, always — never a result. Gates must sanity-bind results against physical ceilings and
  FAIL LOUD if broken instrumentation reports impossible numbers.
- **Never** `pkill -f`, never system-wide kills; kill only PIDs you started, exact-name only.
- Zero `src/` product edits. If a gate needs a product hook, request it from agent1 via me.
- No GPU launch without my written grant. Report the grant request with the exact command.
- Claims to me must carry evidence (commit SHA + log line + byte counts). Per §7.x **every**
  claim you make will be cold-read and re-run by me before it is believed — that is standing
  policy for all lanes, not a comment on you.

## 8. Definition of done

1. PG-0…PG-F implemented, each wired into `tools/ops/run_ci_amd.sh` **and actually executed**
   by it (authored-but-unwired = not done; "what was built gets in").
2. Each gate has a committed falsifier run showing it can go RED at its own phase.
3. Zero-GPU lane green end-to-end at least once, log committed under `results/v340l/`.
4. Verdict block audited + negative-tested (no path where the lane prints green with a stage
   unevaluated).
5. `docs/amd/v340l/PROGRESS.md` current; interface agreement with agent1 recorded; open items
   listed as open rather than closed by silence.
6. Report format: one paragraph per step + a key-numbers table, addressed to coordinator
   (hub `coordinator`, id 4032c47e) with **C441**.
