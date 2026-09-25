# 109 — Agent 1 CPU sprint (interim: until the GPU opens ~11:25)

**Status:** INTERIM — extends docs/105 for the window before the GPU is free.
**Mission:** Use the GPU block productively: implement docs/105 **step 1**
(the 4-stage pipeline) now, prep the D-21 battery harness, and save the
pre-fix binary — so that the ~11:25 GPU window is *verification only* and the
fix lands hours earlier.

---

## Why you are blocked (and what is not)

The GPU is held by the official baseline run
(`~/ninfer/logs/official_baseline2_20260829_090807.log`, ends ~11:25) until
then: no server, no battery. **But docs/105 step 1's implementation is pure
CPU** — code, build, ctest. Only the byte-identity + perf gates need the GPU.

## Task 1 — Implement step 1 NOW (docs/105 §6 step 1, code half)

- Add the `Stages` template parameter (default `kBf16GdnStages`) through
  `launch_bf16_prefill_mma` and the kernel (cp_wait/stage_load arithmetic,
  smem sizing, `cudaFuncSetAttribute`); pass `Stages=4` from
  `bf16_gdn_gating_proj_mma_unsplit_launch` when `is_27`. Nothing else.
  (docs/105 §3-5 have the design + the bit-identity invariant — re-read them
  before touching the kernel.)
- Build: `cmake --build build -j 16`. **Before your first wip rebuild, copy
  the current pre-fix binary:**
  `cp build/apps/ninfer-serve ~/ninfer/bin/prefix_ninfer_serve_$(git rev-parse --short HEAD)`
  — the ~11:25 pre-fix battery capture (docs/105 step 0) runs THIS binary.
- Tests: `/usr/bin/ctest -R "gdn|kvarn|spec"` green.
- **Commit policy (exception to "no untested commits"):** you MAY commit now
  with message `wip(gdn): 27-model MmaUnsplit 4-stage pipeline (docs/105, GPU
  gate pending)`. Plan owner accepts the wip; when the ~11:25 gates pass you
  reword it (interactive rebase) to the formal step-1 message with evidence
  in the body. If you'd rather hold the diff in a stash, that's fine too.

## Task 2 — Battery harness prep (CPU, docs/105 step 0/1 driver)

Assemble the driver so the ~11:25 window is execute-only:
- Script that: starts the server (the pre-fix binary from Task 1's copy, then
  the wip build), runs the D-21 battery — temp>0 robot/sunsets seeds 1–6
  (assert no CJK + coherent, per results/kvarn_temp_sampling_root_cause.md
  "How to re-verify"), H5 at k=1/k=2/k=3 (`tools/ops/h5_retest_mtp_long.sh`),
  6-prompt battery k=2 — and byte-diffs every output against
  `results/105_prefix_battery/`.
- Save the captured pre-fix outputs to `results/105_prefix_battery/` during
  the window (that IS step 0).
- Exact commands are in results/d21_mtp_verify_anchor_fix.md "Pending server
  items" — do not improvise the battery.

## Task 3 (optional, only if Tasks 1-2 are early)

- Perf-gate extractor: script that parses a CI JSON into the docs/105 gate
  table (plain ≥35.0 t/s, verify ≤35.6 ms, mtp ≥79.0) with PASS/FAIL.

## At ~11:25 (GPU opens — plan owner will ping)

Execute docs/105 steps 0-3 exactly as written: pre-fix capture → wip-build
battery A/B → perf gate → (step 2 only if step 1 missed) → full CI → docs.

## Constraints

- Worktree `wo-kvarn-hold` only; no merges; do not touch KVarN/sampler/D-21
  fix/35-model paths (docs/105 §7). No GPU before the ping. `pkill -x`
  only. Commit messages name docs/105 (wip prefix allowed per Task 1).
