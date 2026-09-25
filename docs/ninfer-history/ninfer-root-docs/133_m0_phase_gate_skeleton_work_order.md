# 133 — Work Order: M0 — doc-128 chain skeleton + D1→D2 (unified-kernel certification pipeline)

**Owner:** gemini (test creator) — spawn pending; deliverable on spawn.
**Coordinator:** 01a05fb6. **GPU slot:** claim via intercom (2 short runs at acceptance).
**WBS ref:** docs/132 §11.3 item 3a. **Read first:** docs/128 in full (§6 M0,
§7 risks, §8 execution spec — it IS the spec), docs/131 (test-suite patterns),
docs/132 §11.3 (where this fits).

## 1. Context

The unified kernel (docs/104 Phase 2-3) is built; what's missing is
certification: the doc-128 phased gate pipeline (chained oracles on
phase-boundary artifacts) that localizes any future kernel regression in one
CI run instead of ~1 day of bisection. This work order builds **M0**: the
chain skeleton with the first two phases (D1 commit/quantize, D2
dequant/materialize) — the foundation M1 (full decode chain) and M2 (CI
wiring) build on.

## 2. Environment & test entry points

- **Worktree:** NEW, per docs/128 §8.7 — `cd /home/intel/ninfer/repo && git
  worktree add /home/intel/ninfer/worktrees/wo-phase-gate -b wo/phase-gate
  wo/kv-uniform`. Own `build/` dir — NEVER build into `wo-kv-uniform`'s
  build (agent 1 is actively editing `src/` there) and never into ci-gate.
- **Test entry:** `build/tests/ninfer_phase_gate` (new binary) +
  `tools/bench/phase_gate.sh` (new script, chain table output).
- **Reuse, don't rebuild** (docs/128 §6): D1 oracle = `tests/test_kvarn_codec.cpp`,
  `tests/test_kvarn_codec_edge.cpp`, `tests/kvarn_acc_fwht_cpu_ref.cpp`,
  `tests/kvarn_sinkhorn_convergence.cpp`. D2 oracle =
  `tests/kvarn_materialize_oracle_test.cu` (pattern for the test binary) + the
  CPU dequant path.
- **GPU:** M0–M2 need only the test binary (direct TextContext — no server,
  no port), ~5–10 s per run. Claim slots via intercom before launching;
  `nvidia-smi` + `pgrep` first.
- **Test gates (point-of-failure workflow):** if a chain phase misbehaves, the
  chain table output IS the localizer (first FAIL = point of failure,
  cascade below). If the chain itself misbehaves, check the negative test
  (mutation) first — a gate that can't catch a known bug doesn't ship.

## 3. Scope (and NOT)

IN: env-gated dump hooks for D1 (commit/quantize: codes + scales + layout)
and D2 (dequant/materialize: bf16 K/V tiles); `ninfer_phase_gate` test
binary (pattern: `kvarn_materialize_oracle_test.cu`); `phase_gate.sh` with the
chain table + per-phase artifact files (docs/128 §8.3 format); pinned prompt
(commit it, hash into the artifact); commit per step.
NOT: D3–D7 (that's M1, docs/134), CI wiring (M2, docs/132 item 3c), any
server-side work, any perf work.

## 4. Design (per docs/128 §8 — do not deviate)

- Trigger interface = **test binary, NOT the server** (§8.1).
- Pinned prompt per §8.2 (commit, immutable, hash into artifact).
- Artifact format per §8.3 (one file per phase boundary).
- Tolerance per §8.4: D1/D2 are byte-exact vs CPU oracles — no tolerance
  needed at M0; note where M1 will need FP64 (risk a).
- Gate output + exit code per §8.5 (table + first failing index, 0 = clean).
- First-day checklist per §8.6.

## 5. Design decisions (FINAL — do not re-litigate)

1. Worktree = `wo/phase-gate` based on `wo/kv-uniform` HEAD at start
   (docs/128 §8.7). REJECTED: working in `wo-kv-uniform` (agent 1 is
   editing `src/` there) — revisit never.
2. Dump hooks are **env-gated, zero-cost when unset** (`NINFER_MB_*`
   pattern: null-checked pointers, no device-side `printf` on timing
   paths, no async H2D from stack/heap temporaries). REJECTED: always-on
   hooks — revisit never (docs/128 risk b).
3. D1/D2 oracles = the existing in-tree codec/materialize tests
   (reuse). REJECTED: new oracle implementations — revisit only if an
   existing oracle proves wrong.
4. Chain runs on a fixed small ctx so the whole chain is <10 s (docs/128 §6
   M0 acceptance). REJECTED: long-ctx M0 — revisit never (that's M3's
   domain).

## 6. Execution order (commit + test each step before the next)

1. Create worktree + fresh cmake build; verify baseline compiles (full
   compile — plan the time).
2. D1 dump hook + artifact emission (env-gated). Test: hook on → artifact
   written + valid; hook off → zero behavior change (diff a run with/without).
3. D2 dump hook + artifact. Same test.
4. `ninfer_phase_gate` test binary skeleton: run D1→D2, compare to CPU
   oracles, print chain table.
5. `phase_gate.sh` + artifact format + pinned prompt (commit the prompt,
   hash into artifact).
6. **Acceptance run 1 (GPU slot):** known-good build → D1/D2 PASS, chain
   <10 s.
7. **Acceptance run 2 (GPU slot):** 1-line mutation (dequant scale
   ×(1+1e-3)) → must FAIL at D2 with the table pointing at D2. Then revert
   the mutation and re-run clean.
8. Commit everything as one reviewable unit on `wo/phase-gate`; write the
   debrief (§9).

## 7. Pitfalls

- Hook overhead must be exactly zero when env is off (docs/128 risk b) —
  verify, don't assume.
- Pinned prompts must be immutable (risk c): committed + hashed, ever.
- `fopen` NULL-guarded with a stderr shout. No device-side `printf` on
  timing paths.
- Never build into another agent's `build/` dir.
- If a CPU oracle disagrees with the GPU artifact, the oracle is suspect
  until proven otherwise — report to coordinator, don't "fix" the kernel.

## 8. Definition of done

1. Known-good build: D1/D2 PASS, chain <10 s (evidence: run output in
   results/ + debrief).
2. Negative test: 1-line mutation FAILS at D2 (evidence: run output),
   mutation reverted + clean re-run.
3. All work committed on `wo/phase-gate`; worktree leaves clean (no
   uncommitted junk).
4. Debrief written (§9): findings, dead ends, what M1 needs from M0
   (artifact formats, hook env vars, any caveats).

## 9. Debrief (owner fills before stopping)

- (to be written)
