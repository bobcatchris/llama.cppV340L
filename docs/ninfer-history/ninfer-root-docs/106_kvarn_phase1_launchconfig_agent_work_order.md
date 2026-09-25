# 106 — KV uniformity Phase 1 (launch-config unification) + Phase 0/3/4 audits: Agent Work Order

**Status:** CURRENT — work order for a single implementer agent (second agent;
see GPU queue rule in §2).
**Mission:** Land docs/104 **Phase 1** — make every kernel's launch
configuration explicit per-SKU dispatch instead of hard-coded constants tuned
to specific SM counts — with **zero behavior change on the current SKU**
(byte-identical outputs, no t/s regression vs the standing baseline), complete
**Phase 0** (per-variant identity gates in the baseline matrix), and produce
the CPU audits for **Phase 3** (prefill O(n²) materialize) and **Phase 4**
(prefix-reuse). Read this document fully before writing code.

---

## 1. Context (60-second version)

docs/104 ("KV cache uniform performance plan") root cause R2: per-variant
kernels were tuned to *different* SM counts and hard-code launch constants
(CTA counts, resident-CTA grids, block sizes) that match one SKU and not
another — e.g. GDN cooperative grids assume "340/680 CTAs across 170 SMs"
while the KVarN packed decode split count is tuned to 36 SMs, on a machine
that is 2× RTX 5060 Ti. Phase 1 makes the configuration **explicit**: one
per-SKU dispatch table, built from the runtime `multiProcessorCount`, so
adding a SKU (5070/5080) is a table row, not a code audit — and so the
current SKU's behavior is *provably unchanged*. Deeper background: docs/104
(phases + gates), docs/102 §9/§13 (KV variants + hardware audit).

**Already built and verified (do not redo):**
- Baseline infra: `tools/bench/decode_guard_check.py` +
  `tools/bench/decode_guard_baseline.json` (monotonic ratchet; config-aware;
  overlapped spec kvarn 10k/25k/40k/80k/160k/250k, int8 +80k/160k, bf16 +80k;
  canonical sampling = Qwen thinking temp 1.0/top_p 0.95/top_k 20). Official
  v1 baseline lands ~2026-08-29 11:00 from the `official_*` guard runs.
- docs/103 (GDN T=1 perf fix) is being executed by the FIRST agent on
  `wo/kvarn-hold` — **not your scope; do not touch the GDN gating plan/kernels.**
- D-21/D-22/D-23 closed (see results/d21_mtp_verify_anchor_fix.md, docs/50).

**What you are doing:** steps 0-4 of §6 below.

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; remote `github`).
- **Work protocol (MANDATORY):** ALL work in a worktree + branch, never the
  main tree. **Create your worktree AFTER the first agent's GDN fix lands on
  `wo/kvarn-hold`** (branch from its head so your baseline includes the perf
  fix): `git worktree add ~/ninfer/worktrees/wo-kv-uniform -b
  wo/kv-uniform wo/kvarn-hold`. If `wo/kvarn-hold` has not yet landed the GDN
  fix when you start, do steps 0-2 (CPU) from a worktree off its current head
  and re-branch before any GPU step. Commit per step, push. **No merges** —
  plan owner merges (docs/77 §8).
- Build: `cmake --build build -j 16` (CUDA `sm_120a`; 2× RTX 5060 Ti 16 GB).
- Unit tests: **`/usr/bin/ctest`** (PATH `ctest` is a broken Python wrapper).
  From `build/`: `/usr/bin/ctest -R "<pattern>"`.
- **GPU QUEUE (critical, two locks):**
  1. The official baseline run occupies the GPU until ~11:00 (log
     `~/ninfer/logs/official_baseline_20260829_075416.log`). No GPU before
     that; do not kill it.
  2. The first agent (docs/105) then runs the GDN fix battery + full CI
     (~11:00 → ~16:00). **No GPU for you until that full CI is committed**
     (check `git log wo/kvarn-hold` for the step-4 docs commit, or ask the
     plan owner). One live server at a time; `pkill -x ninfer-serve` (NEVER
     `pkill -f`); `pgrep -x ninfer-serve` + `nvidia-smi` before every launch;
     leave the server restored at the end.
- Server tests: `bash tools/ops/run_ci.sh` from YOUR worktree root (relative
  path — the main-tree symlink would test the wrong code). `--full` only at
  closeout. Must-pass lists: docs/50 §7.1.
- Measurement: decode rate from serve.log `decode=Xtok/s`; perf gates via
  the baseline guard (§6 step 4); `nvidia-smi` clocks with every number.
  Commit all results under `results/`.
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.

## 3. Architecture facts (verified — do not re-derive)

- Hardware: 2× RTX 5060 Ti 16 GB, `sm_120a`, one GPU per rank (TP=2). **Query
  the real SKU at runtime** (`cudaDeviceGetAttribute multiProcessorCount`) —
  do not trust any doc's SM count; docs/102 §13 recorded the audit but the
  constants below predate it and disagree with each other.
- Known hard-coded launch constants (anchors — the FULL list is step 0's
  deliverable; these are the ones already identified):
  - GDN gating plan (`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp`):
    `cooperative_27_grid_is_resident` → 340 resident CTAs;
    `cooperative_35_grid_is_resident` → 340/680 ("across 170 SMs").
  - KVarN packed decode: `kKvarnDecodeSplits` (=54) in
    `src/ops/kernel/gqa_attention_kvarn_decode_packed.inc` (tuned to 36 SMs)
    vs `kMax`-class split counts (=42) tuned to the other SKU.
  - RoPE/attention blocks (e.g. 1020-block grid) and MoE (510-block grid) —
    locate in `src/targets/qwen3_6_27b/impl/variant_kernels.cpp` and the
    attention/rope wrappers.
- The three KV variants share the same model/geometry; they differ ONLY in KV
  storage + dequant (docs/102 §9). Launch-config differences between variants
  are a bug class, not a feature (docs/104 R1/R2).
- Byte-identity contract: greedy decode is deterministic per (prompt, seed,
  build); a launch-config change that alters split/CTA counts **can** change
  reduction order → different bits. On the current SKU you must KEEP the
  current best config exactly (that is why the gates include byte-identity).

## 4. Key call sites (anchors — verify before editing)

- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp` — resident-CTA
  residency checks (see §3).
- `src/ops/kernel/gqa_attention_kvarn_decode_packed.inc` / `.cu` —
  `kKvarnDecodeSplits` + launch config.
- `src/ops/wrapper/gqa_attention.cpp` — variant dispatch (where the per-SKU
  table would be consulted).
- `src/targets/qwen3_6_27b/impl/variant_kernels.cpp` — per-variant kernel
  selection (rope/attention/MoE grid constants).
- `src/core/device.cu` — device property queries (where to cache
  `multiProcessorCount`).
- `tools/bench/decode_guard_check.py` + `tools/bench/decode_guard_baseline.json`
  — the perf gate you must not regress.

## 5. Design decisions (FINAL — do not re-litigate)

1. **One per-SKU dispatch table, built at startup from
   `multiProcessorCount`** (cached once, in device init), consulted by every
   site that currently hard-codes an SM-dependent constant. Table entry =
   the constant(s) that site needs. Unknown SKU = throw (fail loud, never
   guess) — or use an explicitly-marked fallback only where the constant is
   provably SM-independent.
2. **Current SKU = no behavior change.** Every table value for the current
   SKU must equal the current hard-coded value that is actually in effect
   (measure first if a constant is conditionally used — e.g. the 340 vs 680
   residency pair). Gate: byte-identical greedy A/B + baseline guard.
3. REJECTED: auto-tuning at startup (costs seconds per launch site,
   non-deterministic across clocks, unreviewable). REJECTED: compile-time
   per-SKU specialization (defeats the purpose; the point is one binary).
4. The GDN `cooperative_*_grid_is_resident` checks are *correctness* guards
   (cooperative launch must be resident) — keep them as hard checks on top of
   the table, not replace them.

## 6. Execution order (commit + test each step before the next)

**Testing standard:** a launch-config change touches the server-facing path —
byte-identity A/B + the baseline guard are the gate; reading code and
`ctest` are necessary, never sufficient for the GPU steps.

### Step 0 — Audit (CPU only, no GPU)
Produce `results/106_launch_config_audit.md`: the COMPLETE list of
SM/occupancy-dependent launch constants (grep for the anchors in §3 +
`div_up`, `grid(`, `resident`, `Splits`, block sizes in
`src/ops/kernel/`, `src/ops/wrapper/`, `src/targets/qwen3_6*/impl/`), each
with: file:line, the value, what SM count it assumes, and whether it is
currently in effect on this SKU. Flag every constant that does NOT match the
runtime SM count (those are latent bugs on this SKU — report them, do not
fix them in this step).
**Tests:** doc committed; plan owner (QA) reviews it before step 1.

### Step 1 — Dispatch table (CPU + build)
Implement the table (§5.1) + replace the hard-coded constants with lookups.
Build + `/usr/bin/ctest` green. **No GPU yet.** Commit
`refactor(launch): per-SKU dispatch table for launch configs (docs/104 P1)`.
**Tests:** ctest green; diff shows current-SKU values unchanged (table row ==
old constants); a unit test asserts the table returns exactly the old values
for the current `multiProcessorCount`.

### Step 2 — CPU audits (no GPU)
`results/106_prefill_materialize_audit.md` (Phase 3): locate the O(n²) KV
materialize pass in the KVarN prefill path (docs/104 R4), measure its share
of prefill at 80k/160k from the official baseline runs' prefill t/s
(kvarn ~13% slower at 10k vs bf16/int8), and propose the fix design (direct
read / single pass) with the gate it must pass.
`results/106_prefix_reuse_audit.md` (Phase 4): locate the prefix-restore
mechanics per variant (docs/104 R5), document where variants differ, propose
the unification + gate.
**Tests:** both docs committed; plan owner reviews.

### Step 3 — GPU validation (only after the GPU queue is free, §2)
(a) Byte-identity A/B: greedy decode, ≥3 prompts × {bf16, int8, kvarn} ×
{10k, 40k}, current build vs your build → byte-identical.
(b) Baseline guard: run the full overlapped spec (both passes) and
`decode_guard_check.py` → **no FAIL, no WARN beyond ±2% noise** vs baseline v1.
(c) `bash tools/ops/run_ci.sh` fast gate: 0 fails.
Commit results with build commit + model identity.

### Step 4 — Closeout docs
Update docs/104 (Phase 1 = DONE with the table + gates; Phase 0 completion
status; audit cross-refs). Commit
`docs(104): Phase 1 landed + Phase 3/4 audits`.

## 7. Constraints (non-negotiable)

- Worktree only; no merges; commit per step; keep the tree buildable.
- **GPU queue rule (§2) is absolute** — a premature GPU launch corrupts
  someone else's baseline run and you will be stopped.
- Do not touch: GDN gating plan/kernels (first agent's scope), MTP accept
  path (D-22), KVarN numerics (the packed kernel is QA-verified — you are
  changing dispatch, not math), sampler, T19 threshold (≥450 tok/s).
- If the step-0 audit finds a constant that is WRONG on the current SKU
  (latent bug): report it in the audit + a separate `results/106_*` finding;
  fixing it is a new work order (it would change behavior and break the
  byte-identity gate by design).
- Baseline guard is the perf gate — a WARN on any cell must be explained in
  the report before commit.

## 8. Definition of done

1. Steps 0-4 committed to `wo/kv-uniform` with passing tests at each step.
2. **Live proof:** byte-identical A/B (step 3a outputs committed) + baseline
   guard no-FAIL on the full overlapped spec + fast CI 0 fails.
3. `results/106_launch_config_audit.md` lists every SM-dependent constant;
   the table row for the current SKU is provably the old behavior.
4. Phase 3/4 audits committed with fix designs + gates (not implementation —
   that is the next work order).
5. Report: one paragraph per step + the constant table (file:line, value,
   assumed SMs, actual SMs, in-effect?) + A/B/guard verdicts.
