# 105 — GDN gate T=1 perf fix (docs/103): Agent Work Order

**Status:** CURRENT — work order for a single implementer agent.
**Mission:** Restore the −7.5% decode perf regression introduced by `1d3a0533`
(GDN gate routing unified to MmaUnsplit for D-21 correctness) **without breaking
the D-21 byte-identity contract**. End state: CI plain decode back to ≥35.0 t/s,
verify ≤35.6 ms, MTP ≥79.0 t/s, D-21 battery still byte-identical, docs/50 D-21
line + docs/103 closed with evidence. Read this document fully before writing
code.

---

## 1. Context (60-second version)

CI drifted 08-26 → 08-28: plain **35.40 → 32.80 t/s (−7.5%)**, verify
**35.24 → 37.44 ms (+6.2%)**, MTP **80.71 → 75.38 t/s**. The only commit in the
drift window touching the plain decode path is `1d3a0533` ("route 27-model GDN
gate to token-count-independent MmaUnsplit") — a 6-line routing-table change
(`bf16_gdn_gating_proj_plan.cpp`). It was REQUIRED for D-21 correctness: routing
by token count meant T=1 (GEMV) and T>1 (MMA) accumulated the control gate
~1 ulp differently → linear-state divergence → argmax flips (the 1024-boundary
bug). **Do not re-introduce the old split.** `84ba2c8b` (anchor fix) was ruled
out of the plain path: its tp2_backend.cpp additions are `NINFER_D21_DBG`-gated
debug (off by default) + MTP-branch-only fix. Deeper background: docs/103
(attribution + gates), results/d21_mtp_verify_anchor_fix.md (battery + closure
criteria).

**Already built and verified (do not redo):**
- D-22 full-vocab accept fix (`6ad60d42`), D-23 S3 fix (`9a33b896`), D-21 anchor
  fix (`84ba2c8b`) — all QA-verified, CI 20260828_224030 = 0 fails.
- The regression itself, its bracket, and the mechanism (docs/103 §1-3).
- Baseline infra: `tools/bench/decode_guard_check.py` + `tools/bench/decode_guard_baseline.json`
  (official v1 lands ~2026-08-29 11:00 from the `official_sampling`/`official_greedy`
  runs — check `git log` / the baseline file's provenance before using numbers).

**What you are doing:** steps 0-4 of §6 below.

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; remote `github`).
- **Work protocol (MANDATORY):** work in the EXISTING worktree
  `~/ninfer/worktrees/wo-kvarn-hold` (branch `wo/kvarn-hold`). Never edit the
  main tree. Commit per step to `wo/kvarn-hold` and push. **Do not merge** —
  the plan owner merges (docs/77 §8 hold).
- Build (inside your worktree): `cmake --build build -j 16` (CUDA `sm_120a`;
  2× RTX 5060 Ti 16 GB).
- Unit tests: `/usr/bin/ctest` (the `ctest` on PATH is a broken Python
  wrapper). From `build/`: `/usr/bin/ctest -R "gdn|kvarn|spec"`.
- **Test entry points:**
  - `bash tools/ops/run_ci.sh` from your worktree root (relative path!) = fast
    per-build gate. `bash tools/ops/run_ci.sh --full` = entire suite — for
    closeout only (step 3). It stops any running server, builds, tests your
    build, restores the live server at exit. You have full server agency;
    `pkill -x ninfer-serve` (NEVER `pkill -f`); `pgrep -x ninfer-serve` +
    `nvidia-smi` before every launch.
  - **GPU QUEUE (critical):** the official baseline run occupies the GPU until
    ~11:00 (log `~/ninfer/logs/official_baseline_20260829_075416.log`). Do not
    launch anything on the GPU before it finishes; do not kill it. Steps 1-2's
    GPU work starts only after ~11:00.
  - D-21 battery (the byte-identity gate) — commands in
    results/d21_mtp_verify_anchor_fix.md "Pending server items":
    (a) temp>0 re-run: `--spec mtp --draft-tokens 3`, temp 0.8, robot/sunsets
    prompt family, seeds 1–6, assert no CJK + coherent (exact repro in
    results/kvarn_temp_sampling_root_cause.md "How to re-verify");
    (b) H5: `tools/ops/h5_retest_mtp_long.sh` (MTP=3, 1024 prompt,
    MTP-vs-plain byte-identical) at k=1, k=2, k=3;
    (c) 6-prompt battery at k=2 (expect byte-identical; k=1/k=3 allow the
    documented near-tie tolerance — 1 of 6 prompts, ~1 ulp attention).
  - Measurement: decode rate from serve.log `decode=Xtok/s`; CI JSON
    (`results/*_ci.json`) for plain/mtp t/s + verify_ms. `nvidia-smi` clocks
    with every perf number. Commit all results under `results/`.
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.

## 3. Architecture facts (verified — do not re-derive)

- The 27-model GDN gate = `bf16_gdn_norm_gating_dispatch` → rmsnorm +
  `bf16_gdn_gating_dispatch`, problem {heads=48, input_rows=5120, cols=T}.
  Called **once per GDN layer** — 48 times per decode step (plain T=1) and per
  verify (T=4). Weights: a/b [48, 5120] BF16 (983 KiB total).
- `k27Routes` (plan.cpp, post-`1d3a0533`) = single route {1,∞} → MmaUnsplit.
  Pre-regression it was {1,1}→GemvPairedRows, {2,8}→SmallTSplit10, {9,∞}→
  MmaUnsplit. `candidate_is_legal` still encodes the old per-T legality —
  **leave it as-is** (it is the D-21 hard constraint; the route table is what
  actually dispatches).
- MmaUnsplit launch (kernels.cu:413) = `launch_bf16_prefill_mma<Bf16Gdn27Geometry, 1, 8>`
  (SplitK=1, Warps=8). Kernel: grid = (div_up(T,128), 48/16=3, 1) — **at T=1
  that is 3 CTAs on the whole GPU**; 80 K-tiles (BlockK=64, hidden 5120);
  `kBf16GdnStages = 2`-stage cp.async pipeline; 2× `__syncthreads` per
  iteration. T=1 is latency-bound: 80 serial iterations, each gated by the
  ~4 KiB weight-tile HBM round trip divided over 2 stages ≈ 200-300 ns →
  ≈40-60 µs/call × 48 layers ≈ 2.4 ms/step = the entire plain regression
  (T=4 verify delta, +2.2 ms, is the same mechanism: SmallTSplit10's 10-way
  split-K parallelism was replaced by the same 3-CTA latency wall).
- **Bit-identity invariant (the key fact for the fix):** each thread's
  accumulator fragment (mi, ni, lane) is accumulated by the mma sequence
  `for it in 0..79: for ki in 0..3: mma(acc, A[it][ki], X[it][ni][ki])` — a
  fixed order that is **independent of pipeline depth (`Stages`) and warp
  count (`Warps`)**. Those parameters change only when data arrives, never
  the addition order. Therefore a deeper pipeline / more warps is
  **bit-identical by construction** (verify with the battery anyway).
- `kBf16GdnStages` is a global constant (gemm_mma.cuh ~line 29) shared by the
  27-geometry AND 35-geometry paths and by the SplitK>1 cooperative launches.
  The 35 model and the 35 norm-gating split32 path must stay untouched.
- smem: 27-geometry stage = (128×64 + 2×16×64) bf16 = 20480 B; 4 stages =
  80 KiB/CTA (within sm_120's per-SM smem; with 3 CTAs device-wide there is no
  occupancy interaction). 35-geometry stage = 12288 B; 4 stages = 48 KiB.
- Acceptance 80.4% (greedy MTP) is the **corrected** post-D-21 number (was
  82.0 with the bug). Do not "fix" acceptance back up.

## 4. Key call sites (anchors — verify line numbers before editing)

- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh` — kernel +
  constants: `kBf16GdnBlockM/BlockK/Warps/Stages/MFragments` (~26-30),
  `kBf16GdnSmemBytes` (~34-37), `Bf16Gdn27Geometry` (~39-43), main loop
  (`cp_wait<kStages-1>` / `stage_load(..., it + kBf16GdnStages)` ~186-225).
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` —
  `launch_bf16_prefill_mma` template (~262-330, grid/smem/launch, note the
  `cudaFuncSetAttribute` smem sizing), `bf16_gdn_gating_proj_mma_unsplit_launch`
  (~413), gemv (~333) and small_t_split10 (~355) launches (do not touch).
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp` — `k27Routes`
  (~29), `candidate_is_legal` (~175), `bf16_gdn_gating_proj_mma_unsplit`
  dispatch in `execute_resolved` (~150). Route table: **no change expected**.
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h` — schedule enum.
- `results/d21_mtp_verify_anchor_fix.md` — battery commands + closure
  criteria + accepted-tolerance contract.

## 5. Design decisions (FINAL — do not re-litigate)

1. **Fix = deepen the 27-geometry MmaUnsplit pipeline, nothing else.**
   Introduce a per-launch `Stages` template parameter (default
   `kBf16GdnStages`) and pass `Stages=4` from the 27 mma_unsplit launch only.
   Bit-identity by construction (§3 invariant); speeds T=1 (plain) AND T=4
   (verify) AND prefill (same launch); leaves the 35 model, the cooperative
   split launches, and the 35 norm-gating split32 path byte-for-byte as they
   are (default Stages=2).
2. **ESCALATION (only if step 1 misses perf gate):** `Warps 8→16` in the same
   27 mma_unsplit launch (kNFragments 2→1; per-fragment mma order unchanged →
   still bit-identical by construction).
3. REJECTED: restoring GemvPairedRows/SmallTSplit10 routes — that is the D-21
   bug. REJECTED: split-K at T=1 — changes accumulation order (partials
   reduced in a 2nd kernel). REJECTED: new SIMT GEMV kernel "re-verified under
   tolerance" — re-opens the D-21 risk class the hard constraint exists for.
4. **If both 1+2 miss the perf gate:** STOP and report with per-layer
   instrumentation (nsys or env-gated printf timing around
   `bf16_gdn_norm_gating_dispatch`); then re-attribute with a diagnostic build
   of `1d3a0533~1`. Do not ship a partial fix.

## 6. Execution order (commit + test each step before the next)

**Testing standard:** byte-identity is a LIVE-server property (MTP rounds,
chunked prefill, request lifecycle). Unit tests + the battery are the gate;
kernel-only parity does not count.

### Step 0 — Pre-fix capture (CPU + GPU, ~11:00)
On the current build (no code changes), capture the A/B reference: run the
D-21 battery (temp>0 seeds 1–6, H5 k=1/k=2/k=3, 6-prompt battery k=2) and save
every output to `results/105_prefix_battery/` (commit). Also record the
current CI JSON's plain/mtp/verify numbers (20260828_224030 is the reference;
re-run `run_ci.sh` fast gate once to confirm on this tree).
**Tests:** battery outputs saved + coherent; CI numbers recorded.

### Step 1 — Stages 2→4 (27 mma_unsplit)
Add the `Stages` template parameter (default `kBf16GdnStages`) through
`launch_bf16_prefill_mma` and the kernel (cp_wait/stage_load arithmetic uses
it; smem sizing uses it; `cudaFuncSetAttribute` uses it), pass `Stages=4` from
`bf16_gdn_gating_proj_mma_unsplit_launch` when `is_27`. Everything else
unchanged. Build.
**Tests (must pass before moving on):**
- `/usr/bin/ctest -R "gdn|kvarn|spec"` green.
- **Byte-identity A/B:** re-run the step-0 battery on the new build; every
  output byte-identical to `results/105_prefix_battery/` (near-tie tolerance
  only per the documented contract).
- **Perf gate (fast CI):** plain ≥35.0 t/s, verify ≤35.6 ms, mtp ≥79.0.
  Commit `fix(gdn): deepen 27-model MmaUnsplit pipeline to 4 stages (docs/103)`
  with the numbers.

### Step 2 — (conditional) Warps 8→16
Only if step 1 missed the perf gate. Same verification. Commit.

### Step 3 — Full CI + D-21 closeout
`bash tools/ops/run_ci.sh --full` from this tree (this is also D-21 pending
server items 1-3: temp>0 re-run, full-width H5 matrix k=1/k=2, full CI).
Regenerate any pre-`1d3a0533` goldens touched (D-21 item 4) — record what was
regenerated. Commit results with model identity + build commit.
**Tests:** full CI 0 fails; D-21 battery as step 0; goldens recorded.

### Step 4 — Closeout docs
Update results/d21_mtp_verify_anchor_fix.md (all pending items MET),
docs/50 D-21 line (evidence), docs/103 (fix landed: commit, before/after
numbers, battery green). Commit `docs(50,103): close D-21 pending items +
docs/103 (GDN T=1 perf fix)`.

## 7. Constraints (non-negotiable)

- Worktree only (wo-kvarn-hold); no edits in the main tree; no merges.
- GPU queue: nothing on the GPU until the official baseline run finishes
  (~11:00) — see §2. One live server at a time; leave it restored at the end
  (or say so in the report).
- **Do not touch:** the KVarN attention path, the MTP accept path (D-22),
  `candidate_is_legal`'s legality semantics, the 35-model paths, sampler code.
- **Do not "fix" acceptance** (80.4% is correct), do not modify T19 threshold
  (≥450 tok/s), do not modify gate numbers in this doc.
- Keep the tree buildable at every commit; commit per step with the step
  named; results data is project data — commit it.

## 8. Definition of done

1. Steps 0-4 committed to `wo/kvarn-hold` with passing tests at each step.
2. **Live proof:** D-21 battery byte-identical on the final build (outputs
   committed), full CI 0 fails.
3. **Perf restored:** CI plain ≥35.0 t/s, verify ≤35.6 ms, mtp ≥79.0 — the
   pre-`1d3a0533` regime, recorded in the report table.
4. `results/105_*` committed (pre-fix battery, per-step CI JSONs, final
   numbers) with build commit + model identity.
5. Report: one paragraph per step + before/after table (plain/verify/mtp
   t/s + verify_ms, acceptance, battery verdicts) + the exact diff summary
   (files/lines changed).
