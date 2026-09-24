# 138 — i8/bf16 phase gate: Agent Work Order (M0–M4, both lanes)

**Status:** CURRENT — work order for a single implementer agent.
**Implementer:** gemini (both lanes: i8 and bf16+MTP — no split; shared files
are one writer).
**Mission:** Build the docs/137 scope into a working, CI-wired phase gate for
the I8 and BF16 KV-dtype routes: chained per-phase oracles on
**production-route** artifacts for decode (I1–I7 / B1–B7), the MTP round gate
(both dtypes), the prefill chains (I-P1..P4 / B-P1..P4), and `run_ci.sh`
wiring (fast chain + full mutation battery + bf16 correctness battery + i8/bf16
ratchet cells). Done = a broken I5 or B5 build fails CI with a table pointing
at the phase, and a broken quantize/dequant/accept shows up at I1/I2/accept —
all in ~2–4 min of fast CI.
Read this document fully before writing code. **Read docs/137 (the scope)
first** — this order is the execution plan for it.

---

## 1. Context (60-second version)

docs/128 gave KVarN (K4V2) a phase gate: chained oracles on phase-boundary
artifacts, first failing phase = broken phase, lockstep rule, CI wiring.
The other two production KV dtypes — I8 (per-token group-64 int8, 4 planes)
and BF16 (2 planes) — have no phase gate: kernel regressions on those routes
localize via ~1 day of manual A/B (the docs/104 long-ctx I8 fix was exactly
that). The suspected "same bug" (docs/132 §2.1: a gate that runs a
non-production route) **is confirmed for i8** and is fixed in Step 1
(docs/137 §7.1).

**Already built and verified (do not rebuild — reuse):**
- wo-phase-gate `tests/phase_gate.cu`: `PinnedInput::make(seed=1, ntiles_k,
  ntiles_v)`, `write_phase()`, `PHS1` header, `--mutate-*` pattern. **Do not
  change `PinnedInput` output — sha ratchets break.**
- wo-phase-gate `tests/kvarn_decode_cpu_ref.h`: FP64 online-softmax + split-K
  merge emulation — the canonical bf16 body is shared by i8 (post-dequant),
  bf16, and KVarN, so I3–I6/B3–B6 refs are near-duplicates of it.
- `build_slice4_dump.sh` + `tests/phase_gate_slice4_compare.cpp`: the
  production in-kernel null-checked dump hook + comparator + tolerance-JSON
  pattern (docs/133 resolution).
- `tools/bench/phase_gate_mb.{sh,py}`: MTP round-dump driver + converged-state
  comparator (B1/B4/B5 vs R1/R4/R5, exit 0/20/30, rank-gated dumps). The
  **comparator is dtype-agnostic — reuse verbatim**.
- `tests/ops/test_gqa_attention.cpp`: op-level criteria (bf16 rel_l2 2.8e-3 /
  i8 3.15e-3, A1/A3 profiles) — keep; it stays as an independent check, not a
  replacement.

**What you are doing:** Steps 1–7 below (docs/137 §8 milestones M0, M1, M2,
M4, M3, + M5 the MTP-verify-geometry coverage gap found 2026-09-03), each committed and tested before the next.

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo`. Worktrees under
  `/home/intel/ninfer/worktrees/`.
- **Work protocol (MANDATORY):** worktree + branch only — never edit the main
  tree. Create from the current integration branch HEAD:
  ```bash
  cd /home/intel/ninfer/repo
  git worktree add ~/ninfer/worktrees/wo-phase-gate-i8 -b wo/phase-gate-i8
  cd ~/ninfer/worktrees/wo-phase-gate-i8
  cmake -S . -B build && cmake --build build -j 16   # own build dir
  ```
  Commit per step to `wo/phase-gate-i8` and push. Merging to main is done by
  the main-side agent/user — do not merge or push to main yourself.
- Build: `cmake --build build -j 16` (CUDA `sm_120a`, 2× RTX 5060 Ti 16 GB).
- Unit tests: **`/usr/bin/ctest`** (the `ctest` on PATH is a broken wrapper).
- **Never build into another agent's `build/` directory** (the docs/132 §2.2
  trap). One build dir per worktree.
- Test entry points:
  - Your gate binary (you build it): `build/tests/ninfer_phase_gate_i8b`
    (Step 1). CMake target links `ninfer_ops` + `ninfer_artifact` — the
    docs/132 §2.3 link trap; verify `nm` shows the symbols.
  - Server tests: `bash tools/ops/run_ci.sh` **from this worktree's root**
    (relative path — a symlink to the main tree would silently test main's
    code). `--full` for the whole suite. It stops any running server,
    tests your build, restores the live server at exit.
  - Targeted: `tools/smoke/serve_correctness_ci.sh` — env-overridable
    `KV_DTYPE=... SPEC=...` (default int8, no SPEC), `--only T<n>,...`.
    Exit code = number of failed tests. Manages its own server lifecycle.
  - MTP point-of-failure workflow if MTP diverges (docs/131): INVASERT →
    cause map / HASHPT ×16 → T0 batteries → T2 golden. Do not manually
    bisect MTP.
  - Must-pass lists: docs/50 §7.1 is the source of truth.
  - One server at a time on the two GPUs; `pgrep -x ninfer-serve` +
    `nvidia-smi` before every launch; `pkill -x ninfer-serve` (NEVER `pkill
    -f`).
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.
- Measurement: clocks recorded with every perf number (~8%/hr drift);
  results committed under `results/`.

## 3. Architecture facts (verified — do not re-derive)

- **DTypes** (`src/core/dtype.h`): `BF16=0`, `FP32=1`, `I32=2`, `U8=3`,
  `I64=4`, `I8=5`, `FP16=6`, `FP8_E4M3FN=7`, `KVARN_K4V2=8`.
- **I8 storage**: 4 planes — K codes, V codes, K scales (fp16), V scales
  (fp16). Per-token group-wise, group=64, head dim 256. Quantize: scale =
  fp16(absmax/127), code = clamp(RNE(x·inv_scale), −127, 127)
  (`gqa_kv_quant_code`, `src/ops/kernel/gqa_attention_kv_quant.cuh`).
  **There is no standalone quantize/dequant kernel** — quantize is fused into
  the append kernels, dequant into the decode kernel prologues.
- **I8 decode production (default)**: `NINFER_I8_DECODE` unset →
  `I8Route::kUnified` → `launch_tc_partial_unified_i8` →
  `gqa_decode_slice3_i8_v2_kernel` (D-split partition, canonical bf16 body +
  vectorized I8 dequant prologue) + shared
  `gqa_attention_small_t_reduce_output_kernel<Int8=true>`.
  A/B legacy: `NINFER_I8_DECODE=shipped` → `launch_tc_partial_i8` →
  `gqa_attention_decode_i8_tiled_kernel`. Banner `[i8-decode-route]` is
  printed by the launcher — **gate the default route only**; the A/B env
  stays as rollback, covered only if explicitly selected.
- **I8-specific split policy**: `gqa_small_t_split_count` (same launcher file):
  tokens==5 window 128–512 → 32-key tiles; tokens==6 window 128–160 → 24;
  tokens==6 window 5000–8198 → 192 (clamped by 42·DecodeSplitScale). Part of
  the gated surface (I7).
- **BF16 decode production**: `launch_tc_partial_bf16` →
  `gqa_attention_small_t_tc_partial_bf16_kernel`
  (`src/ops/kernel/gqa_attention_decode_bf16.cuh`) + shared reduce
  `<Int8=false>`. Commit = direct bf16 paged copy (fused via
  `GqaAppendInput`). No quantize.
- **Prefill**: `src/ops/launcher/gqa_attention_prefill.cu` — I8
  (`gqa_attention_prefill_i8_kernel`, fill via
  `gqa_attention_prefill_fill_i8_page_kernel` which has **two paths**: tiled
  `tokens>=128 && KVHeads==2` and scalar — both must be gated), BF16
  (`gqa_attention_prefill_bf16_kernel`). Distinct smem ceilings per dtype.
  BF16 prefill is 170-SM-tuned; `--kv-dtype bf16 --max-context 40360` crashes
  with `invalid execution envelope or table` (docs/127 §2) — B-P3 must
  define/pin the valid envelope for 36-SM cards.
- **MTP**: single-sequence verify path is dtype-agnostic
  (`src/ops/kernel/speculative_round.cu`: `prepare_verify_inputs`,
  `accept_greedy_drafts`, `select_accepted_hidden`). MTP-layer KV =
  `mtp_cache()`, a plain paged cache with the **same `kv_dtype`** as the text
  cache (`decoder_state.cpp:94-98`). The **multi-lane batched runner is
  KVarN-only** (`is_kvarn_kv` gates the lane workspace,
  `tp2_backend.cpp` :368/:501; `set_kvarn_batched_active` :2450) — it is
  OUT of scope here (feature, not gate; docs/137 §6).
- **Existing i8/bf16 tests** (today's coverage; docs/137 §7 is the audit):
  - `tests/slice3_i8_test.cu` (`ninfer_slice3_i8_test`): FP64-oracle full
    battery (T=1/5/6, APPEND, MULTI-BATCH, split policy) — but it launches
    the **shipped** kernel (:303/:319), not the production default unified
    v2 (only the T=1 `v2_discriminator` bit-identity touches v2, :639+).
    **This is the clone gate to fix in Step 1.**
  - `tests/slice3_i8_dequant_test.cpp`: host **mirror** of prologue math —
    not production code (fix in Step 1).
  - `tests/slice2_byteid_test.cu`: T=1 byte-identity transcription check
    (shipped bf16 kernel vs slice-2 kernel) — keep, not a chain.
  - `tests/ops/test_gqa_attention.cpp`: op-level criteria, both dtypes —
    keep.
  - `tests/slice3_i8_s8qk_cpu_ref.cpp`: CPU ref for a FUTURE s8-QK port —
    NOT the current production path; do not gate against it.
  - `tests/kvarn_int8_qk_cpu_ref.cpp`: KVarN lever-1 ref — not i8 production.
- **CI**: `run_ci.sh` lives on the CI-only branch
  (`wo/kv-uniform-ci-gate`, docs/126 §0) — CI wiring lands there, syncs to
  `wo/kv-uniform` the same way the decode-guard-cells step landed. Fast gate
  already runs the slice4 D1–D7 phase gate after the unit battery.
  `decode_guard_cells_ci.sh` takes `BASE_DTYPE` (default `kvarn_k4v2`);
  `decode_guard_baseline.json` **already contains** `mtp_on_int8_*` and
  `mtp_on_bf16_*` cells — ungated today. **CI never ratchets baselines** —
  owner decision only (docs/104 §6).
- **Reference contamination** (docs/130 §6): in-process references are
  contaminated — the MTP round gate's reference run is always a fresh
  process.

## 4. Key call sites (anchors — verify line numbers before editing)

- `src/ops/launcher/gqa_attention_decode.cu` — `i8_decode_route` :29 (banner),
  `gqa_small_t_split_count` :76-91 (I8 split policy — I7),
  `launch_tc_partial_bf16` :116-141 (B3–B6 entry), `launch_tc_partial_i8`
  :144 (shipped A/B), `launch_tc_partial_unified_i8` :431 (I3–I6 entry),
  v2 kernel launch :465-467. **The launcher IS the production route — the
  gate drives it, not a re-launch of the kernel.**
- `src/ops/kernel/gqa_decode_slice3_i8_v2.cuh` — production i8 partial kernel
  (dequant prologue + bf16 body + D-split partition) — I2–I6 dump hooks here.
- `src/ops/kernel/gqa_attention_decode_bf16.cuh` —
  `gqa_attention_small_t_tc_partial_bf16_kernel` — B3–B6 dump hooks.
- `src/ops/kernel/gqa_attention_kv_quant.cuh` — `gqa_kv_quant_code` (I1),
  `gqa_kv_dequant_i8x8_from` (I2 math reference).
- `src/ops/launcher/gqa_attention_prefill.cu` + `gqa_attention_prefill_{i8,bf16}.cuh`
  — prefill chains (Step 5).
- `src/ops/kernel/speculative_round.cuh` — MTP accept/verify (MTP round gate
  dump points, Step 4).
- `src/runtime/tp2/tp2_backend.cpp` — single-seq MTP round loop (dump points
  B1/B4/B5 pattern from `phase_gate_mb`); KVarN-only batched gates at :368/:501
  (do NOT touch).
- `src/targets/qwen3_6/impl/state/decoder_state.cpp` :94-98 — `mtp_cache()`
  dtype (context only).
- `tests/phase_gate.cu` (wo-phase-gate worktree) — `PinnedInput::make`,
  `write_phase`, `PHS1` header, `--mutate-*` — **copy verbatim, do not
  change output**.
- `tools/bench/phase_gate_mb.py` — MTP converged-state comparator — **reuse
  verbatim** (dtype-agnostic).

## 5. Design decisions (FINAL — do not re-litigate)

1. **Gate the production default route via the launcher** (unified v2 for i8,
   `tc_partial_bf16` for bf16): split policy, capacity check, partial kernel,
   reduce — exactly what production runs. Clone gates (launching a kernel
   production doesn't launch by default) are REJECTED (docs/132 §2.1).
2. **`PinnedInput` is reused verbatim** (sha ratchets); the i8 chain STARTS at
   bf16 K/V and runs the PRODUCTION append kernel — I1 is production
   quantize, not a CPU pre-quantize.
3. **Byte phases:** I1/I2/B1 gate as sha equality (tolerance 0 — fp16 scale
   + RNE + clamp are exact on CPU). FP phases: FP64 ref, tolerance
   `max(10×observed, 1e-5)` pinned once on the known-good build.
4. **Dump hooks are null-checked and zero-overhead when off** (the slice4
   pattern), land in `src/` on this branch as one reviewable unit, and print a
   route banner (`[i8-decode-route] unified` / `[bf16-decode-route]
   tc_partial`) into the test log — a gate that can't prove which route it
   ran is a clone gate.
5. **MTP round gate = single-sequence** B1/B4/B5 dumps (env-gated,
   `NINFER_MB_PHASEGATE`-style) + fresh-process isolated single-sequence
   reference + `phase_gate_mb.py` comparator verbatim, run at
   `KV_DTYPE=int8` and `bf16`. A multi-lane batched i8/bf16 runner is
   REJECTED for this order (feature, not gate; docs/137 §6; revisit only if
   the owner wants multi-batch MTP on i8/bf16).
6. **Test trigger is a binary, not the server** (`ninfer_phase_gate_i8b`);
   the server is used only for live-path acceptance (Steps 4–5) and the bf16
   battery (Step 3/6).
7. **CI wiring lands on the CI-only branch** (`wo/kv-uniform-ci-gate`), synced
   to `wo/kv-uniform` after merge — never edited in the main tree.
8. **The `NINFER_I8_DECODE=shipped` route is not gated** (rollback path;
   covered only if explicitly selected). A route flip flips the baseline with
   a documented byte-diff record, never silently.
9. **Carve-outs are documented, never silent** (lockstep rule, docs/128 §2.5 /
   docs/137 §2): impacted phase ⇒ its oracle/tolerance updated in the SAME
   commit with a byte-diff A/B record.
10. **`slice3_i8_test` keeps its battery** but retargets at the production
    route in Step 1 (launcher-driven, route banner in log) — it becomes the
    fast standalone version of the I-chain, not a clone gate.

REJECTED: pre-quantizing on the CPU to simplify I1 (loses the point — I1 IS
the production quantize); a new artifact format (PHS1 is ratcheted); a
separate gate binary per dtype (one binary, `--kv-dtype` flag); gating the
s8-QK ref (not production).

## 6. Execution order (commit + test each step before the next)

**Testing standard (applies to every step):** tests must simulate REAL
behavior, not kernel-level interactions in isolation. A step that touches any
server-facing path is NOT done on unit tests alone — the actual call sequence
the server performs (MTP rounds, request lifecycle) must be exercised
end-to-end (test suite or live request). The 2026-08-24 KVarN defect
("append jumped pages") passed every kernel-level test and failed on the
first live request — that is the failure mode this rule exists for. For MTP
regressions, start from the docs/131 workflow (INVASERT → cause map /
HASHPT → T0 → T2), not manual bisection.

### Step 1 — M0: i8 skeleton + I1/I2 + production-route fix (docs/137 §8 M0)
Create `wo/phase-gate-i8` worktree (§2). Add null-checked dump hooks for I1
(codes+scales planes after the append) and I2 (dequantized bf16 tiles) in the
production i8 path. Build `build/tests/ninfer_phase_gate_i8b` (CMake target
like the existing oracle tests; link `ninfer_ops` + `ninfer_artifact`):
`--kv-dtype i8` runs pinned bf16 K/V → PRODUCTION append (I1 dump) →
PRODUCTION v2 decode prologue (I2 dump) → CPU refs (quantize exact ops;
dequant `bf16(code·half(scale))`) → sha compare. `PinnedInput` copied
verbatim from wo-phase-gate. Add `--mutate-i1`/`--mutate-i2` (scale
×(1+1e-3) class). **Fix the clone gate (docs/137 §7.1/§7.2):** retarget
`slice3_i8_test`'s battery at the production route (unified v2 via
`launch_tc_partial_unified_i8`, route banner in log; the shipped kernel stays
covered only under explicit `NINFER_I8_DECODE=shipped`), and make
`slice3_i8_dequant_test` call the production prologue helper instead of a
host mirror.
**Tests (must pass before moving on):**
- `ninfer_phase_gate_i8b --kv-dtype i8`: I1/I2 PASS (sha), chain <10 s.
- `--mutate-i1` (and `--mutate-i2`): FAILS at I1 (resp. I2), exit code =
  phase index.
- `/usr/bin/ctest -R "slice3_i8"`: battery green **on the production route**
  (banner in log), incl. a mutation in the v2 kernel now failing it.
- Full unit battery green (no regressions).

### Step 2 — M1-i8: full decode chain I3–I7 (docs/137 §8 M1)
Dump hooks in `gqa_decode_slice3_i8_v2_kernel` (QK scores, probs, PV
accum, pre-reduce partials) + shared reduce (finals) — null-checked,
zero-overhead. CPU refs reuse the `kvarn_decode_cpu_ref.h` FP64 pattern
(canonical bf16 body is shared). I7 = pure table validation of
`gqa_small_t_split_count` (tokens==5/6 windows + capacity): config exists,
envelope OK, expected pick — no GPU. Test drives the **launcher** (split
policy + capacity + kernel + reduce), route banner required.
**Tests (must pass before moving on):**
- `--kv-dtype i8 --full`-less chain I1–I7 PASS; <10 s.
- **Historical bug**: replay of the docs/104 long-ctx I8 defect (or a
  dequant-prologue mutation) FAILS at I2/I3, not e2e.
- I7 table test: a wrong split pick for a pinned (tokens, window) FAILS.

### Step 3 — M1-bf16: full decode chain B1–B7 (docs/137 §8 M1)
B1 byte-identity (bf16 paged commit/append, `GqaAppendInput` fused path).
Dump hooks in `gqa_attention_small_t_tc_partial_bf16_kernel` + shared reduce
`<Int8=false>`; CPU refs (FP64) near-duplicate Step 2's. B7 = launch config /
envelope table validation (bf16 path; the 40360 crash config documented as
the invalid-envelope case until Step 5 defines the valid range).
**Tests (must pass before moving on):**
- `--kv-dtype bf16` chain B1–B7 PASS; <10 s.
- A reduce-kernel mutation (shared kernel — verify it FAILS at I6 too,
  proving the shared reduce is covered for both dtypes).
- B1: a corrupted append FAILS at B1 (byte identity).

### Step 4 — M2: CI wiring + tolerance table + bf16 battery + ratchet cells (docs/137 §8 M2)
`tools/bench/phase_tolerance_i8bf16.json` (byte phases = 0; fp phases pinned
once on the known-good build). `run_ci.sh` on the CI-only branch: FAST —
`ninfer_phase_gate_i8b` both dtypes + comparator after the slice4 block
(~2–4 min); FULL — mutation battery per dtype (I1/I2/I3–I7, B1/B3–B7),
`BASE_DTYPE=int8` + `BASE_DTYPE=bf16` decode-guard cells (baseline cells
already in JSON), and the **new bf16 correctness battery**
(`KV_DTYPE=bf16`, T1–T11 in `serve_correctness_ci.sh`). Sync the CI branch to
`wo/kv-uniform` the way the decode-guard-cells step landed.
**Tests (must pass before moving on):**
- Fast CI green on the branch: table shows I1–I7/B1–B7 PASS.
- **Negative (CI-level)**: a broken I5 build fails fast CI with the table
  pointing at I5; a broken bf16 reduce fails at B6.
- `--full` green incl. the new bf16 battery (first run: record acceptance +
  step-ms; do NOT ratchet — commit the numbers to `results/` and let the
  owner decide).
- Ratchet cells: `decode_guard_cells_ci.sh BASE_DTYPE=int8` and `=bf16`
  passes run and gate against the existing baseline (no ratchet raise).

### Step 5 — M3: prefill chains I-P1..P4 / B-P1..P4 + envelope gate (docs/137 §8 M3)
Tile-dump hooks (env-gated) in `gqa_attention_prefill_{i8,bf16}_kernel`; both
i8 fill paths (tiled `tokens>=128 && KVHeads==2` + scalar) gated. B-P3:
**define and pin the valid bf16 (ctx, SM) envelope for 36-SM cards** — the
40360 crash config is either fixed (with the fix's byte-diff record) or a
documented carve-out with the pinned valid range; add the envelope table
validation gate (the docs/128 P6 form).
**Tests (must pass before moving on):**
- `--kv-dtype i8|bf16 --prefill`: I-P1..P4 / B-P1..P4 PASS (I-P1 byte, both
  fill paths; B-P1 byte).
- A fill-path mutation (tiled OR scalar — both) FAILS at I-P1.
- Envelope table: every pinned (ctx, SM) pair either in the valid set with a
  config, or explicitly carved out in the table + docs.
- **Live-path (server-facing step):** `serve_correctness_ci.sh` with
  `KV_DTYPE=bf16` (new) and `KV_DTYPE=int8`, SPEC unset: T battery green
  through the changed prefill path.

### Step 6 — M4: MTP round gate, i8 + bf16 (docs/137 §8 M4)
Env-gated B1/B4/B5 dump points (round inputs: vid/vpos/draft/F/ext/anchor;
accept decision: a/lic; converged round state: cur_F/cur_anchor/generated/
next_ext/active) in the **single-sequence** MTP round loop
(`tp2_backend.cpp` / `speculative_round.cuh` — dtype-agnostic region; do not
touch the KVarN-only batched gates). Driver: test binary or
`serve_correctness_ci.sh SPEC=mtp` with the dump env (your choice; record
which). Reference: **isolated fresh-process** single-sequence run (never
in-process). Comparator: `phase_gate_mb.py` verbatim. Run at
`KV_DTYPE=int8` and `bf16`. Pinned geometry set covers the historical MTP
degenerate cases (T=256 terminal ext=0, page-boundary rounds — docs/130 §8).
Wire into CI full (after the decode-guard cells).
**Tests (must pass before moving on):**
- Both dtypes: B1/B4/B5 vs R1/R4/R5 converged-state compare green; token
  exactness.
- **Negative**: a 1-line accept-path mutation (e.g. forced `lic−1`) FAILS at
  the accept phase (exit 20-class signature).
- **Live-path (server-facing step):** live server, `SPEC=mtp`,
  `KV_DTYPE=bf16` AND `KV_DTYPE=int8`: request through MTP rounds succeeds
  (the bf16+MTP live run is NEW coverage — it does not exist in CI today).
- MTP T0–T3 suite (docs/131 T3.1 merge gate) green on the final commit:
  `run_t0.sh` + INVASERT stress + T2(T=64) — you touched the round loop.

### Step 7 — M5: MTP-verify GEOMETRY in the i8/bf16 decode phase gate (coordinator-found gap 2026-09-03)
**Gap (verified in source):** `phase_gate_i8b.cu` runs the decode kernel at `TokenTile=1, MultiBatch=false, Masked=false` (plain single-token decode). So the **MTP verify attention** (T=k+1, masked draft window) is **NOT exercised** for i8/bf16 — a bug in the MTP token-generation/verify path would NOT be caught. The kvarn D-chain runs "1 MTP round (k=3)" (docs/128 §8.1); i8/bf16 do not. (Note: Step 6's MTP *round* gate is batched-vs-seq and is VACUOUS for i8/bf16 — no batched runner — so it does not close this gap either.)
**Task:** extend the i8/bf16 decode phase gate to run the **MTP verify geometry** — `TokenTile=k+1` (tokens 5/6), `Masked=true` — mirroring the kvarn D3/D4/D5/D6/D7 pattern, so the MTP verify attention (dequant/QK/softmax/PV/reduce at the draft window + the causal mask) is gated for both dtypes.
**Tests (must pass before moving on):**
- `ninfer_phase_gate_i8b --kv-dtype i8` and `bf16` run an MTP verify round (T=k+1, masked), dumping per-phase artifacts.
- **Negative**: a mutation injected into the MTP verify path FAILS at the correct phase (exit = phase index) for BOTH dtypes — proves it catches an MTP-verify bug, not just plain decode.
- Route banner in the log proving the production MTP-verify route ran (not a clone).
- Wire into run_ci.sh (fast + full mutation battery) on the CI branch, synced to wo/kv-uniform.

### Step 8 — M6: KVAR PREFILL phase gate (P1-P7) (coordinator-found gap 2026-09-03)
**Gap (verified in source):** `phase_gate_prefill.cuh` gates ONLY i8/bf16 prefill (P1-P4). KVarN prefill is covered ONLY by the oracle test (`prefill_attention_oracle_test.cu`) — there is NO per-phase KVarN prefill gate. (KVarN prefill itself is DONE + working + correct; this gate adds per-phase localization, not correctness.)
**Task:** build the KVarN prefill phase gate (P1-P7), mirroring the D1-D7 decode chain + the I-P/B-P structure, on the production KVarN prefill route (commit/quantize → materialize/dequant-once → shared BF16 FA2 flash, docs/127 §4.1).
**Tests:** route banner (production route, not a clone); per-phase artifacts vs CPU/FP64 refs; a mutation at each phase FAILS at that phase (exit = phase index); wire into run_ci.sh fast + full mutation battery.

## 7. Constraints (non-negotiable)

- **Worktree only:** no edits in `/home/intel/ninfer/repo` outside
  `wo/phase-gate-i8` (and the CI-only worktree for Step 4), ever.
- **Live end-to-end before "done":** Steps 5 and 6 are server-facing — not
  complete until the real server runs them (§6 testing standard).
- **No damage:** commit per step with a message naming the step; keep the
  tree buildable at every commit; never leave uncommitted or untested code.
- **`PinnedInput` output is immutable** (sha ratchets); do not change
  `phase_gate_mb.py` semantics (reuse verbatim; a bug in it is fixed with a
  documented A/B record).
- **Never ratchet baselines** (`decode_guard_baseline.json`,
  `results/baseline.json`) — CI gates against them; raising is an owner
  decision (docs/104 §6). First bf16 numbers are committed to `results/`
  only.
- **Do not touch:** the KVarN-only batched MTP machinery (`tp2_backend.cpp`
  :368/:501, `set_kvarn_batched_active`), the slice4 gate, the KVarN
  tolerance/baseline files, `NINFER_I8_DECODE` semantics (rollback stays).
- **Dump hooks** are null-checked + zero-overhead when off + route banner in
  the test log (decision 4); a gate that can't prove its route is a clone
  gate.
- **Server agency:** full agency (stop/start/reconfigure) — standard way is
  `bash tools/ops/run_ci.sh` from the worktree root; `pkill -x` only; at the
  end leave the live server restored per LAUNCH.md (or say so in the report).

## 8. Definition of done

0. **COMPLETION CI GATE (MANDATORY — run the FULL suite, not the fast gate):**
   `bash tools/ops/run_ci.sh --full` from your worktree root **must PASS** on
   the final commit before you report done. The fast `run_ci.sh` (no `--full`)
   is per-step iteration only and does **NOT** count as done — an agent that
   ran only the fast gate has NOT completed this order. Record the `--full`
   verdict + the auto-generated Desktop results doc (written at the end of
   every run) in the report.
1. Steps 1–7 committed to `wo/phase-gate-i8` (CI step on
   `wo/kv-uniform-ci-gate`, synced) with passing tests at each step —
   including the live-path tests for Steps 5–6.
2. **Live proof:** fresh server launches with the new code serve real
   requests through (a) bf16 prefill (Step 5 T battery) and (b) MTP rounds
   on BOTH `KV_DTYPE=bf16` and `KV_DTYPE=int8` (Step 6) — launch commands +
  nvidia-smi clocks recorded in the report.
3. Fast CI on the branch: `ninfer_phase_gate_i8b` both dtypes green with a
   printed phase table; a deliberately broken I5/B5/accept build fails CI at
   the right phase (record the negative run).
4. **Test gates (docs/131 T3.1):** MTP/kvarn decode path touched →
   `run_t0.sh` + INVASERT + T2(T=64) green on the final commit; merge gate
   via `run_ci.sh --full`.
5. `results/` committed: first bf16 battery numbers, chain timings (<10 s
   per dtype), fast CI wall time (~2–4 min), clocks.
6. Report format: one paragraph per step + key numbers table (phase table
   per dtype, negative-test outcomes, CI wall time, live-run commands).

### M6 — KVarN Prefill Phase Gate Remediation (docs/140)

- **Tile-Dump Hooks Implemented:** Production FA2 kernel `gqa_attention_prefill_bf16_kernel` in `src/ops/kernel/gqa_attention_prefill_bf16.cuh` instruments tile capture for $S_t$ (P4), $m_t, l_t, \alpha_t, P_t$ (P5), and $O_t$ + final $O$ (P6). Hooks are zero-cost when off (separate kernel specialization without instrumentation overhead) and verified bit-identical when off (`bit_off=PASS`, 0 bit differences against production uninstrumented launch).
- **P4–P6 Multi-Key / Multi-Tile:** `kTokens` increased to 256 ($4\times 64$-token key tiles, 3 cross-tile rescales). Verifies token 63 (1 tile) and token 255 (4 tiles, 3 cross-tile rescales) across all 4 KV-heads (20 tiles, 1280 keys total). All intermediate states verified against incremental tile-by-tile FP64 CPU reference simulating online FlashAttention-2 accumulation.
- **Real P3 Launch Envelope Check:** Launch configuration (grid sizing, smem $\le 96\,\text{KiB}$, occupancy $\ge 1$ block/SM) actively verified across contexts $\{1024, 4096, 8192, 16384, 32768\}$. 40360 carve-out actively validated (rejected as exceeding max context envelope $> 32768$), replacing any static tautologies. Paged cache identity table mapping validated.
- **Mutation Battery:** Negative mutations `--mutate-p1` through `--mutate-p7` verified to localize to exit codes 1 through 7 in 7/7 runs. Multi-tile intermediate states targeted for P4, P5, P6.
- **CI Status:** Fast CI passes cleanly with green verdict.

