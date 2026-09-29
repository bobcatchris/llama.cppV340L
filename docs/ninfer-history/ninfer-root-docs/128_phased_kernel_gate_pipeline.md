# 128 — Phased kernel gate pipeline (accuracy localization)

Status: PLAN (2026-09-01). Owner: next agent after MultiBatch.
Goal: when a kernel update breaks accuracy, the gate tells us **which phase**
broke — not just that something broke. Failure localizes to one component →
we know what to fix.

## 1. Problem

The 08-31 decay bug is the template case: the symptom was an e2e sampling
acceptance drop; the root cause sat in the KVarN keypair commit/dequant path
(§3 phase D1/D2). Finding it took ~1 day of manual bisection, A/B builds, and
diffs, because **every accuracy gate today is end-to-end** (token correctness
vs bf16 ref, A2 identity, acceptance %, t/s). When e2e fails, the hypothesis
space is the whole kernel stack.

Owner requirement (2026-09-01): decompose decode and prefill into ~10 phases;
give each phase its own accuracy test; a failing run must localize to a
single phase.

## 2. Principle: chained oracles on phase-boundary artifacts

Each phase boundary produces a **deterministic artifact** (packed codes,
bf16 tiles, fp32 scores, fp32 probs, accumulated outputs, logits, tokens).
A phase test compares the GPU artifact against an **independent reference**
(CPU implementation of that phase, or FP64 oracle) for the same input.

- **Chain mode (CI):** phase N's test consumes the *actual GPU artifact of
  phase N-1*. If the chain passes through N-1, then a deviation at N is
  provably introduced by N. **First failing phase = the broken phase.**
- **Reference mode (dev):** phase N always consumes the *reference*
  upstream artifact. Tells you exactly which phase(s) are wrong in parallel,
  but does not prove the chain.
- **Comparison policy:** byte-identity where the CPU reference uses the same
  fp32 accumulation order (int8/bf16 paths); FP64 + pinned tolerance for
  MMA accumulation (the direct-to-fragment lever already mandates an FP64
  oracle — build it once, use it for both).
- Cascades: once phase N fails, downstream artifacts deviate too; the gate
  marks them `CASCADE (upstream D<n> broken)` and stops the search.

## 2.5 THE LOCKSTEP RULE (read this first)

**The phase gates are part of the pipeline contract, not a one-time
snapshot.** We are actively changing these kernels (Bc=64 tiles,
direct-to-fragment K dequant, every future KV dtype variant — docs/127 §5
queue). Any change that touches a phase's implementation MUST update that
phase's oracle/reference/tolerance **in the same commit**:

1. Before the change: run the phase gate — record the green baseline.
2. Make the change. Re-run the phase gate.
3. If the phase artifact changed (hash/tolerance divergence):
   - (a) **accuracy preserved** (verified against the FP64 oracle and/or
     the e2e token diff) → update the reference/tolerance table in the SAME
     commit, with a byte-diff A/B record attached (the docs/126 Tier 3
     pattern: what diverged, why it is safe, the evidence). Carve-outs are
     documented, never silent.
   - (b) **accuracy changed** → that is a bug in your change; fix it.
4. CI enforces this: a commit touching a phase's source that leaves the
   gate red with an *undocumented* divergence fails — the author must
   either record the new expectation (3a) or revert.

The reference implementations (CPU codec/QK/softmax/PV refs, tolerance
table, pinned prompt, dump format) live in the repo under `tests/` and
`tools/bench/` and are versioned with the kernels they check. When the
pipeline changes shape (new phase, split phase, merged phase), the phase
tables in §3/§4 and the gate script change in the same commit. **A kernel
commit that impacts a pipeline phase and does not update that phase's
gate is an incomplete commit.**

## 3. Phase decomposition — decode (unified KVarN MTP path)

| # | Phase | Artifact at boundary | Oracle | Today | Cost |
|---|-------|---------------------|--------|-------|------|
| D1 | commit/quantize (append → KVarN codes + per-key scales, rotation, field-innermost layout) | codes + scales + layout | CPU codec ref (`tests/kvarn_codec*.cpp`) | partial (unit, not chained) | <1 s |
| D2 | dequant/materialize (codes → bf16 K/V) | bf16 K/V tiles | materialize GPU oracle + CPU ref | test exists, not in chain | <1 s |
| D3 | draft (MTP draft head W8G32 → 3 drafts) | draft tokens | reference draft on pinned input | none (implicit in e2e) | <1 s |
| D4 | QK (slice kernel Q@Kᵀ, Bc tiles) | fp32 scores | CPU bf16 ref (small ctx) / FP64 oracle | none (A2 is e2e only) | <1 s |
| D5 | softmax + rescale | fp32 probs | CPU ref | none | <1 s |
| D6 | PV + per-tile scales + fp32 accumulation | accumulated outputs | CPU ref / FP64 oracle | none | <1 s |
| D7 | reduce (split-K / multi-CTA shared reduce) | per-head finals | CPU ref | none (Tier-3 byte-diff covers some) | <1 s |
| D8 | accept/sampling | accepted token (+ acceptance stats) | deterministic (greedy) / distributional (±pp over N rounds) | partial (e2e acceptance) | ~5 s |
| D9 | GDN checkpoint capture/restore, prefix partial | GDN state hash | byte-diff state | partial (T15-T19, e2e) | ~5 s |
| D10 | serving glue (page mgmt, prefix cache, MTP round assembly) | token stream | e2e: A2 identity + token diff vs bf16 ref | full (regress_unified, T1-T19) | ~60 s |

Fast chain D1–D7: **~5–10 s GPU**. Full chain D1–D10: **~2–3 min**.

## 4. Phase decomposition — prefill (materialize route; shared flash)

| # | Phase | Artifact | Oracle | Today |
|---|-------|----------|--------|-------|
| P1 | commit/quantize | = D1 | = D1 | partial |
| P2 | materialize (dequant-once per tile; C2 sync lives here) | bf16 K/V staged tiles | = D2 + CPU ref | not chained |
| P3 | flash QK tile (Br=64/Bc=64, cp.async staging) | scores tiles | CPU / FP64 ref | none (A3 route-vs-route is e2e) |
| P4 | online softmax/rescale across C tiles | rescaled probs | CPU ref | none |
| P5 | flash PV + epilogue (per-key scale, i8 family) | O tiles | CPU ref | none |
| P6 | launch table / envelope (config pick per (dtype, ctx, SM count)) | selected config | table validation: config exists, envelope OK, expected pick | today: crash = the only test |
| P7 | e2e | tokens | token diff vs bf16 ref, A2 identity | full |

Flash is fused, so fewer boundaries than decode; P3–P5 need env-gated
tile-dump hooks inside the FA2 kernels (the direct-vs-materialize A3 harness
already proves route-vs-route diffs are observable).

## 5. Localization semantics

CI prints a table and exits with the first failing index (0 = clean):

```
D1 commit    PASS  byte-identical (cpu codec)
D2 dequant   PASS  max|Δ|=0
D3 draft     PASS  3/3 tokens
D4 QK        PASS  max rel 2.4e-7
D5 softmax   FAIL  max rel 3.1e-2 (expected <1e-5)   ← fix D5
D6 PV        CASCADE (upstream D5)
...
```

Statistical phases (D8 sampling) are not byte-localizable; they gate as
stats, and if the chain above is clean, the search space collapses to
{sampling code, weights, D8 glue} — which is the actual residual risk.

## 6. Milestones (agent-ready, with acceptance criteria)

**What already exists in-tree (reuse, don't rebuild):**
- D1 oracle: `tests/test_kvarn_codec.cpp`, `tests/test_kvarn_codec_edge.cpp`,
  `tests/kvarn_acc_fwht_cpu_ref.cpp` (rotation), `tests/kvarn_sinkhorn_convergence.cpp`
- D2 oracle: `tests/kvarn_materialize_oracle_test.cu` (GPU) + the CPU dequant path
- D4 CPU ref (dormant): `tests/kvarn_codespace_qk_cpu_ref.cpp` (docs/83);
  int8/s8 siblings: `kvarn_int8_qk_cpu_ref.cpp`, `slice3_i8_s8qk_cpu_ref.cpp`
- D3 oracle pattern: `tests/targets/qwen3_6_27b/test_draft_head.py` (W8G32 draft head)
- Prefill oracle base: `tests/prefill_attention_oracle_test.cu`
- A/B harness pattern: `tests/debug_kvarn_decode_ab.cu`, `tests/bench_kvarn_attention.cu`

**M0 — chain skeleton + D1→D2 (≈0.5–1 day).**
Dump hooks for D1/D2 + test binary `build/tests/ninfer_phase_gate` (pattern:
`kvarn_materialize_oracle_test.cu`) + `tools/bench/phase_gate.sh` with the
chain table + per-phase artifact files (see §9).
*Accept:* (a) known-good build: D1/D2 PASS, chain <10 s; (b) **negative test**:
a 1-line mutation build (e.g. dequant scale ×(1+1e-3)) FAILS at D2. A gate
that can't catch a known bug doesn't ship.

**M1 — full decode chain D1–D7 (≈2–3 days).**
Wire D4 from the dormant `kvarn_codespace_qk_cpu_ref.cpp`; extend the same
file (or siblings) for D5 (softmax+rescale), D6 (PV+scale), D7 (reduce) —
same-input fp32 accumulation order for byte-identity, FP64 + tolerance where
the tile order can't be matched. D3 via the `test_draft_head.py` pattern.
*Accept:* (a) known-good build: D1–D7 PASS; (b) **the 08-31 decay bug**: the
pre-fix build (or a replay of its keypair bug) FAILS at D1 or D2 — the
gate must localize the historical bug; (c) chain <10 s GPU.

**M2 — CI wiring + tolerance table (≈1–2 days).**
`run_ci.sh` fast mode: unit battery + phase gate D1–D7 (~+2 min on every
CI). Full mode: chain D1–D10 (adds D8 acceptance stats, D9 GDN state hash,
D10 A2 identity + token diff — all already exist, just chained). Tolerance
calibration per §9.4. *Accept:* a broken D5 build fails CI with the table
pointing at D5, in one run.

**M3 — prefill chain P1–P7 + D3 formalization (separate, later).**
Tile-dump hooks in the FA2 kernels (A3 harness proves route-vs-route diffs
are observable); reuse `prefill_attention_oracle_test.cu`. P6 (launch table
validation: config exists + envelope OK + expected pick) is a pure table
check, no GPU.

**Total to M2 ≈ 1 week.** (The earlier "2–3 days" estimate was wrong: the
D4–D7 CPU references are the long pole, even with the dormant QK ref.)

## 7. Cost / benefit / risk

- **Benefit:** any future kernel regression localizes in one CI run
  (~2–3 min) instead of ~1 day of manual bisection; makes aggressive
  overnight kernel iteration (Bc=64, direct-to-fragment, future KVarN
  variants) safe.
- **Cost:** ~1 week to M2 (see §6 milestones); ~2 min GPU CI.
- **Risk (a):** accumulation order — byte-identity requires the CPU ref to
  accumulate in the same order; if too fragile per phase, pin an FP64
  tolerance per phase in the table (tolerances are reviewed like code).
  **Risk (b):** dump-hook overhead must be exactly zero when the env is off.
  **Risk (c):** pinned prompts must be immutable — commit them, hash them
  into the artifact file.
- **Non-goal:** does not replace the perf gates (decode guard, t/s ratchet).
  Those answer "is it fast"; this answers "where is it wrong".

## 8. Execution spec (for the first agent)

### 8.1 Trigger interface — a test binary, NOT the server
No server, no HTTP, no serving glue: a new test binary
(`build/tests/ninfer_phase_gate`, CMake target like the existing oracle
tests) that loads the artifact + TextContext directly, runs the pinned
prompt, and drives the phases in order:
```
prefill(pinned prompt)          → dump D1 (codes+scales), D2 (dequant tiles)
1 MTP round (k=3)               → dump D3 (drafts), D4 (scores), D5 (probs),
                                   D6 (accumulated), D7 (per-head finals)
--full: N accept rounds (D8), GDN state hash (D9), A2 identity +
        token diff vs bf16 ref (D10)
```
Kernel hooks take a **dump arena pointer from the context** (null in
production → the hook is a null check, exactly zero cost; no env parsing in
hot paths). Env-gated dumping inside the *server* is deferred to M3
(prefill tile dumps need the live request path).

### 8.2 Pinned prompt
`tests/data/phase_gate_prompt.txt` — committed, immutable, ~512 tokens
(same unit-phrase generator as decode_guard.sh so it's reproducible from a
seed). The file's sha256 is written into every artifact header; the gate
refuses to compare artifacts whose prompt hash differs from the ref.

### 8.3 Artifact format (one file per phase)
`phase_<id>_<runstamp>.bin`:
```
u32 magic 'PHS1' | u32 version | u16 dtype_code (u8|i8|fp16|bf16|fp32|u32codes)
| u64 prompt_sha256_hi | u32 elem_count | raw bytes (row-major; layout per
phase documented in the §3 table column)
```
One file per phase because: byte-identity checks reduce to **hash
comparison** (the gate table prints per-phase sha256), diffs are trivial,
and a broken phase is visible as the first hash that diverges — even when
running two builds side by side.

### 8.4 Tolerance calibration (once, then pinned)
On the known-good build, run the chain and record per-phase max|Δ| vs ref:
- byte phases (D1, D2, D3, D8-greedy): tolerance = 0 (hash equality);
- fp phases (D4–D7): tolerance = max(10×observed, 1e-5), pinned in
  `tools/bench/phase_tolerance.json` (committed, reviewed like code).
Changing a tolerance is a reviewable change, not a silent one.

### 8.5 Gate output + exit code
```
D1 commit    PASS  sha 9c2f… byte-identical (cpu codec)
D2 dequant   PASS  sha 77aa… max|Δ|=0
D4 QK        FAIL  max rel 3.1e-2 (tol 1e-5)  ← first failure, exit 4
D5 softmax   CASCADE (upstream D4)
```
exit code = first failing phase index (0 = clean). CI: `set -e`-safe one-liner;
the table is tee'd into the CI log so a red run says *what* to fix.

### 8.6 First day (checklist)
1. Read: docs/127 (gate status + variant template), docs/126 (verification
   runbook + Tier 3 byte-diff pattern), docs/83 (kernel design, M3/M4
   gates), docs/104 §6/§8 (gate policy + ratchet rules), docs/124/125
   (queue + handoff context).
2. Run the existing oracles to confirm green: `test_kvarn_codec`,
   `ninfer_kvarn_materialize_oracle_test`, `prefill_attention_oracle_test`,
   `kvarn_codespace_qk_cpu_ref` (build it; it is dormant — that is the
   first real task of M1).
3. Build M0 (dump hooks + binary + script) on the CURRENT build; capture
   the per-phase sha256 table as the initial green baseline and commit it
   as `tools/bench/phase_baseline.json` (the chain's own "ratchet": any
   future divergence is visible against it).
4. Do NOT start M1 until the M0 negative test (mutated dequant → FAIL at
   D2) passes.

### 8.7 Branch & worktree (where this work happens)

Work in a **new worktree + branch based on `wo/kv-uniform`** — not in
`wo-kv-uniform` (the main agent is actively editing `src/` there), and not
in `wo-kv-uniform-ci-gate` (CI-only branch per docs/126 §0; this work
touches main-branch kernel source, not just CI):

```
cd /home/intel/ninfer/repo
git worktree add /home/intel/ninfer/worktrees/wo-phase-gate -b wo/phase-gate wo/kv-uniform
cd /home/intel/ninfer/worktrees/wo-phase-gate
# fresh build in THIS worktree (own build/ dir — never build into the
# main agent's build/; plan for a full compile)
```

- Everything — dump hooks (`src/ops/**`), oracles (`tests/`), gate script
  (`tools/bench/`), tolerance/prompt data — lands on `wo/phase-gate` as
  ONE reviewable unit.
- When M1 acceptance is green: **merge `wo/phase-gate` → `wo/kv-uniform`**
  (small diff: hooks are null-checked pointers, tests are additive).
- The `run_ci.sh` wiring edits then sync to `wo/kv-uniform-ci-gate` the
  same way the decode-guard-cells step landed (docs/126 §0 convention).
- GPU/port: M0–M2 need only the test binary (direct TextContext — **no
  server, no port**), ~5–10 s GPU per run. Only M3 (server-side prefill
  tile dumps) needs a server; then coordinate port 8091 as usual
  (check `ss -tln` + `nvidia-smi` first).

## 10. Gate policy (the aggressive part)

- **Lockstep (binding, see §2.5):** impacted pipeline phase ⇒ its gate
  (oracle/reference/tolerance) updated in the SAME commit, with a
  byte-diff A/B record when the artifact intentionally diverges.
- Any commit touching `src/ops/kernel/**` or the launch tables: phase gate
  D1–D7 locally before push; CI runs the full chain.
- New KV dtype (int4 non-kvarn, k5v4, ...): D1/D2 oracle + guard cells
  (docs/127 §4 variant template) must land with the dtype, before merge.
- A byte-identity carve-out (e.g., tile change that changes accumulation
  order) requires: updated tolerance table entry + a Tier-3-style byte-diff
  A/B record (docs/126 Tier 3 pattern). Carve-outs are documented, never
  silent.

## 11. References (what the agent uses)

Docs (all in `docs/` of this worktree):
- **127** — gate status, next-lever queue, KV-dtype variant template (§4)
- **126** — verification runbook (Tiers 1–3; the byte-diff A/B pattern)
- **83** — KVarN code-space kernel design (M3/M4 gates, direct-to-fragment,
  FP64-oracle requirement)
- **104** §6/§8 — unified performance plan: gate policy + ratchet rules
- **124** — optimization queue (what's coming: Bc=64, direct-to-fragment)
- **125** — handoff (MultiBatch state)
- **122** — performance knowledge base (numbers + attribution)
- **120** — Phase 3 status (prefill C2/C3 context for the P-chain)
- **106/114** — launch tables (the P6 validation target)
- **113** — route-vs-route (A3 harness; prefill chain context)
- **78** — G2a gate context (why 40k must stay in every matrix)

In-tree code/scripts (reuse patterns, don't reinvent):
- oracles: `tests/test_kvarn_codec{,_edge}.cpp`, `kvarn_acc_fwht_cpu_ref.cpp`,
  `kvarn_sinkhorn_convergence.cpp`, `kvarn_materialize_oracle_test.cu`,
  `kvarn_codespace_qk_cpu_ref.cpp` (dormant), `kvarn_int8_qk_cpu_ref.cpp`,
  `slice3_i8_s8qk_cpu_ref.cpp`, `prefill_attention_oracle_test.cu`,
  `tests/targets/qwen3_6_27b/test_draft_head.py` (D3 pattern)
- harnesses: `tests/debug_kvarn_decode_ab.cu`, `tests/bench_kvarn_attention.cu`
- scripts: `tools/bench/decode_guard.sh` (prompt generator + cell runner),
  `tools/bench/decode_guard_check.py` (gate/ratchet semantics),
  `tools/ops/run_ci.sh` (CI wiring points), `tools/bench/regress_unified.sh`
  (the D10 pattern)
