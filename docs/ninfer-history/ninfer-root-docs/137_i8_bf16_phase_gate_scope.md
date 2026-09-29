# 137 — i8/bf16 KV-dtype phase gate scope (accuracy localization for the non-KVarN routes)

Status: SCOPE (2026-09-02). Owner: implementer TBC (single writer; gemini available
as second worker for the bf16+MTP lane if split is chosen). Companion docs:
**128** (the KVarN phase gate — this doc is its doc-128-equivalent for the other
two production KV dtypes), 129/130/131 (M0/M2/MTP suite), 132 (review — what
"the same bug" means), 133/134 (resolution, baseline, pinned inputs), 127 (gate
status), 126 (verification runbook + CI-only branch).

## 0. Mission (one paragraph)

docs/128 made KVarN (K4V2) accuracy failures localize to a single phase in one CI
run. The other two production KV dtypes — signed **int8** (`DType::I8`, per-token
group-64, 4 planes) and **bf16** (`DType::BF16`, 2 planes; `src/core/dtype.h`) —
have the same pipeline shape (fused quantize/dequant, split-K decode launcher,
MTP verify round) but **no phase gate**. A kernel regression in the production
i8 or bf16 route today still localizes via ~1 day of manual A/B (the docs/104
long-ctx I8 fix was exactly that). This doc scopes the identical pipeline —
chained per-phase oracles on **production-route** artifacts, first-failing-phase
= broken phase, negative tests, lockstep rule, CI wiring — for I8 and BF16,
**including the MTP phase gate** (§6). Audit in §7: the suspected "same bug"
(docs/132 §2.1 class — a gate that runs a non-production route) **already exists
for i8** and is in scope to fix.

## 1. Problem

- Template case (docs/128 §1): 08-31 decay bug — e2e symptom, root cause in a
  mid-pipeline phase, found by ~1 day of manual bisection.
- i8 precedent: the docs/104 long-ctx I8 fix was root-caused by a shipped-vs-
  unified **A/B build** because no phase decomposition existed; the A/B env
  (`NINFER_I8_DECODE=shipped`) remains the only rollback path.
- bf16 precedent: the bf16 prefill envelope crash
  (`--kv-dtype bf16 --max-context 40360` → `gqa_attention: invalid execution
  envelope or table`; docs/127 §2, 170-SM-tuned) has **crash = the only test**
  today.
- Production KV dtypes: `BF16=0`, `I8=5`, `KVARN_K4V2=8`. Only KVARN has phase
  gates (slice4 D1–D7, `phase_gate_mb`, MTP T0–T3). I8/BF16 have op-level
  tolerance criteria + e2e cells only (§7).

## 2. Principle (inherited from docs/128, unchanged)

- **Chained oracles on phase-boundary artifacts.** Each phase boundary produces a
  deterministic artifact; a phase test compares the GPU artifact against an
  independent reference (CPU implementation or FP64 oracle) for the same input.
  Chain mode: phase N consumes the actual GPU artifact of N-1 → first failing
  phase is provably the broken phase; downstream marked `CASCADE (upstream
  <phase> broken)`.
- **Comparison policy:** byte-identity where the CPU ref uses the same ops/order
  (I1/I2/B1 are byte phases: fp16 scale rounding + round-to-nearest-even + clamp
  are exact on CPU); FP64 + pinned tolerance for MMA accumulation phases.
- **THE LOCKSTEP RULE (docs/128 §2.5, binding, verbatim):** any change touching a
  phase's implementation updates that phase's oracle/reference/tolerance in the
  SAME commit, with a byte-diff A/B record when the artifact intentionally
  diverges. Carve-outs documented, never silent.
- **PRODUCTION-ROUTE REQUIREMENT (binding, docs/132 §2.1):** the gate drives the
  production launcher + production kernel (or dumps from production kernels via
  null-checked hooks). A gate that launches a kernel production does not
  launch by default is a **clone gate — rejected**. This is the bug class the
  owner expects to find: it is already present for i8 (§7.1) and is fixed as
  part of M0.

## 3. Phase decomposition — decode, I8 (production default: unified v2)

Production route (verified, `src/ops/launcher/gqa_attention_decode.cu`):
`NINFER_I8_DECODE` unset → `I8Route::kUnified` → `launch_tc_partial_unified_i8`
(:431) → `gqa_decode_slice3_i8_v2_kernel` (D-split partition, canonical bf16
body + vectorized I8 dequant prologue; :465-467) + shared
`gqa_attention_small_t_reduce_output_kernel<Int8=true>`. A/B: `shipped` →
`launch_tc_partial_i8` (:144) → `gqa_attention_decode_i8_tiled_kernel`.
Quantize is fused into the append kernel (`gqa_kv_quant_code`,
`src/ops/kernel/gqa_attention_kv_quant.cuh`); dequant is fused into the kernel
prologue (`gqa_kv_dequant_i8x8_from` / `prologue_kv_tile_i8`).

| # | Phase | Artifact at boundary | Oracle | Today | Cost |
|---|-------|---------------------|--------|-------|------|
| I1 | quantize/append (bf16 K/V → int8 codes + fp16 scales; scale = fp16(absmax/127), code = clamp(RNE(x·inv_scale), −127, 127); 4 planes) | codes + scales planes | CPU quantize ref (exact ops → **byte phase**) | partial: `slice3_i8_test` APPEND case — shipped route, direct launch, not chained | <1 s |
| I2 | dequant/materialize (prologue: codes → bf16 K/V tiles) | dequantized bf16 tiles | CPU ref `bf16(code·half(scale))` (**byte phase**) | partial: host MIRROR of prologue math (not production code) | <1 s |
| I3 | QK (canonical bf16 body over dequantized K, MMA) | fp32 scores | FP64 CPU ref | none chained (v2 only T=1 discriminator) | <1 s |
| I4 | softmax + rescale | fp32 probs | CPU ref | none | <1 s |
| I5 | PV (bf16 MMA) | accumulated outputs | FP64 ref | none | <1 s |
| I6 | reduce (shared reduce, split-K merge) | per-head finals | CPU ref | none (v2-discriminator byte-identity, T=1 only) | <1 s |
| I7 | split policy / launch config — **I8-specific tiers** (`gqa_small_t_split_count` :76-91: tokens==5 window 128–512 → 32-key tiles; tokens==6 window 128–160 → 24; tokens==6 window 5000–8198 → 192 clamp) + capacity check | selected (splits, capacity) | table validation: config exists, envelope OK, expected pick | none (e2e only) | 0 s |
| I8 | accept/sampling (MTP round, k=3) | accepted tokens + stats | greedy deterministic / distributional (±pp over N) | partial: kv_i8 e2e cells (verify battery) | ~5 s |
| I9 | GDN checkpoint capture/restore, prefix partial | GDN state hash | byte-diff state | shared with KVarN (T15–T19, e2e) | ~5 s |
| I10 | serving glue (page mgmt, prefix cache, MTP round assembly) | token stream | e2e: A2 identity + token diff vs bf16 ref | **full** (int8 T1–T12 battery, CI fast + full) | ~60 s |

Fast chain I1–I7: **~5–10 s GPU** (per dtype). Full chain I1–I10: ~2–3 min.

## 4. Phase decomposition — decode, BF16 (production: tc_partial)

Production route: `launch_tc_partial_bf16` (:116-141) →
`gqa_attention_small_t_tc_partial_bf16_kernel`
(`src/ops/kernel/gqa_attention_decode_bf16.cuh`) + shared reduce `<Int8=false>`.
Commit is a direct bf16 paged copy (fused via `GqaAppendInput`); no quantize.

| # | Phase | Artifact | Oracle | Today |
|---|-------|----------|--------|-------|
| B1 | commit/append (bf16 → k_pages/v_pages) | page bytes | byte-identity (CPU copy ref) | **none** |
| B2 | (no quantize — identity; N/A) | — | — | — |
| B3 | QK (bf16 MMA directly on cache) | fp32 scores | FP64 ref | partial: T=1 byte-identity transcription check + op criteria |
| B4 | softmax + rescale | fp32 probs | CPU ref | none |
| B5 | PV | accumulated outputs | FP64 ref | none |
| B6 | reduce (Int8=false) | per-head finals | CPU ref | none |
| B7 | launch config / envelope (bf16 prefill 170-SM-tuned; the 40360 crash, docs/127 §2) | selected config | table validation + envelope coverage map | none (crash = the only test) |
| B8 | accept/sampling (MTP round) | = I8 | = I8 | **none** (no bf16 CI battery at all) |
| B9 | GDN checkpoint | = I9 | = I9 | shared (e2e) |
| B10 | serving glue | token stream | e2e A2 + token diff | **none** (no bf16 battery; bf16 is the *anchor*, not a gated variant) |

Note: BF16 is the reference anchor for every other dtype (the `regress_unified`
bf16 anchor). B1–B6 are what make "token diff vs bf16 ref" meaningful; today the
anchor is unverified by a chain.

## 5. Phase decomposition — prefill (materialize-free; direct FA route)

Flash is fused (fewer boundaries than decode), same as docs/128 §4. Tile-dump
hooks are env-gated inside the FA kernels (M3).

**I8 prefill** (`src/ops/launcher/gqa_attention_prefill.cu` :29-42):

| # | Phase | Artifact | Oracle | Today |
|---|-------|----------|--------|-------|
| I-P1 | fill/quantize (`gqa_attention_prefill_fill_i8_page_kernel`; TWO paths: tiled `tokens>=128 && KVHeads==2` and scalar — both must be gated) | codes + scales | CPU quantize ref (byte) | none |
| I-P2 | prefill attention (`gqa_attention_prefill_i8_kernel`) | O tiles | CPU/FP64 ref | none (op A3 criteria only) |
| I-P3 | launch/envelope | selected config | table validation | none |
| I-P4 | e2e | tokens | token diff vs bf16 ref | full (int8 T battery) |

**BF16 prefill** (:43-58, `gqa_attention_prefill_bf16_kernel`):

| # | Phase | Artifact | Oracle | Today |
|---|-------|----------|--------|-------|
| B-P1 | fill (bf16 copy) | page bytes | byte-identity | none |
| B-P2 | prefill attention | O tiles | FP64 ref | partial (op A3 criteria) |
| B-P3 | launch/envelope — **the crash config must be fixed or pinned**: define the valid (ctx, SM-count) envelope for 36-SM cards; table validation gate | selected config + envelope map | table validation | none |
| B-P4 | e2e | tokens | token diff | partial (no CI bf16 battery) |

## 6. MTP phase gate (explicit scope)

**State of the art:** the KVarN MTP phase gate = `tools/bench/phase_gate_mb.{sh,py}`
(B1/B4/B5 vs R1/R4/R5 round dumps: round inputs, accept decision, converged
round state; converged-state comparator keyed on (gen, F, anchor); exit 0/20/30;
rank-gated dumps) + the docs/131 T0–T3 suite. The **multi-lane batched runner is
KVarN-only**: `is_kvarn_kv` gates the lane-workspace binding
(`src/runtime/tp2/tp2_backend.cpp` :368/:501; `set_kvarn_batched_active` :2450).
The **single-sequence MTP verify path is dtype-agnostic**
(`src/ops/kernel/speculative_round.cu`: `prepare_verify_inputs`,
`accept_greedy_drafts`, `select_accepted_hidden`; MTP-layer KV = `mtp_cache()`
paged cache, same `kv_dtype`, `decoder_state.cpp:94-98`) — the kv_i8 verify
cells prove i8+MTP runs in production.

**Scope (M4, after M1):**

1. **MTP round gate for i8 and bf16.** Same B1/B4/B5 dump points, env-gated in
   the single-sequence MTP round path (`NINFER_MB_PHASEGATE` pattern), run with
   `KV_DTYPE=int8` and `KV_DTYPE=bf16`, compared against the **isolated
   fresh-process single-sequence reference** (docs/130 §6: in-process references
   are contaminated — never use them). `phase_gate_mb.py` is already dtype-
   agnostic (reads B*/R* bins) — reuse verbatim.
2. **Acceptance:** (a) green on the current build, both dtypes; (b) **negative
   test**: a 1-line mutation in the accept path (e.g. forced `lic−1`) FAILs at
   the accept phase with exit 20-class signature; (c) pinned geometry set covers
   the historical MTP degenerate cases (T=256 terminal ext=0, page-boundary
   rounds, docs/130 §8).
3. **NOT in scope (separate feature, scope separately if wanted):** a
   multi-lane batched runner for i8/bf16 (generalizing `kvarn_lane_ws`). That
   is a feature (multi-batch MTP on i8/bf16), not a gate; it would re-open the
   entire docs/131 battery surface (stride, B-shrink, hydrate) for two more
   dtypes.

## 7. Audit of existing gates (what is gated today; "the same bug")

- **7.1 I8 decode — CLONE GATE (the bug the owner suspects; confirmed).**
  `ninfer_slice3_i8_test` (`tests/slice3_i8_test.cu`) runs the FULL case battery
  (T=1/5/6, APPEND, MULTI-BATCH, split policy, FP64 oracle) on
  `gqa_decode_slice3_i8_kernel` — the **shipped** A/B route (:303/:319). The
  production default is **unified v2** (`gqa_decode_slice3_i8_v2_kernel`),
  covered only by the T=1 `v2_discriminator` bit-identity check (:639+).
  **A regression in the production i8 kernel or its launcher passes the unit
  gate green** — the exact docs/132 §2.1 failure mode. Additionally the test
  launches kernels directly: the I8-specific split policy
  (`gqa_small_t_split_count`) and capacity/envelope validation are unexercised.
- **7.2 I8 dequant:** `ninfer_slice3_i8_dequant_test` is a host **mirror** of the
  prologue math ("written in advance") — it does not call production code.
- **7.3 I8 QK CPU refs:** `slice3_i8_s8qk_cpu_ref.cpp` references a FUTURE s8-QK
  port (not the current bf16-body production path); `kvarn_int8_qk_cpu_ref.cpp`
  is KVarN lever-1. No dormant production-route QK ref exists for i8 — M1 builds
  it (pattern: `tests/kvarn_decode_cpu_ref.h`).
- **7.4 BF16 decode:** only the T=1 byte-identity transcription check
  (`slice2_byteid_test.cu`) + op-level criteria (`tests/ops/test_gqa_attention.cpp`,
  bf16 rel_l2 2.8e-3 / i8 3.15e-3, A1/A3 profiles, mapping patterns, codec
  edges). No phase chain, no long window, no multi-split, no MTP shape.
- **7.5 BF16 prefill:** the envelope crash (docs/127 §2) is the only evidence;
  no launch-table validation gate (the docs/128 P6 gap, same form).
- **7.6 MTP:** i8 = e2e only (kv_i8 verify cells + int8 T cells with
  `SPEC=mtp`); **bf16 = no CI coverage at all** (correctness fast gate defaults
  `KV_DTYPE=int8`; `--full` has no bf16 battery).
- **7.7 CI ratchet:** `tools/bench/decode_guard_baseline.json` **already contains
  `mtp_on_int8_*` and `mtp_on_bf16_*` cells**, but `decode_guard_cells_ci.sh`
  gates only `official_greedy:mtp_on_kvarn_k4v2_<ctx>` (`BASE_DTYPE=kvarn_k4v2`).
  The i8/bf16 ratchet cells exist and are ungated — wiring is a one-env-var
  change per dtype (M2).

## 8. Milestones (agent-ready, with acceptance criteria)

**What exists in-tree to reuse (do not rebuild):**
- Phase-gate skeleton + artifact I/O + pinned inputs: wo-phase-gate
  `tests/phase_gate.cu` (`PinnedInput::make(seed=1, ntiles_k, ntiles_v)`,
  `write_phase()`, `PHS1` header, `--mutate-*` pattern) — **reuse verbatim; do
  not change `PinnedInput` output (sha ratchets break)**.
- Production in-kernel dump pattern: `build_slice4_dump.sh` +
  `tests/phase_gate_slice4_compare.cpp` (null-checked hooks in the production
  kernel, comparator + tolerance JSON).
- MTP comparator: `tools/bench/phase_gate_mb.py` (dtype-agnostic).
- CPU ref pattern: `tests/kvarn_decode_cpu_ref.h` (FP64 online-softmax + split-K
  merge emulation) — the canonical bf16 body is shared, so I3–I6 and B3–B6 refs
  are near-duplicates of it.
- Op-level criteria (keep, do not replace): `tests/ops/test_gqa_attention.cpp`.

**M0 — i8 skeleton + I1/I2 + production-route fix (≈1 day).**
Dump hooks for I1/I2 (append kernel + v2 prologue) + test binary
(`ninfer_phase_gate_i8b`, pattern `kvarn_materialize_oracle_test.cu`) + gate
script with chain table. **Includes the §7.1 fix:** `slice3_i8_test` battery
targets the production route (unified v2 via `launch_tc_partial_unified_i8`,
route banner in the test log); §7.2 dequant test calls the production prologue
helper, not a mirror.
*Accept:* (a) known-good build: I1/I2 PASS (byte), chain <10 s; (b) **negative
test**: scale ×(1+1e-3) mutation in the append path FAILS at I1; (c) a mutation
in the v2 kernel FAILs the unit battery (proves 7.1 fixed).

**M1 — full decode chain I3–I7 + B3–B7 (≈3–4 days; long pole = CPU refs).**
Dump hooks in `gqa_decode_slice3_i8_v2_kernel` /
`gqa_attention_small_t_tc_partial_bf16_kernel` + shared reduce
(null-checked, zero-overhead when off) OR direct launcher-driven dumps (the
§2.1 production-route requirement: the test must run the launcher's split
policy, capacity check, and reduce, not a re-launch). B1 byte-identity
(commit/append). I7/B7 table validation (pure table check, no GPU).
*Accept:* (a) known-good build: I1–I7 and B1–B7 PASS; (b) **historical bug**:
replay of the docs/104 long-ctx I8 defect (or a dequant-prologue mutation)
FAILS at I2/I3; (c) chain <10 s per dtype.

**M2 — CI wiring + tolerance tables + bf16 battery (≈1–2 days).**
Fast CI: `ninfer_phase_gate_i8b` both dtypes after the slice4 block (~2–4 min).
Full CI: mutation battery per dtype; MTP round gate (M4 when ready); **bf16
correctness battery** (`KV_DTYPE=bf16`, T1–T11 — new); **ratchet cells**:
`decode_guard_cells_ci.sh` runs `BASE_DTYPE=int8` and `BASE_DTYPE=bf16` passes
(baseline cells already in JSON; CI still never ratchets — owner decision only).
Tolerance table `tools/bench/phase_tolerance_i8bf16.json` (committed, reviewed
like code).
*Accept:* a broken I5 build fails fast CI with the table pointing at I5; a
broken bf16 reduce fails at B6; a bf16 token divergence shows in the new
battery.

**M3 — prefill chains I-P1..P4 / B-P1..P4 + envelope gate (separate, later).**
Tile-dump hooks in `gqa_attention_prefill_{i8,bf16}_kernel`; both i8 fill paths
(tiled + scalar) gated. B-P3: define + pin the valid bf16 (ctx, SM) envelope for
36-SM cards; the 40360 crash config is either fixed or a documented carve-out
with the pinned valid range.

**M4 — MTP round gate, i8 + bf16 (≈2–3 days; after M1).** Per §6.
*Accept:* §6 acceptance (a)–(c).

**Total to M2+M4 ≈ 2 weeks** (M3 separate). Split option: lane A = i8
(M0/M1/M2), lane B = bf16 + MTP (M1-B/B3–B7, M2, M4) — shared files
(`ninfer_phase_gate_i8b`, tolerance JSON, CI block) are serialized by a single
writer per commit.

## 9. Cost / benefit / risk

- **Benefit:** any future i8/bf16 kernel regression localizes in one CI run
  (~2–4 min) instead of ~1 day of A/B builds; makes the docs/104-style I8 work
  and any bf16 envelope/route change safe; makes the bf16 *anchor* itself
  verified (it is currently assumed correct by construction).
- **Cost:** ~2 weeks to M2+M4; ~2–4 min fast CI; one-time bf16 battery ~15 min
  in full CI.
- **Risk (a) accumulation order** (docs/128 risk a): I3–I6/B3–B6 share the
  canonical bf16 body with the KVarN/bf16 refs — reuse, don't re-derive; if a
  phase "almost matches", it is an order problem, not a tolerance problem.
- **Risk (b) dump overhead** must be exactly zero when off (null-checked hooks,
  docs/128 §8.1).
- **Risk (c) pinned inputs**: reuse `PinnedInput` verbatim (hash ratchets).
- **Risk (d) two i8 fill paths / two i8 decode routes**: the gate covers the
  DEFAULT route (unified v2); `NINFER_I8_DECODE=shipped` stays as rollback and
  is covered by the existing (now corrected) battery only if explicitly
  selected — a route flip flips the baseline with a documented byte-diff
  record, never silently.
- **Non-goal:** does not replace perf gates (decode guard ratchet stays
  separate — the i8/bf16 cells just get wired as gated, M2).

## 10. Execution spec (for the first agent)

- **Worktree/branch:** new `wo/phase-gate-i8` from `wo/kv-uniform` HEAD (or the
  current integration branch at start). Own `build/` — never build into
  another agent's build dir. Test hooks land in `src/` on this branch as one
  reviewable unit (null-checked, zero-overhead).
- **Trigger interface:** a test binary, NOT the server
  (`build/tests/ninfer_phase_gate_i8b`; CMake target like the existing oracle
  tests; link `ninfer_ops` + `ninfer_artifact` — the docs/132 §2.3 trap).
  ```
  --kv-dtype i8|bf16 [--full] [--mutate-i1 ... --mutate-b7]
  prefill(pinned prompt)          → dump I1 (codes+scales) / B1 (pages)
  1 MTP round (k=3)               → dump I2–I6 / B3–B6 (or B1–B6 for bf16)
  --full: N accept rounds (I8/B8), GDN hash (I9/B9), A2 + token diff (I10/B10)
  ```
- **Pinned inputs:** `PinnedInput::make(seed, ntiles_k, ntiles_v)` from
  wo-phase-gate `tests/phase_gate.cu`, default seed=1, sha over k_bf‖v_bf. The
  i8 chain STARTS at bf16 K/V and runs the PRODUCTION append kernel (that is
  I1 — do not pre-quantize on the CPU).
- **Artifact format:** `PHS1` (u32 magic | u32 version | u16 dtype_code |
  u64 prompt_sha256_hi | u32 elem_count | raw bytes; row-major, layout
  documented in §3/§4 table columns) — one file per phase; byte phases gate as
  sha equality.
- **Tolerance calibration:** byte phases I1/I2/B1: tolerance = 0 (hash
  equality — the fp16 scale + RNE + clamp are exact on CPU); fp phases:
  max(10×observed, 1e-5) pinned in `phase_tolerance_i8bf16.json` once on the
  known-good build.
- **Exit codes:** 0 = clean; N = first failing phase index; 77 = no GPU;
  90 = sha self-test (same contract as docs/128 §8.5).
- **Day 1 (checklist):**
  1. Read: 128 (full), 129, 130 §6/§8, 131, 132 (esp. §2.1), 133, 134 §10
     (pinned-input resolution), 127 §2 (bf16 envelope), this doc.
  2. Run the existing unit battery green: `ninfer_slice3_i8_test`,
     `ninfer_slice3_i8_dequant_test`, `ninfer_gqa_attention_test`,
     `ninfer_slice2_byteid_test`.
  3. Confirm the route banner: `[i8-decode-route] unified` is the default
     (grep the launcher; do not assume).
  4. Build M0 + negative tests. Do NOT start M1 until M0's negative test
     (mutated quantize → FAIL at I1) passes.

## 11. Gate policy (the aggressive part)

- **Lockstep (binding, §2):** impacted i8/bf16 phase ⇒ its gate updated in the
  SAME commit, byte-diff A/B record when the artifact intentionally diverges.
- Any commit touching `src/ops/kernel/gqa_decode_slice3_i8*.cuh`,
  `gqa_attention_decode_i8.cuh`, `gqa_attention_decode_bf16.cuh`,
  `gqa_attention_prefill_{i8,bf16}.cuh`, `gqa_attention_kv_quant.cuh`,
  `gqa_decode_body.cuh`, `gqa_decode_unified*.cuh`, or the
  decode/prefill launchers: run the affected chain locally before push; CI
  runs the full chain.
- New KV dtype variants follow docs/127 §4: phase oracles + guard cells land
  with the dtype, before merge.
- A byte-identity carve-out (tile change → accumulation order change) requires
  an updated tolerance entry + a Tier-3-style byte-diff A/B record
  (docs/126). Carve-outs documented, never silent.

## 12. CI wiring (concrete)

- `run_ci.sh` lives on the CI-only branch (`wo/kv-uniform-ci-gate`,
  docs/126 §0) — wiring lands there, syncs to `wo/kv-uniform` the same way the
  decode-guard-cells step landed.
- **Fast (per build):** after the slice4 block — `ninfer_phase_gate_i8b`
  (both dtypes) + comparator, ~2–4 min.
- **Full:** mutation battery per dtype (I1/I2/I3–I7, B1/B3–B7); MTP round gate
  i8+bf16 (M4); bf16 correctness battery `KV_DTYPE=bf16` T1–T11; ratchet cells
  `BASE_DTYPE=int8` + `BASE_DTYPE=bf16` (existing baseline cells).
- GPU/port: test binary only (no server, no port) until M3/M4's prefill+MTP
  legs; then check `ss -tln` + `nvidia-smi` and coordinate as usual.

## 13. References

Docs (all in `docs/` of this worktree):
- **128** — the template (chained oracles, lockstep, milestones, execution spec)
- **129/130** — M0/M2 handoffs; 130 §6 (reference contamination), §8 (MTP bugs)
- **131** — MTP multi-batch test suite (the MTP gate's companion spec)
- **132** — review: clone-gate rejection (§2.1), CMake link trap (§2.3)
- **133/134** — resolution + baseline; pinned-input resolution (§10)
- **127** — gate status (bf16 envelope crash §2), KV-dtype variant template §4
- **126** — verification runbook (Tiers 1–3, CI-only branch convention)
- **104** — long-ctx I8 fix history + I8 split policy provenance
- **113** — route-vs-route (A3 harness); **106/114** — launch tables (I7/B7)
- **135/136** — kv-mtpfix debriefs (MTP pool edge cases; geometry context)

In-tree code (verified pointers; reuse patterns, don't reinvent):
- quantize/dequant: `src/ops/kernel/gqa_attention_kv_quant.cuh`
  (`gqa_kv_quant_code`, `gqa_kv_dequant_i8x8_from`)
- i8 decode: `src/ops/launcher/gqa_attention_decode.cu`
  (`i8_decode_route` :29, `launch_tc_partial_i8` :144,
  `launch_tc_partial_unified_i8` :431, `gqa_small_t_split_count` :76),
  `src/ops/kernel/gqa_decode_slice3_i8_v2.cuh` (production),
  `gqa_decode_slice3_i8.cuh`, `gqa_attention_decode_i8.cuh`
- bf16 decode: `src/ops/kernel/gqa_attention_decode_bf16.cuh`
  (`gqa_attention_small_t_tc_partial_bf16_kernel`)
- prefill: `src/ops/launcher/gqa_attention_prefill.cu`,
  `gqa_attention_prefill_i8.cuh`, `gqa_attention_prefill_bf16.cuh`
- MTP: `src/ops/kernel/speculative_round.cuh`;
  `src/runtime/tp2/tp2_backend.cpp` (batched KVarN-only gates :368/:501,
  `set_kvarn_batched_active` :2450);
  `src/targets/qwen3_6/impl/state/decoder_state.cpp` :94-98 (mtp_cache dtype)
- existing tests: `tests/slice3_i8_test.cu` (§7.1),
  `slice3_i8_dequant_test.cpp` (§7.2), `slice3_i8_s8qk_cpu_ref.cpp`,
  `kvarn_int8_qk_cpu_ref.cpp`, `slice2_byteid_test.cu`,
  `ops/test_gqa_attention.cpp`, `bench_int8_vs_bf16_mma.cu`,
  `slice3_i8_bench.cu`
- phase-gate infra (wo-phase-gate): `tests/phase_gate.cu` (`PinnedInput`,
  `write_phase`), `tests/kvarn_decode_cpu_ref.h`, `tests/phase_gate.cu`
  mutation pattern; `tools/bench/phase_gate.sh`, `phase_baseline.json`,
  `phase_tolerance.json`, `build_slice4_dump.sh`,
  `tests/phase_gate_slice4_compare.cpp`, `phase_gate_mb.{sh,py}`
- CI: `wo-kv-uniform-ci-gate: tools/ops/run_ci.sh`,
  `tools/bench/decode_guard_cells_ci.sh`, `tools/bench/decode_guard_baseline.json`
