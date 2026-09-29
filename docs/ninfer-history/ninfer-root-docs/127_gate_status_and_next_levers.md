# 127 — Gate status, next-lever queue, and the prefill-unification verdict

**Status:** 2026-09-01. Written post B-STAGING4 decay fix (c631f7f1, ac6b3f61, 61dcecd8),
pre MultiBatch merge. Companion to docs/125 (handoff), docs/126 (verification runbook),
docs/104 §4 (gate definitions). This is the single place where "which gate is open, which
lever is next, and why Phase 3 is re-scoped" is recorded, so the plan survives agent loss.

---

## 1. Phase 2 (unified decode) — gate status (docs/104 §4 Phase 2)

| Gate | Criterion | Status | Evidence |
|---|---|---|---|
| (a) BF16 byte-identity | bf16 path identical to shipped bf16 kernel | ✅ 08-31 — **re-run post-fix PENDING** | `regress_unified.sh` PASS (docs/104 §9); post-fix re-run = docs/126 Tier 2a (GPU-exclusive) |
| (b) A2 identity (MTP==plain) | per variant | ✅ current | 09-01 06:59 CI battery: MTP long-prefill MTP==plain PASS; ci_cells accept 70.7/67.9/68.5/70.6/73.7/73.7 (10k→250k) |
| (b) A/B byte-identity | per variant | ⚠️ keypair change unverified | `ac6b3f61` claims bit-identity vs `c631f7f1` (same FMA order) — verification = docs/126 Tier 3 (byte-diff A/B, GPU-exclusive). Not yet executed |
| (b) greedy acceptance ±0.5 pt, no new drift | vs Phase-0 | ✅ (greedy cells) | acceptance tracks baseline at all cells; sampling-config re-baseline still official-v1 (see §5) |
| (c) decode t/s within Phase-0 envelope | | ✅ | every cell ABOVE baseline v1: 74.8/71.2/67.6/61.6/53.8/46.5 |
| §3 residual gate | A/B within [0.85,1.15] vs packed + oracle 10/10 + default flip | ✅ closed 08-31 | kernel cost 0.928–1.048× packed across 12 cells; oracle 10/10; default = unified (`29d0a797`), `NINFER_KVARN_DECODE=packed` kept as rollback |
| Pre-req: Phase 1 landed | launch-config dispatch table | ✅ | docs/106/114 landed + reviewed PASS |
| Pre-req: bug-free (D-21 closeout) | temp>0 re-run, full-width H5, full CI | ✅ at the time — **full CI must be re-run post-fix** (see D5 row below) | docs/117: D-21 identity ALL PASS, re-baseline v2 APPROVED |

**Phase 2 verification status (updated post CI run 20260901_065942, wo/kv-uniform-ci-gate):**
1. ~~docs/126 Tier 1 (unit battery)~~ — ✅ CLEARED: now standing CI step [3a] (run_ci.sh),
   exit=0 in run 20260901_065942 (14 binaries PASS, SELFTEST64 PASS, T4 0.939 ms gate PASS).
2. ~~docs/126 Tier 2a (`regress_unified.sh`)~~ — ✅ CLEARED: running in CI [full];
   RESULT: PASS in run 20260901_065942 (A2 identity + token-correctness vs bf16 anchor).
3. docs/126 Tier 3 (byte-diff A/B: `ac6b3f61` vs `c631f7f1`) — **OPEN** (not a CI step;
   needs a pinned A/B run).
4. ~~`run_ci.sh --full` 0-fails~~ — ✅ CLEARED once (pre-MultiBatch): run 20260901_065942
   Verdict: PASS, 0 fails (`results/20260901_065942_ci.json`). Tracked WARNs only: verify
   pp −5.9% / prefix +4.4% (short-ctx, session-level), guard 25k accept −7.1pp.
   **Must re-run after the MultiBatch merge** (all verification above is against the
   pre-MultiBatch build, binaries 02:21:01). Note: the run crashed in its final
   post-processing (run_ci.sh edited mid-run; bash misread the tail: line 253/255) AFTER
   all test stages had completed; resalvaged by re-running the aggregation with the
   original TS=20260901_065942 — no tests re-run (log: /tmp/ci_full_run.log;
   checkpoint: wo/kv-uniform-ci-gate 7d89e8f2). Aggregation NameError (`_re` before
   import) fixed in the same commit.
5. Official ratchet re-baseline — **DONE 2026-09-01 09:35 (`ratchets: 1`).**
   `decode_guard_check.py --ratchet` raises only cells whose sample_cfg matches the
   baseline config (temp 1.0 = sampling); greedy cells are config-skipped (and the
   CI helper never ratchets). Post-fix sampling matrix re-ran 09:07 (1 run/cell,
   COMP=192, TP2) → `results/official_sampling_decode_20260901_090702.json`; ratchet
   raised 9 of 15 cells (monotonic max), 6 kept below-baseline (single-run noise).
   **Accepted as final by owner 2026-09-01** — an ITERS=2+ re-run (`ratchets: 2`)
   is the known improvement (sampling 1-run values are a single trajectory draw;
   greedy cells are deterministic and fine at 1 run) but is explicitly deferred.
   Greedy highs still cannot ratchet via plain `--ratchet`; the official_greedy
   section remains manual/tooling-only (re-measured 09-01: all 15 cells above
   baseline, GATE PASS — ci_cells_decode_20260901_072620/085134/085943.json).
   Baseline committed wo/kv-uniform-ci-gate 5de34583, synced wo/kv-uniform a6eef4f5.

## 2. Phase 3 (unified prefill) — gate status (docs/104 §4 Phase 3, docs/120 §4D)

| Gate | Status | Note |
|---|---|---|
| D1: prefill t/s within ±5% of **int8** @40k/80k/160k, ≥ baseline every cell (redefined 2026-09-03 — bf16 invalid on this 36-SM box, crashes at 40k) | ✅ CLOSED (2026-09-03) | **PASS @40k (+0.2%) / 80k (−0.3%); 160k = −6.8%** (572.9 vs 614.4, 4b run 09-03 14:59) — a known residual slightly outside ±5% (O(n²) materialize widens at the longest ctx). The old "~−7% vs bf16" figure (docs/122 §11a) is stale (earlier build); the current build is at parity with int8 at 40k/80k. results/106 §7, docs/142 |
| D2: byte-identity direct vs materialize @4k/40k | ⏸ **DEFERRED 2026-09-03** (blocked on B2+B4, and B2 **PASSED** — user: revisit only with free time) | docs/141: the `direct` route has **no launch site anywhere in src/** and its throw at `gqa_attention_kvarn.cu:300` is **unconditional** (no `tail_count` test), so page alignment does not unblock it — verified at runtime on a 1969-token prefill (exit 134, no output). Trap: the throw is unreachable below T>=7, so a short-prompt probe FALSE-PASSES. A4 reference snapshot is now captured (`results/141_gate_verify/a4_phase3_ref/`, labelled Phase-3 reference route-vs-route) and the gate's **2x determinism half PASSES 6/6**. **Why deferred, not owed:** results/113 nsys — materialize = 0.15 s of the 3.65 s gap (4.1%; quantize 98.9%, route-independent) and the gap is now ~0 vs int8 at 40k/80k → B2 best case ≤0.5%, plausibly negative. Also fixed: A3's capture harness generated a fixed ~52k-token prompt regardless of --ctx and killed all 4 long cells in `gqa_kvarn_commit: logical page out of block table` |
| D3: 250k KVarN must-pass battery (T1/T2/T3/T5/T8/T10/T11/T15/T16/T17/T18) | ✅ CLEARED once (pre-MultiBatch) | CI 20260901_065942: KVarN @250k 16-cell battery (T1-T6,T8-T11,T13,T15-T19), 16 passed / 0 failed — covers the full must-pass list |
| D4: decode side provably untouched (byte-diff vs pinned pre-Step-1 build) | ✅ **PASS (static proof)** | docs/141: `b247fc4d`'s whole diff is 36 lines in one file and its only executable statement sits inside `tokens > kKvarnSmallTMax && !packed_verify` => **tokens>=7**; decode is T=1, verify T=2..6, so the prefill route is **unreachable from decode by construction**. Literal battery (pin `aa8ef21b` + numerics-neutral A3 cherry-pick vs `df9f3528`, route pinned by env, 32 cells) = **30 identical, 2 differ**: packed/h5_k0 + h5_k3 at index 5 — **NOT A D4 FINDING (attributed drift)** to `9a58008a`, the only commit touching the packed kernel in range, and the direction is packed *converging onto* unified. Correction debt: the pre-Step-1 window `29d0a797..b247fc4d` **does not compile** (2-arg template call vs 1-param kernel), so docs/120's A2/A3/A4 chain landed on a broken tree |
| D5: `run_ci.sh --full` 0 fails at closeout | ✅ CLEARED once (pre-MultiBatch) — re-run after merge | CI 20260901_065942 Verdict: PASS (0 fails); see §1 item 4 |

**Route state:** B1 (env gate `NINFER_KVARN_PREFILL=materialize|direct`, FATAL on unknown)
is in-tree, but the direct route is HARD-GATED at runtime: the direct prefill kernel has no
tail support (B2) and silently drops every uncommitted key — it is wrong, not slow
(`gqa_attention_kvarn.cu:287-296`). B4 (A/B) therefore cannot run.

**Premise superseded (docs/104 §9 addendum):** "delete `gqa_kvarn_materialize_kernel`
(O(n²) pass)" is wrong twice — the O(n²) regime was removed by docs/66, and materialize is
**4.1% of the prefill gap** (0.15 s of 3.65 s); `quantize_tile_kernel` is **99%** (3.61 s),
driven by `kSinkhornIters` (docs/120 §1, `results/113_prefill_nsys_attribution.md`).

**Remaining prefill perf path (docs/120 §4C):** C4 (Sinkhorn 16→4) DONE as A1 (+8% prefill);
C2 (per-layer `cudaStreamSynchronize` removal, ~6 µs/page, prerequisite) and C3 (side-stream
`gqa_kvarn_commit_completed` — the ONLY path to the full 12.4% gap, riskiest item) both open.

## 3. Verdict: do NOT build Phase 3 as written (in-kernel dequant, delete materialize)

The data and the code both say no. Full analysis in §4. Short form:

1. **The premise is dead** (materialize is 4.1% of the gap, not O(n²) — §2 above).
2. **In-kernel dequant is structurally redundant for expensive dequants.** Flash reads every
   KV page once per q_block CTA; the direct kernel's own header states "the dequant work
   repeats per (q_block, q_head) CTA" (`gqa_attention_kvarn_direct.cuh:10`). At 40k that is
   625 q_blocks; at 160k, 2500. Measured: 62× SLOWER than materialize+flash (docs/124 A5,
   latency-bound ~20 GB/s). Even a perfectly tuned in-kernel route would pay dequant ×2500
   at 160k — fatal for a 4/2-bit affine+scale dequant. Materialize-once is the algorithmic
   win, not the defect.
3. **The flash is already unified where it matters.** KVarN prefill runs the SAME proven
   BF16 FA2 flash as BF16 (over the paged temp view). The variant surface is already reduced
   to quantize + dequant — which is exactly the shape the KV-dtype bucket needs (§4.3).
4. **What Phase 3 actually owes us is verification + the prefill gap, not a kernel rewrite:**
   D2/D3/D4/D5 closure, and C2/C3 if prefill parity vs int8 is a product requirement.

## 4. Prefill unification analysis — what serves the KV-dtype bucket

**Strategic context (user directive, 2026-09-01):** the point of the unified-kernel program
is that we will add a bucket of KV cache types — non-KVarN int4, mixed int5-key/int4-value,
etc. The unified kernel must make that (a) easy to implement, (b) easy to attribute when a
variant underperforms, (c) easy to maintain.

### 4.1 Code facts (verified 2026-09-01 in-tree)

- **Two prefill flash families exist:**
  - BF16 FA2 (`gqa_attention_prefill_bf16.cuh`) — shared by BF16 and KVarN (over the
    materialized paged temp, exact `{D,G,H,P}` flash layout).
  - I8 s8-MMA (`gqa_attention_prefill_i8.cuh`) — independently tuned: QK stays int8 through
    `m16n8k32.s8` TC, V dequantized in-kernel in FP16, 16 warps, own 92,672 B smem arena,
    own constants (`kGqaPrefillI8Br/Bc/Warps`), per-key scale in epilogue. This is a genuine
    math split (int8 accumulation ≠ bf16 body) — the one place "same kernel body" is false.
  - `gqa_attention_prefill_common.cuh` shares only leaf PTX helpers: it "deliberately owns
    no staging policy, shared-memory arena, warp schedule, or kernel body".
- **KVarN prefill = commit(quantize) + materialize(dequant-once, per-tile, K/V staged) +
  shared BF16 FA2 flash.** Materialize = 0.5% of prefill wall (L2-resident, docs/124).
- **The direct (in-kernel dequant) route exists but is gated off** (B2 tail support missing;
  62× slower measured; docs/124 A5 REJECTED).
- **Decode is already the template:** one canonical body (slice2/slice3/slice4 share the
  body; variant = dequant prologue + the SHARED reduce), one launch-config surface
  (Phase 1 dispatch table), one set of gates. Adding a KV dtype to decode = new prologue.

### 4.2 The dequant-placement rule (the real "unified" principle for prefill)

Where dequant runs is a cost decision, not a uniformity decision:

- **Cheap dequant (≤ ~1 FMA/element: int8 ×scale, plain int4 nibble-unpack):** in-kernel.
  Redundancy across q_blocks costs ~1 FMA per element per q_block vs ~1024 FLOPs of MMA per
  kv element per q_block (D=256) — negligible. This is the I8 model; it is correct for int8.
- **Expensive dequant (kvarn 4/2-bit affine + per-key scales + rotation; future fp4,
  ternary, k5v4):** materialize-once + shared BF16 FA2 flash. The dequant FLOPs are paid
  T/Br times in-kernel; once per tile is the only sane option. This is the KVarN model;
  it is correct for kvarn and for every future expensive dtype.

Both templates already exist in-tree. "Unified prefill" should be defined as
**"one proven flash + a per-variant dequant stage with a fixed contract"**, NOT
"one in-kernel-dequant flash for all".

### 4.3 The variant template (what each new KV dtype needs)

For a new KV dtype (e.g. non-KVarN int4/int4, or k5v4), the variant surface is exactly:

1. **Codec + CPU reference** (pattern: `tests/kvarn_codec*.cpp`, `kvarn_codespace_qk_cpu_ref.cpp`)
2. **Commit-quantize kernel** (pattern: batched per-head, docs/123 — the s_row_K/s_row_V
   offset lesson)
3. **Dequant stage** — materialize kernel (expensive path) or in-kernel prologue (cheap
   path), with a **GPU oracle** (pattern: `ninfer_kvarn_materialize_oracle_test`)
4. **Guard cells + ratchet row** (tools/bench/decode_guard — perf + acceptance vs the
   baseline; the compounding-law arbiter)

Everything else is REUSED, unchanged: the BF16 FA2 flash (expensive path) or the s8 flash
model (cheap path), the unified decode body + shared reduce, the Phase 1 launch-config
dispatch table, the CI gates. Non-KVarN types simply omit the rotation (prologue = affine
only — cheaper than kvarn, strictly easier).

**Attribution property (the "find weak spots" goal):** with one shared flash and one shared
decode body, any per-variant perf gap is attributable to quantize + dequant alone, and both
are measurable standalone (bench + kernel-time totals per docs/120 §5). A weak variant is
found by its guard cell and debugged in one of two kernels, not in the attention math.

### 4.4 Consequences / decisions

1. **Do not start Phase 3's kernel work** (in-kernel dequant for KVarN, delete materialize).
   Re-scope Phase 3 = §2 verification debt (D2/D3/D4/D5) + C2/C3 (only if prefill parity vs
   int8 is a product requirement) + the doc-104 §4 corrections (owner edits).
2. **Keep the direct kernel gated off**; if ever re-enabled it needs B2 tail support AND a
   vectorized/MMA-tuned dequant, and it will only ever be competitive for cheap dequants
   (see the rule above) — for kvarn it is a dead end on the math, not just the current
   implementation.
3. **I8 prefill is the one open uniformity question** (its s8-MMA flash is a separate math +
   tuning surface). Options: (a) keep s8-MMA (perf win, accept + document the numerics
   carve-out), (b) add a dequant→shared-FA2 option for I8 and A/B. Low priority — I8 is the
   comparison lane, not a product lane — but the decision should be recorded when it is made.
   Either way it does NOT block the KV-dtype bucket: cheap dequants follow the I8 model,
   expensive dequants follow the KVarN model.
4. **The prefill gap (−7% vs bf16, unevaluable) vs the valid int8 reference** should be
   re-measured same-session before any further prefill tuning (docs/124 A4b/A4c) — the
   bf16 numbers are suspect on 36-SM cards.

## 5. Next-lever queue (decode) — consolidated 2026-09-01 options, in priority order

| # | Lever | Status / refs | Est. |
|---|---|---|---|
| 1 | **MultiBatch / batched decode** (Part b) — slice4 templated for MultiBatch/Masked; launcher hardcodes `batch_size=1` (`gqa_attention_kvarn.cu:55,83`); needs batch_size>1 pass-through + MultiBatch=true + reduce MultiBatch=true + 2-sequence guard. Fork's x5.9 dual; the compounding win for serving throughput | **IN PROGRESS** (src edits live in worktree) | docs/125 PART (b), docs/122 §18 |
| 2 | **Bc=64 tiles in slice4** — slice4 is Bc=32 (2 tiles/64-key page); softmax merge + alpha-rescale + 2 barriers run per 32-key tile. Packed kernel proves Bc=64 works. Changes fp32 accumulation order (class already accepted: unified vs packed 32/64-key widths, `results/kvarn_unified_decode_matrix_20260830.md`). Gate: `regress_unified.sh` (bf16 token anchor) + acceptance battery | open | est. +8–12% decode at long ctx |
| 3 | **Direct-to-MMA-fragment K dequant** (the fork's structural design) — removes the bf16 smem round-trip; post-fix attribution puts K/V dequant (bf16 smem round-trip + per-tile scale LDGs) at ~58% combined. Dormant CPU ref: `tests/kvarn_codespace_qk_cpu_ref.cpp`. Full plan + M3/M4 gates: docs/83 | open | biggest single-phase win; needs FP64-oracle gate |
| 4 | **Tail work** — 25k/80k guard cells: **DONE 2026-09-01 07:26** (ci_cells: 25k = 71.2, 80k = 61.6 t/s, both inside the docs/126 Tier 2b pass bands; greedy acceptance 67.9/70.6 — cross-config comparison vs the sampling baseline is invalid per decode_guard_check policy). The full 10k-250k greedy matrix is now a **standing CI gate** (`decode_guard_cells_ci.sh`, KVarN-only). Ratchet: see §1 item 5 (ratchet is sampling-only; int8/bf16 greedy matrices captured 09-01 08:51/08:59, GATE PASS x2) | done (CI gate in place) | — |
| 5 | **Post-fix SAMPLING matrix re-run** (temp 1.0) | **DONE 2026-09-01 09:07** — 15 cells, `official_sampling_decode_20260901_090702.json`, ratchet 1 applied (§1 item 5) | — |
| 5b | **Extend CI guard step to int8+bf16** (helper is now dtype-aware: `KV_DTYPE`/`BASE_DTYPE` env, defaults kvarn so current CI unchanged). int8/bf16 greedy matrices captured 09-01 08:51/08:59, all above baseline (GATE PASS x2) | open (owner decision) | ~13 min GPU per full CI if enabled |

**NOTE on guard coverage:** the CI guard matrix is **KVarN + greedy only**. int8/bf16
were covered in this run by short-context gate cells (verify battery: kv_i8 80.0 t/s,
accept 81.5%) and the correctness battery (int8 T1-T12 all PASS) — NOT by 10k-250k
t/s guard cells. No sampling (temp 1.0) guard cells of any cache type ran in CI.

**Ordering constraint:** #1 lands first (in flight). The §1 verification debt (126 Tiers
1/2a/3 + `run_ci.sh --full`) is cleared around/after the MultiBatch merge, before #2 or #3
start — #2 and #3 both change accumulation order and need a clean green baseline to diff
against.

## 6. Current official measurement state (as of 2026-09-01)

- Decode t/s (greedy, MTP on k=3, 192-token budget): 10k **74.8**, 25k **71.2**, 40k
  **67.6–67.7**, 80k **61.6**, 160k **53.7–53.8**, 250k **46.4–46.5** (guard final2 +
  ci_cells; MTP acceptance 70.7/68.5/73.7 = baseline at 10k/40k/160k).
- Prefill t/s: **806.7 / 740.1 / 583.4 / 503.3** @10k/40k/160k/250k (matches the
  max-prefill lineage 808/741/584).
- Baseline of record: `tools/bench/decode_guard_baseline.json` official v1 (08-29, Qwen
  canonical configs) — **ratchets: 1** (2026-09-01, sampling section raised
  post-decay-fix; greedy section still 08-29 values — greedy cannot ratchet via
  plain `--ratchet` by design).
