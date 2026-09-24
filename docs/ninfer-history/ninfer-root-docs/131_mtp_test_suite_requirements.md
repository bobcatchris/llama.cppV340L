# 131 — MTP Multi-Batch Pipeline Test Suite: Requirements

Status: requirements (for QA implementation). Owner: QA agent (third agent). Branch: cut from `wo/kv-uniform` (worktree `wo-mtp-tests`, created 2026-09-02).
Companion docs: 127 (kvarn multibatch status), 129 (kvarn multibatch debrief), 130 (multibatch status + MTP WIP + §8 session-3 debrief), 128 (phased kernel gate pipeline — the B-chain this suite feeds).

## 0. Purpose

We keep falling into the same MTP hole: a bug that only shows up as a probabilistic token
divergence at long sequences, found by accident, localized over days. This suite exists so that
that never happens again. Requirements:

1. **Every test either returns green or points at the cause.** A red result must be a structured
   report that identifies (round, layer, lane, invariant, expected, actual) and maps to a component
   via the cause map (§9). No "something is wrong" reds.
2. **The suite is a permanent net, not a one-time sweep.** Any future change to the MTP pipeline
   (kernels, workspace, runner, GDN, accept logic) runs this suite. The tests must keep working
   when the pipeline changes — they test invariants and behavior, not implementation details.
3. **End-to-end tests flush out divergence.** Not just unit checks: the full 2-lane batched MTP
   decode path, token-exact vs the isolated single-sequence reference, across the geometry battery
   that has historically produced bugs (B-shrink, page boundaries, ext=0 terminals, resets).
4. **The suite must be proven sensitive.** Canary tests (§7 T2.5) inject known breakage and
   assert the suite catches it. A net that cannot see a bug is worse than no net.

Acceptance (the suite is done when ALL hold):
- (a) Green on the fixed tree (after the T=256 terminal flake is fixed and verified).
- (b) Mutation proof: on the pre-fix tree, at least one tier goes red with a cause-map signature
  for the known bug (proves the net sees this class of bug).
- (c) Every red output maps to a component via §9 (QA agent fills in actual file/line pointers
  during implementation; the draft map is in §9).
- (d) All canary tests (T2.5) are detected.

## 1. Pipeline under test (self-contained description)

Two-lane batched MTP (k=3 draft depth) on the TP2 backend (`run_tp2_requests_batched` in
`src/runtime/tp2/tp2_backend.cpp`). Topology: ONE process, two rank threads (t0 = rank0, main
thread = rank1), shared host `lanes[]` (barrier-ordered via `sync_bar`), per-rank `TpRankState`
(own KV pool, GDN ring, KVarN workspace, staging), NCCL between ranks. Both ranks run the same
lane data; rank 0 is the reference for host dumps.

### Round loop (one MTP verify round)
1. **B1 publish** (host→device via synchronous `up()` copies): verify positions `[T,B]` (T =
   k+1 verify columns, B = live lanes), `lane_ids` (lane id per batch column — critical under
   B-shrink), `valid_columns` (per-lane live column count), drafts, and per-lane committed-page
   counts (`lpp` device array).
2. **Per-layer verify forward** (n_layers, each on stream `s`):
   - GDN: `causal_conv1d` snapshot + recurrent state snapshot (`recurrent.cuh`), per-lane ring
     slots (ring base is LANE-KEYED, not live-index-keyed).
   - KVarN attend (`gqa_decode_slice4_kvarn_kernel`, MultiBatch=true/Masked=true): reads committed
     pages (K4/V2 codes + scales, via per-lane block-table row `[logical_pages, rows]`) plus the
     open tail tile (`k_tile`/`v_tile`, bf16 pre-quant, 64 slots per lane, lane slices
     `tail_batch_elems = kG*kD*kv_heads` apart in one batched buffer). `window = pos[tokens-1]+1`;
     tail read extent = `[key_base, key_base + tail_count)`, `key_base = packed*64`.
3. **Argmax** (per verify column) → **NCCL allgather** → **accept**: `lic` (accepted prefix
   length incl. anchor), `acc`, `next_ext` (monotone draft-acceptance counter).
4. **Commit + rewind**: accepted tokens' KV stays in the tile (or commits when the page fills:
   64 tokens/page); rejected verify columns are rewound (`gqa_kvarn_rewind_to_token_count`).
5. Next round. `B` shrinks 2→1 when a lane reaches its emitted count (lane0 first in the standard
   test config); after shrink, `lane_ids` remaps columns to lane ids.

### Key buffers / state
- Per-lane `KvarnLayerWorkspace` (`src/ops/kvarn/kvarn_workspace.cpp`): `k_tile [64,256,kvh]`
  bf16 token-fastest, `v_tile [256,64,kvh]` bf16 channel-fastest, `committed_pages`, `tail_count`,
  `tile_page`. MTP tile = separate workspace (`is_mtp`, `layer_index == -1` in append calls).
- Block tables: device matrix `[logical_pages, rows]`, column-major (lane-major memory); row `b`
  = lane `b`'s page mapping. `lpp[b]` = lane `b`'s committed page count.
- GDN ring: per-lane slot base; `active_columns` gating can leave STALE prior-round values in
  unused columns (by design — see T0.3 semantics).
- Host staging: `mb_*` persistent buffers (vpos, rows, valid, lt, tail) — synchronous legacy
  `cudaMemcpy` publish (safe as write→read; the async-H2D variant was a real bug class, fixed).

### Invariants (the suite's backbone — every red report names one of these)
- **INV-1 Token exactness**: batched run's token stream == isolated single-sequence reference
  (fresh processes, same seed/prompt/lengths). This is the golden.
- **INV-2 Round pairing**: batched round's `(cur_F, cur_anchor, generated)` matches the
  corresponding single-sequence round (the M0 gate's pairing key; docs/130 §6).
- **INV-3 Tail consistency**: at every attend, per lane: `window == packed*64 + tail`,
  `tail ∈ [0,64]`, and the tail extent never covers a slot that has already committed
  (tail is the unwritten range only).
- **INV-4 Tile fidelity**: any slot the attend reads contains exactly what the append last wrote
  for that position (or, after a hydrate, exactly `dequant(quant(original))` — a deterministic
  choice, never a mix, never stale-arena content).
- **INV-5 GDN ring**: slot = ring base + slot index, LANE-KEYED; B-shrink never aliases one
  lane's ring onto another's.
- **INV-6 Determinism**: same seed → identical token stream across runs (in-process re-run AND
  fresh process). (Single-seq is deterministic today; batched must be too once fixed.)
- **INV-7 Draft sanity**: drafts in vocab range; an uninitialized/garbage draft never changes
  acceptance (`next_ext` monotone; `lic` only from real tokens).
- **INV-8 Termination**: EMITTED count (not frontier) drives stop; B-shrink keying uses
  `lane_ids`, never live column position.
- **INV-9 lpp/block-table consistency**: `lpp[b] == ws[b].committed_pages` at publish time;
  block-table row `b` == lane `b`'s page mapping.
- **INV-10 No cross-lane bleed**: lane `b`'s tile/block-table/lpp bytes are never read by lane
  `c ≠ b` (stride correctness: `tail_batch_elems == kG*kD*kvh`; `table_row`→lane mapping; lane
  slice contiguity in the batched buffer).

## 2. Bug history → what the suite must prevent

Each bug below cost days. The test IDs in the right column are its permanent guard.

| Bug (doc) | Class | Guard |
|---|---|---|
| j-vs-b / B-shrink keying (130 §5, fixed 93fda35e, f8f926bc) | lane id vs live column | T0.8, T1.1 (shrink cells), INV-8/10 |
| Double-push termination race (130 §5) | data race on shared host vector | T1.1, INV-8 |
| Contaminated in-process reference (130 §6) | methodology | T2.1 methodology (fresh processes only) |
| M0 gate self-corruption: rank 2-writer dump (130 §8) | harness race | T2.3 requirement (rank-gated dumps) |
| lpp async-H2D race (130 §7.3) | legacy-stream ordering | T0.6, T1.3 (content-hash determinism) |
| Uninitialized reads: AR draft window=0 column; masked GDN/conv columns (130 §8) | stale/unwritten memory | INV-7, T0.3 (zero-fill asserts), T0.5 |
| ext=0 terminal degeneracy (all 4 verify columns same position; T=256 only) | degenerate geometry | T0.1(c), T1.1 forced-ext=0 |
| Tile content divergence at r74 (130 §8 — current bug) | stale/lost tile content or OOB read | T0.7 (tile round-trip), INV-4, T1.3 |
| Silent zero-output / bench route mismatch (124) | bench not running the real route | T3.2 (non-trivial output asserts) |
| 32-periodic byte diffs (K4/V2 dequant signature, 130 §8) | lossy hydrate vs original bf16 | T0.7 must distinguish content provenance |

## 3. Test tiers

Tier 0 = ops-level unit (no model, no TP2 threads, one device, CI-friendly, seconds each).
Tier 1 = pipeline micro-harness (real TP2 stack, real model, short T, fast iteration).
Tier 2 = end-to-end (full model, ~1-2 min/cell). Tier 3 = standing gates + cause map.

### T0 — ops-level unit (tests/ops/ or tests/mtp/, `.cu` files next to kvarn_batched_ops_test.cu)

- **T0.1 MultiBatch attend geometry battery.** Cases: (a) B=2 steady; (b) B=1 shrink (lane0
  `valid_columns=0`, lane1 active); (c) ext=0 degenerate (all T columns same position);
  (d) page-boundary crossing (`window` = 64k, 64k+1, 64k+63); (e) `tail_count=0` with
  `committed>0`; (f) `tail_count=63` and `64`; (g) shuffled block tables (non-identity page
  order); (h) `table_rows` permutation (lane reordering); (i) `valid_tokens` = 0/1/3.
  Reference = the SINGLE-SEQUENCE kernel run per lane on the same per-lane inputs.
  Assert per-lane bit-exact outputs. Red report: first mismatch (case, head, token, key slot)
  + expected/actual bytes.
- **T0.2 Tail-count cross-check.** For every T0.1 geometry: compute the kernel-derived
  `tail = window - packed*64` (clamped [0,64]) and the host `tail_count`; assert equal and
  assert INV-3. Dump all inputs on mismatch. (This is the check that was only ad-hoc during the
  §8 hunt; it is silent today — keep it that way.)
- **T0.3 GDN verify battery.** `valid` = 0..4: assert output columns ≥ valid are exactly zero
  (conv side is verified to zero-fill: `causal_conv1d.cuh` `if (column >= valid)` writes 0 and
  never reads `x0`). For the recurrent snapshot: pin ONE semantics for invalid columns
  (write-zero vs skip/stale) — assert it, and document it. Ring-base mapping under B-shrink
  (lane-keyed, INV-5). Reference = CPU single-column GDN.
- **T0.4 Accept kernel battery.** Synthetic (tgt, drafts, ext, len, anchor): all-accept,
  none-accept (ext=0), partial, stop-token mid-chain, room-clamp near max_output. Assert
  `lic`/`acc`/`next_ext` semantics + `next_ext` monotone (INV-7).
- **T0.5 prepare_verify_inputs battery.** Normal / ext=0 / padded: assert the exact `[T,B]`
  vid/vpos matrix including the padding convention (padded columns repeat the anchor).
- **T0.6 Memory robustness.** Pre-fill the tail tile with 3 distinct patterns (zero, 0xFF,
  prior-round data) BEFORE a correct-tail attend. Output must be bit-identical across patterns
  (the kernel reads only `[key_base, key_base+tail)`). Pattern-dependent output = OOB read →
  report the slot range read (computed by pattern fingerprinting).
- **T0.7 Tile round-trip fidelity** (the test that would have caught the current bug;
  reduced repro, no model/TP2/threading):
  Setup: bind batched workspace (2 lanes, K4V2 batch cache, per-lane block-table rows).
  For each `(B ∈ {2, 1-shrink} × tail_count 0..64 × append positions crossing a page boundary)`:
  1. append both lanes' tiles via `gqa_kv_append_kvarn_batched`;
  2. record tile bytes (per lane);
  3. `gqa_kvarn_rewind_to_token_count` to a mid-page frontier;
  4. append again across the next boundary (forcing commit + any hydrate);
  5. assert: for every slot inside the attend's read extent, tile bytes are BIT-IDENTICAL to
     what the append wrote for that position (INV-4). Also assert content provenance is
     deterministic: run the same sequence twice and compare (no run-to-run mixing of original
     bf16 vs dequant content).
  Cover the MTP tile too (`layer_index = -1`).
- **T0.8 Stride/geometry audit** (no GPU): assert `kvarn_tail_elems_` (tp2_backend)
  == `kG*kD*kv_heads` == `kvarn_bind_batched_workspace`'s returned `elems`
  == `GqaKvarnTail::tail_batch_elems`; lane slice `b` at offset `b*elems` in the batched buffer
  (contiguity); `table_row`→lane mapping; block-table row stride. Any mismatch = stride bug
  class (INV-10).

### T1 — pipeline micro-harness (real TP2 stack; new tool under tools/bench/)

- **T1.1 Batched runner stress.** `ninfer_tp2_batched_decode_test`-class harness,
  `T ∈ {64,128,256} × {B=2 both full, B=1 shrink (lane0 max_output=30)} × {normal, forced
  ext=0 (stub drafts that never accept)} × N=200`. Assert INV-1 (token-exact vs in-run isolated
  reference is NOT acceptable — use T2.1's reference for the matrix; here assert INV-6:
  run-to-run identical) per run. Report the FIRST failing (config, run index, token position,
  round). This is the high-frequency version of the current bug hunt.
- **T1.2 Invariant assert mode** (`NINFER_MB_INVASERT`, env-gated, zero cost when unset):
  checks INV-2, INV-3, INV-5, INV-7, INV-9, INV-10 at EVERY round × layer × lane inside the
  real runner; fail-fast. Structured output: `round=<r> layer=<l> lane=<b> INV=<n> expected=<e> actual=<a>`.
  This converts any future probabilistic flake into a deterministic cause report at the first
  violating invariant — the single highest-value instrument in the suite.
- **T1.3 Cross-run content hash** (`NINFER_MB_HASHPT`, env-gated): per-round checkpoint hashes —
  GDN ring slots, tail tile (per lane), lpp, block-table row, and sample q/k/v at layers 0, L/2,
  L-1 — appended to a file. A test runs N=16 and diffs: identical inputs ⇒ identical hashes.
  The FIRST divergent (round, point) localizes the race. This generalizes the ad-hoc DBG-GDN /
  multi-step capture work of the §8 hunt into a standing tool.
- **T1.4 Determinism.** 2× in-process (engine reset between) + 2× fresh process, same seed:
  4× token-identical, for batched AND single-seq. (INV-6.)

### T2 — end-to-end (full model)

- **T2.1 Golden matrix** (merge gate). `T ∈ {64,128,256} × {both-lane-full, B=1 shrink
  (lane0=30), uneven plen} × 3 runs`. Reference = ISOLATED single-sequence runs in FRESH
  processes (`NINFER_SEQ_ONLY` + `NINFER_DUMP_BATCHED`), diffed host-side. (In-process reference
  is contaminated — docs/130 §6; never use it.) Token-exact both lanes.
- **T2.2 Flake battery.** T=256 × 16 runs, in-process token-exact detection (exit code).
  Requirement: 16/16. (This is the battery that currently shows ~40-50% failure.)
- **T2.3 M0 phase gate** (existing: `tools/bench/phase_gate_mb.{py,sh}`). Requirement: dumps are
  rank-gated (`if (gdir) if (rank == 0)` — the §8 self-corruption fix); re-run after every
  change. Gate output must remain all-MATCH on the fixed tree.
- **T2.4 Soak.** 8 sequential T=256 requests on the same engine (reset between). Exercises the
  reset/prefix-reuse path (`kvarn_reset_inflight`, MTP re-prepare) — a historical bug surface.
- **T2.5 Canary (must-fail) tests.** Inject each known breakage, assert the suite detects it:
  (i) publish `tail_count ± 1`; (ii) swap two lanes' block-table rows; (iii) disable the
  masked-column zero-fill in conv; (iv) corrupt one tile slot after append; (v) un-rank-gate
  the M0 dump (expect the self-corruption signature). Each must trip T1.2 or red T2.1/T2.2 with
  a §9 signature. A canary that is NOT detected = a hole in the net → fix the test, not the
  pipeline.

### T3 — standing gates + cause map

- **T3.1 Gate rule.** Any commit touching `tp2_backend.cpp`, `text_context_impl.h`,
  `gqa_decode_slice4_kvarn.cuh`, `causal_conv1d.cuh`, `recurrent.cuh`, `kvarn_workspace.*`,
  `speculative_*`, `gqa_attention_kvarn*` must run: T0 (all) + T1.1 quick (N=20) + T1.2 +
  T2.1 (T=64 cell) + T2.2 (N=4). Full T2 before merge.
- **T3.2 Bench integrity.** Any bench used as a reference must run the EXACT route the model
  runs (docs/124 route-mismatch lesson) and must assert its outputs are non-trivial (a bench
  that silently compares two zero outputs is worse than none). Port 8091 is single-user:
  never two benches at once.
- **T3.3 Cause map** (see §9; QA agent fills in file/line pointers during implementation).
- **T3.4 Runbook.** How to run each tier, expected wall time, GPU coordination (check
  `nvidia-smi --query-compute-apps` + `pgrep` BEFORE launching; coordinate via intercom — the
  GPU is a serial resource), reset procedure, dump locations, log naming (`/tmp/mtptest_<tier>_<config>_<n>/`).

## 4. Implementation requirements

- **Location**: new test files under `tests/` (suggest `tests/mtp/`), harnesses under
  `tools/bench/`. New worktree `wo-mtp-tests` (branch cut from `wo/kv-uniform`); merge back —
  all test files are additive, conflicts expected to be minimal. Instrumentation in `src/`
  follows the existing `NINFER_MB_*` env-gated pattern: ZERO COST WHEN UNSET, never changes
  production behavior when unset.
- **Methodology**: isolated fresh-process references only; dumps rank-gated (`rank == 0` only);
  every `fopen` NULL-guarded with a stderr shout (a harness NULL-dir bug produced fake
  "core dumps" during the §8 hunt); no device-side `printf` on timing paths; E2E cells stay
  short (plen 5/8, T ≤ 256 — long prefills get reaped by the harness).
- **Exit codes**: 0 = green; 1 = divergence (with structured report); 2 = crash (with report);
  3 = harness error (with report).
- **Structured failure lines** (machine-parseable, one per violation, first N=10):
  `TEST=<id> INV=<n> round=<r> layer=<l> lane=<b> expected=<e> actual=<a> [extra=...]`
- **The suite must be runnable headless**: a single `run_all.sh` per tier, machine-parseable
  summary at the end (`PASS=.. FAIL=.. FIRST_FAIL=...`), designed for a coordinator agent to
  re-verify independently (coordinators re-run verification matrices on fresh dumps — never
  trust a single agent's claims).

## 5. Ownership and sequencing

- **QA agent** owns T0-T3 implementation in `wo-mtp-tests`.
- **Executor agent** (01a06233 at time of writing) owns the product fix in `wo-kv-uniform`;
  test instrumentation that must live in `src/` is merged back after the fix lands.
- **Sequence**: T0.8 (minutes, no GPU) → T0.7 (the current bug's reduced repro) → T0.1-T0.6 →
  T1.2/T1.3 (instruments) → T1.1/T1.4 → T2.1-T2.4 → T2.5 (canaries) → T3.
- **GPU discipline**: the executor is using the GPU in ~1-min bursts for the flake fix;
  coordinate any GPU run via intercom before launching.

## 6. Known pitfalls (read before implementing)

1. **j-vs-b class**: anything keyed on a live batch column must be re-checked under B-shrink —
   the key is the LANE ID (`lane_ids[b]`), never the column. This class has produced two of the
   five bugs above.
2. **Termination is by EMITTED count**, not frontier (`cur_F`); the two diverge in the last
   rounds (docs/130 §5.4).
3. **ext patterns are deterministic per T** (greedy + deterministic AR draft) — the terminal
   rounds of T=256 are ext=0 (all 4 verify columns at the same position); 64/128 terminals are
   not. Do not assume randomness where determinism exists (or vice versa).
4. **In-process references are contaminated** (shared host state). Fresh processes only.
5. **Bench route mismatch**: the bench must invoke the same kernel instantiation/route the model
   uses, or it proves nothing (docs/124).
6. **Unwritten tile slots are expected to vary** between runs (stale arena content). Only bytes
   INSIDE the attend's read extent `[key_base, key_base+tail_count)` matter. (The §8 hunt's
   step-74 tile diff was partly exactly this — check the extent before concluding corruption.)
7. **Hydrate is lossy by design** (`dequant(quant(x))`) but its fire decision is
   host-deterministic, and normal decode enters fresh pages at slot 0 (no hydrate). 32-periodic
   byte diffs with K broad / V isolated = K4/V2 dequant signature.
8. **MTP tile is a separate workspace** (`is_mtp`, `layer_index == -1`) with its own
   append/rewind timing — most MTP bugs so far have been text-tile bugs; the MTP tile is the
   least-audited surface.

## 7. Out of scope (for this suite)

- Performance gates (decode_guard / perf ratchets exist separately; T3.2 only requires bench
  integrity, not perf thresholds).
- Prefill-path tests (this suite is decode/MTP; prefill has its own gates in docs/119-122).
- The doc-128 D-chain (unified kernel) — that lives in the phase-gate worktree and has its own
  M1 gate. This suite FEEDS the doc-128 B-chain (multi-batch milestone) once the flake is fixed.

## 8. Definition of green

- T0: all batteries bit-exact vs references; T0.2/T0.8 asserts silent.
- T1: INVASERT silent over N=200 stress; HASHPT identical across N=16; 4× determinism identical.
- T2: matrix token-exact; flake battery 16/16; M0 gate all-MATCH; soak clean; ALL canaries
  detected.
- T3: gate rule in place (CI or documented manual gate); cause map complete with file/line
  pointers; runbook verified by a cold agent (one who has not seen docs/130) executing T2.1
  start-to-finish unaided.

## 9. Cause map (with exact file and line pointers)

| Failure signature | Suspected component | Where to look |
|---|---|---|
| T0.1(b) red, lane1 output wrong, lane0 neutral | B-shrink keying (j-vs-b) | `src/runtime/tp2/tp2_backend.cpp:2517`, `src/ops/kernel/gqa_decode_slice4_kvarn.cuh:335-348` |
| T0.1(c) red at slot S only | tail extent vs written extent | `src/ops/kvarn/kvarn_workspace.cpp:146`, `src/ops/launcher/gqa_attention_kvarn.cu:535` |
| T0.7 red at slot s after step 4 | hydrate/commit interleaving or OOB read | `src/ops/kvarn/kvarn_workspace.cpp:215`, `src/ops/kvarn/kvarn_workspace.cpp:176` |
| T0.6 pattern-dependent | kernel OOB read (extent larger than written) | `src/ops/kernel/gqa_decode_slice4_kvarn.cuh:330-348` |
| T0.8 mismatch | stride bug class (INV-10) | `src/ops/kvarn/kvarn_workspace.cpp:121`, `src/targets/qwen3_6/impl/runtime/text_context.h:219` |
| T1.1 red, B=1-shrink cell, round > shrink point | B=1 phase keying | `src/runtime/tp2/tp2_backend.cpp:2440-2525`, `src/runtime/tp2/tp2_backend.cpp:2740` |
| T1.2 INV-3 trip | tail/packed/window inconsistency | `src/runtime/tp2/tp2_backend.cpp:2520`, `src/ops/kvarn/kvarn_workspace.cpp:153` |
| T1.2 INV-7 trip | uninitialized draft | `src/runtime/tp2/tp2_backend.cpp:2387`, `src/ops/launcher/speculative_round.cu:40` |
| T1.3 first divergence (round r, tail_tile) | tile data race or OOB read | `tests/mtp/t0_tile_roundtrip_test.cu`, `src/ops/kernel/gqa_decode_slice4_kvarn.cuh` |
| T1.3 first divergence (round r, gdn_ring) | GDN slot math (INV-5) | `src/runtime/tp2/tp2_backend.cpp:2517`, `src/ops/kernel/causal_conv1d.cuh:367` |
| T2.1 red, B1/B4 dumps identical, token differs | device-side content (inputs identical) | `tools/bench/t1_stress_harness.py`, `src/runtime/tp2/tp2_backend.cpp:2615` |
| T2.2 red (rate > 0) | probabilistic device race | `tools/bench/t2_golden_matrix.py`, `src/runtime/tp2/tp2_backend.cpp:2405-2415` |
| T2.5 canary NOT detected | hole in the net | `tools/bench/t2_golden_matrix.py:run_t2_5_canary`, `tests/mtp/` |
| M0 gate STATE-DIVERGENCE | round pairing or converged-state | `tools/bench/phase_gate_mb.py:90`, `src/runtime/tp2/tp2_backend.cpp:2571` |
