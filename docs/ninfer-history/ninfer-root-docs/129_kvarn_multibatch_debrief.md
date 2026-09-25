# 129 — KVarN MultiBatch debrief: decay fix, batched decode, and the post-commit bug

Status: RESOLVED (2026-09-01, takeover session — see §10). Original debrief below
(2026-09-01, end of autonomous session). Branch: `wo/kv-uniform`
(HEAD `9bdb7247`, pushed to `github`). Author: overnight agent.

---

## 0. Read this first (30-second summary)

Two tasks were executed back-to-back:

1. **The decay fix (DONE, verified, shipped).** The long-context KVarN decode decay —
   the project's blocking issue — was root-caused to a synchronous global→smem code-copy
   loop in the slice4/UNIFIED attention kernel and fixed with double-buffered cp.async
   prefetch. All six decode-guard cells improved (+4% to +14%), prefill is untouched,
   and the full correctness battery is green.

2. **MultiBatch / option R1 (kernel + ops core DONE and proven; runtime prototype runs
   end-to-end with ONE remaining bug).** Two-lane batched decode on the TP2 backend runs
   to completion: both lanes prefill, decode in lockstep, and lane 1 matches the shipped
   sequential path **token-for-token for the full generation**. Lane 0 diverges at
   exactly **generated index 16** — the first decode step whose attention spans a
   **committed page** (prompt 49 tokens + 16 generated = position 64 = the first
   page-commit boundary). All inputs the kernel consumes have been individually verified
   correct; the capture-replay tool proves the kernel+launch are bit-exact on those
   inputs; the divergence therefore sits in ONE narrow, identified interaction — see §6.

Everything is committed and pushed. Debug tooling is env-gated and production-inert.

---

## 1. Task inception and the chain of documents

| Document | Role in this task |
|---|---|
| **docs/125_handoff_next_agent.md** | **The inception document** for this session. Handed off: the decay bug (unresolved), the R1/R2 MultiBatch options, the measurement approach (decode_guard, port 8091 discipline), the last-known-good commits, and the "route reframe" lesson (bench must measure the kernel the model actually runs). |
| docs/122_kvarn_performance_knowledge_base.md | The performance knowledge base. §22–§23 added this session: the decay root cause + fix record, and the silent zero-output bug record. |
| docs/123_kvarn_batched_quantize_debug.md | Prior art: the batched-quantize scale-offset bug (V scale at +512 vs +1088). The debug playbook followed here. |
| docs/124_kvarn_optimization_queue.md | The optimization queue with verdicts. Updated: B-STAGING4 section (decay resolved), prefill-exhausted verdict superseded note. |
| docs/126_verification_runbook.md | The verification handoff written mid-session for the parallel agent (Tier 1–3 test battery, guard cells, byte-diff A/B). Also records the run_ci.sh --full final-gate rule. |
| docs/127_kvarn_multibatch_status.md (created this session) | The MultiBatch status/design doc: R1 core architecture, proven pieces, the scheduler gap, aggregate-throughput estimates (≈1.6× @N=2 … 2.9× @N=8), and the updated decay table. |
| docs/128_phased_kernel_gate_pipeline.md (on `wo-phase-gate`, by another agent) | The phased kernel gate (D1 commit/quantize → D2 dequant → D4 QK → D5 softmax → D6 PV → D7 reduce, chained oracles + CPU references). **This session adopted its methodology directly**: capture phase-boundary artifacts from the live batched path and replay them through proven kernels (see §6). |
| tools/bench/decode_guard.sh + decode_guard_baseline.json | The measurement arbiter (token-rate + MTP acceptance per context cell). |
| tools/bench/decode_guard_check.py | Baseline comparison (PASS recorded for the decay fix). |
| The beellama-kvarn fork (valujin) — case study in docs/122 §18 | External proof the decay was reducible; its per-commit fix list (read-plan, skip-masked-split, register cap) informed the attack order. |

---

## 2. Part 1 — the decay fix (COMPLETE)

### 2.1 Root cause

The slice4/UNIFIED kernel's per-page code staging was a **synchronous
global→register→smem copy**: 4-byte `LDG` chains with only 2–4 loads in flight per
thread, executed inline at the first tile of each 64-key page and serialized by a
`__syncthreads()`. Measured: ~50% of the entire kernel at 160k context, ~88 GB/s
effective — **latency-bound, not bandwidth-bound** (Little's law predicts ~80 GB/s from
the outstanding-bytes count; 88 measured). KVarN therefore paid ~3× int8's per-key cost
on *half* the bytes, and attention's linear growth with context is what collapsed t/s.

Prior "V-staging is the decay source" hypotheses were wrong in a instructive way: the
problem was never *which* codes were staged — it was the **copy mechanism** (both K and
V), which is why every V-only fix measured neutral.

### 2.2 The fix

Per-32-key-tile **double-buffered banks filled with `cp.async`** (8 B ops for K into a
skewed layout `16*d + 8*(d>>4) + j`, 16 B ops for V), prefetched **one tile ahead** so
copy latency hides under the current tile's dequant+MMA. The skew makes the dequant LDS
hit **32 distinct banks (conflict-free** — strictly better than the old 36-byte pad's
8-way conflict). Banks moved to dynamic smem (12,544 B total); occupancy **stays at
2 blocks/SM** (verified via the occupancy API). T=1 decode now launches Wc=4 (more
cp.async issue threads; the extra padded-row MMA is cheaper than the halved issue cost).

Follow-up micro-win: **keypair K-dequant** — each K code byte holds keys (2j, 2j+1); one
warp now processes both keys per byte (the pair used to be split across two warps,
loading/unpacking the byte twice). Bit-identical math.

### 2.3 Verified results (decode_guard, greedy, ITERS=1, COMP=192)

| ctx  | baseline | fixed | Δ      | vs int8 |
|------|---------|-------|--------|---------|
| 10k  | 69.1    | 74.8  | +8.2%  | 1.07×   |
| 25k  | 68.3    | 71.1  | +4.1%  | 1.02×   |
| 40k  | 64.0    | 67.7  | +5.8%  | 1.02×   |
| 80k  | 58.2    | 61.6  | +5.8%  | 0.97×   |
| 160k | 47.1    | 53.7  | +14.0% | 0.90×   |
| 250k | 43.5    | 46.4  | +6.7%  | —       |

Prefill unchanged at every cell (806.7/740.1/583.4/503.3 tok/s — the max-prefill lineage).
MTP acceptance within the historical band. `decode_guard_check.py`: PASS. Standalone
kernel: 2.110 → 0.938 ms/layer @160k T=4 (**2.25×**), reproduced across runs.

### 2.4 Branch-base decision (recorded for posterity)

The handoff offered two last-known-good points (`7917a1de` max-prefill, `75c416e4`
max-decode@10k). Analysis: HEAD was already a superset — `7917a1de` is an ancestor, and
`75c416e4`'s 73.8 t/s was measured with a different gen budget and its V-staging was
superseded. Worked from HEAD. (Now beaten: 74.8.)

### 2.5 Bonus bugs found and fixed during this part

1. **Silent zero-output in the unified route** (`ac6b3f61`): the launcher inferred
   `packed_pages` from the envelope when `tail.packed_pages == -1` (the isolation-test
   convention) but forwarded the tail STRUCT unchanged — the kernel saw -1, treated
   every tile as an empty tail, and wrote m=-inf → **zeros, silently**. Live serving
   always sets packed_pages ≥ 0, so only tests were affected — but `kvarn_gqa_test` had
   been failing on the default route since the route flip, despite handoff claims of
   PASS. Fixed by writing the inferred value back into the by-value tail struct.
2. **Per-device smem-opt-in race** (`61dcecd8`): `cudaFuncSetAttribute` applies to the
   function on the caller's current device; a process-wide `static bool` let rank 1's
   thread launch with the attribute unset → `cudaErrorInvalidValue` at warmup
   (reproduced 2/2 under decode_guard, 0/2 manual — timing-dependent). Fixed with
   `thread_local`. This was MY regression, caught by the guard run.

---

## 3. Part 2 — MultiBatch (docs/127's option R1)

### 3.1 What the architecture actually is (findings that reshaped the plan)

- **The TP2 backend** (`tp2_backend.cpp` — the only engine that binds KVarN) serves
  requests **strictly sequentially**: `TPEngine::submit` queues, and the entire
  `run_tp2_request` runs under `shared->mutex`, streaming via a per-request
  TokenCallback. `TpRankState` holds single-lane tensors (`ids/cpos/rpos/kvr/slt` are
  `[1]`), and the decode loop is a two-rank `std::barrier` pair.
- **The graph engine** (ConcurrentExecutor + `decode_batch(lanes)` + per-batch CUDA
  graphs) has **zero KVarN binding** and is single-GPU — this 18 GB artifact cannot fit
  one 16 GB card.
- So "wire batch_size>1 + MultiBatch" required building the whole batched stack:
  kernel → ops → TextContext → runner.

### 3.2 Stage A — kernel + launcher (DONE, bit-exact)

- Kernel gained `lane_packed_pages` (device [batch]) + `tail_batch_elems`; under
  `MultiBatch`, each CTA (grid.z = lane) derives its **own** key_base/tail_count
  (invariant `window_b = packed_b*64 + tail_b`) and offsets the batched tail tiles.
  Everything else in the body was already lane-correct through the templated
  MultiBatch/Masked path.
- Launcher: `GqaKvarnTail` gained the batched fields; unified route dispatches
  MultiBatch/Masked=true for 4D q; partials sized width*batch; batched reduce.
- **Proof**: bench BATCHTEST — batch=2, lanes with windows 100/250, different block-table
  rows → **bit-identical** to per-lane single-sequence execution (0/6144).

### 3.3 R1 core — ops layer (DONE, bit-exact)

- `kvarn_bind_batched_workspace`: binds `batch` per-lane workspaces whose layer tiles are
  slices of shared [batch]-strided buffers (k `[G,D,kvh]` token-fastest, v `[D,G,kvh]`);
  returns the attend stride (= 64*256*kv_heads).
- `gqa_kv_append_kvarn_batched`: one call appends+commits all lanes. Key enabler: the
  **batch dimension is outermost** on every batched tensor, so per-lane slices are
  contiguous — per-lane append is the existing single-lane function + a per-lane
  block-table row view. Publishes per-lane committed_pages to device.
- **Proof**: `ninfer_kvarn_batched_ops_test` — 2 lanes (100/250 tokens, 2 layers):
  per-lane tail tiles + committed_pages/tail_count **binary-identical** to two
  independent single-lane sequences.

### 3.4 Public batched attend (DONE)

`gqa_attention_cached_batched` (4D q, batched cache view, valid_columns, kv_table_rows,
batched tail) → `gqa_attention_kvarn_cached_batched_launch` (decode-only, width ≤ 6).
`validate_batch_cache` extended to accept KVARN_K4V2 (U8 code planes + kvarn_scale_pages
side table instead of the bf16/int8 plane checks).

### 3.5 TP2 runtime — batched runner (DONE structurally; ONE bug open)

`run_tp2_requests_batched(backend, requests, callbacks, …)`:

- Pool/GDN sizing scales with the new `TpBackendOptions::max_concurrency`
  (kv_table_rows = lanes; GDN slots = 2*lanes + cache_slot; pool pages ×= lanes).
  **lanes == 1 keeps the shipped layout byte-identical** — the guard's config.
- Per-lane prefill: sequential, each lane against its own block-table row via the new
  `TextContext::set_kv_view`.
- Tail migration: each lane's post-prefill tail is copied into its batched-workspace
  slice **immediately after its own prefill** (later lanes' `kvarn_reset_inflight` would
  otherwise wipe earlier lanes' tails — bug #5 in §5).
- Lockstep batched decode: `[batch]` ids/cpos/rpos/kvr/slt per step, one
  `ordinary_decode_batch`, one per-column `allreduce_argmax` (the one-shot argmax kernel
  already supports T>1 columns), per-lane host commit on rank 0, per-lane callbacks.
- `TextContext::kvarn_attend_text_batched`: per-lane batched append + ONE MultiBatch
  attend, wired into `attn_mix_tp` (the TP2 attention path — the first wiring attempt
  missed this and patched `attn_mix`, which TP2 does not use).

### 3.6 Runtime bugs fixed while bringing it up (each caught by a targeted experiment)

1. **Null `valid_columns` deref**: `ordinary_decode_batch` has no valid_columns parameter
   and the MultiBatch kernel is Masked — the runner now uploads per-lane windows
   (cpos+1) per step (`set_kvarn_batch_valid_columns`) and the attend entry throws on
   null lanes.
2. **Heap corruption**: the batched argmax overflowed the single-int pinned token slot
   (`malloc(): unaligned tcache chunk detected`). Batch-sized pinned buffer.
3. **Per-lane tail wipe**: each lane's prefill calls `kvarn_reset_inflight` on the
   shared single workspace, destroying earlier lanes' tails. Migration moved to
   immediately after each lane's own prefill.
4. **Lane-0 alloc move**: `std::move(kv_alloc)` into lane_allocs[0] broke the
   single-seq prefill path (`Paged KV allocation is not bound`). Lane 0 reuses
   kv_alloc/kv_view.
5. **Barrier deadlock**: host lane-state updates ran on both rank threads (double-push
   races + asymmetric barrier counts). Rank-0-only now.
6. **Stale committed-pages pointer**: the attend's `lane_packed_pages` device buffer was
   allocated from the per-step work arena (`work_.reset()` every step) and cached across
   steps → aliased other allocations → garbage key_base. Now bound from the persistent
   arena via `set_kvarn_lane_packed_pages_dev` (and the upload's missing
   scheduler-pointer assignment was fixed — `cudaErrorInvalidValue` on the memcpy).

---

## 4. Current status of the batched prototype

| Lane | Result |
|---|---|
| lane 1 | **96/96 tokens identical to the shipped sequential path** — full generation |
| lane 0 | **tokens 0..15 identical; token 16 (and after) diverge** |

Token 16 is the first output whose forward **spans a committed page**: position 64
fills the tail to 64 → page 0 commits (codes+scales written) → the tail tile restarts
→ the attention now reads **one packed page (dequant) + one tail token** instead of an
all-tail window.

---

## 5. The debugging campaign (docs/128 methodology, applied)

Following docs/128's "chain oracles on phase-boundary artifacts" principle, every layer
of the pipeline was independently verified:

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | Kernel wrong for the committed+tail shape | Standalone bench, exact shape (`BT_SHAPE="65:1,46:0"` and `65,70`) | **PASSES bit-exact** — kernel excluded |
| 2 | Batched launch/kernel wrong on real captured inputs | NINFER_MB_CAPTURE dumps ALL kernel inputs at step 16 (q, pos, valid, rows, codes, scales, tail tiles, block table, lpp, launch scalars) → replay both kernels on identical inputs | **MATCH (0/3072)** — batched kernel == single-seq kernel; launch excluded |
| 3 | Commit content wrong | Per-lane-attributed checksums of written codes+scales+phys | **batched == sequential per rank** (d14f/6d63 stable) — commit excluded |
| 4 | Prefill wrong | Batched lane-0 t0 vs sequential t0 | **Identical** (11751; 561 on long prompts) — prefill excluded |
| 5 | Tail migration wrong | Tail tile values + counts at step 1/2 and 15/16 (rank-attributed) | **Correct and stable** — migration excluded |
| 6 | Per-step cursors/rows/slots wrong | MB-STEP dumps every step | **Correct** — inputs excluded |
| 7 | GDN all-slot zeroing | Full-slot hygiene experiment | **Broke lane prefill state (both lanes wrong from step 1) — reverted**; per-lane zeroing restored |
| 8 | Sequential-first ordering | NINFER_MB_SEQ_FIRST: sequential runs first, then batched | **Batched becomes fully correct** — the divergence depends on cross-run/cross-state history, not on the commit path itself |

Also built: exhaustive env-gated logging tiers (MB-STEP, ATTEND per-layer-0 state dumps,
COMMIT with lane attribution + physical page + code-plane base pointers, BT per-lane
block-table rows, MB-MAP pool row entries at commit-boundary steps, SEQ-PRE/SEQ-STEP1).
**Keep all of this logging until the feature is complete** — per the owner's instruction.

### 5.1 The CPU-reference probe (one caveat)

`kvarn_decode_cpu_ref.h` (from `wo-phase-gate`) was imported and wired as a full CPU
oracle for the captured lane-0 inputs. Its first run reported 2721/3072 out of tolerance
with 2.4e37 outliers — **the quick reference has a tail-index bug**: tail_k layout is
key-major (`key + G*(d + D*head)`); the probe indexed `[d]`, so the key-64 term read
wrong elements. The leading dims agree within bf16 tolerance, confirming the
dequant/softmax structure. **Fix the probe's tail indexing before trusting its
verdicts** (one-line: tail index = `64*d` for head 0, plus the head stride).

---

## 6. The remaining bug — precisely stated

**Symptom**: batched decode, lane 0 (the lane bound to `kv_alloc`/`kv_view`/GDN slot 0 —
the ORIGINAL single-path resources), produces its first wrong token at the decode step
whose attention window first includes committed-page data (window 65: page 0 committed
at position 64, tail = position 64). Lane 1 (lane_allocs[0]/row 1/slot 1) is correct
for the full generation under the identical code path.

**Verified correct at the failing step**: committed-page codes+scales (checksum match),
tail tile values + counts (rank-attributed dumps), lane_packed_pages upload ({1,0}),
block-table rows ({0,1}), per-step cursors, kernel+launch replay on captured inputs
(bit-exact, matches single-seq kernel).

**Not yet verified**: the Q/K/V *values* flowing into the kernel (the model's
projections at step 16) — everything structural has been checked instead.

**Leading hypothesis (updated)**: lane 0's attention reads a **stale view of page 0's
codes or scales** — the ATT-READ dump at step 16 hashed the code plane at
`[row=0][kvh=0]` to a value that does not match the commit's checksum of the same
address taken moments earlier inside the same append call (03dc/5a32 vs 6d63/d14f,
per rank). Two sub-cases remain:
- (i) the COMMIT-side hash and ATT-READ-side hash read at **slightly different
  addresses** (a base-pointer or layer-offset difference between the kvarn_workspace
  commit path and the batched-attend read path), or
- (ii) something **rewrites page 0's codes between the commit and the same-step
  attention** (e.g. another lane's append resolving phys=0 via a stale row view — note
  lane 1's commit/append path must be re-checked once logging attribution is rank+lane
  tagged end-to-end).

Sub-case (i) is testable in minutes: print `cache.k_pages.data` inside
`commit_completed` (already added: kbase=0x7bc758000000/0x7bc464000000 per rank) and
`bcache.k_pages.data` in the attend (same values printed ✓). **They matched** — so the
bases are equal and the remaining difference is the **offset within the plane**: the
commit wrote at `phys * heads * kb_bytes` (phys=0 → 0) and the attend read at
`(kvh + page*KVHeads) * 8192` (page=0 → 0) — identical… which is why the hunt has been
so stubborn. The next dump must therefore compare **the full 16 KB page region** (not
16 bytes) **at the same instant**, plus the scale row for key 64's page — and, per the
owner's guidance, all of it in one run with the comprehensive logging already in place.

---

## 7. Recommendations / next steps (ordered)

1. **Fix the probe's tail indexing and re-run the CPU oracle** (~15 min): tail index for
   key 64 = `64*d` (head 0) — then the CPU reference gives the mathematically correct
   output for the captured inputs. Compare CPU vs GPU-batched: if CPU == GPU, the bug is
   upstream of the kernel despite the state dumps (i.e., the dumps read copies while the
   kernel reads live data — move the dumps to post-attention and diff outputs instead).
2. **Rank+lane-tag every remaining dump** (COMMIT already has it; add rank to ATT-READ,
   MB-TAIL) — the rank attribution has already caused one misreading.
3. **Diff the full 16 KB committed-page region + full 1152-float scale row** at the
   commit step, read through the ATTEND's pointers, in both orderings
   (NINFER_MB_SEQ_FIRST on/off). One of the four (write-side vs read-side) ×
   (batched-first vs seq-first) combinations will show the stale content.
4. **GDN-slot-0 experiment**: run the batched lanes with slots {2,3} instead of {0,1}
   (one-line change in the runner's slt_h) — if lane 0 becomes correct, the pollution is
   GDN slot 0's interaction with the single path's conventions (single path uses
   (0,1) as (current,rewrite); the batched path uses (0,2) for lane 0).
5. **Then the MTP prototype** (owner-requested): batched verify via
   `target_verify_batch` (already batch-shaped), per-lane drafts/acceptance, and the
   `width > 1` append-stride question (batch outermost ⇒ lane slices contiguous, so the
   append helper needs a width-loop, not a stride change).
6. **Then the engine scheduler** (tp_engine.cpp): collect up to `max_concurrency`
   queued requests and drive `run_tp2_requests_batched`; single-seq path (N==1) stays
   on `run_tp2_request` untouched.
7. **Close out with the docs/126 runbook tiers** (unit battery, regress_unified,
   25k/80k guard cells, byte-diff A/B) and `run_ci.sh --full` as the final gate
   (it rebuilds — run only after the feature settles).
8. **Ratchet the guard baseline** (`decode_guard_check.py --ratchet`) once the owner
   blesses the new decode numbers — the current baseline still holds the pre-fix rates.

---

## 8. Full commit trail (this session, in order)

```
0363c137 kvarn(decode): MultiBatch Stage A — per-lane tail state in slice4 + batched launcher
1a2ccd1b tests: MultiBatch bit-identity check is the DEFAULT ctest behavior
7a9a9c25..2beaea92 docs: 127/128 gate status + ratchet re-baseline (parallel-agent work on the branch)
c631f7f1..ac6b3f61 kvarn(decode): cp.async double-buffered staging + keypair dequant
                    + prefetch-before-wait + unified-route zero-output fix
61dcecd8 kvarn(launcher): thread_local smem opt-in (TP rank-device race)
580d93ab..3b043428 docs: 122 §22-23 / 124 B-STAGING4 / 125 banner / 126 runbook / 127 status
326ec81c kvarn(MultiBatch R1 core): batched attend entry + KVarN-aware validation
1b9c10c5 kvarn(MultiBatch R1 core): batched workspace bind + append/commit + ops test
665bbc6e/3b043428 docs: 127 R1-core proven; scheduler gap scoped
4e5cd082 kvarn(MultiBatch R1, WIP): TP2 batched runner + TextContext batched branch
2589c553 kvarn(MultiBatch R1): 2-lane batched decode runs end-to-end — token-exact until first page commit
9a9431b4 kvarn(MultiBatch): capture-replay oracle + exhaustive logging + CPU-ref probe
9bdb7247 kvarn(MultiBatch WIP): slot hygiene experiment + output dumps
<HEAD>   logging expansion (ATTEND/MB-MAP/COMMIT lane+row attribution) — uncommitted or latest push
```

Key new files: `tests/slice4_kvarn_bench.cu` (batched bit-identity + timing + capture
replay), `tests/kvarn_batched_ops_test.cu` (batched append state proof),
`tests/multi_gpu/tp2_batched_decode.cpp` (2-lane driver + order test),
`tests/kvarn_decode_cpu_ref.h` (imported CPU oracle).

---

## 9. How to reproduce everything

```bash
cd /home/intel/ninfer/worktrees/wo-kv-uniform/build

# Unit + bit-identity battery (GPU free; ~2 min)
cd tests
./ninfer_slice4_kvarn_bench                 # BATCHTEST + timing-sweep gate
./ninfer_kvarn_batched_ops_test             # batched append state proof
./ninfer_slice4_kvarn_test && ./ninfer_kvarn_gqa_test
./ninfer_kvarn_materialize_oracle_test && ./ninfer_slice4_kvarn_dequant_test

# Batched 2-lane decode: correctness + order test (needs both GPUs free; ~2 min)
NINFER_MB_SEQ_FIRST=1 ./ninfer_tp2_batched_decode_test \
  --artifact /home/intel/models/qwen3_8_27b.ninfer --tokens 32
# expect: RESULT: PASS (both lanes MATCH)

# Reproduce the open bug (default order; ~2 min)
./ninfer_tp2_batched_decode_test \
  --artifact /home/intel/models/qwen3_8_27b.ninfer --tokens 32
# expect: FAIL lane 0 divergence at 16; lane 1 MATCH

# Kernel-input capture at step 16 + chain-oracle replay
NINFER_MB_CAPTURE=1 NINFER_MB_CAPTURE_DIR=/tmp/mbcap \
  ./ninfer_tp2_batched_decode_test --artifact /home/intel/models/qwen3_8_27b.ninfer --tokens 32
NINFER_MB_CAPTURE_DIR=/tmp/mbcap ./ninfer_slice4_kvarn_bench   # replay: expect MATCH

# Model-level decode guard (needs port 8091 free; ~10 min)
cd ../.. && bash tools/bench/decode_guard.sh --all-cache-type <tag>
```

Env switches (all default-off, production-inert): NINFER_MB_DBG (logging),
NINFER_MB_CAPTURE(+_DIR), NINFER_MB_SEQ_FIRST, NINFER_MB_SEQ_STEP,
NINFER_MB_SEQ_ATTEND, NINFER_MB_SKIP_APPEND, NINFER_MB_FORCE_B1.

---

## 10. Takeover session (2026-09-01, afternoon) — THE BUG IS SOLVED

**Outcome**: 2-lane batched decode is token-exact vs the shipped path for both lanes at
32/96/128/256 generated tokens, both run orderings, including a >64-token prompt whose
PREFILL commits multiple pages. Aggregate throughput 1.14× (32 tok) → 1.69× (256 tok).
Unit battery green; decode_guard tracking baseline (74.6 vs 74.8 @10k, 71.0 vs 71.1 @25k).

### 10.1 The repro had moved

The step-16 repro (§6) no longer triggered: with the default test prompts (5 and 8 tokens),
the first committed-page attention for lane 0 is at generated index **63**, not 16. The
`--tokens 32` run passed; `--tokens 128` failed deterministically at 63. All step-16-gated
probes and the capture were silently capturing pre-commit steps — every §5 conclusion built
on them was stale. (Probe/capture steps are now env-driven: `NINFER_MB_CAPTURE_STEP`.)

### 10.2 The external review's three claims — all verified true

1. **ATT-READ vs COMMIT hash mismatch was a probe artifact**: the probe read
   `block_tables[0..1]` (lane 0's logical pages 0,1 — identity values) as if they were
   per-lane ROW pointers, hashed 16 bytes vs the commit's 8192, and fired at step 16
   (before any commit existed). The "stale view of page 0" hypothesis (§6) is dead.
   The probe now resolves each lane's physical page through its own row and checksums
   the full 8 KB + full 1152-float scale row, per layer, rank-tagged.
2. **The capture-replay "0/3072 MATCH" was false**: `replay_test` hardcoded
   `scale_page_stride=1152` while the runtime uses `1152*n_layers=18432`. Both kernels
   read identical wrong scales → agreed with each other, proved nothing. Fixed to use
   the captured scalars (stride, physical_pages=130, cap, tail_batch_elems=32768,
   per-lane tail counts).
3. **CPU oracle bugs**: tail-K indexed `[d]` instead of `ti + 64*(d + 256*head)`; the
   comparison looped 3072 elements against a 256-element vector (OOB reads → the 2.4e37
   outliers); code-plane offsets missed the `[phys][head]` stride. Fixed; the oracle now
   AGREES with both GPU kernels on the real step-63 inputs (0/256 out of tol).

### 10.3 Root cause #1 (batched lanes): the identity shortcut

`physical_page_id()` shortcut `if (block_table.ne[0] <= k_pages.ne[3]) return logical_page;`
was justified by "TP2 single-sequence tables are identity" — but batched lane rows are
NOT (lane 1's row is {65,66,...}), and the row view's ne[0]=65 ≤ pool 130 made the
shortcut fire anyway. Debug output was unambiguous: `[COMMIT] lane=1 row0=65 phys=0` —
lane 1's page-0 commit wrote LANE 0's physical page. (Interleaved with lane 0's own
commit at a later step, this produced the cross-lane corruption.)

### 10.4 Root cause #2 (the real one): the shortcut is unsound for the SHIPPED path too,
###     and the batched runner leaked its view binding

After fixing #1 (force-lookup flag), lane 0 still diverged at 63. The decisive experiment
was a **three-way ground-truth diff**: fresh-process sequential run (`tp2_decode --mtp 0
--kv-dtype kvarn`) vs in-test sequential reference (runs AFTER the batched run) vs batched:

| pair | result |
|---|---|
| fresh-truth vs batched | **identical, all 128 tokens, both lanes** |
| fresh-truth vs in-test-seq | diverges at 63 (lane 0) |

The batched path was CORRECT; the sequential REFERENCE was the contaminated one. Two
compounding causes:

1. **Runner leak**: `run_tp2_requests_batched` rebinds `TextContext::kv_` per lane
   (`set_kv_view(lane_views[b-1])`) and never restores it. The shipped single-seq path
   never rebinds `kv_` (it assumes row 0 forever), so every post-batch sequential run
   decoded against lane 1's stale block-table row. Fixed: the batched worker now restores
   `set_kv_view(st.kv_view)` + `set_linear_state_slots(0, 1)` on exit.
2. **The shortcut was always a time bomb**: any sequential run whose pool pages are
   recycled out of order (i.e., any second request in a long-lived server, which is
   exactly what the in-test reference was) gets a non-identity row → commit writes
   logical, kernel reads real → silent page misdirection. Instrumented proof: 128
   `[PHYS-SHORTCUT-WRONG] logical=1 shortcut-phys=1 real-phys=66` events.

### 10.5 The fix (final shape)

- `PagedKVLayerView` gains `const std::int32_t* host_block_table` — the owning
  `PagedKVAllocation::page_ids()` host mirror, read lazily in `layer_view()` (immune to
  vector growth). `execution_view` plumbs the allocation pointer through.
- `physical_page_id()` resolves through the host mirror when present (zero syncs, prefill
  hot path unchanged) or a D2H read of the actual table entry otherwise. **The extent
  heuristic is deleted.** The interim `force_block_table_lookup_` flag is removed.
- Batched runner restores single-seq bindings on exit.
- Ops test gains a **placement assertion**: every committed page must be non-zero at the
  physical page its table row points to. This closes the vacuous-pass hole: the old test
  compared batched vs single-lane, and BOTH were misdirected identically, so it passed
  while the feature was broken. Same lesson as §10.2.2 — an oracle shared by two paths
  cannot see a bug both paths have.

### 10.6 What "should be doing" vs "actually doing" — the recurring failure mode

Every bug in this arc is the same shape: **a component trusted a cached/derived view of
the block table instead of the table itself** (commit shortcut: "extent implies identity";
runner leak: "row 0 forever"; probes: "elements 0..1 are the rows"; replay: "1152 is the
stride"). The table is the single source of truth for logical→physical; anything that
derives from it must either read it or prove it cannot have changed. The placement test
and the three-way diff are the durable artifacts of this session — keep both.

### 10.7 Next steps (supersedes §7 items 1–4)

1. ~~Fix probe/replay~~ DONE. ~~Stale-view hunt~~ moot (probe artifact).
2. GDN-slot experiment: no longer needed for correctness; revisit only for perf.
3. MTP batched verify prototype (owner-requested) — unchanged plan (§7.5).
4. Engine scheduler (§7.6) — unchanged plan.
5. Before merge: `run_ci.sh --full`, byte-diff A/B vs shipped, and re-run the
   long-prompt (≥2 pages/lane) + 3-lane cases; the 1.69× @256-tok aggregate is the
   headline number to beat with the scheduler.
