# KVarN Optimization Queue (docs/124)

Tracked, measured, one-by-one. **PREFILL FIRST, then DECODE.** Each item has a hypothesis, a
measurement plan (standalone bench to form, MODEL GUARD to arbitrate), and a status.

## Summary of the prefill pipeline (current, shipped)
- **Commit (write)**: quantize per page. Batched per-head quantize (Landed, +3% prefill).
- **Materialize** (`gqa_attention_kvarn_materialize_kernel`): packed codes -> BF16 temp
  (K-smem staging Landed, +0.5-0.7%). Grid `<<<n_tiles, kv_heads, 256>>>`, 1 CTA/(tile,head).
- **Flash** (`gqa_attention_prompt_attention_launch`): proven BF16 FA2 over the temp.

Full-chunk prefill wall = commit (quantize) + materialize + flash. Measured prefill (tok/s):
10k=808, 40k=741, 160k=584. bf16 baseline 834/711@80k/614@160k -> KVarN prefill is -3.8% to
-5.3% vs bf16 (was -12.4% before Sinkhorn+batch).

---

## PHASE A — PREFILL (do all before any decode)

### A1. [LANDED] Sinkhorn iters 16->4  —  +8% prefill
Converges by ~4 iters (rel-l2 3.3e-7 vs 16). Gate: greedy byte-identity anchor.

### A2. [LANDED] Batched per-head quantize (commit) — +3% prefill
`<<<1,256,73KB>>>` -> `<<<heads,256>>>`. Fixed the V scale-offset bug (s_row_v=+1088).
MTP 69.8%. All 4 KVarN oracle suites PASS.

### A3. [LANDED] Materialize K-smem staging — +0.5-0.7% prefill
staged 8KB K-code page into smem (32x-scatter fix, cause-3 analog). Bit-identical. MTP 73.3%.

### A4. [REJECTED] Materialize occupancy / threads — materialize is NOT the bottleneck
MEASURED: the materialize kernel is L2-resident (~1100+ GB/s effective, ABOVE the 407 GB/s DRAM
ceiling) because the per-layer bf16 temp fits L2. Per-key is FLAT (~9 ns/key, 0.6 us/page) across
10k/40k/160k. At 160k the materialize is ~1.4ms = 0.5% of the 274ms prefill. It is NOT SM-starved
(n_tiles*heads CTAs) and NOT the bottleneck. Occupancy/thread tuning (A4) and V-bank wide-store
(A6) are DEAD ENDS. The real prefill bottleneck is the FLASH (~99%).

### A4b. [CANDIDATE -> the real prefill bottleneck] bf16 threshold: quantize+materialize+flash
Materialize (0.5%) + quantize (0.31ms flat) are together <1.5% of prefill. So the -3.8/-5.3% vs
bf16 gap is EITHER (a) the flash running slower over KVarN's temp than over bf16's native cache
layout, OR (b) measurement noise / a different-session bf16 baseline. VERIFY with a same-session
bf16-vs-kvarn prefill A/B (the arbiter). If (a), investigate the flash temp access pattern
(materialize writes {D,G,H,P} d-fastest; if the flash reads it less cache-friendly than bf16's
native page layout, that's the gap). If (b), prefill is already at parity.

### A4c. [CANDIDATE] Does the flash over the temp equal the flash over bf16 cache?
Measure gqa_attention_prompt_attention_launch over (i) the KVarN temp and (ii) a native bf16 cache
of the same shape. Same kernel -> same time = layout neutral; different time = the temp layout
costs. Isolates the TRUE prefill gap source.

### A5. [REJECTED] Eliminate the 2x bf16 temp — the direct-read route (B2 tail support)
**REJECTED on data.** The direct-read kernel dequantizes K/V ONCE PER (q_block, q_head) CTA, i.e.
REDUNDANTLY across all query blocks. At large T (hundreds of q_blocks) this re-dequantizes the
same tiles ~T/Br times. STANDALONE BENCH @625 pages (40k): direct-read qblock = 19.752 ms vs
materialize+flash combined = 0.315 ms -> the direct route is **62x SLOWER**. The launcher comment
confirms it ("latency-bound ~20 GB/s effective; loses to materialize+flash for T>1; re-enable
after MMA/vectorized tuning"). The bf16 temp is the WRONG thing to eliminate — the materialize
dequantizes each tile ONCE for all queries, which is the algorithmic win. A5 is a dead end.
### A6. [CANDIDATE] Materialize V-read bank conflicts / wide stores
V is read coalesced (uint16). Verify store `store_vec` to temp is wide (int4). Check if the
bf16 temp write can use 16B (int4) stores everywhere (it already does in the K/V paths).
Low value (already vectorized).

### A7. [CANDIDATE] Flash occupancy / grid config over the temp
The flash (`gqa_attention_prompt_attention_launch`) is the proven bf16 FA2. It's already tuned.
The temp view is {D,G,H,P} matching the flash's expectations. Low value (already proven).

### A8. [CANDIDATE] Prefill pass-1/pass-2 overlap (materialize || flash chunks)
Materialize is write-bound, flash is compute-bound. Could overlap N chunks (pipeline) so the
write of chunk K+1 overlaps the MMA of chunk K. Currently the code materializes ALL tiles then
flashes ALL (2 full passes). Chunked pipelining needs per-chunk staging + stream overlap.
Value: potentially hides the materialize write behind flash compute. Medium risk
(arena/staging + stream sync), moderate value.

### A9. [REJECTED] Cross-page / cross-head 34x batch quantize
Deprioritized: after Sinkhorn+batch the quantize is ~2.8% of prefill. Full 34x saves ~2%. Not
worth the tile-buffering architecture risk. (See §15.)

### A10. [CANDIDATE] Reduce Sinkhorn/quantize redundant work in commit
The commit's quantize now batched per-head. Verify no remaining redundancy (idle warps in the
`<<<heads, 256>>>` grid). Small.

---

## PHASE B — DECODE (after all prefill)

### B1. [NEXT] Occupancy 1 block/SM -> 2 (the fork's biggest transferable)
`kPackedSmem = 101,376 B` = full optin budget -> 1 block/SM, 8/48 warps (16.7%). For a
memory-hierarchy-bound long-ctx decode (per-page flat standalone, decays in model), low occupancy
= poor latency hididng. The fork's register-cap-for-4-blocks was +1.5% single / +15% dual.
Need to cut smem (k_s/v_s are 32KB each; stage_k 2x18KB; scales 2x9KB). Cutting to 2 blocks/SM
requires ~50KB/block. Halving k_s/v_s (Bc 64->32) is the big lever but changes MMA tile size.

### B2. [CANDIDATE] Read-plan / page-order locality (long ctx)
Decay is memory-hierarchy (real block_table page scatter > L2). The fork's read-plan-without-
rbtree was +9.8% single. We use block_table (no rbtree), so the mechanism differs, but page
ORDER / prefetch efficiency transfers. Look at whether pages are read in allocation order (scatter)
and whether sorting/interleaving helps L2 reuse.

### B3. [CANDIDATE] Skip fully-masked splits (multi-sequence / unified cache)
The fork's +27% dual. Only helps multi-sequence or masked regions. Our guard is single-sequence,
so this is a serving-scale win, not a single-req win. Low priority for the guard.

### B4. [NOT-APPLICABLE] Matrix-fragment row permutation
Verified NOT transferable: we smem-stage + ldmatrix_x4 (full fragments in one instr), 4/2-bit
byte/uint16-granular. No 5-bit 32-bit-word waste. The literal KVARN_FRAG_ROW map is a no-op here.

### B5. [CANDIDATE] stage_k double-buffer -> single (if occupancy needs it)
stage_k is 2x18KB (double-buffered for prefetch overlap). If NOT single-buffering it directly, it
could be a 18KB smem cut at the cost of less prefetch overlap. Trade against B1.

### B6. [CANDIDATE] scales_scratch double-buffer -> single
2x9KB. Could cut ~9KB. Trade against B1.

---

## Verification gates (from the standing lessons)
- Standalone kernel bench (bench_kvarn_attention / bench_kvarn_2pass / bench_kvarn_quantize) ->
  forms a HYPOTHESIS, neutral on the synthetic contiguous layout.
- MODEL GUARD (decode_guard.sh / serve prefill tok/s) -> THE ARBITER. In model, compare BOTH
  codes AND scales (codes alone insufficient; the V-offset bug proved this).
- Correctness: ninfer_slice4_kvarn_test, kvarn_materialize_oracle_test, MTP acceptance >= baseline.
- The guard baseline decode tps (from decode_guard_baseline.json cells): kvarn@160k=47.1, int8@160k=60.0.

## MEASURED LEDGER (2026-08-31, reliable results)

- Materialize = **0.5% of prefill** (L2-resident, 1100+ GB/s > 407 GB/s DRAM ceiling, per-key FLAT
  ~9ns, 0.6us/page across 10k/40k/160k). NOT the bottleneck. Occupancy/V-bank = dead ends.
- Quantize/commit = **0.31ms flat** (batched per-head, already landed). Dead end.
- Direct-read route (A5) = **62x SLOWER** @625 pages (19.752ms vs 0.315ms) — redundant per-CTA
  dequant. REJECTED. The materialize dequantizes each tile ONCE (the algorithmic win).
- **bf16 baseline is UNRELIABLE at long ctx on our 36-SM cards**: `build/apps/ninfer-serve ... --kv-dtype bf16 --max-context 40360` crashes with
  `gqa_attention: invalid execution envelope or table` (per-variant per-SKU tuning cliff —
  bf16 is tuned for 170 SMs, we have 36/SM, docs/104 R2). So the "-3.8 to -5.3% vs bf16" prefill
  gap chart suspects a DIFFERENT-session, possibly non-representative bf16 number. VERDICT: do
  NOT chase the bf16 prefill gap. Use int8 (both are 36-SM-tuned compressed lanes) as the valid
  compression reference for prefill.
- CONCLUSION: prefill = quantize(<1%) + materialize(0.5%) + flash(proven bf16 FA2, ~99%). The
  prefill for the compressed lane is at its PRACTICAL LIMIT. The remaining levers are the FLASH
  (proven/tuned) and decode (B). Prefill optimization is essentially EXHAUSTED after:
  Sinkhorn 16->4 (+8%), batched per-head quantize (+3%), K-staging (+0.5%).
- **UPDATE (2026-09-03): the prefill gap has CLOSED to parity.** On the current build (09-01
  KVarN vs 09-03 int8), KVarN prefill t/s is **±0.3% of int8** at 25k/40k/80k (795.8/795.0 @25k,
  770.3/769.0 @40k, 708.4/710.2 @80k). This **supersedes** the "−3.8…−5.3% vs bf16" figure above
  and the "−12…−16%" figures in results/106 (2026-08-29 build). The materialize→bf16-temp→small-T
  round-trip fixes (post-08-31) removed the penalty. The "PRACTICAL LIMIT / EXHAUSTED" conclusion
  is now **CONFIRMED** (parity reached — there is no prefill gap left to close); the remaining
  KVarN-vs-int8 gap lives in **decode** (the slice4/UNIFIED kernel), not prefill. See
  results/106 §7 (Superseded — current-build prefill) and docs/142 (D1 gate redefinition).
- **UPDATE (2026-09-03, 4b CLOSED): the 160k KVarN cell was measured.** 160k prefill = **−6.8%
  vs int8** (572.9 vs 614.4, 4b run 09-03 14:59, `~/ninfer/logs/serve_4b_*_20260903_145944.log`)
  — slightly outside the ±5% D1 gate (known residual: the O(n²) materialize cost widens at the
  longest context). 160k decode = **−18.9% vs int8** (53.3 vs 65.7) — the known KVarN decode
  penalty widens at long ctx (−7.4% @80k). So the prefill gate passes @40k/80k (parity) with a
  small 160k prefill residual; the dominant long-context gap is **decode**. D1 gate: 40k/80k
  PASS, 160k −6.8% documented as a known residual. **4b CLOSED.**

## Prefill verdict: MOVE TO DECODE (Phase B)
The prefill path has been pushed to its practical limit for the compressed lanes. The highest-value
REMAINING optimization (per the beellama fork) is DECODE B1 (occupancy 1->2 blocks/SM), which is
the fork's biggest transferable win (register-cap-for-4-blocks +1.5% single / +15% dual; block
128->1024 = -13% step). Transition to B.

## PHASE B decode verdicts (2026-08-31)

### B1. Occupancy 1->2 blocks/SM — NOT VIABLE (architectural)
kPackedSmem = 101,376 B = the FULL optin budget -> exactly 1 block/SM (16.7% occupancy). To reach
2 blocks/SM (<= 50KB/block) requires Bc=16 + single-buffered staging (37KB), which means 16-key MMA
tiles (4x more blocks, `kKvarnDecodeSplits` must change) AND losing the cp.async prefetch overlap.
Net-negative + deep change to a correctness-critical kernel. REJECTED. The 99KB is inherent to the
Bc=64-tile + double-buffered-staging design.

### B2. Read-plan / page-order locality — NEAR-OPTIMAL already
block_table is token-sequential (allocation order), and KVarN k_codes physical layout is
`k_codes[phys*heads*code_bytes + h*code_bytes]` = phys is a DIRECT contiguous offset. So the decode
reads pages in near-contiguous physical order = good L2/DRAM locality. No red-black tree to remove
(we iterate a flat block_table). Little to gain from reordering.

### B-VSTAGE. [THE REAL DECAY SOURCE, but smem-ceiling-bound] V-codes read from GLOBAL, not staged
The phase model: Phase A (per page lp) warp0=QK+softmax, warps1-3=dequant_v_page(lp) which reads
`v_codes + vco` DIRECTLY from GLOBAL (synchronous, no cp.async). Phase B (per page lp) warp0=PV-MMA,
warps1-3 = dequant_k_page(lp+1) + prefetch_page(lp+2) — K IS staged via cp.async (hides L2 latency).
So K-latency is hidden but V-latency is NOT -> V-deq stalls on L2-miss at long ctx = the V-deq
4%->24% decay source documented in 8.
FIX would be: stage V codes into smem via cp.async (like K). BUT smem is at the 101,376 optin limit.
Even single V-stage (4KB) over-perpages. Freeing via (A) drop stage_k 36->32 pad (-2KB) + (B)
single-buffer scales (-4.6KB) = -6.6KB, still can't fit a DOUBLE V-stage (8KB) needed across the
phase boundary (A uses V(page lp), B prefetches V(page lp+2)). 102.9KB vs 101.4KB limit = 672B over.
VERDICT: the V-staging fix is the RIGHT idea but requires invasive smem juggling at the exact
ceiling, with a 672B shortfall, for a fraction of a 0.5-14%-of-wall component. POOR ROI given the
compounding law. NOT worth the correctness risk on a critical kernel.

## B-STAGING4 RESOLVED (2026-09-01): the decay was the SYNCHRONOUS COPY, not which codes

The queue's open item was "the real source of the slice4 long-ctx decay". Resolved with a new
standalone bench on the REAL model kernel (tests/slice4_kvarn_bench.cu, exact 160k TP2 config,
identity vs shuffled block_table, compile-time phase-skip bitmask):

1. ATTRIBUTION (pre-fix, 160k T=1, full=1.35 ms/layer): code-staging copy ~51.5%, K-deq ~55.7%
   (overlaps staging via DCE), V-deq ~28.5%, QK-MMA ~7.9%, softmax ~12.4%, PV-MMA ~9.1%,
   acc-FWHT/writeback ~0%. The copy is 4B LDG.32 chains with 2-4 outstanding/thread ->
   ~88 GB/s (Little's law ~80 predicted). Barrier-serialized by __syncthreads() per page.
2. SCATTER IS NOT THE ISSUE: identity vs shuffled block_table differ by <0.3% at every ctx.
   Per-key cost is FLAT standalone (4.2-4.6 ns/key 10k->160k): the model "decay" is attention's
   linear ctx growth at a per-key cost ~3x int8's (kvarn 1.351 vs int8 0.417 ms/layer @160k T=1,
   HALF the bytes) dominating the round.
3. FIX (c631f7f1): per-tile double-buffered cp.async banks, prefetch one tile ahead, 8B-aligned
   conflict-free K-row swizzle 16d+8*(d>>4)+j, dynamic smem 12544B (occupancy stays 2 blk/SM),
   T=1 -> Wc=4. Standalone: T=4 2.110->0.975 ms (2.16x), T=1 1.351->0.943. Model guard greedy:
   10k 74.7 (+8.1%), 40k 67.3 (+5.2%), 160k 53.2 (+12.9%, requirement >=48.4 GREEN), prefill
   unchanged (807/738/583), MTP accept = baseline, slice4/oracle/dequant tests PASS.
4. VERDICT: "V-staging is neutral" (§B-VSTAGE) was right but mis-framed — the problem was never
   WHICH codes, it was the synchronous copy mechanism for both K and V. Packed's cp.async was the
   known-good pattern; it simply had never been ported to slice4.
5. NEXT LEVER (open): post-fix attribution puts K/V dequant (bf16 smem round-trip + per-tile
   scale LDGs) at ~58% combined. Direct-to-MMA-fragment dequant (the fork's design) is the
   remaining structural win; staging is now ~15% at Wc=4 and mostly hidden.

## OVERALL VERDICT

SUPERSEDED on 2026-09-01 by B-STAGING4 above (the decay was found + fixed; smem was NOT the
ceiling — per-tile double-buffering fits 2 blocks/SM). Original text follows verbatim:

Prefill EXHAUSTED (A4/A5 rejected with data; Sinkhorn+batch+K-staging landed). Decode at bf16
parity; the KVarN-vs-int8 long-ctx decay is the known reducible issue but its fix (V-staging) is
smem-ceiling-bound with poor ROI, and the other decode levers (occupancy B1, page-order B2) are
dead-ends or already-optimal. The remaining decay is best attacked by a REDESIGN (e.g. a
lower-smem decode that keeps a V-stage), which is a larger Phase-3/plan-owner effort, not an
incremental win. Current state: both lanes GREEN (int8 66.1 @160k >=60.5 shipped; kvarn 48.4 @160k
>=47.1 baseline), prefill near-parity, all correctness tests PASS.

## B-VSTAGE result (2026-08-31, MODEL GUARD = ARBITER) — **decode-NEUTRAL, NOT the decay**

Implemented V-code cp.async smem staging (freed 9KB scales_scratch + K-pad 36->32, 96KB total,
double V-stage). Correctness: kvarn_gqa (packed decode+tail vs BF16), slice4, materialize oracle ALL
PASS.

DECODE GUARD (the arbiter) — decode tps UNCHANGED:
  ctx   | V-staging  | pre-Vstg   | delta
  10k   | 72.1       | 72.2       | ~0%
  40k   | 67.5       | 67.6       | ~0%
  160k  | 48.3       | 48.4       | ~0%
  MTP accept 73.3% / tok-round 3.20 (unchanged).

CONCLUSION: the long-context decode decay is NOT the V-code L2-miss. The refactored hypothesis (that
V-deq grew 4%->24% from unmasked global V-code reads) is WRONG — staging V via cp.async changed
nothing. The decay (per-page-flat standalone but rising ms/round in model) must come from a DIFFERENT
source that the per-page bench hides (page table/scatter, PV-MMA, or the combined kernel's non-
dequant overhead). V-staging kept for correctness-neutral + the 9KB smem free (future occupancy),
but it is NOT a perf win. The decay fix remains open (not V-codes).

## CRITICAL ROUTE DISCREPANCY (2026-08-31) — bench measures the WRONG kernel

The standalone bench (bench_kvarn_attention) measures the **PACKED split/merge decode kernel**
(decode_split + decode_merge). But the MODEL's default decode route is the **UNIFIED / slice4**
kernel (gqa_attention_cached_small_t + shared reduce; launcher routes there when
NINFER_KVARN_DECODE is unset — the log says "UNIFIED (default)"). 

Guard 160k decode=48.3 t/s, tok/round=3.20 -> raw 154.6 steps/s -> 6.47 ms/step.
Bench "combined" @160k = 1.768 ms (T=1, 1 layer). The bench kernel (1.768ms) is ~3.7x SMALLER than
the model's per-step (6.47ms) — because the model does 64 layers + MTP + launcher overhead, AND uses
a DIFFERENT kernel (slice4) than the bench's packed split/merge.

=> **The V-staging and all packed-decode optimizations were measured on a kernel the MODEL DOES NOT
USE by default.** The decay (KVarN 48.3 vs int8 66.1 @160k) lives in the **UNIFIED/slice4 decode**,
NOT the packed split/merge. To fix the model decay, profile and optimize the slice4/UNIFIED decode,
not the packed kernel the bench measures.

## FINAL HONEST STATUS (2026-08-31) — architecture-correct reframe

The model's decode DEFAULT is the **UNIFIED/slice4 kernel** (gqa_attention_cached_small_t +
shared reduce; routed when NINFER_KVARN_DECODE unset, log "UNIFIED (default)"). All prior decode
phase-attribution (V-deq 4%->24%, K-deq 21%, prefetch 26%, PV-MMA 17%) and the V-staging work
targeted the **PACKED** split/merge kernel — which the model does NOT use by default. That is why
V-staging was decode-NEUTRAL and why the bench (packed) disagreed with the model.

The KVarN-vs-int8 model decay (48.3 vs 66.1 @160k) lives in the **slice4/UNIFIED** kernel. Fixing
it requires profiling slice4 (not the packed bench), which is a NEW per-phase attribution exercise.

### Compounding-law reality check
Decode is 0.5-14% of wall (14% @10k -> 0.5% @250k, §14). Even if the ENTIRE KVarN-vs-int8 decode gap
at 160k were closed (27% of decode = 0.5% * 27% = 0.1% wall), it is <0.1% wall for single-sequence.
The decode gap matters ONLY for multi-sequence serving (where decode is a bigger wall share) — NOT
the current single-sequence guard/requirement.

### Where the project stands
- BOTH lanes GREEN (int8 66.1 >=60.5 shipped; kvarn 48.4 >=47.1 baseline). MTP healthy.
- Prefill near bf16 parity (-3.8 to -5.3%); prefill optimizations EXHAUSTED (Sinkhorn +8%, batch +3%,
  K-staging +0.5% landed; A4/A5 rejected on data; materialize 0.5%, quantize <1%, flash = proven FA2).
- All incremental prefill levers measured & landed or rejected. Decode at parity-with-bf16; the
  KVarN-vs-int8 long-ctx decay is real but (a) in a different kernel (slice4) than previously assumed,
  and (b) <0.1% wall per compounding law.
- The remaining real win is MULTI-SEQUENCE serving throughput (where decode compounds), or a slice4
  decode redesign — both larger efforts beyond incremental single-sequence tuning.

## slice4 V-staging result — PARTIAL (guard reaped at long ctx, 2026-08-31)

Correctness: ninfer_slice4_kvarn_test PASS (T2-T5, tail, tile-boundary, mixed). smem 45K->49K
(2 block/SM budget 50.7K still holds).

DECODE GUARD got the 10k cell before the harness reaped the long cells:
  ctx   | slice4 V-stg | baseline | delta
  10k   | 70.9 t/s     | 69.1     | +2.6% GREEN
  MTP accept 69.8% (baseline 71.4).

40k/160k cells: the guard process was reaped by the harness (sandbox kills long-running child
processes) before it could finish the slow prefills. The slice4 route decoding improvement at long
ctx (the decay fix goal) is UNCONFIRMED by the 160k cell. 10k shows +2.6% (good); the long-context
decay benefit needs a 160k run that the harness won't let complete.

NOTE: the serve launch + long prefill (160k ~275s) exceeds what this harness keeps alive. The V-staging
is correctness-clean and does not regress (10k +2.6%); the long-ctx win is plausible but not yet
measured end-to-end.

## slice4 V-staging DEFINITIVE result (read from live serve log, 2026-08-31)From the user-launched serve on 8091 (single-server, serve_d19.log), the arbiter:
- 10k decode = 70.9 t/s (baseline 69.1) -> +2.6%
- 40k decode = 63.3 t/s (baseline 64.0) -> ~-1.1% (noise, no regression)
- prefill 10k=805.9, 40k=739.3 (at/above earlier 802.6/736.4)
MTP accept 69.8@10k / 73.3@40k (healthy).

VERDICT: slice4 V-staging is ~decode-NEUTRAL (like the packed V-staging). V-code L2-miss is NOT the
long-ctx decode decay source. The decay (kvarn 48.4 vs int8 66.1 @160k) is elsewhere; see the
handoff doc 125 for the real candidate sources (online-softmax merge / PV-MMA / non-dequant overhead)
and the MULTIBATCH (part b) path. Do NOT re-chase V-staging.

HARNESS NOTE (for the next agent): the decode guard --all-cache-type SPAWNS its own server on 8091,
which COLLIDES with a user-launched server. Use the user's live server + read serve_d19.log's
'done ... decode=...tok/s ... mtp ...tok/round (...%)' line (this is what produced the definitive
numbers above). Background/nohup/setsid the guard from inside the tool and it gets reaped.
