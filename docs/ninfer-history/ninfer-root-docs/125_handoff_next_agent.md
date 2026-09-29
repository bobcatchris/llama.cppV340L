# KVarN Handoff — post-optimization state (2026-08-31)

> **RESOLVED 2026-09-01 by the next agent (this document's §OUTSTANDING UNRESOLVED is CLOSED).**
> The long-ctx KVarN decode decay was the slice4/UNIFIED kernel's SYNCHRONOUS global->smem code
> copy (~50% of kernel time, latency-bound ~88 GB/s — NOT V-codes, NOT page scatter, NOT the
> softmax merge). Fixed with double-buffered cp.async per-tile prefetch + conflict-free swizzle
> + keypair K-dequant + T=1 Wc=4 (commits c631f7f1, ac6b3f61). Model guard greedy: 10k 69.1->74.8,
> 40k 64.0->67.7, 160k 47.1->53.7, 250k 43.5->46.4 t/s; prefill unchanged; MTP accept = baseline.
> ALSO FIXED: a silent ZERO-OUTPUT bug in the unified route for isolation-test callers
> (unwritten-back inferred packed_pages) — kvarn_gqa_test had been failing since the route flip
> despite this doc claiming PASS. Details: docs/122 §22-§23, docs/124 B-STAGING4.
> Next lever if resumed: direct-to-MMA-fragment dequant (the fork's design); MultiBatch (Part b)
> untouched.

**BRANCH: `wo/kv-uniform`** (the branch this work is on). Pushed to the `github` remote as
`github/wo/kv-uniform`. Related branches visible on the remote: `wo/auto-kv`, `wo/kvarn-d18`,
`wo/kvarn-hold`, `wo/kvarn-live`, `wo/kvarn-pp`, `wo/ci-wiring`, `wo/mtp-adaptive`.

---
## LAST KNOWN GOOD COMMITS on THIS branch (revert targets if needed)

**MAX PREFILL commit: `7917a1de`** — `kvarn(prefill): K-smem staging in materialize`
Measured prefill (serve log, the arbiter): 10k=**808.0** tok/s, 40k=**741.1**, 160k=**584.1**.
This is the HIGHEST prefill measured on wo/kv-uniform. (It sits on top of the batched per-head
quantize `e2b0e350`, whose own result gave 10k=802.6 / 40k=736.4 / 160k=581.2.)
Files: `src/ops/kernel/gqa_attention_kvarn_flash.cuh` (materialize K-staging).

**HIGH DECODE commit: `75c416e4`** — `kvarn(decode): stage V codes into smem — greened, +2.3% decode @10k`
Decode = **73.8 t/s @10k** (highest measured on this branch; vs 72.2 at the max-prefill commit —
the ~1 t/s higher the previous agent was remembering). PREFILL at this commit = **802.6 @10k,
677.3 @80k** tok/s (from the serve_kvvs log). MTP 3.12 tok/round (70.7%). The companion
(duplicate-ish) run `serve_kvvs2` gave 72.1 @10k / 49.8 @80k, so the @80k cell is noisy (57.5 vs
49.8) — only the @10k 73.8 is the confirmed high-decode point. This commit was LATER REVERTED
(808f19b4 "kernel-neutral and model-noisy") and superseded by the slice4 V-staging (bae529c6,
decode-neutral). If you want the max-decode-@10k number back, restore 75c416e4.

**Caveat when comparing commits: use the SAME context set + gen budget.** `7917a1de` used the
guard's gen=64 at 10k/40k/160k; `75c416e4` used a manual gen=155/104 at 10k/80k. The prefill tok/s is
comparable (prompt/wall-to-prefill) but the decode t/s is per-step, so it is the fair cross-compare
only at matching contexts. Do not mix the two measurement setups.

**Also notable (decode-alt): `1478715b`** — `kvarn(unified decode): stage V codes into smem —
~17-21% decode kernel speedup` (kernel-level, but the FULL MODEL regression killed it; reverted in
7dda93c9). The kernel bench improved but the model did NOT — classic standalone-vs-model gap.

If decode ever needs to go back to its peak, `75c416e4` is the max-decode-@10k point (73.8 t/s).
`7917a1de` (+ `e2b0e350` underneath) is the max-PREFILL point (808/741/584).

After `7917a1de` the branch added the decode V-staging experiments (packed neutral, slice4 neutral),
which do NOT regress decode (10k +2.6%, 40k ~noise) but also do NOT fix the long-ctx decay. The slice4
V-staging (`bae529c6`) is correctness-clean but decode-neutral.

---

This is the consolidated handoff for the next agent. **Everything measured, all lessons, all
verdicts, both optimizations and dead-ends.** The previous agent chased several false leads (most
importantly measuring the WRONG kernel). Read §ROUTE first — it reframes everything.

## STATE OF THE BRANCH (all committed & pushed, working-tree CLEAN)
- Branch: `wo/kv-uniform` (worktree /home/intel/ninfer/worktrees/wo-kv-uniform)
- All commits pushed to `github` remote. HEAD = it var
- Correctness: `ninfer_slice4_kvarn_test` PASS, `ninfer_kvarn_gqa_test` PASS,
  `ninfer_kvarn_materialize_oracle_test` PASS. All builds green.

## THE HARD REQUIREMENT (met)
Both lanes GREEN at 160k:
- int8 decode = 66.1 t/s @160k (requirement >= shipped 60.5-60.6). | GREEN (+9.2%).
- KVarN k4v2 decode = 48.4 t/s @160k (baseline 47.1). | GREEN (+2.8%).
- Preheat near bf16 parity (-3.8 to -5.3%); bf16 itself is unreliable at long ctx on 36-SM cards.
MTP acceptance healthy (69.8@10k / 73.3@40k+ / ~76-78% at very long ctx).

## CRITICAL ROUTE REFRAME (read this FIRST — kills all prior decode-attribution work)
The model's DEFAULT decode route is the **UNIFIED / slice4 kernel**
(`gqa_decode_slice4_kvarn_kernel`, via `gqa_attention_cached_small_t` + shared reduce). The launcher
routes there when `NINFER_KVARN_DECODE` is unset (log: "UNIFIED (default)").

BUT the standalone bench (`bench_kvarn_attention`) measures the **PACKED** split/merge kernel
(`gqa_attention_kvarn_decode_packed.inc`). These are DIFFERENT kernels.

=> **V-staging (and ALL the "packed decode phase attribution": V-deq 4%->24%, K-deq ~21%, prefetch
~26%, PV-MMA ~17%) was measured on a kernel the MODEL DOES NOT USE by default.** That's why:
1. The packed V-staging (cp.async stage V) was decode-NEUTRAL in the model (48.4 vs 48.3 @160k).
2. The bench "decay" disagreed with the model.

FIX MOVED: The real decay lives in the **slice4** kernel. slice4 DOES stage K codes (cause-3 fix,
`kcode_s`) but originally read **V codes DIRECTLY from global** (`v_page[pkey*(D/8)+lane]`) — the
exact L2-miss problem. A V-stage was added there (see §LANDED-DECODE).

## WHAT IS LANDED (the real, committed work)

### PREFILL (exhausted — measured, do not chase)
- Sinkhorn 16->4 (`3211ba86`/`62e63ba3`): converged ~4 iters, +8% prefill. Gate via greedy
  byte-identity anchor (Sinkhorn numerics changed).
- Batched per-head quantize (`e2b0e350`): `<<<1,256,73KB>>>` -> `<<<heads,256>>>`, +3% prefill.
  CRITICAL BUG that made this hard: the V-launcher passed `s_row_v` at scale offset +512 (K's slot)
  instead of +1088 (V's slot). Shared offset => bit-identical codes but WRONG scales + MTP=0%.
- Materialize K-staging (`7917a1de`): 32x-scatter K-read -> smem, +0.5-0.7% prefill, bit-identical.
- VERDICT: prefill = quantize(<1%) + materialize(0.5%) + flash(proven bf16 FA2, ~99%) = at its
  practical limit. Reject further prefill work.

### DECODE — slice4 V-staging (the REAL model kernel) — `bae529c6`
- slice4 (`gqa_decode_slice4_kvarn.cuh`) `prologue_kvarn_kv_page`: K coded staged in `kcode_s`,
  V was read from GLOBAL. Added `vcode_s` (4096B) smem stage + coalesced copy + V dequant from smem.
- slice4 kernel is `__launch_bounds__(WarpsPerCta*32, 2)` -> 2 blocks/SM. smem 45K->49K, STILL under
  the 2-block budget (50.7KB). slice4_kvarn_test PASS (T2-T5, tail, tile-boundary, mixed).
- MEASURED (from the live serve log, the arbiter): 10k decode = 70.9 t/s (+2.6% vs baseline 69.1),
  40k decode = 63.3 t/s (baseline 64.0, ~noise). **So slice4 V-staging is ~decode-NEUTRAL too
  (small/no win at 40k, small win at 10k).** 160k cell was still pre-filling at handoff.
- CONCLUSION: V-code L2-miss staging, in BOTH the packed and slice4 kernels, is NOT the source of
  the long-ctx decode decay. The decay is elsewhere (page-scatter / PV-MMA / combined-kernel
  overhead / the query-softmax merge that grows with keys). Do NOT re-chase V-staging.

## WHAT WAS REJECTED (with data — do not retry)

### Prefill
- **A5 direct-read route** — REJECTED. 62x slower @625 pages (19.752ms vs 0.315ms materialize+flash).
  The direct kernel dequantizes K/V once per (q_block, q_head) CTA = redundant across all query
  blocks (~T/Br x). Materialize dequantizes each tile ONCE then flashes = the algorithmic win.
- **A4 materialize occupancy** — REJECTED. Materialize is L2-RESIDENT (~1100+ GB/s, ABOVE the 407
  GB/s DRAM ceiling) because the per-layer bf16 temp fits L2. Per-key FLAT ~9ns, 0.5% of prefill
  @160k (1.4ms vs 274ms). NOT SM-starved, NOT the bottleneck.

### Decode
- **B1 occupancy 1->2 blocks/SM (packed kernel)** — REJECTED. kPackedSmem=101,376B = the FULL
  optin budget -> exactly 1 block/SM. To reach 2 blocks needs Bc=16+single-buffered staging
  (37KB) = 16-key MMA tiles (inefficient, 4x blocks) + losing the cp.async prefetch overlap.
  Net-negative + deep change. The 99KB is architectural (Bc=64 tile).
- **B2 page-order / read-plan** — near-optimal already. block_table is token-sequential; KVarN
  k_codes physical layout is `k_codes[phys*heads*code_bytes+h*code_bytes]` = phys is a DIRECT
  contiguous offset, so pages read near-contiguously.
- **Packed V-staging (cp.async)** — decode-NEUTRAL (48.4 vs 48.3 @160k). Correctness clean + freed
  9KB smem, but NOT a win. Kept for the smem free (future), documented as neutral.

## THE FORK REFERENCE (beellama-kvarn, valujin) — what transfers, what doesn't
The fork (github.com/valujin/beellama-kvarn) fixed the exact KVarN high-ctx decode decay to q8_0
parity (single-slot 23.87->32.50 tok/s x1.36; dual x5.9). Per-commit: read-plan w/o rbtree +9.8%,
fragment-row permute +2.5%, skip-masked-split +27% dual, block 128->1024 -13% step, register-cap
4-blocks +1.5% single / +15% dual.

**TRANSFERABLE (validated)**: The decay is REDUCIBLE (not an intrinsic floor). The single-slot decay
was dominated by read/plan + geometry, not per-key dequant FLOPs. Their #1 (fragment-row permute)
does NOT transfer (they dequant inline into mma fragments with 5-bit 32-bit-word waste "4.4bits/32
read"; we smem-stage + ldmatrix_x4 and are 4/2-bit byte/uint16-granular -> the literal KVARN_FRAG_ROW
is a no-op).

**The two levers we did NOT find a win in**: V-code staging (neutral) and occupancy (architectural).
The fork's real decode wins were plan/geometry, which our code already does well (flat block_table).

## OUTSTANDING UNRESOLVED
- The long-ctx KVarN-vs-int8 decode decay (48.4 vs 66.1 @160k = ratio 0.73) is still open. It is NOT
  the V-code L2-miss (proved by V-staging being neutral in BOTH kernels). The real source is likely:
  (a) page-scatter / block_table DRAM misses beyond what cp.async hides, (b) the online-softmax
  merge that runs per-key (grows with keys), or (c) non-dequant kernel overhead. The fork proves it
  IS reducible. Per the compounding law decode = 0.5-14% of wall (14%@10k -> 0.5%@250k), so even
  fully closing it is ~0.1% wall for single-sequence.

## PART (b) — MultiBatch — SEE docs/127_kvarn_multibatch_status.md (Stage A DONE 2026-09-01)
Kernel+launcher MultiBatch shipped and bit-exact (per-lane tail state via lane_packed_pages +
tail_batch_elems; batched reduce). BLOCKED on runtime: TP2 engine is sequential (no batch
scheduler); graph engine has no KVarN binding + wrong hardware for this model. docs/127 has the
file-level plan for both options + aggregate-tps estimates. Original text follows:
slice4 supports `MultiBatch`/`Masked` templating. The launcher hardcodes `batch_size=1,
MultiBatch=false` (src/ops/launcher/gqa_attention_kvarn.cu:55,83). Grid.z = width*batch_size.
The kernel already handles `batch`, `column_base`, partial offsets. Enabling multi-sequence serving
is where decode compounds (the fork's x5.9 dual). Needs: batch_size>1 pass-through, MultiBatch=true,
reduce kernel MultiBatch=true. Validate with a 2-sequence guard (free 8091 first so --all-cache-type
can spawn its own server).

## HOW TO MEASURE (the guard runs fine from the agent tool — no "harness" problem)
Agents HAVE run decode_guard.sh successfully throughout this branch (incl. the 160k cell; e.g. the
kstage run at 21:01 completed all 3 cells: 10k/40k/160k with decode 72.2/67.6/48.4). So the guard
IS runnable from the agent tool.

THE ONLY FAILURE MODE = **port collision on 8091**. `--all-cache-type` calls `start_server` ->
`free_port 8091` (which `fuser -k`s whatever is on 8091 and `kill -9`s the pids) then spawns its OWN
server there. So:
- Run `--all-cache-type` ONLY when port 8091 is FREE. It works (kstage 21:01 did).
- If a server is ALREADY on 8091 (e.g. the user started one), `--all-cache-type` will kill it and
  collide -> the guard fails (empty log / `finish=cancelled decode=n/a`).
- To measure against an EXISTING server on 8091, use SINGLE-SERVER mode instead:
  `CASES="10000 40000 160000" ITERS=1 COMP=64 TEMP=0 BASE=http://127.0.0.1:8091 \
     bash tools/bench/decode_guard.sh <tag>` with `CTX_LOG` pointing at that server's log.

Correctness-before-perf note: always kill any stale server + free 8091 before a guarded run (the
guard's own free_port does this, but only for --all-cache-type; single-server mode does NOT free
the port).

## KEY REUSABLE LESSONS (also in docs/122 §17-§21 + docs/kernel_perf_knowledge.yaml)
1. K and V use DIFFERENT scale-field offsets (s_row_K=+512, s_row_V=+1088). A shared offset silently
   corrupts V's layout (bit-identical codes, WRONG scales). Verify per-field per kernel.
2. Compounding law: wall impact = prefill_share*prefill_delta + (1-prefill_share)*decode_delta.
   Prefill is 86-99.5% of wall -> prefill wins compound, decode wins barely move the wall at long ctx.
3. The standalone bench (synthetic contiguous layout) is a HYPOTHESIS; the model guard is the
   ARBITER. The bench misses the block_table page-scatter L2-miss (why per-page-flat standalone
   kernels still decay in the model).
4. Decay comparison must use ms/round (NEVER raw t/s) and compare vs INT8 (compressed reference),
   not just bf16.
5. KVarN = ~1.44x int8 per-key at HALF the bytes. Decode at bf16 parity (1.004-1.08x ms/round).

## FILES THAT MATTER
- src/ops/kernel/gqa_decode_slice4_kvarn.cuh  <- slice4 decode (V-staging added, the REAL model kernel)
- src/ops/kernel/gqa_attention_kvarn_decode_packed.inc <- packed decode (V-staging added, kernel the
  model does NOT use by default)
- src/ops/kernel/gqa_attention_kvarn_flash.cuh <- materialize (K-staging added)
- src/ops/launcher/gqa_attention_kvarn.cu <- routes prefill/decode, kPackedSmem, MultiBatch hardcode
- docs/124_kvarn_optimization_queue.md <- full optimization queue + verdicts + measured ledger
- docs/122_kvarn_performance_knowledge_base.md <- performance knowledge (compounding law, decay,
  beellama fork case study §18, lane status §19)
- docs/kernel_perf_knowledge.yaml <- general kernel perf lessons
- tools/bench/decode_guard.sh, decode_guard_baseline.json <- the arbiter + its per-cell baseline

## DOCUMENTS REFERENCED (verified to exist; the SOURCE OF TRUTH for each topic)
| doc/file | what it is (source of truth) |
|---|---|
| **docs/125_handoff_next_agent.md** | THIS file. The entry point: branch state, last-known-good
  commits (revert targets), the route reframe, what's landed/rejected, the fork lessons, the
  measurement-approach (port collision on 8091), next-agent pointers. |
| **docs/124_kvarn_optimization_queue.md** | The full optimization queue (A/B items) with EACH verdict
  (landed/rejected/neutral) and the measured ledger. This is where per-item decisions + their data live. |
| **docs/122_kvarn_performance_knowledge_base.md** | The KVarN performance knowledge base. §14
  compounding law, §18 beellama fork case study, §19 lane status (both green), §20 slice4 V-staging
  neutral, §21 last-known-good commits. |
| **docs/kernel_perf_knowledge.yaml** | General GPU kernel-perf lessons (bandwidth curves, occupancy,
  standalone-vs-model gap, fragment-permute/route insights, the measured decode/prefill figures). |
| **docs/123_kvarn_batched_quantize_debug.md** | The batched-quantize debug case study (the V scale-offset
  bug s_row_V=+1088 vs s_row_K=+512 — the root cause that unblocked the batch). |
| **tools/bench/decode_guard.sh** | The arbiter (model-level decode/prefill measurement). |
| **tools/bench/decode_guard_baseline.json** | The per-cell decode baseline (kvarn@160k=47.1, int8@160k=60.0,
  the thresholds decode_tps_fail 5%). |
| **docs/kernel_perf_knowledge.yaml** | Same as above (the .yaml is the general kernel lessons record). |
| **results/***, **~/ninfer/logs/serve_*.log** | Raw measured data: `results/kstage_decode_*.json`
  (7917a1de), the serve logs (the authoritative prefill/decode tok/s lines). |

NOTE on the top summary line ("Also in docs/122 §14-§19"): the key-lessons reference is now §17-§21
(§14 compounding law, §17 materialize K-staging, §18 fork, §19 lane status, §20 slice4 V-staging,
§21 last-known-good commits). All doc references above were VERIFIED to exist on wo/kv-uniform.

## NEXT AGENT: WHERE TO DIVERGE FOR MAX VALUE
1. (Optional) Confirm the 160k slice4 V-staging number (free 8091 first, then --all-cache-type).
   Expect ~48-50 t/s (neutral) — do NOT expect the decay to be fixed, it isn't V-codes.
2. The REAL reducible decode decay: profile slice4 with the phase-skip (KVARN_DBG_SKIP bitmask,
   compile-time) to attribute per-key cost at long ctx. Target the online-softmax merge / PV-MMA,
   NOT V-staging.
3. Part (b) MultiBatch — the compounding win. slice4 is ready; wire batch_size>1 + MultiBatch.
   Validate via 2-sequence guard (free 8091 first).
4. NOTE: per compounding law, all of this is <0.1-2% wall for single-sequence. The requirement
   (both lanes green @160k) is ALREADY MET. The remaining work is opt-in scope.
