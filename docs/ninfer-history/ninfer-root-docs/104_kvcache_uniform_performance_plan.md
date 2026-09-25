# 104 — KV-cache uniform performance plan ("make every KV cache behave the same")

**Status:** PLAN — design only, no code. Depends on docs/103 (GDN-gate T=1, same disease class) and on the bug-free gate (D-21 closeout items, D-22/D-23 done ✅).
**Companions:** docs/102 §9/§13 (pipeline map + hardware audit), docs/50, docs/82/83 (KVarN code-space work), docs/103.

---

## 1. Goal

Today the `--kv-cache` flag **selects an algorithm**, not a storage dtype (docs/102 §9): BF16, I8, and KVarN each run a different attention kernel family with different numerics, different grid/smem/split tuning (each tuned to a *different* SM count), different prefill paths, different write-time lossiness, and different prefix-reuse mechanics. So per-variant decode t/s, prefill t/s, MTP acceptance, and determinism guarantees all differ, and no single tuning rule works across the RTX 5000 series (docs/102 §13.2/§13.3: KVarN decode tuned for 36 SMs, rope/I8/GDN/MoE tuned for 170 SMs).

llama.cpp and vLLM don't have this problem because **KV dtype is a storage detail**: one attention kernel reads whatever dtype, dequantizes to the compute dtype early, and does the same math; variants differ only in bandwidth + intrinsic quantization loss.

**Target ("uniform"):** for a given (context length, T, MTP on/off, greedy/sampling), across KV variants:

1. **Numerics:** identical kernel body → identical rounding. Variants differ *only* by the quantization loss of their stored values (intrinsic, bounded, measurable). Consequence: MTP acceptance differs by the storage-loss penalty alone — no algorithmic drift.
2. **Decode t/s:** one launch-config rule (grid/smem/splits from a per-SKU table). t/s difference = dequant compute + memory bandwidth only (predictable, bounded, no regime cliffs).
3. **Prefill t/s:** one flash kernel family, in-kernel dequant. No per-variant O(n²) materialize pass.
4. **Determinism:** same guarantees per variant — A/B byte-identical, A2 identity (MTP==plain), prefix-restore identity.
5. **Per-SKU scaling (5060/5070/5080/5090, all sm_120):** one dispatch table keyed on runtime `multiProcessorCount`; no hardcoded per-variant SM constants.

**Explicitly NOT a goal:** making KVarN lossless (impossible at 4/2 bit), making all variants the same size (VRAM is the whole point), or closing the D-21 residual width-4 attention-precision tolerance (Option C — separate, accepted).

## 2. Why they differ today (root causes, from docs/102 §9 + this review)

| # | Root cause | Where | Effect |
|---|---|---|---|
| R1 | Variant selects a different **algorithm**: KVarN decode = FWHT code-space QK + deferred V-FWHT epilogue + split/merge; I8 = s8 mma/SIMT; BF16 = bf16 flash | `gqa_attention_kvarn_decode_packed.inc`, `gqa_attention_decode_i8.cuh`, `gqa_attention_decode_bf16.cuh` | different rounding → different MTP acceptance; different perf shape |
| R2 | Each variant's kernel tuned to a **different SM count** with hardcoded constants | KVarN packed: `kKvarnDecodeSplits=54` ("3 full waves on 36 SMs", `gqa_attention_kvarn.cuh:827`), 99 KiB smem 1 block/SM; I8: `kMax=42×DecodeSplitScale` "one 170-SM wave" (`gqa_attention_decode.cu:53-58`); rope: `kLargeBlockWaveCapacity=1020=170×6`; GDN: `kTargetCtas=680`; MoE: 510 persistent | per-variant per-SKU perf cliffs; wrong wave count on any other SKU (54 splits on 170 SMs = 0.64 waves) |
| R3 | **Write-time lossiness only for KVarN** at 64-token commit boundaries (open tile pre/post-commit) | `gqa_kv_append_kvarn_and_commit` | per-variant acceptance + determinism behavior |
| R4 | **Prefill asymmetry**: KVarN = O(n²) materialize-to-bf16 then flash; I8 = dequant then flash; BF16 = flash | `gqa_kvarn_materialize_kernel` | per-variant prefill t/s; KVarN prefill cost grows with n² |
| R5 | **Prefix-reuse mechanics differ**: KVarN snapshot/restore dance (D-16) vs bf16/I8 exact open page | `kvarn_capture_prefix_snapshot` / `kvarn_restore_prefix_snapshot` | per-variant restore numerics + cost |
| R6 | **Same disease outside the KV cache**: token-count-dependent kernel *routing* splits numerics (D-21: GDN gate GemvPairedRows@T=1 vs SmallTSplit10@T>1) and perf (docs/103: −7% decode) | `bf16_gdn_gating_proj_plan.cpp` | proves the failure mode is routing/numerics-splitting, not KV-specific |

R6 is the key insight: the KV variants are just the largest instance of a general pattern — **any dispatch on (T, variant) that picks different kernels/accumulation orders creates both a numerics split and a perf split**. The fix is a general principle, not a KV patch.

## 3. Design: one kernel family, dequant prologue, canonical math

```
           ┌────────────────────────────────────────────────────────────┐
 gqa_decode<TKV>  (one template, one grid, one smem, one body)          │
   ┌──────────────┐   ┌──────────────────────────────────────────────┐  │
   │ prologue     │   │ body (identical for all TKV)                 │  │
   │ page → bf16  │──▶│ QK mma m16n8k16 bf16                        │  │
   │  BF16: copy  │   │ online softmax (fp32 acc, same order)        │  │
   │  I8: ×scale  │   │ PV mma m16n8k16 bf16                        │  │
   │  KVarN: codes│   │ output (same rounding, same o_proj feed)     │  │
   │  +scale+FWHT │   └──────────────────────────────────────────────┘  │
   └──────────────┘                                                    │
```

- **Dequant happens once, early, in smem** (the KVarN packed kernel already dequants in smem — the change is to *un-rotate V in the prologue* and delete the deferred V-FWHT epilogue + split/merge; PV then runs in the canonical body).
- **The body is the same code for all TKV** → same accumulation order, same rounding → numerics differ only by what the prologue reconstructed (the storage loss). MTP acceptance then differs by the storage-loss penalty *only*.
- **One launch config** per (T, ctx bucket, SKU) — not per variant. The per-SKU table (Phase 1) feeds all variants identically.
- **Prefill:** same family with the prompt route; the KVarN O(n²) materialize pass dies because each page is dequanted in-kernel exactly once per CTA that reads it (flash already reads each page once). R4 gone.
- **Write path / prefix:** storage stays per-variant (that's the point — different compression). R3/R5 stay but become *measured and gated*, not structurally special.

Costs we accept (intrinsic, not fixable): KVarN prologue is more compute than I8's (codes+scale+FWHT) and I8 more than BF16's (×scale). That is exactly the "dequant + bandwidth only" delta of goal 2 — the *shape* of the difference becomes uniform and predictable even though it is non-zero.

**KVarN decode residual (MEASURED, 2026-08-30, full matrix): the unified
KVarN small-T route (slice4, bf16 canonical body) is CORRECT (oracle 10/10,
token-identical to packed 6/6, A2 identity) but 1.3–2.8× SLOWER at long ctx
(greedy: 43.1 t/s @10k vs 60.8, 27.4 @25k vs 67.3, 20.2 @40k, 10.9 @80k).
Root cause (verified 2026-08-30, code-level): the shipped packed decode
kernel (gqa_attention_kvarn_decode_packed.inc) avoids the per-key Hadamard
almost entirely via three conventions + pipelining. LIVE numerics
(verified against the actual code — the kernel's OWN header comments at
lines 213-219 are STALE and describe the REMOVED variant; see the comment-
fix task below): Q tile = FWHT_norm(Q) staged ONCE before the page loop
(kvarn_mma_warp_fwht, register+shuffle warp-level, no smem scratch); K tile
= UNPACK ONLY — the codes already decode to FWHT_norm(K), so score =
MMA(FWHT(Q), FWHT_norm(K)) = Q·K by orthogonality (zero per-key K Hadamard,
kvnorm=1.0); V tile = UNPACK ONLY, NO Hadamard, vvnorm=1/16 (v_s = H(V)/256
rotated) — the PV MMA accumulates Σp·H(V)/256 = H(acc_true)/256 and the
epilogue kvarn_mma_acc_fwht (A1-validated bit-exact, LIVE at .inc:735 in
the mwarpslot==1 partial-write path) recovers acc_true; ~33% win measured
(104→154 GB/s, doc82_a2_kernel_result.md). On top of that, packed is
warp-specialized at 256 threads (6 dequant producer + 2 MMA consumer warps)
with cp.async double-buffered stage_k/scales_scratch. slice4 (unified) does
the opposite: smem-based kvarn_fwht_channel per key for BOTH K and V through
a 32KB float[32][256] scratch (8 smem round-trips per tile, critical path,
no next-tile prefetch), 64/128 threads, 32-key tiling. (Occupancy is NOT a
factor today: both routes 1 block/SM.)
DECISION (2026-08-30, coordinator): KVarN DECODE default STAYS packed
(NINFER_KVARN_DECODE: unset=packed, =unified, =split legacy; flip committed
94218bd6); the unified route ships behind the env var (correct +
oracle-verified) and is made competitive by PORTING THE A1/A2 CONVENTIONS
into the unified prologue (results/104_kvarn_unified_prologue_hadamard_design.md):
A. Q-side warp-level FWHT at q staging.
B. K: unpack+scale straight to k_s — no Hadamard.
C. V: unpack only + DEFERRED acc-FWHT epilogue via kvarn_mma_acc_fwht (what
   packed SHIPS; the helper's documented C-fragment distribution
   c=8n+2*lid+s / row=gid+8*(j>>1) is EXACTLY the slice2 body's acc layout,
   so it drops in without a new butterfly; validated in
   tests/kvarn_acc_fwht_cpu_ref.cpp).
E. TAIL: forward-FWHT the tail K/V tiles (the rotated-domain body cannot mix
   domains — packed escapes via two separate SIMT loops; the unified single-
   MMA body cannot; one tile/step, warp-level, negligible).
D. With A+B+C+E the float scratch disappears → 2 blocks/SM (restore
   __launch_bounds__(Wc*32, 2)).
F. Only if still short: producer/consumer split + double-buffered prefetch.
Gate: long-ctx A/B (10k/25k/40k/80k/160k, greedy+sampling) within
[0.85,1.15] vs packed, oracle 10/10 at every step, then default flip.
STALE-COMMENT TASK (Agent 1, next commit): (1) gqa_attention_kvarn_mma.cuh
kvarn_mma_acc_fwht header says "currently UNUSED" — it is LIVE at
packed.inc:735; (2) packed.inc header lines 213-219 describe the REMOVED
inline-V variant, not the live deferred path — rewrite to match the code.
Note for the harness: a 4k-only A/B cannot gate KVarN perf (attention is a
small share of the step there; -4% at 4k became -59% at 25k) — the long-ctx
cells are the discriminator.

## 4. Phases

### Phase 0 — baseline matrix (IN PROGRESS — first data exists, 2026-08-29)
Per variant × per ctx {10k, 25k, 40k, 80k, 160k, 250k (where it fits)}: decode t/s (greedy + Qwen-thinking sampling), MTP acceptance, tok/round, prefill t/s, later: A2 identity, A/B byte-identity, prefix-restore t/s + identity. **First matrix landed 2026-08-29** (`results/decode_guard_allcache_sampling_greedy.md`, overlapped spec, 1 run/cell, pre-Qwen-config — provisional baseline seeded from it, see §8). Missing: 40k/80k/160k cells (now in spec), Qwen thinking config, 3-run medians, identity gates.

### Phase 1 — launch-config unification (low risk, no numerics change)
- One per-SKU dispatch table: `sm_count × occupancy model → {decode_splits, smem_bytes, grid, tpb, tpb_cap}` for every attention/rope/GDN/MoE launch site. Runtime `multiProcessorCount` query (already available via `DeviceContext::props`) instead of `kRtx5090SmCount=170` / "36 SMs" constants.
- Delete dead regime code: staged shadow (`stage_pages=0` always, `has_staged`, `NINFER_KVARN_NO_SHADOW`), the T=1/T>1 packed routing split (docs/83 option B), and any `if (sm==36||sm==170)`-style special cases.
- Gate: per-variant decode t/s within the Phase-0 per-variant envelope on this box (no numerics change allowed — byte-identity battery must stay green); table validated on both 36-SM and 170-SM boxes as they appear.
- Why first: it fixes the RTX-5000-series rollout blocker (docs/102 §13.7) without touching math, and it de-risks Phases 2-3 (one tuning surface to reason about).

### Phase 2 — unified decode kernel (the big one)
- Implement `gqa_decode<TKV>` per §3 (start: KVarN prologue → canonical body, since that kernel already has the smem dequant scaffolding; then I8; then BF16 as the reference).
- Acceptance gates: (a) BF16 path byte-identical to today's shipped bf16 kernel (it is the reference — any change is a defect); (b) KVarN/I8: A2 identity (MTP==plain) per variant, A/B byte-identical per variant, greedy acceptance within ±0.5 pt of Phase-0 per-variant baseline (the *only* allowed delta vs BF16 is the Phase-0-measured storage-loss penalty, and it must match Phase 0, i.e. no *new* drift); (c) decode t/s within Phase-0 envelope (prologue cost is the only expected delta).
- This is where the deferred V-FWHT epilogue and split/merge die. Budget: this is weeks of kernel work; gate it behind Phase 1 and the D-21 closeout items.
- **Status (2026-08-30): I8 DONE + gated (oracle 8/8, model gates pass, committed c45fb29a). KVarN unified route DONE + correct but NOT the decode default (1.3–2.8× slower at long ctx — see §3 KVarN decode residual); decode default stays packed until the query-fold QK path lands. BF16 reference path is the body itself (trivially unified).

### Phase 3 — unified prefill
- Same family, prompt route, in-kernel dequant; delete `gqa_kvarn_materialize_kernel` (O(n²) pass).
- Gates: prefill t/s per variant within Phase-0 envelope (expect KVarN prefill to *improve* — n² pass removed), byte-identity per variant vs its own Phase-0 prefill output at 4k/40k, 250k KVarN must-pass battery (T1/T2/T3/T5/T8/T10/T11/T15/T16/T17/T18, docs/54 §9) unchanged.

### Phase 4 — prefix-reuse unification
- Measure first: KVarN snapshot/restore cost + numerics vs bf16/I8 exact open page (Phase 0 already collects the restore-identity data).
- Options: (a) make all variants snapshot (uniform, small cost on bf16/I8); (b) make KVarN open-tile handling exact-enough to not need the MTP-seed half (design work). Decide from data.
- Gate: prefix-restore identity + t/s per variant, multi-turn unaligned-restore battery (T17) unchanged.

### Phase 5 — tighten the cross-variant gates (permanent)
Extend `tools/ops/run_verify_tests.sh` + decode_guard:
- Add a KVarN row (today only bf16 default + int8 row exist in CI).
- Cross-variant: decode t/s ratio ∈ [0.85, 1.15] × (bandwidth-adjusted) per pair; greedy acceptance delta ≤ Phase-0 storage-loss penalty + 0.5 pt; A2 identity + A/B byte-identity **per variant**; prefill t/s ratio ∈ [0.8, 1.25] × (n²-adjusted) per pair.
- These gates make "uniform" *enforced*, not aspirational — any future per-variant kernel change that re-splits numerics or perf fails CI.
- Standing per-cell baseline + monotonic ratchet already wired: `tools/bench/decode_guard_baseline.json` + `tools/bench/decode_guard_check.py` (see §8).

## 5. What stays different (honest scope)

- **Storage loss:** KVarN 4/2-bit < I8 8-bit < BF16 exact. The acceptance penalty is intrinsic and is exactly what Phase 0 measures; Phases 2-4 can only guarantee it doesn't *grow*.
- **Dequant compute + bandwidth:** KVarN prologue > I8 > BF16. Intrinsic; bounded; the point is the difference becomes the *only* difference.
- **VRAM:** KVarN 33 KiB/page/head-2 vs BF16 128 KiB — the reason the variants exist.
- **D-21 residual** (width-4 batched attention precision, Option C accepted): orthogonal; the unified kernel may incidentally shrink it (one accumulation order), but we do not depend on that.

## 6. Ordering & dependencies

1. docs/103 (GDN-gate T=1) **first** — same disease class (R6), small, recovers 7%, and validates the "one canonical accumulation, dispatch to the fastest bit-identical schedule" principle on a toy case before the kernel-level work.
2. Phase 0 (baseline) — can start as soon as the GPU frees (server busy 08-29 06:35).
3. Phase 1 — independent of Phases 2-3; safe to land before any per-SKU hardware arrives (docs/102 §13.7 checklist item).
4. Phases 2→3→4 strictly ordered (decode body first, then prefill reuses it, then prefix mechanics).
5. All phases: bug-free-first — no phase starts until D-21 closeout items (temp>0 re-run, full-width H5, full CI — server items in `results/d21_mtp_verify_anchor_fix.md`) are green; D-22/D-23 are done.
6. Per-SKU validation happens as hardware lands: the Phase-1 table is the only thing that should need per-SKU attention.

## 7. Risks

- Phase 2 is real kernel work with a byte-identity target — the deferred-epilogue removal changes KVarN's numerics by design (that's the point) but must not change BF16's; if KVarN's prologue can't dequant V fast enough in smem at 250k, the fallback is keeping the split/merge for KVarN decode only (partial uniformity — decode body unified for QK, epilogue deferred only for KVarN) and documenting the residual split.
- Phase 0 must be done on a clean, committed tree (worktree is currently dirty with in-flight packed-kernel changes — commit or stash first, per docs/101 §4 step 1) or the baseline is contaminated.
- Per-SKU table (Phase 1) needs the occupancy model to be right for *all* sm_120 SKUs, not just 36 and 170 — 5070/5080 SM counts must be measured, not guessed, before shipping the table.

## 8. Standing baseline and gates (updated 2026-08-29)

### 8.1 First matrix (2026-08-29, build 502b42ca + D-22 fix)
`results/decode_guard_allcache_sampling_greedy.md` (overlapped spec, 48 gen tokens, 1 run/cell):

| KV | 10k | 25k | 40k | 80k | 160k | 250k |
|---|---|---|---|---|---|---|
| bf16 decode t/s (samp/greedy) | 69.9/70.0 | 67.6/67.7 | — | 68.3/68.4 | — | — |
| int8 | 70.5/70.8 | 69.5/79.5 | — | — | 63.4/68.1 | — |
| kvarn | 70.2/60.8 | 62.9/67.3 | — | — | — | 36.4/40.7 |

**Findings:**
1. **Short-ctx convergence (≤25k): all variants ≈ 70 t/s** (bf16/int8/kvarn within ~0.5% at 10k). At ≤25k attention is a small fraction of the step — the step is dominated by non-attention work (64 layers of MLP/GDN + collectives; includes the docs/103 GDN-gate 7%). The KV variant barely matters yet. "Old speeds" (≥69 t/s) are already achieved here, including for KVarN — the packed kernel is healthy; the collapse is a long-ctx phenomenon.
2. **Long-ctx collapse is the unsolved problem**: kvarn 10k→250k = 70.2→36.4 t/s (−48%). Attention dominates at 250k and the packed kernel is compute-bound (K-deq 22% + QK-MMA 18% + PV-MMA 18%, docs/83 lever-1 profiling); lever-1 (int8 QK on raw codes) measured NEGATIVE (per-page query-fold overhead > MMA speedup). No known lever; docs/78 step-5 (option c) is the deferred path, gated on "≥128k decode throughput is a product requirement".
3. **Sampling t/s can exceed greedy t/s in a cell** (kvarn 10k: 70.2 vs 60.8). Mechanism, not a compute effect: t/s = tok/round ÷ round time, and round time is identical for both modes. The *acceptance test* differs — greedy accepts only exact argmax matches (P(draft==argmax)), while rejection sampling accepts with probability p_target(draft), harvesting near-miss drafts (2nd/3rd choice with real probability mass) that KVarN's lossy KV produces more often. On bf16 (exact KV) draft≈argmax and both tests converge (69.9 vs 70.0). This is exactly why the sampling-on metric must be guarded — it is the user-facing number and it can sit ±15% from greedy per cell. (48 tokens/cell = 16 rounds → ±~7pp acceptance noise; use 3-run medians for baselines.)
4. **Greedy 10k int8 outlier** (79.5 t/s, 81% accept) is a lucky deterministic trajectory (fixed prompt/seed) — single-cell numbers are quantized to 48 tokens; do not chase per-cell outliers.

### 8.2 Canonical sampling config (Qwen-recommended, qwen3.8-27b)
- **thinking** (serve default, thinking=on; matches CI samp_s42/s43): `temp 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0` — **canonical guard sampling tag**.
- **instruct** (non-thinking): `temp 0.7, top_p 0.80, top_k 20, presence_penalty 1.5` — supported via env (`TEMP/TOP_P/TOP_K/PRESENCE_PENALTY`), add as a second tag if needed.
- `repetition_penalty` is 1.0 (= disabled) in both Qwen sets and is NOT implemented — no gap.
- The guard records the config per cell (`sample_cfg` in results JSON); `decode_guard_check.py` refuses to compare across configs.

### 8.3 Guarded baseline (monotonic ratchet)
- Spec (overlapped so variants compare directly): `kvarn 10k 25k 40k 80k 160k 250k`, `int8 10k 25k 40k 80k 160k`, `bf16 10k 25k 40k 80k` (bf16 max ~89k, int8 ~168k, kvarn ~250k VRAM).
- **40k stays in the spec forever** — it is the G2a gate context (docs/78: decode ≥69 t/s @40k). It was dropped from the spec once (QA C4 class) — the guard now fails the review if it disappears.
- Baseline: `tools/bench/decode_guard_baseline.json` (18 cells, PROVISIONAL — old config, 1 run/cell; first official = next full run under Qwen thinking config: `decode_guard_check.py <run.json> --ratchet --replace-config`).
- Check: `tools/bench/decode_guard_check.py results.json...` → PASS/WARN/FAIL per cell (tps −5% FAIL / −2% WARN; acceptance tracked ±5pp, not gated). Exit 0/2/1.
- **Ratchet policy (keep raising the bar):** t/s baselines only go UP, only after a verified perf win, via `--ratchet`. Re-baseline (3-run medians) on: major kernel changes, new SKU hardware, Qwen config changes. The bar sequence: current provisional → official Qwen-config baseline → G2a @40k ≥69 → 250k floor ≥30 (PASS today: 36.4/40.7) → (step-5 option c, if product requires) ≥69 @250k.

### 8.4 G2a gate status (docs/78, rescope accepted 2026-08-27)
- **@40k ≥69 t/s (target): UNVERIFIED on current tree** — last measured 71 t/s @40k (08-27, pre-wall-removal binary) and 66.7 t/s @40k (08-28, docs/83 option B). 40k cell now in spec; first official run decides.
- **@250k ≥30 t/s (no-collapse floor): PASS** — 36.4 (sampling) / 40.7 (greedy) on 2026-08-29.

---

## 9. Addendum — Phase 2 KVarN decode CLOSED, Phase 3 re-targeted (2026-08-31)

Full detail: `docs/120_phase3_status_and_task_list.md`, `docs/119_*`, `results/119b_*`,
`results/kvarn_unified_decode_matrix_20260830.md`, `results/113_prefill_nsys_attribution.md`.

**The §4 "KVarN decode residual" block above is superseded.** It describes the state at 18:03 on
2026-08-30. What changed since:

| claim above | current state |
|---|---|
| "KVarN DECODE default STAYS packed" | **superseded** — default is unified (`29d0a797`); `NINFER_KVARN_DECODE=packed` kept as the rollback lever |
| slice4 uses smem `kvarn_fwht_channel` per key for BOTH K and V through a 32 KB scratch, 64/128 threads, 1 block/SM | **no longer true** — slice4 now uses register+shuffle `kvarn_mma_warp_fwht`, K is unpack-only, V is deferred to `kvarn_mma_acc_fwht`, **no dynamic smem**, `__launch_bounds__(Wc*32, 2)` restored (SHARED=47360 B, 2 blocks/SM, verified via cuobjdump) |
| residual curve 43.1 / 27.4 / 20.2 / 10.9 t/s | **pre-fix numbers.** Post-fix greedy: 67.8 / 72.0 / 68.3 / 56.9 / 46.3 / **42.1 @250k** (was 4.4) |

The packed-kernel numerics description in that block (Q rotated once, K unpack-only, V deferred to
the LIVE `kvarn_mma_acc_fwht` at `.inc:735`) was **correct and is now also what slice4 does** — it
was ported verbatim, which is what closed the gap.

**Measured result:** acceptance-free kernel cost (ms/round = 1000·tok_per_round/tps) is
**0.928–1.048× packed across all 12 cells** of the 6-context × greedy/sampling matrix. Oracle 10/10;
`regress_unified.sh` PASS for bf16+int8+kvarn.

**Two things Phase 2 did NOT fix, both now known and owned elsewhere:**
1. KVarN's −37% 10k→250k **decode** decay is intrinsic per-key compute (~153 ns/key vs int8's ~51
   ns/key on roughly half the bytes) and is present in **both** routes — no decode change will
   touch it.
2. §4 Phase 3's premise is wrong: *"delete `gqa_kvarn_materialize_kernel` (O(n²) pass)"* — it is
   not O(n²) (docs/66 removed that regime) and it is **4.1%** of the prefill gap. nsys: of a 3.65 s
   gap vs bf16, `quantize_tile_kernel` is **3.61 s (99%)**, cost driven by `kSinkhornIters = 16` —
   quantizer algorithm, not memory. And §4 Phase 3's byte-identity gate references **Phase-0 prefill
   output that does not exist** in any committed artifact (verified: baseline v1 JSONs store t/s,
   acceptance, wall and token *counts* only — no token IDs or text). Use docs/113 Step 1's
   route-vs-route form instead. See docs/120 §2.
