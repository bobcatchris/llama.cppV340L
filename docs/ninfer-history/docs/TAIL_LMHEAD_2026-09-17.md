# TAIL LM_HEAD — the verify logits projection: deep-read, honest budget, fix design (no-GPU desk, 2026-09-17, amd/tp4-cure)

**Seat:** NO-GPU design desk. **Method:** code-read of the M2-wrapped lm_head path at HEAD
`248630d1b` + byte/rate budget against the given ceilings (378 GB/s achievable / 483.8 nominal;
tuned-NVFP4 isolated 265-280 GB/s; 10.75 TF/s F32) + the banked M1/M2 rows
(`results/amd/coherence/VERIFY_DECOMP_row.txt`, `VERIFY_TAIL_row.txt` — read, not re-derived).
Zero GPU, zero builds, zero src//tools edits; this doc is the only file touched.

**Headline:** the M2 `lm_head` field wraps the **full-vocab ColumnN TP4 shard
`[62080, 5120]` in `W8G32_F16S`** — NOT the 40960-id draft-vocab slice, and NOT fp4
(the banked M2 row's parenthetical "(draft-vocab 40960 slice … fp4 weights)" is wrong on
both counts; §1c). Its honest memory budget is **~0.9-1.3 ms/round**; it actually measures
**61.93 ms (5.45 GB/s effective — 49-69x below every ceiling)** because it dispatches to the
W8 *small_t* arm whose NVIDIA-tensor-core schedule runs as an **emulated SIMT `mma_bf16`**
on gfx900 (Vega 10 has no MFMA hardware). This is an **issue/latency pathology class, not a
tuning-gap class**: nothing memory-shaped can produce this number (§2). TOP fix: A/B the
route onto the in-serving-proven SIMT arm (`launch_w8_simt_r8_c4`, ~174 GB/s on the same
format at `[10240,5120]`) — predicted **2-4 ms post-fix**, verify 117.4 → ~66-70, round
133.8 → ~85-90 (§4).

---

## 1. Findings — path, geometry, format, dispatch (file:line @ 248630d1b)

**(a) The wrap site and the verify path.** The M2 tracer's `LM_HEAD` event pair brackets
exactly one call: `ops::linear(flat_hidden, *lm_head_, flat_logits, stream)` at
`src/targets/qwen3_6/impl/runtime/text_context_impl.h:1869` (begin `:1868`, end `:1870`),
inside `target_verify_batch_impl` — the TP4 MTP verify round calls this via
`tp2_backend.cpp:3033` (`st.text->target_verify_batch(...)`, T = rk+1 = 3 at k=2,
`verify_logits` staged `{248320/4, 3, 1}` bf16 at `tp2_backend.cpp:1341`). `lm_head_` is
bound once at `text_context_impl.h:881` (`lm_head_ = &weights_.output_head;`). Local vocab
per rank: `n_vocab = kCfg.vocab / tp_world_ = 248320/4 = 62080` (`:1860`; `kCfg.vocab =
TextConfig::output_rows`, `text_context.h:43`). After the head: local `ops::argmax`
(`:1873`) → LGATHER `rk+1 = 3` allgathers (`tp2_backend.cpp:3050-3069`) →
`allreduce_argmax` (`:3143`) → accept → pinned D2H.

**(b) Weight format and shard.** `output_head` is bound at
`src/targets/qwen3_6_27b/impl/load/bindings.cpp:533`
(`bind_weight(binder, "text/output_head", vocabulary_format, {248320, 5120})`) and
materialized per-rank at `bindings.cpp:679`
(`materialized_weight(backing, plan.output_head, 248320 / tp_world, 5120)`) — the ColumnN
vocab split, sole role of its class (`tp_load.cpp:366-368` names `text/output_head`
`TpRole::ColumnN`; `:410` splits `full_rows / w`). The on-disk format is resolved FROM THE
CONTAINER, not from the requested constant: `bind_weight` passes `actual_format` by
reference and `Binder::require_weight_tensor` overwrites it from the descriptor
(`src/artifact/binder.cpp:68-80`). The banked census print against this artifact names the
triple for BOTH endpoints: **`(W8G32_F16S, row-split-k128-v1, 1,350,860,800)`**
(`docs/amd/NVFP4_AMD_PLAN_agent5.md:419` and the A5 cell at `:492` — 248320 x 5440 B/row =
1,350,860,800 exactly). So the live lm_head is **W8G32_F16S** (1 B s8 code + 2 B f16 scale
per 32-group = 5440 B/row), **NOT bf16** and **NOT FP8**: `endpoint_format()` returning
`FP8_E4M3FN_ROW_BF16S` under `Qwen38Nvfp4` (`bindings.cpp:37-47`) is the already-filed
STALE DECLARATION, unreachable because of the resolve-from-disk rule (agent5's hygiene
item, same cites).

**(c) Read which: shard vs slice.** The tracer header's ambiguous comment
(`src/runtime/tp2/verify_tail_trace.h:17` — "draft-vocab 40960 slice or full-vocab ColumnN
shard") resolves as: **the M2 `lm_head` field = the full-vocab ColumnN shard
`[62080, 5120]` W8G32, 337,715,200 B/rank** (62080 x 5440). The 40960-id draft-vocab slice
is a DIFFERENT tensor — `state->draft_head`, gathered to `40960/4 = 10240` local rows at
boot (`tp2_backend.cpp:1106-1224`), launched by `launch_draft_head` via
`ops::detail::launch_w8_simt_r8_c4` (`tp2_backend.cpp:1387-1393`) — and it is measured
under the `propose`/`chain_head` fields (**0.32 ms**), never under `lm_head`. The verify
must argmax the FULL 248320 vocab (global tokenizer ids, D-22 comment
`tp2_backend.cpp:3046-3052`); the draft slice cannot serve it.

**(d) Kernel dispatch at T=3.** `ops::linear` 4-arg → `LinearPolicy::A16Only`
(`src/ops/linear/linear.cpp:177-180`) → `w8_dispatch` (`linear.cpp:95-96, 83-118`) →
`select_w8_a16_launch(k=5120, n=62080, t=3)`: the vocab bucket
`{20480, 40960, 62080, 124160, 248320}` routes `t <= 33` to **`launch_w8_small_t`**
(`src/ops/linear/w8/w8_dispatch.cpp:50-62` — the family-10 cure, landed in `1533e36ea`,
ancestor of HEAD; before it, world=4 fell to the generic tail → `simt_r8_c4`, then
stub-class). `launch_w8_small_t` instantiates the TP4-vocab launcher table
(`w8_small_t.cu:83-91`), schedule
`W8LinearSmallTProductionSchedule<W8VocabularyTp4ProjectionGeometry, 3>`:
`kTileTokens=8` (T=3 padded to 8), `kKWarps=8`, `kMinBlocksPerSm=2`, Direct scale access
(`w8_config.h:134-156` — "parameters mirrored from the Tp2 vocabulary arm … **Tuning for
62080 rows can follow on the latency ladder; CORRECTNESS-FIRST**"). Grid = 62080/16 = 3880
CTAs x 256 threads. The kernel core (`w8_small_t_mma.cuh`) is an NVIDIA m16n8k16
tensor-core schedule: `mma_bf16` + `ldmatrix_x2` + `cp_async` + per-group double
`__syncthreads()`. **On gfx900 none of that is hardware**: "Vega10 has NO mfma hardware
(that arrives with gfx908/MI100)" (`src/ops/common/mma.cuh:218-226`); `mma_bf16` is the
SIMT emulation — per m16n8k16 op, ~32 `__shfl_sync` + per-pair bf16→f32 converts + scalar
f32 FMA loops per lane (`mma.cuh:242+`), and `ldmatrix` is a plain-LDS emulation
(`mma.cuh:25-37`).

**(e) One round = ONE call.** The head runs once per verify round (all 3 columns in a
single launch; no chunk loop at W8 — the `launch_a16` chunk loop is the FP8 family's,
`fp8_dispatch.cpp:53-73`, not this path). Nothing else moves the 337.7 MB.

---

## 2. Honest bytes/time budget — which hypothesis can produce a 10-30 ms field

Mandatory per-round traffic (per rank, T=3): weight **337,715,200 B** dominates; activation
read 30,720 B; logits write 372,480 B (L2-resident). So the op is a weight-stream at heart,
and the predicted-ms table is weight-bytes / rate:

| scenario (geometry x format) | bytes/rank | @378 GB/s achievable | @280 | @265 | @56 (W8 small_t family, MTP-layer serving rate) | @33 (M1 pre-tune body rate) | verdict vs 10-30 ms |
|---|---|---|---|---|---|---|---|
| **REAL: W8G32 ColumnN shard [62080x5120]** | **337.7 MB** | 0.89 | 1.21 | 1.27 | 6.0 | 10.2 | healthy rates CANNOT; 33 GB/s touches band bottom |
| W8G32 full replicated [248320x5120] (excluded: ColumnN split) | 1350.9 MB | 3.57 | 4.82 | 5.10 | 24.1 | 40.9 | band only below ~34 GB/s |
| bf16 shard (VERIFY_DECOMP §1's 635 MB guess — WRONG format) | 635.7 MB | 1.68 | 2.27 | 2.40 | 11.4 | 19.3 | in band only at body-rate |
| bf16 full replicated (WRONG twice) | 2542.9 MB | 6.73 | 9.08 | 9.59 | 45.4 | 77.1 | in band — but excluded by the census triple + tp_load role |
| fp8 shard (stale endpoint_format arm — unreachable) | ~322 MB | 0.85 | 1.15 | 1.21 | 5.75 | 9.75 | same class as W8 shard |

**Measured reality: 61.93 ms clean-band (M2 row; per-rank 66.0/59.7/60.0/61.6) =
5.45 GB/s effective = 48.6x below 265, 51.3x below 280, 69.3x below 378.** That is BELOW
every memory-shaped hypothesis in the table, including the degraded-33-GB/s one — i.e. the
op is **not memory-shaped at all**. Three independent confirmations:

1. **Clock scaling:** leg1 (deep sclk droop) 153.8 ms vs leg2 clean 66.0 = 2.33x — tracks
   the sclk ratio (M2 row cross-check 2). A DRAM-bound op scales with mclk (held at 945),
   not sclk; an issue-bound op scales with sclk. The op is issue/latency-bound.
2. **Cross-arm, same format, same box:** the draft slice `[10240, 5120]` W8G32 via
   `launch_w8_simt_r8_c4` runs inside `propose` = 0.32 ms → ~55.7 MB / 0.32 ms ≈ **174 GB/s
   in serving**; the MTP-layer W8 small_t projections stream at ~56 GB/s (align 8.03 ms /
   ~451 MB). The TP4-vocab small_t instantiation is 10-32x worse per byte than sibling arms
   of its own family.
3. **Mechanism fits:** the schedule is a 5090 tensor-core import ("RTX 5090 cold-cache
   winner" lineage) emulated instruction-by-instruction on gfx900 — shuffle-heavy m16n8k16
   fragments, per-group double barriers, cp_async emulation — mirrored WITHOUT re-tuning
   onto a 4x-larger row count (`w8_config.h:134` says so in its own words). The NVFP4 lane
   got the SWEEP+HFMA2 retune (V0 won all 4 geometries); the W8 lane's small_t arm did not.

So: a **10-30 ms lm_head field is producible ONLY by (i) the real 337.7 MB shard running at
11-31 GB/s — a pathology (actual: 5.45, even below that band's implied floor), or (ii) a
bf16/unsharded geometry accident — excluded by banked evidence.** The M1 tail (55.21) and
the M2 field (61.93, +11% bracket edges/clocks) are the same owner: lm_head ~50-52
(clock-scaled) + lgather 4.92 (ranks 1-3 waiting on rank0's late finish) + <1 everything
else. **The tail IS the lm_head op plus its gather. And it is NOT a memory-budget problem —
it is an arm-selection problem.**

---

## 3. Fix design (doc only — src/ untouched; diffs are sketches)

Candidates weighed:

- **(A) TOP — route skinny-T TP4 vocab to the proven SIMT arm.** `launch_w8_simt_r8_c4` is
  in-serving-measured at ~174 GB/s on the same weight format on this box (propose field),
  and is what world=4 hit BEFORE the family-10 cure. The cure fixed a determinism/dispatch
  disease (generic-tail → never-tested arm); its route choice just landed on the wrong
  PERFORMANCE arm. The fix is NOT a blind revert — simt_r8_c4 must now pass the tests it
  never ran (bench A/B, accumulation-order parity, count600 fingerprint):

```diff
--- a/src/ops/linear/w8/w8_dispatch.cpp
+++ b/src/ops/linear/w8/w8_dispatch.cpp
@@ select_w8_a16_launch, case k == 5120 (current :50-62)
         case 20480:
         case 40960:
-        case 62080:   // TP4 LM-head shard — routed to the VERIFIED small_t arm (family-10 cure;
-                      // the generic tail would pick the never-on-this-box simt perf arm)
         case 124160:
         case 248320:
             if (t <= 33) { return launch_w8_small_t; }
             if (t <= 48) { return W8_SEL_MMA(t, launch_w8_mma_r64x16_c48_k128_a1); }
             if (t <= 64) { return W8_SEL_MMA(t, launch_w8_mma_r32_c64); }
             return W8_SEL_MMA(t, launch_w8_mma_r64_c128);
+        case 62080:   // TP4 LM-head shard. M2 (VERIFY_TAIL_row 2026-09-17) measured the
+                      // small_t arm at 61.93 ms/round (5.45 GB/s) in serving — emulated-MMA
+                      // issue-bound on gfx900 — while simt_r8_c4 streams the same W8G32
+                      // format at ~174 GB/s (propose field, [10240,5120]). Bench A/B +
+                      // numerics parity + count600 fingerprint are the gate (family-10
+                      // determinism cure is preserved: the arm is now named, tested traffic,
+                      // not generic-tail fallback).
+            if (t <= 4)  { return launch_w8_simt_r8_c4; }
+            if (t <= 33) { return launch_w8_small_t; }
+            if (t <= 48) { return W8_SEL_MMA(t, launch_w8_mma_r64x16_c48_k128_a1); }
+            if (t <= 64) { return W8_SEL_MMA(t, launch_w8_mma_r32_c64); }
+            return W8_SEL_MMA(t, launch_w8_mma_r64_c128);
```

- **(B) Plan B if simt_r8_c4 benches poorly at 62080 rows: purpose-built vocab GEMV**
  (v100-skinny GEMV class — the family that hit 265-280 GB/s isolated on the NVFP4 side):
  grid-stride over rows x whole wavefronts, one wavefront owns rows r..r+kR-1, vectorized
  4-byte code loads, half2 scale math applied per 32-group after a 32-wide f32 dot kept in
  registers, T columns unrolled x3 (verify width is compile-known per launcher table cell —
  the `make_launchers` pattern already keys on ActiveTokens). Skeleton only — the arm is
  NEW code and needs the full cell treatment; do not reach for it before (A)'s bench.
- **(C) NVFP4-quantize the lm_head**: 337.7 MB → ~180 MB (0.5625 B/elem vs 1.0625) AND onto
  the tuned-kernel family (265-280 isolated / 147-192 serving). But the TARGET head drives
  accept/rebase semantics — this is the accuracy-sensitive endpoint; needs the full
  coherence gate (GREEN rows vs W8-head logits, acceptance-rate delta band) before it is
  even legal to measure for serving. Higher lift than (A) for a similar steady-state number;
  keep as the SD-1-era follow-up if the round budget later demands the extra ~1-2 ms.
- **(D) Tiled GEMM / M>=64 gate — DOES NOT APPLY**: the M>=64 gate is the NVFP4 family's
  (`nvfp4_tiled_gemm_hip`); the lm_head is W8, and gfx900 has no MFMA for ANY format — the
  registered W8 MMA arm (`launch_w8_mma_r64x16_c48_k128_a1`, legal t<=48) is emulation-class
  too. Include it as the third bench leg for completeness; expected not to win.
- **(E) Fused argmax epilogue (v100-skinny has one)**: saves a 372 KB logits write+read
  (~2 µs at ceiling) + one launch. µs-class hygiene, NOT a tail fix. The 4.92 ms `lgather`
  is mostly rank0-lateness (M2's asymmetry note) and should collapse to ~0.5-1 ms once
  lm_head is fixed — revisit gather-batching/root-rotation only if a healthy-lm_head re-run
  still shows it.
- **(F) Vocab-slice the verify head to the 40960 draft vocab**: REFUSED — the verify argmax
  over the full 248320 vocab is load-bearing acceptance semantics (D-22); changing it is an
  accuracy redesign, not a perf patch, and saves at most the same ~1-3 ms class.

**Predicted post-fix (A):** weight 337.7 MB at 90-170 GB/s effective for a SIMT-GEMV-class
arm at T=3 (simt_r8_c4 demonstrated ~174 on the same format; halved for the 6x row count
and T=3 surcharge) → **lm_head 61.93 → 2-4 ms**; conservative floor even at the MTP-layer's
56 GB/s class = 6.0 ms. Tail 55.21 → ~4-6; OPTRACE verify 117.4 → ~66-70; B1 round
133.8 → ~85-90 at M1-class clocks → decode ~28-31 tok/s at yesterday's thermals (today's
instant-droop box may cap the absolute number — see §5).

---

## 4. DECISIVE CHECK for the next window (cheap falsification first)

1. **Bench cell (no serve boot, minutes):** extend the roofline harness A/B at exactly
   `n=62080, k=5120`, T in {1, 3, 4}: `launch_w8_small_t` vs `launch_w8_simt_r8_c4` vs
   `launch_w8_mma_r64x16_c48_k128_a1` (all three registered for this shape). RED
   reproduction = small_t ≥ 10 ms-equivalent at serving clocks (confirms the serving field);
   GREEN = the winner arm's measured GB/s names the route. This is the decisive
   arm-ranking measurement; it gates the route flip.
2. **Route-flip serve A/B (one boot, banked binary per BANK-BEFORE-RELINK):** count600
   conc=1, `NINFER_VERIFY_TAIL_TRACE=1` + the M1/M2 trace env, M0 clocks sideband per row-law.
   Expected BEFORE: lm_head ~62, lgather ~4.9, verify ~117, round ~134 (clean-band
   treatment mandatory on today's droop box). Expected AFTER: **lm_head ≤ 4 ms** (RED if
   ≥ 10), lgather ≤ 1, verify ~66-70, round ~85-90. **Fingerprint guard:** 204 rounds /
   acc 0.97 / t/r 2.94 must hold — the arms differ in accumulation order, so this is a
   parity-gated route change, not a byte-identical one (coherence discipline applies;
   warmup-id anchor re-derived on the new binary).

---

## 5. Honesty block

- The M2 row's owner number (61.93 ms, clean-band, 278 rank-rounds) and M1's split
  (117.41 = 62.20 + 9.60 in-loop AR + 55.21 tail) are **banked rows this desk read, not
  re-derived**. This desk's independent contribution is the geometry/format correction and
  the budget that proves the number is pathology-class.
- **Correction served to the banked M2 row:** its "FIX TARGET" line names the owner as
  "(draft-vocab 40960 slice … fp4 weights)". Both halves are wrong: the traced call is the
  full-vocab ColumnN shard [62080x5120] in W8G32_F16S (evidence chain §1b/§1c); the
  40960-slice W8 head is a different tensor measured under propose/chain_head. The FIX
  TARGET verdict (#1 owner, ~90% of the device tail) STANDS — the label did not.
- **VERIFY_DECOMP §1's "~4 ms tail, BF16 lm_head 635 MB/rank, lm_head ≈ 3-4 ms"** was
  wrong twice (format W8 not BF16 → 338 MB not 636; and memory-bound assumption → reality
  is issue-bound at 20x that number). That doc's own honesty row already flagged the row
  as an estimate; M1/M2 replaced it.
- The ~174 GB/s propose anchor includes argmax + remap inside the propose field —
  order-of-magnitude, not a clean kernel rate. The bench cell (§4.1) is what prices arms.
- The emulated-MMA mechanism is **inference from code + the 2.33x sclk scaling + cross-arm
  rates**; no nsys/oprofile of the kernel exists. Do not merge the route flip on this doc —
  merge it on §4's two measurements.
- Post-fix absolute-ms predictions assume M1-class clocks. Today's box instant-droops under
  decode bursts (M2 row: sclk 7→2-5 within 2-6 s of burst; prefill immune) — every absolute
  row needs the clean-band treatment or a re-run when the box returns to last-night
  behavior; rankings/ratios remain thermal-robust.
- Line numbers are from THIS worktree at `248630d1b` and may drift. This desk touched only
  `docs/amd/TAIL_LMHEAD_2026-09-17.md`; src/, tools/ untouched; results/amd/coherence/
  read-only. The GPU window's uncommitted files in the worktree are not mine and were not
  staged.

---

Bases: code @ `248630d1b` — `text_context_impl.h` (:881, :1860-1877), `tp2_backend.cpp`
(:1341, :1387-1393, :3033, :3050-3069, :3143), `bindings.cpp` (:37-47, :514-533, :679),
`binder.cpp` (:68-80), `package.cpp` (:85-104), `tp_load.cpp` (:366-368, :410),
`w8_config.h` (:61-72, :134-156), `w8_dispatch.cpp` (:50-62), `w8_small_t.cu` (:74-91),
`w8_small_t_mma.cuh` (whole), `mma.cuh` (:25-37, :218-226, :242+), `verify_tail_trace.h`
(:17); banked — `VERIFY_TAIL_row.txt`, `VERIFY_DECOMP_row.txt`,
`NVFP4_AMD_PLAN_agent5.md` (:419, :492); given ceilings (378/483.8 GB/s, 265-280 GB/s,
10.75 TF/s).
