# Phase 2 draft — KVarN long-context decode gap (work order draft)

**Status:** READY (plan-owner self-review done 2026-08-29) — drafted per
docs/111 from official baseline v1 (`results/official_baseline_v1_20260829.md`).
Scope: docs/104 Phase 2, scoped to the measurable kvarn long-context decode gap.

## Plan-owner review notes (2026-08-29) — Agent 1 pre-review accepted in full

Pre-reviewed by Agent 1 (kvarn-decode expert); all five findings ACCEPTED. Resolutions:

1. **Baseline v1→v2 re-measure is a MANDATORY step-0 hard gate.** Step 0 re-measures the
   cells on the POST-B' (v2) build BEFORE the pass bar is locked; the cells table + pass
   bar are updated from the fresh numbers (expected shift: B' speeds every T=1 decode
   step, so kvarn cells should move up ~4%). No L-class work starts until the v2 cells +
   pass bar are committed.
2. **L1 vs docs/106 ownership: docs/106 (Phase-1 launch-config table) lands FIRST.** L1
   (docs/115) then tunes ON TOP of the landed table. Conflict rule: **L1's fresh
   measurement wins** — if the tuning disagrees with the table row, L1 updates the row in
   its own commit (newer, final-build measurement).
3. **L3 uses a single reused buffer** — the q8 scratch is per-layer, one layer at a time,
   reused across layers: ~64 MB max @250k, NOT 64 MB × N. State explicitly + add a
   concrete 250k VRAM-budget check to step 0 (baseline ~9059 MB/rank + 64 MB + headroom
   assertion).
4. **kDbgSkip per-bit rebuilds accepted** for step-0 profiling (compile-time, so 6 builds,
   ~10 min each). Note the methodology + a control run with all bits off to confirm the
   phase-skip does not perturb the other phases' timing.
5. **Executor: Agent 1** (kvarn-decode expert). Plan owner sequences GPU slots + reviews
   each class commit.

Execution sequence: docs/106 (Phase-1 table) → docs/115 step-0 v2 re-measure (gate) →
L1 → step-0 profiling → L2 → L3 → re-measure → L4 scope. Post-merge (docs/105 + v2
re-baseline landed).

---

## 1. The problem (baseline v1, greedy, MTP on, k=3, 192-token budget)

Decode t/s gap (kvarn vs best of int8/bf16) widens with context:

| ctx | kvarn | int8 | bf16 | kvarn vs best | round Δ vs int8 |
|---|---|---|---|---|---|
| 10k | 69.1 | 69.6 | 68.9 | −0.7% | +9% |
| 25k | 68.3 | 70.0 | 66.4 | −2.4% | +9% |
| 40k | 64.0 | 66.4 | 65.5 | −3.6% | +11% |
| 80k | 58.2 | 63.4 | 60.2 | −8.2% | +18% |
| 160k | 47.1 | 60.0 | — | **−21.5%** | +31% |
| 250k | 43.5 | — | — | (no comparison) | (no comparison) |

Derived round time (ms = tok/round ÷ t/s): kvarn 45.4/47.6/49.5/55.2/66.9/76.6
vs int8 41.7/43.7/44.7/46.8/51.0. MTP acceptance is NOT the cause — kvarn
tok/round is HIGHER (3.14-3.33 vs 2.90-3.06). The gap is **per-KV-page cost in
the decode attention kernel**, growing linearly with context (attention share
of the round grows with ctx; both variants scale ~linearly in pages, kvarn's
per-page constant is ~30% higher at every long ctx).

Prefill (−13-15% flat at all contexts) is a separate mechanism — Phase 3, out
of scope (docs/113 draft, Agent 2).

## 2. Per-page cost structure (from code + lever-1 profiling)

Packed decode kernel (`src/ops/kernel/gqa_attention_kvarn_decode_packed.inc`),
per page per (layer, kv_head) CTA:
- Phase A: QK bf16 MMA + online softmax [warp 0] ‖ V-dequant 2-bit→bf16 [warps 2-7]
- Phase B: PV bf16 MMA [warp 0] ‖ K-dequant 4-bit→bf16 (affine `code*kc0+kzp)*sr`,
  from cp.async-staged codes) + prefetch(lp+2) [warps 2-7]
- Two `__syncthreads()` per page; **single-buffered** `k_s`/`v_s` (32+32 KiB).
- Lever-1 profiling (doc83_lever1_int8_finding.md, T=4 @40k/250k):
  compute-bound, K-deq ~22% + QK-MMA ~18% + PV-MMA ~18% @250k.
- int8 variant (comparison kernel `gqa_attention_decode_i8.cuh`): s8 MMA
  directly on stored codes (4x TMAC), per-key scale in epilogue — no dequant
  to bf16, no affine, no per-page query fold.

## 3. Ranked levers

### L1 — wave re-tune of kKvarnDecodeSplits (launch config only)
- Site: `src/ops/kernel/gqa_attention_kvarn.cuh:827` — `kKvarnDecodeSplits = 54`
  ("2*54=108 blocks = 3 full waves on 36 SMs").
- A/B 36 (2 waves) / 72 (4 waves) / 54 at 40k/80k/160k/250k. Wave quantization
  and pages-per-split load balance are ctx-dependent (pages_per_split =
  ceil(packed_pages/54)) — the current 3-wave choice may be optimal at 40k but
  not at 160k/250k.
- Expected: 0-15% of attention time at long ctx. Cost: ~1h (constant + A/B).
- **Byte-identity class: NO.** "Fixed-order partial merge" (launcher:148) over
  N splits — changing N changes the partials and the merge accumulation order.
  Gate: greedy acceptance ±0.5 pt per cell + A2 identity (MTP==plain) per
  variant (docs/104 Phase 2 gate (b)).
- This is also the Phase-1 dispatch-table entry for this SKU — coordinate with
  wo/kv-uniform (docs/106) so the table row and this tuning agree.

### L2 — double-buffer k_s/v_s, 2-deep pipeline (scheduling, byte-identity)
- Site: packed.inc page loop (phase A/B, `__syncthreads()` placement); smem in
  `KvarnDecodeShared` (launcher `gqa_attention_kvarn.cu:183-189`, opt-in via
  cudaFuncAttributeMaxDynamicSharedMemorySize, currently ~99 KiB).
- Dequant(lp+1) already runs during PV(lp), but the single buffer + block
  barriers force a full stall before QK(lp+1)/PV(lp+1). Double-buffering
  k_s/v_s (+64 KiB → ~163 KiB ≤ sm_120 opt-in max 227 KiB, 1 block/SM
  preserved) removes the per-page barrier stall.
- Expected: 5-15% of attention time (K-deq 22% @250k is the overlap candidate).
  Cost: ~1-2d (smem layout, barrier restructure).
- **Byte-identity class: YES** (scheduling only — verify with the byte-diff
  battery; any diff is a defect).

### L3 — pre-pass query fold, enabling int8 QK MMA (algorithm, acceptance gate)
- Site: lever-1 code already in tree behind `NINFER_KVARN_DECODE=int8qk`
  (packed.inc, `<bool Int8QK>` template). Lever 1 measured −13% net because
  the per-page query fold (quantize q_rot·kc0 → s8 + warp reductions) sits on
  the MMA critical path.
- Lever: hoist the fold into a pre-pass kernel (per layer: grid-stride over
  pages × kv_heads, T=1 → one query, no reductions on the critical path, ~20 MB
  scratch for q8/c0 at 160k). Packed kernel then only loads q8.
- Expected: recovers the 3.9x s8-MMA win on ~18% of pass-1 → ~3-8% of decode
  time at long ctx. Cost: ~2-3d (pre-pass + fold correctness + bench).
- **Byte-identity class: NO** (int8 numerics ≠ bf16 path; lever-1 rel-l2
  3.78e-03 < 0.0044 gate). Gate: acceptance ±0.5 pt + A2 identity.
- VRAM cost: q8 scratch per layer (~20 MB at 160k, ~64 MB at 250k) — verify
  headroom against the 250k cap.

### L4 — unified gqa_decode<TKV> (docs/104 §3, structural)
- The endgame: one kernel family, dequant prologue, canonical body; kills the
  per-page dequant+MMA split and the deferred V-FWHT epilogue for good.
  Weeks of kernel work; gate behind Phase 1 (launch config) + this phase's L1-L3.
- Include in this work order ONLY as scope marker; do not start it here.

## 4. Hypothesis verification (step 0, before any lever)

The kernel has a phase-skip debug knob (`kDbgSkip` bits: 1=V-deq, 2=K-deq+
prefetch wait, 4=QK, 8=PV, 64=K-deq). Profile per-phase cost at 40k/160k/250k
(T=1 AND T=4) on the pre-L build; confirm the lever-1 breakdown (computed at
T=4 @40k/250k) at 160k and T=1. This decides L2 vs L3 priority: if dequant
dominates (K-deq+V-deq > 40% of pass-1) → L2 first; if MMA dominates → L3 first.

## 5. Measurement plan

Protocol: baseline v1 (greedy, 192-token budget, ITERS=1, MTP on, serve-log
prefill t/s). Cells: 10k/40k/80k/160k/250k per variant, capped by VRAM
(kvarn ≤250k, int8 ≤160k, bf16 ≤80k — guard ALL_CACHE_SPECS).

Per lever (one at a time, smallest first):
1. L-class A/B: L2 → byte-diff battery (robot/sunsets + H5 + 6-prompt +
   temp>0 seeds 1-6, per docs/105 harness). L1/L3 → acceptance battery
   (greedy acceptance ±0.5 pt per cell, A2 identity, determinism 2× identical).
2. Perf: guard cells above, decode t/s + tok/round + prefill t/s.
3. Pass bar (argued): **kvarn@160k ≥ 54 t/s** (within 10% of int8@160k=60.0;
   today 47.1) AND **kvarn@80k within 5% of int8@80k** (≥60.2; today 58.2).
   Rationale: 160k is the comparison ceiling (int8 VRAM cap) and the G2b-
   adjacent product context; 80k is where the gap first becomes visible. A
   lever that doesn't move 160k is not worth the risk class.
4. No cell regresses >2%; G2a kvarn@40k ≥69 re-verified on the final build
   (post-docs/105 fix).
5. Full CI after each landed lever.

## 6. Non-goals

- Prefill (Phase 3 — docs/113 draft, Agent 2).
- MTP acceptance (kvarn's is already the best of the three).
- GDN gating T=1 (docs/105, in flight — re-measure the gap after it lands).
- KV uniformity Phase 1 (launch config table — in flight; coordinate L1 with it).
- D-21 gate itself; lossless KV (impossible at 4/2-bit, docs/104 §5).

## 7. Sequencing & risk

L1 (~1h) → step-0 profiling (~1h) → L2 (~1-2d, byte-identity, lowest risk,
land first) → L3 (~2-3d, acceptance gate) → re-measure → L4 scoping.
Byte-identity contract: any lever that changes accumulation/reduction order is
acceptance-gate class, never byte-identity class — L2 must stay byte-identical
or it is reclassified. GPU: single serial resource; each lever's A/B is one
guard run (~30-60 min) — the plan owner sequences slots.

## 8. References

- Baseline v1: results/official_baseline_v1_20260829.md (+ raw JSONs)
- docs/104 (R1-R5, Phase 2 design §3, gate §4), docs/82 (A2 lever, lever
  history), results/doc83_lever1_int8_finding.md (negative lever + profiling),
  docs/105 (GDN fix, in flight), docs/106 (Phase 1, in flight)
- Kernel: packed.inc (page loop ~L520-560, dequant lambdas ~L233-300),
  gqa_attention_kvarn.cuh:827, launcher gqa_attention_kvarn.cu:148-228
