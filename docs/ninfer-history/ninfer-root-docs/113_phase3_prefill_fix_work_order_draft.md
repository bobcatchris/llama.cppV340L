# 113 — Phase 3: KVarN prefill fix (delete the materialize temp): work order DRAFT

> ## ⚠ RESCOPED 2026-08-31 — the premise of this work order is measurably wrong
>
> **Do not execute Steps 0-2 as written.** Measured attribution is in
> `results/113_prefill_nsys_attribution.md` (9167829f); the design response is
> `docs/119_kvarn_code_layout_and_quantize_critical_path.md` (87130e3d).
>
> This order targets the **materialize temp** on the belief that it causes the −12 to −16%
> KVarN prefill deficit. nsys on identical 21 829-token prompts (both runs clean, rc=0) says:
>
> | | GPU kernel time |
> |---|---|
> | bf16 prefill | 27.74 s |
> | kvarn prefill | 31.38 s |
> | of which `quantize_tile_kernel` | **3.61 s = 98.9% of the gap** |
> | of which `gqa_attention_kvarn_materialize_kernel` | **0.15 s = 4.1% of the gap** |
>
> Deleting the materialize kernel **perfectly** recovers ~0.5% of prefill time against a 12.4%
> deficit. Step 4's gate (±5% of bf16) is therefore **not reachable by this plan** — the direct
> route would have to be perfect and would still leave ~11.9% on the table.
>
> Conversely `31.38 − 3.61 = 27.77 ≈ 27.74`: taking **quantization** off the critical path closes
> the entire gap with no algorithm, numerics or layout change. That is the real Phase 3 work.
>
> Two further blockers found while auditing the code this order assumes is usable:
> 1. `NINFER_KVARN_PREFILL` **does not exist** — Step 0 must build it, not just use it.
> 2. The direct kernel is **unwired dead code with no tail support** (no `tail_k`/`tail_v`/
>    `tail_count`/`packed_pages` params, which the materialize route passes). It therefore
>    silently drops every uncommitted key. Step 0's expectation that direct will merely be
>    "SLOWER ... that is the data point, not a bug" is wrong: on any request with a non-empty
>    tail it is **WRONG**, so the A/B would be measuring a broken route.
>
> ### Revised step order
> - **Step 0′** — profile/attribute `quantize_tile_kernel`. Its cost is `kSinkhornIters = 16`
>   full-tile passes plus an 8-stage FWHT in ~72 KB smem — algorithm, not memory.
> - **Step 1′** — free wins in the commit path: drop the two per-head D2D `cudaMemcpyAsync`
>   copies by quantizing straight into the destination page; de-batch the serialized per-head
>   launch loop.
> - **Step 2′** — move commit off the critical path (side stream + event dependency). Justified
>   because a quantized page is consumed only by a *later* attention reading it as history; the
>   producing chunk reads its own keys via the raw/tail path.
> - **Step 3′** — tail support in the direct kernel + build the env gate + write a **prefill FP64
>   oracle first** (pattern: `tests/slice4_kvarn_test.cu` — real codes, non-identity page
>   permutation, `n_layers>1`, mixed committed/tail). Only then is a direct-vs-materialize A/B
>   meaningful.
> - **Keep** the direct-route work, but **re-label it as a capacity change**: removing a
>   full-size paged BF16 temp (~2 GB at 250k) still matters because it competes with KV capacity,
>   which is the product's whole point. Gate it on memory saved, not t/s.
> - **Demote** the original Steps 0-2 to a footnote under this banner.
>
> Everything below is retained unedited as the original draft for reference.
>
**Status:** DRAFT (plan-owner review pending) — drafted 2026-08-29 per docs/114
from the Phase 3 audit (`results/106_prefill_materialize_audit.md`, commit
8adc2b89) and official baseline v1. Scope: docs/104 Phase 3. Companion:
Phase 4 (prefix-reuse unification) audit `results/106_prefix_reuse_audit.md`
(08eea70b) → its own work order (number TBD) after this one lands.

**Mission:** KVarN prefill t/s reaches parity with BF16/INT8 (within ±5% at
40k/80k/160k, ≥ baseline-v1 number at every cell) by deleting the per-variant
materialize temp and routing large-T prefill through the in-kernel-dequant
direct-read flash. Done = gate §5 passed, closeout doc committed, decode side
provably untouched.

Read this document fully before touching code.

---

## 1. Context (60-second version)

KVarN prefill is the only variant that does a second pass over the KV:
materialize packed→BF16 into a paged temp, then run the proven BF16 flash over
the temp. BF16/INT8 flash over their paged cache directly. The extra pass costs
**−12 to −16% prefill t/s, roughly flat across 10k→160k** (baseline v1,
wall-derived): 720 vs 834 (10k), 598 vs 711 (80k, worst −15.9%), 534 vs 614
(160k, −13.1%), 466 (250k, no comparison). Prefill is 88→99.7% of wall time
10k→250k, so this IS the user-visible number.

The mechanism is linear-in-tokens (extra full-KV write + re-read), **not** a
super-linear blowup — the 2-pass split already removed the old O(n²) regime
(docs/66: 409 vs 97 ms/call). What remains is the temp itself.

The fix machinery exists: `gqa_attention_kvarn_direct_kernel`
(`src/ops/kernel/gqa_attention_kvarn_direct.cuh`) has identical dequant math to
the materialize kernel and stages dequantized pages straight into the FA2
swizzled smem tiles (no temp). It is not used for T>1 prefill today because the
tiled route measured latency-bound (~20 GB/s) at T=1 (D-19 step 1d) — the work
here is to make the large-T route fast enough that direct ≥ materialize+flash.

**Must NOT change:** decode path (packed T≤6 kernels + byte-identity battery),
KVarN numerics (storage loss is the only allowed variant difference, docs/104),
D-21/D-16 closeouts, GDN work, prefix-reuse capacity (D-20).

**Already built and verified (do not redo):**
- Materialize + direct kernels, same dequant math (docs/71 M2, QA-verified).
- Phase 3 audit + gate design (results/106_prefill_materialize_audit.md).
- Baseline v1 wall-derived prefill t/s per cell (results/official_baseline_v1_20260829.md §3 + per-cell JSONs).

## 2. Environment & build/test

- Worktree: `~/ninfer/worktrees/wo-kv-uniform`, branch `wo/kv-uniform`
  (rebased onto wo/kvarn-hold head 5bfd7071 on 2026-08-29; includes the wip GDN
  T=1 fix a73fe948 — do not touch it).
- Build: `cmake --build build -j 16`. Tests: `/usr/bin/ctest` (never the
  PATH `ctest`). Server tests through `bash tools/ops/run_ci.sh` from the
  worktree root (fast gate) / `--full` (closeout only).
- **GPU is a serial resource — a slot is assigned by the plan owner in the
  Telegram handoff; do not start one without it.** Single server at a time;
  `pkill -x ninfer-serve` only; `pgrep -x ninfer-serve` + `nvidia-smi` before
  every launch.
- Model: `/home/intel/models/qwen3_8_27b.ninfer` (meta sha `45912dd83c71a1b3`).
- Commit per step to `wo/kv-uniform`. No merges.

## 3. Code sites (verify before editing — line numbers from the audit)

- Materialize kernel: `src/ops/kernel/gqa_attention_kvarn_flash.cuh:33`
  (`gqa_attention_kvarn_materialize_kernel`) + width-generic `:165`.
- Dispatch: `src/ops/launcher/gqa_attention_kvarn.cu:69-70` —
  `packed_verify = (tokens >= 2 && tokens <= 6) && !force_materialize`; `if
  (tokens > kKvarnSmallTMax && !packed_verify)` → pass 1 materialize + pass 2
  BF16 flash. Note `kKvarnSmallTMax = 1` at `:57` (carries the D-19 step 1d
  live-result comment) — the T>1 route is the materialize route today.
- Direct kernel: `src/ops/kernel/gqa_attention_kvarn_direct.cuh` (identical
  dequant math; stages into FA2 swizzled smem).
- Env gate (new): `NINFER_KVARN_PREFILL=materialize|direct` — default
  `materialize` until Step 4 flips it; FATAL on unknown value.

## 4. Steps

**Step 0 — isolate the materialize cost (GPU, ~1 h).**
Add the env gate (both routes selectable today; `direct` for T>6 is wired but
untuned — expect it to be SLOWER; that is the data point, not a bug). A/B at
10k/40k/80k/160k (kvarn only, MTP on, baseline-v1 protocol, ITERS=1):
wall-derived prefill t/s per route. Deliverable: table in the closeout doc —
materialize-only share = (direct_gap − dequant_compute_estimate) is NOT
required; the end-to-end route delta is the gate metric (audit §2.1).

**Step 1 — wire the direct route for T>6 prefill (CPU+GPU, ~half day).**
Dispatch change behind the env gate; no math changes. Gate: byte-identity vs
the materialize route at 4k/40k (per-variant A/B, same request, diff tokens —
any diff = STOP, per plan-owner review of 8adc2b89) + fast CI 0 fails.

**Step 2 — tune the direct route (GPU, ~1-2 d).**
Make direct ≥ materialize+flash at large T. Tuning surface: tile/block sizing,
cp.async wait-queue depth, smem layout (FA2 swizzled tiles), occupancy. The
D-19 finding (~20 GB/s latency-bound at T=1) bounds what untuned gives you —
measure per-phase (Nsight or phase-skip) before/after each tuning change.
Stop condition: direct within −2% of materialize at 80k with byte-identity
held (Step 1's gate re-run).

**Step 3 — numerics + correctness (GPU, ~half day).**
- Byte-identity vs materialize route: 4k/40k, both cache types if applicable,
  greedy + sampling (temp 1.0 Qwen-thinking), 2× determinism.
- A2 identity (MTP == plain) per variant.
- KVarN 250k must-pass battery unchanged: T1/T2/T3/T5/T8/T10/T11/T15/T16/T17/T18.
- Decode side untouched: packed decode byte-diff battery (robot/sunsets + H5 +
  6-prompt + temp0 seeds 1-6, docs/105 harness) vs the pre-Step-1 build — must
  be IDENTICAL (the prefill route change must not leak into decode).

**Step 4 — flip default + performance gate (GPU, ~1 h).**
Default `NINFER_KVARN_PREFILL=direct`. Full guard pass (kvarn + int8 + bf16 at
their VRAM-capped ladders, baseline-v1 protocol): KVarN prefill t/s within
±5% of BF16 at 40k/80k/160k (int8 at 160k as the co-reference) AND ≥ baseline
v1 at every cell. No cell regresses >2% (any variant — decode included).

**Step 5 — closeout (CPU+GPU).**
Fast gate + `bash tools/ops/run_ci.sh --full` 0 fails (known reds only).
Closeout doc `results/113_prefill_direct_closeout.md`: route A/B table (Step 0),
tuning changes + why, gate numbers vs baseline v1, byte-identity evidence,
remaining risk (untuned T=1 route stays `materialize`-default? — decision:
the T≤kKvarnSmallTMax packed_verify route is untouched either way). Keep both
kernels in tree behind the env gate (deletion is a later cleanup, out of scope
— it touches QA-verified code for zero perf gain).

## 5. The gate (all must pass)

1. KVarN prefill t/s within ±5% of BF16 at 40k/80k/160k; ≥ baseline v1 at every
   cell (wall-derived t/s — serve-log internal clock excludes inter-chunk gaps,
   baseline v1 clock note; do not gate on the internal number).
2. Byte-identity vs the materialize route at 4k/40k (any diff = defect).
3. 250k KVarN must-pass battery unchanged.
4. Decode side: byte-diff battery IDENTICAL vs pre-Step-1 build; no guard cell
   regresses >2% (any variant).
5. Fast gate 0 fails per step; full CI 0 fails at closeout (known reds only).

## 6. Measurement protocol

Baseline v1: greedy + Qwen-thinking sampling, MTP on (draft-tokens 3, k=3),
192-token budget, ITERS=1, 2× RTX 5060 Ti 16 GB, `--devices 0,1`. Prefill t/s =
prompt_tokens / (wall_s − decode_time) from per-cell JSON (wall-derived).
Serve-log prefill progress lines are secondary (internal clock). Record model
meta sha per run (guard JSON carries it).

## 7. Non-goals

- Decode path (Phase 2 = docs/115; separate).
- Phase 4 prefix-reuse unification (audit done; own work order, after this lands).
- KVarN numerics / packed kernel / commit boundaries.
- D-21, D-16 re-opens; GDN T=1 (docs/105, in flight — re-measure after it lands).
- Raising `--prefix-cache-capacity` (D-20).
- Deleting the materialize kernel (kept behind env gate).

## 8. Risks

- **D-19 latency bound** is the main risk: if the direct route cannot be tuned
  to ≥ materialize at large T, the fallback is keeping materialize (default)
  and rescheduling Phase 3 behind the unified-kernel endgame (docs/104 §3 /
  Phase 2 L4). Decision point: end of Step 2.
- Direct route smem/occupancy at large tiles may differ from the FA2 body's
  assumptions — verify 1 CTA/SM occupancy intent is preserved (launcher smem
  opt-in pattern, cf. kv-uniform launch-config commit 90b2ef3c).
- A/B env gate must not leak into committed perf numbers — every gate run
  records which route produced it (closeout doc §routes table).

## 9. References

- results/106_prefill_materialize_audit.md (8adc2b89) — design + gate source.
- results/106_prefix_reuse_audit.md (08eea70b) — Phase 4 companion (not this order).
- results/official_baseline_v1_20260829.md §3 + per-cell JSONs — numbers.
- docs/104 (R4, §3 kernel family, §4 Phase 3 gate), docs/66/68 (2-pass history),
  docs/71 M2 (direct kernel origin), D-19 step 1d (latency-bound finding).
- docs/105 harness (byte-diff batteries), docs/99 (work order template).
