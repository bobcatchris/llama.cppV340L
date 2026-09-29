# 82 — KVarN decode: kill the O(N) dequant transforms, then remove the wall, then MTP/D-21, then magic-dict + ngram + adaptive MTP

**Status:** PARTIALLY SUPERSEDED by **docs/83** (2026-08-27). §0–§1 (the O(N)
dequant diagnosis) stand and are the foundation of docs/83. **Phase A (T=1
acc-FWHT) and Phase B (wall removal via gate) are superseded** — the wall is
abandoned as a non-starter and the real lever is a code-space attention kernel
(docs/83 M1–M4). A1's CPU ref + the banked `acc_fwht`/`stage_tail_v_rotated`
carry forward as docs/83 M1/M2 inputs.

_(original: **LIVE (planned 2026-08-27, user directive). OVERTAKES docs/78 completely.**)_ docs/78 is superseded: its step numbering, G1 ≥150 GB/s gate,
option-(a) re-scope, and "steps 3–4 remain" framing are all void. This doc is
the single plan of record for the KVarN decode effort and the work that
follows it. Owner: agent. QA gate: maintainer. Worktree: `wo/kvarn-hold`
(continues from commit `98db350b`).

## 0. Why docs/78 is superseded (the record, corrected)

docs/78 was written around two numbers that turned out to be wrong or
misattributed. Both are now measured and on the record:

| Claim in docs/78 | Reality (measured 2026-08-27) | Source |
|---|---|---|
| "G1 passed at 184 GB/s" | **Not reproducible.** Real G1 = 104 GB/s @40k, 93 @250k (~23% DRAM efficiency, compute-bound on per-page dequant+FWHT+MMA) | `results/doc78_step2_c2_g1_and_c3_tail_ab.md` §"Correction to the record" |
| "packed-only @40k yields 71 t/s, matches the shadow" | **False.** The 71 t/s @40k was the *shadow* (bf16 flash) path — in-wall decode never reaches the packed kernel (`text_context_impl.h:323-331` early-return). Packed-only @40k = **57.3 t/s** (`NINFER_KVARN_NO_SHADOW=1`, single-run decode_guard) | `results/doc78_step3_scoping.md` §6 |
| "deleting the wall is a free win (−1344 MiB, no regression)" | **Deleting the wall regresses 40k: 71.2 → 57.3 t/s (−20%)** and 10k: 70.5 → 60.8 (−14%). VRAM win confirmed (−1342 MiB/rank) | same |
| "G2 re-scoped via option (a): ≥69 @40k + ≥30 floor @250k" | The ≥69 @40k bar was only ever met by the shadow. Packed-only cannot meet it at current kernel cost. Re-scope is void; gates below replace it | this doc §2 |

What remains **true and carried forward** from docs/78's work:
- The packed decode kernel exists, is numerically verified (Bug A/B, F5 tail
  domain, in-loop cp.async barrier, V code 2× offset — all fixed and QA-read),
  and is ~10× the split baseline (104 vs 10.3 GB/s effective).
- C1 (scale layout) was a false alarm: KVarN scales are runtime VRAM
  (`WorkspaceArena` → `decoder_state.cpp:77/110`), not file-backed; no format
  break, no re-quant needed (`results/c1_scale_layout_versioning_finding.md`).
- The tail A/B sweep (tail ∈ {0,1,8,63}, rel_l2 gate, verified negative test)
  is a real F5 regression guard in `bench_kvarn_attention.cu`.
- Beyond-wall decode today: packed path gives **30.5 t/s @250k** vs the D-18
  collapse of **4.0 t/s** — a ~7.6× win that stands regardless of this plan.
- The architecture diagnosis is unchanged and is the foundation of this plan:
  at long context the per-page dequant+FWHT FLOPs consume exactly the
  bandwidth saving KVarN's compression buys (bf16 KV @250k ≈ same t/s as
  packed @250k). llama.cpp/NVFP4 holds ~390 t/s @195k because it reads KV at
  final precision with **zero dequant FLOPs in the hot loop**.

## 1. The one idea this plan is built on

**Stop doing O(N) transforms in the decode hot loop.** Per-page cost profile
(`results/doc78_step5_option_c_design.md` §1, kDbgSkip sweep, pass-1 = 0.205 ms
@625 pages):

| phase | cost | % |
|---|---|---|
| **V-dequant incl. per-page channel FWHT** | 0.076 ms | **37%** |
| K-dequant (affine) | 0.037 ms | 18% |
| PV MMA | 0.031 ms | 15% |
| loop/barrier overhead | 0.023 ms | 11% |

The dominant term is the per-page V channel-FWHT — an O(N·D) transform that
linearity lets us move to the O(1) side:

- **PV (the fix, A2):** keep `v_s` in the ROTATED domain (drop the inline H,
  keep vvnorm=1/16 → v_s = H(V)/256), accumulate `acc' = Σ p·H(V)/256`, then
  apply **one FWHT per output row** in the epilogue: `H(acc') = Σ p·V_original`
  (H² = D·I). The butterfly maps exactly onto the PV C-fragment layout
  `c = 8n + 2lid + s` — local ops + within-quad shfl only, no smem round-trip
  (design doc §2.3). Tail V is staged rotated to match (§2.4); merge/÷l
  commutes with H (§2.5). Expected: V-dequant 0.076 → ~0.037, **~19% of
  pass-1** → projected packed @40k 57.3 → ~68–70 t/s.
- **QK (deprioritized, A2b only if needed):** K-dequant is affine-only (no
  rotation in the hot loop today) and QK softmax is 3% of pass-1. The earlier
  "GEMM-on-codes / rotate Q" proposals targeted the smaller term at higher
  numerics risk; revisit only if A2 under-delivers GB.

> **Mechanism correction (supersedes docs/78 step 5 and
> `results/g2_rescope_and_option_c_proposal.md` §4):** "rotate softmax weights
> p to move the V FWHT out of the loop" was wrong as stated (the FWHT is a
> 256×256 map on the *channel* dim; scalars p can't absorb it). The correct
> O(1)-side move is the **accumulator** transform above — same idea, applied
> where the linearity actually holds. The prior deferred-FWHT attempt failed
> on fragment-mapping implementation (F7), not theory; the design doc §2.2–2.3
> fixes the mapping and the new bench (realistic zero-mean scales + tail sweep)
> catches this bug class.

### Open design question (resolve in A1 before writing kernels)
Confirm the fragment-mapped butterfly of §2.3 against a CPU reference for all
256 channel values at random p/V (the exact failure mode of F7). If any bit of
the mapping is wrong, rel_l2 will show it — do not trust the paper derivation
alone.

## 2. Gates (replace docs/78 G1–G4 and the option-(a) re-scope)

Measured in the step-0 harness / decode_guard, exclusive GPU, 3-run median,
model identity (path, size, mtime, meta sha) recorded in every results JSON:

- **GA (kernel):** effective packed-code read **≥150 GB/s** at N=40k and
  N=250k, T=1 — the docs/78 G1 bar, now actually required.
- **GB (system, the wall-deletion gate):** decode_guard **≥69 t/s @40k AND
  ≥69 t/s @250k** on the packed-only build (`NINFER_KVARN_NO_SHADOW=1`),
  3-run median. This is the original docs/78 G2, un-re-scoped. @250k is the
  hard part; if GA passes but GB-@250k misses, the wall stays (see §4) and we
  report the number — no partial wins.
- **GC (regression):** MTP acceptance within ±2pp of baseline; T19 ≥450 tok/s;
  battery `--mode ci` green; byte-determinism A/B vs the current build
  (0 mismatch over sampled greedy sequence); tail A/B sweep (C3) green.

## 3. Phase A — O(1)-side transform packed decode kernel (the optimization)

Worktree `wo/kvarn-hold`. Steps commit individually; gates at A2/A3.

- **A1 — Design validation** (CPU-only, no GPU). Most of the design already
  exists: `results/doc78_step5_option_c_design.md` (phase profile, deferred
  acc-FWHT math, fragment-mapped butterfly §2.3, tail domain §2.4, merge
  compatibility §2.5). Remaining: CPU reference implementation of the
  epilogue butterfly checked against direct FWHT on all 256 channels at
  random p/V (the F7 failure mode); precision budget for the extra bf16
  round-trip of acc' vs the existing rel_l2 floor (≈0.004).
  **Gate:** CPU reference matches to < 1e-6; expected rel_l2 ≤ 5e-3 with margin.
- **A2 — Kernel implementation.** Restructure the existing packed kernel per
  the design doc: drop inline V-FWHT (vvnorm stays 1/16, v_s rotated), add
  epilogue acc-FWHT (local + within-quad shfl), rotate tail V staging. Keep
  the current kernel behind `NINFER_KVARN_DECODE=dequant` as A/B reference.
  Determinism: fixed-order split/merge preserved.
  **Gate:** test_kvarn_gqa green (incl. F5/tail cases) at rel_l2 ≤ 5e-3; bench
  tail A/B sweep {0,1,8,63} green vs dequant kernel at rel_l2 < 1e-2;
  kDbgSkip sweep shows V-dequant cost 0.076 → ~0.037 ms.
- **A3 — Measurement.** Step-0 harness G1 @40k/250k (3-run median);
  decode_guard @10k/40k/250k on `NINFER_KVARN_NO_SHADOW=1` build (3-run
  median). **Gate GA + GB.** If GB-@40k passes but GB-@250k misses: see §4
  fallback — the wall decision at @40k is separable from the @250k target.
  If GB-@40k misses: report, stop, re-plan (option A2b K-path or hybrid
  forever).

## 4. Phase B — Wall removal (only if A3 passes)

- **B1:** route ALL decode (in-wall and beyond-wall, T=1 and small-T MTP
  verify) to the packed kernel; remove the `need_pages <= staged_pages`
  early-return in `kvarn_attend_text`/`kvarn_attend_mtp`.
- **B2:** delete the shadow surface (docs/78 step-3 scoping §3 map is still
  accurate): `kKvarnStagedBudgetBytes`, `kvarn_bind_staged_shadow`, staging
  ops, aliasing machinery (docs/75), revert 98b5395e's wall raise.
  KV residency = packed pool only. Keep `NINFER_KVARN_NO_SHADOW` as a no-op
  for one release cycle, then remove.
- **B3 — Gate G3:** ctest `-R ninfer_kvarn|ninfer_tp2_budget` green; budget
  preflight shows **−1344 MiB/rank** at the 250k cap (already confirmed
  achievable: 13707 vs 15049 MiB); k8v8 @96k fit re-checked (closes H6);
  byte-determinism A/B vs pre-B1 build.
- **B4 — Gate G4:** `bash tools/ops/run_ci.sh --full` from this tree; fresh
  server serves a real multi-turn long-prefill request through packed-only
  decode; launch command + clocks in the report; results committed.

If A3 fails GB: the wall stays as-is (current step-2 hybrid: shadow in-wall,
packed beyond-wall). That state is shippable on its own — it already kills
D-18's 4.0 t/s beyond-wall at 30.5 t/s — and Phase A can be revisited with a
re-scoped GB later. **The wall is deleted only by a passing GB, never by
schedule.**

**Partial-pass handling (new, 2026-08-27):** the @40k and @250k bars answer
different questions and may be decoupled. If GB-@40k passes (packed ≥69) but
GB-@250k misses: B1–B4 may proceed on the @40k bar alone — deleting the wall
removes the 40.4k seam entirely, so *every* context then runs the packed path,
and the @250k number becomes "packed decode at 250k" (30.5+ today) with no
shadow to fall back to anyway. The residual risk (contexts between ~40k and
~128k get packed speed instead of shadow speed) must be accepted explicitly by
the plan owner in writing before B1, with the measured @64k/@128k decode_guard
numbers attached. If GB-@40k misses: no deletion, full stop.

## 5. Phase C — MTP fixed (D-21)

After B4 (or after A3-fail + ship of the hybrid state). Branch
`wo/kvarn-d21`, plan `docs/79_mtp_chunked_prefill_d21_boundary_fix_plan.md`.

- **C1:** land the D-21 fix per docs/79 (MTP chunked-prefill 1024-boundary
  divergence; static narrowing already done — state-transition vs
  shared-scratch corruption).
- **C2 — Gate:** mtp_long re-test (H5) on a non-degenerate >512-token prompt:
  MTP-vs-plain byte-identical; full CI green; docs/50 D-21 line closed with
  evidence.

Sequencing note: C is independent of A/B code paths (different kernel), but
runs after them because (a) one live server at a time, and (b) D-21's A/B
byte-compare baseline should be the post-wall-removal build so the fix is
validated against the final attention path.

## 6. Phase D — Magic dictionary + ngram seed + adaptive MTP (integration)

After C2. Three work orders already exist; this phase is their ordered
landing on main:

1. **D1 — Magic dictionary Phase 1** (`wo/magic-dict`, doc 100): corpus
   collector, offline/CPU-only. Land + merge when its QA clears.
2. **D2 — ngram-mod pool seed** (`wo/mtp-adaptive`, doc 80): steps 1–5
   committed, CPU-proven; live baseline/seeded delta parked. Run the live
   step (needs server), gate on acceptance delta vs unseeded baseline, merge.
3. **D3 — Adaptive MTP** (`wo/mtp-adaptive`, docs/69 work order): land the
   adaptive scheduling on top of D2; gate per its work order.

Ordering within D: D1 → D2 → D3 (D2's seed file is built from D1's corpus
pipeline; D3 consumes D2's pool). Each merges independently with its own
evidence; docs/98 status header updated as each lands.

## 7. Roadmap state (single source of truth, 2026-08-27)

| # | Item | Branch/worktree | State | Gate |
|---|---|---|---|---|
| A1 | Deferred acc-FWHT design validation (CPU ref of butterfly) | wo/kvarn-hold | **NEXT** (CPU-only, can start now; design 90% done in `results/doc78_step5_option_c_design.md`) | CPU ref < 1e-6; rel_l2 budget ≤ 5e-3 with margin |
| A2 | Kernel: drop inline V-FWHT + epilogue acc-FWHT + rotated tail V | wo/kvarn-hold | queued | unit + tail A/B green; V-dequant 0.076→~0.037 ms |
| A3 | G1 + GB measurement | wo/kvarn-hold | queued | GA ≥150 GB/s @40k/250k; GB ≥69 t/s @40k/250k packed-only |
| B1–B4 | Wall removal + CI + live proof | wo/kvarn-hold | **only if A3 passes** | G3 (−1344 MiB, ctest) + G4 (full CI, live) |
| — | (else) ship hybrid state: shadow in-wall, packed beyond-wall | wo/kvarn-hold | fallback | GC regression gates only |
| C1–C2 | MTP D-21 fix | wo/kvarn-d21 | queued (docs/79 LIVE) | mtp_long byte-identical + CI |
| D1 | Magic dict Phase 1 (collector) | wo/magic-dict | CPU work exists, QA pending | per doc 100 |
| D2 | ngram-mod seed live step | wo/mtp-adaptive | steps 1–5 committed; live parked | acceptance delta vs baseline |
| D3 | Adaptive MTP | wo/mtp-adaptive | queued | per docs/69 |

Merge hold (docs/77 §8) remains in effect for all of the above: no merges to
main until H0'–H5 clear; this doc's Phase B completion is what clears H0'.

## 8. Constraints (carried from docs/78, unchanged)

- Worktree only; commit per step; no merges while docs/77 §8 hold is in effect.
- Do not change: k4/v2 code format (unless A1 chooses option (i) — then it's a
  quantizer-side re-quant with its own gate), page layout, block-table
  mapping, T19 ≥450, prefill materialize path.
- Live end-to-end before "done" for every server-facing step.
- One live server at a time; port 8091 swap protocol; no concurrent builds.
- Determinism: greedy A/B byte-compare at every routing change.
- Every results JSON records model identity (path/size/mtime/meta sha) and
  the exact build commit.

## 9. Risks

- **A2 fragment mapping** (the F7 class): the epilogue butterfly must match
  the PV C-fragment layout exactly; a wrong bit = structurally wrong output.
  Mitigation: A1's CPU reference + the realistic-scale bench + tail sweep all
  gate A2. Rollback is one flag (`NINFER_KVARN_DECODE=dequant`).
- **Projection risk:** A2's ~19% pass-1 saving projects to 57.3 → ~68–70
  t/s @40k — i.e., it may land *at* the GB-@40k bar, not clearly above it.
  If A3 measures < 69: either accept the partial-pass path in §4 (needs
  @64k/@128k data + plan-owner sign-off) or do A2b (K-path code-space dot,
  deprioritized in the design doc §4) as a second increment. Decide at A3
  with the measured number, not before.
- **GB-@250k misses even after A2:** expected — the acc-FWHT fix removes the
  V transform but K-dequant (18%) and PV MMA (15%) remain O(N). 250k ≥69
  likely needs the full code-space treatment or is a product re-scope. A3
  measures it; the number decides.
- **Scope creep in Phase B:** B is deletion + routing. If it grows beyond
  that, stop and re-scope (docs/78 §7 clause, carried over).
- **Server contention:** A3/B4/C2/D2 all need exclusive GPU. Sequence them;
  do not interleave builds with other agents' servers.

## 10. Supersession record

- **docs/78** — SUPERSEDED by this doc in full (steps, gates, re-scope).
  Kept for history; its measurement artifacts under `results/doc78_*` remain
  valid evidence and are cited above.
- **docs/74 step 3 / C5, docs/76 ordering** — still deleted/superseded as
  before (docs/78 §3 carried over).
- **docs/77 §8 H0'** — updated to point at this doc when Phase B lands;
  until then H0' text stands but its "G1 make-or-break" clause is replaced by
  GA+GB.
- **`results/g2_rescope_and_option_c_proposal.md`** — §2 (analysis) remains
  valid evidence; §3 option-(a) re-scope VOID; §4 mechanism SUPERSEDED by
  this doc §1.
- **`results/doc78_step5_option_c_design.md`** — ADOPTED as the Phase A design
  basis (corrected in place where it conflicted with this doc's gates).

## 11. Immediate worktree hygiene (before A1 starts)

- `tests/test_kvarn_gqa.cpp` still carries ~25 DEBUG printf lines from the
  step-1/2 debugging (committed in 1304ada9). Remove or gate behind an env var.
- Untracked: `results/packed_fix_check_decode_*.json`,
  `results/step2_packed_decode_*.json` — commit with the A1 work or delete.
- `QA_FEEDBACK_FOR_AGENT.txt` is untracked and now historical; fold any
  still-open items into this doc and archive it.
- docs/78 header should carry a one-line "SUPERSEDED by docs/82 (2026-08-27)"
  banner so nobody plans against its stale step numbering.
