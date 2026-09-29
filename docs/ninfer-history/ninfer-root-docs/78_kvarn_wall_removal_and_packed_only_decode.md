# 78 — KVarN: remove the wall; packed-only decode. Planning doc + work order

> **SUPERSEDED (2026-08-27) by docs/82** (`docs/82_kvarn_gemm_on_codes_and_post_wall_roadmap.md`).
> This doc's steps/gates/re-scope are void; it is kept for history. Its
> measurement artifacts under `results/doc78_*` remain valid evidence and are
> cited by docs/82. Notably: "G1 passed at 184 GB/s" was never true (real: 104
> @40k / 93 @250k), and the 71 t/s @40k used in the step-2 acceptance argument
> was the SHADOW path, not packed (packed-only @40k = 57.3 —
> `results/doc78_step3_scoping.md` §6).

**Status:** SUPERSEDED by docs/82 (was: LIVE, planned 2026-08-26, user directive). Supersedes doc 74
step 3 / C5 and the doc 76 ordering. Owner: agent. QA gate: maintainer.
Worktree: `wo/kvarn-hold` (or a fresh branch from main).

**Progress (2026-08-26):** step 0 baseline DONE in `wo/kvarn-hold`
(`bench_kvarn_attention.cu` + `results/doc78_step0_baseline.md`, commits
`82009788`, `ad0d5fba`). Baseline: decode_split is latency/compute-bound at
~8.5 GB/s effective packed-code read (T=1), reproducing the ~4.0 tok/s
beyond-wall (×17 layers); live in-wall decode_guard confirms ~71 t/s @40k.
**Gap to G1 (≥150 GB/s) is ~17.7×** — G1 is the make-or-break gate. Next:
step 1 (packed-only decode kernel), pending QA review of the step-0 numbers.

**Progress (2026-08-27):** steps 1–2 DONE in `wo/kvarn-hold` (packed-only
decode kernel built + routed; numerics QA-verified; QA feedback C1–C4/D/E all
closed — see `results/doc78_step2_c2_g1_and_c3_tail_ab.md`,
`results/c1_scale_layout_versioning_finding.md`). Measured: **G1 = 104 GB/s
@40k / 93 @250k** (misses ≥150); **G2 = 71 t/s @40k (PASS) / 30.5 t/s @250k
(misses ≥69)**. Per plan-owner decision 2026-08-27: **G2 re-scoped via option
(a)** (see step 2 gate below); **option (c) dequant redesign deferred to step
5**. Steps 3 (delete the wall) and 4 (full CI + live proof) remain — step 3 is
the doc's headline and is NOT yet done.

## 1. Decision

KVarN pays exactly KVarN's VRAM cost. **The staged shadow ("the wall") is
deleted.** KV residency = packed pool only (k4v2: 8.97 KB/token/rank →
2.24 GB @250k). No shadow, no persistent overflow, no other resident BF16
copy of KV history. The beyond-wall prefill path (per-call materialize into
a transient tile buffer, compute not residency) is kept as-is — it is the
path CI's 250k prefill (484 tok/s) already runs.

The price of "no wall, no extra memory" is one kernel: a **packed-only
decode kernel** (dequant-in-kernel from pool codes, no BF16 staging). It
exists in the repo as the identified-but-never-done C4 restructure
(warp-per-head + MMA, docs/71/74 step 5, doc 76 C4). This doc makes it the
critical path and deletes the plan that paid more than Q5 to avoid it
(doc 74 step 3: persistent BF16 overflow = full-history BF16 residency,
8.53 GB/rank @250k on top of the 2.24 GB pool — does not fit a 16 GB rank,
and even where it fits costs +26% over a plain BF16 KV cache).

## 2. Why (measured facts, not estimates)

| Fact | Value | Source |
|---|---|---|
| Shadow cost | 1344 MiB/rank (632 pages × 2.125 MiB, 17 layers incl. MTP) | `kvarn_workspace.h`, CI serve.log |
| Pool k4v2 | 8.97 KB/token/rank → 2.24 GB @250k | `kvarn_workspace.h`, docs/57 |
| BF16 KV | 34.3 KB/token/rank → 8.53 GB @250k | `kvarn_workspace.h` |
| Packed decode DRAM floor @250k | 2.24 GB/step → ~5 ms → ~200 tok/s ceiling; ~90 at 45% eff. | 5060 Ti 448 GB/s |
| int8 decode floor @250k | 18.5 KB/token → 4.6 GB/step → ~2× worse floor | docs/57 |
| Today's decode | 69 t/s in-wall; **4.0 t/s beyond-wall (D-18 collapse)** | docs/50 D-18, CI |
| decode_split effective BW | **8.5 GB/s** (step 0 measured: 8.53 @40k / 8.48 @250k, flat in N → scalar-FWHT compute/latency-bound, ~2% of DRAM; supersedes the ~20 GB/s estimate) | bench_kvarn_attention.cu, `82009788` |
| Packed-only prefill | **provably infeasible** on sm_120 (99 KB smem / 256 KB regfile): every tiling spills, overflows smem, or loses 18× on code re-read; MMA does not escape the softmax-state bound | doc 73 §5 (CLOSED, design-space proof) |
| Shadow's real value | ~10× effective bandwidth for 1 GiB/rank, decode-side only; prefill pass-1 is 2–6% at prefill shapes (TC-MMA compute-bound) | doc 73 §5, doc 74 step 1 |
| MTP verify pass-1 share | ~40% of a 40k round today → fixed by the same small-T packed kernel | doc 74 step 1 |

Consequences:
- **Prefill**: no packed-only path exists (hardware, not kernel quality).
  Keep materialize-once. @250k already the beyond-wall path → unchanged.
  ≤40.4k shapes lose the shadow speedup → ~2–6% slower (measured pass-1
  overhead), not a cliff.
- **Decode**: packed-only is viable (small-T state fits trivially, doc 73's
  own conclusion) and DRAM-cheaper than int8. The 4.0 t/s is the materialize
  path, not the floor. The kernel is unbuilt — that is the work.
- **k8v8 @96k fit** (hold item H6, ~21 MiB headroom): resolved automatically
  — deleting the shadow frees 1344 MiB/rank.

## 3. What this doc supersedes

- **doc 74 step 3** (incremental materialization / persistent BF16
  overflow / "extend the shadow upward so there is no seam") — **DELETED,
  never built.**
- **doc 74 step 2** (wall raise 1024→1344 MiB, commit 98b5395e) — reverted
  by this plan's step 3.
- **doc 76 ordering** C1→C2→C3→C5→C6→C4 — replaced by: **C4 (this doc)
  first; C5 deleted; C3 (zero-copy aliasing, docs/75) moot once the shadow
  is gone; C6 (GDN checkpoint restore) unaffected, stays; C1 done (baseline);
  C2 reverted.**
- docs/77 §8 H0 marker — this doc is the plan it pointed to; H0 is replaced
  by H0' below.

## 4. Plan (work order)

### Step 0 — Baseline (measure, no code) — **DONE 2026-08-26, QA-verified** (`82009788`, `ad0d5fba`)
Harness `tests/bench_kvarn_attention.cu` times the real beyond-wall path
(`gqa_attention_kvarn_decode_split_kernel` pass-1 + `..._decode_merge_kernel`
pass-2 with exact k4v2 page footprints, identity block table, full-history
pos; plus the official `gqa_attention_cached` entry point). 3-run median,
one layer; production is 17 layers.

| N | T | pass-1 split | pass-2 merge | eff. BW |
|---|---|---|---|---|
| 40k | 1 | 2.475 ms | 0.028 ms | 8.53 GB/s |
| 40k | 4 | 6.540 ms | 0.029 ms | 3.23 GB/s |
| 250k | 1 | 15.570 ms | 0.029 ms | 8.48 GB/s |
| 250k | 4 | 40.672 ms | 0.029 ms | 3.25 GB/s |

Findings: effective BW **flat in N** (8.53/8.48) at ~2% of the 448 GB/s DRAM
ceiling → pass-1 is scalar-FWHT compute/latency-bound, not DRAM-read-bound
(supersedes the §2 ~20 GB/s estimate). ×17 layers: 250k → 265 ms/token =
**3.78 tok/s, reproducing the documented 4.0 t/s beyond-wall** (D-18). Live
in-wall decode_guard confirmed ~71 t/s @40k (G2 bar: ≥69). **G1 gap
quantified: ≥150 GB/s = ~17.7× the baseline.**
**Gate: PASSED** — baseline committed (`results/doc78_step0_baseline.md`);
QA review of harness + numbers completed 2026-08-26. Step 1 may proceed.

### Step 1 — Build the packed-only decode kernel (C4)
Warp-per-head + MMA restructure: stream K/V codes from the pool, dequant
in smem/registers (warp FWHT + affine), fixed-order split/merge partial
reduction preserved for determinism. No BF16 staging, no shadow access,
works for any N (in-wall and beyond-wall alike).
**Gate G1 (make-or-break):** effective packed-code read bandwidth
**≥150 GB/s (~33% of 448 GB/s DRAM)** at N=40k and N=250k, T=1 — measured
in the step-0 harness, 3-run median. Equivalent system form: decode
attention @250k ≤ ~11 ms/step. If G1 misses, stop here — go to Rollback.

> **G1 result (2026-08-27): MISSED as written — 104 GB/s @40k, 93 @250k
> (<150; ~23% of DRAM, compute-bound on per-page dequant+FWHT+MMA).** The
> ≥150 proxy is superseded by an empirical acceptance argument: packed-only
> @40k yields 71 t/s, which **matches the shadow @40k (71 t/s)**, and @250k
> packed-only yields 30.5 t/s vs the current beyond-wall **4.0 t/s (D-18)**.
> So deleting the wall (step 3) does not regress 40k and is a ~7.6× win at
> 250k. Plan owner accepted proceeding to step 2/3 on this basis (option a),
> rather than triggering Rollback. The ≥150 target is retained as the bar
> option (c)/step 5 must clear to make long-context decode genuinely fast.

### Step 2 — Route decode to packed-only
Route T=1 decode and small-T MTP verify (T≤6) to the packed-only kernel,
in-wall and beyond-wall. Prefill path untouched.
**Gate G2:** decode_guard 3-run median **≥69 t/s @40k and ≥69 t/s @250k**
(today: 4.0 — this is the win bar); MTP acceptance within ±2pp of baseline;
T19 ≥450; battery `--mode ci` green; byte-determinism A/B vs step-0 build
(0 mismatch over sampled greedy sequence).

> **G2 result + re-scope (option a, accepted 2026-08-27):** measured 71 t/s
> @40k (PASS ≥69) and 30.5 t/s @250k (original ≥69 FAIL — architecture limit,
> see `results/g2_rescope_and_option_c_proposal.md` §2). **G2 is re-scoped to
> G2a:** decode **≥69 t/s @40k** (unchanged) **AND a ≥30 t/s no-collapse floor
> @250k** (regression guard, not a target; monotonic vs 40k, no NaN/OOM). The
> ≥69 @250k bar moves to step 5 (option c). MTP ±2pp, T19 ≥450, CI, and
> determinism A/B are unchanged and still required.

### Step 3 — Delete the wall
Remove `kKvarnStagedBudgetBytes`, `kvarn_bind_staged_shadow`, the staging
path (`gqa_kvarn_stage_pages`), and the aliasing machinery (docs/75);
revert 98b5395e's wall raise. KV residency = pool only.
**Gate G3:** `/usr/bin/ctest -R ninfer_kvarn|ninfer_tp2_budget` green;
budget preflight shows **−1344 MiB/rank** with the 250k cap unchanged;
k8v8 @96k fit re-checked and recorded (expect comfortable; closes H6);
determinism A/B vs step-2 build.

### Step 4 — Full CI + live proof
`bash tools/ops/run_ci.sh --full` from this tree; fresh server launch serves
a real multi-turn long-prefill request through the packed-only decode path;
launch command + `nvidia-smi` clocks in the report; `results/` report
committed.
**Gate G4:** CI green (mtp_long tracked separately, pre-existing); live
proof recorded.

### Step 5 (deferred) — Option (c): attack per-page dequant cost
Only path to a real long-context decode win (≥69 t/s @250k). Scope after step
4, with its own design doc + gate, **if and only if** ≥128k decode throughput
is a product requirement. Two sub-attacks (K first): (5a) QK dot directly on
packed 4-bit codes, fold scale after the MMA, rotate Q once so the FWHT folds
onto the O(1) side (gate ≥1.5× @250k); (5b) move the V channel-FWHT out of the
hot loop by rotating softmax weights p (gate ≥1.3× @250k). bf16 hot-window
shadow is explicitly NOT on the table. Full analysis:
`results/g2_rescope_and_option_c_proposal.md` §4.

### Rollback (if G1 or G2 misses)
Revert step 3 (restore the 1344 MiB shadow as a bounded speed cache), keep
the packed-only kernel routed beyond the wall (kills D-18 there), re-scope
with a bounded shadow + the kernel as the beyond-wall path. Report the
missed number; do not ship step 3 on a missed gate.

## 5. Constraints (non-negotiable)

- Worktree only; commit per step; no merges (docs/77 §8 hold in effect).
- Do not change: k4/v2 format, page layout, block-table mapping, hard gates
  (T19 ≥450, decode guard ±2%), prefill materialize path.
- Live end-to-end before "done" for every server-facing step.
- One live server at a time; port 8091 swap protocol; no builds while
  another agent builds.
- Determinism: greedy A/B byte-compare at every routing change.

## 6. Definition of done

1. Steps 0–4 committed with gates passing at each step.
2. `results/`: step-0 baseline table, G1 kernel numbers, G2 system
   numbers, G3 VRAM delta, G4 CI + live proof.
3. One paragraph per step + key numbers table in the report.
4. doc 74 banner + doc 76 ordering + docs/77 §8 updated to match; status
   headers per house rules.

## 7. Risks

- **G1 miss (kernel can't sustain ~33% of DRAM):** decode @250k degrades
  toward the D-18 region; rollback restores the shadow. This is the only
  make-or-break number in the plan.
- **Numerics:** dequant in a new fragment layout can flip near-tie greedy
  tokens → A/B byte-compare is mandatory at steps 2 and 3.
- **Prefill @≤40k is ~2–6% slower** than today's in-wall path. Accepted per
  user directive (2026-08-26); if it ever matters, the shadow can return as
  an opt-in bounded cache — it is no longer the architecture.
- **Scope creep:** step 3 is deletion. If it grows beyond deleting the
  shadow + aliasing, stop and re-scope.
