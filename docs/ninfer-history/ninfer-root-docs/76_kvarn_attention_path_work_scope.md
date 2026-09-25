> **Landed on main 2026-08-26 as docs/76** (QA review + numbering map: docs/77).
> Original: `wo/kvarn-layout-verify` @ fab93b97, file `docs/74_kvarn_attention_path_work_scope.md` (content as of landing).
> Status as of landing: **SCOPING** — component map for docs/74 work; C1 done (docs/74 step 1 baseline), C2 in flight (wall raise uncommitted at landing).
> In-text references to docs/69–74 use the BRANCH numbering: 69=full-matrix(docs/70), 70=prefix-reuse(docs/72), 71=direct-read(docs/73), 72=attn-work-order(docs/74), 73=staged-layout(docs/75), 74=attn-scope(docs/76).

---

# 76 — KVarN attention-path optimization: Work scope (6 components)

**Status:** SCOPING — decomposition of the KVarN beyond-wall attention
optimization into 6 independently-scoped work components. Owner: coordinates
the parallel work (this branch `wo/kvarn-layout-verify` + `wo/kvarn-pp`).
No code on this doc; it is the map. Two of the six are already DONE or
de-risked here.

**Source of truth (do not re-derive):** `wo/kvarn-pp/docs/72_kvarn_attention_optimization_work_order.md`
(the implementer work order, §6 steps 1–6), `docs/kvarn_deep_seek_review.md`
(§5–§7, §B2, §C4–C6), `docs/71_kvarn_branch_review_and_merge_gates.md` +
`wo/kvarn-pp/docs/71_direct_read_tc_prefill.md` (the direct-read proof), and
`wo-kvarn-layout-verify/docs/73` + `tests/test_kvarn_layout.cpp` (the layout
proof). References to "C<n>" below are the deep-seek-review concern letters,
not these components.

**Invariant that underpins the whole set:** the BF16-attention substrate stays
(k4/v2 format, `[D,G,H,P]` d-fastest layout, split/merge ordering,
determinism). Direct-read *prefill* provably loses (docs/71: spill-bound). Only
the *materialization* and *the single-token decode path* are on the table.

---

## Component 1 — Attribute the beyond-wall cost (baseline, no code)

Build a `bench_kvarn_attention.cu` harness that calls
`gqa_attention_kvarn_cached_launch` directly with fixed `packed_pages`,
`tail_count`, `T`; split pass-1 (materialize) vs pass-2 (flash / small-T). Run
at `packed_pages ∈ {482, 625, 964}`, `T ∈ {1,4,16,512}`, `tail_count ∈
{0,1,31}`; record GPU clocks + VRAM. Output = `results/attention_path_baseline.md`.

**Why it's #1:** every later component needs the pass-1 vs pass-2 split as the
before/after reference. No pass/fail; this is the reference table.

## Component 2 — Raise the staged wall to cover target context (config, no kernel)

Bump `kKvarnStagedBudgetBytes` (`kvarn_workspace.h:30`) so `stage_pages`
covers the decode target: 40k → ~625 pages → ~1.33 GiB (currently 1 GiB ≈ 482
pages ≈ 30.8k token wall). `kvarn_staged_page_capacity` does the math. Confirm
startup VRAM still passes the budget preflight (expected ≈ +0.35 GiB/rank for
40k) and the wall log updates.

**Cheapest lever** — a VRAM carve-out, no kernel change. If 40k does not fit,
keep the wall at ~31k and lean on Component 3/5 (delta-only helps regardless).

## Component 3 — Zero-copy aliasing of the below-wall copy (DONE + proven)

**The problem:** the materialize kernel (`gqa_attention_kvarn_flash.cuh`) copies
every below-wall page from the staged shadow into a fresh per-call temp, and the
work order's step 3 wants to alias instead (point the temp `block_table` at the
shadow for `kb < staged_pages`) — but only if the layouts are compatible.

**The proof (this branch, committed):** the staged shadow (`{D,G,H,P}` d-fastest,
`kvarn_workspace.cpp:111-113`) and the beyond-wall temp (`{D,G,H,P}` d-fastest,
`gqa_attention_kvarn.cu`) are the SAME layout, and `paged_kv_element_offset<D,H>`
indexes the identical element. So the below-wall `kd[g·D+d0] = sk[d0 + D·g]` is a
**straight element copy, not a transpose** — it can be eliminated by aliasing,
given one condition: the shadow + overflow live in a **single
block-table-addressable allocation** and `kKvarnAttnG == kPagedKVPageSize`
(64). `tests/test_kvarn_layout.cpp` (CPU-only, no CUDA) asserts all of this:
`g++ -std=c++17 -O2 -o test_kvarn_layout tests/test_kvarn_layout.cpp && ./test_kvarn_layout`.
Full write-up: `docs/73_kvarn_staged_layout_zero_copy_findings.md`.

**Deliverable here is a regression test + proof, not a kernel edit.** The
kernel change itself is small and belongs with Component 5.

## Component 4 — the decode_split kernel (T=1, single token) + merge

- **What it costs:** page-split over 64 splits; each CTA dequants its page range
  once per (kv_head, split) and reuses for the GQA group — so total dequant is
  ~`pages × kv_heads` (efficient, each page dequant once/kv-head), then a
  fixed-order merge.
- **The issue:** per the branch's own measurement it's latency-bound at ~20 GB/s
  effective — the FWHT/scale math is scalar, no MMA. This is the single-token
  decode path, so it's on the critical path of every generated token.
- **Optimization:** the docs/71-identified restructure — **warp-per-head + MMA
  fragments** for the dequant/FWHT. This is the one concrete kernel optimization
  the branch identified but never did. Also vectorize the score reduction and
  replace the per-quad `__shfl_down_sync` with a proper mask.
- **Effort/risk:** high effort (kernel rewrite), medium risk, but only pays off
  if single-token decode latency is a bottleneck (it is, for MTP-verify-free
  decode).

This is the pp work order's Step 5 (+ the Step 6 cleanup item about the
`__shfl_down_sync` quad mask, which belongs here). It is **the one real kernel
rewrite** in the whole set and should not be started until Components 1–3 have
landed (so its before/after is measurable and the layout is proven).

## Component 5 — Incremental (delta-only) materialization, persistent overflow (structural fix)

The work order's "kill the repeated materialize" structural change: make the
beyond-wall temp **persistent** (arena, like the staged shadow) instead of a
fresh per-call allocation, and materialize **only the new page(s) + the tail**
per call. Below-wall pages are aliased (Component 3). Net: per-call pass-1 goes
from `O(packed_pages)` to `O(delta)`.

- Hold a persistent BF16 overflow buffer (or extend the shadow upward so there's
  no seam).
- **Seam correctness gate (mandatory):** prefix/length exactly on a page boundary
  and one mid-page — greedy output identical to a full-re-prefill reference, and
  the mid-page case restores the BF16 tail (docs/70 §4.2 bit-exactness gate).
- **Determinism:** byte-identical vs pre-change for the same request (A/B compare,
  0 mismatch).

## Component 6 — Small-T MTP-verify in-kernel code read + cleanup/hygiene

Two conditional/cleanup pieces that pair with the set:

- **(conditional) Fuse the small-T MTP-verify path** (T∈{2..6}) to read packed
  codes in-kernel (no BF16 temp). docs/71 ruled this out for *prefill*
  (spill-bound at T=1); **MTP verify (T=4) is a different regime** — re-measure,
  do not inherit the conclusion. Gate on Component 1's attribution.
- **Cleanup/hygiene (low risk, do regardless):** (a) fix the formally-UB
  per-quad `__shfl_down_sync` mask in the score loop
  (`gqa_attention_kvarn_decode_split.inc`); (b) correct the "~90 KiB" smem
  comment (actual `sizeof(KvarnDecodeShared)` ≈ 100,328 B ≈ 98 KiB, 1,048 B
  under the sm_120 cap) and add a compile-time `static_assert`; (c) consolidate
  `kKvarnKCodeBytes` (duplicated in `kvarn_workspace.h` vs width-derived in
  `gqa_attention_kvarn.cuh`) into one width-parameterized definition.

---

## How the 6 map to the implementer work order (`wo/kvarn-pp` §6)

| Scope component | Work-order step | Notes |
|---|---|---|
| C1 | Step 1 | baseline, no code |
| C2 | Step 2 | budget knob only |
| C3 | (Step 3 prerequisite) | **DONE** here — proof + test |
| C4 | Steps 5 (+6 shfl) | the kernel rewrite |
| C5 | Step 3 | the structural fix |
| C6 | Steps 4 + 6 | conditional fuse + hygiene |

**Recommended order:** C1 → C2 → C3(done, review) → C5 → C6 → C4.
C4 is the only high-effort kernel rewrite; start it last so its before/after is
measurable against a stabilized C5 and a proven C3.

## Hard gates (never loosen)

- T19 prefill ≥450 tok/s; decode guard ±2% (3-run median @10k/40k/80k);
  battery T16–T19 @250k via `run_ci.sh --full`.
- Byte-determinism across every step (A/B compare, 0 mismatch).
- k4/v2 format, `[D,G,H,P]` layout, block-table mapping, `GqaKvarnTail`
  semantics unchanged.
- `kKvarnAttnG == kPagedKVPageSize` (the alias invariant; guarded by
  `test_kvarn_layout.cpp`).

## Rejected (do not re-litigate)

- Full direct-read replacement of beyond-wall prefill (docs/71: spill-bound).
- Lowering the BF16 shadow to a lossy q8 to double the wall (changes the
  attention input, breaks T18 restore equivalence).
- Touching `kKvarnStagedBudgetBytes` beyond the VRAM that actually fits (the
  2026-08-24 2.4 GiB bump → startup OOM).
