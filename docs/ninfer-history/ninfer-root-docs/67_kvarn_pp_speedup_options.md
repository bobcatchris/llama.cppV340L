# 67 — KVarN pp (prompt processing) speedup options — D-18 follow-up plan

**Status:** PROPOSED — planning document, no code written yet.
**Baseline:** D-18 merged to main (`ad95d5af`, 2-pass KVarN prefill).
**Mission:** raise beyond-wall pp toward the staged-path rate (~700 tok/s) without
violating any D-18 constraint (docs/66 §7): no new persistent allocations,
in-capacity behavior bit-identical, startup VRAM unchanged at 250k.

---

## 1. Where we are (measured)

| Scenario | Rate | Source |
|---|---|---|
| In-capacity (≤30.7k, staged BF16 flash) | ~720 tok/s instantaneous | serve.log |
| Beyond wall, 60k prompt, D-18 build | ~475 tok/s avg (stable ×8) | d18_loop.log |
| Beyond wall, 225k prompt | ~483 tok/s avg, decaying **~364 tok/s @ 220k** | serve.log req 3 |
| Pre-D-18 fused kernel beyond wall | 12.6 → 4.0 tok/s | docs/66 |

The decay is O(T²/C): every chunk × every layer re-materializes the ENTIRE
visible history into a paged BF16 temp (`gqa_attention_kvarn_materialize_kernel`)
and pass 2 (`gqa_attention_prefill_bf16_kernel`) reads it back. Both passes are
serialized on one stream.

## 2. Cost structure of one past-wall chunk (2048 tok, per layer)

1. Pass 1: read codes+scales, warp-FWHT dequant, write temp to DRAM
   (~64 KiB/tile/kv_head written).
2. Pass 2: grid `(q_blocks, q_head)` — **each q-head CTA loops ALL KV blocks
   itself**, so every temp tile is read by 12 CTAs / 2 kv_heads = **~6× redundant
   DRAM reads** (working set ≫ L2).
3. Shadow region (up to 30.7k tokens) is ALSO copied into the temp and re-read,
   even though it already sits in the exact layout flash wants.
4. Passes run strictly serially (same stream).

## 3. Options, in implementation order

### Option A — GQA-shared pass 2 (biggest structural win)

**Change:** launch pass 2 as one CTA per `(q_block, kv_head)` with the GQA group
loop inside, instead of `(q_block, q_head)`. Pattern already proven in-tree by
the step-1 MMA kernel (`gqa_attention_kvarn_mma.cuh`).

**Why:** kills the ~6× redundant temp reads identified in §2.2. Pure pass-2
change; pass 1 untouched.

**Effort / risk:** medium / low-medium. Touches the shared bf16 prefill kernel —
must remain bit-identical for the plain BF16/I8 KV paths or be a KVarN-only
variant (prefer a KVarN-only template instantiation to honor docs/66 §7).

**Expected:** removes the dominant read-amplification term; necessary foundation
for C/D. Profile first (ncu, one past-wall chunk) to confirm reads dominate.

**Gate:** fast-loop 40k probe ≥ 2× current beyond-wall rate trend; T10/T18 green;
battery pp WARN (246 vs 267.6) re-checked.

### Option B — Segmented producer/consumer pipeline (hide pass 1 entirely)

**Change:** split tiles into R segments. Stream A materializes segment k+1 while
stream B flashes segment k; merge partial attention outputs with an online-
softmax split-KV merge (flash-decode style: partial O + running m/l per segment,
combined at the end). Double-buffer the temp (two half-sized temps, alternate).

**Why:** pass-1 latency currently fully serializes ahead of pass 2; bench shows
dequant overhead is only ~0.6% when overlappable, so hiding it returns pass 2 to
pure-flash rate.

**Effort / risk:** medium-high / medium. Needs partial-output workspace
(O fragments + m/l state per (q_block, head, segment)) — transient arena
allocations only, sized like the existing temp. No cooperative launch required
(two streams + events), unlike the grid-sync variant rejected in docs/66 §5.9.

**Expected:** recovers most of the remaining pass-1 share; compounds with A.

**Gate:** same as A + verify identical outputs vs serialized path within the
existing rel_l2 tolerance; determinism check (segment merge must use a fixed
order — fp addition order changes are NOT allowed to alter results run-to-run;
fixed segmentation keeps it deterministic).

### Option C — Skip the shadow-region round-trip

**Change:** with B's merge machinery in place, stop copying shadow pages into
the temp. Flash the staged-shadow view directly (it is already in the exact
{D,G,H,P} layout) for keys [0, staged_pages); materialize+flash ONLY packed
pages ≥ staged_pages (+ tail); combine via the same partial merge.

**Why:** eliminates a write+read of up to ~30.7k tokens × 2 KiB × 2 (K+V)
≈ 126 MB per layer per chunk — the majority of traffic for prompts just past
the wall (60k probe: shadow is ~half of history).

**Effort / risk:** low-medium once B exists (shares merge logic); standalone it
would need its own merge, so schedule after B.

**Expected:** large win specifically in the 30k–80k regime where T19 lives;
diminishing at very long contexts (shadow becomes a small fraction).

### Option D — Coalesce scale gathers in the materialize code branch *(last)*

**Change:** stage each tile's 1152-float scale field through shared memory with
one coalesced load (4.5 KiB), instead of per-lane strided gathers
`T[F * idx]` (~1 sector per float today).

**Placement rationale (owner decision):** scheduled LAST. It is a micro-
optimization inside pass 1; after A/B/C shrink pass-1's relative share its
absolute payoff is small, and the owner prefers to land structural wins first.
Note: the materialize kernel runs on the prefill path only — the decode path
(`gqa_attention_kvarn_kernel`, `tokens == 1` branch) does not execute it, so no
decode regression is expected from this change itself; still, per owner call it
lands last and must show decode-neutral battery numbers before merging.

**Effort / risk:** small / low. Localized to `gqa_attention_kvarn_flash.cuh`
code branch.

**Gate:** bench_kvarn_mma dequant-overhead delta; unit rel_l2 unchanged
(bit-exact expected — same math, different load path); full must-pass subset.

## 4. Explicitly deferred / parked

- **`--prefill-chunk 4096`** (runtime flag, no code): halves rematerialization
  sweep count (∝ T/C) but lengthens the decode stall window between chunk
  boundaries — trades decode interleaving/latency for pp. Owner: keep as a
  manual experiment only, not part of this plan.
- **Persistent dequant mirror** (cache materialized old pages across chunks):
  would turn O(T²/C) traffic into ~O(T), but VRAM headroom at the 250k config
  is ~zero (docs/66 §7, 14,635 MiB/rank). Parked unless capacity is traded.
- **Full grid-sync cooperative fusion**: superseded by B at far lower risk.
- **L2 persistence hints** (`cudaAccessPolicyWindow`): marginal at GB-scale
  working sets; revisit only after A shrinks the working set.

## 5. Pre-work (before any option)

1. **Profile** one 2048-token past-wall chunk (ncu/nsys): attribute time between
   `materialize_kernel`, `prefill_bf16_kernel`, and model non-attention parts.
   Confirms whether reads (A) or serialization (B) dominate before committing.
2. **Re-run the battery twice** to resolve the open flags from the 06:08 report:
   - pp micro-bench 246 vs baseline 267.6 (WARN −8.1%) — staged path, likely
     noise/clocks; confirm.
   - `"MTP long-prefill >512 tok (MTP==plain)"` FAIL (baseline true → false,
     1793-tok prompt) — correctness regression on the long-prefill path.
     **Blocker for all options below until root-caused.**
3. Decode note: `tokens == 1` still uses the legacy per-q-head fused kernel
   (full-history dequant per step, work-order Step 2 open). Out of scope here
   but will dominate long-context latency after pp is fixed.

## 6. Constraints carried over from docs/66 §7 (all options)

- Worktree only (`wo/kvarn-d18` or successor branch); commit per step.
- Live server verification before any step is "done" (real chunked prefill past
  the wall, not unit tests alone).
- No new persistent device allocations; startup VRAM at 250k unchanged.
- In-capacity staged path and I8/BF16 KV paths bit-identical.
- Gates: fast-loop 40k probe → full T19 (≥450 tok/s, do not modify threshold) →
  must-pass subset T1–T18 → MTP acceptance within ~5pp of int8 same-workload.
