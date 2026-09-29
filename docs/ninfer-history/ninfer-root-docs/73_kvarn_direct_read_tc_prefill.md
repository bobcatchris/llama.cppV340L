> **Landed on main 2026-08-26 as docs/73** (QA review + numbering map: docs/77).
> Original: `wo/kvarn-pp` @ 52af56e5, file `docs/71_direct_read_tc_prefill.md` (content as of landing).
> Status as of landing: **CLOSED** (body §5/§6) — direct-read provably losing on this hardware; kernel kept gated, NOT routed.
> In-text references to docs/69–74 use the BRANCH numbering: 69=full-matrix(docs/70), 70=prefix-reuse(docs/72), 71=direct-read(docs/73), 72=attn-work-order(docs/74), 73=staged-layout(docs/75), 74=attn-scope(docs/76).

---

# docs/73 — Direct-read TC prefill (staged-shadow removal): engineering analysis

**Status:** **CLOSED** — landed on main `fe3d8857` (2026-08-26) as docs/73; direct-read provably losing on this hardware; kernel kept gated, NOT routed (body §5/§6).
**Mission:** replace materialize+flash over-wall prefill (and eventually the
staged shadow) with a direct-read tensor-core prefill kernel, beellama-style.

## 1. Facts established (2026-08-25)

- `gqa_attention_kvarn_qblock_kernel` (gqa_attention_kvarn.cuh ~line 675) is
  DEAD CODE — no launcher references it since D-18. Structurally it is the
  desired direct-read design: grid `(kv_heads, q_blocks)`, whole GQA group
  handled per CTA, packed pages dequantized once per (CTA, page) into smem.
- Why it loses: register budget. Each thread owns one dim `d` and carries
  `acc[16][64] + m_i[16][64] + l_i[16][64] + v_cache[16][64]` fp32
  (~4×1024 words) ⇒ massive local-memory spill. This is structural, not
  tunable.
- Materialize+flash baseline (bench_kvarn_2pass, 60k history × 2048 q):
  pass1 0.79 ms + pass2 36.45 ms. Pass-2 traffic: temp written once (32 MB)
  then read once per q_head CTA (16× amp ⇒ ~0.5 GB/chunk).
- Direct-read ideal traffic per (q_block, kv_head): 12 KB codes vs 64 KB
  bf16 tile ⇒ ~10× less DRAM *if* dequant results are reused across the
  GQA group AND enough q rows amortize the FWHT work.

## 2. Design (target)

Two-level tiling, FA2-style online softmax:

- Grid: `(kv_heads, q_super_blocks)`; q_super_block = 512–2048 rows.
- Each CTA streams its history segment ONCE; per page:
  1. dequant K,V page → smem bf16 tiles (2×64×256×2B = 64 KB, fits 99 KB)
  2. for each q sub-tile (Br=64) × head-subset (warp-per-2-heads):
     MMA QK^T against smem K tile, online softmax merge, PV MMA accumulate.
- Accumulators live in MMA fragments (tensor cores), not per-thread arrays —
  this removes the spill that killed the old kernel.
- Tail handling identical to existing kernels (bf16 tail mixed last).
- Numerics gate: rel_l2 ≤ 5e-3 vs staged reference; determinism required;
  bit-exactness NOT required vs staged (documented deviation like step 1d).

## 3. Milestones

- M1: extend bench_kvarn_2pass with a direct-read mode; measure current
  qblock kernel as-is (expect: slow, spill-bound) → fact base.
- M2: warp-per-heads restructure WITHOUT MMA (each warp owns 2 heads,
  128 acc regs/thread) — cheapest possible fix; measure.
- M3: full MMA fragment kernel (mma.cuh pattern) if M2 misses the win bar.
- M4: route over-wall prefill when faster; then evaluate shadow removal
  (in-capacity small-T already has split-K/tiled paths).
- M5 (after routing): docs/69 B2 width templating over surviving kernels,
  then E/F sweeps.

## 4. Win bars

- Over-wall pp: beat 37.2 ms @ 2048q/60k by ≥1.3× to route.
- Decode guard within 2%; T19 ≥450 floor unchanged.

## 5. RESULT (2026-08-25, later): CLOSED — provably losing on this hardware

M1 measurement: unmodified qblock kernel = 2243.9 ms (60x slower than staged
37.2 ms @2048q/60k), confirming spill analysis.

Design-space proof (register file 256 KB/SM, smem 99 KB/CTA):
Online-softmax state for a direct-read CTA = heads_eff x rows x D floats,
invariant under fragment layout (TC redistributes, does not compress):
- 16h x 64r: 262K floats -> 1024 regs/thread -> spill (measured)
- 16h x 32r, 1024 thr: acc fits (128 r/t) but q-cache smem = 256 KB > 99 KB
- 8h x 16r: fits registers, but grid 512 CTAs x 940 pages x 12 KB codes =
  5.5 GB DRAM/chunk vs staged ~0.3 GB -> loses 18x on traffic
Every configuration either spills (>255 regs) or loses on traffic.
Tensor cores redistribute accumulator fragments; they do not reduce the
physical state, so MMA does not escape the bound.

Conclusion: materialize-once + read-16x is information-theoretically cheaper
than dequant-N-times at these shapes (60k history vs 2048-row chunks, consumer
smem/regfile). The staged shadow buys ~10x effective bandwidth for 1 GiB/rank
— correct trade. docs/68 step-0 prioritization stands PROVEN, not just
measured. Direct-read remains viable only for small-T decode (already shipped:
split-K + tiled paths) where rows are few and state fits trivially.

docs/71 CLOSED. Recommended follow-up: none for pp; revisit only if smem/arch
grows (e.g., >=228 KB smem parts) or history/chunk ratio changes.

## 6. M2 RESULT (2026-08-25, final): direct-read TC kernel — BIT-EXACT, 6.8x slower

Implemented `gqa_attention_kvarn_direct_prefill_kernel`
(gqa_attention_kvarn_direct.cuh): the proven FA2 TC kernel with cp.async K/V
staging replaced by in-kernel dequant from packed codes (warp FWHT ->
swizzled smem), synchronous with per-tile syncthreads.

Measured (2048 q rows x 60k history):
- staged materialize+flash: 37.2 ms
- direct-read TC kernel:    253.3 ms (6.8x slower)
- rel_l2 vs staged reference: 0.000000 (bit-exact)

The gap is dequant-redundancy compute: 384 CTAs each dequant all 909 visible
pages (349k page-dequants) vs materialize's 1.9k. Removing it requires
multi-head accumulator state that exceeds the register file (section 5).

DISPOSITION: kernel kept as a validated, gated capability (bench regression
gate rel_l2 < 5e-3); NOT routed. Staged path remains the pp architecture.
Three bench-harness defects found & fixed during validation (raw-byte bf16
NaN inputs; unfilled temp; arena-scope aliasing of the scale table) — these
made every prior "timing-only" measurement of this harness numerically
meaningless. The harness now gates on rel_l2 and will fail loudly.
