# GEMM PIPELINE — the xw-pipelining / coeff-de-scratch design for tiled V0 (no-GPU CODE desk, 2026-09-18, amd/tp4-cure)

**Mission in one line:** LEANMAC's step-0 kill (`results/amd/coherence/LEANMAC_NOTES.md`,
commit 386a94df1) proved V0's wall is NOT arithmetic (1.57 ops/MAC emitted, dequant mul already
hoisted) — it is **129 `s_waitcnt lgkmcnt(0)` per body** (every xw LDS load fully
latency-exposed, serially) **plus coeff spilled to 80 B/thread scratch**. This desk prices the
pipelining/spill fix set from the ISA evidence + V0 source, reconciles the TILED-SWEEP V4
falsification, and pre-registers the bench + zero-GPU acceptance. **No src/ edit, no gate, no
GPU time; this doc is the only artifact.**

## 1. The wall, from the dump (all counts re-verified this desk on the LEANMAC §1 law-flags `-S` output)

Per **body** = one rolled iteration of the emitted compute loop (LEANMAC §2's census unit;
at TN=128 the whole kernel's counts equal the body's — prologue/epilogue contribute zero
`lgkmcnt`): the body covers 16 (g,v) blocks × 8 tokens = 128 xw loads, 1024 `v_fma_f32`,
144 `v_mul_f32`, 208 `ds_read` (128 xw b32 + 64 pair-table b64 + 16 scale b32), 24 scratch
buffer ops (16 store + 8 load), **129 `s_waitcnt lgkmcnt(0)`**.

The emitted per-token shape (v0_tn128 body, verbatim order):

```
ds_read_b32 vN, <xw addr>      ; token t's activation pair
s_waitcnt lgkmcnt(0)           ; FULL LDS round trip (~25-35 cyc) exposed, serially
v_lshlrev/v_and  (unpack xl, xh)
v_fma_f32 ×8                   ; 4 rows × lo/hi
ds_read_b32 vM, <next addr>    ; token t+1 issued only AFTER t's FMA nest
s_waitcnt lgkmcnt(0)           ; ...and waited immediately
```

Token t+1's load is issued after token t's nest and waited on **at once** — the 8-FMA nest
never covers anything. 128 serialized round trips per body vs ~6.9k cycles of pipe work
(1544 VALU ops + 208 ds ops × 4 cyc/wave-instr): the load-use stall, not the op count, is the
ceiling. V0 measures **1989-2444 GF/s on the W4 shards = 18.5-22.7%nom** (`V0PK_row.txt`).

**Why the second wave doesn't already save V0 — the measured decomposition** (TILED_SWEEP
window, full geometries): V0 (VGPR 128 → 2 waves/SIMD) 3083 GF/s; V4 (1 wave/SIMD, §4d)
2064 GF/s. Two readings fit: perfect 2-wave interleave (V0's 2nd wave = ×2.0; V4's body cycles
×0.75) or partial interleave (×1.6; V4 cycles ×0.93 — the leaner reading, consistent with V4's
ISA deltas being only barrier rate + staging overlap). **Both readings agree on what matters:**
the co-resident wave buys 1.6-2.0×, i.e. waves DO interleave, and V4 selling the second wave is
what killed it — while the lgkmcnt wall sat untouched inside each stream. Occupancy was already
bought; the remaining headroom is the in-stream serialization.

## 2. Fix set, each with its VGPR ledger and predicted TF/s

### (a) xw-load software pipelining — batch per (g,v) block, one wait per block

Restructure the source so each (g,v) block issues **all** its LDS reads back-to-back
(4 pair-table b64 + 8 xw b32 = 12 loads, one round trip in flight as a batch), computes the
hoisted pw products, then runs the token nest. One `lgkmcnt` per block; the nest
(64 `v_fma` + 16 unpack ≈ 320+ cycles) dwarfs the ~25-35-cycle round trip, so steady-state
exposed stall → ~0 (16-18 prologue waits per body remain, ≈450 cycles).

- **VGPR cost:** the xw batch = kS registers (8 at TN=128) vs V0's 1-deep xw → **+7-8 net**;
  pair-table pr 8 and pw 8 already live in V0's emitted code. Source-level ledger at TN=128:
  acc 32 + xw 8 + pr/pw shared 8 (pw overwrites pr in place, §2b) + c8 8 + sq 4 + coeff_r 4 +
  xl/xh 2 + addr/idx ~12-20 = **78-86 source regs + allocator slack → predicted
  next_free_vgpr 88-106**.
- **Why the compiler never did this itself:** V0's source interleaves load→use at token
  granularity, and LLVM's scheduler, given a 16-reg coeff array live across the whole k-tile
  and free rein to reorder, hit its 128-reg wall and spilled instead of batching. Explicit
  batching caps the scheduler's freedom and pays for itself (§2b frees the pressure that made
  batching unaffordable).
- **Predicted alone:** ×1.30-1.45 IF occupancy held — but (a) alone pushes V0's 128 to
  ~136-144 → re-spill or 1 wave/SIMD. **(a) alone is not shippable; it is the attribution arm
  inside the combined kernel, never the promotion candidate.**

### (b) coeff de-scratch — per-group live range, 4 regs, not 16

V0 computes `coeff[4][4]` (16 floats) up front per k-tile; LLVM spilled it (80 B/thread:
16 `buffer_store` per body + 8 `buffer_load` per block, vmcnt waits in-loop). But `coeff[r][g]`
is used **only inside group g's 8 blocks**. Restructure: hold the raw scale **quads**
(`sq[4]`, 4 regs, one global dword per row per k-tile, loaded once) and derive the 4 coeff
values **per group** from `s_scale_tab` (same 16 LDS lookups + 16 muls per k-tile as V0 — zero
added ops). coeff's live range: 16 regs whole-k-tile → **4 regs per group, dead at group end**.

- **What gets evicted: nothing.** The spill was pressure, not arithmetic: 16 held + pw 8 +
  spill-reload addressing is what blew the 128. With (a)'s explicit structure capping scheduler
  freedom and pw overwriting pr (−8), the ledger lands at 88-106 — under the bar with slack.
  (The mission's per-half-k-tile split alternative = our split generalized; per-group is finer
  and costs no extra global loads since sq is held.)
- **Ops deleted:** 24 scratch buffer ops per body, their vmcnt waits, and the scratch
  address registers; `.amdhsa_private_segment_fixed_size` 80 → **0**.
- **Predicted alone:** ×1.05-1.15 (2.1-2.8 TF/s) — LEANMAC already said spill is not where the
  headroom is. Its job is freeing the registers (a) needs and deleting the VMEM-serialized
  reloads; alone it is minor.

### (c) Occupancy: ≥2 waves is the CONSTRAINT, not the lever

V0 already runs 2 waves/SIMD (VGPR 128, LDS 19.45 KB × 2 CTAs = 39 KB < 64 KB; VGPR binds).
The fix must **keep** that: hard bar next_free_vgpr **≤ 128** (= 2 waves), target **96**
(headroom; 3 waves would need ≤85 — unreachable without cutting acc 32 + pipeline ≥22, and not
wanted: pipes idle from in-stream stalls, not wave starvation). Enforcement:
`__launch_bounds__(kThreads, 2)` (minBlocksPerMultiprocessor=2) — if the allocator wants more
than 128 it must spill, and the scratch-0 acceptance bar (§5 step-0) catches that at step-0,
not in the window. Predicted delta alone: ×1.0.

### (d) V4 reconciliation — it double-buffered the WRONG side, and paid both occupancy axes

Hard numbers from this desk's re-read of the sweep dump (V4 = `nvfp4_tiled_variant_kernel
<Cfg<64,64,4,16,true>, 80, 128>`, same law flags):

| | V0 shipped | V4 double-buf | PK tiled |
|---|---|---|---|
| `s_waitcnt lgkmcnt(0)` / body | **129** | **131** | 478-570 |
| `next_free_vgpr` | 128 (2 waves/SIMD) | **133 (1 wave)** | 256 (1 wave) |
| `group_segment_fixed_size` | 19456 B (3 CTAs LDS-wise) | **52224 B (1 CTA)** | — |
| `private_segment_fixed_size` | 80 B (24 scratch ops/body) | 0 | 804-820 B |
| in-loop barriers per k-tile | 2 | 1 | — |
| measured (full geoms) | 3083/2756/3103/2366 GF/s | 2064/1908/2008/1491 | 0.21-0.36× |

1. **The wall was never addressed:** V4's `lgkmcnt(0)` count is 131 vs V0's 129 — its
   double-buffer pipelines the **global→LDS staging** side, whose latency vmcnt already hides
   behind 4-deep `dwordx4` loads, while the **LDS→VGPR consumption** side (the 129-wait xw wall)
   is byte-equivalent. The barrier-rate and staging-overlap wins were worth at most ~×1.07-1.34
   of per-body cycles (the two §1 readings).
2. **It paid for that with the second wave twice over:** s_x doubled → 51 KB LDS → 1 CTA/CU
   even before VGPR 133 (>128 → 1 wave/SIMD) independently forced it. Net measured: ×0.63-0.69.
3. **Lesson encoded:** pipeline the **consumption** side (LDS→register) at register cost
   (+≤10 VGPR), never the staging side at LDS cost (+16 KB); and never spend VGPR past 128 —
   the second wave is worth ×1.6-2.0 measured, more than any per-body micro-win on this
   kernel.

## 3. Combined prediction (honest band)

Ratio method on the confirmed body structure (unmodeled-loss factors cancel):

| state | wall per body (cycles/wave, model) | ratio vs V0 |
|---|---|---|
| V0 | VALU ~6.2k + LDS 0.8k + exposed xw stall ~3.2-4.5k + spill/residue ~1k ≈ 11-12k | 1.0 |
| PIPE (a+b+c) | VALU ~6.2k + LDS 0.8k + 16-18 prologue waits ~0.5k + residue ~0.7k ≈ 7.4-7.8k | **1.45-1.60** |

Applied to V0's measured W4 shard band 1989-2444 GF/s (uncertainty widened both sides for the
residue the ratio method can't see):

- **Predicted combined band: 2.9-3.9 TF/s across the five W4 shards at M=128, central ≈ 3.3
  (×1.5 central over V0; per-shard ×1.45-1.6).**
- Cross-check: LEANMAC's independent ops-cut model predicted 3.15-3.87 (central 3.5) for a
  deleted-ops delta of zero — same corridor from a different axis; two models, one range.
- Upside tail, named NOT claimed: if the second wave interleave stacks on the cleaned stream
  as it did on V0's (×1.6-2.0), the pipe-bound ceiling (~6.4 TF/s model) comes into play. The
  floor below is the decisive pre-registration, not this tail.
- Serving projection (bench-class arithmetic, `V0PK_row`/cc455b648): gemm slice ≈ 798.8
  ms/chunk at V0 → **420-590 ms/chunk at the band, ~530 central**.

## 4. Step-0 zero-GPU ISA acceptance — BEFORE any window

LEANMAC §1 command verbatim (law flags off
`build-hip-amd/src/CMakeFiles/ninfer_hip_host.dir/flags.make`, `-x hip`, gfx900, committed
source) on the PIPE kernel; count over the kernel's `.text` + `.amdhsa_kernel` metadata for
**TN=128 at STILES ∈ {80, 24, 68}** (the five W4 shards; census already showed the op mix is
STILES-invariant). The three signatures:

1. **`s_waitcnt lgkmcnt(0)` ≤ 20 per rolled body** (V0: 129). Predicted 16-18 (one per
   (g,v) block + coeff merge into block 0's wait). Whole-kernel count = body count at TN=128.
2. **Scratch zero:** `buffer_store`/`buffer_load` count in the kernel = 0 AND
   `.amdhsa_private_segment_fixed_size` = 0 (V0: 24 ops, 80 B).
3. **`next_free_vgpr` stated** per instantiation in the step-0 table. Bar: **≤128 hard**
   (2 waves preserved — the V4 lesson), **96 target**. Predicted 88-106.

Any signature failing at step-0 → **no window**; iterate the sketch (first fallbacks: drop the
sq hold −4; `launch_bounds(256,2)` forcing a spill is caught by signature 2 — resolve by
structure, never by relaxing the bar). Bank the step-0 table in the run notes before the
window request.

## 5. Bench spec — `nvfp4_prefill_bench --pipe` (extend the results/ bench source, same discipline as --pk)

- **Arms:** V0 (`run_tiled`, unchanged anchor) vs PIPE (`run_tiled_pipe` → new
  `launch_nvfp4_tiled_gemm_pipe`). Five W4 shard geometries (run_pk's probs table):
  AttnInputW4 3584×5120, GdnInputW4 4096×5120, MlpGateUpW4 8704×5120, Resid6144W4 5120×1536,
  Resid17408W4 5120×4352. **M=128** (the production chunk; TN=128 instantiation). 3 warmup +
  20 iters, hipEvent timing, ceilings + 5 s mclk-ramp hammer + sysfs `pp_dpm_mclk` print —
  the identical cells as `--pk` so GF/s are cross-mode comparable. Window ≈ 2-3 min.
- **Correctness gate — FNV byte-identity, not rel-L2:** the restructure claims NO arithmetic
  change (k-ascending fp32 accumulation order untouched; only load placement and coeff's
  recompute point move; pw = pr·coeff fold order identical) → outputs must be **bit-identical**
  to V0. Per geometry: FNV-1a (nvfp4_gemv_bitcheck.cu form) over the full out buffer for both
  arms; **PASS requires hash equality**; rel-L2 printed as diagnostic (must read 0.00e+00).
- **Columns:** ms, GF/s, %nom (10.75e12), xV0, FNV hash pair + PASS/FAIL.
- **Pre-registered floor / READ lines:**
  - **xV0 ≥ 1.3 on EVERY shard** → latency-serialization thesis CONFIRMED; serving leg opens
    (NINFER_TILED_PIPE value-parsed gate, default OFF byte-identical, NINFER_TILED_PK
    pattern; bank-before-relink per BOOT_LAUNCH_RUNBOOK §4; boot battery + ladder).
  - Some shards < 1.3 → partial: report per-shard, ISA post-mortem on the failing shards
    (waitcnt actually emitted? occupancy actually 2 waves?), NO promotion.
  - **ALL < 1.3 → DEAD:** the latency thesis dies, the pipelining/spill family closes on this
    kernel, no blind knobs, no second window.

## 6. Kernel restructure sketch (diff-in-doc — SKETCH, NOT APPLIED; V0 byte-untouched)

```diff
--- a/src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu  (at 386a94df1)
+++ b/src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu
@@ (new kernel after nvfp4_tiled_gemm_hip_kernel; tables/bases/acc-init/epilogue V0-verbatim)
+// TILED-PIPE (2026-09-18, docs/amd/GEMM_PIPELINE_2026-09-18.md): V0's exact tile, staging,
+// arithmetic and k-ascending order — only LOAD PLACEMENT moves. Per (g,v) block the pr+xw
+// LDS reads issue as ONE batch waited ONCE (the 64-FMA+16-unpack nest covers the round
+// trip); coeff keeps a PER-GROUP 4-reg live range derived from the held scale quads
+// (de-scratch: the 80 B segment dies). Outputs BIT-IDENTICAL to V0 by construction.
+template <int STILES, int TN>
+__global__ void __launch_bounds__(kThreads, 2)   // fix (c): minBlocks=2 → VGPR cap 128
+nvfp4_tiled_gemm_hip_pipe_kernel(
+    const std::uint16_t* __restrict__ x, const std::uint8_t* __restrict__ codes,
+    const std::uint8_t* __restrict__ scales, const float inverse_weight_divisor,
+    std::uint16_t* __restrict__ out, const int n_rows, const int k_dim) {
+    constexpr int kS = TN / kColThreads;
+    // ... s_pair_tab / s_scale_tab / s_x + staging loop: V0 verbatim ...
+    const int ktiles = k_dim / kBK;
+    for (int kt = 0; kt < ktiles; ++kt) {
+        __syncthreads();
+        for (int i = tid; i < TN * 8; i += kThreads) { /* V0 staging verbatim */ }
+        __syncthreads();
+        std::uint32_t sq[kRowsPerThread];            // (b): raw quads held, 4 regs
+#pragma unroll
+        for (int r = 0; r < kRowsPerThread; ++r) {
+            sq[r] = *reinterpret_cast<const std::uint32_t*>(
+                scales + scale_base[r] + static_cast<std::int64_t>(kt) * 512);
+        }
+#pragma unroll
+        for (int g = 0; g < 4; ++g) {
+            float coeff_r[kRowsPerThread];           // (b): live range = THIS group (4 regs)
+#pragma unroll
+            for (int r = 0; r < kRowsPerThread; ++r) {
+                coeff_r[r] = amd::bits_to_f32(s_scale_tab[(sq[r] >> (8 * g)) & 0xFFu]) *
+                             inverse_weight_divisor;
+            }
+            uint2 c8[kRowsPerThread];                // u64 code load per row (V0 verbatim)
+#pragma unroll
+            for (int r = 0; r < kRowsPerThread; ++r) {
+                c8[r] = *reinterpret_cast<const uint2*>(
+                    row_codes[r] + static_cast<std::int64_t>(kt * 4 + g) * 8);
+            }
+#pragma unroll
+            for (int v = 0; v < 8; ++v) {            // (g,v) block: batch ALL LDS reads
+                uint2 pr[kRowsPerThread];
+                std::uint32_t xw[kS];                // (a): the pipeline buffer, 8 regs @128
+#pragma unroll
+                for (int r = 0; r < kRowsPerThread; ++r) {
+                    const std::uint32_t word = (v < 4) ? c8[r].x : c8[r].y;
+                    pr[r] = s_pair_tab[(word >> ((v & 3) * 8)) & 0xFFu];
+                }
+#pragma unroll
+                for (int t = 0; t < kS; ++t) {       // 12 ds_read in flight, ONE wait
+                    xw[t] = s_x[tok0 + t][(g * 8 + v) ^
+                                          (static_cast<unsigned>(tok0 + t) & 31u)];
+                }
+#pragma unroll
+                for (int r = 0; r < kRowsPerThread; ++r) {   // pw overwrites pr (−8 VGPR);
+                    pr[r].x = amd::bits_to_f32(pr[r].x) * coeff_r[r];   // same single mul,
+                    pr[r].y = amd::bits_to_f32(pr[r].y) * coeff_r[r];   // same RNE value
+                }
+#pragma unroll
+                for (int t = 0; t < kS; ++t) {       // FMA nest — V0's order, bit-identical
+                    const float xl = amd::bits_to_f32(xw[t] << 16);
+                    const float xh = amd::bits_to_f32(xw[t] & 0xFFFF0000u);
+#pragma unroll
+                    for (int r = 0; r < kRowsPerThread; ++r) {
+                        acc[r][t] = fmaf(pr[r].x, xl, acc[r][t]);
+                        acc[r][t] = fmaf(pr[r].y, xh, acc[r][t]);
+                    }
+                }
+            }
+        }
+    }
+    // ... bit-cast epilogue: V0 verbatim ...
+}
```

Seams (one-liners, same grammar as the sweep/PK precedents): `launch_nvfp4_tiled_gemm_pipe`
declared in `nvfp4_tiled_gemm_hip.h`, defined as `launch_tiled_exact`'s clone routing to the
pipe kernel (m ∈ {32,128,256} switch untouched; TN=256 fits the law: acc 64 + xw 16 + ~20
≈ ≤128 — bar-checked at step-0 too); `run_tiled_pipe` + `run_pipe()` + `"--pipe"` argv in
`results/amd/coherence/nvfp4_prefill_bench.cu`. Bit-identity argument: per (r,t) the emitted
ops are one `v_mul` (pr·coeff, same fold) then one `v_fma` in the same k-ascending
g→v→t→r order as V0 — FNV must confirm at step-1.

## 7. What was NOT done (desk law)

No edit to `src/` (V0 and every variant byte-untouched; FIX-A's `gated_delta_net/` never
read-written), no gate, no bench arm, no GPU time, no window requested. This doc is the sole
commit. Read-only everywhere else; `/tmp` extracts (`/tmp/v0_tn128_full.s`, `/tmp/v4_tn128.s`)
are scratch, re-derivable from `/tmp/leanmac_build/v0_current.s` per LEANMAC §1.

## 8. Provenance

`results/amd/coherence/LEANMAC_NOTES.md` (386a94df1 — the kill, census method, PK post-mortem)
· `src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu` (:53-219 V0 + launcher; :237-260 variant cfgs;
:250 V4; committed source) · this desk's re-census: V0 TN128 next_free_vgpr 128 / LDS 19456 /
private 80 / 129 lgkmcnt(0) / 24 buffer ops; **V4 TN128 next_free_vgpr 133 / LDS 52224 /
private 0 / 131 lgkmcnt(0)** (from `/tmp/leanmac_build/v0_current.s`, law flags) ·
`results/amd/coherence/TILED_SWEEP_notes.md` §5 (V0/V4 measured bands) ·
`results/amd/coherence/V0PK_row.txt` (V0 W4 shard band 1989-2444 GF/s) · cc455b648 (LEANMAC
spec: 3.15-3.87 independent model, 798.8 ms gemm slice) ·
`results/amd/coherence/nvfp4_prefill_bench.cu` (run_pk discipline + probs table) ·
`results/amd/coherence/nvfp4_gemv_bitcheck.cu` (FNV-1a precedent).
