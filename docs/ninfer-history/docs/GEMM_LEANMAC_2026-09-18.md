# GEMM LEANMAC — the leaner-MAC fp32 design after the PK falsification (no-GPU design desk, 2026-09-18, amd/tp4-cure)

**Seat:** NO-GPU design desk. **Method:** instruction-by-instruction code-read of the shipped V0
inner loop (`nvfp4_tiled_gemm_hip.cu`) + banked rows (V0PK_row, SWEEP_row, PK_row, HFMA2_notes
static-ISA precedent) + arithmetic. Zero GPU work, zero builds, zero `src/` edits. **Provenance
chain:** PLOG-045 — V0-PK HFMA2 arm DEAD 0.21-0.36x V0 on the five W4 shards (V0 1989-2444 GF/s
= 18.5-22.7% nominal); PK's failure moved the path forward OUT of fp16 arithmetic and INTO
ISSUING FEWER OPS PER MAC IN FP32 (the GEMM-scale desk's issue-wall census, commit 9a38c0c50 §1-D).
**This doc designs; it measures nothing.**

**Chosen design in two lines:** LEANMAC = pure register-level CSE of V0's dequant·coeff product —
compute `pw = e2m1_pair × coeff[row,group]` once per (row, value-pair) (8 v_mul) instead of once
per (row, token, element) (64 v_mul), so the inner token loop issues pure `v_fma_f32`. Bit-exact
by construction (same multiply, same rounding site, same accumulation order), zero memory-format
change, zero VRAM, +≤8 VGPR. Census 152 → 96 issued ALU ops per 64 MACs = **2.375 → 1.50
ops/MAC**; predicted **3.15-3.87 TF/s bench (central 3.5)** = V0's measured W4 band × 152/96.

---

## 1. THE V0 INNER-LOOP CENSUS, OP BY OP

Per (g,v) iteration of the shipped kernel (g = group 0..3, v = value-pair 0..7; TN=128 ⇒
kS=8 tokens/thread; thread = 4 rows × 8 tokens): **64 MACs** = 4 rows × 8 tokens × 2 pair
elements. Source: `nvfp4_tiled_gemm_hip.cu` `nvfp4_tiled_gemm_hip_kernel`, the `g/h/v/t/r`
nest (this worktree's working tree, :133-162).

| # | op class | count per (g,v) | what it computes | can it shrink? | fate |
|---|----------|-----------------|------------------|----------------|------|
| 1 | `v_mul_f32` | **64** | `bits_to_f32(pr[r].{x,y}) * coeff[r][g]` — applies the per-(row,group) coeff to the LUT-decoded e2m1 pair; re-issued per (row, token, element) though the product is INVARIANT over the kS token loop | 8 of 64 are real work (one per row per element); 56 are re-issues of a loop-invariant product | **HOIST to 8** — the entire design (§3) |
| 2 | `v_fma_f32` | **64** | `acc[r][t] = fmaf(w, x, acc)` — the MAC itself, 1 MAC/instruction | gfx900 fp32 has no 2-MAC form (no dot2/pk in fp32; packed fp16 = PK = falsified PLOG-045) | KEEP — the floor |
| 3 | int VALU: code extract | 8 | per row: `(word >> (v&3)*8) & 0xFF` (shift+mask, constant shift) → pair-table index | `v_bfe` field-extract = 1 op (−4); LLVM may already emit it | KEEP (−≤4, ISA-dump decides) |
| 4 | int VALU: x unpack | 16 | per token: `xw << 16` / `xw & 0xFFFF0000` → two bf16-as-f32 words; 8 FMAs per unpack (0.25 op/MAC, amortized over 4 rows) | eliminating needs fp32-staged x ⇒ s_x doubles to 32 KB ⇒ 35 KB LDS/block = the measured V4 occupancy killer (SWEEP falsifier (a)) | KEEP |
| 5 | LDS (co-issue port, NOT the ALU bill) | 12 | 4× `ds_read_b64` pair-table (one per row, each amortized over 8 tokens = 16 MACs/load) + 8× `ds_read_b32` s_x (8 FMAs/load) | arithmetic e2m1 decode prices 10-14 ALU ops per byte vs 1 co-issue-port LDS ⇒ **+40-56 ALU ops per (g,v)** — a LOSS at the issue level | KEEP the LUT |
| 6 | coeff fold | (16/k-tile) | `coeff[r][g] = scale·divisor` per (row, group) — already hoisted to group level, negligible | — | KEEP |
| 7 | bookkeeping | ~0 | g/v/t/r loops fully unrolled; swizzle indices are compile-time + per-thread constants; only the kt loop rolls | — | — |
| | **total issued ALU-class** | **152** | | | **96 after LEANMAC** |

**Ops/MAC ledger:** V0 = 152/64 = **2.375**; LEANMAC = (8+64+24)/64 = **1.50** (92/64 = 1.44 if
v_bfe lands). Wave math (wave32 on 16-lane SIMD = 2 cyc/instr): V0 304 cyc per (g,v) per wave →
4096 wave-FLOP → 13.5 FLOP/cyc/wave → 54/CU → 42% of 128 ⇒ 4.54 TF/s model (@1.5 GHz × 56 CU).
LEANMAC: 192 cyc → 21.3 → 85.3/CU → 67% ⇒ **7.16 TF/s model**. Measured V0/model on the same W4
shards = 1.989-2.444/4.54 = 0.44-0.54 silicon efficiency — the band carried below.

## 2. CANDIDATES, ANALYZED HONESTLY

### (a) Pre-multiplied weight words (the coeff hoist, two forms)

**(a-full) load-time layout transform: dequant to a dense fp32 (or bf16) weight plane at load,
inner loop = pure FMA on expanded words.** Census at fp32: 64 v_fma + 16 unpack + 12 LDS = 80
ops = **1.25 ops/MAC** — BETTER than LEANMAC, and bit-exact (code×coeff rounds once in fp32 RN
at load = the identical product the kernel computes today). **Priced and rejected on VRAM, not
bandwidth:** NVFP4 weight = 0.5625 B/value; fp32 expansion = 7.11×; the W4 shard working set is
3.43 GB/rank (PREFILL_DECOMP §, the streamed-per-chunk = resident weight plane) ⇒ **24.4 GB/rank
vs the ~8 GB/die class budget** (the "16.4 GB planes do not fit a 2-die budget" datum) — dead on
arrival at TP4. bf16 expansion (2 B, 3.56×, 12.2 GB/rank) also busts the die AND loses
bit-exactness (bf16 rounding of code×coeff ⇒ rel-L2 gate) while keeping the 16-op x-unpack.
Bandwidth leg of the price, for completeness: 24.4 GB / 368.7 GB/s measured read ceiling =
66 ms/chunk DRAM floor (vs 9.3 at NVFP4) — still 12× under the 798.8 ms slice, so DRAM would NOT
have killed it; the 8 GB/die wall does. *If a future world ever carries VRAM headroom, (a-full)
is the known 1.25-ops/MAC ceiling form.*

**(a-lite) register-level pre-multiply — THE DESIGN.** Same product, computed in-kernel, hoisted
out of the token loop: 8 v_mul per (g,v) instead of 64. Zero memory-format change (codes/scales
planes byte-identical; nothing moves on the 378 GB/s wire), zero VRAM, +≤8 VGPR. Bit-exact BY
CONSTRUCTION — see §3. This is the move PK made via fp16 `__hmul2`; the MOVE was right, the
PRECISION was wrong (fp16 cvt/state tax; autopsy §4-R4).

### (b) e2m1 arithmetic dequant vs the LDS LUT — FALSIFIED by census

The `s_pair_tab` byte→(f32,f32) load IS the dequant, in 1 co-issue-port LDS per row per
value-pair. Replacing it with nibble arithmetic (sign shift-or, exponent add, mantissa merge,
zero-fix per nibble ≈ 5-7 int ops × 2 nibbles) costs **10-14 ALU ops per byte** ⇒ +40-56 ALU ops
per (g,v) against 4 LDS slots that do not bill the ALU pipe. A trade of the cheapest resource
(co-issue LDS) for the binding one (ALU issue) — rejected. The LUT already returns BOTH pair
values in one `ds_read_b64` and is amortized over 16 MACs/load; that is its optimal shape.

### (c) int-op elimination — real but NOT the lever

24 int ops per (g,v): 8 code-extract (v_bfe candidate, −≤4, compiler-dependent) + 16 x-unpack
(eliminating doubles s_x to fp32 ⇒ 35 KB LDS ⇒ the measured V4 occupancy death; KEEP). Loop
bookkeeping is already unrolled away. Ceiling of this class: −4 of 152 — do it if the ISA dump
shows LLVM hasn't, but it is worth ~4%, not ~40%. The lever is (a-lite)'s −56.

### (d) TN shape — REJECTED, the sweep closure is respected

No shape constant moves: kTM=64, kBK=64, kThreads=256, 4 rows × kS tokens/thread, TN as
dispatched. Wider per-thread columns (TN=256 ⇒ kS=16) would dilute the now-hoisted fixed costs
(~24 ops over 8·kS MACs) but land exactly on the measured-dead footprints: TN=256 is 35 KB
LDS/block (V4's 33-37% loss, SWEEP falsifier (a)), and shrinking kColThreads is a shape-family
change. The tile-shape family is CLOSED (TILED_SWEEP §5: V0 wins all four geometries by 15-41%);
this arm changes only the INNER LOOP at V0's fixed shape — same law PK's arm ran under.

## 3. THE LEANMAC RESTRUCTURE (bit-exactness argument)

```
// per (g,v), after the 4 u64 code loads and pr[r] = s_pair_tab[byte_r]:
float pw0[kRowsPerThread], pw1[kRowsPerThread];          // +8 VGPR worst case
for (r) { pw0[r] = amd::bits_to_f32(pr[r].x) * coeff[r][g];   // 4 v_mul
          pw1[r] = amd::bits_to_f32(pr[r].y) * coeff[r][g]; } // 4 v_mul
for (t) { xw = s_x[tok0+t][(g*8+v) ^ (tok & 31)];
          xl = bits_to_f32(xw << 16); xh = bits_to_f32(xw & 0xFFFF0000u);
          for (r) { acc[r][t] = fmaf(pw0[r], xl, acc[r][t]);    // pure v_fma
                    acc[r][t] = fmaf(pw1[r], xh, acc[r][t]); } }
```

**Order law:** the g/h/v nest walks K in V0's exact k-ascending sequence; the per-accumulator
add sequence is untouched. **Bit-exactness:** V0 computes `fmaf(round(a×c), x, acc)` where
`a = bits_to_f32(e2m1)`, `c = coeff` — the multiply is a standalone RN op feeding an explicit
fmaf (no fp-contraction seam: a mul whose result feeds fma's multiplicand is not a fusible
mul-add pair). LEANMAC computes the identical `round(a×c)` with the identical instruction at the
identical rounding site, reuses it over the 8 tokens (the reuse V0's source merely forgot to
state), and FMAs the identical bits in the identical order. **IEEE RN determinism ⇒ outputs are
BIT-IDENTICAL to V0 — the gate is byte/FNV identity, NOT rel-L2** (the exact inverse of PK's
gate, which was tolerance-only BECAUSE it re-associated precision). Unlike PK, this arm MAY run
under the standing TILEDV FNV cell — and must PASS it.

**Pre-registered honesty target (mission's "≥1.6 ops/MAC eliminated"):** the census says a
literal elimination of 1.6 ops/MAC (to 0.775 remaining) is **NOT achievable in fp32** — the floor
is 64 FMA (1 MAC/op, no fp32 packed form on gfx900) + 24 int + 8 hoisted mul = 96 = 1.50/MAC;
going below 1 op/MAC requires packed arithmetic = the falsified PK class. Max honest elimination
is **0.875 ops/MAC** (2.375 → 1.50). If the mission's bar reads "land AT OR BELOW 1.6 ops/MAC",
LEANMAC clears it at 1.50.

## 4. PREDICTION (two methods, one arithmetic — stated twice on purpose)

1. **Ops-scaling of the MEASURED V0 band (primary):** 1.989-2.444 TF/s measured
   × (152/96 = 1.583) = **3.15-3.87 TF/s bench, central 3.5** — valid IF the kernel stays
   issue-bound at the same silicon efficiency (same shape, same issue pattern, strictly fewer
   issued ops — the kindest comparison available).
2. **Model-from-census:** 7.16 TF/s model × the 0.44-0.54 model-to-silicon band V0 itself
   measured on these shards = 3.1-3.9 TF/s. Same numbers; no independent confirmation is claimed.

**Gemm slice:** 1556.9 GF/chunk/rank ÷ 3.15-3.87 TF/s = **402-494 ms (central ~445)** — the
mission's "~450". **Serving honesty:** the unexplained 0.66-0.67 serving/bench class factor
(GEMM-scale §header) if multiplicative inflates to **610-750 ms**; both readings are
pre-registered and the same boot decides ([PREFILL-SUM] gemm vs the bench GF/s on one line).
Either way the arm must beat **798.8 ms warm** to exist at all. This raises the fp32-SIMT class
ceiling (~240-330 tok/s position, PLOG-045); it does NOT unlock a new ceiling class — the packed
class is falsified territory at M=128.

**Risk register:**
- **R1 — the compiler may already hoist.** LLVM LICM on the unrolled token loop could already be
  issuing ~96 ops, in which case V0's 2.0-2.4 TF/s is 0.28-0.34 of a 7.2 model and LEANMAC buys
  nothing. Evidence AGAINST full hoisting: this toolchain's own S5 ISA dump kept **34 v_mul_f32**
  alive in small_t's group loop (HFMA2_notes §static ISA) — but V0's invariant is cleaner than
  S5's. **Decided by step-0 (§5), zero-GPU, BEFORE any window.**
- **R2 — VGPR:** V0 estimated ~80-100 (acc 32 + coeff 16 + c8 8 + pr 8 + addressing); +≤8 for
  pw0/pw1 (which overlap pr's live range — pr dies at the pw compute). 2 CTAs/CU retained below
  the 128-VGPR 2-CTA boundary. Step-0 dump reports exact VGPR; **any scratch/spill op in the
  loop body = RED before the window.**
- **R3 — sub-linearity:** as ALU slots shrink, the non-ALU share grows (12 LDS per (g,v) move
  from 8% to 16% of slots; barriers unchanged) ⇒ the real gain may land under 1.583×. This is
  exactly what the 1.4× floor absorbs and the 1.15× branch indicts.
- **R4 — the PK cautionary tale:** PK predicted 2.4-3.3× and measured 0.21-0.36×; V0PK_row's
  "convert/stage bill dominates" is an INFERENCE, not a dump — the static staging bill (~16
  cvt/thread/k-tile, <2% of MACs) cannot explain a 3-5× swing; suspected scalarized half2 or
  spills, never proven (small_t-PK's dump WAS clean: native v_pk_fma_f16, VGPR 56, 0 spills —
  the tiled instantiation was never dumped). Lesson ENCODED: LEANMAC introduces no new types, no
  intrinsics, no precision change — the lowering-surprise class that killed PK is absent BY
  CONSTRUCTION — and step-0 dumps BOTH kernels (V0 for R1, PK-kernel for the free post-mortem)
  before any GPU time.

## 5. DECISIVE CHECK S3 (pre-registered — bench arm + floor + ISA fallback)

**Step 0 — ISA census (zero-GPU, owed by the CODE desk before any window; HFMA2_notes method:
compile at the -O3 LAW line, `llvm-objdump -d` the kernel bodies, /tmp staging like
`/tmp/hfma2_build/*.s`):**
- V0 instantiation (any W4 STILES, TN=128 body): count per-(g,v) `v_mul_f32` / `v_fma_f32` /
  int-VALU / `ds_read_b64`/`ds_read_b32`; record VGPR + spill. **64 v_mul ⇒ LEANMAC lives;
  ~8 v_mul (already hoisted) ⇒ NO BUILD — the census model was wrong about the wall (V0 already
  issues ~96; its 0.28-0.34-of-model gap is LDS-port/barrier class) ⇒ post-mortem desk, the
  fp32-restructure family closes without a window.**
- Post-restructure body: must show 8 v_mul + 64 v_fma, 0 spills, VGPR ≤ V0+~10. Mismatch ⇒ fix
  at the desk, never on the GPU.

**Bench leg — `nvfp4_prefill_bench --leanmac` (mirrors `--pk` machinery exactly):**
- arms: V0 vs LEANMAC at M=128 × the FIVE W4 shard geometries (3584×5120, 4096×5120, 8704×5120,
  5120×1536, 5120×4352); 5 s mclk hammer + ceilings + sclk sideband per house discipline;
  3 warmup + 20 iters; columns GF/s, %nominal (10.75 anchor), xV0, and **BYTE-IDENTITY
  (FNV/byte-compare vs V0) — expected 0.00e+00; ANY nonzero = RED regardless of perf**
  (bit-exactness is this arm's contract; there is NO rel-L2 gate).
- **Pre-registered floor: LEANMAC ≥ 1.4× V0 on EVERY geometry** (the mission floor; expected band
  1.5-2.0×, central 1.58×). Branches:
  - ≥1.4× all five + bit-identical ⇒ serving leg.
  - 1.15-1.4× ⇒ read the step-0 dump: if the 96-op form is confirmed, the miss is sub-linearity
    (R3, LDS/barrier share) ⇒ post-mortem, **no blind knobs, no shape sweep**.
  - <1.15× all ⇒ census-model DEAD (the PK recurrence in fp32 clothing) ⇒ write the post-mortem,
    close the family, the 240-330 tok/s F32-SIMT position stands.

**Serving leg (same window, after the bench):** rebuild per BOOT_LAUNCH_RUNBOOK,
BANK-BEFORE-RELINK (§4), boot from the bank. Gate `NINFER_TILED_LEANMAC` — PK's N7 matrix
verbatim: unset = OFF byte-identical production; `=1` = lean arm; `=0`/`=off`/`=""`/anything else
= **REFUSES LOUD** at first route. `NINFER_TILED_TRACE=1` must print `arm=leanmac n=.. k=.. M=..`
(route proof). **TILEDV FNV cell stays ON for lean legs** (bit-identity preserved — the one gate
PK had to bypass and leanmac must pass). plen-1996 probe + standing ladder battery; decisive
number **[PREFILL-SUM] gemm < 798.8 ms/chunk**; pre-registered: ≤ 500 = confirmed (the bench-class
reading), > 650 with bench ≥ 3 TF/s = the 0.66-0.67 factor is multiplicative (the honest
half-win reading — bank it, do not spin it). Then the promote decision goes to the coordinator
with both rows linked.

## 6. Honesty row

- The census (§1) is source-static, same class as the GEMM-scale desk's §1-D; step-0's ISA dump
  is its falsifier and gates the build (R1). The two numbers this desk cannot settle: whether
  LLVM already hoists the 64 muls, and whether v_bfe lands.
- The 1.4× floor sits under the 1.58× central prediction by design: the gap absorbs R3
  sub-linearity. A 1.15-1.4× result still falsifies the efficiency-holds assumption even if the
  ISA is exactly as censused — the floor, not the hope, is the contract.
- (a-full)'s 1.25-ops/MAC form is priced, not measured; its VRAM rejection rests on the
  3.43 GB/rank weight plane and the ~8 GB/die class budget, both cited from banked docs, not
  re-derived here.
- The serving-side 402-494 vs 610-750 ms fork is a genuine two-reading prediction; the doc does
  not know which leg the 0.66-0.67 factor lives on and claims neither.
- Zero GPU work, zero builds, zero src/ edits at this seat; the working tree's live GPU-window
  artifacts and the three FIX desks' files were read, never written. Only this doc is committed.

---

Bases: nvfp4_tiled_gemm_hip.cu (V0 kernel :53-177, the g/h/v/t/r nest :133-162; PK kernel for the
autopsy) · V0PK_row.txt + V0PK_bench_run.log + PLOG-045 (PK DEAD 0.21-0.36x; V0 on the five W4
shards 1989-2444 GF/s = 18.5-22.7% nom) · PREFILL_GEMM_SCALE_2026-09-17.md commit 9a38c0c50 (§1-D
census 152 ops/64 MACs = 2.4, model 4.5 TF/s, issue-wall verdict; §5 gate grammar; §6 honesty
grammar) · TILED_SWEEP_notes.md + SWEEP_row.txt (tile family CLOSED; V4's 35 KB-LDS occupancy
loss 33-41%; falsifier-(b) issue-stream text) · HFMA2_notes.md (static-ISA method
`/tmp/hfma2_build/*.s`; S5 kept 34 v_mul — the R1 datum; small_t-PK clean dump VGPR 56, 0 spills;
cvt-tax 1.07x decode datum) · PK_row.txt (small_t rel-L2 1.14-1.26e-3 GREEN / perf RED precedent)
· nvfp4_amd_codec.h (:111-114 fold contract = the exact product hoisted; :131-138 order law) ·
math.cuh:59-84 (the D3 bridge PK staged — the inference-vs-dump gap §4-R4 names) ·
PREFILL_DECOMP_2026-09-17.md (3.43 GB/rank/chunk weight plane; 9.3 ms DRAM floor) ·
PREFILL_OPTRACE_row.txt via GEMM-scale header (798.8 ms gemm slice, 1556.9 GF/chunk/rank,
0.66-0.67 serving/bench factor) · code line numbers are THIS worktree's working tree and may drift.
