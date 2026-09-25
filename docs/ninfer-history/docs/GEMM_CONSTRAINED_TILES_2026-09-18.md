# GEMM CONSTRAINED TILES — the pipeline-constrained tile family: derived, priced, KILLED at zero GPU (no-GPU CODE desk, 2026-09-18, amd/tp4-cure)

**Verdict in one line:** the PIPE forward pointer's class ("shapes with ≤ 16 acc/thread can
fund a pinned batch under 128") is **CONFIRMED on funding and on wait coverage** — every
enumerated shape carries a 42-66-source-reg ledger (≥ 46 regs of slack under the 128 cliff)
and its FMA nest covers the 25-35-cycle LDS round trip 3-7x over — **but the family dies
anyway, on two walls the forward pointer did not price: per-flop decode-overhead inflation at
small acc (the census model's issue penalty alone caps the best shape at ×1.21-1.35 vs V0
before grid effects), and token-tile/grid arithmetic that forces either 512-thread CTAs or
doubled CTAs-per-row and pays 15-59% makespan on 4 of the 5 W4 shards.** The measured anchor
closes it: **the class optimum (TM=32, RPT=4, CT=32, kS=4, acc=16) IS sweep variant V1, and
V1 is already measured at 0.72-0.91x V0 unpipelined** — so its absolute ceiling with a
perfect pipeline is **×1.02-1.46, and on Attn/Gdn/Mlp the ceiling is 1.02-1.24, BELOW the
pre-registered ≥1.3x-every-geometry floor by construction.** No shape survives. **NO BUILD,
no window, no bench arm.** Predicted class band (best shape, realistic): **1.5-2.6 TF/s
central ~2.1** vs V0's measured 2.0-2.4 — a family that cannot clear its own floor. This
desk burns zero GPU: the kill is census arithmetic + an existing measured row, the LEANMAC
R1 standard.

## 1. Inputs (read in order; what each contributed)

1. `results/amd/coherence/LEANMAC_NOTES.md` — the real wall: V0's emitted body = **129
   `s_waitcnt lgkmcnt(0)`** (every xw LDS load latency-exposed serially), coeff spilled to
   80 B/thread scratch, **V0 at VGPR 128 = the exact cliff**; and the census method (law
   flags, op-mix table) every number below is calibrated against.
2. `results/amd/coherence/PIPE_NOTES.md` — the fix attempt that honestly died at V0's tile:
   batching the 12 per-block LDS reads needs +8 VGPR the 128 cliff cannot fund (10-cell
   probe matrix: unpinned batches re-sink 8/8; volatile-pinned batches emit — 129 → 67
   waits/body-unit — but spill 2.3-2.4 KB; `launch_bounds(256,2)` byte-identical = zero
   slack); and the **forward pointer this desk was ordered to price**: "shapes with ≤ 16
   acc/thread (e.g. 32-row blocks: acc 16, batch 4-8, byte-granular code loads) can fund a
   pinned batch under 128; the sweep never priced a pipeline."
3. `docs/amd/GEMM_PIPELINE_2026-09-18.md` — the mechanism (batch each (g,v) block's 12 LDS
   reads into ONE wait, 129 → ~16-18/body) and the pre-registration discipline (step-0 ISA
   acceptance before any window; floor ≥ 1.3x V0 EVERY shard else the family closes).
4. `results/amd/coherence/TILED_SWEEP_notes.md` + `SWEEP_row.txt` — the original V0-V4 sweep
   closed WITH V0 winning, **run without pipelining as a constraint and without the ISA
   evidence**; its closure does not bind a pipeline-constrained family — **but its V1 row is
   this family's shape optimum, already measured** (§5), which is what converts this desk's
   model into a measured kill.
5. `src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu` — the shipped V0 (64-row x TN tiles, 4
   rows/thread x TN/16 token-stripes, **acc[RPT][kS] = 4x8 = 32 fp32 acc registers/thread at
   TN=128** — which is exactly WHY no register headroom exists at that tile), the variant
   template (V1 = `TiledVariantCfg<32,64,4,32,false>` :247), and the
   launcher seams any new shape would have to reuse.

## 2. The class definition and its enumeration law (mission item 1)

Class constraints (from the PIPE forward pointer, made exact):

- **acc/thread ≤ 16** fp32 accumulator registers — the funding precondition;
- **batchable LDS reads** — per (g,v) block the thread's pair-table reads (RPT x b64) + xw
  reads (kS x b32) issue as one pinned flight, one wait;
- **total VGPR ≤ 112** (128 cliff − batch funding − margin; the mission bar — note the real
  acceptance is a *census*, not a ledger: source ledgers predict badly, v3 landed 64 VGPR +
  1.6 KB spill where its ledger said 88-106, PIPE §4.3);
- **occupancy ≥ 2 waves/SIMD** (the V4 lesson: the second wave is worth ×1.6-2.0 measured —
  never sell it).

**Enumeration law at the production geometry (TN=128 = the M=128 chunk, one token tile):**
kS = TN/CT (token-stripes per thread), acc = RPT·kS, threads = (TM/RPT)·CT. acc ≤ 16 with
kS ≥ 2 (kS = 1 has no batch to speak of) and pow2 splits forces **CT ≥ 32** for any acc-16
shape — i.e. **kS ≤ 4**. That single fact drives everything: kS ≤ 4 means CT = 32 or 64
token-stripes, which means either 512-thread CTAs (TM=64, RPT=4, CT=32) or doubled grids
(TM=32 → n/32 blocks). Both collide with the co-residency the class is trying to keep
(§4, wall 2). RPT > 4 inflates the pw-mul term (64·RPT issue slots/k-tile) with no
compensating term; acc = 8 shapes relieve funding that is not binding and pay 2.0 ops/MAC.
The class has exactly five concrete members; the plane is fully enumerated.

## 3. The candidates (mission item 1: ledger, waves, M-fit, model)

**Instruction model, per k-tile (K=64) per thread-stream** — census-calibrated: every term
below reproduces the LEANMAC/PIPE emitted census at V0's tile to ±1% (FMA 2·RPT·kS/block,
pw-mul 2·RPT/block, unpack 2·kS/block, misc ≈ 7.5/block from census 120/body, ds =
(RPT+kS)/block + 16 scale-lookups + 16 staging; VALU issue = 4 cyc/wave-instr, ds = 4 cyc —
the PIPE §1/§3 convention, which reproduces V0's measured %nom decomposition). V0 model row:
VALU 3088 (census 1544/body x2 ✓), ds 432 (census 208/body + staging ✓), issue 14080 cyc;
with the census exposure (258 waits x 25-35 cyc) + residue ≈ 800 → **21.3-23.9k cyc/k-tile**
= the PIPE §3 V0 model to the digit.

| shape (RPT, CT, kS) | TM / threads | acc | FMA | pw-mul | unpack | misc | VALU | ds | issue cyc | flops/wave-kt | **issue cyc/kflop S** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 (4,16,8) — anchor | 64 / 256 | 32 | 2048 | 288† | 512 | 240 | 3088 | 432 | 14080 | 262144 | **53.7** |
| **S-A (4,32,4)** | 64 / **512** | 16 | 1024 | 256 | 256 | 240 | 1776 | 288 | 8256 | 131072 | **63.5** |
| **S-B (4,32,4)** | 32 / 256 | 16 | 1024 | 256 | 256 | 240 | 1776 | 288 | 8256 | 131072 | **63.5** |
| S-C (2,16,8) | 32 / 256 | 16 | 1024 | 128 | 512 | 240 | 1904 | 368 | 9088 | 131072 | 69.3 |
| S-D (4,16,2) | 64 / 256 | 8 | 512 | 256 | 128 | 240 | 1136 | 240 | 5504 | 65536 | 84.0 |
| S-E (2,32,4) | 16 / 256 | 8 | 512 | 128 | 256 | 240 | 1136 | 240 | 5504 | 65536 | 84.0 |

† census 288 = 256 pw + 32 coeff muls. S = issue cyc/kflop is the shape's issue-only floor:
**no pipelined kernel of that shape can beat 1000/S x 32 flop/cyc/SIMD = its %nom ceiling**
(32 flop/cyc/SIMD = 10.75 TF/s nominal / 224 SIMD).

**VGPR source ledgers** (PIPE §2 convention; the batch is INCLUDED — xw[kS] staging):

| shape | acc | xw batch | pr/pw | c8 | sq | coeff_r | xl/xh | addr/idx | **source total** |
|---|---|---|---|---|---|---|---|---|---|
| S-A / S-B | 16 | 4 | 8 | 8 | 4 | 4 | 2 | 12-20 | **58-66** |
| S-C | 16 | 8 | 4 | 4 | 2 | 2 | 2 | 12-20 | 50-58 |
| S-D | 8 | 2 | 8 | 8 | 4 | 4 | 2 | 12-20 | 48-56 |
| S-E | 8 | 4 | 4 | 4 | 2 | 2 | 2 | 12-20 | 42-50 |

**Funding verdict: every shape funds the batch with ≥ 46 regs of nominal slack under 128**
(S-B sits ≤ 112 with the batch paid). The de-scratch structure (b) is inherited (per-group
coeff_r from held sq quads — the 80 B segment dies). Caveat stated once: source ledger →
emitted VGPR is a weak predictor on this toolchain (PIPE cell A: 64+spill retreat); a census,
not this table, is acceptance — but no shape here is even nominally near the cliff, unlike
V0's 78-86-source → 128-emitted.

**Waves / grid / M=128 fit** (56 CU; W4 shards n = 3584 / 4096 / 8704 / 5120 / 5120; every
W4 n is divisible by 64, 32 AND 16 → zero row tails for TM ∈ {16,32,64}; TN=128 = the whole
M=128 chunk → zero token tails; LDS = 19.45 KB/CTA for all shapes → 3 CTAs LDS-wise always):

| shape | CTA slots (co-residency) | grid n/… | slot fit per shard | waves/SIMD |
|---|---|---|---|---|
| V0 | 112 @ ≤128 VGPR (2 CTA x 256t) | /64: 56, 64, 136, 80, 80 | **all ≤ 112: single all-resident row, 1.00 everywhere** (136 → ~1.21 rows continuous-packed, measured cost ≈ 4%) | 2 |
| S-A | **56** (1 CTA x 512t; 2 CTAs of 512t need ≤64 VGPR — unreachable) | /64: 56, 64, 136, 80, 80 | rows 1 / 1.14 / 2.43 / 1.43 / 1.43 → eff 1.00 / ≤.875 / ≤.66 / ≤.85 / ≤.85 | 2 |
| S-B @ ≤112 | 112 (2 CTA x 256t) | /32: 112, 128, 272, 160, 160 | rows 1 / 2 / 3 / 2 / 2 → eff 1.00 / .57-.875 / .41-.66 / .70-.85 / .70-.85 | 2 |
| S-B @ ≤85 | **168** (3 CTA x 256t; 3x19.45 = 58.4 KB LDS ≤ 64 ✓) | /32: same | rows 1 / 1 / 2 / 1 / 1 → eff 1.00 / 1.00 / **.62-.81** / 1.00 / 1.00 | **3** |
| S-C | 112 / 168 (as S-B) | /32: same as S-B | as S-B | 2-3 |
| S-D | 112 | /64: as V0 | **all-resident 1.00 everywhere (V0's grid)** | 2 |
| S-E | 112 | /16: 224, 256, 544, 320, 320 | rows 2 / 3 / 5 / 3 / 3 → eff .95-1.00 / .76 / .82-.97 / .85-.95 / .85-.95 | 2 |

m ∈ {32, 256} instantiations: at CT=32, m=32 gives kS=1 — no batch — so any CSHAPE gate is
**m=128-only, loud-refuse otherwise** (N7 pattern). Non-production token tiles are out of
scope by the bench's own geometry.

## 4. Kill analysis, honest order (mission item 2)

**Wall 0 — funding: NOT a kill. The forward pointer's premise is CONFIRMED.** At acc ≤ 16
the batch (xw[kS] + pr[RPT], 4-12 regs) funds under the cliff with margin on every shape;
PIPE's 10-cell "jointly unsatisfiable" verdict was a statement about V0's acc-32 tile, and
the class escapes it exactly as predicted. The pw-overwrites-pr fold and per-group coeff_r
(inherited from PIPE §6) keep the batch cost at xw[kS] net.

**Wall 1 — wait coverage: NOT a kill either, and stated plainly because the mission asks.**
Per (g,v) block the class nest is 2RPT muls + 2kS unpacks + 2RPT·kS FMAs = 48 (S-B) to 28
(S-D/E) VALU wave-instrs = **112-192 cycles of issue vs one 25-35-cycle LDS round trip** —
3-7x coverage. With the next block's batch issued under the current nest (single-deep
cross-block pipeline; volatile-pinning per PIPE cell I is the known mechanism that makes the
batch EMIT), steady-state exposed stall → ~0; the model carries a 0-512 cyc/k-tile residual
for prologue/imperfection. The class CAN both fund the batch and keep the FMA nest busy.
**That is not where it dies.**

**Wall 2 — where it actually dies, two terms the forward pointer did not price:**

**(a) Amortization.** The decode overhead the batch rides on (pw-muls 64·RPT, unpacks
64·kS, misc 240, ds 288-432 per k-tile) is per-thread FIXED work amortized over 64·acc
FMAs. Halving acc 32 → 16 (S-B) inflates issue cyc/kflop 53.7 → 63.5 (+18%); S-C +29%;
S-D/E +56%. Since the entire prize is removing V0's stall share (census bound **×1.43-1.60**
= 21.3-23.9k / 14.9k), the ceiling is
**xV0 ≤ 1.43-1.60 x 53.7/S x grid_eff**: S-B **1.21-1.35 x grid**, S-C 1.11-1.25 x grid,
S-D/E 0.91-1.02 x grid. Even before grid: **S-C, S-D, S-E can never clear 1.3** — killed in-doc.
Shavings priced and rejected: hi/lo pre-masked s_x (kills one unpack op, −11% on S,
but doubles s_x to 35 KB → 1 CTA/CU — sells the second wave, the V4 sin); group-wide batch
(xw[8·kS] = 24-32 regs — gains nothing over per-block batching whose exposure is already 0,
costs the ledger); coeff-folded pair tables (16 tables = 32 KB extra LDS + per-group refill —
dead on LDS and staging).

**(b) Token-tile/grid arithmetic.** acc ≤ 16 at TN=128 forces kS ≤ 4 → CT ≥ 32 (§2), so the
class must choose 512-thread CTAs (S-A: 1 CTA/CU, 56 slots → Gdn/Mlp/Resid queue
1.14-2.43 rows → eff 0.41-0.875, **no VGPR branch can fix it**) or doubled grids (S-B at
≤112 VGPR: 4 of 5 shards overflow the 112-slot row → eff 0.41-0.875) or buy the ≤85-VGPR /
3-CTA branch (S-B: fixes 4 grids but MlpGateUpW4's 272 CTAs still pay 0.62-0.81 AND the 85
landing is a coin-flip the v3 precedent prices pessimistically). **The pinch is symmetric:
V0's acc=32 is the minimum-overhead point that fits the cliff, and V0's 64-row grid is the
only one that covers all five W4 shards in a single all-resident row. Every step toward
pipeline-fundable acc pays amortization + grid for it. The class has no point that clears
the floor — the plane is closed, not merely unpromising.**

## 5. The measured anchor: the class optimum is V1, and V1 already lost (the decisive evidence)

Sweep variant **V1 = `TiledVariantCfg<32,64,4,32,false>` = TM 32 / RPT 4 / CT 32 / kS 4 /
acc 16 / 256 threads = S-B to the digit** (source :247; relL2 0.00e+00 vs V0 on all rows).
It was measured UNPIPELINED at M=128, full geometries (SWEEP_row.txt, PLOG-039):

| geometry | V0 GF/s | V1 GF/s | V1/V0 |
|---|---|---|---|
| AttnInput n=14336 | 3083.0 | 2276.4 | **0.738** |
| GdnInput n=16384 | 2755.7 | 2119.4 | **0.769** |
| MlpGateUp n=34816 | 3103.3 | 2220.1 | **0.715** |
| Resid6144 n=5120 k=6144 | 2365.8 | 2146.4 | **0.907** |

(Both V0 and V1 quantize to whole co-resident rows at those n — 224/112 = 2, 448/112 = 4 —
so the ratios are per-stream economics, not slot artifacts; ±12% quantization uncertainty
noted if V1 hit a 3-CTA landing.) Pipelining cannot add more than the stall-removal bound
the census licenses (**×1.43-1.60** — same wait structure: unpipelined S-B carries one wait
per xw load, 128/k-tile, same exposure class as V0's 258). Therefore the **absolute ceiling
for the best member of the class, with a PERFECT pipeline, is measured-V1-ratio x 1.43-1.60:**

| geometry | ceiling xV0 | floor 1.3 |
|---|---|---|
| AttnInput | **1.06-1.19** | UNREACHABLE |
| GdnInput | **1.10-1.24** | UNREACHABLE |
| MlpGateUp | **1.02-1.15** | UNREACHABLE |
| Resid6144 | 1.30-1.46 | only at ~full recovery, zero grid loss |

The model-only ceiling (§4a, 1.21-1.35 pre-grid) is the OPTIMISTIC reading and it already
sits on the floor; the measured anchor — which is what the floor would actually be judged
against — puts 3 of 4 geometries below it **by construction, before a single line of kernel
code is written**. The one geometry whose ceiling clears (Resid-class) needs the ≤85-VGPR /
3-CTA branch to keep its grid lossless AND near-max recovery — two unmeasured conditionals
stacked on a ×1.46 ceiling whose model class the PIPE desk already watched over-predict.

**Family verdict: NO SURVIVORS.** The honest statement, verbatim per mission: shapes in this
class CAN fund the batch and CAN keep the FMA nest busy; what NO shape in the class can do
is clear the pre-registered ≥1.3x-every-geometry floor — the amortization and grid walls
close the plane at every point. The family dies honestly, like PK (measured 0.21-0.36x),
LEANMAC (census kill), and PIPE (10-cell acceptance kill) before it.

## 6. Predicted band, and the recommendation (mission items 2/4)

- **Predicted TF/s band for the class** (best shape S-B, realistic central: ~60-75% of
  ceiling realization, W4 grids as §3): **1.5-2.6 TF/s across the five W4 shards, central
  ≈ 2.1** — i.e. **×0.75-1.1 vs V0's measured 1989-2444 GF/s (2.0-2.4 TF/s)**, against a
  floor that demands 2589 / 2586 / 3040 / 3177 / 3171 GF/s. Absolute never-never ceiling
  (Resid, full recovery, 3-CTA branch): 3.6 TF/s on ONE shard.
- **Recommendation: NO BUILD.** No kernel, no `NINFER_TILED_CSHAPE` gate, no `--cpipe` bench
  arm, no GPU window, no serving leg. The bench would measure a predicted floor-miss; the
  decisive measurement already exists (V1's sweep row) and the decisive bound is census
  arithmetic. Per the same pre-registration law PIPE died by: the family closes, no knobs,
  no second window. Serving stays at V0, 798.8 ms/chunk gemm slice; the 500-tok/s program's
  gemm-front lever list loses "pipeline-constrained tile re-sweep" and keeps only what
  V0PK_row already named (leaner-issued-ops/MAC rewrite of the DECODE side — the muls/unpacks
  this desk just showed are the class's real tax — or accept the F32-SIMT wall class).

**What would legitimately reopen the family (measured constraint changes only, not
persistence):** (i) an eligibility-law change admitting sched intrinsics/asm-class pinning at
V0's OWN tile — PIPE §5 already proved pinning was never the binder (registers were), so
this only helps if it ALSO changes allocation, i.e. effectively never; (ii) a compiler
upgrade whose allocator funds acc-32 + batch ≤ 128 scratch-0 at V0's tile (then no shape
change is needed — that is PIPE revived, with its own step-0); (iii) a decode-side ops cut
(coeff fold into a per-group staged table or e2m1 arithmetic narrowing that survives the
codec contract) that moves S in §3's table below ~48 cyc/kflop — at which point acc-16
shapes stop paying for their funding. None of these is an order this desk can request.

## 7. Contingency only-if-ordered (pre-registered, predicted to MISS; not a recommendation)

If the coordinator nonetheless orders the window against this kill, the ONLY legal shape is
**S-B** (everything else is ceiling-dead in §4/§5), and the following bars are pre-registered
here so the order cannot be softened in flight:

- Separate template instantiation `nvfp4_tiled_gemm_hip_cshape_kernel<STILES,TN>` (V0
  byte-untouched), m=128-only, `NINFER_TILED_CSHAPE` value-parsed strict-"1" gate, loud-refuse
  (PK mutual-exclusion throw), `[TILED] arm=cshape` trace; `--cpipe` bench = five W4 shards,
  M=128, 3 warmup + 20 iters, ceilings + 5 s mclk hammer, FNV-1a byte-identity vs V0 (no
  arithmetic change claimed — k-ascending order untouched), rel-L2 diagnostic must read
  0.00e+00, floor **xV0 ≥ 1.3 EVERY shard**, all-<1.3 → DEAD with no post-mortem window.
- **Step-0 order (the PIPE lesson, before ANY bench):** sketch-compile first with the LEANMAC
  §1 law flags; census BOTH cells — (i) unbounded: `next_free_vgpr` stated per instantiation,
  bar **≤ 85** (the 3-CTA branch is load-bearing; at ≤112 the grid table kills 4/5 shards);
  (ii) `__launch_bounds__(256,3)`: scratch ops 0 AND `private_segment_fixed_size` 0 (cell I's
  2.4 KB class = fail); waitcnt ≤ 20 per body over TN=128 x STILES ∈ {80, 24, 68}. Any
  signature failing → no bench, family stays closed. Census landmines (PIPE §6, both hit
  there): the .s re-emits kernel symbols at EOF (anchor labels to `^(_\S+):`); `; %bb.*:`
  parses as a label under loose regexes.
- Desk prediction for the record: step-0 likely passes (ledger 58-66) and the bench likely
  lands 0.8-1.2x — a window spent to confirm §5's ceiling table. That is why the
  recommendation above is NO.

## 8. What was NOT done (desk law)

No edit to `src/` (V0 and every variant byte-untouched; the tree stays clean at 9fe491f8f
for src/); no gate, no bench arm, no GPU time, no window requested — the kill is arithmetic
on existing censuses plus an existing measured row. Read-only everywhere except this doc;
`gated_delta_net/` and the bf16_gdn/gqa files never opened. No VRAM-adjacent constant was
touched (none exists in this desk's scope; the VRAM law is cited only as the occupancy
preference it always was).

## 9. Provenance

`results/amd/coherence/LEANMAC_NOTES.md` (the wall: 129 lgkmcnt(0)/body, 24 scratch ops, 80 B,
VGPR 128; census method + law flags) · `results/amd/coherence/PIPE_NOTES.md` (10-cell kill at
V0's tile; cell I pinning mechanism; forward pointer §5; census landmines §6) ·
`docs/amd/GEMM_PIPELINE_2026-09-18.md` (mechanism, step-0 discipline, floor ≥ 1.3x every
shard, commit 9fe491f8f) · `results/amd/coherence/TILED_SWEEP_notes.md` + `SWEEP_row.txt`
(V0-V4 measured at M=128; **V1 = 2276.4/2119.4/2220.1/2146.4 GF/s = the class optimum
measured**, relL2 0.00e+00 all rows) ·
`results/amd/coherence/V0PK_bench_run.log` + `V0PK_row.txt` (V0 W4 shard band 1991.5/1989.6/
2338.1/2444.2/2439.1 GF/s; PK dead 0.21-0.36x) · `src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu`
(:53-177 V0; :237-250 variant cfgs, V1 at :247; :255-260 the LDS/accumulator instantiation
law) · `results/amd/coherence/nvfp4_prefill_bench.cu` (:361+ run_pk probs table = the five
W4 shard geometries with STILES 80/80/80/24/68). All model constants traceable to the LEANMAC
§2 census table; every ratio in §4-6 re-derivable from this doc + those four files.
