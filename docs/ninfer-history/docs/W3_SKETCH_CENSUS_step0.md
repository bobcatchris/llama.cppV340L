# W3 SKETCH CENSUS — step-0 zero-GPU verdict: **KILL** (no-GPU CODE desk, 2026-09-18, amd/wo-w7-body)

**Verdict in one line:** the packed-fp16 (Rapid Packed Math) GEMM tile sketch for gfx900 at
M≥256 compiles CLEAN packed arithmetic — native `v_pk_fma_f16`, zero scalarization, and at the
A=16 accumulator point it is SPILL-FREE at 102 VGPR under the 2-wave cap — but it **fails the
pre-registered acceptance matrix on the wait row (35 `lgkmcnt`-class `s_waitcnt` per rolled
body vs the ≤20 bar), and the kill is overdetermined by the M≥256 re-derivation the desk was
ordered to run: the pk16 group-flush tax (+50% on the MAC instruction count) and the 2A
accumulator-pair pricing make the A=32 point REGISTER-DEAD (136-140 B spill on every attempt)
and leave the A=16 point at a 39-49% of pk16-peak issue model vs the ≥54% G1 bar — so no tile
in the enumerated M≥256 plane reaches the 1000 tok/s physics class even with zero-stall waits.
** KILLED IN-DOC per the PK/LEANMAC/PIPE/CSHAPE precedent: 3 design compiles, 1 probe compile,
0 GPU seconds, one morning of desk time. The family does not go to a window.

## R2 — start from the ledger (closes the desk's first obligation)

The closest prior failures, and what differs here:

- **HFMA2-PK** (V0PK_row + LEANMAC_NOTES §4 post-mortem): packed arithmetic PROVEN innocent
  (1024 native `v_pk_fma_f16`/body, zero scalarization) but died of RESOURCING —
  `next_free_vgpr` 256 (1 wave64/SIMD), 804-820 B/thread scratch, 478-570 waits/body. THIS
  desk differs by design: it encoded LEANMAC §4's lesson ("a fp16 window must shrink the fp32
  accumulator live set, not add to it") as census points acc+window = 16+16 and 32+32 (PK ran
  32+32 PLUS coeff2[4][4] half2 PLUS unbatched loads), rolled the group body so only ONE
  batch unit is live, and batched every (g,v) block's LDS reads (PIPE's mechanism, which PK
  never had). Result: the resourcing half of PK's death IS cured at A16 (102 VGPR, 0 scratch —
  measured) — but the wait wall and a NEW tax (below) are not.
- **PIPE** (PIPE_NOTES): batching the xw loads was confirmed THE wall (129
  `s_waitcnt lgkmcnt(0)`/body at V0), and the 10-cell probe killed every funding attempt at
  V0's acc-32 tile. THIS differs: the pk16 MAC-density doubling was the ledger's own
  "reopen only if" for the acc arithmetic (G1 note; CSHAPE entry) — this desk re-derived it at
  M≥256 as ordered. The re-derivation FAILS the hope: see "the re-derivation" below.
- **constrained-tiles / CSHAPE** (GEMM_CONSTRAINED_TILES_2026-09-18.md): killed at M=128 in
  FP32 (1 MAC/`v_fma_f32`) on amortization (acc≤16 shapes +18-56% overhead) and grid shapes.
  THIS differs: both walls re-derived at pk16 × M≥256. RESULT IS ASYMMETRIC — the grid wall
  LIFTS (measured grid math below: every W4 shard ≥0.76 makespan efficiency at M=256/512, the
  queue-or-overflow kill gone), but the amortization wall REVERSES: pk16 does not remove
  decode overhead, it ADDS a new fixed term — the group flush — so the plane re-closes from
  the other side.
- **LEANMAC**: fp32 census kill (LLVM already hoists); no arithmetic-class overlap, but its
  census METHOD (law-flag `-S` compile, op-mix table vs model, `/tmp` staging) is this desk's
  method, verbatim.

## THE RE-DERIVATION AT M≥256 (what the MAC-density doubling actually buys)

Per rolled body (= ONE 16-value group; the group loop rolls, all else unrolled), thread = 4
rows × kS tokens (A = RPT·kS outputs):

- MAC instructions: `A × 8` pairs × 1 `v_pk_fma_f16` each = **2 MACs/instruction** (the
  doubling, real and measured: A16 → 128 pk/body, A32 → 256).
- The flush (fp32 coeff fold at group end — the pre-registered PK numerics, W=16):
  per output 2 cvt + 1 add + 1 fma = 4 ops → **16·A ops per body = +50% on the pk-MAC count**.
  Measured at A16: 32 `v_cvt_f32_f16` + 16 `v_add_f32` + 16 `v_fma_f32` per body against 128
  pk — exactly the model. The flush is shape-INVARIANT per output (it scales with A like the
  MAC does), so no RPT/kS split escapes it: it is the NVFP4 per-16-value group structure
  itself, priced by the format.
- Accumulators: the fp16 window (A half2) must coexist with the fp32 contract acc (A) → **2A
  accumulator-class registers**. MEASURED pricing at the 128-VGPR/2-wave cap:
  A=16 (2A=32) fits with 0 scratch (102 VGPR); A=32 (2A=64) does NOT — 136 B (256-thread) and
  140 B (512-thread) spill on EVERY attempt, all three iterations. The CSHAPE band's top
  (32 fp32 acc) is unreachable as an A=32 point; it exists only as A16's window+acc pair.
- Issue model at the ONLY fundable point (A16, from the MEASURED op mix, conservative
  ds-billed): ≈260 VALU + 74 ds per body → ≈5,280 cyc/k-tile/wave vs 131,072 FLOP/wave =
  **24.8 FLOP/cyc/wave = 38.8% of the gfx900 pk16 peak (64 F/c/w)**; 49% under the optimistic
  ds-co-issue convention. The G1 bar is **54% sustained**. A32 (had it held registers)
  models 45-56% — at the bar only under every optimism at once, and it cannot hold registers.
  **The MAC-density doubling is fully consumed by the flush tax + accumulator doubling before
  any stall is paid.** This is the R5 honest number: the ledger's "pk16 re-opens the acc/grid
  arithmetic at M≥256" is answered NO for acc, YES-but-irrelevant for grid.

Grid arithmetic at M≥256 (for completeness, since CSHAPE died on it): TN=64 tiles → grid
(n/64)·(M/64); at M=256 = {224, 256, 544, 320, 320} CTAs over 112 slots (2 CTAs/CU at 102
VGPR, LDS 10 KB ×3-CTA fine) → makespan rows 2.00 / 2.29 / 4.86 / 2.86 / 2.86 → eff
1.00 / 0.76 / 0.97 / 0.95 / 0.95 (ceil-quantized); M=512 improves further. The five W4 shard
names per CSHAPE §3 (n = 3584/4096/8704/5120/5120; STILES=80 = the K=5120 class compiled
here). No grid kill — the kills are all per-thread.

## THE ACCEPTANCE MATRIX — measured (source: `w3_final.s` census; counters in
`tools/v340l/w3_census.py`; compile = LEANMAC §1 law flags, standalone TU)

| # | row (pre-registered) | measured, A16 `<80,64,4,16>` | measured, A32 points | verdict |
|---|---|---|---|---|
| 1 | ≤128 VGPR/thread, 0 B scratch | **102 VGPR, 0 B** (`next_free_vgpr 102`, `private_segment_fixed_size 0`; launch_bounds(256,2) honored) | 128 VGPR + **136 B / 140 B scratch** (256-thr / 512-thr) | **PASS** (A16) / FAIL (A32) |
| 2 | ≤16-32 fp32 acc/thread (justify from re-derivation) | **16 fp32 acc + 16 half2 window = 32 accumulator-class**; the re-derivation prices 2A as the real budget and MEASURES A=32 unfundable → 16 fp32 is the only legal point, not a preference | 32 fp32 + 32 half2 = 64 class → spills | **PASS** (with the reversed-direction justification) |
| 3 | ≤20 s_waitcnt (lgkmcnt-class)/body, target single-digit; batched single-wait LDS blocks | **35 lgkmcnt + 9 vmcnt = 44 s_waitcnt/body** (body = 1 group, 68 ds_reads). Full-drain subset: **4 `lgkmcnt(0)` + 4 `vmcnt(0)`**; the other 31 are partial-count drains (`lgkmcnt(1..14)`) — a pipelined batch, physically ~4 round trips per 68 LDS reads | 39 lgkmcnt + 40 vmcnt/body (incl. spill reloads) | **FAIL** — the row counts instructions; 35 > 20. The matrix is not reinterpreted post-hoc (see Honesty) |
| 4 | x staged as fp16 pairs; fp32 coeff folds at group flush (PK numerics) | staging = exact bf16→fp16 pair bridge (`v_cvt_f16_f32` present, 2/word), LDS = packed pairs (all x reads `ds_read_b32`); flush = 2 cvt + add + fma per output folding fp32 `coeff_r` (de-scratch: held scale quads, 4-reg coeff live range, 0 spill) | same structure | **PASS** |
| 5 | native `v_pk_fma_f16` in the inner loop, zero scalarization | **128/body, all MACs; `v_fma_f32` = 16 = exactly the flush fold term; zero `v_mac_f32`, zero scalarized MAC path in ANY iteration** (PK's innocence reproduced at the M≥256 tile) | 256/body, same purity | **PASS** |

Anchor for scale — shipped V0 (banked receipts, LEANMAC_NOTES/PIPE_NOTES): 128 VGPR, 80 B
scratch, 129 `lgkmcnt(0)` per 16-block body (all FULL drains), 1024 `v_fma_f32` + 144
`v_mul_f32` per body, 19.45 KB LDS, measured 2.0-2.4 TF/s/die fp32.

## THE ITERATIONS (3 = the plan's stated 2-3 budget, all censused)

| iter | design delta | A16 result | A32 result | fate |
|---|---|---|---|---|
| v1 | V0's full-unroll body + half2 window/flush + `__launch_bounds__(256,2)`, shift/or half2 build | VGPR 128 (capped), **712 B scratch**, 149 lgkmcnt + 185 vmcnt/body | **1292 B scratch**, 262 lgkmcnt/body | RED everywhere — the 32-block linear body gives the scheduler cross-group freedom; source ledger (~95 regs) predicted badly, exactly PIPE §4.3's warning |
| v2 = **final** | ROLL the group body (1 batch unit live); batch = 2 value-pairs; coeff reads ride batch-0's wait; `__builtin_bit_cast` half2 (v1's construction cost 122-232 real `v_lshl_or`/body); u32 row offsets | **102 VGPR, 0 B scratch, pk 128 native, waits 44/body** | 128 VGPR + 136/140 B spill, waits 79/body | the resourcing cure WORKS; only row 3 remains RED |
| v3 | volatile-pin the batch LDS reads (PIPE cell-I's emission mechanism) | VGPR 112, 0 B, lgkmcnt 10/body — **BUT the volatile `__shared__` read lowers to `flat_load_dword glc`** (generic path, NOT `ds_read`): 73 serial per-load `vmcnt(0)` waits, total **81 s_waitcnt/body = a REGRESSION** | unchanged RED (136/140 B) | REJECTED: the lgkmcnt drop to 10 was a bucket artifact — the loads LEFT the lgkmcnt class. The census caught it (R5: measurement beats the metric). Receipt excerpt banked below; mechanism preserved behind `W3_VOLATILE_PIN` (default OFF) in the artifact |

Probe (banked): `__builtin_amdgcn_fdot2` **exists in this toolchain but requires target
feature `dot10-insts`, absent on gfx900** — "needs target feature dot10-insts" compile error
(`/tmp/w3_sketch/dot2test.cu`, this session). `v_dot2_f32_f16` is NOT reachable on gfx900
from plain C++ (it is a gfx906+/RDNA instruction). The flush tax has no dot2 rescue on this
silicon; on gfx906-class hardware a dot2 form deletes BOTH the flush and the fp16 window
(fp32-only acc, ~0.56 ops/MAC) — that is a HARDWARE reopen condition, not a software lever.

## THE BINDING CONSTRAINT, and why no tile change clears it

Row 3's mechanism, to the digit: with grouped source order, the pressure-driven scheduler
emits the batch as a group but drains it with partial-count waits (35 = 4 batches × ~8
partial drains + prologue/staging), not the single `lgkmcnt(0)` the row was written for. The
only known forced-emission mechanism regresses to the flat/generic path (v3). And the row's
letter cannot be met by ANY shape move: bigger batch units raise the live set (A32 proves the
cap rejects it); smaller batch units raise the wait count linearly; RPT↓/kS↑ shapes raise
x-read and flush share per MAC (worse physics than A16, not better). Overdetermination: even
granting row 3 by its intent (4 full drains), the A16 issue model (39-49%) sits under the 54%
G1 bar before stalls — **the family cannot reach 13.5 TF/s/die at any point of the
enumerated M≥256 plane.** Killed like CSHAPE, but by a re-derivation, not a transfer.

Named reopen conditions (measured constraint changes only — R4):
1. **Compiler/eligibility**: a toolchain (or an eligibility-law change admitting
   sched/asm-class control, banned today) that emits ≤20 (or single-digit full-drain) waits
   per body at ≤112 VGPR. The hardware pipelining mechanism demonstrably works (partial-count
   drains exist in the emitted stream); only the instruction count misses. This is the ONE
   live crack — a window owner may weigh it against the physics line, which says even a pass
   does not clear 54%.
2. **Hardware class**: gfx906+ (`dot10-insts`) where `v_dot2_f32_f16` deletes the flush tax
   and the fp16 window — re-opens A32-flat acc within the cap at ~0.56 ops/MAC.
3. None from tile shape. The M≥256 plane is closed by measured terms.

## Honesty row

- The matrix is NOT weakened: row 3 fails as written (35 > 20) and the desk's verdict is KILL
  on that basis alone. The "4 full-drain waits" analysis is recorded as evidence about the
  WALL's physics (per R6, the observable effect), NOT as a pass — adjudicating intent-vs-letter
  in the desk's favor after a miss is exactly the post-hoc move the pre-registration exists to
  prevent. It is banked because it changes what a compiler-fix reopen would be worth.
- The sketch is a resource-census artifact, not a verified kernel: numerically it is
  unexercised by design (runtime `inputs_ok` guard; epilogue writes zeros when skipped). No
  GF/s claims are made anywhere; all performance content is the issue MODEL from measured op
  counts, labeled as such.
- v1's census is quoted from this session's compile of the superseded v1 form (design delta
  fully described in the iterations table); the banked artifact regenerates the v2/final
  numbers exactly (verified: default rebuild reproduces 102/0/44).
- Zero GPU work, zero servers, zero `pkill`, zero writes outside
  `/home/chris/worktrees/amd-wo-w7-body/` and `/tmp/w3_sketch` (cleaned). The shared checkout
  was read-only. `df -h /` checked (9.3 G free; artifacts ≈ 65 KB total).

## Artifacts (all under the lane worktree)

- `tools/v340l/w3_pk16_sketch.cu` — the census sketch (3 instantiations: A16 256-thr, A32
  256-thr, A32 512-thr; guard; de-scratch; batched reads; `W3_VOLATILE_PIN` default-OFF v3
  receipt; dot2-probe receipt in header).
- `tools/v340l/w3_census.py` — the census counter (label-anchored bodies per PIPE §6
  landmines; `.amdhsa_kernel` metadata parsing).
- `results/amd/W3_sketch_isa_v1.txt` — trimmed ISA: `.amdhsa_kernel` metadata + full hot body
  (label..`s_endpgm`) of the deciding A16 point, with the measured census header.

## Provenance

`docs/amd/PREFILL_1K_PLAN_2026-09-18.md` (G1 rows, ledger, R1-R7, W3 budget) ·
`docs/amd/GEMM_LEANMAC_2026-09-18.md` + `results/amd/coherence/LEANMAC_NOTES.md` §4 (PK
post-mortem: VGPR 256 / 804-820 B / 478-570 waits / 1024 native pk; the shrink-the-fp32-set
lesson; the §1 law-flags command) · `docs/amd/GEMM_PIPELINE_2026-09-18.md` +
`results/amd/coherence/PIPE_NOTES.md` (129-wait wall; batch mechanism; cell-I volatile pin;
§6 census landmines: `^(_\S+):` anchoring, `.amdhsa_kernel` text fields) ·
`docs/amd/GEMM_CONSTRAINED_TILES_2026-09-18.md` (the fp32@M=128 kill this desk re-derived) ·
`src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu` (V0 tile law, XOR swizzle, scale-quad trick,
TILED-PK numerics pattern W=16, bf16x2→f16x2 bridge contract) — all read-only in
`/home/chris/dual_5060_ti_ninfer`. V0/PK/PIPE/V4 anchors cited from the banked receipts, not
re-measured (same flags, R7).

**Ledger row owed (R1):** this desk cannot write the shared checkout — the chair should land
the row in `docs/amd/PREFILL_1K_PLAN_2026-09-18.md`: family "W3 pk16 sketch (A16/A32, rolled
group, batched LDS)", predicted "MAC-density doubling re-opens pk16 at M≥256 (G1 note)",
measured "waits 35/body (row RED), A32 spill 136-140 B, issue model 39-49% < 54% bar — killed
at zero GPU, 3 compiles", reopen "(i) compiler/eligibility wait emission at ≤112 VGPR,
(ii) gfx906+ dot2 class", receipt `worktrees/amd-wo-w7-body/docs/amd/W3_SKETCH_CENSUS_step0.md`.
