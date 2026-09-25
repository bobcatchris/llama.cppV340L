# W7 REPACK DESK — RESUME NOTE (M2+M3: mx-llama repack + v_perm decode)

Updated 2026-09-18 EOD (amended after the vperm3/co-residency push — see section below). Desk
state: **MEASUREMENT COMPLETE — verdicts banked, awaiting coordinator AMBER decision.** If this
desk dies, everything needed survives in git.

## V-PERM-3 AMENDMENT (2026-09-18 late window — co-residency push, FALSIFIED + bug closure)

Two concurrent desks ran the "<=96 VGPR -> 3 blocks/CU" push to completion and agree on every
conclusion; read BOTH rows before touching V-class kernels again:

- **Verdict: co-residency FALSIFIED at this register budget.** Survivor `c` <32r,2r/t,TN64,
  256thr> = 71 VGPR / 0 scratch / MEASURED 3 blocks/CU (vs V's 113/2) benches 1.023x V0
  (coresidency row, runs 2-4) and 0.990-1.055x (primary row, runs 1-3) — every run < the
  pre-registered 1.15 KILL bar, V still beats V3c on every geometry. The V win is ISSUE-side;
  a 3rd co-resident block buys ~nothing at these geometries. Ablations: kS=2-without-the-3rd-
  block = 0.761x (a), 512-thread lowest-VGPR = 0.863x (d) — register relief per se is
  NEGATIVE. Forcing minB=3 spills at 84 VGPR: the honest 3-block bar is <=84 VGPR at 0
  scratch, and even met it buys nothing. Rows: results/amd/coherence/W7_vperm3_row.txt
  (primary) + W7_vperm3_coresidency_row.txt (ablations/band-gates/shas).
- **TWO latent V-family bugs found, closed RED->GREEN by BOTH desks independently** (uniform
  bench data could not see either; the banked row's relL2-0.0 receipts were uniform-data
  artifacts — trap 4 claimed its next victims): (1) restage word order swapped vs the decode's
  declared 0,4,2,6,1,5,3,7 consumption (1/4 of products wrong on real data); (2) stage read x
  at the BLOCK-LOCAL token under the V-family's grid.y=M/TN tiles (tokens >= TN got t%TN's x
  = half the outputs wrong at M=128) — trap 5's twin: the READER was fixed block-local, the
  STAGE was left global-shaped. Production V0 immune (TN = whole M, grid.y = 1). Banked V/PV
  timing verdicts stand (fixes are value-invariant on uniform data); **any integration of V
  MUST carry both fixes**, now landed in the banked bench file (commits 67eb41674/e7985fadf)
  and in the primary desk's harness copy. Permanent guard: `--v3parity` (patterned
  all-256-byte codes + positional scales + per-element x; full-array bit-compare vs V).
  RED->GREEN chain with binary shas in W7_vperm3_coresidency_row.txt.

## Where things are

- Bench + kernels: `tools/v340l/w7_repack_bench.cu` (P, V, PV kernels + repack kernels +
  `--debug` mode). Shipped kernel untouched. Probe: `tools/v340l/w7_vperm_probe.cu`
  (banked in 9670115f9 with the gfx900 perm convention).
- Row: `results/amd/coherence/W7_repack_row.txt`; raw logs `W7_repack_run1/2/3.log`
  (run3 = final). Census .s at /tmp/w7_repack_census_final.s (regenerate: hipcc -S flags in
  the row header). Binaries in /tmp (NOT banked — rebuild from source, ~90 s).
- Phase log: external code read -> probe (PARITY PASS) -> P arm -> V arm -> census (first
  compile RED 256 VGPR, fixed census-first, no bench seconds wasted) -> bench (3 runs: two
  RED correctness runs caught by --debug + rel-L2, run3 GREEN) -> verdicts banked.

## Verdicts (final, pre-registered bars)

- **P (repack planes, mx-llama mechanisms): KILL 0.983x** — bit-identical to V0, measured
  wash/negative. Do not re-derive for the tiled GEMM (V0's scale path was already the quad
  trick; the alias pad does nothing here).
- **V (v_perm decode, fp16-pair consumer): AMBER 1.136x** (bar 1.15; per-geometry
  1.07/1.14/1.09/1.24). Census GREEN: 113 VGPR / 0 B scratch / 32 waits (V0: 212) /
  0.25 v_perm per value. relL2 0.0 vs REF, bit-identical to V0 on bench data. NO requant.
- **PV: AMBER 1.055x, dominated by V alone** — P's planes make V slower; if V integrates it
  uses V0's canonical planes.

## Coordinator decision owed

V is an AMBER lever on a front the pk16 row closed at 2.0x. Integration cost if promoted:
port `w7_v_tiled_kernel` + perm-pool constants behind a `NINFER_TILED_V` gate (variant-
launcher shape, byte-identical off) — the kernel is bench-local, production header untouched.

## Hard-won traps (do NOT re-learn)

1. kS=8 (TN=128) V arm = HFMA2-PK death (256 VGPR + 840 B scratch). kS=4 + ROLLED group loop
   + per-group half2 coeff = 113 VGPR / 0 scratch. Census-first caught it pre-bench.
2. The compiler's product-hoisting (1024-fma body) is dataflow-SHAPE sensitive: P compiled
   2048-fma/256-VGPR until coeff mirrored V0's `coeff[r][4]`-per-k-tile shape.
3. gfx900 perm convention is EMPIRICAL (n[2]: 0->second arg, 1->first; 0x08-0x0C -> 0x00,
   0x0D-0x0F -> 0xFF) — the truth-table probe exists because two same-spelling kernels
   disagreed on the first probe attempt.
4. Uniform bench data (0x11/0x3C memset) HID positional decode bugs (permutation-invariant
   sums). The per-group-coeff `--debug` mode + rel-L2 gate exposed all three RED kernels
   (mask swap = exactly 1/4 outputs; restage lane bug = 0.75; token-domain bug = zero half).
   Any future decode-swap desk keeps both. CONFIRMED IN THE WILD by the vperm3 desks: even
   debug + relL2-on-uniform still missed two real bugs (restage order, stage token domain) —
   only PATTERNED data catches them; `--v3parity` is the proven cell shape.
5. s_x token domain: stage is block-local 0..TN-1; reader MUST use local tokens + the same
   XOR param. V0 (TN=whole M) never shows this; any TN<M port must re-derive it. AMENDED:
   the STAGE side must read x at the GLOBAL token (blockIdx.y*TN + t) — the V-family port
   got the reader right and the stage wrong (the mirror image of this trap).

## Cross-validation cited in the row

Team Green PRMT decode atom +8.2% kernel / +3.2% serve; "PRMT is a table without memory" —
same mechanism class as V; their shared-LUT NO-GO (-7..-14%) mirrors our V0 LDS-LUT baseline.
