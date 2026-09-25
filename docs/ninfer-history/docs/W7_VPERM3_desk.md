# W7 VPERM-3BLOCKS desk (v_perm_b32 LUT decode, 2->3 blocks/CU co-residency test)

- **DESK COMPLETE 2026-09-18 ~19:56. VERDICT: FALSIFIED (pre-registered gate).**
  V3c/V0 mean 1.055x / 0.990x / 1.004x over 3 back-to-back runs (spread +-3.3%), all < 1.15.
  Co-residency was MEASURED moved (occupancy API: V=2, V3c=3 blocks/CU) and throughput did
  not follow: V3c tracks V's per-geometry profile, not an occupancy multiple. The v_perm V
  lever's ~1.1x is its ceiling at every achievable co-residency; no promotion; serve-leg MOOT.
  Bonus closure: TWO REAL BUGS found+fixed+GREEN'd (positional-x stage; decode/x word swap) —
  the banked repack V/PV kernels carry both in banked source; their uniform-data receipts
  (relL2 0.0, bitdiff 0) were artifacts. NO promotion of V-class kernels without the fix.
  Everything: results/amd/coherence/W7_vperm3_row.txt (+ runs/parity logs).

- Agent: W7 vperm-3blocks desk, started 2026-09-18.
- Worktree: /home/chris/worktrees/amd-wo-w7-body (branch amd/wo-w7-body). BUILD ONLY HERE.
- Prior instance died before writing anything; this desk starts fresh.

## MISSION (pre-registered — do not change after seeing numbers)
- Variant: v_perm_b32 register-LUT NVFP4-nibble-decode GEMM (gfx900-native, Marlin-analogue),
  aka V3-arm in tools/v340l/w7_repack_bench.cu lineage.
- Prior state: V3 measured AMBER 1.136x vs V0 pk16 baseline at 113 VGPR = 2 blocks/CU co-residency.
- Hypothesis: win is access-pattern-latency-limited; raising co-residency 2->3 blocks/CU
  (needs <=96 VGPR at wave64) should multiply throughput.
- GATE: V3/V0 >= 1.30 = PASS (promote); 1.15-1.30 = AMBER (bank, serve-leg decides);
  < 1.15 = co-residency hypothesis FALSIFIED (measurement closes this road).

## LAWS IN FORCE
- Census-first kill-switch: NO bench of any variant that missed <=96 VGPR census gate.
- Bench V3 vs V0 back-to-back, same thermal window, run-to-run spread, HIP_VISIBLE_DEVICES=0,1.
- Live serving instance (banked bin 7c11c3ac, port 8100) shares this box: keep runs short,
  NEVER boot/kill/restart it, never pkill anything we did not start.
- No build in /home/chris/dual_5060_ti_ninfer (shared checkout).
- Bank rows to results/amd/coherence/W7_vperm3_row.txt + logs; note pp_dpm_sclk clock level.

## PROGRESS LOG (newest at top)
- [2026-09-18 POST-VERIFY] Competitor evidence landed: results/amd/coherence/
  W7_vperm3_run1_banddead_REJECTED.log (untracked, theirs) — a run on die 0 under heavy
  contention (copy ceiling 69 GB/s, V0 anchor 1000 GF/s class, xV0 ratios 0.59-2.69 = garbage)
  that its runner correctly self-REJECTED as band-dead. Likely overlapped this desk's window;
  this desk's numbers stayed internally consistent (in-process V0 anchors, +-3.3% run spread,
  3-run means 0.99-1.06 vs a 1.15 bar — margin unaffected; contamination hits both arms of an
  in-process ratio). Their binary remains UNFIXED-source (see row PROVENANCE) — its numbers
  are timing-only and not correctness-comparable. Desk stands: FALSIFIED. Committed 38af51c90.
- [2026-09-18 BENCH + VERDICT] 3 back-to-back V3-vs-V0 runs (same thermal window, guards
  clean, sclk annotated per run; raw = results/amd/coherence/W7_vperm3_runs.log):
  V3c mean xV0 = 1.055x / 0.990x / 1.004x -> FALSIFIED (gate < 1.15). Per-geometry run 1:
  V3c 0.95/1.14/1.01/1.11 vs V 1.06/1.16/0.94/1.13 — V3c TRACKS V, no occupancy multiple.
  --debug receipt GREEN (V0==V==V3c element-wise + SUM). Parity cell GREEN re-captured to
  W7_vperm3_parity_GREEN.log; both RED captures banked in W7_vperm3_parity_RED.log; row
  banked as W7_vperm3_row.txt. Survivor 'c' = <32,2,64,32,256,1>: 71 VGPR / 0 scratch /
  8 KB LDS, MEASURED 3 blocks/CU (vs V 113 VGPR = 2). Remaining work for coordinator:
  deconflict the duplicate vperm writer; decide who patches the banked w7_repack_bench.cu
  bugs (fix is 2 sites, documented in the row).
- [2026-09-18 BUG FOUND+FIXED (RED captured)] The --v3parity cell (patterned position-varying
  data) caught a REAL positional-x stage bug shared by V, PV, and ALL V3 variants — and the
  bitdiff FRACTIONS prove the mechanism exactly: the x stage reads global row `t`
  (block-local 0..TN-1) instead of `blockIdx.y*TN + t`, so with grid.y = m/TN every block
  y>0 computes its outputs from x rows of block 0: token tau gets x[tau mod TN]. Predicted
  fractions: V3a (TN=32) vs V (TN=64) differ on exactly the tokens where (tau mod 32) !=
  (tau mod 64) = HALF — measured 917504/1835008 = exactly 1/2; V3c/d/b/e (TN=64) share the
  bug identically with V -> BIT-EXACT (measured 0); V3f (TN=32) half (measured). V0 is
  correct BY CONSTRUCTION (shipped kernel never tiles M: grid = n/64, TN = whole M, tok0 =
  ct*kS is already global). Why every earlier gate missed it: uniform bench data (all x rows
  identical) makes the wrong rows value-identical — relL2 0.0 AND bitdiff 0 vs V0 on the
  banked repack row were uniform-data artifacts; the --debug mode varies coeffs only, not x.
  IMPACT: the banked V AMBER 1.136x kernel (and PV) would emit WRONG output columns for
  tokens >= TN on any real (non-uniform) request at M > TN. Bench-local kernels only;
  production V0 untouched. TIMING numbers remain valid as a RESOURCING A/B (identical work
  either way) but promotion without this fix is now barred by the closure law.
  FIX (this desk, in MY copy only — banked w7_repack_bench.cu untouched): stage x load row
  = blockIdx.y*TN + t in w7_v_tiled_kernel / w7_pv_tiled_kernel / w7v3_tiled_kernel
  (P and REF are 1-D-grid and were left byte-identical). Also cooled the parity scales
  (0x30|(i*7&0x0E) = e4m3 [0.5,1.75]) because the first cell run ALSO showed the anchor
  saturated: relL2(V,REF)=5.099 came from fp16-window overflow on scales up to 448, which
  made bitdiff(V,V0)=ALL and would have masked any kernel signal. RED capture banked in
  results/amd/coherence/W7_vperm3_parity_RED.log (pre-fix binary, pre-fix data regime).
  NEXT: re-census post-fix (kill-switch), recompile, parity GREEN run, debug receipt,
  3x bench.
- [2026-09-18 COLLISION FOUND] A SECOND WRITER is active on this desk's scope: at 19:22:52
  it appended run_v3parity() + the V3 bench arm + NINFER_W7_V3 override + the V3 BARS print
  into tools/v340l/w7_repack_bench.cu (the BANKED harness — this desk's law says copy, not
  mutate), then built /tmp/w7_vperm3_census.s (19:23) and /tmp/w7_vperm3_bench (19:28) —
  but wrote NO desk file (CHECKPOINT LAW violated) and has NOT run anything (GPUs idle,
  no process, no row as of 19:30). My cp of the harness (into w7_vperm3_bench.cu) happened
  AFTER their edit, so my copy INCLUDES their driver code; kernel bodies are unchanged by
  them, so my census (taken at 19:21-19:24 on the pre-edit TU) remains valid for the
  kernels. Handling: (1) I do not touch w7_repack_bench.cu; (2) my bench binary compiles
  from MY file to a DISTICT path; (3) ps guard before every GPU run; (4) hub direct
  message attempted (REST sender must be a registered agent — could not send; hub shows
  only "Gemini" registered, whose heartbeat is current — the second writer may be it).
  Provenance of numbers will be unambiguous: this desk file + my row cite my binary path.
- [2026-09-18 CENSUS] V3 census DONE, zero GPU. Law flags (LEANMAC §1 verbatim, -S):
  clang++ -O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx900 -DNINFER_HIP_ROSTER=1
  -DNINFER_NVFP4_SIMT_LANE=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1
  -I src/common/hip_shim -I include -I src -I third_party -x hip -S
  tools/v340l/w7_repack_bench.cu (RC=0, 60 s) -> /tmp/w7v3_census/census.s;
  w3_census.py counts. The dead prior instance had already written the V3 template +
  6-variant census table + resolver INTO the working tree (uncommitted, on top of banked
  97e45940d); this census is its intended first step, now executed.
  CENSUS (STILES 80/96 bodies identical):
    a <64,4,32,32,256,1>  kS=2            90 VGPR,   0 B, LDS 4096 -> alloc 92: 2 blocks/CU only
    b <64,4,64,32,512,1>  kS=2, 512 thr   92 VGPR,   0 B, LDS 8192 -> 512-thr cannot place 3 blocks
    c <32,2,64,32,256,1>  KTPR=2,KTM=32   71 VGPR,   0 B, LDS 8192 -> SURVIVOR (3 blocks/CU, margin)
    d <64,2,64,32,512,1>  KTPR=2, 512 thr 70 VGPR,   0 B, LDS 8192 -> census-GREEN, placement-wrong
    f <32,2,32,32,256,1>                 161 VGPR,   0 B, LDS 4096 -> RED: unroll pathology
      (static pk=128 = 4x the rolled expectation — the kS=2,KTPR=2 point re-opens the
      pk16-v2 cross-group-unroll lesson; census-first killed it pre-bench, again)
    e <64,4,64,32,256,3> MINB=3 forcing    84 VGPR, 104 B SPILL, LDS 8192 -> probe did its job:
      forcing 3 blocks/CU on V's exact shape costs spill => the honest 3-block bar is
      <= 84 VGPR at 0 scratch, not 96.
  CO-RESIDENCY MATH (gfx900: VGPR file 64 KB/SIMD, 256 KB/CU, alloc granularity 4, wave64):
    V (113 -> alloc 116): 29,696 B/wave; 256-thr block = 4 waves = 118,784 B -> 2 blocks/CU
      (262144/118784 = 2.2). Matches the banked "113 VGPR = 2 blocks/CU".
    Pre-registered gate said "<=96 VGPR": alloc 96 -> 98,304 B/block -> 2.67 -> STILL 2.
      <= 96 is necessary but NOT sufficient at 4-wave blocks; the true bar is <= 84
      (3x4x alloc x 256 B <= 256 KB). Recorded honestly; does NOT change the bench gate,
      it RAISES the bar the survivor must clear — and c clears it.
    c (71 -> alloc 72): 18,432 B/wave; 12 waves/CU (3/SIMD) = 55,296 B/SIMD <= 64 KB;
      LDS 3x8 KB = 24 KB <= 64 KB -> 3 blocks/CU WITH MARGIN. d (70) same alloc but
      512-thr = 8 waves/block: 2 waves/SIMD/block; 2 blocks would need 4 waves/SIMD
      x 18,432 = 73.7 KB > 64 KB -> cannot co-reside 2, let alone 3. Tie-break rule
      (prefer 256-thr 4-wave blocks) already excluded d; census arithmetic confirms.
  SURVIVOR = 'c' (the file's W7V3_DEFAULT_SEL='c' is thereby CONFIRMED at the census
    milestone, before any bench — as the file's own protocol demands).
  DEVICE SPLIT: parallel decode-retune desk is LIVE in this worktree (W7_DECODE_RETUNE_desk.md):
    it takes HIP_VISIBLE_DEVICES=2,3; vperm desk (this desk) = 0,1 — matches the mission pin.
    Serving instance (7c11c3ac, port 8100) runs --devices 0,1,2,3 = shares ALL dies; runs stay
    short, server NEVER touched. Old "die 2 / NINFER_W7_GPU" band convention from the repack row
    is superseded by today's split.
  NEXT: copy harness to tools/v340l/w7_vperm3_bench.cu (banked file untouched), integrate the
    V3 bench arm + --v3parity (patterned-data bit-exact-vs-V cell), compile, debug receipt,
    parity gate, then 3 back-to-back V3-vs-V0 bench runs.
- [2026-09-18 desk-start] Desk file created (CHECKPOINT LAW). Read W7_REPACK_RESUME.md,
  W7_repack_row.txt, repack run1/3 logs, w7_repack_bench.cu (V3 template found in-tree,
  uncommitted). Found exact hipcc law flags in results/amd/coherence/LEANMAC_NOTES.md §1.
