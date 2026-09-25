# Prefill wall assessment — V340L (2026-09-19)

Seat: AMD/V340 line, shared checkout `amd/main`, no-GPU doc desk. Probe: 4x `Vega 10 [Radeon Pro V340/Instinct MI25x2]` gfx9000.
Method: adjudicate the 7-point prefill-wall summary against banked rows. No builds, no device touch.

## 0. Source under review

The 7-point claim: (1) scalar gfx900 + NVFP4 decode tax 52% issue slots; (2) per-chunk ~990 ms = GEMM 756-806 (78%) + AR 101-141 (12%) + non-GEMM 43-74 (6%) + launch/sync 15-69 (4%); (3) GEMM 2.0-2.1 TF/s/die, pure-consume 0.85-0.95x at 36-51 GB/s vs 368, 44 waits, 2 blocks/CU; (4) concurrency/pk16/mad-mix/K-split/swizzle/graph/DRAM/compiler all exonerated; (5) AR 125x ~990 us via host, no overlap, fuse+quant ~6%+6%; (6) 110 W cap, 85 s to slow band, 90 s drain, 3 min reset, 104 cold vs 63 hot tok/s; (7) software ceiling 115-125 tok/s, 200-class behind GEMM 78%.

## 1. Agree: issue wall, not DRAM

Matches `docs/amd/PREFILL_GEMM_SCALE_2026-09-17.md:20-38`:
- DRAM floor 9.3 ms/chunk (3.43 GB/rank at 368.7 GB/s) vs 798.8 measured = 86x slack.
- M=128 is 8x past packed-fp16 crossover M*=16.4, 16x past fp32 M*=8.2.
- Census 2.4 issued ops/MAC; tile shapes CLOSED per SWEEP.
- No MFMA / scalar VOP characterization is correct for gfx900.

## 2. Disagree: slice arithmetic vs P1 banked row

Banked P1 (`docs/amd/PREFILL_BODY_2026-09-17.md:8`, bin `6c8ae21399516750`, plen-1996, T=128, TP4):
`wall 1642.3 = gemm 798.8 (48.6%) + ar 70.7 (4.3%) + body 645.1 (39.3%) + gap 122.7 (7.5%)`

Against the ~990 ms / 78-12-6-4 split:
- GEMM share is ~49%, not ~78%.
- AR is ~4%, not ~12%.
- Body 645.1 ms (delta-net chunked trio, unpacks, norms, conv, attention) is the missing mass; the 43-74 ms non-GEMM row understates it ~10x.
- Prior `docs/amd/PREFILL_DECOMP_2026-09-17.md:90` unnamed ~950 ms (58%) was later named by P1 as body+gap, not launch residue.
- Action: do not concede 78% until `gemm_us` re-measured on the shipping V-arm bin (`c618d356f0cdc401`); P1 bin and current bin differ.

## 3. GEMM rate and pure-consume

- Serving GEMM class per P1: 1556.9 GF/chunk/rank / 0.7988 s = 1.95 TF/s = 18.1% of 10.75 TF/s nominal (`PREFILL_GEMM_SCALE:12-16`), bench V0 2.37-3.10 TF/s cache-served.
- The 2.0-2.1 TF/s/die, 38% fp32 / 19% pk16 claim needs a clock anchor; P1 notes 1138-1500 MHz droop in serving. All absolute TF/s rows must carry sclk sideband.
- Pure-consume 0.85-0.95x at 36-51 GB/s is consistent with issue-bound (loads expose waits), not with a bandwidth roof.

## 4. Exonerations: concur with receipts

- 113→71 regs, 3 blocks/CU = 1.02x: concur, not occupancy-starved.
- pk16 1.34x vs 2.0x required: concur; `GEMM_LEANMAC_2026-09-18.md:5-16` records V0-PK HFMA2 DEAD 0.21-0.36x (PLOG-045); forward path is fewer ops/MAC in fp32 (152→96, 2.375→1.50, predicted 3.15-3.87 TF/s bench).
- mad-mix 0.48x, K-split monotone loss, swizzle/graph/DRAM/compiler nulls: concur; graph-capture correctly noted as non-lever for chunks that are not launch-bound (gap is only 7.5% in P1).
- V-arm +5-8% plausible as next increment, not a wall-breaker.

## 5. AR tax: real but overstated

- No P2P confirmed: all six die pairs `canAccessPeer=0` both directions (`docs/amd/W7_ROUNDWALL_desk.md:103-107`, ROCm 6.2.0).
- Measured AR at prefill size is 70.7 ms/chunk total, not 101-141 ms.
- GATE-T (`W7_ROUNDWALL_desk.md:108-114`, world=2 on dies 2,3): host-staged one-shot 0.31-0.47x RCCL at 10-30 KiB decode sizes, but 1.19x (LOSES) at 1.31 MB prefill size. Zero overlap confirmed (one in-order stream).
- Fusion halves count → ~2% of chunk wall, not ~6%; AR quant at TP4/1.31 MB unmeasured. Queue both, price at ~2-3% combined until measured.

## 6. Thermal multiplier: concur with protocol correction

Per `docs/amd/NIGHT_HANDOFF_2026-09-16_mtp_thermal.md:42-56`:
- Trigger is minutes-scale sustained-load integrator, not edge temp; sclk 1500→560-775 MHz at knee; 2.82x slow after 5.5 min continuous; steady state ~1.75x prefill / ~2.6x decode; full recovery after ~3 min idle.
- 104 cold vs 63 hot (1.65x) fits inside that envelope but is state-dependent. Operating rule: <60 s bursts, 2-3 min gaps, clocks+temps sideband on every row (BOOT_BATTERY step 0c). Unsidebanded comparisons do not count.

## 7. Ceiling and what to do next

- 115-125 tok/s ceiling math assumes GEMM+AR only and omits the 645 ms body lever.
- Banked projection (`PREFILL_DECOMP:171-174`): Fix A (chunk graph, if gaps own) → 100-130 tok/s at plen ~2000; slices floor ~700-760 ms/chunk → hard ceiling ~164-178 tok/s without reopening GEMM family.
- Ranked next:
  1. P2 body-op tracer on current bin — price delta-net trio vs unpacks vs norms; body is the largest software mass.
  2. V-arm/LEANMAC integration leg (ordinal-paired, +-2% within-pair).
  3. AR fusion + quant as ~2-3% combined, measured at 1.31 MB TP4, not assumed.
  4. Keep decode + q4-KV pivot: 93% unattributed round share per summary is where real levers likely live.
- Bottom line: keep ISA/thermal/transport conclusions; re-price GEMM share vs P1 on shipping bin; do not close prefill as roofline until body 39% is killed or harvested.
