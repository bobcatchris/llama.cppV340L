# WORK ORDER: LEANMAC fp32 op-count GEMM bench (A10) — prefill lever, top priority

Owner: desk agent. Coordinator: resolves ALL infrastructure; you execute the task.
Status: READY — this task needs NO server, NO serving window, NO tree build. Pure standalone bench on GPU die 2 or 3.

## Task
Bench the LEANMAC fp32 op-count-reduction GEMM variant against V0. Banked prediction:
3.15-3.87 TF/s vs V0's measured 2.37-3.10 (GEMM_LEANMAC_2026-09-18.md, PLOG-045: V0-PK
HFMA2 dead 0.21-0.36x; forward path = fewer ops/MAC in fp32, 152->96 ops).

## Steps
1. Read docs/amd/GEMM_LEANMAC_2026-09-18.md fully — the variant design + prediction are there.
2. Build the bench as a STANDALONE hipcc cell (pattern: tools/v340l/w7_pk16_bench.cu — same
   harness family; compile with /opt/rocm/lib/llvm/bin/clang++ --offload-arch=gfx900 -O3
   -I src/common/hip_shim -I include -I src -std=gnu++20). Pin HIP_VISIBLE_DEVICES=2 (die 3
   as fallback). If a compile takes >15 min, that is normal for big TUs — do not restart it.
3. Census first (kill-switch): ops/MAC count and VGPR/scratch from --save-temps; bench only
   if it clears the census bar the LEANMAC doc states.
4. Bench V0 + LEANMAC on the 4 standard geometries (same shapes as W7_repack/pk16 rows),
   3 runs each, back-to-back same window. Parity: relL2 vs fp64 reference <= 1e-2 gate.
5. PRE-REGISTERED VERDICT (do not adjust after numbers): >=1.4x V0 mean = PASS -> write the
   integration spec for the V-arm desk pattern (env arm, paired serve-leg) and STOP — the
   coordinator owns integration. 1.2-1.4x = AMBER -> bank + report. <1.2x = KILL -> bank,
   the class closes for good.

## Laws
- Work order doc = THIS file; append progress after EVERY step under "## PROGRESS LOG" (add
  the section header yourself, newest-first).
- Serving window: NOT NEEDED. If you somehow need one, tag "BLOCKER:" in this file and
  continue with cell work — the coordinator resolves windows, boots, and faults. NEVER
  debug boot faults yourself; NEVER pkill anything.
- Disk: 13 GB free; your cell + logs are <500 MB. df -h / if you grow anything >1 GB.
- Numbers carry the clocks note (read /sys/class/drm/card*/device/pp_dpm_sclk) or they
  don't count. Die 2/3 only.

## PROGRESS LOG

### PLOG-2 (2026-09-19, desk agent) — step 3 DONE: CENSUS KILL-SWITCH FIRED — KILL, NO BUILD, NO WINDOW
The work order's step 3 bar is NOT cleared; steps 4/5 are superseded (the pre-registered
census branch closes the family without a window — no bench ran, no GPU was touched, no
binary was linked). Full record: results/amd/coherence/WO_LEANMAC_census_row.txt (+4 census
receipt files beside it). Same-compile w3_census.py facts (AMD clang 18 roc-6.2.0, .s from
--save-temps): V0 <80,128> AND <96,128> = 3143 instr, 128 VGPR, 80 B scratch, 2 CTAs/CU,
**64 v_fma + 9 v_mul per (g,v)** — reproduces the banked madmix-row census exactly. The 8
pw multiplicands are visibly pre-computed registers feeding the FMAs: LLVM's LICM already
hoisted the coeff·e2m1_pair product out of the token loop — V0's shipped binary ALREADY
ISSUES LEANMAC'S TARGET FORM (§5: "8 v_mul + 64 v_fma"). Real ledger ≈ 97 ops/64 MACs
(~1.52 ops/MAC) vs the census model's 152 (2.375): the banked 152/96 = 1.583x prediction
double-counted a gain already inside V0's measured 1.989-2.444 TF/s. The restructured arm
itself trips its own resourcing bar: 256 VGPR (full cap) + 884 B scratch + 611 waits vs
V0's 128/80/212 — RED before any window (madmix FLUSH/wpair death class). Branch quoted
from §5 step 0: "~8 v_mul (already hoisted) ⇒ NO BUILD — the census model was wrong about
the wall ⇒ post-mortem desk, the fp32-restructure family closes without a window." The two
numbers the doc could not settle are settled: LLVM already hoists (yes — v_bfe landed too,
33/body), and the restructure as written buys nothing and costs a resourcing class.
VERDICT: KILL (census class) — fp32 op-count-reduction GEMM family CLOSED; GEMM front stays
at the pk16/madmix measured floor; V0's 0.28-0.34-of-model gap is now attributed to the
LDS-port/barrier/wait floor (the R1 post-mortem the doc reserved).

### PLOG-1 (2026-09-19, desk agent) — step 2 DONE: bench TU written + census compile green
tools/v340l/wo_leanmac_bench.cu written (standalone hipcc cell, w7_pk16/madmix harness
family verbatim: REF arm + V0 via the shipped launcher + bench-local LEANMAC arm = V0's
kernel with the §3 pw0/pw1 hoist as the ONLY delta; 4 standard geometries at M=128; 3
back-to-back runs per arm per geometry, each 3 warmup + 20 iters; parity columns = relL2
vs REF arm 1e-2, relL2 vs true fp64 host subset reference (rows [0,256), codec-header
arithmetic) 1e-2, byte/FNV identity vs V0 expected 0.00e+00; 5 s mclk-ramp hammer + copy
ceiling; sysfs pp_dpm_sclk AND pp_dpm_mclk printed pre+post; banked V0 anchors
3157.5/2830.1/3166.4/2396.1 GF/s with the ~5% harness-OK law). Compile (worktree root):
`/opt/rocm/lib/llvm/bin/clang++ -O3 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1
-D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -DNDEBUG -DUSE_PROF_API=1 -std=gnu++20
--offload-arch=gfx900 -x hip -I src/common/hip_shim -I include -I src -c
tools/v340l/wo_leanmac_bench.cu -o /tmp/leanmac_census/wo_leanmac_bench.o --save-temps=obj`
— rc=0 in 58 s, only family-class nodiscard warnings. GPU preflight: GPU[2]/GPU[3] at 0%
use (a foreign ninfer-serve holds VRAM on 0/1 at 100% — untouched; die 2 was the pinned
target). df: 13 GB free (matches work order). The binary was never linked: the census
(PLOG-2) killed the task at its gate.

### PLOG-0 (2026-09-19, desk agent, branch amd/wo-w7-body) — step 1 DONE: design doc read
Read docs/amd/GEMM_LEANMAC_2026-09-18.md fully. Contract absorbed: LEANMAC = register-level
CSE of the e2m1·coeff product — `pw = pr[r] * coeff[r][g]` hoisted out of the 8-token loop
(8 v_mul per (g,v) vs V0's 64), inner loop pure v_fma_f32; census 152→96 ops/64 MACs
(2.375→1.50 ops/MAC); bit-exact BY CONSTRUCTION (gate: byte/FNV identity vs V0 expected 0.00e+00,
plus this work order's relL2 ≤ 1e-2). Census bar (§5 step 0, KILL-SWITCH before bench):
V0 body must show ~64 v_mul_f32 (if ~8 = LLVM already hoisted ⇒ NO BUILD, family closes without
a window); LEANMAC body must show 8 v_mul + 64 v_fma, 0 spills, VGPR ≤ V0+~10; ANY scratch/spill
op in the loop body = RED. Prediction being tested: 3.15-3.87 TF/s vs V0 1.989-2.444 (xV0 ~1.58x
central). Verdict gates are the WORK ORDER's (pre-registered): mean xV0 ≥1.4 PASS | 1.2-1.4 AMBER |
<1.2 KILL. Plan: standalone hipcc TU `tools/v340l/wo_leanmac_bench.cu` (harness family =
w7_pk16_bench.cu: REF arm + V0 + 4 standard geometries at M=128, 5 s mclk-ramp hammer,
hipEvent 3 warmup + 20 iters, 3 back-to-back runs per arm per geometry), compile
`/opt/rocm/lib/llvm/bin/clang++ --offload-arch=gfx900 -O3 -I src/common/hip_shim -I include -I src
-std=gnu++20`, HIP_VISIBLE_DEVICES=2 (die 3 fallback), clocks note from
/sys/class/drm/card*/device/pp_dpm_sclk carried on every row.
