# WORK ORDER: V-arm integration RE-ENTRY design (reopen key SATISFIED — "+5-8% best case, top remaining prefill software lever", REV2 §9b(a))

Owner: desk agent (slot B). GPU-FREE design desk: build the NEW-MECHANISM design + the
pre-registered serving gate from banked data. NO serving, NO boots. The follow-up
implementation+serving leg needs a separate coordinator grant after your design lands.

## WHY THE RE-ENTRY KEY IS SATISFIED (banked; cite, do not re-derive)

- V-arm integration was TRIED and measured +5.7/+6.6% SLOWER in serving (parity GREEN;
  CLOSED SET row, results/amd/coherence/). The kill was real; the re-entry key was:
  "NEW mechanism (mad-mix census learnings)".
- The mad-mix census learnings now EXIST: PLOG-073 (redemption: attempt-1 GREEN
  1.119x/1.104x cell, relL2 1.65e-3, 122 VGPR no spill; the 0.48x kill was the
  inline-asm DELIVERY — 256-VGPR spill — not the ISA) and PLOG-080 (promoted +8.83%).
  The delivery-class lesson (plain C++ v_mad_mix_f32 + `-fgpu-flush-denormals-to-zero
  -ffp-contract=off`, clang 18) is a NEW mechanism the V-arm desk never had.

## PHASE 1 — AUTOPSY (what exactly lost the 5.7/6.6%)

1. Locate the V-arm evidence (CLOSED SET receipt; the V-arm desk's WO/rows in
   results/amd/**, `git log --all --oneline -- '*varm*' '*v_arm*' '*varm*'`).
2. Decompose the loss: which column grew (GEMM? body? AR? extra passes?). Was the loss
   (a) algorithmic (extra dequant/re-quant passes), (b) resourcing (VGPR/scratch — the
   inline-asm death class), or (c) scheduling? Cite the banked per-column numbers.
3. Write the autopsy into this WO's log — a named mechanism per lost percent.

## PHASE 2 — THE NEW-MECHANISM DESIGN (the deliverable)

1. Redesign the V-arm integration under mad-mix's delivery rules: which V-arm passes fold
   INTO the mad-mix GEMM epilogue/prologue vs run as separate launches; expected VGPR
   budget (<=128, 0 scratch — the PIPE kill constraint); expected column deltas.
2. Price it honestly from the banked columns: predicted serving wall delta (the +5-8%
   band was pre-mad-mix — re-derive against the POST-mad-mix chunk census; the GEMM is
   smaller now, so a body-side win is worth proportionally MORE).
3. Pre-register the serving gate (PLOG-060 ordinal pairs, same skeleton as
   wo_madmix_gmm2_pairs.sh): promote bar, kill conditions, rollback (the V-arm env gate
   default-off byte-identical — kill-switch-first law).
4. Deliverable: `results/amd/varm/VARM_REENTRY_DESIGN.md` — autopsy + design + priced
   prediction + gate. If the honest price is NEGATIVE (the mechanism cannot pay under
   the new delivery rules), say so and close the family with a decisive-reasoning row
   citing this receipt chain — a negative result here is a real result.

## LAWS

- Own worktree + branch amd/wo-varm; progress log newest-first in THIS file after every
  step; resumable from this file alone. NO channel posts. NO builds of any kind (this is
  a reading+design desk; the only tool allowed is git/grep/read + a text editor).
- Numbers cite banked receipts or do not exist.

## PROGRESS LOG (newest first)

**2026-09-20 step 3-5 COMPLETE (desk varm, slot B): DELIVERABLE LANDED — family CLOSED PRICED-NEGATIVE; desk DONE**

- **Deliverable: results/amd/varm/VARM_REENTRY_DESIGN.md** (autopsy + fold design + priced
  band + pre-registered gate + decisive negative closure). Committed on amd/wo-varm.
- **Verdict:** the V-arm family closes PRICED-NEGATIVE under mad-mix delivery rules.
  (a) Arm as integrated: structurally dead (serving conviction W7_varm_ab_row + §1 autopsy:
  N1 byte-priced +2..5 ms/chunk code-plane re-stream at TN=64; N2 latency-hiding deletion
  dominant ~60 ms/chunk class — the M1 insight's serving-leg confirmation). (b) The only
  surviving mechanism (v_perm decode atom, bit-exact per w7_vperm_probe) is designed in full
  as VARM-FOLD into the PROMOTED kernel_mm at V0's exact tile — bit-identical to mm by
  construction, <=136 VGPR, mutual-exclusion contract kept — and priced honestly at
  **-0.5..+1.0% serving wall (expected ~0)** vs the >=1.5% promote bar, on the attempt-3
  dose-response precedent (1.007x cell at -15.7% stream) and the V leg's own conviction of
  the deletion class. The REV2 §8 "+5-8% best case" band is not diluted post-mad-mix — it
  INVERTED (re-serving the arm post-mm would read ~-7.8%).
- **Pre-registered gate (banked, cell-first, self-killing):** G-V1 census (zero GPU,
  <=136 VGPR / body <= 2859 / signatures byte-match) -> G-V2 cell (one die ~30 min, bar
  >=1.05x mean + relL2 0.0 vs mm — deliberately set at the mm-class magnitude, above the
  priced expectation; FAIL = final close) -> G-V3 serving pairs only on a cell pass (exact
  wo_madmix_gmm2_pairs.sh skeleton, alternating ordinals, ±2% within-pair, promote >=1.5%,
  OPTRACE-3 instrument health precondition per PLOG-071, two-layer rollback
  NINFER_VPERM_FOLD=0 / BIN cc122cc0). Boots remain coordinator word only.
- **SD-1 consequence:** the last banked GEMM-column software lever is spent (joins pk16 /
  mad-mix / repack-P / consol / co-residency / LEANMAC / compiler-road in the measured-kill
  ledger). Remaining prefill mass = AR column (fp8-on-ring design, projected -5..-7.5%
  wall) + hardware-class exits. NVFP4-at-TP4 path unaffected.
- Dead-desk resumability: this file + the deliverable + commit chain below carry everything;
  no unbanked state. NO builds/boots/GPU were used (reading, git, grep, editor only).

**2026-09-20 step 1-2 COMPLETE (desk varm, slot B): EVIDENCE GATHERED + AUTOPSY LANDED (design section next; deliverable = results/amd/varm/VARM_REENTRY_DESIGN.md)**

- Evidence read (all paths verified in-worktree or via `git show amd/main:...`):
  results/amd/coherence/W7_varm_ab_row.txt (NO-GO close: +5.7/+6.6% 2k, 10k wash -1.4%,
  parity 1.285e-03 GREEN / falsifier 4.681e-01 RED, PREFILL-SUM unavailable-with-receipts);
  results/amd/coherence/W7_repack_row.txt (V AMBER 1.136x mean, census 113 VGPR / 0 scratch /
  32 waits vs V0 212, v_perm 16/body, TN=64 grid-doubling note, P KILL 0.983x, PV dominated);
  docs/amd/W7_VARM_INTEGRATION_desk.md (window log: interleaved B2 control reproduction,
  route proof, A' collapse quarantine); docs/amd/W7_REPACK_RESUME.md (V3c co-residency
  FALSIFIED: 71 VGPR / 3 blocks/CU = 1.023x only — the V win is ISSUE-side; both latent-bug
  carriage laws); docs/amd/PERF_LOG_AMD.md PLOG-056/060/067/071/073/080 (+line 512: re-entry
  key RESOLVED — measurements never dirty, serving loss REAL); docs/amd/PREFILL_PLAN_REV2
  §1/§8/§9/§9b (chunk census: gemm 756-806 of wall 978-992 ms = 76%, AR 12%, pure-consume
  0.85-0.95x at 36-51 GB/s, M1 latency insight, the +5-8% band provenance); G-MM-2 receipts
  (PLOG-080: MM 41.73s vs V0 45.77s medians n=9/arm = +8.83%; WO_MADMIX_REATTEMPT.md
  integration log: mm census 2859/122/80B, v_mad_mix=1024, V0 flag-invariant; pairs script
  tools/v340l/wo_madmix_gmm2_pairs.sh; row file itself died with the removed madmix-int
  worktree — PLOG-080 is the receipt of record).
- SOURCE AUTOPSY (src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu + nvfp4_dispatch.cpp, this
  branch carries the arm): the V arm is a SINGLE-KERNEL variant — there is NO separate
  decode pass and NO extra launches (route = one launch per a16 GEMM, same 192/chunk as
  V0/mm; the WO-brief premise "eliminates the separate decode pass + its launches" is
  FALSIFIED by source). The convicted package is: 2-D grid (blockIdx.y = M/TN=64 token
  tiles -> 2x CTAs: 448 vs 224 at AttnInput n=14336), fp16-window __hfma2 MAC (s2 windows,
  W=32, declared association class), LDS pair-table + scale-table DELETED (waits 212->32,
  body 2994->1359), TN=64. mm (promoted) changed ONLY the mul-stream arithmetic at V0's
  exact tile/grid/launches and kept every load — and paid +8.83% serving.
- AUTOPSY VERDICT (full decomposition in the deliverable): the +70 ms/chunk clean delta is
  NOT the A7 gap mode (PLOG-071: extraction artifact, walls +-0.2%), NOT box drift (control
  B2 band +-1.5% bracketing A), NOT parity/greedy drift (BLUEPASS 12/12; completion drift
  is ~0.2-0.5% class), NOT launch count, NOT the route gate. Named mechanisms:
  N1 = code/scale-plane re-stream doubling at TN=64 (y-pair CTAs 1.75 waves apart, 4 MB L2
  vs ~189 MB/chunk/die planes -> byte-priced +2..5 ms/chunk = ~5-8% of the delta); N2 =
  latency-hiding material deletion (the M1 insight's serving-leg confirmation: the deleted
  212->32-wait / 1359-instr stream was the scheduler's stall fill at cold-L2; dominant
  residual ~60 ms/chunk class; receipts: M1 census, V3c falsification, the V serving leg
  itself). Instrument honesty: PREFILL-SUM was DOWN both bins (banked receipts) — the +70
  is WALL-derived; per-column attribution is mechanism inference, flagged as such.
- Cell budget for a fold into kernel_mm: mm 2859 instr / 122 VGPR; fold deletes 128
  ds_read/body + ~90 of the 164-op extraction class, adds ~64 perm + ~50 assemble ops ->
  body ~2760 (-3.5%); dose-response precedent (attempt-3: -15.7% stream = 1.007x cell)
  prices the fold cell at ~1.00-1.02x -> below any bar worth a serving window. HONEST
  PRICE: -0.5..+1.0% serving wall, expected ~0 -> the family closes PRICED-NEGATIVE under
  the delivery rules; deliverable will carry the full design + pre-registered cell-first
  gate + the decisive negative closure per the honesty clause.
