# W7 K-SPLIT DESK — RESUME NOTE (checkpoint law; update at every milestone)

Desk: G1 K-split (cross-CTA K-split on V0 nvfp4 tiled GEMM), Team Red lane `amd/wo-w7-body`.
Bench source: `tools/v340l/w7_ksplit_bench.cu` · binary: `bin/w7_ksplit_bench`
(hipcc -O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx900 -Iinclude -Isrc -Isrc/common/hip_shim
tools/v340l/w7_ksplit_bench.cu src/core/device.cu -o bin/w7_ksplit_bench).
Row (being filled): `results/amd/coherence/W7_ksplit_row.txt`.
GPU: device 0 only; serving instance :8100 UNTOUCHED; no server boots.

## Milestone 1 — SOURCE + STEP-0 CENSUS (DONE, this commit)

- Compile law line found: repo hip_shim (`-Isrc/common/hip_shim`) + `-Iinclude` needed.
- STEP-0 CENSUS (w3_census.py on hipcc -S of the bench TU, /tmp/w7_ksplit_census.s):
  - V0 anchor <80,128>: **3143 instr / 128 VGPR / 80 B private** — byte-exact reproduction of
    the banked W7_v0_census/W7_madmix-row census. Compiler agrees the TU is V0's.
  - ksplit_a <80,128> and <96,128> (V0's own launch bounds): **3018 instr / 125 VGPR / 80 B** —
    BELOW V0's 128; same scratch class; 2 blocks/CU preserved. THE SPLIT KERNEL IS
    RESOURING-CLEAN (kill-switch NOT tripped). MAC stream identical (1024 fma).
  - ksplit_b (forced minBlocks=2): 3000 instr / 122 VGPR / 80 B — equivalent class, benched
    arm is **a** (least-modified posture).
  - pureconsume: free-scheduling hoarded to 256 VGPR+276 B (xor soup pipelines); PINNED to
    __launch_bounds__(256,2) → **128 VGPR / 792 B** — V0 occupancy posture; 792 B spill is
    ablation-arm-only (madmix-row benched scratch class), noted in row.
  - reduce kernel: 54 instr / 6 VGPR / 0 B — trivial.
- S is a RUNTIME grid.z param: ONE instantiation covers S∈{2,4,8} (census is S-invariant by
  construction — loop bounds computed in prologue).

## Milestone 2 — PRE-REGISTRATION (committed BEFORE any bench run)

In `results/amd/coherence/W7_ksplit_row.txt` header: PRIMARY = mean(best passing S per
geometry) xV0 ≥1.20 PASS | 1.08–1.20 AMBER | <1.08 KILL; SECONDARY fixed-S4 mean + per-geometry
pattern; relL2(KS,REF)<1e-2 every row; harness = V0 same-run vs banked-today
(3041.4/2238.5/2330.5/2100.8, W7_padded_swx_row AUTO clocks) within ~5%; limiter-kit
KILL-at-step-0.5 condition = k-scaling ratio <1.15 (flat) AND consume <0.5x V0 time.

## Milestone 3 — BENCH + VERDICT (DONE — MEASURED KILL, banked)

- limkit: W7_ksplit_limkit_run1.log (consume shape-1 DEFECT: 792 B spill, 2.5x over V0 — named)
  + run2.log (valid). KIT: (a) consume = V0 sits AT its access-pattern wall (loads alone 0.85-0.95x
  V0 time at HALF occupancy); (b) k-scaling LINEAR 1.97-2.00x (per-tile cost constant); (c) T-scaling
  NOT flat (1834->3135->2230 GF/s at 32/128/256 on Attn); (d) cited closed. KILL-at-0.5 condition
  NOT met (needed flat k-scaling) -> sweep proceeded.
- sweep: W7_ksplit_bench_run1.log. ALL splits LOSE, monotone in S: best-S/V0 = 0.88/0.90/0.88/0.98
  (mean 0.91x; S=4 mean 0.78x). relL2 0.00e+00 everywhere (constant-data caveat + gate-discrimination
  note in row). VERDICT: KILL (<1.08 floor).
- MECHANISM (row "WHY"): V0's 128-VGPR/2-blocks-CU resourcing means k-split cannot add co-resident
  CTAs (unlike Team Green's 1-CTA/SM wide arm) — extra CTAs queue; prologue amortizes worse; reduce
  adds traffic. Named limiter: per-tile access-pattern latency at fixed co-residency.
- RE-ENTRY MAP (REV2 §8): NOTHING fires (pk16 / mad-mix / graph-capture / M=256 ladder / padded-swx
  all conditioned on "K-split lands" or "GEMM shrinks" — neither happened). Conditional-positive
  recorded: a <=96-VGPR (3 blocks/CU) resourcing future or gfx906-class part re-prices this lever
  and those bars.
- Known print defects (source fixed post-run, logs stand as banked): %%nom column 1000x in the sweep
  log (corrected table in row); consume shapes 1-2 defect history in row + resume.
- CLOSE-OUT: clean /tmp census artifacts; final commit+push.

## Laws this desk runs under

Reproduce-first (V0 within ~5% before any arm counts) · same-window same-binary A/B ·
VRAM law (partials slab = bench-side cudaMalloc by shape math, no refusal constants) ·
no pkill · clocks AUTO today (band noted via sysfs print) · do NOT merge to amd/main ·
no PERF_LOG append · do not touch tools/v340l/w7_repack_bench.cu (repack desk's, untracked).
