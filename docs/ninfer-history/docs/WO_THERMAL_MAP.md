# WORK ORDER: THERMAL STATE-FUNCTION MAP — quantify the box's biggest unclaimed performance lever (operator-facing deliverable)

Owner: desk agent (slot B). GPU-FREE desk: banked data + PASSIVE rocm-smi sensor reads only.
NO serve boots, NO benchmarks, NOTHING that perturbs the running battery.

## PROGRESS LOG (newest first)

- **2026-09-20 step 3 — Phase 2 COMPLETE: leaflet delivered.** results/amd/thermalmap/
  THERMAL_MAP.md (one page): per-era response curve table; three quantified levers —
  (a) power-cap raise (110 W/die default verified via sysfs; +52-65%/10k if fast band held,
  magnitude UNPRICED, needs one privileged A/B cell, root-gated), (b) cooling/ambient
  (tau ~90 s => duty-cycle lever, ~3->1.5 min reset if airflow halves tau; no standalone
  tok/s), (c) burst-scheduling idle windows (RECOMMENDED ASK: +52-65%/interactive 10k,
  2.7x interactive decode, +19% on 60k; free, measured across eras); REV2 350-450
  reconciled as projection whose empirical basis is today's 1.5-1.9x multipliers.
  Single-boot rows labeled [1B] in both deliverables. Battery unperturbed (reads only).
  Commits: e7d3c775b (Phase 1), this commit (Phase 2).

- **2026-09-20 step 2 — Phase 1 mining COMPLETE.** Every banked source mined into one table:
  (A) pre-cure era bin 07ad7ecc (COOL10K 10k mean 11.3 plateau 24.5; THARM1 decode 13.79
  cold / 4.89 hot-no-gap / 13.28 hot+3-min-gap — same boot, edge 84C both = integrator, not
  edge); (B) cure era bin 2c8901d3 (PLOG-064 63.1/70.7/104.0; W7_therm_row integrator
  46->80C/~85s load, 90s drain tau, ~53C floor, sclk parks 300 instantly; pinned-vs-auto A/B
  auto 1.8x; serve_10k.log 5th point 82.4); (C) k4v4gates era bin f3312f25 (10k ordinals
  96.9/71.3/50.2 with clocks mclk 945->500->167, sclk 1269->991->560; k4v2 62.2/47.5/43.6;
  soak edge 80->84C); (D) LIVE kvarn battery bin 8ac1ba93 (anchor63150 mean 35.1 cold-boot
  start-91.4; d0.25 29.4; d0.5 28.9; floor 20.7 repeats; sidebands 84-85C; WO's 40.8/20.7/28.0
  reconciled = instantaneous trace warm-sample/park-floor/mid-decay, not leg means);
  (E) fuse microbench cold baseline 34/26/35C; chunk256 ladder 92.4-vs-80.1 class.
  Passive snapshot during battery (~06:37): edge 83-84C x4, junction 87-90, mem 85-91,
  power 62/113/67/96 W, sclk lvl 3-7, mclk 1-3, power1_cap=110W default x4 (sysfs hwmon).
  Deliverable A written: results/amd/thermalmap/MINED_THERMAL_TABLE.md.
- **2026-09-20 step 1 — survey + k4v4fin serve-log parse.** Live battery identified from ps
  (read-only): bin /home/chris/artifacts_bin/ninfer-serve_8ac1ba93eb7fbfad.bin, chunk-128,
  kvarn_k4v4, mc=65536, ws=512, MTP k=2, up 2h07m at 06:39. Per-request traces parsed from
  serve_boot_kvarn_k4v4_mc65536.log (req2 35.1 / req3 29.4 / req4 28.9 prefill means; req5 in
  flight). No writes outside worktree; zero GPU touches beyond rocm-smi reads.


## WHY (banked facts; cite, do not re-derive)

- PLOG-064: the 10k soak is a THERMAL STATE FUNCTION, not a band — hot start 63.1 /
  warm 70.7 / COLD START 104.0 tok/s prefill on the IDENTICAL cure config (1.66x spread).
  The integrator is quantified: 46->80 C edge in ~85 s of bursts; ~90 s drain constant;
  DVFS parks instantly; ~3 min idle resets to the fast band.
- REV2 scoreboard: "M=256 + thermal control -> ~350-450 tok/s class" — thermal CONTROL is
  half of the last projected multiplier, and it is OPERATOR-GATED (power cap / cooling /
  scheduling policy are principal+product decisions).
- Fresh same-boot evidence to mine: the k4v4fin battery's own arm A ran the SAME probe at
  40.8 tok/s (d0.25, warm) then 20.7 tok/s (d0.5 tail, hot) then 28.0 (d0.75) — the
  sideband files (results/amd/k4v4fin/sideband_*.txt) + serve logs hold hours of paired
  (temp, mclk, sclk, tok/s) samples across boots/arms/eras.

## PHASE 1 — MINE THE BANKED DATA (all GPU-free)

1. Collect every banked sideband + serve-log (results/amd/**: k4v4fin, k4v4gates,
   coherence G-MM-2 rows, fuse MEASURE_*.txt, PLOG-064 soak files). Parse into one
   (context, config, start-temp, edge-temp, mclk lvl, sclk lvl, tok/s) table.
2. Deliverable A — the RESPONSE CURVE: tok/s as a function of thermal state, controlled
   per config era (the state function is per-config; do not mix bins without saying so).
3. Deliverable B — the OPERATOR ASK, quantified three ways: (a) power-cap raise
   (rocm-smi show power/cap today; what a raise would hold in DVFS terms — READ-ONLY
   query), (b) cooling/ambient (the drain constant => sustained-vs-burst duty), (c) burst
   SCHEDULING policy (idle windows that reset the fast band; what wall-time fraction at
   ~104-tok/s class is achievable per duty cycle). Each with tok/s and the receipts.

## PHASE 2 — THE LEAFLET (the deliverable that reaches the principal)

`results/amd/thermalmap/THERMAL_MAP.md`: one page — the curve, the three quantified
levers, the recommended single ask (expected tok/s per action), honest caveats (which
numbers are single-boot, which are state-function-solid). This feeds a principal decision;
nothing here flips any serving config.

## LAWS

- Own worktree + branch amd/wo-thermalmap; progress log newest-first in THIS file after
  every step; resumable from this file alone. NO channel posts. NO builds. NO boots.
  rocm-smi sensor/power queries are PASSIVE and allowed; anything that could perturb the
  running battery is not. df -h / before >1G writes (this desk writes KBs).
- Numbers cite receipts or do not exist. Single-boot numbers are labeled single-boot.
