# WORK ORDER: PATH A MEASUREMENT MATRIX — chunk {128,256,512} x capture {off,on} (PA-1 + PA-3 fused)

Owner: COORDINATOR inline (this is the principal's velocity order — the coordinator executes).
Window: shared with the oneshot root-look desk — the matrix fires in the gap AFTER the
root-look's STEP 2 repro phase ends and BEFORE its cure-validation boots. Root-look desk was
notified of the schedule.

## Pre-registered gates (PLOG-060 laws; ±2% within-pair; clocks noted per config)
- G-LADDER: chunk 256 or 512 delivers >=5% wall-per-token improvement vs 128 on stable boots
  (3 ordinal probes each, position-matched). Basis: llama.cpp runs M=512-2048 single GEMMs;
  our M=256 skip was TP-contaminated (per-chunk AR doubled; the cure config changes the math).
- G-GRAPH: NINFER_PREFILL_GRAPH=1 improves or neutral (within ±2%) wall at ANY chunk size,
  with parity BLUE/stop + no mojibake. Basis: the old "not launch-bound" kill predates the
  stall cure + AR structure changes; retry with pre-registered bars.
- COMPOSITE: best (chunk, capture) config vs the 128/off baseline — target trajectory to the
  >=150 t/s @2k program bar (with AR cure to follow as the second stage).

## Matrix (6 configs x [fast boot + 3x plen-2075 mt64 probes])
  1. chunk=128 graph=off  (baseline re-anchor)
  2. chunk=128 graph=on
  3. chunk=256 graph=off
  4. chunk=256 graph=on
  5. chunk=512 graph=off
  6. chunk=512 graph=on
Each config: pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"; boot canonical 2c8901d3
via inline fast-boot (NVMe ART, chunk=N, NINFER_PREFILL_GRAPH per arm); health wait; 3
probes; record 3 walls + parity; append to the row file. ~2.5 min per config, ~15 min total.
NOTE: a wedging config (expect none — canonical posture is 4/4 healthy) = mark SKIP and move
on; two consecutive boot faults on ANY config = halt matrix, restore canonical, report.

## After the matrix
- Restore canonical 128/off (runbook posture) + health-verify.
- Bank results/amd/coherence/WO_PATHA_matrix_row.txt + the 6 boot logs.
- Winning config + graph verdict feed the promotion leg: if (chunk,capture) config holds
  >=5% over 128/off across a re-run bracket, flip the runbook defaults (prefill-chunk +
  NINFER_PREFILL_GRAPH=1), BOOT_BATTERY, 10k grade, PLOG row — the FIRST Path A promotion.

## PROGRESS LOG
- [P-0] 2026-09-19 ~12:00 coordinator: work order written; matrix script ready; waiting on
  root-look STEP 2 repro to end. Root-look notified of the schedule.
