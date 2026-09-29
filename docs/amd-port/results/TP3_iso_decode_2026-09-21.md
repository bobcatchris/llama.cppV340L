# TP3 iso decode cell - draft-device isolation re-verification at -c 10000

Date: 2026-09-21, 23:44-23:53 CDT. Closes the E-037/E-039 contaminated
decode cell on the amd/draft-isolation build, per the E-038 coordinator
request (client exclusivity, distinct port).

- Boot of record (PORT=8081, lane-only client; dies 0-2 serve, die 3 drafts):
  HIP_VISIBLE_DEVICES=0,1,2,3 llama-server -m Qwen3.8-27B-ASCII-P1M.gguf
  --device ROCm0,ROCm1,ROCm2 -ngl 999 -sm tensor -c 10000 -b 512 -ub 512
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp --spec-mtp-device ROCm3
  --port 8081 -t 8
  NAMING CORRECTION of record: the W6 protocol / dispatch line wrote
  --device CUDA0,CUDA1,CUDA2 --spec-mtp-device CUDA3; on this HIP build those
  names do not resolve ("error while handling argument --device: invalid
  device: CUDA0" - boot stub server_tp3_200k_20260921_234338.log). Actual
  ggml device names under HIP_VISIBLE_DEVICES=0,1,2,3 are ROCm0..ROCm3
  (--list-devices). --device still MUST precede --spec-mtp-device (W6 risk 5).
  Isolation boot lines present: "MTP draft device: ROCm3" + "MTP draft
  context runs on ROCm3 (nextn weights + KV pinned to the draft device)".
- Battery: run_tp3_guards.sh, fingerprint aa336055d4d73b00, PORT=8081.
  No other process targeted 8081 during the window; the E-043 contamination
  vector was the TP3 lane harness on 8080, which this lane did not run.

## Cell of record (attempt 1, 23:44-23:52)

| guard        | status | measured                              |
|--------------|--------|---------------------------------------|
| prefill_2k   | PASS   | 116.95 t/s (+1.51% vs 115.21)         |
| decode_8k_10k| PASS   | 15.21 t/s (+1.84% vs 14.94)           |
| mtp_canary   | PASS   | accept 0.66667 (log: 84/126, mean 3.00)|
| determinism  | PASS   | sha 4beb1ba25219ee9b byte-identical   |
| needle_8k    | PASS   | 3/3 depths exact                      |

OVERALL PASS. Retry forensics on server_tp3_200k_20260921_234422.log:
0 "Context size has been exceeded" lines, 0 retry/slot lines - the E-037
defect signature is absent on the iso build under a serial battery.

## Contamination audit (honest, E-043 convention)

- A foreign bench process (bench_tile_chun, the wt-tile-chunked desk's die-3
  bench) was resident on die 3 DURING the window: allocations visible during
  my 180s post-boot cooldown (card3 562.8 -> 742.5 -> 906.6 MiB at 23:45:33 /
  23:46:35, while this server served no requests), non-zero card3 util
  (samples up to 100%) including the 23:48-23:49 decode cell, and still alive
  at my teardown (KFD: 363.7 MiB). It held the campaign boot lock in no way;
  benches run between boots and are not lock-gated.
- Client exclusivity HELD: port 8081 lane-only, zero foreign HTTP (retry /
  context-exceeded lines absent; receipt task set = exactly the 5 guards).
- Therefore: the retry, accept, determinism and needle cells are
  contention-immune and CLOSED - E-037/E-039 contamination does not reproduce;
  the iso build's decode cell passes. Decode t/s 15.21 was measured with die-3
  compute shared, i.e. a lower-bound-flavored number - it already matches the
  canonical same-evening 200k MTP-ON decode (15.12) and exceeds the expected
  14.4-14.9 class, so contention would only push the clean number higher.
- Die-3 VRAM residency NOT cleanly attributable: my draft residency at boot
  is bounded 347.6-551.1 MiB over idle (foreign was already ramping during
  model load); consistent with the W6 estimate (169.3 MiB blk.64 weights +
  ~11 MiB 10k q4_0 draft KV + 100-300 MiB buffers). Serving-level card3 numbers
  are foreign-dominated and not banked.
- Attempt 2 (23:55:50): ABORTED pre-battery per the die-3 discipline - a new
  foreign bench (PID 873236) was already resident on die 3 before boot
  (card3 397-453 MiB pre-boot); server torn down, no numbers banked. The
  abort's artifact receipt tp3_guards_20260921_235710.jsonl (5/5 FAIL, no
  measured values) is the battery firing into the already-killed server after
  its 180s cooldown elapsed; it documents the abort, it is not a measurement.

## WINDOW REQUEST to the coordinator

One clean, exclusive die-3 window (~10 min: boot + 5-guard battery) to
optionally re-bank the iso decode t/s without bench contention. Everything
else in this cell is closed by attempt 1. Requested by the served-measurement
lane (E-047); dies 0-2 are unaffected and were clean throughout.

## Receipts

- tp3_guards_20260921_234422.jsonl; server_tp3_200k_20260921_234422.log;
  server_tp3_200k_20260921_234338.log (CUDA-name failure stub);
  guards_run_iso10k_dev3.txt; mtpvram_iso10k_dev3_vram.csv;
  vram_iso10k_dev3_{preboot,bootready,postbattery,postteardown}.json.
- Attempt-2 abort trail: guards_run_iso10k_dev3_v2.txt;
  mtpvram_iso10k_dev3_v2_vram.csv; vram_iso10k_dev3_v2_*.json;
  server_tp3_200k_20260921_235710.log; tp3_guards_20260921_235710.jsonl.
- Ledger: E-047 in OPTIMIZATION_PLAN_TP3_200K.md.
