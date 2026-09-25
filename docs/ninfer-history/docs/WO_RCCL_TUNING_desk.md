# WO_RCCL_TUNING_desk.md — RCCL env-matrix desk for the #2 prefill wall item (TRUE AR ~123 ms/chunk)

Created 2026-09-19 ~01:20Z by the night coordinator (inline). Receipts this desk builds on:
PLOG-062 (honest in-serve ar column, R6-verified vs rocprof ground truth), PLOG-063 (AR is
structural: 0% overlap, serialized by construction, die spread 101-141 ms/1 s-window,
mean ~956 µs/call, max 11.1 ms straggler), PLOG-060 (ordinal-paired A/B design).

## Why this desk is cheap and real

The AR reduction levers that need engineering (fuse mixer+mlp collectives, quantized AR,
dedicated AR stream + graph rework) are parked as next-window DECISIONS. But one lever class
is ZERO-CODE: RCCL/NCCL environment tuning. The ar column is now measurable in-serve per
chunk on the canonical bin (OPTRACE=3), so an env matrix can be run as boots + probes only.

## Pre-registered design (do not change after first numbers)

- BASELINE: canonical runbook env (no RCCL overrides), ordinal-paired probes — fresh boot,
  auto warmup, 3x 2k probes (plen-2075 mt64 temp0), record [PREFILL-SUM] rank=0 ar column
  per ordinal + wall. PLOG-062 class data: ar=123.0/123.3, wall=991.6/978.2 ms/chunk.
- MATRIX (one boot each, same ordinal design): NCCL_ALGO in {Ring, Tree};
  NCCL_PROTO in {LL, LL128, Simple}; NCCL_MIN_NCHANNELS in {4, 8, 16};
  NCCL_NVLS_ENABLE=0 (Vega: confirm it doesn't mis-set something); NCCL_DEBUG=WARN boot
  once to bank the actually-selected algo/proto/channels (RCCL on gfx900 may ignore knobs —
  that NULL RESULT BANKS TOO: "RCCL env unresponsive on ROCm 6.2/gfx900" would close the
  class as cheaply as a win would).
- GATE (pre-registered): promote any single-env combo only if ar column improves >=10%
  vs baseline WITHIN-ORDINAL (±2% band respected, PLOG-060 design) AND wall improves >=3%.
  Straggler watch: max nccl kernel duration per chunk from the ar column spread.
- LAWS: ordinal-paired design mandatory (PLOG-060); boots only via the anchored retire +
  runbook restore; every perf number carries the clocks note; health-verify at every
  window close; no builds (zero-code desk); disk checked before any log pile-up.
- Effort: ~10 boots x ~4 min = ~40 min + parse. Fits one desk or one long inline session.

## If the matrix is a null result

Bank "RCCL env unresponsive on ROCm 6.2/gfx900 at 1.5 MB bf16 messages" with the boot
matrix table — that closes the cheap class and leaves the AR item to the engineering
options (fuse/quantize/stream-redesign), each of which is a REAL desk with this row as its
premise receipt: 123 ms/chunk = 12% of the 990 ms chunk wall.

## Downstream context

- V-arm integration desk (REV2 §8 SCHEDULED row) may contend for the serving window —
  sequence the two desks, do not interleave boots.
- The 4th soak grade (band datapoint) and day-end synthesis are independent of this desk.
