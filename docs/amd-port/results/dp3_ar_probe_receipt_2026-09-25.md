
# DP-3 PROBE VERDICT: 4-DIE TREE ALLREDUCE WORKS - 123.9us @ 80KB vs 210.8us SERVED - 2026-09-25 18:45

## Instrument: probes/dp_ar_probe.cu v5 (receipt dp3_ar_tree_v5_*.log)
4-rank tree AR built ONLY on hipMemcpyPeerAsync + local reduce kernels.
Tree matches fabric: L1 intra-card {1->0, 3->2} partial copies (parallel,
source streams) + reduce2 on ranks 0,2; L2 cross partial 2->0 (source
stream) + reduce2 on 0; L3 broadcast 0->{1,2,3} serialized on rank0's source
stream. All ordering via same-stream deps + cross-device stream-waits (work
fine). Explicit 12-direction EnablePeerAccess at init.

| size | wall med | min | verify |
|---|---|---|---|
| 4KB  |  75.8 us | 73.7 | PASS x311 |
| 20KB |  84.7 us | 83.7 | PASS (draft-class boundary) |
| 80KB | 123.9 us | 122.6 | PASS (verify-class boundary) |
| 1MB  | 922  us | 882  | PASS (prefill needs the chunked ring instead) |

## THE SERVED COMPARISON (80KB fp32 verify boundary)
- served RCCL ring wall: 128.8-210.8 us/boundary (W8; slowest die = the wall)
- this tree: 123.9 us INCLUDING host enqueue (~15-25us of the wall)
- delta: ~-87us/boundary x 128 verify boundaries ~ -11.1 ms of the ~124 ms
  cycle -> decode ~ +9-10% BEFORE the W39 skew rider (+2.4-4.8% rides).
- 20KB draft boundaries: 84.7us (6/cycle) - additional ms-class saving.
- 1MB class: serialized broadcast is bandwidth-poor; prefill (10.5MB bf16
  boundaries) needs the chunked reduce-scatter/allgather ring - DP-3
  integration work, NOT a blocker for the decode prize.

## TWO ROCm 6.2 BUGS BANKED (integration law, dp_ar_debug2 receipt)
1. CROSS-CARD hipMemcpyPeerAsync SILENTLY WRITES GARBAGE when enqueued on
   the DESTINATION device's stream. Source-device stream = byte-exact
   (matrix probe passed because it always used source streams; the first AR
   probe failed 2/3 broadcasts for exactly this reason - debug2 tests B/C/E
   repro, test A + intra-card D pass in both modes).
   -> INTEGRATION LAW: every peer copy in the meta-backend AR goes on the
      SOURCE rank's stream. (Matches the meta-backend per-die-stream
      structure anyway.)
2. Cross-card copies require EXPLICIT hipDeviceEnablePeerAccess; intra-card
   auto-enables (debug run without enablement: 0->2 garbage while 0->1
   fine; the matrix probe enabled explicitly per pair). Enablement persists
   per process/context.

## Remaining headroom (named, for integration)
- broadcast serialization: 3 x 22.5us on st[0]. Relay variant: 0->2 (st[0],
  cross) parallel with 0->1 (dest-stream OK intra), then 2->3 (intra) =
  depth ~40us vs 66us -> projected total ~98-100us. Chunked ring for
  bandwidth sizes. NOT blocking; 123.9us is the conservative banked floor.
- reduce kernels ~2-4us each; launch overhead ~15-25us of the wall
  (graph-integrated enqueue hides most of it).

## Bottom line
DP-3 is REAL and priced: a correct 4-die allreduce on the existing
substrate, ~41% under the served per-boundary wall at the verify size,
written against the bugs ROCm 6.2 actually has. Integration vehicle: the
meta-backend's boundary dispatch (replaces the RCCL call for decode-class
boundaries behind an env), then ONE served paired window per campaign law.
