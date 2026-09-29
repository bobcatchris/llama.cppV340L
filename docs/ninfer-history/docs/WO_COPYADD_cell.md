# WO_COPYADD_cell — copy-add AR microbench desk (GATE-Q0, PA-2 first gate)

Desk: copy-add AR microbench (option 0 of the AR cure) — price the llama.cpp-style
copy-add partial sum against the MEASURED NCCL SHM ring at PREFILL message sizes, to
decide whether the AR cure needs the one-shot machinery at all (which currently WEDGES
at warmup, 6/7 boots — filed blocker, root-look desk in flight).

- Branch: `amd/wo-w7-body` (worktree `/home/chris/worktrees/amd-wo-w7-body`)
- Cell: `tools/v340l/w7_copyadd_cell.cu` (standalone hipcc, no serve, no RCCL needed for
  the gate arms; a best-effort RCCL world=4-on-2-devs anchor arm X runs last, guarded)
- Dies: **2,3 only** (`HIP_VISIBLE_DEVICES=2,3`) — dies 0,1 belong to the serving window
  (root-look desk cycling boots). No pkill, no :8100 contact, no serve processes started.
- Runs: standalone binary from /tmp; ~seconds per arm; no disk-heavy writes (df checked).

## Measured baseline (the bar — PLOG-062/063, W7_rccl_matrix_row.txt)

NCCL SHM ring allreduce, 4x gfx900, at 1.31-1.5 MB bf16 = **646-771 us/call**;
125 calls/chunk; ~99-123 ms/chunk total AR. Ring is env-immovable (RCCL env matrix =
NULL RESULT, class closed). One-shot alternative wedges at warmup (blocker).

## Geometry actually measured (stated up front)

4 ranks **simulated** via the real transport geometry — pinned host staging buffers, one
per rank, every inter-rank leg is a device copy-engine trip through host memory (the SHM
ring's shape, P2P disabled on this box). **Only 2 physical dies are legal for this desk
(law), so vrank 0,2 -> die 2 and vrank 1,3 -> die 3: 2 active host links, not the
serving ring's 4.** Consequence: any bandwidth-bound arm here pays ~2x the per-link
traffic of the real 4-link ring. Gates below are therefore pre-registered against the
MEASURED 646 us ring number, not against this cell's R arm. R doubles as a model anchor:
if R lands near (or modestly above, given the 2-link handicap) the 646-771 band, the
model is honest; the C-comparison stands against 646 either way. Arm X (RCCL 4 ranks on
2 devs, if RCCL permits duplicate-GPU ranks) measures the 2-link handicap directly.

## Message sizes

- 1.31 MB bf16 = 655360 elems (matches serving partial shape kMaxN)
- 1.50 MB bf16 = 750000 elems

## Arms

- **R** (baseline reference): 4-rank ring-pattern allreduce built from pinned copies —
  reduce-scatter (3 steps, S/4 per step, bf16 in-place add at receiver) + allgather
  (3 steps), per-rank tx/rx streams so D2H(send) overlaps H2D(recv) duplex within a
  step, host barrier per step. Approximates what the SHM ring pays (unpipelined at
  chunk granularity = conservative vs NCCL's pipelining; noted).
- **C** (copy-add one-shot, naive round as specified): 4 ranks concurrently D2H their
  bf16 partial into root pinned slots (4 copies) -> barrier -> 4 H2D into root device
  buffers -> root 4-way add kernel (fp32 accumulate, one bf16 round) -> barrier ->
  root D2H result -> barrier -> 3 broadcast H2D. Full round timed host-side.
- **C2** (copy-add, pipelined publish — family-best-effort): same as C but each slot's
  H2D is event-chained on that rank's D2H (no publish barrier); secondary row, not the
  gate arm. If C fails narrowly, C2 shows whether pipelining alone could save the family.
- **Q** (copy-add + fp8e4m3 quantized partials): same round as C; each rank computes
  per-call amax, quantizes partial to e4m3fn (scale = amax/448, RNE, saturating),
  copies S/2 bytes; root dequant-adds the 4 slots (fp32) -> bf16 result; broadcast.
  E4M3FN encode/decode implemented in-cell (shim has decode only; FNUZ aliasing is
  numerically wrong per shim header). Encode self-tested against golden vectors at
  startup (1.0->0x38, 1.25->0x3A, 448->0x7E, clamp 449->0x7E, subnormals, RNE ties,
  -3.0->0x44 FNUZ-discriminator); startup FAIL = cell refuses to run.
- **X** (anchor, best-effort): RCCL allreduce world=4 ranks on visible devs 0,1
  (duplicate-GPU ranks). If RCCL refuses, row records `anchor=skipped` and nothing else
  depends on it. Runs LAST under alarm(240) so a wedge cannot eat the other results.
- Host-add footnote: single-threaded host 4-way add timed once per size (expected
  disqualifying; measured, not assumed).

## Stats

Warm 8 + 50 timed reps, 2 adjacent reproduce runs per arm per size; mean/med/p5/p95.
Correctness verified on run 1 of each arm against fp64 ground truth of the same inputs
(R: rel bar 0.008 = 2 bf16 ulp; C: max abs vs R output recorded; Q: max abs vs R output
RECORDED and GATED). Random inputs: uniform(-2,2), per-rank LCG seeds, constant across
reps of a run. Clocks (rocm-smi --showclocks, dies 2,3) captured by the runner before,
between, and after arms; every number in the final row cites its clock state. Contingency
for the parallel root-look desk's boot cycles: if a run's p95 > 1.5x median (contention
signature), the run is repeated and both recorded; the reproduced adjacent pair decides.

## PRE-REGISTERED GATES (stated 2026-09-19, before any build or run — not adjusted after)

- **GATE-Q0 (C):** full-round C <= 0.55x ring reference at 1.31 MB, i.e. **C mean <= 355.3 us**
  (0.55 x 646) on the reproduced adjacent pair. The bar is the measured serving ring
  (646-771 us), deliberately NOT this cell's R arm (2-link handicap explained above).
- **GATE-Q1 (Q):** Q <= 1.15x C (per size) **AND** max abs error(Q vs R output) <= 1e-2
  on the uniform(-2,2) bf16-scale random data. Rough screen — the real numerics gate is
  the serve-leg's acceptance bar, not this cell.
- **NULL branch:** if C fails GATE-Q0 AND Q fails GATE-Q1: bank the NULL — the copy-add
  family CLOSES as priced (option 0 cannot beat the ring by 1.8x+ at prefill sizes on
  this transport), and the AR cure falls back to (a) one-shot wedge root-look and
  (b) fp8-on-ring alternatives. Partial passes are reported per-gate, family NOT closed.

## Projections to report

us/call per variant per size; wire GB/s (total counted PCIe payload bytes / round time);
AR busbw = 2*(N-1)/N * S / t (nccl-tests convention); projected chunk saving =
(ring_us - variant_us) x 125 with ring_us = 646 (favorable edge for the copy-add family —
stated) and 771 quoted as range end.

## File plan

- Cell: `tools/v340l/w7_copyadd_cell.cu` + runner `tools/v340l/w7_copyadd_run.sh`
- Binary: `/tmp/w7_copyadd_bin` (never built in the shared checkout)
- Row: `results/amd/coherence/WO_COPYADD_row.txt`
- This doc: checkpoint + PROGRESS LOG (appended after every step)

## PROGRESS LOG (newest last)

- 2026-09-19 desk opened: probe confirmed AMD line (4x V340, gfx900, ROCm 6.2, all dies
  idle at open). Worktree clean checkout of amd/wo-w7-body confirmed. df -h / = 12G free
  (cell writes < 1 MB). Baseline evidence re-read (PLOG-062/063, W7_rccl_matrix_row.txt,
  W7_ar_transport_bench.cu GATE-T pattern). Gates above pre-registered BEFORE build.
- 2026-09-19 cell + runner written: tools/v340l/w7_copyadd_cell.cu (arms R/C/C2/Q/X per
  design above; fp8 E4M3FN encode in-cell with 15-vector golden self-test incl. the
  0xC4 FNUZ discriminator; ring = RS3+AG3 with per-step send events for duplex overlap;
  R re-copies pristine input to dwork per round so timed reps don't compound). Runner
  tools/v340l/w7_copyadd_run.sh: build -> pre/mid/post rocm-smi clock+util snapshots ->
  two invocations (1.31 MB, 1.5 MB) -> row file. Next: build.
- 2026-09-19 build + first runs. FOUR cell bugs caught and fixed, each RED row -> GREEN
  row (cell-internal; noted here per the closure law's spirit):
  (1) fp8 encode rounding conflated guard/sticky (RED: selftest v=1.1875 got 0x39 want
      0x3A) -> proper guard-bit+sticky RNE (GREEN: full 17-vector table passes);
  (2) my own golden vector for 3.0 was wrong (0x40 -> 0x44, exp8/man4);
  (3) ROCm 6.2 context quirk: hipEventCreate binds to the CURRENT device's context —
      C2's events must be created under each recording stream's device (RED: cellfail
      site=hipEventRecord line=346 "invalid resource handle"; GREEN after per-device
      create). rk[r].ev in arm R never hit this because it is created right after
      hipSetDevice(rk[r].dev);
  (4) uint4/bf16x2 union pun must be h[4] not h[2] (RED: C3 verify maxabs=inf — upper
      8 B of each 16 B vector uninitialized; GREEN: C3 verify PASS, maxabs 0.029).
  Also: the verify bar for cancelling sums was relative-to-sum (RED: R verify maxrel
  7.8e6 with maxabs 0.039 on legitimate intermediate-ulp rounding) -> absolute bar
  scaled by per-element partial magnitude sum_r |x_r|, 0.02*(mag+0.25) (GREEN: R PASS
  big=0). Added arm C3 (zero-copy root add over mapped slots, NT 128-bit loads, 8S wire
  = the family's byte-optimal transport) per the decisive-measurement law: close the
  family on its BEST case, not the strawman.
- 2026-09-19 CONTENTION NOTE: during all runs the parallel root-look desk's serving
  window was LIVE on all four dies (rocm-smi pre: 4x GPU 99-100%, 84% VRAM, sclk 1350-
  1500; dies 0,1 at 84 C). Our dies 2,3 shared the box with the serve. Timed numbers are
  tight (p95/med <= 1.02, adjacent runs agree <= 1%), arms share conditions; the NULL
  verdict is additionally protected by byte-counting (see row). No serve process was
  touched; no pkill; no :8100 contact.
- 2026-09-19 arm X made non-silent (RCCL init refusal now prints a ROW before exiting;
  silent `_exit` in the init thread had hidden it), runner gained pipefail + rc echo.
  FINAL OFFICIAL RUN banked: results/amd/coherence/WO_COPYADD_row.txt,
  bin sha16 c3478fc7710febdb. During THIS run the serving window was DOWN (dies 0-3 at
  0% GPU) — clean conditions; numbers match the earlier serve-live runs within 0.3%,
  so the verdict is robust under both box states. Clock note for every number below:
  mclk pinned at max level (945 MHz), sclk idling at 300 MHz between phases (copy-engine
  work), PCIe 8.0 GT/s x8. Anchor arm: RCCL explicitly refuses duplicate-GPU ranks
  (nccl rc=5, ranks 1-3) -> the 4-ranks-on-2-devs anchor is UNAVAILABLE by RCCL policy,
  recorded, not worked around.

## RESULTS (final official run; med us, adjacent reproduce pair, both runs shown)

1.31 MB bf16 = 655360 elems (wire bytes: R/C/C2 = 12S = 15.7 MB; C3 = 8S; Q = 8S+32 B):

| arm | med us run1 | med us run2 | wire GB/s | AR busbw GB/s | chunk saving vs ring 646 (x125) |
|-----|------------|------------|-----------|---------------|--------------------------------|
| R  copy ring (model)        | 2452.0 | 2462.5 | 6.42/6.39 | 0.80 | -225.8 ms |
| C  copy-add naive (GATE)    | 2399.3 | 2396.9 | 6.56/6.56 | 0.82 | -219.1 ms |
| C2 copy-add event-chained   | 2408.3 | 2411.4 | 6.53/6.52 | 0.82 | -220.6 ms |
| C3 copy-add zero-copy root  | 2415.0 | 2415.6 | 4.34/4.34 | 0.81 | -221.1 ms |
| Q  copy-add fp8e4m3 partials| 2852.3 | 2859.6 | 3.68/3.67 | 0.69 | -275.8 ms |
| hostadd footnote (1 thread) | 882.1  | -      | -         | -    | -29.5 ms  |

1.50 MB bf16 = 750000 elems:

| arm | med us run1 | med us run2 | wire GB/s | AR busbw GB/s | chunk saving vs ring 646 (x125) |
|-----|------------|------------|-----------|---------------|--------------------------------|
| R  copy ring (model)        | 2783.5 | 2796.2 | 6.47/6.44 | 0.81 | -267.3 ms |
| C  copy-add naive (GATE)    | 2729.6 | 2729.3 | 6.59/6.60 | 0.82 | -260.5 ms |
| C2 copy-add event-chained   | 2740.6 | 2739.4 | 6.57/6.57 | 0.82 | -261.8 ms |
| C3 copy-add zero-copy root  | 2790.1 | 2785.9 | 4.30/4.31 | 0.81 | -267.5 ms |
| Q  copy-add fp8e4m3 partials| 3229.8 | 3241.6 | 3.72/3.70 | 0.69 | -323.0 ms |
| hostadd footnote (1 thread) | 935.2  | -      | -         | -    | -36.1 ms  |

Correctness: R/C/C3 verify PASS at both sizes (transport bar 0.02*(sum_r|x_r|+0.25),
cross-rank bit-identity); Q maxabs vs R output = 0.2812 at BOTH sizes (gate bar 1e-2).

## GATE VERDICTS (pre-registered 2026-09-19, evaluated on the final banked run)

- **GATE-Q0 (C <= 355.3 us at 1.31 MB): FAIL, 6.7x over.** C = 2399.3/2396.9 us
  (reproduced). Also fails at 1.5 MB. The family's best variants (C2 2411, C3 2416)
  fail identically — this is not a pipelining or copy-shape artifact.
- **GATE-Q1 (Q): FAIL on BOTH halves.** Speed: Q/C = 1.19x (1.31 MB) and 1.18x
  (1.5 MB) > 1.15x. Numerics: maxabs vs R = 0.2812 >> 1e-2, both sizes. fp8 PARTIALS
  (quantize per rank before transport) are dead at bf16-scale activation data, as the
  rough screen predicted — scope note: fp8-ON-RING (quantize after the ring's reduce,
  or reduce on fp8 wire with bf16 accumulation) is a DIFFERENT design and is NOT closed
  by this cell.
- **NULL BANKED (pre-registered branch): the copy-add family CLOSES as priced.** The
  AR cure's option 0 cannot beat the ring at prefill sizes on this transport. Fallback
  per pre-registration: (a) one-shot wedge root-look (the filed blocker), (b) fp8-on-ring.

## WHY (the mechanism, for the coordinator)

Copy-add one-shot moves the SAME 12S wire bytes as the ring (4 publishes + 4 root pulls
+ 1 result + 3 broadcast vs the ring's 2*(N-1)/N per rank x 4 ranks); the in-cell copy
ring R (2452 us) and copy-add C (2399 us) sit 2.3% apart — identical traffic, and C's
leg structure (4 barriers) is no better than the ring's (6 steps, duplex-overlapped).
On the real 4-link transport the NCCL ring does those same 12S bytes in 646-771 us, so
by byte-counting alone NO copy-add variant can reach 0.55x the ring there. The only
byte-reduced variant, C3 (8S, root add reads host-mapped slots zero-copy — the serving
one-shot's transport minus the flag machinery), shows WHY the one-shot machinery buys
nothing here either: kernel NT reads over PCIe on gfx900 run at ~4.3 GB/s effective vs
~6.6 GB/s for copy-engine legs — the 33% byte saving is exactly eaten by the 34% lower
per-byte bandwidth (net wash, measured both sizes). Consequence for the root-look desk:
when the one-shot wedge is fixed, its ceiling at prefill sizes is ~the ring, not 2x
under it; the real prefill AR wins remain fewer collectives (fuse mixer+mlp ARs) and
fp8-on-ring, per the RCCL-matrix closure. host-add (882-935 us, single thread) is
independently disqualifying for the llama.cpp-style host-sum design.
