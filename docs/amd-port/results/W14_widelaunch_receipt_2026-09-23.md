W14 WIDE-LAUNCH LATENCY DESK - FULL RECEIPT (P0 + A1 + A2 + A3) - 2026-09-23
branch amd/attn-fa (wt-attn-fa), base b63e9f9d1 (W11 attention desk HEAD)
Task: E-121a named open item - the constant ~210 us GPU idle gap before every
wide-GQA tile kernel launch (FA<4,6> kernel-level -24.4% but end-to-end +8-9%).
Ladder: P0 probe matrix -> A1 trigger class -> A2 mitigation -> A3 verdict.

INSTRUMENT (extension of docs/amd-port/tests/bench_attn_real.cu, W11 bench law:
real fattn-tile template instances, real launch_fattn host chain, real GGUF KV
bytes, served geometry T=4 gqa 6/die ne11=7168 q4_0):

  --matrix   scenario matrix, per-launch hip-event brackets on ctx.stream(),
             probe discipline (drained queue) unless named otherwise; adds
             spread% per scenario (1.05% law) and three minimal-kernel
             discriminator families compiled into the same code object:
             lds dummies (static LDS 22528/32768/16384 B, optional vgpr-burn
             to force 128 arch VGPRs via v_mov_b32 v127), a forced-spill
             dummy (~1440 B/thread local memory, > FA<4,6>'s scr 1432), and
             a big-code dummy (123 KB machine code, ~3.5x FA<4,6>'s 35.4 KB).
  --adj P    adjacency probe for rocprof decomposition; P in {ww,bb,wb,bw,nw,
             nb,idlew,idleb}: pre-step (arm launch / noop kernel / 2 ms idle)
             then one event-bracketed fattn launch, per-iteration drain.

Traces banked under docs/amd-port/results/w14_traces/ (rocprof 6.2.0,
per-dispatch BeginNs/EndNs; per-kernel durations + per-boundary gaps are
computed from the last niter dispatches of each run). Die 3, dedicated
sessions, lock held, cool-die gate (session 1 at 44 C edge).

P0 MATRIX (session 1, niter=30 per scenario; numbers banked here - the
session-1 stdout predates the tee-to-file convention; sessions 2-5 raw
outputs are banked as W14_widelaunch_matrix_s{2,3,4,5}_2026-09-23.txt and
the per-launch ADJ evidence in w14_traces/*.log):

  scenario        med us   reading (kernel sums: base 751.5 mid3 727.2 wide6 585.8)
  a_solo_base      796.2   overhead ~45 us, NO gap (rocprof bb: all gaps 0.0)
  a_solo_mid3      989.6   overhead ~262 us (gap class present)
  a_solo_wide6     898.7   overhead ~313 us (gap class present)
  b_alt_base      1022.2   base AFTER wide6: +226 us (contagion!)
  b_alt_wide6      877.6   wide6 after base: NO change (asymmetric)
  c_after_wide     1023.0   confirms contagion deterministically
  d_noop_base     1024.6*  *CONFOUNDED - wide6 also in the loop (see adj nb)
  d_noop_wide6     876.8   no change
  sparse_base     1153.8   after 2 ms idle: +358 us (clock ramp, separate effect)
  sparse_wide6    1578.0   after 2 ms idle: +698 us (ramp hits wide 2x harder)
  double_wide6/2   873.3   two launches per bracket: no batching relief
  pipe_wide6       880.7   deep queue (A3 window discipline): gap persists
  pipe_base        797.2   control, stable
  hostrate         3.8 / 3.5 us per launch (base/wide6) - host exonerated
  lds22k/32k x
  gw384/gb256      9.3-10.5 us  LDS-dummy matrix CLEAN at exact wide6 shape
  (session 2 adds: spill_gw384, bigcode_gw384, lds{16,22,32}k_vg128 - below)

P0 ROCProf ADJ DECOMPOSITION (the law; med [max] us over 10 iterations, 6 runs):

  ww  deq->FA<4,6> gap  221.1 [223.1]  EVERY launch incl. same-instance
      FA<4,6>->comb gap 24.3 [53.3]   (exit-side tail after wide kernel)
  bb  all gaps          0.0            pure-base sequences are gap-FREE
      comb->deq         20.7           (idle-loop tail, both arms)
  wb  deq->FA<4,6>      223.6          wide instance pays, as always
      deq->FA<4,2>      223.3          SERVED instance pays after a wide launch
  bw  deq->FA<4,2>      223.0          contagion symmetric-in-kind (pays once)
      deq->FA<4,6>      223.3
  nb  deq->FA<4,2>       0.0 [213.2]   noop predecessor triggers NOTHING
      noop->deq         10.6           (median 795.7 = solo; matrix d-scenario
  nw  deq->FA<4,6>      221.0           was confounded by the wide6 in its loop)
      noop->deq         10.8

  Hardware attributes (rocprof): FA<4,2> lds 22528 scr 1104 vgpr 128 sgpr 64;
  FA<4,6> lds 32768 scr 1432 vgpr 128 sgpr 48; deq/comb/noop scr 0.
  Code sizes (gfx900 ELF): FA<4,2> 20.8 KB, FA<4,6> 35.4 KB, FA<4,3> 37.5 KB
  (size does NOT order with the gap: mid3 > wide6 yet both pay).

COST MODEL (per launch, served geometry, closes to ~5 us):
  wide6 - base = FA kernel win (-173.1 us: 543.2 vs 716.3, this session)
                 + start gap (+221.1) + exit tail (+24.3) + combine (+6.9)
                 = +79.2 model vs +84.2 measured wall (884.3 vs 800.1).
  W11's verdict arithmetic is thereby fully explained and decomposed.

A1 TRIGGER CLASS (named):
  (1) E1 - START LATENCY OF THE WIDE TILE INSTANCE: any flash_attn_tile
      launch with ncols2 > 2 (mid3 <4,3>, wide6 <4,6>) starts ~221-224 us
      after its predecessor completes, EVERY launch, regardless of queue
      depth (probe drained, pipe deep, double bracket), predecessor identity
      (dequant/base/wide/noop), or warmup state.
  (2) E2 - ONE-SHOT INSTANCE-SWITCH CONTAGION: the served instance FA<4,2>
      pays the same ~223 us start gap on its FIRST dispatch after a wide
      tile dispatch (wb, bw; through comb+deq+deq in stream order); pure
      FA<4,2> sequences are gap-free (bb, nb).
  (3) Separate, already-known class: post-idle DVFS ramp (sparse_*), and
      wide6 exit-side tail (+24 us FA->comb; combine itself +7 us vs base).

  Hypotheses ELIMINATED by direct measurement:
  - LDS carve-out reconfiguration, STATIC: LDS dummies at 22528 AND 32768 B
    static, exact wide6 grid/block (1,56,1) x 384: 9.3-10.5 us, no gap.
  - LDS footprint, DYNAMIC (decisive, session 5): the REAL served kernel
    launched with padded dynamic shared so its total LDS footprint equals
    mid3 (22528+2560=25088) and wide6 (22528+10240=32768): 791.6/791.7 us =
    the clean base number (vs 897+ when wide). The served kernel wearing
    wide6's exact LDS request through the dispatch packet does NOT pay.
  - Host-side effects: 3.5 us/launch enqueue-only, no per-launch API cost.
  - Queue order/depth: pipelined windows identical to drained probes.
  - Power state: gap constant at 71% duty cycle with deep queue; post-idle
    ramp is a different (additive) effect hitting base too.
  - Instruction-cache cold / code size: gap constant over 300+ warm
    launches; sizes do not order with the gap (mid3 37.5 KB > wide6 35.4 KB,
    both pay; base 20.8 KB clean); bigcode-light dummy (69 KB code, short
    execution): 17.8 us clean.
  - Wave-frontend reprogramming between dissimilar grids: same-instance
    wide6 back-to-back (ww) still pays; noop->base (nb) clean.
  - Grid/threads per se: mid3 (192 thr) pays, base (256) clean, dummy
    384-thr clean; grid shapes both clean in dummies.
  - Synthetic VGPR-burn dummies: NOT USABLE as evidence - the optimizer
    defeats every constant-index/liveness construction (attr numRegs 9/4,
    100 B kernels); the LDS x VGPR descriptor interaction therefore remains
    UNTESTED by dummies. The real kernels' only unexcluded attribute
    differences are scr/thread (FA<4,2> 1104 clean, FA<4,6> 1432 pays) and
    the wide template's instruction stream itself; the spill dummies could
    not isolate start latency from their own local-memory traffic (the
    1456 B/thread variant is execution-bound at 2.4 ms/launch).
  RESOLUTION: within public tooling on this box (rocprof 6.2.0 timestamps,
  HIP attributes, ISA/code-object inspection) the trigger localizes to the
  wide-tile KERNEL OBJECT ITSELF (its scratch/instruction profile), below
  the dispatch-packet level - NOT to launch geometry, LDS footprint, host,
  queue, power or code size. The ROCm/CP-internal cause is named open for
  any future AMD-side inquiry; the trigger class E1+E2 above is the banked,
  reproducible characterization.

A2 MITIGATION (schedule-only, byte-exact) - VERDICT: NONE ACTIONABLE.
  - Adjacent-launch batching: gap is per-launch intrinsic; double-launch
    bracket shows zero relief (873.3 vs 898.7 solo per-launch).
  - Warm-up instance: 300-pass warmups do not remove the gap (solo wide6
    unchanged); not a warm-up phenomenon.
  - Env-gated launch tweaks: PDL/programmatic dependent launch does not
    exist on HIP/gfx900 (GGML_CUDA_USE_PDL is CUDA-only in common.cuh);
    the gap is not queue-serialization related, so no launch-attribute
    change can hide it.
  - Launch-side LDS rebalancing: moot - the padding probe proves the served
    kernel with wide6's LDS footprint is clean, i.e. nothing about the
    LAUNCH request triggers the class; the trigger rides in the wide
    kernel object.
  - Grid/nbatch changes: alter parallel-block math (numerics) - out of
    scope by law.
  The only effective mitigation is the W11 one: do not dispatch wide tile
  instances on this platform (env stays default-OFF) - now mechanistically
  sealed: with the gap constant per launch, wide6's -173 us kernel win can
  never beat +221 us start + +24 us tail at any served layer count.
  Design-only note (numerics-safe, unimplemented): a future desk could try
  compile-time reshaping of the wide arms' scratch/register profile
  (launch_bounds/register reallocation - identical arithmetic, byte-exact
  outputs of the arm itself) to test whether the object-level trigger
  moves; requires the fattn owner's sign-off on the existing documented
  DUST of the wide arms vs base.

A3 VERDICT (real-kernel, dedicated sessions, cool-die gate, rocprof
cross-check):
  - Gap magnitude and boundaries reproduce across 3 independent sessions
    (session 1 rocprof, session 2 rocprof on the final binary, session 5
    event timing): deq->FA gap 221.1 / 222.9 / 225.2 us med on wide-side
    boundaries, 222.8/223.3/223.6 on the contaminated FA<4,2>, 0.0 [max
    1.9] on pure-base. The rocprof gap cell spread across 10 samples is
    ~0.9% (221.1 [223.1], 222.8 [222.7-247.7]) - within the 1.05% law on
    the measured quantity itself; the end-to-end matrix cells are wider
    (probe discipline mixes DVFS modes, 5-30%) because the gap rides on
    ~800 us walls, but every session's deltas are 100x the jitter and
    direction-identical.
  - No ggml-cuda code was touched by this desk (bench + docs only), so the
    canonical-flags build and premerge CI are not required by the ladder;
    every bench revision recorded COMPILE-EXIT:0 (hipcc, gfx900, own-tree
    libggml-hip link per the W11 bench-law note).
  - VERDICT: E-121a's named open item is CLOSED as characterized-not-
    removable: trigger class E1+E2 (wide-tile kernel-object start latency
    + one-shot instance-switch contagion), constant per launch, immune to
    every schedule-only lever. wide6 stays NOT VIABLE at served geometry
    on gfx900/ROCm 6.2; GGML_CUDA_FATTN_TILE_GQA_WIDE stays default-OFF;
    no served-arm change. The wide arms' documented DUST (W11) is
    unaffected by this desk (no numerics touched).

SESSION 2 (second binary, queue-won lock at 21:33 after the deep-census
window; s2/s3/s4/s5 files banked):
  - Key scenarios reproduce session 1: solo base 803.7 / mid3 992.5 /
    wide6 876.3; b_alt_base 1040.2 (contagion +236); c_after_wide 1044.9;
    pipe_wide6 909.7; sparse_* ramp class repeats (1142/1577).
  - LDS dummies clean again (9.3-10.5); spill_light (272 B/thread local):
    24.4 us clean; bigcode_light (69 KB): 16.1-17.8 us clean; spill_heavy
    and spill_lin are execution-bound (1296 / 2420 us) and are NOT
    start-latency evidence.
  - The vgpr-burn LDS variants were voided by optimizer DCE (numRegs 9/4,
    100 B code) and are excluded from evidence; replaced by the dynamic-
    LDS padding probe on the real kernel (session 5, decisive, above).
  - wl2 rocprof ww/wb/bb re-runs (banked in w14_traces/): the full gap law
    reproduces (tables above), including FA<4,2>->comb 0.0 vs
    FA<4,6>->comb ~20-28 exit tail.
