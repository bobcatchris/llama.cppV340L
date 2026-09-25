# W42 MMVQ SCHEDULE LIFT DESK - the rpb1 register-diet arm - 2026-09-25

Desk mmvq2 (worktree wt-mmvq2, branch amd/mmvq2 off amd/v340-port-v2 @
53a2b70cd). ZERO-GPU desk: diagnosis + one env-gated arm + staged oracle
and bench, per the W27 protocol. No card time touched; every kernel run
below is a COMMAND FOR THE COORDINATOR, not a result.

Inputs of record read: W37_roofline_accounting (the brief), W5_mmvq_bw
receipt (cfull band + s2r), W6_mmvq_rungs_receipt, W7_mmvq_ldsy receipt
(LDS-y killed twice over + s2r wiring fix), W23_mmvq_bw2 (M1+M2 mechanism
of record + bench protocol), W30_bench_output (lb2 no-op, lb4 occ-only,
C4 fusion fail, OCC table), W27_q40prefill (staged-oracle protocol),
OPTIMIZATION_PLAN E-104/E-106/E-137, ggml/src/ggml-cuda/mmvq.cu +
vecdotq.cuh, docs/amd-port/tests/bench_mmvq_real.cu, and the BASEENV
lines in docs/amd-port/scripts/run_deepcensus_window.sh.

## 1. DIAGNOSIS: which inefficiency the 54.7 GB/s is

The brief offered four candidates: (a) weight-line underfetch / kbx
runway starvation, (b) L2 line utilization, (c) occupancy from register
pressure, (d) census. The receipts already narrow this to M1+M2 jointly,
and the code reading confirms it; (b) is ruled out.

- (b) RULED OUT (do not re-open): W6 A1 misalignment law - the 2 B-aligned
  110/98 B strides amplify sectors only ~1.2-1.5x, sectors are shared
  within a block, and W5 A2 wide-x measured NOT-MOVEMENT (-0.9..-1.2%)
  because x-operand loads are 12-20% of per-iteration load instructions.
  A weight-layout swizzle (brief candidate iii) is the ALN class: W4
  measured it NEGATIVE on this schedule, and a rebake-class change also
  fails the byte-identity law. Dead.
- (ii) LDS-Y STAGING IS A CLOSED RUNG (the brief's candidate ii). W7 built
  the literal W5 design (52 B pad, stride 13, verbatim 36 B records) on
  the real kernel: +63..+198% vs base, all five types, T=2..4, runways
  4/8/16, AND the implementability finding shows an LDS-consuming share
  path cannot be both bit-exact and codegen-neutral on gfx900/hipcc-6.2.0
  (contraction divergence, 5030/69632 els). Mechanism: y is CTA-shared
  and already L1-resident; staging re-reads it globally and adds a round
  trip. NOT re-proposed.
- (a)+(c) = M1+M2, the mechanism of record (W23 section 3, W37 section 2):
  the kbx loop (mmvq.cu:576) gives each lane 1.25 independent iterations
  at K=5120 (blocks_per_row_x = 20, blocks_per_iter = 16 for the iq3
  class at nwarps=2, wave64), and the resident-wave count is
  register-capped: W30 OCC dump on the served templates - iq3_s 77-83
  VGPRs = 6 CTAs/CU, iq4_xs 84-85 = 6, q4_K 101-112 = 4, of 40 waves/CU
  (8-12 of 40 = 20-30%). With ~600-cycle HBM/L2 latency, outstanding
  loads per CU = waves x in-flight-per-wave; M1 caps the second factor,
  M2 caps the first. cfull (compute deleted) still only reaches 70-120
  GB/s, and the fast cfull arms are exactly the low-register builds -
  the W7 ldsy build measured 62-64 VGPRs reaching 7-8 CTAs on the same
  bodies. Issue census says 5.5x issue slack: stall-bound, not
  issue-bound. Decode+dp4a tax (+29..+69%) is the residual INSIDE the
  band; this desk does not claim it.

So the one open, in-family, codeable lever is M2 without the lb4 spill:
W30 occ-dumped lb4 at 8 CTAs/CU but never timed it, and lb4 forces 64
VGPRs by spill (16 B/thread local on iq3_s, 208 B on q4_K) instead of by
diet. W37 Opportunity 1 names exactly this: "any register diet that holds
8 CTAs/CU without the lb4 spill".

## 2. THE ARM: GGML_CUDA_MMVQ_RPB1 (rows_per_cuda_block = 1 on the share body)

Design: the T=2..4 share arms hold dec[rows_per_cuda_block][8] (plus
scale/dm share arrays) and tmp[ncols_dst][rows_per_cuda_block]. The two
rows per CTA exist to amortize the y window across a row pair (rpb=2
schedule, upstream constant). The decode state is the register sink:
dropping to ONE row per CTA halves it (dec[1][..], tmp[4][1]) at
unchanged per-lane kbx runway (still 1.25 iterations - each lane's
iteration is one row's block, and the row loop collapses to i=0).

Static VGPR accounting (why the diet reaches the 64-VGPR boundary):

| type (share arm) | served regs (W7/W30) | CTAs/CU | rpb1 static savings | predicted regs | predicted CTAs/CU |
|---|---|---|---|---|---|
| iq3_s  | 83 | 6 (12 w) | dec -8, scale/d -3, tmp -4 = -15 | ~68 | 7 (14 w) |
| q4_K   | 101-112 | 4 (8 w)  | dec -8, sc -4, m -4, dm -2 = -18 | ~85 | 6 (12 w) |
| q5_K   | 104 | 4 (8 w)  | same class as q4_K | ~86 | 6 (12 w) |
| iq3_xxs| 64  | 8 (16 w) | -15 (already at the floor) | ~64 | 8 (no change; CONTROL) |

(gfx900: waves = floor(65536/VGPRs); 8 CTAs = 16 waves needs <= 64
VGPRs, 6 CTAs = 12 waves needs <= 85. LDS at rpb1/T=4 drops 2 KB -> 1 KB
tmp_shared - not a limiter. The compiler may keep the freed registers as
scheduling slack, so the arm COMPOSES with the existing GGML_CUDA_MMVQ_LB
gate: RPB1+LB4 forces the 64-VGPR budget, where the diet means the spill
is 0-16 B/thread instead of lb4's 16-208 B. Both compositions are
benchable.)

The prediction table is deliberately falsifiable: iq3_xxs share is
ALREADY at 8 CTAs/CU (64 regs) - if M2 is the binding term it must NOT
move, making it the built-in control; q4_K/q5_K (4 -> 6 CTAs) must move
MORE than iq3_s (6 -> 7) in wave-proportional terms. If the OCC dump
shows rpb1 regs NOT below share regs, the diet failed and the arm is
void BEFORE any timing (no session spent).

Bit-exactness by construction: the per-(row, kbx, lane) decode and the
per-token apply sequence are unchanged; the second row simply executes
in the sibling CTA with the identical tid mapping, so each row's
accumulation order (kbx order, warp reduce, cross-warp sum through
tmp_shared) is the same as today. No pointer-base change (the W7 LDS
contraction trap does not apply - same global vy/vx operands, same
expression bodies). Codegen may still re-schedule, so the oracle is
mandatory per the W30 law ("codegen changes per arm, every arm needs the
bit-exact oracle"). Can-fail-before-can-green.

What the arm does NOT do: no runway change (M1 untouched: 1.25 iters),
no decode-tax change, y traffic per CTA unchanged (x bytes per CTA halve,
CTA count doubles; total x bytes identical; y is L1/L2-resident per W7
and its doubled L2->L1 stream is priced by cfull-cy at 105-218 GB/s -
not the wall).

### Code changes (this worktree)

- ggml/src/ggml-cuda/mmvq.cu:
  - mul_mat_vec_q<...> gains a trailing template param `bool RPB1 =
    false` (append-only; every existing instantiation stays valid);
    rows_per_cuda_block = (RPB1 && ncols 2..4) ? 1 : calc_rows_per_block.
    All guards (store bound, bias prefetch, tmp_shared shape) already
    parameterize on rows_per_cuda_block - the T=1 shape uses rpb=1 today.
  - calc_launch_params gains `const bool rpb1 = false`; grid becomes
    ceil(nrows/1) when active. MUST agree with the kernel constexpr.
  - mmvq_rpb1_env_enabled(): GGML_CUDA_MMVQ_RPB1=1 gate, GGML_LOG_WARN
    engagement line (E-117: served log drops INFO). Default OFF: unset env
    produces byte-identical launches.
  - mmvq_rpb1_active(type, ncols_dst, s2r): the single predicate shared
    by grid calc and launch gate = ncols 2..4 AND a *_SHARE gate set AND
    NOT (s2r arg AND s2r env). RPB1 supersedes s2r (no row pair to share)
    and aln; it composes with LB=4 as the launch_bounds hint. The fused
    GLU-T4 launch (default OFF) ignores RPB1 (share body has no gate
    accumulator).
  - switch_fusion: rpb1 branch launches
    mul_mat_vec_q<type, c, false, small_k, false, false, lb, true>.
- docs/amd-port/tests/bench_mmvq_real.cu: new arms
  - `lb4s`     = share body at LB4 (the served dispatcher CAN express
    this as GGML_CUDA_MMVQ_LB=4 + share gates; W30 occ-dumped it, never
    timed it - free extra datapoint that separates "diet" from "forced"),
  - `rpb1`     = share body, one-row blocks, grid (N,1,1),
  - `rpb1lb4`  = rpb1 + LB4 hint.
  Timing convention unchanged (niter=30, reps=5, spread law 1.05%,
  oracle-first).

## 3. ORACLE (W27/W7 protocol - every command is coordinator-run)

Instrument builds (die-3 host, HIP_VISIBLE_DEVICES resolves at runtime):

    cd /media/chris/ssd128/llamacpp/wt-mmvq2
    /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 \
      -DGGML_USE_HIP -DGGML_BACKEND_BUILD -DGGML_SHARED \
      -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
      docs/amd-port/tests/bench_mmvq_real.cu -o /tmp/w42_bench_mmvq_real \
      -L /media/chris/ssd128/llamacpp/wt-mmvq2/build-hip/bin \
      -lggml-hip -lggml-base -lamdhip64 \
      -Wl,-rpath,/media/chris/ssd128/llamacpp/wt-mmvq2/build-hip/bin \
      -Wl,-rpath,/opt/rocm-6.2.0/lib

(compile already proven on the desk host: BUILD clean, 0 errors/warnings
against the MAIN tree's libggml for linking; the worktree build-hip build
is the served-codegen proof.)

3a. OCC-first (zero timing; arms void if the diet shows no reg drop):

    HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/media/chris/ssd128/llamacpp/wt-mmvq2/build-hip/bin \
      /tmp/w42_bench_mmvq_real /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf \
      iq3_s 4 share,lb4s,rpb1,rpb1lb4 --occupancy
    # repeat for: q4_K, q5_K, iq3_xxs (the control: expect no CTAs change), q6_K, q3_K

3b. In-process oracle + timing, T=4 (first arm = base reference; every
later arm is device-memcmp'd vs base BEFORE its timing counts; final
base dup = drift control):

    HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=<wt-build>/bin /tmp/w42_bench_mmvq_real \
      <gguf> iq3_s   4 base,share,rpb1,rpb1lb4,lb4s,base 30 5
    HIP_VISIBLE_DEVICES=3 ... iq3_xxs 4 base,share,rpb1,rpb1lb4,base 30 5
    HIP_VISIBLE_DEVICES=3 ... q4_K   4 base,share,rpb1,rpb1lb4,lb4s,base 30 5
    HIP_VISIBLE_DEVICES=3 ... q5_K   4 base,share,rpb1,rpb1lb4,base 30 5
    HIP_VISIBLE_DEVICES=3 ... q6_K   4 base,share,rpb1,base 30 5
    HIP_VISIBLE_DEVICES=3 ... q3_K   4 base,share,rpb1,base 30 5
    (iq4_xs: its share gate is OFF in BASEENV and the rpb1 arm rides the
     share body, so rpb1 cannot engage for iq4_xs under BASEENV - it is
     an irrelevant cell for this arm unless C1 (IQ4XS_SHARE=1) is stacked
     in a later window; skip it here.)
    T=2 (catch-up pool) once on iq3_s: base,share,rpb1,base 30 5.

  Pass bars: every arm ORACLE BITEXACT; base dup within noise; medians
  spread <= 1.05% (cool-die, dedicated per-type sessions per the W5
  hygiene law); any oracle FAIL = the arm is dead, no timing counts.

3c. Default-path law (the W7 A4 discipline) - the arm must not perturb
the default or share paths:

    # reference from the PRISTINE tree (amd/v340-port-v2 build-hip):
    HIP_VISIBLE_DEVICES=3 /tmp/bench_pristine <gguf> iq3_s 4 base,share,base 30 5 \
      --dump /tmp/w42_ref_pristine_iq3_s.bin
    # this worktree, env OFF:
    HIP_VISIBLE_DEVICES=3 /tmp/w42_bench_mmvq_real <gguf> iq3_s 4 base,share,base 30 5 \
      --oracle-file /tmp/w42_ref_pristine_iq3_s.bin     # must print CROSS-ORACLE BITEXACT
    (W30 left refs at /tmp/w30_ref_{iq3_s,iq4_xs,q4_K}.bin - if they
    survive, they serve as the pristine reference directly.)

3d. Served-path engagement (real graph, one boot of the worktree binary):

    default boot: log must show NO rpb1 line, dumps byte-identical to the
                  pristine serve (default-path law).
    BASEENV + GGML_CUDA_MMVQ_RPB1=1 boot: log MUST show
    "GGML_CUDA_MMVQ_RPB1=1, one-row blocks ..." (WARN, E-117), and the
    share-spec logits dump vs BASEENV must be byte-identical.

## 4. EXPECTED GAIN (arithmetic, honestly banded)

Cell level (K=5120 bench, T=4), if M2 is as named and BW scales with
resident waves toward the cfull ceiling:

| type | served GB/s | predicted cell delta | mechanism |
|---|---|---|---|
| iq3_s   | 50.6 | -5..-15% us | 6 -> 7 CTAs (+17% waves) |
| q4_K    | 48.8 | -8..-20%    | 4 -> 6 CTAs (+50% waves) |
| q5_K    | ~29-45 | -8..-20%  | 4 -> 6 CTAs |
| q6_K    | 39.2 | -5..-15% (if reg-capped; OCC first) | worst cfull ratio (33%) |
| iq3_xxs | 41.6 | 0 (CONTROL: already 8 CTAs) | falsifies M2 if it moves |
| iq4_xs  | 75.0 | untouched under BASEENV (share OFF) | |

Pool level (W37 verify table, 56.0 ms at 54.7 GB/s): applying the band to
the gated types: iq3_s 17.6 -> 15.0-16.7, q4_K 8.5 -> 6.8-7.8, q5_K 1.8
-> 1.4-1.7, q6_K 3.7 -> 3.1-3.5, rest flat: verify pool 56.0 -> 50.5-53.5
ms (-4.5% to -9.8%), i.e. 54.7 -> 57-60 GB/s effective - the FIRST STEP
toward the 70-120 GB/s family band, not the band itself (the decode+dp4a
tax and the runway term remain).

Served projection (cycle 123.1 ms, 24.69 t/s of record):
- full pool transfer: -2.5 to -5.5 ms/cycle -> 25.2-26.0 t/s (+2.1% to
  +5.3%). Clears the W37 falsifiable gate (+2%) only near the top.
- w23env-class 1/8 haircut (bench-cell transfers): +0.3-0.7% - BELOW the
  gate. Honest expectation: this is a MUST-LABEL-to-small-WIN arm, and
  its value is half diagnostic (it closes or confirms the M2 half of the
  mechanism of record).

Falsification clauses (pre-registered):
1. OCC dump: rpb1 regs not lower than share regs -> diet void, no timing.
2. iq3_xxs (control) moves >= 5% -> the wave model is wrong, M2 closes.
3. rpb1/rpb1lb4 bench deltas < 2% on q4_K/q5_K at K=5120 while occupancy
   actually rose -> occupancy is not binding on the share arms -> the
   W37 Opportunity-1 occupancy family closes (its own clause), and the
   residual is decode tax + runway - next link is format-level, outside
   this desk.
4. Served paired window < +2% -> falsified per the W37 gate; keep the
   gate default-OFF and bank the negative.

## 5. BENCH SESSION SPEC (one die-3 window, ~35 min card time)

Lock protocol: check-and-hold /tmp/campaign_gpu_boot.lock, cool-die gate
(25-28 C) before the first session, dedicated per-type sessions (the W5
bimodal late-session law), die witness line "device PCI: 0000:0d:00.0"
grepped.

    # 0. build the worktree binary (done): build-hip/bin/{libggml-hip,libggml-base,libggml}.so
    # 1. OCC dump (3.0 min): the 6 runs of section 3a
    # 2. iq3_s session (6 min): section 3b line 1
    # 3. q4_K session (5 min), q5_K (4 min), iq3_xxs control (4 min),
    #    q6_K (8 min - it is 5.9 ms/launch), q3_K (3 min)
    # 4. T=2 iq3_s (3 min)
    # 5. cross-binary default-path oracle vs pristine dump (2 min, 3c)
    Total ~35 min; abort on any ORACLE FAIL (can-fail law) or SPREAD-FAIL
    on the hot late types.

## 6. PAIRED SERVED WINDOW SPEC (only if 3a/3b pass)

- Binary: this worktree's build-hip (the arm is code, not env-only).
- Arm A (control): BASEENV verbatim (run_deepcensus_window.sh line 13).
- Arm B: BASEENV + GGML_CUDA_MMVQ_RPB1=1. Nothing else changes: no s2r
  gates (RPB1 supersedes them by construction), no ALN, no LB (LB=4 rides
  only as a second-order cell if 3b showed rpb1 alone reg-capped above
  64).
- Exactness gate IN-WINDOW before any t/s counts: share-spec logits dump
  vs Arm A byte-identity (W7 A4 protocol). RPB1 predicted bit-exact by
  construction; any byte drift = session VOID and the arm is dead
  regardless of speed.
- Interleaved multi-rep decode A/B on the TP4 config of record
  (24.69 t/s anchor, mean_len 3.00 population), >= 3 reps, medians;
  verdict fields: decode t/s AND MMVQ pool ms from a rocprof census rep.
- Gate: promote to default-ON only at >= +2% served with bit-identity
  (else MUST-LABEL, stays default-OFF per the noise law).

## 7. BUILD STATUS (this worktree, proof)

- mmvq TU + bench binary: /opt/rocm-6.2.0/bin/hipcc -O3 -x hip
  --offload-arch=gfx900 compile of docs/amd-port/tests/bench_mmvq_real.cu
  (includes mmvq.cu verbatim): clean, 0 errors 0 warnings ->
  /tmp/w42_bench_mmvq_real, final link against THIS worktree's
  build-hip/bin (served-codegen proof for the new instantiations).
- Worktree build-hip: configured with the main CMakeCache flags (GGML_HIP
  ON, gfx900, Release, ROCm 6.2.0 clang++, GGML_HIP_GRAPHS/RCCL/NO_VMM
  ON, GGML_LLAMAFILE ON, SCHED_MAX_COPIES 4) and built: ggml-hip target
  TARGET-EXIT:0 (libggml-hip.so in build-hip/bin), full llama target
  LLAMA-EXIT:0 (libllama.so). The serving binary of the paired window is
  this build-hip tree.

UPSTREAM-FACING NOTE: private-fork dev cell; nothing here is upstream-PR
material per AGENTS.md.
