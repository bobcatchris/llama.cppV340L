# W9 RECEIPT: RCCL TRANSPORT DESK (P0 multi-die probe + A1 residual localization + A2 env sweep + A3 served spec) - 2026-09-23

Desk: RCCL TRANSPORT DESK (wt-rccl-transport, amd/rccl-transport, based on
68766bec1). Mission: the U3 multi-die probe the boundary desk named - build
the 4-die RCCL instrument, localize the 59-141 us per-die residual, sweep
RCCL transport env, ship a served arm if one wins.
Foundation: W8_tp4_boundary_receipt_2026-09-23.md, TP4_roundmap_2026-09-23.md,
ledger E-110..E-116a.

## 0. Headline

1. THE SERVED PER-DIE ASYMMETRY DOES NOT REPRODUCE IN ISOLATION. The
   isolated 80 KB fp32 4-die allreduce is FLAT across dies: lockstep dev
   medians 126.5 / 118.1 / 118.1 / 125.0 us (spread x1.07) vs the served
   census 191.6 / 210.8 / 151.9 / 128.8 (x1.64). The W8 residual T(die)
   = 59-141 us is therefore NOT RCCL transport cost - it is interleave
   peer-wait (where each die's kernel sits inside the collective while it
   waits for the slowest arriver's data), set by the compute schedule, not
   by the wire. Flattening the served wall is a compute-balance lever,
   not a transport lever.
2. The transport component itself is real and now measured: 103.2-103.5 us
   per-op burst-pipelined at 80 KB (uniform across dies), vs the W8
   transport-free ring floor 69.5 us -> the SHM wire + launch + sync adds
   ~34 us, the rest of the isolated cost is the on-device ring algorithm.
3. ONE ENV KNOW WINS, REPRODUCIBLY: NCCL_MIN_NCHANNELS=4 lifts the ring
   from the default 2 channels to 4 (INFO fingerprint, both sessions of
   record) and cuts the pipelined per-op to 91.6-92.2 us (-11.0..-11.5%,
   3 sessions) and the lockstep worst die from 126.5 to 113.3-114.4 us
   (-9.6..-10.4%). The mission win bar (>= 15% worst-rank) is NOT MET -
   recorded as a partial: real, reproducible, under bar. Ring order and
   proto are unchanged by the knob, so per-element reduce order is
   structurally preserved; fp-dust class + same-sha two-boot gate still
   applies per convention.
4. Transport of record PROVEN to be SHM: NCCL_SHM_DISABLE=1 explodes the
   per-op to 191-214 us (+85%) with all four dies still connected
   (fallback path); NCCL_P2P_DISABLE=1 is exactly neutral (P2P was never
   available). Tree algo is strictly worse (150 us burst) and REINTRODUCES
   a die gradient (114.8/134.7/154.8/169.8, x1.48) - ring is correct here.
5. Multi-PROCESS RCCL init is a hard defect on this stack: ncclCommInitRank
   world=4 across 4 processes (one per die, HIP_VISIBLE_DEVICES per
   process) fails ncclUnhandledCudaError(1) in every rank instantly,
   regardless of stagger, clique-ignore, registration-off, or bootstrap
   interface. The single-process ncclCommInitAll shape (the served shape,
   ggml-cuda.cu:1421) works and is the instrument of record - consistent
   with W8's single-device refusal and the E-085 probe (InitAll, n=3).

## 1. P0 - the multi-die probe harness (committed instrument)

Files: docs/amd-port/tests/rccl_multidie_probe.cu +
docs/amd-port/tests/run_rccl_multidie.sh (launcher: campaign-lock
check-and-hold JSON line with pid, cool-die gate < 60 C on all four dies,
one clean exec'd process per die or --inproc, session log per config,
lock released per config).
Build: hipcc -O2 -x hip rccl_multidie_probe.cu -o rccl_multidie_probe -lrccl
(COMPILE-EXIT:0, ROCm 6.2.0, RCCL 2.20.5+hip6.2).

- Two regimes: multi-process (one rank per die, worker mode, POSIX shm
  uid exchange + host barrier) and --inproc (one process, ncclCommInitAll
  over 4 devices, grouped ncclGroupStart + 4x ncclAllReduce + ncclGroupEnd
  - byte-for-byte the served call shape).
- Two pacings: lockstep (barrier per iteration = isolated boundary
  latency) and burst (N back-to-back grouped calls = the served
  stream-pipelined regime, per-op = elapsed/N).
- Per-rank metrics: HIP-event dev time around that rank's allreduce
  (census-comparable) AND host wall; median/min/p90/avg, two-half AND
  even/odd spread (even/odd is drift-immune - monotonic clock-ramp failed
  the 3% law on plain halves; even/odd passes at 0.01-0.06%), 2-us bucket
  histogram per rank, correctness gate (exact-sum pattern) before timing.
- Instrument findings: workers must be exec'd from a parent that has not
  loaded HIP is NOT sufficient to fix multi-process init (see 0.5); the
  inproc mode is the of-record instrument.

## 2. A1 - baseline + residual localization

Sessions of record: rccl_multidie_a1_default_20260923_152122.log (env-empty
default, inproc, 2000 iters) and rccl_multidie_a1_inproc80k_20260923_145957.log
(first full session; numbers below). Die map: HIP die0=card1 PCI 05:00.0,
die1=card3 08:00.0, die2=card0 0d:00.0, die3=card4 10:00.0 (card2 15:00.0
is the dead die).

| arm | die0 | die1 | die2 | die3 | spread | burst per-op |
|-----|------|------|------|------|--------|--------------|
| lockstep dev med (us) | 124.4 | 116.1 | 117.3 | 124.2 | x1.07 | - |
| burst per-op dev (us) | 103.2 | 103.2 | 103.2 | 103.2 | x1.00 | 103.2 |

Reading against the foundations:
1. Served census medians 191.6/210.8/151.9/128.8 (x1.64) vs isolated
   x1.07 (lockstep) / x1.00 (burst): the asymmetry is schedule-induced
   wait, NOT wire cost. W8's T(die) residual splits into T_wire (uniform,
   ~34 us over the ring floor) and T_wait(die) (the served-only part,
   59-141 us, set by per-die compute arrival jitter across 65 layers).
2. Implication for the wall: the 31.2 ms/round tax pays max-over-dies of
   (algo + wire + wait). Transport tuning moves the uniform ~103 us
   component only; the wait component needs compute-side balance (out of
   this desk's scope; named for the coordinator).
3. The isolated burst cost (103 us) vs served pooled median (165.6 us)
   confirms ~60 us/op of interleave wait even on the median boundary.

## 3. A2 - env sweep (17 + 10 configs, all sessions committed)

All inproc, 2000 iters, 80 KB unless noted; per-rank medians in the
session logs; every config passed the even/odd spread law except two
20 KB cells marked FAIL in-log (5.6% halves; even/odd ~2.6%).

| config | burst per-op (us) | worst-die lockstep (us) | verdict |
|--------|-------------------|--------------------------|---------|
| default (control x2) | 103.2-103.5 | 125.0-126.5 | baseline |
| NCCL_PROTO=LL / LL128 / Simple | 103.2-106.3 | 120.6-125.0 | neutral |
| NCCL_ALGO=Ring | 103.4 | 124.3 | neutral (default) |
| NCCL_ALGO=Tree (+Simple) | 149.9-150.3 | 169.5-169.8 | REJECTED: worse x1.45, die gradient x1.48 |
| NCCL_ALGO=Tree + LL128 | 103.4 | 124.7 | neutral (LL128 flips tree back to ring-class perf) |
| NCCL_MIN_NCHANNELS=1 | 103.5 | 125.8 | neutral |
| NCCL_MIN_NCHANNELS=4 | 92.2 / 91.6 / 91.9 (x3) | 113.3-114.4 (x3) | WIN: -11.0..-11.5% burst, -9.6..-10.4% worst |
| NCCL_MIN_NCHANNELS=5/6/8/12 | 99.1-100.6 | 112.4-117.0 | worse than 4 AND spread widens to x1.17-1.18 |
| MIN=4 + MAX=4 | 91.6 | 113.3 | same as MIN=4 |
| MIN=4 + NTHREADS=1024 (x2) | 91.9-92.0 | 113.5-114.1 | same as MIN=4 (threads neutral) |
| MIN=4 + PROTO=LL128 | 92.3 | 113.9 | same as MIN=4 |
| MIN=4 + PROTO=LL | 98.7 | 112.0 | LL slightly worse pipelined |
| NCCL_NTHREADS=256/1024 | 103.1 | ~124 | neutral |
| NCCL_P2P_DISABLE=1 | 103.3 | 124.3 | neutral (no P2P existed) |
| NCCL_SHM_DISABLE=1 | 191.4 | 213.8 | REJECTED: +85% - SHM IS the transport |
| RCCL_MSCCL_ENABLE=1 | 103.3 | 124.7 | neutral (no msccl algo selected at this size) |
| RCCL_LL128_FORCE_ENABLE=1 | 103.0 | 124.3 | neutral |
| NCCL_IGNORE_CPU_AFFINITY=1 | 103.2 | 124.6 | neutral |
| 20 KB (ne 5120) default | 68.7 | 105.9 | draft class baseline |
| 20 KB + MIN=4 | 67.5 (-1.8%) | 91.7 (-13.4%) | ch4 helps the draft class too |

INFO fingerprints (sessions a3_fp_default / a3_fp_ch4): default = 2
channels, ring 0-1-2-3, every hop "via SHM/direct/direct"; ch4 = 4
channels, same ring order, same SHM hops. Channel count does not touch
the ring's per-element reduce order.

## 4. A3 - served-arm spec + verdict

VERDICT: the arm is PURE ENV, no code change. The mission's >= 15%
worst-rank bar was not met (-9.6..-10.4% lockstep, -11% pipelined), so
this desk records a PARTIAL win and does NOT claim the 12.6 ms
flattening prize - that prize is compute-balance-bound (section 2), not
transport-bound. What the env arm CAN cash is the uniform wire
component: ~11.3 us/op x 136 boundaries = ~1.5 ms/round upper bound
(~1.2% decode at 23.26 t/s) IF the served boundary pays the same
transport share the probe isolates; the wait share may dilute it.
The provenance-gated 200k battery decides.

Served arm spec (coordinator window, per receipt W8 section 6 pattern):
1. Env set: NCCL_MIN_NCHANNELS=4 exactly (nothing else - Tree, LL,
   >4 channels, threads, MSCCL all measured neutral-to-worse).
2. Provenance: same sha, clean tree, binary+config hashes stamped
   (guard_battery.py law E-112); launch config records the env var.
3. Determinism gate: two boots per arm, same-sha outputs (fp-dust class
   across ARMS is expected under E-085/E-090 convention: channel count
   keeps ring order, but any cross-arm byte diff goes to owner sign-off;
   accept 0.66667 / mean len 3.00 in every cell).
4. Judge per-die: the battery's value is the ROUND wall; the per-die
   NCCL kernel medians should drop ~11 us if the transport share is
   cashed - a quick census diff against TP4_kernel_census_2026-09-23
   confirms mechanism cheaply.
5. Code-level candidate (channel/priority pinning in the RCCL wrapper):
   NOT warranted - the env knob achieves the effect with zero code risk;
   the wrapper (ggml/src/ggml-cuda/ggml-cuda.cu comm paths) already
   inherits NCCL env verbatim. No numerics-class change is proposed.

## 5. A4 - build

- Probe TU: COMPILE-EXIT:0 (hipcc -O2 -x hip, gfx900, -lrccl).
- Full ggml-hip, canonical flags (cmake /home/chris/opt/cmake/bin/cmake,
  -B build-hip, GGML_HIP=ON, Release, GGML_NATIVE=ON,
  CMAKE_HIP_ARCHITECTURES=gfx900, GGML_HIP_RCCL=ON, LLAMA_CURL=OFF):
  BUILD-EXIT:0, 0 warnings in the build log, target ggml-hip linked clean.
  No runtime code was changed by this desk, so the served binary is
  untouched by design.

## 6. Defects found (for the record)

1. MULTI-PROCESS RCCL INIT DEFECT (ROCm 6.2.0 / RCCL 2.20.5+hip6.2,
   gfx900): ncclCommInitRank(world=4) across 4 one-die processes fails
   ncclUnhandledCudaError(1) instantly in every rank. Tried: staggered
   init (3-6 s/rank), RCCL_CLIQUE_IGNORE_TOPO=1, NCCL_LOCAL/GRAPH_REGISTER=0,
   NCCL_SOCKET_IFNAME=lo and default wlo1, fork+exec and shell-exec
   launchers. Single-process ncclCommInitAll over the same 4 dies works
   and is the served shape anyway. Consequence: the campaign's U3 probe
   form-factor is "inproc", not "4 processes"; worker mode remains in the
   committed tool for when the stack is fixed. Sessions:
   rccl_multidie_a1_forkdbg*.log, a1_forkstag_*.log.
2. HALVES-SPREAD INSTRUMENT LAW: monotonic clock-ramp drift fails the W8
   3% halves law on long runs (5.0-6.8%) even when the config is stable;
   the harness now reports even/odd halves (drift-immune, 0.01-0.34% on
   all passing configs). Both metrics are printed per rank.
3. SHM_DISABLE fallback silently degrades 85% - any future "cleanliness"
   env set that disables SHM would cost ~2x on every boundary; recorded
   so nobody ships it.
4. CHECKIN.log carries a future-stamped "ladder complete" line from the
   tp4-bound desk (came in via the base commit); retracted for
   attribution in this desk's 14:52 erratum line. Timestamp hygiene
   matters because the coordinator reads the tail.

## 7. LOG (append-only, newest last)

- 2026-09-23 14:26 desk opened on amd/rccl-transport @ 68766bec1; ledger
  + W8 + round map read. Campaign lock held by mmvq-ldsy (stale pid,
  live desk) - polled, not stolen.
- 2026-09-23 14:39-14:53 P0 harness written (fork + inproc), WIP commits
  be25146ce, c8979d91c. CHECKIN erratum for the foreign future-stamped
  line.
- 2026-09-23 14:53 stale lock removed with evidence (holder pid dead
  since 14:26, ldsy compile-only, dies idle 26-33 C); lock protocol per
  config since.
- 2026-09-23 14:54-15:20 multi-process init defect hunted (6 debug
  sessions); declared a stack defect; inproc = instrument of record.
- 2026-09-23 15:00-15:01 A1 of-record sessions (inproc 80 KB): flat
  x1.07; burst 103.2 uniform; residual does not reproduce.
- 2026-09-23 15:12 commit 1c2d651be (harness hardened + A1 findings).
- 2026-09-23 15:13-15:33 A2 sweep 1 (17 configs) + control: ch4 wins,
  Tree/SHM-off rejected, everything else neutral.
- 2026-09-23 15:29-15:39 A2 sweep 2 (10 configs): ch4 reproduces x3,
  MIN=4+MAX=4 best at 91.6 us; 20 KB class helps too; bar >= 15% NOT met
  -> A3 partial verdict.
- 2026-09-23 15:35-15:36 INFO fingerprints: default 2 channels vs ch4 4
  channels, same ring, same SHM hops.
- 2026-09-23 15:55 A4: configure + build ggml-hip canonical flags
  BUILD-EXIT:0, 0 warnings.
- 2026-09-23 15:58 receipt + ledger E-117a + final commits.
