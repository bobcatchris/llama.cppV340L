# W22 DIE-ASYMMETRY DESK (A1 host census + A2 decomposition + A3 ranked candidates + A4 served-arm spec) - 2026-09-24

Desk: DIE-ASYMMETRY DESK (on amd/v340-port-v2 @ 5ad25e902). ZERO-GPU
throughout: the depth-measurement window holds all four dies behind
/tmp/campaign_gpu_boot.lock (untouched); this desk is read-only host
census + banked-evidence decomposition + a served-arm spec for the
coordinator. Mission: the W8 named lever "die asymmetry" - the served
per-die boundary medians 128.8 / 151.9 / 191.6 / 210.8 us (die 3/2/0/1,
wall pays the slowest, 31.2 vs 18.7 ms/round of NCCL kernel time).
Foundation: W8_tp4_boundary_receipt_2026-09-23.md,
W9_rccl_transport_receipt_2026-09-23.md,
W10_split_balance_receipt_2026-09-23.md,
W18_soak_mechanism_receipt_2026-09-24.md (git 9761f0821, amd/soak-mech),
TP4_roundmap_2026-09-23.md, ledger E-110..E-138.

## 0. Headline

1. A1: the host is a SINGLE-NUMA Intel i7-12700K (hybrid 8 P-cores +
   4 E-cores, 20 cpus); all four dies report numa_node -1 and
   local_cpulist 0-19. The four dies split 2+2 across TWO CPU root
   ports (00:01.0 / 00:01.1), each behind a Microchip PM8533 fanout
   switch; both switch uplinks are x8 Gen3, every GPU link x16 Gen3,
   same hop depth, iommu=pt, no ACS redirect. The PCIe fabric is
   SYMMETRIC between the two pairs - topology cannot generate a die
   gradient of the observed size.
2. A2: the +80 us spread is NOT (a) host enqueue/NUMA (enqueue 0.2 us
   no-op class, single NUMA, and W10's per-boundary compute is a pure
   multiplicative per-die rate - host jitter cannot produce that),
   NOT (b) transport (W9: isolated 4-die allreduce flat x1.07
   lockstep / x1.00 burst; ring estimate in the served census uniform
   114.5-128.9 us), NOT (d) RCCL channels (ch4 changes the uniform
   transport term only; SERVED it LOSES -12.64 pct @200k / -6.16 pct
   @10k, E-119). The evidence supports (c) PER-DIE CLOCK/POWER STATE:
   W10's rate law t_d = 0.893/0.846/0.964/1.000 with the idle-sclk
   ordering matching (die 3 lowest, 1085 vs die 1's 1249 MHz), and
   W18's far-die soak law (0d:00/10:00 latch 775/560 MHz bands under
   soak; decode tracks live sclk; hot-but-unthrottled cells run full
   speed; junction does not discriminate, clocks do).
3. W8's raw per-die medians are wait-redistribution (W10 correction):
   waits 74.0/90.8/37.6/0.0 us (die 0/1/2/3); die 3 is the slowest
   compute die, arrives LAST on 58 pct of boundaries, and pays pure
   ring; the wall pays last-arriver duration + max compute. Under
   soak the rate spread WIDENS toward the far dies (W18), so any
   static per-die calibration is state-dependent.
4. The byte-rebalance form of the lever is CLOSED served (E-125:
   --tensor-split 1.04,1.04,1.02,1.00 lost BOTH paired cells,
   -1.92/-2.33 t/s). The surviving lever family fixes the rate
   spread AT THE SOURCE: C1 per-die clock floors (root-gated; the
   W18 discriminator queued at E-133), C2 host P-core pinning
   (zero code, one runner line), C3 serve-state rate re-measurement
   (analysis-only, doubles as the E-125 post-mortem). Not one
   surviving candidate is env-expressible - the env class (chan4)
   is served-closed; the remaining levers are system-state and
   wrapper-class.
5. First experiment when the GPUs free: the C1 clock-floor paired
   window (10k decode-only, R/F/F/R interleave + one 200k pair),
   with the 4-die sclk sideband (instrument live since E-136(e))
   as the mechanism witness. Prize band: partial spread halving
   ~3 ms/round (~+2.4 pct); full equalization at the fast-die rate
   ~6.1 ms/round (~+4.8 pct) per W10's wall-compute model; plus
   W18-class anti-decay protecting every later window cell.

## 1. A1 - host topology census vs the die PCI map (2026-09-24 11:05 CDT)

Die map of record (W9): HIP die0..3 = PCI 0000:05:00.0 / 08:00.0 /
0d:00.0 / 10:00.0 (enumeration by PCI order; cardN numbers are NOT
stable across reboots - PCI is). Sources: lspci -t -v,
/sys/bus/pci/devices/*/ (numa_node, local_cpulist, current/max link),
lscpu -e, /proc/cmdline.

| die | PCI | root port | switch path | NUMA | link | nearest cores |
|-----|-----|-----------|-------------|------|------|---------------|
| 0 | 05:00.0 | 00:01.0 | 01:00.0 PM8533 -> 02:00.0 -> 03:00.0 -> 04:00.0 | -1 (single node) | x16 Gen3 (switch up x8) | all of 0-19 |
| 1 | 08:00.0 | 00:01.0 | 01:00.0 PM8533 -> 02:01.0 -> 06:00.0 -> 07:00.0 | -1 | x16 Gen3 (up x8) | all of 0-19 |
| 2 | 0d:00.0 | 00:01.1 | 09:00.0 PM8533 -> 0a:00.0 -> 0b:00.0 -> 0c:00.0 | -1 | x16 Gen3 (up x8) | all of 0-19 |
| 3 | 10:00.0 | 00:01.1 | 09:00.0 PM8533 -> 0a:01.0 -> 0e:00.0 -> 0f:00.0 | -1 | x16 Gen3 (up x8) | all of 0-19 |

- CPU: 12th Gen i7-12700K, 1 socket, 1 NUMA node, 20 cpus online:
  8 P-cores (cpu 0-15, HT siblings) + 4 E-cores (cpu 16-19, no HT).
  There is no CCD concept on this host - the real topology split is
  P-vs-E core class. governor-class scaling ~70 pct at census time.
- Fabric symmetry check: pair A (dies 0+1) under root 00:01.0, pair
  B (dies 2+3) under root 00:01.1; both fanout uplinks x8 Gen3
  (7.88 GB/s); both downstream GPU links x16 Gen3 (8.0 GT/s);
  identical hop depth (5 PCIe hops to each GPU); iommu=pt in
  cmdline (DMA passthrough, no translation path divergence); no ACS
  features on the roots (lspci -vv: ACSCtl absent). Note the
  asymmetry SPLIT BY PAIR in the W8 census (pair A 191.6/210.8 vs
  pair B 151.9/128.8) does NOT follow any fabric capability
  difference - the two subtrees are twins.
- Kernel cmdline carries amdgpu.ppfeaturemask=0xffffffff - the
  pp_od clock surfaces C1 needs are already enabled.
- 8-thread unpinned server placement: no launcher pins anything
  (launch_tp3_200k.sh, run_postreboot_hardened.sh: no taskset /
  numactl / affinity lines). Default CFS placement may park any of
  the 8 server threads + httplib workers + RCCL proxy/spin threads
  on the 4 E-cores (cpu 16-19, ~3-4x slower for this class).
- Thread-affinity hypothesis for "why far dies pay +80 us":
  REFUTED as the primary cause. (i) W8 P0: enqueue 0.2 us, launch
  ~12 us, host runs ahead stream-pipelined - there is no host-side
  per-op tax to skew per die; (ii) W10 A1: per-boundary device
  compute = t_d x 579.4 us on ALL FOUR dies to 0.1 pct - a
  multiplicative device-side law that host jitter cannot generate;
  (iii) the fabric is symmetric (above). E-core residency is at
  most a small constant-class jitter term (candidate C2), not a
  per-die gradient.

## 2. A2 - decomposition of the asymmetry from the banked evidence

Candidate causes vs evidence of record:

| cause | verdict | decisive evidence |
|-------|---------|-------------------|
| (a) host enqueue / NUMA bounce | EXCLUDED | W8 floors (enqueue 0.2 us no-op class); single NUMA census (sec 1); W10 multiplicative rate law (kernel durations scale by a per-die constant at equal bytes) |
| (b) board-position transport (trace/retimer) | EXCLUDED | W9: isolated lockstep x1.07 / burst x1.00 vs served x1.64; W10: ring estimate uniform 114.5-128.9 us; fabric symmetric (sec 1); W9 fingerprint: every hop via SHM/direct/direct, P2P never existed (NCCL_P2P_DISABLE neutral), SHM_DISABLE +85 pct |
| (c) clock / power state | SUPPORTED - mechanism of record | W10: t_d = 0.893/0.846/0.964/1.000, stable across window halves, die 3 slowest on EVERY MMVQ class, idle sclk die3 1085 vs die1 1249 MHz; W18: far dies 0d/10 latch 775/560 MHz bands under soak, decode t/s tracks live sclk (23.2-23.6 iff act-mean >= ~1190 MHz; 17.3-17.7 iff 1068-1135; <=991 MHz duty 0-20 vs 44-53 pct), hot-but-unthrottled 10k cells still 21.8-22.5, junction FLAT 83-85 C on all four dies while clocks split, W15 steady state 0d+10 at 788-980 vs 05/08 at 1094-1465 MHz; combowin_chan4_200k_thermal.log shows 560-1500 MHz oscillation through one served 200k battery |
| (d) RCCL channel / lane assignment | EXCLUDED as an asymmetry source | W9: ch4 lifts the UNIFORM transport term only (isolated -11 pct); served chan4 arm FAILED (-12.64 pct @200k, -6.16 pct @10k, E-119) - the probe win did not transfer (2 extra spinning channels/die plausibly starve compute occupancy and slow arrival); Tree REINTRODUCES a die gradient x1.48 (W9) - ring is correct |

Reading:
1. The served per-die boundary medians are duration = ring + wait.
   W10's END-calibrated waits: 74.0 / 90.8 / 37.6 / 0.0 us
   (die 0/1/2/3). Die 3 = slowest compute, arrives LAST on 58 pct of
   boundaries, pays pure ring (its 128.8 median IS the ring+wire
   floor class). The WALL pays last-arriver duration + max compute
   per boundary - not the pooled median, and not die 1's raw sum
   (W10 defect note 2 corrects the W8 framing).
2. Under soak the rate spread WIDENS toward the far dies (W18), so
   the arrival order and the wait distribution drift WITHIN a
   window. Any static per-die calibration taken in one thermal
   state is invalid in another - this is the leading explanation
   for why the W10 ratio vector (calibrated on the cool traced
   window) failed served in E-125 (paired -1.92 / -2.33 t/s, both
   negative at minimal soak for pair 1: the model did not transfer,
   and soak drift alone cannot be the whole story - see C3).
3. The one cheap probe that settles the residual question: serve-
   state 4-die sclk sideband (EXISTS since E-136(e) - the 18-col
   sampler keys on the 4 die PCIs) correlated with per-die MMVQ
   medians from the same window's trace. If t_d(t) tracks
   sclk_d(t)/sclk_fast(t) ~1:1, cause (c) is confirmed at serve
   state and C1's prize is real. If rates spread while clocks are
   pinned equal, the remaining spread is silicon/HBM-class and the
   die-asymmetry lever closes with evidence (that is also C1's
   mechanism kill criterion).
4. Artifact notes: docs/amd-port/results/w14_traces/ are the W14
   wide-launch kernel matrices (per-kernel stats, e.g.
   flash_attn_tile) - no per-die NCCL boundary artifact beyond what
   W10 already mined from TP4_kernel_census_2026-09-23; the W8
   probe logs' floors re-verified
   (tp4_boundary_probe_d3_20260923_142626.log: KLAUNCH ~11-12 us
   size-flat, RING1K n=4 80 KB 69.5 us, n=2 51.5 us).

## 3. A3 - ranked candidates (prize x cheapness x risk)

Mission bar for reference: pulling the slow-die boundary cost
halfway to the fast die's is ~5-6 ms/round (~+10 pct short-prompt
decode on the 24.55 of-record). The only surviving mechanism class
that can pay that bar is C1's.

### C1 - PER-DIE CLOCK FLOORS (root-gated). RANK 1

- Mechanism: remove the low-sclk latch band on the far dies
  (0d:00/10:00) so the per-die compute-rate spread t_d cannot open
  mid-window. Two independently priced effects:
  (i) arrival compression - if t_d compresses toward the fast die's
  rate, W10's wall-compute model prices full equalization at
  579.4 -> 534.2 us/boundary = ~45.2 us x 136 = ~6.1 ms/round
  (~+4.8 pct decode); a partial (spread halved) ~3 ms/round
  (~+2.4 pct);
  (ii) anti-decay - W18: decode decays -26 pct through chained
  cells as far-die sclk latches; a floored window keeps every cell
  decision-grade (instrument repair, compounds with every future
  window).
- Exact ops (root, once per boot BEFORE the window; resolve card
  from die PCI via the runner's die_cards() pattern):
    primary, per die card c:
      echo high > /sys/class/drm/$c/device/power_dpm_force_performance_level
      (revert: echo auto > ...)
    finer variant (Vega10 powerplay; ppfeaturemask already 0xffffffff):
      echo "s 0 1200" > /sys/class/drm/$c/device/pp_od_clk_voltage
      echo "c"     > /sys/class/drm/$c/device/pp_od_clk_voltage
      (sets OD_SCLK min = 1200 MHz, boost above still allowed;
       revert: echo "r" then "c")
  power_dpm_force_performance_level=high is the documented,
  simply-reversible knob; use the pp_od min-sclk floor only if
  level=high trips a thermal artifact.
- A/B served (paired interleaved per the campaign law):
  arms F (floored) vs R (regular regress), 10k decode-only cells,
  order R F F R (2 pairs minimum), position-1 law + void_gate.py on
  every cell; then ONE 200k pair (F vs R) for the anti-decay claim
  (the 200k needle cell is the soak source - the floored arm should
  hold through it). Mechanism witness per cell: hwmon freq1_input
  for all four dies >= 1100 MHz during decode cells (the 18-col
  sideband covers die 4 since E-136(e)); stamp the four
  power_dpm_force_performance_level values into the arm-identity
  header.
- Kill criteria (any one closes C1): paired 10k delta < +0.5 t/s at
  matched position; OR clocks verified pinned but a traced cell's
  t_d spread does not compress; OR mem/junction emergencies or new
  throttle artifacts appear. On a mechanism kill, cause (c) is
  refuted at serve state and the die-asymmetry lever CLOSES
  entirely (C3 becomes the closing post-mortem, no further arms).
- Risk: root-gated system state (not process env); the 110 W/die
  PPT cap is already at max (W18) - floors do not raise power, they
  prevent DOWN-clocking; watch HBM/mem temps (crit 95 C).

### C2 - HOST P-CORE PINNING (zero code, one runner line). RANK 2

- Mechanism: keep the 8 server threads + httplib workers + RCCL
  proxy/spin threads off the 4 E-cores (cpu 16-19). Constant-class
  jitter term only - it cannot produce the observed per-die
  gradient (A1 verdict), but decode at 24.55 t/s is 40.7 ms/token
  and the test is nearly free. Primary mask 0-15 (P-cores with HT);
  variant 0-7 (8 threads on 8 physical cores, HT siblings free) if
  0-15 shows anything.
- Exact delta (runner is coordinator-owned; EXTRA_ENV cannot
  express affinity): in cell(), wrap the binary:
    env HIP_VISIBLE_DEVICES=0,1,2,3 $BASEENV $EXTRA taskset -c 0-15 "$BIN" ...
  (numactl is a no-op on this single-node host - do not bother).
- A/B: paired interleaved 10k cells T vs R, order R T T R, 2 pairs.
- Kill criteria: paired 10k delta < +0.5 t/s. Risk ~zero.
- GPU-to-die affinity of the server process: NOT APPLICABLE in the
  served shape - one process owns all four dies (ncclCommInitAll,
  ggml-cuda.cu:1421); the only affinity axis is host CPUs (this
  candidate). Recorded so it is not re-derived.

### C3 - SERVE-STATE RATE RE-MEASUREMENT (analysis-only). RANK 3

- The E-125 post-mortem and the gate for ever reopening a rebalance
  arm. From the next traced window (or a 60 s rocprof cap on one
  battery): recompute t_d at SERVE state, correlate with the sclk
  sideband, and diagnose the model-to-served failure. Candidates to
  discriminate: (i) soak drift of t_d (cool-trace vector stale),
  (ii) tile quantization at changed row widths (per-byte time is
  not constant across slice widths), (iii) +0.35 GB VRAM on die 1,
  (iv) rotation lumping. Tensor-split stays CLOSED (E-125) unless
  the owner reopens it on this evidence.
- Kill/exit: if serve-state t_d matches the cool-trace vector
  within a few percent, the rebalance failure is structural (not
  rate drift) and ratio-based rebalancing is permanently closed.

### C4 - CLOSED, DO NOT RE-RUN (record)

- RCCL env class: ch4 probe win -11 pct did NOT transfer served
  (E-119: -12.64 pct @200k, -6.16 pct @10k, prefill -10.12 pct,
  mtp canary tripped); Tree worse + reintroduces a gradient (W9);
  SHM_DISABLE +85 pct (W9); PROTO LL/LL128/Simple, NTHREADS, MSCCL,
  P2P-off, IGNORE_CPU_AFFINITY all neutral (W9, 27 configs). The
  env lever on this lever-family is exhausted.
- Tensor-split rebalance: E-125 REJECTED (paired -1.92 / -2.33).
- Boundary count + clustering: W8 A2/A3 NEGATIVE (do not revisit).

### C5 - OBSERVATION, NO ARM

- P2P does not exist on this stack; SHM is the transport and its
  wire term (+34 us over the 69.5 us ring floor) is UNIFORM - not
  an asymmetry term. If a future ROCm/RCCL stack enables P2P across
  the PM8533 pairs (and fixes the multi-process init defect, W9
  defect 1), that uniform term becomes attackable separately.
  Coordinator note only.

## 4. A4 - served-arm spec: exact window deltas, in run order

Run order when the GPUs free (after the depth chain releases the
lock, fresh boot, position-1 law applies):

1. WINDOW 1 = C1 clock-floor pair (the first experiment).
   Pre-window root step (once per boot):
     for c in die_cards:  # die_cards() resolves DIE_PCIS via /sys/class/drm
       echo high > /sys/class/drm/$c/device/power_dpm_force_performance_level
   Arm sequence (COMBO_CELLS=10k): R F F R, then one 200k pair
   (F then R). run_combined_window.sh deltas:
     - arms "cf" (with root step) and "regress" (without) - the
       root step is system state, NOT EXTRA_ENV; EXTRA_ENV stays
       empty for both arms;
     - add to the cell() identity block: `cat /sys/class/drm/*/device/power_dpm_force_performance_level` output + per-die freq1_input, so the arm's system state is stamped.
   Decision: paired deltas per E-119/E-125 convention; ratchet only
   on GREEN (E-137 law); kill criteria per C1 above.
2. WINDOW 2 = C2 taskset pair (independent of Window 1's verdict).
   Runner delta: the taskset -c 0-15 wrap of $BIN in cell();
   arms "ts" vs "regress", R T T R, COMBO_CELLS=10k.
   Kill: paired < +0.5 t/s.
3. CONTINUOUS = C3: the sideband + a 60 s trace cap on any one cell
   of Window 1; verdict into the ledger.

Gates that apply to every cell: provenance stamps (E-112),
arm-identity freshness gate (E-133(c), already in the runner),
void_gate.py adjudication + position-1 law (E-133/E-135),
180 s inter-cell settle (E-134). Numerics class: C1 and C2 are
numerics-NEUTRAL (clock/affinity state changes no kernel order, no
ring order, no sum order) - no owner sign-off beyond the standard
battery; C3 changes nothing (read-only).

Prize table (honest, on the 24.55 of-record):

| candidate | mechanism | expected delta | cost | risk |
|-----------|-----------|----------------|------|------|
| C1 full | floors collapse t_d spread | ~6.1 ms/round ~ +4.8 pct; anti-decay up to W18's -26 pct class on late cells | root step | med (thermal watch) |
| C1 partial | spread halves | ~3 ms/round ~ +2.4 pct | root step | med |
| C2 | E-core avoidance | 0-1 pct class | 1 runner line | ~zero |
| C3 | diagnosis only | 0 direct; gates reopening | trace cap | zero |
| C4 (closed) | - | chan4 -12.64 pct / split -2.1 t/s measured | - | - |

## 5. Identity

No runtime code touched; docs-only desk. Host reads were passive
sysfs/lspci files; no GPU process launched; /tmp/campaign_gpu_boot.lock
untouched (held by the depth chain); guard_battery.py and the
/home/chris runner scripts read-only (their deltas are specified
here for the coordinator, not applied).

## LOG (append-only, newest last)

- 2026-09-24 11:07 CDT desk opened on amd/v340-port-v2 @ 5ad25e902;
  ZERO-GPU behind the depth window's lock (polled, not touched).
  Foundations read: W8 + W9 + W10 receipts, W18 (git 9761f0821,
  amd/soak-mech - not in this branch's tree), round map, ledger
  E-119..E-138. Banked-served-facts list extracted first: chan4
  arm served FAILED (combowin_chan4 cells, E-119); tensor-split
  rebalance served REJECTED (E-125 paired cells) - both closed,
  not re-litigated.
- 11:12 A1 census (lspci -t -v, sysfs numa_node/local_cpulist/
  links, lscpu -e, /proc/cmdline, ACS check): single NUMA, hybrid
  8P+4E, symmetric 2+2 fabric under twin roots with PM8533
  fanouts; ppfeaturemask=0xffffffff already set (C1 surfaces
  live). Thread-affinity hypothesis refuted as primary cause.
- 11:20 A2 decomposition: W10 END-calibrated waits + rate law +
  W18 sclk correlation name (c) clock/power state; W9 flat
  isolation excludes transport; combowin_chan4_thermal.log shows
  560-1500 MHz oscillation served; W14 traces checked (kernel
  matrices, no per-die NCCL artifact); W8 probe floors
  re-verified from the session log.
- 11:24 part 1 commit 67b75992b (sections 0-2).
- 11:30 part 2 commit 2e27d757b (sections 3-5: C1/C2/C3 ranked,
  C4 closed-record, C5 observation, window deltas + kill
  criteria). Desk deliverable complete: the coordinator's first
  run is the C1 clock-floor paired window (R F F R 10k + one
  200k pair) with the 4-die sclk sideband as mechanism witness.
