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
