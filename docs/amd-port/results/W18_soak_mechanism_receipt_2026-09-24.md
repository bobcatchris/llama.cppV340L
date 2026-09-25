# W18 SOAK-MECHANISM DESK RECEIPT (2026-09-24)

Desk: wt-soak on amd/soak-mech (base e116dbb1d). Phase 1 ZERO-GPU: mine the
banked thermal sideband logs + sysfs sensor census; name the mechanism behind
the ~20-minute chained-window decode decay (23.3 -> 17.4 t/s on identical
code, E-127/E-128), or narrow to a decisive discriminator. Instrument:
docs/amd-port/scripts/soak_timeline.py; output: W18_soak_timeline_2026-09-24.txt.
Inputs: /home/chris/hb_p{0a,1a,0b,1b}_thermal.log (E-128 hardened boot,
5 s cadence), /home/chris/combowin_*_thermal.log (E-119 four-arm window +
E-125/E-126 10k chains), W15_deepcensus_2026-09-23.thermal.log (4-die, 20 s),
guard_battery.py sampler source, /sys/class/hwmon + drm sysfs (live).

## 0. MAPPING AND INSTRUMENT GAPS (name before use)

- The 5 s sideband samples rocm-smi card0/card1/card2 = PCI 05:00 / 08:00 /
  0D:00 ("dies 1/2/3"). DIE 4 (10:00) IS NEVER SAMPLED. Per W15, 10:00
  throttles together with 0D:00; its state in the hb windows is inferred, not
  measured. FIX for the next session: add the 4th card to the sampler.
- "c*_mem" IS an HBM temperature channel (hwmon temp3, label "mem") - it has
  been recorded all along but never used in any verdict.
- rocm-smi cardN = PCI order (verified against hwmon values), stable across
  the reboots in this evidence set; the E-120 card-number lesson does not
  bite here because rocm-smi enumerates by PCI, unlike DRM cardN.

## 1. P1 DECAY TIMELINE (the E-128 hardened boot, identical code p0a vs p0b)

Battery shape per cell: boot -> prefill_guard (~15 s) -> decode_guard
(48-57 s, 10k prompt, 128 tokens) -> canary -> determinism -> needle_recall
(127-208 s at 200k - THE SOAK DRIVER); inter-cell gap ~96-100 s (teardown +
boot, light load) leaves the card at junc 47-62 C / mem 50-66 C at next
start (vs 27-35 C on the fresh boot).

  cell | arm    | decode | prefill | start junc (05/08/0D) | die3 sclk during decode
  p0a  | ub512  | 23.33  | 214.59  | 28/27/31              | 1138-1500 (act-mean 1192), no pins
  p1a  | ub1024 | 19.85  | 223.25  | 48/49/60              | 775-1500 (act-mean 1068), 50% <= 991
  p0b  | ub512  | 17.35  | 211.57  | 50/52/62              | 560-1350 (act-mean 1072), 53% <= 991, latches 775 from ~91 s
  p1b  | ub1024 | 17.71  | 222.12  | 51/52/61              | 560-1500 (act-mean 1135), 44% <= 991

- Decode position decay -26% (23.33 -> 17.35) on IDENTICAL code; prefill
  position decay -1.4% (214.59 -> 211.57) in the same cells. The decay is
  decode-specific; compute-bound prefill is almost untouched.
- Die 0D:00 latches into the 775 MHz band mid-cell (p0b elapsed ~91 s, edge
  84-85 C, mem 89 C) and does not recover in-cell; one 560 MHz sighting.
  Dies 05:00/08:00 never pin in any cell (1138-1500 throughout).
- E-127 (pre-hardening boot, 8 h uptime) shows the same shape:
  23.30 / 21.31 / 16.32 / 16.85 - the decay is independent of noretry/SVM
  (already E-128's verdict; the mechanism below is the other half).
- W15 steady state (47 min of traced 4-die load, 20 s cadence, all 4 dies):
  0D:00 + 10:00 sit at 788-980 MHz while 05:00/08:00 hold 1094-1465, with
  junctions FLAT at 83-85 C on all four dies. Junction does not
  discriminate; clocks do. The far dies (0D:00/10:00) are the throttling
  class on this board in every window examined.

## 2. THE CORRELATION (decode tps vs die-3 clock state in the decode window)

  decode tps band        | die3 sclk act-mean | die3 <= 991 MHz duty | die3 mem (mean/max)
  23.2-23.6 (fresh card) | 1192-1331          | 0-20%                | 57-76 / 65-81
  21.8-23.0 (hot 10k)    | 1217-1301          | 0-20%                | 74-85 / 84-89
  19.5-20.3 (soak pos2-4)| 1068-1211          | 18-41%               | 71-85 / 78-93
  17.3-17.7 (deep soak)  | 1068-1135          | 44-53%               | 74-85 / 85-95

Key discriminator cell class: the 10k cells that ran 2-3 min AFTER a
heavy 200k battery (card start junc 46-47, mem reaching 84-88 during the
cell) still decode 21.8-22.5 with die3 at FULL clocks. Hot alone is not
slow; throttled is slow. tps tracks the live clock state, and the clock
state is reached only by sustained heavy load (minutes-class soak).

## 3. RECOVERY EVIDENCE (kills the driver-state hypothesis)

- E-119 10k cells: 22.46-22.52 decode 2-3 min after their own 200k battery
  soaked mem to 90-96 C - same boot, later processes, full speed.
- E-125 s0b 23.51 after a ~13 min mostly-idle gap in a chain that had been
  booting cells since 19:24; the 75-100 s gaps never restore (start temps
  stay 47-62 C), the >= ~10 min gaps do.
- E-127 pos-1 23.30 on an 8-hour-old boot (idle overnight); E-128 pos-1
  23.33 on the fresh hardened boot.
=> Nothing accumulates across process lifecycles; the state variable is
   thermal, clears on idle with a minutes-class time constant, and the
   96-100 s window-gap settle is simply too short to clear it.

## 4. A1 SENSOR CENSUS (all 4 amdgpu dies, identical channels; hwmon5-8 =
   05:00/08:00/0D:00/10:00)

  channel           | label    | value idle      | threshold files
  temp1_input       | edge     | 22-25 C         | crit 85000, emergency 90000 (mC)
  temp2_input       | junction | 24-28 C         | crit 105000, emergency 110000
  temp3_input       | mem      | 26-33 C         | crit 95000, emergency 100000
  freq1_input       | sclk     | 9 MHz (SM clock)| none
  freq2_input       | mclk     | 167 MHz         | none
  in0_input         | vddgfx   | 762-787 mV      | none
  power1_input      | PPT      | 4-6 W           | power1_cap = cap_max = 110 W (cap_min 0) - the 110 W/die cap is AT MAX, unraisable
  pwm1/fan1         | fan      | pwm 24/255, enable=2 (auto) | tach reads implausible (15.2M) or 0 - no usable fan readout/control surface

- HBM/MEM SENSOR EXISTS (temp3 "mem", exposed as rocm-smi "Sensor memory"):
  the campaign has recorded HBM temperature in every thermal log since the
  sideband was introduced. Threshold lore: temp3 crit 95 / emergency 100 C
  (driver-visible); HBM2 refresh-escalation band typically ~85-95 C device.
- NOT exposed on this stack (Vega10/gfx900): per-stack HBM temps, VRM/hotspot
  sensors, throttle-reason file (gpu_metrics is Navi+), per-die board power.
- pp_dpm_sclk/mclk, pp_od_clk_voltage, pp_sclk_od/pp_mclk_od, pp_table EXIST
  but read EMPTY for unprivileged users (root:root 0400 mode content; needs
  root - the OD clock-floor arm is a root-gated action, coordinator/Chris).
- power_dpm_force_performance_level = "performance" (already max);
  amdgpu noretry = 1 (hardened boot confirmed).
- spd5118 hwmon devices = DDR5 DIMM temps (system RAM, not GPU).

Which sensor COULD explain a bandwidth decay invisible at junction 85 C:
junction (105 C crit) cannot; edge (85 C crit) matches the pin onset
(die3 edge sits at 84-85 C whenever latched, but near dies at the same edge
do not pin - per-die limit asymmetry); mem (95 C crit, 87-95 C peaks in
deep-soak cells) is in the refresh-escalation band at deep soak. The true
trigger may be an unexposed VRM/board sensor on the far dies.

## 5. A2 MECHANISM VERDICT: FAR-DIE SOAK

Sustained heavy load soaks the board; the two FAR dies (0D:00 + 10:00) -
which idle and soak 3-8 C hotter than 05:00/08:00 at every position in the
logs - down-step sclk into a 991/775/560 MHz throttle band and latch there
while the near dies hold 1138-1500. TP4 decode is paced by the slowest die,
so decode t/s tracks the far-die clock state (section 2 table); prefill is
compute-bound and barely moves. Recovery is passive cooling with a
minutes-class time constant - no reboot needed, but the 96-100 s cell gap
never clears it. This is thermal power-management throttling (candidate a),
refined: the throttling is per-die, position-dependent, and NOT triggered by
the reported junction.

Honest bound + residual: sampled sclk shrink (-10..-15% on the far dies at
5 s cadence) explains the bulk but likely not all of -26%; aliasing between
5 s samples, an unlogged mclk derate, or HBM refresh-band entry (mem 87-95 C
at deep soak) could carry the remainder. Pure-HBM-refresh (candidate d) is
REFUTED as the primary mechanism (hot-but-unthrottled 10k cells decode full
speed at mem 84-88 C) but survives as a possible contributor at deep soak.
Power-sag (candidate b) has no direct evidence and PPT was never logged;
cap is at cap_max. Driver state (candidate c) is REFUTED (section 3).

## 6. DECISIVE CHEAP DISCRIMINATORS (for the later instrumented session)

1. ONE 1 s CADENCE LOG through a chained battery window: per-die
   sclk + mclk + power1_input (PPT) + vddgfx + edge/junction/mem, ALL FOUR
   dies (fix the sampler card gap). Reading: mclk at max while sclk pins =>
   clock-policy throttling (case closed); PPT pinned at 110 W => power-limit;
   mclk down => bandwidth-side (mclk/refresh) term confirmed.
2. ROOT-GATED CLOCK-FLOOR ARM (one command class, Chris-run): in a decayed
   window force sclk floor via pp_od_clk_voltage / pp_sclk_od (~1200 MHz
   floor on all dies). t/s restores to ~23 while temps unchanged => far-die
   sclk throttle is the whole mechanism; no change => bandwidth term owns it.
3. Optional: dump pp_dpm tables (root) before/after decay to see the
   available states and the active level.

## 7. SERVED-ARM SPEC / CHEAP OPERATIONAL FIX (coordinator)

1. COOLDOWN GATE (no code, scheduling only): bank decision cells at
   position 1 or after >= 5-10 min idle; never bank a 200k-cell decision
   from positions 2+ of a chained window (mechanistic upgrade of the E-119
   soak law; the 96 s gap is measured insufficient, ~13 min is sufficient).
2. OBJECTIVE VOID GATE (no code, from existing sideband): a cell is VOID if
   die-3 sclk act-mean in its decode window < ~1150 MHz or <= 991 MHz duty
   > 20% (the soak_timeline.py table computes it; computable retroactively
   for any window already banked).
3. REORDER, DO NOT LENGTHEN: the 127-208 s 200k needle_recall guard at the
   END of each battery is what soaks the card for the NEXT cell; shortening
   it or running decision cells before it buys recovery for free.
4. PHYSICAL (owner-side): dies 3/4 run 3-8 C hotter at idle and soak first;
   airflow/position fix on the far side of the board is the real cure. The
   110 W/die PPT cap is already at cap_max; fan pwm is auto with no usable
   tach - no software fan lever found.

## 8. PROVENANCE

Zero-GPU desk: existing logs + sysfs/rocm-smi reads only. All tps numbers
are the ledger-banked E-125/E-126/E-127/E-128 values (provenance-stamped in
their own receipts); thermal logs are the guard battery's own sideband
(guard_battery.py ThermalSampler, 5 s cadence) and the W15 desk logger
(20 s cadence). Analysis instrument committed in-tree; regenerate the
timeline dump with: python3 docs/amd-port/scripts/soak_timeline.py.
