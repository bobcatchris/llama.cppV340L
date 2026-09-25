# W7 MAINTENANCE WINDOW RUNBOOK — one reboot, every root-gated lever (2026-09-18)

Purpose: apply the three root-gated levers that the campaign measured as real but could not
touch without root, in ONE reboot, then re-leg everything against the banked same-day logs.
Everything here follows R6 (a knob is "applied" only when its observable effect is verified)
and the band discipline (same leg scripts, sidebands on).

## 1. Kernel cmdline — IOMMU passthrough (user's own proven lever from the 5060 Ti box)

Measured state today: AMD-Vi ON, full-translation DMA-FQ mode, 39 groups, NO iommu param in
/proc/cmdline. Affected paths here: PCIe DMA only — AR (~60 ms/chunk), H2D/D2H, RCCL peer
paths. NOT the GEMM 75% (device-local VRAM traffic never crosses the IOMMU). Expected: a few
percent (AR 60 -> toward the ~35 floor), not a multiplier.

  sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt /' /etc/default/grub
  sudo update-grub
  sudo reboot

Verify after reboot:  cat /sys/kernel/iommu_groups/*/type   -> GPU groups read "identity"
(not "DMA-FQ"). If the box refuses to boot or a GPU vanishes: remove the param (single boot
with the grub edit reverted) — the fallback state is today's, no harm done.

## 2. Clock/power pinning — no reboot needed, root once per boot (or a sudoers rule)

Measured tax: sclk droops to levels 2-4 under compute within seconds of bursts (PLOG-043
sidebands; 979 -> 1032 ms steady-chunk creep inside ONE request); mclk parks at 167 MHz at
idle and ramps with hysteresis. Banked scoreboard figure for this lever: 1.3-1.75x on
hot-machine rows. Gemini's search confirmed the knob names (its pp_compute_power_profile
suggestion is ABSENT on this V340 vBIOS — skip).

  echo "high" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
  echo "high" | sudo tee /sys/class/drm/card2/device/power_dpm_force_performance_level
  # card0 = GPU0 (05:00.0), card2 = GPU1 (08:00.0); card1 = NVIDIA display, NEVER touch.

Verify: pp_dpm_mclk parks at level 3 (945 MHz) and pp_dpm_sclk stops showing levels 0-2
under load (sideband sampler confirms). Optional harder pin (bench windows only — raises
idle power): echo manual > ...performance_level; echo 7 > ...pp_dpm_sclk
Alternative to typing sudo per window: one sudoers rule:
  chris ALL=(root) NOPASSWD: /usr/bin/tee /sys/class/drm/card0/device/power_dpm_force_performance_level, /usr/bin/tee /sys/class/drm/card2/device/power_dpm_force_performance_level

## 3. Post-reboot verification + the A/B re-leg (owner runs, no root)

  1. iommu: groups read identity; rocminfo still enumerates 4x gfx900; health endpoint OK.
  2. Re-run the BANKED leg scripts unchanged (same band discipline, same probe grammar):
       bash results/amd/coherence/W7_finbracket_serve.sh /home/chris/artifacts_bin/ninfer-serve_1e03e5af5fb438b9.bin
     Compare vs today's banked logs (same machine, same scripts, same day):
       AR column (49-83 band), gap (92-152), finalize staircase (1373/1703/1716/2051),
       wall means, tok/s class. Sidebands on; the mclk/sclk behavior IS the mechanism check.
  3. Bank: W7_iommu_clocks_row.txt with before/after columns; PLOG entry via plog_append.py.

## What this window is NOT

Not a GEMM lever (the 75% wall stays — that's the ROCm census desk, still in flight).
Not a substitute for the additive fixes (finalize/AR floor/capture still stack on top).
Expected combined effect if both levers deliver: ~116-121 -> ~135-155 tok/s class, and the
droop-dependent variance that poisons cross-window comparisons shrinks — every future number
gets more honest.
