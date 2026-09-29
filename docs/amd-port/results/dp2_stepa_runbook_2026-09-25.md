dp2_stepa_runbook_2026-09-25.md — DP-2 STEP A reboot and post-boot verification

# DP-2 STEP A RUNBOOK - ACS clear boot + first P2P verdict
# 2026-09-25. Context: DRIVER_PROGRAM_PLAN.md (DP-2), receipt
# dp_p0_p2p_gates_2026-09-25.md (root cause: ACS redirect bits + P2PDMA
# whitelist). Session banked everything before the reboot; this file is the
# post-boot pickup point.

## WHAT IS INSTALLED ON THE BOX (persistent, root-owned)

- /etc/systemd/system/dp-acs-p2p.service (enabled; runs each boot BEFORE
  display-manager): clears ACS redirect (16-bridge list, cap+6 register),
  unloads+reloads amdgpu through host modprobe, logs to
  /var/log/dp_acs_p2p_boot.log (per-bridge before/after values, rmmod
  retries, p2p_links_total, dmesg P2P lines).
- /etc/modprobe.d/dp-acs-p2p.conf + /usr/local/sbin/dp-acs-early.sh:
  EVERY host-side `modprobe amdgpu` clears ACS first (install hook). The
  initramfs-loaded case is covered by the service's reload (the reload goes
  through the hook).
- Reversible: `systemctl disable --now dp-acs-p2p.service; rm
  /etc/modprobe.d/dp-acs-p2p.conf /usr/local/sbin/dp-acs-*` and/or any reboot
  without them (ACS bits reset to firmware defaults at power-on).

## POST-BOOT VERIFICATION (one command chain)

1. cat /var/log/dp_acs_p2p_boot.log
   - WANT: acs_cleared=12, rmmod succeeded, p2p_links_total > 0,
     dmesg "not supported by the chipset" lines ABSENT for same-card pairs
     (05<->08, 0d<->10) and STILL PRESENT for the 4 cross-card pairs
     (whitelist, DP-2 STEP B).
   - FAIL MODES: rmmod failed (log says) -> clear happened live; run
     `sudo modprobe -r amdgpu && sudo modprobe amdgpu` from a text console
     with no X session holding the dies, then re-check links.
2. docs/amd-port/scripts/dp_p2p_recheck.sh   (expect links on all 4 nodes)
3. THE P2P VERDICT (same-card pairs, dies 0+1 under switch A):
     docs/amd-port/probes/run_p2p_allreduce.sh 2
   (holds the campaign GPU boot lock; compiles + runs p2p_allreduce_probe
   on the same-switch pair; W31 instrument verbatim)
   Then dies 2+3 (switch B) by editing RANKS/device selection if the runner
   hardcodes 0,1 (check HIP_VISIBLE_DEVICES in the script).
4. Bank the outputs as results/dp2_stepa_*.log + ledger E-167.

## DECISION TREE AFTER THE PROBE

- canAccessPeer=1 + copies verified + bandwidth > through-host class:
  P2P IS REAL on Vega10 same-switch. Proceed: STEP B (whitelist one-liner
  {0x8086,0x4668,REQ_SAME_HOST_BRIDGE} via test-boot kernel or livepatch,
  owner style choice) for cross-card pairs, then the W8 transport probe
  per pair, then RCCL fingerprint + ONE served paired window (E-090
  numerics class).
- canAccessPeer=1 but copies fail/hang/fault: eligibility unblocked, data
  path NOT. Bank as userspace-refused; escalate to the kernel map path
  (amdgpu_amdkfd map of peer BOs) before concluding hardware-refused.
- canAccessPeer=0 still: check whether links exist but ROCr returned
  NEVER_ALLOWED (fine-grain PCIe branch, dp_p0_p2p_gates receipt STEP C) ->
  one-line ROCr patch behind an env, private-prefix build.

## REMINDERS (campaign laws)

- The svm-watchdog v2 + lock machinery is live; the probe runner respects it.
- One GPU item per boot under the throttled regime: this boot's item IS the
  probe. No server cells on the same boot.
- Xorg holds the AMD dies once gdm starts - any LATER amdgpu reload needs a
  text-console (non-X) context. That is why the service runs pre-gdm.
- OF-RECORD serving config/binary untouched by any of this.
