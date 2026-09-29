
# DP-2 STEP A VERDICT - Vega10 PCIe P2P IS REAL (same-card pairs) - 2026-09-25

## Chain of evidence this boot (16:32-16:40)

1. Boot log /var/log/dp_acs_p2p_boot.log: the modprobe.d install hook cleared
   ACS BEFORE the boot-time amdgpu load (acs_cleared=0 at service time =
   already zero; dmesg refusals at t=20.8s cover ONLY the 4 cross-card pairs).
2. sysfs: p2p_links live on all 4 KFD nodes (1<->2, 3<->4, type 2 PCIe,
   weight 40, max_bandwidth 8000). Same-card pairs {0000:05:00.0<->0000:08:00.0}
   and {0000:0d:00.0<->0000:10:00.0}.
3. W31 probe (run_p2p_allreduce.sh 2): died at its kstore step with
   "Memory access fault ... 0x28000" - the CUDA-UVA raw-pointer assumption.
   DIAGNOSIS: NOT a P2P failure; ROCm coarse memory maps peers at an
   alternate VA per device. The fault itself PROVES canAccessPeer=1 AND
   hipDeviceEnablePeerAccess succeeded (kstore is gated on ena=1).
4. dp_p2p_probe (new, docs/amd-port/probes/dp_p2p_probe.cu; receipt logs
   dp2_probe_pair01_*): pair 0<->1 BOTH DIRECTIONS:
      canAccessPeer=1, enable OK
      hipMemcpyPeerAsync 80KB: OK, sync OK, VERIFY PASS (byte-exact)
      lockstep med: 4KB=11.96us  20KB=14.25us  80KB=23.0us  1MB=159.9us
      1MB effective 6.56 GB/s = PCIe Gen3 x8 WIRE RATE (switch-direct DMA,
      not host-bounce)
5. Cross-card pair 0<->2 (0000:05:00.0 vs 0000:0d:00.0): canAccessPeer=0
   both directions - exactly the P2PDMA-whitelist scope (DP-2 STEP B).
6. IPC (W38's other refusal): hipIpcGetMemHandle STILL invalid-argument
   on same-card pairs (dp2_ipc_probe log). Gate located to
   Device::IpcCreate -> dev_mem->ExportHandle (clr rocclr/device/device.cpp:1228;
   ends in the KFD share/export ioctl). OPEN THREAD for the next session:
   pin ExportHandle's failure; NOTE hipIpcOpenMemHandle ALSO refuses
   same-process use (owners_process_id check) - W31-era probes must fork
   ranks to test the full IPC path.

## THE NUMBERS vs THE CAMPAIGN BASELINES

| metric | before (W8/W38 era) | now (same-card, direct) |
|---|---|---|
| hipDeviceCanAccessPeer | 0 x 12 pairs | 1 x 4 same-card directions |
| 80KB transport floor | 69.5us EMULATED (cooperative kernel) | 23.0us MEASURED peer copy |
| served RCCL boundary | 128.8-210.8us (host-staged SHM ring) | not yet re-measured (needs RCCL/STEP B) |
| 1MB stream | (no P2P existed) | 6.56 GB/s = x8 wire rate |

## WHAT THIS UNLOCKS (next steps, priority order)

1. STEP B (cross-card): one-line whitelist add {0x8086,0x4668,
   REQ_SAME_HOST_BRIDGE} in drivers/pci/p2pdma.c (~line 541). Test-boot
   kernel or runtime livepatch (owner style choice). Cross-card P2P then
   depends only on whether the ADL-S root complex actually forwards peer
   TLPs (same fail-closed probe adjudicates).
2. RCCL engagement: RCCL's P2P transport needs IPC (dead until the
   ExportHandle thread resolves); WITHOUT IPC the near-term serving lever is
   the DP-3 CUSTOM AR on peer copies (meta-backend butterfly exchange via
   hipMemcpyPeerAsync on separate streams: 3 x 23us overlappable + local
   reduce ~= 25-35us class per 80KB boundary vs 210.8us served = the
   +4.5-7% decode / +8-10% prefill prize, with skew rider).
3. UVA note for any custom kernel: peer pointers must be translated per
   agent (hipPointerGetAttributes / hsa_amd_pointer_info agentBaseAddress);
   raw cross-device pointer values fault by design on ROCm.

Campaign status: serving config/binary UNTOUCHED. One boot consumed (the
STEP A service + probes). No machine deaths; the kstore fault was
process-local and recoverable.

## STEP B UPDATE 2026-09-25 17:00 (livepatch route chosen, runtime)

- IPC root cause CLOSED (strace receipt dp_ipc_strace_165236.txt): the export
  dies in KFD ioctl nr 0x81 -> ENOTTY = an OUT-OF-TREE ROCm ioctl absent from
  mainline kernels; ROCm 6.2's runtime predates the dmabuf IPC path. Fix path:
  private newer ROCr (THE ROCK build, DP-4 vehicle) - kernel untouched. Not
  needed for DP-3 (peer copies proven).
- dp_klp_p2p livepatch BUILT and loaded (kernel 7.0.0-31 headers, SecureBoot
  off, MODULE_SIG not forced): forces __host_bridge_whitelist TRUE; livepatch
  transition COMPLETE in dmesg. Sources: /home/chris/driverprog/dp-klp/.
- Boot service extended: ACS clear -> insmod /usr/local/lib/dp_klp_p2p.ko ->
  amdgpu reload -> links count. NEXT REBOOT = the 12-pair test: if links +
  cross-card peer copies PASS, the full 4-die P2P substrate is live and the
  W31 4-rank probe prices the transport floor. If links appear but cross-card
  copies fault: ADL-S RC peer-TLP forwarding is refused by hardware - the
  documented partial outcome (intra-card hops only).
