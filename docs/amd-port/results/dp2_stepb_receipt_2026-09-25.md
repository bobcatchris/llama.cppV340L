
# DP-2 STEP B VERDICT - FULL 12/12 P2P MATRIX LIVE ON PATCHED KERNEL - 2026-09-25 18:15

## Kernel: 7.0.14-dp2 (Ubuntu HWE source + p2pdma.c:540 one-liner {0x8086,0x4668,REQ_SAME_HOST_BRIDGE})
Built: make -j20 + modules_install + make install + dkms nvidia 580.173.02
prebuilt + update-initramfs. Grub default pinned. Display SURVIVED (nvidia
dkms prebuild paid off). dmesg: ZERO "not supported by the chipset" lines.

## THE MATRIX (dp_p2p_probe, receipt dp2_matrix_full_*.log, all 12 directions)

| pair class | directions | 80KB med | 1MB med | 1MB eff |
|---|---|---|---|---|
| intra-card {0<->1} | 2 | 23.2-23.3 us | 160 us | 6.5 GB/s |
| intra-card {2<->3} | 2 | 17.3-17.5 us | 145 us | 7.2 GB/s |
| cross-card (root complex) | 8 | 21.8-22.7 us | 184 us | 5.7 GB/s |

EVERY direction: canAccessPeer=1, 80KB VERIFY PASS byte-exact.

**THE ADL-S ROOT COMPLEX FORWARDS PEER TLPS CORRECTLY** - the last empirical
unknown resolved POSITIVE. Full 4-die P2P substrate is live and CHEAP
(cross-card 80KB within 3% of intra-card latency; 1MB pays ~13% bandwidth).

## W31 4-rank probe: structurally inapplicable (instrument finding, not a failure)
run_p2p_allreduce.sh 4 timed out at its FEAS kstore step: the W31 AR design
moves data as KERNEL PEER STORES via raw cross-device pointers (CUDA-UVA
assumption). On ROCm dGPU, peer access after hipDeviceEnablePeerAccess is
COPY-ENGINE ONLY (hipMemcpyPeerAsync works, instruction-level peer
dereference faults "page not present" - confirmed 2x). The W31 one-shot
kernel AR is therefore IMPOSSIBLE on this stack without IPC; the DP-3 custom
AR must be built on memcpyPeerAsync sequences (which the E-167/E-168 receipts
already anticipated). W31's "+7.6% absolute ceiling" arithmetic was for the
kernel-store design; the memcpyAsync tree design has a DIFFERENT (better)
ceiling - see the tree projection below.

## TREE-AR PROJECTION (per 80KB verify boundary, vs 210.8us served RCCL ring)
level 1: intra-card pairs reduce (23us copy + ~4us reduce kernel)
level 2: cross reduce (22.5us + ~4us)
level 3: broadcast to 3 peers (23us, parallel)
~80-100us class INCLUDING launch overheads vs 210.8 served. If it lands at
90us: 136 boundaries/cycle x 120us saved ~ 16ms/cycle on ~124ms = the
+12-15% decode class, PLUS the W39 skew rider. Prefill: bandwidth-class
(bf16 10.5MB boundaries at 5.7-7.2 GB/s peer vs host-staged SHM today).
NEXT INSTRUMENT: dp_ar_probe.cu (4-rank tree AR over memcpyPeerAsync + local
reduce, CPU-reference checked, lockstep latency) - build next session.

## Persisted machine state
- /boot/vmlinuz-7.0.14-dp2 + initrd + modules + nvidia dkms; GRUB default
  pinned to dp2. Revert: set GRUB_DEFAULT=0 + update-grub.
- /etc/modprobe.d/dp-acs-p2p.conf (ACS clear hook) - STILL REQUIRED every
  boot (firmware re-arms ACS). The klp/wl modules are GONE (not needed; the
  stuck dp_wl v1 cleared with the reboot).
- amd/v340-port-v2 serving config/binary UNTOUCHED (serving still boots the
  old kernel until owner decides; the dp2 kernel is backwards-compatible).
