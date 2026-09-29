# UPSTREAM-READY BUG REPORTS - ROCm 6.2 / THE ROCK findings (owner-fileable)

Context: 4x AMD Vega10 (gfx900, Radeon Pro V340/Instinct MI25), Ubuntu 24.04
(HWE kernel 7.0.14 custom), ROCm 6.2.0 originally; THE ROCK 7.14 nightly
wheels tested. All findings measured on real serving (llama.cpp TP4, 27B
model) plus minimal probes. Drafted 2026-09-28 by the DP driver program;
the owner may file these against ROCm/clr and ROCm/TheRock as they see fit.

## BUG 1 - hipIpcGetMemHandle fails unconditionally on ROCm 6.2 (out-of-tree ioctl)

- ROCm 6.2.0, kernel 7.0.0/7.0.14, gfx900 dGPU, large BAR (BAR = VRAM size).
- hipIpcGetMemHandle returns hipErrorInvalidValue for ANY device allocation,
  both with HSA_ENABLE_IPC_MODE_LEGACY=1 and with the default dmabuf path.
- strace: the failing call is a KFD ioctl (nr 0x81) returning ENOTTY - an
  out-of-tree ROCm ioctl that mainline kernels never implemented; ROCm 6.2's
  runtime uses it on the export path unconditionally.
- Minimal repro: docs/amd-port/probes/dp_ipc_direct.cu (hipMalloc 8MB,
  hsa_amd_pointer_info shows perfect base/size, then hsa_amd_ipc_memory_create
  returns 0x1001; log dp_ipc_direct_dmabuf/legacy.log).
- FIXED in newer stacks: the same operation succeeds under THE ROCK
  7.14.0a20260612 wheels (ipc_memory_create returns 0). Filing so other
  mainline-kernel users of 6.x know the class and the fix path.

## BUG 2 - cross-card hipMemcpyPeerAsync silently writes garbage when
enqueued on the destination device's stream

- ROCm 6.2.0, multi-gfx900, PCIe P2P enabled (peer links present).
- hipMemcpyPeerAsync(dst, dstDev, src, srcDev, size, stream): when the
  stream belongs to the DESTINATION device AND the pair crosses a host
  bridge (same-card pairs are immune), the copy returns hipSuccess but the
  destination contains uninitialized garbage.
- Enqueueing the identical copy on the SOURCE device's stream is byte-exact.
- Repro: 5-line pattern - two devices on opposite sides of a root complex,
  one memcpyPeerAsync per direction per stream choice; verify contents.
  (Full matrix: docs/amd-port/results/dp3_ar_tree_*.log, dp_ar_debug2 repro
  in docs/amd-port/scripts/dp4_paired_final.sh history.)
- Workaround (in production use): always enqueue peer copies on the source
  stream. File so the driver team can add a validation or fix the engine
  routing; also worth checking whether newer stacks still route this wrong.

## BUG 3 - redundant hipDeviceEnablePeerAccess leaves a sticky last-error
that later hipGetLastError() checks misattribute

- After RCCL (or any library) enables peer access, a second
  hipDeviceEnablePeerAccess returns hipErrorPeerAccessAlreadyEnabled and
  sets the thread's LAST ERROR state even when the caller accepts and
  discards that specific code. A later hipGetLastError() - e.g. the
  generic post-launch check in an inference engine - then reads the stale
  "peer access is already enabled" and aborts a healthy pipeline.
- Correct caller discipline is to call (void) hipGetLastError() after
  accepting AlreadyEnabled; but a runtime whose accepted-error codes still
  poison the sticky state turns a documented benign return into an
  application crash class. Request: do not set the sticky error for
  AlreadyEnabled (match CUDA, where the accepted case clears it).
- Repro: enable twice, accept the second return, then hipGetLastError() ->
  hipErrorPeerAccessAlreadyEnabled (docs/amd-port/results dp4 window logs).

## NOTE (not a bug) - PCI peer access for gfx900 on mainline kernels

For other mainline-kernel users of pre-CDNA GPUs: peer-to-peer works on
mainline amdgpu+KFD once two firmware/driver-policy items are handled:
1. ACS P2P redirect bits ship ENABLED by firmware on many boards (root
   ports and GPU downstream bridges alike). They force the PCI core to
   classify peer traffic as through-host-bridge; clearing them
   (setpci, cap+6 = 0) before amdgpu loads lets KFD build peer links.
2. On consumer host bridges (e.g. ADL-S 8086:4668) the P2PDMA whitelist in
   drivers/pci/p2pdma.c needs the device ID added - a one-line kernel patch
   (test-boot kernel or built-in). With both, PCIe P2P over a PLX fanout
   AND across the root complex runs byte-exact at wire rate (x8 Gen3
   measured 6.5-7.2 GB/s; 80 KB copies 17-23 us) on Vega10.
Findings at: docs/amd-port/results/dp_p0_p2p_gates_2026-09-25.md,
dp2_stepa_receipt_2026-09-25.md (E-166..E-178 trail).
