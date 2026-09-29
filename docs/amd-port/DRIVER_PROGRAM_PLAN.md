# DRIVER PROGRAM PLAN - making amdgpu/ROCm work for this box (DP-0..DP-4)

Standing owner direction (2026-09-25): the driver stack is open source (mainline amdgpu
kernel driver, ROCT thunk, ROCr runtime, RCCL, THE ROCK build system); vendor support
matrices are policy, not capability. Older ROCm source trees are donors for mechanism
ports. The only cost is time. This plan runs alongside the kernel ladder (W37/D1 class)
and is additive with it via the W-coupling law.

Laws inherited unchanged (OPTIMIZATION_PLAN_TP3_200K.md, dossier PART 9): append-only
ledger; provenance or it did not happen; every gate fails loud; paired served cells for
any served verdict; same-session bit-exactness for kernel arms; dies are PCI addresses;
churn rationing (one GPU item per boot, <= 6 guard cycles/hour, svm-watchdog lock
respected by every DP GPU cell); thermal sideband on every GPU cell; E-149 hazard law
(memory-temp guard mandatory on ANY clock/power work); system-state writes (sysfs
pp_table, module reloads, test-boot kernels) require owner-go on FIRST application per
configuration, then run under the guard.

Naming: DP desks (DP-0.*, DP-1, ...) with receipts at docs/amd-port/results/dp_*.
Ledger entries: E-166+ (drafted by the desk, appended per append-only law).

## THE PRIZES (why this program exists)

| prize | size | evidence | current status |
|---|---|---|---|
| soak decay (23 -> 17 t/s sustained when far dies latch 991 MHz) | up to +26-30% SUSTAINED decode | W18; E-149 says the surviving actuator is SMU/od8-level = the driver | un-attempted at the driver level (dpm_sclk floors killed, E-149) |
| allreduce transport (P2P over the PM8533 fabric) | +4.5-7% decode, +8-10% prefill; re-opens one-shot AR (+3.2-4.5%, W38) | W8 (69.5 us emu floor vs 210.8 us wall); W31/W38 closures are POLICY closures: canAccess=0 is an eligibility gate, not a silicon measurement | hardware substrate verified (8 GB large BARs per die at 0x5000000000/0x5400000000..., no ACS, P2P-capable switch class) - receipt dp_p0_topo_2026-09-25.md |
| per-die clock spread (skew) | +2.4-4.8% (W39) rides on both above | W22/W39 | rides |
| hard-death rate (E-162: likely power-delivery transients) | availability (fewer lost cells) | E-162 | undervolt plausibly reduces burst current (directional, not promised) |
| compiler wall (8B-store miscompile; E-144 occupancy rungs unbuildable; hipblas compute-type hack hip.h:163-175) | re-opens sealed arms; GEMM selection for the 41% prefill term | WALL 2, E-144 | THE ROCK gfx900 = "Build Passing" tier (SUPPORTED_GPUS.md) - prebuilt tarball path |

Stack math reminder: D1-class kernel wins shrink per-boundary WORK, P2P shrinks
per-boundary TRANSPORT, undervolt holds the CLOCKS both depend on. They multiply.

---

## PHASE 0 - RECON AND ARCHAEOLOGY (zero GPU; STARTED 2026-09-25)

### 0.1 Fabric/topology recon - DONE this session
Receipt: results/dp_p0_topo_2026-09-25.md (lspci tree + BAR dump verbatim).
Facts banked: 4 dies = 2 PM8533 trees off CPU roots 00:01.0/00:01.1; intra-switch pairs
{0000:05:00.0 <-> 0000:08:00.0} and {0000:0d:00.0 <-> 0000:10:00.0} (P2P never crosses
the root complex = the near-certain subset); cross-tree pairs depend on Z690 RC peer-TLP
forwarding (empirical question #2, per-pair probe answers it); every die exposes a full
8 GB 64-bit prefetchable BAR; no ACS (kernel cmdline); iommu=pt.

### 0.2 Power/state recon (read-only sysfs) - DO THIS SESSION
Per die (PCI-address keyed, card2 = the NVIDIA display, excluded):
- /sys/class/drm/cardX/device/pp_table (binary dump, one per die)
- pp_dpm_sclk / pp_dpm_mclk (enabled levels), pp_power_profile_mode
- hwmon power1_cap / power1_cap_max / in*_input (Vddc if exposed), temps
- rocm-smi: --showpower --showclocks --showmem --showpids --showdriverversion
Receipt: results/dp_p0_power_<ts>/ (per-die files + one summary table).
Deliverable: the per-die voltage/frequency/DPM table + paper undervolt headroom calc
(input to DP-1). Kill signal for DP-1 (paper only, no GPU): if power1_cap_max or
current curves show no voltage headroom at the decode band (1100-1350 MHz), DP-1
narrows to "profile_mode + fan curve" only.

### 0.3 P2P gate archaeology (source reading; the DP-2 patch map) - STARTED
Target: pin the EXACT lines that produce today's refusals:
- hipDeviceCanAccessPeer=0 12/12 (W31)
- hipIpcGetMemHandle invalid-argument 0/12 (W38)
- cudaMemcpyPeer 0/12 (W38)
Chain: HIP (open) -> ROCr hsa_amd_agents_allow_access / ipc export (open, rocm/ROCR-Runtime)
-> ROCT thunk (open, rocm/ROCT-Thunk) -> KFD/amdgpu (mainline kernel).
Also map: the Vega-era peer-DMA-buf kernel enablement commits (amd-gfx list, ~2019,
Deucher/Francis series) as the donor mechanism to port if the ioctls need help.
Deliverable: results/dp_p0_p2p_gates.md - file:line for every gate + the patch sketch
(exact hunks, smallest-diff-first ordering) + the probe ladder for DP-2.
Explicit anti-goal: do not touch the serving stack; all patched artifacts run from a
private prefix (LD_LIBRARY_PATH / bundled tarball), never over /opt/rocm.

### 0.4 Toolchain plan (disk-law compliant)
DISK LAW (this box, 2026-09-25): ssd 12 GB free, / 25 GB free. A full THE ROCK build
(~200 GB) is FORBIDDEN until space is freed; the campaign's 100%-disk incident is the
precedent. Therefore:
- DP-2 vehicle: build ROCr + ROCT from source only (small, minutes, /home/chris/driverprog).
- DP-4 vehicle: download THE ROCK PREBUILT gfx900 release tarball (Build Passing tier
  publishes artifacts) into /home/chris/driverprog (25 GB budget; verify size before
  pull). Only if no artifact exists: revisit selective source build after a disk
  cleanup decision (owner).
Gate for DP-4 start: tarball compiles a gfx900 TU and links against its own ROCr stubs;
receipt dp_p0_toolchain.

### 0.5 KFD/SVM + driver version bookkeeping (read-only)
uname -r, /sys/kernel/debug/amdgpu... (if root), dmesg | grep -i amdgpu | tail, KFD
version from rocm-smi --showdriverversion; journalctl per-boot svm hog counter history
(the E-162 dataset). Feeds the long-term ROCm-upgrade decision; zero risk.

## PHASE 1 - SMU/PP_TABLE UNDERVOLT (first GPU work; DP-1)

Mechanism: Vega10 soft pptable (already exposed per die at
/sys/class/drm/cardX/device/pp_table). Undervolt = lower V at EQUAL clocks so the dies
stay out of the 991 MHz throttle band under sustained decode. Unlike the killed dpm
floors (E-149: raised power, HBM 90-93 C hazard), undervolting REDUCES power and heat
at equal clocks. Dies are 110 W-capped (E-162).

### 1.1 Tools
- dp_pptable_dump.py / dp_pptable_write.py: parse/edit the Vega10 pptable (state table
  sclk->mclk->vddc/vddci curves). Donor reference: community Vega pp_table tooling
  (VGTab class). No external binary is trusted blind: every edit is a bounded diff,
  dumped before/after, sha-stamped.
- dp_guard.sh: THE guard (E-149 law): mem temp >= 88 C (crit 95, campaign hazard
  witness was 90-93) -> immediate revert to the stock pp_table + loud log + cell VOID;
  polls the PCI-keyed hwmon at 1 s; runs for the whole cell; exit code feeds void_gate.

### 1.2 Application protocol (first application needs owner-go per the hazard law)
Order: one die first (die 3 = the slowest/throttle-prone, 0000:10:00.0), idle, no server
running; guard live; revert timeout 5 min. Then all four, then the verdict cells.
Never during a serving cell; never mixed with other desks' GPU items (churn law).

### 1.3 Verdict cells (paired, throttled regime, sideband always on)
Cell shape: cool boot -> stock pp_table -> 20-min decode soak (the W18 instrument
class: act-mean sclk, <=991 duty, decode t/s curve) -> idle -> undervolted pp_table ->
identical soak -> idle. Paired windows across two boots if a single boot can't hold
both arms clean (void gates decide, not us).
PROMOTE gate: sustained act-mean >= 1150 MHz on ALL dies; soak decode >= 0.95 x
fresh-boot decode; zero guard trips; determinism double-run byte-identical (crash
detector); acceptance EXACT (0.66667 +- the usual band).
KILL gate: any guard trip at target clocks; any determinism/acceptance anomaly; or
act-mean unchanged (the throttle is not voltage-bound after all -> bank the negative
with the mechanism re-measured, N-01 discipline).

### 1.4 Expected receipts
results/dp1_undervolt_profile_<ts>.md (the profile + per-die before/after tables),
dp1_soak_*.{log,json} (sidebands + decode curves), ledger E-entry at close.

## PHASE 2 - P2P ENABLEMENT (DP-2; the "dead permanently" door re-opened)

### 0.3 FINDING (2026-09-25, receipt dp_p0_p2p_gates_2026-09-25.md) - LADDER REFINED
Root cause of the 12/12 refusal is PINNED: firmware ACS P2P redirect bits
(0x001d: SrcValid+ReqRedir+CmpltRedir+UpstreamFwd) on every fabric bridge
force the PCI core to classify peer traffic as through-host-bridge, and the
ADL-S host bridge (8086:4668) is not in the drivers/pci/p2pdma.c policy
whitelist. Large-BAR/addressing/config/module-param checks all PASS. The
ROCr NEVER_ALLOWED fine-grain branch is probably NOT reachable (HiveId 0==0
equality + coarse-grain path); do not patch ROCr preemptively.

### 2.1 Patch ladder (refined, smallest risk first)
STEP A (RUNTIME, no rebuild, unblocks same-card pairs {05<->08, 0d<->10}):
  scripts/dp_acs_clear.sh --go (dry-run validated 2026-09-25: 12 bridges
  0x001d, switch upstream ports 0x000c, 4 bridges no-ACS skip) then
  amdgpu reload (ONE churn event, server down), then scripts/dp_p2p_recheck.sh
  (first signal: p2p_links non-empty in sysfs), then W31 probes same-card pairs.
STEP B (cross-card pairs): add {PCI_VENDOR_ID_INTEL, 0x4668, REQ_SAME_HOST_BRIDGE}
  to pci_p2pdma_whitelist[] (drivers/pci/p2pdma.c ~line 541) - either a
  test-boot kernel or a runtime livepatch for probing. Empirical risk: ADL-S
  RC peer-TLP forwarding is unproven; same-card pairs do not depend on it.
STEP C: only if canAccessPeer still 0 after links exist: the ROCr fine-grain
  PCIe branch (amd_memory_region.cpp GetAccessInfo) behind an env, private build.
STEP D: W8-class transport probes per pair -> RCCL fingerprint/microbench ->
  ONE served paired window (E-090 numerics class, owner sign-off).

### 2.2 Eligibility probes (the W31/W38 scripts, re-run verbatim)
Per pair, 12 ordered pairs: canAccessPeer / enablePeerAccess / hipIpcGetMemHandle /
memcpyPeerAsync. PASS = the earlier 0/12 flips. Fail-closed: a hang is watchdog-killed
(no WEDGES like W31; every probe has a timeout + kill).

### 2.3 Transport probes (the W8 instrument class, per pair)
- pure P2P copy latency/bandwidth (4 KB..1 MB) - intra-switch pairs first
- 4-rank ring AR emulation on real P2P at the served payload (80 KB fp32)
Deliverable: the real per-boundary floor table replacing the 69.5 us emulation figure.

### 2.4 RCCL on the new substrate
With peer access live, RCCL topo detection should select P2P; fingerprint via
NCCL_DEBUG=INFO (the W9 class). Microbench ring AR; then ONE served paired window
(battery law). Numerics class: transport reorders the fp32 sum = the E-090 precedent
(RCCL promotion, owner sign-off exists for this exact class).

### 2.5 Gates and kills
PROMOTE: paired decode >= +3% AND prefill >= +5% with acceptance/text gates green.
KILL/HARDWARE CLOSURE: if peer maps fault/hang on gfx900 at every ladder rung, bank the
closure as HARDWARE-refused (finally a physics receipt, not a policy one - that alone
closes W38 properly). PARTIAL: intra-switch pairs live, cross-tree dead -> recompute
the prize with RCCL topology awareness and decide by the paired window.

## PHASE 3 - CUSTOM AR OVER P2P (conditional on DP-2 promote)
SM-driven ring/one-shot AR kernel (fp32, 80 KB payloads, doorbell flags in peer BAR
windows): the W38 one-shot ceiling math (+3.2-4.5% with skew untouched; +7.6% absolute)
was substrate-blocked; DP-2 restores the substrate. Gate: >= 15% faster per-boundary
than RCCL-on-P2P at 80 KB on the W8 probe before any integration work. E-090 numerics
class. Not started until DP-2 promotes.

## PHASE 4 - THE ROCK COMPILER/ROCM USERSPACE A/B (DP-4)
- 4.1 Toolchain gate: build-hip tree compiles under the gfx900 prebuilt toolchain;
  oracle battery bit-exact vs the 6.2 binary (the WALL-2 8B-store oracle is the
  canary). Any oracle divergence = toolchain VOID, banked, serving binary untouched.
- 4.2 Re-open E-144 occupancy rungs on the newer LLVM (static census first, the
  P4.10 law - res-usage before GPU time).
- 4.3 hipblasGemmEx compute types (kill the hip.h:163-175 datatype hack) + rocBLAS
  A/B on the W5 prefill GEMM shapes.
- 4.4 When D1 exists: build it with BOTH toolchains; codegen deltas measured like any
  arm.

## RISK REGISTER
- pp_table edits: instability/hang on one die -> single-die-first protocol + stock
  table kept in /tmp + guard revert. Worst case: a reboot restores defaults (soft
  pptable is not persistent across boot).
- P2P probes: hangs -> every probe timeout+kill wrapped (the W31 host-staged WEDGE
  lesson); test boots only, never the serving boot.
- Private ROCr/ROCT: version skew vs 6.2 stack -> probe binaries link the private
  runtime ONLY; serving stack env never modified.
- THE ROCK prebuilt: "Build Passing" tier = artifacts publish, runtime unverified ->
  oracle-gated like any arm before any GPU trust.
- Disk: hard budget - driverprog work lives under /home/chris/driverprog; the ssd law
  (>= 10 GB headroom) is a standing gate for every fetch.

## STANDING RULES (owner-directed)

- **NEVER TOUCH THE NVIDIA DRIVER OR DISPLAY STACK.** The NVIDIA adapter
  (card2, proprietary 580.x driver, DKMS) is off-limits to this program:
  no module load/unload, no parameter changes, no config edits, no DKMS
  rebuilds (the one-time 7.0.14-dp2 compatibility prebuild was the sole
  exception, completed 09-25), no PCI writes to the NVIDIA adapter.
  The program's scope is the four AMD dies only: amdgpu/KFD/ROCm and the
  fabric bridges on the AMD paths.  Display/session risk is unacceptable.
- The owner's own reboots/fixes on the box take absolute precedence: if a
  timer tick finds owner activity in flight, observe only, do not act.

## PROGRAM STATUS (final, 2026-09-28 06:50 - see ledger E-166..E-179)

| phase | state | headline |
|---|---|---|
| DP-0 recon | DONE | topology, power/OD curves, P2P gate map, sources |
| DP-1 undervolt | CLOSED (this cooling) | guard live-fired at 89C, auto-revert proven; no headroom; re-open on hardware change |
| DP-2 P2P enable | DONE | 12/12 directions live on kernel 7.0.14-dp2 (whitelist built-in + ACS boot hook); byte-exact; cross-card within 3% of intra-card |
| DP-3 tree AR | DONE (infrastructure default-off) | serves byte-identical text; thermally-matched win +4-8% decode (E-178); absolute cells thermal-capped |
| DP-4 THE ROCK stack | DONE | full serving path on the 7.14 gfx900 nightly; guards passing; IPC gap closed by vendor code |
| DP-4b IPC | RESOLVED BY UPGRADE | ROCm 6.2 runtime limitation; 7.14 works (E-175) |

## THE ADOPTION (one command once the owner decides)
    sudo docs/amd-port/scripts/dp4_adopt_serving.sh
Verifies dp2 kernel, switches the serving launcher to the THE ROCK binary
+ AR_TREE=1, runs the full guard battery, and rattles the baseline if it
clears. Revert: the script prints the reverse steps; nothing is touched
until the owner runs it.

## STATUS LEDGER (updated per desk; ledger E-entries draft at each close)
- 2026-09-25 DP-0.1 DONE (topology/BAR receipt dp_p0_topo_2026-09-25.md).
- 2026-09-25 DP-0.2 DONE (per-die power/pp_table/OD dumps,
  results/dp_p0_power_2026-09-25/: sclk curve 300..1500MHz @ 800..1250mV,
  power1_cap locked 110W, profile BOOTUP_DEFAULT not COMPUTE).
- 2026-09-25 DP-0.3 DONE (gate map dp_p0_p2p_gates_2026-09-25.md: ACS+whitelist
  root cause pinned end-to-end with file:line; sources cloned to
  /home/chris/driverprog/src: ROCR-Runtime (incl libhsakmt), clr, therock,
  kernel-ref).
- 2026-09-25 DP-1 STAGED (scripts/dp_uv_apply.sh + dp_uv_guard.sh; guard trip
  mem 88C, timeout 30 min; first profile -100mV on sclk levels 4-7, -50mV mclk
  2-3; NOT applied - owner-go + guard-live required).
- 2026-09-25 DP-2 STEP A ARMED (scripts/dp_acs_clear.sh dry-run validated on
  this boot; NOT fired - owner-go; then amdgpu reload = one churn event).
- DP-0.4/DP-4 planned (THE ROCK nightly index path noted; disk law defers).
- 2026-09-25 16:40 DP-2 STEP A VERDICT IN: Vega10 PCIe P2P REAL on same-card
  pairs (E-167; receipt dp2_stepa_receipt_2026-09-25.md). canAccessPeer=1,
  80KB peer copy 23.0us byte-exact, 1MB at x8 wire rate. IPC still refused
  (open thread: clr device.cpp:1228 ExportHandle + same-process probe caveat).
  Cross-card = STEP B scope confirmed (whitelist one-liner).
- NEXT OWNER DECISIONS: (1) STEP B style (test-boot kernel vs livepatch);
  (2) DP-3 custom-AR desk go (peer-copy substrate, no IPC needed);
  (3) first DP-1 undervolt cell under guard.
