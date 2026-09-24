# W31 RECEIPT: P2P-ALLREDUCE DESK (feasibility + flat-allreduce design + staged probe) - 2026-09-24

Desk: P2P-ALLREDUCE DESK (wt-p2p-ar, amd/p2p-ar, based on c2374eaaf).
Mission: the external reviewer's proposal - bypass the RCCL ring with a
FLAT P2P allreduce (each die pushes its slice to peers over PCIe P2P,
one/two concurrent phases instead of 6 serialized ring hops), estimated
8-14 us/boundary vs the banked floors.  Deliver feasibility verdict
(cited), the design (push vs pull decided), a built probe, a staged SHORT
run plan.  GPU status during this desk: U1 (arm=u1-depth) held all 4
dies behind /tmp/campaign_gpu_boot.lock the whole time - this desk is
ZERO-GPU; the probe is built and staged, not run.

Foundation: W8_tp4_boundary_receipt_2026-09-23.md (floors: RING1K 69.5 us
transport-free at 80 KB, KLAUNCH 11.6-12 us, served census 136.0
boundaries/round, die medians 128.8-210.8 us, wall pays die 1's 31.2
ms/round), W9_rccl_transport_receipt_2026-09-23.md (4-die inproc probe:
burst per-op 103.2 us uniform, lockstep worst 126.5 us, transport of
record SHM, NCCL_MIN_NCHANNELS=4 partial win 91.6 us), E-141 cycle math
(boundary wall 24-25% of cycle).

## 1. FEASIBILITY VERDICT (zero-GPU reading): UNRESOLVED, BIASED
## NEGATIVE for direct P2P - and that does NOT kill the desk

### 1.1 Platform facts (measured from the running system, no GPU work)

Topology (lspci -t, 2026-09-24):

    root 00:01.0 -> PLX PM8533 switch A (01:00.0)
      -> 02:00.0 -> [03:00.0 -> 04:00.0] -> 05:00.0  die 0 (HIP dev 0)
      -> 02:01.0 -> [06:00.0 -> 07:00.0] -> 08:00.0  die 1 (HIP dev 1)
    root 00:01.1 -> PLX PM8533 switch B (09:00.0)
      -> 0a:00.0 -> [0b:00.0 -> 0c:00.0] -> 0d:00.0  die 2 (HIP dev 2)
      -> 0a:01.0 -> [0e:00.0 -> 0f:00.0] -> 10:00.0  die 3 (HIP dev 3)

- 4 usable Vega 10 dies (Radeon Pro V340/Instinct MI25x2 class, gfx900),
  2 cards x 2 dies, one card (15:00.0) dead - the W9 die map (HIP die0 =
  05:00.0, die1 = 08:00.0, die2 = 0d:00.0, die3 = 10:00.0) confirmed
  verbatim by the RCCL INFO busIds.
- Same-switch pairs: (0,1) under switch A, (2,3) under switch B.
  Cross-switch pairs traverse the root complex.
- ACS: NO Access Control Services capability anywhere on the path -
  lspci -vvv shows no ACS section on the PM8533 upstream/downstream
  ports, the Vega 10 bridges, or the root ports; no acs_ctrl sysfs file
  on any of them.  Nothing redirects or denies peer TLPs; peer routing
  inside each PM8533 is unrestricted.  (/proc/cmdline has no
  pcie_acs_override because none is needed; it does have iommu=pt - DMA
  passthrough, no IOMMU translation on the peer path.)
- No p2pmem sysfs interface on the devices (amdgpu does not use the
  kernel pci_p2pdma facility; AMD P2P is done via KFD/GPUVM peer
  mappings, so its absence is NOT evidence against P2P).

### 1.2 The evidence FOR direct P2P working

1. The HIP API surface exists on this stack: hipDeviceCanAccessPeer /
   hipDeviceEnablePeerAccess are declared in the installed ROCm 6.2
   headers (HIP 6.2.41133-dd7f95766,
   /opt/rocm/include/hip/hip_runtime_api.h:4940-4981), including the
   explicit error hipErrorPeerAccessUnsupported, and libamdhip64.so
   contains the enable/disable/CanAccess implementations plus the
   failure strings "peer access is not supported between these two
   devices" - i.e. the runtime HAS a code path that both grants and
   refuses peer mappings.
2. Upstream llama.cpp already ships the enablement knob:
   ggml/src/ggml-cuda/ggml-cuda.cu:338-352 - GGML_CUDA_P2P env walks all
   device pairs, cudaDeviceCanAccessPeer, and cudaDeviceEnablePeerAccess
   (the CUDA-name aliases resolve to the hip functions on this build).
   It has never been exercised on this machine (untested in-tree
   machinery - exactly what our probe runs first).
3. Switch-level peer routing is generic on PM8533 and nothing above it
   (ACS) blocks it (1.1).

### 1.3 The evidence AGAINST

1. RCCL 2.20.5's own init on this machine tried and FAILED: the session
   of record results/rccl_multidie_a3_fp_default_20260923_153543.log
   contains (x12 each) "Could not enable P2P between dev 0(=5000) and
   dev 1(=8000)" and "...dev 2(=d000) and dev 3(=10000)" - both
   same-switch pairs - after which every ring hop connects
   "via SHM/direct/direct".  W9 corroborates: NCCL_P2P_DISABLE=1 is
   exactly neutral (P2P was never up), NCCL_SHM_DISABLE=1 costs +85%
   (SHM is the transport).  So RCCL's HIP-level peer enable path fails
   on this stack for the exact pairs a flat P2P kernel needs.
2. The HIP headers mark the whole PeerToPeer group "experimental"
   (hip_runtime_api.h:4946-4948) on ROCm 6.2.
3. Precedent class: AMD's supported P2P fabric story is XGMI-first;
   Vega 10 has no XGMI, and RCCL's own transport table picked SHM.

### 1.4 Why this is still UNRESOLVED, not a kill

RCCL's "Could not enable P2P" conflates (a) hipDeviceCanAccessPeer
returning 0 (platform refusal), (b) hipDeviceEnablePeerAccess erroring
(runtime refusal), and (c) RCCL's own policy gates (its P2P transport is
tuned for XGMI; PCIe P2P can be policy-refused even when the driver
would allow it).  The log line cannot separate them; only a direct call
sequence can (probe FEAS phase: CanAccess matrix -> Enable matrix ->
kernel peer-store round trip -> hipMemcpyPeerAsync round trip, per
ordered pair).  AND the desk survives a negative answer intact: the
host-staged FLAT arm (pinned-memory staging, no P2P dependency) attacks
the same target - it is the RCCL SHM transport class but with 2 phases
instead of 6 ring hops.  Kill condition for the whole desk: BOTH (i)
direct P2P infeasible AND (ii) host-staged flat loses to the banked
RCCL floors (103.2 us burst / 126.5 us lockstep at 80 KB) - that is a
measurement, staged in section 5.

Known constraint noted: RCCL refuses single-die multi-rank
communicators (W8 section 1.1, ncclInvalidUsage "Duplicate GPU
detected") - irrelevant for a custom kernel (no communicators), but it
is why no single-die RCCL baseline exists; our banked baselines are the
W8 in-device floors and the W9 4-die probe.

## 2. SERVED BOUNDARY BYTES (verified at the call site)

The dispatch is size-generic: ggml_backend_cuda_comm_allreduce_nccl
reduces ne = ggml_nelements(tensors[0])
(ggml/src/ggml-cuda/ggml-cuda.cu:1236); the meta executor hands it the
subgraph-final PARTIAL node of every subgraph except the last
(ggml/src/ggml-backend-meta.cpp:2354-2368).  The served shapes come from
the model + census of record:

- hidden size 5120 (the campaign's GEMM K class, OPTIMIZATION_PLAN
  section "K=5120/17408"; n_embd=5120).
- verify/catch-up T=4: ne = 5120 x 4 = 20480 elements = 80 KB fp32;
  draft T=1: ne = 5120 = 20 KB fp32 (W8 census table, section 2.1 -
  measured, not derived).
- fp32 band gate: n_backends >= 4 && ne < 262144
  (ggml-cuda.cu:1259) - all 136 decode boundaries/round are fp32.

Per-die partial: every die holds the FULL ne-element partial of its
row-parallel matmul (feature axis sharded across dies, sequence
replicated).  In the flat scheme each die OWNS chunk d = ne/4 elements
(20 KB at the verify class): "each die pushes its 20 KB slice" = 3 x
20 KB out + 3 x 20 KB in per transport phase.  The external reviewer's
byte framing checks out exactly.

## 3. THE FLAT ALLREDUCE DESIGN (A3)

### 3.1 Algorithm (sliced 2-phase PUSH - "one-shot class")

n = 4 dies, ne elements fp32, chunk C = ne/n (5120 elems / 20 KB at the
verify class).  One kernel launch per boundary per die, NBLK=8 blocks
striping the copies, each block with its own 64-B arrival slot (the
in-tree 2-GPU kernel's structure, generalized):

- Phase A (scatter): die d pushes chunk p of its partial into a staging
  row on die p (direct P2P store) or into its own pinned host row
  (host arm), then writes arrival token = boundary index.
- Phase B (reduce): die d sums ITS chunk: local + peers' rows, fixed
  order (local first, peers ascending).  Writes its result[chunk d].
- Phase C (broadcast): die d pushes reduced chunk d into every peer's
  result buffer (P2P arm) or posts it to its own pinned host slot which
  peers pull (host arm), then token.
- Phase D: spin for peers' broadcast tokens; die d now holds the full
  ne-element sum.

Phase count: 2 transport phases, both fully concurrent across all 12
directed peer edges, vs the ring's 6 serialized hops.  Wire bytes per
die: 120 KB/phase at the verify class (2x the ring's 120 KB total) -
at these sizes the wire is latency/protocol-bound (RCCL moves its 120
KB at an effective 1.2 GB/s in the 103.2 us burst), so paying 2x bytes
to buy 3x fewer phases is the trade.

Synchronization: arrival-token slots (strictly increasing boundary
index, 2-slot ping-pong reuse - the in-tree proof that slot N%2 is safe
absent done-acks, allreduce.cu:226 comment, holds verbatim for this
phase structure: my rewrite of slot s at boundary N+2 is ordered after
my wait for the peer's N+1 tokens, which is ordered after the peer's
kernel N completed, which is after its reads of slot s).  Single writer
per slot, no atomics (in-tree invariant, allreduce.cu:51-53).  Kernels
carry a spin bound that sets an error flag and exits - a wedged
handshake can never hang a die; kill-by-PID stays sufficient.

### 3.2 PUSH vs PULL: PUSH, decided

1. PCIe semantics: a peer store is a POSTED write TLP (fire-and-forget,
   no completion); a peer load is NON-POSTED (a completion must cross
   the fabric back per request).  At 20 KB latency-bound slices posted
   stores are the cheaper primitive and stream better from SM stores.
2. Single-writer slots: with push, exactly one die writes each staging
   slot, so the no-atomics token scheme is sound; pull would have all
   dies reading the same producer buffer (safe, but completions
   serialize on the producer's link).
3. Kernel stores beat the copy engine at this size: hipMemcpyPeerAsync
   pays DMA setup per op that dominates 20 KB; in-kernel stores keep
   the handshake in-kernel (the in-tree design rationale for the
   latency-sensitive case, allreduce.cu:23-27).
4. The host arm necessarily mixes both (post = push to own pinned slot;
   consume = pull from peer's slot) - that is forced by having no peer
   mapping; the P2P arm is pure push.

### 3.3 Numerics note (owner flag, not a blocker)

The flat per-chunk sum order (local first, peers ascending) differs
from RCCL's ring fold order.  Both are deterministic per tensor, differ
only in fp32 rounding dust - the SAME numerics class as the E-090 RCCL
sign-off convention (deterministic, bounded dust, two-boot same-sha
gate per arm; accept 0.66667 / mean len 3.00 cells).  Additionally the
flat scheme guarantees bit-identical results across the 4 dies BY
CONSTRUCTION (each chunk computed once by its owner, then copied) -
strictly stronger cross-die consistency than the ring.  Flagged for
owner sign-off if an arm is ever served; the probe checks bit-exactness
vs a CPU reference in the kernel's exact order plus cross-die identity.

### 3.4 Placement in the tree (env-gated arm, default untouched)

The comm dispatch chain (ggml-cuda.cu): a new arm
GGML_CUDA_ALLREDUCE=p2p -> ggml_backend_cuda_comm_init_p2p (enable peer
matrix, allocate the flat pipeline) -> try_allreduce_p2p, following the
exact existing chain pattern (init_nccl -> init_internal -> init_none,
ggml-cuda.cu:1395-1498; per-call failure returns false and the meta
backend's butterfly handles it, ggml-backend-meta.cpp:2368-2378).  A
size gate keeps prefill-class tensors (>= GGML_RCCL_PREFILL_NE) on the
existing path.  No meta-backend change.  The HIP platform default
(ggml-cuda.cu:1479-1480) is not touched.  NOT IMPLEMENTED in this desk
- the probe must first prove feasibility AND beat the banked floors;
the arm spec is here for the follow-up if it does.

## 4. THE PROBE (built; staged, not run)

docs/amd-port/probes/p2p_allreduce_probe.cu +
run_p2p_allreduce.sh.  Build: hipcc -O2 -x hip, COMPILE/BUILD-EXIT:0,
zero warnings (session log /tmp/p2p_build.log, binary committed).

Phases:
- FEAS: host-mapping check (hipHostMalloc(hipHostMallocMapped) +
  hipHostGetDevicePointer + device post/verify - the APIs the in-tree
  internal AR calls "CUDA-only", allreduce.cu:957; HIP-native names
  exist in the 6.2 headers, so this arm is testable); CanAccess matrix;
  Enable matrix (hipErrorPeerAccessAlreadyEnabled tolerated); kernel
  peer-store round trip using the RAW peer pointer (the same-VA
  assumption CUDA semantics grant - this is what proves or kills it on
  HIP); hipMemcpyPeerAsync round trip.  Per ordered pair, all dies.
- BENCH: the flat kernel of 3.1, arms p2p (device staging, direct peer
  stores) and host (pinned staging); lockstep (host wall, KLAUNCH-
  comparable) and burst (100-op stream-pipelined batches, HIP events,
  worst-die per-op - the W9 served-regime convention); sizes ne
  20480/5120; 2000 iters; halves + even/odd spread law; correctness =
  bit-exact vs CPU same-order reference + cross-die identity, spin
  bound with error flag.
- --ranks 2 mode for the early same-switch-pair window (dies 0,1).

## 5. BENCH PLAN + PREDICTED COST (what remains to measure)

Session budget: FEAS seconds, bench ~2-3 min GPU; total well under the
15-min SHORT window; lock-arbitrated (run_p2p_allreduce.sh), cool-die
gate < 60 C, killed by PID only.

Banked floors to beat (80 KB fp32, n=4): RCCL burst per-op 103.2 us,
lockstep worst-die 126.5 us (W9); transport-free ring floor 69.5 us
(W8, emulation); draft 20 KB class: 67.5-68.7 burst.

Model, honest bands (NOT banked until measured):
- p2p arm: per phase ~ peer-store latency (2-5 us) + 120 KB over the
  die's x16 at ~8-12 GB/s effective (10-15 us) + token/spin (2-4 us);
  two phases + local add (~3 us): ~30-42 us at 80 KB.  Draft 20 KB
  class: latency-dominated ~10-18 us.
- host arm: same structure + the host hop (posted D2H then peer H2D;
  host memory adds ~1-2 us latency and halves effective BW): ~35-55 us
  at 80 KB.
- The external reviewer's 8-14 us single-phase estimate is the
  optimistic edge: it prices reduce-scatter only; a served allreduce
  needs the broadcast phase too (every die consumes the full reduced
  tensor at the next subgraph), so 2 phases is the honest floor.

Wall arithmetic if the p2p arm lands mid-band (~35 us): served saving
~(103.2 - 35) x 136 = ~9.3 ms/round against the burst floor; on the
E-141 cycle (boundary wall 24-25% of ~127 ms/round) that is ~7% decode
- same ballpark as the reviewer's +6%.  The W9 caveat carries: the
served wall also pays arrival wait (0-141 us/boundary, schedule-bound)
which the flat kernel inherits unchanged (it spins for the slowest
arriver exactly like RCCL); the served measurement, not this probe, is
the final arbiter.

Decision tree at run time:
- FEAS: CanAccess=0 or Enable fails on the pairs -> p2p arm skipped,
  host arm carries; if host arm ALSO loses to 103.2 burst -> desk
  KILLED with a measured reason.
- FEAS passes -> both arms measured; winner goes to the coordinator
  with the 3.4 arm spec + E-090-class determinism gate.

## 6. LOG (append-only, newest last)

- 2026-09-24 14:10 desk opened; worktree wt-p2p-ar on amd/p2p-ar at
  c2374eaaf; U1 holds the campaign lock (arm=u1-depth since 12:22);
  zero-GPU phase.
- 2026-09-24 14:17 A2 feasibility reading complete (section 1); ACS
  absent platform-wide; RCCL P2P-enable failure evidence found in the
  W9 session of record.
- 2026-09-24 14:23 A3 design fixed (PUSH, 2-phase sliced, section 3);
  A4 probe written and built BUILD-EXIT:0 zero warnings; runner staged;
  WIP commit 993dfb0b9.
- 2026-09-24 14:3x receipt written; bench staged pending coordinator
  window (U1 still holding at last check).
