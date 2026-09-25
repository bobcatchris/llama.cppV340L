# VERIFY-TRANSPORT DESK RECEIPT (2026-09-22)

Desk: wt-verify-transport on amd/verify-transport (from amd/v340-port-v2 @ dd9b275f6).
Mission: replace the host-staged n=3 allreduce butterfly (the verified dominant
decode lever, ~1 ms/boundary x ~48 boundaries/round vs ~20-25 ms real math)
with RCCL, if and only if the spike + acceptance experiment land clean.

## 1. MAP (code, no GPU)

The transport path already exists in-tree; on this box it is compiled OUT:

- `ggml/src/ggml-backend-meta.cpp`: `ggml_backend_meta_graph_compute` probes
  `ggml_backend_comm_init` + `ggml_backend_comm_allreduce_tensor` proc
  addresses on backend[0]'s registry. If present, every subgraph boundary
  calls `comm_allreduce`; if the callback returns false (or no comm ctx),
  `allreduce_fallback` runs the butterfly.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: registers those proc addresses
  unconditionally (HIP included). `ggml_backend_cuda_comm_init` picks a mode
  from `GGML_CUDA_ALLREDUCE` = `nccl` | `internal` | `none` (Linux default:
  nccl). The nccl path (`ggml_backend_cuda_comm_allreduce_nccl`) reduces
  FP32 for small tensors (n=3: ne < 131072) and compresses to BF16 above.
  The `internal` path (allreduce.cu one-shot pinned-host AR) is compiled out
  on HIP (`#if !defined(GGML_USE_HIP)`) and is 2-rank anyway.
- HIP build: `GGML_HIP_RCCL` (ggml/CMakeLists.txt, default OFF) is the
  upstream switch: `find_package(rccl)`, `-DGGML_USE_NCCL` ("RCCL has the
  same interface as NCCL"), link `roc::rccl`.
- Serving build (`llama.cpp/build-hip/CMakeCache.txt`):
  `GGML_HIP_RCCL:BOOL=OFF` => `GGML_USE_NCCL` never defined =>
  `ggml_backend_cuda_comm_init` proc returns a comm ctx whose try_allreduce
  is the butterfly (init_none) => meta always runs `allreduce_fallback`.
  This is why every boundary is 4 peer copies through host + 3 ADD replays.

Current butterfly sum order (n=3, from `allreduce_fallback`): fold
`r0 += r2`, then exchange `r0 += r1`, `r1 += r0_folded`, copyback `r2 = r0`.
Per element: `((r0 + r2) + r1)`, bit-identical on all ranks (fp32 add is
commutative; both live ranks compute the same association, rank2 is a copy).

## 2. RCCL SPIKE (probe)

Artifact: `docs/amd-port/probes/rccl_probe.cpp` (+ `scripts/probe_rccl.sh`
runner, campaign-lock compliant: check-and-wait / hold / release, 110 s
SIGALRM + 140 s timeout, results in `docs/amd-port/results/rccl_probe_*.log`).

The probe, in one process, one host thread driving collectives (the same
shape as `ggml_backend_cuda_comm_allreduce_nccl`: ncclGroupStart / 3 x
ncclAllReduce / ncclGroupEnd on per-rank streams):

1. prints the hipDeviceCanAccessPeer matrix (expect 0 all pairs),
2. `ncclCommInitAll` across 3 gfx900 ranks (reports init + wall time; run
   with RCCL_DEBUG=INFO to capture the transport picked: P2P vs SHM vs NET),
3. benches the grouped allreduce at 32 KB and 128 KB fp32 (200 warmup,
   2000 timed iters, stream-synced per boundary - the real structure),
4. benches a faithful butterfly emulation on the same boot
   (hipMemcpyPeerAsync staged by the driver through host when peers cannot
   access each other, + device ADDs, + copyback, same enqueue order as
   `allreduce_fallback`),
5. bit-exactness: 64 random seeds per size, host ADD reference in the
   butterfly grouping `((r0 + r2) + r1)`, memcmp vs RCCL output on all 3
   ranks, max ULP over mismatches, cross-rank identity check.

RCCL version on box: 2.20.5 (`/opt/rocm-6.2.0`, librccl.so.1.0.60200,
932 MiB - device code for gfx900 IS embedded, verified via strings).

### Results

Runs (logs: `results/rccl_probe_*.log`, dies 0/1/2 = the serving dies, idle
SCLK, identical conditions for both arms, campaign lock held per protocol):

- INIT: OK in 639-654 ms, ranks=3, canAccessPeer=0 all pairs (verified in
  the probe itself). RCCL 2.20.5.
- Transport (NCCL_DEBUG=INFO): every channel `via SHM/direct/direct` - RCCL
  stages through HOST SHARED MEMORY (its own pinned staging), 2 channels,
  ring + tree both connected, e.g.
  `Channel 00 : 2[d000] -> 0[5000] via SHM/direct/direct nRanks 03`.
  Socket NET is bootstrap-only. So RCCL's pipelined chunked host-SHM
  transport is exactly the no-P2P path this box needs.
- Latency (grouped ncclAllReduce from one thread, sync per boundary):
  - 32 KB fp32: RCCL avg 94.0 us (min 86.6) vs butterfly emulation avg
    212.0 us -> 2.26x, -118 us/boundary.
  - 128 KB fp32: RCCL avg 152.4 us (min 135.7) vs butterfly 340.3 us ->
    2.23x, -188 us/boundary.
- Butterfly emulation = hipMemcpyPeerAsync (driver-staged through host,
  same primitive ggml uses) + device ADDs + copyback in `allreduce_fallback`
  order. The SERVED boundary costs ~1 ms (verified campaign decomposition);
  the emulation's 212 us is the bare primitive, i.e. in situ RCCL also
  eliminates 4 staging round trips + 3 graph dispatches per boundary on top
  of the 2.3x primitive win.
- ACCEPTANCE VERDICT: NOT bit-exact, exactly as fp non-associativity
  predicts. 64 seeds x {8192, 32768} elems: 0/64 seeds byte-identical to
  the butterfly grouping ((r0+r2)+r1); ~3.5% (32 KB) / ~4.4% (128 KB) of
  elements differ, max ULP 64 / 256; bounded dust, no sign-flip wildness.
  RCCL outputs are byte-identical ACROSS ranks in 64/64 seeds (the three
  replicas never diverge from each other - the property the butterfly also
  guarantees via copyback).
  => sum-order reorder = numerics class change = NEEDS CHRIS'S EXPLICIT
  SIGN-OFF per campaign law. Not mergeable on verification-bytes alone.
  No RCCL algorithm can reproduce the butterfly's uniform association
  (ring reduce-scatter reorders per chunk: ((rk+rk+1)+rk+2) rotates with k),
  so bit-exactness is unreachable by env tuning - the one-shot fixed-order
  AR design (allreduce.cu, CUDA-only in-tree) is the only bit-exact device
  collective shape; that is a separate desk if the reorder is declined.

## 3b. SERVED-ARM SPEC (for the sign-off window)

If Chris signs off on the reorder class, the arm is:

- Build: `-DGGML_HIP_RCCL=ON` (compile-clean, 0 warnings, librccl.so.1
  links into libggml-hip - verified in this worktree).
- Runtime: `GGML_CUDA_ALLREDUCE=nccl` (the only gate; unset = butterfly =
  byte-identical to today). Pin `NCCL_ALGO`/`NCCL_PROTO` to the acceptance
  values or leave unset and RECORD the RCCL_DEBUG=INFO connect lines as the
  acceptance fingerprint; drift here silently changes the numerics class.
- Canary (per existing law): greedy decode sha MUST be compared to the
  butterfly baseline - it WILL differ (that is the signed-off change), so
  the gates that must still pass are: acceptance rate 0.66667/3.00 +-band,
  8-needle long-context, determinism (same arm twice => same sha), plus the
  engagement signature: boundary tax lines drop, decode +15-25% class
  (48 x ~1 ms -> 48 x ~0.1-0.15 ms modeled).
- If the reorder is DECLINED: the legacy butterfly stays default; the
  secondary lever (pinned-ring async staging, -2..-6 ms/round, extends the
  merged GGML_PINNED_DEV_COPY pattern) becomes the fallback plan - designed
  here, deliberately not implemented while the bigger lever is on the table.

## 3. ACCEPTANCE EXPERIMENT (protocol, required before any integration)

Law: a sum-order reorder is a numerics class change and needs Chris's
explicit sign-off. fp32 add is commutative but NOT associative, so the only
way RCCL is a drop-in for the butterfly is bit-exact equality of the fp32
sum - byte equality, not tolerance.

Protocol (dump-based, two tiers):

- Tier A (offline, in the probe): N=64 seeds x {8192, 32768} elements.
  Same bytes in on all ranks -> host ADD reference in the exact butterfly
  grouping -> RCCL output -> memcmp on each rank. Random uniform [-1,1]
  exercises both mantissa carries and sign mixes. PASS = every seed, every
  rank, byte-identical.
- Tier B (served dump, behind the integration gate): one boot with
  `GGML_CUDA_ALLREDUCE=nccl` + a dump env that captures the 3 partial input
  tensors at a chosen set of boundaries (e.g. boundary 0 and 23 of round 0,
  200k-class prompt, decode verify ubatch), computes the host ADD reference
  offline, and memcmp's the RCCL output captured in the same boot. This
  covers real activation distributions (denormal tails, extreme scales) the
  probe's uniform RNG does not.

What bit-exactness would PROVE: RCCL's reduction order for n=3 at these
sizes coincides with the butterfly association `((r0+r2)+r1)`. Since fp32
add is commutative, equality of grouping is the only remaining difference
between two sums of the same three floats; byte-equality across thousands
of random triples means the grouping matches (near-certain probabilistically,
and it is checked per element, per seed). Then the transport swap cannot
move any downstream bit: logits, greedy sha, acceptance rate all stay
byte-identical - provable with the existing served canary (greedy decode
sha + accept 0.66667/3.00 gate, as in the R2/R3 arms of E-078).

What a MISMATCH means: sum-order reorder = numerics class change. Over 48
layers x hundreds of rounds the ULP dust integrates; sampled tokens and
acceptance can drift. Per campaign law that arm is NOT mergeable on
verification-bytes alone - it needs Chris's explicit sign-off, with the
measured max ULP and mismatch fraction attached. A WILD mismatch (sign
flips, ULP >> 1) is not a reorder - it is a bug (racy zero-fill of
non-compute ranks, unaligned staging); do not proceed either way.

Determinism pin: RCCL's algorithm/protocol choice depends on message size
and env (`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_SHM_*`). The acceptance result is
only valid for the pinned env recorded with it (captured via RCCL_DEBUG=INFO
connect lines per size class). The integration must record - or fix - these
envs when the gate is on; an env drift at serving time silently changes the
numerics class.

Scope note: the n=3 FP32 path applies while ne < 131072 (decode verify
boundary = 8192 elements = 32 KB: FP32). For prefill-sized reductions
(ubatch 512: ne up to 4.2M) the upstream heuristic compresses to BF16 -
that is a much larger numerics change and is OUT of this desk's acceptance;
decode is the target, prefill keeps the butterfly unless Chris opts in.

## 4. INTEGRATION DESIGN

Zero new subsystems - wire the existing upstream path:

1. Build: campaign serving build adds `-DGGML_HIP_RCCL=ON` (upstream
   option; defines GGML_USE_NCCL, links roc::rccl). Nothing else changes.
2. Runtime gate: `GGML_CUDA_ALLREDUCE=nccl` (upstream env, already parsed).
   The meta backend then calls `comm_allreduce` at every boundary and only
   falls back to the butterfly per-call if it returns false.
3. Byte-identical default (IMPLEMENTED, staged in this worktree): with
   GGML_USE_NCCL defined and env unset, the Linux default in
   `ggml_backend_cuda_comm_init` is nccl - flipping RCCL into the default
   numerics class of every future HIP build. The HIP default is now
   `init_none` (butterfly) so that:
   - env unset => butterfly, byte-identical to today on every build, and
   - `GGML_CUDA_ALLREDUCE=nccl` => RCCL, on CUDA and HIP alike.
   ~7 lines in ggml-cuda.cu. (A separate GGML_RCCL_BOUNDARY alias was
   considered and rejected - it would be a second knob for the same thing.)
4. Non-compute ranks: `ggml_backend_cuda_comm_allreduce_nccl` already
   memsets non-compute tensors to 0 before the reduce (the n_outputs=0
   case) - semantics preserved.
5. Fallback safety: init failure (e.g. RCCL cannot form a 3-rank
   communicator) logs and falls back through internal (unavailable on HIP)
   to the butterfly - boot survives, just slower. Per-call failure returns
   false and the meta butterfly handles that boundary.

Host tests: no host-visible behavior changes when the gate is off (comm
init path compiles identically with GGML_HIP_RCCL=OFF). Byte-exact host
tests apply to anything this desk adds beyond the 3-line default flip.

## 5. SECONDARY (pinned-ring async staging for the legacy butterfly)

Owned but deferred until 1-3 land: event-ordered async pinned-ring staging
for the butterfly's 40 copies/round (-2..-6 ms/round modeled), extending the
merged GGML_PINNED_DEV_COPY pattern (ggml_backend_dev_copy_staging). If
RCCL wins, the butterfly becomes the fallback path and the staging upgrade
applies to it only if still worth it.

## LOG (append-only, newest last)

- 2026-09-22 desk opened; worktree created from amd/v340-port-v2 @ dd9b275f6;
  probe written + compiled clean with hipcc (gfx900, RCCL 2.20.5).
- 2026-09-22 spike run 1 (full2, 2000 iters, 64 seeds): INIT OK 654 ms;
  32 KB RCCL 94.0 us vs butterfly 212.0 us (2.26x); 128 KB 152.4 vs 340.3
  (2.23x); bit-exactness 0/64 seeds, ~3.5/4.4% elems, max ULP 64/256,
  cross-rank identical 64/64.
- 2026-09-22 spike run 2 (transport2, NCCL_DEBUG=INFO): all channels via
  SHM/direct/direct, 2 channels, ring + tree connected; init 644 ms.
- 2026-09-22 integration staged: HIP default comm mode = butterfly (rccl
  opt-in via GGML_CUDA_ALLREDUCE=nccl); compile clean with and without
  GGML_HIP_RCCL=ON, 0 warnings; all five host suites PASS; commit follows.
