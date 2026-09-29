# W38 RECEIPT: ONE-SHOT ALLREDUCE SCOPE DESK (vLLM custom_all_reduce class on TP4 V340) - 2026-09-24

Desk: ONE-SHOT-AR SCOPE DESK (wt-oneshot-ar, amd/oneshot-ar, based on
4bf0d6476 = amd/v340-port-v2 tip).  ZERO-GPU desk: no server launches, no
GPU processes; the only staged executable artifacts are built, not run.
Mission: can a custom one-shot/two-shot allreduce (vLLM
custom_all_reduce style: HIP IPC handles + one cooperative kernel where
each rank reads peers' buffers in-register) beat the RCCL ring at OUR
boundary sizes on our topology (4x gfx900 Vega 10, 2x PLX PM8533, PCIe
Gen3, ROCm 6.2.0-66, RCCL 2.20.5)?

## 1. VERDICT: DEAD ON ARRIVAL - and the mission's own ceiling was overstated

Two independent kills, one residual question (closed by a staged probe,
section 6):

### 1.1 Kill 1 (measured, W31): the peer-read substrate does not exist

A one-shot kernel reduces peers' data IN-REGISTER, which requires every
die to READ other dies' VRAM.  W31's executor run measured the full
CanAccess/Enable matrix (results/p2p_allreduce_probe_r4_linebuf_20260924_155026.log):

    FEAS i->j: canAccess=0 enable=0   for all 12 ordered pairs

Driver-level refusal (hipDeviceCanAccessPeer returns 0 platform-wide;
hipErrorPeerAccessUnsupported exists in the ROCm 6.2 runtime - the
refusal is deliberate, W31 section 1.2/7.1).  This also explains RCCL
2.20.5's own x12 "Could not enable P2P" (session of record
rccl_multidie_a3_fp_default_20260923_153543.log) and why every ring hop
connects "via SHM".  Without a peer mapping, a custom kernel has nothing
to read: the one-shot design loses its transport.  The host-staged flat
fallback (the only substrate that measurably works - see 1.3) WEDGED:
zero completed allreduces in n=4 and n=2 configurations (E-144, W31
section 7.2), and even its design band (35-55 us at 80 KB, W31 section 5)
only marginally beats the 69.5 us transport-free ring floor while the
SHM-class staging it needs is exactly what RCCL already serves.

### 1.2 Kill 2 (arithmetic, banked numbers): the wall is not transport

The mission framed the prize as "slowest-die 210.8 us per boundary vs a
59 us floor" and a ceiling of "allreduce wall 28.7-31.2 ms -> ~8-12 ms
per cycle = +8-10% decode".  Both numbers misread the banked
decomposition (CAMPAIGN_DOSSIER section 1.4, receipts W8/W10/W22):

    per-boundary wall (die 3, served, 29-round window) = 210.8 us
      = ring transport floor  69.5 us   (W8 emulation, 80 KB fp32, n=4)
      + arrival skew          ~141 us   (W10: wait redistribution;
                                        W22: clock/power state)

A one-shot allreduce replaces ONLY the transport term.  Consequences:

- ABSOLUTE ceiling of ANY transport work, including a zero-transport
  allreduce: wall 136 x 141 us = 19.2 ms/cycle vs 28.7 ms = +7.6% MAX.
  The mission's 8-12 ms band is BELOW this bound - it is reachable only
  bundled with arrival-skew compression (per-die clock floors,
  E-147/E-148 - the campaign's live lever, no new transport needed).
- Realistic one-shot at 80 KB (W31 section 5 band: ~30-42 us; the
  reviewer's 8-14 us prices reduce-scatter only and omits the broadcast
  phase every boundary needs): saves 28-40 us per 80 KB boundary.
- Cycle arithmetic: 130 x (28-40 us) [verify+catch-up] + 6 x (52-60 us)
  [draft 20 KB class] = ~3.95-5.56 ms per ~124 ms cycle = **+3.2-4.5%
  decode, IF the skew term is untouched** - and it is (W10: medians are
  wait redistribution; W22: skew is clock state; E-119: channel count
  -12.6% served, transport flat x1.07 in W9).  Any cooperative kernel
  "spins for the slowest arriver exactly like RCCL" (W31 section 5).

So even with a working substrate this desk buys roughly a third of what
the mission promised, and the larger term (skew) is owned by the clock
floor desk, not by transport.

### 1.3 What each candidate substrate is worth (cited status)

| substrate | status on this stack | one-shot viability |
|---|---|---|
| hipDeviceEnablePeerAccess (raw peer pointers) | canAccess=0, enable=0, 12/12 pairs (W31 executor) | DEAD |
| hipIpcGetMemHandle/OpenMemHandle | NEVER TESTED (W31 FEAS short-circuited on enable=0) | the one open question; probe staged (section 6). Priors negative: same driver peer-mapping substrate, in-process TP is not even the IPC use case |
| hipMemcpyPeerAsync (copy engine) | NEVER TESTED (same short-circuit) | closed by the same staged probe; DMA-setup-dominated at 20 KB anyway (W31 3.2) |
| mapped pinned host memory (hipHostMallocMapped) | WORKS (W31 HOSTMAP post/verify token=7) | this is the W31 host arm: WEDGED as implemented (E-144); band 35-55 us at 80 KB overlaps the floor it attacks |

## 2. SERVED MESSAGE SIZES - the mission premise was wrong, exact numbers here

The mission expected "hidden_dim x 2 bytes (f16), ~5-15 KB per boundary".
The served path is FP32, not f16: every decode boundary lands in the
small-tensor band (n_backends >= 4 && ne < 262144 ->
ggml_backend_cuda_comm_allreduce_nccl, ggml/src/ggml-cuda/ggml-cuda.cu:1253-1261)
and reduces ncclFloat in place.  Hidden size verified from the served
model's gguf metadata (/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf):
qwen35.embedding_length = 5120, block_count = 65.  W8 census (section 2,
"verify pass 128 + catch-up 2 + draft 6 = 136") gives the exact table:

| segment | boundaries/cycle | ne | bytes fp32 | share | served per-boundary median (W8 census, die 3) |
|---|---|---|---|---|---|
| verify pass (T=4, 64 full layers x 2) | 128 | 20480 | 81,920 B (80 KB) | 94.1% | 129.0 us |
| catch-up (nextn block, T=4) | 2 | 20480 | 81,920 B | 1.5% | 125.9 us |
| draft chain (3 steps x 2, T=1) | 6 | 5120 | 20,480 B (20 KB) | 4.4% | 89.9 us |
| TOTAL | 136 | - | 10,772,480 B boundary payload/cycle | 100% | - |

Ring wire bytes: 1.5x payload per die per cycle (2(n-1)/n) = ~16.2 MB/die
- noise against the 3.08 GB/die MMVQ stream; the boundary is
latency/protocol/skew-bound, never bandwidth-bound (W9: transport flat
x1.07).  A one-shot pays 2x the wire bytes (3x pushes of ne/n each way,
W31 3.1) to buy 2 phases instead of the ring's 6 - the right trade ONLY
if the substrate latency is low, which is the thing this stack refuses
to provide.

## 3. THE EXISTING PATH (mission item 1) - where a custom AR would hook, and how invasive it is

Allreduce dispatch, ggml/src/ggml-cuda/ggml-cuda.cu:

- 1190-1218: ggml_backend_cuda_comm_context - holds backends, dev_ids,
  comms, ar_pipeline, nccl_up, and ONE function pointer
  `try_allreduce` set by the init chain.
- 1396-1441: init chain comm_init_{none, internal, nccl} - each step
  tries its resource, warns and falls through on failure.
- 1442-1505: top-level comm_init - GGML_CUDA_ALLREDUCE env picks
  nccl | internal | none; HIP DEFAULT = meta-backend butterfly; the
  SERVED config is GGML_CUDA_ALLREDUCE=nccl (E-090, +25-27% decode, the
  campaign's largest single win).
- 1516-1534: per-call ggml_backend_cuda_comm_allreduce_tensor - size
  gate (GGML_RCCL_PREFILL / GGML_RCCL_PREFILL_NE = 131072), then
  comm_ctx->try_allreduce.
- GGML_CUDA_ALLREDUCE=nccl flows into
  ggml_backend_cuda_comm_allreduce_nccl (1232-1315): in-place
  ncclAllReduce ncclFloat for ne < 262144 (all 136 decode boundaries),
  bf16-compress + fp32-convert above that (prefill class, E-101).
- Fallback contract: ggml/src/ggml-backend-meta.cpp:2358-2380 - the meta
  backend calls comm_allreduce per subgraph boundary and runs its
  butterfly on false.  Every served boundary passes through ONE choke
  point that already supports pluggable transports.

Invasiveness assessment (honest): a GGML_CUDA_ALLREDUCE=oneshot arm is
NOT a new subsystem - it is a fourth leaf in an existing chain: one init
step (enable-peer/IPC matrix + staging allocation), one try_allreduce
wrapper, one .cuh with the 4-rank kernel, ~2 small insertions in
ggml-cuda.cu, zero meta-backend changes (W31 section 3.4 had the
placement spec, and it survives this desk's verdict).  The real cons are
not structural: (a) the transport substrate is absent (1.1), so the arm
would be dead code; (b) a fourth transport arm is permanent review
surface on a port branch; (c) numerics sign-off needed (E-090 class:
one-shot's fixed per-chunk owner order differs from the ring fold -
bounded dust, bit-identical across dies by construction, W31 3.3).

## 4. PROTOTYPE STATUS

NOT IMPLEMENTED, deliberately.  Mission item 3 was conditional on P2P
viability; the condition is measured false (W31), so no kernel was
written.  Staged instead: the one measurement that could revive the desk
(section 6), built clean:

    docs/amd-port/probes/ipc_open_probe.cu   BUILD-EXIT:0, zero warnings
                                             (hipcc -O2 -x hip, AMD clang 18 roc-6.2.0)
    docs/amd-port/scripts/run_ipc_open.sh    lock-arbitrated runner, staged

The arm spec of record remains W31 section 3.4 (placement) + 3.1
(2-phase sliced PUSH kernel); this receipt adds nothing to it because
nothing changed that would improve it.

## 5. EXPECTED GAIN, IF THE SUBSTRATE EXISTED (for the record)

| item | value | source |
|---|---|---|
| transport floor today (ring, 80 KB fp32, n=4) | 69.5 us/boundary = 9.45 ms/cycle | W8 |
| one-shot band at 80 KB | 30-42 us/boundary | W31 section 5 (2 phases + local add) |
| one-shot at 20 KB (draft class) | 10-18 us/boundary | W31 section 5 |
| saving per 80 KB boundary | 28-40 us | derived |
| cycle saving | ~3.95-5.56 ms of ~124 ms | derived (130 x 80KB + 6 x 20KB) |
| decode gain, skew untouched | **+3.2-4.5%** | derived |
| absolute zero-transport ceiling | wall 28.7 -> 19.2 ms = +7.6% | derived, 136 x 141 us skew |
| mission's 8-12 ms band | NOT reachable by transport alone; needs skew compression bundled | this section |

For contrast, the mechanism-paired comparator: RCCL burst per-op 103.2 us
/ lockstep worst 126.5 us at 80 KB (W9 inproc probe) - a working one-shot
must beat BOTH to serve, plus the KLAUNCH 11.6-12 us floor it inherits.

## 6. THE STAGED PROBE + PAIRED-WINDOW PLAN (what runs when the GPU queue frees)

Stage 0 - IPC-open FEAS (~30 s GPU, answers section 1.3's two open rows):

    bash docs/amd-port/scripts/run_ipc_open.sh
    # lock check-and-hold (/tmp/campaign_gpu_boot.lock), cool-die < 60 C,
    # die-idle gate, line-buffered log to docs/amd-port/results/W38_ipc_open_<stamp>.log
    # per ordered pair: GET -> OPEN(hipIpcMemLazyEnablePeerAccess) ->
    # kernel read through opened ptr (magic + 1 MiB strided sum) -> CLOSE,
    # plus an independent hipMemcpyPeerAsync round trip; watchdog PROBE-EXIT:3

  - Expected outcome (priors from W31): OPEN=FAIL on all 12 pairs ->
    "IPC verdict: P2P-via-IPC NOT available" -> desk CLOSED with the
    matrix complete; this receipt becomes final and no arm is ever built.
  - Surprise outcome (any pair OPEN+READ ok): go to Stage 1.  Note: the
    probe runs single-process; a HIP same-process-open refusal is itself
    a verdict (IPC path would need multi-process serving - inapplicable
    to llama.cpp inproc TP without a serving-model rearchitecture).

Stage 1 - bench (only on a Stage-0 pass; ~3 min GPU): revive W31's BENCH
(docs/amd-port/probes/p2p_allreduce_probe.cu on amd/p2p-ar, which already
implements the flat kernel and pacings) with the ipc mapping as the
staging substrate: --arms p2p,host --sizes 20480,5120 --pacing
lockstep,burst, 2000 iters.  Gates to pass: bit-exact vs CPU same-order
reference, cross-die bit-identity, and strictly better than 103.2 us
burst / 126.5 us lockstep at 80 KB.  Any wedge = instant kill (E-144
repeat).

Stage 2 - arm build (only on a Stage-1 pass): implement
GGML_CUDA_ALLREDUCE=oneshot per W31 3.4 + section 3 of this receipt
(init step + try_allreduce + kernel .cuh, fp32 band gate ne < 262144,
prefill sizes untouched, env unset = exactly today's behavior); build
with the canonical flags: cmake -B build-hip -DGGML_HIP=ON
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DCMAKE_HIP_ARCHITECTURES=gfx900
-DGGML_HIP_RCCL=ON -DLLAMA_CURL=OFF.

Stage 3 - paired window (the A/B that decides): battery vehicle (E-142),
29-round exact-window convention (W8), lock-arbitrated, cool-die < 60 C,
temp 0, same 4k prompt/128-token probe:
  - control: BASEENV of record + GGML_CUDA_ALLREDUCE=nccl
  - arm:     BASEENV of record + GGML_CUDA_ALLREDUCE=oneshot
  - primary: battery decode t/s; secondary: per-boundary tl_comm_us +
    tl_bounds from the meta telemetry (LLAMA_LAUNCH_TIMELINE,
    ggml-backend-meta.cpp), per-die boundary medians
  - success cell: die-3 served boundary median < ~180 us sustained
    (i.e. the 28-40 us/boundary saving survives serving) => cycle -3.5%
    or better
  - numerics gate: E-090 class - two-boot same-sha determinism + owner
    sign-off of the changed fp32 sum order before the arm is served
  - kill: any wedged boundary (tl_comm_us > 10 ms) or dust beyond the
    E-090 band -> arm env off, desk closed.

## 7. RECOMMENDATION

Keep the desk closed unless Stage 0 surprises.  The campaign's live
allreduce lever stays arrival-skew compression (clock floors,
E-147/E-148): it attacks the ~141 us term a transport cannot touch, is
worth up to +2.4-4.8% realistic by the dossier's own band, and needs no
new transport substrate.  The one-shot idea should be re-filed ONLY
together with that work if Stage 0 passes.

## LOG (append-only, newest last)

- 2026-09-24 16:05Z desk opened; worktree wt-oneshot-ar on
  amd/oneshot-ar at 4bf0d6476; zero-GPU phase.
- 2026-09-24 16:4xZ W31 logs + dossier + W8/W9 read; allreduce path
  mapped (section 3); gguf metadata parsed (hidden 5120, 65 blocks);
  boundary table fixed (section 2); verdict written (section 1);
  IPC-open probe staged + compiled BUILD-EXIT:0, zero warnings; runner
  staged; WIP commit 6b0cbab3b; origin push of real history rejected
  (GitHub 100 MB limit, E-115 known) -> orphan snapshot pushed
  backup/oneshot-ar (tree of 6b0cbab3b) per the campaign backup law.
