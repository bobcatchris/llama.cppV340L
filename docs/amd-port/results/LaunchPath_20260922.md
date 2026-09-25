# LAUNCH/REPLAY-PATH desk: launch/sweep cost table + honest ceiling (2026-09-22)

Desk: wt-launchpath, branch amd/launchpath. Instrumentation first deliverable in
8a9d287cb (LLAMA_LAUNCH_TIMELINE=1, LLAMA_TARGET_LIGHT_SYNC=1,
GGML_CUDA_COMPAT_CACHE=1). This file: profiler output against the logs on disk,
the static audit of the target-side launch/replay path, and the ceiling doc.

## 1. What could and could not be measured offline

The mtpgain_T1/F1 logs on disk (main checkout docs/amd-port/results/) predate
the launch-timeline instrumentation: they carry [decode-timeline] /
[spec-timeline] lines only (captured at -lv 4, LLAMA_DECODE_TIMELINE=1), no
[launch-timeline] lines. T1 was captured at -lv 3 and has no timeline lines at
all. The launch/replay/sweep split therefore requires one boot of this branch
with LLAMA_LAUNCH_TIMELINE=1 -lv 4; that validation arm runs in the
coordinator's window (E-096). The profiler
(docs/amd-port/scripts/profile_launch_timeline.py) was extended with a
decode-only fallback so it still produces the derivable table against the
existing logs, and it parses the [launch-timeline] lines unchanged once the
boot exists.

## 2. Cost table derived from the on-disk logs (F1-F4, 3 dies, butterfly AR)

All numbers ms unless stated. n=85 verify rounds per log, consistent across F1-F4:

| quantity                          | F1 med | F2 med | F3 med | F4 med |
|-----------------------------------|--------|--------|--------|--------|
| verify issue (n=4, issue>40)      | 163.16 | 162.30 | 162.39 | 162.31 |
| verify build (graph reuse)        |  0.000 |  0.000 |  0.000 |  0.000 |
| verify inputs (host setup)        |  0.040 |  0.039 |  0.040 |  0.040 |
| catchup issue (n=4, issue<=40)    | 11.66  | 11.54  | 11.17  | 11.64  |
| catchup inputs                    |  0.026 |  0.027 |  0.026 |  0.027 |
| drain per call (med / max)        | 0.001 / 2.27 | 0.001 / 2.27 | 0.001 / 2.25 | 0.001 / - |
| drain calls per ubatch (med)      | 5.5    | 5.5    | 5.5    | 5.5    |

Graph structure at boot (F1 reserve log): target graph 4279 nodes, 2 sched
splits, 1 sched copy; meta device over 3 ROCm devices; at mtpgain time the
backend allreduce init failed (n_devices != 2) so all boundaries ran the
butterfly fallback (today's serving config has RCCL signed off, E-094).

Reads:
- The verify block is device execution. Host build + inputs are 0.04 ms of a
  163 ms block (0.02%).
- Post-issue drains: median 5.5 calls/ubatch but only the first carries the
  real device wait; the extras find an empty queue at ~1-2 us each. Redundant
  sweep cost is of order 10 us per round, not a target.

## 3. Static audit of the launch/replay path (per verify ubatch)

Structure: 9 subgraphs x 3 dies = 27 die graph_compute calls + 8 allreduce
boundaries. Per die call in steady state (ggml-cuda.cu
ggml_backend_cuda_graph_compute):

- compat scan ggml_cuda_graph_check_compability: O(n_nodes) memcmp-class scan
  per call -> cached per uid by GGML_CUDA_COMPAT_CACHE=1 (8a9d287cb). uid==0
  graphs are never cached (no false hits).
- ggml_cuda_graph_update_required: O(1) when the cgraph uid is stable
  (upstream design); the full properties rescan runs only on rebuild, and
  subgraph uids change only on rebuild of the meta subgraphs (not per ubatch).
- replay path in evaluate_and_capture: no node loop, no allocations, no syncs,
  straight cudaGraphLaunch. The launch enqueue is the floor (~10-30 us on
  ROCm).
- ggml_cuda_set_device per call: already self-guarded by cudaGetDevice
  compare (~1 us).

Findings vs shipped fixes:
- Scheduler input-copy drains (the target-side equivalent of the draft-side
  5-6 -> 1): the meta backend has caps.events=false, so the events==NULL path
  drains the whole meta backend (all 3 devices) before EVERY input copy.
  Shipped env-gated as LLAMA_TARGET_LIGHT_SYNC=1 (8a9d287cb): one drain before
  the copy loop, strictly stronger ordering, identical copied bytes. Verified
  sound in this audit.
- Per-replay compat rescan: shipped env-gated as GGML_CUDA_COMPAT_CACHE=1.
  Verified sound (verdict is topology-fixed per uid; uid==0 never cached).
- Remaining candidates all measured/estimated below ~25 us per ubatch total
  (boundary node vector allocations ~ns each x8; set_device ~27 us; empty
  drains ~10 us). Not worth gating risk against a 163 ms block; left alone.

## 4. Honest ceiling

Host launch/replay/sweep budget per verify ubatch, from the structure above:

| component                          | estimate            |
|------------------------------------|---------------------|
| 27 cudaGraphLaunch floors          | 0.27 - 0.81 ms      |
| compat scans (uncached) x27        | 0.1 - 0.3 ms        |
| 8 boundary enqueues (RCCL or butterfly) | 0.08 - 0.4 ms  |
| sched input copies + inputs        | ~0.04 - 0.10 ms     |
| redundant empty drains             | ~0.01 ms            |
| total recoverable host budget      | <= ~1 ms of 163 ms  |

Even a zero-host-cost launch path recovers under 1% of the verify block. The
163 ms is device execution: die GEMM/attention kernels plus allreduce
transport. Launch-path work (this desk) is bounded and now instrumented;
it cannot move the served decode rate meaningfully by itself.

What the >100x larger wins would require (design-only, not proposals):
- Persistent kernels / device-resident layer loop: removes the per-subgraph
  host serialization entirely; requires the allreduce boundaries to become
  device-side barriers (cooperative groups / grid sync) and a rewrite of the
  meta-backend scheduling model.
- Mega-graph with conditional nodes: one graph for all 9 subgraphs with
  RCCL allreduce captured as graph nodes; collapses 27 launches to 1 and
  removes boundary host round-trips. Needs capture-safe collectives and
  conditional-node support in HIP; medium project.
- Device-side graph launch: host enqueues once per round, device chains the
  graphs; ROCm support for device graph launch is immature; saves the
  remaining per-round host slice only.

These attack the same <=1 ms slice, not the 162 ms kernel+transport body.
The device-side levers (kernel time at bs=4, allreduce transport, MTP round
structure) live on the other desks.

## 5. Served-arm spec for the validation boot (coordinator window)

Boot this branch, 3 dies, serving config, with:
  LLAMA_LAUNCH_TIMELINE=1 LLAMA_DECODE_TIMELINE=1 <served env> -lv 4
Expected per verify ubatch in the log:
- 27 "[launch-timeline] cuda ... mode = replay" lines (3 dies x 9 subgraphs),
  med host expected 10-40 us each, check ~0 when GGML_CUDA_COMPAT_CACHE=1.
- 1 "[launch-timeline] meta nodes = ..., subs = 9, replays = 27, bounds = 8"
  line; comm vs fb splits confirm which allreduce path served.
- 1 "[launch-timeline] sched splits = 2 ..." line per ubatch; with
  LLAMA_TARGET_LIGHT_SYNC=1 the csync delta per round should drop from
  n_inputs per split to 1.
Byte-exactness guards: run the arm twice (env unset vs set) and diff
completion tokens; host suites must stay green (test_launch_timeline_host
covers the counter semantics).
