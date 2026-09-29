# W12 HOST-SLICE DESK - what the host does in the 6.5 ms slice (P0 + A1 + A2 + A3)

Date: 2026-09-23. Desk: HOST-SLICE (wt-host-slice, amd/host-slice; P0 probe + recovery
checkpoint 66bf25a23 from the predecessor agent; continued to A1/A2/A3 by the
continuation agent). Round-map item: "host slice A: post-verify 6.5 ms, all 4 dies
idle 108.4->115.0" (TP4_roundmap_2026-09-23.md row 2 / E-115a), desk prize 2-4 ms.

## 0. Instruments and provenance

- P0 boot (recovered): canonical env + LLAMA_LAUNCH_TIMELINE/DECODE/SPEC_TIMELINE
  at -lv 4, lane 8083, TP4 + draft-mtp, 32k slot (stamp
  W12_hostslice_p0_stamp_20260923_165028.txt).
- A1 INSTRUMENT OF RECORD: docs/amd-port/results/W12_hostslice_diag_tl_165820.log
  (the P0 diagnostic boot, tl arm; 40890 timeline events; warmup-driven spec-decode
  rounds on the canonical stack). Parser: docs/amd-port/scripts/profile_hostslice_rounds.py
  (extended in analysis; per-round state machine over [decode-timeline]/
  [launch-timeline]/[spec-timeline] lines with wall timestamps). 37 rounds parsed,
  34 steady-state. NO additional GPU boot was used (A1 completed from the captured
  log; the one permitted boot went unused).
- Cross-checks: round period med 123.4 ms (vs census 130.5 with rocprof attached,
  vs 129.0 battery) - structure consistent, absolutes shifted by instrumentation.

PROVENANCE DEFECT (recorded): the P0 probe server log
(W12_hostslice_p0_server_20260923_165028.log) holds only boot lines (ends at
"warming up", zero timeline lines) while the probe JSON completed (8551-token
prompt, 27.14 t/s, 128 tokens, accept 90/109) and the stamp lacks the
boot_ready/probe_rc appends. The P0 session's exact history is unrecoverable from
artifacts; the desk re-ground A1 on the diag tl log. The probe JSON does prove the
timeline-env boot serves correctly.

## 1. Predecessor-edit audit (the unverified backend-meta.cpp / ggml-cuda.cu changes)

Both recovered edits are LOGGING-ONLY diagnostics, env-gated under the launch
timeline flag, byte-exact by construction (no device-side or ordering change):

- ggml/src/ggml-backend-meta.cpp:1863-1869: one INFO line when a mid-serving meta
  graph rebuild fires (uid flip), to attribute re-capture cost.
- ggml/src/ggml-cuda/ggml-cuda.cu:5373-5376: one INFO line when CUDA graph warmup
  resets on properties_changed.

Verdict: SOUND. Variables in scope at both sites, %llu casts correct for the
uint64 uid (ggml-impl.h:346, ggml-backend-meta.cpp:1672), compile verified by the
full build below. HYPOTHESIS REFUTED BY DATA: the trigger does not fire in steady
decode - 36/37 verify passes are pure replay (meta replays/ubatch = 516, zero
captures), and CUDA-graph capture happens ONCE at boot (verify pass 0: 516
captures, 222.4 ms host, one-time; diag log capture histogram). The lines are kept
as cheap tripwires for the capture-class regression, but they are not a lever.

## 2. A1: sub-item decomposition of the host slice (steady-state medians, n=34)

Round period 123.4 ms (verify issue start -> next). Wall accounting: verify issue
11.8 + drain 97.6 + slice A 2.04 + bookkeeping 0.16 + draft 10.6 + gap 0.14.

| # | sub-item | ms | code path (file:line) |
|---|----------|----|-----------------------|
| 1 | verify issue (host, OVERLAPPED by device) | 11.836 | llama-context.cpp:1640-1659; meta graph_compute ggml-backend-meta.cpp:1849+ |
| 1a | - graph build | 0.000 | graph cache hit, reused=1 (llama-context.cpp:1586-1592) |
| 1b | - set_inputs | 0.044 | llama-context.cpp:1628-1636 |
| 1c | - RCCL allreduce host enqueues | 5.370 | 128 comm x ~42 us; meta comm path ggml-backend-meta.cpp:2282-2320 |
| 1d | - per-die replay launches + sched walk | 5.17 | 516 replays x ~7 us (meta host 10.543 - 5.370 ar) |
| 2 | verify drain (device-bound, NOT host) | 97.6 | llama-context.cpp:724-763 synchronize() |
| 3 | SLICE A (all-dies-idle window) | 2.04 | drain line -> catchup enqueue |
| 3a | - accept loop: 4-row sample+accept | ~1.7 | server-context.cpp:3982-4005; common/sampling.cpp:837-897 (row path); timer added in A3 |
| 3b | - draft-ctx catchup decode (in process()) | 1.345 | speculative.cpp:1610-1633 (non-staged branch); dec line build 0.031 + inputs 0.030 + meta 0.327 (ar 0.151) + sched residual 0.96 |
| 4 | post-catchup bookkeeping | 0.16 | process line + wait_outputs 0.004 (llama-context.cpp:793-821) |
| 5 | draft section (3 steps) | 10.6 | speculative.cpp:1663-1907 |
| 5a | - draft-ctx decode issue, shape-rebuild steps | 2.593/step | reused=0: sched re-split walk ~1.5 ms penalty (llama-context.cpp:1579-1614) |
| 5b | - draft-ctx decode issue, reused steps | 1.108/step | reused=1 (68/111 of n_tokens=1 decodes) |
| 5c | - packed-fetch drain per step (device wait) | 1.665/step | llama-context.cpp:793-821 wait_outputs; LLAMA_DRAFT_PACKED_GET/LIGHT_SYNC |
| 5d | - draft sampling + batch | 0.053/step | speculative.cpp:1750+; LLAMA_DRAFT_FAST_TOPK |
| 6 | gap: draft end -> next verify issue | 0.142 | server loop |

Reconciliation with the census 6.5 ms all-die-idle window: the tl instrument splits
it into slice A host 2.04 (accept 1.7 + catchup enqueue 0.34 host) + catchup launch
latency + stretches the single-die kernel view lumps into its "catch-up" row; the
draft loop adds ~1.5-3 ms of device idle (serialization around the blocked syncs).
TOTAL host-exposed pool: ~5-8 ms/round - the desk prize lands in the LOWER band of
the round map's 3-5 ms plus the draft slice.

KEY MECHANISM FOUND (new, named): the draft context's decode shapes alternate
4 (catchup) -> 1,1,1 (steps) every round, and llama_context keeps a SINGLE reuse
slot (gf_res_prev, llama-context.cpp:1579-1599) - so 2 of the 4 draft-ctx decodes
per round re-split the sched from scratch (reused=0, issue 2.593 vs reused=1
1.108; the ~1.5 ms delta is the re-split walk, paid on an IDLE device).

## 3. A2: ranked levers

| rank | lever | ms/round | class | size | disposition |
|------|-------|----------|-------|------|-------------|
| 1 | draft-ctx graph/sched shape cache (reuse slot keyed per shape, or last-K for small graphs; draft graphs are 49 nodes) | ~2-3 (2 re-splits x ~1.5) | byte-exact (host-only; reuse path already exercised by the target ctx every round) | M | hand design back - touches llama_context core state |
| 2 | accept-loop greedy fast path: set_logits_row builds a 2.07 MB full-vocab array per row x4 (common/sampling.cpp:164-176) at temp 0 where the outcome is the argmax | ~1-1.5 | numerics-ADJACENT (sampler semantics; needs chain-shape proof at temp 0 + oracle + owner sign-off) | S-M | flag for owner |
| 3 | draft-step pipelining: issue step k+1 under step k's packed-fetch drain tail | 1.5-2.5 | byte-exact (host reordering, event-gated; LIGHT_SYNC/PACKED_GET infra exists) | M-L | hand design back (round map lever 4 confirmed) |
| 4 | NEGATIVE: verify-issue host cost (11.8; RCCL enqueues 5.4) | 0 | - | - | fully overlapped by the ~108 ms device verify; dead as a wall lever |
| 5 | NEGATIVE: steady-state meta rebuild / CUDA-graph recapture | 0 | - | - | does not occur (36/37 replay rounds); predecessor's hypothesized trigger refuted |

## 4. A3: implementation verdict

- Predecessor's edits: AUDITED SOUND (logging-only, gated, correct scope), KEPT.
- Completed the attribution instrumentation with ONE timer: the accept stretch
  (largest unnamed host item, ~1.7 ms) now prints "[spec-timeline] accept:
  rows = N, sample+accept = X ms" under LLAMA_SPEC_TIMELINE
  (tools/server/server-context.cpp:3985-4005). Byte-exact, env-gated.
- No speed lever shipped: every ranked lever is M-size or numerics-adjacent;
  per the desk law the designs are handed back (levers 1-3 above) instead of
  forcing an unsound or unreviewable change.
- Zero additional GPU boots consumed (the one permitted boot went unused).

Build (canonical flags: GGML_HIP=ON, Release, GGML_NATIVE=ON,
CMAKE_HIP_ARCHITECTURES=gfx900, GGML_HIP_RCCL=ON, LLAMA_CURL=OFF,
/home/chris/opt/cmake/bin/cmake, fresh configure of the worktree build-hip):
BUILD-EXIT:0 (see section 6 for the log line).

Pre-merge CI: /home/chris/run_premerge_ci.sh run against THIS worktree tree via a
path-patched copy (the script hard-codes the coordinator's checkout); verdict in
section 6. The coordinator-side gate at merge time re-runs it on the main tree.

## 5. Defects of record

1. Predecessor hypothesis refuted: no mid-serving rebuild/recapture in steady
   decode (section 1) - instrumentation kept as tripwire, not a lever.
2. P0 probe log truncation (section 0): boot-only server log vs completed probe
   JSON; A1 re-ground on the diag tl log.
3. Draft-ctx reuse defect (section 2, the lever-1 mechanism): single-slot graph
   reuse defeats the 4->1->1->1 shape alternation; ~3 ms/round of idle-device
   re-split walk.
4. Instrument caveat: the sched sync_us cumulative counter over-counts relative
   to wall (deltas exceed the enclosing decode, e.g. +3.5 ms inside a 2.6 ms
   issue) - concurrent/overlapping waits; use sync COUNTS, not the us deltas,
   for wall claims.

## 6. Build + CI results (this tree, amd/host-slice @ accept-timer commit)

- Configure: /home/chris/opt/cmake/bin/cmake -B build-hip -DCMAKE_BUILD_TYPE=Release
  -DGGML_HIP=ON -DGGML_NATIVE=ON -DCMAKE_HIP_ARCHITECTURES=gfx900
  -DGGML_HIP_RCCL=ON -DLLAMA_CURL=OFF -> CONFIGURE-EXIT:0
- Full build: BUILD-EXIT:0, zero compiler warnings (only deprecation-warning.cpp
  source filenames match "warning").
- Pre-merge CI (path-patched copy of /home/chris/run_premerge_ci.sh, same tests,
  -L/rpath redirected to this worktree's build-hip): CI-VERDICT: PASS - tree
  hygiene clean, 6/6 host suites, 7/7 gate-wiring types engaged + 7/7 no-env
  controls silent. First run FAILED hygiene only: CHECKIN.log was
  gitignored-but-tracked in this branch (E-116 law fix: untracked, kept on disk).
- GPU: lock discipline - found the boot lock STALE (holder attn-fa W11 A3 pid
  21063 dead by kill -0, dies idle, junctions 27-30 C); removed with evidence,
  held for the CI's die-3 wiring run, CI released it (its own line 81).
