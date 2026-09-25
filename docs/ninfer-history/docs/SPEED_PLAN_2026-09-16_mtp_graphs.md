# SPEED PLAN — MTP + graphs land tonight; the measured A/B and the next 5 levers

**Date:** 2026-09-16 evening · **Seat:** fix seat continuation · **Binary:** banked
`ninfer-serve_b864220912aec151.bin` (coherence-green, BUGTRACK E-14/E-16) · **Grant:**
user direct order ("serve with mtp now… work at maximum capacity"), window G-AMD-MTP*.

## 1. MTP WORKS ON THIS LINE — first boot ever, and it's fast

The artifact always carried the drafter (`mtp/*` classified at `tp_load.cpp:450`, filtered on
every prior boot — WO_MTP_1 was written and never fired). Tonight, k=2:

| config (same request: "Count 1..200", plen 67, mt 600) | decode wall | tok/s out | acc | tok/round |
|---|---|---|---|---|
| MTP k=2, graphs OFF (`--no-cuda-graph`) | 99.50 s | 6.03 | 0.97 | 2.94 |
| **MTP k=2, graphs ON** (default; just drop the flag) | **58.35 s** | **10.28** | 0.97 | 2.94 |
| non-MTP, graphs ON (baseline, same shape) | 180.32 s | 3.33 | — | 1.00 |

- **MTP+graphs = 3.09x over non-MTP+graphs, 1.70x over MTP-no-graphs.**
- `--no-cuda-graph` was debugging-era caution, never a correctness mandate (the written
  rationale, DETERMINISM_TEST_INVENTORY_agent1:36, is about *measurement comparability*).
  Graph capture is shim-implemented (`hip_shim/cuda_runtime.h:408-414`) and PG-A-certified;
  tonight is its first TP4 world=4 service — coherence legs GREEN under it (14/14 fast ladder,
  BLUE@66 exact, 10k leg this window).
- MTP verify-phase conv safety (E-13 gate): verify rounds run T=k+1=3 ≤ 16 → smallt kernel,
  never the T≥65 pairs route. No interaction with the convicted bug.
- **New recommended serve line** (throughput config):
  `ninfer-serve <art> --port P --devices 0,1,2,3 --prefill-chunk 128 --no-prefix-reuse
  --prefix-cache-capacity 256 --greedy --spec mtp --draft-tokens 2` (no --no-cuda-graph).
  Cost: KV capacity 63936 → 36352 tokens (drafter state, measured at boot).

## 2. WHERE THE REMAINING TIME GOES (phase-separated, measured tonight)

- **Prefill (unchanged by MTP):** ~9.7 tok/s mean at 10k (14.5 early → ~9 deep), decay
  measured since A4/M1 (34-35% both dial states, cause unnamed). The 10k MTP leg this window
  re-means it with graphs on — watch MTP3_serve.log.
- **Decode short-ctx:** 10.28 tok/s with MTP+graphs vs 3.33 baseline. The historical "~8"
  figure and today's 3.33 are different shapes; the A/B on ONE shape is the datum that counts.
- **Decode post-deep-prefill:** crisis number was 2.1-2.6 tok/s; the 10k MTP+graphs leg gives
  the MTP answer tonight (banked in BUGTRACK E-17).

## 3. NEXT LEVERS, in order (each priced by tonight's data)

1. **AR-count budget (largest named lever).** Host-staged AR RTT ~98-101 µs flat (COORDINATOR
   measured facts); eager runs ~130 AR/step ≈ 13 ms/step dead; 2-4-layer batching →
   3.3-6.5 ms. With graphs on, per-round cost is ~286 ms (MTP k=2) / ~300 ms (non-MTP) —
   still ~10-30x above the AR floor, so launch/sync structure, not AR alone, dominates; the
   AR-batching falsifier (count collects per round from the k=2 log) prices it before any
   engineering (SD-1a).
2. **MTP k-sweep completion:** k=1 and k=3 at the same shape (draft-tokens range [0,3]).
   k=2 already nets 2.94/3.00 — headroom ≤ 2%; k-sweep is cheap but low-yield at this
   acceptance; re-run the sweep on a REAL (non-synthetic) prompt where acceptance < 0.97.
3. **Per-round sync audit (MTP round path):** 487 ms/round (graphs OFF) → 286 ms (ON) means
   ~200 ms was launch overhead; the remaining 286 ms across draft+verify forwards vs ~125 ms
   single-step says one extra serialization lives in the round (host sync between draft and
   verify, or draft head W8G32 round-trips). Recon report names the sites; one audit cell
   (per-round trace) decides.
4. **L-10k deep row with MTP+graphs** — the pre-designed sheet (NVFP4_AMD_PLAN:863) fired
   tonight in miniature; run the full pre-designed row (early/late decode windows) on a quiet
   window.
5. **Prefill decay hunt** — still unnamed; candidates (KV-growth attention cost, chunk
   boundaries, power/thermal) need the per-chunk tok/s curve the logs already carry.

## 4. LAWS RESPECTED (unchanged)

Phase-axis (never mix prefill/decode columns); VRAM law (all numbers above are measured;
preflight is live); bank-before-relink; named grants, serial TP4 boots, KFD-0 pre-spawn,
exact-PID kills; RED/GREEN closure for any code change (tonight changed NO code — flags only).

— Rows: results/amd/coherence/MTP1_*, MTP2_*, BASE2_*, MTP3_10k_*; BUGTRACK E-17.

## 5. §3 ADDENDUM — TIMING1 (bin 07ad7eccc0b97cc0, NINFER_TP2_TIMING=1, 2026-09-17 ~01:00 UTC)

One instrument boot (commit d32e6c53) fired the count600 shape under the B1 8-phase breakdown
(row: results/amd/coherence/TIMING1_row.txt; block: TIMING1_serve.log:111-122). 211.95 ms/round
mean over 203 rounds, and the verdict is decisive:

- **§3.1 (AR-count) is PRICED AND DEMOTED.** AR Draft Chain = 7.61 ms/round (3.6%) — at the
  98-101 µs RTT law that is ~76-78 serialized collectives, but even zeroing it moves the round
  under 4%. With Alignment Forward 8.56 ms (4.0%) and Propose 0.29 ms, the ENTIRE draft side
  ceiling is 16.46 ms = 7.8%. AR-batching is not where the time is.
- **§3.3 (per-round sync audit) is ANSWERED.** Accept/D2H 0.06 + Rebase 0.01 + Prepare 0.08 +
  Select 0.01 = 0.16 ms — phases 2+3+4+6+7 sum to 0.45 ms (0.2%). There is no hidden host
  serialization between draft and verify at this granularity; the graphs-on win already took
  the launch overhead that lived there.
- **THE OWNER is phase 1, Target Verify (T=k+1=3): 195.34 ms = 92.2%.** The lever that matters
  is the TARGET-MODEL step cost — the TP4 target forward itself (195 ms for 3 tokens vs ~125 ms
  single-step reference; the single-step base is itself 10-30x above the eager AR floor,
  PLOG-005). Next falsifier: a T-sweep of the verify forward (T=1 vs 3 vs 4 timings from the
  k-sweep boots, same instrumentation) to split fixed-step cost from per-token cost inside the
  verify forward. If T=3 ≈ T=1, per-round cost is step-count-bound and the MTP round is already
  optimal at this step cost — the lever moves entirely to making ONE target step cheaper.
- Run-to-run anchor: same shape on MTP2 (b864220912aec151) was 286 ms/round / 10.28 tok/s;
  tonight 211.95 ms/round / 13.87 tok/s (engine-side decode 43.26 s / 600 tok). Instrumented
  diff is flags-only; variance unnamed, both rows cited, not averaged.

## 6. BACKLOG (named 2026-09-17 night, NOT started — queue for the next windows)

1. **Verify-forward T-scaling (the §5 owner, now measured):** verify T-sweep
   intercept ~126.5 ms @T=1 (TIMINGK1/K3) is ~10x off the memory-bound ideal for
   one TP4 NVFP4 step — the lane is LAUNCH-BOUND. Collective capture is blocked
   by RCCL on this stack, so the honest lever list is: launch-count reduction,
   AR-layer batching INSIDE the forward, or a weight-streaming layout. Not
   started; needs the B1 8-phase instrumentation extended per-layer first.
2. **k-table transfer question:** the synthetic count600 k-table (k3 still pays,
   marginal 22.5 -> 19.1 tok/s) does not transfer to prose (PLOG-014: acc 0.54
   at k=3). Needed: real-prompt k=2 vs k=3 head-to-head at the SAME prose
   prompt (tonight measured k=3 real only; MTP1's k=2 real anchor was
   graphs-OFF). One boot per arm, idle-gap law applies.
3. **Thermal integrator identification (from THARM1):** the sustained-load
   trigger is named but not localized (junction/hotspot vs VRM vs rolling
   power-cap). Needs either privileged clocks (sudo grant for rocm-smi set
   verbs — B1a's forced A/B is ready to fire the moment the box grants it) or
   junction-temp telemetry (temp2_input) correlated against the sysfs sampler.
   Cool-recovery time constant (2-3 min bracketed, not measured) belongs here.
