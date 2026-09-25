# SKEW MEASUREMENT DESIGN — deciding H1 (dev0 display-die lag) decisively, no root

**Date:** 2026-09-17 · **Seat:** NO-GPU analysis (amd/tp4-cure) · **Parent:**
`docs/amd/VERIFY_LAUNCH_AUDIT_2026-09-17.md` §4 (~140 ms/round unowned; H1 = cross-rank
skew amplified by 128 serialized ARs/round, dev0 = desktop/display die).
**Analyzer:** `tools/optrace_analyze.py` (selftest: `results/amd/coherence/optrace_selftest.log`).

## Instrument (one boot per arm, GRAPHS-ON, fixed prompt, count >= 200 rounds)

```
NINFER_TP2_OPTRACE=1 NINFER_TP2_OPTRACE_FILE=/tmp/optrace_<arm>.log <same bin, same sha>
python3 tools/optrace_analyze.py /tmp/optrace_<arm>.log --json /tmp/optrace_<arm>.json
```
Bank the `[tp2] single-seq decode` line per arm too (it is the analyzer's wall fallback).

## The hypothesis, made falsifiable

dev0 is the desktop die: Xorg/gnome (and RustDesk when a viewer is connected) render on
card0 = HIP dev0 = rank0 (host mapping measured 2026-09-13: card0 -> rank0's device).
Display contention steals dev0 bandwidth/scheduler slices; rank0's verify phases run long;
all 128 per-round ncclAllReduce wait for the slowest rank, so the lag propagates to every
rank as AR wait. **Prediction:** rank0's phase ms inflated AND other ranks show inflated
`ar` (wait-inclusive) AND round wall tracks rank0's phases.

## Falsifier A — display activity flip (NO root needed)

Same binary sha, back-to-back windows, count >= 200 each:
- **A1 QUIET:** RustDesk DISCONNECTED (no viewer; console idle, no video playback).
- **A2 LOADED:** RustDesk CONNECTED with an active viewer (continuous refresh on card0).

| outcome | meaning |
|---|---|
| A2 vs A1: rank0 verify/align ms up >10%, verdict flips to SKEW rank0, others' `ar` grows with it, wall grows likewise | **H1 CONFIRMED** — display contention on dev0 owns the hole. Mitigate: viewers off while serving, console on another GPU, or reorder (B). |
| A2 vs A1: rank0 phases and wall move <2-3%, skew unchanged | **H1 KILLED** as dominant owner — re-point at H2 (in-graph AR cost) / H3 (fat op class) using the same OPTRACE files. |
| All ranks grow together (verdict COMPUTE), rank0 not special | Load perturbs a shared resource (PCIe/DRAM) — rerun A with RustDesk connected but idle viewer before concluding. |

Corroboration (no root): `rocm-smi --showuse` / `cat /sys/class/drm/card0/device/gpu_busy_percent` sampled during both arms; QUIET = card0 near-idle.

## Falsifier B — die/rank reorder (does the hole follow the DIE or the RANK INDEX?)

Relaunch rotated so the display die is NOT rank0: `--devices 1,0,2,3` (dev0 served by rank1).

| outcome | meaning |
|---|---|
| Skew moves to rank1 (its phases reproduce A2's rank0 pattern) | Skew follows the **DIE** — display contention. Fix is environmental: quiesce display, or serve on dev1-3 only while dev0 is dirty. |
| Skew stays on rank **index 0** (any die) | It is the **RANK-0 ROLE** (lead-rank host work: sampler/egress/log/D2H orchestration). Fix is software: rebalance rank-0 duties. |
| No skew in either ordering (COMPUTE/BETWEEN-PHASES) | H1 dead on both legs; follow the analyzer's verdict block to kernels or gaps. |

## Verdict reading (analyzer contract)

- `SKEW` + named rank = one rank is max in >= 60% of rounds -> run B (die vs index), then A (display).
- `COMPUTE` = ranks burn equally; hole is real kernel work -> per-op probes next (H3).
- `BETWEEN-PHASES` = slowest rank's phases do not fill its wall -> host sync/D2H/scheduler
  gaps own it; extend instrumentation to round boundaries.
- Discipline: same bin sha per A/B pair; warm rounds excluded (default 3); >= 200 rounds;
  decide on means, bank p95 (jitter is the signature); never compare across geometry.

## Cost

Zero new hardware, zero root: 2 boots for A, 1-2 for B, ~1-2 min decode each at count=200. Analyzer + selftest are host-only (no GPU, no KFD context).
