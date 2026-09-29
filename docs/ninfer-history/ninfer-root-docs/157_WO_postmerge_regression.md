# WO-G5 (staged) — post-merge no-regression sweep: groupwise default + NVFP4 path on merged main

**Trigger:** fires immediately after the decisive NVFP4 cell AND the `wo/bf16-w8-requant` → main merge.
**Window slot:** one ~35-min window, both cards, port **8099** (coordinator ruling 22:20Z: probe
convention wins post-merge; 8091 remains the run_ci scripts' internal number), right after the
decisive-cell window releases (no idle gap — this IS the merge gate).
**Owner:** agent1 (execution) + agent2 (backup) + coordinator (interprets with A1) — re-assigned
from the original "gemini (execution) + A1 (interpretation)" at coordinator 22:20Z (gemini down;
WO-G5 was staged before the lane shift). Precondition addition: /tmp prompt bodies are
reboot-volatile — regenerate per results/157_SESSION_DEBRIEF_20260908.md §3 pattern before the
window, don't trust mtimes.
**Purpose:** prove the merged main (NVFP4 path + fail-closed gate + loader fixes) does not regress the
shipped groupwise default, and that both weight paths serve on the same binary. This is the merge bar.

## Preconditions (all CPU, before the window)

- [ ] Merge `wo/bf16-w8-requant` → main is complete; main HEAD recorded in the report.
- [ ] Build green: `cmake --build build --target ninfer-serve` (serve binary only; test targets per
      disk rule — rebuild `ninfer_tp2_decode_test` only, one binary).
- [ ] `git status` clean of unexpected files; HEAD + dirty-hash recorded in the report header.

## Cells (in order; abort on any red, per stop conditions in docs/157_WO_gemini_testing.md)

**C1 — groupwise default, capacity gate** (fail-closed path unchanged):
serve `/home/intel/models/qwen3_8_27b.ninfer`, `--devices 0,1 --kv-dtype int8 --max-context 131072`,
NO nvfp4 env. Gates: starts with `--max-context 131072` accepted; 25k body (`/tmp/p2_body_25k.json`)
prefill = **1,035 t/s ±3%**; two runs content-sha equal.

**C2 — decode guard, kvarn_k4v2** (`ITERS=1 DG_SPECS="kvarn_k4v2|10000 160000"`):
Gates: 80.2 ±2% @10k, 56.3 ±2% @160k, MTP acceptance ≥70%.

**C3 — NVFP4 weights, env-gated** (`NINFER_ALLOW_NVFP4_TP2=1`, artifact
`~/ninfer/incoming/nvfp4/qwen3_8_27b_nvfp4.ninfer`, `--kv-dtype kvarn_k4v4 --max-context 65536`):
serve listens; 25k body prefill recorded — **this row is THE decisive number, re-measured on merged
main** (cross-check vs the in-window value; spread >5% = investigate, don't average).

**C4 — NCCL transport assertion** (`tools/bench/nccl_transport_check.sh` or the C3 log):
all channels `via SHM/direct/direct`, zero `via P2P`.

## Report

`results/157_g5_postmerge_regression.md`: C1–C4 rows, gates, HEAD, dirty-hash, any deviations.
**Merge bar:** C1+C2 green → merge stands. C3 recorded regardless of value (it is the mission answer,
not a gate). Any red → merge holds, escalate to coordinator + A1.
