# 112 — Agent 1: docs/105 GPU window — START NOW (immediate handoff)

**Status:** GPUs are FREE (official baseline finished 10:12, committed as
baseline v1). You are the GPU agent. Execute docs/105 steps 0-3 in order,
back-to-back, no stopping between steps. Report after each step.

All paths relative to `~/ninfer/worktrees/wo-kvarn-hold` (HEAD = `9a31a070`,
your wip `a73fe948` is in-tree). Your harness: `results/105_prefix_battery/`
(MANIFEST.md is the canonical usage — follow it).

## Step 0 — pre-fix capture (~45 min)

Ground truth first. `build/apps/ninner-serve` is the PRE-FIX binary (reverted
after the 09:46 relink incident). Verify before anything else:

```
sha256sum build/apps/ninner-serve ~/ninfer/bin/prefix_ninner_serve_99111d52
# both must be e72baabd605d8a19... — if not, STOP and ping
```

Then run the full pre-fix battery per MANIFEST:
`run_battery.sh MODE=prefix` → `results/105_prefix_battery/capture/`
(robot/sunsets k=0 vs k=3, H5 k=1/2/3, 6-prompt battery k=2, temp>0 seeds 1-6).
Every capture run must complete cleanly — if one fails, fix + rerun before
proceeding (capture is the byte-diff ground truth; a bad capture poisons the
whole A/B).

## Step 1 — wip A/B byte-identity (~20-30 min)

Incremental rebuild of the wip (4-stage) binary — only 2 files changed:

```
cmake --build build --target ninfer-serve -j16
# verify: sha256 of build/apps/ninner-serve != e72baabd...
```

Then `run_battery.sh MODE=wip` → `results/105_prefix_battery/wip/`, and byte-
diff capture/ vs wip/ (`tools/bench/byte_diff.py` or per MANIFEST).

**Any byte diff = STOP.** Save all outputs, ping immediately (D-21 class).
Also run the deferred GPU tests now: `/usr/bin/ctest -R "gdn|kvarn|spec"`.

## Step 2 — perf gate (~30 min)

Fast CI via `parse_perf_gate.py` (per MANIFEST). Gates:
- plain (no MTP) ≥ 35.0 t/s
- verify ≤ 35.6 ms
- mtp ≥ 79.0 t/s
Reference (pre-fix CI): 32.88 / 37.67 / 75.38. Report all three + deltas.

## Step 3 — full CI + closeout (~45 min)

Full CI per MANIFEST. On PASS:
1. Reword the wip commit (`git rebase -i HEAD~2`): `wip(gdn): ...` → proper
   subject per docs/105 §commit policy.
2. Write `results/105_gdn_t1_perf_fix_closeout.md`: byte-identity A/B verdict,
   perf-gate numbers + deltas, full CI verdict, pre/post binary hashes.
3. Commit.

**Do NOT run the decode guard** (post-fix kvarn 40k/250k spot check) — that is
the plan owner's, runs immediately after your step 3 commit.

## Timing & queue

ETA: ~2h → done ~13:00. GPUs are yours until then. After you land:
- 13:00-13:15 plan owner post-fix guard spot check
- ~13:30 Agent 2 (wo/kv-uniform) GPU window
- then mtp-adaptive (docs/107), then magic-dict (docs/108)

If blocked >15 min on anything, ping immediately — do not sit on the GPU.
