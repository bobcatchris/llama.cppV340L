# Verification handoff — gate is now `run_ci.sh --full`

The four post-decay-fix verification gaps from the original 126 runbook were folded
into CI. There is no separate manual handoff runbook anymore; run one command.

## Run it

```bash
cd /home/intel/ninfer/worktrees/wo-kv-uniform-ci-gate
bash tools/ops/run_ci.sh --full --no-build     # gate everything against the existing build
```

- `--full` runs the whole battery (int8 full, KVarN @250k, KVarN T14 zone,
  `regress_unified`, decode-guard 25k/80k cells) — plus the always-on steps.
- `--no-build` is safe RIGHT NOW: the build dir is verified-current (only a
  comment-only commit since the last build) and the main agent is about to
  rebuild it, so `run_ci.sh` must not touch `build/`.
- `run_ci.sh` owns the whole server lifecycle: it stops the live 8091 dev server
  at the start and restores it on exit (even on failure). No manual coordination.

## What the gap actually was → where it lives now

| Old runbook tier | What it verified | Now covered by |
|---|---|---|
| Tier 1       | KVarN unit battery + kernel bench sanity | `run_ci.sh` step [3a] (per-build, always runs; T=4@160256 gate ≤1.5 ms) |
| Tier 2a      | `regress_unified.sh` (token identity vs bf16 anchor) | `run_ci.sh --full` |
| Tier 2b      | decode guard cells, full matrix 10k–250k (greedy, 1 iter/ctx) | `run_ci.sh --full` → `tools/bench/decode_guard_cells_ci.sh` |
| Tier 3       | byte-diff A/B vs old commit (one-shot claim) | NOT in CI — one-shot, see below |

## Tier 3 (one-shot, optional — verifies a past commit claim)

Not a standing gate. Only needed if you want to confirm `ac6b3f61` is
byte-identical to `c631f7f1`; requires building the old commit in a separate
worktree and swapping servers. Keep the procedure from the ORIGINAL docs/126
(`/home/intel/ninfer/worktrees/wo-kv-uniform/docs/126_verification_runbook.md` §Tier 3).

## Pass criteria (CI verdict)

- Exit 0 = PASS (`CI: PASS ✓`) from `run_ci.sh --full`.
- Unit battery (step 3a): all 14 tests PASS, SELFTEST64 nonzero acc, T=4 ≤ 1.5 ms.
- `regress_unified.sh`: `RESULT: PASS` with **no SKIP** (anchor is mandatory).
- Decode guard cells: `GATE: PASS` — each of 10k/25k/40k/80k/160k/250k within the baseline ratchet
  (>5% t/s drop = FAIL; >2% or ±5pp accept = WARN/tracked).
