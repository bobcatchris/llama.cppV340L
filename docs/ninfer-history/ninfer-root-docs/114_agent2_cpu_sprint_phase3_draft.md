# 114 — Agent 2: CPU sprint — rebase + Phase 3 work-order draft

**Status:** DONE (plan-owner takeover 2026-08-29 ~12:10, Agent 2 away).
- Rebase: CLEAN onto wo/kvarn-hold head 5bfd7071 (zero conflicts; the expected
  tp2_backend.cpp profile did not materialize).
- Build: green (`cmake --build build -j16`, all targets incl. ninfer-serve).
- Host-side test: `./build/tests/ninfer_sku_launch_config_test` rc=0 (silent on
  success by design; live device check ran — read-only query).
- docs/113 drafted (commit 6a9c1581) — Phase 3 prefill fix work order, plan-
  owner self-review pending.

**Original status:** CURRENT — Agent 2 (wo/kv-uniform), CPU-only, starts now.
**Mission:** (a) get wo/kv-uniform ready for the ~13:30 GPU window, (b) draft
the Phase 3 work order (prefill O(n²) materialize fix) while waiting. Your
step-2 audits (60ed2911 prefill materialize, 2cff1d42 prefix reuse) are
landed — plan owner review pending, do not rework them.

## 1. Rebase + CPU build (~30 min)

```
cd ~/ninfer/worktrees/wo-kv-uniform
git rebase 9a31a070        # wo/kvarn-hold head (includes baseline v1 + wip GDN)
cmake --build build -j16   # CPU compile only
```

This verifies your dispatch table + Agent 1's wip GDN kernel changes compile
together — cheap insurance before the merge.

**CPU tests only — NO full ctest** (the suite needs the GPU, reserved for
Agent 1 until ~13:00):

```
./build/tests/ninfer_sku_launch_config_test   # your host-side test (rc=0 silent on success)
```

Any other test = wait for your GPU window.

## 2. Phase 3 work-order draft (~1-2h, CPU analysis)

Deliverable: `docs/113_phase3_prefill_fix_work_order_draft.md` — commit to
your branch (docs-only commit `docs(106): ...`). Plan owner reviews before it
becomes a real work order.

Inputs (all in-tree):
- Your audit: `results/106_prefill_materialize_audit.md` (60ed2911) — the
  O(n²) materialize pass, code sites, cost model.
- Baseline v1 §3: `~/ninfer/worktrees/wo-kvarn-hold/results/
  official_baseline_v1_20260829.md` — kvarn prefill is −13 to −15% vs
  int8/bf16 at EVERY context (10k: 719 vs 835; 160k: 427 vs 485; 250k: 340,
  no comparison), and prefill is 87-99.7% of wall time.
- docs/104 R4 (KV materialize root cause), docs/82 (lever history).

Draft content (docs/99 template):
1. **Ranked lever list (2-4 levers):** mechanism, exact code sites, expected
   prefill t/s gain at 10k/160k/250k, byte-identity risk (prefill changes
   must not alter the KV bytes stored or the attention output — the
   byte-identity gate applies to decode output; prefill t/s is measured,
   not gated).
2. **Measurement plan:** prefill t/s per context from serve logs (direct
   measurement, baseline v1 protocol), cells kvarn/int8/bf16 ×
   10k/40k/80k/160k/250k, propose pass bars and argue them (suggestion:
   kvarn@160k within 5% of int8@160k, no context regresses >2%).
3. **Non-goals:** decode (Phase 2), MTP acceptance, KV uniformity (your
   Phase 1, in flight), D-21 gate.
4. If the audit data doesn't support an actionable fix, say so with the
   evidence — a good negative beats a speculative lever list.

## 3. Be ready for your GPU window (~13:30)

- Rebuild against the post-fix head (Agent 1's wip gets reworded before then).
- Step 3 per docs/106: A/B byte-identity (your build vs current, serving
  battery), decode guard, fast CI.
- If Agent 1 slips, your window moves — plan owner will say when it opens.

## Constraints

- No GPU before the window ping. No server launches. One commit per step.
- Docs-only commit for the draft (step 2) — code commits only in the GPU
  window.
