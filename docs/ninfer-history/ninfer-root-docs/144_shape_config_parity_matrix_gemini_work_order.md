# 144 — Shape × Config Parity Matrix, ALL DTYPES (gemini) — close the coverage blind spots

> **Type:** GPU + scripts — test-lane matrix runner + in-process replay corpus, across
> **kvarn / int8 / bf16**. **No src/ feature code** (engine-side hooks go through the
> coordinator to agent1).
> **Base:** `wo/2a-batched-serving` @ `f09d8b8e` (the frontier/batch diagnostics —
> `NINFER_NO_FRONTIER`, `NINFER_BATCH_DISABLE`, batched serve — exist ONLY on this branch).
> Create worktree `wo-shape-parity` off that tip; rebase onto the final 2a tip before merge.
> **Owner:** gemini. **Est:** ~1.5 d (scripting CPU-first; GPU ~2 h total across 2–3 windows
> of ≤60 min, AFTER agent1's current 2a window — coordinate via coordinator).
> **WBS:** new row (coverage hardening). Source findings: docs/143 §4.4 + §4.5 (agent1).

## 0. MISSION

Today's gates compare **each config against itself** — so config-axis bugs are invisible by
construction. Two proved examples from 2026-09-03:

- **docs/143 §4.4 (single-seq!):** the TurnClosure rewrite frontier is **NOT output-neutral** —
  frontier ON (default, present on every chat completion) vs OFF flips the greedy argmax from
  the first decoded token. Every existing gate runs frontier-ON vs frontier-ON.
- **docs/143 §4.5 (batched):** batched-vs-single-seq diverges **only in thinking-mode request
  shape** (5/6 prompts), serve-only — the in-process harness never enables thinking, and the
  decode guard uses raw prompts.

Build the **shape × config parity matrix** that covers every axis combination **on all three
dtypes**, encode the parity invariants explicitly, and mark known violations as tracked xfails
so fixes flip them and new regressions cannot hide. The §4.4/§4.5 mechanisms are NOT
kvarn-specific: the TurnClosure frontier and the chat/thinking request shape apply to every
lane (GDN prefix checkpoints exist for i8/bf16 too — see their serve logs), and no existing
gate A/Bs them on any dtype.

## 1. THE AXES

| axis | values | mechanism |
|---|---|---|
| **dtype** | **kvarn_k4v2 / int8 / bf16** | `--kv-dtype` / `--all-cache-type` per server start |
| shape | raw prompt / chat-templated | serve `/v1/chat/completions` vs raw `prompt` field |
| thinking | ON / OFF | server `--no-thinking` (request `thinking=off` also visible in serve logs) |
| frontier | ON / OFF | `NINFER_NO_FRONTIER=1` at server start |
| batch | single-seq / batched (2-lane self-pair) | `NINFER_BATCH_DISABLE=1` vs concurrent identical pair — **kvarn ONLY** (i8/bf16 have no batched runner; batch axis N/A there until the parked runner lands — revisit trigger) |
| sampler | greedy (parity) / sampling (stability subset only) | temp 0 vs temp 0.8 + seed |

**Context: keep ≤10k for all dtypes.** This matrix tests config-axis neutrality, not long-ctx
perf; bf16 additionally crashes at ≥40k on this 36-SM box (docs/124) — do not probe that here.

**Prompt set:** agent1's 6 discriminators (docs/143 §4.5): ice-floats, lighthouse, water-cycle,
rainfall, three-Moon-facts, reverse-a-string. (ice-floats is the known CLEAN canary; the other
5 are known thinking-mode DIVERGE canaries — exactly what a matrix needs.)

**Core matrix:** greedy only.
- kvarn: 2×2×2×2 = 16 configs × 6 prompts = **96 requests**
- int8 + bf16: batch axis collapses to single → 2×2×2 = 8 configs × 6 prompts = **48 each**
- total **192 requests** + sampling stability subset (6/dtype = 18) + 2× repeat for determinism.
Est. GPU ~2 h incl. per-dtype server starts (group by server-start config to minimize restarts).

## 2. THE INVARIANTS (assert per prompt)

| id | invariant | expected state |
|---|---|---|
| V1 | batch parity, clean shape: single == batched-selfpair, frontier-OFF, no-thinking | kvarn: **PASS** (agent1 verified); i8/bf16: **N/A** (no batched runner) |
| V2 | batch parity, thinking shape: single == batched-selfpair, frontier-OFF, thinking ON | kvarn: **XFAIL — docs/143 §4.5 open defect**; i8/bf16: **N/A** |
| V3 | config neutrality: frontier-ON == frontier-OFF, single-seq, same shape | kvarn: **XFAIL — docs/143 §4.4 open defect**; i8/bf16: **establish** — if frontier is a no-op for a dtype, V3 passes trivially and that fact is itself a recorded finding |
| V4 | determinism: same config ×2 → identical | **PASS, all dtypes** |
| V5 | lane independence: self-pair lane A == lane B | kvarn: **PASS**; i8/bf16: **N/A** |
| V6 | shape canonicalization: chat-templated token-ids replayed as raw prompt == chat-completion output | **establish, all dtypes** (this is what makes the corpus valid) |

Comparison = sha256 over **content + reasoning token stream** (thinking output included —
that's where §4.5 lives). Verdict JSON per cell: config tuple, prompt id, sha, pass/xfail/fail.
**A cell that fails an expected-PASS invariant, or an xfail that unexpectedly PASSES (fix
detector), both go red.**

## 3. THE REPLAY CORPUS (makes this CI-cheap)

Serve runs are the only place thinking/chat shapes exist today — too slow for the fast gate.
So:

1. During the matrix run, **dump the exact chat-templated prompt token-ids** the engine builds
   for each (shape, thinking) cell. If no existing diagnostic prints them (check
   `NINFER_BATCH_DBG` / `NINFER_MB_*` first), FILE THE REQUEST with the coordinator → agent1
   owns engine-side hooks; do NOT add one yourself.
2. Commit the corpus to `tests/multi_gpu/data/shape_parity_corpus/` (token-ids + config tuple +
   expected output sha per cell).
3. Add an **in-process harness cell** (`ninfer_tp2_batched_decode_test --prompt-a/-b` replay of
   the corpus) covering V1/V2/V4/V5/V6 at harness speed — this is what wires into the fast gate.
   The serve matrix itself goes into `run_ci.sh --full` only (docs/99 CI scoping).
   **Known hazard:** the in-process harness had a documented segfault at `--kv-dtype i8|bf16`
   (pre-existing, A1's list). If it still segfaults, i8/bf16 corpus replay is BLOCKED — report
   to coordinator, do not fix the harness yourself; the serve matrix for i8/bf16 still runs.

## 4. STEPS
1. **Read docs/143 §4.4/§4.5/§5/§6 + agent1's probe scripts** (`tools/smoke/diag/`:
   `probe.sh`, `nt.sh`, `frontier.sh`, `det.sh`, `repro2a.sh`) — reuse their launch patterns;
   agent1 already solved the hard invocation problems.
2. Write `tools/parity/shape_config_parity.sh`: matrix runner, dtype × config-grouped server
   starts, emits `results/shape_parity_<ts>.json` + human verdict table. CPU-only authoring.
3. GPU windows (coordinator-authorized, after agent1): run the 192-cell core matrix + stability
   subsets + repeats, dtype by dtype (kvarn first — it has the known defects to pin). Capture
   prompt-token-ids per §3.1.
4. Build corpus + in-process replay cell; verify V1–V6 states match §2's expected column.
5. Write `results/shape_parity_README.md`: invariant table, xfail registry (with defect refs),
   reproduce commands.
6. Propose CI wiring (fast = corpus replay; full = serve matrix) as a docs/144 §7 note —
   actual `run_ci.sh` edit happens at closeout, not here.

## 5. CONSTRAINTS
- Test lane: scripts + harness + corpus + results. **Zero src/ edits.** Engine hook needed →
  ask coordinator → agent1.
- Branch is agent1's live worktree — work in your OWN worktree off `f09d8b8e`; never edit
  `wo-2a-batched-serving` in place; rebase onto final 2a tip when agent1 lands.
- GPU: agent1 holds priority (2a critical path). No GPU until coordinator grants a window.
- Do not "fix" §4.4/§4.5 — they are tracked defects; your job is the net that catches them and
  everything like them.
- Keep provenance per run (tree, binary sha, server flags, env).

## 6. DONE-ON-DELIVERY (DoD)
- [ ] `tools/parity/shape_config_parity.sh` — full matrix runner (3 dtypes), committed.
- [ ] 192-cell greedy matrix + stability subsets RUN (kvarn 96 / int8 48 / bf16 48), results
      JSON committed with provenance.
- [ ] V1–V6 verdicts match expected states per dtype (V2/V3 kvarn xfail-registered; V3 i8/bf16
      established either way; V1/V4/V5 pass where applicable; V6 established or filed).
- [ ] Prompt-token-id corpus committed (`tests/multi_gpu/data/shape_parity_corpus/`).
- [ ] In-process replay cell green at harness speed.
- [ ] `results/shape_parity_README.md` with invariant table + xfail registry + repro cmds.
- [ ] CI wiring proposal noted (§7 of your results doc); no run_ci.sh edit.
- [ ] GPU windows coordinated (agent1 first).

## 7. NOTES
- If the matrix finds a NEW divergence outside the two known defects, stop and report to
  coordinator immediately — that is higher priority than completing the sweep.
- §4.4 consequence for existing CI: B0 batched CI cell must run `NINFER_NO_FRONTIER=1`
  (agent1 owns that fix; do not touch their cell).
