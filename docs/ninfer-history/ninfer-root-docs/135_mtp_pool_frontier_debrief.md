# 135 — Debrief: the T128_uneven_plen single-seq MTP-pool abort (and the CI-gate session that pinned it)

Status: DEBRIEF (2026-09-02, agent-1 ci-gate session). Companion docs: 127 (gate status),
128 (phase-gate pipeline), 130 (multibatch status, §9-§10), 131 (test suite), 132 (roadmap
§11). Branch: `wo/kv-uniform-ci-gate` @ 024189af (all referenced commits pushed).

> **CORRECTION (2026-09-03, GPU-confirmed, `wo/kv-uniform` @ `6857c7ea`) — the abort described
> here is NOT a single-seq MTP-pool-frontier bug. It was the **pre-fix BATCHED `any_draft`
> starvation abort**, mislabeled as a "reference run failed".** The batched phase runs FIRST in a
> `NINFER_SEQ_ONLY` invocation (§5 below), and an abort there is mislabeled as a reference failure.
> On the merged state (post batched-fix `0257878c`) the exact cell `--tokens 128 --mtp 3`
> completes **RC=0**: `[batched] 2 lanes finished` + `[SEQ-ONLY] dumped single-seq MTP + plain
> for both lanes`, 4 seq dumps at 128 tokens each, and the R5 trace is clean — it reaches F=125
> (the `pos=125 slot=61` coordinate) at r036, then F=128/r037 … and completes, no hole. The repo's
> own `tools/ops/pool_frontier_replay` CPU tool also reports the round pattern is
> SELF-CONSISTENT (no lag). See docs/137 (§4–§9). **The "single-seq pool-frontier divergence"
> in §1 is therefore a misattribution; no single-seq fix is needed.**

## 1. The problem, as observed during the pre-fix session

> **Observed on the pre-fix ci-gate tree (`024189af`).** This is the abort that was at first
> mislabeled as a *single-sequence* reference failure; it is actually the batched-phase
> `any_draft`-starvation abort (see the CORRECTION above). The cell and trace are reproduced
> verbatim:

`--tokens 128 --mtp 3` with two prompts of very different lengths ("Explain general
relativity…" ≈14 tok, "Hello!" ≈2 tok) aborts during the reference (`NINFER_SEQ_ONLY`)
invocation (~step 55) with:

```
std::logic_error: gqa_kv_append_kvarn: append left a hole in the tile
  (pool=mtp pos=125 page=1 slot=61 tail=59 tile_page=1 committed=1)
```

> ⚠️ **The mechanism below was the pre-fix hypothesis.** It is now known to be a misattribution:
> the pool advance in the *single-seq* path is consistent (no lag — `pool_frontier_replay`
> self-consistent, R5 clean on the merged binary). The real injector was the pre-fix batched
> `any_draft` gate. Read the rest strictly as the historical account of that abort.

The MTP-layer KV pool (the draft model's cache) advances by the **verified span** per round
(ext-gated: observed +ext/round via REW-DBG pre-tails 55/59), while the sequence frontier
`cur_F` advances by the **accepted** count (`a+1`) per round. `rewind_tail` never extends
`tail_count` forward (early-return when `keep >= tail_count` — correctly refusing to
fabricate). After partial-accept rounds the pool frontier therefore **lags cur_F** by the
accumulated `(a+1 − ext)`, and the next append at a true position ≥ cur_F trips the
sequential-tail invariant (`slot > tail_count` → throw).

## 2. Why it matters / user impact (the honest answer)

Three distinct exposure regimes, in order of severity:

1. **Hard crash (this bug, at threshold).** The throw is uncaught in the worker thread →
   `std::terminate` → SIGABRT → **the whole serving process dies**. All in-flight requests
   drop (connection reset). This is loud, not silent: users get a failed request, never bad
   data from the crash itself. Trigger condition: any request stream whose acceptance
   pattern produces partial-accept rounds (`a+1 > ext`) accumulating ~2+ tokens of deficit
   with the pool near a page boundary — uneven prompts and long generations raise the odds.
2. **Silent slowdown (below threshold).** While the pool frontier lags `cur_F` without
   crossing the invariant, the draft model attends over an incomplete context → its
   proposals degrade → **acceptance rate collapses toward ~1 token/round** → generation
   slows dramatically. Output tokens remain *correct* (committed tokens are target-model
   argmaxes; the target context lives in the TEXT KV + GDN state, which are unaffected).
   Users would experience "the server got slow", not wrong answers.
3. **Wrong data (separate, adjacent bug — still open).** The §5.2/§6.3 class: the shipped
   single-seq MTP path's repeat-padded append overwrites the last-valid slot with a padded
   column's KV (deterministic, prompt-dependent — the lane0@79 case). That CAN produce
   silently wrong continuations. It is a different mechanism from this debrief's bug and is
   tracked as the §6.6 seeding/pool-semantics item.

**Net for the end user:** they do not get bad data from THIS bug — they get a crashed server
(at threshold) or a slow server (below threshold). The wrong-data risk lives in the adjacent
§5.2/§6.3 class, which remains open.

## 3. Everything tried, with outcomes

| # | Attempt | Result |
|---|---------|--------|
| 1 | a13ae028: true positions (base+j) for every column — fixes the T=256 same-slot scatter race | Fixed the batched flake; **regressed** the MTP-layer append contract (idempotent-overwrite lost) → the T128_uneven hole. Root of the current red. |
| 2 | INV-7 INVASERT bound [0,152064) → draft-vocab membership | Correct fix; the old bound was a fossilized false positive (248046 ∈ draft vocab). Landed. |
| 3 | Frontier clamp, batched MTP append (clamp p > frontier → frontier) | **Worse**: collapses the verify's legitimate 4-column extension to one slot → MTP context degrades → new token divergences (T128_both_full rc=1). Reverted. |
| 4 | Frontier clamp, single-seq MTP append | Prevents the throw in isolation, but masks the real question (what should the pool hold?) and does not fix the batched-side absence; kept out of the final state pending the semantics ruling. |
| 5 | Kernel revert to repeat-padding (restore MTP idempotency) | Fixes the hole; **breaks** the TEXT anti-race fix and the seq-vs-batched parity (the seq stomp returns). Reverted — the wrong trade. |
| 6 | INV-3 window assert (window == committed*64+tail+T at the pre-verify binding) | Works; gives NINFER_CANARY_TAIL_COUNT a real trip signal. Landed. |
| 7 | canary_unrank_m0 injection hook + shrink-forcing env | Works; 3/3 canaries DETECTED. Landed. |
| 8 | T3 direct-binary T0 fallback (bypass the broken pip-ctest shim) | Works; T3 exit 0. Landed. |
| 9 | Harness GPU-settle (60s poll) + T1.2 load-OOM retry-once | Works; transient OOM class eliminated. Landed. |
| 10 | Probes: MTPAPPEND (per-append position dump), REW-DBG (rewind pre-state), ALIGN-DBG, MTPDISPATCH (branch decision), HOLE-DBG (detailed throw) | The entire pin came from these. Removed from the committed tree (env-gated versions live in git history); the detailed hole-throw message is kept. |

## 4. Ruled out / disproven (do not re-chase)

- **a+1-vs-ext one-line keying** (my first candidate): disproven — the verify's MTP append
  already writes all 4 true positions (MTPAPPEND width=4 probe); the deficit enters
  elsewhere. The ext-gate's exact code location is STILL UNIDENTIFIED.
- **INV-7 fossil as a product bug**: 248046 is a legitimate draft (draft-vocab member) — the
  bound was wrong, the drafts were fine.
- **Batched-only**: the reference (single-seq) aborts identically — this is a
  single-seq-path bug; the batched path's exposure is unvalidated for this cell.
- **Dispatch skip**: MTPDISPATCH shows the MTP-layer attend always routes to the batched
  branch in the failing run — no skipped dispatch.
- **Hydrate rewrite** (docs/130 §8.4): dead — steady decode never hydrates.
- **Same-slot scatter race** (the T=256 flake): real, but FIXED (a13ae028); unrelated to
  this remaining red except through the shared position tensor.

## 5. Things to know (contracts, layouts, gotchas)

- **Position tensor dual-contract**: `speculative_prepare_verify_inputs` positions feed BOTH
  the TEXT KVarN scatter (needs TRUE base+j — same-slot race otherwise) and the MTP-layer
  appends + RoPE (the REPEAT padding base+min(j,ext) is the MTP append's idempotency
  mechanism). A single tensor cannot serve both — hence the open design question.
- **Tile layouts**: K tile {kG,kD,kvh} is ne[0]=kG-contiguous: elem (g,d,h) @ bytes
  (g + d*64 + h*16384)*2. V tile {kD,kG,kvh}: elem (d,g,h) @ (d + g*256 + h*16384)*2.
  Per-lane slice = 64 KiB; lane b at b*64 KiB. (§8.3's "1024 B slots" were kD-channel
  groups — a decoded-contract trap.)
- **Pool invariants**: sequential appends never hole (tail advances with slot); the hole
  requires a position jump past the tail — i.e., a frontier/bookkeeping desync, never a
  "random" corruption.
- **The MTP pool legitimately lags cur_F** under the current ext-gated advance — the
  rewind's no-extend early-return makes the lag permanent. Any fix must decide which of the
  two is authoritative.
- **Draft-only blast radius**: MTP-pool corruption degrades proposals (perf), never committed
  token identity — UNLESS the corruption reaches the TEXT KV (the §5.2 stomp class, which
  does).
- **Env probes** (all env-gated, re-add from git history if needed): MTPAPPEND, REW-DBG,
  ALIGN-DBG, MTPDISPATCH, HOLE-DBG (kept, in the throw message), MTPCLAMP (removed with the
  clamp).
- **The harness gotchas**: NINFER_SEQ_ONLY invocations still run the batched phase first
  (line 152) — an abort there is mislabeled "reference run failed"; GPU driver memory
  release lags process exit (settling required between steps); the pip `ctest` shim in this
  env is broken (use /usr/bin/ctest or direct binaries).

## 6. Solution paths (ranked)

- **E (ruling first, recommended)**: the plan owner states the intended MTP-pool semantics
  (does the pool advance by ext or a+1? must the alignment append extend the pool?). The
  probes make either answer implementable in ≤1h. Without the ruling, every fix is a guess
  that trades one failure mode for another (attempts 3/5 are the evidence).
- **A (span keying on a+1)**: the coordinator-approved candidate — now known to require
  LOCATING the ext-gate first (attempt 1's lesson). Pair with E.
- **B (alignment width cut to licensed)**: slice the alignment forward to the accepted span
  so masked columns never append. Correct but touches the op call shape; the accepted span
  is ≤ cur_F-1 so no hole ✓; the draft-hidden selection (column a) is within the cut ✓.
- **C (rewind-extend for the MTP pool only)**: let rewind_tail extend the MTP tail to cur_F,
  accepting stale bytes in [pool, cur_F) until the next verify rewrites them — draft-only
  impact, cheapest, but permanently hides real shortfalls (weakens INV-3's spirit).
- **D (decouple mtp_frontier from cur_F)**: honest bookkeeping — track the pool frontier
  explicitly; rewind truncates to min(cur_F, pool_frontier); the alignment appends from the
  pool frontier. Moderate diff across text_context_impl + the runner.
- **Regardless of path**: fix the single-seq §5.2/§6.3 stomp (the wrong-data class) — same
  subsystem, higher user impact (silent wrong continuations).

## 7. Associated records

- Commits (ci-gate): 95b98f72 (merge), 96a1d461 (docs/99), 51e00a7b (126 handoff preserved),
  4aafa044 (CI wiring + suite fixes), 53122cc3 (slice4 callsite signatures), a49ae842
  (INV-7/INV-3/canary/T3/harness), 024189af (§9.8 notes). Branch wo/kv-uniform: a13ae028
  (flake fix), cc8c17cb + fd9c2693 (docs §9/§10), f7da3601 + a0687f04 (roadmap/WO).
- Artifacts: results/20260902_125859_ci.json (the FAIL verdict), results/*mtp_t1/t2/t3.log,
  results/ci_cells_decode_20260902_112923.json (guard PASS), results/ci_sampling_decode_*.json.
- Captures: /tmp/rkT (3-run attend-input captures), /tmp/kv1..kv3 (kin/vin captures),
  analysis scripts /tmp/rkT/{analyze,elem_diff,decisive,elem_diff2}.py.
- Probe envs (committed unless noted): HOLE-DBG detail (in the throw), MTPAPPEND/REW-DBG/
  ALIGN-DBG/MTPDISPATCH (removed — git history), NINFER_MB_CAPTURE kin/vin dumps (committed).
