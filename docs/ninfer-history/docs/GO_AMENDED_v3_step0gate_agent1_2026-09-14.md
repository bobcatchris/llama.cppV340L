# GO-AMENDED v3 — PRED-A FLIPPED GREEN at b412303e (agent1, 2026-09-14 ~10:4xZ, re-derived at the wire)

Supersedes the verdict table of `GO_AMENDED_v2_step0gate_agent1_2026-09-14.md` (v2's predicates,
recipe, and announcement-row text otherwise stand). Baseline: boot tip **b412303e** (WO-TP4-E:
factory relax + `engine_route.h` + `tp_engine_route_host` cell), pulled and merged to lane at
`2b01cfdb`. Zero device, zero builds; cell compiled to /tmp only (single header + one TU).

## Verdict table at b412303e

| # | predicate | v2 state | v3 state @ b412303e |
|---|---|---|---|
| PRED-A | factory silent-downgrade route ABSENT | FAIL (`== 2` live @ :310) | **PASS** — dispatch is `wants_tensor_parallel(asked)` (`engine.cpp:322`, predicate `>= 2` in `engine_route.h:50`), `checked_served_world` throws on any asked≠served pair; host cell ran at this seat: **PASS rc=0, 38/38 arms** |
| PRED-A′ | *(form hardened this hour — see self-catch below)* | — | `grep -nE "if[[:space:]]*\(options\.devices\.size\(\)[[:space:]]*==[[:space:]]*2\)" src/runtime/engine/engine.cpp` must print nothing → **ABSENT rc=1 = PASS** |
| PRED-B | R1 argmax transport PRESENT | FAIL | **FAIL (stands, correctly)** — `require_argmax_transport` still hard-throws (`argmax_routing.h:39-49`), `tp_group.cpp:341-344` names R1 as pending WO-TP4-B; dispatched 10:2xZ per chair. Window stays **HOLD on B alone** |

**Self-catch, same family as the finding (mention-vs-call, #560 law):** v2's PRED-A form —
`grep "devices.size() == 2" <file>` — **still printed a hit at b412303e: a COMMENT** (`:311`,
the relax's own provenance note). The naive form reads FAIL on a green tree — an instrument that
disagrees with the chip it guards is no gate. Hardened to the code-context form PRED-A′ (or,
decision-shaped and better: run the cell, or grep `wants_tensor_parallel` PRESENT at the factory).
The comment is NOT re-litigated as "not merged" — the cell + the code line are the witnesses.

**RED-capture re-verified independently at this seat (closure law, pre-fix direction):** compiled
the cell against a scratch `engine_route.h` with `>= 2` restored to `== 2` →
`FAIL: tp_engine_route_host (22)` — 22 armed failures, including `asked=4 served=1` betrayal arms.
So: RED on pre-fix shape, GREEN on post-fix tip, both measured here, both directions — the cell
joins the standing suite per agent4's commit; this row cites it, does not clone it.

## Disk audit (chair's ask: does step-0 gain a disk predicate, named owner?)

**Measured 10:3xZ:** `df /` = **8.8 G free (93%)**. Attribution of the dark-hours ~2 G:
- Lane build trees are NOT the hole: all `build-hip-amd`/`build-hip` across all 17 worktrees +
  shared checkout sum to ~1.1 G (biggest: p3-serve 394 M, shim-funcattr 393 M).
- The hole is **/tmp = 14 G**: `agent5_probe` 838 M, `wt_now`/`a5_audit`/`wt_mine`/`wt_m2`/`wt_t3`/
  `wt_main`/`mx` ≈ 745 M each (clone-per-scratch-tree pattern), `tmp.txLkvdOsQM` 447 M, plus
  two 120 M `.o` files; and `/home/chris/artifacts_bin` 1.8 G — **which is not waste, it's the
  boot-stamp bank (BANK-BEFORE-RELINK law); anything that prunes it is booting un-pinnable bins.**
- No `git worktree prune` candidates found among the /tmp trees that are still mounted
  (`/tmp/cx` etc. appear in `git worktree list` — pruning must be chair-routed, desks may have
  live runs; I touched nothing).

**RECOMMENDATION (my desk's ruling-as-advice): YES, add PRED-C to step-0, owner = the desk about
to boot, enforcer = allocator-of-grants (chair) at grant time:**
`PRED-C: df / free ≥ 2 G at spawn` — a measured live number, one predicate, no estimate term.
Rationale: a full /tmp during a run kills the trace/log banking mid-boot (request-log-jsonl,
G18*.log capture, core dumps) — a failed window leg with lost artifacts is worse than a refused
spawn, and unlike a VRAM charge this one has NEVER produced a false refusal (8.8 G live vs 2 G
floor = 4.4× headroom). VRAM law's scope is launches; a spawn-time disk witness is a permission-
adjacent fact, same class as KFD-0. Cleanup itself stays owner-scoped per review-law F (a desk
prunes its OWN /tmp scratch + its lane's stale build trees at leg end; never glob another desk's
paths — the `mv`-glob specimen says why).

## Announcement-row text — UPDATED, paste-ready on the R1 tip (v2 block amended, disk clause added)

> **step-0 SOURCE GATE (agent1, v3 @ R1-delivery):** at the BOOT TIP, before spawn:
> (A) factory honest — code-context grep `if (options.devices.size() == 2)` ABSENT in
> `src/runtime/engine/engine.cpp` AND `ctest -R tp_engine_route_host` rc=0;
> (B) R1 argmax transport PRESENT (WO-TP4-B) — else the row names the first-sample R2 throw as
> PRE-DECLARED expected refusal (loud, named, zero-token boot = CAPABILITY datum under P3);
> (C) disk: `df /` free ≥ 2 G printed in the manifest row (spawn-time witness; per-desk /tmp +
> stale-build cleanup at leg end, owner-scoped, bank `artifacts_bin/` exempt);
> bin = bank-stamped `<sha16>` filename, sha re-verified at pre-flight, tip-grep is provenance
> (ONE-BUILD); `compute capability 12.0` at ≥2 devices = step-0 FAIL by definition (factory
> resurrection — ghost-pointer family, factory member; forensics 1363db42).

— agent1, runbook/gates desk · seat worktrees/amd-wo-agent1-support · zero src edits, zero device, zero builds
