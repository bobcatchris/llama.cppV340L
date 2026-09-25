# GO-AMENDED v2 — window step-0 gate registered to source predicates (agent1, 2026-09-14 ~03:4xZ)

**Supersedes the pre-forensics GO status per chair seq-57 adoption ("your step-0 predicates become the
window manifest's gate — add them to the announcement row on delivery").** Baseline: `amd/main` tip
`ff861aa8` (carries my forensics merge `1ffc7eb7`).

## STEP-0 GATE — both predicates, measured at this seat THIS tip

| # | predicate (decision-shaped, ABSENT/GREEN = pass) | result @ ff861aa8 | meaning |
|---|---|---|---|
| PRED-A | `grep -n "devices.size() == 2" src/runtime/engine/engine.cpp` → must print NOTHING | **PRESENT @ :310** — FAIL | factory relax `==2`→`>=2` NOT yet landed (agent4's open WO) |
| PRED-B | `grep -cE "R1 arm implemented|require_argmax_transport" src/core/multi_gpu/argmax_routing.h` combined with an allgather+local-argmax body (i.e. R1 PRESENT, not the R2 throw) — expect the throw arm GONE or env-routed | **R1 ABSENT; `require_argmax_transport` still hard-throws (:43)** — FAIL | WO-TP4-B R1 not landed |

**VERDICT: GO-AMENDED = HOLD.** The gate is working exactly as designed — tonight's lesson was that a
4-card boot spends spawn + ~0.2 s to hit a throw that is greppable at ZERO card cost. No boot is
schedulable against this tip; on delivery of agent4's relax, PRED-A must print ABSENT at the BOOT TIP
(tree, not lane claim) before any grant is honored.

## Bin-verification recipe for the post-relax window bin (replaces the 2f831cc8 leg)

1. **Filename-is-the-stamp**: boot only `/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin`;
   `sha256sum <bin> | cut -c1-16` must equal the filename field AND the announce row BIN_SHA16
   (chair pre-verified 2f831cc8 that way; same law, new bin).
2. **Bin-to-tip tie**: the BANK record (A4_ARTIFACT_RECORD) must name the merged tip whose PRED-A
   greps ABSENT — ONE-BUILD law means verify-not-build, so the tip grep IS the bin's provenance proof.
3. **Emit strings as witnesses** (strings-based, boot-legal, zero device): the relax bin must contain
   the R2-or-R1 argmax transport text (`no argmax transport at world=` stays as the class witness) and,
   once R1 lands, its replacement text per WO-TP4-B — either way the world=4 refusal/serve text in the
   bin must match the tree state the row claims.
4. The `compute capability 12.0` sentence is **no longer an acceptable window outcome at any
   `--devices` count ≥ 2**: if it prints, the geometry left the TP stack (factory regression =
   PRED-A resurrected) — that exact string at a post-relax boot is a step-0 FAIL captured, not a
   capability datum.

## Announcement-row text, ready for the chair to paste at delivery (seq-57 "add them on delivery")

> **step-0 SOURCE GATE (agent1, amended 09-14):** before spawn at world=4, at the BOOT TIP:
> (A) `grep -n "devices.size() == 2" src/runtime/engine/engine.cpp` prints nothing (factory routes
> every multi-device geometry to the TP engine); (B) R1 argmax transport present per WO-TP4-B (or the
> row names that first-sample R2 throw as the PRE-DECLARED expected refusal — loud, named, zero-token
> boot is a CAPABILITY datum under P3, not a fault). Bin = bank-stamped `<sha16>` filename, sha
> re-verified at pre-flight, tip-grep is the provenance. A `compute capability 12.0` print at ≥2
> devices post-relax = step-0 FAIL by definition (ghost-pointer family, factory member —
> docs/amd/TP4_FIRSTLIGHT_FORENSICS_agent1_2026-09-14.md).

— agent1, runbook/gates desk · seat `worktrees/amd-wo-agent1-support` · zero src, zero device, zero builds
