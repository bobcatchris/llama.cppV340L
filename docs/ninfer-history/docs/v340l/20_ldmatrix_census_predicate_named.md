# 20 — The `ldmatrix` census, with every number attached to the predicate that made it

Written 2026-09-13 after a census line was declared adopted into the handoff. **Correction to my own framing,
recorded because I drafted the opposite claim first:** I asserted in a working note that the line "exists only
in chat" — that was false. It is in `results/amd/p3/wave_retirement_plan.md:132` on `origin/amd/main`, which is
better news for the process and worse news for the defect, because the wrong figure is now in a pushed
artifact a successor will trust rather than a message that decays. Numbers here are re-derivable in one
command each; nothing depends on anyone's recollection, including mine.

## Measured, this session, twice (re-run at a fresh tip before quoting)

    git worktree add /tmp/sc origin/amd/main --detach && cd /tmp/sc
    timeout 90 python3 tools/v340l/t3_two_hop_seam_check.py

    root=.
    SCOPE-A consumer call sites (excl. mma.cuh):            140 sites / 40 files
    SCOPE-B name-occurrences incl. mma.cuh defs+guards:     152 sites / 44 roster files (ground_truth=152)
    VIOLATIONS=0
    VERDICT: CLEAN

Identical output at `origin/amd/main` and `origin/amd/t3-wip`. Observed across two successive main tips
(`ac01ad27`, then `953c6671` — main moved 163+ commits during the session, so "current" means *this run*).

## The five numbers, each with its population

| figure | population | whose | status |
|---|---|---|---|
| **152 / 44 files** | SCOPE-B: name applied to arguments, **including** `mma.cuh`'s definitions and guards | tool's ground truth; independently reproduced by me | current, roster==walk asserted |
| **140 / 40 files** | SCOPE-A: consumer call sites, **excluding** `mma.cuh` | tool | current |
| 128 | agent3's earlier fleet-roster predicate | **agent3, not agent2** | historical; not emitted by the current tool |
| 115 / 112 / 39 | `NINFER_LDM_ADDR` occurrences incl. macro-defs+comments / call-sites excl. `mma.cuh` / files | agent3 | historical, reconciled at #377 |
| 159 | *any* `ldmatrix_x` mention, the cross-check bound | mine | used only to assert `mentions ≥ call occurrences` |

The 12-site gap between 140 and 152 needs no narrative: it is `mma.cuh`'s 8 definitions plus guards, stated as
a scope difference by the tool itself. **Prefer the named predicate over the reconstructed ancestry** — the
sentence is shorter and survives the author.

## The defect, and it is in the durable text (amend requested, not applied — `results/amd/p3/` is not my boundary)

The adopted line at `:132` reads, verbatim:

> seam-check tool reads 152/44 files at BOTH origin/amd/main AND t3-wip (identical count, **differing verdicts
> 15 vs 0**)

**`15 at main` does not reproduce.** Measured twice today, at two successive main tips (`ac01ad27`, then
`953c6671` — main moved 163+ commits during the session, so "current" always means *this run*):

    origin/amd/main   953c6671  SCOPE-A 140/40 · SCOPE-B 152/44 (ground_truth=152) · VIOLATIONS=0 · VERDICT: CLEAN
    origin/amd/t3-wip 5b0a8588  identical, VIOLATIONS=0 · VERDICT: CLEAN

So the counts are identical on both trees *and the verdicts are too* — the "differing verdicts" clause is the
error, and it is worse than a stale number because it points a successor at a hole that the tool at current
main says is closed, which is exactly the wasted-window failure the VRAM-law complaints are about. Most likely
`15` was true at an earlier tip (main has moved enormously), which is why the fix is a ref-stamp on the
verdict, not a retraction of the observation: **a violation count without the sha it was taken at is silently a
claim about a tree that no longer exists.**

Second item, chat-only and therefore cheaper to fix: `128` was attributed to "agent2's audit ref" in the
broadcast line. **My audit figure was 152**, decomposed as `118 non-transposed + 34 transposed = 152` with an
independent bound of 159 total mentions; `128` was agent3's fleet-roster predicate. Mis-attributing a count to
the wrong lane is the defect with the longest travel distance, because the citation is what gets reused.

## Cite-or-not line (repaired)

> `152 sites / 44 files` = SCOPE-B name-occurrences incl. `mma.cuh` defs+guards, roster==ground-truth
> asserted, **`VIOLATIONS=0` at `origin/amd/main = <sha-at-run-time>`**; `140 / 40` = SCOPE-A consumer call
> sites excl. `mma.cuh`, same run. Historical: `128` (agent3's roster), `115/112/39` (agent3's occurrence and
> call-site populations) — none of them mine; my independent figure is `152` via
> `grep -rhoE 'ldmatrix_x[24](_t)?\(' --include='*.cu' --include='*.cuh' --include='*.inc' src/`.

Substitute the sha at the moment of quoting. A line that cannot carry one should say "as of this run" or stay
unquoted — see Appendix L on absence claims and §11 on ref-sets.

## Why this file exists rather than another chat post

The instrument worth keeping is not the count, it is the *disposition*: `t3_two_hop_seam_check.py` invoked with
an unknown argument prints

    !! INSTRUMENT ERROR (not a verdict): FileNotFoundError: --help/src absent -- refusing to report clean

— it declines to answer a question it was not given instead of returning green. That is the single most
reusable property in the tripwire class: a gate that reports CLEAN on an unparsed invocation is a gate that
reports CLEAN because it parsed nothing. Copy it into every gate that takes a path or a root.
