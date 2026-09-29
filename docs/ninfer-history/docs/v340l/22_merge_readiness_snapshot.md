# 22 — Merge-readiness snapshot: where the registration rows actually are, and the one conflict to resolve by hand

Taken 2026-09-13 before the successor's three merges, because two claims in the closing census placed the
registration on a branch that never had it. **The shas below are already drifting** — gemini's tip moved
`18f520bb` → `c25d7595` between drafting and committing this file, with every count unchanged. That is not a
defect in the numbers, it is the reason each one carries its ref: a tip-stamp is what lets a reader tell
"moved and still true" from "moved and now false." Re-run the block at the bottom before quoting any of it. Everything here is plumbing that **writes nothing** — safe to re-run
at any time, including while a boot holds the cards.

## Where the rows are (`REGISTERED_EXCEPTIONS` counted as an array block, not by loose grep)

| ref | tip | entries | guard four | agent4's four |
|---|---|---|---|---|
| `origin/amd/wo-gfx900-perm` (gemini) | `18f520bb`, re-read as `c25d7595` **within the hour** | **17** | 0 | 0 |
| `origin/amd/t3-wip` | `e2b15c12` | 28 | 0 | 0 |
| merge-base(gemini, main) | `e1ea5dbb` | **17** | 0 | 0 |
| `origin/amd/main` | `f8fd03f8` | **36** | **4** | **4** |

Main is 392 commits ahead of gemini's branch; gemini is 78 ahead of main.

**Why gemini's count can never be "32": their branch has never edited the array.**

    git diff $(git merge-base origin/amd/wo-gfx900-perm origin/amd/main) origin/amd/wo-gfx900-perm \
      -- tools/ops/gate_pg1_whitelist.sh        # -> EMPTY

gemini's 17 **is** the merge base. So the eight rows exist because of **main's own commits**, and neither
"add the four rows" nor "expect the merge to carry them" describes the real state: the first would duplicate on
main, the second expects nothing of a branch that never had them.

## The array merges clean; the requirements sheet does not

    git merge-tree --write-tree --name-only origin/amd/main origin/amd/wo-gfx900-perm
      CONFLICT (add/add): Merge conflict in docs/amd/TRIPWIRE_GATE_CLASS_SPEC_agent5.md
      merged array: entries=36  agent4four=4  guard4=4

Two independent halves of that output:

1. **No clobber risk on the gate's core artifact.** Because gemini's side is unchanged relative to base, a
   3-way merge keeps main's 36 rows. (I had flagged the opposite as a possibility an hour earlier and it is
   closed — one `--numstat` was enough, which is the whole argument for checking before fearing.)
2. **One genuine landmine: `add/add` on `docs/amd/TRIPWIRE_GATE_CLASS_SPEC_agent5.md`.** Both lines created
   that path independently, so any default resolution — including `--ours`, which is decided by merge direction
   and not by content — discards one lane's edits wholesale. That file is the requirements sheet every tripwire
   law of the session was feeding, so resolve it by hand:

       git show <main-side>:docs/amd/TRIPWIRE_GATE_CLASS_SPEC_agent5.md  > /tmp/spec_main.md
       git show <theirs> :docs/amd/TRIPWIRE_GATE_CLASS_SPEC_agent5.md  > /tmp/spec_theirs.md
       diff -u /tmp/spec_main.md /tmp/spec_theirs.md      # merge by clause, not by side

   Also worth checking the same add/add shape for **any** path both lines authored independently — this one was
   found only because `merge-tree` printed it, not because anyone was looking for it.

## Outstanding item of mine, small but live

`src/common/hip_shim/README.md` on main still cites the shim's shuffle definitions at
`cuda_runtime.h:441/:451/:466` plus "3-arg overloads at `:477-480`". Main's own file says:

    473 __shfl_xor_sync · 483 __shfl_down_sync · 491 __shfl_up_sync · 498 __shfl_sync

30–32 lines off, with the overload range now pointing into the primary definitions. My branch's replacement
(symbol grep, no remembered numbers) was **clipped** in `953c6671` because gate check (a) flags an unregistered
modification to a pre-existing `src/` file — a correct verdict (`--diff-filter=M`, and the file exists at the
baseline). Either register the row or move that guidance to `docs/amd/`, where check (a) doesn't reach; the
second is cleaner, since the array self-documents as being about preserving CUDA code in pre-existing files and
a Markdown file is not that claim.

## Reproduce in one block

    F=tools/ops/gate_pg1_whitelist.sh; G=origin/amd/wo-gfx900-perm; M=origin/amd/main
    for r in $G $M; do echo "$r $(git show $r:$F | sed -n '/REGISTERED_EXCEPTIONS=(/,/^)/p' | grep -cE '^\s*\"')"; done
    git diff $(git merge-base $G $M) $G -- $F | head          # empty == gemini never edited it
    git merge-tree --write-tree --name-only $M $G | head -20   # zero writes; safe on a live board
