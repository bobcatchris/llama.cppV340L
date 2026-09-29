# agent1 step-1 read audit (WO-SUPPORT-1 §6.1) — refs verified at seat, two dead citations found

**Desk**: agent1 · **Date**: 2026-09-13 ~15:2xZ · **WO**: `docs/amd/WO_SUPPORT_agent1.md` @ `dfd5b7ec`

## Registry announcement (forensics subtraction, per WO §2 clause)

- **Worktree**: `/home/chris/worktrees/amd-wo-agent1-support`
- **Branch**: `amd/wo-agent1-support`, created from `amd/main` @ `dfd5b7ec` (chair-issued WO tip)
- Created with `git worktree add ... -b` — new ref, moved nothing, no `fetch --prune` run at this desk
  (forensics-freeze compatible; this line IS the registry row agent3's confound table needs).

## Reads done, with the ref each was read at

| item | read at | verdict |
|---|---|---|
| `WO_SUPPORT_agent1.md` | HEAD `dfd5b7ec` | read in full |
| `RESTART_2026-09-13/COORDINATOR-SUCCESSOR.md` | HEAD `dfd5b7ec` | read in full |
| `BOOT_LAUNCH_RUNBOOK.md` §2–4 | HEAD `dfd5b7ec` | read in full — two defects below |
| G-AMD-27 item-7 verdict | `1d0ff3c6` (`origin/amd/t3-wip`) | read in full (commit body is the row) |
| G-AMD-26 rows | `51dc3dd6` → `results/amd/p3/G17v_verdict_row.txt` | read in full |
| Artifact bank record | `21a095e8` → `results/amd/p3/ARTIFACT_BANK_2026-09-13_record.md` | read in full |
| `results/amd/p3/G3_bringup_kit.sh` v6 | HEAD `dfd5b7ec` | read in full — one defect below |

## Self-report first (my own instrument, not theirs)

My first path-resolution pass printed **all-NO** for seven WO-cited instruments. Cause: I ran
`git cat-file -e <path>` with a *bare working-tree path*, which resolves against the object DB as a
rev-spec, not the commit. Re-run as `git cat-file -e HEAD:"<path>"`: **all seven resolve GREEN at
`dfd5b7ec`** (tps_probe.py, b1_rccl_ar_sweep.hip, run_b1_rccl_ar_sweep.sh, G3_bringup_kit.sh,
WO_TP4_all_lanes.md, 99_agent_work_order_template.md, BOOT_LAUNCH_RUNBOOK.md). The WO's
"all cited paths resolve-checked at my seat before issuing" claim **held**; my check was the faulty
instrument. Recorded because the board's own law is that a check which can fail must be named
before its failure is credited.

## Defect 1 — runbook §3 cites a live sha for the wrong fact (`ref-without-kind`, my class label)

Runbook: "`tools/v340l/tps_probe.py` (on main since `a9d8792`… correction: `a9d38792`)".
The correction names a commit that **exists and is an ancestor of `amd/main`** (verified by
`git merge-base --is-ancestor` across all 11 local+remote refs) — but it is *not* the commit that
introduced that file. `git log --diff-filter=A -- tools/v340l/tps_probe.py` on `amd/main` gives
**`53f9d79a`** ("tools(v340l): tps_probe.py — stream-timed prefill/deco…"). Right object, wrong
proof: the "since <sha>" predicate is false while the "on main" predicate is true. Fix for Task A:
cite `53f9d79a` as introduction and `HEAD:<path>` for presence, which is the only two claims a
reader can re-derive in one command each.

## Defect 2 — kit v6's pair-swap authority is a dead sha

`G3_bringup_kit.sh` header (v6): "pair-swappable per chair ruling `19f39c4c`".
`19f39c4c` resolves **nowhere**: not as an object on any of the 11 refs checked above
(`git cat-file -t` fails; the "e→f disease" class the chair named at #797(2) recurring in a
*committed instrument*, not just in chat). The **ruling itself is real and load-bearing** — the
mask/ordinal law it encodes is independently stamped at **`9c2a7876`** ("kit v5 — the
mask/ordinal interaction: HIP_VISIBLE_DEVICES REMAPS selection…", verified present). So the fix is
a one-token citation swap in the kit comment, **not** a re-litigation of the pair convention.
Handed to agent4 (kit owner) as a comment-only patch candidate; I do not edit their file.

## Also measured at this desk (facts Task C needs, not inherited)

- `sha256sum /home/chris/artifacts_bin/ninfer-serve_0c62ffdfd402a4b8.bin` =
  `0c62ffdfd402a4b87dbf280497244ed13e5ed06a94bb11d04036813d7cdb9412` — matches its filename stamp and
  the full sha in `G17v_verdict_row.txt`. **Task C's bank-pinning predicate is satisfied at bytes at
  my seat**, so the "revisit if the bank hash-checks wrong" branch (§5) is closed.
- `df -h /` at start of my steps: **15 G avail** (WO §8 states 16 G). One full build headroom is the
  binding constraint and it has already tightened by ~1 G at another desk since issue. Tasks A and B
  are zero-build; Task C is **structurally** zero-build — see Defect 3.

## Defect 3 — kit v6 cannot execute Task C as written (blocks nothing, but must be named)

Kit v6 hardcodes `LANE=/home/chris/worktrees/amd-wo-p3-serve`, `cd "$LANE"`, then unconditionally
runs `cmake --build build-hip-amd --target ninfer-serve` and boots
`build-hip-amd/apps/ninfer-serve`. Task C is specified as *no build* (disk law) on the *banked*
binary (boot-from-bank law `f49dfe4a`: filename-is-stamp, never a lane build path, which relinks by
design). So the kit as merged is the wrong instrument for this cell: it would (a) take a build slot
on a 15 G box, (b) relink agent4's lane binary — the exact clobber mechanism the bank law exists to
kill, and (c) boot a moving path rather than the pinned sha. Task A's runbook therefore specifies the
banked-binary launch as an explicit argv row with the kit's *measured recipe* (env + flags kept
byte-identical to the `17f`/`G17v` shape) rather than a kit invocation, and says why. No kit file
edit from this desk.
