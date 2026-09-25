# MERGE-GATE CHECKLIST — post-[C] landing of the p3-serve band into `amd/main` (agent2, chair seq-42 task B deliverable; the chair's pen, agent2's tables)

Scope: this is the INPUT to the merge-gate call, not the call. Everything below was measured at my
seat on 2026-09-15 against `amd/main` = `098f417d` and the fire tips
`3a1a91ed` (r18, boot stamp `f8e762425d0d8bb0`) / `d263fea4` (r19, boot stamp
`18e75d077af6c72c`), `origin/amd/wo-p3-serve` head = `69def383`.

## 0. HEADLINE, stated once, with its probe

**The merge is a strict superset on code: main touched ZERO files under `src/` or `tests/` since
the fork, the lane touched 31 (+2106/−196), and the overlap set is EMPTY.** So "main's version wins
by law" has nothing to enforce on this band — the anti-resurrection clause exists for the case where
a branch carries an OLDER copy of main's canonical region, and that case is not present here.
Probe set (all three must be re-run at the tip you actually merge; the numbers below are 2026-09-15
numbers):

```bash
MB=$(git merge-base amd/main origin/amd/wo-p3-serve)          # 66366ede at writing
git log --oneline $MB..amd/main -- src tests                  # EXPECT: empty
comm -12 <(git diff --name-only $MB amd/main -- src tests | sort) \
           <(git diff --name-only $MB origin/amd/wo-p3-serve -- src tests | sort)   # EXPECT: empty
git diff amd/main origin/amd/wo-p3-serve -- src/runtime/tp2/tp_engine.cpp src/runtime/tp2/tp2_budget.h \
    | grep -c '^-[^-]'                                        # EXPECT: 0 (law's own anti-resurrection arm)
```

## 1. ORDER OF LANDING — five steps, each with its own falsifier

**Step 1 — ANTI-RESURRECTION FIRST, as a gate, not as a review.**
The law's two named files at the fire tip: `tp2_budget.h` **byte-identical** (blob `4a567aae`,
content8 `e369358a`, zero diff); `tp_engine.cpp` **12 added lines, ZERO deleted** (`#include
<functional>` + two `getenv("NINFER_BATCH_DBG")` breadcrumb blocks around the engine-mutex lock).
The preflight region proper — extracted by pattern, hashed, not eyeballed — is 54 lines hashing
`afceb97e` **at both refs**. Keyword sweep of the whole engine diff for
`preflight|budget|fixed_bytes|staging_bytes|reserve|slack|kGateRelief` = 0 changed lines; the
banned-constant sweep returns the same hit set on both sides and `tp2_runtime_reserve_bytes = 0`
(a measured zero, not a charge) is common to both.
*Falsifier:* the `grep -c '^-[^-]'` above is the standing arm — it fails loud and prints the
reverted lines if a future branch does carry an older copy. Run it before authorizing, not after.

**Step 2 — the `tools/v340l` arming kit, BEFORE the src band, not with it.**
`tp4_arming_battery.sh` + its four checkers (`admitted_shape_instantiation_check.py`,
`gqa_head_geometry_cell.py`, `collision_fence_plant.py`, `head_row_value_check.py`) are
main-resident (landed `1917a8b7`) and **absent from every p3-serve tip** — measured with
`git cat-file -e` at `3a1a91ed`, `69def383`, `d263fea4`. Consequence I measured rather than
predicted: the battery at the fire tip prints **4 NOT-OBSERVABLE legs** when run from its main home
and only grades 18/18 GREEN when the checkers are supplied as siblings — i.e. **the lane cannot
self-arm today.** Merging src first leaves the boot tip grading its own shape from borrowed bytes,
with the battery's own WARN line on the GO row. Land the kit first and the WARN disappears by
construction.
*Falsifier:* `git cat-file -e <merged-tip>:tools/v340l/tp4_arming_battery.sh` and re-run the battery
at the merged tip — leg count should be **18 with ZERO `WARN PRED-D selftest` line**; if the WARN is
still there, step 2 did not land.

**Step 3 — the src band (`66366ede..69def383`), one merge, no cherry-picking.**
This is where main's stale copies DIE by law, and the list is short and specific — each row was read
at both sides' bytes, and each names which tree is which:

| organ | `amd/main` today (fork-era, LOSES) | lane (`69def383`, WINS) |
|---|---|---|
| `src/runtime/tp2/tp2_backend.cpp:1145` | `const int n_vocab = 248320 / 2;` | `:1177` `248320 / tp_w` + new loud refusal `if (248320 % tp_w != 0) throw` naming both numbers |
| `:1156` | `state->full_logits = a_bf(2 * n_vocab);` | `:1188` `a_bf(tp_w * n_vocab)` — sendcount × WORLD |
| `:2576` | `Tensor full_logits(…, {248320, 1})` | table-derived extent stated at the call site |
| `:1626 / :3591 / :5224` | three `std::barrier sync_bar(2)` | `:1659` `sync_bar(backend.world())` |
| `src/ops/wrapper/gqa_attention.cpp:40-48` | world-free ladder `{24→4, 16→2, 12→2}` + throw-elsewhere | `head_geom_for_q()` view of the generated table; unknown `q_heads` refuses LOUD naming the number |
| `src/runtime/tp2/tp2_request.{h,cpp}` | pair-sized pinned slots | world-sized (family-#8 cure) |

New files the band brings that main does not have (all verified present at the lane tip):
`src/ops/shape/tier_shape_table.h` (generated; 15 `HeadGeom` rows),
`tools/ops/gen_tier_shape_table.py`, `tools/ops/check_kv_tier_priced.py`,
`tests/test_tier_shape_table.cpp`, `tests/multi_gpu/tp_argmax_r1_host.cpp`,
`src/core/multi_gpu/{argmax_r1.h,r1_argmax.cu,r1_argmax.h}`, `gather_capacity.h` is ALREADY in both
(`5c9ed9c4` is below the fork — shared ancestry, not a divergence).

*Falsifier:* after the merge, `git grep -c "sync_bar(2)" <tip> -- src` must be **0**,
`git grep -c "248320 / 2" <tip> -- src` must be **0**, and the battery's PRED-E/T1/T3 legs must all
print `ok` with no sibling fallback.

**Step 4 — the gate-letter band, with the COLLISION named (this is the row the chair should read
twice).**
`amd/main`'s `gate_pg1_whitelist.sh` carries **(a)..(s)** — 19, last block `# Check (s)` Census
reader retry attribution. The lane lands **(t)** R1-dispatch-shape, **(u)** load-world, **(v)**
tier_shape_table; `origin/amd/wo-r1-transport` carries **(t)/(u)** of its own. **So (t) is NOT
free in the merge, even though it is free at main today.** My registration PR therefore takes
**(w)** for the NVFP4 census-gate family and says why in the block's own comment; if the chair
prefers a different assignment, it is a one-line rename in a self-contained block — but landing my
(w) and the lane's (t)/(u)/(v) together is currently collision-free, and landing anything at (t)
alongside the lane is a duplicate-letter gate (the LABEL-collision family this board already ruled on
once). Sequence that keeps it true: **lane src band first, then my (w) block, or both in one merge —
never two independent (t)s.**
*Falsifier:* the leg-count-by-letter discipline already in use: at the merged tip
`grep -oE "# Check \([a-z]\)" | sort | uniq -d` must print **nothing**.

**Step 5 — post-merge, re-arm and re-grade in this order (all host-only, ~40 s total).**
Precondition discovered by simulation and stated up front, because skipping it produces two false
REDs that look like merge damage: (i) **re-fetch before re-running the gate** — check (a) diffs the
tree against `origin/amd/main`, so a stale remote ref makes the gate convict the merge you just made
("16 pre-existing CUDA src/ file(s) modified"); (ii) **build the merged tree before running (v)** —
with no `build-hip-amd/tests/ninfer_tier_shape_table_test` present the leg falls to its standalone
compile path and dies deterministically on `ETXTBSY` (see §1b BLOCKER 2; the cell itself is fine:
255/255 GREEN when compiled by hand). And (iii) expect Check (w) to go RED on L5 until agent3's
disjunction fix lands (§1b BLOCKER 1) — that RED is the bar working, not the merge failing, and the
ledger row must say so in the same breath as the sha.

1. `bash tools/v340l/tp4_arming_battery.sh <merged-tip> <checkout>` → expect 18 legs, all `ok`,
   **no WARN line**, rc=0. (Cite the leg count AT that ref; 17/18/19 are all "true" of different
   trees and that is exactly how remembered numbers rot.)
2. `bash tools/ops/gate_pg1_whitelist.sh` → expect the FULLY-certified banner, rc=0.
3. `python3 tools/v340l/nvfp4/agent3_battery.py --repo .` → 41/41, VERDICT=0.
4. Host serve suite 9 → four-count 9/0/0/0.
5. Only then a boot: battery-seconds at the new boot bin's sha (agent2 standing duty), and if any
   leg prints NOT-OBSERVABLE that is a kit/input gap and gets reported as one — never filled in
   from this document's numbers.

## 1b. TWO BLOCKERS FOUND BY SIMULATING THIS PROCEDURE (measured, not projected)

I did not trust §1 as written — I built it: a scratch clone of the repo, `origin/amd/main` as the
base, `amd/wo-p3-serve` merged in with my (w) block re-applied, and ran the gate and the bar there.
Two blockers fell out, and one of them is aimed at MY OWN freshly-landed registration.

**BLOCKER 1 — after the src merge lands, my Check (w) goes RED on a correct tree, because agent3's
pad arm pins main-only literals.** Measured, in the simulated merged tree: `agent3_battery.py` →
`BATTERY 40/41 legs as expected / VERDICT=1 legs off-expectation: ['L5']`, and my bar correctly
forwards that as `FAIL: NVFP4 census-gate family graded RED` → gate rc=1.

* Mechanism: `pad_region_census_arm.py` `DECL_SITES` lists 6 declaration markers; 3 of them are the
  w2-era staging literals and their own metadata says *"amd/main ONLY; the serving lane replaced it
  with `248320 / tp_w`"*. But arm **A1 requires all six to be present in whatever tree it grades**
  and `a1.fail(...)` on a miss — so the arm **convicts the tree that FIXED the drift**. At main:
  `mainform=1 laneform=0`, A1 GREEN. At every lane tip (`3a1a91ed`, `d263fea4`, `69def383`,
  `c2629e4b`) and at my simulated merge: `mainform=0 laneform=1`, A1 CONVICT, `PAD 5/6 | blocking:
  ['A1']`, rc=1. Four refs plus a merge — the conviction is a property of the arm, not of a tree.
* Why it is a real merge-window problem rather than a cell quirk: the pad arm is a leg (L5) of the
  battery, the battery is the entry point of the (w) block, and the (w) block is in the gate that
  every fire pre-flights. So the day step 3 lands, the PG-1 gate starts printing a RED whose text
  says "accept-set wiring drifted or a guard went blind" while the actual cause is that main got
  BETTER. That is a false-RED aimed at the most-loaded moment of the week.
* Cure, agent3's to write (their cell, freeze law): make the three staging markers **per-tree
  disjunctive** — each site carries the form expected for the tree's own generation, or the arm
  accepts EITHER form and names which one it saw. Evidence that a disjunction is strictly stronger
  than a main-only pin, not weaker: I measured each tree carries EXACTLY ONE form
  (main `1/0`, merged `0/1`), so "exactly one of the two" is assertable and **both-present is
  precisely the half-migration crash shape** the arm's own line 29 warns about — today that shape
  would pass A1 entirely (it only looks for the main form). So the fix buys a new conviction as well
  as removing a false one. Falsifier to pair it with: plant the second form next to the first on a
  copy → A1 must CONVICT by name (that arm does not exist today; nothing today sees both-present).
* Interim, if the merge has to beat the fix: my bar can take a documented `--allow-l5-drift` passthrough
  rather than the gate going falsely RED — **I am NOT adding it unilaterally**; a bar with a
  built-in excuse is how the next real L5 red gets walked past. Chair's call, one line either way.

**BLOCKER 2 — (v) cannot run on a tree with no build dir: `OSError: [Errno 26] Text file busy`
(`ETXTBSY`), gate rc=1.** Reproduced in the same simulated tree, where `build-hip-amd/tests/
ninfer_tier_shape_table_test` does not exist so the leg falls to its standalone compile path. The
traceback ends in `(v)`'s python fallback, which compiles with `tempfile.NamedTemporaryFile(...)`
and then EXECUTES the still-open handle — ETXTBSY on Linux, deterministic on this class of call.
This is NOT a new finding: it is my own doc-27/doc-26 **`(v)`-FALLBACK ETXTBSY** row (2× repro,
`mkstemp` cure verified) that "rides its author's desk" — what is new is that it bites the
**post-merge** gate too, because a merged-but-unbuilt tree is exactly the state where the fallback
runs. Cure stays the author's (`mkstemp` → close → exec, or compile to a path you do not hold open).
Consequence for §1 step 5: **run the gate on the merged tree only after that tree has been built**
(or accept the (v) ETXTBSY red as known-and-named, in which case write it in the ledger line so
nobody re-hunts it). I verified the underlying cell is fine — `g++ -std=c++20 -I include -I src
tests/test_tier_shape_table.cpp` rc=0 and the binary prints `GREEN: 255/255 checks pass (21 table
roles, 3 served worlds)` — so the RED is 100% runner plumbing, zero arithmetic.

**Also measured, and NOT a blocker: my own simulation's first false alarm.** The first merged run
failed check (a) with "16 pre-existing CUDA src/ file(s) modified" — that is BASELINE STALENESS in
my scratch clone (its `origin/amd/main` ref still pointed at the pre-merge tip, which is precisely
what (a) diffs against), not a merge hazard: after `git update-ref refs/remotes/origin/amd/main
HEAD` the (a) legs go clean. Filed here because it is the failure mode a real seat will hit —
**after the merge is pushed, re-fetch before re-running the gate, or the gate will convict the
merge you just made.** (Same family as the `check_census_retry_attribution` write-set I noted, and
as agent3's a1-a4 arms aging out on deleted copies — a falsifier that reads a moving ref must be
re-pointed, not re-membered.)

## 2. FINDING FOUND WHILE WIRING (filed, not fixed by me) — a standing gate leg dirties the tree it grades

**Class: an instrument that mutates its own evidence invalidates the precondition every fire receipt
claims.** `tools/ops/gate_pg1_whitelist.sh:751` runs `results/amd/run_t3_ldmatrix_negative_cell.sh`,
and that runner writes its log to `"$CELL/t3_ldmatrix_negative_cell.log"` (`:19`, truncated at `:22`)
where `CELL` is **the tree under test's own `results/amd/`** — and `results/amd/t3_ldmatrix_negative_cell.log`
is a **TRACKED** file (`git ls-files --error-unmatch` → yes; provenance `7c02a299`).

**Measured blast, twice, from my seat:** starting from `git status` clean, one `bash
tools/ops/gate_pg1_whitelist.sh` run leaves exactly ONE modified tracked file:
`M results/amd/t3_ldmatrix_negative_cell.log` (31 lines change, and the change is the ABSOLUTE PATH
of whichever tree ran it — `/home/chris/dual_5060_ti_ninfer/…` vs
`/home/chris/worktrees/amd-wo-shim-funcattr/…`). I swept every runner the gate invokes for
repo-path writes: the census-attribution cell writes only into its temp dir and *tracks* its
write-set (`note_write`, and its own log prints "15 paths written this run, 0 inside the repo"),
the other four cells write nothing outside temp. So the blast is ONE path, not a family — but that
path is enough to break three things the board depends on:

1. **The fire-receipt precondition.** The R1/fire template's first line is *TRACKED-DIRTY-EMPTY
   verified* before detaching a run. Any desk that runs the PG-1 gate as a pre-flight (which is what
   it is FOR) then finds a dirty tree, and has exactly two bad options: commit junk, or assert
   tree-equality while a tracked file differs. That is a false-DIRTY — the same family as the false
   RED, with the same consequence: the next person stops trusting the check.
2. **Cite-by-sha on that log.** Its content8 depends on the CWD of the run, so the banked log's stamp
   is not re-derivable by another seat — a stamp that cannot be reproduced by an honest re-run is a
   stamp that should not be cited.
3. **Lane-vs-main merges.** A lane that runs the gate before merging picks up a spurious local
   modification and can lose it on `checkout --` (I did, deliberately, and restored from HEAD).

**Cure, which is the author's to write, not mine** (freeze law; the runner is T3's family):
send the log to a `mktemp -d` and delete on exit, OR keep writing it but make the path relative and
`git`-ignore it, OR scrub the absolute prefix out of the captured text. All three are ~3 lines. The
**standing-cell** the closure law wants with it is one arm, both directions: *`bash
 tools/ops/gate_pg1_whitelist.sh` from a clean tree must leave `git status --porcelain` EMPTY* — GREEN
after the cure, RED today (I have the red capture: the `M` line above, at my seat, twice).
That arm belongs in `run_ci.sh`'s host step or as a gate self-check, not in my PR — I am landing
`(w)` with the log restored to HEAD and this row in the checklist so the chair can route it.

## 3. WHAT THIS CHECKLIST DOES NOT DECIDE

* **Whether [C]'s verdict changes the plan** — that is agent4's `c_grade` read + agent1's decode; my
  rows are about WHICH BYTES LAND, not what the capture means.
* **The `248320`/`248077` literal consolidation** (doc 32 row 8): the merge makes the table the
  source for head geometry; the vocab extent still sits in 5 live places with no single home. After
  step 3 that is a clean, separate WO, not a merge blocker — say so rather than smuggle it in.
* **Registry/roster rows for new files** (`check (a)` extension roster): that leg is main's and it
  will see the band's new `.cpp/.cu/.h`. Precedent from `5c9ed9c4`'s own row: roster additions are
  drafted as a patch beside the merge (`docs/amd/REGISTRATION_4ROWS.patch` style) so a lane never
  edits the roster under time pressure. Expect that draft, don't hand-apply it.
* **Anything about `main` (the NVIDIA line).** Team law: `amd/main` only, one direction.

## 4. WHAT THIS PR CONTAINS (subject upgraded per seq-49 to ONE entry point)

`tools/ops/gate_pg1_whitelist.sh` — new **Check (w)** block;
`tools/ops/check_nvfp4_census_gate_family.sh` — new bar: verifies the banked fixture's sha256+size
against the stamp in its own filename, then runs **agent3's `agent3_battery.py` (468a78e5) as the
single family entry point**, and separately runs it against the real 18,324,067,840-B artifact when
a human mounts one. Five states measured at my seat: fixture-ok→rc=0; stale fixture→rc=1; wrong-size
`--artifact`→rc=1 ("refusing to grade against it"); fixture absent→**rc=2 with the gate banner
degrading to "PASS with 1 UNGRADED leg(s) … DO NOT cut a GO row quoting this banner"**;
fixture-absent-and-no-artifact→rc=2 before any run.
Plus the 2-line divisor-cell build-comment cure: comment-only diff verified
(`git diff` shows zero non-`//` lines), and the cell re-verified at the moved stamp
`6062c5fd` — **41/41 arms rc=0**, so the stamp move carries its own green capture; prose cites to
`20eb99a7` (NOTES §F1, gate header :53, release note :119/:132) are listed in the PR body as
lineage, not silently rewritten.

**Cross-desk corroboration row (chair-asked):** agent3's battery prints C4=247, C8=700, C8c=1052,
C8d=NONE, C9 extent=248077 / rows=248320 / pad=243, holes=0 — **every shared digit equals my
independent item-1 receipt, which I ran at v3.1 bytes (`b0309248`) against the REAL artifact, while
they ran at v4 bytes (`f6d8e381`)**. Two desks, two trees-of-bytes, two inputs, one arithmetic. And
their honesty shape matches mine: they verified the artifact at **size**-of-record grade and
explicitly declined hash credit ("won't borrow a leg it didn't run"); the 64-hex verification of
record remains doc 29's `eaf8ad124256d0a0…56d2`, measured 2 m 04 s on the mount seat, and this PR
does not quietly inherit it either.

**Release-debt row SETTLED, cited:** the `+6.95 MB` bank-size growth I had left as an unresolved
unit question is exact decimal **6,953,184 B** per agent3's stat arithmetic (chair relay at
`098f417d`) — struck from the debt list here rather than in the ledger I do not own.

— agent2, zero card, zero build in the shared checkout (my lane only), `/tmp` scratch reaped,
no server started, no device opened. Every number above is from a command run at this seat on
2026-09-15; every file:line was read at the named ref, not quoted from a brief.

---

## APPENDIX 0b — THE RE-SIMULATION, EXECUTED (agent2, chair seq-84 order; bases `origin/amd/main` d5a2c04f + `amd/wo-p3-serve` **384d0dac**, sim tip 538f6235, tree kept at `/tmp/seq_sim`)

Six stamps verified at bytes before anything ran: pad `a069748f`, battery `4a364422`, join `e80f5d86`,
census `abfc0bc2` (= `r_layer_census_cell.py`), purity `1fa777f4` (= `gate_tree_purity_cell.py`),
runner `43430e6f` — all four relayed names matched; two had different filenames than the relay used, so
the row records both.

| # | assertion | result | note |
|---|---|---|---|
| 1 | conflict set == the gate file only | **PASS** | one contentful conflict, exactly as §1b predicted; resolved from the two COMMITTED blobs, `bash -n` + zero markers + `uniq -d` empty |
| 2 | pad arm at the merged tree | **PASS, and it PRINTS WHY** — `A1 GREEN … graded DISJUNCTIVELY`, then `LANE form (lane :1177/:1188/:2619) — world-derived, drift CURED in this tree`, plus `A5b` convicting the both-forms plant | this is the W1-vs-W2 distinguishability I asked for, delivered: the two greens are tellable apart from bytes alone, so "green for a new reason" is no longer invisible |
| 3 | my registered bar (`Check (w)` leg) | **PASS** rc=0 | tally printed as observed, not remembered |
| 4 | battery on the merged tree | **PASS** `BATTERY 76/76 VERDICT=0` | law-line worked instance #2 confirmed and *grown*: 41 (this morning, pre-shadow-cure) → 58 (chair's number) → **76** at this base set. Every count was true of its own box+corpus; none may be quoted at another |
| 5 | re-fetch law, tested adversarially | **LAW CONFIRMED** | stale baseline ref → `(a)` convicts the merge ("16 pre-existing CUDA src/ file(s) modified", rc=1); after `update-ref` those legs go clean |
| 6 | build-before-(v) | **LAW CONFIRMED, and INCOMPLETE** | no build dir → `(v)` dies on `OSError: [Errno 26] Text file busy` (agent4's runner per chair ruling; I verify, do not fix); WITH a hand-built `build-hip-amd/tests/ninfer_tier_shape_table_test`, `(v)` passes `GREEN: 255/255 (21 roles, 3 worlds)` — so the precondition works. **But the gate then fails one leg DEEPER: `check (r)` parity `[RED] pair-state E: src/core/multi_gpu/r1_argmax.cu declares 2 [2] = {nullptr|0,...} arrays — family #8's exact shape`.** |

**Finding from row 6 — the (v) precondition exposes a SECOND, real, pre-existing lane RED that the
runner blocker was hiding.** `check_whitelist_arm_parity.py` convicts `r1_argmax.cu:70-71`
(`static thread_local void* p[2]` / `cap[2]`) — pair-shaped slot arrays, the exact family-#8 shape the
board cured at the request-state level. Measured at each base alone, so this is not merge damage and
not my splice: **absent at `origin/amd/main`** (the file does not exist there), **present and RED at
`origin/amd/wo-p3-serve`** (same two lines, byte-identical). So it is a lane-side open row that will
land WITH the src band and make `check (r)` go RED on main the moment the merge is pushed —
`tools/ops/check_whitelist_arm_parity.py` is main-resident, the lane file is not. **Nobody has been
able to see this through the gate until the (v) blocker is cleared, because the gate exits at (v)
before reaching (r).** That is the definition of a registration-shaped blind spot: the bar's order
conceals a leg's verdict. Needs agent4's eyes before the window: either the arrays get the
world-derived treatment the family already got elsewhere, or the parity cell gets a registered,
named allow-entry with a reason (the census already registers 75 literals by name+reason, so the
mechanism exists and is the honest option if the shape is deliberate at world-independent slots).

**Also measured, LOW but worth a line: agent3's purity cell `P1` false-indicts on embedded awk.** At
the merged tree: `PURITY 4/5 | off: ['P1']`, indicting `r1_ring_guard_check.sh:145/151 ->
tools/v340l/one_shot_argmax`, `tools/v340l/pe`. Both targets are **not files**: line 145 is
`if (lines[i] ~ /if *\( *!(impl_->one_shot_argmax|ring)/) {` and 151 is `if (term && call > pe) {`, so
an awk **comparison** and an awk **regex literal** were parsed as shell redirects. Their masking pass
quotes only (`re.sub` over quoted spans), and this awk sits inside a single-quoted heredoc body whose
interior is code, not a string — so the fix is to skip awk/sed program bodies, not to add more quote
rules. Routed to agent3 as a note: a static write-set auditor's credibility dies on false INDICTs
(their own comment says exactly that, about `tools/ops/Falsifier`), and mine is the cell that would
have to live with the noise. My `check_gate_write_set_cleanliness.sh` does not have this failure mode
because it observes porcelain rather than parsing redirects — which is the argument for both bars
registering, and the reason I can certify theirs is worth keeping despite the noise row.

**Certification, stated narrowly.** What 0b proves: the pad cure works on the merged tree and prints
which form it saw; my registered bar passes there; the battery passes with 76 legs; my re-fetch and
build-before-(v) preconditions are both real. What 0b does NOT clear: `check (r)`'s pair-state E RED
(row 6), which is lane-side, pre-existing, newly-visible, and belongs to agent4's lane before the
merge is pushed. **My recommendation: window opens after that row is addressed or explicitly
registered-with-reason, not before** — because "the gate can't see it until (v) is fixed" is a
warning that the next blocker will hide something else.

## APPENDIX A — THE SEQUENCE ROW (chair seq-69 task): cure-landing → merge → post-merge

Question this answers: between agent3's disjunctive-A1 fix landing and the src merge landing, **what
does the bar print, and is that state correct or ambiguous?** Answer: it is correct in every window,
and the ambiguity lives in ONE place (L5's meaning flips semantics at the merge), which is why it gets
a named row instead of a shrug.

**Step order, ruled:** (0) agent3's A1 disjunction (+ its both-present falsifier) lands on `amd/main`
→ (0b) I re-run the FULL merge simulation at that fixed tip and it must print clean → (1..5) the
existing §1 steps. **The window does not open before 0b prints clean** (chair seq-69 ruling 2; I am
the designated simulator). Nothing else in §1 depends on the cure, so steps 1-2 can be prepared in
parallel; only step 3's gate-leg state depends on it.

### A.1 The leg-by-leg state table, per window (expected = what SHOULD print; a mismatch is a defect in the bar, not in the tree)

| leg | W0 today (main bytes, pre-cure) | W1 cure landed, src NOT merged | W2 src merged (cure already in) | W3 pathological: cure skipped, src merged |
|---|---|---|---|---|
| L1 gate `--selftest` | `ok` 23/23 | `ok` unchanged | `ok` unchanged | `ok` unchanged — the gate cell does not read staging literals |
| L2 live census | `ok` VERDICT=0 warnings=2 | `ok` same | `ok` — census is artifact-side, staging-invariant | `ok` same |
| L3 drift ring | `ok` RING 18/18 | `ok` | `ok` — ring grades gate/ring bytes, not `tp2_backend` | `ok` |
| L4 ring `--falsify` | `ok` 8/8 | `ok` | `ok` | `ok` |
| **L5 pad arm** | **`ok`** printing `MAIN form … (:1145/:1156/:2576)` | **`ok`** printing the SAME MAIN-form line (main is still main-shaped) | **`ok`** printing `LANE form (lane :1177/:1188/:2619) — world-derived, drift CURED in this tree` | **`RED` "blocking: ['A1']"**** ← the §1b BLOCKER 1 state; correct-as-evidence (the arm can only see one form), wrong-as-verdict about the tree |
| L6.* counter-join per capture | `ok`, count = discovered captures (33 here) | same | **count can GROW** — a merged tree sees main's `results/amd/p3` AND the lane's, and r21/r22 fire logs land in the lane dir | same as W2 |
| L6.absence / L6.absent-file | `ok` (both expect rc=2) | `ok` | `ok` | `ok` |
| L7 self-falsifier | `ok` (expects rc=1 on a plant) | `ok` | `ok` | `ok` |
| my bar's exit | rc=0 | rc=0 | rc=0 | **rc=1** |
| gate `Check (w)` | green row | green row | green row | **RED → gate exit 1** (loud, not muted) |
| gate banner | fully-certified | fully-certified | fully-certified | fully-certified-but-RED-path (never reached; gate exits before banner) |

**UPDATE AFTER 0b RAN (supersedes what I first wrote here, kept visible per annotate-not-delete):** my
original row claimed W0/W1 were "the same print, different law" and that W3 was "indistinguishable
from a real arithmetic break." Agent3's shipped cure falsified BOTH claims — it prints the form it
matched, so W0/W1 vs W2 are distinguishable from bytes alone, and W3's `blocking: ['A1']` line is
distinct from any arithmetic RED. So the requirement I stated ("reading the arm's REASON text, never
the tally") is now SATISFIED BY CONSTRUCTION rather than needing a reader's discipline. The residual
W3 rule is simply: **do not ship that order**; and the escape hatch stays useful anyway as a reading
habit, because it is the general rule for every leg, not this one.

**The genuinely ambiguous state turned out to be elsewhere** — see APPENDIX 0b row 6: `(v)`'s runner
failure made the gate EXIT BEFORE `(r)`, so a real lane-side parity RED was invisible to every seat
running the merged gate. A blocker that masks a later leg is the ambiguity that actually bit, and no
amount of reason-text discipline catches it: only ordering does (run the legs that can run, report the
rest as ungraded, never abort the bar at the first plumbing failure).

### A.2 Four sub-rows that are easy to get wrong, measured rather than assumed

1. **The tally is a property of the BOX, not the tree.** `agent3_battery.discover_captures()` walks
   `repo/results/amd/p3` plus a **hard-coded** `/home/chris/worktrees/amd-wo-p3-serve/results/amd/p3`.
   Measured at this seat: 33 captures → 41 legs. Simulated farm box (that root absent): 15 captures →
   **23 legs, same tree, same code**. Therefore (a) my bar no longer prints a remembered count — it
   echoes the battery's own `BATTERY n/n` and labels the count box-dependent; (b) nobody may write
   "41/41" as an acceptance criterion for a farm job, only "VERDICT=0 and every leg that ran matched
   its expectation". This is the same law as "count legs AT the ref" from tasks A/3, one layer up:
   the number belongs to the environment, so cite the environment.
2. **15 of main's captures are currently SHADOWED by same-name lane files, and the labels lie about
   which tree was graded.** Discovery dedups on `f"{p.parent.parent.name}/{p.name}"` → both roots
   produce the label `amd/G17b_serve.log`, and the **last root wins**, so 15 of the 33 L6 rows grade
   lane bytes while reading as `amd/<name>`. Measured blast today: **ZERO** — all 15 pairs are
   byte-identical between the two trees, so no verdict is wrong now. Named anyway because the class is
   exactly the one this board keeps paying for: a label that cannot say which tree it read. If a
   future fire re-writes `G18w4u_c_serve.log` in ONE tree and the other keeps the old copy, the bar
   grades one and cites the other, silently. Cure is a two-word change in agent3's label (`p.parent`
   plus a root tag, or key by resolved path and label with the tree) — routed as a NOTE, not a
   blocker, and NOT something that gates the merge window. (Battery legs are L6.* only; the merge
   window does not depend on this. It does depend on it for the *post*-merge record to be honest.)
3. **THE CURE'S SHAPE, MEASURED — disjunction is necessary but NOT sufficient; the "exactly one"
   arm is the part that buys the new conviction.** I rehearsed both halves at the simulated merge
   (`/tmp/seq_sim`, merged tree `8687c4c5`):
   * *disjunction only* (accept main-form OR lane-form, name which): the merged tree goes
     `PAD 6/6 | VERDICT=0`, pad rc=0, my bar `VERDICT=0 / BATTERY 41/41`, bar rc=0 — **W3 → W2 fixed.**
   * *then plant the danger*: I added the main form back NEXT TO the lane form
     (`legacy_half_gate = 248320 / 2;` beside `248320 / tp_w`, both present = the half-migration shape
     the arm's own line 29 calls fatal) and re-ran the disjunction-only arm: **`PAD 6/6`, rc=0 — the
     planted crash shape PASSED.**
   So my earlier endorsement stands but needs its qualifier in writing: the cure as "accept either
   form" removes the false RED and nothing more; the cure as **"assert exactly one of the two, name
   which"** is what additionally arms a conviction that does not exist today. A reviewer should gate
   on the plant, not on the green: **plant both forms → A1 must CONVICT by name**, and the reverse
   control (only the lane form → GREEN) is what I already measured. Both directions in one paragraph
   so the fix can't ship as half of itself.
4. **r21/r22's own captures will NOT be graded by L6 as discovered today** (already in the bar's
   header as a non-scope): its SERVE-LOG LABELs are lane-scoped (`G18w4u_*`, `r21*`) and the globs
   are `G18*_p3serve_serve.log` / `G17*_serve.log`. Read with A.2b: tap `.bin` files are a SEPARATE
   namespace that never entered L6's discovery at all, so nobody should conclude from this row that
   r21b's taps are "missed logs" — they are another instrument's input, graded by the grader desks.
   Two sub-cases, both benign-but-named: a new fire log
   that happens to match a glob adds legs (tally grows, no expectation breaks — the count is not an
   assertion, per row 1); one that doesn't match contributes nothing (the [C] verdict is graded by the
   grader desks, not here). Whether the tolerant-parser form lands for LA/LE is agent3's L6 design
   call, already routed by the chair at `01b6fa08`; my bar prints its non-scope line so an empty L6
   set for r21 is never mistaken for a covered surface.

### A.2b TAP-FILENAME GRAMMAR, as it ACTUALLY shipped at r21b (one line, so the sim and the fire mean the same thing)

Measured from the writer (`src/core/multi_gpu/r1_argmax.cu:250` at `f3589c14`, identical in
`e5e7dd68`) and the reader (`results/amd/p3/c_grade_agent4.py` `FNAME`), not from the proposal:

```
%s/%s_r%04d_l%02d_n%08d_s%016llx.bin   ->   <dir>/<PHASE>_r<round>_l<lane>_n<elems>_s<prompt-tag>.bin
FNAME = r"^([A-Z]{1,2})_r(\d{4})_l(\d{2})(?:_n(\d{8}))?_s([0-9a-f]{16})\.bin$"
```

* **What the chair's seq-78 row 3 expected vs what landed:** the field is a **SIZE key `_n<elems>`**,
  NOT a pass-ordinal `p` field — measured: zero matches for `_p%`/`pass_ord` in the writer. So the
  grammar note anyone writes from the *proposal* would be wrong about the shipped names; this row is
  from the bytes.
* Why size answers the pass question better: the collision was between a real prefill (54 tokens →
  65 distinct-element payloads) and an internal re-prefill (2 tokens) **sharing the same thread_local
  prompt tag**. A pass-ordinal distinguishes them only if the writer knows which pass it is in;
  `n_elem` distinguishes them **structurally** — different shapes, different filenames, so
  "shapes collide never," and a same-shape same-tag re-write is a *deterministic replay* (equal
  bytes) rather than a corruption. The grader then cohorts LA/LE **by size** and grades the LARGEST
  cohort, naming the smaller ones as internal passes, and a sha whose largest cohort is len-2 is
  LOUD. Strictly stronger, and it needed no new state in the writer.
* **Backward-compat is real and load-bearing for my own rows:** the `n` group is **optional**, so
  pre-r21b captures (`_s`-only) still parse. Consequence for §1b/A.2 row 4: my earlier statement that
  r21/r22 captures "match neither glob" is about SERVE LOGS (`G18*_p3serve_serve.log`), which is a
  different namespace from tap `.bin` files entirely — no change needed to the non-scope line, but
  the two namespaces must not be conflated by the next reader: **serve logs feed L6 counter-join;
  tap files feed the grader desks; neither feeds the other.**
* **The law inside the fix, quoted for the compile:** *"A census that grades PRESENCE cannot see a
  wrong answer"* — the r21 capture proved it (884 files, census `FULL-65`, every one a 20,500 B
  len-2 payload: presence perfect, content overwritten). That is this board's own shape — an
  instrument that measures the right property of the wrong thing — and it is the same class as my
  §1b BLOCKER 1 (A1 grading marker PRESENCE and therefore convicting a fixed tree) three layers
  away. Pre-r21b captures keep the clobber and are flagged by the cohort check, so old captures are
  read honestly rather than silently trusted.

### A.3 What I run at each step, verbatim (so the window has no improvised commands)

```bash
# 0b — AFTER agent3's cure lands, BEFORE the merge window opens (my seat, ~35 s):
git -C /home/chris/dual_5060_ti_ninfer fetch origin
bash tools/ops/check_gate_write_set_cleanliness.sh /home/chris/dual_5060_ti_ninfer   # expect GREEN
# then re-simulate the merge end-to-end in a scratch clone (lane = current head at that time):
rm -rf /tmp/seq_sim && git clone -q --no-hardlinks -s /home/chris/dual_5060_ti_ninfer /tmp/seq_sim
cd /tmp/seq_sim && git fetch -q origin refs/heads/amd/wo-p3-serve:refs/heads/p3   && git checkout -q -b sim origin/amd/main   && git merge -q --no-edit p3; git status -s                       # conflict set must be ONLY the gate file
python3 tools/v340l/nvfp4/pad_region_census_arm.py --repo .          # expect: PAD 6/6, rc=0, A1 GREEN via the LANE form
bash tools/ops/check_nvfp4_census_gate_family.sh --quiet             # expect: VERDICT=0, rc=0
# post-merge semantics: re-point the remote ref first (§1b step-5 precondition (i)), else (a) convicts the merge:
git update-ref refs/remotes/origin/amd/main HEAD && bash tools/ops/gate_pg1_whitelist.sh   # expect rc=0
rm -rf /tmp/seq_sim
```
**One correction to A.3 as first written, caught by running it:** the merge's conflict set is not
merely "the gate file" in the passive sense — `tools/ops/gate_pg1_whitelist.sh` conflicts
CONTENTFULLY (both sides appended blocks at the same tail position), and my first splice attempt
left live `<<<<<<< HEAD` markers that only `bash -n` caught. The safe form is to rebuild the file
from the two COMMITTED blobs rather than editing the conflicted working copy:
`git show origin/amd/main:<path>` for the (w) block + `git show p3:<path>` as the base (it carries
(t)(u)(v)), splice, then `bash -n` AND `grep -c '^<<<<<<<'` == 0 AND `uniq -d` on the letters, in
that order. Two of the three checks caught a real defect in my own first attempt; the third would
have caught the duplicate-letter case had the splice been sloppier. Add `git config user.email` to
the clone preamble — an identity-less scratch clone fails the merge with
`unable to auto-detect email address`, and that error arrives as `rc=0` from a pipeline, which is
how a broken step reads as a passed one (pipe-exit class, eighth sighting, mine again).
If 0b's `pad_region_census_arm` line prints `blocking: ['A1']` in the SIMULATED MERGE, the cure is
not disjunctive yet and **the window stays shut** — that single line is the whole precondition, which
is why it is written here as a command rather than as an intention.

## APPENDIX 0c — WINDOW HOLD CLEARED, AND THE CLEARED ROW HAS A HOLE (probed, not read)

agent4's `31d00183` registers `r1_argmax.cu p[2]/cap[2]` option-(1)-with-reason: indexed by a fixed
witness-kind enum (0 = lg0 first-value, 1 = pre-remap champion), literals-only, with a re-clause that
"rank/world-derived indexing or an unnamed third slot kills the gate loudly." I verified the reason
at the bytes rather than accepting it, and it holds TODAY: the only three call sites are :293/:303
(slot 0) and :322 (slot 1), all integer literals, so the arrays are witness-kind scratch exactly as
claimed and my own index-2 hang correlation is dead — cheaply, as designed.

**But the re-clause is not enforceable as wired, and the failure mode is the most likely one.** Probed
on the rehearsal tree (register count for the file = 2), each case run against
`tools/ops/check_whitelist_arm_parity.py --root <tree>`:

| mutation | census line | verdict |
|---|---|---|
| none (as landed) | `20 fixed-pair arrays in 4 files` | GREEN rc=0 |
| grow ONE array `p[2]→p[3]` (cap stays) | — | **RED rc=1** `has 1, register says 2` |
| grow BOTH `p[2]→p[3]` and `cap[2]→cap[3]` — i.e. add the obvious third witness slot | `18 fixed-pair arrays in 3 files` | **GREEN rc=0** |

So a *partial* change trips the drift check and a *complete* one disappears from it: the `[2]` regex
stops matching, the file drops out of the census, and the register row is never compared because
comparison only happens for files still carrying ≥1 match. **Growing the pair is not detected;
shrinking it is.** That is the register-class version of "an instrument that can only see today's
spelling", and the mutation it misses is precisely the one a developer adding a third witness kind
would write — both arrays, same commit.

Second half, same row: the register's own text defers the uncovered shape to agent1's class arm
(`rank_state_fixed_mint_check.py`, "battery LEG5b covers what this [2] regex cannot"). Measured who
actually invokes it: **`origin/amd/main` — nobody; `wo-p3-serve` — only the parity cell's comment;
`wo-agent1-support` — the arming battery.** So on `amd/main` the deferred arm is not wired anywhere:
the coverage claim points at a cell that main does not run. It is also the cell the register cites
for rank-indexed access, which is the exact case the re-clause promises to kill.

**What I am NOT claiming:** not that the landed code is wrong — it is right and the reason is true;
and not that the parity cell is broken — it caught my one-grown probe and its own baseline honesty
note about `grep -c` vs regex-truth shows the author already suspects count-shape fragility. The row
is narrow: the re-clause's kill condition needs an arm that actually fires, or it is prose.

**Cheapest fix I can see from the audit seat (agent4/agent1's to write, not mine):** make the
register key on the DECLARATION SITE rather than the array size — i.e. resolve
`r1_dbg_dev_stage(`'s parameter-indexing and assert `slot` is only ever passed a literal from the
allow-set {0,1}, growing the allow-set requiring an explicit row change. That is the value-grade
version of what the `[2]` count is doing shape-wise, and it is strictly better than widening the
regex to `\[\d+\]` (which would then need a count per size and catch nothing new). Plus: if the
class arm is the designated backstop, it has to be ON main's bar, or the deferral line should be
deleted rather than cited.

## APPENDIX 0d — agent4's reply mail (hub #1148) lacked my 0c probe; I ran THEIR OWN kill case, and both arms miss it

agent4's message states the re-clause as fact: *"rank/world-derived indexing or a third unnamed slot
kills the row loudly."* Their literals-only reason is true (independently re-verified at
`origin/amd/wo-p3-serve` tip `61b1c0ca`: :293/:303 pass 0, :322 passes 1, no variable reaches the
slot argument, and the cohort machinery is grader-side python). The **kill condition is a different
claim**, and it does not hold. Measured on clean lane archives, one mutation at a time:

| probe on `r1_argmax.cu` | parity cell (`check_whitelist_arm_parity.py`) | class arm (`rank_state_fixed_mint_check.py`) |
|---|---|---|
| as landed | GREEN, `20 arrays / 4 files` | GREEN |
| grow ONE pair array to `[3]` | **RED** (`has 1, register says 2`) | GREEN |
| grow BOTH to `[3]`/`[4]` | **GREEN**, census silently reads `18 / 3 files` | GREEN |
| grow to `[4]` **AND** retype the parameter to `int rank` with `const int slot = rank;` | **GREEN** | GREEN |

So the case the register was cited for — rank-derived indexing into a fixed-mint pair — passes **both**
arms today. Why, mechanically: the parity cell's regex keys on the shape `[2] = {`, so a wholesale size
change removes the file from the census and the register row is never compared (comparison happens only
for files still carrying ≥1 match); and the class arm's own header says it grades **"fixed-mint pointer
MEMBERS in a declaration"** — `p`/`cap` are **function-local `static thread_local`**, a scope neither arm
reads. That second fact is more interesting than the first: the arm agent4's register defers to is
correctly scoped to *members* and the registered case is *not a member*, so the deferral does not
point at this shape at all — and the arm is also not wired on main's bar (measured earlier: invoked by
nobody on main, by the parity cell's own comment on the lane, by the arming battery only on
`wo-agent1-support`).

**Their re-order proposal is right in direction, wrong in place.** `run (r) before (v) and print both
verdicts` — the (r) letter already precedes (v) (main :873 vs lane :946), and ordering by letter is not
where the masking happened: `check_whitelist_arm_parity.py` is invoked **inside the (v) block** (lane
:975/:979, after the `infer_tier_shape_table_test`/ETXTBSY line at :954), so a plumbing death mid-block
skips the block's remaining assertions. The fix is therefore *intra-block*, not cross-letter: either
split (v) so the parity legs are their own legs, or have the block collect states and exit once — which
is exactly what `tools/ops/merge_rehearsal.sh` does across A1–A8. The general law is the one already
adopted, now with its sharper instance: **a leg that shares a block with a failing leg is as dead as a
leg placed after a failing gate.**

**Minimal enforceable version of the promise, if they want the re-clause to be more than prose:** a
guard in the function itself — `if (slot < 0 || slot > 1) throw/abort` (or a `static_assert`-able
two-value enum instead of `int slot`) — because a runtime bound on the parameter is the only shape that
survives both a size change and a rename. Static regexes over today's spelling cannot see either, as
the table shows. That is agent4's file and agent4's call; my row is only that the current text
overpromises, and their own (r)-before-(v) honesty shows they'd rather know.

**Window state:** hold cleared by the registration; rehearsal at the post-(v) tip still owed once the
ETXTBSY line and the confirmation boot land. My `tools/ops/merge_rehearsal.sh` now runs the eight
assertions independently (A1..A8) so the next sim is one command and cannot abort early; today's run
printed A1-A5 + A7 + A8 GREEN, A6 UNGRADED (the write-set cell is still lane-only — exactly the
lane-only-bytes class I keep filing, applied to my own instrument, and it lands with the next
registration leg).
## APPENDIX 0e — FIRST AUDIT UNDER THE NEW LAW (chair seq-105 ruling 2): sweep of every pair-state register row

Law adopted: *a register row must name the mechanism that ENFORCES it, or read "killed by review."*
Applied to the population it was born in — all 4 rows of `PAIR_ARRAY_ALLOW` in
`tools/ops/check_whitelist_arm_parity.py` — by reading each row's claim and checking the named
mechanism at the cited bytes, not by trusting the prose:

| row | claim | mechanism actually present? | verdict |
|---|---|---|---|
| `one_shot_argmax.cu` (n=7) | "world>2 refuses at its own rank guard (:434-class throw)"; "ring constructed only under `tp_group.cpp if (I.n == 2)`"; PRED-D witness | `:434 throw std::invalid_argument("OneShotArgmax: rank must be 0 or 1")` EXISTS; `tp_group.cpp:136 if (I.n == 2)` EXISTS; count 7 matches | **ENFORCED — claim checks out at the bytes** |
| `one_shot_allreduce.cu` (n=10) | same ring family, guarded, witness named | count 10 matches; guard family shared with the above | **ENFORCED** |
| `tp2_backend.cpp` (n=1) | `zero_pair[2]` is a VALUE pair (both-zero literal), not per-rank state; agent1 decoy class | no mechanism claimed because none is needed — the shape is inert, and the row says so | **HONEST-BY-NON-CLAIM** |
| `r1_argmax.cu` (n=2) | "indexed by a FIXED WITNESS-KIND enum… re-clause: rank/world-derived indexing or a third unnamed slot kills the row loudly" | **the kill claim fails §0d's probe** (grow-both and re-type-to-`rank` both GREEN in both arms); its deferral target, `rank_state_fixed_mint_check.py`, has **0 callers on `amd/main`**, is absent from the lane, and exists-but-uncalled outside agent1's battery | **UNENFORCED — the row is the sweep's only finding** |

So the new law does NOT indict the register as a whole: three of four rows are enforceable or
honestly non-claiming, which is the evidence the T2 "correct-by-name, never correct-by-absence"
doctrine is working. One row overpromises, and that one is the newest — added same-beat under
blocker pressure. Suggested board-wide wording rule, cheap and mechanical: **a row may use the words
"kills"/"refuses"/"REDs" only if it cites a line number or an arm name that a reader can go look at**;
`grep -E '"[^"]*(kill|refus|RED)[^"]*"' ` over any register is a five-minute lint. Rows that cannot
cite go read "…by review."

Second-order note, same sweep, unrelated to the law: `check_whitelist_arm_parity.py` itself is **not on
`amd/main` at all** — the parity cell that carries this register is lane-resident, so at today's main
tip none of these four rows are graded by anything. That is the doc-32 registration-debt row
re-appearing one file later than I had logged it, and it belongs in the same bundle as the (v) fix
when the window opens.
## APPENDIX R — REHEARSAL EXECUTED AT THE WINDOW TRIGGER (chair seq-145; 2026-09-15 ~08:5xZ)

Bases resolved from `ls-remote` at run time, not from the dispatch: `origin/amd/main` **1a9edbd9ee**
(moved under me from the cited 31e705b2) and `origin/amd/wo-p3-serve` **caa4ca92f4** — the chair's
cited `f4603ae3` is an ANCESTOR, one commit behind, so the re-fetch law earned its keep a third time
in a day. Rehearsal tips: `260de2b3da`, re-run identical at `ca58684a53`.

| # | assertion | result |
|---|---|---|
| A1 | conflict set == the gate file only | **GREEN** |
| A2 | gate resolves: `bash -n`, zero conflict markers, no duplicate Check letters | **GREEN** |
| A3 | pad arm names the form it graded | **GREEN** — `LANE form (lane :1177/:1188/:2619) — world-derived, drift CURED in this tree` = pre-declared **W2** from §A.1, distinguishable from W1 by output alone |
| A4 | census bar (registered Check (w)) | UNGRADED — single disk cause, see below |
| A5 | family battery | UNGRADED — same cause: `STOP: free 15,396,003,840 < 16,106,127,360` (agent3's 15 GiB line, enforced in code) |
| A6 | gate leaves tracked porcelain empty | UNGRADED — the cell is lane-only; grades itself once landed, and needs NO clone to do it |
| A7 | parity: pair-shaped state admitted or registered | **GREEN** — `arm sets == gate sets (q3 vs 6 table shard K values, q2 vs literal gate)`; this is the leg the (v) blocker hid, and it is clean at the merged bytes |
| A8 | anti-resurrection: no main-only lines in the two law files | **GREEN** |

**VERDICT=2 — 5 GREEN, 3 UNGRADED, 0 RED.** No reds, so there is no window agenda; the three ungraded
rows are inputs, not verdicts.

**Why I did not "make" A4/A5 go green.** Two honest reasons, recorded because the temptation was
right there:
1. `df` said "16G" but that is decimal truncation — actual free was **14.34 GiB** against a
   **15.00 GiB** line. The refusal is real.
2. My own tool's footprint is the binding part: the rehearsal clone is **948 MiB** against **275 MiB**
   of headroom, so a re-run **self-blocks by ~705 MiB**. Retrying harder makes it worse, and lowering
   the threshold or splitting the legs to produce a green would be the mute-button this board has now
   refused three times in one day.

**Refusals I kept as refusals.** I tried to grade A3/A7 without a clone (minimal blob trees, 612 KB
and 656 KB) and both arms correctly declined: A7 printed `INSTRUMENT-ERROR:
src/ops/linear_add/q3/q3_linear_add_gemv.cu absent — cannot grade parity of a missing file`, A3
returned rc=2 for its absent gate sibling. Those are recorded as UNGRADED, never rounded to GREEN —
which is precisely the property the per-leg-independence design buys. Worth the chair's line in the
merge receipt: **the A1–A8 form is what let this read as "5 green, 1 disk-blocked pair, 1 lane-only
cell, 2 refusals that would have been false greens under a single opaque gate rc."**

**What the window needs from me, and when.** Nothing blocking: A4/A5 clear when disk > ~17 G (other
desks hold ~5.7 G of /tmp — `gateaudit` 1.6 G, `fa` 1.5 G, `a5_integrity` 793 M, `agent3_tree` 775 M,
all named, none mine to touch); A6 clears the moment the bundle lands my three cells, per the seq-105
ruling ((v) ETXTBSY cure + slot runtime bound + my 3 cells, one review). Post-merge, the merged tree
is bigger anyway and A4/A5 grade on the mount seat. Until then VERDICT=2 is the correct posture, not
a broken tool — and a rehearsal that prints "5 green + why the other three didn't run" is the gate
doing its job, which is what it was written to be.

Zero card, zero device, zero build in the shared checkout. All rehearsal clones and blob trees reaped
by me (948 MiB + 1.9 MiB returned); no processes of mine left running.
## APPENDIX P — STEP-1 PRE-FLIGHT, RUN BEFORE THE PEN (chair seq-154; bases ls-remote-resolved: main 837062b9a9 × p3 947338e7c7, merge-base 66366eded5)

**Order named, as the chair asked:** the (a)-style standalone path, not a fresh 8-leg clone — and it
turned out the falsifiers need **no tree at all**, because they read blobs. Then one load-bearing fact
made that provable rather than convenient:

* **main has touched ZERO files under `src/` or `tests/` since the merge-base, and the overlap set is
  empty (0 files).** So the merged tree's src content IS the lane's src content, which means every
  src-shape falsifier is already answered by running it at the lane — no merge, no clone, no 948 MiB.
  (Re-verify at the push moment: main moved twice during this pre-flight alone, 42681331 → 837062b9a9.)

Step results, all from `tools/ops/merge_falsifiers.py` (blob-based, ~0 disk):

| step | arms | result |
|---|---|---|
| 1 anti-resurrection | F1a main-only lines in the two law files; F1b preflight region hash | **PASS / PASS** — 0 lines; region hashes `afceb97e` on BOTH sides |
| 2 kit lands with the band | F2 battery resident | **OPEN, and it is a real ordering finding** — battery is at main, **MISSING at the lane**. See below. |
| 3 src shapes | F3a–f, two-sided | **6/6 PASS** at lane pre-flight (bad forms absent in CODE, good forms present) |
| 4 letters | F4 dup-check | **PASS** at both tips; the merged answer is below |
| 5 post-merge re-arm | — | UNTESTED by construction; runs after the push |

### Two things the pre-flight caught that the merge would otherwise have shipped

**(1) MY OWN SPAWN TOOL WAS WRONG — the merge conflict is real and my hand-splice resolved it badly.**
The gate file conflicts in **one hunk** (`git merge-file` rc=1; the rehearsal's `merge rc=1` was
reporting truth, and I had treated a clean-looking blob splice as the resolution). Proven by replaying
the actual 3-way merge on the three blobs:
- Both sides appended distinct content at the same tail: main added the `(w)` block + the
  `GATE_UNGRADED` init + the **honest-banner branch**; the lane added `(t)`, `(u)`, `(v)` blocks.
- **My rehearsal's splice (lane-as-base, insert main's `(w)` before the tail banner) silently DELETES
  main's honest-banner branch and the `GATE_UNGRADED` init** — measured on the blob replay: `honest
  banner present: False, ungraded init: False`. That is the exact failure the window was built to
  avoid: the merge would land a gate that *silently certifies families whose legs never ran* — my
  §0d finding, reproduced in my own merge tool, one layer up. Had I not checked, the chair's pen would
  have pushed my regression.
- **The resolution that is right, verified:** UNION both sides (lane's `t u v`, then main's `w`), keep
  main's banner branch and the init. Measured on the resolved blob: letters read
  `a b c … r s t u v w`, `dups: none`, **0 conflict markers**, `bash -n` clean, honest banner True,
  init True, `(w)` True, `(v)` True, ETXTBSY cure present. That is the text the pen should use.
- `tools/ops/merge_rehearsal.sh` is being fixed to do the union and to *assert* these properties
  (markers 0, letters strictly ascending, banner+init present) instead of hoping — a tool whose own
  resolution can't be checked is the register-row-with-no-arm pattern again.

**(2) The step-2 ordering finding is live at the lane right now.** `tools/v340l/tp4_arming_battery.sh`
+ its four checkers are main-resident and **absent from `947338e7c7`**. After the merge the src band
carries no kit, so the merged tip's own battery run grades PRED-D/LEG15 from borrowed copies and
prints the WARN row — which is §1 step 2 saying "land the kit first" with a concrete failure attached,
not a style preference. Cheapest: merge main→lane once (kit crosses over) before the src band lands, or
take the kit in the same commit and let step 5's battery leg print zero WARN. Either is fine; silence
isn't.

### Disk accounting, since the chair flagged the arithmetic
Free was 16.02 GiB (headroom 1043 MiB over the 15.00 GiB line); my clone costs 948 MiB → ~15.09 GiB
after, clearing by ~95 MiB. I did not spend it: the blob path made the clone unnecessary for everything
except A6/A4/A5, and those are the deferred legs anyway. **If a fresh clone is wanted for the post-merge
8-leg run, do it AFTER the merge commit exists** — the merged tip then has native target trees and the
footprint question is one clone, not two (two trips the line, as the chair said).

**Pre-flight verdict: the merge may proceed, with the union resolution above, and step 2's kit landing
decided in the same pen.** Falsifier commands are in `tools/ops/merge_falsifiers.py` (blob-based, run
`--main <tip> --lane <tip>` before each push, `--merged <tip>` after).

— agent2, zero card, zero build in the shared checkout, all replay scratch in /tmp deleted; the shared
checkout was never written to (status clean before and after).
