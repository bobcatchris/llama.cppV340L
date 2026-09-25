# G-AMD-18 WINDOW RUNBOOK — the run-of-show for the one 4-card window (agent1, WO-SUPPORT-1 Task A)

**Desk**: agent1 (`amd/wo-agent1-support` @ `5657a174`) · **Issued**: 2026-09-13 ~15:3xZ
**Baseline for every claim below**: `amd/main` @ `dfd5b7ec` unless a ref is named. Refs age at merge
speed (runbook §"what not to trust"); **re-derive the two PREDICATE greps in §0 before booking anything.**

## 0. THE HEADLINE: the window is not the binding constraint — two code gates are

The WO's mission line says "compress the path to TP4's 4-card first-light". Measured at this desk, the
path is blocked by **code, not cards**. Cards are free right now (measured 15:1xZ, `rocm-smi
--showpids` → **0 KFD lines**; `--showmeminfo vram` → dev1/2/3 each 8,314,880 B used, dev0 148,250,624 B
used = the desktop). Both gates below are **pre-flight checks in this runbook** and both are §2-step-0
material: firing the window without them spends device minutes to reproduce a throw.

> **CLOSED ANNOTATION 2026-09-14 ~00:5xZ (chair seq-26 assignment 3, annotate-never-delete): BOTH code
> gates below are CLOSED and CERTIFIED at the flip tip — the §0 headline's "blocked by code" state is
> HISTORY, kept because the predicates are still the window's step-0 arms.** GATE 1 closed by A-4's guard
> narrowing (src merged `6b98623f`) delivered CO-LANDED with GATE 2's routing arm at the roster+flip merge
> **`83d1b00e`** (chair ff'd; re-verified by me at announced tip `98923fda` and again at boot tip
> **`92564abb`** — see `results/amd/G18_FLIPRUN_2026-09-14_agent1.md` postscript and
> `results/amd/p3/G18_PREFLIGHT_2026-09-14_agent1.md` §2). The pair-test the headline demanded — "guard
> gone + routing silent = the silent-null first-light this runbook exists to prevent" — passed with
> dual-parser GREEN/GREEN at every tip since the flip (my `t3_gate2` rc=0 + Gemini's check (q) rc=0,
> `ENFORCE_A4_ROUTING` default-ON at `gate_pg1_whitelist.sh:850`). GATE-2's silent-null is formally dead:
> `allreduce_argmax` now falls back to `require_argmax_transport()` at `tp_group.cpp:307` — a loud throw,
> not a null return. PREDICATE-1 prints GUARD LIVE in its NEW rank-COUNT form (`tp_engine.cpp:728`,
> `if (tp_world < 2)`), which is the expected post-flip shape, not the old refusal.

**GATE 1 — the engine refuses world≠2 (capability gate, fails loud, harmless).**

> **CLOSED @ `6b98623f` (src) + `83d1b00e` (flip co-land), certified at boot tip `92564abb`.** Everything
> in this block below describes the PRE-flip state and is retained as the predicate's provenance; the live
> re-derivation prints the rank-COUNT form cited in the annotation above.
`src/runtime/tp2/tp_engine.cpp:718-723` at `dfd5b7ec`:

    if (tp_world != 2) {
        throw std::invalid_argument(
            "TPEngine: world=" + std::to_string(tp_world) + " requested; only world==2 is "
            "constructible until WO-TP4 A-3 (TpGroup rank loop) + A-4 land — refusing before any "
            "device touch, this is a CAPABILITY gate, not a capacity verdict.");

Verified present at **all three** live tips (`origin/amd/main`, `origin/amd/wo-p3-serve`,
`origin/wo/tp4-gates-d` — the guard string present once each, and `PREDICATE-1` below confirms it is the
live throw rather than a mention). A-3 *has* landed (merged at `516883da`);
**A-4's code has not**: its inventory commit `4d8bfe6e` is NOT an ancestor of `amd/main`
(`git merge-base --is-ancestor 4d8bfe6e HEAD` → false, checked 15:2xZ). So the guard's own text names
the missing prerequisite. Re-derive with a predicate that cannot be satisfied by a comment — counting the
string is not the same as proving the guard is live code, and a future edit that deletes the throw while
leaving a comment mentioning `tp_world != 2` would still print 1 (agent2's `grep -l` mention-vs-call
lesson, #560, applied to my own predicate):

    PREDICATE-1 = grep -nA3 "if (tp_world != 2) {" src/runtime/tp2/tp_engine.cpp \
                  | grep -q "invalid_argument" && echo "GUARD LIVE" || echo "GUARD ABSENT OR INERT"

which asserts the shape (`if (`) **and** its consequence (the throw) rather than the presence of a token.

**GATE 2 — the trap: at world=4, greedy token reduction silently does NOTHING.**

> **CLOSED @ `83d1b00e` co-land — the silent null-path is dead.** `allreduce_argmax` (`tp_group.cpp:307`)
> now ends in `require_argmax_transport()` instead of returning having written nothing; both independent
> parsers certify the routing arm at every tip since the flip (FLIPRUN F2/F3/F4, pre-flight §2). The R2
> interim throw is honest about its successor in-file (ARM 6: an R1/RCCL arm may replace it — the refusal
> can never quietly outlive its reason). If a window boot ever MEETS that throw it is a CAPABILITY
> classification, never a capacity verdict. The text below stays as the original finding — it is why the
> window almost didn't happen, and why PREDICATE-2 is decision-shaped.
Not a throw. `src/core/multi_gpu/tp_group.cpp:109-112` constructs the fused samplers **only for n==2**:

    if (I.n == 2) {
        I.one_shot = std::make_unique<OneShotAllReduce>();
        I.one_shot_argmax = std::make_unique<OneShotArgmax>();
    }

and `TpGroup::allreduce_argmax` (`:297-302`) is guarded by a null-check with **no else**:

    void TpGroup::allreduce_argmax(int rank, const Tensor& logits, Tensor& out_token, ...) {
        if (impl_->one_shot_argmax) { impl_->one_shot_argmax->allreduce_argmax(...); }
    }

Consequence at world=4: `out_token` is **never written**, on any rank, with **zero diagnostics** —
whereas `allreduce_local_bf16` (`:266-272`) DOES fall back to `ncclAllReduce` when `one_shot` is null.
So the AR transport generalizes to 4 and the **argmax transport does not**. If A-4 relaxes GATE 1 without
closing GATE 2, first-light would print tokens sampled from whatever the tensor happened to hold —
greedy, coherent-looking, and completely unverified. That is precisely the fail-loud violation the board's
own laws target, and it is cheap to fix (throw naming world, or an RCCL allgather+local-argmax arm), so it
belongs in **A-4's scope, decided before the window is booked**, not discovered inside the window.
Re-derive with `PREDICATE-2`, which must be **decision-shaped, not a count**. My first version was
`grep -n "if (impl_->one_shot_argmax)"` and that returns **4 matches** at current main — `:292` (step
bookkeeping), `:299` (the collective), `:319`/`:324` (float accessors) — so a reader cannot tell pass from
fail from it, which is the chair's citation law (#482(4): a published command ships with an expected
outcome) applied to my own doc. The runnable form, with its expectation stated:

    PREDICATE-2 = tools/v340l/t3_gate2_silent_nullpath_check.py <tree>
    EXPECTED TODAY (rc=1):  "RED  TpGroup::allreduce_argmax @ ...:297 — returns having done NOTHING"
                            "VERDICT: RED — silent null-path collective(s): allreduce_argmax"
    EXPECTED WHEN CLOSED (rc=0): "VERDICT: GREEN"  -> and then promote the cell to a gate (see the
                            TRACKED-RED cell's own exit-11 rule)
    (selftest: `... --selftest` must print SELFTEST: PASS rc=0; it executes the blind-parse and RED legs,
     so the checker itself cannot be vacuous.)


*Scope note (WO §3, honored): I support, I do not adjudicate or patch. GATE 2 is a **precondition
finding with a citation**, filed for agent4's A-4 and the chair's window decision. No `src/` edit from
this desk; no patch proposal attached.*

## 1. Preconditions the chair must see satisfied in writing (all CPU-side, zero card time)

| # | precondition | how to verify at the keyboard | closes |
|---|---|---|---|
| P1 | A-4 code merged to `amd/main` | `git merge-base --is-ancestor <A-4 sha> origin/amd/main` | GATE 1 |
| P2 | argmax transport world>2 exists or throws | `PREDICATE-2` above | GATE 2 |
| P3 | item-7 prereq (P1 of WO-TP4) — **ADJUDICATED, and the answer is "per-request nondeterminism is LIVE"** (`1d0ff3c6`) | read the row | bars: `same-token` is NOT a valid 4-card gate — first-light must be judged **coherent + zero-fault + arms-traced + cycle class**, per the verdict row's own "BOARD CONSEQUENCE" paragraph |
| P4 | artifact | `ls -l $(cat results/amd/p3/Q3_CANONICAL_PATH)` = 15,446,796,288 B (verified 15:1xZ, EMTEC256 local ext4) | truncated-desktop-copy class (runbook §3) |
| P5 | disk ≥ one build slot | `df -h /` — **15 G measured 15:1xZ**, WO §8 said 16 G; a build is ~5 G+ | runbook law; §2-step-1 owns the number |

## 1b. SEQUENCING HAZARD — the anti-resurrection gate prints a FALSE RED on A-4's required edit

> **RESOLVED-AS-PREDICTED ANNOTATION 2026-09-14 (annotate-never-delete): this hazard was foretold, fired
> exactly as written, and is now a BANKED TRANSCRIPT — not a wave.** The predicted false-RED was captured
> deterministically at the flip tip and banked by the chair at **`ad069459`** with the exhibit
> `results/amd/p3/G35_antires_leg2_predictedRED_83d1b00e.txt` (cross-diff gate rc=1 at `83d1b00e` vs
> baseline `393ce73d`; the single FAIL is A-4's 4 guard-replace lines tripping the any-minus-line
> predicate; leg-1's banned-constant sweep prints ZERO FAILs — the three visible names are tombstone
> comments at `:1156/:1162/:55`, legal by the script's own doc). My F6 overruling receipts (FLIPRUN §F6)
> match the transcript line-for-line — foretold at one desk, caught in the act at another. The
> **permanent decision-aware-leg2 fix is FILED POST-WINDOW to Gemini's tools surface with its full
> RED→GREEN→falsifier triple pre-written** (RED = that transcript; GREEN = same command rc=0 post-fix;
> falsifier = a real reverted constant must still print RED) — see the filing paragraph at
> `ad069459:results/.../G35_antires_leg2_predictedRED_83d1b00e.txt:56`. Until that fix lands, every
> future cross-baseline run across `6b98623f` inherits this RED and the F6 overruling form (leg-1
> predicate cited beside leg-2 RED) is the receipt, never a raw wave — which is precisely the law-decay
> this section was written to catch, now armed with its own precedent.
>
> **UPGRADE 2026-09-14 01:1xZ (chair seq-30): the hazard is now FIXED, not waived.** Gemini's
> decision-aware leg-2 triple merged at **`e2044290`** (`342889a8`): the banked transcript above is the
> cell's own cited RED leg, and the command that produced it (`--baseline 393ce73d`) now prints rc=0 with
> the four guard-replace lines classified benign BY MACHINE — 5/5 arms incl. mutation discrimination
> (benign rc=0 / genuine revert rc=1 / rogue `kGateReliefBytes` revival rc=1), re-measured at my seat on
> the merge forward of this row. The law-wording conflict this section ended in is resolved in the
> instrument's favor: leg 2 now predicates on DECISION content, so the cross-baseline RED every future run
> inherited is dead. Cell caveat banked in the pre-flight row's postscript: it mutates `tp_engine.cpp` in
> place and is NOT re-entrant — never run it concurrently with PG-1 in one tree.

Found while checking agent3's claim (`#601`(C)) that `tp_engine.cpp` is "the one seam file that isn't
wholly unguarded". That claim is **true, and it has a consequence nobody has stated**: the guard it cites
makes A-4's own required change illegal in CI.

`tools/ops/check_anti_resurrection.sh`, production leg 2 (~`:139-152`):

    REVERTED_LINES=$(git diff -U0 "${MERGE_BASE}..HEAD" -- "$f" | grep -E '^\-[^\-]' || true)
    if [[ ${REVERTED_COUNT} -gt 0 ]]; then echo "FAIL: ${REVERTED_COUNT} reverted line(s) detected in ${f} ..."

`$f` includes `src/runtime/tp2/tp_engine.cpp` (`:58`). The predicate is **any `-` line in the lane's own
diff**, which conflates "reverted main's newer content" (what AGENTS.md bans) with "edited the file"
(what A-4 must do). Measured at this desk, zero repo writes:

| commit (all three **merged** into `amd/main`) | lines removed from `tp_engine.cpp` | leg-2 verdict |
|---|---|---|
| `9ce12070` (WO-VRAM-1 fix) | 43 | FAIL |
| `c4b30bf3` (A-2 engine device-vector) | 35 | FAIL |
| `e611c398` (A-3 rank loop) | 2 | FAIL |

And the minimal relaxation of GATE 1 — a **one-character** edit `if (tp_world != 2)` → `if (tp_world < 2)` —
yields exactly 1 removed line + 1 added line (proved by blob diff under `/tmp`, not asserted). So a correct
A-4 commit that narrows GATE 1 is a guaranteed leg-2 RED.

**Boundary, stated narrowly:** I verified the *predicate*, not that it has ever fired. Those three merges
passed, and the chair's own merge message records a different, narrower hand-run check ("main's
tp_engine/tp2_budget region UNMOVED … 0-line diff, chair-run"); leg 2 also goes blind whenever a lane
merges `main` forward first, since the merge-base then advances past the lane's own edits. So the likely
outcome is not a blocked merge but a **false RED that a diligent successor must overrule by prose** —
which is worse than a block, because it trains the board to wave the step-0 gate past `tp_engine.cpp`
edits, precisely where the VRAM-LAW anti-resurrection check is meant to be load-bearing. A false green is
a defect; a false RED on the protected file is one too.

**Not mine to fix** (WO §3: no `src/`/CI edits from this desk; support-not-adjudicate). Re-derive:
`sed -n '55,160p' tools/ops/check_anti_resurrection.sh`.

**Sharpened by an in-tree precedent — and my first recommendation was wrong, refuted by my own test.**
Checked `tools/ops/verify_registered_exception.py` (Gemini's Ruling-1 containment verifier, the sibling
instrument agent4 cross-checked at `feb3d617`): it predicates **only on added lines** (`:263-265`),
never on removals — which is why it PASSed a 26-line `tp_engine.cpp` hunk that leg 2 fails — and it
already ships three-state exits (`:29-31`, `PASS=0/FAIL=1/INSTRUMENT-ERROR=2`, error reachable by
construction at `:127`). I first wrote "adopt a presence-assertion: does the baseline's canonical content
still appear?" and then **tested it against the very case at issue**: after the legitimate A-4 relaxation,
`if (tp_world != 2) {` is present? **NO — legitimately deleted**. So presence-assertion false-REDs too.

The sound predicate is therefore leg 1's **named-content** form, and that is also what the ban actually
says: AGENTS.md prohibits "estimate-based refusal constants … fixed budgets, prefix/ws reserve terms,
'safety' multipliers, slack terms" — a *class of constants*, enumerated by name (`VRAM_SAFETY_MARGIN`,
`kGateReliefBytes`, `headroom_bytes`, `16310ULL * 1024`, …), all of which stay detectable however the
code around them is edited. Any *diff-shape* test — removed-line counting, presence assertions — cannot
separate a resurrection from a legitimate edit, because resurrection and editing are the same text
operation viewed from opposite ends.

**Which relocates the conflict upward, and that is the actionable part.** `AGENTS.md:94-95` states the law
as "`git diff main -- … tp_engine.cpp tp2_budget.h` must show NO stale-region regression for the tree
under test; fail loud with the reverted lines" — so **leg 2 is a faithful implementation of the written
law, not a buggy one**. The finding is therefore not "fix the script" but: *as worded, the anti-resurrection
law and A-4's assignment cannot both be satisfied* — A-4's job is to narrow a guard that lives in one of
the two named files, which is definitionally a "reverted line". Someone must change the law's wording
(suggest: name the banned constants as the regression predicate, as leg 1 already does) or name a
waiver for A-4's specific edit before the merge is attempted. Not mine to decide; both the script
(`sed -n '55,160p' tools/ops/check_anti_resurrection.sh`) and the law text (`AGENTS.md:90-96`) are one
command from any seat, which is the only reason to write this down rather than leave it in chat.

## 2. RUN OF SHOW — one command per step, expected artifact, release line

Window budget 30 min (`WO_TP4_all_lanes.md:42`), cold-CIFS tax included. Steps are ordered so the
**cheapest decisive datum fires first**: if GATE 2 was skipped, step 2c tells you in <1 s at zero VRAM.

**Step 0 — preflight, zero device.** KFD tab-tolerant, raw-dumped, mandatory (runbook law 4):

    rocm-smi --showpids | sed -n '/KFD/,/^====*$/p' > results/amd/p3/G18w_kfd_precheck.txt
    grep -cE '^[0-9]+[[:space:]]' results/amd/p3/G18w_kfd_precheck.txt   # must print 0; refuse nonzero

**Step 1 — build the A-merged tip, then BANK before anything boots** (runbook §4 BANK-BEFORE-RELINK).
NAMED TREE required by the WO; the two candidates and their consequence differ, so name it in the row:

    # A4MERGE=<sha> = the amd/main commit that carries A-4's code
    git -C /home/chris/worktrees/amd-wo-p3-serve checkout <sha>   # or the equivalent in agent1's lane
    /home/chris/opt/cmake/bin/cmake --build build-hip-amd --target ninfer-serve -j4
    sha256sum build-hip-amd/apps/ninfer-serve
    cp -p build-hip-amd/apps/ninfer-serve /home/chris/artifacts_bin/ninfer-serve_<sha16>.bin

Banking is not optional: `build-hip-amd/apps/` **relinks by design** and has already eaten one
pinned artifact (`8e6d79ae`, runbook §4 incident). Every later step boots the **bank path**, and the
row names its sha. Note `results/amd/p3/G3_bringup_kit.sh` v6 hardcodes
`LANE=/home/chris/worktrees/amd-wo-p3-serve` + an unconditional `cmake --build`; it is the wrong
instrument for a pinned bank boot (see `docs/amd/AGENT1_step1_read_audit.md` Defect 3) — the kit's
**recipe** below is copied from its measured launch shape, its build step is not used.

**Step 2 — G-AMD-18 first light (world=4).** Launch shape is the `17f`/`G17v` measured recipe
(`grep '^\[gating\]'` count = arms-traced predicate), mask-ordinal law respected: `HIP_VISIBLE_DEVICES`
**remaps** the selection to logical 0,1,2,3, so `--devices` stays literal-but-masked. dev0 carries the
desktop (148 MB measured) — that is the pair-avoidance law of runbook §3, and a 4-card window is the one
cell that must accept it; the allocator is the gate, per VRAM LAW, no estimated refusal may block this:

    HIP_VISIBLE_DEVICES=0,1,2,3 NINFER_WORKSPACE_MIB=96 NINFER_GATING_TRACE=1 NINFER_VRAM_TRACE=1 \
      setsid nohup /home/chris/artifacts_bin/ninfer-serve_<sha16>.bin \
      "$(cat results/amd/p3/Q3_CANONICAL_PATH)" --port 8093 --devices 0,1,2,3 \
      --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 --no-cuda-graph --greedy \
      > results/amd/p3/G18w_serve.log 2>&1 < /dev/null &

2a. **construction** — if GATE 1 is still in, this dies in <1 s with the `world=4 requested` throw and
    **zero device touch**: that is an admissible, informative outcome (CAPABILITY refusal, not capacity).
    **Second admissible throw, named so it is not mis-read as VRAM**: agent2's TP4 answer (v340l/17,
    `5e89bb0d` corrected form) reports the artifact is rank-agnostic — shapes declared FULL, nothing
    pre-halved — but flags that under a declared group size of 128, `groups_per_row = 9` is not divisible
    by 4 for three `(rows, cols)` shapes (81 tensors), which is a **RowK** axis exposure
    (`check_div(groups_per_row, num_ranks)` would throw), while **ColumnN** is clear (all rows 4-divisible).
    If first light dies on a divisibility throw at construction, that is a **plan-role mapping question**
    (which axis those 81 take) and NOT a capacity verdict, NOT an artifact defect, and NOT GATE 1 — the
    distinction matters because the VRAM-LAW reflex is to read any refusal at boot as a budget problem.
    Whether those tensors are ColumnN (⇒ TP4 needs no artifact change) is agent2's own stated non-claim:
    they offered the role-mapping pass as derivable read-only and explicitly refused to infer it from
    shapes or names. Cite them, don't guess it. My §2b weight derivation is unaffected either way — it is
    arithmetic on a measured row and is labeled as never-a-gate.

    **Citation ceiling — lowered to the author's confidence, then partially RESTORED by reading the sharder
    myself instead of inheriting either the claim or its retraction.** agent2 disclosed (#574(3)) that their
    `4304 % 128 != 0` observation may mean **their reading of the `k128` tag is wrong** for that family, that
    they mis-axed once (used `shape[0]` where the code uses the columns side), and that their published list
    "deserves a re-run rather than trust." So I went to the source at `origin/amd/main` and the MECHANISM is
    independently confirmed:
      - `weight_shard.h:8` defines group id as `row*groups_per_row + group` → `groups_per_row` is
        **columns / group_size**, i.e. the author's *corrected* axis, not their first one;
      - `weight_shard.cpp:70` `check_div(src.groups_per_row, num_ranks, "groups/row for RowK")` — the throw
        exists and **its own message names the axis RowK**;
      - `weight_shard.cpp:65` `check_div(src.rows, …, "rows for ColumnN")` — ColumnN is gated on **rows**,
        and 1152 / 3456 / 4304 are all 4-divisible, so "ColumnN is clear" holds on arithmetic I checked.
    What genuinely remains open is therefore narrower than "is the arithmetic right": it is **which of the 81
    `groups_per_row = 9` tensors the plan assigns to RowK** — a role question the artifact cannot answer, and
    the only thing that decides whether first light meets this throw at all. Derive it from plan role at
    point-of-use, per the author's advice, not from any relayed list including this paragraph. What survived
    the walk-back untouched either way is the *classification*, which never depended on the arithmetic: a
    construction-time divisibility throw is a plan-mapping fact, not a budget fact.
2b. **load floor** — expect `[rank N] decoder state: … MB, … kv pages` ×4 and the `[preflight]
    live-free per rank` 4-tuple; weights/rank **derive** from the measured world=2 number
    (7,102 MiB/rank, `G17v_verdict_row.txt`) ÷ 2 at world=4 ≈ 3,551 MiB/rank — *derived arithmetic on a
    measured row, printed for the record, never a gate* (VRAM LAW: report measured only).
2c. **the first token** — one greedy request, `max_tokens 32, temperature 0`. **This is where GATE 2
    shows up**: valid-looking tokens whose ids are invariant to the prompt, or all ranks' `arm-trace`
    lines present with no cross-rank reduce, means `out_token` was never written. Stop, release, report
    GATE 2 as LIVE rather than reading it as a TP4 bug — the discriminator is `PREDICATE-2`.
    Expected artifact: `results/amd/p3/G18w_response.json` + sha16 of it in the row.

**Step 3 — B-1 (agent5's cell, rides the same window, ~2 min).** Own stamp, never mine to fire:

    LANE=<agent5 lane> CYCLE=<cold|warm> LABEL=B1w \
      tools/v340l/run_b1_rccl_ar_sweep.sh
`CYCLE` is **mandatory** by the runner's own `:?` guard (verified in the file); it writes
`results/amd/p3/G<B1w>_b1_sweep.log` + `.sha256`, self-`hipcc`-builds to `/tmp/b1_rccl_ar_sweep`
(transient instrument, correctly not a truth artifact), thresholds **120/240 µs** are WO law and must
appear in the row, and legs B-1a(w=2 cross-card)/B-1b(w=4)/B-1c(same-card pair) give the first
same-box tree-vs-mesh-vs-RCCL per-call comparisons.

**Step 4 — warm near-capacity margin cell (WO Task C) if Task C was not executed earlier.** Boots the
**banked `0c62ffdfd402a4b8`** binary (independently hash-verified at this desk — audit doc, "measured at
this desk"), same recipe as step 2 with `HIP_VISIBLE_DEVICES=2,3 --devices 0,1`, `CAP_MODE=auto` shape
(no `--max-context`/`--kv-capacity` → the AUTO probe is what boots), immediately after step 2/3 so the
pair is **warm**, and it must print the `[preflight] required … / usable …` composition with the real
slack. Exit truth (WO §2, VRAM LAW): **warm+fit → LAUNCH, margin printed; allocator refusal at the proven
floor + 2 MiB IS THE DATUM — report, never retry, never pre-refuse.**

**Release row — one per step, death included (runbook law 3):**

    RELEASE G-AMD-18 <step> date_utc=$(date -u +%FT%TZ) bin=<sha16> pid=$SPID \
      kfd-post=$(rocm-smi --showpids | grep -cE '^[0-9]+[[:space:]]') \
      serve=results/amd/p3/G18w_serve.log@$(sha256sum results/amd/p3/G18w_serve.log | cut -c1-16) \
      cycle=<cold|warm> — own pid killed only

## 3. FALSIFIER (WO §2: "the check must be able to fail")

Resolve-check for every instrument this doc cites — run it, expect all-OK, then **plant the typo**:

    for p in results/amd/p3/G3_bringup_kit.sh results/amd/p3/Q3_CANONICAL_PATH \
             tools/v340l/run_b1_rccl_ar_sweep.sh tools/v340l/b1_rccl_ar_sweep.hip \
             tools/v340l/tps_probe.py docs/amd/BOOT_LAUNCH_RUNBOOK.md \
             docs/amd/WO_TP4_all_lanes.md docs/amd/A4_PAIR_SITE_INVENTORY_agent4.md \
             src/runtime/tp2/tp_engine.cpp src/core/multi_gpu/tp_group.cpp; do
      git cat-file -e HEAD:"$p" 2>/dev/null && echo "OK   $p" || echo "MISSING $p"; done
    # FALSIFIER DEMO (must print MISSING, proving the check can go RED):
    for p in src/runtime/tp2/tp_engin.cpp; do git cat-file -e HEAD:"$p" 2>/dev/null && echo "OK   $p" || echo "MISSING $p"; done

The planted typo is `src/runtime/tp2/tp_engin.cpp` — one dropped letter, same class as the e→f sha
disease at #797(2). If the loop prints OK for it, **the check is broken, not the doc**.
`A4_PAIR_SITE_INVENTORY_agent4.md` is deliberately absent from `amd/main` (it is agent4's lane tip
`4d8bfe6e`, not merged): resolve it with `git cat-file -e origin/amd/wo-p3-serve:docs/amd/A4_PAIR_SITE_INVENTORY_agent4.md`,
and note that its absence from `amd/main` is itself evidence for GATE 1.

## 4b. AMENDMENTS 2026-09-14 23:2xZ (annotating, never deleting — night's own law, applied to this doc)

1. **§0's GATE-1 evidence half-refuted by the merges:** `4d8bfe6e`/`2c7738c8` (agent4's A-4 *inventory*
   commits) ARE ancestors of `amd/main` since the chair's merges — I checked `git merge-base
   --is-ancestor 4d8bfe6e origin/amd/main` → true at `3a71ab4d`, and GUARD LIVE at `60713b0b`. The
   inventory doc landing while the guard stays live proves the doc was never a valid proxy for the
   code: **PREDICATE-1 is now the ONLY GATE-1 test this runbook stands behind; the §3 "absence from
   `amd/main` is itself evidence for GATE 1" sentence is dead.**
2. **§1-P3 restated at its close:** item 7 CLOSED — triple RED `1d0ff3c6` / GREEN `db09c924` (25/25
   `348e77a1222dea7f`) / WIRED `22ce0122`, check (r) permanent. The first-light bar stands as
   written (coherent + zero-fault + arms-traced + cycle class — never `same-token`); the lag-skew
   caveat dissolved as **latency-class, correctness untouched** (census G-AMD-34 `d752eb54`, rulings
   `60713b0b`: 831 retries, max try 8, median 655, rank0 95.8% one-sided; 50/50 cross-boot text
   identity G18d+G18e, field-scoped to greedy text at world=2 — that field-scope is the chair's
   wording, keep it).
3. **RULING 2 rides the window manifest (chair `60713b0b`, affects step-2 release rows):** the
   rb-**n** world-2 JOIN is INVALID branch evidence — rank1's thread_local counter is structurally
   pinned to 1 by the per-request `std::thread` (inline rank0 vs threaded rank1: zero
   discrimination guilty-vs-innocent). The window row must instead (a) cite rb-**PRESENCE**
   (`52 = 26×2` witness class) and (b) **NAME world=4 attribution as unmeasured** — no rb-join
   inference may appear in a world=4 conclusion until a world=4 log exists to do the join on.
   Pre-declared K=16 re-open trigger to watch from the window's own rows: max try ≥ 12, any death
   at K=16, or world=4's fresh distribution (`60713b0b` item 1).
4. **Desk/baseline drift:** header `5657a174`/`dfd5b7ec` superseded — lane tip `9224da0f`, pre-flight
   row `ba31b690` (merged `586896ef`), empty-guard cells merged (`4c6f7dee`+`9224da0f` on main via
   `26a21f6c`), rulings at `60713b0b`. The §3 resolve-list gains
   `tools/smoke/diag/determinism_empty_guard_check.sh`.
5. **FLIP-DAY closure + live pre-flight (2026-09-14 00:5xZ):** §0's two gates now carry CLOSED-with-citation
   annotations (merged src `6b98623f`, flip co-land `83d1b00e`, dual-parser certification re-fired at
   `98923fda` and boot tip `92564abb` — rows: `results/amd/p3/G18_FLIPRUN_2026-09-14_agent1.md` +
   postscript). §1's P-table: **P1 and P2 satisfied** at the boot tip (P1 by A-4's merge ancestry, P2 by
   PREDICATE-2 rc=0 — decision-shaped, not inherited); P3–P5 unchanged and re-measured in the pre-flight
   row (artifact 15,446,796,288 B exact; disk 11 G; residence 92.87%). New §2-step-0 material from the
   flip era, all in `G18_PREFLIGHT_2026-09-14_agent1.md`: PREDICATE-1's live shape is the rank-COUNT form
   at `tp_engine.cpp:728` (GUARD LIVE there is the PASS condition now); reader legs must cite `a8727267`
   as world=4 certification provenance (pre-fix pairing was BLIND at 4 ranks — 6/47 lines parsed,
   armed-clean printed); window first-light bins read with `CENSUS_STRICT_KINDS=1` (post-A-4 era rule,
   executable at `3374f5c4`, prose at agent3's `d4753a84` on `origin/amd/t3-wip` — name the branch, it is
   not a main ancestor), five pre-A-4 banked bins stay lenient. G-AMD-35 battery released on the wire at
   `b6459cca@origin/amd/wo-p3-serve` (LEG ZERO inert-on-every-axis; M1 premise-unmet; LEG3
   ARMED-NOT-SEEN ⇒ the `kind=ar` evaluation transfers to this window's first boot).
6. **Re-stamp at `e2044290` (01:1xZ, chair seq-30):** boot tip moved (fix is tools/tests/docs-only — src
   equality to the bin tree `83d1b00e` re-measured EMPTY, verify-not-build holds); all step-0 predicates
   re-fired at the new tip and the FULL PG-1 gate re-derived at this seat rc=0 (19/19 checks (a)–(s)) —
   the clean-baseline line is now certified at two seats. §1b carries its FILED→MERGED upgrade above.
   Sequencing: agent4 boots **M4/G-AMD-37 (pair, ~12 min) FIRST**; pair measured free at 01:11:45Z
   (KFD-0, idle bytes exact), window-side physical witness re-stamps on agent4's M4 release minute — GO
   stands gate-side, re-declared by this desk before `d3a0e738`'s first boot.

## 5. What this runbook does NOT close (ORIGINAL — frozen at §0-§4 form)

- **Who removes GATE 1/2** — agent4's A-4 seat; I only establish that both gates exist at all current
  tips and that GATE 2 fails SILENTLY, which is the part that would waste the window.
- **Which** of the 81 `groups_per_row = 9` tensors the plan assigns to RowK (vs ColumnN). The mechanism, the
  axis, and ColumnN's 4-divisibility are now confirmed at `weight_shard.{h:8,cpp:65,70}` by direct read at
  `origin/amd/main`; the *role assignment* is a plan fact the artifact cannot supply, and it is the sole
  remaining variable for step 2a's third exit. Owner is agent2; the chair accepted the derivation as Phase-2
  queue item 1, read-only, run fresh at point-of-use. I did not attempt it (WO §3 scope), and this row is
  written to be **superseded by their result**, not to compete with it.
- The 3,551 MiB/rank figure is derived arithmetic on agent4's measured row, not a measurement; step 2b
  replaces it with the live printout or the row says "unmeasured".
- G-AMD-18's bar is P3's wording, not `same-token` (dead since `1d0ff3c6`); if anyone re-introduces a
  byte-identity gate on the 4-card side, that gate asserts a property this tree does not have at world=2.
- Task C's grant machinery: this desk now runs with **push mail off by user order** (`AGENT_COMM_MAIL`
  unset), so the written-ack exchange in WO §6.4 has no transport. Staged, not requested.
