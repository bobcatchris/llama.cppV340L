# PRE-FIRE RECEIPTS A+B — battery at the [C] lane tip + anti-resurrection diff (agent2, chair seq-26, 2026-09-15 ~01:0xZ)

Zero card, zero build in the shared tree, host-only. Both tasks grade the tip the chair named:
**`amd/wo-p3-serve` @ `3a1a91ed`** (= `[C] r18: seq-16 folds … boot stamp f8e762425d0d8bb0`,
verified as that branch's tip by `git rev-parse`, and its banked bin exists:
`/home/chris/artifacts_bin/ninfer-serve_f8e762425d0d8bb0.bin`, 130,728,536 B, sha256 first-16
`f8e762425d0d8bb0` — filename-is-the-stamp law satisfied).

---

## TASK A — arming battery at 3a1a91ed: **ALL LEGS GREEN, rc=0, no stop-the-world red**

Two runs, both by `git-archive` extract (read-only, discarded on exit), both printed here as they
came out. The difference between them is the finding, so both are banked.

**Run 1 — battery invoked from its main-tip home (305-line script, sha256 `024d97bb…`):**
18 leg lines, **14 ok / 0 FAIL / 4 NOT-OBSERVABLE**, **rc=2** with its own verdict line
*"INSTRUMENT-ERROR: 4 leg(s) NOT-OBSERVABLE — do NOT ship a GO row from this run"*. The four misses
are PRED-D (admitted-shape instantiation), LEG15, LEG15b, LEG15c — all four for the SAME reason the
battery itself prints: *"no checker in the tree nor beside the script"*. Measured, not inferred:
`git cat-file -e 3a1a91ed:tools/v340l/tp4_arming_battery.sh` → **ABSENT**, and the four checkers
(`admitted_shape_instantiation_check.py`, `gqa_head_geometry_cell.py`, `collision_fence_plant.py`,
`head_row_value_check.py`) are **absent at the fire tip too**. They are main-resident (landed at
`1917a8b7`) and live on `amd/wo-agent1-support`; the lane forked at `66366ede` (12:58Z) BEFORE that
landing and never merged it back.

**Run 2 — same battery, same tip, with the four checkers placed beside the script (the kit's own
documented `grade_with_sibling` fallback):** 18 leg lines, **18 ok / 0 FAIL / 0 NOT-OBSERVABLE /
1 WARN**, **rc=0**:

```
ALL LEGS GREEN at 3a1a91edd0018166eaf5d49f9a0c543914b4874f — GO-AMENDED row may be cut
(still owes: lane-green != boot-tip-green)
```

Leg-by-leg, read as printed (the text, not just the rc — my own doc-27 law):

| leg | result at the fire tip |
|---|---|
| PRED-A' factory-dispatch-live | ok — `wants_tensor_parallel` in `engine.cpp` |
| PRED-B route-table-names-Rccl | ok — `ArgmaxRoute::Rccl` count 1 |
| PRED-B r1-call-site-live | ok — `r1_allreduce_argmax` call site in `tp_group.cpp` |
| PRED-B guarded-dispatch-key | ok — `r1_ready` (staged && comms-live) |
| PRED-B ring-arm-vacuous-witness | ok — still the named NO-OP witness (never audited as a gate) |
| PRED-D admitted-shape-instantiation | ok **[agent1-lane, NOT shipped by this tree]** — CLEAN, every admitted shape instantiated in every chain, every chain ends in a throw |
| PRED-D selftest | **WARN — skipped, graded from the lane copy**: read-at-bytes witness, **not** a registered-suite claim |
| PRED-E head-ladder derives from world | ok — delegates to the generated head table (totals+world rows, collision law at emit time) |
| LEG15 / 15b / 15c head-geometry policy / fence plant / row VALUES | ok ok ok [lane copies] — GREEN / FENCE HOLDS 6/6 incl. non-colliding control / 4 emitted rows all equal config-derived, group world-invariant |
| cell `tp_argmax_r1_host` | **ok — rc=0, 30 arms green, src sha8 `23ab4a33`** (this file is ABSENT at main — it is a lane-resident witness, and it is GREEN at the fire tip) |
| cell `tp_argmax_routing_host` | ok — rc=0, 31 arms green, src sha8 `0c65a382` |
| gate2 silent-nullpath | ok — GREEN vs this tree |
| T3 barrier-arity | ok — `sync_bar(2)` **absent** (the family-#4 closure holds at this tip) |
| T1 materializer takes RUNTIME world | ok — `materialize_tp(..., rank, tp_w, ...)` |
| T1 tp_w derived from group.size() | ok — runtime world is the source |
| engine tp_world from device count | ok — measured, not a literal |
| PRED-C disk | ok — **19 G free on /**, report-only, never a refusal |

**Answer to the chair's question ("name any red that is NOT the known non-boot-tip class"): there
are NO reds at all at `3a1a91ed`.** Nothing pre-fire stop-the-world. Pre-fire the tree is clean on
every leg this kit can see.

Two honest caveats, both mine to state rather than let the green slide:
1. **18 legs at this tip, not 17** (main's count this morning) — the count belongs to the tree, and
   this tree grows the LEG15 trio because it carries the HeadGeom emission main lacks. Anyone
   citing "17" for this sha is citing my main-tip run.
2. **The all-green half-depends on borrowed cells.** The PRED-D/LEG15 family graded from agent1's
   lane copies, and the battery says so in its own text. The permanent fix is one line of board
   work, already named in the kit's message: **merge `tools/v340l` (agent1's four checkers + the
   battery itself) onto the lane, or merge the lane onto a tip that carries them.** Until then the
   boot-tip GO row must carry the WARN line verbatim — that is the read-the-text law, not
   decoration. **Recommendation: after the boot sha lands, I re-run at the BOOT tip; if that tip
   still lacks the checkers, the four legs print NOT-OBSERVABLE there and I will report rc=2 as
   what it is (a kit blind spot at that ref), not as a boot defect.**

## TASK B — anti-resurrection check at the fire tip (`AGENTS.md` law, farm-step-0 shape, zero GPU)

Law as written: `git diff main -- src/runtime/tp2/tp_engine.cpp src/runtime/tp2/tp2_budget.h` must
show **no stale-region regression** for the tree under test; if the branch carries an older copy,
**main's version wins by law**. Measured at `amd/main` = `ed4ddca7` vs `3a1a91ed`:

| file | blob (main) | blob (lane) | verdict |
|---|---|---|---|
| `src/runtime/tp2/tp2_budget.h` | `4a567aae…` | `4a567aae…` | **BYTE-IDENTICAL — zero diff, content8 `e369358a` both sides.** No stale region, no resurrection, nothing to arbitrate. |
| `src/runtime/tp2/tp_engine.cpp` | `9b996ded…` (content8 `46bd23e0`) | `20c0a010…` (`a08b65ed`) | **12 added lines, ZERO deleted lines.** The additions are `#include <functional>` + two `getenv("NINFER_BATCH_DBG")` breadcrumb blocks around the engine-mutex lock in `TpSubmission` (the family-#9 organ's instrument, gate-quiet-when-unset). |

**Preflight-region-specific probe** (the part the law actually protects — the region was extracted
by pattern, hashed, and compared, not eyeballed): the block from `// VRAM preflight per rank` to
`DURABLE GUARD` is **54 lines and hashes to content8 `afceb97e` at BOTH refs — IDENTICAL.** A
keyword scan of the whole engine diff for `preflight|budget|fixed_bytes|staging_bytes|reserve|slack|
kGateRelief` returns **0 changed lines**. And the banned-constant sweep (the VRAM law's own list)
returns the SAME hit set on both sides — no new estimate-refusal constant appeared on the lane, and
`tp2_runtime_reserve_bytes = 0` (a measured zero, not a charge) is common to both.

**So: the two files the law names are CLEAN. The lane does not carry an older copy of anything
there — main's canonical home survives intact into `3a1a91ed`, plus 12 lines of debug breadcrumbs.**
A farm step-0 arm running `git diff amd/main 3a1a91ed -- <those two paths> | grep -c '^-'` prints 0
and can gate on it, which is exactly the shape the law asks for.

### The part the law does NOT cover, and the merge-gate call needs (newer-BY-CONTENT table)

The chair asked which side is newer-by-content, not newer-by-commit. Commit dates would mislead
here: `tp_engine.cpp`'s last main-side touch is 09-13 18:21 while the lane's is 09-14 17:09 — but
**the decisive fact is that main has touched NEITHER file's canonical region since the fork, and the
lane's copy is a strict superset of main's.** Established by three probes:

1. `git log 66366ede..amd/main -- src/runtime/tp2/tp_engine.cpp src/runtime/tp2/tp2_budget.h …` →
   **empty** (main changed nothing under `src/` or `tests/` after the fork — its post-fork landings
   are all `tools/v340l/nvfp4/*` cells and docs).
2. `git merge-base --is-ancestor bc34afbd 3a1a91ed` → **true**: main's last `tp_engine.cpp` touch IS
   an ancestor of the lane tip. Same for main's last `tp2_backend.cpp` touch (`5c9ed9c4`, the GATE-3
   capacity guard, below the fork and present in BOTH).
3. `comm -12` of the two post-fork change-sets under `src tests` → **the overlap set is EMPTY**:
   main touched 0 files there, the lane touched 31 (+2106/−196). **There is no file both sides
   changed since 12:58Z, so no merge in either direction can silently revert the other.**

Which means: for these organs **lane-is-newer and main-is-unmoved are the same statement, and
there is no arbitration to do.** The stale-region hazard the law exists to catch is not present at
this tip. What IS present is the known, chair-owned board finding, restated here with which-tree
per cite so it cannot be read as a lane defect:

| organ | `amd/main` = ed4ddca7 (fork-era copy) | `3a1a91ed` (fire tip) |
|---|---|---|
| `tp2_backend.cpp:1145` / `:1177` | `const int n_vocab = 248320 / 2;` | `const int n_vocab = 248320 / tp_w;` + a NEW loud refusal `if (248320 % tp_w != 0) throw` at `:652` naming both numbers |
| `:1156` / `:1188` | `state->full_logits = a_bf(2 * n_vocab);` | `a_bf(tp_w * n_vocab)` — sendcount × WORLD |
| `:2576` | `Tensor full_logits(…, {248320, 1})` | table-derived extent stated at the call site |
| `:1626/:3591/:5224` | three `std::barrier sync_bar(2)` | `sync_bar(backend.world())` at `:1659` (T3 leg GREEN) |
| `tp2_request.{h,cpp}` | pair-sized pinned slots | world-sized (family-#8 cure) |
| `gqa_attention.cpp:40-48` | world-free ladder `{24→4, 16→2, 12→2}` + throw-else | `head_geom_for_q()` view of the generated table, unknown `q_heads` refuses LOUD naming the number (PRED-E GREEN) |
| `src/ops/shape/tier_shape_table.h`, `tools/ops/{gen_tier_shape_table,check_kv_tier_priced}.py`, `tests/test_tier_shape_table.cpp`, `tests/multi_gpu/tp_argmax_r1_host.cpp`, gate letters `(t)/(u)/(v)` | **ABSENT** (that is why my main-tip battery printed 8 reds + 1 NOT-OBSERVABLE this morning — expected-red, named-not-assumed) | present, and the two new host cells run GREEN at the tip (30 and 31 arms) |

**Recommendation to the chair for the post-[C] merge-gate call, stated as a preference with its
reason:** merging the lane's `src/` band into `amd/main` is safe BY MEASUREMENT at this moment
(empty overlap set) and is also what makes the battery's four missing legs and main's eight reds
disappear — but the merge should carry `tools/v340l/{tp4_arming_battery.sh, admitted_shape_
instantiation_check.py, gqa_head_geometry_cell.py, collision_fence_plant.py,
head_row_value_check.py}` ALONGSIDE it, or the boot tip will grade those legs from borrowed copies
again (the WARN row above). Landing the cells with the code they grade is the "land-with" template
this board already uses; landing the code without them recreates the law-with-no-bar row I filed
this morning as table row 4.

## BONUS, from the same hour (banked, chair ruling 3 applied)

The 12.4 MiB census-gate fixture now has a REAL generator instead of a prose recipe:
`tools/v340l/nvfp4/make_census_fixture.py` (my lane, commit below — NOT joining the repo tree as an
artifact; the FIXTURE itself is banked outside git at
`/home/chris/artifacts_bin/nvfp4_census_fixture_f1dab9de89e5018e.ninfer`, 13,009,597 B, sha256
`f1dab9de89e5018e…`, per the bank-location rule; a farm job cites bank path + stamp, and
absent-fixture is rc=2 NOT-OBSERVABLE, which is honest only once agent3's refusal fix lands).

The generator's own three states are measured, not claimed — including a self-conviction:
* `make` → reproduces my hand-built fixture **byte-identically** (`f1dab9de…`, same 13,009,597 B),
  asserts `file_size − max(offset+bytes) == len(header)+len(directory)` before writing, refuses on
  bad magic / short read / missing tokenizer / `model.vocab` layout drift, and prints its LIMIT row.
* `verify --expect-warn 2 --expect-c4 247 --expect-pad 243` → **rc=0** ("gate rc=0 (warnings=2),
  selftest 23/23"), first against the /tmp copy, then re-run against the BANKED path.
* **RED captured on my own file, disclosed:** my first `verify` compared an int row to a str
  expectation and printed `C4 pairing = 247, expected 247 → FAILED`. A checker that can fail on
  agreement is the born-RED class I have been filing against other desks; fixed (one `str()`), and
  the fix is verified in BOTH directions now: wrong expectation (`--expect-c4 248`) → rc=1 with the
  named diff; absent fixture → **rc=2** with `INSTRUMENT-ERROR … NOT-OBSERVABLE is not a pass`;
  happy path → rc=0. All three states of my own instrument are on the record before anyone wires it.

## BANK-vs-GIT: the one open policy question (flagged per chair ruling 3, bank-first adopted)

I proceeded bank-first (fixture at `/home/chris/artifacts_bin/`, cited by path + full sha256, the
registration PR carrying a fetch + hash-check, absent-fixture = rc=2). **If the CI farm's policy is
that every test input must be inside the clone** (no out-of-band mounts, which is a reasonable farm
stance — it is how the 18.3 GB artifact got named NOT-OBSERVABLE in the first place), then a 12.4
MiB blob has to live in git or in LFS and that is a USER/chair call, not a desk call. Costing, so
the call is informed: 12,408 KiB is ~10× the largest doc this repo carries and ~0.01 % of a bin
bank file; it would grow every clone, not just every build. **Middle option I would take if asked:**
don't bank the fixture at all — bank the 20-line generator (already done: `make_census_fixture.py`
is 12 KB of python, main-eligible) and have the farm job that CAN see the artifact produce the
fixture at build time; the generator's `verify` subcommand then proves the produced bytes grade
before anything cites them. That keeps the repo script-only, keeps the fixture regenerable from a
stamp, and makes stale-fixture drift structurally impossible. The only thing it cannot do is run on
a box with NEITHER artifact nor fixture — which is exactly the case the banked copy exists for, so
the choice is: bank it, or accept that those boxes print NOT-OBSERVABLE for this gate family. Both
are honest; the current state (no bar at all) is the only dishonest one.

— agent2, zero card, `/tmp` scratch + one host-only g++ compile (reaped), shared checkout read-only
(all reads `git show`/`cat-file`/`sha256sum`), writes confined to my own lane + the artifact bank.
