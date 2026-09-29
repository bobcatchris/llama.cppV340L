# AUDIT ADDENDUM 2 — reader world-generality, one scope root-cause, and the retry-price verdict
**agent5 desk, 2026-09-13 ~23:5xZ. Zero cards, zero builds, everything re-derived at my seat from
`origin/amd/main`; predicates named per the closure bar (`AR_PARITY_ARM_design_agent5.md` §7).**

## F-C — BUG: the G6/G7 pairing legs I shipped TONIGHT were pair-shaped, i.e. blind at world=4
*Found by me, in my own code, the hour after I shipped it — the legs the window announcement now
cites (`G_AMD_18_WINDOW_ANNOUNCEMENT_PRESTAGED.md` §3 item 3).*

* **Mechanism.** My G6(a)/G6(b)/table/G7 legs and the v2 pairing leg matched ranks with
  `rank=[01]` and paired by the complement `1 - rank`. At world=2 that is the whole universe; at
  world=4 it is half of it, silently. Reproduced with a synthetic four-rank log (4 ranks × 3
  request boundaries × 4 steps = 47 step lines, ONE planted gap: rank3 never prints req2/step3):
  the shipped reader parsed **6 of 47** lines, reported **0** lone sites, printed
  **"armed-clean"**, and exited **0**. That is the F-A defect (kind-blind bucketing) relocated from
  *kinds* to *ranks*, and it points at the first TP4 boot instead of a banked pair bin.
* **Fix (tools-only).** Legs are now universe-driven: a WORLD census line prints the ranks the log
  actually carries (`distinct ranks seen: [0,1,2,3] (world = 4)`) with a reading note per case
  (0 → INSTRUMENT-ERROR; 1 → the pairing legs are VACUOUS and print NEVER-CHECKED, never clean;
  2 → pairwise complement valid; >2 → *no pairwise complement exists at all*, because the shipped
  one-shot family is pair-only by construction (`tp_group.cpp` gates `one_shot` behind `n==2`) — so
  at world=4 these legs report **presence sets**, and a partner-completeness claim there would be a
  category error, not a measurement). A step-line **totality assert** (`MC31-H step lines TOTAL:
  n/n parsed`) mirrors F-A's equality; verdict lines print the universe they were computed over.
* **Triple.** RED = the pre-fix reader retrieved from git at `925148f7` by the cell, run on the
  planted fixture: 6 lines read / 0 findings / "armed-clean" / rc=0 (leg 6, and the leg refuses
  exit 2 if that artifact can't be re-created *or* if it isn't what it claims — the cell asserts
  `rank=[01]` count ≥5 and zero `rank=[0-9]` in the retrieved text). GREEN = this tip: 47/47, the
  planted site NAMED with its missing rank, world printed (leg 7). WIRED =
  `tools/ops/check_census_retry_attribution.py` (now 9 legs, rc=0 with `--falsify --require-real`),
  whose gate leg is **main @ 393ce73d Check (s)** — so this addendum's cell is a suite member on
  arrival, which is the F-B/A self-guard difference from an hour ago, stated rather than assumed.
* **Two of my own defects caught while building the fix, named because they are the class:**
  (i) a blanket sed to repair the step slice (`RSTART+6,-6` → `+5,-5`) **silently broke the G7
  `epoch=` slice** (6-char label, not 5) and the leg answered "1,608 observed>epoch" — a false
  *conviction*, the mirror image of the false clean. Offsets are per-label; leg 9 now pins
  multi-digit `rank=/step=/epoch=/observed=/stamp=/gen=` at known values so no future blanket edit
  survives. (ii) leg 6 first asserted on a line only the *fixed* reader prints ("step lines TOTAL"),
  which read as 0 and would have overstated my own RED leg; and the forward falsifier's first anchor
  blinded the *ordinal counter* instead of the universe — it mutated a knob other than the one it
  claimed to test, and only failed to mislead because I printed both directions.

## Scope root-cause — agent3's 421 vs the reader's 434 on G18c (a naming gap, not a disagreement)
`census_read_analysis.py` sums retries **per request window** (boundaries = `[req N] done`), so lines
outside a window never enter its totals; G18c carries **13 `AR-RETRY` lines after the last request
boundary** (and 2 before the first, which do get counted), 434 raw − 13 = **421 exactly**. G18d/G18e
have zero post-boundary retries, so their per-request sums equal raw (1,010/831) and the coincidence
of the two predicates in two of three bins is what made the third look wrong. Both counts are
correct at their own predicate; any row citing "retries" must name which one (bin total vs
in-window). No verdict moves — the max-try/tail/median rows are identical under either.

## The retry-price verdict agent3 asked for: **87-96 ms/retry survives as an UPPER BOUND; the drift-controlled price is 72-79 ms and it lands inside the source's own named ceiling**
* **Reproduced first, exactly.** Their script at `c2145b99`, run by me on the three banked bins,
  prints **96 ms (G18c, R²=0.952), 87 ms (G18d, R²=0.978), 95 ms (G18e, R²=0.958)** — their numbers
  to the digit, second seat.
* **The control they said they couldn't run.** They named the entanglement honestly (prefill_extra
  alone gives R²≈0.97; the joint fit doesn't separate). But each bin contains **9 zero-retry
  requests** — an *in-bin* drift reference that needs no second predictor. Fitting `extra_decode`
  against the ordinal on those nine, then residualising the retry-bearing requests before the
  through-origin fit:

  | bin | their coefficient | drift slope (9 zero-retry reqs) | **drift-controlled** |
  |---|---|---|---|
  | G18c | 96 ms | 81 ms/req | **78 ms** (R² 0.960) |
  | G18d | 87 ms | 75 ms/req | **72 ms** (R² 0.978) |
  | G18e | 95 ms | 65 ms/req | **79 ms** (R² 0.961) |

* **Why that is a verdict and not a re-pricing quarrel.** The mechanism names its own ceiling:
  `kFlagTimeoutPolls = 1<<15` pinned-host polls at ~2-3 µs = **65-80 ms**
  (`one_shot_argmax.cu:15,:254`; same arithmetic `one_shot_allreduce.cu:20,:142`). The raw fits sit
  *above* that ceiling (87-96) and the drift-controlled ones sit *inside* it (72-79), in all three
  boots, across both dial states — one coincidence would be luck, three-plus-three is a shape. So
  the priced mechanism **stands**, at the controlled value, and the ~20% raw excess is the ordinal
  drift they suspected. Their "the absorb path is not free and its price is on the order of the
  timeout it waits out" is *supported more strongly* by the corrected number than by the original.
* **Limits, mine, stated at the same bar:** n=9 reference requests per bin; ordinal modelled linear
  (agent3's own B1-B6 cause arms are what test that, not this row); this is a rate claim on
  wall-clock so it inherits their collinearity-with-time caveat in full — it prices the absorb path,
  it does not name the cause. And per the VRAM-law polarity it must never be quoted as a bound that
  refuses something: a cost of ~75 ms/retry is an argument about *policy* (K, crossing), not about
  permission.
* **Admissibility under the closure bar:** shelf **(b)** — it names and prices a mechanism, closes
  no bug. Item 7 is closed elsewhere; this row makes the chair's K-ruling (queue item 4) a costed
  decision instead of an adjective, which is what a rate claim is for.

## §5 — ANNOTATION on `fd2393f9` (its message does not describe everything it contains)
That commit's text covers the census reader, the cell, and the runbook; `git show --stat` proves it
**also** carries 36 lines in `tools/v340l/run_b1_rccl_ar_sweep.sh`, which belong to a different item:

**B-1 gains MEASURED board identity per leg**, from agent3's G-AMD-34 sysfs witness, verified at my
seat with `rocm-smi --showuniqueid`: GPU0 `0x21551d485ea3144` + GPU1 `0x21551d485e231a4` share prefix
`21551d485`; GPU2 `0x2155c0cdd0429c4` + GPU3 `0x2155c0cdd1018e4` share `2155c0cdd` — so **HIP 0,1 are
two dies of one package and HIP 2,3 of the other**, confirming agent3's "the working pair is two dies
of one physical board". It belongs in the instrument, not in prose: the A6 convention closure B-1
exists to settle compares a cross-card 6.6-class leg against a same-card sibling 4.86-class leg, and
the runner's defaults label those classes by **index arithmetic** (`DEVS2A=0,2` "cross-card",
`DEVS2B=0,1` "sibling") — true only under a particular mask, and the window announces
`HIP_VISIBLE_DEVICES=2,3`, where logical 0,1 **are** physical GPU2,GPU3 = one package. Every leg now
prints `ROW kind=topology` with resolved count, distinct board keys, the class, and the raw ids.
Two bugs in my own first draft, caught by running it against the four ids *before believing it*:
(a) I sliced **10** hex digits as the board key, which split GPU2/GPU3 into two "boards" — the last
digit is per-DIE; 9 is what this box's ids support (agent3's sysfs form `02155c0cdd` is the same 9
significant digits plus its leading zero). (b) An all-unknown leg (bogus device index) reported
`distinct=0` and then classified **SAME-BOARD** — zero evidence read as a clean, the F-A/F-C class for
the third time tonight; `UNKNOWN` (np==0) and `PARTIAL` (some ids unresolved) are now named refusals.
Measured table (host reads only — opens no compute node, allocates nothing, so it is safe *before* a
grant, unlike everything downstream of it): `2,3` → SAME-PACKAGE; `0,1` → SAME-PACKAGE; `0,2` →
CROSS-PACKAGE (2 keys); `0,1,2,3` → CROSS-PACKAGE (2 keys); `9` → UNKNOWN; `0,9` → PARTIAL.

## F-D — BUG (found by **agent2**, in my cell, 23:5xZ; RED capture reproduced at my seat before anything else): a real-data leg read 27.4% of the bin and labelled it TOTAL
* **Shape.** `check_census_retry_attribution.py` leg 5 capped its input at 60,000 of the banked
  bin's **87,634** lines, saw **277 of 1,010** retries, and asserted
  *"real 1,010-retry bin is TOTAL under the fixed reader: {'argmax': 277} vs raw 277"* — the number
  inside the message disagreeing with the word in its own label. The discarded tail is stamps
  553-804, i.e. **every retry at stamp ≥ 600**, which is precisely the late-census region the
  board's one open datum (steady-lag skew, median 639/655) lives in and the least surprising place
  for an AR-ring (kind-less) retry to appear.
* **Severity, in the finder's own terms and not softened:** no published number was wrong — agent2
  measured zero kind-less lines in the untested region, and 1,010/1,010 stands at three seats
  (agent4's bin, the chair's re-hash, my two rows). This is **future-detection** only, which is
  exactly what a permanent gate is for — and it became load-bearing the moment the chair wired it
  as PG-1 Check (s) with A-4's `kind=ar` co-land next.
* **Class, agent2's fourth instance and the sharpest variant:** the leg **RAN**, the reader was
  **CORRECT**, and the assertion was **TRUE** — 277/277 is internally consistent — so nothing in the
  gate's own output can reveal it. Their generalization belongs in every gate doc:
  *"a subset that self-reports as total is the one shape none of our current falsifiers can see."*
* **Fix (all four parts, `ececc2b0`).** (1) the cap is gone — full file, cost measured at 0.08 s in
  the cell (agent2 timed 0.072 s), so the CI-cost argument never existed; the deleted comment's
  rationale ("present whenever the census is") was a coverage claim the truncation defeated. (2) The
  finder's one-line rule, generalized: a `ground_truth()` whole-file pass is the denominator for
  **every** coverage arm — lines 87,634, retries 1,010, MC31-H step lines 1,608, A1TRACE-K 1,608 —
  and the **fraction is printed** with a mismatch as a refusal. That extends leg 8's vacuity rule
  from *rank-count* to *input*: a part of the data says PARTIAL, never clean. (3) agent2's class
  guard made permanent inside the cell — one kind-less AR-ring retry appended at the **last line**
  of a copy of the real bin must be found (it is: `ar-fallback=1`, raw 1,011) — an arm no truncated
  leg can pass. (4) A both-direction falsifier, which their repro could not be:
  `a2q5_check_s_head_leg_repro.sh` compares a full read against a head read **of the reader**, a
  property the reader cannot change, so it prints RED forever and cannot flip when the gate is
  fixed (it proves the discriminating region is > 60,000 — genuinely useful — but certifies nothing
  about my cell). So the cell gained `A2Q5_CAP`, an env that caps **this leg** on purpose for
  self-test only: `A2Q5_CAP=60000` → **rc=1 with four named FAILs** (coverage 60,000/87,634;
  totality 277 vs 1,010; step-leg 1,104 vs 1,608; G7 1,102 vs 1,608), default path **rc=0**, 37 ok's
  under `--falsify --require-real`. The ground truth always counts the whole file, so a cap cannot
  hide itself in the denominator — that ordering *is* the fix.
* **Receipt state:** `ececc2b0` on `amd/wo-gfx900-perm` (pushed, ls-remote-verified). **`origin/amd/main`
  still carries the capped leg** until the chair picks it, so agent2's finding remains live at main
  and can be verified by the flip test above rather than by my word.

## §6 — REVIEW LAW C/D applied back at this desk (chair 31a79add, adopted same hour; `e5544ceb`)
The two new review questions were turned on my own shipped rows before anyone else's, and the first
answer was a correction rather than a clean pass:

* **C (each direction OBSERVED, inferred twins don't count) — my leg 8 failed it.** The arm says a
  single-rank log must print NEVER-CHECKED, never clean; that is true of the current reader, but the
  RED direction had been *inferred* from the world=4 specimen. Measured against the real pre-fix
  artifact (git `925148f7`, pulled by the cell), a world=1 log produced the **opposite** error: a
  false **conviction** (`2 lone-rank sites`, `world-(a) witness FIRED`, rc=0) because every rank-0
  step looked unpaired when rank 1 did not exist. One blindness, two polarities ⇒ the cell now holds
  **a fixture per polarity** (leg 8b), which is agent3's "the rule is directional" finally enforced
  on the instrument that keeps saying it.
* **D (three-state exit before any text; a number appearing is not a number measured) — caught in my
  own new arm two minutes after I wrote it.** The world=4 assertion first grepped the substring
  `LONE-rank` and tripped on the leg's own section header, i.e. a check reading prose. Replaced with a
  regex over the leg's SUMMARY **numbers** (`sites<47 AND lone==0 AND armed-clean`). Third instance
  of that mistake today, and the second that was mine.
* Registry closure (F-E, `0579b234`, on main) is the same law D one level down: `kind=<anything>` is
  not a measurement of the emitter set; a closed registry (`argmax`, `ar`) with refusal for the rest
  is. The specimen that proved the gap was agent2's probe label `ar2bprobe`, which my reader happily
  bucketed before F-E — total bucketing with an open bucket set absorbs new emitters silently.

Cell state after this section: `--falsify --require-real` → rc=0, **51 ok's** (47 before leg 8b).

## §7 — THE SHARED-STATE FAMILY, three clauses, one root (closed 02:4xZ; register owns the prose so
the rule is findable even if no fourth ruling commit is written)
Tonight's findings F-A, F-C, F-D, F-E are one shape: **a checker's baseline is what it measured in
this run, not what it remembers.** The board rulings carry C (observe, don't infer — `31a79add`),
D (three-state exit before any text — same ruling) and E (delta over remembered constant —
`e4727f67`). The two clauses that are NOT yet in a ruling commit are recorded here with their
mechanical witnesses, so the row points at code rather than at a promise:

1. **Assert a delta over a baseline measured in the same breath** (agent5, `c2ed1b3e`). Specimen: my
   tail-injection guard asserted `kinds['ar'] == 1`, which encoded an implicit constant that nobody
   else touches the tracked bin — agent2's probe appended its own registered line and my healthy tree
   failed its own arm. Witness: the arm now asserts `raw == whole-file + 1` and `ar ≥ 1`, and names
   the >1 case as agreement, not a finding.
2. **A cleanup must be scoped to exactly what the tool added — never a whole-file revert — and its
   witness is a pre-run digest**, with the corollary that a clean-at-start guard makes that digest
   EQUAL HEAD by construction, so the digest alone cannot detect mid-run interference (agent2,
   `19f0c09d`, wording tightened at their insistence in `0628cd7b`; proposed as law clause F, not yet
   a ruling). The ordering matters and my first version of this row had it backwards: agent2's v7 DID
   hold a pre-run digest (SHA_A), and reading "compare to the pre-run state, never pristine HEAD" as
   the remedy would let someone build v7 again and pass review. **Safety is the scoping; the
   comparison is only the proof.** Specimen: their v7's `git checkout -- $LOG` deleted a concurrent
   neighbour's work inside the run window and then certified `restored byte-exact` — failure in the
   direction of looking safe. A baseline taken BEFORE a run goes stale DURING it, so clause 1's
   "measure in the same breath" needs this clause's second half: the baseline must be re-observed at
   the moment of cleanup, or better, the cleanup must not be able to reach anything it did not add.
   Witness in my tree, rewritten after this correction: leg 11 no longer asserts global tree
   cleanliness (that was an exclusivity assumption about shared state — clause 1's error in a new
   costume, and mine); it asserts the cell's own WRITE-SET contains no in-repo path, reports other
   lanes' dirty paths as OBSERVATION rather than grading them, and refuses to tidy a dirty tree.
   Both directions measured on a real git tree: an injected in-repo write → RED naming the offending
   path (`16 paths, 1 inside the repo`); the shipped arms → GREEN (`15 paths, 0 inside`).
3. **A check that mutates shared state must report what it rendered unverifiable** (agent2,
   `6fe52fef`). Specimen is the subtlest of the night and was unfindable from inside either file: my
   leg 11 exists *because* of their v8, could only ever be defeated by their tool, and their tool
   could only see the defeat while their probe held my file — inside their mutation window my
   invariant printed `SKIP-NOT-PASS` (correctly refusing to fake a pass) while their auditor printed
   an unqualified `GREEN`. True statement, scope silently narrowed by who was holding the file.
   Witness, verified at my seat against my tip: their leg2b now counts my SKIP-NOT-PASS lines and
   emits `INFO leg2b COLLATERAL: my probe's own mutation made the cell SKIP 1 invariant(s)…` with
   the verdict scope stated (`rc=0`, my bin digest `8b4a7c84b24b…` unchanged, zero residual), and
   prints `collateral=0` on a tip that has no leg 11 — the absence is measured, not assumed.

The meta, stated once because it is the part that generalises past this repo: two well-behaved
instruments sharing a tree interfered through that shared state, and the interference looked like a
pass **on both sides**. Neither file's own review questions catch that; only the pair does. Which is
the argument for keeping an external auditor whose scope is stated as "no authority over the cell's
verdict" — and for the third clause, since authority is not the only way one tool can cost another
its coverage.

**CANARY (agent2's close, `4ba7466c`, added to leg 11's docstring): under scoped claims there is no
"dirty tree" state left to skip on** — foreign dirt is observation, our own writes are enumerated — so
the surviving SKIP branches are only instrument-unavailable (git missing, status non-zero). If leg 11
ever needs a skip for *"the tree was dirty"*, clause 1 has crept back in as a world-level assertion.
That is the falsifier on the law itself, not on the data, and it is worth keeping because the mistake
is seductive: I shipped it, agent2 shipped it, and both versions print confidently.
