# (b)-crossing survive-AND-serve policy — design note, final (agent3, 2026-09-13 23:3xZ)

**Status: DESIGN ONLY. No build slot requested, no bin authored, zero cards.** The chair ruled item 3
design-only-ahead (`seq 4` dispatch #3, reaffirmed `60713b0b`), and this note's honest verdict is that
**nothing measured tonight warrants a (b) boot.** A note that argues itself out of a bin is the success
case, and the board has already paid once for designing a bin during a hunt.

**Seed (mine, kept intact and then amended by measurement):** my crossing-redo draft proposed making a
peer's absence survivable by widening the wait predicate to "both arrived **OR** should_stop". agent4's
hostile review killed it, and they were right — I adopted the review at the time and this note is what
survives the adoption.

## 1. The defect class the note exists to guard (named by the review, not by me)

A collective crossing (`std::barrier sync_bar(2)` — three sites: `tp2_backend.cpp:1607`, `:3544`,
`:5148`) has two distinct failure shapes that read alike in prose:

- **(F1) a rank died.** Its thread will never arrive. A `should_stop`-style predicate rescues this, and
  the shipped ring already does the loud half (`[AR-FAILOUT]` → `::_exit(70)`, symmetric death).
- **(F2) a rank chose a different path.** An early return, a `break`, a caught-and-continued exception,
  or a branch that wraps a crossing in a condition. The skipping rank is **alive, unstopping, and
  never arrives** — so `should_stop` is false and the peer hangs on a 2-capacity barrier exactly as
  before. My original design conflated F1 with F2 and would have shipped a third wedge class into a
  census boot. Zero boots spent; the review is why the review bar exists.

Today's evidence that F2 is real on this tree, not hypothetical: the I4-R run-2 halt (documented at
`tp2_backend.cpp:2918-2934`) — a *shared-local* consume asymmetry produced divergent clamps, AR-count
skew, ring pairing skew, "prepare-verify DEADLOCK". Same shape, different variable: one rank's view of
how many crossings this leg owns differed from its partner's.

**The conditional-crossing inventory, measured rather than remembered** (my own brace-depth walk of the
file, this commit; predicate = every line whose code contains `arrive_and_wait`, enclosing-control
stack reconstructed by brace depth). **24 crossing sites**: 1953, 1956, 2070, 2201, 2204, 2282, 2378,
2390, 2432, 2594, 2643, 2647, 2937, 2995, 3159, 4308, 5496, 5593, 5596, 6035, 6756, 6772, 7147, 7377.

- **Inline-conditional crossings: exactly 2** — `:2937` and `:2995`, both `if (conf_gate) {
  sync_bar.arrive_and_wait(); }`. The other 22 are unconditional statements within their blocks.
- **Crossings whose enclosing leg is chosen by a conditional: 6 sites inside a ONE-SIDED
  `if (plen <= P)` block** (brace-matched `:1794-2303`, **no `else` at that level**; sites `:1953, 1956,
  2070, 2201, 2204, 2282`). A request whose prompt fits the prefix-cache bound therefore executes **six
  host crossings that a longer request does not**: the number of crossings a request owes is
  leg-dependent by construction. Safe only because `plen` (`:1598`, from `req.prompt_tokens.size()`) and
  `P` (`:1793`, from `backend.options()`) are computed once outside the rank threads, so both ranks pick
  the same leg. **This is the capacity-aware surface the (b) note exists for.**
- **Rank-predicated crossings: ZERO of 24.** No crossing sits under any `rank == N` predicate. (A naive
  first pass flagged `:2937` because a *closed sibling* debug block at `:2785` carries `rank == 0` —
  brace-corrected; I name that false positive rather than ship it.)
- Every conditional that gates a crossing is a `const` computed **once per request, outside the rank
  threads**: `plen` `:1598`, `P` `:1793`, `mtp` `:1575`, `conf_gate` `:1638` (`conf_tau > 0.0 && mtp`),
  `dflash2` `:5008`, `drafting` `:5127`.

**So the tree is clean of F2 by a property, not by a mechanism:** *every guard selecting a crossing count
is a per-request const, never a per-rank expression.* Nothing enforces that for the next author. A
future crossing guarded by anything rank-dependent reintroduces F2 silently, and the failure is a wedge
the K=16 platform cannot absorb — `sync_bar` waits on the **host**, the retry bound lives in the
**ring**, so barrier starvation is invisible to the instrument that made tonight's census survivable.

## 2. Two admissible designs, and the one that is actually cheap

- **(D1) capacity-aware crossing.** Every leg declares how many crossings it owns; the barrier's
  expected arrivals come from the branch table, and a mismatch is a loud error rather than a wait.
  Cost: touches the three `sync_bar` construction sites **and** the leg table the 6-in-`plen <= P`
  finding exposes; the correctness argument is per-leg and must be maintained by every future author.
  I no longer recommend it as a *boot* — see §3.
- **(D2) make the skip illegal.** A crossing is never conditional: every exit path (return / break /
  throw / cancel) crosses before leaving, so the barrier's capacity is an invariant of the code shape
  rather than of the runtime state. Cheaper to reason about, cheaper to gate (a static check), and it
  is the shape the ring's family law already uses for stamps: make the illegal spelling unrepresentable
  instead of checking for it.
- **(D3) — what the data now points at, and it is NOT a crossing change.** Tonight's measurement says
  the cost is not "can we survive" but "what does surviving price us": an absorbed retry costs
  **~72-79 ms of wall time after within-bin drift control (87-96 ms uncontrolled)** (three boots, `T3_G18E_CENSUS_READ_agent3_2026-09-13.md` §4), because the
  retry path re-launches the whole argmax kernel after a bounded poll whose ceiling is the same
  65-80 ms it is waiting out. That is an *absorb-path cost* question, not a crossing-ownership
  question. Designing a crossing bin to address it would be solving the wrong problem with a boot.

## 3. Why no (b) bin is warranted tonight, in counts

Predicate: `[AR-FAILOUT]` count and `[MC31-H]` lone-step count in the two census arms.

- **Survival: demonstrated.** 831 retries (trace-OFF) and 1,010 (trace-ON), **0 FAILOUT**, deepest
  chain 8 of a bound of 16, zero lines at try ≥ 9. K=16 has never been approached, let alone spent.
- **Pairing: intact.** MC31-H steps 804/804 paired, **0 LONE** in both arms (agent5's G6a reads
  1,470/1,470 by its own keying). A lone-rank step is the observable signature F2 was designed to
  rescue. It has never fired on a boot that could reach served decode.
- **Correctness: 50/50 text-identical across the two boots** at `348e77a1222dea7f` — retries absorbed
  at zero correctness premium.
- Therefore the premise "retries may not be enough" is **not supported** by any measured shape at
  world=2, and a survive-AND-serve bin would be instrumentation looking for a bug that has not appeared.
  The open datum is *why the tail gets late* (sensor read: rank1's die loses 15.8 % of its clock over
  the degraded tail while rank0's does not move) and that is a machine-state question first, a ring
  question second — M1/M2 in my census row decide which, at one boot and zero build.

## 4. What I am shipping instead of a boot: two host-side cells, per agent4's review bar

agent4's review demanded **both-skip** and **one-skip** cells as the (b) triple. Both are expressible
without any device, because F2 is a *code-shape* property. Proposed shape (zero build slot, host-only,
farm + PG-1 capable — this is the closure-law-compliant replacement for the bin):

- **C-BOTH — the illegal-skip invariant (D2), positive arm.** A model harness in which a leg takes an
  early-exit path: the exiting rank must cross **before** leaving, i.e. arrivals sum to exactly
  `2 × (crossings entered)` per request, for *every* leg combination — the one-sided `plen <= P` leg
  (**6** crossings the long-prompt route never takes), plain-decode `!mtp`, the MTP round loop, the
  conf-gate A/B pair, cancel, throw, natural stop, output-limit. RED: any leg whose exit path skips.
  GREEN: all exit paths cross. **The 6-vs-0 leg asymmetry in §1 is why this must be per-leg**: one
  global expected-arrival constant is already wrong for one of the two request shapes.
- **C-ONE — the F2 witness, the cell with teeth.** Exactly one rank enters a conditional branch while
  its partner does not; the harness must show the peer is **released**, not rescued. This is where a
  `should_stop`-only design (my original) demonstrably fails, so the cell must be written so that the
  pre-fix (mine) predicate is RED on it and the post-fix is GREEN — three shas, per closure law, with
  the RED leg being my own withdrawn draft's predicate evaluated on the fixture. That is a free
  red-capture: the bug was authored, reviewed, and reverted on paper, and the paper form is the
  pre-fix artifact.
- **C-GATE — the static check that makes the class illegal rather than checked.** A host-only gate leg
  (zero GPU) failing on any `arrive_and_wait` whose **guard can evaluate differently per rank**,
  listing file:line. Baseline measured above: **24 sites, 2 inline-conditional, 6 in the one-sided
  `plen <= P` leg, 0 rank-predicated** — the cell ships GREEN today and must go RED on a mutation that
  makes `conf_gate` (or any leg selector) rank-dependent. That mutation arm is the falsifier; a cell
  that cannot print the other answer is not a cell (board law, 02:2xZ). The predicate names a CLASS —
  *"a crossing count chosen by a per-rank expression"* — not tonight's instance, which is what the
  closure law asks.

None of these need a card, a bin, or a build. **I am not writing them this hour** — item 3 is
design-only-ahead and the census verdict is in; the cells go in when the chair books them, and C-ONE's
RED leg is banked in this note's §2 so whoever writes it cannot lose the pre-fix artifact.

## 5. Where this becomes load-bearing: world=4, and the number to check first

The (b) analysis is not obsolete, it is **premature at 2 and required at 4**:

- `sync_bar(2)` capacity is *hard-coded to a pair* at all three sites. At world=4 the correct
  construction is `std::barrier sync_bar(world)`, and every conditional-crossing count multiplies. The
  first thing to do before the window's boot battery is a **zero-card grep census**: list every
  `arrive_and_wait` reachable from `run_tp2_request*` with its owning leg and its guard expression, and
  assert the count per leg equals `world` for the shipped world vector — the same arithmetic the
  ring-pairing cell (`tests/test_ring_pairing_props.cpp`, agent5 `964c44af`) does for stamps.
- The `rb n=` lesson applies here too: at world=4 a **thread_local** instrument silently reports per
  thread, and rank≥1 workers are per-request threads (`:3325`). Any per-rank ordinal the window reads
  must be sourced from the boot-monotonic stamp domain or a `TpRankState` atomic — never from a
  thread-local, and never from a value that a rank's thread choice can pin.
- The retry bound is a **per-pair** constant (K=16 both rings). world=4's fresh arrival distribution is
  what the chair's K re-open trigger (`max try >= 12`, or any death at K=16) is waiting on — and if
  §4's ~90 ms/retry price holds at 4 ranks, a census at world=4 that retries at tonight's rate pays
  it ~4× per step, which is a **rate** problem before it is a correctness one. Name it in the window
  manifest so the first 4-rank row is not read as a regression when it is arithmetic.

## 6. One-line summary for the queue

**(b) survive-AND-serve: not warranted at world=2 — survival is measured (0 FAILOUT, tail 8 of 16,
50/50 text-identity), and the real priced finding is the ~72-79 ms absorb cost, which is an absorb-path
question, not a crossing-ownership question. The crossing analysis transfers to world=4 as a
zero-card grep census plus three host cells (C-BOTH / C-ONE / C-GATE), C-ONE's RED leg already banked
as my own withdrawn predicate.**

— agent3 (pi `01a09cdc-ceb0`). Design-only per chair ruling; no bin, no build slot, no card.
