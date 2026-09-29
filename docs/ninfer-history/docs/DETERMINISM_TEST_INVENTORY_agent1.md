# DETERMINISM TEST INVENTORY — what the gfx906 tree asserts that ours does not (agent1, differential seat #2)

**Desk**: agent1 · **Date**: 2026-09-13 ~18:2xZ · **Branch**: `amd/wo-agent1-support` (tip `c53b4293` at read time)
**Donor tree**: `/tmp/ninfer-gfx906` — **`7a3c18d9`** ("gfx906: TP2 slice 9c - revert flag-sync default after a card-2…", 2026-09-02), working tree **clean (0 dirty files)** at read time. Stamped because a path is not an object: my first draft hedged "no ref, not fetched" and was wrong — the tree *is* a git checkout and its sha is resolvable with `git -C /tmp/ninfer-gfx906 rev-parse --short HEAD`, so quoting it without one was an unforced citation gap.
**Our tree**: `amd/main` @ `dfd5b7ec`-lineage (`git diff dfd5b7ec -- src/ tools/ apps/` verified empty at this desk).
**Zero-card, zero build, zero lane edits. No file outside my branch written.**
**Feeds**: agent5's transport-side differential. Their split, per the chair: **theirs = WHY, mine = WHAT'S-TESTED.**

## 0. The headline, one paragraph

They already own the suite-member the RED→GREEN closure law (`53c09d17`) demands for item 7 —
`test_engine_tp2_real.cpp:419-422`, *"a second run in the same process must be bit-identical… a difference here
would mean a real ordering hazard."* Our tree carries **no analogue** (grep for the form returns one hit, and it
belongs to a *different* class: `t3_two_hop_seam_check.py:29` reproduces golden files on disk, not decode
tokens). So we do not invent a cell; we port theirs, and because G-AMD-27 (`1d0ff3c6`) already measured
**5/5 distinct greedy outputs from one server**, the port lands **RED on the current tree at first run** — which
is the pre-fix red-capture the law requires, already funded by a boot that already happened. The one honest
complication is §3: their cell asserts a property our tree is currently known to violate, so it cannot join the
suite as a green gate; it joins as a **named, tracked RED** (xfail-equivalent) whose inversion *is* item 7's
closure. Naming that distinction is the whole value of the inventory, and it is a decision for the chair and
agent4's hunt seat, not for me to make by committing a suite change.

## 1. Inventory table

Legend: **present** = we carry an equivalent asserting the same predicate · **absent** = no analogue anywhere ·
**degraded** = we have something in the same shape but weaker (script not cell, host not device, one prompt not a sweep).

| # | their cell/row | predicate (what actually fails) | ours | port cost | what it would have caught here |
|---|---|---|---|---|---|
| 1 | `tests/targets/qwen3_6_27b/test_engine_tp2_real.cpp:419-422` | same engine, same prompt, two `generate_greedy` calls; `repeat != tp2_tokens` → **"tp2 generation is not reproducible within one engine"** | **absent** | low — pure logic, needs an engine+artifact fixture | *exactly item 7's datum*, as a CI cell rather than a one-off boot. This is the closure-law suite-member. |
| 2 | `:284-288` (concurrent leg) | `tp2 != tp2_repeat` on both lanes → *"the two devices' streams are racing"* | **absent** | med — batched runner + 2 prompts | a stream-ordering race between ranks, which is the transport-side class agent5's B-1 measures but nothing asserts |
| 3 | `:290-293` | `tp2.first == tp2.second` → **"the lanes are sharing state"** | **absent** | low | the inverse hazard: a comparison that passes because both sides read the *same* buffer. Cheap, and the board has never written one. |
| 4 | `tests/ops/test_linear_add_split.cpp:188-195` `verify_shards_are_distinct()`, used at **6 sites** across `tests/ops/` | before any equality assert: if the two shard payloads are byte-identical → **"shard identity is untested"** | **absent (0 occurrences tree-wide)** | low, mechanical | every vacuous-green class this board re-derived tonight — a check that cannot fail because its inputs are equal by construction |
| 5 | `:87-104` `synthetic_prompt()` + the second-lane note | deterministic tokenizer-free prompt (fixed cycle `1000+(i*37)%4096`); the **second lane's prompt must be genuinely different, not a rotation**, because a rotation "was measured to put this model on a knife edge (greedy answer changes between batch 1 and batch 2 at tp1, with **no tensor parallelism involved**)" | **degraded** — our determinism corpus uses one prompt (`"Hi."`, 54 tokens) at the serve path | low | **a confound we are currently sitting on.** See §3. |
| 6 | `docs/gfx906/TP2-SLICES.md:247-248` corpus gate | `argmax 7/7`, `KL 0.000451 vs budget 0.00181`, `(1-cos) 7.87e-5 vs 8.53e-5`, `corpus argmax 636/640` over 128 positions × 5 prompts; **"byte-for-byte the S6 eager numbers"** | **PRESENT, different construction** — I first filed this "absent" and that was wrong: we own `tests/phase_gate.cu` (D1 `codes byte-diff <= kTolCodeByteDiff && max_ulp <= kTolScalesUlp`, `:402-425`) driven by `tools/bench/phase_tolerance.json` (schema `phase-gate-tolerance/v1`: byte/token phases D1/D2/D3/D7 gate on a **per-phase sha ratchet with tolerance=0**, FP phases on ulp), plus `tools/bench/byte_diff.py` (docs/105 byte-identity A/B comparator: prints **first divergence position + neighbouring tokens**, `--tol N`). Structurally stronger than theirs in one respect — the tolerance is a named, versioned contract in a file rather than a number in a doc table. | low — the two are comparable, not portable | nothing new; but it means our gap is **not** "no tolerance contract", it is "no *cross-rank* and no *repeat-run* assertion inside it" |
| 7 | `tools/tp2/parity.cpp:121-124` | graphs **OFF on both sides** for the primary comparison, because a captured decode graph takes its GQA envelope from the graph profile's frontier *range* while eager uses the *exact* frontier, and the envelope selects the attention split policy — so graphs-vs-eager would **confound** parallelism with capture. Secondary row allowed. | **present-in-spirit / absent-in-code** — our kit launches `--no-cuda-graph` on *every* boot, so we never compare the arms; their comment *explains why that is the right primary choice* | none (read it) | nothing new, but it is the written rationale our own `--no-cuda-graph` convention never had; worth citing in the runbook instead of re-asserting |
| 8 | `:263-268` `require_batched` precondition | if *neither* concurrency probe is batch-invariant at **tp1** → "the leg has no full-strength lane, **re-choose a prompt**" | **absent** | low | a determinism cell that silently tests a degenerate case. Our 54-token single-prompt corpus is exactly the shape this guards against. |
| 9 | their suite plumbing: `tests/ops/gdn_criteria.h`, `op_tester.h:84` ("EXACTLY the values the kernel will read") | tolerance criteria as named, shared constants rather than per-test magic numbers | **absent** — our checks print numbers without a named criterion object | med | numbers cited without an algorithm, the digest-family defect one level down |

## 2. Answers to the chair's literal question ("what did they test that we don't?")

Four things, in descending order of consequence: **(a)** in-process repeat-determinism of greedy decode at tp2
(#1, #2) — we have no cell of the form *same config, run twice, compare*; **(b)** a **distinctness control**
(#3, #4, #8) — they refuse to *credit* an equality whose operands could be equal for a trivial reason, six times
over, and we have zero instances of that shape anywhere in `tests/` or `src/`; **(c)** a *toleranced* corpus
gate (#6) that can certify near-equality without demanding bit-identity, which is the only bar consistent with
what item 7 has since established about our tree; **(d)** a documented confound-suppression rule (#7) instead of
an undocumented convention.

**(b) is the one I'd generalize past determinism.** It is the same law this board kept discovering in prose —
"the check that cannot fail", "a `0` from a search you haven't sanity-checked", "an empty result is most likely
your own tool" — but they *encoded it as a helper called from six sites*. We have the law; they have the code.

## 2b. Two errors in my own first draft, corrected above, recorded here

- **Filed "absent" for the tolerance-gate family while owning it.** My widened grep (`byte_diff|identity is
  untested|must not be identical`) hit `tools/bench/byte_diff.py` and, behind it, `tests/phase_gate.cu` +
  `tools/bench/phase_tolerance.json`. Row 6 is now PRESENT with the citations. This is the *third* time this
  session a negative claim of mine died to a wider grep — and the second time it was about determinism, where I
  had just finished arguing that absence claims carry the higher burden. Predicate for re-derivation:
  `grep -rln byte_diff tests/ tools/`.
- **Claimed the donor tree had no stampable ref.** It does: `7a3c18d9`, clean. Fixed in the header.

The near-miss is worth naming because of *how* it would have shipped: the table's headline (#1: no repeat-run
assertion) is **true** — `grep` for a second generation of the same prompt returns nothing in `tests/`, and
`test_decode_graph.cpp` (83 lines) has no eager/graph determinism comparison — and a correct headline makes the
adjacent rows feel vetted. One right claim was about to launder two wrong ones.


## 3. The finding that bears directly on item 7, flagged as a hazard, not a verdict

Their `:99-104` note reports that a **rotation of a synthetic prompt** — "the same pattern starting elsewhere" —
was *measured* to flip this model's greedy answer between batch 1 and batch 2 **at tp1, with no tensor
parallelism involved**, i.e. a pure near-tie knife edge, unrelated to any carry-in. Two consequences worth
agent4's and agent5's attention, both mine to *name* only:

- **Our determinism corpus is a single 54-token prompt at the serve path.** If near-tie prompts flip without any
  parallelism or uninitialized state, then a cross-request fork in our corpus is **not by itself evidence of
  carry-in**; it could be prompt-conditioned knife-edge behavior. That does not refute agent4's census or
  agent3's adjudication — the 5/5-distinct datum is 5 forks, and two of them carried a *token-repetition*
  signature that reads as state, not tie-breaking — but it is an **independent confound the donor tree says it
  measured**, and our hunt has not cited it.
- **The donor's own fix is a test-design rule we can copy at zero cost**: make the second lane's input
  structurally unrelated (they used a *quadratic* sequence for exactly this reason), and gate the leg on
  `require_batched` (#8) so a degenerate prompt fails loudly as "re-choose a prompt" instead of quietly passing.

I am not adjudicating which mechanism produced our 5 forks (WO §3: hunt seat is agent4's). I am recording that a
tree with the same engine family documented this confound, quantified it, and built a guard into the cell —
which is the kind of fact worth knowing before the next boot is designed.

## 4. What I did **not** find, stated so nobody hunts it twice

- **No `cmp==0` shell-form cell.** The chair's phrase "same config, repeat, cmp==0" describes #1/#2, but in
  their tree it is C++ value comparison (`std::vector<TokenId> operator!=`), not a `cmp` invocation. Files named
  `md5sum`/`diff -q`/`cmp -s`: **0 hits** in `tests/`+`tools/`. If a *shell-level* repeat-and-compare cell exists,
  it is outside the three surfaces I was pointed at.
- **No git ref on the donor tree.** `/tmp/ninfer-gfx906` — I did not fetch or resolve its HEAD, so nothing above
  carries a sha. Line numbers are the tree as-read; if the donor moves, they move with it.
- **`ARTAG` not encountered.** The chair's note asked me to flag if agent4's ARTAG control-path one-liner touches
  a file I would otherwise read; nothing I read carried that marker, and I made **no lane edits** regardless.
- **Their `tools/parity/qwen3_6_27b/` is vision-only** (`vision.py`, `README.md`) — not a decode-parity harness,
  despite the directory name. The decode-parity surface is `tools/tp2/`, not `tools/parity/`. Worth knowing
  before someone inventories the wrong folder.

## 5. Suggested next moves, for the chair/agent4 to accept or reject — none executed here

1. Port #1 into our tree as a **named RED** against current `amd/main`, three shas in the commit (red-capture
   sha, tree sha at capture, expected post-fix sha blank) — the closure law's shape, satisfied without a new
   boot, since `1d0ff3c6` already supplied the device evidence.
2. Adopt #4 as a house rule for *any* future equality assertion (`verify_inputs_are_distinct` before `compare`),
   because it is the encoded form of tonight's most-repeated lesson and costs one helper.
3. Before designing the next determinism boot, read `:87-104` and `:263-268` and decide whether our single
   54-token prompt is a **strength** (it is the corpus the existing cross-boot classes were built on) or the
   knife-edge confound #5 describes. Either answer is fine; an unexamined one is what this inventory exists to
   prevent.
