# M1 / G-AMD-35 SPACED-CENSUS READ (agent3, reader seat per plan :37 — 2026-09-14 ~01:0xZ)

**Leg:** G-AMD-35 LEG 1.5 = M1, my spaced-census protocol (`M1_SPACED_CENSUS_protocol_agent3.md`,
chair-accepted `cfbc4af7`), booted by agent4. Bin
**`/home/chris/artifacts_bin/ninfer-serve_2a345b3048c1d6b3.bin`** — the same bin as G18d/G18e, verified
at my seat three ways, because I first had this wrong (see §4a): the runner pins
`BANK_SHA=2a345b3048c1d6b3…` with a pre-flight sha gate (`M1_spaced_census_agent4.sh:28-29,40`), agent4's
counts line echoes that sha, and the log's own creation time (19:30:47) is a DIFFERENT boot from the
`d3a0e738` process I saw at 19:23 (that was LEG ZERO, `A4L0_serve.log`, created 19:17:16). **M1 is
therefore a PRE-A-4 bin: the §2a era rule of my protocol does NOT apply to this leg**, which is also
why its argv is byte-equal to G18e — the point of the arm was one-delta-vs-G18e, not an A-4 test.
×25 `'Hi.'` greedy `max_tokens=32`,
`NINFER_GATING_TRACE` OFF, one delta = **2 s sleep between reps**. My witness instrument ran
**from before spawn through the cooldown** (258 rows), which repairs the coverage hole that limited
my own G18e read.

**HEADLINE: B2 FIRES. The spacing did not collapse the retries — it made them WORSE, and my
rank1-clock hypothesis took its own falsifier (B4) and lost. The lag is not machine-duty recovery;
it is coupled to the request sequence.**

## 1. The numbers (predicate named per column, per my 421-vs-434 lesson)

| datum | G18e back-to-back | **G19 M1 SPACED** | reading |
|---|---|---|---|
| retries, raw `grep -c 'AR-RETRY'` | 831 | **961** | **+15.6 % with 50 s of sleeps added** |
| retries, windowed (≤ last `[req N] done`) | 831 | 961 | post-boundary lines **0** in both, so here the two predicates coincide (they did NOT in G18c — name them anyway) |
| kind inventory (`-oE 'kind=[a-z]+'`) | 831 argmax / 0 kind-less | **961 argmax / 0 kind-less** | era: this bin names BOTH kinds, so 0 kind-less = 0 unknown emitters |
| **AR-ring retries** | 0 (inferred by totality) | **0**, field-terminated `kind=ar ` = 0 and `gen=` = 0 — but on a PRE-A-4 bin, where the AR emitter prints NO kind field, so `ar-fallback`/`gen=` is the correct read and **this is still the totality-rescued inference, not the new positive count** | AR ring silent in the spaced geometry, on the same footing as before |
| rank split | 796 / 35 (95.8 % r0) | **951 / 10 (99.0 % r0)** | one-sidedness **sharpened** |
| stamp median | 655 | **643** of 804 calls | late-census concentration unchanged |
| max try | 8 | **8**, zero at 9+ | K=16 untouched, third bin in a row |
| onsets | req #11, first served stamp 327 | **req #11, first served stamp 329** | **identical position** |
| decode tok/s | 4.80 → 2.00 (−58 %) | **4.80 → 2.30 (−52 %)** | same curve |
| prefill tok/s | 14.90 → 9.80 | 14.80 → 9.90 | unchanged |
| text-sha distinct | 1 | **1** — 25/25 at `348e77a1222dea7f`, same digest as G18d/G18e → **75/75 across three boots** | correctness untouched |
| `[AR-FAILOUT]` / `[ARTAG]` / `REJECT-candidate` / `tp2 worker error` | 0/0/0/0 | **0/0/0/0** | B5 did NOT fire — no carrier-class datum |
| MC31-H step lines / KAR | 1,660 / 32 ACCEPT | **1,660 / 32 ACCEPT** | arming equal to the back-to-back bin |

## 2. Branch outcomes, decided against the pre-declared table (branches fixed before the boot)

- **B1 (collapse ⇒ machine state): REFUTED.** 961 ≥ 600. And it did not merely persist — inserting
  50 s of idle made the count **rise by 130**. A recovery-shaped mechanism cannot get worse when you
  add recovery time. That is the cleanest disproof of the DVFS-recovery story I could have been
  handed, and it came from the arm I designed to test my own favourite hypothesis.
- **B2 (persist ⇒ cumulative in-server state, my domain): FIRES.** The strongest single number for
  it: **the onset is at the same request index AND the same stamp** — #11 / stamp 329 vs #11 / stamp
  327 — even though the spaced arm reached that point ~20 % later in wall time (126 s vs 105 s).
  Whatever turns the lag on is counting **requests/steps, not seconds.**
- **B3 (in between): also partly true, and reported as a split, not averaged.** Within-bin drift
  control (the nine zero-retry requests each bin contains, my method) gives
  `6223 + 74 ms/req` drift for this leg, and a raw **88 ms**/retry coefficient falling to **72 ms**
  controlled — inside the source's own 65-80 ms bounded-poll ceiling, fourth bin in a row to agree
  with it (G18c 78, G18d 72, G18e 79, **G19 72**). The *shape* survives spacing while the *count*
  grew, and both facts stay in separate columns.
- **B4 — MY OWN FALSIFIER FIRED AGAINST ME, and it is the finding I least wanted:** in the spaced
  arm the rank1↔rank0 clock asymmetry is **NOT** what it looked like at G18e. `corr(rank1 sclk,
  ordinal) = -0.491` (was −0.925) and `corr(retries, rank1 sclk) = -0.614` (was −0.944) — both
  correlations **weaken by ~40 %** once the geometry changes, and the witness now covers requests
  1-10 which G18e missed: **rank0 and rank1 run essentially TOGETHER at the start** (idle→busy
  transitions: 359/348, 300/300, 474/464 MHz — the two dies are within 3 % of each other while cold)
  and the mean busy ratio 0.873 is a *steady-state* offset, not a drift that tracks the retry curve.
  So, exactly as I pre-committed in `T3_G18E_CENSUS_READ…` §5 and the chair boarded into the M1 row:
  **my G18e clock↔retry correlation was substantially two monotone series coinciding. The clock
  story is VOID, not amended.** I say so in the row that carried it.
- **B7 (the clock channel agent2 added): read, and it DISAGREES with the retry channel — reported as
  a disagreement per its own instruction.** 26 captured idle→busy transitions show the governor
  climbing *on load* (mean first-busy 1032/981 MHz vs steady-busy 1140/994), i.e. **the sleeps did
  not let either die "recover" to a state that removed the lag**; meanwhile edge temperatures during
  the spaced busy windows sit at **78.5 / 78.4 °C**, ~6 °C COOLER than G18e's 84-85 °C, with the
  retries ~16 % HIGHER. A cooler box with more waiting is the third independent nail in the thermal
  story (after agent2's equal-edge-temp and power-cap exclusions).
- **B6 (instrument error): did not fire — but my tool came within one line of causing it, disclosed
  below (§4).** 25 aligned windows, alignment residual 0 s, 184 busy + 74 idle rows parsed.

## 3. What this datum licenses, stated narrowly

**The lag is a function of the request/step sequence, not of elapsed time or package state.** Three
things follow, and none of them is a patch:

1. **My domain re-opened, hypothesis-first.** Cumulative in-server state that grows across ~10
   requests *within one server process* and is indifferent to 50 s of idling: the arena/KV/slot
   history I have not been able to convict all night, the ring's own position arithmetic, or
   per-request allocation drift. The onset-at-fixed-stamp (327/329) is the new handle — it is a
   **number I can aim at**, and it is invariant across the one variable I have manipulated.
2. **Not a cooling/airflow/power story, and I want that on the record in my own voice:** three
   independent exclusions now (equal edge temp at 186 MHz gap; clock and power falling together
   under a 110 W cap; a cooler boot performing worse). Nobody should spend a grant on thermals.
3. **K=16 still stands** — third consecutive bin at max-try-8, zero at 9+; the chair's re-open
   trigger (`max try ≥ 12`) is not approached, and now not approached in a geometry designed to
   stress it.

The honest limit: **the spaced arm confounds "cumulative in-server state" with "cumulative
in-process step count."** The sleeps pause between requests, but the server never restarts, so
anything monotone in *steps served* — including legitimate work — survives spacing by construction.
What M1 killed is the time/duty story, not the arena story; and it cannot by itself distinguish
"arena holds more dirty state" from "we have simply taken 330 more steps". The next discriminator
has to break THAT degeneracy, and **a first version of it already exists on disk, un-banked as a
control: `G17x_serve.log` is a FRESH-process boot that ran 5 requests and was FLAT
(4.9 → 4.8 tok/s, 0 retries)** — so a clean process is not degraded at low step counts, and
`G18c`'s same-family fresh boot degraded from request 11 too. That pair says the onset is a
function of steps-in-this-process rather than of anything inherited from a previous boot, which is
B2-consistent but still cannot separate "~330 steps of accumulation" from "~10 requests of
accumulation" because both logs advance the two together. The arm that separates them is
**M4: fresh process, same geometry, ×15 with `max_tokens` raised so ONE request covers the stamps
where the back-to-back onset sits (~330).** If a single long request degrades at its ~step-330 while
a fresh ten-request sequence stays clean to request 10, the constant is **steps**, and the suspect is
per-step state (slot/arena/ring position). If the long request stays flat and only request-index ~11
is dirty, the constant is **per-request entry** (reset/prefill path), which routes the hunt straight
at the reset family. Zero build for either arm (existing `max_tokens` flag), one card, and it is the
chair's to board — I name it, I do not boot it.

## 4. Two instrument errors in MY OWN tools, caught at the read seat, disclosed not smoothed

1. `census_read_analysis.py` reported **`SAMPLER-ERROR: no usable on-clock samples` on a perfectly
   good witness file** — because my parser called `int()` on values my own sampler writes as floats
   (`1045.0`), raising on every row. The tool could not parse the output of the instrument by the
   same author, an hour earlier. Fixed (float-tolerant, loud-failure semantics preserved and re-
   verified: missing input rc=2, empty/headerless rc=1, good input rc=0). Had I not read the error
   as *my tool being wrong* rather than *agent4's witness being bad*, this leg would have been
   reported unreadable and B7 would have been waved off.
2. Three of my own verification one-liners on the fresh log were about to be published wrong:
   `grep -c 'kind=ar'` returned **961** — a substring match on `kind=argmax`, which would have
   announced "the AR ring retried 961 times", the exact opposite of the truth; the correct
   field-terminated form gives **0**, corroborated independently by the `gen=` field count (0).
   And an unscoped `grep -oE 'rank=[01]'` summed every line in the log (4209/3268) instead of the
   retry lines (951/10). Cause named so it doesn't recur: the fast grep on a fresh log, before the
   pattern is anchored. Agent2's "read the matched line" and agent5's "a claim inherits the weakest
   evidence tier it cites" both apply to me here, minutes after I adopted them.

### 4a. And a third: I nearly published the WRONG BIN for this leg, and the wrong era rule with it

After seeing a `ninfer-serve_d3a0e738…` process on the pair, I wrote this row's bin as `d3a0e738`
(agent4's A-4 candidate) and imported my §2a era rule ("`kind=ar` names both emitters in this bin")
as if it applied here. It does not, and the correction is three independent bytes rather than a
judgement: the runner pins `BANK_SHA=2a345b3048c1d6b3…` with a hard pre-flight sha gate; agent4's
counts line echoes that same sha; and the log timestamps separate the two boots — `A4L0_serve.log`
(d3a0e738, LEG ZERO) created 19:17:16, `G19spaced_serve.log` created 19:30:47. The A-4 bin's
`kind=ar` field is real and my §2a rule is correct **for the window legs that boot d3a0e738**; it was
simply the wrong rule for THIS leg, and reading a 961-line `kind=argmax`-only inventory as "the AR
ring named itself and said zero" would have overstated what this geometry proves. Consequence carried
into the table above: the AR-ring silence at M1 is totality-rescued inference, exactly as at G18c/d/e,
not the first positive count. The first positive count still awaits a `d3a0e738`-era boot.

The general rule, for me specifically: **an observation of a live process is not a provenance claim
about a log.** I saw a real pid running a real bin and attached it to the wrong artifact because both
were in the same lane at the same time.

### 4b. Postscript, third-party confirmation of the bin correction — my mistake became a mechanism

Two hours on, agent5 turned my protocol §2a premise into code on main (`4d5b4ee1`, woven into the
window manifest at `73ef3227`): the reader now **infers the era from the datum itself** — `kind=ar`
is emitted ONLY by the post-A-4 AR emitter, so a log containing it is provably post-A-4 and a
kind-less line there refuses; a log with no such line keeps the lenient reading. Their stated reason
for adopting it is **my git-ancestry false answer** (§4a): pins come from `artifacts_bin`, not from
the tree under your feet.

Re-running that reader against this leg is the independent confirmation my own correction wanted:

    read_census.sh G19spaced_serve.log  ->  961/961 bucketed, ar-fallback=0, rc=0   (LENIENT reading)
    read_census.sh G18e / G18d          ->  rc=0 each                                (five banked bins still read)

M1's log takes the PRE-A-4 reading, so the era is now established by a mechanism I did not write,
against a file I did not touch — not by my prose and not by a sha I copied from a runner. And the
consequence for the window stands in the strict direction: on the first post-A-4 leg, an
unattributable retry line cannot be bucketed silently even if every seat forgets the env exists.
Three of my own errors tonight (float-parsing, substring grep, live-process-as-provenance) each
converted into a permanent guard rather than an apology, which is the closure law working on its
author; this paragraph is the last of the three being paid forward.

## 5. Receipts and next moves

Read banked on lane `amd/t3-wip`; tip and `git ls-remote origin` output sent to the chair with this
row, per COMM law. Reproduce: `tools/census/census_read_analysis.py <G19 log> <G19 witness.tsv>` and
`tools/census/read_census.sh <G19 log>` (agent5's reader, era-aware). No cards claimed by this seat;
I touched nothing but files and `/sys` reads.

**For the flip and the 4-card window, three carry-forwards, each named:**
- **`rb n=` remains inadmissible as branch evidence** (my §2 pool-arithmetic prediction test), and
  agent5's F-C correction stands with it: at world=4 a reading table must expect **presence**
  semantics, not partner semantics — no pairwise complement exists in a pair-only one-shot family.
- **75/75 text identity across three boots/two geometries/two dial states** is the field-scoped D.4
  datum (text-field, greedy, world=2, same prompt). Still not bit-exactness; still licenses no rate
  baseline off any census — M1 is now the proof that a census's rate curve is not a property of the
  workload alone.
- **The absorb price (72-88 ms/retry depending on control) is not a permission constant.** It prices
  K and crossing policy; it gates nothing.

— agent3 (pi `01a09cdc-ceb0`). Reader seat: M1 read complete, B2 fires, my own clock hypothesis
convicted by the arm I designed to test it, and the hunt re-seats on in-server state with
**M4 (fresh-process, same geometry)** as the named next discriminator, awaiting the chair's boarding.
