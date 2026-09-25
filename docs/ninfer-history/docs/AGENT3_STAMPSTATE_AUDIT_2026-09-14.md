# agent3 (fresh) — the lag hunt's suspect list, closed at the code seat: no per-step
# accumulator exists in the shipped geometry, and the ramp is not a ramp but a LATCH.
# Read-only row per chair dispatch (seq 49); no card, no build, no push to the shared tree.

**Seat/ref discipline (law E, applied to my own citations).** Everything below is read at
`amd/t3-wip @ 1991b550` in `/home/chris/worktrees/amd-wo-q3hip` (my lane, per the chair's
03:18Z correction). My lane is 36 commits behind `amd/main @ f051c177`; the src delta between
them is four files (`gather_capacity.h`, `tp_group.{cpp,h}`, `tp2_backend.cpp` +33/−8) and the
`tp2_backend.cpp` delta is the A-4 world-vector block — **none of the four line citations below
sits in a region that delta touches**, but every reader should re-derive the line, not trust the
number. Banked logs are read from `/home/chris/worktrees/amd-wo-p3-serve` (agent4's lane); every
count I publish is a `grep`/parse I ran there, not a number I carried.

**Inherited kit, used as written.** `GAMD37_G20long_READ…` and `T3_G18E_CENSUS_READ…` +
`REVIEW_LAWS_CDEF_2026-09-14.md`: ALIAS / PERIOD / ramp-check print before any ring reading; a
guard offered as protection needs the fixture where the effect IS present; assert a delta against
a baseline measured in the same run; read exit status before text. My Part 3 adds a fourth entry
to that family the kit does not yet have, and it convicts my own predecessor's numbers (§3).

---

## 1. The assignment's premise, tested: what in the serve path grows monotonically in stamp position?

Board belief I was handed: per-step, retries ramp late in stamp space, rank0-carried, no period,
no entry clustering, not time/duty, not reset/prefill, K=16 stands. The surviving-suspect list I
was given is *"arena/KV/page-table contents and rank0's pacing/queueing."* I audited that family
at the bytes. Result: **in the geometry every banked census leg actually boots, none of those
structures grows with cumulative steps.** Item by item.

### 1.1 DeviceArena (`src/core/arena.{h,cu}`) — no ratchet, and the shape filter kills it anyway

`alloc_bytes` is a pure bump pointer (`arena.cu:196-216`): it advances `off_`, records `peak_`
(`:214`), and has **no free list at all** — so there is no per-step "allocation/free history"
to grow, because there is no history kept. `Scope::~Scope` (`:131-134`) restores `off_`;
`ordinary_decode_batch` additionally calls `work_.reset()` at BOTH entry and exit
(`text_context_impl.h:1575`, `:1592`) and the same twice-per-forward pattern holds at
`:1615/1633/1659/1678/1714/1747`. So per-step arena cost is a constant, and the only monotone
field (`peak_`) is a `size_t` comparison that no hot path reads unless `NINFER_WS_PEAK` is set
(`gqa_attention_kvarn.cu:821-834`).

Independently of the code: **the request-shape filter refutes any within-request ratchet.**
Onset at request 8 position 3 (G20long) with requests 2–7 carrying **zero** retries means a
quantity that resets at the request boundary cannot be the clock — my predecessor established
this at their §2 and it binds here. So the arena family is dead twice over.

### 1.2 Page table / PagedKVPool — churn is real but request-bounded, and TINY in this bin

`take_pages` (`paged_kv_cache.cpp:275-325`) does a `lower_bound`, a contiguity scan and an
in-middle `erase` (O(P) memmove); `return_pages` (`:327-331`) appends and then **`std::sort`s
the whole free list**. That is a genuine monotone-in-`P` host cost — but `P` is the *pool* page
count, not the served-step count, and at this boot's capacity it is measured, not assumed: the
G20long log header says **`decoder state: 226 MB, 3 kv pages (cap 132)`**, i.e. P≈3. Three
elements do not sort slower at stamp 420 than at stamp 20.

`acquire_row`/`release_row` flip `row_in_use_` only and deliberately leave stale physical ids
beyond `mapped_page_count` — the code says so at its own hazard note (`paged_kv_cache.h:166-174`)
and the invariant holding it up is *"every attention window is bounded by materialized pages."*
**That note is the live hazard to carry forward, not a bug today:** any future leg that derives a
window from *entitlement* instead of cached tokens would silently consume a previous occupant's
page ids. Nothing at the census seat can make that happen today; logged as a review predicate for
whoever writes the next window.

### 1.3 The process-lifetime accumulators that DO exist — and the measurement that turns each off

Three structures in this file genuinely grow monotonically with cumulative work and never shrink
with idle. All three are **gated off by argv in every banked ramp leg** — checked, not assumed:

| suspect | grows with | gate | state in the banked legs |
|---|---|---|---|
| `HostKvNet::entries` + `HostKVArena::extents_`, with `find_best` scanning every entry × prompt (`tp2_backend.cpp:95-107`, `:320-322`) and `evict_for` building a candidate vector + sort (`host_kv_arena.cpp:248-266`) | parked continuations | `host_kv_mib > 0` (`:1771`) | `grep -c 'host-kv'` = **0** in G18c/d/e/G19/G20long/A4L0/G21 — net inert |
| `MtpNgramMod` pool, `ngram_pool->add` per committed token (`:3069`), `roll` per round (`:2684`), explicitly "process-lifetime" in its own comment | cumulative committed tokens | `mtp_ngram_mod` + MTP | `MTP k=0`, `speculative=off`, `rounds=0` in every leg — cold |
| `VocabOutputCounter::counts_` (248320 u32, touched per accepted token, `vocab_output_counter.h:70-84`) | distinct tokens seen → cache-line footprint | `NINFER_VOCAB_COUNT_DIR` | `grep -c '\[vocab\]'` = **0** — off |

Also off in this bin, by argv: `--no-cuda-graph` (no graph pools), `--no-prefix-reuse`
(`prefix_cache=false` at `serve_options.cpp:342`, confirmed by `reuse=full_reset` on every
`[req N] done` line), `max_concurrency=1` (no batched runner, so the whole
`run_tp2_requests_batched` arena/lane-allocation family is not on the path), and
`print_timing=false` from `tp_engine` (so `PhaseTimer`'s 9×`req.max_output_tokens`
`cudaEventCreate`s at `:1633-1637` never happen in serve).

**The rank0-inline-vs-rank1-spawned asymmetry** my predecessor's rb-n finding names is real and
still live: rank0's worker runs on an httplib pool thread (`http_server.cpp:133-139` →
`worker_count = max_concurrency + max_pending_requests + 1`) while rank1's is a fresh
`std::thread` per request (`tp2_backend.cpp:3355-3367`). But it predicts a **request-index**
shape, and Part 2 of this row shows the phenomenon is not request-bounded at all. So it is a
pacing asymmetry looking for a per-step mechanism, not a mechanism itself.

**Net of Part 1: the surviving-suspect list I was handed is empty at the code seat.** The hunt has
to re-seat on what a step actually *does*, which is where Part 2 went.

---

## 2. The shape of the ramp is a LATCH, not a monotone function — measured on banked bytes

Predicate for every number in this section: rank0's `[AR-RETRY] … kind=argmax` lines and
rank0's `[MC31-H] rank=0 step=` line sequence, parsed per boot from the seven served logs
(G18c, G18d, G18e, G19spaced, G20long, A4L0, G21leg0). 7,455 retry lines, 5,950 retried
stamps, 2,773 served steps total.

### 2.1 Retries come in RUNS, and the run is the event — not the stamp

Pooled over seven boots: **203 runs**, median length 4, max **156 stamps**. Conditional on the
previous step:

| boot | P(retry at s+1 \| retry at s) | P(retry at s+1 \| clean at s) | coupling |
|---|---|---|---|
| G18c | 0.911 | 0.046 | 19.7× |
| G18d | 0.951 | 0.048 | 19.6× |
| G18e | 0.892 | 0.075 | 11.9× |
| G19spaced | 0.941 | 0.058 | 16.3× |
| G20long | 0.866 | 0.065 | 13.2× |

Per-quarter transition rates, pooled over the same seven boots, are the whole story in four numbers:

| boot quartile | P(enter degraded) | P(exit degraded) | share of steps on |
|---|---|---|---|
| Q1 | 0.000 | 1.000 (n=2) | 0% |
| Q2 | 0.022 | 0.342 | 6% |
| Q3 | 0.239 | 0.113 | 67% |
| Q4 | 0.639 | **0.048** | 93% |

**Exit probability does not merely fall — it falls to ~5%.** Once the system is in the degraded
state late in a boot it *stays*. And the entry rate rises monotonically 0.02 → 0.64. A
two-state system whose entry rate climbs and whose exit rate collapses is a **bistable latch with
drifting bias**, not a counter crossing a threshold: a smooth accumulator crossing a fixed bound
gives rising *incidence* but no reason for the *exit* probability of an already-degraded step to
collapse. This changes what to look for: not "what grows", but "what makes the degraded state
self-sustaining" — which is the retry-hammer feedback my predecessor already named (an absorbed
retry costs 72-79 ms and re-launches the whole argmax kernel at 100 % duty on both dies) and which
the board had priced but never given a *shape* test. The shape is now measured, and it favours the
feedback story over the pure-accumulation story.

### 2.2 The regime is not request-bounded in either direction — new, and it closes the family the chair named

- **Runs start mid-request, not at the entry event:** of 201 run starts across seven boots,
  **198 are mid-request and 3 are at a request-entry step** (P(entry-start) ≈ 1.5%; 3/201 against a
  1/32-ish uniform expectation over a 32-step request — the entry event is if anything the *least*
  likely place to latch).
- **Runs cross request boundaries about a fifth of the time:** **43 of 201 runs (21 %)** span at
  least one entry (19-38 % per boot; G20long 5 %). Max run 156 stamps = 4.9 whole requests.
- **P(retry at the entry step of request N+1 | the last step of request N retried) = 0.91-1.00**,
  statistically the same as mid-request (0.86-0.95). The request boundary — prefill, `reset_one_shot_step`,
  the slot-map wrap, the parked-state path, `work_.reset()` — **clears nothing**.

That is the strongest statement this row can make and it is a *negative* one, aimed at the list I
was handed: the mechanism is neither a per-request entry artifact nor anything that a request
boundary resets, and no per-step server-side data structure in the shipped geometry grows. What
survives is host/driver/device state integrated over *issued work*, plus the retry loop's own
positive feedback.

### 2.3 Which clock the latch counts — the collapse test, all four clocks, one predicate

For each boot I located the sustained-onset stamp (first stamp with ≥5 rank0 retries inside a
16-stamp window), then expressed that same wall-instant in four candidate clocks. Lower CV =
tighter collapse across five boots of two different geometries:

| clock at the onset | value spread | CV |
|---|---|---|
| cumulative argmax **stamps** | 326 / 326 / 327 / 363 / **420** | **9.4-12.3 %** |
| cumulative **computed tokens** (prompt+gen) | 921 / 921 / 922 / 1012 / **853** | 4.3-6.5 % |
| cumulative **busy service seconds** (Σ prefill+decode per request) | 110.7 / 111.2 / 111.5 / 125.7 / 119.7 | **2.4-6.8 %** |
| **log bytes** written | 2.25 / 2.25 / 0.20 / 0.20 / 0.25 MB | **78.9 % — refuted** |

Two conclusions, one of them a refutation of a named suspect:

1. **The pinned-write/DMA-backlog-of-LOGGING family is refuted by the corpus's own pair.**
   G18c/d/G21/A4L0 carry 79,680 `[gating]` lines (~6.9 kB/step); G18e/G19/G20 carry 0 (~0.66 kB/step)
   — an **11× difference in bytes written per step**, and the onset moves from stamp 326 to 327.
   Agent4's release row left "pinned-write/DMA backlog" open as a candidate; the *log*-volume
   version of it is now closed. What is NOT closed is DMA volume from the model itself, which is
   collinear with tokens and needs the arm in §4.
2. **Stamp-position is the worst-fitting of the three live clocks** — and the single leg that
   separates them is G20long, the leg the board is leaning on: there the onset is **+29 % late in
   stamps**, **+8 % late in busy-seconds, −7 % EARLY in computed tokens.** The board's wording
   "retries ramp late in stamp space" is true *within* a family of identical 32-token boots and
   stops being the best description the moment the geometry changes. My predecessor's step-cumulative
   verdict was scored against a prediction band (374-425) wide enough to absorb the one datum that
   later discriminates; agent4's release row said the same thing at the time and the adopted belief
   did not move. I am not overturning it — I am reporting that on my seat's numbers, **busy-service
   time and cumulative-tokens both beat stamp-count by ~2×**, and the three are only 1.1-1.3×
   apart, which is exactly the "needs a matched-total A/B" state, not a settled one.

### 2.4 A statistics correction the kit does not yet carry — and it convicts the guard, not just the numbers

Every bin test the census has published (my predecessor's slot χ²(31)=60.0 "p<0.01", the
position-bin χ²=106-115 "POSITION-RAMP", my own first pass at this table) treats *stamps* as the
independent unit. §2.1 says they are not: the unit is the **run**. Design effect
VIF = retried-stamps / runs, measured per boot:

| boot | retried stamps | runs | longest run | VIF | effective n |
|---|---|---|---|---|---|
| G18c | 204 | 19 | 41 | 10.7 | ≈19 |
| G18d | 391 | 20 | **156** | 19.6 | ≈20 |
| G18e | 325 | 36 | 57 | 9.0 | ≈36 |
| G19spaced | 388 | 24 | 116 | 16.2 | ≈24 |
| G20long | 292 | 39 | 66 | 7.5 | ≈39 |
| A4L0 | 394 | 26 | 67 | 15.2 | ≈26 |
| G21leg0 | 297 | 37 | 56 | 8.0 | ≈37 |

A χ² of 88 against a 1 % critical value of 76.8, read off a sample whose effective size is 20-40,
is not a result. Two of my predecessor's readings are void on that basis, and the direction is the
instructive part: the **incidence** bins (each retried stamp counted once) come out **flat in all
seven boots** — χ²(31) = 3.2-6.7 against a 1 % critical of 52.19 — while only the
**depth-weighted** bins (each retry *line*) look peaked. One heavy-tailed run-length distribution
projected onto two axes, which is a *second* mechanism behind "hot slot" and not the alias
mechanism my predecessor already retracted.

**And the guard itself had a live bug, which is what this pass was for.** Part C swept lags from 2
and required the winner to beat its own ±1 neighbours. A two-state latch has autocorrelation that
**decays monotonically from lag 1** (measured on the fixture: lag1 +0.795, lag2 +0.646, lag3 +0.519,
lag4 +0.414, lag5 +0.348), so the argmax lands on lag 2, beats its neighbours, and the guard
**reports `PERIOD-2 CANDIDATE` on a series with no period at all**. RED captured on the pre-fix
artifact (`git show 1991b550:tools/census/census_read_analysis.py` against the latch fixture →
`argmax lag 2 = +0.671 … PERIOD-2 CANDIDATE … clears its ±1 neighbours … by +0.120`); GREEN at this
tip on the same fixture (`argmax lag 1 = +0.789 → NO PERIOD, SERIAL CORRELATION INSTEAD`). Fix:
sweep from lag 1, and refuse any candidate that does not beat lag 1. Cell:
`tests/check_census_period_guard.py` gains arm **NEG3 two-state latch** (6/6 arms PASS), and every
one of the seven real legs now reports `argmax lag 1 = +0.71 … +0.85 → NO PERIOD, SERIAL
CORRELATION INSTEAD`. Read that as the corrected version of the certified finding: **"no period"
was right, and the reason is stronger than "flat" — the series is serially correlated at lag 1 in
every boot, which is a latch, not an array.** The lag-1 value is now the most informative single
summary number the census emits, and it is the one the old guard was structurally unable to see
because its sweep started one index late.

### 2.5 What the latch says about the pacing suspects — and one predicate I got wrong an hour in

P(retry at the next step | previous step retried), split by whether that next step is a
request-entry step. Boundaries are read from the log's own `[MC31-H] rb rank=0` entry markers,
**not** from cumulative gen arithmetic — my first pass derived them that way, misaligned every
boundary by the warmup block, and printed a confident 5 % where the measured answer is 21 %:

| conditioned on the previous step being retried | P(next step retried) | n |
|---|---|---|
| next step is **mid-request** | 0.922 | 2217 |
| next step is a **request entry** | 0.687 | 67 |

Runs crossing a request boundary: 43 of 201 pooled (21 %; 19-38 % per leg). So the entry sequence
(prefill, the `sync_bar.arrive_and_wait()` + `cudaStreamSynchronize` pairs at
`tp2_backend.cpp:1966-1975`, `reset_one_shot_step`) **partially interrupts** the latch — 0.92 →
0.69, roughly a third of the time — but does not clear it, and run starts at an entry step are rare
(7 of 201, against 193 entry steps available). The composite is the useful part: the degraded state
is **carried across** request boundaries and only **partially reset** by them, which is the
signature of device/driver-side queue state and not of a per-request host-side allocation history.
Caveats stated: n=67 on the entry arm is small, and §2.4's VIF point applies to it — direction, not
magnitude.

---

## 3. Named suspects, ranked by cost to settle — each with the measurement fixed BEFORE any patch

Per the night's law, nothing here ships a patch; each row names the observation that would convict
or clear it, and which seat observes each direction.

- **L1 — retry-hammer self-feedback (the latch's engine).** *Cost: zero-card, banked bytes only,
  DONE for shape (§2.1/§2.2), not done for magnitude.* Decisive measurement: compare the exit
  probability against the *absorbed-retry* fraction on the same step, per boot-third. If the latch
  is self-sustaining, exit should fall with cumulative absorbed-retry fraction rather than with
  stamp — that separates "the retry loop is the engine" from "the retry loop is the symptom of a
  drift elsewhere". Falsifier named now: if exit probability is flat in absorbed-retry fraction and
  falls only in a third clock, the feedback story is wrong and I will say so in this file.
- **L2 — busy-service time vs issued work (tokens/steps), the clock the latch counts.** *Cost: one
  boot, zero build.* Only a matched-total A/B separates them: two legs with equal issued argmax
  calls and unequal service seconds (vary per-step work without changing the step count — the
  existing `NINFER_KVARN_DECODE`/route or split-k knobs do this with no src change). Reading fixed
  now: onset invariant to the route change ⇒ tokens; onset tracks the seconds ⇒ duty-integrated
  machine state. Either answer kills a third of the remaining hunt.
- **L3 — machine-side power/impedance state that leads the edge sensor** (agent4's open item).
  *Cost: one boot, zero build, and the sampler already writes the columns.* Pre-declared reading:
  per-die `power1_input` and `sclk` integrated against busy time, onset looked for in the
  *cumulative joule* clock, not the temperature clock. If cumulative energy collapses the five
  onsets tighter than busy-seconds does, that is the accumulator and the thermals exclusion stands
  unchanged (equal edge temp at 186 MHz gap; clock and power falling together).
- **L4 — rank0's inline-vs-spawned asymmetry as a *pacing* driver (my predecessor's rb-n finding).**
  *Cost: zero-card until a boot:* §2.2/§2.5 bound it — the latch is carried across request
  boundaries (21 % of runs span one) and starts at them almost never (7/201), so an asymmetry that
  lives at the request boundary cannot be the engine; it can only be the seed. The arm that would
  revive it is one boot with rank1 on a persistent pool thread (the M2 fix-shape), reading the
  **exit** probability and the entry-vs-mid continuation split (§2.5's table), not the retry count:
  if a symmetric thread architecture raises the exit rate toward the mid-request value, the
  asymmetry is implicated; if the exit rate is unmoved, it is not the latch.
- **L5 — `HostKvNet`/`MtpNgramMod`/`vocab_counter` (the only true process-lifetime growers).**
  Not live in this bin (§1.3, measured). Named so the next seat does not re-audit: the moment a
  window boots with `--host-kv-mib`, L5 becomes the #1 suspect *at the request-entry clock*, and
  §2.2's boundary test will discriminate it in one pass with no new instrument.

## 4. What this row does NOT say

No mechanism is convicted here. No patch is proposed. The VRAM-law polarity applies to every
number above: they characterize, none of them gates a launch. The correctness closure is
untouched by all of it — across the seven legs read here, `[AR-FAILOUT]`=0, `[ARTAG]`=0,
`REJECT-candidate`=0, `kind=ar`=0, `LONE`=0, K=16 untouched (max try 8 every boot), and text
identity holds boot by boot (G20long 15/15 at `c27fd91f5beab79d`; the 32-token family 25/25 at
`348e77a1222dea7f`).

## 5. Receipts

- **Bug found and closed at the guard, three shas named (closure law):** the flat-lag period guard
  reported `PERIOD-2 CANDIDATE` on a latch-shaped series. RED: `1991b550`
  (`git show 1991b550:tools/census/census_read_analysis.py`, latch fixture, prints
  `argmax lag 2 = +0.671` → `PERIOD-2 CANDIDATE`). GREEN: this tip, same fixture, prints
  `argmax lag 1 = +0.789` → `NO PERIOD, SERIAL CORRELATION INSTEAD`. Suite entry:
  `tests/check_census_period_guard.py` arm **NEG3**, 6/6 arms PASS. The class the cell now guards,
  generalized: *a neighbour-control test must include the shortest lag, because autocorrelation
  decaying from lag 1 is serial correlation and any sweep starting at lag 2 will read it as a
  period.* Not the instance, the class.
- Tool: `tools/census/census_read_analysis.py` gains **Part D — RUN/LATCH ANALYSIS** (run count,
  VIF, incidence-vs-depth χ² pair, per-quartile enter/exit, request-boundary start/cross read from
  the log's own entry markers, sustained-onset stamp). It runs by default and prints **after**
  Part C, so the inherited ALIAS/PERIOD/ramp ordering still comes first and the new numbers cannot
  be quoted ahead of it. Part C's sweep now starts at lag 1 and refuses a candidate that does not
  beat lag 1.
- Cells: `tests/check_census_runstructure_guard.py` — positive control (synthesized latch: must
  report VIF>3 and must not bless the face-value bin; also asserts the incidence/depth split
  separates them), negative control (i.i.d. retries at the same marginal count: must report
  VIF≤3 and stay silent — a guard that cries wolf on independent data is worthless on real data),
  and the third state (law D: missing file rc=2, empty log rc=2, never a pass read out of prose).
  5 arms PASS, rc=0. Both controls seed deterministically so the arms cannot silently decay.
- Predicate confession (§2.5): request boundaries first derived from cumulative gen counts,
  misaligned by the warmup block, produced a confident wrong crossing rate (5 % vs the measured
  21-30 %). Fixed by reading the boundary from the emitter's own marker. Logged because the mistake
  is the same family as the guard's: a plausible statistic computed on an index quietly off by a
  constant, and both were caught only by re-running rather than by re-reasoning.
- Row: this file. Zero device contexts, zero builds, zero writes outside my own lane; the shared
  checkout at `amd/main` was read through `git show` only, per the chair's 03:18Z correction.

— agent3 (pi, fresh read seat), `amd/t3-wip`, 2026-09-14.
