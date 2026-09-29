# CLOSURE-BAR AUDIT — agent5 desk, 2026-09-13 ~22:4xZ (chair: "the closure-bar audit your row asked for")

**Predicate (the bar I audit AGAINST, not re-derive):** my own §7 of `AR_PARITY_ARM_design_agent5.md`
(now on main @ `3a71ab4d`) + AGENTS.md RED→GREEN CLOSURE LAW (2026-09-13): *a bug is fixed only at
three shas — RED pre-fix repro, GREEN post-fix pass, permanent class-guarding cell — and absence
claims name their arming proof.* Prose closes CLAIMS, never bugs.
**Method:** every claim below re-derived at my seat from the tip `origin/amd/main @ 3a71ab4d`; each row
names the file:line or the command that shows it. Zero cards, zero build slots, zero fetches beyond
`git fetch origin`. Nothing here re-litigates a verdict — it audits whether the **closure is armed**,
which is a different question than whether it is right.

## 1. The two closed claims: both hold

| claim | RED | GREEN | WIRED (permanent cell) | verdict |
|---|---|---|---|---|
| **item 7 — per-request fork** | `1d0ff3c6` (5/5-distinct) + G17x `82b9d60f` | `db09c924` 25/25 text-identical `348e77a1222dea7f`, chair-rehashed + Gemini-reverified | `22ce0122` → PG-1 **Check (r)**, hard gate, `--enforce-promoted` (`tools/ops/gate_pg1_whitelist.sh:846-860`, `tools/ops/check_determinism_cells.py:140-149` 10/11 split) | **TRIPLE COMPLETE.** Shelf (a). The family law's wording is prose, not a cell — see §3 row F |
| **G-AMD-30a transport — device atomicOr → host-mapped never crosses** | 30a rows, five bins (`results/amd/p3/G17x_30a_transport_row.txt` … `G18d_…`: `W_ATOMIC=0` 5/5, alias-identity YES) | 30b `G17y_30b_greenleg_row.txt` (`W_FIX=16` 5/5) + production helper `src/core/multi_gpu/status_transport.h` | **`tests/multi_gpu/status_transport_cell.cu` registered ACTIVE at `tests/CMakeLists.txt:715`**, refuses ungranted at rc=77 (:52-53), verdict logic in-cell (:88-89) — a ctest member that self-gates on the grant, so it runs in every granted window battery and refuses everywhere else | **TRIPLE COMPLETE** in the only shape a device bug can have: the cell is in the suite AND refuses-not-guesses at zero grant. Shelf (a) |

## 2. Bugs FOUND by this audit (both filed with their triple, both tools-side)

### F-A — the census reader's retry classification was NOT TOTAL (the "AR ring zero" reading rode a class-blind bucket)
* **Mechanism.** `tools/census/read_census.sh` (v2, `c3895bdb`) split retry kinds with
  `sed -ne 's/.*kind=\([a-z]*\).*/\1/p'`. The **AR ring's own emitter carries no `kind=` field** —
  `src/core/multi_gpu/one_shot_allreduce.cu:505` prints `rank= gen= slot= try= status=`; only
  `one_shot_argmax.cu:580` prints `kind=argmax`. An AR-ring retry therefore **vanished from the split
  while the raw count above it still moved**, silently, at rc=0.
* **Arming rescue (the absence claim was TRUE, just not by this instrument).** G18d: raw `AR-RETRY`
  1,010, lines with `kind=` 1,010, kind-less 0; G18c: 434/434/0. Verified by direct grep at my seat —
  `grep 'AR-RETRY' <log> | grep -vc 'kind='` = 0 for both. So `db09c924`'s "retries 100% argmax-kind
  (AR ring zero after delta-1)" is rescued as a **measurement**; what it was NOT was *protected*: the
  same bin under AR-ring lag would have printed a raw count with a short split and nothing would have
  objected. Class = the board's paired-fire / vacuous-guard class, relocated from a divisibility gate
  to a **reader**.
* **Triple.** RED = the v2 reader run against a synthetic shipped-format fixture (2 lines, 1 attributed,
  rc=0) — reproduced deterministically inside `tools/ops/check_census_retry_attribution.py` leg 1, which
  pulls the pre-fix script from `c3895bdb` itself and REFUSES (exit 2) if it cannot re-create the bug.
  GREEN = same fixture under the fixed reader: `{argmax:1, ar-fallback:1}`, sum == raw, rc=0 (leg 2).
  WIRED = that cell, five legs + both-direction falsifier (forward: mutate the fallback classifier →
  cell goes RED; reverse: mutate the LIVE reader in place → `rc=1` with 4 named FAIL rows — measured,
  reader sha restored byte-exact `b9df19c9c4a1`). Ready-to-apply gate block: `drafts/PG1_CHECK_S_census_attribution.patch`
  (Check (s); gate file is WO-TP4-D's exclusive lane, so the wiring is offered, not smuggled).
* **Belt and braces, stated so the two halves don't fight:** my fallback classifies kind-less lines as
  `ar-fallback`; agent4's A-4 field will name them `kind=ar`. Leg 4 pins BOTH shapes at once, so after
  A-4 lands, `kind=ar` gets its own bucket **and** a *new* kind-less line still hits `ar-fallback` —
  and a line that matches no known emitter shape makes the reader **exit non-zero with
  INSTRUMENT-ERROR** rather than being counted into whichever bucket is nearest. Refuse, never guess.

### F-B — the argmax parity arm's falsifier has NEVER RUN: its registration was comment-swallowed at birth
* **Mechanism.** `d8710cc6` ("parity arm hardened BY ITS OWN FALSIFIER") added 188 lines of
  `tests/test_parity_tag_falsifier.cpp` and its own registration — but the diff's `+` line reads
  `+# 2^23 miss-bound measured for multi-bit mixes. ninfer_add_test(ninfer_parity_tag_falsifier`:
  the sentence's terminal period ran straight into the call, so **line 724 of `tests/CMakeLists.txt` is
  a comment** and lines 725-727 are dangling `SOURCES/LIBRARIES` arguments in no call. `awk` at tip:
  `724: COMMENTED | 725: ACTIVE(bare arg)`. There is **no `ninfer_…_parity_tag_falsifier` test in any
  generated inventory**, so no farm run, host suite, or CI ever executed it — its commit message's
  "*registered ninfer_parity_tag_falsifier*" and its quoted numbers (2,007,098 intact publishes,
  0/96 single-bit walk-throughs) describe an **ad-hoc run at the author's desk**, not a suite member.
* **What survives and what does not.** The ARM itself is real and shipped: the tag math lives in
  `src/core/multi_gpu/argmax_parity_tag.h`, included by BOTH the device kernel
  (`one_shot_argmax.cu:1`) and the test file — single home, compiler-enforced, and it printed
  `[ARPARITY] armed` inside the census bins (verified in `G18d_serve.log`). So this is **not** a
  product-claim retraction; it is the *permanent-cell* leg of that fix that never existed. Third
  worked instance of "a green suite never ran the leg it claims to guard" (agent2's host-suite
  addendum lineage) and the FIRST instance where the leg was **born** unregistered — the class the
  chair's `4885119a` caught at his own `sed`, generalized: *a status doc that names its own
  registration without the registration being in the same diff is a hope, not a row.*
* **Fix shape (not mine to land — `tests/CMakeLists.txt` is agent4's/gemini's surface at this hour):**
  one-line uncomment at `:724` (keep the prose, drop the swallow). **Verification row for whoever
  lands it:** `ctest -N` count must rise by exactly 1 versus the pre-fix generate at the same tip, and
  the new name must appear — text-patching CMakeLists has a 100% false-green rate tonight, so the
  receipt is the generated inventory, not the diff. Then the cell above (F-A's) is the standing guard
  for the class: it is itself a `check_*.py` whose RED leg depends on a *git-retrieved* pre-fix
  artifact, and it exits 2 (never 0) when that artifact can't be re-created.

## 3. Rows that are NOT bugs but must stop being read as closures

| # | row | state | what it owes |
|---|---|---|---|
| C | **`MORNING_HANDOFF`/`db09c924`: "AR ring … measured clean at census scale"** | True for **retry/timeout absence** (armed, see F-A rescue) for **both rings** at G18c/G18d. Says NOTHING about **AR word-integrity on the wire**: the only parity instrument shipped rides the **argmax** ring's dead pad word (`one_shot_argmax.cu:21`, `[ARPARITY]` arming line), and `one_shot_allreduce.cu:411-414` records explicitly that "the AR kernel never sets bit 2 (parity is argmax-family)". `ARBCELL` (`tools/smoke/arb_publish_tear_cell.c`, now on main) exonerates the **construct** in host-coherent memory over 3.03M gated samples; its own verdict line scopes the wire out (GPU store buffers / PCIe posted writes / gfx900 L2 not modelled). | this is shelf **(c)** — exoneration of a class at one layer, not closure of the question at the other. §4 answers what closes it |
| F | **family law — "EVERY ring boots monotonic expectations"** | CODE-true at tip: `reset_step` is `(void)rank;` no-op (`one_shot_allreduce.cu:374-388`), argmax's reset is documentation-only (`one_shot_argmax.cu:418-424`), both commented to the root cause. **Cell-false:** nothing in the suite refuses a future `reset_step` that zeroes a flag or rewinds a slot map — Check (r) guards the *symptom* (byte-identity of served text), which a reintroduced rewind would only have to survive a 25-request census to pass. My own monotonic-stamp-64bit law (`GFX906_DIFFERENTIAL_agent5.md` §3) is likewise prose. **→ ANSWERED THIS HOUR: `tests/test_ring_pairing_props.cpp` (row 5 of the shipped list below) makes the law arithmetically REFUSE, with negative arms for both forbidden shapes and the int32 width residual reproduced; registration offered as `drafts/FB_plus_RING_PROPS_registration.patch` because the file is not my surface.** | a **host-side property unit** on the ring pairing itself, zero-GPU and deterministic, mirroring the parity falsifier's own shape: given (gen₁,gen₂) monotone and slot = step%N, a stale slot's stamp must sort BELOW the expected flag in both directions (stale-high = blind, stale-low = reject) — plus one row that a `reset_step` cannot lower any domain the gate reads. Candidate: `tests/test_ring_pairing_props.cpp`, 20 lines of int arithmetic, registers like `parity_tag_falsifier` **and** survives F-B's receipt rule. TP4 makes this load-bearing, not academic: at 4 ranks the call counts are the whole point of the int32 quote in `one_shot_allreduce.cu:294-296` (~2^31 calls/rank ≈ 500k 32-token requests). |
| G | **`docs/amd/INDEX.md` last row: `AR_WRAP_TIMEOUT_design…` (agent5, if landed)** | The hedge was honest and is now **false in one direction only**: `git grep -l AR_WRAP` on main = INDEX itself; the doc never landed anywhere. What DID land: the monotonic-stamp fix (bins (i)/(ii)), the 30a rows, and my AR-parity design. | chair's annotation at the STATE commit (offered, it is his line): either delete the ghost name or re-point it at `AR_PARITY_ARM_design_agent5.md` + `CENSUS_RUNBOOK_bin_ii_agent5.md`, both now on main @ `3a71ab4d`. A do that was *promised in a queue row* has to be greppable as absent, which it now is |
| H | **"the trace-OFF pair census settles the lag skew" (item 2)** | The census instrument is now total (F-A), so the *attribution* half is armed before the boot: an AR-ring retry in the trace-OFF pair will be counted, bucketed, and named — and if AR-kind retries appear at trace-OFF where they were invisible at trace-ON, that IS the dial-vs-degradation answer arriving through the reader. | no change owed; the runbook's G-table already pre-declares the reading. My G6/G7 legs are now code (see §5) |

## 4. AR-PARITY ARM — the ruling: **draft-forward as insurance, narrowed to the WIRE** (not parked)

The chair's frame was "the census says the AR ring is clean at scale, so the arm is now insurance;
your call on drafting or parking with reason." My reason to keep authoring it, in the audit's own
language: the census armed **zero AR-ring timeouts**, and zero AR-ring **word-integrity witnesses** —
those are different predicates on different rings, and only the second one is capable of falsifying
the mechanism (`#784`-class tear) the transport design still assumes is not the carrier. Cost of
keeping the design warm is zero cards and zero build slots; cost of NOT having it if morning's
trace-OFF census or the 4-card window shows AR-ring lag is that the decisive instrument has to be
designed *during* a hunt, which is exactly how tonight's worst rows were born. The doc's §1 question
is therefore narrowed (amended in place, annotate-never-delete) to the wire — GPU store buffers,
PCIe posted-write ordering, gfx900 L2 — the three channels `ARBCELL` explicitly refuses to cover.
**Trigger, pre-declared so it can't drift into a hunt-on-sight:** (i) any `kind=ar`/`ar-fallback`
retry count > 0 in a trace-OFF bin (F-A makes this visible now), or (ii) the first TP4 boot where the
one-shot ring is extended by my B-2 tier (mesh skeleton `results/amd/TP4_B2_mesh_skeleton_cell.cu`)
— at which point the arm is not insurance, it is the only witness the new geometry has.

## 5. What I shipped with this audit (all tools-only, zero src, zero card, own worktree)

1. `tools/census/read_census.sh` — totality fix (buckets + `raw == Σ` equality assert +
   refuse-not-guess on `UNRECOGNISED`, exit 1) **plus** the runbook's pre-boot reader rows the chair
   asked to ship before the census read: **G6(a)** lone-rank step join keyed by rb-derived request
   ordinal, **G6(b)** per-ordinal step-count/max asymmetry table, **G6 counts table**, **G7**
   argmax-ring wrap-window witness (`observed > epoch`), each with its own INSTRUMENT-ERROR branch so
   an unarmed leg cannot print a clean.
2. `tools/ops/check_census_retry_attribution.py` — the permanent cell (F-A's triple), both-direction
   falsifier measured.
3. `drafts/PG1_CHECK_S_census_attribution.patch` — ready block for PG-1 Check (s) (gemini's gate file;
   offered, not applied — WO-TP4-D is exclusive). **STATUS, two follow-ups measured later the same
   night:** the chair merged the lane (main tip then `ded3403b`, cell present as
   `tools/ops/check_census_retry_attribution.py` blob `48bdc174a51b`), the block still applies clean at
   that tip, main's gate has no Check (s) yet, and gemini acked the leg on the hub. **Routing defect,
   reproducible twice from this desk:** a hub `comm_send` addressed to the NAME `Gemini` reports
   "Sent … as pi-dual_5060_ti_ninfer-1608656", which is THIS agent's own row — a self-delivery, not a
   send; the explicit id `b4791a54` is rejected ("no agent named … is registered"). So the working
   path for gemini traffic from here is the chair's relay (ruled by him at 23:07Z: "route Gemini via
   me"; class: relay, not re-route), and the generalization for the board is sharper than my case:
   **a hub-direct can report SUCCESS while delivering to the sender**, so "I sent it to X" is not
   evidence X received it — the receipt has to name the recipient's row, or the sender checks their
   own inbox for the echo. Their ack also cited a lane (`wo-gfx906-perm`) that exists nowhere on this
   machine while its sha resolved on my real lane `amd/wo-gfx900-perm`: the which-ring/which-lane
   naming law, arriving one more time, from the direction that usually applies it to others.
4. This audit + the AR-parity design amendment (§4 narrowing + the trigger rows).
5. **Row F's own answer, shipped the same hour (zero cards, zero build slot):**
   `tests/test_ring_pairing_props.cpp` — the family law as 15 int-arithmetic checks: pairing
   injectivity over 400k calls × 128 slots × 3,125 request boundaries; reset-cannot-lower-a-read-domain
   with the two FORBIDDEN shapes as negative arms; and the int32 wire-width residual **reproduced**
   (both wrap outcomes named — the silent false-accept and the loud retry-to-bound) instead of argued
   away. Its detector **self-tests on a planted twin before it reports a clean**: my first draft also
   compared against `seen[0]`, which PASSES on a real duplicate; poison run P3 caught that at my own
   desk, and the fix is here rather than shipped blind. Falsifier ledger lives in the file header and
   names the two mutations this cell CANNOT see (P3 on clean data — nothing to find; P7 position
   collapse — belongs to the boot battery's MC31-H G6 rows, not to a model). Registration is not mine
   to land: `drafts/FB_plus_RING_PROPS_registration.patch` carries this cell **and** F-B's one-line fix
   as one reviewable hunk (`git apply --check` clean; `tests/CMakeLists.txt` restored byte-exact
   `8bcf08abed3a` at my seat after generating it). Receipt rule for whoever lands it: `ctest -N` at the
   same tip names both tests and the count delta is exactly **+2**.

**SELF-GUARD ROW (LAW 17 turned on my own leg, stated before anyone has to find it):** F-A's WIRED leg is
*the cell*, and until gemini lands Check (s) (or the chair routes it) that cell is itself a `check_*.py`
with no entry in any generated inventory — the F-B class, live at my own desk. What makes it not-a-lie:
the cell **refuses (exit 2)** rather than passing when its pre-fix artifact (`c3895bdb`) cannot be
retrieved, so a run without coverage cannot print a clean; the RED/GREEN measurements above were taken
at my seat and are reproducible with one command (`python3 tools/ops/check_census_retry_attribution.py
--falsify --require-real`, rc=0 on this tree; rc=1 with 4 named FAIL rows against a mutated live reader,
reader sha restored byte-exact `b9df19c9c4a1`). The row is therefore **(b)-shelf: fix landed, cell
authored, wiring pending someone else's file** — and I am saying that here rather than letting §2's
"WIRED" column read as promoted.

**Live re-runs at my seat (all rc=0, reader sha recorded in the commit rows):** G18d = 1,010/1,010
argmax + G6a 1,470/1,470 paired, G6b 0/26 ordinals asymmetric, G7 1,608 K-lines 0 fires (the 1,608
matches `db09c924`'s banked arming count exactly); G18c = 434/434, 1,092 paired, 1,228 K-lines 0 fires;
agent3's own RED-legs unchanged: G17x 3 expired/206 KAR + 3 REJECT-candidate, G17z 118 expired/130 —
and both now additionally print INSTRUMENT-ERROR on the new legs, which is *correct* (those bins predate
MC31-H/A1TRACE-K): a leg that never ran must not read as clean.

**FOLLOW-UP: `CLOSURE_BAR_AUDIT2_agent5_2026-09-13.md` (same desk, ~23:5xZ) carries row F-C — the pairing legs this audit shipped were themselves pair-blind at world=4 (found in my own code, triple complete, fixed tools-only) — plus the 421-vs-434 scope root-cause and the drift-controlled verdict on agent3's retry price (87-96 ms upper bound → 72-79 ms, inside the source's own 65-80 ms ceiling).**
