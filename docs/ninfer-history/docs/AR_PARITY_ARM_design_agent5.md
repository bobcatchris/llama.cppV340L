# AR-PARITY ARM — design (agent5, lane amd/wo-gfx906-diff, chair order 20:1xZ item 1)

**Scope**: design-only. No bin, no patch, no boot — implementation rides a dedicated grant.
(HISTORY: this sentence read 'rides agent4's next queued fix-commit window' when written; A-4 merged at
6b98623f and carried the `kind=ar` field, which was the dependency for the STATUS raise path but is NOT
this arm — the parity word itself is still unbuilt, and its §3 condition precedent (agent4's volatile-RMW
status raise) has been satisfied since `status_transport.h` + the 30b green leg. Annotating the stale
phrase rather than deleting it, per the board's own law.) Companion measurement already banked: ARBCELL host model
@ `89267502` (`tools/smoke/arb_publish_tear_cell.c`) — exonerates the *construct* (payload->tag->
flag chain under in-order gated consumption: 0 mixes at correct epoch / 3.03M samples) and scopes
what THIS device arm must answer: the **wire** (GPU store buffers, PCIe posted-write ordering,
gfx900 L2) — the channel the host model cannot see.


**STATUS AMENDED 2026-09-13 ~22:5xZ (agent5, closure-bar audit `CLOSURE_BAR_AUDIT_agent5_2026-09-13.md`
§4; chair ruling 22:41Z: DRAFT-FORWARD as insurance, not parked).** Census bins G18c/G18d closed the
AR ring's **timeout** question armed (zero kind-less/AR-kind retries, verified by the now-total reader
in `tools/census/read_census.sh`) — and left its **word-integrity** question with ZERO witnesses on the
AR ring: the shipped parity arm rides the **argmax** ring's dead pad word only
(`one_shot_argmax.cu:21`; `one_shot_allreduce.cu:411-414` records that the AR kernel never sets bit 2),
and `tools/smoke/arb_publish_tear_cell.c` exonerates the **construct** in host-coherent memory while
naming the wire (GPU store buffers / PCIe posted writes / gfx900 L2) as out of scope. So §1's question
below is to be read **narrowed to the wire** — the whole channel ARBCELL cannot see — and the arm is
insurance against exactly one thing: a TP4/trace-OFF bin where AR-ring lag or a torn AR tuple is the
carrier. **Pre-declared firing triggers (so it cannot drift into a hunt-on-sight):** (i) any
`kind=ar`/`ar-fallback` retry count > 0 in a trace-OFF census bin — F-A's reader fix is what makes
this visible at all; (ii) the first boot on the B-2 one-shot tier generalized to world>2
(`results/amd/TP4_B2_mesh_skeleton_cell.cu`), where this arm would be the only word-integrity witness
the new geometry has. **TRIGGER-(i) MECHANISM SHARPENED by A-4 landing (6b98623f, ~19:01Z 09-13 — annotated the same
hour it became stale, agent2's tense catch): the AR ring now emits `kind=ar` at
`one_shot_allreduce.cu:523`, so 'AR-ring lag observed' is directly COUNTABLE rather than inferred
from a kind-less shape, and my reader refuses an unattributable line outright under
`CENSUS_STRICT_KINDS=1` (3374f5c4). Consequence: trigger-(i) is now a positive count, not a residue —
the first post-A-4 bin (the G-AMD-18 window) either prints `kind=ar > 0`, which fires the trigger by
name, or prints `kind=ar 0`, which is the strongest AR-ring absence statement this box has ever made
and closes the trigger with a measurement instead of an empty bucket. Across every banked bin to date
the AR ring's own retry line has never appeared in either spelling (kind-less 0 in G18c/G18d/G18e, and
those bins predate the field), so nothing has fired yet.**
**TRIGGER-(i) CHECKED AND NOT FIRED at the first eligible bin** — G-AMD-34/G18e
(trace-OFF census, `results/amd/p3/G18e_traceoff_second_witness_row_agent5.txt` @ 9a11305c): AR-ring retries 0
with the reader's totality assert TOTAL at raw 831 == split 831, so no kind=ar/ar-fallback line appeared in a
trace-OFF bin; the pre-declared trigger did its job — it was evaluated against a count, came up empty, and the
arm stays on the shelf as insurance rather than being promoted by narrative or killed by inconvenience. Implementation still rides a build, still conditioned on agent4's volatile-RMW
raise path (§3 dependency row), still zero cards to author.

## 0. The question the arm answers (one line)

Does the 128-slot AR ring ever deliver a peer payload whose WORDS come from different publish
generations (the #784 tear class), at the ~128-calls/token rate of the plain-decode served path —
and does the answer arrive with an arming proof, not an absence claim?

## 1. Carrier: the per-slot flag word's neighbours — NO payload-layout change

Each `Slot` (ours `one_shot_allreduce.cu:247-250`) owns `host_buf[2]` + `flag[2]`, both pinned.
The argmax family proved the pattern: parity rode a payload word that was ALREADY dead for float
math (`pad`, `one_shot_argmax.cu:15-22`) so the launch was byte-identical when unarmed. The AR
payload has no dead word (every bf16 is data), so the carrier is a **new per-slot parity word
alongside the flag**: `uint32_t *parity[2]` per slot (NOT inside `host_buf` — that region is
peer-read by the combine loop; touching its layout is banned by the "launch byte-identical when
unarmed" precedent law). Storage cost: 128 slots × 2 ranks × 4 B = **1 KiB pinned**, allocated in
`Impl()` beside `cudaHostAlloc(slots[s].flag[r]…)` (:287-290), zeroed same as flags, freed same
dtor path. No VRAM term — host-pinned only; no budget constant touches it (VRAM law unaffected).

## 2. Publish side (kernel `one_shot_ar_pinned_vec_kernel`, writer = tid lane-reduce + leader)

Per call, this rank computes one word over the payload it is publishing:

```
w32[i] = *(uint32_t*)&local_buf[2*i]                       // payload as u32 view (no rounding)
x      = fold-xor of w32[0 .. n_elems/2)                   // deterministic, order-fixed strided
tag    = xor32(x, (uint32_t)step_gen, mix(expected_epoch)) // mix = *(PRIME) like ARBCELL mixw
```

- fold is the SAME strided loop shape as the publish copy (:96-100): thread t folds i ≡ t
  (mod stride), block XOR-reduce through `__shared__` — 128-bit words in flight anyway, ~2 extra
  ALU ops per uint4 on the publish pass; **cost row: single-block 1024-thread kernel at n=5120 is
  5 vec-iters/thread — the fold adds ≤ 5 shfl-steps + one shared round ≈ sub-µs at the measured
  14 µs/call class (A7 anchor, `TP4_AR_TRANSPORT_DECISION_agent5.md` §1)**. It is still a timing
  dial — see §5 arming discipline.
- Store ordering: `payload stores -> __threadfence_system() (existing :103) -> *parity[slot][rank]
  = tag -> existing gen/flag chain (:117-120)`. The parity word rides BEFORE the flag on the same
  fence it already covers — no new fence. Unarmed (env off): the fold loop and store are compiled
  out behind the runtime gate, launch byte-identical (the argmax `parity` arg precedent,
  `one_shot_argmax.cu:433,:86`).
- **Inputs to tag**: the LOCAL payload only (what this rank publishes), plus THIS call's
  `step_gen` and `expected_epoch` — so a tag can only be produced by a kernel that completed its
  own publish pass: same "epoch is inside the tag" property that makes the argmax tag immune to
  coherent-stale slots (`one_shot_argmax.cu:183-188`).

## 3. Consume side (same kernel, after BOTH gates pass)

The combine loop (:189-233) already reads every peer word: fold the SAME xor over `peer_v[i]`
inside the existing loads (zero added memory traffic — the words are being read anyway), reduce,
then **after the combine loop, the leader (tid==0) recomputes `tag' = xor32(x_peer, step_gen,
mix(epoch))` and compares to `*peer_parity`**. Mismatch ⇒ a peer tuple whose words are not from
the generation the flag attested = door-(iv) firing on the wire.

- Comparison sits POST-combine on purpose: the combine's output is already poisoned in a mismatch
  world; the value of this arm is DETECTION (naming the mechanism), not prevention. Prevention
  stays the gen-gate's job.
- Mismatch publication: `atomicOr(status, 4u)` — the AR family's status today carries only the
  timeout bit (`one_shot_allreduce.cu:135,:159` — value 1); argmax uses bits 1|2 with the
  throw-side decode at `one_shot_argmax.cu:371-382`. AR parity takes **bit value 0x4** so the AR
  deferred-throw (:365-373) can name the cause like argmax's PARITY throw does. **CONDITION
  PRECEDENT (board law)**: the raise path is the DEAD mapped-atomicOr channel (eb4846df/b28d25fd)
  — the AR-parity arm MUST NOT ship before agent4's volatile-RMW fix lands for `[ARTAG]`; the same
  one-liner is what makes 0x4 visible host-side. Reviewers reject this arm's PR without that
  dependency row.
- Event-gated print (rate law): first 16 calls per rank (arming witness) + every mismatch:
  `printf("[A1TRACE-ARP] rank=%d step=%d epoch=%d tag_seen=%08x tag_recalc=%08x\n", …)` — the
  ~128/token full-print version is the dial warned in the chair order; it rides ONLY with an
  explicit separate capture decision, event-gating keeps the armed steady-state at zero prints.

## 4. Predicates and the arming law (absence claims name their proof)

**Predicate format (chair 20:5xZ item 1): aligned to agent3's poison-predicate shape from A2Q3
(`0/0/0-with-witness-gap` @ d0c5e381) — every count is FOUR-STATE, never binary:**

| state | meaning | admissible verdict |
|---|---|---|
| CHECKED-AND-CLEAN | `ARP_CHECKED == AR_CALLS_SERVED` both ranks, no mismatch | "AR word integrity held at N=checked-count" — the ONLY row that says clean |
| TRACKED-RED | mismatch observed, tags printed | conviction row (carrier named via §3 classification) |
| NEVER-CHECKED | `ARP_CHECKED < AR_CALLS_SERVED` (witness gap) or arm unarmed | "0 mismatches" MUST be reported as `0/0/0-with-witness-gap(N)` — a hope, per eb4846df's durable rule, never a result |
| INSTRUMENT-ERROR | checked-count exceeds ring steps, throw channel dead (pre-volatile-RMW), print silent where control fires | reject the boot's AR rows entirely; the control predicate catches this class |

The per-rank arithmetic:
- **Host-side count predicate (the verdict gate): `AR_MISMATCH == 0` is admissible ONLY IF
  `ARP_CHECKED == AR_CALLS_SERVED` for each rank, and `AR_CALLS_SERVED == 128 × tokens + warmup`
  per rank within a slot-wrap count of 2.** Implementation: the leader increments a second pinned
  word `*arp_checked[rank]` at each post-compare (cheap, one RMW per call — same channel as gen);
  host reads it at request end via the existing `last_call_timed_out`-style surface (new sibling
  `arp_last_checked(rank)`). The commit-count anchor `impl_->rank_step[rank]` (:376) is the
  independent term — equality `arp_checked == rank_step` at request end is the "every call was
  witnessed" proof; `rank_step` per token-vs-request comes from the existing `reset_step` sites
  (request boundary) and the decode `[tp2] decode=` lines (tokens).
- **Positive control (RED, per closure law)**: a deliberate one-word flip in the peer tag on call
  #3 of a control boot (env `NINFER_ARP_FAULT=1`, writer XORs 1<<0 into the parity store) MUST
  produce exactly one `[A1TRACE-ARP]` + next-entry throw naming the 0x4 cause. RED row pre-fix: fault
  fires, detection silent (proves the old blindness); GREEN row post-arm: fault fires, detection
  loud. Both shas named in the closing row. This is the cell that makes "0 mismatches tonight"
  mean something.
- **Wrap-window row**: mismatches classified by `tag_seen == tag_of(gen_obs-1)` (previous
  generation coherent-stale = ring-reuse window, §3 of the differential) vs arbitrary (true word
  tear) — the print carries both raw tags so the classification is post-hoc mechanical, no
  interpretation needed at the boot seat.
- **BIN (i) COMPATIBILITY ROW (epoch-wrap alignment, agent3 design @ plan :36, retires the
  `dev_epoch` indirection)**: the tag's epoch term is written to survive both flag shapes —
  pre-(i) it is `*dev_epoch` as read today (:83); post-(i) it is `(step+1)/kNumSlots + 1` derived
  host-side from the same counter that feeds the flag (the flag carries step; epoch becomes
  pure-quotient, no device read). The consumer-side recalc derives the SAME value from its own
  `step_gen` argument in both worlds — that is why `step_gen` AND epoch are both in the tag
  (§5 row 2): (i) aligns the domains, this arm verifies the payload across them, and neither
  depends on the other's landing order. If (i) lands first, the design's only edit is deleting
  the :83 read from the tag's epoch term — predicate, control, and states unchanged.

## 5. What the arm does NOT do (boundary rows)

- Does not change the gen-gate, epoch, slot arithmetic, or combine — observer on the existing
  chain (argmax-parity precedent, same additive shape: one new arg, one gate, zero default-path
  change).
- Is not exoneration by itself: the clean rows mean "no cross-gen word mix witnessed at
  (step,epoch) granularity by recomputing witnesses"; a mix that is SELF-CONSISTENT at the wrong
  generation (whole-slot coherent-stale with its own tag) passes — caught only by the
  epoch-inside-tag property, which is why the tag MUST include epoch and gen, both.
- The tag is a 32-bit XOR fingerprint: a mix that happens to preserve it costs 2^-23/publish
  (masking law inherited from `one_shot_argmax.cu:189-193` — exponent-forced positive-float form
  so NaN quieting can fabricate nothing; store as u32 directly here, the parity word is never
  float-read, so the mask law reduces to "same mix function as ARBCELL" for review symmetry).
- Timing: fold + extra pinned word RMW is a dial — sub-µs expected, but the arm's OWN
  determinism-neutrality must be measured, not assumed: pair-run SAME-PROMPT ×5 with
  `NINFER_ARP=0` vs `=1`, G-CELL-compare outputs (if the arm's presence changes fork RATE, that
  is itself data about the suspect — report it; do not drop the arm to keep the bar green).

## 6. Env + gates summary (implementation-ready checklist)

| item | value | anchor |
|---|---|---|
| gate | `NINFER_ARP` (read-once static at launch site, same pattern as the :407 trace gate) | new, argmax-parity precedent `one_shot_argmax.cu:433` |
| storage | +`uint32_t parity[2]`, +`uint32_t arp_checked[2]` per slot/rank, pinned, ~1 KiB + 8 B | `Impl()` :287-303 |
| publish | fold-xor(payload)→tag(gen,epoch)→store BEFORE flag, after existing fence | :96-120 |
| consume | fold-xor(peer reads, already happening)→recalc→compare, leader, POST-combine | :189-233 |
| status bit | 0x4 = ARP mismatch (argmax family keeps its 0x2) | :135,:159,:365-373 |
| print | `[A1TRACE-ARP]`, first 16 + any mismatch only | argmax `:232,:252` prints + KAR rate law `one_shot_allreduce.cu:172-178` |
| predicate | ARP_CHECKED == rank_step per rank per request; verdict admissible only with it | §4 |
| positive control | `NINFER_ARP_FAULT=1` single-bit flip at call 3, must be caught | §4 |
| determinism-neutrality | ×5 same-prompt paired runs armed vs unarmed, G-CELL compare | §5 |
| dependency | agent4's volatile-RMW status-raise fix MUST land first (dead-channel law) | §3 |
| verdict format | four-state poison-predicate (§4 table), never binary | §4 |

## 7. THE CLOSURE BAR for item 7 — written once, in the closeout plan's own terms

(chair 20:5xZ item 2; the closeout doc and bin-(iii)'s release row should quote this, not re-derive it)

**A row CLOSES item 7 iff it carries the closure law's full triple: (RED) a deterministic repro on
the pre-fix artifact naming both shas — tonight's banked REDs (`1d0ff3c6` 5/5-distinct, `82b9d60f`
3+2) qualify as the red-row inventory the fix must name; (GREEN) the SAME geometry ×N (G-CELL's 5/5,
census bar N/N) on the post-fix artifact, CHECKED-AND-CLEAN under §4's four-state predicate — not
`0/0/0-with-witness-gap`; (WIRED) the cell joined to the permanent suite guarding the CLASS (a
wrap-boundary fix owes a boundary cell, a tear fix owes a pack-integrity cell), not the instance.**
Everything else is name-but-not-close, and the board should sort (ii)'s outputs into exactly four
shelves: (a) **Closes**: triple complete, all three shas in the row. (b) **Names the carrier,
closes nothing**: e.g. margins≈0 everywhere with fork-rate >0 — the tie-site is located and the
mechanism class is visible, but per the law's own words a prose closure closes a CLAIM, never a
BUG: this row owes the fix, then (a)'s triple. (c) **Exonerates a suspect, closes nothing**: e.g.
"all KAR REJECTs live in warmup, served region CHECKED-AND-CLEAN" — a legitimate, armed, permanent
row (it retires the ring for the served window — the §1.3 scoping note applies: only if the
witness ran at the ~128/token rate with its checked-count printed), but exoneration narrows the
candidate set; item 7's bug is the FORK, so the fork bar (G-CELL 5/5) is untouched by it. (d)
**Instrument-error**: counts without arming, silent controls, raise-channel dead — the row is void
by the durable rule and re-runs; no shelf claim either direction. The discriminator between (b)
and (a) when (ii) banks: ask for the GREEN sha of a FIX artifact — a mechanism named AND fixed
(plan's own GATE words) is (a); named-and-pending is (b) and keeps item 7 OPEN with the carrier
named. My AR-parity arm's own GREEN leg (and the AR-wrap fix doc's) inherits this bar verbatim:
its (a)-state is `ARP_CHECKED==rank_step` both ranks × mismatch=0 × control-fired × cell in the
boot battery — anything less reports as (b)/(c)/(d) by §4's table, by construction.
