# WO-TP4-B / R1 argmax transport — DESIGN HANDOFF (agent4, 2026-09-14 ~10:3xZ)

**Status: NOT IMPLEMENTED.** My session context ran out at the point where the next keystroke would
have been writing a TP transport into engine custody. A partial R1 is precisely the failure class
this hunk exists to kill ("a half-built R1 is exactly the silent class this hunk exists to kill" —
tp_group.cpp:343, my own comment from the A-4 commit), so the honest deliverable is a design pinned
to bytes plus the verified preconditions, not a half-file. Everything below is MEASURED, not
recalled; every line number was read this session.

## 0. Preconditions — both verified, and one chair claim corrected in the chair's favor
- **The relax is live on main**: `b412303e`, `engine.cpp` carries `runtime::tp2::wants_tensor_parallel`,
  `engine_route.h` present.
- **PG-1 rc=0 at main: TRUE, and my "it stays RED" warning was the one that aged.** I ran the gate in
  the shared checkout at `b412303e` → `PASS: Zero unauthorized pre-existing CUDA src/ files modified.`
  Mechanism, since it matters for anyone re-deriving this: **check (a) compares against
  `origin/amd/main`, so merging the file is what cleared it** — once engine.cpp is ON main the diff is
  empty. My lane-tip FAIL and the chair's main-tip PASS were both correct in their own frames.
  Consequence: `drafts/WO_TP4_E_ROSTER_ADDITION_for_gemini.patch` is **DEAD, unapplied, do not
  review it** (second roster patch of the night superseded by the merge itself — worth a rule: before
  routing a roster patch, ask whether the merge already answers it).
- **Disk: the 2 G is NOT mine.** Zero `*.o` written anywhere under `/home/chris/worktrees` or the
  shared checkout since 02:00Z; my last link is **2026-09-13 20:02Z** (the GATE-3 bin, hours before
  the dark window); `artifacts_bin` = 1.8 G with no new bin after 2f831cc8; my build tree is 394 M,
  statistically tied with `wo-shim-funcattr`'s 393 M. **No lane built in the window the chair is
  asking about**, so the delta is not a build at all — candidates the next desk should price:
  `Desktop` at 6.2 G, `dual_5060_ti_ninfer` at 20 G, logs/census captures banked by the boot legs
  (every leg this night wrote serve logs + 25-response sets + witness files). 8.8 G free is still
  under the 11 G build floor, so **R1's build needs the chair's clean-up ruling before it can be
  sequenced at all** — flagging that as a blocker on leg (1), not a footnote.

## 1. The port, in the shape of the pattern that already proves it
Template is `TpGroup::allreduce_local_bf16`, tp_group.cpp:279-286 — **transport-else-RCCL, in that
order, no estimate**:
```cpp
if (impl_->one_shot && n_elems <= OneShotAllReduce::kMaxElements) {   // fast path, world==2 ring
    impl_->one_shot->allreduce_bf16(...); return;
}
NCCL_CHECK(ncclAllReduce(local_ptr, local_ptr, n_elems, ncclBfloat16, ncclSum, k.comm, ...));
```
Target site is `allreduce_argmax`, tp_group.cpp:349, currently:
```cpp
require_argmax_transport(size(), impl_->one_shot_argmax != nullptr);   // throws at world>2
impl_->one_shot_argmax->allreduce_argmax(rank, logits, out_token, draft_vocab_ids, stream, emit_conf);
```
**THE CHANGE — CORRECTED AT BYTES BEFORE ANYONE BUILDS ON IT.** My draft above, and the chair's
phrasing "allgather+local-argmax", both say to gather the ROW SET. **That is wrong, and the ring
proves it:** `struct ArgmaxPayload` (one_shot_argmax.cu:17-25) is `{float val; int tok; float sumexp;
float pad;}` = **16 bytes per token per rank**, slots sized `kMaxTokens * sizeof(ArgmaxPayload)` (:371).
**The ring never moves logits rows** — each rank argmaxes its own shard and exchanges only the
per-shard CHAMPION plus `sumexp`, which is precisely the term that makes a two-rank
winner-vs-runner-up comparison reproduce a full-vocab softmax. Gathering rows would cost ~242 KiB per
rank and, at world=4, land straight on the 485-vs-244 KiB destination overflow GATE-3 refuses — i.e.
my draft prescribed the one route GATE-3 exists to block. Corrected R1 inherits the ring's algorithm
and widens only its transport:
```
per rank : local shard argmax -> (val_r, tok_r, sumexp_r)    // the SAME computation the ring does
cross-rank: exchange the 16-byte champions (RCCL allgather of kMaxTokens*16B, NOT of vocab rows)
per rank : reduce the W champions by (val, tie-break on GLOBAL token id) -> one winner
conf     : softmax over the full vocab = sum of sumexp_r over ranks, normalised by the global max
```
Reducing W champions is deterministic-equal iff every rank sees all W of them — the requirement below
is unchanged in KIND, but its size is 16 B x world instead of a row set, **which is why this route
needs no GATE-3 exemption at all**.
```
if (impl_->one_shot_argmax) { ->allreduce_argmax(...); return; }        // world==2 ring, unchanged
require_r1_transport(size());                                           // still a loud refusal at world<2
allgather rows (vocab-shard bf16) into a per-rank full-vocab buffer;
argmax locally over the FULL row set; write out_token (+ conf, + draft_vocab_ids mapping).
```
**Two hard requirements, both load-bearing:**
1. **Every rank must reduce the FULL row set.** The equality that makes allgather+local-argmax
   deterministic-equal to the ring is total-row reduction — `argmax` over a *partial* gather is a
   different token on each rank, which is a wrong-number bug with no symptom. The row-width math is
   the same class as GATE-3: **do NOT hardcode `n_vocab/2` or any world-2 stride here.** Sizes come
   from `world` and the real vocab; `gather_capacity.h` is the existing home for "how big is this
   destination" and `require_gather_capacity` must stay on the path so an undersized destination
   refuses instead of overflowing (world=4 measured 485 KiB vs a 244 KiB destination).
2. **The refusal must survive as a capability check, not vanish.** `require_argmax_transport`'s
   `has_transport` early-return keeps working; add the R1 arm as a *replacement* of the throw, so
   the predicate becomes "ring OR r1 OR throw" rather than "ring OR throw". **ARM 6 of
   `tests/multi_gpu/tp_argmax_routing_host.cpp` already pins this future** ("the refusal cannot
   outlive its reason") — read it before writing; it is the spec for what the cell must look like
   after R1, and it will go RED if the throw is deleted rather than widened.

## 2. The triple (closure law — three shas, no exceptions)
- **RED (already banked, zero work needed)**: today's loud throw at world=4, captured at
  `/tmp/G18_*.log` (chair's reference), text = `TpGroup::allreduce_argmax: no argmax transport at
  world=4`. Also pinned by an existing PASS arm in `tp_argmax_routing_host.cpp` (the R1-future arm),
  so the RED is a permanent cell row, not a one-off log.
- **GREEN (host cell, zero card — the determinism-of-the-RULE claim)**: new cell, sibling of
  `tp_argmax_routing_host.cpp`, compiled `/usr/bin/c++ -std=c++20 -I src`. Synthetic 4-rank logits,
  shard-per-rank, and **assert every rank's local argmax over the full gathered rows yields the SAME
  winner token id** — plus the falsifying arm that makes it non-vacuous: a deliberately PARTIAL row
  set (rank r sees only rows [r·W,(r+1)·W)) must yield DIFFERENT winners, proving the cell can tell
  total-reduction from partial. Without that second arm the test passes on a bug. Name both outcomes
  in the output; a cell that can only print "same" has no teeth (tonight's recurring lesson).
  Extract the rule as a pure function (row-set in → token id out) into a bare header, same reason
  `rank_index.h` / `gather_capacity.h` / `argmax_routing.h` exist: **an untestable-at-zero-card
  decision gets tested wrong or not at all.**
- **DEVICE leg (the served boot)**: world=4 first light on the R1 bin — my pre-staged runner is
  `results/amd/p3/G18_world4_firstlight_agent4.sh`, env-parameterised (BIN/BIN_SHA/ARTIFACT/DEVICES/
  W/PORT/LABEL/GEN/REPS), grant-as-code refusing a world=2 ack at rc=77, measuring per-die capacity
  live with **no budget constant**, capturing the refusal text rather than retrying. Its expected
  outcome CHANGES when R1 lands: it currently pre-declares a GATE-2 throw; after R1 it should be
  edited to expect a SERVE, with GATE-3 as the only remaining admissible refusal.
  Note world=4 at this bin needs `--devices 0,1,2,3` and **dev0 is the display card** (148 MB
  furniture, measured) — the manifest must re-measure it, never assume the baseline.

## 3. LEG-ZERO-once-more (chair leg 2) — one command, once the R1 bin is banked
`LEGZERO_BIN=<path> LEGZERO_SHA=<sha256> LEGZERO_LABEL=G22leg0 LEGZERO_ACK=<world2 grant> bash
results/amd/p3/A4_leg0_inertness_replay.sh` — env-parameterised exactly so this leg costs a relink
plus ~6 min, not a rewrite. Comparator grid to quote against (all three measured this session):
onset req #11 at stamp **326 / 326 / 327**, max try 8 ×3, FAILOUT/ARTAG/KAR-REJECT 0, MC31 1,608
identical. Read with `CENSUS_STRICT_KINDS=1` (post-A-4 era; the R1 bin names kind on both emitters)
— the lenient default silently buckets, which is the trap agent5 fell in and I nearly fell in twice.
**Agreeing with the chair's class statement**: every bin owes its own inertness receipt; proximity
of a parent's receipt is not evidence, which is the same law that produced G-AMD-38.
Caveat to carry, not to re-derive: the leg's **cycle class is NOT-COLLECTED** (my thermal predicate
was broken twice — `edge:` sits at column 0, and the unit is a UTF-8 degree sign, not ASCII 'C').
Fixed in the script now; earlier rows must not be read as "box was cold".

## 4b. A SECOND pair-shaped gate the board-clearance did not cover
`OneShotArgmax::allreduce_argmax` opens with (one_shot_argmax.cu:434-436):
```cpp
if (rank != 0 && rank != 1) throw std::invalid_argument("OneShotArgmax: rank must be 0 or 1");
```
So even AFTER a transport decision exists, **the ring's own entry refuses rank 2 and 3**. A world=4
boot needs BOTH (a) the R1 arm at tp_group.cpp:349 AND (b) this gate widened or bypassed — either one
alone still throws. `require_argmax_transport` was never the only pair-shaped thing on the path: this
is the same class as `tp_group.cpp:164`'s hardcoded {0,1} and `checked_rank_index`'s former
`(r==0?rank0_:rank1_)`, one layer further down. Expect a third: `Impl` is internally 2-array-shaped —
`conf[2]`, `host_payload[2]`, pair-sized slots (:349-351, :371-377) — so lifting the rank check still
leaves an object built for a pair. **This is not a one-line widening, and the "last code gap" framing
understates it.**

## 4. Where the token id must agree — the risk a host cell CANNOT see
`out_token` is consumed by the decode loop and `draft_vocab_ids` maps a winner back to the draft
vocab for MTP. **If the R1 path writes a token in gathered-row coordinates while the ring path writes
one in shard-local coordinates, every rank agrees and both are wrong** — a coordinate bug is invisible
to the host cell as I scoped it, because the cell's synthetic input defines the coordinate system.
The next desk must check the ring implementation's output convention in
the ring implementation: `OneShotArgmax::allreduce_argmax` at **src/core/multi_gpu/one_shot_argmax.cu:430** (not one_shot_allreduce.cu — my first grep guessed the file and returned nothing, so the definition was located by re-searching, and its own :257 comment already says the entry "makes that LOUD, never silent") against what the allgather rows
provide, and the device leg is what grades it: greedy text must byte-match the world=2 attractor
`348e77a1222dea7f` at the same arms, or the token coordinates differ. That comparison is the real
first-light test, and it is NOT replaceable by the host cell.

— agent4 (lane `wo-p3-serve`, branch `amd/wo-p3-serve`, tip c25d3299 + this file; pair untouched,
KFD 0, zero own processes, no build this session, disk 8.8 G)


## 5. PAYLOAD SIZING vs THE MEASURED RCCL GRID (chair 11:43Z item 3) — quoted, not adjudicated
Measured on this box by agent5's B-1 (RCCL 2.20.5, cold, eager, shape #4 per-rank thread + blocking=1;
pushed 422d329a, rows `G18B1_b1_shape4_{legA,legB}.log` sha 50e90e1b/404aacfe):

| size | world=2 median | world=4 median |
|---|---|---|
| 10 K | 70.67 us | 129.15 us |
| 128 K | 102.28 us | 196.47 us |
| 1.31 M | 407.98 us | 939.04 us |

**R1's exchange is 16 tokens x 16 B = 256 B per rank** (`kMaxTokens`=16 x `sizeof(ArgmaxPayload)`=16B)
— i.e. **~39x BELOW the smallest measured bin.** The honest consequence, stated as a bound rather than
a number: the grid gives no interpolation at 256 B, so world=4 allgather latency is expected **<= the
10 K bin's 129.15 us**, and the small-message regime is dominated by launch/sync overhead rather than
bytes. First-light row should quote `w=4 @ <=10K bin: 129.15 us` as the expectation ceiling and report
the MEASURED per-step figure beside it; if the measurement comes back far above it, that is a finding
about overhead at tiny sizes, not about bandwidth, and must be named that way. This is exactly why the
champion exchange (16 B/token/rank) beats a row-gather (~242 KiB/rank): a row-gather would land at the
1.31 M bin's **939 us** and additionally hit GATE-3's capacity wall.

**Also from that ruling, recorded so it is not re-litigated at first light:** the chair upheld agent5's
HOLD reading of 129.15 us against the pre-declared 120/240 us lines — REPORT AND HOLD stands, the tree
option stays retired by the row's own math, and the adjudication that matters (real `[AR-RETRY]`
world=4 serve data) comes from the first-light logs. agent5's readers are world=4-certified and waiting.

## 6. STATUS AS OF 11:5xZ (what is and is not done, for whoever picks this up)
DONE, banked and pushed:
- `src/core/multi_gpu/argmax_reduce.h` — the champion-reduction rule, single home, 17-arm cell green.
  §4's coordinate risk CLOSED at bytes there: `.tok` is GLOBAL (`one_shot_argmax.cu:218`), remap LAST
  (`:319`), comparator is a TOTAL ORDER (`:305-308`), conf symmetric so 2->W generalises (`:323-330`).
- Threading assertion (chair item 2) verified: TpGroup = one worker thread per rank
  (`tp_group.cpp:60`, spawned `:129`), every rank op via `dispatch_all` (`:72`, Init `:188`), so the
  R1 collective inherits the per-rank-thread property. Constraint: never hoist it to a shared thread.
NOT DONE — the transport itself, and it is new device code, not a port:
- `one_shot_tp_argmax_kernel` fuses argmax+IPC+reduce in one lock-step handshake; there is NO
  separable shard-argmax kernel to call. R1 needs (a) a shard-argmax kernel writing
  `ArgmaxChampion` per token per rank, (b) an `ncclAllGather` of `kMaxTokens*16B` from the per-rank
  worker context, (c) a reduce kernel over W champions calling `reduce_champions`/
  `champion_softmax_conf` so the rule stays single-home, (d) the draft-vocab remap last, (e) the ring's
  `rank in {0,1}` throw at `one_shot_argmax.cu:434` lifted and `Impl`'s 2-array shape
  (`conf[2]`, `host_payload[2]`, pair-sized slots) widened to world.
- Build slot unused as of this writing; disk 12 G; no `lsof +L1` sample taken, so no disk delta is
  quoted anywhere above.
