# Quality A/B plan — m3_boot4 proposal (CPU-authored; NO boot until stamped)

## Scope established by static arithmetic (this session, pre-plan)
Per-round vpos advance from the EXISTING boot traces ([D2SS] step=vpos deltas;
+1 == a=0 == zero drafts accepted that round):
- p1r1 (the '.007-at-depth / .067-at-round-6' era, tree 2771f066): 28/29 rounds
  advance +1, exactly ONE round advances +2 (a=1) at vpos 11644 ~= step 17 (verified
  request-side, not a warmup seam artifact). Total REAL accepts = 1 in ~29 steps
  (0.034/step-k-normalized: 1/145 = .007 — matches its own stat, ratio was honest).
  Note: the '.067-at-round-6' anecdote belongs to the 27f066 boot itself (p1b-era
  logs), NOT to p1r1 — p1r1 shows ZERO accepts through round 6. Both eras: near-zero.
- m3_boot3 (fall-through-fixed, 9a76439e): 31/31 rounds advance +1. Zero accepts.
CONCLUSION: the '.067 era -> 0.000 era regression' question DISSOLVES — acceptance has
been essentially ZERO since the twin loop first ran; p1r1's 0.007/.067 were the same
near-zero with 1-2 stray matches dressed by the x2 counter eras (ratio unaffected, but
perception was). The REAL quality surface is the ORIGINAL 09:4x framing: 'drafter
proposes plausible-but-rarely-matching at N=1' vs batched .42. No era re-build, no
2771f066 A/B — the delta CORD chased does not exist. The one +2 round in p1r1 is a
usable artifact (what made it different) but it is anecdote-level.

## What IS clean and new in boot3's datum
For the first time the zero-acceptance is measured with: no fall-through contaminating
state after stop; correct round arithmetic; and 32 [DBG-GDN] prints (step<=4 gate)
available. Chains at round 1-2 remain byte-identical to batched lane-0 (closed at
f89f5db1-era) => divergence is at the FIRST COMMITTED-CONTEXT round, i.e. exactly the
two named unread surfaces from the debrief:
(1) A1's queued read: pool-contents-at-depth (hook feed) — what the coverage hook
    actually republishes into the pool at N=1 vs what the next chain reads.
(2) A2's named read: verify .cu GDN write-target semantics (ring-vs-linear) at N=1 —
    starts as CPU static NOW (AB2 below), independent of any boot.
Cross-rank [DBG-GDN] hashes differ at identical slot/consumed (90ee.. vs 5431..):
RECORDED WITHOUT INTERPRETATION — pool is TP-SHARDED; the designed comparison is
cross-route (twin vs batched at same frontier). AB3 makes that comparison real.

## AB plan (cells in order; boot cost: ONE slot, two cells)
- AB1 [DONE, this doc]: scope correction via existing-trace arithmetic. Kills the
  era-regression branch before spending a cell on it.
- AB2 [CPU, A2, starts immediately]: verify .cu GDN write-target read (last unread
  named surface; debrief says MINE in sequence). Deliverable: named mechanism or an
  explicit 'shape-correct, datum-needed' verdict for the boot.
- AB3 [ONE BOOT, CURRENT TREE, dual-cell]: twin body vs batched-lane body on this BIN
  (387150c10eb0 lineage or the AB2-fix delta, whichever lands first), [DBG-GDN]
  cross-route comparison AT the same frontier: batched lane-0 committed-state hash at
  consumed=X vs twin committed-state hash at consumed=X. EQUAL + zero-acc => state
  layer sound, hunt moves to chain INPUT window/lattice contents at round 2+ (hook
  feed, surface 1, A1's); DIFFER => the write-target/rewind asymmetry is named on the
  spot and AB2's read gets its falsifier. Pre-registered third reading: the very first
  round-pair (step 2->3) diverging => seed-vs-carry boundary is wrong even at depth 1,
  escalating the zero_slot/+4251 lineage with clean data.
  Card env: NINFER_D2_GDNHASH=1 stays; ADD explicit batched-cell prints at matched
  frontier (the batched loop's gate must fire at the same consumed values — check
  before launch, adjust gate, re-run fire-check if code moves).
- AB4 [fallback ONLY if AB3 inconclusive]: era-rebuild of 27f066 for a third opinion —
  NOT proposed now (disk 23 G, era question already dead per AB1).
## Sequencing ask
A1: your queued pool-contents-at-depth read and AB3's comparison design are the same
cell's two eyes — review AB3's gate placement before CORD stamps anything. CORD: one
boot slot for AB3 (dual-cell, ~3 min run class), no era rebuild, AB2 free CPU meanwhile.
6th-surface rule stands: anything new-shaped -> report, no freeform boot 5.

## P7 CONVENING AGENDA (pre-derived per CORD seq-38; if P7 lands we start from paper)
P7 = both cells clean 200+stop, cross-route state hashes ALL EQUAL at matched frontiers, acc still 0.
Branches (each with cost):
- B1 accept-kernel inputs: vc/ext mirror re-test with clean arithmetic — CPU static of
  accept inputs vs batched (mirrors shipped 1653619e; re-derive at true 1.0 tok/rd).
  Cost: one bounded A1-or-A2 static, zero boots.
- B2 licensed-token semantics at N=1 (lic_h/a indexing): static twin-vs-batched index
  trace at round 2 (the accept OUTPUT side; complements B1's INPUT side). Cost: CPU
  static + one targeted print-delta cycle if needed (fire-check+word+stamp, ~1 slot later).
- B3 proposal CONTENT (lattice/cand_ids): compare per-round chain draft ids twin vs
  batched lane-0 at matched consumed — the join EXISTS in current dumps (dflash2
  drafts lines print per-rank per-lane); cost: zero code, CPU parse of AB3's own log
  (do this FIRST at convene — it may collapse B1/B2 instantly: equal drafts + equal
  state + acc 0 = accept-logic-only; divergent drafts + equal state = fuse-read-side).
Recommended convene order: B3 (free, from the run's own log) -> B1 -> B2.
