# R1 transport lane — agent5 transfer STATUS (entry commit, 2026-09-14 ~11:5xZ)

**Transfer:** chair ask 5b286d62 (11:48Z), ACCEPTED. Lane `amd/wo-r1-transport` @
`/home/chris/worktrees/amd-wo-r1-transport`, branched from `origin/amd/wo-p3-serve @ 8beb1b9c`
(agent4's banked tip; `b412303e` ancestry verified via `git merge-base --is-ancestor` at the
shared repo). agent4's seat is context-dead; their §6 STATUS was the entry point, read in full.

## Entry verification at MY seat (numbers-paired-to-tree law — agent4's greens are re-derived, not inherited)
- `tp_argmax_reduce_host` (17-arm champion-reduction cell): compiled `g++ -std=c++20` rc=0,
  ran rc=0, PASS line + tie/n=1 arms echoed. Their §6 self-catch (header missing <cmath>)
  noted — the cell's standalone-include guard is real.
- `tp_argmax_routing_host` (routing refusal + index bound + gather capacity): rc=0, "all three
  arm sets passed" — including the world>2 REFUSE arm this transport will legitimately change.
- Throw site verified at bytes: `tp_group.cpp:349` = `require_argmax_transport(size(), ...)`
  immediately before `one_shot_argmax->allreduce_argmax(...)`; predicate lives in
  `argmax_routing.h` (host-testable, zero-device).
- RCCL state carried in from my B-1 (this session): shape #4 (per-rank-thread blocking init)
  = the ONLY shape that ever completed init on this box (4/4 ranks, rows G18B1_b1_shape4_*,
  era acbff97a). agent4's chair-item-2 assertion independently agrees: TpGroup gives each
  rank its own worker thread (`tp_group.cpp:60/:129`, every op via `dispatch_all :72`) —
  the R1 allgather MUST be issued from that per-rank worker context; never hoist to a shared
  thread, never loop ranks serially from one. My deadlock datum is the falsifier behind that rule.

## The contract I build to (chair ruling 5b286d62, verbatim priorities)
1. R1 lives at `tp_group.cpp:349` — NOT inside `OneShotArgmax` (the ring stays, world=2 path
   byte-untouched; the ring's `rank in {0,1}` throw at `one_shot_argmax.cu:434` is NOT mine to lift).
2. `§4b Impl widening DROPPED` — consequence I verified against §1: after the allgather EVERY
   rank holds all W champions, so reduce is LOCAL and deterministic-equal by the total order
   (val desc, global tok asc) — no per-rank pushed-payload arrays needed on the R1 path.
3. Exchange = byte-typed allgather of `kMaxTokens(16) × 16 B = 256 B/rank` champions into a
   world-derived `W×256B` destination; the bf16 collective at `:289` is NOT overloaded
   (dtype + semantics both wrong for it).
4. Reduce via `argmax_reduce.h` single home (`reduce_champions` / `champion_softmax_conf`);
   remap LAST (§4 coordinate law: `.tok` already global, `one_shot_argmax.cu:218/:319`).
5. First-light grader: world=2 attractor `348e77a1222dea7f` byte-match; payload 256B/rank sits
   ~39× below my 10 KiB bin — CEILING quote ≤129.15 µs w=4 with the measured figure beside it;
   overshoot = tiny-message overhead finding, named as such.

## Order (chair-committed, single-serialized-chain): cell -> code -> build -> bank -> grant-ask -> first-light
- CELL (in progress at writing): the NEW contract's falsifiable arms, host-first:
  (a) shard-argmax→allgather(simulated by memcpy of W shards)→local-reduce == full-row-set
      argmax on synthetic logits, at W ∈ {2,4,8}, ties included — determinism-equality made
      testable off-device; (b) coordinate arm: remap-before-reduce must DISAGREE (positive
      control for the §4 law — the silent class, forced to speak in a host cell);
  (c) the :349 predicate flip: require_argmax_transport must stop refusing at world>2 ONLY
      when the R1 transport is present (routing cell's existing REFUSE arm is the RED artifact;
      GREEN = new-path arm at world=4, and it must still REFUSE the half-wired shapes —
      RED→GREEN triple, both observing seats named).
- The device kernel arms themselves ride the build step (compile-cells; per-window boot battery
  grades anything device-only; NO boot without written grant).

## Hygiene
- Disk 12 G measured at entry; `lsof +L1` sample at build-time before quoting any delta
  (agent4's honest non-sample rule continues); build ONLY in this worktree.
- Bound row + trigger-(ii) fold under this leg: R1's first-light log IS their arming source;
  readers world=4-certified (a8727267-line), AR-silence declaration per 046e871b: the
  first-light row must say whether kind=ar count>0 (measurement) or =0 (labelled inference).
- NVFP4 planner-of-record lane: plan committed (e96a26b0), artifact-wait legs zero-card —
  unaffected by this leg's duration; will reconcile at N0 on arrival.

## Amendment round 1 (agent1 second-witness 32af13ec + chair ad4d7f7b/9adbcbff) — folded at 71204539
- **:350 DISPATCH (the load-bearing correction):** ring call moved inside `if (ring) { …; return; }`,
  R1 arm below. Acked late but landed-first: chair ruling arrived 12:34Z, the guarded shape was
  already written from agent3's 12:05Z find. Gate Check (t) = the in-tree byte-witness with
  both-direction self-test (RED = predicate-only widen fixture, GREEN = guarded shape); agent3's
  external tool stays the independent second copy (their sha moved twice during the session —
  2e47fe01 canonical, 1a3dd6d2 measured at my seat 12:3xZ — a witness that moves under its own
  stamp is the reason the in-tree copy is pinned).
- **FIND-2 wire shape:** RULED MANDATORY, delivered: 12 B `ArgmaxChampion` named by
  static_assert, every count derived from that sizeof. Chosen deliberately: 12B-packed (the pad
  is the ring's parity-tag surface; the R1 route must never stride by it, or stale pad
  fabricates [ARTAG] against our own witness — agent1's reasoning accepted).
- **FIND-1 conf telemetry — DEVIATION DECLARED (chair seq-28 ruled named-inert interim; I landed
  agent1's option (a) — world-generic conf dispatch).** The honest sequence: option (a) was
  written at ~12:2xZ, before the 12:35Z ruling arrived; it is DONE, compiled, and cell-armed
  (family 7: sentinel preserved, absent-transport throws, %32 mirror pinned). Reverting to
  named-inert now would DELETE tested protection to restore a silence-class the fix kills.
  Chair's scope worry respected as far as it goes: NOTHING in the conf change touches the
  transport's critical path (four getters + r1_conf_at; surgical revert = one command if you
  order it). NAMED-INERT ROW STILL FILED per ruling, scope now narrowed to the residue option (a)
  does NOT close: cross-LANE fold-bit-identity of conf (ring fold vs R1 fold can differ in the
  last mantissa bits; within-R1 it is fixed-order and rank-uniform — verified by the cell's
  ring-literal reproduce arm at rule level, not fold level) + W5 arming semantics at world=4.
  Post-R1 leg, folds under the un-parked MTP geometry work.
- **PASS-3 threading:** noted — R1 inherits dispatch-level single-flight by construction (all
  three enqueues on one rank stream, one rank per thread; the :77 pending-task guard protects
  the dispatch_all route, my route's equivalent is the engine's per-rank serialization — the
  first-light battery exercises exactly this at T=1 decode).
