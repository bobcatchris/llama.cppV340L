# A-4 INVENTORY — every rank-pair assumption in the TP2 runtime (agent4, pre-guard-removal survey)

Companion to WO_TP4_all_lanes §A-4. Positions at lane tip 51dc3dd6 (= A-3 e611c398 + G-AMD-26 rows);
line numbers move with merges — the PREDICATE is the search, re-run before use:
`grep -n "rank0_\|rank1_\|std::thread\|{0, 1}\|zero_pair\|slot_pair" src/runtime/tp2/tp2_backend.*`.

## Class 1 — vectorizable NOW, mechanical (the guard's own removal set)
| site | shape | world-4 form |
|---|---|---|
| tp2_backend.h :302-303 `rank0_/rank1_` members + ctor params :1486-87 | two unique_ptr | `std::vector<std::unique_ptr<TpRankState>> ranks_` (ctor already fed from the A-3 ranks vector — move the vector through, drop the split) |
| tp2_backend.h :257 `rank(int r) { return *(r==0? rank0_:rank1_); }` | pair selector | `ranks_[r]` with a bounds throw naming r and size |
| tp2_backend.cpp :3267-3291 batched-runner pair threads (`th1` + inline worker(0) + rank0_ex/rank1_ex) | 1 spawned + 1 inline | loop r in [1,world) spawning, worker(0) inline; `std::vector<std::exception_ptr>` err(w), rethrow-first-nonnull |
| :4330-4360 multi-batch runner, same pattern (`err0/err1`) | idem | idem |
| :7367-7379 single-sequence decode runner (`t0` + inline worker(1)! — note inverted: rank 1 inline here) | idem | idem — the inversion itself is a datum: no invariant depends on which rank is inline (none found; flag if A-4 test says otherwise) |
| tp_engine.cpp `b_opts.dev0/dev1` remaining prose (:922 comment) + the `tp_place_capacity(reader, tp_world, …)` | DONE at A-2/A-3 | — |
| backend world guard :1353-1360 (A-3) + engine guard (A-2) | refuse world!=2 | REMOVAL commit: engine guard relaxes to `world>=2` (TpBackend create-guard relaxes only after Class-2 ceilings below derive; 3 stays refused by divisibility throws from A-1 — 6144%3!=0 proves it), and the 81-census/geometry paper test extends to w=4 rows ALREADY PRESENT in tp4_geometry_host.cpp |

## Class 2 — ceilings that must DERIVE at world=4, not be deleted (map §A-4 law)
| site | today | derivation owed |
|---|---|---|
| one_shot_argmax.h :17-24 kNumSlots=32, kMaxTokens=16, payload comment "32 slots x 2 ranks ≈ 12 KiB pinned" | PINNED STAGING IS PAIR-SIZED | payload ring must size `x world`; the W5 fixed-combine order ("rank-0 partial first") generalizes to a fixed rank-ascending order or the conf bit-identity claim dies across ranks — A-4 test: conf equality across all world ranks, per map the ONLY invariants here are order claims |
| tp_group.cpp advance_epoch / S1 monotonic-epoch (agent3's proof instrument) | per-rank epoch pair | epoch vector + the KAR tuple census re-derived at 4 (agent3's T3 gate family owns the witness; coordinate via chair, their seat) |
| make_rank RoundStateSpec output_rows=248320 :780 (vocab rows GLOBAL, split at ColumnN /world) | 248320 literal | = q3 artifact full vocab; derive from manifest (tp_load already reads it) — at w=4 per-rank rows 62080 (paper-pinned tp4_geometry_host ColumnN @4) |
| template floor T~51 at 2 ranks (map §"warmup/cols arithmetic") | measured-era number | the smem/occupancy compile-check at 12-head GDN slices — ZERO-GPU: does it even build + GetAttributes dump (A-4's device-time half; rides the 4-card window ONLY for the measured leg, the compile leg is CI-able now) |
| batched runner `lanes` math (docs/151 §22.x windows: T<=6 x lanes <= kMaxTokens=16) | lane-scaled, not rank-scaled | independent of world; DO NOT touch in A-4 (lane≠rank is the map's own distinction; conflating them is the 768-charged-to-DFlash2 class) |

## Class 3 — world-INVARIANT, do not "parameterize" (decoration risk, LAW 21)
- GDN slot/conv per-rank geometry (state_slots, conv_channels 5120/rank): already `full/world` via tp_local_shape; `zero_pair[2]` :1200 is the LANES-2 initial state-slot width, not a rank pair — leave (predicate proof: it memcpy's into state_slots sized 2*lanes+cache_slot).
- sampler partials (sampling.cuh/.h): token-domain × columns; columns <= kSamplerMaxColumns=16 is a SHAPE ceiling (batch cols), world-independent; and the greedy path is a pure max (commutative+associative) — carry-in hunt §(i) CLOSED at static, banked where? THIS doc + verdict row if the probe confirms.
- AR call SITES (~128/token, map §2.4 census): each is `group().allreduce_*(rank, …)` — call-site world-generic already; transport decision is WO-TP4-B's seat (RCCL-first per chair, agent5's table).

## Guard-removal evidence set (the commit that deletes the world!=2 refuses must ship ALL of)
1. Class-1 vectorization (mechanical, no behavior change at world=2 — PROOF: era-pinned binary replay at w=2 must byte-match the bfa69f17-era serve on the identical corpus — a boot-class truth, name the stamp).
2. tp4_geometry_host extended: w=4 rows ALREADY there; add kMaxTokens/payload-ring sizing assertions from the DERIVED formula once Class-2 lands.
3. A world=4 TpGroup PAPER construction reaching NCCL-init and failing LOUD (DoD wording: "may fail-loud at device time — the CPU-visible half is full"): hipInit/context paths will actually SUCCEED on this 4-die box, so "fails loud" may instead be "succeeds and releases" — pre-declare BOTH outcomes acceptable, the datum is the CONSTRUCT-without-pair-assumption, not the failure.
4. step-0 anti-res green vs merged amd/main 516883da+ (A-1/A-2/A-3 are main's canonical region since the chair's merge — the guard relax is an authorized forward delta, one commit, named).
5. Release-row law applies to any boot in (1)/(3): manifest, own-pid, release row, cycle class.

**Order (ratified by chair seq-33)**: carry-in hunt (agent1 census inbound) + WO-TP4-E fusion FIRST; this inventory is the A-4 plan-of-record it resumes to.


---

## ADDENDUM (chair 920a6225 seq-43, verified at bytes this desk re-read): GATE-2 — the silent-null ARROUTING HOLE, now A-4 scope
`tp_group.cpp:297-302`: allreduce_argmax is `if (impl_->one_shot_argmax) { … }` with **NO else**;
the one-shot is constructed ONLY at `n==2` (:109-112). At world=4 the call SUCCEEDS while
out_token is NEVER WRITTEN — unread device memory, zero faults, plausible shapes: garbage-silent,
the enemy class. My own A-2/A-3 engine/backend guards currently make this UNREACHABLE (world!=2
refused upstream) — the hole is exactly the guard's removal date, which is why it is A-4's, not
someone's future incident.
A-4 now MUST ship one of the two routings, named in its commit:
  (R1) RCCL-reduce + local argmax per rank — deterministic-equal iff every rank reduces the FULL
       row set (allreduce before argmax); rides B-1's transport decision, no new mechanism;
  (R2) THROW on `!one_shot_argmax` naming "argmax routing lands with R1" — acceptable interim,
       loud, zero-silent; default pick for the guard-relax commit, R1 replaces it inside the
       4-card window's WO-TP4-B decision.
PAPER EVIDENCE gains a NEGATIVE CELL (chair law + Gemini §2.1 day-one rule, registers with its
own falsifier): a world=4 construction reaching allreduce_argmax must either WORK or THROW —
`GATE2: silent-null return is a test failure by definition`. Host-side reachability: call the
routing function with n=4 objects and assert throw (no device needed for the null-check leg;
the works-leg rides the window).
Refinement recorded: Gemini's "no new geometry arm needed" STANDS (shapes); this is ROUTING.

## RED→GREEN CLOSURE LAW (user law 53c09d17) — the item-7 cell pre-registered here
- THE RED ROW IS ALREADY BANKED: 1d0ff3c6 (era bin 8e6d79ae…, same prompt ×5, ONE server,
  5/5 DISTINCT). The probe boot's OFF-arm reproduction on the era bin re-certifies it at the
  new era's feet; cite, do not re-burn.
- PRE-REGISTERED GREEN CELL (lands WITH whatever fix the probe points to — this desk writes the
  cell, the fix inherits it): `results/amd/p3/item7_repeat_replay_cell.sh` — kit-v7-gated boot
  (GRANT-ACK law), bin named by FULL sha, HIP_VISIBLE_DEVICES=2,3 + --devices 0,1, cold class
  named, one prompt "Hi." ×5 sequential requests temp=0 max_tokens=32 against ONE server
  instance, text-only compare (agent3's whole-JSON-hash vacuity caution honored: strip
  id/created, compare reasoning+content), VERDICT: 5/5 IDENTICAL = GREEN; any distinct pair =
  RED stands and the mechanism class re-opens with the tag census attached. Suite-permanent,
  three shas (RED era bin / GREEN fix bin / tree) named in the row.
- DISCIPLINE NOTE: the PARITY arm is DETECTION, not the fix; if it closes (a) zero-tag, the
  GREEN cell still owns its bar to whatever successor mechanism (b)/(c) supplies. A cell that
  can only go green by my instrument's being right about its own innocence is not a falsifier —
  the bar is BEHAVIOR (5/5 identical), instrument-independent.
