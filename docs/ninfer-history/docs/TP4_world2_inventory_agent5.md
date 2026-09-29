# TP4 Phase-2 — the world==2 inventory (agent5, first measured pass)

**Scope:** PLAN_2026-09-13_TP2-then-TP4.md Phase 2, "world==2 inventory" item, executed with this
lane's closure method. ZERO device time; every claim below was measured read-only at
**amd/main @ 426e2ff2** (the main checkout's tree — plumbing reads only, per the frozen-refs law;
my lane branch `amd/wo-gfx900-perm` @ aea36f1c carries this doc, and is NOT the measurement tree).
Digits are instrument output tagged by that tree-sha; where a number is PLAN-carried rather than
re-measured by me, the carrying doc is named. **Status of this pass:** items 1–5 measured; §6
(tp2_backend rank surface) is SCOPED, not measured — claimed != proven is invoked against my own
unfinished section, explicitly.

Session identity note: fresh agent5 session (01a09850); the 01:1xZ lane-collision incident
(I executed agent3's sweep in their tree before reading my own kit) is reported separately to the
coordinator and touches NOTHING in this doc — all reads here were made from the main checkout.

---

## 1. The headline, decomposed: what "world==2 by construction" actually is

`tp_engine.cpp` (context of the :862 comment, auto-capacity probe): "This engine is TP2 (world==2)
unconditionally." Measured, the construction has **five distinct load-bearing sites**, not one:

| # | site (path@426e2ff2) | hardness class at world=4 |
|---|---|---|
| 1 | `one_shot_allreduce.cu` kernel signature: `peer_host_buf/peer_flag/peer_gen`, reduce = `local + peer` (kernel body, reduce loop) | **HARD by signature** — one peer, one phase |
| 2 | `OneShotAllReduce::Impl`: `host_epoch[2]`, `dev_epoch[2]`, `host_status[2]`, `host_gen[2]`, `Slot::host_buf[2]` | **HARD by storage** — fixed-size-2 arrays |
| 3 | `one_shot_argmax.cu` kernel: `peer_host_payload`, `peer_flag[t]` per token; winner fusion + `emit_conf`/`my_conf` per rank | **HARD by protocol** — pairwise winner + conf fusion |
| 4 | `tp2_budget.h`: every calibration is per-rank-at-2 (see §3) | **MEASURED-at-2** — re-derive/re-measure, never scale |
| 5 | `tp2_backend.cpp` conf-arm chain: "consume IDENTICAL confidence sequences" (r0+r1 matched-pairs symmetry, e.g. the `prod` loop context at the W5 arm block) | **HARD by symmetry contract** — 4 rings must agree |

**The group layer is NOT a hard-2:** `TpGroup::size()` returns `options.devices.size()`
(tp_group.cpp, `impl_->n`), and `ncclCommInitRank(&kp->comm, this->size(), ...)` already passes the
group size through. A 4-card group constructs; nothing at the group layer refuses world=4.
`tp2_backend.h` holds one `TpGroup` and derives `rank` per thread — the refusal lives above it.

**Load-side placement is NOT a hard-2** (the surprise of this pass — see §2): the sharding
machinery is world-parameterized with pure arithmetic.

## 2. Placement / load: world-generic, derived clean at 4

`tp_load.cpp` `multi_ranges(role, rank, world)` — all seven roles re-derived at world=4 (measured
from the committed constants, not inferred):

| role | full rows (block layout) | /4 quotients | exact? |
|---|---|---|---|
| `MultiRangeGateUp` | [gate 17408; up 17408] | 4352 + 4352 | YES (17408 = 4·4352) |
| `MultiRangeQKV` | [q 6144; k 1024; gate 6144; v 1024] = 14336 | 1536 / 256 / 1536 / 256 | YES |
| `MultiRangeQK` | [q 3072; k 512] | 768 / 128 | YES |
| `MultiRangeGV` | [gate 3072; v 512] | 768 / 128 | YES |
| `MultiRangeGQK` | [q 1024; k 1024] | 256 / 256 | YES |
| `MultiRangeGZ` | [v 3072; z 3072] | 768 / 768 | YES |
| `MultiRangeGQKV` | [q 2048; k 2048; v 6144; z 6144] = 16384 | 512 / 512 / 1536 / 1536 | YES |

- **GQA ratio preserved per-rank by construction** (MultiRangeQKV): 1536:256 = 6:1 = the global
  6144:1024. The split takes proportional q/kv slices, so per-rank GQA semantics hold at 4 without
  new code.
- **The GDN 2:1 skew (PLAN: "must be DERIVED, not divided")**: MultiRangeGQKV keeps the
  split-consumption order [q_l; k_l; v_l; z_l] per rank (the A1 5563465b contract — a contiguous
  half gives rank0 all q/k, garbage GDN). At world=4: q_l=k_l=512 rows, v_l=z_l=1536 rows. The
  v:q skew per rank is 3:1, IDENTICAL to global — because the ranges are proportional slices of
  each sub-block, not halves of the fused row range. **The arithmetic is clean; the open item is
  kernel-tile alignment**: the GDN head count per rank at 4 (12 of 48 per PLAN's 48-heads note)
  must tile the chunked kernels' NS16/NS32 panel geometry — that is a dispatch-table check
  (§5's verify-point), not a load-path change.
- **The REAL hard-2 on the load side is upstream — the artifact shape** (PLAN blocker-question #1,
  agent2's manifests): `multi_ranges` assumes the artifact carries FULL tensors and H2D-shards
  them (`materialize_tp(reader, rank, world, ...)`); a TP2-shaped q3 export (per-rank K-halves
  whitelisted per the promote history) cannot produce a 4-way split from 2-way objects without
  recombination. The `Q3G64_F16S=129` TP-degree question is downstream of the same fact — 129 is
  an artifact-manifest count (q3 ART arms-proof), so its TP-degree dependence is **coupled to
  agent2's artifact-shape answer**, not independently resolvable from the engine tree. This doc
  claims nothing about it beyond the coupling.

> **COUPLING RESOLVED 2026-09-13** (agent2 `922ec410` answered → `5e89bb0d` retracted-and-
> corrected; agent2 #572): **the artifact is NOT TP2-shaped.** All 1118 tensors declare FULL
> shapes — `gate_up [34816,5120]` (full fused gate+up), `qk [7168,5120]` full; layouts
> row-split-k128-v1 ×439 / contiguous-le-v1 ×679. Nothing halved, nothing per-rank. **TP4 is
> an engine port, not an export decision; it does not cross Green's promote machinery; the
> §6 engine-surface slice is UNBLOCKED and re-ranked to the correct next step** (the #366(B)
> note had the hypothesis as the finding — inverted by measurement). Sole residual per
> `5e89bb0d`: a ROLE-MAPPING question answerable in one read-only pass over the plan — 81
> tensors at groups_per_row=9 under declared group_size 128; RowK axis at risk if any take it
> (`check_div` throws at 4 ranks); ColumnN clear (rows 4-divisible); `k128` tag reading
> flagged suspect (one family non-128-multiple). Derive at slice time by name from the plan,
> intersect with the 81 — never infer from shapes (agent2's method note; their original
> "no shape field" premise was absent-key read as absent concept, retracted).
>
> **Coordinator-adopted sentence (ruling #573, verbatim, for this doc's resolution lines):**
> "open coupling resolved — artifact is rank-agnostic; sole residual is a role-mapping
> question, no export required, answerable in one plan pass."
>
> **Derivation caution (agent2 #574(3), farewell note):** the 81-count and four-shape list
> above are agent2-computed under a possibly-wrong `k128` tag reading (4304 % 128 ≠ 0) and
> an already-confessed axis slip — re-run the groups_per_row arithmetic from plan role +
> declared group_size as the slice's OWN measurement; treat their list as the hypothesis to
> test, never the ground truth to intersect against blindly.

## 3. Budget model: four re-derivation rows (VRAM LAW applies to one of them)

`tp2_budget.h` is per-rank by design ("Per-rank VRAM budget model for the TP2 27B backend") —
nothing in it multiplies by a rank COUNT; all hard-2s are calibrations:

1. `static_weights_bytes` (9059 MiB legacy) — already overwritten by
   `tp_place_capacity(reader, world, ...)` at the knowing consumers (rank-independent capacity,
   tp_load.h's own comment). At 4: **no code delta** — the placement reader re-measures; the
   fallback constant remains TP2-era (acceptable: no-manifest fallback only).
2. `kv_bytes_per_token` (18496 I8 / 34816 BF16 / 9792 q4-class) — the comment decomposes as
   "2 sides × 256 ch × bytes/code": **256 ch is the per-rank channel half** (512 full). At 4
   ranks the per-rank channel count halves again ⇒ every tier constant re-derives (128 ch ⇒ codes
   halve; the 4608 B scale side table does NOT — fixed). kvarn_kv_bytes_per_token() is already
   parameterized by (k_bits, v_bits); the missing parameter is the channel count, currently
   hardwired inside the page-bytes arithmetic. **This is a code delta, not just calibration.**
3. `kTp2RuntimeReserveBytes` (1536 MiB, bf16 tiers) — MEASURED FLOOR at TP2 geometry (collective
   buffers, graph memory, draft head, GDN ckpt, fragmentation). At 4: the collective-buffer and
   graph terms change shape. **VRAM LAW: re-measure at 4-rank geometry or the number does not
   exist; scaling it is banned.** The two incident anchors (U ≥ 1047 MiB lineage) are TP2-shaped
   evidence and do not transfer.
4. `decoder_fixed_bytes` (592) and `dflash2_drafter_bytes` — term classification needed at 4:
   GDN ckpt (73.4 MiB) is per-rank REPLICATED (docs/70) — stays; DFlash2 weights replicate per
   rank — stays; the 512 decoder-fixed core needs its shard-vs-replicate decomposition derived
   before any 4-rank figure is quoted.

`required_bytes`/`max_context_fitting` ring math is per-rank and rank-count-agnostic — no delta.

## 4. Transport + argmax: the pairwise wall, and where the AMD line actually stands

- The one-shot AR kernel is a **single-phase pair reduce** (`local + peer`, one peer in the
  signature). The epoch/gen machinery (KAR-v2 stamps, ring-wrap cadence, change-gated dev_epoch
  upload) is pair-local and rank-count-agnostic — it SURVIVES a 4-rank world if the algorithm
  becomes a pairwise tree (2 phases), but the **wrap cadence doubles per token** (2 collective
  phases/token/rank ⇒ epoch advances 2× as often). The host epoch/storage ([2]-arrays, §1 items
  1–2) must generalize regardless.
  **WRAP-CADETNCE PRECISION (agent5, 2026-09-13, on agent4's 32-slot ring correction — my "128-call
  gen wrap" phrase above was the sloppy one):** there are TWO rings, not one: ALLREDUCE
  `kNumSlots=128` (`one_shot_allreduce.h:14`) at 128 AR sites/token = wraps EXACTLY ONCE PER TOKEN
  (steady-state, every slot recycled every token — safe by the per-call epoch handshake, but no
  retro slot read survives even one token); ARGMAX `kNumSlots=32` (`one_shot_argmax.h:14`) at 1
  call/token = wraps every 32 tokens (agent4's wrap caveat for host-side conf reads lives HERE).
  Any geometry/KAR mapping quoted "N calls back" must name WHICH ring it indexes — "128-call" named
  neither and is retired from citation by this row.
- Options at 4 (PLAN names them; costs are PLAN-carried, not re-measured here): staged no-P2P
  pairwise tree (3.13 GiB/s measured class, PLAN §transport), rccl bring-up, one-shot extension.
  **AMD-line fact this pass adds:** at 426e2ff2 the HIP whitelist (src/HipSources.cmake) carries a
  measured NOTE that `one_shot_allreduce.cu` / `one_shot_argmax.cu` / `tp_kernel.cu` are
  DELIBERATELY NOT whitelisted — Step-5 device-port targets (PTX `ld.global.cv/st.global.wt` with
  "l" constraints invalid on GCN). So the pairwise kernels are CUDA-line code TODAY; whatever the
  AMD TP2 path uses for its measured transport, TP4's AR choice intersects the port question from
  the first hour. The delta map must carry both facts together.
- **Argmax conf fusion at 4:** the winner-selection payload protocol is pairwise (peer payload per
  token, t-indexed flags). At 4 ranks: either a 2-phase tree (payloads compose: max of maxes,
  sum of sumexps — associative, tree-safe) or an n-way protocol. The `emit_conf` rings and the
  engine's armed-early chain (tp2_backend W5 arm block: per-rank conf rings read via
  `one_shot_argmax_conf_at(rank, ...)`, window product with stale-check `c_base + c_len > a_now`)
  rest on the **2-rank identical-sequence symmetry** ("matched r0+r1 pairs" in the run-3 note).
  At 4, ALL FOUR rings must stay sequence-identical per step — the stale-check arithmetic is
  per-rank-local and generalizes, but the SYMMETRY is a property of the collective schedule, and
  it is the thing to re-prove (not re-assume) at 4 ranks.

## 5. Template floors: rank-invariant on the token side — one verify-point, no inference

All TP roles are **row-splits** (output rows shard; the 5120 input dim and the TOKEN count T are
replicated per rank — every rank computes its row-shard for all T tokens). Therefore the GDN
gating template floor observed at T~51 (the 17e "template floors at T~51" terms; the cols=51
unsplit step of the 0x8000 fault) is **T-side and rank-count-invariant**: at 4 ranks the same
floor applies to the same T. The ONE verify-point (dispatch table, not inference): whether any
gating/gemm template variant keys on the per-rank N (rows). At 4, MultiRangeGateUp ships 4352
rows/rank (vs 8704 at 2) — if the dispatch ladder has N-keyed rungs, the fired variant changes.
Check: the bf16_gdn_gating_proj dispatch table at 426e2ff2 (launcher's variant selection), a
read-only table read. NOT yet done this pass — flagged as the map's first-hour cell alongside
agent2's artifact answer.

## 6. tp2_backend.cpp rank surface — SCOPED, largest single item, not yet measured

460 `rank` mentions at 426e2ff2 (grep -c, this pass). This is the engine's sequencing core and the
one surface this pass did NOT close. Sub-questions the closure pass must answer (my -E -M
reachability method, next session-hour):
- per-step AR call graph: which steps issue collectives, in what order, and where the pair
  symmetry is ASSUMED vs ENFORCED (barrier pairs, ring resets at :1894/:2120, skew-history lines
  at :3711-3712);
- the armed-early chain's ring-reset interaction (:2304 "zero BOTH" — at 4 ranks, zero ALL);
- rank-indexed storage beyond the one-shot impl (any other `[2]` or `rank^1` idiom);
- the warmup lifecycle (:4059 rounds=8 at ~4 steps) — warmup must exercise the SAME collective
  schedule the real request does, at 4 ranks, or its skew history diverges (the :2302 lesson:
  warmup/real rank flip at identical geometry was already a live class at 2).

**Claimed != proven is invoked here against myself:** no statement above §6's bullet list is a
measurement of the backend surface; it is the scoped plan for one.

## 7. Scheduling + upside rows (PLAN-carried, names attached)

- Card collision: TP4 bring-up consumes all four cards as one unit — dev2-as-window and the
  dev0/1 serve pair both die. Phase-1 items 5/6 must be consumed or explicitly deferred first
  (PLAN Phase 2 "scheduling collision"). Not re-measured (no device work this pass).
- VRAM upside: ~3416 MiB/card weights-class + per-rank floor anatomy ×4 → ~4+ GiB/card headroom,
  120k-class context as TP4's measured goal (PLAN Phase 2 "VRAM"). PLAN-carried; the reserve
  re-measurement (§3.3) is the gate that turns this from estimate to anchor — LITH/VRAM law.

---

## Receipts (all read-only, tree = amd/main@426e2ff2)

- tp_engine.cpp:862 context (auto-capacity comment block, read this pass)
- tp2_budget.h: full-file read; constants and comments quoted verbatim above
- tp_load.cpp: multi_ranges switch (roles/quotients), classify_tp role map (:260-:283 class)
- tp_load.h: tp_local_shape/materialize_tp/tp_place_capacity signatures + measured-basis comment
- one_shot_allreduce.cu: kernel signature + reduce loop; Impl [2]-arrays; advance_epoch/reset_step
- one_shot_argmax.cu: peer payload/flag protocol; emit_conf/my_conf contract
- tp2_backend.cpp: W5 arm block (:2860-class), ring resets, warmup/skew comment lines (scoped §6)
- tp_group.cpp: size()/ncclCommInitRank world plumbing
- HipSources.cmake: one_shot/tp_kernel exclusion note (verbatim, Step-5 class)

Couplings: ~~artifact-shape + 129 → agent2 (blocker #1)~~ **RESOLVED 2026-09-13** (`5e89bb0d`/
#572): artifact rank-agnostic, full shapes, engine port — see the §4a resolution block; residual
= plan role-mapping over the 81 groups_per_row=9 tensors (read-only, one pass, at slice time).
Gating dispatch-table N-keying →
this lane, next pass with §6. Transport cost per token → PLAN's measured anchors, re-quoted at
map time.

---

## §6 ADDENDUM (same session, first closure slice): the backend's hard-2 is CONSTRUCTION-shaped, not arithmetic-shaped

Measured at 426e2ff2 after the doc above was banked (b91a36df carried §6 as scoped-only):

- `grep -c rank` = 460 overcounts the hardness. The pairwise ARITHMETIC lives in the one_shot
  kernels (§1/§4), NOT in tp2_backend: zero `rank^1` / `1 - rank` idioms found in the backend
  itself (grep this pass).
- The world==2 in tp2_backend is concentrated at CONSTRUCTION: `group->ctx(0)` + `group->ctx(1)`
  (literal indices into the world-generic group), `make_rank(..., 0, ...)` / `make_rank(..., 1, ...)`
  (exactly two constructions, literal rank ids), `d2_ranks[2] = {r0, r1}` + `for (int r = 0; r < 2;`
  (d2 bind loop — BOUND hardcoded, BODY rank-generic via `d2_ranks[r]` + `cudaSetDevice(st.ctx.device)`).
- The one_shot ring plumbing is rank-PARAMETERIZED at all 10 measured call sites
  (reset_one_shot_step / one_shot_argmax_rank_step_now / one_shot_argmax_conf_at take rank).
- r0/r1 direct mentions: 11/13 — construction, park/restore, sequencing callsites.

**TP4 delta row (map-grade):** generalize CONSTRUCTION (loop `for r in [0, group->size())`:
  ctx(r), make_rank(..., r, ...), ranks[r] array) and the backend body follows; the remaining
  world-2 work is the one_shot layer (§1 items 1-3, §4) + the conf-symmetry re-proof + the
  "zero BOTH → zero ALL" ring-reset class (the :2304-family). The [2]-array grep noise
  (zero_pair, slot_pair, hash pairs, row_dump) is token/lane-pair storage, NOT rank-pair —
  classified and dismissed this pass (read matched lines, not totals).

Still open in §6 (second slice, next session): barrier-pair map, warmup collective schedule
  equality at 4 ranks (:4059-class), AR-count per step at 4 (feeds the transport decision),
  and the accept/e2e sequencing around the arm chain. Claimed != proven stands on those.

## §6 queue addendum (from agent2's #563 premise retraction, commit 5e89bb0d)

The v340l/17 TP4 answer's premise changed: manifests DO declare shape/layout/format (1118
tensors; row-split-k128-v1 ×439, contiguous-le-v1 ×679) — "no shape field" was absent-key read
as absent concept. Headline survives on the BETTER argument (gate_up [34816,5120] and
qk [7168,5120] declared FULL — nothing halved → rank-agnostic, engine port). NEW PLAN GATE
for the §6 second slice, cited not restated: under declared group_size=128, 81 tensors have
groups_per_row=9 (9%4=1) and one family is non-128-multiple — before parameterising the four
TpRole cases, resolve which of those 81 are RowK-split; if any are, TP4 needs a group-size or
role decision that DOES cross the export (check_div throws at 4 ranks). Axis law from the same
retraction: groups_per_row uses the COLUMNS axis (kSplitK/128), not shape[0]. This lane's docs
never carried the retracted premise (grep-verified 2026-09-13); my §6 slice now STARTS from the
RowK-role resolution, sequenced post-wave as before.
