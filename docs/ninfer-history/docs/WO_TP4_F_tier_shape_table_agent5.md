# WO-TP4-F — generated tier_shape_table (role × world → shard geometry) — agent5 authorship, chair-approved 16:26Z (25a6fc05)

**Class:** the pair-literal family's last member. Tonight killed three occurrences of
"world=2 arithmetic baked into code that receives the world": loader (`tp_world=2`),
GDN state-spec (`5120`/`24` halves), and this one — **shape whitelists that enumerate TP2
halves as data** (`kSupports {5120,6144},{5120,3072},{5120,8704}` in `q3_linear_add_plan.cpp`,
template dispatch arms keyed to the same literals). Quarters verified absent at every family
by grep at e944d237 (`4352`/`1536`: zero hits in q2/q5/q4/w8 plans; `2048`/`1280` hits in w8
are OTHER roles' halves, not quarters — re-check, do not assume).

## Design ruling (chair adopted, this is the spec)
ONE generated table, consumed everywhere:
```
tier_shape_table.h:  per 27B role: {full_n, full_k, alignment} constexpr
                     per world:    {n_w, k_w} = full/W with COMPILE-TIME divisibility
                                   static_asserts (group%64==0 proof rides the entry)
```
- **Compile-time, not runtime:** every legal (role, world) pair is a constexpr instantiation
  site; an illegal world fails the build, not a boot. That is the whole difference from
  tonight's failures — the table cannot disagree with the projection because both read it.
- Consumers to migrate (each = one commit, one cell arm, no big-bang): q3 plan+gemv+gemm_simt
  (STEP 1), then q2, q5, w8, q4, embed-gather rows, attn/gdn input plans, gdn_output_w, the
  mtp/dflash2 shard sites (`n_prop_vocab` already tp_w-derived in the loader fix — fold it into
  the table when its family migrates).
- **Alignment law (VRAM-adjacent):** the group-aligned proofs already computed —
  4352%64=0, 1536%64=0 (down/attn-out quarters); 3072/1536 for {6144}; w8/q4's roles must
  REPEAT the proof per entry — an entry without a static_assert is not allowed into the table.

## STEP 1 = q3 family only (chair staging)
1. New header + q3 `kSupports` becomes a table view; gemv template arms instantiate
   <5120,1536>/<5120,4352>; gemm SIMT same; storage alignment asserts ride the table.
2. Cell registration per chair ruling (3): Check **(v)** = host {1,2,4}×all-27B-roles
   divisibility enumeration over the table (lineage: `check_load_world_consistency.py`'s
   fixture discipline — planted illegal-world RED, legal GREEN, absent-member NOT-OBSERVABLE
   with the member named). Gemini keeps the letter sequence after (v).
3. Bank + warm rebuild; pair taken fresh per chair ("no grant friction, same window terms").

## HONEST ETA answer (chair question 2): can attempt-4-with-q3-only reach first-argmax?
**NO — say it plainly: q3-only moves the wall, it does not clear the path.** The forward that
dies at `linear: unsupported Q3 shape` runs EVERY tier's roles before the first argmax
(embedding/qkv/gate_up/attn-out/gdn-out/down per package.cpp:100: Q3 roles + "rest ref tiers"
= Q4/Q5/W8 per bindings.cpp:302-315). Attempt-4 will die at whichever ref-tier role the
forward reaches next — its NAME unknown until the boot (grep can't order the fused schedule).
So the staged prediction, pre-declared so no boot surprises anyone: **attempt-4's outcome is
"the next named family," exactly as the chair's velocity note intends** — each boot converts
one unknown wall into one table-migration commit; the count is bounded (six families), the
first-argmax moment arrives with the LAST one (likely q4 or w8 — the ones with no quarters
ANYWHERE). First-text grader thereafter: runner 3eeab3b9's LISTENING-BUT-NOT-SERVING state +
sentinel-filtered attractor vs 348e77a1222dea7f, R1's A1TRACE lines as the transport's
in-serve birth certificate. Honest number: **q3 step-1 = half a day incl. cell; full path =
the migration ×5 after it — do not budget first-text on attempt-4; budget attempt-4 on
naming family #2.**

## Table-map input (agent4, feacc4ab — NVFP4 roles/arms + THE CRITICAL DECOY)
Roles needing rows: linear/linear_add/linear_swiglu/attn_input (plan exists, NO gfx900 route -> the row must be a
NAMED LOUD REFUSAL, not a silent BF16 fall-through — pairs with the kv_bytes_for_tier 34*1024 exhaustiveness cell,
agent4 authors it against the generator's emitted enumeration); gdn v/z (G5/G6); KV cache (tier resolves at
tp2_budget.h:224 — NO capacity row, retracted-claim named so nobody re-chases it; the BF16 default is the cell's job);
mtp k<=7 interaction (N5, likely a row); TP-shard admission (materialize_tp — the (u) assertion IS its table row).
**THE TWO /2 FAMILIES MUST STAY DISTINCT IN THE GENERATOR:** nvfp4_shard_image's full_cols/2, src_stride/2 are
FP4-ELEMENTS-PER-BYTE packing halves (int4 lineage — the same decoy class as tp_load.cpp:157-159), NOT world math;
while kSupports/draft-rows halves ARE world math. A generator that sweeps `/2` without reading the callee's own
naming (agent4's retraction + agent1's kv_heads self-catch, two desks, same trap) inherits a false-positive wall.
Row keys: {world 1,2,3,4,8} x {tier NVFP4, FP8_ROW_BF16S, Q2, Q3, Q4, Q5, W8, BF16} x {T=1 GEMV, T>1 GEMM};
worlds 3+8 deliberately included as DIVISIBILITY FALSIFIERS (3 must throw-by-name on 6144 — A-1 precedent).
Coordination: agent4 builds no parallel map; his exhaustiveness cell hooks the generator's grep-able emission (the
new desk emits it first; cell lands with step-1). Attempt-4 runner: 274708da, pins valid (a5ef99e5... until table-
step-1 rebuilds the bank).

## Landing rules (non-negotiable, tonight's ledger baked in)
- Anti-resurrection: tp_engine.cpp/tp2_budget.h ZERO edits in any WO-TP4-F commit (diff-gate
  will show it); table is consumed, never mirrored.
- Every migrated family ships with its (v)-arm extended + RED/GREEN fixtures named at the
  commit that lands it (prose closures don't count).
- Warmup-tell (my cell) is the boot battery's family detector for the duration: a NEW
  `unsupported <tier> shape` throw = next migration, named, no re-scope.
- Credit line for the register: found by boot-3's warmup throw (agent5's cell graded the log
  rc=1 before any human read it), scoped by agent1's tier map generalizing to T2/T3, staged
  by chair; the table itself is tonight's three-class family, final member.

## Neighbor list (agent1, ef220726 — read before touching anything adjacent)
Full file: docs/amd/TP4F_TABLE_NEIGHBOR_LIST_agent1_2026-09-14.md (amd/wo-agent1-support). Binding excerpts for the
table lane: (1) ZERO overlap with t3_gate2 (tp_group.cpp TpGroup:: collectives only; new table file = EXTRA_SOURCES
entry, not a re-scan); (2) legality of 4352/1536 must be COMPILE-proved static_asserts in the table — runtime
divisibility duplicates there would be decoration (the :650 throws already exist for the boot path); (3) DECOYS —
a world-sweep of `2`s MUST NOT touch: tp_load.cpp:157-159 (/2 = int4 PACKING halves), zero_pair[2]/slot_count
= 2*lanes (LANES, A-4 Class-3), kvarn_staged_bytes(16,2,...) — whose 2 is (n_text_layers, kv_heads) MODEL geometry
(agent1 self-caught reading callee PARAMETER NAMES at the declaration — the lesson generalizes: no literal is
pair-shaped until the callee's own naming says so); (4) barrier posture post-T3: arity is world-correct, the live
invariant is arrive-COUNT parity across runners — a hang at the first W5 double-barrier means a rank skipped an
arrive, and that is a POSITIVE finding, not an RCCL ghost; (5) attempt-4's serve-grading first edition = A1TRACE
(R1 birth certificate) + same-arms attractor vs 348e77a1222dea7f (agent1 confirms all three deaths pre-argmax);
step-0/B banner-ABSENT stands as witness-law, no change needed.

**Handoff state:** authored-complete; implementation NOT started by this seat — author's
context at terminal budget, and device-lane work at this edge is how windows get wasted.
Every artifact a builder needs exists: this file, kSupports/gemv/gemm sites grepped, alignment
proofs, (v) cell spec, pin chain, runner 3eeab3b9 armed with BIN/BIN_SHA/TPG_SRC protocol.
