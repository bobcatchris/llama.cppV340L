# WORK ORDER 154 — MultiBatch for all KV dtypes (int8, bf16, int4; kvarn in companion order)

**Status:** ACTIVE (numbered by coordinator 2026-09-05; replaces the DRAFT filename) ·
**Recon:** A1, 2026-09-05, CPU-only · **Mainline+GPU:** A1 ·
**Tests:** gemini (branch `wo/kvarn-multibatch`, target `ninfer_batched_sampling_test`) ·
**Companion doc:** `docs/WORKORDER_kvarn_multibatch.md` (kvarn tiers, same lane) ·
**§8 = RUNNING LOG** — append entries, do not spawn new docs.

---

## 0. RESUME HERE

Fresh session: read §1 (what exists), §2 (the four gaps), §3 (phases). Setup identical to the
companion order §0 (worktree `wo-kvarn-multibatch`, build flags incl.
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DBUILD_TESTING=ON
-DNINFER_BUILD_BENCHMARKS=ON`). GPU keys: no lease paperwork; note usage. Never `pkill -f`.

**NEXT STEP: Phase C (G3) — see docs/HANDOFF_agent1.md (this worktree) for the full session
resumption note: Phase C hot-start (kernel half + coupled gate collapse, alloc sites, DoD),
open items, protocols, mesh identities. Phase S is MERGED to main (cbf613fa); Phases A+B
complete + verified on this branch. After Phase C: Phase D. The S2 k4v2 order-dependence is
a TRACKED OPEN GATE (coordinator-ruled non-blocking; localization instrument designed, §8).**

## 1. What exists today (verified against e002548d+)

| dtype | single-seq | batched attention | notes |
|---|---|---|---|
| bf16 | fused-append `gqa_attention` | **NONE — throws by design** | multi-lane = per-lane, serialized |
| int8 | fused-append `gqa_attention` | **NONE — throws by design** | same |
| kvarn_k4v2 | shared-prologue kernel | **SHIPPED** (MTP+greedy only) | docs/139 2a; docs/132 48/48 bit-exact |
| kvarn_k5v4 / k4v4 | shared-prologue kernel (Step D route) | gated OFF (companion order Step 2) | MultiBatch=true never compiled |
| int4 (q4_0) | refused — **prefill fills I4 cache as bf16** | none | decode prologue+dispatch ARE wired |

Dispatch: `tp_engine.cpp:180 run_batch_dispatch` — `can_batch = kvarn && lanes>1 && self_mtp &&
!NINFER_BATCH_DISABLE`; non-MTP batched is gated OFF after the docs/143 serve-path crash
("gqa_attention_cached_batched: invalid shape for valid columns" — 500s on every concurrent
no-speculation request). The in-process harness proved the non-MTP step bit-exact; only the serve
path is broken.

## 2. The four gaps this order closes

- **G1 (kvarn route):** companion order — widen `tp_engine.cpp:184`, launcher MultiBatch route,
  first compile of `<MultiBatch=true, KBits=4/5>`.
- **G2 (batched non-MTP serve fix):** the valid_columns shape bug that 500s non-MTP batched on
  the serve path. In-process bit-exactness is already proven; this is a serve-plumbing fix.
- **G3 (bf16/int8 batched kernel):** `gqa_attention_cached_batched` is KVarN-only
  (`gqa_attention.cpp:626`). Extend to BF16/I8 caches — the MultiBatch kernel structure carries
  (paged views + valid_columns masking); the KVarN dequant stage is replaced by direct bf16/int8
  reads. No fused-append change: the fused-append single-seq path stays for lanes=1.
- **G4 (int4):** PREREQUISITE — per-TKV prefill cache-fill for Int4Group64
  (`gqa_attention.cpp:141/205`: `code_dtype = I8 ? I8 : BF16` drops int4 into the BF16 arm, so
  prefill fills an I4 cache with bf16 values = plausible garbage). The int4 decode
  prologue+dispatch are already wired; only prefill fill is missing. Then extend G3 to Int4Group64.

## 3. Phases & GPU plan

- **Phase A (G1):** companion order, as written (Steps 0–7 there).
- **Phase B (G2):** reproduce the 500 (no-spec + lanes>1 on kvarn), fix valid_columns shape
  handling, add the rule-14 mutation test (revert → 500 again). ~0.5–1 day.
- **Phase C (G3):** dtype-generic batched attention for BF16/I8 (greedy-only — see §5b); dispatch eligibility widened;
  gemini tests T1(batched-vs-plain bit-exact)/T2(lane isolation)/T4(registry enumeration)
  extended to both dtypes; per-tier perf sanity vs per-lane. ~2–3 days.
- **Phase D (G4):** int4 prefill cache-fill (quantize-to-codes during prefill, mirroring the
  decode commit path), refusal drop, then Int4Group64 through G3's batched path. ~1.5–2 days.
- GPU: A1 runs all validation; gemini authors tests CPU-side and hands commits.

## 4. Falsifiability / mutation gates (rules 14/15)

- Every new test: deliberate-break run recorded (revert the fix/arm → test fails with the named
  assertion).
- Single-seq sealed baselines (`<5,4>`/`<4,2>`/`<4,4>` MANIFEST hashes) must remain untouched —
  re-verified after each phase; drift = finding.
- Batched-vs-plain BIT-EXACT per tier is the contract (docs/132): a near-tie-flip failure there
  is a real defect, not an acceptable attribute.
- Rule 15: the bf16/int8 batched kernel must REPLACE no existing logic (there is none — additive),
  but the `code_dtype` branch fix (G4) must be substitutive: grep the I8?I8:BF16 form empty after.

## 5. Workspace gate decision (`gqa_attention.cpp:470`)

`gqa_attention_workspace_capacity_bytes` rejects every dtype except BF16/I8 — bench cannot size
KVarN workspaces. DECISION: **in scope, Phase C** — when the batched kernel goes dtype-generic,
this gate becomes tier-generic in the same change (one whitelist collapse instead of two), and
bench gains KVarN sizing for free. Risk noted: KVarN workspace shape must be sized from the
paged-view geometry, not copied from the bf16 arm.

## 5b. RULED: GREEDY-ONLY IS UNUSABLE — batched sampling is the FIRST fix (user ruling 2026-09-05)

User, verbatim: Qwen 3.8 recommended settings are temperature ~1.0; greedy is used only for
benchmarking. A greedy-only batched path therefore does not serve the model's normal operating
mode. **Ruling: batched sampling ships FIRST — before the dtype-kernel phases (G3/G4).**

THE PRIMITIVE ALREADY EXISTS (`include/ninfer/ops/sampling.h`):

- `ops::sample(logits[B rows], out[B], token_domain, SamplingConfig[B], logical_positions[B],
  purpose, workspace, stream)` — one token per batch ROW; per-row SamplingConfig on device;
  **greedy and stochastic rows coexist in one invocation** (temp<=0 rows take exact min-argmax).
- RNG is **counter-based on (configs[b].seed, logical_positions[b], purpose) with no mutable
  RNG state** and no dependence on the compact row index — so reproducibility survives lane
  reassignment and batch composition, the exact problem that made batched sampling look hard.
- Penalties (presence/frequency via token_counts), top_k (clamped ≤20), top_p, min_p are per-row.
- `sampling_workspace_capacity_bytes(token_domain, min_lanes, max_lanes)` sizes the substrate;
  "speculative acceptance uses the same workspace substrate for its verification columns" — the
  MTP accept/resample sampling is designed into the same op.
- CITATION CORRECTION (2026-09-05, A1): the single-seq MTP verify callsite previously cited
  here as ":2071 sampled accept/resample" is `speculative_accept_greedy_drafts` (GREEDY verify
  accept that takes a SamplingConfig array). The real single-seq SAMPLED mirror source is the
  **temp>0 plain-decode arm at `tp2_backend.cpp:1754-1763`**: allgather the two 124160 shards
  into full [248320] → `ops::sample(..., kSamplePurposeDecode, ...)`.

THE FIX (new Phase S — BEFORE G3/G4):
  S1. Per-lane `SamplingConfig` array in the batched runner: tp_engine already carries each
      member's `config.sampling` (the single-seq path builds `host_cfg` per request at
      `tp2_backend.cpp:1138-1151` and copies to device `st.sample_cfg`); batched needs the
      array + per-lane `logical_positions`.
  S2. Batched decode step: temp<=0 rows keep cross-rank `allreduce_argmax` (bit-exactness
      preserved); temp>0 rows take `ops::sample` on the full-vocab logits for that lane.
  S3. Batched MTP verify: mirror the single-seq sampled accept/resample (`:2071` pattern) per
      lane using the same workspace substrate.
  S4. Gates:
      - TIE-BREAK EQUIVALENCE (rule 14): the sampler's greedy branch is MIN-argmax ("lower
        token id breaking ties"); `allreduce_argmax`'s tie behaviour must be proven identical
        on exact FP ties, else greedy batch outputs change vs today. Constructed-tie test
        required, not assumed.
      - Reproducibility: same seed + same logical position ⇒ same token, across runs AND across
        lane assignments (re-run the same request in a different lane).
      - Coexistence: a mixed batch (some rows temp=0, some temp=1) must leave greedy rows
        byte-identical to a greedy-only batch.
      - Statistical sanity: acceptance at temp=1.0 < acceptance at greedy (expected for
        speculative sampling); report the delta, no bar.
  S5. Eligibility gate widened: drop `temperature==0` from `eligible()`; keep lookup-draft
      exclusion (single-seq-only drafting source) until separately ruled; keep uniform-mtp_k.

Sequencing: S1-S5 land FIRST (they fix the existing k4v2 batched-MTP path and unblock sampled
traffic), then Phase A/B/C/D proceed on a sampler-complete batched runner.

## 6. Blockers / risks

- **int4 blocked on G4's prefill fix** (correctness, not scope).
- `<MultiBatch=true, KBits=4/5>` and the bf16/I8 batched kernel are FIRST-EXECUTION code — latent
  compile/codegen issues expected at Step G3 start.
- gemini's T-battery depends on Phase A's route landing first (T1 drives the public op).
- docs/143's shape bug (G2) is serve-plumbing: the failure mode was a 500 on every concurrent
  request — the reproduction is cheap but the fix touches admission geometry.

## 7. ETA (A1 mainline + GPU; gemini tests in parallel)

Phase A 1–2d (parallel with this doc) · B 0.5–1d · C 2–3d · D 1.5–2d ⇒ **~5.5–8.5 days** end-to-end,
gemini's test authoring overlapping. Sequencing vs DFlash2 landing: per coordinator, implementation
starts after DFlash2 lands; recon/scoping (this doc) is complete now.

---

## 8. RUNNING LOG (append-only)

### 2026-09-06 ~06:5xZ — A1: (vi) instrument state checkpoint — per-layer VLHASH probe needs scope refinement; FLIP confirmed, cards released to A2 for 2b

Added per-layer per-column hidden hash (`[VLHASH]`, NINFER_MB_HASHPT-gated, Verify phase) inside
`run_layers`, plus B0_seedhid (round-0 prefill hidden) and B2c_verifyhid (verify hidden at write
time). Findings that ARE trustworthy from the ordered-read runs:

- **B0_seedhid matches across orders under correct request pairing** (sampled 5b04 both, greedy
  59d3 both) — the round-0 prefill hidden is order-INVARIANT.
- **B2c_verifyhid (round-1 verify hidden at write time) DIFFERS across orders in all columns for
  both requests** — so the divergence enters INSIDE the round-1 verify forward (`target_verify_batch`),
  with matched inputs (round-0 prefill hidden, drafts, windows, positions) and before the alignment
  forward. This SUPERSEDES the B3-era "alignment forward" pin (same-lane pairing artifact) and the
  "rotation" clue (stale-read artifact).

**NOT trustworthy yet:** the per-layer VLHASH breakdown — the probe fires in more run_layers call
contexts than the batched verify (blocks with only the embedding dump = empty layer ranges), so the
per-layer attribution needs a call-site scope refinement (only hash when the call is the batched
verify, or filter by stage range). This is the next session's first step, plus a read of
`attn_mix_tp`/GDN batched ops for a batch-max-coupled geometry (the remaining candidate class).

Per the coordinator boundary: (vi) stays a tracked open defect (bf16/i8 S2 invalid until fixed +
formal test). Cards RELEASED to A2 for 2b device work (the next (vi) step is instrument redesign +
code reading, CPU-side, until a new device instrument is designed).

**GDN batched-op audit (started, CPU-side):** the round-1 verify's GDN path is
`gdn_mix_tp` (text_context_impl.h:1903) → `causal_conv1d_silu_snapshot` (per-column state
snapshots: read slot = initial_slots[b], write base = snapshot_base_slots[b]) →
`extract_bf16_columns` ×3 → the gated delta-net recurrent ops — all slot-selector driven
(`mb_slt2` slices), per-lane, request-deterministic by inspection. Audit targets for the next
session: `detail::causal_conv1d_snapshot_launch` kernel (per-batch addressing) and the batched
delta-net recurrent kernel — the shared-by-both-dtypes candidate class for the slot-dependent
round-1 verify hidden. The full-attention verify kernel is already read (per-slot pure).

### 2026-09-06 ~06:2xZ — A1: (vi) ordered-read re-run moved the site upstream; supersedes the rotation clue

**Instrument-ordering bug found and fixed in my own B2b/B3 reads:** the diagnostic D2Hs used plain
`cudaMemcpy` (default stream) against NonBlocking stream-s work — unordered, potentially stale
(the handoff's own stream rule). All five diagnostic reads now `cudaMemcpyAsync(..., s)` +
`cudaStreamSynchronize(s)`.

**Ordered-read result (int8, both orders, correct request pairing):** the divergence is visible at
every level from the round-1 alignment INPUT onward — v_vhid (alignment input hiddens) differs in
ALL columns (t=0..3, both requests) at round 1; v_aligid, v_arh, v_prop, and the round-2 drafts
cascade. Round-1 v_vhid = the round-0 accepted span's verify hiddens — so the contamination enters
at or before the round-1 verify hidden stream / staging reuse. That is UPSTREAM of the alignment
forward — deeper than the B3 pin, and inherent multi-hour (staging-buffer lifetime/ordering audit
across the round, likely first-execution MultiBatch code per §6).

**FLIP per the coordinator's boundary: (vi) becomes a tracked open defect** (bf16/i8 S2 gate stays
invalid until fixed + formal test); partial-merge of (iv)+(v)+S2-gating proceeds. State of (vi) for
the tracker: order-dependence CONFIRMED real (multi-token text divergence from ~token 3 under lane
order flip on bf16/i8 fused-append; kvarn bit-exact); eliminated: attention split boundaries,
stale inactive-split partials, order-assigned seeds, B==N remap, verify-forward numerics (round-0/1
accepts matched); pinned-ish: the round-1 v_vhid/alignment stream carries order-dependent bits
(ordered reads); candidate mechanisms: staging-buffer reuse across rounds/lane assignment, or a
batch-max-coupled geometry in the fused-append verify hidden path. Next instrument (fresh session):
hash the round-1 verify hidden per column at WRITE time (inside the verify forward), plus the
round-0 prefill hidden, to split write-side vs read-side.

### 2026-09-06 ~05:5xZ — A1: B3 PINNED the (vi) site — selected accepted hidden (v_arh) differs at ROUND 1; site = the round-1 alignment forward / select

`B3_draft` hash point (per-lane v_arh = selected accepted hidden + v_prop = draft logits, added to
the round loop after the alignment forward + select + draft head). Result (int8, both orders, 37
common rounds): **first arh diff at ROUND 1, lane 0**; first prop diff round 1 lane 0 (consequence
— the draft head is deterministic given arh). Everything upstream is byte-identical across orders:
accepts, frontiers, anchors, positions, windows, and round-0/1 verify logits (B2_verify).

**Pinned site: the ROUND-1 ALIGNMENT FORWARD** (`mtp_forward_decode_batch` over the accepted span
`[T,B]`, `Envelope{maxa,maxa}` — first-execution MultiBatch MTP-layer code, the docs/154 §6 class)
**or its `select_accepted_hidden`** — with all visible inputs byte-identical.

Mechanism hypothesis (unverified): pad columns beyond licensed repeat the anchor position and
re-append its KV slot ("idempotent") — idempotent only if the pad column's hidden equals the
anchor's; masked columns may carry stale/order-dependent hiddens, so the re-append writes
order-dependent values into a slot the valid columns then attend. Next instrument: dump v_aligid +
v_vhid columns at round 1 under both orders.

Per the coordinator boundary (deep = flip): **recommend partial-merge** — (iv) + (v) + S2
kvarn-only gating are correct and verified; (vi) draft-chain order-sensitivity continues as a
tracked lane with the site pinned.

### 2026-09-06 ~05:3xZ — A1: (vi) LOCALIZED TO THE DRAFT-HEAD CHAIN — all visible inputs byte-identical across orders at the diverging round; one bounded instrument remains (B3 draft-logit hash); RECOMMEND partial-merge now

**Instrument iterations:** (1) B1_pack's hash contains lane-id terms (slt/ringbase), so its
"round-1 diff" was an artifact — fixed by printing the RAW fields (win/slt/ringbase/drafts/
anchor/F/slot) alongside the hash. (2) NEW B2_verify hash point: per-round per-column hashes of
the FULL-vocab verify logits (mb_vfull) under the same env.

**Raw round-2 comparison (results/phase_c/hashpt_order2/):** EVERY visible input matches across
orders — win (72/69), ring read slot (5/9 = ringbase+slot, bookkeeping), anchor (25/2716), F
(68/65), cur_slot (3/3) — accepts matched through round 1, and round-0/1 VERIFY LOGITS are
byte-identical (B2_verify). The ONLY divergence: the ROUND-2 DRAFTS themselves — sampled draft-1:
11316 (A) vs 13901 (B); greedy draft-1 matches (328=328) but draft-2/3 differ (7734,799 vs
814,20139). The round-2 verify-logit diff then follows trivially (different proposal tokens →
different verify columns).

**So the order-sensitive stage is INSIDE the round-1 draft-head chain** (embedding(accepted id) →
MTP layer attention → norms → draft LM head → cross-rank argmax), where every VISIBLE input is
byte-identical. The remaining unaudited kernel: the batched MTP draft forward
(mtp_forward_decode_batch — first-execution MultiBatch code per docs/154 §6's own warning). Next
bounded instrument: B3 hash of mb_prop (draft logits) + hid_next per (round, step, lane) — pins
whether the MTP hidden or the draft LM logits diverge first, and at which draft step.

**RECOMMENDATION (coordinator boundary):** flip to the PARTIAL MERGE now — (iv) cb9c12af + (v)
7ff8fd95 + S2 kvarn-only gating are correct, device-verified, and kvarn-CI-safe; track the
draft-chain order-sensitivity as the open (vi) item (same class as S1-i8-composition: bf16/i8 S2
gate stays invalid until fixed + gemini formal test). The draft-chain investigation continues as
its own bounded lane.

### 2026-09-06 ~04:4xZ — A1: (vi) DIAGNOSTIC CORRECTION — the fused-append order-dependence is NOT attention-split numerics; it enters in round 0's draft chain (hash-proven). Ruled fix does not match mechanism; escalated to coordinator before any fix attempt

**Method:** new `tools/smoke/diag/hashpt_order_probe.sh` — int8, two single-batch sessions per
order (A: greedy=lane0/sampled=lane1; B: sampled=lane0/greedy=lane1), staggered fire, per-round
per-lane `NINFER_MB_HASHPT` capture, split per run. The B1_pack hash covers the verify INPUT
GEOMETRY (window F+T, ring read slot, ring base, per-lane drafts) — not state values.

**Result (results/phase_c/hashpt_order/divergence_timeline.txt):**
```
B1_pack:   rounds_common=37  first_sampled_diff=1  first_greedy_diff=1
B4_accept: rounds_common=37  first_sampled_diff=2  first_greedy_diff=2
B5_state:  rounds_common=37  first_sampled_diff=2  first_greedy_diff=2
```

**Reading:** round 0's pack, accept, and state ALL MATCH across orders (byte-identical inputs AND
identical accepted tokens). The first difference appears in ROUND 1's PACK — i.e. the DRAFTS
produced by round 0's draft head differ between orders (win/ring-base components derive from
matched quantities; the drafts are the new information). Both lanes diff from round 1 — the
perturbation is lane-symmetric, and only the sampled chain's chaotic draws make it visible in text
(divergence at the ~3rd generated token, per the s2 artifacts).

**Hypotheses ELIMINATED on the way (each was the working theory at some point):**
1. Attention split boundaries over flattened batch·width (the ruled fix's premise) — the bf16
   MultiBatch small-T kernel and its reduce are per-slot pure (read in full: per-slot pos/row/
   valid-column indexing; the reduce recomputes active splits from positions; inactive splits are
   never read). No cross-slot coupling found.
2. Stale inactive-split partials corrupting the reduce — reduce guards by recomputing
   active_split_count; matches the partial kernel's guard.
3. Order-assigned RNG seeds — translate.cpp: request without seed inherits the SERVER seed
   deterministically (--seed 20260905); both lanes share it; order-invariant.
4. Compact-vs-original config remap post-shrink (the S1 class) — invisible at B==N; the diag
   diverges at B==N round 1.

**Consequence for the ruling:** the per-batch-slot split-alignment fix targets the attention
partitioning, which the hash timeline now EXONERATES (round-0 accept identical — the verify
forward computed identical results under both orders). The mechanism lives in the ROUND-0→1
DRAFT chain: the batched draft-head forward for round 1 (or the state it reads) is
order-sensitive. Next diagnostic (bounded): LICDBG-style round-0/1 full-vocab logit dump per lane
under both orders → confirms whether the round-1 draft logits differ and in which lane/entry —
then the fix targets the confirmed site.

S2 kvarn-only gating (already landed) remains correct under any outcome. Escalated to the
coordinator before attempting the ruled fix, per "flag wrong instructions before executing".

### 2026-09-06 ~04:0xZ — A1: (v) materialize_tp mtp/ filter LANDED + device-verified (241 MB/rank recovered on no-MTP serve; mtp-on unchanged)

**(v) the docs/151 §22.19 (d.5) mtp/ filter candidate is now LANDED.** `materialize_tp(...,
include_mtp = true)` — when false, mtp/ tensors are NOT placed on device; `tp2_backend.cpp` passes
`mtp` (= `options.mtp_k > 0`).

**Consistency argument (the trap, defused):** the binder already holds mtp/ tensors at
ValidateOnly when `StartupFeatures.mtp()` is false (bindings.cpp mtp_placement) — and both the
binder flag and my filter flag derive from the IDENTICAL source (`options.mtp_k > 0`), so placement
and binding cannot disagree. The pattern has a production precedent: text/draft_head is bound
ValidateOnly AND never device-placed today. Object INDICES never shift — the artifact keeps one
slot per reader object (null when unplaced); pass 2 re-checks the name predicate so default-filled
plans for skipped objects cannot alias index 0.

**Device-verified, both directions:**
- no-MTP serve (`--spec none`): "materializing text-only (mtp/ filtered)" — **materialized 9059 →
  8818 MB/rank = 241 MB/rank recovered** on this kvarn-100k config. Note: A2's 430 MB was a
  binder-plan delta at a different config; the direct device recovery measured here is 241 MB/rank
  (W8G32 mtp/ weights ≈ 380 MB full → per-rank sharded ≈ 241 net of ring rounding). Generation
correct through the full binder path ("Blue", finish=stop).
- MTP-on serve: "materializing FULL" — 9059 MB unchanged, filter inactive, full serve CI cell
  green (kvarn default dtype; also re-validates the kvarn-only S2 gating from the (iv) commit).

DFlash2 consequence: serving with the DFlash2 drafter (weights from the sidecar, mtp() false) gets
this recovery automatically. Header-note irony recorded: tp_load.h already claimed "MTP-off" in its
comment — the code now matches what the comment always said.

NEXT: the fused-append lane-slot-sensitivity FIX (per-batch-slot split alignment, per the
coordinator ruling) — the last item before the Phase C merge.

### 2026-09-06 ~03:5xZ — A1: (iv) §5 :470 collapse LANDED + device-validated (4 configs × zero UNDER, all 5 kvarn routes incl. batched) + S2 gate → kvarn-only per ruling

**(iv) the gqa_attention.cpp:470 workspace gate now ACCEPTS the KVarN tiers with a route-exact
bound.** `detail::gqa_attention_kvarn_capacity_bytes` (gqa_attention_kvarn.cu — needs
sku_launch_config + the kernel constants) implements the §8 22:2x draft bound route-exactly:
max(decode pm/pl/pa, materialize k/v/bt at the envelope's n_tiles ceiling =
div_up(max_visible_keys,64)+1) + small-T partials, every sub-term aligned to the arena's 256-B
alloc alignment. Single-seq and batched entries covered (batched ⊆ st_term at T×batch).
`gqa_attention_workspace_capacity_bytes` accepts DType::KVARN_K4V2 (the one marker for every kvarn
tier) and routes there; BF16/I8 paths untouched; layouts planner sites (266/332/366/439) get real
numbers for kvarn plans for the first time. Unit suites green (ninfer_gqa_attention_test,
ninfer_kvarn_gqa_test rc=0).

**TC4 instrumentation: `NINFER_WS_PEAK=1`** ("0" = off) → every kvarn route reports
`[WS-PEAK] route=… base=… peak=… route_peak=… bound=… ok|UNDER` — begin() resets the arena
high-water at route entry and returns the live base; report() diffs. Routes: kvarn_unified,
kvarn_decode_merge, kvarn_materialize_smallt, kvarn_materialize_prompt, kvarn_batched_unified.
Diagnostic-only: it discards the process-level high-water while enabled — never set it in
measurement runs.

**The instrument caught two defects in my own first attempt before any claim was made** (this is
the rule-14 direction working as designed):
1. First form compared the route bound against the arena's CUMULATIVE high-water (71.5 MB — every
   other phase's allocations) → meaningless UNDERs. Fixed: route-local begin/report semantics.
2. Raw-size summation UNDER-counted tokens=1 by exactly 128 B: the arena pads each alloc to 256 B
   (pm/pl/st_m/st_l are non-multiples at T=1). Fixed: every sub-term ceil-aligned. T=4 is now
   BYTE-EXACT (route_peak == bound == 7,332,864); T=1 carries +640 B margin.

**Validation (device, real 2-lane serve rounds, greedy+sampled pairs, batched dispatch confirmed
4×):** kvarn_k4v2 default (unified), NINFER_KVARN_DECODE=packed (decode_merge),
NINFER_KVARN_VERIFY=materialize (materialize_smallt + prompt), kvarn_k5v4 (slice6 unified):
2104/2674/3358/2598 single-seq lines + 2248/2248/2248/1470 batched lines — **ZERO UNDER** across
all four configs and all five routes. gemini's TC4 formal test (the §8 22:2x sketch: arena.reset →
run → peak vs bound) is now directly executable; the in-launcher instrument provides the route
attribution. Formal test remains gemini's per TEST SPLIT.

**S2 gate → kvarn-only (coordinator ruling, this session):** serve_batched_ci.sh now gates S2 only
for kvarn* dtypes; bf16/i8 S2 diffs print the tracked-defect warning without failing the cell,
until the fused-append per-batch-slot split-alignment fix + gemini's formal test land (fix is next
in my queue after the mtp/ filter).

Driver note for posterity: ws_peak_validate.sh's first form hung on a bare `wait` after firing the
request curls — with no args it waits on the SERVER child too, which never exits. Wait the curl
PIDs explicitly; stop sequence polls health before escalating to SIGKILL on its own PID.

NEXT: materialize_tp mtp/ filter (tp_load.cpp:152, binder index consistency per bindings.cpp:181,
own commit + device verify) → the fused-append lane-slot fix.

### 2026-09-06 ~01:5xZ — A1: (iii) bf16 OOM CLOSED + NEW FINDING: fused-append (bf16/i8) sampled chain is lane-slot-SENSITIVE (kvarn is not) — coordinator ruling needed on gate semantics

**(iii) bf16 serve OOM: CLOSED, three verified legs.**

1. **Fail-fast direction:** bf16@100k now refuses at the preflight with arithmetic —
   `required 17443 MiB / usable 16310 MiB (fixed 13595, context 3848, slack −1133)` — instead of
   OOMing at cudaMalloc mid-materialize. (Honest note: on TODAY'S itemized model bf16@100k already
   refused at −365 pre-reserve; the incident's "+404 slack then OOM" premise was an older model
   vintage (fixed 12059 = no headroom/drafter terms). The reserve's real protection: configs
   passing with slack ∈ (0, 1536) on the bf16 tier — the incident class — now refuse.)
2. **bf16 cell passes:** at KV_CAPACITY=MAX_CONTEXT=40960 + media shave (media-cache-mib 0,
   media-live-mib 512), the full serve CI cell runs: P0/B1/B2/B3/S1 all ✓ (bf16 MTP materializes
   and serves — the cell previously died before P0 completed). bf16@60k MTP was attempted first
   per the pre-plan and OOMed at cudaMalloc with +279 MiB nominal slack — that failure is
   LOAD-BEARING EVIDENCE (below), not a scoping miss.
3. **No false refusals:** i8 cell at 100k: P0/B1/B2/B3/S1 ✓ (S2: see the finding).

**The reserve, final form — measured and dtype-conditional.** `Budget::kTp2RuntimeReserveBytes =
1536 MiB`, charged ONLY for BF16 tiers, on BOTH the auto-capacity probe and the preflight
(no-divergence rule). Derivation from two device points on the current model lineage: bf16 MTP
OOMed at +279 MiB nominal slack WITH a 768 reserve charged ⇒ unmodeled U_bf16 ≥ 1047; 1536 = floor
+ ~490 margin. Charging it dtype-blind was TRIED AND RETRACTED on device: i8@100k — a proven config
(2026-09-05 CI + D5 ladder ran it for hours) — falsely refused at −247. The deficit is
bf16-MTP-specific: i8@100k passed at +243 MiB slack on the same vintage ⇒ U_i8 ≤ 243. bf16 cell
defaults encoded in serve_batched_ci.sh (60k default was ALSO tried and OOMed — the fitting config
is 40960).

**NEW FINDING (first exposure — bf16/i8 S-cells never completed before this session): the
Phase-C fused-append batched route is lane-slot-SENSITIVE in the sampled chain.** Truth table
via s2_order_diag.sh (now dtype-parameterized), A1/A2 = same-order, B1 = order flip:

| dtype | batched route | A1 vs A2 | A1 vs B1 (flip) | diag |
|---|---|---|---|---|
| kvarn_k4v2 | KVarN arm (lane views, fixed accumulation) | SAME | SAME — bit-exact | exit 0 (gate CLOSED, 00:3x entry) |
| bf16 | fused-append | SAME | **DIFFER** | exit 2 |
| i8 | fused-append | SAME | **DIFFER** | exit 2 |

Plus the racy-rendezvous CI corroboration: bf16 S2 ✗ (sampled differed across identical servers;
greedy companion differed too), i8 S2 ✗ (sampled differed; greedy reproduced). Reading:
**determinism holds everywhere** (A1==A2 on all three dtypes — no process nondeterminism), **lane
ISOLATION holds** (S1 coexistence ✓ on all three — no cross-lane value flow), but the fused-append
arm's logits shift at last-ulp with the lane SLOT (likely split boundaries over the flattened
batch·width column space); greedy argmax usually survives the perturbation, sampled resample draws
amplify it into different tokens. kvarn's own arm is bit-exact under flips — consistent with its
fixed-order accumulation design.

**Ruling needed (coordinator):** (a) attribute-vs-fix — the fix is a kernel change (pad/align split
boundaries per batch slot in the fused-append arm; formal test per the TEST SPLIT rule is gemini's);
(b) gate semantics meanwhile — S2-as-written is sound for kvarn and flaky-by-attribute for
bf16/i8 (racy rendezvous × slot sensitivity). Options: order-pin the S-cells (stagger, like the
diag) + report-only S2 on fused-append dtypes, or S2 gated on kvarn only. NOT decided unilaterally.

**NEXT:** (iv) §5 :470 collapse per PRE-PLAN 2 while the ruling pends. Evidence:
results/phase_c/{bf16_cell_40k_reserve1536,i8_cell_reserve1536,i8_cell_reserve_bf16only,s2_order_diag_bf16,s2_order_diag_i8}.log
+ s2_order_diag_{bf16,i8}/ artifact dirs + the 60k OOM log (bf16_cell_60k_reserve.log).

### 2026-09-06 ~00:3xZ — A1: S2 GATE CLOSED + S1-i8 NEGATIVE GATE COMPLETE (deterministic trigger, both directions)

Re-grant (coordinator, 00:16Z, in writing, both cards) executed in the ruled order. Fresh session
resumed from docs/HANDOFF_agent1.md §0 + the staged RULE-14 comment.

**1. (ii) S2 re-diag: PASS — tracked-open S2 gate CLOSES.** `s2_order_diag.sh` with the fix in
(b73fd4aa binary): A1==A2 (same order) AND A1==B1 (order flip), exit=0. The k4v2 order-dependence
was the SAME remap defect as S1-i8 (the §8 22:1x candidate attribution confirmed): the sampled
lane no longer shifts when lane order flips, because its accept row now reads its own compacted
config at every B. Evidence: `results/phase_c/s2_rediag_fixin/` (texts byte-equal, 548 bytes
each; dispatch + per-lane lines). S2's "verify-numerics" alternative branch is NOT needed — the
state-leak vs verify-numerics fork resolves to the accept-remap mechanism.

**2. S1-i8 negative gate: COMPLETE — the RULE-14 run-3 inconclusive is retired.** New deterministic
trigger `tools/smoke/diag/s1_det_trigger.sh` (the staggered-fire lever the run-3 comment pinned:
NINFER_BATCH_WINDOW_MS=500 + 300ms stagger, SAMPLED request fired FIRST = deterministic lane 0 /
leader; sampled lane finishes first → B shrinks 2→1 with live=[greedy]; greedy text compared
against a same-binary all-greedy concurrent baseline). Three runs, KV_DTYPE=int8, seed 20260905:

| run | binary | verdict | greedy bytes ref/trig |
|---|---|---|---|
| control | fix | S1_PASS (exit 0) | 496 / 496 |
| mutated (`live[j]`→`j`) | RULE-14 mutation | **S1_FAIL (exit 1)** | 496 / **505** |
| fix re-applied | fix | S1_PASS (exit 0) | 496 / 496 |

The defect is no longer ~50%: pinned order fires it EVERY time (1/1 mutated FAIL vs 1/2
pre-fix concurrent). Baseline G_ref byte-identical across all three runs — mutation-insensitive
as pre-stated, so the reference is valid for both directions. Control run first proves the
stagger machinery itself is output-neutral on the greedy lane (any mutated delta attributes to
the remap, not the harness). Evidence: `results/phase_c/s1_det_trigger/` (control=20260905_202222,
mutated=20260905_202540, reapplied=20260905_203108) + logs `s1_det_trigger_{FIX_control,MUTATED,FIX_reapplied}.log`.
Mutation applied as an uncommitted 1-line working-tree edit, rebuilt, run, reverted via
`git checkout --` — the committed tree never carried it.

**S1-i8-composition fix (b73fd4aa) is now DONE by its own §8 22:2x standard:** fix-in 2/2 CI PASS
(lane-flip, committed at e681ffff) + deterministic mutated-FAIL + re-applied PASS. Source comment
at the vcfg_h block updated to record completion (the stale "NEXT:" text retired).

**NEXT:** (iii) bf16 serve OOM per PRE-PLAN 1 (cell budget shave + preflight runtime-reserve
~768 MiB world==2), then (iv) §5 :470 collapse per PRE-PLAN 2, then (v) materialize_tp mtp/ filter
(tp_load.cpp:152, binder object-index consistency per bindings.cpp:181, own commit).

### 2026-09-05 ~22:2xZ — A1: STATUS MARKER + GPU-window pre-plans

S1-i8-composition fix (b73fd4aa) = COMMITTED, **PENDING GPU VERIFICATION** — NOT done until
BOTH directions pass on device: (i) fix-in → full i8 CI 0 failed ×2 (lane-flip coverage);
(ii) revert-mutation → S1 must FAIL. Incomplete ⇒ fixup/revert. Staging ≠ completion.

PRE-PLAN 1 — bf16 serve OOM (light): symptom = preflight says fits (slack 404 MiB at 100k:
fixed 12059 + context 3847 vs usable 16310) but cudaMalloc OOMs mid-materialize ⇒ the fixed
bucket under-counts real TP2 runtime overhead (NCCL buffers, graphs, draft head 106 MB, GDN
ckpt 73.4 MB/rank, fragmentation). TWO-PART FIX: (a) cell-level — bf16 CI cell runs with
KV_CAPACITY=60000 + shaved budgets (media-cache-mib 0, media-live-mib 512); (b) preflight —
add a runtime-reserve term so it fails fast with arithmetic, not cudaMalloc:
`required += (world==2 ? kTp2RuntimeReserveMiB /*≈768, measured from this incident*/ : 0);`
then re-test bf16 at 100k (expect clean preflight refuse OR successful materialize at 60k).

PRE-PLAN 2 — §5 :470 launcher-peak instrumentation (TC4): measure the arena high-water of a
REAL kvarn round vs the capacity bound; NO launcher edit needed — the arena already tracks
its peak. In the kvarn ops test/bench:
```
arena.reset();
run_real_round(/*k4v2+k5v4, envelope=max_visible_keys, tail present, tokens=1..6, batch 1+2*/);
peak = arena.peak_bytes();                    // device-measured launcher peak
bound = gqa_attention_workspace_capacity_bytes(q_heads, KVARN_K4V2, env, batch, 1, 6);
if (bound < peak) fail("TC4: capacity bound under real launcher peak");   // rule-14 gate
```
plus env-gated `[WS-PEAK] route=... peak=...` stderr in the two kvarn launchers for route
attribution. Implements AFTER the bound lands; the one open read is the prompt-route flash
allocs (tokens>6 materialize tail).

### 2026-09-05 ~22:2xZ — A1: §5 :470 gate-collapse PREP (alloc inventory; implementation waits for GPU validation)

Consumer note: gqa_attention_workspace_capacity_bytes feeds the RUNTIME layout planner
(layouts_impl.h:266/332/366/439), not only bench — the collapse must not OVERSIZE (persistent
pool cost on 16G cards) nor UNDERSIZE. kvarn serve is unaffected today (planner guarded).
ALLOC INVENTORY (route-exact, from gqa_attention_kvarn.cu):
- Unified decode/small-T: slots = kvarn_decode_splits(54 this SKU) × kv_heads × tokens ×
  kKvarnQBlockGroup(16); pm/pl FP32 {slots} + pa FP32 {slots × kKvarnAttnD(256)}. Tokens≤6:
  ~10.7 MB total.
- Materialize (tokens>6 / NINFER_KVARN_VERIFY=materialize): n_tiles = packed_pages +
  (tail_count>0) — RUNTIME tail state; conservative ceiling n_tiles_max =
  div_up(envelope.max_visible_keys,64)+1. k_temp/v_temp BF16 {256, 64, kv_heads, n_tiles}
  (kKvarnAttnD=256, kKvarnAttnG=64) ≈ 262 KB/tile pair + bt I32 {n_tiles}; then small_t
  partials (tokens≤6 forced path) or prompt-route launch (tokens>6 — flash allocs NOT yet
  inventoried — CLOSED: gqa_attention_prompt_attention_launch allocates NOTHING from the
  workspace arena (verified gqa_attention_prefill.cu:16-60) — inventory COMPLETE).
- Batched launcher (:696+): st_acc BF16 {256, q_heads, tokens×q_batch, splits} + st_m/st_l
  FP32 {q_heads, tokens×q_batch, splits} — tier-independent; splits =
  gqa_attention_split_capacity(..., BF16, envelope); route extras per scoping (packed_verify
  / k5v4_small_t_verify / materialize / T=1 partial).
DRAFT BOUND: cap = max(unified_term, materialize_term + small_t_term) with n_tiles_max from
the envelope; VALIDATION (GPU, = gemini TC4): instrument workspace peak on a real k4v2/k5v4
round, bound must be ≥ launcher peak. Prompt-route flash inventory = the one open read.

### 2026-09-05 ~22:1xZ — A1: S1-i8-composition ROOT-CAUSED + FIXED (verification pending GPU window)

Serve i8 S1 (coexistence) failed once / passed once across two runs = intermittent, ~50%.
ROOT CAUSE (Phase-S defect, NOT dtype-specific, surfacing on the newly-opened non-KVarN
path): the verify-accept call passed mb_cfg (ORIGINAL-lane-indexed SamplingConfig array) to
speculative_accept_greedy_drafts instead of the COMPACTED mb_scfg the Phase-S setup builds
for exactly this purpose (the plain sampled arm passed mb_scfg correctly; the verify arm was
the miss — the compaction comment was right, the call missed it). The op reads
configs[COMPACTED row]; at B==N compacted==lane order so it was invisible; once B shrinks,
row b ≠ lane b: greedy lane on original lane 1 + sampled lane 0 finishing first maps the
greedy row to compacted row 0 holding the SAMPLED config → the greedy lane's accept goes
STOCHASTIC → output changes vs the all-greedy batch. Matches: intermittency (rendezvous
lane assignment + shrink order), S2 pass (identical servers), B-cells pass (no sampled
rows), kvarn S1 historical passes.
FIX (one substitutive token + dead-code retirement): accept call now passes mb_scfg;
mb_cfg alloc/upload retired (existed solely for this call); comments state the
compact-vs-original contract at the upload site. CPU-rebuilt clean; NOT yet verified on
device — GPU runs stood down per coordinator (orphan-server incident: my killed loop left
a serve child holding ~9.3G with a dead parent; released, cards clean).
VERIFICATION PROTOCOL when GPUs return: (i) fix in — full i8 CI ×2 (lane-flip coverage),
expect 0 failed; (ii) rule-14 mutation — revert the one-liner, S1 must FAIL. ~18 min.
CANDIDATE ATTRIBUTION: the same remap bug may BE the tracked-open S2 k4v2 order-dependence
mechanism (sampled lane remapped onto the other lane's seed/temperature post-shrink → draw
shift; greedy byte-stable since greedy configs are identical). Cheap test: re-run
s2_order_diag.sh with the fix in — a pass would close the tracked-open gate.

### 2026-09-05 ~21:5xZ — A1: PHASE C (G3) COMPLETE — batched runner is dtype-generic (bf16/i8/kvarn)

SCOPING CORRECTION (found before writing any code): the "multi-hour first-instantiation
campaign" premise was wrong. All 8 {MultiBatch,Masked}x{Append,Cached} template arms were
ALREADY COMPILED for the bf16 and i8 decode kernels (nm-verified on
build/src/.../gqa_attention_decode.cu.o) — the small_t launch dispatches MultiBatch at RUNTIME
on invocation.batch_size, and the kernel bodies already carry per-batch pos/column_base,
table_rows[batch], valid_columns[batch], batch-strided partials. Phase C's real work was
plumbing, and the second scoping premise ("route already wired at :1735, dead only behind
:2426") was also wrong in a load-bearing way: the batched fused-append arm existed only in
attn_mix (non-TP); attn_mix_tp (THE live TP2 path) had no batched non-KVarN arm and would have
attended B lanes' flattened columns as ONE sequence. (First session finding "active_* never
assigned" was a regex artifact — they bind via ScopedValue/ScopedPositions RAII; retracted in
writing to the coordinator.)

IMPLEMENTATION (one substitutive change, coordinator ruling (a)):
1. attn_mix_tp (text_context_impl.h): NEW batched non-KVarN arm — 4D {D,heads,width,batch}
   views + valid + kv_table_rows into fused-append ops::gqa_attention, local head counts
   (mirror of attn_mix :1735). valid_columns is bound only by the MTP verify entry; the
   non-MTP decode entry runs width==1 where Masked=false is correct.
2. tp2_backend.cpp :2426: the docs/138 M4 non-kvarn per-lane FALLBACK REMOVED (rule-15
   substitutive gate per ruling (a): grep "fall back to the production" / "only NON-kvarn
   caches fall back per-lane" EMPTY ✓). kvarn_batched flag now selects the KVarN per-lane
   TAIL STATE, not admission. Guards on every direct st.kvarn_lane_ws[]/st.kvarn_ws deref:
   tail migration (text+mtp) unconditional→guarded; INV-3 (INVASERT), POOLTRACE, MB_DBG
   env-gated sites guarded. st.kvarn_ws.text[l] on an empty vector is UB, not a null deref —
   the guards are load-bearing. Setters/rewind self-guard on kvarn_lane_ws_==nullptr.
3. tp_engine.cpp can_batch: `kvarn &&` dropped — admission is dtype-generic.
4. THE REAL ROOT CAUSE (harness RED → fixed): the batched runner's per-lane PREFILL never
   wrote io_.text_kv_table_row / io_.backend_kv_table_row (only init :550 and single-seq
   :1317/:1543 do). bf16/i8 fused-append prefill keys its cache row off those scalars →
   every lane's prompt K/V appended into ROW 0, each later lane overwriting the earlier
   lanes' prompt at the same positions. bf16/i8 failed with exactly that signature (lane0
   wrong from its FIRST generated token; i8 byte-identical to bf16's failure = dtype-
   independent plumbing; kvarn immune — its append/attend use lane views + lane-id arrays).
   NINFER_MB_SEQ_STEP=1 reproduced the identical divergence (state-determined, not forward-
   determined) — the probe that was VOID for the MTP round is exactly right here. FIX: per-
   lane row writes before each lane's prefill (mirror of the single-seq writes).

VERIFICATION (results/phase_c/, all committed): harness ninfer_tp2_batched_decode_test —
bf16, bf16 --mtp 2, i8, i8 --mtp 2, kvarn (regression): 5/5 PASS rc=0, 48/48 tokens MATCH
sequential per lane per config, ~1.24x aggregate throughput. Rule-15 greps: old fallback
EMPTY, KVarN-only throw STAYS (ruling (a): it is architecturally correct — bf16/i8 ride
fused-append; routing them through gqa_attention_cached_batched would duplicate a batched
route). Marker gate v2: 0. Serve CI bf16 + i8: see the follow-up entry.

REMAINING (was §5-coupled, coupling dissolved by ruling (a)): the gqa_attention.cpp:470
workspace-gate collapse (accept KVARN_K4V2 with a conservative paged-view bound ≥ launcher
peak). NOT in this commit: a numeric capacity model needs the route-exact alloc inventory
(materialize k_temp/v_temp/bt n_tiles ceiling from envelope, unified pm/pl/pa, small_t
partials) done against the launcher, not sketched — next session with fresh budget.
Phase D (G4 int4 prefill fill) unchanged per §0 item 5.

### 2026-09-05 ~16:2xZ — A1: gap-1 (draft-vocab tie-break) CONFIRMED + Phase S design pinned

Verified in `one_shot_argmax.cu`: local tie → min local idx; cross-rank tie → min GLOBAL SLICE
idx (`rank*n_rows+max_idx`); **remap AFTER** (`final_tok = draft_vocab_ids[best_idx]`). Draft
vocab is JSON-loaded (`tp2_backend.cpp:620`) with NO monotonicity guarantee → min-slice→remap
≠ min(final_tok) in general. Design (told gemini + coord):
- Plain-decode stochastic rows: allgather both 124160 shards → `ops::sample` on FULL 248320
  vocab (exact mirror of :1754-1760); tie domain == `allreduce_argmax`+nullptr. ✓
- MTP verify sampled accept: same full-vocab domain; matches greedy verify arm (nullptr). ✓
- Stochastic drafting: MOOT — verified the single-seq sampled path drafts GREEDILY even at
  temp>0 (propose sites stay argmax+remap; `speculative_accept_greedy_drafts` models proposals
  as one-hot). No stochastic drafting exists on any runner; propose sites unchanged.
- Greedy rows keep `allreduce_argmax` on every site (docs/132 bit-exactness untouched).
- gemini's TS1 correction: tie test must assert min-slice-then-remap equivalence, incl. an
  explicitly NON-MONOTONIC remap case (where naive implementations diverge).
Gaps 2 (workspace sized once, no resize) and 3 (eligible negative gates) adopted as ruled.
Doc numbered **154** by coordinator; DFlash2 runs in PARALLEL (coord-confirmed).

### 2026-09-05 ~16:5xZ — A1: Phase S S1/S2/S3/S5 LANDED (wiring compiles; device gates pending)

All edits in `src/runtime/tp2/{tp_engine,tp2_backend}.cpp` on `wo/kvarn-multibatch`:
- **S5** `tp_engine.cpp eligible()`: dropped `temperature == 0.0f`; lookup-draft exclusion and
  uniform-mtp_k kept (negative gates for gemini TS-suite).
- **S1** batched runner (Phase S setup block, decode-loop preamble): per-lane
  `SamplingConfig` built from each `req.sampling` (mirror of the single-seq host_cfg build),
  uploaded ONCE to `mb_cfg` (was mtp-only all-greedy; now unconditional, real configs).
  Per-lane `token_counts` slices [N×248320] allocated+zeroed ONLY when any lane has penalties
  (sampling.h non-aliasing rule). `mb_cfg` no longer re-uploaded per verify round.
- **S3** verify accept: `speculative_accept_greedy_drafts` now receives the real per-lane
  configs — the op was ALREADY sampling-capable (batched path just fed it all-greedy cfgs).
  RNG positions derive from old length inside the op. Seeding/propose/AR sites untouched
  (greedy one-hot drafting, see gap-1 entry).
- **S2** plain batched decode step: kept the fused `allreduce_argmax` on ALL columns (one-shot
  handshake needs identical call counts on both ranks; greedy lanes keep exact results), then
  a sampled arm: compact stochastic lanes → per-lane `allgather_local_bf16` (sendcount in
  ELEMENTS — the verify-allgather lesson) into pre-allocated `mb_sfull` → one `ops::sample`
  over the compacted rows (purpose=Decode, logical_position=cpos — identical keys to the
  single-seq draw) → overwrite those lanes' `tok_pinned`. No per-round staging allocs (arena
  is bump-allocated; runner scope spans all rounds — all buffers pre-allocated like `mb_*`).
- **Gap 2 guard**: backend setup now fail-fasts if `state->work` cannot cover
  `sampling_workspace_capacity_bytes(248320, 1, max_lanes)` + single-seq accept + BATCHED
  accept (`speculative_accept_greedy_drafts_workspace_capacity_bytes(..., 1, max_lanes)`).
  No runtime resize path exists by design.
- `ninfer_engine` builds rc=0. NEXT: S4 device gates (gemini TS1-TS4 on this branch; GPU
  grant requested from coordinator), then serve-path bring-up via the existing CI cells
  (`NINFER_MB_SERVE_MUTATE` battery must still fail a deliberately broken path).
- KNOWN LIMITS (for the record): sampled MTP batched vs single-seq is NOT claimed
  bit-identical (verify-width numerics + statistical acceptance; docs/132 contract is
  greedy-row bit-exactness + sampled reproducibility, not cross-route sampled identity).

### 2026-09-05 ~17:0xZ — A1: two self-caught defects fixed BEFORE device runs (84ee2882, this commit)

Full-diff mechanism review caught two latent bugs the old all-greedy code made invisible:
1. **Compacted-vs-original lane indexing**: both sampled consumers read configs[b] by
   COMPACTED row while mb_cfg/lane_cfgs_h are ORIGINAL-lane-indexed — with B < N (any lane
   finished/cancelled) rows would read another lane's temperature/seed/token_counts.
   Fix: compact live lanes' configs into mb_scfg for BOTH the verify accept and the plain
   sampled arm. RNG keys are row-index-free, so compaction is RNG-safe.
2. **Stream ordering**: compute streams are cudaStreamNonBlocking (device.cu:67) — a
   default-stream cudaMemcpy carries no ordering to stream-s kernels. The sampled-token D2H
   (host commit could read mb_sout before ops::sample finished) and the setup mb_cfg H2D are
   now async-on-s + sync. Same pattern as the single-seq plain_tok_pinned read.
Rule-14 note for gemini: TS-suite should include a lane-finish/cancel mid-batch case (B<N)
so the compaction stays exercised — that is exactly the path the old code never covered.

### 2026-09-05 ~17:1xZ — A1: KVarN closeout addendum — name-table item CLOSED (as distinct, not collapsed)

Coordinator-assigned closeout item ("name-table fix note"). Finding: the two tables A2 flagged
are NOT rule-15 duplicates — `request_log.cpp kv_cache_name` emits the REQUEST-LOG SCHEMA
strings ("int8-group64", "int4-group64"; artifact_type + schema_version consumers) while the
canonical `kv_cache_storage_name` (types.h:100) emits help/warning text ("int8", "q4_0"); the
kvarn tier names match across both. The KvarnK4V4 lag A2 caught was already fixed on main
(0ab1d504). A substitutive collapse would silently change the log schema — recorded as
RATIFIED-DISTINCT with a cross-reference comment at the kv_cache_name switch (comment-only,
zero behavior change; this commit). Remaining addendum items: identity ladder n=24 (needs GPU
— folded into the S4 grant request) + §5 final state (needs the exact §5 reference from the
coordinator — asked).

### 2026-09-05 ~14:1xZ — A1: S2-serve root cause chain + TS-suite defect (device bring-up, grant 1)

Device bring-up executed on the coordinator grant (device 0 + shared TP2):
- **In-process harness: PASS both modes** — no-MTP 48/48 MATCH, mtp_k=2 48/48 MATCH
  (batched == sequential bit-exact through the Phase S accept path, greedy configs).
- **serve_batched_ci.sh: P0/B0-B3 GREEN** (greedy regression clean under real per-lane
  configs; B3 negative detected). **S1 coexistence PASS ×3** — greedy lane byte-identical
  with a sampled partner, incl. across lane flips.
- **S2-serve: RED → root-caused in two layers.** (1) translate.cpp:47-52 assigns a RANDOM
  per-request seed when unpinned (observed seed=15213654020233300322) — fixed by pinning
  --seed on the S-cell servers (118ffd9c). (2) WITH the seed pinned, the sampled lane is
  still order-dependent: diag v2 (s2_order_diag.sh, controlled lane order via
  NINFER_BATCH_WINDOW_MS=500 + 300ms stagger): **A1==A2 byte-identical (same order,
  different processes — the sampled chain IS reproducible), A1!=B1 (order flip changes the
  sampled lane only; greedy byte-stable 0.70/23/72 in all runs)**. => Cross-lane
  ORDER-dependence: the second-prefilled lane's state perturbs that lane's distribution.
  Greedy masks it (argmax); sampling exposes it (draw shift). Prefill-vs-verify localization
  OPEN: the NINFER_MB_SEQ_STEP probe was VOID (seq_step_done only gates the plain step,
  never the MTP round — tp2_backend.cpp:2869/3689). Next: dump round-1 verify logits per
  order and diff, or gate SEQ_STEP into the MTP round as a real bisect arm.
- **TS-suite (gemini, cherry-picked 1c51790f..db41d106): TS1 RED on device 0** —
  `allreduce_argmax non-monotonic remap MISMATCH: got tok=0, want 99999` + sticky
  cudaErrorIllegalAddress killing TS2 (rc=134 ×2). Root-caused to the TEST: dev_draft_vocab
  is a device-0 pointer passed to the rank-1 kernel (no peer access) — production passes
  per-rank copies (make_rank, tp2_backend.cpp:761). Routed to gemini with fix sketch.
  Suite gate stays RED until gemini's fix lands; runner work is independently proven above.
- **Wedge escalation added** to stop_server (SIGKILL after 30s grace, own-PID only) after a
  server ignored SIGTERM and held 13.5 GiB/card (poisoned the first TS run; that run VOID).

### 2026-09-05 ~14:2xZ — A1: TS-suite GREEN post-gemini-fix; S2 localization plan

- gemini's TS1 fix (7fedd1c6: per-rank dev_draft_vocab) landed on the branch; suite re-run on
  device 0: **PASS rc=0** — TS1-TS5 green incl. TS5 negative controls (lookup-draft rejected,
  heterogeneous mtp_k rejected — gap 3 verified on-device) and the workspace invariant
  (max_lanes=8 → 0.62 MiB). Evidence: results/phase_s/ts_suite_fixed_*.log (committed).
- Remaining RED: S2-serve reproducibility = lane-ORDER dependence (proven, see prior entry).
  OPEN LOCALIZATION: prefill-state leak vs batched-verify numerics. SEQ_STEP probe VOID (does
  not gate the MTP round). Next session: NINFER_MB_LICDBG-style round-1 verify-logit dump per
  lane order + diff (deterministic per order, differs across orders ⇒ prefill-side), or a real
  MTP-round SEQ_STEP arm. Then fix (prefill state isolation) or rule (attribute + gate policy).
- Gate policy note for the coordinator: S1 (coexistence) + all greedy gates are GREEN and the
  sampled serve path is FUNCTIONAL (both S runs dispatched, served, reproduced same-order);
  the order-dependence is a quality-of-reproducibility defect, not a corruption.

### 2026-09-05 ~14:4xZ — A1: PHASE A (G1) COMPLETE — batched decode is tier-family-wide

Phase S merged to main (cbf613fa, coordinator gate-policy ruling (b); S2 tracked open).
Phase A landed on wo/kvarn-multibatch (037d47f2 + evidence decc1c8e):
- Gates widened: tp_engine can_batch + batched-runner fallback now is_kvarn_storage
  (tier family); i8/bf16 keep the docs/138 M4 per-lane fallback until Phase C.
- Launcher: the (k5v4||k4v4)&&multi throw replaced by tier dispatch.
  <MultiBatch=true, Masked=true, KSide=5/4> compiled CLEAN on the FIRST build —
  the slice6 template's MultiBatch arms were already tier-generic (the WO's
  "latent compile findings" prediction did not materialize).
- HARNESS (first-ever batched slice6 decode): k5v4 ±MTP, k4v4 ±MTP — 4/4 PASS,
  48/48 tokens MATCH sequential per lane per mode (results/phase_a/).
- SERVE PATH: k5v4 CI 0 failed; k4v4 CI 0 failed (P0/B0-B3 + S1 + S2 green; B1
  dispatch + per-lane stats present at both new tiers; B3 mutation detected).
  S2-serve PASSED at both new tiers with the pinned seed (the k4v2 order-
  dependence did not manifest here — it remains the tracked open gate at k4v2).
- Operational note: a stale-binary serve run was caught and voided (ninfer-serve
  must be rebuilt after ANY source change, not just merges); one orphan server
  SIGKILLed (own PID); stop_server escalation now handles this class.

REMAINING: Phase B (G2 non-MTP serve fix) → C (G3 BF16/I8 kernel + workspace
gate) → D (G4 int4 prefill fill) + the tracked-open S2 localization.

### 2026-09-05 ~15:1xZ — A1: PHASE B (G2) COMPLETE — non-MTP batching un-gated, serve-verified

The docs/143 valid-columns crash was fixed by the 2a runner rewrite; only the self_mtp
admission gate remained. Reproduction with a probe switch (NINFER_BATCH_ALLOW_NOMTP,
now retired): the concurrent no-spec pair DISPATCHED as a 2-lane batch and was
BYTE-EXACT vs the sequential reference — no valid-columns throw. The gate is removed
(can_batch = kvarn && lanes>1 && !NINFER_BATCH_DISABLE); the docs/143 comment replaced
with the Phase B record + the rule-14 mutation note (re-adding a self_mtp-style gate
must make non-MTP fall back single-seq, NOT crash — gemini's test scope).
Final verification (default behavior, no env): dispatched mtp=off k=0 lanes=2; both
lanes parity OK; both replies finish=stop. MTP-path code untouched by this commit
(can_batch expression only adds non-MTP admission; eligible() uniformity keeps
MTP/non-MTP in separate batches; mtp_batch_ok ring arithmetic is MTP-only).
Tool: tools/smoke/diag/phase_b_repro.sh (kept as the no-spec concurrency diag).

### 2026-09-05 ~15:3xZ — A1: PHASE C SCOPING (recon complete; implementation = next session)

Workspace-gate half (gqa_attention.cpp:470) — findings before touching anything:
- The dtype marker for ALL kvarn tiers is DType::KVARN_K4V2 (docs/117: shared marker; the
  kvarn_k/v_bits are the only tier signal). The gate's whitelist is
  `(dtype != BF16 && dtype != I8)` — the collapse adds the KVARN marker.
- The KVarN batched launch (gqa_attention_kvarn_cached_batched_launch) allocates
  acc {kGqaHeadDim, q_heads, tokens*batch, splits} BF16 + m/l FP32 — tier-INDEPENDENT
  (dequant is in-kernel; smem is static per-KSide).
- BUT the unified launcher's OTHER allocations are ROUTE-dependent:
  packed_verify / k5v4_small_t_verify / materialize (k_temp/v_temp/bt at :310-314) /
  T=1 partial (pm/pl at :406-408), with n_tiles = packed_pages + (tail_count>0) —
  RUNTIME tail state, not derivable from (q_heads, widths, batch) alone.
  => The §5 risk note is real: an exact capacity needs the route-exact model; a
  conservative one needs paged-view bounds (pages_per_lane-derived n_tiles ceiling).
  DECISION: the gate collapse lands WITH the kernel half (as the WO originally paired
  it), as one substitutive change — not standalone.
Kernel half (G3 core): gqa_attention_cached_batched (:611, KVarN-only throw) extends to
BF16/I8 — MultiBatch=true/Masked=true instantiations for the BF16/I8 decode kernel family
(gqa_attention_decode.cu + decode_i8.cuh), direct reads replacing the KVarN dequant stage.
Same first-instantiation campaign shape as Phase A. The throw removal there is the
rule-15 substitutive step (grep the KVarN-only form empty after).
PHASE C TEST SCOPE (for gemini, rule-14/15):
- TC1 batched-vs-plain bit-exact at BF16 and I8 (harness --kv-dtype, both ±MTP).
- TC2 lane isolation at I8/BF16 (mixed partners, B shrink to 1 mid-run).
- TC3 registry/planning enumeration: every tier {bf16, i8, kvarn_k4v2, k5v4, k4v4}
  dispatches its batched route; non-kvarn non-MTP still falls back per Phase C's final
  fallback policy (rule 13 site enumeration).
- TC4 workspace capacity: KVarN profile accepted; bf16/i8 capacities UNCHANGED
  (byte-identical formula paths); kvarn bound >= launcher peak on a real round.
- Mutations: flip the new dtype dispatch arm (TC1 fails); shrink the kvarn capacity
  bound below launcher peak (TC4 fails); re-add the KVarN-only throw (TC1 fails).
