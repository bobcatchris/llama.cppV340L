# (vi) Step-0 reconciliation + per-step pin — OFFLINE results (2026-09-06, session 01a07628)

Read this with `HANDOFF_agent1_vi_debrief.md` §6. This doc records what the offline
reconciliation (debrief step 0) and the per-step analysis actually found, on the two
post-debrief captures that survived in /tmp. **No device time was used.**

## Evidence captures (preserved from /tmp, both 2026-09-06)

| dir | binary state | contents |
|---|---|---|
| `results/phase_c/hashpt_051417_vlhash/` | HashTap (landed-but-uncommitted at the time) + restored run_layers hook | server.log with 160 VLHASH blocks (2 TP ranks × 2 emitters, identical hashes), hashpt_A/B.log with B0–B5 points |
| `results/phase_c/hashpt_053551_steps/` | c576d788 (B3b/B3c per-step) | hashpt_A/B.log with B3b_step/B3c_stepout; values match the debrief's anchors exactly (greedy d13aa079 vs 5b5e8664 etc.) |

## 1. Tooling bugs fixed (results/phase_c/vlhash_layer_analysis.py)

1. `--steps` mode was **dead code**: it was appended after the `__main__` guard AND
   `main()` consumed argv first (crashed `FileNotFoundError: '--steps'`). Now dispatched
   inside `main()`; `steps_main` defined before the guard.
2. The steps analysis intersected `(round, sl, lane)` keys across runs — **empty by
   construction** under the order flip (A keys carry lane=0 where B carries lane=1 for the
   same request). Fixed: lane is SELECTED per role (sampled = A.l1 vs B.l0; greedy =
   A.l0 vs B.l1), key = (round, sl). This was debrief rule §4.2 applied inside the tool.
3. New tool `results/phase_c/vlhash_block_compare.py` — block-aligned VLHASH comparison
   (splits on `layer=-1 col=0`, dedups rank/emitter dupes). Used it to prove the old
   "no round-2 divergence" was a parser artifact; the raw VLHASH blocks are too
   fragmented (1-col fragments interleaved between emitters) for naive parsing — the
   hashpt B-point trail is the reliable instrument; VLHASH per-layer is only usable with
   a redesign (single emitter, rank-0 only).

## 2. Findings (role-paired, lane-assigned storage coords normalized)

Per-point-type first diverging round (greedy: A.l0 vs B.l1; sampled: A.l1 vs B.l0):

| point | round-1 | reading |
|---|---|---|
| B0_seedhid | MATCH | prefill order-invariant (re-confirmed) |
| B2_verify (verify logits) | **MATCH** | batched verify compute is order-invariant here |
| B4_accept | **MATCH** | round-1 accept result matches in THIS capture |
| B5_state | **MATCH** | post-accept KV/state writes match |
| B2b_alignout (aligned hidden) | DIFF | first REAL divergence — request-keyed, all-distinct hashes |
| B3_draft (v_arh/v_prop) | DIFF | follows alignout |
| B3c_stepout (r1 draft-step outputs) | DIFF at sl=0, hid only (draft IDs match) | sub-argmax, with B3b step-state MATCHING (pos/rope/valid/win all equal) |
| B2c_verifyhid / B2b_alignin | DIFF but lane-keyed | **PROBE ARTIFACTS** — see §3 |
| B3b_step / B4_accept / B5_state at round 2 | DIFF | downstream consequences of the accept-count divergence |

## 3. PROBE ARTIFACTS: B2c_verifyhid and B2b_alignin are lane-keyed, not request-keyed

The round-1 B2c hash SET for A.lane1 equals the set for B.lane1 (rotated by 2), and the
same for lane 0 — while B2_verify (logits, same underlying compute) is perfectly
request-keyed and matching. Conclusion: these two probe points read a **lane-determined
scratch buffer** whose content does not depend on the request under the flip (likely the
align-forward's shared workspace / write-in-place buffer at capture time). They do NOT
indicate hidden divergence; do not use them as evidence. (B2b_alignout is a genuine
per-request output — 8 distinct hashes, request-keyed.)

## 4. THE PIN (sharpened from the debrief)

- Round-1 verify logits, accept result, and state writes: order-invariant in this capture.
- The first genuine order-dependence is in the **round-1 post-accept chain**:
  `B2b_alignout → B3_draft (v_arh/v_prop) → mtp_forward_decode_batch ×3`.
- B3c: r1 draft-step-0 output hid diverges while the host step-state (pos/rope/valid/win)
  AND the draft IDs match → the divergence is sub-argmax INSIDE the draft forward or in
  how its input (the aligned/accepted hidden) is assembled — **not** in the accept
  arithmetic and **not** in the step-state assembly.
- Per the debrief's interpretation key: `mtp_forward_decode_batch` (the batched [1,B]
  draft forward, first-execution MultiBatch code, docs/154 §6 class) is the
  order-sensitive op — or the align/accept-hidden writeback that feeds it (B2b_alignout
  is the first order-dependent REQUEST-KEYED hash).

Note vs the debrief's root-cause claim: the debrief said the r1→2 ACCEPT-COUNT divergence
is the first visible divergence; in the 05:35 capture B4_accept MATCHES at round 1 and
first diverges at round 2 — consistent with a sub-argmax draft-chain divergence that only
becomes accept-visible after the draft chain has consumed the differing hidden (the
debrief's own "(sub-argmax at round 1, accept-visible at the r1→2 boundary)" reading).

## 5. Session 2 follow-up (same day, ~11:0xZ): probe-bug PROVEN, prepare cleared, suspect isolated

Offline code read continuation (no device):

1. **The B2c/B2b_alignin mirage mechanism is PROVEN, not just suspected.** The probes read
   column `(t*B + j)*5120`; the `[5120,T,B]` layout (ne[0]=5120 innermost) puts column (t,j)
   at `(T*j + t)*5120`. For B=2/T=4 a correct write + transposed probe reproduces EXACTLY the
   observed lane-keyed rotate-by-2 pattern across the order flip (derived and checked against
   the round-1 hashes). Converse: had the verify hidden write-back been transposed, the probe
   would have looked role-paired-CORRECT — it did not, so **the verify hidden v_vhid is
   correct**. B2b_alignout's hashes used the same transposed indexing → its role-paired DIFF
   was mirage-contaminated and is NOT usable as the "first divergence" pin by itself.
2. **What survives as genuine:** B3_draft's v_arh (correct stride-B indexing on [5120,B])
   differs at round 1 → the align forward's OUTPUT genuinely diverges, and v_arh is selected
   FROM v_aligid, so the divergence is in the align forward or one of its inputs.
3. **All verifiable align-forward inputs are cleared**: v_vhid (§5.1), alignment ids
   (mtp_prepare_next_round read: per-row pure, `(row*T + j)` indexing consistent with [T,B],
   no cross-row reads), positions/vcalign/envelope (symmetric max — order-invariant), KV rows
   (lane-keyed, internally consistent), host positions (`host_pos[j*T+t] = F_h[j]+t`, matched).
4. **Round 1 is the FIRST execution of the align path** — round 0 is prefill-only (no
   B3_draft/B3b/B4/B5 points exist at round 0), which is why the divergence starts exactly at
   round 1 regardless of where in the align path the defect sits.
5. **The kvarn-clean / i8-bf16-dirty asymmetry localizes the suspect surface:** the kvarn
   branch (per-lane append `gqa_kv_append_kvarn_batched` + cached batched attend) is
   byte-identical under the flip per the debrief; the i8/bf16 branch is the **fused-append
   `ops::gqa_attention` on `batch_mtp_kv_`** — the attend side was audited per-slot pure
   (debrief §3.5) but the APPEND side into the batched MTP cache was never audited. That
   fused-append arm is the prime suspect.

### Instrument changes landed (build verified: ninfer-serve links clean)

- Fixed the transposed column indexing in **B2c_verifyhid, B2b_alignin, B2b_alignout**
  (`(t*B+j)` → `(T*j+t)`; each carries a comment pointing here).
- New **B1b_alignids** point right after `mtp_prepare_next_round`: per-lane hashes of
  alignment ids + ar_positions + ar_valid_columns (row-pitched stride read from `nb[1]`,
  matching the launcher's `ar_step_stride`). This closes the blind spot between B4/B5 and
  the align forward.

### Decision tree for the next device window (~30 min, role-paired diff at round 1)

| first diverging point | verdict |
|---|---|
| B1b_alignids | mtp_prepare_next_round input defect (accept/anchor/frontier path) |
| B2c_verifyhid (fixed) | verify hidden write-back defect (logits-matched ⇒ unexpected) |
| B2b_alignin (fixed) | something rewrites v_vhid between verify and align |
| B2b_alignout (fixed), with B1b/B2c/B2b_alignin all MATCH | the fused-append batched MTP attention on `batch_mtp_kv_` (i8/bf16 arm) — matches the kvarn-clean asymmetry |

## 6. Next

1. OFFLINE code read: `mtp_forward_decode_batch` + the align/accept-hidden writeback
   (B2b_alignout producer) → name the exact order-sensitive line. Suspects: anything
   reading lane-POSITION-dependent scratch (the B2c/B2b_alignin lane-keying shows such
   buffers exist in exactly this region), in-place reuse of the verify hidden buffer for
   the draft chain, or a batch-index vs slot-index mixup in the draft forward's inputs.
2. Fix design → device window (~30 min incl. build) → S2 diag exit 0 on bf16+i8 +
   accept-count probe → gemini formal test → S2 re-gate.

## 7. WINDOW 9 (07:04Z, cards claimed + released per protocol) — THE TREE RESOLVED

Capture: `results/phase_c/hashpt_070449_window9/` (probe log `hashpt_order_probe_win9.log`).
Role-paired first diverging round (fixed probes):

| point | first_diff | reading |
|---|---|---|
| B0_seedhid | — | prefill clean |
| B1b_alignids arpos/arvalid | role-CLEAN r1 | prepare's position outputs correct (ids sub-hash is whole-tensor — trivially order-sensitive; per-lane ids remain unverified by hash, but see the kvarn control below) |
| B2_verify / B2c_verifyhid | r2 | **verify hidden + logits CLEAN at r1** (now MEASURED with fixed indexing) |
| B2b_alignin | r2 | **v_vhid at align time CLEAN at r1** |
| B2b_alignout | **r1** | **v_aligid (align output) DIFFERS at r1** |
| B3_draft / B3c_stepout | r1 | follows alignout |
| B3b/B4/B5 | r2 | downstream |

**Conclusion: the align forward diverges with clean, measured inputs.**

**The kvarn control (debrief §1: kvarn_k4v2 two-order test is byte-identical) clears ALL
shared code**: the runner append/rewind/prepare/uploads, `mtp_forward_stem`, the
`mtp_forward_tail` TP arm up to the attend branch, and everything after it
(sigmoid_mul/o_proj/post_mixer/final-norm/draft-head) are consumed by kvarn too — a defect
in any of them would diverge kvarn as well. The ONLY structural difference between the
clean-kvarn and dirty-i8/bf16 align forward is the attend branch:

- kvarn: `kvarn_attend_mtp_batched` (per-lane append + `gqa_attention_cached_batched`)
- i8/bf16: **`ops::gqa_attention` fused-append, MultiBatch+Masked, on `batch_mtp_kv_`**

**THE NAMED OP: the fused-append batched attention on the MTP cache (i8/bf16 arm).**
The same kernel template (`gqa_attention_small_t_tc_partial_bf16_kernel` / i8 twins) was
cleared for the TEXT-cache verify instantiation (B2_verify matches r1) — so the residual
audit surface is the MTP-cache-specific instantiation: (a) MTP cache geometry/table_rows
semantics in `batch_mtp_kv_->batch_layer_view(0)` vs `v_kvr`, (b) the append target state
(freshly-rewound slots), (c) `gqa_cache_index`/`gqa_kv_new_index` under the MTP geometry.
The bf16 kernel's append/read boundary was re-audited this session and is internally
consistent (write set == `from_new` read set == [first_pos, first_pos+valid_tokens)).

Next: audit (a)-(c) offline → fix design → device verify → S2 re-gate.

## 8. WINDOWS 11-18 (07:5x-09:1xZ): THE DEFECT IS THE MTP PREFILL'S CACHE CONTENT — LANE-ASYMMETRIC

Instrument ladder built and run (each level committed):
- TH (NINFER_MB_TAILHASH, text_context_impl.h mtp_forward_tail TP arm): stem-ah / rope qn,kn /
  attend-a / post-o_proj / post-mlp per-column hashes. Result: **TH0/TH1 role-MATCH at round 1;
  TH2 (attend output) role-DIFF all 4 columns** — the divergence is inside the align forward's
  attention, with clean q/k inputs.
- Seeding control: the [1,2] batched draft steps (fused-append, same MTP cache, TT=1) are
  role-CLEAN — but their envelope reads no unwritten history; not exculpatory for history reads.
- PARTHASH levels 2→2d (gqa_attention.cpp SmallT arm): per-slot attempts hit two layout bugs of
  mine (ne[0]-innermost convention; a double-counted h that OOB'd — window 11 segfault, fixed);
  final level 2d uses layout-proof CONTIGUOUS per-(lane,split) slices: **all 8 splits of both
  lanes differ in acc/m/l** at the round-1 align, with clean q/k — so the attend's CACHE input
  must differ.
- MTPC (tp2_backend.cpp at B1b): chunked whole-plane hashes of the MTP cache layer-0 K/V(+scale)
  planes + block tables. **Block tables identical; K/V planes differ ONLY at chunk 0 (lane0's
  first pages) and chunks 781-782 (the lane0/lane1 boundary)** — 99.8% identical.
- MTPP (per-lane first-page K/V hashes keyed by PHYSICAL page id): **the same request's page-0
  content differs by lane** — greedy A.lane0 b12180d3... vs greedy B.lane1 1ab267e7...;
  sampled A.lane1 8dddb808... vs sampled B.lane0 5b976680.... Stability patterns (r1 vs r2)
  are fully consistent with plen-mod-64 geometry, so the diffs are not transients.

**CONCLUSION (measured): the MTP-layer prefill writes lane-dependent K/V content for the same
prompt.** The align forward then consumes that divergent history (deterministically — alignout
hashes are identical across the 05:35/07:04/07:59/08:11 runs for the same lane/request), the
draft chain inherits it (B3c sub-argmax), and text diverges. Everything else measured today
(verify hidden/logits/accept/state at r1) is order-invariant.

Remaining question for the fix: WHY the per-lane MTP prefill content differs. Candidates:
(a) warmup residue interacting with lane 0's row (warmup ran single-seq on row 0; lane 1's row
    never saw it — but page 0 is fully overwritten by a 64-token prompt... unless the prefill
    append leaves holes, cf. docs/136 §14 class),
(b) a lane-asymmetric input to the MTP prefill forward (shared staging read before write),
(c) the append targeting (page ids verified identical; content differs).
NEXT DEVICE STEP: byte-level dump of lane0-page0-K (run A) vs lane1-page0-K (run B) and
element-wise diff — the pattern (shift/block/scales) names the mechanism directly. ~10 min.

## 9. WINDOWS 19-25 (09:3x-10:5xZ): ROOT CAUSE NAMED — MTP PREFILL APPEND ROW-TARGETING

Instrument: MTPKV (kn/v pre-quant chunk hashes in mtp_prefill_chunk) + MTPC/MTPP (cache
plane/page dumps at B1b) + RAW byte dumps of each lane's first K page.

**Byte-level matrix (round-1 B1b, both runs, K page 0 of each lane's row):**

| page | content found | expected |
|---|---|---|
| runA lane0 (greedy prefill) | SAMPLED's content (0.03% vs sampled kn) | greedy's |
| runA lane1 (sampled prefill) | sampled's ✓ (0.03%) | sampled's |
| runB lane0 (sampled prefill) | GREEDY's content (0.02% vs greedy kn) | sampled's |
| runB lane1 (greedy prefill) | greedy's ✓ (0.02%) | greedy's |

**In both runs BOTH rows hold the SECOND-prefilled request's prefill K/V; the first-prefilled
request's history is gone before round 1.** runB's pages match the CPU reference quantizer
(FP16(absmax/127) per (head,token,64-dim group), rn codes) to 0.02-0.03% (emulation edge cases
only) — the quantizer is correct; kn/v pre-quant chunk hashes are order-invariant per request.

**DEFECT STATEMENT: the per-lane MTP prefill append does not land in the per-lane row — the
first-prefilled lane's MTP history is destroyed by the second lane's prefill append.** The
align forward then reads row r expecting request r's history and gets the other request's →
order-dependent align output → everything measured downstream (B3c sub-argmax, accept drift,
text divergence). kvarn is immune because its MTP arm uses per-lane TILES (kvarn_lane_ws_[lane]),
not the shared paged MTP pool rows.

Suspects for the WHY (audit next, ~1 window):
1. `set_mtp_view(b==0 ? st.mtp_view : st.mtp_lane_views[b-1])` + `publish_mapping` at the
   per-lane prefill (tp2_backend.cpp ~:2717-2726) — verify lane 1's view/publish actually
   binds pool row 1 (vs both lane views aliasing row 0).
2. `mtp_forward_batch`'s append inside mtp_prefill_chunk — `mtp_kv_.layer_view(0)` uses the
   CURRENT view; check no stale ScopedValue/binding pins row 0 (e.g. io_.backend_kv_table_row
   or a captured table pointer from lane 0).
3. The pool free-list: lane1's first page = 1563 — confirm block_tables row 1 really maps
   positions 0..63 to page 1563 (dump both rows' first 4 entries as VALUES, not hashes).
NEXT WINDOW (~10 min): per-lane POST-PREFILL page dumps (dump at prefill-loop end per b, before
the next lane starts) + block-table row values → distinguishes "lane1 wrote both rows" from
"lane0's write was never landed". Then the fix is a targeting one-liner + regression.

## 10. WINDOW 26-27 (11:1xZ): the smoking gun confirmed + pool addressing fully decoded

MTPROW transition probe (per-lane prefill dumps + block-table VALUES) + the pool source read
decode the MTP pool addressing completely:
- paged_kv plane: 3126 physical pages × 32 KB (64 tokens × 2 local heads × 256 dims, I8);
  block_tables is [page_group_count=1564, table_rows=2] — **group-major, transposed vs the
  kernel's [lane][slot] reading**: bt[group][lane] = the physical page for (lane, group).
  lane0's pages = EVEN ids {0,2,4,...}; lane1's = ODD {1,3,5,...} (lane1's first page = 1563
  would be WRONG under this layout — 1563 is odd... note MTPP printed lane1 front page=1563:
  under group-major that is row 781's lane1 page; the allocation page_ids ordering needs one
  more check, see below).
- **MTPP stride bug (mine)**: I computed stride = plane_bytes/page_group_count = 65536 (a
  GROUP = 2 pages) but used it as a per-PAGE stride with a PAGE id — so the lane1 dump
  pointed 2× too far. The lane0 dump (page id 0 → offset 0) was CORRECT.
- Per-lane prefill kn/v chunk hashes: ORDER-INVARIANT per request (greedy {2103,02a9} in both
  runs; sampled {a6e5,dfef} in both runs) — confirmed twice now.
- CPU reference quantizer matches the cache to 0.02-0.03% (emulation rounding only) — for the
  pages that hold a given request's content.

**THE MEASURED DEFECT (unchanged, now with correct addressing for lane 0):** row 0's first page
(page id 0 — lane0's own first page, offset 0, no stride ambiguity) holds the **SECOND-prefilled
request's** prefill K/V in both runs (runA: sampled 0.03%; runB: greedy 0.02%). The
first-prefilled request's history is not in its row.

**Interpretation:** the second lane's MTP prefill append lands in row 0 (or equivalently, both
lanes' appends land in the same row) — the per-lane `set_mtp_view(...)` switch does not take
effect for the prefill append, OR the lane views alias the same row. Candidate code paths to
audit (next session, offline first):
1. tp2_backend.cpp:2717-2726 `set_mtp_view(b==0 ? st.mtp_view : st.mtp_lane_views[b-1])` —
   verify st.mtp_lane_views is non-empty and each is execution_view of a DIFFERENT allocation
   row (they are constructed at :527-536 — re-verify binding rows 1..).
2. mtp_prefill_chunk's append `ops::gqa_kv_append(kn, v, positions, mtp_kv_.layer_view(0), s)`
   (text_context_impl.h:1322) — confirm mtp_kv_ is the CURRENT lane's view at append time and
   that layer_view(0).block_table is the lane's own table (not pool.block_tables with a
   default row).
3. The MTPROW probe itself needs one fix before rerun: stride must be PER-PAGE (32768), not
   per-GROUP, and the dump must key pages by ALLOCATION page_ids (authoritative), not by my
   transposed block-table walk. mtprow_lane*.bin from win27 are INVALID for lane1.
4. The transition question "does lane1 write both rows or does lane0 never land" is answered
   by dumping row0-page0 + row1-page0 (allocation-keyed) after lane0's prefill and after
   lane1's prefill: 4 hashes per run pin the writer.

Note the win27 mtprow_lane*.bin files are INVALID (transposed walk + group-stride) — do not
use them. The win25 mtpK_lane0_* (page 0, offset 0) remain valid.

## 11. WINDOW 28-29 (11:4x-11:5xZ): TRANSITION TABLE + APPEND TARGET LOG — mechanism 90% named

**The 4-sample transition table (correct addressing, allocation-keyed):**

| sample (runA) | row0-page0 | row1-page0 |
|---|---|---|
| after lane0's (greedy) prefill | **greedy ✓** (0.02%) | neither (93-97%, unwritten) |
| after lane1's (sampled) prefill | **sampled 0..60 + stale greedy 61..63** (4.63% vs sampled — exactly 3 tokens' worth of stale) | **sampled ✓** (0.03%) |

Run B mirrors exactly (row0 = greedy's content after lane1, row1 = greedy ✓).

**INTERPRETATION (mechanism 90% named):**
- lane0's prefill append lands CORRECTLY in row0.
- lane1's prefill append lands in BOTH row0 (slots 0..60, leaving lane0's 61..63 stale — the
  4.63% residue matches plen-61 vs plen-64 geometry exactly) AND row1 correctly. I.e. lane1's
  append runs TWICE (or once with a table covering both rows) — one writer uses ROW 0's table,
  the other the correct row.
- Since lane1 is ALWAYS the second lane (prefill is sequential b=0,1), row 0 always ends up
  holding lane1's content → lane0's align reads lane1's history → order-dependence. kvarn
  immune (tiles). Single-lane unaffected (row0 = own content).

**APPENDLOG evidence:** all appends (warmup T=53, lane0 T=64, lane1 T=61) on a given rank log
the SAME block-table pointer and bt[0]=0 (row0's table flat start). For lane1's call this means
one writer used row0's table. The second writer (row1) is NOT the logged site — there must be a
second append call site for the MTP prefill that does not route through the logged one, OR the
logged site runs twice with different bindings (e.g. a chunked variant vs direct call).

**Remaining open item (one small window or a careful read):** locate the second MTP-append call
site that fires for lane1 with the correct row-1 table. Candidates: mtp_forward_batch's
dual paths (:1482 with logits_column=-1 → the tp2 prefill call; vs another entry), a re-run of
the prefill worker for lane1, or an append inside mtp_forward_batch after mtp_forward_core.

**FIX DIRECTION (independent of the exact line):** the per-lane MTP prefill append must target
lane b's OWN allocation row explicitly — pass the lane's block-table row (or the lane's view)
through mtp_forward_batch → the append, instead of relying on mutable TextContext view state.
Equivalently: bind the row ONCE per lane and thread it through. Then: re-run the 4-sample
transition (expect row0 stays lane0's, row1 stays lane1's) → S2 diag exit 0 on i8+bf16 →
accept-count probe → S2 re-gate with gemini's formal test.
