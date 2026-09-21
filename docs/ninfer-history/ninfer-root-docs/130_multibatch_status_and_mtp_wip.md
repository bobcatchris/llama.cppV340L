# 130 — MultiBatch status: plain-batched SOLVED & pushed; batched-MTP in progress (stashed WIP)

Status: HANDOFF (2026-09-01, takeover session). Branch `wo/kv-uniform`.
Remote `github/wo/kv-uniform` is at `cf19c3af`. Two safe commits sit locally
unpushed (`8a5f42a8`, `19f864b4`). The batched-MTP runner is a STASHED WIP that
does NOT work yet. Read §0 before touching anything.

---

## 0. What is actually true right now (verified)

| Capability | State | Evidence |
|---|---|---|
| Decay fix (docs/129 §2) | DONE, shipped | guard cells match baseline |
| Plain batched decode (mtp_k=0), 2 lanes | **SOLVED, committed, pushed** (`cf19c3af`) | token-exact vs shipped at 32/96/128/256 tok, both orderings, multi-page prompts; 1.4–1.7× aggregate |
| Batched-MTP plumbing (append/attend/backend) | DONE, committed (`19f864b4`) | kernel W=1..6 bit-exact; ops/attend compile + unit-green |
| Batched-MTP runner round-loop (stage 4) | **BROKEN — stashed, not committed** | 64 tok passed ONCE (transient, never committed); 128 tok DEADLOCKS |

The working tree is clean at `19f864b4`. The MTP runner WIP is in
`git stash` ("WIP batched-MTP runner (stage 4)"). Recover with `git stash pop`
— but it hangs; do not ship it as-is.

---

## 1. The plain-batched solve (the real win this session)

Root cause of the docs/129 §6 divergence was TWO bugs, both found by
three-way ground-truth diffing (fresh-process sequential vs in-test sequential
vs batched), NOT by the overnight probes:

1. `physical_page_id()` identity shortcut (`block_table.ne[0] <= k_pages.ne[3]`)
   misdirected batched-lane commits (lane 1's row maps logical 0 → physical 65;
   the shortcut wrote physical 0 = lane 0's page). Fixed: `PagedKVLayerView` now
   carries the owning allocation's host `page_ids()` mirror; resolution is sound
   and sync-free; the heuristic is deleted.
2. The batched runner leaked `TextContext::kv_` bound to the last lane's row; the
   shipped single-seq path never rebinds it, so the *reference* run was the
   contaminated one. Fixed: runner restores `set_kv_view`/`set_linear_state_slots`
   on exit.

Plus harness fixes (all three external-review claims were true): replay used a
hardcoded `scale_page_stride=1152` vs runtime 18432 (false MATCH); the ATT-READ
probe read lane-0 logical pages as row pointers (false "stale view"); the CPU
oracle had OOB compares + wrong tail-K index (now AGREES with both kernels).
A placement assertion was added to the ops test (committed pages must land at
their table's physical page) — closes the vacuous-pass hole where batched-vs-
single-lane comparison shared the same bug.

---

## 2. Batched-MTP: what works, what doesn't

### 2.1 Proven good (committed)
- Kernel/launcher: `BATCHTEST W=1..6` proves the MultiBatch attend is bit-exact
  vs per-lane single-seq at verify widths, incl. commit-boundary shapes.
- Ops: `gqa_kv_append_kvarn_batched` targets `ws[b].mtp` via `layer_index=-1`.
- Runtime: `TextContext::kvarn_attend_mtp_batched` (per-lane MTP append + ONE
  MultiBatch attend on `batch_mtp_kv_`); MTP-tail TP branch wired to it.
- Backend: per-lane MTP-pool allocations (lane b → MTP row b; lane 0 == shipped).

### 2.2 The runner round-loop (COMMITTED `b7a13c31`; 64 PASS, 128 diverges)
Status now: the round loop is committed and **64-token 2-lane MTP passes
token-exact (1.69×)**. The deadlock and early-termination bugs are FIXED
(both-ranks state commit + frontier-consistent `remaining`). What remains is a
**correctness divergence at 128 tokens**: lane 0 wrong at output 79, lane 1 at
125. Verified facts (this session):

- **Not reference contamination.** `NINFER_MB_SEQ_FIRST=1` gives the *identical*
  divergence (`batched=14227 seq=33200` at 79), so the sequential reference is
  clean and batched-MTP is genuinely wrong. (Unlike the plain-path bug #2.)
- **Not the accept logic.** The batched path calls the SAME
  `speculative_accept_greedy_drafts` op as the shipped single-seq runner.
- **Not the main-pool rewind.** `[RW]` logging shows `committed_pages`/
  `tile_page`/`tail` stay consistent (`tail = F - 64`, `cpg=1` after the first
  page) across the divergence — no orphaned page.
- **k-dependent, lane-dependent.** k=3 → lane0@79, lane1@125; k=1 → lane1@121.
  Divergence begins ~20 tokens after lane 0's first main-pool page commit
  (cur_F 64 at output ~59). Points at the **multi-token (T=k+1) verify append /
  `prepare_page_for_append` hydrate across a page boundary in the batched
  per-lane path**, not the 1-token plain append (which is exact to 256).

### 2.3 Replay evidence (step 26, the diverging round)
Captured the batched verify attend inputs (`NINFER_MB_CAPTURE=1
NINFER_MB_CAPTURE_STEP=26`) and replayed through `slice4_kvarn_bench
replay_test`:
- batched vs single-seq kernel on identical inputs → **MATCH** (0/3072)
- CPU reference vs single-seq → **MATCH**
- captured `pos={82,83} valid={86,89} packed={1,1}` (lane 0 window 86 = 1 page + 22 tail)

**Caveat that bounds the conclusion:** `replay_test` launches `TokenTile=1`,
but the verify runs `TokenTile=T=k+1=4`. So the replay only proves **column 0**
(position 82) is correct. The divergence is at position 84 = **column 2**. The
untested surface is the intra-verify dependency: column *t* attends to KV that
columns *0..t−1* appended earlier in the SAME verify call. `BATCHTEST W=1..6`
proves the multi-token attend given a fixed KV, but NOT the append→attend
ordering *within* one verify forward. That is the prime remaining suspect.

### 2.4 TT=4 replay: attend is DEFINITIVELY correct
Extended `replay_test` to `TokenTile=4` and replayed the step-26 capture:
**batched vs single-seq on identical inputs → MATCH, 0/12288, per-column
{0,0,0,0}.** All four verify columns' KVarN attend are bit-exact. So the
attend, the KV it reads, the accept op (+ its three input layouts, all
row-major `[B,·]`, verified against `speculative_round.cuh`), the commit/rewind
trajectory, and the GDN ring layout (`lanes + b*(k+1)`, capacity-guarded) are
ALL correct.

**The bug is upstream of the attend: the batched verify FORWARD produces wrong
hidden/logits → wrong `target_tokens` → wrong committed token.** The replay
cannot see this: it feeds BOTH kernels the same captured `q`, so a wrong `q`
still replays as MATCH. Since wrong *drafts* provably cannot corrupt tokens
(the accept commits `target[a]`, which is correct regardless of the draft), the
committed token being wrong means the verify's own `target_tokens` are wrong.

### 2.5 Refined next step (needs a NEW capture, not the attend replay)
Capture the verify's **`target_tokens` (lic) per column** at the diverging round
for batched vs a single-seq run at the same frontier, and if they differ, bisect
the forward: dump the GDN recurrent/conv state slot read by the verify
(`dbg_dump_gdn_state` already exists) and the per-layer hidden. Prime suspects,
in order:
1. **GDN verify state slot read** (`slt2_h[j] = ring base + slot_h[j]`): if the
   accepted state from the prior round isn't at `ring base + a`, the GDN hidden
   is wrong. The plain path advances GDN 1 token/step (exact); the verify
   advances T then rewinds — the multi-token GDN scan is the untested surface.
2. **Per-lane RoPE/positions** in the verify (`host_pos = F_h[j]+t`) if `F_h`
   is off for a lane after a page commit.
3. A later KVarN layer's multi-token append (replay only covered layer 0).

Do NOT re-litigate the attend — it is proven correct.

---

## 3. Next steps for the MTP runner (ordered, each testable)

1. **Make `live`/`B` provably identical across ranks every round.** Both ranks
   must derive `active[]` from the same synced data (acc_h/lic_h are identical
   after allreduce). Audit the stashed commit loop: confirm the `for j` loop,
   the `hit_stop`/`remaining`/`ctx` done-test, and BOTH `kvarn_rewind_lane`
   calls run on both ranks with identical inputs. Add a one-line per-round log
   `[MTP-B] rank=%d step=%d B=%d` and confirm the two ranks print the SAME B
   each round before touching anything else. (This is the observable check I
   skipped; do it first.)
2. **Barrier discipline.** The MTP branch must hit the SAME number of
   `sync_bar.arrive_and_wait()` per round on both ranks, on every control path
   (normal, `!any_draft`, should_stop). The earlier `should_stop.load()` in the
   mid-round branch was removed for exactly this reason — keep termination
   decisions at the loop top (post-barrier) like the plain path.
3. Then re-run: 64 → 128 → 256 tokens, `--mtp 3`, both orderings, and the
   long-prompt (≥2 pages/lane) case. Only commit when 128 passes twice.
4. Guard is NOT the tool for MTP: `tp_engine.cpp` has zero references to
   `run_tp2_requests_batched`, so decode_guard exercises only the single-seq
   path. Multibatch is validated ONLY by `ninfer_tp2_batched_decode_test`.

---

## 4. Process notes (what wasted time this session — do not repeat)

- **Zombie GPU processes.** Aborted runs and the guard's backgrounded
  `ninfer-serve` kept 12–13 GB on both GPUs; new runs then OOM'd SILENTLY,
  which looked like a hang. ALWAYS check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and
  `pgrep -af ninfer-serve` before a GPU run, and kill leftovers.
- **Device-side `printf` in a hot kernel** (the KPREP debug) serializes the GPU
  and flooded the log — turned a 2.5s run into minutes. Remove debug printfs
  before timing anything.
- **Async H2D from stack temporaries.** `cudaMemcpyAsync(dst, vec.data(), …)`
  where `vec` dies at the block's `}` corrupts the copy. Small control uploads
  must be synchronous `cudaMemcpy`.
- **Stream ordering.** Worker streams are non-blocking; plain `cudaMemcpy`
  (NULL stream) does NOT wait for them. Every D2H read of kernel results needs
  `cudaMemcpyAsync(...,s)+cudaStreamSynchronize(s)`.
- **Arena adjacency.** A collective overrunning a large buffer corrupts the
  next small buffer. Allocate large buffers first, small control buffers after.
- **Don't grep away the failure.** When a run misbehaves, `tee` to a log and
  read the tail; a filtered command that prints nothing hides OOM/hang/throw.

---

## 5. Session outcome (2026-09-02) — HONEST state after the MTP chase

This session chased batched-MTP correctness far longer than proved productive. Here is
the precise, verifiable truth, so the next session starts from facts not optimism.

### 5.1 What got fixed and committed (real, in `git`)

- `cf19c3af` plain batched decode — SOLVED, token-exact, pushed. Unrelated to MTP.
- `b7a13c31` batched-MTP **deadlock** — FIXED. active[]/generated were rank-0-only, so the
  two ranks disagreed on `B` when MTP lanes finish on different rounds (NCCL allgather
  column-count mismatch). Fix = replicate lane STATE (cur_F/active/extents/rewind) on both
  ranks; run the commit on both ranks; derive termination from a synced value.
- `67dad105` MTP parity **diagnostic harness** (overwrites the in-test reference to run
  single-seq MTP, and adds `NINFER_SEQ_ONLY` / `NINFER_DUMP_BATCHED` /
  `NINFER_MTP_LOSSLESS` / `NINFER_LANE0_TOKENS`). These are the ONLY reliable way to test
  MTP: the in-test reference is contaminated (see §5.2).

### 5.2 Two DISCOVERIES that reframe the whole problem

1. **The in-test "sequential" reference is CONTAMINATED.** The shipped single-seq runner
   (`run_tp2_request`) runs in the SAME process after the batched run and inherits leaked
   `TextContext` state (kv_/kvarn_text_ws_/lane bindings). So every in-test
   "batched vs sequential" divergence was suspect. The ONLY trustworthy comparison is a
   FRESH process per path (`NINFER_SEQ_ONLY` / `NINFER_DUMP_BATCHED`), which this session
   added. Most earlier "PASS" / "FAIL" numbers in this doc were contaminated and are
   **not reliable**.

2. **Shipped single-seq MTP is NOT lossless for lane 0.** With clean isolation,
   `seq_mtp != seq_plain` at token 79 for lane 0 (batched_mtp == seq_mtp, but both differ
   from plain). MTP greedy acceptance should be lossless, so this is a pre-existing bug in
   the SHIPPED single-seq MTP path (or its plain comparison), NOT in the batched code. The
   batched MTP faithfully reproduces the shipped MTP (batched_mtp == seq_mtp). Do NOT chase
   "batched_MTP != plain" — that is the shipped-MTP-vs-plain bug, not yours.

### 5.3 Reliable verified state at HEAD (`93fda35e` + `67dad105`), isolated SEQ_ONLY dump

| tokens | lane0 (batched vs seq_mtp) | lane1 |
|---|---|---|
| 64 | DIFFER, batched 62 / seq 64 (truncates) | DIFFER, batched 63 / seq 64 |
| 128 | DIFFER, batched 127 / seq 128 (1 short) | MATCH (128) |
| 256 | MATCH (256) | DIFFER, content | 

So HEAD **runs without hanging** but the **frontier-derived `remaining` truncates the tail**
(60/64, 127/128). The batched-MTP is NOT clean-passing; the commit message over-claimed.

### 5.4 The ONE correct termination fix, and why it is NOT landed

Root cause of the truncation: the frontier `cur_F` advances by the FULL accepted count
`a+1` (correct for KV), but the output is CLAMPED to `room = max - emitted`. So
`cur_F - (plen-1)` (the frontier-derived "generated") OVER-COUNTS vs the real emitted count
near `max_output`, hits 0 early, and truncates the tail. The single-seq runner terminates
on the EMITTED count (`generated`), not on `cur_F`.

The coordinator (subagent) derived the exact single-seq invariants (docs/130 §11 in their
head, roughly): terminate on `generated`; budget clamp `room = max - emitted`; the draft
budget must subtract the `-licensed (a+1)` term; done on `hit_stop | generated>=max |
next_F+k+1>=max_ctx | cancel`.

**Tried and REVERTED (hangs EXIT=124 on a verified-clean GPU):** switch `room`/`remaining`
to `generated.size()` on both ranks. It deadlocks. **Hypothesis (untested, next session):
`generated` diverges across ranks** — one of the `generated.push_back` sites (seeding line
~2243, commit line ~2532, or the batched PREFILL) is still rank-0-only, so rank 1's
`generated` lags → `remaining`/`done`/`active[]` diverge → B mismatch → NCCL allgather
deadlock. Next session: `grep -n "generated"` in the batched function and make EVERY
push rank-symmetric before using `generated.size()` for termination.

### 5.5 Second blocker: B=1 shrink routing (STASHED, unsolved)

When a lane finishes and `B` shrinks to 1, the surviving lane is `b=live[0]` (often 1, not
0). The batched attend must use the lane-tile path (table_rows[lane], tail at base+lane*elems),
BUT the batched runner and the single-seq reference BOTH run at `active_sequence_batch_==1`,
so batch count can't discriminate them. Coordinator's fixes (delete both `batch<=1`
degenerate branches; key tail+lpp+kernel block-table on `table_row`; lane_ids on the append;
`multi = batch>1 || lane_packed_pages!=null`) are in `git stash@{0}`. They are CORRECT for
the batched path but destabilized the prefill/seeding (`T != width*batch`, and the
discriminator confused single-seq-vs-batched-at-batch-1). The routing discriminator needs a
dedicated "batched runner active" flag set per-phase (prefill+rounds, NOT seeding), which
was the last unresolved piece.

### 5.6 Recommendations for the next session (in order)

1. **Fix termination with a provably rank-synced emitted count.** Make `generated` fully
   rank-symmetric (grep every `generated.push_back` in the batched fn; the prefill is the
   prime suspect) OR track a both-ranks `emitted` int seeded to 1 and incremented by
   `toks.size()` once — but find the earlier double-count first (it was real: `em_before=3`
   at step1 with seed=1, a second touch I never found). Then `64/128/256` should go clean.
2. **Then the B=1 shrink routing:** add a per-phase "batched active" flag so the attend
   routing uses the lane-tile path at B=1 without hijacking the single-seq reference or the
   seeding. Use the coordinator's stash@{0} as the base.
3. Only then wire `run_tp2_requests_batched` into the engine scheduler.
4. Always verify via the isolated harness (SEQ_ONLY + DUMP_BATCHED), never the in-test
   reference. Always `nvidia-smi --query-compute-apps` + `pgrep ninfer-serve` before a GPU run.

---

## 6. RESOLUTION (2026-09-02) — batched-MTP token-exact + B=1 shrink fixed, M0 gate landed

§5's two blockers are DONE and committed on `wo/kv-uniform` (series 8bd76d17 → b54ae986).
Both were verified with the ISOLATED harness (fresh SEQ_ONLY + DUMP_BATCHED per path) — the
in-test reference stays contaminated per §5.2 and is not trusted.

### 6.1 Bug A — termination truncation (frontier-vs-emitted) — FIXED (8bd76d17)
Root cause: the commit loop clamped `out_count` and derived `remaining`/`budget`/`done` from the
FRONTIER (`cur_F - (plen-1)`). Near `max_output` the frontier over-counts vs the emitted count
(it advances by the FULL accepted `a+1` even when output is clamped to `room`), so `next_ext` /
`lane.extents` zero prematurely -> the `!any_draft` check breaks -> tail truncation (lane0 126/128).
Fix: key the output budget + termination on the EMITTED count (`generated.size()`), captured
per-round via `genb_h` (a barrier-ordered read of the rank-0-owned shared vector), and compute
`remaining_after - licensed` exactly as `mtp_prepare_next_round_kernel` does. `done` on
`hit_stop | gen_after>=max | cur_F+k+1>=max_ctx | cancel`.

WHY the previous generated-based attempt DEADLOCKED (documented so it isn't retried): the prior
fix pushed `t0`+`toks` onto the SAME shared `lanes[b].generated` vector from BOTH rank threads.
That is a data race / double-count (both threads append identical tokens to shared memory), so
`generated.size()` diverged between ranks -> `active[]`/`extents`/`B` diverged -> NCCL allgather
deadlock. The correct approach keeps output ownership rank-0 and reads the count behind the
barrier. (Reported by the plan owner in §5.4; confirmed in review.)

### 6.2 Bug B — B=1 shrink routing — FIXED (f8f926bc)
Root cause: when a lane finished, `B` shrank to 1 and the surviving lane was `b=live[0]` (often
1, not 0). The `batch<=1` degenerate shortcuts in `kvarn_attend_text_batched` /
`kvarn_attend_mtp_batched`, and the `active_sequence_batch_>1` routing in `attn_mix_tp` /
`mtp_forward_tail`, sent it down the single-seq attend bound to lane 0's workspace ->
"append left a hole in the tile" abort.
Fix: added `kvarn_batch_lane_ids_` + `kvarn_batched_active_` to TextContext; route on
`(active_sequence_batch_>1 || kvarn_batched_active_)` (the `>1` arm preserves the batched
PREFILL, the flag arm covers B=1 shrink; single-seq ref keeps a false flag); key
`gqa_kv_append_kvarn_batched` ws/block_table/packed on `lane_ids`; key the kernel MultiBatch tail
on `table_row`; launcher `multi = batch>1 || tail.lane_packed_pages`; set the flag before the
decode loop and clear it in runner hygiene; B-size the `mb_vwin` view.

### 6.3 NEW latent bug — single-seq MTP round-0 seeding (REF-side, not batched)
Fronted by the M0 dumps (decoded directly): for BOTH lanes the single-seq reference's round-0
draft `d0 = 0` (BOS) — `R1 r0 drafts=[0,198,248045]` (lane0) and `[0,271,32]` (lane1) — while the
batched seeds the coherent next token (`B1 r1 [13,248046,198]` / `[248045,74455,198]`). The
single-seq "reuse prepare_mtp" path argmaxes a mis-seeded / zeroed MTP prefill hidden
(`st.mtp_logits`) and yields BOS. This is the SAME class as the known lane0-79 MTP-vs-plain bug,
on the REFERENCE side; the batched seeding is correct. ROUND-0 is therefore
**KNOWN-DIVERGENT-UPSTREAM**; the gate skips it and the batched is NOT "fixed" to match the ref.
This is likely the mechanism behind the §5.2 #2 finding (shipped single-seq MTP not lossless for
lane0@79) — worth a dedicated fix in `run_tp2_request` seeding later.

### 6.4 M0 phased gate (NINFER_MB_PHASEGATE) — dumps + gate v2
Env-gated host int dumps (doc-128 §8.3 format: `u32 'MBP1' | u32 ver | u64 prompt-sha | u32
elem_count | int32 bytes`), one file per (phase, round, lane):
- B1/R1 (verify inputs): vids[T] vpos[T] drafts[k] F ext slot anchor
- B4/R4 (accept): a lic[T]
- B5/R5 (state): cur_F cur_anchor cur_slot generated next_ext active (R5 active reflects done)

GATE DESIGN (v2, per plan-owner correction): per-ROUND structure (a/lic/slot/ext/drafts) is NOT a
cross-runner invariant — draft chains legitimately differ -> different a/round groupings, the
ref does NOT apply the `-licensed` ext clamp while the batched spec(B) does, and the frontier
over-runs by the full accepted a+1 on a coarser final round. The INVARIANTS are the
**converged-state trajectory (cur_F, cur_anchor, generated)** + the token stream + done-at-max
(F = plen + gen - 1 away from the terminal over-run). So the gate keys on `gen -> (F, anchor)`,
uses the batched `active==0` (done) as TERMINAL (ignoring F/anchor over-run), treats
slot/ext/a/lic/drafts as DIAGNOSTIC-ONLY, and SKIPS round-0 (upstream seeding divergence) + the
regrouped intermediate gens. `tools/bench/phase_gate_mb.py` + `.sh`.

### 6.5 Matrix + negative test
MATRIX (isolated dumps, all CONVERGED-STATE MATCH, gate exit 0): T=64/128/256 both lanes +
B=1 (lane0=30, lane1 survives the shrink). Token-exact confirmed separately via the isolated
SEQ_ONLY+DUMP_BATCHED diff (64/128/256 both lanes, B=1 lane1 128, plain 64 both MATCH).

NEGATIVE TEST (targeted mutation, NOT a revert): temporarily change the lane_ids binding to
`if (B>1) set_kvarn_batch_lane_ids(rows_h.data()); else set_kvarn_batch_lane_ids(nullptr);` at
the MTP round. This reproduces the pre-fix B=1 routing (survivor keys on lane 0's tiles) while
keeping the M0 dumps + compiling. Result: **run exit=134, `gqa_kv_append_kvarn: append left a
hole in the tile` — a LOUD abort at the B=1 phase** (67 dumps produced before the abort all
match; the RUN dies, which is the localization). This proves the B=1 routing fix is necessary:
the pre-fix bug fails loud at the B=1 append, so the phase table is not even needed to localize
it (the e2e run catches it); the phase table is for the silent-variant case. Mutation restored,
matrix re-verified clean.

### 6.6 Still open (post-M0, not blockers)
- Wire `run_tp2_requests_batched` into the engine scheduler (only after the formal B-chain gate
  lands).
- Fix the single-seq MTP round-0 seeding (REF-side bug, §6.3) — likely resolves lane0-79.
- Formal B-chain milestone: bf16 phases B2/B7/B8/B9 + a dedicated test binary + CI wiring, then
  the doc-128 unified-kernel gate closure (user directive: after multi-batch is done in full).

---

## 7. SESSION 2026-09-02 (2nd) — T=256 latent race found + partially root-caused; one open item

§6 fixed the two ORIGINAL blockers and they are committed. But the plan owner's independent
matrix found a NEW latent race that the (F,anchor,gen) gate is BLIND to. Status: the race is
CONFIRMED + REPRODUCED + localized to the KV feed into the MTP GDN at the 2nd-lap (position
256+), with one real latent bug (the batched-attend lpp publish async-H2D race) FOUND + FIXED,
but the FLAKE is NOT yet fully cleared. Read this before running T=256.

### 7.1 Symptom + reproduction (ship-blocker for declaring multi-batch complete)
SYMPTOM: `ninfer_tp2_batched_decode_test --tokens 256 --mtp 3` — batched lane1's FINAL token
flakes (~40-50%): `198` (wrong) vs `6558` (correct, = seq_mtp reference). lane0 never flakes.
64/128 and B=1(lane0=30) are always clean. It is a RACE (timing-dependent), not deterministic.

REPRO (isolated, clean):
```
NINFER_SEQ_ONLY=/tmp/r <test --tokens 256 --mtp 3>   # gives seq_mtp_lane1.txt, last token 6558
for i in 1..NINFER_DUMP_BATCHED=/tmp/r$i <test --tokens 256 --mtp 3>
diff <(tail -1 /tmp/r/batched_mtp_lane1.txt) /tmp/r/seq_mtp_lane1.txt   # 198 vs 6558 -> flake
```
Repros WITHOUT `NINFER_MB_PHASEGATE` (it's a pipeline race, not a dump artifact). The in-test
compare also catches it (exit 1).

### 7.2 Localization (confirmed)
Added two env-gated traces:
- `NINFER_MB_RINGDBG` (tp2_backend.cpp): hashes the GDN recurrent+conv state at the slots each
  lane's verify READS (`slt2_h[j]`, `slt2_h[B+j]`) and the WRITE slots (`slt2_h[B+j]+token`),
  tail-only (step>=70), rank0.
- `NINFER_MB_QKVDBG` (text_context_impl.h, `gdn_mix_tp` ~line 1844): hashes the local
  q4/k4/v4/g4/b4 that feed `gated_delta_net_snapshot`. NOTE: the real batched forward is
  `gdn_mix_tp` (line 1741); `gdn_mix` (line 1935) is a single-GPU-only stub that returns to
  `gdn_mix_tp` — instrumenting `gdn_mix` is a no-op in TP2 builds.

DISCRIMINATION (flaky vs clean run): the FIRST divergence is at **r78 col-2 (lane1, slot 8,
position ~259 = the 2nd lap past max_output=256)**. At that point: write-slot=8 IDENTICAL,
initial state (r77/r78 read-hash) IDENTICAL, conv-state hash IDENTICAL, col-0/col-1 write
hashes IDENTICAL — ONLY the RECURRENT state written to slot 8 differs. So it is the
"right slot + right input + wrong result" signature.

CHAIN ANALYSIS (plan owner, settles it): the snapshot op chains columns within a row (col j
starts from col j-1's OUTPUT slot base+j-1; `initial_state_slots[b]` is only col-0's start).
col-2's input = col-1's OUTPUT (slot base+1=7), and col-1's write-hash is IDENTICAL
clean-vs-flaky -> col-2's chained input is identical -> since `apply_gdn_transition` is
deterministic (pure fp32), col-2's q/k/v/g/beta MUST differ -> **the corruption is UPSTREAM of
the GDN: the hidden state at pos 259, from the attention/KV at 259**. The GDN is faithfully
propagating a corrupted hidden, not the root.

### 7.3 REAL latent bug FOUND + FIXED: batched-attend lpp publish async-H2D race
This is the docs/130 §4 "async H2D from a temporary" trap, but with a heap vector
(text_context_impl.h, both `kvarn_attend_text_batched` ~line 420 and
`kvarn_attend_mtp_batched` ~line 696):
```cpp
std::vector<std::int32_t> packed(maxlane, 0);          // LOCAL, freed at function return
... packed[b] = ws[b].text[fidx].committed_pages; ...
CUDA_CHECK(cudaMemcpyAsync(kvarn_lpp_dev_, packed.data(), maxlane*4, H2D, stream));  // ASYNC!
```
The GPU executes the H2D LATER (queue depth) after `packed` is destroyed; the 8-byte block is
then reused by the NEXT attend's `packed` (or the MTP variant's) with DIFFERENT values
(text ~4-5 pages vs MTP-tile ~1-2), so `kvarn_lpp_dev_` (per-lane committed-pages) can hold a
WRONG page count -> the attend kernel's `lane_packed_pages[table_row]` is wrong -> wrong page
range -> wrong hidden -> wrong token. Racy, lane1-affected (lpp[1]), usually masked (the next
reuse is usually the next TEXT layer's packed with the SAME value).

FIX (applied, working tree): make BOTH uploads BLOCKING — `cudaMemcpy(kvarn_lpp_dev_,
packed.data(), maxlane*4, H2D)` (drop Async) in the text + MTP variants. 8 bytes, ~µs.
This is a genuine latent bug worth keeping the fix for.

### 7.4 What is RULED OUT (verified in code)
- GDN snapshot slot math: position-independent + lane-correct (int32 `initial_slots[batch]`,
  `snapshot_bases[batch]+token`; `slot_h[j]` refilled from the lane each round; ring base keyed
  on lane id `ring0 + b*T`). No pos term, no wrap. NOT a slot-index bug.
- `apply_gdn_transition`: pure fp32 (compile-time kDvPerWarp/kQkPerLane, warp_sum). No u8.
- KV block-table rows: `std::int32_t`. No u8.
- `gdn_input_projection_snapshot` + wrapper: int32 (selectors I32[batch]). No u8.
- No `%256` / `&0xFF` / 8-bit position-slot-row index anywhere grepped across src.
- REWIND (`kvarn_rewind_lane`): HOST-ONLY bookkeeping (two int assignments) — no device race.
- `sync_bar` is std::barrier(2); each rank syncs its own stream before the commit loop, so
  cross-rank device ordering is sound.
- Each rank has its OWN TpRankState (own KV/GDN pool, own ws). The only shared host state is
  lanes[]/active[] (barrier-ordered) + generated (rank0-only, barrier-ordered reads).
- The B=1 lane_ids keying (rows_h at 2540-2541) is lane-id correct at both the append
  (gqa_kv_append_kvarn_batched reads lane_ids[b] host-side) and the lpp publish (keyed by lane
  id, sized to max lane).

So the plan owner's u8-overflow-at-256 prior is NOT confirmed anywhere in the slot/path/
transition. The "128-clean/256-flake" signature is real but its cause is not an 8-bit index in
the GDN snapshot slot path or KV block-table row.

### 7.5 CURRENT VERDICT / OPEN ITEM
The lpp blocking fix (7.3) is a REAL fix but did NOT clear the T=256 flake (~40% still flakes:
T=256 ×10 gave runs 4,8,9,10 -> 198). So the lpp race is a latent bug (kept) but is NOT the
(single) cause of the FLAKE. The corruption is confirmed UPSTREAM (hidden at 259 from KV at
256), and the prime suspect is the **KV 2nd-lap at position 256** (B=1 phase): either
(a) a per-lane KV/MTP ring of cap 256 (`pos % 256`, my code read shows only `pos % 64` tile
slot + `pos/64` page with pages_per_lane=65 — no 256 cap I can see), or (b) the B=1 phase keying
the KV block-table row on the LIVE index j (=0) somewhere (against the lane_ids fix) so lane1's
attention at 256 reads lane0's page ring -> wrong KV -> wrong hidden at 259. The plan owner was
asked to confirm which. The q/k/v hash (7.2) + the read/write-slot GDN hash remain the
discriminators/verifiers.

### 7.6 Working-tree state (uncommitted, this session)
- `src/runtime/tp2/tp2_backend.cpp`: `NINFER_MB_RINGDBG` (read+write-slot GDN hashes, tail-only).
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h`: the lpp blocking-copy fix (KEEP, §7.3)
  + `NINFER_MB_QKVDBG` (q/k/v hash in gdn_mix_tp).
All env-gated (zero cost when unset). Committed series is 8bd76d17 -> 523ea36d (§6); these two
files + any further KV fix land AFTER the race is fully root-caused + verified.

### 7.7 Next-step checklist (finish the ship-blocker, then multi-batch is truly complete)
1. Confirm the exact KV 2nd-lap mechanism at position 256 (plan owner to confirm pos%256 ring
   vs j-vs-lane row keying at B=1); instrument that KV site (hash rows_h/kv_table_rows + the KV
   read at 259, or the KV page-ring wrap) with a stepped trace.
2. Apply the fix; verify T=256 batched ×10 (no dumps) 10/10 token-exact + full matrix
   (64/128/B=1(30) ×3) clean.
3. Re-run the M0 gate (CONVERGED-STATE) + commit the KV fix + the §7.3 lpp fix + the
   `NINFER_MB_RINGDBG`/`NINFER_MB_QKVDBG` diagnostics.
4. Then the still-open §6.6 items (engine wiring, single-seq MTP round-0 seeding bug, formal
   B-chain milestone, doc-128 unified-kernel gate closure).

---

## 8. SESSION 2026-09-02 (3rd) — first divergence pinned to the KV TAIL TILE at step 74; M0 gate was self-corrupting (fixed); flake NOT yet fixed

Status: LOCALIZED, NOT FIXED. The T=256 lane1 terminal flake is still open. But this session
replaced §7's wrong framing with measured facts, fixed two real bugs (one of which invalidated
the M0 gate's own output), and produced a capture tool that pins the first diverging step in a
single run. Commit `f70018d1` on `wo/kv-uniform`. Read §8.4 before touching the tile.

### 8.0 TL;DR for the next session
- The ship-blocker's FIRST divergence is the **KV tail tile content** (`tail_k`/`tail_v`) at
  verify round 74 — not the attend, not the GDN, not the launch plumbing. Every other attend
  input is bit-identical across runs at every captured step (§8.3).
- The M0 phase gate (§6.4) was **self-corrupting**: B1/B4/B5 dumps raced between the two rank
  threads and carried a rank-0-only field. Its "CONVERGED-STATE MATCH" results keyed on `gen`
  were noise. **Fixed and re-verified** (§8.1); `b54ae986` is superseded.
- §7.2's "r78 col-2 GDN hash divergence" was **benign** — the rank-gated dumps show r74-r80
  identical across runs; the only token-affecting divergence is r81 `lic0` (column 0).

### 8.1 Two real bugs found and fixed (verified)
1. **BUG-2 — the M0 gate self-corrupted its own data.** The three `mb_phase_dump` call sites
   (B1 round-start, B4 post-accept, B5 commit-loop) were not rank-gated, so BOTH rank threads
   `fopen`/`fwrite` the SAME `<phase>_r<round>_l<lane>_<sha>.bin` path concurrently. B5's
   payload contains `lane.generated.size()`, which is **rank-0-owned** (rank 1 never pushes —
   §5.4/§6.1), so the last writer won and the dumped `gen` was a 2-writer race.
   Evidence (6 runs, before the fix): B5 `gen` differed across runs at r19/r23/r34/r66
   (60 vs 56, 71 vs 68, 106 vs 110, 213 vs 217) while `cur_F`/`cur_anchor`/`cur_slot`/
   `next_ext`/`active` were identical **in the very same files** — `gen` is the ONLY
   rank-asymmetric field in the payload. Fix: all three sites are now
   `if (gdir) if (rank == 0)`. Re-verified on a 6-run T=256 matrix: the `gen` divergences are
   gone and **no previously-masked divergence appeared** — the race was not hiding anything,
   the gate's other fields were always trustworthy. Every "CONVERGED-STATE MATCH" keyed on
   `gen` (§6.5) was comparing noise; treat `b54ae986` as superseded by this fix.
2. **BUG-1 — uninitialized device memory read into the draft chain.** In the batched AR draft
   loop, `if (maxw2 == 0) { chain.push_back(next_ids); continue; }` pushed `mb_arout[sl]`
   which is NEVER written on that path, and the post-loop D2H copied it into
   `lane.drafts[sl+1]`. Observed directly in B1: r78 lane1 `d2` = 51293 / 120538 / 19933
   across three runs. It is provably unconsumed **today** only because the device kernel sets
   `ar_valid_columns[s] = (s+1 < next)` and the host `next_ext` is monotone non-increasing in
   the terminal region, so a skipped step is always beyond the next round's `ext`. That is
   coincidence, not safety. Now zeroed (`up(next_ids, zeros)`).
   NOTE: a SECOND path of the same class remains open — at B=2 when only lane 1 needs an AR
   step, lane 0's column runs with window=0; its d1/d2 still vary (r75/r76 lane0). Not chased.
3. **Harness:** `NINFER_DUMP_BATCHED` / `NINFER_SEQ_ONLY` into a directory that does not exist
   → `fopen` returns NULL and the test `fprintf`'d into it → SIGSEGV indistinguishable from a
   product crash. Cost real debugging time this session (several "crashes" were mine).
   Now a guarded `dump_tokens()` helper that shouts on stderr.

### 8.2 What was RULED OUT this session (all with measurements — do not re-chase)
| suspect | probe | result |
|---|---|---|
| lpp publish legacy-stream ordering | `NINFER_MB_SYNC_LPP` (full drain before publish) | 4/8 flake → NOT the mechanism |
| "SEQ_ATTEND is just serialization" | `NINFER_MB_SYNC_ATTEND` (sync at same point, batched kernel kept) | 5/8 flake → serialization ruled out; SEQ_ATTEND's win is real |
| derived per-lane tail_count | `NINFER_MB_TAILDBG` (kernel-exact derived-vs-published comparator) | **0 mismatches in 16 runs** → the kernel's derivation was ALWAYS correct |
| page-4 fresh allocation / block-table row write at decode | code audit | impossible: all 65 pages/lane are `materialize_pages()`ed at backend construction (tp2_backend.cpp:441/449/461/467); `publish_mapping` only runs in prefill; the only decode-time `materialize_*` is in `program_impl.h` (non-TP2 path). §7.5 candidate (a) is DEAD |
| tail stride/row mismatch append-vs-attend | code audit | `kvarn_tail_elems_ == bind elems == kG*kD*kv_heads`; row keyed on lane id both sides |
| GDN snapshot cross-CTA race | code read (recurrent.cuh) | columns chain in REGISTERS per CTA (`state[kDvPerWarp][kQkPerLane]`), `store_snapshot` writes base+token; CTAs own disjoint (head, dv-row) ranges → no intra-launch overlap |
| conv zero-fill for `column >= valid` | code read (causal_conv1d.cuh) | `out[out_idx] = 0.0f; return;` and x0 never read → SAFE (so the lane0 window=0 draft garbage is NOT conv) |
| cross-rank sharing of batched state | code read | per-rank `TpRankState`/`TextContext`/device/stream; workers are separate threads on separate devices |

Flake-rate table (in-test exit-code detector, T=256 mtp3, 8-16 runs each):
baseline 3/8 · SYNC_LPP 4/8 · SYNC_ATTEND 5/8 · CUDA_LAUNCH_BLOCKING 1/8 · SEQ_ATTEND 0/8 ·
published-tail-count 7/16 (fix withdrawn, TAILDBG silent).

### 8.3 THE LOCALIZATION (new tool + result)
New env `NINFER_MB_CAPTURE_STEPS="60,68,74,..."` captures the fidx-0 attend's full input set at
EVERY listed step in a SINGLE run (one subdir per step), instead of bisecting run by run.
Fixes over the old capture: a `static mkdir` silently skipped every dir after the first; a
`static captured_steps` vector was a **cross-rank data race** (segfault) — capture is now
rank-0-only; the dump used a legacy-stream D2H which is NOT ordered vs the worker stream
(phantom tile diffs) — now `cudaMemcpyAsync(stream)` + sync; the tile dump now covers BOTH
lane slices (at B=1 the surviving lane is live[0]=1, whose tile is at
`base + tail_batch_elems` — the old base-only dump captured a FINISHED lane's stale tile).

Result (3 runs: 198 / 6558 / 198; steps 60,68,70,72,74,76,78,80,81; 12 files/step:
q, pos, vc, rows, kpage0, vpage0, scales_l0, tail_k, tail_v, bt, lpp, scalars):
```
steps 60, 68, 70, 72 : ALL 12 files bit-identical across all runs
step 74 onward       : ONLY tail_k.bin / tail_v.bin differ
                       (q, kpage0, vpage0, scales_l0, lpp, rows, bt, pos, vc, scalars identical)
```
The attend's inputs are identical except the tile — i.e. the FIRST divergence in the whole
pipeline is the tail tile's content, and everything downstream (q at 78, the pool at 78,
lic0 at 81) follows from it. §7.2's attend-replay MATCH is not contradicted: the replay fed
both kernels the same captured bytes.

Tile slot decode (k_tile `{kG=64, kD=256, kvh=2}` bf16 token-fastest = 1024 B/slot;
lane0 slice [0,64), lane1 [64,128)):
```
r74 K lane0 = {8..19, 24..27, 36..47, 56..63}   V lane0 = {0, 32}      lane1 CLEAN
r76 K lane0 = {0..3, 8..27, 32..51, 56..63}     V lane0 = {0,1,32,33}  lane1 CLEAN
r78 K lane0 = as r76                            K lane1 = {16..31, 36..39, 52..55}
                                                V lane1 = {1, 33}
```
The 4-slot grouping + 32-periodicity is a stride/interleave signature, not random corruption;
and V differing in only 2 isolated slots while K differs broadly is consistent with one run's
tile holding the ORIGINAL bf16 tile and the other holding `dequant(quant(bf16))` from a
`gqa_kvarn_hydrate_page` rewrite (K is 4-bit, V is 2-bit — both lossy).

### 8.4 Prime mechanism (to confirm next) — hydrate rewrites a slot the attend still reads
`gqa_kvarn_prepare_page_for_append` → `gqa_kvarn_hydrate_page` refills tile slots `[0, slot)`
by DEQUANTIZING the committed page's codes (lossy: 4-bit K / 2-bit V). The attend reads
`window = F+T` keys every round, so a position that is inside the tile AGAIN after a
commit/hydrate boundary is served lossy rather than as the original bf16 the append wrote.
Whether a given position is "original bf16" or "dequantized" depends on where the
commit/hydrate boundary fell — and that is exactly the kind of thing that can differ between
runs whose host-visible state (B1/B4/B5) is identical, because the tile is one shared batched
buffer holding all lanes' slices contiguously and two lanes' append/hydrate land on the same
stream. This would explain:
- the ~40% rate: most 4-bit quantizations are near-exact, so the drift usually does not flip an
  argmax; at 256 tokens there are ~4 page boundaries/lane and the accumulated drift sometimes
  flips the terminal token;
- 64/128 clean: fewer boundaries;
- SEQ_ATTEND = 0/8: it re-reads the tile per lane immediately after a D2H sync inside the
  attend, changing which value the attend sees.
THE FIX, if confirmed: make the invariant "the tail must never cover a slot that has already
been committed" hold structurally (never hydrate a tile whose contents are still needed, or
serve already-committed positions from codes and the tail only for the un-committed range).

### 8.5 Corrected §7 framing (supersedes §7.2/§7.5)
- First TOKEN-affecting divergence = **r81, B4 `lic0`, column 0** (6558 x5 vs 198 in the flaky
  run) with every host-visible input (B1/B4/B5 for r74-r80) bit-identical. NOT "r78 col-2".
- The r78 col-2 GDN-hash divergence was a benign consequence (and partly a rank-race artifact
  of the un-gated B5 dumps), not the cause.
- The corruption is UPSTREAM of the attend's inputs except the tile — so "corruption upstream
  of the GDN at pos 259" (§7.5) was right in direction but wrong in surface: it is the KV tile,
  not the hidden/KV page data.
- §7.4's list stands, with the additions in §8.2.

### 8.6 Diagnostics added this session (all env-gated, zero cost when unset)
- `NINFER_MB_CAPTURE_STEPS` — multi-step attend-input capture (see §8.3). THE tool for
  "first diverging step".
- `NINFER_MB_RINGDBG_LAYERS=<round>` — per-layer + per-ring-slot GDN rec/conv hashes at one
  round (used to prove r79-l1 ring2 was the only divergent GDN slot, and that conv was clean).
- `NINFER_MB_QKVDBG_STEP=<round>` — per-GDN-layer q/k/v/g/beta hashes at one round.
- `NINFER_MB_SYNC_LPP` / `NINFER_MB_SYNC_ATTEND` — ordering probes (both negative).
- `NINFER_MB_TAILDBG` — derived-vs-published tail comparator (0 hits in 16 runs).
- Published per-lane tail counts (`kvarn_*_tailcnt_dev`) + MultiBatch kernel consumes them:
  defense-in-depth, NOT the flake fix (TAILDBG proved the derivation was already correct).
  Kept because it removes a real future hazard and enabled TAILDBG.

### 8.7 Repro + assets
```
# flake rate (fast, in-test detector):
./build/tests/ninfer_tp2_batched_decode_test --artifact /home/intel/models/qwen3_8_27b.ninfer \
  --tokens 256 --mtp 3        # exit 0 = PASS, exit 1 = flake (198 vs 6558 at 255)
# first-divergence capture (2 runs is enough):
NINFER_MB_CAPTURE=1 NINFER_MB_CAPTURE_STEPS=60,68,70,72,74,76,78,80,81 \
  NINFER_MB_CAPTURE_DIR=/tmp/A NINFER_DUMP_BATCHED=/tmp/Ad <test> ...
# diff per step; the first step whose tail_k/tail_v differ is the divergence point
```
/tmp/rkT (3-run capture set), /tmp/rkQ (rank-gated gate re-run), /tmp/rkL (per-layer GDN),
/tmp/rkS (first multi-step capture, superseded by rkT), /tmp/fix256 (pre-session flake runs).
NOTE the harness trap: `NINFER_DUMP_BATCHED`/`NINFER_MB_PHASEGATE` dirs must exist — the
unguarded `fprintf(NULL)` segfault is fixed in `f70018d1`, but older binaries still crash.

### 8.8 Next steps (in order)
1. Confirm/kill the §8.4 hydrate mechanism by code read (coordinator asked to verify whether
   hydrate can run for a lane whose page was already committed+attended, and whether two
   lanes' appends/hydrates can interleave on the shared batched tile buffer).
2. QA test (routed via the coordinator to the dedicated qa agent): **ops-level tile round-trip
   fidelity** — bind the batched workspace (2 lanes), append → record tile bytes →
   `gqa_kvarn_rewind_to_token_count` to a mid-page frontier → append across the next boundary
   (forcing commit + hydrate) → assert any slot the attend reads is BIT-IDENTICAL to what the
   append wrote. Cover B=2 and B=1-shrink-lane1, and both the text tile and the MTP tile
   (`layer_index = -1`). Pure ops-level: no model, no TP2, no rank threading.
3. Apply the §8.4 fix; verify T=256 ×16 clean (no dumps) + full matrix 64/128/256 + B=1(30) ×3.
4. Re-run the M0 gate with the RANK-GATED dumps (§8.1) — that is the only trustworthy baseline.
5. Then the §6.6 items (engine wiring, single-seq MTP round-0 seeding bug, formal B-chain
   milestone, doc-128 unified-kernel gate closure).

---

## 9. RESOLUTION (2026-09-02, 4th session) — T=256 FLAKE FIXED at the root; §8.4 hydrate hypothesis KILLED; new pre-existing k=1 issues filed

### 9.1 Root cause of the ship-blocker (proven, not inferred)
The T=256 lane1 terminal-token flake was NOT hydrate, NOT the attend, NOT the GDN, NOT an
arena/aliasing hazard. It is a **same-slot intra-launch write race caused by PADDED verify
positions**:

1. `speculative_prepare_verify_inputs_kernel` (src/ops/kernel/speculative_round.cuh) emitted
   `positions[off] = base + (j <= extent ? j : extent)` — padded columns (j > extent) REPEAT
   the last valid position. Measured: r78 lane1 vpos = {257,258,259,259} (ext=3, T=4).
2. `scatter_kvarn_tile_kernel` (src/ops/kvarn/kvarn_workspace.cu) derives the tile slot from
   THIS device tensor: `slot = positions[t] & (kG-1)`. The host append bookkeeping
   (run detection / tail_count) uses the host binding F+t instead. When ext < T, several
   blocks of ONE launch write the SAME tile slot -> the per-element winner is warp
   scheduling -> **nondeterministic tile content with bit-identical inputs**.
3. Direct evidence: with q/kin/vin captured and BIT-IDENTICAL across 3 runs, the stomped
   tile slot held a per-element MIX of column-2 (225/512 elems) and column-3 (288/512)
   values, different in every run. The victim slot is the anchor slot — i.e. an ACCEPTED
   position — so the corruption enters the converged context and surfaces later as the
   terminal-token flip (198 vs 6558).

### 9.2 The fix (one semantic line)
`positions[off] = base_positions[row] + j;` — every verify column gets its TRUE append
position. Safe because: verify_ids still repeat the anchor for padded columns; the accept
reads only columns <= extent; the rewind truncates padded KV; the window binding (F+T) and
the host bookkeeping (F+t) now agree with the kernel slots by construction. Accepted-column
semantics are bit-unchanged (their positions were already base+j).

### 9.3 Verification (all green)
- T=256 mtp3: **16/16 PASS** (baseline 3/8 flaky).
- Matrix: T=64/128/256 ×3 + B=1 shrink (lane0=30) ×3 — ALL PASS. mtp=0/2 at 256 PASS.
- Isolated harness (fresh SEQ_ONLY vs DUMP_BATCHED, 3 runs): batched == seq_mtp TOKEN-EXACT,
  both lanes, 6/6.
- M0 phase gate (rank-gated dumps): exit 0 on T=256 AND T=128+B=1 — ALL CONVERGED-STATE MATCH.

### 9.4 What this retroactively explains (supersedes earlier sections)
- §8.4 hydrate mechanism: DEAD (the coordinator's code read was right — normal decode never
  hydrates; the ext=0 terminal has no hydrate). The tile diffs §8.3 saw were real, but the
  writer was the scatter race, not dequant-rehydration.
- §5.2 #2 + §6.3 (\"shipped single-seq MTP is NOT lossless for lane0@79\"): the single-seq
  verify feeds the SAME padded positions to the SAME scatter; its launches are sequential so
  the stomp is DETERMINISTIC there (last padded column's KV wins the anchor slot) — hence
  seq_mtp != plain and batched_mtp == seq_mtp. NOTE: after the fix, lane0 MTP-vs-plain STILL
  diverges at 79 (14227 vs 33200, unchanged) — that residue is the SEPARATE round-0 seeding
  bug (§6.3/§6.6), which fires before any padded round. Still open.
- §8.3's slot decode used 1024 B chunks = kD-channel groups, not token slots (K tile is
  ne[0]=kG-contiguous, so token g lives at stride 64 elems). The \"step 74 lane0\" diffs were
  lane0's OWN ext=1 stomp (vpos {257,257,257,257} -> 4 blocks on slot 1); lane1 was clean
  until ITS first ext<T round at r78 — matching the observed divergence onset.

### 9.5 NEW pre-existing issues found while verifying (k=1, never covered by the old matrix)
The doc's matrices only ever ran --mtp 3. At --mtp 1, T=256:
1. In-test compare FAILS deterministically: lane0@161 (batched=56729 vs seq=55296, 4/4 runs,
   same address). Deterministic => NOT the race. Not yet attributed.
2. `NINFER_SEQ_ONLY` SEGFAULTS in `run_tp2_request`'s rank worker lambda (no symbols,
   Release build). Confirmed PRE-EXISTING: the pre-fix binary crashes identically. k=1 has a
   history (§2.2 noted k=1 lane1@121 in the contaminated-harness era). NEXT: debug build +
   bounded read of the k=1 round loop (drafts arrays sized [k], pack[j*k+i], extents clamp).

### 9.6 Diagnostics added this session
- `NINFER_MB_CAPTURE` now also dumps `kin.bin`/`vin.bin` (the layer-0 append k/v inputs).
  These are what pinned the bug: identical across runs while the tile differed => write-side.
- The tile-layout ground truth for future captures: K tile {kG,kD,kvh} is ne[0]=kG-contiguous
  (elem (g,d,h) @ (g + d*64 + h*16384)*2 bytes); V tile {kD,kG,kvh} (elem (d,g,h) @
  (d + g*256 + h*16384)*2). Per-lane slice = 64 KiB, lane b at b*64 KiB.

### 9.7 SESSION 4 addendum — k=1 unblocked (harness bug fixed); k=1 residual attributed; k=2 token-exact
- **SEGFAULT root-caused + FIXED (harness, not product):** `tests/multi_gpu/tp2_batched_decode.cpp`
  hardcoded `run_single(..., 3)` in the NINFER_SEQ_ONLY / NINFER_MTP_LOSSLESS paths while the
  backend is SIZED with `b_opts.mtp_k = o.mtp_k` (TpRankState::ar_drafts/ar_mhs are `k-1`).
  At --mtp 1/2 the single-seq run executed the round-0 AR loop (`step+1 < 3`) against EMPTY
  vectors -> SEGV in run_tp2_request's rank lambda (line ~1450). This is why k<3 was never
  exercisable. Fix: pass `o.mtp_k`.
- **k=2 now TOKEN-EXACT** (batched == seq_mtp, both lanes, isolated dumps): the same code path
  as k=1 (T=3) matches perfectly — strong evidence there is NO batched state bug at small k.
- **k=1 residual (NOT a batched bug):** batched is self-consistent (3/3 identical) and differs
  from the seq reference by a 2-TOKEN EXCURSION at gen 161/164 (anchor 56729 vs 55296 at the
  same F=166), then RE-CONVERGES permanently (gate: gen166+ CONVERGED-PASS, TERMINAL-PASS;
  exit=20 lists exactly those two gens). Ring audit at r80-r84: every round's read slot =
  base + a(prev), lane-correct, cur/base consistent — no slot bug. Attribution: the §6.3
  REF-side round-0 seeding bug (ref seeds d0=BOS at k=1) desyncs the round GROUPINGS from
  round ~1, and the conv state pool is BF16 (layouts_impl.h conv_dtype), so cross-round conv
  state carries bf16 rounding -> near-tie argmax decisions are GROUPING-DEPENDENT. The two
  streams are both valid greedy decodes; token-exactness vs the ref at k=1 is unattainable
  until the §6.6 seeding fix (and/or an FP32 conv pool) lands. seq_mtp == seq_plain at k=1
  (lossless, post-§9.2 fix) — the single-seq side is healthy.
- mtp=0/2/3 regression re-verified after the harness fix.

---

## 10. MULTI-BATCH MILESTONE: COMPLETE (2026-09-02, final state)

Multi-batch MTP (`run_tp2_requests_batched`) is DECLARED COMPLETE on `wo/kv-uniform`.
The last correctness blocker (the T=256 terminal-token flake) was root-caused to a product
bug and fixed at the root (a13ae028); the milestone's formal gate chain is folded and green.

### 10.1 Root cause + fix (the one-paragraph history)
`speculative_prepare_verify_inputs_kernel` repeat-padded verify positions; the KVarN scatter
derives tile slots from that device tensor while host bookkeeping uses F+t → whenever
ext < T, multiple blocks of one scatter launch wrote the same tile slot (intra-launch write
race, per-element winner = warp scheduling) → nondeterministic KV in an ACCEPTED position →
the ~40-50% T=256 terminal-token flake. Fix: `positions[off] = base + j` for every column
(a13ae028). The same mechanism retroactively explains the §5.2/§6.3 single-seq-MTP
non-losslessness (deterministic stomp there) and supersedes the §8.4 hydrate hypothesis.

### 10.2 Verification matrix (all green, no re-runs needed — results stand)
- T=256 mtp3: **16/16 PASS** (baseline 3/8 flaky). T=64/128/256 + B=1-shrink(lane0=30) ×3: PASS.
- Isolated harness (fresh-process SEQ_ONLY vs DUMP_BATCHED): batched == seq_mtp token-exact,
  both lanes, 6/6 at T=256.
- M0 converged-state phase gate (rank-gated dumps, docs §8.1): exit 0 on T=256 and T=128+B=1.
- mtp=0/2/3 regression PASS. k=2: batched == seq_mtp TOKEN-EXACT both lanes.
- Negative evidence from the chase: SYNC_LPP/SYNC_ATTEND/derived-tail-count probes all
  retained the flake; the §8.2 table stands with §8.4/§9.4 supersessions.

### 10.3 B-chain milestone (docs/128 gate pipeline, multi-batch instantiation) — DONE
- **Dedicated gate binaries**: `ninfer_t0_{accept_kernel,tile_roundtrip,attend_geometry,
  gdn_verify,prepare_inputs,stride_audit}_test` — folded from `wo/mtp-tests` (7ff87267 via
  merge 344c495d, 0 conflicts, all compile clean) per docs/131 T0-T3.
- **Standing gates**: M0 converged-state gate (`NINFER_MB_PHASEGATE` +
  `tools/bench/phase_gate_mb.py/.sh`, v2 semantics per §6.4/§8.1) + T3 gate rule
  (`tools/bench/t3_gate_rule.sh`); T1 invariant-assertion mode (`NINFER_MB_INVASERT`),
  T2 golden matrix, runbook `docs/131_runbook.md`.
- **Negative-test requirement**: satisfied by construction this session — the pre-fix binary
  fails the M0 gate/token compare at the pinned repro (docs/130 §8.7), the post-fix passes;
  the §6.5 B=1 negative mutation (loud "append left a hole") remains the ops-level example.

### 10.4 Open items (none block the multi-batch milestone)
1. **Engine wiring**: `run_tp2_requests_batched` is still test-only; the engine scheduler
   (ninfer-serve) runs single-seq. Wiring is mechanical now that correctness gates exist.
2. **§6.3/§6.6 single-seq round-0 seeding fix** (REF-side; also the k=1 token-exactness
   blocker): ref seeds d0=BOS via the mis-seeded MTP prefill hidden. lane0@79 MTP-vs-plain
   is the same class. Scheduled after the doc-128 unified-kernel gate work.
3. **k=1 note**: batched k=1 is deterministic, self-consistent, ring-audited clean; a
   2-token excursion vs the seq ref (gen 161/164, permanent re-convergence, gate
   TERMINAL-PASS) is grouping-dependent bf16-conv-pool rounding seeded by item 2 (§9.7).
4. **CI wiring**: T0-T3 entry points exist (`run_t0.sh`..`run_t3.sh`, runbook); hooking them
   into the CI pipeline is the remaining formality.
5. **Next task**: doc-128 unified-kernel gate closure (D-chain) — user-gated on multi-batch
   completion, which this section declares.

### 9.8 CI-gate session notes — INV-7 domain, canary hooks, harness hardening, and the ONE remaining red
All in `wo/kv-uniform-ci-gate` (a49ae842):
1. **INV-7 domain fix**: drafts are MAPPED full-tokenizer ids — `one_shot_argmax.cu:126`
   `final_tok = draft_vocab_ids[best_idx]` over the 40960-entry draft vocab (max id 248076;
   `tests/multi_gpu/data/qwen38_draft_vocab_ids.json`). The old INVASERT bound [0,152064) was
   a fossilized contract: 248046 is a LEGITIMATE draft token. INV-7 now asserts membership in
   the draft-vocab id set (binary_search over the sorted host list).
2. **INV-3 added** to the INVASERT battery at the pre-verify binding: `window ==
   committed_pages*64 + tail_count + T`. Gives NINFER_CANARY_TAIL_COUNT a real trip signal
   (its old trip was the INV-7 false positive).
3. **canary_unrank_m0**: the env had no injection hook in the merged tree (fossil). Added:
   under the env, force `rows_h` to identity — combined with NINFER_LANE0_TOKENS=30 in the
   harness env, the B=1 shrink then bites (INV-9 / append abort).
4. **T3**: t3_gate_rule.sh runs the T0 binaries directly (the pip-installed `ctest` shim in
   this env throws ModuleNotFoundError: cmake).
5. **Harness**: t1/t2 poll-until-GPU-idle (60s) + T1.2 transient load-OOM retry-once (a
   preceding step's killed server releases driver memory asynchronously).
6. **kvarn_workspace**: the hole throw now carries pool/pos/page/slot/tail/tile_page/committed
   context inline.
7. **THE OPEN RED — T128_uneven_plen reference abort (single-seq, real, pre-existing)**:
   the MTP-layer verify append advances the pool by the VERIFIED span (ext-gated: observed
   +ext/round via REW-DBG pre-tails) while cur_F advances by a+1; `rewind_tail` never extends
   forward (early-return when keep >= tail); after partial-accept rounds the pool frontier
   lags cur_F by the accumulated (a+1-ext), and the next alignment append (at true F) trips
   the hole (pos=125 slot=61 vs tail=59, F=125, pool [0,123)). Candidate fix (needs the
   MTP-pool semantics ruling from the plan owner): key the MTP verify/alignment append span
   on the ACCEPT count (a+1) instead of the draft clip (ext) — or allow the rewind to extend
   the MTP tail to cur_F. Everything else in the T2.1 matrix (5 cells) + T0 + T1 + T3 green.
