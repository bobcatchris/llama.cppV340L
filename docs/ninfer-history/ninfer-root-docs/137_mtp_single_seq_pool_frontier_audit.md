# 137 — Static audit: the single-seq MTP-pool frontier vs the `cur_F` position arithmetic (T128_uneven hole)

Status: **GPU-CONFIRMED — the C1 red is NOT a single-seq bug; it was the PRE-fix batched
`any_draft` starvation abort, mislabeled as a reference failure.** Companion across docs/130 §9.8,
docs/132 §11.7, docs/135 (frontier debrief), docs/136 (§16 batched fix). Branch: `wo/kv-uniform`
@ `6857c7ea` (post-merge; the batched lane-identity fix `0257878c` + CI hardening + aggregation
fix `62084831` all landed here).

**GPU RESULT (2026-09-03, window 102): the single-seq T128_uneven_plen reference does NOT abort.**
Ran the exact cell from the merged binary (`build/tests/ninfer_tp2_batched_decode_test`, built
from `6857c7ea` — `if (!any_draft)` count = 0, i.e. post-fix) with
`NINFER_SEQ_ONLY`+`NINFER_MB_PHASEGATE`: RC=0; `[batched] 2 lanes finished`; `[SEQ-ONLY] dumped
single-seq MTP + plain for both lanes` (4 seq dumps, 128 tokens each); 101 R5 bins. The R5 trace
reaches F=125 (the reported `pos=125 slot=61` abort coordinate) at r036 (`a=3, ext=3, gen=112`),
then F=128/r037 … and completes — **no divergence, no hole**.

**Contrast (pre-fix binary):** `wo-phase-gate-i8` @ `990f3bd3` STILL has `if (!any_draft)` and its
test binary aborts RC=134 (`gqa_kv_append_kvarn: append left a hole in the tile`) in the BATCHED
phase — only B1/B4/B5 bins produced, no R5, no ref dumps. That is docs/135 §5's warning verbatim:
"an abort there [batched phase, runs first in NINFER_SEQ_ONLY] is mislabeled 'reference run failed'."

**Conclusion:** the "single-seq pool-frontier divergence" in docs/135 §1 / docs/132 §11.7 was a
misattribution. The abort was the pre-fix batched `any_draft` starvation, resolved by `0257878c`
(merged at `6857c7ea`). The single-seq path is clean. Cross-check: the repo's own
`tools/ops/pool_frontier_replay` CPU tool also reports the round pattern is SELF-CONSISTENT (no lag).
**No single-seq fix is needed.** Re-run the T2.1 / full MTP suite on the merged state to confirm GREEN.

## 1. Scope (what this bug is NOT)

Distinct from the batched `any_draft` starvation fix (`0257878c`, docs/136 §16). That fix's hunks
are batched-only (`tp2_backend.cpp` batched runner: the `any_draft` gate + `kvarn_attend_mtp_batched`).
The single-seq runner (`run_tp2_request`) is untouched by it. This audit is the **single-seq**
MTP-layer KV-pool bug behind the C1 red cell `full_mtp_t2` / `T128_uneven_plen`:

```
std::logic_error: gqa_kv_append_kvarn: append left a hole in the tile
  (pool=mtp pos=125 page=1 slot=61 tail=59 tile_page=1 committed=1)
```

## 2. The mechanism (as pinned in docs/130 §9.8 / docs/132 §11.7 / docs/135 §1)

The MTP-layer KV pool (the draft model's cache) advances by the **VERIFIED span** per round
(`ext`-gated) while the sequence frontier `cur_F` (`cur_F_mtp`) advances by the **ACCEPTED**
count (`a+1`) per round. `rewind_tail` **never extends** `tail_count` forward (early-return when
`keep >= tail_count`, and `tile_page < page` — docs/135 §1: "correctly refusing to fabricate").
After partial-accept rounds the pool frontier therefore **lags cur_F** by the accumulated
`(a+1 − ext)`; near a page boundary the next alignment append at a true position ≥ cur_F trips the
sequential-tail invariant (`slot > tail_count` → throw). The batched fix's insight (the alignment
forward is the pool-coverage obligation) does transfer: the single-seq alignment forward DOES run
unconditionally — but the divergence is in the **rewind / partial-accept accounting**, not a
skipped alignment.

## 3. The single-seq rewind/pool-advance code path (quiet step-by-step)

All from `run_tp2_request`, mtp branch (`src/runtime/tp2/tp2_backend.cpp`):

1. **State per round** (lines 1640–1661): `cur_F_mtp = plen`; per-round `F = cur_F_mtp`,
   `slot = cur_slot`. Verify inputs at `base_positions[row] + j` for every column
   (`speculative_round.cuh:48`, `positions[off] = base_positions[row] + j`), i.e. `verify_pos`
   spans `[F, F+k]`.
2. **Accept** (lines ~1819–1882): `a` accepted drafts; `next_F = F + a + 1`;
   `cur_F_mtp = next_F`; `cur_slot = a`. R5 dump logs `[cur_F cur_anchor cur_slot generated ext active]`.
3. **Rewind** (line 1893): `st.text->kvarn_rewind_mtp(next_F)` →
   `gqa_kvarn_rewind_to_token_count(*kvarn_mtp_ws_, next_F, true)` (`kvarn_workspace.cpp:320`).
   For the MTP layer, `rewind_layer`:
   - `layer.committed_pages = page(next_F)` (page = next_F/kG, slot = next_F%kG);
   - `if (tile_page < page) return;`  ← **no extend, no commit** (`kvarn_workspace.cpp:348`)
   - `gqa_kvarn_rewind_tail(layer, slot)` → `if (keep >= tail_count) return; tail_count = keep;`
     (`kvarn_workspace.cpp:305`) ← **never grows the tail**.
4. **Alignment forward** (line 1942): `mtp_forward_decode_batch(alignment_ids, verify_hidden,
   verify_pos, verify_pos, licensed_counts, kvr, Envelope{F+k+1,F+k+1}, alignment_hidden)` —
   runs UNCONDITIONALLY (no ext-gated skip). It drives `kvarn_attend_mtp`
   (`text_context_impl.h:716`), which appends `pos = kvarn_positions_1d(positions, T)` =
   `[F, F+k]` via `gqa_kv_append_kvarn_and_commit(kn, v, pos, kvarn_mtp_ws_->mtp, ...)`.
5. **AR draft steps** (line 1977): repeated `mtp_forward_decode_batch` at `ar_positions`
   (`next_F + s`, `s = 0..k-2`), also appending to the same MTP pool.

## 4. The invariant that throws (`kvarn_workspace.cpp`)

`gqa_kv_append_kvarn_with_host_pos` (lines 355–421): processes positions in contiguous runs.
The guard (line ~395):
```
if (workspace.tail_count > 0 && slot > workspace.tail_count) {
    throw ... "append left a hole in the tile (pool=... pos=... page=... slot=... tail=... tile_page=... committed=...)"  // line 402
}
```
A `slot > tail_count` on the same tile page means the append is writing at a position **past the
current tail with a gap** — i.e. the pool frontier is behind the position being appended. The
neighbouring guard (line ~569) `tail_count > 0 && tile_page < page` throws "jumped pages" for a
page-boundary jump — confirming the trigger family is a near-page-boundary bookkeeping desync.

## 5. Root-cause statement (CPU/static)

Two independent owners of the MTP-pool frontier:

| Owner | What it advances | Mechanism | Can it extend? |
|-------|------------------|-----------|----------------|
| Append (`kvarn_attend_mtp` / align + AR) | pool `tail_count` (real written KV) | scatter at `positions` | yes (only within a contiguous run) |
| Rewind (`kvarn_rewind_mtp`) | `committed_pages` + trims `tail_count` | `gqa_kvarn_rewind_to_token_count` | **NO** (returns on `tile_page < page`, and `keep >= tail_count`) |

Meanwhile `cur_F_mtp` is advanced purely by `a+1` (position arithmetic), completely decoupled
from the pool's real frontier. Nothing forces `cur_F` to respect the pool, and the rewind refuses
to bring the pool up to `cur_F`. When partial accepts land near a page boundary, the pool stays
behind `cur_F`; the next append at the true `F` meets `slot > tail_count` → abort. The pool can
also legitimately lag (silent slowdown, docs/135 §2) below the throw threshold.

## 6. Confirmation needed (GPU, coordinator to schedule)

Run the failing T128_uneven_plen cell with `NINFER_MB_PHASEGATE` set and capture the **R5** lines
(`[cur_F cur_anchor cur_slot generated next_ext active]`, `tp2_backend.cpp:1876-1889`) around the
hole (the run aborts near cur_F≈125, pool tail 59). A **two-round diff** of `cur_F` and `cur_slot`
vs the REW-DBG pre-tail should show the exact round where `slot` (pool) and `cur_slot`/`cur_F`
(a+1 arithmetic) stop agreeing — pinning the divergence round empirically and for free (the ABORT
already furnishes the terminal numbers; the diff just localises it).

## 7. Candidate fixes (ranked; docs/135 §6 + coordinator's "ONE owner")

- **D — decouple `mtp_frontier` from `cur_F` (honest bookkeeping, recommended).** Track the pool
  frontier explicitly; `rewind_mtp` truncates to `min(cur_F, mtp_frontier)`; the alignment/append
  extends **from** `mtp_frontier`. One owner, no fabrication. Moderate diff across
  `text_context_impl` + the runner.
- **B — alignment width cut to the accepted (licensed) span.** Slice the align forward to the
  accepted span so rejected/padded columns never append; the accepted span is ≤ cur_F-1 so no hole.
  Smaller but touches the op call shape; needs `column a` selection to stay within the cut.
- **A — span keying on `a+1` (coordinator's A1 first candidate).** Key the MTP verify/alignment
  append span on the accept count (`a+1`) instead of the draft clip (`ext`). Note: docs/135 §4
  warns the verify append already writes all T true positions (MTPAPPEND width=4 probe), so the
  "ext-gate" exact location is still unconfirmed — A likely needs the R5 pin before it lands.
- **C — rewind-extend for the MTP pool only.** Cheapest but weakens INV-3 (hides real shortfalls
  with stale bytes until the next verify).
- **Regardless:** the separate §5.2/§6.3 single-seq stomp (wrong-data class) remains open and is a
  higher user impact — same subsystem.

## 8. Records

- Static read of `wo/kv-uniform` @ `6857c7ea` (post-merge, batched fix + CI hardening + aggregation
  fix all present).
- Cite: `tp2_backend.cpp:1640-1990` (single-seq loop), `text_context.h:342` (`kvarn_rewind_mtp`),
  `kvarn_workspace.cpp:299-348` (rewind), `kvarn_workspace.cpp:355-421` (append + hole),
  `text_context_impl.h:716-753` (`kvarn_attend_mtp`), `speculative_round.cuh:48` (true positions).

## 10. T2.1 golden-matrix re-run (post-fix, merged @ 6857c7ea) — GREEN

`bash tools/bench/run_t2.sh`-equivalent (`python3 tools/bench/t2_golden_matrix.py`, merged
binary) on the merged state: **T2.1 golden matrix 6/6 MATCH** (batched == single-seq reference),
T64/T128 × `both_full`/`b1_shrink`/`uneven_plen` all exact-token match — including the previously
red `T128_uneven_plen`, now `MATCH`. T2.2 flake 4/4 PASS, T2.4 soak 4/4 PASS, T2.5 canaries 3/3
DETECTED (net not weakened). Overall: **`PASS: Tier 2 End-to-End Suite CLEAN`** (0 `TEST=` fails).
Confirms the C1 red is resolved by `0257878c` (merged at `6857c7ea`); no single-seq fix needed.
