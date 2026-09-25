# 64 — MTP rollback-state audit (llama.cpp #27173 AUDIT C)

Status: DONE 2026-08-24 (static audit, read-only). Roadmap item #2, second
audit. Verdict: **PASS — the #27173 failure class does not exist in our MTP
rollback.** No code changes required; one future-proofing note (§3).

Scope: docs/58 §1 item 6 — "Verify a short rejected batch cannot overwrite
the snapshot it rolls back to." Our MTP rollback trims via verify-column
snapshots of the GDN recurrent state (slot ring) + KV overwrite-by-position;
KVarN workspace rewinds per round.

## 1. Mechanism (verified in code)

Per spec round (`tp2_backend.cpp` ~L1195-1280):
- Verify batch is **always width k+1** — `[anchor, d0..d_{k-1}]` at positions
  F..F+k. `slot_pair = {cur_slot, 0}` → initial state slot = `cur_slot`,
  snapshot base = 0; the post-state of column j lands in slot j.
- Budget/stop clamping happens **after** acceptance (`commit_count`), so a
  short remaining budget never shrinks the verify batch — there is no "short
  verify" path at all (unlike llama.cpp's failure shape).
- Acceptance selects `cur_slot = a` (a = accepted drafts, ∈ [0,k]) — the slot
  written by this round's column a. KVarN workspace rewinds to
  `next_F = F + a + 1` in the same step (`kvarn_rewind_text`).

## 2. Why the #27173 bug cannot occur here

| Invariant | Evidence |
|---|---|
| Verify width is never < k+1 | Envelope `{F+k+1, F+k+1}` fixed in `target_verify_batch` call; no valid-column masking of the draft columns (`st.valid_v` masks attention only). |
| Snapshot ring (slots 0..k) fully rewritten each round before selection | Verify writes slots 0..k unconditionally; acceptance reads slot a ∈ [0,k] afterwards. No stale-slot selection possible. |
| Initial state read precedes any overwrite of the initial slot | `recurrent.cuh` `recurrent_bf16_body<Snapshot>`: state is loaded from `initial_state_slots` into registers **before** the token loop; snapshots are stored inside the sequential loop. Even when `cur_slot == 0` (in-place column-0 write), read-before-write holds. |
| Recurrence is sequential across columns | Kernel grid is `(v_heads, batch, state_tiles)` — parallelism is over heads/state tiles only; the token loop is a plain sequential `for` per block. No cross-column race on the initial slot. |
| Prefix-cache slot outside the ring | Slot layout: 0..k snapshot ring, `cache_slot = 2k+1` = prefix slot (`tp2_backend.cpp:187`). Ring max index k < 2k+1 for all k ≥ 1; verify never touches the prefix slot. |
| KV rollback is overwrite-by-position | Paged pool mapping re-published each round; rejected columns' KV positions are rewritten by the next round's verify (or the request ends). Attention is position-bounded, so stale tails beyond committed F are never attended. |

## 3. Future-proofing note

The read-before-write invariant (§2 row 3) depends on the snapshot kernel
staying **sequential across columns**. If a future optimization parallelizes
the token loop across columns (each column starting from its own initial
slot), the "initial read precedes overwrite" property breaks when
`cur_slot` falls inside the ring. Any such change must re-verify this audit
and add a unit test: write distinct states to slots 0..k, run one verify
round with `cur_slot = k`, check slot k's pre-round value was actually used
for column 0 (e.g., via a crafted input where the wrong initial state is
detectable in the output).

## 4. Battery test disposition

docs/58 proposed "long accept → short reject → continue, compare vs no-MTP
greedy". With ~82% MTP acceptance, short rejects occur naturally in every
long generation; T10 (cross-request determinism with MTP on) plus the MTP
acceptance gate in run_ci.sh cover this statistically. A forced-reject test
would need prompts engineered to break the draft — not worth it now. No new
test added.
