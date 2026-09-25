# Catch-up rollback desk: MTP catch-up flow trace + accepted-prefix catch-up

Date: 2026-09-22. Worktree: wt-catchup (branch amd/catchup-rollback, rebased on
campaign HEAD dd9b275f6). ZERO GPU: code trace, offline timeline logs, host
parity test, gfx900 compile only. No boot, no die time, lane 8083 untouched.

## 1. THE ROUND, EXACTLY (code + mtpgain_F1_20260922_133446.log)

Campaign model: heads = 1 MTP (plain growing-KV path, not chain_heads, not
mem_shared). Per decode round, prompt = positions 0..L-1 in both caches,
`sampled` = the last accepted token, `pending_h` = target h at L-1:

| step | where | what | F1 med |
|------|-------|------|--------|
| draft | `draft_mtp::draft` | ctx_dft decodes `sampled` @ L, then each drafted token (1 tok/step, ~2.5 ms/step); cells L..L+m-1 written | 11.0 ms total |
| ckpt block | `update_slots` | `seq_rm(ctx_dft, ckpt.pos_max+1 = L, -1)` DISCARDS the draft-phase cells; draft KV back to 0..L-1 | ~0 |
| verify | server decode | ctx_tgt decodes [sampled @ L, d1..d3 @ L+1..L+3], n_outputs 4 | 163.16 ms |
| catch-up | `draft_mtp::process` | ctx_dft decodes THE SAME 4 rows (logits 0, embd = target h shifted right, row 0 = pending_h), n_outputs 0 | 11.66 ms |
| accept | server `post_decode` | k of 3 drafts accepted; `pending_h` = verify_h[k]; prompt += k tokens; pos_next = L+k+1; `seq_rm(ctx_tgt + ctx_dft, pos_next)` | ~1 ms host |

Log window (4.09.850 - 4.10.035): `[draft: steps = 3, 2.549 ms/step]` ->
`[n_tokens = 4, issue = 166.4, n_outputs = 4]` -> `[n_tokens = 4, reused = 0,
issue = 11.9, n_outputs = 0]` -> accept-loop drains -> next round.

Fate of the 4 catch-up rows, per k (accept = 0.66667, mean len 3.00, E[k] = 2.0
measured 84/126):

| k | rows decoded | rows that survive pos_next = L+k+1 | wasted device work |
|---|--------------|-------------------------------------|--------------------|
| 0 | 4 | 1 (row 0) | 3 rows |
| 1 | 4 | 2 | 2 rows |
| 2 | 4 | 3 | 1 row |
| 3 | 4 | 4 | 0 |

So the baseline catch-up re-decodes 4 - (E[k]+1) = 1.0 row per round of pure
waste (25% of its rows), and the per-k tail is heavy: at iid 2/3 acceptance,
P(k=0) = 1/3 of rounds waste 3 of 4 rows. Also redundant: the draft phase
already decoded the row-0 cell (sampled @ L) this round before the ckpt block
rm'd it (see 4).

## 2. FEASIBILITY VERDICT - IMPLEMENTABLE-CLEAN (the E-078 ordering concern dissolves)

E-078 flagged "reorders KV rollback - not mechanical". That holds only for a
pre-sampling design that writes everything and relies on seq_rm to undo. The
implemented design moves the decode AFTER the accept loop and decodes only
rows 0..k: the rejected cells are never written, so there is nothing to roll
back and the post_decode seq_rm finds nothing beyond pos_next. Both caches end
every round carrying exactly the accepted prefix, identical to baseline.

Every cache / bookkeeping structure involved, checked:

- ctx_dft KV: the only cache the catch-up writes. Prefix arm: cells L..L+k
  written (baseline: L..L+3 then rm to L+k). Cell content is computed from
  identical inputs (same tokens/positions/embd rows; row j attends cells
  0..L+j-1, all identical) - only the ubatch width differs (see numerics).
- ctx_tgt KV: untouched by the catch-up; already accepted-only via its own
  post_decode seq_rm. No change.
- pending_h / verify_h (draft_mtp): copied in process() as before (host
  memcpys); accept() still overrides pending_h = verify_h[i_h]. The staged
  batch keeps PRE-accept copies of the embd rows, so the later pending_h
  writes cannot corrupt the flush inputs.
- server prompt vector / pos_next / spec_i_batch: unchanged.
- checkpoint-restore path (FULL type, or RS with n_rollback > n_rs_seq):
  accept() is not called for the slot, so the flush decodes its staged rows in
  full AFTER the restore - the baseline decoded them before the restore and
  the restore rm'd them again. The prefix arm leaves extra cells beyond the
  restored boundary for one round: causally masked (nothing attends >= its own
  pos), rewritten by the re-verify round, removed by that round's seq_rm.
  Pinned by the host test.
- mtmd chunk path (process() with no catchup() follow-up) and any future
  caller: an unflushed staging self-heals in full at the next process() /
  draft() entry, before anything reads the draft KV.
- mem_shared (gemma4): no catch-up decode exists; the flag is ignored with a
  warning. chain_heads: the flush mirrors the per-head seq_rm + layer-offset
  loop of process().

Numerics caveat (honest): the catch-up ubatch width changes (4 -> k+1 rows).
Per-row inputs are identical, but GEMM tiling/reduction order can differ with
batch width, so last-ulp KV differences are possible and can flip a near-tie
draft sample. Logic-identical, NOT bit-identical on device: the served-arm
gate is accept/mean-len within noise, not identical sha (unlike R2/R3, which
touched only host sync paths).

## 3. IMPLEMENTED (env-gated; unset = byte-identical by construction)

LLAMA_DRAFT_PREFIX_CATCHUP=1:

- `common/speculative.cpp` (draft_mtp): `process()` builds the catch-up batch
  byte-identically (tokens, positions, embd rows incl. the pre-accept
  pending_h row) and stages it instead of decoding; `accept()` records
  `catchup_rows[seq] = i_h + 1` (rows 0..k; row 0 = the sampled token whose
  cell IS a prompt cell at pos_next = L+k+1); new `catchup_decode()` builds
  batch_catchup from the staged rows (full rows when no accept decision
  exists) and runs the same decode loop as process() (per-head seq_rm +
  offset for chain_heads). New base-class hook `catchup()` + free function
  `common_speculative_catchup(spec)`; self-heal at process()/draft() entry.
- `common/speculative.h`: declares common_speculative_catchup.
- `tools/server/server-context.cpp`: one flush call at the end of post_decode,
  after the spec accept iterate; failure throws like the process() path.
- Engagement logging: `[spec-timeline] catchup: rows = N, decode_issue = X ms`;
  the catch-up decode-timeline line now shows n_tokens = k+1 (1..4) instead of
  a constant 4, and the process line's decode_issue drops to the batch build.

Flag unset: staging never happens, every new call site early-returns, the
regular in-process catch-up code path is unchanged.

Host test `docs/amd-port/tests/test_prefix_catchup_host.cpp` (standalone, no
ggml): both arms driven through scripted accept/reject patterns over synthetic
sequences; every ctx_dft decode records the attended-view hash (causal cells
0..p) and drafted tokens derive from it, so any divergence flips the checks.
Pinned: identical drafted-token sequences and draft-phase views across k = 0..
3 rounds; catch-up rows of the prefix arm are exactly rows 0..k of the
baseline's (same token/pos/embd source); post-round cells below pos_next
identical; prefill chunks and 1-token rounds decode full rows; the mtmd-style
unflushed staging self-heals to the baseline cells; the ckpt-restore round
plus its re-verify round stay equivalent; row reduction pinned (48 -> 28 rows
on the 12-round script; 25% expected at the measured accept). ALL PASS;
ASAN/UBSAN clean.

gfx900 compile: cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900
(Release, /home/chris/opt/cmake/bin/cmake), targets llama + llama-common +
llama-server: clean, ZERO compiler warnings (speculative.cpp,
server-context.cpp recompiled).

## 4. EXPECTED EFFECT AND THE NEXT LEVER (not implemented)

At E[k] = 2.0: catch-up rows 4 -> 3.0 avg (25%), med issue 11.66 -> ~8-9 ms,
wall -1.5..-3 ms/round at ~190 ms = +0.8..+1.6% decode t/s class, concentrated
in the k = 0 tail. Not row-linear: the fixed per-decode cost stays.

Next lever, blocked on a shared-code change (documented, no code): the row-0
cell (sampled @ L) is re-decoded by the catch-up although the draft phase's
step 0 decoded the identical (token, pos, embd) row this round and the ckpt
block then rm'd it. Keeping that cell (ckpt-block rm bound L -> L+1, plus a
"draft ran this round" guard for the reuse/no-draft rounds) would drop the
catch-up to rows 1..k = E[k] = 2.0 rows: another ~25% off the same line.

## 5. SERVED-VALIDATION ARM (needs rebuild; one combined window per E-081)

R5: campaign line + LLAMA_DRAFT_PREFIX_CATCHUP=1 + LLAMA_SPEC_TIMELINE=1
LLAMA_DECODE_TIMELINE=1 + -lv 4, decode-only guard. Gate: accept 0.66667 /
mean len 3.00 WITHIN NOISE (ulp ties can move single drafts; a systematic
move is a bug). Expected signature vs R1: catch-up decode-timeline n_tokens
varies 1..4 (not constant 4), new [spec-timeline] catchup lines, catch-up
issue med 11.7 -> ~8-9 ms, wall -1.5..-3 ms, t/s +0.8..+1.6% class; greedy
sha may differ at tie level (documented numerics caveat).

## 6. Files

- common/speculative.{h,cpp}: staging + catchup_decode + catchup() hook +
  common_speculative_catchup.
- tools/server/server-context.cpp: post_decode flush call.
- docs/amd-port/tests/test_prefix_catchup_host.cpp (new).
- Receipt: docs/amd-port/results/CATCHUP_ROLLBACK_2026-09-22.md (this file).
- Ledger: E-085 appended to docs/amd-port/OPTIMIZATION_PLAN_TP3_200K.md.
