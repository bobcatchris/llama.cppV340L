# 167-COMPANION — MTP slice-OOB at fill > 16,384: condensed bug brief for LLM review

**Ask:** walk the pipeline below, read the code excerpts, and name where the out-of-bounds
`Tensor::slice` comes from for MTP k=3 at ~29.9k-token fill — or which listed hypothesis you
find most probable and what single test would decide it.

---

## 1. Environment

- ninfer TP2 engine, model qwen3.6-27b (groupwise-int weights), 2× RTX 5060 Ti 16GB, TP2.
- Failing config: `--spec mtp --draft-tokens 3 --kv-dtype kvarn_k4v2 --max-context 40960
  --kv-capacity 40960 --max-concurrency 2`, two ~14.9k-token prompts in flight (29.9k fill total).
- Works: identical config at ≤16,384 total fill. Fails: any total fill > 16,384.
  Boundary confirmed by bisect: 8,325 ✓ / 14,263 ✓ / 15,457 ✓ / 16,994 ✗ / 20,795 ✗ / ~29k ✗ / ~35k ✗.
- Binary 54f37ab66581 = tree 6a6c6bb0 (carries fix A below + A1's decoder_state transplant).
- Note 16,384 is simultaneously: the default prefill chunk, the prefix-cache capacity, and
  `effective_prefill_chunk = min(plan.prefill_chunk, plan.capacity)` — the boundary is ambiguous
  between these three.

## 2. The error

Both in-flight requests fail: HTTP 500 `slice range out of bounds`.
Throw site: `Tensor::slice` (src/core/tensor.cpp:115):

```cpp
Tensor Tensor::slice(int dim, std::int32_t start, std::int32_t len) const {
    if (dim < 0 || dim >= 4) { throw std::invalid_argument("slice dim out of range"); }
    if (start < 0 || len <= 0 || start > ne[dim] || len > ne[dim] - start) {
        throw std::invalid_argument("slice range out of bounds");   // ← the throw
    }
    ...
}
```

Tensor layout (src/core/tensor.h): `{ void* data; DType dtype; int32 ne[4]; int64 nb[4]; }`.
gdb catch-throw backtrace (both requests, both ranks):

```
#0 __cxa_throw
#1 ninfer::Tensor::slice(int, int, int) const [clone .cold]
#2 run_tp2_requests_batched(...)::{lambda(int, TpRankState&, CUstream_st*)#3}
   (prefill_impl<NullTap> is INLINED into this lambda — invisible as a separate frame)
#3 run_tp2_requests_batched(...)
#4 run_batch_dispatch → TpSubmission::wait → GenerationHandle::wait
```

## 3. Pipeline (batched MTP, per request pair)

```
run_tp2_requests_batched (tp2_backend.cpp, lambda#3, per-rank thread)
 ├─ [A] prefix/staging setup — cache_valid resets, set_mtp_proposal_extent(0)   (:3060-3074)
 ├─ [B] MTP prefill staging block                                                (:1731-1800)
 │      ph_scratch   = staging.alloc(BF16, {5120, mpc})      mpc = text->prefill_chunk()
 │      chunk_pos/ids= staging.alloc(I32,  {mpc})
 │      set_prefill_hidden_override(&ph_scratch)             (:1750 — ride-the-override arm)
 │      chunk loop over prompt (per 512-token sub-chunk):
 │        chunk_pos_h[i]=cursor+i; chunk_ids_h[i]=prompt[next]
 │        ids_c = chunk_ids.slice(0,0,nominal);  pos_c = chunk_pos.slice(0,0,nominal)
 │        ph_c  = ph_scratch.slice(1, 0, nominal);  mh_c = ph_scratch.slice(1, 0, nominal)
 │        text->mtp_forward_batch(ids_c, ph_c, pos_c, Envelope{1, cursor+nominal}, mh_c, ...)
 │        (finalize) final_mh = mh_c.slice(1, nominal-1, 1)
 ├─ [C] chunked prefill loop                                                     (:1518-1560)
 │      cursor = prefix_len; end = plen
 │      while (cursor < end):
 │        set_text_kv_base(cursor)
 │        nominal = min(chunk_limit = text->prefill_chunk(), end-cursor)
 │        chunk = text->prefill_chunk(req.prompt_tokens, cursor, nominal, finalize)
 │        cursor += chunk.processed_tokens
 ├─ [D] decode rounds: mtp_decode_batch_body (mtp_impl.h:78-135) — see §6
```

## 4. prefill_impl (text_context_impl.h:2314+, inlined into lambda#3)

Chunk loop (t0 stepping by 512 inside prefill_impl; `base` = absolute chunk start):

```cpp
const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);   // :2378 ABSOLUTE
int t0 = 0;
for (; t0 < T;) {
    int len = std::min(chunk, T - t0);
    ...
    Tensor xf = prefill_hidden_override_ != nullptr
                    ? prefill_hidden_override_->slice(1, t0, len)        // :2456 RELATIVE
                    : (prefill_hidden_.data != nullptr
                           ? prefill_hidden_.slice(1, t0, len)           // :2458 was prompt_t0 — FIXED (6a6c6bb0)
                           : work_.alloc(DType::BF16, {kCfg.hidden, len}));
    ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);
    if (is_last) {
        Tensor last_xf = xf.slice(1, len - 1, 1);                        // :2463
        ... logits, sample ...
    }
    if (prepare_mtp_prompt) {                                            // :2481 MTP-only block
        const std::uint32_t alignment_tokens = text_prefill->token_ids.size();   // = total prompt tokens
        const std::uint32_t alignment_begin  = prompt_t0;                        // :2489 ABSOLUTE
        const MtpAlignmentWindow mtp_window = plan_mtp_alignment_window(
            alignment_tokens, alignment_begin, len);
        // probe M2 printed: shifted_embedding_begin = alignment_begin + 1
        std::vector<int> mtp_ids_host(len);
        for (j < prompt_columns)
            mtp_ids_host[j] = alignment_ids[mtp_window.shifted_embedding_begin + j];  // host vector
        if (mtp_window.final_column_uses_generated_token) { ...memcpy... }
        Tensor mtp_ids = work_.alloc(DType::I32, {len});
        copy_i32(mtp_ids_host.data(), mtp_ids, s);
        if (is_last && mtp_proposal_extent_ != 0) {                        // :2535 GATE
            Tensor logits = matrix_window(io_.logits, 1);                  // :2536
            Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);          // :2537
            ... mtp_forward_batch ... AR proposal loop ...
        }
    }
}
```

`mtp_proposal_extent` setter: the ONLY assignment found in tp2_backend.cpp is
`st.text->set_mtp_proposal_extent(0);` (:3072, inside the per-request staging resets).
No setter to a non-zero value was found in the batched-MTP path.

## 5. Probe data (instrumented run, MTP@30k, failing)

```
T1 prefill_impl entry T=512 base=1024  ... (chunks 512 apart, per-rank interleaved)
S1 xf-source                           ← printed for EVERY sub-chunk up to the last
M1 before plan_mtp_alignment_window begin=30720 tokens=31013 len=293   ← final chunk
M2 window planned shifted_begin=30721
M3 mtp_ids_host filled                 ← last probe on both ranks
[MB-RANK0-ERR] slice range out of bounds
[MB-RANK1-ERR] slice range out of bounds
[req 1] error slice range out of bounds
[req 2] error slice range out of bounds
```

- M0 probe: `mtp_proposal_extent = 0` at the final chunk (is_last=1), M4gate skipped.
- At 10k fill the same request shape works end-to-end (MTP decode 3.33s/128tok, acc .59/.51).
- S1–S5 (prefill_impl body markers) all print; the throw lands between M3 and the post-loop.
- dflash2 at the SAME 30k fill works on the same binary (its own prefill path — dflash feature
  sink, no `prepare_mtp_prompt` block, proposal via `mtp_forward_batch` in the is_last branch
  gated by the same extent... yet dflash2's decode works).

## 6. Failed hypotheses so far

1. ~~`prefill_hidden_.slice(1, prompt_t0, len)` absolute-offset~~ — FIXED (6a6c6bb0); dflash2
   works post-fix, MTP still fails → second site.
2. ~~Prefix-reuse path~~ — fails identically with `--no-prefix-reuse`.
3. ~~`work_.alloc`/`copy_i32` slice inside the M3→M4 gap~~ — workspace allocs don't slice;
   `copy_i32` is a bare cudaMemcpyAsync wrapper.
4. ~~Cold-load/timeout, orphan servers~~ — ruled out with process-table + drain proofs.

## 7. Live hypotheses (ranked)

- **H1 — MTP alignment/bridge slice at absolute offset**: the `prepare_mtp_prompt` block plans
  `alignment_begin = prompt_t0` (absolute, 29.6k+) and `plan_mtp_alignment_window` +
  `shifted_embedding_begin` (30,721) feed `mtp_forward_batch` → whose internals slice a
  chunk/prefix-sized tensor (16,384-wide) at that absolute offset. Only runs for MTP ✓, only
  breaks >16,384 ✓. NOTE: `alignment_ids[shifted_begin + j]` also walks a host std::vector with
  `shifted_begin + len - 1 = 31,013` on a 31,013-element vector — last index 31,012, borderline.
- **H2 — extent=0 — CLOSED (by-construction, ratified A1 §Q2 + C441)**: `set_mtp_proposal_extent(0)`
  at tp2_backend:3072 is the ONLY batched call-site; the real-extent setter (text_prefill_impl.h:45
  card path) never fires for batched lanes. extent=0 is present in GREEN 10k AND RED 30k runs
  (M0 probe, d167_mtp10k_m0_141911: 10k GREEN, acc .53/.63) — so extent=0 alone isn't the
  differentiator and the block-skipping is by-design. The >16,384 trigger is the staging-copy
  overflow at tp2_backend:3260-3263 — guarded (b), 5b5d1b7a.
- **H3 — io_.mtp frame sized by prefix cap**: `io_.mtp` ingress/egress tensors sized
  {*, prefix_cap=16384} sliced at fill-proportional offsets in the prefill MTP block
  (`target_input_ids.slice(0,0,1)`-style calls are safe; a `slice(1, fill-…, …)` is not).

## 8. Question for the reviewer

Given §3-§5: for MTP (not dflash2), which tensor is sliced out of bounds at total fill > 16,384 —
and at which code line? Constraints from evidence: the throw is inside the inlined
prefill_impl/lambda#3 body; prefill_impl's main chunk loop COMPLETED (final chunk M3 printed on
both ranks); the M4/M5 proposal block was skipped (extent=0); dflash2 identical-shape works.

## 9. Other open items in this lane (context, not the ask)

- dflash2 DARK-BY-PERF confirmed at 10-30k fill (−12%/−31% vs plain decode; verify cost > tokens
  saved despite acc 0.508).
- W5 chain-share 11.1-11.8% << 25% bar → archived with receipts (re-entry: round < ~63 ms).
- 2b-iv guard false-positive class closed (A1 verdict ratified by tail-distinct falsifier,
  FIRES=0); guard v4 redesign pending (ADDENDUM 4 @ staging 94ae03e2).

## 10. TWO-BLOCK DISCOVERY (session end, A1 co-read needed)

The MTP prefill exists TWICE:
- **Block B** (function pre-:2834, :1731-1800): per-chunk `mtp_forward_batch(ids_c, ph_c, pos_c,
  Envelope{1, cursor+nominal}, mh_c)` riding `ph_scratch` {5120, 512} via
  `set_prefill_hidden_override(&ph_scratch)` (:1747) — per-chunk streaming, consumes-at-producer
  by construction, ph_scratch never overflows (nominal ≤ mpc = 512).
- **Block S** (`run_tp2_requests_batched` :3240-3300): the per-lane staging — full-plen
  mtp_ids/mtp_pos, the monolithic cached_ph ← prefill_dummy copy (THE OOB), full-span ph/mh
  slices, per-mpc consume loop, seed extraction, kvarn tile migration.

The batched path (:2834+) runs Block S. The ≤16,384 runs that produced tonight's good rows ran
the SAME block (S) — so S works when the stash fits. The >16,384 failure = the stash+slice
ceiling. A1's content-equivalence proof says the mh intermediate columns are dead writes ⇒
**Block S's stash+consume likely collapses into "stash the LAST chunk's mh tail only"** (the
KV accumulation already happened in [B]/mtp_prefill_chunk per chunk), leaving Block S as:
per-lane MTP KV view setup + seed = mh_last_column + tile migration.

**The remaining unknown (the S-R2 contract):** whether the draft proposal/decode phase consumes
any full-span mh beyond the seed column. If NO (A1's dead-write proof) → the fix is the collapse
(≈15 lines, zero VRAM). If YES → the consumer must be named and preserved in the restructure.

## 11. SAFE STATE (what ships tonight)

BIN 4364c85f3dcf (tree d93f5bf5+probes-reverted): ≤16,384-fill batched-MTP/dflash2 fully
working (tonight's GREEN rows); >16,384 batched-MTP = clean loud refusal
(BATCHED-MTP-FILL-EXCEEDS-PREFIX-CAP) — no corruption, no silent OOB. k=6/7 admission refused
(DFLASH2-K-EXCEEDS-ADMITTED-RANGE) with the ALLOW_K7 diagnostic seam preserved.

## 12. MTP@30k DETERMINISM CORRECTION (self-checked against the cell cards)

d2_40k_final_111632 (the 11:16 GREEN) was **dflash2**, not MTP — my earlier "MTP@30k GREEN
once" was a spec mislabel. Corrected tally: MTP@30k conc=2 = RED 4/4 (slice-OOB
len=31013 > ne[1]=16384 at :3262 — the {5120, prefix_cap} staging buffers cannot hold MTP
hidden beyond 16,384), dflash@30k = GREEN 1/1. DETERMINISTIC, no race. A1's position-scale
boundary model stands unmodified. The fix = (a)-class sizing of the MTP staging buffers
(cached_ph/prefill_dummy → {5120, max_context}: +502MB/rank) OR the (c) consume-at-producer
restructure (zero VRAM) — C441/A1 ruling pending; (b) guard keeps the current tree safe.

## 13. SPEC-LABEL CORRECTION + DISCIPLINE TOOL (post-review hygiene)

The 11:16 30k GREEN row (d2_40k_final_111632) is **dflash2 k=5** — an earlier draft of this
brief mislabeled it MTP. Corrected tally: **MTP@30k conc=2 = RED 4/4 deterministic** (slice-OOB
len=31013 > ne[1]=16384 at the {5120, prefix_cap} staging buffers), **dflash2@30k = GREEN 1/1**.
No race; A1's position-scale boundary model stands unmodified.

Discipline tool landed: `tools/smoke/diag/cell_spec_check.sh <cell-dir>` — resolves the spec
from the serve log ground truth (dispatch lines), flags card/log mismatches, dflash2 conc=1
law violations, and mtp fill>16384 second-site exposure. Every future cell runs it at close.

## 14. (c) WINDOW HANDOFF — the one remaining contract fact

The piecewise stash-consume restructure was BUILT and RUN at 30k (patch preserved, 143601/143849
dirs): the staging slice-OOB is GONE (piecewise consume works), but the run crashes with
cudaErrorIllegalAddress at the kvarn tail copy (text_context_impl.h:826) — because the STASH
SOURCE (prefill_dummy) only holds 16,384 columns of prefill hidden; columns 16,384..31,013 were
never persisted. Reading the stash source at poff ≥ 16,384 = one-past-end read (A1's byte math).

**The remaining contract fact (the only blocker):** WHO writes prefill_dummy, and at what
offsets, for plen > 16,384? Facts established: allocated {5120, prefix_cap} (:586, D-07 sizing);
passed to the decoder constructor (:591) — the decoder publishes per-position prefill hidden
into it during the general prefill; NO other writer found in the tree (grep: text_context_impl,
text_prefill_impl, tp2_backend). For plen > 16,384 the decoder's publish must either clamp
(then the stash is missing columns — the piecewise crash explained) or OOB-write (the D-07
silent-corruption class, pre-staging). ONE instrumented boot (print the decoder publish
offsets/extents at 30k fill) names it.

**The fix once named:** publish-offsets are chunk-local ⇒ the :3260 stash+consume block
collapses (the piecewise patch + seed-from-mh_scratch is the landing); publish is
absolute-OOB ⇒ the decoder publish site needs the max_context resize or per-piece streaming
(the (a)/(d) class, C441's boot-line sizing measurement applies).

Everything else in the chain is DONE and verified (see §11 safe state + the k=6 rule-14 PASS).
