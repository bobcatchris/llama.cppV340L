# 167 — dflash2 prefill slice-OOB at prompt fill > 16,384 (fixed)

**Author:** A2 · 2026-09-10 · fix commit `6a6c6bb0` (habitat) · verified on BIN `54f37ab66581`
**Severity:** product bug — every dflash2 dual-request prompt over 16,384 prepared tokens
failed with HTTP 500 `slice range out of bounds`, well below the 40,960 max-context.
**Status:** FIXED + verified. Rule-14 unit owed (see tail).

## Symptom

dflash2 (batched, conc=2) requests whose **prepared prompt exceeds 16,384 tokens** fail at
prefill: both requests get `{"error":{"message":"slice range out of bounds"}}`; serve log shows
`[MB-RANKn-ERR] slice range out of bounds` per rank. Prompts under the cap work.

## 7-rung bisect (40,960-cap server, kvarn_k4v2, dflash2 k=5, conc=2, BIN 7bc6d54a8fbc)

| fill (prepared tokens, ~5.5 B/tok on this text) | result | evidence dir |
|---|---|---|
| 8,325 | OK (falsifier, acc .39/.45) | `results/falsifierD_r3_101656` |
| 14,263 | OK (decode 6.66s/128tok, acc .47/.29) | `results/d2_16k_bisect_103231` |
| 15,457 | OK (decode 5.84s/128tok, acc .40/.47) | `results/d2_155k_bisect_103559` |
| 16,994 | **FAIL** | `results/d2_17k_bisect_103455` |
| 20,795 | **FAIL** | `results/d2_20k_bisect_103358` |
| ~29,000 | **FAIL** | `results/d2_40k_decode_r3_103003` |
| ~35,000 | **FAIL** | `results/d2_40k_decode_r2_102822` |

Independent of `--no-prefix-reuse` (r3 tested it explicitly — the prefix-REUSE path is not the
culprit). The 16,384 boundary matches the DEFAULT prefill chunk.

## 500-path + root cause

Throw site: `Tensor::slice` (src/core/tensor.cpp:115, `start > ne[dim]`), reached via
`TextContext::prefill_impl<NullTap>` (gdb catch-throw backtrace, both ranks,
`results/d2_167_gdb3_105930/gdb_out.txt` lines 85+96 — first gdb attach attempt was blocked by
yama ptrace_scope=1; launching UNDER gdb worked, with one orphan pair left stopped at a
catchpoint and killed before the verify boot — honest trail in 29df7c84).

Root cause (one line, text_context_impl.h prefill_impl):

```
Tensor xf = prefill_hidden_override_ != nullptr
                ? prefill_hidden_override_->slice(1, t0, len)          // MTP path: RELATIVE — correct
                : (prefill_hidden_.data != nullptr
                       ? prefill_hidden_.slice(1, prompt_t0, len)      // BUG: ABSOLUTE offset
                       : work_.alloc(DType::BF16, {kCfg.hidden, len}));
```

`prefill_hidden` is a **chunk-sized scratch**: `{hidden, effective_prefill_chunk}` where
`effective_prefill_chunk = min(plan.prefill_chunk, plan.capacity)` (layouts_impl.h:96, :205) —
16,384 by default. `prompt_t0 = base + t0` is the ABSOLUTE prompt position of the chunk being
prefilled. Slicing a chunk-sized buffer at an absolute prompt offset is in-bounds only while
`prompt_t0 + len <= 16,384` — i.e. the default chunk size masked the bug for every prompt that
fit one chunk. The DFlash2 verify entry (`dflash_impl.h:359-361`) passes the scratch as
`prefill_hidden` without an override, so round-0 verify at `base = prompt fill` slices at
`prompt_t0 ≈ fill` → OOB for fill > chunk.

**Fix:** `prompt_t0` → `t0` (chunk-relative), matching the override branch's indexing. `xf` is a
per-call output scratch consumed by rmsnorm immediately after — the offset must be in-bounds and
chunk-local, which `t0` (0..T−1) is. The same one-line change also closes the latent MTP-path
exposure at fill > chunk (any caller passing the scratch without an override was affected).

## Verify (BIN 54f37ab66581)

| rung | pre-fix | post-fix | evidence |
|---|---|---|---|
| 17k fill (was-firing) | slice-OOB | **SLICE_OOB=0** — but note: trips 2b-iv 5/5, see below | `results/167_verify_17k_r2_111331` |
| ~30k fill, tail-distinct | (unreachable pre-fix) | **decode 5.28s/128tok, acc 0.53/0.51, per-draft 0.508, tok/round 3.66/3.56, chain 11.1%, FIRES=0** | `results/d2_40k_final_111632` |

## GUARD INTERACTION (input to A1's fix-A redesign)

Post-fix, the 17k rung trips the 2b-iv guard 5/5 (`results/167_verify_17k_r2_111331`): the cyclic
fact-sentence bodies are structurally identical across lanes (values differ, template identical),
so at long fill the two lanes' drafting windows are near-identical and argmax legitimately
converges to the same chain. The planned window-EQUALITY discriminant (164 fix-A) is TOO WEAK:
these windows differ token-wise yet chains converge legitimately. Whatever guard redesign lands
must PASS on 17k-cyclic bodies (correct behavior) and still THROW on true input-collapse
(the FIX-C (d) synthetic row). That tension is the spec.

## Rule-14 unit (owed)

Synthetic `prefill_impl` invocation with fill > chunk must-RED pre-fix (slice throws) and GREEN
post-fix; a mutation re-introducing `prompt_t0` must turn it RED. Not yet written — P-B adjacent.

## FIX-SHAPE SCOPE NOTE (C441 seq-56, farm7-critical-path — the ruling input)

### (0) MAIN-LINEAGE GUARD PORT — REQUIRED, FIRST ITEM

Verified this session: **main:3383-3386 carries the identical unguarded block** (memcpy
`plen*5120*2` into {5120, prefix_cap=16384} cached_ph/prefill_dummy + `slice(1, 0, plen)`).
Main has neither the refusal nor the fix; farm7 cuts from MAIN lineage, and the dflash2/FIX-D
merge CREATES the >16384 consumer there (the fill-gate analysis: single-chunk totals never
exercise it; total fill > 16,384 always does).

**Action:** port the habitat guard (5b5d1b7a, fail-closed throw
`BATCHED-MTP-FILL-EXCEEDS-PREFIX-CAP: plen=N cap=16384`) to main's :3383 block — same
pattern, rides the SAME merge as the dflash2/FIX-D integration (merge-riding, not follow-up).

**Post-merge GREP GATE:** `BATCHED-MTP-FILL-EXCEEDS-PREFIX-CAP` must appear at EVERY site
docs/168's table names, in BOTH trees — a per-hunk tp2_backend merge could otherwise leave
main's copy unguarded (merge-exposure in code form). Grep is the gate.

### The decisive-check table

| check | boundary moves to 8,192? | meaning |
|---|---|---|
| `--prefill-chunk 8192`, capacity held | YES | offending extent = the PREFILL CHUNK dim (basing bug in prefill_impl MTP block) |
| `--prefill-chunk 8192`, capacity held | NO (stays 16,384) | offending extent = PREFIX/CAPACITY dim (staging sizing, H1/H3) |
| `--prefill-chunk 32768` (if allowed) | boundary moves to 32,768 | confirms chunk-dim from the other side |

Zero rebuild if the existing binary honors `--prefill-chunk` at runtime — it is a plan input,
so ONE boot per rung; run the 17k-fill repro at each rung.

### (c) chunked staging — contract surface

The monolithic copy + full-span slices at :3260-3270 become per-mpc pieces, mirroring the
per-512 `mtp_forward_batch` pattern already 20 lines below (:3283-3290 in habitat):
copy/slice `mh_c`/`ph_c` per `off += mpc` piece. Callers of the block: the batched-MTP prefill
staging only (single-seq path is guarded separately at :1358). Bridge contract touch:
`mtp_forward_batch` is ALREADY called per-piece — no signature change; only the staging
copy/slices change shape. Blast radius: 1 block, 1 file, both trees.

### (d) size-at-load — collapses into (a) at full ctx

Sizing `cached_ph`/`prefill_dummy` to `max(prefix_cap, needed-by-shape)` at load: "needed" is
bounded by max_context (plen ≤ max_context by admission), so at ctx 40960 this equals (a):
+502MB/rank persistent. Only differs from (a) when operators cap ctx < prefix_cap.

### (a) memory reality check

+502MB/rank FIXED at 40960 ctx — LINE-M anchor has ~267MB slack @40960 conc=2 ⇒ (a) as-is
likely makes the flagship shape not fit. If (a) is chosen anyway, it must ride a capacity
re-plan, not a silent buffer growth.

### RECOMMENDATION

1. (0) guard port to main — rides the merge (required, cheap).
2. (b) habitat guard — ALREADY LANDED (5b5d1b7a).
3. (c) chunked staging — the real fix, zero VRAM, small blast radius; implement in the
   fix-window; rule-14: 17k-fill must-RED pre-(c)/GREEN post-(c); 15,457-rung stays GREEN.
4. Fold the alignment_ids host off-by-one (30,721+292 = 31,013 on 31,013 elems) into the same
   window (silent heap over-read, Response-5 find, C441 in-scope ruling).
5. L0/L1 farm cells: era-stamped red-on-refuse until (c) lands — farm7 L-green requires the
   fix-shape ruling, making it critical-path as ruled.

### (c) CONCRETE DESIGN (post deep-read of the staging block + mtp_prefill_chunk)

Current flow (habitat :3240-3300, per lane b):
1. General prefill ([C]) writes per-position target hidden into `prefill_dummy` via the
   sub-chunk MTP block (`mtp_prefill_chunk(mtp_ids, xf, ...)` per 512 — consumes xf, appends
   MTP-layer KV at absolute positions into the paged mtp_kv pool, extracts final hidden on
   the last chunk).
2. Staging block (:3260): copies the WHOLE plen of hidden prefill_dummy → cached_ph (so
   prefill_dummy can be reused as the mh output), then consumes ph per 512 via
   `mtp_forward_batch`, writing mh back into prefill_dummy per piece. Seed = mh[plen-1].

The (c) restructure (piecewise stash-consume — removes the plen ceiling):
```
cap = prefix_cache_capacity (16384)
for (piece_off = 0; piece_off < plen; piece_off += cap):
    piece = min(cap, plen - piece_off)
    copy cached_ph[0..piece] ← prefill_dummy[piece_off..piece_off+piece]   // stash piece
    ph_c = cached_ph.slice(1, 0, piece)
    for (off = piece_off; off < piece_off + piece; off += 512):            // existing pattern
        mh_c = prefill_dummy.slice(1, off, len)                            // in-place output
        mtp_forward_batch(mtp_ids.slice(0,off,len), ph_c.slice(1, off-piece_off, len),
                          mtp_pos.slice(0,off,len), Envelope{1, off+len}, mh_c, ...)
    // seed column: after the FINAL piece, mh[plen-1] lives in prefill_dummy[plen-1] ✓
```
Read-before-write safety: each piece's ph is stashed to cached_ph before the mh in-place
write touches prefill_dummy at those offsets; pieces are disjoint.

OPEN CONTRACT FACT (one instrumented card cell, ~15 min): confirm where the general prefill
publishes per-position hidden for plen > 16,384 — if the [C] prefill writes prefill_dummy at
absolute offsets beyond 16,384, the silent OOB WRITE happens during PREFILL (before our
staging guard) and (c) must ALSO repoint that publish (to the piecewise stash or a
max-context-sized buffer). The probe: print the hidden destination tensor's ne + write offset
inside the prefill MTP block at fill 30k. If the publish is already chunk-local (likely —
prefill_impl consumes xf per 512 sub-chunk), then (c) alone closes the bug with zero extra
memory and zero prefill-side changes.

### (0b) THE D-07 CONTRACT (the sizing ruling input)

tp2_backend:457 (D-07, on the record): `cached_ph` and `prefill_dummy` are each
`{5120, P} BF16 where P = min(prefix_cache_capacity, max_context)` — the decoder PUBLISHES
per-position prefill hidden into prefill_dummy within that budget by design. At 40960 ctx:
P = 16,384 columns (168 MB/buffer). A batched-MTP fill of 31,013 needs 31,013 columns ≈ 290 MB
— **outside the D-07 envelope by ~122 MB/buffer**, and the staging copy+slice at :3260-3263
amplifies it (the full-plen memcpy + full-span slices).

FIX-SHAPE CONSEQUENCES against D-07:
- (a) resize buffers to max_context: +502 MB/rank persistent — outside the LINE-M slack
  (267 MB @40960 conc=2) ⇒ likely requires a capacity re-plan (fewer KV pages or a smaller
  default ctx for batched-MTP) — a DESIGN ruling, not a patch.
- (c) piecewise stash-consume (my attempt): fixes the COPY but the CONSUME (mtp_forward_batch
  mh output) still needs a home per piece — solvable with a rolling mpc-sized scratch
  (mh_scratch), which is what the parked patch does — but the round-0 crash showed the mh
  consumer contract (S-R2) is not yet fully named.
- (d) documented limit (the landed (b) guard): batched-MTP fill ≤ prefix_cap, loud refusal
  above — SAFE now, honest now, upgrade path preserved.

THE DECISION C441/A1 owe the merge: D-07 revision (resize) vs D-07-preserving (c) piecewise
with the S-R2 contract named first. Either way the (b) guard ships NOW (it converts silent
corruption into a loud refusal on every lineage).

## LANDED — (c) consume-at-production, batched mirror of single-seq [B] (2026-09-10 ~20:50Z)

**The S-R2 contract read named the whole game (15-min cap, 4 facts):**
1. TP2 builds TextContext with `prefill_chunk_ = min(512, max_context)` (tp2_backend :586-591)
   and `prefill_hidden_ = state->prefill_dummy` — the [C] loop therefore runs **512-token
   chunks**, each publishing its target hidden via the prefill_impl `xf` rmsnorm write.
2. `mtp_forward_tail` is **ATTEND-ONLY** — it never appends KV. The draft KV is appended
   solely by [C]'s per-chunk `mtp_prefill_chunk` at ABSOLUTE positions.
3. `mtp_forward_batch` contract: T = ids.ne[0], `1 <= T <= prefill_chunk_` (=512),
   hidden/mtp_hidden both {5120,T}; envelope = causal visible extent. The k+1=6 proposal
   width exists ONLY decode-side (`mtp_forward_decode_batch` / verify batch), which consumes
   `mb_drafts` + the pool — **the two worlds compose through the {5120,1} seed column**
   (exact `mtp_propose_batch`/`mtp_forward_ar_step` shape), never through shared widths.
4. The single-seq [B] block (ph_scratch override + per-chunk consume + finalize-seed) is the
   proven reference — **the fix MIRRORS it in the batched loop instead of inventing a new
   prefill_impl restructure**: A1's dead-write proof means only the FINAL chunk's consume is
   live; running all chunks is the same cost structure the old Block S already had.

**The change (run_tp2_requests_batched only, one file):**
- BEFORE [C]: bind the MTP lane view + publish the lane pool mapping (moved up; alloc-side,
  idempotent), alloc `ph_scratch {5120,mpc}` + `last_mh {5120,1}` + plen-sized mtp_ids/mtp_pos
  (i32 only), `set_prefill_hidden_override(&ph_scratch)` — the override branch slices at `t0`,
  forcing chunk-relative publication regardless of the non-override branch's behavior.
- PER CHUNK (inside the loop, where the hidden is alive): blocking D2H of t0 at finalize
  (patches mtp_ids[plen-1]), `kvarn_rewind_mtp(0)`, `mtp_forward_batch(ids_c, pc, pos_c,
  Envelope{1, cursor+nominal}, pc /*in-place, single-seq-proven*/, ...)`, finalize copies the
  seed column into `last_mh`.
- AFTER [C]: clear the override; `mb_seed_hid[b] <- last_mh`; kvarn MTP tile migration
  unchanged. **Block S DELETED WHOLESALE** — the BATCHED-MTP-FILL-EXCEEDS-PREFIX-CAP guard
  dies with it: no plen-sized BF16 staging remains on the path, so the fill ceiling is gone
  and prompt admission (the allocator) is the only gate (VRAM-LAW-clean by construction).

**VERIFY CHAIN (C441-ruled order, BIN e830100427b0, kvarn_k4v2 conc=2 mtp k3, evidence
results/fix167_chain/):**
| rung | result |
|---|---|
| 17k (was RED 4/4) | **GREEN** acc .80/.49 t/r 3.50/2.54 det=OK (actual fill ~24.6k — deeper into the was-RED band) FIX17 |
| 30k keystone | **GREEN** acc .47/.77 t/r 2.50/3.40 det=OK 4/4, ERR=0, walls 67.6s FIX30r2 |
| 10k MTP regression | GREEN acc .78/.48 t/r 3.44/2.55 det=OK REGR10 |
| dflash10k insurance | GREEN on W1_dflash_k5 (ctx 8192): acc .35/.31, 2b-iv=0 — shared machinery intact INS10_W1 |
| rule-15 substitution grep | CLEAN — zero functional stash/consume/guard symbols in the batched region; only the deletion-documenting comment |
| must-NO-WRITE (SLICEDBG @30k) | **PASS — 35,126 boundary-crossing slices, ZERO with start+len>ne** |

**30k acceptance side-by-side vs the .67-.70 band:** single-seq MTP reference row in the same
log family reads acceptance=0.67 (the .67-.70 band); the batched keystone lanes read
.47/.77 — lane B sits in the band, lane A below it, mean .62; both inside the suite's legal
[.4,.95] band, and consistent with batched per-lane body variance (10k: .78/.48, 17k:
.80/.49 on the new cyclic bodies). Determinism (md5-identical iters) is the strong signal.

**Honest trails:** (1) first 30k attempt hit ADMISSION (bodies prepared 43,290 > 40960 —
trap-6 inverse: this body family is 3.883 B/token, not 5.5; regenerated + asserted pre-boot).
(2) The chainshare harness at ctx 10240 and the runner 10k cell BOTH trip the 2b-iv
lane-blindness guard on cyclic fact-sentence bodies — the DOCUMENTED pre-my-edit class
(results/167_verify_17k_r2_111331 @ 54f37ab66581, doc §GUARD INTERACTION; A1's 164 fix-A
lane). W1 (2b-iv-clean geometry) GREEN proves the machinery.

**MAIN :3383 port statement (fix-v2-backport, for A1's nod package):** main's Block S copy
(:3383-3386 identical block) must be resolved WHOLESALE to this landed shape — delete the
guard + stash + consume + seed block and adopt the pre-loop staging (override + per-chunk
consume + finalize seed) exactly as in habitat's tp2_backend.cpp (this commit). NO hunk-merge
that keeps main's block and re-adds the guard: the guard's string is correctly DEAD in
habitat — the 168 grep gate is re-homed (C441 seq-82) to the rule-15 absence form: no
stash/consume symbols AND no new plen-sized staging write at every 168-table site.

**Costs measured:** prefill 26.2s @ ~24.6k fill, 33.6s @ ~30.6k fill — the interleaved
per-chunk consume has the same asymptotics as the old Block S (plen/512 iterations, growing causal attend); decode/walls unchanged vs reference rows.
