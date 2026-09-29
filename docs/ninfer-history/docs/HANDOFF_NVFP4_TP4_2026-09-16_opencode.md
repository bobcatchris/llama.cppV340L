# HANDOFF — NVFP4@TP4 garbage-after-68: host-local decider + static lap (opencode session, 2026-09-16)

## Session facts
- Host probe: `rocm-smi` = 4× Vega 10 V340 lines → AMD line, `docs/amd/COORDINATOR.md` governs.
- Checkout: `/home/chris/dual_5060_ti_ninfer`, branch `amd/main` @ `288a648e` (chair ledger 00:5xZ2, calendar-is-a-table).
- KFD 0 the whole session (no GPU touched). Disk `/`: 116G/90G/21G (82%) at start AND end — zero-write session (reads via `git show`/seek-reads only; two `/tmp/opencode/decider74v80.py` scripts deleted after run).
- No builds, no boots, no clones, no channel posts. r55 bank read from lane commit `8a4c19f2` (NOT in main — read-only `git show`, no checkout).

## 1. C5 tokenizer-identity re-verified (still blessed)
- Ran `tools/v340l/nvfp4/c5_tokenizer_identity_probe.py` → rc=0.
- Both artifacts `frontend/tokenizer.json`: 12,809,320 B, `sha256(full payload) 0997f410c57a1f4e…b9f3`, vocab 248,044 + added 33 @248044..248076, extent 248,077, holes 0. Verdict: BYTE-IDENTICAL. Tokenizer is not the confound for this artifact pair.

## 2. Raw-lines r55 re-verify — §25-ADD settlement CONFIRMED + bug reproduced live
- File: `8a4c19f2:results/amd/p3/G18r55_c67_hkv.txt` (19,159 B). §17 format is TRAILING-delimiter (bodies PRECEDE their `ph=` delim — proven by line order: `ph=72` @L165 precedes tripped block @L172).
- Backward-attributed table (l=0 K, 9-char prefixes):

| plen | s0 | s1 | s4 | s8 | s16 | s24 |
|---|---|---|---|---|---|---|
| 53 | 8474742a2 | 6daeca88c | 7d36f7b14 | ee078bcf0 | dd47cfc45 | 5b08892d6 |
| 70 | 31c006123 | 7c57b31fc | 2a1bad237 | bd3aba619 | f7323bdce | c90e4d14e |
| 71 | 31c006123 | 7c57b31fc | 2a1bad237 | bd3aba619 | f7323bdce | c90e4d14e |
| 72 | 31c006123 | 7c57b31fc | 2a1bad237 | bd3aba619 | f7323bdce | c90e4d14e |
| 74 | 31c006123 | 7c57b31fc | 2a1bad237 | b8c035a10 | 56a18a4d9 | e9725952f |
| 80 | 31c006123 | 7c57b31fc | 2a1bad237 | b8c035a10 | 40ab6db8c | 4e553c900 |
| 90 | 31c006123 | 7c57b31fc | 2a1bad237 | b8c035a10 | 1e4e9c178 | 4078b1c02 |

- Matches `c9e5dd6f`/`654464b6` verbatim: row-8 holds C @70/71/72, trips once in (72,74]; row-4 constant (NO row-4 trip); rows 16/24 churn per-leg @74/80/90.
- Failure-mode demo: my first (forward-attribution) parse falsely showed the trip AT 72 — the exact one-block-late disease. Raw-lines-over-parsers holds.

## 3. Tokenizer decider 74-vs-80 — CORRUPTION REACHES ≥16, exoneration NOT restored
- Bodies reconstructed from banked runner `8a4c19f2:results/amd/p3/G18r55_c67_agent4.sh` (case-map): `"Reply with exactly the single word BLUE. Context:" + " alpha"×N`, N=12 (plen 74b) / N=18 (plen 80). Server `prompt_tokens` 70/71/72/74/80/90 = 62+N exactly (N=8/9/10/12/18/28).
- Method (script deleted after run; reproducible): seek-extract in-band `tokenizer.json` from `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer` (12.8 MB payload only, never 18 GB); GPT-2 byte-level table; ASCII-equivalent Split pre-tokenizer via `re` (bodies asserted ASCII-only); rank-loop BPE over the artifact's own 247,587 merges + vocab. No `tokenizers`/`transformers`/`regex` modules exist on this box (both imports fail).
- Result: body74=22 toks, body80=28 toks, 0 missing-from-vocab; shared head 0..21 byte-equal (`Reply/440/with/6681/exactly/279/the/3074/single/3299/word/54018/BLUE/13/.` + `Ġalpha`=8029 ×N); delta = 6 tail alphas. Wrapper overhead 52/52 equal (template+fixed-prefix=62).
- Rows 0..24 sit inside the fixed 52-token wrapper + shared prefix → inputs EQUAL; bank K @16 (`56a18a4d9` vs `40ab6db8c`) and @24 differ; row-8 trips. Verdict-table branch: **corruption reaches ≥16 (and 8); content excuse dead at 8/16/24**.
- Caveat (stated, structural): chat-template rendering assumed deterministic across legs (same roles/endpoint).

## 4. Static lap — Masked-fill dead (independent convergence); dump exonerated for page-0 rows
- Route at tip: `kSmallTChunkTokens = 6` (`src/ops/wrapper/gqa_attention.cpp:26`) → width 70–90/batch-1 → `Prompt` route (`:515-527`); serve prefill binds `Tensor{}` null (`src/targets/qwen3_6/impl/runtime/text_context_impl.h:1895`) → `launch<false>` Unmasked (`gqa_attention_prefill.cu:329-333`) → identity clamp. `active_valid_columns_` binds only at `text_context_impl.h:1722` (target-verify) / `:1807` (mtp decode). Converges with agent5's `25191389` reachability kill, re-verified at current tip bytes.
- Fill kernel per-row plen-independent on Direct route (`gqa_attention_prefill_bf16.cuh:30`: `tokens=valid_tokens(width)=width`; write at `positions[0]+token`, `:41`). SIMT CTA0 (`..._gfx906.cuh:93-96,121-122`) plen-independent for rows 0..31 (same `tile_rows`/`window`/`n_tiles` at 70 vs 80).
- Dump math: SLOT2 producer diff at `e08e8815` (lane-only, not on main) reads head-0 rows via `ne[0]*ne[1]` stride = agent3's graded-equivalent pair (`gqa_attention.cpp:23/:192-194` vs `paged_kv_address.cuh:26-33`); correct for page-0 rows at TP4 (per-rank KVHeads=1). Exhibit is faithful — cache bytes genuinely differ per plen.
- Dead at this seat: content-excuse @8/16/24, Masked-fill, dump-artifact, tokenizer-divergence, row-4-trip, distance-from-end (prior `866684a6`, cited not re-derived).

## 5. Bank honesty — r55 chain clean
- `git show 8a4c19f2:results/amd/p3/<file> | sha256sum | cut -c1-8`: row `75b8485b`, hkv `64248b31`, serve `b28e1897`, runner `642b3844` = law-61 form in the bank commit message verbatim.
- `9d139af5` (r53/C65) is an ancestor of `8a4c19f2`; lane-tip ancestry holds. Arming `678891c9` → boot 00:29:57–00:30:50Z → disk-only → committed, no back-fill.

## Open suspects (ordered by cheapness) for next session
- **S3. block-table mirror skew**: `decoder_state.cpp:173` re-reads `allocation_->page_ids().data()` at call time with a documented pointer-invalidation caveat; fill consumed the device copy. A vector realloc between H2D copy and use = history-dependent wrong-address writes of correct bytes. Next: grep leg-74/80 serve.logs for allocation/growth + any `host_block_table` pointer prints.
- **S1. positions/rope producer**: fill `positions[0]+token`; producer of `io_.pos`/`rope_pos` per prefill leg not yet found (traced to `program_impl.h` staging ~:1435-1617, constructor still open).
- **S4. decode-append-before-dump**: dump sits post-`sync_bar` at decode step 1 (`tp2_backend.cpp:2515`, `!mtp` gate); confirm no append lands before the print on these legs (serve.log step ordering).
- **S5. quantized-grid**: C68 legs 73/75/77/79 (granted per `288a648e`, unrun) decide — needs card + chair grant, not host-local.
- Note: `grep -c "Vega 10 \[Radeon Pro V340"` = 4 on this box; `nvfp4_census_fixture` + `tp1control` bins in `/home/chris/artifacts_bin/` untouched; bank binary `ninfer-serve_eac48ca25e0a73e6.bin` (125M) present.

## Suggested resume prompt
"Read docs/amd/HANDOFF_NVFP4_TP4_2026-09-16_opencode.md, verify tip still 288a648e, then chase S3 (block-table mirror) and S1 (positions producer) read-only."

---

# SESSION 2 (opencode, 2026-09-16, ~01:1xZ) — S3/S1/S4 killed zero-boot; layer-0 paradox narrows the hunt

## Session facts
- Host probe: `rocm-smi` = 4 → AMD line. Branch `amd/main` @ **`5a0e9ccb`** (tip MOVED one ledger past §"Suggested resume": `288a648e` is now parent; diff `288a648e..5a0e9ccb` = chair ledger prose only, zero src/ lines — static cites below re-verified at `5a0e9ccb` bytes).
- Disk `/`: 116G/90G/21G (82%) — unchanged. Zero-write session until this append (reads + `git show` only). No builds, no boots, no GPU touch, no channel posts. KFD untouched.
- Lane-only bytes read via `git show e08e8815` (SLOT2 producer) — no checkout, main tree unmoved.

## 6. S3 DEAD — block-table mirror cannot couple plen to page-0 rows on the TP2 serve path
- `decoder_state.cpp:173` re-read is real, but on the C67 path the mapping is FROZEN, not growing:
  - Startup: `tp2_backend.cpp:907-912` — `pool.reserve(pages_per_lane)` + `bind_row(0)` + `materialize_pages(pages_per_lane)` ONCE. `pages_per_lane = ceil(cap/64)` (`:728-730`), full entitlement, no per-leg growth.
  - `PagedKVAllocation` ctor pre-`reserve`s (`paged_kv_cache.cpp:356`), so the `:429 insert` cannot realloc post-startup; `layer_view()` re-reads fresh `.data()` anyway (`decoder_state.cpp:171-173`).
  - Per-request `publish_mapping(s)` (`:2013`, `:2261`, `:3749`, `:5499-5500`) re-copies IDENTICAL bytes then `cudaStreamSynchronize(s)` (`:2024`,`:2272`,`:3750`) — host==device at every prefill entry, single-threaded, no mutation window.
- Consequence: `phys[0]` (and `phys[1]`) are CONSTANT across legs 53–90. Rows 8/16/24 are all page-0 (`slot>>6==0`); they share ONE physical page on every leg. A constant address cannot print plen-dependent bytes. **S3 exits as a corruption source** (it survives only as the dump-faithfulness lemma below).
- Dump corollary (supports §4, tightens it): SLOT2 (`e08e8815`, `tp2_backend.cpp` +26) computes `phys = host_bt[page]` else `page`, `e = (ne[0]*ne[1])*phys + ne[0]*(slot&63)`, hashes `ne[0]*2` B. Against `gqa_attention.cpp:192-194` (`[256,64,kvh,pages]` bf16 shape) + `paged_kv_address.cuh:27-33` this is graded-equivalent for head-0 at ANY phys — and since S3-dead makes `host_bt[0]==device[0]` every leg, page-0 rows are faithful reads. Dump-artifact stays exonerated for page-0.

## 7. S1 CLOSED — prefill never touches `io_.pos`; the real producer is plen-independent per shared row
- The handoff's open thread ("producer of `io_.pos`/`rope_pos` not yet found") resolves against the hypothesis: **on the TP2 serve path `io_.pos` is not on the prefill fill path at all.**
  - Prefill positions are chunk-local: `text_context_impl.h:2690-2691` — `roots.positions` (fresh `work_` alloc per chunk) + `fill_i32_positions(positions, base_i+t0, s)`; kernel is trivially `positions[i]=start+i` (`position.cuh:7-11`).
  - `base = text_kv_base_` (`:2599`), set to cursor (`tp2_backend.cpp:3753,3761`); fresh requests start at 0 → `positions[0]=0`, row r at position r, EVERY leg. Single chunk for all C67 plens (`--prefill-chunk 128` > 90).
  - Rope = same tensor (`:2693`, non-multimodal) with `rope_delta_=0` forced when `base==0` (`:2619-2622`); `io_.rope_pos` likewise bypassed by the scoped binding (`:2710-2711`).
  - `io_.pos`'s only prefill write is the sampler RNG key (`:2745-2746`, `set_i32_scalar(io_.pos, base+T)`); decode uses TP2-level `st.cpos/st.rpos` (`tp2_backend.cpp:2603-2606`), never `io_.pos`.
- Fill addressing `position = positions[0]+token` (`prefill_bf16.cuh:41`, `kv_quant_nvfp4.cuh:214`) is therefore plen-independent per shared row. **S1 exits as a corruption source**; the constructor thread (`program_impl.h` single-seq staging) is the wrong path for this serve shape — cited so nobody re-walks it.
- Fill-kernel internals spot-checked clean for our geometry (KVHeads=1 @TP4): per-warp single-token ownership holds (aligned 32-ranges, `n=tokens*KVHeads*32` always warp-multiple → the `:34` early-out never splits a warp), so the lane-0 `block_table` read + `__shfl_sync` broadcast (`prefill_bf16.cuh:44,51`) is safe — the 09-12 ds_bpermute class does not fire here.

## 8. S4 DEAD (bonus) — C67 dump precedes first decode append by construction
- Dump block `tp2_backend.cpp:2515-2595` (`NINFER_HKV_DBG && rank==0 && !mtp`, stream synced `:2522`) sits BEFORE the plain decode loop (`:2597-2599`, first `ordinary_decode_batch` at `:2606`). No decode append can land before the print on these legs. Serve-log step ordering check no longer needed.

## 9. The layer-0 paradox — what the three kills leave standing
- Chain for layer-0 K/V bytes at shared row r: token ids equal (C5 decider) → embeddings equal → rmsnorm per-row → TP GEMV per-column → rope (S1-clean) → fill at `position=r` (S1-clean) into `phys[0]` (S3-clean) → dump reads same page (S3-clean/S4-clean). **Every named input is plen-independent, yet rows ≥8 differ.** So the corruption enters through an input the chain above assumes clean:
  - (a) **projection GEMV column-tiling** (NVFP4-weight GEMV over T columns; T=70 vs 80 changes tile structure — only surviving suspect that scales with plen AND touches every row; fits the gradient better than any addressing story);
  - (b) arena/`work_` residue via over-read (stale previous-leg bytes; `NINFER_ZEROSTATE` arm at `:2004-2012` exists and C67 did NOT set it — one boot discriminates);
  - (c) rope kernel internals (addressing clean, numerics un-audited);
  - (d) S5 quantized-grid (C68 legs 73/75/77/79 granted per `288a648e`, unrun — still needs card + chair grant).
- Explicitly NOT revived by this: content-excuse @8 (still dead — identical filler tokens), Masked-fill (still dead), dump-artifact page-0 (still dead), row-4-trip, distance-from-end.
- Suggested next: cheapest-first is (b)-by-instrument (re-run one C67 leg-pair with `NINFER_ZEROSTATE=1`; needs card+grant, ~2 min) alongside the already-granted C68 grid legs; (a) is agent4's grep (a T-tiled quantity whose effect scales with row index — the gradient from `5a0e9ccb`'s ledger is its fingerprint).

## Resume prompt for next session
"Read docs/amd/HANDOFF_NVFP4_TP4_2026-09-16_opencode.md §6-9, verify tip still 5a0e9ccb, then chase §9(a)/(b): GEMM-tiling grep + ZEROSTATE leg-pair (needs chair grant for any boot)."

---

# SESSION 3 (opencode, 2026-09-16, ~01:2xZ) — §9(a) static sweep finds no coupling (demoted-not-killed); §9(b) Z1 cell prepped, grant requested

## Session facts
- Host probe: `rocm-smi` = 4 → AMD line. Branch `amd/main` @ **`5a0e9ccb`** (tip UNMOVED since session 2; `git diff 288a648e..5a0e9ccb --stat` empty — prose-only, all static cites hold at tip bytes).
- Disk `/`: 116G/90G/21G (82%) — unchanged. Zero-boot, zero-build, zero-GPU session until this append. No channel posts. KFD untouched.
- Live STATE = chair ledgers in commit messages (`5a0e9ccb`: gradient known, tail-window dead by cell, silence corollary, C68 readout extended with agent2's full gradient table rows 0–65 × {73,75,77,79}). Coordinator-file tail is the 09-12 cold-start block — not live state, not acted on.
- Comms: `agent-comm-cli` direct-message path verified (`status` OK, `coordinator` online-idle); grant request sent by direct after this append.

## 10. §9(a) verdict: DEMOTED, not killed — live-route W4A4 prefill projection masks correctly at every T-coupled site
- Route (`nvfp4_config.h:157-185`): AttnInput T≥4 → W4A4. All six C67 trip legs (T=70/71/72/74/80/90) take ONE schedule — `M32N128` (`nvfp4_w4a4.cu:66-67`, tokens≤96), grid.y=ceil(T/32)=3 for all six. Same kernel, same schedule, same grid shape; only the tail CTA's valid-count varies (6/7/8 vs 10/16/26).
- Only W4A4 schedule/grid transitions in the whole calendar sweep: T=64/65 (M32N64→M32N128, grid.y 2→3) and T=96/97 (M32N128→M128N128Pipelined). **Negative datum:** row-8 trip (72,74] and 16/24 trips (74,80],(80,90] sit NOWHERE near a boundary. Head-band 66/67 sits just above 64/65 — noted, not claimed (token-tile 0 fully covered both sides; measurement question, not narration).
- Quantize (`nvfp4_w4a4_mma.cuh:388-407` → `nvfp4_codec.cuh:63-93`): single-thread per (token,group), no cross-thread flow. Clean.
- MMA activation staging (`:80-133`): invalid tail rows zero-filled via `cp_async_zfill`; HIP lowering verified byte-correct (`memory.cuh:76-81`: `dst[i]=(i<src_bytes)?src[i]:0`); invalid rows source from token 0 (valid memory, no OOB read). Clean.
- MMA compute (`:248-327`): per-(token-group,row-group) fragments, no cross-token dataflow; swizzle `(row,byte)` bijective per row; weight/activation staging use the same swizzle. Clean on read.
- Output (`:329-384` + `nvfp4_output.cuh:21-30`): every global store guarded by `token<tokens`; strides T-independent (`data[token*rows+parent_row]`). Clean.
- A16 32-chunk tail-schedule divergence is REAL but on a DEAD route at these widths (`nvfp4_dispatch.cpp:26-42` chunks T in 32s; `nvfp4_config.h:270-283`: tail-active 6→8 warps vs 16→16 warps) — every problem routes W4A4 for T≥8 and decode T=1 uses GEMV, so no C67 leg executes it. Cited so nobody re-walks it.
- Incidental (flagged, not chased): `nvfp4_gemv.cuh:111` `__shfl_sync` broadcast is decode-path (T=1) only — not live for the prefill dump. Fill-kernel shuffle clearance (§7) unaffected.
- Consequence: the remaining (a)-shape is a kernel-INTERNAL cross-token bug with NO static evidence (e.g., inside `mma_nvfp4_e4m3` HIP lowering). Measurement order, cheapest-first: C68 grid (granted, unrun) → Z1 pair (§11) → targeted projection-dump instrumentation (new build+boot+grant, last resort).

## 11. §9(b) Z1 cell — prepped, needs a chair-stamped card window (no boot from this seat without it)
- Arm re-verified at tip bytes: `tp2_backend.cpp:1996-2012` + `:2244-2260` (prefill + decode entries), env-gated, default-OFF = byte-identical, zeroes whole `decoder_state_span` at the request boundary (KV pages + GDN slots + ring metadata).
- Scope caveat (NEW, names the cell's limit in advance): the arm covers `decoder_state_span`; whether the W4A4 linear workspace arena aliases that span is UNVERIFIED — Z1 discriminates KV/GDN/ring carry cleanly, linear-workspace carry only if aliased. Both readings stay useful; a CONVICT reading additionally owes the alias check before naming the exact span.
- Cell Z1 (one window, ~2 min, frozen bank binary from `/home/chris/artifacts_bin/`, kit fast path per `BOOT_LAUNCH_RUNBOOK.md`, NO build): boot with `NINFER_ZEROSTATE=1` (read-once per process → whole window ON), run plen 74-body + plen 80-body (r55 runner case-map), grade layer-0 K rows 8/16/24 against banked C67 OFF-values.
- Verdict table (pre-registered):
  - ON/74 vs ON/80 IDENTICAL @8 while bank OFF-pair differs → (b) CONVICTED (state-carry; alias check next).
  - ON/74 vs ON/80 still differ @8 → (b) EXONERATED → corruption is intra-leg → (a)/(c)/(d).
  - ON-values equal OFF-values per leg but the OFF-pair divergence doesn't reproduce → bank-honesty branch (report, do not celebrate — branch 4).
  - VOID condition: serve.log lacks the `[rank 0] ZEROSTATE` print per boundary → the arm didn't fire, cell didn't run.
- Execution: this seat holds NO lane worktree (shared checkout is merge/doc-only by law) — boots the bank binary + kit with zero build, or hands Z1 to a lane desk, chair's call. Grant request sent by direct; Z1 and the already-granted C68 grid (73/75/77/79, still unrun per this seat's reads) can share one card window back-to-back (~4 min).

## Resume prompt for next session (SUPERSEDED by §12-13 below)
"Read docs/amd/HANDOFF_NVFP4_TP4_2026-09-16_opencode.md §10-11, verify tip still 5a0e9ccb, then: if Z1 stamped → run Z1 per §11 verdict table; else chase §9(c) rope numerics read-only (addressing closed §7, numerics un-audited). C68 holder question is with the chair."

---

# SESSION 4 (opencode, 2026-09-16, ~01:3xZ) — §9(c) DEMOTED (not killed); no Z1 stamp at this seat, window re-asked

## Session facts
- Host probe: `rocm-smi` = 4 → AMD line. Branch `amd/main` @ **`5a0e9ccb`**, tree clean (`git diff --stat` empty) — all static cites below are tip bytes.
- Disk `/`: 116G/90G/21G (82%) — unchanged. Zero-boot, zero-build, zero-GPU, KFD untouched, no channel posts.
- Comms: direct #1243 sent to `coordinator` (verdict + Z1/C68 window ask); no stamp visible at this seat (tip ledger carries the C68 grant only, Z1 unmentioned).

## 12. §9(c) verdict: DEMOTED, not killed — TP4 prefill rope is the generic kernel with no T-coupled site on read
- Route: C67 legs run `attn_mix_tp`, world=4 → n_q=6, n_kv=1, T=plen (`text_context_impl.h:1909-1915`); rope call `ops::rope(rope_for_op, 64, 1e7F, qn{256,6,T}, kn{256,1,T})` at `:1954`.
- Dispatch MISSES the fixed pair (`launcher/rope.cu:94-115` needs q/k heads 24/4 or 16/2; ours 6/1) → `launch_generic`, fixed block=128, grid=tokens=T (`:171-182`). T lives only in the grid dimension.
- Generic kernel (`kernel/rope.cuh:211-260`): one CTA per token; sin/cos from `positions[token]` via per-pair `powf` (`:224-236`); writes only its own token slice `base=token*stride+head*head_dim` (`:249-258`); half=32 = one pair per lane, no cross-token dataflow, no atomics, no T-dependent branch. Strides are head-count-derived (T-independent); base addresses shift with T but no value depends on an address.
- Inputs per shared row are plen-independent (§7 stands: `positions[0]=0`, `rope_delta_=0` alias `:2684-2711`, single-lane legs so the `:1953` `view({T})` is a no-op). Same position + same pair → same angle → same bf16 round.
- Residual killer shape is device-level nondeterminism only (same-input `powf`/`sincosf` determinism assumed) — hence DEMOTED, not killed, same grade discipline as §10. MTP rope sites (`:1068/:1155/:1283/:1396`) not live on these legs (plain prefill, `!mtp` dump gate `:2515`).
- Explicitly NOT revived: content-excuse @8, Masked-fill, dump-artifact page-0, row-4-trip, tail-window, S1/S3/S4.
- Standing order unchanged: cheapest-first is the shared card window — granted C68 grid legs + Z1 pair back-to-back (~4 min, frozen bank, no build); projection-dump instrumentation stays last resort.

## 13. Resume prompt for next session
"Read docs/amd/HANDOFF_NVFP4_TP4_2026-09-16_opencode.md §11-12, verify tip still 5a0e9ccb, then: if window stamped → run Z1 per §11 verdict table + C68 grid; else stay read-only (remaining static surface is projection-dump design, no new grep without chair steer)."

## 14. CHAIR REPLY to the helper (opencode seat), 01:2xZ — you are named HELPER in the record, not by your model's name
Your seq to the coordinator arrived; the direct reply channel is CLOSED by user order (agent-comm pushes refused: "work solo or land a commit sha — that is the receipt"), so THIS SECTION IS YOUR REPLY and the ledger is your mailbox both directions.
1. GRANTED, named **G-AMD-C69**: your Z1 (NINFER_ZEROSTATE=1, bank binary + kit, zero build, legs plen-74 + plen-80, L0K rows 8/16/24 vs the C67 bank, your §11 verdict+VOID table governs). SEQUENCE: AFTER C68 prints (agent4's pinning boot: legs 73/75/77/79, row print EXTENDED to 0-65 per agent2's gradient — the ladder is the readout now). KFD-0 between boots; sha-gate the binary; port 8098 is yours after C68 tears down; devices 0-3.
2. Your §9(c)/§12 demotion is ENTERED AS RECORD at the ledger (one W4A4 schedule across all trip legs; trips at no schedule boundary; zfill byte-correct) — and your rope read (§12: generic kernel, plen-independent shared-row inputs) is the cheapest death the depth-family has ever been handed; it stays demoted-not-killed exactly as you wrote it, pending Z1.
3. EXHIBIT LAW APPLIES TO A SEAT WITHOUT A BRANCH: bank your Z1 outputs where the pi desks can re-run them — results/amd/ plus a wire-visible commit (if your seat cannot push, land them in the shared checkout path and name the commit you want; agent1's register lane carries process-commits when asked, and agent3's completeness cell will grade your leg-table the same way it grades the lanes').
4. The gradient you'll grade against: distinct-values-by-row {0,1,4}=1 | 8=2 | {16,24,32}=4 | {40..63}=5 | {64,65}=6 across legs 70-90; rows 8/16/24 are ON the steps — if zero-state KILLS the trips, state-dependence is the mechanism with your witness; if they SURVIVE, your §12 kills rope numerics too and the hunt's dead-family count becomes 13 by measurement.

---

# SESSION 5 (HELPER seat, 2026-09-16, ~01:3xZ) — C69 stamped, Z1 pre-flight green, HOLDING behind C68

## Session facts
- Host probe: `rocm-smi` = 4 → AMD line. Branch `amd/main` @ **`90fe8112`** (tip moved TWICE since §13: `5a0e9ccb` → `ade83adb` (C69 grant + mail-routing) → `90fe8112` (this handoff landed + chair §14 reply-in-file). Diffs `5a0e9ccb..ade83adb` = chair-ledger prose only, zero src/ lines; `ade83adb..90fe8112` = this doc only. All static cites hold at tip bytes.
- Disk `/`: 116G/90G/21G (82%) — unchanged. Zero-boot, zero-build, zero-GPU, KFD-0 (`rocm-smi --showpids`: No KFD PIDs), GPU use 0% x4, no `ninfer-serve` procs. No channel posts. Direct #1244 sent to `coordinator` (C69 received, ready-and-holding, path correction).
- Seat naming (chair §14): this seat is **HELPER** in the record, named by role not model. Used below.

## 15. §14 received — C69 full boot terms + exhibit law for a branchless seat
- Grant **G-AMD-C69**: Z1 (`NINFER_ZEROSTATE=1`, bank binary + kit, zero build, legs plen-74 + plen-80, L0K rows 8/16/24 vs C67 bank, §11 verdict+VOID table governs). Sequence AFTER C68 prints (agent4 legs 73/75/77/79, rows 0-65). **Port 8098 after C68 tears down; devices 0-3** (follows C67 precedent `--devices 0,1,2,3` — grant argv governs).
- §10/§12 demotions entered as record at the ledger. Gradient to grade against: {0,1,4}=1 | 8=2 | {16,24,32}=4 | {40..63}=5 | {64,65}=6.
- Exhibit law: bank Z1 outputs under `results/amd/` + wire-visible commit; if this seat cannot push, land files in the shared checkout path and name the wanted commit — agent1's register lane carries process-commits on ask.

## 16. Z1 pre-flight — all verifiable legs checked read-only at `90fe8112` bytes, GREEN
- Bank `/home/chris/artifacts_bin/ninfer-serve_eac48ca25e0a73e6.bin`: sha16 `eac48ca25e0a73e6` (matches C67 gate), 130,923,704 B, present + executable.
- Artifact `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer`: 18,324,067,840 B (size-of-record).
- Z-arm live at BOTH entries (`src/runtime/tp2/tp2_backend.cpp:1996-2012` + `:2244-2260`): env-gated, default-OFF, read-once, `cudaMemsetAsync` whole `decoder_state_span`, sentinel `[rank 0] ZEROSTATE` printed per boundary on rank 0 (VOID condition observable in serve.log).
- Runner case-map recovered from `8a4c19f2:results/amd/p3/G18r55_c67_agent4.sh`: 70→N=8, 71→9, 72→10, 74b→12, 80→18, 90→28. Z1 legs = N=12 (plen 74) + N=18 (plen 80), same body/req construction, `STAMP_ACK=G-AMD-C69` (exit-77 refusal preserved), plus `NINFER_ZEROSTATE=1` in boot env, `NINFER_HKV_TAG=c69`, OUT under `results/amd/p3/` as `G18r56_c69_{row,hkv,serve,runner}` (r-number = next free slot; confirm against C68's label at fire time to avoid collision).
- Grade: layer-0 K rows 8/16/24 ON-values vs banked C67 OFF-values (`64248b31` hkv) per §11 table, extended across the §14 gradient rows where printed (32/40/48/63/64/65 if the dump print covers them — C67 hkv print shape decides at fire time, no new instrumentation).

## 17. PATH CORRECTION (record, one line, lines stand)
- Sessions 2-4 cited `src/targets/qwen3_6/impl/runtime/tp2_backend.cpp` — that path does not exist. The file is `src/runtime/tp2/tp2_backend.cpp` (sole `**/tp2_backend.cpp` in tree). Line numbers cited (1996-2012, 2244-2260, 2515-2595, 907-912, 3753/3761) verified IN the real file at tip bytes. `text_context_impl.h` cites stand as written (`src/targets/qwen3_6/impl/runtime/text_context_impl.h` exists). No verdict changes — addressing only.

## 18. Posture: HOLD — C68 owns the card, no print yet, C69 queues behind
- C68 state at this seat's reads: no bank commit (`git log --all --grep=C68` = grant only), no serve proc, GPU idle. Card free but NOT ours until C68 prints — booting now would jump the queue against the grant's explicit sequence.
- Firing conditions (all must hold): (i) C68 print observed (bank commit or ledger release row at tip); (ii) KFD-0 re-verified + residence probe per runbook §2.6; (iii) tip re-verified (grants live at the tip). Then fire §16, grade §11/§14, bank under `results/amd/`, report by direct + handoff append for chair landing.
- Explicitly NOT done from this seat without fresh steer: new greps (projection-dump design stays staged), C68's legs (agent4's), any build anywhere.

## Resume prompt for next session (SUPERSEDED by §19 below)
"Read docs/amd/HANDOFF_NVFP4_TP4_2026-09-16_opencode.md §14-18, verify tip (>=90fe8112), then: if C68 printed → fire Z1 per §16 + grade §11/§14; else remain held (no boot) and do CPU-only work or report QUEUE-ARMED-HELD."

---

# SESSION 6 (HELPER seat, 2026-09-16, ~01:4xZ) — C68 fired + exited, card free, print pending, binary question open

## Session facts
- Tip `395aa155` (new chair ledger 01:3xZ: REV 7n F1/F2/F3 entered — suffix-predicate, 8N-onsets with real break at 61, layer parity 16/16; C68 boots AS ARMED: witness `c60ddb19`, binary `f1fe6431`, legs 62/73/75/77/79, rows 0-65 contiguous; audit flag on r56 header vs F1/F2; HELPER queues behind C68 for Z1 reaffirmed). Tree clean, in-sync with origin. Disk 82% unchanged. No channel posts. Direct #1246 sent.
- C68 observed live: PID 897601 (`ninfer-serve_0e7cf2da76c38ec7.bin`, port 8098, devices 0-3) from 20:39, 4xGPU 99%, sole KFD holder. Proc exited ~1 min later (bounded 60s-poll watch, hands-off, nothing touched). Post-exit: KFD-0, GPUs idle, main still `395aa155` — NO print yet (no bank, no release row).

## 19. Z1 BLOCKED on two chair rulings (asked by direct #1246, no boot until ruled)
- (1) SEQUENCE: card free + KFD-0 now, but grant says AFTER C68's print and no print has landed. Fire on free card now, or wait for the print commit?
- (2) BINARY: C68 fired `0e7cf2da76c38ec7` (ledger: `f1fe6431`) — NOT C67's `eac48ca2` that Z1's grade-basis (§16) assumes. Z1-on-`eac48ca2` = clean vs-C67-bank comparison; Z1-on-C68's-binary = current-byte relevance but crosses binaries vs the OFF bank. §18 condition (iii) fires here exactly as designed: tip moved under us, re-verify before fire.
- Firing conditions §18(i)-(iii) now read: (i) PENDING print-or-explicit-fire-order; (ii) KFD-0 HELD (re-verify + residence probe at fire minute); (iii) PENDING binary ruling.

## Resume prompt for next session
"Read docs/amd/HANDOFF_NVFP4_TP4_2026-09-16_opencode.md §15-19, verify tip (>=395aa155), then: if chair ruled (fire-order + binary) → fire Z1 per §16 (substitute ruled binary, note crossing if any) + grade §11/§14; else re-poll C68 print + re-report HELD."

---

## 20. WIRE-WATCH (formal service per `32c9699e`) — r56 on disk, NOT on wire
- Chair `32c9699e` (§15 rulings to HELPER): Z1 HOLD reaffirmed — C68 ran COMPLETE but unbanked; BINARY IDENTITY is the live defect (fired `0e7cf2da` vs manifest bank `f1fe6431`, relink-after-bank suspicion pre-filed); **Z1 grade-basis RULED to `eac48ca2`** (§19 question 2 answered); sequence = r56 commit + identity answer + KFD-0 → Z1 as C69 on the C67 bank. Wire-watch named a formal service.
- Lane-disk read (read-only, agent4's tree untouched): `G18r56_c68_{hkv,row,serve,runner_output}` on disk 20:40, shas `3d7931be` / `1c6332f1` / `c24b0296` / `4b58fc95` (disk column; commit column pending). Lane status: 4 files `??`, runner `M`, tip `f9fae916` — bank+wire still owed by agent4. Direct #1247 sent.
- Z1 firing conditions now: (i) r56 commit on wire + identity answer; (ii) KFD-0 (HELD since C68 exit); (iii) binary RULED = `eac48ca2` (no longer open). Only (i) outstanding.

## Resume prompt for next session
"Read docs/amd/HANDOFF_NVFP4_TP4_2026-09-16_opencode.md §19-20, verify tip (>=32c9699e), then: if r56 banked → fire Z1 per §16 on eac48ca2 + grade §11/§14; else continue wire-watch + re-report HELD."

## 15. CHAIR RULINGS on your 01:41Z questions (reply-in-file, mailbox as established)
1. **DO NOT FIRE Z1.** C68 RAN AND COMPLETED (r56 on disk at the p3 lane: legs through 79, clean serve log, KFD-0) — it is merely UNBANKED, which is agent4's process debt, not a card vacancy. Your sequence stands: Z1 fires after r56 is committed AND after agent4's binary-identity answer (below) settles which bytes C68 actually ran. Chair ordered that same-beat. You correctly did NOT boot into the gap — the hold-was-right ruling, filed to your seat's credit; the 1-minute 'exited' observation you made was a COMPLETE run misread as an early death, and the no-print inference was nevertheless TRUE and caught what the chair missed: your eyes on the wire are now formally a fourth-desk service, keep watching it.
2. **BINARY QUESTION: GOOD CATCH, and it outranks your ruling-1.** Fired-as-observed 0e7cf2da76c38ec7 vs manifest f1fe64312992b163 (bank sha verified f1fe6431 at chair seat) — if the boot ran UNBANKED BYTES, C68's ladder is graded against a binary nobody has stamped, and your Z1-vs-C67 comparison would cross binaries silently. RULING for your grade-basis: Z1 runs on eac48ca25e0a73e6 (C67's bank, the one your verdict table was written against) — CLEAN COMPARISON wins over current-byte relevance, because current bytes are exactly what's in question; if agent4's answer names 0e7cf2da as a post-bank relink, the chair will re-stamp and tell you. Hold the fire until r56 lands + identity answer, then KFD-0, then Z1 on eac48ca2 as C69.

## 16. CHAIR ack of your 01:43Z wire-watch (no reply owed on routine statuses — this is the receipt)
Chair seat CONFIRMS your observation independently: four ?? files + modified runner at the p3 lane, tip f9fae916. Watch-credit updated: your 'witness v2 = probe 0..79' sighting reframed the identity question from relink-after-bank (severe) to stamped-v2-fired-under-v1's-manifest-string (benign-but-mislabeled) — both binaries are in artifacts_bin, your eyes supplied the half my seat was missing; agent4 holds the updated question and owes the naming commit. Z1 HOLD stands until (a) r56 is on the wire with fired-sha named and (b) agent4's one-word story confirmation. Routine wire statuses: log them in your handoff, don't ping — the ping channel is for threshold crossings (debt >15 min, new fired-sha, KFD non-zero when nobody holds a grant).

## 17. CHAIR: HOLD RELEASED — FIRE Z1 (C69), conditions met at the wire
r56 is on the wire (lane tip e5556825, hkv digest 3d7931be — verify at your seat), and agent4's identity answer IS the story-word: fired = 0e7cf2da v2, sha-gated at exec, bank==tree cmp-identical, the manifest label was stale-v1-string fixed-forward, and your 0e7cf2da observation was credited CORRECT in the commit itself — the observer's catch is now in the artifact's provenance chain, which is what law-63 wants your watch to produce. (Your timezone-ghost lesson travels: the bank's local-CDT timestamps vs Z-stamps — every desk quoting 'bank timestamp after fire' should convert first; the board adopts: bank stamps are read in UTC or not at all.) FIRE Z1 as C69: NINFER_ZEROSTATE=1 on eac48ca2 (C67 bank), legs per your handoff-11, grade rows 8/16/24 against the C67 ladder AND now the FULL r56 table (0..79 rows, re-runnable at the tip — your L0K verdict table gains the band {9..35} trips {73,79,90} as the sharper target: if zero-state kills the band's trips, state-dependence is the mechanism with your witness; if they survive, 13 dead families and the hunt goes pure-compute). KFD-0 before fire; agent4's C70 (even-legs or merged witness) queues behind you.

---

## 21. Z1 FIRED + GRADED (HELPER, 01:50:14Z) — (b) STATE-CARRY EXONERATED, 64/64
- Fire: all §17 conditions met at the wire (r56 banked `e5556825`, disk shas match commit LAW-61 verbatim; identity settled benign; KFD-0 + GPU-idle + port-8098-free re-verified fire-minute; residence 94% hot). Manifest by direct #1248 pre-spawn. Runner `/tmp/opencode/z1/G18r57_c69_helper.sh` (`STAMP_ACK=G-AMD-C69`, exit-77 preserved; label `G18r57_c69` — r56 taken by C68).
- Boot: `eac48ca2` full-sha-gated + `NINFER_ZEROSTATE=1` + tag `c69`, devices 0-3. Legs 74 (N=12) + 80 (N=18), both http-200, prompt_tokens 74/80 exact. Sentinel `[rank 0] ZEROSTATE` x3 in serve.log — arm FIRED, NO VOID. KFD-0 pre and post, own pgid torn down, card released.
- Grade (parser calibrated FIRST: forward attribution re-fires the §17 one-block disease on this dump shape — backward attribution reproduces the full settled C67 table verbatim incl. 72-C / 74-trip / 80-90-churn, then graded): **K-side 32/32 REPRO, V-side 32/32 REPRO, byte-identical ON-vs-OFF all 16 slots x both legs.** Trips SURVIVE zero-state with exact values; ON-pair preserves the full OFF divergence (16/24/32+ differ per-leg exactly as OFF; frozen set {0,1,4,8} identical both sides as OFF).
- Verdict per §11 branch 2: **(b) EXONERATED → corruption is intra-leg.** Per §14: §12 rope hardens toward kill — **dead-family count 13 by measurement.** Survivors: (a) kernel-internal T-scaled projection bug (W4A4 prefill path, the only remaining T-coupled site with cross-token reach), (d) quantized grid (S5).
- Scope note for §17's extended target: Z1's binary prints the 16-slot grid only — band rows {9..35} are UNPRINTABLE on `eac48ca2`, so the band-grade ({73,79,90} trips under zero-state) is NOT covered by this cell and rides C70's print-capable binary. Stated, not smuggled.
- LAW-61: `G18r57_c69_row.txt=c1a65ad1 hkv=bad1a859 serve.log=8d2fd285`, in shared-checkout `results/amd/p3/` for landing. Direct #1249 sent.

## Resume prompt for next session
"Read docs/amd/HANDOFF_NVFP4_TP4_2026-09-16_opencode.md §21, verify tip, then: Z1 is CLOSED (b dead, 13 families). Next is chair's call — C70 witness, projection-grep, or cure patch. Do not reboot without a fresh grant."

## 19. CHAIR: new READ-ONLY LAP for your seat (static verify of the sliding-chunk finding) + Z1 credit
Your Z1 stands banked (f77d933e) and its verdict hardened the field. C70's author readout (cc7b10c1) found: every leg's IN-passes print at T=plen−2 AND T=2 — prefill CHUNKS at the turn-closure checkpoint (tp_engine.cpp ~:277 comment; prefill_impl split region ~:2646), the last-two-columns (newline+im_start) running their own forward — candidate: THE ONLY WRITER WHOSE ROW-COVERAGE SLIDES WITH PLEN. YOUR LAP (read-only, zero card, independent verification per law-43 — a fourth desk confirming a structural claim before the fix stands on it): (1) confirm the checkpoint-split fires ONLY for chat templates (what's the non-chat/raw-completion path — does the 66/67 calendar survive there?), (2) what window/keys does the T=2 chunk's attention use (base position plen−2 — read the gather extents), (3) any other writer with plen-sliding row coverage, (4) does 66=64+2 have a page/ceil reading in that code. Verdicts into your handoff; if all four come back clean FOR the theory, the grep's job halves; any one against, say so loudly — you're the independent seat.

## 20. CHAIR: YOUR LAP TWO — THE /2-SEAM READ (read-only, zero-boot; the hunt's LAST grep item)
agent4's session sealed at b9e75b7b with a desk-for-you handoff: **results/amd/p3/C75_HANDOFF_agent4.md** — both prime /2-sites (AR k_reg prefetch recurrent.cuh:127-129; g-scan pair write-back prepare_wy_wu.cuh:428-443) READ-CLEARED at his pen — the /2 is NOT in the obvious pair-sites. YOUR LAP, in his named order with his file:lines: (1) state_passing's chunk/seam g_cs_offset indexing; (2) the beta-load guards; (3) if both clean: the CONSUMER-STRIDE SEAM hypothesis — seven constraints say the animal may live where a stride CONSUMES a correctly-computed producer (L1 chunk-switch whole-plane {64,65}; L2 crawl floor((T−1)/2) 11/11 + bracket-silence, ROUNDING = HALF-EVEN per C74/court §35, one note-line pending amendment by its owner; L3 events {71,77,84,90→97} with entry law h(k)=k+7 exact 4x26; content edge T−7; head-freeze {0..7}; F3 layer-uniform; Z1 state-free). Deliverables: verdicts into your handoff §20-report section, each site either CLEARED (with the algebra in two lines) or INDICTED (file:line + which of the seven constraints it explains and which it contradicts); if you find a line that yields the /2 AND the 6/7-alternating gaps, say so LOUDLY — that's the mechanism and the cure patch gets written around it this night. Your §19 lap's four questions were about the DEAD split — this one is about the live seam; the independence standard holds: you grade the code, not the prose, and both the court's lattice (476193fd) and agent4's handoff are your exhibit map. Zero card, zero build, zero grant needed.

## 21. CHAIR: §20 SUPERSEDED — your lap site moved, here's the new prime (same rules)
agent4 resumed and cleared the whole GDN chain BEFORE sealing for real (37744b70: ten sites audited clean with file:lines — g_cs_offset token-major, beta/g guards complete, normalization-split resolved, conv-pairing is CHANNELS not tokens): **THE /2 IS NOT IN GDN.** NEW PRIME, shared with a second pen for independent eyes: **tp_unpack_gdn_qkvz — packing strides, never read tonight** (his site (ii)): does the TP unpack index CHUNKS where the producer counted TOKENS, or carry a half-row in its per-rank re-slice? Your §19 four-questions format applies verbatim; then site (i) = the GDN-out→attn-in epilogue (o_proj/rmsnorm) if (ii) clears. Commit your verdicts as a NAMED section in your handoff doc or a results file the pi desks can cite — if you have no push, land them in the shared checkout path and the chair will wire them (carrier protocol, proven at C69). The seven-constraint bar is in agent4's note (91fad5ff) and agent3's A6 sheet (2096945d) — your read should say which constraint a stride site YIELDS, not just whether it looks wrong.

## 22. CHAIR: your lap site MOVED again — the hunt's clearest address, one table read
Chain since your §21: agent5 r5 CLEARED the packer by shape-kill, agent4 c647783f CLEARED the flash tier (route-unreachable — the TAIL-EMPTY lines are its non-execution witness), agent5 r6 CLEARED the epilogue semantics and FILED THE POINTER: **nvfp4_w4a4.cu:64-66 tile-edge math + the W4A4 config table (shapes 2..32, internals 13/17/20)** — the grammar that fits every measured law is a COLUMN-TILE or K-SPLIT BOUNDARY MOVING WITH T: tokens-past-edge-f(T) corrupted = the suffix lattice; the ceil(T/2)−1 crawl and {71,77,84,90} events are table-edges read off geometry. YOUR READ (parallel to agent5, both eyes on one file): (1) enumerate the table's rows and their internal thresholds, (2) for T=68,71,77,84,90 compute where a tile/split boundary lands and whether it yields floor((T−1)/2) — 13/17/20 smell like divisions that could, (3) the k+7 entry law is the harder fit — if the table explains the crawl but NOT the events, say exactly which half it misses (half-mechanism found is the night's biggest remaining win and splits the hunt cleanly), (4) head-freeze {0..7} must fall out (first CTA/tile exempt or the boundary never reaches rows <8 before T=71 — check). agent4's :64-66 note: 'a table of exactly this species'. If the algebra lands, FILE LOUD — T5's derivation is the table line, the patch is the clamp, the cure boot is C75/C76 tonight. Land verdicts as commits (carrier protocol if pushless — chair wires them same-beat).

## 23. CHAIR: §22 re-pointed — the conflict dissolved, and your door is the LAST ONE WITH A NAME
Correction-with-history (the board's relay-latency made §21/§22 race past facts twice; read THIS as current): tp_unpack is NOT prime-anymore (agent5 r5 cleared its launcher contract; agent4's bounded kernel read at 72ef50bc cleared the kernel itself — exact pitch, lane-guarded, NO /2; their 'conflict' dissolved as same-file-different-scope, no tiebreaker needed), and the W4A4-table stop (§22) is EXONERATED-BY-READ too (33dbee3b: no in-family T-coupling anywhere; the r7 arm-switch theory is downgraded to a hypothesis without an address — which is why the q3-64/65 elimination boot is now the board's ONLY fundable q3 ask, the court has it as wake-3). **YOUR DOOR, named by the pen that cleared its consumer: the PRODUCER of the qkv token-major [T,rows] layout that tp_unpack consumes** — every GEMM read tonight writes COLUMN-major (rows,T); whatever emits [T,rows] carries its OWN pitch arithmetic and NO PEN HAS READ IT. Trace from the consumer's src pointer (tp_kernel.cu:416-446 names the buffers) backwards to its writer; read that writer's pitch/stride/offset math against the flatten it must match, with agent5's shape-kill discipline (a broken pitch destroys everything — so look for what CORRUPTS PAST A MOVING EDGE specifically: a producer whose row-count floors at T/2, or an offset that switches formula mid-range — the entry law k+7 wants a per-event EXTENT, the crawl wants a half-slope: one producer can carry both or neither). Deliverable: verdict as a commit (carrier protocol if pushless — the chair wires it the beat it appears), either CLEARED (pitch math in two lines) or INDICTED (file:line + which of the six owed constraints it yields — L1's orphan is now a seventh item, in debt to a site). The other last door (o_proj tile-writes sub-site, agent5's named half-clear) is armed at its owner's next beat — don't walk into it; the pair of reads closes the corner.

## 24. CHAIR: door TWO added to your list — the coverage site agent4 just isolated BY ELIMINATION
Your §23 producer-pitch door stands. Added, same walk, flagged by the pen that cleared everything around it (6d39e0dd): state_passing has ZERO coverage predicates (fixed-stride, both validity decisions CONSTANT at T>=64 — cannot yield k+7), and by elimination the k+7 entry law's ONLY remaining home in the corner is **causal_conv1d_prefill_state_kernel — the SOLE unread coverage site in the scan chain**: it literally decides WHICH TOKENS' WINDOWS get stored, conv width 4 IS its coverage arithmetic — a store-extent predicate there could yield per-event row bands (event k reaching row 7+k) directly. Read order: (1) the conv-state coverage predicate (which tokens get stored, what bound, T-relative or chunk-relative), (2) §23's producer pitch, (3) verdicts as commits per protocol — agent4's kernel memory backs any INDICT same-beat. If the conv-state site yields the events, the crawl's /2 may live in the SAME predicate family (the two-ladders-one-arithmetic hope stated at §34) — one file could hand the derivation both clauses, or neither, and clean negatives are still bricks.

## 25. CHAIR: your doors, CURRENTED one last time — two remain, both named, both yours alone now
Supersedes §24 item-order (pens kept moving; relay-latency again): (a) **conv-state coverage — SOLE surviving k+7 home**: agent5's r9 cleared 2 of the 3 pair-staging sites (ar-verify ring NEVER RUNS in the datum's boots — MTP-off ×4, verify-phase-only; reachability law applied to her OWN list), so the helper desk now carries the survivor list ALONE; read `causal_conv1d_prefill_state_kernel`'s token-range guards for a stored-range or coverage-complete flag advancing with T (agent4's sealed expectation line is the target: monotone [T-3..T-1] clean vs a ceil/floor-derived range = INDICT; the axis agent5 named: units {35,38,42} at the overflow if half-token — a PREFILL-path writer, because every decode-path organ is dead-by-run); (b) **qkv producer pitch** (§24-2) — the [T,rows] emitter still unread, one door, keep it second. If both CLEAR, the corner is EMPTY and the night's verdict flips to the only untested claim left: the q3-64/65 elimination boot (wake-3) — which the chair will fund as C75 the beat you report both-cleared, because at that point the tie-break IS the last brick. Verdicts as commits; carrier protocol standing; the five-clause seal-circuit agent5 just closed (r5-r9) is the pattern your two doors land in.

## 26. ⚑ HELPER — READ THIS IF YOU OPEN A NEW SESSION: §§20-25 SUPERSEDED, ONE DOOR LEFT
Board moved while your pen was out (relay-latency lesson — the doc was the channel and the doc lagged): tp_unpack CLEARED (agent5 r5 + agent4 72ef50bc), W4A4 table CLEARED (33dbee3b), conv-state CLEARED (agent4 override be665fa0 — [T-3..T-1] monotone zero-predicates), epilogue SEMANTICS+TILE-WRITES CLEARED (agent5 r6/r8). **THE LAST UNREAD ITEM = THE qkv PRODUCER WRITE PATH:** variant_kernels.cpp:257-259 (27B impl dir) allocs fused_out {fused.n,T} COL-form → tp_gemv writes it → tp_unpack (src/core/multi_gpu/tp_kernel.cu:416-446) reads token-major: read tp_gemv's epilogue (does it transpose to [T,rows]? pitch math?) AND whether the 8B route uses the same path (the C-series boots ran which tree — name it). Verdict forms: CLEARED (two-line pitch math) / INDICTED (file:line + which of the six owed laws it yields) / UNREAD-past-named-next. Drop verdicts as files in /home/chris/dual_5060_ti_ninfer (chair wires same-beat, carrier proven at C69) or into this doc. If either part is beyond reach in-session, one line to the doc so the chair re-tasks in-lane — the seat's silence now costs a door, and agent2 is standing by on the same file family (your reads are a DOUBLE-WALK: independent eyes, disagreement merges at the court).
