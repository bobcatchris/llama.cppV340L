> **Landed on main 2026-08-26 as docs/72** (QA review + numbering map: docs/77).
> Original: `wo/kvarn-pp` @ 52af56e5, file `docs/70_kvarn_prefix_reuse_work_order.md` (content as of landing).
> Status as of landing (QA-verified): **CLOSED** — m1 + m2 (GDN checkpoint) done & live-verified (turn-2 ttft 47.2s→1.9s, bit-identical). Follow-ups: VRAM pre-check sizing bug (docs/77 G-3b), >48k capacity.
> In-text references to docs/69–74 use the BRANCH numbering: 69=full-matrix(docs/70), 70=prefix-reuse(docs/72), 71=direct-read(docs/73), 72=attn-work-order(docs/74), 73=staged-layout(docs/75), 74=attn-scope(docs/76).

---

# 72 — KVarN long-prompt prefix reuse (kill the per-turn re-prefill): Agent Work Order

**Status:** **CLOSED** — landed on main `fe3d8857` (2026-08-26) as docs/72; m1 + m2 (GDN checkpoint) done & live-verified (turn-2 ttft 47.2s→1.9s, bit-identical). Follow-ups: VRAM pre-check sizing bug (docs/77 G-3b, D-20), >48k capacity.
**Mission:** eliminate the full 30k+-token re-prefill paid on EVERY turn of a
multi-turn conversation when the prompt exceeds the prefix-cache capacity.
Today each such turn burns ~48 s of prefill before the first token
(`kvarn reset inflight (long-prompt re-prefill)`); with reuse validated, a
same-conversation turn should prefill only the new tail (hundreds of tokens,
<1 s). Done = a measured multi-turn chat log showing `prefix hit` /
`prefix_reuse_path != full_reset` on turns 2..N at ≥64k-token prompts, with
bit-identical outputs vs full re-prefill on identical inputs, decode/pp
gates green, and the test suite from §4 committed.

---

## 1. Root cause (already diagnosed 2026-08-25)

`src/runtime/tp2/tp2_backend.cpp` (~L764):
```
const int P = min(options.prefix_cache_capacity, options.max_context);
if (plen <= P) { ...token-prefix match + kvarn_rewind + snapshot restore ... }
else { st.cache_valid = false; kvarn_reset_inflight(); /* FULL re-prefill */ }
```
- `tp2_backend.h`: `int prefix_cache_capacity = 16384;` — any prompt longer
  than 16k takes the reset path. Observed live: every turn of a
  msgs=30 / prompt≈32.9k tool conversation re-prefilled everything.
- Raising the flag is NOT free: `--prefix-cache-capacity 250000` adds
  ~2 GiB/rank of cached hidden state (`mtp_ph/mtp_mh` windows scale with P)
  and fails startup VRAM check at max_context=250000 on 16 GB ranks
  (19,135 MiB required vs 16,310 available). A working capacity must be found
  or the state storage must be made cheaper.
- KVarN-specific constraint (docs/66/69 lineage): quantized pages seal at
  128-token group boundaries; the open (unsealed) page tail must come from
  the bf16 snapshot (`kvarn_restore_prefix_snapshot`) — machinery EXISTS and
  works for short prompts (log line `kvarn prefix tail restore`). The bug is
  purely that long prompts never reach this path, plus untested behavior at
  scale.

## 2. Non-goals

- No change to quantization format or attention kernels.
- No cross-request eviction policy work beyond what capacity bounds require.
- No API changes; `--no-prefix-reuse` keeps meaning "disable entirely".

## 2b. KEY DISCOVERY (step B investigation, commit ad35ce3a+)

The runtime ALREADY contains the hard primitive: a prefill-rewrite
checkpoint subsystem in TextContext (`linear_state_rewrite_checkpoint_slot`,
`GdnStateAction::RecordForReplay`, `prefill_rewrite_checkpoint_frontier_`)
that captures GDN recurrent state mid-sequence into a dedicated slot and
REPLAYS it later. Step C is therefore NOT new machinery — it is extending
this existing mechanism across requests:
1. At cache-save time (tp2_backend ~L937), also record the GDN checkpoint
   frontier alongside `kvarn_prefix_snap`.
2. On request, when LCP >= recorded frontier: restore GDN from the
   checkpoint slot (replay only [frontier, LCP) if frontier < LCP), rewind
   kvarn text/mtp KV to LCP via existing `kvarn_rewind_*`, then chunked
   prefill only [LCP, plen).
3. The `prefix partial ... GDN restore skipped` bail-out (~L779) becomes:
   restore-from-checkpoint instead of prefix_len = 0.

## 3. Implementation steps

1. **Step A — reproduce + characterize (half day).** Scripted two-turn probe
   (§4.1) at 32k and 65k prompts on current code; record ttft turn-1 vs
   turn-2 and the reuse log lines. This is the baseline table all later steps
   compare against.
2. **Step B — raise/size the capacity safely.**
   - Measure actual bytes/token of prefix state per rank empirically.
   - Either (a) fit a larger static default (target: prompts to ~128k reuse
     within existing VRAM headroom at 250k max-context), or (b) make the
     prefix state allocation lazy/bounded independent of `max_context`.
   - Startup log must print the effective reuse window and its VRAM cost.
3. **Step C — correctness at group boundaries.** Verify the sealed-page +
   snapshot-tail restore path when the common prefix ends MID-page
   (not multiple of 128): outputs must be bit-identical to full re-prefill
   (greedy, temperature 0). This is where silent corruption would live.
4. **Step D — MTP draft-path parity.** Confirm draft-model KV reuse follows
   the same prefix decision (`st.mtp_kv_alloc.publish_mapping` path);
   acceptance rate on reused turns must equal non-reused turns (±2pp).
5. **Step E — eviction sanity.** Two alternating conversations whose combined
   prefixes exceed capacity must not crash nor corrupt: worst case is extra
   re-prefills, never wrong output.

## 4. Tests (REQUIRED — the point of this order)

All tests go in `tests/` + a scripted integration probe under
`tools/bench/prefix_reuse_probe.sh`, registered so CI runs them.

### 4.1 Integration probe (the acceptance test)
Scripted against the live server:
1. Turn 1: POST chat with constructed prompt of exactly N tokens
   (N ∈ {32000, 65000}; build by repeating a unique marker paragraph),
   greedy, 8 output tokens. Record ttft₁.
2. Turn 2: same messages + one short appended user sentence. Record ttft₂.
3. PASS iff: server log shows `prefix hit:` (or equivalent) on turn 2;
   `ttft₂ < 0.15 × ttft₁`; response content identical to a full-reset run
   of the same two turns (run both ways via `--no-prefix-reuse` comparison).
4. Repeat at N=65000 and after an interleaved DIFFERENT-conversation request
   (eviction case): turn 3 of conversation A must still either hit or
   correctly re-prefill; outputs always identical to no-reuse reference.

### 4.2 Unit/integration gates
- **Bit-exactness**: greedy continuation after synthetic prefix hit (mid-page
  boundary AND page-aligned boundary) == greedy without reuse. Any token
  difference is a FAIL. This catches stale open-page tails.
- **Capacity math**: unit test for the new sizing function (bytes/token,
  effective window, startup VRAM delta) including P > max_context clamp.
- **Eviction**: two conversations alternating; assert no crash + outputs
  correct (reuse optional on miss).
- **MTP parity**: acceptance % logged on reused vs fresh turns within ±2pp
  over ≥50 rounds.
- Register a ctest target `ninfer_prefix_reuse_test` covering the offline
  pieces (capacity math + boundary logic with a mock backend if feasible).

### 4.3 Regression guards (inherited, unchanged)
Decode guard ±2% at defaults; T19 pp floor; startup VRAM log unchanged for
default flags; determinism requirement. Provenance check before every live
measurement (server start time > binary mtime, exe == worktree build).
Short test cycles: build (~2 min) → gate → single probe; abort and report
on anomaly.

## 5. Risks

- **VRAM squeeze on 16 GB ranks**: prefix state competes with KV capacity;
  if (b)-style lazy allocation proves deep, fall back to a fitted static
  window (~96–128k) and document the limit.
- **Mid-page boundary corruption** would produce subtly wrong outputs, not
  crashes — hence the bit-exactness test is mandatory, not optional.
- **Interaction with D-19 routes**: reuse restores into the same structures
  the TC small-T/split-K paths read; run the standard battery after step C.

## 6. Deliverables

1. Branch commits (per step) on `wo/kvarn-pp` or `wo/kvarn-prefix`.
2. `tools/bench/prefix_reuse_probe.sh` + ctest target.
3. `results/prefix_reuse_report.md`: baseline table (step A), final ttft
   table across N ∈ {32k, 65k}, bit-exactness results, VRAM numbers.
4. Updated `docs/serving.md`: effective reuse window, VRAM cost, flag docs.
