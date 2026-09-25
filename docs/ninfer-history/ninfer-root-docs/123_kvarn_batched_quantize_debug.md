# 123 — KVarN batched-quantize: the SM-starvation 34× optimization that breaks the model

**Status: DEBUG IN PROGRESS. Coordinator (lead) + Agent 1 (math/code assist). 2026-08-31.**
**DO NOT REVERT — the goal is to MAKE THIS WORK, not run away to green. Rollback point is the tag
`kvarn-phase2-baseline` / commit `28318ab8` if truly needed.**

This doc gets Agent 1 up to speed on the current state, the exact bug, and the ONE unit of code to
focus on. Everything below is measured, not guessed.

---

## 1. TL;DR — the problem

We found a **34× SM-starvation** in the KVarN prefill quantize kernel, wired a batched grid
(`<<<heads,256>>>`) into `gqa_kvarn_commit_completed`, and **it breaks MTP acceptance at the model
level (accept drops to 0%, output becomes garbage `_FLAGS_FLAGS...`) EVEN THOUGH the batched quantize
is provably bit-identical to the sequential path in isolation.**

The batched kernel is **100% correct** (bit-identical codes+scales, graph-safe) yet the MODEL breaks.
The isolation shows it's a **model-interaction bug**, not a kernel-math bug. We need to find the
unit of code that corrupts the KV cache when the batch is used at the model level.

## 2. The optimization (why it's worth it)

The prefill `quantize_tile_kernel` launches `<<<1, 256, 73KB>>>` = **1 block/SM = ONE SM per tile**,
serialized by the commit host loop. Proven (bench_kvarn_batch_quantize / stream-concurrency):
| mode | per-tile time |
|---|---|
| single `<<<1>>>` x N tiles (serialized) | 38.94 us/tile |
| batched `<<<N>>>` (one block/tile) | 1.14 us/tile |
| **speedup** | **34.1x** |

And it matters: **quantize = 15.6% of the 250k prefill wall** (~79s @250k), and prefill is **99.5% of
wall @long context** (the compounding law, docs/122 §14). So batching the quantize could save ~15%
of long-context wall. High value.

## 3. THE BUG in one sentence

With the batched quantize wired in, **MTP accept=0%, output garbage** — but the same batched kernel
is **bit-identical** to sequential in isolation. Something about USING it in the paged TP2 prefill
corrupts the cache.

## 4. Everything PROVEN (so Agent 1 does not re-derive)

1. **Batched kernel = bit-identical to sequential.** tests/bench_kvarn_batch_commit validated codes
   AND scales byte-for-byte at heads=2 AND heads=4. PASS.
2. **Batched kernel = CUDA-graph-safe.** tests/bench_kvarn_batch_graph: `<<<2,256,73KB>>>` captured
   under `cudaStreamBeginCapture` produces bit-identical output. PASS.
3. **Oracle suites PASS** with the batch: slice4_kvarn_test, kvarn_materialize_oracle_test.
4. **Green baseline (fully sequential)**: correct output, **MTP accept=69.8%, tok/round=3.10**. This
   is the reference. `28318ab8` (HEAD) with a plain rebuild = green.
5. **Isolation (flip which path batches)**:
   - **TEXT prefill batched, MTP-verify sequential** → garbage output, MTP 0%.
   - **TEXT prefill sequential, MTP-verify batched** → CORRECT output, but MTP accept=4.8% (low).
   → So **the TEXT prefill batch is the garbage source** (it corrupts the cache the decode reads).
     Both paths use the SAME batch kernel; the difference is the commit context.

## 5. The current diff (everything that's UNCOMMITTED on top of green HEAD `28318ab8`)

### src/ops/kvarn/kvarn_tile_cuda.cu
- `quantize_tile_kernel`: the per-tile stride. The scale-block stride was changed from `kD` to
  **`1152`** (per-head scale region). Input stride `kD*kG`, output `tstride_q` (full tile bits).
  So block T offsets `scale_a/zp_a/scale_b` by `T * 1152`.
- Added `quantize_k_tile_gpu_batch(..., int ntiles, ...)` and `quantize_v_tile_gpu_batch(...)` that
  call `launch_quantize_tile<IS_V>(..., ntiles)` (which was already in HEAD `85a9e083` as neutral).

### src/ops/kvarn/kvarn_tile_cuda.h
- Declared the two `_gpu_batch` functions.

### src/ops/kvarn/kvarn_workspace.cu
- Added `kvarn_quantize_heads_launch` / `kvarn_quantize_vheads_launch`:
  - `heads==1` → sequential single-tile fallback (calls `kvarn_quantize_head_k/v`).
  - `heads>1` → `quantize_k/v_tile_gpu_batch(..., heads, ...)`. **One `[KVBATCH] heads=2 kb=8192`
    debug print fires once.**

### src/ops/kvarn/kvarn_workspace_launch.h
- Declared the two `kvarn_quantize_*_heads_launch`.

### src/ops/kvarn/kvarn_workspace.cpp
- `scale_scratch` alloc: `{kKvarnKScalesPerTile}` → **`{kKvarnHeadsMax * kKvarnScalesPerTile}`** (8*1152).
- `gqa_kvarn_commit_completed`: replaced the sequential `for (h in heads)` loop with
  `kvarn_quantize_heads_launch` + `kvarn_quantize_vheads_launch` (batched) + per-head
  `kvarn_store_scales_launch(cache, scales + h*1152, ...)`.
- `gqa_kvarn_hydrate_page`: `kvarn_load_scales_launch` now reads into `scales + h*1152` (per-head),
  and the dequant scale pointers are `hs + 0/256/512/576/832/1088` (was `scales + ...`).

### include/ninfer/ops/kvarn_workspace.h
- Added `kKvarnHeadDim = 256` and `kKvarnHeadsMax = 8`.

---

## 6. THE KEY QUESTION for Agent 1 (the one unit to analyze)

**The sequential commit writes scales to a SINGLE transient `scales + 0` (shared, reused per head,
stored immediately inside the h loop). The batched commit writes scales to PER-HEAD `scales + h*1152`
concurrent regions, then stores each AFTER both batches.**

The scale layout for the K and V in the 1152 block (matches the side-table field map):
```
s_col_K[0..255]  zp_K[256..511]  s_row_K[512..575]
s_col_V[576..831]  zp_V[832..895]  s_row_V[1088..1151]
```
Sequential passes (per head h, to `scales + 0` for ALL heads):
- K: `s_col_k=scales+0, zp_k=scales+256, s_row_k=scales+512`
- V: `s_row_v=scales+512, zp_v=scales+832, s_col_v=scales+576`
- store reads `scales + 0`.

Batch passes (per head h, to `scales + h*1152`):
- K: `s_col_k=scales+(h*1152)+0, zp_k=+256, s_row_k=+512`
- V: `s_row_v=scales+(h*1152)+512, zp_v=+832, s_col_v=+576`
- store reads `scales + h*1152`.

**QUESTION: is there an ALIASING/order problem?** The batch K writes `[h*1152+0, h*1152+576)`. The
batch V writes `[h*1152+512, h*1152+832)`. They OVERLAP at `[h*1152+512, h*1152+576)` (`s_row_k`
and `s_row_v` use the SAME offset 512). On the stream, K-batch completes then V-batch overwrites
`[512,576)` — so the final 512-575 region is `s_row_v`. This matches sequential (V wins). **But is the
STORED `s_row_K` supposed to be K's or V's?** (The field map puts `s_row_K` at 512 and `s_row_V` at
1088 — but the commit passes `s_row_v` at 512, not 1088. This is a pre-existing oddity; it works
sequentially because the dequant/quantize use these consistently. Verify the batch reproduces it.)

**Possible causes to check (math + code):**
1. **`kvarn_store_scales_launch` reads `scales + h*1152` — but does it read 1152 floats starting
   there, and does the buffer have `heads*1152`?** kKvarnHeadsMax=8, heads=2 → 2*1152 used, 8*1152
   allocated. Should be fine, but confirm the store's stride matches `kvarn_load_scales_launch`.
2. **Order: the batch launches `<<<heads,256,73KB>>>` = 2 blocks. On a 100KB/SM GPU, 2 blocks*73KB
   go to 2 SMs. But does `cudaFuncSetAttribute(MaxDynamicSharedMemorySize=73KB)` get set for the
   BATCH launch?** (`launch_quantize_tile` calls `configure_kernel` → yes. But the batch path might
   need it set ONCE and it's per-kernel-instantiation; confirm it's set for the `<IS_V,4>`/`<true,2>`
   used here.)
3. **TP2: each rank has kv_heads=2?** (`gqa_kvarn_commit_completed` uses `workspace.kv_heads`.) The
   KVBATCH debug confirms heads=2. Is the batch writing to the CORRECT `k_codes + phys*heads*kb_bytes`
   base, then `+ h*kb_bytes` per head? Compare to sequential's `(h + phys*heads)*kb_bytes`.
4. **V-tile input layout.** Sequential: `v_head = v_tile + h*kD*kG*2` (bytes). The kernel's
   `load_tile<true>` reads `in[j*kD + i]` ([G,D] token-major). Batch uses `tstride_in = kD*kG`
   elements. Confirm this is the RIGHT V-tile head stride AND the V dequant's output offset.

## 7. Reproduce / test

- **Rebuild**: `cd build && make ninfer_ops ninfer-serve -j8`
- **Start server**: `build/apps/ninfer-serve /home/intel/models/qwen3_8_27b.ninfer --port 8099
  --devices 0,1 --spec mtp --draft-tokens 3 --kv-dtype kvarn_k4v2 --max-context 160256
  --kv-capacity 160256` (log to a file; watch for `[KVBATCH]` + `done ... speculative=mtp N tok/round`).
- **Request** (10k): python urllib POST to `/v1/chat/completions`, check output text + `speculative=mtp`.
- **Expected buggy**: garbage `_FLAGS_FLAGS...`, `mp 1.00tok/round (0.0%)`.
- **Expected green (sequential `28318ab8`)**: "Based on the text provided, it is impossible to
  determine...", `mp 3.10tok/round (69.8%)`.

## 8. The 34× is the prize; the bug is ONE unit

Focus on the scale layout / alias / store-load symmetry in `gqa_kvarn_commit_completed` +
`gqa_kvarn_hydrate_page`. The kernel math is proven correct, so the corruption is in how the batch's
per-head scale regions are written/read/stored (or a code-offset/paging mismatch) at the model level.
Agent 1: do the math on the field offsets + the store/load stride + the concurrent-write ordering.
We are CLOSE (MTP-batch gave correct output) — the fix is likely small and in the commit/hydrate
scale layout consistency.

## 9. Files to touch
- `src/ops/kvarn/kvarn_workspace.cpp` (commit + hydrate scale layout) ← LIKELY THE BUG
- `src/ops/kvarn/kvarn_tile_cuda.cu` (kernel stride)
- `src/ops/kvarn/kvarn_workspace.cu` (launchers)

**Coordinate with coordinator (lead) before committing; the diff above is the working state.**

## 10. RESOLVED (2026-08-31): the bug was a V scale offset typo

**ROOT CAUSE (Agent 1's tip + confirmed):** the batch V-launcher passed `s_row_v` at scale offset
`+512`, but the field map + hydrate read `s_row_V` at `+1088`. `+512` is `s_row_K`. So:
- V wrote `s_row_V` to `+512` (USURPING/corrupting `s_row_K`), and
- `s_row_V` never landed at `+1088` (hydrate read stale data there).

**THE FIX:** `kvarn_quantize_vheads_launch` must pass `s_row_v = scales + 1088` (NOT +512). The
batch K-launcher offsets were already correct (`s_col=+0, zp=+256, s_row=+512`).

**CONFIRMED GREEN:** after the fix, model output is correct ("Based on the text provided, it is
impossible to determine...") and **MTP accept=69.8%, tok/round=3.10** (identical to the fully
sequential green baseline). Prefill=802.3 tok/s. slice4_kvarn_test PASS.

**The batched quantize is now CORRECT and MODEL-SAFE.** The 34x SM-starvation fix is real; the
per-head grid=2 batch at the model is green. Cross-page batching (accumulate a chunk's tiles into
one <<<N>>>) is the next step to capture the FULL 34x (>2 heads' worth) — but the per-head batch
alone already eliminates the single-tile <<<1>>> = 1-SM serialization.

KEY LESSON: the K and V quantize use DIFFERENT scale-field offsets (+512 vs +1088) even though they
look similar. A batch/launch that shares one offset for both K and V silently corrupts the V scale
layout (bit-identical CODES, WRONG scales). ALWAYS verify the scale-field map per K vs V.

## 11. POST-RESOLUTION DECISIONS (autonomous continuation)

### Batched per-head quantize (prefill) — LANDED GREEN
- Replaced the SM-starved <<<1,256,73KB>>> with <<<heads,256>>> per-head grid in the commit.
- Green: correct output, MTP 69.8%, prefill +2-3% across 10k/40k/160k.
- Committed e2b0e350. This is the confirmed, model-green prefill win.

### Full 34x cross-page batching — NOT pursued (low ROI)
The per-head batch (grid=2) is already landed. Cross-page batching would accumulate a chunk's
tiles into one <<<N>>> grid — but the commit processes one page at a time (reusing workspace.k_tile),
so it needs tile buffering (huge) or a restructured commit (risky). AND: after the Sinkhorn 16->4
fix + the batch, the quantize dropped from 3.61s to ~0.83s = ~2.8% of prefill wall. So even the
FULL 34x would only save ~2% prefill. NOT worth the architectural risk.

### V-staging (decode) — tried, REVERTED (808f19b4)
- Kernel-neutral (ns/key flat: 0.504 vs 0.500 ms @160k). Model-noisy (+2% to +16% opposite
  directions across runs). Added 4KB smem + a coalesced copy with no measured benefit.
- Reverted to keep the tree focused.

### Decode status: at bf16 parity
- KVarN decode ms/round is 1.004-1.08x bf16 (parity). vs int8 decays 1.015->1.156 (intrinsic
  4/2-bit + bf16-QK vs s8-QK). Decode is 0.5-14% of wall (compounding law) -> not the high-value
  lever. A2b/s8-QK confirmed dead end.

### FINAL OUTCOME: KVarN is near bf16 parity
- Prefill: -12.4% -> -3.8% to -5.3% vs bf16 (Sinkhorn 16->4 + batched quantize).
- Decode: at bf16 parity.
- The 4/2-bit compression is now essentially FREE on performance (near bf16 perf).
Critical knowledge: the K-vs-V scale-offset asymmetry (s_row_K=+512 vs s_row_V=+1088) — a
launcher that shares one offset silently corrupts the V scale table (bit-identical codes, wrong
scales). Always verify per-field per kernel.
