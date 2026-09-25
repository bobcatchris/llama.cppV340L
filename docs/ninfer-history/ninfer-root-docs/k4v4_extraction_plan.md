# k4v4 (KVarN K4/V4) — extraction plan, not a third kernel fork

Status: **plan only, no code changed.** Blocked on a GPU grant to capture the `<5,4>` baseline.
Coordinator approved templating over forking; this refines that into per-side helper extraction.

## Why not a third fork

slice4 (k4v2) = 584 lines, slice6 (k5v4) = 641 lines. Their diff is 237 lines, of which **151 are
comments** — but the remaining ~86 are not derived constants, they are **structural**:

| | K4 side (slice4) | K5 side (slice6) |
|---|---|---|
| row bytes (32 keys) | 16 | 20 |
| cp.async op size | **8 B** (16 B "would collide banks 4-way") | **4 B** (20 not 16-aligned) |
| ops per tile (K) | 512 (256 rows × 2 halves) | 1280 (256 rows × 5) |
| bank layout | **skewed** `16*d + 8*(d>>4) + j` | **linear** `d*20 + byte` |
| source offset | `page_offset >> 1` — **4-bit codes pair 2 keys/byte** | `hw * 20` — no pairing, linear bitstream |
| bank conflicts | conflict-free by construction | recorded 32-way `LDS.U8` debt |

Re-deriving any of that is easy to get subtly wrong. So k4v4 must **lift proven code verbatim**,
not re-implement it.

## The recombination

The two sides are already independent in the fused prefetch loop (it splits at `if (op < N)`), and
every piece k4v4 needs already exists and has oracle coverage:

| tier | K prologue | V prologue |
|---|---|---|
| slice4 `k4v2` | **K4** (skewed, 16 B, 512×8 B) | V2 (linear, 64 B, 128×16 B) |
| slice6 `k5v4` | K5 (linear, 20 B, 1280×4 B) | **V4** (linear, 128 B, 256×16 B) |
| **k4v4** | **K4 — reuse slice4** | **V4 — reuse slice6** |

k4v4 contributes **zero new bit-manipulation code**. It composes two halves that are each already
proven.

**CORRECTION — "every future tier becomes free" is false, and I said it.** It is true only for tiers
whose K width AND V width have each already been built. k4v4 qualifies. **k3v3 does not:**

| tier | K side exists? | V side exists? | new prologue code? |
|---|---|---|---|
| `k4v4` | yes (slice4 K4) | yes (slice6 V4) | **none** |
| `k3v3` | **no** | **no** | both sides, from scratch |

3-bit is genuinely awkward in a way 4- and 5-bit are not: a 32-key K row is 12 B and a 256-channel V
row is 96 B, neither 16-aligned, and 3 does not divide 8, so codes straddle bytes on a period of
3 bytes / 8 codes rather than pairing (4-bit) or running linearly (5-bit). `kvarn_unpack_code(row,
idx, 3)` itself is fine — a 3-bit code never spans more than 2 bytes — but the staging geometry and
the bank skew would both be new, and a conflict-free 12 B row layout has to be designed rather than
lifted.

So the extraction still pays off for k3v3 (it makes the *composition* trivial once K3/V3 exist, and
it gives them a proven shape to copy), but it does not make k3v3 free. Budget it as its own kernel
work, roughly the cost of slice6 originally was.

## Derived geometry for k4v4 (computed, to be confirmed by compile)

- K tile = `kKvarnKTileBytes` = **4224 B** (skewed, from slice4)
- V tile = `kKvarnV4TileBytes` = **4096 B** (from slice6)
- Stage smem = 4224 + 4096 = **8320 B single-buffered** — *smaller* than k5v4's 9232 B, so slice6's
  single-buffered pipeline ordering (dequant kb, sync, issue kb+1) carries over unchanged. No
  `__launch_bounds__` regression.
- prefetch ops = 512 (K, 8 B) + 256 (V, 16 B) = **768**
- K page code bytes = `256*64*4/8` = 8192; V page code bytes = `256*64*4/8` = 8192
- budget `kvarn_kv_bytes_per_token(4,4)` = `18496 * (16384*8/8 + 4608) / 34816` = **11152**
  → `kv_unit` = `11152*17/16` = **11849**
  Sits between k4v2 (8976/9537) and k5v4 (12240/13005), as expected.
- Perf note: the 4-bit skewed bank does **not** have k5v4's 32-way conflict, so k4v4's K side should
  be faster than k5v4's, not slower.

### Incidental finding worth recording: k3v3 is VRAM-identical to k4v2

```
k4v2: 8976 B/token   kv_unit 9537      (4+2 = 6 bits)
k3v3: 8976 B/token   kv_unit 9537      (3+3 = 6 bits)
```

The budget formula depends on `k_bits + v_bits` for the code bytes, so any tier with the same total
width costs the same VRAM. k3v3 is therefore a **VRAM-neutral alternative to the shipped k4v2** —
same footprint, different K/V error split (more K precision at the cost of V, or vice versa). That is
a genuinely useful point on the docs/117 ladder: it means a K/V balance change is free in capacity
terms, and only costs accuracy trade-off. Not part of this work order; recorded because the
arithmetic fell out of the verification.

## Extraction shape

New shared header `gqa_decode_kvarn_prologues.cuh` holding per-side, width-parameterized pieces
lifted **verbatim**:

- from slice4: `kvarn_k_tile_off`, the K half of `kvarn_prefetch_tile`, the K half of
  `prologue_kvarn_kv_tile`, and the K half of `prologue_kvarn_tail_rows`
- from slice6: `kvarn_k5_tile_off` (as the K5 variant), the V half of `kvarn_k5v4_prefetch_tile`,
  and the V halves of `prologue_kvarn_k5v4_kv_tile` / `_tail_rows`

Then each tier is a thin instantiation selecting (K-side, V-side). The MMA/attention body at
slice4:227 / slice6:288 is width-independent (it consumes dequantized bf16 tiles) and is NOT touched.

Splitting the fused `prologue_*_kv_tile` functions per side is the only genuinely delicate part:
K and V share the `dq` scratch and the barrier structure, so the split must preserve the existing
barrier placement exactly. That is where the regression gate earns its keep.

## Plumbing (small, and mostly already paid for)

1. `types.h`: `KvCacheStorage::KvarnK4V4` + `kvarn_tier_widths` case `{4,4}`.
2. `types.h` shared name table: one entry → **parses in serve, CLI, and bench simultaneously.**
   The single-table work from the k5v4 lane means the "N parsers with divergent accepted sets"
   bug class does not recur here.
3. Dispatch route in `gqa_attention_cached` → the new instantiation.
4. `tp2_budget.h`: `kvarn_kv_bytes_per_token(4,4)` already computes correctly (width-generic).
5. `serve_options.cpp`: k4v4 needs **no** §5 refusal once validated — but per the standing rule,
   add the gate rather than assuming; it starts unvalidated like k5v4 did.

## Regression gate (the acceptance criterion, in order)

**Step 0 — baseline BEFORE any edit.** Capture, on GPU:
- `k5v4_oracle` 10/10 + worst error (expect 0.0312 vs tol 0.0616)
- known-answer bytes `0x41,0x0C,0x52,0xCC,0x41`
- `k5v4_dispatch` 10/10
- `kvarn_batched_ops` bit-identity, `kvarn_gqa`, `slice4_kvarn_test`
- plus a **k4v2** baseline, since slice4 is also being refactored.

Store the outputs as a golden file. Without this the "bit-identical" claim is unfalsifiable.

**Step 1** — extract helpers, redirect slice4 + slice6 to them, re-run everything.
**Pass condition: byte-identical to the golden file for BOTH `<4,2>` and `<5,4>`.** Two tiers must
be pinned, not one — both source kernels are being edited.

**Step 2** — add k4v4 as the `<4,4>` instantiation, then its own oracle + dispatch tests (clone the
k5v4 harnesses, which is the cheap part).

## Honest estimate

~1.5–2 days including the two-tier regression proof. My earlier "+0.5d" assumed the differences were
derived constants; they are not (see the table above). The split of the fused K/V prologue while
preserving barrier placement is the risky part, and it is the reason the gate must exist before the
edit rather than after.

## Extraction map (the seam is cleaner than expected)

Read the actual prologue control flow before planning the split. `prologue_kvarn_k5v4_kv_tile`
(slice6:161) and `prologue_kvarn_kv_tile` (slice4:112) share this shape:

```
T = scales + stride * (kv_head + KVHeads*page)          # shared, width-independent
if (!page_ok) { zero-fill k_s AND v_s; __syncthreads(); return; }   # shared
K stage ...                     ; __syncthreads()        # per-side
V stage ...                     ; __syncthreads()        # per-side
```

**K and V are already barrier-separated**, so the split lands exactly on existing `__syncthreads()`
boundaries — no barrier has to move. That removes the risk I flagged earlier ("splitting the fused
prologue while preserving barrier placement is the delicate part"): the fusion is only in the
*prefetch* op loop (`if (op < 1280)`), not in the dequant stages. Corrected below.

### What lifts, verbatim, into `gqa_decode_kvarn_prologues.cuh`

| piece | source | used by |
|---|---|---|
| `kvarn_k_tile_off` (skew `16d + 8(d>>4) + j`) | slice4:66 | k4v2, **k4v4** |
| K4 prefetch ops (512 × 8 B, src `page_offset>>1`) | slice4:73 | k4v2, **k4v4** |
| K4 dequant stage | slice4:112 (K half) | k4v2, **k4v4** |
| `kvarn_k5_tile_off` (linear `d*20 + byte`) | slice6:105 | k5v4 |
| K5 prefetch ops (1280 × 4 B, src `hw*20`) | slice6:112 | k5v4 |
| K5 dequant stage | slice6:161 (K half) | k5v4 |
| V2 prefetch + dequant (64 B rows, 128 × 16 B) | slice4 | k4v2 |
| V4 prefetch + dequant (128 B rows, 256 × 16 B) | slice6 | k5v4, **k4v4** |
| tail-rows stage | both, per side | all |

Composition after extraction: `k4v2 = K4+V2` (slice4 today), `k5v4 = K5+V4` (slice6 today),
`k4v4 = K4+V4` (new, **no new bit-manipulation code**).

### Prefetch loop needs one change, not a split

The fused `for (op = tid; op < K_OPS + V_OPS; op += nthr)` keeps a single loop for issue-rate
reasons. Parameterizing it means `K_OPS`/`V_OPS` and the two branches become width-selected:

| tier | K ops | V ops | total |
|---|---|---|---|
| k4v2 | 512 (8 B) | 128 (16 B) | 640 |
| k5v4 | 1280 (4 B) | 256 (16 B) | 1536 |
| **k4v4** | **512 (8 B)** | **256 (16 B)** | **768** |

k4v4's totals are the two existing halves concatenated — the loop body for each branch is lifted
unchanged. This is the one place a `constexpr` selection is genuinely required.

### Correction to my own earlier risk assessment

I wrote above that splitting the fused prologue "while preserving barrier placement is the risky
part". Having now read the control flow, **that was overstated**: the dequant stages are already
separated by `__syncthreads()`, so the split is on a natural boundary. The real risk is narrower —
the `!page_ok` zero-fill writes *both* tiles in one loop, so it must stay shared rather than being
duplicated per side, or a bad page could leave one tile stale. That is a single decision point, not
a refactor hazard.

### Revised estimate

~1–1.5 d, down from the 1.5–2 d I quoted an hour ago, because the barrier seam turned out to be
clean. Still not the +0.5 d I first guessed — the width differences are structural (op size, op
count, skew, byte-pairing) and the two-tier regression gate is real work. Estimate moved in both
directions as I read the code; the second correction is the one that matters, since it came from
verification rather than optimism.

## Exact lift boundaries (verified against both candidate base branches)

Confirmed the decode kernels are **identical across `wo/k4v4-cpu` and `wo/dflash2-scope`** —
`gqa_decode_slice4_kvarn.cuh` and `gqa_attention_kvarn.cuh` hash-match; `slice6` differs only by the
three stale-comment fixes (28 diff lines, zero non-comment). So this mapping is base-independent.

**slice4 (`k4v2`) — the K side to lift:**
| lines | content |
|---|---|
| 123–159 | K: 4-bit codes `[channel][key/2]`, low nibble = even key; read via `kbuf[kvarn_k_tile_off(d, t0>>1)]` at :149 (shift+mask, **pairing**) |
| 135 | barrier inside the K block |

**slice6 (`k5v4`) — the V side to lift:**
| lines | content |
|---|---|
| 187–196 | `!page_ok` zero-fill (shared, stays shared) |
| 200–227 | K: 5-bit channel-major, `kvarn_k5_tile_off`, `kvarn_unpack_code(row,t0,5)` — **NOT used by k4v4** |
| 228–246 | V: 4-bit **token-major, 128 B rows, nibble-aligned** — this is k4v4's V side |

So `k4v4 = slice4:123–159 (K) + slice6:228–246 (V)`, both contiguous and barrier-delimited. Plus the
per-side prefetch halves (slice4's K: 512×8 B ops; slice6's V: 256×16 B ops) and the two tile-offset
functions (`kvarn_k_tile_off` skewed from slice4; V4 linear from slice6).

Nothing needs re-deriving. The only genuinely new artifact is the composition header itself.

## ⚠️ Base-branch trap (found while starting, 2026-09-05)

Was directed to rebase on `wo/k4v4-cpu@21893588`. **That branch does not contain the budget seam
refactor** — `kv_bytes_for_tier`, `validate_cache_type_widths` and `verify_no_divergence` all have
zero references in its `tp_engine.cpp`, and the superseded inline `768ULL` gate is still present.
`wo/dflash2-scope@b81bc616` has all of it, plus the originals of every commit `k4v4-cpu` carries as a
cherry-pick. The only thing unique to `k4v4-cpu` is `58dd47d0`, the k4v4 tier table.

**Demonstrated, not predicted:** performing the directed merge produced
`Automatic merge went well` with **no conflicts**, and afterwards all three mutation-proven seam
functions had zero call sites — a clean auto-merge that silently reverted the refactor and left the
new functions as dead code. Caught only by running the Appendix F grep *after* the merge rather than
trusting git's success message.

**Correct base: `wo/dflash2-scope@b81bc616` + cherry-pick `58dd47d0`.**
## Progress log

**Step A — DONE (`ca685d7c`).** `gqa_decode_kvarn_prologues.cuh` written: per-side geometry, bank
offsets, cp.async prefetch halves, and dequant stages for K4/K5/V2/V4. Compiles clean.
Namespace must be `ninfer::ops`, **not** `ninfer::ops::kernel` — every sibling kvarn header uses the
former and a nested one makes all lifted symbols invisible (cost me a debug cycle).

**Step B — DONE and PROVEN (`b6608b0a`).** slice6 redirected to `kvarn_prefetch_k5` /
`kvarn_prefetch_v4`; its duplicate constants and `kvarn_k5_tile_off` **deleted, not shadowed**.
All three sealed harnesses reproduce byte-for-byte:
`MATCH oracle.txt / MATCH dispatch.txt / MATCH known_answer.txt`.
The fused-loop → two-loop split is therefore proven data-equivalent, not merely argued.
Also fixed: slice6's `#include` block was duplicated verbatim (six identical includes, twice).

### Step C — remaining work on slice4, with the exact constraint that makes it non-trivial

slice4's symbol names **differ** from the header's (`kKvarnKTileBytes` vs `kKvarnK4TileBytes`,
`kvarn_k_tile_off` vs `kvarn_k4_tile_off`), so there is **no collision** — which is the trap: the
edit will compile and pass with slice4 still carrying its own copies, and the duplication will
survive silently. Rule 15 applies to *values*, so the requirement is that slice4's constants be
**defined in terms of the header's**, not re-stated.

Reference counts (measured, so the blast radius is known before editing):

| symbol | refs | note |
|---|---|---|
| `kKvarnStageSmemBytes` | **12** | used by `gqa_attention_kvarn.cu:575` and `tests/slice4_kvarn_bench.cu` — **must stay defined here**, but as `2 * (kKvarnK4TileBytes + kKvarnV2TileBytes)` |
| `kKvarnKTileBytes` | 5 | alias to `kKvarnK4TileBytes` |
| `kKvarnVTileBytes` | 3 | alias to `kKvarnV2TileBytes` |
| `kvarn_k_tile_off` | 3 | replace call sites with `kvarn_k4_tile_off`, then delete |
| `kKvarnKStageRow` | 2 | legacy, unused by the tile banks — leave alone |

Then replace slice4's fused prefetch loop (`op < 512 + 128`) with
`kvarn_prefetch_k4(...)` + `kvarn_prefetch_v2(...)`, and its K/V dequant blocks with
`kvarn_dequant_k4_tile(...)` / `kvarn_dequant_v2_tile(...)`. Keep the `!page_ok` zero-fill in the
caller — it writes both tiles and must stay shared.

**Gate: `<4,2>` must reproduce byte-for-byte** against the sealed baseline, same as Step B. Rebuild
`ninfer_slice4_kvarn_test`, `ninfer_kvarn_batched_ops_test`, `ninfer_kvarn_gqa_test` and the slice4
bench; all must stay green.

### Step D — the actual k4v4 tier

`k4v4 = K4 + V4`, stage smem `kKvarnK4V4StageBytes` = 8320 B (single-buffered, smaller than k5v4's
9232, so slice6's pipeline ordering carries over). Then: dispatch route in `gqa_attention_cached`,
`is_kvarn_storage`/registry gate opened **only once the prologue exists**, and k4v4's own oracle +
dispatch tests cloned from the k5v4 harnesses. Budget figure 11152 B/token / `kv_unit` 11849 comes
free from `kvarn_tier_widths` — do not restate it.
