# C50 rail verdicts + SHORT-PATH (<256) address list — agent5, 2026-09-15

Lane self-check (law 36, post-reboot): tip `f412d6fa` == `origin/amd-wo-nvfp4`, worktree clean. Survived.

## PART A — C50 (G18r40_tkbisect) rail run: the ids arrived to the judge

Bank: `results/amd/a5/C50_G18r40/` — 15 per-leg `.ids.txt` + `C50_manifest.json` (trace2 tuple per leg)
+ `C50_rail_arms.json` (full arm JSON per leg) + `C50_rail_run_v1.txt` (raw). Runner card released
18:37:58Z, KFD clean, bank sha `2aae4dc2…` verified in-row, **stop_set=2 live at last** (the
vacuous-0 curse is broken — `in_stop` self-attests from this boot forward).

### The calibration twist (archivist note)
agent4's "true-token" targets were template-relative: target+14 = printed prompt
(56→70 … 129→143; warmup/controls land at 53/54). So the probe swept prompts **70..143** — the
pinned PAGE-64 flip window {65,66} sits in the UNPROBED gap (54,70]. The boundary is now
bracketed: clean at ≤54 (27-token coherent legs, byte-EQUAL same-boot pair), noise at every
≥70 leg. That is a coordinate: the crossing lives in **(54,70]** — the 64→65 page crossing is
inside it, the chunk-128 story is dead (see rulings), and the next probe (if the chair grants
one) needs legs at 58/62/64/65/66/68 only.

### Per-leg H-arm table (each leg vs clean54_A ref; §8ai states, no stretching)
| leg | prompt | n | H1 paired/multiset | H4 region | formal rc |
|---|---|---|---|---|---|
| warmup_p53 | 53 | 4 | INSUFFICIENT | NO scatter (1018 vs 1421) | 0 (H2, see note) |
| probe_p54_g1 | 54 | 1 | INSUFFICIENT | NO scatter (1421=1421) | 1 all-miss (acquit) |
| clean54_B (pair) | 54 | 27 | — (identical streams) | NO scatter | 1 all-miss (acquit — correct: both-directions falsifier holds) |
| t56 | 70 | 3 | INSUFFICIENT | SCATTER 96,885 | 1 all-miss |
| t60 | 74 | 3 | INSUFFICIENT | SCATTER 67,805 | 1 all-miss |
| t63 | 77 | 2 | INSUFFICIENT | SCATTER 152,830 | 1 all-miss |
| t64 | 78 | 3 | INSUFFICIENT | SCATTER 52,074 | 1 all-miss |
| t65 | 79 | 3 | INSUFFICIENT | SCATTER 67,805 | 1 all-miss |
| t66 | 80 | 3 | INSUFFICIENT | SCATTER 28,767 | 1 all-miss |
| t70 | 84 | 2 | INSUFFICIENT | SCATTER 94,361 | 1 all-miss |
| t80 | 94 | 3 | INSUFFICIENT | SCATTER 28,767 | 1 all-miss |
| t128 | 142 | 2 | INSUFFICIENT | SCATTER 38,080 | 1 all-miss |
| t129 | 143 | 3 | INSUFFICIENT | SCATTER 100,053 | 1 all-miss |
| enum85 ctl | 85 | 2 | INSUFFICIENT | SCATTER 95,082 | 1 all-miss |

- **INSUFFICIENT at n<8 is said per leg, as named** — H1/H6 refuse the gen=2/3 legs by design; a
  2-3-point stream cannot support or refute any offset. That is the acquittal direction working,
  not a failure of the rail.
- **H4 is the speaking arm** (reporting band, not a threshold conviction): every leg at prompt≥70
  scatters (medians 30k–154k vs coherent 1421); every leg at prompt≤54 does not. The C48
  region-scatter signature REPRODUCES under the new true-token calibration — same shape, cleaner
  bracket.
- **warmup_p53 H2 rc=0 = annotate, not convict**: at n=4 the adjacent-shift is a shared-suffix
  artifact of the SAME clean family (warmup ids `760 1156 1018 328` vs clean head
  `1421 1018 328 …`), and its H4 shows NO scatter. Ruling: warmup_p53 is a member of the clean
  side (53 ≤ 54 boundary holds again), NOT a lane-transposition finding. The tool's rc=0 stands
  mechanically; the judge's note says what it is worth.
- **Structural finding #2 (free to graders, same class as token-0=32198×4):** first-decode id
  **30188 is position-invariant across SIX legs** (prompts 74,78,79,80,84,94), and second-decode
  69226 repeats at 74 & 79. Wrong-decode token 0 is again a function of prefill-END state, not
  content — six different fill bodies, same id.
- Clock cell (this boot): 8 convictions, all A1 LANE-GEN-DESYNC + A2 CANCEL-WITHOUT-WITNESS on
  the three `finish=stop gen=2` legs (63/70/128) + enum ctl — the cancelled trio's streams are
  2-token truncations; their verdicts above already say INSUFFICIENT, so no ruling leans on them.

### RULINGS on agent4's three PINNED predictions (judge, by bytes)
1. **PAGE-64 (flips in {65,66}): UNADJUDICATED — surviving.** No leg landed at 65/66 (the +14
   template offset skipped the window). Bracket (54,70] contains 64→65. Consistent with it;
   cannot convict it yet. NOT falsified — say so precisely.
2. **CHUNK-128 (flips at 129, 65..128 clean): FALSIFIED.** Legs at prompts 70,74,77,78,79,80,84,94
   — all inside the predicted-clean 65..128 span — are noise (H4 scatter + garbled streams).
3. **CONTENT-DIFFUSE (never flips): FALSIFIED.** Noise appears at fillwords 8,12,15,16,17,18,22,32,80,81
   — every content/length combination probed ≥70 flipped; content cannot be the driver if lengths
   70 and 143 both speak noise with the SAME first-decode id.
Net: the boundary is a REAL coordinate in (54,70] and only PAGE-64 remains standing among the
pinned three. agent4's unit self-catch at 17:5xZ stays vindicated (the 55–84 word-count window
was indeed never the axis) — and the SAME unit class bit once more: token-target vs
prompt-after-template (+14) shifted the sweep above the decisive window. Retraction beat carried
to watch-notes: **a calibration oracle that counts the body but not the chat-template wrapper is
half-calibrated** — the era's second "the window was never probed" moment, this time by +14.

## PART B — ZERO-CARD PREP: the SHORT-PATH (<256) page/index/offset address list
Per agent4's static path-map: prompts ≤ 256 ride the single prefill + per-token decode path with
`--prefill-chunk 128` (so ≤256 = at most 2 chunk-appends) — no branch change; crossing candidate
inside. Every position→address computation in that path, with hazard notes (NOT verdicts):

### 0. Constants (the page law)
- `src/core/paged_kv_cache.h:15` — `kPagedKVPageSize = 64`.
- `src/ops/kernel/paged_kv_address.cuh:9-11` — `kPagedKVPageShift = 6`, `kPagedKVPageMask`, static
  assert page==1<<shift.
- `src/ops/kernel/gqa_attention_kv_quant_nvfp4.cuh:36-41` — head-dim 256, group 16, code row 128 B,
  scale row 16 B.

### 1. Canonical addressing (shared by ALL sites below)
- `paged_kv_address.cuh:13-16` — `physical_page = block_table[position >> 6]` (THE crossing op: first
  table read at position 64).
- `:18-24` — `page_head_offset = LeadingExtent*64*(head + KVHeads*physical_page)` (region base).
- `:26-33` — `element_offset = page_head_offset + LeadingExtent*page_offset + leading`.
- `:35-42` — combined block_table form (`position & mask` = page offset).

### 2. WRITE path — nvfp4 prefill/decode KV append (quantizing)
- `gqa_attention_kv_quant_nvfp4.cuh:205-230`: `unit = blockIdx.x*8 + warp_id`, `token = unit / KVHeads`,
  **`position = positions[0] + token`** (:214 — contiguity-by-assumption from `positions[0]`, one
  append run), `page = paged_kv_physical_page(block_table, position)` (:215), `page_off = position & 63`
  (:216); row stores via code/scale index helpers `:54-66`.
  HAZARD-1: if `positions[]` is ever a strided/rotated vector while the kernel assumes `positions[0]+token`,
  appends land one page off exactly when the run crosses 64. This is the write-side first suspect.
- (bf16-family sibling, format-agnostic witness: `kv_cache_append_prefix.cuh` — paged kernel
  `:106-152`: per-warp `lane==0` reads `positions[token]`, `block_table[position>>6]`, shfl-broadcast
  `:138-145`, dst `page_off + 64*(physical_page + physical_pages*kv_head)` `:55-58`; cyclic sibling
  `:68-104` uses `slot = position & 4095` — TWO slot laws for one cache, page (64-modular) vs
  cyclic (4096-window). q3 twin has NO nvfp4 planes yet showed the same scatter → the addressing
  class above is the shared-stack suspect the q3-twin already named.)

### 3. READ path — prefill attention (gather)
- `gqa_attention_prefill_nvfp4.cuh:43-73` — stager: `block_base`/`scale_base` from
  `(physical_page, kv_head, k0 & 63, 0)` (:51-55), then walks **within the tile** `key_l*128` /
  `key_l*16` bytes (:64-67); keys beyond `max_query_abs` zeroed (:50, :61-66 — `full_tile =
  (k0+Bc-1) <= max_query_abs`).
- `:90-93` — `D=256, Bc=64, Threads=128`: key tile == page size EXACTLY, so a tile fits one page
  iff tiles are page-aligned.
- `:201` — **`physical_page = block_table[0]`** (tile 0 starts at table slot 0 — true only if
  key-space base ≡ 0 mod 64; prefix reuse was `full_reset` this boot, so UNTESTED off zero).
- `:239` — rotation `(kb+1 < n_block_max) ? block_table[kb+1] : physical_page`; `:465-467` carries
  it to the next iteration.
  HAZARD-2: causal diagonal: query row i sees keys ≤ base+i; the page holding the LAST visible key
  is `block_table[(base+i)>>6]` — correct only if every tile advance (:373-style `(k0 & 63)==0`
  gating) fires on the same 64-grid as the WRITE. With Bc==page==64 it does — as long as `base` is
  64-aligned. base here = tokens already in cache = 0 for these boots. Keep on the list for the
  chunked case (prompt 142 = chunk 128 + 14: second append's `positions[0]` = 128 = page-aligned —
  clean by arithmetic, another reason the crossing is 64, not 128).

### 4. READ path — decode attention (T=1, engine=5)
- `gqa_decode_slice7_nvfp4.cuh:132-133` — block_table row select (`table_row*table_stride`).
- `:192-197` — **smem page-table preload: `first_page = first_tile >> 6`, `page_count =
  ((split_end-1)>>6) - first_page + 1`, cap 64 entries (=4096 keys)** — prompts ≤256 use ≤5
  entries; HAZARD-3: cap check is a comment (`:194-196`), page_count overflow = silent OOB smem.
- `:212-213` — per-position `paged_kv_physical_page` + `& mask`; `:290-315` tile issue via canonical
  `paged_kv_element_offset` (`byte0*2`, `key & 63`); `:348-376` rotation gated on
  `(next_k0 & 63)==0` — same 64-grid law.

### 5. HOST side — tables & envelopes
- `src/ops/wrapper/gqa_attention.cpp:107-110` — `physical_pages = k_pages.ne[3]`,
  `logical_pages = block_table.ne[0]`, `capacity = logical*64` (validated shapes for NVFP4 planes
  at `:152-168`: `[D/2, 64, kv_heads, physical_pages]` — the region law the kernels index against).
- `src/runtime/tp2/tp2_backend.cpp:728-730,750` — `pages_per_lane = ceil(...)` budget math,
  `pages = pages_per_lane*lanes`; `:798-799` `text_physical_page_groups`.
  HAZARD-4: `tp2_backend.cpp:210-217` — the HKV-FENCE mprotects **4096-byte** pages around a `conv_start`
  (host-page granularity ≠ KV page granularity; a fence computed in BYTES near the conv state region
  deserves one grep at a 64-token crossing).

### 6. Draft/MTP append (engine=5 pool) — LISTED, likely INACTIVE this boot
- `src/runtime/tp2/dflash2_context_append.cu:21,125` — pool append `slot = pa & 2047` (lane-major,
  2048-window cyclic), rope at committed absolute positions `:114-122`; declared
  `dflash2_context_append.cuh:49,89-91`. `speculative=off, rounds=0` in the boot logs → suspect for
  depth>1 only. HAZARD-5 (class): **write cyclic `&2047`, read paged `>>6/&63`** — if any organ
  mixes the two slot laws for the same logical position, the disagreement appears at the FIRST
  position where the laws differ: they never differ at p<64… but `&2047` preserves mod-64 for all
  p<2048, so mixing is silent — the suspect must be a TABLE problem, not a modulo problem. (Said
  out loud so the dive doesn't waste a card minute on modulo-vs-cyclic.)

### Where the dive starts, per prediction outcome
- PAGE-64 convicts (flip at 65): open §2 `:214-216` (write contiguity), §5 allocator fill of the
  block_table row (does table[1] exist / point at the right physical page when a lane's 2nd logical
  page is allocated DURING the prefill launch?), then §3 `:201` base-alignment law.
- No clean bracket tightener available at zero card — the +14 lesson says: next probe must print the
  TOKENIZED length BEFORE firing (the calibration oracle belongs in the request builder, not in the
  runner's arithmetic).

## Law-36 self-check + carry notes
- Reboot survival: tip+push verified above; rail tools at `tools/v340l/nvfp4/ids_offset_triage_agent5.py`
  (selftest 9/9) and `finish_clock_consistency_host.py` untouched since f412d6fa.
- Retraction beats carried: (1) grounded twin killed the 'coherent q3 enum refusal' era-narration
  (PREFIX-EQUAL at bank — bytes outrank prose); (2) agent4's UNBORN falsifier at the pen (filler
  ~4.7 tok/word; the 55–84 word window was never probed) + C50's +14 template offset — same unit
  class, second birth. The hunt's next unit-law candidate: **state every length in ONE unit (tokens
  post-template) and print it from the tokenizer, or say nothing.**
