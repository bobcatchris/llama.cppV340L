> **Landed on main 2026-08-26 as docs/74** (QA review + numbering map: docs/77).
> Original: `wo/kvarn-pp` @ 52af56e5, file `docs/72_kvarn_attention_optimization_work_order.md` (content as of landing).
> In-text references to docs/69–74 use the BRANCH numbering: 69=full-matrix(docs/70), 70=prefix-reuse(docs/72), 71=direct-read(docs/73), 72=attn-work-order(docs/74), 73=staged-layout(docs/75), 74=attn-scope(docs/76).
> LIVE WORK ORDER. **Branch merged into main 2026-08-26 (fe3d8857): this file IS the branch copy** — step 1 (attribution baseline) and step 2 (staged wall 1024→1344 MiB, 98b5395e) are done; full CI 20260826_095551 green except pre-existing mtp_long (triage: results/mtp_long_divergence_20260826.md). **STOP (2026-08-26, user directive): step 3 ("incremental materialization / persistent BF16 overflow") is DELETED and step 2 (wall raise) will be reverted — superseded by docs/78 (wall removal + packed-only decode). Do NOT start step 3 or docs/76 C5. Next code work: docs/78 step 0 (baseline) → step 1 (packed-only decode kernel, C4), under the docs/77 §8 merge hold (H0').**

---

# 74 — KVarN attention-path optimization (kill the repeated beyond-wall materialize): Agent Work Order

**Status:** CURRENT — work order for a single implementer agent.
**Mission:** lower KVarN beyond-wall attention cost (decode + prefill) on the
2× RTX 5060 Ti 16 GB build. The D-19 wall fix changed **which kernel** serves
small-T beyond-wall, but every beyond-wall round/chunk still **re-materializes
the entire visible history** into a fresh per-call BF16 temp. Done = a measured
before/after (methods §4) showing: (a) beyond-wall decode and prefill do not
re-dequant unchanged pages (delta-only), (b) target-context decode no longer
materializes at all when the wall covers it, (c) monochrome — every step keeps
output byte-identical/before-the-fix-deterministic and all gates green (T19
≥450 tok/s unchanged, decode guard ±2%, battery T16–T19 @250k).

Read this document fully before writing code. Background: the review
`docs/kvarn_deep_seek_review.md` §5–§7 and the composite main-side review
(`docs/71_kvarn_branch_review_and_merge_gates.md`) are the source of the
problem statement; docs/68 (§D-19) and docs/71 (`direct_read_tc_prefill.md`)
hold the architecture reasoning.

---

## 1. Context (60-second version)

The KVarN attention path contains **two regimes**, split by whether the
sequence's page count fits a BF16 "staged shadow":

- **In-capacity** (`need_pages <= staged_pages`): the open-page tile is staged
  into the BF16 shadow, then attention runs the **proven BF16** kernel
  (small-T TC split-K for T≤6, else FA2 flash) — **no dequant at all**.
- **Beyond-wall** (`need_pages > staged_pages`): attention calls
  `gqa_attention_kvarn_cached_launch`, which **materializes every visible page**
  (copies the shadow, dequantizes the packed above-wall pages, adds the tail)
  into a fresh BF16 temp, then runs the same proven BF16 kernel on that temp.

The D-19 fix (`acd6f791`) only corrected the *second-stage routing* (route T≤6
to the TC split-K instead of the collapsed 12-CTA flash). The **pass-1
materialization is still paid in full on every call**. For a 40k MTP verify
round (T=4), pass 1 dequantizes ~625 pages (all visible) to serve 4 query rows.

**What must NOT change:** the BF16-attention substrate (docs/71 proved
direct-read *prefill* is spill-bound), the quantized KV format (k4/v2), the
determinism of the split-K/merge ordering, or any hard gate threshold.

**Already built and verified (do not redo):**
- D-19 wall fix `acd6f791`: beyond-wall T≤6 routes to TC split-K; 40k decode
  35.1→60.6 tok/s; MTP acceptance 85.7%.
- Tiled small-T split kernel `060860fe`/`8e579417` (retained, not the live
  route); docs/71 direct-read analysis `0dafe9c7`/`08f00892`.
- Materialize+2-pass flash `gqa_attention_kvarn_flash.cuh` (D-18 step 3).
- BF16 staged-shadow machinery `kvarn_bind_staged_shadow` / `gqa_kvarn_stage_pages`.

**What you are doing:** §6 steps 1–6 (measure → raise the wall → incremental
materialize → conditional fusions → cleanup).

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; `git worktree` workflow — see
  template §2). Use worktree `wo/kvarn-pp` (this branch) or a fresh one.
- Build (inside your worktree): `cmake --build build -j 16`
  (CUDA arch forced to `sm_120a`; 2× RTX 5060 Ti 16 GB).
- Unit tests: **use `/usr/bin/ctest`** (the `ctest` on PATH is a broken Python
  wrapper). From `build/`: `/usr/bin/ctest -R "<pattern>"`.
- Test entry points (standard):
  - GPU unit/bench: `build/tests/<name>` (build first).
  - Correctness battery: `tools/smoke/serve_correctness_ci.sh`; env-overridable
    `KV_DTYPE=kvarn_k4v2 KV_CAPACITY=250000 MAX_CONTEXT=250000`; `--mode ci`
    (fast gate), `--mode full` (int8 + KVarN battery via `run_ci.sh --full`).
  - KVarN battery T16–T19 @250k and T19 (≥450 tok/s) are **only** run by
    `run_ci.sh --full` — use that for closeout.
  - Measurement: prompt sizes from `usage.prompt_tokens`, NEVER char estimates;
    decode rate from serve.log `decode=Xtok/s`; record `nvidia-smi` clocks
    (+ VRAM) with **every** perf number.
  - Live server: one at a time; port 8091 serves an active conversation — swap
    protocol in §7.
- **Task-specific perf harness:** build a small `tests/bench_kvarn_attention.cu`
  (or reuse `bench_kvarn_2pass.cu`) that calls `gqa_attention_kvarn_cached_launch`
  directly with a fixed `packed_pages`, `tail_count`, and `T`, and times pass-1
  (materialize) vs pass-2. Use it to attribute time pre/post change.
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.

## 3. Architecture facts (verified — do not re-derive)

- **Page/tile geometry:** `kKvarnAttnD=256`, `kKvarnAttnG=64`
  (`src/ops/kernel/gqa_attention_kvarn.cuh:13`). **64 tokens per page.**
  `packed_pages = max_visible_keys/64`.
- **Staged shadow size** (`include/ninfer/ops/kvarn_workspace.h:96-119`):
  `per_layer_page = 2·256·64·kv_heads·2 = 128 KiB` (K+V, BF16, `[D,G,H,P]`).
  With 16 text layers + 1 MTP = 17 layers, `per_page = 17 × 128 KiB ≈ 2.125 MiB`.
  `stage_pages = min(pool_pages, kKvarnStagedBudgetBytes / per_page)`.
- **The wall lever** is `kKvarnStagedBudgetBytes = 1 GiB`
  (`kvarn_workspace.h:30`). At 2.125 MiB/page this yields **~482 pages ≈ 30.8k
  token wall**. To cover 40k → ~625 pages ≈ 1.33 GiB; 80k → ~1250 pages ≈ 2.66 GiB;
  250k would need ~8.3 GiB (stays beyond-wall). Raising the budget is the
  **cheapest** lever (no kernel change, just VRAM carve-out).
- **Dispatch** (`src/targets/qwen3_6/impl/runtime/text_context_impl.h:318-395`,
  `kvarn_attend_text`/`kvarn_attend_mtp`): `need_pages = tile_page>=0 ?
  tile_page+1 : committed_pages`; if `need_pages <= staged_pages` → stage the
  open-page tile and run on the **staged BF16 view**; else assemble
  `GqaKvarnTail{packed_pages, tail_count, staged_k/v}` and run on the
  `KVARN_K4V2` cache.
- **Launcher** (`src/ops/launcher/gqa_attention_kvarn.cu:16-143`): for
  `tokens > kKvarnSmallTMax(1)`, passes `packed_pages + (tail_count>0)`,
  allocates `k_temp/v_temp` **in the caller's per-call `workspace.scope()`**
  (`gqa_attention_cached` wrapper, `src/ops/wrapper/gqa_attention.cpp:496`),
  materializes, then T≤6 → `gqa_attention_cached_small_t_launch`, T≥7 →
  `gqa_attention_prompt_attention_launch`. For `tokens==1` → decode_split+merge.
  **The temp is discarded after every call** → full re-materialize per round.
- **Materialize kernel** (`src/ops/kernel/gqa_attention_kvarn_flash.cuh:33-122`):
  grid `(n_tiles, kv_heads)`, one CTA per (tile, kv_head). `kb < staged_pages`
  → vector transpose-copy from the shadow; `kb < packed_pages` → warp dequant
  + FWHT from codes; last tile → element-wise tail (+ zero-fill keys ≥ tail_count).
- **Pass-2 kernels:** BF16 FA2 flash (grid `q_blocks×q_heads`, 768 CTAs) and
  TC split-K small-T (proven, `src/ops/launcher/gqa_attention_decode.cu:385`).
- **`gqa_kvarn_stage_pages` clamps** to `staged_pages`
  (`src/ops/kvarn/kvarn_workspace.cpp:454`, `min(page_count, staged_pages)`);
  a `prefix_len > staged_pages` hit never overflows — good, no change there.
- **GDN checkpoint** (`src/runtime/tp2/tp2_backend.cpp:~989`) captures once per
  request; it only helps cross-request reuse, not within-request per-chunk cost.

## 4. Measurement methods (REQUIRED before/after)

Use the SAME method for baseline and post-change; never compare a single run.

1. **Pass-1 attribution harness** (`tests/bench_kvarn_attention.cu`): time
   materialize (pass-1) vs flash/small-T (pass-2) for `packed_pages ∈ {482,
   625, 964}`, `T ∈ {1,4,16,512}`, `tail_count ∈ {0,1,31}`. Record GPU clocks.
2. **Live decode guard** (`tools/bench/decode_guard.sh`): 3-run median, MTP-on,
   at 10k and 40k and (new) 80k context, 48 generated tokens.
3. **Live pp probe** (`tools/bench/pp_probe.sh`): 3-run median pp tok/s at
   40k/60k/88k; report `usage.prompt_tokens` (never char estimate).
4. **Battery closeout** (`bash tools/ops/run_ci.sh --full`): int8 T1–T11+T12,
   KVarN battery T16–T19 @250k (incl. T19 ≥450), T14 zone.
5. **Pivot table** §8: every number with the byte/usage figure, the config, and
   `nvidia-smi --query-gpu=clocks.sm,memory.used` recorded.

## 5. Design decisions (FINAL — do not re-litigate)

1. **Keep the BF16-attention substrate.** Direct-read prefill loses (docs/71:
   spill-bound, 2243.9 vs 37.2 ms @2048q/60k). Do not replace the flash/small-T
   pass-2 with a from-codes direct-read for prefill.
2. **Order of work:** raise the wall first (§6 step 2 — cheap, low risk, high
   payoff, no kernel change), then incremental materialization (§6 step 3 — the
   structural fix), then fusions (§6 steps 4–5 — conditional on measurement).
3. **Incremental materialization lives in the persistent arena** (like the
   staged shadow), **not** the per-call `work_` scope — otherwise it is freed
   every call and buys nothing.
4. **`kKvarnStagedBudgetBytes` is the single wall knob.** Do not hand-tune
   `stage_pages` at the call site; change the budget (and/or the `per_page`
   layers count if MTP is off) and let `kvarn_staged_page_capacity` do the math.
5. **Determinism is non-negotiable.** The split/merge ordering and the
   fixed-order partial merge must stay byte-deterministic. Any new kernel must
   reproduce the existing summation order.
6. REJECTED: lowering the BF16 shadow to a lossy q8 to double the wall — the
   flash reads BF16 and the shadow IS the attention input for in-capacity;
   changing its precision changes outputs and likely breaks the T18 restore
   equivalence. Revisit only if the flash were made lossy-aware.
7. REJECTED: full direct-read replacement of the beyond-wall prefill — see DI(1).
   Revisit if the MMA/vectorized restructure (step 5) lands and closes the
   spill gap.

## 6. Execution order (commit + test each step before the next)

**Testing standard:** a step touching a server-facing path is NOT done on unit
tests alone — spin up the server, execute the real call sequence (chunked
prefill, MTP rounds, request lifecycle) end-to-end. Kernel/unit isolation is
necessary, never sufficient (the 2026-08-24 "append jumped pages" defect passed
every kernel test and failed on the first live request).

### Step 1 — Attribute the beyond-wall cost (baseline, no code change)
Run the harness §4.1 + decode_guard §4.2 at 40k/80k on the **current** build;
record per-call pass-1 vs pass-2 ms and the wall page count. Confirm the
hypothesis: pass-1 (materialize) dominates and scales with `packed_pages`,
not with `T`.
**Tests:** the numbers table (§4.5) as a committed `results/attention_path_baseline.md`.
No pass/fail — this is the reference all later steps compare to.

### Step 2 — Raise the staged wall to cover target context (config/budget)
Increase `kKvarnStagedBudgetBytes` so `stage_pages` covers the decode target:
40k → ~625 pages → ~1.33 GiB. Only touch `kvarn_workspace.h:30`. Confirm
`kvarn_staged_page_capacity` yields the target and startup logs the new wall.
**Tests (must pass before moving on):**
- `/usr/bin/ctest -R ninfer_kvarn|ninfer_tp2_budget` green.
- decode_guard at 40k: decode tok/s **not** worse than baseline (expect equal or
  better, since beyond-wall materialize disappears for ≤wall context), 3-run median.
- Live: a 40k-token request serves with the wall covering it (no `beyond-wall`
  materialize in serve.log for that range); T19 pp ≥450 unchanged; battery
  `--mode ci` green.
- Record VRAM delta (expected ≈ +0.35 GiB/rank for 40k) and confirm it fits the
  16 GB rank with the KVarN 250k config (budget preflight unchanged).

### Step 3 — Incremental materialization (the structural fix)
Make the beyond-wall temp **persistent and delta-only** instead of
re-materialized every call:
- Keep a persistent BF16 overflow buffer (arena, like the staged shadow) that
  holds the pages above the wall (or extend the shadow upward so there is no
  "overflow" seam).
- On each call, materialize **only the new page(s) + the tail**, not the whole
  history. Subsequent calls reuse the already-materialized pages.
- Below-wall pages: eliminate the transpose-copy by aliasing (point the temp's
  `block_table` at the staged shadow for `kb < staged_pages`) **IF** the temp and
  shadow layouts are compatible (the in-capacity path already feeds the same
  flash kernel directly from the shadow — verify that compatibility first).
**Tests:**
- Per-call pass-1 time must drop from `O(packed_pages)` to `O(delta)` (harness
  §4.1, measure on the 2nd+ consecutive call after a warmup call).
- **Determinism:** byte-identical output vs the pre-step-3 build for the same
  request (reuse the A/B compare harness; 0 mismatch over a sampled sequence).
- **Correctness at the seam:** a prefix/length that lands exactly on a page
  boundary and one that lands mid-page — greedy output identical to a
  full-re-prefill reference; mid-page case must restore the BF16 tail (see
  docs/70 §4.2 bit-exactness gate).
- Live: multi-turn + long-prefill request sequence runs; re-run §4.2/§4.3/§4.4
  (decode guard, pp probe, `run_ci.sh --full`).

### Step 4 — (CONDITIONAL) Fuse the small-T MTP-verify path to read packed codes in-kernel
ONLY if step 3 is insufficient and measurement (step 1) shows pass-1 still
dominates for small T. For T∈{2..6} beyond-wall, a kernel that dequantizes the
visible pages and runs the TC MMA layout **in-kernel** (no BF16 temp) avoids
the full materialize. docs/71 ruled this out for **prefill** (spill-bound at
T=1); **MTP verify (T=4) is a different regime** — re-measure, do not inherit
the conclusion.
**Tests:** 3-run median decode tok/s at 40k MTP-on vs §6 step 3 result; T19
≥450; MTP acceptance within ±2pp of baseline; determinism A/B.

### Step 5 — (CONDITIONAL, D-19 deferred) MMA/vectorized decode_split
The single-token decode path (`decode_split`+`merge`) is latency-bound
(~20 GB/s effective, scalar FWHT). Restructure **warp-per-head + MMA fragments**
(the docs/71-identified but never-done optimization) to raise single-token
decode throughput. Only if single-token decode latency is a bottleneck.
**Tests:** decode_guard 3-run median at 10k/40k; byte-determinism with the
existing split/merge ordering preserved (fixed-order partial merge unchanged);
battery green.

### Step 6 — Cleanup (correctness + hygiene)
Port as small commits, independently testable:
- Replace the per-quad `__shfl_down_sync(0xffffffffu, …)` in the split-kernel
  score loop with the correct quad mask (formally UB today, empirically fine)
  — `gqa_attention_kvarn_decode_split.inc`.
- Correct the "~90 KiB" smem comment (actual `sizeof(KvarnDecodeShared)` ≈
  **100,328 B ≈ 98 KiB**, 1,048 B under the sm_120 cap) and add a compile-time
  `static_assert(sizeof(KvarnDecodeShared) <= kLimit)` — `gqa_attention_kvarn.cuh:426`.
- Consolidate `kKvarnKCodeBytes` (duplicated in `kvarn_workspace.h` vs
  width-derived in `gqa_attention_kvarn.cuh`) into one width-parameterized
  definition so future non-(4,2) widths cannot silently diverge.
**Tests:** existing kvarn unit tests green (`/usr/bin/ctest -R ninfer_kvarn`);
`--mode ci` green; no behavior change (A/B byte compare).

## 7. Constraints (non-negotiable)

- **Worktree only:** no edits in `/home/intel/ninfer/repo`, ever (§2). Commit per
  step to `wo/kvarn-pp` (or a fresh `wo/kvarn-attn-opt`) and push; merging to
  main is the main-side agent/user's job, not yours.
- **Live end-to-end before "done":** a step touching a server-facing path is not
  done until a real server runs it (§6 testing standard).
- **Do not change:** the k4/v2 format, the `V`/`K` page layout, the
  `GqaKvarnTail` field semantics, the block-table mapping, or any hard-gate
  threshold (T19 ≥450, decode guard ±2%).
- **GPU/server:** one live server at a time; port 8091 serves an active
  conversation — do not kill/restart it without explicit user permission. If a
  test needs a different config, use the swap protocol: ask the user →
  `pkill -x ninfer-serve` (exact name, NEVER `pkill -f`) → wait GPUs < 500 MiB →
  launch per LAUNCH.md → run → restore.
- **Do not build while another agent is building/using the tree.** Coordinate;
  if the build dir is in use, stop rather than race.

## 8. Definition of done

1. Steps 1–6 committed to the worktree with passing tests at each step,
   including the live-path test for every server-facing step (§6).
2. **Live proof:** a fresh server launch with the new code serves a real request
   through the changed path (battery T16-style or the own live sequence), launch
   command + `nvidia-smi` clocks recorded in the report.
3. **Measured win:** §4 numbers table showing per-round pass-1 time reduced from
   `O(packed_pages)` to `O(delta)` (or the wall covering target context so
   beyond-wall materialize is absent), with no decode/pp regression and
   byte-identical outputs.
4. `results/attention_path_report.md`: baseline table (step 1), wall change
   (step 2) VRAM + tok/s, incremental-materialize per-call attribution (step 3),
   fusion results (steps 4–5 if run), and the full-battery closeout (T19 ≥450).
5. Report format when done: one paragraph per step + the key numbers table.

## 9. Risks

- **VRAM squeeze:** raising the wall (+~0.35 GiB/rank to 40k) and the persistent
  overflow buffer compete with KV capacity on 16 GB ranks. Re-run the budget
  preflight and cap the wall so startup passes; if 40k+ does not fit, keep the
  wall at ~31k and rely on step 3's delta-only materialize (which helps
  regardless of the wall).
- **Seam correctness:** a mid-page prefix/length after incremental
  materialization can silently produce wrong output — hence the bit-exactness
  gate is mandatory (docs/70 §4.2).
- **Layout incompatibility:** if the temp and the staged shadow layouts differ,
  the below-wall post-aliasing (step 3) can't be zero-copy; fall back to a
  persistent transpose buffer or keep the copy for below-wall pages and only
  make the above-wall delta-only.
- **Determinism drift:** reordering the dequant/summation in any new kernel can
  flip near-tie tokens; always A/B byte-compare.
