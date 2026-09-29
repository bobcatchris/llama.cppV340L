# WORK ORDER: extend the OPTRACE partition to the kvarn route (the map that gates every future window)

Owner: desk agent (slot A). Phase 1 GPU-free design; Phase 2 = ONE instrumented validation
boot, QUEUED behind the coordinator's explicit GRANT line in this file.

## CONTEXT (banked; cite, do not re-derive)

- PLOG-087 / results/amd/census/CENSUS_ROW_union_kvarn_20260920.md: on the PROMOTED line
  (union bin 058a7b85859c5cd0, kvarn_k4v4@65536/ws512), the OPTRACE columns sum to ~305 ms
  of a ~670 ms steady wall — **~54% of wall is untraced device work**. The tracer's
  gemm/ar/body sections describe the bf16 route; the kvarn attention/dequant kernels
  (slice4/slice6 launchers, gqa_attention_kvarn*.cuh family, the prologues) sit outside
  every section. Per PLOG-085's rule, NO column-share claim is valid until this is fixed.
- The tracer: NINFER_PREFILL_OPTRACE (depth 1..3) instruments the prefill loop sections —
  find the wall_begin/wall_end pairs (src/targets/qwen3_6/impl/runtime/text_context_impl.h
  hosts [PREFILL-HOST]; the [PREFILL-OP] gemm/ar/body sections live in the tp2 prefill path —
  grep `PREFILL-OP` and follow). The emit format is `[PREFILL-OP] rank= chunk= tok= t= wall=
  gemm= ar= body= gap= ngemm= gemm_over=`.
- The census raw extract (req3-req7 steady rows) is the ground truth your partition must
  reconcile against: wall 661.1-699.2 ms, traced 305 ms, untraced ~356 ms.

## PHASE 1 — DESIGN + IMPLEMENT (GPU-free)

1. Map the kvarn prefill call path: which kernels execute per chunk on this route
   (launcher -> slice4/slice6 -> prologues/dequant -> mma), and which of them the current
   sections already wrap vs miss. Cite file:line for each boundary.
2. Extend the partition: add explicit sections (candidate: a `kvarn` column, or split
   kvarn-attn vs kvarn-dequant) so that **columns sum to ~wall** on the kvarn route while
   the bf16 route's output stays byte-identical to today's. Constraints (hard):
   - NINFER_PREFILL_OPTRACE unset => byte-identical behavior and output (kill-switch law).
   - Emit format stays parse-compatible: existing fields unchanged, new fields APPENDED
     (the census row parser and the firstchunk tooling must keep working).
   - No estimated VRAM refusals; no behavior change outside the env-gated trace.
3. Commit the implementation with the design map (file:line boundaries) in this WO's log.

## PHASE 2 — VALIDATION BOOT (queued; GRANT line required)

- Window claim per protocol (poll pgrep empty + GRANT line in this file + no open claims
  in WO_BODY_FUSION.md / WO_Q4KV_K4V4.md). ONE boot: promoted line + NINFER_PREFILL_OPTRACE=3,
  3x plen-2075 mt64 probes. **EXTRACT BEFORE RESTORE** (PLOG-087 sequencing error — the
  serve log truncates on restore; extract, bank, THEN restore + +3-min liveness).
- GATE (pre-registered): on the 3 probes' steady chunks, traced columns sum to 95-105% of
  wall on every request; bf16-route parity spot-check: one probe with OPTRACE=1 on the same
  boot shape emits the today-format line unchanged. PASS => bank the re-issued census map
  (results/amd/census/, superseding CENSUS_ROW_union_kvarn_20260920.md) + PLOG row drafted.
  FAIL => the partition report says which section still leaks, no re-boot without
  coordinator word.

## LAWS

- Own worktree + branch amd/wo-tracer (build ONLY there if a build is needed — the
  validation boot may use your own build bin banked per BANK-BEFORE-RELINK, or the
  coordinator rebuilds; coordinate in your log). Progress log newest-first after every
  step; resumable from this file alone. NO channel posts. df -h / before >1G writes.

## LOG (newest first)

### 2026-09-20 ~12:3x — PHASE 1 COMPLETE: kernel boundary map + kvappend/kvattn sections implemented, syntax-validated RC=0 x4 TUs (desk tracer)

**1. Census reconciliation (corrects the census table's internal reading, keeps its numbers).**
The raw OPTRACE lines were lost (PLOG-087 sequencing error), but the columns now reconcile
EXACTLY under this reading: census `wall` = [PREFILL-OP] wall; `gemm`/`ar`/`gap` = the
printed columns (gap = wall − layers = chunk head+tail); census `body` = the **12 NAMED
PBO classes** from [PREFILL-BODY]/[PREFILL-BODYSUM] (24.1 ms), NOT the printed body
column (= layers − ar − gemm ≈ 387 ms — the printed body can never equal the printed gap
when wall=667: 24.1+142+114.5 ≠ 643.6). Traced 305 = gemm+ar+named+gap (114.5+142+24.1+24.1);
**the untraced ~356 ms is [PREFILL-BODY]'s `unbr` class**: in-layer, non-GEMM, non-AR,
outside the 12 named pairs. HEAD's P1 arithmetic already had a bucket for it — what the
kvarn route lacks is NAMES inside that bucket.

**2. Kernel boundary map — kvarn prefill call path, per chunk (T=128, TP4, k4v4, single-seq,
eager — NINFER_PREFILL_GRAPH unset in the canonical env, so all pairs record).**
All text_context_impl.h numbers are POST-EDIT (this commit's file; pre-edit = minus ~114).
Chunk skeleton (src/targets/qwen3_6/impl/runtime/text_context_impl.h unless noted):
- wall pair: `wall_begin` :3850 .. `wall_end` :4237 (fin_mark :4083), drain at the next
  chunk's `run_layers` gate (:3596-3606 → begin_chunk) — wall = embed .. MTP finalize.
- `prefill_impl` plain run_layers call :3931 (graph arms :3947/:3961/:3967/:3998 unreachable
  with the env unset). `run_layers` :3506: layer loop :3671, `vt.layer_begin` :3680 /
  `vt.layer_end` :3738 → every in-layer op lands in `layers`; printed body = layers−ar−gemm.
- FULL layer x16 (attn_mix :2546 → attn_mix_tp :2633):
  - UNNAMED: input rmsnorm :2642.
  - gemm pair :2647-2649 (Variant::attention_projection_tp — nvfp4 tiled).
  - aunpack pair :2661-2663; qknorm pair :2692-2696 (rmsnorm q/k + rope).
  - **KVERN ARM :2799-2801 → kvarn_attend_text :453 — WAS UNPAIRED (the census hole):**
    - `gqa_kv_append_kvarn_and_commit` :476 → ops/kvarn/kvarn_workspace.cpp:589. Per page-run:
      `prepare_page_for_append` :567 (commit_completed :571/:628, hydrate :586) +
      `gqa_kv_append_kvarn_with_host_pos` :625→:364 (`kvarn_scatter_tile_launch` :426).
      **The single-seq call site passes NO host_positions → positions D2H (512 B, pageable)
      + `cudaStreamSynchronize` PER FULL-ATTENTION LAYER (kvarn_workspace.cpp:601-608) —
      16 full-pipeline drains per chunk INSIDE layer bodies.** Page commits are sync-free
      on this route (host_block_table mirror is bound: decoder_state.cpp:188,
      paged_kv_cache.h:55/:269 → kvarn_workspace.cpp:63-68 host path).
    - optional `gqa_kvarn_stage_from_tile` :483 (shadow rebuild; kvarn_workspace.cpp:634).
    - `gqa_attention_cached` :490 (staged arm) / :518 (direct arm) → wrapper
      ops/wrapper/gqa_attention.cpp:932 → `gqa_attention_kvarn_cached_launch`
      ops/launcher/gqa_attention_kvarn.cu:247, T=128 → materialize+flash route (:349-461):
      `gqa_attention_kvarn_materialize_kernel_w` :382 (k4v4 tier dequant packed→BF16 temps;
      legacy k4v2 would take :368), HIP `kvarn_vtemp_bf16_to_f16_inplace_kernel` :413,
      `gqa_attention_prompt_attention_launch` :458 (FA2 over the temp).
  - agate pair :2864-2866; gemm pair :2872-2874 (o_proj tp_gemv); ar pair :2878-2880.
- GDN layer x48 (gdn_mix :3714 → gdn_mix_tp :2884): gate pair :2921, ar pair :3079-3082,
  gemm pair :3093-3095 (gdn_input_projection_tp), upk :3135, conv :3157, gbet :3192,
  dn :3208, gnorm :3225, gemm :3235-3237 (out_proj), ar :3288-3291 — all named.
- mlp_tail x64 (:3468): mnorm pair :3475, gemm pair :3487-3489 (post_mixer_tp; mact pair
  inside variant_kernels.cpp), ar pair :3494-3496.
- NOT bracketed anywhere: MTP-tail kvarn appends (:894/:2026 sites) — they sit in the final
  chunk's tail, already named by the [PREFILL-FIN] prep bracket at depth>=1; out of scope.
- Batched attend (kvarn_attend_text_batched :524) deliberately NOT instrumented: the
  launcher throws for width>6 (gqa_attention_kvarn.cu:795-799 "batched prefill is not
  wired; prefill per lane") — unreachable during prefill chunks, pairs would be no-ops.
- New kv sites (this commit): kvarn_attend_text kvappend :475-476..:488 / :516,
  kvattn :489-492 (staged arm) / :517-519 (direct arm); record methods :1437-1454
  (struct PrefillOpTrace :1248); [PREFILL-OP] append print :1587; [PREFILL-KV-SUM] :1549.

**3. Partition extension (implemented this commit).** Two new sections, ZERO new events:
- `kvappend` = gqa_kv_append_kvarn_and_commit + gqa_kvarn_stage_from_tile (the KV write
  path incl. the per-layer D2H+sync drain).
- `kvattn` = gqa_attention_cached (materialize + v-temp cast + FA2 flash = the dequant/
  attention read path).
Implementation: PrefillOpTrace grows `kv_begin/kv_end` recording into TWO FIXED SLOTS of
the already-paid-for gemm event bank (kKvSlotBase=315, below the fin marks 317-319; bank
regions now disjoint gemm [0,315) / kv [315,317) / fin [317,320); gemm recording capped at
kKvSlotBase so a full bank skips loudly into gemm_over instead of stomping kv/fin —
unreachable on measured shapes, ngemm max 192). No cudaEventCreate (W7 RED evidence: extra
event banks wedged the ~1 MiB-headroom boot). Sites: kvarn_attend_text both arms, via
`kvarn_trace_kv_begin/end` free functions (declared above :449 where the struct is still
incomplete, defined after potrace_for; ids pinned by static_assert).
- Kill-switch law: every site behind `pt.active` + `gemm_suspend` (KS4 graph grammar) —
  NINFER_PREFILL_OPTRACE unset → no records, no prints, no device work, byte-identical.
- bf16-route parity: the kvarn arm never runs on the bf16 route → kv_mask 0 → [PREFILL-OP]
  byte-identical at EVERY depth; OPTRACE=1 parity probe gate satisfied.
- Emit: fields APPEND after `gemm_over=` only when the chunk recorded >=1 kv pair:
  ` kvappend=%.3f kvattn=%.3f` (+ ` kv_over=%d` only when >0). New request-line
  `[PREFILL-KV-SUM] rank= n= mean_ms kvappend= kvattn=` (means over kv-recording chunks);
  [PREFILL-SUM] itself untouched. Reconciliation: wall = gemm + ar + body + gap still holds;
  body = kvappend + kvattn + (12 named classes) + true-slack.

**4. Compile validation (syntax-only, exact -O3 flags from amd-wo-fuse/build-lean
compile_commands.json, paths rewritten to this worktree, -fsyntax-only, no objects
written — disk 6.8G free, no >1G writes):** tp2_backend.cpp RC=0; qwen3_6_27b variant.cpp
RC=0; qwen3_6_27b variant_kernels.cpp RC=0 (0 warnings); qwen3_6_35b_a3b variant.cpp RC=0.
All remaining diagnostics are the pre-existing -Wunused-result class at untouched sites
(vram_trace.h, VerifyLayerTrace :1128-1173).

**5. Falsifiable prediction for the validation boot (pre-registered; 3x plen-2075 mt64,
OPTRACE=3, steady = chunks 2+, rank 0):**
- Byte math bounds the kvarn kernels' pure device time EXACTLY (TP4 rank geometry
  Gqa27Tp4Geometry = 6q/1kv, group 6 — src/ops/kernel/gqa_attention_geometry.cuh:36):
  materialize temps are k+v BF16 {256, 64, kv=1, ~35 tiles} ≈ 2.2 MB written per full
  layer per rank, vtemp cast + flash re-touch ~3.4 MB → ~5.6 MB x 16 layers ≈ 90 MB
  traffic/chunk ≈ 0.2-3 ms even at pessimistic bandwidth. **kvattn > 20 ms/chunk would
  already be a surprise; the H2 falsifier threshold stays 200 ms** (triple margin).
  The append's quant kernels write 2 pages x (8 KB K + 8 KB V codes + scales)/layer —
  negligible. The only structurally new per-layer host crossing on this route is the
  positions D2H+sync (kvarn_workspace.cpp:601-608).
- PRIMARY (H1): the 16 mid-layer positions D2H+stream-sync pipeline drains own the 356 ms
  → **kvappend steady mean ∈ [150, 400] ms/chunk AND kvappend > 3 x kvattn; kvattn <= 60 ms**;
  gemm ~108-125, ar ~136-152, gap ~20-30 (census reproduction within noise).
- H2 (falsifier): kvattn > 200 ms → the dequant/flash device time owns the wall → target =
  materialize route, not the append sync.
- H3 (falsifier): kvappend < 50 AND kvattn < 50 → the time is intra-layer slack OUTSIDE the
  kvarn block (GDN trio internals / launch pacing) → next bisect = depth-3 DN3 splits +
  per-op census; NO second boot without coordinator word.
- Secondary (observation, not gate): ar may read slightly UNDER census (drains re-pad the
  4-rank lockstep so ranks arrive at ARs together).

PHASE 2 remains QUEUED behind the coordinator's explicit GRANT line in this file.

### 2026-09-20 11:3x — COORDINATOR CLOSE: Phase-1 design MERGED to amd/main (kvappend/kvattn sections, append-compatible, byte-identical unset+bf16). PHASE 2 (validation boot) MOOTED by the PLOG-090 pivot (smaller model, NVFP4 dropped) — the boot would validate a partition for the 27B's kvarn route, which the pivot retires. The design is banked for the next model's stack: IF that model runs on ninfer with a quantized-KV route, this partition extends to it by the same recipe; IF bf16-only, the stock sections are already valid (PLOG-087's gap was kvarn-route-only). Desk closed, worktree releasable.
