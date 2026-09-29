# W7_KVHOSTED_desk.md

**MISSION: KV cache (or its cold portion) hosted in system DRAM, OFF BY DEFAULT (env-gated `NINFER_KV_HOSTED=1`), WORKED AND TESTED — target: help long-context prefill (VRAM headroom / beyond-VRAM context). Phase 1 = map existing KV + design doc + standalone hipcc parity/perf cell (NO tree build). Phase 2 = tree integration only after V-arm lands its build AND the cell clears the pre-registered bar.**

Desk: hosted-KV, Team Red (AMD V340L, gfx900/GCN5, ROCm 6.2, NVFP4@TP4).
Worktree: /home/chris/worktrees/amd-wo-w7-body (branch amd/wo-w7-body).
Disk at desk open: 3.4G free (df -h / 2026-09-19 05:28 CDT). Phase 1 needs no tree build.
Build-claim protocol active: W7_VARM_INTEGRATION_desk.md does NOT exist as of desk open
(checked 2026-09-19 05:28 CDT, find across worktrees) — no V-arm build claim to wait on;
if it appears, re-check mtime before any cmake --build.

## LOG (append after every step)

- 2026-09-19 05:28 CDT — Desk opened. Worktree confirmed on amd/wo-w7-body. Disk 3.4G.
  V-arm desk file absent → no build contention. Phase 1 begins: KV map.

- 2026-09-19 05:40 CDT — KV MAP COMPLETE. Findings:
  - Target: qwen3_6_27b (serving line qwen3_8_27b_nvfp4 @ TP4). 64 layers HYBRID: 16 full-attn
    + 48 GDN linear (no paged KV) + 1 MTP attn layer. Full-attn geometry: kv_heads=4, head_dim=256,
    query_heads=24. (src/targets/qwen3_6_27b/impl/config.h)
  - Allocator: ONE persistent DeviceArena per rank (program_impl.h:271 backing =
    persistent.alloc_bytes(plan.persistent.bytes, 256)); decoder KV pool planes bind from it.
    Entire KV is VRAM-resident today, sized at plan time.
  - Layout (src/core/paged_kv_cache.h, decoder_state.cpp plan_cache): page = 64 tokens
    (kPagedKVPageSize); KVarN tiers = 2 U8 code planes/layer (K channel-major D*kb/8 B/tok,
    V token-major D*vb/8 B/tok) + fp32 scale side-table [layers, heads, pages, 1152]
    (1152 = K s_col256+zp256+s_row64 + V same = 4608 B/tile = 72 B/token/head).
  - Per-token math (kvarn k4v2): K 128 B + V 64 B + scales 72 B = 264 B/tok/head/layer
    -> 16,896 B/token text KV (4 heads x 16 layers); +MTP 1,056 B/token = 17,952 B/token total paged.
    BF16 tier reference: 65,536 B/token (3.9x kvarn).
  - Append (prefill): gqa_kv_append_kvarn[_and_commit] (src/ops/kvarn/kvarn_workspace.cpp)
    -> scatter bf16 tile (kvarn_workspace.cu) -> per-(layer,head,page) quantize into plane
    slabs + scale tile -> stage/commit via block_table.
  - Read (attn): gqa_attention_kvarn*.cuh (prefill) + gqa_decode_slice4_kvarn/k5v4 (decode)
    read U8 codes + 1152-float scale tiles through PagedKVLayerView via block_table, dequant
    in-kernel.
  - EXISTING host affordance: PagedKVPool::copy_pages_to_host/from_host (docs/156 §18 P2
    host-KV safety net) — page-major host mirror, PageMajor pools only. Natural substrate.
  - Per-rank note @ TP4: kv_heads=4 over world=4 -> 1 kv head/rank -> 4,224 B/token/rank text KV.
  Measured BW anchors on THIS box: SHM-staged 3.14-3.17 GiB/s (W7_ksplit log, canAccess=0 all
  pairs); pinned PCIe 10-12 GB/s class; device HBM ~full BW (cell will measure all three).
  NEXT: design doc WO_KV_HOSTED_desk.md (mechanism pricing + pre-registered gate).

- 2026-09-19 05:45 CDT — DESIGN DOC BANKED: docs/amd/WO_KV_HOSTED_desk.md written; 
  -> docs/amd/WO_KV_HOSTED_desk.md. Contents: KV budget table (k4v2 17,952 B/token total paged;
  4,224 B/tok/rank @TP4; sweep 10k-500k), three mechanisms priced (mapped-pinned zero-copy /
  page-spill+prefetch / allocator overflow), decode tax stated per mechanism (UNAVOIDABLE:
  C x 4,224 B / BW per token; 38.4 ms/tok @100k cold @11 GB/s; ~35-40x device), RECOMMENDATION
  = (a) mapped-pinned zero-copy cold-KV tail, SELF-BAR pre-registered (parity bit-exact
  mandatory; hosted BW >= 6.0 GB/s @200 MB; 2k-chunk tax <= 5% of chunk wall), Phase 2
  integration gate pre-registered (env arm, capacity-adds-only, RED/GREEN rows, A/B ordinal,
  600+ decode tax, context sweep, BANK-BEFORE-RELINK, plog chain 0d4c30c3272d3092).
  NEXT: cell tools/v340l/w7_kvhosted_cell.cu, then run under HIP_VISIBLE_DEVICES=2.

- 2026-09-19 05:55 CDT — CELL RUN + BANKED (run1, die 2, HIP_VISIBLE_DEVICES=2): PARITY
  bit-exact at 10/50/100/200 MiB both kernel shapes + slab memcmp; hosted mapped-pinned read
  6.33-6.50 GB/s (link-bound: byte-load == vector-load BW; dev arm 134-343 GB/s, matches box
  read ceiling ~368 GB/s from PK_row); clocks mclk 945 top, sclk 1500 MHz TOP sampled mid-run.
  BAR: GO (parity PASS, 6.50 >= 6.0 PASS, prefill tax 1.30 ms/2k-chunk = ~0.08% vs 5% bar).
  Decode-tax unit MEASURED: 32.25 ms/generated-token per 200 MiB/rank hosted (~0.65 ms/1k tok).
  Overlap probe (mechanism b premise): FALSE on this stack (21.8 vs 22.3 ms sum) — no
  copy/compute concurrency on Vega 10 HIP. Banked: results/amd/coherence/
  W7_kvhosted_cell.cu, W7_kvhosted_cell_run1.log, W7_kvhosted_row.txt. Design doc §5 results,
  §6 Phase 2 = NOT STARTED (V-arm desk file still absent; its build has not landed).
- 2026-09-19 05:56 CDT — Health check + final report next. No tree build was run (Phase 1
  standalone cells only); no serving process touched; no build claim needed (no cmake --build).

- 2026-09-19 05:58 CDT — V-ARM DESK APPEARED MID-SESSION: docs/amd/W7_VARM_INTEGRATION_desk.md
  (mtime 05:46) claims the serving window from 05:3x CDT; its BUILD landed 05:45 (incremental
  GREEN, BANKED /home/chris/artifacts_bin/ninfer-serve_c618d356f0cdc401.bin) and MEASUREMENT
  OPENED 05:46 (both legs bin c618d356f0cdc401, control first). No BUILD CLAIM conflict: this
  desk ran ZERO cmake --build (Phase 1 was standalone hipcc only — the protocol held).
  PHASE 2 WAKE CONDITION: V-arm desk's WINDOW EXIT lands in its file (mtime check first) ->
  then execute WO_KV_HOSTED_desk.md §4 gate. Precondition state: (1) V-arm build landed = MET;
  (2) window free = NOT MET (measurement in progress).
- 2026-09-19 05:59 CDT — HEALTH AT END: NO ninfer-serve process running at desk close; this
  desk NEVER retired or booted a server (serving window belongs to the V-arm desk; last
  serve_10k.log write predates this desk at 02:06). No tree build, no pkill, no GPU grant used
  (cell ran on die 2 as a microbench per law, ~90 s total). Disk at close: 3.3G free (delta
  = docs/results text only; cell binary lived in /tmp). PHASE 1 COMPLETE.
