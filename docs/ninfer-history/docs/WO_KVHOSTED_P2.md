# WO_KVHOSTED_P2.md — hosted-KV Phase 2 integration desk checkpoint (Team Red, W7)

**Desk:** Phase 2 tree integration of the mapped-pinned cold-KV tail behind
`NINFER_KV_HOSTED=1` (OFF by default) + G-KV2 VRAM-funder measurement (chunk ladder unlock).
Design authority: docs/amd/WO_KV_HOSTED_desk.md (§5 Phase-1 cell results, §6 Phase 2 gate).
Worktree: /home/chris/worktrees/amd-wo-w7-body (branch amd/wo-w7-body). NEVER build in
/home/chris/dual_5060_ti_ninfer. Serving retire = EXACTLY
`pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"`; canonical restore =
`bash /home/chris/serve_fast.sh` (NVMe model) + health, at every close.

## GATES (from dispatch, pre-registered)

- **G-KV1 parity:** hosted vs device KV, patterned data, >=8k-token synthetic — relL2 <= 1e-2
  class + behavioral BLUE/stop.
- **G-KV2 VRAM-FUNDER (elevated):** NINFER_KV_HOSTED_TOKENS sized for a 20k-token tail —
  report per-die VRAM freed (preflight composition line) and whether a chunk-256 boot with
  NINFER_WORKSPACE_MIB=256+ COMPLETES warmup + serves without bad_alloc (PLOG-069 unlock).
- **G-KV3 perf:** ordinal-paired serve-legs (PLOG-060 design, fresh boots): prefill walls 10k
  hosted-vs-device (gate within 5%); decode t/s at mt=600 — decode tax reported honestly.
- **G-KV4 no-default-impact:** unset env boots byte-identical + 10k grade PASS.
- All gates pass => runbook gains the env line (OFF by default), BOOT_BATTERY, PLOG row
  (chain head 521ced8718cb2dc3). Any fail => bank everything, default stays off, report.

## BUILD CLAIM LEDGER (append "BUILD CLAIM <ts>" before any cmake --build)

- BUILD CLAIM 2026-09-19T18:2xZ — hosted-KV Phase 2 integration build #1 (incremental,
  target ninfer-serve, -j8). Other desks checked: WO_ONESHOT_ROOTLOOK.md mtime 12:10 CDT
  (unchanged since survey; canonical serve idle), WO_DECODE_LANDING.md 10:49 (blocked,
  not building), no active claims anywhere.

## WINDOW CLAIM LEDGER (append "WINDOW CLAIM <ts>" before any boot; check
docs/amd/WO_ONESHOT_ROOTLOOK.md mtime + tail first — that desk cycles boots)

- WINDOW CLAIM 2026-09-19T18:35Z — root-look desk CLOSED 13:07 CDT (gate verdict FAIL 0/5,
  one-shot route multi-factor blocked; box left in canonical posture, health PASS). No
  active desks. Boot battery begins: 4 banked-bin legs (A off-arm, B hosted-arm, B3 funder,
  B4 funder-control), retire law between legs, canonical restore at close.

## PROGRESS LOG (append after every step — LAW)

- 2026-09-19 18:2xZ — BUILD #1 GREEN (first try, incremental -j8; warnings pre-existing
  class only). Binary sha256 9bc7d7e1f764e7cb5a657b0eced764099e0ce8a66d3a37c17ebda1187f99ffbc
  -> BANKED /home/chris/artifacts_bin/ninfer-serve_9bc7d7e1f764e7cb.bin (boots from the bank).
  BANK-BEFORE-RELINK verified: prior build-tree bin eea46f6766f5b9d9 was already banked
  (12:04 copy) BEFORE this build relinked. v340l whitelist parity guard: OK 220/220.
- 2026-09-19 18:1xZ — implementation complete. Diff surface (10 files + 1 new):
  include/ninfer/types.h (EngineOptions.kv_hosted_tokens); src/core/paged_kv_cache.{h,cpp}
  (PagedKVHostedMap, spec.hosted_pages, plane shrink at plan, mapped-pinned slabs owned by
  PagedKVPool, zero_pages run-split, hosted_map view fill); src/runtime/tp2/tp2_budget.h
  (hosted_kv_tokens member; required_bytes + max_context_fitting subtract via kv_unit seam);
  src/runtime/tp2/tp_engine.cpp (env parse NINFER_KV_HOSTED/_TOKENS with BF16-only guard +
  boot print, probe+budget set, composition line hosted clause — OFF log shape unchanged);
  layouts.h/layouts_impl.h (planning inputs + plan impl + spec carry);
  decoder_state.h (export header — spec field)/decoder_state.cpp (plan_cache hosted_pages
  with bf16-only guard, per-pool cap at pages-1, layer/batch view .hosted fill);
  ops/kernel/kv_hosted_route.cuh (NEW — kv_hosted_page_base routing helper);
  gqa_attention_decode_bf16_gfx906.cuh (decode simt: resolve_k/resolve_v + append-branch
  routing), gqa_attention_prefill_bf16_gfx906.cuh (simt + splitk: per-tile slab resolve),
  gqa_attention_prefill_bf16.cuh (fill kernel: routed K/V stores); launchers
  gqa_attention_decode.cu + gqa_attention_prefill.cu (4 launch sites pass cache.hosted).
  KVARN kernel family deliberately NOT routed (bf16-only arm, guarded at plan).

- 2026-09-19 17:33-17:4xZ — DESK OPEN. Survey done: worktree amd/wo-w7-body clean of build
  claims (none active in any desk file); W7_FINALIZE_DESK_RESUME.md stale (09-18);
  WO_DECODE_LANDING.md blocked on the one-shot wedge (its edit is dead code on ring, A/B
  untestable — not building); oneshot root-look desk last wrote 12:10:54 CDT (V1-V6 validation
  plan published), a CANONICAL serve (bin 2c8901d3, ring posture) has been up since 12:16 at
  0% GPU = idle healthy restore; box quiescent. Disk 11 GB free. Baseline build-tree bin =
  eea46f6766f5b9d9 (root-look C2 cure; banked copy verified present in artifacts_bin listing
  to be re-checked before any relink — BANK-BEFORE-RELINK).
- Baseline facts captured (w7_pa_256_off.log): preflight required 8159 / usable 8160 MiB
  (1 MiB slack); composition/rank: weights 5131 + decoder_fixed 1360 + ws 96 + staging 200 +
  arena_padding 160 = fixed 6947 | hidden_state 5 + kv(34816 B/t x 36352 tok + mtp 2) 1207.
  Pool: 569 kv pages/rank (cap 36358 tok), page=64 tokens. Hosted-tail sizing: 34,816 B/t/rank
  => 20k-token tail ~= 664 MiB/rank freed (funds ws 96->256 (+160) with ~4x margin).
- NEXT: study pool/append/read pointer paths (paged_kv_cache.cpp, kvarn_workspace, gqa
  kernels), find minimal-diff routing point; then implement env-gated cold-tail split.

## DESIGN (locked 2026-09-19 18:0xZ, after code study)

**SERVED LINE FACT-CORRECTION (changes the kernel surface):** the canonical line
(qwen3_8_27b_nvfp4, bin 2c8901d3, serve_10k.sh — no --kv-cache flag) runs **BF16 KV**
(default KvCacheStorage::BFloat16), NOT kvarn k4v2. Preflight composition
`kv(34816 B/t x 36352 tok)` = 34 head-layer units x 1024 B (BF16 tier constant,
tp2_budget.h:234). T3 convention: the V plane stores fp16 BITS in a bf16-typed plane
(fill kernel converts; attention reads __half* — gqa_attention_prefill_bf.cuh:58).
World=4 geometry = Gqa27Tp4Geometry (6 q-heads, 1 kv-head/rank). Served kernel surface
(all HIP/gfx900):
  WRITE  gqa_attention_prefill_fill_bf16_kernel        (gqa_attention_prefill_bf16.cuh)
  READ-p gqa_attention_prefill_simt_bf16_kernel        (gqa_attention_prefill_bf16_gfx906.cuh)
         + splitk_partial variant (NINFER_GQA_SPLITK, default OFF)
  READ-d gqa_attention_small_t_simt_partial_bf16_kernel(same gfx906 header;
         launched by launch_tc_partial_bf16, gqa_attention_decode.cu:150)
All address K/V pages via paged_kv_element_offset / gqa_cache_index (LINEAR in phys:
stride = 256*64*kv_heads elems). Decode resolves through 2 lambdas (resolve_k/resolve_v);
prefill computes tile_base per 32-key tile (ONE resolve per tile, zero inner-loop cost).
The KVARN kernel family is NOT on the served path (bf16 line) — kvarn kernels stay
un-routed; hosted arming is GUARDED to bf16-only at plan time (loud throw otherwise),
so kvarn paths can never see a hosted pool.

**Mechanism (matches Phase-1 design §2(a)):** hosted region = LOW physical page ids
[0, cold_pages) — oldest tokens (sequential take_pages => monotone ids per sequence);
device region = [cold_pages, full). At plan time (arm on):
  - pool planes planned at (full - cold_pages) DEVICE pages -> persistent arena bytes
    shrink (real VRAM freeing);
  - PagedKVPool owns mapped-pinned HOST slabs (cudaHostAllocMapped + HostGetDevicePointer),
    one per plane, cold_pages x page_bytes each;
  - PagedKVHostedMap {cold_k, cold_v, cold_pages, full_pages} rides the layer views;
  - kernels resolve per page: (phys < cold_pages) ? cold + phys*stride
    : hot + (phys-cold_pages)*stride — at cold_pages==0 this is EXACTLY today's
    address (intra + phys*stride), so the arm-OFF path is arithmetically identical;
  - budget seam (tp2_budget.h Budget): +hosted_kv_tokens member; required_bytes and
    max_context_fitting both subtract min(hosted_tokens, ctx+mtp+4) x kv_unit_bytes
    (ONE seam, both sites — docs/151 no-divergence rule preserved);
  - zero_pages splits runs at the cold boundary (mapped-host memsets);
  - env: NINFER_KV_HOSTED=1 (unset/0 = off, byte-identical), NINFER_KV_HOSTED_TOKENS
    (default 20000 when HOSTED=1, printed at boot);
  - MTP pool: same treatment, capped at its own page count.
Capacity-adds-only: no new refusal path on the OFF arm; ON arm only shrinks device
requirements (hosted alloc failures throw bad_alloc as any allocation would).
- 2026-09-19 18:2x-18:3xZ — LEG A v1 (off-arm, bin 9bc7d7e1, chunk 128 ws 96) RAN but is
  TAINTED + probe shapes were wrong: (1) the decode-landing desk WOKE mid-window (their file
  13:21, banked fdd9b68e 13:24, booted 13:27/13:28) — their -j8 build overlapped my P2/P3
  probes (contention) and a foreign boot followed my leg; (2) my varied-word prompts hit
  THINKING MODE (content empty, reasoning_content present — live-verified) and pattern
  continuation (finish=length, no BLUE); (3) my "10k" prompt was actually 15,369 tokens
  (words ~1.5 tok). v1 numbers banked as taint evidence only: P1 150.8s/len, P2 364.5s/
  15,369 tok, P3 64.2s gen600. CONFIRMED GOOD in v1: boot health, preflight lines BYTE-IDENTICAL
  to canonical (required 8159/usable 8160, kv 34816 B/t x 36352, decoder state 861 MB, 569
  pages) — the OFF arm is arithmetically transparent at plan time (G-KV4 boot-log evidence).
- v2 leg script: canonical alpha-context shapes (thinking-safe), token-calibrated
  (5,340w=8k tok / 6,500w=10k tok / 330w=600 tok), 180s drain gaps, double health verify.
  Interleave protocol agreed in WO_DECODE_LANDING.md (whoever's server is UP owns the box).
- Coordination note posted to WO_DECODE_LANDING.md (direct, no channel post).
- 2026-09-19 19:12Z — grace expired (fdd9b68e up 44 min, idle 0% GPU ~35 min, decode desk
  dark since 13:31). Retired per anchored law after notes in both desk files. BATTERY START:
  legs A2 (off) -> B (hosted) -> B3 (funder on) -> B4 (funder control refuse), each leg
  ends with the anchored retire so any woken desk can interleave.
- 2026-09-19 19:14Z — A2 leg KILLED by the decode desk's re-fire (they booted 4fb2d1ee at
  14:14 CDT; their boot script's anchored retire took my server mid-prefill at 83%). Their
  file mtime moved 14:12:38 = they read my notes; we are now both actively cycling. Leg v2
  gained a pre-boot BOX BUSY guard (abort, no retire, if any serving instance is up).
  Next A2 attempt at their next retire beat. A2 partial evidence (clean): preflight +
  decoder state lines byte-identical to canonical AGAIN (2nd clean OFF boot).

## PARKED 2026-09-19T19:48Z per principal order — resume checklist

PARK ORDER received via coordinator (principal): Phase 2 parked, bank cleanly, stop.
No further boots or gates were run after the order; the in-flight leg-firing poll was
killed BEFORE it fired (no boot occurred under the park). Box at park: CANONICAL posture
(bin 2c8901d3, booted 14:25 CDT by the decode-landing desk during its own cycle) — left
untouched; no server of this desk is running.

EXACT STATE AT PARK:
- INTEGRATED + COMMITTED (bd46de0e9 on amd/wo-w7-body): full mechanism —
  NINFER_KV_HOSTED=1 / NINFER_KV_HOSTED_TOKENS (default 20000) arm; PagedKVHostedMap;
  pool plane shrink at plan (real device KV freeing); mapped-pinned host slabs owned by
  PagedKVPool (cudaHostAllocMapped + device alias, freed in ~PagedKVPool); zero_pages
  run-split at the cold boundary; views carry the map; routed BF16 readers/writers
  (fill + SIMT prefill + SIMT prefill-splitk + SIMT small-T decode + decode append
  branch); budget seam hosted_kv_tokens in required_bytes + max_context_fitting;
  composition line hosted clause (OFF log shape unchanged); BF16-only plan guard.
- BIN BANKED: /home/chris/artifacts_bin/ninfer-serve_9bc7d7e1f764e7cb.bin
  (sha256 9bc7d7e1f764e7cb5a657b0eced764099e0ce8a66d3a37c17ebda1187f99ffbc), built
  13:11 CDT from this tree BEFORE the F1b desk's 13:22-13:34 edits (clean of them).
- UNTESTED (the entire gate battery): G-KV1 parity, G-KV2 VRAM-funder (chunk-256 +
  ws256 boot; the PLOG-069 unlock measurement), G-KV3 perf legs (prefill 10k + decode
  tax at mt=600), G-KV4 no-default-impact grade. Zero hosted-arm boots have EVER run —
  the integration is build-green only.
- PARTIAL G-KV4 EVIDENCE (off-arm, 2 boots, bin 9bc7d7e1): preflight + decoder-state
  lines byte-identical to canonical both times (required 8159/usable 8160, kv 34816 B/t
  x 36352 tok = 1207, decoder state 861 MB, 569 kv pages). v1 probe walls are
  CONTAMINATION-TAINTED (concurrent foreign build/boot; wrong prompt shapes) — banked
  under results/amd/kvhosted/A_off_128_* as evidence only, NOT a gate row.
- WINDOW CONFLICTS ENCOUNTERED (context for resume): the decode-landing desk woke
  mid-window (banked fdd9b68e 13:24, booted 13:27/13:28; then 4fb2d1ee 14:14 — its boot
  script's anchored retire killed my A2 leg mid-prefill) and the F1b desk edits this
  worktree live (tp2_backend.cpp 13:24, tests 13:22-13:26, w7_f1b_legs.sh 13:34).
  Interleave protocol posted in WO_DECODE_LANDING.md (whoever's server is UP owns the
  box; retire beats are the take-over points).

RESUME CHECKLIST (in order):
1. Re-derive: git log amd/wo-w7-body @ bd46de0e9 (integration diff is the whole change;
   anti-resurrection check: git diff main -- src/runtime/tp2/tp_engine.cpp
   src/runtime/tp2/tp2_budget.h must show only the hosted_kv_tokens seam + print, no
   refusal-math edits).
2. Re-bank check: ninfer-serve_9bc7d7e1f764e7cb.bin in artifacts_bin; if the tree moved
   (F1b d1 fix landed), REBUILD is fine — the integration is committed; df -h / first;
   BUILD CLAIM in this file; bank per BANK-BEFORE-RELINK.
3. Window discipline: check WO_DECODE_LANDING.md + WO_ONESHOT_ROOTLOOK.md mtimes; post
   WINDOW CLAIM here; take the box only at retire beats (leg script has a BOX BUSY guard).
4. Gate battery (tools/v340l/w7_kvhosted_leg.sh v2 — canonical alpha prompts, drain
   gaps, D2-SS-STATS capture): A2 off-arm (G-KV4) -> B hosted 20k tail (G-KV1 parity:
   P1 outputs must be text-identical across arms; G-KV3 P2/P3 walls) -> B3 funder
   (NINFER_KV_HOSTED=1 TOKENS=20000 + chunk 256 + NINFER_WORKSPACE_MIB=256: warmup must
   complete + serve; report composition-line VRAM freed AND the real decoder-state delta)
   -> B4 funder control (same, HOSTED off: expect clean preflight refuse = RED row).
5. Known-suspects at resume: (a) prompt tokens ~= 1.0x words for alpha streams (v1's
   1.54x figure was varied words); (b) probe ordinal = P1,P2,P3 on every fresh boot;
   (c) decode tax at 600-token ctx is small (all-hosted ~10 pages -> ~1.5 ms/tok class);
   the decode-tax signal is bigger at 10k+ ctx if the gate wants it at 600 generated.
6. Promotion path unchanged: gates pass -> runbook env line (OFF by default),
   BOOT_BATTERY, PLOG row (chain head was 74f40321209430b3 at park — re-verify),
   report. Gates fail -> default stays off, bank, report.
