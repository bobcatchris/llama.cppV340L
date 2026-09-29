# WO_KVARN_PORT — KVarN KV tier → HIP lane (k4v4 boots and serves on gfx900)

- **Desk:** amd/wo-kvarnport (one of the ≤2 live agents)
- **Worktree:** `/home/chris/worktrees/amd-wo-kvarnport`, branch `amd/wo-kvarnport`, base = `amd/main` @ `b27a587ae`
- **Opened:** 2026-09-19
- **Mission:** port the KVarN cached-attention route to the HIP roster so `--kv-dtype kvarn_k4v4` BOOTS and SERVES on gfx900. Unlocks (a) q4-KV quality milestone, (b) 200k context ceiling, (c) VRAM funder (~900 MiB/die freed bf16→k4v4 KV = the chunk ladder, PLOG-069).
- **Blockers (proven by k4v4 desk, `results/amd/coherence/W7_k4v4_row.txt`):**
  - **B1:** parse gate `src/serve/serve_options.cpp:437-462` (`#if defined(NINFER_HIP_ROSTER)`, G-AMD-31, `f23197980`) refuses any kvarn tier at parse time. The gate's own text schedules its death: "When T3's port lands on the roster, DELETE this block." Flip = delete, AFTER the port compiles+links.
  - **B2:** `src/ops/launcher/gqa_attention_kvarn.cu` absent from `src/HipSources.cmake` (named remainder, WO-06 S3a); link closes the gap with perf_stub_die arms `apps/serve/hip_link_stub_arms_perf.cpp` (`gqa_attention_kvarn_cached_launch`, `..._cached_batched_launch`, `..._capacity_bytes`). Removing B1 without B2 = k5v4 death-row class (mid-request stub death).
- **Laws in force:** VRAM LAW (no estimated refusals; allocator is the gate), RED→GREEN CLOSURE LAW, BANK-BEFORE-RELINK (`docs/amd/BOOT_LAUNCH_RUNBOOK.md §4`, bank to `/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin`), no bare pkill (retire = EXACTLY `pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"`; restore canonical `bash /home/chris/serve_fast.sh` + health at close), serving-window coordination via desk files (fp8ar desk may hold boots), clocks noted on timings, PROGRESS LOG newest-first after every step.

## PHASE 0 — PORT SURFACE INVENTORY

Target of the port = ONE device TU + its include chain:

| # | File | Role | gfx900 portability evidence |
|---|------|------|------------------------------|
| 1 | `src/ops/launcher/gqa_attention_kvarn.cu` (918 ln) | launcher TU; defines the 3 stubbed symbols (`:218/:729/:866`) | device pass = PROBE (below) |
| 2 | `src/ops/kernel/gqa_attention_kvarn.cuh` (862 ln) | kvarn attention kernels + `decode_split.inc` | int-code dequant math, `__shfl` (SIMT class, already joined) |
| 3 | `src/ops/kernel/gqa_attention_kvarn_mma.cuh` (751 ln) | warp FWHT / deferred acc FWHT / pack_bf16x8 | includes `ops/common/mma.cuh` which carries FULL `__HIP__` gfx900 ldmatrix/mma emulations (agent3 GDN landing, G-AMD-22 ruling A) — the old mma.cuh HOLD is LIFTED (q4/q5/w8 mma arms joined roster "REAL") |
| 4 | `src/ops/kernel/gqa_attention_kvarn_flash.cuh` (295 ln) | flash-style prefill | same mma.cuh basis |
| 5 | `src/ops/kernel/gqa_decode_slice4_kvarn.cuh` (552 ln) | k4v2/k4v4 decode prologue kernel | includes `cuda_pipeline.h` + `math_constants.h` — BOTH present in `src/common/hip_shim/` (first -I in HIP_INCLUDES) |
| 6 | `src/ops/kernel/gqa_decode_slice6_kvarn_k5v4.cuh` (609 ln) | k5v4/k4v4 MultiBatch prologue kernel | same shim basis |
| 7 | `src/ops/kernel/gqa_decode_kvarn_prologues.cuh` | shared per-side prologues (docs/117) | exists in tree |
| 8 | `src/ops/kernel/gqa_attention_kvarn_decode_packed.inc` (790 ln) | packed-only decode kernel | included by launcher TU |
| 9 | pool/workspace seams | `ops/kvarn/kvarn_workspace.cu`, `ops/kvarn/kvarn_tile_cuda.cu` | ALREADY on HIP roster (HipSources.cmake S3b) |
| 10 | host-side dial | `include/ninfer/types.h:89-134` widths, `tp2_budget.h:138` width-generic budget, `tp_engine.cpp:190-196` dispatch | host code, already compiles on HIP lane |

Stub-arm retirement (same commit as roster join, atomic retirement grammar b0f8b1a9): delete the 3 kvarn arms from `apps/serve/hip_link_stub_arms_perf.cpp`. The `bidirectional_gqa` arms stay (text serve never routes; not ours).

**Decisive probe:** isolated hipcc device-pass on the launcher TU with the EXACT whitelisted-TU flag set (flags.make, `ninfer_hip_host`): `-O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx900 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1` + the 7 shim/include -I paths.

**Effort estimate (pre-probe):** roster join + gate delete + stub retirement = minutes; the risk is entirely in the device pass of the 918-line TU and its 4,600 lines of headers. If the probe is clean: ~half a desk-day to serving (build ~450 MB, boots per runbook). If the probe surfaces gfx900 codegen breaks (wave64/smem/dynamic-pipeline class): 1–2 desk-days. >3 desk-days or a CUDA-only-API hard blocker ⇒ STOP and report to coordinator.

## PROGRESS LOG (newest first)

- **2026-09-19 — MISSION COMPLETE: PORT GREEN. kvarn_k4v4 BOOTS AND SERVES on gfx900 at TP4.** Evidence row: `results/amd/coherence/WO_KVARN_PORT_row.txt`. FINAL bin `ninfer-serve_f3312f25c8201da0.bin` (adds the coordinator-audit slab-ring OOB fix; serving sanity 391 PASS, zero worker errors). GREEN measurement bin `ninfer-serve_18993c160d07bf7c.bin`; RED-row bins 73eecb33/3af434bb/63a3d37e also banked (BANK-BEFORE-RELINK honored throughout). Quality: 5/5 battery GREEN, answers byte-identical to the bf16-KV reference (BLUE/ORANGE/391/ZEBRA-48213/ZEBRA-73942); needle@6k PASS at 8192 AND at 36352/chunk-256. VRAM funder MEASURED: kvarn KV 386 MiB/rank @36352 vs bf16 1207 → **821 MiB/die freed**. MTP on kvarn: acceptance 0.99, tok/round 2.99 (mt=600 counting probe, 600/600 tok, ~25.6 tok/s window). Chunk 256 + ws 256 MiB @36352: SERVES, no bad_alloc (perf note: 18.7 tok/s prefill vs 91-125 at chunk 128 — the ladder keeps 128 as default). **Three defects behind B2 found and fixed** (RED→GREEN with banked bins): D1 head-count constant + hardcoded kv_heads=2 binds → table-derived (tp2_backend, 6 sites) + heads<=0 gate (kvarn_workspace; device path is head-generic, quantize has an explicit heads==1 fast path); D2 gfx900 64 KB LDS limit (MEASURED 65536 B; TileShared 69,380 B → X-only smem + global scalars slab; kvarn_tile_cuda_test ALL PASS bit-identical); D3 **the V-plane dtype contract** — HIP-lane bf16-GQA caches carry K=bf16 bits + V=fp16 BITS (decode:156 + SIMT flash cast cache_v to __half); the kvarn materialize wrote bf16 V → flash read garbage → prompt-blind "amnesia" (P1-P5 RED); fixed by an HIP-guarded bf16→fp16 V-temp conversion in the launcher; unit q-block rel_l2 0.69→0.0025, T=1 0.0046. **Coordinator audit items addressed**: the flagged scalars-slab OOB is CLOSED BY CONSTRUCTION (stride-window ring: 128 slots / stride 8 / windows cannot cross the slab end; host bound ntiles<=8 = shape-table fact; cannot fire at kv_local=1 — re-verified tile test + serving sanity post-fix); the 12 bare shuffles in gqa_attention_kvarn.cuh were already width-fixed in this tree. Residual test_kvarn_gqa shadow/F5 failures = the TEST's own bf16-V-bit reference inputs (lane-contract artifact; staged-shadow is dead code in serve at stage_pages=0; packed route is a non-default A/B) — test audit named follow-up, NOT a serving blocker. Phase 0 estimate vs actual: compile probe was clean but the real surface was three runtime contract defects; **~1.5 desk-days total**. Canonical serve restored (serve_fast.sh, 2c8901d3, health OK). Lane: 10 files changed, ready for chair merge to amd/main.

- **2026-09-19 — SERVING WINDOW MANIFEST (announced before spawn).** Bin: `/home/chris/artifacts_bin/ninfer-serve_73eecb3370b6334e.bin` (sha256 `73eecb3370b6334e4fbd53f4bf64c32b172865adc2ca62d971428ecbb7d06adb`, built in this worktree at 15:42 local, `Built target ninfer-serve` rc 0). **nm-verified both sides**: `gqa_attention_kvarn_cached_launch` (0xa0df860 T), `..._cached_batched_launch` (0xa0f5140 T), `..._capacity_bytes` (0xa0f7ad0 T) — REAL TU definitions; `perf_stub_die` remains only for the bidirectional arms. Legs planned (each retires the previous): A1 kvarn_k4v4 small-ctx 8192 no-mtp boot + probe battery; B bf16 small-ctx 8192 no-mtp reference, same battery; A2 kvarn small-ctx + mtp decode probe mt=600 (acceptance/tok-per-round lines); C kvarn @36352 ceiling + chunk-256 attempt. Env = canonical serve_10k.sh exports (NINFER_ALLOW_NVFP4_TP2=1, NINFER_WORKSPACE_MIB=96, NINFER_DRAFT_VOCAB, NINFER_MTP_TAIL_ASYNC=1, NINFER_H2D_PINNED_STAGE=1, NINFER_VOCAB_COUNT_DIR); argv = canonical serve_fast shape + leg-specific --max-context/--kv-capacity/--kv-dtype. Retire = EXACTLY `pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"`; close = restore `bash /home/chris/serve_fast.sh` + health.
- **2026-09-19 — BUILD GREEN, BANKED (BANK-BEFORE-RELINK honored).** Fresh `build-hip-amd` in this worktree: `cmake --build build-hip-amd --target ninfer-serve -j8` rc 0 (ran alongside fp8ar's own-lane build; 20-core host). Bin 172,413,544 B banked as `ninfer-serve_73eecb3370b6334e.bin`. Disk after bank: 5.9 GB free.

- **2026-09-19 — PHASE 0 VERDICT: PORTABLE, no hard blocker. Decisive probe PASS.** Isolated hipcc device-pass on `gqa_attention_kvarn.cu` with the EXACT whitelisted-TU flag set (`-O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx900 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1` + shim/include -Is, `-x hip --hip-path=/opt/rocm-6.2.0`): **rc 0**, 159 device-side warnings + 7 host-side, ALL benign classes (`-Wpass-failed` occupancy-target on slice4/slice6 `__launch_bounds__`, `-Wcuda-compat` ignored-inline-on-kernel, one dangling-else). Wall 6m47s single-TU — heavy TU, adds ~7 min to clean builds. Key portability facts: `ops/common/mma.cuh` carries full `__HIP__` gfx900 ldmatrix/mma emulations (agent3 GDN landing, G-AMD-22 ruling A — the old mma.cuh HOLD is lifted, mma arms joined "REAL" per roster comments); `cuda_pipeline.h` + `math_constants.h` resolve in `src/common/hip_shim/` (first -I in HIP_INCLUDES). Effort: estimated half desk-day→1 day if clean; actual tracking that. Probe artifact: /tmp (transient), re-proof = the real build itself.
- **2026-09-19 — PHASE 1 EDITS LANED (3 files, one atomic commit pending build):** (1) `src/HipSources.cmake`: `ops/launcher/gqa_attention_kvarn.cu` JOINS next to the A3-wave gqa pair (provenance comment above the entry, roster-parser-safe); (2) `apps/serve/hip_link_stub_arms_perf.cpp`: the THREE kvarn perf_stub_die arms RETIRED (atomic retirement, b0f8b1a9 grammar) — bidirectional arms stay; (3) `src/serve/serve_options.cpp`: G-AMD-31 parse gate DELETED per its own scheduled (a)-resolution, tombstone comment left in place. Only ONE NINFER_HIP_ROSTER guard existed repo-wide (grep-verified) — no other refusal site on the kvarn path; tp_engine dispatch + budget are tier-generic host code.
- **2026-09-19 — build window opened.** Fresh `build-hip-amd` configured in this worktree (`cmake -S . -B build-hip-amd -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 -DNINFER_BACKEND=hip -DNINFER_BUILD_APPS=OFF -DNINFER_BUILD_BENCHMARKS=OFF -DNINFER_HIP_BUILD_SERVE=ON -DCMAKE_HIP_ARCHITECTURES=gfx900 -DBUILD_TESTING=OFF`; note: `-DCMAKE_PREFIX_PATH` is REQUIRED — first configure without it died "hip package not found"). Generated build contains the kvarn TU (12 build.make refs). `ninfer-serve -j8` build running alongside fp8ar desk's own-lane build (20 cores host). Disk at configure: 7.0 GB free. Canonical serve still up (PID 530991); boots wait for the bank.


### COORDINATOR RELAY (2026-09-19 ~16:5x) — READ-ONLY AUDIT DESK RESULTS (ranked root causes for your A1/EXP1 RED)
Full log: docs/amd/WO_KVARN_RED_AUDIT.md (committed bdb051b18 on amd/wo-redaudit). Decisive
evidence first — **the RED is a prefill-SCALE signature, not a tier bug**:
- `/tmp/kvarn_boot_k4v2.log` (same bin 63a3d37e): a 68-token prompt (1 chunk = 1 packed page
  + 4-token tail) generates 25 coherent deterministic tokens — quantize/commit/scale-write
  at page 0, materialize, bf16 flash, slice4 decode are ALL CORRECT. Every prompt >=1063 tok
  emits t0=248046 (EOS) at gen=1; t0 comes from the FINALIZE chunk's prefill attend — decode
  kernels never ran at long context. **The defect lives in packed pages >=2 / scale tiles
  >=2 / the 2nd+ prefill_chunk call.**
1. **Page>=1 misaddressing in the packed write/read pair (top)** — reader
   `gqa_attention_kvarn_flash.cuh:80` (`block_table[kb]` + `kvarn_scale_at` page arg) vs
   writer `kvarn_workspace.cpp:614-619` (`prepare_page_for_append`/`commit_completed`).
   Page 0 logical==physical (identity) hides a convention mismatch; from page 2 garbage
   scales/codes -> EOS. Decides with NO BOOT: your drafted `/tmp/test_kvarn_write_path.cpp`
   (append 3 pages, diff read-back vs CPU reference).
2. **Chunk-boundary host seam** — `tp2_backend.cpp:2411-2461` (`set_text_kv_base(cursor)`
   :2415) feeding `text_context_impl.h:468` (`tail.packed_pages = tile_page >= 0 ? tile_page
   : committed_pages`). Decides with 3 curls on your NEXT boot: T=100 / T=128 / T=132 —
   128 passing + 132 failing convicts this class exclusively.
3. **LATENT: 12 bare wave64-unsafe shuffles** in `gqa_attention_kvarn.cuh` (:155, :161,
   :329, :335, :381, :382, :576, :578, :591, :592, :658, :664) — the exact class you
   convicted+fixed in gqa_attention_kvarn_mma.cuh (your own F5 measurement proves the
   mechanism on this chip). Carrier `gqa_attention_kvarn_kernel` (:207) has no launch site
   in the default config today — same `, 32)` width fix when you touch the file.
ALSO FLAGGED (your uncommitted code): **provable OOB hazard** in the new scalars-slab ring
(`kvarn_tile_cuda.cu`: `fetch_add(ntiles) % 64` + `scal_base[blockIdx.x]` — silent
device-heap corruption for any ntiles % 64 != 0; does not fire at chunk=128 but WILL at
other geometries — RED/GREEN cell required before it ships), and a preflight budget
discrepancy (8976 B/t printed vs pool-spec ~4224 B/t — refusal-class only, no impact on the
RED). Note: leg-B (bf16 control) log is truncated — full bank needed as the coherence
control when you next run it.

### COORDINATOR TRIAGE NOTE (2026-09-19 ~17:5x) — A1_FIXED AMBER decomposition + gate tools staged
- A1_FIXED (bin 18993c16): P1/P2/P3 PASS coherent — the empty-output RED is FIXED at short context. P4 needle_3k + P5 needle_6k FAIL with finish=length AND answer='' — generation emits nothing at long context. Pattern check: 3k tok = ~47 pages, 6k = ~94 pages, both failing while 16-page 2k works — consistent with a scale-dependent seam. PRIME SUSPECT re-rank: the audit's flagged OOB in your scalars-slab ring (kvarn_tile_cuda.cu `fetch_add(ntiles) % 64` + `scal_base[blockIdx.x]`) — its wrap fires when tile count crosses the 64 boundary with ntiles%64!=0, i.e. between 2k and 3k the cumulative tile count likely crosses a multiple of 64. Second: decode-side attention reading large block tables (the needle probe's t0 is a DECODE output — prefill coherence at 2k doesn't exercise decode over 47+ pages).
- Phase-2 gate tools are STAGED for you (coordinator-built, on amd/main after 6cb6c9bd3 merge): tools/v340l/w7_needle_gen.py (deterministic needles, --ctx/--depths) + w7_k4v4_gates.sh (G-Q1 behavioral battery + G-Q2 needle runs; transcripts to /tmp). Use them at manifest end or hand back — coordinator or next free body runs Phase 2/3 per WO_Q4KV_K4V4.md (decision (a) recorded there).
- Reminder from your own manifest: leg-B bf16 control needs the full untruncated bank (coherence control for the Phase-3 perf pairs).
