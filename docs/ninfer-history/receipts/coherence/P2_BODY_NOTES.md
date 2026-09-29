# P2 BODY NOTES — per-body-op tracer (NINFER_PREFILL_OPTRACE=2), GPU-owner runbook + staffing tree

**Seat:** CODE desk, `amd/tp4-cure`, 2026-09-17 (compile-only; NO GPU run from this desk).
**Spec:** `docs/amd/PREFILL_BODY_2026-09-17.md` §5 (DECISIVE CHECK P2). **Purpose:** split
P1's measured body **645.1 ms/chunk** (4-rank mean: wall 1642.3 = gemm 798.8 + ar 70.7 +
body 645.1 + gap 122.7; `PREFILL_OPTRACE_row.txt`, bin `6c8ae21399516750`) across the named
suspects so the FIX desks staff in true-priority order. Hunt law: no fix ships before this
split.

**Code (this desk, committed on `amd/tp4-cure`):** tracer bank `src/core/prefill_body_trace.h`
(`ninfer::PrefillBodyTrace`), pairs in `text_context_impl.h` (11 sites + lifecycle), `mact`
in `qwen3_6_27b/impl/variant_kernels.cpp`, depth-3 internals in `gated_delta_net.cpp`,
`chunked/launch.cu`, `bf16_gdn_gating_proj_plan.cpp`. Env gate is a **DEPTH**:
`NINFER_PREFILL_OPTRACE` unset = 0 (all off, byte-identical), `=1` = P1 exactly
(wall/gemm/ar/body/gap, byte-identical output), `>=2` = + the 12 body classes,
`>=3` = + the S1/S2 internal splits. Plain-bool guards only; slots parity-doubled
(512 + 256 per rank, created best-effort only when armed); drained one-chunk-wide at the
existing P1 per-chunk sync points (race-free per P1).

## 1. Class → site map (17 record sites; begin/end brackets, no device work added)

| cls | x/chunk | bracket site (file:line, current tree) | suspect |
|-----|---------|----------------------------------------|---------|
| `gate`   | 48 | `Variant::gdn_norm_control_projection` call — text_context_impl.h:2586 | **S2** wrapper (rmsnorm + gating control proj; gemm splits at depth 3) |
| `upk`    | 48 | `multi_gpu::tp_unpack_gdn_qkvz` — text_context_impl.h:2800 | bandwidth class |
| `conv`   | 48 | `ops::causal_conv1d_silu_split3` — text_context_impl.h:2822 | cleared by doc §2 (fast prefill_pairs path; class proves its ~µs price) |
| `gbet`   | 48 | `multi_gpu::tp_unpack_gbeta_strided` — text_context_impl.h:2857 | bandwidth class |
| `dn`     | 48 | `ops::gated_delta_net` — text_context_impl.h:2873 | **S1** wrapper (l2norms + chunked trio; splits at depth 3) |
| `gnorm`  | 48 | `ops::gated_rmsnorm` — text_context_impl.h:2890 | bandwidth class |
| `aunpack`| 16 | `multi_gpu::tp_unpack_qkv_strided` — text_context_impl.h:2326 | bandwidth class |
| `qknorm` | 16 | q/k rmsnorm + rope block — text_context_impl.h:2357 | bandwidth class |
| `gqa`    | 16 | `ops::gqa_attention` prompt arm — text_context_impl.h:2500 | **S3** (gfx906 SIMT flash, 6x K/V re-stage per kv-group) |
| `agate`  | 16 | `ops::sigmoid_mul` — text_context_impl.h:2529 | bandwidth class |
| `mnorm`  | 64 | `ops::rmsnorm` in `mlp_tail` — text_context_impl.h:3140 | bandwidth class |
| `mact`   | 64 | `ops::silu_mul` in `Variant::post_mixer_tp` — qwen3_6_27b/impl/variant_kernels.cpp:486+490 (both arms) | bandwidth class |
| `l2` (d3)| 48 | l2norm(q)+l2norm(k) — gated_delta_net.cpp:285 | S1 split 0 |
| `wy` (d3)| 48 | `launch_prepare_wy_wu` — chunked/launch.cu:48 | S1 split 1 |
| `sp` (d3)| 48 | `launch_state_passing` — chunked/launch.cu:68 | S1 split 2 |
| `out`(d3)| 48 | `launch_output` — chunked/launch.cu:85 | S1 split 3 |
| `gmma`(d3)| 48 | the whole resolved-schedule launch dispatch in `execute_resolved` — bf16_gdn_gating_proj_plan.cpp:227 (scope-guard closed across the switch's `return` arms) | S2 split |

480 depth-2 pairs/chunk at T=128 (48x6 gdn + 16x4 full + 64x2 mlp); slots 512 ->
`body_over=`/`dn3_over=` appear ONLY on clip (treat touched columns as floors).
Not bracketed (by spec): MTP payload paths (`mtp_post_mixer`, gap/spec-decode region — not
in the probe body) and the traced GEMM/AR launches (P1 already owns them).

## 2. Output grammar + expected readout

Per rank, per chunk, at the P1 drain points:
```
[PREFILL-BODY] rank=R chunk=C tok=T t=<mono-ms> gate= upk= conv= gbet= dn= gnorm= aunpack= qknorm= gqa= agate= mnorm= mact= unbr=
[PREFILL-DN3]  rank=R chunk=C tok=T t=<mono-ms> l2= wy= sp= out= gmma=                (depth>=3)
[PREFILL-BODYSUM]  rank=R n=<chunks> mean_ms <13 columns> body= resid=                (request end)
[PREFILL-DN3-SUM]  rank=R n=<chunks> mean_ms l2= wy= sp= out= gmma=                   (depth>=3)
```
`unbr = body − Σ(12 classes)` — **the intra-layer launch-slack class** (what chunk graph
capture could reclaim inside layer bodies). `body=` is the same chunk's P1 body value;
`resid = body − Σ(13 printed columns)` — expect **|resid| ~ 0 by construction**; the DOC's
reconciliation bar is the 12 columns summing to body 645.1 within ~5% (i.e. `unbr` small).
Per-class ms are device ms on the rank stream (event pairs), summed per class per chunk;
BODYSUM columns are per-chunk means. 4 rank threads interleave on stdout — filter `rank=`
per rank. Expected anchor (doc §4 brackets, to be settled, not assumed): S1 `dn` 250-450,
S2 `gate` 60-160, S3 `gqa` 80-180, bandwidth classes 10-20 combined-ish; elementwise floor
~15 ms/chunk is hard (no fix goes below it).

## 3. GPU-owner runbook (P2 window; ONE command per step; no estimated-VRAM gates anywhere)

0. **Grant** — written GPU grant from the coordinator (intercom) BEFORE any boot; the
   current `serve_10k.sh` server on :8100 keeps serving until ITS owner retires it. Never
   `pkill` by pattern; kill only PIDs your own manifest row recorded.
1. **Rebuild** (this tree, compile verified at -O3, RC=0, zero new warnings):
   ```
   make -C /home/chris/worktrees/amd-tp4-cure/build-hip-amd -j20 ninfer_hip_host ninfer-serve
   ```
   (`df -h /` first — 11G free at desk time. Affects: variant_kernels, gdn gating plan,
   gated_delta_net + chunked/launch, tp2_backend, both variant.cpp TUs — all in
   `ninfer_hip_host`; whitelist parity guard 220/220 passes.)
2. **BANK-BEFORE-RELINK** (runbook §4 law — the filename is the stamp):
   ```
   sha256sum /home/chris/worktrees/amd-tp4-cure/build-hip-amd/apps/ninfer-serve
   cp -p  /home/chris/worktrees/amd-tp4-cure/build-hip-amd/apps/ninfer-serve \
          /home/chris/artifacts_bin/ninfer-serve_<sha16>.bin
   ```
   The P2 window boots **the banked copy** (pinned binaries boot from the bank, never from
   a lane build path). P1's `6c8ae21399516750` stays banked for cross-leg comparison.
3. **BOOT_BATTERY** (per-window cells; rides the window, never boots/kills a server):
   `tools/guards/BOOT_BATTERY.sh 8100` — banks `BOOT_BATTERY_<ts>.row` + clocks sideband.
4. **Boot P2** — same known-good line as P1's window-3 row (`qwen3_8_27b_nvfp4.ninfer`,
   `--port 8100 --devices 0,1,2,3 --prefill-chunk 128 --no-prefix-reuse
   --prefix-cache-capacity 256 --greedy --default-max-tokens 16 --spec mtp --draft-tokens 2
   --allow-nvfp4-weights`; env `NINFER_ALLOW_NVFP4_TP2=1 NINFER_WORKSPACE_MIB=96
   NINFER_DRAFT_VOCAB=<repo>/tests/multi_gpu/data/qwen38_draft_vocab_ids.json
   NINFER_VERIFY_LAYER_TRACE=1 NINFER_TP2_TIMING=1 NINFER_TP2_OPTRACE=1
   NINFER_LMHEAD_ARM=simt NINFER_LMHEAD_TRACE=1 NINFER_TILED_TRACE=1
   NINFER_VERIFY_TAIL_TRACE=1`) with **`NINFER_PREFILL_OPTRACE=2`** (add `=3` only as a
   second same-boot probe after the =2 probe — depth 3 costs another ~240 event
   pairs/chunk). Announce the manifest row (argv + artifact path + window) BEFORE spawn;
   release row after, including deaths.
5. **Cool window + probe**: ≥120 s idle cool window (THARM1/PF2 law: perf numbers are
   thermal-state-dependent), **start the 2 s clocks sideband sampler** (`pp_dpm_sclk` level
   + edge temp per die into `P2_probe_clocks.log`, covering the whole probe window), then
   fire **ONE plen-1996 conc=1 prefill probe, FIRST+ONLY request on the fresh boot**
   (P1 shape: e.g. `BLUE` body + alphas to 1996 tokens, greedy, mt 64). Prefill ~29 s.
6. **Bank** the row: `results/amd/coherence/PREFILL_BODY_row.txt` — P1 row format (bin
   path+sha+HEAD sha, config, full env, serve log name, probe shape+client wall, clocks
   band verdict), then the verbatim `[PREFILL-BODYSUM]` x4 ranks + `[PREFILL-DN3-SUM]`
   lines, one representative `[PREFILL-BODY]` per-chunk table (rank0), and the 4-rank mean
   split. Keep the serve log alongside (grep `[PREFILL-BODY]`/`[PREFILL-BODYSUM]`).
7. **PLOG**: write the entry file, then
   `python3 tools/guards/plog_append.py append <entry-file>` (chain hash is
   script-computed; `verify <start-hash>` walks it). Retire the server per the runbook
   (anchored TERM of YOUR recorded pid, KFD raw-dump check), run BOOT_BATTERY once more on
   the idle box if any cell needs a post-window witness.

## 4. Desk-staffing decision tree (read the BODYSUM 4-rank means, then staff)

Let `DN=dn`, `GQA=gqa`, `GATE=gate`, `UNBR=unbr`, `BW = upk+conv+gbet+gnorm+aunpack+qknorm+agate+mnorm+mact`.

- **(a) Emulation thesis DIES** — `DN < 150` AND `GQA < 60` AND `GATE < 40`:
  the 645 ms is launch-slack (`UNBR` dominant). Staff the **graph-capture desk** (per-chunk
  CUDA/HIP-graph capture of the layer bodies) — NOT the kernel desks. No FIX-A/B/C.
- **(b) Bandwidth census wrong** — `BW > 100` ms combined (doc §3 hard-prices the
  elementwise class at ~15-20 ms): the wrappers hide a pathology. Staff a **wrapper/launch
  audit desk** before any kernel work; re-derive the §3 census from the per-class columns.
- **(c) Suspects own** (the expected branch) — staff by descending measured ms, doc's
  default order unless the measurement reorders it:
  1. `DN` biggest → **FIX-A** (gfx900-native chunked delta-net: drop mma/ldmatrix/cp_async
     emulation for register-tiled SIMT FMA + optional HFMA2, re-grid 24-48 CTAs -> 96-192
     to fill 64 CUs, W deduped per k-head over its 3 v-heads). Read `wy`/`sp`/`out`/`l2`
     from the depth-3 lines to pick the first kernel. Predicted `dn` -> ~20-60 ms.
  2. `GQA` next → **FIX-B** (prompt split-K over key tiles, S=4: 24 -> 96 CTAs, partials
     into the existing SmallT acc/m/l workspace + one flash-reduce; kills the 6x K/V
     re-stage). Predicted `gqa` -> ~15-40 ms.
  3. `GATE` → **FIX-C** (route T>=32 to the plain SIMT gemv/gemm arm or rowsplit-SIMT
     schedule avoiding emulated ldmatrix; phase-2 alternative folds a/b into qkvz's N).
     `gmma` (depth 3) should equal `gate` minus rmsnorm — if `gmma << gate` the wrapper's
     rmsnorm owns, re-staff. Predicted `gate` -> ~5-15 ms.
- Cross-checks: columns + unbr must reconcile to body (resid ~0); a `body_over=`/`dn3_over=`
  on any rank invalidates that rank's clipped columns (re-run with the cap raised); the
  chunk-1 request-first-chunk drain artifact (P1 row) — use chunks 2..n for means, same as
  P1 did.

## 5. Compile-verify receipt (this desk, 2026-09-17)

- `make -C build-hip-amd -j20 ninfer_hip_host` → **RC=0** (all affected TUs rebuilt clean:
  variant_kernels.cpp, bf16_gdn_gating_proj_plan.cpp, gated_delta_net.cpp,
  chunked/launch.cu + all chunked kernel TUs, tp2_backend.cpp, qwen3_6_27b variant.cpp,
  qwen3_6_35b_a3b variant.cpp), whitelist parity guard 220/220 OK;
  `make -j20 ninfer-serve` → **RC=0**, binary linked (137 MB).
- Warnings: 66, all the pre-existing `-Wunused-result` (nodiscard hipError_t / system())
  class on pre-existing lines (tp2_backend 35, text_context_impl 27 at P1/legacy lines,
  vram_trace 4); **zero** warnings reference `prefill_body_trace.h`, any `PBO_*`/record
  site, or any line this desk added (every new event call is `(void)`-cast like P1).
- Two defects of the prior dead attempt found and fixed on the way: (1)
  variant_kernels.cpp never included `core/prefill_body_trace.h` (4x
  "not a member of ninfer"); (2) the qknorm brace-wrap had moved the
  `cache_positions`/`rope_positions`/`rope_for_op` declarations into block scope while
  later attn_mix_tp arms still read `cache_positions` (10x not-declared) — declarations
  restored to function scope, only the three op launches inside the pair braces.
