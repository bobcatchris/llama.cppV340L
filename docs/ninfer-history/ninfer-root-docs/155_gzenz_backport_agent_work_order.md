# 155 — gzenz/ninfer fork optimization backport: Agent Work Order

**Status:** CURRENT — work order for a single implementer agent (A1).
**Mission:** backport the valuable optimizations of the gzenz/ninfer fork
(`https://github.com/gzenz/ninfer`, clone at `/tmp/gzenz_ninfer`, tip `4b882d0`, depth 1)
into this repo, starting with the **NVFP4 KV cache** (`--kv-dtype nvfp4`). Done (phase A) =
a live TP2 server on `--kv-dtype nvfp4` that serves greedy + sampled requests, passes the
serve correctness battery and the MTP order-discrimination gate (S2-style) on the new tier,
halves the per-token KV bytes vs int8 (measured, committed), and leaves every existing tier
(i8/bf16/kvarn_*) bit-untouched. Phases B/C (smaller serve-layer ports, listed in §5) follow
after phase A or in parallel where they are independent of the KV tier.

Read this document fully before writing code.

---

## 1. Context (60-second version)

The gzenz fork is a **clean divergence** of this repo targeted at a single RTX 5090 (32 GB,
`sm_120a`): reliable 555k-context inference with 3 concurrent agentic sessions using an
NVFP4 KV cache (4-bit E2M1 codes + E4M3 group-16 scales), a host-KV spill arena, YaRN
position scaling, and a set of serve-layer robustness features. Our hardware is 2× RTX
5060 Ti 16 GB under TP2 — the fork's single-GPU runtime does not drop in, but its
**codec math, kernel structure, and serve-layer features port**. The user prioritized the
NVFP4 KV cache first ("first thing").

Fork spec: `/tmp/gzenz_ninfer/docs/maintainer/kv-nvfp4-yarn.md` (NVFP4 + YaRN, quality
tables). Fork survey of record: `worktrees/wo-kvarn-multibatch/docs/HANDOFF_A1_gzenz_optimizations.md`
(coordinator's list of 14; this work order re-derives and corrects it in §5).

**Already built and verified (do not redo):**
- Tier mechanism with ONE accepted-name table (`include/ninfer/types.h` — `kv_cache_from_name`,
  `kv_cache_storage_name`, `kvarn_tier_widths`), added by docs/117 so a new tier is selectable
  everywhere at once.
- Paged KV substrate (`src/core/paged_kv_cache.h`, P=64) with per-tier plane planning
  (`src/targets/qwen3_6/impl/state/decoder_state.cpp::plan_cache` — int4 precedent: U8 code
  planes at `head_dim/2` + FP16 scale planes).
- Budget single source (`src/runtime/tp2/tp2_budget.h::kv_bytes_for_tier`, docs/151 §18.1bis).
- The (vi) fix `a4066f5d` + bf16/int8 S2 verification (`b1e99a2c`, `4ada1561`) on
  `wo/kvarn-multibatch` — the MTP machinery NVFP4 must integrate with.

**What you are doing:** phase A steps 1–8 of §6 (NVFP4 KV tier end to end), then phase B/C
items from §5 as assigned.

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; remote `github`).
- **Work protocol (MANDATORY):** ALL work happens in a worktree + branch — never edit the
  main tree at `/home/intel/ninfer/repo` directly (2026-08-24 `git add -A` incident).
  ```bash
  cd /home/intel/ninfer/repo
  git worktree add ~/ninfer/worktrees/wo-nvfp4-kv -b wo/nvfp4-kv wo/kvarn-multibatch
  cd ~/ninfer/worktrees/wo-nvfp4-kv
  cmake -S . -B build && cmake --build build -j 16
  ```
  Branch base is `wo/kvarn-multibatch` (NOT main): the NVFP4 MTP-path work rides the
  batched-MTP machinery that only exists there, and the (vi) fix is its prerequisite.
  Commit per step to `wo/nvfp4-kv` and push. **Merging to main is done by the
  main-side agent/user** — do not merge or push to main yourself.
- **Source repo (read-only reference):** `/tmp/gzenz_ninfer` (depth 1, tip `4b882d0`).
  Never build from it; never edit it. It is `sm_120a`-locked for RTX 5090 (32 GB); our
  build is `sm_120a` on 2× RTX 5060 Ti 16 GB — the instruction set is the same Blackwell
  family, but VRAM budgets and the TP2 runtime reserve differ, so ALL fork VRAM numbers
  must be re-derived through `tp2_budget.h`, not copied.
- Build: `cmake --build build -j 16` inside the worktree. Unit tests via `/usr/bin/ctest`
  (the `ctest` on PATH is a broken Python wrapper), from `build/`.
- Server tests: `bash tools/ops/run_ci.sh` from the worktree root (fast gate);
  `--full` for closeout only. Targeted: `tools/smoke/serve_correctness_ci.sh` with
  `KV_DTYPE=... KV_CAPACITY=... MAX_CONTEXT=...` env overrides — it manages its own server
  lifecycle; never curl-test a server you did not start.
- MTP gates: docs/131 point-of-failure workflow (`tools/bench/run_t1.sh` INVASERT →
  cause map / HASHPT → T0; `run_t2.sh` golden matrix). Any commit touching the MTP/kvarn
  decode path runs T0 + T1-quick + INVASERT + T2(T=64) (docs/131 T3.1).
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer` (int8 weights image; NVFP4 KV is
  independent of weight format — the fork serves NVFP4 KV off NVFP4-weights artifacts, we
  serve it off our existing image).
- **GPU protocol (coordinator rule, MANDATORY):** claim the cards only after a written
  grant from the coordinator (intercom). Re-guard at the moment of claim
  (`tools/smoke/diag/gpu_guard.sh gpu_refuse_if_busy`; expect 15 MiB / 0% / no apps).
  Release immediately after each run (verify via nvidia-smi). Never `pkill` foreign
  processes; kill only PIDs you started. `df -h /` before big builds (disk 95% full).
- Measurement discipline: prompt sizes from `usage.prompt_tokens`, never char estimates;
  `nvidia-smi` clocks recorded with every perf number; every probe/run output committed
  under `results/` with its server config.

## 3. Architecture facts (verified — do not re-derive)

**Fork (reference, `/tmp/gzenz_ninfer`):**
- NVFP4 KV = E2M1 codes (4-bit, 2/byte) + E4M3FN scale byte per 16-element group.
  144 B/token/KV-head/side (128 code + 16 scale) vs int8 264, bf16 512 (fork accounting).
- Codec: `src/ops/kv_cache/nvfp4_g16_codec.cuh` — `kv_cache_nvfp4_{code,scale,src}_index`
  (same `paged_kv_element_offset<leading_extent, KVHeads>` helper family as ours),
  `kv_cache_nvfp4_dequant_e2m1x8_from` (one 32-bit load of 4 code bytes → 8 bf16),
  `kv_cache_nvfp4_quantize_k16` (16 bf16 → 8 code bytes + 1 E4M3 scale), f16 variant.
- Quantization primitive: `detail::quantize_nvfp4_k16(source, input_scale_divisor)` in
  `src/ops/linear/nvfp4/nvfp4_codec.cuh` — per-16-group absmax → E4M3 scale (shared with
  the weight-only NVFP4 path).
- Attention: QK uses NVFP4×NVFP4 `mma_nvfp4_e4m3` (m16n8k64) with built-in E4M3 block
  scales (Q and K both quantized, natural row-major scale layout — NOT the M128x4 swizzle
  of weight MMA). PV dequantizes V to BF16 (`decode_nvfp4_e2m1x2 × E4M3 scale`) then BF16
  MMA. Decode kernel fuses the in-place quantize-append of current K/V.
- **Hadamard rotation applied to K and Q for outlier suppression; V is NOT rotated**
  (`src/ops/kv_cache/hadamard_d256.cuh` — factorized H256 via H32 shuffle butterflies;
  unnormalized fragments with a 2^-4 final leaf). Because H is orthogonal and its own
  inverse, rotating both Q and K leaves QK^T invariant: a dequant-read path needs no
  un-rotate, it only needs Q rotated symmetrically.
- Precompute kernels: `src/ops/softmax_attention/dense/causal_cache/prompt_nvfp4.{cuh,cu}`
  (prefill, 444+88 lines) and `small_t_nvfp4.{cuh,cu}` (decode, 725+239 lines).
- Dispatch chain: `--kv-dtype nvfp4` → `KvCacheStorage::Nvfp4Group16`
  (`include/ninfer/types.h:33`) → `{DType::U8, quant_group=16}`
  (`src/targets/qwen3_6/impl/runtime/layouts_impl.h::target_kv_cache_profile`) → code
  planes `leading_extent = head_dim/2 = 128`, scale planes `leading_extent = 16` with
  `DType::U8` (E4M3 bytes, NOT FP16).
- Fork quality numbers (their hardware/artifact, quoted for expectations only): LongBench
  v2 20-sample: int8 45%, NVFP4-native 30%, NVFP4+YaRN@600k 45%. AIME 2025: NVFP4 29/30.
  Needle 128k 5/5. NOTE the honest reading: NVFP4-native scored BELOW int8 on their
  LongBench set — our acceptance bar for the tier is needle 5/5 + serve battery + no
  regression in MTP acceptance, not "beats int8".
- Storage enum in the fork has Fp8E4M3Row256 as well — NOT in scope for this backport.

**Ours (this repo):**
- `KvCacheStorage` = {BFloat16, Int8Group64, KvarnK4V2, Int4Group64, KvarnK5V4, KvarnK4V4}
  (`include/ninfer/types.h:27`); ONE parse table `kv_cache_from_name` (types.h:83);
  human names `kv_cache_storage_name`. KVarN tiers share the `DType::KVARN_K4V2` marker +
  per-side widths; `Int4Group64` uses the `DType::I4` marker (docs/117 §4.1: I8 template
  at half code width; scale planes FP16 per 64-group).
- `DType` (`src/core/dtype.h:8`): BF16, FP32, I32, U8, I64, I8, FP16, FP8_E4M3FN,
  KVARN_K4V2 (marker only), I4 (marker only). A new tier needs a new marker value here
  (planes themselves are U8).
- Plane planning: `src/targets/qwen3_6/impl/state/decoder_state.cpp::plan_cache` — per
  layer: int8 = 2 planes `head_dim` + 2 FP16 scale planes `head_dim/quant_group`;
  int4 = 4 planes (U8 codes at `head_dim/2`, FP16 scales at `head_dim/64`); kvarn = 2 U8
  code planes + separate `kvarn_scales` tensor [layers, heads, pages, 1152] fp32.
  Validation block throws on unhandled dtype/quant_group combos — a new tier MUST extend
  this block, the silent-fallback class is a known defect pattern (docs/117 §9).
- Budget: `src/runtime/tp2/tp2_budget.h::kv_bytes_for_tier` is THE single source (probe +
  preflight + unit matrix all call it). int8 = 34 × (512+32) = 18496 B/token/rank; bf16 =
  34 × 1024 = 34816; int4 = 34 × (256+32) = 9792. **NVFP4 = 34 × (2×128 + 2×16) = 9792
  B/token/rank — byte-identical to q4_0, 47% below int8** (fork's "45% vs int8" claim is
  consistent under their accounting). `verify_no_divergence()` guards probe/preflight
  divergence — the unit matrix in `tests/test_tp2_budget.cpp` shows an empty cell for a
  tier missing here.
- TP2 decode dispatch: `src/runtime/tp2/tp2_backend.cpp` `decoder_spec` lambda (~:364-410):
  `is_i8_kv / is_kvarn_kv / is_i4_kv` arms select the DType marker + quant_group fed to
  `plan_decoder_state`; local geometry kv_heads=2 (TP halves 4), head_dim=256,
  `kPagedKVPageSize = 64` (`src/core/paged_kv_cache.h:15`). The MTP pool
  (`mtp_kv_heads=2`, `mtp_physical_page_groups`) shares the same spec.
- Attention: launcher `src/ops/launcher/gqa_attention.cu` routes SmallT/ChunkedSmallT/
  Prompt; decode kernels per tier: `gqa_attention_decode.cuh` (bf16),
  `gqa_attention_decode_i8.cuh` + `gqa_decode_slice3_i8_v2.cuh` + unified route
  (`gqa_decode_unified*.cuh`, I8Route env `NINFER_I8_DECODE`), `gqa_decode_slice5_i4.cuh`,
  `gqa_decode_slice4_kvarn.cuh` (k4v2), `gqa_decode_slice6_kvarn_k5v4.cuh`. Prefill:
  `gqa_attention_prefill_bf16.cuh` / `gqa_attention_prefill_i8.cuh` (+ kvarn flash/direct/
  mma variants). KVarN kernels already carry Hadamard machinery (`hadamard_d32_columns_
  inplace` in `gqa_attention_kvarn_mma.cuh` / slice kernels; `src/ops/kvarn/kvarn_codec.h`)
  — the H256 primitive exists in-tree, do not re-derive it.
- Append: `gqa_kv_append_launch` (`src/ops/launcher/gqa_attention_prefill.cu:138`) is the
  ONE append entry used by text prefill AND the MTP prefill path
  (`text_context_impl.h:1348` writer #1) — quantization is FUSED into append kernels;
  index math in `src/ops/kernel/gqa_attention_kv_quant.cuh` (i4 variant already shows the
  "codes at D/2 bytes" pattern). Decode-time appends are fused into the decode kernels.
- MTP surface (the (vi) lane's home): batched MTP cache rows/lanes ride `mtp_kv_` views +
  `batch_layer_view` (`tp2_backend.cpp` per-lane prefill loop ~:2656-2755; the a4066f5d
  hoisted rebind). NVFP4 planes must work identically there — order-dependence gates
  apply to the new tier.

## 4. Key call sites (anchors — verify line numbers before editing)

- `include/ninfer/types.h:27` — `KvCacheStorage` enum: add `Nvfp4Group16`.
- `include/ninfer/types.h:83` — `kv_cache_from_name`: add `"nvfp4"` (+ no aliases).
- `include/ninfer/types.h` — `kv_cache_storage_name`: add the human name.
- `src/core/dtype.h:8` — add `DType::NVFP4_G16 = 10` marker (+ `dtype_size` in dtype.cpp:
  marker, not a plane dtype; follow the I4/KVARN_K4V2 precedent).
- `src/targets/qwen3_6/impl/state/decoder_state.cpp::plan_cache` (~:15-115) — validation
  block + `planes_per_layer` + plane push: NVFP4 arm = 4 U8 planes per layer
  (K codes `{U8, 128, kv_heads, 256}`, V codes same, K scales `{U8, 16, kv_heads, 256}`,
  V scales same); accept `quant_group == 16`, `head_dim % 16 == 0`.
- `src/runtime/tp2/tp2_backend.cpp` `decoder_spec` (~:364-410) — `is_nvfp4_kv` arm:
  `DType::NVFP4_G16`, `kv_quant_group = 16`.
- `src/runtime/tp2/tp2_budget.h::kv_bytes_for_tier` (~:194-207) — `return 34ULL * 288ULL;`
  (= 9792) for `Nvfp4Group16`. Update `tests/test_tp2_budget.cpp` matrix.
- `src/serve/serve_options.cpp` (~:348-400) — the refusal guard: while decode/prefill are
  unwired, `nvfp4` throws like q4_0 does; flip to accept when step 5 lands.
- `src/serve/request_log.cpp:131` — storage name switch: add the case (log correctness).
- `src/ops/launcher/gqa_attention_prefill.cu:138` — `gqa_kv_append_launch`: NVFP4 append
  (fused E2M1+E4M3 quantize) OR a sibling launcher selected by cache dtype.
- `src/ops/kernel/gqa_attention_kv_quant.cuh` — add nvfp4 index math + quantize/dequant
  helpers (port of `nvfp4_g16_codec.cuh` math onto our `paged_kv_element_offset` family);
  scale rows are U8 E4M3 at leading extent 16.
- Decode kernel: new `src/ops/kernel/gqa_decode_slice7_nvfp4.cuh` (or unified-route arm)
  + dispatch in `src/ops/launcher/gqa_attention_decode.cu` (route by cache dtype).
- Prefill kernel: NVFP4 arm in the prefill family (`gqa_attention_prefill_i8.cuh` pattern:
  dequant-to-smem then BF16 flash; NOT the fork's NVFP4-domain QK MMA — see §5.2).
- Hadamard: `src/ops/kv_cache/hadamard_d256.cuh` (new, port of fork's factorized H256) +
  apply at K append (before quantize) and at Q load (before QK) for the NVFP4 tier only.
- Fork reference files: `src/ops/kv_cache/nvfp4_g16_codec.cuh`, `hadamard_d256.cuh`,
  `src/ops/linear/nvfp4/nvfp4_codec.cuh` (quantize_nvfp4_k16),
  `src/ops/softmax_attention/dense/causal_cache/{prompt,small_t}_nvfp4.*`,
  `docs/maintainer/kv-nvfp4-yarn.md`.

## 5. Design decisions (FINAL — do not re-litigate)

1. **NVFP4 enters through the docs/117 tier mechanism** (one enum + one name-table entry +
   one DType marker + plan_cache/budget/decoder_spec arms). REJECTED: aliasing NVFP4 onto
   `Int4Group64` (same 9792 B/token) — different scale semantics (E4M3 u8 per-16 vs FP16
   per-64) cannot share planes, and the tier names must stay distinguishable in logs.
2. **Compute path = dequant-to-BF16 first** (K dequant → BF16 QK; V dequant → BF16 PV, the
   fork's own PV approach generalized to QK). The fork's NVFP4×NVFP4 `mma_nvfp4_e4m3` QK
   fast path is DEFERRED (perf-only; correctness and VRAM win land first). REJECTED for
   phase A: porting `small_t_nvfp4.cuh` wholesale — it is written against the fork's
   engine/scheduler; our kernels own different routes. Revisit the QK MMA as a phase A2
   step only after E2E green and only with the VRAM/quality gates already passing.
3. **Hadamard rotation on K and Q for the NVFP4 tier, V untouched** (fork's outlier
   suppression; orthogonality makes it transparent to the dequant path — Q is rotated at
   load). Applies ONLY when cache dtype is NVFP4; i8/bf16/kvarn numerics byte-untouched.
4. **MTP pool rides the same NVFP4 planes** (`mtp_kv_heads=2`, same 4-plane layout). The
   batched-MTP cache code (the a4066f5d area) is dtype-agnostic above the plane view —
   verify, do not refactor. S2-style order gate runs on NVFP4 before "done".
5. **Refuse-then-flip:** `serve_options.cpp` gains the q4_0-style refusal for `nvfp4` in
   step 1 (so an unwired tier can never silently serve as something else — the docs/117
   §9 defect class), flipped to accept in step 5/6 when the E2E path exists.
6. **Verification bar for phase A** (ours, not the fork's): needle-in-haystack 5/5 at
   128k, greedy-serve battery green on NVFP4, MTP S2-style order gate SAME on NVFP4,
   MTP acceptance within noise of int8 on the same fixture, per-token KV bytes measured
   = ~9792 (±1%), existing-tier regression zero (run_ci.sh gate). Fork's LongBench v2
   harness (`/tmp/gzenz_ninfer/eval/`) is a stretch goal (20-sample), not a gate.
7. **YaRN is OUT of phase A scope.** Native context of qwen3.8-27b (262k) exceeds anything
   our 2×16 GB cards can hold even with NVFP4 (~160k est.) — linear RoPE scaling has no
   user-visible payoff here until >262k contexts are reachable. Revisit only if the
   coordinator prioritizes it (small change: ramp on `rope_positions` only; fork flags
   `--rope-scaling-factor/--rope-scaling-original-context`).
8. **Fork VRAM/quality numbers are quoted, never assumed:** all sizing re-derived through
   `tp2_budget.h` for OUR geometry (TP2, 16 GB cards, our runtime reserve).

### Backport triage (all fork changes, verified against fork code 2026-09-06)

| # | Fork change | In our repo? | Phase / order | Fork anchors |
|---|---|---|---|---|
| 1 | NVFP4 KV cache | absent | **A (this work order §6)** | nvfp4_g16_codec.cuh, {prompt,small_t}_nvfp4.*, types.h:33, layouts_impl.h:75 |
| 2 | Paged KV (P=64, typed pools) | **already present** (own design; P=64, planes, block tables, MTP pool) | N/A — do not port | paged-kv-cache.md (their doc describes THEIR substrate) |
| 3 | Host-KV safety net (`--host-kv-mib`, `--host-state-slots`) | absent | C (needs an eviction story in our pool; big) | host_kv_arena.{h,cpp}, host_kv_safety_net.h, serve_options.cpp |
| 4 | YaRN context extension | absent | deferred (§5.7) | kv-nvfp4-yarn.md §YaRN, serve_options.cpp rope-scaling |
| 5 | OOM recovery (bad_alloc catch, admission backoff) | absent | B | fork worker loop / materialization reserve (locate via `std::bad_alloc` in serve/runtime) |
| 6 | Rewrite checkpoint at turn boundary | absent (verify our checkpoint semantics first) | B (applicability check may reject) | fork context-cache code |
| 7 | Token stability `preserve_thinking=off` | verify (our translate.cpp reasoning handling) | B | fork frontend/template code |
| 8 | Checkpoint lifecycle preservation | verify (same subsystem as 6) | B | fork context-cache code |
| 9 | Tolerant tool-call recovery `--tolerant-tool-calls` | absent | B | serve_options.cpp, fork tool-call parser |
| 10 | Reasoning-effort tier mapping (High/Max → XHigh) | partial (translate.cpp has reasoning-effort; verify reject-vs-map) | B (small) | fork translate equivalent |
| 11 | /stats endpoint | absent | B (medium; pairs with 12) | stats_json.{h,cpp}, http_server.cpp |
| 12 | Monitoring dashboard `tools/monitor/` | absent (wo-dashboard worktree exists — check overlap first) | C | tools/monitor/monitor.py, run.sh |
| 13 | E2E test suite `tools/e2e/` | absent (our run_ci/serve_correctness_ci cover part) | C | tools/e2e/ninfer-e2e.py |
| 14 | Request-log rotation `--request-log-max-mib/--request-log-keep` | absent (request_log.cpp exists, no rotation) | B (small) | serve_options.cpp, request_log.cpp |
| + | Froggeric v22 chat template + `--chat-template`/`--chat-template-semantics` | absent (missed by the coordinator's 14-list) | C (independent; product call) | fork README §chat template, frontend/chat_template.cpp |
| + | `--weights-profile` override | absent (we key profiles off artifact identity) | C | fork serve_options.cpp, targets/*/impl/package.cpp |

Phase B = small, low-risk serve-layer ports, each independently shippable with its own
tests; phase B items do NOT gate phase A. Phase C items are larger/infra; take them only
on explicit coordinator assignment. Every "verify" in the table means: diff the fork's
behavior against our tree's current behavior FIRST and write the finding into the commit —
several fork items fix upstream problems we may not have (clean divergence, not a patch set).

## 6. Execution order (commit + test each step before the next)

**Testing standard (applies to every step):** tests simulate REAL behavior. Any step that
touches a server-facing path is not done on unit tests alone — the actual call sequence
(chunked prefill, MTP rounds, request lifecycle) must run end-to-end on the new tier via
the serve battery / live server. Kernel isolation is necessary, never sufficient (§7).

### Step 1 — Tier plumbing + refusal guard (no GPU)
Add `Nvfp4Group16` to the enum, name/parse entries, `DType::NVFP4_G16` marker,
`plan_cache` NVFP4 arm (4 U8 planes: 128/128/16/16, quant_group 16 validation),
`kv_bytes_for_tier` = 9792, `decoder_spec` arm, request_log name case, and the
serve_options refusal (nvfp4 → "not servable yet", q4_0 pattern). Update
`tests/test_tp2_budget.cpp` matrix.
**Tests:** unit — budget matrix shows the NVFP4 cell = 9792; plan_cache validation accepts
nvfp4+16 and rejects nvfp4+64 (and bf16-style quant_group 0); serve_options refuses
`--kv-dtype nvfp4` with the clear error. `/usr/bin/ctest -R budget` + options tests.

### Step 2 — CPU reference codec + golden vectors (no GPU)
Port the codec to CPU-testable form (E2M1 decode table, E4M3FN decode, `quantize_k16`:
per-16 absmax → E4M3 scale, pack 2/byte). Golden vectors: transcribe ~16 quantize and ~16
dequant cases from the fork's `quantize_nvfp4_k16`/`decode_nvfp4_e2m1x2` semantics
(re-derive on paper, do not run fork code); round-trip tolerance bounds vs bf16 recorded.
**Tests:** unit test with the golden vectors + round-trip max-rel-err bound; document the
0.02-0.03% emulation-floor lesson from the (vi) lane (comparisons against bf16 reference
must state their tolerance, not claim bit-exactness).

### Step 3 — Append path (GPU kernel, build + bench-level test)
NVFP4 fused quantize-append: index math in `gqa_attention_kv_quant.cuh` (codes D/2 U8,
scales 16×U8 E4M3), append kernel arm in `gqa_kv_append_launch` (+ TpGeometry variant).
Hadamard H256 port (`hadamard_d256.cuh`) applied to K before quantize (Q side lands with
step 4). Guard the refusal flip OFF still (attention cannot read the tier yet).
**Tests:** GPU unit (bench binary or ctest CUDA test): append a known bf16 K/V page, read
back codes+dequant on device, compare to CPU codec within recorded bound; both geometry
variants (Gqa27, Gqa27Tp).

### Step 4 — Prefill attention read path (GPU)
NVFP4 arm in the prefill family: dequant K (with Hadamard-rotated Q, per §5.3) and V to
BF16 in smem/stage, then the existing BF16 flash body — the `gqa_attention_prefill_i8.cuh`
pattern (i8 already dequants in-kernel; i4 index math shows the D/2 code rows).
**Tests:** GPU unit vs a BF16-materialized reference on synthetic pages (rel-err bound
recorded); then a single-request live prefill smoke on the worktree build (chunked prefill
against a tiny max_context server) — response must be coherent (needle word retrieval on a
~2k prompt is enough at this step).

### Step 5 — Decode path + dispatch flip (GPU)
NVFP4 decode slice kernel (unified-route arm preferred — one body, vectorized dequant
prologue per the I8Route precedent): dequant K/V per page, BF16 MMA decode, FUSED
in-place quantize-append of the current step's K/V (fork's decode-fused append behavior).
Rotate Q (Hadamard) at QK input. Flip the `serve_options.cpp` refusal to accept.
**Tests:** GPU unit (dequant parity vs step-2 codec; fused-append ↔ prefill-append
agreement on shared pages); then `KV_DTYPE=nvfp4 bash tools/smoke/serve_correctness_ci.sh
--mode ci` (needs coordinator GPU grant + re-guard protocol).

### Step 6 — MTP on NVFP4 + order gate (GPU, the (vi)-lane surface)
Enable/verify `--spec mtp` on NVFP4: the MTP pool planes, `mtp_kv_` views, batched MTP
cache (a4066f5d area) with the new planes; S2-style order discriminator on NVFP4
(`tools/smoke/diag/s2_order_diag.sh` with `KV_DTYPE=nvfp4` + sizing math from
`tp2_budget.h` — derive the MAX_CONTEXT/KV_CAPACITY pair from required_bytes, do NOT
reuse the bf16 40000/45000 numbers blindly).
**Tests:** docs/131 battery subset on NVFP4 (T0 batteries are model-free — run them
regardless); INVASERT stress clean; S2-style gate: A1==A2==B1 exit 0 on NVFP4; MTP
acceptance on NVFP4 within noise of int8 on the same fixture.

### Step 7 — Quality gates (GPU)
Needle-in-haystack 5/5 at 128k on NVFP4 (synthetic needle harness; reuse tools/collect or
write a minimal one under `results/phase_nvfp4/`). AIME-style reasoning spot check (greedy,
a few problems, coherent + correct reasoning; the fork's 29/30 is their artifact + full
AIME, not our bar). Optional stretch: port the fork's LongBench v2 20-sample runner.
**Tests:** committed results dir per run with server config + clocks; needle 5/5 is a
gate, AIME coherence is a gate, LongBench is a stretch.

### Step 8 — VRAM measurement + closeout
Measure per-token KV bytes on NVFP4 via the preflight/probe figures + a real launch
(`required ≈ fixed + 9792/2 B/token/rank` under TP2 — verify the actual MiB/token against
the bf16 0.0618 MiB/token anchor from the (vi) handoff §4, expect roughly half the KV
share); record max_context that fits 16310 MiB/rank. `run_ci.sh` full gate (existing tiers
must be green — zero regression). Final docs note + report.
**Tests:** `bash tools/ops/run_ci.sh --full` green; VRAM numbers committed under
`results/phase_nvfp4/`; live launch command recorded in the report.

## 7. Constraints (non-negotiable)

- **Worktree only:** no edits in `/home/intel/ninfer/repo` outside the worktree (docs
  excluded — this work order itself is the sanctioned docs commit).
- **Live end-to-end before "done":** steps 4-8 each require the real server path (§6
  standard). Kernel parity alone does not count.
- **No damage:** commit per step, tree buildable at every commit, message names the step.
- **Existing tiers are frozen:** i8/bf16/kvarn dispatch, numerics, and budgets must be
  untouched (arms are additive). The (vi) fix and its gates are regression surface — any
  MTP-affecting commit runs the docs/131 T3.1 merge gate.
- **GPU protocol:** written coordinator grant per run-batch; re-guard at claim; release
  immediately after; never touch foreign processes; `df -h /` before big builds.
- **No threshold edits** (T19 ≥450 tok/s unchanged); no `pkill -f`; never test a server
  you did not start.
- **Fork is read-only reference:** port code, do not import fork files wholesale; every
  ported constant re-derived for our geometry. Attribute the fork in commit messages
  (e.g., "ported from gzenz/ninfer 4b882d0: <file>").

## 8. Definition of done (phase A)

1. Steps 1-8 committed to `wo/nvfp4-kv` with passing tests at each step, including the
   live-path tests (steps 4-8).
2. **Live proof:** a fresh `--kv-dtype nvfp4 --spec mtp` server launch serves greedy +
   sampled + a multi-turn MTP session through the changed path; launch command recorded.
3. Functional end state: `--kv-dtype nvfp4` selectable everywhere (serve/cli/bench via the
   one name table), refusal guard flipped, MTP order gate SAME on NVFP4, needle 5/5,
   battery green, existing tiers bit-untouched.
4. Measurement data committed to `results/phase_nvfp4/` (VRAM per-token figure, needle
   transcripts, S2 logs, clocks).
5. docs/131 T3.1 gates green on the final commit (T0 + INVASERT + T2(T=64)); run_ci.sh
   full green.
6. Report: one paragraph per step + key numbers table (per-token bytes, max_context fit,
   MTP acceptance, needle, AIME spot, clocks).
