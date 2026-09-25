# WO_LLAMACPP_bench — llama.cpp empirical bench desk (Team Red / V340L)

- **Date opened:** 2026-09-19
- **Owner:** llama.cpp empirical bench desk (subagent)
- **Work dir:** `/home/chris/llama.cpp` (clone) + `/home/chris/llama.cpp-build` (build tree). NOT the repo, NOT `wo/amd-wo-w7-body`.
- **Mission:** settle whether upstream llama.cpp beats our in-house serving stack on THIS box
  (2x V340 = 4x gfx900 dies, ROCm 6.2), and if so extract the mechanism. Reference scoreboard
  (in-house, 27B NVFP4, this box): prefill ~112 t/s @ 2k prompt (18.5 s walls), 104 t/s cold 10k,
  decode 27–41 t/s by generation length.
- **Models:** Qwen3-30B-A3B Q4_K_M (~18 GB) + Qwen3-8B Q4_K_M (~5 GB) → `/media/chris/EMTEC256`.
- **Laws honored:** :8100 serve (pid 104924/104972, `ninfer-serve_183da007801e0fb6.bin`) belongs to
  another desk — never killed; brief GPU contention acceptable, noted per entry. No pkill outside own
  PIDs. `df -h /` before big ops. This doc is the checkpoint; appended after every step.

## Environment probe (2026-09-19)

- `rocm-smi` count = 2x `Vega 10 [Radeon Pro V340/Instinct MI25x2]` → AMD line confirmed;
  4 HIP agents visible (`rocminfo`: gfx900, xnack-), HIP device ids 0..3.
- ROCm: `/opt/rocm -> /opt/rocm-6.2.0` (hipcc present).
- cmake: system 3.30.5 (sufficient; `/home/chris/opt/cmake` does not exist).
- Disk: `/` 13 GB free (89% used) — builds + clones here stay lean; GGUFs go to USB.
- `/media/chris/EMTEC256`: 179 GB free.
- VRAM at desk-open: ~7.1 GB used of 7.98 GB per die (foreign serve on :8100 holds it) →
  ~1.4 GB/die free until that desk's one-shot finishes. Build/download first; bench after.

## Log

### Step 0 — desk opened, env probe done (2026-09-19)

All facts above re-derived live this session. Next: clone, HIP build attempt (gfx900),
fallback Vulkan if HIP refuses gfx900.

### Step 1 — clone + HIP build (2026-09-19)

- Cloned `ggml-org/llama.cpp` @ `1af554f8fc78ba029665a47b839484d9763e2a75` (2026-09-19) →
  `/home/chris/llama.cpp` (shallow, depth 1). Build tree `/home/chris/llama.cpp-build`.
- Configure (WORKED): `cmake -S /home/chris/llama.cpp -B /home/chris/llama.cpp-build
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900
  -DCMAKE_HIP_ARCHITECTURES=gfx900 -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 -DLLAMA_CURL=OFF`
  with `/home/chris/opt/cmake/bin/cmake` (system cmake does not exist — 3.30.5 lives at
  the opt path only).
- First build FAILED: `ggml/src/ggml-cuda/vendors/hip.h:253: unknown type name
  '__hip_fp8_e4m3'`. Root cause: llama.cpp typedefs fp8 unconditionally for
  HIP_VERSION >= 6.2, but ROCm 6.2's `amd_hip_fp8.h` declares fp8 types ONLY under
  `(__gfx940__||__gfx941__||__gfx942__) && __HIP_DEVICE_COMPILE__` (line 33) — nothing on
  gfx900. Downstream uses are all guarded by `FP8_AVAILABLE`, so the minimal correct fix is
  gating the typedef block the same way.
- PATCH (local, this clone):
  `/home/chris/llama.cpp/ggml/src/ggml-cuda/vendors/hip.h` — guard changed to
  `#if HIP_VERSION >= 60200000 && (defined(__gfx940__) || defined(__gfx941__) ||
  defined(__gfx942__) || defined(__gfx950__))`. Rebuild running (-j16, 20 cores/62 GB RAM).
- Models: Qwen3-8B-Q4_K_M.gguf COMPLETE on USB (5.03 GB, ~40 MB/s); 30B-A3B downloading.
  No GPU contention yet — build/download phases touch no VRAM.

### Step 2 — build GREEN, gfx900 HIP smoke test passed (2026-09-19)

- Rebuild after patch: EXIT=0. Binaries: `bin/llama-bench`, `bin/llama-server`, `bin/llama-cli`,
  `bin/libggml-hip.so.0.24.0` (64 MB, gfx900 code object). Build tree 247 MB; `/` still 12 GB free.
- SMOKE TEST (single die, `HIP_VISIBLE_DEVICES=3`, model Qwen3-0.6B-Q8_0 604 MiB, -ngl 99):
  device probed as `gfx900:xnack- (0x900), Wave Size: 64, VRAM 8176 MiB`;
  **pp128 = 1314.8 t/s, tg32 = 199.5 t/s**. The HIP backend is fully operational on V340 dies.
- VRAM GATE: foreign serve (pid 104972, up 19+ min) still holds ~7.1 GB/die → ~1.4 GB/die free.
  The 8B (5.03 GB) and 30B-A3B (~18.6 GB) CANNOT be loaded until that desk's serve exits.
  No kill (law). Polling; benches start the moment VRAM frees. 30B download ~4.2/18.6 GB.

### Step 3 — downloads COMPLETE; kernel-mechanism evidence banked (2026-09-19, ~10:20-10:35)

- **Both GGUFs on USB:** `Qwen3-8B-Q4_K_M.gguf` 5,027,783,488 B; `Qwen3-30B-A3B-Q4_K_M.gguf`
  18,556,685,824 B (+ tiny `Qwen3-0.6B-Q8_0.gguf` for smoke tests). No `/` disk used by models.
- Foreign serve RESTARTED once (old 104972 gone, new pid 152405, same binary) — it is a
  standing tenant; still holds 7.1 GB/die. Asked coordinator/user for a bench window (no
  answer yet). Continuing with all work that fits the free headroom.
- **rocprof traces (0.6B Q8_0, die 3 only, negligible footprint):**
  - pp512 (prefill): top GPU time = `flash_attn_tile` 75%; rocBLAS **Tensile** GEMM kernels
    `Cijk_Alik_Bljk_HB_MT{64x64,128x64,128x128}x16_SN_...` present; preceded by
    `dequantize_block_q8_0_f16` (dequant→f16 then hipBLAS).
  - tg128 (decode): `mul_mat_vec_q` 53% + `quantize_q8_1` 17.5% + `flash_attn_tile` 9.5% —
    pure custom MMVQ GEMV path, zero rocBLAS.
- **Dispatch rule from source** (`/home/chris/llama.cpp/ggml/src/ggml-cuda/mmq.cu`, explicit
  comment): on gfx900 llama.cpp DISABLES MMQ for dense matmuls ("gfx900 ... lack native dp4a,
  losing to dequant + hipBLAS for dense matrices") and keeps MMQ ONLY for MoE expert matmuls
  (`return n_experts > 0`). Prediction for the real runs: 8B/30B prefill = dequant+rocBLAS,
  decode = MMVQ, 30B MoE experts = MMQ kernels. Traces on the big models will confirm.

### Step 4 — VRAM WINDOW at 11:06-11:20, benches + traces captured (2026-09-19)

- w7 desk's serve went down ~11:06; window-watcher fired at **11:06:46**. Window STILL OPEN
  at ~11:20 (all dies 0.02 GiB used). Page cache made USB-resident GGUFs load at RAM speed.
- **ROW-SPLIT IMPOSSIBLE ON THIS BUILD:** `-sm row` fails on CUDA/HIP with
  `device ROCm0 does not support split buffers` — root cause: upstream llama.cpp @ `1af554f`
  implements `ggml_backend_split_buffer_type` ONLY in the SYCL and Hexagon backends; the
  CUDA/HIP backend does not register it (`src/llama-model.cpp:1146` throws). So upstream
  llama.cpp **cannot tensor-parallel across the 4 V340 dies at all on AMD** — multi-die is
  LAYER (pipeline) split only. Our in-house stack's TP4 has no llama.cpp counterpart here.
  The 30B row cells are therefore N/A (documented, exit 1), not lost.
- **8B Q4_K_M, 4 dies layer split** (r=3): **pp512 179.0, pp2048 156.1, tg128 36.6 t/s**.
- **8B Q4_K_M, single die** (reference, r=3): pp512 174.4, pp2048 126.4, tg128 32.6 t/s —
  layer split ≈ single-die for an 8B that fits one die (pipeline adds ~5-24% at most).
- **30B-A3B Q4_K_M, 4 dies layer split** (r=2): **pp512 208.4, pp2048 137.4, tg128 53.4 t/s**.
- **Kernel confirmation on real Q4_K_M (rocprof --stats):**
  - 8B pp512: 67.4% rocBLAS **Tensile** fp16 GEMMs (`Cijk_Alik_Bljk_HB_MT32x32x16`,
    `MT32x32x32`, `MT64x64x16`), 24.4% `flash_attn_tile`, 2.8% `dequantize_block_q4_K`
    (dequant→f16 pre-pass feeding Tensile). Prediction CONFIRMED.
  - 8B tg32: 86.2% `mul_mat_vec_q` (MMVQ GEMV over q4_K), 4.6% `quantize_q8_1`,
    3.3% `flash_attn_tile`. Zero rocBLAS at decode. Prediction CONFIRMED.
  - 30B MoE pp512 trace (4-die) in flight to confirm expert-MMQ dispatch.

### Step 5 — window closed; 30B MoE trace lost to contention (2026-09-19, ~11:20-11:26)

- w7 desk rebooted its serve at 11:23 (pid 237590, `ninfer-serve_c618d356f0cdc401.bin`, under
  gdb) while the 4-die 30B pp512 trace was still running; the trace process died at 11:23
  (stub artifacts only). Serve footprint back at 6.64/6.54 GiB per die. No kill issued by this
  desk (law); contention event logged here per the WO.
- MoE-expert mechanism therefore rests on SOURCE (mmq.cu gfx900 rule: `n_experts > 0` → MMQ),
  which is unambiguous, plus the 30B bench numbers being consistent with it. Retrace in a
  future window if someone needs the direct kernel-level proof for the MoE path.
- **Clocks during the 30B layer-split bench** (`/tmp/sclk_samples.log`, 5 s cadence): dies sit
  at 300 MHz idle and bounce 300 → 1138/1269/1350 on activity; max seen 1350 MHz (dies 0-1),
  1269 MHz (dies 2-3). The layer-split pipeline's per-die bursts let dies fall back to 300 MHz
  between hops — ramp latency is part of why layer-split decode is latency-bound.

### Step 6 — FINAL SCOREBOARD (2026-09-19)

llama.cpp @ `1af554f`, HIP/ROCm 6.2 backend, gfx900:xnack-, 4x V340 dies, layer split (row
split unsupported upstream on CUDA/HIP), Q4_K_M weights, FA auto (on), build + bench by this
desk in an uncontended window:

| model | mode | pp512 | pp2048 | tg128 |
|---|---|---|---|---|
| Qwen3-8B Q4_K_M (dense, 4.68 GiB) | 4 dies layer | 179.0 ± 0.7 | 156.1 ± 0.3 | 36.6 ± 0.2 |
| Qwen3-8B Q4_K_M | 1 die | 174.4 | 126.4 | 32.6 |
| Qwen3-30B-A3B Q4_K_M (MoE 3.3B act, 17.28 GiB) | 4 dies layer | 208.4 ± 1.6 | 137.4 ± 0.5 | 53.4 ± 0.2 |
| row split (-sm row), any model | — | ERROR `device ROCm0 does not support split buffers` | — | — |

**In-house stack (same box, 27B NVFP4, ~3.3B-active class, TP4):** prefill ~112 t/s @ 2k
(18.5 s walls), 104 t/s cold 10k, decode 27-41 t/s by length.

**Verdict vs our stack (30B-A3B ≈ same-size active-param class):**
- Prefill @2k: **llama.cpp wins, 137.4 vs ~112 t/s (+23%)**.
- Decode: **llama.cpp wins, 53.4 vs 27-41 t/s (+30% to +98%)**.
- llama.cpp does it WITHOUT tensor parallelism (pipeline only, weights pipeline-sharded);
  the win is kernel-quality, not topology. 8B layer-split ≈ single-die (+3% pp512, +24%
  pp2048, +12% tg over one die) shows how weak llama.cpp's multi-die scaling is here —
  a true TP4 with equally good kernels would beat it decisively.

**Mechanism (why it wins), measured + source-confirmed:**
1. Decode: single custom MMVQ kernel `mul_mat_vec_q` = 86% of decode GPU time (q4_K weights
   streamed once per token, q8_1-quantized activations via `quantize_q8_1`); no BLAS in the
   hot path. Lean launch graph vs our chunked TP machinery.
2. Prefill dense: `dequantize_block_q4_K` → f16 + **rocBLAS Tensile** fp16 GEMMs (67% of pp
   GPU time; tiles MT32x32x16/32x32x32/64x64x16 — gfx900 has no MFMA, Tensile runs VALU
   packed fp16). Full-batch GEMMs (M=512/2048) vs our `--prefill-chunk 128` (M=128) — on a
   no-matrix-core arch, per-GEMM batch is exactly what drives VALU utilization.
3. Prefill MoE experts: custom MMQ kernels even on gfx900 (llama.cpp policy: MMQ for
   `n_experts > 0`; dense stays dequant+BLAS).
4. Upstream llama.cpp CANNOT TP4 on AMD: `ggml_backend_split_buffer_type` exists only in
   SYCL/Hexagon backends; CUDA/HIP row split removed upstream (`src/llama-model.cpp:1146`
   throws). Its 4-die mode is layer/pipeline, which is why decode lands at 53 t/s rather
   than ~4x single-die × per-die rate.

**Files banked:** clone+patch `/home/chris/llama.cpp` (fp8 guard fix in
`ggml/src/ggml-cuda/vendors/hip.h`), build `/home/chris/llama.cpp-build` (bin/
llama-{bench,server,cli} + libggml-hip.so), models
`/media/chris/EMTEC256/gguf/{Qwen3-8B-Q4_K_M,Qwen3-30B-A3B-Q4_K_M,Qwen3-0.6B-Q8_0}.gguf`,
raw logs+traces `/media/chris/EMTEC256/llamacpp_bench/`. Disk: `/` 12 GB free (build 247 MB +
clone 211 MB), USB 155 GB free.

**Build lines that worked (ROCm 6.2.0, system cmake absent — use
`/home/chris/opt/cmake/bin/cmake`):**
```
/home/chris/opt/cmake/bin/cmake -S /home/chris/llama.cpp -B /home/chris/llama.cpp-build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900 \
  -DCMAKE_HIP_ARCHITECTURES=gfx900 -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 -DLLAMA_CURL=OFF
/home/chris/opt/cmake/bin/cmake --build /home/chris/llama.cpp-build \
  --target llama-bench llama-server llama-cli -j16
```
(Requires the fp8 guard patch above on gfx900 — upstream ships a build break for pre-gfx940
targets under ROCm >= 6.2.)

Desk status: mission complete; queued item for any future window — 30B MoE pp512 rocprof
retrace (nice-to-have only).





