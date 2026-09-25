# PREFILL BODY — decomposing the 645 ms/chunk in-layer non-GEMM budget (P2 design desk)

**Seat:** NO-GPU design desk, `amd/tp4-cure`, 2026-09-17. Read-only on src/ except this doc.
**Inputs:** P1's measured decomposition (`results/amd/coherence/PREFILL_OPTRACE_row.txt`,
banked bin `6c8ae21399516750`): steady per-chunk (T=128, TP4, 16 chunks of plen-1996 probe)

```
wall 1642.3 = gemm 798.8 (48.6%) + ar 70.7 (4.3%) + body 645.1 (39.3%) + gap 122.7 (7.5%)
```

**Mission:** name the 645.1 ms body op-by-op, design fixes, spec the decisive per-body-op
tracer (P2), and state the honest roofline to the new user goal: **prefill >= 500 tok/s**
(from 75.5 today).

---

## 1. The REAL prefill path at T=128 (dispatch map)

P1 already established the doc's `:2437` cite is the GDN **decode** arm (it `return;`s);
the prefill probe runs the TP arms. Per 128-token chunk, per rank, 64 layers
(48 GDN + 16 full), all in `src/targets/qwen3_6/impl/runtime/text_context_impl.h`:

**GDN layer** — `TextContext::gdn_mix_tp` (:2459), prefill arm (T>1):

| # | op | site | kernel(s) actually dispatched at T=128 |
|---|----|------|----------------------------------------|
| 1 | input rmsnorm + control (a/b) projection + gating | `Variant::gdn_norm_control_projection` :2492 -> `ops::gdn_norm_gating_proj` | rmsnorm warp kernel + **`bf16_gdn_gating_proj_gemm_mma_kernel`** (k27Routes MmaUnsplit route for T=128, `bf16_gdn_gating_proj_plan.cpp:368`) |
| 2 | qkvz projection [4096,5120]x128 | `Variant::gdn_input_projection_tp` :2662 (fused NVFP4 arm, variant_kernels.cpp:240) | tiled SIMT nvfp4 GEMM — **GEMM-traced** |
| 3 | unpack qkvz -> [qkv_conv 2560; z 1536] | `multi_gpu::tp_unpack_gdn_qkvz` :2700 | `unpack_gdn_qkvz_kernel` (tp_kernel.cu:416), grid-stride flattened, scalar bf16 copy |
| 4 | causal conv k=4 SiLU + state | `ops::causal_conv1d_silu_split3` :2715 | T=128 > `kCausalConvSequenceMaxTokens`=64 -> **prefill_pairs** (bf16x2) + prefill_state (`causal_conv1d.cu:38-78`) |
| 5 | g/beta slice unpack -> fp32 | `multi_gpu::tp_unpack_gbeta_strided` :2744 | `unpack_gbeta_strided_kernel` (tp_kernel.cu:333) |
| 6 | **delta-net recurrence** | `ops::gated_delta_net` :2753 | T=128 >= `kChunkSize`=64 -> **chunked trio**: l2norm(q)+l2norm(k), `launch_prepare_wy_wu`, `launch_state_passing`, `launch_output` (`gated_delta_net.cpp:253-313`) |
| 7 | gated rmsnorm | `ops::gated_rmsnorm` :2764 | gated rmsnorm kernel |
| 8 | out_proj RowK [5120,1536]x128 | `multi_gpu::tp_gemv` :2772 | tiled SIMT nvfp4 — **GEMM-traced** |
| 9 | TP allreduce (48/chunk) | :2825 | — **AR-traced** |

**Full-attention layer** — `TextContext::attn_mix_tp` (:2236): input rmsnorm :2245 ->
qkv projection (fused NVFP4, **GEMM-traced**) :2251 -> `tp_unpack_qkv_strided` :2261 ->
q/k rmsnorm :2278-2279 -> rope :2286 -> **prompt attention** :2420 -> `sigmoid_mul` :2442 ->
o_proj (**GEMM-traced**) :2448 -> AR :2454.

**MLP tail** (all 64 layers) — `mlp_tail` :3004: rmsnorm :3008 -> `Variant::post_mixer_tp`
(:3018; gate_up GEMM **traced**, `silu_mul` :481 of variant_kernels.cpp, down GEMM **traced**)
-> AR :3025.

Kernel count per chunk: ~9/layer GDN + ~8/layer full + ~2 mlp + 192 traced GEMM launches
~= 800-900 launches. Embed / finalnorm / last-chunk lm_head / MTP live in `gap`, not body.

## 2. The gates the mission asked about

- **Conv `kCausalConvSequenceMaxTokens=64`:** T=128 takes the `prefill_pairs` path — and it
  is FAST. bf16x2 lanes, grid (C/2/256)xT = 1280 CTAs, ~2.6 MB traffic => ~10-15 µs incl.
  the trailing state kernel and both launches. The conv is **cleared**; the 64-token gate is
  not the pathology. (E-13/E-14/E-15 already own its correctness cells.)
- **Delta-net `kChunkSize`=64:** T=128 fully engages the **chunked** path (T_full=128,
  tail=0) — the correct gate for prefill, but the chunked kernels themselves are the
  suspect (§4).
- **GQA `kSmallTChunkTokens`=6:** width=128, batch=1, q_heads=6 (TP4-local) ->
  `GqaAttentionRoute::Prompt` (`gqa_attention.cpp:515-526`) -> the gfx906 SIMT flash kernel
  `gqa_attention_prefill_simt_bf16_kernel` (self-documented "correctness-first; DPP
  ladders... stage-10 tuning items").

So: **no decode-class kernel is misrouted at T=128.** The body's problem is not routing —
it is that three of the kernels the prefill path lands on are **tensor-core-shaped code
running on gfx900 through a full software emulation, on starved grids.**

## 3. Ceiling prices: what the body WOULD cost at honest bandwidth

Geometry per rank at TP4: hidden 5120 -> one hidden plane at T=128 = 1.31 MB bf16.
Ceilings: 378 GB/s mem, 10.75 TF/s F32 (21.5 packed-fp16). Elementwise/norm ops are pure
bandwidth.

| op class | occ/chunk | traffic or FLOPs | ceiling price |
|---|---|---|---|
| unpack_qkvz + unpack_qkv + gbeta + pack copies | 48+16+48 | ~2.1+1.9+0.05 MB each | ~6+5+0.2 µs -> ~0.6 ms |
| conv pairs + state | 48x2 | ~2.6 MB | ~10 µs -> 0.5 ms |
| rmsnorm (input/q/k/mlp) + gated_rmsnorm | 64+64x2+48+48x2 | 2.6 MB / 1.2 MB each | ~3-7 µs -> ~1.3 ms |
| silu_mul + sigmoid_mul | 64+16 | 3.3 / 1.2 MB | ~0.4 ms |
| l2norm(q)+l2norm(k) | 48x2 | 0.26 MB | ~0.1 ms |
| delta-net workspace+state traffic (W/U/v_new/h_chunk/S) | 48 | ~4 MB | ~0.5 ms |
| **sum, bandwidth class** | | **~1.0 GB** | **~3 ms** |
+ eager launch/inter-kernel slack (~850 launches, fully enqueued back-to-back): ~5-15 ms
**=> the bandwidth+latency class cannot own more than ~15-20 ms of the 645.1.**
Everything else is COMPUTE kernels running far under ceiling. Three candidates exist; their
FLOPs are the census:

- delta-net chunked: ~0.121 GFLOP per (layer, chunk) x48 = **5.8 GFLOP/chunk**
- gating control proj [96,5120]x128: 0.126 GFLOP x48 = **6.1 GFLOP/chunk**
- GQA prompt (avg window 1088 over the 16 chunks): 0.143 GFLOP x16 = **2.3 GFLOP/chunk**
Total ~14 GFLOP — 0.9% of the chunk's 1.559 TFLOP GEMM work. At 10.75 TF/s that is 1.3 ms;
**645 ms implies these kernels run at ~20-25 GFLOP/s effective, ~0.2% of peak.** That is
the lm_head pathology class exactly (small_t at 5.5 GB/s vs 137 after the flip).

## 4. THE NAMED SUSPECTS (code evidence)

**S1 — delta-net chunked trio** (`src/ops/linear_attention/gated_delta_net/chunked/`).
Built for CUDA tensor cores: `mma_bf16`/`mma_tf32`, `ldmatrix_x2/x4`, `cp_async`
(`ops/common/mma.cuh`, `prepare_wy_wu.cuh:18-26`). On `__HIP__`/gfx900 **every one of those
is a software emulation** — ldmatrix = warp shuffles + 16-bit LDS reads (mma.cuh:24-132),
mma = per-lane shuffle-broadcast FMA loop (mma.cuh:217+, ~200+ cycles per 16x8x16 tile,
~28% per-warp efficiency before stalls). Grids at TP4-local H_v=12:
prepare `(NT=2, H_v=12)` = **24 CTAs** (8 warps) on a 64-CU die — most CUs idle;
state_passing `H_v*D_STRIPS` = 12x4 = **48 CTAs** (16 warps, 56 KB smem, serial 2-chunk
loop); output `(2, 12)` = **24 CTAs**. The specializations in the launchers
(`H_v==32`, `H_v>=48`) were tuned for world=1/2 shapes; world=4 falls into the generic arms.
Extra 3x: W is recomputed per v-head while only k_loc=4 k-heads exist
(`head_map` 12 v-heads over 4 qk-heads).

**S2 — gdn gating control projection** (`bf16_gdn_gating_proj_gemm_mma.cuh`).
Uses `ldmatrix_x2/x4` (:224/:234) = same emulated ldmatrix. The k27Routes catalog selects
an MmaUnsplit MMA route at T=128. 6.1 GFLOP/chunk on the emulation layer.

**S3 — GQA prompt SIMT flash** (`gqa_attention_prefill_bf16_gfx906.cuh`).
24 CTAs (width/32 x q_heads = 4x6), 128 threads; serial outer loop over 32-key tiles
(window/32 ~ 34 tiles avg); phase C is a 32-step shuffle-broadcast per row-tile. Worse: with
kv_heads local = 1, **all six q-head CTAs of the GQA group re-stage the identical K/V tile
every iteration** — 6x redundant HBM->LDS traffic for the same keys. Explicitly marked
untuned ("stage-10 tuning items").

**Bracket estimate to be settled by P2 (§5):** S1 ~250-450, S2 ~60-160, S3 ~80-180,
everything else ~10-20 -> spans 400-810 around the measured 645.1. The elementwise floor
(§3) is hard: fixes cannot push body below ~15 ms/chunk.

## 5. DECISIVE CHECK P2 — per-body-op event pairs (extend `NINFER_PREFILL_OPTRACE`)

House pattern (P1): env-gated, default OFF byte-identical, per-rank function-local static
`PrefillOpTrace`, parity-doubled event slots, race-free one-chunk-wide drain behind the
chunk-end `ctx_.synchronize()`. Extension, all in `text_context_impl.h` except two noted
sites:

- **Depth gate:** `NINFER_PREFILL_OPTRACE=2` arms the body pairs (=1 keeps today's
  wall/gemm/ar/body/gap exactly; unset unchanged).
- **Slots:** add `ev_body_op[2][kBodyOpSlots][2]` to `PrefillOpTrace`
  (`kBodyOpSlots = 512` >= 480 occurrences/chunk; ~1024 new cudaEvent_t/rank, same order as
  the existing 640 gemm events; created in `ensure()`, best-effort like P1).
- **Classification:** fixed enum of 12 classes, `body_op_begin(s, cls)/body_op_end(s, cls)`
  no-ops unless depth>=2 && active; each occurrence appends its elapsed ms to
  `sum_body_op[cls]` at drain (per-chunk means printed; occurrences counted).
- **Exact pair sites** (begin/end immediately around, same `if` guard grammar):

| cls | layers | bracket site |
|---|---|---|
| `gate` | 48 gdn | around `Variant::gdn_norm_control_projection` (:2492) |
| `upk` | 48 gdn | around `multi_gpu::tp_unpack_gdn_qkvz` (:2700) |
| `conv` | 48 gdn | around `ops::causal_conv1d_silu_split3` (:2715) |
| `gbet` | 48 gdn | around `multi_gpu::tp_unpack_gbeta_strided` (:2744) |
| `dn` | 48 gdn | around `ops::gated_delta_net` (:2753) |
| `gnorm` | 48 gdn | around `ops::gated_rmsnorm` (:2764) |
| `aunpack` | 16 full | around `multi_gpu::tp_unpack_qkv_strided` (:2261) |
| `qknorm` | 16 full | around q/k rmsnorm+rope block (:2278-2286) |
| `gqa` | 16 full | around the `ops::gqa_attention` prompt call (:2420) |
| `agate` | 16 full | around `ops::sigmoid_mul` (:2442) |
| `mnorm` | 64 mlp | around `ops::rmsnorm` in `mlp_tail` (:3008) |
| `mact` | 64 mlp | around `ops::silu_mul` inside `Variant::post_mixer_tp` (variant_kernels.cpp:481; same env static re-declared there — P1-style) |

- **Output grammar** (per rank, per chunk, at the existing drain points):
  `[PREFILL-BODY] rank=R chunk=C tok=T gate= upk= conv= gbet= dn= gnorm= aunpack= qknorm= gqa= agate= mnorm= mact= unbr=`
  where `unbr = body - sum(classes)` — **the intra-layer launch-slack number** (the share
  graph capture could reclaim inside layer bodies). Plus
  `[PREFILL-BODYSUM] rank=R n= mean_ms <same 13 columns>`.
- **Depth 3 (optional, same boot if dn owns >200):** pairs inside
  `gated_delta_net.cpp` around l2norms / `launch_prepare_wy_wu` / `launch_state_passing` /
  `launch_output`, and inside `gdn_norm_gating_proj` plan around the gemm_mma launch —
  splits S1/S2 internals. Print `[PREFILL-DN3]`.
- **Expected readout:** the 12 columns sum to body 645.1 within ~5%; falsifiers:
  (a) `dn < 150` AND `gqa < 60` AND `gate < 40` => the emulation thesis DIES and `unbr`
  owns (launch-slack class -> graph capture is the lever instead);
  (b) `gate`-`mnorm` elementwise columns > 100 => the bandwidth census (§3) is wrong and
  the wrappers hide a pathology;
  (c) else S1+S2+S3 own => fixes below proceed in that order.
- **Cost:** ~480 event pairs + ~480 elapsed_ms reads per chunk when ON; zero when OFF.
- **Bank:** `results/amd/coherence/PREFILL_BODY_row.txt` + serve log, per P1's row format.

## 6. Fix designs (diff sketches — no src/ edits by this desk)

**FIX-A — gfx900-native chunked delta-net (S1).** New
`chunked/*_simt.cuh` selected at `launch_chunked` by a has-MFMA device gate
(gfx900 -> simt; gfx908+ keeps the mma path verbatim):
```
- drop mma/ldmatrix/cp_async: register-tiled outer-product FMA on __nv_bfloat162->float2
  converts + fmaf; optional v_pk_fma_f16 (HFMA2) for the bf16 A@B products (2x rate).
- re-grid to fill 64 CUs: prepare (NT x H_v x 8 d-strips)=192 CTAs/4 warps;
  state_passing NStrip 32->16 (96 CTAs/8 warps); output +4 d-panels (96 CTAs).
- dedupe: compute W once per k-head (4) in workspace, share across its 3 v-heads
  (-2/3 W-side FLOPs; U stays per v-head).
```
Predicted: ~0.2% -> 3-8% of peak => S1 lands ~645 -> **~20-60 ms** for the `dn` class.
Cell owed: existing chunked-vs-recurrent parity cell (E-4 class) RED/GREEN at T=128,
plus accuracy fingerprint 202/0.985/2.97 family.

**FIX-B — GQA prompt split-K over key tiles (S3).** In the launcher + kernel:
```
- grid (width/Br, q_heads, splits): each CTA covers a DISJOINT key-tile range
  [split*win/S, (split+1)*win/S); partials into the EXISTING SmallT acc/m/l workspace
  (allocate_small_t_workspace already exists), + one flash-reduce launch (same shape the
  ChunkedSmallT route already runs at width<=16).
- 6 q-head CTAs of a kv-group no longer walk the same window serially: 24 -> 24*S CTAs
  (S=4 -> 96), ~Sx wall cut on the attention, K/V staged once per (tile, split).
```
Predicted: `gqa` ~80-180 -> **~15-40 ms**. Cell: prompt-vs-smallT bit-parity at a
plen where both routes are legal (T<=6 forced route), plus existing greedy-text gate.

**FIX-C — gating projection route (S2).** `bf16_gdn_gating_proj_plan.cpp`: at
T >= 32 route to the plain SIMT gemv/gemm arm (`bf16_gdn_gating_proj_gemv_launch`, :218)
or add a rowsplit-SIMT schedule that avoids ldmatrix; alternative (riskier, phase 2):
fold a/b rows into the sharded qkvz projection GEMM's N dimension (96 extra rows) and
slice g/beta out of qkv — kills 48 launches/chunk entirely.
Predicted: `gate` -> **~5-15 ms**. Cell: gating numeric parity (fp32 acc) + the
fingerprint gate.

Order of attack = S1 (biggest, self-contained op), S3, S2. All three are env-gatable
behind a `NINFER_GDN_SIMT=1`-style flag pending their cells; default OFF byte-identical.

## 7. The honest 500 tok/s roofline (per rank, per 128-token chunk, budget 256 ms)

Code FLOP census (this desk, from the shapes in §1): **GEMM class = 1.559 TFLOP/chunk/rank**
(48x(5.37 qkvz + 2.01 out + 11.43 gate_up + 5.71 down) + 16x(4.71 qkv + 2.01 o + 17.14 mlp)
GFLOP; intermediate 17408, I_local 4352). Measured 798.8 ms => **1.95 TF/s effective**
(the 2.93 TF/s W8 bench number is the big-shape kernel best, not the per-chunk mix).

| class | today | 500-budget | honest mechanism |
|---|---|---|---|
| gemm | 798.8 | <= ~160 | needs **9.8 TF/s = 91% of the 10.75 F32 peak** — NOT reachable on the F32 SIMT path (tuned ceiling maybe 3-4.5 TF/s). Requires the **HFMA2 packed-fp16 tiled path** (dequant NVFP4->fp16 in-kernel, v_pk_fma_f16 dot, fp32 accum): 21.5 TF/s peak at >=46% eff -> 140-170 ms. Numerics cell owed (bf16->fp16 range under per-block scales). |
| body | 645.1 | <= ~60 | FIX-A + FIX-B + FIX-C -> `dn` 20-60 + `gqa` 15-40 + `gate` 5-15 + elementwise floor ~15 = **55-90 ms** realistic; 60 is aggressive-but-possible |
| ar | 70.7 | <= ~35 | one-shot AR / RCCL env A-B (the older Fix B; 1.31 MB x 128) |
| gap | 122.7 | <= ~15 | chunk graph capture (P1's demoted Fix A — bounded to this class) |
| embed/finalnorm/lm_head-last | in gap | <= ~10 | existing simt lm_head path (65.8 -> 2.5 ms decode class) |

Sequence that reaches 500:
1. **P2 boot** (§5) — names the 645 exactly; no fix ships before it (decisive-measurement law).
2. **FIX-A/B/C** -> body 645 -> ~60; + graph capture (gap 123 -> ~15-20) + AR A/B (71 -> ~35):
   wall ~799+60+35+15 = **~909 ms -> 141 tok/s**.
3. **HFMA2 GEMM** (the only door to >= 200 tok/s): 799 -> ~150-170 -> wall **~265 ms ->
   ~480 tok/s**; last-mile graph/AR polish crosses 500.
4. Honest verdict: **500 tok/s is unreachable on the F32 SIMT compute path** — with body=0,
   ar=0, gap=0, today's GEMM alone walls at 1.559 TFLOP / 2.93 TF/s = 532 ms -> **~240 tok/s;
   with a heroic F32 SIMT tuning (4.5 TF/s) ~330 tok/s.** The packed-fp16 GEMM rewrite is
   load-bearing for the user goal, not optional polish.
5. **Physical floor** (this die): GEMM at 100% packed-fp16 = 72.5 ms + body floor ~15 +
   ar ~20 + gap ~5 -> **~110 ms/chunk ~ 1150 tok/s absolute**; at a realistic sustained
   50-60% packed efficiency the machine tops out in the **~500-650 tok/s** class. 500 is
   therefore a legitimate goal with ~30-40% headroom above it — but only on the far side of
   two compute-class rewrites (body emulation removal AND packed-fp16 GEMM).

## 8. Sources / provenance

- P1 measurement: `results/amd/coherence/PREFILL_OPTRACE_row.txt`, `P1_PREFILL_NOTES.md`
  (banked bin 6c8ae21399516750, HEAD 61c1ecc0b).
- All file:line cites verified this session on `amd/tp4-cure` @ 7bca3db72+ (working tree).
- No GPU work performed (design desk). P2 runbook = §5, same posture as P1's
  (grant, gpu_guard, bank boot, cool window, clocks sideband).
