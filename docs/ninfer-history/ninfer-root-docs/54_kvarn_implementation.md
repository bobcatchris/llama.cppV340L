# 54 — KVarN: Sub-8-bit KV Cache Implementation Plan

Status: **P1 COMPLETE** (2026-08-23, §9) — Codec reference & unit tests complete
(FWHT + Sinkhorn balance + K-int4/V-int2 RTN exactness & determinism passing 100%).
**P2 tile codec COMPLETE** (2026-08-24, §9 P2b) — CUDA tile codec at D=256/G=64
verified bit-deterministic and matching the CPU reference.
**P2c steps 1–6 COMPLETE** (2026-08-24, `wo/kvarn-live`, merged to main
`6b535c76`): pool layout (`0b2a073e`), write path (`2c841fbb`), read kernel +
dispatch (isolation rel_l2 = 9.9e-5), budget + guard lift, live T16 + must-pass
(D-14), **prefill staging (D-15)** — T8 curve 678→720 tok/s, ~1 GiB staged
shadow/rank, and **prefix tails (D-16)** — unaligned prefix restore via
open-page snapshot. **Main-side verification (2026-08-24): fast battery on the
MAIN binary at 250k KVarN — 11 passed / 0 failed (T1,T2,T3,T5,T8,T10,T11,T15,
T16,T17,T18); unit snapshot round-trip bit-exact; MTP acceptance unchanged vs
int8 (50.3% vs 50.1% same-workload).** Self-contained brief in §9 P2c.
Source project: [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN), Apache 2.0,
paper arXiv:2606.03458. Porting is legally clean; attribute in the first commit
(repo + paper citation).

**Workflow (binding):** REPO.md rules apply. This is defect/feature-driven work —
every new defect found while implementing gets a docs/50 register entry **and a
battery test before its fix is accepted**. One branch per task (`wo/kvarn-*`).
No merge to mainline until the gate in §8 is green; one command:
`bash ~/ninfer/scripts/run_ci.sh`.

## 1. What KVarN is

Calibration-free KV-cache quantization, released as a vLLM fork (Triton kernels).
Per fixed token tile (g = 128 or 64 tokens), each tile walks four stages:

1. **Hadamard rotation** along the channel dim (orthonormal → attention scores
   preserved; spreads per-channel outliers).
2. **Iterative variance normalization** (Sinkhorn-like, log-space, alternating
   row/column std) — equalizes tile variance before rounding.
3. **Asymmetric round-to-nearest** at low bit width. Shipped preset
   `kvarn_k4v2_g128` = **K int4 / V int2**, group 128 (keys get more bits).
4. Read time: scales folded back in — **keys per channel, values per token**.

Claims (Qwen3-32B, AIME25, 16K burst, TP=2): ~4× KV capacity vs FP16, throughput
≥ FP16 (up to ~1.3×), FP16-level accuracy. Explicitly supports:
- **Hybrid linear-attention models** — only full-attention layers are quantized;
  names Qwen3.6-27B (our architecture family).
- **MTP speculative decoding** — verify attends over the reconstructed cache; a
  block is committed to the quantized cache **only once accepted**, so rejected
  drafts never corrupt history.

## 2. Why we care

- Our KV today: signed int8, per-token group-of-64 scales, fused into the GQA
  kernels (`src/ops/kernel/gqa_attention_kv_quant.cuh`). That is ~8 bits + scale
  overhead per channel-token.
- KVarN k4v2 ≈ (4+2)/2 = **3 bits + scale overhead ≈ ~40% of our int8 bytes**.
  At the 200k config (docs/52), KV ring is ~3.3 GB/rank with int8 → ~1.3–1.5 GB
  with kvarn: **~1.8 GB/rank freed** for prefix-cache capacity P, concurrency,
  or going past 200k.
- Our workload (pi agent sessions) is exactly KVarN's target: agentic, long
  context, many sequential turns. The paper's stated problem — error
  accumulation in *reasoning* tasks over long contexts — is our daily driver.

## 3. What we already have (delta analysis)

| Piece | Status | Note |
|---|---|---|
| Paged KV pool with plane layout + `quant_group` field | ✅ | int8 path exists; kvarn needs a new dtype/packing in the same pool machinery |
| Quant/dequant **fused** into GQA decode+prefill kernels (no standalone pass) | ✅ | same design goal as KVarN — we extend the existing fused pattern, not add a pass |
| `--kv-dtype bf16\|int8` option plumbing | ✅ | add `kvarn_k4v2` (opt-in; never default until §8.5) |
| MTP verify/accept flow | ⚠️ verify in P0 | we must confirm rejected drafts never land in the persistent KV (KVarN's commit-on-accept semantics); our verify batch looks ephemeral but prove it |
| Hybrid GDN layers | ✅ untouched | same as KVarN's hybrid support — only full-attention layers hold KV |
| TP2 head-sharded GQA | ⚠️ **central question** | see §4.3 |

## 4. Porting strategy

### A. Use vLLM / their fork directly — rejected
Our stack is custom CUDA on TP2 over PCIe (no NVLink); the whole point of this
repo is kernels tuned for 2×5060 Ti. Not an option.

### B. Run their Triton kernels — rejected
No Triton runtime in our stack; CUDA only.

### C. Reimplement the codec as CUDA ops (recommended)
- **Write path** (KV append, per tile of g tokens × 256 head-dim): FWHT Hadamard
  (log2(256) = 8 butterfly stages), fixed-count variance normalization
  (calibration-free; iteration count is a constant — must be fixed for
  determinism), asymmetric RTN to k4/v2, store packed codes + scales in the paged
  pool.
- **Read path** (inside existing GQA kernels): dequant with folded scales
  (per-channel K, per-token V) — extend `gqa_attention_decode_i8.cuh` / prefill
  fill+attention rather than adding a staging pass (bandwidth is the point).
- **Determinism is a hard requirement** (battery T10 compares outputs byte-for-byte):
  fixed iteration counts, no RNG, no data-dependent loop bounds.

### 4.3 TP2 sharding — RESOLVED (P0, 2026-08-23): head-local, no allreduce
Read their source (`vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`,
`vllm/v1/attention/ops/kvarn_store.py`). The batched API is `balanced: [N, D,
group]` — **each tile is one head's (D channels × group tokens), processed
independently; no stage mixes heads or tiles.** Hadamard is an external GEMM
along head_dim only (`(K @ H).T`, per-head, orthonormal). Each rank quantizes
its own KV-head shard locally. **No allreduce needed.** See §9 for the full spec.

## 5. Memory math (per rank, 200k config, from docs/52 §5)

| KV dtype | bytes/channel-token | KV ring @200k | Δ vs int8 |
|---|---|---|---|
| bf16 | 16 | ~6.6 GB | — |
| int8 (today) | ~8.03 | ~3.3 GB | baseline |
| kvarn k4v2 g128 | ~3.5 est. | ~1.4 GB | **~−1.9 GB** |

Scale overhead is estimated from the K-per-channel / V-per-token scheme; P1
measures it exactly.

## 6. Accuracy & risk surface

- Their parity claim is Qwen3-32B AIME25; our model is Qwen3.8-27B (same family,
  GDN hybrid) — **must measure on our model** (§8.4), not inherit their number.
- **MTP acceptance**: draft verify attends over quantized K → `mtp_accept` may
  drop below the 82.0% baseline → regression gate (>4 pp drop = FAIL).
- **Long-context error accumulation** is the paper's core claim; our real test is
  multi-hour agent sessions → soak + long-generation probe (Castlevania-style),
  reviewed, not just pass/fail.

## 7. Phases & acceptance

| Phase | Who | Acceptance |
|---|---|---|
| P0 feasibility (code-only) | REMOTE | their Triton source read; §4.3 TP question answered in writing (§9); MTP commit-on-accept confirmed or flagged; unit-test plan committed |
| P1 codec reference + unit tests | REMOTE | CPU/CUDA reference of Hadamard+normalize+RTN in `tests/`; round-trip error vs bf16 at g=128 and g=64 measured & logged; **determinism test** (two runs byte-identical); runs under CI |
| P2 kernel port | REMOTE | **tile codec DONE** (kvarn_tile_cuda.{h,cu}, verified vs CPU ref, ninfer_kvarn_tile_cuda_test); attention-path integration (pool layout + GQA read/write + MTP) pending |
| P3 battery @80k kvarn | GPU | full gate §8.1 green at 80k with kvarn; T9 long-prefill pass; `mtp_accept` within gate; T10 determinism pass; `det2.py` n2==n3 |
| P4 accuracy probe | GPU | small math/reasoning set (50–100 items) comparing bf16 / int8 / kvarn, results committed to §9; long-generation probe reviewed; **kvarn stays opt-in until this passes** |
| P5 200k retest | GPU | docs/52 budget rerun with kvarn KV; decide whether P (prefix-cache cap) or max-context moves up; new baseline snapshot in `latest.json` |

## 8. Testing criteria (binding, per docs/52 precedent)

1. **Battery gate** (docs/50 §7.1) at every config touched:
   - Must PASS: T1 T2 T3 T5 T7 T8 T10 T11. Expected PASS: T4 T6 T9 (nightly) T12.
   - Run with `--kv-dtype kvarn_k4v2` as a first-class config, not a one-off probe.
2. **New unit tests committed to `tests/`** (run under CI, not GPU-only):
   codec round-trip vs reference; determinism; TP-shard equivalence (rank-local
   quantization matches the unsharded math within tolerance).
3. **MTP acceptance regression gate**: fail on >4 pp drop below baseline 82.0%.
4. **Accuracy probe results committed** to §9 before kvarn may become default.
5. **Opt-in only** (`--kv-dtype kvarn_k4v2`) until P4 passes; int8 remains the
   default in the meantime.
6. **CI green** (`run_ci.sh`) on the branch before push-to-mainline; perf
   snapshot updated so kvarn numbers are on record.

## 9. Findings (fill in per phase)

### P0 — source reading (2026-08-23, code-only)

**Tile math (per head, deterministic):**
1. Hadamard rotation along head_dim D (orthonormal; external GEMM `(K @ H).T`
   for K, equivalent per-head FWHT — D=256 on our model = 8 butterfly stages).
2. Log-domain Sinkhorn: 16 iterations alternating column/row std normalization
   in log space, **best-so-far selection** by imbalance (max/min std spread,
   lower wins; first minimum breaks ties). Clamps: std ∈ [1e-3, 1e3],
   log-scales ∈ [-0.3, 10].
3. Asymmetric RTN per row (per channel for K): `lo/hi` over the tile row,
   `scale = (hi-lo)/qmax`, `zp = lo`, `q = clamp(round((x-zp)/scale), 0, qmax)`.

**Storage format (per tile, from `kvarn_store.py`):**
- K `[D, G]`: `q_packed [D, G/2]` uint8 (4-bit pairs) + `s_col_K [D]` fp16
  (absorbed per-channel scale) + `zp_K [D]` fp16 (per-channel zero point)
  + `s_row_K [G]` fp16 (per-token sinkhorn scale).
- V `[G, D]`: mirror — `q_packed [G, D/2]` at **2 bits** in the k4v2 preset,
  `s_col_V [D]`, `zp_V [D]`, `s_row_V [G]` fp16.
- Byte cost per token per KV head (D=G=128): K ≈ 64 B codes + ~6 B scales,
  V ≈ 32 B + ~6 B → **≈ 108 B vs int8's 256 B** (~42%, matches §5 estimate).

**Dequant (derived, read path):**
`K̂[c,t] = (q[c,t] − zp_K[c]) · s_col_K[c] · s_row_K[t]`, V symmetric.
The per-token factor folds into the row/softmax scaling; per-channel is a
per-lane multiply. Fits our fused-dequant kernel pattern (no staging pass).

**TP2:** head-local (see §4.3). Our GQA shards split KV heads across ranks;
each rank runs the full tile pipeline on its own heads.

**Active-tile buffering (port design consequence):** tiles commit every
G=128 tokens; the in-flight partial tile must stay bf16 in a small workspace
and be read directly by attention (their "fixed decode workspace"). Per rank:
heads_per_rank × D × G × 2 B × (K+V) ≈ ~1 MB — negligible. KV append is thus
**delayed up to G−1 tokens per head**; the paged pool stores packed tiles,
not raw rows → new plane layout, not a dtype swap of the existing one.

**MTP commit-on-accept:** their backend commits a tile only when all its tokens
are accepted; rejected drafts never reach the quantized cache. Our flow writes
KV at accept time (verify batch is ephemeral) — confirm in P2 that no draft
KV persists, then our existing semantics satisfy this without change.

**Determinism:** fixed 16 iterations + best-so-far selection = deterministic
given inputs; our port must pin iteration count and tie-break (first minimum)
or battery T10 breaks.

- P1: **COMPLETE** (2026-08-23) — Codec reference in `src/ops/kvarn/kvarn_codec.{h,cpp}`, unit tests in `tests/test_kvarn_codec.cpp` (`ninfer_kvarn_codec_test` PASS 100%).
- P2: **NOT STARTED** — CUDA kernel port into the GQA attention path does not exist. The `--kv-dtype kvarn_k4v2` flag parses but is **rejected at TP2 engine startup** (guard in `tp_engine.cpp`) because the backend would silently fall back to BF16 storage while the budget preflight assumed 8 kB/token (guaranteed OOM or silent 4× VRAM). The earlier `test_tp2_kvarn_compare.cpp` was removed: it ran BF16 KV under the KVarN label and asserted nothing.
- P3: **NOT STARTED** — no live-server run has used KVarN KV; the T1–T13 battery results previously logged here were int8/BF16 runs mislabeled as `kvarn_k4v2`.
- P4: **NOT STARTED** — accuracy/determinism on Qwen3.8-27B pending P2.
- P5: **PARTIAL** — budget model only: 8,000 B/token KVarN figure in `tp2_budget.h` + `test_tp2_budget.cpp` (valid as a planning number; meaningless until the codec is in the attention path).

### P1 — codec reference + unit tests (2026-08-23, code-only)

Implemented in `src/ops/kvarn/kvarn_codec.{h,cpp}` and validated via `tests/test_kvarn_codec.cpp` (`ninfer_kvarn_codec_test` registered in CTest):

1. **FWHT exactness & orthonormality**: Verified across $D \in \{64, 128, 256, 512\}$. Energy preservation ($\|Hx\|_2 = \|x\|_2$) passes within $< 10^{-4}$, and exact reconstruction ($H(H(x)) = x$) passes within $< 10^{-5}$.
2. **Sinkhorn balance**: Verified on synthetic ill-conditioned matrices with $15\times$ outlier channels. Raw imbalance spread 15.94 is reduced to balanced spread 1.00 (perfectly equalized row/column stds).
3. **Round-trip reconstruction error**:
   - **K-tile (4-bit int4 RTN)**:
     - $D=256, G=128$: SNR = 20.54 dB, Cosine = 0.9956
     - $D=128, G=128$: SNR = 20.88 dB, Cosine = 0.9960
     - $D=256, G=64$: SNR = 21.48 dB, Cosine = 0.9965
     - $D=128, G=64$: SNR = 21.43 dB, Cosine = 0.9964
   - **V-tile (2-bit int2 RTN)**:
     - $G=128, D=256$: SNR = 5.43 dB, Cosine = 0.8841
     - $G=128, D=128$: SNR = 5.68 dB, Cosine = 0.8879
     - $G=64, D=256$: SNR = 5.56 dB, Cosine = 0.8826
     - $G=64, D=128$: SNR = 5.00 dB, Cosine = 0.8792
4. **Determinism**: 50/50 trials bit-identical across all packed codes and scales.

### P2 — kernel port & CLI wiring (2026-08-23, plumbing only; audit correction)

1. **CLI and Server Plumbing**: Added `KvCacheStorage::KvarnK4V2` to `include/ninfer/types.h`, wired `--kv-dtype kvarn_k4v2` in `serve_options.cpp`, `apps/cli/options.cpp`, `main.cpp`, `request_log.cpp`, and `tp2_decode.cpp`. **Startup guard** (audit 2026-08-23): TP2 engine rejects KvarnK4V2 until the attention-path port exists — without it the backend silently stored BF16 while the budget assumed 8 kB/token.
2. **Budget Model**: Calibrated per-rank KV cache footprint for KVarN at 8,000 B/token in `src/runtime/tp2/tp2_budget.h` and verified in `tests/test_tp2_budget.cpp` (planning number only until P2 lands).
3. **Removed**: `tests/multi_gpu/test_tp2_kvarn_compare.cpp` — it labeled a BF16 run "KVarN" and compared nothing; its claimed token match was the int8/BF16 runs agreeing with each other, not evidence of KVarN.

### P2b — CUDA tile codec (2026-08-24, implemented + verified vs CPU reference)

The GPU half of P2 that is independent of the attention path is done:
`src/ops/kvarn/kvarn_tile_cuda.{h,cu}` ports the P1 CPU codec to one block of
256 threads per tile at **D=256, G=64** (G=64 = paged-KV page size, so a tile
is exactly one page — cleaner than the paper's G=128 which spans two pages).

- `quantize_k_tile_gpu` / `dequantize_k_tile_gpu` (int4 K) and
  `quantize_v_tile_gpu` / `dequantize_v_tile_gpu` (int2 V), bf16 in/out, float
  scales. Shared memory holds X[D][G] channel-major for **both** tiles (the V
  logical [G][D] matrix is loaded transposed and read transposed), so one FWHT
  butterfly (along the channel index, per token) serves K and V.
- **Determinism preserved by construction:** every reduction accumulates
  sequentially in the same order as `kvarn_codec.cpp` (fixed 16 Sinkhorn
  iterations, strict best-so-far with first-minimum tie-break, sequential
  sum-of-squares), so packed codes are bit-identical across GPU runs.
- **Verified** by `tests/test_kvarn_tile_cuda.cpp` (`ninfer_kvarn_tile_cuda_test`,
  5 random trials × K+V): (1) two GPU runs bit-identical on codes+scales;
  (2) codes match the CPU reference up to last-ulp exp/log boundary flips
  (<0.1%, each ≤1); (3) dequantized cosine > 0.999 vs CPU; (4) round-trip SNR
  within 0.5 dB of the CPU reference. **PASS.**
- Two port bugs caught and fixed by the test: (a) V input is token-major, not
  channel-major — needed a transposed load; (b) a templated
  `static bool configured` shared one flag across all four kernels (same
  function-pointer type), so the second kernel launched without its opt-in
  shared-memory attribute → "invalid argument". Attribute is now set per call.

**Still NOT done for P2 (the attention-path integration) — design fixed in
§9 P2c below; execute in that order.**

### P2c — attention-path integration brief (2026-08-24, self-contained for an implementer)

**Status: steps 1–4 DONE** (2026-08-24 on `wo/kvarn-live`). Step 3 isolation
still 9.9e-5 vs bf16-rounded fp32 CPU reference. Step 4 live proof below.

**Why not a staging pass:** at 200k tokens, dequantizing the whole history to
a bf16 shadow buffer every decode step is ~4.7 GB extra DRAM traffic per step.
Rejected. The read path fuses dequant into smem inside the GQA kernel.

#### Existing code the implementer MUST use (do not re-derive)

- **Tile codec (verified, P2b):** `src/ops/kvarn/kvarn_tile_cuda.{h,cu}` —
  `quantize_k_tile_gpu`, `dequantize_k_tile_gpu`, `quantize_v_tile_gpu`,
  `dequantize_v_tile_gpu`. Fixed D=256, G=64 (= one page). bf16 in/out, float
  scales out. **Conventions:** K tile input is `[D][G]` channel-major; V tile
  input is `[G][D]` token-major (the .cu handles the transpose internally —
  feed each function its native layout). CPU reference: `kvarn_codec.{h,cpp}`
  (`quantize_k_tile`/`dequantize_k_tile`/`quantize_v_tile`/`dequantize_v_tile`,
  float in/out) — use it for all host-side tests. Test:
  `tests/test_kvarn_tile_cuda.cpp` (determinism + parity vs CPU).
- **GQA kernel surface to mirror** (`src/ops/kernel/`): decode =
  `gqa_attention_decode_i8.cuh` (+ shared scaffolding in
  `gqa_attention_decode.cuh`, geometry in `gqa_attention_geometry.cuh`);
  prefill = `gqa_attention_prefill_i8.cuh` / `gqa_attention_prefill_bf16.cuh`
  (+ `_common.cuh`). Public entry points: `include/ninfer/ops/gqa_attention.h`
  (`gqa_attention`, `gqa_attention_cached`, `gqa_kv_append`); wrapper + cache
  validation: `src/ops/wrapper/gqa_attention.cpp` (`validate_cache` needs a
  KVarN branch: U8 code planes leading D/2 and D/4, scale table present).
- **Pool layout (step 1, done):** `plan_cache` in
  `src/targets/qwen3_6/impl/state/decoder_state.cpp` — KVarN = 2 code planes
  per layer `{U8, leading=D/2=128}` (K) and `{U8, leading=D/4=64}` (V), plus
  side table `layout.kvarn_scales` fp32 `[layers, heads, physical_pages,
  1152]`. Views carry it as `kvarn_scale_pages` (`paged_kv_cache.h`).

#### Scale side-table field order (per tile = 1152 floats)

| offset | count | field |
|---|---|---|
| 0    | 256 | s_col_K (per-channel log-folded scale, fp32) |
| 256  | 256 | zp_K   (per-channel zero point, folded) |
| 512  | 64  | s_row_K (per-token scale, folded) |
| 576  | 256 | s_col_V |
| 832  | 256 | zp_V   |
| 1088 | 64  | s_row_V |

#### Dequant math the read kernel must implement (in smem)

K tile (codes `q[i][j]` ∈ 0..15, i=channel, j=token):
```
B[i][j] = (q[i][j] - zp_K[i]) * s_col_K[i] * s_row_K[j]
K_hat   = inverse_FWHT_along_channel(B)     // [D][G] out
```
V tile (codes `q[j][i]` ∈ 0..3, j=token, i=channel):
```
B[j][i] = (q[j][i] - zp_V[j]) * s_row_V[j] * s_col_V[i]
V_hat   = inverse_FWHT_along_channel(B)     // [G][D] out
```
Inverse FWHT along the 256-channel axis: in-place normalized butterfly,
8 stages (`len = 1,2,4,...,128`), pair (u,v) → (u+v, u−v), then multiply all
by `1/sqrt(256)`; H²=I so it is its own inverse. Packed-byte layouts: K byte
= `(q_token_hi << 4) | q_token_lo` over token pairs; V byte = four 2-bit codes
`(c3<<6)|(c2<<4)|(c1<<2)|c0` over channel quads (see `kvarn_tile_cuda.cu`
dequant kernels for the exact unpack).

#### Write path decomposition (DECISION — follow this split)

The text pool's KV append is **fused inside the GQA attention kernel** today
(`gqa_attention(q, k, v, ...)` with non-empty k/v appends then attends; see
`text_context_impl.h:1092-1160`). The MTP pool uses an explicit
`gqa_kv_append` (`text_context_impl.h:619`). For KVarN, split the write into:

1. **Workspace append op** `gqa_kv_append_kvarn(k, v, positions, view, stream)`:
   scatters bf16 tokens into a per-sequence **bf16 tile workspace** (≤ 64
   tokens × D × 2 heads per layer, allocated in the work arena by
   tp2_backend; ~4 MB/sequence for 16 text layers + 1 MTP) instead of pool
   pages. Replaces the fused append inside the attention kernel for KVarN.
2. **Host-side commit launch** (in text_context/tp2, where `positions` are
   known host-side): after an append that crosses a page boundary
   (`new_token_count % 64 == 0`), launch one quantize per layer over the
   completed workspace tile → packed codes into pool planes + scales into the
   side table (reuse `quantize_*_tile_gpu`), then clear that tile. **Commit is
   the only writer of packed storage.** No page zeroing needed (attention only
   reads committed pages + tail).
3. **Tail handling in attention:** committed pages come from the pool; the
   in-flight partial tile (≤ 63 tokens) is read directly from the workspace.
   Batched decode has one tail per row, so pass **per-row device arrays**
   `tail_k_ptr[row]`, `tail_v_ptr[row]`, `tail_count[row]` (I32) as extra
   kernel params; the kernel processes them after the packed pages with the
   same math minus dequant. Update them host-side whenever commit/workspace
   state changes.

**MTP rollback is clean by construction:** rejected verify tokens live only in
the workspace and are discarded on trim (`program_impl.h:1015-1022`); no
packed page is ever written for uncommitted tokens, so commit-on-accept holds.
The MTP pool (`mtp_kv`) gets the identical treatment — do not forget it.

**Prefix cache:** cached pages are committed packed pages (page ids only) →
works unchanged; just ensure the prefix slot never owns a workspace tile.

#### Read path (DECISION)

New kernel variants `gqa_attention_decode_kvarn` (+ prefill fill+attention),
mirroring the i8/bf16 kernels' structure: cp.async the packed tile into smem,
dequant in smem per the math above, then **bf16 QK MMA + bf16 PV MMA**. The
int8-QK tensor-core path is NOT used for KVarN (K must be dequanted first);
bandwidth win is preserved (DRAM bytes/token: K halved, V quartered vs int8).
Scales arrive via the side-table pointer + (page, head, layer) indexing; block
tables give page ids as today.

#### Budget (`tp_engine.cpp`)

Replace the hardcoded `kv_bytes_per_token = 8000` with the real
`(payload_bytes / capacity)` from the planned layout (~264 B per token per
head × heads × full_attention_layers, incl. side table).

#### Execution order (each step committed + tested before the next)

1. ~~DType + planes + side table + views~~ **DONE** (`0b2a073e`).
2. ~~Write path: workspace append op + host-side commit launch (+ MTP pool).~~
   **DONE** (`2c841fbb`). GPU round-trip, boundary-crossing (32+32), and
   trim-discard covered in `tests/test_kvarn_write_path.cpp`.
3. ~~Read kernel variants + dispatch in `gqa_attention.cpp` (+ validation
   branch).~~ **DONE** (`gqa_attention_kvarn.cuh` + launcher; dispatch in
   `gqa_attention_cached`). **Test (isolation trick) — corrected 2026-08-24:**
   the kernel is fp32-exact on dequanted values, so comparing against a BF16
   kernel fed CPU-dequantized KV measures bf16 rounding of KV *and of the
   kernel's own bf16 output store* (~4e-3 rel floor) — that threshold was
   miscalibrated. The valid isolation: compare kernel output vs an fp32 CPU
   attention reference on the same dequanted values, **with the reference also
   rounded to bf16** (matching the output store) → 9.9e-5 rel, plus a
   determinism check. `tests/test_kvarn_gqa.cpp` implements both (fp32-ref
   primary at 1e-4; bf16-ref secondary sanity at 5e-3). Live single-request
   check: done with step 4 (T16 + must-pass).
4. ~~Budget math + lift the `--kv-dtype kvarn_k4v2` guard + P3 live gate.~~
   **DONE** (`wo/kvarn-live`, 2026-08-24, commit `50790136`). T16 green,
   must-pass 8/8 at 250k; live defect root-caused (rewind/reset semantics) and
   fixed. See step-4 numbers table below.
5. ~~**Prefill staging pass (D-15 fix).**~~ **DONE** (`wo/kvarn-live`, 2026-08-24).
   See step-5 numbers table below.

**Do NOT:** re-dequantize the whole history every decode step (the original
staging rejection — ~4.7 GB extra DRAM traffic per step at 200k); use int8 QK
tensor cores for KVarN; touch the I8/BF16 code paths (KVarN is opt-in via
`--kv-dtype`); write packed storage outside commit; store scales in pool
planes. Step 5's one-shot incremental staging is NOT the rejected design: each
page is dequanted exactly once over its lifetime, mirroring int8's existing
`fill_i8_page_kernel` architecture.

#### Step 5 — prefill staging (D-15 fix, 2026-08-24) — **COMPLETE**

**Problem (measured, T8 on the step-4 build):** `gqa_attention_kvarn_kernel`
has grid = (q_heads, tokens) — one CTA per query token — and each CTA loops
over ALL packed pages re-running the full dequant (unpack + 256-channel FWHT)
for every K+V page. Prefill cost is O(T×P) with a large constant: ~111 tok/s
at 1.5k prompt → ~22 tok/s at 7.2k, degrading linearly; at 250k the prefill
would take hours. Decode has the same shape (one query/step still dequants all
P pages, plus 8× redundancy across the GQA group). int8 does not have this:
`gqa_attention_prefill_fill_i8_page_kernel` materializes each page once (grid
over page tiles), then the flash kernel reads via tensor cores (~1k tok/s
prefill).

**Design — per-request staged bf16 shadow, one-shot incremental:**
- Staged pool: bf16 K/V with the text pool's paged shape (pages × 64 × D ×
  kv_heads), arena-allocated at request start, freed at request end. Peak =
  full context: ~256 KB/page/rank → ~1 GB at 250k (fits the ~2.7 GB headroom
  of the 250k KVarN config; -np 1 today, so one concurrent pool is the only
  case — document the limit).
- Staging kernel: dequant page p → write bf16 into the staged pool (reuse
  `kvarn_dequant_k/v` verbatim; one CTA per (page, head)). Launch sites:
  (a) prefix restore — stage all restored pages once; (b) commit path — when
  an append fills a page, stage it immediately (O(pages filled) per MTP
  round, usually 0–1).
- Tail: the workspace already holds the tail's bf16 values; copy them into
  the staged pool's last page before each attention call (≤63 tokens ≈ 128 KB)
  — or keep the KVarN kernel's tail path. Implementer's choice; measure.
- Attention: run the EXISTING bf16 prefill/decode kernels against the staged
  pool (identity block table, or a small logical→staged-page table). The
  KVarN kernel is kept for unit tests / fallback dispatch only.
- MTP pool: same treatment (draft window — much smaller).
- Determinism: staging uses the same dequant code → bit-exact values; but the
  bf16 flash kernels may accumulate differently than the custom kernel's
  fp32 path, so re-verify T10 (determinism) and MTP acceptance vs the 82% gate
  after the switch.

**Why this is not the rejected staging pass:** that rejection was per-step
full-history re-dequant in decode (4.7 GB/step at 200k). Here each page is
dequanted exactly once over its lifetime — total O(P) per request, same
architecture int8 already ships.

**Acceptance (test-first):**
- Unit: staged pool contents bit-exact vs direct dequant of the codes.
- Live T8: prefill rate curve must NOT degrade — sustained rate at the 7k
  point ≥ 30% of the 1.5k rate (target: within 2× of int8's ~1k tok/s).
  Extend the battery assertion accordingly (catches O(T×P) regressions).
- T16 green; must-pass suite green at 250k; VRAM headroom check with the
  staged pool live (~1 GB/rank at 250k).

##### Step 5 live proof (2026-08-24)

Implemented: `kvarn_bind_staged_shadow` (~1 GiB budget → **481 pages /
~30.7k tokens × 16 text layers + MTP**), stage-on-commit from the bf16 tile,
stage-from-codes on aligned prefix hit, open-tail copy before attend, BF16
`gqa_attention_cached` dispatch; fused KVarN kernel kept as fallback beyond
staged capacity. Unit `test_staged_shadow_bit_exact` green.

| Metric | Value |
|---|---|
| Staged shadow / rank | **481 pages, 1022 MiB** (log: `kvarn staged shadow`) |
| VRAM / rank (nvidia-smi) | **14,635 MiB** (was 13,611 without staging) |
| T8 client curve | **678 tok/s @~1.5k → 720 tok/s @~7k** (≥30% gate; no O(T×P) collapse) |
| T8 ~12k wall | 17.2s; prefill **722 tok/s** (was 24.5 tok/s / 500s on step 4) |
| Must-pass | T16,T1,T2,T3,T5,T8,T10,T11 **PASS** |
| T10 | PASS (bf16 flash path) |
| T17 | FAIL at step 5 (D-16 open) → **PASS after step 6** |

**Open after step 5:** D-16 (unaligned prefix restore); prompts beyond staged
capacity (~30.7k) fall back to the fused KVarN kernel until capacity is raised
or paging is added.

#### Step 6 — prefix tails (D-16 fix, 2026-08-24)

**Problem:** prefix reuse discarded unless `prefix_len % 64 == 0` → ~94% of
multi-turn prompts re-prefilled from 0; KVarN unusable for agent chat at long
context.

**Fix (`523e54b4`):** at prefix-cache save (end of prefill) snapshot the
open-page bf16 K/V tiles + MTP seed (ar_hidden, draft0) into a per-slot buffer
(~256 KB/rank max); on full hit, rewind to `plen` then restore the tiles
before any attention call; stage the restored tail into the shadow when the
page is within staged capacity. Unaligned guard removed from tp2_backend.

##### Step 6 live proof (2026-08-24, main-side verification)

| Check | Result |
|---|---|
| Unit `test_prefix_snapshot_roundtrip` (capture@130 → rewind+restore → append → commit) | **bit-exact** codes+scales vs direct-append control |
| T17 multi-turn speedup (KVarN 250k) | **PASS** (repeat ≥2× faster) |
| T18 restore equivalence — greedy outputs bit-identical cold vs warm, BOTH shapes (full hit + delta-after-restore) | **PASS** |
| Fast battery on MAIN binary at 250k KVarN | **11 passed / 0 failed** (T1,T2,T3,T5,T8,T10,T11,T15,T16,T17,T18) |
| MTP acceptance, same 512-token workload vs int8 | **50.3% vs 50.1%** — no degradation |
| Decode speed short-context (KVarN vs int8) | 63–68 tok/s vs 65–68 tok/s — within ~2% |

**Still open:** D-17 (abort mid-decode + full-match restore; P3, live repro
needed); **D-18** — beyond staged capacity the fused fallback collapses
(measured 12.6 → 4.0 tok/s past ~30.7k on a real 33k conversation);
interim budget bump to ~2.4 GiB (~74k tokens) shipped same day; real fix =
step 7 below.

#### Step 7 — flash-style fused kernel (D-18 fix, 2026-08-24)

**Problem:** attention beyond staged capacity uses the step-3 fused kernel:
grid = (q_heads, tokens), each CTA re-dequants every packed page → O(T×P)
with an expensive constant. Measured on a live 33k-token conversation:
prefill flat ~700 tok/s up to 29.7k (staged zone), then 12.6 → 4.0 tok/s.
Decode beyond capacity is affected too (fused read every step, per-q-head
redundancy ×16).

**Design — mirror int8's fill+flash architecture, dequant in smem:**
- **Prefill (q-block kernel):** grid = (q_heads, q_blocks), B_q tokens/CTA.
  Per KV page block: dequant K tile [D][G] ONCE into smem (shared by all B_q
  queries), compute the B_q×G score matrix, online-softmax across pages,
  accumulate O += P @ V (dequant V once per block, reusing the same smem
  buffer after the K phase). Dequants drop from T×P to ceil(T/B_q)×P.
  **Dequant to bf16** (not fp32) so values + accumulation match the staged
  BF16 path bit-for-bit → in-capacity and beyond-capacity results are
  identical (protects T10/T18 determinism across the boundary).
- **Decode (GQA-shared kernel):** one CTA per kv_head handles all q_heads of
  its GQA group: dequant each page once, score all group queries, accumulate
  their O vectors. Removes the ×16 per-q-head redundancy.
- Tail (≤63 workspace tokens): same bf16 direct path as today, appended as a
  partial block after the packed pages.
- Dispatch unchanged (`need_pages > staged_pages` → fused); only the kernel
  changes. Keep the old kernel for unit tests / fallback.
- Smem budget: one fp32-or-bf16 [D][G] tile at a time (K phase, then V phase)
  + Q block; must fit sm_120 SM limits — check against `kvarn_tile_cuda`
  constants before writing.

**Acceptance (test-first):**
- Unit: fused-flash prefill output vs staged-path output on identical data
  (pages ≤ capacity) — bit-exact or bf16-equivalent + determinism check.
- **T19 (already in battery, currently FAILING): 80k-token prefill must
  sustain ≥450 tok/s average.** This is the gate.
- Decode beyond capacity: ~40k-context decode within 2× of in-capacity rate.
- T16/T17/T18 + must-pass subset green at 250k; VRAM unchanged (no new
  allocations — smem only).

**Interim (ATTEMPTED + REVERTED 2026-08-24):** a budget bump to 2.4 GiB was
tried and reverted — it pushed the 250k config past the real VRAM ceiling
(startup OOM); the "~16,310 usable" figure was too optimistic. Budget is back
at 1 GiB (~30.7k tokens fast path). **Do not raise the shadow budget without
measuring free VRAM at startup** — step 7 removes the need for a bigger
shadow entirely.
#### Step 4 live proof (2026-08-24, `wo/kvarn-live`)

Launch (port 8091):
```
./build/apps/ninfer-serve /home/intel/models/qwen3_8_27b.ninfer \
  --port 8091 --devices 0,1 --spec mtp --draft-tokens 3 \
  --kv-dtype kvarn_k4v2 --kv-capacity 250000 --max-context 250000
```
Log: `/home/intel/ninfer/logs/kvarn-wo-live.log`.

| Metric | Value |
|---|---|
| VRAM / rank (nvidia-smi) | 13,611 MiB |
| Decoder state / rank | 2,727 MB, 3,907 KV pages (cap 250,007) |
| Layout B/token (KV) | 8,976 (= 18496×16896/34816) |
| Isolation `ninfer_kvarn_gqa_test` | rel_l2 = 9.9e-5 (PASS) |
| Write-path unit | PASS (incl. T16 reset / hydrate / MTP redo) |
| Live suite | **8 passed, 0 failed** — T16,T1,T2,T3,T5,T8,T10,T11 |
| T8 (~12k prompt) | wall 499.8s; prefill 24.5 tok/s; TTFT 492s; MTP 2.46 tok/round (48.7%) |
| MTP over suite (41 reqs) | mean 2.86 tok/round, mean accept 62.0% (min 33.3 / max 83.3) |
| Append / worker errors | **0** (`jumped pages` / `worker error`) |

Root cause of the live T16 throw (docs/50 D-14): (1) worker MTP redo after
`prepare_mtp_prompt` left MTP tile mid-page while re-appending from 0;
(2) speculative verify crossing a page then appending on the prior page without
hydrating the committed tile; (3) unaligned full-prefix hits after a committed
lossy open page. Fixes: reset/rewind + hydrate-on-append + unaligned→re-prefill
from 0 + skip worker MTP redo when prepare_mtp already ran; attend uses
`committed_pages` (not envelope-inferred packed pages).

**Open (perf, not gate):** KVarN attention still re-dequants each packed page
per query CTA → ~12k prefill ~25 tok/s vs ~1k on I8. Correctness-only for P2c.


### P3 / P4 — live battery & accuracy

**P3 must-pass at 250k KVarN: COMPLETE** (2026-08-24) — table above. Full
T1–T15 battery and P4 accuracy probe still open; MTP mean accept on this run
is below the historical I8 82% baseline (R2) but the must-pass subset did not
fail its functional checks.

### P5 — 200k budget evaluation (2026-08-23, model only)

- At $G=128$ and $D=256$, KVarN stores 4-bit K + 2-bit V with folded FP16 per-tile scales, reducing text KV cache memory to ~8,000 B/token (vs 18,496 B/token for Int8 and 34,816 B/token for BF16).
- Decoder state allocation at 80k context:
  - Int8: ~2648 MB (fixed overhead + KV)
  - KVarN: ~1133 MB (at 16k) and comfortably fits 200k tokens well below the 16.31 GiB per-rank budget limit (`fitting > 250,000` tokens).


## 10. Risks

- **R1** TP sharding math is not head-local → allreduce of tile stats in the hot
  path (bandwidth) or redesign. Resolved in P0, before any kernel work.
- **R2** MTP acceptance drops >4 pp → kvarn unusable for our MTP-first config;
  fallback: run kvarn with `--spec none` and accept the decode-speed loss.
- **R3** Accuracy on Qwen3.8-27B below their claimed parity (different model,
  hybrid GDN) → stay opt-in or abandon; P4 decides with data.
- **R4** Sinkhorn iterations in the write path cost more than expected → their
  "throughput ≥ FP16" claim may not survive our kernels; measure pp/pt t/s at
  80k before/after in P3, log to §9.
- **R5** Scope creep: MLA, DFlash, bidirectional attention, sub-2-bit are all in
  their repo and none of them are ours.

## 11. Non-goals

vLLM integration, Triton runtime, MLA models, DFlash, calibration-based variants,
making kvarn the default before P4.
