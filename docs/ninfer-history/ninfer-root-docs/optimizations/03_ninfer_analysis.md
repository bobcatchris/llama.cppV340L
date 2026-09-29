# NInfer (https://github.com/Neroued/ninfer) — how it works & V340L applicability

Cloned & audited 2026-08-19. Apache-2.0. ~88k lines C++/CUDA (src+include+apps), ~36k lines in the kernel tree, 50 per-op benchmark files.

## 1. What it is

A **from-scratch C++/CUDA inference engine for one RTX 5090 (sm_120a, Blackwell consumer)** running a **closed set of 5 registered Qwen artifacts** (Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B; groupwise-int and NVFP4 weight profiles). Not a general runtime: the build **rejects any CUDA arch other than 120a**, and only the five `(model_id, weights_id)` identities are accepted. CLI + OpenAI/Anthropic-compatible HTTP server. Text + image/video, MTP speculative decoding (draft 1–5), DFlash (MoE, draft 1–15), paged KV, INT8 group-64 KV cache, CUDA-graph decode, chunked prefill, 1–8 concurrent requests with true batched decode.

Explicit non-goals (README): **no multi-GPU, no CPU/GPU offload, no distributed serving, no continuous batching/preemption.**

## 2. Measured performance (single 5090, 27B NVFP4)

| | MTP0 | MTP3 (C=1) | MTP3 (C=8) |
|---|---|---|---|
| Decode @ 8k ctx | 86.4 t/s | 202.4 t/s (76–81% accept) | 1,146.9 t/s (5.67× scaling) |
| Prefill @ 7,680 | 11,191 t/s (W4A4) | — | — |
| Prefill @ 260k | 2,510 t/s | — | — |

groupwise-int profile on the same GPU: prefill 3,218 t/s, decode 77.6 t/s → the **NVFP4 W4A4 prefill is 3.48× the int profile**; MTP3 decode +30–32%.

## 3. How it works (architecture)

### 3.1 Artifact format
One `.ninfer` file = weights **in pre-packed layout** + frontend resources (tokenization, vision config). Custom version-2 container with identity metadata. Not safetensors/GGUF/Transformers. Conversion tools (`tools/convert/`) produce the packed layout offline — the same "prepack at load, stream in the kernel" philosophy as v100-skinny's QPN prepack.

### 3.2 Per-shape compile-time kernel specialization (the core trick)
Every GEMM shape in the model gets a **compile-time-exact `Geometry` (N, K as template constants) + a tuned `Schedule`**:

```
Nvfp4GemvSchedule<WarpsPerCta, RowsPerWarp, ValuesPerLane(8|16|32),
                  AccumulatorChains, ScaleAccess(StagedRaw|Direct),
                  CodeCache(Default|Streaming), MinBlocksPerSm>
```

Dispatch is a switch over ~5 problem classes (`AttnInput`, `GdnInput`, `MlpGateUp`, `Residual6144`, `Residual17408`). Because the model set is closed, every shape is known at compile time → full unrolling, no runtime shape branching, per-shape occupancy tuning. **This is architecture-agnostic and is the most transferable asset in the repo.**

### 3.3 Decode GEMV (A16 W4)
`nvfp4_gemv.cu`: bf16 activations, NVFP4 weights (e2m1 codes + e4m3 group-16 scales), **fp32 FMA with multiple accumulator chains** (ILP to hide HBM latency), e2m1x2 decode + scale→coefficient in registers, scales staged to smem (quartet interleave for coalesced reads) or direct, warp-reduce, one store per row. Weights arrive in the offline-interleaved layout (128-row tiles × rmod32 × quartile) so each lane's 8/16/32-code load is a single coalesced vector (`ld.global.cg.v4.u32`). Bandwidth-bound by design — targets the 5090's 1.8 TB/s.

### 3.4 Small-T kernels (M=2..8)
`nvfp4_small_t.cu`: token-tiled GEMV variant for **MTP verification and small batches** — warps split rows *and* tokens, partials in smem, finalization (elementwise or row-vector). The gfx900 analogue of v100-skinny's QPN/WMMA M-band, but SIMT.

### 3.5 Prefill W4A4 (native FP4 tensor cores) — the Blackwell-specific crown
`nvfp4_w4a4_mma.cuh` issues:
```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```
— **Blackwell-only block-scaled FP4 MMA** (activations quantized to e2m1 on the fly, `Nvfp4W4a4MaterializedActivation`). Plus a TMA variant (`nvfp4_w4a4_tma.cu`) using Tensor Memory Accelerator + mbarrier async smem staging, `setmaxnreg` (Hopper+ register reallocation), `cp.async`, `ldmatrix`. This is where the 3.48× prefill number comes from. **Nothing of this exists on gfx900.**

### 3.6 GDN (gated delta net) — the 48 linear-attention layers
Custom recurrent decode kernels (fp32 and bf16-direct variants, with **replay records** so the state read/write pointers can be swapped per graph replay), plus a chunked-prefill implementation. This is Qwen3.6-specific and non-trivial; it's the part of any port that has no off-the-shelf substitute on gfx900 (llama.cpp has GDN support for Qwen3-Next, but not this tuned).

### 3.7 Decode orchestration
- `DecodeGraphDefinition/Executable`: capture → instantiate → upload → launch (standard CUDA graph pattern; `hipGraph*` equivalents exist in ROCm).
- MTP: draft window 1–5, **`--lm-head-draft`** (the MTP module drafts through the *target's* lm_head — shared-head chain-MTP, same as v100-skinny's reference stack), rejection sampling, per-round request compaction.
- Paged KV (471-line `paged_kv_cache.cpp`) + cyclic variant; INT8 group-64 KV halves KV bandwidth.
- Chunked prefill (1,024-token chunks); concurrent serving: startup-fixed capacity 1–8, one batched model traversal per round for all decode-ready requests.
- Host worker pool for offloading host-side work; NVTX ranges throughout.

### 3.8 Verification infrastructure
50 per-op benchmarks (`bench/ops/*_bench.cu`), per-target round benchmarks, an **HBM bandwidth probe** (`tools/hbm_bandwidth_probe.cu`), and a **Python reference implementation** (`tools/reference/`) for numerical parity checks. Same discipline as v100-skinny's benchmarks/ + results/ + byte-diff rule.

## 4. Portability to V340L (gfx900)

### 4.1 Directly portable (architecture-agnostic)

| Asset | Notes for gfx900 |
|---|---|
| Engine skeleton (resident model, paged KV, chunked prefill, graph decode, MTP orchestration, batched decode, 1–8 concurrency) | HIP-ify: `cuda*`→`hip*`, `cudaGraph*`→`hipGraph*` (exists in ROCm). Largest single chunk of reusable C++. |
| **Per-shape compile-time specialization** | Ports 1:1; arguably *more* effective on gfx900 (closed model set, fixed shapes, one arch to tune). |
| Weight prepack philosophy | Concept 1:1; the interleaved layout must be **re-derived** for gfx900 (32 B sectors, 4 MB L2, 64 KB LDS, 64-wide wavefronts, different L1 behavior). |
| GEMV kernel structure (values/lane, accumulator chains, scale staging, reduce) | Ports to HIP; 64-wide wavefront → 2× values per lane or 2× lanes per row-group; retune chains/occupancy. e2m1/e4m3 decode math is pure ALU, verbatim. |
| Small-T (M=2..8) design for MTP verify | Ports; this is the gfx900 substitute for the missing tensor-core M-band (see M-wall in 01 §3). |
| INT8 group-64 KV cache | Storage + dequant in attention kernel; portable. |
| MTP + lm-head-draft + rejection sampling | Portable; keep k=2–4 (M-wall). |
| GDN recurrent decode (structure) | The math ports; the tuned kernel needs gfx900 re-tuning. No off-the-shelf gfx900 GDN kernel of this quality. |
| Bench/parity harness (per-op bench + Python reference + HBM probe) | **Adopt wholesale** — it's the difference between tuning in the dark and tuning against a roofline. |
| License | Apache-2.0 → legally reusable, including in a derived V340L engine. |

### 4.2 Not portable (hardware-specific)

| Asset | Why | gfx900 substitute |
|---|---|---|
| **W4A4 native FP4 MMA prefill** (3.48× prefill win) | `kind::mxf4nvf4...e2m1.e2m1...ue4m3` is Blackwell-only | A16 W4 dequant GEMM (v100-skinny-style, LDS-tiled, large M) → ~450–650 t/s prefill (see 02 §3) vs 11,191. The gap is hardware, accept it. |
| TMA + mbarrier + cp.async async staging | Hopper/Blackwell | Plain global→LDS loads; software pipeline (v100-skinny's WMMA pipelining pattern works on GCN5). |
| `setmaxnreg` (warp-specialized register realloc) | Hopper+ | Not available; tune occupancy statically. |
| bf16 activations | No native bf16 on GCN5 | fp16 (v100-skinny's choice; 2:1 rate on gfx900). |
| sm_120a hardcode + PTX asm tree | CMake rejects other archs; ~36k lines of CUDA/PTX | Full HIP rewrite of the kernel tree (structure reusable, code not). |
| **Single-GPU assumption** | No multi-device path anywhere (no NCCL/HIP-RT, no TP) | **The biggest net-new engineering item** for V340L: a 2-die (within-card) or 4-die (cross-card) layer — TP or layer split + custom one-shot AR over PCIe. Nothing in ninfer to copy; design from the v100-skinny findings (01 §5). |
| 1.8 TB/s GDDR7 roofline | 483.8 GB/s per die | Re-measure per card (units vary). |

### 4.3 Verdict: what's usable

**Yes — substantially.** Ninfer is the best available open template for the *engine* half of a V340L project: closed-model specialization, per-shape kernels, prepack, paged KV, MTP orchestration, graph decode, and a serious bench/parity harness. v100-skinny is the best template for the *kernel philosophy* half on pre-modern silicon (dequant A16 SIMT, fused head, one-shot AR, validation discipline). The V340L project is essentially:

```
ninfer engine skeleton (HIP-ified)
  + v100-skinny-style A16 W4 dequant GEMV/GEMM kernels (gfx900-tuned)
  + net-new multi-die layer (2-way within-card TP first, one-shot AR over the bridge)
  + MTP k=2–4, INT8 KV, paged cache
  + both projects' validation discipline (byte-diff, roofline-first, per-op bench)
```

Effort reality: the kernel tree is a rewrite, not a port (36k lines CUDA/PTX → HIP/GCN5); the multi-die layer is net-new; the GDN kernels need re-tuning. Realistic timeline is months, not weeks — but the two repos together remove most of the design uncertainty, and the bench harness means every step is measured against a roofline from day one.

### 4.4 Quick wins without the full port

1. **Copy the bench methodology now**: per-op GEMV microbench + HBM probe + Python reference parity, run on the actual V340L dies. This alone validates/invalidates assumptions 1–2 in 02 §5 before any engine work.
2. **Copy the per-shape specialization pattern into llama.cpp's ROCm backend** (or a small custom op library): even 2–3 hand-tuned exact-shape GEMVs for the 27B's dominant shapes (5120×5120, 5120×17408 gate/up, 248320×5120 head) would capture most of the decode gap.
3. **Copy the prepack layout idea**: offline-interleave the Q4_1 weights for coalesced gfx900 GEMV loads — a conversion-time change, zero runtime complexity.
4. **Keep the 5090 numbers as the ceiling reference**, not the target: 86.4 t/s MTP0 decode is what *one* 5090 does; the tuned 2×V340L target is ~50 t/s MTP0 / ~110–165 t/s MTP3-structured (02 §4).
