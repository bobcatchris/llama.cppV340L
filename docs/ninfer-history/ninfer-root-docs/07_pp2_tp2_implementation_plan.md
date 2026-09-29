# NInfer Multi-GPU (PP2 → TP2) Implementation Plan — 2× RTX 5060 Ti

**Status:** ARCHIVE

## 0. Context & goal
Run Qwen3.6-27B (16.29 GiB `.ninfer`) on **2× RTX 5060 Ti** (16.31 GiB each). The model does
**not fit on one GPU** (16.29 GiB > 16.31 GiB before KV), so we need 2 GPUs. Goal: beat the
current llama.cpp `-ts 1,1` (TP2) + MTP server using NInfer's tuned kernels + MTP.

### Measured hardware (Window #1, 2026-08-19)
- Sustained HBM: **427 GB/s read / 425 write** (95% of 448 GB/s peak). 36 SMs, 32 MiB L2.
- Decode GEMV efficiency: **76–94% of sustained** (~88% main shapes). lm_head T=1 = 2.63 ms.
- GDN layer (fused, graph): T=1 127 µs, T=16 209 µs. Attention (256 ctx) 18 µs. argmax 8 µs.
- **NCCL 2.30.4 installed.** GPU0↔GPU1 = **PHB** (via CPU host bridge, both NUMA0), PCIe 5.0 x8.
- Allreduce of a 5120-dim bf16 vector (~10 KB) ≈ sub-µs over this link → **comm is not the bottleneck.**

## 1. Architecture facts (from code read)
- `DeviceContext` = single device (device id, stream, load_stream, props). Engine holds one.
- `MaterializedArtifact` loads all weights into **one `DeviceArena`** (single device).
- `Weight` = a **view** (offset) into a contiguous quantized payload + scale metadata (Q4G64/Q5G64/
  Q6G64/W8G32/NVFP4/FP8). Sharding = copy per-device shards of payload **and** scales.
- **Ops are stream-based / device-agnostic**: `linear(x,w,out,policy,ws,stream)`, GDN, attention all
  take a `cudaStream_t` and run on that stream's device. → We can dispatch the same op on 2 streams
  with per-device weights/activations. **The kernel tree is reused unchanged.**
- Decode program is a monolith: `program_impl.h` (2232 ln) + `text_context_impl.h` (1338 ln) +
  `layouts_impl.h` (755 ln) + `dflash_impl.h`/`mtp_impl.h` (speculative). All single-device.
- Model: 64 layers = **16 full-attn + 48 GDN**, hidden 5120, MLP intermediate 17408, vocab 248320,
  MTP head. Per-layer allreduce points: after **O-proj** (attn) / **GDN out-proj**, and after **MLP down**.

## 2. Two parallelisms
| | PP2 (pipeline) | TP2 (tensor) |
|---|---|---|
| Split | layers 0–31 → GPU0, 32–63+lm_head → GPU1 | every layer's N/K split across both GPUs |
| Comm | 1 hidden-state transfer/stage (~10 KB, P2P) | 2 allreduces/layer (128 total, ~10 KB each) |
| Weight load | each GPU loads its layer range (no in-layer shard) | each GPU loads half of every layer (quant shard) |
| Decode latency | sum(stages) ≈ full-model time (**no speedup**, just fits) | max(stages)+allreduce ≈ **half** (2× speedup) |
| Est. decode (MTP0) | ~40 ms → ~25 t/s | ~20 ms → ~37–40 t/s |
| Complexity / risk | **low** | **high** |

**Decision: PP2 first** (correct 2-GPU run, byte-diff validated, low risk), **then TP2** (2× speedup).
They share the same foundation (2-device context, per-device materialization, 2-stream decode loop),
so the foundation is built once.

## 3. Foundation (shared by PP2 + TP2)
New module `src/core/multi_gpu/` (does NOT touch the working single-GPU path):
1. **`TpGroup`** — owns N=2 `DeviceContext`s (one per GPU), an NCCL communicator
   (`ncclCommInitRank`), and helpers:
   - `allreduce_bf16(ptr_local, n, rank)` → `ncclAllReduce` (in-place, bf16).
   - `send_hidden(dst_rank)` / `recv_hidden(src_rank)` → `ncclSend`/`ncclRecv` (PP2 stage transfer).
   - `barrier()` for graph-capture safety.
   - Per-rank `DeviceArena` for weights + workspace.
2. **Weight sharding** (`shard_weight`): given a `Weight` view + a split spec
   (ColumnSplitN / RowSplitK / Replicate / None), produce per-rank `Weight` views into per-rank
   device buffers. Handles the quantized payload **and** the group scales:
   - ColumnSplitN (attn QKV, MLP gate/up, lm_head rows): split output rows; scales split per row-block.
   - RowSplitK (attn O, MLP down): split input cols in **groups of `group_size`** (so a quant group is
     never split across ranks); scales split accordingly.
   - Replicate (embeddings, norms, small control): copy to both.
   - **Unit test**: shard + concat == original (bit-exact) for every QType.
3. **Byte-diff harness**: run single-GPU NInfer on a fixed prompt → save per-layer hidden states +
   final logits (fp32). Run PP2/TP2 → compare (max abs diff < 1e-3 for bf16, argmax token identical).

## 4. PP2 (Phase A) — correct 2-GPU run
1. **Materialization**: GPU0 arena = layers[0..31] + embeddings + norms[0..31]; GPU1 arena =
   layers[32..63] + lm_head + norms[32..63]. (Reuse `Reader`/`Binder`; a PP2 `MaterializationPlan`.)
2. **Decode loop**: 
   - GPU0: embed → layers 0–31 → hidden H (5120 bf16).
   - `send_hidden(1)` / `recv_hidden(0)` (10 KB).
   - GPU1: layers 32–63 → final norm → lm_head → logits → argmax.
   - KV cache: GPU0 holds layers 0–31 KV; GPU1 holds 32–63 KV (no split).
3. **MTP**: draft head on GPU1 (with lm_head); verify runs k+1 tokens through the pipeline.
4. **Validate**: byte-diff vs single-GPU (same prompt, greedy). Expect identical tokens.
5. **Serve**: a thin PP2 server wrapper (or extend `ninfer-serve` with `--pp 2`).

## 5. TP2 (Phase B) — 2× decode speedup
1. **Weight sharding** per problem class (see §3.2): QKV/gate-up/lm_head = ColumnSplitN;
   O/down = RowSplitK; embeddings/norms = Replicate; GDN in/out proj like attn/MLP.
2. **Per-layer 2-stream dispatch** (reuse ops):
   - rmsnorm (replicated input) on both streams.
   - QKV (column) on both → each rank has half the heads.
   - attention / GDN scan **head-parallel** (each rank its heads; KV/GDN state sharded by head).
   - O-proj / GDN out-proj (row) on both → **allreduce** → residual.
   - MLP gate/up (column) → swiglu → down (row) → **allreduce** → residual.
   - lm_head (column over vocab) → allreduce (or gather logits) → argmax.
3. **KV/GDN state sharding**: split heads across ranks (16 full-attn heads / 2, GDN heads / 2).
4. **Graph capture**: capture the whole decode round across both streams (NCCL ops are capturable).
5. **MTP**: verify batch (k+1) on both streams; draft head sharded.
6. **Validate**: byte-diff vs single-GPU; then bench (target ~37–40 t/s MTP0, ~90–105 MTP k=3).

## 6. Phased plan + validation gates
- **P0 (foundation)**: `TpGroup` + NCCL allreduce/P2P (unit test: allreduce of a vector on 2 GPUs is
  correct). Weight-sharding module + bit-exact shard/concat test for all QTypes.
- **P1 (PP2)**: PP2 materialization + 2-stage decode + hidden transfer. **Gate: byte-diff identical
  tokens vs single-GPU.** Serve + measure (expect ~25 t/s MTP0).
- **P2 (TP2)**: in-layer sharding + 2-stream dispatch + allreduce + KV/GDN head-shard. **Gate:
  byte-diff identical tokens.** Bench (expect ~37–40 t/s MTP0).
- **P3 (MTP + graphs)**: MTP k=3 on 2 GPUs + CUDA-graph capture. **Gate: acceptance rate matches
  single-GPU; tokens identical in greedy mode.** Bench (expect ~90–105 t/s).
- **P4 (serve + tune)**: `--pp 2` / `--tp 2` flags, KV int8, concurrency, final A/B vs llama.cpp.

## 7. Risks
- Quantized RowSplitK must not split a quant group → split K in multiples of `group_size` (5120 and
  17408 are both divisible by 64 and by 2 → clean).
- NCCL + CUDA-graph capture: NCCL collectives are graph-capturable since 2.x, but verify on 2.30.4.
- GDN head-sharding: the GDN kernel must support a head offset/subset (check `gated_delta_net` API).
- bf16 allreduce rounding: NCCL bf16 allreduce is deterministic; byte-diff tolerance 1e-3.
- Memory: TP2 each rank holds half weights (~8.1 GiB) + half KV → fits 16 GiB with room for 200K ctx.
