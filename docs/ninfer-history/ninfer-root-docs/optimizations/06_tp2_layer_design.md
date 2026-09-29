# 06 · TP2 multi-GPU layer — design (NInfer, 2× RTX 5060 Ti)

The one piece of net-new engineering. Device-agnostic C++ (CUDA today, HIP later for the V340L). This doc is the implementation guide.

## 1. The key insight: the kernels don't change

NInfer's public linear op is already **single-device and stream-parameterized**:

```cpp
// src/ops/linear/linear.cpp:157
void linear(const Tensor& x, const Weight& w, Tensor& out,
            LinearPolicy policy, void* workspace, cudaStream_t stream);
```

The kernel launches on `stream`, on the current device, reading `w`'s packed pointers. So TP2 is a **weight-sharding + collective overlay**: shard each `Weight` per GPU, call the *same* `linear()` on each device's stream, and insert an allreduce where the math requires it. **Zero kernel-tree changes.** The same overlay ports to HIP for the V340L by swapping `cudaStream_t`→`hipStream_t` and the AR transport.

The decode program's call sites (`src/targets/qwen3_6/impl/runtime/*.h`) use:
`ops::linear`, `ops::linear_add`, `ops::linear_swiglu`, `ops::linear_pair`, all on `state.execution.device.stream`. These are the integration points.

## 2. Architecture: a `TpGroup` abstraction

```cpp
// src/multi/tp_group.h  (new)
struct TpGroup {
    int size;                          // 2 for TP2
    std::vector<DeviceContext> devs;   // one per GPU (stream, load_stream, props)
    AllReduce ar;                      // one-shot + batching state

    void for_each(std::function<void(int rank, cudaStream_t)> fn);
    Tensor shard(const Weight& w, ShardMode m, int rank) const; // per-rank Weight view
    void allreduce(Tensor& buf);       // in-place sum across ranks
    void allreduce_batched(Tensor& buf, int flush); // accumulate, reduce every `flush`
};
```

`state.execution.device` (single `DeviceContext`) becomes `state.execution.tp` (a `TpGroup`). Op call sites change from:
```cpp
ops::linear(x, w, out, policy, ws, state.execution.device.stream);
```
to a TP-aware wrapper:
```cpp
tp_linear(tp, x, w, out, policy, ws, /*shard=*/ShardMode::Column);
```
where `tp_linear` shards `w`, loops ranks calling `ops::linear` on each rank's stream, and (for `Row`) issues `tp.allreduce(out)`.

## 3. Weight sharding (Megatron-style)

Per linear layer, one of three modes:

| Mode | Split | Per-rank `Weight` | Allreduce? | Used by |
|---|---|---|---|---|
| **Column** | output dim `n` → `n/2` | `w.n/2`, qdata/scales offset to rank's row block | **no** (out is split; next op is per-head or row-parallel) | qkv_proj, gate/up_proj |
| **Row** | input dim `k` → `k/2` | `w.k/2`, qdata/scales offset to rank's col block | **yes** (sum partials) | o_proj, down_proj |
| **Replicated** | none | full `w` on both | no (or reduce-scatter) | lm_head (see §7), embeddings |

Per transformer layer the pattern is `Column → (per-head attn / swiglu) → Row(+AR)`, giving **2 ARs/layer ≈ 128 ARs/token** for 64 layers.

**Sharding the packed layout.** `Weight` pointers reference a prepacked layout (`QuantLayout::BlockScaleK128x4` for NVFP4 — 128-row × 4 tiles). Two options:
- **(a) Shard at conversion time (recommended):** the converter emits **per-rank packed shards**, each independently packed + coalesced. Clean, matches NInfer's prepack philosophy, no cross-boundary tiles. Cost: converter change + 2× artifact metadata.
- **(b) Offset into the full buffer:** only correct if the shard boundary aligns with the tile grid (n a multiple of 128). Fragile for `k`-splits.

Go with **(a)**. The `shard()` helper then just returns a `Weight` whose `n`/`k` and pointers point at rank r's shard arena.

## 4. Allreduce (the critical kernel)

2 GPUs over **PCIe 5.0 ×8**, no NVLink/P2P. Message = hidden vector, 5120 × bf16 = **10 KB** (latency-bound, not bandwidth-bound).

- **One-shot AR:** each rank `cudaMemcpyPeer` its partial to the other (works over PCIe without true P2P, via host), each adds the received value in-place. ~2 copies + 1 small kernel per AR.
- **AR batching (the main win):** accumulate `B` layers' partials in a fp32 accumulator, reduce once every `B` layers. Cuts AR count 128 → 128/B. **B=4 → 32 ARs/token.** Validate numerics (fp32 accumulation over 4 layers is safe; byte-diff vs B=1).
- **Target cost:** ~20–40 µs/AR over PCIe 5.0 ×8 → B=4 gives ~1–1.5 ms/token (vs ~4–7.5 ms unbatched).

```cpp
// src/multi/allreduce.h/.cu  (new)
struct AllReduce {
    void init(TpGroup&);                 // set up peer copies + fp32 accum buffers
    void accumulate(const Tensor& partial); // add into fp32 accum (no sync)
    void flush(cudaStream_t* streams);   // one-shot reduce of the accum, reset
};
```

## 5. KV cache sharding (attention)

`PagedKVLayerView` holds `k_pages/v_pages/k_scale_pages/v_scale_pages/block_table`. For GQA, **split KV heads across ranks**: rank r owns heads `[r*H_kv/2, (r+1)*H_kv/2)`. Each rank's `k_pages/v_pages` are sized for its head slice; `block_table` is shared (replicated). The attention op runs per-rank on its heads. No AR for attention (heads are independent); the AR comes from the following `o_proj` (Row).

## 6. GDN (linear-attention) state sharding

The 48 GDN layers carry a fixed-size recurrent state (per value-head). **Split value-heads across ranks** exactly like KV. Each rank's `ssm_state` is its head slice; the GDN recurrent op runs per-rank. The GDN `in_proj` is Column (split), `out_proj` is Row (+AR).

## 7. lm_head (248,320 × 5120)

Two choices:
- **Column-split** the head (each rank computes half the vocab logits) + all-gather the logits for sampling. Extra gather per token.
- **Replicate** the head on both ranks (each computes full logits, discard one). Wastes a GEMV but avoids the gather; the head is ~0.7 GB, fits.
Recommend **column-split + gather** (bandwidth-optimal); the fused-argmax (v100-skinny) can fold the gather+argmax into the head epilogue later.

## 8. Graph capture across 2 streams

NInfer's `DecodeGraphDefinition/Executable` captures one stream. Extend to capture **both ranks' streams** in one graph (CUDA graphs support multi-stream capture via `cudaStreamBeginCapture` with a capturing stream group, or two sub-graphs launched together). The AR peer-copies + reduce kernels are captured as graph nodes. **Hazard (from v100-skinny):** the drafter's GDN recurrent state + KV write-slot indices must be in the graph's owned/refreshed set — prove parity with a state-norm probe after N rounds before trusting it.

## 9. Exact integration points (files to touch)

| File | Change |
|---|---|
| `src/core/device.h` | add `TpGroup` (or new `src/multi/tp_group.h`) |
| `src/multi/allreduce.{h,cu}` | **new** — one-shot AR + batching |
| `src/multi/tp_linear.{h,cpp}` | **new** — TP-aware `linear/linear_add/linear_swiglu/linear_pair` wrappers |
| `src/artifact/*` (loader) | load per-rank weight shards (§3a) |
| `tools/convert/qwen3_8_27b/*` | emit per-rank packed shards |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | swap `ops::linear*` → `tp_linear*`; `device.stream` → `tp` |
| `src/targets/qwen3_6/impl/runtime/dflash_impl.h`, `text_prefill_impl.h`, `vision_context_impl.h` | same swap (prefill + MoE + vision) |
| `src/core/paged_kv_cache.*` | head-sharded page pools (§5) |
| GDN state layout (`layouts_impl.h`) | head-sharded ssm_state (§6) |
| `src/core/decode_graph.*` | multi-stream capture (§8) |
| `apps/` (CLI/serve) | `--tp 2` flag, device selection |

## 10. Validation strategy (adopt NInfer's + v100-skinny's discipline)

1. **Per-op byte-diff:** for each sharded `linear`, compare TP2 output vs single-GPU reference (on a shape that fits one GPU) — must match to fp tolerance.
2. **AR unit test:** known partials → exact sum.
3. **Whole-round byte-diff:** TP2 greedy decode vs single-GPU (or vs the llama.cpp Q5_K_M server, same prompt+seed) — tokens must match (greedy) or stay in-distribution (sampled).
4. **Graph parity probe:** state-norm after N graphed rounds == eager (catches the drafter-state hazard).
5. **Never quote a spec-decode throughput without a byte-level diff** (v100-skinny's rule).

## 11. Implementation order (milestones)

1. **M1 — TpGroup + AR:** build `TpGroup`, one-shot AR + batching; unit-test AR to exact sum. *(no model needed)*
2. **M2 — tp_linear + weight sharding:** converter emits per-rank shards; `tp_linear` wrappers; per-op byte-diff on small shapes.
3. **M3 — KV + GDN sharding:** head-split page pools + ssm_state; attention/GDN per-rank.
4. **M4 — wire the decode program:** swap call sites; greedy byte-diff vs reference.
5. **M5 — multi-stream graph capture:** capture the round; graph-parity probe.
6. **M6 — MTP + serve:** drafter on TP2, `--tp 2` serve, measure tg/pp.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Packed layout doesn't shard cleanly | shard at conversion time (§3a), not by offset |
| AR over PCIe slower than modeled | measure in M1; tune batch factor B; fuse more layers |
| `cudaMemcpyPeer` without P2P is slow | fall back to host-staged AR; measure both |
| Multi-stream graph capture is finicky | keep an eager path; graph-parity probe before trusting |
| Numerics drift from fp32 AR accumulation | byte-diff B=4 vs B=1; drop B if it drifts |
