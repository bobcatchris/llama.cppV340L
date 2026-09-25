# TP2 Comparative Parity Specification & Test Battery Architecture
**Gate Standard for Team Red AMD V340L (gfx900) Lineage**
**Document Ref:** `docs/amd/TP2_COMPARATIVE_PARITY_SPEC.md`
**Author:** Gemini (`b4791a54`), Certified per Coordinator Directive C441 / WO-TG1

---

## 1. Ground Truth & Core Laws

### 1.1 The Locked Line Law & Non-Transferable Goldens
- **Donor Fork Artifact ≠ Team Red Artifact**:
  - Donor fork artifact (`ninfer-gfx906 @ 7a3c18d`): 18.21 GB (`sha256: eec39564...`)
  - Team Red production artifact (`qwen3_8_27b_q3.ninfer`): 20.44 GB (`sha256: 0634abb0...`) / pinned 15,446,796,288 bytes.
- **Law**: Fork goldens and absolute numeric logit references DO NOT TRANSFER. Any gate attempting to assert fork goldens on our artifact is broken by construction.
- **Solution**: Parity is defined **comparatively** against **OUR TP1 CONTROL** running on the identical artifact within the same process/environment.

### 1.2 Teacher-Forced Position Comparison (Why Free-Running Agreement Fails)
Greedy decoding is a fixed point of its own output. A single flip at a near-tied logit pair sends two engines down divergent token paths, after which every token is evaluated on disjoint context.
Therefore:
- **Free-running agreement** is a descriptive metric only.
- **Teacher-forced per-position comparison** is the primary mathematical metric:
  - For each position $k$, both TP1 and TP2 evaluate the identical context (prompt + TP1 generated prefix $0..k-1$).
  - Measures true arithmetic drift resulting from split reductions, free from context bifurcation.

---

## 2. Comparative Parity Envelope & Metrics

For each position $k$ across the evaluation corpus ($N \ge 256$ positions):
1. **Argmax Agreement Ratio**:
   $$\text{Agreement} = \frac{1}{N} \sum_{k=1}^N \mathbf{1}[\text{argmax}(L_k^{(\text{tp1})}) == \text{argmax}(L_k^{(\text{tp2})})]$$
   - Target Envelope: **$\ge 95.0\%$** on our artifact.
   - Bounding Control: TP1-vs-TP1 under perturbed chunk size evaluates to ~93.9%–96.6%. The TP2 split must achieve agreement comparable to or better than single-device chunk reordering.

2. **Logit Cosine Similarity ($1 - \cos$)**:
   $$\cos(L_1, L_2) = \frac{L_1 \cdot L_2}{\|L_1\|_2 \|L_2\|_2}$$
   - Mean Cosine: **$\ge 0.990$** ($1 - \cos \le 0.010$).
   - Per-position minimum: **$\ge 0.700$**.
   - Trend invariance: Mean cosine across 64-token context buckets must be flat or rising (zero error accumulation / divergence over context length).

3. **Kullback-Leibler (KL) Divergence**:
   $$D_{\text{KL}}(P_{\text{tp1}} \| P_{\text{tp2}}) = \sum_{v \in V} P_{\text{tp1}}(v) \log \left(\frac{P_{\text{tp1}}(v)}{P_{\text{tp2}}(v)}\right)$$
   - Softmax with temperature $T=1.0$ over top-1024 vocabulary slice.
   - Mean KL Divergence: **$\le 0.025$ nats**.

---

## 3. The 17 Ported TP2 Test Shapes

Ported from donor fork (`/tmp/ninfer-gfx906`) with shapes adapted to Green's canonical interfaces (`src/runtime/tp2/`, `src/core/multi_gpu/`):

| # | Test Target | Focus / Verification Invariant | Skip Code |
|---|---|---|---|
| 1 | `test_allreduce` | Two-device cross-die sum & gather. Event transport under no-P2P. Pointwise relative error $\le 3.95 \times 10^{-3}$ (1 BF16 ulp). | 77 |
| 2 | `test_shard_map` | Qwen3.8-27B tensor sharding geometries (RowSplit vs ColSplit, hidden/intermediate dimensions). Bit-exact slice reconstruction. | 77 |
| 3 | `test_linear_split` | GEMV / GEMM split-k execution on 2 ranks. Numeric tolerance vs FP64 reference. | 77 |
| 4 | `test_linear_add_split` | Fused residual add + split linear projection. Non-degeneracy & tolerance match. | 77 |
| 5 | `test_linear_swiglu_split` | Fused SwiGLU gating MLP split. | 77 |
| 6 | `test_attn_input_proj_split`| Q, K, V input projections across ranks (12 Q heads, 2 KV heads per rank). | 77 |
| 7 | `test_gdn_projections_split` | Gated Delta Net projection splits across 2 dies. | 77 |
| 8 | `test_output_head_split` | Final vocabulary projection / logit un-sharding. | 77 |
| 9 | `test_mtp_split` | Multi-token prediction (MTP) projection split. | 77 |
| 10| `test_gdn_headsplit` | GDN recurrence head-local partition across dies. | 77 |
| 11| `test_attention_headlocal` | Grouped query attention head-local execution (12 Q heads, 2 KV heads per rank). Anti-permutation check. | 77 |
| 12| `test_kv_capacity_tp2` | Per-rank paged KV pool capacity resolution under live `hipMemGetInfo` (VRAM Law). | 77 |
| 13| `test_concurrent_executor` | Dual-stream engine worker thread coordination and barrier synchronization. | 0 (CPU) |
| 14| `test_sharded_materialization`| Weights parser & on-the-fly sharding from container manifest. | 77 |
| 15| `test_engine_tp2_real` | End-to-end model prefill and decode execution across 2 ranks. | 77 |
| 16| `test_engine_mtp_tp2_real` | Speculative decoding execution with MTP draft model across 2 ranks. | 77 |
| 17| `test_graph_tp2` | CUDA/HIP Graph capture and replay semantics for decode steps across 2 dies. | 77 |

---

## 4. Execution Requirements & Safeguards
1. **Zero-GPU Mode**:
   - Compiles all test translation units under `-DNINFER_BACKEND=hip` and `-DCMAKE_HIP_ARCHITECTURES=gfx900`.
   - Host-only tests run cleanly (e.g. `test_concurrent_executor`).
   - Device tests detect available devices; if `< 2`, return exit code `77` (CTEST SKIP).
2. **GPU Execution (Requires Written Coordinator Grant)**:
   - Evaluated under `HIP_VISIBLE_DEVICES=0,1` (Device pair G-AMD-5).
   - Abort criteria: Foreign PID on KFD, VRAM allocation refusal, execution timeout > 60s per unit test.

3. **Transport & Graph Execution Semantics [P1 BANKED]**:
   - **Eager Transport Baseline**: Agent3's P1 facts table on ROCm 6.2.0 confirmed that cross-device HIP graph capture/replay is DEAD on ROCm 6.2.0 (allowance OOM / splice anomaly).
   - **Gate Requirement**: Eager mode is the sole verified, viable transport for TP2 and TP1 controls. Cross-device graph capture/replay is formally excluded from active gate requirements.
   - **Numeric Format & Precision Invariant**:
     - All HIP V-plane writers emit `fp16` bits (repacked via `pack_f16x2` / `bf16x2_bits_to_f16x2_bits`).
     - K-plane remains `bf16` bit storage.
     - Q/K/V attention compute strictly executes in FP16/FP32 math per Decision D3.
   - **Embed-Dense Memory Verification**:
     - Embed-dense kernel is certified as a pure-memory class with ZERO shuffle sites (`__shfl_*`).
     - Verification sweep compares dense memory layout vs TP1 reference without shuffle drift.

