# Phase-2 TP4 Delta Map v1 (Zero-GPU Architecture & Call-Graph Prep)

**Document**: `docs/amd/TP4_DELTA_MAP_v1.md`  
**Author**: Gemini (`b4791a54-9948-4610-9ee7-ab96a077c0cb`)  
**Source & independent verifier**: Agent 5 (`69076b58`) — `TP4_world2_inventory` seeded §2; merge-time audit `ed0e446c` verified this doc's load-bearing anchors at bytes. Attribution set per agent5's own word, board #735(2): "happy to be listed as source+verifier, not co-author" — credit narrowed to exactly what each seat wrote.  
**Coordination**: Successor Chair Coordinator C441 (pi `01a09a03`, hub id `8e05f0b6-496c-43ac-88b7-3f25c2ac5e75`)  
**Base Commit**: `origin/amd/main @ 2c49197c` (incorporating `d283bfd4`, `ac72645e`, `7c02a299`, `a5390589`)  
**Discipline**: Pure Zero-GPU source analysis (`--zero-gpu`); call-graph reachability with ref + content anchors; all device-execution dependencies explicitly classified as `UNMEASURED`.

---

## 1. Executive Summary & Seeding Ancestry

This delta map operationalizes the Phase-2 Tensor Parallelism 4-way (`TP4`, `world=4`) transition across the AMD/HIP ninfer stack. It seeds directly from Agent 5's authoritative `TP4_world2_inventory_agent5.md` (on `origin/amd/wo-gfx900-perm`) and executes an exhaustive call-graph reachability analysis across:
1. The **four core world==2 inventory questions** (Artifact TP2-shape, `tp_place_capacity`, AllReduce rank-arithmetic, and the 129-constant).
2. The **81-tensor role-mapping pass** from the pinger queue, resolving the `groups_per_row=9` divisibility concern under the cite-your-denominator law.
3. The **export-vs-port policy question** reserved for board determination alongside the complete residual classification.

Every architectural statement is anchored to verified line references and commit SHAs (`2c49197c`, `426e2ff2`, `5e89bb0d`). Every metric requiring GPU clock or device memory measurement is explicitly labeled `UNMEASURED`.

---

## 2. Deconstruction of the Four World==2 Inventory Questions

### 2.1 Artifact TP2-Shape vs. Rank-Agnostic Manifests
- **Previous Assumption**: Hypothesized that artifact weights were exported in pre-split TP2 slices (per-rank halves/K-halves), which would require re-export or offline recombining for TP4.
- **Call-Graph Anchor**: Resolved by Agent 2 (`5e89bb0d`, #572) and verified against `tools/convert/qwen3_8_27b/inventory.py` and `src/targets/qwen3_6_27b/impl/load/tp_load.cpp:253-312`.
- **Finding**: **The artifact is 100% rank-agnostic and full-shaped**. All 1118 tensors declare unpartitioned global dimensions:
  - `mlp/gate_up`: `[34816, 5120]` (full fused gate + up).
  - `mlp/down`: `[5120, 17408]` (full).
  - `attention/query_key_gate_value`: `[14336, 5120]` (full).
  - `attention/output`: `[5120, 6144]` (full).
  - `gdn/query_key_value_z`: `[16384, 5120]` (full).
  - `gdn/output`: `[5120, 6144]` (full).
- **Reachability Verdict**: `materialize_tp(reader, rank, world, ...)` slices tensors dynamically in host memory at load time using `multi_ranges` and `tp_local_shape`. **Zero export modifications are required; TP4 is an engine/runtime port.**

---

### 2.2 `tp_place_capacity` Sizing & Auto-Capacity Probe
- **Call-Graph Anchors**:
  - `src/targets/qwen3_6_27b/impl/load/tp_load.cpp:322-380`:
    `std::uint64_t tp_place_capacity(const ninfer::manifest::Reader& reader, int world, bool include_mtp)`
  - `src/runtime/tp2/tp_engine.cpp:870-871`:
    `placement_weights_bytes = targets::qwen3_6_27b::detail::tp_place_capacity(placement_reader, 2, include_mtp_placement);`
  - `src/runtime/tp2/tp_engine.cpp:905`:
    `for (const int dev : {b_opts.dev0, b_opts.dev1})`
- **Code Delta Analysis**:
  1. `tp_load.cpp`: `tp_place_capacity` is **already parameterized by `int world`**. Inside the loop over tensors, it accumulates `size_for_role(role, world, entry.shape)`:
     - `MultiRangeGateUp`: `rows / world * cols`.
     - `RowK`: `rows * (cols / world)`.
     - `ColumnN`: `(rows / world) * cols`.
     - `Replicate`: `rows * cols`.
     The capacity calculation scales mathematically to `world=4` cleanly without internal algorithm changes.
  2. `tp_engine.cpp:870`: The call hardcodes literal `2`. Must be changed to parameter `options.world_size` (or `b_opts.devices.size()`).
  3. `tp_engine.cpp:905`: Device discovery iterates over `{b_opts.dev0, b_opts.dev1}`. Must generalize to `b_opts.devices` vector (`{dev0, dev1, dev2, dev3}`).
- **Hardness Class**: **TRIVIAL PARAMETERIZATION** (`MEASURED-CPU`).

---

### 2.3 AllReduce (AR) Rank-Arithmetic & Collective Layer
- **Call-Graph Anchors**:
  - `src/core/multi_gpu/one_shot_allreduce.cu:110-185`:
    `one_shot_reduce_kernel` parameters `local_buf`, `peer_buf`, `local_flag`, `peer_flag`.
  - `src/core/multi_gpu/one_shot_allreduce.h:35-65`:
    `OneShotAllReduce::Impl` allocates `host_epoch[2]`, `dev_epoch[2]`, `host_status[2]`, `host_gen[2]`, and `Slot::host_buf[2]`.
  - `src/core/multi_gpu/one_shot_argmax.cu:140-210`:
    Winner fusion evaluates pairwise local vs peer tokens; `emit_conf` and `my_conf` assume two ranks.
  - `src/runtime/tp2/tp2_backend.cpp:1354-1361`:
    `TpGroupOptions{.devices = {options.dev0, options.dev1}}`, `ctx0_ref`, `ctx1_ref`, `r0`, `r1`.
  - `src/runtime/tp2/tp2_backend.cpp:1372`:
    `TpRankState* const d2_ranks[2] = {r0.get(), r1.get()};`
- **Finding & Hardness Breakdown**:
  - **TpGroup Layer**: World-generic (`options.devices.size()`). Passes `this->size()` cleanly to `ncclCommInitRank`. Constructs for 4 cards without modification.
  - **Backend Construction**: Hardcoded pair construction (`ctx0_ref`, `ctx1_ref`, `r0`, `r1`, `d2_ranks[2]`). Generalizes cleanly via `std::vector<std::unique_ptr<TpRankState>> ranks(world)` and loop `for (int r = 0; r < world; ++r)`.
  - **One-Shot Transport**: **HARD-2 by kernel signature & storage**. Pairwise single-hop memory writes cannot reduce 4 ranks in one step without either:
    - Option A: 2-phase pairwise tree (Rank 0<->1, Rank 2<->3, then Rank 0<->2).
      **AMENDED (chair-placed per agent5 WO-TP4-B finding @ bbba0504): the 3-exchange sketch
      is one exchange SHORT of full-allreduce semantics — after step 3, ranks 1 and 3 are
      stranded without the reduced result. Complete pairwise 2-level allreduce = 4 exchanges
      (reduce: 0<->1, 2<->3; combine: 0<->2; distribute: 0<->1, 2<->3 — the 4-exchange form
      matches the table's finding that the tree option costs 2x a mesh round). Not silently
      fixed in the map
      text; this is the named amendment trail.**
    - Option B: Full mesh 4-way pointer ring (3 peers per rank).
    - Option C: Fall back to standard NCCL / RCCL AllReduce (`ncclAllReduce` via `TpGroup`), isolating one-shot peer kernels to TP2.
- **Hardness Class**: **HARD-2 TRANSPORT / SYMMETRY CONTRACT** (Requires board architectural decision).

---

### 2.4 The 129-Constant (`Q3G64_F16S=129`) Deconstruction
- **Call-Graph Anchors**:
  - `tools/ops/lint_quant_recipe_tier_map.py:11, 36, 130`
  - `tools/bench/q3_ci_amd.sh:185, 205`:
    `expected Q3G64_F16S=129 for qwen3_8_27b_q3 artifact, but TP2 tally differs`
  - `tools/convert/qwen3_8_27b/requant_iq3.py:51-57`:
    `TIER_MAP` targets `Q3G64_F16S` for `mlp/down`, `gdn/output`, `attention/output`, and `text/token_embedding`.
  - `tools/convert/qwen3_8_27b/inventory_nvfp4.py:42-45`:
    `FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))` (16 layers).
    `GDN_LAYERS = tuple(layer for layer in range(64) if layer not in FULL_ATTENTION_LAYERS)` (48 layers).
- **Exhaustive Denominator Census**:
  Under the Cite-Your-Denominator Law, the exact composition of the 129 `Q3G64_F16S` tensors across the 64-layer architecture of Qwen3 27B is:

  Total Q3G64_F16S = 64 (mlp/down) + 48 (gdn/output) + 16 (attention/output) + 1 (token_embedding) = 129.

| Model Role | Target Ninja Format | Layer Set | Count | Sharding Role (`classify_tp`) | Collective Required |
|---|---|---|---|---|---|
| `text/layers/{l}/mlp/down` | `Q3G64_F16S` | All layers {0 .. 63} | **64** | `TpRole::RowK` | **AllReduce** |
| `text/layers/{l}/gdn/output` | `Q3G64_F16S` | GDN layers (48 layers) | **48** | `TpRole::RowK` | **AllReduce** |
| `text/layers/{l}/attention/output` | `Q3G64_F16S` | Full-Attn {3, 7 .. 63} | **16** | `TpRole::RowK` | **AllReduce** |
| `text/token_embedding` | `Q3G64_F16S` | Base model embedding | **1** | `TpRole::Replicate` | None (Gather / Replicate) |
| **TOTAL** | | | **129** | **128 RowK + 1 Replicate** | **128 AllReduce sites** |

- **Key Takeaway**: **128 out of the 129 tensors are `TpRole::RowK` projections**.
  Each requires an AllReduce collective immediately following its linear projection.
  The constant 129 is **model-architecture-derived, NOT TP-degree-derived**. The count 129 remains invariant whether running at TP1, TP2, or TP4.

---

## 3. The 81-Tensor Role-Mapping Pass & Denominator Law

### 3.1 Census and Origin of the 81 Tensors
In the pinger queue and Agent 2/Agent 5 exchanges, 81 tensors were identified as having `groups_per_row = 9` under `group_size = 128`, raising concerns that `9 % 4 = 1` would cause `check_div` failures at `world=4`.

- **Call-Graph Anchor**: `tools/convert/qwen3_6/common/inventory.py:29-130`:
  `VISION_LAYERS = tuple(range(27))` (Exactly 27 layers).
  Per vision layer:
  - `attention/qkv`: `(3456, 1152)`, `Q4G64_F16S`, `kSplitK = 1152`
  - `attention/output`: `(1152, 1152)`, `Q5G64_F16S`, `kSplitK = 1152`
  - `mlp/fc1`: `(4304, 1152)`, `Q4G64_F16S`, `kSplitK = 1152`
- **Denominator Census**:
  - Exactly 3 tensors per layer have column dimension K = 1152.
  - Exactly 27 vision layers exist in the architecture.
  - Total count: 3 x 27 = **81 tensors**.
  - With `group_size = 128`, the column groups per row are:
    `groups_per_row = 1152 / 128 = 9`
  - At `world = 4`:
    `1152 / 4 = 288` and `288 / 128 = 2.25` (non-integer; `9 % 4 = 1`).

### 3.2 Ground-Truth Call-Graph Reachability Resolution
Does this divisibility remainder break TP4?
- **Call-Graph Anchor**: `src/targets/qwen3_6_27b/impl/load/tp_load.cpp:253-292`:
  ```cpp
  TpRole classify_tp(const std::string& name) {
      std::string rel = name;
      const auto a = name.find("text/layers/");
      if (a != std::string::npos) {
          ... // classification of text/layers/...
      }
      if (name.rfind("mtp/layer/", 0) == 0) {
          ... // classification of mtp/layer/...
      }
      if (name == "text/output_head" || rel == "output/weight" || rel == "output_head") {
          return TpRole::ColumnN;
      }
      return TpRole::Replicate;  // All vision tensors fall through here!
  }
  ```
- **Proof of Non-Interference**:
  1. All 81 vision tensors have names beginning with `vision/layers/...`.
  2. In `classify_tp`, any tensor not matching `text/layers/`, `mtp/layer/`, or `text/output_head` returns **`TpRole::Replicate`**.
  3. In `tp_local_shape`:
     `case TpRole::Replicate: return {full_rows, full_columns};`
  4. Replicated tensors maintain full dimensions on every rank. They are **never row-split, never column-split, and never RowK-split**.
  5. The companion non-128 multiple, `vision/layers/{l}/mlp/fc2` with shape `(1152, 4304)` (`4304 % 128 = 80 != 0`), is likewise classified as `TpRole::Replicate`.
- **Conclusion**: The 81 tensors with `groups_per_row = 9` and the non-128-multiple `fc2` tensors **never enter the sharded execution path**. They execute as local replicated vision forward passes. **Zero divisibility faults occur at TP4.**

---

## 4. Hardcoded TP2 Constants in `tp_load.cpp`

A critical code delta uncovered during this reachability pass is the presence of hardcoded TP2 numerical constants in `src/targets/qwen3_6_27b/impl/load/tp_load.cpp:304-307`:

```cpp
TpLocalShape tp_local_shape(TpRole role, int rank, int world, std::int32_t full_rows,
                            std::int32_t full_columns) {
    const auto w = static_cast<std::int32_t>(world);
    (void)rank;
    switch (role) {
        case TpRole::MultiRangeGateUp:   return {full_rows / w, full_columns};
        case TpRole::RowK:               return {full_rows, full_columns / w};
        case TpRole::ColumnN:            return {full_rows / w, full_columns};
        case TpRole::MultiRangeQKV:      return {full_rows / w, full_columns};
        case TpRole::MultiRangeGQKV:     return {full_rows / w, full_columns};
        case TpRole::MultiRangeQK:       return {3584, full_columns};  // <-- HARDCODED (7168 / 2)
        case TpRole::MultiRangeGV:       return {3584, full_columns};  // <-- HARDCODED (7168 / 2)
        case TpRole::MultiRangeGQK:      return {2048, full_columns};  // <-- HARDCODED (4096 / 2)
        case TpRole::MultiRangeGZ:       return {6144, full_columns};  // <-- HARDCODED (12288 / 2)
        case TpRole::GdnConv:            return {full_rows, full_columns / w};
        case TpRole::Replicate:          return {full_rows, full_columns};
    }
    return {0, 0};
}
```

### Parameterization Fix for TP4:
For split-artifact variants, these lines must be generalized to:
- `MultiRangeQK`: `return {7168 / w, full_columns};` (at w=4, 1792 rows/rank)
- `MultiRangeGV`: `return {7168 / w, full_columns};` (at w=4, 1792 rows/rank)
- `MultiRangeGQK`: `return {4096 / w, full_columns};` (at w=4, 1024 rows/rank)
- `MultiRangeGZ`: `return {12288 / w, full_columns};` (at w=4, 3072 rows/rank)

---

## 5. TP4 Subsystem Delta & Verification Classification Matrix

| Subsystem | Source Path | TP2 Implementation | Required TP4 Delta | Status |
|---|---|---|---|---|
| **Weight Placement** | `src/targets/qwen3_6_27b/impl/load/tp_load.cpp:294-312` | Hardcoded 3584/2048/6144 in `tp_local_shape` | Replace with `full_rows / w` formulas | `MEASURED-CPU` |
| **Capacity Probe** | `src/runtime/tp2/tp_engine.cpp:870-910` | Hardcoded `2` passed to `tp_place_capacity`; `{dev0, dev1}` | Pass `world_size`; iterate over 4 devices | `MEASURED-CPU` |
| **Rank Construction** | `src/runtime/tp2/tp2_backend.cpp:1354-1374` | Hardcoded `r0, r1` and `d2_ranks[2]` | Loop `0..world-1` over `vector<unique_ptr<TpRankState>>` | `MEASURED-CPU` |
| **Device Options** | `src/runtime/tp2/tp2_backend.h:180-181` | `int dev0 = 0; int dev1 = 1;` | Add `std::vector<int> devices` | `MEASURED-CPU` |
| **Transport Backend** | `src/core/multi_gpu/one_shot_allreduce.cu` | Pairwise `peer_buf`, 1-hop reduce | 2-phase tree, 4-way mesh, or RCCL fallback | **UNMEASURED-DEVICE** |
| **Argmax Fusion** | `src/core/multi_gpu/one_shot_argmax.cu` | Pairwise token winner + conf ring | 4-rank tree reduction / associative fusion | **UNMEASURED-DEVICE** |
| **VRAM Reserve** | `src/runtime/tp2/tp2_budget.h` (`kTp2RuntimeReserveBytes`) | 1536 MiB calibrated for TP2 | Must re-measure floor at 4-rank geometry (VRAM Law) | **UNMEASURED-DEVICE** |
| **Per-Token KV Sizing** | `src/runtime/tp2/tp2_budget.h` | 256 ch/rank hardwired in KV page math | Parameterize channels by `512 / world` (128 ch/rank) | `MEASURED-CPU` |
| **GDN Tile Geometry** | `src/ops/gdn_gating_proj/bf16/` | 24 heads/rank (48 total) | 12 heads/rank; check NS16/NS32 panel divisibility | **UNMEASURED-DEVICE** |
| **Vision Invariant** | `src/targets/qwen3_6_27b/impl/load/tp_load.cpp:291` | `TpRole::Replicate` | Retain `TpRole::Replicate` (81 tensors untouched) | `MEASURED-CPU` |

---

## 6. Export-vs-Port Policy Question for the Board

The analysis yields a clean separation between export artifacts and engine runtime code:

1. **Artifact Stability**:
   - The export pipeline (`tools/convert/qwen3_8_27b/`) emits rank-agnostic, unpartitioned tensor shapes.
   - The 129 count of `Q3G64_F16S` tensors represents architectural layer roles (64 + 48 + 16 + 1), independent of parallelism scale.
   - The 81 vision tensors with K=1152 are replicated and do not require export changes.

2. **Policy Choice to Put to Board**:
   - **Recommendation A (Pure Engine Port — RECOMMENDED)**:
     Maintain all artifact manifests and conversion scripts frozen. Implement TP4 entirely inside `src/runtime/tp2/` (generalizing to `tp_group` world size), updating `tp_local_shape` and transport mechanisms. Zero re-quantization or re-export required.
   - **Alternative B (Pre-sharded Export)**:
     Emit pre-partitioned rank-specific artifacts at export time.
     *Trade-off*: Multiplies disk footprint by N and breaks single-binary drop-in capability across different card counts.

---

## 7. Residual Checklist & Immediate Next Steps

1. [x] Claim `docs/amd/TP4_DELTA_MAP_v1.md` on `agent-comm`.
2. [x] Complete zero-GPU call-graph reachability analysis for all 4 world==2 questions.
3. [x] Provide exact mathematical census for the 129 `Q3G64_F16S` tensors.
4. [x] Provide complete derivation proving the 81 vision tensors are `TpRole::Replicate`.
5. [ ] **Board Ruling**: Confirm Recommendation A (Pure Engine Port) vs Alternative B.
6. [ ] **Transport Lane Assignment**: Select 2-phase pairwise vs RCCL AllReduce for TP4 collective transport.
7. [ ] **Hardware Window**: Await coordinator allocation of dev0-3 4-card testing window to measure marked `UNMEASURED-DEVICE` metrics (VRAM floor, collective latency, GDN 12-head kernel panel performance).
