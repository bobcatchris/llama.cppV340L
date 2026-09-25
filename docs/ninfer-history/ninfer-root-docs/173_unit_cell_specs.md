# Unit-Cell Assertions & Test Specification for Docs/173 §9.4

Author: gemini (test builder)  
Target: `wo/mb-prefix-cache` (owner: agent4)  
Work Order: [`docs/173_multibatch_prefix_caching_inverse_gap_work_order.md`](file:///home/intel/ninfer/repo/docs/173_multibatch_prefix_caching_inverse_gap_work_order.md) §9.4  

---

## 1. Scope & Architecture

Per docs/173 §9.4, three unit cells accompany Step 2 of the multibatch prefix caching implementation:
1. **LCP Resolver Table Test (`test_mb_lcp_resolver.cpp`)** — Table-driven validation of host-side token ledger LCP resolution, capture-point demotion (R1), and greedy dispatch lane assignment (§5 D6).
2. **Snapshot Roundtrip Multi-Lane Extension (`test_prefix_snapshot_roundtrip.cpp`)** — Extension of `tests/test_kvarn_write_path.cpp:737` to concurrent per-lane snapshots with cross-lane isolation.
3. **Per-Lane GDN Park→Restore Bit-Exact Golden Gate (`test_mb_park_restore_golden.cpp`)** — Extension of the docs/156 golden-gate pattern to lane $b \ge 1$ using buffer-backed GDN park tensors (§5 D3).

---

## 2. Cell 1: LCP Resolver Table Test (`test_mb_lcp_resolver.cpp`)

### 2.1 Execution Model
- **Environment**: CPU-only (zero GPU seconds).
- **Target**: Functions `resolve_lane_lcp`, `resolve_capture_demotion`, `assign_dispatch_lanes`.

### 2.2 Table-Driven Test Matrix

| Case ID | Ledger Tokens | Prompt Tokens | Capture Points | Expected LCP | Expected Source | R1 Demoted? |
|---|---|---|---|---|---|---|
| `LCP_EMPTY` | `[]` | `[10, 20, 30]` | `[64, 128]` | 0 | `NONE` | No |
| `LCP_FULL_HIT` | `[1..128]` | `[1..128, 999]` | `[64, 128]` | 128 | `FULL` | No |
| `LCP_SNAP_HIT` | `[1..130]` | `[1..130, 999]` | `[64, 128, 130(snap==park)]` | 130 | `SNAP` | No |
| `LCP_SNAP_ABOVE_PARK` | `[1..130]` | `[1..130, 999]` | `[64(park), 130(snap > park)]` | 64 | `GDNCKPT` | No (demotes to park 64) |
| `LCP_CKPT_HIT` | `[1..100]` | `[1..100, 999]` | `[64]` | 64 | `GDNCKPT` | No (snapped to 64) |
| `LCP_DEMOTED` | `[1..42]` | `[1..42, 999]` | `[64, 128]` | 42 | `NONE` | **Yes (R1)**: lcp=42, source=NONE |
| `LCP_MISMATCH` | `[1, 2, 3]` | `[4, 5, 6]` | `[64]` | 0 | `NONE` | No |

> [!NOTE]
> **Batched D-16 Snapshot Constraint (A4 step-2 amendment, 2026-09-12)**:
> In batched serving, a D-16 snapshot is a valid restore target ONLY when `snap_tc == gdn_ckpt_token_count` (park).
> If `snap_tc > gdn_ckpt_token_count` (e.g. `LCP_SNAP_ABOVE_PARK` where snap is at 130 but GDN park buffer is at 64),
> restoring the snapshot is unsafe because GDN state lacks the delta tokens. The resolver demotes to the highest
> valid GDN park checkpoint (`source=GDNCKPT`, `prefix_len=64`). This is a valid checkpoint hit, NOT an R1 demotion.

### 2.3 Dispatch Lane Preference Assertions (Docs/173 §5 D6)
```cpp
// Given 2 lanes with active ledgers:
LaneCacheEntry lanes[2];
lanes[0].ledger = {1, 2, 3, 4, 5, 6, 7, 8}; // Conv A prefix
lanes[1].ledger = {9, 10, 11, 12, 13, 14};  // Conv B prefix

// Test A: Normal arrival order [Req A, Req B]
std::vector<Request> batch_normal = {req_conv_A, req_conv_B};
auto assign_norm = assign_dispatch_lanes(batch_normal, lanes);
assert(assign_norm[0] == 0); // req A -> lane 0
assert(assign_norm[1] == 1); // req B -> lane 1

// Test B: Inverted arrival order [Req B, Req A]
std::vector<Request> batch_inv = {req_conv_B, req_conv_A};
auto assign_inv = assign_dispatch_lanes(batch_inv, lanes);
assert(assign_inv[0] == 1); // req B -> lane 1 (highest LCP)
assert(assign_inv[1] == 0); // req A -> lane 0 (highest LCP)
```

---

## 3. Cell 2: Snapshot Roundtrip Multi-Lane Extension

### 3.1 Execution Model
- **Environment**: GPU-enabled (ctest / small arena, zero model weights).
- **Target**: `kvarn_capture_prefix_snapshot` and `kvarn_restore_prefix_snapshot` parameterized by `lane_idx`.

### 3.2 Invariants & Assertions
1. **Per-Lane Bit-Exactness**:
   - Lane 0 captured at 130 tokens, Lane 1 captured at 194 tokens.
   - Workspaces rewound independently:
     `kvarn_rewind_views(lane_ws[0]); kvarn_rewind_views(lane_ws[1]);`
   - Restored independently:
     `kvarn_restore_prefix_snapshot(lane_ws[0], lane_cache[0].snapshot);`
     `kvarn_restore_prefix_snapshot(lane_ws[1], lane_cache[1].snapshot);`
   - Append delta tokens to completion.
   - Assert: Lane 0 codes and scales are byte-identical to sequential control 0.
   - Assert: Lane 1 codes and scales are byte-identical to sequential control 1.
2. **Cross-Lane Isolation**:
   - Clobbering `lane_ws[1]` after Lane 0 restore does NOT alter Lane 0 output codes or scales.

---

## 4. Cell 3: Per-Lane GDN Park→Restore Bit-Exact Gate

### 4.1 Execution Model
- **Environment**: GPU-enabled, 48 GDN layers, small synthetic slot allocations.
- **Target**: Buffer-backed GDN park/restore (§5 D3).

### 4.2 Invariants & Assertions
```cpp
// 1. Capture at lane prefill finalize for Lane b (b in {0, 1})
park_gdn_buffers(st, lane_b);
uint64_t h_park_conv = fnv1a_device_tensor(st.lane_cache[lane_b].gdn_park_conv);
uint64_t h_park_rec  = fnv1a_device_tensor(st.lane_cache[lane_b].gdn_park_rec);

// 2. Destructive clobber of active slot b
cudaMemset(st.gdn_conv_state_slot(lane_b), 0x5A, slot_bytes_conv);
cudaMemset(st.gdn_rec_state_slot(lane_b), 0xA5, slot_bytes_rec);

// 3. Restore execution
restore_gdn_buffers(st, lane_b);
uint64_t h_restored_conv = fnv1a_device_tensor(st.gdn_conv_state_slot(lane_b));
uint64_t h_restored_rec  = fnv1a_device_tensor(st.gdn_rec_state_slot(lane_b));

// 4. Golden Gate Assertions
assert(h_park_conv == h_restored_conv);
assert(h_park_rec  == h_restored_rec);

// 5. Deliberate Corruption Self-Test (NINFER_MB_PG_CORRUPT=gdn)
flip_one_byte(st.lane_cache[lane_b].gdn_park_conv);
restore_gdn_buffers(st, lane_b);
uint64_t h_corrupt = fnv1a_device_tensor(st.gdn_conv_state_slot(lane_b));
assert(h_corrupt != h_park_conv); // Triggers INPUTS-DIFFER:PC_RESTORE:L{lane}:gdn
```

---

## 5. Step-5 CI Contract-Lint Cell Pre-Registration (Docs/173 §9.4)

When Step 5 lands live `PC_*` phase dump sites and reconciliation in `run_ci.sh`, `tools/ops/contract_lint.sh` will expand from 29 to 30 evaluated cells. The registered row shape and invariants are pre-registered below:

### 5.1 Registry Entry Definition (`contract_lint.sh:CELLS`)
```python
    {
        "id": "mb_prefix_cache_phase_gate",
        "tier": "full",
        "callee": "tools/bench/phase_gate_mb.py",
        "args": ["$SERVE_LOG", "--prefix-cache", "--mode", "reconcile"],
        "caller_accept": {0},
        "caller_hint": "run_ci.sh:([ $MB_PG_EXIT -eq 0 ] || EXTRA_OK=1)",
        "drift_greps": ["MB_PG_EXIT -eq 0"],
        "synthetic_contract": {
            "EXIT-CONTRACT": {
                0: "ALL_LANES_PARITY_MATCH",
                1: "DIVERGENCE_DETECTED",
                2: "CORRUPT_NOT_LOCALIZED",
            },
            "PASS-SET": {0},
        },
    },
```

### 5.2 Expected Table Row Output
```text
Cell ID                    Mode   PASS-SET     Caller-Accept  Verdict
--------------------------------------------------------------------------------
mb_prefix_cache_phase_gate full   0            {0}            PROVEN ✓
```

### 5.3 Invariant Guarantees
1. **Satisfiability**: Callee PASS-SET `{0}` $\cap$ Caller-Accept `{0}` = `{0}` $\neq \emptyset$.
2. **Exhaustive Bucketing**: Non-zero exits (1, 2) fail the cell and set `EXTRA_OK=1` in `run_ci.sh`.
3. **Contract Completeness**: Callee declared via `--help` or `synthetic_contract` fallback.
4. **No Inverted Acceptance**: Caller-Accept `{0}` $\cap$ FAIL-SET `{1, 2}` = $\emptyset$.

### 5.4 Annex: Step-3a MTP Rewind Phase Grammar & Oracle Routing Specification

Per coordinator directive and A4 §9.5 live finding (`d021ee33`), the batched MTP T2 red is named and diagnosed via `PC_*` emission phases in Step 5.

#### 5.4.1 Step-3a Restore Phase Emission Sequence (Mandatory vs Gated Kinds)
On any active MTP lane (`mtp=1` / `mtp_k > 0`) executing a prefix-cache restore to LCP, the emission contract recognizes four mandatory phases and three mechanism-gated phases:

**Mandatory Phases (always emitted on batched MTP prefix restore):**
1. `PC_RESTORE:L{lane}:gdn` — GDN conv & recurrent state restored from lane park buffers to slot $b$.
2. `PC_RESTORE:L{lane}:tile_reset` — Counter-based tile/page reset for page-aligned parks (normal batched path).
3. `PC_RESTORE:L{lane}:rewind_text` — Text KV cursor and envelope rewound to LCP (`text_kv_base`).
4. `PC_RESTORE:L{lane}:rewind_mtp` — MTP draft KV, recurrent draft seed (`ar_hidden`, `draft0`), and mtp-row tracking rewound to LCP.

**Mechanism-Gated Conditional Phases (emitted only when mechanism gates pass):**
- `PC_RESTORE:L{lane}:kvarn_tail` — Gated on `snap_tc == park` (unaligned tail token restore from snapshot). Page-aligned parks emit `tile_reset` instead.
- `PC_RESTORE:L{lane}:ph` — Gated on separate trunk-hidden store (inactive on batched trunk path where consume-at-production holds per WO §5 D4 amendment).
- `PC_RESTORE:L{lane}:stage` — Gated on staging arena allocations (inactive at production geometry where `stage_pages == 0` per docs/83).

#### 5.4.2 Oracle Routing & Diagnostic Verdicts for `kind=rewind_mtp`
The oracle evaluates the MTP restore path with zero ambiguity:
- **Case A: `kind=rewind_mtp` is PRESENT-but-wrong** (Hash mismatch between ON and OFF arms, or rank 0 vs rank 1 divergence):
  - **Tail Predicate**: `INPUTS-DIFFER:PC_RESTORE:L{lane}:rewind_mtp`
  - **Verdict**: `FAIL` (exit code 1)
  - **Reading Rule**: `R2_PARK_IMAGE_INCOMPLETE: Park image incomplete (park->restore hash mismatch, docs/156 class). Fix = extend capture list.`
  - **Diagnosis**: MTP draft state corrupted during restore or park image incomplete. Pinpoints the point of failure directly to MTP draft tensor restore before speculative decode begins (Rule R2).
- **Case B: `kind=rewind_mtp` is ABSENT on MTP T2 path** (`mtp=1` or `mtp_k > 0` active, but `PC_RESTORE:L{lane}:rewind_mtp` never observed during prefix restore):
  - **Tail Predicate**: `FAIL:PC_RESTORE:L{lane}:MISSING_MTP_REWIND`
  - **Verdict**: `FAIL` (exit code 1)
  - **Reading Rule**: `R2b_MTP_REWIND_BYPASSED: MTP rewind phase absent on active MTP lane during prefix restore. Draft state unaligned; triggers downstream speculative decode divergence.`
  - **Diagnosis**: MTP draft state rewind was bypassed or skipped on an active MTP lane during prefix reuse. Corrupts MTP draft state carry, inevitably causing speculative decode divergence in later rounds (Rule R4 cascade).

---

## 6. Contract Summary

- **Satisfiability**: All unit-cells compile without requiring weight tensors or multi-GPU execution.
- **Fail-Fast**: Any corrupted byte or stale pointer immediately causes an FNV mismatch, localizing to `PC_RESTORE:L{lane}:{kind}` per docs/128 §2.5 lockstep rules.
- **Contract Lint Expansion**: Zero-GPU pre-registration guarantees seamless 29 $\rightarrow$ 30 cell transition upon Step-5 live emission.
- **MTP Rewind Naming**: §5.4 Annex fully specifies the diagnostic routing for the live MTP T2 failure class.


