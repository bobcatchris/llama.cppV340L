# 131 — MTP Multi-Batch Test Suite Runbook

## Overview
This runbook guides engineers and automated CI runners through executing, diagnosing, and maintaining the MTP Multi-Batch Test Suite defined in [`docs/131_mtp_test_suite_requirements.md`](./131_mtp_test_suite_requirements.md).

## Test Tiers

### Tier 0: Ops-Level Unit Tests (C++/CUDA)
Fast, isolated ops tests that validate specific invariants without full model execution:
- `ninfer_t0_stride_audit_test` (**T0.8**): Element strides (`tail_batch_elems = kG * kD * kv_heads`), slice contiguity, block-table row stride, MTP tile slices. Host-only, no GPU required. Validates **INV-10**.
- `ninfer_t0_prepare_inputs_test` (**T0.5**): Tests `speculative_prepare_verify_inputs` under normal, `ext=0` degenerate, partial, and B=1 shrink configurations. Validates **INV-7**.
- `ninfer_t0_accept_kernel_test` (**T0.4**): Speculative greedy draft acceptance (`all-accept`, `none-accept ext=0`, `partial-accept`, `next_ext` monotonicity). Validates **INV-7**.
- `ninfer_t0_gdn_verify_test` (**T0.3**): GDN conv1d & recurrent snapshot column masking (`valid = 0..4`), zero-filling unused columns, lane-keyed GDN ring mapping. Validates **INV-5**.
- `ninfer_t0_tile_roundtrip_test` (**T0.7**): Validates open tile fidelity across page boundary crossings, rewinds, and hydrates for text layers and MTP layer (`layer_index = -1`). Validates **INV-4**.
- `ninfer_t0_attend_geometry_test` (**T0.1, T0.2, T0.6**): MultiBatch attend geometry battery across B=2 steady, B=1 shrink, `ext=0` degenerate, page boundaries, shuffled tables, permuted rows, tail consistency (**INV-3**), and memory robustness (**T0.6**).

**Execution:**
```bash
./tools/bench/run_t0.sh
```

---

### Tier 1: Pipeline Micro-Harness & Runtime Instruments
Validates the full pipeline under stress with invariant assertions and deterministic checkpoint hashing:
- **T1.1**: Batched runner stress over geometry matrix $T \in \{64, 128, 256\} \times \{B=2, B=1 \text{ shrink}\}$.
- **T1.2**: Invariant assertion mode (`NINFER_MB_INVASERT=1`) checking **INV-2**, **INV-3**, **INV-5**, **INV-7**, **INV-9**, **INV-10** on every round.
- **T1.3**: Checkpoint hashing (`NINFER_MB_HASHPT`) comparing state hashes across runs.
- **T1.4**: Multi-run determinism across in-process and fresh processes (**INV-6**).

**Execution:**
```bash
./tools/bench/run_t1.sh [--quick]
```

---

### Tier 2: End-to-End Suite & Canaries
Full-model verification comparing batched MTP against isolated single-sequence execution:
- **T2.1**: Golden matrix evaluation vs isolated fresh-process references (`NINFER_SEQ_ONLY`).
- **T2.2**: Flake battery verifying zero-divergence across repeated runs.
- **T2.4**: Soak battery evaluating 8 consecutive requests on the same backend.
- **T2.5**: Canary (must-fail) injection tests asserting detection of known bug classes (`NINFER_CANARY_TAIL_COUNT`, `NINFER_CANARY_SWAP_BT`, `NINFER_CANARY_UNRANK_M0`).

**Execution:**
```bash
./tools/bench/run_t2.sh [--quick]
```

---

### Tier 3: Standing Gates & Bench Integrity
- **T3.1**: Standing gate rule (`t3_gate_rule.sh`) automatically triggered on changes to core MTP/KVarN/GDN sources.
- **T3.2**: Bench integrity check (`bench_integrity.py`) enforcing non-trivial token outputs and port 8091 exclusivity.
- **T3.3**: Cause map in [`docs/131_mtp_test_suite_requirements.md §9`](./131_mtp_test_suite_requirements.md#9-cause-map-with-exact-file-and-line-pointers).
- **T3.4**: Master runner `run_all.sh`.

**Execution:**
```bash
./tools/bench/run_t3.sh
```

---

## Running All Tiers
To execute the complete test suite:
```bash
./tools/bench/run_all.sh [--quick]
```
Output format:
```
PASS=<pass_count> FAIL=<fail_count> FIRST_FAIL=<first_failure_name_or_NONE>
```
Exit codes: `0` = green, `1` = divergence/failure, `2` = build error.
