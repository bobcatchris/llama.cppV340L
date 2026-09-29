# CI Exit Contract Audit & Manifest

**Date:** 2026-09-07  
**Branch:** `wo/ci-exit-contract` (pinned base `9f453037`)  
**Scope:** Full audit of all test suites, drivers, and harnesses invoked by `tools/ops/run_ci.sh`  
**Purpose:** Evidence base and formal specification for the Step [0/3] CI Contract Guard (`tools/ops/contract_lint.sh`) to eliminate wasted CI runs from unsatisfiable, ambiguous, or mis-contracted cells.

---

## 1. Executive Summary & Contract Guard Architecture

In historical CI passes (most notably Pass 2 and early Pass 3 runs), cells frequently failed not due to kernel or numerics regressions, but due to **caller/callee exit code contract mismatches**. The most severe class of this defect is the **unsatisfiable cell**—where the callee's set of success exit codes (`PASS-SET`) and the caller's acceptance set (`caller-accept`) have an empty intersection:
$$\text{PASS-SET} \cap \text{caller-accept} = \emptyset$$

When this occurs, the test battery is mathematically incapable of passing; running a 90-minute full GPU battery before discovering that a cell's success exit code is rejected by `run_ci.sh` represents a complete waste of hardware and engineering time.

### The 4 Fail-Fast Contract Invariants
The CI Contract Guard (`tools/ops/contract_lint.sh`) runs as **Step [0/3]** at the very beginning of `run_ci.sh` (before code state recording, before CMake build, and before any GPU allocation). It statically validates:
1. **Satisfiability:** $\text{PASS-SET} \cap \text{caller-accept} \neq \emptyset$.
2. **Exhaustive Bucketing:** Every code declared in `EXIT-CONTRACT` must map to a deterministic caller verdict branch (no silent unhandled fallthroughs).
3. **Contract Completeness:** Every CI-invoked callee must provide machine-readable `EXIT-CONTRACT` and `PASS-SET` declarations.
4. **No Inverted Acceptance:** The caller accept-set must never contain codes that the callee explicitly defines as failure or error:
   $$\text{caller-accept} \cap \text{FAIL-SET} = \emptyset$$

---

## 2. Machine-Readable Contract Specification

Every driver, harness, and test binary invoked by `run_ci.sh` must emit standardized contract lines when invoked with `--help` (or `--contract`):

```text
EXIT-CONTRACT: <code>=<NAME>[:<description>]; ...
PASS-SET: <code>[,<code>,...]
```

### Grammar Rules
1. `EXIT-CONTRACT`: Semicolon-delimited list of `<code>=<NAME>`.
   - `<code>`: Non-negative integer ($0 \le \text{code} \le 255$).
   - `<NAME>`: Identifier (`[A-Z0-9_]+`).
2. `PASS-SET`: Comma-delimited list of integers denoting successful termination for the given execution mode.
3. For multi-mode scripts (e.g. `phase_gate.sh --compare` vs `--negative`), calling with `<mode> --help` outputs the mode-specific contract, while bare `--help` outputs the union manifest with mode tags.
4. If a binary is uncompiled (clean checkout before Step [2/3] build), the linter extracts the contract directly from the source file (`.cpp`, `.cu`, `.py`, `.sh`) via AST or regex scanning.

---

## 3. Full CI Battery Exit Contract Audit Table

The table below audits all 40 cell invocations across `tools/ops/run_ci.sh` (both default `quick` gate and `--full` attestation battery).

| # | Cell Identifier | Mode | Callee Path & Arguments | Callee Contract (`EXIT-CONTRACT`) | Callee `PASS-SET` | Caller Site (`run_ci.sh`) | Caller Accept Set | Audit Verdict | Notes / Defect Details |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `ctest_gate` | quick/full | `/usr/bin/ctest -E $CTEST_EXCLUDE` | `0=PASS; 1..255=TEST_FAIL` | `0` | L204–207 | `{0}` | **MATCH** | Standard CTest runner. Exclusions pinned. |
| 2 | `hosttests` | quick/full | `/usr/bin/ctest -R mtp/vocab` | `0=PASS; 1..255=TEST_FAIL` | `0` | L216–224 | `{0}` | **MATCH** | CPU unit suites (MTP ngram/seed/adaptive/depth). |
| 3 | `gpu_lease_test` | quick/full | `tools/smoke/diag/test_gpu_lease.sh` | `0=ALL_PASS; 1=FAIL` | `0` | L243–256 | `{0}` | **MISSING CONTRACT** | Exits 0 on all pass, 1 on fail; lacks machine-readable lines in `--help`. |
| 4 | `unit_battery` | quick/full | `$BUILD/tests/ninfer_*_test` (15 binaries) | `0=PASS; 1=ASSERT_FAIL` | `0` | L261–280 | `{0}` | **MISSING CONTRACT** | Standard C++ assert tests. Return 0 on pass. |
| 5 | `kvarn_bench_selftest` | quick/full | `ninfer_slice4_kvarn_bench` (`SELFTEST64=1`) | `0=PASS; 1=ACC_FAIL` | `0` | L281–291 | `{0}` | **MATCH** | Checks exit 0 and parses `acc_norm > 0`. |
| 6 | `kvarn_bench_decay` | quick/full | `ninfer_slice4_kvarn_bench 160256` | `0=PASS; 1=DECAY_FAIL` | `0` | L295–304 | `{0}` | **MATCH** | Parses `T4_MS <= 1.5` ms. |
| 7 | `run_verify_tests` | quick/full | `tools/ops/run_verify_tests.sh` | `0=PASS; 1=FAIL; 2=ENV` | `0` | L309 | `{0}` | **MISSING CONTRACT** | Documented in header; lacks `--help` contract output. |
| 8 | `mtp_t0_ops` | quick/full | `tools/bench/run_t0.sh` | `0=ALL_PASS; 1=FAILS` | `0` | L316–325 | `{0}` | **MISSING CONTRACT** | Runs 6 unit tests. Exits 0 or 1. Needs `--help` contract. |
| 9 | `run_serve_tests` | quick/full | `tools/ops/run_serve_tests.sh` | `0=ALL_PASS; 1=FAIL` | `0` | L327–328 | `{0}` | **MISSING CONTRACT** | Wraps `serve_battery.py`. Exits 0 on success. |
| 10 | `serve_correctness_ci` | quick/full | `tools/smoke/serve_correctness_ci.sh --mode ci` | `0=PASS; 1..N=FAIL_COUNT` | `0` | L332–333 | `{0}` | **MISSING CONTRACT** | Header documents `0=all pass`; needs `--help` contract. |
| 11 | `slice4_phase_gate_prod` | quick/full | `ninfer_slice4_dump` + `pg_compare` | `0=PASS; 4..7=PHASE_FAIL; 90=PARSE_FAIL` | `0` | L348–356 | `{0}` | **MATCH** | D1–D7 production route comparator. |
| 12 | `i8_phase_gate` | quick/full | `ninfer_phase_gate_i8b --kv-dtype i8` | `0=PASS; 1..7=FAIL_PHASE` | `0` | L364–372 | `{0}` | **MATCH** | C++ phase gate runner. |
| 13 | `bf16_phase_gate` | quick/full | `ninfer_phase_gate_i8b --kv-dtype bf16` | `0=PASS; 1..7=FAIL_PHASE` | `0` | L374–382 | `{0}` | **MATCH** | BF16 decode phase gate. |
| 14 | `i8_prefill_gate` | quick/full | `ninfer_phase_gate_i8b --mode prefill --kv-dtype i8` | `0=PASS; 1..4=FAIL_PHASE` | `0` | L385–393 | `{0}` | **MATCH** | I8 prefill phase gate. |
| 15 | `bf16_prefill_gate` | quick/full | `ninfer_phase_gate_i8b --mode prefill --kv-dtype bf16` | `0=PASS; 1..4=FAIL_PHASE` | `0` | L395–403 | `{0}` | **MATCH** | BF16 prefill phase gate. |
| 16 | `kvarn_prefill_gate` | quick/full | `ninfer_phase_gate_i8b --mode prefill --kv-dtype kvarn` | `0=PASS; 1..7=FAIL_PHASE` | `0` | L405–413 | `{0}` | **MATCH** | KVarN prefill phase gate. |
| 17 | `q4_0_phase_gate` | quick/full | `ninfer_phase_gate_i8b --kv-dtype q4_0` | `0=PASS; 1..7=FAIL_PHASE` | `0` | L426–434 | `{0}` | **MATCH** | Self-activating docs/117 cell. |
| 18 | `k5v4_phase_gate` | quick/full | `ninfer_phase_gate_i8b --kv-dtype kvarn_k5v4` | `0=PASS; 1..7=FAIL_PHASE` | `0` | L438–447 | `{0}` | **MATCH** | Self-activating docs/117 cell. |
| 19 | `mtp_round_gate_kvarn` | quick/full | `tools/bench/phase_gate_mb.sh $ART 64 --mtp 3` | `0=PASS; 1=FAIL; 20=DIVERGE` | `0` | L458–466 | `{0}` | **MISSING CONTRACT** | MTP round gate converged match. |
| 20 | `mtp_round_gate_i8` | quick/full | `tools/bench/phase_gate_mb.sh ... --kv-dtype i8` | `0=PASS; 1=FAIL; 20=DIVERGE` | `0` | L470–478 | `{0}` | **MISSING CONTRACT** | I8 MTP round gate. |
| 21 | `mtp_round_gate_bf16` | quick/full | `tools/bench/phase_gate_mb.sh ... --kv-dtype bf16` | `0=PASS; 1=FAIL; 20=DIVERGE` | `0` | L482–490 | `{0}` | **MISSING CONTRACT** | BF16 MTP round gate. |
| 22 | `mtp_accept_mutation_neg` | quick/full | `phase_gate_mb.py $OUTNEG $OUTNEG` (mutated) | `0=MATCH; 20=DIVERGE` | `20` | L498–505 | `{20}` | **MATCH** | Inverted gate: mutation must produce exit 20. |
| 23 | `serve_batched_ci` | quick/full | `tools/smoke/serve_batched_ci.sh` | `0=ALL_PASS; 1..N=FAIL_COUNT` | `0` | L519–528 | `{0}` | **MISSING CONTRACT** | 2a batched serving cell (P0/B0-B3). |
| 24 | `serve_correctness_full` | --full | `tools/smoke/serve_correctness_ci.sh --mode full` | `0=PASS; 1..N=FAIL_COUNT` | `0` | L544–545 | `{0}` | **MISSING CONTRACT** | Full T1-T12 correctness. |
| 25 | `serve_correctness_kvarn` | --full | `serve_correctness_ci.sh --only T1..T19` | `0=PASS; 1..N=FAIL_COUNT` | `0` | L549–551 | `{0}` | **MISSING CONTRACT** | KVarN 250k production context. |
| 26 | `serve_correctness_t14` | --full | `serve_correctness_ci.sh --only T14` | `0=PASS; 1=FAIL` | `0` | L555–557 | `{0}` | **MISSING CONTRACT** | KVarN T14 overshoot tolerance. |
| 27 | `slice4_prod_mutations` | --full | `ninfer_slice4_dump $mut` + `pg_compare` | `4=D1/T/Q/D4; 5=D5; 6=D6; 7=D7` | `{$exp}` | L568–577 | `{$exp}` | **MATCH** | 7 localized mutation cases. |
| 28 | `phase_gate_compare` | --full | `tools/bench/phase_gate.sh --compare` | `0=MATCH; 1..7=PHASE_FAIL` | `0` | L591–599 | `{0}` | **MISSING CONTRACT** | Clean baseline comparison. |
| 29 | `phase_gate_negative` | --full | `tools/bench/phase_gate.sh --negative` | `0=ALL_CAUGHT; 1=UNCAUGHT` | `0` | L604–620 | `{2..7}` | **CRITICAL DEFECT: UNSATISFIABLE** | Callee exits `0` on success; caller treats `0` as FAIL and demands `2..7`. `PASS-SET ∩ caller-accept = ∅`. |
| 30 | `regress_unified` | --full | `tools/regress_unified.sh` | `0=PASS; 1=FAIL; 2=ENV` | `0` | L629–631 | `{0}` | **MISSING CONTRACT** | Unified kernel token correctness. |
| 31 | `decode_guard_cells_ci` | --full | `tools/bench/decode_guard_cells_ci.sh` | `0=PASS; 1=FAIL; 2=ENV` | `0` | L635–637 | `{0}` | **MISSING CONTRACT** | Multi-dtype greedy context ratchet. |
| 32 | `i8_mutations` | --full | `ninfer_phase_gate_i8b --mutate-i$m` (m=1..7) | `$m=MUTATION_CAUGHT` | `{$m}` | L643–647 | `{$m}` | **MATCH** | 7 localized I8 mutations. |
| 33 | `i8_mutate_mask` | --full | `ninfer_phase_gate_i8b --mutate-mask` | `5=MASK_FAIL` | `5` | L650–654 | `{5}` | **MATCH** | MTP mask mutation. |
| 34 | `bf16_mutations` | --full | `ninfer_phase_gate_i8b --mutate-b$m` (m=1,3..7) | `$m=MUTATION_CAUGHT` | `{$m}` | L665–669 | `{$m}` | **MATCH** | 6 localized BF16 mutations. |
| 35 | `bf16_mutate_mask` | --full | `ninfer_phase_gate_i8b --mutate-mask` | `5=MASK_FAIL` | `5` | L672–676 | `{5}` | **MATCH** | MTP mask mutation. |
| 36 | `kvarn_prefill_mutations`| --full | `ninfer_phase_gate_i8b --mutate-p$m` (m=1..7) | `$m=MUTATION_CAUGHT` | `{$m}` | L687–691 | `{$m}` | **MATCH** | 7 localized prefill mutations. |
| 37 | `serve_correctness_bf16`| --full | `serve_correctness_ci.sh --only T1..T11` | `0=PASS; 1..N=FAIL_COUNT` | `0` | L723–732 | `{0}` | **MISSING CONTRACT** | BF16 68k correctness battery. |
| 38 | `mtp_t1_harness` | --full | `tools/bench/run_t1.sh` | `0=PASS; 1=FAIL` | `0` | L735–736 | `{0}` | **MISSING CONTRACT** | Pipeline stress & micro-harness. |
| 39 | `mtp_t2_golden` | --full | `tools/bench/run_t2.sh` | `0=PASS; 1=FAIL` | `0` | L740–741 | `{0}` | **MISSING CONTRACT** | Golden matrix & canaries. |
| 40 | `mtp_t3_standing` | --full | `tools/bench/run_t3.sh` | `0=PASS; 1=FAIL` | `0` | L745–746 | `{0}` | **MISSING CONTRACT** | Standing gates & pre-merge hook. |

---

## 4. Deep-Dive: The `phase_gate.sh --negative` Defect

### 4.1 Defect Mechanism
In `tools/bench/phase_gate.sh:72-127`:
```bash
    --negative)
        # Tests mutations --mutate-d2 through --mutate-d7 sequentially...
        # Each individual mutation test expects exit $rc == $d:
        rc=0; run --mutate-d2 >/dev/null 2>&1 || rc=$?
        if [[ $rc -ne 2 ]]; then
            echo "[phase-gate] NEGATIVE TEST FAIL on D2 (expected exit 2, got $rc)" >&2; exit 1
        fi
        ...
        echo "[phase-gate] ALL NEGATIVE TESTS PASSED (mutations localized to correct phases D2-D7)"
        exit 0
```
When all mutations are caught as designed, the script finishes with `exit 0`. If any mutation is uncaught, it finishes with `exit 1`.

However, in `tools/ops/run_ci.sh:604-620`:
```bash
    stdbuf -oL bash "$REPO/tools/bench/phase_gate.sh" --negative > "$RESULTS/${TS}_phase_gate_negative.log" 2>&1
    PG_NEGATIVE=$?
    # phase_gate --negative exit semantics (phase_gate.sh:74-95):
    #   exit 0        = broken — the negative battery ran and caught NOTHING
    #   exit 1        = a mutation was NOT caught OR the battery failed to start
    #   exit 2..7     = the Dx mutation was caught as designed
    if [ $PG_NEGATIVE -eq 0 ]; then
      echo "  ✗ Phase Gate --negative: exit 0 — mutations were NOT caught; the negative battery is broken"
      EXTRA_OK=1
    elif [ $PG_NEGATIVE -ge 2 ] && [ $PG_NEGATIVE -le 7 ]; then
      echo "  ✓ Phase Gate --negative: exit=$PG_NEGATIVE — mutation caught as designed"
    else
      echo "  ✗ Phase Gate --negative: exit=$PG_NEGATIVE (ambiguous: mutation-not-caught OR failed to start) — tail:"
      EXTRA_OK=1
    fi
```

### 4.2 Mathematical Inconsistency
- **Callee Contract:** `EXIT-CONTRACT: 0=ALL_MUTATIONS_CAUGHT; 1=MUTATION_UNCAUGHT_OR_ERROR`
- **Callee `PASS-SET`:** `{0}`
- **Caller Accept Set (`caller-accept`):** `{2, 3, 4, 5, 6, 7}`
$$\text{PASS-SET} \cap \text{caller-accept} = \{0\} \cap \{2, 3, 4, 5, 6, 7\} = \emptyset$$

### 4.3 Why It Bit
The author of the `run_ci.sh` check assumed that `phase_gate.sh --negative` was a single-mutation run that directly returned the caught phase index ($2..7$), whereas `phase_gate.sh --negative` is an aggregate test suite that tests all 6 phases in a loop and returns $0$ if all 6 passed! As a consequence, whenever `phase_gate.sh --negative` succeeded completely, `run_ci.sh` declared the cell a FAILURE, causing a false red on `--full`.

### 4.4 The Fix
Either:
1. `run_ci.sh` accepts `0` as success for the aggregate battery (`if [ $PG_NEGATIVE -eq 0 ]; then PASS`).
2. Or `phase_gate.sh --negative` is updated to declare its true aggregate contract: `EXIT-CONTRACT: 0=ALL_MUTATIONS_CAUGHT; 1=MUTATION_UNCAUGHT; PASS-SET: 0`, and `run_ci.sh`'s caller branch is aligned to `{0}`.

---

## 5. Contract Remediation Plan for Pass 3

To pass `contract_lint.sh` cleanly, every callee script must add standardized header lines and `--help` / `--contract` handlers:

### 5.1 Standard Script Template
```bash
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "${1:-}" == "--contract" ]]; then
    cat <<'EOF'
EXIT-CONTRACT: 0=PASS; 1=TEST_FAIL; 2=ENV_ERROR
PASS-SET: 0
EOF
    exit 0
fi
```

### 5.2 Specific Callee Contracts

1. **`tools/smoke/diag/test_gpu_lease.sh`**:
   - `EXIT-CONTRACT: 0=ALL_CASES_PASS; 1=ASSERTION_FAILED`
   - `PASS-SET: 0`

2. **`tools/smoke/serve_correctness_ci.sh`**:
   - `EXIT-CONTRACT: 0=ALL_TESTS_PASS; 1..100=FAIL_COUNT; 101=SERVER_LAUNCH_FAILED; 102=ENV_ERROR`
   - `PASS-SET: 0`

3. **`tools/smoke/serve_batched_ci.sh`**:
   - `EXIT-CONTRACT: 0=ALL_CELLS_PASS; 1..10=FAIL_COUNT; 11=SERVER_LAUNCH_FAILED; 12=ENV_ERROR`
   - `PASS-SET: 0`

4. **`tools/ops/run_verify_tests.sh`**:
   - `EXIT-CONTRACT: 0=ALL_PASS_OR_WARN; 1=REGRESSION_FAIL; 2=ENV_ERROR`
   - `PASS-SET: 0`

5. **`tools/ops/run_serve_tests.sh`**:
   - `EXIT-CONTRACT: 0=ALL_PASS; 1=TEST_FAIL; 2=SERVER_ERROR`
   - `PASS-SET: 0`

6. **`tools/bench/run_t0.sh`**:
   - `EXIT-CONTRACT: 0=ALL_T0_PASS; 1=OPS_ASSERTION_FAIL; 2=BUILD_ERROR`
   - `PASS-SET: 0`

7. **`tools/bench/run_t1.sh`**, **`run_t2.sh`**, **`run_t3.sh`**:
   - `EXIT-CONTRACT: 0=PASS; 1=FAIL; 2=ENV_ERROR`
   - `PASS-SET: 0`

8. **`tools/bench/phase_gate_mb.sh`**:
   - `EXIT-CONTRACT: 0=CONVERGED_STATE_MATCH; 1=TEST_FAIL; 20=STATE_DIVERGENCE`
   - `PASS-SET: 0` (clean run) / `PASS-SET: 20` (mutated run)

9. **`tools/bench/phase_gate.sh`**:
   - `--compare`: `EXIT-CONTRACT: 0=MATCH_BASELINE; 1..7=PHASE_MISMATCH; 90=BASELINE_MISSING; PASS-SET: 0`
   - `--negative`: `EXIT-CONTRACT: 0=ALL_MUTATIONS_CAUGHT; 1=MUTATION_UNCAUGHT_OR_ERROR; PASS-SET: 0`

10. **`tools/regress_unified.sh`**:
    - `EXIT-CONTRACT: 0=TOKEN_IDENTICAL_PASS; 1=TOKEN_MISMATCH_OR_CRASH; 2=ENV_ERROR`
    - `PASS-SET: 0`

11. **`tools/bench/decode_guard_cells_ci.sh`**:
    - `EXIT-CONTRACT: 0=ALL_CELLS_PASS; 1=RATCHET_REGRESSION; 2=ENV_ERROR`
    - `PASS-SET: 0`
