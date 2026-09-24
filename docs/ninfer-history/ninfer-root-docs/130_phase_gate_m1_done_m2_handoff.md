# 130 — Phased kernel gate: M1 done + D3 unblocked, M2 handoff

Status: M1 + D3 SHIPPED (committed + pushed to `github`), M2 in progress.
Worktree: `/home/intel/ninfer/worktrees/wo-phase-gate` (branch `wo/phase-gate`).
Remote: `github` (`git@github.com:chrisconcepcion/dual_5060_ti_ninfer.git`).
Artifact: `/home/intel/models/qwen3_8_27b.ninfer` (17 GB, 64 layers, `text/draft_head` Q4G64_F16S $131072 \times 5120$).

---

## 1. Executive Summary

- **M1 + D3 (docs/128 §6 full decode chain D1–D7 with draft head): COMPLETE & GREEN.**
  - Decode chain: `D1 (quantize) → D2 (dequant) → D3 (draft head) → D4 (QK) → D5 (Softmax) → D6 (PV) → D7 (Reduce)`.
  - Model artifact integration: Successfully linked `libninfer_artifact.a` and mapped `text/draft_head` (Q4G64_F16S $131072 \times 5120$) and `text/draft_head_token_ids` ($131072 \times \text{int32}$) directly from `/home/intel/models/qwen3_8_27b.ninfer`.
  - Independent CPU references:
    - Codec CPU reference for D1 / D2 (`src/ops/kvarn/kvarn_codec.cpp`).
    - Exact dequant GEMV CPU reference for D3 (`tests/phase_gate.cu`).
    - Full KVarN decode CPU reference for D4–D7 (`tests/kvarn_decode_cpu_ref.h`).
  - Algebraic identity verified: Code-space QK ≡ Dequant QK ($3.28 \times 10^{-7}$), End-to-end CPU ref vs direct ($2.15 \times 10^{-6}$), Online Softmax vs Split-K merge ($3.57 \times 10^{-7}$).
  - Clean execution: `tools/bench/phase_gate.sh` → exit 0, runtime <1 s GPU.
  - Baseline comparison: `tools/bench/phase_gate.sh --compare` → MATCH baseline (exit 0).
  - Negative tests: `tools/bench/phase_gate.sh --negative` validates exact single-phase failure localization across all active phases:
    - `--mutate-d2` → caught at D2 (exit 2)
    - `--mutate-d3` → caught at D3 (exit 3)
    - `--mutate-d4` → caught at D4 (exit 4)
    - `--mutate-d5` → caught at D5 (exit 5)
    - `--mutate-d6` → caught at D6 (exit 6)
    - `--mutate-d7` → caught at D7 (exit 7)
  - Blocked phases: D8–D10 (e2e sampling/GDN/token stream) require multi-layer full TP2 execution harness.

---

## 2. Phase Table & Measured Tolerances

| Phase | Description | Status | sha256 (first 16) | Measured vs CPU Ref | Tolerance |
|-------|-------------|--------|-------------------|----------------------|-----------|
| **D1** | commit/quantize | PASS | `4c9821bd85385d12` | 1 byte diff, 76 ULP scales | 64 bytes, 128 ULP |
| **D2** | dequantize | PASS | `d272364e22065333` | 24 ULP bf16 tiles | 64 ULP |
| **D3** | MTP draft head | PASS | `a7505509dfe6ef13` | drafts `[20026, 35138, 40950]`, rel error `0.00e+00` | rel error $10^{-4}$ |
| **D4** | QK scores | PASS | `ac48c9279b56f134` | rel error $1.51 \times 10^{-7}$ | rel error $10^{-4}$ |
| **D5** | softmax probs | PASS | `e91211dc7ea3bb6f` | rel error $1.05 \times 10^{-7}$ | rel error $10^{-4}$ |
| **D6** | PV accumulation | PASS | `88d8add0bffbc72f` | rel error $2.53 \times 10^{-7}$ | rel error $10^{-4}$ |
| **D7** | reduce / out | PASS | `f2461073f8aefaad` | 0 ULP (bit-exact bf16) | 8 ULP |
| **D8** | accept/sampling | BLOCKED | `----------------` | model artifact missing (TP2 harness) | N/A |
| **D9** | GDN state | BLOCKED | `----------------` | model artifact missing (TP2 harness) | N/A |
| **D10**| serving glue / e2e | BLOCKED | `----------------` | model artifact missing (TP2 harness) | N/A |

---

## 3. Files Modified

| File | Role |
|------|------|
| `tests/kvarn_decode_cpu_ref.h` | Pure C++ reference for 1152 scale layout, online softmax rescale, Split-K merge, and artifact generation. |
| `tests/kvarn_codespace_qk_cpu_ref.cpp` | Standalone CPU algebraic identity validation suite (Tests A, B, C). |
| `tests/phase_gate.cu` | Full chained execution of D1–D7 + D3 draft head on GPU with direct `Reader` mmap, CPU reference checks, and mutation flags. |
| `tools/bench/build_phase_gate.sh` | Standalone compiler linking `libninfer_artifact.a`, `libninfer_ops.a`, `libninfer_core.a`. |
| `tools/bench/phase_gate.sh` | Complete test runner for `--baseline`, `--compare`, `--negative` (testing D2–D7). |
| `tools/bench/phase_baseline.json` | Recorded hashes for green baseline across D1–D7. |
| `tools/bench/phase_tolerance.json` | Specification and tolerances for D1–D7. |

---

## 4. Verification Commands

```bash
cd /home/intel/ninfer/worktrees/wo-phase-gate
bash tools/bench/build_phase_gate.sh                 # ~40s build
bash tools/bench/phase_gate.sh                       # clean run -> exit 0
bash tools/bench/phase_gate.sh --compare             # vs baseline -> MATCH baseline (exit 0)
bash tools/bench/phase_gate.sh --negative            # tests D2, D3, D4, D5, D6, D7 -> ALL PASS
g++ -O2 -std=c++17 tests/kvarn_codespace_qk_cpu_ref.cpp -o /tmp/kvarn_cs_qk && /tmp/kvarn_cs_qk  # pure CPU ref PASS
```
