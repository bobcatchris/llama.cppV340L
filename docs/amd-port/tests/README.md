# AMD/V340L TP3 200k Served Regression Guard Battery

Parity implementation of the NVIDIA line's served guard battery (`dual_5060_ti_ninfer/tools/bench/decode_guard.sh`) for the `llama.cpp` port on AMD V340 / MI25 hardware.

---

## 1. House Testing Law

**`llama-bench` is NOT our method.**
Synthetic isolated bench numbers fail to capture:
1. Server HTTP lifecycle, request framing, and batch scheduling.
2. Draft-MTP speculative decode loop across verification steps.
3. Realistic KV cache layout and context pressure at 8k-10k+ tokens.

All optimization gates are evaluated against **`llama-server` booted with the config of record**:
```bash
/home/chris/launch_tp3_200k.sh
# HIP_VISIBLE_DEVICES=0,1,2 ... llama-server -ngl 999 -sm tensor -c 200000 \
#   --batch-size 512 --ubatch-size 128 -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp --port 8080 -t 8
```

---

## 2. Guard Cells & Gates

| Guard | Cell | Context Target | Metric | Baseline (PLOG-101) | Gate Tolerances |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prefill Guard** | `prefill_2k` | ~2,075 tokens | `prompt_per_second` | **96.50 t/s** | `+-5%` warn (`<91.68`), `-10%` fail (`<86.85`) |
| **Decode Guard** | `decode_8k_10k` | ~7,857 tokens | `predicted_per_second` | **15.57 t/s** | `+-5%` warn (`<14.79`), `-10%` fail (`<14.01`) |
| **MTP Canary** | `mtp_canary` | 8k-10k context | `draft_acceptance` | **0.667** | Fail if `< 0.630`, Warn if `< 0.650` |
| **Determinism Guard** | `determinism_greedy` | Greedy, temp=0 | `sha256(text)` | **byte-identical** | Must produce identical text across 2 consecutive runs |
| **Needle Recall** | `needle_recall_8k` | 8k context | `exact_code_match` | **3/3 depths** | Must recall 5-digit code at 25%, 50%, and 75% depth |

- **Draft mean accepted length**: tracked against PLOG-101 baseline of `3.00` tokens/step.

---

## 3. Directory Layout

- `baseline_tp3_200k.json`: Monotonic baseline store & gate thresholds.
- `probe_prompts.py`: Deterministic probe extractors (`probe_2k.txt` prose & 7,857-token prompt).
- `guard_battery.py`: Core Python evaluation suite, HTTP client, timing collector, and receipt writer.
- `run_tp3_guards.sh`: Bash orchestration runner handling PCIe boot, health polling, execution, and teardown.

---

## 4. Usage

### Full Boot (Standard CI / Verification Run)
Batches all three guards into a single 3-4 minute PCIe server boot:
```bash
./docs/amd-port/tests/run_tp3_guards.sh
```

### Running Against an Already Active Server
If `llama-server` is already booted on port 8080:
```bash
./docs/amd-port/tests/run_tp3_guards.sh --no-boot
# or directly via python:
python3 docs/amd-port/tests/guard_battery.py --port 8080
```

### Monotonic Ratchet (After Verified Optimization Gain)
To permanently raise baseline performance gates when a code optimization lands:
```bash
./docs/amd-port/tests/run_tp3_guards.sh --ratchet
```

---

## 5. Receipts & Provenance

Every execution appends an immutable JSON record to `docs/amd-port/results/tp3_guards_<STAMP>.jsonl` with:
- Exact Git commit (`HEAD`), dirty status flag, and diff hash.
- GGUF model path, size, and timestamp.
- Server configuration parameters.
- Per-cell metrics, delta percentages, and individual PASS/WARN/FAIL verdicts.
- Overall run verdict (Exit `0` = PASS, `2` = WARN, `1` = FAIL).
