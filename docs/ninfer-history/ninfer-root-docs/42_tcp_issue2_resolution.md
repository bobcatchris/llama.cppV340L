# TC-P Prefix Caching Issue 2: Root Cause & Resolution Report

**Date:** 2026-08-22  
**Commit:** `0a3b50e3`  
**Status:** **RESOLVED & VERIFIED** (`Prefix identity TC-P(a) on==off` **PASS**)

---

## 1. Problem Summary

Under test condition **TC-P** (prefix caching enabled with 225-token repeated prompt in `/home/intel/verify_battery.sh`), Request 2 produced output tokens that diverged when prefix caching was ON vs when prefix caching was OFF (`--no-prefix-cache`).

---

## 2. Root Cause Analysis

Through step-by-step tensor probing and split isolation across all layers and heads:

1. **Tokens 0..224 (Prompt Prefill):**
   - In Request 1, prefill tokens $0 \dots 224$ executed correctly.
   - At position 224 (the last prompt token), the base model produced logits that correctly emitted `t0_token` (the first generated token).
   - `t0_token` is the model's generated token intended for position $\text{plen} = 225$.

2. **The Bug in MTP Decode Initialization:**
   In [`tp2_backend.cpp`](file:///tmp/ninfer/src/runtime/tp2/tp2_backend.cpp):
   ```cpp
   int cur_anchor = t0_token;
   int cur_F = plen - 1; // BUG: Initialized to 224 instead of 225
   ```
   At Round 0 of MTP speculative verification in Request 1:
   - `base_pos` was set to $F = \text{plen}-1 = 224$.
   - `speculative_prepare_verify_inputs` constructed:
     - `verify_ids = [t0_token, d0, d1, d2]`
     - `verify_pos = [224, 225, 226, 227]`
   - `target_verify_batch` executed `t0_token` **at position 224 instead of position 225**.
   - **`target_verify_batch` overwrote the KV cache entry of prompt token 224 with `t0_token`'s key-value vectors.**

3. **Downstream Effect on Request 2 (`prefix_on`):**
   - Request 2 skipped prefilling tokens $0 \dots 224$ (relying on the KV cache).
   - But position 224 in `text_kv` had been corrupted by `t0_token` during Request 1's decode.
   - In `prefix_off`, Request 2 freshly prefilled tokens $0 \dots 224$, writing the uncorrupted prompt token 224 into position 224.
   - When token 225 was processed, the attention score with key 224 ($Q_{225} \cdot K_{224}$) computed differing scores between `prefix_on` and `prefix_off`, leading to divergence.

---

## 3. Resolution Applied

In [`tp2_backend.cpp`](file:///tmp/ninfer/src/runtime/tp2/tp2_backend.cpp#L1100):
- Initialized `cur_F_mtp = plen;` (225) for MTP speculative verification.
- Round 0 now correctly evaluates positions `[plen, plen+1, plen+2, plen+3]` (`[225, 226, 227, 228]`).
- The entire prompt KV cache at indices $0 \dots \text{plen}-1$ remains pristine across multi-request sessions.

---

## 4. Verification Results

Running [`/home/intel/verify_battery.sh`](file:///home/intel/verify_battery.sh):

| Metric | Result | Verdict |
| :--- | :--- | :--- |
| **Prefix identity TC-P(a) on==off** | **yes** | **PASS** |
| **Prefix skip (fewer tokens, ≤90% ms)** | **yes (228.30 ms vs 6695.90 ms)** | **PASS** |
| **Determinism (2 runs)** | **yes** | **PASS** |
| **A2 token identity (MTP==plain)** | **yes** | **PASS** |
| **Draft vocab active (MTP run)** | **yes** | **PASS** |
| **Sampling determinism (seed 42)** | **yes** | **PASS** |
| **Sampling divergence (seed 43)** | **yes** | **PASS** |
| **Plain decode t/s (40 tok)** | **35.51 t/s** | **PASS** |
| **Prompt processing (pp)** | **32.00 t/s** | **PASS** |
| **MTP $k=3$ Decode Throughput** | **80.19 t/s (acceptance 67.7%)** | **PASS** |

Cleanly committed (`0a3b50e3`) and pushed to `origin` (`dual_5060_ti_ninfer`) and `local` backup.
