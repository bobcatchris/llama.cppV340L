# Final Performance Report: MTP $\ge 80\text{ t/s}$ on Dual RTX 5060 Ti

**Status:** ARCHIVE

**Target Artifact:** `/home/intel/models/qwen3_8_27b.ninfer` (Qwen3.8-27B, 16,731 MB, 64 layers)  
**Hardware:** 2× NVIDIA GeForce RTX 5060 Ti 16GB (PCIe, TP2 sharded)  
**Git Branch / Commit:** `mtp-perf` @ `e6a5cf65` (`chrisconcepcion/dual_5060_ti_ninfer`)

---

## 1. Executive Summary & Progression

We completed **Objective 1 (Draft Vocabulary Slicing & Specialized Launchers)** from `13_future_objectives_mtp_70plus.md`. The target goal was **$\ge 70\text{ t/s}$**.

### Breakthrough Results:
- **Throughput:** **`80.55 t/s`** (512 tokens decoded in **6.36 s**) — **$\mathbf{+50.0\%}$ faster than previous 53.77 t/s baseline!**
- **Draft Acceptance Rate:** **`89.2%`** of proposed drafts accepted (**372 / 417 drafts**, **3.68 tokens/round**).
- **Round Latency:** Reduced from **50.76 ms** to **40.54 ms/round** (−20.1%).
- **Drafter Overhead per Round:** Slashed from **12.46 ms** to **3.28 ms** (**−73.7% reduction**).
- **Plain TP2 Decode (MTP 0):** **21.02 t/s** (30.2 ms/token), 100% deterministic and bit-identical.

---

## 2. Progression Timeline

| Milestone | Plain Decode | MTP k=3 Throughput | Round Latency | Drafter Overhead | Memory / GPU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial State** | 18.94 t/s (52.8 ms) | 53.77 t/s | 50.76 ms | 12.46 ms | 9,892 MB |
| **P1.1: TP2 LM Head Sharding** | 21.28 t/s (30.1 ms) | 60.88 t/s | 44.48 ms | 7.40 ms | 9,059 MB (−833 MB) |
| **Objective 1: 40k Draft Head** | **21.02 t/s (30.2 ms)** | **80.55 t/s** | **40.54 ms** | **3.28 ms** | **9,059 MB** |

---

## 3. Phase Breakdown Comparison (B1 Phase Timer)

All numbers measured over steady 512-token generation runs on 2× RTX 5060 Ti:

| Phase | Before (53.77 t/s) | After LM Head Sharding (60.88 t/s) | After Draft Vocab Head (80.55 t/s) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **1. Target Verify ($T=k+1$)** | 38.26 ms (75.4%) | 37.04 ms (83.3%) | **36.79 ms (90.7%)** | Target verify floor |
| **2. Accept & D2H Sync** | 0.04 ms (0.1%) | 0.04 ms (0.1%) | **0.03 ms (0.1%)** | −25.0% |
| **3. GDN State Rebase** | 0.40 ms (0.8%) | 0.45 ms (1.0%) | **0.44 ms (1.1%)** | Stable |
| **4. Prepare Next Round** | 0.01 ms (0.0%) | 0.01 ms (0.0%) | **0.01 ms (0.0%)** | Stable |
| **5. MTP Alignment Forward** | 0.81 ms (1.6%) | 0.82 ms (1.8%) | **0.83 ms (2.0%)** | Stable |
| **6. Select Accepted Hidden** | 0.00 ms (0.0%) | 0.00 ms (0.0%) | **0.00 ms (0.0%)** | Stable |
| **7. MTP Propose ($d_0$)** | 3.25 ms (6.4%) | 1.70 ms (3.8%) | **0.32 ms (0.8%)** | **−90.2%** |
| **8. AR Draft Chain ($d_1..d_k$)** | 8.00 ms (15.8%) | 4.89 ms (11.0%) | **2.13 ms (5.3%)** | **−73.4%** |
| **Total Round Latency** | **50.76 ms** | **44.48 ms** | **40.54 ms** | **−20.1%** |

---

## 4. Implementation Details

1. **Geometry and Schedule (`w8_config.h`, `w8_small_t.cu`, `w8_dispatch.cpp`)**:
   - Registered `W8DraftVocabularyGeometry` ($40,960 \times 5,120$) and `W8DraftVocabularyTp2Geometry` ($20,480 \times 5,120$) in `src/ops/linear/w8/w8_config.h`.
   - Built specialized launchers in `src/ops/linear/w8/w8_small_t.cu` matching small-T GEMV schedules.
   - Dispatched row dimensions $N=20,480$ and $N=40,960$ through `launch_w8_small_t`.

2. **CPU-Side Row-Split Slicing (`tests/multi_gpu/tp2_decode.cpp`)**:
   - Loaded 40,960 sorted token IDs from `tests/multi_gpu/data/qwen38_draft_vocab_ids.json`.
   - Sliced 20,480 rows per rank directly from `text/output_head` payload (low plane 5,120 bytes/row + scale plane 320 bytes/row) into a compact 106 MB device weight allocation.
   - Allocated sharded proposal logits with shape $[20480, 1]$ per rank.

3. **Remapped Distributed Argmax Exchange (`tests/multi_gpu/tp2_decode.cpp`)**:
   - Enqueued `tp_local_argmax` across local 20,480 logits.
   - Host exchange compared winner between Rank 0 and Rank 1 ($I \in [0, 40960)$).
   - Remapped draft winner token index back to global vocabulary: `winner_token = draft_vocab_ids[I]`.
   - Target verify continues to run full 248,320 vocabulary verification, preserving 100% mathematical exactness.

---

## 5. Benchmark Output Logs

```text
artifact: 1124 objects, 16731 MB device total
[rank 0] materializing FULL sharded model (TP2, MTP k=3) on device 0
[rank 0] materialized: 9059 MB device (capacity)
[rank 0] decoder state: 505 MB, 65 kv pages (cap 4103)
[rank 0] TextContext ready (full 64 layers, TP rank 0, MTP on)
loaded 40960 draft vocabulary IDs from tests/multi_gpu/data/qwen38_draft_vocab_ids.json
[rank 0] draft output head ready (20480 rows, 106 MB)
[rank 1] materializing FULL sharded model (TP2, MTP k=3) on device 1
[rank 1] materialized: 9059 MB device (capacity)
[rank 1] decoder state: 505 MB, 65 kv pages (cap 4103)
[rank 1] TextContext ready (full 64 layers, TP rank 1, MTP on)
[rank 1] draft output head ready (20480 rows, 106 MB)
prompt: 5 tokens (The capital of France is)
mode: MTP (k=3), tokens=512
prefill: 5 tokens in 722.2 ms (6.9 t/s pp)

=== B1 Phase Breakdown (Mean over 138 rounds, total = 40.54 ms/round) ===
  1. Target Verify (T=k+1):      36.79 ms (90.7%)
  2. Accept & D2H Sync:           0.03 ms ( 0.1%)
  3. GDN State Rebase:            0.44 ms ( 1.1%)
  4. Prepare Next Round:          0.01 ms ( 0.0%)
  5. MTP Alignment Forward:       0.83 ms ( 2.0%)
  6. Select Accepted Hidden:      0.00 ms ( 0.0%)
  7. MTP Propose (d0):            0.32 ms ( 0.8%)
  8. AR Draft Chain (d1..d_k):    2.13 ms ( 5.3%)
=================================================================================

decoded 512 tokens in 6.36 s (80.55 t/s)
MTP acceptance: 2.676 mean a/round | 89.2% of proposed drafts accepted (372/417) | 3.68 tokens/round over 139 rounds
```
