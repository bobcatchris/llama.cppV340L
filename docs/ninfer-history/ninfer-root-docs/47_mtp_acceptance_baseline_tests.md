# MTP Acceptance Baseline Tests

**Status:** ARCHIVE

## Purpose
Establish what "correct" MTP acceptance rates should be at k=1, k=2, k=3.
Use as a regression gate and completion criterion.

## Test Methodology
Run the MTP test driver with fixed parameters and measure:
- Acceptance rate (% of proposed drafts accepted)
- Tokens per round (mean tokens generated per MTP round)
- Throughput (tokens/second)

## Fixed Parameters
```
--artifact /home/intel/models/qwen3_8_27b.ninfer
--tokens 512
--ctx 4096
--prompt "The capital of France is"
--no-draft-vocab   # isolate MTP module quality from draft vocab
```

## Expected Results (to be filled in after baseline run)

### k=1
| Metric | Expected | Measured | Status |
|--------|----------|----------|--------|
| Acceptance | ?% | ? | ? |
| Tokens/round | ~2.0 | ? | ? |
| Throughput | ? t/s | ? | ? |

**Theory:** With k=1, there's only 1 draft per round. The MTP module predicts d0
from the target model's hidden state. If the MTP module is well-trained, d0
should match the target ~95%+ of the time. Each accepted draft gives 2 tokens
(1 draft + 1 bonus from target).

### k=2
| Metric | Expected | Measured | Status |
|--------|----------|----------|--------|
| Acceptance | ?% | ? | ? |
| Tokens/round | ~2.5-3.0 | ? | ? |
| Throughput | ? t/s | ? | ? |

**Theory:** With k=2, there are 2 drafts per round. d0 should be ~95% accurate.
d1 is produced by the MTP module using d0's token and its own hidden state.
If the MTP module is well-trained, d1 should be ~70-80% accurate (conditional
on d0 being accepted). Expected tokens/round: 2 + 0.95*0.75 ≈ 2.7.

### k=3
| Metric | Expected | Measured | Status |
|--------|----------|----------|--------|
| Acceptance | ?% | ? | ? |
| Tokens/round | ~3.0-3.5 | ? | ? |
| Throughput | ? t/s | ? | ? |

**Theory:** With k=3, there are 3 drafts per round. d0 ~95%, d1 ~75%, d2 ~55%
(conditional on previous drafts being accepted). Expected tokens/round:
3 + 0.95*0.75*0.55 ≈ 3.39.

## Current Measurements (2026-08-22, with --no-draft-vocab)

### k=1
| Metric | Measured |
|--------|----------|
| Acceptance | **100%** (32/32) |
| Tokens/round | 2.00 |
| Rounds | 32 |

### k=3
| Metric | Measured |
|--------|----------|
| Acceptance | **66.7%** (42/63) |
| Tokens/round | 3.05 |
| Rounds | 21 |

## Analysis

**k=1 is perfect (100%).** The MTP module's d0 prediction is always correct when
using the target model's hidden state from prefill. This confirms:
- MTP module weights are correct
- Target model hidden state is correct
- Draft head is working correctly

**k=3 is degraded (66.7% vs expected ~75%).** The MTP module's d1/d2 predictions
are often wrong when using its own decode output. This suggests:
- Bug in MTP module's decode path (KV cache, positions, or attention)
- OR the MTP module is not well-trained for autoregressive prediction

## Regression Gate

Any change to the MTP code must maintain:
- k=1 acceptance >= 95%
- k=3 acceptance >= 70% (target: 85%+)
- k=3 tokens/round >= 3.0 (target: 3.5+)

If these thresholds are violated, the change has introduced a regression.

## Completion Criterion

MTP is "done" when:
- k=1 acceptance >= 98%
- k=3 acceptance >= 85%
- k=3 tokens/round >= 3.5
- k=3 throughput >= 90 t/s (with draft vocab)
