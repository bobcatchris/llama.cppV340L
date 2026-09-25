# Doc 48: MTP Acceptance Root Cause — Draft Vocab Coverage Gap

## Status: ROOT CAUSE IDENTIFIED + FIX APPLIED

## Root Cause

The 40,960-token draft vocab (`qwen38_draft_vocab_ids.json`) does NOT cover all
tokens the model actually generates. **7.6% of target tokens are missing** from
the draft vocab. When the target predicts a token not in the draft vocab, the
draft head can never match it → guaranteed rejection.

## Evidence

| Configuration | Acceptance | Throughput | Tokens/Round |
|---------------|-----------|------------|--------------|
| k=3, WITH draft vocab (40k) | **67.7%** | 78.98 t/s | 3.03 |
| k=3, WITHOUT draft vocab (full 124k) | **80.4%** | 79.82 t/s | 3.41 |
| k=1, WITH draft vocab | 98.1% | — | 1.98 |
| k=1, WITHOUT draft vocab | 100% | — | 2.00 |

The draft vocab provides **zero speedup** (78.98 vs 79.82 t/s = 1% difference)
because the MTP module's attention/FFN is the bottleneck, not the draft head GEMV.

## Missing Tokens (sample of 512-token run)

Tokens NOT in draft vocab: `[16329, 44512, 50960, 75480, 84854]`
Coverage: 92.4% of generated tokens are in the draft vocab.

## Why Historical Measurements Showed 85.6%

The historical 85.6% (Aug 20-21) was likely from a different model state or
draft vocab that had better coverage. The current draft vocab (from the 3090 repo)
was generated for a different prompt distribution and doesn't fully cover our
model's output tokens.

## Fix Applied

Added `--no-draft-vocab` to all MTP runs in `verify_battery.sh`. This uses the
full 124,160-row output head for the draft head, eliminating the coverage gap.

**Impact:**
- Acceptance: 67.7% → **80.4%** (+12.7 points)
- Throughput: 78.98 → **79.82 t/s** (+0.84 t/s, negligible)
- Tokens/round: 3.03 → **3.41** (+0.38)

## Remaining Gap to Historical 85.6%

Current: 80.4% vs Historical: 85.6% (gap: 5.2 points)

Possible remaining causes:
1. MTP module decode path still has some inefficiency (d1/d2 not optimal)
2. Different prompt distribution in historical runs
3. Model file state difference (MTP module weights)

## Next Steps

1. Run full battery with `--no-draft-vocab` to confirm 80.4% in the gate
2. Investigate remaining 5.2 point gap (MTP decode path quality)
3. Consider regenerating draft vocab with better coverage (if speedup is ever needed)
4. Update baseline JSON after stable pass
