# Doc 46: MTP Acceptance Regression — Narrowed Diagnosis

## Status: ACTIVE INVESTIGATION

## What we know (confirmed facts)

| Test | Result | Conclusion |
|------|--------|------------|
| k=1, 64 tokens | **100% acceptance** (32/32) | d0 is ALWAYS correct |
| k=3, 64 tokens | **66.7%** (42/63) | d1/d2 are often wrong |
| k=3, no draft vocab | **66.7%** | Draft vocab is NOT the cause |
| k=3, ctx=128/512/4096 | **66.7%** all | Context size is NOT the cause |
| Same commit (f571a622, 184e9d15) | **67.7%** | Not a code regression between commits |
| Historical battery (Aug 20-21) | **85.6-92.6%** | Was working before |

## The narrowed problem

**d0 path (always works):**
```
prefill: mtp_forward_batch(prompt_tokens, target_hidden[0..plen-1]) → mh[plen-1]
draft_head(mh[plen-1]) → logits → argmax → d0 ✓
```

**d1 path (often broken):**
```
decode: mtp_forward_decode_batch(d0_token, ar_hidden, pos=plen) → next_hid
draft_head(next_hid) → logits → argmax → d1 ✗
```

The MTP module works when fed the **target model's hidden state** (prefill output),
but produces wrong predictions when fed **its own decode output** (ar_hidden).

## Hypotheses (ranked by testability)

### H-A: Decode path produces wrong hidden state (TOP PRIORITY)
The `mtp_forward_decode_batch` output (`next_hid`) should equal what
`mtp_forward_batch` would produce at the same position if run with prompt+d0.
If they differ → bug in decode path (KV cache, positions, or attention).

**Test:** Add debug print comparing:
- Path 1: `mtp_forward_batch` with plen+1 tokens (prompt + d0) → hidden[plen]
- Path 2: `mtp_forward_decode_batch(d0, ar_hidden, pos=plen)` → next_hid
If they differ → decode path bug.

### H-B: MTP KV cache not properly updated during decode
The MTP module has its own paged KV cache (`batch_mtp_kv_`). If the decode
phase doesn't correctly append to this cache, attention over previous
positions will be wrong.

**Test:** Verify `mtp_kv_alloc.publish_mapping()` is called and the page
table is correct before/after decode.

### H-C: RoPE position mismatch between prefill and decode
Prefill uses positions 0..plen-1. Decode uses pos=plen. If there's an off-by-one
or the RoPE table lookup is wrong, attention will be corrupted.

**Test:** Print RoPE positions used in prefill vs decode. Verify continuity.

### H-D: MTP module weights are subtly wrong
The MTP module (pre_fc_norm_embedding, pre_fc_norm_hidden, attention, FFN)
might have been converted incorrectly from the source checkpoint.

**Test:** Dump MTP module weight norms/means and compare against expected
values (e.g., from a Python reference or the 3090 repo if available).

## What we ruled out
- [x] Draft vocab coverage (all missed tokens ARE in the 40k vocab)
- [x] Context window size (same 66.7% at ctx=128/512/4096)
- [x] Code regression between commits (same result on multiple commits)
- [x] Model file change (mtime unchanged, Aug 20)
- [x] GPU thermal issues (38°C idle)
- [x] Sampling config (MTP path is pure argmax, temp=0)

## Next steps
1. Implement H-A test (compare batch vs decode hidden state)
2. If H-A confirms mismatch → investigate H-B and H-C
3. If H-A shows match → investigate H-D (weights)
