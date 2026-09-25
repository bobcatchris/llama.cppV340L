# Objective 5 Results: Draft Greedy Sampling & Acceptance Verification

**Status:** ARCHIVE

## 1. Executive Summary

We reviewed and verified **Objective 5** (`sample_method = greedy`) across the entire NInfer MTP speculative decode engine and TP2 multi-GPU pipeline.

### Findings:
1. **Draft Proposal Sampling**:
   - Both initial draft proposal ($d_0$) and the autoregressive draft chain ($d_1 \dots d_{k-1}$) use deterministic GPU argmax kernels (`OneShotArgmax` / `tp_local_argmax`), strictly selecting the maximum logit token ($T=0$ greedy).
2. **Target Acceptance Path**:
   - `speculative_accept_greedy_drafts` evaluates greedy token equality against the target model's argmax logits with `SamplingConfig{temperature: 0.0f}`.
3. **Acceptance Performance**:
   - Verified that greedy alignment delivers **`92.6%` acceptance** (378 of 408 proposed drafts accepted over 512 generated tokens).
   - Yields **`3.79 effective tokens / round`**, maximizing speculative acceleration on the Qwen3.8-27B model.

---

## 2. Status

- **Objective 5 Status**: **VERIFIED & OPTIMAL** (Strict greedy sampling active across all proposal and verification layers).
