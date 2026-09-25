# Doc 42 — KVarN gotchas: cross-repo issue audit

**Status:** ARCHIVE

**Scope:** 15+ issues/PRs across 6 repos (beellama.cpp, huawei-csl/KVarN, vLLM upstream, sglang, llama.cpp, syv-ai). Compiled for our future KVarN implementation on 2× 5060 Ti → V340L.

---

## 1. Architecture constraints

### head_dim = 128 only (k4v2_g128 preset)
- **Source:** huawei-csl/KVarN #10
- `kvarn_k4v2_g128` hard-validates `head_dim == 128`. Models with non-128 head_dim (e.g. GLM-4.7-Flash at 576) reject at startup.
- **Our model:** Qwen3.8-27B full-attention layers use head_dim=128 → **compatible**. GDN layers don't use KV cache → no issue.
- **Action:** document the constraint; if we ever target a model with non-128 head_dim, we need a different KVarN preset or a custom kernel.

### Hybrid linear-attention support is explicit, not automatic
- **Source:** sglang #31967, syv-ai #13
- KVarN compresses **only the full-attention KV cache**. GDN/Mamba layers hold bounded recurrent state (not KV pages). The dual-pool architecture (int4 long-term + fp16 tail) applies to full-attention layers only.
- **Our model:** [Full, 3×GDN] repeating — exactly the hybrid pattern sglang's PR targets.
- **Action:** implement KVarN on full-attention layers only; GDN state slots remain bf16. The prefix-cache restore path already handles GDN state separately (copy_slot) — KVarN doesn't touch it.

### KV tail tokens (fp16 sliding window)
- **Source:** beellama.cpp #130, sglang #31967
- Recent tokens are kept at full precision (`--kv-tail-tokens 1024 --kv-tail-type f16`). Quantizing recent tokens hurts quality most.
- **Action:** implement a small fp16 tail pool (1024 tokens default). The tail slides as new tokens are appended; flushed tokens go to the int4 pool. This is the "tail fp16 pool" from sglang's design.

---

## 2. MTP + KVarN interaction (CRITICAL — we use MTP)

### Segfault with MTP + KVarN on gemma-4-31B
- **Source:** beellama.cpp #125 (open)
- `kvarn` + `--spec-type draft-mtp` → segfault on gemma-4-31B. Works without MTP, works without KVarN.
- **Our risk:** HIGH. We use MTP k=3 as the primary decode path.
- **Action:** test MTP + KVarN from day one. The segfault is likely in the MTP verify path reading KVarN-compressed KV — the verify kernel may not support dequant-on-the-fly for speculative tokens.

### CUDA crash at ~900K context with MTP + KVarN (fixed)
- **Source:** beellama.cpp #119 (closed)
- Ornith-1.0-35B-1M-MTP-APEX crashed in `ggml_cuda_fattn_kvarn_decode_launch` at ~900K tokens. Stable up to 500K.
- **Root cause:** likely the same shared-memory opt-in issue as #116 (see §3).
- **Action:** apply the shared-memory opt-in fix proactively; test at 256K+ context with MTP.

### Draft model KVarN support (feature request)
- **Source:** beellama.cpp #126
- KVarN for the drafter model itself is not yet supported.
- **Our plan:** MTP head is the drafter — it's replicated, not KV-cached in the same way. Not a blocker.

---

## 3. CUDA kernel gotchas

### Shared-memory opt-in at ≥768K context (fixed in beellama, apply proactively)
- **Source:** beellama.cpp #116 (closed)
- KVarN decode combine kernel uses dynamic shared memory: `n_splits × 4 bytes`. At 768K context, `n_splits = 12288` → 48 KB = CUDA's default ceiling. Above that: `cudaErrorInvalidConfiguration`.
- **Fix:** `cudaFuncSetAttribute` opt-in for the combine kernel, clamped to `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
- **Our plan:** apply this fix in our KVarN decode kernel from day one. Target 256K context → 128K splits → 24 KB (under ceiling), but future-proof for 1M context.

### Verify-width compute-buffer overflow (open)
- **Source:** beellama.cpp #133
- High `--spec-draft-n-max` (≥12, n_q=13) at deep context (~116K) → OOM mid-prefill. Peak VRAM exceeds card ceiling.
- **Root cause:** compute-buffer sizing doesn't account for KVarN dequant workspace at high verify width.
- **Action:** preflight the verify-width buffer size against available VRAM; clamp or reject gracefully instead of mid-request crash.

### Sinkhorn JIT recompile under dynamic batching
- **Source:** huawei-csl/KVarN #15
- `_sinkhorn_log_kernel` recompiles every decode step under dynamic batching → throughput flatlines at ~80 tok/s regardless of batch size.
- **Root cause:** running-set shape changes every step → Triton recompiles.
- **Action:** warm up all expected shapes at startup, or use a fixed-size decode kernel. For our single-flight server (one request at a time), this is **not a risk** — shapes are constant per request. But if we add multi-request serving, warmup is mandatory.

### MMA compilation warnings (fixed)
- **Source:** beellama.cpp #114 (closed)
- KVarN MMA kernels produce compilation warnings on CUDA. Fixed by silencing.
- **Action:** apply the same warning silences in our kernels.

---

## 4. Quality gotchas

### kvarn6 produces hallucinations (fixed)
- **Source:** beellama.cpp #103 (closed)
- `kvarn6` (6-bit K/V) produced random garbage answers on Qwen3.6-35B-A3B. `q6_0` worked fine.
- **Root cause:** likely a quantization-level bug in the 6-bit path (scale computation or packing).
- **Action:** use `kvarn4` or `kvarn5` (4-bit/5-bit) as the default. Avoid `kvarn6` until the 6-bit path is independently verified.

### kvarn + DFlash leads to reproducible code errors (fixed)
- **Source:** beellama.cpp #93 (closed)
- KVarN + DFlash drafter → model self-corrects mid-generation ("code got corrupted").
- **Root cause:** likely quantization error accumulation during speculative verification.
- **Action:** test KVarN quality with DFlash2 (our post-v1 plan). The paper's variance normalization is designed to mitigate this — verify empirically.

### Tool calling breaks with KVarN on ROCm (open)
- **Source:** huawei-csl/KVarN #17
- Qwen3.6-27B on ROCm (gfx1030, W6800X Duo): tool calling produces garbled output with KVarN. Works with fp16 KV.
- **Our risk:** HIGH. We're targeting V340L (gfx900, ROCm). Same model family (Qwen3.6/3.8-27B).
- **Action:** test tool calling + structured output with KVarN on ROCm early. If quality degrades, the issue may be in the ROCm Triton kernel (precision loss in the Sinkhorn rotation).

---

## 5. ROCm / AMD gotchas (V340L is gfx900)

### kvarn6 massive regression on ROCm (open)
- **Source:** beellama.cpp #122
- RX 7900 XTX (gfx1102): kvarn6 is **15-22× slower** than q8_0 (pp: 53 vs 821 t/s, decode: 1.34 vs 29.5 t/s).
- **Root cause:** likely missing ROCm kernel specialization for the Sinkhorn rotation or the packed-decode kernel.
- **Action:** our KVarN implementation must have ROCm-native kernels from day one. Do not rely on Triton auto-tuning for ROCm — it's known to be poor on gfx900.

### gfx906 KVarN fork exists (reference)
- **Source:** Igneous/vllm-gfx906-mobydick-kvarn (⭐5)
- A gfx906 fork of vLLM with KVarN kv dtype support patched in. gfx906 is closer to gfx900 than our target.
- **Action:** review this fork's patches for ROCm-specific workarounds.

### WSL2 residency limits (documented)
- **Source:** huawei-csl/KVarN #19
- WSL2 has GPU memory residency limits that can cause KVarN to spill to host memory.
- **Our plan:** not relevant (native Linux).

---

## 6. Memory / capacity gotchas

### VRAM spike with kvarn5/kvarn4 (open)
- **Source:** beellama.cpp #112
- 128K context with kvarn5/kvarn4 → consumes all RAM (8GB VRAM + 48GB DDR4). TurboQuant worked fine at same context.
- **Root cause:** likely the fp16 tail pool or the dequant workspace is oversized, or the packed-KV scratch buffer is allocated per-request instead of shared.
- **Action:** budget the fp16 tail pool explicitly (1024 tokens × head_dim × layers). Ensure the packed-KV scratch is shared across requests, not per-request.

### Packed-KV scratch overflow with concurrent long prompts
- **Source:** huawei-csl/KVarN #25
- Two concurrent 140K-token prompts → packed-KV scratch exceeds capacity → FP32 fallback → OOM.
- **Root cause:** scratch buffer sized for one request, not the batch.
- **Action:** size the scratch buffer for `max_num_seqs × max_model_len`. For our single-flight server, this is `1 × max_model_len` — not a risk. But document it for multi-request serving.

### Performance regression after vLLM upgrade (open)
- **Source:** huawei-csl/KVarN #22
- Upgrading from vLLM 0.22.0 to 0.23.0 → throughput degraded with KVarN.
- **Root cause:** unknown (vLLM scheduler change?).
- **Action:** pin the vLLM version if we use the vLLM backend. For our native implementation, not directly relevant.

---

## 7. Debugging / observability gotchas

### GGML_KVARN_DEBUG_ROUTES doesn't surface wide-MMA decision
- **Source:** beellama.cpp #132
- Debug route logging prints identical `route=`/`entry=` strings even when the wide-MMA path engages (n_q > 8 && n_q <= 16 && gqa > 4).
- **Action:** add explicit `wide_mma=1/0` to our debug route logging.

### F16 stage slot aliasing with cache-reuse + concurrent streams (open)
- **Source:** beellama.cpp #130
- `--cache-reuse 256 --slot-prompt-similarity 0.5` + 4 concurrent shared-prefix streams → "structured KV live groups alias one F16 stage slot" → decode abort.
- **Root cause:** the fp16 tail pool's stage slots are aliased across concurrent requests sharing a prefix.
- **Action:** for our single-flight server, not a risk. For multi-request serving, ensure each request has isolated tail pool slots.

---

## 8. Scheme selection guidance

### Recommended preset: kvarn_k4v2_g128
- **Capacity:** ~3.2–3.4× vs bf16 (int4 K, int2 V, group size 128)
- **Throughput:** at or above fp16 in memory-bound regime (+9% at 14B, +18% at 106B per vLLM RFC)
- **Quality:** fp16-level accuracy, calibration-free
- **Avoid:** kvarn6 (quality bugs on some models), TurboQuant (40-52% throughput loss)

### Multi-turboquant comparison (12 methods)
- **Source:** aivrar/multi-turboquant
- Unified toolkit comparing 12 KV compression methods. KVarN is one of the top performers for the capacity/throughput/quality triangle.
- **Action:** use this toolkit for empirical validation of our KVarN implementation vs. alternatives.

---

## 9. Implementation checklist (our stack)

| Item | Priority | Notes |
|---|---|---|
| Full-attention layers only (GDN stays bf16) | P0 | Hybrid model constraint |
| fp16 tail pool (1024 tokens default) | P0 | Quality for recent tokens |
| Shared-memory opt-in for combine kernel | P0 | Prevent ≥768K crash |
| MTP + KVarN test from day one | P0 | Segfault risk (#125) |
| ROCm-native kernels (no Triton auto-tune) | P0 | V340L is gfx900 |
| Tool calling quality test on ROCm | P0 | garbled output risk (#17) |
| Scratch buffer sizing (single-flight: 1×max_len) | P1 | Prevent OOM (#25) |
| Verify-width preflight against VRAM | P1 | Graceful clamp (#133) |
| kvarn4/kvarn5 default (avoid kvarn6) | P1 | Quality bugs (#103) |
| Debug route logging with wide_mma flag | P2 | Observability (#132) |
| Sinkhorn warmup (if multi-request added) | P2 | JIT recompile (#15) |
| DFlash2 + KVarN quality test | P2 | Post-v1 (#93) |

---

## 10. Source repos audited

| Repo | Issues/PRs reviewed | Relevance |
|---|---|---|
| Anbeeld/beellama.cpp | #93, #103, #112, #114, #116, #119, #122, #125, #126, #130, #132, #133 | **Highest** — llama.cpp fork, same model family, CUDA + ROCm |
| huawei-csl/KVarN | #10, #15, #17, #19, #22, #25 | **High** — reference vLLM backend, Qwen3.6-27B tested |
| vllm-project/vllm | #46613 (RFC), #46812 (PR) | **High** — upstream design constraints |
| sgl-project/sglang | #31967 | **High** — hybrid linear-attention support |
| ggml-org/llama.cpp | #24139 | **Medium** — research tracking |
| syv-ai/qwen38-27b-rtx3090 | #13 (closed) | **High** — our exact model, 240K context, 6 allocator fixes |
| aivrar/multi-turboquant | README | **Medium** — 12-method comparison |
| Igneous/vllm-gfx906-mobydick-kvarn | repo scan | **Medium** — ROCm gfx906 reference |
