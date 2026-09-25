# 58 — llama.cpp PR Evaluation (2026-08-24)

**Status:** CURRENT — evaluation of six upstream PRs against our TP2 stack
(2× 5060 Ti, Qwen3.8-27B hybrid GDN, MTP k=3, port 8091). Verdicts are
ACCEPT / ACCEPT-PARTIAL (audits) / REJECT with reasons and revisit
conditions. Revisit rejected items when the stated condition is met.

PRs: ggml-org/llama.cpp #26001, #26048, #26705, #27173, #24891, #25635
(all open as of 2026-08-24).

---

## Verdict summary

| PR | Title (short) | Verdict | Action |
|---|---|---|---|
| #27173 | spec draft perf + rollback bugfix | **ACCEPT-PARTIAL** | 3 audits (§1); 2 items already have better equivalents; 1 rejected w/ reason |
| #24891 | checkpoint invalidation after tool requests | **ACCEPT (as audit)** | prefix-cache correctness checklist, §2 |
| #26001 | GDN chunked prefill kernel | REJECT | we already ship an equivalent (§3) |
| #25635 | XOR swizzle FA smem tiles | REJECT | we already use XOR swizzle (§4) |
| #26705 | branchless Q4_K/Q5_K unpack | REJECT | anti-pattern not in our kernels (§5) |
| #26048 | nvfp4 MMQ epilogue scale fusion | REJECT | not our quant/graph stack (§6) |

---

## 1. #27173 — speculative draft performance (+10–12% t/s) + rollback bugfix

Their setup is close to ours (Qwen3.8-27B, 2× RTX, `-sm tensor`, MTP draft),
so this is the most directly transferable PR. Item-by-item against our code:

| # | Their change | Our status |
|---|---|---|
| 1 | Draft all tokens in one GPU call (`LLAMA_SPEC_CHAIN`) | **ALREADY HAVE** — our MTP speculative round produces d0..d2 in one GPU pass (`speculative_round.cuh`); no per-draft-token CPU round trip. |
| 2 | Return only picked token + prob; score top-32k vocab only (`LLAMA_SPEC_CHAIN_SUB`) | **ALREADY HAVE** — `allreduce_argmax` returns only the argmax (tp2_backend.cpp:881,993,1030) and the draft head searches a narrowed vocab (`qwen38_draft_vocab_ids.json`, docs/48). |
| 3 | Mirror full output layer on each GPU (`LLAMA_META_MIRROR_OUTPUT`) | **REJECTED — we solved it cheaper** (see below). |
| 4 | Prepared work plan per batch size (`LLAMA_SCHED_POOL`) | **AUDIT A — DONE (2026-08-24): NO ACTION.** Measured on live server: 800-token MTP decode, wall 12.46 s, total process CPU 24 ms → ~75 µs/round of CPU planning at ~250 ms/round (0.03%). Decode is fully GPU-bound; plan caching / CUDA graphs would buy nothing. |
| 5 | Remember GPU work split + remove extra sync from GPU-to-GPU add | **AUDIT B — DONE (2026-08-24): NO ACTION.** `one_shot_allreduce.cu` + `tp_kernel.cu` contain zero cudaEvents; the only stream sync in `tp_group.cpp` is the explicit `TaskType::Sync` handler, which no runtime code issues. Kernels rendezvous internally (same end-state as post-#27173 upstream); host-side D2H syncs are only where the sampler/token needs data. |
| 6 | Bugfix: wrong state after undoing tokens (short batches overwrote saved states) | **AUDIT C — DONE (docs/64): PASS.** Verify width always k+1, snapshot ring fully rewritten before acceptance, initial state read before any snapshot write, recurrence sequential across columns, prefix slot outside ring. No battery test needed (short rejects occur naturally at ~82% acceptance; T10 + MTP gate cover statistically). |

**Why item 3 is rejected (revisit condition):** our lm_head is
`TpRole::ColumnN` — vocab rows split 124160/rank (tp_load.cpp:126). For
greedy (our only production sampling mode, pi runs temp=0) we already use a
fused `allreduce_argmax` that exchanges just (argmax_id, score) — zero logit
traffic (tp2_backend.cpp:1143–1145, comment "S3"). Only non-greedy pays the
~500 KB allgather. Mirroring the full lm_head on each rank would cost ~1.3–2.5 GB
VRAM/rank — we sit at 14.9 GB/rank at 200k and would OOM. **Revisit only if**
non-greedy sampling becomes a primary workload AND we have ≥2.5 GB/rank headroom
(e.g., after KVarN lands).

**Expected upside from the audits:** small (single-digit % at best) since the
big items are already ours — but AUDIT C is correctness, not perf, and takes
priority if a bug is found.

## 2. #24891 — server: checkpoint invalidation after tool requests (ACCEPT as audit)

Not a code port (different server architecture), but it's the exact failure
class we've been fighting in our prefix cache (D-01 family, T13/T10 decoy
tests). Their 6 fixes map to an audit checklist for `tp_engine` /
`generation_service` prefix handling:

1. **True token-match prefix must never be shrunk by a stale checkpoint value**
   — do we ever let a cached/older length override the measured common prefix?
2. **Clamp restored values to what actually exists; zero on reset.**
3. **Erasure decisions use physical prompt length, not a corrupted position.**
4. **Generation-phase checkpoints must not evict input-phase ones** — our
   single prefix slot (`prefix_cache_capacity`): verify decode output never
   displaces the reusable prompt prefix, and that tool-call rounds (which
   extend history mid-conversation) keep the prefix valid.
5. **Sequence removal must fail soft, never crash** — check our session
   cleanup paths for hard errors on edge cases.
6. **Multi-turn + tools is the repro shape** — pi sessions with tool calls are
   exactly this; T13-style decoy tests should include a tool round mid-history.

**Action:** audit against items 1–5; add a battery test covering
"long prompt → tool call round → continuation reuses prefix (speed check)".
This is the highest-value item here because multi-turn pi sessions are our
primary use case and we've had real bugs in this area.

## 3. #26001 — GDN chunked kernel for prefill (REJECT)

Adds a vLLM/FLA-style chunked GDN CUDA operator (WY inverse → masked attention
→ state update + fused output) replacing token-by-token recurrent prefill.
**We already ship this**: `src/ops/linear_attention/gated_delta_net/chunked/`
has the same 3-stage structure (`prepare_wy_wu.cu` = their stage 1,
`state_passing.cu`, `output.cu`) and it's in production use (A-3 chunked MTP
prefill fix). The PR adds no capability we lack; their accuracy numbers are on
qwen_3_6_35b_a3b (MoE), not our model.

**Revisit if:** prefill t/s plateaus (~827 tok/s today) and profiling shows the
GDN chunked pass as the bottleneck — then diff their kernel for micro-opts
(their stated tricks: BF16/FP16 mix per stage, "FP32-range exponent prevents
saturation" in state GEMMs).

## 4. #25635 — XOR swizzle flash-attn K/V smem tiles (REJECT)

Replaces row padding with XOR address remapping for smem bank conflicts at
high context (their data: 65K+). **We already use XOR swizzle**:
`gqa_small_t_tc_swz(row,col) = (((col>>3) ^ (row&7)) << 3) | (col&7)` in
`gqa_attention_decode.cuh:121` — the same technique, applied to our GQA
decode tiles (and feeding both bf16 and int8 MMA paths).

**Revisit if:** profiling at 200k shows smem bank-conflict stalls in the GQA
kernels.

**Follow-up DONE (2026-08-24):** all three attention kernel families verified
XOR-swizzled with the identical formula `(((col>>3) ^ (row&7)) << 3) | (col&7)`
— decode `gqa_small_t_tc_swz` (bf16 + i8), prefill `gqa_prefill_swz`
(i8 + bf16, K/V/P tiles), bidirectional `bidirectional_gqa_swz`; 46 call
sites, no padded-stride paths remain. The KVarN read kernel (docs/57) mirrors
these files, so it inherits the swizzle.

## 5. #26705 — branchless Q4_K/Q5_K scale unpack in mmvq (REJECT)

Fixes a predicated runtime branch that made nvcc re-execute scale unpack per
column; mask-select instead. Gains at batch ≥4 on sm_120. **The anti-pattern
is not in our kernels**: our Q4 path is custom row-split GEMV/GEMM
(`src/ops/linear/q4/q4_rowsplit_gemv.cu` etc.) with static schedules and
cp.async-pipelined `scale_pairs` — no per-column runtime layout branch exists
to fold (verified in q4_rowsplit_gemv.cuh). We also don't use their mmvq.

**Revisit if:** decode GEMV profiling shows scale-unpack overhead, or we ever
port llama.cpp's mmvq kernels directly.

## 6. #26048 — fuse w_s scale into MMQ epilogue for nvfp4 (REJECT)

Fuses per-weight scale (+bias) multiplication into the MMQ writeback for nvfp4
checkpoints, removing an intermediate global round trip. **Not our stack**: we
run selective GPTQ Q4/Q5 artifacts through custom fused CUDA linears — no
nvfp4 checkpoints, no graph engine with separate MUL(scale) ops; scale folding
into epilogues is already how our kernels are designed (RMSNorm+RoPE fused,
KVarN scales folded by construction).

**Revisit if:** we adopt nvfp4 quantization or a compute-graph scheduler where
elementwise scale ops become separate nodes.

---

## Sequencing (if/when we act)

1. **#24891 audit** (correctness, primary use case) — ~half day + battery test.
2. **#27173 AUDIT C** (rollback state correctness) — ~half day; fix if broken.
3. **#27173 AUDIT A/B** (planning cost measurement, allreduce syncs) — measure
   first; only optimize if the numbers justify it.
4. Everything else stays rejected with the revisit conditions above.
