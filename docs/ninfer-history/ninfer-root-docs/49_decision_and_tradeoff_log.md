# 49 — Decision & Trade-Off Log (LIVING DOCUMENT)

**Status:** ARCHIVE

> Every significant change, why it was made, what it caused (measured), and the
> trade we accepted. Append new entries at the bottom. Performance numbers are
> measured on 2× RTX 5060 Ti, Qwen3.8-27B Q4_1, 4k ctx unless noted.
>
> Statuses: `ACTIVE` (in current tree), `REVERTED`, `RE-APPLYING`, `CLOSED` (done, kept),
> `ABANDONED` (measured net-negative).

## 1. Decision Table

| # | Decision | Why | Measured effect | Trade accepted | Status |
|---|----------|-----|-----------------|----------------|--------|
| D1 | **TP2 (tensor split) over PP2** for 27B | PP2 serializes; TP2 halves per-rank compute | decode 2× per-GPU efficiency | complex NCCL/one-shot machinery | CLOSED |
| D2 | **MTP k=3** as default | k=4 anomaly (doc 22); k=3 best accept/tokens-per-round | k=3: ~80 t/s, ~82% accept | — | CLOSED |
| D3 | **CUDA graphs OFF** | measured net-negative (doc 20, 27) | — | lost ~2ms/round potential | ABANDONED |
| D4 | **I8 KV cache opt-in, not default** | quality gate; acc 78.9% (I8) vs 82.0% (BF16) | kv_i8 gate passes at 81.5% acc / 80% t/s floor | no VRAM savings by default | ACTIVE |
| D5 | **TC-P requires bit-identity** | prefix cache must be exact, no "close enough" | strictest gate | may fail on benign FP noise | ACTIVE |
| D6 | **Draft vocab DISABLED** (default `""` in tp2_backend.h; `--no-draft-vocab` in battery) | coverage only 92.4% of model outputs (<95% threshold); **zero speedup** — MTP module attn/FFN is the bottleneck, not draft-head GEMV; W8G32 | accept 67.7% → 80.4–82.0% (+12–14 pts) | battery "Draft vocab active" check now FAILs vs old baseline (expected — see §3); no draft-head quantization in serve | ACTIVE (2026-08-22) |
| D7 | **M6 chunked prefill** (T=512) replacing T=1 prefill loop | T=1 prefill was 32 t/s — unacceptable prompt processing | pp 32 → ~265–270 t/s (8.3×) | more complex prefill path; one WIP regression cycle (suspected A2 culprit, proven NOT the cause 2026-08-22) | ACTIVE (WIP, uncommitted) |
| D8 | **S3 sampling fix**: plain decode branches — temp=0 `allreduce_argmax` (greedy, == MTP verify); temp>0 `allgather`→full[248320] + `ops::sample` | argmax-only can't support temp/top-k/top-p; original S3 impl (ncclSum on ColumnN shards) was a bug (A-1, resolved) | A2 identity PASS (temp=0); plain temp=1 seed=42 deterministic, seed=43 diverges | allgather adds ~496KB x2 comm per non-greedy token (negligible vs ~29ms step) | ACTIVE (fixed 2026-08-22) |
| D9 | **H1 MTP position fix** `cur_F_mtp = plen` → `plen - 1` | off-by-one base position corrupted KV | small accept gain; part of ~80% recovery | — | ACTIVE |
| D10 | **One-shot allreduce reset desync fix** | TC-P prefix identity broken by stale epoch state | TC-P PASS on 14:25 binary | — | ACTIVE (regressed again in fresh 18:08 binary — open, see §2 A-2) |
| D11 | **BUILD_TESTING=ON required** for test binaries | tests OFF by default in CMakeLists (line 19) | — | easy to forget | CLOSED (run_ci.sh passes it) |
| D12 | **Explicit CUDA 13.1 compiler** `/usr/local/cuda-13.1/bin/nvcc` | PATH default is 12.9; 13.1 already installed | builds succeed | — | CLOSED |
| D13 | **CI records dirtiness, does not gate on it** (2026-08-22, user directive) | tests must run on dirty trees; tests pass BEFORE pushing; code state (commit + diff hash + file list) recorded in every result | results now carry `_diff_hash`, `_modified_files` | stale/unknown-state results possible if someone ignores metadata | ACTIVE |
| D14 | **Terminology: "test suites" not "batteries"** (2026-08-22, user) | user preference | scripts renamed: run_verify_tests.sh, run_serve_tests.sh, run_ci.sh | — | CLOSED |
| D15 | **Per-check pass/fail in results JSON** (`checks[]`, `_verdict`, `_fails`, `_warns`) (2026-08-22, user) | latest.json had metrics but no verdicts; must be self-service readable | run_verify_tests.sh emits checks table | — | ACTIVE (verify done; serve/ci to follow) |
| D16 | **Target model = Qwen3.8-27B** (not 3.6) | user target; 3.6 retired | all results on 3.8 artifact | — | CLOSED |

## 2. Open Investigation Items (with provenance)

### A-1 — S3 sampling path produces degenerate output at temp=0 (RESOLVED 2026-08-22)
- **Symptom:** with S3 (allreduce_local_bf16 + ops::sample, temp=0), plain decode emits
  "It is beautiful and beautiful…" loop; `allreduce_argmax` path emits coherent
  "The capital of Germany is Berlin."
- **ROOT CAUSE (confirmed):** LM head is **ColumnN-sharded** (tp_load.cpp:125, "split
  vocab rows 248320 / world"): each rank owns `n_vocab = 248320/2 = 124160` rows = half
  the vocab; rank r owns tokens `[r*124160, (r+1)*124160)`. S3 called
  `allreduce_local_bf16(rank, logits, 124160)` → `ncclAllReduce(ncclSum)` which **sums
  disjoint token slices** (logit(global_i) + logit(global_i+124160)) = garbage.
  `ops::sample` greedy argmax over garbage → degenerate loop. S3 never actually worked:
  the old 14:25 binary predated S3, which is why A2 "passed" before.
- **Fix (in tree):** plain decode branches on temperature:
  - `temp==0`: `allreduce_argmax` (fused, exact, matches MTP verify) → A2 identity holds.
  - `temp>0`: new `TpGroup::allgather_local_bf16` (ncclAllGather) assembles the full
    [248320] distribution into `st.full_logits`, then `ops::sample(..., 248320, ...)`.
- **Verified:** plain temp=0 == MTP temp=0 (A2); plain temp=1 seed=42 deterministic
  (A==B), seed=43 diverges.
- **Lesson:** ColumnN-sharded logits must never be combined with a sum-based allreduce.
  Only (a) fused argmax over shards or (b) gather-to-full-distribution. Do NOT revert S3
  again (D8, user directive).

### A-2 — TC-P prefix identity regressed in fresh binary (RESOLVED 2026-08-22)
- **Repro (deterministic, not flaky):** `--prompt2 "<LONGPROMPT> And to summarize in one
  word."` with/without `--no-prefix-cache`; outputs matched through generated token ~57,
  then diverged (char 1286: ON "provided the[i]r" vs OFF "provided [space]answer").
- **Root cause (confirmed):** GDN recurrent state was stored in **FP16**
  (`tp2_backend.cpp` `.recurrent_dtype = DType::FP16`). Fresh path (232 tokens in one
  prefill) keeps the state in fp32 registers the whole way and quantizes to FP16 only at
  the end. Cached path (req1=225 tokens, then req2=7 on top) quantizes the state to FP16
  at the 225 boundary, then continues from the quantized state → tiny drift → one close
  argmax flips at ~token 57. (The GDN kernel is already a *sequential ordered* recurrence,
  not a parallel scan — the "non-associative scan" premise was wrong.)
- **Fix (in tree):** `.recurrent_dtype = DType::FP32` — lossless state across the
  prefix-cache boundary, so cached-then-continue (225 + 7) is bit-exact with a single
  fresh pass (232). All GDN kernels, the state pool, and replay support FP32.
- **Verified:** prefix on/off output2 now `IDENTICAL` (1344/1344 chars). Decoder state
  204→725 MB (FP16→FP32); total VRAM ~9.8 GB (still well under 16 GB).
- **Trade:** ~0.5 GB extra state memory for bit-exact prefix caching. Chosen per user
  (2c: proper fix, not a gate relaxation).

### A-3 — Acceptance gap 82.0% vs 85.6% historical (OPEN)
- 85.6% came from dirty-tree binaries (docs 46–48); clean tree + no-draft-vocab = 82.0%.
- Candidate lever: **W6 — MTP quantized head** (share target's Q4 lm_head with MTP
  draft head so quantization error correlates). Added 2026-08-22 per user.
- Baseline gates (doc 47): k=1 ≥95%, k=3 ≥70% (target 85%+), tok/round ≥3.0 (target 3.5+).

## 3. Baseline & Gate Housekeeping

- `baseline.json` is **STALE**: captured pre-M6 (pp 32, round 37.8ms, acc 67.7%,
  draft_vocab ON). Current reality (2026-08-22 18:51 run): pp ~265, round 42.7ms
  (identical on old and fresh binaries → not a regression), acc 82.0%, draft vocab OFF.
- Consequence: "Round phase (B1) FAIL +13.2%" and "Draft vocab active FAIL" are
  **baseline-staleness false alarms**, not code regressions.
- **Rule (2026-08-22):** baseline may only be re-captured after a full green run,
  with user sign-off on which gates move (D6 gate must flip to expect draft vocab OFF;
  B1/round gate must move to 42.7ms or the metric must be re-derived).
- **Rule (user, 2026-08-22):** never disable a feature to make a test pass — fix the
  feature or fix the test, and record it here.

## 4. Performance Trajectory (Qwen3.8-27B, 2× 5060 Ti)

| Date | State | plain t/s | pp t/s | MTP k=3 t/s | accept | notes |
|------|-------|-----------|--------|-------------|--------|-------|
| ~08-20 | early MTP | ~33 | ~32 | ~50s | ~50s% | first 3.8 runs |
| ~08-20 | 94 t/s baseline | 33.3 | 32 | **94** | ~85 | dirty tree, doc history |
| 08-21 | f571a622 dirty | 35.5 | 32 | 81 | **85.6%** | best recorded |
| 08-22 07:29 | clean tree regression | 35.5 | 32 | ~60 | **67.7%** | draft vocab root cause found |
| 08-22 17:56 | no-draft-vocab (14:25 bin) | 35.5 | 269.6 (M6 in tree) | 81.0 | 80.4% | kv_i8 fixed, TC-P pass |
| 08-22 18:11 | fresh build (S3 in) | 35.6 | 265.9 | 81.0 | 82.0% | A2 BROKEN (S3 bug) |
| 08-22 18:51 | S3 reverted (T=1 pf) | 35.5 | 32.1 | 80.95 | 82.0% | A2 pass; pp lost (M6 reverted) |
| (target) | v1 | ≥35 | ≥250 | **≥94** | **≥85%** | 70+ t/s hard floor |

## 5. Change Log (append-only, newest last)

- **2026-08-22** Renamed battery scripts → test suites (D14); run_ci.sh created:
  record state → build → verify suite → serve suite → combined verdict JSON (D13).
- **2026-08-22** Draft vocab disabled by default (D6) — accept 67.7→80.4.
- **2026-08-22** Battery results metadata: `_timestamp/_commit/_binary_time/_tree_dirty`
  + `checks[]`/`_verdict`/`_fails` (D15); latest.json + timestamped JSONs under
  `/home/intel/ninfer/results/`.
- **2026-08-22** A-2 RESOLVED: root cause = GDN recurrent state stored in FP16
  (`tp2_backend.cpp` `.recurrent_dtype`), quantized at the 225-token prefix boundary
  in the cached path but not in the fresh 232-token path → tiny drift → late argmax
  flip. Fix: `.recurrent_dtype = DType::FP32` (lossless across chunk boundary,
  zero perf cost, +0.5 GB state; all GDN kernels/state pool/replay support FP32).
  Prefix on/off now bit-identical. (The "non-associative parallel scan" premise was
  wrong — the GDN kernel is already a sequential ordered recurrence.)
- **2026-08-22** v1 DECLARED: verify suite fully green (0 fail/0 warn): pp 266.2 t/s,
  MTP k=3 80.76 t/s @ 82.0% acceptance, I8 pass, TC-P + A2 + sampling + prefix skip
  all pass. Baseline re-captured from the green run (B1 37.8→42.8 approved). Removed
  all remaining `[SAMP*]` debug prints. Committed + pushed to
  `chrisconcepcion/dual_5060_ti_ninfer` (branch mtp-perf).
- **2026-08-22** A-1 RESOLVED: root cause = S3 used ncclSum on ColumnN-sharded logits
  (disjoint vocab slices) → garbage argmax. Fix: temp=0→allreduce_argmax; temp>0→
  new TpGroup::allgather_local_bf16 + sample(full[248320]). Added `st.full_logits`.
  Removed 3 uncommitted [DEBUG] prints. M6 chunked prefill exonerated for A2.
- **2026-08-22** Doc 49 (this decision/trade-off log) created per user directive: no
  feature may be disabled to pass a test; every change + measured effect + trade must
  be recorded here (D15/D16 housekeeping).
- **2026-08-22** CUDA 13.1 build path established (D12); BUILD_TESTING=ON in CI (D11).
- **2026-08-22** M6 chunked prefill WIP verified: pp 32→265 t/s (D7).
- **2026-08-22** S3 sampling fix (allreduce_local_bf16 + sample) (D8, pre-A-1 discovery).
- **2026-08-21** H1 position fix (D9); one-shot reset desync fix (D10).
- **2026-08-21** I8 KV gate established, opt-in (D4); TC-P bit-identity policy (D5).
- **2026-08-20** Qwen3.8-27B becomes target (D16); draft head quant (W8G32) — later
  found zero-benefit, superseded by D6.
