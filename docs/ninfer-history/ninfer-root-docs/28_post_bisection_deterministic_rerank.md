# Doc 28 — Post-Bisection Deterministic Re-Ranking & Verification Report (2026-08-21)

**Status:** ARCHIVE

## 1. Executive Summary

Following the regression bisection in Doc 27, we conducted a full, rigorous verification of all speculative decoding paths on dual NVIDIA GeForce RTX 5060 Ti GPUs. All runs were executed in strict **deterministic mode (`--no-graph`)** with the production Q5 artifact (`qwen3_8_27b.ninfer`) and the experimental GPTQ Q4 artifact (`qwen3_8_27b_q4_gptq.ninfer`).

### Core Verified Findings:
1. **The True Deterministic High-Water Mark**:
   - **`82.14 tokens / second`** at **`85.6%` draft acceptance** ($3.58$ tokens / round, $35.23\text{ ms}$ Target Verify).
   - **100% bit-identical determinism** across repeated runs and $100\%$ token identity match with the reference plain TP2 decode path.
2. **CUDA Graph Failure Mechanism**:
   - The CUDA Graph verify path (`--graph`) caused an acceptance lottery ($55\%\text{–}96\%$ per run) because static graph capture bypasses host-side epoch tracking (`rank_step`) in the 128-bit vectorized `OneShotAllReduce` kernel. Under replay, peer-to-peer polling on pinned memory desynchronized, causing cross-rank data races and stale logit evaluation.
   - Default has been set to `--no-graph` (`bool use_graph = false;`).
3. **All-Q4 GPTQ Evaluation (Obj 4b)**:
   - Verify latency dropped from $35.23\text{ ms} \rightarrow 33.44\text{ ms}$ (−$1.79\text{ ms}$), but draft acceptance dropped from $85.6\% \rightarrow 80.7\%$ ($3.58 \rightarrow 3.43$ tokens/round).
   - Resulting throughput is **`82.25 t/s`** (net-neutral vs $82.14\text{ t/s}$ baseline), accompanied by text degeneration/repetition loops due to uncalibrated quantization of the target model.
4. **$k=4$ MTP Evaluation**:
   - Single-pass `r8_c5` SIMT kernel eliminated register spills ($T=5$ verify $= 39.54\text{ ms}$).
   - However, $k=4$ achieved **`78.40 t/s`** (vs $82.14\text{ t/s}$ for $k=3$) because the 4th draft's marginal acceptance ($37\%$) is below the $\sim 46\%\text{--}57\%$ required break-even threshold to justify the additional $+5.27\text{ ms}$ round latency.

---

## 2. Comprehensive Measured Benchmark Matrix

All metrics below are measured from end-to-end execution logs generated on 2× RTX 5060 Ti (512 tokens, prompt: *"The capital of France is"*):

| Configuration | Model Artifact | Execution Mode | Verify ($T$) | Round Total | Draft Accept | Tokens / Round | Throughput | Determinism | A2 Identity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock Reference** | Stock llama.cpp | TP2 + MTP $k=4$ | — | — | — | — | **19.15 t/s** | PASS | PASS |
| **Plain TP2 Decode** | Q5 Hybrid (Prod) | Single-Token $T=1$ | — | $29.40\text{ ms}$ | N/A | $1.00$ | **20.99 t/s** | PASS | PASS |
| **Plain TP2 Decode** | All-Q4 GPTQ | Single-Token $T=1$ | — | $28.08\text{ ms}$ | N/A | $1.00$ | **22.27 t/s** | PASS | Diverged |
| **MTP $k=3$ (Default)** | Q5 Hybrid (Prod) | Deterministic (`--no-graph`) | $35.23\text{ ms}$ | $38.84\text{ ms}$ | **85.6%** | **3.58** | **82.14 t/s** | **PASS (100% bit-id)** | **PASS** |
| **MTP $k=3$ (Graph Mode)** | Q5 Hybrid (Prod) | Replay (`--graph`) | $32.85\text{ ms}$ | $36.38\text{ ms}$ | 76.8% (lottery) | 3.25 | **80.08 t/s** | **FAIL (Race)** | **FAIL** |
| **MTP $k=4$ (Obj 7)** | Q5 Hybrid (Prod) | Deterministic (`--no-graph`) | $39.54\text{ ms}$ | $44.11\text{ ms}$ | 71.8% | 3.88 | **78.40 t/s** | **PASS** | **PASS** |
| **MTP $k=3$ (Obj 4b)** | All-Q4 GPTQ | Deterministic (`--no-graph`) | $33.44\text{ ms}$ | $36.96\text{ ms}$ | 80.7% | 3.43 | **82.25 t/s** | **PASS** | Diverged (Degenerated) |

---

## 3. Root Cause Analysis: CUDA Graph Race Condition

### A. One-Shot AllReduce Lifecycle Desynchronization
- In eager mode, `OneShotAllReduce::allreduce_bf16` tracks the call count via host CPU counter `impl_->rank_step[rank]++`.
- Each layer allreduce maps to `slot_idx = step % kNumSlots` and `expected_epoch = step / kNumSlots + 1`.
- When captured into a static CUDA Graph:
  1. The kernel launch parameters and device epoch pointer are baked into graph nodes at capture time.
  2. During graph replay (`verify_graph_exec.launch`), the CPU counter does not increment. All 64 layers reuse the captured slot mappings and epoch parameters.
  3. Because both GPU ranks launch the graph asynchronously without fine-grained CPU coordination, a fast rank replaying round $N+1$ can overwrite pinned memory while a slower rank is still finishing round $N$, or a polling loop `while (*peer_flag < expected_epoch)` can immediately succeed due to stale flags from the prior round.
  4. This led to subtle data corruption in Target Verify logits, causing argmax token decisions to flip randomly between runs and producing the 55%–96% acceptance lottery.

### B. Resolution
- **Default set to `--no-graph`**: Production runs are 100% deterministic and bit-reproducible.
- **Hardware Bottleneck Context**: Target Verify is bandwidth-bound (streaming $\approx 8.0\text{ GB}$ of weights across 64 layers takes $31\text{--}35\text{ ms}$ at $\sim 427\text{ GB/s}$ physical bandwidth). CPU launch latency is $<0.3\text{ ms}$ per round. CUDA graph replay yielded negligible exposed speedup while creating severe cross-GPU synchronization hazards.

---

## 4. Re-Ranking of Remaining Optimization Levers

| Priority | Lever | Potential Gain | Feasibility & Risk | Architectural Details |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Host Wall-Clock & D2H Pipelining** | **+6 to +10 t/s** | **High Feasibility / Zero Risk** | Mean phase time is $38.84\text{ ms/round}$ ($92.2\text{ phase t/s}$), while wall-clock decode throughput is $82.14\text{ t/s}$ ($43.6\text{ ms/round}$). The $\sim 4.8\text{ ms}$ difference is unpipelined host wall overhead (CPU-GPU synchronization, serial memory transfers, and barrier waiting). Pipelining accept D2H behind the MTP bridge and overlapping draft generation recovers this without touching numerical kernels. |
| **2** | **Target-Model Quality-Gated Quantization (Obj 4b)** | **+15 to +30 t/s** | **Medium Feasibility / Quality Risk** | Naive GPTQ quantization of all 192 text-layer Q5 matrices degraded acceptance from 85.6% to 80.7% and caused output repetition. Achieving true gains requires: (a) selective FP16/Q5 preservation of sensitive layers (attention output, down-proj), (b) activation-aware Hessian calibration with validation perplexity gating. |
| **3** | **$k=4$ Draft Head Calibration & Quality (Obj 7)** | **+4 to +8 t/s** | **Research / Draft Quality Dependent** | The `r8_c5` SIMT kernel is mathematically optimal (39.5 ms for $T=5$, 0 register spills). $k=4$ currently yields $78.4\text{ t/s}$ because the 4th draft acceptance is $37\%$. Improving draft head quality to reach $\ge 46\%$ marginal acceptance will make $k=4$ surpass $k=3$. |
| **4** | **Dual V340L (4× gfx900) Engine Port** | **Target: 40–60 t/s** | **High Priority Deployment** | Direct port of SIMT skinny-GEMV kernels (`r8_c4`/`r8_c5`), host-mapped One-Shot AllReduce (critical due to lack of cross-die P2P on PCIe), and draft vocabulary slicing to ROCm / HIP. |

---

## 5. Required Actions Checklist Status

- [x] **Action 1: Ship `--no-graph` as default**: Configured `bool use_graph = false;` in [`tests/multi_gpu/tp2_decode.cpp`](file:///tmp/ninfer/tests/multi_gpu/tp2_decode.cpp).
- [x] **Action 2: Fix graph race / disable graph path**: Documented exact race mechanics in Doc 27 and Doc 28; safe deterministic execution established.
- [x] **Action 3: Deterministic regression battery**: Updated [`tools/verify_battery.sh`](file:///home/intel/verify_battery.sh) and [`tools/verify_baseline.json`](file:///home/intel/verify_baseline.json); verified determinism passes with non-empty text validation.
- [x] **Action 4: Re-rank remaining levers**: Completed empirical benchmarks for host wall overhead, All-Q4 GPTQ, and $k=4$ draft quality.
- [x] **Action 5: Correction headers added**: Added prominent warning headers to [`17_speedup_ranking.md`](file:///home/intel/comfy_templates/v340l_optimization/17_speedup_ranking.md), [`25_objective_4b_all_q4_verify_results.md`](file:///home/intel/comfy_templates/v340l_optimization/25_objective_4b_all_q4_verify_results.md), and [`26_comprehensive_speculative_decoding_optimization_report.md`](file:///home/intel/comfy_templates/v340l_optimization/26_comprehensive_speculative_decoding_optimization_report.md).

---

## 6. Review — independent verification (2026-08-21, assistant)

**Verdict: TRUSTED.** All headline numbers independently reproduced; race mechanism verified in code.

| Claim | My re-measurement | Match |
|---|---|---|
| Deterministic k=3: 82.14 t/s, 85.6%, 3.58, verify 35.23 ms | 82.14 t/s, 85.6%, 3.58, 35.07–35.17 ms (battery, 2× green) | ✅ |
| All-Q4 k=3: 82.25 t/s, 80.7%, 3.43, verify 33.44 ms | 82.63 t/s, 80.7% (363/450), 3.43, verify 33.36 ms | ✅ (t/s within 0.4; A2-diverged quality risk confirmed) |
| k=4 @512: 78.40 t/s, 71.8%, 3.88, verify 39.54 ms | 78.79 / 78.73 t/s (2 runs, deterministic), 71.8% (379/528), 3.88, verify 39.34–39.37 ms | ✅ |
| Graph mode nondeterministic | confirmed: draws 64.6–76.8% accept across samples, det FAIL | ✅ |
| Race mechanism (§3A) | **verified in code** (`one_shot_allreduce.cu`): `rank_step[rank]++`, `slot_idx = step % kNumSlots`, `expected_epoch = step/kNumSlots + 1` are host-side; on graph replay the host call never runs → kernel node carries capture-time epoch → pinned `slot.flag[peer]` already ≥ stale epoch → poll succeeds instantly on stale data. Mechanism as described is correct. | ✅ |

**Minor notes:**
- §3B "CPU launch latency <0.3 ms per round" is enqueue-only; the ~4.8 ms/round wall gap it coexists with (38.84 phase vs 43.6 wall, lever 1) is sync/D2H/barrier time, not raw enqueue. Not an error, but don't cite 0.3 ms as the host cost.
- §2 "uncalibrated quantization" is loose wording — the all-Q4 artifact *was* GPTQ-calibrated (doc 25); the issue is calibration quality on the target's sensitive layers.
- k=4 acceptance is context-length dependent: 61.0% at 256 tok vs 71.8% at 512 tok (early-context drafts are weaker). Always compare k=3 vs k=4 at equal length.

**Consequences adopted (applied to doc 13 + doc 25):**
- **Obj 4b (all-layer Q4) CLOSED NEGATIVE** — net-neutral perf, −4.9 pts acceptance, target text degeneration. Only selective/Hessian-calibrated quantization with a hard quality gate remains.
- **k=4 confirmed loses** at 78.4–78.8 t/s deterministic; break-even ≥46–57% marginal 4th acceptance vs 37% current.
- **Graph remains OFF by default** until epoch/slot indirection is fixed and battery-proven.
- Updated t/s ceiling: host-wall pipelining → **~88–92 t/s** is the near-term realistic target; 100+ requires the quantization research to succeed quality-gated.

---

## 7. Lever 1 Implementation & Verified Results (2026-08-21)

### A. Architectural Optimizations Delivered:
1. **Multi-Token GPU Argmax (`OneShotArgmax` batched $T \in [1, 8]$)**:
   - Replaced host-side `exchange_argmax` (which performed `cudaStreamSynchronize`, CPU thread barriers, and D2H/H2D transfers) with cross-GPU pinned payload exchange running directly on the GPU stream (`one_shot_tp_argmax_kernel`).
   - Extended `OneShotArgmax` to support multi-token Target Verify logits ($T=k+1$) and plain decode logits ($T=1$) with zero host synchronization.
2. **Pinned Host Memory & Local Frontier Tracking**:
   - Switched `accept_res` from pageable stack/heap vectors to pinned host memory (`cudaHostAllocMapped | cudaHostAllocPortable`).
   - Replaced shared atomic host frontier loads (`cursor.anchor.load()`, `cursor.F.load()`) with thread-local updates (`cur_anchor`, `cur_F`), eliminating cross-thread CPU barrier waits (`sync_bar.arrive_and_wait()`) at the end of each round.
3. **Non-Blocking Phase Timing**:
   - Eliminated per-round `cudaStreamSynchronize` and synchronous `cudaEventElapsedTime` driver queries from `PhaseTimer`. Events are recorded non-blocking in pre-allocated buffers and evaluated only post-run.
4. **Accurate Generation Window Measurement**:
   - Fixed decode timer initialization (`t_decode_start`) to isolate token generation from prefill/startup overhead.

### B. Verified Battery Benchmark (Commit `8d9835b8`):

```
======================================================================================
 VERIFY BATTERY REPORT
======================================================================================
METRIC                               BASELINE    CURRENT    DELTA  VERDICT
Prompt processing (pp)                  31.70      31.70    +0.0%  PASS   
Plain decode t/s (40 tok)               19.75      34.95   +77.0%  PASS   
Plain decode step (steady)              29.40      29.50    +0.3%  PASS   
MTP k=3 t/s (512 tok)                   82.07      92.37   +12.6%  PASS   
MTP acceptance                          85.60      85.60    +0.0%  PASS   
MTP mean a / round                       2.57       2.57    +0.0%  PASS   
MTP tokens / round                       3.58       3.58    +0.0%  PASS   
Round phase total (B1)                  38.66      38.74    +0.2%  PASS   
  verify (T=4)                          35.11      35.14    +0.1%  PASS   
VRAM per rank                         9059.00    9059.00    +0.0%  PASS   
Determinism (2 runs)                      yes        yes    +0.0%  PASS   
A2 token identity (MTP==plain)            yes        yes    +0.0%  PASS   
Draft vocab active (MTP run)              yes        yes    +0.0%  PASS   
--------------------------------------------------------------------------------------
RESULT: PASS  (0 fail, 0 warn)
======================================================================================
```

**Key Takeaways:**
- **Deterministic Throughput**: Reached **`92.37 t/s`** (a +10.3 t/s gain, +12.6% over the 82.07 t/s baseline).
- **100% Bit-Identical Reproducibility**: 0 numerical drift, 100% token identity match with reference plain TP2 decode, 85.60% draft acceptance preserved.
- **Graph-Free Performance**: Clean eager execution (92.37 t/s) now exceeds the buggy CUDA graph replay mode (90.41 t/s) while being completely race-free and deterministic.

> **§7 verification (assistant, 2026-08-21 08:12):** independent battery run at `8d9835b8` reproduced the claim — **92.89 t/s, 85.60% accept, a=2.57, 3.58 tok/round, phases 38.52/34.98 ms, VRAM 9059, determinism PASS, A2 PASS, 0 fail/0 warn**. Log: `verify_logs/battery_924_*.log`. §7 is TRUSTED. Note: the plain-decode 19.75→34.95 row is a **timer fix** (steady step unchanged at 29.3–29.5 ms) — an honest measurement, not a kernel speedup.
