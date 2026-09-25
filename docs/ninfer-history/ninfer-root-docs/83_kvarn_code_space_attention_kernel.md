# 83 — KVarN done right: code-space attention (no wall, no materialize)

**Status:** LIVE plan of record for the KVarN attention implementation
(directive 2026-08-27). **Supersedes docs/82 Phase A/B** (the wall-removal +
T=1 acc-FWHT framing) and the docs/78 lineage. docs/82 §0–§1 (the diagnosis:
O(N) dequant transforms eat the bandwidth win) is **kept and is the foundation
of this doc**; docs/82's *execution* (A2 = PV-only, on the 0.7% T=1 kernel;
wall kept as a fallback) is abandoned. Owner: agent. QA gate: maintainer.
Worktree `wo/kvarn-hold`, continues from `7aa5b4be`.

## 0. Post-mortem (what went wrong, so we don't repeat it)

KVarN k4v2 is a good idea (4-bit K / 2-bit V, ~2.7× KV compression) that was
implemented badly. The failure modes, all real and measured:

1. **The hot loop dequantizes codes to bf16.** Attention is done by
   materializing K/V back to bf16 and running bf16 MMA. The V-side needs an
   O(N·D) Hadamard (FWHT) per page — 37% of pass-1 — that is too slow to run
   inline every decode round.
2. **The wall was a band-aid for (1).** To dodge the per-round Hadamard, we
   pre-materialized a **bf16** staged shadow. But the shadow is bf16, so for
   any context that fits in it (≤40.4k = 632 pages) KVarN stores *bf16 shadow +
   k4v2 codes* = **more** VRAM than plain bf16. The compression only pays off
   beyond the wall. The wall defeated the entire purpose of the cache.
3. **Four code paths, none of them the real one.** shadow-bf16-flash (in-wall),
   packed T=1 (draft, 0.7% of decode), materialize+bf16-flash (T=4 MTP verify,
   21%), and the legacy split path. Each patched toward its own gate.
4. **Gates got re-scoped to accommodate regressions** (G1 184→104, G2 option-(a),
   GB @250k 69→30) instead of the kernel being made fast. A2 optimized the
   0.7% path and moved end-to-end decode 57.3→57.6.

The wall was a non-starter. The fix is not to tune around it — it is to make
the dequant **disappear from the hot loop** so there is nothing to shadow.

## 1. Root cause, stated once

**We dequantize in the O(N) loop when we should do attention directly on the
codes.** Both halves of attention have an exact, cheap code-space form:

- **QK** contracts over the channel dim `d`; the K codes are already in the
  Hadamard domain (`codes_K = H(K)/16`), and `H` is orthogonal & symmetric, so
  `Q·K = H(Q)·H(K)/D`. Rotate **Q once** (O(1) per query row), then dot against
  the raw K codes with the affine scale folded per-group. **No K dequant.**
- **PV** contracts over the key dim `k`; `H` acts on the channel dim, so it
  commutes with the contraction: `P·H(V) = H(P·V)`. Accumulate `P·codes_V`
  (rotated), then apply **one** `H` to the accumulator in the epilogue
  (`H²=D·I`). **No V dequant.** This is the deferred acc-FWHT, already proven
  **bit-exact on CPU** in `tests/kvarn_acc_fwht_cpu_ref.cpp` (A1, `436a6618`).

Do both and the decode kernel reads 0.75 B/elem of codes and does *no* O(N)
transform — bandwidth-bound by construction, which is the llama.cpp/NVFP4
target docs/82 §0 named. The wall, the materialize kernel, and the bf16 decode
flash all become dead code.

## 2. Target architecture: one code-space attention kernel

A single tiled decode/verify kernel, `gqa_attention_kvarn_codespace_kernel`,
serving **T = 1..6** (single-token decode *and* MTP verify), grid
`(kv_head, page_split)`, each page's codes loaded to smem **once** and reused
for all T query rows:

- **Q preprocessing (O(1) per query row, outside the page loop):**
  - `q_rot = H(Q)` (256-pt Hadamard, warp-local) — enables QK on K codes.
  - fold the per-channel `vc` into Q: `q' = q_rot * vc` — removes the V-style
    per-channel scale from the K inner loop.
  - carry per-64-group `sr`, `zp` for the K codes.
- **QK on codes (no K dequant):** unpack K 4-bit codes → int8 lanes, `m16n8k32.s8`
  MMA of `q'` against the codes, then per-group rescale by `sr` and subtract the
  `zp` rank-1 correction (`Σ_d q'_d` per group, precomputed). This is the exact
  pattern the int8 decode kernel already uses for its K path.
- **PV on codes (no V dequant):** unpack V 2-bit codes → bf16/int8, accumulate
  `acc' = Σ p·codes_V` (rotated domain), then **one** `kvarn_mma_acc_fwht(acc)`
  in the epilogue (A1-validated, banked in `gqa_attention_kvarn_mma.cuh`).
- **Merge:** fixed-order split-K partial merge (reuse the packed kernel's
  determinism-preserving merge); `÷l` and merge commute with `H`.
- **Tail:** stage tail K/V in the **rotated** domain to match (helper
  `kvarn_mma_stage_tail_v_rotated` already banked; add the K-side rotated stage).

Numerics: code-space QK is **≥ as accurate** as the current path — it uses the
same 4-bit codes but skips the extra bf16 rounding of dequantized K. PV matches
the A1/A2 result (fp32-truth rel_l2 ≈ 0.0044, closer than the bf16-dequant
reference). No new precision risk beyond what k4v2 already accepts.

## 3. Reusable assets (this is not from-scratch)

| asset | where | use |
|---|---|---|
| s8-MMA-on-codes QK pattern | `gqa_attention_decode_i8.cuh` (78.8 t/s MTP, proven) | QK-on-K-codes |
| deferred acc-FWHT, **bit-exact CPU-validated** | `tests/kvarn_acc_fwht_cpu_ref.cpp` (A1) | PV epilogue |
| `kvarn_mma_acc_fwht` + `stage_tail_v_rotated` | `gqa_attention_kvarn_mma.cuh` (banked `de8b6439`) | PV epilogue + tail |
| packed page-load / split-merge / cp.async | `gqa_attention_kvarn_decode_packed.inc` | kernel skeleton |
| tail A/B sweep {0,1,8,63} + realistic scales | `bench_kvarn_attention.cu` | regression guard |

## 4. Staged plan (mature process: validate → build → profile → collapse)

Each step commits; gates are measured against a **bandwidth-bound target**, not
re-scoped to fit a regression.

- **M1 — Code-space design validation (CPU-only, no GPU, server stays up).**
  **DONE — PASS** (`tests/kvarn_codespace_qk_cpu_ref.cpp`,
  `results/doc83_m1_codespace_qk_result.md`): QK folding identity 9.1e-08,
  end-to-end code-space attention 2.8e-07. Both halves validated on CPU.
  Extend the A1 harness to the **full** code-space attention: (a) QK-on-codes
  with `vc`-folded rotated Q + per-group `sr`/`zp` correction vs the dequant
  reference, over random K codes; (b) the PV deferred-acc-FWHT (already done);
  (c) end-to-end `softmax(QK)·V` code-space vs dequant-space. **Gate:** matches
  the dequant path to <1e-6 modulo the accepted k4v2 quant error; the `zp`/`vc`
  folding is the F7-class landmine — prove it on CPU before any kernel edit.
- **M2 — Implement `gqa_attention_kvarn_codespace_kernel` (T=1..6).** Behind
  `NINFER_KVARN_DECODE=codespace` for A/B against the current paths. Determinism:
  fixed-order split/merge preserved. **Gate:** `test_kvarn_gqa` green (incl. F5 +
  tail), tail A/B sweep green, numerics ≤ the current dequant path vs fp32 truth.
  **Two scoping findings (2026-08-27, `results/doc83_step2_hazard_and_m3_design.md`)**
  fix the shape of M2:
  - **Register wall:** the packed T=1 kernel keeps online-softmax `acc` in
    registers (128/lane for 16 rows); T=4 = 64×256 = 512 regs/lane, infeasible.
    So M2 must use **global-memory split-K partials** (the `small_t` structure),
    NOT a tiled copy of the packed kernel. This is why `kKvarnSmallTMax=1` and
    why D-19 step 1d's direct-read T>1 route was latency-bound.
  - **Materialize hazard:** the legacy 4/2-bit `materialize` kernel is
    byte-identical-protected (`flash.cuh:158-163`: neutral body edits empirically
    shift MTP acceptance on prefix-restored requests). So the "defer the Hadamard
    inside materialize" shortcut is **abandoned**; M2 is a **new** kernel that
    *replaces* materialize+small_t for decode, reading codes directly.
  The full dataflow (rotated-Q × affine-K-codes, rotated-V accumulate, reduce-side
  `Ĥ`, global partials) is specified in `results/doc83_step2_hazard_and_m3_design.md`.
  - **STATUS (2026-08-28): M2 SHIPPED via a simpler route than the from-scratch
    codespace kernel.** Option B (token-block the existing packed kernel to serve
    T=2..6 verify, `grid.y` splits tokens into ≤16-row blocks) + A2 (defer the V
    Hadamard to the epilogue `kvarn_mma_acc_fwht`) delivers @40k 57.3→66.7 and
    @250k 30→40 (beats the removed shadow), 6/6 greedy outputs identical, tests
    green. See `results/doc83_optionB_A2_win.md`. The from-scratch codespace kernel
    (int8 s8-MMA on raw codes) remains a future upgrade, not a prerequisite.
- **M3 — Profile vs the bandwidth-bound target (nsys, exclusive GPU).** Confirm
  the decode window has **no** `materialize`, **no** bf16 decode flash, **no**
  shadow reads — decode is the code-space kernel. Measure decode_guard
  @10k/40k/250k packed-only. **Gate:** code-space decode **≥ the shadow numbers
  it replaces** (≥71 @40k, and materially >30 @250k since the Hadamard tax is
  gone). This is the gate the wall could never pass — here it is the *point*.
- **M4 — Collapse the paths.** Route ALL decode (T=1..6) to the code-space
  kernel; delete the wall (shadow bind/staging/aliasing, `kKvarnStagedBudgetBytes`,
  revert 98b5395e), the `materialize` decode route, the bf16 decode flash, and the
  split path. **One** KVarN decode path. **Gate:** full CI (`run_ci.sh --full`),
  byte-determinism A/B vs pre-M4, VRAM **−1342 MiB/rank** at the 250k cap, live
  multi-turn long-prefill request served through code-space decode.

## 5. Success criteria (the honest ones)

- **Bandwidth-bound:** code-space decode kernel reads ~0.75 B/elem with no O(N)
  transform; effective code read ≥150 GB/s (GA) at 40k **and** 250k, T=1..6.
- **No regression on removal:** decode ≥ shadow at 40k, and better than the
  current 30 t/s at 250k (the Hadamard tax is gone). Wall removal is a **win**,
  not a tradeoff — that is the whole difference from docs/82.
- **VRAM:** −1342 MiB/rank, KVarN actually saving memory at every context.
- **One path:** shadow/materialize/split/bf16-flash-for-decode all deleted.

## 6. Anti-patterns this plan forbids

- No wall, no shadow, no "keep the hybrid as a fallback."
- No re-scoping a gate downward to accommodate a regression; fix the kernel.
- No optimizing a path that profiling shows is <5% of decode (the A2 lesson).
- No new code path without deleting an old one by M4.
- No kernel edit before its mapping is proven on CPU (the F7/A1 lesson).
- No editing the byte-identical-protected legacy `materialize` kernel (docs/69 B2
  MTP-acceptance hazard) — build a new path that replaces it instead.

## 7. Supersession

- **docs/82** — §0–§1 diagnosis kept; Phase A (T=1 acc-FWHT) and Phase B (wall
  removal via gate) **superseded by this doc's M1–M4**. A1's CPU ref and the
  banked `acc_fwht`/`stage_tail_v_rotated` helpers carry forward as M1/M2 inputs.
- **docs/78** — remains superseded (history only).
- **docs/82 §4 partial-pass / option-(a)** — void; there is no wall to keep.
