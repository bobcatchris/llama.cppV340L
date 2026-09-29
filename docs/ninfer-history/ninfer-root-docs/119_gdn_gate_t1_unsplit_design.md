# 119 — GDN-gate T=1 MmaUnsplitT1 design (docs/103 Option A)

**Status:** OPEN — design for review (plan owner drafting the work order, docs/118).
**Author:** TP2 agent (Agent 1). **Reviewer:** user / plan owner.
**Companions:** docs/103 (remediation work order), docs/105 (attribution),
results/105_attribution.md (probe evidence), docs/117 (pipeline board).

---

## 1. Recap / motivation

Attribution (probes P0–P3, results/105_attribution.md) is now conclusive:

> **`1d3a0533` is the ENTIRE docs/105 regression.** Old GDN routing (P2 revert)
> restores ALL gates: plain 35.42 / verify 35.49 ms / mtp 79.18 (baseline
> 35.40 / 35.24 / 80.71). P1 (anchor) exonerated. P3 (checksum env) off by default.

`1d3a0533` changed the 27-B GDN control-gate routing from a **T-dependent split**
(Gemv T=1, SmallTSplit10 T=2–8, MmaUnsplit T≥9) to a **single `MmaUnsplit`** for all
token counts. It was required for D-21 (the old split accumulated the gate ~1 ulp
differently at T=1 vs T>1, drifting the recurrent state and flipping greedy argmax).
The cost: `MmaUnsplit` is slower at small T (T=1 decode, T=2–8 verify) than the
split routes it replaced.

**Tension:** the D-21-safe `MmaUnsplit` is slower at small T; the D-21-unsafe split
routes are faster. docs/103 Option A aims to recover small-T perf **while keeping
bit-identical accumulation**.

## 2. The D-21 invariant (must not regress)

T=1 decode must remain byte-identical to T≥2 (verify) for the same token. Concretely,
for a given token, the gate projection `g`/`beta` must be bit-identical whether the
projection runs at cols=1 or cols≥2. docs/103's acceptance gate for this change is the
full D-21 battery (P3 k=1 / k=2, H5 1024-token, 6-prompt) staying byte-identical.
**This is a pure routing/schedule swap — any output change is a defect, full stop.**

## 3. Why `MmaUnsplit` is slow at T=1 (code-level)

`bf16_gdn_gating_proj_gemm_mma_kernel` (src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh)
tiles **`BlockN = Geometry::kBlockN = 128`** tokens per CTA:

- At decode T=1, `token0 = blockIdx.x*128 = 0`, and only **token 0** is valid. Tokens
  1–127 are zero-padded in shared (`store_vec(dst, make_int4(0,0,0,0))`, Predicated
  variant since `1 % 128 != 0`). The kernel still runs the **full 128-token-wide MMA**
  and 128-token x-tile to produce **one** token.
- Launch is `grid = (ceil(1/128)=1, kBf16GdnHeads/kBf16GdnBlockM = 48/16 = 3, 1)` = **3 CTAs**,
  each `Warps*32 = 256` threads, on a 170-SM device.

So the waste is twofold: (a) 127/128 of the MMA + x-tile work is empty, (b) only 3 CTAs
means low SM occupancy for the gate at T=1.

## 4. The bit-identity key

`mma_bf16` computes `D[m,n] += Σ_k A[m,k]·B[k,n]` per `m16n8k16` tile. Each output
**column** `n` depends **only** on `B[*,n]` (that token's x) and `A` (the weight) — it is
**independent** of every other token in the tile. Therefore:

> token 0's fp32 accumulation is **identical** whether the MMA tile is 8, 16, 64, or 128
> tokens wide — **by construction**, provided the kernel keeps the same `mma_bf16`
> instruction, the same fragment layout, and the same K-order for that column.

This is what makes `MmaUnsplitT1` safe: it can shrink the token tile (cutting the empty
tensor-core work) and keep token 0 bit-identical to the `MmaUnsplit` T≥2 path **by
design, not by luck**.

## 5. Design

### 5.1 Schedule

- Add `Bf16GdnGatingScheduleId::MmaUnsplitT1` to
  `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h`.
- Route (k27Routes): `{{1, 8} → MmaUnsplitT1 (narrow tile), {9, kAnyCols} → MmaUnsplit-128}`.
  - **MUST cover cols 1–8, not just cols==1.** The verify gate (≤35.6 ms, currently
    37.64) is driven by T=4 (cols 2–8) having gone `SmallTSplit10→MmaUnsplit` in
    `1d3a0533`. A T=1-only fix would leave verify at ~37.6 and FAIL the verify gate.
  - `MmaUnsplitT1` legal at `cols == 1..8` (covers plain T=1 AND verify T=2–8).
  - `candidate_is_legal` for the 27 B model: `MmaUnsplitT1` →
    `problem.cols >= 1 && problem.cols <= 8`.
- Named "gdn_gating_proj.bf16.mma.unsplit_t1". `schedule_uses_mma` → true (so the token
  variant is `Predicated` at T=1, since `1 % 128 != 0`). `schedule_split_k` → 1.

### 5.2 Kernel

Two candidate shapes (decide by A/B measurement):

- **A1 — narrow-tile reuse (minimal code):** launch the *same*
  `bf16_gdn_gating_proj_gemm_mma_kernel<Bf16Gdn27Geometry, 1, ..., Stages>` but with a
  narrower `BlockN` that is exactly one MMA N-tile for the warp count
  (`kWarpN = BlockN/Warps ≥ 8`, so `BlockN=64` with `Warps=8` ⇒ `kWarpN=8`, `kNFragments=1`).
  Token 0's fragment position is unchanged ⇒ bit-identical. Cuts empty-token MMA work
  ~2× (128→64) and lets grid.x stay 1.
- **A2 — dedicated narrow kernel:** a small, single-token kernel that reads the weight
  row(s) coalesced (Gemv-like memory pattern) and accumulates via `mma_bf16`
  (tensor-core, bit-identical), avoiding the 128-token tile entirely.

**Recommendation:** start with A1 (smallest diff, reuses the bit-identity guarantee), and
measure. If A1 does not recover enough of the ~7.5% plain / +2.2 ms verify gap, implement
A2, then Option B (\u00a76.3).

### 5.3 Why bit-identity holds for A1

With `BlockN=64`, token 0 maps to `col0 = token0 + warp*kWarpN + ni*8 + 2*lid`. For
`warp=0, ni=0, lid=0` ⇒ `col0=0` — the same fragment position as in the 128-tile path.
The K-loop (`it` over `kTilesPerSplit`) and `ldmatrix_x2/x4` fragment loads for that column
are identical. `mma_bf16` therefore produces the same `a_acc`/`b_acc` for token 0.

**Extension to cols 2–8:** the same column-independence argument applies. Each *real*
token column `n` (0..cols-1) depends only on `B[*,n]` + `A`, and its fragment position in
the narrow (BlockN=64) tile is unchanged from the 128-tile path (`col0` for the first
`kNFragments` = same). The zero-padding of token columns ≥cols does not touch the real
columns, so every real token is bit-identical to the 128-tile `MmaUnsplit` path.

## 6. Expected performance + risk

- The small-T GDN gate is bandwidth-bound on the **weight read** (≈1 MB across 48 layers per
  step). Both `MmaUnsplit` and `Gemv` read the same weight; the 7.5% plain / +2.2 ms verify
  delta is from the empty-token MMA work + low CTA occupancy, which A1/A2 target.
- **Verify target for cols 2–8:** the A/B must compare the cols 2–8 narrow tile against the
  OLD `SmallTSplit10` numbers (verify 35.49 ms) — the target is to **beat or match**
  ≤35.6 ms, not merely "recover part of the gap".
- **Risk (documented in docs/103 §7):** the MMA (`m16n8k16`) tile shape **caps CTA
  parallelism at small T** — an unsplit MMA can only launch ≈3–6 CTAs (48 heads / 16-row
  tile; +2 if a/b split). So `MmaUnsplitT1` (SplitK=1, bit-identical) may **not** reach
  GEMV-class occupancy and may not hit the verify gate.
- **Escalation ladder:** A1 (narrow tile) → if it can't hit BOTH gates (plain ≥35.0,
  verify ≤35.6) → A2 (dedicated kernel for the shortfall) → Option B (make the batched
  accumulation bit-identical to the T=1 GEMV; more invasive, re-runs D-21 both ways). The
  A/B measurement decides which is needed.

## 7. Verification plan (in order)

1. **Build** from working base (the `cudaFuncSetAttribute`
   re-issue fix `aa576027` + `MmaUnsplit` routing).
2. **Byte-identity** (MUST pass): D-21 battery — P3 k=1/k=2, H5 1024-token, 6-prompt —
   byte-identical to plain (the same battery the `1d3a0533` fix was verified against).
3. **Perf gate** (decode_test, bf16, same harness as CI):
   - plain ≥ **35.0**, verify ≤ **35.6 ms**, mtp ≥ **79.0**.
4. **Full CI** → `PASS` with all WARNs cleared.
5. **Record** `results/103_gdn_gate_t1_remediation.md` with A/B tables, provenance
   (commit + sha256 + mtime), model identity.

## 8. Non-goals

- No change to D-21 byte-identity (the invariant we protect).
- No re-introduction of the old split (`GemvPairedRows` + `SmallTSplit10`) — that is the
  D-21 bug.
- No KVarN / packed-kernel work (docs/83 lever-1 stays parked).
- No serve-path work; decode_test only.
- No baseline re-baseline without a written justification.

## 9. Files touched (expected)

- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h` — enum `MmaUnsplitT1`.
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp` — routing, name,
  `candidate_is_legal`, `schedule_uses_mma`, `schedule_split_k`, `execute_resolved`.
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` — new
  `bf16_gdn_gating_proj_mma_unsplit_t1_launch` (A1: narrow-tile `launch_bf16_prefill_mma`;
  A2: dedicated kernel).
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh` — only for A2.

## 11. CHOSEN PATH — Option B' (GEMV unification for T≤8)

**Status: CHOSEN (plan-owner decision; supersedes §5/§6 A1/A2).** The D-21 invariant is
"T=1 gate bit-identical to T≥2 gate **for the same token**" — NOT "everything must equal
MmaUnsplit". `1d3a0533` satisfied it by making everything slow; the old split routes were
fast but the two GEMV-class kernels (`GemvPairedRows` T=1, `SmallTSplit10` T=2-8) accumulated
1 ulp differently. The root fix shares ONE per-column reduction across T=1 and T=2-8:

- **T=1:** `GemvPairedRows` (unchanged) — the proven-fast single-token GEMV (35.42 t/s).
- **T=2-8:** `SmallTGemv` (new) — a batched GEMV that loops over the tokens, computing each
  token column with the **SHARED per-column reduction code** (same exact FMA order over K,
  same striding, same block reduction as `GemvPairedRows`). Weight read amortized over the
  token batch. This is the perf class of the old `SmallTSplit10` (verify 35.49 ms).
- **T≥9:** `MmaUnsplit-128` (unchanged, split-K=1).
- k27Routes: `{{1,1}→GemvPairedRows, {2,8}→SmallTGemv, {9,kAnyCols}→MmaUnsplit}`.
  - `candidate_is_legal`: `SmallTGemv` → 27-model && 2≤cols≤8. `schedule_uses_mma=false`,
    `schedule_split_k=1`.

**Why D-21 holds by construction:** T=1 and T=2-8 run the SAME per-column fp32 FMA reduction
(identical K order, identical striding, identical block reduce) → identical gate bits →
identical recurrent-state trajectory. Prefill (T≥9) is `MmaUnsplit` in both the decode and
verify MTP runs, so it cancels. The T=1 kernel is UNCHANGED, so plain-decode output is
unchanged by definition → **no "both directions" rework needed**. If `SmallTGemv` at T=4
comes in >35.4 ms (unexpected — weight read is amortized over 4 tokens, strictly better
than T=1), we re-discuss.

**Perf anchors (empirical, not guesses):**
- plain: `GemvPairedRows` unchanged → 35.42 t/s.
- verify (T=4): batched GEMV, weight read amortized over 4 tokens → ≥35.49, target ≤35.4 ms.
- mtp: inherits → ≥80.7 t/s.

**Gates:** build → ctest unit (`test_gdn_gating_proj`, extended with a cols 2-8 `SmallTGemv`
case) → D-21 battery byte-identical → plain ≥35.4 / verify ≤35.4 / mtp ≥80.7 → full CI →
`results/103_gdn_gate_t1_remediation.md`.

**Files touched (Option B'):**
- `bf16_gdn_gating_proj_plan.h` — enum `SmallTGemv`.
- `bf16_gdn_gating_proj_plan.cpp` — routing, name, `candidate_is_legal` (2≤cols≤8),
  `schedule_uses_mma=false`, `schedule_split_k=1`, `execute_resolved`.
- `bf16_gdn_gating_proj_kernels.cu` — shared per-column reduction + new
  `bf16_gdn_gating_smallt_gemv` kernel + `bf16_gdn_gating_proj_smallt_gemv_launch`.
- `bf16_gdn_gating_proj_kernels.h` — `smallt_gemv_launch` declaration.
