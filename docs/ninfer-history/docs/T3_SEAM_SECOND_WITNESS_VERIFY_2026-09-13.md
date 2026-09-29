# T3 (A)-seam — second-witness verification of the landed sweep (agent3, live session)

**Session:** agent3, pi `01a0984f-cd62` (qwen3.8-flash), hub line `pi-dual_5060_ti_ninfer-1060980`.
**Scope:** I did **not** author the commit. The atomic commit `007bfa27` (+ comment-only `4c9dbac6`)
was landed by the *other* session sharing the `agent3` name (`01a0984f-4a6c`, glm-5.3) and merged to
`amd/main@35bdf211` via `3baaf367`. This file is my **independent re-verification** of the landed
bytes, plus one proven residual. Per tonight's law: *re-verification must vary the instrument.*

## 1. What I verified, and how (instruments differ from the original author's)

| # | Claim on the board | My instrument | Result |
|---|---|---|---|
| 1 | Guard is live | Minimal synthetic HIP TU calling `ldmatrix_x4(…, smem_addr(&s[0]))` | **fires** — `error: call to deleted function 'ldmatrix_x4'` |
| 2 | Guard is live, both arms typed | `git show 007bfa27:mma.cuh` | 4× `= delete` in `#if defined(__HIP__)`; `ldm_addr_t` defined **on both arms** (`:36` ULL, `:106` unsigned) — the classic half-landed CUDA break is absent |
| 3 | Fleet invariant 0 | Own mutation-testable checker (§3) | **128 sites, 0 violations**; injected-mutation → rc=1 (the check *can* fail) |
| 4 | Negative cell passes | Ran `results/amd/run_t3_ldmatrix_negative_cell.sh` myself | **exit 0** — Arm A: 4 deleted-function diags; Arm B (guard stripped): compiles clean → **non-vacuity proven** |
| 5 | Sweep compiler-green on the crash sites | Real `hipcc` `-c` device pass, gfx900, my own flags | **8 TUs rc=0, 0 deleted diags**: q5, q4, q4-swiglu, w8-rowsplit, vision, bidirectional, gqa_prefill, gqa_kvarn |
| 6 | FRAGCELL-512 extension sound | `-x hip -c` compile of `t3_b4_frag_parity_cell.cu` | rc=0. **Device legs W1/W2 NOT run** — no grant held; rides a later dev2 window |
| 7 | Single-writer boundary | `git diff --name-only` filtered | **0 edits** to agent4's residuals (kernels.cu / plan.cpp / tp_engine / HipSources) |

## 2. q5 provenance note
q5:360/:367 were the *proof site* in the debrief. Verified: raw `smem_addr` there at `010dc213`;
`NINFER_LDM_ADDR` at HEAD; TU compiles rc=0. The specific fault that opened G-AMD-17d is closed
by the compiler, on my own run, not on anyone's word.

## 3. Counting rules — the board's numbers are ONE measurement, not three
`NINFER_LDM_ADDR` counts disagreed (89 / 106 / 115). Named rules reconcile them:
- occurrences incl. macro definitions + comments: **115**
- call-site occurrences, `mma.cuh` excluded: **112** (96 ldmatrix-arg wraps + 16 `lane_base`/`sbase` assignments)
- distinct lines: **112**; files touched: **39**
- ldmatrix call sites fleet-wide (ground truth, `grep -o` = python spans): **128**
Every figure was correct; each was counting a different population. *Numbers ship with counting rules.*

## 4. PROVEN RESIDUAL — the guard is blind one level up (32/128 = 25% of sites)
`*_swz_addr` helpers (`gqa_prefill_`, `bidirectional_gqa_`, `vision_attention_`) now take
`ldm_addr_t` in **and** out. The deleted overloads sit on `ldmatrix_*` only, so a **raw 32-bit
token routed through the helper** into an emulation is still parse-green:

```c
ldmatrix_x4(..., gqa_prefill_swz_addr(smem_addr(p), 0u, 0u, 0u));   // rc=0, NO diagnostic
```

Measured: guard absent → `rc=0, deleted=False`; guard added → `rc=1, deleted=True`.
So 96 sites are compile-refused and **32 are protected by a remembered type** — the exact weakness
that produced G-AMD-17d. Today it is **inert** (0 live violations, §1 row 3); it is a
*future-regression* surface.

**Fix (authored-in-concept, NOT applied — main is merged, needs coord ruling):** deleted
`unsigned`-first-param overloads on the three helpers, inside the `#if defined(__HIP__)` arm only.
On the CUDA arm `ldm_addr_t == unsigned`, so an unguarded deleted overload is a **duplicate
signature = CUDA build break** — must be arm-scoped. Same rule the sweep already follows.

## 5. Honest caveats
- CUDA-arm text is **by-construction** preserved (`NINFER_LDM_ADDR(p)` → `smem_addr(p)` on non-HIP),
  **not** guard-proven: this is an AMD host (rocm-smi → 4× V340), no `nvcc`, so no CUDA compile
  was possible for me either.
- The 32 helper-routed sites include families that are HIP-unreachable today (header-blocked or
  arm-guarded) — reachability was never the point; the type seam is.
- 6 residual `smem_addr(` calls are correct (mbarrier / TMA `.shared::cta` PTX operands, 32-bit
  by hardware contract) — deliberately untouched.

## 6. Instrument worth keeping
`/tmp/t3_fleet_check.py` — counts ldmatrix spans, asserts 0 raw-token spans, and has a `--mutate`
mode that MUST make it fail. Two earlier greps of mine under-reached (6-TU, then 24-site) because
ERE `_t?` is not PY `_t?`: `ldmatrix_x[24]_t\(` requires the literal `_t` and silently drops the
103 non-transposed calls. A bounded listing that misses 80% of its population "passed" twice before
the count-vs-ground-truth check caught it. **Always assert the enumerated count equals an
independently-measured ground truth, then mutation-test the checker.**

— agent3 (01a0984f-cd62), second witness. Chain state: sweep merged at `amd/main@35bdf211`;
17e stamped to agent4; FRAGCELL W1/W2 device legs + B4 await a dev2 window.
