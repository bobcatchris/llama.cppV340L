# Doc 32 — Next Work Order (for the agent)
> **⚠ SUPERSEDED (2026-08-21):** Active work order is now **Doc 34 — TP2 Serve Integration Plan** (`docs/optimizations/34_tp2_serve_integration_plan.md`). WI-1/WI-2/WI-3 are closed; WI-4/WI-5 are deprioritized behind Doc 34. 

**Date:** 2026-08-21
**Status:** Authoritative. Supersedes any roadmap in docs 30/31 §5 that conflicts with this.
**Current HEAD:** `b939e95b` · **Verified baseline:** 94.18 t/s MTP k=3 (battery `20260821_085515`, `/home/intel/verify_baseline.json`)

---

## 0. Verified state (do not re-derive, do not re-run to "confirm")

| Metric | Value | Source |
|---|---|---|
| MTP k=3 | **94.18 t/s** (deterministic, `--no-graph`) | battery 20260821_085515, PASS |
| Acceptance | 85.6% (a=2.569, 3.58 tok/round) | same |
| Round / verify | 37.99 ms / 34.79 ms (91.6% of round) | same |
| Plain decode | 35.42 t/s (29.0 ms/step) | same |
| pp | 32.00 t/s (225 tok) | same |
| VRAM | 9059 MB/GPU | same |
| Gates | determinism PASS, A2 PASS, draft vocab 40960 PASS | same |

Phase 1 (1A rolling GDN ring, 1B fused OneShot AR+AXPY, 1C vectorized strided unpack) is
**done, verified, and merged** (`7bc4c41f`, `518adffc`, `e31d5bce`). Do not redo it.

## 1. Rules (non-negotiable)

1. **Every perf claim must come from a full `verify_battery.sh` run**, not a single manual run.
   The battery auto-updates `/home/intel/verify_baseline.json` only on full PASS. Logs land in
   `/home/intel/verify_logs/` — the report log + raw per-run logs are the record of truth.
2. **No CUDA graph anywhere.** Graph mode is default-off and currently *hangs* (exit 124).
   Work item 2 deletes it. Do not debug it, do not "fix" it, do not re-enable it.
3. **Quality gate before any quantization claim:** perplexity (fixed prompt set), GSM8K-style
   fixed eval, acceptance ≥ 85.0% (−0.6 pts floor vs 85.6), A2 token identity, battery PASS.
   Gate fails ⇒ close the objective as negative (like doc 25 did for bulk INT4). A closed
   negative is a valid, valued result.
4. **Never cite a number you did not measure in this session.** Stale anchors live in this
   doc's §0 and in doc 29 rev 2 (V340L).
5. **Hardware facts (measured, do not "correct" from datasheets):** 5060 Ti = 36 SMs,
   16 GB **GDDR7**, 448 GB/s theory / ~427 GB/s measured, sm_120a, CUDA ≥ 13.1.
   V340L = 4 × gfx900 dies, 400 GB/s/die + PCIe 3.0 x8/card **user-confirmed specs (not yet
   measured on-system)** — see doc 29 rev 2.

## 2. Work items, in order

### WI-1 (highest value): Objective 4c — selective Hessian-calibrated INT4, quality-gated

**CLOSED NEGATIVE — 2026-08-21. See doc 33.** Value_z scope (15% of GEMV bytes):
−4.1% t/s net (acceptance −5.7 pts vs verify −0.8%). GPTQ ≈ RTN at 1024-token
calibration. Acceptance cost concentrates, gain doesn't — full scope extrapolates
worse (doc 25 all-layer: −11.8 pts for +13.2% verify ≈ break-even). Both attempt
budget slots used. Infrastructure (cal-dump hook, strict converter, **fixed** GPTQ
quantizer — two pre-existing bugs found and repaired) is committed and reusable.

<details><summary>original spec</summary>

- **Why:** verify GEMV (34.79 ms, 91.6% of round) is the wall. Bulk Q4 proved bulk
  quantization is dead (doc 25). Layer-selective is the only untested quant path.
- **Scope:**
  - Keep W8/F16 on sensitive matrices: embeddings, attention Q/K/V/O, layers 0–1 and 62–63,
    MLP `down_proj`.
  - Hessian-calibrate Q4G64 on the bulk: MLP `gate_up_proj` + intermediate GDN projections
    (~60% of weight bytes).
  - Use the GPTQ-calibration machinery pattern from `syv-ai/qwen38-27b-rtx3090` (technique
    only — that repo is vLLM/Python, nothing ports verbatim).
- **DoD:** quality gate of §1.3 passes **and** verify ≤ ~28 ms **and** battery PASS with
  t/s ≥ 105. Expected success case: **~115–120 t/s**.
- **Failure path:** gate fails ⇒ close negative, document which matrix class broke acceptance,
  keep W8 default. Stop. Do not iterate beyond 2 calibration attempts without asking.

</details>

### WI-2 (30 min, do first or immediately after): delete the CUDA graph path

- Remove `--graph` / `--no-graph` flags, capture/replay code, and graph references from
  `tp2_decode.cpp` and one-shot AR/argmax. Eager is the only path; keep the current default
  behavior bit-exact (A2 must still pass).
- **DoD:** grep finds no graph code; battery PASS with 94.18 ± 1%.
- Why: it currently hangs (exit 124 in the last battery); every remaining line is risk.

### WI-3: LOOKUP=1 context-lookup drafting (zero-risk, workload-value)

- Suffix-match drafts from prompt + generated history (n-gram table), injected into the draft
  buffer before MTP forward. Lossless — target verify still gates every token.
- **DoD:** add a **code-file-copy prompt case** to the battery (e.g. prompt = first 200 tokens
  of a source file, expected continuation = next tokens of the same file). On that case,
  drafts from lookup must raise tok/round materially vs 3.58 with zero rejected-token
  corruption. On the standard "capital of France" case, nothing may regress (≥ 94 t/s).
- Value is for code/RAG/copy workloads, not the benchmark prompt.

### WI-4: prefill GEMM tiling

- pp is 32 t/s — weakest metric, compute-bound. Target: large-M GEMM tiling at 30–45% of
  fp16 peak. **DoD:** pp ≥ 45 t/s on the battery's 225-token prompt, no decode regressions.
- Also raises confidence in the V340L pp estimate (doc 29, weakest row).

### WI-5 (light start only): V340L port preparation

- Safe now: HIP port of the SIMT r8c4/c5 kernels (near 1:1 from CUDA), host-mapped one-shot
  AR shim, 2-level TP4 reduce design (intra-card 2-die → cross-card 2-die, doc 29 §5).
- **Not yet:** any tuning, any V340L perf number, any commit that claims V340L throughput.
  The card is in transit; arrival-day microbench (BW + GEMV efficiency) decides the plan.
  If measured per-die BW ≪ 400 GB/s, re-rank before continuing.

### PARKED (do not start)

- **DFlash2 block drafter** — 1.92B non-AR drafter, own artifact, large architecture.
  Revisit only after WI-1's outcome is known.
- **k=4 draft quality** — needs marginal 4th-draft acceptance ≥ 46–57% (current 37%);
  model-side work.
- **AR batching (2–4 layers)** — WI-1B's fusion already took most of the win; revisit only
  if a profile shows exposed AR > 0.5 ms again.

## 3. Where things live

- Battery: `/home/intel/verify_battery.sh` · baseline: `/home/intel/verify_baseline.json`
- Logs: `/home/intel/verify_logs/` (report + raw per-run)
- Repo: `chrisconcepcion/dual_5060_ti_ninfer` (`mtp-perf` branch) — push docs to
  `docs/optimizations/`, never to upstream
- V340L source of truth: doc 29 rev 2 (commit ~135 t/s MTP, range 120–145, plain ~51;
  single gating unknown = GEMV efficiency on gfx900)
- Optimization catalog: doc 12 (technique provenance) · speedup history: doc 17 (corrected)
