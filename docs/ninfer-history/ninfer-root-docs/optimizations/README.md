# Optimization research — Qwen-27B class, multi-GPU (5060 Ti dev, V340L target)
Research + implementation log for getting NInfer MTP to 70+ t/s. Numbered chronologically; read from `12` onward for the current state.

| Doc | What |
|---|---|
| 01–04 | v100-skinny findings → V340L (gfx900) projections + NInfer analysis |
| 05 | 5060 Ti dev plan |
| 06–07 | TP2 layer design + PP2→TP2 implementation plan |
| 08–09 | MTP acceptance bug → fix |
| 10 | MTP perf hitlist (original worklist) |
| 11 | **Hitlist results** — measured 53.77 t/s, verify=38.26ms floor, the honest gap |
| 12 | **Cross-repo optimization catalog** (v100-skinny / NInfer / qwen38-27b-rtx3090) — reusable across GPUs |
| 13 | **Future objectives** — the prioritized NInfer worklist to reach 70+ t/s |

| 14–16 | One-shot AR, draft-head INT4 (null), CUDA graph, verify profiling |
| 17 | **Speedup ranking** — all speedups ranked with V340L portability notes (⚠ corrected in doc 27) |
| 18–24 | Per-objective results (AR, INT4, graph, argmax, k=4, T=5, greedy) |
| 25–26 | All-Q4 verify GEMV + comprehensive report (⚠ both carry correction banners — stale baseline) |
| 27 | **Regression bisection + CUDA-graph nondeterminism** — the authoritative correction; read this last |
| 29 | **2× V340L final estimates (rev 2)** — 400 GB/s/die + PCIe 3.0 x8 confirmed; commit ~135 t/s MTP k=3 (120–145), plain ~51; only gating unknown = GEMV efficiency |

Reusable artifact: `tests/multi_gpu/data/qwen38_draft_vocab_ids.json` (40,960-id draft-vocab slice for the MTP drafter, from the 3090 repo — same model; validate coverage before use, see doc 13 Objective 1).
| 32 | **Next work order (authoritative)** — WI-1 selective Hessian INT4 (gated) → WI-2 delete graph path → WI-3 LOOKUP drafts → WI-4 pp tiling → WI-5 V340L prep (light); parked: DFlash2, k=4 quality |
| 44 | **Radiance MXFP4 optimization survey** (2×R9700 Reddit build, 280 tok/s) — by-technique catalog with code locations both sides, measured wins, low-effort drop-in shortlist (§9), phase-gate interaction (§10). Brought over from `wo/kvarn-prefill` (`f679568e`) |
| 45 | **Scope: radiance decode launch-gap stack (the 22 ms win)** — what `db9dba6` actually was; NInfer already has 2 of its 3 components (one-shot AR+residual, no activation quants); remaining items W0–W5 (baseline census, GDN extract-trio fusion, MTP small-op folds, skinny GEMV audit, dynamic draft depth / in-chain confidence break — §6) |
