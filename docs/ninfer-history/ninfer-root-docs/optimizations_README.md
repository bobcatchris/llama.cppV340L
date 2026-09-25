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

Reusable artifact: `tests/multi_gpu/data/qwen38_draft_vocab_ids.json` (40,960-id draft-vocab slice for the MTP drafter, from the 3090 repo — same model; validate coverage before use, see doc 13 Objective 1).
