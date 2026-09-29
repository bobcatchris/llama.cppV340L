# STACK DECISION BRIEF — ninfer vs llama.cpp on this box (everything measured 2026-09-20)

For the smaller-model stack decision. Every number below was measured on THIS box this
session (receipts: results/amd/llamacpp/FAIR_MATRIX_2k_llamacpp.txt,
results/amd/census/FAIR_MATRIX_2k_ninfer.txt + BF16_SUSTAINED_2k.txt, PLOG-087..096).

## The measured matrix at 2k context (27B-class model, all 4 V340 dies unless noted)

| config | prefill | decode | notes |
|---|---|---|---|
| llama.cpp TP2 layer, q4_0 KV | **103.2** | 8.95 | best prefill measured at 2k |
| llama.cpp TP2 layer, bf16 KV | 100.9 | 8.67 | |
| llama.cpp TP4 layer, q4_0 KV | 90.1 | 8.98 | |
| llama.cpp TP4 layer, bf16 KV | 90.4 | 8.87 | KV width changes nothing |
| llama.cpp TP4, q4_0 + MTP (default ~3) | 87.6 | **11.93** | acceptance 0.65 |
| llama.cpp TP4, bf16 + MTP d2 / d3 | 86.4 | 11.02 / 11.5 | knob: --spec-draft-n-max |
| llama.cpp TP2 + MTP | REFUSED | — | ~151 MiB short at 10k ctx |
| llama.cpp row (tensor) split | no tests produced (upstream partial support) | | |
| ninfer bf16 + draft-2 | 75.8–101.0 | **16.9 med (16.3–23.5)** | acceptance 73.5%, cap 23,808 |
| ninfer bf16 + draft-3 | 64.9–69.5 | 14.2–16.6 | acceptance 61.4% — draft-3 loses |
| ninfer k4v4 + draft-2 (promoted) | 64.3–67.5 | 10.4–10.6 | cap 65,536 |
| ninfer k4v4 + draft-3 | 64.5–65.6 | 10.8–11.9 | |

Context-scaling (ninfer bf16+d2): 36.3 tok/s sustained at ~0 ctx → 16.9–23.5 at 2k →
decode falls with context bytes. The historical "40 t/s" was the near-zero-context number.

## What the numbers say

1. **Prefill: the stacks are comparable at 2k** — llama.cpp TP2 (103.2) sits at the top of
   ninfer's bf16 band (75.8–101.0) and above ninfer kvarn (64–67.5). The earlier "ninfer
   3.5× faster" claim was geometry-mixed and is RETRACTED (PLOG-095).
2. **Decode: ninfer bf16+d2 wins (16.9 med vs llama's best 11.93)**; llama.cpp draft-mtp
   beats ninfer *kvarn* decode (11.93 vs 10.5) — kvarn pays ~35-40% decode for capacity.
3. **llama.cpp decode is flat ~8.9 no-spec** across TP2/TP4 and q4_0/bf16 KV — it does not
   scale with world count or KV width on gfx900; MTP is its only decode lever (+12-33%).
4. **MTP: both stacks have it.** llama.cpp: --spec-type draft-mtp, --spec-draft-n-max knob
   (2 vs 3: 11.02 vs 11.5 — d3 mildly better there). ninfer: 0.88 acceptance/2.59 tok-round
   at d2 beats llama's 0.65/2.95; ninfer d3 collapses (0.61) while llama d3 holds —
   different MTP implementations, different acceptance behavior.
5. **Capacity (27B, per die):** kvarn k4v4 65,536 tok; bf16 23,808 (ws512) / 36,352 (ws96
   era); llama.cpp GGUF 12.17 GiB splits 2 dies at 10k ctx (TP2+MTP refused; TP4+MTP fine).
   Historical max bf16 boot on TP4: 98,944 (era manifest, PLOG-088).

## The decision criteria (what each stack maximizes)

- **Decode speed at matched context:** ninfer bf16+d2.
- **Prefill at 2k:** llama.cpp TP2 (by a nose; within thermal noise of ninfer bf16).
- **Long context:** ninfer kvarn (65,536 verified 8/8 needle-exact; llama.cpp at 12.17 GiB
  weights + KV runs out of 8 GB dies faster per token — its 10k test fit, larger untested).
- **Operational simplicity:** llama.cpp (one binary, GGUF ecosystem, no bake pipeline).
- **What we control:** ninfer (every kernel, the ledger, the thermal levers) vs llama.cpp
  (upstream's roadmap, our patches: the FP8 gate; the gfx900 path is unattended upstream).

## Open items before the call

- Model identity (the GGUF tested is the 27B; if the smaller model differs, re-run §matrix
  — the runner is staged: /media/chris/ssd128/llamacpp/matrix.sh + fair_llama2c.sh pattern).
- Long-context llama.cpp numbers (only 10k/2k measured; 40-80k needs the GGUF split across
  4 dies + smaller KV, untested).
- The 09-09 NVIDIA-line reference (2,835 pp / 85-101 tg at 40-80k, TP2 16 GB ranks) — the
  performance target worth naming out loud, on hardware we do not have on this box.
