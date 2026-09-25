# 118 — DFlash implementation plan (Path B: DFlash2 block drafter)

**Status:** DESIGN — 2026-08-30. This is the intensive plan for the DFlash
implementation (the next item after int4 + KVarN cache types). GATED behind
int4 + KVarN cache types (docs/117) + the merge kernel (docs/104 Phase 2-3)
being done. See docs/56 (DFlash scoping) for the full context.

## 0. The constraint (user, 2026-08-30)
The DFlash implementation is the next item after int4 + KVarN cache types
(docs/117). It is GATED behind:
1. The merge kernel (docs/104 Phase 2-3) being done (Agent 1's lane).
2. The int4 + KVarN cache types (docs/117) being done (the VRAM headroom for
   the drafter).

## 1. What DFlash is (from docs/56)
A speculative-drafting backend that replaces MTP's 1-layer AR draft chain with
a small separate drafter. Two variants:
- **In-tree DFlash** (`SpeculativeBackend::DFlash`): 6-layer local drafter —
  BLOCKED (no 27B artifact, no training recipe).
- **DFlash2** (Path B — ACTIONABLE): separate **1.92B 5-layer NON-autoregressive
  block drafter**, GPTQ W4A16 (1.19 GB artifact). Predicts a full 7-token block
  in one parallel pass from target hidden states at layers 5/19/33/47/61 + path
  selector over 16 candidates/slot. Measured: **4.80 tok/round vs MTP 3.58 →
  122–132 t/s** on a single RTX 3090. DFlash2 is **lossless** (greedy matches
  target exactly) — speed only, no quality risk.

## 2. The artifact dependency (RESOLVED — docs/56)
- **Weights:** `incoai/Qwen3.8-27B-DFlash2` (BF16 GGUF) + mirrors `z-lab/Qwen3.8-27B-DFlash2-GGUF`
  (Q4_K_M 1.1 GB / Q8_0 2.0 GB / BF16 3.8 GB) + `HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF`
  (535 MiB, mixed 2–3-bit, imatrix-calibrated). All Apache 2.0.
- **Reference implementation:** llama.cpp **PR #27342** ("spec : add DFlash2
  support") + HermiG's `fix/dflash2-tool-tg-collapse` branch: adds `p_min`
  (1deefcca39), **grammar fix** (fe3e373e3f — lazy single-token grammar
  acceptance while a grammar is active), and `--spec-draft-ubatch-size`
  (9207d48915) to cap drafter VRAM.
- **The grammar fix matters for us:** we serve an agent (pi) with tool calls;
  porting the unfixed acceptance path would cripple exactly our workload. Port
  from the fixed branch, not the PR head.
- **GGUF as the DFlash head — yes as weight source, no as runtime format:** load
  the official GGUF (Q4_K_M or BF16) and run it through our conversion toolchain
  into a .ninfer draft artifact (our GPTQ pipeline already does imatrix-
  calibrated low-bit).

## 3. The 5 work items (from docs/56)
1. **Download the official GGUF** (`z-lab` or `incoai`) + clone HermiG branch
   as port reference.
2. **CUDA port** (like KVarN P2): reimplement the 5-layer non-AR block pass +
   path selector as our ops; the in-tree DFlash plumbing (feature sink,
   checkpoints, graph profiles) is a starting skeleton but shapes differ
   (non-AR block ≠ AR local drafter).
3. **TP2 wiring:** drafter reads full hidden vectors from 5 target layers →
   replicate the 1.92B drafter on both ranks (~1.2 GB/rank, simple).
4. **Memory:** ~1.2–1.5 GB/rank (weights + draft-window KV). Tight; needs
   kv-capacity trimmed or KVarN P2 landed first (KVarN frees ~1.9 GB/rank).
5. **Validation:** battery T1–T14 + acceptance vs the 82.0% MTP baseline gate;
   determinism (T10) — non-AR path selector must be deterministic.

## 4. The expected gain (honest estimate — docs/56)
- Our TP2 verify is already faster (~75 t/s at 3.5–4.0 tok/round). If DFlash2
  delivers 4.8 tok/round here, decode ≈ **95–110 t/s (+25–45%)**, assuming the
  replicated drafter's per-round cost (5 layers, one parallel pass) stays below
  MTP's AR chain. Not guaranteed — measure at P-equivalent of T9.
- **Updated anchor (2026-08-24):** on a single 24 GB GPU against the same target,
  DFlash2 Q4_K_M measured ~92/103/108 tok/s at n_max 2/3/4 (acceptance
  0.694/0.595/0.510, draft len 2.39/2.78/3.04). That is a real measured ceiling
  to compare our port against.

## 5. The effort estimate
- Work item 1 (download the artifact + clone the reference): ~0.5 day.
- Work item 2 (CUDA port — the 5-layer non-AR block pass + path selector): ~1-2
  weeks (the biggest item — the non-AR block pass + the path selector).
- Work item 3 (TP2 wiring — replicate the drafter on both ranks): ~2-3 days.
- Work item 4 (memory — kv-capacity trim + KVarN P2): ~1-2 days (if KVarN P2 is
  already landed).
- Work item 5 (validation — battery T1–T14 + acceptance + determinism): ~2-3
  days.
- **Total: ~3-5 weeks** (the CUDA port is the biggest item).

## 6. The adjacent lever (no artifact needed — docs/56)
- **Lookup drafting** (`LOOKUP=1`, `req.use_lookup` already wired in
  `tp_engine.cpp`): scans prompt + generated history for suffix matches, drafts
  verbatim tokens for free. +30–140% on copy/code/RAG workloads (up to 381 t/s
  on the 3090). Zero new weights; worth an A/B on real agent traffic before any
  DFlash effort.

## 7. The recommendation (docs/56)
- **Now:** nothing (MTP k=3 is fine; lookup A/B is the cheap experiment).
- **After KVarN P2** (VRAM headroom exists): open a work order for Path B with
  the 5 items above as phases.
- **Path A:** drop unless upstream ships a 27B DFlash artifact.
