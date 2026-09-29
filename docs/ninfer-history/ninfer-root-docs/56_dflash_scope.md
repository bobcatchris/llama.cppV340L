# 56 — DFlash scoping for Qwen3.8-27B TP2

Status: SCOPE, **Path B unblocked 2026-08-24** (official DFlash2 GGUF artifacts + llama.cpp PR #27342 reference impl are public; see Path B section). No code yet. Register: none yet.

## What DFlash is

A speculative-drafting backend that replaces MTP's 1-layer AR draft chain with a
small separate drafter. Two variants are in play:

- **In-tree DFlash** (`SpeculativeBackend::DFlash`): 6-layer local drafter
  (5 local layers, hidden = target hidden, intermediate 6144, 32q/8kv heads ×
  128) that reads features from 8 target layers and drafts up to 15 tokens
  autoregressively. Full mechanism already exists in our shared qwen3_6
  runtime: `dflash_context.{h,_impl.h}`, feature sink
  (`text_prefill_impl.h`), rewrite checkpoints, CUDA-graph profiles
  (`Variant::dflash_graph_profiles`).
- **DFlash2** (cross-repo catalog item 3.4, from the qwen38-3090 machine):
  separate **1.92B 5-layer NON-autoregressive block drafter**, GPTQ W4A16
  (1.19 GB artifact). Predicts a full 7-token block in one parallel pass from
  target hidden states at layers 5/19/33/47/61 + path selector over 16
  candidates/slot. Measured there: **4.80 tok/round vs MTP 3.58 → 122–132 t/s**
  on a single RTX 3090 running the same Qwen3.8-27B weights.

## Current state for OUR model

`src/targets/qwen3_6_27b/impl/config.h`:

```cpp
struct DFlashConfig { static constexpr bool supported = false; ... };
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
```

**Qwen3.8-27B has no DFlash drafter weights and the config is zeroed out.**
The in-tree backend only works for Qwen3.6-35B-A3B (MoE), which ships with a
DFlash artifact upstream. `--spec dflash` on our model would fail at layout
(`kMaximumDFlashDraftTokens == 0` check in `layouts_impl.h`).

## Two paths, one is blocked

### Path A — in-tree DFlash drafter for 27B: BLOCKED

Requires a 6-layer DFlash draft model derived/trained specifically for
Qwen3.8-27B. Upstream (Neroued) ships it only for their artifacts; no training
recipe exists in either repo. **No artifact, no procedure → not actionable.**

### Path B — port the DFlash2 block drafter: ACTIONABLE (unblocked 2026-08-24)

**Artifact dependency RESOLVED** — official public artifacts for our exact
target now exist, no 3090 machine needed:

- Weights: `incoai/Qwen3.8-27B-DFlash2` (BF16 GGUF) + mirrors
  `z-lab/Qwen3.8-27B-DFlash2-GGUF` (Q4_K_M 1.1 GB / Q8_0 2.0 GB / BF16 3.8 GB)
  and `HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF` (535 MiB, mixed 2–3-bit,
  imatrix-calibrated). All Apache 2.0.
- Reference implementation: llama.cpp **PR #27342** ("spec : add DFlash2
  support", still OPEN as of 2026-08-24 — code lives on the PR branch / forks,
  MIT) + HermiG's `fix/dflash2-tool-tg-collapse` branch: adds `p_min`
  (1deefcca39), **grammar fix** (fe3e373e3f — upstream DFlash2 applies the
  tool-call PEG grammar over the full vocab at every verify position, ~90 ms/
  step CPU collapse → tool-call generation drops to ~¼ speed; fixed via lazy
  single-token grammar acceptance while a grammar is active), and
  `--spec-draft-ubatch-size` (9207d48915) to cap drafter VRAM.
- Measured baselines (their table, single 24 GB GPU, Qwen3.8-27B target):
  acceptance 0.694/0.595/0.510 at n_max 2/3/4 (Q4_K_M); ~103 tok/s at
  n_max=3. DFlash2 is **lossless** (greedy matches target exactly) — speed
  only, no quality risk.
- **The grammar fix matters for us**: we serve an agent (pi) with tool calls;
  porting the unfixed acceptance path would cripple exactly our workload.
  Port from the fixed branch, not the PR head.

**GGUF as the DFlash head — yes as weight source, no as runtime format.**
GGUF is just a tensor container: load the official GGUF (Q4_K_M or BF16) and
run it through our conversion toolchain into a .ninfer draft artifact (our
GPTQ pipeline already does imatrix-calibrated low-bit — we can match or beat
the Q2_K_S-MIX recipe). Direct GGUF loading in ninfer-serve is not worth it:
the runtime is built on .ninfer artifacts + materialize_tp sharding; a second
load path for one ≤1.1 GB model violates the single-format policy.

Work items (updated):

1. ~~Obtain artifact from 3090 machine~~ → **download official GGUF**
   (`z-lab` or `incoai`) + clone HermiG branch as port reference.
2. **CUDA port** (like KVarN P2): reimplement the 5-layer non-AR block pass +
   path selector as our ops; the in-tree DFlash plumbing (feature sink,
   checkpoints, graph profiles) is a starting skeleton but shapes differ
   (non-AR block ≠ AR local drafter).
3. **TP2 wiring**: drafter reads full hidden vectors from 5 target layers →
   either replicate the 1.92B drafter on both ranks (~1.2 GB/rank, simple) or
   shard it (allreduce of layer outputs per round — avoid). Replication is the
   sane default at 1.2 GB.
4. **Memory**: ~1.2–1.5 GB/rank (weights + draft-window KV). Current 80k/100k
   config has ~1 GB/rank headroom → **tight; needs kv-capacity trimmed or
   KVarN P2 landed first** (KVarN frees ~1.9 GB/rank). At the 200k config it
   does not fit at all without KVarN.
5. **Validation**: battery T1–T14 + acceptance vs the 82.0% MTP baseline gate;
   determinism (T10) — non-AR path selector must be deterministic.

### Expected gain (honest estimate)

3090 numbers are single-GPU where verify was the bottleneck. Our TP2 verify is
already faster (~75 t/s at 3.5–4.0 tok/round). If DFlash2 delivers 4.8
tok/round here, decode ≈ **95–110 t/s (+25–45%)**, assuming the replicated
drafter's per-round cost (5 layers, one parallel pass) stays below MTP's AR
chain. Not guaranteed — measure at P-equivalent of T9.

**Updated anchor (2026-08-24, from the public artifact card):** on a single 24 GB GPU against the same target, DFlash2 Q4_K_M measured ~92/103/108 tok/s at n_max 2/3/4 (acceptance 0.694/0.595/0.510, draft len 2.39/2.78/3.04). That is a real measured ceiling to compare our port against — not just the 3090 projection. Their target ran on one GPU; our TP2 verify is faster, so the relative gain may be smaller than +45% but the absolute tok/s floor is now known.

## Adjacent lever (no artifact needed)

**Lookup drafting** (`LOOKUP=1`, `req.use_lookup` already wired in
`tp_engine.cpp`; catalog item 2): scans prompt + generated history for suffix
matches, drafts verbatim tokens for free. +30–140% on copy/code/RAG workloads
(up to 381 t/s on the 3090). Zero new weights; worth an A/B on real agent
traffic before any DFlash effort.

## Recommendation

- **Now:** nothing (MTP k=3 is fine; lookup A/B is the cheap experiment).
- **After KVarN P2** (VRAM headroom exists): if the 3090 artifact is
  obtainable, open a work order for Path B with the 5 items above as phases.
- **Path A:** drop unless upstream ships a 27B DFlash artifact.
