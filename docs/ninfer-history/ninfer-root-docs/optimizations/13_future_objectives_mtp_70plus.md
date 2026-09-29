# Future Objectives — NInfer MTP → 70+ t/s (2× 5060 Ti, Qwen3.8-27B)
**For the next agent.** This is the concrete, prioritized worklist derived from the cross-repo optimization catalog (doc 12). It turns the *techniques* found in `qwen38-27b-rtx3090` (and `v100-skinny`) into **NInfer C++/CUDA tasks**.

**State (updated 2026-08-21, verified by battery + doc 27 bisection + doc 28 Lever 1):** MTP k=3 = **92.37 t/s** deterministic (`--no-graph`, eager execution) at **85.60% acceptance** (a=2.57, 3.58 tok/round); verify 35.1 ms, round 38.7 ms. Plain decode **34.95 t/s** (timer fix — steady step unchanged at 29.3–29.5 ms; the old 21.7 figure was startup-polluted). **DoD IS MET and SURPASSED.** Lever 1 (Host Wall-Clock & D2H Pipelining via batched `OneShotArgmax`, pinned memory, and non-blocking timers) delivered +10.3 t/s gain with 100% bit-identical determinism and 0 quality risk. Objective 2 CLOSED negative (doc 16); all-Q4 layer quantization CLOSED negative (doc 28: net-neutral 82.25 t/s at −4.9 pts acceptance + text degeneration); k=4 measured deterministic at 78.4 t/s (loses; needs ≥46–57% marginal 4th acceptance vs current 37%, doc 28). Next: **quality-gated selective quantization → k=4 draft quality → Dual V340L port**.
Housekeeping: the `--graph` path is still in-tree and still racy (lottery draws, doc 27) — fix the epoch/slot indirection or delete the path (doc 27 action 2).

**The reusable artifact from the 3090 repo** (same model) is copied to the code repo at **`tests/multi_gpu/data/qwen38_draft_vocab_ids.json`** — 40,960 token ids (max 248076 < our 248320 vocab), frequency-shaped, counted over the model's own outputs. See Objective 1.

---

## Work order (do in this sequence; each gates the next)

### Objective 1 — Draft vocabulary for the MTP drafter  *(BIGGEST lever, no target-quality risk)*
**Source:** qwen38-3090 `prepare/build_draft_vocab.py` + `draft_vocab_ids.json` + `qwen3_5-mtp-draft-vocab.patch` (catalog 3.1).
**Why:** the MTP head re-scores the full 248,320-row `output_head` ~4×/round (11.25 ms). A drafter that scores only 40,960 rows is ~6× cheaper per draft → the MTP head drops from ~12 ms to ~3 ms → round ~41 ms → **70+ t/s**.
**Spec (NInfer):**
1. Load `tests/multi_gpu/data/qwen38_draft_vocab_ids.json` (40,960 full-vocab ids).
2. Build a **40,960-row slice** of `text/output_head` (same W8G32 row-split format) → store as a new object `mtp/draft_output_head` (~221 MB vs 1.35 GB; also **saves ~1.1 GB VRAM**). Keep the id→row map on host.
3. Wire `mtp_propose` (d0) and the AR chain (d1..d3) to score **only the 40,960 draft rows**, argmax within the draft head, then **map the draft row back to the full-vocab id**.
4. If the target's greedy sample for a position is **outside** the draft vocab, that draft is a **guaranteed rejection** — the target's own token is used (spec decode stays exact). No correctness change to the target.
5. The **verify** still uses the full `output_head` (target path unchanged).
**Accept / gate:** A2 token-identity vs plain MUST still pass (the draft set only shrinks proposals). Then measure steady `--tokens 512 --mtp 3`.
**⚠️ First, validate the id list against OUR model:** run a coverage check — sample ~1M tokens of our model's own greedy output and confirm the list covers ≥ 95% (regenerate/count if not; the method is in `build_draft_vocab.py`). Don't assume cross-checkpoint identity.
**Expected:** MTP head 12.46 → ~3 ms; round ~41–43 ms → **~68–72 t/s**. This alone likely crosses 70.

### Objective 2 — Split-KV attention for the T=4 verify  — **CLOSED, NEGATIVE RESULT** (see doc 16)
**2026-08-20 review:** nsys profiling (doc 16) shows GQA attention is **0.2 ms = 0.5%** of the 36.56 ms verify phase. The real verify walls are GEMV 23.5 ms (64%), NCCL AllReduce 7.5 ms (20%), host launch 3.8 ms (10%). Split-KV has ≤0.2 ms to win; the 3090 patch targeted a T=1 single-GPU config that does not apply to our T-batched TP2 verify. Objective 2 is closed. Superseded by Objective 2a below.

### Objective 2a — Replace NCCL AllReduce with 2-rank P2P one-shot AR in the MTP round  *(NEW, top lever after review)*
**Source:** our own `TpGroup` P2P send/recv (validated, microbench 6–9 µs/call, `tests/multi_gpu/bench_ar.cpp`) + doc 16 finding (128 NCCL RING_LL AR/round × 58.2 µs = 7.5 ms, exposed/serialized).
**Note:** P1's "direct non-blocking allreduce" (f184738f) is still `ncclAllReduce` — it only removed the host worker-thread barrier. The 58 µs is GPU-side RING_LL protocol latency for 10 KB messages over PHB.
**Spec:**
1. Microbench a 2-rank one-shot AR at the exact 5120-elem bf16 size using the existing P2P primitives (bidirectional exchange + local sum).
2. If ≤ ~30 µs/call: add `TpGroup::allreduce_p2p_local_bf16` and route the MTP round's 128 ARs through it (plain decode too).
3. Gate: A2 token-identity + determinism; re-run battery.
**Expected:** −3–6 ms/round → **~95–105 t/s**. No quality risk; reuses existing plumbing.
**Source:** qwen38-3090 `patches/spec-decode-attn.patch` (catalog 4.1/4.2).
**Why:** the verify step has k+1=4 query rows; GQA decode may be running one block per (kv-head, head-group) without splitting the KV sequence across SMs → idle SMs. On the 3090 this was 57 µs→23 µs/layer; our verify is 75% of the round, so any win compounds.
**Spec (NInfer):**
1. Profile the current GQA decode kernel on the T=4 verify (nsys / the B1 harness) to confirm the SM-utilization gap.
2. Add a **split-KV** path: NUM_SEGMENTS blocks per (request, kv-head), each over a KV range, online-softmax partial, then a combine. Add **query-row tiling** (BLOCK_M) so T can grow past 4 later.
3. **Partial buffers must be fixed-size** (allocated once) — they are captured by the CUDA graph (Objective 3).
**Gate:** A2 token-identity (attention output must be bit-identical within tolerance — same math, different tiling). Then re-measure verify phase.
**Expected:** verify 38.26 → ~20–28 ms (context-dependent). Needed to go past ~77 and to make k=4/5 viable.

### Objective 3 — CUDA-graph the MTP round  *(DONE but BLOCKED — race, docs 20/27/28)*
**2026-08-21 status:** graph implemented (doc 20) but is **nondeterministic** (doc 27 bisection): host-side `rank_step`/`expected_epoch` in the one-shot AR is baked at capture; on replay the pinned `slot.flag` is already ≥ the stale epoch so peer polling succeeds instantly on stale data → verify logits corrupt → acceptance lottery 55–96%. Default is now `--no-graph` (driver flip, commit 268ef04f). **Unblock = epoch/slot indirection through stable device memory (no baked ints, no stale-flag ambiguity), then re-prove determinism in the battery.**

### Objective 3 (original spec) — CUDA-graph the MTP round  *(kills the launch/D2H tail)*
**Source:** v100-skinny `native_round_design.md` (catalog 3.7) + our `DecodeGraphExecutable::update`.
**Why:** the round is fixed-shape (verify T=4, align T=4, AR 3 steps, rebase); only buffer *contents* change. Capturing it as one graph removes the ~4 ms launch + per-op D2H tail (the gap between the 50.76 ms phase sum and the measured round).
**Spec:** after Objectives 1–2 land (so the round shape + buffer addresses are stable), capture the whole MTP round; `update` the input buffers each round. Ensure all workspace/partials (incl. split-KV from Obj 2 and the draft head) are pre-allocated and never reallocated.
**Gate:** A2 token-identity; measure the round end-to-end.
**Expected:** −3–5 ms/round → pushes 70 → ~75.

### Objective 4 — GPTQ int4 `output_head` (and drafter)  *(quality-aware deeper quant)*
**Source:** qwen38-3090 `prepare/quant_mtp.py`, `drafter/gptq_lm_head.py` (catalog 2.2).
**Why:** our `output_head` is W8G32 (already 8-bit). int4 halves the read again. On the 3090, int4 cost ~5% acceptance at default sampling, **0 at greedy**.
**Spec:** GPTQ-quantize (calibrated, not naive) — at minimum the **draft head** from Objective 1 (40,960 rows → ~124 MB int4). Optionally the full verify `output_head`. Gate hard:
**Gate:** A2 token-identity at greedy **and** perplexity + a GSM8K-style check (catalog 8.1 quality battery) vs the W8G32 baseline. If quality regresses, keep int8.
**Expected:** draft GEMV ~2× faster again on top of Objective 1.
**2026-08-21 addendum (doc 28, measured):** the *all-layer* Q4G64 variant ("4b") is **CLOSED NEGATIVE**: verify only −1.79 ms (35.23→33.44, far below the naive GEMV-halving projection), acceptance −4.9 pts (85.6→80.7), net **82.25 t/s ≈ neutral**, plus target-model text degeneration (A2 diverged). Plain bulk quantization is dead; any quantization lever now requires **layer-selective quantization + Hessian calibration + perplexity/quality gate** (doc 28 lever 2) — treat as research, not a sure win.

### Objective 5 — Draft `sample_method = greedy`  *(possible free win)*
**Source:** v100-skinny `acceptance_gap_notes.md` (catalog 3.3).
**Spec:** inspect our MTP draft sampling path; if drafts are sampled with temperature/top-p rather than greedy, switch to **greedy**.
**Expected:** +10–25% acceptance at low effort. Measure before/after on the A2 harness.

### Objective 6 — fp16 GDN recurrent state  *(halve GDN state traffic)*
**Source:** qwen38-3090 optimizations #3 (catalog 5.1).
**Why:** the GDN state is read/written every decode step + our rebase (0.40 ms). If it is fp32, fp16 halves footprint + traffic with no quality loss (the delta-net kernel is already ~85% BW — the dtype is the lever, not the kernel).
**Spec:** change the GDN/Mamba state dtype fp32→fp16 (not bf16 — 10 vs 7 mantissa bits). Re-run A2 + perplexity.
**Expected:** −0.2–0.4 ms rebase + faster GDN verify steps.

### Objective 7 — k=4  *(measured: loses deterministically; research-level lever, doc 28)*
**2026-08-21 status:** T=5 kernel fixed (r8c5 single-pass, verify T=5 = 39.4–39.5 ms, deterministic). k=4 @512 tok = **78.4–78.8 t/s, 71.8% accept, 3.88 tok/round** vs k=3 82.1 — still loses. Break-even needs marginal 4th-draft acceptance ≥ ~46–57% (currently 37%). Lever is **draft quality, not kernels**.

### Objective 7 (original spec) — Revisit k=4 (only AFTER 2a + 3; first probe the T=5 anomaly)
**Source:** catalog 3.2/3.5.
**Why:** k=4 still loses (47.11 t/s vs 79.20 at k=3, verified 2026-08-20). **Open anomaly (doc 16 review):** verify T=5 = 58.66 ms vs T=4 = 36.55 ms — one extra query row costs 22.1 ms, which GEMV (weights read once) and attention (µs-scale) cannot explain. The small-T dispatch fix (t≤7, cdf0f758) improved the k=4 round 68.19→63.88 ms but the 1.6× T-ratio remains. Profile the T=5 GEMV path (kernel selection / padding / MMA vs SIMT) before spending time on k=4.
**Gate:** measure t/s per k; keep the best.

---

## Out of scope now (log for later)
- **DFlash2 block drafter** (catalog 3.4) — separate 1.92B non-autoregressive drafter; the >90 t/s bet. Architecturally large; needs its own GPTQ W4A16 artifact.
- **Lookup drafting** (3.6) — long-context/RAG only; not our low-context agent.
- **W4A8 int8 tensor-core GEMMs** (2.3) — high-concurrency lever; we are single-stream T≤4.
- **int8 KV** (5.2) — matters at 4k+ context, not at our ~200-token test.
- **Prefix caching** (8.2) — multi-turn, not single-shot.

---

## Definition of Done (overall)
- ~~MTP k (best) steady `--tokens 512 --ctx 4096` **≥ 70 t/s**, acceptance ≥ 55%.~~ **MET 2026-08-20: 79.20 t/s, 85.6% acceptance (battery-verified).**
- Ceiling target now: **≥ 100 t/s** via Objectives 2a + 3 (no quality risk).
- **A2 token-identity green** after every change (P2P AR, graph, int4, fp16 state, greedy). Run `tools/verify_battery.sh` (rebuilds only if stale; ~90 s).
- Plain decode (`--mtp 0`) still works (A1 stays fixed).
- Per-phase B1 harness re-run and pasted into doc 11 (verify the verify-vs-MTP-head split actually moved).
- Quality battery (perplexity + GSM8K) run for any precision change (Obj 4).
- Build clean; commit to `mtp-perf`; push to `chrisconcepcion/dual_5060_ti_ninfer`. **Do not sweep up unrelated uncommitted work** — commit only your files.

## Commands
```
cd /tmp/ninfer && git checkout mtp-perf && cd build && make -j24 ninfer_tp2_decode_test
ART=/home/intel/models/qwen3_8_27b.ninfer
cd build
# steady-state (the DoD number)
timeout 300 stdbuf -o0 ./tests/ninfer_tp2_decode_test --artifact $ART --tokens 512 --ctx 4096 --mtp 3 --prompt "The capital of France is"
# plain baseline (A1 regression)
timeout 120 stdbuf -o0 ./tests/ninfer_tp2_decode_test --artifact $ART --tokens 40 --ctx 4096 --mtp 0 --prompt "The capital of France is"
# draft-vocab coverage check (validate the id list before Objective 1)
python3 - <<'PY'
import json
ids=set(json.load(open('tests/multi_gpu/data/qwen38_draft_vocab_ids.json')))
# sample N greedy tokens from a plain MTP run and report coverage = len(sample & ids)/len(sample)
PY
```
