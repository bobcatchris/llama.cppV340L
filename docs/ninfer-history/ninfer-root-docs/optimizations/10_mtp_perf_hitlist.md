# MTP k=3 Performance Hitlist — 2× RTX 5060 Ti (**Qwen3.8-27B**, TP2)

> ## ⚠️ TARGET MODEL: `qwen3.8-27b` (NOT 3.6)
> - Target artifact: **`qwen3_8_27b.ninfer`** (model_id `qwen3.8-27b`, target_key `qwen3_8_27b`) at
>   `/home/intel/models/qwen3_8_27b.ninfer` (source: `https://huggingface.co/neroued/Qwen3.8-27B-NInfer`).
>   The 3.6 model was deleted. **All commands below use the 3.8 artifact.**
> - NInfer's single `Qwen3_6_27B` target handles BOTH `qwen3.6-27b` and `qwen3.8-27b`
>   (`src/targets/registry.cpp:183-189`); C++/sampling defaults are identical for the two, so the only
>   difference is the baked-in artifact config+weights. **All TP2/MTP work + kernels transfer unchanged**
>   (shape-driven). **BUT the numbers below were measured on the 3.6 artifact — the next agent MUST
>   re-run B1/B2 on the 3.8 artifact first** to confirm the shape matches (64 layers, hidden 5120, GDN,
>   MTP) and get the true 3.8 baseline. If the shape differs, re-derive the TP split (machinery adapts)
>   and re-measure before starting C.

> Hand-off worklist for the next agent. Each item is scoped with **WHAT / WHY / HOW / VERIFY / EXPECTED**.
> Order the work as: **A (correctness) → B (measure) → C (perf levers)** — do not start C until B has
> broken down the per-round cost, so we optimize the real bottleneck.

> ## 🔴 REGRESSIONS FOUND IN REVIEW (fix before anything else)
> 1. **Plain decode (`--mtp 0`) HANGS after ~17 steps.** Root cause: the MTP-fix prefill now emits t0 and
>    sets `cursor.F = plen-1`, so the plain loop decodes t0 at cache position `plen-1` instead of `plen`
>    (off-by-one vs the pre-agent code, which set `anchor = prompt_ids.back()`). See **A1**.
> 2. MTP numbers in this doc are from the **deleted 3.6 artifact** — re-measure on 3.8 (see B).

## Ground truth (measured 2026-08-20, `ninfer_tp2_decode_test`) — **on the 3.6 artifact (reference only)**

| Config | t/s | notes |
|---|---|---|
| Plain TP2 decode (`--mtp 0`) | ~32.5 (target) | **currently HANGS after ~17 steps** (see A1) |
| MTP k=3, 64 tok | 34.1 | prefill-dominated |
| MTP k=3, 128 tok | **44.7** | 64.4% draft acceptance, mean a=1.93, 2.95 tok/round, 44 rounds |
| Target | 70–90 | 2.2–2.8× over plain baseline |

**Per-round cost ≈ 49 ms**, commits ~3 tokens → compute ceiling ≈ 61 t/s. To hit 70–90 t/s the round
must drop to **33–43 ms**. So the goal is: **cut 6–16 ms/round**, and/or commit more tokens/round.

Round breakdown (to be measured in B1, NOT assumed):
`target_verify (T=k+1=4)` → `accept + D2H sync` → `prepare_next_round` → `mtp_forward_decode_batch`
(MTP head alignment, T=4) → `select_accepted_hidden` → `mtp_propose_batch` → 2× AR
(`mtp_forward_decode_batch` T=1 + `mtp_propose_batch`) → GDN `copy_slot(a,0)` rebase.

Key files:
- Driver: `tests/multi_gpu/tp2_decode.cpp`
  - Prefill + t0 + MTP prefill + round-0 draft chain: ~L500–566
  - Plain decode path: ~L567–597
  - MTP round loop: ~L601–752
  - Accept + host sync: ~L653–672
  - Alignment pass + AR: ~L700–740
- MTP forward: `src/targets/qwen3_6/impl/runtime/text_context_impl.h`
  - `mtp_forward_stem` (L360) — FC GEMM: `ops::linear(fc_in, *mtp_.fc, x)`
  - `mtp_forward_tail` (L398) — `Variant::mtp_attention_projection` (qkv+gate), GQA via
    `batch_mtp_kv_`, `ops::linear(*mtp_.o_proj)`, `Variant::mtp_post_mixer` (MLP), final norm
  - `mtp_forward_decode_batch` (L887) — batched MTP head (used by alignment + AR)
- CUDA graph: `src/core/decode_graph.{h,cpp}` — `DecodeGraphExecutable{capture,instantiate,update,upload,launch}`
  **already supports `update`** (per-round kernel-arg changes). Driver does NOT use it yet.
- TP machinery (reuse for MTP head): `src/core/multi_gpu/tp_kernel.{h,cu}`, `weight_shard.{h,cpp}`,
  `tp_load.{h,cpp}`; classification in `src/artifact/plan_split.h` (MTP weights currently REPLICATED —
  `weight_shard.cpp` has NO mtp handling yet).

---

## A. Correctness fixes (do first)

### A1 — Plain decode (`--mtp 0`) hangs after ~17 steps  [BUG]
- **WHAT**: `--mtp 0` completes ~17 steps (position ~21) then hangs (timeout). Output is coherent up to
  the hang.
- **WHY**: It is the correctness reference and a baseline. Likely a position/slot bookkeeping regression
  introduced by the prefill change that now emits t0 and sets `cursor.F = plen-1` / `generated = 1`
  (the plain loop then decodes t0 at the wrong cache position, and/or the GDN committed-slot index runs
  past a valid slot).
- **HOW**:
  1. Trace `cursor.F` / `cursor.anchor` through the prefill → first plain step. Confirm t0 is decoded at
     position `plen` (5), not `plen-1` (4). The prefill's last logit predicts position `plen`; the first
     decode step must write t0's KV at position `plen`.
  2. Check the GDN committed-slot (`st.slt`) index each plain step; confirm it stays within the pool and
     matches the position (no wrap/aliasing that corrupts a later step).
  3. Confirm KV page map (`st.kvr`) advances and does not OOB at position ~21 (check `ctx` capacity vs
     `plen + step`).
- **VERIFY**: `timeout 60 ./tests/ninfer_tp2_decode_test --artifact ... --tokens 40 --ctx 4096 --mtp 0
  --prompt "The capital of France is"` → prints `decoded 41 tokens in ... (≈30 t/s)` and coherent output.
- **EXPECTED**: restores the ~30–32.5 t/s baseline reference.

### A2 — MTP determinism + token-identity vs plain  [CHECK]
- **WHAT**: Prove MTP emits the exact greedy sequence (not just "looks coherent").
- **HOW**: (a) Run MTP k=3 twice, byte-diff the token stream. (b) Once A1 is fixed, run plain decode for
  the same N and diff token-by-token against MTP.
- **VERIFY**: two MTP runs are identical; MTP tokens == plain-decode tokens for the first ≥64 tokens.
- **EXPECTED**: both pass (early tokens 13, 271, 248068 already match plain).

---

## B. Measurement (prerequisite for C — do before optimizing)

### B1 — Per-phase round timing  [INSTRUMENT]
- **WHAT**: Break down the ~49 ms/round into phases so we know which lever pays off.
- **HOW**: Wrap each phase in CUDA events on rank 0 (a small `PhaseTimer`), accumulate over all rounds,
  print a mean breakdown at the end:
  `verify`, `accept+sync`, `prepare_next_round`, `mtp_align`, `select_hidden`, `propose`, `ar_total`,
  `gdn_rebase`, `other`. Reuse the existing `t0`/`t1` pattern already in the loop.
- **VERIFY**: a single summary line, e.g.
  `phase-ms: verify=31.0 accept=1.2 align=6.1 select=0.3 propose=1.0 ar=4.8 rebase=0.4 other=4.2`.
- **EXPECTED**: tells us whether verify (unmovable) or the MTP-head/launch/overhead tail dominates.

### B2 — Steady-state throughput  [MEASURE]
- **WHAT**: The 44.7 t/s figure is 128 tokens; confirm the true asymptote and acceptance at scale.
- **HOW**: Run `--tokens 512`. Report t/s + the acceptance line (already printed).
- **VERIFY**: t/s plateaus; acceptance stable (~60–65%).
- **EXPECTED**: baseline for all C-phase gains (report before/after against this number).

---

## C. Performance levers (data-driven, after B)

Expected path: **44.7 → ~55 (C2) → ~70–85 (C1+C3)**. Verify each step against B2.

### C1 — CUDA-graph the MTP round  [HIGH VALUE, MEDIUM EFFORT]
- **WHAT**: The round shape is **fixed** (verify T=k+1, align T=k+1, AR k−1) regardless of acceptance `a`;
  only the seed/positions change per round. Capture the whole round as one graph and `update`+`launch` it
  each round.
- **WHY**: ~8 kernel launches/round → launch-submission latency. B1's `other`/launch bucket is the target.
- **HOW**: Use `ninfer::DecodeGraphExecutable`. Capture the round body once (fixed buffer addresses —
  the driver already uses fixed staging buffers: `st.verify_ids`, `st.ar_positions`, `st.kvr`, ...).
  Per round: write new positions/drafts into the **same** device buffers, call `update` (or just `launch`
  if only buffer *contents* change, not kernel args), then do the single unavoidable `a` D2H.
  - If any kernel arg (not just buffer contents) changes, use `update()`.
  - The one data-dependent host value is `a` (needed for next anchor/F + rebase) — keep that single sync;
    do NOT let it drain the graph (sync after launch, not before).
- **VERIFY**: B1 breakdown shows the launch/`other` bucket shrink; t/s up; A2 still passes (deterministic).
- **EXPECTED**: −8 to −15 ms/round.

### C2 — TP-split the MTP head  [SOLID, MEDIUM EFFORT]
- **WHAT**: The MTP head is REPLICATED — `fc`, `attention`, `o_proj`, `post_mixer` run full-width on BOTH
  GPUs. Split them RowSplit (world=2) exactly like the text layers.
- **WHY**: MTP head ≈ the `align + propose + ar` buckets in B1. Halving it saves several ms/round.
- **HOW**:
  1. `src/artifact/plan_split.h`: classify the MTP weights — `mtp.fc` / `mtp.attention` (q,k,v,gate) /
     `mtp.output` → RowSplit; `mtp.post_mixer` → RowSplit on gate/up + allreduce on down.
  2. `src/core/multi_gpu/weight_shard.{h,cpp}` + `tp_load.{h,cpp}`: add MTP weight rows to the sharded
     load (reuse the existing `RowSplitK128V1` / `MultiRange` paths; the MTP layer is a single layer).
  3. `text_context_impl.h` `mtp_forward_stem`/`mtp_forward_tail`: when TP active, route the four GEMMs
     through `ops::tp_gemv` (see `attn_mix_tp`/`mlp_tail` for the exact pattern) and add allreduce on
     `o_tail` + `mlp_tail` (reuse `TpGroup::allreduce_local_bf16`).
  4. MTP attention KV (`batch_mtp_kv_`) is per-rank (each rank holds its head slice) — confirm the GQA
     local path is used (same as text layers).
- **VERIFY**: `tp2_shard_plan`/`tp2_load` report the MTP weights sharded; MTP t/s up; A2 passes.
- **EXPECTED**: −4 to −6 ms/round (halves `align + ar + propose` GEMV cost).

### C3 — Tune k  [LOW EFFORT, AFTER C1+C2]
- **WHAT**: 64% draft acceptance means the drafts are decent. Test k=4 and k=5.
- **WHY**: More tokens/round amortizes the verify; the MTP-head cost is already halved by C2.
- **HOW**: `--mtp 4` and `--mtp 5`; compare t/s + acceptance from B2-style runs. Watch VRAM (MTP pool,
  KV) and per-round cost.
- **VERIFY**: highest stable t/s; acceptance doesn't collapse at higher k.
- **EXPECTED**: k=5 can push past 85–90 t/s once C2 is in (5 tok/round).

### C4 — Collapse host syncs / reduce launches  [LOW EFFORT, FILLER]
- **WHAT**: Each round does `ctx.synchronize()` + 2 D2H (accept, lic). Merge into one 5-int D2H; drop the
  redundant `ar_hidden` D2D copies in the AR loop if the buffers can be ping-ponged.
- **WHY**: Shrinks the accept/sync bucket from B1.
- **HOW**: single `cudaMemcpy` of a `{accepted, lic0..3}` struct; reuse the acceptance workspace buffer.
- **VERIFY**: B1 accept bucket drops; t/s up; A2 passes.
- **EXPECTED**: −1 to −3 ms/round.

---

## D. Portability (flag now, do at V340L port time)

- **D1**: Confirm the MTP fix + C-phase changes contain **no sm_120a-specific intrinsics** — the MTP path
  should be portable to HIP/gfx900. (The MTP fix uses `mtp_forward_decode_batch` + stock kernels, which
  are arch-neutral; verify C1/C2 don't reintroduce arch locks.)
- **D2** (deferred): V340L HIP port — separate effort, after 5060 Ti MTP is at target.

---

## Definition of done (whole list)
1. A1: `--mtp 0` runs 40+ tokens, ~30 t/s, coherent.
2. A2: MTP deterministic and token-identical to plain for ≥64 tokens.
3. B1: per-phase breakdown printed.
4. C-phase: **steady-state MTP t/s ≥ 70** (ideally 85+) at `--tokens 512`, with the acceptance line
   showing ≥55% draft acceptance.
5. All runs: `timeout` + `stdbuf -o0`, output logged.
6. Build clean, no new warnings; changes committed to a branch (`mtp-perf`), **push only to
   `chrisconcepcion/dual_5060_ti_ninfer`**.

## Commands
```
cd /tmp/ninfer/build && make -j24 ninfer_engine ninfer_tp2_decode_test
cd /tmp/ninfer/build
# steady-state
ART=/home/intel/models/qwen3_8_27b.ninfer   # <-- 3.8 TARGET, not 3.6
timeout 300 stdbuf -o0 ./tests/ninfer_tp2_decode_test --artifact $ART --tokens 512 --ctx 4096 --mtp 3 --prompt "The capital of France is"
# baseline (after A1)
timeout 120 stdbuf -o0 ./tests/ninfer_tp2_decode_test --artifact $ART --tokens 40 --ctx 4096 --mtp 0 --prompt "The capital of France is"
```
