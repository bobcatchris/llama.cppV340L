# P1 — PREFILL PER-OP EVENT TRACER (DECISIVE CHECK P1, PREFILL_DECOMP_2026-09-17 §5)

**Seat:** CODE desk, amd/tp4-cure, 2026-09-17. No GPU work — compile-only; the tracer boots
in a GPU-OWNER window. Implements the P1 named in
`docs/amd/PREFILL_DECOMP_2026-09-17.md` (commit e25dd8479): prefill runs 75.7 tok/s at
plen 1996 but ~58% of the chunk (~950 of 1647 ms) is UNNAMED non-GEMM time; P1 names it
with per-op device event pairs, one boot, zero new kernel code.

**What landed (all in `src/targets/qwen3_6/impl/runtime/text_context_impl.h`):**
`struct PrefillOpTrace` (next to `VerifyLayerTrace`, same house pattern — per-rank
function-local-static instances, env-gated, default OFF byte-identical), the
`VerifyLayerTrace` gate relaxed so the ALREADY-EXISTING layer + AR event pairs also
bracket the prefill path, one missing AR pair added (see Deltas), and the chunk-wall pair
in `prefill_impl`.

## Env

- `NINFER_PREFILL_OPTRACE=1` — arms the prefill per-op tracer. Default OFF is
  byte-identical: every added site is a plain-bool/`active`-flag guard (no device work,
  no prints, no state mutation when unset), the structural-gate pattern of
  `NINFER_VERIFY_LAYER_TRACE` / `NINFER_VERIFY_TAIL_TRACE`.
- Arming adds, per chunk: the existing `vt.layer_begin/layer_end` pairs (64),
  the existing + one new `vt.ar_begin/ar_end` pairs (128), 256-class gemm pairs at the
  tiled-GEMM call sites, and 1 chunk-wall pair. Drain is race-free: prefill_impl is one
  chunk per call and every call ends in the hard `ctx_.synchronize()` — chunk k reads
  chunk k-1's completed events (the LTRACE argument, one chunk wide). The request's LAST
  chunk is drained by `finish()` at the request-end sync (finalize_at_end chunk).

## Output (stdout, [TAIL] grammar, per rank; 4 rank threads interleave — filter by rank=)

    [PREFILL-OP]  rank=R chunk=C tok=T t=<mono-ms> wall= gemm= ar= body= gap= ngemm= gemm_over=
    [PREFILL-SUM] rank=R n=<chunks> mean_ms wall= gemm= ar= body= gap=

- `wall` = device time of the chunk body (embed .. MTP finalize), the denominator.
- `gemm` = sum of the tiled-GEMM call-site pairs (the doc's 6-problem/256-launch census;
  `ngemm` sanity-checks the count — expect ~256 on full chunks; `gemm_over>0` means the
  320-slot per-chunk cap clipped, treat the gemm column as a floor).
- `ar` = sum of the 128 per-chunk TP allreduce windows (1.31 MB bf16 each).
- `body` = layer bodies − ar − gemm (in-layer non-GEMM non-AR ops).
- `gap` = wall − gemm − ar − body = **the decisive number** (the unnamed residual).
  Things outside layer bodies (embed, finalnorm, last-chunk lm_head, MTP prefill chunk)
  land in gap — that is exactly the doc's "wall − layers" tail/sync/MTP discriminator.
- Warmup prefills also print (each finalized prefill = one [PREFILL-SUM]); the analyzer
  keys on the probe request's chunk=1..16 run and its SUM line.
- Side effect when ON (documented, env-gated): the batched verify pass during decode also
  arms VerifyLayerTrace (its gate's first disjunct is unchanged, but `vt.on`-style arming
  now occurs via the prefill disjunct's shared `ensure/begin_pass`), so [LTRACE] stderr
  dumps appear every 64 passes; they are supplementary per-layer means, not part of P1.

## Expected P1 readout (S1 central, per FULL 128-token chunk, per rank)

    wall ~1650 = gemm ~530 + ar ~130 + body ~90 + gap ~900

with wall/gemm/ar/body/gap ALL measured in the same boot for the first time (the 530 and
130 are today derived/extrapolated; P1 makes them rows).

## The three falsifiers (same boot, re-rank the levers — verbatim from the doc §5)

1. `gemm` > ~1100 ms/chunk ⇒ hypothesis **I** (cold-weight DRAM/TLB streaming in the
   tiled kernel's streamed-weights path) — the GEMM kernel lever REOPENS (Fix C).
2. `ar` > ~250 ms/chunk ⇒ hypothesis **H** owns (TP AR bandwidth term at 1.31 MB × 128
   over the 2-root PCIe topology, W2-class anchor extrapolation broke) — Fix B (RCCL env
   A/B + axpy epilogue fold), predicted 1650 → ~1300-1500 ms ⇒ 85-98 tok/s.
3. `body` (= layers − ar − gemm) small (< 300 ms) AND gap (= wall − layers) large ⇒ the
   tail/sync/MTP class owns the residual; ELSE hypothesis **G** owns (eager inter-op
   stream gaps, prefill NOT graph-captured while decode is GRAPHS-ON) — Fix A (chunk
   graph capture/replay via the existing DecodeGraph machinery), predicted 100-130 tok/s.

## Deltas vs the doc's sketch (both flagged in-code)

- **GEMM pairs at the call sites, not the `[TILED]` seam:** the doc offers "6 sites, or
  piggyback the `[TILED]` dispatch seam nvfp4_dispatch.cpp:132/:194". The seam file
  (`src/ops/linear/nvfp4/nvfp4_dispatch.cpp`, linear dispatch) is owned by a concurrent
  desk (LMHEAD route-flip) — this desk may not edit it. The pairs bracket the five
  call sites in text_context_impl.h (`Variant::attention_projection_tp`, `tp_gemv` o_proj,
  `Variant::gdn_input_projection_tp`, `tp_gemv` out_proj, `Variant::post_mixer_tp` — the
  last wraps mlp gate_up+down), which enqueue exactly the same device work; at
  T=128 every one routes to the tiled kernel. Event cost when ON: 2 records per site.
- **The gdn prefill-arm AR had NO pair:** the doc's ":2437-2443" cite is the gdn DECODE
  arm (it `return;`s before the prefill arm). The prefill arm's allreduce (48 of the 128
  per-chunk ARs) got a pair in this change, byte-identical style (`if (vt.active)`).
  Without it, `ar` would have covered only 80/128 ARs and systematically under-priced H.

## GPU-OWNER RUNBOOK (one boot, cool window)

0. Grant: written GPU grant from coordinator; `gpu_guard` foreign-CUDA check; server on
   :8100 (PID 310132 today) is NOT this window's — do not touch; sole-process discipline,
   <60 s bursts / 150 s idles, clocks sideband 2 s banked (TILED probe-b pattern).
1. **Rebuild** in THIS worktree (branch amd/tp4-cure, this commit):
   `cmake --build /home/chris/worktrees/amd-tp4-cure/build-hip-amd -j$(nproc)` (or the
   window's normal build path — the TUs touched are tp2_backend.cpp and the qwen3_6
   target impls; compile RC=0 verified at the exact -O3 build commands). Bank the binary
   sha per BANK-BEFORE-RELINK (docs/amd/BOOT_LAUNCH_RUNBOOK.md §4) and boot from the bank.
2. **Boot battery first** (standing cells, GREEN required): same posture as
   TILED_DATUM boot 1 minus nothing:
   config `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer --port 8100 --devices 0,1,2,3
   --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 --greedy
   --default-max-tokens 16 --spec mtp --draft-tokens 2 --allow-nvfp4-weights`,
   env `NINFER_ALLOW_NVFP4_TP2=1 NINFER_DRAFT_VOCAB=/home/chris/dual_5060_ti_ninfer/tests/
   multi_gpu/data/qwen38_draft_vocab_ids.json NINFER_WORKSPACE_MIB=96 NINFER_TP2_TIMING=1
   NINFER_TILED_TRACE=1` **plus `NINFER_PREFILL_OPTRACE=1`**.
3. **P1 probe:** ONE plen-1996 conc=1 prefill probe (the TILED_DATUM (b) reconstruction:
   BLUE body + 1938 space-separated alphas, mt 64, temp 0, finish=output_limit), cool
   window. Capture stdout ([PREFILL-OP] 16 lines + [PREFILL-SUM] per rank × 4 ranks) and
   the clocks sideband.
4. **Bank:** `results/amd/coherence/PREFILL_OPTRACE_row.txt` (manifest, bin sha, config/
   env line, ttft/tok/s, the 4 ranks' [PREFILL-SUM] lines verbatim, per-chunk gap column,
   falsifier verdict 1/2/3-or-G, clocks band) + the raw serve log. Default-OFF
   byte-identity needs no cell: OFF adds only `std::getenv` per process (compile-verified
   RC=0, zero new warnings).
