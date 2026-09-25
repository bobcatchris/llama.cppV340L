# M2_NOTES — verify-tail per-op tracer (`[TAIL]`), no-GPU desk, 2026-09-17, amd/tp4-cure

M1 (VERIFY_DECOMP_row.txt, PLOG-042, bin 409450b83ffa4be8) named the tail: of the 117.41 ms
verify round, the post-layer-64 TAIL is **55.21 ms/round**, COMPUTE-verdict (OPTRACE skew
0.010 ms, gap 0.028 ms). M2 = split that tail PER OP so the next fix targets the right op.
This commit lands the instrument only — zero GPU claims inside it.

## Env var

    NINFER_VERIFY_TAIL_TRACE=1     # arms the [TAIL] tracer; unset = byte-identical default

Gate pattern (structural byte-identity): the diff is **74 insertions, 0 deletions** across
`src/runtime/tp2/tp2_backend.cpp` + `src/targets/qwen3_6/impl/runtime/text_context_impl.h`,
plus the new header `src/runtime/tp2/verify_tail_trace.h`. Every added functional line is
either `if (tail_trace.on) { ... }` around a plain bool seeded from the env, or a
`tail_trace.on ? now : 0.0` host-stamp ternary — grep-verified (0 ungated added lines).
Env unset → no event records (zero stream mutation), no extra syncs, no prints, no data-flow
change: same guarantee class as NINFER_TP2_OPTRACE / NINFER_VERIFY_LAYER_TRACE.

## What each timer wraps (field → op → site)

Device ops = one cudaEvent pair on the rank stream, per round (DEVICE time). Host fields =
chrono host walls. One instance per TP rank (`VerifyTailTrace::get(rank)`; the inline
accessor's function-local static merges across TUs — `tp2_backend.cpp` owns the round
boundary + printing, the TextContext verify impl feeds the in-forward ops into the SAME
instance).

| field | op | where |
|---|---|---|
| `prep` | speculative_prepare_verify_inputs kernel | tp2_backend.cpp (round top) |
| `embed` | verify embedding lookup | text_context_impl.h target_verify_batch_impl |
| `finalnorm` | post-layer-64 final rmsnorm | text_context_impl.h |
| `lm_head` | logits projection — **draft-vocab 40960 slice when NINFER_DRAFT_VOCAB resolves (M1 posture), full-vocab ColumnN shard otherwise** (tp_engine.cpp ~:1067 resolution order; B3 loud-warning guards the silent full-lm_head degradation ~3x/round) | text_context_impl.h `ops::linear(flat_hidden, *lm_head_, ...)` |
| `argmax` | verify-column argmax over local logits | text_context_impl.h |
| `lgather` | rk+1 `allgather_local_bf16` NCCL gathers assembling the full logits distribution | tp2_backend.cpp (post-verify) |
| `ar_argmax` | verify `allreduce_argmax` (cross-rank target tokens, device, no host sync) | tp2_backend.cpp |
| `accept` | speculative_accept_greedy_drafts kernel | tp2_backend.cpp |
| `sync` | HOST wall at the round's ONLY hard `cudaStreamSynchronize` (accept D2H drain) — where any device backlog lands | tp2_backend.cpp |
| `d2h` | HOST wall enqueueing the pinned accept D2H copies | tp2_backend.cpp |
| `book` | HOST wall sync-exit → accept-phase stamp (conf-arm host-mapped reads, history push, ngram training) | tp2_backend.cpp |
| `rebase` | kvarn_rewind_text + kvarn_rewind_mtp | tp2_backend.cpp |
| `prepnext` | mtp_prepare_next_round | tp2_backend.cpp |
| `align` | MTP alignment forward (T=1) | tp2_backend.cpp |
| `select` | speculative_select_accepted_hidden | tp2_backend.cpp |
| `propose` | d0 draft head + its allreduce_argmax | tp2_backend.cpp |
| `chain_fwd` | draft-chain per-step T=1 forward (summed over the round's rk−1 steps) | tp2_backend.cpp AR chain |
| `chain_head` | draft-chain per-step draft head + allreduce_argmax (summed) | tp2_backend.cpp AR chain |
| `chain_enq` | HOST wall around the whole chain loop (host-staging discovery) | tp2_backend.cpp |

**Code-read finding (task item b):** there is NO literal verify-logits D2H in the greedy
path — logits stay on device end to end (accept kernel consumes `verify_logits_full`
in-device; only the `accepted`/`licensed` ints go D2H pinned). The tail's "D2H copies"
class is exactly the `d2h` (pinned int copies, µs) + `sync` (drain wait — device backlog
lands here) fields.

**Join math:** fields are per-op times; residuals are the discriminators.
`verify(B1) − loop(LTRACE) ≈ 55.21` should decompose as
`embed + finalnorm + lm_head + argmax + prep + lgather + ar_argmax + <in-verify gap>`,
with `sync` telling you how much of the tail the host waits on at the drain. Rebase..chain
fields decompose the 16.3 ms draft-side class (align 8.03 / AR 7.85 / propose 0.31 / accept
0.08 B1 anchors); `chain_fwd` vs `chain_head` splits the 7.85 AR phase into forward-vs-AR.

## [TAIL] output format (stdout, flushed; one line per round + windowed means)

    [TAIL] rank=0 round=57 t=123456.789 prep=0.003 embed=0.012 finalnorm=0.045 lm_head=3.212 argmax=0.185 lgather=0.410 ar_argmax=0.230 accept=0.055 sync=12.400 d2h=0.001 book=0.090 rebase=0.020 prepnext=0.140 align=8.010 select=0.010 propose=0.300 chain_fwd=4.100 chain_head=3.600 chain_enq=0.050
    [TAIL-AVG] rank=0 n=100 mean_ms prep=... (same fields; every 100 collected rounds)
    [TAIL-SUM] rank=0 n=204 mean_ms prep=... (request end; totals then reset)

Per-rank lines (4 ranks at TP4). Collection drains the PREVIOUS round's events at the
existing hard sync — everything of round R−1 (chain included) precedes round R's verify on
the same stream, so elapsed reads are race-free: LTRACE's argument, one round wider.
Ring of 2 rounds, drain-before-reuse; ≤6 chain steps (kD21MaxDraftDepth−1). Overhead when
ON: ~34 event records/round (µs-class); OFF: zero.

## GPU-owner one-window runbook (do NOT skip BOOT_BATTERY)

1. **Rebuild in the LANE worktree only** (never the shared checkout):
   `cmake --build /home/chris/worktrees/amd-tp4-cure/build-hip-amd --target ninfer-serve -j$(nproc)`
2. **Bank before relink** (docs/amd/BOOT_LAUNCH_RUNBOOK.md §4) →
   `/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin`; boot THE BANKED PATH (filename = stamp).
3. Pre-boot strings check: `strings <banked.bin> | grep NINFER_VERIFY_TAIL_TRACE` must hit.
4. **BOOT_BATTERY GREEN** (per-window law), bank its row + clocks log.
5. **Boot posture** = serve_10k.sh line MINUS the 10k body, plus the traces env:
       export NINFER_ALLOW_NVFP4_TP2=1 NINFER_WORKSPACE_MIB=96
       export NINFER_DRAFT_VOCAB=/home/chris/dual_5060_ti_ninfer/tests/multi_gpu/data/qwen38_draft_vocab_ids.json
       export NINFER_VERIFY_TAIL_TRACE=1 NINFER_TP2_TIMING=1 NINFER_TP2_OPTRACE=1 NINFER_VERIFY_LAYER_TRACE=1
       setsid nohup "$BIN" /media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer --port 8100 \
         --devices 0,1,2,3 --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 \
         --greedy --default-max-tokens 16 --spec mtp --draft-tokens 2 --allow-nvfp4-weights \
         > /home/chris/serve_M2.log 2>&1 < /dev/null &
   Verify in the log: "loaded 40960 draft vocabulary IDs" (the lm_head slice arm — the
   full-vocab fallback would change what `lm_head` measures) + healthy boot ~5 s.
6. **Workload:** count600-class, conc=1 (105-token counting prompt, mt=600, temp=0 greedy,
   one request at a time; M1 anchor family: 201-204 rounds, acc 0.97-0.99, t/r ~2.94).
7. **M0 law:** 2 s clocks/temp sideband sampler (pp_dpm_sclk + edge) through every leg;
   legs <60 s or ≥2-3 min idle gaps; bank the sideband log with the row.
8. **Post-leg:** anchored kill of ONLY the PIDs you started; KFD=0 check.
9. **Bank `VERIFY_TAIL_row.txt`:** bin sha, boot posture, the `[TAIL]`/`[TAIL-AVG]`/
   `[TAIL-SUM]` lines (all ranks) + B1 block + `[LTRACE]` loop sums from the serve log,
   clocks sideband; PLOG via `tools/guards/plog_append.py`.
10. **Read:** Σ(embed..ar_argmax) vs the 55.21 tail names kernel-vs-gap;
    `lm_head` big → lm_head fusion/tiled lever; `sync` big with small Σ → drain/host class;
    `chain_head` big → parked W4 AR class; `lgather` big → gather-batching lever.

## Compile-verify (law line, RC=0, no new warnings)

- Law command = the tree's own compile_commands.json line for each affected TU
  (`/usr/bin/c++ ... -O3 -DNDEBUG -std=gnu++20 -c ...`), run from
  `/home/chris/worktrees/amd-tp4-cure/build-hip-amd/src` — all 243 tree TUs already build -O3.
- Affected TUs (the only includers of text_context_impl.h): `runtime/tp2/tp2_backend.cpp`,
  `targets/qwen3_6_27b/impl/variant.cpp`, `targets/qwen3_6_35b_a3b/impl/variant.cpp`.
- Baseline (pre-edit): RC=0 ×3, warnings 9/9/48 — all `-Wunused-result` (hipError_t nodiscard),
  the pre-existing class. Post-edit: RC=0 ×3, warnings **9/9/48 — identical counts and class**
  (the new header `(void)`-casts its HIP calls so it adds zero instances).
