# COMPLETED_OPTIMIZATIONS.md — what works, what it did for us (clean list)

Companion to REMAINING_ITEMS.md. Every landed optimization, its measured effect, and where
the receipt lives. Serving config of record = the runbook line in /home/chris/serve_10k.sh
(bin 2c8901d3d18adef1 + env arms). All receipts in docs/amd/PERF_LOG_AMD.md (PLOG chain,
58+ byte-verified links) unless noted. Last updated: 2026-09-19 ~05:20Z.

## THE HEADLINE (where the campaign has moved the box)

| metric | window-7 start (2026-09-18) | now (2026-09-19 morning) | factor |
|---|---|---|---|
| 10k prefill (10k-token prompt) | 62.5 tok/s (first-soak era) | **63-104 tok/s** = thermal state function; **104.0 from a cold start** | **1.66x** at the cold end |
| 2k burst prefill (plen-2075) | ~109.7 tok/s, decayed under load | **~112 tok/s HELD FLAT** under sustained load (18.4-18.8 s walls) | the decay is gone |
| decode | 41 tok/s anchor (2k ctx, short gen) | unchanged engine; quantified 40.7 -> 27.0 t/s across 100->600 generated (KV decay) | mapped, not yet moved |

## LANDED OPTIMIZATIONS (active in the serving binary/config)

1. **GDN SIMT rewrite** (NINFER_GDN_SIMT, promoted default-ON) — the chunked linear-attention
   trio rewritten for gfx900 SIMT after 5 root-caused defects. Effect: dn column 480 ->
   ~30 ms/chunk (**x16**); prefill class 76 -> ~116-121 tok/s (+~55%) at its landing.
   Receipt: PLOG-052/053, dn_simt_chunk_cell (poison canaries, ALL PASS ~1e-6).
2. **GQA split-K promotion** (gate + splitk=on default route) — prompt-attention split-K on
   gfx900. Effect: part of the 76 -> 109.7 daytime progression; G4 ladder +15% at chunk-256.
   Receipt: PLOG-052, GQA cell RED->GREEN (cell's own unsigned-RNG bug found and fixed).
3. **GGATE bf16 prefill arm** (gdn_gating_proj SIMT col_tile, promoted default-ON).
   Receipt: PLOG-052 era, [GGATE] route proof from default.
4. **Finalize-tail cure** (NINFER_MTP_TAIL_ASYNC=1) — the last-chunk host stall: 108-byte
   pageable H2D cost 659.6 ms on all 4 ranks; now composed on-device.
   Effect: chunk-17 prep 922-1602 -> ~14-17 ms; request mean gap/chunk 126-167 -> 73-92.
   Receipt: PLOG-055 (bracket chain, 6-bin A/B ledger).
5. **Stem warmup** (boot warmup at the T=27 width class) — cures the one-time 330-676 ms
   first-request width-class cost. Effect: first request runs at full speed (109.7, no
   penalty). Receipt: PLOG-055 follow-on, generation_service warmup().
6. **Pinned-staging cure** (NINFER_H2D_PINNED_STAGE=1) — the ~995 ms/chunk pageable-H2D
   enqueue stall (512-B chunk-ids upload blocking until chunk end on every rank), cured by
   a per-thread ring of 64 pinned slots, event-guarded reuse (torn-payload impossible).
   Effect: **paired A/B -18.6/-27.1/-32.6% per ordinal; pinned FLAT vs unset RISING**;
   10k soak state function 63.1/70.7/104.0 (hot/warm/cold) vs 62.5 era; sustained-load
   decay eliminated. Receipts: PLOG-058/059/060/061/064, w7_h2d_stall_cell (300x collapse).
7. **Trace instrument fix** (ar-bracket subslot ring) — the OPTRACE ar column read 60.3 ms
   while truth was 119.5 (mlp re-recorded the mixer's begin event); now reads
   **123.0/123.3 = profiler ground truth** at wall +-0%. Effect: measurement honesty for
   every future leg; revealed AR as the true #2 prefill item (12% of chunk wall).
   Receipt: PLOG-062 (R6 verify).
8. **Platform: iommu=pt** (39 DMA groups -> identity) — part of the current platform
   baseline; P2P remains absent (hardware fact). Receipt: W7_p2p_pt_row.
9. **Thermal policy: clocks AUTO** (reversal of the pin-high experiment) — pinned-high hurt
   sustained 1.8x; auto holds the fast band under the 110 W cap. Receipt: PLOG era +
   W7_therm_row (integrator quantified: 46->80 C in 85 s bursts; ~90 s drain; ~3 min reset).

## LANDED TOOLS & GUARDS (not perf, but they hold the gains)

- BOOT_BATTERY.sh per-window battery (guard + ladder + clocks sideband) — every landing runs it.
- --v3parity patterned-data cell (the two V-family bugs can never silently return).
- OPTRACE prefill brackets (env-gated, honest ar column post-fix).
- NINFER_VOCAB_COUNT_DIR coverage counter (armed every boot; free draft-vocab tripwire).
- PERF_LOG hash chain (tools/guards/plog_append.py) — 58+ byte-reproducible links.
- RED->GREEN cell suite: GQA Rng cell, dn chunk cell, h2d stall cell, draft-arm roofline cell.

## TRIED AND CORRECTLY REJECTED (so we never pay for them twice)

Clock pinning (hurt sustained 1.8x) · graph capture (not launch-bound) · pk16 sketch
(1.34x vs 2.0 bar) · mad-mix (0.48x, resourcing) · K-split (monotone loss) · co-residency
3-blocks (occupancy raised, no time moved — 2 independent desks) · RCCL env matrix
(NULL — SHM ring at 2 channels is the envelope) · padded swizzle (wash) · draft-vocab
widening (coverage already 97.6-99.4%) · decode draft-arm GEMV retune (already 1.15x of
roofline). Each has a named reopen condition in REMAINING_ITEMS.md §DORMANT-RETRY.
