# TP4 ROUND MAP - where the 23 ms/round goes (zero-GPU desk, census-grounded)

Date: 2026-09-23. Desk: TP4 ROUND MAP (wt-tp4-round, amd/tp4-roundmap, based on
387805bec). ZERO GPU: ledger + receipts + census analysis only.

## 0. Facts of record and method

Serving of record: TP4 (4 dies) + RCCL (GGML_CUDA_ALLREDUCE=nccl) + full
optimization set. Decode 23.26 t/s @200k, accept 0.66667, mean len 3.00
(/home/chris/combowin_regress_200k + .server, task 14: eval 5503.28 ms /
128 tokens = 42.99 ms/token; commit 9be54f82f, binary c691079db528bd72,
provenance PASS 5/5, E-113).

Round wall of record = 42.99 ms/token x 3.00 mean len = 129.0 ms/round.

Primary instrument: the fresh TP4 kernel census (coordinator drop,
docs/amd-port/results/TP4_kernel_census_2026-09-23_kernel_trace.csv,
589,000 dispatches; .server/.stamp/.agent_info copied into this tree).
Conditions: real serving path, TP4 + RCCL + full opt set, ctx slot 32768,
one probe = 6964-token prefill + 98 decode tokens, accept 0.7556,
mean len 3.27 -> 30 rounds; decode wall 3911 ms / 30 = 130.5 ms/round.
Census context check: 130.5 ms/round @32k-slot vs 129.0 @200k-slot -> the
round is CONTEXT-INSENSITIVE at served prompt depths (~7-8k effective
tokens in both; see UNBOUNDED item U1 - these are 200k/32k CACHES, not
depths). Per-die busy cross-check: census MMVQ implies 54.7 GB/s effective
weight stream vs the bench share-kernel band 48-52 GB/s + cfull ceiling
70-120 GB/s (W5 receipt) - the trace is NOT in the 2-3x rocprof inflation
regime of E-102; absolute times are usable.

Model constants: GGUF 12,252,882,720 B (offset: file size), /4 dies =
3.063 GB/die; 65 blocks, n_embd 5120, ffn 17408, n_head 24, n_kv 4,
head_dim 256, vocab 129272 (GGUF metadata). HBM ceiling 183.8 GB/s
(W0_census_decode_receipt.md measured). Weight-stream floor per full pass
per die = 3.063 GB / 183.8 GB/s = 16.7 ms.

## 1. Round structure (census timeline, one die)

Sampled round (agent 1, 140.2 ms window): verify pass (65-block cadence,
~1.65 ms/layer, one big flash_attn every ~4th layer = 16 full-attn layers
+ 49 gated-delta-net layers) ending in the T=4 head MMVQ (1379.5 us);
ALL 4 DIES GO IDLE SIMULTANEOUSLY 108.4 -> 115.0 (6.5 ms - a true HOST
gap, not cross-die skew); catch-up (draft ctx, T=4, one 718 us flash) ~2.3 ms;
three draft steps at 2.98 ms period (118.9 / 122.0 / 125.0, each = block
MMVQs + 60-67 us flash + 2 NCCL + 695 us draft head); next verify starts
128.1. Dies are 113.6/130.5 = 87% busy across the strict decode window.

## 2. ms/round decomposition table (per die, TP4 @ ~8k effective ctx)

wall 130.5 ms (census) / 129.0 ms (200k battery - same structure)

| # | component                                | ms/round | source (receipt path or derivation) |
|---|------------------------------------------|----------|-------------------------------------|
| 1a | MMVQ weight-stream kernels, total       | 60.6     | census strict window: 533.7 launches/die/round; per-type (ms): t21/IQ3_S 17.6, t18/IQ3_XXS 14.8, t23/IQ4_XS 10.9, t12/Q4_K 8.5, t14/Q6_K 3.7 (incl. heads), t11 2.1, t13/Q5_K 1.8, t8 0.6, rest 0.7 |
|     | - of which verify pass (T=4)            | 56.0     | 60.6 minus draft 3.4 minus catch-up 1.2 (timeline alloc); = 54.7 GB/s effective on 3.063 GB |
| 1b | RCCL allreduce boundaries               | 22.7-25.6| census: 137.2 NCCL launches/die/round; median 165.6 us, avg 186.7, p90 261.2, min 76.2; 137.2 x median = 22.7, x avg = 25.6. VERIFY share ~129 x 165.6 us = 21.4 |
| 1c | flash_attn_tile + combine               | 16.1     | census: ~20 tile calls/die/round; verify 16 x 716-804 us; catch-up 1 x 718; draft 3 x 60-67; combine 21 us each |
| 1d | small kernels + copies                  | 11.3     | census residual of 113.6 busy: quantize_q8_1 2.4, rms_norm 1.8, k_bin_bcast 1.2, cpy_scalar 1.1, copyBufferRect 1.1, get/set_rows 1.4, concat 0.7, gated_delta_net 0.8, dequant q4_0 0.8, rope 0.4, norms/unary 1.1, ssm_conv+fwht 0.6, tail ~1 |
| 2 | host slice A: post-verify (drain, 4-row sample+accept, batch build, issue) | 6.5 | census timeline: all 4 dies idle 108.4->115.0; corroborated E-078 host-only 5.5-8 ms (VERIFY_ROUND_2026-09-22.md) |
| 3 | catch-up (draft ctx, T=4) device + gap  | 3.5      | census timeline 115.0->118.6; device ~2.3 (MMVQ 1.2 + flash 0.74 + NCCL 2 x ~0.24) |
| 4 | draft loop 3 x T=1 (device 1.3-1.6/step + host/latency ~1.4/step) | 8.9 | census timeline: 2.98 ms step period; device/step = block MMVQ ~0.43 + head 0.695 + flash 0.06 + 2 NCCL ~0.1-0.37; TP3 wall class 11.05 ms (E-073) improved by RCCL + packed arms |
| 5 | draft->verify host gap + in-verify microgaps | ~4  | arithmetic: 130.5 - (113.6 busy + 6.5 + 3.5 + 8.9) = ~4; in-verify launch gaps ~0.1 ms x 65 layer groups |
|   | TOTAL                                   | 130.5    | sums to wall within rounding |

Memo vs the of-record floor: E-105 states "4-rank floor ~106 ms/token-pass
-> ~28 t/s theoretical; 23.34 = 83% of ceiling". The ledger never wrote the
106 ms composition. Reconciliation against the census: the 106 figure is
consistent with pricing MMVQ at TWO full passes x 91 GB/s (67 ms), boundary
tax 48 x 94 us (4.5 ms), draft 11, host 8, flash ~15. Against that math the
census says: boundaries 25.6 (+21.1 - the count TRIPLED to 137/round and
per-boundary is 165.6 us, not 94), small-kernel class 11.3 (never priced),
partly offset by MMVQ 60.6 (-6.6: catch-up is draft-ctx, nearly free, and
the T=4 rate is 54.7 GB/s) and host/draft idle 16.9 (-2.1). THE UNNAMED
~23 MS IS THE TP4 BOUNDARY TAX DELTA, with the small-kernel class as the
second unnamed item.

Boundary-count finding of record: E-085/E-105-era structure said "~24 cut
points per full target pass, ~48 boundaries per decode round". Measured TP4:
137.2/round = ~65-67 per target pass ~= ONE PER LAYER plus extras (attn/
delta out-proj + ffn_down partial nodes both cut at n=4). The mechanism
(which tensors became partial when ranks went 3 -> 4) is UNBOUNDED from
existing data - needs the graph audit in Desk T1 below.

## 3. Ranked lever table

| rank | lever | ms/round available | concrete action | feas | risk |
|------|-------|--------------------|-----------------|------|------|
| 1 | MMVQ T=4 bandwidth (weight stream) | 12.2-30.5 (cfull band 70-120 GB/s: 56.0 -> 25.5-43.8); 35.6 if 150 GB/s (mission hypothetical); floor 16.7 at 183.8 | W5 C-rung LDS-y staging (design banked in W5_mmvq_bw_receipt: 52 B pad, stride 13, ~10 syncs/CTA, occupancy budget 7 CTAs/CU); targets the x/y mix tax (+41-100%) on the rpb=2 schedule | L (build-heavy kernel schedule) | M-H: s2r replica wins did NOT transfer (E-113); share DID (E-104). Requires interleaved multi-rep served A/B |
| 2a | per-boundary latency to probe parity | 5.3-9.7 (165.6 -> 126.5 us x 137, or -> 94 us at small sizes) | TP4 n=4 RCCL probe at the served 80 KB fp32 size (4x5120); NCCL env sweep (NCCL_PROTO/NCCL_MIN_NCHANNELS - transport fingerprint says 2 channels, ring+tree, SHM); boundary clustering so consecutive collectives pipeline | S/M | L: env-only, bytes unchanged; verify numerics class stays signed-off fp32 dust |
| 2b | boundary COUNT 137 -> TP3-class ~48 | 12-17 (137 -> ~48-70 x 165.6 us) | graph/meta audit of why n=4 triples the cut-point count; sharding choices that keep small tensors whole (n_kv=4 heads now split 1/die); subgraph merge where consumers allow | L (touches split/graph structure) | M: E-078 called count architectural at TP3; TP4 structure differs - audit first, do not force |
| 3 | host slice A (post-verify 6.5 ms) | 2-4 | split drain vs sample vs accept vs batch vs issue with one -lv 4 TP4 timeline boot; overlap candidate-build with catch-up issue where legal | M | L: host-only, hygiene arms byte-exact |
| 4 | draft loop host slice (~4.2 ms) + gaps | 1.5-2.5 | pipeline next-step set_inputs+launch under the current drain (inputs are event-gated, LLAMA_ASYNC_INPUT=1 in stack); onsample fetch already off critical path | M | L: env-gated class; step-batching stays BLOCKED (E-072 design verdict) |
| 5 | small kernels + copies (11.3 ms) | 2-3 | copies class ~3.9 ms (cpy_scalar 1.1 + copyBufferRect 1.1 + concat 0.7 + get/set_rows 1.4): eliminate redundant convert/copy nodes in the TP4 graph; the rest is spread too thin | M | M: graph edits risk capture/replay (E-090 T3 lesson) |
| 6 | flash/KV path | UNBOUNDED (16.1 ms at ~8k depth) | no lever named until deep-context measured (U1) | - | - |
| 7 | catch-up waste | 0 (CONFIRMED) | catch-up device is only 2.3 ms/round; PREFIX_CATCHUP ceiling 0.6 ms - E-091 neutrality upheld and now explained | - | - |

Closed doors re-confirmed by this map: graph replay structure is NOT
first-order (dies 87% busy; E-099 <=1 ms verdict holds), catch-up row
reduction neutral, T3 grouped neutral (E-093), aln negative (E-106),
layer-split dead.

## 4. TOP-3 next desks (order matters)

1. T1 TP4 BOUNDARY DESK (expected 5-10 ms/round first, 12-17 later).
   Zero-GPU audit: partial-axis census of the meta splitter at n=3 vs n=4
   (name the tensors that became cut points; why 137 vs 48) + one profiling
   window: TP4 RCCL probe at served sizes (80 KB fp32, n=4, grouped vs
   per-boundary) + NCCL_DEBUG=INFO connect fingerprint + launch-timeline
   boot. Rung a = env/clustering (S/M), rung b = count reduction (L, only
   if the audit names a legal merge). Every RCCL ms here is also a prefill
   ms (same collectives).
2. T2 MMVQ C-RUNG DESK (expected 10-20 ms/round). Build the banked W5
   LDS-y staging design on the real rpb=2 schedule; bit-exact device
   oracle first (house law); served gate = interleaved multi-rep A/B vs
   the ratcheting baseline (E-110 law) - the s2r lesson is binding.
3. T3 HOST SLICE DESK (expected 3-5 ms/round). One -lv 4 decode-timeline
   boot at TP4 ( census shows the 6.5 ms gap is host, all dies waiting);
   attribute drain/sample/accept/batch/issue; ship byte-exact overlap arms.

Realistic stack: 129.0 -> ~100-108 ms/round = ~28-30 t/s at current
acceptance. The 40 t/s class still runs through weight-bit reduction or an
acceptance-rate lever (E-107 road note unchanged).

## 5. UNBOUNDED - needs a coordinator profiling window

- U1 DEEP-CONTEXT DECODE (highest value): every served decode number in the
  campaign (14.86 -> 23.26) used ~7-8k-token prompts in big caches. Flash/KV
  scaling to real 50-200k depth is unmeasured; flash_attn_tile is 16.1
  ms/round at ~8k and ~linear in depth. REQUEST: decode-only census, real
  prompt >= 64k tokens (256k slot, 512 forced tokens, lane 8083, full opt
  set, fresh boot) - bounds flash/KV growth + boundary-size drift + host
  slice at depth before any 200k-depth claim is made.
- U2 TP4 -lv 4 decode-timeline boot (host-slice internals for Desk T3):
  full opt set + timelines + -lv 4, one decode request, 200k slot.
- U3 TP4 RCCL boundary micro-probe: n=4 ranks, 80 KB fp32 in-place
  allreduce, grouped-per-16 vs one-at-a-time, NCCL_DEBUG=INFO; fixes the
  per-boundary floor that lever 2a prices. (Existing probes: 94.0 us @32 KB
  was n=3 E-085; 126.5 us @80 KB from the prefill probe table,
  RCCL_EXT_2026-09-22.md.)

Census caveats: 30-round sample; one 7.03 ms NCCL outlier excluded from
medians but in sums; single probe, single boot; kernel-name type mapping
(t21=IQ3_S, t18=IQ3_XXS, t12=Q4_K, t13=Q5_K, t14=Q6_K, t23=IQ4_XS) per the
E-102/W5 projection tables.
