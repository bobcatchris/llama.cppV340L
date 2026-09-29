W15 DEEP-CENSUS DESK - U1 DEEP-CONTEXT DECODE - RECEIPT - 2026-09-23
Desk: wt-deep-census on amd/deep-census. Question: bound attention/KV
ms-per-round as a function of KV depth and name the depth at which deep
context threatens the 129.0 ms/round of-record.

0. WINDOW OF RECORD (one boot, lane 8083, lock-held 20:45:44-21:32:48)

Stack: canonical TP4 + full opt set, ctx 131072, rocprofv3 --kernel-trace,
binary /media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin/llama-server.
Boot 20 s (page-cached model), health ok. Pre-window: the boot lock was held
STALE 823 s by a dead pid (desk=widelaunch, /proc/60372 absent, zero GPU
processes, dies at idle VRAM 18.5 MB); removed WITH EVIDENCE per the E-121b
law (W15_deepcensus_2026-09-23.stalelock.log) - this also unblocked the
coordinator's own waiting tsab chain. Window length 47:04 vs the ~35 min law
(overrun cause chain below, documented, not hidden).

INSTRUMENT FAILURES (honest accounting):
(a) probe A (63.2k prompt): client urllib timeout 1500 s fired while the
    server was still prefilling (thermal-slowed to ~34 t/s); server canceled
    task 0 at n_tokens = 48,682. NO decode.
(b) probe B (119.4k prompt, same-64k-prefix for slot-cache reuse): client
    killed per plan at 21:24:30, but a non-streaming request is NOT canceled
    by client disconnect mid-prefill - the server kept processing until the
    21:32:43 teardown SIGTERM. Traced prefill reached ~66.5k total KV.
    NO decode at depth. The primary U1 metric (flash_attn_tile per-launch
    us at deep KV on the REAL decode path) is therefore UNMEASURED; the
    verdict below is a model anchored at the banked 7.2k census plus one
    component (the f16-KV-pool dequant) that IS directly measured to 68k.

THERMAL ACCOUNT: within ~4 min the 4-die traced prefill drove all junctions
to 84-85 C; dies 3/4 (0d:00/10:00) sagged to 800-950 MHz while dies 1/2 held
~1250-1430 MHz (W15_deepcensus_2026-09-23.thermal.log, 20 s cadence). The
early healthy phase reproduces the of-record rates (218 t/s untraced at 4k
vs census-era 188 traced at 7k). Per-die curves below use die 1 as the
clock-held channel.

1. A1 MEASURED (W15_deepcensus_depth_extract_2026-09-23.txt; 2,027,544
dispatches; depth per launch read from grid geometry, not wall-clock:
dequantize_block_q4_0 Grid_Size_X/32 = ne11 exactly, verified against the
banked 7168 anchor; tile launches paired with the preceding deq's depth)

- f16-KV-POOL DEQUANT (dequantize_block_q4_0), die 1, MED US BY 4k DEPTH BINS:
  8k:25.0  12k:39.1  16k:56.0  20k:72.4  24k:87.1  28k:107.0  32k:122.7
  36k:134.8  40k:151.6  44k:169.9  48k:186.2  52k:203.0  56k:215.3
  60k:234.0  64k:249.2  68k:264.0
  LINEAR at 3.98 us per 1k tokens per launch, and the 8k bin lands ON the
  banked census anchor (24.9 us at 7168, W11 P0). This is the first DIRECTLY
  measured deep-depth kernel curve of the campaign; it validates the whole
  depth-extraction method. (Dies 3/4 show thermal spikes on top of the same
  slope, e.g. die3 48k bin 468 us - excluded; die 1 clean.)
- PREFILL TILE (flash_attn_tile T=512 instance), die 1: 23.5 ms @4k bin,
  70.3 @8k, 163 @16k, 394 @32k, 984 @64k, 1060 @68k. Early slope 5.9
  us/token reproduces W13's 41 ms @7168 (5.7); late slope inflates x1.4
  (13.6 -> 19.0 us/token beyond 36k) - consistent with the f16 pool leaving
  L2 as depth grows (DRAM-resident re-read), NOT with die-1 clocks (held).
  If real, the decode tile's deep slope is if anything UNDERSTATED.
- SERVED PREFILL WALL (.server progress lines, untraced-rate regime):
  218 t/s @4k -> 190 @7k -> 137 @14k -> 90 @16-22k -> 62 @27k -> 46 @34k ->
  34 @46k -> 15 @62k -> 13.9 @65k. Deep context collapses PREFILL too; a
  64k-token prompt costs ~10-25 min of prompt processing on this card
  (thermally self-reinforcing: the collapse IS the card heating).

2. A2 VERDICT: THE ROUND IS NOT FLAT IN REAL DEPTH

Decode-tile depth model (wave model): the decode tile instance fixes each
CTA to a 192-col KV scan (~730 us, the banked 722 us at 7168 = one wave);
depth grows parallel blocks pb = d/192, waves = pb x 3 z / 112 CTAs, so
tile(d) = 722 us x d/7168 -> LINEAR at 0.102 us/token/launch. This is the
only physical reading of the banked anchor; the prefill-instance evidence
(48 waves and still 0.54 GB/s effective on unique KV) rules out any
wave-count efficiency rescue. Attention/KV class per round per die:

  depth    tile(v+c) draft combine deq40 | class  | round (129 base)
   7168      12.49   0.19   0.44   0.94 |  14.1  |  131  (banked 129)
  16000      27.62   0.41   0.98   1.88 |  30.9  |  148
  32000      54.90   0.83   1.96   3.98 |  61.7  |  179
  64000     109.81   1.66   3.92   8.18 | 123.6  |  241
 120000     205.48   3.10   7.35  15.54 | 231.5  |  349
 200000     342.58   5.17  12.24  26.04 | 386.0  |  503

- the class DELTA vs 8k exceeds 5 ms/round at d ~= 9,400 tokens.
- the "context-insensitive to 32k" of-record claim is a CACHE-SIZE claim
  (7-8k prompts in 32k/200k caches); at 32k REAL DEPTH the round is
  ~179 ms (+39%), at 64k ~241 ms (+87%), at 200k ~503 ms (3.9x).
- CAVEAT (binding): the tile/combine/draft depth terms are MODEL, anchored
  at 7.2k; the direct decode-at-depth measurement is UNMEASURED (section 0).
  The deq40 term is DIRECTLY measured linear to 68k. Any promotion decision
  that hinges on the exact 32k-64k round wall needs the follow-up probe
  (section 4) before banking.

3. F16-KV-POOL TAX VS Q4_0-DIRECT TILE (the W11 forward lever)

Deep-delta split at 200k: tile KV re-read 342.6 ms = 92% of the delta;
pool dequant 26.0 ms = 7%; combine 12.2; draft 5.2. The f16-KV-pool tax
(U1's named unbounded item) is REAL and directly measured (0.82 ->
26.0 ms/round from 8k to 200k) but it is the MINOR term. The major term is
the tile kernel's KV scan itself (z=3 re-reads of an f16 pool at 30.5 GB/s
effective, W11). VERDICT: a q4_0-DIRECT decode tile kernel DESERVES A DESK -
it deletes the dequant pass (7%) AND attacks the 92% by scanning 3.6x
denser KV bytes with no f16 pool round-trip; the W11 wide-arm lesson
(constant launch latency ate a kernel-level win) defines the gate: the
bench must win END-TO-END per launch, not kernel-sum only. Secondary lever
named by this window: the prefill tile instance is latency-bound at
sub-1 TF/s (W13) and its deep superlinearity makes 64k+ prefill a
minutes-class cost - same kernel family, same desk umbrella.

4. FOLLOW-UP SPEC (decisive, cheap): extend docs/amd-port/tests/
bench_attn_real.cu (banked W11 instrument, real launch_fattn host chain,
real GGUF KV bytes, single die, no TP4 window) to sweep ne11 = 7168 / 16k /
32k / 64k / 120k / 200k on the served decode instance; measure tile+combine
+deq per launch vs depth directly, thermally gated. That converts this
desk's model band into a measured curve without spending another TP4 boot
window. If a served decode-at-depth number is still wanted, the window law
learnings apply: no fixed client timeout, chunked prefill with cooldown
gaps, kill = SIGTERM to the server (client disconnect does not cancel a
non-streaming prefill), probes sized to the ~35 min envelope (<= 32k depth).

5. ARTIFACTS
- trace: W15_deepcensus_2026-09-23_kernel_trace.csv (634 MB, 2.03M
  dispatches; gzipped after analysis per disk hygiene)
- W15_deepcensus_depth_extract_2026-09-23.txt (per-die depth curves)
- W15_deepcensus_phase_split_2026-09-23.txt (phase split; burst-ramp
  section's ubatch-bucket depth labels superseded by the extract file)
- W15_deepcensus_2026-09-23.server / .stamp / .probes (probe A traceback =
  instrument-failure evidence) / .thermal.log / .stalelock.log
- scripts: run_deepcensus_window.sh, deepcensus_phase_split.py,
  deepcensus_depth_extract.py, deepcensus_fit.py
