# W10 RECEIPT: SPLIT-BALANCE DESK (P0 byte audit + A1 arrival model + A2 rebalance design + A3 served spec) - 2026-09-23

Desk: SPLIT-BALANCE DESK (wt-split-balance, amd/split-balance, based on
676cbc806). Zero-GPU throughout: GGUF offset-delta parsing + the exact
loader split-rule simulation + the TP4 kernel census. Named by the
transport desk's A1 finding (W9): the served-only boundary peer-wait is
compute-arrival, and the hypothesis on dispatch was per-die weight byte
imbalance. Foundation: ledger E-110..E-119, W8, W9, TP4_roundmap.

## 0. Headline

1. P0 VERDICT: BYTE IMBALANCE IS NOT THE MECHANISM - and not by a small
   margin: the served TP4 split is BYTE-EXACT EVEN. This tree does not
   assign layers to dies; it shards EVERY weight tensor by name-keyed
   axis across ALL dies (src/llama-model.cpp:333-357 patterns, :383-501
   axis table), so every die holds an identical 1/4 slice of every
   sharded tensor + full copies of the mirrored ones. Per-die weight
   bytes = 3,348,768,560 B on ALL FOUR dies (max-min = 0 B, 0.0000%;
   per-block spread 0 on all 65 blocks). Die 1 does not carry more
   bytes. The mission's premise is refuted at the byte level.
2. A1 VERDICT: the served per-die boundary band (128.8-210.8 us) is a
   PER-DIE COMPUTE-RATE spread at EQUAL bytes. With the per-agent clock
   skew calibrated out (rocprof agents do not share a timebase; die 1's
   clock is +731 us - the raw starts that made "die 1 arrives last" are
   an artifact), die 3 is the LAST ARRIVER on 58% of boundaries and the
   corrected waits are 74.0 / 90.8 / 37.6 / 0.0 us (die 0/1/2/3). The
   identical-kernel duration medians invert that order exactly: die 3 is
   the slowest compute die on EVERY MMVQ class (per-byte time
   t_d = 0.893 / 0.846 / 0.964 / 1.000), and per-boundary device compute
   = t_d x 579.4 us on all four dies to 0.1% - pure multiplicative rate
   scaling. Ring estimate is uniform (114.5-128.9 us), matching W9's
   isolated x1.07. Mechanism candidate of record: per-die SCLK during
   serve (idle sysfs snapshot: die 3 lowest at 1085 MHz vs die 1's
   1249; PCIe uniform 8.0 GT/s x16 - topology excluded).
3. A2: the byte lever survives in modified form - equalize COMPUTE TIME
   per boundary (bytes proportional to per-die rate), not bytes. Two
   hard findings shape it: (i) the sharding granularity quantizes any
   flag split (attention tensors only move in head units = 25% steps,
   GDN stack 12.5% steps, only the FFN is fine at 1.47% steps); (ii)
   floor+rotation makes achieved shares lumpy. Best legal launch-flag
   split --tensor-split 1.04,1.04,1.02,1.00 -> achieved shares
   26.45/27.90/23.53/22.12% -> wall compute 579.4 -> 547.6 us/boundary
   = ~31.8 us x 136 = ~4.3 ms/round (~3.3%, decode 23.26 -> ~24.0).
   In-code granularity relaxation (GDN only; attention is head-locked
   by semantics) reaches the ideal 534.2 us = ~6.1 ms/round; stacked
   with NCCL_MIN_NCHANNELS=4 (W9) the ceiling is ~8.1 ms/round.
4. A3: pure launch flag for the A/B (no code); numerics class stated
   precisely in section 5: per-die row sets move, each row's
   computation is bit-identical, and the allreduce ring order per
   element is unchanged - only the fp32 association of the four rank
   partials changes = fp-dust class (same as W9's channel change),
   same-sha two-boot gate + owner sign-off for cross-arm byte diffs.

## 1. P0 - the split rule and the byte table

Instrument: docs/amd-port/probes/gguf_split_balance_audit.py (GGUF read
= metadata only; every tensor size from offset deltas; a full run's
output is the law for every number below). Offset-delta law verified:
sum of deltas 12,248,245,184 B + 4,637,536 B header/padding = the
12,252,882,720 B file, and delta = gguf n_bytes on all 866 tensors.

### 1.1 The actual split rule (file:line)

The split unit is NOT the decoder layer. The meta sharding shards each
tensor by NAME-KEYED AXIS across all devices:

- patterns: src/llama-model.cpp:333-357
- axis table (AXIS_0 = split ne[0] = partial-sum axis -> the per-layer
  allreduce; AXIS_1 = split ne[1] = feature-sharding axis;
  MIRRORED = full copy per die): src/llama-model.cpp:383-501
  (attn_output/ssm_out/ffn_down -> AXIS_0 at :441/:461/:481; q/k/v and
  qkv/gate/ffn_up/ffn_gate and output -> AXIS_1 at :426-432/:451-464/
  :472-478/:492; everything else MIRRORED at :501)
- per-layer rotation of segment order: rotation = get_il_eff(il) %
  n_devices, src/llama-model.cpp:386-393 + :403-407 (non-block tensors:
  n_layer % n_devices = 64 % 4 = 0)
- QWEN35 interleave segments: src/llama-model.cpp:505-559 (GDN qkv one
  segment ne_s=key_dim=2048 x nr=5; gate/ssm_out x3; dt/a x3 on 16)
- granularity: src/llama-model.cpp:566-645
- boundaries: src/llama-model.cpp:649-683: scan[j] = cumsum of
  tensor_split[(j + rotation) % n_devices] (:652-658); DEFAULT
  (all-zero flag) high_j = ne_s*(j+1)/n_devices, CUSTOM high_j =
  ne_s*scan[j]/sum, both FLOORED to the granularity (:668-674); die
  (j + rotation) % n_devices takes [low, high); last takes the rest
  (:675-681). The flag parses at common/arg.cpp:2565-2583 into
  params.tensor_split; the model reads it via llama_model::tensor_split
  (src/llama-model.cpp:1685). Serving of record is -sm tensor with NO
  flag (TP4_kernel_census_2026-09-23.server; launch_tp3_200k.sh), so
  the default all-zero branch = exact equal split is what served every
  number in the campaign.

### 1.2 Per-die byte table (default split, the served configuration)

Inventory: 866 tensors; 65 blocks (blk.0-63 trunk = 16 attn + 48 gdn at
interval 4; blk.64 = the MTP block, dense attention + nextn tensors -
is_recr fallback src/models/qwen35.cpp:21-26; no recurrent_layer_count
array in this GGUF).

| class | bytes |
|---|---|
| total tensor bytes (offset-delta) | 12,248,245,184 |
| sharded (AXIS_0 + AXIS_1) | 11,865,968,832 -> 2,966,492,208 per die |
| mirrored per die (token_embd 351,619,840 + nextn.eh_proj 27,852,800 + 216 norms 2,803,712) | 382,276,352 |
| PER DIE, all four | 3,348,768,560 B (3.349 GB) |

Imbalance: max-min = 0 B (0.0000%). Every sharded dimension divides by
4 exactly after granularity flooring (ffn 17408->4352; attn_q 12288 ->
3072; k/v 1024->256; attn_output 6144->1536; gdn qkv/gate/ssm_out/
conv1d 2048->512 per segment; dt/a/alpha/beta 16->4; output vocab
129272->32318), so the per-layer rotation has no remainder to
redistribute; per-block die spread = 0 on all 65 blocks (script output,
table in W10 P0 log). KV cache (runtime, not weights) splits 1 kv head
per die - also even.

"Which tensors live on which die": every die holds 1/4 of every sharded
tensor and the full mirrored set. There is no per-die tensor
assignment to rebalance - only slice widths.

## 2. A1 - arrival model vs the census

Instrument: docs/amd-port/probes/census_arrival_model.py; full output
committed as results/W10_census_arrival_model_2026-09-23.log. Source:
TP4_kernel_census_2026-09-23_kernel_trace.csv (589k dispatches, decode
window = last 5.0 s), read read-only from the coordinator's drop.

1. Reproduction gate PASSED: per-die NCCL medians 192,009 / 211,122 /
   152,155 / 128,895 ns (die 0/1/2/3, agent = die+1 by Location_Id)
   vs W8's 191.6 / 210.8 / 151.9 / 128.8 us - same trace, same band.
2. DEFECT FOUND IN NAIVE CROSS-AGENT ANALYSIS: rocprof per-agent
   timestamps are not on a common timebase. Calibration from the
   collective END alignment (an allreduce completes near-simultaneously
   on all ranks; per-die end offset vs the cross-die median end = the
   clock offset): die0 -472 ns, die1 +731,468 ns, die2 -10,330 ns,
   die3 +474 ns. Raw starts would say "die 1 arrives last on 100% of
   boundaries" - an artifact; W8's per-die DURATIONS are in-die and
   immune, which is why W8's table stands.
3. Corrected arrivals: start spread median 126 us; last arriver per
   boundary = die 3 on 58%, die 1 on 21%, die 2 on 21%, die 0 on 11%.
   Median peer wait: die0 74.0, die1 90.8, die2 37.6, die3 0.0 us.
   Duration - wait = ring estimate 118.0 / 120.3 / 114.5 / 128.9 us -
   uniform within 12%, consistent with W9's isolated flat x1.07.
4. The rate table (identical kernels, skew-immune durations, us):

| kernel | die0 | die1 | die2 | die3 | d1/d3 |
|---|---|---|---|---|---|
| mul_mat_vec_q t21 (IQ3_S) | 151.7 | 148.6 | 171.9 | 177.0 | 0.839 |
| mul_mat_vec_q t18 (IQ3_XXS) | 164.6 | 159.1 | 181.6 | 190.5 | 0.835 |
| mul_mat_vec_q t23 (IQ4_XS) | 94.4 | 90.5 | 100.3 | 106.5 | 0.850 |
| mul_mat_vec_q t12 (Q4_K) | 108.3 | 106.7 | 112.5 | 113.3 | 0.941 |
| mul_mat_vec_q t11/t14/t13 | 170.1/635.1/26.5 | 168.3/629.2/25.8 | 183.3/711.1/26.2 | 198.8/758.0/27.1 | 0.847/0.830/0.951 |
| flash_attn_tile 256x256x4 | 722.4 | 714.9 | 716.5 | 724.6 | 0.987 |

   Non-NCCL device-busy per die (window): 3397.6 / 3280.9 / 3545.4 /
   3666.7 ms -> per-byte time t_d = 0.927 / 0.895 / 0.967 / 1.000.
   Per-boundary inter-NCCL device-busy medians: 517.5 / 490.2 / 558.7 /
   579.4 us -> t_d = 0.893 / 0.846 / 0.964 / 1.000, and meds_d =
   t_d x 579.4 us holds on ALL FOUR dies to 0.1% - compute time is a
   pure multiplicative function of the die rate at equal bytes.
   Stability: first/second window halves give 0.934/0.906/0.965/1.000
   and 0.9187/0.8832/0.9685/1.000 - the order is stable at serve state.
5. VALIDATION OF THE BAND: ring (~115-129) + corrected waits
   (0-91 us) spans 128.9-211.1 vs observed medians 128.9-211.1. Does
   byte imbalance PREDICT the arrival order? NO - bytes are 0.0000%
   imbalanced. The arrival order is the exact INVERSE of the compute
   rate order (fastest die waits longest; slowest die - die 3 - arrives
   last and pays pure ring). Mechanism candidates for the rate spread,
   ranked: (a) per-die SCLK during serve - idle sysfs snapshot
   (/sys/class/drm/card*/device/hwmon): sclk die0 1204, die1 1249,
   die2 1213, die3 1085 MHz, temps 55/49/57/53 C - die 3 lowest clock
   matches die 3 slowest; the ~19% MMVQ spread is consistent with a
   ~10-15% clock delta on a bandwidth/issue-bound kernel; E-111 already
   showed per-die sclk collapse under heat soak on this card.
   (b) per-card HBM/power state - not resolvable zero-GPU. PCIe
   EXCLUDED: all live cards at 8.0 GT/s x16 (card2 = the dead die at
   x4). Decisive serve-state instrument for the coordinator: sample
   hwmon sclk per card during one battery (host-only) or pin clocks
   equal - if the band flattens, clocks are THE mechanism and the
   rebalance compensates what the system could fix at the source.

## 3. A2 - rebalance design

Equalize COMPUTE TIME per boundary: choose per-die byte shares
s_d proportional to 1/t_d. Ideal shares 25.81 / 27.24 / 23.90 / 23.05%
-> ideal per-die compute = 534.2 us/boundary (vs 579.4 today).

### 3.1 The granularity wall (flag lever bound)

The loader floors every boundary to the tensor's granularity
(llama-model.cpp:670-674): attention splits are head-locked by
semantics - attn_q in 3072-wide units (12 q-head PAIRS; the q+gate
doubling, :611-616), attn_k/v in 256-wide units (1 kv head, :628-631 -
the KV cache shards per head), attn_output in 1536-wide units (6 out
heads, :618-621) = 25% steps on all attention tensors; the GDN stack
granularity is lcm(blck,128) = 256 on ne_s = 2048 = 12.5% steps
(:570-577); only the FFN (256 on 17408, :639-643) is fine (1.47%
steps). Floor+rotation additionally lumps the achieved shares (the
fat segment's rounding residue follows the weighted die, not the
round-robin). A grid search under the EXACT boundary math
(docs/amd-port/probes/split_ratio_search.py,
results/W10_split_ratio_search_2026-09-23.log) converges to a plateau
at wall compute 547.6-548.2 us/boundary.

### 3.2 Lever (a) - pure launch flag (RECOMMENDED A/B ARM)

  --tensor-split 1.04,1.04,1.02,1.00   (device order = die 0,1,2,3)

Audit-engine verified achieved sharded shares 26.45 / 27.90 / 23.53 /
22.12% -> max s_d t_d = 0.2362 (die0) -> per-boundary wall compute
579.4 -> 547.6 us. Expected: ~31.8 us/boundary x 136 = ~4.3 ms/round
of the 129 ms round = ~3.3% decode (23.26 -> ~24.0 t/s). No code, no
numerics-structure change (section 5 class), trivially reversible.
Stacked with NCCL_MIN_NCHANNELS=4 (W9 PARTIAL): ring 128.9 -> ~114.7
est -> total ~6.3 ms/round (~4.9%).

### 3.3 Lever (b) - in-code (ceiling, only if (a) verifies)

Env-gated (default-off) rate-proportional split in
llama_meta_device_get_split_state: (i) accept
GGML_CUDA_TP_RATE_SPLIT="r0,r1,r2,r3" as the scan weights when
params.tensor_split is all-zero (anchor: src/llama-model.cpp:652-658);
(ii) when a custom split is active, relax ONLY the GDN granularity
256 -> 64 (anchor: src/llama-model.cpp:570-577) - legal because those
are AXIS_1 (whole-row) slices and quant blocks live along ne[0];
attention granularity stays (head semantics). This makes the GDN stack
fine-grained and reaches ~534.2 us/boundary = ~45.2 us x 136 =
~6.1 ms/round (~4.8%); stacked with ch4 ~8.1 ms/round (~6.3%). Cost: M
(sharding code + capture/replay risk per the E-090 T3 lesson); only
worth it if (a) cashes its ~3% first.

## 4. A3 - served-arm spec for the coordinator

Arms (interleaved multi-rep per the E-119 soak law; provenance stamps
per E-112; the serving of record is UNCHANGED until a GREEN win):

- S0 control: launch_tp3_200k.sh as-is (no tensor-split).
- S1 balance: + `--tensor-split 1.04,1.04,1.02,1.00`.
- S2 balance+ch4: S1 + NCCL_MIN_NCHANNELS=4 (W9's pure-env PARTIAL).
- C0 attribution (optional, zero-code, host-only): pin equal SCLK on
  all four cards for one default-split boot; a flattened served band
  confirms the clock mechanism and re-ranks the levers.

Checks and gates:
1. Device-order witness at boot: ggml-cuda init prints devices in
   order; index 0..3 must map to die0..3 per the campaign die map
   (card1 05:00 / card3 08:00 / card0 0d:00 / card4 10:00, W9). If the
   enumeration order differs, permute the ratios accordingly (the
   audit script documents the assumed order).
2. VRAM: die 1 takes ~+0.35 GB weights vs default; KV is even; the
   boot either fits or fails loudly - covered by the battery.
3. Correctness: accept 0.66667 / mean len 3.00 in every cell; same-sha
   two-boot determinism per arm.
4. Mechanism witness (cheap): post-boot census diff - per-die NCCL
   medians should compress from 128.8-210.8 toward the ring floor and
   per-die MMVQ medians should equalize; if they do not, the rate
   spread moved (thermal) - re-measure t_d before judging S1.
5. Win bar: 200k decode +2% interleaved (>= ~0.5 t/s); baseline
   ratchets only on GREEN (E-119 law).

## 5. Numerics class, stated precisely

The rebalance moves WHICH die computes WHICH rows. Each output element
of a sharded mul_mat is still computed by exactly one die from exactly
the same weight rows and the same activation values - each row's
computation is bit-identical wherever the row lives. The four per-die
partial vectors that the allreduce sums are the SAME four values per
element, redistributed across ranks; the RCCL ring order and its
per-element reduce order are unchanged, but the fp32 ASSOCIATION of
the four partials per element changes because different values now sit
on different ring positions. Class = fp-dust (association-order only),
the same class W9 signed for channel-count changes: NOT bit-exact
across arms, bit-deterministic within an arm (same sha, same boot
class). Gate: same-sha two-boot per arm + owner sign-off on cross-arm
byte diffs (E-085/E-090 convention). No per-tensor sum order inside a
die changes; no quantization path changes.

## 6. A4 - build/identity

No runtime code was changed: the desk's output is two python
instruments + logs + this receipt (git diff vs 676cbc806 touches only
docs/amd-port). Default-path byte-identity is trivial; no ggml-hip
build is required or performed. The GGUF was opened read-only via
header metadata; the GPUs were never touched (no server, no probe, no
ROCm call; sysfs reads are passive host files).

## 7. Defects and notes for the record

1. ROCPROF PER-AGENT CLOCK SKEW: cross-agent Start_Timestamp
   comparisons in the census are invalid without END-median
   calibration (die 1 offset +731 us). Any future cross-die timeline
   work must calibrate first; per-die DURATIONS are safe.
2. The round map / W8 framing "the wall pays die 1's 31.2 ms/round of
   NCCL time" counts die 1's WAIT as wall; the wait overlaps other
   dies' compute. The wall-relevant NCCL slice is the LAST ARRIVER's
   duration (die 3, ~128.9 us median = ~17.5 ms/round) plus the max
   compute per boundary - which is exactly the component the rebalance
   attacks. W8's per-die medians themselves are correct and reproduced.
3. results/.gitignore *.log needed a force-add for the two instrument
   logs (precedent: committed session logs).

## 8. LOG (append-only, newest last)

- 2026-09-23 desk opened on amd/split-balance @ 676cbc806; ledger
  E-110..E-119 + W8 + W9 + round map read; sharding rule extracted
  (llama-model.cpp:333-683); GGUF metadata + inventory dumped.
- P0: audit instrument written; default split byte-exact even
  (3,348,768,560 B/die, spread 0); commit 01ac568c0.
- A1: census arrival model; skew defect found + calibrated; waits
  74.0/90.8/37.6/0.0; rate table + stability; sysfs check (PCIe
  uniform, sclk candidate). A2: granularity law + grid search; best
  flag split verified in the audit engine. Commit 3f0eb3fe6.
- D: this receipt + ledger E-120a; final commit.
