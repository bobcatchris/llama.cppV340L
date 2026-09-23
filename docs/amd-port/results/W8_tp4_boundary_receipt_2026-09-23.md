# W8 RECEIPT: TP4 BOUNDARY DESK (P0 probe + A1 census + A2/A3 verdicts) - 2026-09-23

Desk: TP4 BOUNDARY DESK (wt-tp4-bound, amd/tp4-bound, based on d143fbb5a).
Mission: the round map's T1 - price the TP4 boundary tax per-op (P0), name
why the count is 137/round (A1), and run the two levers to a verdict:
rung A clustering (A2), rung B count reduction (A3, design only).
Foundation: TP4_roundmap_2026-09-23.md, ledger E-110..E-115a.

## 0. Headline

1. The boundary count is 136.0/die/round, EXACTLY, on all 4 dies
   (29-round exact window): 128 verify + 2 catch-up + 6 draft. The round
   map's 137.2 avg included window-edge effects.
2. The per-boundary cost is NOT uniform: die medians 128.8 / 151.9 /
   191.6 / 210.8 us. The round map's pooled 165.6 us median is a mixture.
   The WALL pays the SLOWEST die: 31.2 ms/round of NCCL kernel time on
   die 1 vs 18.7 on die 3.
3. P0 cost model: enqueue floor 0.2 us (RCCL-1R, kernel elided), launch
   floor ~12 us (size-flat), transport-free 4-rank ring floor 69.5 us at
   80 KB (RING1K single cooperative kernel). The served boundary is
   ring floor + T(die) with T = 59 us (die 3) to 141 us (die 1) - host
   software is a no-op class; the levers are transport and die asymmetry.
4. A2 clustering: NEGATIVE - no legal clustering candidate exists in the
   served graph (boundaries are data-dependent, one tensor each, separated
   by nonlinear consumers; enqueue is already stream-pipelined).
5. A3 count reduction: NEGATIVE for the served plan - the 2-cuts-per-layer
   structure is forced by the sharding table + the PARTIAL cut rule; every
   legal alternative either multiplies weight streaming (the #1 ms pool)
   or keeps the count. Design + anchors below; do not implement.

## 1. P0 - single-die RCCL micro-probe (die 3, fp32, 16-128 KB)

Instrument: docs/amd-port/probes/rccl_tp4_boundary_probe.cpp
(runner run_tp4_boundary_probe.sh: check-and-hold /tmp/campaign_gpu_boot.lock,
cool-die gate < 60 C (max 31.0 C at run), HIP_VISIBLE_DEVICES=3, no server,
no ports; die idle, solo on the machine). Sessions of record:
results/tp4_boundary_probe_d3_20260923_142008.log (RCCL refusal) and
results/tp4_boundary_probe_d3_20260923_142626.log (floors of record).

### 1.1 HARD INSTRUMENT FINDING: RCCL refuses single-die multi-rank

ncclCommInitAll with n >= 2 ranks on the one visible device returns
rc=5 ncclInvalidUsage immediately (the "Duplicate GPU detected" class -
RCCL 2.20.5 enforces one rank per GPU inside a communicator). A
single-die RCCL collective DOES NOT EXIST on this stack, so the mission's
"n=4 ranks on one die through RCCL" is impossible as specified; the
per-op cost curve is measured as floors + model (1.2/1.3), and the
cross-die transport term remains the coordinator's U3 probe.

### 1.2 Measured floors (die 3, 2000 iters, halves spread; the 3% law)

| arm | meaning | 16 KB | 32 KB | 64 KB | 80 KB | 96 KB | 128 KB |
|-----|---------|-------|-------|-------|-------|-------|--------|
| RCCL-1R | single-rank ncclAllReduce (kernel elided) = enqueue floor | 0.26 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 us |
| KLAUNCH n=4 | one kernel + sync = launch floor | 12.0 | 11.4 | 12.4* | 11.6 | 11.7 | 12.0 us |
| RING1K n=4 | one cooperative kernel, 4-rank ring in-device = transport-free algorithm floor | 33.3 | 42.6 | 60.4 | 69.5 | 78.7 | 97.5 us |
| RING1K n=2 | same, 2 ranks | 27.1 | 33.1 | 45.4 | 51.4 | 57.5 | 69.7 us |

*the n=4 64 KB KLAUNCH cell tripped the 3% spread law (12.32%) - reported
as-is, marked FAIL; every other cell passed (worst 2.59%), and the size-flat
KLAUNCH class is confirmed by its neighbors at 0.1-1.1%.

Cooperative launch support: present on gfx900 (grid sync across the
4 rank-blocks; the RCCL kernel shape - ONE launch per collective).

### 1.3 The per-op cost model us(size, n)

  us_enqueue      ~ 0.2 us (RCCL-1R: the host API enqueue is a NO-OP class;
                    even x4 ranks the group enqueue is < 1 us)
  us_launch       ~ 11.5-12 us, size-flat over 16-128 KB (KLAUNCH)
  us_ring(S, n)   = a(n) + b(n) x KB   (RING1K least-squares, r > 0.999)
                    a(4) = 24.1 us, b(4) = 0.573 us/KB   (80 KB: 69.5 us)
                    a(2) = 21.0 us, b(2) = 0.380 us/KB   (80 KB: 51.4 us)
  us_served(die)  = us_ring(80, 4) + T(die)   -> the transport+wait residual:
                    die 3: 128.8 - 69.5 =  59.3 us   die 2: 151.9 - 69.5 = 82.4 us
                    die 0: 191.6 - 69.5 = 122.1 us   die 1: 210.8 - 69.5 = 141.3 us
  draft boundary (20 KB, n=4): ring floor 35.5 us; served median 89.9 us ->
  T ~ 54 us - consistent with the 80 KB residual on the best die.

Reading:
1. Host software cost is ZERO-class: enqueue 0.2 us, launch 12 us - the
   served 128.8-210.8 us per boundary is ~85-95% ring-algorithm +
   cross-die transport + peer wait. There is no launch-tax lever.
2. Even with PERFECT (zero-cost) transport, a 4-rank ring at 80 KB pays
   ~70 us of pure on-device algorithm time on gfx900 - the "boundary tax
   at probe parity" fantasy floor is ~70 us, not the 12 us launch class.
3. The dominant residual T(die) spans 59-141 us across dies on the SAME
   machine and topology - the asymmetry is the lever, not the mean. Any
   RCCL env win (algo/proto/channels) must be judged per-die: the wall
   pays die 1's 210.8 us median, not the pooled 165.6.
4. RING1K is an emulation bound (in-device traffic, no SHM wire, 6 grid
   syncs vs RCCL's pipelined 2-channel kernel) - treat 69.5 us as the
   transport-free FLOOR, not as RCCL behavior; U3 multi-die pins the
   real transport curve.

## 2. A1 - boundary census (zero-GPU, TP4_kernel_census_2026-09-23)

Method: phase-split of the 589k-dispatch kernel trace (all 4 agents,
symmetric 6418 NCCL launches/die). Decode window = last 5.0 s of trace
(traced decode runs ~5 s for the 3.9 s untraced eval, ~+28% rocprof
overhead - inside the usable band per the round map caveat). Rounds
segmented by the T=4 head MMVQ (mul_mat_vec_q t14, 1250-1700 us).

### 2.1 Count

| phase | NCCL/die/round | tensor class | median us |
|-------|----------------|--------------|-----------|
| verify pass (T=4, 65 layers) | 128.0 | 80 KB fp32 (ne 20480) | 129.0 (die 3) |
| catch-up (nextn block, T=4)  | 2.0   | 80 KB fp32 | 125.9 |
| draft 3 x T=1 (nextn block)  | 6.0   | 20 KB fp32 (ne 5120) | 89.9 |
| TOTAL                        | 136.0 | | |

Exact 29-round per-die table (all dies carry exactly 3944 NCCL = 136.0 x 29):

| die | median us | avg us | p90 us | NCCL kernel-time ms/round |
|-----|-----------|--------|--------|---------------------------|
| 0   | 191.6     | 210.3  | 284.3  | 28.60 |
| 1   | 210.8     | 229.7  | 295.9  | 31.23 |
| 2   | 151.9     | 170.2  | 229.3  | 23.15 |
| 3   | 128.8     | 137.2  | 173.2  | 18.66 |

Wall consequence: all dies are lock-stepped by the collective chain, so the
verify wall pays the slowest die's NCCL time: the boundary tax is
31.2 ms/round (die 1), not 22.7-25.6 (the round map priced the pooled
median x count; the wall pays max across dies, not mean).

### 2.2 Why 128 verify boundaries - the mechanism, with anchors

The meta backend cuts the graph exactly at PARTIAL-axis nodes:
- ggml/src/ggml-backend-meta.cpp:2028
  `new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == PARTIAL`
- ggml/src/ggml-backend-meta.cpp:2282-2293: after every subgraph except the
  last, one comm_allreduce call (the 4 per-rank partial copies of the
  subgraph-final node).

PARTIAL is produced by mul_mat over two AXIS_0 (feature-sharded) operands:
- ggml/src/ggml-backend-meta.cpp:590-592 (handle_mul_mat).

The sharding table pins the served model's row-parallel weights by NAME,
independent of device count:
- src/llama-model.cpp:444 attn_output.weight -> AXIS_0
- src/llama-model.cpp:467 ssm_out.weight -> AXIS_0
- src/llama-model.cpp:480-481 ffn_down.weight -> AXIS_0
and the column-parallel companions:
- src/llama-model.cpp:426/432 attn_q/kv/qkv weights -> AXIS_1
- src/llama-model.cpp:472/478 ffn_up/gate weights -> AXIS_1.

Per layer, both layer types (17 full-attn, 48 gated-delta-net) therefore
end in TWO row-parallel matmuls whose activations are feature-sharded by
the preceding column-parallel projection:
- attn: qwen35.cpp:332 (wo) and :476-479 (ffn_down)
- delta-net: qwen35.cpp:463 (ssm_out) and :476-479 (ffn_down).
Each is a PARTIAL node -> one cut + one allreduce. 65 layers x 2 = 130
cut points; measured verify reductions = 128.0/pass. The -2 against the
skeleton is a bounded residual (the delay/merge rule
ggml/src/ggml-backend-meta.cpp:1918-2016 (get_i_delayed) is the in-tree
mechanism that can absorb a boundary; the census cannot name the exact
pair). Cadence check: inter-boundary gaps carry a ~1.2-1.4 ms class
(flash-attn body) exactly every 8th boundary = the [attn, delta, delta,
delta] layer rhythm x 2 cuts - the 2-per-layer structure is visible in
the timeline, not just the code.

The draft/catch-up side: the nextn (MTP) block has the same 2 row-parallel
matmuls -> 2 boundaries per pass; catch-up runs it once at T=4, the three
draft steps at T=1 (20 KB): 2 + 6 = 8. Sum 128 + 8 = 136.0 - matches the
census exactly.

Decode boundary dtype: ne 20480 is inside the n>=4 fp32 band
(n_backends >= 4 && ne < 262144, ggml/src/ggml-cuda/ggml-cuda.cu:1259),
so all decode boundaries are fp32 ring/shm collectives
(ggml-cuda.cu:1264-1276: one ncclGroup per boundary, per-rank
ncclAllReduce, NO host sync inside - the host runs ahead and the
collectives are stream-pipelined behind the layer compute).

### 2.3 Was the count really "tripled" from TP3?

The TP3 of-record figure (~24 cuts/pass, ~48/round, E-085/E-105-era) was
structure math, never a measurement. Evidence from the TP3 census
(census_decode_20260921_224043, llama-bench T=1, butterfly served):
37296 rocclr_copyBuffer dispatches / 4 copies-per-boundary (n=3 fallback:
fold + 2 butterfly + copyback, ggml/src/ggml-backend-meta.cpp:2220-2264)
/ 257 steps = ~36 boundaries/step - same class as the of-record, still
~3.5x below TP4's 128/pass. So the count DID multiply; but the sharding
table above is name-keyed and n-INDEPENDENT, so the divergence cannot
come from the table. Candidate mechanism (flagged, unresolved): the
split-state propagation has T- and n-dependent branches (e.g.
handle_flash_attn_ext asserts src axes AXIS_2,
ggml-backend-meta.cpp:748-755; handle_gated_delta_net has a MIRRORED
fast path, ggml-backend-meta.cpp:769-784) whose resolution differs
between the TP3 (T=1, n=3) and TP4 (T=4, n=4) graphs. The decisive
instrument already exists in-tree: one n=3 NCCL boot with the launch
timeline prints "meta nodes = N, subs = M, bounds = K" per graph
(ggml-backend-meta.cpp:2321-2324) - a one-boot measurement that names
the TP3 subgraph count exactly. LEFT FOR THE COORDINATOR (no boot spent
here; TP4 was the desk's scope).

## 3. A2 - rung A, boundary clustering: VERDICT NEGATIVE (no implementation)

Law: candidates must preserve per-tensor sum order (cluster only
same-tensor boundaries) or be flagged numerics-class-changing.

1. No same-tensor adjacent boundaries exist: every boundary reduces ONE
   tensor (the subgraph-final PARTIAL node,
   ggml-backend-meta.cpp:2285-2293) and its reduced output is consumed by
   the next subgraph (residual add -> RMS norm -> next projection). The
   next boundary is a DIFFERENT tensor produced through the nonlinear
   consumer. There is nothing to batch.
2. Merging DIFFERENT tensors' reductions (concat-allreduce of out-proj +
   ffn partials) is illegal twice over: (a) data-dependence - the ffn
   partial is computed FROM the reduced out-proj result (RMS norm in
   between, qwen35.cpp:332-479), so the second boundary cannot be issued
   early; (b) even where independent, it changes the class the owner
   signed off (sum order / algo selection per size).
3. Host enqueue cost is already amortized: comm_allreduce is a single
   grouped ncclGroup per boundary, no host sync
   (ggml-cuda.cu:1264-1276), issued stream-ordered from the meta executor
   (ggml-backend-meta.cpp:2270-2293). The census gaps between NCCL
   kernels (0.5-0.7 ms) are layer compute on the device, not launch
   overhead.
4. Pricing for the record (P0): the enqueue floor is 0.2 us per op
   (RCCL-1R, section 1.2) - host-side per-op cost is a no-op class;
   there is nothing to cluster away on the enqueue path either.

Conclusion: GGML_CUDA_RCCL_CLUSTER is not implemented - there is no legal
candidate in the served graph to cluster. The env knob would be dead code.

## 4. A3 - rung B, count reduction: DESIGN ONLY, VERDICT NOT FEASIBLE-CLEAN

Goal: eliminate the per-layer cut on the allreduce path (130 -> ~65 or
fewer). Designs evaluated (all file:line anchored):

- B-1a column-parallel out-proj (attn_output/ssm_out -> AXIS_1,
  replicated attention input): removes the out-proj PARTIAL but requires
  every die to hold and stream ALL of q/kv/qkv weights (currently 1/4).
  Weight stream is the #1 ms pool (60.6 ms/round census); +3x qkv bytes
  on 17+48 layers costs tens of ms/round to save ~9-13 ms of boundary -
  NET LOSS. REJECTED.
- B-1b column-parallel ffn_down: same shape, worse - ffn up/gate are
  ~80% of model bytes. REJECTED.
- B-1c reduce-scatter + all-gather (sequence parallel): same collective
  count (2/layer), half the wire bytes per op. At 80 KB latency-bound
  sizes there is no per-op win (bytes are not the constraint; P0 probe +
  the 126.5 us floor show latency dominance), it is T-legal only at
  T=4/4-dies (draft T=1 cannot shard), and it forks the graph structure
  by T (replay/class risk, E-090 T3 lesson). REJECTED for decode; the
  prefill class (10 MiB boundaries, bf16 branch) is a different,
  bytes-bound lever - out of scope here, parked for the coordinator.
- B-1d fold the two per-layer reductions into one: blocked by the RMS
  norm + gating nonlinearity between them (qwen35.cpp:332 attn_output ->
  residual -> :209-class norm -> ffn); E-078 section 3.2 reached the same
  verdict for the nextn block. CONFIRMED.
- B-1e delay the out-proj reduction across the residual ADD (algebraic:
  allreduce(m + p) = m + allreduce(p) for MIRRORED m): legal only until
  the next norm, which sits immediately after the add - saves zero
  boundaries, changes sum order (numerics class) if forced. DEAD.

VERDICT: the 2-per-layer cut structure is ARCHITECTURAL to the sharding
plan at n=4 (same conclusion as E-078 3.2 for TP3). The remaining honest
levers on this tax are transport-class, not count-class:
  (a) die asymmetry: dies 0/1 pay +49%/+64% median vs die 3
      (128.8 -> 191.6/210.8 us). If the critical die were flattened to
      the best-die level: -12.6 ms/round. Candidate: RCCL channel/transport
      mapping (NCCL_DEBUG=INFO fingerprint per die, NCCL_ALGO=Tree vs Ring
      at 80 KB, NCCL_MIN_NCHANNELS) - env-class, needs the coordinator's
      multi-die U3 probe.
  (b) count on the draft side: 8 of 136 boundaries are nextn-block
      boundaries; E-078's accepted-prefix-only catch-up remains the only
      count lever named (KV-rollback reorder risk, NOT mechanical).

## 5. A4 - build + byte-identity

- Probe TU: hipcc -O2 -x hip rccl_tp4_boundary_probe.cpp (gfx900,
  RCCL 2.20.5) - clean, COMPILE-EXIT:0 (P0 log header).
- Full ggml-hip build, canonical flags: GGML_HIP=ON, Release,
  GGML_NATIVE=ON, CMAKE_HIP_ARCHITECTURES=gfx900, GGML_HIP_RCCL=ON,
  LLAMA_CURL=OFF, cmake /home/chris/opt/cmake/bin/cmake.
  [BUILD-RESULT-PLACEHOLDER]
- Default-path byte-identity: NO runtime code was changed (probe + docs
  only) - the served binary and its bytes are untouched by this desk; the
  env-gated A2 knob was deliberately NOT added (dead code, section 3).

## 6. SERVED-ARM SPEC for the coordinator

This desk ships no served arm (no runtime change). What the coordinator's
window should run to cash the boundary findings:

1. U3 multi-die RCCL probe (the desk's P0 isolates single-die software
   cost; the transport curve needs 4 dies): allreduce 16-128 KB fp32,
   n=4 real dies, arms = per-boundary vs grouped-16 vs pipelined-16,
   PLUS NCCL env sweep {NCCL_ALGO=ring,tree} x {NCCL_PROTO=default,ll,ll128}
   x {NCCL_MIN_NCHANNELS=1,2,4}, per-die timings (the asymmetry is the
   finding: watch die 0/1 vs die 3 spread, not just the mean).
   Win bar: critical-die median < 128.8 us class = served delta
   (31.2 - 136 x new_median) ms/round.
2. NCCL_DEBUG=INFO served fingerprint (one boot): per-die connect lines at
   the 80 KB class; compare channels/transport per die against the
   asymmetry table in 2.1.
3. ONE n=3 NCCL boot with the launch timeline
   (ggml-backend-meta.cpp:2321-2324 prints subs/bounds per graph):
   settles the TP3-vs-TP4 count mechanism (section 2.3) and whether any
   legal count recovery exists at n=4. Zero-code, one boot.
4. Any RCCL env arm that moves the served median is numerics-class-safe
   ONLY if the algo/proto change keeps per-element reduce order stable -
   same-sha determinism across two boots per arm is the gate (E-085/E-090
   convention); accept 0.66667 / mean len 3.00 in every cell.

## LOG (append-only, newest last)

- 2026-09-23 desk opened; workspace wt-tp4-bound on amd/tp4-bound at
  d143fbb5a. Ledger E-110..E-115a + round map read. Census phase-split
  done (A1). P0 probe written + queued on the campaign lock (held by
  mmvq-ldsy P0 anchors; single 150 s sleeps).
