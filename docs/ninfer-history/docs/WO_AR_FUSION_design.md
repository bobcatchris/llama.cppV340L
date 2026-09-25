# WO-AR-FUSION — per-layer collective fusion design + parity-gate spec (DESIGN ONLY, no code)

Desk: AR-fusion DESIGN desk, Team Red (AMD V340L, gfx900, NVFP4@TP4), lane `amd/wo-w7-body`,
2026-09-19. Deliverable per coordinator: a design a follow-up implementation desk executes
verbatim — chosen option, exact call-site/phase changes, buffer/stream plan, pre-registered
gates, rollback, effort. **This desk edits NO src, builds NOTHING, touches NO GPU.**

Companion premise rows (all banked, this box):
- PLOG-062: honest in-serve prefill AR column = **123.0/123.3 ms/chunk** (OPTRACE ar, cure bin
  2c8901d3d18adef1), chunk wall 978-992 ms -> AR = 12% of the prefill wall.
- PLOG-063: rocprof on the cure bin — nccl ARs **0.0% overlapped** with GEMM on all four dies;
  single in-order stream; overlap is structurally impossible. AR reduction must be FEWER BYTES
  or FEWER COLLECTIVES; scheduling buys zero.
- PLOG-065: RCCL env matrix NULL RESULT (16ch/Tree/LL128 all no-fire) — the 4x-gfx900 SHM ring
  at ~1.31-1.5 MB bf16 is **host-SHM-bandwidth-bound, env-immovable** (646-771 us isolated,
  ~961 us mean in-serve per collect). The algorithmic-class door is the only one open.
- PLOG-066 + W7_roundwall_row: decode round 59.2 ms cold p50; in-loop AR **6-9.6 ms/round**
  (64 verify collects, 30 KiB each = 3x5120xbf16, ~150 us/collect hot, ~94-125 us cold);
  host-additive 0.21 ms/round (NOT an AR item); Branch-B (one_shot_allreduce.cu deferred
  status) sized at 3-4.5 ms/round saving, owned by the parallel W4AR desk.
- GATE-T (W7_ar_transport_run{1,2}.log): host-staged one-shot vs RCCL ring, world=2 dies 2,3:
  10 KiB H/R 0.39x, 30 KiB 0.47x (pipelined 0.31-0.47x), **1.31 MB H/R 1.19x (H LOSES at
  prefill size)**; capture/replay liveness GREEN (32-collect, zero wedge).
- PLOG-060: ordinal-paired fresh-boot A/B is the controlled measurement design (within-pair
  +-2%); PLOG-064: every soak/decode absolute carries its start temperature.

---

## 1. GROUNDED DATAFLOW AUDIT (read from the tree, not from the mission summary)

### 1.1 Layer census and AR count
`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/hybrid_topology.h`: `is_full_attention(layer)
= (layer+1) % 4 == 0` -> at 64 layers: **16 full-attention + 48 GDN**. Per layer exactly TWO
ARs: mixer-out AR (o_proj at full-attn layers, out_proj at GDN layers) + mlp down AR
(mlp_tail). In-code census (text_context_impl.h, GDN prefill arm comment): **128 per-chunk
allreduces** = 16 o_proj + 48 gdn-out + 64 mlp. The mission's "~125" is the same census within
rounding; the PLOG-062 ar column (123 ms) is TIME, so in-serve mean = 123.0/128 = **961
us/collect**.

### 1.2 Chunk geometry and message shapes
- `kPrefillChunkAlignment = 128` (text_context.h:83); plen 2075 = 16x128+27 -> every prefill
  chunk is T=128 (tail chunk T<=128), hidden = 5120.
- Prefill AR message: 5120 x 128 x bf16 = **1,310,720 B (1.31 MB)**, n_elems = 655,360.
- Decode verify AR: 5120 x 3 (T=3 MTP verify) x bf16 = 30 KiB, n_elems = 15,360.
- MTP draft head ARs (:1650/:1659, :1984/:1996): 5120 x T, T in {1,3,128}; at prefill-chunk T
  they are full-size 655,360-elem collectives (2 per chunk) — they ride the same dispatch as
  everything below.

### 1.3 Transport dispatch — the ONE fact that shapes the whole design
`src/core/multi_gpu/one_shot_allreduce.h:17`: `kMaxElements = 65536` (128 KB/slot, "covers up
to T=12"). `tp_group.cpp:381-398 allreduce_local_bf16` routes:
- `n_elems <= kMaxElements` AND one-shot constructed (NINFER_TP_ONESHOT_AR, world gate) ->
  **host-staged one-shot** (pinned publish -> threadfence_system -> gen/flag -> peer NT loads ->
  fp32-class combine -> bf16 out -> **residual fold epilogue** `__hadd2(x, __hadd2(l,p))`, the
  W2a fold). DECODE (15,360 elems) RIDES THIS TODAY.
- otherwise -> **NCCL ring** in-place + separate `one_shot_axpy_bf16` residual fold (2 kernels).
  **ALL PREFILL COLLECTIVES (655,360 elems > 65,536) RIDE THE NCCL SHM RING TODAY**, and pay
  128 extra axpy kernels per chunk for the residual fold.
- The residual fold is ALREADY fused into the collective on the one-shot arm. There is no
  remaining "fold the residual" win on decode; on prefill it is a free side-effect of any
  custom-transport change (Option C), worth ~0.2-0.3 ms/chunk (128 axpys x ~1.5-2.5 us).

### 1.4 The dependency chain (code-level, the legality arbiter)
Full-attn layer (prefill arm, `text_context_impl.h`):
```
x (residual, replicated)                          # mixer entry (coh_dump("X0", x) comment)
  gqa_attention -> sigmoid_mul -> a               # :2689-2697
  partial = o_proj(a)          [RowK-sharded]     # :2708-2712
  AR#1: allreduce_local_bf16(partial -> x)        # :2714-2719   x += sum_r partial_r
  h = rmsnorm(x, post_attn_norm)                  # mlp_tail :3307  <-- NONLINEAR SEAM
  partial = post_mixer_tp(h)   [MoE, rank-local]  # :3323-3330
  AR#2: allreduce_local_bf16(partial -> x)        # :3331-3335   x += sum_r partial_r
```
GDN layers identical shape (delta-net -> gated_rmsnorm -> out_proj partial -> AR#1' at :3120-
3131 prefill / :2913-2922 decode). MTP head identical (norm between its two ARs, :1650/:1659).

**Every AR#2 operand is a function of AR#1's output through an RMSNorm.** On the single
in-order stream there is no slack anywhere around either collective (PLOG-063).

### 1.5 Kernel internals that the design must respect (one_shot_allreduce.cu, current tree state)
- Launch: `<<<1, 1024>>>` — a SINGLE CTA does publish + poll + combine. This is why H loses at
  1.31 MB (GATE-T 1.19x): one CTA's LSU/PCIe issue rate (~10 GB/s effective over the ~7.9 MB
  of publish+load traffic) is the ceiling, not the wire.
- Publish choreography: device -> pinned host_buf via `st_writethrough_uint4`;
  `__threadfence_system()`; then gen, then flag (volatile dwords; the G-AMD-30a measured
  transport — atomic RMW does NOT cross host-mapped memory on this box, status_transport.h).
- Consumer: flag+gen gate -> `ld_uncached_uint4` peer loads -> combine -> write out + residual
  fold. Slot = `step % kNumSlots` (128 slots x world buffers x 128 KB = 64 MB pinned/rank).
- **Branch-B is landing RIGHT NOW, uncommitted, in this file**: `NINFER_AR_DEFER_STATUS` arm
  (defer_status gate + the `break` single-shot path ~:1035-1048 replacing the per-call
  `cudaStreamSynchronize` at :1050). The world=2 path at :1010-1026 is marked VERBATIM
  (bit-frozen, NVIDIA line). **This design adds code around that region and edits nothing
  inside it.** Sequencing law for the implementation desk: land on top of Branch-B's committed
  state; rebase if it moves; never touch :1000-1060 sync/status logic (composition matrix in
  §8 GATE-Q1c).
- gfx900 has NO native fp8 convert instructions (v_cvt fp8 class is gfx950+); any fp8 leg is
  SOFTWARE encode/decode (bit math or a 256-entry LDS LUT). Cost is trivial vs wire time but
  it must be written, not assumed.

---

## 2. OPTION A — fuse / restructure the two per-layer ARs: LEGALITY MATRIX

The hypothesis from PLOG-065 consequence (a) was "fuse mixer+mlp collectives = halve AR count".
Priced honestly against §1.4:

| # | restructure | verdict | why | ms/chunk |
|---|---|---|---|---|
| A-i | one fused collective per layer (AR#1+AR#2) | **ILLEGAL** | AR#2's operand = rmsnorm(AR#1 output) -> MoE GEMMs. The fused collect needs both payloads enqueued; the mlp partial does not exist until AR#1 completes. On the in-order stream the fused collect either waits mid-collect (serialization unchanged, strictly worse: attention payload waits for the mlp GEMMs) or reads garbage. Wrong-results/deadlock class. | 0 (cannot ship) |
| A-ii | batch AR#2(L) with AR#1(L+1) across the layer boundary | **ILLEGAL** | x entering layer L+1 (and its input_norm) is a function of AR#2(L)'s output. Same cycle class as A-i one level up. | 0 |
| A-iii | accumulate mixer-partial + mlp-partial into one buffer pre-collective ("where the math allows") | **ILLEGAL — the math allows NOWHERE** | the seam between the two ARs contains rmsnorm (nonlinear) + MoE routing/activation. sum_r mlp(partialmix_r) != mlp(sum_r partialmix_r) under any norm; and the MoE gate/up weights are output-sharded so each rank cannot even form the others' mlp pieces. | 0 |
| A-iv | eliminate AR#1: feed mlp the rank-local partial mixer output | **ILLEGAL** | norm nonlinearity (same as A-iii) AND sharded-weights infeasibility. | 0 |
| A-v | reduce-scatter + all-gather (sequence-parallel Megatron restructure) | LEGAL but **worthless here** | ring AR is literally RS+AG; wire bytes identical (2(N-1)/N x S); PLOG-065 says bytes are the binding term. All cost, no gain. | ~0 |
| A-vi | batch the last AR of chunk N with the first AR of chunk N+1 (cross-chunk; the residual streams of different chunks ARE independent) | LEGAL, **not worth it** | only 1 boundary pair per chunk pair; saves only the per-collect fixed cost (~50-100 us in-serve) -> ~0.1 ms/chunk against new cross-chunk queue coupling. | ~0.1 |
| A-vii | residual-add folded into the collective epilogue | LEGAL, **already done on decode; free side-effect on prefill** | one-shot arm has the W2a fold; prefill's NCCL arm pays a separate axpy. Any custom prefill transport absorbs it. | 0.2-0.3 (only as C's side-effect) |
| A-viii | "batch across the 4 ranks' message layout" (one 2S collect instead of two S collects) | **ILLEGAL within a layer (A-i); and even where legal it saves only fixed cost** | ring wire bytes of one 2S collect = same as two S collects (1.5 x 2S both ways); SHM is byte-bound, not count-bound, at 1.31 MB. | ~0 |

**OPTION A VERDICT: the count-halving hypothesis is FALSIFIED at design level.** The two
per-layer ARs sit on opposite sides of a nonlinear seam and cannot merge; every same-depth
batching position is likewise dependency-closed. Legal content of A = the residual-axpy fold
(~0.2-0.3 ms/chunk) which arrives free inside Option C. A standalone A desk ships 0 ms.
Decode side: A = 0 ms (fold already fused; the 64 verify collects are already one-per-layer).

## 3. OPTION B — quantized AR payload (bf16 -> fp8e4m3 + per-rank scale)

Premise: the collectives are byte-bound on host SHM (PLOG-065), so **wire time scales with
payload bytes**. Halve the published bytes -> halve the bandwidth term. The AR payload is a
rank-partial (pre-residual o_proj/down_proj output) — exactly the tensor class quantized
allreduces are designed for: the dequant-sum can accumulate in fp32 and the residual stream x
STAYS bf16 full precision; quantization noise enters only through the summed contributions.

### 3.1 Scheme
- Payload: **fp8e4m3** (4-bit exponent, 3-bit mantissa, max 448, saturating convert) over
  **int8-class storage** (1 B/elem). Chosen over int8-linear: e4m3 carries ~2^8 of relative
  dynamic range inside one scale, which fits partial tensors with outliers without crushing
  small elements; int8's uniform step gives small elements proportionally worse error.
- Scale: **per-rank, per-collect, per-tensor** s_r = amax(|partial_r|) / 448, computed ON
  DEVICE by an amax pre-pass kernel (grid-stride block-reduce into a device scalar). Each rank
  publishes ITS OWN scale beside its payload — dequant multiplies each peer contribution by
  its own s_r, so heterogeneous scales across ranks are mathematically exact per payload.
- Publish: quantize bf16->fp8 (software encode, §1.5), write-through into the slot's pinned
  buffer as u32x4 (4 bytes/thread-iter instead of 8 — same uint4 128-bit stores, double
  element count); publish order **payload -> fence -> scale word -> gen -> flag** (scale INSIDE
  the existing fence chain, BEFORE the flag, exactly the ARP parity-word slot pattern — the
  gen-gate then proves the scale belongs to this call's generation).
- Combine: after both/N gates pass, each thread loads peer fp8 words, decodes via bit-math
  (or 256-entry LDS LUT) to fp32, multiplies by the peer's scale, accumulates fp32, converts
  once to bf16, folds residual — the W2a epilogue shape verbatim (`__hadd2(x, sum)`).
- Error model (for the gate bars, not as a claim): per-element quant RMS ~ s_r x 2^-4/sqrt(12)
  x |p|; four-rank fp32 dequant-sum keeps it at that class; RMSNorm downstream re-normalizes
  but the residual stream accumulates 128 draws per chunk — this is exactly what GATE-Q2's
  paired acceptance bars adjudicate. No theory row substitutes for the A/B.

### 3.2 Prefill capacity — the prefill ring (NEW instance, not a kMaxElements bump)
The existing 128-slot decode ring cannot carry 655,360 elems (slot = 128 KB bf16). Do NOT
raise kMaxElements globally (128 slots x 4 ranks x 640 KB fp8 = 320 MB pinned — absurd).
Construct a SECOND OneShotAllReduce instance with runtime {slots, capacity}:
- `capacity = 655,360` (one prefill chunk x 5120, fp8 -> 655 KB/peer buffer),
- `slots = 8` (prefill ARs are stream-serialized; 8 slots cover any conceivable skew;
  decode's 128 exists because decode retried in-place — Branch-B's deferred arm needs the
  same anti-recycle law, 8 is its prefill analog),
- pinned cost: 8 x 4 x 655 KB ~= **21 MB host-pinned per rank** (84 MB total; host RAM, not
  VRAM — no VRAM-law surface, and the allocator is the only gate: cudaHostAlloc failure ->
  log + instance stays null + NCCL fallback serves the chunk. NO REFUSAL PATH, VRAM law).
- Own rank_gen/rank_step ring (independent call identity); `advance_one_shot_epoch` /
  `reset_one_shot_step` forwarding extended to it (tp_group.cpp:446-460 shape).

### 3.3 Decode leg — priced NO-GO (state the arithmetic, park it)
At 30 KiB the one-shot collect is latency-dominated: GATE-T pipelined world=2 30 KiB = 21.7 us
H, of which the bytes term is ~10-12 us. fp8 halves that term (~5 us saved) but adds an amax
pass + quant pass (~2-3 us each at n=15,360) on the critical path of EVERY collect: net
~0 +- 1 ms/round — noise against a 3-4.5 ms Branch-B prize, bought with per-token numerics
risk. **Decode rides Branch-B only.** Re-fire clause: if Branch-B's serve A/B lands worse
than 0.55x in-serve ratio, GATE-Q3 (§8) re-opens the decode-quant question with the same
cells.

## 4. OPTION C — quantized AR + the legal residue of A
= Option B (prefill fp8 ring) + the residual-axpy fold absorbed into the quant combine
epilogue (kills 128 axpy kernels/chunk, ~0.2-0.3 ms/chunk) + nothing else from A (§2 matrix).
There is no additional C content beyond B; the recommendation below is therefore "C",
implemented as B-plus-fold.

## 5. PRICING (expected savings, honest bands, both sides)

| option | prefill ms/chunk (of 123) | prefill wall (of ~990) | decode ms/round (of 59.2) | verdict |
|---|---|---|---|---|
| A standalone | 0 legal (0.2-0.3 as fold only, not standalone-shippable) | ~0 | 0 | REJECT standalone |
| B (prefill fp8 ring) | **-49 to -73** (single-CTA projection: 128 x ~390-415 us = 50-53 ms; band widened for in-serve straggler skew, die spread 101-141 ms) | **-5 to -7.5%** | 0 (decode NO-GO by §3.3) | the prize |
| B stage-2 (multi-CTA publish/combine, per-CTA completion words) | to ~-95..-105 (toward the ~25 GB/s staging ceiling: ~160-200 us/collect) | to ~-10% | n/a | upside tier, separately gated |
| C = B + fold | B's numbers + 0.2-0.3 | B + ~0.03% | 0 | **RECOMMENDED** |
| decode fp8 (any option) | n/a | n/a | ~0 +- 1 (savings eaten by added passes) | PARKED (GATE-Q3 clause) |

Projection receipt chain: 771 us measured H bf16 at 1.31 MB (7.9 MB staging traffic) = 10.2
GB/s effective single-CTA; fp8 halves traffic -> ~385-400 us + ~5 us amax + ~10 us quant-store
slack; in-serve ring baseline 961 us/collect x 128. The band's LOW end assumes in-serve skew
hurts the one-shot as much as it hurts the ring; GATE-Q0/GATE-Q2 decide, the design promises
only the GATE-Q2 bar (-30% ar column), not the projection's best case.

## 6. RECOMMENDATION

**Option C — prefill-only fp8e4m3 quantized host-staged one-shot ring (new big-slot instance),
with the residual fold absorbed into its combine epilogue; decode unchanged (rides Branch-B);
decode-quant parked behind GATE-Q3.**

Why: it is the ONLY priced option that moves the 123 ms column (A is dependency-dead — §2);
it rides the byte-bound mechanism PLOG-065 left open; it composes with Branch-B (additive
kernel/route code, zero edits to the :1000-1060 region); its risk is bounded by env-gated
construction + the paired-A/B parity gates; and it leaves the NCCL fallback as the
always-available rollback path inside the same binary.

---

## 7. IMPLEMENTATION SPEC (file:line level; the implementation desk executes this verbatim)

Sequencing law FIRST: **land on top of Branch-B's committed one_shot_allreduce.cu state.**
Branch-B owns the defer_status region (~:1000-1060 current line numbers will shift). This work
order's edits are ADDITIVE around it; any hunk touching the status/sync/slot logic of the
existing 2-rank or world paths is out of spec and reviewers reject it.

### 7.1 `src/core/multi_gpu/one_shot_allreduce.h`
1. Keep `kNumSlots = 128`, `kMaxElements = 65536` untouched (decode ring's constants; the
   comment at :17 stands). Add runtime-capacity construction:
   `explicit OneShotAllReduce(int world, std::size_t n_slots, std::size_t max_elements,
                              bool fp8_payload);`
   Default-arg-compatible with the existing ctor — existing callers compile unchanged and the
   decode instance stays byte-identical (argmax-parity "launch byte-identical when unarmed"
   precedent law).
2. Impl gains `n_slots`, `capacity`, `fp8` fields; the fixed `Slot slots[kNumSlots]` C array
   stays (Slot is pointers-only; ctor loop allocates only the first `n_slots` — empty Slot
   structs cost nothing). New per-slot pinned words: `float* scale[ /*world*/ ]` allocated
   beside `flag` in the same cudaHostAlloc pattern (AR_PARITY_ARM §1 storage precedent:
   ~n_slots x world x 4 B = 256 B for the prefill ring).
3. Declare the two new device entry points used by tp_group: `amax_bf16(rank, ptr, n, stream,
   out_dev_scalar)` and the fp8 allreduce (see 7.2).

### 7.2 `src/core/multi_gpu/one_shot_allreduce.cu` (new code ONLY; :1000-1060 untouched)
1. `ar_amax_bf16_kernel<<<kBlocks, 256, 0, stream>>>`: grid-stride max(|x|) over the bf16
   partial -> one device fp32 scalar in the impl's per-slot device scratch. kBlocks 8-16
   (1.31 MB read, ~3-5 us). Capture-safe: pure kernel, no host sync.
2. `one_shot_ar_fp8_kernel_world` (mirrors `one_shot_ar_pinned_vec_kernel_world`; the world=2
   bf16 kernel is NOT duplicated — decode keeps its VERBATIM path, prefill is world=4 only):
   - pass A (fused into the publish kernel; reads the amax scalar): compute
     `s = amax/448f`; software-encode bf16->e4m3 (no native instr on gfx900 — bit-encode or
     shared 256-entry LUT); publish fp8 as u32x4 write-through (2x elems per 128-bit store);
     `__threadfence_system()`; store `s` to `slot.scale[rank]`; then gen, then flag (order
     EXACTLY as the bf16 kernel's payload->gen->flag chain; scale sits inside the fence,
     before flag).
   - pass B (after the N-gate): NT-load peer fp8, decode to fp32 (bit-decode or LUT), FMA by
     `slot.scale[p]` into fp32 accumulator, convert once to bf16 RN, write out, fold residual
     (`__hadd2(x, sum)` — W2a epilogue shape verbatim).
   - Status/timeout/retry behavior: identical constants (`kFlagTimeoutPolls`), identical
     status bits, identical diag rows. The B3 heartbeat needs no change (the wrapper in
     tp_group.cpp stamps it around whatever route runs — verify this reading at
     implementation time; it is the :388-397 wrapper).
3. `allreduce_bf16` entry: add a capacity check against the INSTANCE's capacity (not the old
   constant) and branch to the fp8 kernel when `impl_->fp8`; the host-side loop structure,
   gen/slot arithmetic and defer_status arm are shared code — the fp8 branch changes ONLY
   launch args (+PeerArgs.scale[p]) and the kernel symbol. PeerArgs gains `float* my_scale;
   const float* peer_scale;` (tail nullptr at world=2, never read — same fixed-capacity
   pattern as today).
4. The prefill ring instance NEVER runs the defer-status region differently — it inherits
   whatever Branch-B landed, which is the point of the composition cell (GATE-Q1c).

### 7.3 `src/core/multi_gpu/tp_group.cpp` + `tp_group.h`
1. TpGroup ctor, after the existing one-shot construction block (:160-168): read-once static
   `NINFER_AR_QUANT_PREFILL` (same getenv pattern as `tp_oneshot_ar_gate()`); when set AND
   world>1: `I.quant_prefill_ring = make_unique<OneShotAllReduce>(I.n, 8, 655360, true);`
   cudaHostAlloc failure at any slot -> free partial, set null, `fprintf(stderr, "[AR-QUANT]
   pinned alloc failed rank=%d — NCCL fallback serving")` -> NCCL path serves. NO THROW, NO
   REFUSAL (VRAM-law shape: the allocator is the gate, in real time).
2. `allreduce_local_bf16` (:381): after the heartbeat-entry stamp and BEFORE the one-shot
   check, insert the route:
   `if (impl_->quant_prefill_ring && n_elems > OneShotAllReduce::kMaxElements) {
      impl_->quant_prefill_ring->allreduce_bf16(rank, local_ptr, n_elems, k.ctx.stream,
                                                residual_ptr); if (hb) exit; return; }`
   Everything else (one-shot check, NCCL+axpy fallback) is byte-identical. Net effect: ALL
   prefill collectives (655,360 elems, including the MTP head's chunk-width ARs) route to the
   quant ring when armed; decode (<=65,536) untouched; fallbacks unchanged.
3. `advance_one_shot_epoch` / `reset_one_shot_step` (:444-460): forward to
   `quant_prefill_ring` when present (request-boundary epoch/step hygiene, same as argmax).
4. Header: +`std::unique_ptr<OneShotAllReduce> quant_prefill_ring;` member.

### 7.4 Call sites — NONE
`text_context_impl.h` sites :1650/:1659/:1984/:1996/:2717/:2918/:3127/:3333 all funnel through
`allreduce_local_bf16` and need ZERO edits. This is the design's central property: routing is
central, parity of the unarmed path is trivially byte-identical.

### 7.5 Cells and benches (new files; no src edits)
- `tools/v340l/w7_ar_quant_bench.cu` — GATE-Q0 microbench (world=2 on dies 2,3): arms
  {R ring bf16, H one-shot bf16 (GATE-T reproduction), H one-shot fp8} x sizes {655 KB fp8 /
  1.31 MB bf16 payload, 15 KB fp8 / 30 KiB bf16}; correctness vs CPU fp32 reference with the
  §8 error bars; graph-replay leg (32-collect, byte-check per replay — GATE-T's proven shape).
  Build per the desk pattern (`/opt/rocm/lib/llvm/bin/clang++ --offload-arch=gfx900
  -I src/common/hip_dev_shim -O3`), BIN BANKED before any run.
- Host-model cell (CI-farm + PG-1 joinable): fp8 encode/decode reference + error-bound +
  saturation + torn-scale/wrong-gen falsifiers (GATE-Q1a/b, both directions).
- Device cells join the per-window BOOT_BATTERY under the runbook's LABEL law once promoted.

## 8. PRE-REGISTERED GATES (numbers + bars, written BEFORE any bench; firing order fixed)

- **GATE-Q0 (transport microbench, the stage-1 decider).** World=2, dies 2,3, <2 min,
  tagged in the implementing desk's log per the window law; box state + sclk annotated
  (M0 law). FIRE stage-1 (serve A/B) iff: fp8-H mean per-collect <= **0.55x** R-ring-bf16 at
  the 1.31 MB class (the byte halving showing through the fixed costs) AND <= **0.75x** H-bf16
  at the same size, reproduced +-2% in two adjacent runs, graph-replay leg GREEN (byte-equal
  outputs across replays; dequant outputs within GATE-Q1a bars vs CPU ref). Marginal
  (0.55-0.75x R) -> bank, no serve window. >0.75x R -> fp8 transport NO-GO, park everything.
  HONEST SCOPE (GATE-T precedent): world=2 is a REDUCED model of the serving world=4; the
  world=4 decisive cell is GATE-Q2 — Q0 only kills clearly-dead transports cheaply.
- **GATE-Q1 (correctness cells, closure-law triple).**
  (a) error-bound cell: random + adversarial payloads (outliers, cancellation) —
      `L2_rel(ar_fp8, ar_fp32) <= 3.0e-2` AND `max_abs_err <= max_r(s_r) * 2^-3 * 1.01`;
      RED direction: injected torn-scale word and stale-slot wrong-gen payload MUST fail the
      cell (both-direction falsifier). GREEN row names both shas (pre-cell artifact + cell).
  (b) saturation cell: scale-floor/amax-infinity payload -> finite output, no NaN propagation
      (e4m3 saturates; assert).
  (c) compose cell: `NINFER_AR_DEFER_STATUS=1` x quant armed — 32-collect graph replay, zero
      wedge, byte-equal per replay (Branch-B's own proven cell shape, now on the fp8 path).
  Cells join the standing suite permanently (host-only parts in CI farm + PG-1; device parts
  in the boot battery).
- **GATE-Q2 (serve A/B — the decisive prefill cell).** PLOG-060 ordinal-paired design: TWO
  fresh boots, same bin, arm A `NINFER_AR_QUANT_PREFILL=1`, arm B unset; identical life-story
  per arm (boot -> warmup -> 3x 2k-class probes, plen-2075, OPTRACE armed for the ar column;
  start temperature noted per PLOG-064). FIRE/PROMOTE iff, at EVERY ordinal:
  ar column **-30% or better** (projection target -50%; the bar is the promise),
  wall **-3% or better**, within-pair agreement +-2%, parity CLEAN (all probes finish=stop
  BLUE-class, no mojibake), MTP acceptance within **-3% relative** of the paired arm (the
  2k-probe acceptance is the logits-path bar), tok/round within +-2%. Then: BOOT_BATTERY
  GREEN + one 10k soak with start-temp noted, acceptance >= baseline -3% relative (0.71-class
  -> >=0.689), prefill within the thermal band accounted. PROMOTION = runbook BIN flip +
  env line added, per the BOOT_LAUNCH_RUNBOOK pattern (env is the serving config of record).
- **GATE-Q3 (decode-quant re-fire, optional, parked by default).** Opens ONLY if Branch-B's
  serve A/B lands an in-serve H/R ratio worse than 0.55x. Same cell family, decode sizes;
  bar: paired round **-2% or better beyond Branch-B's own arm** + acceptance unchanged
  (+-0.01 abs). Otherwise stays parked (the §3.3 arithmetic).

## 9. ROLLBACK / KILL-SWITCH

- Primary kill: unset `NINFER_AR_QUANT_PREFILL` -> the instance is never constructed, every
  prefill collective takes the exact current NCCL+axpy path, byte-identical binary behavior
  (env read-once at ctor; the armed/unarmed duality is the argmax-parity precedent law).
- Within-bin fallback: pinned-alloc failure or any future fault -> null instance -> NCCL
  fallback serving (no refusal path anywhere; VRAM-law clean — no estimated-charge gate was
  added by this design and none may be added by the implementation).
- Bin rollback: bank-before-relink per BOOT_LAUNCH_RUNBOOK §4 — the A/B boots a BANKED
  artifact (`/home/chris/artifacts_bin/<name>_<sha16>.bin`), rollback = previous banked BIN
  line in the runbook (one-line flip), prior canonical 2c8901d3d18adef1 / c618d356f0cdc401
  lineage per the boot-incident ledger in PLOG-066.
- Stage-2 (multi-CTA) is behind its own env bit (`NINFER_AR_QUANT_CTAS`, default 1) so the
  promoted stage-1 posture can always be narrowed without a rebuild.

## 10. RISK REGISTER

- R1 numerics drift (acceptance/prose quality): the whole reason GATE-Q2's acceptance bars
  exist; fp8 partial-AR is production practice (TE-class) but THIS model's 128x/chunk cadence
  is the untested variable. Mitigation: env kill + paired bars; no partial arming.
- R2 single-CTA bandwidth ceiling worse than linear-bytes projection: GATE-Q0 kills it before
  any serve window is spent; stage-2 multi-CTA is the named escape (per-CTA completion words —
  new sync choreography, separately celled, not in stage-1 scope).
- R3 torn-scale word (NEW correctness class this design introduces): scale published inside
  the fence chain, before flag, consumed only past the gen-gate; GATE-Q1a's RED direction
  convicts the class, not the instance.
- R4 graph-capture interaction (NINFER_PREFILL_GRAPH arm exists): amax+publish are pure
  kernels on one stream — capture-safe by construction; GATE-Q1c replays the composed shape.
- R5 Branch-B file collision: sequencing law §7 (land on their committed state; :1000-1060
  untouchable). If Branch-B slips, this desk queues behind it — the designs are independent
  and the compose cell proves coexistence.
- R6 host-pinned growth +21 MB/rank: host RAM, not VRAM; no law surface; allocator-gated.
- R7 epoch/step forwarding bug (double-reset or missed reset) would desync the prefill ring's
  slots: covered by extending the EXISTING forwarding functions (§7.3.3) and by GATE-Q1c's
  replay (a desync shows as stale-flag timeout within kFlagTimeoutPolls, loud per the
  AR-FAILOUT law).

## 11. EFFORT ESTIMATE (desk-hours)

- Stage-1 code (§7.1-7.4: ctor/fields, 2 kernels + software fp8 codec, PeerArgs scale words,
  routing, forwarding): **6-8 h**.
- Cells + benches (§7.5: GATE-Q0 bench, host-model cell suite, compose cell): **2-3 h**.
- Windows: GATE-Q0 ~15 min (one <2 min bench + build/bank); GATE-Q2 ~60-75 min (two boots x
  boot+warmup+3 probes, per the V-arm desk's ~25-30 min/arm measured budget); battery + soak
  ~30 min. Call it **2 windows**.
- **Stage-1 total: 10-12 desk-hours to a banked promote/no-promote verdict.**
- Stage-2 multi-CTA (only if GATE-Q0b prices it after stage-1 fires): +6-10 h (partitioned
  publish/combine, per-CTA completion words, tear-class re-cells, its own A/B).
- Decode fp8: parked (0 h unless GATE-Q3 opens).

## 12. PROGRESS LOG (append-only; checkpoint law — this file is the desk's checkpoint)

- [step 1] 2026-09-19 — Desk opened as AR-fusion DESIGN desk, lane amd/wo-w7-body. Probe:
  rocm-smi 4x "Vega 10 [Radeon Pro V340" -> AMD line confirmed, docs/amd/ rules apply.
  Read: W7_ROUNDWALL_desk.md (GATE-T/GATE-K + window forensics), W7_roundwall_row.txt
  (full round table + cold-band amendment), PERF_LOG PLOG-060..066, W7_rccl_matrix_row.txt
  (null result + transport receipt), AR_PARITY_ARM_design_agent5.md (parity-word slot
  precedent + closure bar). No GPU acts; parallel desk owns the window (dies 2/3 untouched).
- [step 2] 2026-09-19 — Source recon (read-only): AR call sites enumerated
  (text_context_impl.h :1650/:1659/:1984/:1996/:2717/:2918/:3127/:3333 — mission's four plus
  the MTP-head pair); census 16 full-attn + 48 GDN = 128 ARs/chunk (hybrid_topology.h
  interval-4 + in-code "48 of the 128" comment); chunk=128 (kPrefillChunkAlignment); W2a
  residual fold already fused in the one-shot epilogue; transport dispatch = kMaxElements
  65536 (one_shot_allreduce.h:17) -> ALL prefill collectives (655,360 elems) ride NCCL+axpy
  TODAY, decode rides the host-staged one-shot; kernel is single-CTA <<<1,1024>>> (the
  measured H-loss mechanism at 1.31 MB); Branch-B's uncommitted defer_status arm read in
  place (:1000-1060 region) — composition constraints written into §1.5/§7.
- [step 3] 2026-09-19 — Options priced. Option A: legality matrix §2 — straight fuse and
  every same-depth batching ILLEGAL (RMSNorm+MoE seam between the ARs; code-level chain §1.4);
  count-halving hypothesis of PLOG-065(a) FALSIFIED at design level; legal content = axpy
  fold only (~0.2-0.3 ms/chunk, rides C). Option B: fp8e4m3 + per-rank per-tensor device
  amax, new 8-slot/655,360-elem prefill ring instance (+21 MB pinned/rank), decode priced
  NO-GO (~0 +-1 ms/round) with GATE-Q3 re-fire clause. Pricing §5: prefill -49..-73 ms/chunk
  (bar -30%), wall -5..-7.5%; decode 0 (Branch-B's 3-4.5 is the parallel desk's prize).
- [step 4] 2026-09-19 — Work order completed: recommendation C (= B + fold, prefill-only),
  implementation spec §7 (file:line, zero call-site edits, Branch-B sequencing law), gates
  GATE-Q0..Q3 pre-registered §8, rollback §9, risks §10, effort 10-12 desk-hours stage-1 §11.
  DESK DELIVERABLE COMPLETE — hand-off to the implementation desk; next desk act per
  ping-pong: await coordinator tasking (this desk holds no window, built nothing, touched
  no GPU).

## ADDENDUM (2026-09-19 ~09:4x, principal benchmark landed): THE RACE CONTEXT — this work order is now the race-deciding desk

Principal benchmarked llama.cpp on IDENTICAL hardware (4x gfx900, nerd-dell): 27B dense Q4_1
-sm tensor = pp512 159.66 / pp2048 152.09 t/s; tg 19 t/s. Our stack same box: ~112 t/s
prefill (18.4 s / 2075 tok), decode 27-41 t/s (MTP). Decomposition: GEMM rates at PARITY
(theirs 2.18 TF/s/die end-to-end = our GEMM-only 2.0-2.27); the ENTIRE deficit is our TP
overhead (ar 123 + body 43-74 + gap 15-69 ≈ 230 ms/chunk vs their ~free copy-add sum).
Decode: WE WIN 1.4-2x (MTP) — adoption of llama.cpp would cost decode for prefill.

CONSEQUENCE: this desk's implementation is the race-deciding landing. ADD one more option to
price in GATE-Q0: option 0 = llama.cpp-style copy-add partial sum (each rank copies its
partial to the reduce root over pinned transfers + the root adds; no NCCL, no quant) — if it
beats the 990 us ring at 1.31 MB, it is the numerics-free base case and fp8 rides on top.
llama.cpp reference build: nerd-dell ~/llama.cpp @ 7775f6e62, ROCm backend, Q4_1 GGUF,
-tensor split across 4 dies. Our target once this lands: >=150 t/s 2k prefill with decode
unchanged >=27 t/s.
