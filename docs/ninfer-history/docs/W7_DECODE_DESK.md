# W7 DECODE DESK — web-findings audited against OUR decode stack (2026-09-18)

Seat: decode-optimization desk, lane `amd/wo-w7-body`, **no GPU / no server** (desks running).
Inputs: the search desk's four findings (mx-llama.cpp = github.com/mxxm-t/mx-llama.cpp,
iacopPBK/llama.cpp-gfx906 — both shallow-cloned and read at file level) audited against this
tree + the banked decode ledger. Baseline all numbers: master scoreboard
`docs/amd/PLAN_50TPS_master.md` — round 60.0 ms, decode arithmetic 49.5 tok/s, **e2e 41 tok/s**,
MTP k=2, graphs ON, TP4 = 4 dies PCIe, acceptance 0.65 prose / 0.97 counting, tok/round 2.31.
Round anatomy (PLOG-041/044): verify 55.7 (bodies 52.6 + **in-loop AR 9.6** + tail
~lm_head 2.53 / lgather 0.67 / align 2.16 / chain_fwd 1.39 / propose 0.34 / chain_head 0.33)
+ bookkeeping 4.3. The plan's own last-14 ms: **tp_gemv draft-arm retune (~10)** + **AR (~9.6)**.

R2 rule: every claim below cites a file in THIS tree (banked rows) or a file:line in the
cloned forks. Nothing is estimated from memory.

---

## RANKED LIST (expected e2e tok/s ÷ port cost; verdicts at each section)

| # | Finding | Expected on our round | Cost | Verdict |
|---|---------|----------------------|------|---------|
| 1 | F4a tp_gemv draft-arm retune (wave64-shape census added) | −8..−10 ms → 41→~46-47 e2e | 1-2 d (kernel bench, window) | **MEASURED 2026-09-18 (W7_decoderetune_row): RETUNE NO-GO — no misretuned arm exists post-PLOG-044. All six serving draft GEMVs ride simt_r8_c4 and stream 218-246 GB/s isolated (o_proj 83 = 60.7% of floor, 0.10 ms absolute); forward sum 1.01 ms ⇒ 2.01 ms/round; the "~10 ms" was the pre-flip small_t class, already harvested. Remaining draft-side owners re-priced: F2 AR > align machinery ~1.15 ms > per-forward non-GEMV ~0.8 ms.** |
| 2 | F1a draft-vocab coverage leg (`NINFER_VOCAB_COUNT_DIR`, in-tree) | conditional 0.65→0.70+ = +8-9% e2e if coverage <90-95% | ~1 h boot flag | **MEASURED 2026-09-18 (W7_decoderetune_row): REAL prose = 98.4-99.4% (offline emitted-id membership = counter semantics; live battery 334/336 + REALK anchor 63/64; dedicated boot VRAM-blocked). ABOVE the 90-95 bar → widening NOT triggered; coverage is NOT the prose-acceptance suppressor. Serve-leg counter recipe banked for the high-volume confirmation.** |
| 3 | F2 peer-write / capture-safe decode AR arm | −6..−8 ms (AR 9.6→1.5-3) → +11-13% e2e | 3-4 d incl. device cells | **DO NEXT WINDOW** (step-0 P2P re-probe 30 min first) |
| 4 | F1b BF16 exact draft-head slice (their "drafter/target identity" class) | conditional; removes quant-noise acceptance loss | 0.5-1 d + A/B | **DO NEXT WINDOW**, gated on F1a's decomposition counter |
| 5 | F3 whole-token graph + concurrent lane dispatch | 0 (already ours) | — | **NOT APPLICABLE** (see §F3) |
| 6 | F4b wave64 shuffle-hazard census (q4 GEMV arms) | correctness class, off hot path | 0.5 d (cells) | **FILE** (RED-cell hunt, no close without cell) |

**Nothing was implemented by this desk.** After audit, no candidate change is both obviously
safe and byte-identical-when-unset: the AR arm is a new correctness-critical transport (needs
RED/GREEN device cells per closure law), the acceptance counters need egress plumbing, and the
wave64 shuffle hazard cannot be closed prose-only (RED→GREEN cell required). Designs are banked
below so the window legs are copy-ready.

---

## F1 — Drafter/target consistency + split-lm_head top-k fix (mx-llama.cpp)

**What they did (their fork):**
- DSpark: drafter replicated per lane; **target output projection replicated so every device
  holds the full logit row**; target left **un-repacked** by default — "on the same topology
  this raised draft acceptance from 63.8% to 83.5% and generation from 26.8 to 31.5 t/s"
  (`mx-llama.cpp/FEATURES.md:271-283`, `LLAMA_DSPARK_TARGET_REPACK=1` restores repacking).
  Mechanism: their drafter and target were reading the SAME weights stored two different ways
  (repacked vs not) → drafter argmax ≠ target argmax on near-ties.
- DFlash: drafter ran top-k over logits that each lane held only a SLICE of; they added a
  meta-backend pre-op gather (axis-0 split → whole rows before top-k/argmax) so the drafter
  sees the whole vocabulary (`FEATURES.md:285-301`).

**What we have:**
- Drafter = separate 40960-id draft-vocab head, W8G32, per-rank slice 10240×5120 = 53 MiB
  (`docs/amd/ACCEPTANCE_LEVERS_2026-09-17.md` §b; boot line "[rank r] draft output head ready
  (W8G32, 10240 rows, 53 MB)"); rows gathered from the full 248,320-row head by
  `src_row = draft_vocab_ids[rank*n_local + i]` (`src/runtime/tp2/tp2_backend.cpp:1106-1220`).
  Proposal = fused `allreduce_argmax` over that head + remap
  (`src/targets/qwen3_6/impl/runtime/text_context_impl.h:1894-1919` `proposal_argmax`).
- Verify = FULL-vocab greedy: lm_head ColumnN shard (248320/4 = 62080 rows/rank,
  `text_context_impl.h:2180-2194`), per-rank argmax, cross-rank reduce via the R1 ring = 12 B
  champion wire allgather + local reduce, "deterministic-equal to the full-row-set argmax"
  (`src/core/multi_gpu/tp_group.cpp:458-517`, `argmax_r1.h`).
- Accept = `speculative_accept_greedy_drafts` prefix compare of target argmax vs drafts
  (`src/targets/qwen3_6/impl/runtime/speculative_target_impl.h:26-30`).
- Coverage counter ALREADY in-tree: `NINFER_VOCAB_COUNT_DIR=<dir>` → "[vocab] coverage=XX.XXX%"
  per 10k accepted tokens (`src/runtime/tp2/vocab_output_counter.h:44-77`).
- Banked ceiling semantics: a target-argmax OUTSIDE the 40960 slice is a guaranteed miss at
  that position (`ACCEPTANCE_LEVERS_2026-09-17.md` §b; docs/48: slice covers only ~92% of the
  SIBLING-repo outputs — never validated on OUR distribution).

**The gap (mapped to their fix classes):**
- Their class (ii) top-k-gather-over-a-slice: **we already do the minimal thing.** Our wire is
  12 B/rank/token (R1 champion), and the verify argmax reduces the FULL row set exactly. There
  is no full-vocab round-trip and no sliced-argmax bug. NOT APPLICABLE as a fix.
- Their class (i) drafter/target weight identity: ours are different BY DESIGN (W8G32
  re-quantized slice of the target head, plus 208k rows absent). So the analog is NOT a bug to
  fix but two measurable acceptance suppressors:
  1. slice coverage of OUR prose argmax stream (unmeasured; ceiling semantics);
  2. quantization argmax flips (drafter sees 8-bit codes + f16 scales, target sees the full
     head) on near-tie positions.
  Plus the intrinsic MTP-head quality on prose vs counting (0.65 vs 0.97 is dominated by the
  task, not the transport — their own 83.5% at n_max 2 is the same league as our counting 0.97).

**Decision rule (banked, copy-ready):**
- Step 1 (DO NOW): boot any prose-serving leg with `NINFER_VOCAB_COUNT_DIR=<dir>`; read
  coverage. <90-95% → widen the slice (~100k ids ≈ +130 MiB/rank, draft-head GEMV +0.3 ms —
  noise; `ACCEPTANCE_LEVERS` §b already priced this). Master MISSING item 4 is this exact row.
- Step 2 (NEXT WINDOW): decompose the MISSES — per accepted position, distinguish
  "target-argmax ∉ slice" (coverage) from "target-argmax ∈ slice but drafter argmax ≠ it"
  (quant noise). Cheap env-gated device counter on the accept path; needs a small egress
  extension (NOT byte-identical-unset if it widens `MtpDecodeEgress` — do it as a separate
  gated D2H of two u32 counters, only when the env is set).
- Step 3 (conditional): BF16 draft-head slice (exact target rows for the 40960-id subset):
  40960×5120×2B/4 ranks = 105 MiB/rank (+52 vs today). This is the honest analog of their
  "replicated lm_head so drafter and target read identical weights". Only worth firing if
  Step 2 attributes a meaningful share to quant flips.

**Expected gain:** acceptance 0.65→0.70 would lift tok/round 2.31→~2.5 (+8% e2e at the same
round wall); 0.65→0.75 (+15%) is the historical MTP-head ceiling band. All CONDITIONAL on the
Step-1/2 measurements — do not resize the slice before the counter speaks (the doc's own
decision rule).

---

## F2 — Peer-write AllReduce (mx-llama.cpp) — the 9.6 ms in-loop AR

**What they did (their fork, `ggml/src/ggml-cuda/tp-allreduce.cu`):**
- `k_broadcast_reduce` (:298-415): each rank **writes its input directly into every peer's
  VRAM staging** (offset rank*n_elements; (N-1)·S posted PCIe writes/rank), in-kernel
  system-scope RELEASE/ACQUIRE flag barrier (:46-160, `st.release.sys` / `ld.acquire.sys`),
  then reads peers' slots with **nontemporal loads** and reduces locally. F32 on wire,
  "lossless". No host sync anywhere — fully graph-capturable.
- `k_cross_device_reduce_1stage` (:189-257): one-shot peer-reads variant.
- `k_twoshot_f32` (:465+) for ne ≥ 8192/rank (decode crossover :793-813,
  `GGML_TP_AR_TWOSHOT_MIN_NE`; 4-rank small-message two-shot auto :800-813).
- Coherence gate (:675-711): peer-write needs HW coherence — XGMI (gfx90a/94x/95x) OR
  **`HSA_FORCE_FINE_GRAIN_PCIE=1`** (validated on gfx906 only; experimental elsewhere).
  Without it they fall back to staging+event handshake — the same shape as OUR one-shot.
- Measured: +9% decode on 4×MI50 TP (43.9→48.0 and 66.7→72.2 t/s; `FEATURES.md:70-92`).

**What we have:**
- Default: RCCL ring, bf16, per-collect measured **135.62 us median @10 KiB world=4** (200
  reps, `results/amd/p3/W7_b1_ar_floor_row.txt`); in-serving 9.6 ms / 128 collects ≈ 75 us
  (PLOG-041). 1.31 MB @849 us — F-ENV/NCCL-tuning arms REFUTED by that row.
- `NINFER_TP_ONESHOT_AR=1` arm (built, world≤8): `one_shot_ar_pinned_vec_kernel_world`
  (`src/core/multi_gpu/one_shot_allreduce.cu:459-618`) — publishes bf16 to **host-mapped
  pinned staging** (write-through), polls peer flag/gen words (the G-AMD-30a volatile
  transport), reads peers with nontemporal loads, canonical-order reduce. World=4 GREEN
  functionally (PLOG-049).
- **Why our one-shot was a wash** (PLOG-049: "perf wash at decode sizes 7.6-12 vs 7.6-9.6 per
  ARTAIL"): two structural reasons, both visible in code —
  1. it stages through **host memory** (device→host write, host→device read = 2 PCIe crossings
     at the measured 3.13 GiB/s / 10 KiB one-way 14.1 us, `results/amd/p1/p2p_probe.out`)
     because **P2P measured UNAVAILABLE** (canAccess=0 both directions, same probe — and
     ONESHOT_AR_notes.md:68 calls this "host-staged over PCIe, RTT 98-101 µs flat");
  2. it does a **per-call `cudaStreamSynchronize`** (`one_shot_allreduce.cu:1033`) to consume
     the status word — a host round trip per collect (~128/round), killing pipelining;
  and a third, decisive for the port: that sync makes the arm **illegal inside graph capture**
  (cudaStreamSynchronize on a capturing stream fails), so the one-shot arm cannot ride the
  graphed decode round at all — any W4AR serving A/B must have run graphless or outside the
  captured region; the window owner should state which posture produced the ARTAIL numbers.

**Gap → port design (banked for the window):**
- `NINFER_DECODE_PEERWRITE_AR=1`, implemented against the `TpGroup::allreduce_local_bf16`
  interface (same call site `tp_group.cpp:381-404`), world=4 template instantiation of the
  k_broadcast_reduce shape: bf16-on-wire (halves their F32 bytes; our reduce is already bf16
  canonical-order), per-peer VRAM staging + system-scope flag barrier + NT peer reads,
  capture-safe (NO host sync; deferred status consumption at next call entry — the loud
  `_exit(70)` path already exists at `:884-899` — plus the B3 watchdog).
- **Step 0 (30 min, decisive): fresh ROCm P2P probe on all 6 die pairs** — `hipDeviceCanAccessPeer`
  + `hipEnablePeerAccess`. The banked probe (`results/amd/p1/p2p_probe.out`) says canAccess=0
  with a "GeForce driver restriction likely" note whose provenance may be the NVIDIA-era tool;
  the AMD box has never had this re-derived. gfx900 is NOT in their coherence allowlist
  (gfx90a/94x/95x), so even with peer access, `HSA_FORCE_FINE_GRAIN_PCIE=1` is required and
  correctness must be cell-proven (their own warning).
- Branch A (P2P works): port as designed. Expected: 10 KiB one-shot over PCIe ≈ publish 30 KiB
  (T=3 width, 5120×3×2B) + flag ≈ 5-10 us/collect + reduce ⇒ AR 9.6 → ~1-2 ms ⇒ round 60→~52-53
  ⇒ **e2e 41→~45-47**. Their +9% is the measured precedent on the same GPU family.
- Branch B (P2P stays closed): the honest fallback is a **capture-safe host-staged one-shot**
  (same staging, delete the per-call sync, deferred detection) — expected AR 9.6→~5-6 ms by
  pipelining alone (publish of call n+1 overlaps compute), a smaller but real win, and it
  makes the one-shot arm graph-compatible for the first time.
- Cost: Branch A 3-4 days (transport + world=4 cells + RED/GREEN + wedge drills reuse the
  PLOG-049 drill suite); Branch B 1-2 days.
- Verdict: **DO NEXT WINDOW**, gated on the step-0 probe. Do NOT bench any arm that still
  contains a per-call host sync inside the graphed round.

---

## F3 — Whole-token graph + concurrent lane dispatch (mx-llama.cpp)

**What they did:** decode token = subgraph+AllReduce+subgraph… recorded into ONE graph per
GPU, replayed once per token, bit-exact, +6% at 4-GPU (`FEATURES.md:133-…`); lanes issued
concurrently instead of device order (+2.5% at 4, +32% at 8 GPUs, `FEATURES.md:120-131`).

**What we have — already past this:**
- The ENTIRE MTP round — verify forward (incl. per-layer ARs), `mtp_prepare_next_round`, both
  draft AR steps, proposals, accept bookkeeping, and the egress D2H — is captured as ONE CUDA
  graph per rank profile and replayed once per round:
  `src/targets/qwen3_6/impl/runtime/mtp_impl.h:70-195` (`mtp_decode_batch_body` /
  `capture_mtp_decode_batch`), generic capture/launch in
  `src/targets/qwen3_6/impl/runtime/graph_impl.h:9-24` + `src/core/decode_graph.cpp:58-138`.
- ARs are INSIDE the graph: `allreduce_local_bf16` enqueues directly on the rank's own stream
  (`tp_group.cpp:381-404`), and graph-replay chunks "enqueue no per-AR host crossings"
  (`tp_group.cpp:405-411`, the heartbeat comment). Graphs ON = banked 1.70x decode (master #3).
- Concurrent dispatch: tp2/tp_engine drives each rank from its own host thread on its own
  stream; the `std::barrier sync_bar(world)` sites (`tp2_backend.cpp:1752,2136-2139`) are
  per-REQUEST staging barriers, not per-op — there is no 80-130-subgraph issue stagger to
  cure, because there is only 1 graph launch per round.

**Gap:** none on the decode round. Residual (already ledgered elsewhere): the per-round host
ingress/egress memcpys and the OUTSIDE-tail class (PLOG-041 M2 follow-up, master item 10) —
different lever, already queued.
**Verdict: NOT APPLICABLE — we already have their end state.** (Their +6%/+2.5% measured a
host-submission architecture we never had.)

---

## F4 — Half-warp dispatch for small matrices on wavefront64 (iacopPBK)

**What they did:** `ggml/src/ggml-cuda/gfx906/matmul/mmvq-q4_0.cuh:6-11,31-34,86-92` —
64-thread block = **two half-warps, each owning one output row** (2 rows per wave64), K
strided by 32 lanes, half-warp DPP reduction (`warp_reduce_sum<32>`); wins for small matrices
(ncols < 1024). Plus the wave64 CORRECTNESS hazard they fixed: sub-warp shuffles in a 32-lane
abstraction don't work once the sub-warp width reaches `warp_size/2` on wavefront64 — fallback
added (`ggml/src/ggml-cuda/mmid.cu:139-152`; README:70).

**What we have (launch-shape census, decode GEMV family):**
- The SIMT GEMV/GEMM decode family **already dispatches 32-lane workers per row** — 8 rows per
  256-thread CTA (`kBlockThreads = 8*32`, `src/ops/linear/w8/w8_rowsplit_gemm_simt.cu:20-26`;
  `q4_rowsplit_gemm_simt.cuh:39-41,224-226`: `lane = tid&31`, `warp = tid>>5`, row per warp),
  and the nvfp4 small_t tuned kernel is the same shape (`kWarpsPerBlock=8`,
  `lane=tid&31`, `nvfp4_small_t_hip.cu:135-138,293-298`, plus the T=2..4 token-sharing
  "1 worker = 1 row × all T" from PLOG-033). On gfx900's wave64, 256 threads = 4 hardware
  waves, each carrying TWO 32-lane row-workers in lockstep — **that is exactly
  iacopPBK's winning 2-rows-per-wave shape, by construction.** The w8/q4 SIMT kernels carry no
  shuffles at all (LDS/accumulator paths), so there is no width-mismatch hazard on the hot
  decode arms.
- The perf lever this desk confirms is the ALREADY-NAMED one: the draft-arm serving class
  `[14336..34816,5120]` runs ~54-56 GB/s vs 137-220 GB/s proven achievable — the
  "tp_gemv draft-arm retune (~10 ms/round)" residual (master 0b; w8_dispatch.cpp:109-118
  names the fc seam and the `NINFER_LMHEAD_ARM` application site). iacopPBK's additions to the
  retune checklist: (a) keep the 2-rows/wave mapping (don't regress to 1 row/wave);
  (b) use DPP-based 32-lane reductions for any cross-lane phase (their `warp_reduce_sum<32>`);
  (c) check the two co-scheduled row-workers' control flow stays identical (divergence
  serializes the whole wave64).

**F4b — a REAL wave64 hazard filed (off the hot path, correctness class):**
`src/ops/linear/q4/q4_rowsplit_gemv.cuh:294,350,352` issue **full-wave**
`__shfl_sync(kFullWarpMask, lane_scale_bits, local_group)` with `srcLane = local_group < 32`
inside a 32-lane-per-warp mapping whose schedules include `kWarpsPerRow = 8`
(`Q4GemvR1W8DirectSchedule`, `q4_rowsplit_gemv.cuh:101-106` → serves the q4_q5 gdn-input /
attn-input arms, `q4_q5_gdn_input_independent.cu:28-29`, `q4_q5_attn_input_small_t.cu:25-27`).
On gfx900 (wave64) hardware lanes 0-63 = logical warps 0+1 of the SAME row, so a `srcLane<32`
shuffle in logical warp 1 reads the SIBLING WARP's scale bits (wrong group range) — the exact
class iacopPBK had to fall back for. Our NVFP4 decode path does NOT route through these arms
(they serve Q4/Q5 gdn/attn-input weights), which is consistent with banked byte-correct decode;
but per the closure law this is a FILED HUNT, not a close: it needs a RED device cell (run the
R1W8 arm on gfx900, compare vs the mma/reference) before any guard/fix, then the cell joins the
battery permanently. Also matches the G-AMD line-law: guard the CLASS (any full-wave shuffle
under a <64-lane ownership split), not the instance.

**Verdict:** hot-path half-warp dispatch: **NOT APPLICABLE (already our shape)** — fold their
DPP-reduction + uniform-control-flow checklist into the DO NEXT WINDOW draft-arm retune.
F4b shuffle hazard: **FILE** (0.5 d, cells; no prose close).

---

## PROVENANCE / R2 CHAIN

- Their fork: mxxm-t/mx-llama.cpp @ shallow clone /tmp/w7refs/mx-llama.cpp — FEATURES.md:70-92
  (custom AR +9%), :120-131 (concurrent dispatch), :133-140 (whole-token graph), :271-283
  (DSpark 63.8→83.5), :285-301 (DFlash whole-row gather); tp-allreduce.cu cites inline.
- Their fork: iacopPBK/llama.cpp-gfx906 @ /tmp/w7refs/llama.cpp-gfx906 — README:70,
  mmid.cu:139-152, gfx906/matmul/mmvq-q4_0.cuh (half-warp per row, DPP reduce).
- Ours: PLAN_50TPS_master.md (round anatomy, last-14 ms, MISSING 0/2/4, 0b), PLOG-041
  (M1 decomposition), PLOG-044 (LMHEAD flip, 60.0 ms), PLOG-049 (W4AR GREEN + wash verdict),
  results/amd/p3/W7_b1_ar_floor_row.txt (135.62 us @10 KiB w4; F-ENV refuted),
  results/amd/p1/p2p_probe.out (P2P canAccess=0; staged 3.13 GiB/s / 14.1 us @10 KiB),
  results/amd/coherence/ONESHOT_AR_notes.md (RTT 98-101 us flat), ACCEPTANCE_LEVERS_2026-09-17
  (slice pricing, coverage counter, greedy-already). Source cites inline (tp_group.cpp,
  one_shot_allreduce.cu, tp_kernel.cu, w8_dispatch.cpp, nvfp4_small_t_hip.cu,
  q4_rowsplit_gemv.cuh, mtp_impl.h, graph_impl.h, decode_graph.cpp, text_context_impl.h,
  speculative_target_impl.h, vocab_output_counter.h).
- Desk scope note: read-only on the shared checkout; no GPU; nothing committed but this doc.
