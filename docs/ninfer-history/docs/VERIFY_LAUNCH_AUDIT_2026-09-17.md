# VERIFY LAUNCH AUDIT — where the 195 ms verify round actually goes (TP4, T=3)

**Date:** 2026-09-17 · **Seat:** GEMV-TUNE perf seat (step 3 of EXTRACTING_NUMBERS Part C
sequencing) · **Method:** code-read with file:line cites + banked rows + ONE single-die
micro-bench (§5; zero TP4 work — box is TP4-blocked per `TP4FAULT_row.txt`).
**Governing banked row:** `TIMING1_row.txt` (bin 07ad7eccc0b97cc0, TP4, GRAPHS-ON, count600,
mean of 203 rounds): **Target Verify = 195.34 ms of a 211.95 ms round (92.2%)**; round
bookkeeping (accept/rebase/prepare/select/propose) = 0.45 ms (0.2%); draft side
(align 8.56 + AR-chain 7.61 + propose 0.29) = 16.46 ms (7.8%).

---

## 0. MEASURED — HOLE1 window (2026-09-17, same day; supersedes the ranked hypotheses in §4 and re-points §7)

Instrument landed and banked: **NINFER_TP2_OPTRACE=1** (per-rank per-round 9-phase event
lines in `tools/optrace_analyze.py` format — the B1 machinery was rank-0-only AND its
PhaseTimer was a shared-instance race once all ranks record; now one per-rank instance
inside the worker lambda) and **NINFER_VERIFY_LAYER_TRACE=1** (per-layer body + per-layer
AR cudaEvent ring inside `run_layers`, drained next-pass, `[LTRACE]` lines every 64 passes).
Binary banked `5735aaf3e9ddd410` (branch amd/tp4-cure). Three count600 boots: instrumented
remap-BROKEN, instrumented remap-ON, and the certified `e341a2072c6f5078` A/B. Rows:
`results/amd/coherence/HOLE1_row.txt`, `HOLE1_ltrace_table.txt`, logs `HOLE1_serve.log`,
`HOLE1AB_e341_serve.log`, `HOLE1REMAP_serve.log`.

### 0.1 The hypotheses, measured

- **H1 cross-rank skew — ACQUITTED.** Per-rank phase times are identical to <0.1 ms on all
  4 ranks (verify 216.657/216.665/216.662/216.664 ms; analyzer sum-skew 0.017-0.019 ms,
  no consistent laggard). Dev0's desktop does NOT gate the 128 ARs. The §4-H1 mitigation
  list (quiesce desktop, rank remap) buys nothing.
- **H2 AR cost above the law — ACQUITTED (inside verify).** Measured in-graph... rather,
  IN-EAGER verify ARs = 14.5-15.1 ms/round = 128 × ~114 µs ≈ the eager RTT law. ARs are
  ~7% of the verify phase and are NOT the hole.
- **H3 one fat op class — REFUTED in the single-op form.** The interior is a UNIFORM
  ~1.9-2.0 ms/layer across all 64 layers; full-attention layers cost only ~+0.15 ms vs GDN
  layers. Attention/KV is NOT the owner. No single op owns the hole.
- **H4 graph-replay occupancy — MOOT: §1's premise is wrong for this path.** The TP2/TP4
  round loop (`tp2_backend.cpp` worker) is EAGER — there is no `cudaStreamBeginCapture` in
  `src/runtime/tp2/` at all; the only capture site (`decode_graph.cpp`) serves the
  single-process program path TIMING-style TP4 boots never take. "GRAPHS-ON" in the boot
  config does not capture this path. The ~950 nodes are launched fresh every round — which
  also re-opens a small honest lever (graph capture at TP4 would need the tp2 path wired to
  `DecodeGraphDefinition`, none of §6's hazards exist yet because nothing is captured).

### 0.2 Where the time actually goes (remap-ON boot, verify phase = 203 ms p50)

| segment | ms/round | note |
|---|---|---|
| layer loop (LTRACE body sum) | **138.9** | 64 × ~2.15 ms |
| — of which TP ARs (2/layer, axpy incl) | **14.5** | 128 × ~114 µs ≈ RTT law |
| — of which non-AR kernels + intra-layer gaps | **124.4** | ~1.94 ms/layer, uniform, width-independent (T=1 intercept 126.5 ms per NIGHT_HANDOFF) |
| outside the loop (emb, finalnorm, full-vocab lm_head GEMV, argmax, prepare, 3 logits allgathers, allreduce_argmax) | **~64** | measured by subtraction phase-vs-loop |
| whole verify phase | **203-215** | rest of round: align 8.9 + draft chain 8.8 + propose 0.34 + bookkeeping 0.3 |

Effective per-rank weight-stream rate implied by the non-AR layer time: **~33 GB/s vs the
265 GB/s isolated-GEMV ceiling — an ~8x kernel-exec gap**. During decode the cards run at
100% GPU / power pinned at the 110 W cap (SCLK 1269-1350 MHz, MCLK 945 MHz —
`HOLE1REMAP_clocks_t12.txt`), so part of the gap is power-cap clock behavior; the rest is
per-kernel inefficiency at T=3 under interleaved small ops. THE NEXT DISCRIMINATOR is
per-op (tpv_probe-tagged) events inside ONE layer, or a rocprof kernel timeline, to split
kernel-exec from intra-layer stream gaps; and the micro-bench-vs-serving gap needs a
dedicated cell (same GEMV shape inside a 40-kernel mixed stream).

### 0.3 SECOND FINDING — the draft-vocab remap CWD trap (+142 ms/round, landmine class)

Boot 1 (CWD inside `results/amd/coherence/`) silently degraded the DRAFTER to the full
lm_head: `propose` 0.34 → **59.5 ms**, `ar` chain 8.8 → **66.4 ms**, acceptance 0.97 → 0.55,
round 233 → **376 ms**. The certified e341 binary shows the SAME numbers without my
instrumentation (tree property, not instrument distortion), and TIMING1-era boots show the
remap loading (`loaded 40960 draft vocabulary IDs`). The B3 guard prints a loud WARNING —
which is how this was caught — but the runbook should mandate
`NINFER_DRAFT_VOCAB=<absolute path>` (or repo-root CWD) for every TP4 boot. Any historical
row whose log lacks the `loaded 40960 draft vocabulary IDs` line is a remap-broken row and
must not be compared to remap-on numbers.

### 0.4 Round budget now (remap-ON, this window, 290-round means)

verify 215.0 (92.1%) + ar 8.8 + align 8.9 + propose 0.34 + bookkeeping 0.3 = **233.4
ms/round** vs TIMING1's 211.95 — cross-boot thermal band ±25% on this box (B3HARDEN saw
294.9 warm); the STRUCTURE (92% verify) reproduces exactly.

---

## 1. THE ROUND FRAME: it is already ONE captured graph

The whole MTP round — target verify → prepare-next → alignment forward → select-accepted →
propose chain — is captured and replayed as a single graph:
`capture_mtp_decode_batch` (`src/targets/qwen3_6/impl/runtime/mtp_impl.h:185`) captures
`mtp_decode_batch_body` (`mtp_impl.h:~100-182`, includes `card.target_verify_batch` at :17 of
`speculative_target_impl.h` call shape, the k−1 propose loop with its per-step D2D at
`mtp_impl.h:175`, and the egress D2H at :179) via `DecodeGraphDefinition::capture`
(`src/core/decode_graph.cpp:58`, `cudaStreamBeginCapture ... ModeThreadLocal`). TIMING1 ran
this path GRAPHS-ON and its verdict is explicit: "the ~200 ms graphs-off → graphs-on win
already removed the launch overhead that lived there; what remains is NOT inter-phase sync."

**Consequence:** the classic "count host launches" audit is already priced at ~zero by
replay. What matters inside a replayed graph is (a) how many NODES (each node still costs
device-side schedule time), (b) the fat kernels, (c) the collectives — which serialize on
cross-rank arrival. §2 counts the nodes.

## 2. (a) PER-ROUND LAUNCH CENSUS (code-derived, T = k+1 = 3, batch = 1, TP4)

Layer mix: 64 layers = **48 GDN + 16 full-attention** (`hybrid_topology.h:9-14`,
`kHybridAttentionInterval = 4`: every 4th layer is full; comment "3 GDN + 1 full-attention
per block" `text_context_impl.h:2158`). Loop: `run_layers`
(`text_context_impl.h:2764-2830`) — full → `attn_mix_tp` + `mlp_tail`; GDN → `gdn_mix_tp`
(verify arm :2167-2307) + `mlp_tail`.

| site | kernels per instance | cites |
|---|---|---|
| **Full-attention mixer** | rmsnorm 1 · projection 1–3 (split arm: 2 tp_gemv + 1 pack; NVFP4 fused: 1) · qkv unpack 1 · q-rms 1 · k-rms 1 · rope 1 · attend 1–2 (kvarn-batched or gqa) · sigmoid_mul 1 · o_proj tp_gemv 1 · **AR 1 + axpy 1** | `attn_mix_tp` `text_context_impl.h:1909-2115` (:1918, :1921, :1929, :1946-47, :1954, :2049/:2084, :2110, :2113, :2114); proj arms `variant_kernels.cpp:203-240` |
| **GDN mixer (verify arm)** | control norm+gating proj 1 · input proj 2 (NVFP4 fused: 1 tp_gemv + 1 unpack) · conv1d+silu+snapshot+split3 1 · gbeta unpack 1 · gated_delta_net_snapshot 1 · gated_rmsnorm 1 · out_proj tp_gemv 1 · **AR 1 + axpy 1** | `gdn_mix_tp` `text_context_impl.h:2150, :2188, :2216, :2227, :2281, :2288, :2301, :2303`; fused-proj arm `variant_kernels.cpp:265-293` |
| **MLP tail (every layer)** | rmsnorm 1 · gate_up tp_gemv 1 · silu_mul 1 (strided-split fast path, W4a) · down tp_gemv 1 · **AR 1 + axpy 1** | `mlp_tail` `text_context_impl.h:2638-2657`; `post_mixer_tp` `variant_kernels.cpp:464-509` |
| Round head/tail | embedding 1 · final rmsnorm 1 · lm_head linear 1 · argmax 1 (+ cross-rank argmax reduce) | `target_verify_batch_impl` `text_context_impl.h:1721-1746` |

**Totals per verify round (TP4):**

- Mixer+MLP kernel launches (excl. AR): GDN 48 × ~12 + full 16 × ~13 + mlp 64 × ~4
  ≈ **~850–950 kernel nodes**.
- Collectives: **2 ARs per layer × 64 = 128 ncclAllReduce** — call sites
  `text_context_impl.h:2114 (attn), :2303 (gdn), :2651 (mlp)` →
  `TpGroup::allreduce_local_bf16` (`tp_group.cpp:336-347`).
- **TP4-specific: every AR is ncclAllReduce + a SEPARATE axpy kernel** (the residual fold)
  because `impl_->one_shot` is constructed **only at world==2**
  (`tp_group.cpp:136-139`: `if (I.n == 2)`). One-shot capacity would cover TP4 T=3
  trivially (`kMaxElements = 65536` = "covers up to T=12" `one_shot_allreduce.h:15`) —
  the blocker is design scope (2-rank wire shape, "for rank (0 or 1)"
  `one_shot_allreduce.h:22-23`), i.e. the roadmap's "one-shot AR OPEN w/ named GCN
  blocker": at 4 devices over PCIe (doc 01 §3: no cross-die fabric) the peer-flag
  spin-wait topology is the unbuilt part. The one-shot already carries fused residual-add
  ("−1 launch/step" `text_context_impl.h:1117`) and graph-replay epoch support
  (`one_shot_allreduce.h:36-38`).

**What the census is worth (measured today, §5):** empty-kernel launch = **1.56 µs
device-side** (deep queue), host enqueue 1.83 µs; a verify-shaped tiny kernel
(15360 elems) = **2.57 µs**. So ~950 nodes ≈ **2-3 ms/round** pure schedule+floor — an
upper bound on what ANY launch-count reduction (fusions, graph coarsening) can recover
from the verify forward. Launches are NOT the owner. This independently confirms TIMING1's
0.45 ms round-bookkeeping exoneration and re-points the audit at the collectives and the
unaccounted bulk (§4).

## 3. (b) AR BATCHING — the concrete sites, and what the dependency order allows

The RTT law (Part C: 128 × 98-101 µs ≈ 13 ms, "eager AR floor" PLOG-005) and the 2-4-layer
batching idea meet THREE facts from the code:

1. **The ARs are semantically per-layer and on the critical path.** Each layer's next op is
   an rmsnorm over the FULL hidden row (`attn_mix_tp:1918`, `mlp_tail:2642`, GDN control
   proj `:2150`), and each rank's column-parallel GEMV consumes the full normalized row
   (K = 5120, not the rank shard). You cannot defer an AR past the next norm without
   re-plumbing to a sharded-residual + allgather scheme — and an allgather costs the same
   latency class as the allreduce it replaces. The "batch 4 layers' ARs into one call"
   pattern therefore has **no adjacent 4 tensors to concatenate** at the current layer
   structure: the 128 ARs are interleaved with 128 norms, not clustered.
2. **The honest fusion site that DOES exist: kill the axpy + shrink the collective.** At
   TP4 each AR is 2 device ops (`tp_group.cpp:342-346`: ncclAllReduce + one_shot_axpy_bf16).
   Wiring the existing OneShotAllReduce for world=4 removes 128 axpy launches AND replaces
   ring-AR (latency ~O(world × link RTT)) with one-shot (~O(1) flag protocol + 30 KB
   exchange; doc 01 §5 measured 11-15 µs/op on NVLink, PCIe estimate 2-3×). Ceiling:
   ~13 ms → **~2-4 ms/round**. This is the roadmap's "one-shot AR" item and it is
   code-complete for world=2 — the work is the 4-rank wire, not a rewrite.
3. **TIMING1 caps the whole AR family:** "AR does NOT dominate — batching the draft chain
   can recover AT MOST ~7.6 ms/round... even zeroing it moves the round <4%". The 13 ms
   eager floor inside verify is likewise <7% of 195 ms. **Verdict: AR work is
   second-order; do it for the constant factor (one-shot at TP4), not as the lever.**
   Contract note: element-wise AR means batching/one-shot changes NOTHING in fp summation
   order per element (same 4 rank-partials, same sum) — bit-preservation is expected but
   must be proven by the standard byte-diff cell before any landing (rule doc 01 §4).

## 4. THE HOLE THE AUDIT FOUND: ~140 ms of the verify forward has NO owner

> **MEASURED the same day — see §0. All four hypotheses below are resolved there (H1/H2
> acquitted, H3 refuted as single-op, H4 moot); the hole decomposes as ~124 ms of uniform
> non-AR kernel-exec across the 64 layers + ~64 ms outside the loop.**

Bytes audit (Part C, GEMV-TUNE-updated): ~4.1 GB/rank weight stream per verify round.
At the PRE-TUNE measured GEMV rate (~150 GB/s) ≈ 27 ms; at the tuned 265+ GB/s ≈ **15.5 ms**
(`GEMV_TUNING_row.txt`). Add AR floor ~13 ms, launches/small-kernels ~2-3 ms (§5),
attention at short context (small, unmeasured), lm_head ~4 ms. Total accounted:
**~40-55 ms — but the verify forward MEASURES 195.34 ms** (pre-tune; the tuned GEMV
predicts ~183 ms). **~140 ms per round is unexplained by any banked number.**

Named candidate hypotheses (falsifiable by the same instrument):
- **H1 — cross-rank skew amplified by 128 serialized collectives.** Each ncclAllReduce
  waits for the slowest rank's arrival; dev0 hosts the desktop (runbook §3: Xorg/gnome
  measured 139 MB-8 GB swings), so rank-0 jitter accumulates and EVERY subsequent AR pays
  the skew. 128 × ~1 ms average skew ≈ the whole hole. Prediction: per-op probes show the
  wait concentrated INSIDE AR events, growing through the layer loop; mitigation is
  environmental (quiesce/pin desktop, or ranks on dev1-3 + one die pair swap).
- **H2 — in-graph AR cost far above the 98-101 µs RTT law** (the law was measured eager;
  RCCL-in-graph protocol/channel behavior may differ). Prediction: uniform AR cost ~3-5×
  the law.
- **H3 — a fat op class hiding in plain sight** (e.g. gated_delta_net_snapshot or the conv
  snapshot ops at 48×/round being tens-of-µs each, or hidden device syncs).

**The decisive instrument (ranked #1, one boot):** an env-gated per-op event-probe arm
riding the EXISTING `tpv_probe` sites (`text_context_impl.h:945` — today it is only a
cudaGetLastError check, zero timing) at :2188/:2216/:2281/:2288/:2301/:2303 (gdn),
:1918-:2114 (attn), :2642-:2651 (mlp), :1742-:1746 (tail) — hipEvent pairs per op, summed
per tag per rank, printed at round end (the B1 8-phase pattern,
`tp2_backend.cpp:597-635`, is the template; NINFER_TP2_TIMING=1 gates it today). This
splits the 195 ms into AR-wait vs kernel-exec vs gaps, per rank, in one GRAPHS-ON boot.

## 5. MICRO-BENCH (measured this session, dev0, single process, gfx900, ROCm 6.2.0)

```
EMPTY kernel: 10000 launches, 15.642 ms total, 1.56 us/launch (device-side, deep queue)
EMPTY kernel host-wall: 1.83 us/launch (enqueue)
TINY kernel (15360 elems, rmsnorm-shaped): 2.57 us/call device-side
```
Source pattern: `/tmp/launch_overhead_bench.cu` (empty kernel N=10000 + 5120×3-elem kernel
N=20000, hipEvents; KFD=0 pre/post; ~10 s GPU). Prices: (a) the ~950-node census at
1.56-2.57 µs/node ≈ 2-3 ms/round; (b) the axpy-class op at TP4 ≈ 128 × ~2 µs ≈ 0.3 ms;
(c) graph-node scheduling cost — consistent with TIMING1's bookkeeping exoneration.

## 6. (c) WHOLE-ROUND GRAPH CAPTURE — status and the remaining hazards

Status: **shipped and running** (§1; roadmap "graphs SHIPPED 1.70x"). The audit therefore
only enumerates what a future re-capture (e.g. after the one-shot TP4 AR lands inside the
graph) must keep in the captured/refreshed set:

1. **KV write-slot indices** — per-round positions arrive via device tensors bound before
   capture (`cache_positions`/`rope_positions` views, `text_context_impl.h:1710-1720`
   ScopedPositions) — pointers must be capture-stable, contents refreshed per round
   (already the design; any new epilogue must follow it).
2. **GDN conv + recurrent (SSM) state** — slot-indexed state pools
   (`state_.conv`/`state_.recurrent`, `text_context_impl.h:2200/:2233`) with
   initial/base slot tensors per round (`:2176-2182`); the verify arm writes snapshots into
   the W+1 ring and acceptance selects the committed slot afterwards (`:2168-2173`).
   Doc 01 §5's measured V100 failure ("graph inert in their env — drafter's recurrent state
   did not persist, τ→0, all drafts rejected") is the hazard: the drafter's conv/SSM state
   and any capture-time constants must be in the graph's owned/refreshed set.
3. **RCCL-in-capture**: at TP4 the ncclAllReduce calls ARE inside the captured round
   (graphs-on measured 1.70x with the nccl path at world=4 — they replay). The named
   residual hazard is the one-shot AR's epoch surface: it has explicit replay support
   (`advance_epoch`/`reset_step`, `one_shot_allreduce.h:36-38`); a 4-rank one-shot must
   carry the same epoch discipline or be excluded from capture (which would forfeit its win).
4. **Validation plan (doc 01 §5's own law):** after capture, a **state-norm probe** — run
   N≥3 rounds on a fixed prompt; after each round dump/compare GDN conv+recurrent state
   and KV tail vs the eager path (byte-diff rule doc 01 §4); PLUS the acceptance-rate
   parity guard: captured-vs-eager first-miss profile (aτ→0 is the failure signature,
   TIMING1 records a0/a1/a2 per round) must match within noise before the graph is
   trusted for serving.

## 7. (d) RANKED PATH TO 46 ms/ROUND (updates Part C; per-round, TP4, T=3, batch=1)

Anchor: round 211.95 ms (TIMING1, PRE-GEMV-TUNE) → predicted ~200 ms post-tune. Budget
requires ≤46 ms.

| # | lever | owner today (ms) | best evidence | projected | measurement plan | risk |
|---|---|---|---|---|---|---|
| 0 | **INSTRUMENT the verify forward** (per-op event probes at tpv_probe sites, per rank) | — | §4 hole: ~140 ms has no owner; tpv_probe is error-check-only today (`text_context_impl.h:945`) | enables ALL rows below | one GRAPHS-ON boot, NINFER_TP2_TIMING-style arm; split AR-wait vs kernel vs gap per rank | none (env-gated) |
| 1 | **Resolve cross-rank skew / H1** (desktop on dev0; rank drift × 128 ARs) | inside the ~140 ms | H1 fits: dev0 = display card (runbook §3); ARs serialize on slowest rank | 0 → up to ~100+ ms IF H1 owns it | row 0's probes; then A/B with desktop quiesced or rank map dev1-3+swap | env, not code |
| 2 | **GEMV weight stream** | ~27 (150 GB/s) | GEMV_TUNING_row.txt: 265+ GB/s bit-equal | **~15.5** | DONE (this window); verify in-boot at next TP4 window | landed |
| 3 | **One-shot AR at TP4** (kill 128 ring-ARs + 128 axpys) | ~13 (eager floor) + 0.3 axpy | `tp_group.cpp:136` world==2 gate; doc 01 §5 economics; TIMING1 <4% ceiling | ~13 → **2-4** | row 0 probes pre/post; byte-diff cell (expect bit-preserving, elementwise sum) | 4-rank wire unbuilt (GCN blocker) |
| 4 | **Attention + small kernels + lm_head** | ~10-15 (unmeasured) | §5 tiny-kernel floor 2.57 µs; lm_head audit (B.2): 0.75 MB logits + 635 MB weights ≈ 4 ms | few ms | row 0 probes name them; fuse only what probes convict | low |
| 5 | AR batching across layers (RTT-law 128→32-64) | ≤13 ceiling | TIMING1: AR family <4% of round; §3.1: no adjacent tensors at current structure | ~0 net | only if row 0 shows per-AR cost >> law | re-plumb norms; LOW priority |
| 6 | Draft side (align + AR chain + propose) | 16.46 | TIMING1 phases 5+7+8 | ≤8 (chain batching) | priced; <4% ceiling — skip unless free | low value |

**Honest arithmetic:** even with #2 landed and #3+#4 perfect, the round is ~200 − 12(GEMV)
− 10(AR) − few(small) ≈ **~175 ms unless #0/#1 convict and fix the ~140 ms hole.** The 46
ms budget therefore hinges on the instrument-first row: nothing else in the ranked list
physically reaches 46 ms without knowing what owns the hole. This is the next window's
first boot (after the TP4FAULT root reset).

---
Bases: TIMING1_row.txt · GEMV_TUNING_row.txt · TP4FAULT_row.txt ·
`text_context_impl.h` / `variant_kernels.cpp` / `tp_group.cpp` / `one_shot_allreduce.h` /
`mtp_impl.h` / `decode_graph.cpp` / `hybrid_topology.h` (line cites inline) ·
docs/optimizations/01 §5 · launch micro-bench §5 (this session).
