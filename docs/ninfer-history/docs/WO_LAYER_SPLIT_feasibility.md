# WO — LAYER-SPLIT (pipeline parallelism) FEASIBILITY — design desk deliverable

Desk: layer-split feasibility design (Team Red, AMD V340L, 4x gfx900, NVFP4@TP4 campaign).
Worktree: `/home/chris/worktrees/amd-wo-w7-body` (branch `amd/wo-w7-body` @ 90c9ecf22). READ-ONLY
on `src/`; this doc + receipts only. No builds, no serving window, zero GPU (all numbers banked).
Queued by: STRATEGIC UPDATE 2026-09-19 (90c9ecf22, `docs/amd/REMAINING_ITEMS.md`).
STATUS: **DELIVERED — VERDICT: feasible mechanically, NOT worth it for the 27B@2k target;
RECOMMEND stay-TP4 cure ladder; keep layer-split as the MoE-lineage / long-context design.**

## PROGRESS LOG (newest first)

- 2026-09-19 DELIVERABLE COMPLETE: verdict + tables + gates written (this commit). No src touch,
  no build, no GPU, :8100 untouched.
- 2026-09-19 pricing pass: pipeline arithmetic derived from banked decomp (990 ms/chunk wall =
  gemm ~770 + AR ~123 + body/gap ~97; P1/W2 logs; W7_p2p_pt_row transport floor); decode pricing
  from decode-landing row (round 129.38 = verify 112.8 + align 8.48 + AR-chain 7.63); VRAM from
  measured-manifest 3820 MiB/rank placement + tp_load roles; effort census from the site inventory.
- 2026-09-19 inventory pass complete: StageSpec pipeline-range machinery EXISTS in-tree
  (bindings.h:210-222, bindings.cpp:586-679, text_context.h:177-187, text_context_impl.h:364/1005/
  2147/2189/3509); TpGroup::send/recv P2P exists (tp_group.h:108-109, tp_group.cpp:228-232/575-600);
  SPMD round engine mapped (tp2_backend.cpp:1752 barrier, ~30 arrive_and_wait, three runners);
  AR sites pinned (text_context_impl.h :2717 attn, :2918 gdn, :3333 mlp, MTP mirrors
  :1650/:1659/:1984/:1996, argmax :2017/:2027/:3887; ~20 backend sites); weight placement roles
  pinned (tp_load.cpp:371 Replicate embedding, :367-368 ColumnN head, :705-707 shard calls);
  KV/GDN state world-derived (tp2_backend.cpp:716/819-836); MoE variant runs SINGLE-DIE
  (variant.cpp:237-239).
- 2026-09-19 desk-open: worktree confirmed, strategic-update diff read, `src/core/multi_gpu/` +
  `src/runtime/tp2/` + target impl census started.

---

## 0. VERDICT (executive)

**Layer-split is mechanically feasible — about a third of its plumbing already exists in-tree —
but on every banked number it LOSES to the stay-TP4 cure path at the campaign's prefill target
(2k) and wins nothing at decode that the AR cure doesn't win cheaper.** The llama.cpp 152-160 t/s
existence proof does not transfer: it is an M=512-rocBLAS + thin-body proof, not a layer-split
proof (their own multi-die scaling is +3-24%, i.e. ~single-die rate either way). Layer-split's
real strategic value is elsewhere: **it is the only AR-free multi-die architecture for the
35B-A3B MoE lineage** (which today runs single-die only, `variant.cpp:237-239`, and whose NVFP4
weights ~4.4-4.8 GB/die fit layer-split but have no dense-TP support). Recommendation, in order:
(1) stay-TP4 + chunk ladder + AR-quant/copy-add + prefill graph capture (2-3 d ladder + 1.5 d AR
+ 2-3 d capture; projected 135-150 t/s at 2k, decode unchanged); (2) layer-split PARKED with
pre-registered re-entry triggers (MoE adoption, >3k-token window mix, TP4-infeasible model);
(3) dual-engine llama.cpp-prefill serving: INCOHERENT for this stack (dead below).

## 1. INVENTORY — where TP4 is assumed (file:line, this worktree)

Depth summary: **the TP4 assumption is WIDE but not DEEP.** It is a placement role + a per-rank
SPMD schedule, not a baked-in data layout. The forward itself branches on `tp_group_ != nullptr`
per layer (`text_context_impl.h` attn :2390, mlp :3318), the loader branches per-tensor role
(`tp_load.cpp:371`), and the layer RANGE axis already exists independently of the shard axis.

### 1.1 Weights layout / loading
- `src/targets/qwen3_6_27b/impl/load/tp_load.cpp`
  - :323/:366-371 — per-tensor `TpRole`: `ColumnN` (output_head), `RowK` (o_proj, GDN out,
    mlp down), `Replicate` (embedding, norms, GDN control). Embedding is **replicated full**
    [248320,5120] on every die; lm_head is ColumnN 62080 rows/rank BF16 (635 MiB/rank).
  - :408-412 — shard shapes {rows/world, cols} etc.; :654-707 — `shard_row_split` RowK/ColumnN
    copies (the only three `weight_shard` call sites in the repo, all here).
  - `materialize_tp(reader, rank, world, ...)` — placement happens at artifact materialization:
    **each die receives the SHARD of every layer**. Layer-split needs a new mode: full-width
    tensors for a layer RANGE, nothing elsewhere.
- `src/targets/qwen3_6_27b/impl/load/bindings.h`
  - :21-23 — `kTextLayers=64`, `kFullAttentionLayers=16`, `kGdnLayers=48` (hybrid topology).
  - :206-222 — **`StageSpec{first_layer, last_layer, include_embedding, include_lm_head}` —
    pipeline-stage selection EXISTS**; `LoadedModelData` ctors take (StageSpec, tp_world) as
    orthogonal axes (:226-243).
- `src/targets/qwen3_6_27b/impl/load/bindings.cpp`
  - :586-679 — `initialize(plan, mat, stage, tp_world)`: binds ONLY layers in
    [first_layer, last_layer) (:610-664) into local ModelView slots; `include_embedding`
    (:599-602) and `include_lm_head` (:676-679) are already conditional. `output_head`
    materialized at `248320/tp_world` (:679), o_proj/down at `6144/tp_world` k-dim
    (:634-635, :658-659), GDN conv `10240/tp_world` (:648-650).
- Weights per die (measured-manifest anchor): **3820 MiB/rank at TP4**
  (G18_firstlight_FAILURE_PLAYBOOK_agent4.md:118 "placement=3820 MiB/rank (measured-manifest)").
  Derived composition: stage-local quarter-width layers ~2.0-2.1 GiB + replicated embedding
  (~0.7 GiB NVFP4-class) + lm_head shard 0.62 GiB + mtp/ (~0.4 GiB W8G32). **The 3.8 GB/die
  NVFP4 claim is VERIFIED** (15.3 GB total across 4 dies).

### 1.2 Forward pass (TextContext)
- `src/targets/qwen3_6/impl/runtime/text_context.h`
  - :177-187 — ctor takes `first_layer`/`last_layer`: "Pipeline-parallel range: only layers in
    [first_layer, last_layer) are bound and executed; local KV/linear-state layer slots are
    indexed relative to the first local full/GDN layer."
  - :203-207 — `set_tp(group, rank, world)` arms the TP arms.
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h`
  - :364-372 — stage range ctor + validation; :1005, :3509 — BOTH layer loops (GDN-state and
    hidden forward) iterate `[stage_first_layer_, stage_last_layer_)`; :2147/:2189 —
    embedding/finalize are range-aware (stem only on the first stage, head only on the last).
  - **In-layer AR sites (2/layer x 64 = 128/chunk)**: full-attn o_proj :2717; GDN out-proj
    :2918; mlp_tail down-proj :3333 (ColumnN gate_up + RowK down + AR pattern, :3318-3337).
    MTP mirrors: :1650, :1659, :1984, :1996. Vocab tail: argmax :2017, :2027, :3887;
    non-greedy allgather :2783/:3056/:4401/:5231/:6819/:7716 (backend sites).
  - Batched-attention sites: MultiBatch runner is backend-side (below); in-context the
    `active_sequence_*`/kvarn batched arms (:1605-1628, :2491-2513) are head-shard-shaped
    (`n_q/world`) — full-width restorage needed under layer-split.
- `src/core/multi_gpu/tp_group.h` / `.cpp`
  - :50-77 — allreduce/allgather collectives; :88-90 one-shot argmax (ring world==2-gated, R1
    RCCL arm at world 4 — `argmax_routing.h:67`, `r1_argmax.cu:300`).
  - :108-109 + tp_group.cpp:228-232, :575-600 — **`send`/`recv` point-to-point pipeline
    primitives ALREADY EXIST** (RCCL ncclSend/ncclRecv, bf16, one pending task per rank,
    post-then-synchronize_all contract).

### 1.3 Round engine / workspace / KV (the SPMD heart)
- `src/runtime/tp2/tp2_backend.cpp`
  - :705-1000 `make_rank`: world-derived everything — LM-head rows divisibility throw :716-719,
    GDN state dims `10240 % world`/`48 % world` throw :819-822, KV head geometry from
    `head_geom_for_config(24, 4, tp_w)` tier table :832-836 (per-rank kv_heads = 4/world = 1 at
    TP4), DecoderStateSpec `full_attention_layers=16` :838-854, KV pool reserve :967-995
    (paged, per-rank). `LoadedModelData(..., StageSpec{0, 64, true, true}, tp_w)` :767-768 —
    the ONE call site, full range today.
  - :1752 `std::barrier sync_bar(world)` + 24 `arrive_and_wait()` sites (single-seq runner
    :1714-3787) — **every rank executes every round phase in lockstep**. Same SPMD shape in the
    D2 runner (:4788-4831) and the batched runner (:7856-7869, world-vectorized threads).
  - Prefill chunk driver :2166-2212 (re-enqueue after per-chunk hard sync) — rank-symmetric.
  - Backend AR/argmax sites: :2323, :2474, :2519, :2783-2794, :3056, :3164, :3567, :3650,
    :4196, :4395-4401, :5223-5231, :5967, :5995, :6804-6819, :7464, :7556, :7644, :7686, :7716.
  - Per-rank state: `TpRankState` (tp2_backend.h:51-179) — own arenas (work/persistent/staging),
    own DecoderState + paged KV + GDN checkpoints, own MTP staging/draft head, own DFlash2
    state+weights (~1 GiB/rank).
- `src/runtime/tp2/tp_engine.cpp` — one TpGroup over `devices`, one host worker thread per rank.
- VRAM preflight: `src/runtime/tp2/tp2_budget.h` + tp_engine preflight region — per-rank
  placement math assumes world-sharded weights (measured-manifest 3820 MiB/rank class).

### 1.4 MoE variant (compounding fact)
- `src/targets/qwen3_6_35b_a3b/impl/variant.cpp:237-239` — "35b is MoE and does not support the
  dense TP split; **tp_group_ is never set for this target**" → the 35B-A3B lineage is
  **single-die-only in-tree today**. At NVFP4 its weights (~17.5-19 GB) do NOT fit one 16.4 GB
  die, so the lineage has NO multi-die story except layer-split/expert-parallel.

## 2. LAYER-SPLIT DESIGN SKETCH (if built)

### 2.1 Placement
- Die r owns layers [16r, 16r+16) full-width (all 24 Q / 4 KV / 48 GDN v-heads), via the
  EXISTING `StageSpec{16r, 16r+16, r==0, r==3}` + `tp_world=1` binding. The forward then runs
  the SINGLE-GPU arms everywhere (`tp_group_` unset) — zero in-layer AR by construction, no new
  kernel path, the exact arithmetic of the single-die forward.
- Embedding on die0 only (~0.7 GiB), full lm_head + final norm + MTP draft head + DFlash2 +
  sampler on die3 only. Head VRAM on die3: 2.44 GiB BF16 full vs 0.62 GiB shard today →
  mitigation options: NVFP4 head (~1.2 GiB; `draft_head_quant q4` precedent exists) or a 2-way
  head split die2+die3 with one finalize-only mini-AR (~1-3 ms/chunk, 128x cheaper than today's
  per-layer ARs).

### 2.2 Pipeline flow
- Prefill: die0 embeds chunk c, runs layers 0-15, hands hidden [5120, T] bf16 to die1, ... die3
  runs 48-63 + final norm + head. Steady state (C chunks in flight) all four dies busy; bubbles
  only at prompt start/end: wall(C) = (C+3) x stage(C-shape) + hops.
- Hop cost (banked transport): `W7_p2p_pt_row.txt` — **NO P2P on this platform** (canAccess=0
  all pairs, iommu=pt, V340 display class); host-relayed staged transport 3.14-3.17 GiB/s,
  10-KiB latency ~14 us. Per hop: T=128 hidden = 1.31 MB → 0.42 ms + 14 us ≈ 0.43 ms; 3 hops =
  ~1.3 ms/chunk ≈ **0.13% of a 990 ms chunk wall — negligible**, and overlappable in steady
  state. Decode: 10 KB x 3 hops ≈ 42-50 us/token vs a 24-37 ms round — 0.15%. (The tasking's
  "6.5 GB/s pinned" figure is the same class; the banked row's 3.15 GiB/s one-way staged is the
  conservative binding number — the conclusion is identical.)
- Transport work: TpGroup::send/recv exists but allows ONE pending task/rank and bf16 only —
  needs a 2-deep double-buffer (chunk c+1 produced while c in flight) and multi-column verify
  tensors [5120, k+1]. Small (0.5-1 d).

### 2.3 Decode + MTP
- Token (or verify batch of T=k+1<=8) walks die0→3 serially. Round wall = sum of 4 stage walks.
  Aggregate per-die work identical to TP4 (quarter-width x 64 layers = full-width x 16); the
  saving is the ~128 decode ARs (40 KB class, ~12.8 ms/round) + lockstep skew, minus ~50 us
  hops.
- **MTP coexists**: draft head, lm_head, argmax, sampler, ngram pool, DFlash2 all live on die3;
  the seed hidden crosses the pipeline; verify T-column batches walk as one payload. Acceptance
  parity risk is LOW (pipeline hops are bit-exact copies, not reductions — unlike the AR-fusion
  ulp-drift case, PLOG-068).

### 2.4 VRAM per die (derived from the 3820 MiB measured-manifest anchor)
| die | today (TP4) | layer-split | delta |
|---|---|---|---|
| die0 (stem) | ~3.73 GiB | ~2.7-2.8 GiB (layers + emb) | ~-1.0 GiB |
| die1/die2 | ~3.73 GiB | ~2.0-2.1 GiB (layers only) | ~-1.6 GiB |
| die3 (head) | ~3.73 GiB | ~4.9-5.1 GiB BF16 head (or ~3.7 NVFP4 head) | +1.2 GiB (or ~0) |
- KV per die: UNCHANGED (16 layers x 4 kv-heads x 256 = 64 x 1 x 256 today). GDN state
  unchanged (12 stage-local full-width = 48 quarter-width). Freed headroom on die0-2 is
  UNEVEN — the symmetric KV ring can grow only if die3's head is quantized/split first.
  Workspaces: full-width per-op intermediates on 1/4 the ops — roughly neutral, absorbed by the
  freed headroom. **No VRAM-law exposure: allocator stays the gate; no new constants.**

## 3. PRICE IT

### 3.1 The measured base (banked, current bin class)
Steady 128-token chunk at TP4, plen ~2k: **wall ~990 ms = GEMM ~770 + AR ~123 + body/gap ~97**
(AR=123 ms/990 = 12.4%: WO_RCCL_TUNING_desk.md:39, PERF_LOG PLOG-065; body/gap band 60-140:
W2_GAP_CAPTURE_desk + ledger; end-to-end ~112 t/s / 2075 tok: WO_AR_FUSION_design.md:421).
Earlier-generation log (P1/W2, bin a943225a): wall 1490 = gemm 770 + ar 68-73 + body 645 + gap 20
— the body mass has been curing all campaign; both decompositions bracket the same conclusion:
**AR is 5-12% of the wall; the GEMM is at bench class (2.9 TF/s/die); everything else is
per-op body.**

### 3.2 Prefill projection (the decisive arithmetic)
Stage time per die per chunk (16 full-width layers) = per-die TP4 work for 64 quarter-width
layers minus AR = **(990 - 123) ≈ 867 ms** — identical FLOPs and bytes by construction
(quarter-width x 64 = full-width x 16; same 5-problem GEMM census, same body ops).

| geometry (plen ~2k) | TP4 today (990/chunk) | layer-split, pipelined (C+3)x867 | verdict |
|---|---|---|---|
| C=2 (chunk 1024) | 2 x 3.96 s = 7.9 s | 5 x 3.47 s = 17.3 s | **2.2x SLOWER** |
| C=4 (chunk 512) | 4 x 1.98 s = 7.9 s | 7 x 1.73 s = 12.1 s | **1.5x SLOWER** |
| C=16 (chunk 128) | 16 x 0.99 s = 15.8 s | 19 x 0.867 s = 16.5 s | **~4% SLOWER** (~108 vs ~112 t/s) |
| C=64 (plen ~8k) | 63.4 s | 58.1 s | +8% (crossover C≈21 ≈ 2.7k tok) |
| C→∞ | 1.00x | 1.142x | asymptote = the AR share |
- Pipeline bubble tax at 2k is structural: a 2k prompt is 2-16 chunks; the +3 bubble stages
  swamp the 12% AR saving until C≈21. **llama.cpp's 152-160 does not contradict this**: their
  pp512 is ONE M=512 pass whose stages are ~pure GEMM at rocBLAS fp16 class (their end-to-end
  2.08-2.18 TF/s/die aggregates four ~77%-of-peak single-die GEMM stages); our M=128 tiled-NVFP4
  GEMM runs 2.9 TF/s/die already ABOVE their end-to-end rate — their win is M-scaling + a thin
  non-GEMM body, not the pipeline. Their own multi-die scaling is +3-24% (single-die gets ~all
  of it) — layer-split on their stack adds the AR-share-sized last few %, exactly as the
  arithmetic above predicts.
- Against the CURED TP4 path (chunk ladder M>=256 + AR fp8/copy-add [-49..-73 ms] + prefill
  graph capture; projected wall 780-880 ms/chunk-eq → 135-150 t/s at 2k): layer-split needs the
  SAME cures (its stages carry the same GEMM class and the same body ops — the capture prize is
  per-die there), then lands at ~4-9% BEHIND cure-path TP4 at 2k. Small tailwind not in the
  table: full-width single-die GEMMs halve the launch count (256→64/chunk) — inside the noise
  band.

### 3.3 Decode projection
Round (k=3): 129.38 ms = verify 112.8 + align 8.48 + draft-AR chain 7.63 (decode-landing row).
Layer-split removes the ~12.8 ms verify-AR mass + skew, adds ~0.05 ms hops:
~109-113 ms/round → **+13-16% round → decode 27-41 → ~30-46 t/s**. Modest positive; Branch-B's
priced decode win was 3-4.5 ms — same class, and the AR cure targets the same mass.

### 3.4 VRAM
Section 2.4: roughly neutral, die3-heavy; no refusal-constant exposure; KV unchanged per die.

### 3.5 Effort (honest number) — site census
| work | sites | est |
|---|---|---|
| `materialize_tp` layer-range/full-width placement mode (bypass shard_row_split; role table) | tp_load.cpp:323-712 | 1-2 d |
| StageSpec fan-out + ModelView local-slot verification (machinery exists) | bindings.cpp:586-679 | 0.5 d |
| make_rank per-stage DecoderState/KV/GDN/head-geometry/mtp re-homing; world-throw table rework | tp2_backend.cpp:705-1000 | 1-2 d |
| **SPMD→pipeline round restructure: 3 runners, 24 barrier sites, prefill driver, verify/accept/ngram/DFlash2 flow** | tp2_backend.cpp:1714-7900 | 3-5 d |
| Head/sampler/argmax collapse to die3 (allreduce_argmax→local; allgather class deleted) | ~20 backend sites + 3 context sites | 0.5-1 d |
| MTP/verify seed-hidden handoff + batched-runner lanes per stage | backend | 1-2 d |
| send/recv double-buffer + verify-width payloads | tp_group | 0.5-1 d |
| watchdog/liveness/traces/preflight world-semantics + NEW parity/boot cells | multi | 1-2 d |
| **Total to first GREEN world=4 layer-split serve** | | **11-19 desk-days (central ~14)** |
Compare: chunk ladder 2-3 d (banked design), AR option-0/fp8 stage-1 10-12 h (banked §11),
prefill graph capture 2-3 d (banked W2 design) — independently landable, sequentially summing
to ~5-7 desk-days for the full cure path.

## 4. THE HONEST COMPARISON + RECOMMENDATION

| | A: stay-TP4 + AR cure + chunk ladder + capture | B: layer-split refactor | C: llama.cpp prefill-only |
|---|---|---|---|
| prefill @2k | today 112 → **135-150 t/s** (projected, banked per-cure prices) | **~108 t/s** (4% behind today; -25%+ behind A) | n/a for OUR model (below) |
| decode | unchanged 27-41 (AR decode cure +3-4.5 ms optional) | ~30-46 (+13-16%) | their dense tg 19 < our MTP — LOSS |
| effort | 5-7 d total, each cure lands alone | **11-19 d**, one-shot big-bang | weeks (dual-engine) |
| risk | per-cure RED/GREEN, existing cells | new world-semantics everywhere; watchdog/wedge class; acceptance re-proof | KV/quant/scheduler split-brain; llama.cpp lacks our NVFP4 + hybrid-GDN (their bench models are standard-attention dense/MoE) |
| VRAM | unchanged | ~neutral, die3 +1.2 GiB | doubles residency |
| strategic fit | SD-1 (NVFP4@TP4 operational) | enables MoE lineage multi-die (real, but a DIFFERENT work order) | abandons SD-1 |

**RECOMMENDATION: Path A.** The premise "llama.cpp's e2e rate = our GEMM rate, therefore AR
dominates our deficit" is half-true and the half matters: AR is 123 ms of a 990 ms wall; the
cure path attacks AR (49-73 ms) AND the body (capture) AND the GEMM M-class (ladder) — all
banked designs, days not weeks, each independently falsifiable. Layer-split attacks ONLY the AR
share, forfeits the within-layer 4-way parallelism that pays for it, and pays a pipeline tax
that is -4% at exactly the campaign's 2k target. **Adopting layer-split to chase llama.cpp's
number is buying their bottleneck-structure (pipeline) without their bottleneck-remover
(M=512 GEMMs + thin body).** Path C is incoherent for this stack: same-process dual-engine
would duplicate KV pools/schedulers/VRAM, llama.cpp cannot read our NVFP4 artifact or serve the
48-GDN hybrid, and their decode (19) loses to our MTP (27-41) — we would lose the side we win.
**Keep layer-split PARKED as the designed multi-die architecture for the 35B-A3B MoE lineage**
(single-die today, no dense TP, ~4.4-4.8 GB/die NVFP4 fits, MoE is naturally AR-free) — that is
the one scenario where its effort buys something TP4 cannot.

## 5. PRE-REGISTERED GATES

Path A (recommended; first implementation desk's pass bars):
- **G-LADDER (chunk 256/512 re-leg)**: same-window A/B, cure config, plen 2075, temp 0, 2 boots.
  PRIMARY: end-to-end prefill t/s >= +8% over the chunk-128 leg on the same bin. Byte-identical
  outputs (standing parity cell), boot battery GREEN, workspace growth at T=512
  cudaMemGetInfo-measured (VRAM law: no constants).
- **G-AR (option-0 copy-add, then fp8 ring)**: RED/GREEN per closure law — RED = ar_us column
  (standing `vt.ar` event pairs) + wall on the pre-fix bin; GREEN = wall -30..-73 ms/chunk within
  window noise, output byte-identity (bf16 path), MTP acceptance within noise (Q3 re-fire
  clause), k=2 ring stays default (one-shot route stays root-looked, WO_ONESHOT_ROOTLOOK.md).
- **G-GRAPH (prefill capture)**: the four W2 kill-switches (per-chunk allocation census,
  dynamic-shape audit, ar_watchdog false-kill audit under replay, OPTRACE disposition) all
  pass BEFORE capture code; RED/GREEN = steady-chunk wall + [PREFILL-BODYSUM] unbr before/after;
  byte-identical outputs.
- **Composite bar**: >=150 t/s at 2k with decode >=27 t/s unchanged — reaching it empirically
  closes the layer-split question on its own terms.

Layer-split re-entry triggers (pre-registered — any ONE reopens, none fire today):
- **RS-1 MoE adoption**: campaign charters the 35B-A3B lineage at multi-die → layer-split is the
  architecture study for THAT model (this doc's design + inventory carry over).
- **RS-2 long-window mix**: serving windows move to prompts >~3k tokens at T=128 (C>21
  crossover) AND Path A has landed (else both architectures lose to the cure).
- **RS-3 TP4-infeasible model**: a target whose NVFP4 weights+KV exceed 4-die TP4 placement
  (the original TP4 justification inverted).
First implementation desk pass bars IF triggered: (1) layer-range materialize produces per-die
full-width stage weights (sha-stamped artifact); (2) world=4 pipeline serve boots GREEN with a
device parity cell vs single-die reference on a <=16-layer slice; (3) prefill t/s >= SAME-WINDOW
uncured TP4 at plen 2075 (the -4% projection must flip before further investment); (4) decode
round wall within +-2% of TP4 baseline; (5) MTP acceptance within noise, same corpus.

## 6. HONESTY ROW

- The 867 ms stage and the (C+3) pipeline arithmetic are DERIVED from the banked 990/770/123/97
  decomp, not a layer-split measurement — no such binary exists. The identity
  (quarter-width x 64 = full-width x 16 per die) is exact in FLOPs/bytes; per-op efficiency at
  4x N is interpolated from the same SWEEP family (closed), not measured at n=16384.
- The 990 ms wall (RCCL-desk class) and the 1490 ms wall (W2 log bin) are different bins; both
  decompositions are cited; the verdict is invariant across both (AR share 5-12%).
- llama.cpp model identity: their 27B dense Q4_1 is a DIFFERENT, standard-attention
  architecture (our 27B is the 48-GDN hybrid); their MoE row is qwen3-30B-A3B, not our
  35B-A3B hybrid variant. Both rows are existence proofs for M=512 GEMM class on these dies,
  not for layer-split on our model.
- Embedding format/composition of the 3820 MiB anchor is derived (role table + shapes), not
  per-tensor inspected (artifact inspect.py needs torch; unavailable here). Bounds are tight
  (emb 0.6-2.4 GiB depending on format shifts die0's delta, verdict unchanged).
- Decode +13-16% projects the W2-class 30-40 KB AR floor onto the round; the per-AR decode floor
  is banked-class (98-101 us RTT), the 128-AR sum at verify width is not directly measured.
- Line numbers are THIS worktree @ 90c9ecf22 and drift.

## BASES (anchors consumed)

`docs/amd/REMAINING_ITEMS.md` (strategic update, 90c9ecf22) · `docs/amd/PREFILL_DECOMP_2026-09-17.md`
(chunk census, G_tiled ~530 ms class, 1556 GF/chunk/rank, 128 AR/chunk x 1.31 MB) ·
`docs/amd/W2_GAP_CAPTURE_desk.md` (P1 measured per-chunk rows: 1490 = 770+70+645+20; unbr 60-84;
finalize 1376 open) · `docs/amd/WO_RCCL_TUNING_desk.md` + PERF_LOG PLOG-065 (123 ms/990 wall,
host-SHM-bound, env-immovable) · `docs/amd/WO_AR_FUSION_design.md` (:421 llama.cpp row + our
112 t/s/27-41; fp8/copy-add pricing -49..-73 ms, 10-12 h) · `docs/amd/VERIFY_DECOMP_2026-09-17.md`
+ decode-landing row (round 129.38 = 112.8+8.48+7.63; k=2 keep, Branch-B banked-unpromoted) ·
`results/amd/coherence/W7_p2p_pt_row.txt` (NO P2P; staged 3.14-3.17 GiB/s, 14 us) ·
`docs/amd/G18_firstlight_FAILURE_PLAYBOOK_agent4.md` (3820 MiB/rank measured-manifest) ·
code: bindings.h/.cpp, tp_load.cpp, tp_group.h/.cpp, weight_shard.h, text_context.h/
text_context_impl.h, tp2_backend.h/.cpp, tp_engine.h, variant.cpp (35b) — paths in §1.
