W24 PREFILL-AT-DEPTH DESK RECEIPT - 2026-09-24
branch amd/v340-port-v2 (HEAD 4bf711608 at desk open), ZERO-GPU desk behind
the U1 depth window; U1 logs read read-only (u1_window_console.log,
u1_depth_*_thermal.log)

MISSION: served prefill collapses with depth (218 t/s short, 170.35 t/s
@10k u1_10k, 13.9 t/s @65k E-124). Decompose WHERE the time goes at depth
(quadratic vs excess), then design the prefill-side extension of the fa40
q4_0-direct arm (today decode-only) so a desk can implement + bench it
when dies free.

MODEL GEOMETRY OF RECORD (GGUF header re-parsed this desk):
- Qwen3.8-27B-ASCII-P1M: 65 blocks, 17 FULL-ATTENTION blocks (attn_k.weight
  count) + 48 SSM/GDN blocks (hybrid), n_head 24, n_head_kv 4, head_dim
  256, emb 5120. TP4 serve: 6 Q heads + 1 KV head per die, gqa_ratio 6.
- Served config (run_u1_window.sh:55, canonical-200k-inline): -c 200000
  -b 512 -ub 512 -ctk q4_0 -ctv q4_0 -fa on, GGML_CUDA_FATTN_TILE_Q40_DIRECT=1
  in BASEENV (decode arm on).

1. A2 - THE DECOMPOSITION: THE COLLAPSE IS THE TILE SCAN, CLOSED FROM
   FIRST PRINCIPLES

1.1 Prefill execution structure (code of record)
- Prefill runs in 512-token ubatches (llama-context batch split). Each
  ubatch rebuilds the graph; per full-attention layer per die one
  flash_attn_ext launch (17 launches per ubatch per die) plus per-layer
  K/V q4_0->f16 pool conversions.
- K/V views handed to attention are n_kv-sized, padded to 256
  (llama-kv-cache.cpp:1243-1258 get_n_kv, max(n_pad,256); get_k :1259),
  so per-launch conversion + attention cost grows stepwise with fill.
- The tile instance at prefill M: on HIP the 64-col branch is gated
  DKQ<=128 (fattn-tile.cuh:1429-1441), so DKQ=256 falls to the
  cols_per_block=16 branch (:1458-1468): instance
  flash_attn_tile<256,256,8,2> with the AMD config row
  (256,256,16 -> nthreads 256, occ 2, nbatch_fa 32, nbatch_K 128)
  (fattn-tile.cuh:214-218). Grid per launch: ntiles_x = M/8, 3 z-tiles.
- launch_fattn (fattn-common.cuh:972) converts the FULL current-depth K
  and V q4_0 tensors to f16 before the kernel when need_f16_K/V
  (:1019-1064): the "f16 pool" (dst-append scratch, fattn.cu
  get_alloc_size). This is the W15-measured 3.98 us/1k-token/launch line.

1.2 The empirical law (two clean points, both fresh-thermal)
- Served short cell: 218 t/s @4k prompt (E-124): 4.59 ms/token avg.
- u1_10k (this window, 2026-09-24 11:48): 10235 tokens in 60.08 s =
  170.35 t/s = 5.87 ms/token avg.
- Model c(D) = L + q*D with request-average c_avg(M) = L + q*M/2:
  L = 3.73 ms/token, q = 0.42 ms/token per 1k depth. Fit closure: both
  points land within 0.5% (4.59 vs 4.59; 5.87 vs 5.88).

1.3 First-principles closure (no free parameters)
- W13 census (6964-token prefill, 36.84 s wall): Cijk GEMM 41.4%,
  flash_attn_tile 29.4%, nccl 16.9%, other (norms/gdn/copies) 12.3%;
  dies 96.6% busy - host/graph overhead < 3.4% (meta/rebuild churn
  CLOSED as minor).
- Linear floor from the census: 2.19 (GEMM) + 0.89 (nccl) + 0.65 (other)
  = 3.73 ms/token. MATCHES L. These classes are depth-linear (dense GEMMs
  on 512x512 tiles, allreduces per layer, KV set_rows) - they do NOT
  collapse; they set the floor rate 268 t/s.
- Quadratic term from the banked decode anchor: the tile class per round
  @7168 is 12.49 ms (E-124 table) = 17 layers x 722 us/launch. The same
  per-row scan rate applies at prefill (W13: 10.47 s/die / (14 ubatches
  x 17 layers) = 44 ms/launch at avg depth 3482 = 12.6 us/KV-token;
  decode: 722 us/7168 KV-tok x 4 Q-rows = 25.2 ns/row-KV-token; prefill:
  44 ms / (512 rows x 3482 KV-tok) = 24.7 ns/row-KV-token - SAME RATE,
  both instances latency-bound per row-scan, extra blocks buy no
  throughput).
- Prefilled M tokens = sum over ubatches of D_i = M^2/(2*512) row-KV
  scans per layer x 17 layers x 25.2 ns = 0.219 ms per ubatch per 1k
  depth -> q = 0.42 ms/token per 1k. MATCHES the empirical q. THE MODEL
  IS THE MEASUREMENT: c(D) = 3.73 + 0.42*D/1000 ms/token.

1.4 Expected prefill(D) curve (fresh-thermal) vs measurements

  depth M   model wall   model avg t/s   measured          status
  ------    ----------   -------------   --------          ------
    4096      18.9 s        217           218 (E-124)      EXACT
   10235      60.3 s        170           170.35 (u1_10k)  EXACT
   51200     ~759 s        ~67            (u1_50k landing) PREDICTION
  102400    ~2642 s        ~38            (u1_100k queued) PREDICTION
  153600    ~5285 s        ~28            (u1_150k queued) PREDICTION
  199000    ~9058 s        ~22            (u1_199k queued) PREDICTION

  (Predictions banked BEFORE the cells land; deltas = thermal multiplier.
  W15 served points already show that multiplier: 46 t/s @34k vs model
  55 = 1.2x, 15 t/s @62k vs model 33 = 2.2x - thermally self-reinforcing,
  NOT a second mechanism. The runner's D^1.5 wall model embeds it: at 10k
  it predicts 35 t/s vs 170.35 fresh - 4.9x pessimistic.)

1.5 VERDICT: how much of the collapse is EXCESS?
- The quadratic term (tile scan, q) is 0% of the wall at D->0, 39% of
  u1_10k, and grows to ~91% at 199k. It is algorithmically legitimate
  O(N^2) attention (M^2/2 row-KV pairs) - BUT the instance executes it
  at 25 ns/row-KV-token against a ~2 ns bandwidth-perfect bound
  (1024 B unique f16 K+V per row-KV-token / ~500 GB/s): ~12x headroom
  that bytes, not FLOPs, own.
- EXCESS inside the tile class: the f16 pool read amplification
  (2048 B/KV-token written+re-read vs 288 B q4_0 = 7.1x). The W17 decode
  arm realized -37.5%..-39.4% wall FLAT in depth from exactly this cut;
  that is the banked expectation for the prefill instance too.
- EXCESS outside the tile class: the per-launch pool conversions
  (17 layers x 2 launches x 3.98 us/1k x sum(D_i)) = 14.5 ms @10k,
  0.35 s @50k, 5.2 s @199k = 0.02-0.6% of wall. NEGLIGIBLE but free.
- EXCESS at depth: the W15 x1.4 late superlinearity beyond 36k (f16 pool
  leaving L2) is pool-byte-specific - direct q4_0 reads shrink it.
- NOT this arm's: thermal multiplier (W22 desk: clock floors).
- RESIDUAL (the optimization target) = f16-pool tax inside tile
  (est -37.5% of tile) + conversion launches + late superlinearity.
  At fresh-thermal walls: 8.8 s of 60.3 @10k, ~213 of 759 @50k,
  ~847 of 2642 @100k, ~3179 of 9058 @199k.

2. A3 - THE PREFILL q4_0-DIRECT EXTENSION DESIGN

2.1 Does the var-11 template cover prefill M? YES - structurally.
- The q40 machinery lives entirely in the J (KV) pipeline:
  flash_attn_tile_load_tile_q40 (fattn-tile.cuh:534-597) is templated on
  (I=nbatch_fa/J-head-dim slice, J=nbatch_K) ONLY - no ncols1/M
  dependence; it writes the SAME shared half2 tiles the f16 loader
  produces, so the KQ/VKV pipeline downstream is untouched. The decode
  instance (<256,256,4,2,.,11>) and a prefill instance
  (<256,256,8,2,.,11>) share every line of the q40 path.
- static_asserts pass at prefill shape: DKQ,DV % 32 == 0 (:1076);
  nbatch_K=128 % QK4_0 == 0 (:536).
- oob/mask: prefill uses the same causal mask + oob_check template arms
  as decode; K->ne[1] is 256-padded (get_n_kv) so the J-tiling pattern
  matches decode's. KV_max pre-scan is skipped at both (Q->ne[1] < 1024
  gate, fattn-common.cuh:1085).
- The ONLY blocker is the launch gate fattn-tile.cuh:1610
  (Q->ne[1] <= 4): prefill (Q->ne[1]=512) falls through to the f16
  cols_per_block=16 branch and instantiates q40_var=0.

2.2 The extension (new arm, ~15 lines in launch_fattn_tile_switch_ncols2)
- In the same `if constexpr (DKQ == 256 && DV == 256)` HIP block, add a
  second gate beside :1608-1625:
    static const bool q40_prefill = getenv("GGML_CUDA_FATTN_TILE_Q40_PREFILL") != nullptr;
    if (q40_prefill && use_gqa_opt && Q->ne[1] > 4
        && K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0) {
        nwarps    = get_nthreads(DKQ, DV, 16, cc) / warp_size;  // = 8
        nbatch_fa = get_nbatch_fa(DKQ, DV, 16, cc);             // = 32
        launch_fattn<DV, 8, 2>(ctx, dst,
            flash_attn_tile<DKQ, DV, 8, 2, use_logit_softcap, 11>,
            nwarps, nbytes_shared, nbatch_fa, false, false, false, warp_size);
    }
  ncols1=8/ncols2=2 mirrors the f16 fall-through instance exactly
  (:1458-1468), so launch geometry and the parallel-block scan match the
  f16 arm; need_f16_K/V=false deletes the pool conversion AND the
  dst-append scratch (graph alloc stops growing with n_kv).
- use_gqa_opt preconditions (:1525) already hold at prefill on AMD
  (gqa_limit=INT_MAX; mask present causal; K->ne[1] % 256 == 0). Any
  non-conforming shape still falls through to f16 unchanged.
- ENV SPLIT: keep GGML_CUDA_FATTN_TILE_Q40_DIRECT exactly as committed
  (decode arm, validated + in served BASEENV). New env
  GGML_CUDA_FATTN_TILE_Q40_PREFILL gates ONLY the prefill arm:
  (a) independent A/B against U1's baseline cells without re-running
      decode; (b) exactness isolation - a prefill regression cannot
      pollute the validated decode arm; (c) both envs set = candidate
      serve config, one boot.
- SAME template var (11, scalar half2 stores). The W17 miscompile is an
  8-byte store-granule codegen defect; scalar stores were exact at every
  depth AT THE DECODE INSTANTIATION. A new instantiation = new codegen:
  exactness must be RE-PROVEN (next), but v11 is the only arm to try -
  all other store shapes failed identically (W17).

2.3 Exactness risks at prefill + oracle recipe (die 3, lock-arbitrated)
- Risk 1 (primary): the <256,256,8,2,.,11> instantiation may miscompile
  differently (per-instantiation codegen). W17 staged-oracle tooling
  (GGML_FATTN_Q40_DBG KT/VT/KQ dumps) applies unchanged.
- Risk 2: parallel_blocks from occupancy of the new instantiation may
  differ from the f16 arm -> KV-split combine order -> fp dust, not
  bit-exactness. Classify per W16/W17 rules: EXACT (bit-identical dst),
  DUST (max_rel <= ~1.3e-3, basedup bit-exact - needs owner sign-off),
  GARBAGE (fail).
- Risk 3: partial tiles at prefill edges - tail ubatch M=268 (33 x 8 + 4
  Q rows) and non-multiple-of-256 n_kv boundaries; the causal mask must
  zero oob J after q40 dequant exactly as the f16 path does. Include an
  oracle case at used=10235 (n_kv pads to 10496) and M=268.
- Bench recipe (extends W17 bench_attn_real.cu, already carries the
  q40_kv template + depth CLI):
    arms: base = f16-pool prefill instance <256,256,8,2> (served today),
          v11p = <256,256,8,2,.,11>, basedup x2 determinism control
    M:    512 and 268 (tail)
    depths: 7168 10240 32768 65536 102400 199936
    per arm: staged oracle dumps first (KT/VT/KQ bit-compare), then dst
    memcmp vs base, then timing (niter sized so a 199k launch ~90 ms
    still gets 10 iters, rep 0 discarded, W17 protocol)
    record: pb printed from occupancy, per-launch med us, oracle class.
- Acceptance: oracle EXACT-or-DUST at all cells AND v11p >= 30% faster
  per launch at 10k+ -> proceed to served window. GARBAGE anywhere ->
  stop, file codegen characterization, do NOT sweep more variants (W17
  law: scalar stores are the only exact shape).

2.4 Prize estimate (fresh-thermal walls, tile -37.5% + conversions free)

  depth M   baseline        post-arm        avg t/s         recover
            wall  (t/s)     wall  (t/s)     gain            of collapse
  --------  --------------  --------------  --------------  -----------
   10235     60.3 s (170)     51.5 s (199)   +17%           ~74%
   51200    759 s   ( 67)    546 s   ( 94)   +39%           (vs model)
  102400   2642 s   ( 38)   1795 s   ( 57)   +51%
  199000   9058 s   ( 22)   6048 s   ( 33)   +50%

  Post-arm law: c'(D) = 3.73 + 0.26*D/1000 ms/token. Upside NOT counted:
  the L2 superlinearity recovery beyond 36k and any latency relief from
  7.1x fewer bytes (decode measured -39..-45% at deep, better than
  -37.5%). Downside NOT counted: served thermal multiplier (applies to
  both arms; paired cells control it).
  The tile class is 39% of the wall @10k but 91% @199k - the arm's
  leverage GROWS with depth; it is the only lever that bends the 200k
  prefill wall below the hour class on this card.

3. SERVED-WINDOW PLAN (U2, when dies free)
- Prereq: U1 window COMPLETE (baseline cells 10k/50k/100k/150k/199k
  banked on the f16-pool prefill arm) + bench oracle PASS (2.3).
- Boot: BASEENV (decode arm on) + GGML_CUDA_FATTN_TILE_Q40_PREFILL=1,
  same canonical config; re-run U1 cells (10k/50k/100k minimum).
- Metric: prompt_tps per cell, paired vs U1 baseline (same recipe,
  cache_prompt=false, void-gate adjudicated); decode_guard per cell must
  hold the E-137 ratchet (decode arm untouched = canary).
- Model expectation: +17% @10k, +39% @50k, +51% @100k. Kill criteria:
  < +5% @10k, or any oracle/GARBAGE regression, or decode_guard breach.
- Provenance: stamp every cell (arm-identity gate as U1 does).

4. DESK LOG
- 12:08 desk open, zero GPU. A2 closed from GGUF re-parse + W13 census +
  E-124 anchors + u1_10k; model closes with zero free parameters
  (25.2 vs 24.7 ns/row-KV-token decode vs prefill; L matches the W13
  linear census to 0.1 ms).
- A3 closed: template coverage verified (J-pipeline only), gate + env
  split + oracle + bench specified; prize banked.
- Open: u1_50k/100k/150k/199k land every ~40-90 min - update section 1.4
  measured column before the final commit of this receipt.
