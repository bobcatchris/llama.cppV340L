# TP3 V340L Optimization Plan - Qwen3.8-27B-ASCII-P1M @ 200k

Date: 2026-09-21. Branch: amd/v340-port-v2 (pin lineage d9df11006 + 24e9e5dc4).
Horizon: up to 6 months, no programming-hour budget.

## Goal and constraints

Goal: >2x decode (17.8 -> 36+ tok/s) and >2x prefill (96.5 -> 200+ tok/s),
holding at BOTH short (2k) and long (10k+) context, in sustained real use.
Config of record: TP3 tensor split + draft-mtp d3, q4_0 KV, FA on, 200k ctx.

Constraints (decided 2026-09-21):
- Weights bit-exact. No requant. (Optional 5-min sanity check: download the
  real Q4_1 ASCII GGUF from HuggingFace and bench it once; expectation low,
  the old 144-vs-97 receipt was a near-zero-context number and capacity at
  200k is why we run the small file.)
- No power-cap or memory-clock games. Better cooling is installed; one
  afternoon of cold/warm/hot re-measurement to refresh the thermal function
  is all the thermal work budgeted. Vega56 BIOS on one card raises power and
  core clocks but memclk will not hold 1k MHz - mclk 945 -> ~183 GB/s/die
  is the permanent bandwidth wall, and everything above it is ALU.
- 4th die is a free helper: draft device, dev/bench cell, or precompute -
  never a 4th TP rank (TP4 measured worse: decode 15.2-15.5 vs 17.8, and AR
  grows with world).

## Baseline truth (measured, see ninfer-history + amd-port receipts)

- Decode cycle at MTP d3: ~175 ms for 3.12 accepted tokens. Kernel census:
  86.2% mul_mat_vec_q (MMVQ), 4.6% quantize_q8_1, 3.3% flash_attn. Zero
  rocBLAS. Effective MMVQ bandwidth ~30 GB/s vs 183.8 GB/s/die measured
  ceiling -> fat-tier-loss verdict: the deficit is kernel execution, not
  physics. Ideal cycle ~21-25 ms -> ~7x headroom on the dominant kernel.
- Prefill: 67.4% rocBLAS fp16 GEMM at ~1.7-2 TF/s/die vs ~21.5 TF/s/die
  packed-fp16 peak (no MFMA on gfx900; VALU fp16x2 only), 24.4% flash_attn,
  allreduce 12-21% of wall. AR today = meta-backend butterfly staged through
  host memory (no P2P, canAccessPeer=0 all pairs, ~50 us/hop latency, PCIe
  3.0 x8 per card ~15.75 GB/s/dir). The fast one-shot host-pinned AR design
  (allreduce.cu) exists in-tree but is compiled out on HIP
  (`#if !defined(GGML_USE_HIP)`).
- GDN: 48 of 64 trunk layers are recurrent; graph shows per-layer chains of
  small kernels (ssm_conv, ssm_scan, l2_norm, silu/softplus/mul glue) that
  are unmeasured on this port at decode batch 4.
- MTP draft: per draft step the head re-runs a full attention block + FFN +
  the 0.5 GiB q6_K LM head on the same 3 dies, serialized before the verify
  pass. 3 draft steps per cycle at accept 0.708.
- Thermal: new cooling; expected flatter cold/hot curve, to be re-stamped
  once (W1). Historic function was 1.65-2.8x cold/hot - verify, don't assume.

## Method

Every optimization cell runs the ninfer-kernel-opt protocol: lock env,
reproduce baseline (+-2%), one rocprof capture squeezed to death, classify
the bound (weight-bytes floor AND compute floor, plus the four-ablation kit
where counters are unreliable on gfx900), pull ladder rungs in yield order,
gate every arm same-BIN behind an env switch with >=3 interleaved reps and an
in-run control (>1% drift = void session), bank a receipt for wins AND
negatives, stop only at a named measured ceiling. The 4th die is the
permanent dev/bench cell so serving windows are never disturbed.

## Workstreams

### W0 - Harness (week 1-2)
- bench.sh: cold/warm-stamped llama-bench + server protocol (3-min idle
  rule), 2k/10k/50k/200k ladder, decode+pp+acceptance per run.
- Census script: rocprof around one canonical request per phase (decode-
  verify, draft, prefill), kernels ranked by total seconds, plus GPU-idle
  timeline (launch-gap share) and AR latency histogram.
- Correctness battery: 8-needle long-context check + MTP acceptance canary
  (gate 0.708 at 2k, fail at -4pp). Every promoted arm re-runs it.
- Route banner: per-config print of split/world/spec/quant so receipts
  self-attribute.

### W1 - Config and environment wins (week 1-3, existing binary)
- --spec-draft-backend-sampling (GPU-side draft sampling; kills per-step
  vocab-logits host roundtrips, 129k-wide f32).
- ubatch sweep 128/256/512 at 2k/10k/200k with thermal stamps.
- Host tuning: CPU affinity for HIP driver threads, NUMA local to the PCIe
  roots of the three dies; env matrix (GGML_CUDA_ALLREDUCE,
  GGML_CUDA_PDL, OP_OFFLOAD_MIN_BATCH).
- One-afternoon thermal re-stamp (cold/warm/hot prefill+decode) with the
  new cooling; update the thermal leaflet numbers.
- Optional: the 5-min HF Q4_1 bench (closes the format question).
- Expected: decode 19-22, pp 105-125.

### W2 - MMVQ decode engine (month 1-3, the big one)
Target: mul_mat_vec_q for this file's exact (qtype, K, T) cells at T=1..8.
Bound expectation: latency/occupancy + decode-ALU (LUT unpack chains on
wave64, sub-wave grids on 56 CUs). Ladder, in order:
1. C2/C1 K-split (cross-CTA first; ninfer measured +32-58% on sub-wave
   serial chains; many-wave rows expected dead - band by measurement).
2. A2 128-bit coalesced loads (uint4-per-lane; sector math).
3. A3 LDS double-buffer prefetch (cp.async concept, software pipelining).
4. B1 v_perm_b32 byte-permute decode atom (PRMT analog) for the 3-bit iq
   LUT chains; count SASS inst/weight before and after.
5. C4 launch_bounds/occupancy budget (+1 KiB smem pad law).
6. C7 per-shape qualification: pin winners for K in {5120, 6144, 10240,
   12288, 17408} x T in {1..8}; fail loud off-list.
Priority by bytes: iq3_s (3.39 GiB), iq4_xs (3.12), iq3_xxs (2.35), q4_K
(1.58), q6_K head (0.55). All arms behind env switches; oracle = bit-exact
math so no acceptance price.
Exit: >=100 GB/s effective on the top-3 types, decode share re-censused.

### W3 - GDN chain + glue (month 2-3)
- Census the recurrent chain at decode batch 4 on die 4 (dev cell): kernel
  list, gap share, per-op us for ssm_conv / ssm_scan / norm / glue.
- Fuse per-layer elementwise glue (norm+silu+mul, sigmoid+mul) into the
  neighboring ops; single fused GDN-recurrent kernel candidate if the
  census says the chain dominates.
- Verify cparams.fused_gdn_ar / fused_gdn_ch are active on this port.
- Kill quantize_q8_1 (4.6% of decode): emit q8_1 activations directly from
  the producing norm kernel where the MMVQ path allows.
- Prefill half: chunked GDN scan (ninfer measured x16 on the dn column) if
  the prefill census shows ssm-scan share.

### W4 - Allreduce fast path (month 2-4)
- Port allreduce.cu (2-GPU pinned-host one-shot) to HIP; generalize to 3
  ranks; wire as the meta-backend AR for small payloads, butterfly kept as
  fixture behind env. Respect the CLOSED SET: no one-shot wedges (acceptance
  drift), no fp8-ring numerics, no copy-add.
- Then chunked pipelined butterfly for prefill (split M; reduce slice k
  while computing k+1) to hide the 12-21% AR share.
- Expected: -10-20 ms/decode cycle at ~130 ARs x ~40 KB; prefill +10-15%.

### W5 - Prefill engine (month 2-5)
- GEMM: bench rocBLAS on the exact shapes (M=ubatch, K=5120/17408, N=per-die
  row share). If <50% of achievable, write packed-fp16 VALU tiles per the
  GEMM_CONSTRAINED_TILES_2026-09-18 spec (32 flop/cyc/SIMD bound; 56 CU x
  4 SIMD x 32 = 10.75 TF/s/die fp32-class realistic top) instead of fighting
  Tensile. Pre-quantize activations once per tile, not per GEMM.
- FA: 13 fattn-tile instances exist for hd=256 GQA 24:4; bench all, retune
  the winner for 56 CUs (tile/wave sizing), target the 24.4% share.
- Long-context: measure pp at 50k/100k/200k after W4 overlap lands; the
  mild context scaling (17.8@2k -> 15.6@10k) should hold to 200k.

### W6 - 4th die as draft device (month 3-5)
Primary: run the draft-mtp context (nextn block + shared head + its KV,
~0.9 GiB) on die 3, overlapping draft steps with the verify pass on the TP3
dies. Needs: device pinning for the draft context in speculative.cpp, small
KV/state sync (last-token hidden state + accepted tokens), and the draft
graph's 0.5 GiB LM head staying on die 3. Removes draft time (roughly a
third of the cycle at accept 0.708, 3.12 tok/step) from the critical path.
Fallback if overlap stalls: die 3 stays the dev/bench cell (already paid
for by W0-W2).
After W2-W4 land, re-sweep TP2/TP3/TP4 - the balance shifts once kernels
stop dominating, and KV headroom at TP2 returns if MTP moves off-box.

### W7 - Long-context gating + delivery (month 4-6)
- Full 2k/10k/50k/100k/200k curves at every milestone; q4_0 KV kept
  (3.9 GiB at 200k; hybrid model scales mildly).
- Watch logits buffer at -b 256+ (512 x 129,272 x f32 = 264 MB); confirm
  inp_out_ids gating active in the MTP path.
- Sustained-rate characterization (the user-facing number): 30/60-min soaks
  at 10k and 200k, receipts with cold/hot labels.

## Milestones

- M1 (end week 3): harness live, W1 banked. Decode 19-22, pp 105-125.
- M2 (end month 2): W2 first blood (K-split + wide loads on iq3_s/iq4_xs/
  iq3_xxs), W4 initial port. Decode 26-32, pp 130-160.
- M3 (end month 4): W2/W3/W4 complete, W5a+5b in. Decode 35-45, pp 200-260.
  Both 2x gates green at 2k AND 10k.
- M4 (month 6): W6 draft overlap + sustained characterization. Decode 40+
  stretch, receipts complete.

## Verification gates (every arm, no exceptions)

Same-BIN env-switch A/B, >=3 interleaved reps, in-run control, median per
axis-class, noise classes <2% not-movement / 2-5% must-label / >=5%
unambiguous. Oracle bit-equality for exact kernels; acceptance price +
served-content equality for anything that changes numerics; needle battery
+ MTP canary before any promotion; old arm stays alive behind its switch;
every rate row self-attributes; receipts in docs/amd-port/results/.

## Closed set (do not re-open without new evidence)

graph capture (twice), fp8-ring AR numerics, one-shot AR wedges, copy-add
AR, pinned-high clocks (1.3-1.8x backfire at 110 W), kT-widening without
occupancy data, smem LUT decode post-K-split, RCCL env matrix, static-count
projections as rate claims, requantizing the shipped file.

## Risks

- rocprof counter limits on gfx900: the four-ablation kit needs no hardware
  counters (pure-consume, k-scaling, T-scaling, CTA-tiling).
- Meta-backend surgery breaks TP3 boot: arms staged behind env; butterfly
  kept as fixture; die 4 dev cell catches this before serving windows.
- Decode ceiling ~60-90 tok/s (memclk-bound once kernels are fixed): 2x has
  margin even if W6 slips; prefill ceiling is ALU-bound -> custom tiles
  budgeted in W5.
- Upstream drift: pin lineage, cherry-pick only; the port is 2 code commits.

## First week, concretely

1. W0 scripts (bench protocol, census, control cells) on die 4.
2. W1 flag sweeps on the existing binary (backend-sampling, ubatch, env).
3. W2 P1 classification: census the decode pass; run the byte-floor +
   ablation kit on the iq3_s 17408x5120-class and iq4_xs cells.
4. File receipts; update this doc's baseline numbers with the W1 re-stamp.

## Appendix - pointers

- Launch line of record: HIP_VISIBLE_DEVICES=0,1,2 llama-server -m
  Qwen3.8-27B-ASCII-P1M.gguf -ngl 999 -sm tensor -c 200000 -b 512 -ub 128
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp (d3 default).
- Build: cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900
  -DCMAKE_HIP_ARCHITECTURES=gfx900 -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0
  -DLLAMA_CURL=OFF (ROCm 6.2.0 pinned; FP8 gate patch required).
- Measured constants: 183.8 GB/s/die D2D ceiling; mclk 945 MHz; sclk floats
  1269-1500; AR RTT ~50 us/hop host-staged; weights 5.7 GiB/die at TP3;
  200k boot uses 8419/8131/8130 of 8573 MiB/die.
- Quant recipe: docs/amd-port/QUANT_RECIPE.md + quant/ascii-p1m-tensor-types.txt
- Method source: ninfer-kernel-opt skill (kernel ladder + gating laws).

## PROGRESS LEDGER (append-only)

Law: this ledger is the current state of the campaign. New entries are
APPENDED at the end, never edited or removed; corrections are new entries
that supersede. Each entry: id, date, one-line state, receipt pointers.
Agent-work law: agents work in git worktrees on side branches (never the
main tree), one worktree per workstream; served world (dies 0,1,2) belongs
to Gemini's guard battery, die 3 is the dev cell.

- E-001 2026-09-21 W0 baseline reproduced: TP3-tensor 10k no-spec pp 94.69 /
  tg128 12.24 (new receipt cell). results/W0_baseline_tp3_tensor_10k.txt
- E-002 2026-09-21 W0 decode census banked (rocprofv3; absolute-path lesson):
  MMVQ 77% of decode kernel budget; allreduce only ~6% (W4 demoted); GDN
  5.6%; targets named. Correction: the 249 us/call rows are FUSED-GLU
  variants, not small_k. results/W0_census_decode_receipt.md
- E-003 2026-09-21 Guard suite landed (Gemini, modeled on dual_5060_ti_ninfer)
  and validated GREEN on first run: pp 93.77 / decode 15.31 / accept 0.6667
  (== PLOG-101 receipt). Hardening assigned: thermal sideband, version-lock,
  prefill-first ordering, cache_prompt=false. docs/amd-port/tests/
- E-004 2026-09-21 Fusion A/B banked: fused-GLU MMVQ stays (decode 16.21 vs
  14.66 t/s mean without it, -8% consistent); prefill leg VOID (monotone
  thermal drift 49.8->55.9 across arms). results/AB_fusion_off_2026-09-21.md
- E-005 2026-09-21 W2 microbench cell banked (bench_mmvq_gfx900.cu, die 3):
  shipped clone 105.2 GB/s = 1.75x off sequential floor. SEVEN schedule
  rungs NEGATIVE: LDS-LUT, balanced slots, NW4, NW8, multi-row r2/r4/r8,
  NW1 r4/r8, ILP2 x2. results/W2_mmvq_cell_2026-09-21.md
- E-006 2026-09-21 Pure-consume ablation names the wall: coalesced 327 GB/s;
  same-geometry loads 245 (pattern NOT the wall); +constant-LUT 178 (-28%);
  full decode 105 (-41% further). Door: decode-atom redesign, target
  130-150 GB/s, bit-exact integer atom preferred.
- E-007 2026-09-21 Incident + law: stale per-tensor byte figure (28.2 MB vs
  real 38.3 MB) -> OOB bench reads -> memory fault -> numbers void. LAW:
  per-tensor sizes come from offset deltas (next_offset - offset), never
  recomputed formulas. GGUF layout verified standard (866/866 tensors).
- E-008 2026-09-21 Protocol adopted (user-directed): agents work in git
  worktrees on side branches; this ledger is the append-only current state;
  timer automation (every 30 min) drives progress without user input.
- E-009 2026-09-21 Worktree + agent split (user-directed protocol): worktree
  /media/chris/ssd128/llamacpp/wt-decode-atom on branch amd/w2-decode-atom
  dispatched to the W2 decode-atom desk (oracle-first mandate, target
  >=130 GB/s on the iq3_s cell, die 3 only). Gemini retains dies 0-2 for the
  served guard battery and its hardening tasks. Timer automation (30 min)
  now carries this protocol.
- E-010 2026-09-21 Hardened battery + cold-stamp baseline re-run validated GREEN:
  version-fingerprint ab3dfba685c5cfb7, session lock enforced, 5s thermal sideband
  (109 samples, drift +47.0C, max edge 85C/junc 90C). 180s idle cooldown executed;
  all 5 guards PASS: cold-stamped pp 93.87 t/s (-2.72% vs 96.50), decode 15.34 t/s
  (-1.47% vs 15.57), MTP accept 0.6667 (gate >=0.63), greedy determinism byte-identical
  (sha256 4beb1ba25219ee9b), needle recall 3/3 exact ([25, 50, 75]%). W0/W1 test harness
  setup officially closed. Receipt: results/tp3_guards_20260921_141900.jsonl,
  thermal log: results/tp3_guards_20260921_141900_thermal.log

- E-010 2026-09-21 (recorded by ZCode from Gemini msg #1316; Gemini's own
  copy did not land in-tree) Battery hardening + cold-stamp baseline re-run
  COMPLETE: 5/5 guards GREEN - cold prefill 93.87, decode 15.34, accept
  0.6667, determinism byte-identical, needle 3/3. Thermal sideband banked
  (109 samples). Battery fingerprint ab3dfba685c5. Receipt:
  results/tp3_guards_20260921_141900.jsonl (+ _thermal.log). This is the
  official baseline of record for all future arms. W0/W1 setup CLOSED.
- E-011 2026-09-21 Second worktree dispatched: wt-prefill-gemm on branch
  amd/w5-prefill-gemm for the W5 prefill desk (rocBLAS shape survey +
  GEMM microbench harness authoring, NO GPU runs while the W2 decode-atom
  desk holds die 3; benching deferred until W2 integration is recorded
  in this ledger). Decode-atom agent still running (no report yet).
- E-012 2026-09-21 W5 zero-GPU phase COMPLETE (desk: wt-prefill-gemm). Shape census
  banked from the GGUF header + qwen35.cpp graph: 48 GDN + 16 attn layers; attn_q
  5120x12288 carries q+gate fused; TP3 row-split rounds per-die shares to 128-row
  blocks (get_mmq_y_host gfx900), canonical shares e.g. 17408 -> 5760/5760/5888. Trunk
  = 6.23 TFLOP/ubatch (48.7 GFLOP/tok, ~2.04 TFLOP/die); FFN gate/up/down = 70.3% of
  GEMM FLOPs, then gdn_qkv 10.3%, ssm_out+gdn_gate 6.2% each. Dispatch survey: every
  dense prefill GEMM runs dequant-to-f16 + hipBLAS f16-acc/f16-out (gfx900 MMQ
  disabled dense by rule mmq.cu:371-376; FORCE_MMQ/CUBLAS are compile-time only);
  implied served rate ~2.0-2.2 TF/s/die vs 10.75-12.5 fp32-class / ~21.5 packed-fp16
  ceilings, so the custom packed-fp16 tile route stays W5-live pending the bench gate.
  Harness READY, compile-validated for gfx900, NOT RUN: die-3 runs deferred until the
  W2 desk releases the cell. Receipt: results/W5_gemm_survey_2026-09-21.md;
  harness: tests/bench_gemm_gfx900.cu (census table + build line in header).
- E-013 2026-09-21 Integration: amd/w5-prefill-gemm merged into
  amd/v340-port-v2 (merge resolved at the ledger tail; duplicate E-010
  entries kept - same fact, recorded independently, both stand under
  append-only). W5 zero-GPU phase ACCEPTED: shape census (FFN trio = 70.3%
  of GEMM FLOPs, 6.23 TFLOP/ubatch), rocBLAS harness ready
  (tests/bench_gemm_gfx900.cu, compile-validated, deferred die-3 run),
  survey finding of record: dense prefill GEMMs run dequant-to-f16 +
  hipBLAS f16-acc/f16-out every call (mmq.cu:371-376 rule) - the f16
  dequant write traffic (~16 GB/die/ubatch) is a newly-named prefill
  target. GPU phase queued behind W2 decode-atom desk.
- E-009 2026-09-21 W2 decode-atom cell goes oracle-gated (bench v2): host C++
  mirror of vec_dot_iq3_s_q8_1 gates every arm (rel err < 1e-4, 64 rows). Three
  traps caught and guarded: __device__-only table symbols (host silently read
  garbage; pull via kernel D2H + pinned values), HIP __half host fields
  VALUE-convert on assignment (ds.x fill became fp16(15360); q8_1 now built as
  raw bytes), iq3s_grid = 512 entries (9th index bit from qh). Two v1 clone
  bugs found vs mmvq.cu/vecdotq.cuh and fixed: u1 index l0+4 -> l0+1, y
  pointer never offset by kby=kbx*8. results/W2_decode_atom_2026-09-21.md
- E-010 2026-09-21 CORRECTION (supersedes E-005/E-006 absolutes): fixed-clone
  control = 90.4-90.9 GB/s (422-425 us), stable across 3 sessions; the v1
  105.2 GB/s figure was an accidentally lighter kernel - its missing kby
  offset made all 8 x loads loop-invariant (compiler hoisted them out of the
  kbx loop). Same-era A/B (old binary from git): 360.0 us/106.6 vs fixed
  424.6/90.4. Perf-neutral claim FALSIFIED; old ratios stand only within their
  own sessions. consume-ablation ladder unchanged structurally: loads-only
  245, +LUT 178, full decode ~91 (corrected).
- E-011 2026-09-21 W2 decode-atom ladder ALL NEGATIVE (all oracle-passing,
  bit-exact, no numerics change, control drift < 1 percent): gmem (grid global
  vs constant) 1.00x NEUTRAL; pip (reorder + 2 accumulators) 1.04x; sgn (32 KB
  sign-folded LUT, kills whole vcmpne4/vsub4 chain) 2.20x - 32 KB > 16 KB L1,
  L2-resident divergent lookups; partial folds net zero by instruction budget;
  sexp (2 KB pre-expanded sign masks) 1.16x - 4 extra divergent lookups cost
  more than ~30 vops saved; hand dp4a (v_bfe_i32+v_mad_i32_i24) 1.08x
  volatile / 1.21x non-volatile - library 16-bit SDWA emulation wins. Static
  census: base loop 314 vops/call (dp4a ~130, sign ~35, index/addr ~90).
  results/W2_decode_atom_2026-09-21.md
- E-012 2026-09-21 W2 decode-atom VERDICT (stop condition reached): the
  shipped iq3_s atom is at its practical gfx900 ceiling at ~91 GB/s; the
  130-150 GB/s target is ISA-walled - no v_dot4 on gfx900 (dp4a emu
  irreducible for bit-exact math), fp16 V_DOT2 path blocked by x int8->half
  conversion cost (and numerics), 8 divergent 9-bit LUT reads per 32 weights
  irreducible for this format. Next levers named OUTSIDE the atom: T4
  quantize_q8_1 elimination first (also the door to an fp16x2 activation
  format for a dot2 atom), T3 tiny-tensor tax second. Handoff to W2 owner.
- E-014 2026-09-21 Integration: amd/w2-decode-atom merged (desk-local ledger
  entries kept verbatim alongside main-branch entries - some E-numbers now
  appear twice from parallel desks; authoritative sequence continues E-015+).
  W2 VERDICT OF RECORD: shipped iq3_s atom at its practical gfx900 ceiling -
  8 arms all negative/neutral (oracle-gated, 3 reps x 200 iters, bit-exact);
  binding terms: 8 divergent 9-bit LUT reads / 32 weights, ~130 vops dp4a
  emulation (no v_dot4 on gfx900), sign-fold dead (32 KB > 16 KB L1). TRUE
  shipped-atom figure 90.4-90.9 GB/s - v1 bench's missing kby offset hoisted
  x loads out of the loop, so the banked 105.2 was an accidentally lighter
  kernel (correction addendum on the cell receipt). 130 target ISA-walled,
  not forced. NEXT LINK (desk ruling, adopted): T4 kill quantize_q8_1 BEFORE
  T3 tiny-tensor tax - the fp16x2 activation producer is also the only door
  to a cheap exact V_DOT2_F32_F16 atom. Die-3 window now OPEN for the W5
  gemm harness run.
- E-015 2026-09-21 W5 GEMM harness RUN (die 3): shipped-route rocBLAS fp16
  (h16 mode) = 4.62 TF/s/die call-weighted (f32-out mode 3.04 - h16 stays).
  37% of fp32-class / 21% of packed-fp16 ceiling -> W5 GATE MISSED: custom
  packed-fp16 tile route OPENS. Small shapes worst (attn k/v ~1.6-2.1 TF/s).
  Custom tile must beat 4.62 TF/s/die to promote. Full output:
  results/W5_gemm_run_2026-09-21.txt; harness tests/bench_gemm_gfx900.cu.
- E-016 2026-09-21 Two parallel desks dispatched (user granted 2-agent
  assist): wt-tile-fp16 on amd/w5-tile-fp16 (W5 custom packed-fp16 GEMM
  tile, die 3 PRIMARY - promotion bar 4.62 TF/s/die) and wt-t4-producer on
  amd/t4-producer (T4 per-layer q8_1 activation-quant cache + fp16x2/V_DOT2
  producer design; ZERO die-3 access until the tile desk completes - GPU
  arbitration: tile desk owns die 3 first).
- E-016 2026-09-21 T4 producer desk dispatched (worktree wt-t4-producer,
  branch amd/t4-producer; ZERO-GPU phase while die 3 is busy). Target: kill
  the redundant activation re-quantization in decode (quantize_q8_1 =
  7.1% of kernel budget, ~1400 launches/step; 8+ vec-q matmuls per layer
  re-quantize the same 2-3 unique x). DESIGN BANKED: per-ubatch q8_1
  activation cache in ggml_backend_cuda_context, env-gated
  GGML_CUDA_Q81_ACT_CACHE=1 (static-once getenv). Key = (producing tensor
  NODE, src1 data ptr, ne10-13, strides, device, stream): the node field is
  the load-bearing safety choice - ggml-alloc free-block reuse can hand a
  freed intermediate's address to a later same-shape tensor WITHIN one
  compute, so a data-pointer-only key can false-hit; a node pointer cannot.
  Entries are freed at every graph_compute entry, so nothing spans two
  computes or outlives the compute buffer; cache is force-inactive during
  CUDA graph capture/replay (captured quantizes must stay in-graph, replays
  skip the host side). Vec-q path only; mul_mat_id excluded (ids!=nullptr
  and op==NONE scratch slices). Bit-exact by construction: same kernel +
  same key params + unchanged x = identical bytes (kernel is deterministic,
  full padded rows written).
- E-017 2026-09-21 T4 implementation banked (zero-GPU): ggml-cuda/common.cuh
  ggml_cuda_q81_act_cache (64 MiB cap, hit/miss counters, INFO line every
  256 computes) + hooks at both quantize sites (ggml_cuda_op_mul_mat split
  path = the TP3 route, ggml_cuda_mul_mat_vec_q non-split) + begin_compute
  gate in ggml_backend_cuda_graph_compute. Env unset = zero behavior
  change (find/insert early-return, no counters). Compile-validated:
  ggml-cuda.cu and mmvq.cu for gfx900 (rocm 6.2 clang, build-mirrored
  flags) both clean. Host unit test
  docs/amd-port/tests/test_q81_cache_host.cpp ALL PASS: hit bytes ==
  fresh-quantize bytes (byte-exact quantize mirror incl. warp-reduce
  butterfly), node-recycle must-miss, shape/stride/stream/device
  sensitivity, per-compute invalidation frees 100% of pool memory, byte
  cap stops inserts without eviction, nbytes() == both upstream size
  formulas. Receipt: results/T4_producer_2026-09-21.md. DEFERRED to the
  die-3 window: served greedy determinism byte-identical vs cache-off arm,
  rocprof quantize_q8_1 count ~1400 -> ~450-500/step, decode t/s A/B.
- E-018 2026-09-21 Integration: amd/t4-producer merged (desk-local ledger
  numbering kept verbatim; parallel E-numbers stand per append-only). T4
  zero-GPU phase ACCEPTED: env-gated per-ubatch q8_1 activation cache
  (GGML_CUDA_Q81_ACT_CACHE=1; unset = zero change), node-pointer cache key,
  per-compute invalidation, CUDA-graph-capture-safe, hooks on the TP3 split
  path + non-split mmvq; compiles clean for gfx900; host unit tests ALL
  PASS (byte-exact reuse, must-miss on node recycle, cap, invalidation).
  Saving bound: ~63-65% of quantize_q8_1 launches removed ~= 4.5% of decode
  kernel time. Served validation (dies 0-2, Gemini lane) protocol banked in
  results/T4_producer_2026-09-21.md: guards-off GREEN -> cache-ON byte-
  identical determinism + accept >= 0.63 + needle 3/3 -> census launch
  count check -> 3-rep OFF/ON A/B, promote at >= +2% decode.
- E-019 2026-09-21 Two more desks dispatched (user directive, 2-agent
  assist): wt-t3-grouped on amd/t3-grouped (T3 tiny-tensor tax: grouped
  GEMV for the per-layer small tensors sharing one x; zero-GPU impl phase,
  die-3 bench queued behind the tile desk) and wt-draft-dev on
  amd/w6-draft-device (W6: dedicated draft-device support - run the MTP
  draft context on the 4th die overlapping the TP3 verify pass; code +
  compile-validated implementation, served validation coordinated with
  Gemini). GPU arbitration: die 3 = tile desk -> T3 bench -> W6 validation
  windows (with Gemini).
- E-020 2026-09-21 W1 ubatch sweep completed (interleaved 4-boot A/B, ub256 vs ub512 vs ub128 control, 8 reps total, dies 0-2):
  ub256 wins on decode (15.51 t/s mean cold vs ub512 14.94 t/s, +3.8% higher decode throughput than ub512, +1.1% vs control)
  with large prefill gains (107.52 t/s mean cold, +14.5% vs ctrl 93.87 t/s). ub512 wins pure prefill (114.06 t/s mean cold,
  +21.5% vs ctrl) but regresses decode (-2.6% vs control, -3.8% vs ub256). MTP acceptance held at 0.6667 (gate >=0.63),
  greedy determinism byte-identical (PASS 8/8), needle recall 3/3 (PASS 8/8). Recommendation: promote ub256 to launch
  script of record for balanced decode/prefill optimization. Receipt: results/W1_ubatch_sweep_20260921_160303.md.

- E-021 2026-09-21 W1 ubatch sweep RULING (receipt
  results/W1_ubatch_sweep_20260921_160303.md): ub512 PROMOTED to the
  launch script of record - prefill +20.6-22.4% (113.0-114.9 vs 93.87)
  at decode -2.2 to -4.5% (14.87-15.00); acceptance 0.6667, determinism
  PASS, needle 3/3 in all 8 arms. For 200k-context workloads the prefill
  gain dominates the small decode give-back; rollback = revert the script
  line. Baseline gates to be re-stamped on the promoted config via
  --ratchet (prefill ~113.5, decode ~14.9 expected). DISCOVERY: server
  warns "backend sampling not supported with SPLIT_MODE_TENSOR; using
  CPU" - GPU-side draft sampling never engages under tensor split; logged
  as a future server-side fix candidate, closes the W1 backend-sampling
  item as not-applicable to this split mode.
- E-016 2026-09-21 W5 custom-tile cell RUN (die 3, oracle-gated): packed-fp16 GEMM
  tile (v_pk_fma_f16 packed along k-pairs, weights resident f16 in HBM, M=128 tiles,
  interleaved column ownership, split-K + fixed-order f32 reduce in the timed region)
  BEATS THE W5 GATE: best arm a8 (128x64, TY4 TX8 KC32, auto split-K) = 5.53 TF/s/die
  all-die call-weighted vs the 4.62 bar (+19.7%) and vs same-session rb16 control
  4.55 (+21.5%); control reproduces banked E-015 (4.52/4.62) within ~1.5% drift.
  Per-shape (die0): ffn_down +55% (6.15), ssm_out +67% (4.98), attn_out +57%,
  attn_q+gate +48% (5.36), gdn_gate +24%, gdn_qkv +19% (5.18), attn_k/v +88-97%
  (~3.0), ffn_gate/up parity 0.99-1.00x (5.60 vs 5.67). ORACLE: host mirror of the
  exact per-output accumulation (chunk-local flush schedule, per-slice f16 state,
  fixed-order f32 slice sum) with double-precision fused-half emulation = bit-exact
  vs v_pk_fma_f16; 90/90 arm-shape gates PASS mismatch 0.000%; tile L2-vs-f64 err
  2.3-5.4e-3 = 10-19x better than shipped h16 (K/ks f16 slices + f32 sums beat
  full-K f16 accumulation). Two design bugs caught by the oracle before any timing
  counted: (a) size_t 64-bit index math compiled to v_mad_u64_u32 chains, ~2x the
  VALU stream - 32-bit indexing is the single biggest arm win; (b) MT64 arm A-stage
  missed the M-tile row offset (50.8% outputs wrong). Arm ladder (all-die agg):
  a8 5.53 > a3 5.20 > a4 4.88 > a2 4.85 > a1 4.85 > a6 4.44 > a5 4.23 > a9 4.21 >
  a7 4.14; register-prefetch pipeline arm (ldg->reg before compute, stlds after,
  1 barrier/chunk) REJECTED: VGPR 93->246 (a8), 155->256 (a6), all arms regress
  (a6 4.25->1.34) - occupancy loss outweighs hidden HBM latency. Census
  (hipFuncGetAttributes + --save-temps): a8 VGPR 93, LDS 25 KB, 2 waves/SIMD,
  512 v_pk_fma_f16 + 975 other VALU per chunk loop, zero spills. Ceiling named:
  W-frag/A-frag LDS service 48 cyc per 128 VALU-cyc per wave-kpair = 1.5x
  oversubscribed on the per-CU LDS at 4 waves -> ~14 TF/s model cap; measured 5.6
  on ffn_gate = staging exposure (lockstep barriers, HBM latency per chunk) + split-K
  reduce dominate the gap. Next levers: 3-4-deep LDS buffering or W-frag direct from
  L2 (without the VGPR blowup), cheaper W-frag, epilogue atomics to drop the reduce.
  PROMOTION RECOMMENDED: runtime-env-gated dispatch at the ggml-cuda.cu:2629 cublas
  fallback (E-012 option (a), static-once getenv pattern at 1600-1626), a8 geometry
  for all census shapes, weights kept as load-time f16 dequant output (steady state)
  - the per-call dequant tax (~16 GB/die/ubatch f16 writes) is deleted on top of the
  +21% GEMM. Receipt: results/W5_tile_2026-09-21.md; raw:
  results/W5_tile_full_2026-09-21_v1.txt; harness: tests/bench_tile_gfx900.cu.
- E-022 2026-09-21 Integration: amd/w5-tile-fp16 merged. W5 TILE VERDICT OF
  RECORD: PROMOTE-bar MET - arm a8 (128x64, TY4 TX8 KC32, split-K + fixed-
  order f32 reduce, weights resident f16) = 5.53 TF/s/die call-weighted vs
  4.62 gate (+19.7%), +21.5% vs same-session rb16 control; wins on every
  shape class except ffn_gate/up parity; biggest: ssm_out +67%, attn_out
  +57%, ffn_down +55%, attn_q+gate +48%, attn k/v ~1.9x. Oracle 90/90
  bit-exact (f64 fused-half emulation of v_pk_fma_f16). Also deletes the
  per-call dequant tax (~16 GB/die/ubatch f16 writes) and is 10-19x more
  accurate than shipped h16. Named negative: register-prefetch pipeline
  (VGPR 93->246, all regress). Named ceiling: LDS W/A-frag service ~1.5x
  oversubscribed -> ~14 TF/s model cap; staging exposure at lockstep
  barriers dominates the gap. NEXT: in-ggml env-gated dispatch at the
  ggml-cuda.cu:2629 cublas fallback (a8 geometry, load-time f16 weights),
  then 3-4-deep LDS buffering + cheaper W-frag toward 6-8 TF/s.
- E-019 2026-09-21 W6 draft-device desk dispatched + design banked (worktree
  wt-draft-dev, branch amd/w6-draft-device; ZERO-GPU, no boots by the desk).
  Device-inheritance audit of the MTP draft context (ctx_dft = second
  llama_context on the shared target model) names three independent
  inheritance points, all now pinnable: (1) context backends from
  model.devices (meta over dies 0-2 under -sm tensor), (2) weight bufts from
  the loader's meta buft lists, (3) MTP KV from model.dev_layer(il) (meta
  AXIS_0 shard today). Load-bearing facts: the GGUF carries
  blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm} but NOT
  shared_head_head/embed_tokens, so the draft graph falls back to the shared
  model.output (517.8 MiB Q6_K, TP3 row-sharded) and model.tok_embd
  (335.3 MiB IQ4_XS, mirrored) - both MUST stay on the TP group for the
  target verify pass, only blk.64 itself (169.3 MiB, 15 tensors) is
  draft-exclusive. The sched GGML_ABORTs on pre-allocated weights outside
  the backend list (ggml-backend.cpp), so a naive [die3, CPU] draft backend
  list crashes on the first head matmul: placement must be exact at load
  time. Die-3 budget ~0.5-0.7 GiB of 8 GiB (weights 169 + KV ~220 at 200k
  q4_0 + compute/output buffers).
- E-020 2026-09-21 W6 implementation banked (zero-GPU): --spec-mtp-device
  <DEV> (name or GPU index; env LLAMA_ARG_SPEC_MTP_DEVICE) + three pins:
  llama_model_params.dev_mtp (weights: loader buft rule bid >= n_layer in
  llama-model-loader.cpp buft_for_tensor; KV: llama_model::dev_layer override
  for il >= n_layer - target unaffected, its cache filters the MTP layer out)
  and llama_context_params.extra_device (draft ctx backends [meta, die3,
  CPU]; server sets cparams_mtp.extra_device). Head stays on meta (keeps 3x
  bandwidth); two ~20 KiB meta<->die3 activation hops per draft step are the
  added PCIe cost; the h handoff crosses PCIe per token exactly as before,
  relocated not added. Guarded: flag unset = zero behavior change;
  dev_mtp-in-target-devices refused (arg + server, order-proof);
  --device CUDA0,CUDA1,CUDA2 mandatory in 4-visible boots or the TENSOR
  split builds TP4. Compile-validated gfx900 (8 touched TUs, exact
  build-hip flags, zero warnings); host logic tests ALL PASS
  (docs/amd-port/tests/test_w6_mtp_device_host.cpp). NOT RUN: served A/B
  (alternating boots, 3 reps, guard battery, determinism within-arm only -
  not vs control) queued for a negotiated window (dies 0-2 serve, die 3
  drafts); honest overlap estimate and full protocol in
  results/W6_draft_device_2026-09-21.md (v1 single-stream win 0-15%,
  expected ~+5-10%, strict-dataflow overlap is zero until the v2
  draft-ahead scheduler lands).
- E-023 2026-09-21 Integration: amd/w6-draft-device merged. W6 zero-GPU
  phase ACCEPTED: --spec-mtp-device <DEV> pins the MTP draft head (weights,
  KV, context backends) to die 3 via three load-time placements; shared
  output/tok_embd stay on TP3 (GGUF has no draft-exclusive head); die-3
  budget ~0.5-0.7 GiB. Compile-clean 8 TUs, host tests PASS. HONEST ESTIMATE:
  v1 = TP-tax removal (AR chain + triple-issued draft kernels), -2% to +15%,
  expected +5-10% (serial dataflow: no true overlap until later desks).
  Validation protocol in results/W6_draft_device_2026-09-21.md - Gemini lane,
  alternating-boot A/B, promote at >= +2% decode with guards green; within-arm
  determinism only (reduction order changes across devices).
- E-024 2026-09-21 Tile-integration desk dispatched: wt-tile-integ on
  amd/tile-integ - wire the a8 tile into the ggml-cuda dispatch behind
  GGML_CUDA_TILE_FP16=1 at the cublas fallback branch point; die 3 open
  for its validation runs. Gemini ACKed E-021 ruling (msg #1325); ubatch
  re-stamp -> T4 validation -> W6 A/B queue stands.
- E-019 2026-09-21 T3 grouped-GEMV implementation banked (zero-GPU, desk:
  wt-t3-grouped, branch amd/t3-grouped; die 3 held by the tile desk). Env
  GGML_CUDA_MMVQ_GROUP=1 (static-once getenv, unset = zero change): one
  launch computes a batch of 2-8 small quantized vec-q MUL_MATs sharing the
  same src1 (byte-identical x). Grouping point = graph node loop (try_fuse
  precedent): same-src1-pointer members within a 16-node window, members
  excluded if any try_fuse MUL_MAT pattern would consume/span them, early-
  dst-write liveness enforced (intermediate WAW/WAR vs candidate dst, src1
  mutation, pairwise member-dst overlap - the T4 node-pointer lesson applied
  to allocator range recycling). Kernel = mmvq_group_row<type>: verbatim solo
  mul_mat_vec_q schedule per member (same kbx partition, reduce order,
  vec_dot), mixed types per launch, geometry gate (uniform nwarps, 1
  row/block, no small_k; gfx900 GCN = nwarps 2 all types) aborts to solo on
  any mismatch; per-device row whitelist <= 256 (env-tunable MAX_ROWS).
  Split-path runner mirrors ggml_cuda_op_mul_mat T=1 verbatim (row split,
  events, peer copies; one shared q8_1 per device per group - composes with
  T4 cache, same key). By-value member params = capture/replay-safe. Host
  oracle docs/amd-port/tests/test_mmvq_group_host.cpp ALL PASS: 144 rows
  float-BIT-identical solo-vs-grouped over census-shaped mixed batch
  (q4_K/q8_0), f64 references honest, geometry + small_k gates verified.
  Compile-clean gfx900 (ggml-cuda.cu, mmvq.cu). Saving bound: GDN
  beta/alpha pairs 96 -> 48 launches/step (2x on the pair class), latency
  overlap ~halves its floor time; ~3-4% of decode kernel time expected,
  census optimistic bound 12-15%. Receipt: results/T3_grouped_2026-09-21.md.
  DEFERRED to die-3 window: guards byte-identical determinism gate, census
  launch-count check, 3-rep OFF/ON A/B (promote >= +2%), MAX_ROWS=384 sweep
  for attn k/v pairs, occupancy check.
- E-025 2026-09-21 Integration: amd/t3-grouped merged. T3 zero-GPU phase
  ACCEPTED: GGML_CUDA_MMVQ_GROUP=1 graph-level grouping of same-src1 vec-q
  MUL_MAT nodes (16-node scan, T4 node-key + dst-overlap recycling checks,
  try_fuse span exclusion), grouped kernel bit-identical by construction
  (verbatim solo schedule per member, mixed types per launch, by-value
  params = replay-safe), host oracle 144 rows bit-identical, gfx900 compile
  clean. Saving bound: pair class 96 -> 48 launches/step, ~3-4% of decode
  kernel time expected, +2-5% t/s if fully realized. Served validation
  queued on the Gemini lane AFTER T4 (same protocol class: env-unset GREEN
  -> byte-identical determinism gate -> census launch counts -> 3-rep
  OFF/ON A/B, promote >= +2%; MAX_ROWS=384 sweep for attn k/v pairs).
- E-026 2026-09-21 PRIORITY CORRECTION (Chris): activated MTP typically
  costs OVER 2 GB PER DEVICE - the TP2 feasibility desk's hypothesis is
  updated accordingly: the analytic ~202 MiB/die figure (weights+KV slice)
  is a LOWER BOUND; the true cost likely lives in the draft context's
  compute/logits buffers and pool allocations, ALL of which
  --spec-mtp-device should relocate. Desk instructed to measure the true
  per-device MTP VRAM cost first (MTP off vs on), then the flag's savings.
  If >2 GB/device confirms, the serving-die savings at TP2/200k are
  ~10x the historical 151 MiB shortfall - TP2+MTP@200k becomes
  comfortably viable, and TP3 gains ~2 GB/die of context headroom.
- E-027 2026-09-21 Promoted ub512 baseline re-stamp + T4 producer correctness gate banked (Gemini lane, dies 0-2):
  (1) Re-stamped baseline on promoted ub512 config via cold battery run with --ratchet
  (receipt: results/tp3_guards_20260921_171802.jsonl). 5/5 GREEN: prefill 115.21 t/s (ratcheted),
  decode 14.93 t/s, MTP accept 0.6667, determinism PASS, needle 3/3. tests/baseline_tp3_200k.json updated.
  (2) T4 Step 1 (env unset): 5/5 GREEN (receipt: results/tp3_guards_20260921_170742.jsonl).
  (3) T4 Step 2 (cache-ON gate, GGML_CUDA_Q81_ACT_CACHE=1): 5/5 GREEN (receipt:
  results/tp3_guards_20260921_172641.jsonl). Greedy determinism is BYTE-IDENTICAL to cache-off
  (sha256: 4beb1ba25219ee9b), MTP canary 0.6667, needle 3/3.
  (4) Teardown assertion noted: GGML_ASSERT(pool_size == 0) triggers at server exit due to
  un-freed q81_act_cache pool buffers (clean fix: call clear() in cache destructor).
  (5) Resource arbitration: window granted on dies 0-2 for TP2-feasibility desk MTP VRAM profiling.
  T4 Step 3 (3-rep perf A/B) paused until TP2 desk completes.

- E-027 2026-09-21 T4 SERVED GATE PASSED (Gemini, dies 0-2): cache-ON
  greedy determinism BYTE-IDENTICAL to cache-off (sha256 4beb1ba25219ee9b),
  accept 0.6667, needle 3/3, 5/5 guards on rebuilt a62fc0f5a. Baseline
  re-stamped on promoted ub512: prefill ratcheted to 115.21 t/s, decode
  14.93 (receipt tp3_guards_20260921_171802.jsonl). OPEN BUG: teardown
  GGML_ASSERT(pool_size == 0) - q81 cache buffers not returned at context
  free; fix assigned to the T4 desk (gates promotion, not the running
  OFF/ON perf A/B). Remaining for T4 promotion: 3-rep OFF/ON A/B >= +2%.
- E-018 2026-09-21 T4 CORRECTION (served gate PASSED, teardown bug fixed):
  determinism gate GREEN on the merged rebuilt branch - cache-ON greedy
  output sha256 4beb1ba25219ee9b identical to cache-off, accept 0.6667,
  needle 3/3, 5/5 guards. BUG: on clean server shutdown
  GGML_ASSERT(pool_size == 0) fired - q81_act_cache buffers are raw
  pool->alloc's, freed only at the next begin_compute, so at context
  teardown the pools still had outstanding allocations when their
  destructors ran the accounting. FIX: ggml_backend_cuda_context dtor now
  calls q81_act_cache.clear() first - pools are context members destroyed
  after the dtor body, so every entry returns to its live pool; unset-env
  path untouched (empty map, no-op). Fix gates PROMOTION only; Gemini
  proceeds with the OFF/ON 3-rep perf A/B on the pre-fix binary.
- E-028 2026-09-21 EMPIRICAL NEGATIVE: TP2+MTP at 200k does NOT boot even
  with --spec-mtp-device (desk boot attempt died at allocation time; empty
  logs = died before serving). Historical refusal class confirmed stronger
  than the draft-block offload - TP2 at 200k is closed as a topology unless
  a future KV/weights change lands. Desk re-scoped: measure the true
  per-device MTP VRAM cost (>2 GB hypothesis) on TP3, which boots, plus
  --spec-mtp-device relocation savings there; one small-context TP2+device
  boot for flag mechanics.
- E-024 2026-09-21 W5 tile-integ desk COMPLETE (worktree wt-tile-integ, branch
  amd/tile-integ; die 3 only, ~11 min of windows; dies 0-2 untouched). The a8
  tile is WIRED INTO ggml behind GGML_CUDA_TILE_FP16=1: branch at the top of
  ggml_cuda_op_mul_mat_cublas (the dense fallback op), kernel ported verbatim
  from bench_tile_gfx900.cu (one change: dst row stride ldc decoupled from N
  for the strided main-device slice), static-once env, census whitelist
  (gfx900, quantized src0, src1 f32, PREC_DEFAULT, M<=512 %128, K in
  {5120,6144,17408}, N_d %64, 32-bit/align guards) with one-time WARN +
  shipped fallback off-list. FINDING: use_fp16 requires row_diff ==
  src0->ne[1] (ggml-cuda.cu:1660) so under TP3 split the shipped route is
  dequant-to-f32 + cublasSgemm (NOT the h16 route E-012 named - that holds
  only single-GPU); the branch therefore sits before the whole bf16/h16/f32
  chain and the tile also deletes the 2x-byte f32 dequant tax in-server.
  VALIDATION (die 3, integrated build, compile-clean gfx900 zero warnings):
  (1) oracle on the PORTED kernel 10/10 bit-exact vs spec (max rel 0.00e+00),
  L2-vs-f64 2.9e-3..6.1e-3 inside the banked band; (2) env-unset byte-identity
  proven EMPIRICALLY: same bench linked against the pre-change served
  build-hip lib, f32 dst dumps cmp 10/10 byte-identical; (3) integrated
  per-shape GEMM-class TF/s (rocprof kernel-only, 3x20 median): call-weighted
  die0 agg 6.10 TF/s/die vs same-session shipped h16 4.79 (+27.5%) - the 5.53
  banked class REPRODUCED AND EXCEEDED, no kernel-level integration cost;
  per-shape vs shipped: ffn_down 1.75x, attn k/v 1.86x, ssm_out 1.57x,
  attn_out 1.51x, gdn_qkv 1.50x, attn_q+gate 1.31x, gdn_gate 1.20x,
  ffn_gate/up 0.94-0.95x (parity class, control ran hot this session);
  (4) wall op-level (v1 still pays per-call dequant + src1 convert): 3.52 ->
  4.33 TF/s/die (+23.0%); (5) M=256/512 route and hold 5.0-5.6 wall class,
  M=640 off-list -> one WARN + shipped. HONEST COUPLINGS: v1 GEMM reads f16
  weights just written by the retained dequant (L2 credit, attn k/v most) -
  5.53 stays the post-loader reference; numerics are the arm's own class
  (L2 vs shipped arm 1.1-2.0e-2, tile 10-19x more accurate vs f64). DEFERRED
  to the Gemini lane (dies 0-2): rebuild served binary, guards GREEN unset,
  ON-arm greedy determinism WITHIN arm + accept >= 0.63 + needle 3/3 (NOT
  byte-identity vs OFF - arm is numerics-changing by contract), TP3 trace
  check (tile replaces SGEMM under split; no off-whitelist tile at ub512),
  3-rep OFF/ON A/B; then the loader step deletes the retained dequant.
  Receipt: results/W5_tile_integ_2026-09-21.md; traces:
  results/W5_tile_integ_trace_{on,off}_2026-09-21.csv; harness:
  tests/bench_tile_integ.cpp + tests/test_tile_integ_oracle.cu.
- E-029 2026-09-21 Integration: amd/tile-integ merged (bd825a43e lineage).
  a8 TILE IS IN-GGML behind GGML_CUDA_TILE_FP16=1: dispatch branch at the
  top of ggml_cuda_op_mul_mat_cublas, census whitelist + fail-loud shipped
  fallback, env-unset byte-identity proven empirically (f32 dst dumps cmp
  10/10 vs pre-change lib). DISPATCH DISCOVERY OF RECORD: under TP3 split
  use_fp16 is impossible (row_diff != ne[1], ggml-cuda.cu:1660) -> the
  served shipped route was dequant-to-F32 + cublasSgemm all along; the
  tile deletes the 2x-byte f32 dequant write tax under split. Integrated
  per-shape (die 3, M=128): call-weighted 6.10 TF/s/die vs shipped 4.79
  same-session (+27.5%), zero kernel integration cost; ffn_down 1.75x,
  attn k/v 1.86x, gdn_qkv 1.50x; ffn gate/up parity-class (0.95x).
  Numerics-changing by contract (tile 10-19x more accurate vs f64, but
  different class from shipped h16/f32). Served validation queued on
  Gemini lane (rebuild, within-arm determinism, census trace, OFF/ON
  A/B); loader desk (delete retained per-call dequant) queued after.
- E-030 2026-09-21 TP2 feasibility desk COMPLETE (worktree wt-tp2-mtp, branch
  amd/tp2-mtp-feasibility, base 597fcaee5; dies 0-2 window, Gemini ACK #1328;
  numbering note: shared tree already at E-029 when written, E-026..E-029 live
  there). THREE-WAY MTP MATRIX at TP2/-c 10000 (b512/ub512, q4_0 KV, FA, 2 reps
  per arm, receipt results/TP2_feasibility_2026-09-21.md): TRUE per-device MTP
  cost in-split = 1072.8 MiB/die over MTP-OFF at boot-ready (7599.6 vs 6526.8,
  byte-identical across reps), of which draft weights+KV are only ~90 MiB - the
  cost is DRAFT-CONTEXT BUFFER dominated (~983 MiB/die on the TP meta group),
  confirming the coordinator's buffer-dominance instinct and refuting the
  202 MiB weights+KV analytic figure as a ~5x undercount (the >2 GB/device
  hypothesis does NOT reproduce at TP2/10k either). --spec-mtp-device ROCm2
  relocates only 251.1 MiB/die off the serving dies (draft-context backend list
  [meta, die2, CPU] keeps most draft compute on meta; die 2 holds 558.0 MiB at
  boot, 1437.1 after probes). Decode: OFF 10.54/11.39, in-split 14.79/14.81,
  flag 14.43/14.43 (flag = -2.4% vs in-split, MTP = +38-40% vs OFF); pp
  87.5-89.2 all arms; accept 0.66667 / 3.00 tok-per-step everywhere. Consequence:
  in-split TP2+MTP post-probe headroom is 67 MiB/die at 10k (ctx ceiling
  ~17-18k); the flag's 231 MiB/die + draft-KV moved off-die extends the ceiling
  to ~36k at the -2.4% decode price. Flag stays opt-in; promotion only pays for
  TP2 mid-context classes where MTP would otherwise not fit.
- E-031 2026-09-21 TP2@200k failure NUMBERS (supersedes the "empty logs"
  assumption in E-028; same desk as E-030): the MTP-OFF TP2/200k boot aborts at
  target-context graph_reserve, not at first request - "allocating 1057.78 MiB
  on device 0: cudaMalloc failed: out of memory" (ggml-backend-meta.cpp:1512
  GGML_ASSERT), sampler peak 7956.6/7956.3 MiB on dies 0/1 = ~219 MiB free at
  the 1057.78 MiB ask, i.e. >= 839 MiB/die short BEFORE any draft allocation.
  KV at 200k = 3519.00 MiB total (16 full-attn layers, K/V q4_0 1759.50 each,
  in-split); the 1057.78 MiB ub512 compute buffer is ctx-independent (same size
  allocated by the TP3/200k control), so no ubatch reduction closes the gap.
  TP2@200k remains closed as a topology. Desk stood down from the TP3 total A/B
  per the hub de-conflict (#1336/#1337 - Gemini's lane); one TP3 control boot
  VRAM record handed over (boot-ready 7809.8/7582.0/7634.2 used MiB, die 3
  idle; results/tp2feas_t3on1_20260921_183902_vram.log).
- E-032 2026-09-21 Integration: amd/tp2-mtp-feasibility merged. TP2 desk
  FINAL: TP2@200k closed (fails MTP-OFF, >=839 MiB/die short on a
  ctx-independent 1057.78 MiB buffer - no ubatch reduction closes it; the
  historical 151 MiB was a 10k layer-split-era number, prediction refuted).
  TRUE in-split MTP cost = 1072.8 MiB/die at TP2/10k (context-buffer
  dominated: weights+KV only ~90 MiB; >2 GB does not reproduce at 10k).
  Flag relocates only 251.1 MiB/die (draft ctx [meta, die, CPU] backend
  list keeps compute on serving dies); value = extends TP2 MTP-capable
  ceiling ~17-18k -> ~36k ctx for -2.4% decode; opt-in. MTP itself +38-40%
  TP2 decode. Full three-way table + receipt:
  results/TP2_feasibility_2026-09-21.md. W6 follow-up named: draft-context
  buffer placement must prefer the extra device for full isolation.
- E-033 2026-09-21 Two more desks dispatched (keep-3-agents directive):
  wt-draft-isolation on amd/draft-isolation (FULL draft isolation: draft
  ctx backends must lead with the dedicated device, shared output/embd
  duplicated there ~0.9 GiB budget, zero draft allocations on meta group;
  zero-GPU impl + host tests, die-3 windows negotiated) and
  wt-loader-dequant on amd/loader-dequant (load-time f16 weight residency
  to delete the retained per-call dequant; die-3 bench vs 6.10/4.79
  references). Die-3 primary user: served-validation desk (boots are on
  0-2; die 3 free between its censuses).
- E-034 2026-09-21 W6-isolation FULL DRAFT ISOLATION banked (zero-GPU, desk:
  wt-draft-isolation, branch amd/draft-isolation, base b4ad90c5a; receipt
  results/W6_isolation_2026-09-21.md). Closes the E-032 follow-up
  ("draft-context buffer placement must prefer the extra device"): when
  --spec-mtp-device is set the draft context now allocates NOTHING on the TP
  meta group. Four changes: (1) llama_context ctor inserts params.extra_device
  at index 0 - the draft backend list LEADS [die3, meta, CPU] so the sched
  resolves ops to the device holding the weights; (2) loader create_tensor
  duplicates the draft-visible shared tensors WHOLE onto the MTP device at
  load time (TOKEN_EMBD + OUTPUT, name-dedup for tied embeddings, bytes added
  to size_data not n_created, copies kept OUT of tensors_by_name so
  get_tensor/meta split-state still resolve the originals; originals untouched
  - the target verify pass is not moved or re-buffered); exact GGUF sizes of
  record via offset deltas: output.weight Q6_K 542,942,400 B = 517.79 MiB,
  token_embd.weight IQ4_XS 351,619,840 B = 335.33 MiB, combined 853.12 MiB
  (inside the ~0.9 GiB budget); (3) graph_mtp builders (qwen35, qwen35moe,
  step35, cohere2moe) prefer model.tok_embd_mtp/model.output_mtp -> single
  die3 split, zero meta-group draft allocations; (4) sched_reserve isolation
  audit logs "<dev> isolation audit: A MiB on dev, B MiB on the model split"
  (B must read 0.00), LLAMA_SPEC_MTP_STRICT=1 turns B>0 into a boot failure.
  Flag unset = byte-for-byte unchanged (no dup, no reorder, no audit). PREDICTIONS
  at TP3/200k ub512: serving dies free ~548 MiB/die vs the current flag arm
  (1643.4 MiB meta-side draft compute / 3 dies; ~785 MiB/die vs the no-flag
  arm) -> served 8419/8131/8130 used of 8573 drops to ~7870/7583/7582, free
  154/442/443 -> ~703/990/991 MiB; die 3 residency 1.44 GiB today (10k
  post-probe anchor) + 208.6 KV 10k->200k + 853.1 dups + 387.1 relocated
  compute = ~2.82 GiB central (2.6-3.4 range; the task's ~2.4 GiB guess is
  optimistic), die 3 keeps ~5.3-5.6 GiB free. Per-step PCIe improves slightly:
  the two ~20 KiB meta<->die3 activation hops disappear; the 517,088-byte
  logits row (vocab 129,272 x f32) changes bus side only. Host evidence: test
  test_w6_isolation_host.cpp ALL PASS (dup rule, name dedup, loader
  accounting, tensors_by_name exclusion, backend leading, sched placement
  matrix, audit/STRICT bucketing) + predecessor W6 suite still ALL PASS;
  full clean build-hip gfx900 (ROCm 6.2.0) llama-server 100% exit 0 ZERO
  warnings. NOT RUN: served A/B queued for a negotiated die-3 window
  (protocol in the receipt; smoke = boot-log dup/audit lines with 0.00 MiB
  residue, rocm-smi banked against the analytic numbers).
- E-035 2026-09-21 Integration: amd/draft-isolation merged (8a4ebbcaa).
  FULL DRAFT ISOLATION implemented behind --spec-mtp-device: backend list
  leads with the dedicated die, token_embd+output duplicated whole there
  (853.12 MiB, offset-delta sizes), graph redirects to the mtp copies,
  sched audit logs meta-group residual (STRICT=1 hard-gates 0.00).
  Compile-clean, host suite ALL PASS, unset = byte-identical. PREDICTED
  (200k): serving dies free 154/442/443 -> ~703/990/991 MiB (+548/die);
  die-3 residency ~2.82 GiB. Awaiting die-3 served window (coordinated
  with Gemini) + A/B per the receipt protocol. Desk served its window
  request to Gemini on the hub.

- E-033 2026-09-21 Loader dequant-elimination desk COMPLETE (worktree
  wt-loader-dequant, branch amd/loader-dequant; die 3 only, ~19 min of windows;
  dies 0-2 untouched). LOAD-TIME F16 WEIGHT RESIDENCY IS IN behind
  GGML_CUDA_TILE_FP16_RESIDENT=1 (composes with GGML_CUDA_TILE_FP16): the tile
  route's per-call dequant of the quantized weight slice is replaced by a
  persistent per-(tensor, device) f16 cache, produced ONCE at first use by the
  same dequant kernel the per-call route runs. Design: entries keyed by {device,
  weight-slice device pointer, K, row_diff, type} (weights immutable post-load;
  the census whitelist admits only model-weight slices), raw cudaMalloc NOT
  graph-pool memory (pools recycle within a graph build), freed in the context
  dtor via clear() per-entry device-current (T4 teardown lesson applied); budget
  = cumulative per-device cap GGML_CUDA_TILE_FP16_RESIDENT_MIB (default 4096)
  AND live cudaMemGetInfo free minus 256 MiB margin at first use; refusal is
  final per tensor, one WARN, per-call dequant fallback. Residency whitelist =
  tile census whitelist by construction (fetch sits behind tile_fp16_should_use).
  VALIDATION (die 3, M=128 census shapes, IQ3_S, zero-warning gfx900 build):
  (1) ORACLE 10/10 shapes BIT-IDENTICAL f32 dst (cmp of dumps, resident vs
  per-call, same protocol as E-024); (2) cap-refused arm (CAP_MIB=16 -> 8
  refusals + 2 sub-cap residents) 10/10 byte-identical vs per-call - fallback IS
  the per-call path; (3) free-VRAM gate exercised with a 7900 MiB hog (234 MiB
  free): 10/10 refusals, 0 residents, clean exit - the served-200k behavior
  reproduced on demand; (4) RESIDENT without TILE_FP16 inert (zero route logs);
  (5) teardown clean: VRAM before/after resident run identical (12,550,144 B).
  PERF: wall op-level call-weighted die agg per-call 4.28-4.34 (median 4.32,
  reproduces E-024's 4.33) -> resident 5.41-5.52 (median 5.46) = +26.4%; biggest
  per-shape walls ssm_out 1.38x, attn_q+gate 1.36x, gdn_gate/attn_out 1.30-1.32x,
  attn k/v 1.10-1.13x. GEMM-only kernel trace: 6.32 -> 6.13 TF/s/die aggregate
  (-3.0%) - the E-024 L2-credit prediction CONFIRMED and small; post-loader
  integrated reference is 6.13, above the 5.53 standalone anchor. Deleted tax
  measured: dequantize_block_iq3_s 141.3 ms per 63x10-compute pass vs 3.4 ms
  one-time residency production (10 launches). VRAM delta at bench scale: 304.2
  MiB raw residency for the 10-shape set (analytic 304.25), end-state free-VRAM
  delta 254 MiB vs per-call (pool slice difference included). Receipt:
  results/Loader_dequant_2026-09-21.md; raw runs + traces:
  results/loader_dequant_{percall,resident}_{run.txt,trace csv}_2026-09-21;
  harness tests/bench_tile_resident.cpp + tests/parse_tile_resident_trace.py.
- E-034 2026-09-21 TP3 RESIDENCY BUDGET VERDICT: NEGATIVE for served use - full
  f16 residency of the whitelist shapes does NOT fit at TP3, any context. Exact
  arithmetic (die shares = equal thirds, 48 GDN + 16 full-attn layers, f16 = 2
  B/elem): FFN 64L = 10,736 MiB/die (gate 3600 + up 3600 + down 3536); GDN 48L =
  3,456 (gdn_qkv 1560 + gdn_gate 960 + ssm_out 936); attn 16L = 1,032 (q+gate
  640, out 312, k 40, v 40); FULL WHITELIST = 15,224 MiB/die = 14.87 GiB/die =
  1.78x the entire 8573 MiB die (44.6 GiB across TP3); FFN-only = 1.25x the die,
  does not fit even EMPTY. f16 = 2.61x the 5837 MiB/die quantized whitelist
  bytes. At the 200k boot of record (154 MiB free/die) the free-VRAM gate
  refuses every slice (smallest FFN ask 55.25 + 256 margin = 311 > 154); at
  10k-class boots (763-991 MiB free) only attn k+v (80 MiB/die, +10-13% wall on
  those cells, ~0.7% call-weighted) fits and is not worth a served A/B.
  PRODUCTION: switch stays OFF at TP3 - behavior and numerics unchanged (tile v1
  with per-call dequant, 4.33 wall class); the +26% wall is banked for
  headroom-ful environments (single-GPU short-ctx, >=16 GiB dies). The residency
  question at TP3 is CLOSED unless a weights/KV format change opens >= 15.3
  GiB/die (none on the roadmap). Remaining tile-wall levers named: src1 f32->f16
  convert (16 ms/pass at bench scale) and the residual GEMM gap to 6.13.
- E-036 2026-09-21 Integration: amd/loader-dequant merged (015bf0dbc).
  RESIDENCY VERDICT: full-weight f16 residency does NOT fit at TP3 200k
  (needs 15,224 MiB/die = 1.78x the die; FFN-only 1.25x; at the 154 MiB
  free 200k boot the gate refuses every slice) - NEGATIVE banked with full
  arithmetic; switch OFF in production, +26.4% wall win banked for
  headroom-ful environments (>=16 GiB dies). Oracle byte-identical 10/10,
  budget gates exercised (CAP refusals + 234 MiB-free hog reproduce the
  200k behavior on demand), RESIDENT alone inert, teardown clean. GEMM-only
  -3.0% confirms the E-024 L2-credit prediction. Remaining honest levers:
  src1 f32->f16 convert (16 ms/pass) + residual GEMM gap 4.33 -> 6.13.
- E-038 2026-09-22 W6-isolation desk: E-037 route-back ROOT-CAUSED - the
  TP2@10k decode-cell failure ("Context size has been exceeded. off = 69",
  28 KV retries, HTTP 500) is NOT a defect of the E-035 isolation build and
  the dual-cache cell-accounting-aliasing hypothesis is REFUTED. Forensics
  on tp2feas_t2flag10k4_20260921_213414_server.log: the failing run served
  FIVE completion requests; the battery posts exactly two (serial,
  guard_battery sha identical to the v1 run) - tasks 20 (3122 tok), 35
  (7857 tok) and 46 (32 tok) were FOREIGN clients (TP3 A/B lane harness
  landing on whichever server held the contested port 8080, per E-037's own
  collision story). Two concurrent 7857-token prompts on one 10240-cell
  unified cache (n_parallel auto=4, kv_unified=true) demand 15714 cells;
  the log closes to the exact cell: task 14 placed 6075 + 512 in-flight =
  6587, task 35 placed 3584 + 64 + 4 + 1 trickle = 3653, total 10240 = zero
  free (even nb=1 then fails); the retry storms sit at batch offsets 0/64/
  68/69 (batch offsets, not cell offsets) and the "28 retries" is the
  counted retry-line total. No aliasing exists in code: create_memory for
  mtp_on_hybrid_qwen35 builds a plain llama_kv_cache with mem_other =
  nullptr (the only mem_other consumer is the GEMMA4_ASSISTANT iswa path);
  the duplicated token_embd/output are weight buffers and never enter cell
  arithmetic. Same build, serial load (10k3, killed externally at t+120s):
  ZERO retries through cached 5120 of the identical request. v1's "zero
  retries" comparison was clean-serial vs contested-port - not the same
  geometry. The decode cell was never cleanly measured on the iso build;
  v1's 14.43 t/s / 0.66667 stands until a clean window. RESIDUAL
  build-independent exposures surfaced (flagged, out of desk scope): per-
  slot n_ctx admission ignores global unified occupancy; slot fill order
  starves an older mid-prompt request for a newer one. HARDENING shipped:
  server now logs "MTP draft cache: independent cells (not shared with the
  target)" at boot on --spec-mtp-device; host suite extended with the
  cache-independence rule table + incident capacity arithmetic as
  regression documentation (ALL PASS, predecessor suites ALL PASS,
  llama-server gfx900 rebuild zero warnings). Receipt:
  results/W6_isolation_defect_2026-09-22.md. REQUEST to the coordinator:
  re-verify the TP2@10k decode cell on the iso build under CLIENT
  exclusivity (distinct port, e.g. 8081, or hub-GO quiescence of the TP3
  lane harness) - serial battery; expected zero retries, accept ~0.66667.
- E-039 2026-09-21 Integration: amd/draft-isolation hardening merged
  (772bf7cdc). ISOLATION DEFECT REFUTED - root cause was PORT
  CONTAMINATION: the failing TP2 server log shows 3 foreign requests
  (TP3 A/B lane harness) landed on it mid-flight; demand 2x7857 vs 10240
  unified KV cells closes the context-exceeded arithmetic exactly; no
  cell-accounting aliasing exists in code (draft cache is a separate
  cells object; duplicated tensors never enter cell arithmetic). The v1
  zero-retries comparison was clean-serial vs contaminated-port. Shipped:
  boot-log cache-independence line + host suite incident regression doc.
  NEW LAWS: (1) every desk boot uses its OWN port (validation 8081,
  TP2 desk 8082, Gemini lane 8080) - client exclusivity, not just boot
  exclusivity; (2) flag for the coordinator: per-slot n_ctx admission
  ignores global unified occupancy + slot fill order starves older
  requests - upstream-relevant server exposures, out of desk scope.
  Iso decode cell re-verification pending on a clean exclusive window.
- E-040 2026-09-21 T4 q8_1 act cache SERVED 3-rep OFF/ON A/B COMPLETE (desk3,
  dies 0-2, canonical ub512 boot per arm, 180s cold-stamp idle, settle window
  edge <= 35C + min 120s; per-boot temps in desk3_ab_manifest.jsonl): decode
  OFF 14.95/14.98/14.92 (mean 14.950) vs ON 14.97/14.97/15.07 (mean 15.003)
  = +0.36%, prefill +0.03% (115.36 vs 115.39) - FAILS the +2% promotion gate.
  6/6 boots all-guards PASS, accept 0.66667 everywhere, greedy sha
  4beb1ba25219ee9b byte-identical in every arm (matches the E-027 gate of
  record). ENGAGEMENT PROVEN via one -lv 4 diagnostic boot (ggml INFO is
  verbosity-filtered at the default server thold, which is why no gate lines
  appear in served logs): enable line present + begin_compute stats
  "401 hits / 427 misses over 1024 computes" (~48% hit) - the cache populates,
  hits, and still does not move served decode at ub512/TP3; honest negative.
  (b) CLEAN-SHUTDOWN SMOKE GREEN on the fixed binary: 0 GGML_ASSERT lines of
  any kind in all 6 logs - the E-018 pool_size teardown fix is confirmed
  served. NOT PROMOTED (gate); teardown fix CONFIRMED. Receipts:
  tp3_guards_20260921_{191620,192953,194338,195732,201059,202424}.jsonl;
  diag log desk3_diag_q81_20260921_214339.log; full table
  results/TP3_served_validation_2026-09-21.md.
- E-041 2026-09-21 tile (GGML_CUDA_TILE_FP16=1) SERVED verdict: FAIL -
  STOPPED per protocol. env-unset GREEN arm PASS (dec 15.10, pre 115.17, sha
  4beb1ba25219ee9b). ON arm: env delivery proven (/proc environ), boot healthy,
  then SIGABRT on the FIRST prefill request - "ROCm error: out of memory" at
  ggml-cuda.cu:452 ggml_cuda_pool_leg::alloc inside ggml_cuda_tile_fp16_mul_mat
  <- ggml_cuda_op_mul_mat_cublas (core dump); one off-whitelist WARN
  ("shape off the census whitelist") fired 4s earlier, i.e. draft-model dense
  shapes are OFF-list while target trunk ub512 shapes are ON-list. Mechanism:
  E-031 boot-ready free VRAM is ~200-400 MiB/die at TP3/200k and the tile's
  per-call f16 src0/src1 pool allocs abort die 0; the +27.5% bench-level win
  does not survive the 200k memory envelope. Census cell: census_decode.sh
  (-p 8) reaches NO dense GEMM under either arm (mmvq covers M <= 8 =
  MMVQ_MAX_BATCH_SIZE; zero GEMM-class kernels in both -p 8 traces);
  supplementary -p 512 -sm tensor rocprofv3 pair DOES show the replacement:
  OFF 2976 rocblas Cijk launches / 6729 ms vs ON 1632
  tile_fp16_gemm<128,64,4,8,32> (4325 ms) + 1632 tile_fp16_reduce (331 ms),
  dominant MT128x128x16 class fully displaced, zero off-whitelist WARN at
  -p 512. NOT PROMOTED; re-try gated on the E-036 loader-dequant residency
  step deleting the per-call f16 alloc. Receipts:
  tp3_guards_20260921_{203825,205202}.jsonl; traces
  desk3_c2_p512_{off,on}_kernel_trace.csv.gz; server log
  server_tp3_200k_20260921_205202.log (backtrace of record).
- E-042 2026-09-21 T3 grouped mmvq (GGML_CUDA_MMVQ_GROUP=1) SERVED 3-rep A/B
  COMPLETE: decode OFF 15.05/15.10/15.02 (mean 15.057) vs ON
  14.89/15.11/15.06 (mean 15.020) = -0.24%, prefill -0.04% - FAILS +2%. 6/6
  all-guards PASS, accept 0.66667, greedy sha byte-identical in every arm
  (bit-identity contract holds). CENSUS launch-count cell NOT MET:
  quantize_q8_1 358413 and mul_mat_vec_q 358413 launches IDENTICAL OFF vs ON,
  zero mul_mat_vec_q_grouped launches (census_decode_20260921_224002 vs
  _224043 traces) - the grouping never forms on the canonical config:
  ggml_cuda_try_group_mmvq declines while stream_context().concurrent_events
  is non-empty ("never reorder around the multi-stream machinery", active
  under the TP3 meta-backend butterfly) and/or no eligible same-src1 window
  survives the eligibility filters under split. NOT PROMOTED: mechanism inert
  in this topology; candidate would need the concurrency gate relaxed plus a
  demonstrated group formation before a re-run. Receipts:
  tp3_guards_20260921_{205728,211330,214646,215954,221301,222609}.jsonl.
- E-043 2026-09-21 served-lane incident + contamination audit (desk3): two
  mid-battery SIGKILLs (21:11:15, 21:27:16) and one serving kill (21:34:14)
  were collisions with the tp2-feasibility lane's boots on shared port 8080
  (journal: WINDOW START 21:10:54, GAP CLAIM 21:25:09, GO boot 21:34:16);
  affected manifest rows VOID-RETRIED, retries clean. Post-alert audit of all
  14 completed arm server logs (tests/desk3_contamination_audit.py): exactly
  7 tasks per battery, prompt-token signature [3122, 7857, 32, 32, 6043,
  6043, 6043] identical everywhere, zero serial task overlaps, zero KV-retry
  lines - NO foreign requests landed on any completed arm, so the E-040/E-041/
  E-042 numbers stand as measured. Lane adopted the campaign GPU lock
  convention (/tmp/campaign_gpu_boot.lock) mid-run: wrapper check-and-waits,
  holds through teardown, removes at exit. Same-binary guarantee for every
  arm above: single build-hip rebuild 19:14-19:15 from the b4ad90c5a tree,
  binary/libs mtimes unchanged through the last boot (battery fingerprint
  aa336055d4d73b00 constant; receipt commit field only reflects docs-HEAD
  moves by other desks).
