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
