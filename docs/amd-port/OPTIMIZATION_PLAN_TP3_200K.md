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
- E-044 2026-09-21 SERVED VALIDATION VERDICTS OF RECORD (13 batteries,
  5/5 guards each, greedy sha identical everywhere, contamination audit
  CLEAN on all arms): (1) T4 q8_1 cache +0.36% decode (engagement proven:
  401 hits/427 misses) - NOT PROMOTED, honest negative, teardown fix
  confirmed served; (2) a8 tile - ON arm CRASHED at 200k first request
  (tile f16 pool allocs OOM at 200-400 MiB/die free; tile itself proven
  working: displaces rocblas fully at -p 512) - NOT PROMOTED at 200k;
  reopen path = pool-fitting or the loader residency that E-036 closed;
  (3) T3 grouping -0.24% AND census shows ZERO grouped launches - the
  grouping never forms under the TP3 multi-stream machinery - NOT
  PROMOTED; reopen needs graph-level grouping design. Config of record
  unchanged: ub512, prefill 115.21, decode ~14.95, accept 0.66667.
- E-045 2026-09-21 Two desks dispatched: wt-tile-chunked on
  amd/tile-chunked-dequant (CHUNKED-DEQUANT tile: dequant K-chunks into a
  single reusable scratch so the tile's pool footprint drops from
  100s of MiB to a few MiB - the direct fix for the E-044 200k crash;
  die 3) and the served-measurement lane for TP3 200k MTP-OFF/ON +
  isolation decode cell (dies 0-2 + coordinated die-3 window; answers the
  >2 GB-at-scale question and the W6/isolation A/B).
- E-046 2026-09-21 served-measurement lane: TP3 200k MTP OFF vs ON per-die
  VRAM MEASURED (port 8080, one boot + 5-guard battery per arm, campaign
  lock held per boot, thermal settle <= 35C edge, same-binary E-043 build,
  battery fingerprint aa336055d4d73b00): TRUE per-device MTP cost at 200k =
  1061-1255 MiB/die at boot-ready (mean 1125.4; OFF 6554.6/6520.8/6574.5 vs
  ON 7809.9/7582.1/7634.2), 968-976 MiB/die post-battery (mean 972.4);
  >2 GB/device does NOT reproduce at 200k and cost is nearly ctx-independent
  vs TP2/10k's 1072.8 (+4.9% for 20x ctx) - draft-context-buffer dominated
  per E-030. Die 0 carries ~195 MiB more than dies 1/2 (rank-0 draft-side
  buffers, KV symmetric). ON boot-ready matches the E-031 control record
  within 0.1 MiB (cross-boot reproducibility). OPERATIONAL CEILING
  quantified: MTP-ON serving peak 8150.9-8165.7 of 8176.0 MiB/die = 10.3-25.1
  MiB free at peak - the E-041/E-044 tile-OOM threshold is now a number
  (any per-call pool alloc > ~10 MiB aborts die 0 at the 200k served point).
  Served decode 8k-10k: OFF 12.23 vs ON 15.12 t/s = +23.6% (matches baseline
  untraced_no_spec_reference 12.24); accept 0.0 vs 0.66667; greedy sha
  4beb1ba25219ee9b byte-identical in BOTH arms; needle 3/3 both. Arm A FAIL
  verdict = expected control signature. Single-rep prefill -6.2% (123.14 vs
  115.50) noted, not this lane's cell. Receipt:
  results/TP3_mtp_vram_2026-09-21.md (full table + receipts list).
- E-047 2026-09-21 served-measurement lane: iso decode cell (E-038 request)
  CLOSED on the draft-isolation build: dies 0-2 + --spec-mtp-device on die 3
  at -c 10000, battery on PORT=8081 (lane-only client): 5/5 PASS, prefill
  116.95, decode 15.21 t/s, accept 0.66667 exact (84/126, mean 3.00),
  determinism sha 4beb1ba25219ee9b, needle 3/3; server log 0
  "Context size has been exceeded", 0 retry lines - the E-037 defect
  signature does not reproduce under a serial battery. NAMING CORRECTION of
  record: the W6/dispatch line's --device CUDA0,CUDA1,CUDA2 --spec-mtp-device
  CUDA3 does not resolve on this HIP build (invalid device: CUDA0); actual
  names are ROCm0..ROCm3 per --list-devices; --device must still precede
  --spec-mtp-device. CONTAMINATION AUDIT (E-043 convention): the
  wt-tile-chunked bench was resident AND computing on die 3 during the
  window (card3 allocs during my cooldown, 100% util samples in the decode
  cell, alive at teardown with 363.7 MiB; benches are not lock-gated).
  Client exclusivity HELD (8081, zero foreign HTTP), so the retry/accept/
  determinism/needle cells are contention-immune and CLOSED; decode 15.21 is
  a lower-bound-flavored number under die-3 compute sharing - it already
  matches the canonical 200k MTP-ON decode (15.12) and exceeds the expected
  14.4-14.9 class. Die-3 VRAM residency not cleanly attributable (my draft
  residency bounded 347.6-551.1 MiB over idle; foreign transient dominated
  serving). Attempt 2 (23:55) aborted pre-battery: a new foreign bench was
  already on die 3 before boot. WINDOW REQUEST to the coordinator: one clean
  exclusive die-3 window (~10 min) to optionally re-bank the iso decode t/s
  uncontended; all other cells closed. Receipt:
  results/TP3_iso_decode_2026-09-21.md.
- E-048 2026-09-21 Integration: served-measurement lane COMPLETE (E-046/
  E-047 receipts, commit 269cc31fa). ANSWERS OF RECORD: (1) true per-device
  MTP cost at TP3/200k = ~1.06-1.26 GB/die (mean 1125.4 boot-ready) - the
  >2 GB hypothesis does not reproduce at any measured scale; cost is nearly
  ctx-independent (TP2/10k 1072.8 -> TP3/200k 1125.4, +4.9% for 20x ctx),
  confirming draft-context-buffer dominance; die 0 carries ~195 MiB extra
  (rank-0 draft-side buffers). (2) MTP-ON serves within 10.3-25.1 MiB/die
  of ceiling at peak -> the tile-OOM threshold is now quantified (~10 MiB
  per-call alloc aborts die 0) - direct input to the chunked-dequant desk.
  (3) MTP value confirmed: +23.6% decode (12.23 -> 15.12). (4) Iso decode
  cell CLOSED CLEAN on 8081: 5/5 guards, prefill 116.95, decode 15.21,
  accept 0.66667 exact, zero retries - the E-037 "defect" is fully
  attributed to port contamination. (5) Naming: --device takes ROCm0..3
  on this HIP build, not CUDA0 (receipt-documented). ARBITRATION NOTE:
  bench desks must hold /tmp/campaign_gpu_boot.lock too - the tile-chunked
  bench was die-3-resident during the iso decode window (result banked as
  lower-bound; uncontended re-bank optional via E-047 window request).
- E-045 2026-09-21 W5 tile-CHUNKED desk COMPLETE (worktree wt-tile-chunked,
  branch amd/tile-chunked-dequant; die 3 only, ~20 min of windows; dies 0-2
  untouched). The E-044 C2 OOM class is DELETED by construction: behind
  GGML_CUDA_TILE_FP16=1 + GGML_CUDA_TILE_FP16_CHUNKED=1 the a8 tile now
  dequantizes ONE split-K slice at a time into a 17.1 MiB reusable window
  (f16 N_d x (ksl+bs) + gathered quant blocks; raw cudaMalloc, per-stream,
  grow-only, 64 MiB budget + 64 MiB free-VRAM margin, refusal -> shipped route
  with one WARN - never aborts). Bit-identity CONTRACT MET: chunk boundaries
  are the unchunked kernel's own gridDim.z partition from pick_ks, each slice
  launch runs the same half2 schedule over the same f16 values (gather feeds
  the same to_fp16 kernels the per-call route uses), ACC epilogue adds the
  slice f32 partials in the reduce's fixed order - oracle 10/10 PASS
  (unchunked still 0.00e+00 vs the host spec, L2 band 2.9-6.1e-3, chunked
  bit-diff 0/N on every shape) AND full-glue f32 dst dumps cmp 10/10
  byte-identical on real IQ3_S; env unset + M=640 off-list behavior unchanged.
  VRAM proof: measured peak transient delta (0.4 ms sampler, per-shape
  steady-to-min) chunked 52 MiB @ M=128 / 56 MiB @ M=512 vs unchunked
  104/144 MiB, envelope 200 MiB - FIT with margin; no partials buffer (P path
  unused), pool only sees small src1 blocks. PERF (die 3, census, M=128,
  3x20 median): wall call-weighted 2.77-2.81 TF/s/die vs unchunked 4.28-4.33
  (the E-024 4.33 reference reproduced) = -35%; kernel GEMM-only call-weighted
  3.68 vs 6.39 (6.10 reference class) - the slice launch gives up the
  unchunked gridDim.z fill: per-launch grid (N_d/64, M/128, 1) is 26 blocks
  for ffn_down / 4 for attn_k, so ks=8 shapes lose 40-87% GEMM while ks=4
  big-N shapes lose only 4-6%; dequant+gather work itself is NOT the tax.
  M=512 (served ub512 point): wall 5.17 vs 5.72 = -9.6%, grid.y=4 closes most
  of the fill gap. VERDICT: memory-viable at 200k (bench level; served boot
  confirmation = Gemini lane), perf-negative vs the unchunked tile at M=128,
  but the unchunked tile is exactly the arm that OOM-aborts at 200k - the
  honest 200k comparison is vs the shipped f32 SGEMM route. NEXT LINK named:
  (1) N-span chunking with full-K windows - contiguous quant ROW slabs need
  NO gather, launch keeps gridDim.z=ks + P/reduce per chunk so the fill tax
  closes while bit-identity holds (same per-element slice schedule + reduce
  order); (2) endgame = fused dequant-in-staging tile (per-quant-type kernel
  project, deletes window and extra pass). Receipt:
  results/W5_tile_chunked_2026-09-21.md; traces + run logs:
  results/W5_tile_chunked_{trace_unchunk,trace_chunked,bench_unchunk,
  bench_chunked}_2026-09-21.{csv,txt}; parser tests/parse_tile_chunked_trace.py.
- E-049 2026-09-21 Integration: amd/tile-chunked-dequant merged
  (24f7430cc, GGML_CUDA_TILE_FP16_CHUNKED=1). SPLIT VERDICT: memory SOLVED
  (window 17.1 MiB steady, peak transient 52-56 MiB - fits the 200k
  200 MiB envelope with 3.5x margin, deletes the E-041 abort class;
  dedicated cudaMalloc never the OOM-ing pool; refusal falls back to
  shipped) - performance NEGATIVE at M=128 (-35% wall: 2.77-2.81 vs
  4.28-4.33; -42% GEMM-only) because per-split-K-slice chunking forfeits
  the unchunked launch's gridDim.z fill on ks=8 shapes (-40-87%; ks=4
  shapes lose only 4-6%). At M=512 only -9.6%. Oracle 10/10 bit-identical;
  env-unset unchanged. NAMED NEXT LINK: N-span chunking with full-K
  windows (contiguous quant row slabs, no gather, keeps z-fill) - the
  design that could give memory-fit AND parity. Until then the 200k tile
  question stays: unchunked crashes, chunked is slower than shipped f32
  SGEMM at M=128 (2.8 vs ~4.3 wall).
- E-050 2026-09-21 N-span chunking desk dispatched: wt-tile-nspan on
  amd/tile-nspan - the named E-049 next link (N-span windows with full-K
  contiguous slabs: no gather, keeps gridDim.z fill, targets memory-fit
  AND M=128 parity vs the 4.28-4.33 unchunked reference). Die 3.
- E-050 2026-09-21 W5 tile-NSPAN desk COMPLETE (worktree wt-tile-nspan, branch
  amd/tile-nspan; die 3 only, ~30 min of lock-held windows; dies 0-2 untouched).
  E-049's named next link LANDED and CLEARED BOTH 200k GATES at bench level.
  Behind GGML_CUDA_TILE_FP16=1 + GGML_CUDA_TILE_FP16_NSPAN=1 (tried before the
  chunked arm) the a8 tile now dequantizes contiguous N-SPANS of weight rows with
  the FULL K dimension straight from the quant tensor (rows are contiguous slabs;
  the per-type to_fp16 kernels are linear over quant blocks, so the gather of the
  chunked arm is deleted entirely) into ONE reusable window per stream (f16 slab
  span x K + span partials ks x M x span; default budget 128 MiB total via
  GGML_CUDA_TILE_FP16_NSPAN_MIB; raw cudaMalloc, grow-only, 64 MiB free-VRAM
  margin, refusal -> shipped route with one WARN, never aborts). Every span GEMM
  keeps the unchunked gridDim.z = ks partition and its P + fixed-order f32 reduce
  (new tile_fp16_reduce_span: same slice-sum order, float4 mapped into strided
  dst, needs ldc % 4 == 0); tile_fp16_gemm itself UNCHANGED (C base offset + N =
  span rows are call-site params). Bit-identity CONTRACT MET: output columns are
  disjoint across spans and each element's slice schedule + f32 sum order are the
  unchunked ones - oracle 10/10 PASS twice (single-span AND forced multi-span
  schedules; unchunked still 0.00e+00 vs host spec, L2 band 2.9-6.1e-3, chunked
  and nspan bit-diff 0/N on every shape), full-glue f32 dst dumps cmp 10/10
  byte-identical nspan-vs-unchunked at M=128 AND M=512 on real IQ3_S, env-unset
  10/10 byte-identical vs the pre-change served library. BUG OF RECORD found +
  fixed (2 lines, latent in the merged chunked arm too): in-place window growth
  updated the buffers but not the cached entry sizes, so the next call re-entered
  the realloc path and - on the CUDA-graph capture pass - the realloc's stream
  sync aborted ("operation not permitted when stream is capturing"); reproduced
  on gdn_qkv (first shape needing a bigger P slab), fixed in both arms, oracle
  re-gated, crash scenario passes in the clean battery. PERF parity (die 3,
  census, 3x20 median): M=128 wall call-weighted nspan 4.31 vs unchunk 4.32
  same-session control (bar >= 4.28 MET; the E-024 4.33 class reproduced) = -0.2%
  vs the chunked arm's -35%; M=512 5.65 vs 5.68 (-0.5%); kernel traces show
  launch-count parity (1 gemm + 1 reduce + 1 dequant + 1 convert per compute,
  reduce renamed tile_fp16_reduce_span) and GEMM-only within 1-3% per shape,
  route call-weighted 4.44 vs 4.48 - the chunked z-fill tax is GONE by
  construction. Budget dial banks the span-loop cost curve: NSPAN_MIB=8 forces
  ffn_down to 9 spans x 192 rows (24 blocks/launch vs 208) = 2.15 TF/s on that
  shape (-52%, same class as chunked's 2.18 - tight windows converge to the same
  fill physics: blocks/launch ~ (span/64) x ks must stay >= ~112); NSPAN_MIB=16
  refuses everything above attn_k/v -> clean shipped fallback. VRAM proof (0.4 ms
  sampler): max census window 67.5 MiB @M=128 / 101.2 @M=512 (f16 56.25 + partials
  11.8/47.2), one-time, refusal-protected; measured peak transient 102 MiB @M=128
  / 140 @M=512 vs the 200 MiB envelope - FIT; steady 10-shape retention 88 MiB vs
  unchunked 126 (M=128), 206 vs 248 (M=512) - the E-041/E-044 multi-size-class
  per-call f16 pool class is deleted, the pool only sees small src1 blocks.
  VERDICT: N-span gives memory-fit AND parity - the design door E-049 named is
  REAL; the 200k tile question now reduces to the served boot confirmation
  (Gemini lane: guards with both envs, in-arm greedy determinism, MTP accept,
  p512 trace cell - tile_fp16_gemm + tile_fp16_reduce_span displacing cublas,
  zero off-whitelist launches; watch die-0 first-prefill: window ask + 64 MiB
  margin vs boot-ready free). Stretch 5.5+ remains M=512-only (5.65), as with the
  unchunked arm - M=128 wall is dequant-bound, not a span effect. Endgame design
  door unchanged: fused dequant-in-staging tile (per-quant-type kernel project).
  Receipt: results/W5_tile_nspan_2026-09-21.md; run logs + traces:
  results/W5_tile_nspan_{oracle,bench,capture_realloc_bug,trace_summary}_
  2026-09-21.txt, results/W5_tile_nspan_trace_{nspan,unchunk}_2026-09-21.csv;
  parser tests/parse_tile_nspan_trace.py.
- E-051 2026-09-21 Integration: amd/tile-nspan merged (d3e9eda3f,
  GGML_CUDA_TILE_FP16_NSPAN=1 composing with TILE_FP16=1). BOTH 200k
  GATES MET at bench level: memory-fit (peak 102-140 MiB incl. one-time
  window 67.5-101.2, vs unchunked 102-144 pool class that aborted) AND
  parity (4.31 TF/s at M=128 vs 4.28 control; 5.65 at M=512; kernel
  within 1-3% per shape; launches unchanged at 4). The multi-size-class
  per-call f16 pool class that aborted die 0 is GONE. Bug of record
  found+fixed (also latent in the merged chunked arm): in-place window
  growth never updated cached entry sizes -> realloc re-entry + stream
  sync abort on the graph-capture pass; fixed in both arms, oracle
  re-gated. NAMED NEXT: served boot on the Gemini lane (guards both envs,
  within-arm determinism, p512 trace cell, watch die-0 first-prefill vs
  window ask + 64 MiB margin - clean refusal if tight). Endgame door:
  fused dequant-in-staging tile deletes even the window.
- E-053 2026-09-21 Two zero-GPU desks dispatched (GPU grant stays with the
  TP2 ladder desk): wt-server-fixes on amd/server-exposures (fix the two
  E-039 server exposures: per-slot n_ctx admission vs global unified KV
  occupancy; slot fill order starving older mid-prompt requests - both
  found in the contamination forensics, both host-testable) and
  wt-t3-graph on amd/t3-graph (T3 reopen: design graph-level grouping that
  forms under the TP3 multi-stream machinery - the E-044 zero-launch
  finding). No die usage: both are impl + host-test desks.
- E-054 2026-09-22 W-server-fixes desk COMPLETE (worktree wt-server-fixes,
  branch amd/server-exposures; the E-053 zero-GPU dispatch; ZERO die time).
  Both E-039 server exposures FIXED in tools/server/server-context.cpp with
  host-test reproductions (no GPU anywhere): (1) GLOBAL UNIFIED KV ADMISSION
  (default ON, --kv-admission / LLAMA_ARG_KV_ADMISSION, kv_unified-only):
  process_single_task now launches a task only if its remaining prompt cells
  (kv_unified_cells_needed: task tokens minus the candidate slot's cached
  common prefix) PLUS the remaining prompt cells of every other in-flight
  request fit in n_ctx - total occupied cells; failure path = purge idle
  cached prompts first (try_clear_idle_slots gained a skip param so the
  candidate's fresh prefix cache is never purged), 400
  EXCEED_CONTEXT_SIZE if the request exceeds the whole cache, otherwise
  DEFER into the existing deferred queue (re-admitted on the next slot
  release) - clean queueing instead of the E-037 accept-then-starve that
  ended in "Context size has been exceeded. off = 69" HTTP 500 for ALL
  in-flight requests; (2) FIFO SLOT FILL ORDER (default ON, --kv-fifo-fill /
  LLAMA_ARG_KV_FIFO_FILL): the pre_decode prompt-fill loop iterates
  processing slots by task id (arrival order) instead of slot index, so the
  older mid-prompt request wins the remaining cells first (the incident's
  newer-on-slot-0 trickle-while-veteran-starves pattern). Default ON is
  justified per exposure because the legacy behavior is objectively a
  starvation bug; both knobs are documented escape hatches back to legacy;
  admission engages only under kv_unified (non-unified slots own private
  cells regions and per-slot admission is already exact). HOST TESTS
  (docs/amd-port/tests/test_server_exposures_host.cpp, house convention,
  ALL PASS): unit mirrors of both new pure functions; defect 1 reproduction
  closing the incident to the exact cell (6587 + 3584 + 64+4+1 = 10240,
  fatal at off = 69, both requests 500); defect 2 reproduction (legacy fill
  round gives the whole 512 batch to the newer task, FIFO flips it);
  admission decision table (defer on the incident geometry, admit empty,
  prefix-reuse boundary flip, knobs-off legacy, REJECT_TOOBIG,
  purge-then-admit); full fixed trajectory (task 35 deferred, task 14
  completes 7857 + generates + releases, deferred popped, stale cache
  purged, task 35 completes, zero aborts); FIFO-only control proving
  admission is the load-bearing fix. Predecessor test_w6_isolation_host
  re-run ALL PASS. Compile-clean: direct g++ -std=c++17 -O1 -c -Wall
  -Wextra of server-context.cpp + common/arg.cpp exit 0 (only pre-existing
  header-static warnings; cmake/ninja unavailable in this environment,
  full linked build deferred). Receipt:
  results/Server_exposures_2026-09-22.md (includes served-window validation
  plan: 10k-class -kvu boot, two concurrent 7857-token prompts on one port,
  expect defer line + zero retries + zero 500s; and residual exposures:
  generation pressure vs the nb=1 TODO, parent/child copy_state_to cell
  duplication not modeled at admission).
- E-055 2026-09-22 Integration: amd/server-exposures merged (7b1e43505).
  BOTH E-039 SERVER EXPOSURES FIXED, host-tested to the exact incident
  cells: (1) global unified-KV admission control (default ON,
  --kv-admission/--no-kv-admission): a task launches only if its remaining
  cells plus every in-flight request's remaining cells fit n_ctx; purge
  idle cached prompts first; 400 if over-cache, else defer to the existing
  deferred queue - the incident trajectory (6587+3584 trickled 64/4/1 ->
  off=69 HTTP 500) can no longer occur; (2) FIFO slot fill order (default
  ON, --kv-fifo-fill): older mid-prompt requests get priority over newer
  on remaining cells. Host suites ALL PASS incl. predecessor W6 suite;
  compile-clean. Served validation queued (10k boot, two concurrent 7857
  prompts: second defers with log line, completes after first, zero
  context-exceeded). Residual exposures documented (generation pressure,
  parent/child n_cmpl>1 cell duplication) - out of desk scope, ledgered.
- E-037 2026-09-22 TP2 desk: E-035 full-isolation build VERIFIED on TP2 at
  10k (worktree wt-tp2-mtp synced to 7d3ab351a, binary v152; receipt
  results/TP2_feasibility_2026-09-21.md Result 4). AUDIT GATE PASSES:
  "ROCm2 isolation audit: 132.02 MiB on ROCm2, 0.00 MiB on the model split"
  (STRICT=1 armed), duplication lines confirm token_embd 335.3 + output 517.8
  moved whole to the draft die. VRAM (boot-ready -> post-probe): serving dies
  7344.3/7343.9 -> 7930.2/7929.8 used (v1 flag: 7348.5/7347.9 -> 7945.3/7944.7);
  draft die 1412.5 -> 2290.8 (v1: 558.0 -> 1437.1). Duplication delta exact
  (+854.5 vs 853.12); draft-die total matches the ~2.3 GiB prediction; the
  serving-die ~800 MiB further-drop prediction is REFUTED (measured -15 MiB):
  serving-die request-time footprint is TARGET-compute dominated, identical in
  both builds - full isolation buys the audit guarantee and a draft-free meta
  split, not serving-die headroom. Prefill 88.74 t/s PASS. Decode/accept cells
  for this arm lost twice to port-8080 collisions with the TP3 A/B lane (both
  servers free_port-kill each other; hub #1343-#1345); v1 flag numbers (14.43
  t/s, accept 0.66667) stand as the arm reference pending a clean decode-only
  slot. LAW CANDIDATE from these two incidents: cross-desk free_port on a
  shared port needs a hub GO handshake, not just a gap claim - a claimed gap
  with a duration estimate still raced. LOCK CONVENTION adopted per
  coordinator: /tmp/campaign_gpu_boot.lock (desk + ts + duration,
  check-and-wait, release at teardown) implemented in run_tp2_feasibility.sh;
  third run under the lock completed cleanly. FOURTH FINDING (defect route-back
  to the draft-isolation desk): on the E-035 build the TP2/10k decode cell
  FAILS - 7857-token request exhausts KV-space retries (28, down to n_batch=1)
  and errors "Context size has been exceeded. off = 69" -> HTTP 500; v1 build
  on the identical request/config: zero retries. KV geometry identical between
  builds (n_ctx_seq 10240, unified, Meta KV 90 MiB, draft KV 40 MiB ROCm2) -
  suspect dual-cache cell-accounting aliasing (draft cache seq state marking
  target unified cells occupied). Iso decode/accept numbers NOT bankable at
  10k until fixed; v1 flag numbers (14.43 t/s, 0.66667) remain the arm
  reference. Prefill 88.91 t/s PASS under the lock (88.74 first run).
- E-038 2026-09-22 TP2 desk: FOUR-ARM definitive table complete, zero VOIDs
  (receipt Result 5/6; commits e5169d35d + successor). Arm C decode FILLED on
  the isolation build: 7.42/7.41 t/s (reproduced, zero KV-retries, accept
  0.66667/3.00) = -48.5% vs the v1-flag build's 14.43/14.43 on identical
  flags/config. Savings ladder per serving die (post-probe): in-split MTP
  costs ~961 MiB/die over MTP-OFF; v1-flag costs ~790; full isolation costs
  ~786 - i.e. full isolation saves only ~4.5 MiB/die vs the v1 flag (169 vs
  in-split) while DOUBLING the per-token draft cost: the audit's 0.00
  meta-side residual is bought by running the draft's embedding+LM-head on
  the 1x-bandwidth draft die. VERDICT: E-035 full isolation is not a
  promotion candidate for latency-sensitive TP2 serving; the v1 flag (partial
  relocation, head stays on meta) is the operating point, and full isolation
  only matters if serving-die VRAM is the binding constraint. Acceptance
  0.66667/3.00 in every arm. TP2@200k remains closed (E-028/E-031).
  TP3 ladder paused mid-10k (off1/on1/flag1/off2 banked: pp 122.58/120.74 OFF,
  117.39/84.64+81.67 in-split, 117.64 flag; decode 12.17-12.18 OFF, 15.38
  in-split, 7.57 flag = the same isolation decode collapse on TP3) - on2
  interrupted by priority correction, to resume after the TP2 bank.
- E-056 2026-09-22 Integration: amd/tp2-mtp-feasibility final four-arm
  table (8cddf5229). VERDICT OF RECORD for TP2@10k: v1-flag is the
  operating point (decode 14.43, 231 MiB/die free, 0.66667). FULL
  ISOLATION REJECTED for latency serving: decode 7.42 t/s (-48.5% vs v1)
  reproduced clean with zero retries - the mechanism is the design itself:
  the draft's embedding+LM-head run on the 1x-bandwidth draft die plus 2
  host-staged hops/step; the 0.00 audit and the decode collapse are the
  same choice. VRAM: D-vs-C = ~4.5 MiB/die (nothing); D-vs-OFF = ~786.
  MTP value class confirmed: in-split +38-40% decode vs OFF at 10k-era;
  TP3@10k flag arm shows the same isolation collapse (7.57 vs 15.38
  in-split). Isolation remains viable ONLY if serving-die VRAM is the
  binding constraint; buffer-placement follow-up would be needed to make
  it latency-viable. Receipt Result 6; full four-arm table banked.
