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
- E-054 2026-09-22 MTP-overhead desk dispatched: wt-mtp-overhead on
  amd/mtp-overhead (zero-GPU). TARGET: the per-draft-step fixed overhead
  (~37 ms/step measured envelope vs a few ms of draft compute) that caps
  the MTP decode multiplier at ~1.15x instead of ~3x. Named components:
  the CPU sampling sync per draft step (backend sampling unsupported
  under SPLIT_MODE_TENSOR - server warning of record), the full-vocab LM
  head GEMV per step, and pipeline drains. Primary work item: make
  sampling run on-device under tensor split (env-gated), plus timeline
  instrumentation to attribute the rest.
- E-057 2026-09-22 Coordinator: DESK LAW AMENDMENT (Chris direct).
  Waits/status polls use single sleeps of up to 150 s; 10-second
  busy-polling is banned (token waste, Chris: "150 second timeouts are
  fine, 10 second is wasting token usage"). Relay complete: MTP-overhead
  desk (ACKed, adopted), T3-reopen desk, Gemini hub #1357, and the
  30-min automation prompt. Stay-active law unchanged (no indefinite
  sleeps, no external ACK dependencies). Also: amd/server-exposures
  (7b1e43505) confirmed merged in campaign HEAD lineage; its served
  validation (10k-class dual 7857-token concurrent prompts on one port:
  expect defer log line, second request completes after the first, zero
  "Context size has been exceeded", zero 500s, then --no-kv-admission
  legacy-failure control) is QUEUED behind the TP3 200k definitive
  ladder, lane 8081, boot-lock convention. Probe script staged at
  docs/amd-port/tests/probe_kv_admission.sh. TP3 200k ladder state:
  t3on1 under lock since 07:11:57 (runner chain tp3def200_on1/flag1/
  on2/flag2 in /home/chris/).
- E-058 2026-09-22 TEMP PROGRAM, coordinator direct execution (Chris:
  "what about the temp idea? do not message other agents"). CARD MAP
  CORRECTION of record: card2 is the NVIDIA boot display (0x10de); the
  V340L dies are card0 (0d:00.0), card1 (05:00.0), card3 (08:00.0),
  card4 (10:00.0); serving = card0/1/3, draft = card4. Historical VRAM
  log labels that say card1/2/3 need reinterpretation against PCI
  order. Live thermal evidence during the t3flag1 ladder arm: card3
  77-81 C junction - the confound the sweep targets. Undervolt path
  verified: pp_od_clk_voltage writable on all four dies; stock OD_SCLK
  level 5 = 1269 MHz @ 1150 mV, stock curve puts 1200 MHz at ~1125 mV;
  OD_MCLK level 3 = 945 MHz @ 1150 mV is NEVER written. Sweep script
  docs/amd-port/tests/voltage_sweep.sh: pin sclk 1200 via level-5
  rewrite + manual perf level + level force; baseline decode cell at
  1125 mV; steps 1100..850 in -25 mV; per-step gates = mclk still 945,
  decode t/s >= 97% of baseline, accept >= 0.63 (when canary reports),
  junction < 95 C with INSTANT revert to stock on any failure; phase 3
  applies the winner to all four dies and re-stamps the 200k guard
  battery; EXIT trap restores stock tables + perf level auto. Uses the
  proven wt-tp2-mtp binary, lane 8081, guard battery as the only
  measurement. Watcher launched detached: waits (150 s cadence) for the
  ladder chain to end + lock-free + dies drained (< 500 MiB), then
  holds the lock (desk=temp-sweep, 75 min) and runs the sweep under
  sudo; watcher log /home/chris/tp3_temp_sweep_watcher.log (exact-name
  LOG LAW applies). Receipt lands at docs/amd-port/results/
  voltage_sweep_<ts>.log.
- E-059 2026-09-22 TEMP PROGRAM REVISION (Chris: "the priority after
  their tp3 run is get temps under control through a bunch of short
  experiments, 10k runs and mtp overhead in parallel"). Sweep script
  rewritten to be ALL-SHORT/10K: single server boot for the whole
  sweep (lane 8081, 10k, in-split MTP, wt-tp2-mtp binary); phase 0
  stock reference cell (perf auto, no pin) for the thermal + t/s
  reference; phase 1 pin 1200 MHz @ 1125 mV reference cell; phase 2
  steps 1100..850 with the gate kit (mclk 945 untouched, decode
  t/s >= 97% of pinned baseline, accept >= 0.63 when canary reports,
  junction < 95 C, instant stock revert); phase 3 winner on all four
  dies + FULL guard battery at 10k; EXIT trap restores stock. The
  200k confirmation phase is dropped - 200k re-stamps happen only
  after temps are controlled. Junction sampled every 2 s during each
  cell (voltage_sweep_<ts>_temps.log) so every step reports its own
  max temp. MTP-overhead desk continues in parallel (zero-GPU), as
  ordered. Ladder t3flag2 was the last 200k boot (started 07:29:44);
  watcher fires the sweep on its next 150 s tick after teardown + die
  drain.
- E-060 2026-09-22 TP3 200k DEFINITIVE TABLE (coordinator read of the
  tp2-feasibility desk receipts, wt-tp2-mtp/docs/amd-port/results/
  tp2feas_t3*_{on,flag}{1,2}_20260922_07*_battery.jsonl; desk may add
  its own entry on resume - keep both). In-split MTP (t3on) @ 200k x2:
  decode 14.73 / 14.67 t/s PASS (-1.4% vs 14.94 baseline), prefill
  113.69 / 113.43, accept 0.66667, mean_len 3.0. v1-flag draft-device
  (t3flag) @ 200k: decode 7.58 / 7.58 / 7.57 t/s = -49.3% (decode gate
  FAIL is the ONLY failing cell), prefill 117.6 / 117.9 / 117.5 (+2%),
  accept 0.66667, determinism byte-identical, needle PASS except one
  rep with 'Remote end closed connection without response' (server
  died mid-needle on that boot - needs the server-log crash check).
  VERDICT OF RECORD: at TP3@200k the draft-device relocation costs
  ~2x decode exactly as at TP2@10k (7.4 vs 14.4); in-split MTP is the
  operating point at every measured geometry; serving-die VRAM under
  the flag arm at 200k ~1127/1075 MiB free + draft die 2.29 GiB. The
  decode collapse mechanism (draft embedding+LM-head on the 1x-
  bandwidth die + host hops) is now confirmed at 200k, doubling the
  weight behind the MTP-overhead desk's on-device sampling work.
- E-054 2026-09-21 T3-reopen desk COMPLETE (worktree wt-t3-graph, branch
  amd/t3-graph; zero GPU). E-042 "grouping never forms" ROOT-CAUSED with a
  host-proven mechanism: ggml-alloc RECYCLES the ssm_beta MM dst range for the
  ssm_alpha MM dst in the exact qwen35 GDN-layer graph (192 B block, identical
  address, proven by allocating the real topology with ggml_gallocr on host -
  results/T3_graph_2026-09-21.md), so the grouped path's pairwise dst-overlap /
  WAR safety checks drop the only sub-whitelist same-src1 pair on every GDN
  layer on every die -> zero grouped launches, census identical OFF/ON. The
  rejection was CORRECT for the early-write design (an early grouped write of
  alpha would clobber the range the sigmoid still reads as beta); the design
  needed copy-back, not a weaker check. E-042's concurrency-gate hypothesis
  REFUTED (concurrent_events only populate under GGML_CUDA_GRAPH_OPT=1 AND a
  single visible device - inert at 3 dies), the meta-partitioning hypothesis
  REFUTED (per-device subgraphs are order-preserving index-range copies with
  stable per-device src pointers; the 16-node window sees the pair; served
  decode proven T=1x1x1 from the census quantize/mmvq grid shapes). SECOND
  structural blocker proven: attn k/v weights split at granularity
  lcm(1536,256)/6 = 512 -> per-die shards 512 rows (or 0 on the rotating third
  die) -> default MAX_ROWS=256 aborts every k/v group, and E-025's planned 384
  sweep would have failed too (512 is the floor). FIX IMPLEMENTED behind the
  same GGML_CUDA_MMVQ_GROUP=1 switch: aliased members admitted as COPY-BACK
  (grouped launch writes a per-device pool temp; the graph loop issues one D2D
  copy temp->dst at the member's OWN graph position - solo timing,
  capture/replay-safe, bit-identical by construction); direct members and the
  head keep the unchanged early write; copy-back rejected over split buffers
  and off-device dsts (legacy paths keep strict rules); x-mutation and
  fusion-span exclusions factored into ggml_cuda_mmvq_group_x_fusion_safe as
  hard rules for both classes. INSTRUMENT: GGML_CUDA_MMVQ_GROUP_DEBUG=1 logs
  gate-naming decline reasons + formation summaries at WARN (survives default
  verbosity filtering, 64-line budget) - closes the evidence gap where the ON
  census arm had no artifact proving env delivery (enable INFO line filtered).
  HOST TESTS (no GPU): tests/test_t3_detect_host.cpp ALL PASS on a real
  gallocr-allocated graph (pair forms 2 members, later member copy-back,
  recycled-range overlap fires on real ranges, big-cell whitelist aborts +
  raised-sweep admission, T=8 tail ineligible, kv 512/512/0 mirror);
  tests/test_t3_alias_host.c banks the order-sensitivity trap (short chain does
  not recycle, full chain does). gfx900 compile clean (build-hip flags,
  syntax-only, zero diagnostics). Env unset = zero behavior change. DEFERRED
  DEVICE VALIDATION (one dies 0-2 boot): MMVQ_GROUP=1 + DEBUG=1 server log
  shows "group formed ... (copy-back)" per layer; census expects
  mul_mat_vec_q_grouped = 144/step class with mmvq dropping equally,
  quantize halving on the pair class, +1 copy kernel per pair; greedy sha gate
  vs env-unset; then 3-rep A/B at the 200k config (bar >= +2%); optional kv
  arm at MAX_ROWS=512 (not 384). Receipt: results/T3_graph_2026-09-21.md.
- E-061 2026-09-22 Coordinator integration: amd/t3-graph merged into
  amd/v340-port-v2 (ffbce3dc3, ledger conflict resolved keep-both).
  Copy-back grouped-mmvq design is now in the campaign tree behind
  GGML_CUDA_MMVQ_GROUP=1; env unset = zero behavior change. DEVICE
  VALIDATION QUEUED behind the temp sweep (one dies 0-2 boot, lock
  convention): boot with MMVQ_GROUP=1 + MMVQ_GROUP_DEBUG=1 -> expect
  per-layer "group formed ... (copy-back)" WARN lines; rocprofv3
  census -> expect mul_mat_vec_q_grouped ~144/step class, solo mmvq
  dropping equally, quantize halving on the pair class, one 192 B D2D
  copy per pair; greedy determinism sha vs env-unset; then 3-rep 200k
  A/B (bar >= +2%); optional kv-class arm at MAX_ROWS=512 (512 is the
  structural floor, not 384). Dispatched as a queued validation desk
  (waits on the boot lock at 150 s cadence while the temp sweep owns
  the dies).
- E-063 2026-09-22 TEMP SWEEP FINAL + REAL v1-FLAG MEASUREMENT
  LAUNCHED. Sweep complete (receipt results/voltage_sweep_20260922_
  073449.log + _temps/_thermal): winner 1100 mV @ pinned 1200 MHz;
  stock ref 15.60 t/s @ 77 C junc; pinned 1125 mV ref 15.98 @ 82 C;
  1100 mV stable 15.94; 1075 mV UNSTABLE (t/s < 97% + 92 C) with
  instant revert; full winner guard battery max_junc 92 C (inside the
  95 C gate but close - watch it at 200k); stock tables restored by
  the EXIT trap. Winner at 200k deliberately NOT re-stamped yet - the
  E-062 v1-flag correction takes the dies first. Duplicate sweeps
  stood down (Gemini hub #1359, TP2 desk messaged): temp program is
  the coordinator lane per Chris. REAL TP3 v1-flag cell launched:
  worktree wt-v1flag at b4ad90c5a (= 8a4ebbcaa^, the tree where
  --spec-mtp-device is the v1 PARTIAL relocation, server-side impl in
  tools/server/server-context.cpp, target head stays on the meta
  group); arm t3flag via the desk's runner staged in that worktree
  (BIN auto-resolves to the v1 build), VISIBLE 0,1,2,3,
  --spec-mtp-device ROCm3, STRICT unset (does not exist in this tree);
  200k x2 then 10k x1, lane 8081, lock honored, guard battery vs
  baseline_tp2_200k.json; exact logs /home/chris/v1flag_200k_rep1.log,
  v1flag_200k_rep2.log, v1flag_10k_rep1.log. ARM IDENTITY LAW
  evidence: provenance = commit b4ad90c5a + binary sha256 recorded at
  launch; expected signature if v1 is real: decode ~14-15 t/s class
  (vs 7.58 full isolation) with draft die NOT carrying duplicated
  output weights.
- E-057 2026-09-22 MTP-overhead desk COMPLETE (worktree wt-mtp-overhead, branch
  amd/mtp-overhead; the zero-GPU draft-step 37 ms attack; NO die time).
  ROOT CAUSE BANKED: the "backend sampling not supported with
  SPLIT_MODE_TENSOR; using CPU" line fires from llama_context::set_sampler
  (src/llama-context.cpp), triggered by the draft-mtp ctor's
  llama_set_sampler(ctx_dft, top_k(10) chain) offload attempt
  (common/speculative.cpp; --spec-draft-backend-sampling defaults ON).
  Provenance: introduced by the same upstream PR that built backend sampling
  (#23287, ad2775726) as a defensive "-sm tensor" fallback - UNIMPLEMENTED,
  not proven fundamental. KEY QUESTION ANSWERED: NO die holds the full summed
  logits row after the draft LM head in this build - output.weight is
  vocab-axis sharded (llama-model.cpp split config AXIS_1 -> meta head result
  AXIS_0-disjoint), the head triggers NO allreduce (meta boundaries only
  after PARTIAL nodes, i.e. the ffn_down class inside the block), and the
  full ~517 KB row materializes only on the host via the spliced 3-way
  get_tensor_async. Also banked: the drafted token is cur_p->data[0].id =
  top-1 logit (greedy; the chain's trailing dist only fills p for p_min), and
  id(N) is a true compute input of draft step N+1 (graph_mtp consumes the
  token embedding via enorm + the h row via hnorm through eh_proj) - the
  draft loop is inherently lockstep, so "overlap CPU sampling with the next
  draft block" is dead on the dependency, not on effort. Naive unrefusal
  aborts in ggml_backend_meta_get_split_state (TOP_K/ARGSORT handle_per_row
  asserts src != AXIS_0; handle_pad split-axis assert; no (AXIS_0 weight,
  MIRRORED x) MUL_MAT case; the segment model cannot express a top-k whose
  output size differs from the input partition) - a full on-device sampling
  project needs either head K-axis re-shard (FP-sum-order change =
  acceptance-risk + model-wide layout) or a new top-k-per-shard op + meta
  state. IMPLEMENTED (all env-gated, unset = byte-identical):
  LLAMA_TP_BACKEND_SAMPLING=1 experimental unrefusal with abort WARNs (the
  served instrument that closes "where does meta actually fail");
  LLAMA_DRAFT_FAST_TOPK=1 minimal-sync arm - common_sampler_sample_topk
  heap-selects the top-k straight off the logits row (same
  make_heap/scan/sort_heap algorithm + comparator as the regular
  std::partial_sort path) and runs the same chain on the k survivors,
  deleting the 129272 x 16 B full-vocab candidate build + full-size
  partial_sort per draft step, with eligibility guards (no grammar/rbudget,
  no backend token, chain exactly [top-k(+dist)]) falling back to the regular
  path; LLAMA_SPEC_TIMELINE=1 + LLAMA_DECODE_TIMELINE=1 draft-step profiler
  (per step: decode_issue, build/reused/inputs/issue, outputs issue, drain,
  sample+batch, total) - the 37 ms attribution instrument for the served
  cell. HOST TESTS ALL PASS (docs/amd-port/tests/test_mtp_sampling_host.cpp,
  ~23 s): distinct-value trials (4 vocab sizes to 129272, ~222k trials)
  bit-exact arrays + identical drafted token + identical dist draw;
  k-boundary +-12 ULP bands bit-exact; exact-tie rows keep the logit
  multiset and the unique-max drafted token identical (tie ORDER among
  exactly-tied ids may differ - the one documented divergence, able to move
  only the seeded draw among tied candidates); full-scale top-1 and edge
  cases pass. Predecessor W6 MTP host suite re-run ALL PASS. Compile-clean
  on all three changed TUs (plain g++ and -DGGML_USE_HIP -DGGML_HIP mirror).
  DEFERRED SERVED PROTOCOL (priority order): (1) attribution cell -
  campaign config + LLAMA_SPEC_TIMELINE=1, ~50 rounds, split decode_issue vs
  drain vs CPU chain (rule: chain ~= 1-3 ms => the 37 ms is device/launch/AR,
  next desk is device-side); (2) LLAMA_DRAFT_FAST_TOPK=1 A/B, expect greedy
  byte-identical, accept 0.66667, +1-3% t/s; (3) LLAMA_TP_BACKEND_SAMPLING=1
  single-round abort capture; (4) optional cb_eval flag for LM-head vs block
  ms. REVISED ESTIMATE at 12-15 ms/step: round 121-140 ms for 3.0 tokens =
  21.4-24.8 t/s = +40-61% vs today's 14.43-15.38 and a 1.76-2.04x MTP
  multiplier vs OFF (12.17-12.18) - up from 1.15-1.26x. Receipt:
  results/MTP_overhead_2026-09-22.md.
- E-064 2026-09-22 REAL TP3 v1-FLAG CELL MEASURED (single rep, Chris:
  one run). Provenance per ARM IDENTITY LAW: binary from wt-v1flag @
  b4ad90c5a (= 8a4ebbcaa^, v1 partial-relocation semantics; boot log
  has ZERO full-isolation lines - no "leading the backend list", no
  isolation audit - and the v1 "MTP draft device: ROCm3" load line),
  lane 8081, 200k, t3flag boot line identical to the ladder arms.
  RESULT: decode 12.81 t/s, accept 0.66667, mean_len 3.0 (receipt
  /home/chris/v1flag_single.jsonl, server log /home/chris/
  v1flag_single_server.log). PLACEMENT: between in-split 14.73/14.67
  (-13%) and full isolation 7.58 (+69%) - v1 is decisively NOT the
  full-isolation collapse; the TP2 ordering (in-split >= v1-flag >>
  full iso) holds at TP3@200k, with v1's cost vs in-split larger at
  200k than the TP2@10k -2.4%. Full-battery second rep + 10k v1 cell
  queued to complete the table of record; MTP-OFF @ 200k (never run
  in this campaign) also queued x2. Table of record follows when the
  cells land.
- E-065 2026-09-22 Integration: amd/mtp-overhead merged (2ad756a59,
  ledger keep-both). DELIVERABLES (all env-gated, unset =
  byte-identical, host-tested): LLAMA_DRAFT_FAST_TOPK=1 (heap top-k
  off the logits row, deletes full-vocab candidate build + full-size
  partial_sort per draft step; bit-exact arrays + drafted token in
  ~222k trials, ties documented), LLAMA_TP_BACKEND_SAMPLING=1
  (instrumented unrefusal; meta aborts at TOP_K/ARGSORT split state
  prove on-device sampling needs head re-shard or a per-shard top-k op
  - a subsystem project), LLAMA_SPEC_TIMELINE=1 + LLAMA_DECODE_TIMELINE=1
  (step/batch/drain attribution). FACTS OF RECORD: the refusal is the
  defensive "-sm tensor" fallback from #23287, unimplemented not
  fundamental; output.weight is vocab-axis sharded and the 517 KB row
  materializes only via spliced host gets (no post-head allreduce in
  this fork); the draft loop is inherently lockstep (id(N) is a
  compute input of step N+1 via eh_proj) so CPU/draft overlap is dead
  on dependency. REVISED MODEL: 12-15 ms/step -> 21.4-24.8 t/s class
  (+40-61% vs today's 14.7, MTP multiplier 1.76-2.04x vs OFF);
  fast-topk recovers only the CPU slice; the ~20 ms/step drain/
  relaunch needs the TIMELINE attribution cell then device-side work.
  SERVED VALIDATION QUEUE: TIMELINE attribution cell, FAST_TOPK A/B
  (greedy byte-identical gate), backend-sampling abort capture.
- E-039 2026-09-22 TP2/TP3 desk: FULL DRAFT-DEVICE LADDER on the isolation
  build (binary v152, 7d3ab351a; 16 boots: TP2@10k A/B/C x2, TP3@10k A/B/C x2,
  TP3@200k B/C x2 + audit-capture boots). TP3@10k (receipt Result 7): OFF
  decode 12.17/12.18, in-split 15.38/15.58, draft-die 7.57/7.58 = -51% vs
  in-split; boot-ready savings: in-split costs ~765 MiB/die over OFF, flag
  saves ~224/die vs in-split; post-probe flag saves ~503/die vs in-split.
  TP3@200k (Result 8): in-split (config of record) 14.73/14.67 t/s decode,
  pp 113.69/113.43, 5/5 GREEN x2 - my binary reproduces the re-stamped
  baseline; draft-die arm decodes 7.58/7.58 = -48.5% at HALF the serving-die
  footprint (boot-ready C-vs-B saves ~602 MiB/die mean; die 3 carries 2.29 GiB
  boot / 3.17 GiB post-probe). AUDIT GATE 0.00 on the model split captured at
  200k (VERBOSE boot: "ROCm3 isolation audit: 263.52 MiB on ROCm3"), plus
  die-3 physical signature on every C boot. ACCEPTANCE 0.66667/3.00 in every
  arm of every topology; all quality guards PASS. CONSOLIDATED VERDICT: the
  full-isolation decode collapse (-48 to -51%) reproduces at every topology
  and context (TP2/10k 7.42/7.41, TP3/10k 7.57/7.58, TP3/200k 7.58/7.58) -
  root cause class: the draft's embedding-row + LM-head run on the
  1x-bandwidth draft die plus 2 host-staged hops per draft step; the v1-flag
  (partial relocation, head on meta) does NOT pay this. PROMOTION: E-035 full
  isolation is NOT a candidate for any latency-sensitive lane; its value is
  headroom (~602 MiB/die at TP3/200k, ~177 MiB/die at TP2/10k vs in-split).
  Voltage/clock sweep authorized next (Chris-directed, separate receipt).
- E-066 2026-09-22 REGRESSION CANDIDATE #3 + pipeline state. MTP-OFF
  @ 200k rep 1 (merged tree, stock clocks, lane 8081, boot
  tp3off-rep1 08:05:43): decode 10.47 t/s, prefill 100.52. Decode is
  -29.9% vs the MTP baseline (expected for OFF) but -14% vs the
  12.24 no-spec reference at 200k (baseline_tp3_200k.json config
  block) - the merged tree's OFF path is a THIRD regression candidate
  alongside the ub512 decode cost (-4%) and thermal slide (~-2%).
  Rep 2 running; if confirmed, the b512-vs-ub512 A/B (assigned to
  Gemini, hub #1362) plus an arm-level bisect (which merge moved OFF
  from ~12.2 to ~10.5) goes to the top of Priority Zero. STEP A
  merged this tick (eca7a14df; server-context conflict resolved
  keep-both-invariant); Gemini's duplicate STEP B cancelled (hub
  #1362). Cell chain: off1 done, off2 running, v1-flag 200k rep2 +
  10k queued, then the interleaved 2k A/B (in-split vs draft-device,
  merged build, probe_2k_decode.py, mode-line artifacts, per-die
  VRAM) - answers Chris's origin question: was the draft device ever
  actually faster at 2k on real code.
- E-067 2026-09-22 THERMAL PROGRAM FINAL FACTS + measurement reset
  (Chris: "reduce heat now, all runs invalid if too hot"). (1) POWER
  CAP: hard SKU ceiling 110 W/die (cannot raise); driver accepts
  90-110; at 90 W decode CHOKES (card3 775 MHz, 9.04 t/s); at 100 W
  busy dies slip (775-991 MHz under load); only ~105-110 W holds
  1200 MHz on all serving dies - the dies were power-limited at 110 W
  all along, stock and pinned. Cap reduction is NOT a heat lever at
  maintained performance on this SKU. (2) UNDERVOLT PIN: proven at
  10k only (morning sweep, 15.94 stable); at 200k ALL pinned cells
  timed out (8/8 arms, no GPU-hang markers in server logs, prime
  suspect manual-perf-level MCLK ramp inhibition at large-KV boots).
  Pin at 200k = open investigation, not a serving option. (3)
  MEASUREMENT RESET: every stock-clock number from ~07:30 onward is
  VOID as thermally confounded (the 14.73->12.81->10.47->8.78 slide
  across arms = die soak, idle-wait 0 back-to-back boots; stock boost
  power-throttles at the 110 W cap). The only clean stock numbers
  remain the 07:11-07:33 ladder era. (4) FINAL A/B CHAIN RUNNING
  (exec_0334c85d, lane 8081): STOCK clocks, cooldown-gated (no boot
  until hottest serving die < 60 C), interleaved 200k in-split vs
  v1-flag x2, then 2k in-split vs v1-flag pair - Chris's origin
  question (was the draft device ever faster at 2k). Mode lines +
  per-arm junc/draw/sclk telemetry on every arm. This chain is the
  table of record. Environment: stock restored (perf auto, OD stock,
  cap 110).
- E-068 2026-09-22 REBOOT RECOVERY + CLEAN RE-STAMP BEGINS. Post-crash
  forensics: 3x llama-server segfaults INSIDE libamdhip64.so.6.2
  (dmesg, same IP) beginning with the OD-table churn on live cards;
  sysfs GPU resets cleared the segfaults but left HIP enumeration
  dead (0 devices; amdgpu unload refused, 13 dependency refs). Chris
  rebooted; state after: 4/4 dies enumerate (8160 MiB free each),
  junctions 22-27 C, stock OD/cap. DISCRIMINATOR + FIRST CLEAN CELL:
  in-split @ 200k on the MERGED build (STEP A + mtp-overhead + T3 +
  admission all in), stock clocks: decode 15.10 t/s PASS (+1.05% vs
  14.94 baseline; 5-cell green). VERDICTS: (a) the morning crashes
  were DRIVER CORRUPTION, all of today's code merges EXONERATED;
  (b) 15.10 > both re-stamp waypoints (14.93, 15.57-era) on a cold
  cooldown-gated machine - the "decode decline" at 200k was largely
  thermal soak + driver decay, and the config of record holds ~15.1
  when measured clean; (c) origin decomposition stands: 17.8@2k era
  number remains context-scaling, not loss. FINAL A/B CHAIN running
  (exec_221aa1c5): 200k flag-r1/on-r2/flag-r2 + 2k on/flag pair,
  stock, cooldown-gated, mode lines + junc/draw/sclk telemetry per
  arm. Table of record lands from these receipts. OD/cap experiments
  are CLOSED on this hardware pending Chris's efficiency-trade
  decision (cap floor ~95-100 W chokes clocks; 110 W = SKU max).
- E-069 2026-09-22 PARITY VERDICT - THE DAY'S CONTAMINATED CONCLUSIONS
  REVISED (post-reboot clean receipts, cooldown-gated, mode lines
  verified "partial (v1)" on every flag boot). TP3 @ 200k, decode
  guard, accept 0.66667/3.00 EVERYWHERE: in-split 15.10 / 15.23
  (walls 88.1/88.4 s) vs v1-flag 15.18 / 15.08 (walls 89.2/89.2 s) -
  PARITY within 0.7%. The morning's "-13% v1-flag" (12.81) and the
  ladder-era "-18%" (12.24, wall 112 s) were the DEGRADING DRIVER +
  thermal soak, not the mode: forensics = the flag1 jsonl holds both
  cells (12.24 pre-reboot vs 15.18 post) on the same arm. TP3 @ 2k:
  in-split 22.25 vs v1-flag 21.19 (-4.8%) - near-parity, and the
  tree is +19% faster than the origin-era 17.8@2k draft-device code.
  ANSWERS OF RECORD: (1) Chris's hypothesis REFUTED - the draft-device
  path is healthy at 2k AND 200k; (2) full isolation remains the only
  catastrophic mode (-49%, 7.58 x3, morning receipts); (3) the draft
  device costs ~nothing at any depth measured today on healthy
  silicon, so its VRAM dividend (v1 = partial relocation) is nearly
  free - the operating-point question between in-split and v1-flag
  reopens on VRAM grounds alone (in-split post-probe free was 17-114
  MiB/die at 200k; v1's 200k VRAM capture queued next boot);
  (4) OFF@200k clean cells in flight (fin_200k_off1/2) to complete
  the 4-arm table - the morning 10.47/8.78 OFF numbers are VOID
  (driver decay + soak). ENVIRONMENT LAW: after any OD/cap churn or
  unexplained crash, dmesg first (libamdhip64 IP signature), and no
  number banked post-incident without a clean-state discriminator.
- E-070 2026-09-22 TABLE OF RECORD - TP3 four-arm, clean silicon,
  post-reboot, cooldown-gated, mode-verified (receipts fin_*.jsonl +
  fin_*.log; walls 80-89 s class on every cell). MTP-OFF @ 200k x2:
  12.19 / 12.16 t/s = the no-spec reference (12.24) EXACTLY - OFF
  decode has NO context-scaling penalty (12.2 at 10k AND 200k; decode
  is weight-bound, q4_0 KV reads are small) and the morning VOID
  numbers (10.47/8.78) are confirmed as driver decay, closed.
  DERIVED: MTP value at 200k = +24% over OFF (15.1-15.2 vs 12.17),
  matching +26% at 10k; v1-flag = PARITY with in-split at 200k
  (15.08-15.18 vs 15.10-15.23) and -4.8% at 2k, so on healthy
  silicon the draft device's cost is ~zero and its VRAM dividend is
  nearly free - the E-056 "v1 rejected" verdict is SUPERSEDED (it was
  decided on mislabeled full-isolation + contaminated-driver data);
  the in-split vs v1 operating-point decision now hinges on the v1
  200k VRAM capture (in-split post-probe free was only 17-114
  MiB/die). Path to the 36 t/s goal unchanged: MTP-overhead desk's
  +40-61% modeled recovery (21.4-24.8 t/s class) via the ~20 ms/step
  drain/relaunch attack, then kernel work.
- E-071 2026-09-22 GAINS OFFENSIVE DISPATCHED (Chris: "launch your
  agents and get to work on the mtp and other items to make those
  gains"). Three lanes: (1) MTP-GAINS SERVED DESK (lane 8083, main
  tree, docs-only commits): the E-065 deferred protocol - TIMELINE
  attribution cell (CPU chain vs drain/relaunch verdict),
  FAST_TOPK interleaved 2x2 A/B with timeline-drop engagement proof,
  Arm G backend-sampling abort capture, and the v1 VRAM capture at
  200k (completes the E-070 table's VRAM column - decides
  in-split-vs-v1 serving on VRAM grounds). (2) DRAFT-STEP DRAIN
  ATTACK DESK (wt-mtp-drain on amd/mtp-drain, zero-GPU): packed
  single-copy head get (LLAMA_DRAFT_PACKED_GET), synchronize audit
  (LLAMA_DRAFT_LIGHT_SYNC), drain-budget table, step-batched design
  doc if needed - targeting the 21-25 t/s class. (3) GEMINI (hub
  #1363): GO for the ub512-vs-b512 decode-recovery A/B on lane 8080
  (Priority Zero refund test; clean reference now 15.10/15.23).
  Environment: post-reboot clean state; stock clocks; OD/cap
  experiments closed.
- E-072 2026-09-22 DRAIN-ATTACK DESK COMPLETE (wt-mtp-drain on
  amd/mtp-drain, zero-GPU: audit, implementation, host tests, compile).
  SYNCHRONIZE AUDIT: one draft step executes 5-6 full scheduler sweeps
  (the sample drain + backend-token guard + the 3 set_logits sampled-getter
  probes + the h-row getter - every C-API output getter synchronizes), and
  only the sample drain is a real dependency: the logits row is spliced from
  3 dies, so waiting all 3 die streams is irreducible. EXTRACT AUDIT: the
  "517 KB spliced get" is already 3 linear async DMAs into pinned host
  memory at final offsets - the full row is never on one die, so "one
  contiguous transfer" is physically 3 and the splice was already single-
  copy; the real per-step fat is the redundant syncs, not the DMA. Offset
  note: the meta AXIS splice's get_2d degrades to a linear copy per shard at
  n_copies = 1 (the per-draft-step case). IMPLEMENTED (env-gated, unset =
  byte-identical): (1) LLAMA_DRAFT_PACKED_GET=1 - decode() skips its logits +
  h_nextn extraction for the draft ctx; one llama_fetch_nextn_outputs call
  per step (staged API, src/llama-ext.h) issues the same two meta gets and
  hands the raw row pointers to new no-sync sampler entries
  (common_sampler_sample_row / common_sampler_sample_topk_row, common/
  sampling.{h,cpp}); PACKED alone keeps the full-vocab candidate build
  (arrays byte-identical to baseline), with FAST_TOPK it keeps the
  heap-select; (2) LLAMA_DRAFT_LIGHT_SYNC=1 - llama_wait_outputs drains only
  the output-owning backends; per-step blocking syncs drop 5-6 -> 1 (draft
  ctx perf counters stop closing in this mode, documented). Guards: packed
  path auto-off with backend sampling attached or shared draft ctx (WARN);
  fetch failure stops the draft round rather than read stale rows. HOST
  TESTS ALL PASS (docs/amd-port/tests/test_packed_get_host.cpp: meta splice
  mirror byte-exact on uneven 3-way shards at the real 129272 vocab x 1-4
  output rows plus 2/4/5-way stress, row-pointer resolution, mirrored h-row
  get, sample_row parity BIT-EXACT vs the regular path across 64/1k/32k/
  129272 vocabs, tie-order caveat re-measured 2/200 at 129272; ASAN/UBSAN
  clean); test_mtp_sampling_host and test_w6_mtp_device_host re-run ALL
  PASS. Compile-clean gfx900 (cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900:
  libllama, common, llama-cli all built, zero warnings). DRAIN BUDGET
  (results/DRAIN_BUDGET_2026-09-22.md): per draft step = 9 main subgraph
  replays (nextn block cut at its 2 PARTIAL nodes into 3 meta subgraphs x 3
  dies) + 2 allreduce boundaries x (4 inter-die host-staged peer copies + 3
  ADD replays; n=3 butterfly fallback) + 4 output DMAs + 1 true drain + 4-5
  idle sweeps + 4-6 blocking H2D input sets; real device work ~0.2-0.5
  ms/die at 200k. VERDICT: (a)+(b) delete ~0.1-0.5 ms/step of host slice -
  the 21.4-24.8 t/s class is NOT reachable by host call-site changes; it
  requires the device-side serial chain attacked (on-device sampling
  subsystem, launch/replay latency on the 15 device issues, AR transport).
  STEP-BATCHED DESIGN: two draft steps per replay is blocked by the
  id(N)->eh_proj dependency unless the top-2 candidates are branched
  speculatively with branched draft KV (2x slots + commit-on-choice) -
  verdict NOT clean, do not implement; doc lists the follow-up options
  (async input sets, splice-plan cache, HIP-graph replay health check).
  SERVED ARMS for the measurement desk: LLAMA_DRAFT_PACKED_GET=1 alone
  (greedy outputs byte-identical, accept 0.66667 gate; expect sample+batch
  -0.1..-0.5 ms/step in SPEC_TIMELINE, t/s ~0-+2%), then +LLAMA_DRAFT_
  LIGHT_SYNC=1 (same outputs; exactly 1 wait/step), each with and without
  LLAMA_DRAFT_FAST_TOPK=1 for the full 2x2 - all four arms should read
  accept 0.66667/3.00; any accept move is a bug, not noise.
- E-073 2026-09-22 TIMELINE ATTRIBUTION VERDICT (E-065 deferred protocol,
  gate 1; T1 boot, lane 8083, 5x 7857-probe/n_predict-128 requests = 210
  draft rounds, accept 0.66667 everywhere; receipt MTPgains_2026-09-22.md).
  E-065 RULE FIRES: sample+batch 1.13 ms/step (<< 5 ms) => CPU chain minor,
  device-side owns it. STRONGER: the 37 ms/draft-step model was
  MISATTRIBUTED - the whole 3-step draft loop is 11.05 ms/round (draft
  graph 2.3-2.7 ms/step device + 1.1 ms/step CPU; steps = 3 every round).
  The round (~190-195 ms wall at 14.9 t/s) is owned by the TARGET VERIFY
  decode: [decode-timeline] shows the 4-token verify ubatch issue = mean
  85.3 / MED 138 ms BLOCKING (build 0.03, inputs 0.03; synchronize drain
  0.04 ms => the wait sits inside graph_compute), plus ~40-55 ms host
  sampling/batch per round. Cross-check vs E-072's drain budget: agrees -
  host call-site changes cannot reach the 21.4-24.8 t/s class; the lever is
  the verify-batch device serial chain + launch/replay latency. DRAIN-ATTACK
  RETARGET (served evidence): verify-batch device critical path (3-die
  pipeline + allreduce latency at 4 tokens) + the ~40-55 ms host slice; the
  draft loop is already cheap (11 ms/round). 21.4-24.8 t/s needs round
  121-140 ms. LOG LAW NOTE: library INFO maps to LOG_LEVEL_TRACE=4
  (common/log.cpp common_get_verbosity) - [decode-timeline] lines need
  -lv 4; default-verbosity boots only show [spec-timeline].
- E-074 2026-09-22 FAST_TOPK INTERLEAVED 2x2 (E-065 gate 2; F1 unset / F2
  set / F3 unset / F4 set, all + timelines + -lv 4; decode-only guard +
  determinism per arm; cooldown-gated boots, stock clocks). ENGAGEMENT
  PROVEN: sample+batch 1.131 -> 0.966 ms/step (-15%), round total 10.9 ->
  10.27 ms in both set reps. ACCEPT 0.66667 / mean 3.0 / draft 126-84 in
  every cell. DETERMINISM: 2-run greedy byte-identical per arm AND identical
  sha256 across all four arms (4beb1ba25219ee9b) - byte-exact vs the
  regular path at temp 0. t/s: unset 14.93/14.94 vs set 14.91/15.02 = +0.2%
  mean, inside noise - the removed ~0.66 ms is 0.35% of an ~190 ms round,
  so the modeled +1-3% overestimated the CPU share. VERDICT: fast-topk is
  free and byte-exact but NOT a decode t/s lever at 200k today; its value
  scales only if the round shrinks (or at smaller contexts).
- E-075 2026-09-22 ARM G: BACKEND-SAMPLING ABORT CAPTURE (E-065 gate 3;
  G1 boot, LLAMA_TP_BACKEND_SAMPLING=1 + timelines, one decode request;
  NOT a perf arm). Boot WARNs captured (set_sampler experimental enabled +
  "sampler ops on vocab-sharded logits are not modeled ...; aborts are
  likely"). First draft-round process() aborts:
  GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN) at
  ggml/src/ggml-backend-meta.cpp:814 via ggml_abort in
  ggml_backend_meta_get_split_state <- draft_mtp::process <- server
  update_slots. The first host GET of vocab-sharded sampler tensors walks
  into the un-modeled split state - empirically closes the MTP_overhead
  desk's "where does meta fail" question: on-device sampling needs head
  re-shard or a per-shard top-k op (subsystem project). Controlled
  GGML_ABORT: dmesg clean of libamdhip64 segfaults at the event, 4/4 dies
  enumerated on the next boot.
- E-076 2026-09-22 v1 VRAM TABLE AT 200k (E-065 gate 4; completes the
  E-070 VRAM column; V1 boot: campaign line + --spec-mtp-device ROCm3 with
  HIP_VISIBLE_DEVICES=0,1,2,3 - the draft die must be HIP-visible, else
  -devm cannot resolve - STRICT unset, mode line "draft-device mode:
  partial (v1)" verified at V1 log line 7). Probe: 14.935 t/s, accept
  0.66667 = parity with in-split (matches E-069/E-070). VRAM (MiB, dies
  total 8176; HIP0/1/2 = card1/card3/card0, HIP3 = card4; jsons
  mtpgain_vram_{insplt_T1,v1_V1}_*): in-split boot 7809.7/7591.8/7643.9/18.0
  vs v1 boot 7074.8/7052.9/7105.0/1440.8; post-probe in-split
  8166.5/8069.9/8122.0/18.0 (FREE 9.5/106.1/54.0 - OOM-adjacent) vs v1
  7675.4/7605.6/7665.5/2320.8 (FREE 500.6/570.4/510.5, draft die 5855.2
  free). VERDICT: v1 saves ~456-491 MiB on EVERY serving die post-probe at
  zero decode cost - the in-split-vs-v1 operating point CLOSES in favor of
  v1 on VRAM grounds (E-069 item 3 resolved): v1 is the safer 200k serving
  default. HANDOFF: E-072's served arms (PACKED_GET / LIGHT_SYNC x
  FAST_TOPK 2x2) require a rebuild - the merged code (3964a90b1, 13:44)
  postdates the current build-hip binary (libllama.so.0.0.187, 08:13) - so
  they are queued for the next measurement window, not run in this one
  (ladder was scope-locked; no rebuild in the measurement grant). Campaign
  state: lane 8083 clean, lock released, stock clocks throughout, no
  driver events.

- E-077 2026-09-22 OPERATING POINT CLOSED + ATTRIBUTION CORRECTION
  (MTPgains receipt, E-073..E-076 integrated; coordinator entries).
  (1) VRAM VERDICT: at 200k, v1 (partial, mode line verified) frees
  ~456-491 MiB per serving die at ZERO decode cost (probe 14.935 =
  parity; post-probe free in-split 9.5/106.1/54.0 MiB = OOM-adjacent
  vs v1 500.6/570.4/510.5; draft die 2.32 GiB used). CANONICAL LAUNCH
  SCRIPT /home/chris/launch_tp3_200k.sh FLIPPED to v1 default
  (--spec-mtp-device ROCm3, HIP_VISIBLE_DEVICES 0,1,2,3); full
  isolation stays opt-in behind LLAMA_SPEC_MTP_STRICT=1. Baseline
  config block to be re-stamped on the v1 config at the next green
  battery. (2) ATTRIBUTION CORRECTION (210 rounds): the 37 ms/step
  draft-loop model was MISATTRIBUTED - the draft loop costs 11.05
  ms/round total (2.3-2.7 ms/step device, 1.1 ms/step CPU). The round
  (~190-195 ms at 14.9 t/s) is owned by the TARGET VERIFY decode: the
  4-token verify ubatch issues blocking at mean 85 / median 138 ms
  inside graph_compute (sync drain only 0.04 ms - the wait is the
  device serial chain: 9 subgraph replays x 3 dies + n=3 host-staged
  butterfly allreduce boundaries), plus ~40-55 ms/round of host
  sampling/batch. FAST_TOPK verdict: engaged (-15% sample time,
  round 10.9 -> 10.27 ms), byte-exact (sha 4beb1ba25219ee9b across
  all arms), +0.2% t/s = noise at 200k - keep enabled, not a lever.
  Arm G: on-device sampling blocker pinned at ggml-backend-meta.cpp:814
  (SPLIT_AXIS_UNKNOWN assert) - head re-shard or per-shard top-k op
  required, as designed. (3) RETARGET: the gains road is now (a) the
  40-55 ms/round host slice (plain C++ - profile then optimize),
  (b) the verify-ubatch device chain (allreduce transport + replay
  count - the device-side subsystem), (c) drain arms (PACKED_GET/
  LIGHT_SYNC) validation on the rebuilt binary. Verify-round attack
  desk dispatched; timeline logs (T1/F1, -lv 4) on disk for offline
  profiling.

- E-078 2026-09-22 VERIFY-ROUND DESK COMPLETE (wt-verify-round on
  amd/verify-round, campaign HEAD f097c9f6e; zero-GPU: offline profiling of
  the T1/F1 timeline logs, code, host tests, gfx900 compile; receipt
  VERIFY_ROUND_2026-09-22.md + scripts/profile_verify_round.py).
  (1) HOST-SLICE PROFILE - the E-077 "~40-55 ms/round host" partition is
  WRONG: segmenting F1 by wall-clock timestamps (verify = n_tokens 4 with
  n_outputs > 0; catch-up = the second n_tokens 4 decode, n_outputs 0) gives
  round med 190.0 = verify issue 163.2 (85.9%) + catch-up issue 11.7 (6.1%)
  + draft total 11.0 (7.55 device + 3.44 host) + D2H drain 1.9 + HOST only
  ~5.5-8 ms (target sampling ~0.7, batch builds ~0.7, residual ~0.7). The
  40-55 was an artifact of pooling verify+catch-up lines (E-073/E-077 pooled
  n=170, med 138.1 understated verify) and of counting the catch-up DEVICE
  decode as host. T1 (210 rounds) cross-checks. Round is 91-97% device
  blocking. Candidate audit: no redundant logits copies exist (one tensor
  get_async per extract; get_logits_ith is a pointer); target candidate
  builds are ~0.66 ms/round and fast-topk does NOT transfer (request-shaped
  chain); the E-072 redundant-sweep finding IS alive on the target accept
  path (~24 syncs/round) - FIXED (2). (2) IMPLEMENTED (env-gated, unset =
  byte-identical): LLAMA_VERIFY_ROW_SAMPLING=1 - one llama_wait_outputs +
  new llama_peek_logits_rows (src/llama-ext.h) hand the 4 verify rows as
  raw pointers to common_sampler_sample_row (E-072 API) with the same
  break-on-mismatch loop (common_sampler_sample_and_accept_n_rows,
  common/sampling.cpp; server gate + regular-path fallback in post_decode);
  grammar/budget/backend-sampler guards fall back; LIGHT_SYNC-style perf
  window caveat documented. GGML_PINNED_DEV_COPY=1 - the no-peer-copy
  butterfly peer copies run 10 boundaries x 4 copies = 40x
  [sync+sync+malloc+unpinned D2H+H2D+free] per round (5 decodes x 2
  boundaries; ground truth in ggml_backend_tensor_copy_async); new
  ggml_backend_dev_copy_staging (ggml-backend.cpp) replaces the per-call
  malloc/free with a grow-only cached PINNED buffer (src device host buft),
  get/set sequence unchanged = bytes identical by construction; falls back
  to the historical path when unset / no host buft / alloc fail. HOST TESTS
  ALL PASS: new test_verify_row_sampling_host (accepted sequences + break
  positions bit-exact, greedy + seeded top-k/dist, 64..129272 vocab,
  ~1900 trials, ties, guards; ASAN/UBSAN clean) and test_pinned_staging_host
  (reuse, geometric growth, byte-exact staged vs malloc, fallbacks;
  ASAN/UBSAN clean); test_mtp_sampling_host, test_packed_get_host,
  test_w6_mtp_device_host, test_server_exposures_host re-run ALL PASS.
  gfx900 compile clean, ZERO warnings (llama + llama-common + llama-cli).
  (3) DESIGNED, no code: transport - die-to-die does not exist (PCIe through
  host regardless); the structural lever is comm_allreduce (meta already
  probes ggml_backend_comm_init / ggml_backend_comm_allreduce_tensor; HIP
  registers none; RCCL integration = subsystem project, fp sum order is the
  acceptance-critical question), the host-side lever is event-ordered async
  pinned-ring staging (sync+sync+blocking copy -> get_async + event +
  dst-wait + set_async, >= 4 slots/pair; est -2..-6 ms/round); copy
  consolidation is unavailable (4 distinct (src,dst) pairs per boundary).
  (4) REPLAY COUNT VERDICT - NOT CLEAN, do not implement: the meta splitter
  cuts exactly at PARTIAL-axis nodes and each reduction feeds replicated
  consumers (norm/activation between attn.out and mlp.down); boundary count
  is architectural, the two reductions are sequentially dependent. Device
  desk note: the catch-up decode re-decodes ALL verify tokens (incl. ~1.3
  rejected/round, KV seq_rm'd later) at 11.7 ms med vs 2.5 ms per draft
  step - an accepted-prefix-only catch-up would cut real device work but
  reorders KV rollback, not mechanical. (5) SERVED ARMS (need rebuild;
  timelines + -lv 4; accept 0.66667/3.00 gate in every arm, any move is a
  bug): R1 baseline reproduces; R2 +LLAMA_VERIFY_ROW_SAMPLING (greedy sha
  identical, ~22 fewer drain lines/round, G3 -0.3..-0.6 ms, +0.2..+0.5%);
  R3 +GGML_PINNED_DEV_COPY (sha identical, verify med -2..-6 ms,
  +1..+3% class); R4 = R2+R3 additive (-3..-7 ms/round).
- E-079 2026-09-22 SERVING DEFAULT = IN-SPLIT (Chris: "we will mainly
  serve split for now but should be able to turn it on via a flag").
  launch_tp3_200k.sh reverted to in-split; v1 documented as the flag
  path (--spec-mtp-device ROCm3 + HIP 0,1,2,3). v1's warm full-battery
  cell (disc3) lands for the record but no longer gates anything.
  MEASUREMENT PAUSE: no more repeated A/B batteries - the remaining
  die windows are for NEW optimization validation only. OPTIMIZATION
  OFFENSIVE DISPATCHED (Chris: "work on some optimizations instead of
  running the same crap over and over"): (1) MMVQ KERNEL DESK
  (wt-mmvq-kernel on amd/mmvq-kernel): the 77%-of-decode-time kernel,
  gfx900 iq3_s vecdot - LUT residency, dot2-instruction probe, tiling,
  q8_1 layout; ninfer-kernel-opt protocol, oracle bit-exact gate,
  bench harness exists; single-die short benches on card4 only.
  (2) VERIFY-TRANSPORT DESK (wt-verify-transport on
  amd/verify-transport): event-ordered async pinned-ring staging for
  the 40 boundary copies/round (est -2..-6 ms/round), copy-engine
  overlap with a dependency diagram, RCCL feasibility spike (gfx900
  x4 no-P2P; fp-sum-order acceptance experiment designed, not run).
  (3) disc3 (v1 warm battery) result = record only.
- E-080 2026-09-22 disc3 RECORD (v1 full battery, cooled dies, single
  boot after idle): 5/5 GREEN - decode 15.20 (+1.72% vs 14.94),
  prefill 116.72, canary/determinism/needle clean. The 13.27 re-stamp
  FAIL is thereby attributed: THERMAL SOAK from the fifth consecutive
  boot, not a v1 cost. Final mode ranking on clean silicon: v1
  15.08-15.20 across every regime >= in-split 14.87-15.23 (parity to
  slightly ahead) >> full isolation 7.58. The in-split serving
  default stands per Chris's call; the v1 flag path is proven healthy
  at full battery and carries the ~490 MiB/die dividend whenever
  serving-die headroom is needed.
- E-081 2026-09-22 COOLING UPGRADE + COMBINED VALIDATION DESK
  DISPATCHED (Chris: "we increased cooling on gpus"). Idle junctions
  now 26-28 C at 3-4 W. The two optimization desks (MMVQ kernel,
  verify-transport) continue unchanged - their benches and arms
  benefit directly. NEW: combined validation window on lane 8083
  (drift-controlled: ref re-run between arms) for every env-gated fix
  merged today - FAST_TOPK+PACKED_GET, +LIGHT_SYNC,
  VERIFY_ROW_SAMPLING, PINNED_DEV_COPY, and the ALL composite - with
  engagement signatures per arm (1-wait/step, -22 drains/round,
  verify issue -2..-6 ms); then the T3 grouped-mmvq device validation
  (GGML_CUDA_MMVQ_GROUP=1 + DEBUG: formation lines or named gate
  declines - either is the deliverable) and the E-054 admission
  validation (probe_kv_admission.py at 10k class). With the cooling
  headroom, +0.5-3% class deltas are finally resolvable in a single
  window instead of being eaten by soak.
- E-082 2026-09-22 COORDINATOR LOCK VIOLATION (own-goal, full
  transparency per campaign law). While running Chris's direct TP2
  10k flag request (TP2/10k/v1-flag decode guard = 14.41 t/s, PASS,
  receipts /home/chris/tp2flag_dummy.jsonl - reproduces the
  historical 14.43 on the merged tree) the coordinator's launch
  command (a) killed ALL llama-server processes and (b) removed and
  replaced /tmp/campaign_gpu_boot.lock - killing the combined
  validation desk's freshly-booted ref0 server one second after
  start and destroying its held lock. This is the exact incident
  class E-039/#1355 banned, committed by the coordinator. REMEDIATION
  executed immediately: TP2 server torn down, lock removed, dies
  drained and returned to the desk, desk notified (void ref0,
  re-run first, rest of ladder clean). LAW AMENDMENT: coordinator
  direct-request boots MUST go through the same gate as desks -
  check-and-hold the lock, targeted kill by PID only (never a global
  llama-server sweep), and if a lock is held, the request WAITS.
- E-083 2026-09-22 SPEED OFFENSIVE WIDENED (Chris: "make progress on
  speed, heat shouldnt be an issue anymore"). Two more zero-GPU desks
  dispatched so every known lever has an owner: (1) CATCHUP-ROLLBACK
  DESK (wt-catchup on amd/catchup-rollback): the 11.66 ms/round
  catch-up re-decodes rejected drafts - map the accept/reject flow,
  feasibility verdict on accepted-prefix catch-up (the KV-rollback
  ordering concern from E-078), implement behind
  LLAMA_DRAFT_PREFIX_CATCHUP=1 if clean, blocker doc if not.
  (2) ASYNC-INPUT DESK (wt-async-input on amd/async-input): the 4-6
  blocking H2D input sets per step - map the critical path, pinned
  ring + event-gated consumption behind LLAMA_ASYNC_INPUT=1, honest
  latency-floor sizing. With the running MMVQ-kernel and
  verify-transport desks, all five identified levers are owned:
  kernel math (77% share), boundary transport, host syncs (merged),
  catch-up waste, input staging. Validation: one combined served
  window for all winners once the desks land (no per-desk batteries).

- E-084 2026-09-22 ASYNC-INPUT DESK COMPLETE (wt-async-input on
  amd/async-input, campaign HEAD c20b3f547; zero-GPU: set_inputs path audit,
  implementation, host tests, gfx900 compile; receipt
  ASYNC_INPUT_2026-09-22.md). (1) MAP - every per-round blocking H2D set
  (blocking = ggml_backend_tensor_set on a device tensor; the HIP buffer set
  is cudaMemcpyAsync + cudaStreamSynchronize + cudaSetDevice per tensor, and
  behind the meta backend MIRRORED inputs fan out to one blocking round trip
  PER DIE): per MTP round = draft steps 3 x (tokens 4 B + h row 16-32 KB +
  pos) + verify (tokens 16 B + pos) + catch-up (tokens + h rows 64-128 KB +
  pos) = 13 blocking tensor sets = 39 die-level round trips (E-072's 4-6/step
  counted tensors, not die fan-out). out_ids/kq_mask/k-v idxs/k_shift/s_copy
  are HOST-buffer writes, not DMA sets (sched moves them stream-ordered inside
  graph_compute) - audited, untouched. Ground truth (F1 decode-timeline):
  verify/catch-up inputs mean 0.027 ms/ubatch (max 0.051), draft-step inputs
  mean 0.010 ms -> 0.08-0.15 ms/round total; E-072's ~50-300 us/step upper
  range was pessimistic. (2) IMPLEMENTED (env-gated, unset = byte-identical):
  LLAMA_ASYNC_INPUT=1 - llama_input_tensor_set (llama-graph.cpp) routes the
  per-ubatch input setters through a pinned ring (8 grow-only slots, device
  host-buft, GGML_PINNED_DEV_COPY house pattern): memcpy into slot +
  ggml_backend_tensor_set_async on the sched-resolved owning backend (meta
  MIRRORED fan-out posts per-die DMAs and returns). Slot reuse gated by a
  per-slot backend event when the device supports events, else by a
  synchronize of the issuing backend (meta drains all dies; quiescent at that
  point in the decode loop, so the gate costs ~0). Device tensor addresses
  unchanged -> graph reuse/capture-replay unaffected; bytes identical by
  construction. Fallbacks: no sched (K-shift/opt call sites), host tensors,
  no host buft, alloc fail, offset != 0. set_inputs grew a scheduler param.
  HOST TESTS: new test_async_input_host (mirror control flow vs fake backend
  layer): byte-exact landings + ring-overwrite safety over the real round
  shape with two backends and varied drain timing, no-event device gate,
  event re-creation on device change, growth, all fallbacks - ALL PASS,
  ASAN/UBSAN clean; test_packed_get_host, test_verify_row_sampling_host,
  test_pinned_staging_host, test_mtp_sampling_host re-run ALL PASS. gfx900
  compile clean, ZERO warnings (llama, llama-common, llama-cli). (3) HONEST
  SIZING: the lever deletes host round trips only - the posted DMA still must
  land before the consuming graph reads the tensor. Decode ceiling = the
  set_inputs wall => ~0.05-0.10 ms/round recoverable = < 0.1% of the ~190 ms
  round, t/s +0.0-0.1%; floor = host enqueue ~2-5 us/copy + quiescent gate.
  Secondary: draft-ctx prefill h-row sets (8-16 MB at 512, pageable blocking
  today) may recover ~0.3-0.6 ms/chunk, rig-measurable only. VERDICT: S1 is
  a hygiene win and a NEGATIVE for the speed offensive - it is two orders
  below the 163 ms verify device chain and cannot move 15 -> 21+ t/s; the
  round remains owned by the verify device serial chain + boundary transport.
  (4) SERVED ARMS (fold into the combined validation window, no dedicated
  battery): A1 unset reproduces; A2 +LLAMA_ASYNC_INPUT=1 - identical greedy
  sha, accept 0.66667/3.00 gate, engagement = one INFO line at first decode +
  [decode-timeline] inputs columns collapse to ~0.00-0.01 ms/ubatch; round
  -0.0..-0.15 ms (inside noise). Composable with all merged arms.
- E-085 2026-09-22 CATCHUP-ROLLBACK DESK COMPLETE (wt-catchup on
  amd/catchup-rollback, rebased on dd9b275f6; zero-GPU: code trace, offline
  timeline logs, host parity test, gfx900 compile; receipt
  CATCHUP_ROLLBACK_2026-09-22.md). (1) FLOW MAP: per round the draft_mtp
  catch-up (process(), ctx_dft, 4 rows, n_outputs 0, med 11.66 ms = 6.1% of
  the round) re-decodes the full verify batch [sampled @ L, d1..d3]; rows
  0..k survive post_decode's seq_rm(pos_next = L+k+1) as prompt cells, rows
  k+1..3 are thrown away unread - at accept 0.66667/mean_len 3.00
  (E[k] = 2.0 measured 84/126) that is 1.0 wasted row/round (25%), worst in
  the k=0 tail (P = 1/3 at iid, 3 of 4 rows wasted). Also redundant: the
  ckpt block rm's the draft-phase cells (incl. the row-0 cell the draft
  step 0 already decoded identically) before the catch-up rebuilds them.
  (2) VERDICT - IMPLEMENTABLE-CLEAN: the E-078 "reorders KV rollback"
  concern dissolves because the flush runs AFTER the accept loop and
  decodes only rows 0..k - the rejected cells are never written, so
  seq_rm finds nothing beyond pos_next; both caches end the round
  accepted-only, identical to baseline. Audited: ctx_tgt untouched,
  pending_h/verify_h unchanged (staged batch keeps pre-accept embd
  copies), ckpt-restore path decodes full staged rows after the restore
  (dead cells causally masked, rewritten by the re-verify round), mtmd
  path self-heals unflushed stagings at the next process()/draft() entry,
  mem_shared ignores the flag, chain_heads mirrors the per-head rm+offset
  loop. Numerics caveat: the ubatch width changes (4 -> k+1), so last-ulp
  KV differences can flip a near-tie draft - logic-identical, NOT
  bit-identical on device; served gate is accept/mean-len within noise,
  not identical sha. (3) IMPLEMENTED (env-gated, unset = byte-identical):
  LLAMA_DRAFT_PREFIX_CATCHUP=1 - process() stages the byte-identical
  catch-up batch, accept() records rows 0..k, new catchup_decode() (+
  base-class hook catchup() + common_speculative_catchup, common/
  speculative.{h,cpp}) decodes the prefix rows (full rows when no accept
  decision), server flush call at the end of post_decode; engagement line
  [spec-timeline] catchup: rows = N; catch-up decode n_tokens now varies
  1..4. HOST TEST ALL PASS (new test_prefix_catchup_host, standalone):
  identical drafted-token sequences + draft-phase attended views across
  k = 0..3, catch-up rows = exact prefix of baseline rows, post-round
  cells below pos_next identical, prefill/1-token rounds full rows,
  mtmd self-heal parity, ckpt-restore + re-verify parity, row reduction
  pinned (48 -> 28 rows on the 12-round script); ASAN/UBSAN clean.
  gfx900 compile clean, ZERO warnings (llama, llama-common, llama-server).
  (4) SIZING: rows 4 -> 3.0 avg, catch-up med 11.66 -> ~8-9 ms, wall
  -1.5..-3 ms/round = +0.8..+1.6% t/s class; NEXT LEVER documented not
  implemented: keep the draft-phase row-0 cell (ckpt rm bound L -> L+1 +
  draft-ran guard) to drop the catch-up to E[k] = 2.0 rows (another ~25%).
  (5) SERVED ARM R5 (fold into the combined window): campaign line +
  LLAMA_DRAFT_PREFIX_CATCHUP=1 + timelines + -lv 4; accept 0.66667/3.00
  within noise, catch-up issue med -> ~8-9 ms, wall -1.5..-3 ms,
  +0.8..+1.6% class; greedy sha may differ at tie level (caveat above).
- E-086 2026-09-22 COMBO WINDOW COMPLETE (combined validation desk, lane
  8083, cooled dies, ref-interleaved ladder ref0,A,B,ref1,C,D,ref2,ALL,ref3;
  receipt ComboVal_20260922_175500.md). Incident: first ref0 boot killed at
  launch by the coordinator's TP2 boot (killed ALL llama-server processes
  and replaced the campaign boot lock - E-082); voided, dmesg clean, ladder
  re-run. GATES: accept 0.66667 / mean_len 3.00 / draft 84-126 in every
  cell of every arm (zero moves - no bugs); greedy sha 4beb1ba25219ee9b
  identical across all nine boots (byte-exact vs the regular path, same as
  E-074). t/s (2-ref drift control, ref band 14.79-14.93, spread 0.94%):
  ref0 14.93, A 15.00 (+0.47%), B 14.88 (-0.34%), ref1 14.83, C 14.89
  (+0.40% vs ref1), D 14.74 (-0.61% vs ref1), ref2 14.92, ALL 14.85
  (-0.47% vs ref2), ref3 14.79 - every arm inside the ref band; even with
  cooled dies sub-1% boot noise is the floor at these effect sizes.
  ENGAGEMENT PROVEN per arm: A sample+batch timer 1.146 -> 0.055 ms/step
  (mostly re-attribution into the packed fetch; honest host deletion =
  draft total med 10.93 -> 10.26 ms/round, -0.67) + drains/round 232 ->
  214; B wait_outputs/draft-step = exactly 1.000 (126/126, 3.00/round,
  0.90 ms = the real drain); C drains/round 232.0 -> 215.0 (-17.0; brief
  estimated ~22) + one 0.003 ms light wait/round, wall med -3.08 ms vs
  ref1 (largest wall move of the window); D SIGNATURE NOT REPRODUCED -
  verify issue med -0.6 ms only (modeled -2..-6), wall -1.09 ms, t/s -0.61%
  (low side of band); engaged by construction (unconditional on the
  no-peer-copy fallback path), the two blocking syncs per boundary copy
  dominate and stay; ALL stacks all signatures (waits 4/round, drains/round
  194.0 = window minimum). VERDICT: all five arms free and byte-exact
  (defaults-safe), none is the t/s lever at 200k in-split - wall med stays
  192-194 ms in every arm; E-078's device-owned round confirmed on cooled
  silicon.
- E-087 2026-09-22 T3 GROUPED-MMVQ DEVICE VALIDATION (combined validation
  desk; one boot GGML_CUDA_MMVQ_GROUP=1 + DEBUG, receipt section 2).
  FORMATION PROOF CAPTURED: 18 "group formed" lines (2-member groups:
  node_14 q5_K rows 256, node_43 q5_K+q3_K, node_115 iq3_xxs+q5_K, node_187
  q4_K+q3_K, rows 12-256, prefill graphs included), 12 "admitted as
  copy-back" lines, verbatim max_rows declines ("mtp_Qcur_full-64 has 6144
  rows ... > max_rows 256; attn k/v shards are 512-granular under TP3").
  DEFECT: the server ABORTED on the first decode graph - controlled
  ggml_cuda_error (dmesg clean of libamdhip64): "ROCm error: invalid
  argument" at hipMemcpyAsync(node->data, it->second.temp, ... D2D) in the
  pending copy-back path, ggml-cuda.cu:5108, right after "group formed at
  Vcur-3: 2 members ... member Kcur-3 (copy-back)". Prefill grouped
  launches with copy-back members ran fine; the decode graph's copy-back
  placement fails (the off-device dst guard appears not to cover this
  case). t/s + accept UNDECIDABLE, determinism cell not runnable - handed
  back to the T3 desk with verbatim evidence; formation is NOT the
  bottleneck, copy-back execution is.
- E-088 2026-09-22 ADMISSION VALIDATION (E-054 fix, combined validation
  desk; one boot -c 10240, kv unified verified, probe_kv_admission.py).
  MECHANISM ENGAGED AND CLEAN: two defer lines ("unified KV occupancy is
  too high, defer task 2 (n_need = 7862, n_used = 512, n_pending = 7345,
  n_ctx = 10240)" and the retry at n_used = 7872), zero "Context size has
  been exceeded", zero HTTP 500, request A 200 in 80.1 s (prefill 99.8 t/s,
  mean acc len 3.00). DEFECT FOUND - DEFERRED REQUEST STARVES: A released
  with its 7872-token KV retained by the prompt cache (slot picked by LCP
  similarity 0.999 - B is A's prompt + a 5-token suffix), and the admission
  formula counts B's FULL n_need = 7862 against n_used = 7872 with no
  credit for the 7857 already-cached prefix tokens (incremental need ~5
  against 2368 free); no eviction path exists, B is never scheduled and is
  cancelled at 12.3 min when the probe's 600 s client timeout closes.
  Probe verdict line: FIX FAIL (non-200) - captured as the deliverable.
  Fix direction: credit prompt-cache prefix reuse in n_need (admit when
  incremental need fits) or evict cached KV after N defers. --control mode
  left for a later window per brief (would hit the same starvation).
- E-089 2026-09-22 COMBO VERDICT + TWO FIX DESKS DISPATCHED. Combined
  window verdict (E-086..E-088): every merged fix byte-exact
  (sha 4beb1ba25219ee9b across 9 boots), engagement proven, but ALL
  arms inside the ref band (14.79-14.93) - the hygiene stack is
  defaults-safe and kept, none is the t/s lever; the round stays
  192-194 ms device-owned. TWO defects found by the window, both
  dispatched: (1) T3 COPY-BACK DECODE FIX DESK (wt-t3-graph,
  amd/t3-graph, synced to HEAD): groups FORM (18 formation lines,
  12 copy-back admissions, prefill grouped launches fine) but the
  first DECODE graph aborts with hipMemcpyAsync invalid argument at
  ggml-cuda.cu:5108 right after "group formed at Vcur-3 ... member
  Kcur-3 (copy-back)" - graph-capture legality of the copy-back D2D
  is the diagnosed class (unstable temp address / capture-stream
  mismatch); fix must keep solo timing + bit-exact + capture safety;
  served validation boot at the end under the lock. (2) ADMISSION
  STARVATION FIX DESK (wt-server-fixes, amd/server-fixes): the probe
  caught OUR E-054 fix starving the deferred request - after A
  completes, its 7872-cell cache is retained and B's FULL need (7862)
  is counted with no credit for the ~7857 already-cached prefix
  cells; fix = credit incremental cells (need minus cached prefix) on
  both admission paths + purge idle slots before denying a deferred
  task; probe re-run in both modes as validation. (3) RCCL transport
  and MMVQ kernel desks continue (the +15-25% and unknown-big levers).

- E-085 2026-09-22 VERIFY-TRANSPORT DESK COMPLETE (wt-verify-transport on
  amd/verify-transport, campaign HEAD dd9b275f6; RCCL spike run under the
  campaign boot lock, dies 0/1/2, receipts rccl_probe_full2_20260922_174518.txt
  + rccl_probe_transport2_20260922_174701.txt, desk receipt
  VERIFY_TRANSPORT_2026-09-22.md). (1) MAP - the transport swap needs no new
  subsystem: ggml-cuda already registers ggml_backend_comm_init /
  comm_allreduce_tensor proc addresses (HIP included) and the meta backend
  already prefers comm_allreduce over the butterfly; the serving build has
  GGML_HIP_RCCL=OFF so GGML_USE_NCCL is never defined and every boundary
  falls to allreduce_fallback. The internal AR (allreduce.cu) is compiled
  out on HIP and is 2-rank only. Butterfly fp32 sum order for n=3 derived
  from the meta code: ((r0+r2)+r1), bit-identical across ranks (commutativity
  + copyback). (2) RCCL SPIKE - FEASIBLE, WINS: RCCL 2.20.5 (gfx900 device
  code embedded in librccl.so.1.0.60200) initializes 3 ranks in ~650 ms with
  canAccessPeer=0 all pairs; transport = SHM/direct/direct on every channel
  (host shared memory, pipelined, 2 channels, ring+tree). Grouped
  ncclAllReduce from one host thread (the exact ggml integration shape):
  32 KB fp32 avg 94.0 us vs 212.0 us for a faithful butterfly primitive
  emulation on the same boot (2.26x); 128 KB: 152.4 vs 340.3 us (2.23x).
  Against the SERVED ~1 ms/boundary (4 staged copies + 3 graph dispatches
  on top of the primitive) RCCL removes ~85-90% of the boundary tax:
  modeled ~35-45 ms/round of the ~48 ms tax = +15-25% decode class.
  (3) ACCEPTANCE EXPERIMENT (probe tier, 64 seeds x 2 sizes): NOT BIT-EXACT,
  as fp non-associativity predicts: 0/64 seeds byte-match the butterfly
  grouping; ~3.5% (8192 elems) / ~4.4% (32768 elems) of elements differ,
  max ULP 64/256, bounded dust (no sign flips); RCCL outputs byte-identical
  ACROSS ranks 64/64 (replica consistency preserved). VERDICT per campaign
  law: sum-order reorder = numerics class change = NEEDS CHRIS'S EXPLICIT
  SIGN-OFF. No RCCL algo/proto can reproduce a uniform association (ring
  reduce-scatter reorders per chunk), so bit-exactness is unreachable by
  env tuning; the only bit-exact device collective shape is a fixed-order
  one-shot AR (allreduce.cu class) - separate desk if the reorder is
  declined. (4) INTEGRATION (implemented, gated): ggml-cuda.cu HIP default
  comm mode = init_none (butterfly) - unset env keeps today's bytes on
  every build, RCCL stays opt-in via the existing GGML_CUDA_ALLREDUCE=nccl
  (a GGML_RCCL_BOUNDARY alias was rejected as a second knob). Served-arm
  spec for the sign-off window: build -DGGML_HIP_RCCL=ON (compile clean,
  librccl links), run GGML_CUDA_ALLREDUCE=nccl, record RCCL connect lines
  as the acceptance fingerprint; greedy sha WILL differ (that is the
  signed-off change) - gates that must hold: same-sha determinism across
  two boots of the arm, accept 0.66667/3.00 band, 8-needle, engagement =
  boundary tax lines gone + decode +15-25% class. Host tests:
  test_pinned_staging_host, test_verify_row_sampling_host,
  test_mtp_sampling_host, test_packed_get_host, test_async_input_host all
  PASS; gfx900 compile clean 0 warnings with and without
  GGML_HIP_RCCL=ON. (5) SECONDARY (pinned-ring async staging for the
  legacy butterfly, -2..-6 ms/round): designed, deliberately NOT
  implemented while the ~35-45 ms RCCL lever awaits sign-off; it becomes
  the fallback lever only if the reorder is declined.
- E-090 2026-09-22 RCCL SERVED VALIDATION - +25-27% DECODE CONFIRMED
  WITH CONTROL. Two RCCL boots (GGML_CUDA_ALLREDUCE=nccl, RCCL-linked
  binary): decode 19.01 / 18.72 t/s (+27.2% / +25.3% vs 14.94
  baseline), accept 0.66667/3.00 in both - the numerics dust did NOT
  move the accept loop. CONTROL (same binary, env unset, cooled to
  43 C, cooldown-gated): 14.98 PASS = the butterfly reference exactly;
  an interim un-cooled control at 9.38 was voided as inherited soak
  (third consecutive boot). ARM IDENTITY: the only difference between
  the arms is the env; the modeled 15-25% delivered as 25-27%.
  SERVING ADOPTION REMAINS CHRIS'S SIGN-OFF (numerics class: RCCL sum
  order differs from butterfly by bounded dust - ~3.5-4.4% of
  elements, max ULP 64/256, ranks consistent 64/64; byte-exactness
  unreachable by any RCCL algo). Until signed off: binary ships
  RCCL-linked with env UNSET = byte-identical butterfly (init_none
  default, init failure falls back). The ~48 ms/round boundary tax is
  now ~21 ms - round ~165-170 ms, decode 18.7-19.0 at 200k on stock
  clocks. At 10k, 20 t/s is at the doorstep. Remaining levers: MMVQ
  kernel desk (running), catch-up rollback (+0.8-1.6%, merged),
  T3 copy-back decode fix + admission starvation fix (desks running).
- E-090 2026-09-22 ADMISSION STARVATION FIXED + SERVED-VALIDATED
  (admission starvation fix desk, wt-server-fixes, amd/server-fixes
  synced to HEAD 4ab107998, receipt AdmStarvFix_20260922_181538.md).
  FIX (two parts, E-088's "credit the cached prefix" direction):
  (1) kv_unified_cells_needed now credits the candidate slot's cached
  common prefix regardless of cache_prompt - launching on the slot
  keeps or drops that prefix, either way those cells are not
  additional pressure (exact with cache_prompt, conservative without:
  the slot drops the retained prefix before placing new cells, and
  LCP <= cached cells keeps the fit inequality sound); applied on the
  single admission path that serves BOTH initial admission and the
  deferred re-admission, and mirrored in the in-flight n_pending
  accounting. (2) PURGE BEFORE STARVE: after the existing idle-slot
  purge pass, a last relief evicts the CANDIDATE slot's own cached
  prompt (saved to the prompt cache first) before denying - only when
  evicting actually flips the verdict (the full task then fits) - so
  no purge pass can starve on a skip-param blind spot. E-054 guarantee
  intact: in-flight n_pending reservation untouched, purges/evictions
  only ever touch idle (non-generating) slots. HOST: sections 9-11
  added to test_server_exposures_host.cpp at the exact served cells
  (defect arithmetic verbatim, non-overlapping C must still defer
  against an in-flight incumbent, eviction-only-when-it-flips); all
  existing host suites green on the branch. SERVED (lane 8083, -c
  10240 in-split, kv unified verified, probe both arms): FIX PASS -
  1 defer line ("defer task 2 (n_need = 7857, n_used = 512,
  n_pending = 7350)") while the incumbent prefills, then re-admitted
  30 ms after release on LCP similarity 1.000 with the incremental
  credit against the retained 7877-cell cache, both requests 200
  (80.1 s / 160.8 s), zero exceeded, zero 500 - the E-088 starvation
  (600 s timeout cancellation) is gone. With cache_prompt = false the
  server re-prefills the full prompt after dropping the retained
  prefix (conservative case, correct); with cache_prompt = true the
  prefix reuse collapses the re-prefill to the incremental cells
  (host-test level). CONTROL ARM (--no-kv-admission) CAPTURED, NEW
  SIGNATURE: no textbook 500/exceeded class - the overcommit halves
  n_batch to 2 on KV-full then ABORTS the server (GGML_ASSERT(task)
  at server-context.cpp:365 + "speculative batch index 2 is not
  inside the current sub-batch [0, 2)" via ggml_abort in decode);
  A got 500, B got a dropped connection. On today's HEAD the
  admission gate is the only thing between this geometry and a
  server-killing abort - the legacy knob is not a safe escape hatch
  for concurrent ~8k prompts (abort root-cause out of desk scope,
  evidence verbatim in the receipt). Control rerun without draft-mtp
  (textbook-signature hunt) left unrun: the t3-copyback desk holds
  the lane (lock law).
- E-091 2026-09-22 CATCH-UP VERDICT: t/s-NEUTRAL under RCCL at 10k
  (interleaved 2x2, cooled, engagement captured). ref 18.63 / 18.66 vs
  PREFIX_CATCHUP 18.59 / 18.45 - the modeled +0.8-1.6% did not
  materialize; the 11.66 ms catch-up is dominated by fixed decode-issue
  cost that the row reduction does not remove (consistent with the
  async-input desk's fixed-cost finding). The earlier 15.95 stacked
  cell was thermal (uncooled late-chain boot). Flag stays merged,
  OFF by default, not counted in the stack-up. RCCL reference
  reconfirmed at 10k: 18.59-18.66. Desktop review document updated to
  match.
- E-090 2026-09-22 T3 COPY-BACK DECODE FIX COMPLETE (wt-t3-graph, branch
  amd/t3-graph synced to d98e8cdcb; one served boot lane 8083 under
  the boot lock; receipt results/T3FIX_20260922_183334.md). E-087 ROOT
  CAUSE: the copy-back registration sized the D2D copy as the full 2-D
  area (src0->ne[1] * ne[0] * sizeof(float)) while a T=1 mmvq result is
  one float per weight row - the temp/dst hold row_diff floats; the count
  was ne[0]-times oversized (Kcur-3: 256 KiB from a ~1 KiB pool block).
  Eager prefill survived because first-fit pool blocks made the oversized
  ranges stay in-allocation (silent tail corruption, no error); the first
  DECODE capture crossed out-of-allocation ranges and HIP rejected the
  CALL itself -> "invalid argument" at ggml-cuda.cu:5108. All three
  briefed hypotheses EXCLUDED EMPIRICALLY (hipcc gfx900 capture matrix on
  an idle die, /tmp/t3_diag/capture_diag.hip): plain D2D hipMemcpyAsync
  on ctx.stream() captures, instantiates, replays x2 and verifies MATCH
  (cases 3/5/6) - the API form, stream and pool-temp address class are
  capture-legal; only the oversized count returns invalid argument at the
  call (cases 2/4, eager AND captured). FIX: one line, cb.nbytes =
  ggml_nbytes(member); test_t3_detect_host ALL PASS. SERVED EVIDENCE
  (combo line + MMVQ_GROUP=1 + DEBUG): zero ROCm errors, clean teardown;
  E-087's killer group ran - "group formed at Vcur-3 ... candidate Vcur-3
  admitted as copy-back + candidate Kcur-3 admitted as copy-back" in the
  decode-graph window (log lines 10729-10740) plus node_43/node_310
  (rows 12/18); prefill chunks grouped too (node_14, 64-line debug budget
  consumed exactly). Decode cell 14.82 t/s vs non-grouped combo refs
  14.79-14.93 (mean 14.87) - INSIDE the band, -0.81% vs baseline file;
  accept 0.66667 / mean len 3.00 exact; greedy sha 4beb1ba25219ee9b x2 =
  the E-086 cross-arm sha -> BYTE-EXACT vs env-unset (and the copy-back
  tail is now correct bytes; the old eager path silently corrupted it).
  Design constraints hold: solo timing, bit-exact, capture/replay-safe.
  T/s verdict needs the planned 3-rep A/B (single cell cannot resolve
  under the 0.94% noise floor). LOCK LAW route-back: a released lock does
  not imply a drained die - first boot attempt raced the coordinator's
  teardown and lost 1154 MiB of KV alloc to OOM (dmesg clean); launch
  gates now check live llama-server + per-die VRAM use, not just the
  lock; convention text should add a post-teardown settle delay.
- E-092 2026-09-22 T3 FIX INTEGRATED + ZOMBIE RELAUNCH + LOCK SETTLE
  RULE. (1) amd/t3-graph merged (e7dbc7785): the copy-back size fix
  (cb.nbytes = ggml_nbytes(member) - was the full 2-D area, ne[0]x
  oversized; eager prefill silently tail-corrupted, decode capture
  rejected) is in the campaign binary; served proof: grouped launches
  in decode graphs, zero ROCm errors, decode 14.82 in-band, accept
  0.66667, byte-exact sha. Remaining: a 3-rep A/B (single cell cannot
  resolve under the 0.94% noise floor) - queued for the final
  combined window. (2) LOCK SETTLE RULE adopted per the T3 desk's
  route-back (its first boot raced a coordinator teardown and OOM'd):
  a released lock does not imply a drained die - 60-90 s settle +
  per-die VRAM verify before booting; kills PID-targeted only.
  Automation text updated. (3) MMVQ kernel desk found ZOMBIE (4 h,
  zero artifacts) and relaunched as wt-mmvq-kernel2 on
  amd/mmvq-kernel2 with a 60-minute first-deliverable milestone and
  an end-turn-on-blocker requirement; zombie check (>90 min silent,
  zero artifacts) added to the automation.
- E-093 2026-09-22 T3 GROUPED DECODE 3-REP A/B VERDICT: t/s-NEUTRAL.
  Interleaved 3 reps at 200k, decode cells, accept 0.66667 everywhere:
  ref 15.05/14.97/15.05 (mean 15.02) vs GROUP=1 15.00/14.96/14.89
  (mean 14.95) = -0.5%, inside the 0.94% noise floor. The launch-count
  saving does not move decode (consistent with the drain budget:
  device math per step is small vs structure). The copy-back size fix
  itself stays merged (removes the abort AND a silent eager tail
  corruption); grouped decode stays env-gated OFF by default, not in
  the stack-up. Receipts t3ab_*.jsonl.
- E-094 2026-09-22 RCCL SIGN-OFF + SERVING PROMOTION (Chris: "i
  approve rccl"). Canonical launch script /home/chris/
  launch_tp3_200k.sh now sets GGML_CUDA_ALLREDUCE=nccl; numerics class
  change accepted (RCCL sum order, bounded dust, ranks consistent,
  boot-deterministic - E-090). BASELINE RE-STAMPED UP on the RCCL
  config: decode gate 14.94 -> 18.72 (measured 19.01/18.72/18.74
  across three boots; conservative median), allreduce=rccl recorded in
  the baseline config block. WAYPOINT STATUS vs origin: 17.8@2k era ->
  18.72-19.01 @200k = the origin number is BEATEN at 5x the context
  (first time any config has passed 17.8 at deep context). 20 t/s at
  10k is one lever away (MMVQ kernel arms pending). Byte-exact
  butterfly remains available by unsetting the env.
- E-093 2026-09-22 MMVQ RELAUNCH: BASELINE REPRODUCED, FIRST DELIVERABLE
  BANKED. wt-mmvq-kernel2 (amd/mmvq-kernel2 @ 075d1501b) reran the
  oracle-gated 8-arm bench on die 3 (lock-compliant, 75 s settle):
  base 421.1 us/call = 91.1 GB/s, -0.24% vs the banked 422.1/90.9
  (P0 gate +-2% PASS), all 8 arms within +-1.3% of the 09-21 session,
  oracle PASS 1.7e-06 everywhere. Control rep spread 1.6% noted
  (median agreement 0.24%; 1% law enforced on future A/B verdict
  sessions). Receipt: results/W2_mmvq2_baseline_2026-09-22.md.
  Pre-kill recorded: arithmetic derivation of iq3s_grid is dead (the
  512-entry grid is a trained codebook, not arithmetic). Live doors
  named: T=2-4 tiled shapes (never benched; MTP verify band runs
  T>1 in served decode graphs), gfx900 dot-product ISA compile probe,
  q8_1 operand layout. Next: T>1 harness extension.
- E-094 2026-09-22 MMVQ T-BAND WIN: DECODE-ONCE SHARE -34.1/-46.2/-43.0%
  KERNEL AT T=2/3/4, BIT-EXACT. The shipped mul_mat_vec_q re-executes the
  y-independent decode (8 const-LUT lookups + sign chain) once PER TOKEN;
  share computes the 8 signed quads once per (kbx, lane) and the per-token
  loop keeps only x-int loads + dp4a in the same order. Session of record
  (10 configs, 3 interleaved reps x 200 iters, oracle-gated, all PASS
  <=5.4e-06): t1 418.5; t2 base 753.8 -> share 497.0; t3 1190.7 -> 640.2;
  t4 1523.0 -> 868.9. share+48B-aligned-y (aln) adds -12.8% at T=4 only
  (757.7, -50.3% vs base) and needs the q8_1 producer relayout - banked as
  named lever, not shipped. T-scaling of the shipped kernel: 1.80x/2.85x/
  3.64x per T-step (floor is T-invariant ~209 us) - the verify band
  (MTP k=3 -> T=4 every round) was paying 3.6x for 1x of weights.
  Static-count served projection (N-19: candidate only): 0.77 x -43.0%
  = -33.1% decode kernel time -> t/s upper bound x1.49 pending served A/B.
  ISA probe closed arm (b): v_dot2_i32_i16, v_dot2_f32_f16, v_dot4_i32_iu8,
  v_dot4_i32_i8 all refuse to assemble on gfx900 (rocm-6.2.0) - dp4a
  emulation is irreducible, consistent with the 09-21 receipt. Oracle-gate
  defect closed: NaN-blind green (rel >= gate is false for NaN -> all-NaN
  arm passed); fail condition is now !(rel < gate). Arithmetic derivation
  of iq3s_grid pre-killed (trained codebook). Next: env-gated share in
  mmvq.cu, then real-kernel bit-exact gate, then served A/B. Receipt:
  results/W2_mmvq2_tband_2026-09-22.md.
- E-095 2026-09-22 MMVQ SHARE SHIPPED ENV-GATED + BIT-EXACT GATE + STOP
  CEILING. (1) vecdotq.cuh: vec_dot_iq3_s_q8_1_decode/apply split pair;
  mmvq.cu: GGML_CUDA_MMVQ_IQ3S_SHARE=1 (tile-gemm env pattern) wires the
  share path for IQ3_S, ncols_dst 2..4, rows_per_block 1 (GCN); default
  OFF, env unset = byte-identical shipped path. (2) GATE: the shipped
  split pair benched as t2/t3/t4_ship arms is BIT-IDENTICAL to base
  (memcmp 64 rows x T tokens, all three) and timing-identical to the
  harness share arm; three consecutive 3-rep interleaved sessions agree
  within ~1%: base 753.8/1190.7/1523.0 vs share 496-500/635-640/869-873
  us/call = -33.9/-46.7/-42.8% at T=2/3/4. Served projection (static-
  count class): at MTP k=3 verify T=4, 0.77 x -42.8% = -33.0% decode
  kernel time, t/s upper bound x1.49 pending served A/B; with the aln
  producer relayout (-50.4% total) x1.63. (3) STOP CEILING: T=1 closed
  at ~91 GB/s (decode chain = divergent const-LUT + emulated dp4a, both
  ISA-walled on gfx900, no dot instruction assembles); T=2-4 residual
  after share is 4.2x floor at T=4, next link = aln 48 B q8_1 producer
  relayout, then the T=1 chain wall. Queue: campaign build + served A/B
  in the final combined window. Receipts:
  results/W2_mmvq2_baseline_2026-09-22.md, W2_mmvq2_tband_2026-09-22.md.
- E-095 2026-09-22 MMVQ SHARE SERVED A/B VERDICT: t/s-NEUTRAL served
  (share 18.82 vs ref 18.99, in noise; byte-exact so zero risk; keep
  merged env-gated OFF). The -43% kernel-time bench win did not
  translate: the verify round's wall time is structure-bound (boundary
  tax, replay count, DMA) - kernel math was already only a slice of
  it. FOURTH independent confirmation of the device-structure
  attribution. ref2/share2 cells failed client-side (no crash, no
  segv - dmesg clean); verdict stands on ref1/share1. CAMPAIGN SPEED
  OUTCOME OF RECORD: RCCL mode = THE win (+25-27%, signed off,
  serving default, baseline 18.72); all other levers measured neutral
  or are designed-parked (RCCL deeper integration, on-device
  sampling). The 20 t/s @10k line needs the next subsystem (on-device
  sampling or RCCL extension to more transfer classes), not more
  micro-optimization.
- E-096 2026-09-22 THE THREE SUBSYSTEM PROJECTS STARTED (Chris: "start
  on the 3 designed subsystem projects now"). All zero-GPU
  (design/implement/host-test), fresh worktrees off 3f17b0f28:
  (1) ON-DEVICE SAMPLING (wt-onsample, amd/onsample): per-shard
  argmax op for the greedy draft path - 3 tiny (value,index) pairs
  per step replace the 517 KB row consumption; deterministic
  tie-break matched to the host argmax, meta-carve-out designed to
  avoid the E-065 SPLIT_AXIS_UNKNOWN abort; LLAMA_DRAFT_ONDEVICE_
  ARGMAX=1. Strategic value: prerequisite for step-batching.
  (2) RCCL COVERAGE EXTENSION (wt-rccl-ext, amd/rccl-ext): prefill-
  sized reductions (ne >= 131072) currently fall back to butterfly/
  bf16-compress - bench RCCL vs butterfly at real prefill sizes,
  gate-extend if it wins; numerics: the owner's sum-order sign-off
  covers the class globally, but prefill numerics feed KV - served
  full battery gates required; plus a transfer-class audit table.
  (3) LAUNCH/REPLAY-PATH (wt-launchpath, amd/launchpath): per-replay
  and scheduler-sweep cost measurement (LLAMA_LAUNCH_TIMELINE=1
  counter as first deliverable), TARGET-side sweep audit (draft side
  already 5-6 -> 1), per-replay overhead attack if measured high,
  honest ceiling doc for persistent-kernel/mega-graph options.
  All env-gated, unset = byte-identical; validation arms run in the
  coordinator's window when the desks land.
- E-097 2026-09-22 SUBSYSTEM DESKS PAUSED AT USAGE LIMIT (account 5 h
  limit; resets 2026-09-23 09:56). All three worktrees hold WIP:
  wt-onsample clean (died in design), wt-launchpath committed its
  LLAMA_LAUNCH_TIMELINE first deliverable (8a9d287cb) + profiler
  script, wt-rccl-ext rescued (a2dcaeff2). RELAUNCH PLAN: after
  reset, resume all three with "continue your predecessor's WIP in
  worktree X" briefs. THE RCCL-EXT DESK'S PRE-DEATH FINDING NEEDS
  CHRIS'S EYES (numerics of the CURRENT serving config): today's
  nccl mode runs BF16-COMPRESS at every PREFILL boundary (ne >=
  131072) - a MUCH wider numerics class than the "bounded dust" the
  E-094 sign-off text describes: ~99.998% of elements differ vs fp32
  reference, max relative error 3.3e4 at cancellation sites (bf16
  mantissa loss). Decode boundaries (ne < 131072) ARE the signed-off
  fp32 dust class. The desk's bench (rccl_prefill_probe, 3 clean
  boots): RCCL-F32 2.26-2.4x faster than butterfly at EVERY size
  (-7.69 ms/boundary at the 10 MiB prefill size); RCCL-BF16 (today's
  branch) is 1.9x faster than RCCL-F32 at 10 MiB - so the choice at
  prefill is precision (+2.61 ms/boundary, ~+1% prefill wall, worst
  +5.5%) vs today's bf16 class. Served gates passed on the bf16
  branch, but the sign-off text should not be read as covering it.
  The desk's gate implementation (WIP) gives Chris the fp32 switch.
  Decision queued for Chris alongside the reviewer's read.
- E-098 2026-09-22 SUBSYSTEM DESKS RELAUNCHED (Chris: "we got a early
  reset, you can run agents"). All three resumed with sync-onto-HEAD
  + resume-WIP briefs: RCCL-EXT (agent d22a9b55) finishes the
  size-class gate (fp32/bf16/butterfly selectable, default = today's
  behavior; the owner's prefill-class decision stays open), the
  transfer-class audit table, and the gate host tests; LAUNCHPATH
  (agent 6ee2a401) runs the predecessor's profiler offline, produces
  the launch/sweep cost table, then target-side sweep + per-replay
  fixes (env-gated) + ceiling doc; ONSAMPLE (agent d3c0455a) restarts
  the per-shard argmax implementation (predecessor died in design,
  zero WIP). All zero-GPU, env-gated unset = byte-identical, 90-min
  zombie rule armed.
- E-099 2026-09-22 LAUNCHPATH DESK LANDED: launch/sweep cost table +
  honest ceiling - the launch path is NOT the decode bottleneck
  (<= ~1 ms recoverable of the 163 ms verify block, <1%). The
  on-disk mtpgain_T1/F1 logs predate LLAMA_LAUNCH_TIMELINE (T1 was
  even -lv 3): no [launch-timeline] lines exist anywhere on disk, so
  the measured launch/replay split needs the validation boot
  (served-arm spec written). The profiler now falls back to a
  decode-only table when meta lines are absent; against F1-F4 it
  pins: verify issue med 162.3-163.2 ms (n=85 each), build 0.000,
  inputs med 0.040 ms (0.02% of the block), catchup med 11.2-11.7,
  drain med 5.5 calls/ubatch with only the first carrying the real
  wait (extras ~1-2 us, empty queue). Static audit of the target
  path (27 die replays + 8 boundaries per verify ubatch): replay
  path has no node loop, no allocations, no in-replay syncs;
  update_required already O(1) on stable uid; set_device self-guarded;
  the two real waste items were already shipped env-gated in
  8a9d287cb and this audit verified both sound (TARGET_LIGHT_SYNC:
  meta backend caps.events=false so the events==NULL gate engages;
  COMPAT_CACHE: uid==0 never cached, no false hits). Remaining
  candidates ~ns-25 us/ubatch: left alone, not worth gating risk.
  Ceiling doc (results/LaunchPath_20260922.md): persistent kernels /
  mega-graph with conditional nodes / device-side graph launch all
  attack the same <=1 ms host slice, not the 162 ms kernel+transport
  body - decode-rate levers live on the device-side desks. All six
  host suites green incl. test_launch_timeline_host; gfx900 compile
  clean (llama-server, no boots). Env work remaining for the window:
  boot with LLAMA_LAUNCH_TIMELINE=1 -lv 4, expect 27 "mode = replay"
  lines/ubatch, meta subs=9 replays=27 bounds=8, csync delta 1 with
  TARGET_LIGHT_SYNC.
