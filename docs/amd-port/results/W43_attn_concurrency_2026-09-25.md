W43 ATTENTION-CONCURRENCY DESK RECEIPT - 2026-09-25
branch amd/attn3 (worktree wt-attn3, off amd/v340-port-v2 @ 6408d20fc), v2
desk (v1 died in the 06:59:50 reboot ~17 min in; its uncommitted fill-arm
sketch was discarded per reset law and re-derived here). ZERO-GPU desk
behind the boot queue: receipts + code + arithmetic + staged runner.

MISSION (W37 Opp-3 / kernel ladder): the post-fa40 attention tile kernel is
~10.7 ms/cycle on the verify pool and LATENCY-BOUND (W17: KV stream
1.87-2.35 GB/s payload; depth law grows). W37 priced concurrency at
-2..-4 ms/cycle. Rungs offered: (i) KV-block software pipelining, (ii)
split-K along KV + combine, (iii) grid geometry / head-splitting at T=4,
(iv) own finding.

Inputs of record: W17_fa40codegen (fa40 oracle + 4-depth v11 timings),
W37_roofline_accounting (row 3 + falsifiable prediction 3), W14_widelaunch
(E1/E2 gap + rocprof kernel attributes), W27_q40prefill (oracle protocol),
fattn-tile.cuh / fattn-common.cuh @ 6408d20fc.

## 1. THE SERVED GEOMETRY, EXACTLY (code citations)

Instance of record (served decode arm, fattn-tile.cuh:1602-1627):
flash_attn_tile<256,256,4,2,use_logit_softcap,11> via launch_fattn<256,4,2>
with need_f16 = false,false. AMD config row (fattn-tile.cuh, amd table):
CASE(256,256,8,256,2,64,128) = 256 threads (8 warps), nbatch_fa 64,
nbatch_K 128. Static LDS (fattn-tile.cuh:1124-1126):
  Q_tmp  8 x 128 half2            =  4096 B
  KV_tmp 64 x (64+4) half2        = 17408 B
  KQ     8 x 64 half              =  1024 B
  total                            22528 B
Matches the W14 rocprof attributes of this shape class EXACTLY:
FA<4,2> lds 22528, scr 1104, vgpr 128, sgpr 64 (W14 receipt).

Grid at serve T=4 (TP4 die, bench = served geometry: 6 Q heads / 1 KV head
per die, gqa_ratio 6, docs/amd-port/tests/bench_attn_real.cu:7):
  ntiles_x   = ceil(4/4)              = 1   (fattn-tile.cuh launch chain)
  ntiles_z   = ne02/ncols2 = 6/2      = 3
  ntiles_dst = 3                          (fattn-common.cuh:1173-1175)
  ntiles_KV  = ceil(ne11/64)          = 112 at d=7168
KV tile loop: blockIdx.y strided (fattn-tile.cuh:1233-1245); per-tile work
= 2 K phases + 2 V phases (:783-815 K, :921-1015 V), q40 loader
:535-597 (per thread: 2 q4_0 blocks x [2x memcpy_1<4,2> + 2B scale]).

## 2. DIAGNOSIS: THE LAUNCH IS ALREADY RESIDENCY-SATURATED

The parallel_blocks scan (fattn-common.cuh:1108-1171) starts at
max_blocks_per_sm = hipOccupancyMaxActiveBlocksPerMultiprocessor(...).
For this kernel that occupancy is capped at 2 CTAs/CU by BOTH resources:
  VGPR: 128/thread x 64 lanes x 4 B = 32 KB per wave64; 64 KB SIMD file
        -> 2 waves/SIMD = 8 waves/CU = 2 CTAs/CU (4 waves each, 1/SIMD).
  LDS:  22528 B/CTA -> floor(64 KB/22528) = 2 CTAs/CU.
Device residency cap = 64 CUs x 2 = 128 CTAs.

The scan then picks pb = 42: blocks_per_wave = 64 x 2 = 128; efficiency
3x42/128 = 98.4 percent at one "wave"; pb=43 crosses to 2 waves at 50
percent and the 95-percent break fires. Result: 126 CTAs = 98 percent of
the 128-CTA residency cap, each CTA walking ceil(112/42) = 3 KV tiles at
d=7168. THE LAUNCH ALREADY FILLS EVERY RESIDENT SLOT THE KERNEL CAN HOLD.

Consequences for the offered rungs:
- (ii) split-K fill: raising pb cannot raise in-flight work (residency is
  hard-capped); it only repartitions the same 336 tile-jobs
  (ntiles_KV x ntiles_dst = 112 x 3) over the same 128 slots. Makespan
  under any per-tile-latency model = ceil(336/128) x L = 3L regardless of
  pb (chains of 3 at pb=42; three passes of 1-tile CTAs at pb=112).
  Predicted delta: 0 percent +- tail noise.
- (i) software pipelining: on gfx900 there is no async global->LDS copy
  (cp.async is NVIDIA; the skill's Team-Red table says the port is
  prefetch-to-REGISTERS double buffering). One K phase = 256 q4_0 blocks
  over 256 threads = 1 block = 16 half2 per thread = 32 live VGPRs of
  register buffer per stage. 128 + 32 = 160 VGPRs -> 40 KB/wave -> 1
  wave/SIMD -> 1 CTA/CU: residency HALVES to buy latency hiding the warp
  scheduler already provides across 16 warps/CU. The LDS side does not
  rescue it: a second 17.4 KB tile buffer at nbatch_K=128 gives
  22528+17408 = 39936 B x 2 CTAs = 78 KB > 64 KB (occupancy drops there
  too); smaller phases fit LDS but still die on the VGPR term.
  KILLED BY BUDGET MATH, not by taste.
- (iii) head-splitting at T=4: ntiles_x is already 1 (T=4 = ncols1) and
  the merge direction (ncols2 = 6, wide6) was MEASURED in W14: kernel
  -24.4 percent but the E1 start gap +221 us per launch made the wall
  +8-9 percent WORSE at serve depth; the gap is fixed while the win
  scales, crossover ~900-1000 us base - far above the 552 us serve cell.
  DEAD by W14 receipt.

Why the kernel is slow at all (the W17 anomaly, quantified): W17 v11
timings fit t = 226 us + 108.7 us x chain on the (7168, 10240) pair:
  d=7168   chain 3    552.4 us     d=65536  chain 25   4021.8 us
  d=10240  chain 4    661.0 us     d=199936 chain 74  16310.9 us
L = 109-220 us PER 64-KEY TILE is 100x above load-latency arithmetic
(payload 6.2 MB/launch at d=7168 = 11 GB/s; a full tile-load round trip
is ~1-2 us). The only receipt-consistent magnitude is PER-TILE SCRATCH
TRAFFIC: scr 1104 B/thread (W14) x 256 threads x 336 tile-jobs x 2
(write+read) = ~190 MB/launch -> 552 us at ~344 GB/s, i.e. AT the
measured consume wall (327 GB/s, E-006). This is a HYPOTHESIS with the
right magnitude, not a proof; the register diet that would fix it is
BLOCKED (any dot-chain/store-shape codegen change reopens the W17
miscompile minefield; W17 stop law: no further variant warranted).

VERDICT: bound = latency/serial (W17) with residency saturation (this
desk). All three named rungs are killed or capped by arithmetic +
receipts BEFORE card time. What survives is the cheapest measurement
that can falsify this desk: the KV-split fill sweep (rung ii), staged
here, predicted ~0 by this desk and -25..-45 percent by W37's model.
One bench session decides; W37's Opp-3 gate closes either way.

## 3. THE ARM (implemented, default OFF)

ggml/src/ggml-cuda/fattn-common.cuh:
- :970-973 launch_fattn gains `const int fill_parallel_blocks = 0` (defaulted;
  every existing caller unchanged, byte-identical codegen).
- :1182-1188 fill logic: 0 = the shipped efficiency scan; > 0 pins
  parallel_blocks (clamped to [1, ntiles_KV]); < 0 = max split (one KV
  tile per block). Fixed value per ne11 -> deterministic combine order.

ggml/src/ggml-cuda/fattn-tile.cuh (decode arm only, :1602-1628):
- :1613 env GGML_CUDA_FATTN_TILE_Q40_FILL (static getenv, default off) is
  read and passed through to launch_fattn (:1627); the WARN line gains
  fill=%d.
- Kernel body UNTOUCHED: var-11 scalar half2 store shape untouched (the
  W17 blade), no new template instantiation, no numerics change when the
  env is unset (negative-control law).

docs/amd-port/tests/bench_attn_real.cu (extended, not rewritten):
- launch_arm_q40_fill<VAR, FILL> (:144-164) mirrors launch_arm_q40 plus
  the fill param; arms fill56 / fill84 / fillmax / filldup / v11dup.
- `--conc` arm set: { v11 (pb=42 scan control), v11dup (in-run control),
  fill56, fill84, fillmax (pb=ntiles_KV), filldup (fill84 determinism
  control) }; CENSUS line prints numRegs/static_shared/local for the v11
  instance into every W43 log (W14-attr class, T-18 lineage stamp).
- Oracle: v11dup and filldup must be BIT-EXACT vs their references (hard
  fail); fill56/fill84/fillmax vs v11 are DUST-class by design (the
  combine reduction order changes with pb; W27 DUST-PASS <= 1.3e-3,
  owner sign-off for serve). Existing arms/modes untouched.

BUILD (this worktree, canonical flags):
- /home/chris/opt/cmake/bin/cmake -B build-hip -DGGML_HIP=ON
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx900 -DGGML_HIP_RCCL=ON -DLLAMA_CURL=OFF
- ggml-hip target: BUILD-EXIT:0 (libggml-hip.so relinked)
- bench: /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900
  -DGGML_USE_HIP -DGGML_BACKEND_BUILD -DGGML_SHARED -I ggml/include
  -I ggml/src -I ggml/src/ggml-cuda docs/amd-port/tests/bench_attn_real.cu
  -o /tmp/bench_attn_real_w43 -L build-hip/bin -lggml-hip -lggml-base
  -lamdhip64 -Wl,-rpath,<abs>/build-hip/bin -Wl,-rpath,/opt/rocm-6.2.0/lib
  -> CC-EXIT:0
- llama-cli: CLI-EXIT:0 (full main-binary compile proof)
No GPU touched at any point (boot-queue law).

## 4. ORACLE + TIMING - STAGED (lock-arbitrated, one short process per
invocation; resolve the HIP index from PCI at runtime per the W27 law -
on the current boot PCI 0000:0d:00.0 = HIP index 2; NEVER hardcode):

  cd /media/chris/ssd128/llamacpp/wt-attn3
  D=$(HIP_VISIBLE_DEVICES=$IDX /tmp/bench_attn_real_w43 2>/dev/null); # IDX per rocminfo BDFID

  # (a) serve depth, all arms, oracle + timing in one process:
  HIP_VISIBLE_DEVICES=2 /tmp/bench_attn_real_w43 \
      /media/chris/ssd128/models/Qwen3.8-27B-ASCII-P1M.gguf 30 5 --conc --depth 7168
  # (b) depth decomposition (F vs chain-slope, same session class):
  HIP_VISIBLE_DEVICES=2 /tmp/bench_attn_real_w43 <gguf> 30 5 --conc --depth 10240
  HIP_VISIBLE_DEVICES=2 /tmp/bench_attn_real_w43 <gguf> 30 5 --conc --depth 65536

Adjudication per cell:
  v11dup  BIT-EXACT vs v11     else session VOID (rerun)
  filldup BIT-EXACT vs fill84  else fill arms VOID (nondeterministic)
  fill56/84/max vs v11: BIT-EXACT or DUST with max_rel <= 1.3e-3 (W27
  DUST-PASS class; anything else = GARBAGE -> stop, no variant sweep)
  staged channel (GGML_FATTN_Q40_DBG build, optional): fill arms must
  print KT=ok VT=ok KQ=ok - block (0,0,0) computes tile 0 at every pb, so
  the stage dumps are pb-independent and must match v11 exactly.

Timing read: per-launch med us (niter=30 back-to-back, rep 0 discarded,
SPREAD-FAIL flag > 1.05 percent per the W11 law). All fill windows within
+-2 percent of v11 = rung (ii) DEAD (bank the negative, this desk's model
stands, next link = section 5). Any fill arm <= -5 percent = UNAMBIGUOUS
falsification of the residency model; <= -25 percent = W37 prediction 3
fires (552.4 -> <= 414 us) and the winner goes to the paired window.

Expected ms/cycle arithmetic (both models, falsifiable):
- This desk (residency): 0 ms/cycle; the sweep costs one bench session
  and closes W37 Opp-3 with a measured negative.
- W37 model: -25..-45 percent per launch at d=7168 -> 16 launches x
  -138..-248 us = -2.2..-4.0 ms/cycle -> 24.69 -> 25.1-25.7 t/s
  (+1.6..+4.0 percent), growing with depth (chain 25 -> 10 at 64k would
  be ~-2.4 ms/LAUNCH under this model - the 64k cell prices it).

## 5. NEXT LINK IF THE SWEEP IS FLAT (named, not owned)

The residual is the FA<4,2> register/spill profile: vgpr 128 + scr 1104
(W14 attrs) with per-tile scratch churn the only mechanism that closes
the W17 magnitude gap. The lever is a register diet (VGPR <= 64 would
double residency; killing the spill would attack L directly), and BOTH
routes are codegen changes inside the W17 miscompile blast radius. This
desk does NOT open that; it names it as the standing next link, gated on
new evidence (compiler bump off hipcc clang 18, or a gfx90a-class
re-derivation), per the W17 stop law.

## 6. SERVED PAIRED-WINDOW SPEC (U-W43, only if a fill arm fires)

Prereq: section 4 oracle EXACT/DUST-PASS at all cells AND a fill arm
<= -25 percent at d=7168. Kill criteria (any one kills):
- paired prompt_tps gain < +0.5 percent at the serve-depth cell (W37
  falsifiable 3 served leg)
- any oracle breach, any v11dup/filldup nondeterminism, any decode-arm
  regression (decode canary = the promoted fa40 var-11 arm with FILL
  unset must reproduce the of-record window within +-2 percent)
- MTP acceptance moves beyond window noise (D3 law: the combine reorder
  is DUST-class per element; acceptance is the exchange rate - measure
  it, do not assume it)
Command shape (coordinator):
  GGML_CUDA_FATTN_TILE_Q40_DIRECT=1 GGML_CUDA_FATTN_TILE_Q40_FILL=<winner> \
  ... llama-server (paired window, same prompt population, fa40-of-record
  window as control; FILL-unset boot = negative control, must be
  byte-identical served behavior per the default-off law).
Promotion bar: default-ON only after TWO independent windows clear the
gates; the scan (FILL unset) stays the shipped fixture.

## 7. LEDGER

- Rung (i) pipelining: KILLED pre-card (VGPR budget math, section 2).
- Rung (iii) head-splitting: KILLED by W14 E1 receipt (section 2).
- Rung (ii) KV-split fill: ARMED as the falsifier (default-off env +
  staged bench); predicted 0 (this desk) vs -2..-4 ms/cycle (W37).
- Named next link: FA<4,2> vgpr128/scr1104 register profile (section 5),
  blocked by the W17 miscompile stop law.
- Blade respected: zero store-shape changes; var-11 instance untouched;
  every change behind a default-off env; all builds green.

Assisted-by: GLM-5.3-Flash
