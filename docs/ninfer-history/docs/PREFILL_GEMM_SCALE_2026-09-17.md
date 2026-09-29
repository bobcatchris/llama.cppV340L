# PREFILL GEMM SCALE — the wall, the HFMA2 tiled arm, the chunk ladder (no-GPU desk, 2026-09-17, amd/tp4-cure)

**Seat:** NO-GPU design desk. **Method:** code-read of the shipped tiled kernel + banked rows
(P1 OPTRACE, SWEEP, TILED_GEMM notes, SMALLT-PK precedent) + arithmetic. Zero GPU work, zero
builds, zero src/ edits. **Question:** user goal prefill ≥500 tok/s (from 75.5) needs the GEMM
slice off its 798.8 ms/chunk measured level — issue wall or memory wall, and what kernel change
moves it. **This doc designs; it measures nothing.** Companion: `docs/amd/PREFILL_DECOMP_2026-09-17.md`
(the chunk decomposition), `PREFILL_FRONT_2026-09-17.md` (why the M>8 band needs a tiled GEMM at all).

**Measured base (P1, `PREFILL_OPTRACE_row.txt`, bin 6c8ae21399516750, plen-2000 probe):**
chunk wall 1642.3 ms = gemm **798.8** (48.6%) + ar 70.7 (4.3%) + body 645.1 (39.3%) + gap 122.7 (7.5%),
4-rank means, n=16, rank-uniform, gemm_over=0 (no cap clipping). Serving GEMM class =
**1556.9 GF/chunk/rank ÷ 0.7988 s = 1.95 TF/s = 18.1% of the 10.75 TF/s fp32 nominal anchor**.
(Bench V0 = 2.37-3.10 TF/s cache-served, SWEEP_row — serving runs at 0.66-0.67× bench class;
the ~270 ms delta is unexplained, suspects cold-code-load latency exposure + partial sclk droop
1138-1500 MHz. A GEMM lever must beat **798.8 ms warm**, per the P1 row's own words.)

---

## 1. THE WALL: ISSUE, not memory — three independent proofs

**Proof A — DRAM slack.** Weights per chunk per rank = 3.43 GB streamed (codes + scales, all
five problems); DRAM floor at the measured 368.7 GB/s read ceiling = **9.3 ms/chunk** vs 798.8
measured — **86x slack**. Even a zero-arithmetic weight scan cannot go below 9.3 ms; the kernel
is 86x above it.

**Proof B — cache-served rate.** SWEEP_row: V0's effective weight-stream rate at M=128 is
5.2-6.8 TB/s (cache-served timing loop) — 14-18x ABOVE the 368.7 GB/s ceiling. The bench never
waited on DRAM; its 22-29%-of-nominal ceiling is therefore an EXECUTION ceiling, not a
bandwidth one.

**Proof C — crossover M.** Arithmetic intensity of the dequant GEMM = 2M FLOP per 0.5625
weight-bytes = 3.556·M FLOP/B. Machine balance at the measured ceiling:
fp32: 10.75e12 / 368.7e9 = 29.2 FLOP/B ⇒ **M\* = 8.2**;
packed fp16 (2x): 21.5e12 / 368.7e9 = 58.3 ⇒ **M\* = 16.4**.
M=128 is **8x past even the packed-fp16 crossover** (16x past fp32). No memory-wall regime
exists at any M the production chunk sizes use. (The 378 GB/s achievable anchor gives the same
M\* = 8-16 within rounding.)

**Proof D — the instruction census (the design-relevant one).** V0 inner loop at TN=128
(thread = 4 rows × 8 tokens; per (g,v) iteration = 64 MAC/thread = 4 rows × 8 tokens × 2
elements):

| op class per (g,v) | count | notes |
|---|---|---|
| v_mul_f32 (dequant·coeff) | 64 | re-issued per (row, token, value) though coeff is per (row, group) |
| v_fma_f32 | 64 | 1 MAC per instruction |
| int VALU (code extract + bf16 unpack) | 24 | 4×2 extract + 8×2 unpack |
| LDS (pair table b64 ×4 + s_x b32 ×8) | 12 | co-issue port, not the ALU bill |
| **total issued VALU-class** | **152** | **2.4 issued instructions per MAC; 2.0 fp32 math ops per MAC** |

Wave32 on gfx900's 16-lane SIMD = 2 cycles/instruction ⇒ 304 cycles per (g,v) per wave for
4096 useful FLOP (wave-wide) ⇒ **54 FLOP/cycle/CU model = 42% of the 128 FLOP/cyc/CU fp32
peak** ⇒ theoretical 4.5 TF/s @1.5 GHz × 56 CU. Measured V0: 2.37-3.10 bench / 1.95 serving ⇒
**model-to-silicon efficiency 0.52-0.69** (LDS issue contention, barriers, tails, sclk). The
census brackets the measurement from above — the kernel is ALU-issue-bound exactly as the
SWEEP's falsifier-(b) text anticipated: "limiter is the FMA/pair-table issue stream itself …
cut decode ops per FMA instead."

**VERDICT: ISSUE WALL.** Two killers named: (1) the coeff multiply is 50% of the fp32 math and
is hoistable per-group; (2) every fp32 FMA carries 1 MAC where `v_pk_fma_f16` carries 2 at 2x
rate (gfx900 packed-fp16 pipe, ISA-verified in SMALLT-PK: `__hfma2` lowers to native
`v_pk_fma_f16`, zero emulation). Tile shapes are CLOSED (SWEEP falsifier (a) fired, V0 wins
15-41%); the arm below changes the **instruction mix per MAC at V0's fixed shape** — not a
shape.

## 2. THE HFMA2 TILED ARM ("V0-PK") — the PK pattern generalized to large M

Precedent: `nvfp4_small_t_hip.cu` `nvfp4_small_t_hip_kernel_pk` (SMALLT-PK) — issue-bound at
3x ALU ops per weight byte, packed-fp16 window + fp32 flush, tolerance-gated. V0 has the same
disease (2.4 issued ops/MAC, §1-D). The arm keeps V0's tile EXACTLY (block 64 rows × TN,
thread 4r × TN/16t, K-tile 64, XOR-swizzled s_x, u64 code loads, scale-quad trick) and changes
only the arithmetic, mirroring SMALLT-PK's five changes:

1. **e2m1 pair table as half2 bits.** `s_pair_f16_tab[256]` (u32 per entry, 1 KB — half the
   uint2 table), filled FROM `amd::e2m1_bits` via `__float2half_rn` — exact by construction
   (e2m1 values {0,±0.5,±1,±1.5,±2,±3,±4,±6} are dyadic, ≤2 significand bits, all fp16
   normals; RNE of an exactly-representable value is the value). Codec stays the single
   arithmetic home.
2. **x staged as packed fp16.** After the per-k-tile uint4 global load, each u32 crosses
   `ninfer::ops::bf16x2_bits_to_f16x2_bits` (math.cuh:74, D3 bridge, exact-class C441: bf16's
   7-bit mantissa embeds in fp16's 10 bits) before the swizzled LDS store. Same word count,
   same swizzle, same banking. Conversion cost: 16 u32 bridges/thread/k-tile vs ~2048 MACs
   amortized there — <1%.
3. **coeff stays the fp32 contract fold** — `bits_to_f32(e4m3_lut_decode(..)) × inverse_weight_divisor`
   per (row, group), exactly as V0 computes it — then cast once per (row, group) to
   `__half2 coeff2 = __float2half2_rn(coeff)` (16 cvt/thread/k-tile, hoisted out of the v
   loop). The e4m3 scale plane NEVER narrows to fp16 as a load path; its fp16 rounding happens
   once, on a value the flush design (below) treats as data.
4. **Inner loop: coeff folded into the weight pair, one pk-FMA per (row, token, value-pair).**
   Per (g,v): `pw2[r] = __hmul2(pair_f16_tab[byte_r], coeff2[r])` ×4 (invariant over the 8
   tokens — this deletes V0's 64 v_mul_f32 class), then per (r,t):
   `s2[r][t] = __hfma2(pw2[r], x2[t], s2[r][t])` — 32 v_pk_fma_f16, each carrying 2 MACs at
   2x rate. New census per (g,v): **36 pk ops + 8 int + 12 LDS vs V0's 128 fp32 + 24 int +
   12 LDS** — issued-ops per MAC drops 2.4 → ~0.6 with each op carrying 2 MACs.
5. **fp16 accumulation window W + fp32 flush.** `s2[r][t]` (half2, 32 VGPR at kS=8) sums the
   window; flush to the fp32 contract accumulator `acc[r][t]` (32 VGPR, unchanged epilogue)
   as fp32 adds of `__low2float + __high2float`. W is THE design knob:

| W (values) | coeff lives | flush bill/window | model FLOP/cyc/CU | %packed peak | predicted bench TF/s* | ×V0 |
|---|---|---|---|---|---|---|
| 16 (precedent-exact) | fp32 at flush (SMALLT-PK shape verbatim) | 2 cvt+add+fma ×32/group | 114 | 44% | **5.0-6.6** | 2.1-2.8 |
| 32 (recommended) | fp16 in-window (pw2 mul) | half the cvt rate | 141 | 55% | **6.1-8.2** | 2.4-3.3 |
| 64 (one k-tile) | fp16 in-window | minimal | 161 | 63% | **7.0-9.3** | 2.7-3.7 |

\* bench class = model × the measured 0.52-0.69 model-to-silicon band, @1.5 GHz × 56 CU
nominal anchor; sclk 1269-1500 in the SWEEP window already sits inside the measured
percentages' caveat. **Central prediction: W=32 ⇒ ~6-8 TF/s bench class = 57-76% of fp32
nominal = 28-37% of the packed ceiling — the "different compute ceiling" the mission named,
realized.**

**Register/LDS budget (V0 shape, TN=128):** acc fp32[4][8]=32 VGPR + s2 half2[4][8]=32 +
coeff2 4 live + pw2 4 + addressing ⇒ ~90-115 VGPR ⇒ **2 CTAs/CU (VGPR-bound; LDS 18 KB would
allow 3)** ⇒ 16 waves/CU = 4/SIMD — ample for an issue-bound loop. LDS: 1 KB f16 pair table +
1 KB scale table + TN×128 B s_x = 18 KB. If the ISA spills at kS=8, first knob is
flush-per-group (shrinks s2 live range), second is 2 rows/thread × TN/8 (TILED_GEMM_notes §5
grammar) — shape stays V0's family, no sweep reopen.

**k-ascending law.** The codec contract (nvfp4_amd_codec.h:131-138) fixes per-row consumption:
groups ascending, values 0..15 ascending, one fp32 accumulator, `fmaf(code*coeff, x, acc)`.
V0-PK preserves the ORDER (the g/h/v nest walks K identically) and changes only the
ASSOCIATION — the codec text itself prescribes the declaration grammar: "a perf pass that
changes ORDER changes the digest and must say so at its gate." This arm does NOT change order;
it changes precision association, which the SMALLT-PK precedent already declared
tolerance-gated, NOT bit-equal.

**The exact tolerance gate (verbatim bar):** per problem, **rel-L2(out_PK, out_V0) < 1e-2**
(the A16-class bar the roofline bench already applies S5-vs-PK), expected error ~1e-3-class
relative — under the NVFP4 quantization noise itself (~1e-1). Envelope guards, printed as a
bench census (reported, never a launch refusal — VRAM-law spirit):
(a) x fp16-normal envelope |x| ∈ [2^-14, 65504]; production post-rmsnorm hiddens ~±30 ⇒
≥20x headroom (SMALLT-PK's own numbers);
(b) window overflow: |s2| ≤ W · 6 · max|x| (W=16, coeff-at-flush: max|x| < 682) or
W · 6 · max|x| · max|coeff| (coeff-in-window; W=32/64) < 65504 — the bench prints
max|window| per problem so the guard is measured, not assumed;
(c) NO FNV/bit-equality cell may gate this arm — bit-identity is false BY DESIGN. The
in-serving `NINFER_TILED_VERIFY` FNV cell (TILEDV) will RED under PK; it must be OFF for PK
legs and its correctness role taken by the bench rel-L2 column + the standing ladder /
known-answer battery.

**Serving prediction (the honest one).** Bench central 6-8 TF/s (W=32). The unexplained
serving/bench class factor 0.66-0.67 (§ header) may be multiplicative (clock/first-touch
scaling with compute) ⇒ serving 4.0-5.4 TF/s ⇒ gemm slice 290-390 ms/chunk; or partly a fixed
per-launch cold term ⇒ add back up to ~1 ms/launch × 256 ⇒ up to ~470. Band:
**798.8 → ~250-470 ms/chunk (central ~350), i.e. 1.7-3.2x**, best case (factor proven
clock-only, PK rides bench class) ~190-240. Prefill impact with all other slices frozen:
wall 1642 → ~1090-1190 ms ⇒ **~107-117 tok/s** — the single biggest measured-available lever,
not the goal by itself (§4).

## 3. CHUNK LADDER (`--prefill-chunk` 128 → 256/512)

What M-scaling does NOT buy: total GEMM FLOPs for fixed plen are chunk-invariant
(12.16 GF/token/rank; 1556.9 GF per 128 tokens either way) — bigger chunks change
EFFICIENCY terms only, never the 1556 GF.

What it buys, per 1996-token request (16 chunks → 8):
- **gap:** 122.7 ms/chunk × 8 fewer chunk boundaries ≈ **−0.98 s/req** (the chunk-entry
  sync/re-enqueue bubble + eager tail; halves again at 512);
- **ARs:** count halves (128/layer-set/chunk × half the chunks), payload doubles (1.31 →
  2.62 MB). Measured 552 µs/AR at 1.31 MB is bandwidth-paced (2.4 GB/s ≪ the 98-101 µs RTT
  floor ⇒ latency is NOT dominant) ⇒ per-request AR bytes are invariant ⇒ **≈ wash** unless
  the 2.62 MB regime turns latency-mixed — UNMEASURED, bounded [0, −0.6 s/req];
- **launches/MTP/embed tails:** count halves (~−0.2 s/req, mostly inside gap's arithmetic
  already — do not double-count).
Total standalone prize: **+5-8% prefill (75.5 → ~79-81)**.

What it costs:
- **the V4 occupancy law:** TN=256 ⇒ s_x = 32 KB + 3 KB tables = 35 KB ⇒ **1 block/CU** — the
  exact footprint that cost V4 33-37% vs V0 (SWEEP). If M=256 GF/s < 0.9× M=128 GF/s, the
  gemm slice per chunk is ≥2× the M=128 slice and the gap/AR prize is eaten alive.
- kernel launcher: `switch(m)` registers {32,128,256} — 256 is instantiated; **512 is not**
  (and its 64 KB x-tile busts the LDS static_assert): chunk 512 means splitting into 2×256
  sub-launches at the seam (code cost, zero new math) for a further ~−0.5 s/req of gap —
  diminishing; only worth measuring if 256 comes back occupancy-clean AND gap+AR still >5%
  of wall.
- remainder chunks: 1996 = 15×128+76 = 7×256+204; tails (<64 or unregistered M) ride the
  legacy small_t chunk loop by design (dispatch threshold law) — unchanged either way.
- VRAM: the tiled arm allocates NO workspace (whole-M out view); chunk-256 doubles per-chunk
  activation buffers (largest projection out 256×8704 bf16 = 4.4 MB) inside the existing
  measured preflight/workspace posture (`NINFER_WORKSPACE_MIB=96`). Per the VRAM LAW: the
  allocator is the gate, the ladder arms measure, and **no estimated charge refuses anything**
  — a 256 boot either fits or reports cleanly in real time.

**Crossover verdict for the 500 goal:** chunk scaling touches ≤ ~1.6 s of the 26.3 s request
(gap+AR combined) — it is a SECONDARY stacking lever, never decisive. Decide it on the same
bench leg as the PK arm (one boot adds the `--prefill-chunk 256` leg), pre-registered
falsifier: M=256 GF/s < 0.9× M=128 ⇒ chunk stays 128, ladder closed on the occupancy law.

## 4. THE 500 GOAL — the honest statement

Budget: 500 tok/s at plen ~2000 = **≤256 ms/chunk**. Today's non-GEMM slices are
ar+body+gap = 838.5 ms/chunk. Two consequences, stated plainly:

1. **No GEMM-side lever alone reaches 500 — or even 152.** With the GEMM slice at literally
   ZERO ms, the wall is still 843.5 ms/chunk ⇒ 152 tok/s ceiling. The mission's "<~160 ms gemm
   slice" (which at the FLOP=2MNK convention = 1556.9 GF/0.16 s = **9.7 TF/s/rank = 90.5% of
   fp32 nominal = 45.3% of packed** — the mission's "5.3 TF/s" figure corresponds to a 294 ms
   slice at this convention) leaves ≤96 ms for ar+body+gap vs 838.5 measured: a 8.7x cut
   OUTSIDE the GEMM.
2. **The 500 goal is a three-front program.** Front 1 (this doc): V0-PK gemm 798.8 → ~250-470
   (best ~190). Front 2: the P1-named WINNER, the in-layer non-GEMM body, 645.1 → must reach
   ≤~60-100 ms/chunk (~7-10x, a kernel-efficiency program over norms/conv/rope/unpack/
   quant-dequant/silu + the every-chunk MTP prefill forward — NOT yet designed, owner desk).
   Front 3: gap 122.7 → ~20 via chunk graph capture (P1 Fix A, bounded upside already
   measured at 7.5%) + the chunk-256 stack. Illustrative combination ladder (per request,
   1996 tok):
   - today: 26.3 s ⇒ 75.5 tok/s
   - +V0-PK (central 350 ms/chunk): ~19.0 s ⇒ ~105 tok/s
   - +graph capture (gap→~20): ~18.2 s ⇒ ~110 tok/s
   - +chunk 256 (clean occupancy assumed): ~17.4 s ⇒ ~115 tok/s
   - +body 645→100: ~10.0 s ⇒ ~200 tok/s
   - +body 645→60 AND PK at best case (~200): ~6.3 s ⇒ ~317 tok/s
   - **500 tok/s additionally requires the body at ~40-60 ms/chunk AND gemm ≤ ~160-200 —
     i.e. every front at or beyond its best case.** Anything claiming 500 from a GEMM patch
     alone is arithmetic false.

## 5. DECISIVE CHECK S2 — bench + serving spec (next GPU window)

**Env gate (house pattern, magic-static read-once, default OFF = byte-identical production):**
`NINFER_TILED_PK=1` — flips `launch_nvfp4_tiled_gemm` to the V0-PK kernel, exactly mirroring
`NINFER_SMALLT_PK` (smallt_pk_enabled(), nvfp4_small_t_hip.cu:386-389). Explicit A/B launcher
symbol `launch_nvfp4_tiled_gemm_pk` for the bench, mirroring `launch_nvfp4_small_t_pk`.
Implementation home: the tiled TU (`nvfp4_tiled_gemm_hip.cu`) as a sixth arm at V0's shape
(shape constants shared, arithmetic swapped) — NOT a new tile variant id in the closed sweep
family; W is a `constexpr` template param {16,32,64}, one compile each.

**Bench leg (the decisive one) — `nvfp4_prefill_bench --pk`**, same LAW compile line, 5 s mclk
hammer + ceilings first:
- arms: V0 vs V0-PK at M=128 (primary) and M=256 (ladder) × **the five W4 SHARD geometries**
  (3584×5120, 4096×5120, 8704×5120, 5120×1536, 5120×4352 — production problems; prior sweeps
  ran the four FULL geometries, never the shards, never V-PK);
- W sweep: 16/32/64 (three compiles, ~1 min);
- columns: GF/s, %nominal, **rel-L2 vs V0 output (bar 1e-2, every row)**, max|window| census
  (the overflow guard, measured), peak sclk sideband. NO FNV column (§2 gate).
- **Pre-registered prediction + falsifiers:** W=32 PK ≥ 1.8x V0 GF/s at M=128 (conservative
  floor of the 2.4-3.3 model band). PK < 1.8x ⇒ the flush/cvt/int bill dominates the model —
  dump the ISA before ANY further tuning; next knobs in order: W=64, 2 rows/thread, x-half
  LDS prefetch. All-PK-arms < 1.3x ⇒ the arm is DEAD, the issue wall needs a different class
  of attack (write the ISA post-mortem, reopen nothing).

**Serving leg (same window, after the bench):** boot banked posture + `NINFER_TILED_PK=1`
(winNer W), `--prefill-chunk 128`, plen-2000 probe + standing ladder battery:
- decisive number: **[PREFILL-SUM] gemm < 798.8 ms/chunk** (P1 instrumentation already in the
  bin — zero new code for the readout);
- pre-registered: gemm ≤ 500 ms/chunk = arm confirmed in serving; gemm > 700 = the
  serving/bench class factor ate the arm — read sclk sideband before concluding;
- ladder + known-answer coherence per the standing battery (tolerance posture; the TILEDV FNV
  cell stays OFF for PK legs per §2);
- then ONE `--prefill-chunk 256` boot (PK and non-PK probes): [PREFILL-SUM] wall/16 vs /8,
  gemm ms/chunk vs 2× the M=128 value (occupancy law), AR column at 2.62 MB (prices §3's
  unmeasured term). Bank per-LABEL; BANK-BEFORE-RELINK per the runbook if a relink follows.

## 6. Honesty row

- The instruction census (§1-D) is a static count of the C++ source's unrolled form, not an
  ISA dump; it brackets the measured class from above (model 4.5 TF/s vs measured 1.95-3.10),
  which is the direction the issue-bound verdict needs, but the predicted PK TF/s inherit its
  uncertainty (hence the 0.52-0.69 silicon-efficiency band carried from V0's own
  model-to-measured ratio, and the pre-registered 1.8x floor).
- The packed-fp16 2x-rate anchor is doc-01 inference + SMALLT-PK's ISA verification of the
  lowering, NOT a measured v_pk_fma_f16 throughput on THIS die; the bench leg measures it
  end-to-end. gfx900 fp16 denormal behavior in the D3 envelope is part of what rel-L2 gates.
- The serving/bench factor 0.66-0.67 is a two-point derivation (T1 A/B + P1) with an
  UNEXPLAINED mechanism; §2's serving band spans both its plausible shapes (multiplicative vs
  fixed-per-launch). Only P1's own gemm column decides.
- The M=256 tiled GF/s has never been measured at the W4 shard geometries; §3's occupancy
  prediction transfers V4's measured 35 KB-LDS loss — a transfer, flagged as such, falsified
  by the S2 ladder leg.
- The body-lever sizes in §4 (645 → 100/60) name REQUIRED sizes for the 500 arithmetic; no
  design exists for them here, and none is claimed.
- This seat touched only this doc; the working tree's live GPU-window artifacts
  (results/amd/coherence/*) were read, never written.

---

Bases: PREFILL_OPTRACE_row.txt (798.8/70.7/645.1/122.7, ngemm=192 pairs = 5 site classes, gemm_over=0) ·
SWEEP_row.txt + TILED_SWEEP_notes.md (V0 2366-3103 GF/s = 22.0-28.9% nominal, cache-served
5.2-6.8 TB/s, ceilings 354.4/368.7 GB/s, tile family CLOSED, falsifier-(b) issue-stream text) ·
TILED_GEMM_notes.md (V0 design, M=256 35 KB/1-block occupancy risk, §6 incident) ·
TILED_DATUM_row.txt (75.7/42.1/41.6 serving A/B, ROUTE PROOF five W4 problems) ·
nvfp4_small_t_hip.cu SMALLT-PK (hfma2 window+flush pattern, D3 bridge, exact-class C441,
NINFER_SMALLT_PK gate, 1e-2 bar, |x| envelope + 682 bound) · nvfp4_amd_codec.h (:111-114 fold
contract, :131-138 order law = the declaration grammar) · math.cuh:59/:74 (half2_from_bits,
bf16x2_bits_to_f16x2_bits) · PREFILL_FRONT_2026-09-17.md §4 (M-wall arithmetic, 28 FLOP/B
balance, M\*≈8) · PREFILL_DECOMP_2026-09-17.md (chunk decomposition, P1 spec, Fix A/B/C
ownership) · nvfp4_dispatch.cpp (a16_tiled_route_enabled {128,256}, TILEDV FNV cell) ·
nvfp4_tiled_gemm_hip.cu (V0 kernel + variant machinery, switch(m) {32,128,256}) ·
text_context_impl.h PrefillOpTrace (the serving readout S2 reuses). Code line numbers are THIS
worktree's working tree (amd/tp4-cure) and may drift.
