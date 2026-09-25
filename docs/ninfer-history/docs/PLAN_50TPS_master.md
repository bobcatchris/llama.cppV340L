# PLAN 50TPS — master scoreboard (pure MTP; ngram-mod EXCLUDED by user order)

**Goal:** ≥50 tok/s decode AND much higher prefill, NVFP4@TP4, **pure MTP only** (no
ngram-mod — struck from the path by user order 2026-09-17). **SCOPE (user order 2026-09-17):
CONCURRENCY 1 ONLY — decode and prefill at a single stream. Concurrent performance (mc≥2)
is NOT of interest at the moment; every bench, window, and lever here is single-stream.**
Serving posture: k=2, graphs ON,
`--allow-nvfp4-weights`. This doc is the single scoreboard; rows cite banked files; PERF_LOG
holds the chain. Updated live as levers land.

## THE ARITHMETIC (single-stream, real prose, acc 0.65 ⇒ 2.31 tok/round)

| | now (measured) | 50 tok/s needs | status |
|---|---|---|---|
| round wall | **60.0 ms** (window-3 flip leg, PLOG-044; was 128.4 post-SMALLT, 212–286 before) | **≤46 ms** | **decode arithmetic 2.97 tok/60.0 ms = 49.5 tok/s — the 50 goal REACHED to within 1%**; last 14 ms: tp_gemv draft-arm retune (~10) + W4 one-shot AR (9.6) |
| verify forward | **55.7 ms** (was 112.0 post-SMALLT, 195.3 pre) | ~35–40 ms | SMALLT (PLOG-033) + **LMHEAD route flip: lm_head 61.9→2.5 ms (PLOG-044)** |
| — GEMV streams | ~27 ms → tuned ~15 ms | ✔ kernel landed (76% ceiling) | DONE, in-round +4% |
| — layer-body linears (small_t) | **tuned: ~19 ms/round in-serving** (was ~63) | ✔ T=2..4 token-sharing 3.2x | DONE (SMALLT/PLOG-033) |
| — **the remaining hole → FIXED (window 3, PLOG-044)** | M2 split the tail (lm_head 61.93 @5.45 GB/s misretuned arm); window 3 FLIPPED it: A/B/C roofline B simt_r8_c4 24.4–24.7x over A small_t at T=3 (2.47 ms @137 GB/s vs 60–61 @5.5); serve leg `NINFER_LMHEAD_ARM=simt` lm_head **65.8 → 2.53 ms** (p50 2.47, max 3.46 ≤4 bar), lgather 0.67 ≤1, align 8.33→2.16 + chain_fwd 7.83→1.39 = exactly the DRAFT-FC-priced fc share; fingerprint 202/0.985/2.97 IDENTICAL + greedy text byte-identical (merge gate PASS); promote-to-default = coordinator ORDER pending (boots stay byte-identical default until then) | done at t≤4 verify widths; residual draft-side 2.16+1.39 = tp_gemv-arm retune follow-up | DONE (gated); LMHEAD_flip_row.txt + W8_LMHEAD_row.txt |
| — AR (128 ring collects) | ~13 ms | ~3 ms | W4 one-shot built, needs device test |
| bookkeeping (accept/align/propose/chain) | **~4.3 ms** post-flip (was 16.3) | ~3 ms | align 2.16 + chain_fwd 1.39 + propose 0.34 + chain_head 0.33; residual = tp_gemv-arm retune |
| acceptance | 0.65 | ≥0.65 (pure-MTP levers only) | coverage meter ready; vocab slice = only lever |
| aggregate alternative | 10.8 ×1 | 4 streams ×3–3.5x ≈ 35–38 → +tuning = 50 | bench ready, untested |

**Prefill (co-equal goal):** 11.3 tok/s @10k mean (cool), ~14.5 short; plen-2000 class runs **75.5 tok/s**
(TILED, P1 probe) — ON the 75.7 anchor. **P1 decomposition LANDED (PLOG-044, PREFILL_OPTRACE_row.txt):
per-128-tok-chunk wall 1642.3 = gemm 798.8 + ar 70.7 + body 645.1 + gap 122.7. THE LEVER IS NAMED: the
in-layer NON-GEMM/NON-AR body (645 ms/chunk, 39% — GDN mixer/conv/norm/activation class) owns the chunk;
GEMM is warm at 799 (49%, under its 1100 reopen bar), AR 71 (4%), and the eager GAP is only 122.7 (7.5%) —
chunk graph capture is DEMOTED to that 7.5% class (NOT the 100–130 tok/s the sketch predicted). Next
prefill lever: non-GEMM op kernel efficiency; the large-M NVFP4 dequant GEMM (doc 01 Part A) remains the
10k-row class lever.**

## DONE (banked, each battery-gated + hash-chained)

| # | win | measured | receipt |
|---|-----|----------|---------|
| 1 | coherence | 16/16 ladder + 10k PASS | BUGTRACK E-14/E-16 |
| 2 | MTP k=2 serving posture | 3.09x vs non-MTP | PERF_LOG PLOG-011..014 |
| 3 | graphs ON (drop --no-cuda-graph) | 1.70x decode, 1.10x prefill | PLOG-011..013 |
| 4 | causal-conv + bf16-low fixes guarded | suite green | guards/BOOT_BATTERY |
| 5 | **GEMV tuned** | 42%→76% ceiling, 1.8x kernel | PLOG-030, GEMV_TUNING_row |
| 6 | N7 un-gated (--allow-nvfp4-weights) | 3-arm proof | PLOG-029 |
| 7 | thermal law + sidebands | 2.82x swing characterized | PLOG-017..019, 027 |
| 8 | W4 one-shot AR + gfx900 port | compile-clean, canonical-order bit-exact | 2688e6a29, 4967b90cd |
| 9 | instrumentation (B1 phases, optrace analyzer) | self-tested | PLOG-011, ebad7e1c7 |
| 10 | **SMALLT: verify-width kernel identified + tuned (T=2..4 token-sharing) 3.2x, bit-equal** | round -39.6%, decode +66% | PLOG-033, SMALLT_row |
| 11 | **TILED prefill GEMM landed + dispatch-routed (M≥64)** | prefill 37.3 → **75.7 tok/s** (1.8x serving, 6.7x vs 11.3@10k anchor); race fix certified | PLOG-034, TILED_DATUM, 6feebd190 |
| 12 | **10k meltdown PASS via durable one-command serve** (`/home/chris/serve_10k.sh`) | 10k grade finish=stop 'BLUE' PASS, sustained boot, full clocks | PLOG-037 |
| 13 | Tile-shape sweep V0–V4 **MEASURED (PLOG-039): V0 (shipped) wins ALL 4 geometries by 15-41%; prediction V4>V3>V0 falsified, falsifier (a) fired — V0 stays, tile-shape family CLOSED** | relL2 0.00 all 20 rows; V0 3083/2756/3103/2366 GF/s (28.7/25.6/28.9/22.0% of 10.75 TF/s) | SWEEP_row, PLOG-039 |
| 14 | HFMA2 packed-fp16 arm **MEASURED (PLOG-040): numerics GREEN (rel-L2 1.1-1.3e-3, 8x under bar), perf RED vs adoption bar (mean 1.07x, 1.02x on MlpGateUpW4) — S5 stays default, lever dead-as-built** | 5 W4 gates PASS rc 0; cvt tax ate the packed-FMA win; follow-up named (LDS-activation staging) | PK_row, PLOG-040 |
| 15 | **M1 verify decomposition (decisive cell, coordinator add): verify 117.41 = layer loop 62.20 (bodies 52.60 + in-loop AR 9.60) + OUTSIDE-TAIL 55.21 ms/round; tail UNCHANGED from pre-tune (54.3) while the loop fell 141→62 — the #1 owner is now NAMED: post-layer-64 tail (lm_head/logits/sampling/D2H)** | count600 204 rounds in-class; OPTRACE COMPUTE verdict (skew 0.010 ms, gap 0.028 ms); 4 ranks identical; clocks TOP 113/113 | VERIFY_DECOMP_row, PLOG-041 |
| 16 | **WINDOW 3 (PLOG-044): LMHEAD route flip GREEN end-to-end — roofline B/A 24.4–24.7x at T=3; serve leg lm_head 65.8→2.53 ms (≤4 gate), fingerprint 202/0.985/2.97 + greedy text IDENTICAL, round 134.5→60.0 ms = 49.5 tok/s decode arithmetic (50-goal reached within 1%); bin 6c8ae21399516750, env-gated (default boots byte-identical until a promotion order); P1 prefill: gap 122.7 ms/chunk — winner = non-GEMM body 645, graph capture demoted to 7.5%** | A 60–61 ms @5.5 GB/s vs B 2.47 ms @137 GB/s; C loses; DRAFT-FC 23.7x; T=33/40 unflipped | W8_LMHEAD_row + LMHEAD_flip_row + PREFILL_OPTRACE_row, PLOG-044 |

## MISSING (the honest list, in fire order) — UPDATED after HOLE1 (9f6f81dfd)

0. **HOLE1 MEASURED (2026-09-17, HOLE1_row.txt): H1 skew ACQUITTED (ranks identical <0.1 ms);
   the hole is KERNEL-EXEC EFFICIENCY — the 64-layer body implies ~33 GB/s weight stream in
   serving vs the 265-280 GB/s isolated tuned ceiling (~8x). Uniform ~1.94 ms/layer, T=3 and
   T=1 alike (width-independent). AND: draft-vocab CWD trap measured at +142 ms/round when it
   falls back (propose 0.34->59.5 ms) — MANDATE NINFER_DRAFT_VOCAB=<abs> on every TP4 boot
   line; rows whose log lacks 'loaded 40960 draft vocabulary IDs' are incomparable.**
   **+ SMALLT PHASE 1 (2026-09-17, SMALLT_row.txt): the verify-width kernel is now PROVEN —
   at T=3 every layer projection (all five W4 geometries) executes `nvfp4_small_t_hip_kernel`
   via the A16Only SIMT binding -> tp_gemv -> nvfp4_dispatch -> launch_a16; the tuned decode
   GEMV never runs at T>=2. Serving-side [SMALLT] env-gated trace landed (NINFER_SMALLT_TRACE).**
1. **SMALLT TUNED + LANDED (2026-09-17, PLOG-033): nvfp4_small_t_tuned_kernel (T=2..4,
   token-sharing: 1 warp = 1 row x all T, weights stream ONCE) + S1/S2 in the legacy kernel
   (T=5..32 prefill chunks). 3.2x kernel at T=3 (52-57 -> 147-192 GB/s, 14-15% -> 39-50% read
   ceiling), BIT-EQUAL vs pre-edit ref (5 W4 x 2 passes). Same-boot A/B count600: round
   212.68 -> 128.43 ms (-39.6%), verify 194.01 -> 111.95 ms, decode 14.0 -> 23.2 tok/s (+66%),
   prefill 22.4 -> 37.3 tok/s. Byte-identity GREEN; BOOT_BATTERY GREEN; bin
   1e5745f9f8febcc3. 60%-of-ceiling bar honestly NOT met at T=3 (3x ALU/weight-byte issue
   wall; banned levers needed — SMALLT_row.txt honesty block). Next levers for the 46 ms
   round: W4 one-shot AR (item 2) + the remaining ~80 ms layer-body non-linear ops + TILED
   prefill GEMM greenlit (prefill bench: 3.5x PROD GF/s at M=128, relL2 PASS).**
2. **W4 one-shot device test** (ready): gate flip behind NINFER_TP_ONESHOT_AR=1, A/B vs ring,
   byte-identity per re-scoped E-17. ~10 ms/round (14.5 measured -> 2-4).
3. **Prefill large-M GEMM — LANDED + TUNING CLOSED (2026-09-17, PLOG-039): V0–V4 sweep fired,
   V0 wins all four geometries by 15-41%; falsifier (a) fired (LDS occupancy beats barrier/overlap),
   and the smaller-tile follow-up knob (V1) also lost 16-26% — tile-shape family closed, V0 ships.
   Prefill stays 75.7 tok/s; next prefill work is NOT tile shapes.**
4. **Acceptance coverage measurement** (pure-MTP lever): NINFER_VOCAB_COUNT_DIR on a real
   workload; raise the 40960 slice only if measured <90–95% (+130 MiB/rank, priced).
5. ~~**Concurrency bench** (ready): mc=2/4, k=2 forced (k=3 throws at mc=4), M-wall read.~~
   **DE-SCOPEd by user order 2026-09-17: FOCUS IS CONCURRENCY-1 ONLY (decode + prefill).
   Concurrent performance is NOT of interest at the moment.** mc=2/4 N-arms stay parked
   (script + KV posture findings remain banked from PLOG-036 for whenever scope reopens).
6. **Attention/KV pricing** (inside hole, may need its own trace).
7. Root-grant items (clock pinning): worth ~1.3-1.75x on hot-machine rows.
8. **VRAM-law refuse-constants (owner desk, cross-line): 4 banked (VRAM_LAW_AUDIT_2026-09-17.md,
   2b4fe8dfe) — V1/V2 tp2 reserve 1536 MiB (conc>=2 only = de-scoped path, but single-canonical-home
   region: needs cross-line coordination per anti-resurrection rule), V3 types.h:141 1024 MiB KV
   headroom (resurrection-adjacent twin of WO-VRAM-1 #1), V4 program_impl graph-capture fixed
   allowance (both off the AMD TP2 boot path). Legal replacements spec'd in audit SS3. New
   refuse-constants are now CI-guarded: tools/guards/check_vram_refuse_constants.py (selftest GREEN).**
0b. **WINDOW-3 SHEET — DELIVERED 2026-09-17 (PLOG-044): lm_head route flip GREEN end-to-end (bench
   24.4–24.7x, serve parity 202/0.985/2.97 + text identical, lm_head 2.53 ms, round 60.0 ms =
   49.5 tok/s decode arithmetic) + P1 prefill gap number 122.7 ms/chunk (winner: non-GEMM body 645,
   graph capture demoted to 7.5%). Owed follow-ups: PROMOTE-TO-DEFAULT order (NINFER_LMHEAD_ARM
   default is still smallt; gated boots proven); tp_gemv draft-arm retune (align 2.16 + chain_fwd
   1.39 residual; the ~54-56 GB/s serving class at [14336..34816,5120] vs 137-220 GB/s proven
   achievable); W4AR GREEN leg (amd/oneshot-ar @7dd270662 — 6 device cells owed incl. the 2/2-wedge
   repro boot expected GREEN-or-loud<=15s); clock-droop control-variable follow-up (window-3
   observation: 60 ms rounds no longer trip the instant droop — the droop tax now lands mainly on
   long prefill/bench bursts).**
10. **M2 tail per-op probe (2026-09-17, next window): M1 (PLOG-041) named the round's #1 owner —
   55.21 ms/round verify tail OUTSIDE the layer loop (lm_head/logits + sampling + D2H + tail
   collectives), UNCHANGED by the SMALLT loop tune. Fire the tpv_probe env-gated arm (one GDN +
   one FULL layer + tail ops) and split the 55.21; fusion/graph-capture of the tail is the
   lever class behind it.**

## PREFILL STATE (2026-09-18, window 6 / BODYFIX — post PLOG-050)
**GEMM FRONT CLOSED (2026-09-18): three families falsified with receipts — HFMA2-PK (0.21-0.36x measured),
LEANMAC (compiler already hoists), PIPE (batch unsatisfiable at the 128-VGPR cliff), constrained-tiles
(funding+coverage feasible but amortization+grid kill all shapes; optimum = V1 already measured slower).
V0 at 2.0-2.4 TF/s is the honest kernel-class state; a compiler upgrade or a different arithmetic class
reopens it (conditions named in GEMM_CONSTRAINED_TILES doc). Prefill road: body promotions (window 6) +
gap capture (~123->15-30) + RCCL arms (0-33) => the ~240-330 wall class stands as the machine ceiling
without new arithmetic.**

**Current: ~76.4 tok/s @2k class (arm-B clean band: wall 1674.7 → 1579.0 ms/chunk, client 29.34 → 27.20 s) /
35.3 @10k.** Window-6 verdicts: **gate fix PROMOTED default-ON** (FIX-B SimtColTile: gate 80.4 → **11.31 ms**
4-rank, x7.1, parity byte-identical, bin 93515c1f5854b346; "0" rolls back); **gqa AMBER** (FIX-C split-K S=4:
57.2 → 29.26 ms = x1.95, <20 acceptance missed, S=2 A/B worse 31.35 — stays env-gated NINFER_GQA_SPLITK);
**dn RED BY WEDGE** (FIX-A SIMT trio: T=51 clean, T=128/chunks=2 deterministically hangs 3 dies — 510 ms STILL
ON the body). Op-level split (baseline 710 body / 1675 wall): gemm 821 + body 710 (dn 513 / gate 80→11.3
promoted / gqa 57→29.3 gated) + ar 67 + gap 124. Next: desk cures the GDN T=128 wedge (+the flaky combined-boot
warmup wedge + the GQA cell's NaN harness, PLOG-050); then gqa acceptance pass, gap capture, RCCL arms, gemm
rewrite per the standing ladder. **WINDOW-6 HARDWARE NOTE: die 1 in a reset-proof 82 W / 92C-junction idle
state (firmware class, host-reboot-only) — the window's GPU side ended at the incident; serving restore +
any further timing legs are BLOCKED until the host reboots (W6_THERMAL_INCIDENT_row.txt).**


## LAWS UNCHANGED

Every landing: RED/GREEN cell or byte-identity, battery GREEN, banked sha16, phase-axis rows,
clocks sideband, hash-chained PERF_LOG. ngram-mod: documented as EXCLUDED (ACCEPTANCE_LEVERS
doc), never counted toward 50.
