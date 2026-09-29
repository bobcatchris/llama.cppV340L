# Doc 29 — 2× V340L final performance estimates (2026-08-21)

**Status:** ARCHIVE

Consolidates and supersedes the estimate tables in docs 01–04 with everything measured since:
the deterministic 5060 Ti engine state (doc 27 bisection, doc 28 Lever 1, **92.89 t/s verified**),
the closed objectives (Obj 2 split-KV, Obj 4b bulk INT4, k=4 on sm_120a), and the corrected
speedup ranking (doc 17, post-doc-27 corrections).

**Read doc 27 before trusting any older estimate in this folder.**

## 1. TL;DR (rev 2 — user-confirmed 400 GB/s/die, single-point estimate)

| Quantity | Conservative (GEMV eff 60%) | **Central (70%)** | Optimistic (75%) |
|---|---|---|---|
| Plain decode | ~44 t/s | **~51 t/s** | ~54 t/s |
| **MTP k=3 (production)** | ~121 t/s | **~138 t/s** | ~146 t/s |
| pp512 | ~350 | **~400–450** | ~500+ |
| Stock reference | 20–21 t/s tg (MTP on, user's current run); pp ~120 (user); doc 04 historical: 19.15 tg / 144.66 pp | | | |

**Commit number for planning: ~135 t/s MTP k=3 (range 120–145), plain ~51 t/s.** Bandwidth is
no longer the unknown — the single gating variable is **GEMV kernel efficiency on gfx900**
(assumed 70%, v100-skinny measured 72–78% on V100; ±10 pts of efficiency = ±~18 t/s MTP).
All inputs except that one are confirmed: 400 GB/s/die (user), PCIe 3.0 x8 (user), TP4
weights 4.6 GB/die (artifact), acceptance 85.6%/3.58 tok/round (model property, measured).

Arrival-day microbenches (§7) replace the 70% assumption; nothing else changes.

## 2. Hardware assumptions — and the spec contradiction

| Unit | Value |
|---|---|
| Cards | 2 × V340L, each 2 × gfx900 dies = **4 dies total**, 8 GB HBM2 per die, 32 GB total |
| Inter-die | **No fabric.** Same-card dies: PCIe bridge inside the card. Cross-card: PCIe 3.0 through the host (PHB topology, analogous to the 5060 Ti pair) |
| PCIe | **3.0 x8 per card** (user-confirmed): 15.75 GB/s/dir/card |
| CU/die | 56 (Vega 20), 32-lane SIMT — same warp structure as the skinny-kernel design |
| Compute | ~21.5 TFLOPS fp16/die (doc 04 figure) |
| **Bandwidth/die** | **CONTRADICTED IN OUR OWN DOCS — see below** |

**The contradiction (must be resolved on arrival, do not paper over it):**
- Doc 04 (original research): **483.8 GB/s/die** (1935 GB/s aggregate). This is scenario S4 and
  is physically suspect: 483.8 GB/s would require a 1024-bit interface at ~3.8 Gbps, which is
  beyond HBM2 for gfx900. It may be a card-level or doubled figure recorded per-die by mistake.
- Doc 17 (§2 rows 7/8): **~165 GB/s/die**. This is scenario S2 and is consistent with the
  closest relative part, the Radeon Pro **V340** (16 GB HBM2, 307.2 GB/s total → 153.6 GB/s per
  8 GB die if 2-die).

## 3. What transfers from the 5060 Ti (measured, deterministic, post-Lever 1)

These are the transferable ratios, all measured at HEAD `8d9835b8` (battery-green, doc 28 §7):

| Ratio | 5060 Ti value | Why it transfers |
|---|---|---|
| Acceptance / tok-per-round | **85.6% (a=2.57, 3.58 tok/round, k=3)** | model+quant property, hardware-invariant |
| verify(T=4) / plain-step | **1.19×** (34.98 / 29.3 ms) | both GEMV-bound; GEMM@T=4 streams weights once |
| MTP head / verify | **~10%** (3.5 ms of 38.5 ms round) | head GEMV scales with BW like everything else |
| Host wall / round | **~0 ms** (Lever 1: wall ≈ phases) | post-Lever-1 host path is async; assume 0–1 ms on gfx900 |
| GEMV efficiency | 99% of measured peak on sm_120a | does **not** transfer — assume **70%** on gfx900 (v100-skinny measured 72–78% on V100; 64-wide wavefronts may shave points — arrival-day microbench settles it) |

Weights per die at TP4: artifact ≈ 16.2–17.5 GB total → **~4.1–4.6 GB/die** (use 4.6 GB in
the cost model). VRAM fit: 4.6 GB weights + <0.1 GB KV/GDN (4096 ctx, split heads) + ~1.5–2 GB
arena/workspace ≈ **6–7 GB of 8 GB** — fits, with ~1–2 GB headroom. **Keep ctx ≤ 4096 for
comfort; 8k ctx will be tight.** (5060 Ti TP2 uses 9059 MB/rank incl. arena — halved at TP4.)

## 4. Cost model (single point: 400 GB/s/die, parameterized on GEMV efficiency)

`plain_step = 4.6 GB / (400 × eff) + 3.3 ms` (AR 1.5 + GDN 0.8 + attn/KV 0.5 + host 0.5)
`round = plain_step × 1.19 + head + 0.5`, head = 3.2 ms × 424/(400×eff) (5060 Ti head scales ∝ 1/BW)
`t/s = 3.58 / round`, 3.58 tok/round (85.6% acceptance, transfers exactly)

| | eff 60% | **eff 70%** | eff 75% |
|---|---|---|---|
| GEMV plain | 19.2 ms | 16.4 ms | 15.3 ms |
| Plain step | 22.5 ms | 19.7 ms | 18.6 ms |
| **Plain t/s** | **44.5** | **50.7** | **53.7** |
| MTP head | 2.3 ms | 1.9 ms | 1.8 ms |
| MTP round | 29.5 ms | 25.9 ms | 24.5 ms |
| **MTP k=3 t/s** | **121** | **138** | **146** |
| Plain roofline (100% BW) | 87 t/s | | |
| MTP vs stock (20.5) | 5.9× | **6.7×** | 7.1× |

Sanity anchors:
- **Doc 04's "~150–190 MTP combined"** assumed 483.8 GB/s/die + 65–75% eff; with the confirmed
  400 GB/s the same method gives ~120–150 — i.e. doc 04's *method* was right, its bandwidth
  input was high by ~20%.
- **Doc 28's "40–60 t/s" deployment target** was set under the low-BW uncertainty; it is now a
  floor well below expectation, not a central case. Safe minimum commit only.
- **v100-skinny multiplier anchor**: stock→tuned ≈ 4.5× on 4× V100 → ~90 t/s here. That
  multiplier depends heavily on stock quality (their stock had no working MTP); treat as a
  pessimistic cross-check, not a ceiling.

**pp (prompt processing)** is compute-bound (CU count/freq, not BW): stock ~120 (user's
current run; doc 04 historical 144.66) → **~350 (floor) / ~400–450 (central) / 500+ (optimistic)**
with large-M GEMM tiling at 30–45% of fp16 peak. Our 5060 Ti pp (31.7 t/s @ ~220 tok) is **not**
a tuned reference — prefill tiling was never optimized; pp is the weakest number in this doc.

## 5. PCIe 3.0 x8 per card — AR analysis

Per verify round: 64 layers × 10 KB (5120×bf16) = **640 KB/rank** → trivially within 15.75 GB/s.
AR is **latency-bound, not bandwidth-bound**: ~2–4 µs/op host-mapped one-shot × 64 ≈ 0.13–0.26 ms
exposed. Two port-critical consequences:
1. **Host-mapped staging (wt/cv coherency) is mandatory** on V340L, not optional — there is no
   cross-die GPU P2P. The doc 18 one-shot pattern ports directly (PCIe-ordered coherency works
   the same).
2. **Our one-shot AR is 2-rank only.** TP4 needs either a 4-rank generalization (ring over the
   4 dies: 2 same-card hops + 2 cross-card hops) or a **2-level reduce** (intra-card 2-die
   reduce → cross-card 2-die reduce). The 2-level design matches the hardware topology and is
   the recommended port plan. (AR batching every 2–4 layers — a v100-skinny technique, never
   implemented even on the 5060 Ti — cuts the op count 4× and is worth doing in the same pass.)

## 6. Port checklist — status of every known optimization

| Optimization | 5060 Ti status | V340L port |
|---|---|---|
| TP multi-GPU foundation (sharding, MTP protocol, GDN slots) | DONE, verified | Port (HIP kernels). Highest-value target; SIMT r8c4/c5 shape is native to gfx900 |
| Draft vocab (40,960 rows) | DONE, +18.3 t/s, **+20.7 pts acceptance** | Port as-is (artifact reusable). Also cuts ~0.3 GB/die VRAM — matters at 8 GB |
| MTP stabilization (prefill/snapshot/lockstep/A2 harness) | DONE | Port; keep acceptance-probe + A2 gates per execution mode |
| ColumnN lm_head sharding | DONE, +7.1 t/s | Port; 8-byte argmax exchange over PCIe |
| One-shot AR (host-mapped pinned + wt/cv) | DONE, +1.2 t/s (1.5 ms exposed) | Port — **mandatory** pattern there; 2-level TP4 design (§5) |
| OneShotArgmax batched T≤8 | DONE (post-Lever-1, in the 92.89 state) | Port |
| Lever 1: pinned accept_res, thread-local frontier, async timers | DONE, +10.3 t/s | Port (host-side, GPU-agnostic) |
| FP16 GDN recurrent state | DONE | Port; **proportionally more useful** at lower HBM2 BW |
| SIMT small-T GEMV dispatch (T≤7) + r8c5 T=5 | DONE | Port + **T-schedule analysis** — gfx900 has 254 VGPR headroom, no NVFP4 tile cliff → **k=4 legitimately reopens there** |
| CUDA graph | DONE but **racy, default-OFF** (doc 27) | **Deprioritize / delete** — hipGraph on gfx900 is less battle-tested; revisit only after the epoch/slot race fix proves out |
| Split-KV attention | CLOSED NEGATIVE (0.2 ms of verify) | **Do not port** |
| Bulk INT4 verify GEMV (Obj 4b) | CLOSED NEGATIVE (net-neutral, −4.9 pts, degeneration) | **Do not port as-is.** Selective/Hessian + quality gate only — but its relative value is *higher* there (lower BW → bigger byte-halving win) |
| k=4 draft quality (needs 46–57% marginal vs 37%) | PARKED (model-side) | Same wall; draft-quality work, not kernel work |

**Not yet explored anywhere** (open possibilities, both platforms):
1. **AR batching (2–4 layers)** — v100-skinny's biggest combined-only win; never implemented.
2. **DFlash2-style block drafter** (catalog 3.4) — separate 1.92B non-autoregressive drafter;
   the only >2×-on-top-of-MTP-class idea, but architecturally large (needs its own artifact).
3. **Selective INT4 with Hessian calibration + quality gate** — the only remaining +20-class lever.
4. **Prefill GEMM tiling** — the pp unknown on both platforms.
5. **NUMA/clock pinning** — minimal on the 5060 Ti box (single socket); may matter on the V340L
   dual-card host (pin ranks to the correct card's NUMA node).

## 7. Arrival-day verification plan (< 30 min)

1. `rocm-smi` + `lspci -tv` → confirm 4 devices, per-card topology, **PCIe link width** (expect
   x8 per user; watch for x4 degradation on some slots — x4 does NOT hurt AR (§5) but is noted).
2. HBM2 bandwidth microbench per die → **validates the 400 GB/s spec** (if it measures materially
   lower, scale §4 linearly: t/s ∝ BW at fixed efficiency).
3. GEMV efficiency microbench (our prepack layout, Q5) → replaces the 70% assumption — the only
   remaining gating input.
4. Re-run the stock llama.cpp config → confirms the current 20–21 t/s / pp ~120 baseline.
5. 2-die sync probe (small model, single card) → measures same-card vs cross-card AR latency split.
6. VRAM fit: materialize TP4 at ctx 4096 → confirm per-die headroom (~6–7 of 8 GB expected).

Then §4's eff column picks one, and the port worklist (§6) is the plan.

## 8. Bottom line

- **Commit ~135 t/s MTP k=3 (120–145 range, 6.7× stock); plain ~51 t/s (44–54).**
- The port is the whole engine at technique level — nothing about the 5060 Ti work is
  hardware-specific except the SM-count tuning, and the SIMT skinny kernels are the *right*
  shape for gfx900 by construction.
- Biggest port risks: (a) GEMV efficiency on gfx900 (±10 pts = ±~18 t/s; measured on arrival),
  (b) TP4 AR design (2-level reduce, §5), (c) hipGraph immaturity (avoid by keeping the graph
  path deleted), (d) 8 GB per-die VRAM headroom (ctx budget).
