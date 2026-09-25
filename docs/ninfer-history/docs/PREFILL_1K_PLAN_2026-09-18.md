# PREFILL 1K PLAN — the road to 1000 tok/s prompt processing (NVFP4@TP4, gfx900)

Written 2026-09-18 at session close; the successor roadmap to PREFILL_1K from today's 76.4.
Read PLAN_50TPS_master.md "PREFILL STATE" + GEMM front closure first — this doc builds on those receipts.

## THE PHYSICS (what 1000 tok/s means on this silicon)

- 27B model ⇒ 2·27e9 = 54 GFLOP/token ⇒ **1000 tok/s = 54 TFLOP/s aggregate = 13.5 TF/s/die sustained**.
- gfx900 **fp32 peak ≈ 10.75–12.5 TF/s/die** ⇒ fp32 arithmetic CANNOT reach 1k. Full stop.
- gfx900 **packed fp16 (Rapid Packed Math) peak ≈ 21.5–25 TF/s/die** (2:1) ⇒ 1k = **54–63 % of packed-fp16
  peak, sustained**. This is the ONLY arithmetic class that reaches the goal.
- Weight streaming is NOT the limit: 3.4 GB/rank per chunk-forward at chunk 128 ⇒ ~27 GB/s ≪ 378 GB/s.
  **1k is compute-bound, at 54 % of the second-highest precision class the hardware has.**

Honest framing: today's entire stack runs V0 at ~2.0–2.4 TF/s/die fp32 (~20 % of fp32 peak) — and we
measured, not guessed, why: latency-exposed LDS waits + register spills (LEANMAC/PIPE census). 1k asks for
**6.75×** today's effective rate in a precision class whose one prior attempt (PK) died of spills, not
arithmetic (1024 native v_pk_fma_f16/body, zero scalarization — the ISA proved the math was fine).
**1k is the everything-lands-perfectly outcome; 400–700 is the realistic band of the same program.**

## THE GATES (each independently falsifiable, in dependency order)

- **G1 — the arithmetic:** a packed-fp16 GEMM tile family that sustains ≥6.75 TF/s/die (≥54 % pk16 peak)
  under the FULL constraint set the falsifications bought us: ≤128 VGPR (0 scratch — PIPE's kill),
  ≤16–32 acc regs/thread (CSHAPE ledger), batched single-wait LDS blocks ≤20/body (the wall itself),
  x staged as fp16 pairs, fp32 coeff folds at group flush (PK's numerics were exact — reuse the pattern),
  spill 0 B (PK's kill). **PK died of resourcing, not math — the class is open; the tile is unsolved.**
  NOTE: CSHAPE's amortization kill was computed at M=128 fp32 — at pk16 the MAC density doubles, which
  re-opens the acc/thread and grid arithmetic; re-derive at M=256/512 before believing any kill.
- **G2 — the body at T=128 ≤ ~60 ms/chunk:** dn trio (510 ms) must run — the wedge is root-caused from the
  evidence row (T=51 ran CLEAN at 12 ms ⇒ the class is capable; T=128 path hangs ⇒ an index/launch-shape
  bug, not a physics wall); gate+gqa fixes landed (11.3 / 29.3); GQA cell harness fix + promote owed.
- **G3 — orchestration ≤ ~60 ms/chunk:** gap capture (chunk graph replay, 123→15–30, AR-desk sketched),
  AR at transport floor (~35–70 measured-bound), last-token lm_head trivial (route-flip class).
- **G4 — chunk ladder:** M=256/512 re-derives G1's register/grid math (fewer chunk-forwards/s amortizes
  everything; VRAM law: measure-first allocations only). CSHAPE kills do NOT transfer to M≥256 blindly.
- **G5 — the box:** sustained 54 TF/s draw at the 110 W/die cap ⇒ junction heat is a real ceiling factor
  (tonight: dies idled at 84–92 °C after wedges; FLR playbook v3 recovers them). The TF/s-vs-clock curve
  under sustained pk16 load is UNMEASURED — measure before promising 1k.

## THE LEDGER — TRIED AND FAILED (do not redo; every entry cites its receipt)

| family | predicted | measured | WHY it failed (mechanism, not narrative) | receipt | reopen only if |
|---|---|---|---|---|---|
| **HFMA2-PK** (packed-fp16 inner loop; decode small_t + prefill tiled) | 1.15–1.6× / 2.4–3.3× | **1.07× / 0.21–0.36×** | NOT arithmetic: ISA dump proved native `v_pk_fma_f16`, zero scalarization. Died of **resourcing**: 256 VGPR ⇒ 1 wave64/SIMD (¼ occupancy), 804–820 B/thread scratch spill, 478–570 waitcnt/body | V0PK_row + LEANMAC_NOTES §PK post-mortem | a pk16 tile that holds ≤128 VGPR and 0 scratch (G1) |
| **LEANMAC** (register CSE of dequant·coeff in fp32) | 3.15–3.87 TF/s | **never built** | The census model was wrong: LLVM **already hoists** (8 v_mul/body, not 64) — zero instructions available to save. Static ISA dump killed it pre-build | LEANMAC_NOTES | never (fp32 class); a *different* transform would need its own census |
| **PIPE** (batch the 12 LDS reads → one waitcnt, at V0's tile) | 1.45–1.6× | **never benched** | Diagnosis confirmed to the digit (129 waits, 80 B scratch, 128 VGPR) — but the **+8 VGPR batch funding is unsatisfiable at the 128-Cliff**: 10-cell probe matrix (unpinned loads re-sink 100%; volatile pins emit but spill 1.4–2.4 KB; launch_bounds = zero slack) | PIPE_NOTES | fundable registers (PIPE's own §4) — i.e. a smaller-acc tile (CSHAPE entry) or compiler change |
| **constrained-tiles** (acc≤16 shapes to fund PIPE's batch) | ≥1.3×/shard | **never built** | Funding CONFIRMED (42–66 regs, fits ≤112), wait coverage CONFIRMED (3–7× nest vs round trip) — killed by **scheduling arithmetic**: decode overhead inflates 18–56 % as acc drops; TN=128 at acc≤16 forces grid shapes that queue or overflow; class optimum = sweep-V1, already measured 0.715–0.907× unpipelined | GEMM_CONSTRAINED_TILES_2026-09-18.md | pk16 MAC-density doubling re-derivation at M≥256 (G1 note) — the kill was computed fp32@M=128 |
| **V4 double-buffer** (original tile sweep) | best-in-family | **0.63–0.69× V0** | Double-buffered the **staging side the hardware already hides (vmcnt)**, not the consumption side the code waits on (lgkmcnt): 131 waits REMAINED + VGPR 133 + 51 KB LDS both forced 1 wave | TILED_SWEEP_notes + LEANMAC_NOTES | subsumed by PIPE/CSHAPE rows |
| **W4 one-shot AR, pre-cure** | faster AR | **17-min silent wedge 2/2** | Lockstep liveness: monotone arrival gates + one-shot kernel queued behind unbounded RCCL on the same stream ⇒ bounded wait unreachable. FIXED by the cure (B1/B2/B3 + watchdog; wedge byte-match GREEN ×2) — but perf is a wash at decode sizes and size-dead at 1.31 MB prefill (4S vs 1.5S wire) | PLOG-049 + ONESHOT_W4_* | never for prefill sizes; decode adoption = robustness only |
| **server boots on poisoned VM state** | — | **page faults at warmup, every boot** | A device fault (wedge-abort, memory access fault) leaves KFD/GPU-VM state that survives process death AND `rocm-smi --gpureset` | PLOG-046/047/051 | PLAYBOOK v3: retire → per-die gpureset → **idle-power check (~3–8 W)** → if stuck hot: **sysfs FLR the specific card** (`/sys/class/drm/cardN/device/reset`; card1 = NVIDIA display, untouchable) → recheck → serve |
| **W3 pk16 sketch tile** (spill-free packed-fp16 GEMM at M≥256, census-first) | G1: ≤128 VGPR, 0 scratch, ≤20 waits/body, ≥54% pk16 peak | **KILLED AT ZERO GPU** (3 design compiles = the 2–3 budget): resourcing CURED vs PK (102 VGPR ≤128, 0 B scratch, native `v_pk_fma_f16` ×128/body zero scalarization) but the LDS batch cannot fund: **44 waits/body (35 lgkmcnt) vs ≤20** — group-flush tax is +50% MAC count and shape-invariant; A=32 acc spills 136–140 B at the 128 cap (2A register-dead); measured issue model at the only fundable point = **39–49% of pk16 peak < 54% even at zero stall**; grid arithmetic DOES lift at M≥256 (CSHAPE grid kill gone — irrelevant, kills are per-thread); `v_dot2_f32_f16` needs `dot10-insts`, ABSENT on gfx900 | W3_SKETCH_CENSUS_step0.md + W3_sketch_isa_v1.txt (lane amd/wo-w7-body) | (i) a compiler that emits batched waits at ≤112 VGPR (eligibility change), (ii) gfx906+ `dot10-insts` hardware — then re-derive G1 |
| **W3 pk16 sketch — REOPENED under goal change, MEASURED KILL** (same tile, NEW acceptance) | ≥2.0× V0 GF/s at M=128 (bench) | **1.34× mean (1.28–1.43) — KILL, pre-registered <1.5×.** Validated methodology (V0 reproduced SWEEP_row ±2%). The 44 waits/body do NOT hide: pk16 sustains ~4.0 TF/s/die = 38% fp32-nom / ~19% pk16 peak; the 2× MAC density nets only +34% after flush tax + stalls. Numerics FAIL (relL2 ~146) recorded as by-design-undebugged — a fix only adds work, cannot change the verdict. GEMM front now closed with a measured floor; 300+ tok/s requires gfx906 dot10 / ROCm wait-emission reopeners | W7_pk16_bench_row.txt + W7_pk16_bench_run1.log (lane) | bench closes the row — no reopen. Third and final kill of the pk16 class (PK resourcing → W3 1k-bar → W7 700-class bench) | USER ORDER 2026-09-18 ("as close to 1k as possible, 700 is okay") re-derives the bar: 700 tok/s = ~43 % pk16 peak, INSIDE the sketch's measured 39–49 % census band. The 1k kill (54 %) STANDS — this row is a new acceptance under a new target, not an overwrite. Kill-switches: census-first (VGPR >128 or scratch >0 = PK's death = STOP), rel-L2 within 2× V0's, bench <1.5× = family dies permanently | PREFILL_PLAN_REV2_2026-09-18.md §2 + W3_SKETCH_CENSUS_step0.md | bench verdict (PASS/AMBER/KILL) closes this row either way |
| **dn SIMT trio T=128 wedge** (FIX-A delta-net SIMT; was the 510 ms body owner) | dn 510 → <100 ms/chunk at NINFER_GDN_SIMT=1 | **LANDING COMPLETED WINDOW 7: PROMOTED default-ON, dn 480 → ~30 ms (×16), wall class 76→116-121 tok/s (+~55%).** Five defects total: h_chunk base token-offset-vs-chunk-index (64× OOB, both kernel sides, chunks≥2 only); prepare Phase-G v_row double-offset; Phase-G DELTA-1 W-product K column (`[s][d2]` → `[s][d_off+d2]`); Phase-G DELTA-2 block-Schur missing both unit-diagonal block factors; plus the premise fix (T<64 never ran the trio — the wedge call was its first-ever device execution). Cell (poison canaries + watchdog): pre-fix 20 FAILURES with the chunk-1 stomp at exactly the predicted byte; post-fix ALL PASS at rel-L2 ≤ 1.9e-6 on all six tensors. Serve leg at the exact former wedge geometry: 17 chunks, no wedge, parity BLUE/stop (completion 41→52 = documented summation-order trajectory drift, not a text change) | W1_DN_WEDGE_ROOTCAUSE_desk.md + W1_PHASEG_DELTA_desk.md + W7_dn_cell_*_run.log + W7_row_promote_dn.txt (lane amd/wo-w7-body) | no reopen — promoted; rollback NINFER_GDN_SIMT=0 (byte-identical mma path) |

Cross-references for wider (non-1k) history: BUGTRACK_NVFP4_TP4_INCOHERENCE.md (the bf16-intrinsic family),
PERF_LOG_AMD.md (tamper-chained), docs/amd/ONESHOT_W4_BOUNDED_WAIT_DESIGN.md (the class closure).

## THE RULES (governance — how this document stays true)

- **R1 — RECORD ON COMPLETION.** Every desk, window, bench, or falsification is recorded in THIS
  document's ledger (one row: predicted / measured / why / receipt / reopen-if) in the same session it
  completes — before push, before sleep. The PERF_LOG is the tamper chain; **this ledger is the
  redo-prevention map**. An unrecorded completion is an unfinished completion.
- **R2 — START FROM THE LEDGER.** Any new attack desk cites, in its first paragraph, the closest prior
  failure in this ledger and the specific reason it differs. A desk that cannot name the difference has
  not read the ledger — reject its spec.
- **R3 — KILL-SWITCH FIRST.** Every desk ships its zero-GPU/cheap falsification step before any build
  (step-0 census = LEANMAC; acceptance matrix = PIPE; in-band floors = PK). A design without a named
  kill condition is a wish, not a spec.
- **R4 — CLOSURE NEEDS A NAMED REOPEN CONDITION.** A falsified family stays closed until its documented
  "reopen only if" condition fires (new compiler capability, new precision class, M-ladder re-derivation).
  Reopening on enthusiasm is the waste this ledger exists to prevent.
- **R5 — MEASUREMENT BEATS MODEL.** When a model and a measurement disagree, the model is dead — record
  the correction in the ledger (LEANMAC: the census predicted 64 muls; the dump showed 8; the model died,
  not the kernel).
- **R6 — VERIFY THE CHECK.** A passing check counts only when its observable effect is verified
  ("Successfully reset" ⇒ idle-power ~3–8 W/die; "listening" ⇒ health endpoint; battery GREEN ⇒ known-answer
  grade). Proxy-pass = not-pass.
- **R7 — THE CHAIN IS THE PROOF.** Receipts live in results/amd/coherence/ + PERF_LOG (hash-chained via
  tools/guards/plog_append.py); ledger rows MUST point at their receipt file. A row without a receipt is
  a claim, not knowledge.

## WORKSTREAMS (each pre-registered, cheapest falsification first)

- **W1 — dn T=128 wedge root-cause → promote dn/gqa/ggate fully.** Evidence row W6_row_arm-a. Outcome:
  prefill ~140 class. (This is also on the critical path to ANY 1k future — the body must exist.)
- **W2 — gap capture + RCCL arms** (AR-desk F-GRAPH/F-ENV specs, both decisive checks written).
  Outcome: ~160–180 class.
- **W3 — THE LOAD-BEARING DESK: spill-free pk16 tile at M≥256.** Design from the three falsification
  receipts as a constraint set (see G1). Method: step-0 ISA census of a SKETCH compile before any bench
  (PIPE lesson); pre-registered acceptances: ≤128 VGPR, 0 scratch, ≤20 waits/body, ≥2× V0 GF/s bench
  AND ≥25 % pk16 peak; kill in-doc on any miss (PK/LEANMAC/PIPE/CSHAPE precedent — 4 falsifications cost
  ≈2 windows total; this is the cheap-artifact discipline that works). Budget: 2–3 design iterations.
  Outcome if it lands: 300–400 class at W1+W2 stacking.
- **W4 — stack the rest:** chunk ladder (G4), AR to floor, body to ≤60, D2H/orchestration audit.
  Outcome: 500–700 band.
- **W5 — sustained-thermal truth:** TF/s-vs-clock curve at 50+ TF/s draw; power-cap interplay with the
  110 W firmware lock; FLR playbook v3 as the recovery standard. This decides whether 1k is a software
  summit or a cooling problem wearing a software hat.

## RESUME POINTER (new session starts here)

- **REV2 GOVERNS THE ORDER NOW (user orders 2026-09-18): read docs/amd/PREFILL_PLAN_REV2_2026-09-18.md
  FIRST — goal changed to "as close to 1k as possible, 700 okay"; attack order re-issued by measured
  payoff: (1) GEMM pk16 bench (THE ×3.5-4 multiplier, reopened under the new bar), (2) finalize-chunk
  anomaly (1393 ms/request unbracketed, staircase evidence + code region named), (3) AR floor + capture,
  (4) W5 thermal + M=256 re-leg after GEMM. Honest projection: ~350-450 with everything landing; 1k
  stays killed (54% bar, gfx906/compiler reopeners).**
- State at window-7 close (2026-09-18, lane amd/wo-w7-body): decode **41 tok/s e2e promoted**
  (permanent); prefill **~80 tok/s chunk-wall class, band-uncontrolled** (W7 default-route leg
  2075 tok / 17 chunks / 1521.8 ms mean) with **FIX-C gqa PROMOTED default-ON** (bin
  a943225ad6da661e = ggate+gqa, BOOT_BATTERY GREEN in serving posture, serving :8100,
  serve_10k.sh flipped to it; rollback NINFER_GQA_SPLITK=0; certified 409450b8 fallback banked).
  W1 dn wedge ROOT-CAUSED + OOB class cured at cell level (see ledger row) — trio not promotable
  until the Phase-G W/U relL2 residual lands. W3 pk16 sketch **KILLED at zero GPU** (waits row;
  reopen = compiler wait-emission at ≤112 VGPR or gfx906+ dot10 hardware) — **the honest program
  band is now 400–700 with W3's arithmetic family closed; 1k requires a named reopener.**
- Next actions: (1) land the Phase-G W/U fix + cell GREEN (desk in flight, W1_PHASEG_DELTA_desk.md);
  (2) then dn acceptance legs (dn 510 → <100 bar) + GDN_SIMT promotion decision; (3) W2 gap capture
  + RCCL arms (~160–180 class); (4) W4/W5 stacking per the gates — G1 reopens only on the named
  hardware/compiler conditions.
- The falsification ledger to respect: PK / LEANMAC / PIPE / CSHAPE / W3-sketch — five families,
  five cheap kills. Every new desk ships its kill-switch FIRST.
