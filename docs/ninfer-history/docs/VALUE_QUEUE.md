# VALUE_QUEUE.md — THE RANKED QUEUE (canonical; every body works by this list)

THE RULE (principal order 2026-09-19 ~15:00): **work by value. The top executable item owns
the next available body (agent). No desk starts below an executable higher-value item.
Discarded optimizations are RE-EVALUATED at every window end — a discard is only valid after
3 genuinely distinct implementation attempts or a decisive measurement against the CURRENT
goal. Bars set for abandoned goals are VOID (the pk16 lesson: its 2.0x bar belonged to the
dead 1000 t/s goal and killed a measured 1.34x kernel).**

## BOARD STATE (2026-09-20 ~14:0x): PIVOT HOLD — reads beat stale pointers
Principal order PLOG-090: moving to a SMALLER model, NVFP4 dropped (SD-1 superseded).
All 27B-era queues are CLOSED or retired-with-model (see CLOSED SET + PLOG-084..089).
Executed-and-banked this session: kvarn promotion (65,536), G-BF-1 + chunk256
NO-PROMOTE closes, firstchunk refutation, census + sampler RED->GREEN, historical
reconciliation (98,944 bf16 TP4 era), 300k/250k/200k/TP2 refusal ladder.
DISPATCH IS BLOCKED ON THREE INPUTS (see docs/amd/NEW_MODEL_LANDING.md — the
landing recipe is complete): (1) which model, (2) target, (3) stack (ninfer vs
llama.cpp). Until then the only duty is holding the promoted line healthy.

## PROCEDURE (mandatory per principal order)
1. **Every agent gets its own git worktree**: `git worktree add /home/chris/worktrees/<desk-name> amd/main`
   then branch (`git checkout -b amd/wo-<desk>`). Build dirs are per-worktree
   (`cmake --build build-hip-amd --target ninfer-serve -j8` inside YOUR worktree). No shared
   build-tree races, no cross-desk file contamination (the d1 incident class).
2. **Every task runs from a WORK ORDER document** (docs/amd/WO_<task>.md): mission, steps,
   pre-registered gates, rollback — written BEFORE dispatch.
3. **PROGRESS LOG appended as work is made** — after every step, newest-first. A dead desk
   must be resumable from its work order alone.
4. **Infrastructure is coordinator-owned**: boot faults, windows, restores, wedges → tag
   "BLOCKER:" in the work order and keep executing. Coordinator resolves within 30 min.
5. Boot recipe: `bash /home/chris/serve_fast.sh` (40 s, NVMe model, no 10k grade). Retire =
   EXACTLY `pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"`. Restore + health-verify
   at every close. Clocks notes with every perf number.

## THE RANKED QUEUE (by expected value; re-ranked at every window end)

### LIVE
- **[EXEC] k4v4 KV quality upgrade** — TIER 1 rank 0, principal-designated. Doc: WO_Q4KV_K4V4.md.
  Port landed (amd/wo-kvarnport); RED debug in progress with audit-desk hypotheses relayed
  (prefill-scale signature, page>=1 write/read pair top suspect).
- **[EXEC] mad-mix attempt-1 INTEGRATION + G-MM-2 serving pairs** — rank 1 (fp8 slot's
  successor). Cell GREEN 1.119x/1.104x (PLOG-073); integration = 2 declared flags
  (-fgpu-flush-denormals-to-zero -ffp-contract=off) + C++ wpair in nvfp4_tiled_gemm_hip.cu;
  serving leg = PLOG-060 ordinal pairs, wall -1.5% minimum to promote (G-MM-2).

### PARKED (principal order 2026-09-19 ~15:2x)
- **Hosted-KV Phase 2** — PARKED (integration state banked in WO_KVHOSTED_P2.md with resume
  checklist). The hosted-KV FEATURE (Phase 1 results + design) remains banked; Phase 2
  integration resumes by principal decision only. NOT the same as the cancelled
  KV-capacity trade (that stays cancelled).

### CLOSED THIS WINDOW
- **F1b execution**: COMPLETE — d1 RED/GREEN closed (half-vocab sampling fix, same-class
  :3122 + tap position race), E-1 anatomy (HEAD-MISS 81-92% everywhere), gates b+a1 NO-FIRE.
  Row: W7_f1b_row.txt. d2 ngram-mod moved to TIER 2 per principal (acceptance/drafting
  tweak, not a base-stats prefill/decode upgrade).

### TIER 1 — highest value, executable now
0. **k4v4 KV QUALITY UPGRADE** — PRINCIPAL DESIGNATION (2026-09-19 ~15:1x): "Q4 KV CACHE GOES
   UP BECAUSE ITS SOMETHING YOU MIGHT ACTUALLY BE ABLE TO DELIVER. MAKE IT TIER 1." The
   deliverable is real and near-complete as a path: KVarN k4v2 (V@2-bit, ~3 bits effective)
   -> k4v4 (K4/V4 true 4-bit both planes) — PORTED AND SERVING (PLOG-074/076/078): family
   k4v2/k4v4/k5v4 all boot coherent on HIP; budget constants verified x3 (8976/11152/12240
   vs measured). HONEST VALUE (PLOG-078 repricing): the 821 MiB/die funder + short-ctx
   parity + MTP 0.99 — NOT prefill speed (cold cold-pair: no advantage at 2k; 10k
   unmeasured). Remaining gate: G-Q2 long legs (113664 paired) gated on the ~10.1k
   arena-exhaustion fix (ovf desk; real mechanism: 96 MiB work-arena ratchet + port's 3rd
   5 MiB temp — int32 hypothesis falsified). WORK ORDER EXISTS: docs/amd/WO_Q4KV_K4V4.md (phases:
   dial+ceiling -> needle/acceptance quality gates -> paired perf -> optional NVFP4-KV).
   Value: quality upgrade on the serving path + 200k-ceiling validation. Effort: config +
   validation desk (no new kernels). DISPATCH: next free body.
1. **fp8-on-ring AR** — quantize partials, ring transports half bytes, dequant on receive.
   Value: prefill AR 123 -> ~62 ms/chunk (**-6% wall**, AR column 12%). Transport door fully
   measured closed otherwise (ring byte-optimal; one-shot wedges+drifts; copy-add = same
   bytes; SDMA 3.3-3.6 GB/s). Numerics gates required (ulp-drift lesson: one-shot drifted
   acceptance 0.96->0.85 — the fp8 gate must bar acceptance -3% rel). Effort: ~1-2 desk-days.
2. **PARKED (principal order 2026-09-20: "dont worry about ngram... we need performance gains")** — was: d2 ngram-mod draft correction — not a base-stats
   prefill/decode upgrade; it is an acceptance/drafting quality lever. Mechanism measured:
   HEAD-MISS = 81-92% of misses in every register (drafter picks runner-up, margin median
   0.24-0.35; F1b anatomy, 7218 slots). An ngram/continuation lookup promoting the runner-up
   on margin+match attacks that class. Effort: pricing then implementation.
3. **CLOSED — REFUTED PREMISE (PLOG-086, 2026-09-20): first-chunk request-start cost.** The instrument leg proved no per-request gap exists on the current stack (ttft-minus-steady +-14 ms; the 918ms lines were stale-drain artifacts; the historical finding was the cured pageable-H2D stall). Lever dead. Was:
   chunk 1 of EVERY request pays 305-329 ms hot / ~900 ms cold of gap (per-request warm/
   pre-touch of the workspace/page path — NOT boot-once, NOT parity-linked). Value: ~1.8%
   of request wall at plen-2075, ~0.4% at 10k. Effort: 1 instrument boot to name the
   allocator/init events (sampler+correlator tools banked: w7_gap_clock_sampler.sh,
   w7_gap_correlate.py), then a pre-touch fix.
4. **CLOSED NO-PROMOTE (2026-09-20, PLOG-085): body-op fusion/harvest + the launch-shape family** — ksplit (2.03x cell) measured 0.0% at the serving leg (G-BF-1); gate-column residual (~116 µs/launch re-reads) named for a future pass, must clear the 1% noise floor. Was: the named non-GEMM ops column (43-74 ms/chunk: norms, gate,
   unpacks, conv) has NEVER been attacked as a fusion target (rmsnorm folds, gate+silu fuses,
   unpack-in-GEMM absorbs). PRICING NOTE (2026-09-19, arfix BODYSUM): flat fusible part ≈
   gate 11.6 + conv 4.6 + norms ~3.6 + upk 1.6 ≈ ~21 ms/chunk (~2%); gqa (+3.2 ms/chunk,
   attention-vs-depth physics) and dn (~30 ms, delta-net trio) are algorithmic terms, NOT
   fusion targets. Honest value: ~2%.

### TIER 2 — high value, gated or in queue
5. **Q3G64 artifact decode leg** — 3.25 vs 4.5 bits/weight; decode GEMV is bytes-bound ->
   ~30-40% decode candidate. Bake pipeline supports it (tools/artifact/numeric.py
   Q3G64_F16S); GEMV lane ships. Quality gate on OUR model required. Effort: bake+bench desk.
7. **[CLOSED -> see CLOSED SET] L2 stall** — 5th cure executed on the mandate; family
   closed on a decisive mechanism map (PLOG-077): platform-class
   (gfx900/ROCm 6.2/KFD-under-serve), cell clean at all worlds, serve wedges on a cured
   bin, KFD WAIT_EVENTS named. Reopen = platform change only.
8. **F1b design resume** — parked (cap); 7-step resume checklist ready.

### TIER 3 — gated on the principal or product
- **kvarn materialize B2 direct route** (structural cure for the 10.1k arena class; ovf desk
  flagged as lane decision — DECIDED 2026-09-20 by coordinator: PARKED with receipt. The
  in-place fp16 V-temp + NINFER_WORKSPACE_MIB=512 lever suffice for the current goal and are
  byte-identical below the old wall; B2 is NOT byte-identical and only pays once kvarn
  long-context is a promoted serving default. Reopen: the day promotion flips the default.)

- q4-KV / pre-dequant (principal: parked until q4-KV + prefill maxed)
- upp power-cap raise (operator-gated; lifts the whole thermal state function)
- MoE 35B-A3B (principal: off the table)
- Thermal scheduling policy (product decision)

## CLOSED SET — receipts banked; RE-EVALUATION at every window end under THE RULE
| item | closed by | the honest why | re-attempt key |
|---|---|---|---|
| Chunk ladder | clean no-gain at 512 (18.69/18.86 vs 18.43-18.80); M=256 re-leg on the promoted stack ALSO NO-PROMOTE (+0.17% ord2-3 vs 5% bar, PLOG-085) — the per-chunk costs it amortized were already cured | big-M does not pay in our fused kernel; chunk count left the request critical path | a NEW request-critical-path map (post-cures) showing per-chunk fixed costs again material |
| Graph capture | twice: no win, then degradation 20->27 s | capture-cache pathology | after AR cure changes the chunk structure |
| k=3 | round +44% beat tok/round +29% | breakeven <=76 ms not met | acceptance rises via F1b/d2 |
| Copy-add AR | same 12S wire bytes as ring (2.3% apart) | byte-counting forbids it | none (bytes are bytes) |
| fp8-on-ring AR | G-FP8-0 FAIL all arms (WO_FP8_ring_row.txt: stage-FS 3159us vs 646-771 band; RCCL ncclFp8E4M3 sums CORRECTLY on 2.20.5 but is SLOWER than bf16 1300-1394 vs 843-847; exact 6-bit I8P ring -39% vs own anchor still misses band) | **AR quant is NUMERICS-bound, not transport-bound** — even a half-byte exact-integer ring cannot meet a 1e-2 relL2 budget; falsifiers caught both directions (torn payload 4.11 vs clean 0.17) | principal numerics-budget call (~2e-2 passes I8P at kappa=1, needs G-FP8-2-class behavioral evidence) or upstream RCCL fp8 perf fix |
| V-arm integration | +5.7/+6.6% slower in serving, parity GREEN | RE-ENTRY EXECUTED (PLOG-082, desk varm 2026-09-20): closed PRICED-NEGATIVE — fold band -0.5..+1.0% vs the 1.5% bar post-mad-mix; dominant loss = deleted latency-hiding material (bench hot-L2 vs serving cold-L2) | cold-L2-class bench harness that can price latency-hiding deletions before a serving leg |
| L2 stall (armed-route submission) | 5th cure: cell matrix CLEAN at all worlds vs serve wedge on a CURED bin (E1, device-side `GPU Hang`); API audit clean; R7 ioctl = AMDKFD_IOC_WAIT_EVENTS — PLOG-077 | PLATFORM-class: gfx900/ROCm 6.2/KFD interaction under full serve context — no our-side defect exists to fix | platform change only (ROCm upgrade / gfx906+ / SR carrying R7+E1+cell matrix) |
| Gap alternation ("bimodality") | request-segmented re-analysis of arfix log (321f2967e): NO odd/even structure — even/odd gap 18-22 ms flat, walls +-0.2%, all 6 multi-chunk requests x 2 ranks | extraction artifact (interleaved-request time-sorted reads); real costs = per-request first-chunk gap (new #3) + thermal drift | none (artifact); first-chunk cost is the surviving lever |
| Co-residency | 1.02x, two desks | issue-side wall confirmed | compiler/ISA change |
| RCCL env | NULL matrix | SHM ring is the envelope | new RCCL version |
| Decode GEMV retune | already 1.15x of roofline | optimal | none |
| Draft-vocab widening | coverage 97.6-99.4% | slice not the suppressor | counter drops <90-95% (armed) |
| pk16-as-was | numerics fix ATTEMPTED and LANDED (relL2 ~146 -> 1.69-1.83e-3) but perf 0.237-0.262x vs >=1.15x bar (256 VGPR + 820 B scratch) — PLOG-073 | closed on decisive measurement, all three legs (numerics, perf, resourcing) | none — ISA change |
| mad-mix | **REDEEMED (PLOG-073): attempt-1 GREEN — 1.119x [96,128] / 1.104x [80,128] cell, relL2 1.65e-3, 122 VGPR no spill; the 0.48x kill was the inline-asm DELIVERY, not the ISA** | single attempt + void 2.0x bar; the 3-attempt mandate produced a real kernel win | — (now LIVE: TU integration + G-MM-2 serving pairs, coordinator word) |
| P2P transports | canAccess=0, twice | hardware | platform change |
| Compiler road | LLVM 17/19/22 invariant | wait structure unchanged | new major ROCm/LLVM |

### PARKED (principal order 2026-09-20 ~07:0x) — llama.cpp alternative-base probe
- Repo assessed (anng-pptk/rocm-vega-llama.cpp): NOT a llama.cpp fork — a Docker/ROCm-stack
  setup guide (gfx900 Tensile payloads from ROCm 6.3.4 into a 7.2.1 container, 6 commits, no
  llama.cpp code changes). "Layer split" = stock `--split-mode layer`. Perf claim (~58 tok/s)
  is a 26B-A4B MoE decode, not comparable to our dense-27B prefill. REAL value if revisited:
  cheap container probe of the "new major ROCm" reopen keys (L2 stall, compiler road) + a
  matched-silicon prefill datapoint. REOPEN: on better cooling + storage landing (principal:
  expected 2026-09-20), or principal word. Entry blockers solved then: no docker on box, no
  source weights for GGUF conversion, / disk 96% (EMTEC 157G is the data-root).
