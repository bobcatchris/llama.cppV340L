# REMAINING_ITEMS.md — everything left to try, including retries (clean list)

Companion to COMPLETED_OPTIMIZATIONS.md. Full context per item lives in BACKLOG.md and the
linked receipts. Statuses: LIVE (desk in flight) / READY (spec'd, next) / GATED (user or
decision) / DORMANT-RETRY (failed once; named retry key — do not attempt without it).
Last updated: 2026-09-19 ~08:50Z.

## LIVE — desks in flight

2. **Decode landing: Branch-B + k=3** — (a) env-gated sync removal at
   one_shot_allreduce.cu:1033, sized 3-4.5 ms of the 59 ms cold round; (b) k=3 draft-depth
   pricing (data-backed: acceptance 0.96, tok/round 2.91, PLOG-066). Window granted.
   [WO_DECODE_LANDING.md]

## READY — spec'd, next in queue

3. **Hosted DRAM KV Phase 2** (principal order) — tree integration behind NINFER_KV_HOSTED
   (off by default); Phase 1 bar already cleared (parity GREEN, 6.5 GB/s, prefill tax
   1.3 ms/chunk ~60x under bar; decode tax 32.25 ms/tok/200MiB documented).
   Receipts: WO_KV_HOSTED_desk.md, W7_kvhosted_row.txt. Starts when the decode desk's
   window frees.
4. **AR quantization (prefill-only)** — DESIGN DELIVERED (WO_AR_FUSION_design.md, option C):
   fp8e4m3 quantized AR via a second one-shot instance (+21 MB pinned/rank), routes centrally
   at tp_group.cpp:381. Projected: AR column 123 -> ~50-53 ms, wall -5..-7.5%. Implementation
   desk 10-12 h, lands ON TOP of Branch-B's committed state (same file region). Gates Q0-Q3
   pre-registered (microbench <=0.55x ring; correctness cells w/ both-direction falsifiers;
   ordinal-paired A/B bars: ar -30%, wall -3%). NEXT DISPATCH after decode-landing lands.
4b. **AR count-halving (fusion)** — FALSIFIED AT DESIGN LEVEL: the two per-layer ARs sit on
   opposite sides of a nonlinear seam (rmsnorm + MoE GEMMs between them) — no legal batching
   position exists (WO_AR_FUSION_design.md). Closed without code. The residual-axpy fold
   (128 axpy kernels/chunk, ~0.2-0.3 ms) arrives free inside option C.
6. **Gap bimodality attribution** — post-cure gap alternates 15/69 ms per chunk (A-prime
   leg's +70 ms signature); one OPTRACE boot reads it free. Blocking clean measurement of
   any 5-8%-class prefill arm (the V-arm lesson).
7. **F1b drafter quality** — EXECUTION CLOSED (2026-09-19 ~14:5x): d1 RED/GREEN CLOSED
   (guard + device cell 6/6, both-direction falsifier, bins fdd9b68e0cf14421/4fb2d1ee4ec73890;
   served effect at TP4 nil — slice 95% lower-half). E-1 anatomy (7218 slots): HEAD-MISS
   81-92% of misses in EVERY register (drafter picks runner-up, margin median 0.24-0.35);
   gates b (vocab top-up) and a1 (head bf16) both NO-FIRE by their pre-registered bars.
   REMAINING LEVER NAMED: **d2 ngram-mod draft correction** (targets the HEAD-MISS class
   directly) — needs pricing desk before implementation.
   drafter = 1 MTP layer, W8G32 int8 unconditionally (not NVFP4); proposals = pure argmax;
   draft vocab = runtime row-slice of lm_head. Execution path: E-1/E-2 error analyses
   (zero build, NINFER_D22_PDBG tap, 3-class taxonomy: OUT-OF-SLICE/NEAR-MISS/HEAD-MISS)
   -> d1 BUG FIX (tp2_backend.cpp:3182 sampling-branch accept sees HALF the vocab at
   world=4 — greedy unaffected; one line + RED/GREEN cell) -> a1 head-slice W8->bf16
   (gated on E-3 flip-rate >=2%) -> b reasoning vocab top-up (gated on E-1 OUT-OF-SLICE
   >=30%). top-2/tree drafting FALSIFIED by arithmetic (verify owns 87% of round).
   [E-1/E-2 + d1 desk IN FLIGHT]
8. **Decode small levers** — align/embedding ~1.15 ms/round + non-GEMV ~0.8 ms. After #2.

## GATED — needs the user or a decision

9. **q4 KV cache migration** — user's stated next milestone; unlocks ~200k context AND is
   the decode KV-decay killer (27 -> 40+ t/s path at long generation).
10. **VRAM-funded pre-dequant** (kills the measured 52% conversion tax) — PARKED by user
    order; unlocks only when prefill is maxed AND #9 landed.
11. **upp power-cap raise** (110 W -> higher) — operator-gated (can freeze the box); lifts
    the entire thermal state function (104 tok/s cold-start was AT 110 W).
12. **Thermal scheduling policy** — heavy requests >=3 min apart (measured drain rule) or
    accept the state function. Product decision, free to apply.

13b. **LEANMAC fp32 op-count family** — KILLED AT CENSUS, zero GPU (2026-09-19 ~09:10):
   V0's shipped binary already contains the design (LLVM LICM hoists the coeff·e2m1
   product; measured 1.52 ops/MAC vs LEANMAC's modeled 1.50 — the 152-op premise was
   source-static and wrong about the binary); the restructured arm trips its own
   resourcing bar (256 VGPR + 884 B scratch + 611 waits — the madmix death class).
   BONUS: V0's 0.28-0.34 gap-to-model attributed to the LDS-port/barrier/wait floor.
   RETRY WHEN: a compiler that stops hoisting (never), or LDS-floor relief (hardware).
   Receipts: WO_LEANMAC_census_row.txt (+ V0 body extract = primary receipt).

## DORMANT-RETRY — failed/closed with a named key

13. **V-arm integration** — CLOSED NO-GO (2026-09-19 ~08:30): bench 1.1x did NOT convert
    (paired: +5.7/+6.6% slower within-pair; 10k wash; parity GREEN 12/12; route proof
    confirmed). Arm stays in-tree behind NINFER_VPERM_ARM (unset = byte-identical).
    RE-ENTRY (banked in W7_varm_ab_row.txt): clean boot phase + working PREFILL-SUM +
    clocks sidebands + >=3 interleaved pairs + gap-bimodality (#6) attributed FIRST —
    the observed +6% equals that unattributed mode's amplitude.
14. **bf16-KV dial** — CLOSED BY STRATEGY (principal decision): dequant removed is noise at
    short context while decode bytes grow 4.1x permanently.
15. **Clock pinning** — hurt sustained 1.8x pre-cure. RETRY WHEN: power cap raised (#11),
    or one re-leg under the cure config.
16. **Graph capture** — RETRY WHEN: GEMM wall shrinks >=1.4x (LEANMAC A1 passing would meet
    this key).
17. **Compiler road** — RETRY WHEN: new major ROCm/LLVM ships (30-min CPU census).
18. **P2P / peer-write** — RETRY WHEN: hardware changes (canAccess=0, measured twice).
19. **RCCL env knobs** — NULL matrix; RETRY WHEN: new RCCL ships.
20. **Draft-vocab widening** — AUTO-RETRY TRIGGER ARMED: coverage counter on every boot;
    widen if true-stream coverage drops <90-95%.
21. **pk16 / mad-mix / swizzle / K-split / co-residency** — killed with receipts. RETRY
    WHEN: gfx906-class hardware or a compiler that changes wait-emission/VGPR allocation.
22. **Decode draft-arm GEMV retune** — NO-GO: already 1.15x of roofline. Optimal; closed.

## RESOLVED INCIDENTS (record so the class never repeats)

- **Canonical bin boot fault (GQA width=53 warmup page fault, random dies, 0/6)** —
  RESOLVED 2026-09-19 ~08:45; RECURRENCES #2 (08:00.0, 2026-09-19 ~19:3x — cleared by
  gpureset alone) and #3 (0d:00.0, 2026-09-20 ~02:1x — gpureset did NOT clear; FLR-all-4
  + settle cleared it, NO reboot needed). PLAYBOOK v4 (supersedes v3 ordering):
  kern.log capture -> retire -> gpureset the faulted die -> try serve ->
  FLR ALL FOUR V340 CARDS (PCI-VERIFIED — drm cardN numbering SHIFTS ACROSS BOOTS:
  2026-09-19 card1 was the NVIDIA card, by 2026-09-20 card2 was; go by PCI address,
  NVIDIA display = 0000:15:00.0 TODAY, never by remembered index) -> settle ~1 min ->
  serve. Reboot ONLY if FLR fails. Post-FLR dead sensors (0.0 W) are normal — wait for
  re-init before declaring death. Standing rule unchanged: do not retry through the fault.

## READY (added 2026-09-19 ~09:30Z)

23. **Q3G64 artifact empirical leg (decode)** — the bake pipeline natively supports Q3G64_F16S
    (tools/artifact/numeric.py) and the q3 GEMV decode lane ships (roundtrip_q3q2.py). At
    3.25 bits vs NVFP4's 4.5, decode weight bytes drop ~28% -> at the measured GEMV roofline
    (1.15x of floor) that is a ~30-40% decode t/s candidate. PREFILL at q3 needs a q3-decode
    tiled kernel (NVFP4-specific today) — measure decode first; prefill stays NVFP4 unless a
    q3 tiled kernel is separately justified. Quality gate: Team Green ships Q3G64 as a serving
    line, so the format is production-class elsewhere — still gate on OUR model's perplexity
    or golden set before any promotion. [QUEUED — dispatch when a slot frees; bake via
    tools/convert/qwen3_8_27b + roundtrip_q3q2.py patterns]

## BLOCKER (2026-09-19 ~10:3x) — ONE-SHOT AR ROUTE WEDGES AT WARMUP POST-FLR/REBOOT

Signature: oneshot-arm boots (NINFER_TP_ONESHOT_AR=1) wedge at warmup ~4/5 attempts across
TWO bins (183da007 + LKG c618d356 — no defer edit involved); watchdog full matrix on one
catch: world=4, wedged_rank=0, gen=0, slot=-1, inflight_ms=0, hb_seq=1 — rank0 stuck at the
FIRST crossing with nothing in flight = host-side handshake deadlock, NOT a page fault
(dmesg clean post-reboot). Canonical ring-posture boots 4/4 healthy.

IMPACT CHAIN: gates Branch-B validation (the edit is dead code on the ring route) AND the
AR-quant implementation (its foundation is a second one-shot instance). NEXT DISPATCH:
root-look desk on the KAR-v2 first-crossing handshake (one_shot_allreduce.cu — note the
code's own flagged hazard at :661-665 slot-recycling/gen ambiguity; docs/156 + PLOG-049
lineage). Until resolved: Branch-B stays banked-unpromoted (parity GREEN, A/B untestable),
k=2 confirmed optimal (k=3 closed: acc 0.96->0.92, round 59->84.8 ms, t/s -10.5%).

## STRATEGIC UPDATE (2026-09-19 ~12:00, llama.cpp desk findings — RE-OPENS TWO ITEMS)

- **M-scaling premise OVERTURNED by external evidence**: llama.cpp prefills the WHOLE prompt
  as one GEMM (M=512-2048) through dequant+rocBLAS Tensile fp16 on the same dies and lands
  2.08 TF/s/die END-TO-END (159.66 t/s 27B dense) — equal to our GEMM-ONLY rate at M=128.
  Our M=256 CLOSED-SKIP (§8) was decided on OUR TP-heavy pipeline where per-chunk AR doubled;
  the premise "bigger M doesn't pay" was TP-contaminated. RE-LEG: chunk 256/512 ladder on the
  cure config with honest instruments (2 boots) — dispatch when the window frees.
- **TP4 itself is now a question, not a law**: llama.cpp runs NO tensor parallelism on AMD
  (row-split removed upstream; layer-split only) and its multi-die scaling is weak (+3-24%)
  — the win is no per-layer AR at all. At NVFP4, 27B weights = 15 GB = 3.8 GB/die —
  LAYER-SPLIT (pipeline) FITS our VRAM, which is why TP4 existed (bf16 54 GB forced
  sharding; NVFP4 removed the pressure). Feasibility study queued (design desk).
- Decode: llama.cpp dense-27B decode = 19 t/s vs our 27-41 (MTP wins); their MoE 30B-A3B
  decode = 53.4 t/s (3.3B active) — the MoE model option also compounds here.

## PATH A CONFIRMED (2026-09-19 ~12:15, layer-split feasibility verdict)

Layer-split: FEASIBLE (StageSpec pipeline axis already in-tree, bindings.h:210-222) but NOT
worth it for 27B@2k — pipelined wall 16.5 s vs TP4 15.8 s at 2k (AR saving < pipeline
serialization at C=16); crossover ~2.7k tokens; decode +13-16% is the one genuine win
(27-41 -> ~30-46 t/s). PARKED. Re-entry triggers: (1) 35B-A3B MoE revival (principal has
MoE off the table), (2) >2.7k-token steady-state workloads, (3) decode-latency product
priority. Effort if ever: 11-19 desk-days. Receipts: WO_LAYER_SPLIT_feasibility.md.

PATH A = the program (5-7 desk-days total, composite bar >=150 t/s @2k, decode >=27):
  PA-1 chunk ladder 256/512 (2-3 d, needs window — queues behind root-look repro phase)
  PA-2 AR cure: copy-add microbench (GATE-Q0, no window, dies 2/3) -> fp8 if one-shot
       route is cured by the root-look, else copy-add carries it
  PA-3 prefill graph capture retry (2-3 d, pre-registered gates — the launch-bound
       premise shifted after the cure + AR changes)

## PARKED (principal order 2026-09-19; park executed cleanly)
- **Hosted-KV Phase 2** — PARKED POST-INTEGRATION: integration COMPLETE and committed
  (NINFER_KV_HOSTED / NINFER_KV_HOSTED_TOKENS arms, OFF by default, budget seam, BF16-only
  plan guard; bin 9bc7d7e1f764e7cb banked; whitelist parity guard 220/220; independent
  serving evidence via the F1b desk's control boot GREEN). Gates G-KV1/2/3 NOT RUN (parked
  before any hosted-arm boot); G-KV4 partial-positive (2 clean OFF boots byte-identical).
  RESUME CHECKLIST (6 steps) in docs/amd/WO_KVHOSTED_P2.md. FINDING FOR RESUME: the served
  line's KV reserve model is BF16-tier (34,816 B/t/rank) = ~2x the actual k4v2 pool — the
  reserve-vs-actual discrepancy gates the true VRAM-funder math. NOT the same as the
  cancelled KV-capacity trade (that stays cancelled). The hosted-KV FEATURE stays banked;
  Phase 2 validation resumes by principal decision only.
