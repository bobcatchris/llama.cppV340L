# ROADMAP STATUS — the v100-skinny parts list, mapped to what's built (2026-09-17)

**The roadmap you're remembering:** `docs/optimizations/01_v100_skinny_findings_for_v340l.md`
(the audited V100-skinny parts list; `docs/optimizations_README.md` is the series index), with
the derived worklists `12_cross_repo_optimization_catalog.md`, `13_future_objectives_mtp_70plus.md`,
`30_decode_path_review…`, `31_phase_1a_1c_decode_optimizations.md`, `32_next_work_order.md`.
Doc 01 §8 is the canonical parts list: (a) SIMT dequant GEMV at 70–80 % of per-die BW + fused
argmax head; (b) small-k chain-MTP, greedy drafter; (c) graph-captured round + one-shot AR;
(d) validation discipline. This doc maps every part to its 2026-09-17 status. Evidence rules
inherited from doc 01 §7 (roofline-first, byte-diff validation, retraction discipline) are
ADOPTED and in force — BUGTRACK VOID rows and the E1 byte-diff oracle are exactly that.

## PART A — SIMT dequant GEMV + fused argmax head

| part (doc 01) | status | evidence |
|---|---|---|
| NVFP4 weights on no-tensor-core silicon (in-kernel e2m1/fp8 decode, fp16 2:1 ALUs) | **SHIPPED** — NVFP4@TP4 serves (`NINFER_NVFP4_SIMT_LANE=1` build lane; artifact `qwen3_8_27b_nvfp4.ninfer`) | every boot since C-series; coherence GREEN 56..10000 (BUGTRACK E-14/E-16) |
| e2m1 `dequant8_tm` + fp8 re-bias bit-twiddle (pure ALU port) | **SHIPPED** (nvfp4 SIMT lane kernels in-tree) | builds clean gfx900; serves |
| **Roofline: % of 483.8 GB/s per die** (target 70–80 %) | **OPEN — never measured on V340L.** Doc 01 §7 rule 1: roofline-first. One microbench session prices ALL kernel work | nothing banked |
| Fused argmax lm_head epilogue (logits never touch HBM; 0.7 GB head at 4-bit) | **PARTIAL** — `allreduce_argmax` R1 path (shard-argmax→allgather→reduce, `r1_argmax.cu`) avoids host logits, but whether the full-vocab buffer round-trips HBM per step is unverified | tp2_backend.cpp:3047; OPEN question named |
| Native 4-bit lm_head codes (no requant round-trip) | **SHIPPED for the drafter** (40960-id draft-vocab slice `qwen38_draft_vocab_ids.json`, W8G32 head, 53 MB) | boot log "draft output head ready" |
| Split-K, two-rows-per-warp tuning | OPEN (tuning options, price after roofline) | — |

## PART B — chain-MTP, small-k, greedy drafter

| part | status | evidence |
|---|---|---|
| Chain-MTP serving | **SHIPPED + measured** (first boot 2026-09-16; k-sweep k∈{1,2,3}) | PERF_LOG PLOG-011..014 |
| M-wall prediction (k≤4–7 economic, no TC escape) | **CONFIRMED by measurement** — k=2 wins real prose (10.79 tok/s, +10.7 % over k=3); synthetic k=3 marginal | PLOG-025/026 |
| Acceptance = domain curve (counting ≫ prose) | **CONFIRMED** — 0.97 counting vs 0.65 prose (k=2) | TIMING/REALK2 rows |
| Greedy drafter free points | SHIPPED (greedy serve posture) | — |
| **Byte-diff validation rule (adopt verbatim)** | **ADOPTED** — E1 oracle run; divergence chased to bf16 near-tie floor, accept path exonerated by 206-round audit tap (BUGTRACK E-17, `NINFER_ACCEPT_AUDIT`) | E15/E16 rows, PLOG-024 |
| Draft-vocab provenance trap | HARDENED (NINFER_DRAFT_VOCAB env → exec-relative → CWD, loud fallback warning; guard G5) | commit 754f50e80 |

## PART C — graph-captured round + one-shot AR

| part | status | evidence |
|---|---|---|
| CUDA graphs on gfx900/HIP | **SHIPPED, measured** — graphs-ON decode 1.70x (99.5→58.35 s same shape); flag reaches the shared target runtime forwards (`program_impl.h` consumes `use_cuda_graph`); whole-round capture exists as `capture_mtp_decode_batch` (mtp_impl.h:185-195) | PLOG-012 rows; NOTE: the "TP is graph-free" recon claim is WRONG — resolved 09-17 |
| Drafter-state-persistence hazard (their inert-graph lesson) | WATCHLIST — prove state parity after N rounds if the round is ever captured whole | doc 01 §5 |
| Custom one-shot AR (theirs: 11–15 µs) | **OPEN — named blocker in-tree**: `one_shot_allreduce.cu` deliberately NOT HIP-whitelisted (PTX inline-asm `ld.global.cv` invalid on GCN; needs the volatile-access rewrite). Current host-staged AR = 98–101 µs flat (COORDINATOR measured facts); round's AR share ≤ 3.6 % — price AFTER verify-step pricing | HipSources.cmake note; TIMING1 |
| NUMA/clock pinning checklist (worth 3–4 % on V100) | **URGENT+BLOCKED ON ROOT** — thermal throttle measured 2.82x on sustained load (sclk 1500→560-775 MHz); every set verb root-gated. Operating rules banked (<60 s bursts / 2–3 min gaps; sidebands mandatory) | THARM1/THARM2 (PLOG-018..019, PLOG-027) |
| GDN chain-spec device-side metadata (−1.4 ms/step precedent) | OPEN — unexamined on this line | — |

## Ranked next builds (roofline-first, per doc 01 §7)

1. **NVFP4 GEMV roofline microbench on gfx900** (memcpy ceiling → % BW per die) — prices
   every kernel tuning option and tells us whether the SIMT lane needs Marlin-style work or is
   already at the wall. Half a session, zero risk.
2. **lm_head HBM round-trip audit** (does the 248,320-vocab head stream HBM per decode step?
   if yes, the fused-argmax epilogue is the biggest single kernel part left).
3. **One-shot AR GCN rewrite** (volatile-access port of `one_shot_allreduce.cu`) — bounded by
   verify-step pricing first (TIMING1 says AR ≤ 3.6 % of round; verify forward = 92.2 % is the
   real target — which itself is launch-bound per the T-sweep intercept).
4. **Root grants batch** (clock pinning + persistence) — converts the thermal RC law from
   "operating rules" into "removed".
5. GDN device-side metadata port — small, precedented −1.4 ms/step.

— Written from docs/optimizations/01 (§1–8) against PERF_LOG_AMD.md head PLOG-029 and
NIGHT_HANDOFF_2026-09-16_mtp_thermal.md. Claims cite banked rows; the two recon corrections
(graphs ARE live on TP4; AR-share measurement) are recorded above.
