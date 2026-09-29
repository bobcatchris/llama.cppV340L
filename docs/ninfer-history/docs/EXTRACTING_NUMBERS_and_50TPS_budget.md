# EXTRACTING NUMBERS ON THE V340L — methodology, measured ceilings, and the 50 tok/s budget

**Date:** 2026-09-17 · **Seat:** fix/perf seat · **Context:** user target >50 tok/s decode on
the NVFP4 route ("think big"). This doc is the single home for (A) how to get TRUSTWORTHY
performance numbers on this box — every trap we paid for — (B) today's measured ceilings and
kernel position, (C) the physics budget for 50 tok/s. Rows cite banked files; anything not
measured is labeled [design]/[inference] per the doc-01 convention.

---

## PART A — HOW TO EXTRACT A TRUE NUMBER (each trap cost us a row)

1. **Compile benches at the PRODUCTION -O level.** clang defaults to -O0; the first roofline
   run produced copy=27 GB/s and GEMV=0.1 GB/s — an order-of-magnitude hallucination. The
   production build uses `-O3` (verified in compile_commands.json). Harness law: any perf
   number from this tree is invalid unless the bench compile line carries the same -O.
   (VOID row noted in `results/amd/coherence/ROOFLINE_row.txt`.)
2. **HBM idles at 167 MHz and short bursts never wake it.** `rocm-smi --showclocks` shows
   mclk level 0 (167 MHz) at idle; a workload measured in that state sees a ~27 GB/s ceiling
   and — the trap — that low ceiling *trivially satisfies demand*, so the governor never
   ramps. Fix that works without root: hammer ~5 s wall-clock continuously BEFORE timing
   (bench carries this), and verify the hot state externally via
   `cat /sys/class/drm/card0/device/pp_dpm_mclk` DURING the run (level 3 = 945 MHz expected,
   starred entry is current). `--setclock/--setperflevel` are root-gated (THARM1).
3. **Cross-check events with host wall-clock once per harness.** hipEvents agreed with
   chrono in the triage, but only the cross-check proves it; events on the legacy default
   stream with unusual sync semantics are a known-flaky surface on old ROCm.
4. **Sample clocks+temps as a sideband on EVERY perf row.** Sustained load degrades rounds
   2.82x via a saturating thermal integrator (THARM1/THARM2: cold 213 ms/round → hot 606 →
   full recovery at the SAME 84 °C edge temp after 3 min idle). A row without a sideband is
   unreadable. Heavy legs: <60 s continuous or 2–3 min idle gaps.
5. **Ceiling conventions:** report the copy ceiling in the 2x convention AND a pure-read
   ceiling; a decode GEMV is read-dominated and must be graded against the READ ceiling.
6. **Units:** GB = 1e9 B; GB/s = bytes / (1e6 × ms). The first run's kMiB² formula printed
   0.0 — check every printed number against a hand division once.
7. **Name the ARM you measured.** Same kernel at different geometry/compile/clock state is a
   different number. Every row: binary sha16, compile flags, clock state, phase (prefill vs
   decode never mix).
8. **Kernel time ≠ step time ≠ tok/s.** Convert explicitly through the bytes audit (Part C).
   The biggest analysis error of the night was reading a 1.9x kernel headroom as a 1.9x
   end-to-end win — the verify forward has ~100 ms of non-GEMV time.

## PART B — MEASURED CEILINGS AND KERNEL POSITION (2026-09-17, cool die, hot clocks)

| quantity | value | source |
|---|---|---|
| D2D copy ceiling (2x conv) | **369.5 GB/s** | ROOFLINE_row.txt |
| Pure-read ceiling | **377.9 GB/s** | ROOFLINE_row.txt |
| Nominal HBM2 (doc 01) | 483.8 GB/s — **confirmed optimistic** (76–78 % achievable) | doc 01 §6.4 warned; now measured |
| Production NVFP4 GEMV (`nvfp4_gemv_hip_kernel`) | **122.9–156.4 GB/s = 32–42 % of read ceiling** (AttnInput 14336×5120 → 152.0; GdnInput 16384×5120 → 153.7; MlpGateUp 34816×5120 → 156.4; Residual6144 5120×6144 → 122.9) | ROOFLINE_row.txt |
| **TUNED NVFP4 GEMV (GEMV-TUNE, 2026-09-17)** | **211.8–295.1 GB/s = 55–77 % of read ceiling** (AttnInput → 282.0 = 73–74 %; GdnInput → 289.5 = 75–76 %; MlpGateUp → 290.2–295.1 = 75–77 %; Resid6144 → 211.8 = 55 %) — **1.68–1.88x, BIT-EQUAL** to the shipped kernel (15-problem bit-equality cell, permanent) | GEMV_TUNING_row.txt |
| v100-skinny reference | 648 GB/s = 78.5 % of their ceiling | doc 01 §1 [measured] |
| **GEMV tuning headroom** | **~1.9x** (to ~70–78 % ceiling ≈ 265–295 GB/s) | derived |
| Decode GEMV per-call (full rows) | AttnInput 0.272 ms · GdnInput 0.307 · MlpGateUp 0.642 · Resid6144 0.144 | ROOFLINE_row.txt |

Why the kernel is at 42 %: the HIP port is deliberately untuned (its header: "PERF IS A NAMED
LATER PASS") — one warp/row, K-groups strided by 32 lanes, 8-bit-scale loads per group, no
LDS staging, no XOR swizzle, no vectorized weight loads, no two-rows-per-warp. The v100-skinny
technique list (doc 01 §2) is the tuning menu.

### B.1 WHAT THE TUNING PASS FOUND (2026-09-17, one change at a time, all bit-equal)

The ISA pointed where the menus didn't: the e2m1 16-entry value table compiled to GLOBAL
memory with a dynamically-indexed `global_load_dword` **per nibble** (16 loads + a 64-bit
VGPR address computation each ≈ 110 of ~160 instruction slots per group) — the data loads
and the math were never the limiter (vectorizing them: FLAT, iters 1–2). Winning techniques,
in order landed (full table in `results/amd/coherence/GEMV_TUNING_row.txt`):

1. **e2m1 pair table in LDS** (2 KB, filled FROM `amd::e2m1_bits` — the codec stays the
   single arithmetic home): 1 `ds_read_b64` per code byte. 40 % → 60 % ceiling ALONE.
2. **e4m3 scale table in LDS** (1 KB, from `amd::e4m3_lut_decode`): replaces ~20 closed-form
   int ops/group. 60 % → 70 %+.
3. **8 warps × 2 rows/warp** (16 rows/block): two independent single-accumulator chains
   (contract-clean), x unpack and table reads shared per row pair. 70 % → 73–77 %.
4. Supporting (order-preserving): codes as one u64/group; x as 2× uint4/group; scale bytes
   via aligned u32 quad loads ((g&3)==(lane&3) is loop-invariant).
Named failures (kept out, each bit-equal): persistent grid (cross-rowgroup pipelining loss
> tail gain), 4 rows/warp (VGPR collapse, 115 GB/s), 32-row blocks (occupancy −20 %),
unroll-2 on K=5120 (helps only K=6144; per-geometry option noted, not landed).
Still banned by contract (each worth ~3–8 % [inference], needs the documented re-scope):
split-K across warps, lane→group remap, extra accumulator chains.
Landing bar (≥70 % read ceiling on MlpGateUp AND AttnInput, bit-equal): **MET** (74.8–77.5 %
and 73.2–74.2 % across confirmation runs). Residual 22–27 % on the big shapes is DRAM
mixed-stream efficiency + per-launch ramp/drain (BW scales with row count at fixed K);
in production the drain is absorbed by adjacent kernels in the stream/graph, so the
in-isolation bench UNDERSTATES the production gain.

### B.2 lm_head ROUND-TRIP AUDIT (step 2; answer, no implementation)

Read: `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1683-1749`
(`target_verify_batch_impl`) + the TP4 load path (`tp2_backend.cpp`, `row_split_geometry`).

**Does the [T,248320] logits buffer round-trip HBM per verify round on TP4?** It is
MATERIALIZED (written by `ops::linear(flat_hidden, *lm_head_, flat_logits)` at :1743, then
read back by `ops::argmax` at :1746) — but the premise overstates it twice:

1. It is the rank SHARD, not the full vocab: `n_vocab = kCfg.vocab / tp_world_` →
   [T, 62 080] bf16 at TP4 = **0.37 MB/rank at T=3** (batch=1), i.e. write+read ≈ 0.75 MB
   per round ≈ **0.02 % of the ~4.1 GB/rank round traffic**. The round-trip is NOT a lever
   (Part C's "~13 ms ×T" line was the WEIGHT stream mis-attributed to it).
2. The weight stream is already T-amortized: `flat_hidden` is [hidden, columns=T×batch] —
   ONE `ops::linear` call computes all T columns, so the 635 MB/rank bf16 lm_head shard is
   streamed ONCE per round (~1.7 ms at ceiling; the achieved rate of THIS kernel on this
   shape is the open question, not the bytes).

**Where a fused-argmax epilogue would attach** (doc 01 §2 answer): inside the lm_head arm
of `ops::linear` dispatch (a16 linear path at this shape) behind a caller flag — per output
tile keep a running (max, index) per column in registers/LDS and publish per-column winners
to a small [columns] pairs buffer; the existing downstream cross-rank argmax reduce
(E1MARGIN's "committed allreduce argmax") consumes the pairs instead of materialized
logits. Value: kills the logits allocation + 0.75 MB round-trip + one launch ≈ 10-30 µs/round
and frees the logits VRAM — REAL but small vs the 46 ms budget; v100-skinny's bigger
motivation (fatter vocab/rank) does not transfer at TP4 shards. The bigger lm_head lever,
if one is ever wanted, is quantizing the main output head (the draft head already has a q4
arm in tp2_backend) — 635 → ~160-320 MB/rank ≈ 1-2.5 ms/round — but it CHANGES logits bits
and needs its own digest gate. Neither is recommended now: step 3 (the ~100 ms launch
overhead audit) dominates both.

## PART C — THE BYTES AUDIT AND THE 50 tok/s BUDGET [inference, constants measured]

Per-layer full-model NVFP4 rows: AttnInput 14336 + GdnInput 16384 + MlpGateUp 34816 +
Residual6144 5120 + Residual17408 17408 = **89,088 rows**; bytes/row = K/2 codes + K/16
scales ≈ 2880 B at K=5120 → **~250 MB/layer all-ranks, ~64 MB/layer/rank at world=4 →
~4.1 GB/rank per verify forward** (matches the ~4.9–5.1 GB/rank materialized minus KV).

Decode budget at the SAME shape as tonight's A/B (count600, k=2):

| component | now | at 70 % ceiling | notes |
|---|---|---|---|
| GEMV weight stream (T-scaled ≈ ×1.4 at T=3) | ~35–40 ms | **~15.5 ms — LANDED** | GEMV-TUNE 2026-09-17: tuned to 265+ GB/s bit-equal (`GEMV_TUNING_row.txt`, Part B.1); the "at 70%" column is now the MEASURED state |
| lm_head stream (635 MB/rank × T, bf16 shard) | ~13 ms ×T-ish | ~5 ms | fused-argmax epilogue (doc 01 §2) can cut the round-trip |
| attention + KV reads at T=3 | part of the ~100 ms "other" | unchanged | KV tiny at short ctx |
| AR (~128 × 98–101 µs) | ~13 ms | ~13 ms | one-shot AR port = the named lever |
| launch/serialization overhead | dominant in "other" | must be crushed | whole-round graph capture exists (`capture_mtp_decode_batch`); RCCL-in-capture is the named hazard |
| **verify total** | **195 ms measured (92 % of round)** | **~40 ms required** | TIMING1 |
| accept/align/propose/bookkeeping | 16.5 ms measured | ~6 ms | already lean |

**The 50 tok/s equation (real prose, acc 0.6 ⇒ tok/round ≈ 2.3):** 50 tok/s ⇒ ~21.7 rounds/s ⇒
**≤46 ms/round** vs today's 212–286 ms. The budget above shows a physically coherent path —
*every* line must land near its roofline: tuned GEMV (measured headroom), fused/tuned head,
graphs-or-equivalent on the whole round, and the verify forward's launch overhead crushed.
v100-skinny's 86–91 tok/s on 4×V100 is the existence proof of exactly this shape of stack;
their k=15 economics do NOT transfer (M-wall, confirmed at k=2–3), but 50 tok/s at k=2–3 with
real-prose acceptance ~0.6 does not need them.

**Sequencing (each step independently measurable in the bench, then the boot battery):**
1. GEMV tuning pass vs the ROOFLINE bench (no boot needed) — target ≥70 % read ceiling.
   **DONE 2026-09-17: 73-77.5 % bit-equal, landed** (Part B.1).
2. lm_head round-trip audit → fused-argmax epilogue decision. **DONE: Part B.2** — round-trip
   is 0.02 % of round traffic; epilogue is small; weight stream already T-amortized.
3. Verify-forward launch audit (the ~100 ms "other") — **DONE: `VERIFY_LAUNCH_AUDIT_2026-09-17.md`**
   — the round is already one captured graph; ~950 nodes ≈ 2-3 ms (measured 1.56 µs/launch);
   the audit finds ~140 ms of the verify forward has NO owner (AR ~13 + GEMV ~27 + small
   ~3 are only ~43) and names instrument-first (per-op event probes at the existing
   tpv_probe sites) + the cross-rank-skew hypothesis (desktop on dev0 × 128 serialized
   collectives) as the decisive next step.
4. Whole-round graph capture on the TP path (mind the drafter-state hazard, doc 01 §5).
5. One-shot AR GCN rewrite (priced last: ≤3.6 % of round).

**Sanity anchor:** non-MTP step today = 125 ms; a full roofline-tuned stack budgets ~25–35 ms
⇒ 3–4x on decode is the hardware-honest ceiling envelope; 50 tok/s output rides MTP ×2.3 on
top of a ~40 ms round. Anything claiming 50 tok/s without (a) clocks sideband, (b) byte-diff
vs plain decode, (c) named binary sha is not a number — it's a wish (doc 01 §4 rules, in
force).

— Bases: ROOFLINE_row.txt · GEMV_TUNING_row.txt · TIMING1_row.txt · THARM1/THARM2 ·
PERF_LOG_AMD.md (head PLOG-030) · BUGTRACK E-14/E-16/E-17 · docs/optimizations/01 §1–8 ·
NIGHT_HANDOFF_2026-09-16.
