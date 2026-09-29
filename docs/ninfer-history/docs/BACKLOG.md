# BACKLOG.md — the canonical "what we should try" list (Team Red, AMD V340L)

Living document. Owner: session coordinator. Updated at every window end (and whenever an
item changes state). Everything here is either READY (spec banked, waiting on capacity/go),
GATED (needs the user or a decision), or DORMANT (failed with a named reopen condition —
never re-litigate without its key). Completed work lives in docs/amd/PERF_LOG_AMD.md
(PLOG chain); this file is only the FORWARD list. Last full revision: 2026-09-19 ~05:15Z.

## A. READY — spec'd, waiting on capacity or a go

| # | item | expected size | spec/receipt pointers | notes |
|---|---|---|---|---|
| A1 | **V-arm integration** (v_perm LUT kernel into src pk16 path) | ~+5-8% prefill best case | REV2 §8 SCHEDULED row; fixes in w7_repack_bench.cu + w7_vperm3_bench.cu; PLOG-056 | MANDATES: carry both bug fixes; ordinal-paired serve-leg (PLOG-060 design). Dispatch at agent reset ~12:42 2026-09-19 |
| A2 | **AR fusion** — fuse per-layer mixer+mlp collectives into one | up to ~6% of chunk wall (AR = 123 ms/990) | PLOG-062/063/065; W7_curetl_row | Engineering desk: layer-phase restructure so ONE collective covers both partials. Numerics = pure reordering (same values, same summation order per element? VERIFY — gate on parity cell) |
| A3 | **AR quantization** (bf16 -> fp8/int8 messages, halve bytes) | up to ~6% of chunk wall | PLOG-063/065 | Numerics-risk class; needs parity gates + acceptance check (it feeds attn/mlp inputs). RCCL may support custom; else hand-rolled ring over SHM |
| A4 | **Decode Branch-B sync removal** — delete per-call cudaStreamSynchronize at one_shot_allreduce.cu:1033, deferred status consumption | decode W4AR lever (round-wall fire order #1) | W7_ROUNDWALll_desk.md GATE-T section; one_shot_allreduce GREEN world=4 (PLOG-049) | One-line-ish host change; wave64 checklist clean; gate: decode tok/s A/B (GATE-B baselines banked: 40.7/30.8/27.0 t/s at 100/300/600 gen) |
| A5 | **Decode small levers** — align/embedding machinery ~1.15 ms/round + per-forward non-GEMV ~0.8 ms | ~2 ms of 3.5 ms residual | W7_DECODE_RETUNE_desk.md re-priced fire order | After A4 |
| A6 | **F1b drafter quality** — acceptance 0.65-0.71 vs 2.93 ceiling; suppressor = drafter (quant argmax flips / head quality), NOT vocab (97.6-99.4% coverage) | decode multiplier (each +0.1 acceptance ~= +0.2 tok/round) | W7_DECODE_DESK.md; W7_decoderetune_row.txt; coverage counter armed every boot (NINFER_VOCAB_COUNT_DIR) | Options unpriced: drafter head quant tweak, draft-vocab top-up slices for the reasoning register (96.28% vs content 98.95%), acceptance-aware sampling |
| A7 | **Chunk-end sync / gap bimodality** — gap column 15 vs 69 ms/chunk never fully attributed post-cure | ~1-5% prefill | PLOG-062 books (wall 978-992 = gemm+ar+body+gap) | OPTRACE legs on a fresh boot read it free; cheap attribution desk |

## B. GATED — needs the user or a decision, not engineering effort

| # | item | expected size | gate | pointers |
|---|---|---|---|---|
| B1 | **q4 KV cache migration** (bf16 KV is the working path only) | unlocks ~200k context; the user's stated next milestone | USER GO | VRAM headroom is plentiful per user; plan the migration desk after A-tier |
| B2 | **VRAM-funded pre-dequant** (kills the 52% conversion tax) | the biggest single known lever (52% of GEMM issue slots) | USER ORDER: parked until prefill maxed AND B1 landed (double-keyed) | §2.5 PARKED notice + NVIDIA-precedent warning (visibility trap); W7_therm/cold-start findings are the informed context |
| B3 | **upp power-cap raise** (110 W firmware cap -> higher via runtime table) | shifts the ENTIRE thermal state function up (104 cold-start was at 110 W) | OPERATOR GO (can freeze the box; needs user at console) | W7_MAINT_WINDOW_RUNBOOK.md has the knob inventory; THARM1 integrator characterized (W7_therm_row) |
| B4 | **Sustained-serving thermal policy** — schedule heavy requests >=3 min apart (measured drain rule) or accept the state function | keeps the fast band; free | product decision | W7_therm_row.txt |

## C. DORMANT — failed with a named key; never re-litigate without it

| # | item | killed by | THE KEY that reopens it | cheap re-check |
|---|---|---|---|---|
| C1 | Clock pinning (hurt sustained 1.8x, pre-cure) | thermal: pinned-high trips the integrator faster | upp cap raised (B3), OR one idle-afternoon re-leg under the cure config (the cold-start discovery makes the interaction untested) | burst+sustained A/B ~30 min |
| C2 | Graph capture (no steady win) | steady chunks not launch-bound (async queue hides launches) | GEMM wall shrinks >=1.4x (not met; A1 alone won't get there) | single A/B leg ~30 min |
| C3 | Compiler road (44-wait bodies invariant) | LLVM 17/19/22 all identical | each new major ROCm/LLVM release | re-census ~30 min, CPU-only |
| C4 | P2P / peer-write AR / NVLink-class transport | canAccess=0 on ALL pairs (measured twice, iommu=pt too) | hardware change only | P2P probe per pair ~10 min if platform changes |
| C5 | RCCL env knobs | NULL matrix (16ch/Tree worse, LL128 wash) — SHM ring at 2ch IS the envelope | none (class closed) | — |
| C6 | Draft-vocab widening | coverage 97.6-99.4% >= the 90-95% trigger | true-stream coverage (counter armed every boot) drops below trigger | free — read the counter at shutdown |
| C7 | pk16 sketch / mad-mix / padded swizzle / K-split / co-residency | measured kills xN each (PLOG-053/054/056; falsification set complete) | gfx906-class hardware, or a compiler that changes wait-emission/VGPR allocation | re-census ~1 hr if that day comes |
| C8 | Decode draft-arm GEMV retune | NO-GO: already at 1.15x of roofline on the routed winner (nothing to flip) | none — it is optimal; A4/A5 are the decode levers | — |

## D. UNPRICED — known-knowns we have never measured (idea parking, not commitments)

- **Prefix reuse is OFF in the serving line** (`--no-prefix-reuse` + `--prefix-cache-capacity 256`): why it was disabled and what multi-turn/repeated-prefix workloads would gain — never priced on this box.
- **Batching/continuous batching**: all grades so far are single-stream; multi-request prefill interleave is unmeasured.
- **MTP draft length k=3+** at the measured acceptance (0.71-0.94): tok/round ceiling math was only ever done for k=2.
- **Reasoning-register vocabulary slice** (see A6 third option).
- **Chunk-size ladder on the cure config** (M=256 was CLOSED-SKIP pre-cure — the stall cure changes the fixed-cost math slightly; re-leg is one probe if ever needed).

## STANDING MEASUREMENT LAWS (earned 2026-09-18/19 — apply to every row above)

1. Ordinal-paired serve-legs (same bin, fresh boots, position-matched probes, ±2% within-pair) — PLOG-060.
2. Every soak/soak-class number carries its start edge-temp — PLOG-064.
3. No estimated VRAM refusals; no un-gated flips; kill-switch first; RED→GREEN before any close — AGENTS.md.
4. Cross-boot absolutes are thermally poisoned; within-boot/within-pair deltas are the only currency.
