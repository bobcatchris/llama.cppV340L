# W2 AR FLOOR — the prefill allreduce column as a one-window measurement plan

NO-GPU desk (W2, companion to `docs/amd/W2_GAP_CAPTURE_desk.md`), 2026-09-18, lane
`amd/wo-w7-body`. This doc only SUMMARIZES and sequences existing specs; every number below is
quoted from a banked row or the named desk doc.

## Source specs (the plan's "AR-desk F-GRAPH/F-ENV specs, both decisive checks written")

- **`docs/amd/PREFILL_ARTAIL_2026-09-17.md`** — the AR-desk spec the plan's W2 line cites.
  F-ENV = its §C1, F-GRAPH = its §E, F-BISECT = §C2, one-shot refusal = §B, fp8/fp4 refusal = §D,
  and **the single decisive check B-1 = its §G** (pre-declared readings included).
- **`docs/amd/TP4_AR_TRANSPORT_DECISION_agent5.md`** — bus anchors A4/A6/A7/A7b/A8 behind the
  3.36 vs 6.65 GB/s convention dispute (unresolved by law until measured).
- **`docs/amd/AR_PARITY_ARM_design_agent5.md`** — robustness insurance only (AR-ring word-tear
  witness; triggers checked-and-not-fired at G18e). **Not a perf arm; out of scope here.**

## The floor, honestly priced (ARTAIL §A)

Payload 1.31 MB bf16 (128 tok × 5120 × 2 B) × 128 ARs/chunk (2/layer × 64). Ring wire per rank
1.5S = 1.97 MB ⇒ **floor 296 µs/AR (optimistic 6.65 GB/s) … 585 µs/AR (conservative 3.36 GB/s)**
⇒ **ar column floor 38–75 ms/chunk** (the plan quotes "~35–70"). Measured: 552 µs/AR = 94% of the
conservative floor (P1); W7 promoted-GQA leg shows the same picture with rank slosh —
ar r0 66.0 / r1 82.8 / r2 62.9 / r3 49.3 ms/chunk, anti-correlated with per-rank gemm
(gemm+ar near-constant per rank). **Verdict already on file: transport-bound + arrival slosh; the
slosh is NOT recoverable ms** (de-skewing gemm shrinks ar spans, not the wall).

## Env arms that exist (all byte-identical when absent)

| arm | where | prefill effect |
|---|---|---|
| `NCCL_MIN_NCHANNELS`, `NCCL_P2P_LEVEL`, `NCCL_DEBUG=INFO` | RCCL env (ARTAIL §C1 F-ENV) | the ONLY live prefill AR arm; predicted ar 70.7 → **38–71 ms/chunk**, sweep decides |
| `NINFER_TP_ONESHOT_AR=1` | `tp_group.cpp:16-18` (default OFF) | **SIZE-DEAD at prefill in every arm**: 655,360 el ≫ `kMaxElements` 65,536 (`one_shot_allreduce.h:17`) — routes to ring identically. Decode-only A/B per ARTAIL §F, and there a wash band (0 to ±2 ms/round) |
| `NINFER_PREFILL_GRAPH=1` (proposed) | gap-capture desk | does not touch ar transport |

Any additional env arm (e.g. NCCL algo/protocol overrides) must be declared before the window per
R3 — not pre-registered, not run.

## One-window executable plan

1. **B-1, the decisive check (runs first, ~minutes, rides any boot)** — RCCL `ncclAllReduce`
   micro-sweep: S ∈ {10 KiB, 128 KiB, 1.31 MB} × world ∈ {2, 4}, 200 timed reps, median + p5,
   clocks sideband banked (same discipline as the boot battery). Pre-declared readings (ARTAIL §G,
   verbatim): 1.31 MB median **<520 µs ⇒ F-ENV has real bus headroom**; **≥520 µs ⇒ book zero and
   promote F-BISECT to the only AR lever** (and F-BISECT stays gated on the GEMM/BODY collapse —
   at today's 1490 ms wall it is −2%); 10 KiB w=4 median ≤120 µs ⇒ RCCL carries decode; ≥240 µs ⇒
   mesh gets a decode perf WO. B-1 also closes the A6 direction-convention dispute that by law
   blocks all AR budgeting.
2. **F-ENV A/B on the same boot** — the three env arms above, plen-2075 leg on the promoted
   posture, cool window, `[PREFILL-SUM]` ar column per rank + greedy text byte-identity (E-17 bar).
   Acceptance: ar-column delta vs the OFF leg on the SAME boot; byte-identical unset required.
3. **Book only what the row says**: expected outcome band **ar 38–75 ms/chunk**; anything below
   38 claims a faster-than-measured bus and must name which B-1 cell proves it. Do not book slosh.

## Refusals to respect (already priced, do not re-open without new evidence)

- **One-shot AR at 1.31 MB: REFUSED** — 4S wire vs ring 1.5S ⇒ 2.3–3.1× WORSE, +95–150 ms/chunk
  (ARTAIL §B; re-opens only if a B-1-class cell ever measures one-shot better at this size).
- **fp8/fp4 compress-then-allreduce: REFUSED** — E-17 numerics (greedy near-tie flips), ≤33 ms
  prize, no RCCL fp8/fp4 collective on this stack (ARTAIL §D).
- **F-BISECT: HIGH complexity, gated** on the GEMM/BODY collapse making ar first-order (ARTAIL
  §C2/G).
- **F-EPILOGUE** (fold the AR axpy residual into the next op's read, −1–2 ms/chunk, 128 launches
  deleted): trivial diff, rides any body-desk window, never its own window (ARTAIL §C3).

Receipt discipline (R7): B-1 row + F-ENV A/B rows land in `results/amd/coherence/`, chained into
the plan ledger per R1, bin shas named in both directions of every A/B.

## VERDICT (2026-09-18, window owner, B-1 executed) — BOOK ZERO FOR F-ENV

B-1 ran on the FINAL platform (pinned+pt, cycle=warm, RCCL 2.20.5, 200 reps, all cells ok):
row `results/amd/p3/W7_b1_ar_floor_row.txt`, log `W7_b1_ar_floor_b1_sweep.log`
(sha256 1f68acba…). Pre-declared readings applied:

- **1.31 MB @ w=4 med 849.47 µs ≥ 520 µs ⇒ F-ENV BOOKED ZERO** (the NCCL env arms are refuted
  by measurement before being run; step-2 A/B legs cancelled per the decision tree).
- 10 KiB @ w=4 med 135.62 µs — between the 120/240 triggers; neither fires; no decode WO.
- **A6 convention CLOSED**: one_way 6.691-6.705 GB/s / traversal 3.451-3.491 GB/s — the
  optimistic 6.65 GB/s convention is the one-way number.
- w=4 1.31 MB floor confirmed at ~849 µs/AR ⇒ the engine's measured ar column (44-95 ms/chunk
  ÷ 128 ARs = 344-742 µs/AR) already runs at or below the clean-sweep floor via overlap; there
  is no transport win to book. **F-BISECT is the only remaining AR lever and stays gated** on
  the GEMM/BODY collapse — at the post-finalize-cure wall (~1037 ms/chunk) it is ≪2%.
