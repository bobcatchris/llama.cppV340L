# PREFILL-ARTAIL — the 552 µs/AR priced honestly, and the prefill AR + orchestration fix set

NO-GPU design desk, 2026-09-17, branch `amd/tp4-cure` (worktree `amd-tp4-cure`). Seat laws
honored: zero GPU work, zero builds, **no `src/` edits** — every patch shape below is prose/diff
SKETCH inside this doc only; only this doc is committed. Read-only inputs:
`results/amd/coherence/PREFILL_OPTRACE_row.txt` (P1, bin `6c8ae21399516750`, HEAD 61c1ecc0b),
`docs/amd/PREFILL_DECOMP_2026-09-17.md`, `docs/amd/TAIL_ARTAIL_2026-09-17.md` (rev 2),
`docs/amd/TP4_AR_TRANSPORT_DECISION_agent5.md` (anchors A4/A6/A7/A7b/A8),
`docs/amd/ONESHOT_W4_BOUNDED_WAIT_DESIGN.md`, `results/amd/coherence/ONESHOT_AR_notes.md`,
BUGTRACK E-17 (PLOG-024/028). Code cites are from THIS worktree (HEAD 7bca3db72 line).

MISSION framing (coordinator, 2026-09-17): user goal prefill **≥500 tok/s** (from 75.5); the
premise handed to this desk was "552 µs/AR is ~4.5x the pure bandwidth time and ~5.5x the RTT —
something is reclaimable". **The honest ring-time computation refutes the 4.5x premise** (§A):
the AR slice is already AT the ring bandwidth floor. This doc still does its full job: price it,
design what is actually reclaimable, and refuse what is not.

---

## A. The honest pricing — VERDICT: TRANSPORT-BOUND (bandwidth regime), not latency, not orchestration

### A.1 The numbers, verbatim anchors

P1 (PREFILL_OPTRACE_row.txt), steady per-chunk 4-rank means, n=16, chunk=128 tok:

```
wall 1642.3 = gemm 798.8 (48.6%) + ar 70.7 (4.3%) + body 645.1 (39.3%) + gap 122.7 (7.5%)
ar per-rank: rank0 71.08  rank1 92.02  rank2 71.85  rank3 47.97   (spread 44 ms!)
gemm per-rank: rank0 803.9 rank1 769.7 rank2 801.0 rank3 840.4    (anti-correlated with ar)
```

Payload per AR: 128 tok × 5120 hidden × 2 B bf16 = **1,310,720 B (1.25 MiB)**, 128 ARs/chunk
(2/layer × 64 layers; sites bracketed by the vt event pairs, PREFILL_DECOMP §5). Mean measured
span = 70.7 ms / 128 = **552 µs/AR**.

### A.2 Ring allreduce time, computed honestly (the computation the mission asked for)

Standard ring model at world n=4: wire per rank = **2(n−1)/n × S = 1.5 × 1.3107 MB =
1.966 MB**, plus per-phase latency terms. Two bus anchors exist on this box and BOTH are cited
(TP4_AR_TRANSPORT_DECISION §1 A6, §5-1: direction-convention dispute, unresolved by law —
"nobody budgets AR off either until explained"):

| model | arithmetic | per-AR | per-chunk (×128) |
|---|---|---|---|
| Ring BW term, conservative anchor 3.13 GiB/s (3.360 GB/s) | 1.966 MB / 3.360 GB/s | **585 µs** | 74.9 ms |
| Ring BW term, optimistic convention 6.61–6.70 GB/s one-way | 1.966 MB / 6.65 GB/s | **296 µs** | 37.9 ms |
| Latency terms (honest size) | 6 phases × A7-class fixed ~8–14 µs (measured 10 KiB eager copy, size-flat 64 B–10 KiB) | ≤ 84 µs, pipelined inside the transfer | ≤ 10.8 ms, mostly hidden |
| Die-mem cross-check (ROOFLINE: copy 369.5, read 377.9 GB/s) | ring reduce touches 1.5S local = 1.97 MB r+w | ~11 µs | **the ceiling is 100x clear — NOT binding at 1.31 MB** |
| **MEASURED (P1)** | 70.7 ms / 128 | **552 µs** | **70.7 ms** |

**Readout:** measured 552 µs sits at **0.94× the conservative ring floor (585 µs)** and 1.87×
the optimistic one (296 µs). The mission's "~4.5x the pure bandwidth time" premise is
**REFUTED by arithmetic**: there is no 4.5x on the table; against the honest ring floor there is
between **zero** (if the 3.13 GiB/s convention is the real bus) and **~33 ms/chunk** (if the
6.65 GB/s convention holds at this shape) of transport headroom. The "~5.5x the RTT" reading is
also a category error the P1 pair dissolves: the 98–101 µs "RTT" is the G-AMD-5 **API+sync
round-trip class** (2 hops × ~50 µs with sync overhead, TP4_AR_TRANSPORT_DECISION §5-2), not a
per-hop cost inside an RCCL collective; the measured data-path fixed cost is A7's 14 µs class,
which at 1.31 MB is ≤3% of the span and already pipelined.

### A.3 The three candidate regimes, each killed or kept by evidence

- **Host-orchestration-bound (eager launch + sync per AR × 128)? NO — exonerated by code-read
  AND by measurement.** The default-W4 route for a 1.31 MB AR is plain `ncclAllReduce` on the
  rank stream with **no per-call host sync** (`tp_group.cpp:352`; the one-shot arm at `:348` is
  size-dead at prefill: 655,360 > `kMaxElements` 65,536, `one_shot_allreduce.h:15` — and note
  that gate holds **even under `NINFER_TP_ONESHOT_AR=1`**, so the per-call
  `cudaStreamSynchronize` of the one-shot host loop, `one_shot_allreduce.cu:811`, never runs at
  prefill shapes in ANY arm). Prefill sync census: **exactly ONE hard sync per chunk** (chunk-end
  drain, `text_context_impl.h:3305`, driver loop `tp2_backend.cpp:2166-2205`) — the TAIL_ARTAIL
  rev-2 default-route census (§B.2 there) holds a fortiori at prefill shapes. Host enqueue for
  the whole ~1300-op chunk is 2–3 ms (VERIFY_LAUNCH_AUDIT §5 law) ≪ 70.7 ms; the ar column is
  device event-pair time, not launch tail. The launch tail lives in the GAP column (§E), not here.
- **Latency-bound? NO.** At 1.31 MB the wire term (296–585 µs) is 3.5–7× the entire fixed-cost
  budget (~14 µs/leg, A7/A7b — the latency regime is the 10 KiB DECODE regime, §F). RTT-class
  terms cannot own a 552 µs span whose bandwidth floor is 296–585 µs.
- **Transport-bound? YES — bandwidth regime, at or below the conservative ring floor.** And the
  per-rank spread proves the spans additionally absorb **intra-chunk arrival slosh**:
  gemm+ar per rank is near-constant (r0 874.9, r1 861.8, r2 872.8, r3 888.4 — spread 26.6 ms ≈
  1.6%) while gemm and ar are anti-correlated (fast-GEMM rank1 waits 92 ms inside ARs for
  slow-GEMM rank3, which waits only 48). The AR is doing its lockstep-absorber job; the
  pure-transport content of the 70.7 ms is ≤70.7 and ≥ the floor the sweep (§G) pins. De-skewing
  rank GEMM time (r3 840 vs r1 770) would shrink AR SPANS on ranks 0/1/2 and NOT the wall (the
  wall is the pacer rank) — do not book slosh as recoverable ms.

**VERDICT (one line): the 552 µs/AR is TRANSPORT-BOUND — measured at 94% of the conservative
ring bandwidth floor, with latency ≤3-15% (pipelined) and orchestration zero (async RCCL, one
chunk-end sync, 2-3 ms/chunk total enqueue). The only true AR-side levers are fewer bytes
(refused, §D), overlap (§C2), or a faster bus (§C1).**

---

## B. Fix (a): one-shot AR at 1.31 MB — **PRICED, REFUSED for prefill** (2.3–3.1x WORSE)

Design **assuming the W4AR GREEN leg lands** (`amd/oneshot-ar` @ 7dd270662; B1/B2/B3 bounds +
watchdog + injectors per ONESHOT_W4_BOUNDED_WAIT_DESIGN §2/§6; device leg running in a parallel
window). **Contingency if RED:** nothing in this doc's prefill sequence depends on GREEN — the
prefill AR stays RCCL ring in every branch; RED only parks §F's decode A/B and keeps the
default-route census (TAIL_ARTAIL §B.2) the governing truth.

1. **The GREEN flip alone does not touch prefill.** The route gate is
   `impl_->one_shot && n_elems <= kMaxElements` (`tp_group.cpp:348`); prefill payload 655,360 el
   is 10.0× the 65,536 cap ("covers up to T=12.8" at hidden 5120 — A4). All 128 prefill ARs
   route to RCCL ring with env=1 exactly as with env absent. Adoption for prefill requires
   widening `kMaxElements` ≥ 655,360 (16×), a capability-constant change that must carry its own
   cell arms (the `kR1MaxWorld` construction-refusal pattern, `tp_group.cpp:130-135`).
2. **The buffer trade is fine — capacity is NOT the refusal.** Slot staging =
   `kMaxElements × 2 B × kNumSlots(128) × world(4)` per rank process (`one_shot_allreduce.cu:601-610`):
   16 MiB → **256 MiB/rank process (1.0 GiB host-pinned total across 4 processes)**. This is
   pinned HOST memory, not device VRAM — the VRAM LAW (no estimate-refused launches) is not
   engaged, and the only "16.4 GB" law on the books is SD-1b's FIT-LAW (NVFP4 weight planes need
   all 4 dies) — three orders of magnitude above this staging and a different resource. 1.31 MB
   FITS the trade.
3. **The refusal is PERF, and it is robust across both bus conventions.** One-shot per-rank wire
   = publish S + read 3 peers = **4S = 5.24 MB** (vs ring's 1.5S) at one hop latency:

   | convention | one-shot @1.31 MB | ring @1.31 MB | verdict |
   |---|---|---|---|
   | conservative 3.36 GB/s | max(publish 390 µs, 3 reads 1170 µs) + poll ~100 µs ≈ **1.3–1.7 ms** | 585 µs floor; 552 measured | one-shot ≈ **2.3–3.1x worse** → 166–218 ms/chunk |
   | optimistic 6.65 GB/s | ≈ **0.7–0.85 ms** | 296 µs floor | still ~1.4–1.9x worse |

   One-shot is a LATENCY play (1 lockstep round); at 1.31 MB the bandwidth term owns and the
   ring's 1.5S/rank is optimal. **Adopting it would take the ar slice 70.7 → 100-218 ms/chunk.
   REFUSED.** This is the same shape as the honest B-1 pre-declaration in
   TP4_AR_TRANSPORT_DECISION §4 — its "extend-the-mesh" branch was conditioned on the sweep and
   applies to DECODE shapes only (§F).

---

## C. The fix set that survives pricing

### C1. F-ENV — RCCL bus-headsweep, zero code (rides any window; THE decisive check's home arm)

`NCCL_MIN_NCHANNELS`, `NCCL_P2P_LEVEL`, `NCCL_DEBUG=INFO` (name the transport) A/B'd on the
banked P1 posture (`6c8ae21399516750`), plen-2000 probe, cool window. Predicted: ar 70.7 →
**38–71 ms/chunk** (the sweep decides whether the 6.65 GB/s convention is reachable at this
shape; the conservative read is zero recovery). Both-direction falsifier: env arms must be
byte-identical OFF.

### C2. F-BISECT — bi-segment AR overlap with independent compute: **−30–35 ms/chunk** (the only true exposure-killer)

Split each AR's 128-token payload into two 64-token halves on the AR stream; the FIRST half's
completion event gates the downstream GEMM on half A while half B is still on the wire.
**Dependency story for the 5-problem layer** (per-rank tiled problems: GdnInput 4096×5120 ×48,
AttnInput 3584×5120 ×16, MlpGateUp 8704×5120 ×64, out-proj 5120×1536 ×64, down-proj 5120×4352
×64 — ROUTE PROOF, PREFILL_DECOMP §1):

- GDN layer chain: input-GEMM → conv/WY delta body → out-proj → **AR#1** → norm → gate-up →
  silu → down-proj → **AR#2** → (next layer input-GEMM).
- **What bisects cleanly:** everything from AR#1 onward is row-parallel over tokens — AR#1(A/B),
  gate-up(A) gated on AR#1(A)'s event (gate-up(B) concurrent with it, gated on AR#1(B)),
  silu/down-proj likewise, AR#2(A/B), next layer's input-GEMM(A) gated on AR#2(A). The GDN
  body (conv state + chunked WY delta-net, `kChunkSize=64`, state-passing sequential over
  tokens) does **NOT** bisect — it stays full-width AFTER both AR#2 halves complete (it needs
  the full hidden). Attn layers analogous (GQA prefill is token-blocked already).
- **AR call count doubles (256 × 0.66 MB/chunk) — same total wire bytes, same floor**; the
  saving is EXPOSURE: half of every AR hides under the first half's downstream GEMM.
  Predicted exposed ar ≈ 70.7 → **35–40 ms/chunk**, i.e. the "ar <~35 ms at the 500 goal" bar,
  met at the boundary.
- Complexity grade: **HIGH** (multi-stream + event choreography through every layer body;
  composes with F-GRAPH only as per-half capture). **Gate it on the GEMM/BODY collapse**: at
  today's wall it is −2%; it becomes first-order exactly when the 500-goal kernel work lands.

### C3. F-EPILOGUE — fold the AR axpy residual into the next op's read: **−1–2 ms/chunk** (cleanup, rides any body-desk window)

128 axpy launches + 1.31 MB r+w each deleted (PREFILL_DECOMP Fix B's code arm). Bandwidth-class
saving ~0.9 ms at the 369.5 GB/s copy anchor + 128 launches ~0.25 ms. Never a window of its own.

---

## D. Fix (c): bf16→fp8/fp4 compress-then-allreduce — **REFUSED, honestly, on E-17**

Predicted wire saving is real (ar 70.7 → ~38 fp8 / ~20 fp4 at the same convention): halving or
quartering S quarters... halves the floor. It is refused anyway, on the closed lesson:

- **E-17 (BUGTRACK, PLOG-024 + PLOG-028 follow-up): the flips are bf16 near-ties/degenerate** —
  at temp 0 the bar is byte-identity, and near-ties flip on ulp-class perturbation. A reduced-
  precision AR changes the reduce VALUES (not just order) in EVERY layer of EVERY chunk; the
  hidden diverges in ulp class; greedy argmax near-ties flip; the token stream diverges — the
  exact wound E-17 took a dedicated window and an audit tap to classify. Re-opening it to buy
  ≤33 ms of a 1642 ms wall (2%) is the SD-1 drift signature in miniature.
- Mechanically it is also not a flag: RCCL has no fp8/fp4 collective on this stack; it means
  quant/dequant kernels + a custom collective = the one-shot-class machinery plus a NEW numerics
  surface, i.e. maximum risk for the minimum prize. **REFUSED.** (Mirror image of TAIL_ARTAIL's
  fp32-payload audit: that one was free to refuse because the payload was already minimal; this
  one refuses a real saving because the correctness cost is banked.)

---

## E. Fix (d): the 122.7 ms gap — orchestration disposition, and the 500-scale re-ranking

- **The gap is NOT per-AR eager overhead.** There is no per-AR sync to shrink (§A.3); the gap is
  the chunk-end drain + post-sync re-enqueue bubble class (driver re-enqueues chunk N+1 only
  after the `:3305` drain) — the P1 falsifier's own disposition, and F-BISECT's extra
  streams/events would if anything INFLATE it unless composed with graphs.
- **F-GRAPH — chunk graph capture/replay (PREFILL_DECOMP Fix A): predicted gap 122.7 → ~10–30 ms
  (−90–110 ms/chunk).** DEMOTED by P1 to the 7.5% class at the 75-tok/s scale — but **re-ranked
  UP at the 500 goal**: 122.7 ms is 48% of the 256 ms/chunk budget that 500 tok/s implies. The
  same fixes that shrink the ar slice do NOT shrink the gap; the gap needs its own lever, and
  this is the named one (decode is already GRAPHS-ON, `decode_graph.cpp`; RCCL-capture fallback
  = per-layer-body graphs between ARs, PREFILL_DECOMP §5). Cheaper first move: chunk N+1
  pre-enqueue (double-buffered driver loop) — kills the re-entry bubble, a fraction of the
  class, ~zero risk, rides any window.
- Honest ceiling of this desk's whole portfolio at today's wall: ar (0–33) + epilogue (1.5) +
  gap (90–110) + bisect (30–35) ≈ **≤165 ms of 1642 = 10%** → prefill 75.5 → **≤82 tok/s with
  the AR and gap both perfect**. **The 500 goal is owned by gemm 798.8 + body 645.1 (88% of the
  wall) — the FRONT tiled-GEMM family (predicted ≥3–5x, PREFILL_FRONT §6) and the BODY kernel
  class (P1's named winner).** This desk's contract at the 500 scale: keep ar ≤35 ms and gap
  ≤30 ms so the collective plane is never the straggler while the 7x kernel-class collapse
  (gemm+body 1444 → ~196 ms, reachable only in the fp16-2:1/nominal-TF class per
  PREFILL_DECOMP §5's 164–178 tok/s no-reopen ceiling) happens above it.

---

## F. The decode-side AR (9.6 ms/round in-loop): one flip does NOT serve both fronts

- Decode ARs are 10,240 B, 128/round, measured ~75 µs each pipelined (9.60 ms/round, TAIL_ARTAIL
  §A.2 context) — the **latency regime** (A7: 14.1 µs eager staged copy at exactly 10 KiB,
  size-flat; wire term 4.6–6.3 µs is noise). Opposite regime from prefill's 1.31 MB — the two
  fronts do not share a transport conclusion.
- **Post-GREEN one-shot at decode: predicted 7.6–12 ms/round (bracket 28–90 µs/AR, A7/A8) vs
  ring's measured 7.6–9.6** — a WASH band, not a lever. And the flip ACTIVATES the per-call
  `cudaStreamSynchronize` drain (`one_shot_allreduce.cu:811`; TAIL_BOOK's 8-drain census becomes
  live), so adoption additionally requires the parked FIX-A/FIX-B deferred-drain pair
  (TAIL_ARTAIL §F6 ordering: bounded-wait cells first). Decode round impact of adopting: **0 to
  −2 ms/round possible, +2.4 possible the other way — decided only by the GREEN leg's own
  measured A/B row (W4AR-W4-IDENTITY + a 200-round timing leg), never by this bracket.**
- Prefill under the same flip: unchanged (size-gated to ring, §B.1). **Verdict: one flip, two
  fronts, zero prefill ms and a wash band on decode — the flip's value is the liveness net
  (watchdog/bounded death), not speed. Adopt for safety if GREEN; do not book it as perf.**

---

## G. Recommended sequence, predicted ms, and THE decisive check

| # | fix | predicted Δ wall/chunk | cost/risk | gate |
|---|---|---|---|---|
| 1 | **B-1 sweep** (below) | 0 (it measures) | one window, ~minutes | none — run first |
| 2 | F-ENV RCCL arms | 0 to −33 ms (ar 70.7→38–71) | zero code | rides #1's boot |
| 3 | F-EPILOGUE axpy fold | −1–2 ms | trivial diff | cleanup; byte-identity cell |
| 4 | F-GRAPH chunk capture (+ pre-enqueue first move) | −90–110 ms (gap → 10–30) | medium; decode-graph pattern exists | own window + battery; re-ranked UP at 500 scale |
| 5 | F-BISECT AR overlap | −30–35 ms (ar → 35–40 exposed) | HIGH | ONLY after GEMM/BODY collapse makes ar first-order |
| — | one-shot @1.31 MB | **+95–150 (REFUSED)** | — | §B, robust across conventions |
| — | fp8/fp4 AR payload | **REFUSED** (≤33 ms prize) | — | §D, E-17 |

Cumulative honest band if all surviving fixes land: **1642.3 → ~1475–1520 ms/chunk (84–87
tok/s)** — every further multiple belongs to the GEMM/BODY desks.

**THE SINGLE DECISIVE CHECK — B-1, the RCCL ncclAllReduce micro-sweep** (chartered in
TP4_AR_TRANSPORT_DECISION §4, still unrun; P1's 552 µs is the serving-embedded point that makes
it decisive rather than exploratory): S ∈ {10 KiB, 128 KiB, 1.31 MB} × world ∈ {2, 4}, 200 timed
reps, median + p5, clocks sideband banked, same binary discipline as the boot battery. It
settles in one cell: (i) the prefill transport floor — is 552 µs the 585-class conservative
floor (F-ENV recovers ~0, F-BISECT is the only AR lever) or the 296-class optimistic floor (up
to ~33 ms/chunk of bus headroom for F-ENV to chase); (ii) decode ring-vs-bracket at 10 KiB —
sizes the one-shot decode adoption band (§F); (iii) the A6 direction-convention dispute that
the law says blocks all AR budgeting. Pre-declared reading (inherited, cause-neutral): median
at 10 KiB w=4 ≤120 µs ⇒ RCCL carries decode, mesh stays safety-only; ≥240 µs ⇒ mesh gets a
decode perf WO; 1.31 MB median <520 µs ⇒ F-ENV has a real bus headroom; ≥520 ⇒ book zero and
promote F-BISECT to the only AR lever.

## Honesty block

1. No silicon touched; every measured number is quoted from a banked row (P1, TAIL_ARTAIL/M2,
   A6/A7/A8, ROOFLINE) and every other figure is labeled arithmetic or code-read with file:line.
2. The headline is a REFUTATION: the mission premise (4.5x recoverable) fails the honest ring
   computation; what survives is a ≤165 ms/chunk portfolio whose largest single item (F-GRAPH)
   is another doc's fix re-ranked, not a new mechanism.
3. The F-BISECT −30–35 ms is a model number (exposure = half the AR floor), never device-run;
   the GDN-body non-bisectability is a code-read claim (state-passing, `kChunkSize=64`) that the
   landing seat must re-verify against `gated_delta_net.cpp:253-313` before design freeze.
4. F-ENV's band is wide (0 to −33) precisely because the A6 convention dispute is open by law;
   the sweep closes it. No fix above may book its predicted ms without its A/B row (RED/GREEN
   closure law); none of these closes a bug — they are perf claims with their checks named.
5. The one-shot refusal (§B) is robust across both conventions but prices the poll term at the
   A8 prose anchor (~60–100 µs); if B-1's w=4 one-shot cell ever measures better, the refusal
   re-opens — the bracket, not the verdict, is the durable artifact.
6. The 500-tok/s statement (§E) is a ceiling allocation, not a promise: gfx900 has no matrix
   instructions (PREFILL_FRONT §2/§4); the 7x gemm+body collapse lives in the fp16-2:1
   nominal-TF class and is other desks' to measure.
