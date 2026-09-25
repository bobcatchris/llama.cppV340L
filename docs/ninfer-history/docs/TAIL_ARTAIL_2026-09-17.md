# TAIL-ARTAIL — the verify-tail collective inventory, latency math, and fix designs

NO-GPU design desk, 2026-09-17, branch `amd/tp4-cure`. REVISION 2 (coordinator refocus:
"lgather is wait-on-root, promote drains") — folds in the MEASURED M2 verdict
(results/amd/coherence/VERIFY_TAIL_row.txt + PLOG-043, landed by 9cee65ae2) and cross-refers
docs/amd/TAIL_BOOK_2026-09-17.md (ebfb2eaf4) and docs/amd/TAIL_LMHEAD (fb86a0fdd). Revision 1's
transport framing is corrected where the measurement convicts it; the collective inventory and
floor math survive as the authority on how much DEVICE-COLLECTIVE time can hide inside the
measured spans — now the centerpiece (§D). Laws honored: no GPU runs, no builds, **no `src/`
edits** (every patch below is a SKETCH inside this doc only). All file:line citations are from
THIS worktree at the M2 tracer commit `248630d1b`.

Posture throughout: single-seq MTP, `--draft-tokens 2` (k=2, k_buf=2), verify width T=rk+1=3,
world=4 (TP4 = 4×MI25 dies), greedy, draft-vocab remap file resolves (40,960 ids), conc=1,
count600-class, tracer ON, **NINFER_TP_ONESHOT_AR unset (the default W4 route)** — that last
one is load-bearing for §D.

MEASURED M2 VERDICT this doc now aligns to (clean-band): **lm_head 61.93 ms/round ≈ 90% of the
tail** (rank0 66.0 > ranks1-3 59.7-61.6; issue-pathology not memory-shaped — LMHEAD desk owns
it); **lgather 4.92 ms on ranks 1-3 BUT 0.78 ms on rank 0 = WAIT-ON-ROOT** (ranks1-3 waiting on
rank0's late lm_head; the transport itself is sub-ms); ar_argmax **0.20**; accept **0.017**;
align **8.28**; chain_fwd **7.90**; propose **0.32**; book exonerated **0.10**; sync **117.46**
(whole-verify overlap, not additive); d2h 0.026.

---

## A. Collective inventory (per verify round, world=4, k=2, T=3) — authoritative, unchanged by rev 2

### A.1 Geometry (allocation-site derived, not assumed)

| buffer | shape | site | bytes/rank |
|---|---|---|---|
| `verify_logits` (target shard) | BF16 {62,080, 3, 1} | tp2_backend.cpp:1341, `n_vocab = 248320 / tp_w` (:1242) | 372,480 |
| `verify_logits_full` (gathered) | BF16 {248,320, 3, 1} | tp2_backend.cpp:1342 | 1,489,920 |
| `proposal_logits` / `mtp_logits` (draft shard) | BF16 {10,240, 1} | tp2_backend.cpp:1326, `40960/4` (:1108, :1243) | 20,480 |
| draft head (MTP proposal head) | {10,240, 5120} Q4G64/W8, per-rank slice | tp2_backend.cpp:1106-1226 | ~11 MB encoded |
| target lm_head per-rank shard | {62,080, 5120} W8G32_F16S (337.7 MB/rank — LMHEAD desk's corrected label) | tp2_backend.cpp:1112 | 337.7 MB weight read per GEMM |
| R1 argmax wire | `ArgmaxChampion` 12 B × T | argmax_r1.h:29-43 (static_assert sizeof==12) | 36 B (verify), 12 B (draft) |

The mission's two candidate payload shapes resolve as: the **verify** logits are the
FULL-VOCAB ColumnN shard (62,080 rows/rank — NOT the 40,960 draft slice; text_context_impl.h:1823
shape gate + the R1 witness "sumexp ~= n_rows (6.2080e+04 at w4)", r1_argmax.cu:135); the
**draft** head is the 40,960-id slice (10,240/rank). All logits payloads are bf16; the R1
exchange is int8 byte-typed (§A.4).

### A.2 The tail collectives, in round order — AND THE ROUTE EACH ACTUALLY TAKES AT TP4

| # | op | call site | count/round | payload per op (send/recv per rank) | ROUTE at world=4 default |
|---|---|---|---|---|---|
| 1 | `lgather` | tp2_backend.cpp:3056 loop `j=0..rk` → tp_group.cpp:358-382 | **3** (one ncclAllGather PER COLUMN) | 124,160 B send / 372,480 B ring-recv | RCCL ncclAllGather ncclBfloat16 (:380) — async, NO host sync |
| 2 | `ar_argmax` | tp2_backend.cpp:3164 → tp_group.cpp:442-455 → r1_argmax.cu:300-348 | **1** (all T=3 columns in ONE call) | kernel-1 re-READS 372,480 B local; wire = 3×12 B = **36 B** ncclInt8 allgather | **R1 allgather-reduce** — the one-shot ring is NULL at w4 (tp_group.cpp:144-146), so :436's ring arm is dead; R1 arm has NO unconditional host sync (r1_argmax.cu syncs at :328/:378 are inside `if (r1_trace)`) |
| 3 | `align` ARs | tp2_backend.cpp:3535 → mtp_forward_core (text_context_impl.h:1326; ARs :1551, :1563) | **2** | 5,120 el bf16 = 10,240 B each | RCCL ncclAllReduce (tp_group.cpp:352 fall-through — `impl_->one_shot` is NULL at w4 default, :346-348 guard) — async, NO host sync |
| 4 | `propose` d0 AR | tp2_backend.cpp:3566-3568 | **1** | 12 B champion | R1 (as #2) — no host sync |
| 5 | `chain_fwd` ARs | tp2_backend.cpp:3640 → mtp_forward_core | **2** (rk−1 = 1 step × 2) | 10,240 B each | RCCL ncclAllReduce — async |
| 6 | `chain_head` AR | tp2_backend.cpp:3649-3651 | **1** | 12 B champion | R1 (as #2) — no host sync |

**TAIL TOTAL: 10 collectives/round** (3 big allgathers 124 KB, 4 hidden ARs 10 KB, 3 tiny
champion allgathers 12-36 B). **Hard `cudaStreamSynchronize` count inside the tail on the
DEFAULT W4 route: exactly ONE** — the accept drain (tp2_backend.cpp:3222). Context (the loop):
64 layers × 2 ARs = 128 × 10,240 B allreduces, measured 9.60 ms/round ≈ 75 µs each (pipelined).
Per-round grand total: 138 collectives; per-rank tail wire ≈ 0.41 MB sent + 1.16 MB received.

### A.3 Redundancy found while counting (feeds F1/F2a)

- **The 3 lgathers are DEAD DATAFLOW in greedy.** Their only consumer,
  `speculative_accept_greedy_drafts_kernel`, returns in its greedy arm
  (src/ops/kernel/speculative_round.cuh:104-121) **before any read of the `logits`
  parameter** — greedy reads only `target_tokens` (R1's global argmax) and `drafts`;
  `token_domain` is unused there. The temp>0 arm (:123-192) is the sole reader. The batched
  runner already knows the class ("the vfull allgather stays for non-greedy",
  tp2_backend.cpp:4177); the single-seq loop never got the gate.
- **The verify shard argmax is computed TWICE**: text_context_impl.h:1873 local
  `ops::argmax` (372,480 B read) → overwritten at tp2_backend.cpp:3164 by R1's kernel-1,
  which re-reads the same 372,480 B.

### A.4 bf16→fp32 payload audit — NO fp32 collective exists (unchanged, verdict stand)

`ncclAllGather(ncclBfloat16)` tp_group.cpp:380 (the only allgather entry); ncclAllReduce
bf16/f32 dispatchers at :188/:192 with `allreduce_f32` having ZERO call sites in the MTP round
loop; `ncclAllGather(ncclInt8, T×12 B)` r1_argmax.cu:341; FP32 tensors on this path are
device-local/pinned staging, never collective payloads. Even a hypothetical fp32 lgather adds
~0.75 MB ≈ 0.08 ms. **DEAD hypothesis, now measurement-backed too** (rank0's full 3-gather set
ran in 0.78 ms — transport is sub-ms with the real bf16 payloads).

---

## B. Latency math — REVISED against the measured M2 row

### B.1 Floor vs measured, per span (the authoritative split the coordinator asked for)

Device floor per span = local compute (at the loop's measured effective GEMM class) + the
span's own collective transport (RTT ~0.1 ms/op + wire; rank0's 3-gather lgather measured
0.78 ms ⇒ ~0.26 ms per 124 KB allgather end-to-end, the best transport datum we now have).

| span (M2 field) | measured ms (clean band) | device-collective floor (this desk) | wait/drain residue | residue share |
|---|---|---|---|---|
| lgather | 4.92 (r1-3) / **0.78 (r0)** | 0.6-0.9 | ranks1-3: ~4.0 = **wait-on-root** (r0's late lm_head); r0: ~0 | 81% on r1-3; 0% on r0 |
| ar_argmax | 0.20 | 0.12-0.2 | ~0 | 0% |
| align | **8.28** | ≤ 1.2 (0.8 T=3 draft-layer compute + 2×0.1 AR + wire) | ≥ 7.0 | ≥ 85% |
| chain_fwd | **7.90** | ≤ 0.7 (0.3-0.5 T=1 compute + 2×0.1 AR) | ≥ 7.2 | ≥ 91% |
| propose | **0.32** | ≤ 0.35 (draft GEMM ~0.1-0.2 + 12 B R1 AR) | ~0 — **the clean control** | 0% |
| accept | 0.017 | ≤ 0.02 | 0 | 0% |

**Centerpiece verdict: of the 16.5 ms draft-side collective-adjacent class (align 8.28 +
chain_fwd 7.90 + propose 0.32), AT MOST ~2.2 ms can be device-collective + compute time; ≥14.3
ms is WAIT.** propose (same R1 AR class as chain_head, same stream discipline) ran at its
floor — so the wait is POSITIONAL, concentrated in the two spans that immediately follow the
round's only hard sync (accept drain, :3222) and its enqueue segment.

### B.2 What the wait CAN be on the default W4 route (mechanism candidates, post-census-correction)

TAIL_BOOK's host-drain-quantization hypothesis priced the right phenomenon but cited the wrong
sinks for THIS route: its 8-drain census counts `one_shot_argmax.cu:567-570` and
`one_shot_allreduce.cu:808-811` — both inside the ONE-SHOT arms, which are UNREACHABLE at TP4
default: `Impl::one_shot` and `Impl::one_shot_argmax` are constructed ONLY at world==2, or at
any world under `NINFER_TP_ONESHOT_AR=1` (tp_group.cpp:140-149 — "default OFF"); every
`allreduce_argmax`/`allreduce_local_bf16` at w4-default therefore routes to R1
(tp_group.cpp:442-455) / plain ncclAllReduce (:352), neither of which host-syncs per call
(§A.2 route column). **The per-round hard-sync census on the default route is 1 (accept
:3222), not 8.** Consequently FIX-A (deferred-consume) and FIX-B (single-drain) as specced
against the one-shot status loops would be fixing sinks that do not run in this posture — they
become relevant ONLY on a future `NINFER_TP_ONESHOT_AR=1` boot (post bounded-wait GREEN leg).

The surviving mechanism family for the ≥14.3 ms residue, both host-paced, distinguishable
WITHOUT new code (§D checks):

- **M-skew (peer wait)**: the accept drain empties every rank's host pipeline at :3222; the
  four hosts then re-enqueue rebase→prepnext→align→chain at their own pace. Each RCCL AR
  re-synchronizes the device streams, so every early rank's device waits inside align/chain AR
  spans for the slowest HOST to arrive — the exact signature already measured at lgather
  (4.92 vs 0.78) and in-loop (ranks 0/1 AR 10.4-11.1 vs ranks 2/3 8.0-8.8). Predicts
  align/chain_fwd spans LARGE on early ranks, SMALL on the LAST-arriving rank.
- **M-starve (self host-pace)**: each rank's own host cannot enqueue the ~25-35-launch
  segment fast enough after the cold start, so its own device idles inside the spans.
  Predicts align/chain_fwd spans LARGE ON ALL RANKS EQUALLY (each rank waits on itself).
- TAIL_BOOK's enqueue-pricing (5-15 µs/launch) makes pure M-starve hard to stretch to 8 ms
  (~270 µs/launch would be needed); M-skew needs only the ~2.5 ms/round rank asymmetry ALREADY
  measured in-loop. Prior: M-skew dominant, M-starve secondary. **This is a prediction, not a
  finding — D2 decides.**

### B.3 Hypothesis verdicts (rev 2)

| hypothesis | verdict | arithmetic |
|---|---|---|
| lgather transport (bytes/dtype/algorithm) | **DEMOTED — measured sub-ms** (rank0: 3 real AGs in 0.78 ms). No payload/algorithm fix can beat ~0.3 ms | §B.1, §A.4 |
| lgather as WAIT-ON-ROOT | **CONFIRMED by the per-rank split**; the only lgather-side fix that pays is removing the sync point or the ROOT (rank0's lm_head — LMHEAD desk) | 4.92 vs 0.78 |
| draft-chain spans = one-shot host drains (TAIL_BOOK census) | **REFUTED for the default route** — cited sinks unreachable at w4 (guards by line, §B.2); census correction is the desk's duty. FIX-A/B deferred to the oneshot-gated world | §B.2 |
| draft-chain spans ≥75% wait (device-collective excluded by floor) | **CONFIRMED by floor math** — this is the authoritative split the refocus asked for | §B.1 |
| M-skew vs M-starve | OPEN — **decided free-of-charge by the per-rank [TAIL] rows** (D2) | §D |
| lm_head GEMM = the tail's owner | **CONFIRMED (61.93 ms)** — LMHEAD desk owns the fix (simt_r8_c4 route-flip, predicted 2-4 ms); not a collective | fb86a0fdd |
| RTT×count floor | CANNOT own the tail (≈1.1-1.3 ms) | rev-1 §B.1, unchanged |
| fp32 payload inflation | DEAD (audit + measurement) | §A.4 |
| batch draft-chain ARs into one | CANNOT — autoregressive dependence (d0 AR output = chain step 0's input token; chain-head AR output = next round's drafts) | code order :3567→:3640→:3650 |
| one-shot-AR swap in the tail | NOT a tail lever (≤0.45 ms + W4 wedges 2/2 today); loop lever ~5 ms, post bounded-wait GREEN leg only | design doc §1 |
| fold-argmax-into-allreduce | **KEPT ONLY under the coordinator's criterion: a fold counts if it removes a DRAIN/sync point, not bytes.** On the default route NO fold qualifies (there are no per-call drains; F2a removes a local KERNEL, not a sync; F2b keeps the call count). On a future oneshot-route world the criterion activates — re-derive then | §C |

---

## C. Fix designs (SKETCHES ONLY — src/ untouched per law)

### F1 — kill the dead greedy gather (3 collectives → 0 in greedy): CLEAN WIN, DEMOTED LEVER

Rev-1 predicted the saving = the lgather field. Rev-2 correction under wait-on-root: skipping
the gathers removes the re-sync POINT, but the ranks1-3 wait RELOCATES to ar_argmax (they will
wait there for rank0 instead). The net round saving = the ROOT's own gather cost ≈ **rank0's
0.78 ms** (rank0 is the critical rank — its lm_head is the latest). Secondary wins: 1.49 MB of
useless per-round buffer writes deleted, 3 fewer lockstep points, and the temp>0 route keeps
its gather. Land it as cleanup riding any LMHEAD window; do not spend a window on it alone.

```diff
--- a/src/runtime/tp2/tp2_backend.cpp   (single-seq MTP round loop, before :3052)
+++ b/src/runtime/tp2/tp2_backend.cpp
@@
-                    if (tail_trace.on) { tail_trace.op_begin(s, ninfer::tp2::VerifyTailTrace::LGATHER); }
-                    {
+                    // F1 TAIL-ARTAIL: verify_logits_full's ONLY reader is the sampling
+                    // branch of speculative_accept_greedy_drafts_kernel (greedy arm
+                    // returns before any logits read). Greedy = the 3 AGs are dead
+                    // dataflow. Predicate mirrors the kernel's own greedy arm;
+                    // NINFER_D22_PDBG (reads the buffer) keeps it armed.
+                    const bool accept_is_greedy =
+                        !(req.sampling.temperature > 0.0f) || req.sampling.top_k == 1;
+                    const bool need_full_logits =
+                        !accept_is_greedy || getenv("NINFER_D22_PDBG") != nullptr;
+                    if (tail_trace.on && need_full_logits) { tail_trace.op_begin(s, ninfer::tp2::VerifyTailTrace::LGATHER); }
+                    if (need_full_logits) {
                         const std::int32_t nv = (std::int32_t)st.verify_logits.ne[0];
                         for (int j = 0; j <= rk; ++j) {
                             backend.group().allgather_local_bf16( ... );   // unchanged body
                         }
                     }
-                    if (tail_trace.on) { tail_trace.op_end(s, ninfer::tp2::VerifyTailTrace::LGATHER); }
+                    if (tail_trace.on && need_full_logits) { tail_trace.op_end(s, ninfer::tp2::VerifyTailTrace::LGATHER); }
```

`req` is in scope (host_cfg built from it at :1888-1895). Follow-on: the batched MTP twin
(:6819 region), same gate, if not already gated.

### F2a — delete the redundant local verify argmax on the TP route (launch-count cleanup; NOT a drain fix)

Per the coordinator's criterion this stays ONLY as cleanup: it removes a local kernel + one
372 KB read, zero sync points. Sketch unchanged from rev 1 (gate text_context_impl.h:1872-1874
on `tp_group_ == nullptr || tp_rank_ < 0 ||` the diagnostic envs). F2b (feed the impl's local
champions into R1, skip kernel-1) stays PARKED — it saves bytes, which the refocus correctly
demotes, and removes no call.

### F3 — the fold that COULD qualify (one sync point fewer): merge the draft-head's GEMM with the R1 exchange? NO — merge the ROUND's first two sync points? NO. Honest census: none qualifies today

Stated plainly so the criterion is on record: the tail's only eliminable sync point is the
lgather→ar_argmax pair in greedy (F1 deletes the pair's first member; the second must stay —
it produces target_tokens). The align/chain ARs are dependence-ordered (§B.3). The accept
drain is the D2H semantic point. The one-shot status drains do not run on this route. **Net:
no fold-into-allreduce fix qualifies under the drain criterion in this posture.**

### F4 — the actual lever class for the ≥14.3 ms residue: de-skew / de-starve the post-drain segment

Mechanism-agnostic form (safe under either M-skew or M-starve): **shorten the host path from
the accept drain (:3222) to the last tail-collective enqueue (chain loop end, :3655)**, and
make it rank-UNIFORM:

- F4a: defer rank-asymmetric host work (rank-0 token callback, prints, vocab counter, the
  [TAIL] collect/print) to AFTER the chain enqueue — rev-1's F6 sketch, blocks named by
  anchor (:3283-3348 conf-arm, :3360-3399 history/ngram, :3404-3423 rank-0 emissions; conf-ring
  depth safe: deferred reads target the previous staged span, 32-slot ring wraps at >8 rounds,
  impossible at k_buf ≤ 6).
- F4b: cut the segment's launch count (the ~25-35-launch mtp_forward_decode_batch is walked
  TWICE — align + chain step — after a cold pipeline; graph-capture/replay of the draft-side
  segment is the M1-named lever class).
- F4c: if D2 says M-skew, add the cheap symmetricizer first: rank0's per-round [TAIL] printf +
  fflush alone is a stdout-pipe stall candidate; move ALL per-round prints to a request-end
  flush (tracer-internal change, byte-identical decode).

DECISIVE CHECKS for F4 are §D's D2/D3 — F4 lands only after they read.

### F5 — ADJACENT BUG FLAG (unchanged): token_domain = `2 * nv` at :3182 is the world=2 literal

At TP4 the true sampling domain is 4·nv = 248,320. Dead in greedy (token_domain unused in the
greedy arm), WRONG in temp>0 (sampling would see half the vocabulary). One-line fix
(`backend.world() * nv`) rides its own RED/GREEN on a temp>0 TP4 cell, never folded into F1.

```diff
-                        2 * (std::int32_t)st.verify_logits.ne[0], st.sample_cfg, st.work, s);
+                        static_cast<std::int32_t>(backend.world()) *
+                            static_cast<std::int32_t>(st.verify_logits.ne[0]),
+                        st.sample_cfg, st.work, s);
```

### F6 — TAIL_BOOK FIX-A/FIX-B disposition (cross-ref per coordinator)

FIX-A (deferred-consume of one-shot status) / FIX-B (single-drain) are correctly designed
against the sinks they cite — but those sinks run only at world==2 or under
`NINFER_TP_ONESHOT_AR=1`. Disposition: **PARK both specs against the oneshot-gated world**
(where they also collide with the ONESHOT_W4_BOUNDED_WAIT_DESIGN.md ordering: B1/B2/B3 bounds
and the 6 GREEN-leg cells come first, else the swap wedges 2/2). Do not aim them at the
default W4 tail; the default route's residue is F4's.

---

## D. DECISIVE CHECKS (centerpiece per refocus) — split device-collective vs drain-wait inside the measured spans

**D1 — route/sink census (zero-cost, first):** on the M2 banked log + binary: (a) grep
`AR-RETRY|AR-FAILOUT|NINFER_TP_ONESHOT_AR` — expect ZERO hits on the default-route boot
(any hit falsifies §B.2's unreachable-sink claim and revives TAIL_BOOK's mechanism); (b)
assert the strings census of the banked bin shows the one-shot symbols present-but-ungated
(they compile in; the GUARD is the constructor gate, tp_group.cpp:144-149 — cite, don't
strings-alone). Expected readout: per-round hard syncs = 1 (accept) on the default route.

**D2 — per-rank span split (zero new code; THE discriminator):** the [TAIL] lines are
per-rank and already banked. Read align/chain_fwd per rank:
- LARGE-on-early-ranks / small-on-last (the lgather 4.92-vs-0.78 shape) ⇒ **M-skew**: the
  residue is peer-wait at RCCL ARs; cure = F4a/F4c (rank-uniform host path), NOT drain fixes;
  predicted align+chain_fwd 16.2 → ~2-4 ms after cure.
- LARGE-equally-on-all-ranks ⇒ **M-starve**: each host paces its own device; cure = F4b
  (graph-capture of the draft-side segment) ± launch-count diet; same predicted band.
- Either way, ANY per-rank-minimum ≥ 1.5 ms that survives ⇒ real device time hides there,
  CONTRADICTING this desk's floor (§B.1) — falsifier duty: re-derive the floor (the suspect
  would be the mtp_forward_core weight-read class, not the collectives).

**D3 — the one-call-fewer A/B (F1), now with the honest expectation:** greedy boot ±F1,
count600 conc=1, tracer ON. Expected: `lgather` field 0.000 all ranks; rank0 round −0.5-0.8 ms;
ranks1-3 ar_argmax field GROWS by ~the relocated wait (predicted +3-4 ms) with ROUND time
unchanged on those ranks — this relocation readout is itself the wait-on-root proof row. Token
byte-identity ([ids] stream) + temp>0 leg keeps lgather > 0 (both-direction falsifier). Closure
cell joins the boot battery: greedy ∧ lgather==0 ∧ ids-identical.

**D4 — LMHEAD handoff (the tail's real clock):** the collective desk's residue (≤2.2 ms) +
F1 (~0.8) + F4 (≤14 ms if D2 reads favorably) bound what ANY collective-side fix can recover;
the tail's owner is lm_head 61.93 → the fb86a0fdd route-flip A/B (small_t vs
launch_w8_simt_r8_c4, T={1,3}) decides the 50-tps claim. Round-level predicted bands:
collective-side total recovery 1-15 ms; LMHEAD-side 57-60 ms. The two are independent levers.

---

## E. Honesty block

1. **No silicon touched by this desk.** Measured numbers are quoted from the banked M2 row +
   PLOG-043 (9cee65ae2) and M1 row; everything else is code-derived floor or labeled
   prediction. Clean-band treatment of absolute-ms rows is assumed per the droop finding.
2. **The §B.2 census correction of TAIL_BOOK is a desk-vs-desk claim, and I state my
   falsifier first**: a single [AR-RETRY]/[AR-FAILOUT] line, or any evidence the M2 posture
   set NINFER_TP_ONESHOT_AR, revives the 8-drain mechanism and my correction is WRONG. The
   guards I cite (tp_group.cpp:144-149, :346-348, :436) are one read away for any reviewer.
3. M-skew vs M-starve is a PRIOR (65/35), not a finding; D2 is one log-read away and I have
   not read the per-rank align/chain_fwd columns (they were not in the material reachable to
   this desk; if they exist in VERIFY_TAIL_row.txt the desk that reads them settles §B.2 for
   free).
4. The align AR payload (10,240 B) is read off the T=1 arm of mtp_forward_core; at align's
   width-3 call the ARs may move 3× that. Changes wire µs, not the floor's verdict (compute
   dominates the floor there).
5. F1's greedy byte-identity is structural (zero readers cited by line) but the closure law
   wants the measured row (D3). F4a/F4c defer host blocks whose only couplings are the conf
   ring (depth argument in rev-1 §F6, still: 32 slots, ≤8 rounds distance, k_buf ≤ 6) and
   next-round host consumers; the landed diff must re-verify the W5 A→B barrier pairing
   survives the move before any boot.
6. This doc closes NO bug. F1's dead-gather finding is a measured-waste claim whose RED row is
   the banked lgather field and whose GREEN row does not exist until D3 runs (RED→GREEN
   CLOSURE LAW). The §A.3 dual-argmax and §F5 domain findings are flagged, not fixed.
