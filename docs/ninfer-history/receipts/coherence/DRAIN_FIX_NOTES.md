DRAIN_FIX_NOTES — scope renamed per coordinator STOP-and-PIVOT (2026-09-17): M2 skew/starve
verdict (ARTAIL D2), not a FIX-A/FIX-B landing. Lane amd/tp4-cure, CODE desk, no-GPU window.

== 0. MISSION STATE ON THE PIVOT (what was reverted / kept) ==

- FIX-A/FIX-B (TAIL_BOOK 2026-09-17, ebfb2eaf4): **HALTED BEFORE ANY EDIT.** The desk had
  completed only the read/design phase (TAIL_BOOK + tp2_backend.cpp round loop :2890-3677,
  round_state.cpp:1328 aliasing, tp_group.h / one_shot_*.h contracts, ARTAIL rev 2) when the
  STOP arrived. Zero lines were written to src/ or tools/ — nothing to revert; `git status`
  src/ shows only another agent's in-flight `w8_dispatch.cpp` edit, untouched by this desk.
- Host-only deferred-consume FSM cell: **NOT written** (order: keep only if already written
  and selftested — it was not). Nothing to park. The cell spec (stamp/slot/status/retry FSM,
  RED = entry-check death on a deferred leftover, GREEN = retry-before-consumer, K=16 loud
  bound) stays in TAIL_BOOK §(c)/cell spec for the future NINFER_TP_ONESHOT_AR world and is
  now double-parked behind ARTAIL's bounded-wait GREEN leg.
- Corroboration read (D1, zero-cost, run anyway): `grep -c` on the banked M2_serve.log →
  AR-RETRY **0**, AR-FAILOUT **0**, NINFER_TP_ONESHOT_AR **0** — the one-shot sinks are
  silent on the default-route boot exactly as ARTAIL §B.2 claims; no evidence revives the
  drain mechanism.

== 1. D2 — PER-RANK SPAN SPLIT (THE DISCRIMINATOR), LEG 2 CLEAN BAND ==

Parse: per-rank `[TAIL]` rows from M2_serve.log (leg 2 = the 204-round request ending at
[finish-trace] line 34803; begins line ~24254), joined by (rank, round) with
`[OPTRACE] phase=verify ms<=125` (the VERIFY_TAIL_row clean band). Parity with the banked
row achieved before splitting: n=278 rank-rounds, 4-rank means align 8.276 / chain_fwd 7.896
/ chain_head 0.317 / propose 0.320 / lm_head 61.85 / lgather 5.00 — all within 0.1 of
VERIFY_TAIL_row.txt. One torn stdout-interleave line (leg1 line 15793) skipped; all other
rows carry the full field set. Per-rank means, ms/round:

  field        rank0    rank1    rank2    rank3   4-rank-mean  max/min
  align        8.303    8.259    8.269    8.272     8.276       1.01x
  chain_fwd    7.900    7.891    7.891    7.902     7.896       1.00x
  chain_head   0.314    0.322    0.320    0.314     0.317       1.03x
  propose      0.320    0.323    0.319    0.319     0.320       1.01x
  lgather      0.829    7.077    6.868    5.234     5.002       8.54x  <- the skew shape
  lm_head     66.034   59.723   60.003   61.627    61.847       1.11x
  ar_argmax    0.201    0.198    0.206    0.211     0.204       1.07x

Medians (clean band) match means within 1% per rank per draft field — not a tail artifact.
Unclean leg-2 rounds tell the same shape story (align 10.75-10.93 across ranks, 1.02x).

**D2 READ: align/chain_fwd are RANK-UNIFORM to 0.5% (max/min 1.00-1.01x) while lgather on
the SAME rows shows the classic wait-on-root split (8.54x, rank0 0.83 vs ranks1-3 5.2-7.1).
Neither doc branch's signature fits — and the ARTAIL falsifier fires:**

- M-skew (peer wait, lgather-shaped: LARGE early ranks / SMALL last-arriving root): REFUTED.
  rank0 — the lm_head laggard every wait story leans on — has the LARGEST align span
  (8.303, +0.4% over the field), not the smallest. A wait-on-rank0 mechanism requires rank0
  small the way lgather's 0.83 is small; the observed spread is 0.04 ms where lgather's is
  6.2 ms on the same rounds. There is no lgather shape in the draft spans.
- M-starve (self host-pace): REFUTED by two independent banked controls:
  (a) host-exclusion: `chain_enq` — host wall inside the chain span — is 56-73 us/round
      (all ranks) against a 7900 us device-event span. The host finished enqueueing the
      entire chain loop ~7.8 ms before the span's end event; the device cannot have been
      waiting on its own host inside the span. (TAIL_BOOK's 25-35-launch x 5-15 us pricing
      is itself refuted downward: real enqueue is ~2 us/launch here.)
  (b) clock-law: host-class fields are CLOCK-FLAT (chain_enq leg1/leg2-clean = 1.00x, book
      0.95x) while align and chain_fwd SCALE HARD with sclk — see the control table below.
- Falsifier branch (ARTAIL §D D2, verbatim trigger: "ANY per-rank-minimum >= 1.5 ms that
  survives => real device time hides there, CONTRADICTING this desk's floor (§B.1)"):
  **FIRED — align min 8.259, chain_fwd min 7.891, both >= 1.5 ms, rank-uniform.**

== 2. CLOCK CONTROL (device-time confirmation; the falsifier's re-derive duty) ==

Per-rank means, deep-droop leg 1 (ALL rounds, sclk collapsing to 2-5 under burst) vs leg 2
clean band; ratio of mean-of-rank-means:

  field       leg1(ALL) mean   leg2(CLEAN) mean   ratio     clock law
  lm_head         87.87           61.85            1.42x    device (rank0 153.76/66.03 = 2.33x, M2's own clean-vs-clean row)
  align           18.76            8.28            2.27x    DEVICE — scales like the proven compute-bound lm_head
  chain_fwd       17.96            7.90            2.27x    DEVICE
  propose          0.62            0.32            1.93x    device (GEMV + argmax kernel)
  chain_head       0.63            0.32            1.97x    device
  ar_argmax        0.31            0.20            1.53x    device
  lgather         68.08            5.00           13.61x    WAIT (positive control: 91/88/90 ms ranks1-3 = rank0's 153.8 lm_head excess)
  chain_enq        0.06            0.06            1.00x    HOST — clock-flat negative control
  book             0.11            0.11            0.95x    HOST — clock-flat negative control

Internal leg-2 split agrees: clean vs droop rounds (same leg, same boot) → align 1.50x,
chain_fwd 1.51x, lm_head 1.18x (rank-mix), sync 1.42x; a wait-class span does not scale
with clocks, and a host-paced span cannot (the hosts are clock-flat in the same table).
2.27x at the box's ~2.3x droop = COMPUTE-BOUND — the same law M2 used to convict lm_head
(leg1/leg2 2.33x == clock ratio).

Honest residual: a coupled AR wait with perfectly-aligned arrivals is numerically
indistinguishable from own-compute at 0.5% spread — but then the span still EQUALS the
per-rank T=1 compute, and the cure is identical (make the T=1 forward faster). Either
reading kills de-skew and round-restructure; nothing in the banked data supports a >40 us
wait component.

== 3. VERDICT ==

**NEITHER M-skew NOR M-starve — M-COMPUTE (the falsifier branch).** The draft-side
16.5 ms/round (align 8.28 + chain_fwd 7.90 + chain_head 0.32 + propose 0.32, clean band) is
REAL, RANK-UNIFORM, CLOCK-SCALING DEVICE TIME of the T=1 draft-side forwards — not wait.

Floor re-derivation (falsifier duty, as the doc predicted): ARTAIL §B.1's "<=2.2 ms device
admitted" premise was the error — it priced the mtp_forward_core weight read at healthy
memory rates. At the rate LMHEAD convicted for the same box's skinny-T W8 path (lm_head
337.7 MB / 61.93 ms = 5.45 GB/s effective, a tensor-core schedule emulated as SIMT mma_bf16
on gfx900), the align span alone implies ~45 MB of weight read per forward per rank — the
right order for one 5120-hidden W8-quantized layer shard at TP4 read by a T=1 pass. The
T=1 draft forwards appear to ride the SAME issue-bound skinny-T kernel class as lm_head
(exact shapes/arms are D4's roofline A/B to confirm — byte-count consistency check here is
order-of-magnitude, not a derivation).

Desk consequences:
- D2 answer to ARTAIL §D: **falsifier branch, not either named hypothesis.** F4a/F4c
  (de-skew, rank-uniform host path) DEMOTED — there is no skew to remove (rank0 is already
  the max span; uniformity is 0.5%). F4b (graph-capture of the draft segment) DEMOTED as a
  ms-lever — it can hide only the 56-73 us host gaps; the 16 ms is device execution and
  graph replay runs the same kernels at the same clocks. The prior "M-skew 65 / M-starve 35"
  splits 0/0 into a third cell.
- The 16.5 ms move to the LMHEAD-class lever: route the T=1 MTP-layer linears off the
  skinny-T issue-bound arm (same lever family as the lm_head simt_r8_c4 flip). If the T=1
  forwards reach even the 3x band, align+chain_fwd 16.2 -> ~5.5 ms; at a 10x band -> ~2 ms.
  D4's roofline A/B (T={1,3}) already measures exactly this arm — the draft-side fields
  ([TAIL] align/chain_fwd per rank) are its readout, no new instrument needed.
- Round-total bookkeeping: tail owner remains lm_head 61.93 ms (LMHEAD desk); the draft-side
  16.5 is now ALSO device-compute (not recoverable by collective/orchestration work). The
  only orchestration-class ms left in the whole tail: lgather ranks1-3 ~5 ms (collapses
  automatically when rank0's lm_head lands) and ar_argmax/heads < 1 ms combined.
- FIX-A/FIX-B stay parked per ARTAIL rev 2 (unreachable sinks at TP4 default), now with a
  second independent reason: even on a oneshot-gated boot, the 16.5 ms these fixes targeted
  would be device compute, not drain-wait — their predicted yield was priced against wait
  that does not exist. Re-price before any unpark.

== 4. REPRODUCTION ==

  # D1 (route census):
  grep -c "AR-RETRY\|AR-FAILOUT\|NINFER_TP_ONESHOT_AR" results/amd/coherence/M2_serve.log  # 0
  # D2: parse per-rank [TAIL] from M2_serve.log; leg 2 = 204-round request after the
  # gen=16 warmup ([finish-trace] line 24240; [TAIL] rows from ~line 24254); join
  # (rank,round) with [OPTRACE] phase=verify ms<=125; split align/chain_fwd by rank.
  # Parity gate: 278 rank-rounds; 4-rank means must land on VERIFY_TAIL_row.txt within 0.1 ms.

Controls banked: leg1-vs-leg2-clean clock table (sec 2); internal leg2 clean/droop split;
host-exclusion chain_enq; lgather positive wait-control; chain_enq/book negative controls.
All numbers from the already-banked M2 window (bin ninfer-serve_248630d1b110c2bf) — zero new
runs, zero new code.
