# WO: W5 break-path deadlock — shortened round hangs at the next chain step

**Status:** OPEN defect, filed by agent1 2026-09-07. Parent scope: docs/optimizations/45 §6 (W5).
**Ships:** W5 stays OFF (`NINFER_MTP_CONF_TAU` unset). W0–W4a merge is NOT blocked by this WO.

## 1. Symptom (observed, reproducible recipe below)

With `NINFER_MTP_CONF_TAU=0.5 NINFER_MTP_CONF_DBG=1`, adaptive depth (k=3 base, max 7),
K4V4, ctx-8000 greedy prompt:

- Rounds 0–22 run normally (conf prints sane: 0.34–0.999).
- Round 23: first real break — `[conf-break] round 23 depth 3->2`.
- Round 24: top-of-round `rk=2 T=3 src=adaptive` (verify-depth log), verify + accept +
  prep + align + select + d0-propose all COMPLETE (`[conf] d0 conf=0.9567` printed).
- Round 24 chain step 0: **hang**. GPU pinned at 100% on both cards (spin-wait), request
  never returns, SIGTERM ignored (spin kernel), needs SIGKILL of the server pid.

So the shortened round itself verifies correctly; the hang is one step later, inside the
next chain iteration.

## 2. Why a hang (not wrong output) implies rank asymmetry

The one-shot argmax (`one_shot_argmax.cu`) is a lock-step handshake keyed on a per-rank
`rank_step` counter → slot = step % 32, epoch = step/32+1. If the two ranks make a
DIFFERENT number of `allreduce_argmax` calls, rank A's kernel spins forever on
`peer_flag >= expected_epoch`. A hang at exactly the first post-break chain step means the
ranks took different control flow at the break boundary. The break decision is host-side
and must be bit-identical on both ranks; everything downstream (chain length, extents,
drafts) derives from it.

## 3. Hypotheses (rank order = my current suspicion)

**H1 — conf value divergence between ranks.** The gate breaks iff `prod_conf < tau`.
`prod_conf` is built from `one_shot_argmax_last_conf(rank)`, which reads the rank's OWN
host-mapped slot `slots[(rank_step-1)%32].conf[rank][0]`, written by the kernel as
`1/S` with fixed rank-0-first combine order. Divergence sources to check:
  a. The 16 B payload exchange (`st_volatile_payload` v4 wt / `ld_volatile_payload` v4 cv):
     is a 16 B write to *host-mapped pinned* memory observed atomically by the peer GPU
     after the flag handshake, or can the peer's `ld.global.cv` see a torn payload
     (val/tok from epoch N, sumexp from epoch N-1)? The old 8 B payload predates the
     sumexp field; 16 B crosses no page but the wt/cv guarantee on mapped host memory is
     the suspect. A torn sumexp on ONE rank → different S → different conf → one breaks,
     one doesn't. **This is my leading hypothesis.**
  b. `rank_step` skew: any argmax call site executed by one rank only (sampling arms,
     reuse-path d0 proposals at the prefill branches that do NOT pass emit_conf) shifts
     which slot `last_conf` reads. If rank X reads a stale slot (conf=0 from init or an
     old value), it breaks unilaterally.
  c. fp32 combine order: verify the emitted conf is bit-equal by construction (it should
     be: same four operands, same order) — cheap to confirm in H2's unit sim.

**H2 — pending_depth / extents write race vs the next round's drafts2 view.**
`write(st.extents, n_draft)` at chain end is an async H2D from a *stack local*
(`cudaMemcpyAsync(t.data, &value, ...)` — the existing `write` lambda pattern). If the
copy executes after the local dies (scope exit before stream sync), extents gets garbage
→ next round's device-side width disagrees with the host rk. Both ranks have the same
code, so this alone shouldn't desync — but it can produce a width mismatch that deadlocks
the verify/argmax pairing differently per rank. Check: does the chain-end write even need
to exist (the next round's top rewrites extents from the host rk anyway)?

**H3 — epoch pairing across a break.** After a break, round 24's chain runs fewer argmax
calls than the k_buf-based buffers/graph assumptions? No graph here. But `reset_step`
is called per REQUEST, not per round — verify the slot ring (32 slots) cannot wrap
mid-request into a slot whose peer flag is still at an older epoch when chain lengths
vary (rk=2 rounds consume fewer slots → wrap timing differs from full-depth runs).
32 slots × ~4 argmax/round ≈ 8 rounds before wrap; a 25-round request DOES wrap. If
wrap interacts with the break (fewer calls → different slot at the same round index),
a stale-flag race becomes possible exactly after depth changes.

## 4. Decisive checks (ordered, cheap first)

| # | Check | GPU? | Decides |
|---|---|---|---|
| C1 | Re-run the repro with `NINFER_MB_ARGMAX_TRACE=1` (exists, one_shot_argmax.cu L266 — prints rank/step/slot/epoch/T/n_rows per call). Diff the two ranks' traces around the hang: the first divergent step count IS the desync point; its call-site label says which branch diverged. | short (~3 min) | H1b vs H1a/H3 |
| C2 | Remove the `rank == 0` guard on the `[conf]` print (one line), re-run repro: if the two ranks print DIFFERENT conf for the same d0/step → H1a (torn payload) proven; if identical → H1a dead, control-flow divergence confirmed instead. | short (~3 min) | H1a |
| C3 | Unit sim (no GPU): extend `tests/test_mtp_confidence_break.cpp` — feed two `MtpConfidenceBreak` instances the same conf stream with a ±1 ULP perturbation on one; show a single-ULP conf difference near tau flips the break and produces different chain lengths (quantifies how tight H1a must be; expected: yes, trivially). | none | H1a plausibility |
| C3b | **H1a mechanism refinement (code-read, this WO):** the conf write `my_conf[t] = 1/S` happens AFTER the flag handshake, so the peer's flag can reach rank0 before rank0's conf store lands. A rank that syncs fast reads its own conf at a stale value (0.0 at init → unconditional break; or the previous epoch's value). This is a host-visible write race, not a payload tear — it explains "hangs exactly at the first break" and predicts the divergence direction (the FASTER rank breaks spuriously). Fix candidates: (i) write conf into the SAME payload slot the peer already polls (peer reads it via the exchange — but the write must precede the flag, and 1/S needs the peer's partials → needs a second flag epoch); (ii) simplest correct: after the chain's LAST argmax the host does a 4-byte `cudaMemcpyAsync` D2H of the conf slot on the stream (ordered after the kernel), then `cudaStreamSynchronize` — stream-ordered copy removes the visibility race entirely (the mapped-memory read is not stream-ordered). | none | H1a design |
| C4 | Static audit (no GPU): grep every `allreduce_argmax` call site; annotate which are unconditional-on-both-ranks vs rank-gated vs path-gated (reuse vs redo prefill branches, temp>0 arms). Any path-gated call that can differ across ranks under a break = H1b. | none | H1b |
| C5 | Delete the chain-end `write(st.extents, n_draft)` (redundant — next round top rewrites from host rk), re-run repro. Still hangs → H2 dead. | short | H2 |
| C6 | Slot-ring stress: set `kNumSlots=8` (test-only build), run tau-unset (gate off) with adaptive depth for a long request — if wraps alone can deadlock at depth changes, this reproduces WITHOUT W5, exonerating/implicating H3 independent of the conf path. | short (~10 min) | H3 |

## 5. Fix gate (any hypothesis)

1. Repro recipe runs ≥ 5 requests × 256 tokens, tau ∈ {0.35, 0.5, 0.7}, adaptive + fixed
   k=3, zero hangs, zero SIGKILLs.
2. Greedy byte-identity tau-on vs tau-off at ctx 500/5000/20000 (the prefix argument —
   must hold once the gate actually breaks; this was never truly tested because the
   broken conf formula pinned the gate open).
3. `serve_batched_ci.sh` green (the break must not desync lanes; lanes path is phase 2 —
   gate must stay single-request-only until then).
4. Re-measure the cost finding with a WORKING gate: ms/round tau-off vs {0.2,0.35,0.5}
   on the mix; the +13 ms/round launch-ahead loss may be offset by real saved steps at
   depth ≥ 5. Report ms/round, not tok/s. Update docs/45 §6 with the verdict either way.

## 6. Repro recipe

```
env NINFER_NO_FRONTIER=1 NINFER_MTP_CONF_TAU=0.5 NINFER_MTP_CONF_DBG=1 \
    NINFER_VERIFY_DEPTH_LOG=1 setsid build/apps/ninfer-serve \
    /home/intel/models/qwen3_8_27b.ninfer --host 127.0.0.1 --port 8099 --devices 0,1 \
    --kv-dtype kvarn_k4v4 --kv-capacity 100000 --max-context 80000 \
    --spec mtp --draft-tokens 3 --mtp-adaptive --mtp-draft-max 7 --no-prefix-reuse \
    --model-id qwen3.8-27b
python3 tools/ops/mtp_byteid.py http://127.0.0.1:8099 8000 256 /tmp/out.txt diag
```
Binary: `wo/radiance-launch-gap` @ 7bf8cbb7 (post conf-fix). Observed hang point:
round 24 chain step 0, first break at round 23. (Break round varies with prompt; any
first break can trigger it.)

## 7. What is NOT the bug (already excluded)

- The conf formula (exp(M−log S) overflow) was real and IS fixed (1/S, 7bf8cbb7) — it
  explains the sweep's zero-breaks, not the deadlock (deadlock appeared AFTER the fix,
  with real breaks).
- The verify-at-shorter-width machinery itself: round 24's rk=2/T=3 verify completed and
  accepted correctly; per-round views + extents (docs/69 plumbing) work.
- W1/W2a/W4a: hang requires tau>0 (gate on); tau-unset runs of the same binary are clean
  (byteid gates, all censuses).

## 8. Progress log (agent1, same day — CPU checks done, GPU verification pending window)

**C4 (static audit) — DONE, H1b EXONERATED at call-site level.** Every `allreduce_argmax`
in the single-request MTP path (L1535 redo-d0, L1659 common-d0, L1708 round-0 chain,
L2151 verify, L2365 steady-d0, L2418 steady chain) executes unconditionally on both rank
threads; no `rank ==` guard wraps any call. The redo path proposes d0 twice (1535+1659)
but symmetrically on both ranks. `reset_one_shot_argmax_step` is per-request, both ranks.
=> rank_step cannot diverge from call-site asymmetry; it can only diverge if a rank takes
different control flow (i.e., a different break decision) — which points at H1a.

**C3 — DONE (test_ulp_sensitivity_vs_desync in test_mtp_confidence_break.cpp, green).**
One ULP in the observed conf at the tau boundary flips should_break() on one rank only.
Decision inputs must be shared-state, not per-rank computed.

**H1a mechanism, sharpened by code-read (supersedes C3b's guess):** the device-written
conf (`my_conf[t] = 1/S` in the argmax kernel) executes AFTER the peer-flag poll, so it
has no cross-rank ordering guarantee; and each rank's host reads its OWN device-written
slot — two independent writes of "the same" fp32 combine. Any non-determinism in the
device path (e.g. a peer payload read racing a slot reuse at the 32-slot wrap, H3-adjacent)
diverges the ranks directly. The payload exchange itself IS ordered (write -> fence ->
flag; poll -> read), so the payloads are the safe shared state.

**FIX IMPLEMENTED (pending repro): decision conf computed HOST-side from the exchanged
payload pair.** `OneShotArgmax::conf_from_payloads(rank)` reads both ranks' pinned
payload slots for the last call and runs the identical fp32 combine (rank-0-first order)
on the host; both hosts read the SAME two 16-byte payloads -> bit-identical decision by
construction, no device-write ordering assumptions. The four observe sites now use it;
`[conf]` prints both sources (`conf=... (dev=...)`) — if they ever differ, the device
path's non-determinism is caught red-handed (C2 folded into the shipped debug).

**Remaining before this WO can close (needs GPU window, ~10 min):**
1. Repro recipe x5 requests, tau {0.35,0.5,0.7}, adaptive + fixed k=3: zero hangs?
2. If still hangs: C1 (NINFER_MB_ARGMAX_TRACE both-rank diff pinpoints the first
   divergent call) and C2' (compare conf vs dev= in the [conf] lines).
3. If clean: tau-on greedy byte-identity (ctx 500/5k/20k) — the gate never truly had
   this test (broken conf pinned it open) — then the re-measure sweep (gate item 4).
4. C5/C6 ablations only if the hang survives the fix.

## 9. Round-0 / drafts1 / extents audit (CPU, 2026-09-07 — bc42b29a)

Static trace of every consumer of a shortened round-0 proposal:
- `mtp_prepare_next_round` (src/ops/wrapper/mtp_round.cpp): consumes per-round
  self-describing views (verify_ids {rk+1,1}); writes next_extents, which the next
  round's top OVERWRITES via host `write(st.extents, rk)` with rk = pending_depth.
  Consistent at both chain sites. T∈[2,8] guard: rk=1 (break before step 0) gives
  T=2 ✓ in range.
- Round-0 "Pack drafts" loop is a SELF-COPY no-op (ar_drafts[i] ARE drafts1[i+1]
  slices); stale tail slots [n_draft, rk) are never viewed because drafts2 is
  {rk_actual,1}. first_drafts.resize(rk) D2Hs garbage tail into a stats vector —
  cosmetic (rank0-only diagnostic), NOT control flow.
- use-lookup overwrite replaces the full rk from host-computed drafts — deterministic
  and rank-symmetric.
- Reuse vs redo prefill branches: both converge on the common d0-propose (L1659,
  emit_conf=conf_gate); the redo-only L1535 call is symmetric across ranks.
=> No static consumption divergence found. The mechanism must be observed, not read:
   the REAL C2 run (both-rank [conf rN] prints, guards actually removed in bc42b29a)
   discriminates (i) rank1 computing a different prod from (ii) identical decisions
   with divergent calls. If (i): per-rank state upstream (slot read timing). If (ii):
   the break bookkeeping itself.

## 10. W5-ON gate script

tools/bench/radiance_w5_gate.sh — hang-check x taus {0.35,0.5,0.7} x {adaptive, fixed},
break counts, conf-vs-dev divergence scan (correct parser), tau-on greedy byte-identity
vs the tau-off reference captured IN THE SAME RUN. Run it after the mechanism fix lands.

## 11. REAL C2 verdict (run 15:24-15:33Z, log /tmp/c2r2.log, bc42b29a binary)

Both ranks, warmup request, tau=0.35, adaptive:
```
[conf r0] step 0 conf=0.1062 (dev=0.1062) prod=0.1062 tau=0.35
[conf r1] step 0 conf=0.1062 (dev=0.1062) prod=0.1062 tau=0.35
[verify-depth] round 0 rk=2 T=3 src=adaptive
```
argmax trace: step0+step1 PAIRED on both ranks, zero unpaired. Then spin (100% util,
never reached listening; killed).

VERDICT: **(ii) — decisions are IDENTICAL (both ranks break at the same point, both
shorten round 0 to rk=2/T=3), and the deadlock is INSIDE the shortened round-0 verify
forward itself**: the next argmax (round-0 verify, would be step2 T=3) is never launched
by EITHER rank => the hang is inside target_verify_batch at T=3, before its argmax.
H1a/H1b (decision divergence) are DEAD as mechanisms.

Narrowed suspect set for the shortened-verify hang (next diagnostic, needs a window):
 (a) one-shot AR epoch/slot pairing across the chain-exit->verify transition: the break
     skips chain AR calls; if any per-rank AR counter absorbed an extra call earlier
     (e.g. the round-0 d0-propose at L1535 redo-branch fires on one path only —
     reuse-vs-redo is HOST-deterministic, but check the MTP-KV seed-reuse branch),
     verify's AR handshakes pair against the wrong peer epoch -> spin.
 (b) the round-0 pack loop + stale drafts1[2] (nominal rk=3 copied 3 entries, verify
     consumes 2): verify_ids built from drafts2{2,1} — but mtp_kv/positions were seeded
     for 3 drafts at prefill (prepare_mtp_prompt); a width mismatch there could make
     one rank's attend read a slot the other never wrote.
 (c) decisive cheap test: tau=0.35 but force the break to be a NO-OP (patch
     pending_depth to never < rk) -> if it still hangs, the bug is in the break
     bookkeeping, not the shortened verify; if clean, (a)/(b) remain.
NOTE the earlier c1b run (steady rounds, break at round 12) hung AFTER a completed
shortened round 24-verify — so shortened verifies CAN complete; the warmup case
shortens ROUND 0 specifically (pre-KV-seed interplay) — (b) moves up in priority.

## 12. Discrimination matrix (GPU runs 16:09-16:30Z, payload-conf binary + round-0 NOOP temp patch)

| run | round-0 shorten | steady shorten | result |
|---|---|---|---|
| c2r2 (tau .35, no patch) | YES (break in pre-loop chain) | not reached | HANG in round-0 verify (step2 never launched, both ranks) |
| w5c  (tau .35, NOOP)     | no | none fired (prod stayed >= .35) | LISTENING, 64-tok req OK (sha bcaf0135) |
| w5c7 (tau .7, NOOP)      | no | YES (round 0 -> rk=2 at round 1) | 64-tok req COMPLETED, shortened verify fine (same sha) |

CONCLUSION: the defect is SPECIFIC to shortening the PRE-LOOP round-0 chain (the block
before the round loop). Steady-round shortening (the round-loop chain-exit -> next-round
verify path) works on the payload-conf binary — c1b's steady hang was the OLD
device-conf decision divergence, dead since bb26f9e3. Prime mechanism candidate (code
read, unconfirmed): the pre-loop chain-exit sets pending_depth but does NOT rewrite
st.extents (the steady site does), and the round-0 "Pack drafts" loop copies NOMINAL rk
entries including stale tail slots — the round-loop top's write(st.extents, rk) should
cover extents, so the remaining asymmetry is the prefill-seeded MTP-KV/positions state
vs the shortened first verify. FIX CANDIDATE (NOT landed per coordinator): mirror the
steady-site behavior at the pre-loop exit (write extents + pack only n_draft), or
simply forbid round-0 shortening (break only from round 1 on — costs one round of
savings, kills the defect surface). Temp NOOP patch reverted; tree clean at f7d625bb.

## 13. c2r2 trace re-read (CORRECTION of §11's hang location): the stall is AFTER the
round-0 verify argmax, not inside the verify forward.
Full ordered trace: (1) TWO T=1 argmax pairs (steps 0,1) = round-0 chain step0 + one
more; (2) [conf rN] step 0 lines, prod=0.1062 (single observation — the d0-propose conf
block NEVER RUNS on the warmup's reuse path: the reuse branches propose d0 at L1534
without emit_conf/observe, so conf_break's first observe is chain step0; the "missing d0
print" is explained, no PCIe-lag mystery); (3) verify-depth round 0 rk=2; (4) a T=8
argmax pair = the round-0 verify argmax (T=8 = k_buf-ceiling buffer width, not round
width — it COMPLETED on both ranks). The hang is AFTER that: in the accept -> prep ->
ALIGN-FORWARD region of shortened round 0 (align = mtp_forward_decode_batch at T=rk+1=3
— the only AR-bearing forward between the verify argmax and the steady d0-propose,
which never appears). REVISED PRIME SUSPECT: the alignment forward at shortened width
versus the prefill-seeded MTP-KV/frontier state (the align consumes st.licensed_counts/
st.next_extents from mtp_prepare_next_round — check whether its AR call count is width-
dependent: a T=3 align on one side vs T=4-seeded state on the other would mispair the
one-shot ARs exactly here). Next step (CPU): read mtp_prepare_next_round + the align
path for width-dependent AR counts.

## 14. Post-§13 code-read (CPU): AR-count mispairing is WEAKENED — mtp_forward_tail's TP
branch issues exactly 2 one-shot ARs per call, width-independent (both T=3 and T=4 well
inside the 128-slot x 65536-elem one-shot capacity at hidden=5120). Remaining candidate:
round 0's ALIGNMENT inputs (alignment_ids/licensed_counts/...) are PREFILL-SEEDED
(prepare_mtp_prompt, nominal k+1=4 columns) on the reuse path — there is no prior
mtp_prepare_next_round at round 0. A shortened round-0 verify (T=3) feeds the accept/prep
arithometry against 4-column seeded state; if any index/extent derived from the seeded
columns drives a device-side loop bound or a slot the AR/attend reads, one rank can spin
while the other waits on it. DECISIVE NEXT TEST (short GPU): round-0 NOOP vs not, at
tau=0.7 (forces breaks in both configs), plus a variant that forces the REDO prefill path
(NINFER_MTP_NO_REUSE=1) with shortening ENABLED — if the hang disappears on the redo
path, the prefill-seed-width theory is confirmed and the fix is to re-seed or clamp the
alignment inputs when round 0 is shortened (or simply forbid round-0 shortening — one
line, costs one round of savings).
STATUS: W5 stays OFF. All evidence committed; branch tip clean.

## 15. §14 executed on the MERGED tree (slot 19:56-20:04Z, integration binary cbad568d) — theory NOT confirmed; a REAL accounting bug found instead

* nvfp4 post-merge ctest: 3/3 GREEN (attention 1.7s batch) on cbad568d.
* reuse control (tau .35): WARMUP-HANG reproduced on the merged tree (expected).
* redo (NINFER_MTP_NO_REUSE=1, tau .35 AND .7): LISTENING + 64-tok request, sha
  bcaf013545724cc0 == known-good reference at BOTH taus — but breaks=0 at tau=0.7
  DESPITE [conf r0] lines showing prod=0.1626 < 0.7. The redo "pass" is VACUOUS.

ROOT CAUSE of the vacuous pass (accounting bug, the real §14 finding):
mtp_confidence_break.h models d0 as FREE (observe() is for chain steps only;
drafts_produced() = 1 + steps_run). My wiring ALSO observes d0 -> steps_run is
inflated by 1 -> on the redo path (d0 observed) n_draft = 1+steps_run == rk whenever
the chain would shorten -> pending_depth never arms -> redo never shortens -> never
hangs. On the reuse path the L1534 d0 proposal bypasses the conf block (no observe)
-> accounting accidentally correct -> shortening arms -> hang. So the hang correlates
exactly with SHORTENING BEING ARMED, and the reuse/redo difference is which d0 site
fires — the §14 discrimination was confounded by the off-by-one.

CONSEQUENCES: (1) the steady-site breaks w5c7 recorded ("depth 3->2") shortened by ONE
LESS than intended (n_draft inflated) yet verified green + byte-identical — the stale
drafts1 slot was rejected, harmless but the savings are over-claimed by one step.
(2) Fix set for W5-ON: (a) correct the accounting (observe d0 WITHOUT incrementing
steps_run, or drafts_produced = steps_run when d0 was observed); (b) THEN re-run the
reuse-vs-redo matrix — the seed-width theory remains UNTESTED, not refuted.
WO stays open; W5 stays OFF.

## 17. ROOT CAUSE (matrix 2026-09-08 + instrumented dig, slot ~04:05-04:15Z): pending_depth rank-race, not state sizing

Matrix verdict (results/radiance_w5/ @ cf1e22ad): C(NO_R0) HUNG past round 0 — round-0-only
protection falsified; shortening broken in steady state. Instrumented dig ([w5d] markers,
conf_dbg-gated) on cell A (reuse+armed) caught the mechanism live:

    [w5d] round 19 armed pending=1        <- chain-end arm, rank 0
    [conf-break] round 19 depth 3->1
    [w5d] round 20 top rk=3 pending_was=3 <- NOTHING writes 3 to pending_depth

Mechanism: `pending_depth` (tp2_backend.cpp:1135) is a plain int captured BY REFERENCE by
BOTH rank threads (th1 @ :2492, sync_bar(2) @ :1100). The round-top consume pair
    rk_mtp = pending_depth > 0 ? pending_depth : nominal;   // :1933
    pending_depth = 0;                                       // :1935
is an unsynchronized read-then-zero between two threads. Interleaving R0-read(1), R1-zero,
R0-read(0)->nominal(3) — or the mirror — yields RANK-DIVERGENT verify widths (1 vs 3) with
identical extents/drafts expectations. The next TP collective (verify/accept allreduce) then
desyncs: permanent 100%-GPU stall, or garbage-dependent abort — exactly the W5 hang AND the
historic -9/-11 "crash signatures" class.

Why every prior observation fits:
- Hang ALWAYS immediately after a [conf-break] line (race needs an armed value; pending==0
  reads are benign — nominal is the correct value when nothing was armed).
- Hang round non-deterministic (A: 28 run-1, 1 run-2; C: 14 both runs) — pure thread timing.
- NO_R0 didn't help: same race at any steady break; round 0 was merely the first armed round
  (the WO §11-15 "round-0 seed state" localization was this race seen through round 0).
- B (NO_REUSE) hung too — race is in the conf/shortening path, reuse-independent.
- D clean — tau off => conf_gate false => never arms.

FIX DESIGN (NOT landed — dig-only grant; needs coordinator decision):
Two-barrier consume at round top, matching the file's sync idiom:
    sync_bar.arrive_and_wait();                    // both ranks at round top
    const int rk_mtp = pending_depth > 0 ? ... ;   // both READ (nobody zeroed yet)
    sync_bar.arrive_and_wait();                    // both have read
    pending_depth = 0;                             // now safe to consume
Safe because the arm precedes the iteration's last TP collective (rendezvous), so the value
is visible to both ranks before the first barrier; the second barrier orders the zero after
both reads. Cost: one extra 2-thread barrier per MTP round (~ns). Alternative: rank 0 decides
and broadcasts rk via an existing exchange. ALSO FLAG: audit the batched runner's
prepare_next_round site (:3853) and conf_break's shared observe() for the same pattern.

Dig artifacts: results/radiance_w5/w5dig_A.log ([w5d] trace), instrumentation @ (this commit),
binary rebuilt 04:05Z (serve target only; test-binary relink ate the disk to 78M — 9.3G
reclaimed per protocol, objects kept).

## 18. SECOND DEFECT (2026-09-08 evening session): tau-ON losslessness violation at long prompts

Post-rank-race-fix (6be15a2e) follow-up: the hang is FIXED (per-rank conf_break §17b —
validation: 5/5 green on the config that hung 100% pre-fix, deterministic across runs).
NEW finding: tau-ON vs tau-OFF on a 3,258-token prompt produces DIFFERENT emitted text
(deterministic both sides; divergence at char 233; lengths 722 vs 659). Losslessness
(doc 45 §6) is violated on long prompts. Tau-OFF unaffected.

Discriminator data: tau-ON deterministic-with-itself ⇒ NOT a race (the §17a/§17b shared-state
defects are closed); a tau-caused acceptance/state divergence ⇒ the §11-15 "shortened round
consumes mis-sized prefill-seeded state" class is REAL at deep positions: multi-chunk prefill
(GDN checkpoint @ chunk boundary, e.g. @9728 for a 10k prompt) + first armed rounds verify
against alignment/MTP-KV state positioned for the wrong boundary. The 2,000-token matrix never
reached a second chunk, which is why post-fix matrices were byte-identical.

Status: W5-ON flip remains BLOCKED (losslessness broken). Next dig: acceptance/frontier
arithmetic at shortened rounds following multi-chunk prefill — compare GDN checkpoint boundary
vs armed-round verify anchor positions, single-seq, i8 AND k4v4.

### §18.1 ISOLATION RESULT (traced runs, led_off/led_on/led_t001 in results/radiance_w5/)

| Config | Shortened rounds | Output vs tau-OFF |
|---|---|---|
| tau OFF | 0 | reference sha 1c00977cd794fa7b |
| tau=0.01 (gate machinery fully ON, 0 breaks) | 0 | IDENTICAL (1c00977cd794fa7b) |
| tau=0.7 (44×rk=1, 18×rk=2, 68×rk=3) | yes | DIVERGES (ed6a297758e23b39) |

MECHANISM (named): the width-2 shortened verify round ACCEPTS a draft token the width-4
verify rejects at the same stream position (first at round 17: accepted 11346, reference
11782) — kvarn staged-shadow window-slide × reduced-verify-width interaction. Gate machinery
itself is FREE (tau=0.01 byte-identical). Rank-consistent, deterministic (not a race).
Length-dependence: 2k-prompt decode never leaves the first 2048-token staged-shadow window
(2000+43<2048) — the matrix was blind; every ≥3.2k-prompt decode position sits past the slide
boundary and the first width-2 round past it diverges. Full-width rounds slide correctly
(tau-OFF green at 200k). Per-round accept traces: [w5d-acc] lines in the committed logs.
Fix design → A2 (ring/tile/staged-shadow owner). W5-ON blocked until width-2 verify is
correct at ≥ slide-boundary positions AND tau-ON==tau-OFF byte-identity holds at 200k.

## §18.3 OPT-1 IMPLEMENTATION MAP (handoff, 2026-09-08 — A1 context window exhausted mid-implementation)

Machinery located (all sites verified in tree):
- Rewind call site (single-seq decode round): `tp2_backend.cpp:2774` — `st.text->kvarn_rewind_mtp(next_F)`
  → `text_context.h:355` → `gqa_kvarn_rewind_to_token_count(*kvarn_mtp_ws_, keep_tokens, true)`
  → `kvarn_workspace.cpp:321` `rewind_layer`: branches `tile_page > page → discard_partial`
  vs `tile_page == page → rewind_tail` — THE non-canonical-geometry divergence point (A2 §8).
- Commit: `kvarn_workspace.cpp:429` `gqa_kvarn_commit_completed(workspace, scratch, cache,
  layer_index, stream)` — quantizes the open tile into committed pages.
  **KEY OPEN ITEM: it early-returns unless `tail_count == kG` (64)** — full-tile-only.
  OPT-1 needs the accepted-prefix tail committed BEFORE rewind; for partial tiles either
  (a) a commit-partial variant (quantize valid rows only + pad — needs A2's sign-off on
  scale-side-table semantics for padded rows), or (b) commit only when tail is full and
  keep the partial rewind (may not fully restore I2). DECIDE BEFORE CODING.
- Traces committed: [w5d-tgt] per-round target argmax (tp2_backend.cpp:2217), [w5d-acc]
  per-round accept stream (:2248), [w5d-brk] per-rank break decisions (:2455). All
  conf_dbg-gated (NINFER_MTP_CONF_DBG=1). Uncommitted in worktree: the [w5d-tgt] edit +
  this note.

Implementation order when resumed:
1. Commit the [w5d-tgt] trace edit (this worktree, uncommitted).
2. Resolve the partial-tile commit question with A2 (his pool).
3. Insert commit-before-rewind at :2774 (only under conf_gate? NO — commit for ALL
   rounds: I3 requires no gate-dependent bookkeeping. Commit is the canonical path.)
4. 6-gate battery (tools/bench/w5_gates.sh to be written): tau=0.01 non-reg,
   round-17 accept==11782, tau=0.80 byteid (width-3+2), tau=0.7 full-gen byteid,
   200k tau-ON int8 completes, matrix green.

## §18.4 S1 FIX DESIGN (A2 proposal, A1 answers + precedent — READY TO IMPLEMENT, fresh window required)

**S1 (primary): eliminate per-chain-step host syncs; round-end armed shortening, adaptive-style.**
- DELETE: both per-step `cudaStreamSynchronize(s)` + `should_break()` host checks (pre-loop chain ~1698 + round-loop chain ~2455) — the mid-chain host decisions are the reorder trigger.
- The chain then runs FULL nominal width, enqueues appends back-to-back exactly like the non-conf path → rewind sees identical fill level → I2/I3 hold trivially (A2's trivial-invariant argument).
- ADD: at the round-end accept sync (already required for accept_res_pinned D2H), read ALL chain conf values from the HOST-MAPPED one-shot slots (A2: slots[(step-1)%32]; steps ≤ k-1 ≤ 7 ≪ 32, no wraparound), reconstruct the prod prefix host-side, compute n_draft = 1 + longest prefix with prod ≥ tau, arm pending_depth = n_draft for the NEXT round.
- SAVINGS PROFILE: one-round-delayed shortening (round N full width, round N+1 armed short) — matches how the SHIPPED adaptive controller already saves. NOT a regression vs adaptive.
- F1-secondary (commit plumbing): the [w5d-tgt] evidence shows verify reads correct at reduced width — reduced-width verify + rewind is PROVEN by the shipped adaptive path; no commit-side change required. Prior F1 pack-fix is MOOT: d0/ar_drafts are VIEWS into drafts1 (tp2_backend.cpp:858-864) — the pre-loop "pack" is self-copies; no stale reads exist.
- DECISIVE PRECEDENT: adaptive_ctrl.n_cur ships rk=1/2/3 rounds losslessly today (T2-golden) — reduced-width verify + rewind is production-proven when arming follows the adaptive pattern (no mid-chain host interference). S1 mimics it exactly.

**Why the conf path diverged at all (answers the mystery):** per-step host syncs interleave the speculative AR appends with host logic — appends/rewind are ordering-sensitive; the different legal interleave leaves non-canonical (but legal) pool geometry → proposals shift → stream diverges. tau=0.01 clean (no conf path), 2k clean (timing lottery won), ≥3.2k hangs/diverges (lottery lost). All observations explained.

**Implementation notes for the next session:**
- one_shot conf slot accessor for round-end bulk read: see one_shot_argmax_conf(rank) impl (src/core/multi_gpu/one_shot_allreduce.cu) — slots are host-mapped; add `one_shot_argmax_conf_at(rank, step)` or read the mapped base directly.
- The [w5d-brk]/[w5d-acc]/[w5d-tgt] traces stay; add [w5d-arm2] logging the round-end computed n_draft.
- 6-gate bar per W5_PAUSED_STATE.md §RE-ARM + round-17 accept==11782 + tau=0.80 (width-3) byteid.
- Cards: verify free (guard ≤20 MiB per tightened fence), absolute binary paths, kill by pgrep-exact PID (setsid forks — $! group-kill unreliable, cost us a cascade tonight).

## §18.5 S1 ACCESSOR DESIGN + IMPLEMENTER NOTES (A2 WIP at stand-down — commit as-is)

**R1 arbitration (coordinator, seq 152/153): APPROVED.** Round-end arm reads
conf slots post-their-own-sync; R2 (reordering the d0 argmax into the chain slot
sequence) VETOED — frozen handshake, don't touch one_shot_argmax.cu's kernel body.

### The accessor
- conf values live in `OneShotAllreduce::Impl::slots[kNumSlots]` — `Slot {
  __nv_bfloat16* host_buf[2]; int* flag[2]; }` (one_shot_allreduce.cu:128-133),
  host-mapped (cudaHostAlloc :152-155). Slot index = `step % kNumSlots` where
  step = `impl_->rank_step[rank]++` (the GLOBAL argmax-call counter, :207).
- Today only `conf_from_payloads(rank)` (one_shot_argmax.cu:301, exposed via
  TpGroup::one_shot_argmax_conf, tp_group.cpp:307-309) reads the LAST-written
  slot. The bulk read needs `conf_from_payloads_at(rank, step)`: same
  field-extraction as conf_from_payloads applied to `slots[step % kNumSlots]`
  payloads for BOTH ranks (conf = 1/S with rank-0-first combine — mirror it
  exactly; do not re-derive the formula).
- **Round-relative step mapping (the open bit):** the per-round argmax-call
  count is NOT constant across shapes — a round emits d0-propose(1) +
  AR steps + verify-side argmaxes. The arm needs each chain step's global slot.
  Cleanest: capture `base = one_shot_rank_step_now(rank)` (add a counter getter
  to TpGroup — a plain host read of `impl_->rank_step[rank]`, no sync) at
  chain-start; the chain's AR confs then live at
  `base + 1 + step` for step ∈ [0, rk-1); d0's slot lives at the d0-propose
  call index (one before chain start) — READ IT AT THE D0 PROPOSE POINT (its
  own post-argmax value, via the existing conf accessor — one read, no sync
  beyond what the accept path already does), and carry it in a local for the
  round-end combine.
- **First-armed-round sentinel (coordinator caution):** before any chain has
  run (round 1 post-prefill) or after any rewind that resets the slot
  sequence, the previous boundary's confs DON'T EXIST. Define carry state
  explicitly: `int carry_d0 = -1` → "no-armed-info" → treat as full width
  (never arm a shortening from a stale slot). Battery gate 2 (round-17
  accept==11782) catches any violation.

### Deletion sites (verified, current tree @ 95846bb1)
- pre-loop Round-0 chain: tp2_backend.cpp ~2034-2045 (d0 `if (conf_gate) {
  cudaStreamSynchronize; observe_d0 }`) + ~2057 loop-top `should_break()`.
- round-loop chain: ~2846-2863 (d0 observe block) + ~2881 loop-top
  `should_break()`; the per-step observe at ~2905-2915 syncs + observes.
- keep `conf_gate`/`conf_break` structures (they hold tau); the round-end
  combine can reuse MtpConfidenceBreak as a plain host-side struct — its
  relaxed reads/writes are safe once it is PER-RANK (e278bd4f made conf_break
  instances rank-local? VERIFY: MtpConfidenceBreak conf_break is a local
  per-worker object at ~1463 — good, no shared state; the atomics were for
  the mapped-slot visibility).

### Battery (6 gates, from §18.4 + grants) — cards protocol
guard ≤20 MiB first; absolute binary paths (`/home/intel/ninfer/repo/build/apps/
ninfer-serve`); kill by pgrep-exact PID ONLY (setsid detaches — $! is wrong;
this cost a cascade tonight); tau cells: (1) tau=0.01 vs tau-OFF byteid (0
breaks, must stay clean); (2) 3258-tok k4v4 tau=0.7 → round-17 accept==11782
(was 11346); (3) tau=0.80 byteid (width-3+2 mixture); (4) tau=0.7 full-gen
byteid vs tau-OFF; (5) 200k int8 tau-ON completes (the hang cell); (6) W5
matrix A/B/C green (breaks fire, shas match baseline D). W5 default stays OFF
until 6/6. STOP-on-red. `[w5d-arm2]` log at the arm point.

Refs: spec §18.4 (95846bb1); design consult A2_PAUSED doc (wo/feature-matrix-
roadmap @ 5fd8fe8a, §8); traces results/radiance_w5/ + /tmp/led_*.log;
tau-OFF ref sha 1c00977c.

## §18.5a GATE-2 AMENDMENT (coordinator ruling 2026-09-09 04:03Z, agent2 S1 flag)

Gate 2 as written ("3258-tok k4v4 tau=0.7 → round-17 accept==11782") pinned an
exact round INDEX. The S1 bulk-arm consumes at iteration N+1's accept sync, so the
armed shortening first applies at top(N+2) — one round later than the old
mid-chain arming the projection assumed. AMENDED GATE 2:

> **acceptance VALUE 11782 observed within rounds 16–18. Value-match = GREEN
> (the correctness criterion is convergence to the tau-OFF reference stream);
> exact round-17 hit = bonus. If BOTH the value match (any round in 16–18) AND
> the ±1 round match fail → REAL RED: STOP-on-red as designed, escalate.**

Rationale: 5 of the 6 gates are lag-insensitive (all compare against tau-OFF
byte-identity or completion); only gate 2 encoded a timing assumption. A
bookkeeping lag shift must never trip the battery — a VALUE divergence would.

## §18.5b INVARIANT I4 — emitted == verified (named, binding for ALL depth-varying mechanisms)

**Statement:** a round's VERIFY width must equal the width its predecessor chain
EMITTED. Any mechanism that varies MTP depth must change EMISSION before
verify consumes; carrying "what was emitted" across the round boundary is the
only sound verify input. min(arm, nominal) is NOT sufficient — adaptive drift
between the two reads reopens a verify>emitted gap.

**Evidence (discriminant, results/w5_discriminant_20260909_073554):** fixed-k
vs shipped-adaptive byte-identical at 4 ctxs x 2 passes — the shipped
controller satisfies I4 by construction (step-6 updates n_cur BEFORE the
chain emits). S1's original round-top arm violated I4 (verified a prefix of
what was emitted) and reproduced G3's divergence EXACTLY (2k sha
1afe030d, deterministic) while adaptive-baseline stayed clean -> I4 breach is
the single cause of Problem 1 AND Problem 2 (mtp-pool appends at nominal
width, rewind at reduced = the committed=52/slot=62 holes the fail-closed
detector refused).

**Enforcement (22f3c8c5, wording corrected per A1 review):** arm clamps
emission post-step-6; the clamped width is PUBLISHED (pending_depth) and is
the next verify's ONLY input. **verify < emitted is structurally impossible**
(clamp branch: verify==emitted; skip branch — arm fires but pending_arm >= rk,
possible when adaptive rk < armed_conf_len — verify falls to nominal >=
emitted, which is the shipped adaptive-safe zero-on-read regime and SAFE).
What a mechanism must never reintroduce is verify BEHIND emission. Future depth mechanisms (k5v4 W5
variant, D1 fusion interplay, adaptive+v5): they satisfy I4 or they don't
ship. Battery run-2 = proof.

## §18.6 INVARIANT I5 — cross-rank arm symmetry (run-2 G2 death, binding)

**Statement:** under tau>0, the two rank threads' ARM/CLAMP decisions must be
identical BY CONSTRUCTION, not by value-agreement luck. The one-shot AR ring
pairs strictly by enqueue index; any decision that changes a rank's emission
COUNT must be made on rank-symmetric inputs under thread lockstep, else counts
skew -> pairing skew -> collective deadlock (run-2 G2: r0 armed, r1 never
evaluated, verify 2-vs-3, hang at prepare-verify, round 3).

**Why I4 alone was insufficient:** I4 couples emission==verified PER RANK. It
says nothing about the two ranks agreeing with EACH OTHER. The shared-by-
reference carry (armed_conf_*, old pending_arm) made single-shot consume a
first-writer-wins race; the post-propose reset made the staleness verdict
thread-timing-dependent. Both violated I5 pre-33ef6b3d.

**Enforcement (33ef6b3d, pending A1 re-review):** double-barrier consume
(snapshot between A/B, identical-value zeroing after B) + worker-local my_arm
(no cross-thread carrier at the clamp) + post-propose reset deleted (Round-0
carry refers to its own fresh slots 0.. via branch-entry resets). The
loop-top pending_depth pattern (W5 FIX, §17) is the same idea — future W5
state carriers follow it or stay per-thread; shared mutable arm state is
BANNED. tau=0 untouched.

**Battery status:** run-2 = G0 ok, G1 GREEN, G2 RED-BY-DEADLOCK (pre-fix
binary bd999b5a). run-3 gates on A1 review of 33ef6b3d + coordinator grant.
Evidence: results/w5_run2_halt_analysis.md + w5_battery_20260909_102659/.
