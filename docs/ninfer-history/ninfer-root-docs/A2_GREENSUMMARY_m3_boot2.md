# m3_boot2 — WITH-ERROR CLOSE (CORD seq-24 label, not GREEN): decode loop green end-to-end, COMPLETION path is the named wall; seq-16 stop-wall CONFIRMED

Era bundle (rule 6, ALL self-printed by the wrapper at claim): tree `7a1b2ea1` (src
`fdd2327c`, script-only commits after), BIN md5 **e74560a03a65cdd176db5c29df0b9566**
= CORD GO hash, verified pre-launch by coord AND wrapper; body m2b_10k_a (card-derived);
slt=[0 2]; fresh guard at claim in card line 6 (no foreign servers, port free, GPU 0 apps,
df 23 G); GO = CORD seq-20 written stamp; A1 cumulative static GREEN (drafts/a1_cumulative_static_go_gate.md).

## Result in one paragraph
No freeze, no divergence mid-run: 31 decode steps, both ranks posted every SYNC print in
perfect 70/70 parity INCLUDING `step=31 entering round_end_bar`; then the completion path
threw `Paged KV allocation is not bound` on BOTH ranks (paged_kv_cache.cpp:459/481,
bound_row_=-1) followed by `[rank 0] kvarn reset inflight (re-prefill from 0)` and the
request returning the error JSON (curl rc=0, wall 20.0 s). Warmup bled the same error at
its ~4-step boundary (era-consistent, lines 893-896). Teardown clean by the wrapper itself.

## THE UNIFICATION (the run's real gift)
`:3868 acc_rounds += 1; acc_accepted += a;` executes on BOTH rank threads (worker lambda
captured by reference — the gen_before family of 1653619e, stats side missed). Observed:
rounds=59 at step 31 (2x31 minus race-bound ~1), warmup rounds=8 at ~4 steps. Retro:
p1r1's 'died at 58 rounds' = ~29 steps; the 58 x 17.5 MiB = 1 GiB DRAIN ARITHMETIC was
built on a phantom denominator; every era's wall (p1r1 round-58 throw, m3_boot1 step-29/30
hang, m3_boot2 step-31 throw) sits at gen ~= 30 = the NATURAL STOP boundary of this body
(greedy, temperature 0, zero cap fields). One wall, three costumes. The arena commit
(67b18499) cured a disease that measurement now says never existed — NOT proposed for
revert (reviewed, pushed, harmless hygiene; A1/CORD weigh), but the '58x17.5=1GiB' story
must be marked SUPERSEDED wherever it lives (COORDINATOR.md 22:4x/d709563c arena-finding,
09-11 00:0x night-bank) so farm7 sizing arguments never cite it.
Boot1-vs-boot2 difference (hang vs throw at the same wall) = timing race on the teardown
path (who reaches the missing-binding commit first; boot1 = rank-1 silent-exit variant,
boot2 = both-rank named-throw variant) — consistent, both are branch 5d's
'at-the-stop-token' wall, not the 'never-enters' wall (rank-1 posted through the last bar).

## Quality datum (pre-registration (ii), clean)
accepted = 0 over all 31 steps (acc_rate 0.000 end-to-end; warmup stats irrelevant post
counter-understanding). The drafter proposes plausible-but-never-matching at N=1 from
round 1 — unchanged by everything shipped tonight. NEXT IN SEQUENCE (A1's declared hole):
verify .cu GDN write-target semantics read (ring-vs-linear).

## Readings scorecard (card 5a-5d)
- exact-stop-repeat (seq-16/19): CONFIRMED — step-31 ~= same-boundary as boot1's 29-30
  (2-step wobble = where the rate story says stop sits; within the deterministic band).
(CARDS RELEASED by CORD seq-24.) - run-2: SKIPPED per seq-16 corollary (exact-repeat happened; CORD to release the slot).
- divergent-round: not applicable. green-loop: PARTIAL — the DECODE LOOP is green
  end-to-end for the first time; the COMPLETION path is the named wall.
- capture-names-throw: rank-1 threw on the completion path with immediate print compiled
  in, but the post-loop error path in this twin run used the worker-error line
  ('[tp2 worker error rank 1]') before join — no bare silent exit occurred, so the
  immediate print did not need to fire. It stays armed for the fix-cycle boots.

## Files
serve.log (8416 lines, complete), card.txt (era+readings self-printed), wrapper log,
kill_list.txt (PIDs reaped by wrapper), respA.json (the error JSON = the datum), no bt
(wrapper never armed — nothing froze).

## POST-CLOSE MECHANISM UPDATE (same session, supersedes the teardown-boundary framing above)
The 'completion path' wall is an ENGINE ROUTING FALL-THROUGH, byte-verified:
tp_engine.cpp if(dflash2) block (:348-359, route + done=true) has NO `return`;
the pre-existing unconditional run_tp2_request at :361+ then re-runs the SAME
request through the batched/MTP-shaped runner (the §22.6 class the replaced
throw used to guard — the comment says 'INSTEAD of ... below'; control flow
disagrees). That runner's start block (kvarn reset :1872 -> publish_mapping
:1880) hits bound_row_<0 on state the twin left -> both-rank '[tp2 worker
error]' (:3234, sole producer) -> '[req 1] error' replaces the client's 200.
Consequences: (a) the twin itself ran CLEAN to natural stop BOTH eras —
boot1's hang and boot2's throw are the fall-through's two collective-timing
costumes, not decode-loop walls; (b) p1r1's round-58 'drain' was the same
fall-through after a ~29-step clean twin run (x2 phantom denominator,
dc92bddf); (c) warmup bleed = same fall-through on the warmup request.
FIX (proposed, awaiting A1 static + CORD stamp): `return;` after done=true in
the dflash2 block. Predictions for the fix-boot: P1 warmup not-bound vanishes;
P2 HTTP 200 + twin text; P3 rounds == steps exactly (counter fix riding).
Ledger note (A2 self-report): an intermediate message RETRACTED this finding
citing an unverified 'git -L' — itself wrong; re-retracted against direct
bytes same session. Claim discipline: read-the-file beats remembered-layout,
especially when the audience is the static gate.
