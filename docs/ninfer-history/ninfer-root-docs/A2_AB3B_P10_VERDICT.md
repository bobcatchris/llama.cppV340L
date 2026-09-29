# ab3b cell-T — P10 datum + the determinism CORRECTION: identical bytes, different outcome = real race; tie-break static REFUTED; stale-pairing mode documented

Era: same BIN e24951a1800b as m6_boot1, tree 0826dfaa, GO = CORD seq-51 (cell-B
decision), p1ab3.sh dual-cell (cell-B NEVER reached: cell T froze; FREEZE-SOP bt
captured 06:18:20 before kill; release row: own PIDs, both cards 15 MiB).

## THE DECISIVE COMPARISON (the boot's real output)
m6_boot1 (COMPLETED, acc .369) vs ab3b cell-T (FROZE at step 13) are
byte-identical through every comparable line: steps 12/13 D2SS traces (vids, vpos,
slt, win), round-12 drafts on BOTH ranks, SYNC counts (15/15/16 vs 15/15/16), same
warmup stats row (rounds=3 accepted=1 at .067/1.33/4 — deterministic warmup in both).
Same BIN, same body, same request bytes.
=> **CORD seq-52 confirmation (a) 'DETERMINISM PROVEN' IS REFUTED BY m6 vs ab3b**:
   the exact-step-13 repeat across ab3/ab3b was real, but the COMPLETION of the same
   content in m6 proves the outcome is a per-boot coin-flip, not a content function.
   Race class RE-OPENED (and with it: my C4-fix 'verified by completion' claim needs
   this doc as its correction — m6 completed, but the wall is not dead; the reset did
   not fix a race it cannot reach).
   The 4/4 script correlation (hang-on-p1ab3 / complete-on-p1bt) was checked: server
   argv + env blocks are IDENTICAL between the scripts (diff proven); no mechanistic
   difference found -> script identity is an observation proxy (timing perturbation
   from the poll/watch structure), NOT a cause.

## (b) C4-as-CAUSE: REFUTED (CORD-concurs) — hunk KEPT
Reset present, placement A1-GREEN, wall unmoved at ab3b. Hygiene parity with every
other route retained per seq-45(3) logic. C4-as-seeding-source also dies with (a):
the intra-request pairing state IS the race candidate now (see below).

## (3) TIE-BREAK STATIC — REFUTED as the divergence vector, with the replacement named
allreduce_argmax combine (one_shot_argmax.cu:163): tie resolves to MIN TOKEN ID on
both ranks symmetrically (`peer.val > my.val || (equal && peer.tok < my.tok)`) —
rank-flip impossible for equal payloads. The float payloads (:175-178 rank-paired
m/s extraction) read the SAME (m0,s0,m1,s1) on both ranks. So per-rank lic divergence
cannot come from the COMBINE. It CAN come from the documented PAIRING failure,
:263-266 comment verbatim: 'the fused argmax is a lock-step handshake keyed on a
per-rank [step]... pairing breaks and a rank reads a stale (or never-written) peer
payload as a token id' — plus the DEVICE-SIDE peer poll :155
`while (peer_flag[t] < expected_epoch)` — the ONLY mechanism found that explains BOTH:
(i) rank-divergent tokens from identical content (stale/never-written payload =
timing), (ii) GPU0-at-100% during freezes (that poll loop IS the device spin — the
ab3-era captures showed exactly that), and (iii) the clean rank-1 exit BEFORE rank-0
reaches the next argmax (rank-1 exited; rank-0's verify-13 forward died in a D2H
queued behind... — refinement: rank-0 bt frame-class here is D2H-BLOCK-IN-APPEND
(cuMemcpyDtoHAsync under gqa_kv_append_kvarn_and_commit), i.e. rank-0 died one step
EARLIER than the argmax-poll theory predicts: the append D2H waits on stream-order
behind an EARLIER one-shot kernel of round 12's chain path OR the epoch flag written
by... — unresolved at this layer; naming stops at 'pairing/epoch race on the fused
one-shot handshake is the live mechanism-class; exact desync origin NOT statically
identified by this pass'.

## Verdict status
- m6_boot1 P8 STANDS as banked history (first complete twin request, .369/37.4) —
  and now understood as the WINNING SIDE of the coin, not a proof of wall-death.
- G1 status: decode path PROVEN at parity WHEN THE COIN LANDS RIGHT; the wall is a
  7th-surface-class TIMING RACE in the fused one-shot pairing — convene-with-user
  threshold reached per CORD seq-52 ('no third boot on the same wall without his
  eyes'). No boot 7 moved or proposed by A2.
- Cell-B: never ran; join still pending-not-lost (third time — it must NOT run on a
  coin-flip server; a clean cell-B needs the race resolved or cell-T re-rolled to a
  win).

## CORD seq-53 STATIC DELIVERABLE — three questions answered + the precedent comment + the one-line defect
Q3 FIRST (cleanest): twin per-round T is CONSTANT: both seed (:3612) and rounds (:3794)
pass d2_log with ne[1]=T=k+1=6; wrapper derives T from logits.ne[1] identically both
ranks. NO T-variance skew. (a>0 changes emitted COUNT, never the argmax call shape.)
Q1 (call-count parity): twin consumes one_shot_argmax at EXACTLY two sites; the chain
uses plain NCCL allgather only (verified: no group() calls in run_dflash2_chain). Per
request: 1 seed + 1/round, both ranks, reached identically until the anomaly itself.
Reset now guarantees cross-request symmetry. No host branch in the twin loop changes
collective count (cancellation/token_cb/stats are rank-0-gated NON-collectives).
Q2 (forward-only guard): **NONE — AND THAT IS THE STRUCTURAL DEFECT.**
`expected_epoch = step/32 + 1` (kNumSlots=32, one_shot_argmax.h:14). For the FIRST 32
CALLS the epoch is the CONSTANT 1: my_flag[t]=1 every call, the poll
`while (peer_flag[t] < expected_epoch)` passes on ANY payload the peer ever published
to that slot, and payload slots are overwritten each call with NO generation stamp
(ArgmaxPayload 4th field = 0.0f unused). Within-epoch call pairing is therefore PURE
ORDERING — a rank whose block-t reads late can consume the peer's NEXT-step payload
(the peer's N+1 kernel overwriting payload_B[t] with N+1 values while its flag stays
1 passes A's check trivially). Lockstep + round_end_bar make this window narrow, but
the bar is NOT a full device barrier (host-side arrive; the peer's NEXT argmax can be
ENQUEUED and its blocks scheduled while this rank's lagging block-t of step N still
polls-then-reads). THE PRECEDENT: tp2_backend.cpp:1629-1640 (W5 FIX §17b, 2026-09-08)
documents EXACTLY this terminal signature — 'rank-divergent chain lengths ->
ALLREDUCE_ARGMAX MISMATCH -> PERMANENT STALL, GPU0 SPIN 100%/GPU1 0%' — born from
rank-DIVERGENT CALL COUNTS; the twin has equal counts but the SAME unguarded
future-read window under equal counts + scheduling jitter. Datum-fit: content-identical
m6 vs ab3b with opposite outcomes (TIMING RACE ✓); rank-1 lic gets a stop-adjacent
future-token at round 12 -> hit_stop rank-1-only -> silent clean exit ✓; rank-0 then
stalls in verify-13 at a D2H ordered behind stream work whose peer participant is gone
(GPU0 100% device-side poll ✓, rank-1 absent from LWP dump ✓).
FIX-SHAPES (propose, none landed): (S1) one-line: monotonic step-based epoch
(`expected_epoch = step + 1`, flags then carry step identity; reset keeps per-request
steps in range) — kernel + both flag writes; costs nothing, kills future-reads
completely. (S2) generation field in payload + verify-after-read (heavier, exact).
(S3) DIAGNOSTIC-FIRST, ZERO CODE: the kernel ALREADY has A1TRACE
(NINFER_MB_ARGMAX_TRACE, :268-273: per-rank step/slot/epoch/T prints) — a boot-7 with
that env on, ROLLED N TIMES (the race needs repetition; N runs of p1bt.sh, cheap ~90 s
each, hang-detection automated), would name the divergent step AND count hangs vs the
script-perturbation correlation. RECOMMENDED: S3 boot battery (CPU-scripted, no src)
in PARALLEL with S1 implementation — S1 is cheap enough that verification can be the
same battery: fix-BIN N-runs should show ZERO hangs if S1 theory holds.
NOTE: this WEAKENS but does not restore C4-as-fix: the reset (fcd045ce) fixed
cross-request skew; the INTRA-request window stands with or without it. m6_boot1's P8
remains banked history (the winning coin), G1-CLOSED DECLARATION IS WITHHELD pending
the race: parity .369 is REAL as a ceiling datum, the route is not yet RELIABLE.
