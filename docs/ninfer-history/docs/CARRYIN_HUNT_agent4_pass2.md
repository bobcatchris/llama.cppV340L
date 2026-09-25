# Carry-in hunt, agent4 pass 2 — suspect (a): the #784 16-byte publish shape (K-BLIND gap), adjudicated as FAR as statics go + the instrument I'd trust

Chair dispatch 9c2b177d assigned (a) to whoever converges first; this desk took it (reading
reported in-thread same hour). Predicate for re-derivation: `sed -n '22,60p;163,213p'
src/core/multi_gpu/one_shot_argmax.cu` at any tip quoted.

## What the code actually is (bytes @ f96b729a, verified, not inherited)
- HIP publish lane (:24-29): FOUR componentwise volatile u32 stores (`val, tok, sumexp, pad`),
  then `__threadfence_system()` (:177), THEN the flag epoch store (:178), then another fence
  (:179). CUDA lane (:40-48): one `st.global.wt.v4.u32` (16-byte store), same fence-flag-fence
  sandwich.
- Reader (:209-213): polls `peer_flag[t] >= expected_epoch`, (KAR trace prints the observed flag),
  then `ld_volatile_payload` — four volatile u32 loads on HIP (:31-38).

## The static verdict, with its honest boundary
The flag is write-ordered AFTER the payload by `__threadfence_system()` — classic message-passing:
any observer that sees the new epoch must see all four components, **IF the fence implementation
covers host-pinned (system-scope) writes correctly on gfx900/ROCm 6.2.** The per-component store
SHAPE is not itself a tearing hazard under a working fence; the fence is the load-bearing promise.
So the live question is exactly what the chair's K-blindness finding says: **the corpus cannot see
it.** A1TRACE-K (:205-209) compares flag epochs only; it cannot distinguish "all four words landed
whole before the flag" from "the fence did not flush the pinned window and a late-arriving component
was read as the PREVIOUS call's word". The second case is a real mechanism with the right shape:
previous-call = *same slot, different request's winner* (32-slot ring), it perturbs only the cross-rank
combine of ONE rank (the other rank's poll may land a full epoch later, clean), it forks EARLY tokens
(the observed char 37–107), it is stable-within-boot patterns varying between boots (link/pinned
timing), and it needs no sampler/split-k/AR-reject involvement — the ONLY suspect class that survives
agent3's §2 exoneration of the AR window for 4/5 forks, because the AR *decision* path was checked
(REJECTs) but the AR *payload integrity* was instrument-blind.

## The instrument I'd trust, named BEFORE any patch (agent3's rule, applied)
`ArgmaxPayload.pad` is a DEAD FIELD — every writer publishes `0.0f` (:175, `my_p{max_v, idx, sumexp,
0.0f}`). Turn it into a parity tag, env-gated (`NINFER_AR_PARITY=1`, unset = byte-identical publish
shape except one constant store — and the gate itself is the A/B arm):

    pad = __float_as_uint(epoch) ^ bits(val) ^ (uint)tok ^ bits(sumexp)   // writer, before stores
    reader: after ld, recompute from its own three words; mismatch => atomicOr(status, 2u)
            + printf [ARTAG] t=%d rank=%d epoch=%d (once per offending call)

- What it CAN see: any reader that combines words from different publish generations (tearing,
  fence failure, or store-reorder) flips the tag check — per-publish, per-request coverage, exactly
  where the corpus says to look. Cost: 4 XORs device-side.
- What it CANNOT see: a coherent-but-wrong payload (e.g. epoch race with an identical-tag stale
  write of the SAME call) — no realistic path: epoch is in the tag, so stale-epoch pairs
  cannot reproduce it. State-the-residual, no overclaim.
- Falsifier both directions (law 17): unit arm injects a shuffled 4-store order into the writer
  host-side simulation (it's a pure function of the four words) => tag MUST fail; untagged run on
  the 5-sample replay => if the tags ALL pass across 164 calls x 2 ranks, the fence promise is
  CERTIFIED-BY-MEASUREMENT on this machine and (a) closes with the strongest possible negative —
  the mechanism had a detector tuned to its own signature and stayed quiet.
- Comparability: the probe must run on a NEW bin (additive env-gate, unset = publish-path unchanged
  apart from one dead-field store — near-identical, but NOT the pinned era bin; per the bank law the
  row names its own bin sha and claims output-CLASS comparability to the 17g corpus only, exactly
  the honesty pattern the G-AMD-26 row set). This makes it a stamp event: written request when the
  reading pass says the tag design is final, NOT before.

## Cross-links
- Suspect (b) (warmup rank-step asymmetry) is agent3's consumer read — not double-booked here;
  note (b) and (a) are NOT exclusive: a fence blind-spot would produce exactly an asymmetry visible
  at the next consumer read, so (b)'s finding, whatever it is, is data FOR this arm's A/B.
- WO-TP4-E fusion: the pad-tag trick also pays the 4-card window a dividend — at world=4 the
  pairwise publish ring is the bring-up transport (map §2.3 RCCL-first is for AR, the argmax
  one-shot stays pairwise); the same tag certifies the transport BY CONSTRUCTION on the way to
  first-light. E-pass item, filed.

## Kit law-#5 gap, self-caught while reading the chair's ruling
Kit v6 header SAYS "RUN ONLY with a written coordinator stamp" — PROSE, not enforcement; the
standing law from agent3's VOID boot is a GRANT-ACK GATE. Cell script has one (exit 77); the kit
itself does not. Fixed forward in kit v7 (next commit): `STAMP_ACK` required env, refused with the
same permission-not-capacity wording as the cell. Nobody's ghost boots on my instrument.


---

## ADDENDUM ROW (chair e06c6738, annotate-never-delete): exoneration moved to 5/5 — (a) stands, strengthened

Agent3's self-correction (232d1797, merged 4f1f8bd7) relocated ALL THREE REJECTs to warmup:
every SERVED fork (5/5) is AR-window-exonerered, and their print-gate (step_gen<=16 ||
gen_at_flag<step_gen) makes silence at later steps full coverage, not absence of evidence.
The fence-blind-spot does not merely survive that shift — it is the ONLY AR-boundary class
left standing under it: payload INTEGRITY was never instrumented for ANY served fork.
The warmup anomaly agent3 holds as suspect (b) (rank1 at AR step 100/101 with
peer_gen=99/peer_flag=0 — 'code says it cannot pass silently, yet none appear') sits in the
SAME blind spot: a cross-generation read is exactly a datum no epoch-comparing instrument
can see. The PARITY arm therefore instruments the (b) window as a free side-effect:

**OFF/ON capture plan (answers the chair's stamp-request question NOW): the parity boot runs
NINFER_AR_PARITY=1 + NINFER_MB_ARGMAX_TRACE=1 from SPAWN, covering warmup and all requests —
agent3's (b) gets its tag census at zero extra card time; a warmup-only [ARTAG] hit is their
datum, a served-step hit is (a)'s, zero hits anywhere closes both blind-spot classes
certified-by-measurement on this machine.**

Implementation landed source-only at the next commit: tag = sampling_tag_mask(epoch ^ bits(val)
^ tok ^ bits(sumexp)) on the dead pad field; mask forces a normal positive float exponent —
a raw tag could be an sNaN pattern and FP moves may legally QUIET NaNs, fabricating mismatches
on intact publishes; masked, false positives are impossible by construction and a missed
cross-generation mix costs 2^-23 per publish (named residual, beats silence). Mismatch fires
[ARTAG] UNCONDITIONALLY + status bit 2 + the entry throw distinguishes parity from timeout.
UNSET env = pad stores 0.0f as every historical writer did, one dead-if branch: default path
byte-identical (step-0 reviewer note: one new kernel arg + one read-once env gate).

## ADDENDUM 2 (self, d8710cc6): the falsifier caught the design's blind spot before any boot
The first mask (keep low 23 XOR bits) was BLIND to single-bit tears landing on bits 23-31 of any
field — i.e. EXPONENT/SIGN bits of val/sumexp, precisely what corrupts a float payload usefully.
Planted-flip leg walked 27/96 through. Fixed by the GF(2)-triangular fold (h = r ^ r>>15 ^ r>>23
before masking): single-bit deltas in any position now caught EXACTLY, multi-bit mixes miss at the
named 2^-23, zero-false-positive sweep still clean (2,007,098 cases incl. every exotic float
pattern). Headline law re-earned: **an instrument's falsifier must be hostile enough to kill the
instrument — a check designed to pass is a receipt pre-printed.** The (a)-arm design section above
stands as written (annotate-never-delete); its final shape is d8710cc6's.
