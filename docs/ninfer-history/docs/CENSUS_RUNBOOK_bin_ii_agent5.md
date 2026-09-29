# CENSUS RUNBOOK — bin (ii) ×25 (agent5, chair order 21:0xZ item 2)

**Purpose:** the ×25 same-request census on the post-(i) bin, with the RED baseline pre-named, so no
lane improvises at the boot seat. Zero design choices made here that aren't already banked — every
row cites its source sha/site. Pair with agent1's verdict script (four states) and agent3's
interpretation seat per plan :37.

**RED baseline (pre-banked, the author's-own-adoption-boot irony included):**
`G17x_serve.log:755-756` @ `9dd78c30…` (banked shared checkout `results/amd/p3/`):
`[A1TRACE-KAR] rank=0 step=126 … verdict=REJECT-candidate` + same `step=127` — two warmup expiries,
skew +1/+2 at a run boundary (agent4's re-read, adopted by agent3: boundary-clock lag, NOT the epoch
race — and lag SURVIVES per-request resets, which is why (i) does not close them). Poison predicate
armed: these two lines are the expected-residual witnesses in the post-(i) bin's warmup window.
Fork-rate RED (the carrier hunt's own baseline): G17x text-sha 3-identical/2-distinct
(`G17x_repeat_summary.txt`, `82b9d60f` row).

## 1. Pre-boot (every time, in order)

1. `python3 results/amd/p3/residence_probe.py` — RESIDENCE-BEFORE-SPAWN clause (agent2 `1ff254b3`).
   Cold road ⇒ run the one-line re-warm (`dd if=/media/chris/EMTEC256/qwen3_8_27b_q3.ninfer
   of=/dev/null bs=8M`, ~1 s if only a slice evicted) BEFORE spawn.
2. Written GPU grant from chair; GRANT-ACK echoed in the boot's own arguments (exit-7/77 kit v7 law).
3. Bin from `/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin` ONLY (the filename is the stamp);
   manifest row records tree sha + bin sha256 + bank path (G17x_manifest_row.txt is the template).
4. Mask: `HIP_VISIBLE_DEVICES=2,3` (dev0 = display, untouchable); KFD precheck zero foreign contexts
   (`gpu_guard.sh` pattern, own-pid kills only).

## 2. Boot argv (identical to the proven census-capable shape — G17x manifest row, verbatim flags)

    HIP_VISIBLE_DEVICES=2,3 NINFER_WORKSPACE_MIB=96 \
      NINFER_MB_ARGMAX_TRACE=1 NINFER_AR_PARITY=1 NINFER_MC31=1 \
      setsid nohup /home/chris/artifacts_bin/ninfer-serve_<post-(i)-sha16>.bin \
      "$(cat results/amd/p3/Q3_CANONICAL_PATH)" --port 8093 --devices 0,1 \
      --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 --no-cuda-graph --greedy \
      > results/amd/p3/G19census_serve.log 2>&1 < /dev/null &

- Capacity: AUTO, no capacity flags (G17x law: "capacity is NOT this cell's truth").
- Env note: `--devices 0,1` stay literal-under-mask (G18 runbook masking law).
- Arms at post-(i) tip already print their own arming asserts (`[MC31] armed gate=1 kNumSlots=32`
  once-per-process; `[ARPARITY] armed:…`; MC31-H per-call HOST lines `rank= step= slot= epoch=` —
  the wedge lesson made them host-channel, zero device-printf timing pressure).

## 3. Workload: the ×25

One server, ONE greedy request shape, ×25 sequential: prompt `'Hi.'`, `temperature 0`,
`max_tokens=32` (G17x's exact corpus shape scaled 5→25). Same curl per rep; each response banked
`G19census_resp_<n>.json`; per-rep text_sha16 line appended to a running summary (the valid
instrument is TEXT-sha; `raw_sha16` provenance-only per agent4's manifest law). Between reps:
none — sequential, no restart, no cache clear (the census measures in-process per-request state;
a restart between reps is a different experiment and voids the point).

## 4. Expected greps + reading table (the pre-declared branch tree)

Mechanical greps on `G19census_serve.log` after the run (all counted, none interpreted at the seat —
interpretation is agent3's, per plan :37):

| # | grep | expected | branch if seen |
|---|---|---|---|
| G1 | `^\[MC31\] armed` / `^\[ARPARITY\] armed` | ≥1 each | 0 = INSTRUMENT-ERROR (unarmed boot: counts from this log are NEVER-CHECKED, re-run) |
| G2 | `^\[MC31-H\] rank=` | both ranks, per served call; step/slot/epoch visible | count mismatch vs G6 expected = INSTRUMENT-ERROR leg |
| G3 | `verdict=REJECT-candidate` | ZERO in served window; warmup-boundary expiries (≤ a few, skew ≤2, steps 125-127 region) = the KNOWN RESIDUAL (agent3's signature: expiries ONLY at run-transition steps with skew ≤2 — distinguishable by grep from a carrier event: those cluster at a fork site, not a boundary) | served-window REJECT = carrier witness, cite step/slot/epoch + nearest fork ordinal |
| G4 | `^\[ARTAG\]` | ZERO | any HIT = C6-window convict (poison predicate format: report as TRACKED-RED with tags; per interim rule the run is QUARANTINED not stopped — tokens not quotable, `b28d25fd`) |
| G5 | `^\[AR-FAILOUT\]` + `tp2 worker error` | 0 / 0 | FAILOUT = symmetric death fired = FAILLOUT state (agent1's script), report last MC31-H line as boundary witness; worker-error>0 = the replaced throw class resurrected = INSTRUMENT-ERROR on (i)'s consumer leg |
| G6 | per-request AR call counts: `rank=` lines with slot arithmetic OR derive `25 × tokens × 128 ≈ steps-per-rank` from MC31-H last step per rank per request | EQUAL both ranks, every request (pairing-law cross-check: rank_step parity at each reset boundary) | rank-asymmetric step count at any boundary = the call-count-parity suspect (`tp2_backend.cpp:3729-3740` named it) — that IS a census outcome, branch (3)-equivalent. **TWO WORLDS, distinct fixes (agent3 21:1xZ, adopted): (a) LONE-rank step-N MC31-H line = crossing-strand (host-entry asymmetry at the collective); (b) rank-DIVERGENT req-ordinal (rb-n join, §6 addendum) at MATCHED step = re-arm desync (both ranks reached the same ordinal through DIFFERENT request boundaries — the A2Q3 freeze shape — NOTE (b) is ALREADY detectable at the census bin without the format change: the rb join is per-rank, so rank0 req=3 vs rank1 req=2 at overlapping steps is visible in the shipped log format). Read (a) and (b) as separate rows; a bin where (b) fires but (a) never does indicts the reset path, not the ring** |
| G7 | `\[A1TRACE-K\] .* observed=` | observed == epoch (pairing exact) | observed > epoch = wrap-window live (the named blindness witness from §5-4's instrument-gap amendment — its first capable firing, report either way) |

**Fork table (the census's namesake output):** text_sha16 across the 25 reps →
`n_distinct ∈ {1, 2..25}`. Pre-declared readings: **1/25-distinct (=G-CELL green shape)** =
carrier absent at this bin/corpus (does NOT close item 7 — shelf (c) exoneration per §7 bar in
AR_PARITY_ARM_design §7: names-nothing-closes-nothing without the triple); fork-rate >0 = carrier
census fires, per-fork attribution = first diverging token per rep pair + G3/G4/G6 rows at that
ordinal decide the worlds (three-worlds table, agent3):
margins≈0+tie evidence ⇒ decision-rule; identical-logits-different-winner ⇒ branch (2) ring
(ours-alone, cite QUICKDIFF pair); values-differ ⇒ upstream state / AR-carrier (branch 1/3,
per-request counters + C5/C6 classing, AR-counter arming is minimum boot).

**×25 statistics law:** fork-rate is reported as k/25 with the bin sha; a 1/25 event is data not
noise — the RED baselines were 5/5-distinct and 3-of-5, so any nonzero k is a live carrier census,
and 0/25 with all greps in their expected column is the strongest exoneration this corpus can
offer — still shelf (c), never shelf (a), per the closure bar.

## 5. Post-run

Release: kill own pids only, KFD recheck, verdict row file `G19census_verdict_row.txt` (agent1's
script's four-state output + sha of this runbook + sha of bin + fork table + the G1-G7 counts).
If the boot wedges pre-serving: pre-declared branch = bin-level failure, NOT a census outcome
(agent4's wedge-row law, `0ec28ea1`), one retry line via chair, log the branch signature
(gating-line count vs zero A1TRACE/MC31 = never-reached witness).

## 6. The one-line addition from my sites-census (chair item 1, this cycle)

Bin (i)'s sites list is **complete as designed: the epoch-indirection class is AR-only.**
Predicate walked: `grep dev_epoch|host_epoch src/` = 20 hits, ALL in
`one_shot_allreduce.cu` (post-(i)-tip incl. rank_gen block); `one_shot_argmax.cu` = **zero** —
its epoch is already host-arg `step+1` (`:470@origin tip`; `advance_epoch` is a documented no-op
`:409`), and its `reset_step` zeroes rank_step AND flags with the same bracket shape
(`:413-422@origin tip`).
No third sibling: `tp_kernel.cu`, `multi_gpu_hip_addendum.h`, `tp2_request.cpp` (pinned words are
write/read-back result slots, no poll-compare), `host_kv_arena.cpp` — zero epoch constructs.
**So the closeout sentence stands with one precision: argmax already has the HOST-ARG-FROM STEP
shape (i) gives AR — argmax got the indirection-deletion early (S1 fix, `step+1`); what argmax does
NOT have from (i) is the monotonic-stamp epoch domain: its epoch = step+1 RESTARTS per request
(reset zeroes rank_step at `:415@origin tip`), exactly the pre-(i) AR disease in a smaller ring — 32 slots ⇒
epoch restarts at 1 each request with flags zeroed same reset (the bracket protects the cross-request
leg ONLY while the sync_bar pairing holds; the strand world agent3 named: a lagging peer re-publishes
flag=1 after the partner's reset = new-request slot-0 pass on a stale flag = coherent-stale read of
the PREVIOUS request's slot-0 payload).** That is not (i)'s miss (the chair's handoff cited allreduce
sites because allreduce is where the indirection class lives) — it is the pairing-law follow-on:
`expected_flag = gen+1 across-boot` closes the argmax cross-request leg by the same one variable, and
if (i) lands flag-carries-step for AR only, argmax keeps the reset-bracket dependency as its known
residual — one named row for (iii)'s candidate list, zero new evidence needed to arm it
(MC31-H already prints slot/epoch per call at :516 — the stale-flag pass would show as observed>epoch
via the G7 grep above). **ADDENDUM 21:1xZ (my own re-read at tip): the req-index need NOT wait on a
format change — the rb arm already prints the ordinal:** `[MC31-H] rb rank=%d n=%u site=short/long-
reprefill` fires per-rank, per-reset-entry (:1915/:2164 @tip), thread_local counter monotone across
boot. Mechanical join: a per-rank single awk pass keys each non-rb MC31-H step line by the count of
prior same-rank rb lines = request ordinal; **rb-n vs step monotonicity is additionally the G6
parity witness in the same pass — rb counts diverging across ranks while step counts stay equal (or
vice versa) IS the strand-world fingerprint, derived from two channels already shipping.**
  **→ WORLD-(b) CLAIM RETRACTED AS STATED, by its author, 23:4xZ (agent4's retraction upheld by
  agent3's `c2145b99` §2, and the mechanism is decisive: `rank_step`/`rb n=` are `thread_local`,
  rank1 runs one fresh `std::thread` per request against a pool of 1+16+1 = 18 ⇒ rank0's `n=2`
  CANNOT appear before arrival #19, and it is observed at #19/#19/#20 while rank1 prints `n>1`
  ZERO times across five banked rb logs. So an rb-n cross-rank compare is equal-by-construction
  in one direction and pinned-to-1 in the other: it has no discriminating power, and my
  'strand-world fingerprint' sentence overstated it.** The tool survives the retraction — my
  shipped `G6b` derives the ordinal from the per-rank rb COUNT and compares per-ordinal step
  counts/max-steps, which is a different predicate from `rb n=` and keeps its power — but at
  world=4 neither the n-field nor a pairwise complement exists at all, so from `964c44af`→F-C the
  legs report PRESENCE SETS over the rank universe the log actually carries and print that
  universe; see `docs/amd/CLOSURE_BAR_AUDIT2_agent5_2026-09-13.md` row F-C for the world-blindness
  bug this sentence hid and its triple. Field-in-the-line remains cleaner for post-census logs
  (agent4's custody), and (ii) is not gated on it. Field-in-
the-line remains cleaner for post-census logs (agent4's custody), but (ii) is NOT gated on it.
