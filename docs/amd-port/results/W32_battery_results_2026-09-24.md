# W32: POST-U1 LEVER BATTERY RESULTS (E-139 FIVE-WINDOW QUEUE)

Date: 2026-09-24. Desk: BATTERY-ANALYST (zero-GPU; the battery unit
campaign-postu1-battery owns the machine - read-only on /home/chris/postu1_*).
Branch: amd/v340-port-v2. Ledger entry: E-145.

## SCOPE

The post-U1 battery (E-139 queue + W28 addition) run as ONE ordered unit,
/home/chris/run_post_u1_battery.sh, five paired served windows, all 10k
decode-only R/A/A/R cells at ctx 10240 on the TP4 stack of record:

  1. w23env  - MMVQ env gates: IQ4XS_SHARE=1 + IQ3XXS_S2R=1 + IQ3S_S2R=1
               (W23 C1+C2; expected ~-3.4 ms/round pool, ~+2.6% decode)
  2. ub1024  - --batch-size 1024 --ubatch-size 1024 re-promotion (E-139a)
  3. w19     - LLAMA_DRAFT_SHAPE_CACHE=1 on the E-138-fixed tree
  4. w22c1   - per-die clock floors (root set_clock_floor.sh), R F F R
               + one 200k F/R anti-decay pair
  5. w28nmax - --spec-draft-n-max 4 (W28 chain-shape window)

Per window: pairs = (arm1-ctrl1, arm2-ctrl2), position-1 law (control
bookends), 180 s inter-cell settle + die_idle + die_hot<60 re-gates,
void_gate.py --die3-group 2 on every cell's thermal sideband, machine
adjudication appended to /home/chris/postu1_window_results.txt.

## RUN IDENTITY

- Unit active 16:25:47 CDT 2026-09-24 (clean reboot 16:22-16:25 after the
  pre-reboot battery instance was stopped while queued behind an
  svm-watchdog lock; pre-reboot partial cells discarded, NOT results).
  BATTERY COMPLETE 18:02:16 CDT ("POSTU1-BATTERY-DONE", unit inactive,
  wall 1h36m29s; machine summary 0/5 promotable - see desk corrections).
- Boot arm-identity stamp 1560e872b (rev5); HEAD moved to 260425807 (rev6,
  PLAN_CURRENT docs-only) at 16:26:52, one minute after boot - docs-only
  drift, accepted precedent. Binary 0d5eb814b0653589 identical to the
  pre-reboot launch (rev4 rebuild).
- Per-cell PROVENANCE lines: all checked below; any tree_clean=False or
  code-commit drift is an anomaly (none so far).

## RESULTS TABLE (filled as windows land)

Machine verdict = the battery adjudicator's own line (postu1_window_results.txt).
Desk verdict = BATTERY-ANALYST adjudication on the full measured data
(jsonl receipts + server logs + standing exactness proofs), including the
structural-void corrections of section ADJ.

| window  | paired mean (t/s) | paired % | machine verdict | desk verdict |
|---------|-------------------|----------|-----------------|--------------|
| w23env  | +0.08             | +0.32%   | EXACT-VOID      | WIN-marginal, promotion blocked (no in-window sha proof) |
| ub1024  | +0.34             | +1.41%   | SPLIT           | PROMOTE (decode gate passed; prefill co-metric +5.0/+4.1% from server logs) |
| w19     | -24.38 (0.00 arms)| -100%    | ARM-CRASH       | ARM-CRASH x2 - lever re-closes (E-135(c)) |
| w22c1   | none (no cells)   | none     | CTRL-FAIL       | STRUCTURAL SKIP (hook not root) - lever UNMEASURED, not killed |
| w28nmax | -10.54            | -43.49%  | ACCEPT-MISSING  | LOSS + double ACCEPT-VOID - chain stays at 3 |

## ADJ. STRUCTURAL FINDING: decode-only cells cannot feed 3 of the 5 window laws

guard_battery.py --decode-only runs ONLY the decode guard
(guard_battery.py:833-895: prefill/determinism/canary/needle guards are all
skipped when args.decode_only is set). The battery's cell() launches every
10k cell with --decode-only, so cell logs NEVER contain
determinism_guard / prompt_tps / draft_accept lines, and the machine
adjudicator - which greps the cell log for exactly those lines - structurally
cannot pass three windows regardless of what was measured:

- w23env: det=None -> unconditional EXACT-VOID (happened, see below).
- ub1024: prefill=None -> prefill_ok always false -> best case SPLIT
  ("hold at 512/512") even on a decode WIN; decode is the gate per the
  E-127/E-133 law, prefill the co-metric - the co-metric was never measured.
- w28nmax: accept=None -> ACCEPT-MISSING unless the analyst recovers
  acceptance from the jsonl receipts (decode_res.accept_ratio IS recorded
  in decode-only mode).

The w23env ratchet text ("text_sha256 diverged") is factually wrong - the
sha was never measured. Analyst corrections applied per window below.

## PER-WINDOW SECTIONS

### 1. w23env (MMVQ env gates: IQ4XS_SHARE=1 + IQ3XXS_S2R=1 + IQ3S_S2R=1)

Cells 16:25-16:43 CDT. All 4 cells PROVENANCE binary=0d5eb814b0653589
config=c2c5656128f0d531; boot stamps 1560e872b (r1: tree_clean=True) and
260425807 docs-only (a1 True; a2/r2 tree_clean=False - SELF-INFLICTED by
this desk's untracked W32 draft file present in the repo 16:31-16:52,
disclosed in CHECKIN.log, zero code impact, file moved to /tmp scratch).

| cell | decode t/s | draft_n/acc | mean_len | void-gate (indep re-run) |
|------|-----------|-------------|----------|--------------------------|
| w23r1 ctrl | 24.69 | 126/84 | 3.00 | PASS 1375 MHz |
| w23a1 arm  | 24.80 | 126/84 | 3.00 | PASS 1278 MHz |
| w23a2 arm  | 24.73 | 126/84 | 3.00 | PASS 1206 MHz |
| w23r2 ctrl | 24.68 | 126/84 | 3.00 | PASS 1171 MHz |

PAIR1 +0.11 t/s (+0.45%), PAIR2 +0.05 t/s (+0.20%), PAIRED MEAN +0.08 t/s
(+0.32%). Controls 0.01 t/s apart - window noise is tiny, the paired mean
is a real positive. Machine verdict EXACT-VOID (structural, section ADJ).

Desk findings:
- Draft-chain invariance witness: draft_n=126, accepted=84, mean_len=3.00
  IDENTICAL in all 4 cells (same greedy path under the arm).
- Exactness: NOT proven in-window (decode-only never runs the sha guard).
  Standing evidence: W7 A4 exactness checks + W23 receipts (share/s2r arms
  exact by construction/offline oracle). Consistent with the in-window
  invariance witness, but the W23env kill law as written demands the
  in-window PASS -> promotion BLOCKED by this battery.
- Magnitude: +0.32% vs the +2.6% prediction (W23 pool estimate -3.4 ms/round
  on ~128 ms rounds). Direction right, size ~1/8 of prediction. Even if a
  re-run proves sha-exact, the served prize at 10k is ~1/4 of a glide step.
  E-104 replica numbers stay dead in practice.

Desk verdict: WIN-marginal on measurement; promotion blocked pending a
sha-proving re-run (arm cells without --decode-only), and the magnitude
makes the lever a low priority.

### 2. ub1024 (--batch-size 1024 --ubatch-size 1024 re-promotion, E-139a)

Cells 16:50-17:20 CDT. All 4 cells PROVENANCE tree_clean=True, binary
0d5eb814b0653589, config c2c5656128f0d531; commit drift during window is
docs/tools-only (d0507102d rev7 PLAN_CURRENT; 1c30abe7 W34 ft_score.py
instrument) - served binary byte-identical throughout.

| cell | decode t/s (guard) | prefill t/s (server log) | accept | void-gate (indep) |
|------|--------------------|--------------------------|--------|-------------------|
| ubr1 ctrl | 24.48 | 187.08 | 0.6667 | PASS 1250 MHz |
| uba1 arm  | 24.84 | 196.47 | 0.6614 | PASS 1352 MHz |
| uba2 arm  | 24.75 | 195.02 | 0.6614 | PASS 1288 MHz |
| ubr2 ctrl | 24.42 | 187.26 | 0.6667 | PASS 1230 MHz |

PAIR1 +0.36 t/s (+1.47%), PAIR2 +0.33 t/s (+1.35%), PAIRED MEAN +0.34 t/s
(+1.41%). Prefill pairs +5.02% / +4.14% (predicted +4.8/+5.3% - inside
band). Acceptance arm 0.6614 vs ctrl 0.6667 (-0.5 pts, arm draft chain
127 vs 126 generated - benign, no exactness law on this window).

Machine verdict SPLIT - structural (section ADJ: prefill=None in cell
logs); the ratchet text "prefill not; hold at 512/512" is contradicted by
the server-log prefill measured above.

Desk verdict: E-139a promotion criteria MET - decode gate passed (both
pairs positive, E-127/E-133 decode-default law) AND prefill co-metric
confirmed inside the prediction band. RECOMMEND PROMOTE: --batch-size 1024
--ubatch-size 1024 into the of-record launch (decode ~24.8, prefill
~196 t/s at 10k-class depth).

### 3. w19 (LLAMA_DRAFT_SHAPE_CACHE=1 on the E-138-fixed tree) - CRITICAL

Cells 17:21-17:41 CDT. The freshness pre-check passed (libllama.so carries
the engagement text) and the arm ENGAGED cleanly - and then the server
CRASHED in both arm cells, deterministically:

- w19r1 ctrl: PASS, decode 24.47 t/s, engagement 0 (correct), prefill
  186.16 t/s, void-gate PASS 1313 MHz.
- w19a1/w19a2 arms: guard request DIED ("Remote end closed connection
  without response", decode 0.00 FAIL both).

Crash signature (identical both arms, identical uptime timestamps):
1. "draft shape cache enabled (2 slots)" x2 at boot (target + draft
   contexts) - the cache is ON and ENGAGED (machine engagement 2/2).
2. Shape re-entries begin: 4 successful re-registrations across two graph
   shapes (warmup + request), each re-registering "norm-64" (simple tensor)
   and "cache_k_l64 (view)" - the E-138 re-register heal WORKS for these.
3. The NEXT shape re-entry hits
   ggml/src/ggml-backend-meta.cpp:832:
   GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN) failed
   -> abort, server death mid decode-guard request.

CRITICAL NEWS: the E-138 heal failed SERVED. It extended the crash-free
path by two shape re-entries (the pre-fix crash point is passed cleanly)
but a third draft-graph shape variant carries a split axis the meta buffer
cannot resolve (UNKNOWN), and the assert kills the process.

Machine verdict: ARM-CRASH, mean -24.38 t/s (-100%), ratchet "lever
re-closes (E-135(c) law)" - the classification is CORRECT (the failed arms'
"decode_tps 0.00 t/s" lines parse as numbers, so the paired math ran and
the assert check fired). Adjudicator robustness note: had the guard died
before printing any decode_tps line, the same state would have been
mislabeled CTRL-FAIL ("fix the control cell") - that did happen in w22c1
(next section).

Desk verdict: lever RE-CLOSES per the E-135(c) law (draft-path assert, no
retry in this battery). Heal docket for whoever reopens W19: either capture
the split axis at clear time or add an UNKNOWN-axis fallback (skip
registration, force re-derive) at ggml-backend-meta.cpp:832. Control cell
numbers confirm the underlying draft path itself is healthy without the
cache (24.47/24.30 t/s, acceptance 0.6667).

### 4. w22c1 (per-die clock floors R F F R + 200k F/R pair) - STRUCTURAL SKIP

NOT RUN - NOT A KILL. All 6 cells failed in ~1 s at the pre-boot hook:
"CLOCK-FLOOR-FAIL: root required (use: sudo /home/chris/set_clock_floor.sh
high/auto)". The battery unit was armed as user chris (non-root systemd
transient unit); set_clock_floor.sh needs root to write
power_dpm_force_performance_level and refused LOUDLY per its fail-loud
design (cell logs contain only "CELL-RESULT: <cell> FAIL
clock-floor-hook-refused"). Zero cells booted, zero clocks pinned, zero
measurements.

The W22 C1 kill law is untouched: it requires either clocks verified pinned
without t_d compression, or a measured paired delta < +0.5 t/s - neither
could happen. Lever state: UNMEASURED. Machine line CTRL-FAIL ("re-run
window after fixing the control cell") - the actual fix is a root context
for the hook (run the battery as root, or a NOPASSWD sudo wrapper for
set_clock_floor.sh), then re-run with --only w22c1.

Note: the orchestrator handoff prescribed "sudo /home/chris/
set_clock_floor.sh status once" before arming - the systemd arming path
skipped the root-context assumption. Arming-context defect, not a battery
logic defect.

### 5. w28nmax (--spec-draft-n-max 4, W28 chain-shape window)

Cells 17:45-18:02 CDT. All 4 cells PROVENANCE commit=444017736de3
tree_clean=True, binary 0d5eb814b0653589.

| cell | decode t/s | draft_n | accepted | accept ratio | void-gate (indep) |
|------|-----------|---------|----------|--------------|-------------------|
| w28r1 ctrl | 24.18 | 126 | 84 | 0.6667 | PASS 1274 MHz |
| w28a1 arm  | 13.68 | 167 | 84 | 0.5030 | PASS 1280 MHz |
| w28a2 arm  | 13.70 | 167 | 84 | 0.5030 | PASS 1288 MHz |
| w28r2 ctrl | 24.27 | 126 | 84 | 0.6667 | PASS 1228 MHz |

PAIR1 -10.50 t/s (-43.42%), PAIR2 -10.57 t/s (-43.55%), PAIRED MEAN
-10.54 t/s (-43.49%). Machine verdict ACCEPT-MISSING (structural - no
draft_accept console lines in decode-only mode); the guard's own cell
verdicts tell the story: both arms OVERALL FAIL on decode (-44%).

Desk adjudication on the jsonl receipts (accept_ratio IS recorded):
- ENGAGED: the 4-token chain fully engaged - draft_n 126 -> 167 for the
  same 128 decoded tokens (~3.97 drafts/round vs ctrl 3.00, W28 spec
  expected ~3.9).
- ACCEPTANCE COLLAPSED: accepted stayed EXACTLY 84 in all four cells - the
  4th chain slot contributed 0 of the 41 extra drafts. Per-draft acceptance
  0.6667 -> 0.5030 (-16.4 pts), below the 0.63 canary AND beyond the -5 pt
  hard-VOID band (double ACCEPT-VOID per the W28 law).
- The -43% decode loss is pure verify overhead: same 84 accepted tokens,
  41 more drafts verified per request.
- Invariance witness: identical accepted-token count and draft pattern in
  every cell (in-window sha proof structurally unavailable, section ADJ -
  same standing limitation as w23env).

Desk verdict: decisive LOSS + hard ACCEPT-VOID. The chain stays at 3.
W28's "acceptance is head-calibration-bound" census verdict is CONFIRMED
EMPIRICALLY served: extending the chain buys zero marginal accepts at this
depth/model. No re-run recommended; --spec-draft-n-max 4 is dead on this
model.

## ANOMALY LOG (chronological)

1. Pre-reboot battery instance (rev4, 16:06-16:22) was stopped while queued
   behind an svm-watchdog lock; machine rebooted; relaunch 16:25:47. The
   pre-reboot partial cells are NOT results (discarded).
2. w23a2/w23r2 tree_clean=False - SELF-INFLICTED by this desk: the
   untracked W32 draft file (written 16:31 into docs/amd-port/results/)
   dirties git status --porcelain, which guard_battery.py stamps at server
   boot. Binary hash + config hash identical across all cells (zero code
   impact). W32 moved to /tmp scratch at 16:52; all later cells
   tree_clean=True. Disclosed in CHECKIN.log.
3. svm-watchdog pauses (normal per watchdog law, battery resumed
   automatically each time): ~16:41:42-16:50 (high=4->5), ~17:08-17:15
   (high=7), ~17:30-17:35 (high=11).
4. w19 ARM-CRASH x2 (critical; section 3) - E-138 heal failed served,
   ggml-backend-meta.cpp:832 UNKNOWN split-axis assert.
5. w22c1 structural skip (section 4) - unit armed without root context.
6. Machine-adjudicator structural voids (section ADJ) - w23env EXACT-VOID
   and w28nmax ACCEPT-MISSING were predetermined by --decode-only guard
   coverage, not by measurements; ub1024 SPLIT likewise. Desk corrections
   applied per window above.
7. PROVENANCE commit drift during the run (all docs/tools-only, binary
   0d5eb814b0653589 constant): 1560e872b (rev5, boot) -> 260425807 (rev6,
   16:26) -> d0507102d (rev7, 16:48) -> 1c30abe71 (W34 tools, 17:15) ->
   444017736 (rev9 docs, 17:19). All stamps tree_clean=True after 16:52.

## E-145 LEDGER ENTRY

E-145 (post-U1 lever battery, W32): five paired served windows of the
E-139 queue + W28, one ordered unit run 16:25:47-18:02:16 CDT 2026-09-24
(1h36m; systemd campaign-postu1-battery; R/A/A/R 10k decode-only cells at
ctx 10240 on the TP4 stack of record; void_gate --die3-group 2 PASS on
every measured cell, independently reproduced). Verdicts (machine / desk):

- w23env (MMVQ IQ4XS_SHARE+IQ3XXS_S2R+IQ3S_S2R): EXACT-VOID (structural,
  sha guard never runs in decode-only) / WIN-marginal +0.08 t/s (+0.32%),
  both pairs positive, controls 0.01 apart, draft-chain invariance witness
  perfect; NOT promoted - in-window exactness law unsatisfiable in this
  harness and magnitude ~1/8 of the +2.6% prediction. E-104 stays.
- ub1024 (batch/ubatch 1024): SPLIT (structural, prefill guard never runs)
  / decode gate PASSED +0.34 t/s (+1.41%; pairs +0.36/+0.33) AND prefill
  co-metric +5.02/+4.14% from server logs (predicted +4.8/+5.3) -
  E-139a promotion criteria MET ON MEASUREMENT. PROMOTED by desk
  adjudication: --batch-size 1024 --ubatch-size 1024 into the of-record
  launch. (Ratchet procedure per E-137: single paired WIN -> baseline
  ratchet; this is the only promotable window, machine line blocked by the
  harness defect below.)
- w19 (LLAMA_DRAFT_SHAPE_CACHE): ARM-CRASH x2 (real, deterministic) -
  engagement 2/2 then ggml-backend-meta.cpp:832 UNKNOWN split-axis assert
  on the third draft-graph shape re-entry; E-138 heal works for 2 shape
  re-entries (4 clean re-registers) then fails served. Lever RE-CLOSES per
  E-135(c), no retry in this battery. Heal docket: capture axis at clear
  time or UNKNOWN-axis fallback (skip+re-derive).
- w22c1 (per-die clock floors): STRUCTURAL SKIP - unit armed without root,
  all 6 hooks refused loudly, zero measurements. NOT a kill (W22 C1 law
  untested). Re-run: root context or NOPASSWD wrapper, then
  run_post_u1_battery.sh --only w22c1 (re-runnable, skips nothing else).
- w28nmax (--spec-draft-n-max 4): ACCEPT-MISSING (structural) / decisive
  LOSS -10.54 t/s (-43.49%) + double hard ACCEPT-VOID (0.5030 < 0.63
  canary; -16.4 pts < -5 band); chain fully engaged (3.97 drafts/round)
  but 0/41 marginal accepts. Chain stays at 3; W28 head-calibration-bound
  census CONFIRMED served; lever dead on this model.

HARNESS DEFECT for the docket: guard_battery.py --decode-only runs only the
decode guard, so decode-only cells can never feed the w23env exactness law,
the ub1024 prefill co-metric, or the w28nmax acceptance law - the machine
adjudicator structurally voids 3 of 5 windows regardless of measurements.
Fix: either run decision cells with the full battery (determinism+prefill+
canary guards on) or extend --decode-only to record text_sha256 +
prompt_tps + accept_ratio lines in its console summary (accept_ratio is
ALREADY in the jsonl receipt).

Net campaign state after E-145: promoted = ub1024 (desk-adjudicated).
Closed = w19 (re-close), w28nmax chain-4 (falsified). Unmeasured = w22c1
(re-run pending root fix). Marginal/blocked = w23env (sha-proof re-run
optional, low priority). Remaining served levers: U2 q40-prefill window
(rev7 merge-order law: merge after battery, before U2) and the W22 C1
re-run.

## RUN TIMELINE (cell boots, CDT)

w23r1 16:25:48, w23a1 16:30:41, w23a2 16:34:55, w23r2 16:39:09 (verdict
16:43 EXACT-VOID) | watchdog 16:41-16:50 | ubr1 16:50, uba1 ~16:59, uba2
17:02, ubr2 17:16 (verdict 17:21 SPLIT; watchdog 17:08-17:15) | w19r1
17:21, w19a1 17:26 (CRASH), w19a2 17:30 (CRASH), w19r2 17:36 (verdict
17:41 ARM-CRASH) | w22c1 6x hook-refused 17:45 (verdict 17:45 CTRL-FAIL) |
w28r1 17:45, w28a1 17:49, w28a2 17:53, w28r2 17:58 (verdict 18:02
ACCEPT-MISSING) | POSTU1-BATTERY-DONE 18:02:16, unit inactive.
