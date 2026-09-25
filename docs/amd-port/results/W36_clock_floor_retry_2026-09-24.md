# W36 CLOCK-FLOOR RETRY (w22c1) - ADJUDICATION RECEIPT - 2026-09-24

Desk: W22C1 CLOCK-FLOOR ADJUDICATION DESK (on amd/v340-port-v2 @ 4bf0d6476).
Machine: root unit campaign-postu1-battery, `--only w22c1` (started 18:18,
final cell landed 19:39:56 CDT). This desk is ZERO-GPU: read-only on
/home/chris artifacts + docs-only in-repo. Foundations: W22 die-asymmetry
spec (C1 spec + kill law, 2026-09-24), W18 soak-mechanism receipt
(git 9761f0821, amd/soak-mech), ledger E-133 (void gate), E-147(4)
(unprivileged CTRL-FAIL -> root retry queued), E-148 (root retry + ub1024
promotion, of-record decode 24.69).

## 1. Design (as run)

Window w22c1, R F F R 10k decode-only pairs + one 200k anti-decay pair
(F then R). Floor arm F = `power_dpm_force_performance_level=high` on all
four dies via /home/chris/set_clock_floor.sh (root step, SYSTEM STATE, not
env; EXTRA_ENV empty both arms). Arm stamps in every cell header:
`clock_floor_state` + `clock_freqs_mhz` readback. Per-cell gates:
wait_free, 300 s cooldown after each clock-state change (W22), 180 s
inter-cell settle, die-hot gate, 18-col thermal sideband + void_gate.py
(E-133, die-3-of-record = sampler group 2). Provenance identical on all
six cells: binary=0d5eb814b0653589 config=c2c5656128f0d531
tree_clean=True, built from 68654e245 (E-148).

## 2. Six-cell dataset (complete)

| cell   | pos | arm | floor stamp      | ctx   | decode t/s | vs 24.69 | gate  | die-3 act_mean |
|--------|-----|-----|------------------|-------|-----------|----------|-------|----------------|
| w22r1  | 1   | R   | auto x4 / 9 MHz  | 10240 | 24.66     | -0.11%   | PASS  | 1255 MHz       |
| w22f1  | 2   | F   | high x4 / 1496   | 10240 | 24.34     | -1.42%   | PASS  | 1310 MHz       |
| w22f2  | 3   | F   | high x4 / 1496   | 10240 | 24.13     | -2.26%   | PASS  | 1192 MHz       |
| w22r2  | 4   | R   | auto x4 / 9 MHz  | 10240 | 24.49     | -0.82%   | PASS  | 1302 MHz       |
| w22f200| 5   | F   | high x4 / 1496   | 200000| 23.90     | -3.20%   | VOID  | 1209 MHz       |
| w22r200| 6   | R   | auto x4 / 9 MHz  | 200000| 24.04     | -2.65%   | PASS  | 1238 MHz       |

200k cells carry full five-guard batteries: f200 prefill 212.18 (-2.54%),
r200 prefill 214.89 (-1.29%); determinism + needle_recall + mtp_canary
PASS everywhere; draft acceptance 0.66667 IDENTICAL in all six cells
(clock state is numerics-neutral, as the W22 spec predicted).

Provenance note: `commit=unknown` on all six cells is a root-run git-read
quirk in the guard; the chain is intact via identical binary/config
hashes + tree_clean=True. Harness nit, not a cell failure.

## 3. Verdict per the W22 kill law: KILL

PROMOTE bar: floor arm gains >= +0.5 t/s over the paired regular arm WITH
a pinning witness. Measured - the floor arm lost every pair:

    PAIR1  w22f1  - w22r1  = -0.32 t/s  (-1.30%)
    PAIR2  w22f2  - w22r2  = -0.36 t/s  (-1.47%)
    200K   w22f200- w22r200= -0.14 t/s  (anti-decay receipt, F then R)
    PAIRED MEAN (10k)      = -0.34 t/s  (-1.38%)

Position-matched, both 10k pairs negative -> machine base LOSS; desk
verdict KILL per the W22 C1 kill criterion 1 (paired delta < +0.5 t/s at
matched position). No ratchet: baseline of-record UNCHANGED at decode
24.69 t/s (cells/decode_8k_10k/decode_per_second); set_clock_floor.sh
stays OUT of the boot chain.

Order strength: the 200k pair ran F then R, so r200 held the warmer,
later, soak-penalized position and STILL won - the true floor deficit at
200k is >= 0.14 t/s. Boot-decay drift (R1->R2 = -0.17, F1->F2 = -0.21)
is deconfounded by the within-pair deltas, which are consistent
(-0.32 / -0.36), and by position: r2 started warmer than f1 (37-46 C vs
39-52 C edge) and still ran faster. The floor cost is real, ~0.2-0.3 t/s
at matched position, not a soak artifact.

## 4. Thermal / clock evidence (mechanism)

- PINNING WITNESS AT STAMP: both F arms stamped high x4 / 1496 MHz x4 at
  boot; R arms auto x4. The FLOOR-VOID clause does not fire - the arm was
  correctly stamped. The witness then testifies AGAINST the lever:
- f1 PINNED CLEAN IN-RUN and was still slower: <=991 MHz duty 0.0% on all
  four dies, >=1100 MHz duty 100% (full span), die-3-of-record mean
  976 -> 1240 MHz vs r1 - and the pair still went -0.32 t/s. Decode on
  this stack is NOT sclk-limited in the ~1100-1300 MHz band.
- f2 PIN LEAKED under warm soak: 991-band dips at 9-27% duty, fleet-wide
  991 dips at t=15/25/45 s; its die-3 act_mean (1192) came in BELOW the
  regular arm's r2 (1302). The floor only added heat, not clocks.
- f200 FLOOR DID NOT SURVIVE THE NEEDLE: gate VOID - die-3-of-record
  act_mean 1209 MHz with <=991 MHz duty 22% (> 20% threshold); die2
  latched the 775 MHz band from ~100 s to end (one 560 sample); die3 sat
  the 991 band through the needle. Junction max 91 C. The W18 anti-decay
  claim is refuted inside the very arm built to prove it.
- R CELLS DO DIP (the target phenomenon exists): sub-1496 samples are
  67-100% in every cell, R or F; R-cell far-die act_means 1255/1302/1238
  with gate PASS. So the W18 latch is real at serve state - but it is
  driver/power-state-controlled and costs nothing measurable at this
  shape: there was something for the floor to prevent, and preventing it
  (where the pin held) bought negative t/s.
- HAZARD WITNESS: under the 200k needle HBM runs at 90-93 C vs crit 95 C
  on BOTH arms (die1 hit 93 C with floors set; edge start 43 C). Clock
  forcing during 200k-class needles spends the last 2-3 C of HBM margin
  for a measured NEGATIVE return. Recorded for the future: any
  clock-forcing retry must come from SMU/od8 level (not dpm_sclk) and
  must carry a memory-temp guard. NOT queued - the lever is closed.

## 5. Mechanism conclusion (one line)

The power_dpm level=high 1496 MHz floor does not survive decode load
(leaks under warm soak, latches through the 200k needle), and where it
did pin cleanly (f1) decode got SLOWER - short-prompt decode on this
stack is not sclk-limited in the 1100-1300 MHz band, and the W18 far-die
latch is unreachable from this knob.

## 6. What this closes

- C1 PER-DIE CLOCK FLOORS: CLOSED on kill criterion 1. No boot-chain
  change; set_clock_floor.sh remains a diagnostic, not a launch component.
- The ANTI-DECAY form of the lever (W18's -26% class): REFUTED at serve
  state - the floored 200k cell voided through the throttle band anyway.
- Die-asymmetry lever family: byte-rebalance already closed (E-125);
  with C1 closed, no further arms. Because clocks were never truly pinned
  under load, the W10 t_d-compression discriminator never ran clean - the
  silicon/HBM-class residual question passes to C3 (serve-state rate
  re-measurement) as the analysis-only closing post-mortem, per the W22
  spec. C2 (host P-core pinning) is independent and unaffected.
- Of-record UNCHANGED: decode 24.69 / prefill ~217.71 gate / accept 0.67,
  TP4+RCCL+fa40+ub1024.

## 7. Harness notes (for the record)

- r200 floor-clear hook hit a TRANSIENT TEARDOWN RACE on its first
  attempt (clear fired while amdgpu still held f200's forced-high level;
  write refused; fail-closed refusal stub landed). The runner re-attempted
  and the cell ran clean (auto x4 stamp, full PASS summary, 19:39:56).
  Coordinator-owned fix queued: post-teardown settle or one ~90 s retry
  in the hook path before declaring a cell refused.
- `commit=unknown` in PROVENANCE under the root-run battery (root cannot
  read the user git context): nit; binary/config hashes identical across
  all six cells carry the chain.
- The machine adjudicator's console/result line lands after the runner's
  post-cell settle gates; this receipt adjudicates from the six landed
  cell files per the same law the adjudicator encodes (desk adjudication
  precedent, E-148). A stale `CELL-RESULT: w22r200 FAIL
  clock-floor-hook-refused` stub from the 17:45 unprivileged attempt
  (E-147(4)) existed on disk until the root run overwrote it at r200 boot.

## 8. Identity

Zero GPU work: battery owns all four dies throughout; this desk performed
read-only host reads of /home/chris artifacts (cell summaries, thermal
sidebands, void_gate.py runs in observe mode) and docs-only repo edits.
No server launches, no lock contact, no reboots.
