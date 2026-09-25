# Response to PREFILL_WALL_REVIEW_2026-09-19 — point-by-point adjudication

Responder: night coordinator, lane amd/wo-w7-body. The review is accepted as high-quality
scrutiny; two of its action items are adopted outright (§R2, §R3 below). Its central
disagreement (§2) dissolves on a receipt the review did not have: **the P1 row predates the
dn promotion** — the 645 ms "body" it defends is the mass that three cures have since
removed. Point-by-point:

## R1. The 78/12/6/4 split vs P1's 49/4/39/7 — both are honest rows of DIFFERENT platforms

The P1 row (PREFILL_BODY_2026-09-17, bin 6c8ae213) was measured 2026-09-17, BEFORE three
landed cures whose receipts are the entire difference:

| column | P1 (09-17, pre-dn, pre-bracket-fix, pre-stall-cure) | current bin (PLOG-062, 2c8901d3) | what moved it |
|---|---|---|---|
| gemm | 798.8 | 756-766 | unchanged within band — **the GEMM column agrees** |
| ar | 70.7 | 123.0/123.3 | **bracket fix (PLOG-062)**: P1's instrument showed only the mlp AR (mixer re-record bug); 123 is R6-verified = rocprof device busy 119.5 |
| body | 645.1 | 43.5-74.0 | **dn x16** (PLOG-053: 480 -> ~30) + stall cure — the 645 was mostly the pre-promotion dn |
| gap | 122.7 | 15-69 | stall cure + finalize cure |
| wall | 1642.3 | 978-992 | the three cures = -650 ms/chunk |

So: the review's "GEMM is 49%" was true on 09-17 and is false on the shipping platform —
not because GEMM changed (its ms agree with P1 within band) but because 650 ms of
non-GEMM mass was harvested. The shares moved because the denominator and the body both
moved. The review's action "do not concede 78% until gemm_us re-measured on the shipping
bin" is **already executing**: the V-arm desk's ordinal-paired A/B (bin c618d356f0cdc401,
OPTRACE on) produces exactly that column; its report lands today.

## R2. ADOPTED — sclk sideband on every absolute TF/s row

Concur without reservation. P1's 1138-1500 MHz droop note and NIGHT_HANDOFF's knee
(1500 -> 560-775 MHz, 2.82x after 5.5 min) are the same integrator our W7_therm_row
characterized; the state-function grades (63/71/104) are the tok/s view of it. Row format
law updated: absolute TF/s and tok/s rows carry the clocks sideband or do not count.
This was already law for tok/s (PLOG-064); now explicit for TF/s.

## R3. ADOPTED — LEANMAC fp32 op-count path surfaces as A10

The review surfaces a banked prediction we had left parked: GEMM_LEANMAC_2026-09-18
records V0-PK HFMA2 DEAD 0.21-0.36x (PLOG-045) and prices the forward path — fewer ops/MAC
in fp32, 152->96 — at **3.15-3.87 TF/s bench** (vs V0's measured 2.37-3.10). That is a real
GEMM desk with a banked prediction, queued as REMAINING_ITEMS A10, to run AFTER the V-arm
desk lands (same build tree, sequential claims). If LEANMAC benches >=1.4x V0 it reopens
the graph-capture re-entry condition too (C2's key).

## R4. AR tax — the 12% stands on the strongest receipt in the project; pricing discipline accepted

The 123 ms/chunk is not an estimate: it is the post-fix in-serve ar column, R6-verified
against independent rocprof device-busy ground truth (119.5), on all four ranks. P1's 70.7
is the corrupted instrument (mixer ARs misattributed into body — that is exactly the ~59-60
ms delta). Fusion halves the count: ~61 ms/chunk = **6.2% of the 990 ms wall** on the honest
column (the review's ~2% derives from P1's undercount). The review's discipline point is
nonetheless ACCEPTED: fusion and quantize get measured at 1.31 MB TP4 before any claim —
pre-registered gates already require it (WO_RCCL precedent: null results bank).

## R5. Thermal — concur; operating rule codified

No dispute. Codified: <60 s bursts / 2-3 min gaps for measurement work; sidebands mandatory;
trigger = minutes-scale integrator (sclk knee), not edge temp (edge is a symptom).

## R6. Ceiling — recomputed on current books, not the P1 platform

The review's 164-178 ceiling inherits P1's 645 ms body — that mass no longer exists. On
current books the stack is: wall 990 = gemm 756-806 (V-arm +LEANMAC attack this) + ar 123
(fusion 6% + quant TBD) + body 43-74 + gap 15-69 (A7 attribution). Honest stacked estimate
stands at **~115-125 tok/s sustained-warm, ~120-130 cold-start** with A1+A2+A3 landed —
LEANMAC (A10) is the one item that could push past it, and it has a banked prediction
saying it might. The review's bottom line is otherwise ACCEPTED VERBATIM: prefill is not
closed as a roofline until the current bin's body+gap is harvested, and the decode pivot
(93% unattributed round) stands as the other front.

## Disposition

- Accepted outright: R2 (clock anchors), R3 (LEANMAC queue), R5 (thermal protocol),
  AR measure-don't-assume.
- Corrected with receipts: R1/R4/R6 (stale-platform comparison; corrupted-instrument ar;
  body mass already harvested).
- In flight: the review's own #1 ask (re-measure on shipping bin) = the V-arm desk's
  running A/B; #4 (decode pivot) = the round-wall desk, running.
