# DP DRIVER PROGRAM - DECISION BRIEF (one page, 2026-09-28)

**For:** the owner deciding what happens next on the V340 box.
**Status:** program CLOSED at its measured plateau (ledger E-166..E-179, all
committed). 15 guard cells, 3 stacks, 4 AR configs, 2 kernels. Every valid
cell: byte-identical text, all quality gates green.

## WHAT WAS DELIVERED (all measured, all receipted)

1. **Vega10 PCIe P2P works on mainline.** 12/12 die-pair directions,
   byte-exact, cross-card within 3% of intra-card, 6.5-7.2 GB/s at 1 MB.
   The old blockers were firmware ACS bits (runtime-clearable) and a
   one-line P2PDMA whitelist gap - not silicon.
2. **A self-healing boot stack**: the ACS-clear service + whitelist kernel
   survived 6+ hard crashes unattended and rebuilds the P2P topology on
   every boot.
3. **THE ROCK 7.14 runtime path is serving-capable**: full guard battery
   passing, text byte-identical to of-record, IPC works (6.2's didn't),
   RCCL 7.14 inits natively on the P2P topology.
4. **A custom tree allreduce** (default-off) with a thermally-matched
   +4-8% decode win over RCCL-SHM in interleaved pairs - banked
   infrastructure that converts to a served win whenever the thermal
   ceiling moves.
5. **Findings for upstream** (drafted, owner-fileable): ROCm 6.2 IPC
   ioctl gap, the dst-stream peer-copy corruption, the sticky-error class.

## THE NUMBERS THAT MATTER

- Served decode at the 8k-10k guard cell: **25.3-25.8 t/s regardless of
  transport** (thermal state swings it 22.9-26.5). Transport config is a
  second-order effect; the wall is die-skew + thermal.
- Prefill: 236-246 t/s on all configs (the tree does not touch prefill).
- The tree's per-boundary floor is real: 124 us vs RCCL's 210 us matched -
  it shows as the matched-window win, not in absolute cells, because the
  boundary wall is arrival-skew dominated (W39 confirmed).

## THE DECISIONS

1. **ADOPT THE STACK** (one command, reversible, re-stamps the baseline):
       sudo docs/amd-port/scripts/dp4_adopt_serving.sh
   Switches serving to the dp2 kernel + THE ROCK 7.14 binary + AR_TREE.
   All elements measured; the 6.2 stack stays available as a boot entry.
2. **THE HARDWARE INSPECTION** (E-162): power delivery + HBM cooling.
   This is the only lever that moves the decode wall itself - every
   software rung is pulled, and the crash pattern (burst-churn deaths,
   instant resets, non-ECC board) points here.
3. **OPTIONAL**: file the upstream bug drafts
   (results/dp_upstream_bug_reports.md) - they are written, receipted,
   and ready.

## WHAT NOT TO RE-OPEN (all closed with receipts)

Transport tuning at 8-10k without thermal matching; dpm clock floors;
iq4_xs requant ladder; PP/tp2xpp2; NCCL channel/env sweep; LDS-y;
undervolt on this cooling (re-open on hardware change only).
