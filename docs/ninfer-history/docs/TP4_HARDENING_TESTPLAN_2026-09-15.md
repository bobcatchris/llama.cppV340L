# TP4 HARDENING TEST PLAN — real scale, team-green precedent (chair, 2026-09-15 09:5xZ)

User order: diligence beyond 10k — "10k is not a real scale." The bar this repo already carries from the NVIDIA line: **int8@200k = 199.8k tok @ 59.0 tok/s** (docs/59_roadmap L2, VRAM_LEDGER), bf16 context sweeps 10k..80k with latency curve 71.9→60.8, 40k/80k/160k parity rows, bf16@120k field capability. The AMD w4 line has produced **zero** numbers at any of these scales. This plan closes that gap with the era's own laws attached.

## Laws this plan runs under (non-negotiable, all banked)
- **VRAM law**: allocator is the only gate. No estimated refusal, no reserve constants, no budget arithmetic that can say no. A rung that doesn't fit dies by cudaMalloc's own words, logged verbatim.
- **LITH**: any launch-safety number cited must be MEASURED on THIS machine at comparable geometry, or the number does not exist.
- **Capture-before-kill**: a hang at depth is a datum that cannot be read twice (stall-watch law) — the boot race's first real reappearance chance is at 80k+, and the capture protocol is pre-stated.
- **Determinism-equality**: every rung carries its pair leg; digests must match or the rung reads as a finding, not a flake.
- **Budget-armed expectations**: cancelled ≠ failed, length ≠ stop — the runner's facet-3/4 cures apply; PROBE_CURL_M per rung = measured arithmetic shown in the row, never a guess.

## The ladder (world=4, untraced, temp=0, one boot session per rung-group, serial)
| rung | context | deliverables |
|---|---|---|
| H1 | 1k | baseline tok/s, sanity |
| H2 | 10k | first real decode number (user minimum, already dispatched) |
| H3 | 40k | green-parity scale #1 |
| H4 | 80k | green-parity scale #2 — where their bf16 curve lives |
| H5 | 160k | green-parity scale #3 — allocator answers, we log its words |
| H6 | 200k | their headline number's scale, if H5 fit |

Per rung: (a) THROUGHPUT tok/s total + early-vs-late window deltas (KV/GDN-state growth cost becomes visible, measured not modeled); (b) STABILITY: finish_reason, pad/sentinel witness (ids ≥248,077 = 0 expected), crash/hang = capture-first; (c) COHERENCE at depth: first/last 200 chars banked — attractor regression at depth means the cure is budget-limited and joins the residue list as a real member, said plainly; (d) DETERMINISM pair same-boot; (e) state-growth rows: KV pages + GDN state footprint at rung depth, measured.

## Controls and follow-ons
- **w2 parity control**: same prompts at w2 where capacity allows — the era's original grader (per-leg digest vs w2 oracle) extended to scale.
- **SOAK**: 10× repeated 10k-generation legs, one boot — throughput variance + depth-stability; any single-leg digest drift = finding.
- **MTP**: stays its own WO (never-booted frontier, same organ class as the soup) — sequenced AFTER the ladder, not inside it.
- **Near-tie audit + 59-token completion**: folded into H2's captures where the dump budget allows, else stay queued.

## Failure semantics (pre-declared, no post-hoc goalposts)
Any rung failing = the rung's row banks RED with its capture; the ladder continues at the largest passing rung; nobody retries a failure away; the residue list gains members, never loses them. If H2 (10k) fails, that is exactly the user's suspicion confirmed and the hardening story starts there with facts, not vibes.

Owners: boots = agent4 (serial, grant per session); grading rows = agent1's instruments (map, checker, grader — all in main now); cells/battery = agent3/agent2 at each new tip; ledger per rung. This doc is the ladder's pre-registered expectation table: numbers before adjectives, controls before claims.
