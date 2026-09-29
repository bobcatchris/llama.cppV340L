# WO-VRAM-1 — de-constant the preflight: allocator is the gate (three measured paths, one family)

**Issued**: coordinator pi 01a09a03, 2026-09-13 ~11:0xZ. **Owner**: agent4 (serve lane — holds the adjudicated tp_engine carries: the +26 env-gate, measured-floor context); any restarted session picks this up from the ledger. **Class**: VRAM LAW conformance — banned-constant eradication, NOT a re-baseline. **Region caution**: this is the ANTI-RESURRECTION canonical home (main's tp_engine preflight region + tp2_budget.h). Before coding, `git diff main -- src/runtime/tp2/tp_engine.cpp src/runtime/tp2/tp2_budget.h` and re-read AGENTS.md's law: main's version of the region wins; carry only the adjudicated amd deltas (ws-warrant env-gate et al.), named in the merge msg.

## The three decision-path constants (all seat-verified; receipts named)

1. **`headroom_bytes = 1024 MiB` unconditional** — `tp2_budget.h:54`; summed by `fixed_bytes()` (:85-89); `max_context_fitting` returns 0 when `usable <= fixed_total` (:281) → auto-KV refusal. Composition arithmetic reproduces at two seats: 7102+592+96+200+160+1024 = 9174 = 17g2's own printout (agent3's instance #6 @ #740; agent5's re-verification @ `ITEM6_decode_anomaly_agent5.md` Appendix, bcb6c1da; chair reads of :54/:85/:281 this minute). This is the phantom that made agent2's auto-KV attempt-1 refuse at "legacy-estimate = 9059" while the measured need was 7102.
2. **placement-basis throw→estimate fallback** — `tp_engine.cpp:~868-876`: reader THROW (e.g. truncated artifact!) → catch → `placement_basis` stays `"legacy-estimate"` (9059-era constant) → charges against genuinely-feasible cards. This is the exact path agent2's #698 attempts took. A corrupt-artifact case and a capacity case share one misdirected message ("stale-residue or foreign contexts") — the refusal is wrong AND the diagnosis is wrong.
3. **`kGateReliefBytes = 1200 MiB` static slack** — `tp_engine.cpp:~1062-1067`: the measured-floor gate is `claimed − 1200 MiB`, a remembered number on the refusal path (directionally lenient, still a constant that decides; the user-signed `docs/VRAM_LEDGER.md` era produced it before the law's final form). Conform it in the same pass or name it in the row as a known residual — do not leave a fourth surprise for the next audit seat.

## Conformance target (the law verbatim)

- **Report measured only**: preflight/budget printouts carry live `cudaMemGetInfo` (hip) free-bytes + per-term composition **with each term's name and provenance** (agent5: "make the log do the work" — agent3 proved a reader can reconstruct the composition anyway; the log should not require it).
- **Decide by allocator**: if capacity doesn't fit, `cudaMalloc` says so, cleanly, in real time. No constant, multiplier, or slack term may flip a launch from try→refuse. A refusal must cite a LIVE free-bytes read vs an ACTUAL requested-bytes figure (or the real malloc error), never a budget expression.
- **Distinguish corrupt from tight**: a throwing placement-reader must refuse *as a reader error* (surface the exception, exit loud) — never fall through to an estimate that produces a capacity-shaped lie.

## Removal ORDER (polarity law, agent5 @ 892891ee, chair-embodied): the three constants are NOT the same polarity — do not delete them alike.
- **headroom_bytes voted STRICTLY** (adds to fixed → refused fits): straight DELETE is monotone-safer — un-refusing what the allocator can judge live.
- **kGateReliefBytes voted LENIENTLY** (claimed − 1200 → un-refused false kills; its comment cites the 16,679/15,069 rows): deleting it WHILE ANY CLAIMED-ESTIMATE BASIS REMAINS makes the gate STRICTER and re-factories the exact recurrence class the law exists to end. **Order: basis-to-measured FIRST; the relief dies with the basis it was patching, not before it.** (Bandage-off-healed-wound vs take-the-wound-out-first.)
- The throw→legacy fallback (path 2) converts in the same first act: a throwing reader must exit as INSTRUMENT-ERROR, never revive a capacity-shaped estimate.

## Test gate (mandatory, per VRAM LAW's own clause + LAW 17)

- **Synthetic near-capacity cell**: geometry engineered so every ESTIMATE says no but the live reads say maybe → the cell must **LAUNCH and MEASURE**, not refuse. This is the cell the law demands ("a synthetic near-capacity cell must LAUNCH and measure, not refuse") and it doubles as the falsifier: plant it in the negative suite so a future constant on the decision path goes RED at CI, not at a user's boot.
- Falsifier both directions (law 17): a planted rogue constant charge must be CAUGHT by a lint/predicate leg (grep the decision path for literal MiB terms feeding `return 0`/refusal — bounded by the three named today; a bounded listing proves nothing about absence, so the cell, not the grep, is the witness).
- Step-0 anti-resurrection diff clean vs main's stale region; three-state exit; name the baseline of every run.

## Sequencing

17g4 (agent4, swap boot) lands first — it closes Phase-1 item 1 and this WO is not boot-blocking (explicit-capacity launches bypass path 1; the kit already names capacity). WO-VRAM-1 rides agent4's next CPU pass or the first restarted session's dispatch; the decode-anomaly item 6 (agent5's seat) and this share a log line — the composition printout serves both.

**Definition of done**: three constants off the decision path (or named-and-warranted residuals with a follow-up cite), near-capacity synthetic cell LAUNCHING green in CI, reader-throw refusal as instrument-error class, preflight printout self-explaining. Until then: use explicit `--max-context/--kv-capacity` (the kit does) — the auto path stays a known hazard, which this WO exists to end.
