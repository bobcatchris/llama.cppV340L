# CHAIR HANDOFF — full debrief, 11:3xZ 2026-09-14 (for the incoming chair; the outgoing desk wrote this at the user's request)

**READ ME FIRST, then execute §"NOW" — the board is fast and idle is the only failure mode left.**

## Position (all tool-verified at 11:2xZ, main tip `7e65b35b`)
- **TP2/item-7 era: CLOSED.** Determinism triple (RED 1d0ff3c6/GREEN db09c924/WIRED check-(r) hard-gate); A-4 guard-relax + GATE-2 routing + rank-aliasing fix + GATE-3 memory-safety refusal + WO-TP4-E factory relax ALL MERGED; roster complete and **certified by two seats with an executed RED twin** (cross-baseline `gate --baseline 393ce73d` rc=0).
- **TP4 = ONE CODE LEG FROM FIRST-LIGHT: R1 argmax transport (WO-TP4-B).** Everything else is paper-green: window G-AMD-18 announced (slots filled), step-0 source gate in §5 of the announcement (agent1's PRED-A/B/C), grants live, pair+quads free, disk 12 G, build measured to fit (~17 min incremental), first-light runner pre-staged+env-parameterised.
- **R1 spec = the merged handoff, not a redesign**: `docs/amd/WO_TP4_B_R1_DESIGN_HANDOFF_agent4.md` @ main (from 65081d2e). §1 corrected port (per-rank shard argmax → RCCL exchange of kMaxTokens×16B champions+sumexp → reduce by (val, global-token tie-break) → conf from summed sumexp — NO logits-row gathering: my chair phrase "allgather+local-argmax" was WRONG and would hit GATE-3's overflow refusal). §4b SECOND pair-shaped gate: `one_shot_argmax.cu:434` rank∈{0,1} throw + Impl 2-array shape (conf[2], host_payload[2]) — expect a third, same lineage one layer down. §2 triple with the RED leg ALREADY BANKED (today's loud throws = /tmp/G18_*.log + first-light must DECLARE AR-silence kind: kind=ar>0 = measurement, =0 = inference EVEN AT world=4, per 046e871b clause).
- **B-1 four-die: RUNNING as I type** (agent5; witnessed alive 11:18+11:20Z, cross-package 0,2 leg, 200 reps). Rows ETA ~11:3xZ. It stalled TWICE on an unbounded rocm-smi precheck (root-caused, hardened to exit-4-unknown=occupied — read the release row when it lands).
- **Hunt state**: lag = per-step monotone late ramp, rank0-carried, NO period (guard certified BOTH directions), not time/duty/thermal/entry-path; suspect list closed EMPTY by agent3; latch one-row `docs/amd/LATCH_ONE_ROW_2026-09-14.md`. K=16 stands four bins. MTP un-parked by user, +2 GB fixed cost known, draft-tokens ∈ [0,3], waits for post-R1 geometry.

## NOW — the parallel graph (this is what "velocity" means tonight)
| seat | leg | due |
|---|---|---|
| **agent4** (fresh) | **WRITE R1** from the handoff spec, own lane `amd/wo-p3-serve` tip; cell first (RED already banked), then incremental build, bank bin BEFORE relink, then fire first-light runner (its one edit: expect-throw → expect-serve) | rows as they land; ~2-3 h |
| agent1 | second-witness R1 spec vs GATE-2/GATE-3 predicate family (any THIRD pair-shaped site = your class, name it pre-build); refresh step-0 PRED-B flip-check at the R1 tip; flip GO-AMENDED when green | same-build window |
| agent3 (fresh) | read seat armed for B-1 rows (world-banner + era-declaration per 3d2a5c95) + independent §4b gate-map read of one_shot_argmax.cu (two-desk pattern; conflicts = data) | B-1 read ≤15 min of rows |
| agent2 | host-suite restatement at the R1 tip (cross-baseline, per your 1105Z template) + CI audit of Gemini's NVFP4 cells on landing | rolling |
| Gemini | NVFP4 codec host goldens + CC-gate pin cell NOW (dispatched 10:3xZ twice — if stuck, say so); then Check-wiring for R1's world>2 arms | tonight |
| agent5 | finish B-1 rows + bound row + trigger-(ii) evaluation, then R1 second-seat if agent4 stalls, else rest | immediate |

## Standing doctrine (do not re-derive; each has banked specimens)
Directs-only comm; receipts = pushed+ls-remote'd shas **tool-echoed before quoting** (the chair's own hallucinated-cite + 3 roster-false-green specimens are in the ledger — resolve, then speak); RED→GREEN triples, absence claims carry arming proofs; laws **C/D/E/F** are file-content in `docs/amd/REVIEW_LAWS_CDEF_2026-09-14.md` and apply to chair rulings too — a merge you didn't re-derive didn't happen, a gate on empty diff is vacuous, a guard without a positive-control fixture is uncertified; disk = `lsof +L1` in-row, never du-attribution for transients; one build at a time, lane worktrees only, shared checkout = chair-merge; grants route through this desk; MTP/thermal/entry-path/K-bound = do-not-spend list.

## Machinery
- **Pinger WORKS**: `~/.pi/agent/tmp/pinger_amd/` — name-based claim (`echo coordinator > coordinator_id.txt` AT TAKEOVER — Move 0, before anything), 20-min auto ticks verified (10:46/11:05/11:17 db-signed), age-guard refuses stale wakes LOUDLY. Rewrite `ping_msg.txt` FAST-FACTS every phase change; wake≠act, sweep lane tips every beat (0-ahead is the health check).
- **NVFP4-on-AMD domain is LIVE** (user directive): artifact arrives within hours; agent5 planner-of-record (plan doc pending); Gemini test-first lanes above; recipe anchors fetched and banked in the 10:3xZ dispatches; q2/q3 SIMT tiers = the AMD dialect. Do not let TP4 starve it or vice-versa: different desks, same evening.
- Ledger entry for the unowned datum: the 10:58Z 278,454,272-B KFD boot (pid 1094894) matches agent5's sweep byte-size but sits inside their claimed-dead window — tension flagged to agent5 for self-naming; next boot, sample pid+cmdline+cwd AT SIGHT.
