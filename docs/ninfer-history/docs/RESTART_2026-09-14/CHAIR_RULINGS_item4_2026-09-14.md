# Chair rulings, 2026-09-14 ~23:1xZ — issued on G-AMD-34 census (d752eb54), chair-re-verified at 26a21f6c

## RULING 1 — item 4, K-census policy: **K=16 STANDS, with a measured re-open trigger.**
Datum (both arms, chair re-derived at own seat from the banked log, matching agent4's release row
exactly): try-histogram trace-ON 410/180/148/118/78/43/22/11 and trace-OFF 350/159/124/84/52/35/17/10
— **max try = 8 in both, zero events at 9+, zero FAILOUT both arms.**
- K=8 (old) sat exactly at the observed ceiling: try=8 events occurred 10–11 times per census; a
  tail that touches the bound is a bound that will be crossed by noise. G18c's death at K=8 is
  consistent. K=16 = 2× measured max with zero observed premium — correct shape: platform bound must
  absorb the box's own distribution, and the loud death on sustained absence survives intact.
- **RE-OPEN TRIGGER (pre-declared):** any bin with max try ≥ 12 (three-quarters rule), or ANY
  death-by-retries at K=16, or the 4-card window's world=4 retry histogram (first census at
  world>2 re-derives the ceiling from zero — 4-rank publish fan-in may be a different distribution).

## RULING 2 — rb-n attribution: **world-(2) rb-n JOIN is INVALID as branch evidence; presence-witness stands.**
agent4's mechanism finding (census release row): rank1's worker runs on a per-request std::thread
(:3325) → its thread_local counter is structurally pinned to n=1 (26/26 across G18c/d/e), while
rank0's counts thread reuse — the join has zero discrimination between a clean 25/25 bin and the
bin that died at req 20. Board law: no row may cite rb-n **attribution** in either direction at
world=2. rb **presence** (lines = arrivals × ranks) remains a valid live-arming witness.
**Consequence for the window manifest (agent1):** cite rb-presence, not rb-n-join, as the
branch-parity arm until a cross-request counter exists; at world=4 the whole question is
un-measured, so the window row must NAME it as unknown rather than inherit the world-2 reading.

## RULING 3 — F-B registration: **the :724 uncomment + RING_PROPS registration ride agent4's A-4
co-land** (agent5's drafts/FB_plus_RING_PROPS_registration.patch, git apply --check clean).
Receipt demanded per agent5's own rule (chair-endorsed): **+2 in `ctest -N` generated inventory at
the merged tip** — cmake/ctest live at /home/chris/opt/cmake/bin (chair-located this hour; the
"no cmake on this host" lore is stale for PATH but the tools exist off-path). A comment-swallowed
ninfer_add_test that ships "registered" in its commit message is the born-unregistered class;
first closure receipt is the inventory line, never the diff.

## RULING 4 — D.4 cross-boot bar: datum EXISTED at census, field named.
G18d (trace ON, 25) × G18e (trace OFF, 25): all 50 responses text-sha **348e77a1222dea7f**
(reasoning+content concat, sha256-16, whole-JSON-vacuity law honored — raw JSON differs at
created:/usage: only), chair-re-hashed at seat twice. Cross-boot text-identity at world=2, greedy,
same bin family: 50/50. Gemini to consume for the cross-boot wording; the bit-exact claim is
scoped to the TEXT FIELD of greedy same-prompt serve at world=2 — it is not a general byte-identity
claim and must not be quoted as one.
