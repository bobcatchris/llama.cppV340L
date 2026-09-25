# FLIP-DAY SCRIPT — PREDICATE-2 + check (q) + donor-cell parity audit (agent1, staged 2026-09-14)

Run at the merged-A-4 tip in the shared checkout (reads only; no build here — law). The chair's
announcement quotes the outputs verbatim into the window manifest.

    cd /home/chris/dual_5060_ti_ninfer && git fetch origin amd/main && git pull --ff-only
    T=$(git rev-parse --short origin/amd/main); echo "FLIP TIP=$T"

    # F1 — GATE 1 post-flip state is SHAPE-DEPENDENT, so record it, don't pre-adjudicate it:
    grep -nA3 "if (tp_world" src/runtime/tp2/tp_engine.cpp | head -8        # paste output in row
    # Either "no tp_world guard remains" (A-4 fully relaxed) OR a guard whose throw is replaced by real
    # world>2 construction is the flip. PREDICATE-1 printing GUARD ABSENT OR INERT is EXPECTED here —
    # the safety property is that GATE 2 (F2/F3) closed TOGETHER with it: guard gone + routing silent =
    # the silent-null first-light this runbook exists to prevent. A bare deletion with F2/F3 still RED =
    # NOT a flip, flag loudly and do not book.

    # F2 — my decision-shaped checker (independent parser):
    python3 tools/v340l/t3_gate2_silent_nullpath_check.py .; echo "t3 rc=$?"          # EXPECT GREEN rc=0
    python3 tools/v340l/t3_gate2_silent_nullpath_check.py --selftest; echo "st rc=$?" # EXPECT PASS rc=0

    # F3 — Gemini's routing checker + the enforce flip:
    python3 tools/ops/check_tpgroup_routing.py; echo "q rc=$?"                        # EXPECT rc=0 (GREEN cert)
    grep -n "ENFORCE_A4_ROUTING" tools/ops/gate_pg1_whitelist.sh                       # flip = default 1 at :~834
    ENFORCE_A4_ROUTING=0 bash tools/ops/gate_pg1_whitelist.sh >/dev/null 2>&1; echo "PG-1 rc=$?"  # EXPECT 0

    # F4 — PARITY AUDIT (the donor-cell leg): F2 and F3 parse the SAME function with INDEPENDENT code.
    # Verdict-parity across two independent parsers is the stronger gate; a divergence is a datum about
    # a parser, not about A-4 — name which parser and re-read before citing either:
    #   GREEN/GREEN = certified; RED/GREEN or GREEN/RED = STOP, report divergence to chair + Gemini, do
    #   NOT book the window on the optimistic one.

    # F5 — determinism suite still stands at the flip tip (my empty-guard now promoted-citatable):
    bash tools/smoke/diag/determinism_empty_guard_check.sh; echo "eg rc=$?"           # EXPECT GREEN rc=0
    python3 tools/ops/check_determinism_cells.py --enforce-promoted >/dev/null 2>&1; echo "r rc=$?"  # EXPECT 0

    # F6 — anti-resurrection leg vs A-4's tp_engine.cpp edit (the §1b hazard):
    bash tools/ops/check_anti_resurrection.sh 2>&1 | tail -6
    # If leg 2 FAILS on A-4's removed guard line = the PREDICTED false-RED class (runbook §1b): it is
    # overruled ONLY by the named-constants predicate (leg 1 form), chair to cite both outputs; if
    # AGENTS.md wording was amended/exception registered pre-merge, cite the register line. A raw leg-2
    # FAIL waved without either receipt is the law-decay this section was written to catch.

    # F7 — window-manifest rows owed (RULING 2 form, from docs/amd/RESTART_2026-09-14/CHAIR_RULINGS_item4_2026-09-14.md):
    #   cite rb-PRESENCE (lines = arrivals x ranks, 52=26x2 class), NOT rb-n JOIN; NAME world=4
    #   attribution unmeasured; K=16 re-open watch (max try>=12 / death at K=16 / world=4 distribution);
    #   ctest -N +2 receipt for F-B+RING_PROPS at merged tip (chair's inventory-line-is-the-receipt rule);
    #   dev0 used-bytes baseline line in the announcement (chair mitigation (a)); boot bin = agent4's
    #   banked A-4-tip sha16 from /home/chris/artifacts_bin (ONE-BUILD law), bank sha re-verified at
    #   pre-flight not assumed (agent4's own census-manifest discipline, adopted).
