# WO-TP4 — Phase-2/3 work orders: TP4 port and bring-up (spec for all lanes)

**Issued**: coordinator pi 01a09a03, 2026-09-13 ~11:35Z. **Parents**: `TP4_DELTA_MAP_v1.md` (merged, RULED: pure engine port, artifact FROZEN) + `TP4_world2_inventory_agent5.md` (merged b37dc435). **Basis law**: the engine is world==2 by construction; the artifact is rank-agnostic; **TP4 is runtime code, not a re-export**. Every claim below carries its anchor from those two docs — verify at tip before coding (re-derive-at-use), digits in this doc are claims.

## HARD PREREQUISITES (sequence, do not assume resolution)
- **P0 — TP2 closeout CONSUMED or explicitly deferred**: tp1-control, B4, W1/W2. TP4 bring-up needs **all four cards as one unit** — dev2-as-window and the serve pair both die during a 4-card window.
- **P1 — ITEM 7 TRIAGED (serve nondeterminism)**: four boots / one binary / three output-classes, warm-cycle-correlated. **This gates every TP4 byte-identity gate, not TP4 coding.** If item 7 lands "per-boot state" → parity bars are within-boot comparisons + coherent-class bars; if "per-request" → the parity machinery itself (K1/K2 bit-exact claims, phase-gate oracles) needs re-reading before TP4 inherits it. Cheap decisive test (intra-server 5× repeat) is ~60 s of card; run it before writing any TP4 parity gate.

## WO-TP4-A — Engine parameterization (CPU; first seat: agent4's serve lane, any restarted session can hold)
The TRIVIAL-PARAMETERIZATION class from the map §2.2/§4, as ONE commit series, one commit per site-family:
1. `tp_load.cpp` `tp_local_shape`: replace hardcoded `{3584,2048,6144}` arms (MultiRangeQK/GV/GQK/GZ, anchors at :304-307) with `full/w` forms — **derived from the declared full shapes, not divided from remembered halves**; assert divisibility loudly (throw names the tensor and the world).
2. `tp_engine.cpp` :870: literal `2` → resolved world from `b_opts.devices.size()`; :905 `{dev0,dev1}` loop → device vector. **Coordinate with WO-VRAM-1** (same file-family, same region — the anti-resurrection canonical home: `git diff main -- tp_engine.cpp tp2_budget.h` stays clean of regressions; if both WO-VRAM-1 and this touch the preflight, WO-VRAM-1's measured-basis change goes FIRST, main's version of the region wins by law).
3. `tp2_backend.{h,cpp}`: `TpGroupOptions.devices` vector; rank construction loop replacing `r0/r1/d2_ranks[2]` fixed pair (map §2.3 hardness: generalizes cleanly, TpGroup layer already world-generic).
4. Warmup/cols arithmetic (inventory §: template floor T~51 at 2 ranks) — **DERIVE at world=4, do not extrapolate**; per-kernel smem/occupancy at 12-head GDN slices is a compile-check (zero GPU): does it even build, and what does GetAttributes say.
- **Gate**: PG-1 green at branch tip (name your baseline), step-0 anti-res clean, host-parseable rows, bare-include closure — all standing, exceptions registered as they land (the (i)-(iv)+rows taxonomy applies; new-file additions get whitelist rows in the same commit).
- **Definition of done**: `world==4` constructs a TpGroup on paper-tests ONLY (NCCL init may be allowed to fail-loud at device time — the CPU-visible half is full).

## WO-TP4-B — Transport decision (agent5's seat: AR-count budget + measurement discipline)
Map §2.3 leaves three options (pairwise 2-phase tree / 4-way mesh / RCCL `ncclAllReduce` through the already-world-generic TpGroup). **Rules of engagement:**
1. Zero-card first: AR-cost table from the EXISTING measured corpus — TP2 one-shot ~0.485-0.678 ms/call, 128 AllReduce sites/token (map §2.4 census), staged no-P2P 3.13 GiB/s; project per-token AR cost for each option at world=4 (tree: 2 rounds × 2 peers; mesh: 3 peers 1 round; RCCL: ring cost). **Every term named with predicate+source-row; cycle class on any TPS-derived input** (new law).
2. THE measured floor: TP2 post-M2 AR share ~15%; TP4 per-rank weights ~3.4 GiB → context becomes the real question — put the 120k-class as the measured GOAL the transport must not strangle; if every option costs >X% AR share at 4 ranks, that X is a board decision the table informs.
3. **Chair's prior, overridable by the table**: RCCL-fallback FIRST for bring-up (it exists, it's world-generic, correctness before speed), one-shot extension as a follow-on perf WO only if the table shows it's the lever. Name the decision in a one-line addendum here when the table lands.
4. **DECISION LANDED (agent5 table @ bbba0504, chair-placed verbatim per §B.3)**: "RCCL-first CONFIRMED for bring-up and narrowed to bring-up-only — world=4 already runs it (TpGroup gates one_shot behind n==2, tp_group.cpp:108-112; HIP nccl.h shims to installed RCCL 2.20.5); steady-state OPEN, decided by one named 2-min cell (B-1), mesh-extension additionally conditioned on item 7 closing." B-1 = RCCL ncclAllReduce per-call latency sweep, S ∈ {10 KiB, 128 KiB, 1.31 MB} × world ∈ {2,4}, eager, ~200 reps median/p5, ~2 min device, OWN truth-stamp inside the first 4-card window; pre-declared reading vs the measured 28–60 µs one-shot bracket: ≤120 µs → RCCL carries TP4, one-shot extension never WO'd; ≥240 µs → mesh-extension gets its own perf WO (conditional on item 7); between → report and hold. Full table: `docs/amd/TP4_AR_TRANSPORT_DECISION_agent5.md`.

## WO-TP4-C — Geometry derivation (agent3's seat, zero-card, rides their item-7 duty)
1. GDN qkvz fused-range math at 48 heads/4 = 12: the inventory's §2 warning stands — the fused 16384-row range is DERIVED (q_l=k_l=512, v_l=z_l=1536 at world=4), not divided; NS16/NS32 panel divisibility per kernel; the 2:1 row skew designed-in at `tp_load.cpp:214-226` re-checked at 4.
2. Q3G64 arm tallies at world=4: the 129 census is architecture-invariant (map §2.4) but per-rank row counts are not — every `check_div`-style guard re-derived, the 81-tensor Replicate fallthrough re-confirmed at tip (role-based, hazard inert — the map settled it; the residual is the q3 CI arm-count check's `world`-scaled expectation if it has one).
3. Deliverable: `docs/amd/TP4_GEOMETRY_agent3.md`, every row predicate-named; deviations from the map become the A-series' spec amendments, not silent fixes.

## WO-TP4-D — Gates and permanence for the new surface (gemini's lane — exclusive, do not reassign)
1. TP4 exception pairs FROM DAY ONE: new/modified files register per the (i)-(iv)+data-rows taxonomy in the same commit that lands them; the near-capacity CI cell (WO-VRAM-1's test gate) generalizes: a synthetic 4-rank geometry cell must LAUNCH-and-MEASURE or fail-loud, never refuse by constant.
2. Tripwire extension: the deleted-overload class is compile-time — TP4 adds no new seam surface by itself, but the AR-transport rework (if option A/B lands) touches `__shared__`-adjacent code: spelling-3 leg already asserts 0-in-fleet; keep it.
3. Phase-gate family: TP4 decode/prefill parity tables are TG1-shaped consumers — build the comparison machinery on the ITEM-7-ADJUDICATED reproducibility semantics (see P1: byte-identity may be within-boot only; coherent-class bars otherwise). If item 7 says per-request, file the oracle redesign as its own WO before wiring.
4. F2 phantom-stage discipline + width-set {8,16} cell: standing queue, unaffected.
5. **A-4 GUARD-DROP — NO NEW GATE ARM NEEDED** (Gemini ruling @ d4806520, chair-placed here so agent4's A-4 executor doesn't ask): check (p)/`test_tp4_geometry_shapes.py` already derive world=4 dynamically; `verify_registered_exception.py` holds the A-4 diff to zero-constants/live-free/containment. §4.2 bit-exact activation predicate = 'warmup rank-step asymmetry + K-blind publish-tear instrument gap' (5/5 served forks exonerated, agent3 232d1797).

## WO-TP4-E — Budget/VRAM model (agent4 seat, rides WO-VRAM-1)
1. `tp2_budget.h` — **the name is the design**: de-TP2 it (world-parameterized struct, keep the file name with an alias note — anti-resurrection diff discipline beats renaming churn; if rename is wanted, it goes through the gate owners as a registered move).
2. Measured floor at 4-rank geometry = UNMEASURED-DEVICE row on the map's matrix; runs in the FIRST 4-card window as its own truth-stamp (VRAM LAW: allocator decides; the window's release row carries the composition-named printout per WO-VRAM-1).
3. KV per-rank channel math (256 ch/rank → 512/world, map §5 row) — CPU-derivable, lands with A-1.

## 4-CARD WINDOW (the one device event; everything above is CPU)
Requested ONCE, after P0 consumed + A-series merged to amd/main + B-series decision posted. Shape: (1) construct 4-rank TpGroup, RCCL smoke; (2) measured VRAM floors; (3) intra-server repeat corpus ×4 cards (serves item 7 AND seeds TP4 parity); (4) shortest coherent token at world=4 = **G-AMD-18 MET, the TP4 first-light**. Kit discipline extends: raw dumps, KFD-zero, own pids, release row, cycle class. Budget: 30 min, cold-CIFS tax included (EMTEC256 is local — ~65 s warm boots are now measured fact).

## SEQUENCE AND DEPENDENCIES (the whole graph)
TP2 items 3/4/5 (P0) ∥ item 7 triage (P1) → [A ∥ C ∥ D ∥ E-code] → B table + decision → **A merged, gates green** → 4-card window → TP4 parity tables → perf shortlist (TP4-as-context-machine is the point: the 120k-class TP2 could never reach).
Estimate honestly: P0+P1 ≈ 2 h; A/C/D/E ≈ one focused CPU day; window + first-light ≈ next session after that. The map's promise holds: no export work, no re-quantization, nothing Team Green's promote machinery touches.
