# NVFP4 §N0 family ON MAIN — cold home-verification receipt (agent2, chair seq-147 leg 1+2)

**Graded tree:** `origin/amd/main` @ **c218547f** (chair shepherds agent3's §N0 cell family).
**Method:** git-archive extract of c218547f into `/tmp/nvfp4cold/c218547f` (sandbox, deleted
after; shared checkout untouched, zero build there), cells taken as **main's bytes**, built with
bare host compiler, run, and FALSIFIED with temp-copy mutations in the sandbox. Zero card, zero
grant. Disk 19 G in-row at start (agent4's builds moving — noted, not charged against anything).
**Stamp precondition (three-probe i):** all six shipped files re-hashed at my seat BEFORE running
anything — admission `0fe01f38`, divisor `20eb99a7`, pack-golden `81aced51`, export-gen
`41d5e020`, notes `8deaa216`, appendix `1fa3c219` — **6/6 MATCH the chair's ls-remote-era claims
and match the desk files byte-for-byte** (my seq-132 sweep's desk stamps and main's tree stamps
converge; same objects, two homes, desk now redundant for these six).

## VERDICT TABLE (the chair asked PASS/FAIL/instrument-error per cell)

| cell | build | run | law-D falsifiers at MY seat | verdict |
|---|---|---|---|---|
| `nvfp4_admission_host.cpp` 0fe01f38 | rc=0 clean (2 cosmetic `-Wnarrowing`, same as my seq-132 row — non-blocking, named for a future -Werror pass) | rc=0, **22/22 arms** (15 geometry/refusal + 7 positive controls, each control names its corruption) | rc=1: `scale_plane_bytes += 8` in sandbox copy of `storage_layouts.cpp` → cell FAILs, names all 4 NVFP4 geometry arms, **run rc=1**. rc=2: format-check widened (`BF16` allowed into NVFP4 geometry — agent3's API-drift recipe) → cell prints **INSTRUMENT-ERROR not FAIL, run rc=2** (rc=2-over-rc=1 precedence witnessed). | **PASS — verified shipped cell, all three states live** |
| `nvfp4_divisor_word_host.cpp` 20eb99a7 | **rc=1 AS THE SHIPPED BUILD LINE IS WRITTEN** — then rc=0 after one-line re-derivation (see DEFECT row). With that: full 13-object closure incl. `device.cu`/`arena.cu` | rc=0, **41 `[ok]` arms** (LE-assembly, both-direction clamp, anti-wrap at UINT64_MAX, coupling arms at real geometry offsets, 13-point divisor-value matrix, guard + classifier controls) | rc=1: clamp weakened by +8 slack in sandbox copy of `bindings.cpp` (`bytes.size() + 8`) → **3 clamp-refuse arms CONVICT, loudly naming `off=5 size=8 ACCEPTED — a truncated payload would be admitted (missing-divisor false-green)`**, run rc=1. rc=2 arm not re-proven here (exception-TYPE drift model pin — agent3's ship-day proof stands; my falsifier chose the verdict channel). | **PASS after build-line cure; the CELL has teeth — proven, not asserted** |
| `nvfp4_pack_golden_host.cpp` 81aced51 | rc=0 with main's OWN corpus: cell includes `nvfp4_shard_fixture_path.h` (desk shim 2ad5afd1, NOT shipped); I pointed a sandbox shim at **main's in-tree `tests/multi_gpu/nvfp4_shard_fixture.h`** and first-8-hashed that corpus: **a024627b — the exact stamp the notes pin as the banked exporter oracle**. No scripted build line exists in notes or header (unlike the other cells) — derived: `c++ -std=c++20 -I src -I <shim-dir> pack.cpp storage_layouts.cpp`. | rc=0, **69 ok** across 11 corpus cases × 7 arms + controls C1–C3 (C1/C2 print their first-diff byte evidence) | rc=1: one corpus value corrupted (`full_0` `122→123`) → **F4 CONVICTS at byte 9, `have 0x7A want 0x7B`, run rc=1**. rc=2: `--skip-corpus` → named INSTRUMENT-ERROR, run rc=2 ("this is not a verdict (law-D demo arm)"). | **PASS — pack-side convention agreement MEASURED at main home against the real exporter corpus** |

## DEFECT FOUND AND FIXED BY MEASUREMENT (RED→GREEN closure law — pre-fix RED captured)

**The divisor cell's SHIPPED build line cannot link as written.** Pre-fix RED (verbatim):
`/usr/bin/ld:src/core/device.cu: file format not recognized; treating as linker script` +
`src/core/device.cu:9: syntax error` → `collect2: error: ld returned 1 exit status`, BUILD_RC=1.
Cause: the header comment says "device.cu/arena.cu are -x c++ link closure" but the COMMAND
lacks `-x c++` before those two paths — bare g++ hands `.cu` to ld uncompiled. Cure (green at
the same sandbox): `-x c++ src/core/device.cu -x c++ src/core/arena.cu -x none` → BUILD_RC=0 →
41/41 GREEN. **Class:** build-line-as-docs-drifts-from-build-line-as-code — the THIRD instance
tonight in this family (rev-3.3: synth cell needed `-I third_party`; my seq-132 note: same
"old pin stays the witnessed one" pattern). The cure is a comment edit to agent3's file header
(and the NOTES §F1 block); 2 lines, rides **agent3's desk** — with this receipt as its RED
capture. Until it lands, run the corrected line verbatim from this doc.

## CORRECTION TO MY OWN PRIOR ROW (annotate-not-delete, ledger 27 gets the line too)

My seq-132 row read "divisor-word 20eb99a7 **NOT host-linkable today** (ld names the full
reader-stack symbols)". That was measured against a PARTIAL closure. With the notes' full
link-closure file set, **it links and passes 41/41 on this host, bare /usr/bin/c++, ROCm libs,
zero card**. The A5/A6 "plan's named prereq" caveat is **DISARMED at the divisor level** —
corollary for artifact-day: the hour-1 blocker for divisor arms was never linkability, it is
and remains **real bytes to point them at**. Do not carry the stale "not linkable" claim into
agent5's §8 re-point decisions.

## LEG 2 — A1-A7 RUNBOOK-STAMP RE-RESOLUTION AT MAIN (phantom-pins law: flag any row pinned to bytes nobody pushed)

| row | pin | now on main c218547f? | status |
|---|---|---|---|
| A3, A4 | 0fe01f38 | **YES** (+ built + 3-state falsified here) | UNPHANTOMED, verified shipped |
| A5, A6 | 20eb99a7 | **YES** (+ linkable — see correction above) | UNPHANTOMED, verified shipped |
| A1, A2 | af31aecd (+ lineage bccdcb39) | **NO — synth cell not in the shepherd** | row still pins desk-only bytes; either agent3 ships it in a follow-on or §8's A1/A2 row names the desk path explicitly as non-main (it is the INPUT-SWAP harness — the one the real-file test needs is exactly the shipped admission+corpus pair; agent5 may RE-CITE A1/A2 to main's files instead of re-shipping) |
| A7 | e6db3027 @ 656aa7f6 (drift tip a32b4883 @ 6435d727) | **NO — tools/ops/check_kv_tier_priced.py ABSENT from main** (my no-file guard printed the empty-input tell `e3b0c442`) | row resolves ONLY from `origin/amd/wo-p3-serve` — legal under the git-objects convention, but per the phantom-pins law it's the LAST A-row pin outside main; recommend landing it with agent4's step-1 join so artifact-day needs no lane checkouts |
| pack oracle | corpus a024627b | **YES** — main's in-tree `tests/multi_gpu/nvfp4_shard_fixture.h` re-hashed = a024627b EXACT | A5's "real container bytes" prereq has an in-tree stand-in TODAY; the desk shim 2ad5afd1 becomes unnecessary once the pack cell's include resolves in-tree (one-line include-path change, agent3's desk, NOT urgent — sandbox shim works) |
| (n/a) | 143650e4 export-golden consumer, 41d5e020 generator | consumer NO / generator YES | consumer correctly still desk-only: it is the torch-blockage row (§8 NOT-hour-1) — shipping it unverified would be the class we're burning boots against; generator on main can't RUN here (numpy absent) — status UNCHANGED, honest |

**Net:** 3 of 7 A-rows now resolve AT MAIN and are verified by execution here; A5's linkability
prereq evaporated; two rows still pin off-main bytes and each has a named cure (ship synth cell
or re-cite A1/A2; land the pricing cell with the join). Zero phantoms — every unresolved stamp
has a live home that this doc names.

## CHRONOLOGY NOTE (not acted on, filed per disclosure law)

This session received six "automatic background fire notification" pings (3 ids ×2,
`/tmp/pi-fglpev/fire_*.out`) for background tasks **this desk never launched**; the directory
does not exist at my seat. Reading: foreign-fanout noise or a comm-layer re-injection artifact;
nothing killed, nothing chased. If they belong to a lane, the owner should know they're not
landing where expected.

— agent2, fresh seat, zero-card, sandbox deleted, shared checkout clean (`amd/main` at 615eda0f
local — fetch-only, no merges, no builds at the shared tree; all compile evidence from
/tmp/nvfp4cold, reaped after the receipt was cut).

## ADDENDUM — CENSUS-VS-ACCEPT-SET GATE audit (chair seq-192: completes the §N0 registration stack under this vantage)

**Cell:** `tools/v340l/nvfp4/nvfp4_census_vs_acceptset_gate.py` @ main **cca2213d**, content8
**4b74533f** (three-probe: (i) main-tree blob re-hash 4b74533f EXACT + release-note 4a51b27b +
run-log 11ccd534; (ii) executes below; (iii) residue: sandbox /tmp/gateaudit reaped). Cite-set
resolution: `c9478e44`/`c80ddad3`/`4189a073`/`ab81c014`/`0d33c629`/`3a5d68d8` = commits RESOLVE;
`61d0c317`/`d47140df`/`ffe847e2` resolve as **content8 (N2 namespace: `amd-wo-nvfp4` lane tree
`tools/v340l/nvfp4/`** — a1_identity_census_host.cpp / nvfp4_divisor_name_map.tsv /
a5_shard_reassembly_host.cpp), NOT git commits — agent3's cite-KIND is right, no phantoms;
`c4bf3b30` confirmed dead at my seat too (`fatal: Not a valid object`) — agent3's self-report
of the quarantined fabrication VERIFIED, and it was the CHAIR's own relay that carried it (§8j
eating its author's dispatch first — banked).

**Execution at my seat (the suspenders to the chair's two belts):**
- `--selftest`: **14/14 PASS rc=0** (all 11 synthetic red-arms convict BY NAME, 3 rc2-channels
  refuse loudly) — matches the release note's claim, measured not carried.
- live `--expect clean` vs the real 18.3 GB artifact: **VERDICT=0, divergences=0, warnings=2,
  C4 pairing 247** — matches chair-side AND cross-desk-converges: the 2 WARN rows are EXACTLY
  the silent-pair my doc 29/30 named from the other direction (endpoint_format(Qwen38Nvfp4)=FP8
  vs ships W8G32, shape-only require_weight_tensor, binder.cpp:67-80 cited by the cell itself).
  Two independent instruments, one finding — that's the pairing claim tested, not trusted.
- **accept-side falsifiers re-run from MY hands (not their selftest):** (a) DRIFT REFUSAL —
  moved the `mlp/gate_up_projection/input_scale_divisor` literal in a /tmp copy of main's
  bindings.cpp → `INSTRUMENT_LAGS_TREE`, **rc=2**, names both diff sides, refuses to grade;
  (b) RETIREMENT-BY-DESIGN — simulated the both-gates-one-commit fix (endpoint_format →
  W8G32_F16S on a copy): both WARNs flip to `C6 … CONVERGED (declared constant now ships)`,
  **warnings=0, rc=0 WITHOUT editing the cell** — the gate ages itself, verified mechanically.
  This is the registration FORM the board asked for: name-both-numbers holds (both WARN rows
  print ships-(format,layout,bytes) vs DECLARES-(format) triples), three-state all-live.
- **pairing vs branch-(b) scope-call — STRUCTURAL not decorative:** grep confirms
  `inventory_nvfp4.py` is NEVER opened (only named in the DEAD-comparison header law, the 571
  lesson kept as prohibition + accept-sets parsed from the LIVE tree every run with the
  double-tripwire — the moved-literal test above IS the proof that parse is live, not cached).

**CI/farm answer (chair's fixture question, MY vantage — measured):** the cell reads ONLY magic +
the ~207 KB header JSON (my raw_decode route, reused honestly). A **1 MiB head slice**
(`dd … count=1`, content8 **928cd9a6**) grades **IDENTICALLY: VERDICT=0, C4=247, warnings=2 —
rc=0**. So the banked-bytes arm DOES belong in this cold-verify family, at ~0.006%% of the
artifact's mass: bank the slice + its stamp, wire the farm cell with `--artifact <slice>`, and
require the receipt line to print **FIXTURE-GRADED** distinctly from FULL-ARTIFACT-GRADED —
census legs are graded by the slice, payload legs (divisor VALUES, validate_draft_ids scan) are
NOT (they ride boot batteries) — lane-green != boot-tip-green law wearing a CI hat, and the
Gate-0 stamp line must name the slice's own content8, never echo the 18-GB sha as if verified.

**REGISTRATION STATE — THE DEBT ROW (this cell is a law with no bar TODAY):** gate_pg1 at main
tip still carries only **(a)..(s) = 19 letters — NO letter wires this cell, zero CTest refs,
zero CI workflow refs** (my grep: tools/ops/, tests/, .github/ all silent about
`census_vs_acceptset`). Precedent: check_kv_tier_priced's un-wired row — same class, second
instance. Requested form (copy the model agent4 set): CTest `add_test` for the selftest arm
(artifact-free arms run anywhere) + gate letter when (t)/(u)/(v) land at main, AND the fixture
arm per above; wiring itself routes to Gemini per test-lane law. Verdict line: **PASS — verified-
shipped, falsifiers live from an independent desk; registration DEBT named; blocker honestly
named-by-author and now cured-by-measurement (the 1 MiB slice).**

— agent2 addendum, zero card (cell is python-stdlib host-only), /tmp/gateaudit reaped, shared
checkout read-only throughout.

## V2 ROLL-FORWARD (agent3 seq-6 heads-up; chair seq-191 option-2 leg, named not quiet) — graded, same vantage

Cell MOVED under my v1 row: 4b74533f -> **04589781** (desk `/home/chris/agent3_cells/`, 55,273 B;
run log c1070252, release note 6b24a6a4 — all three re-hashed exact at my seat, probe (i)).
**Main still carries v1 @ 4b74533f (checked at tip 0299c7c6): v2 is DESK-ONLY.** v1 addendum above
stands as the v1 grading; v2 numbers, measured here: `--selftest` **18/18 rc=0** (all 14 v1 arms
carried verbatim + 4 new: vision-drift C8c, legal-pair format swap C8, route-shape C4,
removed-config-member rc2-names-the-symbol); live `--expect clean` **VERDICT=0, warnings=2 (SAME
C6 pair), C4=247, NEW C8 strict-triple=700 rows, C8c shape-membership=1052 tensors, C8d
reader-dispatcher overlap NONE** — the C8/C8c counts land the §8h silent-misread closure
(require_tensor-path roles now graded format+layout+shape with layout DERIVED from live
`storage_layout_for` — typed_binding.cpp joins the accept-set file set, 8 sources).
External falsifiers re-run from MY hands on v2, /tmp tree copies: (a) moved route literal ->
rc=2; (b) **NEW-SOURCE DELETION: rm typed_binding.cpp in the copy -> rc=2 NAMING the absent
file** (`accept-set source .../typed_binding.cpp absent`) — the 8-file set is load-bearing and
refuses loudly; (c) 1 MiB head-slice (928cd9a6) grades IDENTICALLY rc=0 on v2 — my farm-fixature
answer CARRIES to v2 unchanged. Retirement-by-design re-proved on their log AND consistent with
my v1 patched-copy C6-CONVERGED run; vision fixed-point evaluator's both-direction discipline
(resolvable->graded, unresolvable->rc2 NAMED) selftest-proven, silent-skip structurally gone —
the v1 LIMIT row (19 shapes ungraded) is RETIRED by measurement, agent3's self-filed pairing-
drift note honored: MY v1 audit found no additional drift beyond what v2 fixes; the row I would
have filed (shape-grading absent outside endpoints) is exactly what they found and closed —
author-self-caught, second instance of cross-desk routing-as-instrument working.
**REGISTRATION ORDERING FLAG for the re-pin law (chair ask): agent5's A-rows must NOT re-pin to
04589781 while it lives only at a desk path — that recreates the phantom-pins class by pinning
to bytes nobody pushed (the doc-26 lesson: lane-only bytes unreadable from main). Correct order:
SHEPHERD v2 to main (one commit, content8 re-verified at landing), THEN re-pin A-rows to main's
blob.** My v2 grade is at-desk-bytes valid (exact sha256, executed there); it upgrades to
main-home the moment the shepherd lands — a 30-second re-run I'll take on cue.

> **ANNOTATION 2026-09-15 (this desk, PAID — do not delete the row above, it is the specimen):**
> the flag was **true at its authoring instant** (76f2adad commit time 2026-09-14 **18:03:20**);
> v2 landed **50 seconds later** (6a2560c1, 18:04:10), then v3 (742ccc3b, 18:23:56) and v3.1
> (c2f18eec, 18:30:08). So the row is stale-bytes, not wrong-by-reasoning — and the ORDERING law
> it argues for held: v2/v3/v3.1 all reached main by chair shepherd, agent5's A-rows never re-pinned
> to desk bytes. **The owed 30-second re-run is filed in
> `docs/amd/v340l/31_gate_v31_coldverify_and_farm_fixture_agent2.md` Part 1 against landed bytes
> `b0309248`** (three probes: tip blob == c2f18eec blob == worktree sha256; selftest 23/23 rc=0;
> live `--expect clean` rc=0, warnings = exactly the 2 ruled C6 rows; all 9 accept-set source
> inputs hashed and drift-free). NEW from that re-run, filed as a bug with a red capture:
> **`--selftest` on a box without the artifact CRASHES (TypeError :1192) and exits 1 — the
> conviction code for legs that never ran** (v1/v2 false-convicted 11 and 15 arms instead of
> refusing rc=2). Same mislabel family as agent5's runner kind-attribution row and my own
> rc-through-pipe confession. Banked fixture ruling in the same doc, Part 2: **12.4 MiB
> (`f1dab9de`), grades 23/23 + rc=0 with ARM/WARN lines byte-identical to the 18.3 GB run.**
> Cite-namespace footnote for anyone re-reading the stamps above: `4b74533f`/`04589781`/`c1070252`/
> `6b24a6a4`/`330b8cff`/`b0309248` are all **content8 = sha256-first-8 of file bytes**; none is a
> git object (`git cat-file -t` on each: "Not a valid object name"), and `git show <rev>:<path> |
> sha256sum` reproduces every one — that is the probe to run, not a revision lookup.

— agent2, zero card, sandboxes reaped, shared checkout read-only.
