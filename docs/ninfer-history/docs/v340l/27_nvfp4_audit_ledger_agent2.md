# NVFP4 AUDIT LEDGER — agent2, standing shift (chair seq-132 leg-3: registration / blast / letter-count vantage on every NVFP4 landing)

Opened 2026-09-14T19:5xZ. Rule of the desk: **nothing NVFP4 is trusted before this ledger says
where it is registered (gate letter / CTest / standalone), what its blast surface is, and what
count was actually measured — at the bytes, by execution, this beat.** Every row: content8 by
sha256 over the exact file, verdict by running it.

## LANDINGS AUDITED (19:4x-19:5xZ, this session)

| instrument | owner | content8 | where it's at | registered? | verdict at my seat (measured) |
|---|---|---|---|---|---|
| `nvfp4_admission_host.cpp` (§N0 cell) | agent3 | **0fe01f38** (267 ln — matches every ledger cite, re-derived) | UNCOMMITTED `/home/chris/agent3_cells/` BY DESIGN (agent5 REV-2 record) | **NO gate letter, NO CTest** — pure standalone | **GREEN rc=0: 22/22 arms** (15 geometry+refusal + 7 positive controls, all corruptions caught) — links bare-host vs `storage_layouts.cpp` ONLY (one TU; the §N0-linkability question ANSWERED for this cell), g++ -std=c++20, zero HIP. Cosmetic: 2 `-Wnarrowing` warnings (u64→u32 brace-init in the positive-control table) — non-blocking, named so a future -Werror pass doesn't call it a surprise |
| `nvfp4_divisor_word_host.cpp` | agent3 | 20eb99a7 (375 ln) | same uncommitted bank | none | **NOT host-linkable today**: compiles, then ld names 4+ undefined artifact-stack symbols (`bind_tensor`, `bind_device_tensor`, `Binder::finish/payload/has_tensor/materialize_on_device`, `MaterializedArtifact::device_arena/device_data`) — the FULL reader stack, HIP-side. This is the measured face of the plan's named prerequisite ('nvfp4_shard_image host-linkability unverified'): the divisor-word arm rides the artifact-day link, the admission arm already proved it doesn't. Rides: whoever wires §8 A-rows to CI must either grow a host-linkable binder slice or run it in the HIP build |
| `nvfp4_export_golden_host.cpp` (consumer) | agent3 | 143650e4 (241 ln) | same | none | **three-state CORRECT as designed**: links vs storage_layouts only; no fixtures → rc=2 INSTRUMENT-ERROR naming the reason ('fixture-absent is NOT a verdict'); given a dir without manifest.json → same refusal, different sentence. Law-D compliant from day one |
| `gen_nvfp4_export_golden.py` (oracle generator) | agent3 | 41d5e020 | same | n/a | **BLOCKED AT THIS DESK — honestly, not secretly: `ModuleNotFoundError: numpy`** on this host's python3. The torch-host blockage the plan keeps OUT of hour-1 (§8 NOT-hour-1 rows) — my audit adds: it also blocks the consumer's corpus HERE, so the export-golden pair cannot go gate-green on this machine until a torch/numpy host runs the generator or the synth harness (ec45cfde) supplies the corpus. Naming it so nobody 'fixes' the consumer to pass without a corpus |
| `check_kv_tier_priced.py` | agent4 | e6db3027 @ 656aa7f6 (p3-serve) | in-tree at b82e4f45 | **NOT wired**: zero refs in gate_pg1_whitelist.sh, zero in any CI workflow — the generator's docstrings cite it as the law but nothing MACHINICALLY grades it at merge | GREEN standalone (audited 1855Z beat: every tier explicitly priced, BF16-default reachable only by its naming tier) — REGISTRATION DEBT ROW: it's a law with no bar; when (v) proved that un-on-the-bar cells rot, the same finding applies here. Audit ask: gate letter (w) slot or --falsify leg inside (v) at next substantive landing |
| `check_whitelist_arm_parity.py` | agent4 | (in b82e4f45 tree) | in-tree | **YES — wired at (v) legs 2+3 (falsify + live)** | GREEN both directions (8-arm plant selftest incl halfarmed/census_new/census_shrank) — the model registration form: cell + falsifier + live leg, all in-gate |
| `gen_tier_shape_table.py --check` (drift proof) | agent4 | in-tree | in-tree | **YES — (v) leg 4** | GREEN at 6d4a5c08 (table byte-identical to generator output). Cosmetic debt: `--check` lacks a `--root` flag (gate's `\|\|` fallback carries it); fix direction = ADD the flag, don't delete the fallback |
| role-geometry × world-legality tables | agent1 | — | **NOT YET LANDED** (chair seq-131 pivot is minutes old; §8b coordination contract exists in agent5's plan @ 0dae91d4) | pre-registration | nothing to grade; when they land: same battery — content8, linkability, three-state, registration slot named. Their input edges (§N-later order, generator NVFP4-row flip) are three-party by contract — my audit flags any TWO-party pre-bake |
| artifact-day runbook §8 | agent5 | plan @ 0dae91d4 `docs/amd/NVFP4_AMD_PLAN_agent5.md` | in-tree on amd-wo-nvfp4 | doc, not cell | VERIFIED as written: A1–A7 every row pre-pinned to a SHIPPED instrument with rc contract; two NOT-hour-1 rows explicitly named (torch blockage + card grants) — burn-down honest, matches my measurements above line for line (admission=hour-1-ready; divisor-word/export-golden correctly NOT-hour-1) |

## BLAST + REGISTERED-BLACKBOARD STATE AT THE LIVE TIP (b82e4f45)

- Gate letters at-tree: **22 blocks (a)..(v)** — (v) prints (header :953). Registration FORM, as
  the chair and agent5 asked for on record: **double** — in-gate block (4 legs) AND CTest
  (`tests/CMakeLists.txt:60 add_test(ninfer_tier_shape_table_test)`): gate catches lane-merge
  regressions, CTest catches farm/cmake-door regressions. That's the form the kv_tier cell should copy.
- (v)'s **ETXTBSY fallback false-RED** (my 1940Z finding): unchanged at b82e — the live tip still
  cannot pass (v)'s gate leg without the prebuilt binary path; 3-line cure rides the author's desk.
- Roster debt unchanged: (a) 8 files / (b) 1 addition — same gate short-circuit; graded via
  continue-harness at the tree-equal 6d4a5c08 fire.
- Host serve suite 9 at b82e4f45: **9 passed / 0 skipped / 0 failed / 0 not-reached (four-count,
  rc=0)** — caveat stated: linked against p3-serve lane archives at that worktree (serve-layer TUs;
  the table-step touched none of the nine tests' code paths — zero-diff named).
- df IN-ROW 23G/80%; census/attach/twins/anti-res: carried by PROVEN TREE-EQUALITY
  b82e4f45 == 6d4a5c08 on src/tests/tools/include subtrees (all four MATCH by git tree-hash) + the
  1940Z receipt's rows; re-fired fresh on any new sha.

## STANDING WATCH-ROWS (this desk, every beat)

1. check_kv_tier_priced wiring → letter (w) or (v)-leg-5; audit on landing.
2. (v) ETXTBSY cure commit → GREEN confirmation (my /tmp probe = ready falsifier, both directions).
3. agent1 tables land → full battery, three-party contract checked.
4. Artifact-day A-rows → each instrument re-run at its A-row named shape; NOT-hour-1 rows stay NOT unless the plan itself changes.
5. Gemini silence ledger: last held timestamps 10:3xZ ×2 dispatches, 11:5xZ door, ~12:1xZ & 12:2xZ checks, 13:15Z fire-time recheck, 17:4x/18:5x/19:4xZ this session's checks — still zero traffic; CI wiring (their dispatched leg) remains unlanded, which is WHY rows 1 and the un-wired cells above read as debt rather than defect-by-someone-else.

## agent1's tp4_arming_battery.sh — received, WEAPONS-REVIEWED, COLD-VERIFIED (seq-12 handoff, 19:4xZ)

Reviewed before trusting (arming-proof-is-an-arm, applied outward): `set -uo pipefail`,
`mktemp -d /tmp/a1_battery_XXXX` + EXIT-trap reap, git-archive extract into scratch (never a live
worktree, never a build, zero device), no self-invocation anywhere in 20 KB of script — depth
bounded by construction, no recursion to starve. rc=2 paths tested at my seat: unresolvable ref
(`6d4a5c08:src` → NOT-OBSERVABLE + exit 2, refuses to grade a subtree as a tree), no-arg usage,
failed archive each named. rc=0 at 6d4a5c08 AND at live tip b82e4f45 (2.5 s wall — this is the
seconds-each form my witness-only TP4 role needs). **Teeth proven, not taken:** at old tip
c5ab9226 it prints 4 honest FAILs (world-free head ladder, resurrected pair-literal barriers
:1626/:3591/:5224, materializer world-laundering, tp_w ungrouped) + exit 1 — the battery convicts
pre-fix trees, so its greens mean something. The two reading rules ADOPTED VERBATIM into this
desk's vocabulary: (1) rc=2 = a leg didn't run — never a pass, never a remembered fill-in;
(2) `[agent1-lane (NOT shipped by this tree)]`/WARN = read-at-bytes witness, NOT a registered-
suite claim — say exactly that; the WARN is evidence, deleting it to read cleaner is the falsifier-
gaming class. Status: THIS is the procedure for attempt-8's boot-sha grading; my parallel battery
(gate letters + twins + census + host-suite-9) stays the suite-continuity home — the two
instruments OVERLAP on PRED-B/T1/T3 legs and agree at three tips tested; where they disagree,
both run, the disagreement gets a row.

## FRESH-SEAT CONTINUITY BEAT 2026-09-14T20:2xZ (chair re-brief; legs 1-3 all closed, zero-card)

**LEG 1 — doc-26 incident row: LANDED, ticked.** 8120be3b (ancestor of live amd/main tip
615eda0f) carries `docs/amd/v340l/26_ring_guard_twin_interop_2026-09-14.md` — blob
**4b0132e8**, and it is BYTE-IDENTICAL at main AND at my lane tip b9c80be1 (three-probe
`(i)` via `git rev-parse <ref>:<path>` on both refs). Rows present in landed bytes: recursion
incident (~15,900 levels / ~17 GiB, :285-296), RE-PIN VERIFICATION (:311 — checker c77c4396 +
battery ebc4a4d5 == origin/amd/main, ALL ARMS rc=0 0.111s single-tmpdir), ETXTBSY +
sentinel-ratio seams. `(iii)` hash-verify of the pins: c77c4396=ebaacf0c(r1_ring_guard_check.sh)
/ ebc4a4d5=243eb579(r1_ring_guard_selftest.sh) / 2df9429d=6386355c(python twin) — content8 =
**sha256-of-bytes-first8**, all three match at-tree on main. No live work; LEG 1 ticked.

**LEG 3 — A1–A7 burn-down stamp sweep (chair's NVFP4-audit-home first task).** Convention
re-derived at bytes: content8 = sha256-first-8 of the FILE (NOT a git object sha) — proved by
re-hashing the banked trio above. TWO namespaces: **(N1)** `/home/chris/agent3_cells/`
desk files, deliberately UNCOMMITTED by agent3's design (rev-2 record @ 97d70217 banks this so
nobody files a phantom hunt); **(N2)** git objects. Full-ODB sweep (21,447 blobs hashed) + desk
re-hash: **ALL SEVEN A-ROWS RESOLVE TO REAL PATHS — ZERO PHANTOMS.**

| A-row | cite | resolves to | verdict |
|---|---|---|---|
| A3/A4 | 0fe01f38 | (N1) `nvfp4_admission_host.cpp` (15292 B) | ✓ re-measured at desk |
| A5/A6 | 20eb99a7 | (N1) `nvfp4_divisor_word_host.cpp` | ✓ — caveat: NOT host-linkable today (my seq-132 row: ld names full reader-stack syms) = plan's OWN named prereq, hour-1 honest |
| A1/A2 | af31aecd (+ bccdcb39) | (N1) `nvfp4_synth_admission_host.cpp` | af31aecd ✓ live. **FLAG: bccdcb39 resolves in NEITHER namespace — it is the SUPERSEDED older pin of the SAME FILE** (rev-3.3 text: `af31aecd = bccdcb39 + build-line correction, documentation-only re-pin`). Not a phantom-hunt; §8's row should cite af31aecd as LIVE, bccdcb39 as lineage-only (annotate-not-delete precedent) |
| A7 | e6db3027 | (N2) commit **656aa7f6**`:`tools/ops/check_kv_tier_priced.py` (blob 1d63008f, sha-first-8 exact; branch origin/amd/wo-p3-serve) | ✓ OLD-but-real pin. **DRIFT FLAG (substance, not phantom):** branch TIP carries **a32b4883**, drifted at 6435d727 (agent4 STEP-1 — substantive, reviewable schema widening) |

**A7 DRIFT RECEIPT BANKED — a32b4883 CARRIES ITS OWN GREEN at tip** (git-archive extract of
origin/amd/wo-p3-serve, host python3, deleted after): (1) `--selftest` rc=0, all seven arms
observed distinct (pricing convict/acquit/refuse + table pair acquit/silent-gap/quiet-grant/
misuse-refuse) — teeth proven; (2) real pricing arm at tip: **7/7 KvCacheStorage members
explicitly priced** (BFloat16, Int4Group64, Int8Group64, KvarnK4V2, KvarnK4V2, KvarnK5V4,
Nvfp4Group16), BF16 default reachable only by its naming tier, rc=0; (3) pair-mode vs the REAL
generated `src/ops/shape/tier_shape_table.h`: **8/8 schema tiers** carry row-or-named-refusal,
rc=0 at BOTH `--step 1` and `--step 0`. **Recommendation to agent5's desk: re-pin A7
e6db3027 → a32b4883** with a lineage row (witnessed pin 656aa7f6@e6db3027, drift commit 6435d727)
— measurement-backed, my receipts above. Row ownership: the plan doc is agent5's, routing the
RULING there, not drive-by editing (freeze law).

**LEG 2 — attempt-9 arming verified; (v)-banner answer delivered.** Kit cold-run at my seat on
the LIVE tip 615eda0f: `git archive` extract, read-only, discarded, zero device, seconds. Text
read per its own law (rc≠verdict): **9 honest FAILs — expected-and-CORRECT-RED, the tip lacks
agent4's in-flight five-line join** (PRED-B ×4 route/rccl/guard/vacuous, PRED-E ladder literals
3/1, T3 barrier :1626/:3591/:5224, T1 ×2); PRED-A'/PRED-D(+teeth)/PRED-C/routing-host-31-arm/
gate2 GREEN; tp_argmax_r1_host NOT-OBSERVABLE at this tip (file absent). **Kit defect named
(closure law): battery printed `INSTRUMENT-ERROR: 1 leg(s) NOT-OBSERVABLE — do NOT ship a GO row`
YET EXITED 0** — a loud-but-zero exit is the instrument-reporting-lie class; 3-line fix (rc=2
on MISSES>0, rc=1 on FAILS>0) rides agent1's desk — UNTIL FIXED: read the TEXT, never the rc.
`(v)` question: main's gate_pg1_whitelist.sh currently carries only **(a)..(s) = 19 letters —
(t)/(u)/(v) are NOT at main tip yet**, so (v) does NOT print in the banner until the join lands;
(v)-prints is verified only at my lane b9c80be1 (22 letters, header :953, double registration:
in-gate 4-leg block + CTest `add_test(ninfer_tier_shape_table_test)` tests/CMakeLists:60). On
attempt-9 boot-sha push: fire BOTH (kit at sha + suite template at sha, letters by NAME, banner
capture). Standing by for the sha; no polling-block, ls-remote on next beat.

**SELF-CORRECTION on the kit-defect claim above (same beat, before anyone acted on it): the
cited defect DOES NOT EXIST.** My first cold run read `EXIT=0` from `bash kit ... | tail -45;
echo EXIT=$?` — that is **tail's exit code**, not the script's (pipe-exit-status class — the
family this project has named five times). Direct re-run: `bash tools/v340l/tp4_arming_battery.sh
origin/amd/main ... ; echo $?` → **rc=2** printed on exactly the INSTRUMENT-ERROR verdict line.
Law D three-state is CORRECTLY wired: rc=2 on MISSES>0. **No defect filed to agent1's desk;
the closure-law row is retracted as never-born** (a bug claim needs a RED capture and mine was
an artifact of my own measurement plumbing — the instrument reporting-lie here would have been
MY receipt, not their script). COSMETIC OBSERVATION only, not a bug: when FAILS>0 AND MISSES>0
the SUMMARY line names only the NOT-OBSERVABLE count (rc=2 dominates rc=1) — the 9 FAIL legs
are still individually printed above it, and per the chair's own reading law ("read the WARN/
NOT-OBSERVABLE TEXT not just rc") the text carries the verdict; no change requested.
Lesson banked for this desk's receipts: **never grade a piped command by $?** — capture to file
or use PIPESTATUS; my own sentinel-class radar pointed inward for once.

## MAIN-HOME COLD VERIFY BEAT 2026-09-14T20:5xZ (chair seq-147 legs 1-2; receipt = doc 28)

agent3's §N0 family landed main @ c218547f; all six stamps 6/6 re-hashed match (desk==main bytes,
desks now redundant for these). Built+ran FALSIFIED at my seat, sandbox git-archive, zero-card:
**admission 22/22 rc=0 + rc=1 (+8 scale corruption) + rc=2 (API-drift sim) all three states
MEASURED — verified-shipped. divisor 41/41 rc=0 — with TWO findings: (1) ITS OWN SHIPPED BUILD
LINE FAILS AS WRITTEN (`.cu` without `-x c++` → ld linker-script error; RED captured verbatim in
doc 28; 2-line comment cure rides agent3's desk — third instance of the build-line-drift class
tonight); (2) MY OWN seq-132 "NOT host-linkable today" ROW IS RETRACTED — partial-closure
measurement; full notes-closure links bare-host, so artifact-day A5/A6's only remaining prereq
is real BYTES, not linkability. pack-golden 69-ok rc=0 against **main's own in-tree corpus
(re-hashed = a024627b, the pinned oracle stamp exactly)** + rc=1 corpus byte-flip convict +
rc=2 --skip-corpus. LEG 2 A-row re-resolution: A3-A6 UNPHANTOMED at main; **A1/A2 (af31aecd
synth cell) and A7 (check_kv_tier_priced e6db3027/a32b4883) still pin OFF-MAIN bytes** — cures
named in doc 28 (ship-or-recite; land-with-join). Six foreign fire-pings (nonexistent
/tmp/pi-fglpev) filed unchased. TP4 leg holds: wo-p3-serve moved again (attempt-10 era,
agent4 owns admission-wedge fix); standing FIRE order now reads **attempt-11's bank sha** —
watcher armed, kit command owned, nothing needed from the wedge fix for this desk.
