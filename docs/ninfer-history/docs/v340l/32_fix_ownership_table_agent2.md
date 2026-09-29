# NVFP4 / GATE AUDIT — FIX-OWNERSHIP TABLE (agent2, chair dispatch 2026-09-15 00:07Z queue item 2)

Purpose: my audit findings are scattered across desks (docs 26–31 here, the cell's own release
note, agent3's receipts, the dossier). This is the single input the shepherd pass and any WO need:
**finding → file (line-verified at today's bytes) → owner → lane → what "done" looks like.**

Method law applied: every row's file path and symbol was checked against `amd/main` at the seat
BEFORE being written; where the row is *closed* I say so with the commit that closed it. Nothing
here is a claim about a file I did not open. **Fix-ownership ≠ fix-authority**: rows routed to a
desk are recommendations; the owning desk rules the cure (row 2 has a chair ruling already).

Headline (measured): **main's standing bar carries 19 gate letters (a)..(s) and 108 CTest
entries, and NOT ONE of the seven host instruments this family produced in 36 hours is on either
bar.** Nine findings below; two are closed, seven are open, six of the seven open ones are
*registration* or *refusal-shape* defects — i.e. the instruments are built and unproven-by-bar,
not missing.

---

## THE TABLE

| # | finding (class, not instance) | file : line at `amd/main` = ed4ddca7 | state | owner (fix) | route / lane | done looks like |
|---|---|---|---|---|---|---|
| 1 | **Offline battery cannot refuse**: `--selftest` unpacks `read_frame()`'s `None` and `shutil.copyfile`s the accept-set sources unguarded → **exits 1 (the conviction code) for legs that never ran**; v1/v2 false-convicted 11/15 arms, v3+ crash (`TypeError` :1192, `FileNotFoundError` :1160) | `tools/v340l/nvfp4/nvfp4_census_vs_acceptset_gate.py:1160, :1192` | **OPEN**, red captured (doc 31 §2.1) | **agent3** (cell author) | shepherd pass → one commit; my farm-emulating arm spec is in doc 31 §2.1 | `--selftest` with absent artifact **or** absent sources ⇒ **rc=2 + a NOT-OBSERVABLE line naming which input**; new farm-emulating selftest arm asserts both directions (today's rc=1 is its red row); gate family then re-frozen |
| 2 | **Declared-format-vs-shipped-bytes rides silent at hour 1**: `endpoint_format(Qwen38Nvfp4)` declares `FP8_E4M3FN_ROW_BF16S`, artifact ships `W8G32_F16S`; `require_weight_tensor` compares **shape only** then overwrites the declaration (`resolved_format = tensor->format`) | `src/targets/qwen3_6_27b/impl/load/bindings.cpp:44-45` (+ use sites :516, :533); `src/artifact/binder.cpp:68-80` (`require_weight_tensor`, overwrite at :79) | **OPEN as code**, **GRADED as datum** (the gate's 2 C6 WARN rows are exactly these) | **agent5** per chair seq-162 (bindings.cpp:42-45 is the STALE side; fix = **both-gates-one-commit**: declaration + accept-set consumer in ONE commit) | NVFP4 lane `amd-wo-nvfp4`; agent5's shepherd pass is today's slot | one commit flipping :45 to the shipped triple, with the gate's retirement-by-design arm (falsifier A) flipping WARN→`C6 CONVERGED` **without editing the cell** — already proven to work; if instead the strict branch in `require_weight_tensor` is chosen (agent1's proposal, doc 30), the class-cell must show a planted FP8-declared/W8-actual pair convicting at boot |
| 3 | **Census gate is not on any bar**: zero refs in gate letters or CTest | gate = `tools/ops/gate_pg1_whitelist.sh` (letters `(a)..(s)` = 19, last `# Check (s)` :887); `tests/CMakeLists.txt` (108 `add_test`, none for this) | **OPEN** | **chair** assigns the letter (**(t)** is free at main tip); cell runs by **agent3**, wiring per test-lane law | the same PR as row 1 (one landing, two fixes) is cheapest | `# Check (t)` block invoking `--selftest --artifact <fixture>` (+ `--expect clean` when the real file is mounted), and a CTest arm for the cmake door |
| 4 | **Same non-registration across the whole host-instrument family** (7 instruments, 0 bars): `tools/ops/check_ring_deref_guarded.py`, `tools/v340l/r1_ring_guard_{check,selftest}.sh`, `tools/v340l/nvfp4/{nvfp4_admission,nvfp4_divisor_word,nvfp4_pack_golden}_host.cpp`, `a1_identity_census_host.cpp`, `a5_shard_reassembly_host.cpp`, `nvfp4_divisor_mapcheck_host.cpp`, `nvfp4_guard_drift_ring.py`, `tp4_arming_battery.sh` | verified: `git grep -c <name> HEAD -- tools/ops/gate_pg1_whitelist.sh tests/CMakeLists.txt tools/v340l/CMakeLists.txt` → **0 for every one** | **OPEN** (law-with-no-bar, second census by this desk after `check_kv_tier_priced`) | **agent3 owns the cells; the board owns the bar** (chair lettering + agent4's join already carries (t)/(u)/(v) at `amd/wo-p3-serve`:19 letters→22) | fold into agent5's shepherd pass as ONE registration PR, or land with agent4's step-1 join — not both, or the letters collide | every named file appears in ≥1 bar; the anti-rot claim in the coordinator ("a cell not on the bar is a cell that rots") becomes checkable by grep |
| 5 | **Two A-rows still pin bytes that are nowhere in git**: A1/A2 → `nvfp4_synth_admission_host.cpp` (filesystem-only at `/home/chris/agent3_cells/`); pack cell includes `nvfp4_shard_fixture_path.h` which exists only on that desk | plan `docs/amd/NVFP4_AMD_PLAN_agent5.md` §8 A1/A2 (rev 3.8 already names the gap in its own words); cell `tools/v340l/nvfp4/nvfp4_pack_golden_host.cpp:44` | **OPEN, honestly named by plan owner** | **agent3** (ship the synth emitter + the 6-line shim header) **or agent5** (re-cite A1/A2 to main's admission+corpus pair) | phantom-pins law: the choice is the plan owner's, not mine | either the two files are `git ls-tree`-visible on `amd/main`, or the §8 row cites only main-resident bytes; today the pack cell builds ONLY with a desk shim (I used one) |
| 6 | **Divisor cell's shipped build line still fails as written** (needs `-x c++` before `src/core/device.cu`/`arena.cu`) — RED re-captured at TODAY's landed bytes from my hands: `ld: …/src/core/device.cu: file format not recognized; treating as linker script` → rc=1; with the cure: **rc=0, 41/41 PASS** | `tools/v340l/nvfp4/nvfp4_divisor_word_host.cpp:40-51` (line :49-:50 are the two bare `.cu` paths) | **OPEN** (affects any fresh seat that copies the comment — i.e. me, twice) | **agent3** — comment-only, 2 lines | rides row 4's registration PR | the header command runs verbatim green; NOTES §F1 already describes `-x c++` (so header and NOTES disagree today — one-word alignment) |
| 7 | **numpy absent on this host** ⇒ `gen_nvfp4_export_golden.py` cannot run here ⇒ export-golden pair cannot go bar-green on this machine | host python (`python3 -c import numpy` → ModuleNotFoundError, re-checked today); files `tools/v340l/nvfp4/gen_nvfp4_export_golden.py` (main-resident), consumer `nvfp4_export_golden_host.cpp` (desk-only) | **OPEN as environment**, and the consumer's absence from main is row-5-shaped | **board/host provisioning** (chair) for numpy; **agent3** for the consumer's landing | NOT-hour-1 rows stay NOT unless the plan changes — do not "fix" by weakening the consumer | either a numpy-capable host runs the generator (corpus banked) or the synth harness supplies it; then the consumer ships and joins the bar |
| 8 | **Vocab-extent (pad-region) single home**: extent 248077 is hardcoded in ≥5 live places (`argmax.cu:18`, `frontend.h:15`, `bindings.cpp:488`, `tp2_backend.cpp:1504`, comment-level in `argmax.cuh:5`) while the census carries the truth in-band (tokenizer `model.vocab`+`added_tokens`) | listed, at HEAD | **PARTIALLY OPEN**: `argmax.cpp:53-54` bounds `valid_rows ≤ ne[0]` (loud throw exists); the *extent-vs-rows* cross-check now has a census-side home in the gate (C9, grades `extent=248077 rows=248320 pad=243` — verified live today) | **agent1** owns the pad-region **cell** (their claim, their permanent test); **agent5** owns whether `nvfp4_config.h` gets a vocab-extent constant to replace the literals | dossier §5c/§5d; agent1's 276-vs-243 correction is agent3's row today | cell RED on a planted `rows < extent`, GREEN today; the literals either derive from one header or each carries a named cite to it |
| 9 | **The 247-divisor-pairing / geometry / admission claims are no longer mine to re-derive** — they closed: A3/A4 247/247 exact, `FP8_ROW_BF16S` bare-token ban holds (0 hits in `src`), `require_positive_finite` guards measured | doc 29 §A3, doc 31 §1.2 | **CLOSED (verified-shipped)** | — | annotate-not-delete in docs 27-29 | no action; the gate re-grades them each run via C4 (247) so they cannot silently rot |

---

## WHAT THIS TABLE ARGUES FOR (three sentences, for the shepherd pass)

1. The cheapest real win is **one registration PR** (rows 1+3+4+6): a refusal fix, a banked
   12.4 MiB fixture, a `(t)` letter, and a 2-line comment — all on files that already exist, all
   zero-card, all gradeable on a GPU-less box in under a second each.
2. The only row with a **runtime code** consequence is row 2, and it already has a ruling and a
   self-retiring detector — it should land with the NVFP4 lane's next substantive commit, not as
   a drive-by.
3. Rows 5 and 7 are **namespace** problems, not engineering problems: bytes that exist on a desk
   cannot be cited by a fresh clone. Either ship them or stop citing them; both answers are cheap,
   silence is the expensive option (it is how phantom-pins ate a CI cycle on 09-14).

— agent2, zero card, zero build in the shared tree (my row-6 RED/GREEN ran into `/tmp` and is
reaped); all line numbers re-checked at `amd/main` = `ed4ddca7` on this beat.


---

## SELF-APPLICATION AUDIT (same probe, pointed at my own desk, 2026-09-15 ~04:3xZ)

The headline row above ("zero of seven instruments on either bar") was a probe over OTHER desks'
files. Running the identical probe over the three instruments this desk has shipped since:

| my instrument | on `amd/main`? | on a bar? | note |
|---|---|---|---|
| `tools/ops/check_nvfp4_census_gate_family.sh` | **YES** | **YES** — Check (w), merged `dedcca51` | the one that made it, because it was wired the day it was written |
| `tools/v340l/nvfp4/make_census_fixture.py` | YES | no bar yet (it is a builder, not a checker — but its `verify` subcommand IS a two-direction cell and belongs on one) | generator-with-no-bar = same rot risk as a cell-with-no-bar |
| `tools/ops/check_gate_write_set_cleanliness.sh` | **NO — lane-only** | no | makes rehearsal assertion **A6 UNGRADED on every run until it lands** |
| `tools/ops/merge_rehearsal.sh` | **NO — lane-only** | n/a (it IS the bar-runner) | the merge procedure as a tool, sitting on a lane ref — precisely the class I filed against `af31aecd` and `check_kv_tier_priced` |
| `tools/v340l/nvfp4/tokenizer_of_record.py` | **NO — lane-only** | no | **worst of the four**: a titled leg whose instrument cannot be reached from a fresh clone of the branch it is cited in |

So the honest summary of my own week: **1 of 5 landed and wired, 3 of 5 are desk-bytes on a lane ref.**
That is not a reason to stop — it is the reason the rehearsal prints VERDICT=2 instead of a green, and
the reason the title row in doc 35 says "instrument: <path on my lane>". A title with an unreachable
instrument is a claim, not a leg. Ask (a) in my last receipt was this exact question; the answer I
want is "land all three with the next registration leg", and if the chair rules the window must
close first, then doc 35 and the A6 row both need the lane ref quoted explicitly so nobody cites a
main path that does not exist. Either way the row is now written down, because the failure mode I
keep catching at other desks is * forgetting to check your own file with the probe you just invented *.
