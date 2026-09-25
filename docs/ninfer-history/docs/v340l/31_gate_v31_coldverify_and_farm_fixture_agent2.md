# GATE v3.1 COLD-VERIFY RECEIPT + FARM-FIXTURE RULING (agent2, morning seat 2026-09-15, chair dispatch 00:07Z queue items 1 and 4)

Zero card, zero build, host-only. Everything below was EXECUTED at this seat against
LANDED bytes, not quoted. Artifacts of the run live in `/tmp/agent2_v31_verify/` (transient);
their content8 stamps are printed so a re-run can be compared, not believed.

---

## PART 1 — queue item 1: the owed 30-second re-run, filed against LANDED bytes

### 1.1 The version question, resolved by three probes (my own law: before grading any
### short-hex cite, run all three probes and name the namespace)

My 76f2adad row said "main still v1 @ 4b74533f — v2 DESK-ONLY". **True when written, dead now.**
Timings from the objects themselves (author-date of each landing):

| event | sha | time (−0500) |
|---|---|---|
| my flag written (lane tip) | 76f2adad | 2026-09-14 **18:03:20** |
| v1 lands (agent3's cell) | cca2213d | 17:48:32 |
| **v2 lands** | 6a2560c1 | **18:04:10 — 50 s after my flag** |
| v3 lands | 742ccc3b | 18:23:56 |
| **v3.1 lands (current)** | c2f18eec | 18:30:08 |

Probe set on `tools/v340l/nvfp4/nvfp4_census_vs_acceptset_gate.py`, run at `amd/main` = 1f61a58c:

* **(i) rev-parse** → `amd/main:…gate.py` = blob `4fba2cc44db05567f2d4a0604430181bbac08981`;
  same blob as `c2f18eec:` — so the landing commit's bytes are STILL on the tip (no drift since).
* **(ii) cat-file blob | sha256** (content8 = sha256-first-8 of BYTES, the content namespace):
  landed gate = **`b0309248`**, landed `GATE_CELL_RELEASE_2026-09-14.md` = **`44bb203a`**.
  Chain cross-check, same probe at each landing commit — v1 `4b74533f` / v2 `04589781` /
  v3 `330b8cff` / v3.1 `b0309248`. Each maps 1:1 to its ledger cite; the chair's c2f18eec row's
  "v3.1 @ b0309248" is **exact**.
* **(iii) sha256sum over the working-tree file** in the shared checkout → `b0309248…` (full
  64-hex `b03092489e2a284cbcc160af305411a57d820a16d83d0021db51b6010774df31`), 1281 lines.
  Tree == tip == landing commit. **No drift anywhere in the chain.**

**Name the kind of my old stamps, once and for all:** the trio in doc 28 (:134) — gate `04589781`,
run log `c1070252`, release note `6b24a6a4` — are content8 (sha256-of-bytes). `git cat-file -t
04589781` / `b0309248` / `330b8cff` all return `fatal: Not a valid object name` AT THIS SEAT
today: they were never revisions and must not be probed as if they were. **VERDICT: amd/main IS
v3.1 @ content8 `b0309248`. No drift to name; my flag was stale-desk-bytes, superseded 50 seconds
after it was written.** Annotated in place in doc 28 (annotate-not-delete) rather than edited out.

### 1.2 Execution at the landed bytes (this is the owed 30 s; it cost 0.54 s)

* `--selftest --repo <main checkout>` → **rc=0, SELFTEST 23/23**, 0.537 s. Log content8
  `69529609` (full `69529609e0167a48…bbca7dd0`). The 23rd arm is the §8m fork-guard (live C9
  path must route through the SAME `c9_grade` the red-arms fire) — it PASSES here, i.e. the
  anti-ceremony condition holds at the landed bytes, not just at agent5's desk.
* `--expect clean` against the REAL 18.3 GB artifact (`/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer`
  present, 18,324,067,840 B = `ARTIFACT_BYTES_OF_RECORD` exactly) → **rc=0, VERDICT=0,
  divergences=0, warnings=2**, 0.171 s. Log content8 `32c56615`. The 2 WARNs are EXACTLY the two
  ruled C6/stale-constant rows (`text/token_embedding`, `text/output_head`, W8G32_F16S shipped vs
  FP8 declared, `bindings.cpp:42-45` ruled the stale side) — byte-identical text to the v1/v2-era
  runs, so the contract "warnings = exactly the 2 ruled rows" still holds at v3.1. Full log
  content8 `32c56615` (`32c5661515876eb5…65a48e23`).
* New-at-v3 arms, seen live: **C9 extent=248077 rows=248320 pad=243** (print, not demand);
  C8 strict-triples **700**, C8c role-shape membership **1052**, C8d overlap **NONE**,
  C4 pairing **247**. All match the numbers in the chair's landing rows — re-derived, not quoted.
* **Accept-set inputs hashed, not assumed** (registration-input cite-by-sha law): all 9 `SRC`
  files the gate parses are present in the checkout AND byte-identical to `amd/main`:
  b27 `d56f5483`, b27h `66fb1836`, vis `66edd349`, vish `631cf066`, res `20409f72`,
  lay `51cfd19c`, rd `e6868243`, tb `98bbc070`, cfg `8cf4be27`. **Zero drift** — the receipt is
  against one consistent tree, not a mid-merge sandwich.

### 1.3 Where my earlier grades stand now

My v1 audit (doc 28, `cca2213d` bytes) and v2 roll-forward (76f2adad, desk bytes `04589781`)
both remain TRUE OF THEIR PINS. The v2-era substantive rows carry forward unchanged because the
lines they grade did not move in v3/v3.1: the 8→9-file accept-set source set is still
load-bearing and refuses loudly (re-proved below), and the §8h silent-misread closure (C8/C8c)
is still the arm that sees the class my doc 30 named from the other side. v3/v3.1 ADDED the C9
vocab-extent organ and the fork-guard; nothing I convicted has been weakened.

**Item 1 status: CLOSED. Receipt filed. No drift to report.**

---

## PART 2 — queue item 4: can a banked fixture make the census gate's farm arms observable?

Chair framing: cca2213d's named non-coverage says the **rc=2 path is NOT-OBSERVABLE without the
18.3 GB artifact mounted**. My §N0.4 verdict already established `nvfp4_shard_image` IS a host
cell (zero `/dev/kfd`). RULING requested: does a banked fixture unblock farm wiring of the whole
gate family? **Answer: YES, and the fixture is 12.4 MiB, not 18.3 GB — with one design
constraint that is NOT obvious and one real defect it exposes.** Measured, not argued.

### 2.1 The defect the question exposes first (this is a RED capture, file it as a bug)

At landed v3.1 bytes, `--selftest` **cannot refuse honestly when the artifact is absent**:

```
python3 gate_v31.py --selftest --repo <valid checkout> --artifact <absent path>
  -> TypeError: cannot unpack non-iterable NoneType object   (gate_v31.py:1192)   rc=1
```

`read_frame()` correctly returns `None` with a `NOT-OBSERVABLE` blocker appended, and `main()`'s
live path handles that (`run_gate` :855-857 → rc=2, measured clean). But `selftest()` unpacks it
directly at :1192 — `ident, objs, jb, fs = read_frame(artifact, r)`. Same class, second site:
:1160 `shutil.copyfile(Path(repo)/rel, dst)` raises `FileNotFoundError` (rc=1 traceback) when the
accept-set sources are absent — i.e. the selftest's own "RC2 accept-set sources absent" arm can
never be reached by running the selftest against a tree that lacks them; the arm passes only
because `run_gate` is called with a synthetic repo copy *inside* the function.

Consequence for the farm, measured across the whole lineage (no-artifact selftest runs):

| bytes | no-artifact `--selftest` outcome |
|---|---|
| v1 `4b74533f` | **3/14 rc=1** — 11 arms falsely convicted (CONTROL real-census rc=2 ≠ 0) |
| v2 `04589781` | **3/18 rc=1** — same failure mode, 15 arms falsely convicted |
| v3 `330b8cff` | **crash** TypeError rc=1 |
| v3.1 `b0309248` | **crash** TypeError rc=1 (line 1192) |

So the cca2213d note's proposed CI form — "the selftest arm is the CI-farm-safe form ONLY if a
fixture artifact is banked" — was right about the fixture and wrong about the mechanism: on a
GPU-less, artifact-less farm box the cell today returns **1, the conviction code, for a leg that
never ran.** That is precisely the mislabel family this project has caught at four desks
(agent5's runner kind-attribution, my own rc-through-pipe, the INSTRUMENT-ERROR-wearing-a-kind
row). It is worse than the rc=2 it was supposed to avoid: a farm run would print a RED and the
board would read it as "the gate found a divergence."

**Bug class (generalized per RED-GREEN law): an offline battery whose inputs are not all
present must print NOT-OBSERVABLE and exit 2 — it must never exit the conviction code, and must
never crash into it.** Fix direction, ~4 lines at both sites, cell owner's desk:

```python
frame = read_frame(artifact, r)
if frame is None:
    print("SELFTEST NOT-OBSERVABLE: artifact/directory unreadable — arms cannot run:",
          r.blockers[-1]); return 2
ident, objs, jb, fs = frame
```
and for the source-copy block: `if not (Path(repo)/rel).exists(): print(...); return 2`.
The closure cell that makes it permanent: a farm-emulating arm — `--selftest` with a valid
`--repo` and an absent `--artifact` must assert **rc=2 AND a NOT-OBSERVABLE line**, both
directions (today's rc=1 is the red row; post-fix green row named in the same arm). Not filed
against a person: v1/v2 shared the false-conviction half, v3 added the crash half.

### 2.2 The fixture, built and graded (this is the ruling's evidence)

What the cell actually reads from the artifact: the 8-byte magic, the 8-byte directory length,
**the 206,636-byte JSON census**, the file's total size, and **ONE payload region — the in-band
`frontend/tokenizer.json`, 12,809,320 B at the data base** (C9's extent arithmetic). `grep`
over the landed bytes: exactly one `f.seek` (line 926), no other payload read. So the other
~18.31 GB is never opened by this gate. A fixture therefore only has to be truthful about
magic + census + the tokenizer payload + a file size that closes the base arithmetic.

**The non-obvious constraint (why a truncated or sparse file fails):** C9 derives
`base = file_size − max(offset + bytes)`. Truncate the artifact and `base` goes hugely negative; a
sparse file with the directory VERBATIM fails for the same arithmetic reason (its declared
extents still run to 18.32 GB, so `base = 13,018,216 − 18,323,858,944 < 0`). Both measure the
same way: `VERDICT=2 … C9 NOT-OBSERVABLE: tokenizer payload unparseable ([Errno 22] Invalid
argument)`, selftest 21/23 (the two failures are exactly the two arms that need the payload:
CONTROL real-census and the §8m GUARD — honest refusals, not silent grades). So the fixture must **re-derive the
offset column** so the arithmetic closes at the fixture's own size. That is safe precisely
because no arm but C9's base uses offsets — I verified field-by-field: the fixture differs from
the real artifact's directory in `offset` on 1306 of 1307 objects and in **no other field**
(identity equal, all names/formats/layouts/shapes/bytes verbatim).

Fixture of record (built by this desk, generator reproducible from part 2.3):

| property | value |
|---|---|
| path (this seat, transient) | `/tmp/agent2_v31_verify/fixture_v2.ninfer` |
| size | **13,009,597 B (12.4 MiB)** = 0.071 % of 18,324,067,840 B |
| content8 (sha256 first 8) | **`f1dab9de`** — full `f1dab9de89e5018e1cab4a49510f5d5b659eaecc2d8e2768da4ae54ff616db80` |
| construction | real magic + REAL 206,636-B census JSON re-serialized with `offset := 0` for the tokenizer and `offset := −bytes` for every other row; then header + directory + **the real tokenizer bytes appended contiguously**, so `file_size − max(offset+bytes) == len(header)+len(dir)` exactly (asserted at build time) |

Grades at landed v3.1 bytes with the fixture, `--repo` = a 9-FILE COPY of main's accept-set
sources (the farm-realistic shape — not the full checkout):

* `--selftest` → **rc=0, SELFTEST 23/23**, 0.613 s (log content8 `69529609` — byte-identical to
  the run against the real 18.3 GB artifact; same arms, same verdicts, 1400× less mass).
* live `--expect clean` → **rc=0, VERDICT=0 divergences=0 warnings=2**, and the ARM and WARN
  lines are **byte-identical** to the real-artifact run (`diff` of the ARM/WARN blocks: empty) —
  including `C9 … extent=248077 rows=248320 pad=243` and C4=247 / C8=700 / C8c=1052.
* Backward compatibility of the same fixture: v1 **14/14 rc=0**, v2 **18/18 rc=0**, v3 **22/22
  rc=0** — one fixture serves the whole family and any lineage re-grade.

### 2.3 RULING (what the farm should actually wire)

1. **YES — a banked fixture makes the gate family farm-observable, and it is 12.4 MiB, not
   18.3 GB.** Bank `f1dab9de…` (or its regenerated twin) in the artifact bank next to the
   boot-stamped binaries, cite it by full sha256 + the fixture-stamp-not-artifact-echo law.
2. **Fix the selftest refusal FIRST (2.1) or the farm arm is born-wrong.** With the fixture
   mounted the crash is unreachable, but a farm box where the fixture failed to mount must
   exit 2, not 1. Wire order: refusal fix + its two-direction cell → fixture banked → farm
   job runs `--selftest --repo <checkout> --artifact <fixture>` as the standing arm, plus
   `--expect clean` against the fixture as the live-shape arm.
3. **Label what a fixture proves.** Fixture-graded = census/arithmetic/accept-set wiring, incl.
   C9's tokenizer extent (the payload IS the real payload). Fixture-graded does NOT prove
   tensor-payload bytes (no arm reads them anyway — divisor VALUE and A3 geometry are artifact
   legs, they stay on PG-1/host-rig with the real file) and does NOT prove offset truthfulness
   (offsets are re-derived by construction — a fixture can never catch an offset-overlap bug;
   the real-artifact run must stay on the mount-capable seat, and its honest rc=2 is a
   *legitimate* NOT-OBSERVABLE, which the fix in (2) makes distinguishable from the false RED).
4. **Generator belongs next to the cell** (`tools/v340l/nvfp4/make_census_fixture.py`, ~20 lines:
   header+dir+tokenizer read, offset re-derive, the base assert) so a re-export regenerates the
   fixture instead of anybody hand-poking bytes — the same "legality by instantiation, not by
   whitelist" instinct agent1 banked: the fixture's validity is ASSERTED at build
   (`file_size − hi == len(hdr)+len(dir)`), not assumed from its author's intent.
5. **Cost, honestly stated:** my fixture was built by this desk from the mounted artifact
   (2 seeks, 12.8 MB read, no full-file hash) and is a derived work — if the artifact moves,
   regenerate; never let a stale fixture grade a new tree silently. The fence against that is
   already in the cell: `ARTIFACT_BYTES_OF_RECORD`/sha of record are printed as OF RECORD, and
   C9's datum rows would drift — but the strongest fence is (4)'s generator + re-bank on the
   next export, which I recommend as the WO text.

**Item 4 status: RULED, with a filed bug (2.1) and a byte-stamped fixture (2.2).**

---

## PART 3 — queue item 3 precondition (checked, not assumed)

`tools/v340l/tp4_arming_battery.sh` EXISTS at `amd/main` tip: sha256 first-16
**`024d97bbe0b42c45`**, 305 lines, header carries the AUTHOR-ABSENT use ruling, law-D three-state
contract and the PRED-E/LEG-15 growth since the 19-leg era. Precondition SATISFIED; no gap to
name.

**Cold-run done at this seat, and the leg count is MEASURED, not remembered** (`bash
battery.sh amd/main <checkout>`, log content8 in the transient dir): **17 leg lines printed =
7 ok / 8 FAIL / 1 n-a / 1 NOT-OBSERVABLE, rc=2** with the honest closing line
"INSTRUMENT-ERROR: 1 leg(s) NOT-OBSERVABLE — do NOT ship a GO row from this run". Every red reads
as expected for a NON-boot tip: `amd/main` still lacks agent4's step-1 join (no
`src/ops/shape/tier_shape_table.h`, no `tools/ops/gen_tier_shape_table.py`, no
`tests/multi_gpu/tp_argmax_r1_host.cpp` → the NOT-OBSERVABLE; world-free gqa ladder at
`src/ops/wrapper/gqa_attention.cpp:40-48` → PRED-E red; pair-literal barrier lines still present).
So the kit is ALIVE and its teeth are intact — 17 legs, not 19, not 14: **cite 17 or re-count, the
number belongs to the tree it ran on**. **Armed and waiting on agent4's [C] boot-bytes stamp** —
battery-seconds at that sha is the next fire-side receipt from this desk (leg count read from
output at the sha; text read before rc, per my own self-correction row in doc 27).

— agent2, zero card, zero build in the shared tree, `/tmp` scratch only (reaped on request),
shared checkout read-only (all reads via `git show`/`cat-file`; writes only inside
`/tmp/agent2_v31_verify/` and my own lane docs).
