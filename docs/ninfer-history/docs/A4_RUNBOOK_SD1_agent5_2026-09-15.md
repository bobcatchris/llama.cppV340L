# A4 RUNBOOK — the SD-1 goal-line boot, executable form (agent5, plan owner; chair seq-190, 2026-09-15)

**What this is:** the procedure the board RUNS the day the NVFP4 ports go warmup-green. The rulings it
executes live in `docs/amd/REVIEW_ACCEPT_v2_SD1_agent5_2026-09-15.md` (RULING 2 = the six conditions,
RULING 3 = artifact reality) and plan §8t/§8v/§8w — this doc cites them, it does not re-argue them.
Every era's endgame stumbled on a runbook that lived in memories; this one does not.

**Slot notation:** `«SLOT»` = fill at fire time, never earlier. A blank printed in a receipt row is a
VOID row, not a passing row.

---

## §R0 — PRE-FLIGHT (all zero-card; every leg fails STOP, none may be waived)

| # | leg | command / check | expected datum | STOP if |
|---|---|---|---|---|
| PF-1 | GPU grant | written grant from coordinator, id `«GRANT_ID»`; guard: `tools/smoke/diag/gpu_guard.sh` + `rocm-smi --showpids` | grant names this boot; zero foreign contexts on dev 0-3 | no written grant ("clear to claim" is not a grant) |
| PF-2 | BIN slot | `«BIN_SHA16»` → boot `/home/chris/artifacts_bin/ninfer-serve_«BIN_SHA16».bin` (bank = filename = stamp, BOOT_LAUNCH_RUNBOOK §4); record `sha256sum` full at fire | bank file exists; its name-stamp == its sha | any boot from a lane build path |
| PF-3 | ARTIFACT slot | `«ARTIFACT»` = `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer`; `stat -c %s` == 18,324,067,840; sha-of-record `eaf8ad12…56d2` cited from the mount-seat row (do NOT re-hash 18 GB at this desk) | size matches; artifact named by sha16 in the row header | size ≠ size-of-record (wrong file = verdict void, law-D rc=2 shape) |
| PF-4 | **C5 re-run falsifier** | `python3 tools/v340l/nvfp4/c5_tokenizer_identity_probe.py` (0.8 s) | rc=0, `BYTE-IDENTICAL` — REQUIRED only if «ARTIFACT» ≠ eaf8ad12-pair-of-record (new convert file ⇒ probe fires FIRST) | pair check fails ⇒ C5 cannot convict; flag row, do not mint against A2/A1 |
| PF-5 | **FORMULA — RATIFIED, slot filled from the ratified authority exactly as designed** (chair seq-2 ratification + agent1 signature; canon wording: agent1 `DIGEST_DIALECT_CENSUS_agent1_2026-09-15.md` §3, durable @ `e6bebd6a` on `origin/amd/wo-agent1-support`; ACCEPT §1b amendment landed by agent1 @ `bcff9d2d`; **digit-check at agent5's seat this beat**: recompute on `G18r28_hladder_g1000_1.json` (len_r=185, len_c=52) → concat prints `e764922b0dfe1a7f` == agent4's banked row, reason-only prints the census-predicted dead value `7b1ad95b6099355e` — both directions predicted, neither remembered) | see **PF-5 VERBATIM** block below the table — runners paste IT, not this row's summary | the A4 row's first line is `FORMULA: ruling4-concat` printed BEFORE its first mint | any runner whose arithmetic cannot be named by that exact block; ACCEPT's old reason-only column is annotated DEAD via the census (annotate-not-delete — the column survives struck through with the cite, at the doc owner's next legitimate touch) |
| PF-6 | Tree pin | `«TREE_SHA»` = lane tip the bin was built from; anti-resurrection diff `git diff origin/amd/main -- src/runtime/tp2/tp_engine.cpp src/runtime/tp2/tp2_budget.h` == EMPTY | tree sha in row header; diff empty | stale-region regression (VRAM law gate) |
| PF-7 | **SF-1 host-stability — RESOLVED-AS-PHYSICAL; slot filled from the row's own words (chair §SF-1 @ `9d912423`), formal-clearance distinction KEPT** | faithful repro ran witness-armed (same bin `3da20992`, same 160,110 B file-routed body, 4-die sync 40k prefill): 1k pair green, leg-1 walked >=43% (15,232/35,607), peak 528 W, **zero voltage sags**, host survived the plateau; user hypothesis *"maybe my cat hit the power button"* — front-panel cut fits ALL forensics (zero journal trace, empty pstore, 53-s dead zone, wtmp unclosed) | row status, verbatim from `9d912423`: **`UNCONFIRMED-PHYSICAL, not software, not load-proven — deliberately NOT issued as a formal clearance`**. Near-clearance evidence citable as: repro-trace SNAPSHOT `26,549 lines, sha256-16 276a31ef44d51aa3` (taken at agent5's seat this beat; the row's own count 21,039 was true at stamp-time — annotate-not-correct — the LIVE file still grows because the witness was never stopped: `vwatch.py` PID 35999, not mine to kill, flagged to chair for teardown; a citable artifact is a stopped file, never a growing one) | **A4 FIRE RULE (retained distinction): 1k/512-token legs may fire NOW under the two kept policies; every 40k+ deep leg (incl. §R9 row L-40k) rides ONLY under witness + someone awake at the chassis — §SF-1 residual policy, not optional. No formal clearance exists; nothing in any receipt may say one does** |

**PF-5 VERBATIM — paste exactly this into every runner and row header (agent1's ratified block, `e6bebd6a` §3):**

```
FORMULA: ruling4-concat :: sha256_16( utf8(reasoning_content) || utf8(content) )
  field values are the RAW JSON string values as decoded — never the display-wrapped form,
  concatenated reasoning-then-content; digest = first 16 lowercase hex chars of the SHA-256
  of exactly those bytes, in that order, with no separator or added newline.
  At max_tokens=32 mint legs content="" (so the dead reason-only dialect agrees here BY EMPTINESS,
  not by law); at natural-stop legs content is load-bearing and the dialect is the difference.
```

## §R1 — OPENING GATE: WARMUP GREEN (free composition test; graded BEFORE any request is spent)

Boot argv (fill-in; shape measured at the honest-first boot 37ec0ba1, runner template
`results/amd/p3/G18r29_nvfp4_honest_agent4.sh`):

```
/home/chris/artifacts_bin/ninfer-serve_«BIN_SHA16».bin \
  /media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer \
  --port 8098 --devices 0,1,2,3          # WORLD:4 — the row names the world
  [ws96, untraced — carry the honest-first row's knobs unless the WO names a change] \
  > «LABEL»_serve.log 2>&1 &  echo $! > «LABEL».pid
```

Gate arithmetic on the serve.log (verbatim strings from the 37ec0ba1 reference sample):

```
grep -c 'warming up\.\.\.'                    «serve.log»  == 1
grep -c 'warmup failed (continuing)'          «serve.log»  == 0    # pre-cure sample line: '…: nvfp4 linear: unsupported shape'
grep -c '\[tp2 worker error rank'             «serve.log»  == 0    # pre-cure sample: rank 0..3 each threw the pinned text
grep -c 'listening on http://'                «serve.log»  == 1
```

**Why counts and not just 'listening': the pre-cure boot ALSO listened** (37ec0ba1: warmup threw on all
four ranks, server survived, socket opened — `LISTENING-BUT-NOT-SERVING` is exactly this shape). An
always-green-looking log is the mute button; the two zero-counts are the gate. Print `warmup=GREEN` only when all four arithmetic
arms hold as above; else STOP: kill the process GROUP you started (`kill -TERM -«PID»`, PID from the
.pid file — own pgid only, never pkill), bank the RED log, report — the port
isn't in yet and no request was spent to learn that.

**Cross-home note (agent1 seq-3):** these two directions — RED `…unsupported shape` ×4 + `warmup failed
(continuing)` pre-P1, the same greps going 4→0 post-ports as BOOT-B's inverse with a named site — are
**pre-registered in WO-NVFP4-1's CT-6 counter-test** (agent1, both arms per the ratification). The
runbook gate and CT-6 are one grammar with two citations; if they ever disagree, the CENSUS-side
string (`e6bebd6a`) and the serve.log verbatim (37ec0ba1) are the arbiters and this paragraph gets
annotated, not silently re-derived.

Capacity line (measure, never assume): `auto KV: … capacity=«N» tokens` must satisfy
`prompt_tokens + max_tokens ≤ «N»` with the arithmetic printed. Cited ceiling anchor at this
(bin-family, artifact, world): 63,936 tokens @ ws96 (37ec0ba1, measured). A4's budget (54 + 32) rides
this with enormous margin — the line exists for A6/A7 legs and for moved geometry.

## §R2 — CONDITION i: the PREREQUISITE PAIR, at THIS (bin, artifact)

BOOT-A (pre-port refusal) is already minted at bin `3da20992` (37ec0ba1). Pair law: if «BIN_SHA16» ≠
`3da20992…`, the pair RE-MINTS against the new bin — the refusal leg runs on the ported tree anyway
and must now print **zero** refusals (§8t inverse): grep counts `warmup failed` = 0 (== §R1) AND, as
the pair's B-leg datum, the pinned text `nvfp4 linear: unsupported shape` appears NOWHERE in the log.
Rows name WHICH guard fired (the `NINFER_ALLOW_NVFP4_TP2` env-vs-world=4 trap — a row that says only
"env set" measured nothing).

## §R3 — CONDITION ii: 25/25 SAME-BOOT DETERMINISM @w4

Request template (model id PINNED — the 404 lesson from 37ec0ba1's burned probe: the serving id is
`qwen3.8-27b`, never `probe`):

```
for i in $(seq 1 25); do
  curl -s http://127.0.0.1:8098/v1/chat/completions -d \
  '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hi."}],
    "max_tokens":32,"temperature":0, "«THINKING_FIELD»":true}' \
  > «LABEL»_rep$(printf %02d $i).json
done
```

(`«THINKING_FIELD»` per the ACCEPT thinking-mode-at-the-mint's-bytes note: read ONE rep's JSON, name
the thinking flag from the schema, print it; do not assume.) Digest per PF-5's ratified formula over
`.choices[0].message` fields; field-path + budget + formula + WORLD stated together on every row —
four-part contract. Expected datum: **one digest ×25**, finish_reason uniform across all 25, all same
boot. A2's profile tolerance (≤1 NAMED near-tie fork) governs CROSS-world comparisons only —
WITHIN-boot disagreement at temp 0 is STOP, no exceptions: capture-before-kill (STALL()+wchan on any
non-200), bank RED, report; never retry a disagreement away.

## §R4 — CONDITION iii: THE MINT RECORD (and mint-conditionality mechanics)

The row: `A4/@w4 · ARTIFACT nvfp4 18,324,067,840 B eaf8ad12…56d2 (or «ARTIFACT») · BIN «BIN_SHA16» ·
TREE «TREE_SHA» · GRANT «GRANT_ID» · FORMULA ruling4-concat · WORLD 4 · budget 32 · field-path
.reasoning_content + content (concat order per PF-5 VERBATIM) · digest «A4_MINT_SHA16» · finish «uniform value»×25 ·
capacity «N» · allocator words verbatim ('none' is a logged word) · C5-pair «YES/FLAGGED»`.
`«A4_MINT_SHA16»` is computed by EXACTLY the PF-5 arithmetic — raw JSON field bytes, reasoning-then-
content, no separator; at budget-32 the content term is empty by observation, which the block says out
loud so nobody mistakes the agreement for law.

**Re-mint mechanics (annotate-not-delete, checkable):** the mint is a property of the
(artifact, bin) PAIR. Either component moves ⇒ old row annotated `SUPERSEDED-BY <new row id> (moved:
artifact|bin <old>-><new>)`, old digest kept, NEW row id mints (`A4'`, `A4''`, …). No row's EXPECTED
cell is ever edited after fire — if a moved component "explains" an old mismatch, the explanation
goes in the annotation, not into the number. This is what prevents tomorrow's phantom A4-FAIL on a
re-exported artifact and tomorrow's silent A4-PASS on a stale bin.

## §R5 — CONDITION iv: the weights-loaded proof at the goal geometry

`«A4_MINT_SHA16»` **must differ from A2's mint digest** (q3 @w4 — the world-4 pair form of C5).
Equality = weights-not-loaded: this is a DEFECT CAPTURE, not a pass; bank everything, route to the
loader lane, report. (If «ARTIFACT» is new: PF-4 ran first or this proof is void.)

## §R6 — CONDITION v: the HUMAN-COHERENCE sub-procedure (C6 — capture-before-signature ORDER)

1. §R3 froze `«A4_MINT_SHA16»` into the banked row FIRST. (A coherence signature predating its digest
   grades vapor — the chair's words; the order is the protection.)
2. Human reader: chair or user — a desk never signs its own mint. Reader opens `«LABEL»_rep01.json`
   .. `rep25.json` reasoning text (all 25 should read identically — that's C4; the human grades
   MEANING, not consistency).
3. Optional depth leg: natural-stop pair at max_tokens=512 (world-4 sibling of A6) if the reader wants
   sentence-completion evidence; its digests ride the same formula name.
4. Signature file: `results/amd/nvfp4/C6_A4_«A4_MINT_SHA16»_«YYYY-MM-DD».txt` — **the digest is in the
   FILENAME** so a signature cannot pre-date what it read — body: reader name, date, digest echoed,
   verdict (COHERENT / INCOHERENT+named-shape), and `FORMULA: ruling4-concat` copied from the row.
5. Until the file exists, A4's max state is **MEASURED-PENDING-HUMAN** — no green, no quote, no
   ledger flip. Never marker-in-file-without-reader.

## §R7 — CONDITION vi + refusals + teardown (last words, in order)

- Row carries the provenance sentence verbatim from RULING 3 (QAT-exported planes; real-decode
  fresh-convert mints A4', does not edit A4).
- R2 env-gate refusal leg: any `NINFER_ALLOW_NVFP4_TP2`-family refusal row names which guard fired
  and at which world — never which env var was set.
- Teardown: kill own pgid only; post-kill `ninfer-serve=0`, `rocm-smi --showpids` KFD row; release the
  grant in the report. Bank the boot-stamped bin if the run changed anything (BANK-BEFORE-RELINK).

## §R8 — ANSWER TO THE CHAIR'S seq-186 QUESTION: put the split IN the WO text — paste block for agent1

**RULING: YES, it belongs in WO-NVFP4-1 itself** — §8s is plan-receipt prose; port desks read the WO,
and "it lives in a plan section someone may not open" is the prose-only-home failure this board has
already paid for three times. Three sentences, agent5's words, relay verbatim:

> **GOLDEN-LEG DEPENDENCY SPLIT (plan-owner ruling; NVFP4 plan §8s/§8v/§8w):** The port-side grading
> in this WO — the G2 determinism bar, G3 corpus, G4 long-generation, and the warmup-green composition
> test — compiles against the SHIPPED pack-convention leg (`nvfp4_pack_golden_host.cpp` @ 81aced51)
> plus agent3's host-side decode+classify golden cell, ALL runnable on this host with zero card and
> zero torch. The host-blocked fourth leg (export-golden consumer `143650e4`, needs a torch host) is
> CONVERT-SIDE ONLY: it gates admission of a FRESHLY PRODUCED artifact and has no role that precedes
> any port. Therefore no WO row may cite `143650e4` as an input; if a row appears to need it, that row
> is a convert-lane row wearing a port label and belongs beside the §8s owner sentence, not here.

— agent5. Companion rows: plan §8x; slots fill at fire time; nothing in this doc is quotable as a
result — it is the shape a result has to arrive in.

---

## §R9 — FIRE SHEET: nvfp4 FIRST-DECODE SESSION (measurement readiness, chair seq-11; pre-filled 2026-09-15 13:2xZ, zero-card)

**Purpose:** the day the first warmup-green nvfp4 bin exists, the w4-q3-vs-w4-nvfp4 decode comparison must
NOT be re-derived — these rows are the q3@w4 ladder's OWN geometry (runner `G18r28_hladder_agent4.sh`,
sha256-16 of script `1cbf252617205ae6`, p3 lane; file-routed body class per the chair's seq-180
40k-RED fix; `WORLD_FIELD=4` built into its header line), restated as exact commands with every slot
filled except the four boot-slots `«BIN_SHA16» «GRANT_ID» «LABEL» «tree-sha»`.

**Artifact identity — re-stamped THIS beat so the mint record carries it from second zero:** full-file
`sha256sum` of `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer` run at agent5's seat, completing
2026-09-15 13:2xZ: **`eaf8ad124256d0a0c1ebbbca442ca58eee4f97ab34a60a0b4d57e2b41e2c56d2`** —
EQUALS the hash-of-record (independent second full-carriage: agent2's mount-seat measure of Sep-14 +
this beat's fresh read; the mount file has not bit-rotted between eras). A4/§R4 rows cite BOTH stamps.

**Formula line (every row, first line, verbatim from PF-5):**
```
FORMULA: ruling4-concat :: sha256_16( utf8(reasoning_content) || utf8(content) )
  field values are the RAW JSON string values as decoded — never the display-wrapped form,
  concatenated reasoning-then-content; digest = first 16 lowercase hex chars of the SHA-256
  of exactly those bytes, in that order, with no separator or added newline.
```

**Shared preconditions for all rows:** §R0 PF-1..7 green (PF-7 now resolved-filled: 1k/512 legs fire
under policy; the L-40k row REQUIRES witness-armed + chassis-awake); §R1 gate `warmup=GREEN`; one boot
session, both legs, same server process (pair law: cross-boot equality proves nothing).

**Standing per-leg witness (chair seq-44 order, law-51 made mechanical; added at §R9 top so it binds
every row below):** after each leg's response lands, run the clock-agreement cell over the session log
and print its per-leg line INTO the row — a leg whose clocks are not printed is **UNGRADED**, whatever
its digest says:

```
python3 tools/v340l/nvfp4/finish_clock_consistency_host.py «serve.log» --json-dir «OUT»
# per-leg datum: delta_mode + lane/done/usage triple + finish-class + verdict (0/1/2 three-state);
# conviction or missing-print => leg UNGRADED (law-51: grades print the cross-clock comparison or mark it)
```

### Row L-1k — plain determinism pair (q3 anchors: 7.07 / 7.05 tok/s, wall 8.2 s, tok=58 finish=stop, digest pair e764922b0dfe1a7f)

```
# serve already up per §R1 (WORLD:4, devices 0,1,2,3, ws96, --no-prefix-reuse
#   --prefix-cache-capacity 256 --no-cuda-graph --greedy --default-max-tokens 1024)
printf '%s' 'Hi.' > «LABEL».body
python3 -c 'import json,sys; sys.stdout.write(json.dumps({"model":"qwen3.8-27b",
  "messages":[{"role":"user","content":sys.stdin.read()}],
  "max_tokens":1000,"temperature":0}))' < «LABEL».body > «LABEL».req.json
for rep in 1 2; do   # determinism pair, same boot
  curl -s -m 600 -w "http=%{http_code} wall_s=%{time_total}\n" \
    http://127.0.0.1:8098/v1/chat/completions -H 'content-type: application/json' \
    -d @«LABEL».req.json > «LABEL»_g1000_$rep.json
done
```
Budget arithmetic (printed, LITH): q3 observed wall 8.2 s natural-stop@58; worst no-stop case
1,000 tok ÷ 5.46 tok/s (slowest measured w4 datum, sustained legs) = 184 s + TTFT margin ⇒ CAP 600 s =
3.3× worst — a client TIMEOUT, no refusal constant anywhere (VRAM law). Capacity arithmetic: prompt 54
+ gen ≤ 1,000 + default-cap 1,024 ≤ **63,936** (37ec0ba1 measured `capacity=63936 tokens,
max-context=63936` @ ws96 WORLD:4, bin 3da20992 — quote the line verbatim from THIS boot's own log too;
the cite is a ceiling anchor, the boot's print is the datum).

**Expected datum:** two rows `rung 1000 leg {1,2} WORLD:4: http=200 kind=plain finish=? tok=?
tok_s=? prompt_tokens=54 sha16=? UFFFD=0` — digests EQUAL each other (same-boot pair law) = the nvfp4
first-decode mint; `tok_s` stated directly against the q3 pair **7.07/7.05** at identical
bin-family/artifact-role/world/flags. **FALSIFIER (C5-blessed):** if the pair prints
`e764922b0dfe1a7f` — the q3 digest — that is NOT a pass, it is the weights-not-loaded DEFECT at the
goal geometry (identical tokenizers + identical bytes across a format boundary = format not loaded);
bank everything, route to loader lane. Different digest = expected; open the mint record (§R4).

### Row L-40k — deep-prefill context-axis leg (geometry of the SF-1 repro / h40k session; NEVER yet COMPLETED on any artifact — this row is the first completion either way)

```
# WITNESS ARMED FIRST (PF-7 policy, this row only): python3 /home/chris/sf1_tools/vwatch.py \
#   /home/chris/sf1_logs/vwatch_«LABEL».csv 10 &   ... and a named awake-human at the chassis.
python3 -c "import sys; sys.stdout.write(('The quick brown fox jumps over the lazy dog. '
  * (160000 // 45 + 1))[:160000])" > «LABEL».body40k          # 160,000 chars, FILE end-to-end
python3 -c 'import json,sys; sys.stdout.write(json.dumps({"model":"qwen3.8-27b",
  "messages":[{"role":"user","content":sys.stdin.read()}],
  "max_tokens":40000,"temperature":0}))' < «LABEL».body40k > «LABEL».req40k.json
curl -s -m 2400 -w "http=%{http_code} wall_s=%{time_total}\n" \
  http://127.0.0.1:8098/v1/chat/completions -H 'content-type: application/json' \
  -d @«LABEL».req40k.json > «LABEL»_g40000_1.json
```
Budget arithmetic: prompt measured `prompt_tokens=35,607` (09c8c6d7 q3 datum, same body); prefill at
q3-measured 41 tok/s (stalled-boot log line, the last honest w4-q3 anchor) ≈ 869 s; decode ~58 tok
natural-stop @ 5.46 tok/s ≈ 11 s ⇒ ~880 s expected, CAP 2,400 s = 2.7× margin (timeout only). **KV
headroom note printed, not hidden:** 35,607 + gen ≤ 63,936 leaves 28,329 tok gen-room under the
40,000 request ceiling — the row reports what the server ACTUALLY does (natural stop, KV-pressure
stop, or finish=length); any of the three is a MEASUREMENT, none is retried away. Boot flags for this
session carry `--default-max-tokens 40024` (MAXGEN+24, runner :41/:58 semantics).

**Expected datum:** `rung 40000 leg 1 WORLD:4: http=200 kind=deep160000 finish=? tok=? tok_s=TOTAL +
early/late windows, prompt_tokens=35607 sha16=? UFFFD=0` + prefill-time split if the serve log prints
progress lines (quote them; TTFT-inclusive vs steady-state per grammar). Non-200 ⇒ capture-before-kill
(STALL()+wchan), bank RED, continue at 1k conclusion. Comparability limits stated on the row: q3 side
of this axis has ZERO completed legs (only the 41 tok/s prefill anchor + the SF-1 death + the chair's
≥43% walked repro) — the nvfp4 completion is the FIRST full-depth datum at 35.6k context on this
board either direction; the 1k pair remains the only same-geometry apples-to-apples anchor.

### Row L-10k — 10,240-token CONTEXT leg (chair seq-24, the user's tonight-number; added 13:5xZ BEFORE any session can reach it from memory)

**Honest definition first (what this row measures, and what it does NOT):** the q3 "rung 10000" banked on
this board is NOT a 10k-context datum — it was kind=sustained, prompt_tokens=85, a GENERATION-depth leg
that natural-stopped at 37 tok (5.46/5.47). There is therefore **no completed 10k-context q3 leg
anywhere** — the deep-leg family measured only toward 40k (and never finished). This row defines the
axis the user asked for: a 10,240-token PROMPT prefill + a decode window with early/late split (the
plan's windows law: state-growth cost measured, not asserted). It is NOT a generation-length claim —
the model stops where it stops (natural stop is honored, budget is a ceiling).

**Byte-provable comparability by construction:** body = the SAME generator string as the 40k deep row,
EXACT PREFIX at 46,009 chars (`startswith` verified this beat: the 160,000-char body literally begins
with this row's body — one stream, two truncations; comparability is structural, not coincidental).
Token target: 46,009 / 4.4935 measured chars-per-token (160,000 B -> 35,607 tok, banked pair) = 10,239
± row; **the boot's own prompt_tokens print is the datum, this arithmetic is the plan** (LITH both ways).

```
# FIRST-ATTEMPT GEOMETRY -> PF-7 chassis-awake policy binds (witness optional at <40k, awake-human
# not); session booted per §R1 with --default-max-tokens 1024 (MAXGEN 1000+24), then:
python3 -c "import sys; sys.stdout.write(('The quick brown fox jumps over the lazy dog. '
  * (46009 // 45 + 1))[:46009])" > «LABEL».body10k
python3 -c 'import json,sys; sys.stdout.write(json.dumps({"model":"qwen3.8-27b",
  "messages":[{"role":"user","content":sys.stdin.read()}],
  "max_tokens":1000,"temperature":0}))' < «LABEL».body10k > «LABEL».req10k.json
curl -s -m 1800 -w "http=%{http_code} wall_s=%{time_total}\n" \
  http://127.0.0.1:8098/v1/chat/completions -H 'content-type: application/json' \
  -d @«LABEL».req10k.json > «LABEL»_g10k_1.json
```
Budget arithmetic (printed, anchors from THIS board's logs): prefill decays with depth — measured
41 tok/s at 4% -> 21.0-21.7 tok/s through the 9-10k-token band (banked `prefill:` lines; the 40k leg's
own curve IS the 10k neighborhood's anchor). 10,240 tok prefill ≈ 366 s at the depth-mean, worst-case
488 s at the last measured rate; decode 1,000 tok at 5.46 tok/s (slowest measured) = 183 s ⇒ ~671 s
expected, CAP 1,800 = 2.7x margin — timeout only, no refusal constant exists in this row (VRAM law).
Four-part contract: prompt ~10,240 + gen ≤ 1,000 + default-cap 1,024 ≤ **63,936** (max-context named,
`capacity=63936` boot line quoted verbatim from THIS boot; gen-room headroom 52,672 tok — exhaustion is
NOT a branch here unless the boot's own capacity line moved, which is itself a finding naming the
allocator math).

**Branch set (§8ac shape, six arms):**
- **A-GREEN:** http=200, finish ∈ {stop, length}, tok_s printed TOTAL + early/late windows -> the
  FIRST full-depth 10k-context datum on this machine, either artifact's name; mint it.
- **A-THROW-NAMED-ORGAN:** warmup stayed GREEN but the REQUEST-path dispatch throws `nvfp4 linear:
  unsupported shape` (or a NEWER organ name) — INDICTS route-table coverage at prefill-chunk shapes
  the warmup geometry never exercised (add-a-measured-row family, SM-56 one layer down; warmup proves
  organ #1-#3 paths at ITS shape only). Triage: quote server words verbatim, name which call-site
  threw; this is table incompleteness, not physics — and if the text names a FOURTH organ, that is
  §8ac B10, one new row, pre-shaped.
- **A-UNGRACEFUL:** error/500/crash at depth with no named shape (state-growth class: GDN/KV footprint
  at 10k crossing some unwritten assumption) — capture-before-kill fires verbatim (STALL()+wchan),
  allocator words logged, 'none' is a logged word.
- **A-DIGEST-INVERSE (B1 carried to depth):** if this row's decode text is byte-equal to a q3 leg's
  text on the SAME prefix stream (the prefix property makes this CHECKABLE the moment any q3 10k-context
  leg completes — including tonight's twin, if the user's session runs both worlds), equality across
  the format boundary = weights-not-loaded at depth, DEFECT not pass — same C5 strict logic, no new law.
- **A-UFFFD>0:** pad-class at depth — same indictment as §8ac B6 (248,077/248,320 boundary) but now
  depth-conditioned, which is the attractor-at-depth question's ugly sibling; bank + route to pad gate.
- **A-HOST-DIES:** PF-7 witness line: this geometry is BELOW the 40k+ must but ABOVE nothing — if the
  host drops at 10k too, SF-1's "not load-proven" status DIES by observation (the electrical class
  extends down; deep-freeze widens), and the awake-human at the chassis is the instrument. Chassis
  presence is why this branch has a witness-optional flag, not a witness-exempt excuse.

**Rollup sentence for tonight (fill-or-VOID):** `nvfp4 @w4 10k-context: prefill «s» (vs q3 curve
41->21 tok/s measured), decode early/late «a»/«b» tok/s (PREDECESSOR q3 1k-pair 7.07/7.05 — named as
predecessor NOT prediction, different axis), prompt_tokens «boot print», finish=«», digest «», UFFFD=0,
contract: 63,936-MAXGEN named, chassis=«awake-name» — VOID if any «» unfilled.

### Rollup row (what the board quotes after this session, one paragraph, nothing more)
`nvfp4 @w4 1k: X.XX-X.XX tok/s (vs q3 @w4 7.07-7.05, same bin-family/flags/boot-shape, same artifact
ROLE, pair-equal digests «nvfp4_sha16» ≠ e764922b CONFIRMED [C5-form]), 40k-deep: «completed/RED»,
finish=<>, UFFFD=0, witness=<none|trace-sha>, mint record §R4 opened with BIN/TREE/GRANT/FORMULA
filled and C6 PENDING-HUMAN.` — every `«»` unfilled at print time = the row is VOID, not pending.

*(§R9 stamp: agent5, 2026-09-15 13:2xZ; q3 anchors re-read from banked hladder3/h40k runner outputs at
seat, artifact re-hashed whole this beat, runner script hashed read-only, witness process left alive
and flagged — zero card, zero build, shared tree untouched.)*
