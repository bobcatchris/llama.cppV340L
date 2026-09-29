# ITEM 6 (perf baseline): decode-anomaly first pass — zero GPU, log-derived

**Source:** `results/amd/p3/G17f_serve.log` (merged tree `246928b6`), the ONLY two gen/decode
rows in it (prompt=54 each):

    gen=4  prefill=4.28s decode=0.63s    -> <=0.157 s/token
    gen=32 prefill=5.72s decode=36.65s   -> marginal tok5..32 = (36.65-0.63)/28 = 1.287 s/token

**FINDINGS (log settles shape):**
1. Slowdown is MID-GENERATION and TOKEN-INDEX dependent: first four tokens <=0.157 s/tok,
   tokens beyond that average 1.287 s/tok marginal. Neither warm-up nor fixed-cost
   amortization fits (those move the AVERAGE the other way; agent2's #698 reading, confirmed).
2. `full_reset` hypothesis REJECTED as primary cause: once-per-request cost amortizes to
   <=0.04 s/token across 28 tokens; cannot produce a 1.29 marginal.
3. Remaining candidates: per-token re-prefill engaging after a few tokens, or a paging/
   eviction path, or a mode switch (graph-off fallback per-token cost at this geometry).
   Distinguishing them needs PER-TOKEN timestamps, which this log does not carry.

**NEXT decisive check is on-card and instrumented:** agent2's `tools/v340l/tps_probe.py`
(on main since a9d38792, both-direction stub-proofed, prints no numbers it can't source)
at the 2048/kvarn_k5v4 geometry agent2 held stable >230 s — that requires a written grant
request; will file one when 17g's release row frees the traffic picture. No card time
before then. One-boot-per-truth; marginal not average, as ruled.

**Block-sentence amended per #733(3) the hour it landed** ("every blocked-on-X owes: did I
check X is the only place X can happen"): the original sentence gated on dev0/1's release
row by old pair convention. Equivalence is MEASURED — chair's showtopo: all four dies
weight-40/2-hop/PCIe, dev2/3 desktop-free (18.5 MB baseline) — and TP2 geometry needs any
two equivalent dies, so the probe is NOT blocked on dev0/1: target pair = dev2/3 the moment
17g3's row returns them to pool (or any pair the chair stamps), grant request filed at that
point, not before. Blocked-on remains true only for the WRITE of the grant, which is the
chair's, which is where it should stay.

Citation: figures at predicate "rows in G17f_serve.log @ 246928b6"; the 7.3x in #698 is
the average-vs-average pair; this note's 1.287 is the marginal — same data, named predicates.

## Appendix: the 9174 phantom — #740 VERIFIED at main's bytes + the sibling path

agent3's headroom claim (#740) fully confirmed at my seat (`246928b6` archive): `:54`
`headroom_bytes = 1024 MiB` ("reserved as margin", unconditional) → `fixed_bytes()` :85-89
sums it → `:281` `usable <= fixed_total → return 0` = refusal. Composition 7102+592+96+200+
160+1024 = **9174 exact** (agent4's log's own printout); without it 8150 vs free 8160 = fits
by 10 MiB. Instance #6 of the banned class, as filed. The file QUOTES the law at :40-44 to
justify `static_weights_bytes` as fallback-only — the same reasoning simply never reached
the adjacent constant.

**SIBLING, verified earlier at my audit seat (tp_engine.cpp):** a SECOND false-refusal path
at the same geometry — :870-873, placement-basis reader THROW → catch prints "falling back
to legacy" → `legacy-estimate` basis (the 9059 constant) charges against cards that measured
7102 feasible. Agent2's attempt-1 refusals ran through THIS path (truncated artifact =
throwing reader = estimate revived). Fixing headroom_bytes while leaving the catch
estimate-fallback means: healthy manifest → fixed; THROWING manifest (truncated/corrupt
artifact) → estimate-gated refusal again, misattributed to "stale-residue cards" by the
same misdirecting message. Per the law, the conformant shape for both is identical: refuse
nothing on constants — measure (allocator-is-the-gate), and print the TERM COMPOSITION of
any capacity number so a near-miss reads as a near-miss, not a trespass. Review test for
whichever fix lands: synthetic near-capacity cell must LAUNCH and MEASURE (AGENTS.md,
verbatim). Not my file — filed here as third-seat witness for agent4/coord's call.

## Second pass (item-6, zero GPU): the anomaly does NOT reproduce in the merged tree

**Sources (stamped):** G17f rows in `results/amd/p3/G17f_serve.log` @ `246928b6`; 17g3 rows —
cited at the time from `/tmp/g3_serve.log` (sha1-prefix `bb984de4`); SUPERSEDED per #797: the
canonical citation is the banked `results/amd/p3/G17g3_serve.log` @ `bc122675` (the /tmp file was
later clobbered by the g5 boot — same-kit instrument defect, see Third pass §2).

    run      gen=4 decode   gen=32 decode   marginal tok5-32   marginal/first-4
    G17f     0.63 s         36.65 s         1.287 s/tok        8.2x   <- the cliff
    17g3     0.63 s          6.48 s         0.209 s/tok        1.33x  <- flat

1. **The cheap regime is run-invariant**: gen=4 decode = 0.63 s in BOTH logs (identical to
   the printed digit) — tokens 1-4 cost the same old-tree-on-dev0 and merged-tree-on-dev2/3.
2. **The mid-generation cliff is GONE**: marginal 1.287 -> 0.209 s/tok (6.2x), and 0.209 is
   corroborated internally by 17g3's own progress print (4.8 tok/s at 25/32). The per-req
   ~4.x the chair read = this marginal. Window-vs-per-req gap narrows 17x -> 2.4x.
3. **ATTRIBUTION HONESTLY LIMITED**: 17f->17g3 changed THREE things at once (tree contents,
   card pair, hour). Non-reproduction is settled; WHICH change killed the cliff is not, and
   cannot be from these two logs. Mechanism candidates (per-token re-prefill / paging engage /
   mode-switch) are MOOT for main — no cliff to chase there; the identification only matters
   if the cliff ever returns, at which point the two-row marginal test (this table's shape,
   zero extra card cost on any serve log that already carries gen=4 + gen=32 rows) is the
   standing first check. Review test still owed per board law: tps_probe.py multi-gen sweep
   (5/17/32/64) under a written grant, to confirm flatness beyond 32 tokens — 32 tokens is
   inside kv-cap 260; the engage-after-N hypothesis predicts the cliff, if it ever existed,
   had an N near 5 and 32 is already past it, so gen-64 is the load-bearing new point.
4. The decode-anomaly item as ROUTED (#698 "check the marginal") is CLOSED with agent2's
   finding promoted from open-defect to fixed-and-verified-in-shape; the perf BASELINE proper
   still waits on real-context cells under grant — smoke shapes are cited as smoke shapes.

## Pre-registered analysis: environment-sensitivity (17g3's 107-char divergence -> 17g4 swap)

No action until agent4's 17g4 row (ruled). Method filed so the data answers in one pass when it lands:
1. Inputs: the kit's per-boot raw token/logit dumps, 17g3 (launch-labels 0,1 = physical dev2,3)
   and 17g4 (swap 3,2, option A), plus both release rows' rank-baseline tables (the 18-vs-8 MB
   second-rank delta is the candidate cause).
2. Test: byte-diff the token sequences; locate FIRST-DIVERGENCE INDEX in each; cross-reference
   divergence position against the rank-baseline asymmetry (if divergence position is STABLE
   under card swap -> content-determined attractor, environment secondary; if it MOVES with
   physical slot -> environment-asymmetry confirmed, exactly the 17g4 stamp's pre-declared split).
3. Predicate discipline for the number: "107 chars" names WHICH run-pair, WHICH scanner
   (char-vs-token vs logit-ULP), and which comparison baseline; the greedy-determinism-otherwise-
   intact claim constrains scope (single sequence, single prompt, no sampling).
4. If 17g4 reproduces 17g3's output: divergence is pair-stable -> file as this pair's answer,
   move on to gen-64 marginal flatness (the grant cell). If it follows the physical card: the
   perf baseline proper MUST name its pair per row — pair-independence is now an open variable,
   not an assumption. Either branch is a result; this file gets the table, not prose.

## WO-VRAM-1 addendum: the third constant verified + the removal-ORDER trap

Chair's fourth wheel CONFIRMED at bytes: kGateReliefBytes = 1200 MiB @ tp_engine.cpp:1067,
feeding the refusal gate as `claimed − relief` (:1069), user-signed ledger provenance in the
printout (:1071). WO doc 14367878 resolves; all three constants now seat-verified across
three seats.

**The trap in "take the constants off the decision path"** (for the implementer, from the
comment block's own history): the relief votes LENIENTLY and exists precisely because a
claimed-estimate gate false-refused launches that then ran (16,679 refused vs 15,069 real,
int8@200k — the ledger rows at :1059-1061). Deleting the 1200 while LEAVING a
claimed-basis gate makes the gate STRICTER and resurrects the VRAM-LAW's named recurrence
class — the opposite of conformance. The conformant move is the one the WO words correctly:
the BASIS goes measured-only (allocator decides, cudaMalloc is the refusal), and then the
relief has nothing to vote on and dies with the basis, not before it. Order matters: kill the
estimate-gate first; a constant's removal-direction determines whether it was a bug or a
bandage. Headroom (:54) voted STRICTLY — delete-and-measure is monotone-safer there; the
relief voted LENIENTLY — its deletion alone is a tightening, tightening is refusal, refusal
by constant is the crime. Same law, opposite polarity: report measured-only, allocator
decides, and the near-capacity synthetic cell must LAUNCH.

## Third pass (#754): digit-chains named, a rounding self-correction, a row-provenance split, and the gen-64 cell as proposed geometry

**1. Self-correction, third-decimal:** my 1.287 was mis-rounded. Named chain, no intermediate
rounding: (36.65 − 0.63) = 36.02; 36.02 / 28 = **1.28643** → quote 1.286. The chair's 1.286 was
right; my file's 1.287 gets amended here (earlier text stands as written, this row supersedes —
annotate, never delete).

**2. Row-provenance split (same label, two artifacts — the family continues): RESOLVED by the
chair's #797 ruling — the mechanism was a KIT INSTRUMENT DEFECT, not a relay error.**
`G3_bringup_kit.sh:32` redirects EVERY boot's serve log to the same `/tmp/g3_serve.log`
regardless of LABEL — my cited file (sha1-prefix bb984de4, the 17g3 boot, mtime 06:02) was
CLOBBERED by the 17g5 boot (mtime 06:20; the file's current content reads 0.63/6.49 with the
g5 request line, sha1 now a0f76192…). Both readings were honest at their moments; a /tmp NAME is
not a citation. Banked truth: committed `results/amd/p3/G17g3_serve.log` @ `bc122675` carries
`gen=4 decode=0.62s / gen=32 decode=6.42s` (line ~710, chair's rows verified at my seat — the
"those rows are NOT in my file" sentence was true of the FILE and false of the CORPUS). And my
`8d090f69` dead-sha flag was a ONE-DIGIT transcription: the verdict row is **`8d090f6b`**, resolves
on origin/amd/wo-p3-serve, `git cat-file -t` = commit, verified at my seat this pass — the e/f
disease, caught by my own cat-file-before-declaring law applied belatedly. Board law adopted (the
chair made it with my ghost-file line as the precedent): cite banked path+sha, never a /tmp name;
kit fix (per-label log path) directed to agent4's queue, not mine. Both value-pairs give the SAME
verdict (flat ~1.33x) so nothing load-bearing moved — the provenance is the whole finding:
    mine:   (6.48 − 0.63) = 5.85; 5.85 / 28 = 0.20893 → 0.209 @ bb984de4 [file since CLOBBERED — superseded]
    banked: (6.42 − 0.62) = 5.80; 5.80 / 28 = 0.20714 → 0.207 @ results/amd/p3/G17g3_serve.log, bc122675
The 4.8-vs-4.83 corroboration survives because both round to the log's own progress print; the
DISCIPLINE outcome is the point: every serve-log figure carries file+sha1-prefix with its row.

**3. GEN-64 CELL — proposed exact geometry (rides agent2's window; reader-only, zero boot,
zero reservation; tps_probe.py @ main, stream-timed, both stub directions proven).
STATUS per chair #797 2026-09-13: **APPROVED AS PROPOSED** — attaches to agent2's window as a
zero-boot reader add-on per the standing plan; written grant request naming host:port when fired,
marginals per pair. Do not fire before agent2's window announcement.**
- Arm A (smoke-comparable, continuous with the 17f/17g3 rows): --prompt-tokens 54, gen sweep
  {4, 32, 64, 128}, --repeats 2 each. Marginals by consecutive pairs, division chains printed.
  Load-bearing NEW coverage: 32→64 and 64→128 (beyond every observed point; the boot log shows
  5 kv pages at cap 260 — gen-128 forces page growth, so an engage-after-N paging mechanism
  with N in (32,128] can ONLY be caught by this arm).
- Arm B (agent2's stable geometry, high-prompt regime): --prompt-tokens 1800 (fits their
  ctx-2048+k5v4: 1800+128=1928 ≤ 2048), gen {32, 128}, --repeats 2. Tests whether the flat
  marginal holds where GDN/prefix state is actually working.
- Flatness rule, pre-declared: marginal(32→128) within ±15% of marginal(4→32) = flat; outside =
  the cliff relocated and dated. Repeats give the run-to-run variance the margin-stable
  language needs. Total read time ~2-3 min inside whatever window agent2's user-task holds;
  host:port named when agent2 announces the live server. Written grant request per chair's
  #753 sequencing: reader-add-on to THEIR stamped boot, no card reservation of my own.
- Both branches are verdicts: flat -> "~4.8 tok/s marginal, prompt- and length-stable through
  128" becomes the board's first honest decode number on merged main; cliff -> mechanism list
  (#698's three) reopens WITH the N-value located, which no average ever prints.

## §method field-lesson (17g4 third outcome, #763): divergence numbers need their comparison FIELD named

Re-derivation at my seat, one JSON in hand (`/tmp/g3_response.json` sha1-prefix 165d2343):
the chair's three first-divergence figures (17f↔g3=107, 17f↔g4=15, g3↔g4=15) are **internally
consistent** (17f can share g3's prefix to char 107 while both fork from g4 at 15) — the
triangle is not the suspect. My only 17f-labeled artifact (`/tmp/17g_baseline_17f.txt`, a raw
**content string**, not a JSON) is **162/162 BYTE-IDENTICAL to g3's `reasoning_content`** —
which makes d=107 impossible IF that file is 17f's true output. So either the baseline file is
a regenerated/extracted copy (name says baseline, provenance unknown) or the 107 lives in a
field my file doesn't carry. **Lesson generalized from the digest family to divergence
numbers: `first-diff(char)` is meaningless without naming the comparison field (raw JSON bytes
include `created:`/`usage:` stamps -> position-of-timestamp divergences dominate content
divergences; content-field-only -> my 162/162 identity).** 17g5's machine-verdict row should
declare WHICH field was hashed, plus publish the three JSONs' paths+shas so the numbers are
re-derivable at other seats — pre-registration means re-derivable, not just reproducible.
Within-run determinism test (two requests, one process) remains IMPOSSIBLE from g3's log:
5 progress lines, one request (1-based ids) — the replay IS the experiment. Slot held.

## ESCALATION SEED (17g5 DIFFER confirmed; my pre-declared branch) — full divergence matrix re-derived at third seat

Machine-decided matrix, content-field, sha256-prefix per response, all four artifacts at my
seat (archived bankings @ agent4's lane refs; comparison FIELD named per my own law):

    17f  = 348e77a1222d (162 ch)   g3 = a0b2a5a518ce (152 ch)
    g4   = e2b4a3f41581 (134 ch)   g5 = 348e77a1222d (162 ch)  == 17f BYTE-EXACT

    first-diff(content): 17f↔g3=107 · 17f↔g4=15 · g3↔g4=15 · 17f↔g5=0 · g3↔g5=107 · g4↔g5=15

1. **The chair's triangle (107/15/15) CONFIRMS at my seat** — his numbers were right, and
   the "un-locatable" gap from my last pass is resolved by the ugliest mechanism of the night:
   my /tmp/g3_response.json was a MISLABELED ARTIFACT (162-ch content == 17f == g5-era capture,
   file named g3 at 06:20Z). The BANKED G17g3_response.json (152 ch) diverges at 107 as ruled.
   Ghost-file class: an unprovenanced /tmp artifact outranked the banked corpus by name alone.
   The lesson stands generalized: cite the BANKED path+sha, never a /tmp filename.
2. **Streams tally: THREE outputs across FIVE coherent boots** {17f-stream (hit twice: 17f,
   g5), g3-stream (once), g4-space-parse (once)}. Identity claims are FIELD-SCOPED: g5==17f is
   byte-exact at content-field; their raw JSONs still differ at created:/usage: — the
   field-naming law is load-bearing for the escalation's own headline sentence.
3. **Order-as-determinant REFUTED by the matrix itself**: dev3,2 produced BOTH the space-parse
   (g4) AND a bit-exact 17f reproduction (g5); dev2,3 produced the stylistic third (g3).
   Pair maps to nothing stable; agent4's timing-dependent-state pointer (arena zero-fill /
   comm-init ordering, warm-cycle correlation) is the surviving hypothesis, and early-token
   localization (char 15/107 = first reasoning span) bounds WHERE in the pipeline to hunt:
   pre-first-logit state, not mid-decode numerics.
4. Item-6 consequences filed with the branch that closed: cross-replay 'same token' identity
   is dead as the chair's row says (within-stamp bar only — the coherent/zero-fault/arms-traced
   bars survive untouched, and every receipt in my lane is exit-code/static/within-run form,
   none required cross-boot identity); the gen-64 marginal cell is UNAFFECTED as a measurement
   (marginal cost doesn't require identical tokens, only same-geometry timing) but its
   PRESENTATION must carry per-boot token-count variance, and agent2's window is now even more
   valuable: it would add request-count variance on top — the same-process second-request test
   my log-pass showed never happened in ANY boot to date.

## Self-corrections from #774 at my own seat (annotate, never delete)

1. BOOT COUNT: my 'three streams, five boots' double-counted — the mislabeled /tmp g3 file and
   the banked g5 response are ONE boot. Truth: FOUR boots (17f, g3, g4, g5), THREE output
   classes, g5==17f byte-exact; the chair's matrix at his desk matches mine at char indices
   (only the boot count differed, by my artifact-confusion). Corrected tally supersedes.
2. CYCLE CLASS IS NOW MANDATORY on every item-6 row: warm boots print LOWER prefill
   (11.7-12.6 vs cold 16.5 tok/s), decode spread 6.9-9.8 SAME CONFIG — so my gen-64 cell
   inherits agent2's window class UNKNOWN until stamped, and any marginal it yields must be
   labeled cold|warm|unclassified; the two-row marginal test gains a cycle column. My earlier
   '0.209 marginal' from g3 is therefore cycle-unclassified historical, quotable with that
   caveat — which is precisely the predicate discipline the sheet preaches; it binds its
   author's numbers first.
3. The intra-server 5x-repeat decisive test is folded from my open-slot list to ITEM 7's
   seat (agent3) — same datum, now owned, not pending on me. Item 6 stays the marginal/TPS
   line; item 7 owns which state selects which stream; the boundary is drawn at the chair's
   ruling, and the determinism corpus (agent4's rows) is the shared substrate.
