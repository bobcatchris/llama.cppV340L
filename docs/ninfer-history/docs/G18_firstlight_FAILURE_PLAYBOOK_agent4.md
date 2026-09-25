# G-AMD-18 first-light FAILURE PLAYBOOK (agent4, 2026-09-14 ~12:2xZ) — zero-card, pre-staged

Purpose: tonight's forensics on the load-refusal took hours on exactly these unknowns. Each branch
below is a NAME + a DECISIVE CHECK that is already written + the CONCLUSION each outcome licenses.
Run the check BEFORE hypothesising; do not re-derive a branch that has already been excluded.

Order of operations after a bad boot: **F0 → then exactly one of F1..F5.** F0 exists because the most
expensive failure tonight was reasoning from a log whose provenance was never checked.

---
## F0. PROVENANCE — is the artifact I'm analysing the one that ran? (do this first, always)
```
BIN=<the .bin you booted>; sha256sum "$BIN" | cut -c1-16        # must equal the filename stamp
grep -m1 -aoE "world=[0-9]+|ranks=[0-9]+" results/amd/p3/<LABEL>_serve.log
ls -la results/amd/p3/<LABEL>_manifest.txt                        # written BEFORE spawn?
```
Licenses: if the sha doesn't match the filename, NOTHING below applies — you are analysing a
different binary than you think (the pinned-bin law exists because a lane build path is a moving
target). If no manifest exists, the boot was ungranted and the row is void regardless of outcome.

---
## F1. INIT HANG / no "Ready for requests" → the RCCL SHAPE law, not a capacity problem
Decisive check (zero card, ~5 s): did every rank's comm init come from its OWN thread?
```
# 1a. is the allgather issued from the per-rank worker (safe) or hoisted to one thread (deadlock)?
grep -nE "ncclAllGather|allgather_and_local|allgather_r1" src/core/multi_gpu/tp_group.cpp
grep -nE "dispatch_all|worker = std::make_unique<std::thread>" src/core/multi_gpu/tp_group.cpp
# 1b. did any rank reach the ring's own entry (means it did NOT hang in init)?
grep -ac "rank must be 0 or 1" results/amd/p3/<LABEL>_serve.log
# 1c. foreign occupancy at the time (a stale peer looks exactly like a hang)
rocm-smi --showpids 2>/dev/null | awk 'NR>3 && $1 ~ /^[0-9]+$/'
```
Conclusion mapping: hang with 1b=0 and no KFD growth ⇒ **shape violation** — init serialised across
one thread. This box has ONE proven-working shape (agent5's B-1 shape #4: per-rank `std::thread` +
`blocking=1`, 4/4 ranks, rows `G18B1_b1_shape4_*`). Do NOT debug transport internals until the
threading matches that shape. Hang with 1c nonzero ⇒ foreign occupant, not your bug: report, don't kill.
VRAM-law note: a hang is never evidence of a capacity problem; capacity reports itself via a clean
`hipMalloc` failure. If you find yourself writing "it probably didn't fit", stop — that is the banned
estimate class and it has cost this project CI runs and user hours.

---
## F2. rc=139 / Segmentation fault → the R1 null-deref class (PRED-D should have caught it pre-boot)
```
# witness comes from the BANKED copy at the tip under test, never a working copy:
git show HEAD:tools/ops/check_ring_deref_guarded.py > /tmp/witness.py && sha256sum /tmp/witness.py | cut -c1-8
python3 /tmp/witness.py src/core/multi_gpu/tp_group.cpp; echo "rc=$?"
# which ranks died, and did the ring's OWN message ever print? (it must NOT have — see below)
grep -aoE "rank must be 0 or 1" results/amd/p3/<LABEL>_serve.log | head -1
```
Ground truth, verified at bytes so nobody re-derives it: the ring is constructed **only** under
`if (I.n == 2)` (`tp_group.cpp:115-117`), so at world=4 `one_shot_argmax` is null on **all four**
ranks. An unmasked unconditional deref therefore SIGSEGVs every rank at the call site. The
"`ranks 0/1 segfault, ranks 2/3 stale :434 throw`" split that circulated at 12:05Z is **impossible**:
`:434`/`:435` are inside the method, unreachable through a null object. A boot log showing rc=139
and NO `:434` text is fully consistent with the null-deref class — do not go hunting for a throw
that cannot print.
**Witness stamp, and how to read it (the F2 check is only as good as the cell it runs):** the
banked cell is `tools/ops/check_ring_deref_guarded.py`, current reviewed content8 **1a3dd6d2**
(chair `087a797e`, 6/6 fixtures both-directions). Two earlier stamps are KNOWN-INCOMPLETE and must
never grade the source: `9e4475de` (line-scoped guard → false-RED on a legal brace-on-next-line) and
`2e47fe01` (missed the guarded-but-**no-return** double-writer, where both the ring and the R1 arm
write `out_token`). Resolve the witness through git rather than a path — `/home/chris/agent3_cells/…`
is the author's working copy and changes under their own edits, so a row quoting one stamp can
silently execute another. The runner's GATE 5 enforces this; if you are checking by hand, compare the
sha8 you get against that list before believing a GREEN.

If rc=139 appears WITH `:434` text present, that is a DIFFERENT bug: someone constructed the ring at
world>2 and ranks 2/3 entered it. Then check whether the rank guard was deleted (PRED-D2 in the
runner) — that path is silent-wrong-token territory, not a crash, and is worse.

---
## F3. Token MISMATCH vs the world=2 attractor → COORDINATE class, not "argmax is wrong"
The attractor is `348e77a1222dea7f` (measured, world=2, `--gating-trace`, greedy, same corpus).
```
# 3a. did the ring already go silent at world=4? (the D1/D2 accessor class — see F4 first!)
CENSUS_STRICT_KINDS=1 bash tools/census/read_census.sh results/amd/p3/<LABEL>_serve.log | grep -aoE "kind=[a-z-]+ [0-9]+"
# 3b. coordinate test: is the winner a plausible LOW index (shard-local) or full-vocab-range?
grep -aoE "MC31.*win=[0-9]+" results/amd/p3/<LABEL>_serve.log | grep -oE "win=[0-9]+" | sort -u | head -6
python3 -c "print('shard width at W=4 =', 248320//4, '-> local-only winners cluster under that')"
```
Reading: `3b` winners all **< 62080** while the true global winner is elsewhere ⇒ shard-local
coordinates leaked into `out_token` (the law is `.tok` global, remap LAST —
`one_shot_argmax.cu:218`/`:319`). Winners in full range but still disagreeing with world=2 ⇒ check
the tie-break DIRECTION (`val desc, tok asc`; a `<`/`>` flip is byte-identical text and different
tokens) and that the draft-vocab remap is applied once, not twice.
Do NOT conclude "R1 argmax is broken" from a mismatch while F4 is unresolved: if the conf/step
accessors are returning silent zeros at world=4, part of your evidence stream is fabricated.

---
## F4. CONF mismatch / conf == 0.0 → ACCESSOR SILENCE before SUMEXP MATH (the ordering is the point)
agent3's finding, five rank-guards, verified at bytes:
`:605 → return 0.0f`, `:619 → return 0`, `:624 → return false`, `:629 → return 0.0f`, all guarded
`rank != 0 && rank != 1` and **returning quietly** — plus `:434` the throw. At world=4 these are
reached by our own instrumentation (`tp_group.cpp:353-356`, and `tp2_backend.cpp:2308` feeding
`armed_conf_base`), so **conf=0 / step=0 / timed_out=false are the expected silent values, not
measurements.**
```
# 4a. is the conf you're reading even from a live accessor? count accessor calls vs zero values
grep -aoE "conf=[0-9.]+" results/amd/p3/<LABEL>_serve.log | sort | uniq -c | sort -rn | head -5
# 4b. only THEN test the sumexp math on the values that are not suspect
grep -aoE "my=\(vbits=[0-9a-f]+,tok=[0-9]+\)" results/amd/p3/<LABEL>_serve.log | head -4
```
Rule: **no world=4 row may cite conf/step/timeout as a measurement** until either (i) the four
accessors throw like `:434` does, or (ii) a world-general accessor exists. Until then a zero is a
property of the instrument. This is my own thermal-line bug (`0 edge sensors`, box showed 4) promoted
into the runtime: an instrument that cannot see its subject must not emit a value.
If 4a shows non-zero confs (i.e. the accessors were fixed), THEN the sumexp branch is live:
`conf = 1/Σ exp(val_r − M_GLOBAL)·sumexp_r`; a **local** max shift makes the answer rank-dependent —
`argmax_reduce.h`'s arm 5 asserts the global-max value differs from the local-max one, so re-run
`/tmp/standalone`-style: `g++ -std=c++20 -I src tests/multi_gpu/tp_argmax_reduce_host.cpp -o /tmp/c && /tmp/c`.

---
## F6. LOAD-TIME OOM with preflight placement << materialized ⇒ **LOADER PAIR-MATH, not capacity**
The banked specimen is the 14:41:55Z world=4 fire: `results/amd/p3/G18w4_serve.log` (in agent5's lane,
`/home/chris/worktrees/amd-wo-r1-transport/`; the runner's step-0 verdicts and output are
`G18w4_step0.txt` / `G18w4_runner_output.txt`). Verbatim, both numbers from the same log:
```
[tp2] auto KV: free=8160 MiB (min both ranks), placement=3820 MiB/rank (measured-manifest)
[preflight] live-free dev0..dev3: 8160 MiB each          <- FOUR dies, MEASURED, world-derived
[rank 0] materializing text-only (mtp/ filtered) sharded model (TP2, MTP k=0) on device 0
[rank 0] materialized: 7102 MB device (capacity)          <- PAIR number, at world=4
```
**Decisive check (zero GPU, no re-run, works on any log):** compare the two printed numbers.
```
grep -aoE "placement=[0-9]+ MiB/rank|materialized: [0-9]+ MB" <serve.log> | head -4
# tell: materialized/rank  >>  placement/rank  => the loader sized its arena from PAIR math.
# Here: 7102 vs 3820. And the load line itself says `TP2` at world=4 -- the tag contradicts the
# device count in the SAME log, which is the whole branch in one grep.
```
**Why this is not a capacity question, and cannot be argued as one:** every capacity number in the log
is MEASURED (8160 MiB free on four dies, preflight placement 3820 MiB/rank), and 7102 would have
FITTED 8160 — the OOM came from what the loader allocated on top of a capacity it sized for two
ranks, not from a card that was too small. **VRAM law applies exactly here: reading this as "4 cards
can't hold it" is the banned estimate class, and the log disproves it.** The correct conclusion is
Class-2 (world-derived sizing missing in the loader), i.e. the `tp_load` lineage.

**TWO TRAPS IN THE TELL, both from my own verification pass — read these before quoting it:**
1. **UNITS DIFFER BETWEEN THE TWO LINES.** preflight prints **MiB**, materialized prints **MB**
   (`%zu MB device (capacity)`). A naive numeric compare is unit-ambiguous; for the ratio conclusion
   it is irrelevant (7102 vs 3820 is ~1.86x either way), but for any arithmetic that decides a
   boundary, convert. Do not publish "7102 MiB".
2. **DO NOT ASSERT "materialized is exactly half the artifact."** It is not, and the mismatch is
   informative rather than disqualifying: 15,446,796,288 B = 14,731 MiB, while 7102 x 2 = 14,204 MiB
   — **96.4%, not 100%**, because rank 0 loads *text-only* with `mtp/` **filtered** (see the load
   line). So the check is `placement << materialized` plus the `TP2`-at-world-4 contradiction, NOT
   `materialized*2 == artifact`. Anyone using the exact-halves form will call a real instance a false
   positive the first time a target filters tensors.

**Landing site to check when the branch is opened:** the printed capacity is
`mat.stats().device_capacity_bytes` (`tp2_backend.cpp:689`), which is `plan.device_capacity_bytes`
(materializer.cpp:115/129) — i.e. the pair math is in the TP plan/materializer, one level above both
files I just named. `materialize_tp(reader, rank, world, ...)` receives `world`, so verify whether it
uses it or re-derives 2.

**Where this branch goes:** it is the same class as GATE-3's hardcoded `n_vocab=248320/2` and the
`rank0_/rank1_` pair A-4 replaced — a world=2 assumption living in a shared path. Agent5's class-guard
cell (`materialized == placement at any world`) is the right permanent form; it belongs in the boot
battery, not only in the host suite, because it is only decidable against a real load.

---
## F5. Nothing above, but no serve → the two GATEs firing as designed (these are RESULTS)
*(if the log shows an allocation failure at load instead, go to F6 — it is neither of these two.)*
```
grep -aoE "no argmax transport at world=[0-9]+.*|destination too small at world=[0-9]+.*" results/amd/p3/<LABEL>_serve.log | head -2
```
After R1 lands, GATE-2 must be **gone** from the greedy path (that's the leg). `destination too small`
= GATE-3 stated capacity, which is an admissible refusal and means the R1 staging is mis-sized — a
real finding, reportable, not a failure of the window. If BOTH texts are absent and there's no
"Ready", you are in F1.

---
## Cross-cutting tells, cheap to check and each has burned hours this cycle
- **A stale card looks like a code bug.** `rocm-smi --showpids` before and after; kill only your own pgid.
- **`grep -c` returning 0 is an instrument reading, not a fact about the world.** Twice tonight a
  "site X is absent" came from my own malformed command (a `\\\\`-escaped class, and `^\s+` against
  text at column 0). Confirm with the raw line before any conclusion of ABSENCE.
- **The naive PRED-A form is a false RED** — `grep "devices.size() == 2"` matches a COMMENT at
  `engine.cpp:311`. Use PRED-A′ or run the cell.
- **Latency ceiling, not prediction:** expectation ≤ 129.15 µs (w=4 @ 10 KiB bin) for a 256 B/rank
  exchange, ~39× below the grid's smallest bin. An overshoot is a tiny-message-overhead finding.
- **`df` deltas are unattributable after the fact** — 10,646 deleted-open handles on this box. If you
  want to name a disk change, sample `lsof +L1 | sort -k7 -nr | head` AT the event.
