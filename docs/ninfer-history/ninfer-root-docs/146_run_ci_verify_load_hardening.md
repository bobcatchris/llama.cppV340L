# 146 — Hardening the `run_ci.sh` verify-data load (agent2)

> **Renumbered from docs/147 on 2026-09-04** (coordinator ruling seq 21). This document
> **was authored as docs/147; commits `8c4cf7d3` / `f32c4dbf` / `5cee82ce` refer to it under
> that number** — commit messages are immutable, so the pointer line is the reconciliation.
> `docs/147` now means agent1's `147_2a_debrief_round3.md`. The `results/141_gate_verify/`
> scripts kept their `146_` prefix so the results namespace follows the doc, not the
> debrief number.

> **Type:** CI-plumbing fix. No GPU, no feature code, no numerics.
> **Assigned by:** coordinator, seq 12. **Gates D5** — `d5_go.txt` is not stamped until this lands.
> **Scope:** `tools/ops/run_ci.sh` report block only. Explicit pathspec per team commit rule.
> **Fixer-certifies concern:** acknowledged and covered by protocol — after my D5 pass the
> coordinator independently re-runs D5 and re-verifies the verdict.

## 1. Trigger

While pre-flighting D5 I found that a file **I** landed (`results/4b_decode_20260903_145944.json`,
commit `2bff1d5d`, already merged to `wo/kv-uniform`) would have made D5 report a **false PASS**.
Reproduced against the real code path, not inferred.

## 2. The original code

`tools/ops/run_ci.sh:556-559`:

```python
verify_jsons = sorted(glob.glob(f"{results_dir}/[0-9]*.json"))
verify_data = {}
if verify_jsons:
    with open(verify_jsons[-1]) as f:
        verify_data = json.load(f)
result["verify"] = verify_data
```

### Three defects, and the reason they compound

1. **`sorted(...)[-1]` over `[0-9]*.json` is an ASCII *name* sort, not a recency sort.** Any
   filename whose first digit is greater than `2` sorts after every `2026…` timestamp. My
   `4b_decode_…` file (`4` > `2`) therefore *became* the loaded file, silently displacing the real
   verify output.
2. **The pattern also matches `run_ci.sh`'s own output, `<ts>_ci.json`**, because `_` (0x5f) sorts
   after `.` (0x2e). Demonstrated on this repo's real `results/`: the old selector resolves to
   `20260903_124533_ci.json` — a CI *summary* — where the actual verify output is
   `20260903_124533.json`. In a normal live run the fresh verify file usually sorts last anyway
   (its timestamp is newer than the previous run's `_ci.json`), so **this defect is largely latent,
   not actively wrong on every run** — stating that precisely rather than overclaiming. It becomes
   real whenever the two share a timestamp prefix, and it makes the selector's contract
   ("the newest verify result") unenforceable.

   **Latent — now confirmed empirically, per coordinator ruling seq 13.** Historical audit
   (`results/141_gate_verify/146_historical_load_audit.py`, read-only, CPU-only) compared every
   recorded `<ts>_ci.json` against its sibling `<ts>.json` across **all 13 worktrees + `repo`**:

   ```
   testable pairs          : 152
   primary load mismatches :   0     (1036+ keys compared per the two largest trees)
   decision-affecting      :   0
   ```

   Every recorded `_ci.json` has a nested `verify` block **identical** to its sibling verify output,
   so the old selector loaded the correct file in every run still testable. **No historical verdict
   needs re-verification; this does not block the GO.** 25 pairs were skipped for a missing sibling
   (pruned files) — that is absence of evidence, not evidence of correctness, and is stated as such.
   Fixed anyway as a side effect of (b): the strict pattern excludes `_ci.json` by construction.
3. **No error handling, and the block's exit code was discarded.** This is what turns defects 1–2
   from "wrong data" into "false green": the final `RC` (`:725-733`) is computed *only* from the
   shell step exits (`VERIFY_EXIT`, `SERVE_EXIT`, …). So when `json.load` raised on my JSONL file,
   the block died, `${TS}_ci.json` and `latest.json` were never written, and the script **still
   printed `CI: PASS ✓`** — while `determinism_ok`, `a2_identity_ok`, `kv_i8_ok` and the
   `mtp_accept_pct` regression gate (`:690-699`) were never evaluated.

**Exposure at time of writing:** three trees carried the bad file (`wo-kv-uniform`,
`wo-2a-batched-serving`, `wo-shape-parity`). agent1 applied the local `git mv` after a direct
message; the trunk is fixed by my merge; gemini's tree was notified.

## 3. The fix — coordinator scope (a)–(d), all four implemented

| # | ruling | implementation |
|---|---|---|
| (a) | parse failure = HARD CI error, never "CI: PASS" | `try/except` around `json.load` → `sys.exit(4)` with `REPORT BLOCK FAILED:` on stderr |
| (b) | explicit selection, not any digit-prefixed JSON | strict `_re.fullmatch(r'\d{8}_\d{6}\.json')` — the exact shape `run_verify_tests.sh:362` writes — then `max(..., key=os.path.getmtime)` |
| (c) | propagate the block's exit code into `RC` | `REPORT_EXIT=$?` immediately after `PYEOF`; new first branch in the verdict |
| (d) | assert the loaded file is a single JSON object | `isinstance(verify_data, dict)` → `sys.exit(4)`, so a future stray file fails **loudly at load**, not silently at report |

Plus, non-fatally: a staleness **warning** if the chosen verify file is >1 h older than the run's
`TS`. Deliberately a warning, not an error — `run_verify_tests.sh` stamps its own timestamp and
making that fatal risks failing legitimate runs.

The verdict line for a report-block failure is distinct and unambiguous:

```
CI: FAIL ✗ REPORT BLOCK FAILED (exit=4) — verify-derived gates were NOT evaluated; do NOT treat as a pass
```

with `RC=2` (the existing `BUILD_FAIL` code), so it can never be read as a pass by a human or a
caller keying on exit status.

## 4. Verification — `results/141_gate_verify/146_test_load.sh`

The battery extracts the load block **from the live `run_ci.sh`** by `awk` and executes it, so it
tests the shipped code rather than a copy. Six scenarios:

| # | scenario | expected | result |
|---|---|---|---|
| S1 | verify file + `_ci.json` present | loads the **verify** file | ✅ `20260903_124533.json`, exit 0 |
| S2 | stray `4b_*.json` JSONL present — **old code crashed here** | ignored by the selector | ✅ loads verify file, exit 0 |
| S3 | newer verify file exists | newest **by mtime** wins | ✅ `20260904_000100.json` |
| S4 | no strict-verify file at all | hard fail | ✅ exit 4, `REPORT BLOCK FAILED` |
| S5 | verify file is JSONL | hard fail | ✅ exit 4, names the parse error |
| S6 | verify file is a JSON **list** | hard fail | ✅ exit 4, "is a list, expected … object" |

Shell wiring tested separately in both directions: `sys.exit(4)` → `REPORT_EXIT=4` →
`CI: FAIL ✗ REPORT BLOCK FAILED`, `RC=2`; `sys.exit(0)` → `CI: PASS ✓`, `RC=0`. (Comment lines
between `PYEOF` and `REPORT_EXIT=$?` do not clobber `$?` — verified, since that is a real hazard in
this edit.)

`bash -n tools/ops/run_ci.sh` clean.

## 4b. Audit methodology error, recorded because it nearly produced a false alarm

The **first** version of `146_historical_load_audit.py` reported **27 "decision-affecting"
mismatches** and printed `VERDICT: ... BLOCKS the GO`. That was wrong, and the error was mine:

- I compared the sibling's `mtp_long_ok` against the `_ci.json` **top level**. But `mtp_long_ok` is
  not in `run_ci.sh`'s copy list, so it is legitimately absent there — "absent vs True" is an
  artifact of the comparison, not a mis-sourced load.
- The correct primary test is `ci["verify"]` (the nested block, which *is* exactly what the selector
  loaded) against the sibling, over all shared keys. On that test the same file matched: nested
  `mtp_long_ok=True`, sibling `mtp_long_ok=True`.

Rewritten with the primary/secondary split and a copy-list restricted to the keys `run_ci.sh`
actually copies. Result went 27 mismatches → **0**.

The lesson is the same shape as §6 and belongs with it: **an alarming result is not a finding until
the measurement method has been checked against a known-good case.** I had the disconfirming
evidence available in one command (`does the nested verify block match?`) and reached the verdict
before looking. The conservative wording in §2 was right, and it is now actually tested rather than
merely asserted.

## 5. What this does NOT do

- Does not touch any test, gate threshold, or numerics.
- Does not delete or move anyone else's result files.
- Does not fix the underlying habit that produced the bad file — `decode_guard.sh` writing
  `results/<TAG>_decode_*.json` into the **root** of a directory CI globs. That stays a hygiene
  item; the load site is now immune to it either way.
- Does not make `run_ci.sh` reject a *stale* verify file — warns only, per §3.

## 6. Lesson for the closeout doc

**"`results/` is inert data" is FALSE when CI globs it.** A file added under a results directory,
with no `src/` or `tools/` change, altered CI's behaviour. "Docs-and-results-only" is therefore not
synonymous with "CI-outcome-neutral" and should not be accepted as a safety argument without
checking what CI globs. This is the self-correction I raised on my own docs/141 merge claim
(coordinator accepted into the record, seq 12).

---

## Appendix — pp_tps rebaseline, 2026-09-04 (coordinator ruling seq 36)

> **Timestamps in this appendix: disk/log = America/New_York (EDT); intercom messages = UTC**
> (EDT = UTC−4). Recorded because a coordinator ruling read local-time disk mtimes as UTC,
> mis-dated a set of runs by four hours, and from that concluded a verification step had
> reported a false clean state. It had not. Lesson: convert clock spaces before attributing
> state to a time. See also `results/141_gate_verify/dg100k_RESULTS.md`.

**pp_tps rebaselined to 245.1** by controlled warm capture on the step-0-fixed tree.

### The regression is real, and the prior attribution was wrong

`/home/intel/verify_logs/20260830_075038_pp.log` records
`prefill: 225 tokens in 811.7 ms (277.2 t/s pp)` — a genuine Aug-30 measurement. Current
warm pp is ~245–247, so **−10.9% against a real baseline is a real regression.**

**Retracted:** the claim that "277.2 exceeds all recorded readings (max 269.6) and is
unreproducible". Both the original agent2 scan (1449 `results/*.json`) and the coordinator's
independent verification (543 readings) covered only **derived** JSON and missed the **primary**
`verify_logs/*_pp.log` files, 15 of which are on disk. Lesson recorded under attribution
discipline: **scan the primary source first.**

**`037a9c73` is excluded as the cause.** pp was already 246.8 t/s at `20260903_084325` on
commit `9e7cc5b7`, and `merge-base --is-ancestor 037a9c73 9e7cc5b7` is **false** — the blamed
commit landed 12:51, four hours later. It is also the only commit that ever touched
`gqa_attention_prefill_bf16.cuh`, so it cannot act through anything else. Post-landing readings
246.4 / 247.5 put its cost at **≈0.4 t/s**, not 30.

**Cause: unattributed.** The drift lies between Aug-30 07:50 (811.7 ms) and Sep-03 08:43
(246.8 t/s). Live **candidate, not proven**: the Sep-01/02 MultiBatch series (`0363c137`,
`ac6b3f61`, `326ec81c`, `4e5cd082`) refactoring shared launcher/cache code that single-sequence
prefill also traverses. The pp cell itself exercises **no** batching (`--mtp 0 --tokens 1`, and
`grep -E "max.concurrency|batch|lanes" tools/verify_battery.sh` returns nothing), so batching is
not in the measurement path even if MultiBatch refactoring is in the causal chain. Follow-up:
bisect `6857c7ea`→`4fd1001a` (~4 GPU runs, owner agent1, their lane).

### Cold/warm bimodality — a measurement hazard, not a regression

pp is bimodal by ~12% depending on GPU clock state:

| band | wall time | t/s |
|---|---|---|
| warm | 908–919 ms | 245–247 |
| cold | 1023–1095 ms | 205–220 |

**Any single-run pp measurement is unreliable without a discarded warm-up run.** This is what
produced the 219.8 and 205.6 values written into the baseline on Sep-3 23:05–23:09.

### The tool defect that made this confusing, and its fix

`verify_battery.sh` passed bash `"$UPDATE"` into Python argv as a **string**; `'0'` is truthy in
Python, so the write gate `if update and fails == 0 and warns == 0` degenerated to
`fails == 0 and warns == 0` — **every passing battery run rewrote the baseline, flag or not.**
Third instance of the session's stringly-typed-zero family (after `getenv("X")`-is-true-for-`X=0`).

Fixed in step 0 (`8f27e46f`) by restoring `update, k4 = int(update), int(k4)` verbatim from
`main`. Proven in practice, not just by reading: after the fix, the post-write verification run
(measured 246.0) left the baseline at 245.1, and a further plain run left the file's md5
unchanged (`26d5cc52…` before and after).

Why 277.2 survived at all: the gate's `fails==0` term was the only part working — pp ~247 vs
277.2 → FAIL → no write. **The baseline was frozen by the regression, not by the flag.** Once it
was lowered, runs passed and the bug began self-overwriting.

### Write timeline (Sep-3, local)

```
22:24–22:27  diagnostics, warm      247.1 / 247.7 / 247.3   (0.24% spread)
23:05:15     COLD capture           219.8 (1023.5 ms)  -> written; backup holds the original 277.2
23:06:05     verify, warm           245.3 (917.4 ms)   -> self-overwritten by the truthy bug
23:07:39     COLD capture           205.6 (1094.6 ms)  -> written
23:08:27     verify, warm           244.8 (919.1 ms)   -> self-overwritten
23:13:54     plain run, no flag     247.1              -> self-overwritten (the defect, live)
--- step 0 fix, step 1 restore of 277.2 from bak_20260903_230515 ---
23:38:48     warm-up, DISCARDED     245.4 (917.0 ms)
23:39:48     controlled capture     245.1  -> WRITTEN (backup: verify_baseline.json.bak_20260903_233948)
23:4x        verify run             246.0  -> baseline UNCHANGED at 245.1 (fix proven)
```

### Mechanism, on the record (condition 4)

The write used the tool's own full captured dict (`verify_battery.sh:215-216`) with the tool's own
`keep` filter expression (`:218`) applied verbatim. No value hand-chosen; no row selectively
edited. Capture runs FAIL against 277.2, so the tool's own gate blocks its write even with
`--update-baseline`; the explicit write+backup **is** the controlled path.

### Reproduce byte-for-byte (conditions 1–3)

```bash
cd /home/intel/ninfer/worktrees/wo-4a4d-gate-verify          # step-0-fixed tree
BUILDDIR=/home/intel/ninfer/worktrees/wo-2a-batched-serving/build REPO=$PWD \
  bash tools/verify_battery.sh --no-build                    # WARM-UP — DISCARD this one
TRUNK=$PWD ./results/141_gate_verify/rebaseline_pp.sh --run  # capture -> keep -> backup -> write -> verify
# report : /home/intel/verify_logs/20260903_233948_report.log
# backup : /home/intel/verify_baseline.json.bak_20260903_233948
# verify : diff of baseline vs tool-filtered report dict == EMPTY (asserted by the script)
```

`--run` also re-runs the battery afterwards; with step 0 in place that run **cannot** mutate the
baseline, which is what makes the post-write check meaningful. Before step 0 it was circular.

Battery bands unchanged (FAIL < 0.90×, WARN < 0.97×) — the rebaseline preserves future-regression
detection against current reality rather than an unreachable Aug-30 number.
