# W35 Harness Fix - guard battery SKIP rows + GUARD-CAPS (E-145/E-147)

Date: 2026-09-24. Desk: HARNESS-FIX. Branch: amd/v340-port-v2.
Commit: a795c09e4 (amd-port tests: guard battery emits SKIP rows + GUARD-CAPS under --decode-only).
Surface: docs/amd-port/tests/guard_battery.py + new replicating host test. Zero GPU, zero kernel
touches, build-hip untouched (clock-floor window live 18:00-19:15 CDT), /home/chris runner scripts
read-only.

## 1. Defect chain (E-145, twice-observed; E-147 same class)

run_post_u1_battery.sh arms windows with `--decode-only` (w23/w28/w19/ub cells, runner lines
442-479). guard_battery.py flag-gates its cells with bare `if not (...)` conditions and, in the
skip case, simply never appends the row to guard_results:

- prefill_guard: skipped by any of --decode-only/--canary-only/--determinism-only/--needle-only
- mtp_canary row: withheld under --decode-only (metrics computed, then dropped, old line 858)
- decode_guard row: withheld under --canary-only
- whole decode branch (both rows): skipped under --prefill-only/--determinism-only/--needle-only
- determinism_guard / needle_recall_guard: skipped by any competing mode flag

Row-absent is ambiguous downstream. Adjudicators that demand the rows mis-read absence:

- w23env window VOIDED as "text_sha256 diverged" - the sha was never measured (E-145). The old
  summary table derived `m_val = "diverged"` from `r.get("identical")` on a row that could not
  exist, and analyst-side parsers read the missing determinism row as a failed/voided guard.
- w28nmax printed ACCEPT-MISSING the same way (E-147 false-void class): no canary row in the
  receipt, adjudicator filled the gap with a miss verdict.

W32 battery: 3 of 5 window laws were structurally voidable by this one harness gap.

## 2. The fix (docs/amd-port/tests/guard_battery.py)

1. SKIP rows. Every flag-skipped cell now appends
   `{guard, cell, status: "SKIP", reason: "<flag>", notes: "<guard> SKIP (<flag>)"}`
   in its canonical slot, so row order stays prefill, decode, canary, determinism, needle in
   every mode. Reason is the responsible exclusive flag ("decode-only", "determinism-only", ...).
   The verdict loop treats SKIP as neither pass nor fail: a real FAIL still yields
   OVERALL VERDICT: FAIL / exit 1 (covered by the test), PASS runs stay PASS.
2. Summary table. SKIP rows print `SKIP <guard> <metric> - - <notes>`: MEASURED/BASELINE are "-"
   (never measured - no fake numbers, no "diverged"), but guard and metric names are kept
   (SKIP_METRICS map: prompt_tps / decode_tps / draft_accept / text_sha256 / exact_recall) so the
   runner's existing grep
   (`grep -E "PROVENANCE|prompt_tps|decode_guard|draft_accept|text_sha256|exact_recall|VERDICT"`)
   still captures SKIP rows into the cell logs adjudicators read. No runner change needed.
3. GUARD-CAPS line at battery start, exact format:
   `GUARD-CAPS: decode=1 prefill=0 determinism=0 canary=0 needle=0`
   (1 = this row will carry a real measurement under the armed flags; printed order
   decode prefill determinism canary needle).
4. Receipt JSONL carries the same dict as top-level `guard_caps` - the machine channel that
   survives even where console lines are grepped away.
5. Full-battery behavior: with no mode flag, no skip branch fires - guard rows, table, verdict,
   PROVENANCE and thermal lines are byte-identical to the old code (proven below, section 3).
   The battery fingerprint changes because BATTERY_VERSION hashes guard_battery.py itself - that
   is the version identity doing its job, not an output regression.

## 3. Proof

- Replicating host test (new): docs/amd-port/tests/test_guard_battery_skiprows.py - 14 checks,
  SKIPROWS-VERDICT: PASS, exit 0. Drives the real main() with network, session lock, provenance
  gate and thermal sampler stubbed (zero GPU, no lock contact, no sockets). Covers: decode-only
  caps line exact, 5-row emission, reasons, row order, no "diverged" text, verdict/rc semantics
  under decode FAIL, determinism-only mode (this case caught a second hole: the outer decode-branch
  skip still dropped both rows - fixed), full-battery no-SKIP identity, receipt guard_caps.
- Before/after harness (/tmp/w35_harness): HEAD copy of guard_battery.py vs fixed, same stubs,
  full battery + --decode-only.
  - Full battery diff after masking fingerprints and mkstemp names: exactly one added line,
    `GUARD-CAPS: decode=1 prefill=1 determinism=1 canary=1 needle=1`. All else byte-identical.
  - --decode-only before: single decode_guard row, three guards invisible (the E-145 hole).
    After: GUARD-CAPS line + 5 rows (4 SKIP + real decode row), verdict unchanged.
- CI: /home/chris/run_premerge_ci.sh with CI_SKIP_GPU=1 (E-121b zero-GPU desk law; the merge gate
  re-runs section 3) and CI_OUT to /tmp: tree clean, 7 host suites PASS, engagement routing PASS,
  meta re-entry PASS, gate-wiring SKIP (GPU) -> CI-VERDICT: PASS.
- Safe-invocation note: a live served battery could not be used for the before/after demo
  (lock law + clock-floor window); the stubbed-main harness executes the real code path
  (provenance gate print through receipt write) instead, so the outputs above are the true
  console formats.

## 4. Deferred (documented, intentionally not done this session)

1. Kernel gate-log severity (rebuild-window item): the mmvq share/s2r gate lines in
   ggml/src/ggml-cuda/fattn/mmvq still log INFO, so the served "engagement" evidence is dropped
   from filtered server logs. Needs INFO->WARN in kernel code + a build-hip rebuild window -
   both forbidden this session (clock-floor window boots servers from build-hip).
2. Wire test_guard_battery_skiprows.py into /home/chris/run_premerge_ci.sh: one entry in the
   host-suite pattern (python3, zero-GPU). /home/chris scripts are untouchable by this desk.
3. run_post_u1_battery.sh grep does not pass the GUARD-CAPS console line into cell logs (row
   names pass; the caps line itself does not match the alternation). Add GUARD-CAPS to the
   alternation if console-level caps are wanted; until then adjudicators read guard_caps from
   the receipt JSONL, which is complete.
4. Pre-existing non-ASCII degree sign on the THERMAL DRIFT console line left as-is: changing it
   would break full-battery byte-identity for existing parsers (no new non-ASCII introduced).

## 5. Adjudicator guidance (E-145 close-out)

- A guard row with status SKIP was never measured: it can neither void nor promote a window.
  w23env-style "diverged" verdicts from an absent/sha-less row are harness artifacts, not data.
- Configure expected row sets from GUARD-CAPS / receipt guard_caps instead of assuming a full
  battery: decode-only windows have prefill=determinism=canary=needle=0 by design; prefill
  co-metrics for those windows come from server-log print_timing extraction (W32 protocol).
- ratchet law untouched: --ratchet still keys on PASS rows only; SKIP rows cannot ratchet.
