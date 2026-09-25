# W21 - PERSISTENT-SERVER BATTERY PROTOCOL: DESIGN + OWNER DECISION BRIEF

Date: 2026-09-24. Desk: persistent-server protocol (zero-GPU; no server was
launched, no GPU touched, no lock created). Branch: amd/v340-port-v2.
Prototype runner: docs/amd-port/scripts/run_persistent_battery.sh (--dry-run
verified; gates proven, including the stale-binary refusal).

## 1. PROBLEM (ledger E-134)

Five hard machine deaths; forensics pin KFD svm_range_deferred_list_work
instability under ALLOCATION CHURN, with every death clustered at
llama-server boot/teardown cycles. Every battery cell today = one full boot
(27B q4_0 weights to 4 dies + 200k-context KV pool), ~5 min run, teardown.
noretry=1 did not prevent deaths; the svm-watchdog (hog >= 3 claims
/tmp/campaign_gpu_boot.lock) converts deaths into pauses but does not remove
the churn. The named REAL fix is the persistent-server protocol: boot ONCE,
run N batteries, tear down ONCE. It changes the measurement protocol, so it
needs an owner decision before it becomes the default.

## 2. SERVER RESET SURFACE TODAY (verified, tools/server)

- GET  /slots            tools/server/server.cpp:244 -> server-context.cpp:4645
  (slot metrics; needs --slots, default-on).
- POST /slots/:id_slot   tools/server/server.cpp:245 -> server-context.cpp:4688
  (post_slots). Dispatches action=save/restore/erase at
  server-context.cpp:4707-4714. Erase handler handle_slots_erase at
  server-context.cpp:5368; SLOT_ERASE task at server-context.cpp:2765-2788
  calls slot->prompt_clear(false) (server-context.cpp:255-267): seq_rm of the
  WHOLE sequence on the target ctx AND the draft ctx (ctx_dft), plus
  prompt.tokens.clear(). Returns n_erased witness. Defers while the slot is
  busy (server-context.cpp:2776) - safe, but caller must retry.
- GATE: post_slots refuses ALL slot actions unless --slot-save-path is set
  (server-context.cpp:4688-4692, "Start it with --slot-save-path"). The flag
  requires an existing directory (common/arg.cpp:3234-3243).
- PER-REQUEST STATE: guard_battery.py sends cache_prompt=false on every
  request (guard_battery.py:390,447,541,601). With cache_prompt=false the
  server computes n_past=0 (server-context.cpp:3307-3383), empties the token
  list (keep_first, server-context.cpp:3507) and seq_rm's everything past
  pos 0 on target AND draft caches (server-context.cpp:3531-3538). Every
  request already rewrites the full KV from position 0: numerically
  stateless per request.

MINIMAL SERVER ADDITION NEEDED: NONE. Adding --slot-save-path
<existing-empty-dir> to the launch line unlocks /slots/0?action=erase with
zero code change (the flag also enables save/restore; harmless). Optional
smallest-diff alternative if the owner rejects the flag: relax the
slot_save_path gate for action=="erase" only at server-context.cpp:4688-4692
(~4 lines; erase never touches the filesystem). NOT required for the
prototype runner.

## 3. STAGE-BY-STAGE STATE ANALYSIS (a)

Per cell, guard_battery.py runs in strict order; every request is
cache_prompt=false (full KV rewrite, see above). So "state" risk is not
prompt-cache reuse; it is PROCESS warmth and thermal regime.

| stage (order kept)        | warm-server safe? | notes |
|---------------------------|-------------------|-------|
| prefill_guard (2k, COLD-STAMPED) | numerics yes; regime no | deliberately cold-stamped after --idle-wait; under a persistent server, cell 1 pays one-time costs (ggml workspace pool growth, ROCm allocator, draft shape cache warmup). MITIGATION: cell 1 = SCRUB cell (run, stamped VOID, never banked); banked cells 2..N share one warm regime. |
| decode_guard (8k-10k -> 128 tok) | yes, with soak law | soak-sensitive (E-133: decode -26% under far-die throttle, ~13 min recovery), cache-insensitive. The runner keeps the same gates: settle between cells, die-hot/cooldown policy, VOID rule via thermal sideband. Warm draft shape cache can nudge acceptance up a few 0.001s - visible in paired deltas on BOTH arms equally. |
| mtp_canary (>= 0.63 gate) | yes | absolute gate, not a delta; warm regime raises acceptance if anything (safe direction). |
| determinism double-run | yes, MORE honest | compares two consecutive greedy requests within the same server process - exactly the property claimed; cache_prompt=false on both runs (guard_battery.py:531-553). |
| needle recall (25/50/75% @ 8k) | yes (functional) | exact-code retrieval gate; stateless per request. E-133 note: it stays LAST in the cell (soak source). |

CELL-BOUNDARY RESET: POST /slots/0?action=erase between cells (target KV +
draft KV + token list cleared, n_erased logged as a witness). Belt-and-braces
on top of cache_prompt=false; also prevents the slot-similarity/catchup
machinery from ever seeing stale slot tokens.

SERVER RESTARTS: between ARMS only (arms differ by env/cmdline that is read
once at process init); NEVER between cells.

## 4. PAIRED ALTERNATION UNDER ONE SERVER (c)

- Runtime-switchable arm (env per request): impossible - the arm envs
  (GGML_CUDA_MMVQ_*_SHARE, LLAMA_DRAFT_*, ALLREDUCE) and BS/UB args are read
  once in ggml/backend/server init. Making them HTTP-switchable would be an
  invasive per-arm subsystem; rejected.
- Two CONCURRENT persistent servers (arm A :8081, arm B :8082): rejected -
  double weights + double 200k q4_0 KV against 4 dies x 16 GiB does not fit
  at the 200k decision context, and a simultaneous double boot is exactly the
  churn we are eliminating.
- CHOSEN: SEQUENTIAL PER-ARM PERSISTENT EPOCHS. Boot arm A once -> N cells
  (cell 1 scrub) -> teardown -> settle -> boot arm B once -> N cells ->
  teardown. Churn per paired decision: 2 cycles, independent of N.
  COST (stated honestly): interleaved A/B alternation within a window is
  lost; arm comparison becomes between two thermal eras. Mitigations:
  (1) position-matched cell deltas (both arms run the identical scrub + N-1
  schedule with identical gates), (2) a mid-arm anchor cell (re-run the
  baseline cell at the midpoint; drift > noise band voids the window),
  (3) TIERING: persistent windows are the SCREENING tier only. Any positive
  delta must replicate in ONE classic interleaved window (E-119 law) before
  it banks a decision. The classic protocol is not retired; it is reserved
  for arbitration.

## 5. PROVENANCE + ARM IDENTITY (d)

- Nothing changes in the receipts: guard_battery.py's provenance gate runs
  PER CELL invocation (commit, clean tracked tree, binary sha256, cmake cache
  sha256, launch-config verbatim, battery fingerprint) and stamps every cell
  row (guard_battery.py:139-193, 697-713, 894). The runner passes the SAME
  --launch-config to every cell of an arm, so all banked rows of a window
  carry identical binary identity by construction.
- The runner inherits the E-133 arm-identity freshness gate verbatim
  (non-empty git stamp with safe.directory, binary mtime >= HEAD mtime,
  missing binary/model -> exit 2 before any boot). PROVEN IN DRY-RUN: it
  currently refuses the serving binary as older than the E-134 commit until
  a rebuild lands.
- NEW stamps per cell (log head + cell block): persistent epoch id, arm,
  cell_pos, mark (SCRUB-VOID/BANKED), server_uptime_s, KFD hog count before
  the cell, erase witness (n_erased). Verdict lines (prompt_tps /
  decode_guard / draft_accept / text_sha256 / exact_recall / VERDICT) are the
  unmodified guard_battery output, grepped exactly as the classic windows do
  - coordinator tooling reads them unchanged. The runner also emits a final
  "PERSISTENT-VERDICT: ... worst=..." line.

## 6. CHURN MATH (e)

Server boot/teardown cycles per decision window (each cycle = the death
site; E-134 deaths at per-boot hog 4-5):

| window                                   | today (1 cycle/cell) | persistent |
|------------------------------------------|----------------------|------------|
| combined window, 1 arm (200k + 10k cell) | 2                    | 1          |
| hardened paired prefill window (p0a/p1a/p0b/p1b) | 4            | 2 (1/arm)  |
| N-cell soak study per arm (N=4)          | 4                    | 1          |
| screening sweep, 3 levers x 2 cells      | 6                    | 3          |

Churn per DECISION at the screening tier drops 2-4x; hog exposure becomes
one boot-worth per arm instead of one per cell. Watchdog semantics are kept:
the runner waits on /tmp/campaign_gpu_boot.lock before boot, re-checks it at
every cell boundary (watchdog claims pause the run instead of churning into
it), and releases the lock only once, after the single teardown, followed by
the standard settle.

## 7. VALIDITY RISKS + CROSS-CALIBRATION (A4)

R1 WARM-CACHE DECODE/PREFILL INFLATION: banked persistent cells are not
fresh-boot cells. The banked regime starts AFTER the scrub cell, so banked
vs banked is consistent; banked vs classic-fresh-boot is NOT directly
comparable. Classic anchors show the scale: E-133 hardened window decode
23.2-23.6 (fresh) vs E-134(e) soaked-boot 16.7-17.7; warm-process effects are
expected far smaller than soak but must be measured, not assumed.

R2 KV CARRYOVER BETWEEN CELLS: refuted by code for numerics - every request
rewrites full KV from pos 0 (cache_prompt=false; server-context.cpp:3307,
3531-3538) and the runner additionally erases the slot at each boundary.
Residual risk is none at the value level; the witness line (n_erased) plus
the determinism guard detect any surprise.

R3 DETERMINISM UNDER REUSE: the guard itself (two greedy runs, byte-compare,
guard_battery.py:531-577) remains valid against one warm process; if
in-process state ever leaked into sampling, this guard fails loudly.

R4 THERMAL REGIME / SOAK: unchanged laws (E-133): decision cells at
position-1-of-regime or after >= 5-10 min idle; VOID if die-3 sclk
act-mean < ~1150 from the existing sideband; needle stays last. The
persistent schedule makes these gates MORE important (more cells per era),
and the mid-arm anchor cell catches slow drift the per-cell gates miss.

R5 FIRST-CALL JITTER ASYMMETRY: absorbed by the scrub cell; both arms run
the identical scrub+banked schedule so the asymmetry cancels in
position-matched deltas.

ROLLOUT (recommended):
- Phase 1 (shadow, no protocol change): after the next rebuild, run ONE
  persistent epoch of arm=regress (anchor env, 4 cells) between classic
  windows on the same binary. Cross-calibrate: banked persistent cells 2..4
  vs the classic fresh-boot anchors on the same commit. PASS GATE: decode
  and prefill deltas within the classic cell-to-cell noise band (use the
  E-133 spread, ~1-2%, as the first cut; tighten with the shadow data).
- Phase 2 (screening tier): persistent epochs for lever screening and soak
  studies; classic interleaved windows stay mandatory for any promotion
  decision (one replicate per E-119).
- Phase 3 (optional, only if owner accepts): paired persistent windows for
  lever pairs that need tight deltas - accepting the two-era cost with
  position-matched cells + mid-arm anchor.
- Unchanged throughout: svm-watchdog, cooldown/VOID gates, provenance gate,
  guard_battery.py (untouched), /home/chris runners (untouched).

## 8. DECISION REQUESTED (AWAIT-OWNER)

RECOMMEND: ADOPT the persistent-server protocol as the SCREENING tier with
the Phase 1 shadow cross-calibration against one classic window, and the
--slot-save-path launch flag (zero server code change) for the per-cell
erase reset. Keep the classic interleaved window as the mandatory
arbitration tier for decisions.

AWAIT-OWNER on exactly one decision:
  approve persistent-screening + classic-arbitration tiering
  (A) YES - adopt as recommended; shadow epoch first
  (B) YES, but with the 4-line server diff (ungated erase) instead of
      --slot-save-path
  (C) NO - keep 1-cycle-per-cell; rely on watchdog + 180s spacing only

The runner ships ready: docs/amd-port/scripts/run_persistent_battery.sh
(dry-run verified; refuses stale binary, missing binary/model/baseline;
holds the campaign lock convention; never pushes, never touches GPUs).
