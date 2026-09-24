# 26 — Session debrief at relaunch (successor entry point, supersedes v340l/25 for STATE; 25 stays as its detail annex)

**Author:** agent2 (pi 01a09af0), 2026-09-13T18:2xZ. User is recycling this session. **Read in order:** this file → `25_lane_state_note.md` (same branch, queue-(3) geometry + verified loader facts) → `23_lane_handoff_at_session_end.md` (@ `ea3e1e7b`, still the doctrine doc). Branch: `amd/wo-shim-funcattr` @ `eebb3233`, merged to `origin/amd/main = 785bca6f` at close, ANTI-RES diff 0 lines, surface = docs/results/tools + the one registered-exception README fix (`src/common/hip_shim/README.md`, verifier rc=0). Rev-list at close: 33 ahead (behind 0 — main fully merged) (re-derive with `git rev-list --count origin/amd/main..HEAD`; moves with the wave).

## What this session did (one line each, receipts in parentheses)

1. **Host suite queue-(1): 9/0/0/0 three times** at 88af9c78/9cbcbadf/83fba46e — shas + four-counts sent to chair; receipt `results/amd/host_suite/2026-09-13_merged_tip_88af9c78.txt` (append-only file — add to it, don't fork it).
2. **G-AMD-28 tp1-control: COMPLETE as a death row.** Ref artifact is **binder-dead on this tree** — 66 unconsumable `dflash2/*` objects, zero dflash2 bind sites in either 27B loader route (`grep -c` = 0 on bindings.cpp AND tp_load.cpp; both routes share `bind_artifact`→`Binder::finish`), q3 carries 0 dflash2 = why every q3 boot passed. **DO-NOT-REFIRE** — deterministic 0.2 s repro. Row: `results/amd/p3/G28_death_row.md`; manifest was pre-spawn (`686e799a`); bin banked `/home/chris/artifacts_bin/tp1control_a28_9dc56c390815bda0.bin`. The cell re-opens only at a chair product ruling (strip-export / sidecar-shape / fixture-class); if it does, the closure row cites the death row + re-runs the binder repro to flip (RED→GREEN closure law triple: `53c09d17`).
3. **Agent3's #784 item-7 review** (`docs/amd/v340l/24`): corpus triangle re-derived at my bytes; A1TRACE-K sees flags-not-payload defect filed (later vindicated by agent5's "K-blind publish-tear" and the chair's pad-tag instrument design `69d9b3c8` — my §4 proposal and their NINFER_AR_PARITY converge; instrument law stands, hunt moved to carry-in/`full_reset` under agent4).
4. **Instrument work on the lane:** tps_probe per-token ARRIVAL CURVE + FLAT/CURVED (max/MEAN rule — median self-masked its own planted spike before the fix; evidence in-file) at `3e818d23`; README symbol-grep fix `c2e6106b`; suite script `71ea8590`; INDEX rows 22/24/25 + v340l/21 row flipped OPEN→CLOSED (main's verifier now self-plants poison fixtures, `:592/:621` — do not re-file the gap).
5. **Backlog-relay drain (#691–#768):** ~25 stale relays triaged against current bytes, none allowed to re-enter stale state; live residue routed: agent4 `wave_retirement_plan.md:132` "15 vs 0" stamp ask (by hub id `pi-…-1607751` — NAME ROUTING FAILS, use ids), chair RESTART-kit stale ticket line, phantom 04:00→08:00 grant closed by three-seat locus exhaustion.

## The ONE live obligation: queue-(3), USER-TASK real-context TPS cell

**Status: written-requested, NOT fired, grant basis = user directive + chair 13:31Z/13:45Z dispatches (NOT any 04:00→08:00 grant — struck, see v340l/25 + this session's locus search).** Successor: re-read the chair's CURRENT STATE first — dev2,3 may since have been consumed by agent4's G-AMD-26/parity probe/carry-in hunt or agent1's G-AMD-29, and the hunt may have ended. Then:

- **Request in writing to the chair** naming host:port + window (my draft: dev2,3, 35 min, kit-v6 launch shape + `--max-context 2048` + `--kv-dtype kvarn_k5v4`, artifact `/media/chris/EMTEC256/qwen3_8_27b_q3.ninfer` 15,446,796,288 B — **never the Desktop copy (truncated, 6.61 GB)**; desktop_f = same bytes over CIFS, don't pull 15 GB across SMB). Flags validated against the banked bin's `--help` (`--max-seq-len` does NOT exist).
- **Manifest row BEFORE spawn** (template = `results/amd/p3/G28_tp1_manifest.md`), release row after (deaths included), kill recorded pgid only — **contractual clause on my next grant** (LAW 20 lineage). `HIP_VISIBLE_DEVICES=2,3` + `--devices 0,1` (mask/ordinal law). KFD tab-tolerant raw dump at spawn minute; never carry a "pair is free" claim.
- **Attach the reader cells:** agent5's APPROVED gen-64 (Arms A 54/{4,32,64,128}×2, B 1800/{32,128}×2, ±15% flatness rule — exact text in v340l/25, fires only after MY manifest row posts), second-request determinism reader (same prompt ×2-3 identical length, every generation incl warmups + `--request-log-jsonl` banked per-LABEL) — data source for agent4's carry-in hunt; and report the ARRIVAL CURVE, not the average, on every row (the phantom-cliff error class).
- **Ceiling discipline:** predecessor measured 260–512 boot-AND-serve ceiling at 256-class, 4096 dies at first REQUEST (not phantom), 2048+k5v4 held >230 s idle-stable. If the band dies under service, THAT is the datum — no mid-stamp re-tune, new stamp for a new truth (gate-pass-but-same-site-death law, runbook).

## Facts a successor must not re-derive at card cost (all at my bytes this session)

- ref `artifacts/qwen3_8_27b.ninfer`: 20,437,336,576 B, sha `0634abb0…467eec3` (salvage receipt), 1190 objs {text:773, vision:333, **dflash2:66**, mtp:12, frontend:6}, Q3G64=0. Binder-dead, above.
- tp1 route = plain `Engine` (`engine.cpp:308-313`, devices.size()==2 → TPEngine else single); single-Engine never reaches tp2 code.
- GATE-2 real: `tp_group.cpp:297-302` null-guard no-else (chair FYI, verified; A-4's scope, not mine).
- Item 7 chain (for reading context, not action): 17f==g5 content sha `348e77a1222d`/162ch, g3@107, g4@15; per-request nondeterminism LIVE 5/5 (chair STATE 13:5xZ), sampler exonerated 5/5, suspects = carry-in state (`full_reset` park/restore, GDN slot recycling), agent4 owns the hunt, pad-tag instrument named-before-patch.
- WO-VRAM-1 unit fully on main: decision path measured-only (headroom deleted `tp2_budget.h:57-60`, INSTRUMENT-ERROR classes at `tp_engine.cpp:875/:891/:931`, near-cap LAUNCH cell `tools/ops/wovram1_nearcap_auto_launch_cell.sh`).

## Hygiene / boundaries the successor inherits

- **COMM LAW absolute: no channel posts, hub-enforced; directs by name-or-id (agent4 = hub-id-only); receipts are commit shas.** ~25 stale relays will keep arriving (09:0x–12:0xZ band replaying); treat every one as unverified history, byte-check before any action — several re-smuggle struck claims (phantom grant, dev0/1 pair, same-token bar, "15 violations", W1/W2-open).
- Watcher daemon (pid 2054034, predecessor-era) still alive, writing `results/amd/watch_events/*.tsv` every 30 min — bank-commit them (last banked 17:55Z; untracked rows may exist at your start). Do not kill it without checking ownership; it is lane furniture.
- Shared-scratch state I left: `/tmp/ht` detached at 83fba46e (suite TREE design; harmless, re-pointable); `/tmp/host_suite_9_*` + `/tmp/sv_G17*.log` transient; `/tmp/fake_sse*.py` probe test servers (dead).
- **DISK: 14 G free at close (was 17 G mid-session, 19 G at their cold start — it is tightening again).** Re-verify `df -h /` before ANY build; my tp1 build cost ~500 MB; the >5 G ask-first rule is live.
- Pi-id in every commit message (LAW 19): mine `01a09af0-4bd8-7686-b22e-7face3f2c931`.
- My lane files nobody else should touch: `src/common/hip_shim/*` (README row is registered; verifier-passing edits OK), `tools/v340l/*`, `docs/amd/v340l/*`. `tools/ops/`+tests = gemini's; tp2 canonical = main-wins-by-law. Report, don't patch, outside.

## Open questions the session leaves (with owners, not mine)

1. Chair: merge the lane? (three suite greens + death row + instrument upgrades + README fix queued behind their decision; my surface claim is re-derivable with the two diff commands in v340l/25.)
2. Chair/agent4: G-AMD-28 closure awaits the product decision on the control-table cell (fixture-class likely the honest disposal on this box).
3. Agent4: the `:132` sha-stamp — my direct may have died with routing; re-ping if silent at their next row.
