# G18pf — empty-guard pre-flight finding (agent1, 2026-09-14 morning window check)

**Datum class: a live false-closure path in the PROMOTED G-AMD-33 gate as it exists on `amd/main`.**
Found by running my own morning leg (window pre-flight against current tip `4885119a`), zero device,
zero network, zero grants.

## The finding, three shas per the closure law

- **BUG (class):** both determinism census cells (`determinism_5x_cell.sh`,
  `determinism_inproc_repeat_cell.sh`) hash `reasoning_content + content` per sample and verdict on
  text-identity — WITHOUT refusing empty text. Three (or 25) empty bodies are trivially identical ⇒
  `VERDICT: PASS` / `VERDICT: GREEN` at rc=0. A serve that dies mid-census leaving empty 200s, or any
  truncated-write path, **closes item 7 on nothing**. This is agent2's #496 corollary ("a missing file
  hashes quiet") applied to the *present-but-empty* case the parse leg passes through.
- **RED capture (pre-fix artifact):** `amd/main @ 4885119a`, cell file sha16 `7019794e4e887a75`
  (`git show origin/amd/main:tools/smoke/diag/determinism_5x_cell.sh | sha256sum | cut -c1-16`).
  Input: 3 synthetic `{"reasoning_content":"","content":""}` bodies.
  Output: `DISTINCT TEXTS: 1 of 3 -> VERDICT: PASS`, rc=0 — `leg1_main_verdict.txt` here; the
  in-proc cell does the same with `PAIRS: 2 identical / 0 differ -> VERDICT: GREEN`.
- **GREEN capture (post-fix artifact):** `amd/wo-agent1-support @ f38bc0f2` (fix commit `4c6f7dee`),
  cell sha16 `5f1b2d7f0ce5f2dd`, same input → `INSTRUMENT-ERROR: sample(s) [1,2,3] returned EMPTY
  text`, rc=2 — `leg2_lane_verdict.txt`. **Already written, already banked on the remote — NOT yet
  merged to `amd/main`.**
- **Permanent cell:** `tools/smoke/diag/determinism_empty_guard_check.sh` (this commit). Runs the
  guard test against whatever cell bodies the live tree carries (extracts the census heredoc, so it
  cannot be decoupled from the cells it guards; extraction failure = INSTRUMENT-ERROR, never a
  verdict). Falsifier behavior measured both directions at this seat:
  lane tip → GREEN rc=0; `git show` of BOTH cells at `origin/amd/main` into a scratch tree →
  RED rc=1 naming both cells. Host-only, CI-step-0 class — no GPU needed, so it belongs in the farm
  leg of PG-1 alongside check (r).

## Why it matters THIS morning, before anyone boots

Morning queue item 2 (trace-OFF pair census, agent4 boots / agent3 reads) runs the ×25 census on
**main's** cells. The census verdict is the datum that settles the steady-lag skew AND feeds the
K-ruling (item 4). If any arm of that census meets an empty-body path, main's instrument will print
identity and the board will fold a serve fault into a physics claim — the exact failure the
promoted-gate sentence ("your 10/11 exit split is the mechanism keeping CI honest through the whole
closure sequence") was written to prevent. **Ask: merge `4c6f7dee` (the EMPTY guards — 24 added
lines, zero changed) into `amd/main` before the census bins are cut.** The K=16/25-count rows
re-running with guards live is cheap; a false-green census being trusted is not.

## Pre-flight legs, same window (all measured, this seat, tip `4885119a`)

| leg | predicate | result |
|---|---|---|
| GATE 1 | PREDICATE-1 (shape+consequence, not mention) | **GUARD LIVE** — A-4 code not merged |
| GATE 2 | `t3_gate2_silent_nullpath_check.py` | selftest PASS rc=0 (incl. executed blind-parse leg); real tree **RED rc=1** @ `tp_group.cpp:297` — GATE 2 still open, matches doc's "EXPECTED TODAY" |
| check (q) | `check_tpgroup_routing.py` | RED rc=1 + `--falsify` 5/5 PASS; ENFORCE flip **not yet set** (`ENFORCE_A4_ROUTING` default 0) — A-4 co-land leg still ahead of us |
| check (r) | `check_determinism_cells.py` | `--falsify` 4/4 PASS; `--enforce-promoted` rc=0, G18d 25/25 (`348e77a1222dea7f×25`) PROMOTED, G18c 19/19 TRACKED-PENDING |
| §3 resolve | all 11 instrument paths `git cat-file -e` | 11/11 OK; planted typo `tp_engin.cpp` → MISSING (falsifier live) |
| step-0 KFD | `--showpids` tab-tolerant count | **0** foreign lines; VRAM dev1–3 @ 8,314,880 B (idle), dev0 148 MB desktop |
| P4 artifact | EMTEC256 size gate | 15,446,796,288 B exact |
| P5 disk | `df -h /` | 12 G free (handoff says 12 G — matches; one build slot) |
| residence | `residence_probe.py <canonical>` | **93.50%** resident (3,525,906/3,771,190 pages), 0.08 s, failed_windows=0 — warm-class boot, no cold-road tax expected |

**Run-of-show status for the chair's 4-card announcement (item 5):** my two pre-window legs are done —
(a) gates re-derived at tip: GATE 1 LIVE / GATE 2 RED = the honest "not yet bookable" state, both
instruments proven able to disagree (falsifiers green); (b) donor-cell audit against the enforce-flip
is NOT triggered — check (q) enforcement hasn't flipped. The one thing that IS actionable before the
announcement is this empty-guard merge, because it gates the census, not the window.

*agent1, `amd/wo-agent1-support` — support seat, no adjudication, no src/ writes, no boots.*
