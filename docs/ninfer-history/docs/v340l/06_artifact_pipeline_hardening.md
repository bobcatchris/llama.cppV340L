# v340l/06 — artifact pipeline: measured failure modes and fetcher hardening

**Owner:** Agent-B (artifact pipeline lane, C441 17:42Z assignment).
**Date:** 2026-09-12 ~17:5xZ. **Zero GPU, zero device touch** — agent3 holds G-AMD-14.
**Scope:** the 19 GiB reference pull in flight plus the Q3 v1a goal file (CROSSLANE ask-4).
Everything below is measured on this box, not inferred from the script's comments — where a
comment and a measurement disagree, the measurement is reported and the comment is called out.

## 0. What is actually running (baseline at 17:5xZ)

| item | value |
|---|---|
| fetcher | `artifacts/fetch_ref.sh`, pid **1315090** (coordinator's — I never signal it) |
| target | `artifacts/qwen3_8_27b.ninfer`, expected **20,437,336,576 B** (19.03 GiB) |
| registry sha256 | `0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3` (`docs/amd/README.md:28-30`) |
| progress | 6.22 GB ≈ 30% at 17:5xZ → 7.34 GB / 35.9% at 18:12Z; **measured rate 1.45 MB/s** (45 s two-point sample: 65,122,304 B) |
| ETA | **~2.9 h from the sample**, i.e. ~20:45Z, at that rate |
| `artifacts/fetch.status` | **absent** (by design — see §1) |
| git exposure | `artifacts/` was UNTRACKED, not ignored, and `*.ninfer` was not in `.gitignore`: `git add --dry-run -A` in the main checkout stages the in-flight partial (6.2 GiB then, 19 GiB at completion). Guard added in 72f00dd0 after confirming `git ls-files \| grep -c '\.ninfer$'` = 0, so nothing tracked is affected |
| `artifacts/fetch.err` | 0 bytes |
| goal file (Q3 v1a) | not on disk; `/home/chris/models/` does not exist; `q3_ci_amd.sh:36` still defaults ART to that missing path |

Rate/ETA note, stated because a schedule was built on it: the 17:2xZ report of "~1.7 GiB/min,
ETA ~2.5 h" and the measured 1.45 MB/s (= 0.083 GiB/min) differ by ~20×. The 2.5 h ETA and the
"4 h+ of wall time" framing are consistent with each other and with my sample; the 1.7 GiB/min
figure is not, and looks like GB-vs-GiB-per-minute slip. Use the byte-rate for scheduling.

## 1. `fetch.status` cannot be the liveness signal (it is terminal-only)

`fetch_ref.sh` writes `fetch.status` at exactly two points — the size-match branch and the
SIZE-MISMATCH branch, both after the loop exits. While a pull is in flight the file **does not
exist**, so "watch `fetch.status`" silently watches nothing for ~3 hours: absence is
indistinguishable from "still downloading" *and* from "the process died an hour ago".

Correct liveness = three signals together, which is what `tools/v340l/artifact_watch.sh`
implements: pid in `/proc` **and** bytes advancing between passes **and** `fetch.err` content.

## 2. The script's header overstates its own check — "sha256 checked against registry" is not wired

`fetch_ref.sh:3` says *"sha256 checked against registry"*. Line 17 is:

```bash
sha256sum "$OUT" > $D/fetch.sha256  # registry: 0634abb0...467eec3
```

That **records** the digest; nothing compares it to the registry value, and no branch can
produce a hash-failure status. So `COMPLETE` in that status file means **"size matched"**, not
**"artifact verified"**. The registry digest is a comment.

Consequence with teeth: a 19.03 GiB file whose bytes are wrong but whose length is right —
a duplicated block, a mid-flight truncate-then-refill, a CDN serving a different object of the
same size — yields `COMPLETE`, an empty `fetch.err`, and a green-looking pipeline. The
size-only definition of "done" is precisely the vacuous-green pattern this sprint has now
caught several times in other lanes; the artifact path has the same shape.

**Fix (H1 below): the compare belongs in the fetcher, and `COMPLETE` must be reserved for
size+hash agreement.** Until then, nobody should forward that word as a pass. My watcher does
the comparison itself and is the thing I will sign off on, not the status file.

## 3. THE HEADLINE FAILURE MODE: a 200-with-body error poisons the partial **silently**, then the loop spins at 176 attempts/sec

Measured against a local stand-in returning an HF-style HTTP **200** with an HTML
"our servers are busy" body — the shape error pages take when a CDN degrades, which is the
common case rather than a exotic one, because rate-limits and maintenance interstitials
frequently come back 200, not 4xx.

Reproducing `fetch_ref.sh`'s exact curl line (`-sL -C - --retry 5 --retry-delay 10
--speed-time 120 --speed-limit 1000 -o`, stderr appended to `fetch.err`), three iterations:

| iteration | curl rc | file size | `fetch.err` bytes |
|---|---|---|---|
| 1 (fresh start, no file) | **0** | **64** ← the HTML body, written as artifact content | 0 |
| 2 (resume at 64) | 0 | 64 | 0 |
| 3 | 0 | 64 | 0 |

And the loop rate, measured over a 5 s window: **880 curl invocations = 176/sec**, `fetch.err`
still **0 bytes**. So every observable channel says success: exit status 0, no stderr, no
status file, no error. The only signal is 64 bytes where 19 GiB should be.

Why it is unrecoverable in place: the 64-byte HTML prefix becomes part of the artifact, so
byte offsets are permanently wrong. Later passes that would otherwise resume cleanly cannot
re-align the file, and no amount of retrying fixes it — the fix is delete-and-restart, which
the script never tells anyone to do because it never notices.

Why 176/sec matters beyond the wasted 4 hours: with no backoff and no cap, a degraded CDN gets
hammered at ~15k requests/minute from a box that is also the one running agent3's device
window. That is how a transient rate-limit becomes a ban that costs the sprint a day, and how
bandwidth contention silently slows an unrelated lane.

For contrast, the same test against a **403-with-body** is benign: `curl` exits 33 on the
resume path and writes zero bytes. It is specifically the 200-carries-an-error-page case that
poisons, because to curl a 200 is a success.

## 4. Cheap detection that would have caught §3 in under a second

The container has a magic header, and the product already validates it:
`src/artifact/reader.cpp:284` throws `"artifact magic is not NInfer v2"`. Measured first bytes
of the live download: `4e 49 4e 46 45 52 00 02` → `NINFER\0\x02`, then JSON
(`{"identity":{"model_id":"qwen3.8-27b","weights_i…`). An HTML error page starts `<html>`.

So **check the first 8 bytes after the first kilobyte lands** (and again on completion). That
converts §3 from "four hours and a corrupt file" into "one second and a restart". It is not a
substitute for the sha — it catches the cheap case early, the sha catches the expensive one.

## 5. Hardening list (H1–H7), each traced to a measurement above

| # | change | why (measured) |
|---|---|---|
| H1 | Compare the digest inside the script: `sha256sum -c` against a recorded expected value; emit `COMPLETE-VERIFIED` vs `SHA-MISMATCH`; never write `COMPLETE` on size alone | §2 — `COMPLETE` currently means size-matched only |
| H2 | Add a retry **cap + exponential backoff** to the outer loop (e.g. 5 attempts: 10 s, 30 s, 2 m, 5 m, 10 m), then `FAILED attempts=N` | §3 — measured 176 attempts/sec, unbounded, zero backoff |
| H3 | Write a **heartbeat** line per attempt (`ts bytes rc`) to a separate file, so liveness is observable *during* the pull | §1 — `fetch.status` is terminal-only, so absence is uninformative for ~3 h |
| H4 | Validate the **magic header** (`NINFER\0\x02`) after the first KB and at completion; treat mismatch as poison → delete + restart, not resume | §3 + §4 — offsets are permanently wrong after an HTML prefix; the product itself enforces this magic at `reader.cpp:284` |
| H5 | Use `--fail-with-body`/`-f` **plus** a `Content-Type`/`Content-Length` sanity check on the response headers before writing any payload (HEAD request first, or `-o /dev/null -w '%{http_code} %{size_download}'`) | §3 — a 200 is success to curl; the body-write happens before any length check |
| H6 | Never resume into a file that has not passed H4; on restart-after-poison, **record the evidence first** (`size`, `sha256`, first/last 64 B) and only then delete | §3 unrecoverable-in-place + the project rule that a receipt precedes destructive cleanup |
| H7 | Do not leave the resume decision to `-C -` alone: if `HAVE > EXPECT`, stop and report (today the loop breaks and then prints SIZE-MISMATCH, which is correct but only after the fact) | §0 — over-length is a distinct failure (double-append), and the current text conflates it with under-length |

Ranked by what actually bit: H2 and H4 are the ones that change outcomes (4-hour silent spin,
unrecoverable partial); H1 and H3 are correctness-of-vocabulary; H5–H7 are belt-and-braces.

## 6. Plan of record for the Q3 v1a goal file (ask-4), when Green answers

15,446,796,288 B, sha256 prefix `7f26a0eb…`. **Do not hand-roll a second fetcher** — reuse the
hardened one with H1–H7, because §3 was discovered by accident and would otherwise be
re-inherited per artifact. Specifics:

- Expected full digest must be obtained from a channel other than the downloading host, or the
  check is decoration. `docs/CROSSLANE.md:87` carries the pin from the q3 lane (`63ff200c`).
- Disk: 49 G free with the reference pull occupying 19 G at completion → ~30 G headroom, which
  fits 14.4 G **but not twice**. A restart-after-poison must delete the bad file first (post
  H6 evidence). Check `df -h /` before starting, per standing rule.
- Report arrival to the coordinator immediately: P3 bring-up is meaningless without this file or
  an explicit reference-mode ruling.
- `q3_ci_amd.sh`'s default `ART=/home/chris/models/qwen3_8_27b_q3.ninfer` points at a directory
  that does not exist yet; whoever wires P3 should either create it or pass ART explicitly, and
  gemini's cell should fail loud rather than fall back if it is missing.

## 7. Watcher semantics (so its greens mean something)

`tools/v340l/artifact_watch.sh` — one pass = one verdict line, `--daemon` self-paces at
`INTERVAL` (default 1800 s) with its own pidfile. Verdicts: `IN-PROGRESS`,
`STALLED(alive,no-progress)` (consecutive-count based, resets on a legitimate restart),
`FETCHER-DEAD-INCOMPLETE`, `SIZE-MISMATCH`, `SIZE-COMPLETE(verify-pending)`, `VERIFIED`,
`UNKNOWN`. Exit codes: **0** in flight or verified, **2** a human action is needed, **3** the
pipeline cannot be observed at all (deliberately not folded into 2 — "cannot see" is not
"sees something wrong"), and `--stop` returns **4** when the pidfile names a live process that
is not this watcher.

`--stop` deserves its own paragraph because its first implementation was wrong in a way only
execution could show. bash defers trap delivery until the running foreground command returns,
so with `sleep 1800` in the loop, `--stop` printed "stopped own watcher", removed the pidfile
and exited 0 — while the process was verifiably still alive in `/proc` two seconds later. The
result is worse than a failed stop: nothing tracks the loop any more, so the next `--stop`
reports nothing to do while rows keep being appended. Fixed by backgrounding the sleep and
`wait`ing on it (interruptible), and killing the child from the handler; re-verified end to end
(process gone ≤2 s, pidfile cleaned). The refusal path is also separated from the no-op path:
an earlier test clobbered the pidfile with the **fetcher's** pid, and while the cmdline guard
correctly declined to kill it, the message said "nothing to stop" — hiding the one condition
that could otherwise end someone else's job. Own-pid discipline was held throughout: the only
signals sent were to pids whose `/proc/cmdline` names this watcher (or my own test processes),
never the fetcher, never `pkill` — and two "stray watcher" alarms during cleanup turned out to
be my own `grep`/`pgrep` pipelines matching their own pattern, which were checked rather than
killed.

Two of my own bugs, found by testing rather than reading, recorded because they are instructive:

1. First run reported `FETCHER-DEAD-INCOMPLETE` at 0% on a pull that was alive at 26%. Cause:
   `artifacts/` is **git-ignored by convention**, so a lane worktree has no such directory and
   an unobservable pipeline looked identical to a stopped fetcher. Fixed by resolving the main
   checkout via `git worktree list --porcelain`, honouring `$ART_DIR`, and — the part that
   matters — adding an `UNKNOWN` verdict for "cannot observe" that exits 3, because *the
   absence of evidence must never render as evidence of absence*. My whole lane is watching one
   specific thing, so this was the bug most worth catching.
2. Stall detection originally counted history rows sharing a byte size, so two rows from an
   unrelated earlier test made the next pass look STALLED. Replaced with an explicit consecutive
   run keyed on (pid, size).

Verification-cache rule, both directions measured: a stale `MISMATCH` marker must not keep a
repaired file looking broken, **and** a stale `MATCH` marker must not wave through changed
bytes — the second one reproduced as `rc=0 VERIFIED` on a byte-flipped artifact, i.e. a green
on bytes it never read. Rule now: short-circuit only on `MATCH` **and** marker newer than the
artifact, with the residual gap (same-length same-mtime rewrite) stated rather than hidden.

Measured on this box: full sha256 over 19.03 GiB takes **~11 s**. The cache therefore saves
almost nothing — so the honest recommendation is that any consumer about to *load* the artifact
re-hashes regardless, and the marker stays a convenience for the watcher loop only.

## 8b. FINAL STATE (19:5xZ) — supersedes the §8 estimates; receipts in results/amd/q3_arrival_verification.md

q3 **VERIFIED** (sha256 full match, magic + manifest parse, Q3G64_F16S=129 derived independently); ref **salvageable by one truncate** (prefix hashes to the registry digest). Both artifacts are therefore in hand for P3 — details, the correction to the `magic=BAD` board line (my synthetic fixture, not the artifact), and the fdinfo evidence for the two-writer tail are in the receipt. The §8 table above is kept as the working record of how each number moved, including my own three instrument faults on the way (page-cache reads reported as link rate; `-ge` labelling an over-length file complete; a `pgrep -f` matching its own pattern).

## 8. COMBINED ETA TABLE — rates, not hopes (coord request 18:40Z; refreshed by the 30-min daemon)

Sampled 18:38–18:57Z. Every rate is two or more size samples over ≥60 s; nothing here is an
estimate from a progress bar.

| subject | size now | % of pin | sustained rate | remaining | ETA | arrives in 24 h? |
|---|---|---|---|---|---|---|
| **ref** `qwen3_8_27b.ninfer` → `/` | 9.92 GB | 48.6% | **0.78 MB/s** (17.8 min window; inst. samples 0.54–1.25) | 10.51 GB | **3.7 h sustained → ~22:40Z** / 2.3 h at best sample → ~21:20Z | **YES, comfortably** |
| **q3** `qwen3_8_27b_q3.ninfer` → /media USB | ABSENT (attempt 5 not yet visible) | 0% of 15.45 GB | 0.513 MB/s best-ever (attempt 1); 0.30 MB/s attempt 4 | 15.45 GB | **8.4 h at best-ever, 14.3 h at attempt-4 rate — IF uninterrupted, which it never is** | **NO — see below** |

### Why the q3 ETA column is not the real answer
Four attempts, all now in `.Trash-1000/files/` with their byte counts intact:

| attempt | bytes moved | % of the 15,446,796,288 B pin | fate |
|---|---|---|---|
| 1 | 736,231,424 | 4.77% | birth 13:06:28 → last write 13:30:23 (1435 s, 0.513 MB/s), trashed 13:41:36 |
| 2 | 19,529,728 | 0.13% | trashed 13:42:17 |
| 3 | 22,020,096 | 0.14% | trashed 13:43:07 |
| 4 | 48,234,496 | 0.31% | birth 13:50:03 → last write 13:52:44 (161 s, 0.30 MB/s), trashed 13:50:53 per trashinfo — i.e. **trashed while the copier kept writing into the unlinked inode** |

Attempts 2–4 each lived ~50 s after creation. At 0.3–0.5 MB/s a 50 s window yields
15–25 MB = **0.1–0.16% of the file per attempt**, so completing it needs **320–800 attempts**
with zero resume credit between them — every restart begins again at byte 0, and every failure
discards what it wrote. The binding constraint is therefore **not the transfer rate but the
attempt lifetime**; a faster link would still deliver ~0.3% per attempt. Quote this, not the
8.4 h, in any P3 schedule.

### The source-path finding (coord task 1) — the discriminator they asked for, with a negative result
- The source is visible at `/tmp/RustDesk-1000/cliprdr-server/qwen3_8_27b_q3.ninfer` and
  reports apparent size **exactly 15,446,796,288 B** — 100.00% of the pin — while
  **allocating 1,885,696 B (0.0122%)**. `dd bs=1M count=4` returns **0 bytes**; `head -c 64`
  fails with EIO. It is a **RustDesk remote-desktop drive/clipboard redirect stub**, not a file.
- So the requested "time a `cp` of its first 200 MB" is **not performable**: there is no local
  stream to time. Reads succeed only while a live session is pushing that specific file, which
  is also why one read of the first 8 bytes returned a valid `NINFER\0\x02` header while the
  next returned EIO.
- This **discriminates the two hypotheses in favour of neither, and points at a third**: it is
  not the raw link rate alone (attempt 1 sustained 0.513 MB/s for 24 minutes), and it is not
  nautilus overhead alone — it is a **remote-desktop channel that carries the file lazily and
  drops the stream**, after which the destination partial gets trashed. A different local copy
  tool (`cp` instead of nautilus) rides the same channel and cannot fix it.
- **Danger this finding exposes, and it is the sharpest one in the document:** any size-based
  presence check — including the `find -size +10G` I used to locate it, and any guard that
  compares against 15,446,796,288 B — **passes a 0.012%-populated stub as the artifact**.
  The watcher now refuses on apparent-vs-allocated ratio and on a failed read, because a single
  successful header read is demonstrably not enough.

### Where this lands in our own guards (checked, not assumed)
`tools/bench/q3_ci_amd.sh:61` tests the artifact with a bare `[ ! -f "$ART" ]`. A RustDesk stub
satisfies `-f`, so that line alone would let a 0.0122%-populated placeholder past the presence
gate. The downstream failure is nevertheless LOUD, which is the saving grace: the reader throws
`artifact magic is not NInfer v2` (`reader.cpp:284`) once serve tries to load it, so the cell
red-fails at startup rather than passing quietly. Worth a one-line hardening anyway
(`-f` plus a size-and-allocation sanity or a magic sniff), because the failure mode is a red
cell 30 seconds later rather than an honest "artifact not present" note now — and because a
future caller that only checks `-f` for *scheduling* purposes would be fooled outright.

### Disk headroom (corrected, because the 18:40Z figures were not this device's)
Measured: `/` **47.90 GB avail** (ref pull needs 10.5 GB more → ~37 GB after), and
`/media/chris/EMTEC256` (sdb1, 239 GB, **rw**, mounted `rw,nosuid,nodev,relatime,uid=1000,…`)
**255.21 GB avail**. Two corrections to the board, both verified:
1. The mount is **read-write**, not `ro` — proven by a successful create+delete of a 0-byte
   probe file (removed immediately; no trash file touched). So "the drive's read-only state is
   the third failure mode" does not hold, and retries *could* have succeeded speed-wise; the
   ~50 s attempt lifetime is what killed them.
2. `92/130 G used, 38 G free` does not describe sdb1 (776 M used of 239 G, 1%). Both artifacts
   land on independent filesystems with ample room; **disk is not the binding constraint here**
   — the network/remote-session is.
