# 145 — Worktree & branch cleanup inventory (task #3)

Status: **INVENTORY ONLY — nothing deleted, nothing moved.** CPU-only; no GPU used.
Author: agent2. Date: 2026-09-03. Requested by: coordinator (seq 4, "worktree cleanup inventory").
Scope: the 13 worktrees under `/home/intel/ninfer/worktrees` + `/home/intel/ninfer/repo`, their
branches, builds, and disk. Every number below is from a command shown in §6 — re-run before acting.

> **Do not act on this doc without the two STOP items in §1.** The coordinator's ask said "12
> worktrees"; there are now **13** (`wo-shape-parity` appeared, gemini's docs/144 lane).

---

## 1. STOP items — read before deleting anything

### 🔴 S1. `github/main` is 194 commits stale — the GitHub default branch is 10 days behind

> **RULING (coordinator, seq 8): HOLD.** Do **not** push to `github/main` until the user decides
> which line is canonical. This is not a cleanup task — it is a branch-policy decision.

**The two-line explanation of why neither side can simply win:**

- **`main`** is the **radiance / optimization line** — it carries `docs/optimizations/44`–`45` and
  the `wo/kvarn-hold` merges (`55c383a3` merged hold: KVarN wall removal, packed-only decode, GDN
  T=1 Option B', mtp-adaptive, magic-dict, ctest CI gate).
- **`wo/kv-uniform`** is the **KVarN line** — the phase-gate / unified-KV / MultiBatch / docs/141
  verification work that everything since 2026-09-01 has been landing into.

Neither is an ancestor of the other (`main` = +67 unique vs `github/wo/kv-uniform`), so there is no
fast-forward and no objectively-correct merge direction: choosing one silently discards the other
line's identity. Hence HOLD + surface to the user, rather than "push main and move on."

The measured facts that motivate the ruling:

```
github/main tip:  c752d72c  2026-08-24
local  main tip:  3526ca5d  2026-09-03     (194 commits ahead of github/main)
main vs github/wo/kv-uniform: NOT an ancestor, +67 unique commits
```

### 🔴 S2. `/home/intel/ninfer/repo` (main) has 23 dirty paths including live source

**Composition, corrected after the coordinator's check — both numbers are right, they count
 different things: 5 *modified* + 18 *untracked* = 23.**

The 5 modified paths are **doc-45 W5 instrumentation**, not 2a work, and they form one coherent
in-progress feature (verified: `mtp_confidence_break` / `FirstMissHistogram` appears in all four
code files):

```
 M src/runtime/tp2/tp2_backend.cpp        } W5 instrumentation:
 M src/runtime/tp2/tp2_rounds.h           }  FirstMissHistogram, mtp_confidence_break.h include,
 M tests/CMakeLists.txt                   }  stats.first_miss
 M docs/optimizations/45_...md            }
 M docs/optimizations/README.md
?? src/runtime/tp2/mtp_confidence_break.h        <- new header, part of the same feature
?? tests/test_mtp_confidence_break.cpp           <- its test
?? results/w5_replay_bound.py                    <- its analysis script
```

> **RULING (coordinator, seq 8): protect before ANY cleanup** — commit these to a dedicated **W5
> branch** (or stash with a labeled ref) as part of the hygiene pass. Added to §7.

The remaining untracked paths are the four work orders (docs/139–142 — the **only** copies),
`docs/optimizations/46_kv_capacity_sizing_runbook.md`, `tools/bench/dg_table.py`, and 10 result
JSONs.

This is the **shared main checkout**, so `git worktree remove` / branch deletion is safe here, but a
`git checkout .` or `git clean -fd` would destroy someone's in-flight W5 work and the only copies
of four work orders.

### 🟠 S3. **Four** result files are named `--all-cache-type_decode_*.json`

**Correction to my first pass: I reported two; there are four.** I had listed the dirty set through
a `head -12` and counted what I saw instead of counting the files. Re-checked with `ls -1
results/--all-cache-type* | wc -l` → 4:

```
results/--all-cache-type_decode_20260830_164407.json
results/--all-cache-type_decode_20260830_171331.json
results/--all-cache-type_decode_20260830_180317.json
results/--all-cache-type_decode_20260830_195818.json
```

A `TAG` argument was captured as the literal string `--all-cache-type` (the flag was passed where
the tag was expected), so the filenames begin with `--`. They are real data, but any tool that
globs or `cat`s them without `--` will misparse. Cheap fix: rename all four + make
`decode_guard.sh` reject a `TAG` beginning with `-`. **Ruled into the hygiene pass (seq 8).**

---

## 2. Disk

```
/dev/sda2   228G total   201G used   16G free   94%   (snapshot at ~15:16; see §2b — it is 88% now)
```

| consumer | size | note |
|---|---|---|
| `worktrees/wo-2a-batched-serving/build` | **16.0G** | agent1's ACTIVE 2a build — **do not touch** |
| `worktrees/wo-shape-parity/build` | 992M | gemini's ACTIVE docs/144 build — **do not touch** |
| `bin/` | 338M | pinned binaries (e.g. `prefix_ninfer_decode_99111d52`) |
| `magic_dict_burnin_20260830/` | 17M | dated artifact, 2026-08-30 |
| `repo/` | 209M | main checkout |
| remaining 11 worktrees | ~88M each | **source only, no build dir** |

**Two builds account for ~94% of all worktree disk, and both belong to active agents.** So "worktree
cleanup" cannot recover meaningful space without stopping agent1 or gemini. The realistic
reclamation from inactive worktrees is ~1G, not 16G.

Corollary worth stating plainly: **disk pressure is not caused by worktree sprawl.** If space is
the actual problem, the lever is the 16G active build (ccache, `-j` object retention, or a
Release-without-debuginfo config), not the 11 small worktrees.

### Note on my own worktrees
`wo-4a4d-gate-verify`, `wo-4a4d-pinA`, `wo-4a4d-pinA2` each had a build dir (~1.8G total).
**All three are now gone** — deleted between my two `du` passes (~15:16 and ~00:0x) without a
message. The committed results are intact (`.ids` verified, compare still reproduces
30-identical/2-differ), but **re-running docs/141 now needs a fresh build** (~15 min each,
`cmake --build build --target ninfer_tp2_decode_test`). If that was a disk-pressure reaper rather
than a person, the same will happen to the next build anyone leaves in place.

---

## 2b. Build-dir disk analysis (coordinator's explicit ask, seq 6)

Disk moved 94% → **88% / 28G free** between my passes, so ~12G was reclaimed by someone. Only
~1.8G of that was my three build dirs, so the rest came from elsewhere — worth knowing the cause
before it recurs.

### Every build dir on the box

| build dir | size | owner / status |
|---|---|---|
| `worktrees/wo-2a-batched-serving/build` | **16G** | agent1 — **ACTIVE, do not touch** |
| `worktrees/wo-shape-parity/build` | 992M | gemini — **ACTIVE, do not touch** |
| `repo/build` | 18M | main checkout, essentially unbuilt |
| other 10 worktrees | 0 | no build dir at all |

### Root cause: it is NOT stale builds, debug info, or sprawl — it is static linking

Inside the 16G build:

```
build/tests   16G   <-- 100% of it
build/src    423M
build/apps   373M
build/CMakeFiles 7.9M
```

`build/tests` holds **134 executables, 15.0 GB, averaging 112 MB each; 84 of them exceed 100 MB.**
The reason is structural — every library in `src/CMakeLists.txt` is `STATIC`
(`ninfer_core:17`, `ninfer_ops:75`, `ninfer_engine:336`, `ninfer_serve:361`, …), and
**`libninfer_ops.a` alone is 191 MB** — the whole CUDA kernel set. Each of the 134 test binaries
statically re-links a large slice of that same archive.

Measured, not assumed:

```
ninfer_sampling_defaults_test   184.5 MB
  after `strip`:                178.5 MB   <- only 3% smaller
```

So **debug info is not the problem** — ~178 MB of each binary is real embedded kernel code. A
`strip` pass, `-g0`, or a no-debuginfo config would reclaim almost nothing. Anyone reaching for
those first will waste the afternoon.

### Ranked recommendations

| # | action | est. reclaim | effort | risk |
|---|---|---|---|---|
| 1 | **Link `ninfer_ops`/`ninfer_engine` SHARED** (or an `OBJECT` lib + one shared consumer). 134 binaries × ~112 MB → 134 × ~3 MB + one ~191 MB `.so` | **~14 GB** | real task | CUDA fatbin registration + symbol visibility; tests poking internals may need rework. **Evaluate, do not flag-flip.** |
| 2 | **Target-scoped builds in dev worktrees** — build only what you run | ~90% per worktree | trivial | none; this is what I did (3 targets ≈ 1 GB vs 15 GB) |
| 3 | **`ccache`** — not installed on this box (`which ccache` empty) | 0 GB, but ~15→2 min rebuilds | small | none; helps time, not space |
| 4 | Prune never-run test targets (the `_real_test` model-loading variants) | ~1–2 GB | small | loses coverage — needs owner sign-off |
| 5 | `strip` / `-g0` / debuginfo config | **~3%** | small | pointless here |

**The one-line version: the box's disk problem is that `run_ci.sh:138` runs
`cmake --build . -j$(nproc)` with no target filter, and every one of ~134 test binaries statically
re-links a 191 MB CUDA kernel archive. Fix the link model (rec #1) and a full CI build drops from
~15 GB to well under 1 GB, which also makes per-worktree builds affordable instead of a luxury.**

Corollary for cleanup priority: **removing all 4 clearly-safe worktrees reclaims ~350M** — they
have no build dirs. Worktree sprawl is a legibility problem. Disk is a build-system problem, and
neither agent1's nor gemini's build is the culprit; both are active and correctly sized for what
they contain.

---

## 3. Worktree table (13)

| worktree | branch | dirty | vs `github/wo/kv-uniform` | build | last commit | verdict |
|---|---|---|---|---|---|---|
| `repo` | `main` | **23** | +67 unique (diverged) | 18M | 09-03 | 🔴 S1+S2, resolve first |
| `wo-2a-batched-serving` | `wo/2a-batched-serving` | 4 | +9 (active) | **16G** | 09-03 | 🟢 ACTIVE agent1 |
| `wo-shape-parity` | `wo/shape-parity` | 0 | +8 (active) | 992M | 09-03 | 🟢 ACTIVE gemini |
| `wo-kv-uniform` | `wo/kv-uniform` | 6 | trunk | none | 09-03 | 🟢 TRUNK checkout |
| `wo-kvarn-prefill` | `wo/kvarn-prefill` | 1 | +10 unique, pushed | none | 09-03 | 🟡 unmerged, 1 dirty — ask owner |
| `wo-phase-gate-m1` | `wo/phase-gate-m1` | 0 | +5 unique, **LOCAL-ONLY** | none | 09-02 | 🟠 unmerged + unpushed |
| `wo-kv-uniform-ci-gate` | `wo-kvarn-lane-fix` | **15** | MERGED | none | 09-02 | 🟠 **name mismatch** + 15 dirty |
| `kv-mtpfix` | `wo/kvarn-mtpfix` | 0 | MERGED | none | 09-02 | ⚪ removable (merged) |
| `wo-phase-gate` | `wo/phase-gate` | 0 | MERGED | none | 09-02 | ⚪ removable (merged) |
| `wo-phase-gate-i8` | `wo/phase-gate-i8` | **14** | MERGED | none | 09-03 | 🟠 merged but 14 dirty — inspect |
| `wo-4a4d-gate-verify` | `wo-4a4d-gate-verify` | 0 | MERGED | none | 09-03 | ⚪ removable (mine, merged+pushed) |
| `wo-4a4d-pinA` | **(detached)** @ `2b6f0019` | 0 | n/a | none | 08-31 | ⚪ removable — see §4 |
| `wo-4a4d-pinA2` | `wo-4a4d-pinA2-harness` | 2 | MERGED | none | 08-30 | 🟡 D4 baseline, keep until B2/B4 |

**Branch↔worktree name mismatches** (each is a footgun: `cd worktrees/X; git status` reports a
different branch than the directory implies):
- `wo-kv-uniform-ci-gate` → `wo-kvarn-lane-fix` (worst case: the dir name says "ci-gate", the
  branch says "lane-fix", and it has 15 dirty paths)
- `kv-mtpfix` → `wo/kvarn-mtpfix`
- `wo-4a4d-pinA` → detached HEAD
- `wo-4a4d-pinA2` → `wo-4a4d-pinA2-harness`

---

## 4. Why `wo-4a4d-pinA` is safe to drop but `wo-4a4d-pinA2` is not

`pinA` is a detached checkout of `2b6f0019`, created solely to test whether the natural D4 pin
builds. **It does not compile** (docs/141 §5.1: `29d0a797` introduced a 2-arg template call
`packed_kernel<false, skip>` against a 1-param kernel `template <bool Int8QK>`; fixed only at
`aa110a22`). That finding is now permanently recorded in the commit message of `2bff1d5d`, in
docs/120 §D-D4, and in docs/127 row D4. The worktree adds nothing.

`pinA2` is the **D4 baseline build recipe**: `aa8ef21b` + cherry-pick `80b8a661`. Its value is
that the cherry-pick is provably zero-`src/`, which is what makes the pinned comparison legitimate.
Recommend keeping until B2/B4 (direct-route wiring) lands, or until the coordinator signs off
D4 as closed-for-good. It is only 85M without its build dir.

---

## 5. Branches (33 local)

**17 fully merged into `github/wo/kv-uniform`** — safe to delete as branches (worktrees first if attached):
`v1-integrate`, `wo-4a4d-gate-verify`, `wo-4a4d-pinA2-harness`, `wo-kvarn-lane-fix`,
`wo/200k-context`, `wo/auto-kv`, `wo/ci-wiring`, `wo/kv-uniform`, `wo/kv-uniform-ci-gate`,
`wo/kvarn-attn-verify`, `wo/kvarn-mtpfix`, `wo/kvarn-pp`, `wo/mtp-tests`, `wo/phase-gate`,
`wo/phase-gate-i8`, `wo/remote-ports`, `wo/tool-calls`.

**11 NOT merged** — do not delete:

| branch | unique | pushed? | risk if deleted |
|---|---|---|---|
| `main` | +67 | yes (but github/main 194 stale) | 🔴 S1 |
| `wo/2a-batched-serving` | +9 | yes | agent1 active |
| `wo/shape-parity` | +8 | **LOCAL-ONLY** | 🟠 gemini active, no remote copy |
| `wo/kvarn-prefill` | +10 | yes | 🟡 owner unknown |
| `wo/mtp-adaptive` | +44 | yes | content in `wo/kvarn-hold`? verify |
| `wo/magic-dict` | +36 | **LOCAL-ONLY** | 🟠 no remote copy |
| `wo/magic-p2` | +35 | **LOCAL-ONLY** | 🟠 no remote copy |
| `wo/phase-gate-m1` | +5 | **LOCAL-ONLY** | 🟠 no remote copy |
| `wo/kvarn-layout-verify` | +4 | **LOCAL-ONLY** | 🟠 no remote copy |
| `wo/kvarn-d21` | +2 | **LOCAL-ONLY** | 🟠 no remote copy |
| `wo/kvarn-hold` | +62 | yes | hold branch, likely intentional |

**The 6 LOCAL-ONLY unmerged branches are the real data-loss risk in this inventory.** Verified
per branch rather than assumed — the picture splits into two different severities:

| branch | unique | content in local `main`? | actual exposure |
|---|---|---|---|
| `wo/shape-parity` | +8 | **NO** | 🔴 **genuinely single-copy on disk** (gemini, active) |
| `wo/phase-gate-m1` | +5 | **NO** | 🔴 **genuinely single-copy on disk** |
| `wo/kvarn-layout-verify` | +4 | **NO** | 🔴 **genuinely single-copy on disk** |
| `wo/kvarn-d21` | +2 | **NO** | 🔴 **genuinely single-copy on disk** |
| `wo/magic-dict` | +36 | yes | 🟠 in the local object store, but see below |
| `wo/magic-p2` | +35 | yes | 🟠 in the local object store, but see below |
| `wo/mtp-adaptive` | +44 | yes | 🟠 in the local object store, but see below |

So **19 commits are genuinely single-copy** (the four NOT-in-main branches), not the ~120 the raw
counts suggest — I checked rather than summing. The magic-dict/magic-p2/mtp-adaptive content is
replicated into local `main`, so it survives deletion of those refs.

**But the second row is not as safe as it looks:** `main` itself is **194 commits ahead of
`github/main`** (§1 S1). Content that exists only "in main" therefore exists only on this disk
too. Pushing `main` (S1) is what actually de-risks all seven rows — which is why S1 is first in §7
and not the worktree removals.

---

## 6. Commands used (re-run before acting — this is a snapshot)

```bash
cd /home/intel/ninfer/repo && git fetch github wo/kv-uniform main
git worktree list --porcelain
du -sh /home/intel/ninfer/worktrees/* | sort -h
df -h / | tail -1
# merged?  (per branch)
git merge-base --is-ancestor <branch> github/wo/kv-uniform && echo MERGED
# unpushed
git rev-list --count github/main..main
git rev-list --count github/wo/kv-uniform..<branch>
# local-only
git rev-parse --verify -q github/<branch> || echo LOCAL-ONLY
# dirty
git -C <worktree> status --porcelain | wc -l
```

No `git worktree remove`, `git branch -d`, `git prune`, `rm -rf`, or `git clean` was run for this
doc. `git worktree prune --dry-run` reports nothing prunable; all 13 worktree directories exist.

---

## 7. Recommended order (updated with coordinator rulings, seq 8)

0. **If disk is the actual concern, fix the build system, not the worktrees** — §2b rec #1
   (shared linking, ~14 GB) or rec #2 (target-scoped builds, trivial). This outranks everything
   below on the disk axis.
   - **Shared-link = separate task.** Per ruling: propose a WO **at closeout, not now**.
   - **Target-scoped dev builds = cheap immediate win.** Per ruling: note it in `run_ci.sh` hygiene.
   - Disk root cause folds into the **closeout doc** (§2b is written to be lifted whole).
1. **S1 — HOLD.** No push to `github/main` until the user picks the canonical line. Surfaced.
2. **Push the 4 genuinely-single-copy branches** (`wo/shape-parity`, `wo/phase-gate-m1`,
   `wo/kvarn-layout-verify`, `wo/kvarn-d21` — 19 commits total, §5). The magic-dict/magic-p2/
   mtp-adaptive refs are covered by pushing `main`, which is itself held pending S1.
3. **S2 — protect the W5 work before ANY cleanup**: commit the 5 modified + related untracked paths
   to a dedicated W5 branch (or a labeled stash ref), and commit docs/139–142. Nothing in §4 may
   run against `repo/` until this is done.
4. **S3 — rename the four `--all-cache-type_*.json`** files + guard `TAG` in `decode_guard.sh`.
5. Only then remove worktrees: `wo-4a4d-pinA` (spent, §4), `wo-4a4d-gate-verify` (merged+pushed),
   `kv-mtpfix`, `wo-phase-gate`. ~350M. Reclaim is negligible — the point is clarity, not space.
6. **Inspect before removing** the 4 merged-but-dirty worktrees (`wo-kv-uniform-ci-gate` 15,
   `wo-phase-gate-i8` 14, `wo-kv-uniform` 6, `wo-kvarn-prefill` 1) — dirty means unreviewed work.
7. Leave `wo-2a-batched-serving`, `wo-shape-parity`, `wo-kv-uniform` alone: active + trunk.

### §2b reaper note — status after investigation
The coordinator checked cron / crontab / systemd timers and found **no automated disk cleaner**.
Attribution is unresolved; most likely the prior agent1 session (`01a067cc`, since restarted — its
intercom ID is now `01a0697c`) or a human. **D5 risk accepted-low**: a fresh build is ~15 min on
one tree, with the caveat to watch for mid-build directory deletion. Keeping this note per ruling.

**Bottom line:** worktree sprawl is a legibility problem, not a disk problem. The disk problem is
one 16G active build; the durability problem is `github/main` being 194 commits stale plus 19
genuinely single-copy commits on 4 unpushed branches. I'd spend the cleanup effort on §7 items 1–3
and treat the worktree removals as cosmetic.

---

## 8. Hygiene items queued by this pass (recorded, deliberately NOT actioned)

Per coordinator ruling seq 6, these are listed so they are not lost, and left unedited until
assigned to a commit that can carry them:

1. **Stale route comment** — `src/ops/launcher/gqa_attention_kvarn.cu:376` reads
   *"Packed decode kernel (default route)."* Unified has been the default since `29d0a797`. In a
   file whose `[kvarn-decode-route]` banner exists precisely because "the number alone is not
   evidence", a comment naming packed as the default is what leads a reader to the hypothesis that
   was just ruled out for the 2a MTP deviant. 1-liner; belongs in the 2a-land or closeout commit.
2. **`direct`-route false-pass hazard** — the `NINFER_KVARN_PREFILL=direct` throw at `:300` is
   unreachable below T≥7 (`packed_verify` early-returns at T=2..6), so a short-prompt A/B appears
   to succeed. Worth a comment at the throw site when B2/B4 starts. docs/120 §B2 + docs/127 row D2
   carry the prose; code is what gets read at that moment.
3. **`packed_verify` naming trap** — `:226` does **not** select the packed decode kernel despite the
   name; it is the predicate that keeps a verify round out of the materialize prefill branch. Cost
   me a minute and could mislead the 2a bisect.
4. **S3 filename bug** — rename `results/--all-cache-type_decode_*.json`, and make
   `decode_guard.sh` reject a `TAG` that begins with `-`.
5. **Correction-debt already amended** (coordinator seq 6): docs/141 §5.4 "convergence/fix"
   language dropped — packed/unified non-identity is by design (`:349-356`, 64-key vs 32-key MMA
   accumulation order). Retained: packed output change at 1024-token ctx attributed to `9a58008a`
   (sole commit in range), and the perf-neutral-label-with-no-long-ctx-model-test observation.

### 8.6 `run_ci.sh` verify-data load is fragile — found while pre-flighting D5

`tools/ops/run_ci.sh:556-559`:

```python
verify_jsons = sorted(glob.glob(f"{results_dir}/[0-9]*.json"))
verify_data = {}
if verify_jsons:
    with open(verify_jsons[-1]) as f:      # "latest" by NAME SORT, not mtime
        verify_data = json.load(f)
```

Two independent weaknesses, and they compound:

1. **`[0-9]*.json` + `[-1]` is an ASCII sort, not a recency sort.** Any filename whose first
   digit is greater than `2` sorts after every `2026…` timestamp. I landed exactly such a file —
   `results/4b_decode_20260903_145944.json` (`4` > `2`) — which therefore became the file CI loads,
   silently displacing the real latest `_ci.json`.
2. **The load is not wrapped in try/except, and the block's exit code is discarded.** The final
   `RC` at `:725-733` is computed only from the shell step exits (`VERIFY_EXIT`, `SERVE_EXIT`, …).
   So when `json.load` raises, the report block dies, `${TS}_ci.json` and `latest.json` are never
   written, and the script **still prints `CI: PASS ✓`**.

The gates that live inside that block — `determinism_ok`, `a2_identity_ok`, `kv_i8_ok`, and the
`mtp_accept_pct` regression check (`:690-699`) — are therefore **skipped without any signal**.
For a completion gate that is the worst available failure mode: a false PASS.

**This was live and would have fired on the next `run_ci.sh` run, i.e. D5.** Confirmed by
reproducing `:556-559` verbatim: `JSONDecodeError: Extra data: line 2 column 1` (the file is
JSONL — six concatenated objects, not one document). 39 other digit-prefixed JSONs all parse, so
mine was the only bad one, and it was the only one that mattered because it sorted last.

Mitigated in docs/141 branch by `git mv` into `results/141_gate_verify/` (the glob is not
recursive), which restores `20260903_124533_ci.json` as the loaded file — verified by re-running
the exact load path. **The structural fix is NOT done, deliberately:** it is a `tools/ops` change,
and the D5 merge was ruled docs+results-only so it stays CI-outcome-neutral.

Recommended separate fix (needs its own WO, not mine to fold into a closeout merge):
- wrap the load in `try/except` and treat a failure as a hard CI error, not a silent skip;
- select by `max(..., key=os.path.getmtime)` or match `*_ci.json` explicitly rather than any
  digit-prefixed JSON;
- propagate the report block's exit code into `RC`.

---

## 9. Two cross-cutting items for the closeout doc (raised jointly with agent1, seq 8)

### 9.1 Repo-level coverage gap — state once, not per-battery

**Chat-templated prompts are an input class that no MTP-losslessness or A2 parity cell in this
repo currently exercises.** Raw-text harness prompts cannot produce the greedy near-tie that breaks
A2 identity, so no amount of length or repetition on raw text will surface it.

This applies identically to three separate bodies of evidence, which is why it belongs at repo
level rather than as a caveat inside any one battery:
- agent1's §4.5 single-seq MTP probes (the deviant survived a full session of green gates for
  exactly this reason),
- my docs/141 A2 identity cells (raw `--prompt` / `--prompt-file` only — see §5.6 there),
- the M4 gate's raw-text prompts.

Standing caveat for anyone reusing an A2 or MTP-lossless PASS: **check which input class produced
it.** Raw-text parity is not shipped-traffic parity.

### 9.2 The session's named anti-pattern

Three of us hit the same failure from three different directions:

| who | instance | what vanished |
|---|---|---|
| agent2 | `results/4b_decode_*.json` hijacked `run_ci.sh`'s verify load | the gate never ran; `CI: PASS ✓` still printed |
| agent1 | `getenv("NINFER_MB_SERVE_MUTATE)=0` is **true** — `X=0` set the var | the clean arm was actually the mutated arm |
| agent1 | `sha_of` swallowed exceptions → `[ "" = "" ]` compared equal | determinism gate passed on a run where every request errored |
| agent2 | §5.4 "convergence" retracted in chat at seq 12, never deleted from the file | the retraction existed only in a message |
| agent1 | approved B0/B1 re-point shipped as something different, unannounced | the approved design and the landed design diverged |

**Generalization (agent1's formulation, seq 8): the authoritative artifact is the one that gets
executed, not the one that gets asserted.** A green verdict whose gate vanished, a retraction that
lives only in chat, an approved design that quietly became a different design — all three are the
document and the message disagreeing, with the message winning in people's memory.

Mechanical consequence adopted for D5: `146_audit_verdict_block.sh` re-derives the gate structure
from the file at stamp time rather than trusting a merge report or a verdict string. Short form for
the closeout: **verify the gate ran; don't trust the verdict string.**
