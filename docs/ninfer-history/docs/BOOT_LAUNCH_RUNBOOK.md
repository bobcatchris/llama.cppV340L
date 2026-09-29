# BOOT-LAUNCH RUNBOOK — the dev0 margin incident, and how a short-context benchmark takes minutes (not an hour)

## 0. PROMOTED SERVING BUILD (2026-09-20 — read this first before any boot)

- **Serving default = `ninfer-serve_cc122cc0ab616e4e.bin`** (mad-mix NVFP4 GEMM build, G-MM-2
  promoted at +8.83% serving wall, PLOG-080). `serve_10k.sh` BIN points here; `serve_fast.sh`
  inherits. Cutover-verified end-to-end (FAST PROBE 16.9 s, BLUE + 3k-needle exact,
  WO_MADMIX_cutover_row.txt).
- **Rollbacks, two layers**: boot-rollback = `ninfer-serve_2c8901d3d18adef1.bin`; runtime
  rollback = env `NINFER_MADMIX=0` on ANY bin (restores the V0 kernel path, proven on the
  promoted binary: FAST PROBE PASS + BLUE at stop).
- **KV-tier note (2026-09-20)**: serving KV stays BF16 until the k4v4 promotion decision
  lands (k4v4fin desk); when it flips, `--kv-dtype kvarn_k4v4` + `NINFER_WORKSPACE_MIB=512`
  are the verified long-context posture (7755/8160), k4v2 = rollback dial.
- Probe tooling contract (REQUIRED by the serve): `"model":"qwen3.8-27b"` field + answers
  may land in `reasoning_content` (reader: reasoning first, then content;
  `tools/v340l/w7_probe.sh` encodes both; needle probes need max_tokens >= 128).

**Author**: coordinator pi 01a09a03 (chair), 2026-09-13 ~10:5xZ. **Trigger**: user order — "the issue with the card launch needs to be documented so we shouldn't take an hour to run a short context benchmark." Everything below is chair-measured this session or copied from committed rows (citations inline); nothing is convention.

## 0a. STANDING BOOT RULES (2026-09-20)
- **EXTRACT BEFORE RESTORE** on any instrumented boot: serve_fast/serve_10k truncate their
  logs on every boot — a restore before extraction destroys the raw trace (PLOG-087 error;
  tooling fix: tools/v340l/wo_census_leg.sh enforces it by construction).
- **+3-MIN LIVENESS**: a canonical restore is not a close until health passes AND stays
  passing at +3 minutes (PLOG-084: one restore exited gracefully ~90 s post-probe).
- **ERA-MYOPIA LAW**: every ceiling/capacity claim cites its manifest era — boot logs beat
  derivations (PLOG-088: bf16 TP4 once booted 98,944 tokens at placement 3,820 MiB/rank;
  today's 5,131 MiB manifest caps bf16 at 36,352).

## 0b. THE PROMOTED KV TIER (2026-09-20, PLOG-084)

- Default line (both scripts): **kvarn_k4v4 @ max-context 65536 / capacity 65536, NINFER_WORKSPACE_MIB=512**, bin **ninfer-serve_058a7b85859c5cd0.bin** (UNION: mad-mix + full kvarn; amd/main ece88d382).
- **ws512 is MANDATORY with kvarn**: the 96 MiB chunk arena bad_allocs kvarn prefill past ~12.3k real tokens (PLOG-079); 512 lifts the position ceiling to 65,536 (KV-B2 cell). Contexts >65,536 need docs/120 B2 (parked).
- Rollback ladder: `--kv-dtype kvarn_k4v2` (V@2-bit dial, same posture) -> strip kvarn flags + ws512->96 + BIN cc122cc0ab616e4e (bf16 36,352 default).
- **Lineage law**: a bin refusing `--kv-dtype kvarn_*` at parse predates the cached-route link — do not "fix" the script, rebuild from amd/main tip (cc122cc0 incident, PLOG-084).
- Every kvarn window closes with the KVARN_BOOT_BATTERY cells (KV-B0/B1/B2) + liveness at health-OK AND +3 min (the 90 s observation, PLOG-084).

## 1. What actually happened (timeline, 2026-09-13)

- **09:25:12Z** — 17g attempt #1 dies OOM: dev0 free=816 MiB at spawn (agent4 death row `31ab20ef`/`c30351b1`). Cause: agent2's user-task boot held ~7.2 GiB. Legitimate work (user directive), but unannounced — the release-row gap.
- **09:51Z, 10:0xZ** — attempts by agent4 self-refused twice on their own freshness gate (correct instrument behavior, fixes at `a2ac1f2d`).
- **10:0xZ** — 17g2: **no contention** (dev0 free=8158 at spawn), ladder byte-identical to the green 17f boot, dies at ONE 396 MiB reserve. Binary sha256 **identical to 17f's** → the entire merge wave is codegen-inert (row `252d5b5b`). Diagnosis: environmental margin — 17f fit by **22 MiB = 0.27% of the card** because GPU0 (PCI 05:00) carries the desktop: Xorg pid 2035, gnome-shell 2856, rustdesk 1595, measured between 139 MB and 8 GB-swings through the hour.
- **10:2xZ** — agent4 gate-samples dev0: 8043/8043/8042 vs my ≥8100 floor → DEFERS, third time, correctly. Chair escalates to the user for ~60 MB of desktop.
- **10:3xZ — the user's correction**: "we have 2 cards." Measured answer: all four dies are mutually **weight-40 / 2-hop / PCIE** (`rocm-smi --showtopo`) — `dev2,3` is a topologically IDENTICAL, desktop-free pair (baseline 18,575,360 B each). The "dev0/1 = boot pair" line in the restart kits was a CONVENTION, not physics. Pair swapped. **17g3 then booted green on dev2,3 in one attempt, end-to-end ~4 minutes of card time.**

**The hour was not compute. It was: convention nobody measured (pair lock), a shared-resource holder nobody announced (desktop on dev0), and a config that fit by 0.27% (17f geometry). All three were resolvable by two commands in minute one.**

## 2. Standing laws from this incident (board-wide)

1. **A fixed resource assignment is a claim, not a fact.** Any "blocked on device X" must be accompanied by "and I checked whether X is the only place X can happen" — equivalence decided by `--showtopo` + baseline VRAM reads (minutes), not by the convention's age. (Posted at #board 10:4xZ; this doc is its permanent home.)
2. **A stamp that outlives its measurement is a claim about the past in present tense — re-derive or delete, never carry** (LAW 20 corollary, agent2). Gate verdicts carry their ref: `VIOLATIONS=n @ sha`.
3. **Device rights are announced rights**: any boot posts a manifest row (argv + artifact path + expected window) BEFORE spawning, and a release row AFTER, including deaths. Contention between silent boots is the one failure mode this runbook exists to kill.
4. **KFD checks must be tab-tolerant and raw-dumped** — the space-vs-tab grep false-green class (agent4's kit confession; kit v3+ fixed, raw dump per boot mandatory).
5. **GRANT-ACK-IN-ARGUMENTS (device-kit law, 2026-09-13, from the VOID_UNGRANTED incident, agent3):** any script that can touch a device must REFUSE at zero device touch unless invoked with the written grant's name in its ARGUMENTS/ENV (`T3_I7_GRANT_ACK=G-AMD-27` pattern, agent3's kit), on its OWN exit code that no finding ever wears (exit-7 class: a broken gate must never masquerade as a result). Root cause generalized: **a verb is not a precondition — "dry run" described the intent, not the command's effect.** Falsifier once lanes adopt it: invoke any device script un-stamped; anything but the named refusal exit at zero device touch is a violation. Prior same-shape instances: agent4's G-AMD-26 cell (exit-77 self-gate), kit v6 freshness-refuse.
6. **RESIDENCE BEFORE SPAWN (standing clause, agent2 WARM_CACHE b06ce67b):** run `results/amd/p3/residence_probe.py` (mincore, 0.09 s, no root) before any boot window — it says whether the boot pays disk (cold road ~43 MB/s ≈ 6 min full-file; hot ~1 s; a served boot self-warms ~93%). Post-pressure re-warm: `dd if=/media/chris/EMTEC256/qwen3_8_27b_q3.ninfer of=/dev/null bs=8M`. Page cache yields to GPU-pinning boots and builds — expect 90-100% across build-free gaps, less after pressure. Residence before verdict.

## 3. HOW TO RUN A SHORT-CONTEXT BENCHMARK — the minutes path

Cards (measured 2026-09-13): **dev0 = DISPLAY CARD — DO NOT BOOT PAIRS ON IT** (desktop furniture 60–140 MB, nobody owns it, nobody should kill it). Clean pairs: **dev2,3** (preferred) or dev1 + (dev2 or dev3). Topology is symmetric, so any pair is equivalent for correctness; AR bandwidth identical (no P2P on any pair).

**The five facts that cost an hour to learn:**
- `HIP_VISIBLE_DEVICES=2,3` **remaps the selected cards to logical 0,1** — with the mask set, `--devices` must STAY `0,1`; literal `--devices 2,3` is refused "invalid CUDA device id" in 0.1 s (kit v5, `9c2a7876`).
- Artifact: use `/media/chris/EMTEC256/qwen3_8_27b_q3.ninfer` (15,446,796,288 B). `/home/chris/Desktop/…q3.ninfer` is **TRUNCATED** (6.6 GB, 740/1124 objects past EOF — throws at layer/31 gate_value; agent2 manifest parse, `9df53116`). EMTEC256 is local ext4: desktop_f is the same bytes over CIFS (SMB pull of 15 GB is the slow road).
- Load is ~31 s from EMTEC256 (17f row); whole smoke cycle ≈ 4–5 min per boot.
- The 17f config (ws96, ctx/kv256, chunk128) fits by 22 MiB on a polluted card; **on dev2,3 the same config ran first-try green.**
- Kill only your own recorded pid/pgid (kit does this); never a pattern.

**Fast path, single source of truth — the kit** (`results/amd/p3/G3_bringup_kit.sh` on `amd/wo-p3-serve`, v5): it carries the canonical artifact path (`Q3_CANONICAL_PATH`), `DEVS` parameter (default `0,1` launch shape — pass `HIP_VISIBLE_DEVICES=2,3`, keep `--devices 0,1`), ws96/ctx256/kv256/chunk128/no-prefix/no-graph/greedy, KFD tab-tolerant precheck + raw dump, freshness gate, argv+bin-sha stamping in the window log, release-row emission. Read its own header — kit v7+ (agent4, 0aa55e1d era): bare `bash G3_bringup_kit.sh` REFUSES at zero device touch (STAMP_ACK law #5, exit-77); pass the named grant in env. Two independent gates (kit+cell) between a ghost boot and any lane. Expected: precheck-zero → listening in ~35 s → probe → release row, all in one invocation.

**For TPS specifically**: `tools/v340l/tps_probe.py` (on main since `a9d8792`… correction: `a9d38792`) is stream-timed, stdlib-only, both failure directions stub-proven (`ee380b7d`). Point it at the live port; report MARGINAL per-token cost at several gen-lengths (0.8-vs-14.1 lesson: averages hide the quantity you mean; smoke rows prompt=54/gen=32 are NOT throughput results).

**First numbers, 17g3 (dev2,3, 2026-09-13T10:46Z, /tmp/g3_serve.log)**: prefill 16.5 tok/s (window) / 14.9 (per-req), decode 9.8 tok/s (window) / ~4.x (per-req) at ctx-256 smoke shape — provisional, pending agent4's coherence row and the real-context cell.

## 4. Residuals this doc does NOT close

- **BANK-BEFORE-RELINK (law, 2026-09-13, from the era-binary clobber):** any binary or archive stamped by a boot row (`FRESHNESS bin_sha256=...`) is a **truth artifact**: before re-running a build that can relink it, `cp -p` it to `/home/chris/artifacts_bin/<name>_<sha16>.bin` and commit a RECORD row (path, full sha, recipe, date) — the 122 MB artifact never needs to enter git, its *name and sha* do. Incident: agent4's A-1/A-2 verification build relinked the SOLE copy of the 17g-era binary (`8e6d79ae…`, pinned by agent3's item-7 replay method) at 13:12Z, six minutes before the preservation order existed. Prior named instance (agent5, 3cde3b9b): same source+flags produced md5 b1033fa2 ≠ captured 58dfe1e3 — **linker build-id drift is why preservation is a COPY, never a rebuild-to-match promise**: a re-link is a new artifact by default; the era only survives if you bank the bytes that booted. Recovery of the era ran as a machine-decided test — rebuild at `d32d7d23` (era inputs; `git diff` of the three stamped heads' build inputs EMPTY) + full-sha match against banked FRESHNESS rows; match ⇒ era restored, mismatch ⇒ era closes loudly and every pin re-scopes. Lane conduct note: agent4 self-reported, searched the inode, froze builds, and surfaced the reconstruction argument WITHOUT acting on it — the good shape. An in-place relink is normal lane life; a pinned-by-sha replay artifact makes preservation part of the grant. **Corollary (chair law f49dfe4a): pinned binaries BOOT from the bank — `/home/chris/artifacts_bin/<name>_<sha16>.bin`, filename-is-stamp — never from a lane's `build-hip-amd/apps/` path, which is a moving target by design.**

- The 22 MiB fit-margin of 17f geometry is still thin — dev2,3 green doesn't make it roomy; a ws/ctx relief boot remains a candidate future truth (own stamp, capacity-deviation logged).
- The auto-KV legacy-estimate (9059 MiB) placement-basis fallback still exists as a measured refusal hazard on polluted cards; explicit capacity bypasses it; the VRAM-law fix (measured-floor gate) is the filed defect, not this doc's job.
- If a future incident needs the dev0 pair anyway (lease rows pin history there), the human decision is the user's: move the desktop, or boot elsewhere — "elsewhere" now officially exists.
