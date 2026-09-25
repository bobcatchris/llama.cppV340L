# ONESHOT-W4 IMPLEMENTATION NOTES — the W4-LOCKSTEP-LIVENESS cure, landed arms + GPU-owner one-window runbook

Lane: `amd/oneshot-ar` (branched from `amd/tp4-cure` @ `248630d1b`). CODE-desk seat,
NO GPU touched (compile-only + host cells). Design of record:
`docs/amd/ONESHOT_W4_BOUNDED_WAIT_DESIGN.md` (eb365119f) — this file is the
landing report; the design doc's §4a GREEN-leg plan is what the GPU window runs.

RED being cured: `NINFER_TP_ONESHOT_AR=1` at TP4 wedges the boot deterministically
(2/2, bin `409450b83ffa4be8`, commit `b420b86fe`): warmup never completes, 3 GPUs
pinned at 100% (1 CU each), stranded rank varies, NO AR-FAILOUT — 17+ min silent.
CLASS `W4-LOCKSTEP-LIVENESS`: system-unbounded collective wedge behind per-launch
bounded waits ((i) monotone `>=` gates silently accept wrong-generation stamps;
(ii) the one-shot kernel queues behind unbounded RCCL work on the shared rank
stream so its bounded wait is unreachable; (iii) no end-to-end watchdog).

---

## 1. What landed (bounds)

All three bounds arm ONLY the env-gated world>2 one-shot arm. Default behavior
(env absent, every world) is byte-identical by construction — see §3.

### B1 — per-launch poll + stale-pass census (design §2 B1, §2d, §2e)
`one_shot_allreduce.cu`, `ar_wait_peer` (HIP arm; the CUDA arm is R7-untouched —
NVIDIA seat's cell):
- Outcome PASS-fresh: peer exactly at my call — unchanged semantics.
- Outcome PASS-stale/ahead: gate passed with `gen_at_flag > step_gen` (peer AHEAD
  — the silent wrong-generation combine channel). NEW: sets status bit 4
  (`kArStatusStalePass`) and writes the diag cell
  `(flag=kArDiagAheadMark(-1), gen=gen_at_flag)`. The monotone `>=` comparison
  itself is unchanged; the silence is not.
- Outcome TIMEOUT: unchanged bounded poll → status bit 1, PLUS the diag cell
  records the last-observed peer `(flag, gen)` — the §2d matrix.
- The world kernel passes its observer row `my_diag_row` per peer
  (`&my_diag_row[p]`); the row is a new mapped `ArDiagMatrix` per rank
  (~64 B, world>2 allocation only).
- Host (`allreduce_bf16`, world>2 arm): on bit 4 → one
  `[AR-STALEPASS] rank= gen= slot= peer= gen_at_flag= delta=` line per ahead
  peer via `ar_diag_max_ahead_delta()` (single home, in `ar_liveness.h`);
  `delta > kDesyncWindow (64)` → `dump_wedge_matrix_and_exit(73, "[AR-DESYNC]")`.
  A stale-pass call NEVER completes silently: pending != 0 forces the existing
  retry law (same gen/slot, kArLagRetries=16) → `[AR-FAILOUT]`/`[AR-DEADLINE]`
  death. Every world>2 retry prints `[AR-RETRY] kind=ar` unconditionally (the
  NINFER_MC31 gate is dropped for the N-rank arm only; world==2 keeps it).

### B2 — per-collective host deadline (design §2 B2)
`allreduce_bf16`, world>2 arm:
- Progress triple `ArProgress {inflight_since_ns, inflight_gen, inflight_slot}`
  (mapped, per rank) stamped BEFORE the first launch (gen/slot first, stamp
  last), cleared to 0 after success. This is also the B3 clock.
- `deadline_ns = ar_now_ns() + kArCallDeadlineNs (2 s)` before the loop; after
  every `cudaStreamSynchronize + status consume`, expiry →
  `dump_wedge_matrix_and_exit(71, "[AR-DEADLINE]")`. Order per the design
  sketch: census → deadline → clean-break.
- Anchor: 2 s sits BELOW the R2 law's own ~7-10 s declared death and >1000x
  steady state — it fires only where the loop itself is stuck. Liveness bound
  (R2 shape): ends a wedged crossing, refuses no launch, not a VRAM constant.

### B3 — system watchdog thread (design §2 B3) — `ar_watchdog.h` (new, host-only)
- `ArWatchdog::arm(view)` starts ONE thread, 250 ms period, scanning every
  rank's progress clock (kind 1: `inflight_since_ns` stale > 2 s — covers the
  kernel-never-launched case: host spinning inside cudaStreamSynchronize behind
  a queued kernel or dead RCCL collective cannot clear its own clock) and every
  rank's crossing heartbeat (kind 2: `ArHeartbeat` odd-seq stale > 2 s — covers
  any collective crossing on the dispatcher, including the NCCL arm where the
  T2 wedge actually lived).
- On declaration: §2d dump + `[AR-WEDGE-WATCHDOG]` + `_exit(72)`.
- Armed ONLY in `tp_group.cpp`'s env branch (`tp_oneshot_ar_gate() && n <=
  kMaxWorld`, world>2); disarmed first thing in `~TpGroup` (plus RAII dtor,
  declared after `one_shot` so it dies before the views). Heartbeat
  entry/exit bumps live in `allreduce_local_bf16`, gated
  `tp_oneshot_ar_gate() && n > 2`.

### §2d diag matrix (the triage artifact)
`ar_watchdog_dump()` prints, one glance:
```
[AR-WEDGE-WATCHDOG] world=4 wedged_rank=2 t=+2310ms since inflight
  rank0 gen=141 slot=13 inflight_ms=2310 hb_seq=44 hb_crossing_ms=2308 peers_last_seen={1:(f141,g141) 2:(f0,g0) 3:(f0,g0)}
  ...
```
- `f-1` marks an ahead-pass cell (`kArDiagAheadMark`); real flag stamps are >=1,
  0 = never-published — unambiguous.
- Barrier-phase census (design §2 B3 last clause / §5 tp2 hook) NOT landed —
  `tp2_backend.cpp:1751` is the tp2 owner's site and this seat did not touch it.
  The dump omits the phase field entirely (never prints a placeholder), per the
  design's own landing note. Honest residual; the progress clock + heartbeats
  name the wedged rank without it.

### New files
- `src/core/multi_gpu/ar_liveness.h` — host-only leaf (no HIP includes):
  constants (`kArCallDeadlineNs=2000000000`, `kDesyncWindow=64`,
  `kArStatusStalePass=4`, `kArDiagAheadMark=-1`, `kArWatchdogPeriodMs=250`),
  PODs (`ArProgress`, `ArDiagCell/Matrix`, `ArHeartbeat`, `ArWatchdogView`),
  `ar_now_ns`, `ar_watchdog_scan`, `ar_watchdog_dump`,
  `ar_diag_max_ahead_delta`, heartbeat entry/exit. Single home of the decision
  logic so the CI cell tests the real shipped code.
- `src/core/multi_gpu/ar_watchdog.h` — host-only B3 thread (injectable
  clock/death for the CI cell; default death = dump + `_exit(72)`).
- `one_shot_allreduce.h` — accessors `world()/progress(r)/diag(r)`,
  `dump_wedge_matrix_and_exit(71/73)`; `static_assert(kArLivenessMaxWorld ==
  kMaxWorld)`.

## 2. What landed (W4AR drill injectors — env-gated, dormant by default)

All in `one_shot_allreduce.cu` (`ArTestInject`, read once at the first world>2
call; L7 of the guard pins them to this file only):
- `NINFER_AR_TEST_STALL_RANK=<r>` — rank r's world kernel returns before
  publishing anything (no flag/gen/status). W4AR-STALL-DRILL.
- `NINFER_AR_TEST_AHEAD=<k>` (+ optional `NINFER_AR_TEST_AHEAD_RANK`, default 1;
  the rank selector for all drills is STALL_RANK if set) — on that rank's FIRST
  one-shot call: phantom-publish an advanced stamp (`rank_gen += k`) at the
  CURRENT lockstep slot (GEN→release-fence→FLAG, KAR-v2 order), skip the call.
  Peers' `>=` gates pass on the wrong-generation stamp → B1 census (k <= 64) or
  `[AR-DESYNC]` (k > 64). W4AR-AHEAD-PEER.
- `NINFER_AR_TEST_STUCK_NCCL=1` — once, on the selected rank: enqueue a bounded
  fake-collective spin kernel (`ar_test_stream_block_kernel`, 1<<33 cycles ~
  8-17 s, far past the 2 s deadline) ahead of the AR kernel, so the kernel is
  QUEUED and never launches; only B3 can see it. W4AR-STREAMFRONT-DRILL.

## 3. Default-identity statement (tasking §2, the env matrix)

- `NINFER_TP_ONESHOT_AR` unset (default):
  - world==2: bit-frozen path — the 2-rank kernel, its launch site, the entry
    deferred-consume, the retry loop and the mc31h-gated `[AR-RETRY]` are
    UNTOUCHED (guard L4 pins zero cure/injector tokens in both the kernel body
    and the launch site; the only world==2-visible diffs are the comment line
    in the dtor and the hoisted gate lambda — zero behavior lines). The
    liveness allocations are `world > 2`-gated, so even the ctor allocation
    sequence is the historical one; dtor frees are null-guarded over the same
    loop. The watchdog thread is NEVER started (arm is inside the env branch);
    the heartbeat gate reads `tp_oneshot_ar_gate() && n > 2` — first touch of
    the hoisted static is the same getenv-once the ctor always did.
  - world>2: one_shot never constructed (pre-existing gate) → NCCL ring, as
    before; the watchdog/heartbeat code is never reached.
- Watchdog when OFF: **provably zero-cost, not merely idle** — the thread does
  not exist (chosen over "started-but-polling" because the arm site is already
  the env gate, so default boots pay neither a thread nor a wakeup; and a
  running watchdog on a default boot would be a new failure surface watching
  nothing).
- `NINFER_TP_ONESHOT_AR=1`: world==2 unchanged (branch not taken at n==2);
  world>2 gets B1/B2/B3 + census + injectors-at-zero.

Env matrix (design §4b) — full matrix at boot-battery cadence; per-sha gate =
the five RED-direction cells in §4.

## 4. Cells landed (host CI class) + receipts

`tools/guards/check_oneshot_w4_liveness_guard.py` (house pattern:
`check_bf16_low_guard.py`; comments stripped before matching; `--root`;
`--selftest` mutation coverage; a missing compiler is a FINDING, never a skip):

| guard | locks | direction captured |
|---|---|---|
| L1 | ar_liveness.h constants at measured anchors + scanner/dump/census single home | RED: constants/file missing |
| L2 | ArWatchdog + default death = dump + `_exit(72)` + `[AR-WEDGE-WATCHDOG]` | RED: file missing |
| L3 | ar_wait_peer HIP arm: ahead detection (`gen_at_flag > step_gen`), bit 4, diag writes on ALL outcomes; world kernel diag + stall injector | RED: 6 findings @ pre-fix tree |
| L4 | frozen world==2 path: kernel body + launch site carry ZERO cure/injector tokens (W4AR-W2-BITFREEZE host leg) | GREEN-only (it pins the absence) |
| L5 | B2 arm: progress stamp/clear, deadline, `[AR-STALEPASS]`, `[AR-DEADLINE]`/71, `[AR-DESYNC]`/73, unconditional N-rank `[AR-RETRY]` | RED: 9 findings @ pre-fix tree |
| L6 | default-OFF gate shape, watchdog arm INSIDE the env branch only, dtor disarm, world>2-gated heartbeats | RED: 5 findings @ pre-fix tree |
| L7 | injectors read ONLY in the .cu; defaults dormant | RED: 6 findings (envs absent) |
| BH | behavioral harness — the REAL headers compiled host-side (no HIP/GPU), 20 scenarios: stalled-peer declare, stuck-crossing declare (stream-front shape), deadline strict boundary, census delta 64/65 window, red-model (pre-fix `>=` silently accepts ahead), §2d dump content, watchdog THREAD end-to-end on a fake clock, disarm, real `_exit(72)` death in a forked child | RED: compile failed (no cure) / GREEN: PASS |

Receipts (banked, this commit):
- `results/amd/coherence/ONESHOT_W4_guard_prefix_RED.out` — guard @ pre-fix tree
  (`git archive 248630d1b`) → **RED, exit 1** (L1/L2 missing, L3/L5/L6/L7
  findings, BH compile-failed): the CLASS fired on the pre-fix artifact.
- `results/amd/coherence/ONESHOT_W4_guard_green.out` — guard @ this tree →
  **GREEN, exit 0**, L1-L7 clean + BH PASS (20 scenarios incl. `_exit(72)`).
- `results/amd/coherence/ONESHOT_W4_guard_selftest.log` — **SELFTEST PASS**
  (27 mutations all detected, 7 pristine fixtures clean).

Device cells (design §4a) still owed on the GPU window: W4AR-WEDGE-REPRO,
W4AR-STALL-DRILL, W4AR-AHEAD-PEER, W4AR-STREAMFRONT-DRILL, W4AR-W2-BITFREEZE,
W4AR-W4-IDENTITY — runbook below. The host legs above close the
deterministic-logic portion of the CLASS in both directions; the device legs
close the transport reality (the T2 wedge itself).

## 5. Compile-verify (house law line)

Law command = the tree's `compile_commands.json` line
(`build-hip-amd`, per docs/amd law + M2 precedent), run with this lane's paths:

- `one_shot_allreduce.cu`:
  `/opt/rocm-6.2.0/lib/llvm/bin/clang++ -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1
  -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -I{src/common/hip_shim,
  include, src, third_party,...} -O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx900
  -x hip -c src/core/multi_gpu/one_shot_allreduce.cu`
  → **RC=0, warnings 24 = baseline 24** (pre-edit byte-run of the same command;
  all pre-existing `-Wunused-result`, counts identical incl. class).
- `tp_group.cpp`: `/usr/bin/c++ ... -isystem /opt/rocm/{include,6.2.0/include}
  -O3 -DNDEBUG -std=gnu++20 -Wswitch -c src/core/multi_gpu/tp_group.cpp`
  → **RC=0, warnings 16 = baseline 16**.
- Zero new warnings was enforced during the edit: the liveness code
  `(void)`-casts its new unchecked HIP calls (M2 precedent) and uses
  load+store (not `++`) on the volatile heartbeat seq (-Wdeprecated-volatile).

Builds stayed in `amd-wo-oneshot-ar/build-scratch` (object files only; the
shared checkout was never built in; df stayed >=12 GB free throughout, abort
threshold 3 GB).

## 6. GPU-owner ONE-WINDOW RUNBOOK (closes the design §7 ledger)

Sha law: RED = banked `409450b83ffa4be8` (2/2, `b420b86fe`). GREEN = the
post-fix binary YOU boot this window — bank it per
`docs/amd/BOOT_LAUNCH_RUNBOOK.md §4` (BANK-BEFORE-RELINK; boot from
`/home/chris/artifacts_bin/<name>_<sha16>.bin`, never a lane build path).

0. Grant: this window IS the GPU grant; `df -h /` before building (~12 GB free
   at impl time — if < 3 GB, ABORT and report).
1. Rebuild in the lane that owns the window (clone `build-hip-amd` config if
   the lane lacks one): build ONLY the affected targets —
   `cmake --build build-hip-amd --target ninfer-serve -j$(nproc)` (or the
   lane's equivalent app target). The cure touches
   `one_shot_allreduce.{h,cu}`, `ar_liveness.h` (new), `ar_watchdog.h` (new),
   `tp_group.cpp` — expect relinks of the serving binary only.
2. `sha256sum` the binary; `cp -p` to `/home/chris/artifacts_bin/`; strings
   sanity: `strings <bin> | grep -c 'AR-WEDGE-WATCHDOG\|AR-DEADLINE\|AR-STALEPASS\|AR-DESYNC\|AR-TEST-'`
   ≥ 5 (the bounds + injectors are IN the default binary, DORMANT).
3. Step 0 guard (zero-GPU): `python3 tools/guards/check_oneshot_w4_liveness_guard.py`
   → GREEN exit 0 (host class, already banked; re-run in-window for the row).
4. **W4AR-W2-BITFREEZE first** (the cure must not touch the frozen path):
   world=2 count600 at env-absent AND env=1; `[ids]` byte-compare vs the
   pre-fix banked stream. Byte-identical or RED.
5. **W4AR-WEDGE-REPRO** (the previously-wedged boot, now the headline):
   `NINFER_TP_ONESHOT_AR=1`, TP4 (`--devices 0,1,2,3`), boot + warmup, 300 s
   cap. Expected — **no third outcome**:
   - GREEN: warmup completes; `[ids] == 760 1156 1018 328` byte-match + cross-rank
     post-AR witness, OR
   - LOUD death ≤ ~15 s: one of `[AR-DEADLINE]`(71) / `[AR-WEDGE-WATCHDOG]`(72) /
     `[AR-DESYNC]`(73) / `[AR-FAILOUT]`(70) with the §2d matrix block banked to
     console.
   - A `warming up...` line older than 2 min with NONE of those tags = watchdog
     missing = protocol violation — file against the design doc's class, do not
     debug the boot.
6. **Injector cells** (one boot each; all env-gated, default-absent):
   - `W4AR-STALL-DRILL`: `NINFER_TP_ONESHOT_AR=1 NINFER_AR_TEST_STALL_RANK=3`
     (repeat rank=1): expect ALL ranks loud-dead ≤ ~12 s, `[AR-DEADLINE]`(71) or
     `[AR-WEDGE-WATCHDOG]`(72), matrix names the stalled rank `f0,g0` (never-arrived).
   - `W4AR-AHEAD-PEER`: `... NINFER_AR_TEST_AHEAD=2` → `[AR-STALEPASS]` census
     lines (delta 2) then loud death (71/70 — census never completes silently);
     `... NINFER_AR_TEST_AHEAD=200` → immediate `[AR-DESYNC]`(73), delta 200 > 64.
   - `W4AR-STREAMFRONT-DRILL`: `... NINFER_AR_TEST_STUCK_NCCL=1` →
     `[AR-WEDGE-WATCHDOG]`(72) ≤ ~3 s while the AR kernel NEVER ran (proves B3
     covers the unreachable-bounded-wait case).
   Cap every drill at 60 s with anchored kill of ONLY your PIDs
   (`gpu_guard.sh`, absolute repo path; never system-wide pkill); KFD=0 after.
7. **W4AR-W4-IDENTITY**: env=1, TP4, count600 vs the ring arm — `[ids]` +
   completion byte-compare vs `ONESHOT_A_count600.json`; within-arm 4-rank
   post-AR bitwise equality. (If step 5 went GREEN-on-warmup, this is the
   serving-leg adoption datum; the one-shot arm still owes its measured
   ms/round before any adoption talk — T2 verdict 1 stands.)
8. Bank per-LABEL rows (`BOOT_BATTERY_<ts>.row` + clocks sideband; PLOG via
   `tools/guards/plog_append.py`), fill the design §7 ledger GREEN column with
   the boot-stamped sha.

## 7. Honest residuals
- Barrier-phase census (tp2 `sync_bar` hook) not landed — owner's site; the
  dump omits phase rather than placeholder (design note 2 honored).
- The CUDA (#else) arm of `ar_wait_peer` keeps its unbounded-poll semantics
  (R7: NVIDIA seat's cell, same CLASS) — new params accepted-but-unused there.
- The §1c trigger remains top-hypothesis until a device census captures a live
  `[AR-STALEPASS]`; every variant lands in the B1/B2/B3 nets regardless.
- The ahead-pass retry law (a bit-4 pending forces the existing retry loop to
  exhaustion) means small ahead-deltas end in `[AR-FAILOUT]`(70) at ~1.3 s —
  loud by design; if a window shows benign small-delta stalepasses that SHOULD
  ride, that is a policy conversation for the design doc, not a silent change.
