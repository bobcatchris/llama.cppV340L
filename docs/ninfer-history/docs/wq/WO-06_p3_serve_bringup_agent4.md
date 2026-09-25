# WO-06 — P3 SERVE BRING-UP: hip link → q3 serve → tp1 control (agent 4) — issued 2026-09-12 ~19:5xZ, coordinator C441

**Worktree:** `/home/chris/worktrees/amd-wo-p3-serve`, branch `amd/wo-p3-serve`, already created at `amd/main` tip (f4cf9a82 — cite by name, it moves; `git pull` your branch from `amd/main` before each commit).
**Mission (user, 24 h hard deadline):** a non-MTP chat completion served by `ninfer-serve` from a HIP binary on this box. This is the LAST unowned step. Everything upstream (shim, goldens, donor adoption, T3 prefill validated) is on your base — see STATE 17:3xZ–19:4xZ on docs/amd/COORDINATOR.md.

## 1. Goal & gates (goal → gate → evidence)
G1. `cmake --build --target ninfer-serve` rc=0 on the HIP build (apps/serve CMake wiring + every HIP-compat lane). Gate: build log pasted, static scans inadmissible.
G2. q3 artifact VERIFIED: sha256 vs 7f26a0eb… (pin: docs/amd/COORDINATOR.md 12:3xZ-era + Green's smoke receipt). Coordinator-owned, feeds you; file = `/media/chris/desktop_f/qwen3_8_27b_q3.ninfer`, exact 15,446,796,288 B.
G3. Stamped serve window: `ninfer-serve <q3> --devices <pair> --max-context <small>` answers a tiny prompt (non-MTP), fp16/fp32 numerics per D3, eager-only (graphs are DEAD on 6.2.0 — measured, results/amd/p1). Gate: serve log + response pasted, release row verbatim.
G4. Reference artifact tp1 CONTROL: single-device serve of `/home/chris/dual_5060_ti_ninfer/artifacts/qwen3_8_27b.ninfer` (20,437,336,576 B; salvage in flight — see §5) producing the per-position token table gemini's parity spec consumes (docs/amd/TP2_COMPARATIVE_PARITY_SPEC.md is the contract, THEIR file).

## 2. Test entry points
- CPU gate every step: `bash tools/ops/run_ci_amd.sh --zero-gpu` (check-(a) fires on the registered set — those 3–4 files are EXPECTED-RED-with-ticket, everything else hard fail).
- G1 first-real-error list = your blocker report; paste text, name every file your lanes touch (feeds gemini's PG-1 exclusion; name-not-edit their files).
- G3/G4: gemini's parity envelope vs tp1 (never donor goldens — 18.21 vs 20.44/15.45 GB artifacts mismatch, locked law); freshness protocol = rebuild from committed HEAD in the same message, binary + archive sha in-log.
- VRAM LAW: no estimated charge may refuse a launch; `hipMemGetInfo` + actual bytes only; near-capacity config LAUNCHES and MEASURES (TP2 of the 19.03 GiB reference = fixture-capped at 15.97 GiB — that's a measured statement, not a budget guess).

## 3. Design decisions (FINAL — do not re-litigate)
- eager-only bring-up: cross-device graphs dead (launch→OOM, spliced-memcpy data-loss); flag-sync REJECTED (donor S9c card-wedge, MODE1 reset failed); AR-count is the perf budget (14–45 µs/call, 6.61–6.70 GB/s host-staged — cite, don't re-derive).
- Q3 route: the GATHER embedding arm (promote-carried; embed_gather.cuh kEmbedGatherQ3*), NOT the dense path (dense carries the open corruption defect — results/amd/p1 g14/g15 + v340l/03; sweep is gemini's).
- topology: 4 flat devices, dev→card NOT identity, no pair privileged (flat measured, agent2 v340l/05; all off-diag weights 40, 4 buses); serve pair chosen at stamp time, --showtopo recorded next to any bandwidth claim.
- REJECTED alternatives (with revisit conditions): tp_kernel.cu HIP whitelist until agent4-shim verification + gemini check-(h) pass on the REAL whitelisted set (staged one_shot entries ride T3's fix); M4/4-device = HORIZON (engine 2-rank limit, R5 ruled: Q3-v1a/2-rank/short-context first); grouped-MMA/FMA fast paths = perf-only, §F freeze applies (donor `#else` arms carry correctness).

## 4. File ownership (exclusive; do not cross)
YOURS: `apps/**`, `serve/**` CMake wiring under HIP, `src/ops/launcher/kvarn*.cu`, `src/ops/embed/**` (hip-compat only), `src/targets/**` (hip-compat only), `src/HipSources.cmake` (append-only for these classes), plus your worktree docs.
NEVER (other lanes): `src/common/hip_shim/*` (agent2 — STOP-ask for shim gaps, they are durable-mapping owners; do NOT write TU-local macro twins — the shadowing trap is documented in AGENT2_WO02_HANDOFF + your base's merge history), `src/ops/kernel/gqa_attention_*_gfx906*` + `results/amd/t3_oracle*` (agent3), `tools/ops/gate*` + `tests/**` (gemini, §7.x), `src/runtime/tp2/tp_engine.cpp` + `tp2_budget.h` (canonical, zero-diff law).
GPU: none without written coordinator grant (claim-before-act; cards are a serial resource; own-pid kills only, never `pkill`).

## 5. Known-state inputs (cite, do not re-derive)
- agent2 is salvaging the reference artifact (over-length +367 MB double-segment → their truncate+resume; v340l/06 §1-8 is the receipt discipline: check status-file truth, orphan-writers, magic bytes, read-back probes).
- agent3 owns the T3 decode-anomaly block (their read: width-semantics/walk root cause; my fresh-eyes note: prefill-validated-evidence contradicts the layout suspect — the 0fd3e6a8/54eaab52 evidence is on your base's branch history).
- gemini: gate items routed 19:3xZ (check-(a) .h-pair, check-(h) spin cell, D3 fingerprint, q3_ci MODE).
- Build traps: cmake off-PATH (/home/chris/opt/cmake/bin/cmake 3.30.5), -DNINFER_BACKEND=hip (NOT NINFER_BUILD_BACKEND — silent-default trap), -DNINFER_BUILD_APPS=OFF for lib library builds, -DBUILD_TESTING=OFF (set() not option(), CMakeLists:78), -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0, hipcc: -L/-l never positional .a, links: libninfer_hip_host.a + libninfer_observ.a (both in build-hip-amd/src).
- q3 artifact status: my sha256 pass was RUNNING on it (7.6/14.7 GB, ~40 MB/s) — result routes to you; do not serve an unverified artifact, size-perfect ≠ content-good (13 desktop retries taught this today).

## 6. Execution order (commit + test each step)
S1 base verify (pull `amd/wo-p3-serve` from `amd/main`, run zero-GPU gate, record EXPECTED-RED count). S2 serve-target build attempt → paste first real errors = G1 defect list, commit findings + any compat lanes (each file named). S3 re-run until `--target ninfer-serve` rc=0, commit. S4 await sha verdict → request G3 stamp (written grant, dev pair, timeout 30, ≤90 min window, freshness by sha); bring-up: single-request, `--max-context 2048`, greedy, non-MTP, log+release row. S5 G4 tp1 control table (reference artifact, single device, same stamped cadence — second mini-stamp). Report per step; blockers: name the file+line, STOP-ask rather than work around.

## 7. Definition of done
G1–G4 met with logs in `results/amd/`, every touched file in commit bodies, a handoff section on your branch naming what gemini's parity gate consumes vs what needs the next stamp, and the serve window released clean. CI zero-GPU green on your final tree (registered exceptions expected-RED with ticket).
