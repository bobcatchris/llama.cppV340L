# v340l port — progress notes (append-only)

Log format: `## YYYY-MM-DD HH:MM UTC — <agent>` then facts. Never edit or
delete earlier entries; append corrections as new entries. Measurement data
goes in `results/v340l/` with config + clocks; this file holds narrative
and gate status.

**Gate status legend:** ○ not started · ◐ in progress · ● green · ✗ red

| Phase | Gate | Status |
|---|---|---|
| 0 | PG-0 env verify + measured per-die roofline | ● measured 2026-09-12T01:27–01:33Z under G-AMD-1 (results/v340l/step0_roofline.md) |
| 1 | PG-1 HIP build lane (zero-GPU) | ● whitelist + dynamic derivation + anti-resurrection + parity certified (7bf82b40) |
| 2 | PG-A single-device primitives | ● runtime primitives + graph capture certified under G-AMD-3 (results/v340l/pga_device_smoke.log) |
| 3 | PG-B minimal op set parity | ○ |
| 4 | PG-C layer/shard forward | ○ |
| 5 | PG-D TP2 fixtures on 2 devices (M2) | ○ |
| 6 | PG-E TP4 full artifact on 4 devices (M4) | ○ |
| 7 | PG-F run_ci_amd.sh green | ○ |

---

## 2026-09-12 — scoping session (dual_5060_ti_ninfer#2bf0)

- Scoped the port from this repo (not upstream ninfer). Key scoping facts
  verified this session:
  - HF artifact `neroued/Qwen3.8-27B-NInfer`: single file
    `qwen3_8_27b.ninfer`, **20,437,336,576 bytes**, sha256
    `0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3`,
    container v2, `model_id=qwen3.8-27b`, **`weights_id=groupwise-int`**,
    min runtime revision `385b30ce…`. Source verified from the HF
    `artifact-manifest.json` (fetched 2026-09-12).
  - 20.4 GB artifact does NOT fit one V340L card (≈16 GB) → M2 (2 devices)
    is fixture-based correctness only; first full-model milestone is M4
    (4 devices / 2 cards).
  - This repo already has the TP2 multi-GPU layer
    (`src/runtime/tp2/`, `src/core/multi_gpu/`) with a fixture test ladder
    in `tests/multi_gpu/` — the port is a HIP conversion of an existing
    working design, not a from-scratch multi-GPU build.
  - `one_shot_allreduce.cu` uses plain malloc/P2P patterns (no
    NVLink/multimem) — portable.
  - Artifact quant profile is groupwise-int → NVFP4/FP8 kernel trees are
    NOT on the port path (also impossible on gfx900: no tensor cores, no
    FP4/FP8 hardware).
  - gfx900 has no bf16 ALU → fp16 activations / fp32 accumulate (decision
    D3 in the work order); numerics re-baselined on AMD, never diffed
    byte-exact across backends.
- Wrote `00_scope_and_work_order.md` (steps 0–8, gates PG-0…PG-F) and this
  log. Doc-numbering note: folder-local numbering (docs/v340l/00_…) used to
  avoid the global `docs/NN_` series; coordinator claim still required.
- Disk at scoping time: 58 GB free on `/` (artifact needs 20.4 GB + build
  space — check `df -h` again before downloading).
- Open questions carried in work order §8 — **ALL RESOLVED 2026-09-12 by
  coordinator ruling C441** (devices-not-cards; gfx900 measured; ROCm
  6.2.0-66 pinned; same box, no nvcc → CUDA guarantee here is
  source-diff/whitelist only, real CUDA proof rides the merge package).
  Work order §2/§5/§6/§7/§8 corrected accordingly.
- Test-lane boundary: PG gate implementation + `run_ci_amd.sh` wiring are
  gemini's exclusive lane (docs/v340l/01 WO, accepted); this lane owns the
  PG specs and cold-reads/re-runs gemini's gates.

## 2026-09-12 (later) — branch landed; board facts verified first-hand

- Branch `wo/v340l-hip` at `~/worktrees/wo-v340l-hip`, based on 34b209d6;
  merged main (ea9595bd = gemini's test-lane WO docs/v340l/01) so the lane
  folder is whole on the branch. Commit SHAs reported to coordinator.
- Board facts re-verified by THIS session directly (not just cited):
  `rocm-smi --showmeminfo vram` → 4 GPUs × 8,573,157,376 B (7.98 GiB)
  total, 18,575,360 B used (idle); `rocm-smi --showproductname | grep -c
  "Vega 10 \[Radeon Pro V340"` → 4; `rocminfo` → 4× gfx900 GPU nodes;
  `hipconfig` → HIP 6.2.41133, ROCm 6.2.0, amdclang 18. Groups: video +
  render present (device access OK). No nvcc on PATH (GTX 1050 Ti is
  display-only).
- Coordinator pre-registered M4 binding constraint (C441): weights ≈
  4.76 GiB/device ⇒ ~3.2 GiB/device for KV+activations — the slack kills
  the first launch, not the weights. VRAM LAW: hipMemGetInfo + actual
  bytes; near-capacity cell LAUNCHES and MEASURES.
- No GPU grant issued yet. Next: request Step 0 roofline grant (exact
  command) from coordinator; meanwhile proceed with Step 1 (zero-GPU HIP
  build lane) which needs no grant.

## 2026-09-12 ~01:21–01:35Z — GRANT G-AMD-1 EXECUTED (PG-0 complete)

- Grant received in writing (C441): build + run the per-device D2D memcpy
  probe, 40-min window, sequential, no server/artifact. All conditions met.
- Probe: `tools/v340l/bw_probe.hip` (~90 lines, 2×512 MiB buffers = 1 GiB
  footprint, 10 warmup + 200 timed iters, hipcc 6.2.41133
  `--offload-arch=gfx900`). CMake wiring: root CMakeLists gained the
  `NINFER_BACKEND` switch (cuda default byte-identical; hip branch gated to
  gfx900) + HIP-only `tools/v340l` subdirectory with `NINFER_HIP_PROBE_ONLY`
  scoping — zero changes to the CUDA branch. Compiled directly with hipcc
  this run because **this box has no cmake binary at all** (the run_ci.sh
  nvcc paths reference the old box; noted as a Step-1 environment item).
- MEASURED (payload convention, memcpy-equivalent): dev0 191.55, dev1 185.06,
  dev2 187.12, dev3 183.80 GB/s; 3× repeats per device stable ±2.5%, no
  clock-ramp trend. Full table + PIDs + clocks in
  `results/v340l/step0_roofline.md`; raw log alongside.
- INTERPRETATION (not a measurement): the docs/01+04 483.8 GB/s per-die
  figure does not match this box's measured memcpy anchor; every later
  "% of roofline" claim must cite the step0 table, not the docs figure.
  Note the payload-vs-traffic convention explicitly when comparing (§ of
  step0_roofline.md).
- Release row verified first-hand: `rocm-smi --showpids` empty, all 4
  devices back to 18,575,360 B baseline. Grant closed within window.

## 2026-09-12 ~01:36–01:42Z — GRANT G-AMD-2 EXECUTED (PG-0b: in-load operating clocks)

- Sustained ~32 s copy per device (NINFER_BW_ITERS=11000, same 1 GiB footprint) with
  concurrent read-only sysfs sampler (200 ms, pp_dpm_*/gpu_busy_percent, cards 0-3).
- MEASURED sustained: dev0 183.34 / dev1 183.26 / dev2 182.58 / dev3 181.09 GB/s —
  ~1-4% below short-burst (droop, as coordinator predicted).
- Operating point: **mclk at TOP DPM level (3 = 945 MHz) for 100% of load samples**
  on every sampled device; sclk floats 1269-1500 MHz. The anchor above is therefore
  AT the max memory-clock operating point available without root. PG-0 relabeled
  'anchor at default unpinned clocks' per ruling; pinned-clock peak = PG-0c, needs user.
- Device→card mapping is NOT identity: dev0→card1, dev1→card3, dev2→card0, dev3→card4
  (card2 = NVIDIA display). Recorded in docs/174. dev3's in-load clocks not sampled
  (sampler covered cards 0-3) — marked unmeasured, not inferred.
- Release row verified first-hand: showpids empty, all 4 devices at 18,575,360 B.
- Toolchain: cmake 3.30.5 tarball pinned at ~/opt/cmake (sha256 recorded in docs/174),
  NOT pip (D5 wrapper-shadow lesson); absolute-path rule published for gemini.
- Merged main d6173bb7 (gemini WO corrections).

## 2026-09-12 ~01:44–02:05Z — GRANT G-AMD-2b closed; STEP 1 whitelist + shim LANDED

- G-AMD-2b (74a89677): dev3/card4 sampled directly — sustained 181.17 GB/s at mclk
  top level (159/159). All 4 devices now measured at the same in-load operating
  point. First attempt failed rc=127 (build-hip wiped for reconfigure) — recorded.
- Step 1 product-side deliverable landed (ac75b742 + 144cb6b8):
  - `src/HipSources.cmake` — whitelist (13 host-side sources; whitelist-beats-
    blacklist so new CUDA-only files fail the HIP build loudly).
  - `src/common/hip_shim/{cuda_runtime.h,cuda_bf16.h,nccl.h}` — shim dir FIRST on
    the include path; cuda_bf16 shim covers storage/type usage only (gfx900 has
    no bf16 ALU — kernel numerics go fp16/fp32 per D3); nccl→rccl/rccl.h
    (rccl-dev installs under include/rccl/, not the root — first shim attempt
    missed that).
  - `src/CMakeLists.txt` dispatches to HipSources.cmake under NINFER_BACKEND=hip;
    CUDA content textually preserved (shared-file rule). Wording per C441: the
    cuda CONFIGURATION is unchanged; the file is edited only under hip
    conditionals.
  - Verified: `libninfer_hip_host.a` compiles and links with cmake 3.30.5 +
    hipcc 6.2.41133 `--offload-arch=gfx900`. Shim growth so far: +cudaGetDevice,
    +cudaMemcpyDefault (compile-error-driven, exactly the loud-failure contract).
- Denominator discipline adopted (C441 ruling): any "% of roofline" claim must
  name its denominator — theoretical 483.8 GB/s vs measured sustained D2D copy
  ceiling ~183 GB/s (≈37.8% of theoretical). PG-0b sustained is the reference;
  PG-0 is the burst record.
- OPEN: `/reload` needed on this session (user action) before my comm_send to
  gemini can work — still running the pre-fix extension copy per C441. Coordinator
  is relaying interface notes meanwhile.
- NEXT (Step 2 + remaining Step 1): extend whitelist toward runtime primitives
  (PG-A scope); gemini wires PG-1 static cells against the landed targets.

## 2026-09-12 ~02:05–02:30Z — STEP 1 EXTENDED + SILENT-DROP DEFECT FOUND & FIXED (own work, self-caught)

- Whitelist extended with graph/KV/state runtime + the 3 multi_gpu device .cu
  files. **Build returned EXIT 0 — but an object-count audit showed the
  archive held 16/21 objects: CMake silently DROPS `.cu` sources in a HIP-only
  project (no LANGUAGE assignment).** The two earlier "green" Step-1 reports
  (13/13 claimed) were affected the same way — RETRACTED: device.cu/arena.cu
  had NOT been compiled in them. Corrected now.
- Fix: `set_source_files_properties(... LANGUAGE HIP)` for every .cu +
  `src/CheckHipArchive.cmake` POST_BUILD parity guard (archive object count
  MUST equal whitelist source count; a drop = build failure). Guard message
  verified firing; parity now 18/18 green.
- With .cu ACTUALLY compiling, the multi_gpu AR trio failed loudly on real
  port work (as designed): PTX inline asm `ld.global.cv/st.global.wt` with
  "l" constraints is invalid on GCN (needs volatile-access rewrite — Step-5
  port item), `__float2bfloat16_rn` → HIP has scalar `__float2bfloat16`
  (storage conversion available), `cudaHostAlloc{Mapped,Portable}` →
  hipHostMalloc{Mapped,Portable} flags exist. The three files are REMOVED
  from the Step-1 whitelist with a NOTE that they join per-file as their
  ports land — keeping them listed-but-dropped would gut the guard's meaning.
- Shim updates: cudaMemGetInfo alias (VRAM LAW API), cudaMallocHost/cudaFreeHost
  via hipHostMalloc/hipHostFree wrappers (hipMallocHost is deprecated → 2
  warnings per TU), cudaMemset2DAsync, cuda_runtime_api.h shim header,
  cudaGraph* aliases + 2 signature-delta wrappers (3-arg cudaGraphInstantiate
  → 5-arg hip; cudaGraphExecUpdate ResultInfo struct wrapper — HIP's is a
  plain enum), and cuda_bf16.h rewritten to the DELIBERATE option-(b) contract
  per C441 (typedefs only; arithmetic stays compiler-enforced-dead).
- bf16 arithmetic being compiler-enforced-dead on gfx900 (C441 probe:
  __hadd on __hip_bfloat162 fails) = D3 is not policy, it's the toolchain.
- WO §6 PG-1 spec extended with the PARITY cell (gemini implements).

## 2026-09-12 ~02:30–03:10Z — DISCREPANCY RECONCILED (C441 03:51 measurement) + guard hardened + tp2 host layer compiles

- Reconciliation of C441's list=27 vs archive=22 observation (NOT a live silent
  drop, but two REAL guard weaknesses):
  1. "18/18" (my earlier message) was TRUE at commit 465ddeab; then the 4
     runtime/engine files landed → "22/22" green (UNCOMMITTED when C441
     measured); then the 5 tp2/contract files → build was RED (nvtx3 fatal).
     C441's 22-object archive was STALE from the 22-file green build. The
     build failure was LOUD (that's the contract), but a stale archive sitting
     beside a longer list is exactly the masquerade class — see guard fixes.
  2. Guard weakness A: `-DSOURCES="list"` through add_custom_command gets
     SEMICOLON-FLATTENED (guard saw "1 source" vs 27 objects) — now a
     file(GENERATE) stamp + SOURCES_FILE.
     Guard weakness B: count-only compare → now NAME-SET compare + sha1 of the
     source list printed in the OK line (a stale "N/N" line cannot match a new
     list's hash). Also: CMP0057 NEW needed for IN_LIST in `-P` script mode;
     regex `[ \t]*` → `[ \t]+` (empty-match rejected by CMake). Falsifiers run:
     mismatched list → PARITY FAIL (verified); clean build → 27/27.
- NVTX decision: `src/common/hip_shim/nvtx3/nvToolsExt.h` = deliberate NO-OP
  (profiling annotation only, zero numerics; header names rocprofiler/hipRange
  as the future real hookup point — never at call sites).
- Shim additions this stretch: cuda_fp16.h (new), cuda_bf16.h TRANSITIVE note
  (CUDA's cuda_bf16.h pulls cuda_fp16.h; hip_bf16.h does NOT — tp2_backend.cpp
  proved it by using __half through that chain; the shim now mirrors CUDA's
  transitive surface), cudaHostAlloc(+Default/Mapped/Portable),
  cudaMemcpy2DAsync, cudaGetLastError, cudaMallocHost/FreeHost wrappers
  (hipHostMalloc/Free; the deprecated hipMallocHost aliases caused warning
  spam), cudaMemGetInfo (VRAM LAW API).
- STATE: whitelist = 27 sources; CLEAN configure+build rc=0;
  `v340l whitelist parity OK: 27/27 objects, name-sets equal,
  source-list sha1=73698627d8`. tp_engine.cpp (VRAM-LAW canonical home) now
  genuinely compiles under HIP. AR device trio + ops kernels remain Step-3/5.

## 2026-09-12 ~03:15–03:35Z — OPS HOST LAYER 100% GREEN (b0a37455) + PORT INVENTORY (results/v340l/step2_symbol_inventory.md)

- All 75 ops/wrapper+plan+dispatch+kvarn .cpp files compile CLEAN under
  HIP/gfx900 — 102/102 objects, zero errors. Guard line at b0a37455.
- Inventory (nm-based): 795 ninfer:: symbols referenced; 435 truly unresolved;
  300 attributed to 128 distinct .cu kernel files = UPPER BOUND of remaining
  port surface; minimal-path estimate ~40–50 files (nvfp4/fp8/q6/mma-family/
  sparse_moe stay D2-excluded). core/multi_gpu trio listed with its measured
  blockers. Family table + method caveats in the results doc.

## Handoff note (C441 pattern, 2026-09-12 close) — negative tests must prove they can SEE their subject
Five tonight-wrong-answers were all measurement-side, not tree-side. The one that
binds THIS lane: an untracked probe file is invisible to `git diff main
--diff-filter=A`, so a gate never asked the question looked proven (rc=0). Three
costumes: untracked-vs-tracked, staged-vs-committed, worktree-vs-index.
Applied to our guards: (1) the parity guard's SOURCES_FILE stamp is generated by
file(GENERATE) at CONFIGURE time — a standalone `cmake -P CheckHipArchive.cmake`
call must pass the stamp from the SAME build dir that produced the archive, never
a hand-list or a stale configure; (2) gemini's zero-diff/resurrection cells must
run against committed tree state (git add -N at minimum) and their falsifiers
should plant a case the gate CAN see. Any cell that cannot demonstrate "my probe
reached the gate" is unproven, not green.
Merge main BY NAME on resume (never a cited SHA — pinned numbers are tonight's moving-tree trap in a new costume); assert tip-correctness by presence: docs/174 carries the 'additions convention' section after the merge.

## Slice: targets/* + serve/ whitelisted (137/137) — host half of the product now compiles end-to-end
- +whitelist: 35 files (22 targets .cpp incl. qwen3_6_27b tp_load, 13 serve .cpp). +shim: __half_as_ushort/__ushort_as_half bit-casts (hip_fp16 has none — grep-verified 0), cudaUUID_t = plain alias (hipUUID exposes .bytes identically — hip_runtime_api.h:84), CUDART_VERSION/runtime/driver version maps, cpp-httplib include for serve.
- Parity line at this state: 'v340l whitelist parity OK: 137/137 objects, name-sets equal, sha1=50ba7f80f5'. Remaining unresolved = kernel .cu symbols + apps main() (apps target not yet wired HIP).

## 2026-09-12 ~04:10–04:25Z — GEMINI PHASE-GATE LANE: PG-1 certified & falsifier design updated

- **Falsifier Design Update (Chronology / Non-Mutating Scratch)**:
  - Falsifier design evolved from initial tracked-file append (`ad857772`) to strictly non-mutating scratch-copy architecture in commit `59fc3ac2` (per C441 option b preference).
  - `check_anti_resurrection.sh` copies target files to `/tmp/anti_resurrect_scratch_XXXXXX` with `trap 'rm -rf ...' EXIT INT TERM` installed prior to mutation. Zero tracked files are modified during test or falsifier runs; crash-safe under SIGKILL/SIGTERM.
- **Whitelist Parity Guard & Dynamic Derivation**:
  - Parity guard (`src/CheckHipArchive.cmake`) integrated into `gate_pg1_whitelist.sh` Check (f) using `SOURCES_FILE` strong name-set comparison.
  - Falsifier Test 5 added demonstrating RED on name swap (`BOGUS.cpp.o`).
  - Whitelist additions refactored from hardcoded array to dynamic derivation by construction (`7bf82b40`): permits `src/common/hip_shim/**`, root HIP infrastructure, and all sources listed in `src/HipSources.cmake`.
- **Lane Verdict Isolation**:
  - `run_ci_amd.sh` routes falsifier runs to `results/v340l/falsifiers/` avoiding canonical `ci_amd_verdict.json` clobbering.
  - Zero-GPU CI certified: `run_ci_amd.sh --zero-gpu` -> `PASS (zero-GPU subset)`.
- **Status**: PG-1 is ● green (static whitelist, anti-resurrection, bf16 compile probe, and archive parity certified).

## 2026-09-12 ~04:47–04:54Z — GRANT G-AMD-3 EXECUTED: PG-A single-device runtime primitives CERTIFIED 5/5 ●

- **Grant Scope**: G-AMD-3 issued by coordinator (C441) for dev0 only (`v340l_device_smoke`), 30-minute window, footprint <256 MiB.
- **Pre-run check**: `rocm-smi --showpids` clean ("No KFD PIDs currently running"), all 4 GPUs at 18,575,360 B baseline.
- **Measured Execution Facts (results/v340l/pga_device_smoke.log)**:
  * Check 1: `DeviceContext(0)` initialized; device name `AMD Radeon Graphics`, GCN arch `gfx900:xnack-`. Disambiguated SM / CC code: 90 (props.major*10+minor = 90; physical CUs = 56 on Vega 10). Total VRAM 8,573,157,376 B (7.98 GiB). Measured free VRAM via live `hipMemGetInfo`: 8,401,190,912 B (7.82 GiB).
  * Check 2: `DeviceBuffer` (64 MiB) allocated; byte-exact H2D and D2H roundtrip verified across 16,777,216 floats.
  * Check 3: Stream synchronization and `CudaEventTimer` verified (elapsed = 0.029 ms).
  * Check 4: `DecodeGraphDefinition` and `DecodeGraphExecutable` capture, instantiate, and launch with `trivial_add_kernel` verified with exact numeric output.
  * Check 5 (Hardened): `DeviceArena` (128 MiB) suballocation and D2D copy bandwidth timed via `hipEventElapsedTime` with full stream synchronization: **sustained 179.81 GB/s across 50 iters** (payload: 3,355,443,200 B in 0.0187 s). Cross-validates PG-0b measured ceiling (~183 GB/s) within 1.7%. Sanity ceiling bound `[10.0, 350.0] GB/s` verified.
- **Demonstrated Falsifiers (results/v340l/falsifiers/)**:
  * Out-of-range device ID (`pga_device_smoke_falsifier.log`): `v340l_device_smoke 99` -> rc=134.
  * Check 5 broken instrument falsifier (`pga_check5_falsifier.log`): `v340l_device_smoke --falsify-check5 0` -> reports 33,554,432.85 GB/s, fails loud with `FAIL: Impossible bandwidth reported! Instrumentation broken.`, exits rc=1.
- **Bonus Recording (Card Mapping Corroboration)**:
  * Sysfs sampling on drm card1 (`/sys/class/drm/card1/device/`) corroborated dev0 mapping; idle baseline verified across all cards.
- **Post-Run Release Row**:
  * `rocm-smi --showpids` confirmed empty ("No KFD PIDs currently running").
  * All 4 GPUs returned to 18,575,360 B baseline. Grant G-AMD-3 closed within window.
- **Status**: PG-A is ● green (5/5 checks passed).

## First-token list delivered (results/v340l/first_token_files.md) — C441 scheduling change
- T1 17 glue files (0 asm) -> T2 7 w8 GEMV decode -> T3 attention (bounded ldmatrix/cp.async -> LDS-rewrite; plain-KV first, int8-KV T3b) -> T4 prefill/GDN (bf16_gating 614 loc straight->fp16; swiglu-prefill route = design Q-B) -> link+load+first token. ~26 .cu + 5 .cuh, ~9k loc. AR trio + tp_kernel EXPLICITLY not first-token.
- Single-device serve PROVEN to exist: --device N + engine.cpp:310 size()==2 branch. T5 not runtime-needed on path; tp2 link refs remain (stub-or-port decision at link time).

## 2026-09-12 ~09:55–10:15Z — GRANT G-AMD-4: T1 BATCH 1 PORTED + DEVICE-VERIFIED (3 files green)

- +whitelist: ops/launcher/{scalar.cu, scatter.cu, cast.cu} — compile green,
  parity "140/140, name-sets equal, sha1=c8f20faa96". +shim: cuda_pipeline.h
  (synchronous staging impl — gfx900 has no cp.async), cuda_bf16.h converters
  (__float2bfloat16_rn alias, __floats2bfloat162_rn float2 + 2-float overloads,
  __host__ __device__ — first attempt host-only failed from __global__).
- HELD: ops/launcher/add_bias.cu (and residual_add.cu, batch 2) — blocked by
  PRODUCT header src/ops/common/memory.cuh (hand-written cp.async PTX, "l"
  constraints, __cvta_generic_to_shared). Include-path overlay possible
  (shim dir is FIRST on -I) but carries an ODR-split risk (relative quoted
  includes inside src/ops/common would get the real header) — REQUESTING
  COORDINATOR RULING: overlay+ODR-audit vs sanctioned minimal product edit.
- DEVICE VERIFICATION (dev0, HIP_VISIBLE_DEVICES=0, ad-hoc harness /tmp/
  batch1_verify.cu linked against the archive; durable gate = gemini's PG-B):
  SCALAR set/inc/add ok; CAST fp32->bf16 4096/4096 BIT-EXACT vs C-side RNE;
  SCATTER bf16 columns 16x64 permutation complete. All 5 checks pass.
- HARNESS LESSONS (both mine, both cost a device run): (1) cudaStreamNonBlocking
  + legacy-stream sync memcpy = RACE — first run's 4 "failures" were my
  harness, kernels were fine; use stream 0 or explicit sync. (2) scatter_launch
  is 2-D column semantics {d, vision}, bf16-typed — my 1-D i32 assumption was
  wrong; read the kernel, not the header name. A red first run is information
  about BOTH sides — name it.
- Release: No KFD PIDs; all 4 devices at 18,575,360 B (×4 verified).

## Pipeline-shim exposure inventory (C441 bounded-acceptance requirement 1)

cuda_pipeline.h sync-staging accepted for T1; where the lost cp.async latency-
hiding will bite, recorded HERE so T3 inherits a known map, not a discovery:

| body | cp.async hits | tier | status |
|---|---|---|---|
| kernel/gqa_decode_slice4_kvarn.cuh | 12 | KVarN (out of first-token) | note for prod-parity phase |
| kernel/gqa_decode_slice3_i8.cuh | 4 | **T3b int8-KV (production parity)** | measure-at-T3 per ruling 3 |
| kernel/gqa_decode_unified_slice2.cuh | 1 | **may be on plain path** | check when T3 wires the dispatcher |
| kernel/gqa_decode_body.cuh | 0 | T3 plain | clean |

T1 files HELD pending the same decision (they include the hand-PTX product
header ops/common/memory.cuh): add_bias.cu (4), gelu.cu, sigmoid_gate_mul.cu,
residual_add.cu. RULING REQUESTED: include-path overlay (with ODR audit of
src/ops/common relative includes) vs sanctioned minimal product edit.
Rule held per C441 requirement 2: if a T3 body genuinely overlaps staging with
compute, the fix is a manual gfx900 double-buffer (issue next load before
consume), NOT a wider shim.

## 2026-09-12 ~10:20–10:40Z — T1 BATCH 2 COMPILE (147/147) + second product-header blocker

- +whitelist: rope, embed_gather, argmax, sampling, position, l2norm,
  layer_norm .cu — parity "147/147, sha1=5c26e0306f".
- +shim: cuda_fp8.h — **NOT a typedef**: ROCm 6.2 ships only E4M3_FNUZ while
  CUDA's __nv_fp8x2_e4m3 is E4M3FN (different encodings; aliasing would be
  numerically wrong) — shim implements exact E4M3FN decode (bit-splice);
  audited usage = embed_gather.cuh `.____x`+float2 cast. cub/block/
  block_merge_sort.cuh → hipcub (API-compatible form verified against
  sampling.cuh call sites); math_constants.h (CUDART_INF_F).
- **NEW BLOCKER CLASS:** second product PTX header `ops/common/math.cuh`
  (`ex2.approx.f32` with '=f' constraint — rmsnorm path reaches it via
  chain; first was memory.cuh). HELD files now: add_bias, residual_add,
  gelu, sigmoid_gate_mul (memory.cuh) + rmsnorm, silu_and_mul, gdn_gating
  (math.cuh chain). **The overlay-vs-sanctioned-edit ruling gates 7 of the
  17 T1 files — it is now the critical-path decision, not an edge case.**
- Warp-shuffle emulation added (CUDA width-32 semantics on 64-lane wavefronts:
  out-of-group accesses return caller's own value per CUDA spec; guarded
  __HIPCC__). NOTE for review: segs assume power-of-two width — true for every
  current call site (32 / warpSize).
- Device verification of batch 2 NOT run — per new cadence requesting launch
  window (compile done, compile-window use only).

## Decision-support for the memory.cuh/math.cuh ruling (pre-work, no tree change)

ODR audit complete: ALL includers of both headers use the path form
"ops/common/{memory,math}.cuh" (zero bare-relative includers, including the
headers' own self-includes) → the include-path OVERLAY (option A) has no
split-brain exposure it initially feared. Drafts of BOTH options prepared in
/tmp/ptx_hdr_options/ (ready to apply same-minute either way):
- A: overlay copies with #if !defined(__HIP__) guards around every PTX
  section (6 asm sites memory.cuh, 2 math.cuh) and sync-staging/zfill-
  faithful/exact-RNE emulations.
- B: same bodies applied as sanctioned edits to the two product files
  (identical diff content; visible to CUDA-line reviewers).
Numeric notes for the gate design: pack_bf16x2 emulation is BIT-EXACT (RNE);
exp2_approx→exp2f is approximation-class, NOT bit-identical to ex2.approx —
rmsnorm/silu numerics need tolerance bands (consistent with D3 re-baseline).
Exposure: 7 T1 files held; T2 direct includes on 3 of 7 GEMV decode files
(w8_rowsplit_gemm_decode, swiglu_decode, pair_decode); T3 i8/slice bodies all
includers. Recommendation stands: B (two small honest guards beat invisible
shadowing), but A is now known-safe too — either applies cleanly.

## G-AMD-5 EXECUTED — peer probe (results/v340l/peer_probe.md), window 10:47-10:50Z of 20
- HEADLINE: hipDeviceCanAccessPeer = 0 ALL 6 pairs (incl. same-card siblings). No P2P exists; every cross-device byte is host-staged. Device-write one-shot AR design CANNOT port as written — it is a host-staged REDESIGN.
- Matrix: cross-card staging 6.61-6.70 GB/s (one-way payload); same-card-sibling pairs 4.86 (contention, both legs of one card's upstream); RTT ~98-101 us per 2-hop API round trip (~50 us/hop, flat vs size => latency-bound). All under the 183 GB/s anchor — sane.
- Physical grouping derived: card A = 05:00(dev0)+08:00(dev1); card B = 0d:00(dev2)+10:00(dev3). TP2 pair choice: CROSS-CARD (dev0+dev2).
- 130 ARs x 100us = ~13 ms/step unbatched = not viable; batched (33-65) = 3.3-6.5 ms = visible but workable with pinned-buffer double-buffering design.
- Release: No KFD PIDs, 4x 18,575,360 B verified. Sampler log: peer_probe_sampler.log.

## T2 first contact: 5/7 joined (159/159 sha1=eadacf35a7); mma.cuh = THIRD product PTX header, ruling requested
- Joined: attn_input w8 decode, gdn_input w8 decode, gdn_projected_conv, swiglu w8 decode, pair_decode — compile clean, device-verify pending.
- Held: w8_rowsplit_gemm_decode.cu + w8_pair_gemm_concat.cu — include chains reach ops/common/mma.cuh (7 mma wrappers incl. m16n8k16 bf16/f16 '+f' asm). NOT a cheap sync-emulation class: fragment-layout-correct m16n8k16 emulation is a DESIGN ITEM (or abort-stub the never-dispatched variants behind the same registered-exception mechanism — minimal-path GEMV dispatch never calls them at runtime, only the header must parse).
- Proposal for the third exception: mma.cuh guard = mma_bf16/mma_f16 exact outer-product fragment emulations (correctness insurance, ~40 lines, slow is irrelevant if never dispatched) + remaining five as loud-abort stubs (calling one on HIP = dispatch defect, mirrors the AR-stub doctrine C441 set). Awaiting ruling; product-file exceptions stay named-registered either way.

## G-AMD-8 post-analysis (CPU): argmax RED = harness again — valid_rows is the VOCAB SCAN LIMIT (argmax.cuh:65), I passed T=2 (scanned 2 of 1024). Sample-vs-argmax red was downstream of the crippled argmax. Harness v6 fixed+relinked (marker present). REMAINING TRUE RED: l2norm only (scale implies ~half the row's squares missing; prime suspect my __shfl emulation or reduce group semantics — needs a targeted device probe, next stamp).
## LAYOUT RULE (for docs/174 + gemini): {d, rows} with ne[0]=feature-dim holds across T1/T2 launchers (l2norm/layer_norm/scatter confirmed); argmax logits token-major [t*V+v]; argmax valid_rows = scan limit.

