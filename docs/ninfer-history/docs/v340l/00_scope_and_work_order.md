# 00 — V340L HIP port (2-device fixture gate → 4-device full model): Agent Work Order

> Filled from `docs/99_agent_work_order_template.md`. Hand an implementer:
> "work on this task in this doc".
>
> **Status:** CURRENT — work order for a single implementer agent (needs a
> worktree `wo/v340l-hip` and, before any GPU launch on the AMD box, a
> written GPU grant from the coordinator per AGENTS.md).
>
> **Mission:** Stand up a HIP/ROCm build lane of THIS repo (not upstream
> ninfer) that serves the `groupwise-int` Qwen3.8-27B artifact on AMD Radeon
> Pro V340L hardware: milestone M2 = 2 HIP devices (one card — fixture-based
> TP2 correctness only, the artifact does not fit on one card), milestone M4
> = 4 HIP devices / 2 cards (full 20.4 GB artifact, plain-decode serving via
> tensor split). Every step lands with a phase-gate test; a dedicated AMD CI
> lane (`tools/ops/run_ci_amd.sh`) gates all of it. The CUDA build and the
> existing CI are never modified in behavior.
>
> Read this document fully before writing code. Read `docs/01`,
> `docs/02`, `docs/04` (V340L hardware/perf background — ARCHIVE but the
> hardware facts are the planning basis) and `docs/03` §4 (portability
> analysis of upstream ninfer; note this repo has since BUILT the multi-GPU
> layer that doc says upstream lacks — `src/core/multi_gpu/` + `src/runtime/tp2/`).

---

## 1. Context (60-second version)

This repo runs Qwen3.8-27B (`groupwise-int` weights, 20.4 GB artifact) on
2× RTX 5060 Ti over a TP2 path (`src/runtime/tp2/` + `src/core/multi_gpu/`)
with CUDA 13.1, `sm_120a`-only builds. The user is porting the project to
AMD V340L cards (per docs/01+04: each card = 2 gfx900 dies = 2 HIP devices,
8 GB HBM2 @ 483.8 GB/s per die, fp16 2:1, **no tensor cores, no bf16 ALU, no
FP4/FP8**). AMD is not CUDA-compatible; ROCm/HIP is the only viable route.
The port must be the SIMPLEST implementation that works: the minimal kernel
subgraph on the active serving path, plain decode (spec off), dual-backend
CMake so CUDA stays intact, phase-gate tests per step, and an AMD CI lane.

**The one fact that shapes the whole plan:** the artifact is **19.03 GiB**
(20,437,336,576 B; HF `neroued/Qwen3.8-27B-NInfer`, sha256 `0634abb0…2a467eec3`,
`weights_id=groupwise-int`, container v2, verified from the HF
`artifact-manifest.json` and independently re-verified by the coordinator).
One V340L card = 2 dies = **15.97 GiB**; one die/device = **7.98 GiB**
(measured on this host 2026-09-12 — this box IS the target, docs/174).
**"2 GPUs" (2 devices, one card) cannot hold the model.**
Therefore M2 (2 devices) is a *fixture-based correctness* milestone using the
existing shard fixtures in `tests/multi_gpu/`; M4 (4 devices / 2 cards) is
the first full-model milestone. Resolved (coordinator ruling C441,
2026-09-12, docs/174): "2 GPUs then 4" = **HIP devices**, not cards.

**Already built and verified (do not redo):**
- TP2 serving path on CUDA: `src/runtime/tp2/` (tp_engine, tp2_budget,
  tp2_backend), `src/core/multi_gpu/` (one_shot_allreduce — P2P-memcpy based,
  **no NVLink/multimem dependency** — tp_group, weight_shard, tp_kernel).
- TP2 test ladder: `tests/multi_gpu/` — `test_tp_group.cpp`,
  `test_weight_shard.cpp`, `tp2_shard_plan.cpp`, `tp2_load.cpp`,
  `tp2_decode.cpp`, `test_one_shot_correctness.cu`,
  `test_tp2_interleaved_prefill.cpp`, plus shard fixtures
  (`bf16_requant_fixture.h`, `gen_bf16_requant_fixture.py`).
- Phase-gate infrastructure: `tests/phase_gate.cu` (995 lines),
  `tests/phase_gate_artifacts.h`, `tools/ops/run_ci.sh` (1337 lines) — the
  pattern to clone for AMD, not to modify.
- CPU reference implementations for op parity: `tests/kvarn_*_cpu_ref.cpp`,
  `tools/reference/` (Python).
- Artifact identity/format: `src/artifact/` (container v2),
  `src/targets/qwen3_8_27b/` (closed-model specialization — portable as-is).

**What you are doing:** steps 0–8 of §6, in one line each: environment
verify → HIP build lane → single-device runtime primitives → minimal op set
parity → single-device layer parity → TP2 fixture gate (M2) → TP4 full-model
gate (M4) → AMD CI lane wired end-to-end.

## 2. Environment & build/test

- **ONE BOX, dual roles (measured 2026-09-12, docs/174):** this host is
  BOTH the repo checkout AND the AMD target — 4 HIP devices (kfd nodes 1–4),
  **gfx900** (not gfx906), **7.98 GiB/device** (8,573,157,376 B), 2 cards ×
  2 dies (user confirmed), **ROCm 6.2.0-66 pinned** (do not chase newer
  without a stated reason), hipcc 6.2.41133 at `/opt/rocm`. The only NVIDIA
  GPU is a GTX 1050 Ti (display) and there is **no nvcc** — the CUDA build
  cannot be compiled or run here. All GPU work requires a **written GPU
  grant from the coordinator (intercom) before launching anything** —
  AGENTS.md standing rule. Kill only PIDs you started; `pkill -x`
  exact-name only, never `pkill -f`. Step 0's hardware half is DONE
  (docs/174 measured board facts) — do not re-measure, cite it.
- **Work protocol — MANDATORY worktree + branch (this lane's branch is
  `wo/v340l-hip` at `~/worktrees/wo-v340l-hip`):**
  ```bash
  git worktree add ~/worktrees/wo-v340l-hip -b wo/v340l-hip
  cmake -S . -B build && cmake --build build -j 16
  ```
  Commit per step to `wo/v340l-hip`. Merging to main is done by the
  main-side agent/user.
- **Dual-backend build rule (NEW, load-bearing):**
  - CMake gains `NINFER_BACKEND` (`cuda` default | `hip`). The `hip` path
    sets `LANGUAGES C CXX HIP`, `CMAKE_HIP_ARCHITECTURES=gfx900`
    (measured), and **excludes from the build** every kernel dir not on the
    minimal path: all `src/ops/*/nvfp4/*`, all `src/ops/*/fp8/*`, the
    MMA/splitk GEMM variants not needed for groupwise-int decode, DFlash,
    MTP drafter kernels, kvarn flash variants beyond the minimal decode
    path. Exclusion is by explicit file list in the backend CMake (whitelist
    beats blacklist — new CUDA-only files must fail the HIP build loudly;
    that negative-test cell lives in **Step 1**, not Step 7).
  - **CUDA-path guarantee on THIS host is source-diff/whitelist only**
    (coordinator ruling C441): HIP-side work ADDS files and makes ZERO
    diffs to CUDA-side files; a CI cell asserts it. The genuine CUDA
    regression proof (build + gates) **cannot be produced on this box** —
    it rides the merge package as an explicit OPEN item for the
    NVIDIA-line host. `run_ci_amd.sh` must never print a CUDA-side green
    it cannot produce (same defect class as D5's silent section).
  - No `-W` flag changes; keep the narrow `-Wswitch` behavior.
- **Artifact acquisition (Step 6):** download
  `https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer`
  (19.03 GiB). **`df -h` first — disk is the shared constraint** (58 GB free
  on `/` at scoping time; the artifact + extraction headroom must fit).
  Verify sha256 against `artifact-manifest.json` (0634abb07024221de141456cf
  04a42ab74b18bc38e1b781c6eb2e062a467eec3). Record bytes verified in
  `docs/v340l/PROGRESS.md`.
- **Tests:** `/usr/bin/ctest` from `build/`. Every step's phase gate is a
  named ctest target AND a script entry in `tools/ops/run_ci_amd.sh`.
  **Ownership boundary (coordinator ruling C441, hard):** phase-gate
  **implementation + CI wiring belong to gemini's exclusive test lane**
  (docs/174 §3; gemini ACCEPTED, hub #14) — not reassignable to A1/A2, and
  not authored inside this product work order. This doc keeps the **PG
  specs** (what each gate must prove, call sites, whitelist semantics) as
  design input for gemini's WO (`docs/v340l/01_amd_phase_gate_test_lane_work_order.md`);
  the implementer hands specs over and cold-reads/re-runs gemini's gates
  before believing any PG row.
- **Measurement discipline (VRAM LAW applies on the AMD box too):**
  preflight/budget code reports MEASURED numbers only — `hipMemGetInfo` +
  actual bytes; no estimate-based refusal constants anywhere in new code
  (user order 2026-09-10; see AGENTS.md). Every perf number ships with
  `rocm-smi` clocks. Prompt sizes from `usage.prompt_tokens`, never char
  estimates. Measurement outputs committed under `results/` with config.
- **Pre-registered M4 binding constraint (coordinator C441):** weights
  19.03/4 ≈ 4.76 GiB/device on 7.98 GiB leaves **~3.2 GiB/device** for KV +
  activations + context. That slack, not the weights, is what will kill the
  first launch. First real launch reports the measured floor
  (`hipMemGetInfo` + actual bytes; a near-capacity cell LAUNCHES and
  MEASURES — it never refuses on an estimate).
- **AMD CI (§6 Step 7):** `tools/ops/run_ci_amd.sh` mirrors `run_ci.sh`'s
  shape (state record → build → gates → restore) but targets the HIP build
  dir and runs: zero-GPU unit tests → device smoke → op parity → TP ladder
  → (nightly) decode-guard cells. It must be runnable on the AMD host
  WITHOUT a grant only in its zero-GPU stages; GPU stages gate on the
  coordinator grant check. Owned by gemini (test lane); this lane consumes
  it and supplies the PG specs.
- The artifact lives on THIS box (same host); download location recorded in
  PROGRESS.md at Step 6.

## 3. Architecture facts (verified — do not re-derive)

- **Board (MEASURED on this host 2026-09-12 — docs/174, do not re-derive):**
  this box is the v340l target: 4 HIP devices (kfd nodes 1–4), **gfx900**,
  **7.98 GiB/device** (8,573,157,376 B, rocm-smi), 2 cards × 2 dies,
  ROCm **6.2.0-66** pinned. Only-just facts to keep straight: 19.03 GiB
  artifact > 15.97 GiB/card ⇒ M2 is fixture-only; M4 weights ≈ 4.76 GiB/device
  leaves ~3.2 GiB/device for KV+activations — the binding constraint.

- **Repo = this tree**, not upstream ninfer: it HAS multi-GPU (docs/03 §4.2
  "single-GPU assumption" row is stale for this repo).
- **CMake** (CMakeLists.txt:1–60): `LANGUAGES C CXX CUDA`, hard-forced
  `CMAKE_CUDA_ARCHITECTURES=120a` (FATAL otherwise), CUDA ≥ 13.1 required,
  C++20. HIP build adds a parallel arch gate; the CUDA gate is untouched.
- **CUDA surface:** 284 `.cu`/`.cuh` files; 375 files touch CUDA APIs. The
  port touches only the minimal subgraph (§6 Step 1 produces the exact list);
  everything else is compile-excluded under HIP.
- **Quant profile that matters:** the artifact is `weights_id=groupwise-int`
  (HF artifact-manifest.json). The NVFP4 W4A4 path (docs/03 §3.5 "Blackwell
  crown") is **irrelevant** — gfx900 has no tensor cores. The port's kernel
  core is the w8/groupwise SIMT dequant path (`src/ops/linear_add/w8/`,
  `src/ops/attn_input_proj`, `src/ops/gdn_*` non-nvfp4 variants), which
  dequantizes in-kernel and was designed bandwidth-bound — exactly the
  gfx900 shape.
- **Activation dtype:** current kernels are bf16-activation. gfx900 has NO
  bf16 ALU (docs/01 §1, docs/03 §4.2). Decision D3 below: fp16 activations
  (fp32 accumulate) on HIP; bf16 storage → fp16 at load. Expect tolerance
  re-baselining, not byte equality, vs CUDA outputs.
- **Multi-GPU primitives are portable:** `one_shot_allreduce.cu` uses plain
  `cudaMalloc`/P2P memcpy patterns (no `multimem`, no NVLink intrinsics);
  `hipMemcpyPeer`/`hipGraph` equivalents exist in ROCm. TP2 topology:
  shard plan (`src/artifact/plan_split.h`, `weight_shard.cpp`) + per-rank
  engine + one-shot AR per layer.
- **GDN is load-bearing:** 48 of 64 layers are gated-delta-net linear
  attention (`src/ops/gdn_input_proj/`, `gdn_gating_proj/`,
  `linear_attention/`, `linear_swiglu/`, `linear_pair/`). No off-the-shelf
  gfx900 substitute exists; the recurrent-decode + chunked-prefill kernels
  must be ported. This is the largest single kernel-work item.
- **Decode orchestration:** `src/core/decode_graph.cpp` (CUDA graph
  capture→instantiate→upload→launch) → `hipGraph*` 1:1. Paged KV +
  INT8 group-64 KV (`paged_kv_cache.cpp`, `cyclic_kv_cache.cpp`) are
  storage+dequant — portable.
- **Serving path identity:** TPEngine (NOT ConcurrentExecutor) is the live
  path (REPO.md §2a). The TP2 preflight region in `src/runtime/tp2/
  tp_engine.cpp` + `tp2_budget.h` is the canonical VRAM-decision home
  (ANTI-RESURRECTION RULE): the HIP tree must keep this region in sync with
  main and use `hipMemGetInfo`; the CI cell below (Step 7) asserts it.
- **gfx900 kernel constraints** (docs/01 §1–2): 64-wide wavefronts (2×
  register pressure per logical thread vs 32-wide warp), 64 KB LDS/CU
  (halve K-chunks vs 128 KB), LDS is 32×4B banks (XOR swizzle math
  survives), fp16 2:1 rate, no DP4A-on-separate-pipe (don't bother), M-wall
  at M≈6–8 (plain decode M=1 and small verify widths only — fine for this
  scope; MTP is out of scope).
- **Hardware numbers are planning-only until measured:** 483.8 GB/s/die and
  "2 dies/card, 8 GB/die" come from docs/01+04 [inference]/community tags;
  Step 0 measures the real memcpy ceiling per die and re-derives the
  roofline. docs/04 §5.6: V340L units vary (power caps, limp-mode reports) —
  measure per card, slowest-die gating applies.

## 4. Key call sites (anchors — verify line numbers before editing)

- `CMakeLists.txt:6–13` — arch gate to clone (not modify) for HIP.
- `CMakeLists.txt:24–52` — CUDA standard/version checks: HIP branch skips
  these, sets its own.
- `src/CMakeLists.txt` + per-dir CMake — where the backend-conditional file
  whitelist lands.
- `src/core/device.cu/.h` — device init/stream management; first port
  target.
- `src/core/arena.cu`, `src/core/tensor.cpp`, `layout.cpp` — allocation +
  layout layer.
- `src/core/multi_gpu/one_shot_allreduce.cu` — AR port (P2P memcpy based).
- `src/core/multi_gpu/tp_group.cpp`, `weight_shard.cpp` — TP topology +
  sharding (host-side, mostly portable as-is).
- `src/runtime/tp2/tp_engine.cpp` (esp. lines ~680–800: probe + VRAM
  preflight region) + `tp2_budget.h` — ANTI-RESURRECTION region; HIP tree
  keeps it main-identical except `cuda*`→`hip*`.
- `src/targets/qwen3_8_27b/` + `src/targets/registry.cpp:94–195` — the
  closed-model load plan; per-shape `Geometry/Schedule` specialization
  (docs/03 §3.2) ports 1:1 and is the port's biggest design asset.
- `src/ops/linear_add/w8/*` — the groupwise-int GEMV/GEMM family to port
  first (SIMT variants, not mma).
- `src/ops/common/warp.cuh` — `__shfl_xor_sync` etc. → HIP intrinsics shim.
- `tests/phase_gate.cu` — phase-gate pattern to clone per phase (PG-A…PG-E).
- `tests/multi_gpu/tp2_load.cpp`, `tp2_decode.cpp`,
  `test_one_shot_correctness.cu` — M2 gate bodies to port.
- `tools/ops/run_ci.sh` — the lane shape `run_ci_amd.sh` mirrors.

## 5. Design decisions (FINAL — do not re-litigate)

1. **Dual-backend CMake, not a fork.** One tree; `NINFER_BACKEND=hip`
   whitelist-gates sources. Reason: the CUDA path must never regress — and
   on THIS host that assurance is **source-diff/whitelist only** (no nvcc;
   HIP work ADDS files, zero diffs to CUDA-side files), with the genuine
   CUDA regression proof riding the merge package as an explicit OPEN item
   for the NVIDIA-line host (coordinator ruling C441). The repo's whole
   test culture (run_ci, phase gates) is the asset — forking the tree would
   orphan it. REJECTED: separate HIP fork repo (doubles maintenance, loses
   CI history); hipify-perl whole-tree conversion (284 files — converts
   dead nvfp4/fp8 code for nothing and risks silent numeric changes in live
   code).
2. **Port the minimal serving subgraph only; whitelist-build under HIP.**
   Plain decode, `--spec none`, concurrency 1 to start. MTP/DFlash/nvfp4/fp8
   stay CUDA-only until plain decode is green. REJECTED: porting MTP in the
   first pass (docs/01 §3: k≤4 only, acceptance re-derivation needed —
   separate work order after M4).
3. **fp16 activations + fp32 accumulate on HIP** (bf16 converted at load).
   gfx900 has no bf16 ALU; v100-skinny's measured recipe on equivalent
   silicon is fp16/fp32-accum. Consequence: numerics are re-baselined on
   AMD (self-consistent greedy anchors), never diffed byte-exact across
   backends. REJECTED: fp32 activations everywhere (2× activation traffic
   in bandwidth-bound kernels for no correctness need).
4. **Fixture-first multi-GPU (M2) because the artifact cannot fit one
   card.** Reuse `tests/multi_gpu` shard fixtures on 2 devices; the real
   artifact debuts at M4/4 devices. REJECTED: shrinking KV to squeeze the
   artifact onto one card (0 headroom = untestable; and 19.03 GiB >
   15.97 GiB cannot fit at all).
5. **TP = tensor split, one-shot AR over PCIe.** Same topology the repo
   already runs on CUDA (tp2), and docs/02 §2 shows within/across-card AR
   is latency-bound but µs-scale for ~10 KB hidden vectors. No cross-die
   fabric exists — layer-split alternative is strictly worse (serial weight
   read, docs/02 §1).
6. **Phase gates before speed.** No perf tuning before PG-E is green;
   targets come from measured rooflines (Step 0), not docs/02 estimates.
7. **VRAM LAW from day one on the AMD host:** any preflight in the HIP path
   uses live `hipMemGetInfo` + actual bytes; the CI step-0 cell diffs
   `tp_engine.cpp` + `tp2_budget.h` against main and fails loud on stale
   regions (extends the existing anti-resurrection cell to the HIP tree).
   REJECTED: any fixed budget constant in new HIP code.

## 6. Execution order (commit + phase gate each step before the next)

**Testing standard (from the template, applies to every step):** unit/kernel
isolation is necessary, never sufficient. Any step touching a server-facing
path must end with the real server (or the real engine entry) executing the
code path end-to-end on the AMD hardware. A kernel passing against its CPU
ref does not make a step done.

**Spec-level naming + provenance rules (per coordinator C441, 2026-09-12):**
the test target names used below (`v340l_device_smoke`, `v340l_op_parity_*`,
`pg_layer_forward`, `v340l_op_bench_*`) are SPEC names — the ctest targets and
labels are created and owned by the test lane (gemini, docs/v340l/01), and
every label/name-based cell must carry a non-zero-test-count guard
(`ctest -L x` exits 0 on an empty selection — that is a vacuous pass).
Gate IMPLEMENTATION is gemini's; this WO owns what each gate must prove.
Until gemini's PG-1 negative test lands, the whitelist's
"new CUDA-only files fail the HIP build loudly" property is
**designed-not-demonstrated** and must be reported that way. Any
"% of roofline" claim must name its denominator: theoretical per-die
483.8 GB/s vs the measured sustained D2D copy ceiling (~183 GB/s, PG-0b —
≈37.8% of theoretical, at max mclk 945 MHz, all four devices at parity).

### Step 0 — Environment verify + measured roofline (hardware half DONE)
**DONE, measured on this box 2026-09-12 (docs/174 — cite, do not re-measure):**
4 HIP devices (kfd nodes 1–4), gfx900 (kfd `gfx_target_version 90000`),
7.98 GiB/device (8,573,157,376 B), 2 cards × 2 dies (user confirmed),
ROCm 6.2.0-66 pinned, hipcc 6.2.41133. **Remaining (needs a written grant):**
the memcpy ceiling probe per die (port `tools/hbm_bandwidth_probe.cu` first
if needed — it is a ~50-line bandwidth loop, not the engine), record per-die
GB/s + clocks + link widths (`rocm-smi`, `lspci -vvv | grep LNK` — x8 vs x4
per die, docs/04 §5.3).
**Gate PG-0:** `results/v340l/step0_roofline.md` with per-die memcpy GB/s,
link widths, clocks. This is the roofline every later perf number is judged
against.
**Tests:** none (this step PRODUCES the measurement anchor).

### Step 1 — HIP build lane + backend CMake gate
Add `NINFER_BACKEND` to CMake (cuda default unchanged; hip sets HIP language,
gfx900, source whitelist). Create the shim header (e.g.
`src/common/hip_compat.h`) aliasing the CUDA runtime/intrinsics surface to
HIP (`cudaMalloc→hipMalloc`, `__shfl_*_sync`, `__syncthreads`, warp size,
`cudaGraph*`→`hipGraph*`, `cudaMemcpyPeerAsync`→`hipMemcpyPeerAsync`, stream/
event types). Verify what hipify-perl leaves behind on the whitelisted files
before hand-aliasing — prefer mechanical conversion + shim for intrinsics.
The whitelist at this step covers only what compiles host-side: `artifact/`,
`targets/`, `runtime/tp2/` host files, `serve/`, `apps/`.
**Gate PG-1 (zero-GPU):** HIP build compiles host-side; a CI cell asserts
the whitelist FAILS LOUDLY when a non-whitelisted CUDA file is included from
a HIP TU (negative compile test or CMake-level check) — **this cell lives
here, Step 1, not Step 7** (coordinator C441: the lane must never be behind
the code); a cell asserts CUDA-side files carry ZERO diff vs main (the
only CUDA guarantee provable on this box — no nvcc here; real CUDA proof
rides the merge package as an OPEN item); **and a cell asserts whitelist
PARITY — every listed source produces an object** (`src/CheckHipArchive.cmake`
POST_BUILD guard is the in-tree precedent; a drop must be a build failure,
never a review discovery — measured 2026-09-12: CMake silently drops `.cu`
sources in a HIP-only project with build EXIT 0). Host-only unit tests green
under HIP build (artifact reader, shard plan, tp2 budget host math).

### Step 2 — Single-device runtime primitives (1 die)
Port `device.cu`, `arena.cu`, `tensor.cpp` alloc/free/stream paths.
**Gate PG-A:** device smoke test binary (init, alloc, H2D/D2H, P2P memcpy
within card, stream sync, `hipGraph` capture-instantiate-launch of a trivial
kernel) + the Step-0 bandwidth probe re-run through the engine's own alloc
path (sanity: engine allocations don't degrade BW).

### Step 3 — Minimal op set parity (1 die, kernel level)

> **Port-surface inventory (mechanical, Step 2 output): see
> `results/v340l/step2_symbol_inventory.md`** — 435 unresolved `ninfer::`
> symbols attributed to 128 `.cu` files (UPPER BOUND); the minimal-path port
> set is ~40–50 files (D2 keeps nvfp4/fp8/q6/moe excluded). Files join the
> `src/HipSources.cmake` whitelist ONE AT A TIME, only once they compile —
> the parity guard's meaning depends on it. Largest single item: the GDN
> kernel family; Step-5 AR trio blockers already measured (PTX `l`-constraint
> asm → volatile-access rewrite).
Port, in dependency order: rmsnorm/elementwise, w8 groupwise GEMV (M=1) and
small-T (M=2..8) from `src/ops/linear_add/w8/` (SIMT variants), attention
decode + prefill (int8 group-64 KV dequant in-kernel), GDN recurrent decode
step + conv, rotary, argmax/sampling. Each op: parity vs the existing CPU
ref (`tests/kvarn_*_cpu_ref.cpp` pattern) with fp16-activation tolerances,
plus a per-op microbench vs the Step-0 roofline (% of BW for GEMV; % of
fp16 peak for prefill GEMM) — **bandwidth-bound GEMV must land ≥60% of
measured per-die BW before proceeding** (docs/02 §5 assumption 2; if it
can't, STOP and re-tune — everything downstream inherits the loss).
**Gate PG-B:** ctest set `v340l_op_parity_*` + `v340l_op_bench_*` green,
roofline percentages recorded in `results/v340l/`.

### Step 4 — Single-device layer + whole-model-shard forward (1 die)
Run one full transformer block (attn + GDN + MLP) and then a full forward of
a shard-sized stack on one device against golden activations captured from
the CUDA build on identical inputs (fixture tensors, fp16 tolerance bands
fixed in the test, not eyeballed).
**Gate PG-C:** `tests/v340l/pg_layer_forward` green (block-level tolerance
match); greedy decode of ≥64 tokens on a shard model produces a stable,
self-consistent token stream (no NaN/divergence class bugs).

### Step 5 — TP2 on 2 devices (M2 milestone, fixtures)
Port `tp_group`, `weight_shard` runtime, one-shot AR, and the TP2 engine
host path. Run the ported `tests/multi_gpu` ladder (tp2_shard_plan →
tp2_load → tp2_decode → test_one_shot_correctness → interleaved prefill)
on 2 HIP devices of one card.
**Gate PG-D (M2 DONE):** full ported ladder green on 2 devices; AR
correctness test (N≥1000 iterations, no epoch/race flake); decoded tokens
from fixture shards match the CUDA-run fixture goldens within tolerance;
AR latency measured and recorded (docs/02 §2 predicts µs-scale — verify,
don't assume).

### Step 6 — TP4 with the real artifact (M4 milestone, full model)
Download + sha256-verify the 19.03 GiB artifact (§2; `df -h` first). TP4
tensor-split across 4 devices / 2 cards. Launch per LAUNCH.md pattern with
`--spec none`, modest `--kv-capacity` (VRAM LAW: measured
`hipMemGetInfo` floors only — the preflight computes from live free bytes
+ actual bytes; NO constants; a near-capacity cell LAUNCHES and MEASURES),
concurrency 1, greedy. Remember the pre-registered binding constraint:
~3.2 GiB/device slack for KV+activations after weights (§2).
**Gate PG-E (M4 DONE):** the real server on 4 devices answers a real HTTP
request end-to-end; a ≥30-minute soak at concurrency 1–2 without leak/OOM;
greedy output is stable across restart (self-consistency); measured plain
decode t/s + pp t/s recorded in `results/v340l/` with clocks, against the
Step-0 roofline. **This is the deliverable end state.**

### Step 7 — AMD CI lane wired end-to-end (implementation: gemini's lane)
`tools/ops/run_ci_amd.sh` is **implemented by gemini per its WO
(`docs/v340l/01_amd_phase_gate_test_lane_work_order.md`), from the PG specs
in §6** (ownership boundary, §2). Shape: stage 0 zero-GPU (backend whitelist
negative test, tp2 anti-resurrection diff vs main, CUDA-side zero-diff vs
main, unit tests), stage 1 build-hip, stage 2 PG-A/B (single device),
stage 3 PG-D ladder (2 devices), stage 4 PG-E smoke (4 devices; nightly =
PG-E full + soak). It must NOT print a CUDA-side green it cannot produce on
this box. This lane's role: hand over specs, cold-read + re-run gemini's
gates before believing any PG row, write evidence not agreement.
**Gate PG-F:** one full `run_ci_amd.sh` green run recorded (log in
`results/v340l/`).

### Step 8 — Report + handoff
PROGRESS.md updated per phase-gate; one paragraph per step; key-numbers
table (per-die BW, GEMV %BW, AR latency, decode t/s @ ctx, pp t/s); open
items (MTP work order next; kvarn extras; multi-stream batching).

## 7. Constraints (non-negotiable)

- **Worktree only; no main-tree edits** (§2). This lane: `wo/v340l-hip` at
  `~/worktrees/wo-v340l-hip`.
- **GPU grant before every launch on this host** (coordinator, written,
  intercom). Guard on foreign CUDA/ROCm contexts. Kill only your own PIDs;
  repo copy of gpu_guard.sh by absolute path; never system-wide pkill.
- **VRAM LAW:** no estimated-charge refusals anywhere in new code — live
  `hipMemGetInfo` + actual bytes only (AGENTS.md, user order 2026-09-10).
- **CUDA path untouched — enforced as source-diff on this box:**
  `NINFER_BACKEND=hip` is additive; HIP work ADDS files and makes ZERO
  diffs to CUDA-side files; the tp2 preflight region semantics stay
  main-identical (ANTI-RESURRECTION RULE). Real CUDA build/gate proof rides
  the merge package as an explicit OPEN item (no nvcc here — ruling C441).
- **Test-lane boundary:** phase-gate implementation + CI wiring are
  gemini's exclusive lane (docs/174 §3); not authored in this lane, not
  reassigned to A1/A2. This lane owns the PG specs and verifies gemini's
  gates by cold-read + re-run.
- **Doc numbers:** global `docs/NN_` numbers are claimed from the
  coordinator only (docs/174 = the AMD line's pointer, claimed by C441);
  folder-local `docs/v340l/NN_*` is blessed.
- **Disk:** `df -h /` before the artifact download and any large build;
  clean your own scratch; never delete another lane's files.
- **No perf claims without measurement:** every quoted number carries
  clocks + config + committed output (§2).
- Live end-to-end before "done" for every server-facing step (§6 standard).

## 8. Definition of done

1. Steps 0–7 committed to `wo/v340l-hip` (or the AMD-side branch) with each
   phase gate green at its step.
2. **Live proof (M4):** a real launch of `ninfer-serve` (HIP build) on 4
   HIP devices serving the groupwise-int Qwen3.8-27B artifact answers real
   requests; launch command + serve log recorded.
3. `tools/ops/run_ci_amd.sh` green end-to-end once, log committed.
4. `results/v340l/` holds: roofline (Step 0), op/AR/decode measurements with
   clocks, artifact sha256 verification record.
5. `docs/v340l/PROGRESS.md` current; open questions resolved or explicitly
   carried forward; next work order (MTP on gfx900, k≤2–4) drafted as a
   stub.

## Former open questions — ALL RESOLVED (coordinator ruling C441, 2026-09-12)

1. **"2 GPUs then 4" = HIP devices**, not cards (the cards reading demands
   M4 = 8 devices/4 cards; this box has 2 cards). No §6 rescope.
2. **Silicon: gfx900** (kfd `gfx_target_version 90000`, rocminfo), 2 dies
   per card, each die = 1 HIP device; 7.98 GiB/device.
3. **ROCm pinned 6.2.0-66.**
4. **Same box** — this checkout IS the target host; no nvcc present, so the
   CUDA guarantee here is source-diff/whitelist only (§2, §7) and real CUDA
   proof rides the merge package as an OPEN item.

## Late additions (2026-09-13, session end) — read these three first if you are merging or citing counts

The folder's numbered docs `01`–`19` are session history; these three are forward-looking and each one changes
an action, so they are indexed here because nothing in the successor's reading path names them.

- **`20_ldmatrix_census_predicate_named.md`** — the `{128, 140, 152}` site-count family, each figure attached to
  the predicate that produces it, plus the ref-stamp rule for violation counts. Cite from this, not from chat.
- **`21_shim_verifier_poison_check_is_textual.md`** — measured: the shim verifier prints
  `PASS: poison-free, width-32 contract certified` (rc=0) on a header with
  `#define __shfl_down_sync(v,x,y,z) (poisoned)` appended. One regex, three names, textual not semantic.
  Proposed fix is in the file; the gate is not mine to patch.
- **`22_merge_readiness_snapshot.md`** — the eight registration rows are **main's** (36 entries; gemini's branch
  never edited the array, so its 17 *is* the base and "gemini's 32" is unreproducible), and the one real merge
  blocker is an **`add/add` conflict on `docs/amd/TRIPWIRE_GATE_CLASS_SPEC_agent5.md`** — default resolution
  discards one lane's edits to the requirements sheet. Zero-write reproduction:
  `git merge-tree --write-tree --name-only origin/amd/main origin/amd/wo-gfx900-perm`.
