# WO-04 — Q3-on-HIP bring-up (agent 3) — issued 2026-09-12 ~13:35Z, coordinator C441

**Base:** branch from `amd/main` (cite by name; was `2c4b8110` at issue). Verify base: `src/ops/linear/q3/q3_rowsplit_storage.h` present at sha256 `e66a4270d64ed5dedb9cda97ef05c0d5224e42cf0620dc20490f2b7d8bed607b`, and `src/common/hip_shim/cuda_runtime.h` contains the string `unconditional: all lanes publish`. If either fails, stop and ask coordinator — you are on a stale base.

## 1. Goal
Make the Q3 ops layer COMPILE and PASS CPU goldens under the HIP build, and land the `embed_gather` Q3 arm — the load-gate for any artifact with a quantized embedding. Product of this WO: the Q3 chain is build-verified on our box, so serve bring-up (WO-05) has nothing left to discover.

## 2. Test entry points (gates — this limits your possibility space)
- **CPU goldens FIRST, device later:** Team Green's host-twin goldens came with the promote — `tests/ops/` q3 files (`q3_a16_cpu` 13/13 per their promote log incl 4 TP2 per-rank shapes, `embed_gather_golden`, `q3_quantizer`, `quant_recipe`). Run them under plain g++/python on CPU before touching HIP; a host-twin that fails here is a BASE problem, report it, do not "fix" the golden.
- **Compile gate:** HIP configure+build of the q3-touched targets (trap list below). First genuine compiler error = your blocker list; static scans over-claim (agent1's 4/6 lesson, STATE 13:3xZ).
- **Dequant finite-check** (theirs, per their plane law): full-tensor dequant, not per-chunk cosines. Per-chunk encode + concat yields GARBAGE AT CORRECT SIZE.
- Device cells: **none in this WO.** All GPU time comes later via coordinator-stamped slots.

## 3. File ownership (exclusive; do not cross)
- `src/HipSources.cmake` — append q3/q2 ops lines ONLY (whitelist growth). If you need to touch `src/CMakeLists.txt` outside adding files to lists, stop and ask (registered-exception zone).
- `src/ops/linear/q3/*`, `src/ops/linear_add/q3/*`, `q2/*` — HIP-compat edits allowed INSIDE new `#if defined(__HIP__)` lanes, CUDA text preserved in `#else`, zero reflow (RULING-1 pattern — see `git show 758b1322` for the exact style; PG-1 check (a) is being fixed by gemini to whitelist these files, interim: their expectation is registered exceptions).
- `src/ops/embed/embed_gather.cu` / `.cuh` — the Q3 arm (kEmbedGatherQ3*). Group-block decode from `q3_rowsplit_storage.h` `q3_decode_one` (host+device single source of truth — use it, do not re-derive the codec).
- **NOT yours:** `hip_shim/cuda_runtime.h` (agent2's), `math.cuh`/`memory.cuh` (RULING-1 exceptions — read their patterns, don't edit), `run_ci_amd.sh` + gates (gemini's), `targets/`/`serve/` (agent1's).

## 4. Environment traps (all measured on this box; each cost someone a cycle)
- `cmake`/`ctest` NOT on PATH: `/home/chris/opt/cmake/bin/cmake` (3.30.5 tarball). No system cmake, no pip cmake, no passwordless sudo.
- Configure: `-DNINFER_BACKEND=hip` (**flag name corrected 14:3xZ by agent3 — `NINFER_BUILD_BACKEND` silently defaults and make then hunts nvcc**) `-DCMAKE_HIP_ARCHITECTURES=gfx900 -DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0`. `BUILD_TESTING` is `set()` not `option()` (CMakeLists:78) → re-raises libcurl (absent dev pkg); clear the raisers, `-D` cannot override.
- `hipcc` parses a positional `.a` as source: link `-L build-hip-amd/src -lninfer_hip_host`.
- Toolchain: `__bfloat16_as_ushort` is a NUMERIC cast on ROCm (CUDA reinterprets) — use `__hip_bfloat16_raw`/memcpy. bf16 ARITHMETIC is compiler-rejected on gfx900. Q3 scales are pure **fp16** bits (`__ushort_as_half` semantics OK) — the FNUZ trap cannot bite this path (their word, verify cheaply).

## 5. Design decisions (FINAL)
- Q3 tier reality (verified against main's bytes): served v1a = **zero Q2**, roles = {down, attn-out, gdn-out, embedding}×layers at Q3G64; **embedding is Q3-GATHER, not GEMV** — the GEMV pair alone cannot load the 1.26→0.47 GiB embedding. Do not build a Q2 arm beyond keeping their fail-loud T>1 guard compiling.
- Pack: G=64, 24 code bytes + 1 fp16 scale = 3.25 bpw; 3-bit two's-complement, 8 vals per 3 bytes, `(v24>>(w*3))&7`, `(v^4)-4`, ×float(scale). Quantize: scale=max_abs/3.5, clamp [−4,3].
- REJECTED: porting the grouped-MMA extension (their own comment says perf-pass later); TP4/4-die work (engine supports 2-rank or single-device only; HORIZON per docs/59); any device-time detour (no grant without coordinator).

## 6. Execution order (commit + test each step)
0. **FIRST COMMIT — guard-patch, coordinator-ruled (measured defect, 13:3xZ):** `src/ops/linear/q3/q3_rowsplit_storage.h:23` and `src/ops/linear/q2/q2_rowsplit_storage.h:25` neutralise `__host__/__device__/__forceinline__` under `#if !defined(__CUDACC__)`; hipcc defines `__HIPCC__`+`__clang__` and NOT `__CUDACC__`, so the empty defines fire during the device pass and neuter ROCm's own `__device__` intrinsics (20 errors in `amd_hip_bf16.h` — `__ocml_fma_f32`, fmax/fmin/ceil/cos/exp*/exp2 — for any TU including `embed_gather.cuh`). Reproduced + root-caused by agent2 in G-AMD-13 preflight; q2 instance found by coordinator byte-verification. Patch: `#if !defined(__CUDACC__) && !defined(__HIPCC__)` in BOTH files, one commit, message cites `docs/CROSSLANE.md` AMD-log defect row (registered exception pending gemini's PG-1 exclusion-list update — name both files in the commit body so the exclusion is copy-pasteable). Proof-of-fix cell (CPU): `hipcc -x hip --offload-arch=gfx900 -fsyntax-only $(grep -rl "q2*_rowsplit_storage\|q3_rowsplit_storage" src/ops --include=*.cu)` → exit 0 — paste it.
1. Base verify (above) → branch `amd/wo-q3hip`. 2. CPU: run their q3 host-twin goldens on this box unmodified → commit log artifact. 3. Whitelist q3/q2 files → compile gate (CPU-side; first real error pasted). 4. HIP-compat lanes per RULING-1 pattern until compile green. 5. embed_gather Q3 arm + its CPU golden. 6. Dequant finite-check on a synthetic tensor + full-tensor roundtrip vs the fp64 oracle bar (cos ≥0.999/role). Each step: commit before reporting.

## 7. Definition of done
q3+q2 ops files and embed_gather Q3 arm in `HipSources.cmake`, HIP build green (log pasted), their CPU goldens green on this box (log), embed_gather Q3 CPU golden green, dequant finite-check green, all on `amd/wo-q3hip` pushed, and a handoff section naming what the next lane (WO-03's serve bring-up continuation / WO-05 attention work) can trust vs must re-derive. Report to coordinator with the parity line + sha1.
