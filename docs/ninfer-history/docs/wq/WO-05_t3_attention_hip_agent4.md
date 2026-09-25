# WO-05 — T3 attention under HIP, plain bf16-KV route (agent 4) — issued 2026-09-12 ~13:36Z, coordinator C441

**Admission bar:** this is the numerics lane — take it only if you are strong on fp16/fp32 kernel work. A weak pass here creates rework, not speed. Base: branch from `amd/main` (name, not SHA); verify `src/common/hip_shim/cuda_runtime.h` contains the string `unconditional: all lanes publish` (the merged shuffle fix — without it every cross-lane reduce is silently ~√2 wrong and your validation would be measuring a bug).

## 1. Goal
The attention op family (T3) compiles and is device-verified on gfx900 for the **plain bf16-KV route** (int8-KV is T3b, deferred, separate WO). This is the long pole between "artifact loads" and "server answers".

## 2. Test entry points
- **Phase gate pipeline is gemini's (`run_ci_amd.sh`, PG stages, docs/amd/v340l/01).** Do not author gates/tests — run theirs. New device cells require a coordinator-written grant; none exists at issue.
- **Decisive CPU reference FIRST:** every kernel claim gets an fp64 CPU oracle comparison (their bar: cos ≥0.999/role) before any GPU slot is requested. Hypothesis without a cheap decisive check is a hope (§5 rule).
- Golden route: single-device, tiny grids first (d=64/128/256 fast domain + d=66 generic — the pattern that cracked l2norm). Deliverable is an index→value table, never a bare PASS.
- Known trap this lane will hit: gates on this line must test **ISA-level behavior** (`ds_bpermute` under conditional execution) not call-site semantics — multi-step butterflies are invisible to single-step enumeration. Any new reduce you write: forward `width`, never branch around a wavefull instruction (see docs/amd/AGENT2_WO02_HANDOFF.md — read it first, it's the mechanism write-up).
- **Predecessor landmine (yours to sweep, 13:4xZ finding):** the promoted `q3_rowsplit_storage.h:24` / `q2_rowsplit_storage.h:26` (line-cites corrected post-merge by agent5 verifier + re-verified at amd/main bytes this session) neutralise `__host__/__device__` under `#if !defined(__CUDACC__)` — hipcc takes that branch and it poisons ROCm's `__device__` intrinsics tree-wide through any include chain that reaches it. agent3 patches both headers as WO-04 step 0; if your include chain breaks on 20 `amd_hip_bf16.h` errors, check that landed before debugging anything of yours.

## 3. File ownership (exclusive)
- `src/ops/attention/*` (T3 set per `results/v340l/first_token_files.md` whitelist classes), new HIP-lane kernels you add under the same tree.
- **READ-ONLY for you:** `math.cuh`/`memory.cuh` — they carry the RULING-1 registered exceptions (`exp2_approx` = APPROXIMATION-CLASS via `exp2f`, not bit-exact; `pack_bf16x2` = bit-exact RNE). If attention needs a helper that must change there: STOP, file the exact ask to coordinator — that file pair is a registered-exception surface, edits need a ruling not convenience. D3 numerics: route through fp16/fp32 — but **enforcement correction (14:3xZ): bf16 arithmetic is NOT compiler-rejected on gfx900; it compiles and silently emulates in fp32 (~6 instr/op — agent2 `v340l/03(b)`, coord-reproduced). Self-enforce D3; a passing build is not evidence of bf16 compliance; gemini's ISA-fingerprint CI cell is the gate.**
- NOT yours: shim (agent2), q3/q2/embed_gather + the guard-patch (agent3, WO-04), serve/targets (agent1), CI/gates/oracle files (gemini).

## 4. Measured facts to build on (cite, do not re-derive)
7.98 GiB/die, 56 CUs, wavefront 64, ROCm 6.2.0-66; no P2P at all (host-staged cross-card 6.61–6.70 GB/s); intra-device ~183 GB/s ceiling — any % of roofline names its denominator; dev→card NOT identity (dev0→card1, dev1→card3, dev2→card0, dev3→card4); shape convention `{d, rows}`, `ne[0]`=feature dim. Build traps: cmake at `/home/chris/opt/cmake/bin`; `-DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0`; hipcc: `-L … -l…` never positional `.a`.

## 5. Design decisions (FINAL)
- bf16-KV plain route first; KVarN/int8-KV variants explicitly out. Single-device correctness is the milestone; TP2 attention sharding rides WO-03's AR design, not yours.
- REJECTED (don't re-litigate): porting mma.sync/ldmatrix/cp.async families to MFMA (the prefill-MMA mapping risk Team Green was warned about — the SIMT portable shape sidesteps it by design; attention goes the same way until someone measures a reason); env-gated printf inside product headers (instrument in test-local TUs, the exception-pair rule).

## 6. Execution order
1. Base verify + read `docs/amd/AGENT2_WO02_HANDOFF.md` + `docs/amd/v340l/00` §T3 scope. 2. Static: enumerate your kernel set's dependency surface vs the shim/exceptions — report the STOP-asks BEFORE writing kernels. 3. Compile-only pass in your worktree. 4. CPU-oracle tests per kernel (no GPU). 5. Request stamped device slot(s) — 90–600 s each, dev assignment by coordinator. 6. Commit per step; every report pastes bytes (compiler errors, golden tables), never claims.

## 7. Definition of done
T3 kernel set: HIP compile green + CPU-oracle green (tables in `results/amd/`) + one stamped device run green per kernel class, handoff section naming what T3b (int8-KV) inherits. Serve bring-up (agent1) is unblocked when this lands.
