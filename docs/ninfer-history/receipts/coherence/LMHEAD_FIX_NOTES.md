# LMHEAD_FIX_NOTES — the TAIL lm_head route flip + roofline A/B/C (CODE desk, 2026-09-17, amd/tp4-cure)

Implements the fix spec in `docs/amd/TAIL_LMHEAD_2026-09-17.md` (commit `fb86a0fdd`): the verify
lm_head — full-vocab ColumnN TP4 shard `[62080, 5120]`, W8G32_F16S, 337,715,200 B/rank — measured
**61.93 ms/round (5.45 GB/s effective)** at T=3 because `select_w8_a16_launch` routes the TP4-vocab
bucket to `launch_w8_small_t`, whose 5090 tensor-core schedule runs as emulated SIMT `mma_bf16` on
gfx900. The sibling arm `launch_w8_simt_r8_c4` streams the SAME format at ~174 GB/s serving-side.
Mission: bench-gated, env-gated route flip + the bench that prices it. NO GPU was touched by this
desk (server PID 310132 on :8100 is not ours; zero launches executed).

## What landed

1. **Route flip — `src/ops/linear/w8/w8_dispatch.cpp`** (the seam named in the spec §3-A).
   TWO application sites, ONE gate (coordinator scope extension after the D2 verdict,
   `DRAIN_FIX_NOTES.md` commit `981386566`):
   - `case 62080` (k=5120) — the TP4 LM-head shard: `NINFER_LMHEAD_ARM=simt` AND `t <= 4` →
     `launch_w8_simt_r8_c4`; everything else (default unset, `=smallt`, t>4) → **byte-identical
     current route** (small_t at t<=33, then the registered mma ladder exactly as before).
   - `case 10240` (n=5120) — the MTP draft-chain **fc** stem projection `[5120, 10240]`: the ONLY
     draft T=1 forward that rides THIS seam; same gate, same t<=4 scope, same flip arm.
   - The family-10 determinism cure is preserved: the simt arm is now NAMED, bench-gated traffic,
     not a generic-tail fallback. The route change is **PARITY-GATED, not byte-identical** (the
     arms differ in accumulation order): the serving gate is the count600 fingerprint
     (204 rounds / acc 0.97 / t-r 2.94) + identical greedy text, run by the GPU owner below.

   **Seam topology of the draft chain (verified in code, window-3 expectation input):** the
   mtp_forward T=1 round has FIVE W8 linears, and they do NOT all dispatch through
   `select_w8_a16_launch`:
   - `mtp_.fc` [5120,10240] — `ops::linear` → **THIS seam** (was small_t at T=1; flips under the
     gate) — `text_context_impl.h` mtp_forward_stem (:1295).
   - attention proj (packed qkv+gate) [14336,5120], o_proj [5120,6144], gate_up [34816,5120],
     down [5120,17408] — via `multi_gpu::tp_gemv` (`variant_kernels.cpp:504+`, tp_kernel.cu:138)
     → **SIBLING seam**, and that seam ALREADY routes W8G32 t<=4 to `launch_w8_simt_r8_c4`
     (tp_kernel.cu:139-141). Nothing to flip there — the flip arm is already the served arm.
   Consequence for window 3: the count600 A/B expects `lm_head` to collapse (61.93 → 2-4 ms) and
   `align`/`chain_fwd` to move ONLY BY THE FC SHARE (one of five linears, ~56 of ~450 MB per
   draft forward). The bulk of the D2 16.5 ms draft compute already runs on simt_r8_c4 — its
   ~54-56 GB/s serving rate at the [14336..34816,5120] geometries (3x below the ~174 GB/s
   [10240,5120] propose-anchor class) is the NEXT lever (retune/schedule class), NOT this route
   flip. The bench's DRAFT-FC leg prices the fc A/B directly; a future tp_gemv-arm bench (T=1
   rows at the four sibling geometries) is the named follow-up for the remainder.
2. **Env gates, N7 house pattern** (value-parsed, never presence-parsed; `tp2_backend create()`
   precedent):
   - `NINFER_LMHEAD_ARM` = `simt` (or `simt_r8_c4`) | `smallt` (or `small_t`) | unset = default
     smallt. Set-but-unrecognized (incl. `=0`, `=off`, `=""`) **refuses loud** naming the value
     and the accepted set. Read once per process (magic static).
   - `NINFER_LMHEAD_TRACE=1` gates the `[LMHEAD]` label (mirrors `[SMALLT]`/`[TILED]` grammar):
     one stderr/stdout line per (n, k, T) first-seen — `[LMHEAD] arm=<kernel> n=62080 k=5120 T=3
     gate=simt|smallt (launch #N)` — plus a `[LMHEAD] total launches #N` line every 16384 launches
     so a count600 run can prove WHICH arm served every round. Zero cost when unset (single bool
     test per launch).
3. **Bench — `results/amd/coherence/w8_roofline_bench.cu`** (W8 sibling of
   `nvfp4_roofline_bench.cu` / `nvfp4_smallt_roofline_bench.cu`, same house discipline: -O3 LAW
   compile, 5 s continuous mclk hammer before timing, best-effort sysfs `pp_dpm_mclk` print,
   hipEvent timing, GB/s + %read-ceiling). Arms called DIRECTLY (timing independent of the gate):
   - **A** `launch_w8_small_t` (current serving arm; RED reproduction leg)
   - **B** `launch_w8_simt_r8_c4` (flip arm)
   - **C** `launch_w8_mma_r64x16_c48_k128_a1` (registered mma arm, legal t<=48 — gfx900 executes
     it as emulated-SIMT mma too; priced for completeness, expected not to win. The SERVE build
     keeps this symbol on the die-loud HIP link stub; the bench links the REAL TU to price the
     schedule itself.)
   - EXACT geometry n=62080, k=5120, T in {3, 1, 4} (T=3 primary), W8G32_F16S byte convention
     (codes n·k B + f16 scales n·(k/32)·2 B + x + out), fixed-seed splitmix64 data with a
     rel-L2(A,B) / rel-L2(A,C) arm-agreement leg (bar 1e-2, statistical — NOT the serving gate).
   - plus: a DRAFT-FC leg (see the seam topology above — prices the fc [5120,10240] T=1 A/B,
     the seam-member share of the D2 draft compute), a draft-slice anchor row `[10240, 5120]`
     T=3 via simt_r8_c4 (the geometry the ~174 GB/s propose-field number rode), and a SELECTOR
     leg that drives the REAL `w8_dispatch.cpp` `select_w8_a16_launch` for the lm_head problem
     (62080,5120) at T in {1,3,4,33,40} AND the fc problem (5120,10240) at T in {1,3,4} under
     the process's `NINFER_LMHEAD_ARM` — the flip is proven by RE-RUNNING the binary with
     `=simt` (the gate is a process-lifetime magic static).

## Compile verification (this desk, no execution)

```
# server (route flip): RC=0
/home/chris/opt/cmake/bin/cmake --build /home/chris/worktrees/amd-tp4-cure/build-hip-amd \
  --target ninfer_hip_host -j8
#   -> [100%] Built target ninfer_hip_host; v340l whitelist parity OK: 220/220 objects
/home/chris/opt/cmake/bin/cmake --build /home/chris/worktrees/amd-tp4-cure/build-hip-amd \
  --target ninfer-serve -j8
#   -> [100%] Built target ninfer-serve  (link RC=0)
```

Bench compile (the -O3 LAW line, house-identical to the NVFP4 benches; from the worktree root;
device/tensor/dtype close the core symbols the arm TUs' launch_route token-slicing needs):

```
/opt/rocm/llvm/bin/clang++ -O3 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 \
  -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -Isrc/common/hip_shim -Iinclude -Isrc \
  -Ithird_party -DNDEBUG -std=gnu++20 --offload-arch=gfx900 -x hip \
  results/amd/coherence/w8_roofline_bench.cu src/core/device.cu src/core/tensor.cpp \
  src/core/dtype.cpp -o /tmp/w8_roofline_bench_cure
```

(Note: `clang++` = `/opt/rocm/llvm/bin/clang++`, ROCm 6.2 AMD clang 18; `cmake` is not on PATH in
this env — the lane's is `/home/chris/opt/cmake/bin/cmake`.)

**This desk's compile-verify evidence (2026-09-17):** server rebuild RC=0 (whitelist parity guard
220/220, ninfer-serve relink RC=0, includes the fc-cell extension); bench compile RC=0, 0 errors,
72 warnings (the benign hip-shim nodiscard / -Wpass-failed unroll class, house tolerance) — bench
binary sha256 `ba8ae5a99eab65e0896b5d2bbd3f2de991835bd62443fbd36a0f8b5d84907413`
(/tmp/w8_roofline_bench_cure, NOT executed — no GPU from this desk; recompile per the law line in
the GPU window if the tree moved).

## Bench run command (GPU window; zero-serve; ~700 MB device alloc; no port claimed)

```
cd <worktree root>
/tmp/w8_roofline_bench_cure 2>&1 | tee results/amd/coherence/W8_LMHEAD_roofline_run_<ts>.log
NINFER_LMHEAD_ARM=simt /tmp/w8_roofline_bench_cure 2>&1 | tee -a results/amd/coherence/W8_LMHEAD_roofline_run_<ts>.log
```

RED/GREEN reading (spec §4.1): RED reproduction = arm A at ms-equivalent ≥ 10 at serving clocks
(confirms the 61.93 ms serving field is the kernel, not the box); GREEN = the winner arm's
measured GB/s names the route. Expected per the budget table: A ~5 GB/s class (tens of ms),
B at 90-170+ GB/s class (1.5-4 ms), C not expected to win. Selector leg default run must print
`lm_head (62080,5120) T=1/3/4 -> launch_w8_small_t` and `draft-fc (5120,10240) T=1/3/4 ->
launch_w8_small_t`; the `=simt` rerun must print `launch_w8_simt_r8_c4` for BOTH problems at
T=1/3/4 (T=33/40 stay small_t / lane-fallback in both — the flip is scoped to verify widths).

## GPU-owner ONE-WINDOW runbook (bank every leg; M0 clocks sideband law applies to all legs)

1. **Rebuild in the LANE worktree only** (never the shared checkout):
   `/home/chris/opt/cmake/bin/cmake --build /home/chris/worktrees/amd-tp4-cure/build-hip-amd --target ninfer-serve -j$(nproc)`
2. **Bank before relink** (docs/amd/BOOT_LAUNCH_RUNBOOK.md §4) →
   `/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin`; boot THE BANKED PATH (filename = stamp).
3. Pre-boot strings check: `strings <banked.bin> | grep -E 'NINFER_LMHEAD_ARM|NINFER_VERIFY_TAIL_TRACE'`
   must hit BOTH.
4. **BOOT_BATTERY GREEN** pre-leg (per-window law): `tools/guards/BOOT_BATTERY.sh <port>`; bank
   row + clocks log.
5. **Roofline A/B/C bench** (zero-serve, minutes — the decisive arm-ranking; compile line above;
   run both gate settings, tee the log into `results/amd/coherence/`). The flip is legal ONLY if
   B beats A decisively here AND the serving leg (step 6) holds the fingerprint.
6. **Route-flip serve leg**: boot posture = serve_10k.sh line MINUS the 10k body, PLUS
   `NINFER_LMHEAD_ARM=simt NINFER_LMHEAD_TRACE=1` and the M2 trace env
   (`NINFER_VERIFY_TAIL_TRACE=1 NINFER_TP2_TIMING=1 NINFER_TP2_OPTRACE=1 NINFER_VERIFY_LAYER_TRACE=1`,
   `NINFER_ALLOW_NVFP4_TP2=1`, `NINFER_WORKSPACE_MIB=96`, `NINFER_DRAFT_VOCAB=<draft_vocab_ids.json>`,
   TP4 `--devices 0,1,2,3 --spec mtp --draft-tokens 2 --prefill-chunk 128 --no-prefix-reuse
   --greedy`). Verify in the log: "loaded 40960 draft vocabulary IDs" AND the `[LMHEAD]`
   first-seen lines naming `arm=launch_w8_simt_r8_c4 gate=simt` for BOTH application sites —
   `(62080, 5120)` (lm_head) and `(5120, 10240)` (draft fc).
7. **Workload:** count600 conc=1 temp=0 greedy (M1/M2 family: 67-tok prompt, mt=600), warmup +
   one clean leg; **clean-band treatment mandatory** (today's box instant-droops under decode
   bursts — sclk 7→2-5 within 2-6 s; rankings are thermal-robust, absolute ms are not).
   Expected: `lm_head` **≤ 4 ms** (RED bar ≥ 10), `lgather` ≤ ~1, verify ~66-70, round ~85-90
   at M1-class clocks (conservative floor: lm_head 6.0 ms even at the 56 GB/s MTP-layer class).
   `align`/`chain_fwd` move ONLY BY THE FC SHARE (seam topology above — the other four draft
   linears already ride simt_r8_c4 via tp_gemv); their residual is the named follow-up lever.
   BEFORE-arm anchors for the same boot posture with env unset (optional second boot): lm_head
   ~62, lgather ~4.9, verify ~117, round ~134.
8. **Fingerprint guard (THE gate):** 204 rounds / acc 0.97 / t-r 2.94 must hold on the flipped
   arm (M1/M2 anchor family: 201-204 / 0.97-0.99 / 2.94-2.99). The arms differ in accumulation
   order — a small acc/t-r move is the failure mode this step exists to catch; if the fingerprint
   holds but the greedy completion TEXT diverges from the M2-banked count600 completion, record
   loud and escalate before promoting the default.
9. **Greedy known-answer + restore:** fixed-prompt greedy completion vs the banked anchor
   (determinism continuity), then restore the standing serve_10k.sh posture (env unset ⇒ smallt;
   relink/default boots stay byte-identical until a promotion order flips the default).
10. **Bank** `W8_LMHEAD_row.txt` (bin sha, both logs, [TAIL]/[TAIL-AVG]/[TAIL-SUM] lines, the
    `[LMHEAD]` first-seen + volume lines as the route proof, clocks sideband) + PLOG.

## Env matrix

| env | value | behavior |
|---|---|---|
| `NINFER_LMHEAD_ARM` | unset / `smallt` / `small_t` | byte-identical current route (small_t) |
| | `simt` / `simt_r8_c4` | verify widths t<=4 route to `launch_w8_simt_r8_c4` |
| | anything else | serve refuses loud (N7 value-parsed rule) |
| `NINFER_LMHEAD_TRACE` | `1` | `[LMHEAD]` first-seen + volume lines (route proof) |
| | unset | zero cost |
| `NINFER_VERIFY_TAIL_TRACE` | `1` | existing M2 `[TAIL]` per-op ms (lm_head field) — unchanged |

## Honesty block

- This desk touched ONLY: `src/ops/linear/w8/w8_dispatch.cpp`,
  `results/amd/coherence/w8_roofline_bench.cu`, this notes file. The GPU window's uncommitted
  results/logs in the worktree are not ours and were not staged.
- Predicted post-fix numbers are from the spec §3 table (weight 337.7 MB at 90-170 GB/s SIMT-GEMV
  class, halved for the 6x row count and T=3 surcharge; the ~174 GB/s propose anchor includes
  argmax+remap, order-of-magnitude only). The bench prices the arms; the serving leg prices the
  round. Nothing is merged on the doc's numbers — only on steps 5-8.
- The C arm note: on the HIP lane the SERVE dispatch can never reach the real mma TU
  (`NINFER_NVFP4_SIMT_LANE=1` collapses W8_SEL_MMA to the simt fallback, and the link stub dies
  loud); the bench links the real TU purely to price the schedule per spec §3-D.
- The selector leg's T=33/40 rows intentionally exercise the unchanged t>4 ladder (small_t /
  W8_SEL_MMA lane fallback) to prove the flip is scoped to verify widths only.
