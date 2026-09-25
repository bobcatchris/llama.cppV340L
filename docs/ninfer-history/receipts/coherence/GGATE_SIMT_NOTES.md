# GGATE_SIMT_NOTES — FIX-B: GDN gating control projection OFF the emulated-MMA path (P2 owner #2)

**Desk:** CODE-desk, `amd/tp4-cure`, 2026-09-17. Compile-only (NO GPU runs; server on :8100
undisturbed). Scope: `src/ops/gdn_gating_proj/bf16/` + one host cell. NOT touched (per
staffing): `src/ops/linear_attention/gated_delta_net/` (FIX-A), `gqa_attention_prefill*`
(FIX-C), existing coherence rows.

**Target (measured, PREFILL_BODY_row.txt / PLOG-048, bin 74219298c4f3d4a4):** the `gate`
class (rmsnorm + a/b control projection + gating) owns **78.1-82.1 ms/chunk** at T=128 TP4
(48 GDN layers x ~1.67 ms/call). Route at measurement: k27Routes cols{9,inf} ->
`MmaUnsplit` = `bf16_gdn_gating_proj_gemm_mma.cuh` = software ldmatrix_x2/x4 + per-lane
shuffle-broadcast MMA emulation on gfx900, dispatched at grid (1,3,1) = **3 CTAs on the
die** for 0.127 GFLOP/call (6.1 GFLOP/chunk) — the S2 pathology exactly as spec'd in
`docs/amd/PREFILL_BODY_2026-09-17.md` §4.

## 1. Route change (commits 768956ba4, 7f937d635)

New schedule `SimtColTile` (`gdn_gating_proj.bf16.simt.col_tile`), selected **only** by the
env gate at the prefill arm of `bf16_gdn_gating_resolve_plan`:

- Kernel `bf16_gdn_gating_proj_simt_tile_kernel<4,8>` (in
  `bf16_gdn_gating_proj_kernels.cu`): gfx900-native **plain fp32 FMA** — no `mma.cuh`, no
  ldmatrix emulation, no cp_async, no dynamic smem, no cooperative launch, no smem
  attribute raises (the whole G-AMD-21 launch-safety surface is structurally absent).
- Shape: one 32-thread CTA per (4 logical rows x 8 token cols); per lane the k loop walks
  vec8 (uint4) chunks `lane, lane+32, ...` (coalesced; k-ascending within each lane's
  stride set), 8 fused FMAs per (row, col, vec) into 32 fp32 accumulators, then each
  accumulator closes with the 32-lane `warp_reduce_sum` shfl_down butterfly (offsets
  16,8,4,2,1). Epilogue identical in form to the incumbent GEMV arms:
  `g = -exp(A_log[row]) * softplus(acc + dt_bias[row])`, `beta = sigmoid(acc)`, out layout
  `[t, 48]` token-major (same as GemvPairedRows/SmallTGemv).
- Grid `(96/4, ceil(T/8), 1)` = **(24, 16) = 384 CTAs at T=128** (~6.9 CTAs/CU on the
  56-CU device; fills 56-64 CU dies; vs MmaUnsplit's 3 CTAs = ~127x more CTAs).
- **Reduction-order change (documented in `bf16_gdn_gating_proj_kernels.h` on the kernel
  declaration): NOT bit-equal to either incumbent** — GemvPairedRows/SmallTGemv use the
  256-thread block-reduce (decode's bit-stable contract, unchanged and still serving
  cols<=8), MmaUnsplit used tile-MMA order. Parity gate is therefore **rel-L2 < 1e-2**
  (fp32 accumulation-order class; measured ~1e-7-1e-6 everywhere it has been measured —
  see the cell below). MmaUnsplit remains the default route and is untouched.

## 2. Gate: NINFER_GGATE_SIMT (value-parsed, default OFF byte-identical)

Predicate lives in the **dependency-free** `bf16_gdn_gating_simt_gate.h` (tp_argmax_routing
cell pattern: the plan funnel AND the zero-card cell call the same function — a gate
regression is a red cell, not a red narrative). Substitution point:
`bf16_gdn_gating_resolve_plan`, is_27 branch, `ggate_substitute_simt(cols, arm==MmaUnsplit)`.

| NINFER_GGATE_SIMT | effect | trace |
|---|---|---|
| unset (default) | OFF — resolve output byte-identical to pre-FIX-B (MmaUnsplit at cols>=9) | **silent** |
| `1` | ON — cols>=9 (27 model) resolves SimtColTile | one stderr line |
| `0` or anything else | OFF (strict parse: exact `"1"` only, not truthy) | one stderr line |

`[GGATE]` first-seen trace (exactly once per process, thread-safe magic static, only when
the var is PRESENT — unset never prints anything):

```
[GGATE] NINFER_GGATE_SIMT='1' -> route=gdn_gating_proj.bf16.simt.col_tile (bf16 27-model prefill arm, cols>=9; decode cols==1 and smallT cols 2..8 unchanged)
```

**Decode path untouched by construction:** cols==1 (GemvPairedRows) and cols 2..8
(SmallTGemv / SmallTSplit10) never substitute, gate ON or OFF; `candidate_is_legal` bounds
SimtColTile to `cols >= 9` = exactly the arm it replaces. The 35-model catalog is untouched
(SimtColTile illegal off the 27 geometry). Cross-check handle at runtime: with
`NINFER_GATING_TRACE` set, the plan funnel prints `[gating] schedule=...` — gated boots
must show `gdn_gating_proj.bf16.simt.col_tile` at T=128 and the old names at T<=8.

Workspace: SimtColTile needs 0 bytes (split_k=1), same as MmaUnsplit — capacity scan and
arena sizing unchanged in both gate states (verified: `route_capacity` still bounds by the
SmallTSplit10 arm at small cols).

## 3. Host-only unit cell (zero-card, ALL GREEN)

`tests/ops/gdn_gating_simt_route_host.cpp` (CMake target `ninfer_gdn_gating_simt_route_host`
for the CI farm; build-hip-amd keeps BUILD_TESTING=OFF, so the desk receipt is the house
direct-compile form, per the WO-TP4-E cell precedent).

- **Gate matrix, both directions:** production predicate over env {unset, `0`, `1`, `yes`}
  x cols {1,4,8,9,16,128}: decode/smallT stay, prefill arm flips only at (ON, cols>=9),
  non-MMA arms never flip at any cols. 4/4 ok.
- **Paired-fire falsifiers (cell must be able to go RED):** always-substitute poison goes
  RED (would flip decode cols=1); never-substitute poison — **the exact pre-FIX-B
  behavior — goes RED (prefill arm stuck)**. This is the RED capture for the gate: on the
  pre-fix artifact the resolve output at (env would-be-ON, cols=128) is MmaUnsplit, which
  is precisely the never-substitute arm this cell rejects.
- **Route-order parity, old-vs-new, both directions** (rel-L2 < 1e-2, **NOT bit-equal** —
  stated; the order change is the documented delta): order-faithful AND op-faithful
  (std::fma = IEEE fused; exact bf16->f32 converts; butterfly adds in fp32; epilogue same
  formula both sides) host emulations of the OLD GEMV house order (GemvPairedRows/
  SmallTGemv contract; MmaUnsplit's GPU-side parity vs this contract is owned by the
  standing ctest sweep bound kGdnProjectionFp32 rel 3e-6) vs the NEW SimtColTile
  stride+butterfly order, at the cell-contract reduced shape **T=16, rows=8 (heads=4),
  K=512** (2 stride iterations/lane — the stride is exercised):

```
rel-L2 g    new->old 7.065e-08 | old->new 7.065e-08
rel-L2 beta new->old 8.253e-08 | old->new 8.253e-08
rel-L2 g    new->oracle 8.790e-08 | old->oracle 7.815e-08     (fp64 sequential oracle)
rel-L2 beta new->oracle 5.666e-08 | old->oracle 6.032e-08
```
  All 4 orders of magnitude inside the 1e-2 bar. Determinism pin: new-order emulation
  rerun bit-equal. **Verdict: ALL GREEN, exit 0.**
- Receipt (this desk, zero GPU):
  `g++ -std=c++20 -O2 -Wall -Wextra -I src -I include tests/ops/gdn_gating_simt_route_host.cpp -o <bin> && <bin>`
  → compile RC=0 **zero warnings**, run exit 0, full output above (banked in this file).
- What the cell does NOT cover (honest boundary): device-kernel execution parity and the
  perf number — those are the GPU owner's probe below (parity gate vs un-gated boot).

## 4. Compile receipts

`make -C build-hip-amd -j20 ninfer_hip_host ninfer-serve` → **RC=0**, 0 errors.
Warning audit (both gating TUs rebuilt clean + full link): every warning is a pre-existing
class (unused-result on the HIP shim + FIX-C's gqa_attention_prefill cuda-compat);
**zero warnings on added lines**. One initial defect found and fixed during bring-up:
`-Wpass-failed` unroll remark on the new kernel's epilogue loop (unroll pragma + break) —
removed the provably-dead row guard (grid x tiles the 96 logical rows exactly,
static_assert'd); 0 pass-failed in the final tree. Lane serve binary in the build tree at
commit time: `build-hip-amd/apps/ninfer-serve` sha16 `ed1c888613c4ec6d` (lane build path —
per law it is a moving target: BANK it before any relink, boot from the bank).

## 5. GPU-owner runbook (next card window; NO GPU work was done at this desk)

Predicted saving: `gate` column **78-82 ms/chunk -> 2-6 ms** (conservative acceptance
ceiling < 12 ms), i.e. **~70-80 ms/chunk off the 677-698 ms body** — 0.127 GFLOP/call
moves from ~0.2%-of-peak emulation to the fp32 FMA pipes at a sane fraction of the
10.75 TF/s ceiling (even 10% of peak = ~120 µs/call = ~5.7 ms/chunk; 384 CTAs is ~6.9
waves of occupancy, L2-resident operands: weights 0.94 MB + x 1.25 MB fit the die's L2).

1. **Rebuild** in the lane tree: `make -C build-hip-amd -j20 ninfer_hip_host ninfer-serve`
   (expect RC=0; warning classes unchanged). No GPU needed for this step.
2. **BANK-BEFORE-RELINK** (BOOT_LAUNCH_RUNBOOK §4): `cp -p` the freshly linked
   `ninfer-serve` to `/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin` (filename-is-
   stamp) and commit the RECORD row (path, full sha, recipe, date). Pinned binaries boot
   from the bank, never from the lane build path.
3. **Boot battery** per runbook (grant + gpu_guard discipline; BOOT_BATTERY.sh rides the
   window, banks `BOOT_BATTERY_<ts>.row` + clocks sideband — thermal law: no row without
   its sideband compares to anything).
4. **Gated probe:** with `NINFER_GGATE_SIMT=1 NINFER_PREFILL_OPTRACE=2` on the known-good
   nvfp4 line, run the plen-1996 conc=1 first-request probe (same grammar as P1/P2;
   cool window + clocks sideband first). Expect:
   - serve log: exactly one `[GGATE] ... route=gdn_gating_proj.bf16.simt.col_tile` line
     per rank process (first-seen), present before first prefill chunk;
   - `[PREFILL-BODY]`/`[PREFILL-BODY-SUM]`: **gate 80 -> <12 ms** (predicted band 2-6);
     `dn` and `gqa` columns unchanged vs the un-gated reference within noise (their routes
     untouched); body sum drops by ~the gate delta; wall improves by the same order on the
     prefill-class tok/s readout (P2 anchor ~70.7 tok/s);
   - optional cross-check: `NINFER_GATING_TRACE=1` shows `schedule=gdn_gating_proj.bf16.simt.col_tile`
     at the T=128 calls and the incumbent names at T<=8.
5. **Parity gate vs un-gated boot:** same probe un-gated (env unset) on the same banked
   binary family — request finish/type/content sane, and the standing coherence/text
   fingerprint gates apply to the gated run. The gated-vs-un-gated outputs are NOT
   expected bit-identical (documented fp32 order change, ~1e-7 rel-L2 class per projection;
   model-level drift must stay inside the existing fingerprint/text gates — that IS the
   acceptance bar, alongside the ctest sweep bound if a GPU ctest run is in scope:
   kGdnProjectionFp32 rel 3e-6 covers MmaUnsplit; the SimtColTile arm joins the sweep the
   same way its numbers were cell-cleared at 1e-2 with ~1e-7 measured).
6. **Bank** the row (this file's sibling naming: suggest `GGATE_SIMT_row.txt`) + serve log
   + clocks sideband, `tools/guards/plog_append.py` for the PLOG entry, and flip the
   default ONLY after the row is green (route stays env-gated until then; making
   SimtColTile the default at cols>=9 is a one-line catalog change that must carry its own
   row + this cell's green).

## 6. Files

- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_simt_gate.h` (new, dependency-free gate predicate + [GGATE] trace)
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h` (enum + comment)
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp` (name/uses_mma/split_k/legality switches, substitution, dispatch case)
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h` (decl + order-change/tolerance doc block)
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` (kernel + launcher)
- `tests/ops/gdn_gating_simt_route_host.cpp` (new, zero-card cell) + `tests/CMakeLists.txt` (target)

Commits: **768956ba4** (route + gate), **7f937d635** (cell + receipts). Base: 342f3861c
line (P2 BODY SPLIT, PLOG-048).
