# GDN_SIMT_NOTES — FIX-A: chunked delta-net trio OFF the emulated-MMA path (P2 owner #1)

**Desk:** CODE-desk, `amd/tp4-cure`, 2026-09-17 (resume of the usage-cap-killed FIX-A desk;
kernel rewrites secured as commit `c79c59e3b` "FIX-A WIP"). Compile-only + zero-card host
cell (NO GPU runs; server on :8100 undisturbed). Scope:
`src/ops/linear_attention/gated_delta_net/chunked/` + one host cell. NOT touched:
`bf16_gdn_gating_proj*` (FIX-B, landed), `gqa_attention*` (FIX-C, landed),
`text_context_impl.h`, existing coherence rows.

**Target (measured, PREFILL_BODY_row.txt / PLOG-048):** the `dn` class (chunked trio:
`launch_prepare_wy_wu` + `launch_state_passing` + `launch_output`, 48 GDN layers/chunk)
owns **501.6-513.7 ms/chunk = 73% of the ~690 ms prefill body** at T=128 TP4. Route at
measurement: CUDA-tensor-core-shaped kernels (`mma_bf16` / `ldmatrix` / `cp_async` via
`ops/common/mma.cuh`) — **every one a software emulation on gfx900** (no MFMA hardware
before gfx908) — on starved grids: prepare (NT=2, H_v=12) = **24 CTAs**, state_passing
12x4 = 48 CTAs (16 warps, serial 2-chunk loop), output (2,12) = **24 CTAs** on the
56-CU die (`docs/amd/PREFILL_BODY_2026-09-17.md` §4 S1).

## 1. What each kernel rewrite does (gfx900-native SIMT, identical algebra)

All three live in new `*_simt.cuh` headers; the mma path in the sibling `.cuh` files is
**untouched and remains the default**. Math re-derived from the recurrent ground truth
(`delta = beta*(v - alpha*(k.S)); S = alpha*S + delta*k`, `recurrent.cuh`), which fixes
the decay-then-read variant: `M[t][s] = -beta_t * exp(g_t - g_s) * (k_t.k_s)` (strict
lower), `T_inv = (I - M)^{-1} = sum_k M^k` (M nilpotent), `U = T_inv @ (beta . V)`,
`W = T_inv @ (beta*e^g . K)`. Plain fp32 FMA on bf16-converted operands; no `mma.cuh`,
no ldmatrix emulation, no cp_async, no dynamic smem.

- **`prepare_wy_wu_simt.cuh` (stage 1/3):** per (chunk, v-head, 32-col d-strip) CTA:
  cooperative K load to LDS (+2 row stride killing the 32-way bank conflict), warp-0
  Hillis-Steele inclusive g scan (strip 0 publishes `g_cumsum_out` — identical values,
  one deterministic store), strict-lower KKT, decayed M (select-not-mask: g_cumsum is
  monotone decreasing, upper triangle is large-positive exponent — no inf*0=NaN),
  16x16-block in-place unit-lower inversion + block-Schur completion (documented
  barrier/ordering constraints, both load-bearing), then U and W for the strip's 32
  columns in one pass.
- **`state_passing_simt.cuh` (stage 2/3):** per (v-head, 16-row d-strip) CTA walks the
  chunks serially: h_chunk snapshot (bf16, [d][k] rows), `v_new = U - W.h` stored
  UNDECAYED, `h = exp(g_C)*h + k^T.(v_new * exp(g_C - g_t))`, next-chunk W/K/g prefetch;
  typed state I/O (fp16/bf16/fp32 in/out via `if constexpr` — same dtype dispatch as the
  mma launcher). In-place `state_in == state_out` safe: each CTA reads its own (h_v,
  strip) cells once, up front.
- **`output_simt.cuh` (stage 3/3):** per (chunk, v-head, 16-token band) CTA: decayed
  causal `A[t][s] = dot(q_t, k_s) * exp(g_t - g_s)` (s<=t, upper cells WRITTEN 0),
  then `out = scale * (exp(g_t) * dot(q_t, h_chunk[d,:]) + sum_s A[t][s] * v_new[s][d])`
  over eight 16-wide d-panels. Each output cell produced by exactly one thread, one
  reduction tree.

**ORDER / PRECISION CHANGE (codec contract law, declared on each header):** dots
k-ascending fp32 FMA vs the mma path's 16-deep fragment trees (the HIP mma emulation is
full fp32 FMA — no tf32 truncation); decays uniformly `exp2_approx(x*kLog2E)` vs the
mma prepare's `expf` (C441 approximation class). **NOT bit-equal**; acceptance gate
rel-L2 < 1e-2 (measured: §4). bf16 rounding boundaries (W/U/v_new/h_chunk/attn_out
stores) identical to the mma path.

## 2. Grid + occupancy math (gfx900 / V340 die: 56 CUs, 64 KB LDS/CU, 10.75 TF/s F32)

| stage | simt grid @ TP4 (H_v=12, T=128, NT=2) | threads | static smem | CTAs/CU | CTAs | mma grid |
|---|---|---|---|---|---|---|
| prepare_wy_wu_simt | (2, 12, 4) d-strips | 256 (8 warps) | 34.3 KB | 1 (2x34.3 > 64) | **96 = 1.7 waves** | (2,12) = 24 |
| state_passing_simt | (12, 8) d-strips | 256 | 44.5 KB | 1 (2x44.5 > 64) | **96 = 1.7 waves** | (12,4) = 48, serial loop |
| output_simt | (2, 12, 4) 16-row bands | 256 | ~31.4 KB | 2 | **96 = 0.86 waves** | (2,12) = 24 |

Every stage re-grids to >= 96 CTAs over the 56 CUs (mma: max 24 CUs busy at prepare/
output). The prepare d-strip split recomputes KKT/T_inv 4x (+55 MFLOP/layer-call,
bounded ~0.1 ms/chunk at gate-level efficiency) but changes NO output value (each output
cell still owned by exactly one thread; deterministic — cell-verified). The output
T-band split has ZERO redundant compute (disjoint 16-row bands; A row built exactly
once; h_chunk/v_new panels re-staged per band from L2).

## 3. W-dedupe: the design sketch's "compute W once per k-head" is INVALID — declined with math, guarded by a cell mutation

`docs/amd/PREFILL_BODY_2026-09-17.md` §4 S1 said "Extra 3x: W is recomputed per v-head
while only k_loc=4 k-heads exist" and §6 sketched "dedupe: compute W once per k-head (4)
in workspace, share across its 3 v-heads (-2/3 W-side FLOPs)". **The premise is wrong
for this model and the dedupe would break the op:**

- `W[t][d] = sum_s T_inv[t][s] * (beta_s * e^{g_s}) * K[s][d]`. The decay factors and
  beta are **per-v-head quantities**: the a/b control projection is per-v-head
  (gating proj rows = 2 x value-heads; qwen3_6_27b: `gdn_key_heads=16` vs
  `gdn_value_heads=48` -> head_map 12 local v-heads over 4 local k-heads, group 3), and
  BOTH kernels index them per v-head (`beta_in[(cs+t)*H_v + h_v]`,
  `g_in[cs*H_v + h_v + t*H_v]` — mma `prepare_wy_wu.cuh` Phase WY-A and the SIMT stage-1
  Phase A alike). Therefore M, T_inv and W are per-v-head **by construction**; the only
  k-group-sharable term is the raw KKT `K.K^T` (no beta/g factors).
- **Cell-level proof:** the parity cell plants exactly this sharing as mutation m3
  (`--selftest`, "[mut khead-W-share]"): W rel-L2 vs the honest path blows up to
  **4.042e-1** (40x the 1e-2 bar), vnew 3.2e-2, out 2.3e-2, state 1.2e-2 — all RED.
- The legitimate share (raw KKT, ~30% of stage-1 FLOPs) was **declined with math**:
  consuming it needs a k-head-gridded stage = 4 k-heads x NT(2) = **8 CTAs at TP4** —
  re-starving the grid (the exact pathology being fixed) plus a new workspace + launch.
- The dedupe that IS real and implemented: **T_inv is built once per (chunk, v-head) in
  LDS and consumed by both products** (U and W), and the g cumsum is computed once and
  published once. The 4x strip recompute of stage-1 (§2) is priced (~0.1 ms/chunk) and
  buys the 24 -> 96 CTA re-grid.

## 4. Gate: NINFER_GDN_SIMT (value-parsed per N7 house law, default OFF byte-identical)

Predicate lives in the **dependency-free** `chunked/simt_gate.h` (the FIX-B
`bf16_gdn_gating_simt_gate.h` pattern: the three stage launchers AND the zero-card cell
call the same function; reads getenv FRESH per consult — cell-flippable in one process).
Substitution point: top of each of the three stage launchers in `output.cu`,
`state_passing.cu`, `prepare_wy_wu.cu`. Decode (T=1 -> recurrent), the chunk tail
(recurrent inout) and the launch brackets in `launch.cu` are reached by none of them.

| NINFER_GDN_SIMT | effect | trace |
|---|---|---|
| unset (default) | OFF — all three launchers byte-identical to pre-FIX-A | **silent** |
| `1` | ON — trio routes to the `*_simt.cuh` kernels | two stderr lines (below) |
| `""`/`0`/anything else | OFF (strict parse: exact `"1"` only, NOT truthy) | one stderr line |

Line 1 (`[GDN]` gate decision, exactly once per process, on the FIRST consult that finds
the variable PRESENT — unset never prints):

```
[GDN] NINFER_GDN_SIMT='1' -> gfx900 SIMT kernels (prepare_wy_wu_simt/state_passing_simt/output_simt) (chunked-prefill trio route; strict parse: exact "1" arms, unset stays silent and byte-identical)
```

Line 2 (`[GDN]` dispatch receipt, armed path only, once per process, fired by the first
simt-routed chunked launch; names the grids actually dispatched for that shape):

```
[GDN] NINFER_GDN_SIMT=1 chunked prefill -> gfx900 SIMT kernels: prepare_wy_wu_simt grid=(2,12,4)x256, state_passing_simt grid=(12,8)x256, output_simt grid=(2,12,4)x256 (H_qk=4 H_v=12 T=128 chunks=2 kChunkSize=64); env unset keeps the mma path byte-identical
```

Note: this gate deliberately DIVERGES from the adjacent `NINFER_GDN_FORCE_RECURRENT`
parse (first-char truthiness, `gated_delta_net.cpp:268`): that var only downgrades
routing (loose is safe); this one swaps in rewritten kernels, so it carries the strict
armed/off grammar. **The WIP gate (`c79c59e3b`) was first-char truthiness — that defect
is fixed by this desk** (audit + RED/GREEN receipts, §5).

## 5. Host-only unit cell — RED + GREEN receipts (zero card)

`tests/test_gdn_simt_parity.cpp` (CMake target `ninfer_gdn_simt_parity_test` for CI/PG-1;
build-hip-amd keeps BUILD_TESTING=OFF, so the desk receipt is the house direct-compile
form per the FIX-B/FIX-C precedent). HONEST PROVENANCE: there is NO CPU reference for
this op in the tree (recurrent.cu is device code), so the cell compares CPU-EMULATED
LOGIC at the mission shape **H_v=4, H_qk=2, T=16, kChunkSize=8** (a shape the device
kernels' BT=64 assert forbids directly): old_emu = scalar mirror of the mma trio's order
(anti-diagonal-wave block Schur, `expf` decays) vs new_emu = mirror of the SIMT kernels'
order (block-rows-ascending Schur, uniform `exp2f(x*log2e)`), plus the token-recurrent
ground truth anchor (fp32) AND an **fp64 oracle arm** (double, no bf16 rounding).
Gates rel-L2 < 1e-2 on W/U/vnew/out/state old-vs-new, both-vs-anchor, both-vs-oracle;
determinism rerun bit-equal (memcmp); production gate matrix over
{unset, "", "0", "1", "yes", "true"} with **stderr-captured** trace-once asserts;
`--selftest` plants 3 defects as RED captures (M sign flip, dropped decay, the INVALID
per-k-head W share of §3).

- **Compile:** `g++ -std=c++20 -O2 -Wall -Wextra -I src -I include
  tests/test_gdn_simt_parity.cpp -o <bin>` -> RC=0, **zero warnings**, both runs.
- **RED row (gate defect, PRE-FIX artifact = `simt_gate.h` at `c79c59e3b` + cell at this
  commit):** run RC=**1** —
  `FAIL gate matrix: only exact "1" arms; ""/"0"/"yes"/"true" OFF` (the truthy parse
  arms "yes"/"true") and `FAIL trace-once: exactly ONE [GDN] line ...` (the WIP gate had
  NO consult-path trace: garbage was OFF-silent). Numeric sections already GREEN.
- **GREEN row (POST-FIX artifact = `simt_gate.h` at this commit):** run RC=**0** —

```
rel-L2 vs oracle  old: out=2.675e-03 state=4.020e-03 | new: out=2.675e-03 state=4.023e-03
determinism: old-path rerun bit-equal / new-path rerun bit-equal
gate matrix: only exact "1" arms; ""/"0"/"yes"/"true" OFF   [ok]
trace-once: exactly ONE [GDN] line across the whole matrix  [ok]
truthy-parse poison (pre-fix gate) goes RED on this matrix  [ok]
dispatch receipt: exactly one kernel+grid line              [ok]
pristine rel-L2 old/new: W=1.653e-04 U=1.989e-04 vnew=2.056e-04 out=1.129e-04 state=3.365e-04
             old/anchor: out=2.793e-03 | new/anchor: out=2.793e-03 state=4.023e-03
PASS, RC=0
```

- **`--selftest`: pristine=GREEN mutations_caught=3/3, RC=0** (M-sign: W 1.3e-1; no-decay:
  W 9.8e-2; khead-W-share: W 4.0e-1 — §3's proof).
- What the cell does NOT cover (honest boundary): device-kernel execution parity and the
  perf number — the GPU owner's probe below owns both (parity gate vs un-gated boot +
  the E-4-class chunked-vs-recurrent arm in the boot battery).

## 6. Compile receipts (this desk)

`make -C build-hip-amd -j20 ninfer_hip_host ninfer-serve` -> **RC=0**, 0 errors (fresh
link of all three gate TUs + full relink). Warning audit on the forced rebuild: 85
warnings, ALL the pre-existing `-Wpass-failed=transform-warning` unroll class on the OLD
mma kernels' lines (`state_passing.cuh:169` x40, `prepare_wy_wu.cuh:355` x31,
`output.cuh:347` x14) — **zero warnings on the SIMT kernels, the gate header, or any
added line**. Lane serve binary at commit time: `build-hip-amd/apps/ninfer-serve` sha16
`a43312bb9634bef9` (lane build path — a moving target per law: BANK it before any
relink, boot from the bank).

## 7. GPU-owner runbook (next card window; NO GPU work was done at this desk)

Predicted (P2 anchor: wall 1642.3 = gemm 798.8 + ar 70.7 + body 645.1 + gap 122.7;
`dn` 501.6-513.7 ms/chunk at ~0.2%-of-peak emulation -> plain FMA at even 3-8% of the
10.75 TF/s ceiling): **`dn` 510 -> <100 ms** (acceptance ceiling), body ~280 (with
FIX-B's gate <12 and FIX-C's gqa <20 landed), chunk ~1250, prefill-class **~95-105
tok/s** (from 75.5). These are PREDICTIONS for the probe to confirm — the number does
not exist until measured on this machine at this geometry.

1. **Rebuild** in the lane tree: `make -C build-hip-amd -j20 ninfer_hip_host
   ninfer-serve` (expect RC=0; warning classes unchanged per §6). No GPU needed.
2. **BANK-BEFORE-RELINK** (BOOT_LAUNCH_RUNBOOK §4): `cp -p` the freshly linked
   `ninfer-serve` to `/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin`
   (filename-is-stamp; this desk's link sha16 `a43312bb9634bef9` — re-sha at bank time)
   and commit the RECORD row. Pinned binaries boot from the bank, never the lane path.
3. **Boot battery** per runbook (written GPU grant + gpu_guard discipline; the battery
   banks `BOOT_BATTERY_<ts>.row` + clocks sideband — no row without its sideband
   compares to anything). The E-4-class chunked-vs-recurrent parity arm and the
   accuracy fingerprint (202/0.985/2.97 family) ride the window as the device-truth
   arm of the parity gate.
4. **Gated probe:** `NINFER_GDN_SIMT=1 NINFER_PREFILL_OPTRACE=2` on the known-good
   nvfp4 line, plen-1996 conc=1 first-request probe (P1/P2 grammar; cool window + clocks
   sideband first). Expect per rank process:
   - exactly ONE `[GDN] NINFER_GDN_SIMT='1' -> gfx900 SIMT kernels ...` gate line,
     present before the first prefill chunk;
   - exactly ONE `[GDN] ... dispatch receipt` line naming grid=(2,12,4)x256 /
     (12,8)x256 / (2,12,4)x256 at T=128 (chunked geometry hit at all — if the shape
     line shows something else, the TP4 head counts moved and §2's waves math must be
     re-run before trusting the ms);
   - `[PREFILL-BODY]`/`[PREFILL-BODY-SUM]`: **`dn` 510 -> <100 ms**; `gate`/`gqa`
     columns unchanged vs the un-gated reference within noise (FIX-B/FIX-C routes
     untouched); body sum drops by ~the dn delta; wall improves by the same order on
     the prefill tok/s readout;
   - optional depth split: `NINFER_PREFILL_OPTRACE=3` `[PREFILL-DN3]` across
     prepare/state_passing/output if the <100 ceiling is missed and the split is needed
     to name the laggard stage.
5. **Parity gate vs un-gated boot:** same probe, env unset, same banked binary family.
   Outputs are NOT expected bit-identical (declared fp32 order + expf->exp2f change;
   cell-measured ~1e-4 rel-L2 W/U/vnew/out, ~4e-3 state per tensor, and the recurrent
   anchor agrees with BOTH arms at 2.8e-3/4.0e-3) — model-level drift must stay inside
   the existing coherence/text fingerprint gates; that IS the acceptance bar, alongside
   the boot battery's chunked-vs-recurrent arm.
6. **Bank + promote decision to coordinator:** bank the row (suggest
   `GDN_SIMT_row.txt`) + serve log + clocks sideband; PLOG entry via the coordinator-
   owned chain helper (`tools/guards/plog_append.py`) — do NOT hand-edit
   `PERF_LOG_AMD.md`. The route stays env-gated default-OFF until the row is green;
   making SIMT the default for gfx900-class (no-MFMA) devices is the coordinator's
   promote call carrying its own row.

## 8. Files

- `src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu_simt.cuh` (stage 1/3 kernel + launcher; W-dedupe ruling header)
- `src/ops/linear_attention/gated_delta_net/chunked/state_passing_simt.cuh` (stage 2/3, typed state I/O)
- `src/ops/linear_attention/gated_delta_net/chunked/output_simt.cuh` (stage 3/3)
- `src/ops/linear_attention/gated_delta_net/chunked/simt_gate.h` (new dependency-free gate; REWRITTEN by this desk from the WIP truthiness parse to the N7 strict grammar + trace-on-present)
- `src/ops/linear_attention/gated_delta_net/chunked/{output,state_passing,prepare_wy_wu}.cu` (gate hooks only — `c79c59e3b`, unchanged by this desk)
- `tests/test_gdn_simt_parity.cpp` (zero-card cell; fp64 oracle + determinism rerun + gate matrix + stderr-captured trace asserts added by this desk) + `tests/CMakeLists.txt` (target)

Commits: `c79c59e3b` (FIX-A WIP, secured by coordinator), then this desk's
SIMT-kernels+gate / cell / notes commits (shas in the session report). Base: `amd/tp4-cure`
line.
