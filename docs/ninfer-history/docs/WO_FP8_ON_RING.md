# WORK ORDER: fp8-ON-RING AR (A5) — the last AR lever, prefill −6% wall target

Owner: desk agent (implementation + gates). Coordinator owns infrastructure; window shared
with the k4v4 desk (k4v4 takes first boots; your legs follow its quality gates — coordinate
via desk files, claim protocol in VALUE_QUEUE.md §PROCEDURE).

## THE MEASURED OPPORTUNITY
AR column = 123 ms/chunk (12% of the 990 ms wall), 125 calls/chunk at ~1.5 MB bf16, riding
the NCCL SHM ring. The ring is BYTE-BOUND: halving the bytes halves the AR time. fp8 partials
halve the bytes. MEASURED EXCLUSIONS (do not re-try): copy-add transport (same wire bytes as
ring, GATE-Q0 NULL 6.7x over), one-shot route (warmup wedge 6/7 boots + end-to-end -34% +
ulp-drift acceptance collapse 0.96->0.85 — PLOG-068), SDMA (3.3-3.6 GB/s), RCCL env (null).
THE ONLY SURVIVING SHAPE: quantize partials to fp8e4m3, transport half bytes over the
EXISTING ring, dequant at consumption.

## THE DESIGN PROBLEM (solve honestly; a wrong design fails the acceptance gate)
fp8 does not commute with summation: you cannot ncclAllReduce raw fp8 codes. Viable shapes:
(a) QUANTIZED RING: each rank quantizes its partial per output shard (per-channel or per-16
    group scales, fp16 sidecar), reduce-scatter over the ring with dequant-add-requant at
    each hop (2 extra requants), allgather fp8 codes, dequant at consumption. Noise: ~3
    quantizations + fp8 sum error per element.
(b) ONE-SHOT-LAYOUT over pinned staging WITHOUT the one-shot flag machinery: all 4 ranks
    stage quantized partials to a pinned buffer (plain pinned, NOT the mapped KAR staging —
    that path wedges), each rank reads all 4 quantized partials and dequant-adds LOCALLY
    (single quantization per rank, no hop compounding). NOTE: this is per-rank redundant
    compute (each rank sums all 4) but AR bytes leave VRAM once — on the SHM-staged ring the
    wire is host memory either way; price both.
Prefer (b) if the numbers hold: single quantization, no requant compounding, and it composes
with the ring only at the staging layer (NCCL stays out of it).

## IMPLEMENTATION
- Env arm: NINFER_AR_FP8=1 (+ NINFER_AR_FP8_SCALE=per_channel|per_tensor for the gate
  matrix). Unset/'0' = byte-identical ring path. No new refusal paths. Minimal diff at the
  AR entry in tp_group.cpp (the NCCL call site) — one branch, one quant kernel, one dequant
  kernel (or fused into the consuming op's epilogue).
- fp8e4m3 codec: reuse the copy-add cell's codec (tools/v340l/w7_copyadd_cell.cu — it is
  RED/GREEN proven, including the union-pun and encode-guard fixes).
- Build in YOUR worktree: git worktree add /home/chris/worktrees/amd-wo-fp8ar amd/main ->
  branch amd/wo-fp8ar -> build-hip-amd clone config (build dirs are per-worktree; the w7
  body desk's tree is not yours). df -h / first.

## PRE-REGISTERED GATES (write nothing after numbers exist)
- G-FP8-0 transport cell (standalone, dies 2/3): fp8 staged round-trip + dequant-add vs the
  646-771 us ring reference at 1.31-1.5 MB — bar: >=35% AR-time cut cell-level (byte-halving
  minus codec overhead). FAIL here = close the desk, bank the row.
- G-FP8-1 numerics cell: 4-rank fp8 sum vs fp32 reference on realistic partial magnitudes —
  max abs error and relL2 budget: relL2 <= 1e-2 per AR output (bf16-class), both directions
  falsifier.
- G-FP8-2 serve parity: temp-0 probes vs pre-arm bin — outputs will drift (different AR
  numerics): gate is BEHAVIORAL (BLUE/stop/no-mojibake x10) + MTP acceptance drop <= 3 pp
  vs the same bin pre-arm (docs/54 §6 gate class).
- G-FP8-3 perf: ordinal-paired fresh boots (PLOG-060), 3x plen-2075 mt64 per arm: ar column
  (NINFER_TP2_OPTRACE=1) -35% minimum AND wall -3% minimum => PROMOTE: runbook env + battery
  + 10k grade + PLOG row (chain head 74f40321209430b3). Miss => bank, report, close.

## LAWS
- This file is your checkpoint; append "## PROGRESS LOG" (newest-first) after EVERY step.
- Infrastructure = tag "BLOCKER:" and continue; coordinator resolves in 30 min.
- Window: k4v4 desk takes first boots; yours follow. Claim protocol via desk files.
- Clocks notes with every perf number; no bare pkill; no estimated VRAM refusals;
  restore canonical + health at every close.

## PROGRESS LOG

### 2026-09-19 fp8ar desk — step 5: G-FP8-0 FAIL, G-FP8-1 FAIL => DESK CLOSED (banked; no promotion)

Rows: `results/amd/coherence/WO_FP8_ring_row.txt` (three banked runs: 16:18 staged-INVALID
[superseded, see step 4], 16:18 rccl, ~16:2x staged-VALID rerun). All numbers below are
med_us, reproduced pair; clocks at run: sclk 300Mhz idle-class, window verified DOWN, no
foreign processes (row file snapshots).

**G-FP8-0 (transport; bar med <= 419.9us @1.31MB / <= 500.9us @1.5MB = 0.65x the 646-771
PLOG band) — FAIL, every arm:**
| arm | @1.31MB | @1.5MB | verdict |
|---|---|---|---|
| stage-R (bf16 staged ring, CONTROL) | 2522-2531 | 2850-2870 | anchor: reproduces the copyadd host-staged floor (2452-2796) within noise; 3.9x the band |
| stage-FR (WO shape a) | 3861-3866 | 4265-4268 | FAIL — 9.2x over bar (also 1.5x over its own stage-R floor: requant kernels drown the wire halving; PLUS my hop-requant impl has a real chunk-routing bug — ranks diverge, relL2 0.69, see below) |
| stage-FS (WO shape b) | 3159-3164 | 3593-3611 | FAIL — 7.5x over bar; transport CORRECT (crossrank=0) |
| rig-RING (bf16 NCCL, CONTROL) | 843-847 | 942 | anchor: real 4-die ring; +31% over the band (rig overhead: single-thread enqueue+sync vs the server's concurrent rank threads) |
| rig-FP8 (ncclFp8E4M3) | 1300-1394 | 1339-1342 | FAIL — 3.1x over bar and SLOWER than the bf16 ring itself: RCCL 2.20.5 accepts fp8 and sums correctly (banked fact — this was unknown) but the path is software-slow (wire_gbs 2.8-3.0 vs bf16's 18.6) |
| rig-I8P (packed 6-bit int32 lanes, exact integer ring AR) | 516-524 | 572-577 | FAIL as registered — misses the band bar by 23%/14% |

**G-FP8-1 (numerics; relL2 vs fp64 <= 1e-2 at the pre-stated NOMINAL kappa=1) — FAIL,
every quant arm:**
| arm | kappa=0 | kappa=1 (nominal) | kappa=3 |
|---|---|---|---|
| stage-FS (single requant-free sum) | 0.0256 | 0.0168 | 0.0144 |
| rig-FP8 (RCCL per-hop requant) | 0.0534 | 0.0454 | 0.0423 |
| rig-I8P (single quant + exact integer transport) | 0.0297 | **0.0160** | 0.0108 |
Falsifiers (both-directions witness): rig-I8P torn-payload CAUGHT (maxabs 4.109 vs clean
0.172); stage-FS torn-slot CAUGHT (2.141 vs clean 0.2812) on the VALID rerun.

CLOSURE PER THE PRE-REGISTERED LAW ("FAIL here = close the desk, bank the row"):
- **The fp8e4m3-on-ring shape is dead on this stack for two independent reasons:** RCCL's
  fp8 path is slower than bf16 (so "half bytes over the EXISTING ring" buys nothing), and
  fp8's 3-bit mantissa puts relL2 at 4.2-5.5% — 4-5x the numerics budget.
- **The class-level finding (the bug-class generalization the laws demand): byte-halving
  AR quantization is NUMERICS-BOUND, not transport-bound.** rig-I8P proves the transport
  side is solvable — exact integer sums, single quantization, -38.6%/-39.2% vs its own
  in-cell bf16 ring anchor, wire halved through the existing ring untouched — and still
  fails: no 6-bit-class code can meet relL2 <= 1e-2 on the nominal distribution (kappa=3
  only reaches 0.0108). Meeting the budget requires either a wider code (which forfeits
  the byte halving) or a principal-level numerics-budget call (e.g. relaxing to ~2e-2,
  which rig-I8P at kappa=1 would pass) backed by behavioral evidence of the G-FP8-2 class.
  That is a re-evaluation-mandate item, not this desk's call.
- Host-staged shapes (a)/(b): measured dead, as the copyadd desk predicted — the
  host-sync-per-step floor (2522us) is 6-7.5x the bar before any codec cost; shape (b)'s
  all-pull is 1.25x worse than the floor it replaces. CLOSED on numbers.
- stage-FR's chunk-routing bug (ranks diverge, relL2 0.69-0.75) is REAL in my cell
  implementation but does not affect the closure (the arm is 1.5x over its own floor even
  before correctness). Noted for honesty; not fixed — the shape is closed.
- The INVALID first staged run (16:18, g_n unset — every staged arm moved 0 bytes and
  verified zeros, stage-R control "failed") is superseded by the valid rerun and kept in
  the bank only as the cell-bug record. The cell fix is committed.
- SERVE ARM NEVER BOOTED: bin 4a7533a7b5f4e16c stays boot-gated, unbanked, on this
  branch (the ncclFp8E4M3 route it implements is measurably slower than bf16 — do not
  ship). No canonical state was changed (window found DOWN, left DOWN; no foreign
  process touched; watcher exited 16:18:29; zero pkill issued by this desk).

BLOCKER: none. DESK CLOSED.

### 2026-09-19 fp8ar desk — step 4: cell run 1 (staged rows INVALID — g_n bug) + rccl rows VALID

- Watcher fired at window-down 16:18; `run.sh both` banked three sections into
  results/amd/coherence/WO_FP8_ring_row.txt.
- **staged run 1 INVALID — cell bug:** `run_staged()` never set the global g_n, so every
  staged arm moved 0 bytes and verified freshly-allocated zeros (identical maxabs/relL2
  across arms, stage-R control "failing", wire_gbs 48-55 — impossible for real PCIe).
  Rows superseded by the rerun; fix committed (g_n = n in both loops). Kept in the bank
  as the cell-bug record.
- **rccl run VALID** (4 physical dies, window down, rig-RING control PASS: maxabs 0.039,
  big=0, crossrank=0 — a real bf16 ring AR at world=4): rig-FP8 supported-but-slow,
  rig-I8P fast-but-over-bar, falsifier CAUGHT. Rows in step 5's tables.

### 2026-09-19 fp8ar desk — step 3: serve arm implemented + built green (BOOT GATED)

- `src/core/multi_gpu/fp8_ring_allreduce.{h,cu}` (new): the RED/GREEN-proven copyadd
  codec verbatim + host selftest at arm-enable (mismatch disables the arm — bf16 stays),
  grow-only per-rank code staging (allocator is the gate; grow failure = documented bf16
  fallback, NOT a refusal — VRAM law), quant/dequant kernels in the house CUDA-API/shim
  dialect. Whitelisted in src/HipSources.cmake (device set) + src/CMakeLists.txt (CUDA
  lane keeps parity).
- `tp_group.cpp` one-branch diff at the AR entry (`allreduce_local_bf16`), pre-stated
  scope: **world>2 AND n_elems > OneShotAllReduce::kMaxElements** — exactly the ARs the
  one-shot machinery already declines (the ~1.31-1.5 MB prefill chunk ARs, 12% of wall).
  Decode-size ARs keep the byte-identical bf16 NCCL path (MTP acceptance surface
  untouched by construction). Env NINFER_AR_FP8=1; NINFER_AR_FP8_SCALE has exactly one
  legal value on the rccl-native design ("native": e4m3 precision is scale-free; no
  per-rank scale can ride a raw ncclSum over codes — a differing setting logs once and
  is ignored). Residual axpy unchanged, shared by both paths. Unset env = byte-identical,
  zero allocations. world==2 attractor path byte-frozen (arm cannot construct).
- Build green in this worktree: arm bin sha16 **4a7533a7b5f4e16c** (pre-arm baseline
  02e8cbbbde477975). Both binaries NOT booted — BOOT GATED on the G-FP8-0 cell verdict
  (a hunt may not ship a patch before its decisive measurement).
- G-FP8-2/3 legs script: `tools/v340l/w7_fp8ar_legs.sh` (window-guarded: refuses rather
  than touches another desk's server; ordinal-paired off/on boots of the SAME arm binary;
  behavioral x10 + acceptance extraction; NINFER_TP2_OPTRACE + NINFER_PREFILL_OPTRACE
  both on). Instrument note pre-stated: the WO's "ar column (NINFER_TP2_OPTRACE=1)"
  covers the decode verify-round ar phase, which the arm does not touch by design; the
  decisive prefill ar column is the [PREFILL-BODY] ar= field at NINFER_PREFILL_OPTRACE=1.
  Both collected; gate = prefill ar column, decode ar = no-regression witness.
- Commits: a7181a933 (cell+runner+desk), 47bff682c (serve arm), this step (legs script).
- Still BLOCKED on the window (k4v4 legs; watcher fires the cell when it drops).

### 2026-09-19 fp8ar desk — step 2: cell built; BLOCKER: window UP (k4v4 first boot)

- Cell `tools/v340l/w7_fp8_ring_cell.cu` written + compiled clean (sha16
  e0a4a4e589d52655… stamp at build; runner `tools/v340l/w7_fp8_ring_run.sh` stamps per
  run). Arms: staged mode (stage-R anchor / stage-FR shape-a / stage-FS shape-b /
  stage-FS-FALS) on the pre-registered dies-2,3 geometry; rccl mode (rig-RING anchor /
  rig-I8P exact-integer packed 6-bit / rig-I8P-FALS / rig-FP8 ncclFp8E4M3 support probe,
  FP8 LAST under alarm so a hang cannot eat the other rows) on dies 0-3.
- Commit a7181a933 on amd/wo-fp8ar (cell + runner + desk file + this log).
- **BLOCKER: serving window UP at 15:23 CDT** — pid 564256, the canonical pinned bin
  `ninfer-serve_2c8901d3d18adef1.bin` on :8100 (the k4v4 desk's first boot per the WO
  window law). No device work launched; zero GPU contact from this desk. The desk runner
  is window-guarded (refuses to start while any ninfer-serve process is live, re-checks
  immediately before each launch) and is queued behind the window via a watcher that
  fires `run.sh both` the moment the window drops. Nothing to kill; nothing killed.
- Serve baseline for G-FP8-2/3 built in this worktree (pre-arm, byte-clean tree):
  `build-hip-amd/apps/ninfer-serve`, sha16 02e8cbbbde477975. NOT booted, NOT banked yet.

### 2026-09-19 fp8ar desk — step 1: survey + analytic pricing (BEFORE any new measurement)

Setup: worktree `/home/chris/worktrees/amd-wo-fp8ar` branch `amd/wo-fp8ar` from `amd/main`
@ 3780414e0 (per VALUE_QUEUE §PROCEDURE). Build config cloned (HIP lane, serve-only,
mirrors w7-body cache). Desk file: `docs/amd/WO_FP8_ON_RING_desk.md`. df -h / at start:
11G free (91%), build tree measured 446M in the sibling desk — no disk risk.

Survey anchors (existing, not re-derived):
- AR entry: `src/core/multi_gpu/tp_group.cpp` `allreduce_local_bf16` — at TP4 default the
  one-shot is NULL (env-gated, wedges: excluded by this WO), so every AR rides
  `ncclAllReduce` at :398 -> RCCL SHM ring, 646-771 us/call at 1.31-1.5 MB bf16 (PLOG-062/063).
  Residual axpy (`one_shot_axpy_bf16`) follows the AR — any fp8 arm must preserve it.
- Codec: `tools/v340l/w7_copyadd_cell.cu` `f32_to_e4m3fn` + shim decode
  `ninfer_hip_shim_fp8::e4m3fn_to_f32` — RED/GREEN proven incl. union-pun + encode-guard
  fixes; reused verbatim.
- Copyadd row (results/amd/coherence/WO_COPYADD_row.txt, this box, 2026-09-19): the
  host-staged ring-pattern arm R (12S wire, 6 host-synced steps) measured med 2452-2462 us
  @1.31 MB and 2783-2796 us @1.5 MB — 3.8x ABOVE the 646-771 NCCL band. The floor is
  per-step host sync (~400 us/step), not wire bytes.

Analytic pricing of the WO's shapes (stated before measurement):
- (a) QUANTIZED RING host-staged: wire 6S but still 6 host-synced steps. From R's measured
  floor, predicted ~1400-1600 us @1.31 MB — 3x over the 419.9 us G-FP8-0 bar (0.65x646).
- (b) ONE-SHOT-LAYOUT all-pull over plain pinned staging: wire 10S (2S D2H publish + 8S H2D
  all-pull), ~2 host syncs. Sibling arm Q (8S wire, 3 syncs) measured 2852-2859 us —
  (b) prices >= Q, i.e. ~4x over the bar. The copyadd desk already paid for this lesson.
- DECISIVE IMPLICATION: no host-synced staging shape can meet G-FP8-0 on this stack. The
  literal "EXISTING ring" arm — RCCL carries the fp8 bytes itself
  (`ncclAllReduce(..., ncclFp8E4M3, ...)`; declared in rccl.h 2.20.5, reduction support
  UNVERIFIED) — is the only candidate with the ring's device-driven floor. A sibling
  byte-halving arm, per-channel int8 codes with a shared delayed scale (call k uses the
  amax measured at call k-1, clamped +-63 so 4-rank sums never wrap mod 256),
  `ncclAllReduce(ncclInt8, sum)` — integer sums commute exactly, mantissa 7 bits vs fp8's
  3 — is priced in the same cell because the fp8 numerics budget analysis above says it
  may be needed.
- G-FP8-1 risk analysis (pre-stated): e4m3 mantissa = 3 bits -> per-element quant rms
  ~2^-4/sqrt(3) ~= 3.6% of element magnitude; 4 independent quantized partials summed with
  random signs -> relL2 vs fp32 ref ~= 3.6e-2 on zero-mean data — 3.6x OVER the 1e-2
  budget. Scaling (per-channel/per-tensor) changes range use, NOT mantissa error. The
  budget can only be met if realistic partials carry enough common-mode (per-channel DC)
  that ||ref|| is dominated by it; the cell therefore prices realistic distributions
  explicitly, and int8 (0.46% class) as the numerics-safe codec.

Cell plan (arms, all pre-registered against the WO's bars — no bar moved):
- rig-RING: RCCL bf16 world=4 allreduce, expects the 646-771 band (validates the rig).
- rig-FP8: fixed-constant scale bf16->e4m3, ncclAllReduce(ncclFp8E4M3), dequant.
- rig-I8: per-channel delayed scale + clamp +-63, ncclAllReduce(ncclInt8), dequant.
- stage-FR / stage-FS: WO shapes (a)/(b) on the pre-registered dies-2,3 geometry, to bank
  their measured row (the desk closes them on numbers, not on prediction).
- falsifier: corrupted-slot arm must FAIL verification (both-directions witness).
- G-FP8-0 bar (unchanged): shipped-candidate arm <= 419.9 us @1.31 MB / <= 500.9 us @1.5 MB
  (>=35% cut vs the 646-771 band), reproduced pair.
- G-FP8-1 bar (unchanged): relL2 <= 1e-2 vs fp32 ref on realistic partials + corruption
  must trip.
- Die-law note: RCCL arms need 4 physical dies (duplicate-GPU refusal at 2 is banked
  evidence); window verified DOWN at claim time (desk file); host-staged arms stay pinned
  to dies 2,3.

BLOCKER: none.

