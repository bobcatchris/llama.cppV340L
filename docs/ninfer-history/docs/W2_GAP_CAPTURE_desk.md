# W2 GAP CAPTURE — what the prefill chunk gap actually is, and whether chunk graph capture is worth building

NO-GPU design desk (W2, gap-capture seat), 2026-09-18, lane `amd/wo-w7-body`. Read-only analysis +
this doc; zero builds, zero device work. Log under analysis (the promoted-GQA serving leg):
`results/amd/coherence/W7_boot_promote_gqa_serve.log`, request `n=17` (plen-2075, chunk 128,
bin a943225ad6da661e = ggate+gqa, `NINFER_PREFILL_OPTRACE` depth 2).

## R2 — start from the ledger

Closest prior rows: **P1 / PLOG-044** (`docs/amd/PERF_LOG_AMD.md` line ~240: prefill decomposition
wall 1642.3 = gemm 798.8 + ar 70.7 + body 645.1 + **gap 122.7**, with "chunk graph capture DEMOTED
to the 7.5% gap class it can actually touch (~+7.5%, NOT 100-130 tok/s)") and the plan's own
**G3/W2 line** (`docs/amd/PREFILL_1K_PLAN_2026-09-18.md`: "gap capture (chunk graph replay,
123→15–30, AR-desk sketched)"). The AR-desk spec the plan cites is
`docs/amd/PREFILL_ARTAIL_2026-09-17.md` §E (F-GRAPH re-ranked UP at the 500-goal scale, predicted
gap 122.7 → 10–30, "cheaper first move: chunk N+1 pre-enqueue").

**Why this desk differs: it builds, and in building it re-derives the number first.** ARTAIL priced
F-GRAPH against P1's *request-mean* gap (122.7); neither row ever read the PER-CHUNK rows of a
serving log. This desk did (W7 log, per-chunk `[PREFILL-OP]`/`[PREFILL-BODY]` lines, rank 0,
chunks 1–17) and the decomposition **changes the prize**: the steady-state gap is already ~20
ms/chunk; the 122.7/180 means are a two-chunk artifact (request-first-chunk accounting bug +
request-final-chunk anomaly). R5 (measurement beats model) applied one level down: the model of
"gap = per-chunk drain/re-enqueue bubble" dies against the per-chunk rows, and the capture design is
re-scoped to what the rows say is actually on the table.

## 1. What the gap column measures (code definition)

`gap = wall − layers` where `wall` is the chunk-wide device event pair
(`PrefillOpTrace::wall_begin/wall_end`, `src/targets/qwen3_6/impl/runtime/text_context_impl.h:3481`
and `:3679`) and `layers` is the sum of the 64 per-layer `VerifyLayerTrace` body pairs plus the
bracketed gemm pairs (`text_context_impl.h:1301-1321`, semantics comment `:1087-1090`). So gap =
**everything inside the chunk's wall window that is not inside a layer body pair**: the chunk head
(ids copy → embedding), the chunk tail (final norm, last-chunk lm_head/argmax, the whole MTP
prefill-chunk block), and any device idle inside the 64-layer loop not covered by a layer pair.
Note `prefill_impl` is ONE chunk per call (the loop `break`s at `:3691`); the multi-chunk driver is
`src/runtime/tp2/tp2_backend.cpp:2164-2212`, and every chunk ends in a hard `ctx_.synchronize()`
(`:3696`).

## 2. The finding: the W7 log decomposes the "180 ms gap" into three very different things

Per-chunk rows, rank 0, request n=17 (log lines 305-1535):

| chunk | tok | wall | gemm | ar | body | gap |
|---|---|---|---|---|---|---|
| 1 | 128 | 1505.4 | 783.8 | 10.5 | 0.0 | **1454.5** |
| 2 | 128 | 1469.1 | 764.4 | 71.6 | 653.0 | **0.0** (clamped) |
| 3–16 | 128 | 1466–1517 | 762–774 | 68.6–73.2 | 616–655 | **13.8–24.1 (mean 20.1)** |
| 17 | 27 | 2027.9 | 437.1 | 49.5 | 165.1 | **1376.2** |

Reconciliation: (1454.5 + 0 + 280.9 + 1376.2)/17 = **183.0 ≈ the printed 181.9 mean**. The famous
"gap 122.7" (P1, n=16) is the same arithmetic with 16 chunks. Components:

### (a) Steady-state gap ≈ 20 ms/chunk — ALREADY at the plan's post-capture target (15–30)

Fixed per-chunk cost, NOT token-proportional. Named owners (estimates are arithmetic, the ~20 is
measured):

- **MTP bulk chunk** (`mtp_prefill_chunk`, non-final path, `text_context_impl.h:3588-3666`,
  `:3663`): stem = embedding + 2×rmsnorm + `mtp_pack_fc_input` + `ops::linear` fc
  (hidden×2·hidden·T-class GEMM ≈ 13.4 GFLOP ⇒ ~6–8 ms at today's ~2 TF/s/die) + MTP kv projection
  + rope + `gqa_kv_append` (:1696-1710) ⇒ **~8–13 ms, all ranks symmetric** — the largest *named*
  steady-gap item. It runs EVERY chunk and lands entirely in gap (outside layer pairs).
- **`begin_chunk` event drain inside the wall window** (`text_context_impl.h:1219-1239`, called at
  `:3270` from `run_layers` BEFORE the layer loop, i.e. right after `wall_begin`): reads ~192 gemm +
  up to 512 body event pairs (`cudaEventElapsedTime` ×~1500) + two printf/fflush
  (`:1301-1321`, `prefill_body_trace.h:208-249`) — host work of ~5–15 ms during which the device has
  almost nothing queued (chunk just started).
- **Chunk head/tail unbracketed kernels**: `copy_i32` ids H2D (pageable, `:3504`),
  `fill_i32_positions` (`:3507`), `set_i32_scalar` rope_delta (`:3420`), `ops::embedding`
  (`:3533`), final rmsnorm (`:3552`): <5 ms combined.
- **Inter-layer launch slack**: between `vt.layer_end(L)` (`:3369`) and `layer_begin(L+1)`
  (`:3311`) the only host work is dispatch — a few ms/chunk when enqueue lags.
- Between-chunk turnaround (host sync `:3696` → tp2 driver loop `tp2_backend.cpp:2166-2212` → next
  `wall_begin`) is **outside both wall brackets** — invisible in the columns, bounded by the
  2–3 ms/chunk enqueue census (ARTAIL §A.3). Not the prize.

### (b) `unbr` ≈ 60–84 ms/chunk — the REAL graph-capture class, and it lives inside `body`, not `gap`

`[PREFILL-BODYSUM]` (rank 0: unbr 63.6; ranks 1/2: 84.1/75.4) = layer-body time NOT covered by the
12 bracketed op classes — i.e. launch slack BETWEEN the ~1300 in-layer kernel launches (the dn trio
alone is 48 launches/chunk of multi-kernel bodies; the depth-3 splits l2/wy/sp/out/gmma exist in
`NINFER_PREFILL_OPTRACE=3`, `prefill_body_trace.h:32-34`, to say which sub-class owns it — **never
run at depth 3 yet**). Fixed at T=128 (unbr 73.9 chunk-3 → 62.6 chunk-16). Whole-chunk graph capture
eats this class; per-layer capture eats most of it. **gap 20 + unbr 65 ≈ 85 ms/chunk is the honest
capture prize — 5.7% of the 1490 ms steady wall, not the 7.5-12% the ledger rows imply.**

### (c) Finalize chunk (chunk 17): 1376 ms of in-wall, unbracketed, UNEXPLAINED time — the single biggest open item

Chunk 17 (T=27, `finalize=true`) wall = 2013–2028 with only 651 named. Its BODY is fine (165: dn 8.0
+ **gqa 97.7** — the KV-length-bound attention class, grown 11.8→35.2→54.9→97.7 across the request;
plus unbr 39.2), so the 1376 gap is all in the finalize tail the layer pairs never bracket:

- last-token lm_head + `allreduce_argmax` R1 arm (`:3554-3586`, `:3563-3564`) — physics says 2–6 ms;
- `mtp_prefill_chunk` final (stem + q/kv proj + MTP attention over the ~2.1k-token MTP KV + head,
  `:3640-3645`) — the gqa class again, ~20–120 ms;
- the MTP ar-step loop ×k−1 (`:3649-3661`) — ~5–50 ms;
- a hard `cudaStreamSynchronize` + D2H when the MTP window needs the generated token
  (`:3614-3617`) — host round-trip, device idles only for the round-trip;
- **named-op physics sum ≈ 30–180 ms vs 1376 measured — a >7× remainder no code path read so far
  owns.** It is per-REQUEST (amortizes to ~80 ms/chunk of the request mean — BIGGER than the whole
  steady capture prize), rank-uniform, and repeats on every long request.

Per R5/R7 this remainder must be *measured, not modeled* before anyone builds capture: it is
plausibly either (i) real device work whose owner is a kernel we have not bracketed (MTP-final
attention class, lm_head route), or (ii) device idle from a host stall in the finalize path. Both
dispositions kill or re-scope the capture build.

### (d) Chunk-1 line is an accounting artifact — free fix, metric honesty

Chunk 1 prints body=0.0 / ar=10.5 / gap=1454.5 = wall − 51: the `vt.begin_pass` drain /
`begin_chunk` snapshot order at request start (`:3257-3272` vs `:1219-1239`) attributes chunk 1's
layer events to no line (chunk 2's line then clamps to gap 0.0). The wall itself is sane (1505 ≈
steady). Fix = correct the drain/parity handoff so `[PREFILL-SUM]` means stop carrying a phantom
(1454−1505)/17 ≈ −3 ms and a +81 ms finalize-only view; zero device risk.

### (e) AR peer-wait slosh — ~17 ms/chunk on early ranks, NOT transport, NOT capture-class

ar is anti-correlated with gemm per rank (W7: r0 66.0 vs r3 49.3 ar; r3 gemm 772.6 vs r1 717.3 —
identical pattern to P1, ARTAIL §A.3: gemm+ar near-constant per rank). The lockstep AR absorbs rank
GEMM skew; de-skewing gemm shrinks ar spans on fast ranks and NOT the wall. Do not book it as
recoverable.

## 3. Capture design (if the build is chartered after §4's checks)

**Kill-switch first (R3) — zero-GPU checks, in order:**

1. **Per-chunk allocation census**: capture freezes pointers; every chunk must produce IDENTICAL
   arena offsets. `work_.reset()` (`:3466`) + `workspace_recipe::text_prefill_roots` (`:3501`) +
   the deterministic `work_.scope()` nesting in `run_layers`/mixers must be proven
   allocation-identical chunk-to-chunk (any growth-path `cudaMalloc` inside a chunk defeats
   capture permanently). Static read; failure = capture infeasible without an arena freeze.
2. **Dynamic-shape / per-chunk-host-value audit** (each named with its fix or eager-escape):
   positions base (`fill_i32_positions`, `:3507` — host scalar ⇒ must move to a device scalar
   written pre-replay; the `io_.pos` `set_i32_scalar` pattern at `:3561` already exists), rope_delta
   (`:3420`), gqa envelope `{visible, visible}` (`:3529-3530` — host values each chunk; audit the
   `ScopedEnvelope` plumbing to device-side), ids H2D (pageable `copy_i32`, `:3504` — memcpy from
   host cannot be captured; pre-stage into a fixed device staging buffer pre-replay), GDN state
   slots + MTP KV page/tail growth (device-side block tables — audit `tail_count/tile_page`
   capture-compat in `mtp_prefill_chunk`), and the **finalize machinery** (`:3554-3667`): stays
   eager, every chunk's `is_last` runs eager by design.
3. **Watchdog interaction audit (W4-wedge class)**: the B3 heartbeats post per-AR host crossings
   (`src/core/multi_gpu/tp_group.cpp:377-395`); under REPLAY there are no per-AR host crossings, so
   `ar_watchdog` sees a silent rank for the whole chunk and false-kills. Requires a per-chunk
   heartbeat or widened tolerance under the env gate — a missed one here resurrects the 17-min
   wedge class.
4. **OPTRACE disposition**: ~1600 event pairs per chunk become graph nodes (overhead) or go dark
   under replay. Decide: capture with instrumentation off; wall pairs stay outside the graph so the
   before/after wall+gap columns remain measurable.

**What can be captured** (at T=128, non-final chunk): the 64-layer body incl. both GEMM families
(fixed shapes: 5-problem census, ARTAIL §C2) + all 128 per-layer NCCL allreduces + the MTP bulk
chunk (fixed shapes). **Feasibility on this stack is de facto proven for the machinery**: decode
runs `use_cuda_graph` default-ON at TP4 (`src/serve/serve_options.h:53`, `--no-cuda-graph` at
`serve_options.cpp:291`; profiles `src/targets/qwen3_6/impl/runtime/layouts.h:94`), captures via
`cudaStreamBeginCapture(ThreadLocal)` (`src/core/decode_graph.cpp:61`) around the ordinary-batch
body (`program_impl.h:1324`) which **enqueues `ncclAllReduce` inside the captured region** (the
same `tp_group.cpp:398` call prefill uses) — i.e. graph capture WITH RCCL collectives at world=4
already boots, serves, and passes the boot battery on ROCm 6.2/gfx900. Unproven residue: capture of
the 1.31 MB AR specifically (capture-compat is size-independent in principle; only 10 KiB is
measured) — keep the named fallback.

**Fallback (the cheaper 80%) if capture is refused**: per-layer-body graphs BETWEEN the ARs (no
collective inside any captured region — ARTAIL §E already names this), which still eats most of
unbr; plus, independent of capture: (i) the chunk-1 parity fix (free, §2d), (ii) the finalize
brackets + hunt (§2c), (iii) ARTAIL's pre-enqueue/double-buffer driver move (few ms), (iv) moving
the `begin_chunk` drain after the chunk-end sync (host reorder, 5–15 ms), (v) batching the dn-trio
sub-launches once depth-3 names unbr's owner.

**Capture trigger / replay path**: first T=128 non-final chunk of the first request post-boot runs
eager *under capture* (slow once); instantiate; replay all subsequent T=128 non-final chunks
(pre-replay host prologue: stage ids, write position-base/rope_delta/envelope device scalars,
`work_.reset()`, `cudaGraphLaunch`); any chunk with T≠128 (51/27/2 tails) or finalize runs eager.
**Env gate: `NINFER_PREFILL_GRAPH=1`, default OFF; unset = byte-identical binary behavior** (same
structural-gate pattern as `pf_optrace_on`, `text_context_impl.h:3392`). One-shot AR
(`NINFER_TP_ONESHOT_AR`) is irrelevant here — size-dead at prefill (655,360 ≫ kMaxElements 65,536,
`one_shot_allreduce.h:17`).

**Decisive GPU measurement**: same leg, same discipline (plen-2075, conc=1, cool window, banked bin
+ boot battery GREEN before and after, greedy text byte-identity — E-17 bar):
`[PREFILL-SUM]` wall and gap columns + per-chunk `[PREFILL-OP]` rows, ON vs OFF, plus the
steady-chunk wall (chunks 3–16) as the primary endpoint (the request mean carries the finalize
anomaly as noise). **Acceptance: steady-chunk wall −40…−90 ms/chunk with byte-identical output;
kill in-doc if <20 ms/chunk or any byte mismatch.** Record per R1 in the plan ledger with the
before/after bin shas.

## 4. Honest risks

- **The prize is 85 ms/chunk (5.7%), not 123→15-30.** The ledger's gap number was a mean of a
  20 ms steady state and two per-request transients. Anyone quoting −90-110 ms/chunk for capture is
  booking the chunk-1 artifact and the finalize anomaly, neither of which replay touches.
- **RCCL-in-graph at 1.31 MB is argued, not measured** (decode proves 10 KiB). If the first capture
  attempt fails at the AR nodes, the per-layer fallback survives.
- **The finalize anomaly (1376 ms/request) may dominate everything** — if §2c's brackets name a real
  unbracketed kernel (MTP-final attention or lm_head route class), fixing IT is worth more than
  capture, and capture built first would be measured against a denominator it cannot move.
- Multi-rank lockstep capture/replay ordering is a new liveness surface (the W4 lesson); the
  watchdog audit (kill-switch 3) is not optional.

## 5. Recommended build order (one GPU window)

1. (host-only, ~hours) Chunk-1 drain-parity fix + finalize-chunk bracket pairs
   (lm_head+argmax / MTP-final / ar-step — 3 event pairs, printed in the existing
   `[PREFILL-OP]` finalize line). Decides §2c's disposition. **Build this first.**
2. (same window) Run the depth-3 OPTRACE (`NINFER_PREFILL_OPTRACE=3`) on the same leg: names
   unbr's owner (l2/wy/sp/out/gmma) — decides capture vs dn-launch-batching for the 65 ms class.
3. (window 2, only if 1+2 say the 85 ms class is the best remaining lever) The capture build per
   §3, env-gated, kill-switches 1-4 signed off, acceptance as stated.
