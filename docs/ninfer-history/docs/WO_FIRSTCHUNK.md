# WORK ORDER: first-chunk request-start cost — name the mechanism, one instrument boot (VALUE_QUEUE TIER 1 item #3)

Owner: desk agent (1 slot). Phase 1 is GPU-free prep; Phase 2 is EXACTLY ONE serve boot,
QUEUED behind the open G-BF-1 claim (do not take the window; poll). Phase 3 is
design-only this session.

## CONTEXT YOU MUST KNOW (banked facts; cite, do not re-derive)

- The finding (PLOG-071 window): chunk 1 of EVERY request pays a 305-329 ms hot /
  ~900 ms cold request-start gap; ~1.8% of request wall at plen-2075, ~0.4% at 10k.
  NOT boot-once, NOT parity-linked. Gap "bimodality" was closed as an extraction
  artifact — first-chunk cost is the SURVIVING lever.
- The instrument is ALREADY BANKED: bin `ninfer-serve_05628598c977e42a.bin` in
  /home/chris/artifacts_bin/ carries the OPTRACE depth-3 `[PREFILL-HOST]` instrument
  (src/targets/qwen3_6/impl/runtime/text_context_impl.h: h2d_trace_bytes counter +
  per-parity h3_tb/h3_span/h3_free_b/h3_free_e/h3_h2d_b/h3_h2d_e sampled in
  wall_begin/wall_end via cudaMemGetInfo, emitted in emit_line; commit 5d95f60fb on the
  w7-body lane). READ THE SOURCE on branch amd/wo-w7-body (`git show
  amd/wo-w7-body:src/targets/qwen3_6/impl/runtime/text_context_impl.h`) to learn the
  exact env gate + line format — do not guess.
- Analysis tools BANKED on amd/wo-w7-body: tools/v340l/w7_gap_clock_sampler.sh (clock
  sideband) + tools/v340l/w7_gap_correlate.py (request-segmented gap analysis).
- Thinking-model serve contract: bodies need "model":"qwen3.8-27b"; probes max_tokens>=128.

## PHASE 1 — PREP (GPU-free; deliverable = a ready one-command leg)

1. Read the instrument source (cited above); write down the enabling env vars and the
   OPTRACE line schema in this WO's log.
2. Stage `tools/v340l/wo_firstchunk_leg.sh` in YOUR worktree: boot the banked
   05628598 bin (canonical env shape = serve_10k.sh's exports; NOT the promoted line —
   this bin predates it; do not add posture flags beyond the canonical set), BLUE sanity,
   6x plen-2075 mt=128 probes (the standard body), capture serve log + clock sideband,
   retire via EXACT law form, then run the correlator; restore canonical serve_fast +
   health. Bin check first: file exists + sha16 matches the filename stamp (bank law).
3. Dry-check the runner line-by-line (bash -n) — a dead agent must not leave a broken
   script for the window.

## PHASE 2 — THE LEG (ONE boot; window-queued)

- WINDOW CLAIM: post "WINDOW CLAIM: firstchunk-leg" in this WO ONLY when
  `pgrep -f '^/home/chris/artifacts_bin/ninfer-serve'` is EMPTY and the G-BF-1 claim in
  docs/amd/WO_BODY_FUSION.md (amd/main tail) is CLOSED. Poll every ~10 min; between
  polls, work Phase 3.
- Execute the leg. Deliverable: the first-chunk gap NAMED at mechanism level with
  receipts — which allocation/init event(s) make the 305-329 ms, from the h3_* spans
  (h3_free vs h3_h2d ordering answers "allocator OOM-scan? pin? staging alloc? page
  touch?"). Bank: results/amd/firstchunk/ (serve log, OPTRACE extracts, sideband,
  correlator output, a NAMED-mechanism row).
- Bar for "named": a mechanism claim that predicts a FIX LEVER (e.g. "X MiB pinned alloc
  at first chunk = Y ms of the gap; pre-touch or pool would erase it").

## PHASE 3 — PRE-TOUCH FIX DESIGN (no code this session)

Design the fix the mechanism names (pre-touch / pool / alloc hoist — let the data pick).
Pre-register its serving gate (PLOG-060 pairs; first-chunk gap reduction bar from YOUR
measured split; no regression elsewhere). Do NOT implement without a fresh coordinator
grant — pricing first, code second.

## LAWS

- Own worktree + branch amd/wo-firstchunk; progress log in THIS file newest-first after
  every step; resumable from this file alone.
- NO channel posts (COMM LAW). NO cmake builds. df -h / before any >1G write.
- Retire EXACTLY `pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"` and only your
  own boots. Restore canonical + health at every close. NO estimated VRAM refusals —
  the allocator is the gate.

## PROGRESS LOG (newest first)

### 2026-09-20 firstchunk — LEG EXECUTED + CLOSED: first-chunk gap NOT REPRODUCED; mechanism named (artifact + cured pageable-H2D class)

Leg ran 10:33:25-10:40:53 (window granted via intercom; handed-over serve retired with
the EXACT law form first; restore serve_fast + health + PLOG-084 +3min liveness OK).
7/7 probes behavioral PASS (blue + 6x standard body; this era's tokenizer yields
prompt=2134 — same 16x128+finalize chunk class as plen-2075).

**NAMED MECHANISM (full row + receipts: results/amd/firstchunk/MECHANISM_ROW.md):**
1. **No per-request first-chunk cost exists on the era-faithful posture.** ttft −
   prompt/steady-tput = +1/−7/−5/+4/+8/+14/+6 ms across blue+p1..p6 (cold AND hot);
   arrival→first-chunk-done ≈ one steady chunk wall + ≤14 ms. Chunk-1 device wall is
   FASTER than each request's own steady mean on all six long requests (thermal drift
   explains rising walls: gemm 770→1063 ms across the series). PLOG-071's 305-329 ms
   hot / ~900 ms cold class: ABSENT.
2. **h3_* fields exonerate all four candidate mechanisms at hot chunk 1:** host_span
   5.9-9.8 ms (flat), Δfree = 0.000 MiB (allocator silent; pool plateaued 1.394 GB
   after req 1 and never moved), h2d_kb = 2-4 KiB. Phase-3 pre-touch/pool design is
   MOOTED BY MEASUREMENT — nothing per-request to pre-touch.
3. **The 918.2/922.3 readings (first two long requests' chunk-1 [PREFILL-OP] lines)
   are a stale-drain ARTIFACT, not time:** ar field byte-equal to the preceding
   MTP-redo line's ar (49.197/17.956), body=0.000, layers=wall−gap < gemm (impossible —
   gemm pairs run inside layer windows); the walls themselves (1004/975 ≈ steady) leave
   no room for real 900 ms idle. Coordinator's "early-request transient, every-request
   premise refuted" reading CONFIRMED and sharpened: not pre-chunk latency either
   (Receipt 1 bounds it at ≤14 ms on the same requests).
4. **Residual first-occurrence costs (not per-request):** one-time pool growth
   (~208 MiB total across warmup+p1, then frozen); p1 finalize chunk-17 host_span
   1037.1 ms = KNOWN legacy finalize-stall class (this bin predates NINFER_MTP_TAIL_ASYNC
   consumption; once-only — p2-p6 finalize spans 6.7-8.5 ms).
5. **FIX LEVER: none for first-chunk on this posture.** If the class is chased further,
   the named re-attempt is ONE control boot with NINFER_H2D_PINNED_STAGE unset
   (pre-PLOG-059 era): hypothesis = the historical gap WAS the pageable copy_i32 stall
   (167-185 ms/rank class, cured in canonical since 09-18) + the Receipt-3 artifact.
   Pricing first, coordinator grant required.
6. **TOOLS BUG BANKED:** w7_gap_clock_sampler.sh emits a header-only CSV (its heredoc
   replaces python's stdin → json.load(sys.stdin)=EOF → every sample dropped; raw
   clock CSV is the RED evidence; correlator clock_samples=0). One-line fix class
   (read the pipe); left to the w7 lane per pricing-first.

Artifacts: results/amd/firstchunk/{serve_firstchunk_20260920_103325.log,
optrace_rank0_20260920_103325.txt, firstchunk_summary_20260920_103325.txt,
probes_20260920_103325.txt, correlate_20260920_103325.txt, clock_*.csv,
MECHANISM_ROW.md}. Runner: tools/v340l/wo_firstchunk_leg.sh (executed as staged;
grant via FIRSTCHUNK_ALLOW=1 under the logged intercom provenance).

### 2026-09-20 firstchunk — WINDOW CLAIM: firstchunk-leg — GRANTED via coordinator intercom; executing the staged leg

Coordinator grant received 2026-09-20 ~10:35 via intercom (verbatim key line: GRANTED,
"WINDOW CLAIM: firstchunk-leg GRANTED" — the window is ours NOW; not yet appended to
this file, hence the runner runs with FIRSTCHUNK_ALLOW=1 under this logged provenance).
Coordinator facts adopted:
1. Standing serve :8100 = coordinator's canonical (058a7b85859c5cd0, promoted
   kvarn_k4v4@65536/ws512) — HANDED TO THIS DESK: retired with the EXACT law form
   immediately before the leg boot.
2. ERA-FAITHFUL ENV: ws96 EXPLICIT (the 305-329 ms finding was measured at ws96;
   serve_10k.sh has since flipped to ws512 — runner does NOT inherit it; its export
   block is hardcoded canonical + ws96). No kvarn flags on the leg boot (era: bf16 KV,
   pre-promotion posture).
3. Close rule extended (PLOG-084): restore via bash /home/chris/serve_fast.sh (now
   boots the promoted kvarn line — EXPECTED, not a bug) + health + liveness RE-CHECK
   at +3 minutes.
4. If spans name a GEMM-adjacent term, price against POST-mad-mix columns
   (PLOG-085 conversion lesson: only request-critical-path reductions convert).

### 2026-09-20 firstchunk — POLLING for the window (Phase 1+3 complete and committed)

Phase 1 done (branch tip e751b5056 runner + schema; dec7ed848 Phase 3 tree). Polling
both WO copies every ~10 min since 07:07 for the coordinator's grant line (the literal
phrase "WINDOW CLAIM:" + "firstchunk-leg" + "GRANTED", as its own line — CAUTION:
grep it ANCHORED `^WINDOW CLAIM:...`, unanchored matches this very prose note, which
caused a false grant read at 08:04). Pipeline ahead of the leg (coordinator's queue):
kvarn battery arm B (running — serve up since ~06:43, bin 8ac1ba93) -> KV default
flip verify -> G-BF-1 pairs window -> chunk256 pairs window -> THIS leg. RESUME =
grepped grant line, then run
`FIRSTCHUNK_ALLOW=1 bash tools/v340l/wo_firstchunk_leg.sh` ONLY after (a) grant line
present, (b) `pgrep -f '^/home/chris/artifacts_bin/ninfer-serve'` EMPTY, (c)
G-BF-1 claim CLOSED in docs/amd/WO_BODY_FUSION.md (amd/main tail). The script
re-checks (a) and (b) itself and refuses otherwise. Post my claim line
"WINDOW CLAIM: firstchunk-leg" into this WO at take-over.

### 2026-09-20 firstchunk — PHASE 3: mechanism decision tree from the h3_* span fields (design-only)

Inputs per chunk (rank 0): `host_span` (host wall of the enqueue/launch section),
`Δfree = free_b − free_e` (device bytes allocated inside the chunk; negative Δ = drop),
`h2d_kb` (copy_i32 staged bytes), plus `[PREFILL-OP]` wall/gemm/body/gap and the
per-probe client epoch marks. Request segmentation: `chunk=1` boundaries; req1 = BLUE
sanity, req2 = first standard body (COLD), req3-7 = HOT repeats.

**Q1 — does hot chunk-1 `host_span` exceed steady-chunk host_span (req3+)?**
- **NO** → the 305-329 ms is not host-blocking in the launch section. Two sub-cases:
  - `[PREFILL-OP]` chunk-1 wall also high → device-side chunk-1 work (state/KV init
    kernels, first-touch page-in by kernels). Fix family: device init hoist into
    boot/between-request idle; NOT a host pre-touch.
  - chunk-1 wall ≈ steady but the client-visible request start is still slow → the gap
    lives BETWEEN requests (request setup before the first prefill chunk: tokenizer,
    request struct alloc, KV reservation). Join client epoch marks to first
    `[PREFILL-OP] t=` to price it. Fix family: request-setup path pre-touch/pool.
- **YES** (host blocked inside chunk-1's launch section) → Q2.

**Q2 — Δfree at hot chunk 1: MiB-scale DROP or flat?**
- **DROP** → a device allocation happens inside chunk 1 → **Q3**:
  - free RESTORED between requests (next request's free_b back at baseline) →
    **per-request alloc/free cycle + ROCm allocator re-alloc (OOM-scan/suballoc) is the
    mechanism**. Named claim shape: "X MiB device re-alloc at every chunk 1 = Y ms
    (measured hot span minus steady span); retaining/pooling the workspace across
    requests erases Y". FIX = POOL (retain) or boot-time HOIST sized to X MiB.
  - free NOT restored (pool already holds; later requests show Δfree≈0) and the stall
    STILL recurs → the alloc is one-time; persistent stall must be page-touch of the
    mapped region or non-alloc host-prep → fall through to Q4/Q5 (record the
    contradiction; the one-time part is the COLD ~600 ms excess, priced separately).
- **FLAT (Δfree≈0)** → not a device alloc → **Q4**:
  - `h2d_kb` SPIKE at chunk 1 vs steady → transfer-side: pinned ring `grow()`
    (cudaHostAlloc, 64 KiB granular) or staged-copy first-touch. Ring persists
    (thread_local, grows to max seen) → if hot requests still stall here the host
    page-touch of re-grown slots is named only if growth recurs; else → Q5.
  - `h2d_kb` flat → **Q5 HOST-PREP class** (banked READ table): enqueue thread blocks
    on host-prep with no device-alloc and no copy counter movement — candidates: host
    page-touch of per-request staging mallocs, small pinned allocs (device-Δfree
    invisible), sync queries. Depth-3 fields cannot split these further → follow-up is
    strace/phase-print at the next instrument grant, but the FIX FAMILY is already
    determined: **pre-touch/pool at request-accept**.

**COLD vs HOT split:** req1/req2 carry boot-time one-time costs (module warm, first
alloc, ring first-grow); the persistent hot 305-329 ms is the target. Cold − hot ≈ 600
ms prices the boot-warm lever separately (serve_10k's stem warmup already owns part of
the width class — do not double-count).

**Fix selection rule (let the data pick):**
- per-request Δfree drop recurs → POOL/HOIST the device workspace (pool preferred:
  allocator stays the gate, no new budget constants — VRAM LAW safe).
- Δfree flat, host-prep → PRE-TOUCH at request-accept (touch the staging/working set
  before the first chunk is enqueued) + a boot-time warm at max observed width.
- h2d/pinned-ring growth recurs → pre-grow the ring at boot (kSlots warm-up), a
  ~5-line patch in PinnedCopyRing::ensure (NOT this session).

**Pre-registered serving gate (for the fix, after this leg prices it):** PLOG-060
ordinal-paired fresh boots, 3x plen-2075 per arm, ±2% within-pair; PRIMARY BAR =
chunk-1 hot surcharge reduced by ≥50% of the measured (leg) chunk-1 vs steady host_span
delta; SECONDARY = no steady-chunk wall regression >±1%, BLUE parity PASS both arms,
steady-state Δfree footprint unchanged (pool must not raise peak residency; the
allocator remains the only launch gate — no estimated VRAM refusals).

### 2026-09-20 firstchunk — PHASE 1 steps 1+2: instrument source read; bin verified; runner STAGED

**Step 1 — instrument gate + line schema (read from `git show amd/wo-w7-body:src/targets/qwen3_6/impl/runtime/text_context_impl.h`, commit 5d95f60fb line; corroborating `src/core/prefill_body_trace.h`, `src/runtime/tp2/verify_tail_trace.h`):**

- **Enabling env var (exact): `NINFER_PREFILL_OPTRACE=3`.** Depth parse
  (`PrefillBodyTrace::env_depth()`, prefill_body_trace.h:103): unset=0; else
  `clamp(atoi,1..3)`. Depth>=1 arms P1 `[PREFILL-OP]`; >=2 adds `[PREFILL-BODY]`
  op pairs + `[PREFILL-DN3]`; **>=3 additionally arms the depth-3 host instrument:
  `[PREFILL-HOST]`** (guarded at wall_begin/wall_end/emit_line by
  `env_depth() >= 3`). Default OFF is byte-identical (structural gate).
- **`[PREFILL-HOST]` line schema (emit_line, text_context_impl.h:1493):**
  `[PREFILL-HOST] rank=<r> chunk=<c> host_span=<ms .1f> free_b=<bytes> free_e=<bytes> h2d_kb=<KiB .1f>`
  - `host_span` = host wall (ms) across the chunk's launch section, sampled
    `mono_ms()` (steady_clock ms == CLOCK_MONOTONIC domain) in wall_begin→wall_end;
    same clock domain as the `[PREFILL-OP] t=` field and /proc/uptime.
  - `free_b`/`free_e` = `cudaMemGetInfo` FREE bytes at chunk begin/end — **a drop =
    allocation inside the chunk**.
  - `h2d_kb` = delta of the program-wide `copy_i32` byte counter
    (`h2d_trace_bytes()`, counts BOTH pinned and legacy paths) across the chunk =
    staged H2D KiB inside the chunk.
  - Chunk lifecycle: sampled in `wall_begin` (h3_tb, free_b, h2d_b) / `wall_end`
    (h3_span, free_e, h2d_e); the `[PREFILL-HOST]` line PRINTS at the NEXT chunk's
    gate (begin_chunk drain) — request's last chunk prints at `finish()` after the
    request-end sync. `chunk=1` = first chunk of a request. 4 rank threads
    interleave on stdout — filter `rank=0 `.
- Companion lines used: `[PREFILL-OP] rank= chunk= tok= t= wall= gemm= ar= body= gap= ngemm= gemm_over=`
  (t = mono ms; correlator joins this to the sampler), `[PREFILL-SUM] … mean_ms wall= gemm= ar= body= gap=`
  per request end, `[PREFILL-BODYSUM]`/`[PREFILL-DN3-SUM]` (depth>=2/3).
- `NINFER_H2D_PINNED_STAGE` is a CANONICAL env (serve_10k.sh line, PLOG-059 cure),
  NOT part of the instrument — keep it ON (canonical posture the 305-329 ms finding
  was measured under); `h2d_kb` counts either way.

**Step 2 — bank law verified:** `/home/chris/artifacts_bin/ninfer-serve_05628598c977e42a.bin`
exists (139,456,840 B, mtime Sep 19 16:09); `sha256sum | cut -c1-16` = `05628598c977e42a`
== filename stamp. MATCH.

**Canonical env shape (serve_10k.sh exports, verbatim — the bin predates the promoted
line; no posture flags beyond this set):**
```
NINFER_ALLOW_NVFP4_TP2=1
NINFER_WORKSPACE_MIB=96
NINFER_DRAFT_VOCAB=/home/chris/dual_5060_ti_ninfer/tests/multi_gpu/data/qwen38_draft_vocab_ids.json
NINFER_MTP_TAIL_ASYNC=1
NINFER_H2D_PINNED_STAGE=1
NINFER_VOCAB_COUNT_DIR=/home/chris/vocab_counts
```
Boot line (serve_10k.sh shape): `$BIN $ART --port 8100 --devices 0,1,2,3 --prefill-chunk 128
--no-prefix-reuse --prefix-cache-capacity 256 --greedy --default-max-tokens 16
--spec mtp --draft-tokens 2 --allow-nvfp4-weights` — BIN = the 05628598 banked bin.
Standard body: `'Reply with exactly the single word BLUE. Context:' + ' alpha'*2072`
→ prompt_tokens=2075 (madmix parity cell asserts ==2075; W2_GRAPH_LEG_RECIPE §65:
16 steady 128-chunks + 27-token finalize), max_tokens=128 per this WO.

**Runner STAGED:** `tools/v340l/wo_firstchunk_leg.sh` (bash -n clean). Adds over the
banked protocol: sha16 gate, stand-down guard on a pre-existing serve (window law),
df guard, BLUE sanity abort, 6x plen-2075 mt=128, per-probe PROBE rows, OPTRACE
rank-0 extract, correlator run **with a units guard** — `t=` is steady_clock MS but
the sampler mono column is /proc/uptime SECONDS; the banked correlator was only
parser-dry-run (PERF_LOG_AMD.md:511), never joined live, so the script auto-detects
a dead join (max dt_s > 5 s) and re-runs on a mono×1000 converted CSV, banking both.


### 2026-09-20 10:5x — COORDINATOR CLOSE (checkpoint law: desk stalled post-leg, nudge unanswered; data complete — close executed from banked receipts; if the desk wakes, its report joins as addendum)
- **PREMISE REFUTED (the finding that created this WO):** "chunk 1 of EVERY request pays 305-329 ms hot / ~900 ms cold" is FALSE on the current stack. The instrument run (bin 05628598, ws96 era-faithful, 6x plen-2075 mt128 + sanity, 11 request segments): the ~918-922 ms pre-chunk latency appears on the FIRST TWO multi-chunk requests only (req3, req6); req7-req11 — five consecutive multi-chunk requests under RISING thermal load — show chunk-1 pre-chunk latency **0.0 ms flat** (correlator rows, requests_segmented=11).
- **NAMED MECHANISM (honest split):** (a) the cost is a BOOT-WARMUP TRANSIENT, not a per-request tax — pre-chunk latency on early big requests (first-touch/initialization of request-scale structures; KV-pool page warm-in class), cured by repetition by the 3rd multi-chunk request; (b) the one-time VRAM event is separate and small: 30 MiB / 67.8 ms span on the boot's first request, then ZERO drops; (c) in-window chunk-1 gap is only 25.2 ms (correlator: 990.3 = gemm 807.5 + ar 108.5 + body 49.1 + gap 25.2); (d) h2d <= 4 KiB throughout — the pinned-staging cure (PLOG-058/059) holds; nothing left on the H2D path.
- **REPRICING/CLOSE:** VALUE_QUEUE #3 ("first-chunk request-start cost, ~1.8% of request wall") collapses to ~2 s ONCE PER BOOT (~0.03% amortized over any real session) — CLOSED, lever dead; a pre-touch fix has nothing to sell. The original 305-329 ms hot readings were mid-warmup transients of the pre-cure era mis-attributed to every request. The REAL chunk-1 terms are gemm 807 ms + ar 108 ms — i.e., the request-start "gap" the census saw was the gemm column itself plus the now-cured stalls; no separate first-chunk mechanism exists on the current stack.
- RECEIPTS: results/amd/firstchunk/{firstchunk_summary,correlate,probes,optrace_rank0,serve_firstchunk,restore_serve_fast}_20260920_103325.* (all on this branch). Canonical restored (union bin) + liveness verified well past the +3 min rule. Stage: 6/6 probes behavioral PASS (walls 19.6->32.0 s, pure thermal).

### 2026-09-20 11:0x — COORDINATOR ACKNOWLEDGMENT: the desk's final report SUPERSEDES my checkpoint reading — the 918/922 ms "transient" is itself an artifact (stale-drain: ar byte-equal to the preceding MTP-redo line; walls exclude real idle). Final close state: the PLOG-071-era per-request first-chunk gap DOES NOT EXIST on the era-faithful posture (ttft-minus-steady: +1/-7/-5/+4/+8/+14/+6 ms across all 7 requests, cold and hot; chunk-1 device wall FASTER than each request's own steady mean on all six long requests; h3_* exonerate host-span/dfree/h2d/pool-restored — host_span 5.9-9.8 ms flat, Δfree 0.000 MiB at hot chunk 1). Residuals priced and harmless: ~208 MiB one-time pool growth (frozen after request 1); p1's single 1037 ms finalize stall = legacy class on this pre-MTP_TAIL_ASYNC bin only, never recurs. Bonus RED banked for the w7 lane: w7_gap_clock_sampler.sh heredoc swallows its own rocm-smi pipe (header-only clock_*.csv is the receipt). VALUE_QUEUE #3: CLOSED, lever dead — the historical "first-chunk gap" was the uncured pageable-H2D enqueue stall + extraction artifact, both since cured/retired.
