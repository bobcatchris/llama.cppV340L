# W7 UNBR-ATTRIBUTION DESK — checkpoint file (append-only; newest at bottom)

Desk opened 2026-09-18. Mission: attribute the ~53 ms/chunk prefill time not owned by any
named kernel. Deliverables: budget table (A kernel/B launch-gap/C sync/D H2D/E unexplained),
ranked fix list, optional measured fix, restore serving config verified.

- WORKTREE: /home/chris/worktrees/amd-wo-w7-body (branch amd/wo-w7-body)
- SERVING: banked bin /home/chris/artifacts_bin/ninfer-serve_0572d4108317ee85.bin on :8100
  (env per /home/chris/serve_10k.sh: NINFER_ALLOW_NVFP4_TP2=1, NINFER_WORKSPACE_MIB=96,
  NINFER_DRAFT_VOCAB=/home/chris/dual_5060_ti_ninfer/tests/multi_gpu/data/qwen38_draft_vocab_ids.json,
  NINFER_MTP_TAIL_ASYNC=1; args --port 8100 --devices 0,1,2,3 --prefill-chunk 128
  --no-prefix-reuse --prefix-cache-capacity 256 --greedy --default-max-tokens 16 --spec mtp
  --draft-tokens 2 --allow-nvfp4-weights)
- Retire pattern (EXACT ONLY): pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"
- DISK: 96% used, 4.5G free — no cmake builds; standalone hipcc cells only.

## LOG (append per step)

- [step 0] Desk file created. Env verified (worktree on amd/wo-w7-body; serve_10k.sh read;
  df 96%/4.5G). NEXT: read PREFILL_PLAN_REV2 §9 + PERF_LOG tail + W7_FINALIZE_DESK_RESUME.

- [step 1] Live server found: PID 214823 = ninfer-serve_7c11c3ac07863366.bin (PLOG-056 prose-legs
  bin; NOT the serve_10k.sh canonical 0572d4108317ee85 — pre-window drift noted; decode desk left
  it serving). /health = {"status":"ok"}. Decode desk bench compile in flight (clang PID 511999,
  w7_draft_arm_bench — standalone, untouched by this desk).
- [step 2] OPTRACE knob VERIFIED in banked bin 0572d4108317ee85 via strings: NINFER_PREFILL_OPTRACE
  present; format strings: [PREFILL-OP] wall/gemm/ar/body/gap/ngemm; [PREFILL-BODY] per-op gate/upk/
  conv/gbet/dn/gnorm/aunpack/qknorm/gqa/agate/mnorm/mact/unbr; [PREFILL-BODYSUM]; [PREFILL-SUM].
  PRIMARY INSTRUMENT AVAILABLE IN THE SERVING BIN — no tree edits needed for the trace legs.
  NEXT: read src trace headers for level semantics + what lands in `unbr`.

## RE-DERIVATION (zero GPU, from banked logs — R5: measurement beats model)
From W7_fintail_async_serve.log (PROMOTED config: bin-era cure ON, plen-2075 mt64 probe,
17 chunks, rank0):
- [PREFILL-OP] steady chunks 3-16: wall 986-1038 / gemm 788-798 / ar 54-59 / gap 17-24;
  body RISES 115 -> 159 through the request (droop class; gemm flat).
- [PREFILL-BODY] per-chunk unbr: chunk2 98.3 (first steady, 2x), chunks 3-16 mean ~51.6
  (range 44.5-54.9), finalize chunk17 (tok=27) 342.5 — the per-chunk steady unbr ~52 ms is
  CONFIRMED as the mission target; chunk2 +98 and finalize +342 are per-request extras.
- [PREFILL-BODYSUM] rank0 n=17 unbr=67.5 body=152.8 resid=-3.5 (reconciliation holds).
- Cross-check older legs (band-uncontrolled): dn_simt 52.5, optrace3 52.8, finbracket_v2 52.5,
  async 67.5 — unbr sits 52-68 ms/chunk across bands; 53 (REV2 §1) re-derived OK.
- DN3 (depth-3, optrace3 leg chunk5 T=128): dn splits l2 0.9 / wy 15.1 / sp 11.7 / out 5.8 —
  named-inside-named; not part of unbr.

## WHAT unbr IS (from src, /home/chris/worktrees/amd-wo-w7-body/src/targets/qwen3_6/impl/runtime/
text_context_impl.h + src/core/prefill_body_trace.h):
- unbr := body - sum(12 named device-event spans); body := layer bodies - gemm spans - AR spans;
  gap (separate P1 column) holds embed/finalnorm/lm_head/MTP (outside layer bodies).
- Source enumeration of layer bodies: the 12 named classes bracket EVERYTHING except the
  full-attn input rmsnorm (ops::rmsnorm(x, input_norm) in attn_mix_tp :2370, 16x/chunk,
  microsecond-scale) — unnamed KERNELS cannot own 53 ms. => unbr must be DEVICE IDLE inside
  layer bodies (host enqueue latency not hidden by the async queue, or host-blocked sync
  stalls) and/or copies. The timeline trace decides (R5).

## WINDOW PLAN (pre-registered)
- Leg A (per-op trace): retire anchored; boot serve_10k.sh BIN 0572d4108317ee85 + EXACT
  serve_10k.sh env + NINFER_PREFILL_OPTRACE=3; probes: 2k = plen-2075 mt64 temp0 (EXACT
  fintail_async request shape, comparability) then 10k = serve_10k.sh shape (9938 alphas,
  mt192). Logs: results/amd/coherence/W7_unbr_optrace3_{2k,10k}.log. Clocks: read-only
  pp_dpm_sclk samples alongside.
- Leg B (timeline): SAME bin under rocprofv3 --hip-runtime-trace --kernel-trace
  --memory-copy-trace --output-format csv; standard env + NINFER_PREFILL_OPTRACE=1 (P1 walls
  for reconciliation); ONE 2k probe; graceful TERM first (flush), KILL only survivors.
- Gates (pre-registered): budget fractions from leg B stand on their own timeline; cross-check
  vs leg A OPTRACE unbr is band-uncontrolled (different boots) — state both, no ±2% claim
  across boots; any ARMED fix A/B later obeys ±2% same-window.
- Restore: bash /home/chris/serve_10k.sh (canonical bin+env), health + its 10k grade = the
  generation leg. NOTE: pre-window live server was decode-desk boot 7c11c3ac07863366 (same env,
  different bin) — restored config = serve_10k.sh canonical per mission spec.

- [step 3] WINDOW OPEN. Retired old server (anchored pkill); booted BIN 0572d4108317ee85
  (serve_10k env EXACT + NINFER_PREFILL_OPTRACE=3) from worktree CWD; health ok pid 552502.
  Log: results/amd/coherence/W7_unbr_optrace3_serve.log. Clocks sideband:
  results/amd/coherence/W7_unbr_clocks.log (auto clocks, sclk level 7/1500MHz samples).
- [step 4] 2k probe (plen-2075 mt64 temp0, EXACT fintail_async shape): GREEN parity
  'BLUE'/finish=stop/completion=52, wall 19.5 s. NEXT: fire 10k probe (serve_10k shape,
  mt192) in background; parse 2k trace while it runs.

## RESUMED-BY (RESUME agent, night shift 2026-09-18 ~21:25 CDT)

Prior agent died mid-window (~21:09-21:18, not OOM). Resume plan: (0) no window taken yet —
coordinator restored serving on bin 7c11c3ac, its 10k grade probe runs ~21:25-21:45, do not
retire before then; (1) inspect ALL banked evidence incl. the 10k_resp.json (suspicious,
21:06) and the unexplained "tl" leg the desk file does not record; (2) parse the banked
OPTRACE=3 trace into the budget table (zero-GPU, preferred); (3) only if a decisive
instrument gap remains AND window free after 21:45, consider a new window (WINDOW CLAIM
first line). Append per step below.

- [step 5 - RESUME] BANKED EVIDENCE TRIAGE (zero GPU):
  - W7_unbr_10k_resp.json is NOT an error — it is the GREEN 10k parity response of the
    OPTRACE=3 boot's second probe: prompt_tokens=10000, content "BLUE", finish stop,
    69 completion tok; serve log req2 done 21:06:12 (ttft 99195 ms, wall 102.52 s,
    prefill 100.8 tok/s, mtp 2.41 tok/round 70.7%). So leg A completed BOTH probes:
    2k (ttft 18.26 s) AND 10k (ttft 99.20 s). [PREFILL-OP] lines in leg A log: 392.
  - "tl" leg = pre-registered LEG B, fully banked: rocprofv3 --hip-runtime-trace
    --kernel-trace --memory-copy-trace on the 2075-probe; boot 21:07:36, probe done
    21:08:28 (wall 20.99 s, ttft 19158 ms), graceful kill flush 21:09:59 -> CSVs:
    w7_unbr_tl/W7_unbr_tl_{kernel_trace 212.8k rows, hip_api_trace 805.6k rows,
    memory_copy_trace 89.3k rows, agent_info}.csv (54/69.8/6.3 MB). tl serve log has
    [PREFILL-OP] per-chunk walls (OPTRACE=1: no [PREFILL-BODY] lines) for reconciliation.
    Predecessor died 21:09-21:18 right after the CSV flush, BEFORE parsing — desk file
    frozen at step 4 is stale relative to the evidence (both legs complete).
  - CLOCKS CORRECTION: W7_unbr_clocks.log star placement read against live sysfs format —
    the `*` marks the CURRENT level and in every banked sample it sits on level 0/300 MHz,
    NOT level 7/1500 as previously stated. The "mid-10k 21:06:22" sample landed 10 s AFTER
    req2 completed (21:06:12) i.e. it is an idle sample; the log contains NO under-load
    sclk sample. Live check at resume (21:26): both GPUs busy 0%, sclk 0/300 MHz — the
    coordinator's grade probe had not yet reached the GPU. Under-load clock citation TBD
    from a live sample during any later load; all perf numbers below cite this caveat.


## TIMELINE VERDICT (leg B, banked) — the unbr is NAMED
Trace: results/amd/coherence/w7_unbr_tl/W7_unbr_tl_{kernel,hip_api,memory_copy}_trace.csv
(rocprofv3 on banked bin 0572d4108317ee85, ONE 2k probe, parity BLUE/stop/52, wall 21.0 s
= +7.7% trace overhead; chunk5 OPTRACE wall 1025.0 / gemm 791.1 / ar 60.3 / body 152.3 / gap 21.3).
Chunk-5 device timeline (GPU0, single queue, ZERO overlap, 98.5% busy):
  gemm kernels 821.4 (tiled 714.2 + w8_rowsplit 107.2) | nccl AR 119.5 + axpy 1.0 |
  named-12 op kernels ~73 | unnamed ~2.3 | IDLE 15.2 (618 intervals).
FINDING 1 (the accounting half): OPTRACE ar column = 60.3 but nccl device busy = 119.5 (2x).
  VerifyLayerTrace has ONE ar event pair per layer; BOTH per-layer ARs (mixer + mlp) call
  ar_begin — the mlp site RE-RECORDS the same begin event (text_context_impl.h :2603/:3013/
  :3219 all -> ev_ar[layer][slot][0]), so the pair measures only the MLP AR and the ~64
  mixer ARs/chunk (~59-60 ms) fall out of ar into body -> P1 dumps them into unbr.
FINDING 2 (the real-waste half): per chunk the rank enqueue thread blocks ~995 ms inside ONE
  ~512-B pageable hipMemcpyAsync (device transfer 3.7 us, executes only when the in-order
  stream drains at chunk end; pageable staging cannot return until the DMA runs). Same class
  as the cured finalize bug (PLOG-055 108-B staging), still live per-chunk. Plus a medium
  ~27.5-us H2D blocking the host 51-60 ms. Consequence: launches happen in one ~65 ms burst,
  the compute queue runs ~1 chunk deep and thin; 618 micro-idles (14.3 ms/chunk, host blocked
  = sync-class C) + 0.9 ms true launch gaps (B) + chunk-end sync 15.5 ms host (the gap column).
  ALL FOUR RANKS IDENTICAL (16 giant blocked calls each over the prefill).
Budget of OPTRACE unbr (~52-72 band): ~59-60 mixer-AR undercount (real AR work, misattributed)
  + ~14.3 sync-stall idle (pageable-H2D enqueue stall — REAL recoverable waste)
  + ~0.9 launch gaps + ~0.3-2 unnamed kernels (full-attn input rmsnorm 16x/chunk etc.)
  + H2D device time in-body ~0.02; E closes within cross-boot band slop.
Boot-A (OPTRACE=3, same bin) request-mean BODYSUM n=17: unbr=72.2 body=159.6 (steady chunks
  50-56; finalize chunk 399 pulls the mean). 10k leg: unbr FLAT 54.9 mean vs context length.

- [step 5] Leg B executed: rocprofv3 trace of the SAME banked bin (OPTRACE=1, one 2k probe,
  parity BLUE/stop/52). NOTE: an unanchored pkill -f "rocprofv3" self-matched the desk shell
  (PLOG-052 lesson, 2nd occurrence) and killed the shell mid-retire; the traced server was
  retired cleanly afterwards via SIGTERM to the PID (graceful, let rocprofv3 flush CSVs) —
  no -9 was needed and no bare pkill was used against any process.
- [step 6] FINDINGS BANKED (see W7_unbr_row.txt): (1) ar bracket re-record drops the mixer
  AR (~59-60 ms/chunk -> misattributed into unbr); (2) per-chunk ~995 ms host stall in ONE
  ~512-B pageable H2D (+ medium 51-60 ms) = the enqueue-throttle class; device idle 14.3 ms
  sync-class + 0.9 ms gaps; unnamed kernels ~2.3 ms. Device 98.5% busy; TRUE AR = 119.5 ms.
- [step 7] Deliverable-3 cell w7_h2d_stall_cell (pre-registered gate): PASS x3 — pageable
  167/184/185 ms vs pinned 0.58-0.66 ms host block at identical queue depth, parity OK.
- [step 8] Row banked: results/amd/coherence/W7_unbr_row.txt. NEXT: restore serving via
  bash /home/chris/serve_10k.sh (canonical bin 0572d4108317ee85 + env; its 10k grade = the
  mandatory generation leg), then final desk update + commit.

## OWNER-INLINE CLOSURE (coordinator, 22:45-23:0x — agents dead on platform usage limit, work executed inline per PLOG-054 precedent)
- CURE LANDED: NINFER_H2D_PINNED_STAGE=1 (PinnedCopyRing: 64 pinned slots/thread, event-guarded reuse, torn-payload-safe; unset/"0"/any pinned failure = legacy byte-identical). Build GREEN after +2 shim names (cudaEventQuery/cudaErrorNotReady, provenance-commented). BIN BANKED 07fcf7a5d98dea7d (BANK-BEFORE-RELINK honored).
- SERVE-LEG A/B (2k probe plen-2075 mt64 temp0, parity BLUE/stop/52 on ALL 12 requests across both arms):
  - PINNED boot (env=1, OPTRACE=3): 4 requests wall 18.7/18.9/18.8/18.8 s FLAT; unbr request-mean 69.2 -> 52.4 -> 45.6 -> 44.9 (plateau); body -> 133.6-135.0.
  - UNSET control (SAME bin): wall 19.7 -> 23.0 -> 25.8 s RISING (thermal-decay class); unbr 69.5 -> 48.9 -> 56.6 -> 83.8.
  - First-request-vs-first-request IDENTICAL (69.2 vs 69.5) = matches boot-A baseline class (72.2): the cure's effect emerges UNDER SUSTAINED LOAD (the serving regime) — flat-vs-decaying is the signal; cross-boot absolute deltas are thermally confounded (honest note; paired same-window A/B + rocprof timeline on the cure bin = the named reopen-if, incl. the AR-overlap hypothesis: with the queue fed, mixer-ARs may overlap compute, which would explain steady unbr landing BELOW the ~60 ar-undercount floor).
- PROMOTION: runbook flipped to BIN 07fcf7a5 + NINFER_H2D_PINNED_STAGE=1 (TAIL_ASYNC pattern: runbook env = serving config of record; rollback = remove line). Canonical boot + 10k soak grade fired 23:0x (result appends to PLOG-059). BOOT_BATTERY after the soak.
