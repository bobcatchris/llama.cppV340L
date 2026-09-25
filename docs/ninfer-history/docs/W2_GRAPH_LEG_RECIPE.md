# W2 GRAPH LEG RECIPE — fire the NINFER_PREFILL_GRAPH lever in one window

Build desk: `amd/wo-w7-graph` (no-GPU desk; the GPU legs below belong to the window owner).
**BANKED BIN: `/home/chris/artifacts_bin/ninfer-serve_75957b8c62822b8a.bin`**
(sha256 `75957b8c62822b8a9e2844197c8b228a87570529c4849e07986632a4a0574d0a`, 137,654,456 B,
built 2026-09-18, branch `amd/wo-w7-graph` @ f0356f538+recipe, `-O3` gfx900 ROCm 6.2.0).
Spec of record: `docs/amd/W2_GAP_CAPTURE_desk.md` §3/§5 (capture design + build order step 3);
charter: `PREFILL_PLAN_REV2_2026-09-18.md` §4 "capture medicine". R2 citation: the P1 row
(`docs/amd/PERF_LOG_AMD.md` ~:240, wall 1642.3 = gemm 798.8 + ar 70.7 + body 645.1 + gap 122.7)
and the W2 per-chunk decomposition (steady gap ≈ 20 ms + unbr ≈ 53–65 ms = the honest ~85 ms/chunk
capture prize; the 122.7/180 means are a chunk-1 artifact + finalize anomaly — replay touches NEITHER).

## 0. What the lever is (one paragraph)

`NINFER_PREFILL_GRAPH=1` captures the STEADY non-final prefill chunk's 64-layer body (the single
`run_layers(x, Phase::Prefill, tap)` call inside `TextContext::prefill_impl`) into a CUDA graph on
the first eligible chunk (request chunk 2: `base==len==128`), instantiates, launches that same
chunk through the graph, and REPLAYS the graph for every subsequent eligible chunk (len==128,
non-final, text-only, no rewrite checkpoint). The chunk head (pageable ids H2D, positions fill,
embedding) and the whole chunk tail (final rmsnorm, lm_head/argmax, the entire MTP block incl.
`gqa_kv_append` and the finalize sync+D2H) stay eager every chunk. Decode-graph machinery
(`src/core/decode_graph.cpp`) is the template; decode graphs already run default-ON at TP4 with
`ncclAllReduce` captured inside — this boot shape has proven RCCL-in-capture on ROCm 6.2/gfx900.

Files (branch `amd/wo-w7-graph`): `src/targets/qwen3_6/impl/runtime/prefill_graph.h` (new: gate
parse, no-go envs, owner registry, capture key + arena anchors), `text_context.h` (cache members),
`text_context_impl.h` (prefill_impl capture/replay orchestration + run_layers tracer suspend +
PrefillOpTrace::gemm_suspend), `core/multi_gpu/tp_group.{h,cpp}` (`heartbeat_ping`).

## 1. The bin is already banked (BANK-BEFORE-RELINK law, runbook §4)

The desk banked the built binary (path + sha above). If the window owner rebuilds from this
branch instead, re-bank by sha16 BEFORE any relink:

```bash
cd /home/chris/worktrees/amd-wo-w7-graph
SHA=$(sha256sum build-hip-amd/apps/ninfer-serve | cut -c1-16)
cp -p build-hip-amd/apps/ninfer-serve /home/chris/artifacts_bin/ninfer-serve_${SHA}.bin
sha256sum /home/chris/artifacts_bin/ninfer-serve_${SHA}.bin   # record the full sha in the row
```

Pinned boots load from the bank path (`/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin`),
never from `build-hip-amd/apps/` (post-clobber law 2026-09-13).

## 2. Legs — A/B, same bin, same prompt, cool window

Copy `results/amd/coherence/W7_boot_promote_gqa_serve.sh` (W7 known-good leg, on `amd/main` and
`amd/wo-w7-body`) and change EXACTLY three things:

1. `LANE=` → this lane's worktree; log/resp names → `W2_graph_{a,b}_*`.
2. Env: add `export NINFER_PREFILL_GRAPH=1` for the **B leg**; leave it UNSET for the **A leg**
   (the A leg is the byte-identity + denominator baseline; unset must be today's path bit-for-bit).
   Keep the rest of the known-good env verbatim, i.e. `NINFER_PREFILL_OPTRACE=2` (the P2 env the
   finalize desk uses) plus `NINFER_ALLOW_NVFP4_TP2=1 NINFER_WORKSPACE_MIB=96
   NINFER_DRAFT_VOCAB=... NINFER_VERIFY_LAYER_TRACE=1 NINFER_TP2_TIMING=1 NINFER_TP2_OPTRACE=1
   NINFER_LMHEAD_ARM=simt NINFER_LMHEAD_TRACE=1 NINFER_TILED_TRACE=1 NINFER_VERIFY_TAIL_TRACE=1`;
   `NINFER_GDN_SIMT/GGATE_SIMT/GQA_SPLITK` deliberately unset. WARNING: do NOT set any env from
   the lever's no-go list in §4 of `prefill_graph.h` (`NINFER_MB_HASHPT`, `NINFER_HKV_DBG`,
   `NINFER_ZBIS`, `NINFER_MB_NANPROBE`, `NINFER_MB_PARTHASH`, `NINFER_COH_OPS`, `NINFER_COH_ALLOC`,
   `NINFER_C_LAYERS`, `NINFER_C_DUMP`, `NINFER_GQA_SPLITK`, `NINFER_MB_DBG`) — any of them
   disables capture (one loud `[PREFILL-GRAPH] disabled:` line) and the leg degenerates to A.
3. Probe: plen-2075 first+only, greedy, mt 64 (or the desk's standard), e.g.
   `'Reply with exactly the single word BLUE. Context:' + ' alpha'*2072` — verify
   `prompt_tokens` from the resp JSON (the fixed prefix carries a few tokens; nudge the repeat
   count until `prompt_tokens=2075`, i.e. 16 steady 128-chunks + a 27-token finalize).

Retire/boot exactly as the known-good script does (bank-path-anchored `pgrep -f`, TERM→KILL, the
`rocm-smi --showuse` four-dies-idle check, `setsid nohup "$BIN" ... --devices 0,1,2,3 --prefill-chunk
128 --no-prefix-reuse --prefix-cache-capacity 256 --greedy --spec mtp --draft-tokens 2 ...`).
**Get the written GPU grant first (intercom); the boot manifest row (argv + bin sha + window)
posts BEFORE spawn and the release row AFTER, per runbook §2 law 3.**

Expected boot-line evidence (B leg, serve log stderr):
```
[PREFILL-GRAPH] captured steady chunk body: rank=R len=128 arena=[....] bytes=.. — replaying subsequent chunks
```
exactly once per rank (4 lines), at chunk 2 of the first request.

## 3. What to grep and the acceptance bar

Per-request, rank 0 (filter `rank=0`), B leg vs A leg, same prompt:

```bash
grep -a "PREFILL-OP] rank=0" "$LOG"      # per-chunk wall / gemm / ar / body / gap rows
grep -a "PREFILL-SUM] rank=0" "$LOG"     # request means
grep -a "PREFILL-GRAPH" "$LOG"           # capture/replay counters + any disable line
```

- **PRIMARY ENDPOINT: steady-chunk wall** = mean `[PREFILL-OP] rank=0` `wall=` over chunks 3–16
  (chunk 1 is the accounting artifact; chunk 2 carries the one-time capture cost; chunk 17 is the
  finalize anomaly — none of the three are the lever's denominator). Acceptance (pre-registered,
  W2 spec): **B − A in [−90, −40] ms/chunk with byte-identical greedy text.**
- **Byte identity**: A-leg and B-leg resp JSON `choices[0].message.content` must be byte-equal
  (same prompt, `--greedy`, temp 0). ANY mismatch = kill (revert the env, file the row).
- **Kill if wall improvement < 20 ms/chunk** (same discipline as every pre-registered bar).
- READING THE B-LEG COLUMNS: for graphed chunks (3–16) `gemm=/ar=/body=` read **0** and
  `gap==wall` — the in-body event pairs are deliberately suspended inside the graph
  (kill-switch 4; re-recorded event nodes would race the drains). `wall=` stays truthful (the
  wall pair brackets the chunk from OUTSIDE the graph). The B-leg `[PREFILL-BODY]` rows print
  zeros for graphed chunks for the same reason. The A leg keeps full columns; the A/B comparison
  is wall-vs-wall.
- `[PREFILL-SUM]` means mix eager chunk 17 into both legs equally — use the per-chunk rows for
  the endpoint, not the request mean.
- Capture/replay counters: `grep -a "PREFILL-GRAPH"` also surfaces `eager fallback` causes —
  any `permanently disabled` line means the lever silently turned OFF mid-leg: bank the log, stop,
  read the named reason, do not re-leg without understanding it.

## 4. What the GPU leg MUST verify (the no-GPU desk could not)

Honest list, one by one — these are the falsifiers the build desk could not run:

1. **Capture legality of every op in the 64-layer body at TP4.** Argued, not measured, per op:
   RCCL `ncclAllReduce` inside capture is proven by shipped decode graphs (same
   `allreduce_local_bf16` call, world=4, default-ON) — but at the 1.31 MB prefill AR size
   (655,360 elems) it is only ARGUED size-independent, never measured (W2 §4 risk). The GDN
   trio / conv / rope / rmsnorm / prompt-route GQA / tiled GEMM launches are plain kernel
   enqueues, but the FIRST capture attempt is the only proof. Failure mode: capture throws →
   `[PREFILL-GRAPH] permanently disabled` → boot continues eager (by design); the leg then
   measures nothing — file the named reason.
2. **Replay numeric identity.** Byte-identical greedy text (acceptance above). The desk's proof
   is by-construction (device-driven inputs behind baked pointers; Prompt-route GQA is
   envelope-independent at width=128/batch=1); only the GPU leg can show the model agrees.
3. **The arena anchor holds under the real allocator.** `work_.used()` at run_layers entry must
   be chunk-invariant for len=128 (kill-switch 1). If the `[PREFILL-GRAPH]` log shows repeated
   eager fallbacks (counter in the boot log), the determinism assumption is wrong — bank and report.
4. **Actual milliseconds.** Nothing here measured time. The −40..−90 ms/chunk band is the W2
   arithmetic (unbr 53–65 + gap 20 − head/tail costs the graph cannot touch); the GPU leg decides.
5. **Watchdog interaction under the oneshot env.** Analytic result (ar_liveness.h
   `ar_watchdog_scan`: kind=2 needs ODD heartbeat seq — an idle rank never false-kills), not
   exercised. `heartbeat_ping` fires per replay launch when `NINFER_TP_ONESHOT_AR` is set.
   If the leg runs with that env set, confirm no `[AR-WEDGE-WATCHDOG]` during steady chunks.
6. **Multi-request replay.** The cache persists across requests (st.text is the backend's
   persistent card); request 2+ replay without recapture. Verify the second request's
   `[PREFILL-GRAPH]` log shows NO new capture line and its output is still correct.
7. **Lockstep at TP4 under replay** — all four ranks capture at their own chunk 2 and replay
   thereafter; per-chunk `ctx_.synchronize()` re-syncs ranks between graphs. Confirm no
   desync/timeout class appears (the W4 lesson): a wedged replayed chunk has NO 2 s B3 cover
   until the chunk-end sync (same posture as shipped decode graphs) — the leg window's own
   timeout is the bound of last resort.

## 5. Host-side proof already banked (no-GPU desk)

- `tools/guards/check_oneshot_w4_liveness_guard.py` → **GREEN** (exit 0) on this branch: L1–L7
  clean + behavioral harness PASS (20 scenarios, incl. a real `_exit(72)` child). This cell pins
  the exact scanner semantics the KS3 analysis relies on (`ar_watchdog_scan` kills only on an ODD
  heartbeat seq) with my `heartbeat_ping` addition in the tree.
- `tools/guards/check_vram_refuse_constants.py` → **GREEN** (exit 0): 0 new refuse-constant
  instances. The lever adds no VRAM estimates — the capture region's arena slices come from the
  existing shape-math recipes; the replay reserve is a measured capture-time delta.
- `tests/test_decode_graph.cpp` (capture/instantiate/launch semantics) requires a CUDA device and
  self-skips (exit 77) without one; this desk is barred from device touch, so it was NOT run —
  it is a candidate for the GPU window alongside §4's legs (it exercises the same
  DecodeGraphDefinition/Executable pair my cache embeds).
- Full `ninfer-serve` target compile at `-O3` gfx900 (ROCm 6.2.0): clean, 0 errors; the only
  new diagnostic during development (a `-Wsubobject-linkage` note on the local suspend struct)
  was restructured away before the final link.

## 6. Kill/revert discipline

Env-gated, default OFF, byte-identical unset — a kill is `unset NINFER_PREFILL_GRAPH` and a row;
no code revert needed to return the lane to today's behavior. Record per R1 in the plan ledger
with the A/B bin shas (same bin for both legs — the gate is runtime).
