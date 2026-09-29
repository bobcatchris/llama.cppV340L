# CONCURRENCY BENCH — N=1/2/4 concurrent streams on the MTP serve line (ready-to-fire)

**Date:** 2026-09-17 · **Seat:** no-GPU analysis desk (bench written, never fired by this desk)
**Runner:** `results/amd/coherence/concurrency_bench.sh` (this directory)
**Mission:** measure the M-wall on THIS stack — doc/optimizations/01 §3 predicts 4 concurrent
streams aggregate ~3–3.5× single-stream (M=4 costs ~1.2–1.3× M=1 per round). If it holds,
concurrency is the second budget line to 50 tok/s (EXTRACTING_NUMBERS Part C) alongside
acceptance (ACCEPTANCE_LEVERS_2026-09-17.md).

## 1. How concurrency routes (code facts, amd/tp4-cure tree 2026-09-17)

- `--max-concurrency` (default **1**, legal [1,8]; `src/serve/serve_options.cpp:467-468`) flows
  to the engine (`src/serve/generation_service.cpp:308`). Concurrent requests are PARKED
  (docs/139 2a `TpBatchMember`); the mutex leader runs every parked member — single-sequence
  when one is dispatchable, `run_tp2_requests_batched` when several are
  (`src/runtime/tp2/tp_engine.cpp:128-135`, call at `:444`).
- **The boot line MUST carry `--max-concurrency 4`** (or ≥ planned N): batched MTP admission
  throws `N > max_concurrency` (`tp2_backend.cpp:5508-5510`). Default mc=1 + a 2nd parked
  request = throw, not queue.
- **MTP batches cleanly through the batched runner** (docs/129 §7.5: verify [k+1,N],
  draft/AR [1,N], round ops batch-native). NOT dflash2's gates — those are separate
  (`tp2_backend.cpp:5391+`).
- Known batched+MTP limitations (embed in any read of the results):
  1. **GDN slot-ring admission:** need = mc·T ≤ 2mc+2k+1. At mc=4: k=2 fits (12≤13);
     **k=3 THROWS** ("batched MTP: GDN slot ring capacity exceeded", `:5503-5510`). The bench
     therefore runs k=2 — which is also the recommended serving k (REALK23 VERDICT).
  2. **Every lane must request MTP and disable prefix/lookup** (`:5513-5514`). The serve line
     already carries `--no-prefix-reuse`; the body must be plain chat (no tools/lookup).
  3. **Greedy-only**: batched decode is greedy ("allreduce_argmax is per-column", banner above
     `run_tp2_requests_batched`, `:5355+`). Fine — the recommended line is `--greedy`.
  4. **Fixed-k only in batched**: ngram-mod / adaptive / CONF_TAU wiring exists ONLY in the
     single-seq loop (ACCEPTANCE_LEVERS §c/§d) — the N≥2 arms measure plain fixed-k=2 MTP.
  5. **BF16 conc≥2 runtime reserve:** 1536 MiB/rank charged when `is_bf16_kv && mc>=2`
     (`tp_engine.cpp:1147-1149`, `tp2_budget.h:89`) plus `drafter_fixed_bytes(..., mc)` —
     auto-KV therefore resolves LESS capacity at mc=4 than the 63936-token mc=1 boots. Harmless
     for ~700-tok legs; **record the boot "[tp2] auto KV" line in the manifest** (the script does).
- Batch-formed evidence to bank per arm: "[tp2] batched decode: dispatched N-lane batch
  (mtp=on k=2 lanes=N)" (`tp_engine.cpp:441`) and per-lane "[tp2] batched lane i:
  acceptance=… tok/round=… rounds=… gen=…" (`:452`). If the dispatched line shows fewer lanes
  than fired (a lane arrived after the leader dispatched), that SPLIT is a recorded finding —
  stagger is 0.25 s to make splitting unlikely, not impossible.

## 2. Body — the count600 shape

chat completions, single user turn, `max_tokens=600`, `temperature=0` (server `--greedy`),
thinking defaults ON server-side — exactly the TIMING1 probe shape
(`TIMING1_serve.log:64`: msgs=1 max_tokens=600 greedy thinking=on; 67-tok prompt, gen=600,
acc 0.97, 13.87 tok/s engine-side at N=1). Body text: "Count from 1 to 200, one number per
line, no commentary." Each stream fires the SAME body.

## 3. Hard rules embedded in the runner (laws, not etiquette)

- **GPU grant:** refuses to run without `CONC_BENCH_GRANT=1` — the operator sets it only after
  the coordinator's written grant (intercom). "Clear to claim" is not a grant.
- **Foreign-context guard:** sources the canonical
  `/home/chris/dual_5060_ti_ninfer/tools/smoke/diag/gpu_guard.sh`; `gpu_refuse_if_busy` before
  boot; `gpu_spawn`/`gpu_kill_own` (PID-scoped) only. **Never a system-wide pkill.**
- **Bank law:** binary must be `/home/chris/artifacts_bin/<name>_<sha16>.bin`; the sha16 is
  recomputed and must match the filename stamp (refuse otherwise; `CONC_ALLOW_UNBANKED=1`
  escapes for a explicitly-blessed lane build, printed loud in the row).
- **KFD hygiene:** `rocm-smi --showpids` raw dumps pre/post (`CONC_kfd_pre.txt`/`_post.txt`),
  tab-tolerant read (BOOT_LAUNCH_RUNBOOK.md rule 4); final `KFD=0` verification.
- **Thermal discipline (<60 s bursts; THARM law, NIGHT_HANDOFF §3):** arm order N=1 → 2 → 4
  (coldest first), ≥150 s idle between arms, projected walls ≈ 46 / 39 / 53 s — each burst
  under the ~60 s knee. Clocks+temps sideband at 2 s through every arm
  (sysfs `pp_dpm_sclk` star line + hwmon edge, V340 cards auto-discovered; the same class of
  sideband as REALK23). **Sampler is killed by its own PID** (REALK23 correction: killing the
  parent nohup does not kill the loop). If the sideband shows an sclk collapse mid-arm, the
  arm is VOID — re-run with `CONC_MT=300` (half body, ~25 s walls).
- **Numbers law:** a row without (a) clocks sideband, (b) binary sha, (c) the [tp2] lines is
  not a number (EXTRACTING_NUMBERS Part C closing rule). Output byte-diff across streams is
  NOT required here (all four streams are distinct identical-prompt legs; instead record the
  per-lane acceptance + finished content heads).

## 4. Procedure (what the script automates)

1. Preconditions: `df -h /` printed (shared constraint), gpu_refuse_if_busy, KFD precheck raw.
2. Boot the CURRENT recommended serve line + `--max-concurrency 4` (NIGHT_HANDOFF §1 line,
   N7 `--allow-nvfp4-weights`, ws96, chunk128, no-prefix, prefix-cap 256, greedy, k=2):
   `setsid nohup env --allow-nvfp4-weights NINFER_WORKSPACE_MIB=96 <BIN> <ARTIFACT> --port 8100
   --devices 0,1,2,3 --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 --greedy
   --default-max-tokens 16 --spec mtp --draft-tokens 2 --max-concurrency 4`
3. Poll `/health` (≤180 s); bank the boot lines: auto-KV, preflight composition,
   "loaded 40960 draft vocabulary IDs", "draft output head ready (W8G32, 10240 rows, 53 MB)".
4. Arms N ∈ {1,2,4}: 150 s idle → start clocks sideband (PID captured) → fire N identical
   curls (0.25 s stagger) → wait all → stop sampler by PID → harvest:
   per-stream `done finish=… decode=NN.Ntok/s`, per-lane `[tp2] batched lane` acceptance/tok-
   per-round, dispatched-lanes line, wall from first fire to last done,
   **aggregate tok/s = Σ completion_tokens / wall** → `CONC_N{N}_row.txt` + responses banked.
5. Teardown: `gpu_kill_own $SERVE_PID`, verify no serve remains, KFD post raw dump == 0.

## 5. Read-out and verdict shape

- Calibration: the N=1 arm should reproduce the TIMING1 class (≈13–14 tok/s, acc ≈0.97 at
  counting shape; if it reproduces REALK-class prose numbers instead, the body is wrong — VOID).
- Verdict line: `aggregate scaling N=4/N=1 = X.XXx (M-wall predicts 3–3.5×)`, plus N=2/N=1.
- Interpretation guardrails: all arms share ONE boot — thermal integrator carries across arms,
  so the 150 s idles and arm ORDER are part of the measurement; compare only against the
  cold-first N=1 arm, never against REALK/TIMING1 absolute walls from other boots.
- Aggregate tok/s rides on top of acceptance: at counting-shape acc 0.97 the M-wall number is
  a pure-concurrency read; repeat the N=1/N=4 pair on the REALK prose body (`CONC_BODY=prose`,
  mt=300, sky-blue prompt per REALK23) only after the counting pair lands.

## 6. Manifest template (script emits the filled version)

```
== MANIFEST <UTC> CONC bin <sha16> (recomputed==stamp: yes)
grant: <coordinator written grant, intercom ref>
config: TP4 --devices 0,1,2,3 --spec mtp --draft-tokens 2 --max-concurrency 4 graphs-ON port=8100
        N7(--allow-nvfp4-weights) NINFER_WORKSPACE_MIB=96 --prefill-chunk 128 --no-prefix-reuse
        --prefix-cache-capacity 256 --greedy --default-max-tokens 16
boot: auto-KV <capacity line> | preflight <composition line> | draft-vocab <loaded line>
body: count600-class chat mt=600 temp=0 thinking=on (TIMING1_serve.log:64 shape), stagger 0.25s
KFD: pre 0 (CONC_kfd_pre.txt) / post 0 (CONC_kfd_post.txt)
== ARM N=1: wall Xs; streams: decode A/B tok/s; acc=; tok/round=; aggregate X tok/s; clocks CONC_N1_clocks.tsv
== ARM N=2: … dispatched 2-lane (mtp=on k=2 lanes=2) …
== ARM N=4: … dispatched 4-lane …
== VERDICT: N=2/N=1 = X.XXx; N=4/N=1 = X.XXx (M-wall predicts 3–3.5×, doc/optimizations/01 §3)
== RELEASE: sampler killed by PID (verified), serve killed anchored (verified), KFD=0
```
