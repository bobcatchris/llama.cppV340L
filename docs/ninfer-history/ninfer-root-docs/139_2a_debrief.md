# 139 — 2a Debrief: Batched KVarN Serving Wiring (agent1)

**Status at writing:** Step 1 committed (`59affb99`); Steps 2–5 implemented in the
working tree, buildable, live validation **blocked on one remaining serve-path
defect** (garbage-token exception → runner barrier deadlock, details §4).
GPUs released, no server running. Branch `wo/2a-batched-serving`, worktree
`~/ninfer/worktrees/wo-2a-batched-serving`, base `wo/kv-uniform` @ `c5f598b9`
(note: base has since moved to `5b6f027b` per coordinator — see §7.5).

---

## 1. The task (docs/139, verbatim scope)

Make "batched decode" a *served* KVarN feature on TP2. The tested batched runner
`run_tp2_requests_batched` (`tp2_backend.cpp:2072`) was dead code in production —
the engine ran every request through single-sequence `run_tp2_request`
(`tp_engine.cpp:223`). Required after the WO:

- `--max-concurrency N` server + N concurrent KVarN (KVARN_K4V2) requests route
  through `run_tp2_requests_batched` (true MultiBatch attention).
- Per-lane stats (acceptance, tok/round) reported, mirroring single-seq `TpRunStats`.
- Overflow (`N > max_concurrency` or MTP ring `need > have`) rejected/queued with
  a clear error — no silent truncation.
- A decode_guard / `run_ci.sh` batched cell fails CI on a serving-path regression.
- **N=1 must remain byte-identical.** No attention-kernel changes. Worktree only,
  commit per step, live e2e proof for every server-facing step (docs/50 §7.1).

Execution order: 5 steps — (1) engine batch dispatch, (2) per-lane context+stats,
(3) serve-path admission, (4) live e2e proof (MTP on/off, mixed ctx, determinism),
(5) CI batched cell with negative test.

---

## 2. What is delivered (per step)

### Step 1 — engine batch dispatch: **COMMITTED** (`59affb99`)
- `TPEngine` ctor now plumbs `options.max_concurrency` into `TpBackendOptions`.
  **Finding:** it was never plumbed before — the backend always built with
  `max_concurrency=1`, so the entire per-lane machinery (lane KV allocs, GDN
  slots, batched workspace) was dead configuration in the serve path.
- `TpSubmission::wait()` restructured into a leader/follower rendezvous:
  requests register a `TpBatchMember` (config + token callback + per-lane
  cancellation view) in `SharedState::batch_waiting`, then contend on
  `SharedState::mutex` as before. The lock holder (leader) collects the parked
  set for a bounded window (`batch_window_ms()`, default 250 ms,
  `NINFER_BATCH_WINDOW_MS` to tune, `NINFER_BATCH_DISABLE=1` kill-switch),
  then dispatches: 1 member → the untouched single-sequence call (N=1
  byte-identical by construction); ≥2 eligible members →
  `run_tp2_requests_batched` with per-lane cancellations; per-lane `TpRunStats`
  delivered back to each member's thread via the mutex hand-off.
- Eligibility guard: KVarN backend, `max_concurrency > 1`, `temperature == 0`
  (both runners are exact argmax at temp 0 — sampled requests stay single-seq),
  no lookup drafts, uniform `mtp_k`. Overflow is requeued (logged), never dropped.
- Serve-path admission (Step 3 scope, implemented here): reuses the runner's
  MTP ring arithmetic (`need = lanes*T` vs `have = 2*lanes + 2k + 1`); batched
  MTP unavailable when the ring geometry cannot carry a 2-lane batch (stated in
  the log, falls back to single-seq).
- `run_tp2_requests_batched` gained an optional `lane_cancellations` parameter
  (null = historical shared-view semantics; BATCHTEST callers unchanged): lane b
  stops on its own OutputSession EOS/limit or client disconnect without
  affecting other lanes.
- Prefix-cache hygiene: the batched worker start invalidates
  `st.cache_valid / cached_tokens / kvarn_prefix_snap / gdn_ckpt_*` — the
  batched lane-0 prefill reuses the pool row / slot 0 / workspace tile that the
  single-seq prefix restore trusts (the "append jumped pages" failure class).
- TextContext per-run state hygiene: batched worker start clears
  `prefill_rewrite_checkpoint_frontier`, `mtp_proposal_extent`, `decode_step`
  (the single-seq runner sets/clears these per run; the batched runner
  previously inherited whatever the last single-seq run left bound — e.g. the
  serve warmup's rewrite frontier would split the batched prefill at a stale
  absolute offset).
- **Tests passed:** clean build (CUDA 13.1); BATCHTEST W=2 bit-exact, MTP k=3
  PASS and plain (no-MTP) PASS.

### Step 2 — per-lane context + stats: **implemented, live validation pending**
- Runner now fills per-lane `prefill_ms/prefill_tps`, `decode_seconds/
  decode_tps`, `total_seconds`, `rounds`, `accepted_drafts`,
  `mean_a_per_round`, `acceptance_rate`, `tokens_per_round`,
  `prefix_reuse_path=FullReset` (rank-0-owned counters, read post-join).
- Engine logs one line per lane after each batch, e.g.
  `[tp2] batched lane 0: acceptance=0.43 tok/round=2.29 rounds=112 gen=256 ...`.
- **Root-cause fix landed here** (see §4.2): batched draft-head output buffer
  `mb_prop` was sized `{nv_l=124160, N}` but the draft head has **20480
  rows/rank** (`n_prop_vocab`); the W8 GEMM sizes its grid from `out.ne[0]` →
  6× too many row blocks → ~530 MB OOB weight reads. Fixed to
  `{n_prop_vocab, N}` (mirrors single-seq `proposal_logits`).

### Step 3 — admission: **implemented** (inside `run_batch_dispatch`, see Step 1)
- Not yet live-tested with `max_concurrency + 1` concurrent requests.

### Step 4 — live e2e proof: **script ready, not run**
- `tools/smoke/serve_batched_measure.sh`: {MTP on, off} × {2-concurrent,
  sequential baseline} at mixed context, greedy determinism repeat, sampling
  fallback probe; writes `results/batched_serving_<ts>.json` with config,
  nvidia-smi clocks, per-lane stats, decode rates.

### Step 5 — CI batched cell: **script ready, not wired, not run**
- `tools/smoke/serve_batched_ci.sh`: B0 sequential baseline; B1 concurrent pair
  must produce the `dispatched 2-lane batch` log line + per-lane stats + outputs
  byte-equal to B0; B2 overflow (3 concurrent) all served; B3 negative —
  `NINFER_MB_SERVE_MUTATE=1` (mutation hook added to the batched MTP commit)
  must produce a detectable mismatch, else the cell is blind.
- **Not yet** added to `run_ci.sh`.

---

## 3. Infrastructure / environment problems (and fixes)

1. **CUDA toolkit:** default `nvcc` is 12.9; the build needs
   `cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc`. Also
   `-DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON -DCMAKE_BUILD_TYPE=Release
   -DCMAKE_CUDA_ARCHITECTURES=120a` (tests+serve binary; default OFF).
2. **Disk full** (228 GB root at 100%) during the test build; freed ~1.5 GB of
   stale `/tmp` profiling dumps (`pf*_kvarn.sqlite`, `nsys_kvarn250.nsys-rep`,
   Aug 30). **The disk is still ~99% full** — the next agent should budget for
   this (a full `--full` CI run writes several GB of logs).
3. **GPU contention:** agent2 (docs/141, 4a/4d) was running concurrent GPU
   batteries. Coordinated over intercom; protocol that worked: treat *any*
   foreign `ninfer` process holding a GPU context as a guard; poll
   `nvidia-smi --query-compute-apps`; take gaps; announce windows. One OOM race
   happened (my test vs their D4 start) — harmless, avoid by announcing.
4. **gdb cannot attach** (`ptrace_scope=1`): to debug a *hang*, launch the
   server **under** `gdb -batch -x script` with `handle SIGINT stop print`,
   reproduce, then `kill -INT <gdb pid>` to dump all thread stacks. This worked
   and is the recommended technique here.

---

## 4. The debugging story (the important part)

### 4.1 Timeline of the serve-path failure
1. First live batched attempt crashed at warmup with `cudaErrorIllegalAddress`
   in the batched draft-head path. **Cause:** my dispatcher drained the queue
   but never removed *itself* from it — the leader re-took its own member, so
   the batch was `[self, self]` and two lanes wrote one request's state. Fixed
   with a dedupe in `take_if_eligible`. (The work order's live-testing standard
   caught this immediately; it passed no kernel test because it is pure
   dispatch logic.)
2. Second attempt: a *real* 2-lane batch dispatched and crashed with
   `cudaErrorIllegalAddress` — surfacing at `text_context_impl.h:826` /
   `w8_rowsplit_gemm_simt.cu:39` depending on `CUDA_LAUNCH_BLOCKING`.
3. Harness could not reproduce: BATCHTEST W=2 MTP passed with short prompts,
   long prompts, `NINFER_MB_SEQ_FIRST`, `NINFER_MB_REF_PREFIX`, 1 GiB
   workspace, ctx 80000 — all PASS (one long-prompt harness run throws a
   separate pre-existing "text prefill chunk does not match its full prompt"
   error — see §6.4).
4. **compute-sanitizer localized it**: invalid 16 B reads ~0.3–1 MB past the
   draft-head weight codes, in blocks whose rows are ≥ 20480 — the W8 GEMM grid
   is sized from `out.ne[0]`, and `mb_prop` was `{124160, N}` for a 20480-row
   head. Fresh harness layouts left the OOB region mapped (silent garbage);
   the serve layout faults. **Fixed** (`n_prop_vocab`).
5. After the fix, BATCHTEST W=2 MTP passes, plain batched serving WORKS
   end-to-end (2-lane dispatch, 256 tok/lane, per-lane stats, no errors).

### 4.2 The remaining defect (blocks Steps 2–4 live validation)
Serve batched-MTP now **hangs**: dispatch succeeds, both lanes commit t0
(callbacks fire), then at the first MTP round commit a lane callback delivers a
garbage licensed token (`-1065828196` — looks like a float bit pattern);
`OutputSession::preview` throws `generated token is outside the checkpoint
vocabulary`; the rank-0 worker catches, sets `should_stop`, exits — **but the
other rank is blocked inside `sync_bar.arrive_and_wait()`**, which
`should_stop` cannot wake. Result: the runner never joins, both HTTP requests
time out, the server stays up but is wedged for new work on that mutex chain.

Evidence chain: gdb thread dump under `gdb -batch` (Thread 19 = leader HTTP
thread = rank-1 worker parked in the barrier; **no rank-0 thread exists**);
added temporary `[MB-RANK0-ERR]` print →
`generated token is outside the checkpoint vocabulary: -1065828196`.

So there are **two distinct problems** here:
- **4.2a (root):** garbage licensed tokens (`lic_h`) in the serve batched-MTP
  path, first round, after a prior single-seq MTP warmup. Diagnostics in place:
  `NINFER_MB_LICDBG` dumps `acc`/`lic[4]` per lane for steps ≤ 3.
- **4.2b (robustness):** the `std::barrier` deadlock class — ANY single-rank
  exception between barrier arrivals strands the other rank permanently and the
  exception is never delivered. This is pre-existing runner structure, but the
  serve path makes it fatal (a wedged server, not a clean 500).

### 4.3 Separate finding: batched-MTP vs single-seq-MTP bit-exactness gap
Independent of the crash: on longer prompts (~70 tokens, ctx 80000) the harness
shows batched-MTP vs sequential divergence at a late token (lane 0: batched
98936 vs seq 98012 at position 52), reproducible with `NINFER_MB_FORCE_B1=1`
and on a fresh backend, i.e. not MultiBatch column interaction and not state
pollution. The M0 phase gate shows all rounds CONVERGED except one
(gen53: `batched F=133 a=98936 | ref F=133 a=98012`); the draft chain differs at
that round (batched d1=98012 vs ref d1=96983). Looks like an argmax near-tie
broken by a numeric parity difference in the batched target-verify/draft chain
vs the single-seq path. Note: this may be *related* to the mb_prop bug (the
oversized buffer fed the AR-chain argmax) — **re-test after the fix**; the
re-test had not run at debrief time. Evidence preserved: `/tmp/wo2a_gate`
(phase dumps) and repro commands in §6.3. Kernels are out of 2a scope; if it
survives the mb_prop fix, hand to the kernel/runner owners with that evidence.

### 4.4 Transient HTTP 400s
During some concurrent-fire attempts one of two curls got instant HTTP 400
"request body is not valid JSON" with a valid body; standalone retry succeeds.
Suspected stale-server/zombie on the port during the fast kill/relaunch cycles;
not reproduced against a clean single server. Watch port hygiene
(`pgrep -x ninfer-serve` + health check *before* launch, which the scripts do).

---

## 5. Current exact state
- Branch `wo/2a-batched-serving` @ `59affb99` (Step 1). Working tree: Steps 2–5
  code + fixes + temporary diagnostics; buildable (`cmake --build build -j 16`
  green with the CUDA-13.1 flags above).
- Temporary diagnostics currently in the tree (remove or deliberately keep
  env-gated before closeout): `[MB-RANK0-ERR]`/`[MB-RANK1-ERR]` catch prints,
  `NINFER_MB_LICDBG`, `NINFER_DRAFT_DBG`, `NINFER_BATCH_DBG`,
  harness env hooks (`NINFER_MB_REF_PREFIX`, `NINFER_MB_REF_CTX`,
  `NINFER_TEST_WS_BYTES`), `NINFER_MB_SERVE_MUTATE` (keep — Step 5 negative test).
- No server running; GPUs free.
- Validated green: build; BATCHTEST W=2 MTP + plain (post-fix); plain batched
  serve (2 lanes, 256 tok each, per-lane stat lines, 0 errors); N=1 warmup and
  single requests on the new engine path.
- Blocked: serve batched-MTP (§4.2a); everything downstream of it (Steps 2–4
  live proofs, run_ci wiring evidence).

## 6. Suggested next steps (ordered)
1. **Chase the garbage token** (§4.2a): relaunch the licdbg server
   (`--max-concurrency 2 --spec mtp --draft-tokens 3 --kv-dtype kvarn_k4v2
   --kv-capacity 100000 --max-context 80000`), fire 2 concurrent short greedy
   requests, read `[LICDBG]` (acc/lic per lane, steps ≤ 3). Decision tree:
   garbage on both lanes → inspect `mb_tgt_rm`/`v_tgt1`/`v_full` chain (verify
   argmax + allgather); lane-1-only → lane-view/row mapping; values that look
   like float bits → a bf16/int32 buffer confusion (the -1065828196 ≈ 0xC07A4C84
   pattern suggests exactly that — check whether `lic_pin`/`acc_pin` offsets
   alias anything when N=2 with the warmup-dirtied arena).
   Also re-run the harness long-prompt divergence check (§4.3) post-fix to see
   if the tie-flip was the same bug.
2. **Fix the barrier-deadlock class** (§4.2b) so a lane-level failure degrades
   to a clean per-request 500 instead of a wedged server. Two candidate shapes:
   (a) wrap lane callbacks so OutputSession throws are converted to a lane
   cancel (retire the lane, keep the batch) — this alone would have turned
   today's hang into a clean partial failure; (b) replace the bare
   `sync_bar.arrive_and_wait()` with an abort-aware wait (bounded waits +
   `should_stop` re-check loop) — larger change, do (a) first.
3. **Finish Step 2–4 live validation** (serve_batched_ci.sh cells B0–B2,
   serve_batched_measure.sh, ±0.5 pt acceptance vs single-lane, determinism
   repeat), then wire `serve_batched_ci.sh` into `run_ci.sh`'s fast gate next to
   the M4 MTP round gate, plus the B3 negative run.
4. **N=1 regression proof:** run `bash tools/ops/run_ci.sh` (fast gate) on the
   branch — all single-request suites exercise the N=1 path unchanged.
5. **Commits:** Step 2 (stats + reporting + mb_prop fix + hygiene), Step 3
   (admission — already in the dispatch commit scope, may fold into Step 2's or
   separate), Step 5 (cell + hook + run_ci wiring). Push each.
6. **Base moved:** `wo/kv-uniform` is now `5b6f027b` (4a/4b/4d closed). Rebase
   or merge the branch onto it before closeout; expect conflicts only in
   `tp_engine.cpp`/`tp2_backend.cpp` neighborhoods.
7. **Cleanup before closeout:** strip or consciously keep the temporary
   diagnostics (§5); keep `NINFER_MB_SERVE_MUTATE` (negative test depends on it).
8. **Report the §4.3 parity gap** (if it survives the fix) to the runner/kernel
   owners with the /tmp/wo2a_gate dumps and repro commands — docs/130/136
   claim bit-exactness, and this is the first known counterexample at near-ties.

## 7. Key numbers (measured this session)
| Check | Result |
|---|---|
| BATCHTEST W=2 MTP k=3 (post-fix) | PASS, bit-exact both lanes |
| BATCHTEST W=2 plain (post-fix) | PASS, 1.26× aggregate vs sequential |
| BATCHTEST W=2 MTP (pre-fix) | PASS on fresh layouts (masked bug) |
| Plain batched serve (2 lanes × 256 tok) | works; lane prefill 0.15 s, decode 8.17 s wall for the pair |
| Single-seq MTP serve (baseline, 256 tok) | ~57 tok/s decode, wall ≈ 4.6–4.9 s |
| Batched MTP serve | **hangs** at first MTP round commit (§4.2a) |
| Harness long-prompt MTP parity | diverges at token 52 (§4.3, re-test post-fix) |
| Warmup single-seq on lanes=2 backend | works; prefix capture/snap fine after hygiene fix |

## 8. Launch commands used (record)
```bash
# serve (batched validation)
./build/apps/ninfer-serve /home/intel/models/qwen3_8_27b.ninfer \
  --port 8095 --devices 0,1 --spec mtp --draft-tokens 3 \
  --kv-dtype kvarn_k4v2 --kv-capacity 100000 --max-context 80000 \
  --max-concurrency 2 --model-id qwen3.8-27b
# harness
./build/tests/ninfer_tp2_batched_decode_test --artifact /home/intel/models/qwen3_8_27b.ninfer \
  --tokens 64 --mtp 3 [--ctx 80000 --prompt-a ... --prompt-b ...]   # + NINFER_MB_SEQ_FIRST=1 etc.
# configure
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a
```
