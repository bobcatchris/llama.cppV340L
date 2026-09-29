# 143 — 2a Debrief Round 2: Batched KVarN Serving (agent1)

> **SUPERSEDED ON §4.5 AND STEPS 2/4/5 — read `docs/147_2a_debrief_round3.md`.** The §4.5
> "batched-vs-single-seq parity gap, isolated to thinking mode" below is **not** a batching bug
> and is **not** thinking-mode specific: replaying the exact chat-templated prompt ids in-process
> shows the batched runner is bit-exact against plain greedy decode and the deviant is the
> pre-existing single-seq MTP path on KVarN (a near-tie argmax flip — and already on the record as
> `docs/130` §5.2 #2). Steps 2, 4 and 5 are now closed, the serving cell is wired into
> `run_ci.sh` and green, and N=1 byte-identity is proven. Four further real defects were found by
> running the scripts this round left unrun — including a product bug that failed **every**
> multi-chunk batched request. Everything below remains accurate as the round-2 record.

**Status at writing:** Steps 1–3 and 5 **implemented + live-validated**; the two blockers
inherited from `139_2a_debrief.md` (§4.2a garbage licensed token, §4.2b barrier deadlock) are
**solved, committed and verified live**. Step 4 measurement **not run**. One **new real defect**
is open and it is the only thing keeping the serving CI cell red: a batched-vs-single-sequence
output parity gap **isolated to thinking mode** (§4.5). `serve_batched_ci.sh` is therefore
**deliberately not wired into `run_ci.sh`** — it must not gate green on a known-red assertion.
GPUs released, no server running, tree clean, branch pushed.

Branch `wo/2a-batched-serving`. Worktree `~/ninfer/worktrees/wo-2a-batched-serving`.
Base `wo/kv-uniform` @ `c5f598b9`; **base has moved to `8b386f3b`** (4a/4b/4d + B2/B3/B4
closeout) — rebase still owed (§6.6).

Commits this session, on top of `59affb99` (Step 1) and `09cd4a06` (139 debrief):

| Commit | What |
|---|---|
| `0a0161d5` | Step 2+3: per-lane stats, admission, **allgather root cause** (+ prior mb_prop fix) |
| `854db9e6` | §4.2b: lane-retirement desync that wedged the server |
| `b5f76fdd` | Step 4+5: `serve_batched_ci.sh` (B0–B3) + `serve_batched_measure.sh` |
| `14422140` | docs + preserved probe scripts (`tools/smoke/diag/`) |

Pushed to `github/wo/2a-batched-serving` (new branch on the remote).

> Doc-number note: this file was first written as `docs/140_2a_continuation.md`. **140 collides**
> with `repo/docs/140_kvarn_prefill_gate_m3_remediation_work_order.md` (a different work order).
> Renumbered to 143. Highest doc number in the repo at writing is 142.

---

## 1. The task (docs/139, verbatim scope)

Make "batched decode" a *served* KVarN feature on TP2. The tested batched runner
`run_tp2_requests_batched` was dead code in production — the engine ran every request through
single-sequence `run_tp2_request`. Required after the WO:

- `--max-concurrency N` server + N concurrent KVarN (KVARN_K4V2) requests route through
  `run_tp2_requests_batched` (true MultiBatch attention).
- Per-lane stats (acceptance, tok/round) reported, mirroring single-seq `TpRunStats`.
- Overflow (`N > max_concurrency` or MTP ring `need > have`) rejected/queued with a clear
  error — no silent truncation.
- A decode_guard / `run_ci.sh` batched cell fails CI on a serving-path regression.
- **N=1 must remain byte-identical.** No attention-kernel changes. Worktree only, commit per
  step, live e2e proof for every server-facing step (docs/50 §7.1).

Definition of done (docs/139 §8): steps 1–5 committed with passing tests incl. live-path tests;
live proof with the launch command recorded; **N=1 byte-identity proven**; determinism; CI cell
wired + negative test fails a broken build; measurement data committed to `results/` with server
config + clocks.

---

## 2. What is delivered (per step)

### Step 1 — engine batch dispatch: **COMMITTED** (`59affb99`, prior session)
Unchanged from docs/139 §2. Leader/follower rendezvous on `SharedState::mutex`; 1 member →
untouched single-seq call; ≥2 eligible → `run_tp2_requests_batched` with per-lane cancellations;
eligibility = KVarN ∧ `max_concurrency > 1` ∧ `temperature == 0` ∧ no lookup drafts ∧ uniform
`mtp_k`; overflow requeued, never dropped.

### Step 2 — per-lane context + stats: **COMMITTED + LIVE-VALIDATED** (`0a0161d5`)
Runner fills per-lane `prefill_ms/prefill_tps`, `decode_seconds/decode_tps`, `total_seconds`,
`rounds`, `accepted_drafts`, `mean_a_per_round`, `acceptance_rate`, `tokens_per_round`,
`prefix_reuse_path=FullReset` (rank-0-owned counters, read post-join). Engine logs one line per
lane. Live proof (serve, `--max-concurrency 2`, MTP k=3):

```
[tp2] batched lane 0: acceptance=0.60 tok/round=2.85 rounds=27 gen=77 prefill=0.14s decode=1.77s
[tp2] batched lane 1: acceptance=0.70 tok/round=3.13 rounds=23 gen=72 prefill=0.15s decode=1.77s
```
Not yet done: the WO's "within ±0.5 pt of the single-lane baseline" comparison — needs the
Step 4 measurement run.

### Step 3 — admission: **COMMITTED + LIVE-VALIDATED** (`0a0161d5`)
Ring arithmetic (`need = lanes*T` vs `have = 2*lanes + 2k + 1`) reused in `run_batch_dispatch`.
Live proof at `max_concurrency + 1` = 3 concurrent:
```
[tp2] batch admission: admitted 2 lane(s), queued 1 request(s) (max_concurrency=2 mtp=1 k=3)
```
all three requests served, none truncated (CI cell B2 PASS).

### Step 4 — live e2e proof: **SCRIPT READY, NOT RUN**
`tools/smoke/serve_batched_measure.sh` exists ({MTP on, off} × {2-concurrent, sequential} at
mixed context, greedy determinism repeat, sampling fallback probe → `results/batched_serving_<ts>.json`).
Blocked on §4.5 (thinking-mode parity) or on running the matrix with `--no-thinking`.

### Step 5 — CI batched cell: **SCRIPT READY + RUN, NOT WIRED**
`tools/smoke/serve_batched_ci.sh`: B0 sequential baseline; B1 concurrent pair must show the
`dispatched 2-lane batch` line + per-lane stats + outputs byte-equal to B0; B2 overflow; B3
negative via `NINFER_MB_SERVE_MUTATE=1`.
**Verdict on this branch: B0 PASS · B1 lane 0 PASS / lane 1 FAIL · B2 PASS · B3 PASS.**
B3 proves the cell is not blind. Not added to `run_ci.sh` — see §6.1.

---

## 3. Infrastructure / environment problems (and fixes)

1. **Wedged server holds both GPU contexts.** A hung batch leaves `ninfer-serve` holding
   ~13.5 GiB across both cards; the next launch dies with `cudaErrorMemoryAllocation: out of
   memory`, which reads like a capacity problem but is not. **Always `pkill -9 -x ninfer-serve`
   + `sleep 3` before relaunching.** (WO says `pkill -x`; a wedged process ignores SIGTERM —
   use `-9`.)
2. **Disk 94% / ~16 G free.** A `--full` CI run writes several GB. Budget for it.
3. **Build flags** unchanged from docs/139 §3.1: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
   -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON -DCMAKE_BUILD_TYPE=Release
   -DCMAKE_CUDA_ARCHITECTURES=120a`.
4. **`serve_batched_ci.sh` takes ~4 min** (two server launches + 4 cells). Run it backgrounded
   with `nohup` and poll — it will otherwise eat your wall-clock budget (it ate two of mine).
5. **`tp2_backend.cpp` is plain C++, not CUDA.** bf16 intrinsics (`__bfloat162float`) do not
   compile there; decode bf16 by hand (`u32 = (u32)half << 16; memcpy` to float).
6. **GPU contention:** agent2 (docs/141) is closed and confirmed zero pending GPU work. Their
   handover note is worth keeping: **guard on foreign CUDA contexts, not ports** — a server
   moved ports under `compute-sanitizer` and a port test read "free" while both cards held
   14.1 GiB.
7. **gdb cannot attach** (`ptrace_scope=1`). For a *hang*, cheaper than the docs/139 gdb trick:
   read `/proc/<pid>/task/*/stat` field 3 — the thread stuck in `R` while the batch makes no
   progress is the spinning rank. That plus the argmax trace (§4.3) localised the deadlock in
   one step.

---

## 4. The debugging story (the important part)

### 4.1 §4.2a SOLVED — batched verify allgather passed BYTES where NCCL wants ELEMENTS

`run_tp2_requests_batched` called:

```cpp
allgather_local_bf16(rank, mb_vlog + col*nv_l*2 /*bytes*/, nv_l * 2 /*<-- WRONG*/,
                     mb_vfull + col*2*nv_l*2);
```

`allgather_local_bf16`'s third argument is `ncclAllGather`'s **`sendcount` in elements of the
datatype**. The single-sequence reference in the same file passes `nv` (elements). The batched
call passed `nv_l * 2` — the *byte* width of an `nv_l`-element bf16 shard. The 2× did two things:

1. each column's allgather wrote `sendcount * world = 4*nv_l` elements into a `2*nv_l`-element
   column, so column `c` clobbered column `c+1` and the "full-vocab" distribution the accept
   kernel reads was rank-0-only data;
2. the **last** column ran `2*nv_l` elements (496,640 B) **past the end of `mb_vfull`**, straight
   through `mb_prop`, `mb_seed_hid`, `mb_arh`, `mb_arnext`, `mb_arh2` and into `mb_tgt` /
   `mb_tgt_rm` / `mb_drafts` / `mb_lic`.

That overwrite *is* the garbage licensed token: `0xC090C090` (= `-1064255344`) is two bf16 `-4.5`
values — raw logit bits in the licensed-token buffer, exactly the "bf16/int32 buffer confusion"
§4.2a predicted.

**The test that found it** (keep this one): a host-side cross-check added at the existing
transpose point compares the fused TP argmax output against a host argmax of the allgathered
full-vocab logits, per column, env-gated by `NINFER_MB_LICDBG` (`[ARGB]` lines). It cleanly
separates "verify logits are garbage" from "argmax result is garbage":

```
before:  [ARGB] step=1 col=0 argmax_out=-1064255344 host_argmax=1156 *** MISMATCH ***   (8/8)
after:   [ARGB] step=1 col=0 argmax_out=1156       host_argmax=1156  MATCH              (8/8)
```

**This also closed §4.3 of docs/139.** The long-prompt batched-vs-sequential divergence was
never an argmax near-tie or a kernel parity gap — it was this overwrite. Post-fix the harness is
bit-exact both lanes at ctx 80000 with ~70-token prompts and at `--tokens 128`.

### 4.2 §4.2b SOLVED — two independent defects, both "server stays up but wedges"

Reproduced with a concurrent pair whose lanes close their turn on **different rounds** — the
ordinary case once any lane hits EOS. Neither showed up in the harness, which never cancels.

**(a) `active` was declared INSIDE the rank worker lambda**, so each rank thread retired lanes
in its own private copy and the ranks silently diverged on the live set. Caught directly with
the one-shot argmax handshake trace (`NINFER_MB_ARGMAX_TRACE`, added to `one_shot_argmax.cu`):

```
[A1TRACE] rank=0 step=92 slot=28 epoch=3 T=4 n_rows=124160 raw
[A1TRACE] rank=1 step=92 slot=28 epoch=3 T=8 n_rows=124160 raw
```

Step counters agreed, column count did not → rank 0's kernel spun on peer columns rank 1 never
published. No exception, no join, both HTTP requests time out. `active` is now shared host state
beside `lanes`, as `std::atomic<int>` — **not** `std::vector<bool>`, whose packed bits let two
rank threads lose an update by read-modify-writing the same word.

**(b) Per-lane cancellation was evaluated inside the MTP commit loop**, which runs on both rank
threads. The `OutputSession` lives on the leader thread and only rank 0 runs the token
callbacks, so a cancellation view can flip between rank 0's read and rank 1's — the same
disagreement as (a). Cancellation is now swept by **rank 0 only** after the round barrier and
published to rank 1 by a second barrier (single writer, both readers), matching the plain path's
rank-0-only commit loop. The in-loop `done` keeps only deterministic conditions (output limit /
context guard / stop token).

Plus debrief shape (a): the engine **wraps each lane's token callback** — a throw is captured on
the `TpBatchMember`, the lane retires through its own cancellation view, only that request fails.
Verified: uneven-EOS pair returns HTTP 200 on both lanes (gen 74 / 71, `finish=stop_token`,
27 vs 23 rounds) and a following lone request still serves.

### 4.3 Why the deadlock was hard, and what actually cracked it
docs/139 spent the session on the *exception* path (garbage token → throw → stranded rank). The
garbage token was one bug; the wedge was a **second, unrelated** bug (private `active`) that only
appeared once the first was fixed and lanes could retire at different rounds. The two traces that
did the work — `[ARGB]` (host-vs-device argmax cross-check) and `[A1TRACE]` (per-rank step/slot/T
pairing) — are both cheap, env-gated, and worth keeping permanently.

### 4.4 NEW FINDING — the TurnClosure rewrite frontier is NOT output-neutral

`rewrite_checkpoint_frontier` is deliberately not an eligibility exclusion (every chat completion
carries one) and the batched config clears it. Measured with `NINFER_NO_FRONTIER=1`, all requests
forced single-sequence (`NINFER_BATCH_DISABLE=1`), same prompt:

| config | output |
|---|---|
| frontier ON | `Ice floats because water expands when it freezes: hydrogen bonds form…` (327 B) |
| frontier OFF | `Ice floats on water because water expands when it freezes, making ice less dense…` (294 B) |

The frontier splits the prefill at the TurnClosure offset; the changed reduction order flips the
greedy argmax **from the first decoded token**. The frontier-OFF text is byte-identical to what
the batched lane produces — **the batched lane is not at fault.** docs/70 presents this as a
prefix-restore optimisation, which implies output-neutrality. **Report upstream; out of 2a
scope.** Consequence: CI cell B0 must run with `NINFER_NO_FRONTIER=1` or it compares two
different effective configs.

### 4.5 OPEN — batched-vs-single-seq parity gap, isolated to THINKING MODE

The only thing keeping B1 red. Probe: same prompt fired alone (single-seq) then fired as a
2-lane self-paired batch; frontier off; sha of content+reasoning.

| prompt | alone | batched | |
|---|---|---|---|
| ice floats | `48de073e7b` | `48de073e7b` | CLEAN |
| lighthouse | `0e2cc3a4bf` | `a03991e784` | DIVERGE |
| water cycle | `d77631b77b` | `65297ea57d` | DIVERGE |
| rainfall | `42e053bad0` | `b1c749523f` | DIVERGE |
| three Moon facts | `141ea14048` | `ae3eee075e` | DIVERGE |
| reverse a string | `29ac857893` | `28f240277e` | DIVERGE |

**5 of 6 diverge — systematic, not a rare near-tie.** Each of these was ruled out by direct test,
do not re-walk them:

* *memory corruption* — fixed in §4.1; batched output is now fully self-consistent.
* *cross-lane contamination* — a prompt paired with **an identical copy of itself** produces
  byte-identical output on both lanes.
* *the B 2→1 shrink path* — pairing with a shorter prompt (early retirement) gives the **same**
  batched output as pairing with itself.
* *unstable reference* — single-sequence serve fired 3× is deterministic (`0e2cc3a4bf` ×3).
* *prefix reuse* — still diverges under `--no-prefix-reuse`.
* *rewrite frontier* — still diverges with `NINFER_NO_FRONTIER=1` (that's §4.4, separate).
* *prompt length / shared preamble* — harness with a ~90-token shared-preamble chat-style pair at
  `--tokens 96` and `--tokens 128`: **PASS, bit-exact both lanes**.

**The decisive discriminator is thinking mode:**

| server flags | alone | batched | verdict |
|---|---|---|---|
| (thinking on, default) | `0e2cc3a4bf` | `a03991e784` | **DIVERGE** |
| `--no-thinking` | `afae3d361e` | `afae3d361e` | **MATCH** |

So the gap is specific to the thinking-mode request shape and is **serve-only** — the in-process
harness never enables thinking, which is why every harness cell is green. Divergence appears
early in the reasoning stream ("I should keep it simple…" vs "I need to keep it to one
sentence…"), i.e. within the first ~10 decoded tokens → a prefill/early-decode state difference,
not a late near-tie.

---

## 5. Current exact state

- Branch `wo/2a-batched-serving` @ `14422140`, **pushed** to `github`. Tree clean.
- GPUs free (15 MiB both), no server running.
- **Validated green:** clean build (CUDA 13.1); BATCHTEST W=2 MTP k=3 PASS bit-exact both lanes
  (short prompts, ~70-token @ ctx 80000, and `--tokens 128`); serve batched-MTP e2e with
  per-lane stats and 0 errors; uneven-EOS pair serves cleanly (the old wedge); CI cells
  B0/B2/B3 PASS; B3 negative test detects a deliberately corrupted batched commit.
- **Red:** CI cell B1 lane 1 — the §4.5 thinking-mode parity gap.
- **Not done:** Step 4 measurement run; ±0.5 pt acceptance-vs-single-lane comparison; N=1
  regression proof via `run_ci.sh` fast gate; rebase onto `wo/kv-uniform @ 8b386f3b`; wiring
  `serve_batched_ci.sh` into `run_ci.sh`; diagnostic cleanup decision.
- Temporary diagnostics in the tree (all env-gated): `[MB-RANK0/1-ERR]` catch prints,
  `NINFER_MB_LICDBG` (+ `[ARGB]` cross-check), `NINFER_MB_ARGMAX_TRACE`, `NINFER_DRAFT_DBG`,
  `NINFER_BATCH_DBG`, `NINFER_NO_FRONTIER`, harness `NINFER_MB_REF_PREFIX` / `NINFER_MB_REF_CTX`
  / `NINFER_TEST_WS_BYTES`. `NINFER_MB_SERVE_MUTATE` is permanent (B3 depends on it).
- Probe scripts preserved under `tools/smoke/diag/` (`repro2a.sh`, `eos2b.sh`, `cmpseq.sh`,
  `shrink.sh`, `probe.sh`, `nt.sh`, `frontier.sh`, `det.sh`, `txt.py`). Server logs under
  `logs/` in the worktree.

---

## 6. Suggested next steps (ordered)

1. **Chase the thinking-mode parity gap (§4.5) — this is the critical path.** Next concrete
   move: dump the exact chat-templated **thinking** prompt token ids the engine builds and feed
   them verbatim to `ninfer_tp2_batched_decode_test --prompt-a/-b`. The harness tokenizes raw
   text and has therefore never seen this prompt shape. If the harness then reproduces the
   divergence, it is runner-level → bisect with `NINFER_MB_FORCE_B1=1` and the M0 phase gate
   (`NINFER_MB_PHASEGATE`) against the single-seq reference. If the harness does **not**
   reproduce, the difference is in engine-side request construction for thinking requests, not
   the runner. Thinking mode is the last thing 2a has not exercised.
2. **Only after B1 is green:** wire `serve_batched_ci.sh` into `run_ci.sh`'s fast gate next to
   the M4 MTP round gate, and re-run the B3 negative in-run. Do not wire it red.
3. **Step 4 measurement:** `bash tools/smoke/serve_batched_measure.sh` → commit
   `results/batched_serving_<ts>.json`. Then do the WO's ±0.5 pt acceptance comparison against a
   single-lane baseline. If the gap blocks this, run the matrix with `--no-thinking` and label it.
4. **N=1 regression proof:** `bash tools/ops/run_ci.sh` (fast gate) on this branch. Note the
   engine now has a 250 ms collection window on `--max-concurrency > 1` servers only; a
   `--max-concurrency 1` server never enters it, and `NINFER_BATCH_DISABLE=1` is the kill-switch.
5. **Diagnostic decision:** keep `[ARGB]` + `[A1TRACE]` permanently (they are the two tests that
   found §4.1 and §4.2a and cost nothing when unset); keep `NINFER_NO_FRONTIER` (the CI cell
   depends on it); strip `[MB-RANK0/1-ERR]`, `NINFER_DRAFT_DBG`, `NINFER_BATCH_DBG`,
   `NINFER_MB_PACKDBG` and the harness bisect hooks at closeout.
6. **Rebase** onto `wo/kv-uniform @ 8b386f3b`. Expect conflicts only in the `tp_engine.cpp` /
   `tp2_backend.cpp` neighbourhoods.
7. **Report §4.4 (frontier non-neutrality) and §4.5 upstream** to the runner/kernel owners.
   docs/130/136 claim batched/single-seq bit-exactness; §4.5 is the first known counterexample
   and it is prompt-shape-dependent, so the claim needs qualifying.

---

## 7. Key numbers (measured this session)

| Check | Result |
|---|---|
| `[ARGB]` argmax cross-check, pre-fix | 8/8 columns MISMATCH (garbage = bf16 logit bits) |
| `[ARGB]` argmax cross-check, post-fix | 8/8 MATCH |
| BATCHTEST W=2 MTP k=3, short prompts | PASS, bit-exact both lanes |
| BATCHTEST W=2 MTP k=3, ~70-token @ ctx 80000 | PASS, bit-exact both lanes |
| BATCHTEST W=2 MTP k=3, `--tokens 128` | PASS (4.3 s batched vs 4.1 s seq = 0.97×) |
| BATCHTEST W=2 MTP, shared-preamble 90-token pair | PASS, bit-exact both lanes |
| Serve batched-MTP e2e (2 lanes, uneven EOS) | HTTP 200 both; gen 74 / 71; 27 vs 23 rounds |
| Serve per-lane stats | lane0 acceptance 0.60 / 2.85 tok/round; lane1 0.70 / 3.13 |
| Serve admission (3 concurrent, max_conc=2) | "admitted 2, queued 1"; all 3 served |
| CI cell B0 / B1 / B2 / B3 | PASS / lane0 PASS lane1 FAIL / PASS / PASS |
| Thinking-mode probe (6 prompts, alone vs batched) | 1 CLEAN, 5 DIVERGE |
| Same probe with `--no-thinking` | **MATCH** |
| Frontier ON vs OFF, single-seq, same prompt | 327 B vs 294 B — **not output-neutral** |
| Single-seq serve determinism (3× same prompt) | identical sha `0e2cc3a4bf` ×3 |

---

## 8. Launch commands used (record)

```bash
# configure
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j 16            # or --target ninfer-serve / ninfer_tp2_batched_decode_test

# serve (batched validation) — add NINFER_NO_FRONTIER=1 for like-for-like parity tests
env NINFER_NO_FRONTIER=1 ./build/apps/ninfer-serve /home/intel/models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8095 --devices 0,1 --spec mtp --draft-tokens 3 \
  --kv-dtype kvarn_k4v2 --kv-capacity 100000 --max-context 80000 \
  --max-concurrency 2 --model-id qwen3.8-27b
#   diagnostics: NINFER_MB_LICDBG=1 (accept/argmax cross-check) NINFER_MB_ARGMAX_TRACE=1
#                NINFER_MB_PACKDBG=1 NINFER_BATCH_DBG=1
#   kill-switch: NINFER_BATCH_DISABLE=1 (force all-single-seq)   window: NINFER_BATCH_WINDOW_MS

# harness (batched vs sequential bit-exactness)
./build/tests/ninfer_tp2_batched_decode_test --artifact /home/intel/models/qwen3_8_27b.ninfer \
  --tokens 128 --mtp 3 --ctx 80000 --prompt-a "..." --prompt-b "..."

# serving CI cell (B0-B3), ~4 min — run backgrounded
nohup timeout 600 bash tools/smoke/serve_batched_ci.sh > /tmp/ci.log 2>&1 &

# thinking-mode parity probe (the §4.5 discriminator)
bash tools/smoke/diag/nt.sh                       # thinking on  -> DIVERGE
FLAGS=--no-thinking bash tools/smoke/diag/nt.sh   # thinking off -> MATCH
```
