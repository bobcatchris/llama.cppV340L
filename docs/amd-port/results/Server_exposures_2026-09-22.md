# Server exposures E-039: unified KV admission + fill order fairness

Date: 2026-09-22. Desk: wt-server-fixes (branch amd/server-exposures), the
E-053 zero-GPU dispatch. Fixes the two build-independent server exposures
flagged in E-038/E-039 during the E-037 contamination forensics: a
7857-token request starved to death (28 KV retries down to n_batch=1, then
"Context size has been exceeded. off = 69", HTTP 500 for EVERY in-flight
request) while a server served concurrent traffic on a unified KV cache.
Strictly host-only desk: zero die time, no GPU build.

## Defect 1: per-slot n_ctx admission ignores global unified KV occupancy

Under kv_unified (n_parallel auto = 4), cparams.n_ctx_seq == n_ctx, so every
slot's server-side n_ctx equals the FULL cache (10240 cells in the incident)
and the only admission checks (task tokens vs slot.n_ctx, in pre_decode)
admit any request that would fit an EMPTY cache - with no look at how many
cells the other slots currently hold. Two concurrent 7857-token prompts
demand 15714 cells from a 10240-cell cache. The newcomer gets admitted, then
cannibalizes the cache: 6587 (task 14) + 3653 (task 35 trickle) = 10240 =
zero free, the decode path halves n_batch 512 -> 256 -> 128 -> 64 (commit
64/4/1 at batch offsets 0/64/68/69) and throws at nb=1, aborting both
requests with HTTP 500. Reproduced number-for-number by the incident
trajectory sim (test section 2).

## Defect 2: slot fill order starves an older mid-prompt request

The pre_decode prompt-fill loop iterates slots in slot index order. In the
incident the NEWER request (task 35) sat on slot 0 and took the whole
512-token batch every round while the older mid-prompt request (task 14,
slot 1, still needing 1782 cells) received none - the "task 35 placed 3584 +
64 + 4 + 1 trickle" line of the forensics. Reproduced by the one-round fill
sim (test section 3). Slot index is arbitrary with respect to arrival, so
the victim is luck, not policy.

## Fixes (tools/server/server-context.cpp, common/arg.cpp, common/common.h)

1. Global unified KV admission control (default ON, `--kv-admission` /
   `LLAMA_ARG_KV_ADMISSION`): in process_single_task, after slot selection
   and only when params_base.kv_unified, a task launches only if
   kv_unified_admission_fits(): its remaining prompt cells
   (kv_unified_cells_needed: task tokens minus the candidate slot's cached
   common prefix, cache_prompt-aware, 1 token entry = 1 cell media included)
   PLUS the remaining prompt cells of every other in-flight request
   (n_pending: STARTED slots via the same helper, PROCESSING_PROMPT slots as
   task - placed) fit in n_ctx - sum(all slots' prompt cells). The pending
   reservation is exactly the "don't take the last cells an in-flight
   request still needs" rule. When the check fails: idle slots' cached
   prompts are purged first (try_clear_idle_slots gained a `skip` parameter
   so the candidate's own freshly loaded prefix cache is never purged - same
   relief policy as the decode() retry path); a request larger than the
   whole cache is rejected immediately with the clean 400
   (ERROR_TYPE_EXCEED_CONTEXT_SIZE); anything else is DEFERRED - it stays in
   the existing deferred queue and is re-admitted when an in-flight request
   releases its slot (pop_deferred_task on callback_on_release), i.e. clean
   queueing instead of accept-then-starve. The HTTP client sees a delayed
   response, never a 500 from this class.

   Default ON because the current behavior is objectively a starvation bug
   (E-037/E-038: one foreign request turned into HTTP 500 for the whole
   server). Escape hatch: --no-kv-admission restores legacy behavior. The
   check engages only under kv_unified: with kv_unified=false each slot owns
   a private n_ctx/n_seq_max cells region and per-slot admission is already
   exact.

2. FIFO slot fill order (default ON, `--kv-fifo-fill` /
   `LLAMA_ARG_KV_FIFO_FILL`): the fill loop iterates processing slots sorted
   by task id (monotonic arrival order; deferred tasks keep their original
   id, so re-queued requests preserve age) instead of slot index. With
   admission ON this makes the older request win the remaining cells first
   and finish its prompt before a newcomer places anything; with admission
   OFF it changes only who starves first, not whether (test section 7
   documents that admission is the load-bearing fix). Escape hatch:
   --no-kv-fifo-fill restores slot-index order. Generative slot handling,
   can_batch_with anchoring and parent/child logic are unchanged; the batch
   anchor simply becomes the oldest compatible task.

## Host tests (docs/amd-port/tests/test_server_exposures_host.cpp)

House convention per test_w6_isolation_host.cpp: standalone, no ggml
linkage, mirrors of the exact server-side arithmetic. ALL PASS:

- unit mirrors: kv_unified_cells_needed (prefix credit, cache_prompt off,
  clamp), kv_unified_admission_fits (exact fit admits, +1 cell rejects)
- defect 1 reproduction: the incident trajectory (6587 + 512 in-flight;
  newcomer admitted; 3584 chunk rounds; 64+4+1 trickle; offsets
  0/64/68/69; fatal at off = 69; used == 10240; BOTH slots aborted with
  HTTP 500; the older request never got another cell)
- defect 2 reproduction: one legacy fill round hands all 512 batch tokens
  to the newer task on the lower slot index while the older request gets
  zero; FIFO ordering flips it
- admission decisions: incident geometry defers (need 7857 + pending 1270
  vs free 3653); empty server admits; prefix reuse flips the decision at
  the exact boundary; knobs-off falls back to legacy; request > n_ctx is
  REJECT_TOOBIG; purge-then-admit reclaims idle caches
- fixed trajectory: task 35 deferred at admission; task 14 completes its
  full 7857-token prompt and generates; release pops the deferred task;
  stale cache purged; task 35 completes; zero aborts, occupancy never
  exceeds n_ctx
- FIFO-only control: the older request now completes its prompt but the
  still-admitted newcomer still drains the cache to zero and hits the nb=1
  abort - documents that admission is primary, FIFO is fairness

Build + run: g++ -std=c++17 -Wall -Wextra -o /tmp/test_server_exposures_host
docs/amd-port/tests/test_server_exposures_host.cpp && /tmp/test_server_exposures_host
=> "ALL PASS". Predecessor suite test_w6_isolation_host re-run: ALL PASS.

## Compile-clean

No cmake/ninja in the worktree environment; direct host compile of the
touched TUs: g++ -std=c++17 -O1 -c -Wall -Wextra -Itools/server -Itools/mtmd
-Icommon -Iinclude -Iggml/include -Isrc -Ivendor -I. on server-context.cpp
and common/arg.cpp - both exit 0, zero warnings from the changed lines (the
only warnings are pre-existing -Wunused-function on unrelated header
statics). Full linked build deferred to the coordinator's build lane.

## Needs a served-window validation (coordinator schedules)

Host arithmetic proves the decision logic; a short served window on the
gfx900 build should confirm end-to-end behavior: (1) boot -kvu with a small
-c (10k class, the incident geometry), fire the two concurrent 7857-token
prompts on one port and verify the second is queued (defer log line) and
completes after the first, with zero "Context size has been exceeded" lines
and zero 500s; (2) confirm no retry storms in the log; (3) confirm
--no-kv-admission reproduces the legacy failure signature (regression
hatch). No 200k boot needed: the incident geometry is 10k-class.

## Residual exposures (documented, out of desk scope)

- Generation pressure: admission reserves cells for in-flight PROMPTS, not
  for generation (unbounded without n_predict). Under kv_unified, several
  long generations can still jointly fill the cache and hit the nb=1 abort
  ("terminate only the largest active slot/sequence" - the existing
  server-context.cpp TODO). Per-slot n_ctx stop (STOP_TYPE_LIMIT) only
  triggers at a slot's own full-cache-sized limit under unified.
- Parent/child shared-prompt tasks (n_cmpl > 1): the parent passes admission
  as one task, but copy_state_to duplicates the prompt cells into each child
  sequence afterwards; child cell demand is not modeled at admission.
- The fill-time batch accounting still counts filled-but-uncommitted tokens
  server-side on KV-full retry rounds (pre-existing overcount; conservative
  for admission, which reads the same counter).
