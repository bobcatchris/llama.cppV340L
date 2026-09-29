# Drain-attack desk: draft-step drain budget, synchronize audit, gated fetch/sync fixes

Date: 2026-09-22. Worktree: wt-mtp-drain (branch amd/mtp-drain). ZERO GPU:
code audit, implementation, host tests, compile checks only.

Attacked number: each of the 3 draft steps per MTP round costs ~37 ms
(accept 0.66667, mean 3.0) against a few ms of real device work, capping MTP
at 1.15-1.26x where the modeled ceiling is 1.76-2.04x (21.4-24.8 t/s class
vs 15.1 today). Desk inputs: the sampling-path map of record
(MTP_overhead_2026-09-22.md) + this desk's full read of the per-step path
(llama-context decode/extract/synchronize, common/sampling, the meta backend
graph_compute + get_tensor_async, ggml-backend tensor_copy paths).

## 1. Synchronize audit - every blocking point in one draft step

The draft loop (common/speculative.cpp draft_mtp::draft, bs=1, non-shared
KV, single nextn head) executes per step:

| # | site | what blocks | class |
|---|------|-------------|-------|
| S1 | decode entry: set_inputs (llama-graph.cpp) | 4-6 buffer-level `ggml_backend_tensor_set` H2D copies (tokens 4 B, embd/h rows 16-32 KB, pos, out_ids); the CUDA/HIP buffer set is `cudaMemcpyAsync` + `cudaStreamSynchronize` + `cudaSetDevice` per tensor (ggml-cuda.cu buffer iface) | 4-6 blocking host round trips, ~50-300 us |
| S2 | graph issue: meta `graph_compute` | host-async internally; per boundary the allreduce fallback (no host block: peer copies are `cudaMemcpyPeerAsync` + event handoff, ADDs are async 1-node graph replays). Without `GGML_CUDA_P2P` peer copies are driver-staged through host memory (2 DMA legs, async) | device latency, not host block |
| S3 | extract: 2 meta `get_tensor_async` calls (3 logits shards + 1 mirrored h row, ~517 KB + ~24 KB) | none - all 4 copies are async DMAs into the pinned output buffer; the AXIS splice already writes each byte exactly once at its final offset (the 2D call degrades to a linear copy per shard at n_copies=1) | 0 host block |
| S4 | `common_sampler_sample[_topk]` -> `llama_synchronize` | THE drain: sched sweep = meta backend (3x `cudaStreamSynchronize`, one per die) + CPU backend | 1 blocking wait, the only one that must block |
| S5 | `llama_get_sampled_token_ith` (backend-token guard) | full `llama_context::synchronize` again - idle | redundant sweep |
| S6 | `set_logits`: `llama_get_sampled_probs_ith` + `llama_get_sampled_logits_ith` + `llama_get_sampled_candidates_ith` | 3 more full syncs - idle (the getters synchronize at the C-API layer) | 3 redundant sweeps |
| S7 | `llama_get_embeddings_nextn_ith` (h row for the next batch) | another full sync - idle | redundant sweep |
| S8 | one-time: output_reserve realloc sync, sched_reserve sync | n/a | init only |

Net: exactly ONE sync point per step needs to block (S4, waiting all 3 die
streams is the true dependency - the logits row is spliced from 3 dies).
The code as it stands executes FIVE TO SIX full scheduler sweeps per draft
step (S4 + S5 + S6 x3 + S7, in the regular path; S4 + S5 + S6 x2 + S7 with
FAST_TOPK). Each idle sweep costs the meta backend 3 stream-sync round trips
plus sched bookkeeping; the drain sweep additionally carries the whole device
critical path.

Device-side serial chain per step (what the drain sweep waits for):
1 nextn block + head cut at 2 PARTIAL nodes (attn out-proj, ffn_down) into
3 meta subgraphs, replayed on 3 dies = 9 main graph replays; 2 allreduce
boundaries in butterfly fallback for n=3 = per boundary 4 inter-die copies
(fold 2->0, butterfly 0->1 + 1->0, copy-back 0->2) + 3 ADD replays = 8 copies
+ 6 ADD replays per step; then the 4 output DMAs. Every die's stream carries:
3 subgraph replays + 2 full allreduce chains + its D2H. At 200k the KV read of
the 1-layer block and the head-shard read sum to ~0.2-0.5 ms of real memory
work per die - everything above that in the ~37 ms envelope is per-op
latency: replay/launch overhead x 15 device issues, the host-staged AR round
trips, and the host relaunch slice (S1 + graph-issue host work + batch
rebuild).

## 2. What was implemented (all env-gated; unset = byte-identical)

### 2.1 LLAMA_DRAFT_PACKED_GET - one packed fetch per draft step

- decode() skips its raw-logits and h_nextn extraction for the ctx
  (`packed_fetch`, src/llama-context.cpp: gated in both extract branches).
- One call per step, `llama_fetch_nextn_outputs(ctx_dft, i_last, &row, &h)`
  (staged API in src/llama-ext.h + llama-context.cpp): issues the SAME two
  meta gets (identical tensors, byte counts, pinned destinations) and returns
  the two host row pointers. No sync, no copies beyond the skipped ones -
  the bytes are identical by construction.
- Sampling runs on the returned rows: `common_sampler_sample_row`
  (full-vocab candidate build from the row -> candidate arrays and chain
  application IDENTICAL to the regular path) or, with FAST_TOPK also set,
  `common_sampler_sample_topk_row` (heap-select on the row). Both new
  entries never synchronize (common/sampling.h/.cpp; the ctx-based
  entries are refactored onto the same guts: set_logits_row /
  set_logits_topk_row / topk_eligible).
- Guarded off (with a WARN) when a backend sampler is attached (its token
  would bypass the host rows) or when the draft shares the target context
  (gating decode() would hide the target's own outputs); on a fetch failure
  (unsorted outputs, unsupported layout - not reachable for the draft batch)
  the draft round stops rather than read stale rows.
- Deleted per step vs baseline: the decode-internal extract fan-out moves to
  one call, and the sampling path's getter syncs are not paid (the row
  entries touch no C-API getters).

### 2.2 LLAMA_DRAFT_LIGHT_SYNC - the drain count goes from 5-6 to 1

- `llama_wait_outputs(ctx)` (staged API + llama-context.cpp): synchronizes
  ONLY the backends owning the fetched outputs (for the meta backend that is
  all 3 die streams - the true dependency). Skips the scheduler sweep and the
  perf-stat closure of llama_context::synchronize.
- In the draft loop the packed fetch drains via wait_outputs instead of
  llama_synchronize; the row-based sampler entries and the returned h-row
  pointer then add ZERO further syncs. Audit result S5/S6/S7 are deleted;
  exactly one blocking wait remains per step.
- Requires PACKED_GET (ignored otherwise). With PACKED_GET alone the loop
  still uses the full llama_synchronize (stats close as before) - the
  sampler/h-row syncs are still gone.
- Perf-counter caveat (gated mode only): with LIGHT_SYNC the draft ctx's
  n_queued_tokens/t_eval_us bookkeeping stops closing; the draft context is
  not the reported perf surface (the server reports the target ctx).

Budget for the arm, per draft step: baseline = 1 drain + 4-5 idle sweeps +
4-6 blocking H2D sets + extract in decode; PACKED_GET = 1 drain (explicit) +
0 sampler syncs; PACKED_GET+LIGHT_SYNC = exactly 1 wait + 0 sweeps. Static
estimate of the deletable slice: 4-5 idle sweeps x (3 stream syncs +
bookkeeping) ~ 0.1-0.5 ms/step - real, but one order below the ~20 ms/step
target. These arms are the PROOF instrument for that attribution as much as
they are a fix.

## 3. Drain budget verdict - can (a)+(b) reach the 21-25 t/s class?

No. The audit says the host-side slice that (a)+(b) can delete is the
redundant sweeps + extract call-site overhead (~0.1-0.5 ms/step), while the
measured envelope is ~37 ms/step against ~0.2-0.5 ms of real per-die memory
work. Even the whole CPU chain (already cut by FAST_TOPK) is small. The
~20+ ms/step lives in the device-side serial chain and the relaunch:
9 main replays + 14 AR/ADD issues per step, each carrying launch/replay
latency, and 8 host-staged inter-die AR copies per step (the "2 host-staged
hops" of E-039 grow to 8 DMA legs at n=3 in the butterfly fallback). The
served TIMELINE cell (decode_issue vs drain vs sample) decides which of the
two dominates; both are device-side work, out of reach for host call-site
changes. Expected signature of this desk's arms: small (0-2%), with
sample+batch down ~0.1-0.5 ms/step in the SPEC_TIMELINE lines - the value is
the deletion of the redundant syncs AND the negative result that licenses the
next desk to go device-side.

## 4. Step-batched design (two draft steps per replay) - feasibility verdict

Goal: amortize the per-step relaunch by computing two draft steps in one
graph replay. Structure of the dependency (facts of record, re-derived from
graph_mtp): step N+1 consumes id(N) (via nextn.enorm get_rows) and h(N)
(via nextn.hnorm) - both exist only AFTER step N's graph completed and the
host sampled id(N) out of its head output. A lockstep two-step graph is
therefore impossible without changing what is computed:

Option A - speculative branching (batch the top-2 candidate ids of step N as
two rows through step N+1's block+head). Feasible only with branched KV: the
two rows write different KV at the draft position, needing 2x KV slots for
the draft region, a commit-on-choice rule feeding the target verify, and
acceptance-bookkeeping changes (the second token is drafted from the top-1
branch only). Not a patch: KV layout, batch plumbing and verify semantics
all change. Verdict: NOT clean; out of desk scope.

Option B - on-device sampling + device-side loop (remove the host from the
critical path entirely): already established as a subsystem project (head
K-axis re-shard or per-shard top-k op + meta split state, #23287 refusal).
The drain budget here strengthens the case: it removes S4-S7 AND the host
round trip between steps, which is the single largest structural item left
on the host side.

Option C - micro relief inside the current structure (candidate follow-up
desk, no new subsystems): (1) async the S1 H2D input sets onto the compute
streams (quiescent-context safe, deletes 4-6 blocking round trips/step;
needs backend resolution threaded into the input setters - mechanical but
touches every llm_graph_input class); (2) cache the meta get splice plan per
logits tensor to skip split-state resolution per step (micro); (3) CUDA/HIP
graph capture health-check on the 15 device issues (if any subgraph misses
the replay cache per step it re-captures - a possible ms-scale item the
TIMELINE cell would show as decode_issue).

Verdict: two-steps-per-replay is NOT implementable cleanly; the desk's
calibration answer is that the 21-25 t/s class requires device-side work
(Option B or launch/AR-latency reduction per option C), not host call-site
changes.

## 5. Host tests

docs/amd-port/tests/test_packed_get_host.cpp (standalone, house convention):
- splice equivalence: mirror of the meta AXIS splice (uneven 3-way shards at
  the real 129272 vocab, 2/4/5-way stress, small vocabs, 1-4 output rows) -
  byte-exact rows + row-pointer resolution vs the reference row.
- mirrored h-row get (shard 0 read) - byte-exact.
- sampler parity through the fetch: sample_row vs regular path = BIT-EXACT
  candidate arrays + identical drafted token + identical seeded dist draw
  (64/1k/32k/129272 vocabs); sample_row_topk vs the fast path = identical,
  with the documented exact-tie order caveat re-measured (2/200 at 129272).
  ALL PASS; ASAN/UBSAN clean. Existing suites re-run: test_mtp_sampling_host
  ALL PASS, test_w6_mtp_device_host ALL PASS.

## 6. Files

- src/llama-context.cpp/.h: packed_fetch gate in decode's two extract
  branches; fetch_nextn_outputs / wait_outputs / set_packed_fetch.
- src/llama-ext.h: llama_set_packed_fetch / llama_fetch_nextn_outputs /
  llama_wait_outputs.
- common/sampling.h/.cpp: set_logits_row, set_logits_topk_row, topk_eligible
  refactors; common_sampler_sample_row / common_sampler_sample_topk_row.
- common/speculative.cpp: draft_mtp ctor gating (packed_get/light_sync) and
  the draft loop fetch+drain+row-sampling restructure.
- docs/amd-port/tests/test_packed_get_host.cpp: parity suite (new).
- Receipt: docs/amd-port/results/DRAIN_BUDGET_2026-09-22.md (this file).
