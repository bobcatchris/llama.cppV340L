# MTP-overhead desk: draft-step sampling path root cause + gated fixes + timeline profiler

Date: 2026-09-22. Worktree: wt-mtp-overhead (branch amd/mtp-overhead). ZERO GPU:
design, implementation, host tests, compile checks only. No boot, no die time.

The attacked number: with --spec-type draft-mtp (accept 0.66667, mean 3.0
tokens/round) each of the THREE draft steps costs ~37 ms for a computation that
is one nextn block plus one LM head pass. The MTP multiplier is capped at
~1.15-1.26x by this per-step overhead.

## 1. The sampling-path map (root cause, Task A)

### 1.1 Where the warning lives

The "backend sampling not supported with SPLIT_MODE_TENSOR; using CPU" line is
NOT in tools/server. It fires from `llama_context::set_sampler`
(src/llama-context.cpp, pre-edit line ~1249) when it is called on a context
whose model is in SPLIT_MODE_TENSOR. The server only triggers it:

    tools/server load_model
      -> common_speculative_init (server-context.cpp)
        -> common_speculative_impl_draft_mtp ctor (common/speculative.cpp)
          - builds a backend sampler chain per seq: top_k(10)
          - backend_sampling defaults TRUE (--spec-draft-backend-sampling /
            LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING)
          - calls llama_set_sampler(ctx_dft, seq_id, chain)
            -> refusal (returns false)
          - SPC_WRN "backend offload failed for seq_id=0; using CPU sampler"

Both lines appear in server logs; the refusal site is the llama-context one.

### 1.2 Provenance of the refusal: unimplemented, not proven fundamental

git shows the check was introduced by upstream PR #23287 ("Move to backend
sampling for MTP draft path", ad2775726) - the SAME PR that built the backend
sampling feature. Its own commit message: "fallback to CPU on failure cases,
such as with '-sm tensor'". It is a defensive guard for a path the author did
not implement, not a documented fundamental blocker.

### 1.3 The draft-step sequence today (per one of the ~3 steps per round)

In common/speculative.cpp, common_speculative_impl_draft_mtp::draft():

    llama_decode(ctx_dft, batch)          // async: 3 dies, per-subgraph
                                          // allreduce, nextn block + head
    common_sampler_sample(...)            // common/sampling.cpp:
      llama_synchronize(ctx)              //   FULL pipeline drain (all dies
                                          //   + output D2H copies)
      set_logits                          //   builds cur_p = 129272 x
                                          //   llama_token_data (~2 MB) from
                                          //   the host logits row
      chain apply                         //   top_k(10): std::partial_sort
                                          //   over 129272; dist: softmax over
                                          //   the 10 + RNG draw (fills p for
                                          //   the p_min gate)
      return id
    h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last)
                                          // host h row (n_embd f32)
    rebuild batch (id + h_row) -> next llama_decode

Key mechanics discovered:

- The drafted token is cur_p->data[0].id = the top-1 LOGIT (greedy). The
  trailing dist sampler never picks the draft token; it only fills the
  probabilities that the p_min gate reads. Draft-token parity at temp 0 is
  therefore argmax identity.
- id(N) is a genuine compute input of draft step N+1: the MTP block graph
  (graph_mtp, src/models/qwen35*.cpp; V340L is qwen3.8-class) consumes the
  token embedding via nextn.enorm AND the h row via nextn.hnorm, concatenated
  through nextn.eh_proj. So the host round trip is serially dependent - there
  is NO host/device overlap available inside the draft loop as built; the GPU
  idles during [drain tail + CPU chain + batch rebuild + decode entry] and the
  host idles during device compute. Any "overlap the CPU sampling with the
  next draft block" design is dead on this dependency, not on implementation
  effort.

### 1.4 Where draft logits come from and whether a die holds the full row

KEY QUESTION ANSWER: NO device holds the FULL summed logits row after the
draft LM head in this build.

- Split config (src/llama-model.cpp llama_meta_device_get_split_state):
  output.weight -> GGML_BACKEND_SPLIT_AXIS_1 (vocab axis). MUL_MAT with a
  MIRRORED input maps that to a head result sharded along its vocab axis
  (ggml-backend-meta.cpp handle_mul_mat): each die holds a DISJOINT vocab
  slice, not a partial sum.
- The meta backend's allreduce boundaries fire only after PARTIAL-axis nodes
  (the ffn_down-class K-sharded matmuls inside the block). The head output is
  never reduced on-device; there is no all-gather in the machinery either.
- The full row materializes ONLY on the host: llama_context::decode's
  "extract logits" branch (needs_raw_logits -> true because backend samplers
  were refused) calls ggml_backend_tensor_get_async on the meta backend,
  whose AXIS_0 branch splices the row from all three dies
  (ggml-backend-meta.cpp ggml_backend_meta_get_tensor_async). 129272 x f32 =
  ~517 KB per draft step, then the drain completes the transfer.

So the brief's "after the LM head's allreduce" premise does not hold for the
head in this fork: the allreduces are inside the block; the head output stays
vocab-sharded and is assembled at D2H time on the host.

### 1.5 Why (1) "sample on the device holding the summed logits" is blocked

With no die holding the full row, on-device sampling needs either:

- Head re-shard to the K axis (row-parallel head -> PARTIAL result -> existing
  boundary allreduce -> full row MIRRORED on every die -> the EXISTING backend
  chain (build_sampling -> ggml_top_k -> get_rows -> t_sampled -> tiny int D2H
  -> common_sampler_sample fast path at common/sampling.cpp) would run
  unmodified). BLOCKED because: handle_mul_mat has no (AXIS_0 weight,
  MIRRORED x) case (GGML_ABORT); it is a model-wide layout change (target head
  too); and splitting the head's K reduction across dies changes the FP sum
  order - logits differ in ulps, so temp-0 argmax parity vs the CPU chain is
  not guaranteed (acceptance-affecting).
- Per-shard top-k on device (3 x top-10, ~240 B host combine): the meta
  segment model (device slices partition a logical tensor) cannot express
  ggml_top_k - its output size k differs from the input partition n_vocab/3
  per die; handle_per_row asserts src axis != AXIS_0 for GGML_OP_TOP_K /
  GGML_OP_ARGSORT; handle_pad asserts zero-pad along the split axis
  (build_sampling pads rows of the sharded logits); the downstream get_rows
  over per-shard-local indices has no split-state case. This is a new ggml op
  plus meta split-state semantics project, not a desk patch.

Naively unrefusing (attaching the chain today) aborts at the first draft
decode in ggml_backend_meta_get_split_state - which is why the upstream guard
exists.

## 2. What was implemented (all gated; unset env = byte-identical)

### 2.1 LLAMA_TP_BACKEND_SAMPLING - experimental unrefusal (the (1) instrument)

src/llama-context.cpp set_sampler: with the env set, SPLIT_MODE_TENSOR no
longer refuses; the normal backend-offload path runs, with explicit WARNs that
sampler ops on vocab-sharded logits are not modeled and aborts are likely.
Default (env unset): the original refusal, unchanged. Purpose: the served rig
can measure exactly where the meta machinery fails instead of trusting static
analysis.

### 2.2 LLAMA_DRAFT_FAST_TOPK - the minimal-sync host-side fix (the (2) arm)

The one real host-side lever: shrink the exposed CPU sampling cost per step.

- common/sampling.h/.cpp: common_sampler_sample_topk(gsmpl, ctx, idx, k).
  Heap-selects the top-k logits directly from the row (make_heap + bounded
  scan + sort_heap: the same algorithm and comparator as the regular path's
  std::partial_sort), builds a k-candidate candidate array marked sorted, and
  applies the SAME sampler chain on just those k (top_k then skips the
  re-sort; dist fills p as before). Avoids the full-vocab cur_p build
  (129272 x 16 B writes) and the full-size partial_sort per draft step.
- Eligibility guards return nullptr and the caller falls back to the regular
  path: grammar or reasoning budget present, a backend-sampled token exists,
  or the chain is not exactly [top-k] or [top-k, dist].
- common/speculative.cpp draft_mtp::draft: with the env set, the draft loop
  uses the fast path; unset keeps the exact regular call sequence.

Parity contract (host-tested, section 3): identical top-k ids/logits/order and
identical drafted token whenever logits are distinct; exact-tie behavior
documented below.

### 2.3 Timeline instrumentation (Task C)

- LLAMA_SPEC_TIMELINE=1 (common/speculative.cpp draft_mtp):
  "[spec-timeline] draft: steps = N, decode_issue = X ms/step, sample+batch =
  Y ms/step, total = Z ms" per draft round, plus a per-target-batch
  "[spec-timeline] process" line for the catch-up decode.
- LLAMA_DECODE_TIMELINE=1 (src/llama-context.cpp): per process_ubatch -
  n_tokens, graph reused, build ms, inputs ms, issue ms; per ubatch outputs
  issue ms (with n_outputs and backend-sampled flag); per synchronize - drain
  ms.
- Attribution per draft step: decode_issue (host: memory prepare + graph
  build/reuse + set_inputs H2D + launch issue) | drain (device critical path:
  block + head + allreduces + output D2H transfer completion) | sample
  (drain wait + CPU chain; the decode-timeline drain line inside it splits the
  two) | batch + relaunch (next step's decode_issue). Limitation, documented:
  the block-vs-LM-head split inside the graph is not host-measurable without
  node timing (the cparams.cb_eval hook exists for a follow-up flag).

## 3. Host tests

docs/amd-port/tests/test_mtp_sampling_host.cpp (standalone, house convention,
mirrors of the two paths + llama_sampler_dist_apply). Build + run:

    g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_mtp_sampling_host \
        docs/amd-port/tests/test_mtp_sampling_host.cpp && /tmp/test_mtp_sampling_host

Result: ALL PASS (~23 s wall):

- distinct random, n_vocab 64/1k/32k/129272, ~222k trials total (uniform +
  gaussian, deduplicated to guaranteed-distinct floats): candidate arrays
  BIT-EXACT (memcmp), drafted token identical, dist draw identical.
- k-boundary near-ties (24 candidates on distinct +-12 ULP offsets straddling
  the boundary): bit-exact.
- exact ties (quantized 33-value rows + all-equal rows): survivor LOGIT
  multiset identical; drafted token identical whenever the max is unique; tie
  ORDER among exactly-tied ids may differ between heapselect variants -
  measured (43.6k/50k and 18k/20k order diffs on the saturated-tie rows). This
  is the only divergence and it can only move the seeded stochastic dist draw
  among exactly-tied candidates; the drafted token (top-1 logit) is unaffected
  unless the MAXIMUM logit is exactly tied between two ids.
- full-scale top-1 (129272, 400 trials, unique max): identical.
- edges: k clamp (k > n_vocab, k = n_vocab, k = 0), +-inf and +-1e30 extremes,
  boundary ties.

Predecessor suite test_w6_mtp_device_host.cpp re-run: ALL PASS.

Compile: g++ -std=c++17 -O1 -c -Wall -Wextra of common/sampling.cpp,
common/speculative.cpp, src/llama-context.cpp - clean; repeated with
-DGGML_USE_HIP -DGGML_HIP (host TU mirror of the gfx900 HIP build flags from
the plan: cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900 ...) - clean. (Changed
TUs are host-only C++; gfx900 codegen is unaffected.)

## 4. Deferred served-validation protocol (no GPU used by this desk)

1. Attribution cell (gate 1): campaign config + LLAMA_SPEC_TIMELINE=1
   (everything else unset), ~50 draft rounds at accept 0.66667. Fill the
   per-step table (decode_issue | drain | sample | total). Verdict rule: if
   sample_total minus the interleaved drain is ~1-3 ms, the CPU chain is minor
   and the 37 ms is device/launch/allreduce - next desk is device-side; if
   decode_issue dominates, the per-step cost is host scheduling.
2. Arm F (gate 2): LLAMA_DRAFT_FAST_TOPK=1 A/B against arm 1. Expect identical
   greedy output and accept 0.66667; t/s delta expected ~+1-3% (2-6 ms/round).
3. Arm G (gate 3, abort-expected): LLAMA_TP_BACKEND_SAMPLING=1, single draft
   round then capture the abort/behavior + exit; empirically closes the
   "where does meta fail on sampler ops" question. Do not serve on it.
4. Optional follow-up: expose cparams.cb_eval behind a flag to split LM-head
   vs block ms on the rig.

## 5. Revised estimate at 12-15 ms/draft-step

Round today: verify ~85-95 ms + 3 x ~37 ms = ~196-206 ms for ~3.0 tokens =
14.43-15.38 t/s (measured). If per-step overhead drops to 12-15 ms:

    round = 85-95 + 3x(12..15) = 121-140 ms
    t/s   = 3.0 / 0.121..0.140 = 21.4-24.8 t/s

= +40-61% over today, and an MTP multiplier of 1.76-2.04x vs MTP-OFF
(12.17-12.18 t/s measured) - up from the current 1.15-1.26x. The fast-topk arm
contributes the CPU-chain slice only (~2-6 ms/round, ~1-3%); the remaining
~20 ms/step lives in the drain/relaunch and must be taken out device-side
(launch count, allreduce latency, graph capture) - the attribution cell above
is what directs that desk.

## 6. Files

- src/llama-context.cpp: set_sampler LLAMA_TP_BACKEND_SAMPLING gate; decode
  timeline (process_ubatch build/inputs/issue, outputs issue, drain).
- common/sampling.h/.cpp: common_sampler_sample_topk + set_logits_topk
  (heap-select fast candidate selection, eligibility-guarded).
- common/speculative.cpp: draft_mtp fast-topk wiring; LLAMA_SPEC_TIMELINE
  per-step/process profilers.
- docs/amd-port/tests/test_mtp_sampling_host.cpp: parity suite (new).
- Receipt: docs/amd-port/results/MTP_overhead_2026-09-22.md (this file).
