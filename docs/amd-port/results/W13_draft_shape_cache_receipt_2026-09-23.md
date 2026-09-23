# W13 DRAFT-SHAPE-CACHE DESK - P0 + A1 + A2 + A3 (+ A4 spec for the coordinator)

Date: 2026-09-23. Desk: DRAFT-SHAPE-CACHE (wt-draft-cache, amd/draft-cache @ fdb235c80 base).
Lever of record: W12 host-slice receipt section 3 lever 1 (draft-ctx graph/sched shape cache,
~2-3 ms/round). Zero GPU (host chain under test is CPU logic; interleaved A/B window in flight).

## 1. P0 - mechanism verdict

### 1a. The trigger is CONFIRMED as briefed

- Single reuse slot: `llama_context::process_ubatch` consults exactly one previous graph result,
  `gf_res_prev` (src/llama-context.cpp:1579), reusing it only when
  `!graph_reuse_disable && res->can_reuse(gparams)` (src/llama-context.cpp:1586); any miss runs
  `res->reset()` + `ggml_backend_sched_reset()` + full rebuild + `ggml_backend_sched_alloc_graph()`
  (src/llama-context.cpp:1600-1627).
- The reuse predicate `llm_graph_params::allow_reuse` (src/llama-graph.h:706-775) keys on the ubatch
  shape (n_tokens, n_seq_tokens, n_seqs, n_seqs_unq, equal_seqs, token/embd presence, seq ids),
  n_outputs, samplers, nextn offset, embeddings/causal flags, arch, gtype and the adapter/cross
  pointers. A 4-token catchup ubatch can never match a 1-token step ubatch, so the draft context
  thrashes the single slot.
- Observed pattern (W12_hostslice_diag_tl_165820.log, steady-state round, [decode-timeline] lines,
  draft ctx = the 49-node graphs; the 4278-node reused=1 lines are the target ctx verify):
  `4 reused=0 (catchup) -> 1 reused=0 (step 1) -> 1 reused=1 -> 1 reused=1`, i.e. 2 misses/round;
  issue 2.5-2.9 ms on the 1-token misses vs 1.08-1.17 ms on hits (matches W12 table 5a/5b,
  2.593 vs 1.108 ms). Cross-checked in W12_hostslice_a1_rounds.txt (draft section 10.6 ms/round).

### 1b. Cost attribution CORRECTED (the W12 lever text is incomplete)

W12 attributed the ~1.5 ms/miss to the "sched re-split walk" at llama-context.cpp:1579-1614. That
walk is NOT what the miss pays inside the measured `issue` window: the issue timer brackets only
`graph_compute` (src/llama-context.cpp:1646-1656). The actual miss chain, bottom-up:

1. `ggml_backend_sched_reset` wipes ALL tensor->backend-id assignments and tensor copies
   (ggml/src/ggml-backend.cpp:1931-1941), so every re-split re-derives the schedule from scratch
   (ggml_backend_sched_split_graph, ggml/src/ggml-backend.cpp:1080).
2. Every re-split stamps FRESH graph uids: the input graph (ggml/src/ggml-backend.cpp:1099,
   `graph->uid = ggml_graph_next_uid()`) and the split views (ggml/src/ggml-backend.cpp:1551).
3. The meta backend rebuilds its per-die subgraphs whenever the incoming uid differs from its
   SINGLE stored uid (`needs_rebuild`, ggml/src/ggml-backend-meta.cpp:1862), including the
   simple-tensor double-buffer flip (ggml/src/ggml-backend-meta.cpp:1896-1906), the per-die
   subgraph re-derivation and fresh per-die graph uids (ggml/src/ggml-backend-meta.cpp:2126).
4. Fresh per-die uids defeat the HIP device-graph replay: `ggml_cuda_graph_update_required`
   replays only when `cgraph->uid == graph->uid` (ggml/src/ggml-cuda/ggml-cuda.cu:3455-3464);
   a uid change re-scans node properties and re-captures. This is the dominant term and it lands
   inside the decode issue timer on the first compute after the re-split.

Consequence: a llama_context-only shape cache would save only the graph rebuild
(measured build = 0.02-0.04 ms/decode in the W12 log), i.e. ~0.06 ms/round - NOT the ~3 ms prize.
The prize is collectable only if the ggml sched + meta layers also recognize the shape re-entry.

### 1c. Design conclusions drawn in P0 (verified in code)

- The cache key is the EXISTING reuse predicate: `can_reuse(gparams)` on the cached
  `llm_graph_result` (same params copy compared by `allow_reuse` + per-input `can_reuse`,
  src/llama-graph.cpp:1396-1429). No new hashing; decisions per shape are identical to a fresh
  build by construction.
- Memory cost per cached entry (draft ctx, 1-layer block): `llm_graph_result` holds
  buf_compute_meta = ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes)
  (src/llama-graph.cpp:1305-1336) with max_nodes = max(1024, 8*n_tensors) for the draft arch
  (src/llama-context.cpp:2674-2687) -> order 0.5 MB host per entry; device temporaries are shared
  (single sched/galloc). 2 entries are trivial.
- Invalidation triggers of the single slot today, which the shape cache must mirror:
  (a) construction at init (src/llama-context.cpp:493);
  (b) `memory_update` after the memory module performed an update - context shift, KV defrag or
      resize all arrive here (src/llama-context.cpp:979-984, `gf_res_prev->reset()`);
  (c) `graph_reserve` after a scheduler reset (src/llama-context.cpp:2705-2707);
  (d) adapter changes set `sched_need_reserve` (src/llama-context.cpp:1560) which routes through
      sched_reserve -> graph_reserve, and `allow_reuse` additionally compares cvec/loras pointers
      (src/llama-graph.h:772-773).
- Address stability for device-graph replay on re-entry: the galloc re-applies the saved
  placement plan for an unchanged graph (ggml_gallocr_needs_realloc/alloc_graph,
  ggml/src/ggml-alloc.c:1021-1078) and skips re-allocation entirely for tensors that already
  carry data (ggml_gallocr_init_tensor, ggml/src/ggml-alloc.c:996-1018); cached-entry tensors keep
  their device addresses across re-entries, and the meta simple-tensor pool recreates per-die
  structs deterministically (ggml_reset + identical creation order), so captured per-die graphs
  keyed by a STABLE uid replay validly.

## 2. A1 - implementation (3 layers, all inert with LLAMA_DRAFT_SHAPE_CACHE unset)

(to be filled at A1 completion)

## 3. A2 - host-level verification

(to be filled at A2 completion)

## 4. A3 - build + pre-merge CI

(to be filled at A3 completion)

## 5. A4 - served spec for the coordinator

(to be filled at A4 completion)

## 6. Defects of record

(to be filled at D)
