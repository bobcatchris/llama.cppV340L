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

Committed 68f729074 (A1) on amd/draft-cache, one amendment 47a822aef (A3, defect 2 below).

- L1 llama-context (src/llama-context.cpp/h): `gf_res_shape` (2-4 extra `llm_graph_result`
  slots, env LLAMA_DRAFT_SHAPE_CACHE, default OFF), `gf_res_active()` routes
  process_ubatch/wait_outputs/fetch_nextn_outputs/opt_epoch_iter to the active slot;
  process_ubatch scans the other slots with the SAME reuse predicate
  (`cand->can_reuse(gparams)`), keeps the built graph on a shape re-entry and only
  re-splits the sched (`gf_res_shape_sched` tracks which entry the sched is allocated
  with); full misses rebuild into the LRU slot. memory_update + graph_reserve reset all
  slots (invalidation mirrors 1c b/d).
- L2 ggml-backend (ggml-backend.cpp): `ggml_backend_sched_split_graph` keeps a built
  graph's uid (fresh uid only when uid == 0); single-split views carry the graph uid.
- L3 ggml-backend-meta (ggml-backend-meta.cpp): 2-slot memo of seen graph uids ->
  per-die cgraph uids; on a re-seen uid the rebuild restores the recorded per-die uids
  so the HIP device graphs replay (ggml-cuda.cu uid-keyed
  ggml_cuda_graph_update_required) instead of re-scanning + re-capturing.
- Inert with the env unset: `gf_res_shape` empty -> `gf_res_active()` returns
  `gf_res_prev` and process_ubatch takes the unchanged single-slot path; fresh graphs
  never repeat uids, so L2/L3 code paths never fire.

## 3. A2 - host-level verification

Instrument: docs/amd-port/tests/test_draft_shape_cache_host.cpp (committed
9c229b3ae; CI replicate for the defect class, no GPU, no ggml linkage - it mirrors the
exact host decision chain: allow_reuse predicate fields, single-slot vs shape-keyed
process_ubatch policy incl. the scan/LRU/sched_is_res branches, sched split uid rule
with the L2 change, and the meta needs_rebuild + per-die uid memo with the L3 change;
every mirrored rule cites file:line). Transcript:
results/W13_draft_shape_cache_a2_host_test_2026-09-24.txt. hipcc COMPILE-EXIT:0,
RUN-EXIT:0, ALL PASS first run:

1. DEFECT replicated: single-slot steady round reused = 0,0,1,1 - 2 misses/round,
   20 graph + meta rebuilds in 10 rounds, per-die uids churn every round (HIP
   recapture class). Matches the W12 capture (issue 2.593 ms reused=0 vs 1.108
   reused=1).
2. FIX proven: shape-cache steady round reused = 1,1,1,1 from round 2 on - each shape
   built exactly once (2 builds total), 18 cached re-splits, per-die uids stable after
   warmup (HIP replay class).
3. Schedule identity: the cached re-entry sees the identical params signature as the
   fresh build ("t4_st4_sq1_o4_g1") - identical schedule decisions by construction
   (same predicate).
4. Invalidation mirrors: memory update resets both slots -> both shapes rebuild, then
   reuse resumes (0,0,1,1 warmup).
5. Predicate safety: an unseen third shape never gets a stale hit - degrades to
   today's rebuild.

Mirror-vs-real fidelity re-audited this session against 68f729074 (scan order, LRU
victim, sched_is_res gating, `params.res` used only at build time, meta
`needs_rebuild = uid == 0 || uid != cur`) - faithful for the exercised decisions.

## 4. A3 - build + pre-merge CI

- WORKTREE DAMAGE FOUND + REPAIRED (both from the dead predecessor session, both
  uncommitted): the committed test file was truncated to 0 bytes on disk (restored
  from HEAD 9c229b3ae) and build-hip/CMakeCache.txt was 0 bytes (tree wiped, fresh
  configure).
- Configure: /home/chris/opt/cmake/bin/cmake -B build-hip -DCMAKE_BUILD_TYPE=Release
  -DGGML_HIP=ON -DGGML_NATIVE=ON -DCMAKE_HIP_ARCHITECTURES=gfx900 -DGGML_HIP_RCCL=ON
  -DLLAMA_CURL=OFF -> CONFIGURE-EXIT:0.
- Full build: BUILD-EXIT:0, zero compiler warnings (only deprecation-warning.cpp
  source filenames match "warning"). Incremental rebuild after 47a822aef:
  BUILD-EXIT:0.
- CI wiring: test_draft_shape_cache_host added to the host-suite list of
  /home/chris/run_premerge_ci.sh; the script gained two additive, default-preserving
  toggles: CI_TREE (run against a desk worktree; default stays the coordinator
  checkout - supersedes W12's path-patched-copy hack) and CI_SKIP_GPU=1 (skips the
  die-3 gate-wiring section for zero-GPU desk runs; the merge gate re-runs it; the
  skip prints a loud CI-SKIP line). rpath follows CI_TREE.
- CI verdict (this tree @ 47a822aef, zero GPU): CI-VERDICT: PASS - tree clean, 7/7
  host suites (incl. test_draft_shape_cache_host), gate-wiring section CI-SKIP
  (log: results/W13_premerge_ci_2026-09-24.txt). The coordinator-side merge gate
  re-runs the full CI including the die-3 wiring section on the merged tree.

## 5. A4 - served spec for the coordinator

Arm (copy of the of-record /home/chris/launch_tp3_200k.sh, ONE line added to the env
block next to the other LLAMA_DRAFT_* envs, binary from the draft-cache build or the
merged tree, everything else identical):

    LLAMA_DRAFT_SHAPE_CACHE=1 \   # -> 2 slots (catchup/step pair), engagement line:
                                  #    "llama_context: draft shape cache enabled (2 slots)"

Gates for the verdict (E-117 law: engagement REQUIRED; E-119 law: interleaved A/B for
sub-5% claims):

1. Engagement: the INFO line above present in the server log.
2. Mechanism witness (free, same boot): [decode-timeline] draft-ctx decodes read
   reused=1 on the catchup AND step 1 in steady rounds, issue medians ~1.1 ms
   (vs 2.593 reused=0 in W12). NOTE: "[launch-timeline] meta rebuild:" now fires
   ~2x/round by design (the uid alternation still re-derives the per-die subgraphs;
   the memo keeps their uids stable so the device graphs REPLAY) - do not misread it
   as the W12 steady-state-rebuild regression; the recapture tripwire
   (ggml-cuda.cu warmup reset) must stay SILENT after boot.
3. Battery: provenance-gated guard_battery.py vs baseline_tp3_200k.json, all 5 cells
   PASS (decode 23.23/23.34, prefill 217.71, accept 0.66667, needle recall,
   determinism_greedy byte-identical within boot - the change is host-only byte-exact
   class).
4. Expected delta: up to ~3 ms/round on a ~123-129 ms round = ~+2-3% decode - at the
   edge of the instrument; bank only via interleaved A/B (E-119 soak law) plus the
   mechanism witness in (2), which is decisive on its own.
5. Rollback: unset the env - default OFF path is untouched (single-slot behavior
   byte-identical; L2/L3 unreachable).

## 6. Defects of record

1. Predecessor session left two 0-byte truncations in this worktree (test file on
   disk, build-hip/CMakeCache.txt); both repaired (git checkout / fresh configure).
   The committed test blob itself was intact.
2. A1 DEFECT (found in the A4 spec review, fixed 47a822aef): LLAMA_DRAFT_SHAPE_CACHE=1
   allocated ONE slot - with one slot the scan always skips itself and the LRU wrap
   returns to the same slot, i.e. the default single-slot thrash path verbatim: the
   served arm would print the engagement INFO line yet deliver ~0 ms (E-117-class
   inert gate). Fix: 1 -> the default 2 slots (the block's own stated intent);
   2-4 remain explicit slot counts.
3. Residual cost (honest bound): the two shape re-entries per round still re-derive
   the meta per-die subgraphs (remap, 49-node draft graph - small); the eliminated
   terms are the graph rebuild and the HIP recapture (the P0-dominant term). Served
   delta may therefore land under the ~3 ms headline; the [decode-timeline] witness
   decides.
4. Cached-params lifetime note: a shape-cache hit reads the SAVED ubatch params from
   the previous round (samplers output[]/seq_id content compares) - the same
   one-round lifetime class the target ctx exercises cross-round today on the
   single-slot path; no new lifetime class introduced.
5. CI gap closed this session: the suite list edit + CI_TREE/CI_SKIP_GPU toggles live
   in /home/chris/run_premerge_ci.sh (outside the repo); the die-3 gate-wiring section
   did NOT run on this desk tree (zero-GPU law, served window in flight) - the merge
   gate must run the full CI.
