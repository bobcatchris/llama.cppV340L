# Verify-round desk: host-slice profile table + env-gated fixes + transport/replay designs

Date: 2026-09-22. Worktree: wt-verify-round (branch amd/verify-round, campaign
HEAD f097c9f6e). ZERO GPU: offline profiling of the T1/F1 timeline logs, code,
host tests, gfx900 compile only. No boot, no die time, lane 8083 untouched.

## 1. HOST SLICE PROFILE (work item a) - the 40-55 ms attribution is wrong

Method: the raw logs carry wall-clock timestamps (min.sec.milli.micro) on every
[spec-timeline] / [decode-timeline] line. Segmented all F1 rounds (verify
decode = n_tokens 4 followed by outputs with n_outputs > 0; catch-up = second
n_tokens 4 decode with n_outputs 0) and measured every host gap from the
decode-entry back-calculation (entry = line_ts - build - inputs - issue).
Profiler: docs/amd-port/scripts/profile_verify_round.py (re-runnable).

F1 (85 rounds, decode-timeline on), median ms/round:

| component                                            | med ms | share | class |
|------------------------------------------------------|--------|-------|-------|
| verify decode issue (ctx_tgt, 4 tok, in graph_compute)| 163.16 | 85.9% | DEVICE |
| catch-up decode issue (ctx_dft, 4 tok, in process()) |  11.66 |  6.1% | DEVICE |
| draft loop total (3 steps)                           |  11.02 |  5.8% | 7.55 DEVICE issue + 3.44 host |
| verify output D2H wait (drain inside h_tgt getter)   |   1.90 |  1.0% | DEVICE/PCIe |
| target sampling + post_decode rest + loop (G3)       |   1.03 |  0.5% | HOST |
| catch-up batch build (process entry to decode entry) |   0.29 |       | HOST |
| outputs extract + h-row copies + residual            |   0.75 |       | HOST |
| round wall (draft-line deltas)                       | 190.04 |       |       |

T1 cross-check (210 rounds, spec-timeline only): wall med 195.2, draft total
med 10.92, process total med 14.47, residual (wall - draft - process) med
168.3 = verify 163.2 + catch-up path host + rest. Consistent.

VERDICT (supersedes the E-077 host/device partition): the "~40-55 ms/round
host slice" DOES NOT EXIST as host time. It was derived from the pooled
n_tokens=4 population (E-073/E-077 pooled verify + catch-up lines: n=170,
med 138.1) which UNDERSTATED the verify issue (true med 163.2) and counted
the catch-up decode (11.7 ms of device) as host. The true host slice is
~5.5-8 ms/round (3-4%): draft sample+batch 3.44 (already attacked by
FAST_TOPK / PACKED_GET / LIGHT_SYNC), target sampling ~0.7 of the 1.03 G3
window, batch builds ~0.7, residual ~0.7. The round is ~91-97% device
blocking: verify 163.2 + catch-up 11.7 + draft issue 7.55 + D2H 1.9.

Target-side candidate audit (from the brief):
- per-token candidate builds on the target sampler: 4 rows x 129272
  candidates (~2 MB each) per verify = ~0.66 ms - real but small; the target
  chain is request-shaped (temp 0 guard here), so the draft FAST_TOPK trick
  does NOT transfer (eligibility, not implementation).
- redundant logits copies: NONE. decode() extracts with ONE
  ggml_backend_tensor_get_async of the whole [n_outputs, n_vocab] block;
  llama_get_logits_ith returns a pointer (no copy).
- redundant scheduler sweeps: CONFIRMED on the target side - every
  common_sampler_sample in the accept loop pays 6 synchronize calls
  (1 explicit + 3 sampled-getter probes + the logits getter + the
  backend-token probe), x4 rows = ~24 syncs/round. This is the E-072 audit
  finding, alive on the target path. FIXED (below).

## 2. IMPLEMENTED (env-gated; unset = byte-identical)

### 2.1 LLAMA_VERIFY_ROW_SAMPLING - target verify-accept row path

The accept loop (server post_decode, common_sampler_sample_and_accept_n)
samples the 4 verify rows through the ctx getters, one synchronize per
getter. The row path: ONE llama_wait_outputs light drain, a backend-sampled
token probe on the first row, one llama_peek_logits_rows call (new staged
API, src/llama-ext.h) handing the raw row pointers, then per-row
common_sampler_sample_row (E-072 API) + common_sampler_accept with the same
break-on-mismatch semantics. Same rows, same set_logits_row build, same
chain: the accepted tokens are identical by construction; only ~22 of ~24
per-round sync sweeps are deleted. Guards return empty and the caller falls
back to the regular loop unchanged: grammar, reasoning budget, backend
sampler attached (probe on row 0 - a backend sampler produces tokens for
every row), short peek. Caveat (same as LIGHT_SYNC): the light drain does
not close the ctx perf evaluation window while the flag is set.

Files: src/llama-ext.h + src/llama-context.{h,cpp} (llama_peek_logits_rows,
no-sync multi-row pointer resolution), common/sampling.{h,cpp}
(common_sampler_sample_and_accept_n_rows), tools/server/server-context.cpp
(env gate + fallback in post_decode).

### 2.2 GGML_PINNED_DEV_COPY - pinned staging for the peer-copy fallback

Transport audit (work item b, ground truth): with no peer copy
(GGML_CUDA_NO_PEER_COPY, canPeerPeer = 0), EVERY butterfly peer copy takes
ggml_backend_tensor_copy_async -> cpy_tensor_async false ->
synchronize(src) + synchronize(dst) -> ggml_backend_tensor_copy ->
ggml_backend_cuda_buffer_cpy_tensor returns false (NO_PEER_COPY) ->
malloc + BLOCKING unpinned D2H + BLOCKING unpinned H2D + free. The verify
ubatch runs 2 boundaries, and per decode round the split runs 5 decodes
(verify + catch-up + 3 draft steps) x 2 boundaries = 10 boundaries x 4 peer
copies = 40 slow copies + 80 full device drains per round.

The quick win: ggml_backend_dev_copy_staging (ggml/src/ggml-backend.cpp) - a
grow-only cached PINNED host buffer (1 MiB start, x2 growth) from the source
device's host buffer type replaces the per-call malloc/free; the get/set
sequence is unchanged so the bytes are identical by construction. Falls back
to the historical malloc path when the env is unset, the device has no host
buffer type, or the alloc fails. Locked (shared cache). Expected on the rig:
removes 40 malloc/free + unpinned-transfer penalties per round; the 80
device drains remain (semantics) - see the design below for those.

### 2.3 Host tests

- docs/amd-port/tests/test_verify_row_sampling_host.cpp (new): both accept
  loops mirrored end-to-end over the same rows; accepted token sequences and
  break positions BIT-EXACT for greedy (campaign temp-0) and seeded
  [top-k(10), dist] chains across vocabs 64 / 1k / 32k / 129272 (uniform +
  gaussian, ~1900 trials); candidate arrays memcmp-identical; exact-tie rows
  accept identically; guard cases return empty (grammar / budget /
  backend-token). ALL PASS; ASAN/UBSAN clean.
- docs/amd-port/tests/test_pinned_staging_host.cpp (new): cache logic
  mirrored - reuse without realloc, geometric growth with old-buffer free,
  byte-exact staged vs malloc copies at 1 KiB..5 MiB + odd sizes, env-unset
  and no-host-buft fallbacks. ALL PASS; ASAN/UBSAN clean.
- Existing suites re-run on this tree: test_mtp_sampling_host,
  test_packed_get_host, test_w6_mtp_device_host, test_server_exposures_host
  - ALL PASS.
- gfx900 compile: cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900
  (Release, /home/chris/opt/cmake/bin/cmake), targets llama + llama-common +
  llama-cli: built clean, ZERO warnings (all changed TUs compiled:
  ggml-backend.cpp, llama-context.cpp, sampling.cpp, server-context.cpp).

## 3. DESIGNED (no code)

### 3.1 Allreduce transport (work item b, the rest of the lever)

The transport is PCIe-through-host regardless of design: canPeerPeer = 0,
so a die-to-die DMA does not exist. Levers, in order of value:

1. comm_allreduce on the HIP backend (the structural fix): the meta backend
   already probes ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init")
   and "ggml_backend_comm_allreduce_tensor" and, when present, bypasses the
   host butterfly entirely (one call per boundary, device-side, no host
   staging). The HIP/ROCm backend does not register a comm - RCCL
   integration is the project. This deletes the 40 copies + 80 drains per
   round, not just their overhead. Subsystem-scale: comm_ctx lifetime,
   multi-die collectives, fp sum order = the acceptance-critical question
   (same caveat as head re-shard).
2. Event-ordered async staging (host-side, no sum-order change): replace
   sync(src)+sync(dst)+blocking-copy with get_async into a pinned ring slot
   on the src stream, event record on src, dst stream waits the event, then
   set_async on the dst stream, plus a per-slot done event for slot reuse.
   Semantically identical data flow (stream ordering preserves every
   dependency), but the dst die stops draining on every copy and the copies
   of the two independent butterfly pairs can overlap. Requires >= 4 ring
   slots per (src,dst) pair (2 concurrent pushes per boundary round x
   safety). GGML_PINNED_DEV_COPY staging buffers are the natural slot
   storage. Est. -2..-6 ms/round on today's numbers.
3. Copy consolidation: NOT available at this granularity - the 4 copies per
   boundary are 4 distinct (src,dst) pairs (n=3 fold + butterfly + copy
   back); there is nothing to merge into fewer larger copies.

### 3.2 Replay count / graph cut (work item c): VERDICT - NOT CLEAN, do not implement

The meta splitter cuts the graph exactly at PARTIAL-axis nodes
(new_subgraph = last node || split_state.axis == PARTIAL); 2 PARTIAL nodes
per nextn block -> 3 subgraphs x 3 dies = 9 replays + 2 boundaries. Raising
the cut means delaying a reduction past a replicated consumer (norm /
activation between attn.out and mlp.down run on every die and need the fully
summed input) - the boundary count is ARCHITECTURAL to tensor parallelism,
not a tuning knob. The two reductions are sequentially dependent through the
block, so they cannot be folded into one either. The honest replay-count
lever is not fewer cuts but cheaper boundaries: 3.1.1 above (comm_allreduce)
or 3.1.2 (async staging). A note for the device desk: the catch-up decode
(11.7 ms med for a 4-token pass of the nextn block alone, vs 2.5 ms for a
1-token draft step of the same graph) re-decodes ALL verify tokens including
the ~1.3/round that get rejected, then the rejected KV is seq_rm'd next
round - an accepted-prefix-only catch-up (post-sampling) would delete
boundary+replay work, but it reorders KV rollback and is NOT mechanical.

## 4. SERVED-VALIDATION ARMS for the measurement desk

Rebuild required (this tree). All arms: campaign line (v1 launch script
default) + LLAMA_SPEC_TIMELINE=1 LLAMA_DECODE_TIMELINE=1 + -lv 4, decode-only
guard + determinism, accept 0.66667 / mean_len 3.00 / draft 126-84 gate in
every arm - ANY accept move is a bug, not noise.

| arm | env | expected signature (vs F1 baseline) |
|-----|-----|-------------------------------------|
| R1  | (unset)             | baseline reproduces: verify med ~163, catch-up ~11.7, draft total ~11, wall ~190-195 |
| R2  | LLAMA_VERIFY_ROW_SAMPLING=1 | identical greedy sha; the drain-line groups between the [spec-timeline] process line and draft step 1 collapse (about 22 fewer [decode-timeline] drain lines per round); G3 window -0.3..-0.6 ms; t/s +0.2..+0.5% class |
| R3  | GGML_PINNED_DEV_COPY=1 | identical greedy sha; verify issue med -2..-6 ms (boundary stalls), draft step issue -0.2..-0.5 ms; t/s +1..+3% class |
| R4  | R2 + R3             | additive within noise; combined round -3..-7 ms |

Engagement proofs: R2 - drain-line count per round in the log; R3 - presence
is binary (env), effect reads on the verify issue distribution.

## 5. Files

- src/llama-ext.h, src/llama-context.{h,cpp}: llama_peek_logits_rows.
- common/sampling.{h,cpp}: common_sampler_sample_and_accept_n_rows.
- tools/server/server-context.cpp: LLAMA_VERIFY_ROW_SAMPLING gate + fallback.
- ggml/src/ggml-backend.cpp: GGML_PINNED_DEV_COPY staging cache.
- docs/amd-port/tests/test_verify_row_sampling_host.cpp (new).
- docs/amd-port/tests/test_pinned_staging_host.cpp (new).
- docs/amd-port/scripts/profile_verify_round.py (new; offline profiler).
- Receipt: docs/amd-port/results/VERIFY_ROUND_2026-09-22.md (this file).
- Ledger: E-078 appended to docs/amd-port/OPTIMIZATION_PLAN_TP3_200K.md.
