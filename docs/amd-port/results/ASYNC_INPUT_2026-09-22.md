# Async-input desk: blocking H2D input-set map, pinned-ring staged sets, honest floor

Date: 2026-09-22. Worktree: wt-async-input (branch amd/async-input, campaign HEAD
c20b3f547). ZERO GPU: code audit of the set_inputs path, implementation, host
tests, gfx900 compile only. No boot, no die time, lane 8083 untouched.

Attacked number (ledger E-072 S1 / E-083): the 4-6 blocking H2D input sets per
decode step - decode entry pays `ggml_backend_tensor_set` per input tensor,
which on the HIP backend is `cudaMemcpyAsync + cudaStreamSynchronize +
cudaSetDevice` per tensor (ggml-cuda.cu buffer iface), and behind the TP3 meta
backend that fans out to one blocking round trip PER DIE (MIRRORED inputs).

## 1. MAP - every per-round blocking H2D set (MTP round, campaign config)

Blocking set = `ggml_backend_tensor_set` on a device-resident input tensor;
die-level = the meta MIRRORED fan-out (3 dies). Sizes at n_tokens 1/4; the
draft h row is n_embd f32 (16-32 KB, E-072).

| ubatch per round | ctx | input class | tensor | bytes | blocking sets | die-level round trips |
|---|---|---|---|---|---|---|
| draft step 1-3 (x3) | ctx_dft | embd_h | tokens | 4 B | 1 | 3 |
| draft step 1-3 (x3) | ctx_dft | embd_h | h row | 16-32 KB | 1 | 3 |
| draft step 1-3 (x3) | ctx_dft | pos | pos | 4 B | 1 | 3 |
| verify | ctx_tgt | embd | tokens | 16 B | 1 | 3 |
| verify | ctx_tgt | pos | pos | 16 B | 1 | 3 |
| catch-up | ctx_dft | embd_h | tokens | 16 B | 1 | 3 |
| catch-up | ctx_dft | embd_h | h rows (4) | 64-128 KB | 1 | 3 |
| catch-up | ctx_dft | pos | pos | 16 B | 1 | 3 |

Total per round: 13 blocking tensor sets = 39 die-level blocking round trips
(E-072's "4-6 per step" counted tensor granularity, not die fan-out).

NOT blocking H2D sets (audited, stay untouched): out_ids, kq_mask,
k/v idx tensors, k_shift, s_copy, cls/mean - all live in HOST buffers
(`ggml_backend_buffer_is_host` asserts in the setters) and are written in
place; the sched moves them device-side inside graph_compute as stream-ordered
copies. The pos M-RoPE 4D conversion (n_pos_per_embd = 4) builds pos_data on
the host, then one set - same path. attn_temp (llama4) and cross_embd (mmproj)
route through the new helper but are not on this workload's path.

Measured host wall of set_inputs (ground truth, F1 decode-timeline log,
170 n_tokens=4 lines + 254 n_tokens=1 lines): verify/catch-up inputs mean
0.027 ms (max 0.051), draft-step inputs mean 0.010 ms -> ~0.08-0.15 ms/round
total in set_inputs, i.e. ~2-3% of the ~5.5-8 ms host slice and ~0.05-0.08%
of the ~190 ms round. E-072's "~50-300 us per step" upper range was
pessimistic; the F1 rig truth is ~10-30 us per ubatch.

## 2. IMPLEMENTED (env-gated; unset = byte-identical)

LLAMA_ASYNC_INPUT=1 - pinned-ring staged input sets:

- `llama_input_tensor_set` (src/llama-graph.cpp) replaces the
  `ggml_backend_tensor_set` calls in the per-ubatch input setters (embd,
  embd_h, pos, attn_temp, cross_embd). With the env set and a scheduler
  handed to `llm_graph_result::set_inputs(ubatch, sched)`, the bytes are
  memcpy'd into a pinned ring slot and copied with
  `ggml_backend_tensor_set_async` on the backend that owns the tensor
  (resolved via `ggml_backend_sched_get_tensor_backend`): for the meta backend
  the MIRRORED fan-out posts one async DMA per die and returns; the device
  tensor addresses are unchanged, so graph reuse and capture/replay see the
  same stable destinations. The copied bytes and their order are identical by
  construction; only the staging and the synchronization differ.
- Ring: 8 grow-only slots (1 KiB start, x2), pinned host buffers from the
  owning device's host buffer type (the same GGML_PINNED_DEV_COPY house
  pattern). A slot is reused only after its previous staged copy completed:
  gated by a per-slot backend event when the device supports events, otherwise
  by `ggml_backend_synchronize` of the backend the copy was issued on - for
  the meta backend that drains all die streams, and in the decode loop it is
  quiescent at that point (outputs of the previous ubatch are consumed before
  the next set_inputs; pipeline_parallel is false in TP mode), so the gate
  costs ~0. Static ring + mutex (thread-local sched pointer for the setter);
  safe with ctx_tgt + ctx_dft alternating on the same ring.
- Fallbacks to the historical blocking set: env unset, no scheduler (K-shift
  and opt/training call sites pass none), host-buffer tensors, missing host
  buft, alloc failure, offset != 0, size == 0.
- One INFO line at first engagement for the served-log signature.

Files: src/llama-graph.{h,cpp} (staging + setter routing + set_inputs
signature), src/llama-context.cpp (process_ubatch passes sched).

## 3. HOST TESTS

docs/amd-port/tests/test_async_input_host.cpp (new, house convention, mirrors
the setter control flow 1:1 against a fake backend/event/stream layer):

- byte-exactness + ring-overwrite safety: 50 rounds of the real round shape
  (draft steps + verify + catch-up, tokens/pos/h, two backends alternating)
  with drain timing varied; every issued transfer lands exactly the bytes the
  setter handed over (shadow at issue == landed at completion), so no slot was
  ever overwritten while its copy was in flight;
- no-event devices (the meta device reports caps.events = false): the gate is
  the issuing backend's synchronize, correct across contexts;
- event-capable devices: per-slot event gate, re-created on device change;
- growth: geometric slot growth, no realloc within capacity, no frees;
- fallbacks: env unset / no sched / host tensor / no host buft / alloc fail /
  offset != 0 / size 0 all keep the blocking set and issue nothing staged.

ALL PASS; ASAN/UBSAN clean. Existing suites re-run on this tree:
test_packed_get_host, test_verify_row_sampling_host, test_pinned_staging_host,
test_mtp_sampling_host - ALL PASS. gfx900 compile clean via
/home/chris/opt/cmake/bin/cmake (-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900,
Release): llama, llama-common, llama-cli.

## 4. HONEST SIZING - the floor

The lever deletes host round trips only; it does NOT delete the DMA time
itself (the posted copy still must land before the consuming graph's first op
reads the tensor - stream ordering, unchanged).

- Decode (the attacked path): ceiling = the whole set_inputs wall,
  ~0.08-0.15 ms/round, minus the added pinned memcpy (~us) and the gate (~0
  in the quiescent loop) => ~0.05-0.10 ms/round recoverable. Against the
  ~190 ms round: < 0.1%, t/s +0.0-0.1% (14.94 -> at most ~15.0). The verify
  decode's 163 ms device chain is untouched; this lever is two orders below
  it. Floor statement: preposted tiny copies bottom out at the host enqueue
  (~2-5 us per copy) + a quiescent gate.
- Draft-ctx prefill chunks (secondary, not the target): the h-row set is
  n_tokens x n_embd f32 (8-16 MB at 512); today that is a driver-staged
  pageable blocking copy (~0.3-0.9 ms of the measured 0.27-0.91 ms inputs per
  chunk). Staged pinned + posted before graph issue may recover a fraction and
  overlap the issue host work - rig-measurable only, +0.0-0.5% prefill class.
  Target prefill chunks (tokens/pos only) shrink by the round-trip share,
  ~0.03-0.09 ms per chunk.

VERDICT: S1 is real (13 blocking sets / 39 die round trips per round, all
mapped) but its wall-clock share at 200k decode is ~0.1 ms/round - a hygiene
win that removes the per-die cudaSetDevice/synchronize round trips and the
driver's pageable staging from the decode critical path, and a NEGATIVE result
for the speed offensive: async input cannot move the 15 -> 21+ t/s needle. It
confirms E-072/E-078: the round is owned by the verify-batch device serial
chain and the boundary transport (verify-transport desk's lever).

## 5. SERVED-VALIDATION ARMS (need rebuild; fold into the combined window)

All arms: campaign launch line + LLAMA_SPEC_TIMELINE=1 LLAMA_DECODE_TIMELINE=1
+ -lv 4, decode-only guard + determinism; accept 0.66667 / mean_len 3.00 gate
in every arm - any accept move is a bug, not noise (bytes identical by
construction).

| arm | env | expected signature vs baseline |
|-----|-----|--------------------------------|
| A1  | (unset) | baseline reproduces: inputs 0.01-0.05 ms/ubatch, verify med ~163, wall ~190-195 |
| A2  | LLAMA_ASYNC_INPUT=1 | identical greedy sha; one "LLAMA_ASYNC_INPUT staged input sets enabled" INFO at first decode; [decode-timeline] inputs columns collapse to ~0.00-0.01 ms/ubatch; round wall -0.0..-0.15 ms (inside noise); t/s +0.0-0.1% |

Engagement proof: the INFO line + the collapsed inputs columns (decode
timeline). Composable with every merged arm (PACKED_GET, LIGHT_SYNC,
VERIFY_ROW_SAMPLING, PINNED_DEV_COPY) - no interaction: staging changes where
the input bytes come from, nothing else.

## 6. Files

- src/llama-graph.h: set_inputs takes the scheduler (default nullptr).
- src/llama-graph.cpp: LLAMA_ASYNC_INPUT staging ring + llama_input_tensor_set
  + setter routing.
- src/llama-context.cpp: process_ubatch passes sched.
- docs/amd-port/tests/test_async_input_host.cpp (new).
- Receipt: docs/amd-port/results/ASYNC_INPUT_2026-09-22.md (this file).
- Ledger: E-084 appended to docs/amd-port/OPTIMIZATION_PLAN_TP3_200K.md.
