# W6 - 4th die as draft device: zero-GPU design + implementation

Date: 2026-09-21. Desk: wt-draft-dev, branch amd/w6-draft-device.
State: ZERO-GPU phase COMPLETE (design + implementation + compile-check + host
logic tests). NOT RUN - served validation needs a negotiated window where dies
0-2 serve and die 3 drafts (protocol at the end of this file).

## What ships

`--spec-mtp-device <DEV>` (aliases `-devm`, `--device-mtp`; env
`LLAMA_ARG_SPEC_MTP_DEVICE`), accepted as a device name (`CUDA3`) or a bare
index into the visible GPUs (`3`). When set, together with `--spec-type
draft-mtp`, three pins move the MTP draft head off the TP3 group onto die 3:

1. nextn-layer weights (`blk.64.*`, 15 tensors, 169.3 MiB) load whole into a
   plain die-3 buffer instead of the meta (TP3) buffer;
2. the draft context's KV cache (layer 64 only, q4_0, ~220 MiB at 200k)
   allocates on die 3;
3. the draft context's backend list becomes `[meta(TP3), die3, CPU]` instead
   of `[meta, CPU]`, so the draft graph runs its block on die 3 while the
   shared LM head and token-embedding tensors stay on the TP group.

Unset (default) = byte-for-byte today's behavior; every new code path is
gated on `dev_mtp != nullptr` / `extra_device != nullptr`.

## Device-inheritance audit (why three pins are needed)

The draft context is a second `llama_context` on the SAME loaded model
(`server-context.cpp:1245`, `llama_init_from_model(model_tgt, cparams_mtp)`).
It inherits the target's placement at three independent points:

1. Context backends: `llama_context` ctor builds its backend list from
   `model.devices` (src/llama-context.cpp, GPU-backends loop). For a TP3 boot
   that list is the single meta device over dies 0-2, so the draft context
   used to get exactly the target's backends. Fix: new
   `llama_context_params.extra_device` appended after the model's devices
   (skipped if duplicated); server sets it at server-context.cpp
   (`cparams_mtp.extra_device`).
2. Weights: tensor buffer types come from the loader's per-layer buft lists
   (the meta buft under `-sm tensor`). Under the meta device every tensor is
   either row-sharded (AXIS_0/1) or mirrored (MIRRORED) across dies 0-2 only;
   die 3 holds nothing. Fix: new `llama_model_params.dev_mtp` threaded into
   the loader; in `buft_for_tensor` (llama-model-loader.cpp) any block tensor
   with `bid >= hparams.n_layer()` (the MTP layers) goes to
   `ggml_backend_dev_buffer_type(dev_mtp)`. Explicit `-ot` overrides still win.
3. KV cache: `llama_kv_cache` picks the per-layer device via
   `model.dev_layer(il)` (src/llama-kv-cache.cpp). Fix: `llama_model::dev_layer`
   returns `dev_mtp` for `il >= hparams.n_layer()`; the target context is
   unaffected because its hybrid cache filters the MTP layer out
   (`filter = il < hparams.n_layer()`, llama-model.cpp create_memory).

Notes on alternatives rejected:
- No "split mode forced to none/layer" is needed for MTP: the model is shared,
  and per-context placement (above) achieves the pin without re-splitting.
  (The separate-draft-model path already has full placement control via
  `--spec-draft-device`; `dev_mtp` is deliberately cleared for that path -
  server-context.cpp, "draft model placement is owned by --spec-draft-device".)
- Overriding `output.weight` or `token_embd.weight` to die 3 is impossible
  without penalizing the target: the target verify pass computes its logits
  through `model.output` (517.8 MiB Q6_K, TP3 row-sharded with the allreduce),
  and the input layer does `get_rows(model.tok_embd)` (335.3 MiB IQ4_XS,
  mirrored). The GGUF carries `blk.64.nextn.{eh_proj,enorm,hnorm,
  shared_head_norm}` but NOT `shared_head_head`/`embed_tokens`, so the draft
  graph falls back to those same shared tensors (qwen35.cpp graph_mtp:
  `head_w = shared_head_head ? ... : model.output`).
- The scheduler hard-aborts on pre-allocated weights that no sched backend
  supports (ggml/src/ggml-backend.cpp, GGML_ABORT "pre-allocated tensor ... in
  a buffer ... that cannot run the operation"), so a naive
  `[die3, CPU]` draft backend list would crash on the first head matmul.
  Load-time placement, not per-step copies, is the only sane route.

## Scheduler placement that follows from the pins

Draft graph (graph_mtp) per step, n_out = 1..4 rows:

- enorm/hnorm/eh_proj/attn/ffn/shared_head_norm (weights on die 3): run on
  die 3; KV reads/writes die 3; no allreduce (single device).
- `get_rows(tok_embd)` and the LM-head matmul (weights in the meta buffer):
  run on the meta backend (TP3, keeps the 3x aggregate bandwidth for the head);
  the sched inserts two ~20 KiB activation copies per step (token row
  meta->die3, head input die3->meta; host-staged, no P2P on this box).
- Backend sampling stays attached to ctx_dft; logits live on the meta side,
  so the sampled id crosses to host once per step (as today).
- `h_nextn` output: graph result on die 3, copied into buf_output (host-pinned
  via the meta device's host buft) exactly like today's meta->host copy.

## Die-3 memory budget (8 GiB / 8573 MiB cards)

- blk.64 weights moved whole: 169.3 MiB
- MTP KV (1024+1024 dim, q4_0, GGML_PAD(200000,256) ctx): ~220 MiB
- compute + output buffers (ubatch 128-512 draft/warmup graphs): ~100-300 MiB
- total: roughly 0.5-0.7 GiB, leaving die 3 usable as the dev/bench cell
  between serving windows (and as the draft device during them).

The fit-params estimate (`common_get_device_memory_data`) attributes per-buft
breakdown against the model's own device list (meta only under `-sm tensor`),
so the MTP context bytes were already not added to the TP3 reservation before
this change; after the change they physically land on die 3. No regression,
slightly more TP3 headroom in practice.

## Risks (honest list)

1. KV cache sync overhead per draft step: KV is now single-device, so seq_rm/
   state copies touch only die 3 - this removes work rather than adding it.
   The residual risk is the two ~20 KiB meta<->die3 activation hops per step
   (latency-bound, host-staged ~50-100 us each; <= ~0.6 ms per cycle, ~1-2%).
2. Embedding round-trips host-side: unchanged in cost class. The draft hidden
   state already crosses to host between target and draft contexts
   (`verify_h`/`pending_h` in common/speculative.cpp); the destination device
   changes from the meta group to die 3. The h handoff crosses PCIe per token
   exactly as before (target d0-2 -> host -> draft device); this feature does
   not add a new PCIe crossing, it relocates one.
3. First-ever `[meta, plain-GPU, CPU]` sched topology: the target today runs
   meta+CPU only. Multi-split draft graphs are new territory for graph reuse /
   CUDA-graph capture on this port. Mitigation for the first boot:
   `LLAMA_GRAPH_REUSE_DISABLE=1` if anything misbehaves; capture behavior is a
   named validation item.
4. LoRA edge: `llama_model::select_buft` (adapter placement) still answers from
   the meta buft list for layer 64. A LoRA on the MTP layer would place adapter
   tensors meta-side (sched copies make it correct, not fast). Not exercised by
   the campaign; documented, not fixed.
5. CLI ordering: the arg-level guard "mtp device must not be a target device"
   only sees devices already parsed, so `--device` must precede
   `--spec-mtp-device` on the command line; server-context.cpp re-checks and
   refuses to boot on overlap regardless of order. The target device list MUST
   be pinned explicitly (`--device CUDA0,CUDA1,CUDA2`) when 4 GPUs are visible,
   otherwise the TENSOR split builds a 4-die meta device (TP4 - forbidden).
6. Fit estimate: the MTP context share of the estimate now lands outside the
   target device table (dropped, not misattributed). Worst case is a slightly
   conservative target fit; `-c 200000` is pinned in the launch line anyway.

## Overlap-win estimate (honest)

Strict dataflow: draft step k+1 consumes the hidden state and sampled token
produced by verify k, so in single-stream decode NOTHING can overlap v1 - the
chain verify -> draft1 -> draft2 -> draft3 -> verify is fully serial, and die 3
idles during verify while dies 0-2 idle during draft, exactly as today. The
v1 win is efficiency, not concurrency:

- Draft steps leave the TP3 dies: the 4-5 row-split matmuls per step stop
  paying the host-staged allreduce chain, and ~40 small kernels per draft step
  stop being triple-issued across three devices (the launch-gap tax the W3/GDN
  census already flagged).
- The LM head keeps its 3x bandwidth (stays on meta), which is why the head was
  NOT moved to die 3: 517.8 MiB at ~90 GB/s/die single-device would be ~5.7 ms
  per step versus ~1.9 ms + allreduce sharded.
- The blk.64 block runs at 1x bandwidth on die 3 (~1.9 ms) instead of 1/3-shard
  (~0.6 ms) + AR + launch tax (~0.5-1.5 ms) - roughly a wash on arithmetic,
  positive if the launch-gap tax is as large as the census suggests.

Estimate for single-stream decode at the served point (8k-10k cell, accept
0.667, 3.00 tok/step): between -2% (hop + single-device block costs dominate)
and +15% (AR + launch tax dominates); expected value around +5-10%. This is a
genuinely uncertain cell - it hinges on the unmeasured launch/AR tax inside
the draft window - and the A/B protocol below decides it.

The real prize this flag unlocks (v2, separate desk): draft-ahead scheduling.
The draft graph produces its own h_nextn, so the draft chain can keep drafting
speculatively while the previous verify runs, rolling back to the accepted
point; and with n_parallel >= 2, one request's draft overlaps another's
verify/prefill. Both require the draft to not contend for dies 0-2, which is
exactly what this change buys. Fallback per plan: if v1 A/B is negative and v2
is not pursued, the flag stays opt-in and die 3 remains the dev cell - zero
cost to the served lane.

## Coordinator validation protocol (served window: dies 0-2 serve, die 3 drafts)

Coordinate with Gemini's lane owner for a window where die 3 is free of bench
work; the served guard battery (dies 0-2) must not be disturbed. No GPU work
was done at the desk; all of the following is for the negotiated window.

Build once (no GPU needed for configure):
    cd /media/chris/ssd128/llamacpp/wt-draft-dev
    cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900 \
      -DCMAKE_HIP_ARCHITECTURES=gfx900 -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build build-hip -j$(nproc) --target llama-server

Boot A (control) = existing /home/chris/launch_tp3_200k.sh unchanged
(binary swapped to the new build to keep the binary identical across arms).

Boot B (arm):
    HIP_VISIBLE_DEVICES=0,1,2,3 <new>/bin/llama-server \
      -m /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf \
      --device CUDA0,CUDA1,CUDA2 \
      -ngl 999 -sm tensor -c 200000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --spec-mtp-device CUDA3 \
      --port 8080 -t 8

(`--spec-mtp-device 3` is equivalent; `--device` MUST come first and MUST be
present whenever 4 GPUs are visible.)

Smoke checks (arm, before any battery):
1. Boot log contains: "MTP draft device: CUDA3", "using extra device CUDA3 for
   this context", "MTP draft context runs on CUDA3".
2. `curl -s localhost:8080/health` -> ok; one greedy completion returns sane
   text; accept ratio reported in the timing line ~0.667.
3. `rocm-smi` during decode: die 3 shows utilization and ~0.5-0.7 GiB used;
   dies 0-2 within their usual envelope.
4. Greedy determinism WITHIN the arm (two identical runs, byte-identical).
   Do NOT expect byte-identity versus control: blk.64 matmuls change reduction
   order (full-tensor fp32 accumulate vs row-shard + allreduce).
5. Guards battery (existing harness): prefill, decode, mtp_canary (gate
   >= 0.63), needle 3/3. LLAMA_GRAPH_REUSE_DISABLE=1 fallback only if (1)-(4)
   misbehave, and note it in the receipt if used.

A/B protocol: alternate boots A,B,A,B,A,B (3 reps each), same cold-stamp rules
as the hardened battery; median decode at the 8k-10k cell, median pp at 2k;
noise classes < 2% not-movement / 2-5% label / >= 5% unambiguous. Promote the
flag into the launch line only on >= +2% decode with all guards green;
otherwise keep it opt-in and record the number - both outcomes bank.

Host-side unit evidence already banked (no GPU):
- docs/amd-port/tests/test_w6_mtp_device_host.cpp - ALL PASS
  (MTP-layer classification against the real tensor names, overlap guards,
  index bounds, extra-device dedup rule).
- All 8 touched TUs (common/arg.cpp, common/common.cpp, src/llama-context.cpp,
  src/llama.cpp, src/llama-model.cpp, src/llama-model-loader.cpp,
  src/models/llama.cpp, tools/server/server-context.cpp) compile warning- and
  error-free with the exact build-hip flag set (gfx900, ROCm 6.2.0, gcc 13).
