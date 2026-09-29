# W6b - full draft isolation: the MTP draft context allocates nothing on the TP3 group

Date: 2026-09-21. Desk: wt-draft-isolation, branch amd/draft-isolation (base b4ad90c5a,
the tree that carries the merged --spec-mtp-device). ZERO-GPU phase COMPLETE:
design + implementation + full gfx900 build + host logic tests. NOT RUN - the
served-validation desk owns boots on dies 0-2; die-3 windows are negotiated
(protocol at the end). Direct predecessor: results/W6_draft_device_2026-09-21.md.

## Problem being closed (from TP2_feasibility_2026-09-21.md / E-032)

The merged flag relocates only 251.1 MiB/die of the measured 1072.8 MiB/die MTP
cost at TP2/10k, because the draft context's backend list [meta(TP3), die3, CPU]
lets the scheduler keep the bulk of the draft compute buffers on the meta group
(821.7 MiB/die still on the serving dies). Also, the GGUF carries no
draft-exclusive LM head/embeddings, so the draft graph reads the shared
model.output and model.tok_embd, which live as split tensors on the TP group.

Exact sizes of record (GGUF offset deltas of Qwen3.8-27B-ASCII-P1M.gguf, never
recomputed; 866 tensors; each delta equals the raw size, no alignment padding):

    output.weight      Q6_K    542,942,400 B = 517.79 MiB (542.94 MB)
    token_embd.weight  IQ4_XS  351,619,840 B = 335.33 MiB (351.62 MB)
    combined duplication              894,562,240 B = 853.12 MiB (894.56 MB)

    blk.64.* (15 tensors, draft-exclusive, already pinned by W6) 169.3 MiB
    vocab = 129,272 -> one logits row (f32) = 517,088 B = 517.1 KB

## What ships (4 changes, all gated on the flag being set)

1. Backend list LEADS with the dedicated device (src/llama-context.cpp ctor):
   `params.extra_device` is inserted at index 0 instead of appended, so the
   draft context's list is [die3, meta(TP3), CPU]. The scheduler resolves a
   pre-allocated weight to the first backend supporting its buffer type
   (ggml-backend.cpp sched_backend_from_buffer), so compute follows weights.
2. Whole-tensor duplication at load time (src/llama-model-loader.cpp
   create_tensor): when dev_mtp is set, TOKEN_EMBD and OUTPUT tensors are
   additionally created in a die3-buffer-type context (dedup by name, so tied
   embeddings produce a single shared copy). The loader's normal load path
   fills them from the GGUF alongside the originals - no post-load device
   copies, no host staging. The original meta tensors are untouched: same
   pointers, same buffers, target verify pass unmodified. Dup bytes are added
   to the progress accounting (size_data), never to n_created
   (done_getting_tensors guard). The copies are kept OUT of tensors_by_name
   (get_tensor() answers meta split-state queries by name and must resolve the
   originals - host-tested).
3. Draft graph redirect (src/llama-model.cpp): model.tok_embd_mtp /
   model.output_mtp members are resolved from the MTP context after tensor
   creation; the graph_mtp builders of qwen35, qwen35moe, step35 and
   cohere2moe prefer them (with fallback to the originals when the flag is
   unset or the GGUF carries draft-exclusive weights). With all weights
   die3-resident, the draft graph becomes a single die3 split: zero
   draft-context allocations on the meta group.
4. Isolation audit (src/llama-context.cpp sched_reserve): after reserve, the
   context logs "sched_reserve: CUDA3 isolation audit: <A> MiB on CUDA3, <B>
   MiB on the model split" (CPU bucket excluded; the host-pinned output buffer
   and the KV cache - already pinned via dev_layer - sit outside this audit).
   B > 0 warns (isolation incomplete, e.g. an output_norm fallback); env
   LLAMA_SPEC_MTP_STRICT=1 turns the warning into a boot failure for gate
   runs.

Unset (default) = byte-for-byte today's behavior: no dup, no reorder, no
audit, graph builders unchanged (both *_mtp members stay null).

## Per-die VRAM relocation prediction at TP3/200k (ub512, q4_0 KV, FA)

Absolute draft allocations (die-count independent, from the TP2 measurements
at 10k b512/ub512, byte-reproducible across reps):

    blk.64 weights (die3 since W6)                       169.3 MiB
    draft KV 1 layer, 1024+1024, q4_0, 200192 cells      220.1 MiB  (110.0 x K+V)
    draft context compute (TG+PP, meta group today)   ~1643.4 MiB   [a]
    shared output+embd duplicates (NEW, die3)            853.1 MiB

    [a] TP2 flag arm: 821.7 MiB/die x 2 serving dies on the meta group.

Predictions at TP3 (meta group spreads over 3 dies):

- Serving dies, current flag arm: ~547.8 MiB/die of draft compute on the meta
  group (1643.4 / 3; TP2 showed near-equal thirds, 7348.5/7347.9). Full
  isolation frees ~548 MiB/die (expect 500-650 given the observed ~228 MiB
  unevenness of the TP3 control boot 7809.8/7582.0/7634.2).
- Against the served desk's current numbers 8419/8131/8130 used of 8573:
  prediction ~7870/7583/7582 used -> free goes 154/442/443 to ~703/990/991 MiB
  per die. Note the die-0 excess (288 MiB over dies 1-2) cannot be draft
  allocation - the draft meta share is near-equal thirds; it predates the draft.
- Against the no-flag in-split arm the same change frees ~785 MiB/die
  (784.7 = (169.3 + 220.1 + 1643.4)/3 - weights/KV shards also leave meta).
- The task's "~2.4 GiB die-3 residency" guess is optimistic; the accounting
  below lands higher.

## Die-3 residency prediction at 200k

Anchor: TP2 flag arm, die2 at 10k post-probe = 1437.1 MiB (boot-ready 558.0 =
169.3 weights + 11.5 KV + 377.2 own compute share; post-probe the buffers grew
to 1256.3 total compute share on the draft die).

    1437.1  today (10k post-probe)
  + 208.6   draft KV 10k -> 200k (11.5 -> 220.1)
  + 853.1   output.weight + token_embd.weight duplicates
  + 387.1   meta-side compute residue that relocates (1643.4 - 1256.3)
  = 2885.9 MiB ~ 2.82 GiB central prediction (range 2.6-3.4 GiB: a single
    die3 split buffer can come out smaller than the sum of today's meta+die3
    split buffers - no copy inputs - or larger after probe growth)

Die 3 total is 8176 MiB (E-031 idle record 18.0/8158.0): ~5.3-5.6 GiB remain
free at the served point - the dev/bench cell stays usable between windows,
with less slack than the W6 flag arm (~6.7 GiB free).

## Risks (honest list)

1. The 517,088-byte logits handoff (129,272 x f32) per draft step is now
   computed on die3 and crosses die3 -> host instead of meta -> host. Same
   size, different bus. Compensation: the two ~20 KiB meta<->die3 activation
   hops the W6 arm inserted per step (token row meta->die3, head input
   die3->meta) are GONE, and the h handoff (n_out x 5120 x 4 = ~20 KiB/row) is
   unchanged in class. Net per-step PCIe should improve slightly.
2. Meta-group graph-reuse interaction: the draft graph collapses from a
   2-3-split [meta, die3] topology to a single die3 split. Simpler than W6,
   but it is still a new topology for graph reuse / capture on this port -
   LLAMA_GRAPH_REUSE_DISABLE=1 remains the named fallback for the first boot.
3. Load-time cost: the two duplicated tensors add 853 MB of file reads
   (page-cache warm: seconds, once per boot).
4. GGUFs without blk.64.nextn.shared_head_norm would keep model.output_norm on
   meta: the head-norm op splits to meta, the audit warns (STRICT fails the
   boot). Qwen3.8-27B-ASCII-P1M carries shared_head_norm (20480 B, verified) -
   not hit by the campaign model.
5. LoRA on the head still matches by tensor name (get_weight is name-keyed), so
   adapter tensors apply through sched copies - correct, not fast; same class
   as the W6-documented LoRA edge, not exercised by the campaign.
6. Fit estimate (--fit / common_get_device_memory_data): the dup buffer and all
   draft context bytes are attributed to a buft that is not among the model's
   devices and are dropped from the target fit table - same "dropped, not
   misattributed" behavior W6 documented. -c 200000 stays pinned in the launch
   line.
7. Wasted 853 MiB if a future GGUF carries draft-exclusive
   shared_head_head/embed_tokens: the dup would be redundant (die3 has room,
   but the loader cannot know the graph's fallback choice). Documented, not
   fixed.

## Host evidence already banked (no GPU)

- docs/amd-port/tests/test_w6_isolation_host.cpp - ALL PASS: dup rule
  (TOKEN_EMBD/OUTPUT only, flag-gated), name dedup (tied case), loader
  accounting (size_data vs n_created), exact GGUF byte sizes of record,
  tensors_by_name exclusion, backend-list leading + dup-ignore + CPU-only
  edge, scheduler placement matrix ([die3, meta, CPU]: weights -> die3, inputs
  via CPU, meta weight -> meta fallback), audit bucketing + STRICT trip.
- docs/amd-port/tests/test_w6_mtp_device_host.cpp (predecessor suite) - ALL
  PASS, unchanged.
- Full build from clean: cmake -B build-hip -DGGML_HIP=ON
  -DAMDGPU_TARGETS=gfx900 -DCMAKE_HIP_ARCHITECTURES=gfx900
  -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 -DLLAMA_CURL=OFF
  -DCMAKE_BUILD_TYPE=Release ; --build build-hip -j20 --target llama-server
  -> 100%, exit 0, ZERO warnings (all 10 touched TUs compiled:
  llama-context, llama-model, llama-model-loader, models/qwen35,
  models/qwen35moe, models/step35, models/cohere2moe, common/arg,
  server-context, headers). --help renders the new flag text.

## Coordinator validation protocol (served window: dies 0-2 serve, die 3 drafts)

Request a die-3 window from the Gemini lane owner (agent-comm
b4791a54-9948-4610-9ee7-ab96a077c0cb) such that dies 0-2 may serve undisturbed;
the served guard battery must not be disturbed. No GPU work was done at this
desk; all of the following is for the negotiated window.

Binary: rebuild in this worktree (command above) or use
wt-draft-isolation/build-hip/bin/llama-server (built 2026-09-21, exit 0).

Boot A (control) = /home/chris/launch_tp3_200k.sh unchanged (swap the binary to
this build so arms differ only by flags).

Boot B (arm):
    HIP_VISIBLE_DEVICES=0,1,2,3 <build>/bin/llama-server \
      -m /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf \
      --device CUDA0,CUDA1,CUDA2 \
      -ngl 999 -sm tensor -c 200000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --spec-mtp-device CUDA3 \
      --port 8080 -t 8

(--device MUST come first and MUST be present whenever 4 GPUs are visible.)

Smoke checks (arm, before any battery):
1. Boot log contains, in this order:
   "MTP device CUDA3: duplicated token_embd.weight (335.3 MiB) and
    output.weight (517.8 MiB) for the draft context"   [loader]
   "using extra device CUDA3 for this context (leading the backend list)"
   "CUDA3 isolation audit: <A> MiB on CUDA3, 0.00 MiB on the model split"
   "MTP draft context runs on CUDA3 (fully isolated: ...)"
   The load-bearing number is the second audit figure: it MUST be 0.00 MiB.
   Optional hard gate: LLAMA_SPEC_MTP_STRICT=1 on the arm boot.
2. Control boot A shows NONE of these lines (flag-unset semantics unchanged).
3. health ok; one greedy completion returns sane text; accept ~0.667.
4. rocm-smi during decode: die 3 ~2.6-3.4 GiB used (prediction 2.82); dies 0-2
   ~548 MiB/die below the flag-arm envelope. Run the vram_sampler.py CSV
   (boot-ready + post-probe) and bank the actual numbers against this receipt's
   predictions - this desk's numbers are analytic and expected to move.
5. Greedy determinism WITHIN the arm (two runs byte-identical). Do NOT expect
   byte-identity vs control: blk.64 and head matmuls change reduction order.
6. Guards battery: prefill, decode, mtp_canary (>= 0.63), needle 3/3.
   LLAMA_GRAPH_REUSE_DISABLE=1 only if (1)-(5) misbehave; note it if used.

A/B: alternate boots A,B,A,B,A,B (3 reps each), same cold-stamp rules as the
hardened battery; median decode at the 8k-10k cell, median pp at 2k. Promote
into the launch line only on >= +2% decode with all guards green; otherwise
keep the flag opt-in and record the number - both outcomes bank.
