# W41 SMALL-KERNEL + COPIES ARMS RECEIPT - W37 Opportunity 2 - 2026-09-25

Desk: small-kernels-and-copies arm, v3 (v1/v2 died to session bounce / usage
limit before producing artifacts; v2 left two CHECKIN notes that this desk
verified from code). Worktree wt-smallk, branch amd/smallk off
amd/v340-port-v2 @ f766a6a04. ZERO-GPU desk: code + build + design only.
Accounting of record: W37_roofline_accounting_2026-09-24.md section 3
Opportunity 2 ("quantize_q8_1 dedup + copy/convert elimination, expected
-2.5..-3.5 ms/cycle -> +2.0-2.9%").

## 0. THE WHICH-3X ANSWER (call-site evidence, pinned from code)

The census kernel symbol `quantize_q8_1` (census_decode_ranked.txt row 6:
358413 calls / 1570.2 ms) is exactly the standalone MMVQ pre-quant launch
`quantize_q8_1<aln>` at quantize.cu:407-409 via
`quantize_row_q8_1_cuda_layout`. It has exactly two host callers on the
served stack:

1. `ggml_cuda_mul_mat_vec_q` (mmvq.cu, launch at :1711 pre-patch) - the
   DIRECT non-split mul_mat dispatch (ggml-cuda.cu:2748). This is the
   SERVED path: under meta-TP4 every weight shard is a plain per-die
   buffer (llama-model.cpp:334 `llama_meta_device_get_split_state`
   sharding; AXIS_1 column shards for q/k/v/qkv, AXIS_0 row shards for
   out/down), so `split == false` at ggml-cuda.cu:2741-2748 and every
   T=4 matmul goes direct.
2. `ggml_cuda_op_mul_mat` (ggml-cuda.cu:2090) - the SPLIT-buft path.
   NOT served under meta-TP4 (no split buffers). For completeness: this
   path does NOT re-quantize per split - it quantizes once on the src1
   device (ggml-cuda.cu:2089-2095, only when `src1_on_device`) and the
   peers `cudaMemcpyPeerAsync` the q8_1 (ggml-cuda.cu:2171). The
   "meta-backend mul_mat splits re-quantize per split" suspect from the
   trace brief is RULED OUT by this code.

Ruled out as contributors to the census symbol:
- `quantize_mmq_q8_1` (mmq.cu, launchers mmq.cu:142/203) - different
  symbol, MMQ path (large T), not decode.
- `quantize_q8_1_to_shared` (fattn-common.cuh:332, used at
  fattn-vec.cuh:181) - in-kernel shared-memory quantization of Q tiles
  inside the FA kernel; different symbol, never a standalone launch.

The three launches are therefore THREE MUL_MAT NODES ON THE SAME src1
NODE, per attention layer per die, in src/models/qwen35.cpp
`build_layer_attn`: `wq(cur)`, `wk(cur)`, `wv(cur)` (qwen35.cpp:269, 281,
284 - three separate `build_lora_mm` calls on the same post-norm tensor).
Each is an independent MMVQ node, each quantizes the same `cur`.
"Draft+target double-quant of the same hidden states" is RULED OUT:
separate graph computes (the cache clears at every
`ggml_backend_cuda_graph_compute`, ggml-cuda.cu:5405), separate tensors.

The qkv triple is NOT the whole story - the complete same-x census of the
target verify pass (T=4, per die, from qwen35.cpp + delta-net-base.cpp):

| layer class (count) | mul_mat nodes on the shared x | launches -> deduped | launches saved |
|---------------------|-------------------------------|---------------------|----------------|
| full-attn (16)      | wq,wk,wv on attn_norm out (:269/281/284) = 3 | 3 -> 1 | 2 |
| full-attn (16)      | ffn_gate,ffn_up on attn_post_norm out (build_ffn LLM_FFN_PAR) = 2 | 2 -> 1 | 1 |
| GDN (48)            | wqkv,wqkv_gate,ssm_beta,ssm_alpha on layer input (:236/240/361/368, `build_qkvz(cur)` + beta/alpha both on `cur`) = 4 | 4 -> 1 | 3 |
| GDN (48)            | ssm_out on GDN output (:463) = 1 | 1 | 0 |
| head (1)            | output on result_norm (:222) | 1 | 0 |
| **total**           | **353 quantize launches** | **161 distinct x** | **192 (54%)** |

(wo, ffn_down, head quantize their own unique x - no redundancy there.
The draft/nextn block has the same wq/wk/wv triple + gate/up pair per
draft step; the catch-up pass repeats the full target-pass structure.)

CORRECTION TO W37: the "re-quantized ~3x/layer/die" shorthand (W0 T4)
holds only for the attention layer (3x on one x). Launch-weighted across
the pass the redundancy is 353 -> 161 = 2.2x, not 3x - because most
matmuls (down/out/head/micro) own a unique x. W37's quantize component
(2.4 -> 0.8 ms) is therefore an overestimate; honest band below.

The 2x already exists but is INERT: `GGML_CUDA_Q81_ACT_CACHE`
(common.cuh:1401, default off) caches the q8_1 BUFFER per (producing
node, data ptr, stream, device, shape, strides, layout) key within one
graph compute, but both served call sites re-launch the quantize kernel
UNCONDITIONALLY on a hit - the cache deduped the allocation, never the
launch:
- mmvq.cu:1694-1711 (pre-patch): `find`/`insert` at :1696-1702, then an
  unconditional `quantize_row_q8_1_cuda_layout` at :1711. (v2's
  CHECKIN note, verified.)
- ggml-cuda.cu:4813-4829 (group path, T=1 draft): same pattern,
  unconditional `quantize_row_q8_1_cuda` at :4826.
- ggml-cuda.cu:2076-2096 (split path): correct (launch inside the
  miss branch) - but not served anyway.

## 1. ARM 1: Q8_1 RE-QUANT DEDUP (implemented in this worktree)

Design (minimal diff over the existing infrastructure - no new
subsystem): the launch is skipped when the cache HITs. A hit guarantees
bit-identical bytes:
- the key carries the producing node pointer + device pointer + stream +
  device + ne/strides (common.cuh:1402-1416), so a pool address recycled
  for a different tensor cannot hit;
- entries are cleared at every `begin_compute` (ggml-cuda.cu:5405), so
  they never span computes and never outlive the compute buffer;
- within one compute the src1 node's bytes are fixed once produced (DAG;
  matmul src1 is read-only), and `quantize_q8_1` is deterministic per
  block: same input bytes + same grid -> identical output bytes. Skipping
  the relaunch cannot change any bit the matmul reads.
- stream ordering: all direct-dispatch nodes of a device run on the same
  ctx stream, so the producer's quantize is ordered before every hit
  consumer's matmul.
- the cache is force-disabled during CUDA graph capture/replay
  (ggml-cuda.cu:5403-5405); on gfx900 graphs are disabled anyway
  (cc < AMPERE, ggml-cuda.cu:5310-5315), so direct mode always serves.

Diff (all in this worktree, uncommitted at desk open):
1. mmvq.cu (solo direct path, the served T=4 path): `src1_q8_1_fresh`
   flag; the :1711 launch now runs only on miss/alloc.
2. mmvq.cu: cache key now passes `layout = mmvq_aln ? 1 : 0`. Before,
   aln and legacy producers shared layout=0 keys; with launch-skip live,
   a legacy-only-sized entry hit by an aln consumer would read an
   unallocated aln region. ne11 (in the key) plus the env decide
   mmvq_aln, so under a single boot's env the layout is a function of
   the key; the explicit field makes cross-layout aliasing impossible.
   The aln dual-region producer writes legacy at +0 and aln behind it,
   so any same-layout hit serves both regions (vecdotq.cuh:1702
   `block_q8_1_aln`, 48 B).
3. ggml-cuda.cu group path (T=1 draft): same fresh-flag hit-skip at the
   :4826 launch.
4. ggml-cuda.cu:5329 engagement line INFO -> WARN (E-117: served log
   drops INFO; W27 precedent).

Env gate: `GGML_CUDA_Q81_ACT_CACHE=1` (pre-existing, default OFF). The
mission's "GGML_CUDA_Q81_REUSE=1" name maps to this existing gate;
reused per AGENTS.md instead of adding a second flag for one lever.

Expected arithmetic (per cycle, per W37 row 4 + this census):
- quantize_q8_1 = 2.4 ms/cycle; launch-weighted dedup deletes 54% of
  verify-pass launches, and every deleted launch quantizes the same
  [5120 x 4] x as its group (uniform per-call cost), so the verify-pass
  term scales ~2.2x; draft triple/gate-up and catch-up add a little.
  Band: 2.4 -> ~1.0-1.3 ms = **-1.1 to -1.4 ms/cycle raw**; with the
  w23env-class partial-transfer haircut: **-0.8 to -1.4 ms/cycle ->
  +0.6-1.1% t/s** (cycle 123.1 -> 121.7-122.3).
- W37's Opp2 total (-2.5..-3.5) stacked this as -1.6; see section 2 for
  why the copies half shrinks under code audit.

Falsifiable gate (W37 gate-2 discipline, restated for this arm): in a
`-lv 4` + `LLAMA_LAUNCH_TIMELINE=1` boot with the gate on,
quantize_q8_1 launches/cycle must drop >= 1.8x vs gate-off AND served
paired delta must be >= +0.3% t/s (W33 noise floor). If launches drop
but t/s does not, the class is launch-latency-free dead: close.

Oracle (bit-exact, law: oracle-before-timing):
- Bit-exactness argument as above (deterministic producer, read-only
  consumers, per-compute lifecycle, graph-replay excluded).
- Command (one paired window, two boots, same binary, same session):
      # boot A: BASEENV (W40 U2 preflight stack), canonical config
      #   -c 200000 -b 512 -ub 512 -ctk q4_0 -ctv q4_0 -fa on
      #   guard battery -> record text_sha256 (determinism guard)
      python3 docs/amd-port/tests/guard_battery.py --determinism-only \
          --server-binary <wt>/build-hip/bin/llama-server \
          --launch-config "BASEENV q81=off" --output-jsonl <receipt>.jsonl
      # boot B: BASEENV + GGML_CUDA_Q81_ACT_CACHE=1
      grep -c "GGML_CUDA_Q81_ACT_CACHE=1" server.log   # > 0 (WARN line)
      python3 docs/amd-port/tests/guard_battery.py --determinism-only \
          --server-binary <wt>/build-hip/bin/llama-server \
          --launch-config "BASEENV q81=on" --output-jsonl <receipt>.jsonl
  PASS = boot B text_sha256 byte-identical to boot A (and to the
  in-boot repeat). Then the timed paired delta adjudicates the gate.
  Counter check in any verbose log: "q81 hits / misses over computes"
  line (common.cuh:1467) must show hits > 0 on the decode workload.

## 2. ARM 2: TP4 COPY/CONVERT NODES - DELETABLE VS LOAD-BEARING

The class of record (W37 row 4 / roundmap lever 5): cpy_scalar 1.1 +
copyBufferRect 1.1 + get/set_rows 1.4 + concat 0.7 = 4.3 ms/cycle, with
"2-3 ms recoverable by deleting redundant convert/copy nodes". Code
census of where these launches come from under meta-TP4 (every die
evaluates the full logical graph on its shards; cross-die movement is
RCCL allreduce at subgraph boundaries, ggml-backend-meta.cpp:2345-2388;
the butterfly fallback with staging copies (ggml-backend-meta.cpp:2257-
2339 `set_tmp_data` + `ggml_backend_tensor_copy_async`) only runs when
`comm_allreduce` fails - census shows 137.2 NCCL launches/die/round, so
the fallback is cold in the served config and its copies are NOT the
census class):

| census kernel (ms/cycle) | source node(s) (file:line) | per cycle per die | verdict |
|---|---|---|---|
| k_get_rows_float (get_rows half of 1.4) | embd gather qwen35.cpp:525; inp_out_ids gathers :178/179/215/635; GDN state loads via build_rs get_state_rows + states_extra (llama-graph.cpp:3240/3244) | 1 + 1 + 48x2 | LOAD-BEARING (semantic gathers + recurrent state reads) |
| k_set_rows_quant q4_0 (set_rows half of 1.4) | KV cache writes, 16 attn layers | 16 | LOAD-BEARING (KV write path) |
| concat_cont (0.7) | GDN conv-state concat conv_states||transpose(qkv_mixed) delta-net-base.cpp:470 (48x/pass); MTP e_norm||h_norm qwen35.cpp:550 (draft, 3x/cycle) | ~50-200 | LOAD-BEARING (semantic operand assembly; not deletable without kernel-level dual-pointer conv) |
| cpy_scalar f32->f32 (1.1) | GDN conv-state store ggml_cpy(conv_state_last -> conv_states_all view) delta-net-base.cpp:509 (+ K-slot loop :539 when n_rs_seq>0); GDN ssm snapshot store ggml_cpy(src view -> ssm_states_all view) delta-net-base.cpp:597; rs extra-state cpy llama-graph.cpp:3246; token_shift store llama-graph.cpp:3330 | ~2-4 per GDN layer + head glue | LOAD-BEARING (these write the recurrent/snapshot state). NOT no-op: n_rs > n_seqs in the spec cycle (rollback slots), so the extra-state cpy runs real rows |
| copyBufferRect (1.1) | the strided snapshot/conv stores above execute as 3D copies (non-contiguous dst views with cache-size row strides); plus zero-sized-guarded views that launch nothing | - | LOAD-BEARING, same nodes as cpy_scalar row |

Findings:
- There are NO redundant convert nodes on the served qwen35 path: no
  f32->f16 cpy nodes exist (KV conversion rides inside set_rows quant;
  FA reads q4_0 KV directly post-fa40). No ggml_dup nodes are built.
  No "redundant post-split cpy" exists: meta-TP4 boundaries allreduce
  the partial node in place; there is no sched-style staging copy.
- The zero-sized case is already free: when n_rs == n_seqs, s_copy_extra
  is a 0-row view (llama-graph.cpp:3267) and the extra get_rows/cpy are
  zero-sized (no launches).
- AR-fallback staging copies (the only true deletion class) are cold
  with healthy RCCL. If a `-lv 4` boot shows `meta ar fb > 0`, killing
  those fallbacks changes fp32 summation order vs the ADD-fold butterfly
  -> NOT bit-exact-deletable; would need its own oracle baseline.

So the honest Opp2 copies component is NOT -2.0..-3.0 by node deletion.
What remains, in two classes:
- ARM 2a (this desk, graph-level): nothing deletable survives the
  bit-exactness bar. Verdict: DEAD as a pure node-deletion lever; the
  measurement boot (below) is still the law before closing - if the
  timeline shows >= 1 ms/cycle in cpy/convert launches NOT in the table
  above, that residue is the deletion target.
- ARM 2b (kernel-class, NOT this window): fuse the two GDN state stores
  into their producers (gated_delta_net writes snapshots directly into
  ssm_states_all; conv kernel writes conv state directly), deleting
  ~2-4 launches x 48 layers of cpy_scalar/copyBufferRect. Same bytes,
  same values -> bit-exact achievable; class ceiling ~1.0-1.5 ms/cycle
  of the 2.2 ms cpy_scalar+copyBufferRect. M-class kernel work under the
  W17 oracle discipline; banked as the follow-on, not priced in Opp2.

Revised Opp2 arithmetic: arm 1 -0.8..-1.4, arm 2a 0 (pending timeline
residue check), arm 2b banked -1.0..-1.5 (kernel class). Opp2 total
**-0.8 to -1.4 ms/cycle this window (+0.6-1.1% t/s)**, vs W37's
-2.5..-3.5 - the gap is the copies half, which this audit re-classifies.

Arm 2 oracle command (if any residue deletion is ever attempted):
same paired-boot guard battery as arm 1, boot B differing only in the
arm's gate; PASS = byte-identical text_sha256 (node deletion must be
bit-exact, no DUST class).

## 3. PAIRED-WINDOW SPEC (one window, both arms adjudicated)

Preconditions: GPU lock free (paired window), cool-die < 60 C, quiet
machine check; binary from this worktree's build-hip (section 4),
binary sha + CMakeCache sha stamped into the receipt.

1. Boot A (baseline): W40 U2 preflight BASEENV stack unchanged, canonical
   config `-c 200000 -b 512 -ub 512 -ctk q4_0 -ctv q4_0 -fa on` TP4.
   Run guard battery (determinism + decode guards); record text_sha256,
   t/s battery.
2. Boot B (arm 1): BASEENV + `GGML_CUDA_Q81_ACT_CACHE=1`. Verify
   engagement: `grep -c "GGML_CUDA_Q81_ACT_CACHE=1" server.log > 0`.
   Same battery; PASS requires text_sha256 identical to boot A.
3. Adjudication (arm 1 gate): paired decode t/s delta vs boot A;
   predicted +0.6-1.1%; kill if < +0.3% (noise floor W33) or any
   determinism mismatch.
4. Measurement boot (class of record, W37 gate 2): BASEENV +
   `GGML_CUDA_Q81_ACT_CACHE=1` + `-lv 4` + `LLAMA_LAUNCH_TIMELINE=1`,
   short decode run; from the timeline: per-cycle counts and us for
   quantize_q8_1 (expect >= 1.8x fewer launches), cpy_scalar,
   copyBufferRect, concat, get/set_rows. If the copy class totals
   < 3.5 ms/cycle, W37 Opp2-gate falsifies the REMAINING copies lever
   (arm 2a closes; 2b stands on its own kernel-class case).
5. Teardown by PID only; lock release with evidence.

## 4. BUILD (compile proof, ZERO-GPU)

Configured + built in this worktree with the main build-hip flags
(read from /media/chris/ssd128/llamacpp/llama.cpp/build-hip/CMakeCache.txt):
cmake /home/chris/opt/cmake, GGML_HIP=ON, Release, GGML_NATIVE=ON,
CMAKE_HIP_ARCHITECTURES=gfx900, GGML_HIP_RCCL=ON, LLAMA_CURL=OFF,
ROCm 6.2.0 clang. Result: BUILD-EXIT:0, full build then an incremental
pass after the last source edit; mmvq.cu.o and ggml-cuda.cu.o both
newer than their sources (05:17:48 / 05:20:00 vs 05:08:12 / 05:17:32),
llama-server relinked - the compile proof covers the final arm-1 state.

## 5. DESK LOG

- (v1/v2) session-bounced; v2 left CHECKIN notes: arm1 mapped (cache
  exists, mmvq.cu:1711 re-quant on hit; 3x = q/k/v sharing cur) - both
  verified from code this session.
- 09:58Z desk open, worktree clean @ f766a6a04, CHECKIN law started.
- which-3x pinned: qkv triple + GDN quad + gate/up pair; 353 -> 161
  launches (2.2x), split/draft suspects ruled out from code.
- arm 1 implemented: hit-skip (mmvq.cu solo + group path), layout in
  key, engagement line to WARN.
- arm 2 classified: served copies = GDN state glue + KV/embd gathers;
  nothing deletable at the node level; AR-fallback copies cold; 2b
  fusion banked.
- build-hip BUILD-EXIT:0 (full + incremental pass on the final sources;
  objects newer than sources, llama-server relinked 05:20).
- receipt written; commit amd/smallk + push_backups snapshot.
