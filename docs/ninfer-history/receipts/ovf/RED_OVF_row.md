# RED_OVF_row — G-OVF-0 RED capture (kvarn >10.3k std::bad_alloc)

- Date: 2026-09-19 (boots 19:34-20:4x local). Desk: ovf (Team Red agent #2).
- Bin: /home/chris/artifacts_bin/ninfer-serve_f3312f25c8201da0.bin (gates-PLOG-074 GREEN).
  AS-BUILT source of record: branch amd/wo-kvarnport tip 4178169ef (bin embeds
  /home/chris/worktrees/amd-wo-kvarnport/src/ops/launcher/gqa_attention_kvarn.cu; confirmed
  via the cuda_check filename string at .rodata 0x247f92 in the bin).
- Boot flags: canonical env (serve_10k.sh exports; NINFER_WORKSPACE_MIB=96) + --port 8100
  --devices 0,1,2,3 --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256
  --greedy --allow-nvfp4-weights --max-context 36352 --kv-capacity 36352
  --kv-dtype kvarn_k4v4 --spec mtp --draft-tokens 2 --default-max-tokens 16.
- Replay body: gates banked needle20k_req.json (19,709 real tokens). 4 fires, 4/4 throws:
  red_resp/red2/red4/red6, wall 116/127/127/108 s. Response body every time:
  {"error":{"message":"std::bad_alloc","type":"internal_error"}}; all 4 ranks log
  [tp2 worker error rank N] std::bad_alloc. RSS flat (host) — consistent: the refusal is a
  DEVICE workspace arena, not host memory.

## Decisive capture (RED #6, results/amd/ovf/red_gdb5.log, gdb `catch throw bad_alloc`)

Backtrace: __cxa_throw <- DeviceArena::alloc_bytes <- DeviceArena::alloc <-
ops::detail::gqa_attention_kvarn_cached_launch (ret 0xa0e0a44) <- gqa_attention_cached <-
TextContext::kvarn_attend_text <- attn_mix_tp <- run_layers <- prefill_impl <- prefill_chunk
<- run_tp2_request lambda (rank thread).

- Throw site in arena: src/core/arena.cu:299 `if (end > cap_) { throw std::bad_alloc(); }`
  (mapped by disassembly of alloc_bytes 0x9db4880: r12=aligned off, r15=end,
  cmp cap_[this+8], r15 / ja -> throw).
- Failing call (init-list read from frame-3 stack at sp+0xe0): shape {256, 64, 1, 160} =
  kKvarnAttnD, kKvarnAttnG, kv_heads(TP4 per-rank=1), n_tiles=160, DType 6 = FP16 ->
  the HIP-port temp `v_temp_h` — amd/wo-kvarnport src/ops/launcher/gqa_attention_kvarn.cu
  lines 396-397 (added by WO_KVARN_PORT for the gfx900 V-plane fp16-bits contract).
- n_tiles = 160 pages x 64 tok/page = 10,240 tokens = EXACTLY the banked death band
  (9984-10112 progress marks, position-locked on 19.7k and 49.3k prompts).
- Arena state at throw: cap_ = 100,663,296 (= 96 MiB, NINFER_WORKSPACE_MIB override),
  off_ = 95,421,568, peak_ = 100,651,264 (12 KB under cap — the arena rides the ceiling),
  end = 100,665,088. Requested bytes = end - off = 5,242,880 = 256*64*1*160*2 EXACTLY.

## Verdict

- WO prime hypothesis (int32 overflow producing a huge/negative size) FALSIFIED: the
  requested size is honest (5 MiB). The defect is 96 MiB work-arena EXHAUSTION:
  chunk-entry offset is already ~81 MiB and ratchets ~1.2 MB per prefill chunk until the
  materialize route's own temps (k_temp 5 MiB + v_temp 5 MiB + v_temp_h 5 MiB + bt) hit
  the cap at chunk 79. CUDA-line pre-port route had only TWO temps; the HIP port added the
  third full-size temp (+5 MiB/call), moving the wall to the observed position on this bin.
- Fix direction (source hunt, GPU-free): (1) PRIMARY — find the ~1.2 MB/chunk un-freed
  residue in the prefill pass (root cause; removes the wall for every route); (2) make the
  v bf16->fp16 conversion IN-PLACE (same 2-byte width, elementwise, race-free) to restore
  CUDA-line temp parity. (2) alone only moves the wall ~2 chunks — (1) is required.

G-OVF-0: MET (RED reproduced 4/4 with allocation site named). Window released to
coordinator after this capture; canonical restore owned by coordinator (warmup-fault
class 414405c32).
