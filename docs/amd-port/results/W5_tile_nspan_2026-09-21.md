# W5 tile-nspan desk - N-span full-K dequant window, bit-identical, memory-fit AND parity

Date: 2026-09-21. Worktree wt-tile-nspan, branch amd/tile-nspan. GPU: die 3 only
(HIP_VISIBLE_DEVICES=3; dies 0-2 untouched). Scope: E-049's named next link -
chunk the a8 tile's f16 weight staging along N instead of K: each launch handles a
contiguous N-span of weight rows with the FULL K dimension, keeping the unchunked
launch's gridDim.z = ks fill that the chunked arm forfeited (-35% wall at M=128).
Targets: memory-fit (window + peak under 200 MiB, measured) AND wall parity vs
unchunked (>= 4.28 TF/s call-weighted at M=128).

## 1. What landed (on top of the merged tile-gemm.cu)

- tile-gemm.cu: GGML_CUDA_TILE_FP16_NSPAN=1 (on top of GGML_CUDA_TILE_FP16=1)
  routes to tile_fp16_nspan_run, tried before the chunked arm. Per call, weight
  rows are processed in contiguous N-spans: span size = the largest NT(64)-aligned
  row count whose full-K f16 slab fits GGML_CUDA_TILE_FP16_NSPAN_MIB (default 128;
  window f16 + partials per stream), a single span when the whole slice fits.
  Multiple spans are evened (NT-aligned, last takes the remainder). Per span:
  (a) the vetted per-type to_fp16 kernel dequantizes the span's quant rows
  STRAIGHT from src0 - quant rows are contiguous slabs, the kernels are linear
  over quant blocks, so the window bits are the per-call dequant bits and NO
  gather exists (the chunked arm's gather kernel is unused here);
  (b) tile_fp16_gemm UNCHANGED - the span launch passes the window (row stride K),
  N param = span rows (span-local partials), and C base = dst + span column
  offset, with grid (span/64, M/128, ks) - the unchunked z partition intact;
  (c) new tile_fp16_reduce_span: the same fixed-order f32 slice sum as
  tile_fp16_reduce, mapping float4 lanes into the strided dst (row stride ldc,
  column offset). Elements are disjoint across spans; every element sees the same
  slice schedule and f32 sum order as the unchunked arm -> bit-identical.
- Window: one f16 slab (span x K) + one span partials slab (ks x M x span) per
  stream, raw cudaMalloc (never the per-call pool), grow-only, freed at context
  teardown (ggml_cuda_tile_fp16_nspan in common.cuh + clear in the ctx dtor).
  Budget = NSPAN_MIB total per stream AND free VRAM minus a 64 MiB margin at
  decision time; refusal falls back to chunked/shipped with one WARN - never
  aborts. ks > 1 additionally requires ldc % 4 == 0 (float4 strided reduce).
- BUG OF RECORD fixed (2 lines, also latent in the merged chunked arm): the
  in-place window growth updated the buffers but NOT the cached sizes
  (e->f16_bytes/e->p_bytes, chunked e->q_bytes), so the next call re-entered the
  realloc path; when that call was the CUDA-graph capture pass, the realloc's
  hipStreamSynchronize aborted ("operation not permitted when stream is
  capturing"). Reproduced on gdn_qkv (first shape needing a bigger P slab than
  ffn gate's 11.8 MiB); crash log results/W5_tile_nspan_capture_realloc_bug_
  2026-09-21.txt. Fixed in both arms; oracle re-gated after; the exact scenario
  (gdn_qkv window realloc then capture) passes in the clean battery.

## 2. Oracle (die 3, docs/amd-port/tests/test_tile_integ_oracle.cu extended)

Per census shape: unchunked vs host-mirror spec (unchanged gate), chunked
bit gate (unchanged), NEW nspan bit gate at two schedules - single full-N span
and forced multi-span (3*NT rows per span). 10/10 PASS: mismatch 0.000%, max rel
0.00e+00 (unchunked still bit-exact vs spec), L2-vs-f64 2.9e-3..6.1e-3 (banked
band), chunked bit-diff 0/N, nspan bit-diff 0/N on every shape. Re-run clean
after the window fix (results/W5_tile_nspan_oracle_2026-09-21.txt).

## 3. Full-glue byte identity (real IQ3_S weights, bench --dump)

bench_tile_nspan.cpp, both arms through the real ggml mul_mat dispatch, f32 dst
dumped per shape: cmp 10/10 byte-identical (nspan vs unchunk) at M=128 AND at
M=512. Env unset: this worktree's library vs the pre-change served library
(../llama.cpp/build-hip): 10/10 byte-identical. Budget-refusal fallback exercised
live (dial arms below): one WARN + shipped per-call tile route, no abort.

## 4. Perf, die 3, census shapes (3 reps x 20 iters, median)

Wall op-level (dequant + convert + GEMM in both arms), M=128:

```
shape        K     N_d  ks | unchunk ms  TF/s | nspan ms  TF/s
ffn_gate   5120  5760   4 |  1.7096   4.42 |  1.7042   4.43
ffn_up     5120  5760   4 |  1.7157   4.40 |  1.7261   4.37
ffn_down  17408  1664   8 |  1.6457   4.51 |  1.6450   4.51
gdn_qkv    5120  3328   8 |  0.9997   4.36 |  1.0016   4.36
ssm_out    6144  1664   8 |  0.6673   3.92 |  0.6676   3.92
gdn_gate   5120  2048   8 |  0.6420   4.18 |  0.6596   4.07
attn_q+ga  5120  4096   4 |  1.3658   3.93 |  1.3299   4.04
attn_out   6144  1664   8 |  0.6586   3.97 |  0.6690   3.91
attn_k     5120   256   8 |  0.1582   2.12 |  0.1585   2.12
attn_v     5120   256   8 |  0.1579   2.13 |  0.1586   2.12
call-weighted die agg: unchunk 4.32 (reproduces the 4.28-4.33 reference)
                       nspan 4.31 = parity (-0.2%); PARITY GATE MET (bar 4.28)
M=512 (served ub512 point): unchunk 5.68 | nspan 5.65 (-0.5%)
```

Kernel-only (rocprofv3 traces of the same binary, medians over 60 computes;
parser docs/amd-port/tests/parse_tile_nspan_trace.py): launch counts are
IDENTICAL to the unchunked arm (630 gemm + 630 reduce + 630 dequant + 630
convert over the run - the nspan reduce is tile_fp16_reduce_span, everything
else the same kernels). GEMM-only per shape within 1-3% (ffn_gate 6.35 vs 6.41,
ffn_down 6.82 vs 6.85 TF/s); route call-weighted 4.44 vs unchunk 4.48. The
z-fill the chunked arm forfeited is intact by construction; the residual -1% is
the span reduce's strided dst mapping.

Budget dial (the span loop's cost curve, M=128 wall):
- NSPAN_MIB=16: every shape needing > 16 MiB refuses -> shipped per-call tile
  route; attn_k/v take a 3.5 MiB single-span window; call-weighted 4.34.
- NSPAN_MIB=8: ffn_down runs 9 spans x 192 rows = 24 blocks/launch vs 208
  unchunked -> 2.15 TF/s on that shape (-52%), call-weighted 3.48. Mechanism:
  span fill scales with (span/64) x ks blocks per launch; below ~112 blocks the
  56-CU machine underfills, converging to the chunked arm's slice-launch tax
  (chunked ffn_down 2.18). N-span with a TIGHT window is not a fill cure - the
  cure is the budget admitting full-N spans, which the 200k envelope does.

## 5. VRAM proof (the 200k gate, 0.4 ms sampler methodology)

Window: ONE f16 slab + ONE partials slab per stream, allocated once at first use
with a free-VRAM check (64 MiB margin), reused across spans, shapes, computes.
Max census window: ffn gate/up f16 56.25 MiB (5760 x 5120 x 2 B) + P 11.8 MiB
(M=128) / 47.2 MiB (M=512) = 67.5 / 101.2 MiB one-time. The pool only ever sees
small src1 convert blocks (1.25-4.4 MiB at M=128). Measured peak transient delta
(every transient the route creates, including the one-time window alloc on the
first shape) and steady retention over the 10-shape pass:

```
                unchunk peak   chunked peak   nspan peak   envelope
M=128             102 MiB         52 MiB       102 MiB      200 MiB
M=512             144 MiB         56 MiB       140 MiB      200 MiB
steady 10-shape retention: unchunk +126 MiB vs nspan +88 (M=128);
                           unchunk +248 MiB vs nspan +206 (M=512)
```

FIT at both points. The E-044 C2 abort class (per-call full-weight f16 pool
blocks of a new size class per shape + growing partials) is deleted by
construction - the multi-size-class pool retention that aborted die 0 at E-041
is gone; what remains is one dedicated, refusal-protected cudaMalloc.

## 6. Verdict

- Bit-identity contract: MET (oracle 10/10 x2, single-span AND multi-span
  schedules; full-glue dumps cmp 10/10 at M=128 and M=512; env-unset 10/10 vs
  the pre-change served library).
- 200k memory gate: MET at bench level (peak 102 MiB @M=128, 140 @M=512 vs the
  200 MiB envelope, window one-time and refusal-protected; served boot
  confirmation remains the Gemini lane's cell).
- Parity gate: MET at M=128 (4.31 vs 4.32 same-session control; bar 4.28) and
  M=512 (5.65 vs 5.68). The 5.5+ stretch is reached only at M=512 (5.65), same
  as the unchunked arm - the M=128 wall is dequant+convert-bound, not a span
  effect (GEMM-only 6.35-6.85 TF/s in both arms). N-span SOLVES what chunked
  could not: window-class memory (67.5 MiB steady vs 100s of MiB per-call) with
  NO fill tax (-0.2% wall vs -35%).
- Vs the SHIPPED f32 SGEMM route the tile replaces, the honest comparison stays
  a served-lane cell (E-041 framing unchanged).

## 7. Next link

1. Served boot confirmation (Gemini lane, dies 0-2): guards with
   GGML_CUDA_TILE_FP16=1 + GGML_CUDA_TILE_FP16_NSPAN=1, in-arm greedy
   determinism (two ON boots), MTP accept >= 0.63, needle 3/3, plus the p512
   trace cell: tile_fp16_gemm + tile_fp16_reduce_span displacing the cublas
   kernels, zero off-whitelist launches. Watch die-0 first-prefill: the window
   ask (67.5 MiB + 64 margin at M=128; 101.2 + 64 at M=512) must fit boot-ready
   free VRAM - refusal is clean if not.
2. Endgame design door unchanged: fused dequant-in-staging tile (per-quant-type
   kernel project) deletes the window AND the dequant pass, dropping the M=512
   peak below the chunked arm's 56 MiB while keeping parity.

## 8. Run inventory

- Build: cmake -S . -B build-tile-nspan -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON
  -DAMDGPU_TARGETS=gfx900 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON
  -DGGML_HIP_MMQ_MFMA=ON -DLLAMA_BUILD_{EXAMPLES,TOOLS,TESTS}=OFF -DLLAMA_CURL=OFF
  (cmake at /home/chris/opt/cmake/bin; ggml target only). gfx900, zero warnings.
- Oracle: /tmp/test_tile_nspan_oracle (~2 min). Bench: /tmp/bench_tile_nspan
  built from docs/amd-port/tests/bench_tile_nspan.cpp against
  build-tile-nspan/bin (plus a serve-lib link for the env-unset A/B). Traces:
  rocprofv3 --kernel-trace, CSVs at results/W5_tile_nspan_trace_{nspan,unchunk}_
  2026-09-21.csv, summary results/W5_tile_nspan_trace_summary_2026-09-21.txt;
  clean battery log results/W5_tile_nspan_bench_2026-09-21.txt.
- All die-3 windows held /tmp/campaign_gpu_boot.lock (E-043/E-048 convention).
  Total die-3 time ~30 min. Dies 0-2 untouched.

Files: ggml/src/ggml-cuda/tile-gemm.cu, ggml/src/ggml-cuda/tile-gemm.cuh,
ggml/src/ggml-cuda/common.cuh, ggml/src/ggml-cuda/ggml-cuda.cu,
docs/amd-port/tests/test_tile_integ_oracle.cu,
docs/amd-port/tests/bench_tile_nspan.cpp,
docs/amd-port/tests/parse_tile_nspan_trace.py. Ledger: E-050.
