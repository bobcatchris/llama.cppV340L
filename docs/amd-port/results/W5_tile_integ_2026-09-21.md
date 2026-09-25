# W5 tile-integ desk - env-gated a8 tile dispatch in ggml, validated die 3

Date: 2026-09-21. Worktree wt-tile-integ, branch amd/tile-integ. GPU: die 3 only
(HIP_VISIBLE_DEVICES=3; dies 0-2 untouched). Scope: wire the oracle-gated a8 tile
(E-016/E-022, 5.53 TF/s/die banked) into the shipped dense-prefill dispatch behind
GGML_CUDA_TILE_FP16=1, validate the integrated build per shape, and name what is
still deferred.

## 1. What landed

- ggml/src/ggml-cuda/tile-gemm.cuh - geometry defines + split-K picker + glue decl,
  device-code free.
- ggml/src/ggml-cuda/tile-gemm.cu - a8 kernel ported verbatim from
  docs/amd-port/tests/bench_tile_gfx900.cu (one change: dst row stride ldc decoupled
  from weight N, for the strided main-device slice), fixed-order f32 reduce,
  static-once env read (same pattern as cublas_force_compute_type), census whitelist,
  f16 dequant + convert + launch.
- ggml/src/ggml-cuda/ggml-cuda.cu - one include + one branch at the top of
  ggml_cuda_op_mul_mat_cublas (the dense fallback op; HEAD line ~1663). Returns false
  with zero side effects when env unset or shape off-list; the shipped branches then
  run unchanged.

Branch point note: the branch sits BEFORE the bf16/h16/f32 sub-branches because
use_fp16 requires row_diff == src0->ne[1] (ggml-cuda.cu:1660) - true on single GPU,
FALSE under TP3 row-split. So the shipped route the tile replaces is:
- single GPU: dequant-to-f16 + rocBLAS hgemm h16-acc/h16-out + f16->f32 convert
  (this bench, matching the banked E-015/E-022 anchor), or
- TP3 split (served): dequant-to-f32 + cublasSgemm f32-acc/f32-out - the E-012
  survey's "h16 branch engages" holds only non-split. The tile arm therefore also
  deletes the 2x-byte f32 dequant write tax in-server on top of the GEMM delta.

Whitelist (all must hold, else one WARN + shipped route): GGML_CUDA_CC_VEGA (gfx900,
the benched platform), quantized src0 (f16 weights produced by the same per-call
dequant the shipped h16 route uses; no loader change in v1), src1 f32,
GGML_PREC_DEFAULT, contiguous src0, M <= 512 and M % 128 == 0, K in {5120, 6144,
17408}, per-die N share >= 64 and % 64 == 0 (covers the 128-row-block share wobble),
32-bit index and 16-byte alignment guards of the kernel.

## 2. Oracle (gate before timing)

docs/amd-port/tests/test_tile_integ_oracle.cu includes tile-gemm.cu and gates the
PORTED kernel vs the tile spec: host mirror of the exact accumulation (chunk
schedule, per-slice half2 state via f64 fused-half emulation, fixed-order f32 slice
sum) + fp64 reference. 10/10 census shapes (die0 shares, ks as dispatched):
mismatch 0.000%, max rel 0.00e+00 (bit-exact vs spec), L2-vs-f64 2.9e-3 to 6.1e-3 -
inside the banked a8 band (2.3-5.4e-3 class). Build/run line in the file header.

## 3. Byte-identity when env unset

Same bench binary linked against the pre-change served library
(../llama.cpp/build-hip/bin) and the integrated build; identical deterministic inputs
(real IQ3_S quantization of sane random f32), single compute per shape, f32 dst
dumped and cmp'd: 10/10 shapes byte-identical. Env unset = byte-identical shipped
path, empirically proven, not just argued.

## 4. Integrated per-shape TF/s vs banked (die 3, M=128)

rocprofv3 kernel trace, 3 reps x 20 iters, median; tile arm = tile_fp16_gemm +
tile_fp16_reduce; shipped arm = rocBLAS HB hgemm (h16 route, kernels visible in
results/W5_tile_integ_trace_off_2026-09-21.csv).

```
shape        K     N_d   | integ TF/s  bank a8  bank rb16 | shipped TF/s  vs ship
ffn_gate     5120  5760  |   6.22       5.60      5.67     |   6.54        0.95x
ffn_up       5120  5760  |   6.18       5.61      5.66     |   6.54        0.94x
ffn_down    17408  1664  |   6.58       6.15      3.72     |   3.77        1.75x
gdn_qkv      5120  3328  |   6.17       5.18      4.36     |   4.11        1.50x
ssm_out      6144  1664  |   5.50       4.98      3.52     |   3.51        1.57x
gdn_gate     5120  2048  |   5.73       5.34      4.49     |   4.79        1.20x
attn_q+gate  5120  4096  |   5.34       5.36      3.66     |   4.09        1.31x
attn_out     6144  1664  |   5.33       4.94      3.71     |   3.54        1.51x
attn_k       5120   256  |   3.09       2.86      1.56     |   1.66        1.86x
attn_v       5120   256  |   3.09       2.86      1.57     |   1.66        1.86x

call-weighted die0 agg (E-015 weights): integrated tile 6.10 TF/s/die
  same-session shipped h16 control 4.79 -> +27.5%
  vs banked a8 5.53/5.56 class -> REPRODUCED AND EXCEEDED (bar was "~5.5 within noise")
```

GATE MET. No integration cost at the kernel level. Two honest couplings found:
1. The integrated GEMM reads f16 weights that the retained per-call dequant JUST
   wrote (Vega 16 MB L2) - small slices (attn_k/v, 2.6 MB) get real L2 residency,
   which is part of why integrated > standalone-harness banked numbers. This credit
   disappears once the loader keeps f16 steady state; the standalone harness numbers
   remain the reference for the post-loader state.
2. This session's rocBLAS control ran hot (4.79 vs banked 4.52, +6%; ffn gate/up
   6.54 vs 5.67) - same direction as the ~1.5% control drift the tile desk saw;
   per-shape ratios vs same-session control are the honest comparison.

Wall op-level (dequant + convert + GEMM, still IN both arms in v1), call-weighted
agg: OFF 3.52 -> ON 4.33 TF/s/die (+23.0%) (e.g. ffn_down 2.50 -> 1.67 ms/iter,
attn_k 0.26 -> 0.158 ms). The remaining wall gap to the GEMM numbers is exactly the
v1-retained per-call dequant + src1 f32->f16 convert (110 launches/10-shape pass,
24.1 ms vs 70.5 ms of tile GEMM) - deleted by the later loader step, not by dispatch.

## 5. Whitelist tiers + fallback behavior

- M=256/M=512 census spot checks (env ON): wall 5.0-5.6 TF/s class, all shapes route
  to the tile (gridDim.y = M/128); full per-shape re-derivation at M>128 deferred to
  a Gemini-lane bench (E-012's "re-derive at M=256/512" caveat still stands for perf,
  correctness is oracle-covered by construction of the M tiles).
- M=640 off-list, env ON: exactly one WARN "shape off the census whitelist, shipped
  route in use", shipped times, no tile engagement. Route logs fire once per process
  ("tile route engaged (K=.. N_d=.. M=.. ks=.. ldc=..)") - die-3 stderr captured.

## 6. Numerics - the arm is numerics-changing

Tile = f16-in, half2 k-pair accumulation, split-K f32 slice reduce, f32 out.
Shipped h16 (this bench) = full-K f16 accumulate, f16 out + convert; shipped TP3
f32 route = f32 in/acc. Arm-vs-arm delta on identical inputs: L2 rel 1.1-2.0e-2,
both arms finite with sane weights. vs f64 the tile was 10-19x MORE accurate than
h16 (E-016). Greedy determinism WITHIN the arm holds by construction: fixed grid,
fixed ks, no atomics, fixed-order reduce. This arm MUST pass the served
byte-difference protocol before promotion - NOT byte-identity vs the shipped arm.

## 7. Deferred served-validation steps (Gemini lane, dies 0-2)

1. Rebuild the served binary from this worktree (build-hip flags: GGML_HIP=ON,
   AMDGPU_TARGETS=gfx900, GGML_HIP_GRAPHS=ON, GGML_HIP_NO_VMM=ON,
   GGML_HIP_MMQ_MFMA=ON). Guards GREEN with env unset (baseline re-stamp).
2. Tile arm (GGML_CUDA_TILE_FP16=1): hardened guard battery. GATES: greedy
   determinism byte-identical WITHIN the arm (two consecutive ON boots, NOT vs the
   OFF arm), MTP accept >= 0.63, needle 3/3. Expect a different sha256 vs the OFF
   arm - that is the numerics-changing contract, the canary is acceptance.
3. Trace arm: census kernel-trace on TP3 - confirm tile_fp16_gemm/reduce replace the
   cublas kernels (under split these replace the SGEMM f32 route: expect a larger
   delta than this single-GPU bench) and no tile launches off-whitelist (M>512 ub512
   chunks must NOT appear).
4. Perf arm: 3-rep cold-stamped OFF/ON A/B at pp (and decode sanity - M<=8 routes to
   mmvq, unaffected). Promote per campaign bar; watch ffn gate/up parity.
5. Then the loader step (separate desk): load-time f16 steady-state weights delete
   the retained per-call dequant (24.1 ms/10-shape-pass here; ~16-32 GB/die/ubatch
   in-server) - after which the standalone-harness numbers (5.53) are the reference,
   not the L2-coupled v1 numbers.

## 8. Run inventory

- Build: cmake -S . -B build-tile-integ -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON
  -DAMDGPU_TARGETS=gfx900 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON
  -DGGML_HIP_MMQ_MFMA=ON -DLLAMA_BUILD_{EXAMPLES,TOOLS,TESTS}=OFF -DLLAMA_CURL=OFF
  (cmake at /home/chris/opt/cmake/bin). ggml + tile TUs compile clean, zero warnings.
- Bench: docs/amd-port/tests/bench_tile_integ.cpp, linked against
  build-tile-integ/bin (and against the pre-change ../llama.cpp/build-hip/bin for
  the byte-identity arm). Traces: results/W5_tile_integ_trace_{on,off}_2026-09-21.csv.
- Die-3 windows used: oracle (~2 min), wall/trace arms (~6 min), edge checks
  (~3 min). Dies 0-2 untouched.

Files: ggml/src/ggml-cuda/tile-gemm.cuh, ggml/src/ggml-cuda/tile-gemm.cu,
ggml/src/ggml-cuda/ggml-cuda.cu, docs/amd-port/tests/test_tile_integ_oracle.cu,
docs/amd-port/tests/bench_tile_integ.cpp. Ledger: E-024.
