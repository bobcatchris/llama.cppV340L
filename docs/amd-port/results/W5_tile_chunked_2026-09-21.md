# W5 tile-chunked desk - chunked-dequant tile arm, bit-identical, window 17.1 MiB

Date: 2026-09-21. Worktree wt-tile-chunked, branch amd/tile-chunked-dequant. GPU:
die 3 only (HIP_VISIBLE_DEVICES=3; dies 0-2 untouched). Scope: make the a8 tile
fit the 200k memory envelope (E-044 C2: per-call full-weight f16 pool allocs OOM
at 200-400 MiB/die free) by dequantizing one split-K slice at a time into a small
reusable window, WITHOUT changing a single output bit vs the unchunked tile.

## 1. What landed (on top of the E-024 integrated tile)

- tile-gemm.cu: tile_fp16_gemm gains a bool ACC template param and two runtime
  params: ksl (slice iteration length; the unchunked path passes K/ks - the same
  value the kernel used to derive internally) and xs (X row stride; unchunked
  passes K). The unchunked arm is the same instruction stream as before (ACC
  false compiles out, same math) - re-proven by the oracle below. New launch
  classes: <...,true> (accumulate into C) and tile_chunk_gather_q (one block per
  weight row, copies the row's slice-spanning quantized blocks into the
  contiguous gather window; 16B/4B/byte paths selected by alignment).
- tile-gemm.cu: GGML_CUDA_TILE_FP16_CHUNKED=1 (on top of GGML_CUDA_TILE_FP16=1)
  routes to tile_fp16_chunked_run: per slice z in 0..ks-1 (ks = the SAME
  tile_fp16_pick_ks choice the unchunked arm makes): gather the row's blocks
  covering [z*ksl, (z+1)*ksl) -> the vetted per-type to_fp16 kernels dequantize
  the gathered contiguous window -> tile_fp16_gemm<ACC = z>0> writes C directly
  (no P, no reduce). Slices can start mid quant block (ffn_down ksl = 2176,
  bs = 256): the window covers the enclosing blocks, L = ceil((r+ksl)/bs)*bs,
  and the GEMM reads it at offset r = kb0 - jb0*bs with row stride L.
- common.cuh / ggml-cuda.cu: ggml_cuda_tile_fp16_chunk - per-stream window pair
  (f16 window + q gather buffer), raw cudaMalloc at first use (NOT the per-call
  pool), grow-only in the max padded slice size, freed at context teardown.
  Budget: GGML_CUDA_TILE_FP16_CHUNKED_MIB (default 64) total per stream AND
  free VRAM minus a 64 MiB margin at decision time; any refusal falls back to
  the shipped route with one WARN (never aborts - the 200k posture is "fit the
  window or get out of the way").
- Bit-identity argument (why the schedule is exact): the chunk boundaries ARE
  the unchunked kernel's gridDim.z partition, each slice launch runs the same
  half2 k-pair schedule over the same f16 values (the gather feeds the same
  to_fp16 kernels the per-call route uses, so the window bits are the per-call
  dequant bits), and ACC-accumulate adds the slice f32 partials in the reduce's
  fixed z order. Same adds, same order, same operands.

## 2. Oracle (die 3, docs/amd-port/tests/test_tile_integ_oracle.cu)

Per census shape: unchunked kernel vs the host-mirror spec (as E-024), plus the
new chunked-vs-unchunked bit gate (chunked arm gathers each slice into a
contiguous N x ksl window via hipMemcpy2D - the f16-level equivalent of the
glue's quant gather - then launches per slice with ACC). 10/10 PASS:
mismatch 0.000%, max rel 0.00e+00 (unchunked still bit-exact vs spec), L2-vs-f64
2.9e-3..6.1e-3 (banked band), chunked bit-diff 0/N on every shape. First oracle
run caught a real harness bug (W indexed as if the slice were contiguous); the
glue itself gathers by construction and was correct.

## 3. Full-glue byte identity (real IQ3_S weights, bench --dump)

bench_tile_chunked.cpp, both arms through the real ggml mul_mat dispatch, f32
dst dumped per shape: cmp 10/10 byte-identical (unchunk vs chunked), M=128.
Env unset: shipped path (3.66 TF/s wall class, no tile lines) - unchanged.
M=640 with env ON: one off-whitelist WARN + shipped route, no tile launch.

## 4. Perf, die 3, census shapes (3 reps x 20 iters, median)

Wall op-level (dequant + convert + GEMM in both arms), M=128:

```
shape        K     N_d  ks | unchunk ms  TF/s | chunked ms  TF/s
ffn_gate   5120  5760   4 |  1.730      4.36 |  1.897      3.98
ffn_up     5120  5760   4 |  1.715      4.40 |  1.899      3.98
ffn_down  17408  1664   8 |  1.655      4.48 |  3.404      2.18
gdn_qkv    5120  3328   8 |  1.001      4.36 |  1.535      2.84
ssm_out    6144  1664   8 |  0.681      3.85 |  1.362      1.92
gdn_gate   5120  2048   8 |  0.672      4.00 |  1.265      2.12
attn_q+gate 5120 4096   4 |  1.370      3.92 |  1.602      3.35
attn_out   6144  1664   8 |  0.665      3.93 |  1.363      1.92
attn_k     5120   256   8 |  0.160      2.10 |  0.903      0.37
attn_v     5120   256   8 |  0.159      2.11 |  0.898      0.37
call-weighted die agg: unchunk 4.28-4.33 (reproduces the E-024 4.33 reference)
                       chunked 2.77-2.81 = -35%
```

Kernel-only (rocprofv3 traces of the same binary, medians over 60 computes;
parser docs/amd-port/tests/parse_tile_chunked_trace.py):

```
                    GEMM-only      route (gemm+reduce / gemm+gather+dequant)
unchunk call-wtd       6.39 TF/s   6.08 TF/s   (gemm class = the 6.10 reference)
chunked call-wtd       3.68 TF/s   2.88 TF/s
```

Per-shape kernel split shows WHERE the chunk tax is: GEMM-only per shape is
-4..-6% for the ks=4 big-N shapes (ffn_gate 6.37 -> 5.96) but -40..-87% for the
ks=8 shapes (ffn_down 6.84 -> 2.65, attn_k 3.21 -> 0.41). Mechanism: the
unchunked launch fills the machine with gridDim.z = ks (ffn_down 208 blocks in
one launch); a slice launch has grid (N_d/64, M/128, 1) - ffn_down 26 blocks,
attn_k 4 blocks - so 8 sequential launches each run at 2-18% of the 56-CU
machine, and the gather+dequant pair (2 more launches per slice) adds 13-25
launches per call vs 4. The dequant+gather work itself is NOT the problem
(unchunk weight dequant is ~460 us of the ffn wall; chunked gather+dequant
totals ~550 us over 4-8 slices).

M=512 spot (the served ub512 point): unchunk wall 5.72 TF/s vs chunked 5.17
(-9.6%) - grid.y = 4 quadruples the per-launch block count and most of the fill
tax closes. VRAM peak deltas at M=512: unchunk 144 MiB vs chunked 56 MiB.

## 5. VRAM proof (the 200k gate)

Static: the chunked route holds ONE window per stream, sized N_d x (ksl+bs) x 2
+ gathered quant bytes - 17.1 MiB for the largest census slice (f16 14,745,600 B
+ q 3,168,000 B, ffn gate/up; M-independent), plus the src1 f16 pool block the
unchunked arm allocates too (1.25 MiB at M=128). No partials buffer (the P path
is unused; ks*M*N*4 = 11.8 MiB/call at M=128 saved by construction).
Empirical: bench sampler thread polling free VRAM every 0.4 ms, per-shape
steady-to-min dip = every transient the route creates:

```
                unchunk peak   chunked peak   envelope
M=128           104 MiB        52 MiB         200 MiB
M=512           144 MiB        56 MiB         200 MiB
```

FIT in the 200 MiB envelope at both points; the E-044 C2 abort class (per-call
58 MiB weight f16 blocks of a NEW size class per shape + growing partials) is
deleted - the window is allocated once, reused across chunks, shapes and
computes, and the pool only ever sees small src1 blocks. Caveat of record: one
early bench pass showed a spurious ~734 MiB free-VRAM swing - traced to the
desktop session (Xorg/compositors hold card 3) churning VRAM mid-run, not to
the route; the clean back-to-back rerun above declines smoothly (per-shape
post-alloc vs post-warmup free printed to stderr in the saved run logs).

## 6. Verdict

- Bit-identity contract: MET (oracle 0 bit diffs x10 shapes; full-glue dumps cmp
  10/10). The chunked arm IS the a8 tile numerically - the served protocol stays
  "greedy determinism within the arm", not byte-identity vs shipped.
- 200k memory viability: MET at bench level (17.1 MiB steady window, 52-56 MiB
  measured peak transient vs 200 MiB envelope; the unchunked arm's 104-144 MiB
  peak plus multi-size-class pool retention is what aborted die 0 at E-041).
  Served confirmation still needs a TP3 boot on the Gemini lane (dies 0-2).
- Perf: NEGATIVE vs the unchunked tile at M=128 (2.8 vs 4.3 wall) because the
  bit-identity-preserving slice partition forfeits the unchunked launch's
  gridDim.z fill; at M=512 the gap is -9.6% (5.17 vs 5.72). Honest 200k
  comparison is vs the SHIPPED f32 SGEMM route the tile replaces (E-041: tile
  displaced rocblas 6729 -> 4656 ms at p512 kernel time) - that A/B is a served
  lane cell, not a die-3 bench cell.

## 7. Next link

Two candidate replacements for the schedule, both keeping the oracle contract:
1. N-span chunking with FULL-K windows: chunk = contiguous quant ROW block
   (no gather needed at all - to_fp16_cuda runs on the row slab directly),
   window = rows x K x 2 (budget ~16 MiB -> ~1800 rows at K=5120, ~480 at
   K=17408), launch keeps gridDim.z = ks and the P+reduce path per chunk, so
   per-launch blocks scale with ks again and the fill tax closes; bit-identity
   holds because every output element still sees its whole unchunked slice
   schedule and reduce order.
2. Fused dequant-in-staging tile (decode quant blocks inside the kernel's W
   staging): deletes the window AND the extra pass, the real endgame, but a
   per-quant-type kernel project.
If the memory win is needed before either lands, this arm is promotable as-is
for the 200k envelope at the M=512 wall price (-9.6% vs the OOM-ing unchunked
tile; the alternative at 200k remains the shipped f32 route).

## 8. Run inventory

- Build: cmake -S . -B build-tile-chunked -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON
  -DAMDGPU_TARGETS=gfx900 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON
  -DGGML_HIP_MMQ_MFMA=ON -DLLAMA_BUILD_{EXAMPLES,TOOLS,TESTS}=OFF -DLLAMA_CURL=OFF
  (cmake at /home/chris/opt/cmake/bin; build the `ggml` target only - this
  worktree's llama-app target needs flags the desk does not use). gfx900, zero
  warnings.
- Oracle: /tmp/test_tile_chunked_oracle (~2 min). Bench: /tmp/bench_tile_chunked
  built from docs/amd-port/tests/bench_tile_chunked.cpp against
  build-tile-chunked/bin. Traces: rocprofv3 --kernel-trace, CSVs at
  results/W5_tile_chunked_trace_{unchunk,chunked}_2026-09-21.csv; run logs
  results/W5_tile_chunked_bench_{unchunk,chunked}_2026-09-21.txt.
- Die-3 windows used: oracle ~2 min, benches+traces ~15 min, edge checks ~3 min.
  Dies 0-2 untouched.

Files: ggml/src/ggml-cuda/tile-gemm.cu, ggml/src/ggml-cuda/tile-gemm.cuh,
ggml/src/ggml-cuda/common.cuh, ggml/src/ggml-cuda/ggml-cuda.cu,
docs/amd-port/tests/test_tile_integ_oracle.cu,
docs/amd-port/tests/bench_tile_chunked.cpp,
docs/amd-port/tests/parse_tile_chunked_trace.py. Ledger: E-045.
