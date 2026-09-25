W11 ATTENTION DESK - FULL RECEIPT (P0 + A2 + A3 + A4) - 2026-09-23
branch amd/attn-fa (wt-attn-fa), HEAD bd93ac3d1 at receipt time

P0 DECOMPOSITION: see W11_attn_p0_2026-09-23.md. Served instance pinned:
flash_attn_tile<256, 256, 4, 2, false>, T=4, gqa 6/1 per die, ne11 7168
q4_0 KV, LDS 22528 B, grid (1, 37, 3), pb=37; flash class 13.0 ms/round
of which 0.82 ms/round is the f16-KV-pool dequant tax (unbounded, item U1).

A2 (committed 49b21dbc9): wide-GQA tile arms, env GGML_CUDA_FATTN_TILE_GQA_WIDE,
configs (256,256,12) nthreads 192 and (256,256,24) nthreads 384. Hypothesis:
one block covering all Q heads of a KV head streams the f16 KV pool once
instead of gqa_ratio/ncols2 = 3 times, attacking the 22.0 MB/call KV re-read.

A3 REAL-KERNEL MEASUREMENT (the W7 bench law: real template instances, real
launch_fattn host chain, real GGUF bytes)

Instrument: docs/amd-port/tests/bench_attn_real.cu - includes fattn-tile.cu
verbatim, drives launch_fattn<DV,ncols1,ncols2> with the exact served launch
parameters (warp_size 32, nthreads/nbatch_fa from the committed config table),
F16 KV extra data after dst as in production, K/V dequant + parallel-block
scan + combine all on the real path. KV bytes: pread from
Qwen3.8-27B-ASCII-P1M.gguf with q4_0 scales sanitized to fp16(0.03125).
Arms: base <4,2>, mid3 <4,3>, wide6 <4,6>, basedup <4,2> determinism control.
Link: own-tree build (build-hip, canonical flags); linking the campaign
tree's libggml-hip aborts at exit (context dtor layout differs across
branches - noted for future benches, see bench comment).

Protocol (evolved over 8 sessions, final for sessions 6-7): hip events on
ctx.stream() around niter=30 back-to-back launches, 12 rep windows per arm,
arms interleaved per rep; 2500-pass (~8 s) DVFS warmup; rep window 0 is a
measured DVFS warm-in window and is discarded (per-rep diagnostic: rep 0
runs in the post-warmup boost state, later windows in hot-steady state);
spread law 1.05% per config on the remaining 11 windows.

ORACLE: basedup vs base BIT-EXACT every session (in-run determinism).
mid3/wide6 vs base DUST (6143/6144 and 6142/6144 els differ, max rel
1.283e-3) - NOT bit-exact, as the A2 comment predicted: the different
z-split changes the parallel-block split and combine grouping. No
bit-exactness is claimed for the wide arms.

SPREAD LAW: base 0.21% / 0.45% and basedup 0.37% / 0.68% (sessions 6/7) PASS.
mid3 and wide6 windows are bimodal-class (2.67-3.19%, alternating operating
points ~2% apart, visible in the REPS lines) -> VOID as cells per campaign
law; re-run twice with the same verdict; the deltas below are 8-12x the
jitter. Position bias: basedup (4th arm) runs +0.9% vs base (1st arm), so
the printed wide deltas slightly UNDERSTATE the wide cost.

RESULTS (per launch, us, sessions 6 and 7; die 3, cool, lock held):

  arm      s6 med   s7 med   spread s6/s7    vs base
  base      796.6    789.4   0.21%/0.45% PASS    -
  mid3     1007.2    987.3   2.79%/3.19% VOID   +26.47% / +25.07%
  wide6     858.9    861.0   2.67%/2.80% VOID   +7.85%  / +9.07%
  basedup   803.9    796.5   0.37%/0.68% PASS   +0.95%  / +0.90%

Direction-consistent across all 8 sessions (earlier protocol eras included):
wide6 +7.85..+12.44%, mid3 +24.67..+29.20% SLOWER than base.

ROCProf CROSS-CHECK (rocprof --stats + trace, kernel decomposition):

  kernel            med us   grid (flattened)          decode
  dequant_q4_0 x2    23.3    229376x32                 (= 7168 KV rows)
  FA<4,2>           680.7    28416x256 = (1,37,3)      pb=37 (census-exact)
  FA<4,3>           626.4    21504x192 = (1,56,2)      pb=56
  FA<4,6>           515.0    21504x384 = (1,56,1)      pb=56
  combine (pb37)     21.0    6144x256
  combine (pb56)     26.5-30.1

Kernel sums per launch: base 751.5, mid3 727.2, wide6 585.8 us. The wide
arms' MAIN KERNELS WIN - wide6 -24.4% vs base: the KV-streamed-once
hypothesis is CONFIRMED at kernel level. The launches still lose end-to-end
because every wide launch pays:

  (a) a CONSTANT ~210 us GPU idle gap between the K/V dequant pair and the
      fattn kernel (trace: gap_before = 209.5-218.1 us on every FA<4,3>/<4,6>,
      0.0 us on FA<4,2>). --probe mode (same arm back-to-back) reproduces it
      (824 us/launch vs 585.8 kernel sum), so it is intrinsic to the wide
      launches, not a sequence or pool artifact. Host is NOT stalled (HIP API
      trace: every call sub-3 us, no per-launch hipMalloc; pool reuses).
      Wide kernels spill slightly more (scr 1432-1652 vs 1104 B/thread) and
      use 192/384-thread blocks; the ROCm-side mechanism of the 210 us is
      NOT identified - named open item for any future wide-tile attempt.
  (b) combine cost scales with pb (56 vs 37 partials): +26-51% combine.

VERDICT: NEGATIVE - the A2 wide-GQA arm does NOT win at served geometry.
GGML_CUDA_FATTN_TILE_GQA_WIDE stays default-OFF; served-arm spec: NONE
(unchanged instance flash_attn_tile<256,256,4,2,false>). The f16-KV-pool
dequant tax (P0) is unchanged by A2 (equal dequant cost in all arms).
Recommendation: keep the env-gated arm in tree as a documented negative
(zero cost when the env is unset; engagement witness = the INFO line,
negative control verified), OR revert before any campaign merge if the
coordinator prefers minimal surface - the bench stays either way.

A4 BUILD + CI: ggml-hip full target, canonical flags (GGML_HIP=ON, Release,
GGML_NATIVE=ON, CMAKE_HIP_ARCHITECTURES=gfx900, GGML_HIP_RCCL=ON,
LLAMA_CURL=OFF): BUILD-EXIT:0. /home/chris/run_premerge_ci.sh:
CI-VERDICT PASS (tree clean, 6/6 host suites, 7/7 gate-wiring types).

LEVERS NAMED FORWARD (in prize order):
  1. The ~210 us wide-launch GPU start latency (mechanism unknown). If a
     future wide-tile attempt solves it, the kernel-level -166 us (wide6)
     becomes a real ~-18% cell win; today it is fully eaten.
  2. The f16-KV-pool tax (P0 item U1) remains the big flash-class prize:
     0.82 ms/round at 7.1k KV, unbounded in KV depth; a q4_0-direct tile
     kernel (no pool) would also delete the dequant traffic the wide arms
     still pay (46.6 us/launch).
