# W13 PREFILL DESK - served prefill decomposition + ranked levers (gfx900 TP4)

Date: 2026-09-23. Desk: PREFILL (wt-prefill-2, amd/prefill-2 off 676cbc806). Zero-GPU:
census phase-split + in-tree path audit + lever enumeration. No runtime code changed.

Instruments: docs/amd-port/scripts/prefill_phase_split.py (phase-split analyzer),
raw dump docs/amd-port/results/W13_prefill_phase_split_2026-09-23.txt. Census of
record: TP4_kernel_census_2026-09-23_kernel_trace.csv (589,000 dispatches, ns
timestamps; one probe = 6964-token prefill + 98 decode tokens; the kernel_trace
CSV lives on disk in the main tree only - gitignored per E-115).

## 1. P0 - phase split (per the round-map method, by Cijk-burst boundary)

Windows detected in the trace: warmup mmvq burst 14.3-14.7 s (model-load warmup,
excluded), PREFILL = Cijk burst [19.002 .. 55.843] s = 36.84 s (server log says
prompt eval 37.04 s - 0.5% agreement), DECODE = [55.843 .. 59.882] s = 4.04 s.

Decode-window cross-check vs the round map: mmvq 53.7% / nccl 22.7% / flash 11.6%
of kernel time, 4.04 s for 98 tokens - consistent with 130.5 ms/round x 30 rounds.
The decode structure of record is unchanged; the rest of this receipt is prefill.

PREFILL: 231,285 launches, 142.35 s kernel time across 4 dies = 35.6 s/die busy
in a 36.84 s wall = 96.6% busy (no host-slice lever in prefill, unlike decode's
6.5 ms/round). Zero MMVQ launches in pure prefill: every dense GEMM has
src1_ncols = 512 (or the 268-token tail), above MMVQ_MAX_BATCH_SIZE.

Per-die prefill families (kernel time share, 36.84 s wall):

| family | die 1 | die 2 | die 3 | die 4 | notes |
|---|---|---|---|---|---|
| Cijk (hipBLAS GEMM) | 14.73 s 41.4% | 14.12 s 39.7% | 14.95 s 42.0% | 15.73 s 44.2% | all HALF kernels, see below |
| flash_attn_tile     | 10.47 s 29.4% | 10.42 s 29.3% | 10.45 s 29.4% | 10.59 s 29.8% | prefill is flash-heavy |
| ncclDevKernel       | 6.03 s 16.9% | 6.94 s 19.5% | 5.91 s 16.6% | 4.68 s 13.2% | x1.48 die asymmetry |
| other (norms, gdn, copies, staging) | 4.38 s 12.3% | 4.12 s 11.6% | 4.25 s 11.9% | 4.56 s 12.8% | |

## 2. P0 - the prefill GEMM table (which library, which instantiation)

In-tree path of record (file:line):
- mmq.cu:368-372 - gfx900 (GGML_CUDA_CC_VEGA) `return n_experts > 0`: MMQ is OFF
  for dense prefill (upstream d9df11006/25c55df18 class decision), so quantized
  dense matmuls fall through ggml-cuda.cu:2756 to ggml_cuda_op_mul_mat_cublas
  (ggml-cuda.cu:1724).
- ggml-cuda.cu:1761 - the W5 tile route (tile-gemm.cu) is tried FIRST, env-gated,
  default OFF (served = off).
- ggml-cuda.cu:1801 use_fp16 requires row_diff == src0->ne[1]: TRUE for every
  shard the meta backend hands the CUDA op (each die's shard carries local ne[1],
  rows full-range), so ALL dense prefill GEMMs take the fp16 branch:
  dequant-to-f16 + cublasGemmEx CUDA_R_16F / CUBLAS_COMPUTE_16F, f16 out + f32
  convert (ggml-cuda.cu:1838-1861). The bf16 branch (:1769) and the f32 Sgemm
  branch (:1868) NEVER run in the census - there are zero non-HB Cijk kernels.

hipBLAS kernel census (prefill, all 4 dies, ALL are rocBLAS half "HB" kernels,
ISA900, WS64): TOTAL Cijk = 59.54 s = 14.89 s/die avg.

| variant (MT_MxNxK, WG) | launches | ms | us/launch | ms/die | class |
|---|---|---|---|---|---|
| MT32x32x32 WG8x8 | 8288 | 38448.1 | 4639.0 | 9.61 | gate/up-class (grid 8704x16 x6760 = 4 dies x 65 layers x 2 x 13 full ubatches) |
| MT128x64x16 WG16x16 | 6760 | 10351.6 | 1531.3 | 2.59 | single grid family 10240x8 |
| MT64x64x16 WG16x16 | 5704 | 7651.0 | 1341.3 | 1.91 | grids 10240x8 + 6144x8 + 20480x5 |
| MT16x16x24 WG8x8 | 7800 | 2168.8 | 278.0 | 0.54 | small projections (grids 64x32, 1024x32) |
| 5 tail variants (MT32x32x24, MT64x32x32, MT64x32x16, MT16x16x24 b) | 1856 | 922.6 | - | 0.23 | tail ubatch (268 tokens, grid y=9) + odd shapes |

ubatch anatomy: 13 full 512-token ubatches + 1 x 268-token tail = 14 (6964 tokens;
grid.y=9 x MT32 = 288 = padded tail). GEMM calls: 30,240 = 7,560/die = 540/ub.
Effective GEMM rate ~5.6 TF/s/die average (2 x 512 x ~5.8 B params / 1.06 s per
ubatch per die) - in the W5 rb16 band; M=512 is already a good shape for rocBLAS.

Staging class (the convert_unary + dequant pool, per-call, NOT cached):

| staging kernel | launches | ms (4 dies) | role |
|---|---|---|---|
| dequantize_block_* <__half> (12 types) | 31,098 | 5125.1 | weight dequant to f16, PER GEMM CALL PER UBATCH (same weights re-dequantized 14x per prefill) |
| convert_unary<float,__half> | 30,239 | 1699.1 | src1 f32->f16, one per GEMM call |
| convert_unary<__half,float> | 30,240 | 1682.2 | f16 GEMM out -> f32 (COMPUTE_16F f16-out tax) |
| convert_unary<float,bf16> + <bf16,float> | 14,555 | 1293.1 | E-101 bf16 boundary compress/decompress, 4 launches per boundary per ubatch (130 x 14 x 4 = 7280 each way) |
| TOTAL staging | ~106k | 9799.5 | 2.45 s/die = 6.9% of prefill busy |

The task's "75k convert_unary launches" = 30,239 + 30,240 + 7,279 + 7,276 = 75,034
in prefill. Verdict on caching: the weight->f16 dequant is per-ubatch and UNCACHED;
full f16 residency is VRAM-infeasible (f16 weight copies ~13+ GB/die vs 16 GB dies
at a 200k KV budget), and the only in-tree residency mechanism
(GGML_CUDA_TILE_FP16_RESIDENT, tile-gemm.cu:214-299, 4 GiB cap) is tile-route-gated.

flash_attn detail: flash_attn_tile<256,256,16,2> = 41.78 s / 1020 launches =
40.96 ms avg = 16 full-attn layers x 16 ubatches x 4 dies - ~10.5 s/die at an
effective utilization of well under 1 TF/s (T=512 x KV<=7k x 6 q-heads/die class).
flash is 29.5% of prefill: the largest single pool after GEMM, and the top future
kernel-desk target (decode's flash uses the <...,4,2> variant at 716-804 us).

NCCL detail: 7,799 launches / avg 3.02 ms (10.5 MB f32 -> 5.2 MB bf16 boundary
volumes, vs the 80 KB decode class). Per-die spread 4.68-6.94 s (x1.48): the
E-117a/E-119 compute-arrival asymmetry EXISTS IN PREFILL TOO; the wall pays
die 2's 6.94 s. Full flatten to die 4 = 2.25 s = 6.1% of prefill wall.

## 3. A1 - ranked levers (numerics class per entry)

| rank | lever | class | expected prefill delta | basis |
|---|---|---|---|---|
| 1 | ub1024 prefill arm: --batch-size 1024 --ubatch-size 1024 (serving of record is b512/ub512, launch_tp3_200k.sh) | launch flag only; per-row math unchanged; dust-risk: rocBLAS may pick a different instantiation (split-K order) at M=1024, so treat as possible byte-diff until the determinism protocol says otherwise | +5-10%: staging passes halve (~6.9%/2), boundary count halves (latency + peer-wait share), M=1024 GEMM shapes | census 2; watch VRAM at 200k slot (auto boot gate) |
| 2 | NCCL_MIN_NCHANNELS=4 | pure env, ring order unchanged (E-117a fingerprint) | unknown at 5.2 MB; bounded by the 6.9 s pool; probe-proven -11%/op at 80 KB and helps the 20 KB draft class too | W9/E-117a |
| 3 | GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1 (exists in-tree, ggml-cuda.cu:1705) | NUMERICS-CHANGING (f16-in/f32-acc/f32-out; deletes the 1.68 s f16-out convert = 4.7% of busy) - FLAG FOR OWNER; perf risk: gfx900 packed-f16 rate may halve with f32 acc - bench before serving | -4.7% staging, GEMM-side sign unknown | census 2 |
| 4 | tile-fp16 route A/B (GGML_CUDA_TILE_FP16[+_NSPAN], in-tree, compiled, gfx900-whitelisted at tile-gemm.cu:623) | NUMERICS-CHANGING (fp16-tile class; E-022 contract: within-arm determinism + accept canary, not byte-identity) - FLAG FOR OWNER | DOWNGRADED by this census: served rocBLAS HB already ~5.6 TF/s avg at M=512 vs the tile's M=512 wall 5.65 (W5_tile_nspan) = parity-to-negative; the tile's own bench wins (ffn_down 1.75x, ssm_out 1.57x) sit on K=17408/6144 weights that are K-sharded under TP4 to local K {4352,1536,1280} = NOT in the whitelist trunk-K set {5120,6144,17408}. Covering them needs a whitelist + bench extension (design below) | census 2 + W5 receipts |
| 5 | split-balance (E-119 desk, already dispatched) | pure flag | prefill-side prize priced HERE: up to ~2.25 s = 6.1% if die 2 flattens to die 4 | census 1 |
| 6 | hipBLASLt / algo tuning | DEAD on gfx900: hipBLASLt does not support Vega on ROCm 6.2.0; rocBLAS internal heuristics expose no supported algo-selection env; GGML_CUDA_FORCE_CUBLAS / MMQ toggles would move OFF the measured-optimal route | 0 (negative of record) | this desk |

Designed-not-built (owner-gated, none bit-exact-safe enough to land without a
served window):
- tile whitelist extension to the K-sharded local set {1280, 1536, 4352} +
  tile_fp16_pick_ks coverage + real-kernel bench (bench_tile_integ class) before
  any serve; targets the 2.59+1.91 s/die MT128/MT64 classes and (via NSPAN+
  RESIDENT) the dequant tax on the biggest classes.
- shipped-route f16 weight residency for a VRAM-budgeted subset (bit-exact class:
  same f16 bits into the same hgemm; same grow-only-refusal pattern as
  tile-gemm.cu:214-299 but feeding ggml-cuda.cu:1838) - superseded in practice by
  lever 1 which halves the same tax with zero code.
- prefill flash kernel desk (29.5% pool, sub-1 TF/s utilization) - biggest single
  future prize, kernel-build class, out of scope here.

## 4. A2 - implementation verdict

NO CODE CHANGE SHIPPED. Every lever that survives ranking is a launch flag or env
(lane 1/2/5), and every numerics-changing route (lane 3/4 + designs) is owner-flag
territory per the brief (E-101's bf16-compress decision is NOT reopened - the
boundary class stays; lane 3/4 are GEMM-class changes, a different domain, but the
same flag-for-owner law applies). The bit-exact code candidate (shipped-route
weight residency) loses to lever 1 on risk-adjusted value: same saving, zero code.
No ggml-cuda file touched -> A4 build/CI not required by the ladder law.

## 5. A3 - served-arm spec for the coordinator (next validation window)

All arms on the current serving of record (launch_tp3_200k.sh env set unchanged),
provenance-gated per E-112, each arm = 200k battery + 10k decode cell +
decode-guard vs the re-stamped baseline (E-110 law), interleaved or order-rotated
per the E-119 soak finding (200k cells are position-sensitive; 10k cells are the
decision-grade tiebreak; prefill cells are the primary metric here).

ARM P1 - ub1024 (primary, schedule-only class):
- launch_tp3_200k_ub1024.sh = launch_tp3_200k.sh with
  `--batch-size 1024 --ubatch-size 1024` (both, ub <= b required).
- Gates: (1) within-arm greedy determinism, two consecutive ON boots
  byte-identical; (2) cross-arm vs ub512: IF outputs are byte-identical, the arm
  is schedule-only and banks on prefill+decode gates alone; IF they differ, the
  arm is a dust-class change - owner call, then accept >= 0.63 + needle 3/3
  become the canaries (E-022/E-101 contract style); (3) decode-guard cell MUST
  pass vs baseline (decode ubatches are T<=8 and structurally unaffected, but
  the E-110 law does not carve exceptions); (4) VRAM: boot must clear the 200k
  slot (activation doubling; refusal = clean fail, no partial credit).
- Expected: prefill +5-10% (188 -> 198-207 t/s census-class; 195-200 -> 205-220
  served-class), decode neutral.

ARM P2 - chan4 (pure env, may ride the same window as a separate arm):
- `NCCL_MIN_NCHANNELS=4` appended; engagement = NCCL INFO fingerprint line in the
  server log (REQUIRED in the verdict, E-117a). Expected: prefill 0 to +3%
  (bounded by the 6.9 s die-2 pool), decode 0 to +1%.

ARM P3 - f32-acc hgemm (ONLY with owner sign-off, numerics class change):
- `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1`; engagement = the
  "Detected GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F" INFO line (ggml-cuda.cu:1711).
- Cheap pre-bench first (die 2 microbench of the MT32x32x32 shapes f32-acc vs
  f16-acc) - if the GEMM itself slows more than ~5% the arm is dead before
  serving. Upside ceiling: the 1.68 s f16-out convert (4.7% of busy).

Tile arm: not recommended for this window (census pricing parity-to-negative at
ub512); revisit only together with the whitelist-extension design.

## 6. Process

- Zero GPU used (census + tree audit only); no lock taken, no server touched.
- CHECKIN.log per the ~30-min law (fresh log; the inherited rccl-transport lines
  were replaced at desk start per the coordinator erratum).
- Commits: P0 analyzer + raw dump (b049b7d10), A1 window-fix + levers (03b20f6a2),
  receipt + ledger (final). No pushes, no PRs.
