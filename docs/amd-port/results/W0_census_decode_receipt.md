# W0 Receipt: Decode Census (TP3, no-spec) - 2026-09-21

Instrument: rocprofv3 --kernel-trace, census_decode.sh, run
census_decode_20260921_135523 (90 MB trace + kernel_trace.csv).
Workload: llama-bench -p 8 -n 256 TP3 tensor, ASCII-P1M, q4_0 KV, FA.
Bench during trace: pp8 14.06 / tg256 11.51 t/s (trace overhead ~+7% vs the
12.24 untraced receipt - coherent). Measured decode segment = last ~25 s,
256 steps, ~87 ms/step wall untraced, GPU ~90%+ busy (kernel-bound, NOT
launch-bound: the graph-capture door stays closed).

NOTE: trace includes the warmup decode pass, so the ms/step table below
overstates absolute time (~123 vs ~87 ms); RATIOS are the deliverable.

## Per-step decode decomposition (share of kernel budget)

| component                                     | share | note |
|-----------------------------------------------|-------|------|
| MMVQ iq3_s (FFN/qkv/gate bulk)                | 31.9% | 93 us/call, ~1.8x off byte floor |
| MMVQ iq3_xxs (FFN/attn_out bulk)              | 16.3% | 94 us/call |
| MMVQ iq3_xxs FUSED-GLU (has_fusion=true)      | 12.9% | 249 us/call = gate+up 2x bytes at 2.4x off floor (unfused same work: 2x94=188 us -> fusion is a net LOSS on gfx900 iq types) |
| MMVQ q4_K (attn_q + hundreds of tiny ssm ts)  | 12.1% | dominated by sub-wave calls on <100 KB tensors (latency floor ~40 us) |
| MMVQ iq4_xs (embd/gate/ffn hi-imp)            | 11.7% | |
| MMVQ iq3_s FUSED-GLU                          | 10.5% | 252 us/call, same fused-GLU loss class |
| quantize_q8_1 (x re-quantized per matmul)     | 7.1%  | 1400 calls/step; x is shared by ~500 matmuls/die |
| MMVQ q3_K                                     | 5.0%  | |
| k_bin_bcast add (AR fold adds + residuals)    | 4.5%  | |
| MMVQ iq4_xs FUSED-GLU                         | 4.2%  | |
| flash_attn_ext_vec<256,1,q4_0,q4_0>           | 4.0%  | 48 calls/step (16 attn layers x 3 dies) |
| rms_norm<1024>/<256>                          | 5.0%  | |
| MMVQ q6_K (LM head + hi-import)               | 2.9%  | |
| k_get_rows_float (embd gather)                | 2.4%  | |
| MMVQ q5_K                                     | 2.0%  | |
| GDN chain (gated_delta_net+fwht+l2+conv+glue) | 5.6%  | across 48 layers x 3 dies - cheaper than feared |
| cpy_scalar + copyBuffer (AR staged copies)    | ~1.5% | |
| rope_multi + set_rows q4_0 (KV write)         | 2.1%  | |

MMVQ total = ~77% of decode kernel time. Allreduce total (adds + copies) =
~6% - W4 demoted from 10-20 ms/cycle projection to a minor item.

## Named targets (order by expected yield)

T1. FUSED-GLU MMVQ penalty (correction 2026-09-21 19:1xZ: the 249-252 us/call
    rows are has_fusion=true variants, NOT small_k - template arg 3 is
    fusion, arg 4 small_k). Fused calls move 2x the bytes (gate+up) at 2.4x
    off floor vs 1.85x unfused: fusion is a net loss on gfx900 iq types.
    Pricing probe (zero-code): GGML_CUDA_DISABLE_FUSION=1 A/B via guard
    battery. Scoped fix: gate ggml_cuda_should_fuse_mul_mat_vec_q off for
    iq types on gfx900 (cc<=PASCAL escape exists; AMD offset hides gfx900
    from it). NOTE: DISABLE_FUSION also kills non-GLU fusion classes - a
    positive delta names the family, the scoped switch prices it honestly.
T2. Big-cell MMVQ efficiency (iq3_s/iq3_xxs/iq4_xs/q3_K): 93-96 us/call vs
    ~51 us byte floor on the 9.4 MB/die FFN cells = ~1.8x headroom. The
    gfx900 emulated dp4a (6 VALU) + divergent constant-memory grid LUT
    lookups (GGML_TABLE_BEGIN -> `static constant` on HIP) are the named
    mechanisms; rungs: LDS-resident tables, wide loads, K-split on
    sub-wave rows.
T3. Tiny-tensor tax: ~250 q4_K + ~114 q8_0 + iq3 tiny calls/step at
    5-42 us/call on tensors totaling <1 MB = ~12-15% of step in launch +
    DRAM latency floors. (The only real small_k=true users are a handful
    of q4_K calls at 33.7 us/call - healthy, leave them.) Candidates:
    per-layer grouped GEMV (they share the same x), or fuse the
    ssm_alpha/beta/dt micro-chain per GDN layer.
T4. quantize_q8_1 (7.1%): x is re-quantized ~3x/layer/die; cache/fuse so x
    is quantized once per graph section (or once per matmul group).
T5. FA (4.0%) + GDN (5.6%) + rms (5.0%): hold for W3/W5 after T1-T4.

## Byte floors (TP3 per-die, 183.8 GB/s measured ceiling, T=1)

- FFN 5120x17408 iq3_xxs/iq3_s: 9.4 MB/die -> 51.2 us floor.
- attn_qkv 5120x10240: 5.5 MB/die -> 30 us floor.
- LM head 5120x129272 q6_K: ~137 MB/3 = 45.7 MB -> 249 us floor.
- Full model 11.0 GiB / 3 dies = 3.67 GiB -> 20.0 ms/token floor
  (= 50 t/s no-spec ceiling; x MTP ~1.44 -> ~72 t/s served ceiling).
  Current: 12.24 t/s no-spec = 25% of the bandwidth ceiling.

Files: census_decode_20260921_135523_kernel_trace.csv (delete after
archiving the ranked tables - 603 MB), census_decode_ranked.txt,
census_decode_phases.txt, census_decode_20260921_135523.bench.log.
