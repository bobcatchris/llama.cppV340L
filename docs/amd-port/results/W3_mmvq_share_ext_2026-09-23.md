# W3 MMVQ Share-Extension Receipt: engagement fix + 6 served types - 2026-09-23

Instrument: docs/amd-port/tests/bench_mmvq_share_gfx900.cu v1 (memcmp-gated,
one binary), tree amd/mmvq-kernel2 (post amd/v340-port-v2 merge), hipcc
rocm-6.2.0 gfx900, die 3 (HIP device 3), lock-compliant. Weights = real GGUF
tensor bytes per type (offset-delta law), synthetic valid q8_1 x 4 tokens.
SCHEDULE OF RECORD: the REAL mul_mat_vec_q GCN shape at ncols_dst 2..4 -
rows_per_cuda_block = 2, nwarps = 2 (block 64x2), kbx = tid/(qi/vdr),
kqs = vdr*(tid % (qi/vdr)), blocks_per_iter = vdr*128/qi. Arms: base = the
exact shipped vec_dot_<t>_q8_1 per (token, row); share = the exact new
vec_dot_<t>_q8_1_decode/apply split pair from vecdotq.cuh.

## RUNG 0 (engagement fix): the shipped iq3_s share branch was DEAD on GCN

calc_rows_per_block(MMVQ_PARAMETERS_GCN, ncols_dst 2..4) returns 2, but the
shipped branch required rows_per_cuda_block == 1 -> compile-time dead for
every served T=2..4 instantiation on gfx900. The E-095 served A/B neutral
and its "structure-bound attribution" had a simpler mechanical cause: the
share path never ran. Fix: decode once per (row, kbx, lane) - rows share
nothing - then apply per (j, i); rows_per_cuda_block 1 and 2 both covered
(RDNA tables use 1, GCN uses 2). The iq3_s real-schedule re-measure after
the fix: share wins only **-2.8%** at T=4 (816.5 -> 793.6 us, PASS session)
vs the -43% of the W2 rpb=1 clone. The clone schedule, not the served one,
produced the headline. Both served-neutral causes now named.

## EXTENSION RUNGS (per-type, each with its own env gate; bit-exact gate =
device memcmp of the full dst vs the base arm at the same T, N rows x T
tokens, before any timing counts)

| type | trace T=4 share | session of record | T=2 / T=3 / T=4 verdict | acceptance |
|------|-----------------|-------------------|-------------------------|------------|
| 12 q4_K    | 837 ms (14.3%) | S2 (spread 0.84% PASS) | -20.0 / -21.5 / **-21.6%** | BITEXACT T=2,3,4 |
| 11 q3_K    | 228 ms (3.9%)  | S2 (spread 0.99% PASS) | -27.5 / -31.9 / **-28.4%** | BITEXACT T=2,3,4 |
| 21 iq3_s   | 1528 ms (26%)  | S2 (spread 0.87% PASS) | -8.5 / -5.7 / **-2.8%** (rung 0) | BITEXACT T=2,3,4 |
| 18 iq3_xxs | 1112 ms (18.9%)| S3 dedicated (0.67% PASS) | -0.4 / -3.5 / **-3.8%** | BITEXACT T=2,3,4 |
| 13 q5_K    | 154 ms (2.6%)  | 5 sessions, stable | -10.0 / -20.0 / **-21.5%** (+-1.2) | BITEXACT T=2,3,4 |
| 14 q6_K    | 145 ms (2.5%)  | 5 sessions, stable | -2.4 / -4.9 / **-5.8%** (+-1.3) | BITEXACT T=2,3,4 |
| 23 iq4_xs  | 789 ms (13.4%) | 4 sessions, stable | NEGATIVE +6.1 / +2.4 / **+5.8%** | BITEXACT T=2,3,4, KEPT OFF |

Session-law notes (honest): q4_K/q3_K/iq3_s/iq3_xxs have PASS sessions of
record (per-config rep spread < 1%). q5_K/q6_K/iq4_xs trip the 1% spread
gate on a monotonic thermal ramp that hits BOTH arms (A/B interleaved per
rep; dup controls t4_base2/t4_share2 bit-identical in every session; verdict
range across 5/5/4 sessions +-1.3 points). The ramp is the known platform
thermal class (E-098/E-100). The iq4_xs T=4 decision cells themselves are
clean (t4_base 0.36%, t4_share 0.35% spread); its gate trips are on the
losing arm at T=2/3 and do not affect the keep-OFF decision.

## HEADLINE: bank -20..-28% on the three K-quant cells; IQ packed types ~-3..-6%; iq4_xs negative

Share wins scale with how decode-bound the base is on the REAL schedule
(per-sub-block scale walks: q4_K/q5_K/q3_K win big; packed one-word-scale
IQ types: iq3_xxs/iq3_s win little; LUT-select iq4_xs: negative), NOT with
type weight share. The W2 -43% class does not survive the rpb=2 schedule.

## Served projection (static-count class, trace CSV shares; candidate only)

- q4_K 837.2 ms x -21.6% = -180.8 ms
- q3_K 228.1 ms x -28.4% = -64.8 ms
- q5_K 154.4 ms x -21.5% = -33.2 ms
- iq3_xxs 1112.4 ms x -3.8% = -42.3 ms
- q6_K 145.2 ms x -5.8% = -8.4 ms
- iq3_s 1528.4 ms x -2.8% (rung 0) = -42.8 ms
- iq4_xs: 0 (OFF)
Total: -372 ms of 5873 ms MMVQ = **-6.3% MMVQ** (-7.6% of the T=4 band) ->
-4.1% of traced kernel time -> served decode upper bound ~x1.04-1.05 IF
decode wall scales with MMVQ kernel time (E-095 structure-bound caveat now
measures a REAL -6.3% candidate instead of a dead branch). The prior
"-25-30% MMVQ" projection was the rpb=1-clone artifact; do not chase it.

## SERVED-ARM SPEC (for the coordinator's window; byte-exact, zero numerics risk)

GGML_CUDA_MMVQ_IQ3S_SHARE=1 GGML_CUDA_MMVQ_IQ3XXS_SHARE=1 \
GGML_CUDA_MMVQ_Q3K_SHARE=1 GGML_CUDA_MMVQ_Q4K_SHARE=1 \
GGML_CUDA_MMVQ_Q5K_SHARE=1 GGML_CUDA_MMVQ_Q6K_SHARE=1
# GGML_CUDA_MMVQ_IQ4XS_SHARE  stays UNSET (negative, +5.8% at T=4)
Each gate independent; unsetting any reverts that type to the shipped path.

## CEILING STATEMENTS (binding term per type at T=4 on the real schedule)

- q3_K: after share 56.2 GB/s; decode-ALU removed was the binding term
  (-28.4%); residual = y-operand traffic + emulated dp4a (ISA-walled).
- q4_K: after share 64.9 GB/s; same class as q3_K (-21.6%); residual = traffic.
- q5_K: after share 78.1 GB/s; scale-walk decode was binding (-21.5%);
  residual approaching the traffic class.
- q6_K: base already 83.6 GB/s on the lm_head shape - traffic-bound; only
  -4.7% available from decode amortization.
- iq3_xxs: base 45.0 GB/s but ONE packed scale/sign word per 8 quads - the
  redundant decode is small; binding = x/y operand stream. -3.8%.
- iq3_s: base 47.3 GB/s, same packed-scale class; -2.8% real. The E-006
  decode-chain wall was measured on the rpb=1 clone; on the served schedule
  the binding term is operand traffic, not the decode chain.
- iq4_xs: NAMED NEGATIVE. Cheapest decode of the set (4 LUT-select ints, one
  scale byte), highest base bandwidth (77.8 GB/s); the 10-live-register
  share state costs more than the removed decode. Do not retry without a
  register-pressure lever.

Negative catalog respected: constant-LUT and smem-LUT decode pre-killed
(E-006 class); no LUT rungs attempted. ISA walls re-confirmed: no dot
instructions assemble on gfx900 (09-21 probe).

## Gates

- gfx900 compile: hipcc TU clean (zero warnings) + full cmake build
  (/home/chris/opt/cmake/bin/cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900,
  ggml-hip target) clean.
- Host suites: 16 built, 15 PASS + test_t3_alias_host documented exit-2
  informational (short-chain no-recycling, pre-existing outcome).
- Oracle defects: none found this cell (the W2 NaN/undersize lessons are
  baked into the instrument: full-dst memcmp, no rel-err path).

UPSTREAM-FACING NOTE: private-fork dev cell; nothing here is upstream-PR
material per AGENTS.md.
