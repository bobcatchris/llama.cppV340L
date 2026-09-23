# W5 MMVQ BW Receipt: weight-stream rungs A1-A4 - s2r banked, ceilings named - 2026-09-23

Instrument: docs/amd-port/tests/bench_mmvq_share_gfx900.cu v2/v4 (memcmp-gated,
one binary, arms base/share/aln/wide/cfull/cx/cy/s2r), tree amd/mmvq-bw
(worktree wt-mmvq-bw, branchpoint = amd/mmvq-kernel2 567c508cd incl.
v340-port-v2 merge 95f304d04 + banked aln arm), hipcc rocm-6.2.0 gfx900,
die 3, lock-compliant sessions (/tmp/campaign_gpu_boot.lock, desk mmvq-bw).
Weights = real GGUF tensor bytes (offset-delta law), synthetic valid q8_1
x T tokens, REAL GCN schedule clone (rpb=2, nwarps=2, block 64x2).

## P0 REPRODUCTION: PASS

W3 anchors reproduce (session W5_bw_p0_repro_2026-09-23.txt): T=4 share
verdicts iq3_xxs -3.4 / q4_K -22.0 (PASS 0.64%) / q5_K -20.4 / q3_K -28.4
(exact) / iq3_s -4.7 / q6_K -4.4 / iq4_xs +5.4 (negative stands); iq3_s
t4_base 840.4 vs 816.5 banked = +2.9% (thermal band, E-098/E-100 class).
aln T=4 negative reproduced on every type (+7.8..+375% vs share) - sibling
desk's W4 verdict confirmed on this stack.

## A1 STAGING/CHUNK-ALIGNMENT AUDIT (zero-card, per-type byte-laws)

Block strides vs the 64 B memory sector, x = weight stream, y = q8_1 stream
(36 B blocks; per-kbx group of 8 = 288 B, always 16 B-aligned):

| type | bs B | bs mod 16 | base align | x block bytes read/lane | x loads/lane/kbx | x sectors/warp-iter (useful 13.75) |
|------|------|-----------|-----------|--------------------------|------------------|---------------------|
| iq3_s  | 110 | 14 | 2 B | 14 B (qs 8 + signs 4 + qh 1 + scales 1) | 8 (6 ushort + 2 byte) | ~21 (1.53x) |
| iq3_xxs| 98  | 2  | 2 B | 12 B | 6 | ~21 (1.53x) |
| q3_K   | 110 | 14 | 2 B | ~9 B  | ~5 | ~21 |
| q4_K   | 144 | 0  | 16 B | 12 B (2 dword + 2 ushort + dm) | 4-5 | ~18 (1.2x) |
| q5_K   | 176 | 0  | 16 B | ~14 B | 5-6 | ~18 |
| q6_K   | 210 | 2  | 2 B | ~10 B | ~5 | ~22 |
| iq4_xs | 136 | 8  | 8 B  | 19 B (16 contig + scales + d) | 4 dword | ~16 (1.15x) |

- 2 B-aligned 110/98/210 B strides wander all residue classes mod 64: every
  block straddles 2-3 lines; sector amplification 1.5x class on the IQ3 pair.
- The aln (48 B y repack) result + this table name the y-operand stream as
  already optimal in its 36 B layout: per-kbx groups are 16 B-aligned.
- Load-instruction amplification: iq3_s x-side 8 scalar loads per 14 B
  (dword-ideal = 1 per 4-16 B); y-side = 8 dword loads per 32 B block,
  block-per-lane at 36 B stride -> a wave32 load touches ~36 distinct lines
  (the L1-transaction storm).

## A2 WIDE X-OPERAND LOADS (arm 6): NOT-MOVEMENT - named negative

funnel-shift trio (iq3_s/iq3_xxs, 3 dwords vs 4 ushorts per 8 B window) and
uint2 (iq4_xs, 2 vs 4 dwords): T=4 verdicts iq3_s -0.9% vs base / iq3_xxs
-0.9% / iq4_xs -1.2% - all < 2% NOT-MOVEMENT, per the noise law. Mechanism
confirmed by the consume decomposition: x-operand loads are only 12-20% of
per-iteration load instructions; widening them cannot move the cell.
gfx900 dwordx2 does NOT decompose (no aln-class blowup), but there is
nothing for it to win on the x side. Session W5_bw_v2.

## A4 PURE-CONSUME CEILINGS (arms 3/4/5, T=4): the stop-condition instrument

Same loads as share, xor-consume instead of decode+dp4a:

| type | share us | cfull us (GB/s) | cx (x only) | cy (y only) | decode+dp4a tax |
|------|----------|-----------------|-------------|-------------|-----------------|
| iq3_s  | 801.3  | 488.1 (79)  | 364.4 (106) | 346.2 (112) | +69%  |
| iq3_xxs| ~750   | 455.5 (76)  | 312.7 (110) | 327.4 (105) | +60%  |
| q4_K   | 775.8  | 511.3 (99)  | 407.5 (124) | 295.7 (171) | +52%  |
| q5_K   | 576.3  | 397.2 (109) | 320.3 (136) | 199.2 (218) | +45%  |
| q3_K   | 699.8  | 546.2 (71)  | 458.7 (84)  | 307.5 (126) | +29%  |
| iq4_xs | 647.1  | 487.1 (98)  | 380.7 (125) | 333.1 (143) | +33%  |
| q6_K   | 6284.7 | 4530.1 (120)| 4025.0(135) | 2515.9 (217)| +39%  |

Named ceilings: (1) the ACCESS-PATTERN ceiling itself is 70-120 GB/s - even
zero-compute versions of the exact schedule never approach the 327 GB/s
sequential-read figure; (2) x/y mix costs +41..100% over either stream alone
(L1 line-revisit storm: cfull > max(cx, cy)); (3) decode+dp4a adds another
+29..69% (dp4a = 4x v_mul_i32_i24_sdwa + 2x v_add3 = 6 VALU, SASS-counted;
kernel text 1446 inst/iteration, 61 VGPR, issue demand ~144 us vs 800 us
measured = STALL-BOUND, not issue- or DRAM-bound). Binding term: latency/
stall on scattered 2-4 B operand streams with only 1.25 kbx iterations of
runway per lane.

## A3 ROW-SHARED Y LOADS (s2r, arm 7): BANKED WIN - bit-exact, all 7 types

The rpb=2 schedule makes both rows' apply read the SAME y words; the
compiler does not CSE them. preload once per (token, block, lane) + apply_u
per row. Defect history: first build read ds at word 8 (upstream qs-first
layout assumption); this tree's block_q8_1 is ds-FIRST - oracle caught NaN
immediately, fixed to word 0 (N-row: layout assumptions must be tree-checked).

T=4 verdicts (sessions of record; s2r arm spread <= 1.05% everywhere):

| type | t4 base | t4 share | t4 s2r | s2r vs share (served increment) | verdict |
|------|---------|----------|--------|-------------------------------|---------|
| iq3_s  | 833.3 (ded) | 793.9 | 741.2 | **-6.6%** | WIRE |
| q4_K   | 993.9 (v4, PASS 0.97%) | 778.4 | 729.7 | **-6.2%** | WIRE |
| q5_K   | 723.9 (v4) | 579.1 | 548.1 | **-5.3%** | WIRE |
| iq3_xxs| 764.0 (v4) | 733.3 | 711.5 | **-3.1%** | WIRE |
| iq4_xs | 614.5 (v4) | (share OFF +5.7) | 567.4 | **-7.7% vs its served BASE** | WIRE |
| q3_K   | 954.5 (ded) | 693.6 | 691.8 | -0.3% NOT-MOVEMENT | OFF |
| q6_K   | thermal-void sessions; s2r = share class (+3.5..5%) | | | no increment | OFF |

iq3_s T=2/T=3 (dedicated session): s2r 500.2 / 661.6 = -6.1% / -6.4% vs
share - the increment spans the whole verify band. All s2r arms BITEXACT vs
base at T=2/3/4 on all types (device memcmp, N rows x T tokens) = zero
numerics risk, no acceptance exposure.

## SERVED-ARM SPEC (for the coordinator window)

GGML_CUDA_MMVQ_IQ3S_S2R=1 GGML_CUDA_MMVQ_IQ3XXS_S2R=1 \
GGML_CUDA_MMVQ_Q4K_S2R=1 GGML_CUDA_MMVQ_Q5K_S2R=1 GGML_CUDA_MMVQ_IQ4XS_S2R=1
# stacked ON TOP of the six existing *_SHARE=1 gates (Q3K/Q6K shares stay
# ON per E-104; their s2r stays OFF per this receipt). IQ4XS: set S2R only,
# do NOT set IQ4XS_SHARE. LLAMA_MMVQ_ALN must stay UNSET (s2r reads the
# legacy 36 B layout; the top-level gate hard-excludes aln).

Served projection (static-count, trace shares, candidate only): iq3_s 1528.4
x -6.6% = -100.9 ms; iq4_xs 789.0 x -7.7% = -60.8; q4_K 837.2 x -6.2% (on
the post-share kernel) = -47.6; iq3_xxs 1112.4 x -3.1% = -34.5; q5_K
(post-share 154.4 x 0.785 = 121.2) x -5.3% = -6.4. Total ~ -250 ms of
5873 ms MMVQ = -4.3% MMVQ on top of E-104's -6.3% -> ~ -3% of traced kernel
time, decode t/s upper bound ~+3% at 10k class.

## KILLED / NAMED NEGATIVES (this desk)

- A2 wide-x (funnel trio, uint2): NOT-MOVEMENT (<2%), x-loads are 12-20% of
  the stream. Do not re-open without a schedule that makes x dominant.
- q3_K/q6_K s2r: no increment over their share arms (share already cashed
  the same win).
- Session hygiene: v3 full-battery session went bimodal late (q3_K 737->1020,
  iq3_s 780->1607 - clock/power confound, whole-session VOID); verdicts
  re-grounded in dedicated per-type sessions. The 1% law is enforceable only
  per-config on dedicated runs for the late, hot types.

## NEXT LINKS (cost class)

- cfull ceiling (70-120 GB/s) is the wall: the lever class is schedule
  restructuring (more kbx runway per lane, LDS staging of the y stream with
  cooperative contiguous loads, C-rung) - a build-heavy cell, expected to
  recover the x/y-mix tax, NOT the decode tax.
- decode+dp4a tax (+29..69%) is B-rung territory: dp4a emulation is already
  6 VALU (SASS-optimal per-op); only format-level decode reduction helps.
- LDS-y staging arm: designed (52 B pad, bank-conflict-free stride 13, ~10
  syncs/CTA), not built - banked as the named next link with its occupancy
  budget (7 CTAs/CU at 8.6 KB LDS vs 8 resident by regs).

UPSTREAM-FACING NOTE: private-fork dev cell; nothing here is upstream-PR
material per AGENTS.md.
