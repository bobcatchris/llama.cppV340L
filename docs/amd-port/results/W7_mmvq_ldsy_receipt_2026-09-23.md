# W7 MMVQ LDS-Y Receipt: staging killed (slow when exact, unlandable otherwise) - s2r wiring defect found and fixed - 2026-09-23

Desk mmvq-ldsy (worktree wt-mmvq-lds, branch amd/mmvq-lds, base 387805bec).
Instrument: docs/amd-port/tests/bench_mmvq_real.cu - NOT a replica: it
includes ggml/src/ggml-cuda/mmvq.cu verbatim and launches the actual
mul_mat_vec_q template instances at the exact served GCN geometry (rpb=2,
nwarps=2, block 64x2, grid ceil(N/2)) on REAL GGUF tensor bytes (offset-delta
table from bench_mmvq_share_gfx900.cu, ASCII-P1M). Oracle: full dst (N x T)
device memcpy + host memcmp vs the base arm BEFORE any timing counts; a base
dup at the end of every arm list doubles as drift/determinism control.
hipcc rocm-6.2.0 gfx900, die 3 only, /tmp/campaign_gpu_boot.lock held per
session, cool-die gate (25-28 C) before sessions. This desk was built around
the E-113 lesson: the previous desk's replica harness lied; every number here
is the served template.

## P0: REAL-KERNEL BASE ANCHORS - instrument PASS, replica anchors OVERTURNED

Anchor numbers below are from the FINAL tree (pristine kernel + the s2r
wiring fix) - the only tree whose numbers are served-relative. Earlier
intermediate designs measured different absolute bases (see the
implementability finding: code shape moves these kernels by up to 2.3x);
they are instrument-development history, not anchors.

Final-tree real-kernel base vs share at T=4 (K=5120; rocprof on the served
build-hip cross-validates: base 1700.0 median vs bench 1843.1, share 664.2
vs bench 687.6 - the bench tracks the served path within the clock class):

| type   | base    | share  | share delta | share replica (W3/W5/W6) |
|--------|---------|--------|-------------|--------------------------|
| iq3_s  | 1843.1  | 687.6  | -62.7%      | -4.4/-4.7/-6.6           |
| iq3_xxs| 1923.3  | 706.2  | -63.3%      | -4.1/-3.1/-3.4           |
| q4_K   | 1108.4  | 849.3  | -23.4%      | -21.5/-22.0              |
| q5_K   | 821.1   | 639.1  | -22.2%      | -20.4                    |
| iq4_xs | 618.7   | 502.8  | -18.7%      | +5.3/+5.4                |
| q3_K   | 797.1   | 798.2  | +0.1%       | -28.4                    |
| q6_K   | 5908.3  | 5996.0 | +1.5%       | (thermal-void class)     |

T=2 (same build): iq3_s +0.6%, q4_K -12.4%, iq3_xxs +3.0%, iq4_xs -5.4%,
q5_K -15.7% (share vs base).

The replica anchor set is overturned: the served schedule pays 2.3x more for
the per-token base path than any replica showed, so the decode-once share
gates are worth up to -63% per type (iq3_s/iq3_xxs), not single digits;
q3_K's replica -28.4% is really +0.1% (its served gate does nothing);
iq4_xs share flips from +5.3% replica-negative to -18.7% real-positive (its
gate is OFF served per E-104 - re-opening that gate is the coordinator's
call with these numbers). All oracles BITEXACT, base dup controls
bit-identical.

## A1/A2: LDS-Y STAGING - BIT-EXACT BUILD MEASURED: NAMED NEGATIVE

The W5 design (52 B padded LDS records, bank-safe stride 13, runway budget
8.6 KB/CTA, ~10 syncs) was implemented behind GGML_CUDA_MMVQ_LDSY=1 for the
five s2r-class types at T=2..4, composing with the *_SHARE/*_S2R namespaces.
The cooperative store copies each staged kbx window's q8_1 y blocks into LDS
verbatim; consumers read the window through the same share_consume body as
the plain path (one compiled body, generic pointer, both call sites - see
the implementability finding below for why this is the ONLY bit-exact shape).

Occupancy at T=4 (hipFuncGetAttributes, bit-exact build): ldsy shared =
6656 B tile + 2048 B reduction = 8704 B - the W5 budget verified exactly.
numRegs: q4_K 101 (= base), q5_K 104 (= base), iq3_s 68 (vs 83),
iq3_xxs 62 (vs 64), iq4_xs 62 (vs 85). ctas_per_cu: q4_K/q5_K 4 (unchanged,
register-limited), iq3_s 6 (unchanged), iq3_xxs 8 -> 7, iq4_xs 4 -> 7.

Formal session W7_ldsy_a2_formal_2026-09-23.txt (lock held, cool die, every
arm ORACLE BITEXACT before its timing). Medians, T=4:

| type   | base  | share | s2r   | ldsy:4 | ldsys2r | best ldsy run |
|--------|-------|-------|-------|--------|---------|---------------|
| iq3_s  | 792.8 | 691.8 | 675.1 | 1973.8 | 1980.0  | 1714.0 (RUN8) |
| q4_K   | 1112.7| 833.4 | 831.2 | 2168.3 | 2153.0  | 1816.1 (RUN8) |
| iq3_xxs| 800.1 | 781.8 | 732.5 | 1791.3 | 1791.2  | -             |
| iq4_xs | 570.9 | 813.4 | 600.6 | 1439.5 | 1452.8  | -             |
| q5_K   | 808.7 | 626.0 | 627.0 | 1608.0 | 1592.2  | -             |

ldsy vs its own-build base: iq3_s +149.0% (best run +116.2%), q4_K +94.8%
(best +63.4%), iq3_xxs +123.9%, iq4_xs +152.1%, q5_K +98.9%. T=2:
+97.0..+198.0% on all five. Spreads on the ldsy arms <= 1.35%.

Served-relative accounting: that build's own base branch proved to be a
codegen outlier (its default path differs from served; see the
implementability finding - the restructured base branch ran up to 2.3x fast
on the 2 B-aligned types). Against the FINAL tree's served-class base, the
staged arm's absolute cost (~1974 us for iq3_s) is +7.1%. The candidate is
negative in every accounting: -0% win nowhere, +7% served-relative at best,
+63..+198% inside its own build.

VERDICT: LDS-y staging killed as a named negative on the real kernel, all
five types, both T. Nothing wires into the served spec. Mechanism: the y
stream is shared by every CTA (8704 CTAs read the same ~23 KB of q8_1 per
token), so the direct path's "scattered" 36 B loads are already L1-resident;
staging re-reads those bytes from global an extra time (the cooperative
load), adds an LDS store+load round trip plus two full-CTA barriers, and
replaces nothing that L1 was not already providing for free. Sub-pass runway
(RUN=4, the W5 budget point) additionally serializes lane groups (32/128
lanes consuming per stage on the iq3 class); full-pass RUN=8/16 removes that
and still loses by 63-165%.

## A3: NO WINNERS - RUNWAY SWEEP DOCUMENTED, EXTENSION NOT TRIGGERED

No type won, so the q3_K/q6_K extension was not performed (ladder:
winners-only). Runway sweep (T=4, medians): iq3_s RUN4 1973.8 / RUN8 1714.0 /
RUN16 2096.3 (best +116.2% vs base); q4_K RUN4 2168.3 / RUN8 1816.1 (best
+63.4%). The negative is schedule-deep, not parameter-deep.

## THE IMPLEMENTABILITY FINDING (why the candidate is reverted, not landed)

After the negative, the desk attempted the E-106-style landing (candidate
code default-OFF, reproducible via env). Three shapes were built and tested
against the A4 byte-identity oracle; each fails a law:

1. Separate LDS-pointer consume (staged type blocks calling the unchanged
   apply functions with an LDS base, plain path pristine): NOT BIT-EXACT.
   The gfx900 backend forms the apply-tail fused multiply-adds differently
   when the operand base is LDS vs global: 5030/69632 outputs differ in the
   low bits (max rel 1.5e-4). Scoping `#pragma clang fp contract(off)` to the
   staged path only moves the failure to the 694 elements where the PLAIN
   path contracts and staged now does not. Global -ffp-contract=off makes all
   arms mutually bit-exact, proving contraction is the entire mechanism - but
   a global switch changes served bits (rejected).
2. Shared-body consume (one generic-pointer lambda called with the global y
   or the LDS tile; the plain and staged paths execute literally identical
   instructions): BIT-EXACT (all arms, all types, both T - the formal session
   above was measured on this build). But the generic-pointer body shifts
   address-space inference for the whole kernel: the served default path's
   base-branch outputs change (default-env oracle dump vs the base tree:
   q3_K T=1 485/512 floats differ) and the plain base arm slows 2.3x on the
   2 B-aligned types (iq3_s 792.8 -> 1848.9, iq3_xxs 800.1 -> 1919.9) at
   identical occupancy (77 regs, 6 CTAs). Violates the default-path law.
3. 13-dword padded records with dedicated apply_lds twins (the literal W5
   record design): same contraction failure as (1) with its own store-index
   defect history; superseded.

CONCLUSION: on gfx900/hipcc-6.2.0 an LDS-consuming share path cannot be both
bit-exact against the served kernel and free of served-codegen perturbation.
Combined with the +63..+198% perf negative on the bit-exact build, the
candidate is dead twice over. LANDING: mmvq.cu reverted to the pristine
387805bec source with exactly one change kept (the s2r wiring fix, 22 lines,
below); the LDSY kernel code, gate and template parameters are removed from
the tree; the instrument (bench_mmvq_real.cu) is kept reduced to the
base/share/s2r arms. The full LDSY implementation remains reviewable in this
branch's history (WIP commits).

## FINDINGS OF RECORD (the desk's banked output beyond the negative)

1. SERVED DEFECT FIXED (kept in tree): the s2r host flag never reached the
   launch. mul_mat_vec_q_switch_type hardcoded `false` for its s2r parameter
   at all 21 type cases (introduced by 079951eb7, the bw desk's own commit;
   carried through the 36b1fe72f merge). Every *_S2R=1 gate was a no-op
   served: the E-106/E-109 "served s2r" numbers and E-113's "s2r exactly
   neutral at 10k" verdict measured the REGRESS arm, not s2r (10k 23.21 vs
   23.21 to the second decimal - bit-identical because it was the same code).
   Re-verified on the served build during A4: base-tree share+s2r dump ==
   base-tree default dump, byte-identical. Fix: all cases pass s2r through
   (22 lines). Default path unchanged (the flag is false unless the env
   gates fire).
2. REAL-KERNEL s2r deltas (first honest measurement, wiring fixed):
   T=4 vs share: iq3_xxs -6.3%, iq3_s -2.9%, q4_K -0.3%, q5_K +0.2%,
   iq4_xs -26.2% vs its share arm BUT +5.2% vs its served base (iq4_xs share
   is OFF served; s2r alone is WORSE than its base). T=2 vs share: -2.3 to
   +1.3% (nothing). The W5 replica wire set (-6.6..-7.7%) does not hold on
   the real kernel; only iq3_xxs carries a real s2r increment at T=4 and the
   set is stale at T=2. Coordinator decision needed before any s2r promotion
   attempt; an interleaved multi-rep served A/B remains mandatory.
3. REAL-KERNEL share table (P0 above) supersedes the replica anchor set for
   the T=4 cell. Served implication: the q3_K share gate (E-104, ON) is
   likely worth ~0 of its replica-claimed -28%; the E-104 20.01 result must
   have been carried by the other types. Re-litigation is the coordinator's
   call; the instrument is now in-tree.

## A4: BUILD GATES (final tree)

- mmvq.cu TU standalone: clean (BUILD-EXIT:0, no warnings).
- Full ggml-hip build of this tree (build-gfx900, canonical flags): PASS.
- Diff vs 387805bec: 22 lines, all inside mul_mat_vec_q_switch_type
  (the s2r threading fix). No kernel-body changes of any kind.
- Default-path byte-identity: default-env real-path oracle dumps (8 types x
  T=1..4 x q81-cache) this tree vs the served build-hip tree: byte-identical.
- Share-spec dumps: byte-identical vs the served tree.
- s2r engagement: served tree share+s2r dump == served tree default dump
  (the defect, confirmed); this tree share+s2r dump == this tree default dump
  (the fix engages AND s2r is bit-exact on the real path).
- Details: results/W7_ldsy_a4_checks_2026-09-23.txt.

## KILLED / NAMED NEGATIVES (this desk)

- LDS-y staging (any shape): +63..+198% vs base on the bit-exact build, all
  five s2r-class types, T=2..4, runways 4/8/16, both consume forms. The y
  stream is L1-resident by construction (CTA-shared); there is no coalescing
  dividend to collect. Re-open only on a schedule where y is NOT CTA-shared
  (fused multi-K layers reading disjoint y) or an arch class where scattered
  L1 loads cost real transactions.
- LDS-y staging as a landable default-OFF arm: not implementable within the
  bit-exact law on gfx900/hipcc-6.2.0 (the implementability finding).
- 13-dword padded LDS records (the literal W5 record design): superseded by
  verbatim 36 B records; killed with the parent.

## NEXT LINK (cost class)

The 70-120 GB/s access-pattern ceiling survives its last scheduled lever:
staging (this desk), wide-x (W5/W6), K-split (W6), perm atoms (W6) are all
dead on the real schedule. What remains for the MMVQ cell: format-level
weight-bit or decode reduction (owner's bit-exact rule guards) and schedule
fusions that change what y IS - both outside this desk's scope. The
immediately actionable outputs of this desk: the s2r wiring fix and the
corrected real-kernel share/s2r tables for the coordinator's served window.

UPSTREAM-FACING NOTE: private-fork dev cell; nothing here is upstream-PR
material per AGENTS.md.

Sessions of record: W7_ldsy_p0_2026-09-23.txt, W7_ldsy_a2_formal_2026-09-23.txt,
W7_ldsy_a4_checks_2026-09-23.txt.
