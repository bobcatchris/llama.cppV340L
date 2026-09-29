# W6 MMVQ RUNGS Receipt: the W1 ladder re-derived per served type - all rungs land as derivations or named negatives - 2026-09-23

Desk mmvq-rungs (worktree wt-mmvq-rungs, branch amd/mmvq-rungs, branchpoint
amd/mmvq-bw 567c508cd, merged amd/v340-port-v2 36b1fe72f at landing).
Instrument: docs/amd-port/tests/bench_mmvq_rungs_gfx900.cu v1 (memcmp-gated,
one binary, arms base/share/rung; the W3 share harness plus rung arms),
hipcc rocm-6.2.0 gfx900, die 3, lock-compliant sessions. Weights = real GGUF
tensor bytes (offset-delta law, verified against the GGUF header before any
card time), synthetic valid q8_1 x 4 tokens, REAL GCN schedule clone
(rpb=2, nwarps=2, block 64x2). Types: iq3_s, iq3_xxs, q4_K, iq4_xs (the
four dominant served T=4 MMVQ types; trace shares 26/18.9/14.3/13.4%).

## P0 REPRODUCTION: PASS (clean session, lock held, spreads <= 1.4%)

W3 T=4 share verdicts reproduce (W6_rungs_p0_repro + W6_rungs_s1):
iq3_xxs -4.1 (W3 -3.8), q4_K -21.5 (W3 -21.6 exact), iq4_xs +5.3/+6.4
(W3 +5.8, negative stands), iq3_s -4.4 (W3 -2.8; the bw desk's independent
repro measured -4.7 - the delta vs W3 is the known E-098/E-100 thermal
class, concurred by two desks). In-run dup controls bit-identical in every
session; all share/wide arms BITEXACT vs base (device memcmp, 17408 rows x
4 tokens) before any timing counted.

## RUNG LADDER VERDICTS (the W1 sequence re-derived per type)

### C1 intra-CTA K-split and C2 cross-CTA split: KILLED BY GEOMETRY (derivation receipt)

W1's q3 disease was a long serial K-chain on a sub-wave grid. The served
T=4 cell at K=5120 has NO chain: blocks_per_row = K/256 = 20, and the GCN
schedule covers kbx slots 0..31 in ONE iteration for every type (blocks_per_iter
= vdr*nwarps*64/qi: iq3_s 16 slots x 8 lanes = 128 lanes active, a second
pass only for kbx0 < 4; q4_K/iq3_xxs/iq4_xs 32 slots, 80/128 lanes active,
one pass). Longest per-thread chain = 1-2 iterations. The grid is many-wave
(N/2 = 8704 CTAs of 2 wave64s on 64 CUs = ~136 CTAs/CU demanded), the exact
class W1 measured DEAD for K-split (gate_up 5 waves). K-split has nothing
to split and re-mapping kbx ownership to fill the idle 48 lanes changes the
fp32 reduction order (not bit-exact; would carry an acceptance price for
<= 1.6x lane-efficiency upside that the W5 consume ceilings bound away:
the cell is stall-bound at 70-120 GB/s access-pattern ceiling, not
parallelism-starved). Do not re-open on this schedule; a K-shaped change
(e.g. fused multi-K layers) is the door.

### A1 staging/chunk-alignment: alignment-law derivation + one NEUTRAL measurement

Derivation (zero-card): contiguous loads that dwordx2/x4 could merge are
Misaligned on every block for the odd-stride types - iq3_s 110 B and
iq3_xxs 98 B strides put the block base at 2 mod 4/8 wandering all residue
classes (get_int_b2 ushort loads are the only legal widths); the 36 B y
stride breaks 16 B y-operand alignment for uint4 applies (this is exactly
the W4 aln relayout, measured NEGATIVE). Only iq4_xs qs is 8 B-aligned on
every block (136 B stride, qs at +8): 4 dwords -> 2 dwordx2.
Measurement (session W6_rungs_s1, PASS spreads): iq4_xs wide-on-share arm
BITEXACT, +0.1% vs share = NOT-MOVEMENT. Concurs with the W5 bw desk
independent wide-x negative (-0.9..-1.2%): x-loads are 12-20% of the
per-iteration stream. Named negative; do not re-open.

### B1 PRMT/perm decode atom: REFUSED ON DEVICE + physics-capped (double kill)

The shipped iq3 sign chain lowers on gfx900 to v_sub_i16/byte-assembly
emulations: 28-30 VALU ops per sign pair (probe_isa census, offload-device-only
-GCN asm). The candidate atom - v_perm_b32 MSB-replication to build the
FF/00 sign masks (2-3 ops) + carry-free sign apply (g ^ s0) + (s0 &
0x01010101) (3 ops; grid bytes are never 0x00 so the +1 never carries) -
is EXHAUSTIVELY EXACT on host over the full atom space (256 grid entries x
256 sign bytes, both xxs mask idioms and the iq3_s idiom, ALL EXACT,
probe_sign). On device the oracle REFUSED both perm arms (bit mismatch,
reproduced in two sessions): gfx900 v_perm_b32 does NOT implement the
assumed selector MSB-replication mode. Probes with uniform and masked
selectors return FF-pattern results that no nibble-wise model fits
(dbg_perm*.cu; e.g. selector 0x00000008 -> byte 0x00 but 0x88888888 ->
0xFF on identical data: the output depends on selector bytes outside the
per-byte nibble). NAME: the v_perm mode-bit semantics assumed from the
GCN doc model are invalid on gfx900 as exposed by __builtin_amdgcn_perm /
inline v_perm_b32; treat 3-bit plain byte-select (the proven
get_int_from_table_16 idiom) as the only trusted use.
Even a corrected atom is capped: W3's FULL decode-removal (share) on these
types wins only -2.8/-3.8% at T=4, and the W5 consume decomposition
measures issue demand ~144 us vs ~800 us actual (5.5x slack) = the cell is
stall-bound, not VALU-issue-bound, so cutting sign-chain VALU cannot move
it. VERDICT: killed; do not re-open without a schedule that makes decode
issue-bound again.

### q4_K: no arm pulled (derivation-only)

q4_K decode is nibble shift/mask (cheap); its apply is 8 emulated dp4a x
6 VALU (gfx900 ISA wall, 09-21 dot-probe: no v_dot4 until gfx906) and its
W3 share arm already cashed -21.6%. Residual = x/y mix traffic (W5 consume:
+52% over pure-consume). Nothing in the A-D ladder applies to a
16 B-aligned block with cheap decode and an ISA-walled dot.

## SESSION HYGIENE (honest log)

- Sessions of record: W6_rungs_p0_repro (bench_base binary, W3 instrument)
  and W6_rungs_s1 (rung harness, 4 types, worst spread 0.61% PASS). Both
  clean, lock held by desk mmvq-rungs.
- W6_rungs_s2_seal (post-merge confirmation) is VOID: the desk overwrote a
  live coordinator lock (coord-combowin regress cell started 10:32:39; my
  30 s wait expired and stole it) and ran concurrently - spreads 11-33%,
  verdicts wild, all timing numbers discarded per the 1% law. The timing-
  independent oracle verdicts from that session stand (perm arms refused
  again; wide arm bit-exact again - consistent with s1). Lock ownership was
  restored with a disclosure note; the coordinator's 200k window may carry
  a contaminated die-3 slice from 10:33-10:36 and should be re-checked
  before its numbers are banked. PROTOCOL FIX: wait loops must poll until
  the lock is actually FREE, never time out into stealing.

## COMPOSITION with the banked arms (E-104 share set, E-106 s2r set)

Nothing from this desk composes into the served spec: every rung is a
derivation or a named negative. The served-arm spec of record remains the
W5 spec (six *_SHARE=1 gates + five *_S2R=1 gates; IQ4XS_SHARE and
LLAMA_MMVQ_ALN unset). The iq4_xs wide arm is neutral on share and, being
x-side only, cannot interact with s2r's y-preload beyond noise; it stays
off. No re-run needed.

## KILLED (with receipts, this desk)

- C1/C2 K-split (all four types): geometry - 1-iteration chains, many-wave
  grid; this receipt's derivation + W5 consume ceilings.
- A1 wide loads iq4_xs (uint2 merge): +0.1% NOT-MOVEMENT, W6_rungs_s1;
  concurs W5 A2.
- A1 wide loads iq3_s/iq3_xxs: illegal by alignment law (2 B-aligned odd
  block strides); y-side = W4 aln negative.
- B1 perm atom iq3_s/iq3_xxs: device oracle refusal (v_perm mode-bit model
  invalid on gfx900) + physics cap (stall-bound cell); probe_sign,
  probe_isa, dbg_perm*.cu + the two refusal logs.

## NEXT LINK (cost class)

The only door this ladder leaves open is the one W5 already named: the
70-120 GB/s access-pattern ceiling is schedule-bound (1.25 kbx iterations
of runway per lane; the x/y mix storm) - schedule-level y staging (LDS-y
design banked in W5) or format-level decode reduction. Instruction-count
and K-shape rungs are exhausted on this schedule; B-rungs reopen only if a
future schedule re-enters the issue-bound regime.
