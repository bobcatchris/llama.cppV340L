# W2 Decode-Atom Receipt: iq3_s MMVQ atom redesign on gfx900 - 2026-09-21

Instrument: docs/amd-port/tests/bench_mmvq_gfx900.cu v2 (oracle-gated, 8 arms in
one binary), die 3, 3 interleaved reps x 200 timed iters per arm per session.
Weights = REAL blk.21.ffn_up bytes (38,297,600 B, offset-delta law), synthetic
valid q8_1 (raw-byte built), schedule cloned from mul_mat_vec_q (GCN, T=1,
wave64, NW2, kqs=2*(tid&7), 16 kbx slots).

## ORACLE (mandatory gate, now passing)

Host C++ mirror of the shipped vec_dot_iq3_s_q8_1 over the same weight bytes +
same synthetic x; every arm must match rows 0..63 at rel err < 1e-4 before its
timing counts. All 8 arms pass at rel err 1.7e-06 (identical across arms).
Getting the oracle green surfaced three traps, each now guarded in the bench:

1. iq3s_grid is __device__-only under GGML_COMMON_IMPL_HIP. Host code reading
   the symbol silently read garbage; hipMemcpyFromSymbol aborts on internal
   static symbols ("Cannot create GlobalVar Obj"). Fix: 512-thread pull kernel
   D2H, pinned-value check [0]=0x01010101, [322]=0x07070b0b.
2. HIP host-side ggml_half fields go through __half: `ds.x =
   ggml_fp32_to_fp16(1.0f)` VALUE-converts (stored fp16(15360) = 0x7380, not
   0x3C00). The q8_1 buffer is now built as raw bytes (ds.x@0, ds.y@2, qs@4,
   stride 36, verified device-echo). Note the compiled block_q8_1 has ds FIRST.
3. Table facts (task question answered): iq3s_grid has 512 entries
   (ggml-common.h:1042, uint32_t); the kernel indexes with a 9th bit from qh:
   idx9 = qs_byte | ((qh >> l) & 1) << 8.

## CLONE CORRECTIONS (supersede the E-005/E-006 instrument claims)

The v1 bench deviated from the shipped math in two places:
- u1 = get_int_b4(..., l0 + 4) instead of l0 + 1 (reads ints {4,6,8,10} of the
  q8 block: duplicates ints 4,6 and spills into ds/next block);
- y pointer never offset by kby = kbx*(qk/QK8_1) (weights of block kbx were
  paired with x of blocks 0..7; mmvq.cu:573,724 pass &y[kby]).

PERF-NEUTRAL CLAIM: FALSIFIED by same-era A/B (old binary rebuilt from git
29db8633e, run back-to-back with the fixed bench): old 360.0 us = 106.6 GB/s
vs fixed 424.6 us = 90.4 GB/s. The missing kby offset made all 8 x-int loads
LOOP-INVARIANT (old bench read the same y blocks for every kbx), so the
compiler hoisted them out of the kbx loop: an accidentally lighter kernel.
Consequences:
- The old 105.2 GB/s "shipped clone" figure was an artifact; the true shipped-
  atom number of record on today's package-load era is 90.4-90.9 GB/s
  (422-425 us), stable across 3 sessions (+-0.8 percent).
- All E-005/E-006-era ratios remain valid RELATIVELY within their own sessions;
  absolute re-anchor is the fixed clone above.

## RESULTS (final session; oracle PASS on every arm; control drift < 1 percent)

| arm        | us/call | GB/s  | vs base | verdict                                            |
|------------|---------|-------|---------|----------------------------------------------------|
| base       | 422.1   | 90.9  | 1.00x   | fixed verbatim clone = number of record            |
| gmem       | 423.4   | 90.6  | 1.00x   | NEUTRAL - constant vs global LUT class irrelevant  |
| pip        | 440.5   | 87.1  | 1.04x   | NEGATIVE - intra-call reorder + 2 int accumulators |
| sgn        | 929.8   | 41.3  | 2.20x   | STRONG NEGATIVE - 32 KB sign-folded LUT            |
| sgnpip     | 958.9   | 40.0  | 2.27x   | NEGATIVE                                           |
| sexp       | 488.7   | 78.5  | 1.16x   | NEGATIVE - 2 KB pre-expanded sign-mask table       |
| dp4a       | 508.9   | 75.4  | 1.21x   | NEGATIVE - hand v_bfe_i32+v_mad_i32_i24 (asm)      |
| sexpdp     | 490.5   | 78.2  | 1.16x   | NEGATIVE                                           |
(earlier session: dp4a with volatile asm = 459.7 us = 83.5 GB/s, 1.08x - still
negative; non-volatile asm is WORSE. All bit-exact integer math; no arm changes
numerics, so no acceptance price anywhere in this receipt.)

## MECHANISM READINGS (each negative carries one)

1. sgn (32 KB): the complete 4-sign-bit fold needs 2^13 x 4 B = 32 KB - twice
   the gfx900 16 KB vector L1. The lookups fall to L2 and the dependent per-
   lane chain eats full L2 latency: 2.2x. Partial folds (8/16 KB) leave the
   expensive __vsub4 sign application in place - net zero by instruction
   budget. Sign-fold-in-LUT is STRUCTURALLY DEAD on gfx900.
2. sexp: replacing 8 per-code vcmpne4 chains (~30 vops) with 4 additional
   divergent table lookups LOSES: divergent-lookup replay costs more than ALU
   on this cell. Confirms the LUT-replay term dominates decode ALU.
3. hand-dp4a: 12-vop bfe+mad loses to the 16-vop library emulation under both
   volatile and non-volatile asm (LLVM schedules the library's 16-bit SDWA
   form better; my serial 4-deep mad chain does not overlap).
4. pip: the compiler already schedules the base loop well; manual reordering
   plus accumulator splitting costs registers and wins nothing (-4 percent).
5. Static census (amdgcn -S): base loop body = 314 v-ops/call (dp4a emu ~130,
   sign chain ~35, index+address ~90, loads 16); sgn loop is far leaner and
   still 2.2x slower -> the wall is NOT static instruction count alone but
   divergent per-lane LUT access (8 lookups per 32 weights, one address per
   lane = replay serialization) plus dp4a emulation.

## VERDICT (stop condition reached)

The shipped iq3_s decode atom is at its practical gfx900 ceiling: every
redesign arm measured negative; best measured = the shipped atom itself at
90.9 GB/s (today-era). The 130-150 GB/s target is ISA-walled:
- no v_dot4_i32_i8 on gfx900 (dp4a emulation ~130 vops is irreducible for
  bit-exact integer math);
- the fp16 V_DOT2_F32_F16 path is blocked: q8 int8 -> half conversion on the
  x side costs more ALU than the 16 dot2 instructions save (no linear bit
  mapping from two's-complement byte to half; a LUT would add 32 more
  divergent lookups), and changes numerics besides;
- the 8 divergent 9-bit LUT reads per 32 weights are irreducible for this
  format (every 4-weight group has its own code; lane-to-code remapping does
  not reduce distinct addresses per row).
Cross-check ablation (E-006) ladder stands: loads-only 245 GB/s, +LUT 178,
full decode ~91 (corrected instrument).

## NAMED NEXT LEVERS (outside the atom; for the final report)

- T4 quantize_q8_1 elimination (7 percent of decode step): also the only path
  that changes the x-side format - if the producer emitted fp16x2 activations,
  a V_DOT2_F32_F16 atom becomes both cheap and exact for integer-valued halfs;
  that is an acceptance-priced (numerics-identical but layout-changing) door.
- T3 tiny-tensor tax (~12-15 percent of step): schedule/launch class, not
  atom class.
Recommendation: T4 first (shares the atom's x-format constraint, single
producer to change), T3 second.

## REPRODUCE

  cd wt-decode-atom
  /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP \
    -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
    docs/amd-port/tests/bench_mmvq_gfx900.cu -o /tmp/bench_mmvq \
    -L /media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin -lggml-base
  HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin \
    /tmp/bench_mmvq /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf 200 3

UPSTREAM-FACING NOTE: private-fork dev cell; nothing here is upstream-PR
material per AGENTS.md.
