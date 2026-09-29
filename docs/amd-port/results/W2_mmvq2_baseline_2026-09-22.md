# W2 mmvq2 Baseline Receipt: relaunch reproduction - 2026-09-22

Instrument: docs/amd-port/tests/bench_mmvq_gfx900.cu v2 (oracle-gated, 8 arms,
one binary), tree amd/mmvq-kernel2 @ 075d1501b (worktree wt-mmvq-kernel2,
branchpoint = e7dbc7785 + E-092 docs), hipcc rocm-6.2.0 --offload-arch=gfx900,
ggml-base 0.15.3 from build-hip. Die 3, lock-compliant session
(/tmp/campaign_gpu_boot.lock, desk mmvq-kernel2, 75 s settle, released).

## P0 REPRODUCTION GATE: PASS

Banked baseline (W2_decode_atom_2026-09-21, 3 sessions +-0.8%):
base 422.1 us/call = 90.9 GB/s. Today: 421.1 us/call = 91.1 GB/s, delta -0.24%.
Per-arm agreement vs the 09-21 final session, all within +-1.3%:

| arm    | today us/call | today GB/s | vs base | 09-21 us/call | delta |
|--------|---------------|------------|---------|---------------|-------|
| base   | 421.1         | 91.1       | 1.00x   | 422.1         | -0.2% |
| gmem   | 425.0         | 90.3       | 1.01x   | 423.4         | +0.4% |
| pip    | 440.3         | 87.2       | 1.05x   | 440.5         | -0.0% |
| sgn    | 928.4         | 41.3       | 2.20x   | 929.8         | -0.2% |
| sgnpip | 946.8         | 40.5       | 2.25x   | 958.9         | -1.3% |
| sexp   | 489.8         | 78.3       | 1.16x   | 488.7         | +0.2% |
| dp4a   | 513.1         | 74.8       | 1.22x   | 508.9         | +0.8% |
| sexpdp | 490.4         | 78.3       | 1.16x   | 490.5         | -0.0% |

ORACLE: all 8 arms PASS at rel err 1.7e-06 (64 rows, gate 1e-4). Control
(base) rep spread today 415.8..422.5 = 1.6% peak-to-peak - marginal vs the 1%
session-void law; median agreement with the banked number is 0.24%, inside the
+-2% P0 gate, so the session STANDS for baseline purposes. The 1% law is
enforced on all subsequent A/B verdict sessions (any winning arm will be
re-judged in a fresh interleaved session against an in-run base control).

## STATE OF THE CELL (from the banked receipts, unchanged by today's rerun)

- Shape: blk.21.ffn_up.weight [5120,17408] iq3_s, 38,297,600 B (offset-delta
  law), T=1 GCN schedule (NW2, kqs=2*(tid&7), 16 kbx slots).
- Byte floor 208.8 us at the 183.8 GB/s copy figure (sequential-read ceiling
  is higher: consume_seq measured 327.5 GB/s).
- CLOSED: T=1 schedule class - 7 banked negatives (NW4, NW8, NW1 r4/r8,
  multi-row r2/r4/r8, balanced kbx, LDS-LUT, ILP2 unroll).
- CLOSED: decode-atom class - 8 oracle-gated banked negatives (gmem, pip, sgn,
  sgnpip, sexp, dp4a, sexpdp), all confirmed again today.
- Ablation ladder: loads-only 245.5 GB/s, +constant-LUT 177.6 (-28%), full
  decode 90.4-91.1 (-41%). Number of record for the shipped atom on this cell:
  90.9-91.1 GB/s (421-423 us).

## LIVE DOORS FOR THE RELAUNCH (arms a-d not closed by banked receipts)

1. T=2-4 tiled shapes: never benched (all receipts are T=1). The served decode
   runs MTP verify at T=2-4, so the T>1 MMVQ cells (different GCN schedule
   row: nwarps and rows_per_block differ; qi unroll factor) are open and carry
   served weight. Extension of the harness to the mul_mat_vec_q T>1 schedule
   is the first relaunch arm.
2. v_dot2 / dot-product ISA probe on gfx900: zero-card compile test; the 09-21
   fp16-dot2 analysis covered numerics and conversion cost, not the ISA probe
   itself (and v_dot2_i32_i32 as named in the task does not exist in GCN5
   documentation; v_dot4_i32_iu8 is gfx906+ - the probe settles it).
3. Arithmetic derivation of iq3s_grid: PRE-KILLED by table inspection - the
   512-entry grid is a trained codebook (pinned values 0x01010101, 0x07070b0b
   et al.), not an arithmetic sequence; no derivation exists. LUT residency
   constant-vs-global already banked NEUTRAL (gmem arm).
4. q8_1 operand layout/alignment (quantize ~7.1% of decode step): open; pairs
   with the T>1 extension since x-side effects scale with T.

## REPRODUCE

  cd /media/chris/ssd128/llamacpp/wt-mmvq-kernel2
  /opt/rocm-6.2.0/bin/hipcc -O3 -x hip --offload-arch=gfx900 -DGGML_USE_HIP \
    -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
    docs/amd-port/tests/bench_mmvq_gfx900.cu -o /tmp/bench_mmvq_k2 \
    -L /media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin -lggml-base
  # under /tmp/campaign_gpu_boot.lock, desk mmvq-kernel2, die 3 only
  HIP_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin \
    /tmp/bench_mmvq_k2 /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf 200 3

UPSTREAM-FACING NOTE: private-fork dev cell; nothing here is upstream-PR
material per AGENTS.md.
