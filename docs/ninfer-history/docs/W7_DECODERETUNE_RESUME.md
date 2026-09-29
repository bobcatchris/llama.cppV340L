# W7 decode-retune desk — resume note (lane amd/wo-w7-body, 2026-09-18)

Desk: F4a draft-arm retune + F1a draft-vocab coverage. This file is the running resume; the
closing row lands in `results/amd/coherence/W7_decoderetune_row.txt`.

## State

- [DONE 2026-09-18] BOTH LEGS CLOSED — see `results/amd/coherence/W7_decoderetune_row.txt`.
  F1a: real-prose coverage 98.4-99.4% (live battery 334/336 SE+-0.4%; REALK anchor 63/64;
  counting 98.0%; probe corpus 83.7%; corruption logs 73.1% = tripwire). Widening NOT
  triggered. F4a: all six serving draft GEMVs measured 218-246 GB/s isolated on the CURRENT
  serving arm (simt_r8_c4); forward sum 1.007/1.010 ms (2 runs, <=1.5% spread); NO shape below
  the pre-registered 60%-of-floor candidate line -> NO-GO on kernel retune, nothing to flip,
  the "~10 ms/round" was the pre-flip small_t class already harvested by PLOG-044 (head
  measured 0.232 ms vs 0.20 ms floor = 1.15x; 50x-gap hypothesis falsified). Master-plan row
  re-priced: remaining draft-side owners = F2 AR chain > align machinery ~1.15 ms/round >
  per-forward non-GEMV ~0.8 ms/round. Wave64 checklist verified clean (no kernel edit, census
  exempt). Desk commits: 69db827f3 (F1a) + the F4a closing commit.
- Serve-leg for the owner: NOTHING to integrate on F4a (serving already routes every draft
  GEMV to the winning arm). F1a high-volume confirmation rides the NEXT prose-serving boot via
  `NINFER_VOCAB_COUNT_DIR=<dir>` (recipe in the row).

- [DONE] F1a coverage (offline path — see constraint below): the in-tree counter
  (`src/runtime/tp2/vocab_output_counter.h`, wired at `tp2_backend.cpp:1658/3410/1686`) counts
  ACCEPTED tokens = the emitted stream, so an offline membership check of banked emitted-id
  streams (`[ids]` lines, first 64/response, `tp_engine.cpp:677-683`) against
  `tests/multi_gpu/data/qwen38_draft_vocab_ids.json` has IDENTICAL semantics to the counter.
  Measured over all banked serve logs (both this worktree and the shared checkout, 250 files,
  exact-tuple deduped):
  - REAL prose anchor (REALK sky-blue leg, the lane's only real-prose response): **63/64 =
    98.4%** (miss id 89661).
  - Counting class: **98.0%** (matches the 0.97-0.99 acceptance anchor class).
  - Healthy probe corpus (89 distinct probe streams, 1525 tok): **83.7%**; top-6 missing ids
    (54018, 30188, 62158, 32198, 20206, 89661) = 41% of all misses.
  - Corruption-era logs (G18r36_nvfp4ids, G18r65_nan, E12b — banked NaN/corruption verdicts):
    73.1% — SIDE FINDING: slice coverage doubles as a cheap corruption tripwire.
  - Decision rule (banked): anchor >= 95% -> widening NOT triggered by real prose; the 83.7%
    probe-corpus number is prompt-distribution-dependent. High-volume decider = the serve-leg
    counter (recipe below).
- [BLOCKED->RECIPE] F1a serve-leg: the :8100 posture boot holds ALL FOUR dies at 84% VRAM
  (7.27/8.59 GB), so a second boot cannot fit anywhere and the anchored instance must not be
  touched. Recipe for the owner's next prose-serving boot (zero code):
  `NINFER_VOCAB_COUNT_DIR=<abs dir> NINFER_DRAFT_VOCAB=<abs> <canonical boot line>`; drive real
  prose; read `grep "\[vocab\]" serve.log` — periodic lines every 10k accepted tokens + the
  `flush(shutdown)` line reports cumulative coverage at ANY run length.
- [CLOSED] F4a bench: `results/amd/coherence/w7_draft_arm_bench.cu`, gates A1-A5 in its header,
  bin sha fd764d9fa134a552, runs W7_draftarm_run{1,2}.log. Verdicts in the row file.
- Serving anatomy verified from code: all SIX draft-round W8G32 GEMVs ride simt_r8_c4 at T=1
  (five via `tp_kernel.cu:139-145` W8G32 t<=4 -> simt_r8_c4; fc + proposal head via the
  w8_dispatch generic tail / LMHEAD gate). Per-rank serving shapes: fc [5120,10240],
  attn-qkv [3584,5120], o_proj [5120,1536], gate_up [8704,5120], down [5120,4352],
  head [10240,5120] = 210.5 MB/forward, 2 forwards/round (align + 1 chain step at k=2).

## GPU posture

- Benches standalone, pinned `HIP_VISIBLE_DEVICES=2` (die 2). The other desk benches die 0;
  :8100 serve (PID 214823, bin 7c11c3ac07863366) UNTOUCHED. Kill only PIDs this desk started.

## Next on resume (desk CLOSED — pointers for the next owner)

1. Nothing owed on F4a/F1a from this desk; the closing row + row commit are the deliverables.
2. If a future window re-opens draft-side work, the fire order in the row stands: F2 W4AR
   (its own lane), then align/embedding machinery (~1.15 ms/round), then T=1 non-GEMV body.
3. The F1a serve-leg counter (NINFER_VOCAB_COUNT_DIR) should ride the next prose-serving boot
   as a free rider — recipe in the row.
