# W7 DECODE-RETUNE DESK — live log (2026-09-18, fresh session)

Seat: W7 decode-retune desk, worktree `/home/chris/worktrees/amd-wo-w7-body`, branch
`amd/wo-w7-body`. Prior desk instance died before writing anything — this session starts
fresh. Authoritative spec: `docs/amd/W7_DECODE_DESK.md` (read in full this session).

Legs:
- **F1a** — draft-vocab COVERAGE measurement (zero-code). Mechanism per desk file:
  `NINFER_VOCAB_COUNT_DIR=<dir>` → "[vocab] coverage=XX.XXX%" per 10k accepted tokens
  (`src/runtime/tp2/vocab_output_counter.h:44-77`). A live serving instance EXISTS (banked
  bin 7c11c3ac, port 8100). MAY send it requests; MUST NOT boot another server, kill,
  restart, or pkill anything not started by this desk.
- **F4a** — draft-arm GEMV retune. MTP draft-head GEMV `[1,5120]×[5120,40960]` W8G32 4-bit,
  53 MB weights/rank, ~10 ms/round vs ~0.19 ms memory floor (~50x gap). Kernel per desk file:
  w8 SIMT rowsplit family (`src/ops/linear/w8/`), dispatch seam `w8_dispatch.cpp:109-118`
  (`NINFER_LMHEAD_ARM` application site). Bench devices: **HIP_VISIBLE_DEVICES=2,3 ONLY**
  (0/1 belong to the parallel vperm desk). NO builds or cmake in
  `/home/chris/dual_5060_ti_ninfer` — build ONLY in this worktree.

## PRE-REGISTERED PASS GATE (F4a) — stated BEFORE any benching

- Primary gate (kernel level): best retuned variant achieves **>= 1.5x** mean speedup vs the
  current kernel on the serving shape `[1,5120]x[5120,40960]` (per-rank slice rows used at
  bench = the dispatch-resolved row count; recorded next to the number) → then the variant
  graduates to a serve-leg A/B proposal (NOT executed this session unless a GPU grant
  exists; proposal written here with expected round math).
- If best < 1.5x: verdict is NO-GO for serve-leg; bank the numbers and the reason.
- Reproduce-first law: every counted number must reproduce within **±2%** vs a same-window
  baseline (baseline re-run adjacent in time, same device, same clock posture). Single-shot
  numbers do not count.
- Clock law: every perf number reported next to the sclk level read from
  `/sys/class/drm/card*/device/pp_dpm_sclk` at bench time. Un-annotated numbers don't count.
- Box laws in force: no estimated VRAM refusals; no system-wide pkill; no builds in the
  shared checkout.

## LOG (newest at bottom; one entry per step)

- [step 0] Desk file created (checkpoint law). Spec `W7_DECODE_DESK.md` read. Next: inspect
  vocab_output_counter.h + locate live server posture (F1a), locate kernel + build dir (F4a).

- [step 1, F1a] Live instance posture: PID 214823 (`ninfer-serve_7c11c3ac07863366.bin`,
  port 8100, `--devices 0,1,2,3 --greedy`), up ~2h40m. **Boot env has NO
  `NINFER_VOCAB_COUNT_DIR`** (only `NINFER_DRAFT_VOCAB`, `NINFER_WORKSPACE_MIB`,
  `NINFER_ALLOW_NVFP4_TP2`, `NINFER_MTP_TAIL_ASYNC`) → the in-tree counter is INACTIVE in
  this instance and this desk may not reboot it. Counter domain re-derived from source and
  confirmed: `tp2_backend.cpp:3410` `vocab_counter.add(lic, a+1)` — anchor + accepted drafts
  per round = exactly the server's emitted token stream. Stopgap method (zero-code, zero
  engine risk): sample prose from the live instance (greedy), re-tokenize the output text
  with the tokenizer of record, and count membership in
  `tests/multi_gpu/data/qwen38_draft_vocab_ids.json` (40,960 ids).

- [step 2, F1a] Tokenizer of record reconstructed WITHOUT transformers/torch/regex (none
  installed): `export_tokenizer.py` flow executed with a torch stub →
  `build/collect/frontend/tokenizer.json` (12,809,320 B), **sha256
  0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 — matches the pin in
  tools/convert/qwen3_8_27b/convert.py exactly.** Pure-python byte-level BPE at
  `results/amd/w7_f1a/qwen38_bpe.py` (NFC + exact \p{L}/\p{M}/\p{N} category ranges +
  exact Rust-\s White_Space set + rank-ordered merges + 33 added tokens). Validation:
  (a) project_vocab.py spot check `"def foo(int x):" -> [727, 14785, 1494, 830, 1590]`
  EXACT (these ids were verified against the engine tokenizer per docs/98 §5A); (b)
  round-trip battery PASS; (c) live `/v1/messages/count_tokens` battery: 11/12 cases agree
  EXACTLY modulo a constant +42 wrapper (32-token server default system prompt + 10-token
  template wrapper `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n`); the
  12th (adversarial leading-whitespace case) is explained by server-side leading-trim of
  user content (" " counts == ""), i.e. input parse, NOT BPE. Known method caveat banked:
  re-encoding detokenized text can re-segment a small minority of positions vs the true
  emitted id stream (e.g. `Ġ`+`lighthouse` vs `Ġlighthouse`); the true-stream instrument
  of record is `NINFER_VOCAB_COUNT_DIR` at the NEXT prose boot (recommended below).
  `</think>` = added token 248069 (handled).

- [step 3, F4a] Kernel/seam identification (all file:line in THIS worktree):
  - Prompt's literal shape `[1,5120]x[5120,40960]` = the draft OUTPUT head; per-rank slice is
    `[10240,5120]` (53 MB) and it rides the GENERIC row of `select_w8_a16_launch`
    (`w8_dispatch.cpp:276-280`) -> `launch_w8_simt_r8_c4` at t<=4 — the propose field measured
    it healthy (0.32-0.34 ms, ~156-174 GB/s, TAIL_LMHEAD §3 item 2). The ledger's "~10 ms
    draft-arm" is the PRE-flip framing of the bucket; post-flip residual = align 2.16 +
    chain_fwd 1.39 (PLAN_50TPS_master.md:21,101-103).
  - The retune class the desk file names = `[14336..34816,5120]` (MTP draft-layer projections:
    gate_up [34816,5120], mlp-A [14336,5120], down [5120,17408]) served at ~54-56 GB/s vs
    137-233 GB/s proven for the same kernel family at other shapes (W8_LMHEAD_row.txt: simt_r8_c4
    233.4 GB/s @T=1 [62080,5120], 137.7 @ [10240,5120] T=3, DRAFT-FC 207-220 GB/s @T=1).
  - Serving seam: tp_gemv (`src/core/multi_gpu/tp_kernel.cu:137-146`) routes W8G32 t<=4 to
    `launch_w8_simt_r8_c4` (warp-per-row, 8 rows/CTA, 256 threads, K in 1024-value smem slabs,
    2-stage cp.async pipeline; `w8_rowsplit_gemm_simt.cu:13-27`, kernel `.cuh:147-246`).
    fc [5120,10240] is on the ops seam, flipped to simt at t<=4 (`w8_dispatch.cpp:107-121`).
  - **So the CURRENT SERVING ARM for the whole class at decode widths is simt_r8_c4, and the
    open question F4a answers by measurement: does simt_r8_c4 reproduce 54-56 GB/s at
    [14336..34816,5120]@T=1 (kernel-level pathology at mid shapes), or is the residual already
    cured post-flip (stale ledger number)? Then: do variants beat it?**

- [step 4, F4a] PRE-REGISTERED BATTERY + GATE (refining step-0 gate; stated before any run):
  - Shapes (W8G32_F16S, per-rank): S1 [34816,5120], S2 [14336,5120], S3 [5120,17408],
    S4 [10240,5120], S5 [40960,5120] (prompt's literal full-head reading), anchors A1
    [62080,5120] T=3 and A2 [10240,5120] T=3 (banked 137.0/137.7 GB/s cross-window refs),
    A3 fc [5120,10240] T=1 (banked 207-220 GB/s ref). T=1 primary (draft decode width), T=3
    secondary.
  - Arms: (A) simt_r8_c4 = CURRENT SERVING ARM = the baseline the gate is judged against;
    (B) simt_r8_c8; (C) small_t (the arm the 54-56 GB/s ledger class rode pre-flip — context);
    (D) NEW r4_c4 (RowsPerCta 8->4: 2x CTAs, half smem/CTA -> deeper co-residency);
    (E) NEW r4_c4 split-K S=4 and (F) S=8, two-STAGE DETERMINISTIC reduce (fixed-order fp32
    partials, no atomics — greedy byte-identical discipline preserved).
  - GATE (unchanged from step 0, now concrete): best variant mean >= 1.5x vs (A) on a shape at
    T=1, reproduced within +-2% in the same window (re-run adjacent in time), rel-L2 vs (A)
    < 1e-2, with sclk/mclk levels printed -> serve-leg A/B proposal. Otherwise NO-GO verdict
    with numbers banked.
  - Reproduce-first: (A) itself must reproduce the banked anchors (A1/A2/A3) within ~10%
    cross-window (clocks differ) before shape numbers are read; ±2% law applies to the
    same-window variant comparisons.

- [step 5, F1a] PROSE COVERAGE RESULT (banked raw data in results/amd/w7_f1a/):
  - Sample: 28 greedy prose generations from the LIVE instance (port 8100, bin 7c11c3ac),
    default thinking posture, 28 diverse essay/story/explainer prompts
    (collect_prose.py; raw JSONL banked). 18,688 output tokens re-tokenized with the
    validated BPE.
  - **COVERAGE = 97.597% (18,239 / 18,688; binomial SE ±0.112 pct-pts).**
  - Segment cut: content 98.95% (9,083/9,179); reasoning 96.28% (9,128/9,481) — the
    thinking register spends a wider vocabulary than final prose. Per-generation spread
    86.9-99.9% (one outlier at 86.9%).
  - Missed mass: 449 tokens across ~200 distinct ids; top-40 missed ids listed in
    coverage_report.md (rare morphological/numeric/foreign-script pieces — consistent with
    the slice being top-40,960 by frequency).
  - **Implication vs the banked decision rule (ACCEPTANCE_LEVERS §b / desk file F1 step 1):
    coverage is ABOVE the 90-95% widening trigger → do NOT widen the slice.** Ceiling math:
    a draft position's acceptance is capped by coverage at 0.976; k=2 tok/round ceiling
    1 + c + c^2 = 2.93 at c=0.976 — the current prose acceptance 0.65 is far below that
    cap, so the 40,960 slice is NOT the binding acceptance suppressor; the suppressor is
    the drafter itself (MTP-head quality + quant argmax flips = the desk file's suppressor 2,
    F1b's gate). Slice widening (+130 MiB/rank, +0.3 ms) would buy at most ~2.4% of
    per-position acceptance mass — REFUSED by the decision rule, not by the VRAM law.
  - RECOMMENDATION (instrument of record): the NEXT prose-serving boot (any lane) should
    set `NINFER_VOCAB_COUNT_DIR=<dir>` — zero code, counter already in-tree — to re-measure
    on the TRUE emitted id stream (kills the re-segmentation caveat) and to feed the Phase-2
    merge (project_vocab.py --merge-counts). Until then 97.6% stands as the practice number
    with method banked above.

- [step 1, session 2 2026-09-18 ~19:00-21:00] F1a MEASURED: live real-prose coverage 99.40%
  (334/336, 21-prompt battery, engine-printed [ids]; SE +-0.4%); REALK anchor 98.4%; counting
  98.0%; healthy probe corpus 83.7%; corruption-era logs 73.1% (tripwire side-finding). Widening
  NOT triggered (<90-95 rule false). Serve-leg counter recipe banked (dedicated boot was
  VRAM-blocked: :8100 holds all 4 dies at 84%). NOTE: my probe run shared the serve with a
  foreign 768-tok decode bench (their streams independently scored 98.75% - kept separate).
- [step 2, session 2] F4a bench written (results/amd/coherence/w7_draft_arm_bench.cu) with
  pre-registered gates A1-A5; compiled at -O3 LAW line, bin sha fd764d9fa134a552. Two runs on
  die 2, ALL serving shapes 218-246 GB/s (o_proj 83 = 60.7% of floor, above the candidate line);
  forward sum 1.007/1.010 ms; A2/A3 no candidate -> A5 fires: the "~10 ms/round draft-arm"
  is NOT kernel-isolatable post-PLOG-044 (head measured 0.232 ms vs 0.20 floor = 1.15x, the
  50x-gap hypothesis falsified). NO-GO on serve-leg per the pre-registered rule; closing row
  results/amd/coherence/W7_decoderetune_row.txt + master-plan re-price (draft-arm term DONE;
  remaining owners: F2 AR chain > align machinery ~1.15 ms > per-forward non-GEMV ~0.8 ms).
  Wave64 checklist verified clean in code (2 rows/wave by construction, width-32 segmented
  shuffles, CTA-uniform trip counts).
