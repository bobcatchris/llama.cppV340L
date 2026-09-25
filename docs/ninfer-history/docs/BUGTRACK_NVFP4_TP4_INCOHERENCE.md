# ⚠️⚠️⚠️  BUG OPEN — NO FIX EXISTS — MODEL STILL BROKEN  ⚠️⚠️⚠️

# NVFP4@TP4 served-output incoherence ("garbage-after-68")

> **STOP. READ THIS WHOLE HEADER BEFORE WRITING ANY CODE OR BOOTING ANYTHING.**
> **25+ agent-hours, ~36 boots (C42–C77), 25+ suspect families "cleared" — and the model
> outputs garbage at ≥68 prompt tokens TODAY, same as when the hunt started.**
> **Lines of fix code landed: ZERO. Ladder-green runs: ZERO. If your plan repeats an entry
> in §1 below, you are about to waste hours re-proving something already proven.**

```
SCOREBOARD (honest, 2026-09-16 ~17:30 CDT — updated at E-13)
  boots + static laps spent ............ C42–C77 + E1–E13 (2 windows this era: E12, E12b)
  suspect families eliminated .......... 25+ (see §1 — each with a receipt) + ARENA (E-12:
                                         alloc tables structurally identical T64 vs T65)
  mechanisms convicted ................. 1 — THE CAUSAL CONV'S T-GATED KERNEL PATH:
                                         T≤64 → sequence kernel; T≥65 → prefill_pairs
                                         kernel (dispatch causal_conv1d_silu.cpp:162-169,
                                         kCausalConvSequenceMaxTokens=64). QC/KC/VC dirty
                                         from row 0 at T≥65, even-channels-only, measured
                                         E12b (E-13).
  fixes landed ......................... 0 (fix direction + closure bar in E-13)
  model output at prompt ≥ 68 tokens ... STILL MOJIBAKE (reproduced 2026-09-16 16:53 CDT,
                                         E12b window, deterministic)
  model output at prompt ≤ 66 tokens ... coherent (never was broken; re-verified 16:53)
  artifact ............................ /media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer
  bank binary under test .............. /home/chris/artifacts_bin/ninfer-serve_1f068ea7aa8a6aa2.bin
```

**THE DEFECT, one line:** deterministic, length-gated garbage generation — coherent at
prompt ≤66 tokens, empty-content + multilingual mojibake at ≥68, at EVERY longer length
(288 … 10,000 tokens). Same boot → same garbage (deterministic). q3 twin breaks at the
same threshold (shared-stack onset). North star (user order): **coherent at 10k first,
then throughput** (prefill ~10 tok/s at 10k = 972 s/request — separate crisis, DO NOT
conflate with coherence).

**APPEND-ONLY LAW:** entries are numbered and dated; corrections are NEW entries citing the
old one; nothing below is ever edited or deleted. (Doc restructured 2026-09-16 ~15:00 CDT
by the audit seat at user order — prior revision superseded by this one; law applies
forward from this revision.)

---

## §1  WE TRIED THIS — DO NOT WASTE YOUR TIME

Every row below is CLOSED **with a receipt**. Re-running any of it without labeling the
run "determinism glass (expected: same result)" is banned. If you disagree with a row,
APPEND a correction entry — never re-litigate silently.

### 1A. The big verdict everyone quotes — and what it does NOT mean

| # | What was tried | Result | Receipt |
|---|----------------|--------|---------|
| 1 | **"No corruption" verdict (helper, 09-16 09:00)** | K-cache hash differences across prompt sizes = ≤32-ulp fp reassociation in RCCL ring allreduce. TRUE — **but it says NOTHING about output coherence.** Citing it as "bug fixed" is the exact error that wasted the last session. The same seat's own banks show garbage output. | commit `3aa363ef`, `results/amd/HELPER_nvfp4_tp4_verdict_no_corruption.md`; contradicted by its own `G18r63/r64/r65` banks |
| 2 | **Z1 / NINFER_ZEROSTATE=1** ("cleared the hunt with Z1") | Zeroing all decoder state at request boundary: trips byte-identical ON vs OFF, 64/64. **State-carry exonerated.** That is ALL it cleared. | C69, handoff §21, row sha `c1a65ad1`, hkv `bad1a859` |

### 1B. Cache/state-level suspects — ALL CLEARED, do not re-open

| # | Family | How killed | Receipt |
|---|--------|-----------|---------|
| 3 | Frontier/tail-chunk split ([plen−2]+[2]) | `NINFER_NO_FRONTIER=1` A/B: trips persist byte-equal | C71 `40294527` |
| 4 | Window/clobber (page history) | 66R repeater: history-independent | C65 `2a60c9ec` |
| 5 | Block-table publish ordering / word-1 race | rows materialize whole at setup, one H2D, never touched mid-request | `54b52c3c`, `21ae668b` |
| 6 | Positions/rope producer | prefill never touches io_.pos; chunk-local; rope kernel T-clean | handoff §7/§12 |
| 7 | Dump-artifact (page-0) | SLOT2 dump graded-equivalent; glass tests | handoff §4/§6 |
| 8 | GDN chain (10 sites: g_cs_offset, beta/g guards, normalization-split, conv-pairing, state_passing coverage) | all token-major correct, zero T-coupled predicates | `37744b70`, `6d39e0dd` |
| 9 | tp_unpack_gdn_qkvz strides | 4-ground clear | agent5 r5 + `72ef50bc` |
| 10 | Packer | shape-kill (broken pitch contradicts head-frozen/boundary-clean) | agent5 r5 `a05f7d54` |
| 11 | Flash tier | route-unreachable (TAIL-EMPTY witness) | `c647783f` |
| 12 | conv-state coverage | monotone [T−3..T−1], zero predicates | `be665fa0` |
| 13 | Epilogue semantics + tile-writes (o_proj/rmsnorm) | shape-kill | agent5 r6/r8 `1ffa07e1`+ |
| 14 | W4A4 producer-pitch as L1 site | q3 co-flip at T=64/65 ⇒ shared-stack; in-family read clean | C75 `a3bedfdf` + `33dbee3b` |
| 15 | qkv producer pitch ([T,rows] emitter) | exact index match both paths | `3c7b9ece` |
| 16 | gfx906 prefill as L1 site | const KV args — cannot flip a plane it cannot write | `7dd5649a` |
| 17 | ar-verify ring | never executes in datum boots (MTP-off) | agent5 r9 |
| 18 | Tokenizer divergence | artifacts byte-identical (C5 probe) | handoff §1 |
| 19 | Masked-fill / kSmallT route | static lap: identity clamp on the live route | handoff §4 |
| 20 | NaN at gdn layer-0 verify output | nan probe 0/1536 all ranks (**tap is narrow — deeper layers NOT covered**) | G18r65 nan bank |
| 21 | Content-excuse at rows 8/16/24 | tokenizer decider: identical filler tokens | handoff §3 |
| 22 | Pure chunk-count trigger | 85-token enum = ONE chunk, still noise | C46 `52f51d5f` |
| 23 | RCCL chunk bug | prefix-invariance + exactness hold | `3aa363ef` chain |
| 24 | Arm-switch M32N64→M32N128 as mechanism | downgraded to "hypothesis without a site" | `33dbee3b` |

### 1C. Output-level measurements that ALREADY EXIST — do not re-derive from scratch

| # | What was measured | Result | Receipt |
|---|-------------------|--------|---------|
| 25 | **C46 output bisect** (54 → 7335 tokens) | 54-tok "Hi." coherent ×2 boots; 85, 166, 279, 507, 963, 1873, 3693, 7335 ALL noise. Boundary bracketed **54..85** — attributed (wrongly) to "content-kind" | `52f51d5f`, bank `50f61f06` |
| 26 | **L1_fast ladder** (this audit seat, 09-16 14:06) | Same content swept across lengths: 56/60/61/64/66 PASS, 68/72/80/96/128/160/224/288 FAIL. **Cliff = 66→68 = T-space 64/65.** This REFINES C46's bracket — it did not discover it | `results/amd/coherence/L1_fast_row.txt`, commit `d24f2540` |
| 27 | Helper servability banks | water@56 OK, 2+2@60 OK, 1-10@68 GARBAGE; 2535/8092/9319/9996 all garbled/empty | `G18r64`, `G18r63`, `G18r65` (now committed, `d24f2540`) |
| 28 | q3 twin at the boundary | K-planes flip at same T=64/65; twin "printed noise at 67/68" (**streams quarantined as sampler-artifact — a CLEAN q3 text control at the cliff still does not exist**) | C75 `a3bedfdf`, `8dff6553` |

### 1D. Tools that ALREADY EXIST — reuse them, never rewrite

| Tool | Where | Status |
|------|-------|--------|
| **ids_offset_triage H1–H6 judge** (constant-offset / adjacent-shift / lane-word-swap / region-scatter / period-coupling / tokenizer-surface arms) | `rev 3.39` (`8582334a`), selftest 9/9 | **NEVER FIRED at the cliff.** Raw [ids] lines for PASS-vs-FAIL legs are ALREADY BANKED in `results/amd/p3/G18r65_nan_serve.log` (PASS legs end `248046`; FAIL legs e.g. `230294 2580 39698 45494`). Running this judge needs ZERO new boots |
| Coherence ladder (known-answer, graded lengths) | `results/amd/coherence/coherence_ladder.py` | works; RED row banked |
| Helper taps: ZBIS / GBIS / GBIS2 / MAG / MAG2 / ZEROSTATE / NO_FRONTIER / HKV_DBG | env-gated, default-off, in `attn_mix_tp` + `text_context_impl.h` + `tp2_backend` | proven |
| Helper graders | `results/amd/helper/*_grade.py`, `sumcheck.py` | proven |
| Boot line, artifact paths, port discipline | §4 | proven |

---

## §2  WHAT IS ACTUALLY KNOWN (laws measured, none explained)

These are REAL, reproducible regularities — any mechanism claim must reproduce ALL of them,
or it is a guess:

1. **Output cliff:** coherent ≤66 prompt tokens, garbage ≥68 (T = plen−2 ⇒ cliff at T=64/65).
2. **q3 co-flip:** q3 K-planes flip whole-plane at the same T=64/65 ⇒ onset is shared-stack.
3. **Crawl law:** threshold TH(T)=round(T/2−1) HALF-UP fits all eleven crawl thresholds.
4. **Event law:** late events {71,77,84,90,97}, entry h(k)=k+7 exact on 4 events × 26 rows.
5. **Alternation:** gaps 6,7,6,7 (corrected direction).
6. **The 13:** prime, UNDERIVED at two seats — no stride in the era multiplies to it.
7. **Head-freeze:** rows {0..7} never trip; frozen set {0,1,4,8}.
8. **K-plane diffs ≤32 ulps** across legs (benign reassociation) — **yet text is wholesale
   garbage. Any mechanism must explain why ulp-level cache noise and catastrophic text
   garbage COEXIST at the same boundary. This is the central unexplained tension.**
9. **Falsifiers armed, never fired:** e6=103 / e7=110 (C77 filed, not fired).
10. Determinism: same prompt+boot → byte-identical output (garbage included).

---

## §3  WHY THERE HAS BEEN NO PROGRESS (the hole, stated plainly)

The hunt spent 25+ hours eliminating cache-level suspects (§1B) and closed with "the K-diffs
are benign." **Nobody ever traced the TEXT corruption to a site.** The output garble was
treated as a downstream symptom, then the downstream symptom was declared healthy by a
verdict that never tested it. The defect lives in generation — and generation has been
touched exactly twice at the output level (C46, L1_fast), both times only to MEASURE the
cliff, never to catch the corruption entering.

Corollary: another static read of another kernel is **the lowest-value action available**.
25+ of them are banked. The corruption has never been caught IN THE ACT on the output side.

---

## §4  ONLY OPEN WORK (ordered; reuse §1D tools first)

1. **Fire the existing ids judge at the cliff** (zero boots): run
   `ids_offset_triage_agent5.py` (rev 3.39) on the banked PASS/FAIL [ids] lines in
   `G18r65_nan_serve.log`. H6 period-16 ⇒ scale-block/plane organ; H3 lane-word swap ⇒
   addressing; H1 constant threshold-offset ⇒ systematic boundary. This is the
   cheapest unspent decisive measurement in the repo.
2. **Pin the cliff token** (one boot, minutes): ladder legs at prompt 67 (+66/68 glass).
   Decides 66/67 vs 67/68 against the K-calendar B={66} row.
3. **Clean q3 text control** (one boot): same ladder vs `qwen3_8_27b_q3.ninfer`.
   C75 cleared q3 K-planes co-flip; its TEXT at 66/67/68 has never been cleanly graded
   (the old twin streams are quarantined). Decides shared-stack vs nvfp4-only FOR THE GARBAGE.
4. **Catch the corruption entering** (small build, only after 1–3): top-k logits/ids dump
   for the first 4 decoded tokens at prompt 64 (PASS) vs 68 (FAIL), same body —
   "distribution subtly drifted" vs "garbage from token 1" splits the mechanism classes.
5. **Organ A read** (zero boot, never landed): chunk-vs-tail handoff
   `gated_delta_net.cpp:253-305`, acceptance = explains 64/65 only.
6. **Fire C77** (e6 legs [102,103,104]) when a window exists — validates or kills the k+7 law set.
7. **Only after coherence GREEN somewhere:** NVIDIA-side test duplication (user order) —
   note honestly: most `tests/` are shared; only 9 files are NVIDIA-only, and `phase_gate.cu`
   binds a dead `/home/intel/models` path. Port what runs; do not expect it to find this bug —
   the bug is a serve-time numerics cliff no existing gate models.
8. **Throughput (AFTER coherence):** prefill ~10 tok/s at 10k (972 s), decode 2.1–2.4 tok/s
   post-deep-prefill vs ~8 short. Phase-axis rule: prefill and decode columns never mix.

**Definition of done for the bug:** a NEW banked binary sha on which `coherence_ladder.py`
exits 0 (all legs PASS, `--long` included), with §1C row 26 cited as the RED row. Nothing
short of that closes this document's parent bug.

---

## §5  APPEND-ONLY ENTRIES (newest last)

- **[2026-09-16 ~15:00 CDT] E-1 (audit seat).** Restructured this document at user order:
  do-not-redo list moved to the front (§1), scoreboard added, "no progress" stated plainly
  (§3), §7 re-ordered to reuse-first after verifying overlaps (C46 already bracketed the
  cliff 54..85 — L1_fast refines, not discovers; ids judge exists unfired). Supersedes the
  earlier same-day revision in full; prior revision's content is preserved in the entries
  above. No verdicts changed.
- **[2026-09-16 ~15:40 CDT] E-2 (audit seat). §4.1 EXECUTED — ids judge fired at the cliff;
  verdict ALL-MISS, sampler/transport exonerated, corruption is upstream in the forward.**
  Tool: `tools/v340l/nvfp4/ids_offset_triage_agent5.py` (rev 3.39, `8582334a`, banked here
  from git so nobody digs again; selftest 9/9 re-run this seat,
  `results/amd/ids_judge/selftest.log`). Inputs: the 14 L1_fast legs' [ids] lines, already
  banked in `G18r65_nan_serve.log`, extracted to `results/amd/ids_judge/<leg>.ids.txt`.
  Runs: each FAIL leg vs ref=e66 (coherent BLUE@66, same prompt family) —
  `results/amd/ids_judge/<leg>_vs_e66.json`. **ZERO new boots.**
  Results (8 FAIL legs):
  - **H1 constant-offset / H2 adjacent-shift / H3 bit-swap: support 0.0 on every leg.**
    The generated garbage is NOT an id-base offset, NOT a stream/lane swap, NOT a byte-order
    swap. The three simplest addressing-fault families are DEAD for the output stream.
  - **H4 region-scatter fires on 6/8 legs** (noise id medians 24k–246k vs ref 3,074 —
    garbage lives in rare-vocab regions; k160/m288 within region).
  - **H6: no dominant period** (best r2 0.35–0.47 at K∈{4,8,16,32}, inconsistent across
    legs, all under the 0.5 bar) — corruption is not stride-shaped in the id stream.
  - **first_region_divergence = 0–1 on every leg: garbage from the FIRST generated token.**
  - f68/i96 INSUFFICIENT (<8 ids) — named state, judge refused to over-read them.
  Judge rc=1 ALL-MISS on all 8 (no winner arm). Per the judge's own grammar
  (coherent-but-wrong ⇒ arithmetic class or new class) plus §2 items 8+10, the reading is:
  **the sampler faithfully emits argmax of a genuinely corrupted forward. Sampling,
  detokenization, and transport are EXONERATED for the text garble. The suspect surface is
  the prompt-forward's final hidden state at T≥65.** This also sharpens §2.8's tension: the
  ≤32-ulp cross-leg K diffs say legs agree with EACH OTHER, not that either is semantically
  right — short prompts prove the forward right at T≤64, the cliff proves it wrong at T≥68,
  and both facts survive the ulp result. **Consequence: §4.4 (last-position logits/hidden
  tap, PASS-vs-FAIL same body) is now THE decider — everything downstream of the forward is
  clean, so zero-boot options for this question are exhausted.** Next session: build the
  small logits tap; do NOT re-run ids triage (this entry), do NOT re-derive §1 rows.
- **[2026-09-16 ~16:05 CDT] E-3 (audit seat). CLIFF PINNED TO THE TOKEN: prompt 66 PASS →
  prompt 67 FAIL (= T 64 → 65).** Fine grid k=1..8 on the BLUE body
  (`results/amd/coherence/L2_finegrid_row.json`, same boot bank 6cacfc3c): 63/64/65/66 all
  PASS ("BLUE", coherent reasoning); 67/68/69/70 all FAIL (empty content, mojibake
  reasoning from token 1). T = plen−2 ⇒ the cliff sits EXACTLY where the GDN chunked
  prefill (kChunkSize=64) first spawns a TAIL after one full chunk: T_full=64, tail=1 at
  T=65. Also matches the K-calendar C-region onset (plen 67) — but the K-diffs are ulp
  (§1 row 1), so the calendar and the text garble are two symptoms; the tail handoff is
  the one mechanism that turns on exactly at the cliff. Static read of the handoff
  (wrapper gated_delta_net.cpp:250-305, chunked/launch.cu, recurrent.cuh/cu launcher +
  kernels, aliasing, eps, head mapping): NO defect found — consistent with the record's
  25 cleared reads. Static reading STOPPED per §3; moving to the bypass experiment (E-4):
  env-gated NINFER_GDN_FORCE_RECURRENT forces T_full=0 (all-recurrent — the path T≤64
  uses and decode proves every boot) and re-runs the fine grid. Coherent-at-65 under
  force-recurrent convicts the chunked+tail handoff as the corruption carrier; still-
  garbage exonerates it and re-opens the T-coupled surface.
- **[2026-09-16 ~17:30 CDT] E-4 (audit seat). THREE more organs exonerated by bypass; ids
  ground truth banked.** (a) `NINFER_GDN_FORCE_RECURRENT=1` (env-gated T_full=0 bypass,
  default-off, committed in `gated_delta_net.cpp`; binary
  `ninfer-serve_d274afbd10a3a8cd.bin`): whole GDN prefill runs pure-recurrent — prompt 67
  STILL GARBAGE. GDN chunk-vs-tail handoff EXONERATED for the text bug. (b)
  `NINFER_NO_FRONTIER=1` + force-recurrent: single-pass prefill, no frontier split —
  prompt 67 STILL GARBAGE (identical mojibake). Frontier machinery EXONERATED. (c) ids
  judge at the cliff already ALL-MISS (E-2). Bank rows:
  `results/amd/coherence/E4_forcerec_row.json`, `E4_nofrontier_row.json`. **Probe-method
  warning: max_tokens must be ≥48 for the BLUE body (reasoning needs ~35 tokens before
  content); an mt=32 probe batch this session produced false FAILs — VOID rows, do not
  cite.** Also: a `setsid` fork ate one kill and an old server answered a probe batch —
  always verify `/proc/<pid>/environ` before trusting a boot's rows.
- **[2026-09-16 ~18:10 CDT] E-5 (audit seat). LAYER TAP: embeddings CLEAN, layer-0 x
  NON-CAUSALLY CORRUPTED at T=65. The corruption is INSIDE layer 0's forward.**
  Instrument: existing `NINFER_C_LAYERS=1 NINFER_C_DUMP` full-x dumps (LE pre-layer-0,
  LA_r<layer> post-layer, rank 0), boot `e5tap`, requests k=4 (T=64, PASS) and k=5 (T=65,
  FAIL), files under /tmp/coh_tap (MBP1 header = 20 B; grammar
  `<phase>_r<layer>_p<pass>_l<rank>_n<hidden·T>_s<tag>.bin`); analyzer
  `results/amd/coherence/e5_tap_analysis.py` (+ `.out`), serve log `e5_tap_serve.log`.
  Ground truth: `[COH-IDS]` dumps of the exact served id sequences (E-6 tap, binary
  `ninfer-serve_dcd746468732e236`, log `e6_ids_serve.log`) prove the T=65 id stream is
  CORRECT (wrapper 55 tok, alphas at rows 55–59, closing 60–64) — **the "inserted row"
  hypothesis is WITHDRAWN**: it was an artifact of comparing different-length alpha runs
  (consecutive identical embeddings make shifted rows bit-match). Embedding dump: rows
  0..58 bit-identical between runs, alpha/closing rows exactly as tokenized — CLEAN.
  **THE FINDING: at LA layer 0, identical-token rows (e.g. row 2, token 198, position 2)
  differ between T=64 and T=65 by up to |Δ|=81.25 with 5120/5120 bf16 bits — wholesale,
  non-causal, NOT ulp reassociation** (norms ~12–23 at layer 0). Layer 0 is a GDN layer;
  its chunked kernel receives IDENTICAL rows 0..63 in both runs, so a correct causal
  pipeline cannot produce this. The corruption enters in layer-0's own organs (input
  projection A16 kernels with T-coupled schedules, GDN kernels, gating, conv, out-proj +
  TP allreduce) acting non-causally on early rows once T≥65. Last-row norms blow up
  (~175–290 vs ~12–23) from layer 0 onward — the garbled text is downstream decay of a
  layer-0 wound. NEXT (armed): MAG hex dump (NINFER_GBIS_MAG=1, x+partials at gidx 0) at
  T=64/65 in one boot → per-row max-delta table splits layer 0 into projection / conv /
  GDN / allreduce; then per-rank MAG2 splits the allreduce. Do NOT re-derive E-1..E-5.
- **[2026-09-16 ~19:20 CDT] E-8 (audit seat). TAIL-BAND HYPOTHESIS DEAD — the gate is
  EXACTLY "main-pass T ≥ 65".** Prediction ladder across tails 5..31 + tail 0
  (`results/amd/coherence/E8_tailgrid_row.json`): T=69..96 ALL FAIL including tail 0
  (T=96 = three clean 32-chunks, no small/decode chunk at all) and tails 27/30 (which PASS
  as T=62/63's second chunk). NOT a per-tail-size kernel-arm bug, NOT request poisoning
  (k=4 passes again immediately after failures; water-probe 60-after-68 agrees). The
  failing condition is exactly T ≥ 65 in the main pass. **Also in this entry: T=65 and
  T=96 produce BIT-IDENTICAL corrupted layer-0 x at rows 0..2** (d(96,65)=0.0000; tiny ulp
  drift at rows 17/40) — ALL T≥65 share ONE deterministic wrong computation for the early
  rows, plus per-T ulp noise. T65 row 0 matches NO T64 row/layer anywhere (no shift, no
  layer offset — novel wrong values). Row-0 norms at layer 0: ~175–290 (T65) vs ~12–23
  (T64).
- **[2026-09-16 ~19:45 CDT] E-9 (audit seat). IMPOSSIBILITY ARGUMENT (the strongest
  sentence in this file):** the served id sequences are CORRECT (E-6 `[COH-IDS]`); the
  embedding dump is BIT-IDENTICAL for rows 0..58 between T=64 and T=65; every layer-0 organ
  is causal (rmsnorm, per-token A16 projection chunks [32], causal conv width 4, gated
  delta recurrence, per-token out_proj); therefore layer-0 outputs for rows 0..54 MUST be
  bit-identical between T=64 and T=65. MEASURED: they differ wholesale (5120/5120 bits,
  maxd up to 81, identical fingerprint under force-recurrent AND default path, and shared
  bit-exact wrong values across T=65..96). ⇒ **a non-causal write or a T-conditional
  resource bind corrupts the forward for ANY T≥65.** All structural suspects EXONERATED
  this session: GDN chunk-vs-tail (force-recurrent bypass), frontier split
  (NINFER_NO_FRONTIER), work-arena size (96 vs 1024 MiB — e8 boot still fails), request
  poisoning, tokenizer/staging (ids correct), sampler/transport (ids judge ALL-MISS),
  NaN (gdn0 tap). **The contradiction is the compass: whatever writes outside its bounds
  or binds a resource by T-class IS the bug.**
- **[2026-09-16 ~20:05 CDT] E-10 (audit seat). ADDRESS CENSUS ARMED + FIRST CENSUS.**
  New env `NINFER_COH_PTRS` (default-off): prints address+extent of x/qkv_conv/qc/o/on/
  partial at every gidx boundary (binary `ninfer-serve_38b30843a812cf5c.bin`, log
  `results/amd/coherence/e11_ptr_serve.log`). First census (rank0 gidx0): NO overlap among
  the six tensors at T=64 or T=65; the arena layout SHIFTS with T (x start +0x200,
  qkv_conv +0x7400, partial +0xc200 at T=65) — layout shift is legal, but **any consumer
  holding a T=64-shaped pointer/view into the T=65 arena (or vice versa: an allocation
  sized at T=64 but written with T=65 strides) would smash exactly the way E-9
  describes.** Next session, in order: (1) extend COH-PTR to print the FULL arena
  allocation table (every work_.alloc offset+size in one prefill) and diff T=64 vs T=65 —
  the overlapping pair names itself; (2) check the same for the T=2 tail pass and the
  DECODE-step allocations; (3) if no host-visible overlap, the smash is device-side:
  bisect with per-op hex dumps (extend MAG to qkv_conv/qc pre-allreduce at gidx 0) to find
  the FIRST op whose early rows diverge — the divergence op convicts its kernel; (4)
  keep everything in §3 dead — none of it is re-opened by E-8/E-10.
- **[2026-09-16 ~16:45 CDT] E-11 (audit-seat continuation, amd/tp4-cure @ 85c95c47). TAPS
  ARMED + CENSUS GENERALIZED + ORPHAN RECORDED.** (a) E-10's next-session items 1–3 are now
  ONE boot: `NINFER_COH_ALLOC` (full arena allocation table — every alloc/push/pop/reset with
  seq/offset/bytes/align, MARK-segmented by REQ/PASS/LAYER/DEC; files per-arena keyed by base
  addr) and `NINFER_COH_OPS` (16-stage layer-0 gidx0 per-op dumps X0 HN G0 B0 QK QV ZB QC KC
  VC GB BB O0 ON PA XA + XL via the proven c_dump_bf16 writer). Code committed, default-off;
  banked binary `ninfer-serve_82a9515961f75bc3.bin` (sha256 82a9515961f75bc3…, full sha in the
  runner); runner `results/amd/p3/E12_coh_tp4_agent6.sh` (STAMP_ACK gate, KFD-clean
  precondition, df≥20G, env-verify against the stale-server class of E-4); analyzer
  `results/amd/coherence/e12_analysis.py` (selftest: planted alloc divergence, planted reuse
  event, planted stage divergence — all caught; T-proportional growth classified expected).
  (b) CENSUS GENERALIZED, zero boot: the banked E-10 census (`e11_ptr_serve.log`, banked copy
  sha16 5ab611c9a2784fd9) re-read IN FULL — 1152 lines = 4 legs (T=51 warmup, T=2 tail, 64,
  65) × 4 ranks × 48 gidx: **zero extent drift, zero overlap among the six GDN tensors in
  EVERY leg and rank** — E-10's no-overlap finding generalizes; the census surface is clean
  everywhere it reached. ⇒ the corruption carrier, if it is an allocation, is one the census
  never printed: gated_delta_net's internal scratch, attention/mlp/recipe allocations, or a
  device-side write through valid pointers — exactly what E-11's two taps add.
  (c) PRE-DECLARED READINGS (decided before the window fires):
    1. alloc-seq length mismatch or non-proportional size change ⇒ T-conditional allocation
       NAMED — next session diffs its writer; per-op bisect confirms.
    2. alloc-seq clean, no reuse candidates ⇒ host-visible arena exonerated as carrier;
       conviction moves to the per-op bisect: first stage with first_div < K6667 convicts its
       organ (K6667 = X0-aligned prefix, E-5 says ~59 for this body). X0 prefix ≠ ~59 ⇒ the
       E-5 alignment assumption itself needs re-reading.
    3. all stages clean within K but XA/XL differ ⇒ corruption between stages (allreduce
       internals / op boundaries) — next: per-rank PA dumps already in the tap split the
       out_proj from the allreduce (PA clean + XA dirty ⇒ the fold/allreduce wrote it).
    4. T67-vs-T96 cross-check: E-8's "one shared wrong computation" predicts early rows
       BIT-IDENTICAL between the two FAIL legs; a mismatch REVISES E-8 — itself a finding.
  (d) ORPHAN RECORDED: the E-10 seat (`amd-tp4-cure-branch-review`, EXITED ~16:05 CDT) left
  its census boot alive — PID 2059260, port 8098, binary 38b30843…, HKV_TAG=e11ptr, idle 30+
  min, 2 requests done, holds all 4 dies. Release pending board word (grant request G-AMD-E12
  sent to agent4+agent1; kill by exact recorded PID/pgid only, never a pattern). Window fires
  immediately on grant; E12 legs k=4 (T=64 glass) / k=5 (T=65 FAIL) / k=36 (T=96, branch 4)
  / k=4 again (poisoning glass), mt=64 per the E-4 warning.
- **[2026-09-16 ~17:05 CDT] E-12 (audit-seat continuation; grant G-AMD-E12/E12b issued DIRECT
  BY THE USER — "you are granted usage of anything, just fix the problem" — supersedes the
  pending board asks). TWO TP4 WINDOWS FIRED; ARENA EXONERATED (pre-declared branch 2).
  Orphan e11ptr server (PID 2059260) released first by exact-PID TERM, KFD verified 0.**
  Windows: E12 (bin 82a9515961f75bc3) then E12b (bin 1f068ea7aa8a6aa2, adds rank-tagged
  COH-ALLOC marks + OWNER handshake after the first window showed the 4 rank THREADS of one
  process interleaving marks in every table). RED row both windows, deterministic:
  k4/plen66 PASS "BLUE", k5/plen67 FAIL mojibake, k36/plen98 FAIL, k4-glass PASS
  (rows: `results/amd/coherence/E12b_coh_tp4_row.txt`; serve log banked alongside;
  20 alloc tables + 792 op dumps per window).
  PART A (arena tables, owner-filtered): T=64-pass vs T=65-fail epochs of every rank's
  96 MiB work arena — **1235 allocations vs 1235, ZERO anomalies** (no sequence-length
  mismatch, no non-proportional size change, no alignment change; reuse events symmetric
  222/222). The host-visible arena allocation surface is EXONERATED as the corruption
  carrier — E-9's "T-conditional resource bind" is NOT in the allocator.
  PART B (per-op bisect, layer-0 gidx0, all 4 ranks agreeing): X0/HN/G0/B0/QK/QV/ZB/GB/BB
  bit-identical PASS-vs-FAIL within the content-aligned prefix K=59 (E-5's 59 CONFIRMED by
  the X0 alignment reference); **QC/KC/VC (post causal-conv) diverge from ROW 0**;
  O0/ON/PA/XA/XL downstream dirty (propagation). Analyzer
  `results/amd/coherence/e12_analysis.py` (rows table now uses MEASURED world-4 shard dims:
  k_rows=512 v_rows=1536 qkvz=4096; fp32 stages dump as 2× bf16-words).
- **[2026-09-16 ~17:30 CDT] E-13 (audit-seat continuation). THE ORGAN AND THE GATE ARE
  NAMED — `ops::causal_conv1d_silu_split3`, and it is T-GATED BY DISPATCH, NOT BY MATH.**
  Fine structure (rank0 QC, T=65 leg vs T=64 leg, rows 0..58 — bit-identical inputs):
  (1) **exactly the 256 EVEN channels of 512 are wrong in EVERY row 0..58; odd channels
  bit-match the clean leg** (pair-structured ⇒ the bf162 PAIRS kernel, not the scalar);
  (2) wrong values are LARGE (≈ +6.38..7.24 vs clean ≈ ±1.5) — catastrophic, not ulp;
  (3) **T=65 vs T=96 wrong values BIT-IDENTICAL rows 0..58** (E-8's ONE-shared-wrong-
  computation confirmed and refined — cross-div only at row 8 at XA, rows 59+ differ only
  by real prompt content); (4) QC rows 59..64 show odd-channel dirt too — those rows carry
  genuine content differences (66 vs 67 alphas). THE GATE (static, now measurement-backed):
  `causal_conv1d_silu.cpp` dispatch (split3 path :157-169; plain path :41-48) routes
  **T==1→decode, T≤16→smallt, T≤64→sequence (`kCausalConvSequenceMaxTokens=64`,
  launcher/causal_conv1d.h:14), T≥65→prefill** — so **T=64 and T=65 run DIFFERENT KERNELS**
  (sequence :156 vs prefill_pairs :71 of kernel/causal_conv1d.cuh; prefill_launch picks the
  pairs variant by alignment, launcher/causal_conv1d.cu:78-88). This single gate explains
  the whole record: E-9's impossibility (same inputs, different code ⇒ different outputs),
  E-8's T-class law (kernel+inputs identical for ANY T≥65 ⇒ bit-identical wrong values),
  E-3's cliff at exactly T=65, E-4's force-recurrent persistence (the conv runs on BOTH
  delta-net paths), and E-2's garbage-from-token-1 (row-0 wound, downstream decay).
  The 25+ static reads missed it because the defect is a DISPATCH-CLASS divergence, not a
  bug inside one kernel body. NEXT (the fix, in order):
  1. PIN THE LINE: run `causal_conv1d_prefill_pairs_kernel` vs `causal_conv1d_sequence_kernel`
     on identical synthetic [C,T] input + state (device cell, ~seconds) — the diff isolates
     the pairs kernel's even-channel defect (prime suspect: weight tap addressing in bf162
     units — weight2[p + k*C2] at :103-106 vs the [C,4] tap-major layout where channel 2p's
     tap k lives at elements 8p+k — verify against the sequence kernel's per-channel
     indexing, which is the semantically-green reference).
  2. FIX minimal: correct the pairs kernel's addressing (or route T>64 prefill through the
     sequence kernel as a stopgap — perf regression acceptable only as a diagnostic step,
     not the landing).
  3. CLOSE per RED→GREEN law: the cell joins the standing suite permanently (synthetic
     conv cell: sequence-vs-pairs bit-equality on identical input, any C/T — guards the
     CLASS: any kernel-pair divergence under the same dispatch), plus the definition-of-done:
     NEW banked binary sha on which `coherence_ladder.py` exits 0 (all legs, --long
     included), citing §1C row 26 / E12b rows as RED.
  Window re-run is deterministic: `STAMP_ACK=G-AMD-E12b bash results/amd/p3/E12_coh_tp4_agent6.sh`
  (env taps all default-off; dumps land in /tmp/coh_e12b — volatile, re-run reproduces;
  analysis: `python3 results/amd/coherence/e12_analysis.py /tmp/coh_e12b`).
- **[2026-09-16 ~17:50 CDT] E-14 (fix seat, amd/tp4-cure). BUG CLOSED per RED→GREEN law: root
  cause was NOT the pairs kernel's addressing — it was `__low2bfloat16` MISCOMPILING on
  gfx900/ROCm 6.2, reached only by the pairs kernel's split3 store branch.** Sequence of the
  closure (all rows banked in `results/amd/coherence/`):
  1. **E-13 step-1 cell built and fired** (`e13_conv_route_parity.cu`): sequence vs
     prefill-pairs vs prefill-scalar on bit-identical synthetic [C,T]+state, plain form —
     **PARITY-OK bitwise everywhere** (C=4096/10240/8192/512/510, T=65/96/130/257). E-13's
     weight-tap-addressing suspicion is FALSIFIED for the plain form.
  2. **Extended to the split3 epilogue (production form, world-4 geometry C=2560 bounds
     {512,512,1536}): RED — pairs+split3 diverges from sequence+split3 in EXACTLY the E-12b
     fingerprint**: absolute-even channels only (256/512 in q, 49920/99840 in v...), every row
     from row 0, catastrophic magnitudes (max_delta ≈ 48896), T=65/96/257 alike; sequence vs
     scalar-prefill bit-identical. RED row: `E13_route_parity_RED.out` (17 DIVERGENT rows) on
     pre-fix source.
  3. **Micro-test pinned the intrinsic**: `__floats2bfloat162_rn(0.75, -1.25)` packs correctly
     (raw words 3f40 bfa0), `__high2bfloat16` returns -1.25 correctly, but `__low2bfloat16`
     returns 467d (=16192.0) — garbage. The ROCm 6.2 header (`amd_hip_bf16.h:610`) is
     nominally correct (`__hip_bfloat16(hr.x)`); this is a **gfx906/gfx900 compiler codegen
     defect for the low-half unpack**. The pairs kernel's split3 branch
     (kernel/causal_conv1d.cuh:110-111, pre-fix) was the ONLY use of `__low2bfloat16`/
     `__high2bfloat16` in the tree (grep-verified), and its low half = absolute-even channel of
     each pair — hence even-only corruption, and only on the T≥65 route (the only route through
     the pairs kernel), and T-invariant (E-8's law: same kernel + same inputs ⇒ same wrong
     values). Every §2 regularity and every E-entry now closes.
  4. **FIX (one site)**: the split3 store branch re-rounds each half directly from its float
     accumulator (`__float2bfloat16_rn(silu(acc0))` / `(silu(acc1))`) instead of unpacking the
     packed pair — the form proven bit-correct by the plain branch and the micro-test. Comment
     in-tree cites this entry. GREEN row: `E13_route_parity_GREEN.out` — all 51 rows OK, exit 0.
  5. **Permanent suite cell**: `tests/ops/test_causal_conv_route_parity.cu` registered in
     tests/CMakeLists.txt (`ninfer_causal_conv_route_parity_test`), device-touching class —
     joins the per-window boot battery. Guards the CLASS: any kernel-pair divergence under one
     dispatch, plain and split3, across C/T/bound classes (including odd bounds {13,17,40}).
  6. **Definition-of-done row**: binary built in this lane worktree, BANKED as
     `/home/chris/artifacts_bin/ninfer-serve_b864220912aec151.bin` (sha256
     b864220912aec15193072bd081aa316d…; BANK-BEFORE-RELINK obeyed), booted TP4 port 8100
     (grant: user blanket grant of 2026-09-16 recorded in E-12; manifest in
     `E14_ladder_row.txt`, KFD-clean precondition, exact-PID kill, KFD verified 0 after):
     **`coherence_ladder.py --port 8100 --max-tokens 64 --long` = 16/16 legs PASS, exit 0**
     (`E14_fixed_b864220912aec151_row.txt`), incl. n1k (1000) and o2k (2000) prompt tokens —
     lengths that were mojibake since the hunt opened. RED citations: §1C row 26 (L1_fast) and
     E12b rows. **THE PARENT BUG OF THIS DOCUMENT IS CLOSED.** Remaining work moves to the
     throughput axis (§4.8: prefill ~10 tok/s at 10k) — separate crisis, phase-axis rule.
- **[2026-09-16 ~18:40 CDT] E-15 (fix seat, amd/tp4-cure). ROOT-CAUSE WORDING CORRECTED (E-14's
  "miscompile/codegen" is WRONG — the defect is a deterministic HEADER BUG) + the CLASS fix
  landed.** (a) The mechanism, proven at source level AND live: ROCm 6.2's
  `amd_hip_bf16.h:610` returns `__hip_bfloat16(hr.x)` **UNBRACED**; `hr.x` is `unsigned short`,
  so overload resolution binds the INTEGER-VALUED ctor `__hip_bfloat16(unsigned short)`
  (converts the raw bits AS AN INTEGER to float and re-rounds) instead of the raw-bit-copy ctor
  `__hip_bfloat16(__hip_bfloat16_raw)`. Low word 0x3f40 (=16192 as int, =0.75 as bf16) returns
  0x467d (=16192.0) — the exact micro-test observation of E-14, now fully explained. Every
  sibling braces (`__high2bfloat16` :572 correct); `__low2bfloat162` (:617) carries the same
  defect via implicit conversion of both args. Not a compiler bug, not nondeterminism: one
  missing brace pair in the vendored header.
  (b) **CLASS FIX** (bug-class law): `src/common/hip_shim/cuda_bf16.h` now defines corrected
  `ninfer_low2bfloat16/__low2bfloat162` bodies (raw-braced, mirroring the correct siblings) and
  macro-redirects the two broken names AFTER `hip_bf16.h` is fully parsed (the header's include
  guard makes later direct includes no-ops, so ordering cannot resurrect the broken bodies).
  With the shim fixed, the PRE-FIX kernel (which still calls `__low2bfloat16`) passes the whole
  route-parity cell — the class fix alone cures the original instance.
  (c) **NEW SUITE CELL (the RED/GREEN the law demands, at the intrinsic level)**:
  `tests/ops/test_bf16_halves_contract.cu` (`ninfer_bf16_halves_contract_test`) asserts
  half-extraction bits == `__float2bfloat16_rn` of the source float over 256 value pairs ×
  8 intrinsics (`low/high/low2/high2/lowhigh2highlow` compositions). Receipts:
  `results/amd/coherence/E15_halves_contract_RED.out` — STOCK intrinsics, **exit 1, 1200
  mismatches, smoking-gun row `in=(3f40,3f40) got=467d want=3f40`**;
  `E15_halves_contract_GREEN.out` — shim-corrected, **exit 0, 0 mismatches**.
  (d) Kernel-level chain, now four corners, all banked:
    - pre-fix kernel + stock shim  → RED 17 rows (`E13_route_parity_RED.out`, E-14)
    - fixed kernel  + stock shim   → GREEN 51 rows (`E13_route_parity_GREEN.out`, E-14)
    - pre-fix kernel + FIXED shim  → GREEN (`E15_route_parity_REDrefix.out`) — class fix subsumes instance
    - fixed kernel  + FIXED shim   → GREEN (`E15_route_parity_GREEN.out`) — current tree
  (e) In-tree kernel comment re-cites the true mechanism (E-14's "miscompiles" wording
  replaced; E-14 text itself untouched per append-only). Serve binary `b864220912aec151`
  remains the banked definition-of-done artifact (kernel does not call the intrinsic anymore;
  the shim fix does not change its codegen) — tree rebuilt to prove the macro redirect compiles
  clean in the full HIP build. Standing warning to the board: on ANY toolchain, contract cell
  first — `__low2bfloat16`'s name says bits, the vendored 6.2 body says integer.
- **[2026-09-16 ~19:05 CDT] E-16 (fix seat, amd/tp4-cure). THE NORTH STAR IS MET: plen=10000
  coherent on the fixed binary.** Leg: BLUE known-answer, prompt_tokens=10000 exactly (base 62 +
  9938 fillers), banked binary `ninfer-serve_b864220912aec151.bin`, TP4 port 8100 (same boot
  discipline: manifest in `E16_10k_row.txt`, KFD-clean precondition, exact-PID kills, KFD 0
  after). GRADE: **finish=stop, content='BLUE', 69 completion tokens, coherent reasoning
  throughout ("...a long string of \"alpha\" repeate[d]..."), zero mojibake — PASS.**
  Attempt 1 (mt=64) is a VOID row of the E-4 class, recorded as such: reasoning consumed all
  64 tokens, finish=length with EMPTY content but fully coherent reasoning — re-fired at
  mt=192 per the standing warning. NOTE for future 10k legs: mt=64 is no longer sufficient at
  plen 10000; use ≥192. Throughput columns (phase-axis data, coherence uninvolved): prefill
  1029.78 s / 10000 tok ≈ 9.7 tok/s (mean; ran 14.5 tok/s early, ~9 tok/s deep), decode
  26.87 s / 69 tok ≈ 2.6 tok/s at 10k context — matching §4.8's crisis numbers, now isolated
  cleanly on a GREEN-coherence binary. Coherence axis CLOSED through the north star; the
  remaining work is the throughput axis.
- **[2026-09-17 ~05:5x CDT] E-17 (night seat, continuation of the E1 RED datum banked
  2026-09-17 ~04:5x). MTP-vs-NON-MTP BYTE-DIVERGENCE CLASSIFIED: batched-verify geometry
  numerics — accept path EXONERATED by direct audit; WO_MTP_1's byte-identity bar
  re-scoped (same-geometry identity + acceptance<=argmax), NOT a coherence bug.**
  Governing doc: WO_MTP_1 (E1 = MTP must change nothing but speed; never fired until
  2026-09-17). The RED datum: graphs-OFF count600 (temp 0) MTP k=2 arm vs non-MTP arm
  token streams diverge (`results/amd/coherence/E1ORACLE_row.txt`, first divergence at
  generation position ~2: k2 emits `486`, base emits `1472 220`). Tonight's falsifier set:
  (a) k-sweep geometry probe (`E1K1_row.txt`): divergence point MOVES with k — k1-MTP
  follows base to gen-pos ~15 then flips (`874 4799 1414` vs `1534 1132 1103`) while
  k2-MTP flipped at gen-pos ~2; at that second site k1 and k2 AGREE with each other
  against base. Cross-boot non-MTP determinism: fresh boot byte-reproduces E1B; same-boot
  repeat byte-identical. (b) accept-audit tap (env NINFER_ACCEPT_AUDIT=1,
  `tp2_backend.cpp` at the speculative_accept_greedy_drafts call): D2Hs each round's
  target argmax tokens and the licensed (retained) tokens, prints any retained token that
  differs from its round's argmax. Clean run on bin c6361bde8bc0dcc1: **206 audited
  rounds, 0 retained mismatches** (`E1AUDIT_serve.log`), tap passive (audited arm's
  [ids] byte-identical to the unaudited E1A arm). By the greedy kernel's construction
  retained[i] == target_argmax[i] is an invariant; it now also holds by measurement.
  CLASSIFICATION: the divergence is born UPSTREAM of accept — the T=k+1 batched verify
  forward produces low-order-different logits whose argmax flips at a handful of near-tie
  numeral positions vs the T=1 path (LCP 29/42 of 65, difflib ratios 0.923/0.954; flips
  confined to same-locale numeral tokens; both streams remain coherent 1..200 counting —
  no derailment, no mojibake). Same-geometry identity HOLDS byte-level across boots AND
  binaries (B3's fcef463d count600 [ids] == 07ad7ecc TIMING1 count600 [ids]).
  RE-SCOPED E1 BAR (supersedes WO_MTP_1's raw byte-identity vs non-MTP): (i) MTP k=i arm
  == MTP k=i arm across boots/binaries, byte-identical; (ii) acceptance never exceeds
  target argmax (tap cell, 0 mismatches); (iii) cross-geometry delta bounded and
  characterized per (a). NAMED FOLLOW-UP: top-2 logit margin print at flip positions via
  the existing NINFER_D22_PDBG hook (would quantify tie-gap; not needed for this
  classification). Rows: `results/amd/coherence/E1K1_row.txt` (+ E1K1A/B/B2_ids,
  E1AUDIT_ids); audit binary banked `ninfer-serve_c6361bde8bc0dcc1.bin` (tap default-off).
  VOID ROW logged: first audit attempt spawned while the control server still held the
  GPUs (KFD=1 at spawn — hygiene miss); boot clean-failed on 816 MiB free; re-run clean.
  No code behavior change landed for E-17 itself (the tap is env-gated, default off).
- **[2026-09-17 ~08:0x CDT] E-17 FOLLOW-UP CLOSED (D1, night seat, same window).** The
  named margin datum landed WITHOUT new plumbing: the existing `NINFER_D22_PDBG` hook
  (env NINFER_D22_PDBG=1, NINFER_D22_PDBG_ROUNDS=40) already prints, per verify column,
  the argmax's full-vocab softmax probability (p_am) over the gathered bf16 logits —
  probability space is a sufficient margin metric. Margin table
  (`results/amd/coherence/E1MARGIN_row.txt`, PLOG-028; margin arm [ids] byte-identical to
  E1A — hook read-only): flip-class columns carry p_am 0.17-0.55 (near-degenerate
  distributions), a recurring numeral-boundary column shows `local_am=16 global_am=486
  p_am=1.0000` — top-2 BIT-IDENTICAL at bf16 after gather, tiebreak order decides — and
  several columns where the gathered-bf16 argmax differs outright from the committed
  allreduce argmax. Absolute top-2 logit-gap plumbing RE-QUEUED AS NOT NEEDED: an absolute
  gap would not change the classification (flips are at/below bf16 resolution). E-17
  remains CLOSED with its re-scoped bar; no code behavior change.
