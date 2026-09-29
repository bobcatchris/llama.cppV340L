# WORK ORDER: F1b drafter-quality design (decode acceptance lever — design + spec only)

Owner: desk agent (design desk — NO src edits, NO builds, NO serving window except light
probes; the decode-landing desk owns the heavy window). Coordinator owns infrastructure.

## Mission
Acceptance is the decode multiplier: measured 0.65-0.96 by register (PLOG-066: 0.96 cold
counting-class, 0.71 at 10k soak, tok/round 2.45-2.91). The vocab slice is NOT the
suppressor (coverage 97.6-99.4%, PLOG-066-class receipts). The suppressor is the DRAFTER:
MTP-head quality + quant argmax flips. Design the highest-value intervention and spec it to
implementation grade.

## Steps
1. MAP THE DRAFTER: grep the MTP head path (src/targets/qwen3_6/ + src/runtime/tp2/) —
   the draft head's architecture (1 layer?), its quant format (same NVFP4 weights? which
   dials?), how draft logits -> proposals (top-k? argmax? temperature), and where acceptance
   is decided (verify pass). Bank the map with file:line references.
2. ERROR ANALYSIS DESIGN (not execution): spec HOW to measure where acceptance is lost —
   (a) drafter-argmax vs target-teacher-forced comparison on N sampled positions (how often
   does the drafter's top choice differ from what the verifier accepts, and by what margin —
   near-miss vs wrong-register), (b) per-register acceptance split (prose vs reasoning vs
   code vs counting — the 0.94-vs-0.69 spread says register matters), (c) quant-flip probe:
   if the draft head weights dequant to bf16 in a cell, does acceptance move? Price each.
3. OPTIONS TO SPEC AND PRICE (work order grade each — steps, gates, effort):
   a. Draft-head precision bump (the head is 1 layer — dequant-to-bf16 cost is tiny; price
      the ms/round cost and the expected acceptance delta).
   b. Reasoning-register vocab top-up (the reasoning register spends wider vocab: 96.28% vs
      content 98.95% — price the slice edit + VRAM).
   c. Acceptance-aware draft selection (draft top-2 fallback, confidence threshold).
   d. Anything the code map reveals as cheaper (e.g., a drafter scale/zp bug).
4. RECOMMEND + pre-register gates for the winner (acceptance delta bar, decode t/s paired
   A/B design per PLOG-060, no-prefill-impact proof).

## Laws
- THIS file is the checkpoint; append "## PROGRESS LOG" (newest-first) after every step.
- Light probes against :8100 allowed (the decode-landing desk owns boots — if its server is
  mid-swap, wait 5 min and retry; do NOT boot or kill anything).
- No builds, no src edits. Infrastructure = tag "BLOCKER:" and continue.

## DRAFTER MAP (F1b step 1 — banked 2026-09-19, branch amd/wo-w7-body, all paths relative to repo root)

### Identity
- The drafter is the **MTP-1 head**: ONE transformer layer (`TextConfig::mtp_layers = 1`,
  `src/targets/qwen3_6_27b/impl/config.h:42`) + a sliced lm_head proposal head. Tree-free
  sequential chain: d0 from the align hidden, then rk-1 autoregressive (AR) steps.
- Served shape (W7 windows, bin era c618d356/2c8901d3): TP4, MTP k=2, 64 layers
  (16 full-attn + 48 GDN), hidden 5120, output_rows 248320 (`config.h:12-17`).
- Draft depth: k from `--draft-tokens` (served k=2); buffers to `k_buf` capped at
  `kD21MaxDraftDepth = 7` (`src/runtime/tp2/tp2_backend.cpp:733-754`); verify width
  T = rk+1 <= 8 (`src/ops/wrapper/mtp_round.cpp:57-59`; `kMaximumMtpDraftTokens = 7`,
  `src/targets/qwen3_6/impl/runtime/instance.h:40`).

### Decode round flow (single-seq loop, `src/runtime/tp2/tp2_backend.cpp` run_tp2_requests)
1. prepare verify inputs `:3037-3041` (padded cols carry TRUE positions — docs/130 §8.9 fix).
2. verify forward `target_verify_batch` `:3048-3056`; per-rank logits shard
   `verify_logits` [248320/tp_w, rk+1] (alloc `:1341`, `n_vocab = 248320 / tp_w` `:1242`);
   **LGATHER allgather to `verify_logits_full` [248320, rk+1] runs UNCONDITIONALLY on the
   MTP path** `:3051-3066` (pair-era `2*nv` byte-stride comment = GATE-3 capacity note);
   target argmax per column `allreduce_argmax` `:3164`.
3. accept `speculative_accept_greedy_drafts` `:3179-3183` — kernel
   `src/ops/kernel/speculative_round.cuh:86-212`: greedy branch `:104-121` = longest draft
   prefix whose target ARGMAX matches + correction token at the divergence column (bit-exact,
   reads target_tokens only, never logits); sampling branch `:149-192` = truncated rejection
   sampling over the full logits, bonus from last column. Per-request `sample_cfg`
   (greedy default `:1269-1274`, overwritten per request `:1902`) SELECTS the branch —
   temp>0 requests DO take the sampling branch.
4. accept D2H + the round's ONLY hard stream sync `:3197-3205`.
5. align forward `mtp_forward_decode_batch` `:3538-3547`; select accepted hidden
   `:3552-3555`; **propose d0** `:3559-3572` = `launch_draft_head` + `allreduce_argmax`
   with `dev_draft_vocab` remap; **AR steps d1..d_{rk-1}** `:3620-3666` (each =
   mtp_forward_decode_batch -> launch_draft_head -> allreduce_argmax).

### Proposal path (draft logits -> token)
- Draft head = **runtime row-slice of the full W8G32 lm_head**, NOT the artifact's
  `text/draft_head`: JSON 40960 ids `tests/multi_gpu/data/qwen38_draft_vocab_ids.json`
  (`:1081-1105`; path resolution + loud-fallback guard `src/runtime/tp2/tp_engine.cpp:1071-1155`
  — missing file degrades to FULL lm_head ~3x propose cost); slice
  `n_draft_local = 40960/tp_w = 10240` rows/rank `:1107-1226` — default `draft_head_quant="w8"`
  (`tp2_backend.h:222`) raw-copies W8 rows (53 MiB/rank, matches boot line), `"q4"` arm
  dequant int8->float->requant Q4G64 `:1114-1181` (LOWER precision — see OPTIONS d3).
- GEMV `launch_draft_head` `:1369-1394` (`launch_w8_simt_r8_c4` / `launch_q4_simt_r8_c4` /
  full lm_head fallback) -> `allreduce_argmax` (`src/core/multi_gpu/tp_group.cpp:458`)
  fuses cross-rank argmax + row->global-id remap.
- **Proposal = pure argmax; no temperature, no top-k anywhere on the draft path**
  (re-confirms ACCEPTANCE_LEVERS (a)).
- Target-side twin `TextContext::proposal_argmax`
  (`src/targets/qwen3_6/impl/runtime/text_context_impl.h:2007-2032`) uses the artifact
  `optimized_proposal` head (Q4G64, 131072 ids, `src/targets/qwen3_6_27b/impl/load/bindings.cpp:535-541`)
  or full lm_head — NOT on the tp2 serve path (tp2 uses launch_draft_head).

### MTP head internals (`text_context_impl.h`)
- Stem `mtp_forward_stem :1484-1520`: embedding(ids) -> rmsnorm(pre_fc_norm_embedding) +
  rmsnorm(pre_fc_norm_hidden) -> concat [e;h] (`mtp_pack_fc_input`) -> fc linear
  [5120 <- 10240] -> input_norm.
- Tail `mtp_forward_tail :1522+`: TP GQA attention (packed qkv; q_norm/k_norm; ONE MTP KV
  layer, `layer_view(0)`; rope; sigmoid_mul gate) -> o_proj -> AR/residual ->
  post_attention_norm -> post_mixer MLP (TP row-shard + AR) -> final_norm -> propose.
- Decode entries: `mtp_forward_decode_batch :2344`, `mtp_propose_batch :2375`;
  capture-graph batch body `src/targets/qwen3_6/impl/runtime/mtp_impl.h:70-183`.

### Quant formats (which dials)
- MTP head matmuls are **W8G32 int8, unconditionally** (`bindings.cpp:550-569`): fc
  {5120,10240}, qkv packed {14336/tp_w,5120}, o_proj {5120,6144}, gate_up {34816,5120},
  down {5120,17408}; all norms BF16. The MTP head is NOT NVFP4 even when the backbone is.
- Backbone: NVFP4 blockscale + Q4/Q5 groupwise mix (`bindings.cpp:271-315`); endpoint
  lm_head W8G32 for the served profile (`endpoint_format`, `bindings.cpp:38-48`).
  Draft head slice inherits the W8 head rows verbatim (no NEW quant error in the w8 arm).

### Acceptance accounting + existing measurement taps (zero-build instruments)
- stats `:3766-3774`: `acceptance_rate = acc_accepted/(rounds*k)`,
  `tokens_per_round = gen/rounds`, `mean_a_per_round`.
- FirstMissHistogram (`src/runtime/tp2/mtp_confidence_break.h:110-160`), fed `:3407`
  (MTP rounds only), printed via `NINFER_VERIFY_DEPTH_LOG` `:3775+` — per-depth first-miss
  distribution + exact E[saved].
- VocabOutputCounter (`src/runtime/tp2/vocab_output_counter.h`; init `:1660`, add `:3410`)
  — `[vocab] coverage=` per 10k accepted tokens under `NINFER_VOCAB_COUNT_DIR`.
- **NINFER_D22_PDBG classifier** `:3101-3158` (env-gated, rank 0): per verify column prints
  `local_am`, `p_am` (target prob of argmax), `global_am`, `draft`, `p_draft` (target prob
  of the draft) computed over the FULL allgathered logits — legal on greedy serves because
  LGATHER is unconditional. Plus PDBG-COUNT drafts-outside-local-shard counter.
- NINFER_ACCEPT_AUDIT `:3205-3252` — retained-vs-target-argmax misaccept audit (C1b).
- Existing adaptive machinery: `mtp_adaptive.h` (depth hysteresis),
  `mtp_confidence_break.h` (CONF_TAU confidence-product chain break; wiring `:3560-3572`,
  `:3664-3669`), ngram-mod pool (`mtp_ngram_mod.h`, replaces drafts `:2913-2930`).

### Measured anchors (cited, not re-derived)
- Round budget cold p50 59.2 ms (PLOG-066): verify-loop bodies 52.6 (87%) + in-loop AR
  6-9.6 + lm_head 2.36 + align fwd 2.03 + chain/propose/head 1.98 + micro 1.05 + host 0.21;
  kv 34,816 B/token. Thermal doctrine: 59 cold <-> 106 hot ms (PLOG-064/066).
- Acceptance: counting-class cold 0.96 / tok/round 2.91 / 49.3 tok/s (PLOG-066); prose
  0.65 / 2.31 / 10.79 (REALK23); 10k soak 0.71; k=3 on prose LOSES (0.51, 9.75 tok/s).
- Coverage (F1a live-instance receipts, `results/amd/w7_f1a/coverage_report.md`): overall
  97.597% +-0.112; **reasoning segment 96.277% vs content 98.954%** -> >=3.7% of reasoning
  target-argmaxes are UNPROPOSABLE (hard ceiling per position).

### MAP FINDING d1 (bug candidate, exact site): world=4 sampling-accept domain is HALF
- `:3182` passes `2 * st.verify_logits.ne[0]` (= 124160 at world=4) as `token_domain` to
  the accept kernel; the batched-runner sites `:4470` and `:6954` pass
  `backend.world() * nv_l` (= 248320) — the world-general form. `2*nv` is the pair-era
  literal: correct at world=2, HALF the vocab at world=4.
- Effect (sampling branch only): drafts with global id >= 124160 get p_draft = 0 ->
  spurious reject + resample; correction/bonus draws exclude the upper half of the vocab.
  Output-distribution incorrectness (the branch is documented "distribution-correct"),
  not a crash. **Greedy branch unaffected** (returns at `speculative_round.cuh:104-121`
  before touching logits) — every banked probe (all greedy) is clean.
- Fix shape: one line (`tp_w * n_vocab`, matching the batched sites) + RED/GREEN cell
  per closure law (see OPTIONS d1). Not fired from the design desk.

## ERROR-ANALYSIS DESIGNS (F1b step 2 — SPEC ONLY, not executed here)

### E-1 Flip anatomy: drafter-argmax vs verifier-accepted, per position (PRIMARY, zero build)
- Question: at each verify column, WHY is the draft not accepted — (i) OUT-OF-SLICE
  (target argmax not in the 40960-id draft vocab: unproposable, drafter blameless),
  (ii) NEAR-MISS / flip-able (draft is the target runner-up or within margin eps of the
  argmax under TARGET logits: a precision/numerics bump could flip it), or
  (iii) HEAD-MISS (p_draft ~ 0, rank of draft under target logits large: the MTP head
  itself is wrong — no in-tree precision lever can fix it).
- Instrument: `NINFER_D22_PDBG=1 NINFER_D22_PDBG_ROUNDS=<N>` on the STANDARD greedy serve
  line (map §taps: legal because LGATHER is unconditional). One leg per register class;
  parse `[PDBG]` lines (script, ~40 LOC: classify each column by the three classes above;
  margin m = p_am - p_draft and rank of draft under the full logits — both already printed).
  Cross-check drafts-outside-local via the same tap's PDBG-COUNT counter.
- Sampling plan: N >= 2000 classified columns per register (counting mt=600 gives ~1200
  rounds x 2 cols; prose REALK ~4300 cols; reasoning + code prompt sets same grade).
- Pre-registered decision rule: class-(ii) share >= 50% of MISSES in a register -> precision
  lever (a1/E-3) has headroom there; class-(i) >= 30% -> vocab top-up (b) for that register;
  class-(iii) dominant -> drafter quality is the ceiling, only (c)-class or d2 levers remain.
- PRICE: 0 build; 1 window leg (4 register probes x3, ordinal-paired per PLOG-060);
  ~1.5 desk-hours incl. parser. Delivers the denominator every option below is priced against.

### E-2 Per-register acceptance split (co-spec with E-1, same legs)
- Question: where is acceptance LOST by register — the 0.96 (counting) vs 0.65 (prose)
  spread, plus reasoning/code not yet cleanly split.
- Instrument: per-request stats block (map: acceptance_rate / tokens_per_round /
  mean_a_per_round printed per request) + `NINFER_VERIFY_DEPTH_LOG` first-miss histograms
  per leg + `NINFER_VOCAB_COUNT_DIR` coverage lines per leg (coverage % per register —
  reproduces the 96.28-vs-98.95 split mechanically). Register = the probe set's prompt
  class; no code changes (stats are per request; one class per leg).
- Design law: ordinal-paired boots (PLOG-060), start-temp recorded (thermal doctrine,
  PLOG-064/066) — acceptance deltas quoted within-pair only.
- PRICE: rides E-1's legs (same boots); ~0.5 h extra analysis. Output: acceptance x
  first-miss-histogram x coverage per register = the option-pricing table.

### E-3 Quant-flip probe: does dequant-to-bf16 move the drafter's argmax? (prices option a)
- Question: of the class-(ii) columns E-1 finds, how many does an fp16-precision draft head
  actually flip — and does acceptance move?
- Design: env-gated bf16 arm in `launch_draft_head` (`:1369-1394`): dequantize the W8 slice
  rows to bf16 at slice-build (one host loop variant beside `:1183-1226`) + a bf16 GEMV
  route; `NINFER_DRAFT_HEAD_BF16=1` arms it, unset = byte-identical legacy. ~20-30 LOC,
  one function + one env read, no new refusal paths.
- Two-sided falsifier: (a) OFF arm must be byte-identical to control boot on a fixed
  prompt (greedy); (b) ON arm flip-rate measured by E-1's tap on the same prompts —
  flip-rate ~ 0 falsifies the quant-flip hypothesis cheaply and closes option a.
- PRICE: 0.5 desk-hour edit + ~15 min build + 2 fresh boots x 3 probes (paired). NOTE:
  the build/serve leg belongs to the decode-landing desk's window — this desk hands the
  spec over; design-desk cost ends at the edited-cell spec.

## OPTIONS — SPEC + PRICE (F1b step 3; modeled figures marked (m), measured cited)

### (a) Draft-head precision bump — TWO SCOPES
- **a1: proposal-head slice W8 -> bf16** (53 -> 106 MiB/rank; +53 MiB/rank VRAM = KV
  -1,593 tokens at 34,816 B/t; KV capacity is auto-computed from the measured manifest —
  VRAM-LAW-compliant, no refusal constants). Compute (m): propose GEMV reads double the
  bytes: +~0.2 ms x 2 proposes/round = +0.4 ms/round (+0.7% of the 59.2 ms cold round;
  byte-rate anchor 265 GB/s from ACCEPTANCE_LEVERS (b)). Break-even: tok/round +0.7%
  relative. Expected upside = E-3 flip-rate x P(flip lands on the accept path) — DELIBERATELY
  NOT claimed before E-1/E-3 run. PRICE: the E-3 cell itself + 0.5 h; gate G-A below.
- **a2: whole MTP layer W8 -> bf16** (+106 MiB/rank = KV -3,186 tok; (m) +1-2 ms/round
  from doubling the ~158 MB/rank the head reads per pass x ~2 passes — +2-3% round).
  Only fires if E-3 shows the HEAD is clean but the LAYER owns the flips — requires a
  deeper hidden-state tap (not spec'd until E-3 says so). PRICE: 1-2 desk-days + cells.
  A priori disfavored: the layer runs on bf16 activations with per-group int8 scales —
  the same numerics class the backbone already runs at Q4/Q5 without an acceptance anomaly.
- Verdict: a1 = cheap, gated, worth its cell; a2 = parked pending E-3 evidence.

### (b) Reasoning-register vocab top-up
- Slice edit is DATA, not code: superset JSON (40960 -> 65536 or 98304 ids) rebuilt from
  the banked `VocabOutputCounter` flush bins + `tools/collect/project_vocab.py` merge,
  stratified so the reasoning register's measured out-of-slice argmax ids enter the slice
  (the out-of-slice set is exactly what E-1 class (i) + the coverage bins name).
- Price: rows/rank 16,384 (65536) -> ~89 MiB (+36 MiB, KV -1,082 tok) or 25.6k (98304) ->
  ~134 MiB (+81 MiB, KV -2,437 tok); propose GEMV (m) +0.15 / +0.35 ms/round respectively.
  Ceiling lift: reasoning-register coverage 96.28% -> ~99% IF the top-up targets the
  measured misses; end-to-end bounded by the reasoning-round share of the mix.
- Gate: fire ONLY if E-1 shows class-(i) >= 30% of reasoning-register misses (the WO's own
  framing — "the vocab slice is NOT the suppressor" — says overall it will not clear this
  bar; the reasoning register is the one place it might). PRICE: 2-3 h data work + 1
  paired window leg; zero src edits.

### (c) Acceptance-aware draft selection
- **Top-2 / tree drafting: FALSIFIED AT DESIGN LEVEL BY ARITHMETIC** (same kill class as
  A4-fusion). The verify bodies own 87% of the round (52.6 ms for T = rk+1 <= 3 columns):
  one extra verify column (m) ~ +17.5 ms; the chain step it might save is (m) ~ -1.0 ms
  (1.98 ms / 2 steps). Backwards by >16x at the served geometry; no gate can rescue it.
- **Confidence-threshold selection (CONF_TAU): real, in-tree, LOW at k=2** — max save is
  one chain step/round; lossless by construction (prefix proposals, byte-identical stream).
  The tau sweep prices itself OFFLINE from banked FirstMissHistogram logs (doc 45 §6.1
  bound -> exact E[saved]); zero GPU. Do not spend a window; revisit only for a >=0.8-
  acceptance class where depth has room (counting already 0.96).
- PRICE: tau sweep 1 h desk (offline); window legs 0.

### (d) Code-map-revealed items (cheaper than all of the above)
- **d1 (BUG, map finding): world=4 sampling-accept token_domain HALF** (`:3182` vs the
  world-general `:4470`/`:6954`). Fix = one line + cell. RED: world=4 temp>0 MTP request
  vs world=2 reference, same seed/prompt — accepted stream diverges when a sampled/corrected
  token id >= 124160 (deterministic fixture: force a draft from the upper half). GREEN:
  post-fix streams match; cell joins the boot battery (device-touching arm). Correctness,
  not perf — no tok/round claim. PRICE: 0.5 h + 1 cell.
- **d2 ngram-mod** (`--mtp-ngram-mod`, host-only, in-tree): already priced (ACCEPTANCE_LEVERS
  (c)) — cheapest live lever for counting/repetitive registers; composes with everything;
  dormant in the batched loop (known limitation). PRICE: 1 paired leg, no build.
- **d3 the `"q4"` draft-head dial is an ANTI-lever** (`:1114-1181`): requantizes int8 -> int4,
  strictly more quant noise in the exact term (drafter logits) the suppressor hunt targets.
  Marked NEVER-FIRE; recorded so no future desk "discovers" it.
- **d4 boot-file guard** (`tp_engine.cpp:1071-1155`): already loud (G5); no action.

## RECOMMENDATION + PRE-REGISTERED GATES (F1b step 4)

### Ranked
1. **E-1 + E-2 (flip anatomy + register split) — fire first, zero build.** Every other
   option's price is conditioned on its output. One window leg, ~2 desk-hours.
2. **d1 (sampling-accept domain fix) — fire independently on correctness law**, one line +
   RED/GREEN cell per closure law (three shas: pre-fix repro, post-fix pass, suite entry).
3. **a1 (head-slice bf16) — the implementation bet**, conditional on E-3's flip-rate gate
   (fires only if bf16 flips >= ~2% of draft positions AND E-1 class-(ii) >= 50% of misses
   in prose/reasoning).
4. **b (reasoning vocab top-up)** — conditional second, gated on E-1 class-(i) >= 30% in
   the reasoning register.
5. c/top-2 FALSIFIED (arithmetic); c/CONF_TAU dormant (offline pricing only); a2 parked;
   d2 opportunistic; d3 never.

### Pre-registered gates for the winner path (a1, then b)
- **G-A acceptance delta bar**: promote only if paired acceptance rate rises >= +2.0 pp
  absolute on the target register AND tok/round >= +1.5% relative (a1) / +2.0% relative (b)
  — each above its modeled break-even (+0.7% / +0.6%) with margin. Below bar: bank the bin,
  close the option with the receipt, no third attempt without new evidence.
- **G-B paired A/B (PLOG-060)**: same banked bin both arms, fresh boot per arm, 3x mt=600
  probes per arm per register, within-pair +-2%; decode t/s from the [tp2] engine lines;
  start-temp recorded per leg (thermal state function — absolutes without start-temp are
  quarantined per PLOG-066 doctrine).
- **G-C no-prefill-impact proof**: prefill wall + TTFT within +-2% of control on the same
  paired boots (a1 touches decode-only tensors; the MTP prefill chunk path shares the
  layer but not the slice — G-C verifies that claim rather than assuming it).
- **G-D VRAM law**: the +53/+81 MiB/rank is charged in the measured manifest
  (drafter_fixed_bytes seam / slice-build alloc), KV capacity delta READ OFF the boot line
  and recorded; no refusal constant may appear anywhere in the diff (reviewer rejects on sight).
- **G-E byte-identity falsifier (a1)**: greedy output must be byte-identical to control
  when no drafter argmax flip occurs (E-1 tap cross-check on the same prompts); any diff
  not attributable to a measured flip is a defect, not a win.

## PROGRESS LOG (newest-first)

### 2026-09-19 — F1b EXEC desk COMPLETE: d1 CLOSED (3 shas), E-1/E-2 anatomy banked, gates read
- **d1 CLOSED per closure law.** Fix live on amd/wo-w7-body @ e9d6565a2 (accept :3204 full
  domain + PDBG tap same-class fix). RED: guard rc=1 @ pre-fix tree 99bfac4ad58624e1 +
  control bin 9bc7d7e1f764e7cb e2e tap witness (PRE-FIX-BROKEN-TAP, 63.4% corrupted cols).
  GREEN: guard rc=0 + --selftest PASS @ post-fix tree; device cell ninfer_accept_domain_w4_cell
  PASS 6/6 on hardware (world=4 geometry, multiblock route, both-direction falsifier);
  fix bins fdd9b68e0cf14421 + 4fb2d1ee4ec73890 banked. Suite entries: guard -> zero-GPU
  farm step-0 class; device cell -> boot battery (CMake entry landed; standalone-built here
  since this tree has BUILD_TESTING=OFF).
- **INSTRUMENT CLASS-FIND (banked as its own lesson):** the pre-accept PDBG tap raced the
  async LGATHER — paired CURRENT drafts with ONE-ROUND-STALE logits (measured 47%-vs-96%
  counting mismatch; one-round shift was exact). Tap moved post-accept-sync + acc/lic
  outcome fields added; parser v2 implements the kernel's exact evaluation rule (accepted
  slots j<acc; miss = divergence column; TRUNCATED class). Validated: per-request slot
  acceptance matches the engine's own per-request accounting within ~1-2 pp.
- **E-1/E-2 ANATOMY (7218 evaluated slots, 4 registers x3 greedy legs, bin 4fb2d1ee):**
  counting 95.65% slot-acc / prose 65.74 / reasoning 90.77 / code 80.88. Miss taxonomy:
  HEAD-MISS dominates EVERYWHERE (81-92% of misses; drafter picks the target's runner-up,
  rank median 1-2, but at margin median 0.24-0.35). OOS 0-10.6%, NEAR-MISS 8-12%.
  **GATE b (vocab top-up): DOES NOT FIRE** (max OOS 10.6% prose vs 30% bar) — option b
  closed at gate level; the vocab slice is NOT the suppressor, now proven per-column.
  **GATE a1 (head-slice bf16): DOES NOT FIRE** (max NEAR 11.1% reasoning vs 50% bar) —
  a1 dead at E-1 gate level; E-3 bf16 flip-rate measurement MOOT for a1 (not worth its
  window; spec stays banked). The acceptance ceiling is the 1-layer MTP head's knowledge
  (class-(iii)); remaining cheap lever: d2 ngram-mod on counting-like registers.
- **E-2:** engine per-request acceptance counting 0.96/0.96/0.89 (matches PLOG-066 anchor
  0.96/2.91), prose 0.55/0.53/0.53, reasoning 0.87x3, code 0.73/0.71/0.74; tok/round
  2.91/2.11/2.75/2.41-2.47; first-miss histograms + [vocab] coverage=98.460% banked.
- **BONUS d1 A/B at temp=0.5** (BEFORE=9bc7d7e1 vs AFTER=4fb2d1ee, paired fresh boots):
  accounting AND streams BIT-IDENTICAL (~2900 tokens, all channels) — the bug's served
  effect at TP4 on these registers is nil (draft slice is 95% lower-half; 0/5784 accepted
  drafts were upper-half; divergence/bonus upper-half argmaxes 0.57%/0.29%). Correctness
  close stands on the class cells. ONE transient WEDGE observed on the pre-fix bin's first
  temp=0.5 request (unreproduced 2/2 on both bins afterward; receipt d1_ab/before_serve.log
  from the first attempt) — logged for the boot battery's sampling arm.
- **ROW BANKED:** results/amd/coherence/W7_f1b_row.txt (full tables + gate readings + shas).
- CLOSE: canonical restored via /home/chris/serve_10k.sh + health-verify (see final row).

### 2026-09-19 — F1b EXEC desk PAUSED per coordinator order (cap enforcement / window conflict) — RESUME CHECKLIST BANKED
- **PAUSE ORDER received ~10:55 CDT**: the WO_ONESHOT_ROOTLOOK desk now owns the serving
  window AND the build tree (instrumented boots). I stopped BEFORE any build, BEFORE any
  window claim write, BEFORE touching :8100. Nothing of mine is in flight.
- STATE AT PAUSE (all committed, branch amd/wo-w7-body):
  - d1 fix: `docs/amd/W7_F1B_d1_fix.patch` (accept site :3182 -> full domain + same-class
    PDBG tap :3122 fix + `rank=` field). Tree is PRE-FIX by law (reverted; verified).
  - Guard cell `tools/guards/check_accept_domain_pair_literal.py`: **RED captured on the
    pre-fix tree (HEAD 99bfac4ad58624e1, rc=1, findings P1:3122/P1:3182/PIN-1:3180/PIN-2:3122);
    GREEN proven on a patch-applied copy (rc=0)**. --selftest both-direction ready.
  - Device cell `tests/ops/test_accept_domain_w4_cell.cu` — syntax-verified CLEAN via hip
    -fsyntax-only (fixed unqualified SamplingConfig + Tensor-temporary/non-const-ref bind).
    CMake entry (append-only, after the E-15 block in tests/CMakeLists.txt:
    `ninfer_add_op_test(ninfer_accept_domain_w4_cell SOURCES ops/test_accept_domain_w4_cell.cu
    LIBRARIES ninfer_ops CUDA::cudart)`) NOT yet applied — apply at resume, before build.
  - E-1/E-2: `tools/v340l/w7_f1b_pdbg_parse.py` (validated on synthetic fixture),
    `w7_f1b_boot.sh` (canonical env + PDBG/DEPTH_LOG/VOCAB_COUNT; I_HOLD_THE_WINDOW guard),
    `w7_f1b_legs.sh` (4 registers x3: counting mt=600, prose/reasoning/code mt=1200),
    `w7_f1b_d1_ab.sh` (bonus before/after temp=0.5: BEFORE=183da007 pre-fix bin, AFTER=my
    d1 bin; D2-SS-STATS paired extraction, start-temp recorded).
- RESUME CHECKLIST (in order):
  1. Confirm WO_ONESHOT_ROOTLOOK closed + window free (its desk file + :8100 state).
  2. BUILD CLAIM row in docs/amd/WO_DECODE_LANDING.md PROGRESS LOG (desk inactive by then).
  3. Apply W7_F1B_d1_fix.patch (git apply) + CMake cell entry; build
     `cmake --build build-hip-amd --target ninfer-serve ninfer_accept_domain_w4_cell -j8`
     (cmake at /home/chris/opt/cmake/bin/cmake); df -h / first (12G free at pause — tight,
     re-check).
  4. BANK-BEFORE-RELINK: cp -p to /home/chris/artifacts_bin/ninfer-serve_<sha16>.bin; name
     the sha in the closing row (three-shas law).
  5. GREEN captures: guard --selftest on the post-fix tree (rc=0 expected) + run
     ninfer_accept_domain_w4_cell (device; exit 77=skip if no VRAM next to a live server —
     then run it in a server-free gap). Both receipts -> results/amd/coherence/.
  6. Window: I_HOLD_THE_WINDOW=1 tools/v340l/w7_f1b_boot.sh <my sha bin> -> E-1/E-2 legs
     (~40-60 min) -> bonus d1 A/B (183da007 vs mine, temp=0.5) -> restore canonical per
     /home/chris/serve_10k.sh + health-verify.
  7. Bank results/amd/coherence/W7_f1b_row.txt (anatomy table + gate readings + shas) +
     final PROGRESS LOG entry here.
- Decode-landing desk closed mid-pause-watch (adfa4fc6e/e3d705374/c9b9c402b): :8100 was
  restored canonical 2c8901d3, 10K PASS — the window was free; the ROOTLOOK claim re-takes
  it. My contamination worry is MOOT: their bin 183da007 was built 08:50, before my edit
  ever touched the tree (my edit window 10:12-10:20 had no build running).

### 2026-09-19 — F1b EXECUTION desk: d1 fix written + cells + E-1/E-2 prep (build queued behind decode desk)
- D1 FIX WRITTEN, then REVERTED FROM THE TREE under coordinator arbitration (decode desk
  holds the build claim and is building THIS tree): full diff banked as
  `docs/amd/W7_F1B_d1_fix.patch`. Contents: (1) the one-line fix — :3182 passes
  `(std::int32_t)st.verify_logits_full.ne[0]` (full domain, wrapper-contract-legal) instead
  of the pair-era `2 * st.verify_logits.ne[0]`; (2) SAME-CLASS fix in the E-1 instrument
  itself — the NINFER_D22_PDBG tap (~:3122) computed its "full-vocab domain" as `2*nv_l`
  too, which at world=4 read HALF of column 0 as column 1 (wrong stride) and softmaxed half
  the vocab (wrong domain); fixed to `verify_logits_full.ne[0]` + `DRAFT_OUTSIDE_LOCAL` flag
  re-based to the true local shard, + a `rank=` (1-based rank of the draft under the FULL
  logits) field appended to the [PDBG] line — the E-1 design assumed rank was already
  printed; it was not. Syntax-checked (-fsyntax-only via build compile db): clean.
- CLASS SWEEP (closure law, guard the class): all live `2 *` domain literals in
  tp2_backend.cpp are exactly :3182 (accept) + :3122 (PDBG tap); the other `2*` hits are
  slot arithmetic or GATE-3 capacity comments (intentional). Batched sites :4470/:6954
  already world-general.
- WORKSPACE SAFETY of the fix verified: accept workspace is pre-sized at :1278 with the
  FULL 248320 domain — post-fix demand exactly fits (pre-fix was under-using it); no new
  allocation, no refusal path.
- GUARD CELL (zero-GPU, joins farm step-0 class):
  `tools/guards/check_accept_domain_pair_literal.py` — P1 pattern (pair-era
  `2 * verify_logits.ne[0]` literals) + PIN-1 (accept call span must pass a world-general
  domain) + PIN-2 (PDBG nv pin). **RED CAPTURED on the pre-fix tree** (HEAD
  99bfac4ad58624e1): P1:3122, P1:3182, PIN-1:3180, PIN-2:3122, rc=1. **GREEN proven on a
  patch-applied copy** (rc=0). Honest scope note in header: textual, no dataflow — the
  semantic witness is the device cell below.
- DEVICE MECHANISM CELL written: `tests/ops/test_accept_domain_w4_cell.cu`
  (target `ninfer_accept_domain_w4_cell`, CMakeLists entry lands with my build claim —
  cell-addition only): world=4 served geometry (physical_rows=248320, k=2, batch=1,
  multiblock sampling route — NOTE: the d1 fix must be validated against the MULTIblock
  route (partial_topk + group_finalize), which is what token_domain=248320 actually
  dispatches to; the single-block kernel only serves domain<=kSamplerTileItems). ARM A:
  upper-half draft (200000) accepted at full domain / spuriously rejected at 124160 (the
  mechanism, kept as RED-direction witness). ARM B: correction draw reaches an upper-half
  argmax only at full domain. ARM C: greedy accepts an upper-half draft under BOTH domains
  (byte-exact greedy property). Exit 77 = no device (house SKIP).
- E-1/E-2 INSTRUMENTATION banked: `tools/v340l/w7_f1b_pdbg_parse.py` (3-class classifier —
  ACCEPTED / OUT-OF-SLICE via the 40960-id JSON membership / NEAR-MISS at pre-registered
  eps=0.05 with 0.02/0.10 sensitivity / HEAD-MISS with rank percentiles; gate lines for
  b (>=30% of misses) and a1 (>=50% of misses); provenance guard auto-EXCLUDES pre-fix-tap
  logs from p_draft-dependent classes; bonus-column out-of-slice rate; local_am-vs-global_am
  sanity) — validated on a synthetic 4-column fixture. `tools/v340l/w7_f1b_boot.sh`
  (canonical posture + PDBG/DEPTH_LOG/VOCAB_COUNT env; I_HOLD_THE_WINDOW=1 guard — refuses
  to fire outside a held window). `tools/v340l/w7_f1b_legs.sh` (4 registers x3 probes:
  counting mt=600, prose/reasoning/code mt=1200; per-probe D2-SS-STATS + first-miss +
  PDBG byte-bracket extraction; auto-classifies per register).
- CONTAMINATION NOTE for the decode desk (flagged to coordinator 2026-09-19 ~10:2x): my d1
  edit sat in the tree ~10 min before the arbitration landed; reverted via git checkout
  (tree verified pre-fix, :3182 pair-era literal). If their build log shows
  tp2_backend.cpp compiled in that window, their Branch-B bin carries the d1 fix — harmless
  for their within-bin A/B (defer ON vs OFF on the SAME binary), but their record should
  note it.
- NEXT: decode desk banks -> I take the build claim (BUILD CLAIM in WO_DECODE_LANDING.md
  PROGRESS LOG) -> re-apply patch -> build + BANK-BEFORE-RELINK -> guard --selftest on the
  post-fix tree + device cell run (boot battery) -> E-1/E-2 legs in my window slice ->
  W7_f1b_row.txt.

### 2026-09-19 — F1b steps 2-4 DELIVERED (design desk, amd/wo-w7-body @ a99488c13 + this commit)
- ERROR-ANALYSIS DESIGNS banked: E-1 flip anatomy (NINFER_D22_PDBG tap, 3-class
  OUT-OF-SLICE/NEAR-MISS/HEAD-MISS taxonomy + pre-registered decision rule, zero build),
  E-2 per-register split (rides E-1's legs; stats + FirstMissHistogram + VOCAB_COUNT),
  E-3 quant-flip probe (env-gated bf16 draft-head arm spec, two-sided falsifier).
- OPTIONS priced: a1 head-slice bf16 (+53 MiB/rank, (m)+0.4 ms/round, break-even +0.7%,
  gated on E-3); a2 whole-layer bf16 PARKED; b reasoning vocab top-up (+36..81 MiB/rank,
  data-only, gated on E-1 class-(i) >= 30% in reasoning); c top-2/tree FALSIFIED BY
  ARITHMETIC (extra verify col (m)~+17.5 ms vs chain step (m)~1 ms — verify bodies own 87%),
  CONF_TAU dormant (offline tau sweep); d1 world=4 sampling-accept domain bug (one-line fix
  + RED/GREEN cell, correctness law); d2 ngram-mod (priced, composes); d3 q4 dial marked
  NEVER-FIRE.
- RECOMMENDATION: E-1+E-2 first (zero build) -> d1 independently (closure law) -> a1 as the
  implementation bet (E-3-gated) -> b conditional second. Pre-registered gates G-A..G-E
  (acceptance bar, PLOG-060 paired A/B with start-temp, no-prefill-impact, VRAM manifest
  charge only, byte-identity falsifier).
- No src edits, no builds, no boots. :8100 probe returned no listener (decode-landing desk
  mid-swap per WO law) — not needed; all receipts read from banked logs/artifacts.

### 2026-09-19 — F1b step 1 DELIVERED: DRAFTER MAP banked (commit a99488c13)
- MTP-1 head (1 layer, W8G32 int8 unconditionally) + runtime lm_head row-slice proposal
  head (40960 ids -> 10240 rows/rank at TP4, W8 default); pure-argmax propose via
  launch_draft_head + fused allreduce_argmax+remap; greedy first-prefix accept kernel;
  full file:line bank + existing zero-build taps (D22_PDBG / ACCEPT_AUDIT /
  VERIFY_DEPTH_LOG / VOCAB_COUNT) + measured anchors (PLOG-066 budget, REALK23, F1a
  coverage receipts).
- MAP FINDING d1 recorded: tp2_backend.cpp:3182 passes pair-era `2*nv` token_domain to the
  accept kernel (batched sites :4470/:6954 pass world*nv_l) — world=4 sampling branch sees
  HALF the vocab; greedy path unaffected (kernel returns before reading logits).
