# W28: MTP ACCEPTANCE CENSUS + DRAFT-KV EXPERIMENT SPEC

Date: 2026-09-24. Desk: MTP-ACCEPTANCE (zero-GPU; U1 depth window held
/tmp/campaign_gpu_boot.lock throughout - untouched). Branch: amd/v340-port-v2.

Owner question: "we might be able to raise MTP acceptance with possibly
another quant." Served acceptance 0.66-0.68 (temp 0), mean_len ~3.0 on the
3-token draft-mtp chain.

## VERDICT (one paragraph)

The hypothesized lever does not exist in the direction asked. The draft
context's KV is ALREADY f16 - the highest useful cache type - because the
server builds the draft-mtp cparams from the speculative.draft cache-type
fields, which -ctk/-ctv never touch, and the launch passes no -ctkd/-ctvd.
"Raise the draft KV precision" would A/B two identical arms (moot window,
VOID by census). Per-token acceptance is head-calibration-bound: the MTP head
is a trained 1-layer approximation of the 65-layer stack shipped inside the
GGUF, and no served-side flag or cache quant changes what it proposes
(cross-project law L8: without a head re-fit there is likely no meaningful
acceptance lever; L3: any numerics delta is a population draw, not a
pairing gradient). The only draft-side quant lever in-tree points DOWN
(-ctkd/-ctvd q4_0): it frees 140.4 MiB/die and engages the fa40 direct arm
in the draft graph, but its t/s prize is <0.1% of a deep decode round and it
carries a real acceptance lottery risk - NOT RECOMMENDED. The one
acceptance-adjacent window worth a slot is chain shape (--spec-draft-n-max
4): a t/s experiment with acceptance as a guarded side-effect, spec'd in
section A5.

## A2. PRECISION CENSUS: the draft KV is f16 today (file:line)

The launch of record (BASEENV, /home/chris/launch_tp3_200k.sh) passes
`-ctk q4_0 -ctv q4_0` and NO -ctkd/-ctvd. Flow:

- `-ctk`/`-ctv` set `params.cache_type_k/v` only (common/arg.cpp:2215-2237).
  These reach exactly one place: the TARGET context cparams
  (common/common.cpp:1602-1603, `cparams.type_k = params.cache_type_k`).
- The draft-mtp draft context is a separate llama_context built in the
  spec_mtp branch of the server (no draft model is loaded:
  tools/server/server-context.cpp:1270-1291). Its cparams are:
  `cparams_mtp.type_k = params_base.speculative.draft.cache_type_k` and
  same for type_v (tools/server/server-context.cpp:1279-1280).
- `params.speculative.draft.cache_type_k/v` default to GGML_TYPE_F16
  (common/common.h:340-341, struct common_params_speculative_draft) and are
  set only by `-ctkd`/`-ctvd` (common/arg.cpp:3684-3706), which the launch
  does not pass.
- The has_draft fit-params copy at server-context.cpp:1146-1147 is the
  measurement path for a standalone draft model and does not apply here.

ANSWER: the draft (MTP/nextn) context's KV cache is served at F16 right now.
The target context's KV is q4_0. There is no inheritance in either direction:
the two contexts take their cache types from two independent param fields.

Secondary census - LLAMA_DRAFT_ONDEVICE_ARGMAX (set nowhere in serving;
src/llama-context.cpp:74 reads the env into cparams.draft_ondevice_argmax):
it does NOT change selected tokens. Contract, src/llama-ext.h:113-120: the
draft graph carries a per-shard argmax node (src/llama-graph.cpp:3602-3612,
`ggml_argmax_shard`) and "merging the pairs by max (first pair on equal max)
reproduces the greedy top-1 of the spliced row" - exact equivalence by
construction; decode() only skips the full n_vocab row fetch
(src/llama-context.cpp:2275-2300, buffer sizing at 2501-2505). Bandwidth
lever, acceptance-neutral. Not an acceptance lever; census closed.

## A3. DRAFT KV GEOMETRY AND VRAM COST

Geometry of the draft context (draft-mtp on this model):

- Model: /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf, arch `qwen35`
  (LLM_ARCH_QWEN35, hybrid - src/llama-arch.cpp:946), GGUF header:
  block_count 65, nextn_predict_layers 1, head_count 24, head_count_kv 4,
  key_length 256, value_length 256, full_attention_interval 4.
- The MTP draft context caches ONLY the nextn layer: for hybrid qwen35 the
  MTP context takes a plain KV cache with layer filter
  `il >= hparams.n_layer()` (src/llama-model.cpp:2084-2087 mtp_on_hybrid_qwen35
  dense-attention special case; filter at 2180-2181). The nextn layer is the
  dense MTP block (src/models/qwen35.cpp:19-25: "MTP layers are dense";
  recr pattern `(i+1) % full_attn_interval != 0` -> layers 3,7,...,63 =
  16 full-attn KV layers among the 65 target layers; the other 49 are
  gated-delta-net linear layers with no KV).
- So: draft KV = 1 layer. Target KV = 16 layers. The "17 full-attn layers"
  of the E-124/W24 law = 16 (target) + 1 (draft).

Per-die sizes at -c 200000 (TP4 tensor split: 4 KV heads / 4 dies = 1 KV
head = 256 dim per die per layer; q4_0 = 18 B per 32 elts = 0.5625 B/elt):

| cache                                | per die          | 4-die total   |
|--------------------------------------|------------------|---------------|
| draft KV f16 (SERVED TODAY)          | 195.3 MiB        | 781.2 MiB     |
| draft KV q4_0 (downward lever)       | 54.9 MiB         | 219.7 MiB     |
| draft KV f32 (the only "raise")      | 390.6 MiB        | 1562.5 MiB    |
| delta f16 -> q4_0                    | -140.4 MiB       | -561.5 MiB    |
| target KV q4_0 (contrast, 16 layers) | 879 MiB          | 3.44 GiB      |

Computation: f16 200000 x 256 x 2 B x 2 (K+V) = 204.8 MB = 195.3 MiB/die;
q4_0 200000 x 256 x 0.5625 B x 2 = 57.6 MB = 54.9 MiB/die.

Envelope correction: the mission premise "4x16 GB dies" is WRONG. sysfs
mem_info_vram_total = 8573157376 B = 8176 MiB = 8 GiB per die (all four,
read 2026-09-24 13:05 CDT). The f16 draft KV FITS - it is the live U1
serving config right now: used 6978.0/6977.4/6978.0/6977.8 MiB of 8176 MiB
per die, ~1198 MiB/die free at depth. No fit question exists.

GB/s-time cost of the draft KV:

- The draft reads its KV on every draft step (~3 steps/round, draft_n/rounds
  = 2.97 at the 10k anchor). At 200k depth that is 195.3 MiB/die per step
  (f16); at 10k, 10 MB. Against the E-124 deep law (~500 ms/round at 200k)
  and an effective tile-read rate of TB/s class, the draft KV read is
  ~0.05-0.15 ms/round at 200k: ~0.03% of the round. Nothing.
- Does the draft benefit from GGML_CUDA_FATTN_TILE_Q40_DIRECT? NO, today:
  the arm is q4_0-KV-gated and the draft KV is f16, so the draft graph's
  attention runs the standard f16 fattn-tile path. The fa40 boot line
  ("q4_0 KV in-kernel (gqa_ratio 6, T 4)", /home/chris/u1_server.log:38)
  is the target graph; the nextn layer has the same gqa_ratio 6 (24/4), so
  IF the draft KV were quantized to q4_0 the same direct arm would engage
  for the draft graph - the per-launch -37.5% tile-class effect would then
  apply to a launch that is ~0.03% of the round. Prize unchanged: nothing.
- The f16 -> f32 "raise" would ADD 195.3 MiB/die and slow the draft step,
  for a numerics delta far below the +/-3 pt acceptance resolution (L2).
  Null arm by construction.

## A4. MECHANISM ANALYSIS: what creates draft-vs-target divergence

Setup (self-speculative draft-mtp): the target's last hidden h_t (computed
by the target with q4_0 KV) feeds the MTP block
(input tensors are F32, "mtp_h_input" - src/models/qwen35.cpp:527-531;
enorm/hnorm RMS norms + eh_proj concat projection, qwen35.cpp:541-554),
which runs ONE dense transformer layer over the draft context's own f16 KV
cache and the shared LM head to propose tokens (chain of n_max = 3,
common/common.h:325, --spec-draft-n-max common/arg.cpp:3738; the verify
loop consumes them on the target). Acceptance = greedy argmax agreement.

Ranked sources by expected acceptance delta, with the code term:

1. MTP head approximation (STRUCTURAL, dominant). The head is a trained
   1-layer mimic of the remaining 65-layer stack; its disagreement with the
   full stack's argmax IS the 0.32-0.34 rejection rate. The head weights and
   its input block (eh_proj/norms) ship quantized inside the GGUF
   (12.25 GB for 27B ~ 3.6 bpw class); there is NO served-side re-fit or
   recalibration lever. Per L8 this is head-calibration-bound; per the
   coordinator's law a head RE-FIT could exceed native - not available
   in-tree. Expected delta from any served flag: 0.
2. Chain compounding (STRUCTURAL, second). Draft steps 2-3 condition on the
   draft's OWN previous outputs/hidden states, so per-position acceptance
   decays with draft depth; mean accepted clusters early
   (mean_len 3.04 of max 4 >> iid 0.68 sum 2.46). Not a precision term.
   Served lever exists but changes chain SHAPE, not per-token p:
   --spec-draft-n-max (section A5). Expected per-token acceptance delta: 0
   by definition; t/s delta: real.
3. Depth interaction (L4, confound + possible free gain). Cross-project law:
   acceptance rose +25 pts from 10k to 72k on the ninfer stack. Registered
   prediction already in W26 (draft_accept climbs above 0.68 in the U1 deep
   cells). Not a lever; a confound that forbids cross-depth acceptance
   comparisons and a possible compounding gain at deep decode if it holds.
4. Draft KV precision (ABSENT). The draft cache is f16 - the top useful
   cache type (q8_0 is LOWER precision at 1 B/elt; f32 is the only raise).
   Term does not exist today. If the draft KV were q4_0 it would ADD
   proposal-logit noise (a genuine divergence term, direction: worse); it
   is not. Delta available by "raising": below L2 resolution (= 0).
5. Verify-side q4_0 (CANCELS - this is the subtle one). The verifier's own
   argmax is computed under q4_0-KV noise, but the draft is conditioned on
   the SAME q4_0-derived h_t (the target produced it), and the drafted
   tokens are scored by that same verifier. Symmetric-quant divergence does
   NOT arise: both sides chase the same q4_0-noisy argmax, so the target's
   cache quant shifts the universe both sides share rather than opening a
   gap between them. A draft-cache quant change moves only side 1 of the
   comparison (term 4); a target-cache change moves both. Neither is a
   served acceptance lever.
6. Sampling (NO term). temp 0 greedy of record; the served fast-path envs
   are bit-exact/byte-identical class (owner sign-off, launch header), and
   LLAMA_VERIFY_ROW_SAMPLING carries the "identical copies by construction"
   contract (src/llama-ext.h:105-112). LLAMA_DRAFT_ONDEVICE_ARGMAX likewise
   (A2). No divergence, no lever.

## A5. EXPERIMENT SPEC (post-U1 battery)

A5.0 MOOT-WINDOW RECORD. The originally sketched arm "of-record +
-ctkd f16 -ctvd f16" is VOID BY CENSUS: the draft cache is already f16
(A2). Running it A/Bs two identical launches. Do not spend a window.

A5.1 NOT-RECOMMENDED ARM (for the record): downward draft-KV quant
(of-record + `-ctkd q4_0 -ctvd q4_0`). Prize: 140.4 MiB/die VRAM + the
draft step joining the fa40 direct arm (~0.1-0.2 ms/round at 200k, <0.1%
decode - below the battery's resolution floor). Risk: L8 acceptance
lottery (quantized proposal logits change the error DIRECTION against head
biases; the 0% parity artifacts on the ninfer stack bracket both ends).
Verdict: only consider if a future arm needs ~140 MiB/die of headroom
(e.g. an ub1024 or longer-context push); never as an acceptance window.
If run anyway: V-quant requires flash-attn (src/llama-context.cpp:3985-3988)
- satisfied by -fa on; engagement stamp = per-die sysfs used drops
~140 MiB vs a control boot at identical ctx; kill: any arm cell accept
< control - 5 pts or < 0.63 canary floor -> arm VOID, lever stays closed.

A5.2 RECOMMENDED WINDOW: w28nmax (chain shape 3 -> 4), the only
acceptance-ADJACENT lever that is real in-tree. It is a t/s experiment;
acceptance is a guarded side-effect (L3: no "better pairing" claims).

- Slot: FIFTH window in /home/chris/run_post_u1_battery.sh, appended after
  w22c1 (cheapest-first law bends for the W22 cooldown machinery already
  last). Standalone script not needed: the battery's cell() launcher already
  parameterizes arm identity via (cenv, cargs); the delta is
  (a) windows list += w28nmax, (b) adjudicate() CFG +=
  {"w28nmax": dict(metrics=("decode",), eng=None, exactness=True)},
  (c) two cell() calls + one plan block. This desk edited nothing in
  /home/chris; the coordinator applies the delta.
- Cells: interleaved CTRL1/ARM1/ARM2/CTRL2 at ctx 10240 decode-only
  (COMBO_CELLS=10k shape), 2 pairs per the E-133 position-1 law, 180 s
  settles + die_idle + die_hot<60 re-gates, void_gate.py --die3-group 2
  per cell - all inherited from cell() unchanged.
- Control launch: BASEENV + of-record line verbatim (cenv="", cargs="").
- Arm launch: identical + cargs="--spec-draft-n-max 4". Zero new env,
  zero code, zero VRAM (the chain loop extends; no new buffers).
- Engagement stamp (guard's own fields, no code): control draft_n/rounds
  ~3.0 (10k anchor: draft_n 125 over ~42 rounds = 2.97); arm must read
  ~3.9-4.0. Read draft_n and rounds from the guard jsonl; a stamped arm
  with draft_n/rounds < 3.5 = ENGAGEMENT-FAIL -> arm VOID (E-117 law).
- Primary metric: paired decode t/s (arm1-ctrl1, arm2-ctrl2). Secondary:
  mean_len, draft_accept as served side-effects.
- Expected prize (honest): chain-4 adds the depth-4 acceptance term.
  iid p=0.68 gives p4 ~ 0.21, but measured acceptance clusters early
  (mean_len 3.04 vs iid 2.46), so the marginal token is plausibly worth
  mean_len +0.15-0.3 -> +4-8% decode per the campaign conversion law
  (~+8% per +0.1 accept-equivalent), partially spent on the 4th verify
  token's KV read (negligible at 10k) and one more draft step (one nextn
  layer + shared LM head - the LM head, not KV, is the real draft-step
  cost; watch ms/round, not just t/s).
- Kill criteria (quantified per L2):
  - Acceptance band: paired draft_accept delta within +/-3 pts = NUMERICALLY
    NULL (metric resolution, 103-row study + our draft_n ~ 125-165/cell).
    Never promote or kill on a delta inside the band.
  - Promotion: paired decode mean > 0 AND arm accept not below
    control - 3 pts (a chain-4 that pays for itself in t/s while inside the
    accept band is a WIN on t/s grounds alone).
  - Hard VOID: any arm cell accept < control - 5 pts, or any cell < 0.63
    (mtp_canary floor stands regardless of window); any determinism_guard
    FAIL (text_sha256 must match control: a longer draft batch cannot
    change the greedy token sequence - content-invariance is the W7-class
    exactness law for this arm; a mismatch means the reuse/ckpt path is
    not content-safe -> arm VOID, lever re-closes, E-135(c) class).
  - Engagement-FAIL (above) -> arm VOID.
  - Window SKIP on control-cell failure per battery fail-loud law.
- Wall: 4 boots x ~12-14 min (boot + 10k battery + settles) ~ 50-55 min.
- VRAM pre-boot: die_idle (< 200 MiB all four dies) is already the cell()
  gate; no new check needed. Arm VRAM = control VRAM (A3 table: chain
  length does not touch cache size).
- L4 confound: all cells are 10k. Acceptance-vs-depth belongs to W26's
  live U1 table; w28nmax adjudication must not compare acceptance across
  depths, and if L4 holds (deep accept > 0.68) the chain-4 marginal token
  is worth MORE at deep ctx - a follow-up 200k pair (W22C1 pattern, F then
  R) is the optional extension, not part of the base window.

## Receipt chain

- A2/A3 census: this file, cites inline; sysfs reads 2026-09-24 13:05 CDT
  (live U1 server, lock untouched); GGUF header parsed read-only.
- W26 registered prediction (acceptance vs depth, L4): results/
  W26_u1_depth_table_2026-09-24.md - the depth leg of this question.
- Cross-project laws L2/L3/L4/L6/L7/L8 (ninfer 103-row study, owner
  relay 2026-09-24): folded into A4 ranking and A5 kill criteria as cited.

## Disclosure

Census, analysis, and this document were produced with AI assistance
(ZCode, GLM-5.3-Flash) from code reading and banked campaign artifacts;
the contributing human reviewed the cited code paths and the experiment
design. No GPU was touched; no /home/chris file was edited; no push.
