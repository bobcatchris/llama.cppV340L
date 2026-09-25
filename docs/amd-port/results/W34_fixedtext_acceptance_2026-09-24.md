# W34: FIXED-TEXT ACCEPTANCE INSTRUMENT (design, build, run spec)

Date: 2026-09-24. Desk: FIXED-TEXT-ACCEPTANCE (zero-GPU; 5-window battery
owns the machine; /tmp/campaign_gpu_boot.lock never touched, own-stamp law
only). Branch: amd/v340-port-v2.

## 1. THE GAP

Acceptance comparisons across arms are confounded by trajectory divergence
(cross-project law L3, adopted E-139 cont.): a numerics change can flip a
greedy token, the arms then decode DIFFERENT texts, and per-text head
agreement varies enormously by text type (T-10 population law). A served
draft_accept delta between arms is therefore a draw from a per-text
population, not a pairing gradient. W28 (results/W28_mtp_acceptance_
2026-09-24.md, section A4 term 3) registered the confound and left the
fixed-text cell out of scope; this desk built it.

The prescribed instrument (ninfer receipt s5): feed BOTH arms the SAME
continuation text and measure per-token head hit-rates directly. In our
stack greedy verify makes the target's argmax the ground truth, so the
measurement is: score the draft-mtp head's top-1 (and chain prefix) against
the target's realized greedy tokens over a FROZEN corpus.

## 2. WHAT THE SERVER EXPOSES PER-TOKEN TODAY (census, file:line)

Per-request, machine-readable (this is the guard jsonl witness surface):

- POST /completion resp.timings.draft_n and timings.draft_n_accepted
  (tools/server/server-task.cpp:255-258, result_timings::to_json; filled
  from the slot counters at tools/server/server-context.cpp:520-522).
- The counters are PER-REQUEST: slot.reset() clears n_draft_total /
  n_draft_accepted / n_draft_verif_steps / n_accepted_per_pos at task
  start (tools/server/server-context.cpp:322-326, struct fields 294-298).

Per-request log lines (slot.print_timings() at request end,
tools/server/server-context.cpp:610-636, called at 3953 and 4096):

- LOG_INF (always on): "draft acceptance = X (A accepted / G generated),
  mean len = M" (server-context.cpp:627-630). This is what
  guard_battery.parse_server_log_tail regexes.
- LOG_TRC (needs -lv 5 or higher, common/arg.cpp:3533 -lv/--verbosity;
  levels in common/log.h:24-25): "acc per pos = (p1, p2, ...)"
  (server-context.cpp:619-625, SLT_TRC at 632-633). Counters filled at
  4065-4070. This IS acceptance-vs-position-in-chain, per request,
  exposed today with zero code.

Per-ROUND log lines (env LLAMA_TRACE=1, read at
tools/server/server-context.cpp:1433-1437): "accepted k/n draft tokens"
(server-context.cpp:4046-4047) and "... (restore checkpoint)"
(server-context.cpp:4016-4017). One line per verify round, in order,
inside the accept loop. LLAMA_TRACE also enables the spec timeline
(server-context.cpp:3988-4004).

Target per-token top-k logprobs: request field n_probs (alias "logprobs")
on /completion (tools/server/server-common.cpp:1133-1141 for the OAI
"logprobs"/"top_logprobs" spelling; schema tools/server/server-schema.cpp:
176-178). Emission at tools/server/server-task.cpp:288-334. CRITICAL HOLE:
probs are populated ONLY on the non-speculative path
(tools/server/server-context.cpp:3947-3948 -> populate_token_probs at
2049-2103, which reads the target context logits). The speculative accept
loop emits tokens with EMPTY probs - the literal "TODO: set result.probs"
at tools/server/server-context.cpp:4091. So a spec-on boot cannot emit
per-token target top-k today; a spec-off boot can.

Draft head proposals per token: NOT exposed. The LLAMA_DRAFT_* family
(ONDEVICE_ARGMAX src/llama-context.cpp:74, SHAPE_CACHE :230-231,
PREFIX_CATCHUP speculative.cpp:1348+, PACKED_GET :1323, LIGHT_SYNC :1324,
FAST_TOPK :1692) are bandwidth/bit-exactness levers; none log proposals.
LLAMA_SPEC_TIMELINE logs accept timing, not proposals. The drafted ids are
consumed inside common_sampler_sample_and_accept_n; only aggregate counters
and the per-round "accepted k/n" line surface.

All surfaces verified present in the served binary's impl library
(build-hip/bin/libllama-server-impl.so): strings "acc per pos" (1),
"draft acceptance" (1), "accepted %2zu/%2zu draft tokens" (2),
"draft_n_accepted" (1), "n_probs" (2). The runner pre-checks these
fail-loud before any boot.

## 3. THE INSTRUMENT (design)

Two passes per cell, one SHORT boot (ctx 10240, battery 10k cell shape):

FT-GEN (freeze + served witness). One greedy continuation per corpus item
(temperature 0, cache_prompt false). Records served acceptance per item,
per-round lines, per-position line, and the text sha256. The CONTROL
cell's continuations are FROZEN: tokenize(prompt + text, add_special=true
to match the server's string-prompt tokenization,
server-context.cpp:2333) gives full_ids; the continuation is
full_ids[prompt_len:].

FT-CASCADE (divergence-proof fixed text). For each frozen item and each
grid point j (default j = 8, 20, ..., 152): POST /completion with
prompt = full_ids[:prompt_len + j] AS A TOKEN-ID ARRAY and n_predict =
chain_nmax + 1. The server passes token-id prompts through unchanged
(tokenize_mixed, tools/server/server-common.cpp:627-662; add_special is
not re-applied to arrays). Round 1 of each request drafts off the TRUE
frozen history: the per-round "accepted k/n draft tokens" line
(LLAMA_TRACE=1) gives the accepted-prefix length k. Over all j and items:

  p_i = P(round-1 accepted >= i)

p1 is the head's top-1 hit rate conditioned on true target history - the
pure head-quality datum. p2/p3(/p4) add the draft-conditioned chain terms.
n_predict = nmax + 1 makes round 1 propose the full chain
(get_n_draft_max = min(n_ctx - n_tokens - 2, n_remaining - 1),
tools/server/server-context.cpp:426-445); rounds after a partial accept are
also recorded (rounds[] list per request) but round 1 is the fixed-text
datum. The fixed text lives on the PROMPT side, so the input is IDENTICAL
across arms BY CONSTRUCTION: no trajectory divergence can enter, which is
the whole point (L3). An arm whose numerics diverge from the reference is
still scored against the same frozen tokens.

Note the acceptance being measured IS the head-vs-its-own-verifier rate:
for an arm that changes target numerics, the verifier's argmax at a
frozen prefix may differ from the frozen token - which is correct, that
disagreement is part of the arm's head-quality reality (the head should
predict what THIS arm's target actually says).

Cells (E-133 bookend law): r1 (control --spec-draft-n-max 3, FREEZE) /
a1 (arm) / a2 (arm) / r2 (control, re-freeze). All four cascade against
FROZEN1 from r1. The r1-vs-r2 compare is the instrument's own A/A null
calibration (same arm, same frozen text -> any p1 delta there is pure
instrument noise); a1-vs-a2 is the arm replicate. GEN runs in every cell,
giving the cross-arm served-text sha matrix (the L3-in-the-wild witness).

T-10 law folded in: the corpus is 6 items spanning prose / repetitive /
list / code / factual / dialogue kinds; the scorer reports per-kind and
per-item p1 (the spread itself is a law witness) and the compare verdict
requires consistent per-item sign, not just a pooled delta.

## 4. AS BUILT AND HOST-VERIFIED (zero GPU)

Files:

- docs/amd-port/scripts/ft_corpus.json - corpus (6 items, ASCII).
- docs/amd-port/scripts/ft_cell.py - cell driver (GEN / freeze / CASCADE,
  log-window round parsing, selftest mode).
- docs/amd-port/scripts/ft_score.py - scorer (summarize / compare, Wilson
  95 pct intervals, per-kind, per-item sign consistency, sha matrix,
  union-of-positions compare so a chain-4 arm's p4 prints against a
  chain-3 control's n/a).
- /home/chris/run_fixedtext_acceptance.sh - window runner (NOT repo-side;
  house law). Lock law verbatim from the window pattern: own-stamp
  /tmp/campaign_gpu_boot.lock, campaign_busy = lock + live battery / U1 /
  U2 / fa40 / guard / watchdog / any llama-server process, pre-boot
  die_idle < 200 MiB all four dies + die_hot < 60 C (E-134e), 180 s cell
  settle (E-134), sleeps <= 150 s, per-cell provenance stamp (git HEAD,
  binary sha256, corpus sha, frozen sha, full launch line), arm-identity
  freshness gate, fail-loud capability pre-check (exit 2 pre-GPU),
  dry-run refuses (exit 3) while the machine is busy. Light thermal
  sideband sampler per cell writes a void_gate.py-compatible csv
  (timestamp,cN_edge,cN_sclk; verified parseable by void_gate.py
  --die3-group 2 --observe).

Verification done host-side (no GPU touched):

- bash -n OK on the runner; py_compile OK on both python files; JSON
  loads; all four files byte-clean ASCII (LC_ALL=C grep '[^ -~]' = 0).
- Round-regex unit test against the REAL format strings (slot prefix,
  "%2zu/%2zu" space padding, restore-checkpoint variant, multi-round,
  0-accept, per-pos line, accept line): all pass.
- Scorer synthetic cells (signal arm +0.09, null arm -0.11 with sha
  mismatch, chain-4-vs-chain-3 union/p4 row): verdict structure correct.
- END-TO-END MOCK: a local HTTP mock implementing /tokenize, /completion
  and appending real-format log lines; freeze cell (gen -> freeze ->
  cascade on own frozen file) and arm cell (cascade vs frozen) both ran
  clean; frozen file, per-request records (rounds, round1_*,
  prompt_ids = prompt_len + j, sha, acc_per_pos, log_mean_acc_len) and
  the final compare table all verified. Mock dir removed.
- Dry-run against the live machine: capability pre-check 6/6 PASS on
  build-hip/bin/libllama-server-impl.so + --help; selftest OK; correctly
  BLOCKED exit 3 (battery pid 6879 live + watchdog lock) - the refusal
  path is exercised, not just the happy path.

## 5. EXACT RUN COMMANDS (when the machine frees)

Defaults are the w28nmax chain-4 comparison:

  /home/chris/run_fixedtext_acceptance.sh --dry-run   # re-check gates
  /home/chris/run_fixedtext_acceptance.sh             # 4 cells, ~45-55 min

  outputs: /home/chris/ft34_{r1,a1,a2,r2}.{jsonl,server,sideband.csv},
           /home/chris/ft34_frozen_r{1,2}.json,
           /home/chris/ft34_results.txt  (adjudication: r1-vs-a1,
           r1-vs-a2, r1-vs-r2 A/A null, a1-vs-a2 replicate)

Any future numerics arm reuses the same cells by env, e.g. the q40 prefill
arm:

  FT_ARM_ENV=GGML_CUDA_FATTN_TILE_Q40_PREFILL=1 FT_ARM_ARGS="" \
    /home/chris/run_fixedtext_acceptance.sh

(state the arm's chain args in FT_ARM_ARGS explicitly if it is not the
of-record 3; keep FT_NMAX_ARM consistent with them). Wider resolution:
FT_GRID_STEP=4 FT_GEN_NP=240 gives 60 points/item = 360 samples/cell
(p1 95 pct band ~ +/-0.05 pooled, ~ +/-0.035 across both arm cells) at
roughly +4 min/cell; the default (13 points/item, 78/cell) has a p1 band
of about +/-0.10 per cell, ~ +/-0.07 pooled over the two arm cells.

## 6. HOW TO READ TONIGHT'S w28nmax CHAIN-4 VERDICT

WITHOUT this instrument (the W28 A5.2 window alone):

- The served draft_accept side-effect is a population draw (L3). A
  +/-3 pt served delta is inside metric resolution (L2) and must not be
  read as head quality even at identical sha - it CAN be read as
  chain-shape arithmetic only because greedy verify pins served text to
  the target's own argmax (identical sha across arms makes the two
  cells' per-text populations the same text, which is why the sha gate
  is a hard gate there).
- The verdict on chain-4 is then purely the t/s verdict: promote iff
  paired decode mean > 0 and no accept band breach (W28 A5.2 laws).

WITH this instrument (run it before or after the window; same machine
law, 4 short boots):

- p1/p2/p3 measured on the SAME frozen corpus under chain-3 and chain-4
  arms should be IDENTICAL up to instrument noise (the head and the
  text are the same; only the chain loop extends). The A/A compare
  (r1-vs-r2) prices that noise. If p1..p3 agree within the A/A band,
  any served acceptance change in the window is chain-SHAPE arithmetic
  (mean_len up by construction), not head change - the cleanest possible
  support for the t/s verdict.
- p4 (arm-only row, control prints n/a) directly prices the marginal
  4th token: p4 x (token value) vs (one more draft step + one more
  verify row) is the W28 A5.2 expected-prize estimate measured instead
  of extrapolated.
- If p1 DOES move beyond the A/A band with consistent per-item sign,
  the chain-4 path perturbs earlier positions (a real finding: the
  extended chain loop is not content-neutral for the draft), and the
  window's acceptance side-effect gains a causal reading.
- For any FUTURE numerics arm (q40 prefill, mmvq shares, RCCL order),
  the served draft_accept delta stays a population draw, but the
  cascade p1 delta on frozen text is a head-quality measurement; pair
  every such window with these 4 cells (or 2: r1 + one arm cell) when
  acceptance is load-bearing.

## 7. MINIMAL SERVER ADDITION (if exact proposal logging is ever wanted)

Not needed for this instrument. If per-token draft PROPOSALS (not just
accept counts) are ever required, the two smallest sufficient changes are:

1. Per-round proposals: in the speculative accept loop, right after the
   draft is sampled and before accept, trace-log slot.spec_draft at
   tools/server/server-context.cpp:3971 (where n_draft is saved) - one
   SLT_TRC line with the proposed ids alongside the existing "accepted
   k/n" line at 4046-4047. ~3 lines, LLAMA_TRACE-gated, zero hot-path
   cost when off.
2. Per-token top-k under spec: fill result.probs for accepted ids in the
   accept loop at tools/server/server-context.cpp:4084-4091 (the standing
   "TODO: set result.probs"), reusing populate_token_probs (2049-2103)
   against ctx_tgt rows via slot.spec_i_batch. Larger change (the logits
   for drafted rows are already computed by the verify batch; the row
   index mapping is the work).

## Receipt chain

- W28 acceptance census + w28nmax window spec: results/
  W28_mtp_acceptance_2026-09-24.md (terms 1-6, A5.2).
- Guard jsonl witness + decode-guard request pattern:
  docs/amd-port/tests/guard_battery.py (run_decode_guard:494+,
  parse_server_log_tail:411-438).
- Serving env of record: /home/chris/launch_tp3_200k.sh (BASEENV copied
  verbatim into the runner so p_i is measured under serving numerics).
- Window/lock law pattern: /home/chris/run_u2_prefill_window.sh.

## Disclosure

Census, design, scripts, and this document were produced with AI
assistance (ZCode, GLM-5.3-Flash) from code reading and banked campaign
artifacts; the contributing human reviewed the cited code paths and the
instrument design. No GPU was touched; no existing /home/chris file was
edited (only the new runner added); the campaign lock was never touched;
no push.
