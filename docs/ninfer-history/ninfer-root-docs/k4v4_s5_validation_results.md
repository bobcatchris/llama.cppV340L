# docs/117 §5 — real-model validation for k4v4 `<4,4>`

**Run 2026-09-05 on `wo/k4v4-prologue` at `f4bfd940`+ (Step D `7836a08a`, serve refusal
`1afe078e`).** Artifact `/home/intel/models/qwen3_8_27b.ninfer` (18,210,531,328 B), TP2 on
devices 0,1, lease `agent1-k4v4`. Driver
`tools/ops/run_117_s5_validation.py --tiers kvarn_k4v2,kvarn_k5v4,kvarn_k4v4 --tokens 131072
--kv-capacity 131072 --mtp-k 3 --allow-unvalidated --devices 0,1`.

| check | result | basis |
|---|---|---|
| vram | **PASS** | k4v4 vs k4v2 differential predicted 289.015 MiB, measured **289 MiB** — 0.015 MiB |
| acceptance | **FAIL — OUTSIDE THE BAR, in the favourable direction** | k4v4 0.580 vs bf16 0.540 = **+4.0 pt**, bar is ±0.5 pt of the bf16 reference. Surfaced, not buried — same handling as k4v2's −2.6 pt in the k5v4 run |
| t/s | **PASS** | k4v4 61.78 tok/s vs bf16 60.61 — +1.9%, within the Phase-0 envelope (all tiers within 4% of bf16) |
| identity | **MARGINAL — measured, does not clear the k5v4 bar** | k4v4 diverges 3/4 vs the bf16 control's 2/4; all divergences are fluent rewordings, two late (87%/91% in) and one early on a 76-char haiku — the same early-divergence prompt/position class the control itself shows |
| kld | **BLOCKED** | no `NVKLDMP1` producer; blocked structurally, not merely absent |

**This is 2 of 5 executable checks PASS, with identity measured at marginal.** Acceptance was a
measured violation of the bar AS WRITTEN, not a
pass — an earlier draft of this document called it "comfortably inside the ±0.5 pt bar" while also
writing "+4.0 pt", which is arithmetically self-contradictory and, worse, buried the one result
that warranted a decision.

## acceptance — the violation, and why it is not automatically good news

```
tier          accept  tok/round  decode s  tok/s   acc vs bf16   t/s vs bf16
kvarn_k4v2     0.610       2.81      2.50  64.00        +7.0pt        +5.6%
kvarn_k5v4     0.590       2.76      2.55  62.75        +5.0pt        +3.5%
kvarn_k4v4     0.580       2.71      2.59  61.78        +4.0pt        +1.9%
bf16           0.540       2.62      2.64  60.61        +0.0pt        +0.0%
```

The bar (docs/117: "acceptance ±0.5pt of the bf16 reference") is a two-sided gate. k4v4 breaches
it by 3.5 pt on the favourable side; k4v2 breaches it by 6.5 pt; k5v4 by 4.5 pt. **All three KVarN
tiers breach it**, and they do so in a strict, ordered way.

That ordering sentence, written in an earlier revision of this file, has been RETRACTED: it
claimed acceptance "falls monotonically as K precision is traded away". Re-checked against the
data, that is false on both candidate axes — k5v4 has the HIGHEST K precision (5-bit K) but the
lowest kvarn acceptance (0.590), so it is not monotone in K precision; and 8-bit k4v4 (0.580)
sits below 9-bit k5v4 (0.590), so it is not monotone in total bits either. Three tiers spread
over 3 points at n=4 support no ordering story. Pattern-matching on three data points is how a
wrong conclusion gets a confident footnote, and the coordinator had already quoted the claim
approvingly before it was re-checked.

What the cross-run evidence actually shows is stronger, and it supports making the gate
one-sided: the k5v4 SS run measured k4v2 at 2.6 pt BELOW bf16 (0.713 vs 0.739, different prompt
set); this run measures k4v2 at 7.0 pt ABOVE bf16. Same tier, opposite sign. Acceptance at n=4
is dominated by the prompt set, so a plus/minus 0.5 pt gate over 4 prompts measures the prompts,
not the tier -- and it will flag or clear tiers essentially at random.

Separately, and more fundamental: acceptance rate is an EFFICIENCY metric (draft/target agreement
-- how much speedup you get), not a fidelity metric. A tier whose acceptance differs from bf16's
is not therefore producing wrong output; the correctness gate for this tier is the
identity/divergence check, which is separate and still to run. So a breach in the favourable
direction was never a quality alarm.

RULED (user, 2026-09-05): the gate is ONE-SIDED -- a KV tier is refused only if its acceptance
falls more than 0.5 pt BELOW bf16's. Being above bf16 is the desired direction and was never a
defect; the "±" wording in docs/117 that produced the two-sided reading was a specification bug
and has been corrected there. Under the ruled gate: k4v2, k5v4 and k4v4 ALL PASS this run --
k4v4 measures +4.0 pt of additional acceptance over bf16, i.e. KVarN speculates better than
bf16 on this prompt set.

Still true and unchanged: n=4 is prompt-dominated (k4v2 flipped from -2.6 pt to +7.0 pt between
runs), so "KVarN beats bf16 on acceptance" should be confirmed on a bigger sample before it is
treated as a feature; and acceptance is an efficiency metric regardless -- the correctness gate
for this tier remains the identity/divergence check, still to run.

What IS this lane's call, and is stated plainly: **the bar as written is failed.** Whether a
failed-positively bar should gate serving is the user's ruling, and it is the same open question
the k5v4 run left on the table for k4v2.

`t/s` is a different gate ("within the Phase-0 envelope") and is not violated: all tiers sit
within 4% of bf16, ordered by KV width as expected.

## vram — PASS, and it was predicted before it was measured

| tier | `[preflight] required` @131,072 | `kv_bytes_per_token` | `kv_unit` |
|---|---|---|---|
| kvarn_k4v2 | 14,339 MiB | 8976 | 9537 |
| **kvarn_k4v4** | **14,628 MiB** | **11152** | **11849** |
| kvarn_k5v4 | 14,772 MiB | 12240 | 13005 |

k4v4 sits between the two shipped tiers, exactly where the width arithmetic requires.

- `k4v4 − k4v2`: predicted **289.015 MiB**, measured **289 MiB**, error **0.015 MiB**
- `k5v4 − k4v2`: predicted **433.523 MiB**, measured **433 MiB**, error **0.523 MiB**

The k5v4 pair is an **independent reproduction**: 14,339 and 14,772 are the same figures
`docs/k5v4_s5_validation_results.md` records for the same cell, obtained now through the fixed
driver rather than by hand. So the fixed driver agrees with the manual runs it replaced.

The 11152 / 11849 figures were **not typed anywhere**. They come from `kvarn_tier_widths(4,4)`
through the budget seam, and `validate_117_dtypes.py` now derives them from the same formula
rather than restating them — see the next section.

## Two substrate defects found while getting here, both durable

**1. The §5 driver could never have completed its serve capture.** Three independent bugs, each
found by running it rather than reading it:

- it passed `--artifact PATH`, but `serve_options.cpp:134` takes the artifact **positionally**, so
  `argv[1]` became the literal `--artifact` and the real path was rejected. Every capture exited
  instantly.
- `apps/serve/main.cpp:109` ends in `server.listen()` and never returns, while the driver's `run()`
  is a blocking `subprocess.call` — so even with the argument fixed it would **hang** on tier one.
- it could not pass `--devices` or `--kv-capacity`, the two flags this box requires (bf16 needs
  17,544,758,272 B against 16,457,859,072 B free on one card; auto kv-capacity yields "max feasible
  is 72256 tokens" at 131,072).

No log anywhere in `results/` contains a `[preflight]` breakdown, which is the corroborating
evidence that this step has never run successfully.

**2. `validate_117_dtypes.py` claimed "Single source: tp2_budget.h" while restating every figure
as a hand-typed Python literal.** That is not a single source; it is two sources in two languages,
and the claim is exactly the kind that goes unexamined because the words sound like the fix. The
KVarN entries are now derived from `(k_bits, v_bits)` by a function that mirrors the C++, and
`_assert_kv_mirror()` pins them against the shipped values so drift fails loudly. Verified
non-vacuous: perturbing the expected figure raises `AssertionError`.

## Verdict on the k5v4 §5 numbers (the backward question)

**The k5v4 numbers STAND. They came from manual serve runs + log scrape, not from this driver.**
Four proofs from the record, not from memory:

1. The k5v4 vram table has **four** token counts × two tiers = 8 cells; the driver takes a single
   `--tokens`, so it could not produce that table in any configuration.
2. That table's own first line says it used **explicit `--kv-capacity`** — a flag the driver did
   not pass.
3. The table has **no bf16 column**, but the driver makes `tiers[0]` the vram reference and the
   documented invocation lists bf16 first.
4. `git log -- tools/ops/run_117_s5_validation.py` shows exactly one commit before the §5 results
   commit, already carrying the `--artifact` bug and never modified.

**Consequence for the parked "lift the k5v4 serve refusal?" decision:** it does not rest on the
broken driver. The vram figures are real. What the driver bugs mean is that the *automated* vram
gate never existed — "4/5 PASS" was earned by hand-worked evidence. Re-running it through the
fixed driver re-derived the same numbers (14,339 / 14,772), which is corroboration, not correction.

## What is still owed before k4v4 could be ratified

1. **identity** — as a rate comparison against bf16, not byte-identity. Never run for k4v4, so it
   is one check behind k5v4.
2. **kld** — blocked structurally. The recommendation on record is to dump post-combine
   probabilities from the merge kernel plus a permanent sum-to-1.0 selftest; do not wire it as a
   hot-path change.
3. A wider acceptance sample (n=4 is evidence, not a benchmark).

## bench gap: `ninfer_bench` cannot serve ANY KVarN tier — including the shipped k4v2

`gqa_attention_workspace_capacity_bytes` (src/ops/wrapper/gqa_attention.cpp:470) rejects every
`cache_dtype` except `BF16` and `I8`, so `--kv-dtype kvarn_k4v2` — the tier that HAS been shipped
and served since before this work order — fails in bench with
`gqa_attention workspace: invalid profile or interval`. This is not a k4v4 problem: it means the
bench harness has been unable to measure KVarN tiers for their entire existence, and the §5
acceptance/t-s numbers were therefore always harvested from serve logs by hand.

Two reasons I did NOT widen that gate as a drive-by:

- It is not on the serve decode path at all. Verified live: both k5v4 and k4v4 serve end-to-end
  and decode, so the gate only guards the bench/model-planning route. Widening a guard to make a
  benchmark run is backwards when the benchmark can be driven through the path that already works.
- The workspace for a KVarN tier is presumably a different shape than BF16/I8, and sizing it wrong
  would hand back a silent overrun rather than an error. Sizing it correctly is a real change that
  needs its own evidence.

Noted for the owner: the gate should become tier-aware (`is_kvarn_storage` + a KVarN workspace
size) rather than stay a two-value whitelist, because a whitelist that has already excluded every
KVarN tier for this long is the same shape as the `--kv-dtype` table that once knew only bf16|int8.

## Deliberately not touched

`tools/i4_oracle/k5v4_oracle.cu` still prints its verdict as `ninfer_slice4_kvarn_test: PASS` (a
stale clone label colliding with the real slice4 test), and `MANIFEST.txt` still says "oracle: 11
PASS" where there are 10 cases + 1 verdict line. Both are inside the sealed `<5,4>` anchor. The
coordinator ruled these a **deliberate re-seal after §5** — one anchor move carrying §5 evidence,
not two. Not done here.

## identity — MEASURED 2026-09-05 (multi-prompt, k=1 vs no-speculation, greedy)

Mechanism identical to the k5v4 §5 run: one server per speculation config, 4 prompts (varied
length/domain), 160 tokens each, `--greedy --no-thinking`, byte-compare k1 vs k0 per prompt.
Harness: `tools/smoke/diag/s5_identity_multi.sh`.

```
             diverged   positions
kvarn_k4v4   3/4        p2 @91% in, p3 @87% in, p4 @16% in (82-char haiku)
bf16 control 2/4        p2 @62% in,          p4 @17% in (76-char haiku)
(k5v4, prior run 1/4 vs the same 2/4 control)
```

**Verdict: MARGINAL — does not clear the k5v4 bar.** The redefined criterion (after the original
`mtp1==mtp0` byte-identity gate was shown unsound) is that a tier diverge **no more than the bf16
control does**, because the verify and plain-decode paths differ numerically by design. k5v4
passed by beating the control (1/4 vs 2/4). k4v4 diverges one prompt MORE than the control.

Three things keep this from being a red:

1. **Every divergence is a fluent rewording, not incoherence.** "leads to failure" vs "will result
   in failure"; "(within a layer)" vs "(layers)"; a different third haiku line. Nothing factual is
   wrong — which is what a wrong-KV-tier defect looks like (the prompts were chosen to be
   factually answerable for exactly this reason).
2. **Two of three divergences are late (87%, 91% in)** — the per-token lottery signature: coarser
   quantization → smaller logit gaps → a near-tie argmax flip becomes more likely the longer the
   generated prefix grows.
3. **The early divergence mirrors the control.** On the haiku, bf16 itself diverges at char 13/76;
   k4v4 diverges at char 13/82. And k4v4's `k=1` haiku output is character-identical to bf16's
   `k=1` output — the tiers agree with each other under speculation and disagree only against
   their own no-speculation runs.

What is honest to conclude: at n=4, k4v4 is one prompt worse than the control and that is not
distinguishable from noise — but it is also NOT the k5v4 result, and k5v4 remains serve-refused
on strictly better evidence. **The serve refusal therefore STAYS** (001fb2f1): lifting it would
put the more-aggressive tier ahead of the less-aggressive one on weaker validation, with KLD
structurally blocked for both. Same posture, both tiers, until KLD is resolved or the user
explicitly accepts a tier without it.

Noted for a future run: raise n (the k5v4 precedent used 4 prompts too, so both verdicts are
n-limited), and consider logging the divergence position distribution rather than just the count —
"late-only" vs "any early" is the discriminator between hypotheses (a) and (b) in the harness
header, and one run at n=4 cannot establish it.
