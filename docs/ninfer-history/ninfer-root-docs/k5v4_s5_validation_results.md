# docs/117 §5 — real-model validation results (k5v4)

Run 2026-09-05 on the real artifact, TP2, both RTX 5060 Ti. Not a doc-number claim —
promote/rename at the coordinator's discretion.

**Artifact** `/home/intel/models/qwen3_8_27b.ninfer` (18,210,531,328 B), 27B target.
**Build** `wo/117-int4` @ `321ddc1e`. **Devices** 0,1 (TP2), usable 16310 MiB/rank.
**Gate lifted** `NINFER_ALLOW_UNVALIDATED_KV_DTYPE=1` — validation runs only; the k5v4
process gate stays in force for normal serving until this section's verdict is accepted.

## Verdict

| check | result | basis |
|---|---|---|
| vram | **PASS** | differential model validated <0.2% against 6 measured cells |
| acceptance | **PASS** | k5v4 0.738 vs bf16 0.739 — 0.1 pt, spec is ±0.5 pt |
| t/s | **PASS** | k5v4 1.54× over no-spec; bf16 1.54× — same envelope |
| identity | **PASS vs reference / GATE UNSOUND AS SPECIFIED** | bf16 itself diverges 2/4 |
| kld | **BLOCKED** | no `NVKLDMP1` producer exists anywhere in `src/` |

## vram — PASS

Explicit `--kv-capacity` (auto-sizing overshoots on this box, see "driver findings").
All figures are `[preflight] required ... MiB` per rank, **no drafter charged** (DFlash2
is not implemented, so these are the pre-drafter baselines).

| tokens | kvarn_k4v2 | kvarn_k5v4 |
|---|---|---|
| 131,072 | 14,339 | 14,772 |
| 160,000 | 14,602 | 15,131 |
| 163,840 | 14,637 | 15,179 |
| 204,800 | 15,009 | 15,687 |

Per-token slopes recovered from measured pairs, against the model's `kv_unit`:

- k4v2: `(14637-14339)/32768 -> 9542 B/token` vs predicted **9537** (+0.05%)
- k5v4: `(15179-14772)/32768 -> 13017 B/token` vs predicted **13005** (+0.09%)
- k5v4−k4v2 differential @204,800: measured **678 MiB** vs predicted **677.4**

Anchor cross-check: measured k4v2@204,800 = 15,009 → back-computed to 200,000 = **14,965**,
against the claimed anchor **14,968** (3 MiB). This confirms the anchor is a **k4v2** cell,
not I8, and independently corroborates agent2's §15.1 provenance fix.

## acceptance / t/s — PASS

`[tp2] single-seq lane` decode stats harvested from the serve logs; 4 prompts × 160 tokens,
5 measured rounds per config. Acceptance is gen-weighted.

| tier | acceptance (mtp k=1) | Δ vs bf16 | decode tok/s k=1 | k=0 | speedup |
|---|---|---|---|---|---|
| bf16 (reference) | 0.739 | — | 53.8 | 35.0 | 1.54× |
| **kvarn_k5v4** | **0.738** | **−0.1 pt** | 52.8 | 34.3 | **1.54×** |
| kvarn_k4v2 | 0.713 | −2.6 pt | 51.8 | 34.4 | 1.51× |

k5v4 sits 0.1 pt from the unquantized reference, inside the ±0.5 pt the spec allows.

**Side finding, not §5's to fix:** the *shipped* k4v2 tier is 2.6 pt below bf16 — outside the
±0.5 pt criterion. Consistent with 4-bit K shrinking logit margins. Flagged for the coordinator.

## identity — passes against reference, but the gate as written is unsound

docs/117 line 141/181/260 specify "A2 identity (mtp1==mtp0)" — MTP at `draft_tokens=1`
producing byte-identical greedy output to no speculation. No implementation of this gate
existed before this run; `serve_correctness_ci.sh` does not cover it (T1/T2/T3/T5/T8 are
prompt isolation, prefix equivalence, prefix divergence, mid-stream disconnect, admission
oversize). Implemented as `tools/smoke/diag/s5_identity_gate.sh` and `s5_identity_multi.sh`.

Single prompt, 96 tokens: bf16 PASS, k5v4 PASS, **k4v2 FAIL**. That looked like a k4v2
defect. Four prompts at 160 tokens shows it is not tier-specific in the way it first appeared:

| tier | prompts diverging (n=4) | divergence position |
|---|---|---|
| bf16 (no quantization at all) | **2/4** | 62%, 17% |
| kvarn_k5v4 | **1/4** | 16% |
| kvarn_k4v2 | 4/4 | 88%, 50%, 28%, 16% |

bf16 — which has no quantization to blame — diverges from no-speculation on half the prompts.
So **byte-identity is not a property the MTP path guarantees on any tier**: the verify path and
the plain decode path differ numerically (reduction order, tile shape — the same class as the
documented unified-vs-packed non-identity), and a near-tie argmax flips. Every divergence here
is late, and both sides are coherent paraphrases ("accuracy"/"quality", "leads to failure"/
"will result in failure") — never garbage, never token 0. The haiku prompt diverges on *every*
tier, which is the tell: it is the maximally unconstrained prompt, so near-ties are guaranteed.

**Consequence:** k5v4 diverges *less* than the unquantized reference (1/4 vs 2/4), so it passes
against the only baseline that means anything. But the gate needs redefinition — a rate/position
comparison against bf16, or semantic equivalence, not byte-identity. Blocking a KV tier on a
criterion bf16 fails would be a spec bug.

Caveat: n=4 prompts is a characterization, not a precise rate.

## kld — BLOCKED, and blocked deeper than "no producer"

`NVKLDMP1` has no producer. slice4/slice6 accept `dump_scores`/`dump_probs`/`dump_q` and every
caller in `src/` passes `nullptr`. Wiring it means an allocation plus file I/O on the decode
path; deliberately not done as a rushed hot-path change.

**But passing a pointer would not have been enough.** Reading the hooks before wiring them found
two structural problems that would have produced a *confidently meaningless* KLD number:

### 1. The KVarN dump is a partial distribution — roughly 1/54th of the key window

The KVarN decode kernel is split-K: `split = blockIdx.y`, launched as
`dim3 grid(KVHeads, splits, batch)` with `kKvarnDecodeSplits = 54`
(`src/ops/launcher/gqa_attention_kvarn.cu:576`). Each split computes its **own local softmax**
over `[split_start, split_end)`; keys outside that range are forced to `-CUDART_INF_F`
(`gqa_decode_slice6_kvarn_k5v4.cuh:550-553`), and the per-split `partial_m`/`partial_l`/
`partial_acc` are combined in a later merge kernel.

The dump is guarded `if (dump_probs != nullptr && kv_head == 0 && split == 0)`
(`:581`). So it writes **split 0's local softmax only** — a vector that sums to 1 over its own
key subset and is silent on the other ~53/54 of the window. KLD against that is not a measure of
attention divergence; it is a measure of one arbitrary slice.

And this is not a long-context edge case. `gqa_small_t_active_splits<Geometry, false>` routes
KVarN to `gqa_small_t_default_splits`, which has a **floor** of `4 * DecodeSplitScale` — so
split-K is on by default at essentially every window, not just the 128k-200k regime §5 cares
about.

### 2. The four kernels' hooks do not share an indexing convention

| kernel | dump indexing | shape |
|---|---|---|
| `gqa_attention_decode_bf16.cuh` | `dump_scores[key0]` | flat, per-key, split-range guarded |
| `gqa_decode_slice3_i8_v2.cuh` | `dump_scores[key0]` | flat, per-key, split-range guarded |
| `gqa_decode_slice4_kvarn.cuh` | `dump_scores[row0 * window + key0]` | 2-D row-major, `split == 0` only |
| `gqa_decode_slice6_kvarn_k5v4.cuh` | `dump_scores[row0 * window + key0]` | 2-D row-major, `split == 0` only |

So there is no single producer that serves all tiers: bf16/i8 emit a flat per-key vector, KVarN
emits a `rows × window` matrix, and only the KVarN pair has a row dimension at all. An
`NVKLDMP1` header's `rows`/`cols` would mean different things per tier, and a cross-tier KLD
would silently compare differently-shaped distributions.

### What the producer pass must therefore do

Pick one, explicitly, before writing code:

- **(a) force `splits = 1` in a dump-only mode.** Simplest, and single-split is arguably the
  cleaner measurement of a distribution over keys. But it changes the reduction order from what
  ships, so the number describes a debug configuration, not production.
- **(b) dump per-split partials (`m`, `l`, and the unnormalized accumulator) and reconstruct the
  normalized distribution host-side.** Most faithful — measures the real production path — and
  the most work: a new hook shape plus host-side combine.
- **(c) dump the post-combine probabilities from the merge kernel.** Faithful and correctly
  normalized, but `gqa_attention_kvarn_decode_merge_kernel` has no dump hook today, so it needs
  one added.

Recommendation: **(c)** — it is the only option that yields the distribution production actually
served, and it needs one hook rather than a host-side reimplementation of the combine. Pair it
with a selftest that asserts the dumped row sums to 1.0 within tolerance; that assertion alone
would have caught the partial-distribution trap immediately, and it is cheap enough to keep
forever.

Until then §5's KLD stays BLOCKED — correctly, and now with a reason that is about the hooks'
structure rather than their absence.

## Driver findings (fixed or filed)

1. **`ninfer-serve` takes the artifact positionally**, not `--artifact`. Driver corrected.
2. **TP2 needs `--devices A,B`.** Driver corrected.
3. **Auto KV capacity overshoots.** At `--max-context 8192` auto proposed 409,344 tokens for
   k4v2 (4043 MiB of context) against ~2459 MiB actually free after fixed (12,827) + workspace
   (1024), so preflight refused a config that fits. Reproduced on bf16 and k4v2. Worked around
   with explicit `--kv-capacity`; **filed as a real sizing bug, not mine to fix in this lane.**
4. `ninfer_bench` is gated behind `NINFER_BUILD_BENCHMARKS=OFF` in this build dir; the driver
   now treats its absence as a partial run instead of skipping all five checks.
