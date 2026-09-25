# 111 — Phase 2 draft: KVarN long-context decode gap (drafting assignment)

**Status:** CURRENT — for the free agent, CPU-only, starts now.
**Mission:** DRAFT (do not implement) the docs/104 Phase 2 work order for the
KVarN long-context decode gap. Deliverable: a draft work order (docs/99
template) at `~/ninfer/drafts/phase2_draft.md` (NOT a repo commit — plan owner
reviews, numbers, and commits it). You are an analyst here, not an implementer.

---

## 1. The data (use this — do not re-derive, do not run anything)

Official baseline v1, committed: `~/ninfer/worktrees/wo-kvarn-hold/results/
official_baseline_v1_20260829.md` (full matrix + raw JSONs + serve logs).

**Greedy decode gap (kvarn vs best of int8/bf16) widens with context:**

| ctx | kvarn | int8 | bf16 | kvarn vs best |
|---|---|---|---|---|
| 10k | 69.1 | 69.6 | 68.9 | −0.7% |
| 25k | 68.3 | 70.0 | 66.4 | −2.4% |
| 40k | 64.0 | 66.4 | 65.5 | −3.6% |
| 80k | 58.2 | 63.4 | 60.2 | −8.2% |
| 160k | 47.1 | 60.0 | — | **−21.5%** |
| 250k | 43.5 | — | — | (no comparison) |

**MTP acceptance is NOT the cause** — kvarn tok/round is HIGHER than int8/
bf16 at every context (3.15-3.33 vs 2.90-3.06, greedy). The gap is
**round time**. Derived round times (ms = tok/round ÷ t/s × 1000):

| ctx | kvarn | int8 | bf16 | kvarn vs int8 |
|---|---|---|---|---|
| 10k | 45.4 | 41.7 | 42.1 | +9% |
| 25k | 47.6 | 43.7 | 44.5 | +9% |
| 40k | 49.5 | 44.7 | 46.7 | +11% |
| 80k | 55.2 | 46.8 | 50.8 | +18% |
| 160k | 66.9 | 51.0 | — | **+31%** |
| 250k | 76.6 | — | — | (no comparison) |

The per-round cost gap grows ~3× faster than the decode t/s gap → something
scales with KV size that hits the kvarn path harder. Prefill (−13-15% at ALL
contexts, flat) is a separate mechanism — that's Phase 3, OUT of scope here.

## 2. What to read (before writing)

- docs/104 — R1-R5 root causes + Phase 2 scope (this is the track you're drafting).
- docs/82 — post-wall roadmap + lever history (what was tried, what failed).
- results/doc83_lever1_int8_finding.md — negative lever: int8-ifying the KV
  load did NOT help. Your draft must explain why that result doesn't rule out
  (or does rule out) bandwidth hypotheses — this is where most drafters fail.
- results/official_baseline_v1_20260829.md + its JSONs (numbers above).
- The T≤16 packed decode kernel: src/ops/kernel/gqa_attention_kvarn_decode_
  packed.inc + src/ops/launcher/gqa_attention_kvarn.cu (where per-round cost
  lives: kKvarnDecodeSplits=54 wave sizing, 99 KiB smem, dequant warps).
- docs/102 §13 — KV variant storage differences (what kvarn stores/loads that
  int8 doesn't, per token, per layer).

## 3. Deliverable: draft work order (docs/99 template)

1. **Ranked lever list (3-5 levers), each with:** mechanism, exact code sites,
   why it hits kvarn specifically (vs int8/bf16), why the effect should scale
   with context (cite the round-time table), expected effect (t/s at 160k),
   byte-identity risk (the D-21 contract: any change to reduction/accumulation
   order is a NO-GO without a byte-identity gate — launch-config-only changes
   are the safe class).
2. **Hypothesis space** (investigate, don't conclude): wave quantization
   (54-split grid vs 36 SMs at long KV), per-token KV load bandwidth
   (dequant path vs int8 direct), attention tile geometry at long KV, per-
   round fixed overhead vs KV-proportional cost (the data says
   KV-proportional dominates — verify the split).
3. **Measurement plan:** baseline v1 protocol (greedy, 192-token budget,
   ITERS=1, MTP on), cells kvarn/int8/bf16 × 40k/80k/160k/250k, per-lever A/B
   (one lever at a time, byte-identity A/B first), propose pass bars and argue
   them (suggestion: kvarn@160k ≥ 55 t/s AND kvarn@80k within 3% of int8@80k;
   no cell regresses >2%).
4. **Explicit non-goals:** prefill (Phase 3), MTP acceptance (it's fine),
   sampler, GDN gating (docs/105 scope), D-21 gate itself, KV uniformity
   (Phase 1, in flight).

## 4. Constraints

- CPU only. No GPU, no server, no worktree in the repo, no commits anywhere.
- Deliverable is one file: `~/ninfer/drafts/phase2_draft.md`.
- Cite every number to the baseline doc/JSONs. Mark every code-site claim you
  have not verified in-tree as UNVERIFIED (you may read the code in
  ~/ninfer/worktrees/wo-kvarn-hold, read-only).
- If the data doesn't support a clear Phase 2 (i.e., the gap isn't attributable
  to anything actionable), say so with the evidence — a good negative is
  worth more than a speculative lever list.
