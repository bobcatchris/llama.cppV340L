# ACCEPTANCE LEVERS — in-tree audit + pricing (no-GPU desk, 2026-09-17)
Seat: analysis (no GPU), branch `amd/tp4-cure`. Baseline: MTP k=2 real prose acc 0.65,
tok/round 2.31, 10.79 tok/s (`results/amd/coherence/REALK23_row.txt`, bin 07ad7eccc0b97cc0).
Target: 50 tok/s decode (EXTRACTING_NUMBERS_and_50TPS_budget.md Part C). Code cites are the
`amd/tp4-cure` working tree 2026-09-17; `src/runtime/tp2/tp2_backend.cpp` has in-flight owner
edits, so quoted anchors are the durable part of each cite.

## (a) Drafter sampling — ALREADY GREEDY, lever banked
Every proposal (round-0 d0 `:2308`/`:2459`, chain `:2504`/`:3518`/`:3596`) is a fused
`allreduce_argmax` over the sliced draft head — no temperature, no sampling kernel on the
draft path. The verify accept is greedy ALWAYS (`speculative_accept_greedy_drafts`, `:3156`;
target argmax `:3143`), regardless of request temperature; temp>0 sampling exists only on the
PLAIN decode path (`:2758`). doc/optimizations/01 §4's "greedy drafter = free 10–25 pts" is
already captured in-tree AND inside the 0.65 baseline (probes run `--greedy`,
`results/amd/coherence/E14_serve.log:40`). **Nothing to fire.** Corollary: a temp>0 A/B on
the MTP serve path re-measures the same greedy stream — do not spend a window on it.

## (b) Draft-vocab slice — coverage is the ceiling; measure before resizing
Slice = `tests/multi_gpu/data/qwen38_draft_vocab_ids.json`, 40960 ids (16.5% of 248320),
frequency-shaped over the sibling 3090 repo's outputs; provenance in
`docs/13_future_objectives_mtp_70plus.md` Objective 1 ("BIGGEST lever"; ⚠️ validate ≥95%
coverage on OUR model — **not yet banked**; B3HARDEN_row.txt proves the file loads, not coverage).
**40960 vs "10240 rows" reconciled:** per-rank head is SLICED to `n_draft_local = 40960/tp_w`
= 10240 at world=4 (`:1107`; proposal width `:1242`; rows gathered from the full 248320-row
head by `src_row = draft_vocab_ids[rank*n_local + i]`). W8G32 row = 5120×1B codes + 160×2B
scales = 5440 B → 10240×5440 ≈ 53 MiB — matches the boot line "[rank r] draft output head
ready (W8G32, 10240 rows, 53 MB)" (printf `:1224`; `B3ENV_serve.log:17`). `dev_draft_vocab`
carries all 40960 ids for row→global-id mapping (`:1233`).
**Ceiling semantics:** a draft can only propose slice ids; a target-argmax outside the slice
is a guaranteed miss at that position (docs/13 Obj 1 step 4 — spec stays exact). Per-position
acceptance ceiling = slice coverage of the target's argmax stream.
**THE measurement exists in-tree, zero-cost:** boot with `NINFER_VOCAB_COUNT_DIR=<dir>` →
"[vocab] coverage=XX.XXX%" every 10k accepted tokens (`vocab_output_counter.h:77`, wired
`:1656-1660`). Put it in every acceptance window.
**Raising to ~100k ids:** 25000 rows/rank ≈ 130 MiB/rank (+77 MiB; 8160 MiB usable at boot,
E14_serve.log:3); draft-head GEMV per chain step 53→130 MB ≈ +0.3 ms at the measured 265 GB/s
— noise vs the 212 ms round. **Slice SIZE is not the constraint.** Decision rule: fire the
coverage counter first; only measured coverage < ~90–95% justifies a bigger slice.

## (c) `--mtp-ngram-mod` (+ `--ngram-mod-seed`) — cheapest live lever, host-only
Mechanism: 24-gram → last-seen-next-token host pool, 4,194,304 entries ≈ 16 MB RAM,
process-lifetime, shared across requests (`mtp_ngram_mod.h`; pool created `:1621`). Each round
`roll()` greedily extends a chain (stuck-loop guard); if length ≥ `--ngram-mod-n-min` it
REPLACES the MTP drafts for that round (`:2913-2930`), else falls through to the head. Every
drafted token is still verified → output distribution unchanged. `--ngram-mod-seed` replays a
corpus into the pool at boot (`mtp_ngram_seed.h`; requires `--mtp-ngram-mod`,
`src/product/speculative_options.h:74-78`).
Compatibility: MTP-only; DFlash2 refuses it (`speculative_options.h:91-94`); bookkeeping stays
honest (ngram rounds excluded from first-miss/adaptive feeds, `:3367`).
Expected: big on repetitive streams (the count600 counting body is the easy win), upstream
wins on code/retrieval; on real prose the pool misses often (doc 01 §4 domain curves) —
unproven on prose until measured. COST: zero GPU (host roll/add; a winning round skips the
draft chain entirely).
LIMITATION: ngram lives ONLY in the single-seq round loop — the batched MTP loop
(`run_tp2_requests_batched`, `:5355+`) has no ngram/adaptive/conf wiring, so at ≥2 concurrent
lanes the pool is dormant. Ngram gains do NOT carry into the concurrency arm.

## (d) `--mtp-adaptive` / `NINFER_MTP_CONF_TAU` — real mechanisms, priced LOW at k=2
Adaptive (`mtp_adaptive.h`, llama.cpp PR #27210 port): depth climbs on full-accept streaks
(climb thresholds 2/4/10/6/3/2; 3→4 hardened, `:30-40`), drops on misses; buffers sized to
`k_buf = mtp_draft_max` at boot (`:733-734`); needs `--mtp-draft-max ≥ 2`
(`speculative_options.h:41-55`). CONF_TAU (`mtp_confidence_break.h`, env read `:1772`):
in-chain confidence-product early break; shortened round = prefix of the full proposal →
byte-identical output; conf bulk-read at the accept sync, no mid-chain syncs.
Pricing: deeper k already LOSES on prose (k=3: 9.75 vs 10.79 tok/s, acc 0.51 — REALK23
VERDICT), so adaptive's climb has nowhere profitable to go (it would sit at the floor = fixed
k), and CONF_TAU's max save at k=2 is one chain step/round. **Do not spend a window.** Revisit
only for a ≥0.8-acceptance class (count600 hits 0.97 — TIMING1_row.txt).

## (e) Thinking mode — already ON in the measurement posture
Every banked serve probe runs `thinking=on` (E14_serve.log:40; REALK output streams carry the
reasoning prefix). doc 01 §4's thinking lift (+7–10 prose pts) is therefore already inside
acc 0.65. A thinking-off arm would measure a posture we do not serve. Skip.
## Fire order for the next windows (cheapest decisive first)

1. `NINFER_VOCAB_COUNT_DIR` on the standard serve line — one leg answers the slice-ceiling question (b).
2. `--mtp-ngram-mod` A/B, count600 body + REALK prose body — free mechanism, one arm each (c).
3. Concurrency N=1/2/4 per `results/amd/coherence/CONCURRENCY_BENCH.md` — orthogonal to
   1–2 at k=2 (M-wall: 4 streams ≈ 3–3.5× aggregate, doc/optimizations/01 §3).
4. Slice resize / adaptive / CONF_TAU: only if 1 or 2 say so.
