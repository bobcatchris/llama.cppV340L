# docs/161 — DFlash vs MTP: comparison test spec (seed, CORD; owner A2)

## Why they are NOT interchangeable metrics (the trap)
MTP = per-position AR chain, acceptance measured per drafted token (accept% = accepted/verified; tok/round 1..k+1). DFlash2 = block-diffusion: EVERY masked position has a top-16 lattice + greedy/sampled chain; "acceptance" upstream (z-lab) counts token-level agreement of the CHAINED proposals vs target argmax — but our verify counts accepted-prefix per round. Same word, different denominators. **A test that compares "accept%" across backends without naming the denominator is vacuous.**

## Established reference numbers (this box, this artifact, greedy, k=5)
| metric | MTP k=3 | DFlash2 k=5 |
|---|---|---|
| tok/round | ~2.97–3.21 (int8/k4v2 ladder) | 2.53–2.82 (fixd cell) |
| accept% | 65.8–73.7 | 29.5–35.3 |
| chain share of round | ~5.8% (B1 receipts) | 11.4% pre-ghost / D2 re-measure |
Do NOT read DFlash 35% < MTP 70% as "worse": per-draft acceptance is low BY DESIGN for block diffusion (z-lab corpus: 37-39%); the currency is **tok/round × net decode t/s**, not accept%.

## Test matrix to build (JSON-driven, per workload)
Workload classes (bodies live in repo results/ + /tmp bodies regen pattern; commit bodies to results/bench_bodies/ — /tmp is volatile):
- W1 short-reasoning (EN, 60-70 tok prompt, 48-256 gen, greedy)  ← tonight's cell family
- W2 long-prompt (25k/40k/80k ctx, 250-tok gens, warm-discarded)
- W3 multilingual/translation (ZH body used in ghost cells)
- W4 sampling (temp>0, seed-pinned — dflash is stochastic by design; MTP greedy byte-match tests DON'T transfer: assert DISTRIBUTION match or accept-rate band, never bytes)
- W5 dialogue/multi-turn (prefix-reuse on — different verify shapes)
Per cell, per backend (off / mtp-k3 / dflash-k5 / dflash-k7 when the seam allows): {tok/s, tok_per_round, accept_denominated ("prefix-accepted/verified" + "token-agree" BOTH), rounds, first-divergence position vs OFF baseline, VRAM peaks}.
Pass bars: dflash must BEAT mtp on at least the t/s column for its winning workloads (user's regime claim) or the merge exhibit says so; byte-identity vs OFF applies to greedy MTP only — dflash gets band assertions + determinism-with-pinned-seed checks.
## JSON test shape
tests/dflash_vs_mtp_cell.json: {body_id, dtype, ctx, conc, spec:"mtp|dflash2|off", k, seed, iters, gates:{tps_min, tok_round_min, accept_band:[lo,hi], determinism:true|n-a, byteid:"off-only"}} — one runner, matrix over JSONs (mirror decode_guard_cells_ci.sh style so farm can ingest it).
## Open research (A2)
(1) does single-seq dflash even admit (F3 law said conc>=2 — that's exactly what the single-seq work will settle); (2) tok/round parity point — at what ctx/gens does dflash k=5 overtake mtp k=3 on THIS box; (3) sampling determinism contract: what NINFER_DFLASH2_* env pins the walk tie-breaks reproducibly.
