# WORK ORDER: d2 ngram-mod draft correction — pricing from banked F1b anatomy + host-cell implementation (VALUE_QUEUE TIER 1 item #2)

Owner: desk agent (1 slot). GPU-FREE desk: Phases 1-2 need NO GPU and NO cmake build.
Phase 3 is GPU and QUEUED — do not wait on it; deliver Phase 1-2 then message coordinator
`QUEUE EMPTY` via your final report (never a channel post — COMM LAW).

## CONTEXT YOU MUST KNOW (banked facts; cite, do not re-derive)

- F1b anatomy (COMPLETE, last window): HEAD-MISS = 81-92% of draft misses in EVERY
  register — the drafter picks the runner-up; decision margin median 0.24-0.35; anatomy
  over 7218 slots. Row: results/amd/coherence/W7_f1b_row.txt (locate on amd/main; per-slot
  data banked alongside — find it, do not regenerate).
- The lever (principal-named): an ngram/continuation lookup that promotes the runner-up
  into the draft set on margin+match attacks exactly the HEAD-MISS class. This is the
  prompt-lookup idea hybridized with the drafted top-1: when the draft's top pick has low
  margin AND an ngram match in the request context predicts the runner-up, promote.
- d1 (half-vocab sampling + tap-position race) is CLOSED — do not reopen; your surface is
  the PROMOTION POLICY, not the sampler bugs.
- Serve contract facts: spec=mtp draft-tokens=2; acceptance is measured as tok/round
  (baseline ~2.59-2.75 at plen-2075 class, PLOG-080/076-era probes); thinking-model
  bodies need "model":"qwen3.8-27b" and max_tokens>=128 (tool-law class).

## PHASE 1 — PRICING (GPU-free; deliverable = a number with receipts)

1. Locate the banked F1b per-slot anatomy data (results/amd/**, amd/main and lane
   branches: `git log --all --oneline -- '*f1b*'`). Read its schema.
2. Bound the win FROM DATA: among HEAD-MISS slots, what fraction has (a) the true target
   token == the runner-up, and (b) an ngram match in the request context that predicts
   that runner-up? Fraction x current miss rate = the acceptance pp upper bound; the
   margin+match gate curve (sweep match length, margin threshold) = the honest priced
   band. Deliverable: `results/amd/d2ngram/D2_PRICING_ROW.md` with the table + the
   pre-registered Phase-3 gate this pricing implies.
3. If the banked data cannot answer (b) (no stored request context per slot), price the
   (a) bound only and SAY SO — a bound with a named gap beats an invented number.

## PHASE 2 — IMPLEMENTATION + HOST CELL (GPU-free; NO cmake — direct g++ only)

1. Find the draft-selection site in src/runtime (spec mtp, draft-tokens=2 path): where the
   draft tokens are picked and where a host-side post-process could promote the
   runner-up. Cite file:line in this WO's log.
2. Implement NINFER_D2_NGRAM (strict-parse gate, same class as
   src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_simt_gate.h:89-115): unset = byte-identical
   behavior (LAW: kill-switch first); "1" = on; "0"/garbage = off, traced once. The ngram
   table builds from the request's own context tokens (in-flight, bounded memory — state
   the bound; no estimated-VRAM refusals anywhere: allocator is the gate).
3. Host cell (tests/d2ngram/ or tools/v340l/): pure host TU, compiled with DIRECT g++
   (NOT cmake — disk law; df -h / before any >1G write). Cell = synthetic draft logits +
   synthetic ngram table: (i) RED row — gate unset => promotion never fires, outputs
   byte-identical; (ii) GREEN row — gate on => runner-up promoted exactly on
   (margin<th, matchlen>=L) and never otherwise; (iii) falsifiers BOTH directions —
   corrupted table => no fire (not a crash), margin above threshold => no promote.
   All rows + g++ command lines banked in results/amd/d2ngram/.
4. RED->GREEN CLOSURE LAW: the cell guards the CLASS (any low-margin runner-up promotion
   decision), not the instance. The cell joins the standing suite (zero-card class).

## PHASE 3 — SERVING GATE (GPU; QUEUED behind G-BF-1 and the firstchunk-leg window)

Pre-registered (do not loosen): ordinal-paired fresh boots (PLOG-060), 3x plen-2075
mt=512 per arm, NINFER_D2_NGRAM on vs off on the SAME promoted bin; gates: acceptance
(tok/round) must NOT drop (any -1pp = fail), decode t/s within +-2%, plus the pricing
gate from your Phase 1. Window claim protocol: "WINDOW CLAIM: d2-serving" in this WO +
poll `pgrep -f '^/home/chris/artifacts_bin/ninfer-serve'` empty + check
docs/amd/WO_BODY_FUSION.md tail for an open G-BF-1 claim. Retire law EXACT form for own
boots only; restore canonical serve_fast + health at close.

## LAWS

- Own worktree + branch amd/wo-d2ngram; progress log in THIS file newest-first after
  every step; a dead desk must be resumable from this file alone.
- NO channel posts (COMM LAW — server-side rejected; directs only).
- NO cmake builds; NO serve boots outside Phase 3's window protocol.
- Numbers cite measured anchors or do not exist (VRAM LAW extended).
