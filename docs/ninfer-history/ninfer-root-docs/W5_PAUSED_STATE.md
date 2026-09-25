# W5 PAUSED STATE — fix-design material frozen at the user's pause (08:50 EDT)

**Status: PAUSED by user decision ("potential huge win, but paused").** No active design
or implementation work. This document freezes everything gathered so far so the dig
resumes with zero loss.

**RE-ARM TRIGGER (the coordinator surfaces this at every queue planning step):**
> When a GPU window is free AND the user says "W5 re-arm" — resume at §RE-ARM below.

**Nothing is lost:** the rank-race fix + per-rank conf_break fixes are already in pushed
main (`e278bd4f`, dormant while W5 is off); the open defect's isolation evidence lives in
`results/radiance_w5/`, `/tmp/led_off.log`, `/tmp/led_on.log`, `/tmp/led_t001.log`, and
WO §18 (trace shas: tau-OFF reference = `1c00977c`).

---

## 1. The defect (A1's completed dig, reproduced + isolated)

SHORTENED VERIFY ROUNDS diverge in the kvarn staged-shadow path past the first shadow
window. Isolation: 3,258-token prompt, k4v4, single-seq, merged fix binary:
* tau=0.01 (confidence gate ON, 0 breaks): **byte-identical** to tau-OFF → the confidence
  machinery (prod_conf, thresholds, gate) is exonerated.
* tau=0.7 (62 shortened rounds): **diverges at the FIRST rk=1 round** (round 17): the
  width-2 verify accepts draft 11346 where the width-4 reference accepts 11782 at the
  same stream position. Rank-consistent (both ranks agree), deterministic.
* Length dependence: a 2k prompt never leaves the first staged-shadow window
  (2000+43 < 2048 staged capacity) → never reproduces. Past the shadow boundary
  (2048 tokens), the FIRST width-2 round diverges.
* Conclusion: **staged-shadow window-slide × reduced-verify-width interaction** — either
  the slide/rebase arithmetic or the KV read window at width 2 mis-positions the target's
  column-1 logits.

## 2. Route map — ⚠️ SUPERSEDED-PREMISE (user-caught, 2026-09-08 morning)

**The §2 route map below rests on a STALE PREMISE and is retained only as history.**
The staged bf16 shadow / "the wall" was REMOVED per docs/83 — verified in the current
tree: `tp2_backend.cpp:763-770` ("the staged bf16 shadow ('the wall') is REMOVED...
stage_pages is pinned to 0; has_staged stays false so the (now-dead) shadow early-returns
and staged-tail wiring never fires") and `kvarn_workspace.cpp:269/291`
(`ws.staged_pages = 0` / `= stage_pages` with stage_pages pinned 0). The
"text_context_impl.h prefer-BF16-flash branch" cited below still EXISTS in the source
but is DEAD CODE under stage_pages=0 (`need_pages <= staged_pages <= 0` is never true).

**Consequence for the route map:** there is no shadow↔pool data-source switch at 2048 —
every verify round (any width, any length) runs the packed/materialize path from the
codes. The observed asymmetry is REAL (width-2 diverges past 2048, width-4 does not;
sub-2048 never diverges) but its mechanism is NOT the shadow boundary.

**NEW 2048-BOUNDARY CANDIDATES (A1 re-deriving, fresh suspects):**
  * N1 — committed-vs-open-tile boundary: 2048 = 32 × 64-token kvarn pages. The first
    32 pages may take a different serve/commit path than pages ≥ 32 (workspace-tile
    residency, commit cadence, or a capacity constant sized 32 pages).
  * N2 — GDN chunk-frontier at 2048: the chunked prefill's GDN checkpoint
    ("within one chunk of the end", tp2_backend) and/or the chunk-limit arithmetic
    crosses a boundary at ~2048 tokens; the first armed shortened round after a
    deep-position checkpoint may combine state differently than the 2k case.
Both are consistent with the isolation table below (which remains VALID: the divergence
is real, deterministic, width-2-at-first-shortened-round, past ~2048 tokens).

---
**(SUPERSEDED historical route map — do not design against it):**

The kvarn attention path branches on verify width T and shadow capacity (staged_pages,
in tokens = staged_pages × 64; first window = 2048):

* **Sub-wall** (need_pages ≤ staged_pages): BF16 flash over the STAGED SHADOW
  (bf16 mirror), tail staged from the workspace tile each call
  (`text_context_impl.h` kvarn attend, "prefer BF16 flash" branch).
* **Over-wall** (need_pages > staged_pages):
  * T=1 (plain decode) and T=2..6 (MTP verify rounds) → **PACKED decode kernel by
    default** (`packed_verify = tokens>=2 && tokens<=6 && !force_materialize && !k5v4`;
    `NINFER_KVARN_VERIFY=materialize` is the A/B escape hatch) — reads POOL codes +
    side-table scales + the in-flight tail directly; the shadow is not read.
  * T>6 → materialize + 2-pass flash (prefill chunks).

**[SUPERSEDED — see the premise flag above. The surviving empirical core:]** the
first SHORTENED round past ~2048 tokens diverges; width-4 rounds (and tau=0.01, no
shortening) are correct at every length. The 2048 boundary is real; its mechanism is
being re-derived (N1/N2 above).

## 3. Design questions (the three assigned)

**(a) Slide/rebase path when verify width < 4.** The shortened round verifies positions
[p .. p+rk] with rk+1 = width-1 columns instead of the nominal k+1=4. The round's
position/column bookkeeping (anchor, prep/align, d0-propose, verify extents) derives
column offsets from the NOMINAL chain shape; if any component derives from the nominal
width while the actual verify ran short, column-1 (the second verified position) reads
the wrong KV column or the wrong draft alignment. Code-read needed: the width plumbing
from `drafts_produced()` into the attend call (which of anchor/rope/envelope are
width-derived vs nominal).

**(b) KV read window at reduced width past the slide boundary.** The packed kernel takes
`packed_pages`, `tail_count`, and infers `packed_pages = envelope.max_visible_keys/64`
when the caller passes -1 (launcher note: a wrong value there historically zero-filled
every tile — "silently ZERO output", the unified-route incident). At the first over-wall
width-2 round, the visible-keys envelope and the tail geometry must describe the
SHORTENED chain, not the nominal one. Code-read needed: `envelope.max_visible_keys` and
`tail.tail_count` at the shortened round — if they encode the width-4 window, the packed
kernel reads one stale column for the target's column-1.

**(c) Where the fix belongs.** Working conclusion: in the SLIDE/REBASE + WIDTH
BOOKKEEPING (runtime), NOT the kernel. The kernels are width-generic by design
(`materialize_kernel_w` exists for non-legacy tiers; the legacy k4v2 kernel body is
byte-frozen by the docs/69 B2 hazard note — "even semantically neutral edits shift MTP
acceptance on prefix-restored requests" — so a kernel-side fix on the legacy tier is the
LAST resort and would need its own byte-identity proof). The runtime must hand the
kernel the true width-derived geometry; the kernel already handles arbitrary geometry.

## 4. Candidate fix shapes

* **F1 — explicit-width plumbing (preferred shape):** carry `width = drafts_produced()+1`
  from the break decision into the verify attend call (positions, envelope
  max_visible_keys, tail_count) as the ONLY source of truth; remove every nominal-width
  derivation on the shortened path. Lowest risk; touches runtime bookkeeping only.
* **F2 — post-shorten rebase:** after a shortened round, re-run the shadow/rebase
  normalization (stage-from-tile + recount) so the next round starts from a
  width-independent state. Heavier (extra copies) but also heals any residual
  state drift.
* **F3 (last resort, kernel-side):** fix the packed kernel's column-1 indexing for
  width<4 on the legacy tier — requires byte-identity proof against the frozen-body
  hazard note. Only if F1/F2 demonstrably cannot reach the misaligned read.

## 5. Open question: is it width-2-specific or width<4-general?

tau=0.7 diverged at the first rk=1 round; no width-3 round occurred before divergence.
The dig must produce a width-3 isolation case (tune tau so the first break fires at step
2 → width 3) before the mechanism is called "width-2-specific". If width-3 also
diverges, the defect is "any width < nominal"; if width-3 is clean, the defect is
specific to the LAST-step break (drafts_produced()=1 — the d0-only round), which points
harder at the d0/anchor alignment (candidate (b)).

## 6. Validation plan (the bar, per the coordinator)

1. tau=0.01 == tau-OFF byte-identity stays green (already proven — must not regress).
2. tau=0.7 @ 3,258 tok: the width-2 verify must accept the SAME draft as the width-4
   reference at the same stream position (round-17 acceptance: 11782, not 11346).
3. New width-3 isolation case (per §5) byte-identical to the width-4 reference.
4. The 200k-scale bar: tau-ON int8 10k+ guard cell (currently hanging) completes.
5. Full-generation byte-identity vs tau-OFF at 3,258 tok with W5 firing throughout.
6. The matrix (docs/optimizations/45 §6 A/B/C cells) re-proven green with breaks firing.

## §RE-ARM (resume checklist)

1. Confirm GPU window free (guard; cards ≤20 MiB).
2. Re-read this doc + WO §18 + A1's traces (/tmp/led_*.log, results/radiance_w5/).
3. Reproduce round-17 divergence on the merged fix binary (tau=0.7, 3,258 tok, k4v4).
4. Execute the code-reads from §3(a)/(b) — the width plumbing and the envelope/tail
   geometry at the shortened round.
5. Fix shape per §4 (F1 preferred), validate per §6, then W5-ON flip decision returns
   to the user (with the flip-decision table from 23:36 EDT).

---

## 8. FIX-DESIGN CONSULT (A2, for A1's §18 dig — the append/rewind reordering)

**The invariant, decomposed for the pool model.** "Conf-gate must not change
speculative append/rewind state outcome" decomposes into:
* I1 — the append SET is a function of drafts_produced (inherently variable under W5;
  fine — fewer drafts is the feature).
* I2 — after the round's rewind, the MTP pool/tile state must be INDISTINGUISHABLE for
  every next-round read from "the accepted tokens had been appended canonically" —
  regardless of how many drafts were produced.
* I3 — no pool bookkeeping decision may depend on sync timing (only on the decided
  chain shape, which is deterministic).

**Where the reordering happens (code-path localization).** The conf path syncs PER
CHAIN STEP to read the confidence (host-mapped argmax slot). The pool appends for chain
step i are enqueued before that step's sync; the break decision arms `pending_depth`;
the chain exits; the VERIFY runs at width pending_depth+1; then `kvarn_rewind_mtp` +
accept bookkeeping. The non-conf path enqueues the FULL chain's appends before any
rewind. The rewind therefore executes against a DIFFERENT tile fill level:
* full chain: the open tile crossed (or neared) a page boundary per the nominal k;
* shortened chain: the open tile holds produced+accepted codes only.
`kvarn_rewind_mtp` → `rewind_layer` then takes different branches (`tile_page > page` →
`discard_partial` vs `tile_page == page` → `rewind_tail`) and leaves a different-but-
LEGAL tail geometry (tile_page/tail_count/content) → the next round's draft head
attends its own simulated KV through that different geometry → proposals shift → the
accept stream diverges. This is I2's violation: the post-rewind state is legal but not
CANONICAL for the accepted prefix.

**Option evaluation (pool-model risk):**
* **OPT-1 — commit-at-round-end (RECOMMENDED).** After the verify + accept, commit the
  accepted-prefix tail (`gqa_kvarn_commit_completed` on the open tile) BEFORE the
  rewind bookkeeping, so all accepted codes land in the pool and the workspace tile
  re-opens EMPTY; the rewind then only drops pages ≥ the accepted boundary. Post-round
  state = canonical by construction (all accepted codes committed; open tile empty and
  identical in both paths) → I2 holds for any chain length; I3 holds (one decision,
  post-verify, derived from the accepted count). Uses only the existing commit/hydrate
  machinery — no kernel changes, no new buffers. Cost: one tile commit per round
  (bounded, the commit path is the proven hot path). Frontier interaction: the commit
  advances `tile_page`/resets `tail_count`, so the NEXT round's
  `prepare_page_for_append` starts from a page-aligned empty tile and hydrates the
  unaligned slot from the committed page — the standard path.
* **OPT-2 — unconditional appends/rewind + gate verify-width only.** Invariant holds
  TRIVIALLY (the append/rewind sequence is bit-identical to the non-conf path), and it
  removes the sync from the pool path entirely. COST: the draft chain must run to full
  length (the chain-shortening savings vanish) and the verify-width reduction still
  needs width-neutral reduction geometry (derive the split/tiling schedule from the
  NOMINAL k_buf+1, not the actual width — otherwise the same near-tie flip re-enters
  through the kernel). Fall-back shape if OPT-1 cannot be proven; changes W5's
  savings profile.
* **OPT-3 — double-buffered draft state.** Conf-path speculative appends land in a
  scratch mirror; the accepted prefix is materialized canonically post-verify. Restores
  I2 but is the most invasive (scratch sizing, extra copies, a second rewind path) —
  hold as the heavy hammer.

**Sequencing note for A1's dig:** the discriminator between OPT-1 and "the appends
themselves differ" is already in the traces: if `[w5d-acc]` accept divergence appears
ONLY after the first post-shorten round (and the shortened round's own verify argmax is
correct — which the tgt2 logs confirm), the append SET is fine and the state
canonicalization (OPT-1) is the whole fix. If proposals diverge WITHIN the shortened
round itself, an append-side reorder exists too and OPT-2's unconditional sequence
becomes primary.
