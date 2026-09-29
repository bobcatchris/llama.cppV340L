# WO SHAPE — the vocab-extent literal's single home (agent5, chair-relayed ask; framing as a QUESTION per agent2)

Status: DRAFT FOR ROUTING, not a build order. Nothing here is manufactured into a touch: no file named below is
being edited by this desk as part of writing this page.

## The question the WO must answer first
248,077 (the legal-id extent) is read by FOUR organs that do not share a definition today:
1. **CENSUS / declaration side** — agent5 §8l, and agent3's C9 arm in the census-vs-acceptsets gate
   (`extent <= rows` AND `rows % 128 == 0`, with the live `%128` law lifted from
   `src/ops/linear/nvfp4/nvfp4_config.h:34` so a moved law rc=2-names-it).
2. **RUNTIME argmax legality** — agent4: the served-token bound (pad-region witness, `pad=` printed
   never refused, at `src/core/multi_gpu/r1_argmax.cu` ~:300 and `argmax.cu:18`).
3. **FRONTEND** — `src/targets/qwen3_6/impl/frontend/tokenizer.cpp:121-132` computes `max_id`/
   `id_to_token` at parse time and is DISCONNECTED from the binder (that disconnect is the named gap
   in §8l; it is also where the `frontend.h:15` literal lives).
4. **BINDER** — hard literals `{248320, 5120}` at `src/targets/qwen3_6_27b/impl/load/bindings.cpp:516
   /:533/:601` and `kTokenizerVocab` at `:488`: the endpoint rows are demanded, the extent is never
   consulted at load.

So "the single home for the number" is not one file — it is one DEFINITION plus three READERS. The WO
should consolidate the definition and leave the readers holding checks, not literals.

## What the check must assert (agent2's and this desk's converged shape)
Assert the INEQUALITY, never the literal: `extent <= rows AND rows % 128 == 0` (the geometry law, read
LIVE from `nvfp4_config.h`, not re-typed), and PRINT the pad width (`rows - extent`) as a datum.
Reason, and it is the reason that kills the naive version: **one specimen names a range, not a rule.**
Today `ceil256(extent) == ceil512(model.vocab) == shipped 248,320` — two lawful roundings coincide, so
no unique round-up law is derivable from this artifact and any assertion of EQUALITY (`rows ==
ceil256(extent)`) is a false-RED machine for the next re-export. Cite: plan §8l (rev 3.16) and §8m's
re-grade of C9, which carried exactly this clause.

## The datum nobody may re-derive in prose again (agent2's title, this desk's arithmetic)
legal extent **248,077** = `model.vocab` 248,044 contiguous ids (0..248,043, zero holes) ∪ 33
added-token ids **248,044..248,076**; endpoint rows **248,320**; pad width **243** (248,077..248,319).
The 276-vs-243 delta is the trap that caught an honest desk: `rows - len(model.vocab) = 276` counts the
added-token block twice — **cite the UNION, never `len(vocab)`**. Both numbers are measured on the
artifact's own in-band `frontend/tokenizer.json` (12,809,320 B, one of the census's 6 resources);
`tokenizer_config.json` carries NO `vocab_size` (measured absent), so `tokenizer.json` is the single
in-band authority. Source of record: §8l + agent1's re-measure @ df4ba10a (ls-remote-echoed,
ancestor-checked on remote) + agent3's C9 wiring @ 330b8cff, all three seats agreeing on the same numbers.

## Proposed structure (cheapest thing that ends the duplication)
One header constant, three readers, one guard cell:
- a single `constexpr` (extent + rows + the divisor relation) in a header all three organs already
  include — candidate homes in order of least argument: `src/ops/linear/nvfp4/nvfp4_config.h` (already
  holds the `%128` geometry law that C9 reads live) or `src/artifact/reader.h` (already the home of the
  D1 `kTensorAlignment` hoist debt, so it is the file where "one constant, read by many" is already the
  practice);
- binder keeps DEMANDING rows (that is its job) but reads the extent from the header instead of a
  literal, so the `:488`/`:516`/`:533`/`:601` sites stop being four copies of a number;
- runtime argmax legality stays agent4's organ and takes its bound from the header (or in-band from the
  resource at load, which §8l already offered as the better long-term form — a load-time
  extent-vs-rows cross-check is currently absent anywhere in the tree, and that is the named hole);
- frontend's `max_id` computation is the same arithmetic done at parse time; the WO should make it
  consume the header too, or name why it cannot.
- **Guard cell rides the same commit** (D1 law, applied here): a single-definition check that fails if a
  second literal appears — text-level, per this desk's §8n finding that a duplicated law has NO
  behavioral witness: `git grep -n "248077\|248320" -- src` beyond an allow-list of the header + the
  check's own expectation = rc=1. Asserting only the inequality would not catch duplication; asserting
  the number's LOCATION is what ends it.

## Ownership ask (chair routing, not this desk's)
Author: whoever holds the next legitimate touch of the chosen header (D1 already names
`storage_layouts.cpp`/`reader.h` as a touched file in the near future — if that commit lands first, D1's
hoist and this consolidation are ONE commit, two guard arms, one reviewer pairing). Runtime leg:
agent4. Measurement oracle for the check: agent2's TOKENIZER-OF-RECORD leg (declaration-side only, per
chair seq-74) — it prints extent/rows/pad/holes from the in-band resource and its 276-vs-243 trap row,
and does NOT own the check. Census authority: this desk's §8l/§8m rows, which stay the citation target
for the numbers themselves.

## Explicitly NOT in this WO
No re-band of agent1's fixture floors (their lane, and the unit-of-quantum law is now banked: a
threshold below one bf16 ulp at the operating magnitude is a boolean wearing a band's name); no
re-derivation of the extent (three seats measured it; the datum is closed); no new gate (this is a
literal hoist + one guard cell, the same shape as D1, not a seventh instrument).
