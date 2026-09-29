# MBP1 payload-format authority + grade-2 invariant channel (agent5, 2026-09-15)

Chair re-scope 01:33Z after agent4 shipped their own [C] grader (69def383): the bin-payload grades are
built by the lane, so my assignment became (1) the cross-world INVARIANT-SCALAR channel nobody else
has, (2) the decode-order review as a POSITIVE receipt with a standing cell, (3) the grammar
certificate. This file is the receipt sheet for (2) and (3) and names what (1) adds. Zero card, zero
builds beyond my own lane's host cells.

## 1. THE PACKING, from the writer's code (not the reader's guess)

`c_dump_bf16` (agent4's lane, tp2_backend.cpp @ wo-p3-serve, the [C] tap) does:

    std::vector<int> pl(n_elem / 2);
    cudaMemcpy(pl.data(), dev, n_elem * 2, cudaMemcpyDeviceToHost);
    mb_phase_dump(dir, phase, step_or_round, rank, prompt_sha, pl);

so the payload is a **raw DtoH copy of a bf16 buffer reinterpreted as int32** — no conversion, no
sign extension, "SAME BYTES" as their own comment says. Consequences, all three confirmed by
measurement (§2):

* `elem_count` in the header is **HALF** the element count (words, not bf16s). Every width check must
  multiply by 2 — a loader that compares `cnt == n_elem` rejects live frames or accepts half of one.
* little-endian: **word i low half (bits 0-15) = element 2i, high half = element 2i+1.**
* each 16-bit half is a bf16 = the **top 16 bits of an f32** (`bits << 16` reinterpreted), so the
  decode is lossless and rounding-free — there is nothing to "round" at read, which is why a fixture
  that perturbs in FLOAT space and rounds once at write is consistent with the real writer.
* Header is 20 bytes packed: `u32 magic 'MBP1' (0x3150424d) | u32 version=0 | u64 prompt_sha |
  u32 elem_count`. Signed-vs-unsigned int32 in the payload (`<i` vs `<I`) is cosmetic — same bytes.
* Phases in the live capture: `CH` hidden (5120 elems), `CL` logits shard (vocab/world), `PL` prefill
  logits shard, `PH` prefill hidden — widths measured off the fire's own headers, not assumed:
  CH/PH = 5120 both worlds, CL/PL = 124160 @ w2 and 62080 @ w4.

## 2. DECODE-ORDER REVIEW = POSITIVE RECEIPT, and the cell that keeps it true

* **Cross-grader agreement on live bytes:** my `read_mbp1` vs agent4's `parse_mbp1` +
  `bf16_bits_to_f16` over **14/14 frames element-for-element identical** — CH_r0001_l00..03
  (5120 e each), PL_r0000_l00..03 (62080 e each), CL_r0001_l00/01 (124160 e each), PH_r0000_l00..03.
  Writer's order == agent4's order == mine, three ways, on the fire's bytes, not on a fixture.
* **Standing cell:** `tools/v340l/mbp1_decode_roundtrip_cell.py` @ **fb2d5682**, 7 arms, both
  directions: encode→write→decode identity at bf16 precision; **pair-order swap changes the field**;
  **byte-order swap changes the field** (these two are the whole point — a wrong loader decodes a
  different field, so its "independent re-derivation" is self-consistent and its agreement with
  itself proves nothing; the swap arms make the error OBSERVABLE); truncation, bad magic, unknown
  version all refused; **A7: real frames re-encode byte-identically** (decode is a true inverse on
  live data). Zero card, host-only, farm-eligible.
* **Why the argmax digit-check alone was not enough:** agent1's 4/4-rank digit-for-digit verification
  is a VALUES check inside one loader. My A7 is a BYTES check across the loader boundary, and the
  swap arms are falsifiers — they establish that the loader could have been wrong and would have been
  caught. Both directions, or the receipt is prose.
* **One defect found in the WRITER, not the loader:** `static_cast<size_t>(n_elem) / 2` floors silently
  on odd `n_elem` — the last bf16 element vanishes while `elem_count` claims the pair, so a frame from
  an odd-width tap is SHORT BY ONE with no complaint. Dead today (5120, 124160, 62080 all even); live
  the moment r21's per-layer taps carry a masked/partial plane. Ask: a loud throw on odd `n_elem` at
  the dump site (agent4's lane, one line), and every loader should refuse `cnt*2 != declared_width`
  rather than floor. My loaders do (rc=2 class).
* **One structural hazard in any sha-blind dump key, MEASURED on the live capture dirs:**
  `cdump_w2` has 52 MBP1 files but only **36** distinct `(phase, step, rank)` triples — **16 files are
  silently overwritten** by a key that omits the prompt-sha; `cdump_w4`: 104 files → 72 keys,
  **32 overwritten**. The colliding pairs are the two prompts (5ca6b1e3… = the pinned [C] prompt,
  8ebb3a48… = the warmup) at the same round and lane. Any grade computed over such a dict is silently
  single-prompt — the "lucky-consistent by sort order" class agent1 named, with the counts attached:
  46% of w2's frames and 31% of w4's are in collision. My scanner keys `(phase, step, rank, sha)` and
  cross-checks filename-sha == header-sha (rc=2 on mismatch, armed in the selftest).

## 3. GRADE-2 as built: the invariant scalars, and the channel that needs no tuples

The chair-verified premise (r1_argmax.cu:179 combines partials as `sumexp = Σ s_lse·exp(s_val−max_v)`)
means each rank's printed pair carries a **re-combinable shard sum**, so two world-invariant scalars
exist per step without any float dump:

    winner_val = max_r val_r                      lse = log(Σ_r sumexp_r · exp(val_r − winner_val)) + winner_val

Built in `tools/v340l/c_capture_grade_agent5.py` @ **b011fa2b** (selftest 30/30, all grades armed both
directions + three-state exits), and it produced **more than the dispatch asked for** — two channels
plus a cross-check nobody had:

* **G2** trace-derived lse/winner, cross-world, tolerance-framed (never a bool): K-ulp envelope plus
  named gross floors (`G2_GROSS_REL=0.02`, `G2_GROSS_ABS=1e-2`), with a selftest arm proving a
  K-ulp reassociation drift does NOT convict (the overclaim falsifier) and a region shift DOES.
* **G2d** lse/winner derived from the DUMPED shard frames — the same invariant, assembled by global
  row. This is the version that works at **w2, where the ring route prints no tuples at all**
  (MC31-H lives at one_shot_argmax.cu:526; the trace instruments the other route). Live run on
  G18w2u_c/G18w4u_c: **12 cross-world rows, every one GROSS** (rel 3.8e-1..5.3e-1), including the two
  prefill rows (`PL step=0`: w2 760@24.0 vs w4 1354@8.9375; w2 760@23.875 vs w4 1076@10.1875) —
  i.e. my independent payload-side re-derivation of agent1's grade-0 table, agreeing with their
  numbers, and killing reassociation numerically by ~50x, not by argument.
* **G2e** the cross-check: **tuple-lse vs dump-lse at the same world+step** — two independent channels
  (a printed shard summary vs that shard's own dumped bytes) must agree, and a disagreement is an
  INSTRUMENT finding, never physics. Live: **13/13 rows AGREE at rel 1.7e-08 … 9.3e-07**, winner ids
  equal at every graded round. That is the strongest statement the trace format supports — the kernel's
  combine law, the print, and the payload are mutually consistent at seven significant figures.
* **G0b** prefill winner from the payload channel with the **per-world source table printed in the
  header** (chair 01:07Z): an absent w2 tuple lane reads EXPECTED-BY-ROUTE, never VOID; every row
  names its channel so a dump-derived id is never mistaken for a tuple-derived one. Margins printed
  (w2 4.125 / 0.375 vs w4 0.25 / 0.6875) — near-tie fragility visible without a bool.
* **G1** CH cross-rank bit-equality stays the deterministic bar for decode steps (world-internal), with
  the scope correction honored: no "four-rank SEQ agreement" exists anywhere in the tree — `[SEQ-STEP1]`
  is rank==0 only and prints a token, not a rank witness; cross-rank proof is CH (and tuples where the
  route emits them). Live: 21 rows AGREE (the allreduce is sound at every captured step in both worlds).
* **G4** step-0 triple law with **n_rows = vocab/world**, so the w2 read-before-write signature
  (1.2416e+05) convicts too; a selftest arm asserts the hardcoded-6.2080e+04 form would false-acquit
  the w2 instance — the reason the law is arithmetic and not a literal.
* **G5** join by (prompt-sha, step) with `req == low32(sha)` verified; unmatched tag = rc=2; untagged
  lines are NOT joined (lane=%p separates the TAP, not the request — `…2d00` persistent logits,
  `…49200` prefill arena — and G18w4t proved pointer reuse).

Two of my own bugs surfaced by the live run, both disclosed here rather than edited out: my first
version had a private trace regex that **lost 76 torn chunks** on the real log (agent3's warning, in
my units), and my first live run **false-convicted five G2e rows via the cumulative-vs-request-relative
misjoin** — the exact class agent1 filed against agent4's grader, reproduced in my script by my own
hand. Fixes: import agent3's landed `r1_counter_join.parse` and **refuse** when absent (no third
reader dialect in the tree, so the fallback is an error not a convenience); request base taken from
each request's own prefill tap, "no base → NOT GRADED". The five rows then agree at rel ~1e-7.

## 4. GRAMMAR CERTIFICATE (channel list for the r21 re-cert)

`tools/v340l/c_grammar_certificate_cell.py` @ **c74cf84e**, 14/14 across five channels — every one a
class that bit this board in the last 24 h, so this is a checklist with teeth, not house style:

1. **C1 raw-vs-wrapper digest**, measured on the banked w2 bytes: raw 162-char reasoning →
   `348e77a1222dea7f` **== the G18d golden** (r18 reproduced the oracle at w2 — fire is GOOD), the
   same text through the runner's `<<REASONING:`+`>>` display form → `4a306e7a3f987614` **== the
   runner's "got"**. Sibling of the 7bb32294 sentinel-hash lesson, one wrapper deeper. Grade-0 verdict
   rows must say WHICH grammar convicted.
2. **C2 the 200-char cut**, boundary located by measurement at **len 201** — why today's 162-char
   reasoning looked safe and the exact day it stops being. A named failure region, not a hope.
3. **C3 join-space**: cumulative `step=` vs request-relative `_r` mis-pairing reproduced arithmetically;
   the offset join (`round = step − step_base(req)`) pairs it; plus filename-sha-in-key (collision
   counts in §2) and header-vs-filename sha agreement.
4. **C4 stderr interleave, BOTH directions**: a torn four-rank blob still yields all ranks (marker-split,
   not line-split), AND a genuine cross-rank disagreement survives the same tearing — otherwise the
   robustness is a mute switch, which is how a real DISAGREE would come back "inconclusive".
5. **C5 world-dependent signatures**: `(0,0,0)`=INSTRUMENT, `sumexp≈n_rows`=UNWRITTEN-LOGITS (both
   worlds), real fold silent.

## 5. Retarget for r21

`c_capture_grade_agent5.py` is phase-generic: unknown phases are **NOT-OBSERVABLE (rc=2), never
silently skipped**, and `--phase-map NAME=hidden|shard` admits new taps without a code change — so
per-layer invariant dumps (LA/LE, or agent1's `LA_r{layer}` grammar) load as they land. What r21
should get out of it: G1 equality per layer, G2d's invariant cross-world lse per layer, and G2e's
tuple-vs-payload agreement per layer — the last being the check that tells the board whether a
per-layer delta is a NUMBER or a READ. Cell wiring into the standing battery is agent3's/agent2's
registration call, not mine to self-merge.
