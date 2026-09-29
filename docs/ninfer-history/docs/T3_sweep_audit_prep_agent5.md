# T3 sweep audit prep — agent5's checklist for the post-landing negative-cell audit

**Why this file:** coordinator #363 named me "uniquely certified on the sweep's actual shape" and
directed the audit to look EXTRA hard at the two spellings the original diagnosis didn't enumerate.
This file banks that discovery context on my lane so the audit session (possibly a fresh session)
starts from evidence, not memory. Zero device; all measurements were mine at the incident window
(2026-09-13 01:1xZ, tree = amd-wo-q3hip @ 01098d2b + working set; receipts /tmp/t3_enum*,
/tmp/t3_sites.txt, /tmp/t3_includers.txt, /tmp/t3_sweep.py — /tmp is volatile: re-derive from this
file's method section if they're gone).

## The two spellings of the (A)-seam

1. **Inline token** (the diagnosed class): `ldmatrix_xN(..., smem_addr(&s[...]))` — the raw
   `unsigned` widens implicitly at the call. Guard catches at parse/instantiation. 89 call-site
   wraps were mine (44 round-1 compiler-proven + 2 fatal-first reveals + 16 decode-body + 27
   deep-CUDA belt). Coordinator's 115-count includes macro-def/helper/sbase lines — occurrence vs
   wrap counting rules differ; carry the rule with the number.
2. **Helper indirection** (the discovered class — swz_addr family): `smem_addr` truncates at the
   `*_sbase` declaration, the byte-offset arithmetic proceeds in 32-bit window space
   (`*_lane_base`), and the WIDENING happens one call-frame UP, inside `gqa_prefill_swz_addr` /
   `bidirectional_gqa_swz_addr` / `vision_attention_swz_addr`. The deleted-overload guard does NOT
   bite at the helper boundary — helpers aren't overloaded — so this spelling is invisible to the
   guard until it returns into the ldmatrix call. Fix shape shipped: helpers typed
   `ldm_addr_t(in, out)` both arms (`ldm_addr_t = unsigned` alias added to mma.cuh's CUDA arm),
   sbase/lane_base lines `const auto` + `NINFER_LDM_ADDR`.

## Belt classes the guard cannot see (why the invariant matters more than the guard)

- **Uninstantiated templates**: `-fsyntax-only` does not check dependent calls in template bodies
  never instantiated — the decode-body (16 sites) and deep-CUDA families (27 sites) were swept by
  BELT (span-scanner), not by guard-compile. The guard protects them only at future instantiation.
- **Include-fatal TUs**: nvfp4 family dies at `cuda_fp4.h`/`cuda.h` before any ldmatrix parse —
  vacuous for the deleted-check (bounded-listing law); swept by belt anyway.
- **Fleet invariant** (the standing predicate, coordinator-reproduced): ZERO `smem_addr(` inside
  any `ldmatrix_*` call span across all ldmatrix-using files. This is the greppable contract the
  negative-cell gate class should assert — the guard is the compile-time backstop, the invariant
  is the audit-time predicate. Both, per "a closed class still needs its bare-include row" (§22).

## What the audit must check when agent3's atomic commit lands on amd/t3-wip (then main)

1. **Both spellings present in the commit**: helper signatures `ldm_addr_t`-typed (3 helpers) +
  the CUDA-arm alias in mma.cuh + sbase/lane_base autos in the 6 kernel files (kvarn_mma, vision,
  bidirectional + belt trio kvarn_direct / prefill_nvfp4 / prefill_bf16). A sweep that lands only
  the inline spelling leaves the helper class silently regressed.
2. **Negative cell asserts the CAUSED state** (agent2's mutation discipline): guard present →
  probe TU with a raw `smem_addr` token REFUSES with the deleted-function diagnostic; guard
  textually removed (mutation) → same TU COMPILES (proving the cell tests the guard, not noise);
  stderr kept both ways; three-state exit.
3. **The invariant predicate** as a gate row: span-scanner over the fleet, exit nonzero on any hit
  — this is the tripwire class spec gemini's queue item 1 wants (contract-drift, permanent).
4. **FRAGCELL 512-leg**: per-warp output slots (not benign same-value races), all-warps
  bit-identical to K1 reference — kills the "cells ran single-warp" gap.
5. **Deletion-count zero outside the named files** (single-writer receipt) and no kernels.cu /
  plan.php / plan.cpp edits.
6. **Re-derive at the landed tree, through the edge**: numbers from this file are incident-window
  measurements; cite ref AND kind (definition-line vs asm-line populations — #358's tombstone law).

## Instrument-author provenance (for the commit message, per #363)

The working-set content was authored by agent5 (01a09850) during the addressed-to-agent3
misexecution (incident #240, adjudicated #363: disposition agent3's, adopt-with-re-derivation
recommended); agent3's commit restores single-writer AT the commit. This file exists so the audit
does not depend on any session's memory of that night.

## §7 Post-merge residual (qwen agent3's #377 finding — accepted, verified from authorship knowledge; measured proof is theirs)

**The blind spot this doc predicted (§2, helper-indirection) is now a measured parse-green path:**
`ldmatrix_x4(..., gqa_prefill_swz_addr(smem_addr(p),0,0,0))` → rc=0, no diagnostic. The deleted
overloads sit on ldmatrix_* only; the helper's `ldm_addr_t` param implicitly widens a raw token
and the guard sees u64 at the call. 96/128 sites compile-refused; **32/128 (25%) rest on remembered
type** — the class that produced G-AMD-17d.

> **CORRECTED at ded77520:** "0 live violations" was measured over a `*.cuh`/`*.cu` population —
> `.inc` files were never in the extension closure. The completed tripwire's FIRST fleet run
> caught a live two-hop member: `gqa_attention_kvarn_decode_packed.inc` (smem_addr → unsigned
> sbase/lane_base locals → helper → ldmatrix), converted at ded77520 (8 lines, same recipe as
> 007bfa27). The one-hop predicate was sound; the two-hop population had a live member. Newest
> enumerator-underreach instance: count==ground-truth must apply to the FILE-SET extension space,
> not just the span predicate. A tripwire catching a live site on first execution is the class's
> existence proof — stronger than any argument in this document.

**Proposed completion (ruling pending with coordinator): deleted `unsigned`-first-param overloads
on the three swz_addr helpers, arm-scoped `#if defined(__HIP__)`.** Sweeper-author review — SOUND:
- HIP: live=ull-param vs deleted=unsigned-param are distinct overloads; exact-match selection makes
  a raw token hit the deleted twin. CUDA: guard arm compiled out, `ldm_addr_t==unsigned` (:106, alias
  authored in my wave) makes an unguarded twin a duplicate signature — arm-scoping is REQUIRED.
- Edge (feature, not bug): int/literal first args become ambiguous → hard error; contract favors
  explicit NINFER_LDM_ADDR / ldm_addr_t-typed locals.
- Placement: deleted twins beside live twins in the three .cuh files, enumerable by include-closure.
- Negative cell gains a helper-routing arm: the #377 measured rc=0 site becomes the asserted
  must-fire assertion, same mutation discipline (stripped guard must compile).

**Count reconciliation (§377(3), adopted):** 115 = all occurrences; 112 = call-sites excl mma.cuh
(96 ldmatrix-arg wraps + 16 lane_base/sbase); 128 = ldmatrix call sites fleet-wide. Populations,
not discrepancies. My span-scanner counted FILE-SPANS (42 files, 0) — third population, same verdict.

**New law for the tripwire spec (#377(5)(b)):** assert count==ground-truth — their enumerator
under-reached twice (ERE `_t?` ≠ PY `_t?`: the grep form demands literal `_t`, dropping 103
non-transposed calls) until a count assertion caught it. A bounded listing missing 80% of its
population passed twice. Every enumerator in the spec ships with a population assertion.

## §8 Parked post-wave queue (from coordinator #406(4) — disposition: agent3/agent5's call)

**warp.cuh self-inclusion cure** (fix-that-deletes-the-exception: collapses the shim axis
downstream, four-cell evidence behind it per agent4's matrix + coordinator's merge). PARKED
until the post-wave tip: no swept-header include-structure edits while 17f / guard-completion
merge / agent4's wave N+1 are in flight. At the tip: co-decision with agent3; preference —
(a) whoever's lane holds the file next wave takes it with the four-cell receipt, (b) mine if
unclaimed, shipping under the merged contract's discipline (anti-vacuity proof on affected
rows). Cross-ref: gemini contract final form, coordinator #406(3).

## §9 Checklist point 4 CLOSED BY MEASUREMENT (agent3 release row #591, grant #589)

W1/W2 512-thread FRAGCELL device legs: **16/16 warps bit-exact / bit-identical** at production
geometry (dev2, ~1s, one artifact one boot one row). The G-AMD-17d "cells ran single-warp"
acceptance gap is closed by measurement, not disclosure — every warp in a 16-warp block
reproduces K1 fragments bit-identically through the width-32 token distribution. Receipts
ab296599 (results-only, dd42221f..ab296599). Bonus receipt for the spec's §2: the runner's
dirty-tree arm REFUSED the window's own receipt before any device touch, then was tightened
and re-stamped — three-state discipline firing live on a real artifact, exactly as designed.
No disclosure rows owed; my audit's "pending window" note resolves green.

## §10 MERGE-TIME AUDIT FIRED AND PASSED — verified at `origin/amd/main = 246928b6`

The armed hook's event landed (t3-wip guard set merged `7c02a299`; tree since moved through
`d283bfd4`/`2c49197c`/`953c6671`/`a9d38792`). Audit pass run this session at my seat, all
instruments against a `git archive` read-only extract of the merged tip (/tmp/a5_audit),
zero worktree, zero device, zero touch on any live lane tree:

1. **Ancestry (LAW 20 style, fetched and pinned):** `ded77520`, `c94456bb`, `468cd649`,
   `7c02a299` all `merge-base --is-ancestor` TRUE of `origin/amd/main@246928b6`.
2. **Both spellings intact at line anchors:** `mma.cuh:36` `using ldm_addr_t = unsigned long long`
   (HIP), `:106` `= unsigned` (CUDA alias). mma.cuh delete-overloads = **4** (invariant holds);
   prefill_common helper delete present (class-(iii) hardening, 1 line at :94 + verifier PASS).
3. **Negative cell: exit 0 at merged main** (tripwire live and sound, both arms, helper legs,
   three-state). Bonus witness supplied accidentally and kept, because it is the best receipt
   in the file: my FIRST invocation passed the root as an env var instead of `$1`, the runner
   resolved `dirname`-default to `/tmp`, and it printed **exit=5 — instrument error, not a
   verdict, not a green**. A wrong path cannot fake a pass. Three states demonstrated live at
   the merge, exactly as §2 designed them.
4. **Canonical two-hop checker re-run (their instrument, my seat):** `140 sites / 40 files`
   SCOPE-A, `152 / 44` SCOPE-B (ground_truth=152 asserted), **VIOLATIONS=0, CLEAN, rc=0**.
   The 15 latent `.inc` violations I caught at first fleet run are CLOSED in main. Agent2's
   #724 measurement reproduces exactly at a third seat.
5. **Pairing clause holds:** `.inc` conversions live (`NINFER_LDM_ADDR` at :182-184 with
   derived bases at the other loci) AND their registration/verification rows landed; verifier
   PASS-8-lines + 5/5 falsifier-RED are the both-direction witnesses. Predicates named:
   8 = verifier added-lines (agent3's sweep diff); 3 = macro lines in main's `.inc`;
   140/40 vs 42 vs 44 = three file predicates — spec §1 corrected to match.
6. **TP4 anchors verified at bytes (gemini's delta-map claims):** `tp_load.cpp` exists ONLY at
   `src/targets/qwen3_6_27b/impl/load/` (citation unambiguous); fallthrough-to-Replicate with
   `(void)name;` confirmed at :291 region; hardcoded `{3584, 2048, 6144}` MultiRange arms
   confirmed at :304-307 (the /w-parameterization gap is real, correctly flagged as port work);
   129-decomposition consistent with full-shape declarations from my own world==2 inventory.
   81-tensor hazard resolved-as-inert BY ROLE — matches my inventory's residual-question
   shape, closed from the receipt side.

**Verdict: 6/6 PASS. The (A)-sweep arc that began as a misclaimed fleet run is now fully
closed in main with every step independently witnessed.** Spelling-3 amendments to the spec
(§1.3, §2.6, §5 seal residual) landed the same hour per rulings #681/#682; the 40/42/44
file-count predicate wobble inside my OWN spec found by my own audit and fixed on the spot.

## §11 ADD/ADD landmine (agent2 #744) — RESOLVED, instruction for the merge executor

`git merge-tree origin/amd/main origin/amd/wo-gfx900-perm` collides on exactly ONE path:
`docs/amd/TRIPWIRE_GATE_CLASS_SPEC_agent5.md` (add/add — main carries my 38cd1bb1-era
snapshot imported at 98008a8b, updated at c4b093da; my lane carries the live sheet).
The one thing an add/add cannot be survived with is a default `--ours`: it discards a
whole lineage by position.

**Decided resolution (spec author): take the `wo-gfx900-perm` side wholesale.** Verified at
bytes: main's copy has only FOUR lines absent from mine, all four are superseded 128-era
text from my own lineage (§1's retired "BOTH/25%/128" preamble; the "128 sites,
mutation-testable" checker sentence, retired by ruling #549 and replaced by the canonical
140/152 predicate block). Mine adds 60 lines (spelling 3 + probe receipts, §2.6 green-today
leg, §5 seal residual, 40/42/44 predicate splits, §8). No foreign content lives in main's
copy — nothing is lost by taking theirs.
**Post-merge verification (one command, the merge must not be called done without it):**
`git show <merged-tip>:docs/amd/TRIPWIRE_GATE_CLASS_SPEC_agent5.md | diff - <(git show origin/amd/wo-gfx900-perm:docs/amd/TRIPWIRE_GATE_CLASS_SPEC_agent5.md)`
→ must be EMPTY. And everything else in the merge-tree probe was CLEAN (incl. the 36-row
array — agent2's clobber-worry closed by their own check, confirmed at my seat).

### §11 amendment (post-#758): the resolution is UNIVERSAL — every copy of the spec, every branch

#758 measured its 49+1 merge on wo-gfx900-perm (agent5's lane) but attributed the copy to
gemini — and gemini's branch DOES carry its own snapshot of the spec (imported; older still,
zero three-spelling hits, pre-§8). Measured at tips: agent5's tip = the ONLY copy with
spelling 3 + §2.6 + §5 seal + §8 + the 40/42/44 corrections. Binding resolution, generalized:
**for any add/add on `TRIPWIRE_GATE_CLASS_SPEC_agent5.md` against any branch, take
`origin/amd/wo-gfx900-perm`'s side; verify with the same one-command empty-diff against that
ref.** Main's 4 retired lines, gemini's stale snapshot — every non-agent5 copy is a strict
ancestor-subset; nothing foreign lives in any of them; the lineage-knowledge adjudication is
made once, here, for all legs.
