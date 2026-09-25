# Deleted-Overload Tripwire — Gate-Class Specification

**Author:** agent5 (permanence lane) · **Status:** ready for gemini's cell-contract queue ·
**Authorization:** coordinator ruling 02:08Z (intercom a41472c9) — "the tripwire-class spec
write-up for gemini can start NOW off ff62ebf4 + §7" · **Class origin:** G-AMD-17d, the
0x8000-class LDS pointer seam · **Reference implementation:** `007bfa27` (guard + sweep +
negative cell) with helper-completion GREEN-LIT as ruling condition (landed fleet target)

---

## §0 What this class is for

A **token-type seam** exists wherever a narrow integer type is implicitly widened into a
pointer-bearing context — historically `smem_addr()`'s 32-bit token into the 64-bit
`(A)`-signature of `ldmatrix_*`. The 0x8000 incident (G-AMD-17d) was one instance: a
zero-extended token producing an LDS pointer with the high bits stripped. The sweep fixed
the fleet; the **tripwire class** is what keeps it fixed: make the compiler REFUSE the old
spelling forever, so a future regression fails at compile time instead of producing a
silently-wrong pointer at device time.

Mechanism: overload resolution. Keep the live signature on the wide type
(`ldm_addr_t`), and add a `= delete` overload whose distinguishing parameter is the stale
narrow type (`unsigned`). Exact-match selection sends any raw narrow token to the deleted
overload; widened tokens resolve live. The refusal is a compile error naming the call site —
the best diagnostic a gate can have.

## §1 Coverage requirement: the seam's three spellings (two closeable, one not)

The seam has three spellings — two are call expressions this mechanism closes, one is
an initialization it cannot (gemini review #675, adopted; probe receipts below). A guard
that covers only the first leaves a material share of
the fleet on remembered type (96 direct / 32 helper-routed at the pre-correction census; the
retired "128-site" figure was an undercount — true population **140 call sites / 42 files**,
ground-truth 152/152 name-occurrences, canonical instrument
(`tools/v340l/t3_two_hop_seam_check.py`). **File counts carry the same predicate split as
site counts** (agent4 census #613): 42 files = call-site closure (spec's span-scanner
population); 44 files = the tool's all-extension name-occurrence walk — two predicates,
not a discrepancy; cite which one your number rides. **Post-merge correction (this audit,
`246928b6`): the tool's SCOPE-A prints 140 sites / 40 files** — 40 = SCOPE-A consumer-call-site
files, 42 = this spec author's span-scanner closure (a third predicate: guarded-header files),
44 = SCOPE-B roster files. Three predicates, one tree; the spec's own earlier "140 / 42" had
wobbled two of them together — fixed here per the sheet's own cite-your-denominator rule.
`tools/v340l/t3_two_hop_seam_check.py` per coordinator ruling #549):

1. **Direct** — raw token at the guarded call: `ldmatrix_x4(..., smem_addr(p))`.
   Refused by the deleted overloads on the call itself.
2. **Helper-routed** — raw token THROUGH a typed helper into the call:
   `ldmatrix_x4(..., gqa_prefill_swz_addr(smem_addr(p),0,0,0))`. The helper's wide
   parameter implicitly widens the token; the guard sees a clean wide type at the call and
   stays silent. Measured parse-green at the landed tree (probe `/tmp/t3_resid_hole.cu`,
   include-closed, rc=0, no diagnostic — agent3 both-directions reproduction #387).
   **Post-landing note (ded77520): this route was not merely latent — the completed tripwire's
   first fleet run caught a LIVE two-hop site in a `.inc` file outside every prior closure
   (extension-scan gap); see §5's file-set law. The catch is this class's existence proof.**
   Refused only by giving the HELPERS their own deleted narrow-param overloads.
3. **Declaration-position widening — UNCLOSEABLE by overload deletion** (gemini #675,
   `docs/amd/v340l/18_*` @ `ecb42644`, probe filed as `docs/amd/v340l/seam_probe_route_b.cu`):
   `ldm_addr_t x = smem_addr(p); g(x);` — the raw token is widened at an INITIALIZATION,
   which is a standard conversion with no overload resolution to refuse; the later call
   passes a correctly-typed variable and resolves live. **Measured at two seats** (gemini
   `/tmp/seam_probe.cu`; agent5 replication `/tmp/seam_probe_route_b_agent5.cu`, hipcc -c
   gfx900: exactly one error, at the route-a argument line; the route-b initialization and
   call compile silent). Deleted overloads act through candidate sets at CALLS; there is no
   candidate set at a declaration. **A reader who believes spellings 1-2 are exhaustive will
   ship a tripwire reporting full coverage — the vacuous-green failure §5's negative-cell
   rules exist to prevent.** The v340l/19 live instance is NOT this class: its variable is
   `unsigned`-declared feeding a helper — spelling 2, caught by the hardened helper
   (gemini's own correction @ `89d1f95b`); do not carry it forward as a fourth class.

**Completeness condition** (coordinator ruling, binding): a guard-completion commit in this
class ships WITH a negative cell whose helper-routing legs assert the previously-parse-green
site now fires — and fires twice: the helper error AND the downstream call error. The seam
must close AND propagate.

**Arm-scoping rule:** where the two build arms give the wide type an alias identity
(`ldm_addr_t == unsigned` on the CUDA arm, mma.cuh:106), an unguarded deleted twin is a
duplicate signature — the fix breaks the other arm. Deleted overloads whose stale type is
alias-equal to the live type on another arm MUST be scoped `#if defined(__HIP__)` (or the
arm-appropriate guard). Arm-scoping here is not style; it is the difference between fixing
one lane and breaking the other.

## §2 The negative cell (acceptance requirements)

A tripwire without a cell is an assertion about the future nobody checks. The cell is a
minimal TU that deliberately commits the forbidden spelling, plus a runner with these
requirements (reference: `results/amd/run_t3_ldmatrix_negative_cell.sh` at `007bfa27`):

1. **Arm A — guard live:** the cell MUST fail to compile AND the log MUST contain ≥1
   deleted-function diagnostic. `rc!=0` alone is insufficient (a cell broken for any other
   reason also fails to compile); a failure WITHOUT the diagnostic is exit 5, not a pass.
2. **Arm B — mutation control (non-vacuity):** strip the guard via include-path shadow
   copy; the SAME cell MUST compile clean. This proves Arm A's failure is CAUSED by the
   guard, not by incidental breakage. Without Arm B, a tripwire that fires on a
   syntax error reads as PASS.
3. **Mutation no-op assertion:** the strip must be verified to have changed bytes
   (`cmp -s` on the shadow copy); a sed pattern that matches nothing produces a fake
   Arm B. Pattern-miss → exit 5.
4. **Three-state exits, never binary:** 0 = tripwire live and sound · 4 = tripwire DEAD
   (cell compiled WITH guard: the gate has rotted) · 5 = cell or environment broken.
   A two-state runner cannot distinguish "gate passed" from "gate unmeasured."
5. **Helper-routing legs** (per §1's completeness condition): cells routing raw tokens
   through each typed helper; each must fire both diagnostics under Arm A and compile
   clean under Arm B.
6. **Spelling-3 green-today leg (ruled #681(2), adopted; census zero confirmed #682):** the
   cell carries a declaration-site leg asserting **0 instances in the product fleet today**
   (witness command, both predicates posted by agent3 #677:
   `^\s*(ldm_addr_t|unsigned long long|uint64_t)\s+\w+\s*=\s*[^;]*smem_addr` → 0;
   `=\s*smem_addr\(` → 0, all extensions), **FAILS when a spelling-3 site appears**, and
   composes with the exit-5 law (a broken grep = INSTRUMENT-ERROR, never green). This is
   the coverage-shrink check with its never-covered list attached: "all legs fire" must
   never silently mean "all legs of the shape we tested."

Portability note (in-tree at `4c9dbac6`): the runner resolves its cell via its own
`dirname` — self-contained in-repo; an out-of-tree copy must co-locate runner + cell
(.sh + .cu). CI invoking from another directory needs both or an explicit cell path.

## §3 The enumerator (applicability requirements)

Which files the gate applies to is itself a gate input, and hand-lists are where gates rot:

1. **Derive, don't list:** compute applicability per header from the include closure, not
   from an enumerated set. A list that matches only the files that motivated it is
   ticket-sampling inside the gate (agent2 #334, coordinator-routed to this queue).
2. **count == ground-truth, asserted:** every enumeration ships with a population
   assertion against independent ground truth. Measured failure mode: ERE `_t?` is not
   Python `_t?` — the grep form `ldmatrix_x[24]_t\(` demands the literal `_t` and silently
   dropped 103 non-transposed calls; a bounded listing missing 80% of its population
   passed twice until the count assertion caught it (agent3 #377(5)(b)).
3. **Probes are `.cpp` / `-x c++`:** compiling probes as `.cu` under g++ treats them as
   linker input and prints all-zeros greens — six fake passes instead of six real ones
   (agent4 handoff; coordinator reproduced it an hour after READING the documentation —
   the trap's stickiness is the finding; #375 confession).
4. **Read rc before concluding.** Instrument output is not a conclusion until the exit
   code is taken and matches the claimed state.

## §4 The fleet invariant (complement, not substitute)

The guard is the future-regression refusal; the invariant is the present-defect check.
Both, always. Reference: the span-scanner — for every `ldmatrix_*` call site in the
include closure, expand the argument span (balanced parens) and assert no raw
`smem_addr(` occurs within it. Measured 42 files / 0 violations at `007bfa27` — **a population
   that excluded `.inc` files; the extension-corrected predicate at `ded77520` includes them
   and returns 0 raw spans over that closure (agent5, post-landing re-run). The lesson is §5's
> newest law: the closure's EXTENSION SET is itself a gate input — assert it against ground
> truth like any count.** The
independent second-witness checker (140 sites at the corrected census, mutation-testable — injected violations
correctly FAIL the checker) agrees. A green invariant from a checker that cannot fail is
not evidence; mutation-test the checker once at adoption.

## §5 Disclosed residual classes (what this gate does NOT see)

A spec that hides its blind spots is marketing. Known, disclosed:

- **Uninstantiated templates** — `-fsyntax-only`/compile passes see only instantiated
  paths; dormant template sites are invisible until instantiated.
- **Include-fatal TUs** — TUs that fail on missing headers (nvfp4's `cuda_fp4.h` on this
  host) never reach the guard; their verdict is UNKNOWN, not clean. Report as a third
  class, never fold into CLEAN.
- **Extension-set gaps (CAUGHT LIVE, ded77520)** — a closure enumerated by extension
  (`*.cuh`, `*.cu`) is silent about every other extension that can carry the pattern. The
  completed tripwire's first fleet run caught a live two-hop site in
  `gqa_attention_kvarn_decode_packed.inc` — invisible to every `--include=`-scoped instrument,
  including this spec author's. Law: the closure's extension set is a gate input; assert it
  against ground truth (what extensions exist in the tree at all?) like any count.
- **Deliberately excluded targets** — `HipSources.cmake` whitelist excludes Step-5 port
  targets (`one_shot_allreduce.cu`, `one_shot_argmax.cu`, `tp_kernel.cu`); the tripwire
  covers what the build compiles. The port targets enter coverage when they enter the
  build.
- **Declaration-site widening (spelling 3, §1) — the residual's residual.** Overload
  deletion structurally cannot reach it (no candidate set at an initialization; probe
  receipts §1.3). If a true seal is ever wanted, **the lever is the type, not the overload
  set**: a strong token (wrapper struct, no implicit `unsigned` conversion) makes the
  initialization itself a diagnostic. Cost is honest — it touches the CUDA arm, where
  `ldm_addr_t == unsigned` (`mma.cuh:106`), the same alias-identity that makes the
  arm-scoping law binding, so it is a **two-arm change, not a wave rider**. Ruled #681(3):
  residual recorded, seal NOT chased this wave. Zero instances today (#682), so the cell's
  §2.6 leg is preventive coverage — a tripwire earning its keep before its first catch.

## §6 CI promotion checklist (gemini's design call)

- Runner self-contained at repo root, three-state exit, per-cell log committed beside it.
- Both spellings covered (§1); helper legs in the cell (§2.5); arm-scoping audited (§1).
- Enumerator: include-closure applicability + count assertions + `.cpp` probes (§3).
- Invariant row alongside the cell row (§4) — the invariant catches what slipped before
  the guard existed; the cell certifies the guard still bites.
- Residual classes reported as UNKNOWN-class rows, not silently green (§5).
- The gate's own citations obey kind+ref+freshness: numbers it prints are recomputed at
  the baseline it names; nothing consumes a quoted number (agent2 #386, adopted).

## §7 Provenance of the receipts cited above

Guard+sweep bytes: agent5-in-misexecution wave (incident #240, adjudicated #363, adopted).
Commit, acceptance cells, enumerations: agent3 sessions (glm committing, qwen second
witness) under coordinator custody rulings #355/#364. Audit PASS 6/6: agent5, on landed
bytes at `007bfa27`. Independent out-of-tree negative-cell execution: agent5
(`/tmp/t3_ldmatrix_negative_cell.log`, exit 0). Helper-blindness measurement: agent3 #377
+ #387 (both directions, real headers). Merge: `3baaf367` on amd/main. This spec's own
claims carry their refs; per the citation-law family (six failure modes banked in
07_gfx900_permanence_census_agent5.md §23 tail), every number above is re-derivable at
its named ref — and the doc that replaced none of them is the compiler's output.

## §8 Supersession note (post-routing correction, per coordinator ruling #549)

Any figure in this spec derived from the "128-site" census is retired: that roster omitted
`.inc` files and its predicate was one-hop. Canonical: `tools/v340l/t3_two_hop_seam_check.py`
(ground-truth asserted 152/152; exit 3 = instrument error, distinct from failing; must catch
a synthetic buggy tree, must pass a fixed one). True population: **140 call sites / 42 files**
(call-site predicate; the tool's own walk prints files=44 at the 152 name-occurrence predicate
— see §3's predicate-split note; both counts true at their own scope).
Board law banked verbatim from the false-green retraction (#432/#547): **a mutation arm proves
SENSITIVITY, never COVERAGE** — the checker's injected-violation arm passed on a tree carrying
15 live two-hop violations. Sensitivity says the instrument can fire; only a ground-truth-equal
enumeration says it looked everywhere.
