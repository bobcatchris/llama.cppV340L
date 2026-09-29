# 21 — The shim verifier's poison check is textual: a `#define` that redefines the shuffle primitives passes it

Measured 2026-09-13, CPU-only (`python3` + `hipcc -fsyntax-only`; **no GPU, no boot, no grant**). Found while
answering a different question — whether my unmerged `cuda_bf16.h` fix would pass the gate that now covers the
shim. Mutation-tested the gate because that is the discipline the whole board adopted, and it failed.

## What changed first (a stale claim of mine, corrected)

All night I've carried "the arm skips `src/common/hip_shim/` entirely." **False at current main.**
`tools/ops/verify_registered_exception.py:385-386` now routes shim paths to a dedicated checker:

    if args.file.startswith("src/common/hip_shim/"):
        code, msg = verify_shim_header(args.file)

and the tool is properly tri-state — `EXIT_PASS 0 / EXIT_FAIL 1 / INSTRUMENT-ERROR 2`. So the hole closed and
I had not re-derived it. Good news first: **my branch's `cuda_bf16.h` PASSES the current gate (rc=0)**, so the
`grep-trap` fix (`0ddf0cdc`, still unmerged, `+11 −3`) is merge-ready by the gate's own lights.

## The gap: what "poison-free" actually tests

`verify_shim_header`'s poison check is one regex (offsets ~62-63 of the function):

    re.search(r"#[ \t]*define[ \t]+(__host__|__device__|__forceinline__)[ \t]*$", content, re.M)

and the width-32 check only inspects *existing* wrapper bodies:

    shfl_defaults = re.findall(r"__shfl_[a-z_]*sync\([^)]*\)\s*\{\s*return\s+__shfl_[a-z_]*sync\(...", content)
        -> requires len == 4, and rejects a default if "warpSize" in it or no ", 32 )"

So: three names, object-like only, and only for `__host__`/`__device__`/`__forceinline__`. Nothing scans for a
macro that **redefines the shuffle primitives themselves**, and the wrapper test is textual presence, not
post-preprocessor semantics.

## Mutation results (the gate's advertised claim vs measured behaviour)

Baseline: unmodified `src/common/hip_shim/cuda_bf16.h` @ `origin/amd/main` → `rc=0 PASS`. Each row appends one
line to a copy of that file, then runs the gate:

| mutation | appended | rc | verdict |
|---|---|---|---|
| `FUNC_MACRO` | `#define __shfl_down_sync(v,x,y,z) (poisoned)` | **0** | PASS — "poison-free, width-32 contract certified" |
| `OBJ_MACRO` | `#define __shfl_xor_sync __shfl_xor` | **0** | PASS — same claim |
| `WRAP_REMOVE` | comment line marking the contract as deleted | **0** | PASS (nothing removed; see caveat) |

Exit codes captured **directly** (`out=$(…); rc=$?`), not after a pipeline — an earlier run of this same test
reported `rc=0` from `tail` and I discarded it as evidence. Three of three mutations certified clean.

Why this matters beyond form: `cuda_bf16.h` is a **shim header that everything else includes**, so a
function-like `#define` of `__shfl_down_sync` at its tail changes what every downstream consumer means by that
call — while the gate that exists to certify the width-32 logical-warp contract prints "certified". It also
compiles: `hipcc -fsyntax-only` on the header alone cannot notice, because poisoned macro text is valid syntax.
The check is reading for the *shape* of three specific spellings, not for whether the file still means the same
program.

## Caveats on my own finding, since the claim is about absence

- **Bounded listing.** I tested three forms against one header. A fourth form (`#undef` + redefinition, a
  `#pragma`, a `#define __shfl_sync(...)` before the wrappers) may fail where these pass. The claim is
  "these three pass," not "any poisoning passes."
- **`WRAP_REMOVE` was a weak mutation** — appending a comment removes nothing, so its PASS is not evidence.
  The two macro rows are the load-bearing ones. A real wrapper-deletion test needs the 4 wrappers actually
  edited out, which I did not do.
- **The file is not mine to fix** in the gate's sense: `tools/ops/` is gemini's boundary and
  `verify_registered_exception.py` lives there. `src/common/hip_shim/` is mine, and this is a defect in the
  *check*, so it is reported, not patched.

## Proposed fix, one clause

Extend the poison regex to any function-like macro whose name matches the primitives the contract is about —
`#define[ \t]+__shfl[a-z_]*[ \t]*\(` (and `__ballot`/`__any`/`__all`, same family), object-like or not — and make
the width-32 test **semantic** rather than textual: compile a small consumer TU that `#include`s the shim and
*calls* the four wrappers, asserting the emitted `width=32` argument, so a macro that changes meaning cannot
pass by leaving the wrapper bodies untouched. Same principle the seam checker already implements — a check that
cannot fail has not been run.

## Reproduce

    git worktree add /tmp/vr origin/amd/main --detach && cd /tmp/vr
    git show HEAD:src/common/hip_shim/cuda_bf16.h > src/common/hip_shim/cuda_bf16.h
    printf '\n#define __shfl_down_sync(v,x,y,z) (poisoned)\n' >> src/common/hip_shim/cuda_bf16.h
    python3 tools/ops/verify_registered_exception.py --file src/common/hip_shim/cuda_bf16.h; echo "rc=$?"
    # observed: "PASS: ... poison-free, width-32 contract certified" with rc=0

## Closure status: fixed by the INCOMING gate (gemini leg), verified by the same two mutations

Re-measured 2026-09-13 against `origin/wo/v340l-phase-gate` — gemini's branch, the one that carries
`verify_registered_exception.py` improvements not yet on main. Their poison check is **two** regexes where main
has one:

    main  :62   #define[ \t]+(__host__|__device__|__forceinline__)[ \t]*$
    gemini:109  same, ^-anchored, \b instead of end-of-line
    gemini:112  #define[ \t]+(__shfl[a-z_]*|__ballot[a-z_]*|__any[a-z_]*|__all[a-z_]*)   <- the missing family

Run main's tree with gemini's verifier dropped in (= the state after leg 1 lands), one appended line per row:

    #define __shfl_down_sync(v,x,y,z) (poisoned)   -> rc=1  FAIL: "Warp/shuffle primitive macro shadowing detected"
    #define __shfl_xor_sync __shfl_xor             -> rc=1  FAIL: same

Same two mutations on current main: **rc=0 PASS "poison-free, width-32 contract certified"** (re-confirmed today).
So the gap is real on main **now** and closed by the merge that is already planned.

**This is also the measured basis for the chair's merge ordering, and it refutes my own argument.** I claimed the
gate was "already in place on main" so legs 1↔2 were swappable. False twice over: gemini's branch edits
`gate_pg1_whitelist.sh` (12 diff lines vs base) and their verifier blob differs from main's
(`ac592cbcb5` → `32354537ba`), so the merge *changes the gate on main*. Ordering therefore has a consequence, not
just an aesthetic: **merging the gate leg first is what removes a live false-PASS from the tree everything else
merges through.** If it lands second, every earlier merge was vetted by a check that passes a poisoned shim
header — which is exactly the failure mode this file documented, and exactly why "gate protects the tree first"
was ruled the way it was.

Corrected recommendation: **v340l/21 → "open on main, closed by gemini leg 1; do not re-file; verify with the two
appends above rather than trusting this note."** The proposed fix I sketched (§"Proposed fix") is what gemini
implemented, minus the semantic consumer-TU compile — the wrapper-body check is still textual, so a mutation that
edits a wrapper body's `width` argument rather than defining a macro is still not covered. That residual stands.

## Postscript: the residual I left open is CLOSED for the case I named — measured, and my reasoning was the flaw

I wrote that "a mutation editing a wrapper body's `width` argument rather than defining a macro is still not
covered," reasoning from the check being **textual**. That inference was wrong in this direction: a textual
assertion of a *constant* catches edits to that constant, which is precisely what a textual check is good at.

Measured on `origin/amd/main` after the merge, one mutation in `src/common/hip_shim/cuda_runtime.h:512`:

    return __shfl_sync(m, v, src, 32)   ->   return __shfl_sync(m, v, src, 16)

    MUTATED:                 rc=1  FAIL: "Violation of width-32 logical warp contract in shuffle default wrapper"
    CONTROL (restored file): rc=0  PASS: "poison-free, width-32 contract certified, post-preproc…"

So both hazard classes this file documented are now refused on main: macro shadowing (`:112` regex) **and**
default-wrapper width. The gate that printed a false PASS here earlier in the session is closed, and the row in
`v340l/18`/this file should be read as historical triage, not open work.

**What genuinely remains, and it is not a host item.** `verify_shim_consumer_witness` already *calls* the 3-arg
overloads inside its witness TU —

    out[0] = __shfl_sync(0xffffffff, v, 0, 32);   // explicit width
    out[0] = __shfl_sync(0xffffffff, v, 0);        // 3-arg default, also present

— so "exercise the 3-arg forms" is already in place; what a compile cannot establish is that the default
**lowers to 32** rather than to `warpSize`. C++ gives no `static_assert` handle on a defaulted argument's
value, so the only ways to observe it are (i) the textual invariant on the shim body, which exists and fires, or
(ii) a device-time comparison of the two call forms — a behavioral check, therefore a **card** item, not a
queue item. I'd resist filing it as if it were free.

**Method note worth keeping:** I got this wrong by reasoning about the check's *form* ("textual, therefore
blind") instead of running the mutation it supposedly misses. The check was textual *about a constant*, so it was
exactly strong enough. Absence of coverage is a claim you test by mutating, not by inspecting — the same rule that
made three other findings tonight wrong in the helpful direction.
