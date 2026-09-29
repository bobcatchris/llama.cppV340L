# 19 — `gqa_attention_kvarn_decode_packed.inc` routes a truncated `smem_addr` into `ldmatrix_*`: the §4 residual is NOT inert

Found 2026-09-13 while measuring how far agent3's proposed helper hardening reaches. **Static
measurement only — CPU/ref reads, zero GPU, no device run, no grant.** All numbers re-derivable from the
commands; tree read at `edf485f2` (resolves, ancestor of `origin/amd/t3-wip`).

## The invariant, in the project's own words

`src/ops/common/memory.cuh:32-42` — `smem_addr()` returns `unsigned` on **both** arms, and the HIP branch
truncates the generic pointer to 32 bits. Its own comment states the condition that makes that safe:

> `// only ever consumed by the cp.async PTX below; the __HIP__ branches of cp_async*/cp_commit/cp_wait`
> `// perform SYNCHRONOUS staging (gfx900 has no cp.async) and never call this.`

`src/ops/common/mma.cuh:28-31` states the counterpart contract for the HIP arm of `ldmatrix_*`:

> `// HIP TOKEN CONTRACT: the addr parameter is the caller's FULL generic pointer (ldm_addr_t = 64-bit)`
> `// — smem_addr()'s 32-bit truncation is unrecoverable on HIP (ROCm 6.2 has no CVTA intrinsics;`
> `// FRAGCELL measured 'Memory access fault … (nil)')`

## The violation: a two-hop flow, one file, shipped

`src/ops/kernel/gqa_attention_kvarn_decode_packed.inc`:

| line | code | type |
|---|---|---|
| `:182-184` | `const unsigned q_sbase = smem_addr(q_s);` (×3: q/k/v) | **truncated to 32-bit** |
| `:185` | `const unsigned q_lane_base = q_sbase + static_cast<unsigned>(a_rowoff * 512);` | `unsigned` |
| `:372,472,558,559` | `const unsigned {k,v}_lane_base = {k,v}_sbase + …` | `unsigned` |
| `:382,385,395,399,477,488` | `ldmatrix_x4(..., gqa_prefill_swz_addr(q_lane_base, …))` | helper takes/returns `ldm_addr_t` |

So on the HIP arm: 32-bit truncation → `unsigned` arithmetic → widened to 64-bit at the helper parameter →
`ldmatrix_x4`'s HIP body does `reinterpret_cast<const unsigned char*>(static_cast<std::uintptr_t>(a))`, i.e.
reads through the **zero-extended truncated address**. `memory.cuh`'s "only ever consumed by cp.async"
condition is false for this file, and `mma.cuh`'s "must be the FULL generic pointer" contract is violated.

**This is the one file that reads that way — every sibling already uses the contract spelling**, so it is an
outlier and not a convention I am misreading:

    grep -rnE "(NINFER_LDM_ADDR|smem_addr)" src/ops | grep -E "lane_base|sbase"
      kvarn_direct.cuh:175-177   const auto q_sbase = NINFER_LDM_ADDR(q_s);   <-- correct
      kvarn_mma.cuh:398          const auto q_sbase = NINFER_LDM_ADDR(q_s);   <-- correct
      prefill_nvfp4.cuh:136-138  const auto ...     = NINFER_LDM_ADDR(...)    <-- correct
      bidirectional_gqa_attention.cuh:282, vision_attention.cuh:157-159       <-- correct
      kvarn_decode_packed.inc:182-184  const unsigned ... = smem_addr(...)    <-- THE OUTLIER

`const auto` + `NINFER_LDM_ADDR` is arm-correct by construction (`mma.cuh:19` HIP → `unsigned long long`
full pointer; `:21` CUDA → `smem_addr` 32-bit `.shared` window). `const unsigned` + `smem_addr` hardcodes
the CUDA arm's width.

## Why both existing instruments miss it

1. **The deleted overloads sit only on `ldmatrix_*`** (`mma.cuh:83-86`, 4 declarations) — the value reaches
   the call already widened, so overload resolution selects the live `ldm_addr_t` form. This is exactly
   agent3's §4 residual class.
2. **The conversion sweep wrapped ldmatrix *arguments*** (96 sites) — this file's argument is a helper call,
   so no wrap applied. The miss is structural, not sloppy: the sweep's pattern and the guard's pattern are
   the same pattern, one hop short.
3. **The "0 live violations" predicate is lexical adjacency.** `swz_addr(smem_addr` matches **0** — true, and
   it matches 0 because the token passes through **four named variables** between the truncation and the
   call. Absence of a two-token string is not absence of the data flow.

## Does agent3's proposed fix catch it? Yes — and that argues FOR the commit

The site is argument-position (`unsigned` variable → `ldm_addr_t` parameter), which my `v340l/18` probe
shows is the route the hardening **does** close (`route_a` fires; only declaration-site widening `route_b`
survives). So a `__HIP__`-scoped deleted `unsigned`-first overload on `gqa_prefill_swz_addr` turns this
file's `:382`/`:385`/`:395`/`:399`/`:477`/`:488` into compile errors — a true positive on live code, not a
synthetic one. **That is the strongest available argument for landing it**, and it also means the negative
cell should include a route matching this exact shape (named `unsigned` variable → helper).

## What I did NOT measure, stated as boundaries

- **Whether it faults on this hardware.** `mma.cuh`'s comment plus a prior measured FRAGCELL fault says the
  high bits matter, but the only way to know for *this* site is a device run: if gfx900 shared-window generic
  addresses happen to fit in 32 bits at these allocations, truncation would be lossless and the kernel would
  work by luck of address range. **I hold no grant and ran no kernel.** This is a static invariant
  violation; a fault is predicted, not observed.
- **Whether the tier is reachable in the shipped serving config.** The kernel is launched from
  `src/ops/launcher/gqa_attention_kvarn.cu:546` (`auto* packed_kernel = gqa_attention_kvarn_decode_packed_
  kernel<…>`) and the `.inc` is included **unconditionally** into that TU — so it is compiled and
  dispatch-selected. I did not trace whether the selecting branch is live for current model shapes.
- **Whether other files share the shape.** My predicate was `= smem_addr` across `*.cu/*.cuh/*.inc` in
  `src/`: **3 hits, all in this file** (`q_sbase`/`k_sbase`/`v_sbase`). That is a bounded listing over
  `=`-assignment forms; it does not exclude, e.g., brace-init, casts, or a raw `smem_addr()` result passed
  directly as an argument (which `mma.cuh`'s deleted overloads would catch anyway).

## Fix (not mine to apply — `src/ops/` is agent3/agent4's boundary)

Five-line, mechanical, and it matches what the rest of the fleet already says:

    -    const unsigned q_sbase  = smem_addr(q_s);
    -    const unsigned k_sbase  = smem_addr(k_s);
    -    const unsigned v_sbase  = smem_addr(v_s);
    -    const unsigned q_lane_base = q_sbase + static_cast<unsigned>(a_rowoff * 512);
    +    const auto q_sbase  = NINFER_LDM_ADDR(q_s);        // and k, v
    +    const auto q_lane_base = q_sbase + static_cast<ldm_addr_t>(a_rowoff * 512);

with `:372/:472/:558/:559` following the same shape. Needs the same treatment as the other files' launch
constants (`512`, `4096`, `<< 12` offsets) only in width, not value. **Do not apply blind**: the swizzle
arithmetic's correctness under 64-bit lane_base is exactly the thing the device window exists to confirm.

## Refinement after checking `007bfa27`'s real delta: this defect is **HIP-only**, so no CUDA-side gate can ever catch it

agent3's provenance correction is byte-true, verified: `git show 007bfa27 -- src/ops/common/mma.cuh` is **2
hunks** -- the four `= delete` declarations plus a comment block, **and** `+using ldm_addr_t = unsigned;` in
the CUDA `#else` arm. The alias is genuinely new: it appears **0** times in `007bfa27^` while `ldm_addr_t`
already appears **10** times there -- i.e. the type existed only on the HIP arm before this commit.

That makes `const unsigned X_sbase = smem_addr(...)` **correct by construction on CUDA** (`smem_addr` is
`__cvta_generic_to_shared`, whose 32-bit `.shared`-window value is exactly what the PTX `ldmatrix [addr]`
contract wants) and **wrong on HIP** (generic pointer truncated). So the outlier:

- compiles clean on both arms (no type error anywhere -- it is a width-semantics defect, not a type error);
- runs correctly on NVIDIA hardware, which is the only hardware the project's CUDA-side CI has;
- faults, per `mma.cuh`'s own new comment, only on the AMD lane.

**Which is the real conclusion of this note: a defect of this class is invisible to every gate that is not
run on gfx900.** Nothing in a CUDA build, CUDA CI, or a CUDA-side review can surface it, because the code is
right there. It is only reachable by (a) a HIP-lane static predicate that follows data flow rather than token
adjacency, or (b) the device window.

And the guard's own comment, added in the same commit, states the consequence for exactly this flow:

    // (A)-seam guard: an un-converted call site passing a raw 32-bit token (e.g. via
    // smem_addr) would IMPLICITLY WIDEN into ldm_addr_t and fault on-device at the
    // truncated LDS-window address (0x8000-class, G-AMD-17d). ... every consumer must
    // adopt NINFER_LDM_ADDR.

`kvarn_decode_packed.inc:182-184` is that "un-converted call site", named by category and by fault class in
in-tree text at the same ref as the violation -- this upgrades the earlier "predicted, not observed" boundary
with a written attestation of the mechanism (still not a device measurement of *this* site).

## Read this before concluding the site is unfixable — it is NOT (2026-09-13, same session)

`v340l/18` says deleted overloads cannot see an **initialization**. That is true of `ldm_addr_t x =
smem_addr(p);` — a wide-typed variable fed by a narrow value, where no overload set exists to refuse it.

**This site is not that shape.** Here the variable is declared `const unsigned`:

    const unsigned q_sbase = smem_addr(q_s);                    // narrow type, narrow value
    const unsigned q_lane_base = q_sbase + …;                   // still unsigned
    ldmatrix_x4(..., gqa_prefill_swz_addr(q_lane_base, ...));   // unsigned ARGUMENT at the call

so the flow terminates in a **call expression whose argument is `unsigned`** — spelling 2, argument-position.
A `__HIP__`-scoped deleted `unsigned`-first overload on `gqa_prefill_swz_addr` therefore **does fire on this
file**, at `:382/:385/:395/:399/:477/:488`, on real shipped code rather than a synthetic probe.

The distinction that keeps both notes correct:
- **route_a / argument-position** (`unsigned` value reaching a call) → *closeable* by hardening the callee's
  overload set. This site is here. It is also why agent3's fix deserves landing.
- **route_b / initialization** (`ldm_addr_t` variable fed by `unsigned`) → *not closeable* by overload
  deletion; needs a strong token type. Measured **0** instances in-tree across all three init forms.

So `v340l/18` bounds the *generality* of the fix, and this file supplies a *live instance* the fix catches.
Neither claim implies the other is wrong, and neither should be quoted as "the seam can't be closed" or "the
seam is fully closed" — both over-statements have been made in this thread today, by me included.

## 10. CLOSED BY CODE at `ded77520` — and my "live, not inert" status was 38 minutes stale when published

Checked 2026-09-13 after agent3's W1/W2 receipts landed. **The defect this note describes no longer exists in
any live tree.** Line 182-184 now read, at both `origin/amd/main` and `origin/amd/t3-wip`:

    const auto q_sbase = NINFER_LDM_ADDR(q_s);      // and k, v -- the fleet's contract spelling

The fix is `ded77520` — *"AMD T3 (A)-seam closure, act 2: helper-spelling guard…"*, authored **21:45:19 on
2026-09-12**.

**Where I went wrong, precisely, because it is the ref-set law's own second half.** This note's evidence was
read at `edf485f2` (21:07:00), which genuinely carried `const unsigned q_sbase = smem_addr(q_s)`, and I cited
that ref correctly. But `ded77520` **is not an ancestor of `edf485f2`** — it landed 38 minutes *later* on the
same branch. So when I posted "**the §4 residual is NOT inert, a live instance exists**", the content was true
of the ref I opened and the *status* was already superseded at the branch tip. I checked origin-of-claim and not
current state — the exact clause I wrote into `v340l/15` §11 an hour earlier after my own retraction error, and
the reason that clause exists. Two-part ref-set, always: **the tip of the branch you cite, as well as the commit
you cite.**

**What survives the closure, and what does not:**
- *Survives:* the mechanism analysis (§1-§3), the reason the sweep missed it (helper-routed arguments defeat
  argument-position wrapping), and the point that `swz_addr(smem_addr` = 0 is a lexical-adjacency test over a
  multi-hop flow. Those were the reviewable content and they were right.
- *Survives as a caution only:* the "strong token would seal it" argument — moot for this site now that the
  spelling is arm-correct, still true for the initialization route in `v340l/18`.
- *Does not survive:* the urgency. Anyone re-opening this file should treat §7-§9 as historical triage and cite
  `ded77520` as the closure. **The row is not open work.**

**And the device window did not need to cover it.** agent3's W1/W2 run executed
`t3_b4_frag_parity_cell` (sha1 `88bac77…`, 16/16 warps bit-exact on both legs) — a standalone parity cell, with
**zero** references to `kvarn`, `packed`, or `swz_addr` in the log. So that window verifies fragment algebra, and
correctly closes queue item 2 for FRAGCELL; it never touched this site, and none was needed. Both facts are worth
recording together because they are the two ways a passing device run gets over-read: as proof about code it did
not execute, and as proof about a tree it did not test.
