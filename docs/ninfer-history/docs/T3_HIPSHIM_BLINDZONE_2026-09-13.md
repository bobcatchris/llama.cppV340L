# hip_shim blind zone: the width-32 shuffle contract has no guardian

> ## ✅ CLOSED — guardian now exists and is mutation-tested (2026-09-13 ~09:0xZ)
> This finding is **correct as filed and now fixed upstream**. main carries
> `tools/ops/verify_registered_exception.py :: verify_shim_header()`, which asserts exactly the four
> 3-arg shuffle wrappers exist and **fails if any contains `warpSize` or omits `, 32 )`** — the
> detection-only lint this document proposed, implemented (via `ad77faa0`, gemini's taxonomy clauses
> (i)-(iv)). **I mutation-tested it rather than reading it and nodding**: true file → `PASS`, rc=0;
> one wrapper injected to `warpSize` → `FAIL: Violation of width-32 logical warp contract`, rc=1;
> reverted → `PASS`, rc=0. It also carries `EXIT_INSTRUMENT_ERROR`, i.e. it adopts the
> broken-gate-own-exit-code rule from item 2 below.
> **Correction to my own mechanism claim, in agent4's favor and against my own text:** I wrote that
> the bash verifier `return 0`s unconditionally on hip_shim at `:109-111`. That was **true at the ref
> I measured** (my lane's older copy still contains it verbatim) but **stale at main**, where the
> function is a one-line delegation to the python tool. My *conclusion* (no guardian) was right at
> the time and is now wrong; agent4's *explanation* (that I'd confused `is_registered_exception()`'s
> membership `return 0` with the verifier's) is also wrong — I read the right function at the wrong
> ref. Both errors are the same class; only the ref matters.

**Measured at:** `origin/amd/t3-wip @ 2addef12` (lane tip), 2026-09-13 ~05:2xZ.
**Trigger:** agent4's seq-52 caution that any CI promotion of the negative cell must land
*outside* the gate's `hip_shim` skip, "or the cell inherits an invisible blind zone on shim files."
Checking that caution against the code turned up something sharper than a placement caveat.

## The finding

`src/common/hip_shim/cuda_runtime.h:510-513` — the four 3-arg CUDA shuffle wrappers hard-code the
logical-warp width:

```cpp
template <class T> __device__ __forceinline__ T __shfl_xor_sync (unsigned m, T v, int lm)      { return __shfl_xor_sync (m, v, lm, 32); }
template <class T> __device__ __forceinline__ T __shfl_down_sync(unsigned m, T v, unsigned d)   { return __shfl_down_sync(m, v, d,  32); }
template <class T> __device__ __forceinline__ T __shfl_up_sync  (unsigned m, T v, unsigned d)   { return __shfl_up_sync  (m, v, d,  32); }
template <class T> __device__ __forceinline__ T __shfl_sync     (unsigned m, T v, int src)      { return __shfl_sync     (m, v, src, 32); }
```

Its own comment records that this constant is **load-bearing and has already broken once**:

> On 64-lane wavefronts the default MUST stay 32 (logical-warp semantics) — measured failure mode
> when it was `warpSize`(64): broadcast-from-lane-0 leaked row 0's scale into the sibling logical
> warp (l2norm batch-2 RED). *(G-AMD-12)*

And this is exactly the machinery tonight's device result depends on: `ldmatrix_*` distributes
per-lane row addresses via `__shfl_sync(m, addr, 8*j + row)` (mma.cuh), i.e. **the W1/W2
512-thread legs I just certified as 16/16 BIT-IDENTICAL rest on that literal `32`.**

## Why it is unguarded — two skips, both verified in source

`tools/ops/gate_pg1_whitelist.sh`:
- check (a) loop `continue`s on `^src/common/hip_shim/` (~:211), and
- `verify_registered_exception()` `return 0`s on the same pattern at **:109-111** — unconditionally,
  even though those shim paths *are* listed in `REGISTERED_EXCEPTIONS` (`:82-85`).

So flipping `32` → `warpSize` is a one-token edit to a pre-existing file that **no gate check
examines**. Worse, it is *parse-green and type-correct*: `warpSize` is a valid `int`, so nothing at
compile time distinguishes it. It fails only numerically, and only on wavefront-64 parts —
returning exactly the silent cross-warp contamination G-AMD-12 measured.

By this project's own boarded law, this is the defect class the sweep was created to end:
**"a contract change needs a compile-refusing mechanism, not a remembered list."** Here there is
neither a refusal nor even a review trigger — only a comment, inside a blind zone.

## Honest scope of what I have and have not shown
- **Not shown:** that anyone will flip it. This is exposure, not an active bug. Verified the seam's
  own guardians are outside the zone: `NINFER_LDM_ADDR` / `ldm_addr_t` occur in **no** `hip_shim/`
  header (`grep` → none); they live in `src/ops/common/mma.cuh` and `src/ops/kernel/*.cuh`, which
  the skip does not cover. So the *(A)-seam negative cell is **unaffected** by this blind zone.
- **Not shown:** that a compile-time tripwire for the width is possible — and this is now measured
  rather than assumed. Both spellings were compiled with the same driver the negative cell uses
  (`hipcc -x hip --offload-arch=gfx900`): the *bad* default `warpSize` → **rc=0**, the correct
  literal `32` → **also rc=0, zero errors**. **A compiler cannot tell them apart**, so there is no
  deleted-overload-style refusal available for this seam, unlike the token-type seam. The asymmetry
  is the reason it needs a different control, and I am flagging rather than shipping a fix I have
  not made work.
- **Measured, and it strengthens the case:** gfx900 is built with **`wavefrontsize64_on`**
  (read off the compiler's own cc1 args), so `warpSize == 64` on this hardware. The bad default is
  not hypothetical — it evaluates to exactly the value G-AMD-12 recorded as causing the l2norm
  batch-2 RED cross-warp leak.
- **Harness confession, because a false negative here would have been filed as evidence:** my first
  attempt compiled with bare `clang++` and died on `use of undeclared identifier '__shfl_xor_sync'`
  (the shim's emulations sit under `#if defined(__HIPCC__)`). That rc≠0 looked like "the compiler
  DOES catch it" and was purely my wrong driver. Re-run under `hipcc` gave the result above.

## Cheapest controls that would actually fail (gemini's lane to weigh; agent4's caveat applies)
1. **Detection-only lint** (CPU, zero-device): assert the four 3-arg wrappers pass a literal `32`
   and that no shuffle default resolves to `warpSize`. Fails loudly on the one-token edit that no
   current check can see — and it must be a LINT, because the measurement above shows no compiler
   diagnostic is available. Detection-only is honest here: it guards a constant, not a type.
2. **Rule, not code:** any diff touching `hip_shim` shuffle defaults requires a stamped
   shuffle-semantics window before merge. Today the only thing standing between that edit and a
   production wave64 leak is a prose comment.
3. **Placement caveat, as agent4 asked me to record:** if the negative cell is promoted into CI,
   it must sit *outside* both `hip_shim` skips — otherwise a shim-file contract drift is invisible
   precisely where the tripwire's first arm cannot reach. The cell's current seam needs no such
   relocation (see scope above); the *width* contract does.

## Note on the exchange this came from
`#367`/`#368` ("INTENT TO COMMIT" / "THE ATOMIC COMMIT HAS LANDED") and `#371` (the window ask) are
**not this session's messages** — they posted at `01:36-01:38Z` from hub line `1060980`, which
three sessions have resolved as "agent3" tonight. This session began `01:09:21Z` and its first two
posts (`#359` 01:19, `#361` 01:23) were the collision flags, not commit announcements. The
"messages racing pushes" pattern agent4 observed is real, but it belongs to the twin lane. Cite
`pi <id>`: it is the only field that disambiguates.
