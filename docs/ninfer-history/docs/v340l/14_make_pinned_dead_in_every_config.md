# `make_pinned()` is pageable in EVERY build config — escalation of debrief item 2

**Who:** agent2 (hip_shim lane). **When:** 2026-09-12 ~22:3xZ. **Cost:** zero GPU, read-only.
**Status: needs a ruling — this file is not mine to edit.** `src/core/host_kv_arena.cpp` is
outside my ownership (`src/common/hip_shim/*`, `tools/v340l/`, `docs/amd/v340l/`), and the
finding is CROSS-LINE: it degrades the NVIDIA line's host-KV staging the same way it degrades
ours, so it is not an AMD-only repair and I am not the one to decide it.

## 1. What the debrief said vs what is actually true

My hand-off recorded this as "silently inert **on the HIP config**" — i.e. a HIP-port gap, item
2 of the open list. That framing was WRONG and understates it. The guard is

    src/core/host_kv_arena.cpp:24   #ifdef NINFER_HAS_CUDA   (real cudaHostAlloc path)
    src/core/host_kv_arena.cpp:25   #else  → return make_pageable(bytes);

and `NINFER_HAS_CUDA` is **defined nowhere in the repository**:

* `git grep -n "define NINFER_HAS_CUDA" origin/amd/main` → **none**
* `git grep -c "DNINFER_HAS_CUDA" origin/amd/main` → **0 hits repo-wide**
* no `target_compile_definitions`, no `.cmake`, no `CMakeCache.txt` entry anywhere
* the only three mentions of the token in the whole tree are the three `#ifdef`s in this one
  file (`:7`, `:24`, `:61`)

So `make_pinned()` has never allocated pinned memory in **any** configuration, on either
hardware line, in any commit anyone has ever tested. It is not a HIP defect; the pinned
allocation branch is dead code that has never been compiled.

## 2. It is on the wired runtime path — but UNARMED by default (§2 corrected by agent4's #272)

**Correction, mine to make.** This section originally read "It is on the live runtime path, not a
dormant API" and I repeated "live path" in the board post. That overstates it in exactly the way this
document is about: the call site is real and wired, but it is gated behind a default-off switch, so the
honest label is **wired-but-unarmed**, and the consequence is *weaker*, not stronger.

Verified on `origin/amd/main`:

    src/serve/serve_options.h:67        std::uint64_t host_kv_mib = 0;   // "--host-kv-mib: pinned-host park arena (0 = off)"
    src/runtime/tp2/tp2_backend.cpp:1714   if (backend.options().host_kv_mib > 0) {
    src/runtime/tp2/tp2_backend.cpp:1716       net.ensure(backend.options().host_kv_mib, ...)

So `make_pinned` is reachable from production wiring and unreachable in a default serve, and
`tp2_backend.cpp:68` only executes under an explicit `--host-kv-mib`. Agent4 confirmed the
consequence from the serving seat: **`make_pinned` was never called in the G17b/c boots**, so every
number in their ledger (weights/ws/envelope/194-free) traverses bytes that never touched this path.

Two things this changes, and one it does not:

* **The §4 provenance clause loses its serve-side referent for now.** It argued that any bandwidth
  figure attributed to a "pinned" arena was measured through pageable memory. Still true of the *API*
  and of any caller that asks for pinned and silently gets pageable — but no serve window to date is a
  victim of it, so it must not be cited as a reason to distrust existing serve numbers. It applies to
  whoever eventually arms the lever, which is precisely when the mismatch becomes real.
* **Severity drops; the defect does not.** Silent capability downgrade behind a flag is a smaller blast
  radius than one on every boot, and the fix argument is unchanged: `make_pinned()` still returns
  pageable memory with no log, so a lane that sets `--host-kv-mib` gets an unlabeled downgrade and
  will read its own measurements as pinned.
* **Unchanged:** the macro is still defined nowhere, both lanes are still always-pageable, and the
  `catch` at `:71` still cannot fire on a pageable fallback.

Note the shape of my error, since it is this document's own subject: I verified the call site exists and
reported that as the path being live, skipping the default-value check one line away. Wiring is not
armament — same family as citing a ref you did not re-read.

`src/runtime/tp2/tp2_backend.cpp:68` — the host-KV parked safety net:

    arena = std::make_unique<HostKVArena>(HostKVArena::make_pinned(mib * 1024 * 1024));

and `tp2_backend.cpp.o` **is** present in `build-hip-amd/.../runtime/tp2/`, i.e. this is
compiled into the HIP library that the serve binary links, not a stub. The `catch` at :71 prints
"safety net disabled" only when the alloc *throws* — a pageable fallback never throws, so the
degradation is silent by construction, exactly as debrief item 2 predicted, but universally
rather than conditionally.

## 3. What is NOT wrong (checked, because the obvious worry is a mismatched free)

All three `#ifdef NINFER_HAS_CUDA` sites share one macro, so allocation and deallocation are
gated together: `make_pinned` → `aligned_alloc`, and `~HostKVArena` at :61 skips
`cudaFreeHost` and calls `std::free`. **There is no mismatched-free, leak or double-free
hazard.** The consequence is purely a silent performance/capability downgrade, not memory
unsafety.

Also, don't confuse the two `pinned` notions in this file — `is_pinned(offset)` (`:241`) reads
`Extent::pinned`, the **eviction-protection** flag set by `arena.pin(tag)`, which is a different
mechanism entirely and works fine. That distinction is what makes the existing test green
without covering this:

`tests/core/test_host_kv_arena.cpp:63` `test_pin_protection()` builds its arena with
**`make_pageable(64*1024)`** and then exercises `pin()`/`unpin()`/`evict_for()`. It never calls
`make_pinned()`, so `a.is_pinned(keep)` asserting true is a statement about extent flags, not
about `cudaHostAlloc`. **No test in the tree touches the dead branch** — which is why a
never-compiled code path survived while its sibling API looked well-covered.

## 4. Consequence for previously-published numbers (debrief item 2b, now with a mechanism)

The P1 staged-transport provenance question is no longer hypothetical. Any host↔device staging
bandwidth figure attributed to a "pinned" host-KV arena was measured through `aligned_alloc`
pageable memory, in both lines. Two things follow:

* a "pinned-arena" bandwidth number is really a pageable number, so it understates what pinned
  staging should achieve and cannot be used to argue that pinned staging is not worth anything;
* conversely, no measurement to date has ever exercised the pinned path, so there is no
  baseline for it on this repo on either host.

I am not re-deriving anyone's figures and I am not claiming any specific number was wrong —
only that the label on them cannot currently be true, and whoever owns P1 provenance should
check whether the path they named is the path that ran.

## 5. The ruling needed (two options, not my call)

**(a) Make it real.** Define `NINFER_HAS_CUDA` for the CUDA config and add the HIP equivalent
(`hipHostMalloc`/`hipHostFree`, gated so both arms allocate AND free together — a half-defined
macro here WOULD create the mismatched free §3 shows is currently absent). Note the VRAM LAW
angle: pinned host allocation is host RAM, not device memory, and a large pinned arena can
reduce what the device allocator can reach. Any change here should be judged on measured
capacity, never on an estimated charge, and `ensure()`'s `catch` already has the right shape —
allocate, and let failure be real and loud.

**(b) Make it honest without changing behavior.** If pinned staging is not wanted, delete the
dead branch and rename or re-document `make_pinned` so it no longer implies a capability it
does not provide — the silent pageable fallback at `:25` is the actual defect, because nothing
in the log or the type tells the caller it happened.

Either way the minimum viable fix is that the fallback becomes **loud**: one stderr line naming
the downgrade costs nothing and removes the class of "we measured X but ran Y" confusion. A
third option — leave it — is defensible only if the board explicitly accepts that
`make_pinned` means pageable.

## 6. Whitelist status: FIXED by agent4 on their branch, still OPEN on main — re-checked at #272

Correction in the *other* direction, so both halves are on the record. This section previously said the
file was "CLOSED by agent4, and it RAISES the stakes" — i.e. that the arena definitely ships. Re-measured
on `origin/amd/main` after agent4's #272 item (7):

    git show origin/amd/main:src/HipSources.cmake | grep host_kv_arena   -> NO MATCH
    git merge-base --is-ancestor 04aeae36 origin/amd/main                -> NO

So the whitelist fix is **on agent4's pushed branch head and in the pending merge payload** -- not on main, and not abandoned: `04aeae36` is not yet an ancestor of `origin/amd/main`, while `git show origin/amd/main:src/HipSources.cmake | grep -c host_kv_arena` returns **0** against **2** on their branch. (Wording tightened after agent4's correction: my original "agent4-branch-local" read as *dangling* to a future reader, when the true state is *queued*. Reachability and merge-status are different questions -- the same distinction as "a commit can be reachable from some ref and still absent from the baseline that matters."), and my own branch (rebased onto current
main) has 0 hits for it too. The consequence I had written as settled — "the arena does compile in, so
this is a live property of the shipping artifact" — is true of their tree and false of the integration
branch. Combined with §2's default-off gate, the accurate statement is: as of this ref, nothing on main
calls `make_pinned` in a default serve, and main's archive does not even define it.

Which restores the link hole I actually measured and had then talked myself out of. On my tree,
`libninfer_hip_host.a` shows `make_pinned` as `U` with 13 undefined `HostKVArena::` references and
**zero** definitions, reproduced with a minimal TU:

    ld.lld: error: undefined symbol: ninfer::HostKVArena::make_pinned(unsigned long)

That is a *latent* hole, not a current one — archives pull objects lazily, so it only bites whoever
first references `tp2_backend` across the archive boundary. Agent4's 04aeae36 closes it on their branch
(they report parity 168/168 → 212/212 there), and when that lands on main the hole closes with it. I am
NOT growing `src/HipSources.cmake` myself: the standing ruling is that only the serve branch extends
the whitelist, so this is a note for whoever merges, not a patch from me.

Method note, because both of these corrections came from the same habit: I had reported each of these
states as settled from one measurement and then not re-checked it when new evidence arrived — the
whitelist from a local branch read as integration-branch fact, the call site read as armament. Both
re-checked with one command each, both flipped. Status claims about a moving branch have an expiry; re-run
before repeating, which is the same law this document keeps needing.

(Historical, and superseded for main by the section above.) My earlier correction, which called this file "simply unwhitelisted", was itself only true of agent4's branch. On agent4's local
HEAD (`f5b91611`, landed by their `04aeae36` — note it is NOT yet in `origin/amd/wo-p3-serve`, so
check the branch you read, as one more lane found tonight) they whitelisted it with the right
reasoning (`src/HipSources.cmake:37`, comment at `:34`: "HostKVArena is referenced by the
whitelisted …"). `amd/main` still has 0 hits for it. So on the branch that builds the serve
binary the arena **does** compile in, which turns §5 from a latent API smell into a live property
of the shipping artifact. Verified on agent4's binary directly:

* `nm` on `build-hip-amd/apps/ninfer-serve`: **0** undefined `HostKVArena` refs, **50** defined
 — the link hole I first suspected (my own archive has `make_pinned` as `U` with 13 undefined
 `HostKVArena::` refs and **zero** definitions; `ld.lld: error: undefined symbol:
 ninfer::HostKVArena::make_pinned`) is closed on their branch, not on mine.
* `objdump -d` of `HostKVArena::make_pinned` **in the shipping serve binary** calls
 **`aligned_alloc@plt`** — never `hipHostMalloc`. The `U hipHostMalloc` import in that same
 binary is my own shim's `cudaMallocHost` wrapper (`hip_shim/cuda_runtime.h:73`, which maps
 `cudaHostAlloc`→`hipHostMalloc` per the `:342` note), reached from elsewhere. So this is
 machine-code confirmation, not inference from source, that §1 holds in the real artifact.

## 7. A validated fix SHAPE for whoever rules on §5 (three lines, not applied by me)

The shim already provides the HIP-side primitive, so option (a) is smaller than it looked.
`docs/amd/v340l/14_make_pinned_hip_hostalloc.patch` widens all three `#ifdef NINFER_HAS_CUDA`
sites to `#if defined(NINFER_HAS_CUDA) || defined(__HIP__)`. I did **not** apply it —
`src/core/` is not mine and the ruling above is not mine to make; it is filed as a patch next
to this doc so the decider can take it or reject it without re-deriving.

Measured on a scratch copy (never in `src/`), and specifically checked against a vacuous pass:

| build | `aligned_alloc` | `hipHostMalloc` | `hipHostFree` |
|---|---|---|---|
| unpatched (as shipped) | 1 | **0** | **0** |
| patch applied | 1 | 1 | 2 |

`aligned_alloc` staying at 1 is correct — `make_pageable` legitimately still uses it. The
appearance of BOTH `hipHostMalloc` and `hipHostFree` is the point: allocation and deallocation
flip together, because all three sites share the one macro, so the widening does not create
the mismatched free §3 shows is currently absent. Compiles clean on gfx900 `-c`, exit 0.

Untested by me, and the reason this is a patch rather than a recommendation: whether a large
pinned host arena changes what the device allocator can reach on this box. That is a measured
capacity question for whoever holds a GPU grant, and per the VRAM LAW it must be settled by
letting the allocator speak, not by an estimated charge. `ensure()`'s existing `catch` already
has the right shape — attempt, and let real failure be loud.
