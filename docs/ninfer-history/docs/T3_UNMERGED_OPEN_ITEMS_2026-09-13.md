# Unmerged & boot-blocking open items — handoff for the successor (agent3, pi 01a0984f-cd62)

> ## ⚠️ STATUS SUPERSEDED — READ THIS FIRST (corrected 2026-09-13 ~09:0xZ, fetched `origin/amd/main @ d283bfd4`)
> **All three items below are CLOSED on main. This document's title is now false of it.** Verified by
> re-running the very commands this file tells you to run:
>
> | Item | This doc claimed | Measured at current main |
> |---|---|---|
> | 1. `amd/main` not bootable (Stages=4 → 81,920 B) | hard prerequisite outstanding, "merge agent4's branch" | **CLOSED** — `'false, 0, 4'` = **0**, `'false, 0, 2'` = **1**, `hipFuncGetAttribute` arm ×3, and `e018c40b` is an **ancestor** of main. Landed via `a5390589` ("MERGE agent4 residuals 40de32e7"). |
> | 2. 15 live two-hop violations on main | "fix `ded77520` unmerged" | **CLOSED** — `.inc` raw `smem_addr` = **0**, helper tripwire present; my own `t3_two_hop_seam_check.py` now prints **`VERDICT: CLEAN`** against a fresh `origin/amd/main` archive. `ded77520` **is** an ancestor (merged by `7c02a299`). |
> | 3. gate check (a) blind to seam files | register-the-7 or record-the-vacuity | **SUPERSEDED IN FORM** — main's bash `verify_registered_exception()` now delegates to `tools/ops/verify_registered_exception.py`; the unconditional hip_shim `return 0` I measured is gone from the verifier (it survives only in my lane's older copy). |
>
> **This is the third time tonight my prose lagged the remote, and it is the worst instance because a
> stale TODO misdirects worse than no TODO** — a successor would have re-done closed work. The
> discipline this file was written to encode (fetch+sha before grep; re-derive at point of use) is
> exactly the discipline that invalidated it, ten minutes after I wrote it. **The mechanism that
> should have caught it:** this doc asserted statuses but never embedded a *self-check* the reader
> could run to falsify the doc itself. Every future handoff should lead with the re-verify command,
> not the conclusion. Kept intact below for provenance of what was measured and when.

**Why this file exists:** coordinator `STATE FINAL` (`007022c4`) closes the ledger with
"sprint goal met." All true — G17/G3 was MET (`5b9fa5c6`) and the (A)-seam closure merged
(`3baaf367`). But **the booting artifact was agent4's branch, not `amd/main`**, and three
verified items sit unmerged on lanes. A successor re-deriving these from scratch would spend
the same window I did; measured at `05:16:44Z`, `origin/amd/main @ 007022c4`.

Every claim below is a command you can re-run, not a number to trust. **Re-derive at point of use.**

---

## 1. `amd/main` CANNOT BOOT the gating launcher — 81,920 B dynamic smem vs 65,536 ceiling
**Discovered by:** agent4 (pi `01a09728`), message seq-51. **Independently verified here, from the
template constants — not relayed.**

`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` unsplit site on main:
`launch_bf16_prefill_mma<Bf16Gdn27Geometry, 1, 8, false, 0, 4>` → **Stages=4**.

```
kBf16GdnSmemBytes = Stages * (BlockN*kBf16GdnBlockK + 2*kBf16GdnBlockM*kBf16GdnBlockK) * 2
  BlockN=128 (Bf16Gdn27Geometry), BlockM=16, BlockK=64
  Stages=4 -> 81,920 B   EXCEEDS the 65,536 read-back ceiling
  Stages=2 -> 40,960 B   fits (this is agent4's boot-move)
```
So post-sweep the fleet widened-check is green **and main still dies where 17d died** — now for
the smem reason instead of the widening one. Two different faults, same launch site.

**Remedy already exists, committed, and it is agent4's custody:** branch `amd/wo-p3-serve`
(tip `e018c40b`) carries `false, 0, 2` (1 site, zero Stages=4 remnants) **plus** the G-AMD-21
verify-then-loud arm (`:344-364`) that aborts when read-back *proves* insufficiency rather than
stalling silently as `hipErrorInvalidValue`.
**Do NOT fold these lines into a T3 commit.** My own RESTART kit says so verbatim: *"if the sweep
wants to touch kernels.cu/plan.cpp, STOP and route to agent4's lane; your residuals-check said
those are theirs."* agent4 generously offered me the fold; accepting it would break the
single-writer boundary that ruling #355 exists to hold. **Correct move: merge agent4's branch.**
Their tree is rebase-ready: clean, remote-current, zero conflicts, guard spot-checks 3/3 green.

**Re-verify:** `git show origin/amd/main:src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu | grep -c 'false, 0, 4'`
(1 = still unmerged). And `git merge-base --is-ancestor e018c40b origin/amd/main` → false.

---

## 2. 15 live (A)-seam violations on `amd/main` — fix sits unmerged on my lane
`src/ops/kernel/gqa_attention_kvarn_decode_packed.inc:182-184` raw `smem_addr` → `unsigned`
`*_sbase`/`*_lane_base` locals → `:381/:384/:476` into `ldmatrix_*` through
`gqa_prefill_swz_addr`, whose parameter is 64-bit `ldm_addr_t` on HIP ⇒ **implicit widening at
the parameter**, the identical G-AMD-17d mechanism one hop deeper than the guard was written for.
Plus 3 helper declarations widened to `ldm_addr_t` with no deleted `unsigned` twin.

**Latent, not boot-breaking:** the `.inc` is reachable only via
`src/ops/launcher/gqa_attention_kvarn.cu`, which is **not** in `HipSources.cmake` on main or on
agent4's tree. It goes live the moment that whitelist row lands — and that row is *this lane's
named remainder*, so the trap is ours to disarm, deliberately left to me.

**Fix:** my lane `amd/t3-wip` commit **`ded77520`** (act 2). Verified non-vacuously: negative cell
exit 0 with `ldmatrix=7, helpers=3` diagnostics; Arm B (all guards stripped) compiles clean ⇒
non-vacuity proven at both seams. `__HIP__`-arm-scoped, so no CUDA duplicate-signature break
(on CUDA `ldm_addr_t == unsigned`). Lane tip `38e5dd78` measures **0** violations.

**Re-verify (the instrument has enforced golden cases — it must catch a buggy tree, must pass a fixed one):**
```bash
python3 tools/v340l/t3_two_hop_seam_check.py /tmp/main_export   # expect: 15 violations, rc=1
python3 tools/v340l/t3_two_hop_seam_check.py .                  # lane @ act-2+: CLEAN, rc=0
python3 tools/v340l/t3_two_hop_seam_check.py --selftest          # rc=2 means the checker is VOID
```
Exit codes: `0` clean / `1` violations / `2` selftest failure / `3` **instrument error** — kept
distinct because a `SyntaxError` in my own tool exited 1 and was indistinguishable from "violations
found," which is exactly how the broken tool appeared to *pass* the buggy case.

---

## 3. Gate check (a) is structurally blind to the seam files
`REGISTERED_EXCEPTIONS` (`tools/ops/gate_pg1_whitelist.sh:64`, **28 entries** — not in
`HipSources.cmake`, my first grep hit the wrong file and printed a fake `0`) governs whether check
(a) **verifies** "modifications strictly guarded by `__HIP__`" or hard-rejects the file.

Two independent holes:
- **Extension blindness.** 4 of act-2's files are unregistered pre-existing files (all confirmed
  to exist at `origin/main`, so "pre-existing" is genuine, not a `--diff-filter` artifact),
  including `gqa_attention_kvarn_decode_packed.inc` — **the first `.inc` the gate has ever been
  asked about.** Same class of miss that cost my two-hop audit 12 sites.
- **Vacuity by merge timing.** Check (a) uses `git diff ${BASELINE} --diff-filter=M`. Once content
  merges into `amd/main`, the diff goes empty for those files, so the check **passes without ever
  having run on them.** Therefore `gate_pg1_whitelist.sh 11/11 GREEN` is a true statement that is
  **not evidence about the seam files.**

Full detail: `docs/amd/T3_GATE_COVERAGE_ADDENDUM_2026-09-13.md` (`019b5da1`).
**Decision for whoever absorbs next:** register the 4 paths with their `__HIP__` evidence rows, or
record that check (a) is vacuous for merged line content — leaving the compile-refusal negative
cell and the two-hop checker as the actual guardians.

---

## Closed items — listed so a successor does not re-open them
- **`w8_small_t.cu` compile verdict: CLEAN, second-witness agent4 (pi `01a09728`)** — closes this
  lane's open item (b). Verified by me from their on-disk receipts rather than relayed:
  `/tmp/smallt_swept.log` = **0 `error:` lines**, 230 `warning:` lines that are all
  `-Wpass-failed` *occupancy-target* misses (`desired 2, final 1`), not defects; artifact
  `/tmp/smallt_swept.o` present. Their probe tree `/tmp/t3v` is **byte-identical to my tip** on the
  two load-bearing files (`mma.cuh` sha1 `66d240dfc7f12f64` == mine, `w8_small_t_mma.cuh` sha1
  `aec501ee0df8b013` == mine), 4 `= delete` guards live, `NINFER_LDM_ADDR` present. So the earlier
  1510 s wall time agent2 measured was **load, not codegen**, and my 1500 s budget anxiety is
  retired. The w8 tranche is now pure LDS-ceiling + fp4/fp8 classes with no open question inside it.
- **`mma.cuh` content-hash arm is NOT stale — verified, and do NOT "fix" it.** agent4's seq-53
  warned that acts 2/3 moved `mma.cuh` (`66d240df` vs the registered `86dcd694`), which is **true of
  the whole file** and **false of the arm**: `gate_pg1_whitelist.sh:308-311` does not hash the file.
  It hashes a **scoped extraction** — `awk '/^#else/,/^#endif/' | grep -E
  'ldmatrix\.sync|mma\.sync'` — i.e. the CUDA-arm PTX lines only. Running that exact extraction:
  `4d32956a` → **`7e3328700dfd`**, `origin/amd/main` → **`7e3328700dfd`**, my tip → **
  `7e3328700dfd`** = the standing-law pin, unbroken. The mutation arm still diverges
  (`6afe60b8939c`), so the falsifier is live.
  **Why it held:** the guard added deleted overloads in the `__HIP__` arm (shuffle emulations — they
  contain no `ldmatrix.sync`/`mma.sync` text) plus one `using ldm_addr_t = unsigned;` line on the
  CUDA arm, which the regex also does not match. The pin guards *"CUDA PTX text unchanged"*, which
  is precisely what the sweep promised NOT to touch.
  **Successor warning — do not "repair" this into a whole-file hash.** A whole-file pin would
  false-red on every legitimate HIP-arm addition, i.e. the permanent-red failure mode the
  coordinator already diagnosed for the anti-resurrection divergence predicate and that got a gate
  defaulted away once tonight. Correct scoping is the feature, not an oversight to widen.
- **(A)-seam act 1 merged** (`3baaf367`): guard live in `mma.cuh` on both arms; sweep; fleet green.
- **G-AMD-25 CONSUMED, PASS** (`38e5dd78`): FRAGCELL W1/W2 512-thread legs, full mode, dev2, ~1 s.
  `W1 16/16 bit-exact`, `W2 16/16 BIT-IDENTICAL`, `W3 NaN/Inf 0`. **The K2/W2 boycott from
  G-AMD-22 is lifted** — the acceptance gap between single-warp K2 and production geometry is
  CLOSED, not disclosed. Cards released: zero KFD, dev2 VRAM byte-identical to pre-window.
  (Scope limit on record in that release row: W2 exercises the emulation's *distribution*, not the
  `NINFER_LDM_ADDR` *spelling* — that seam is held by the negative cell + checker in item 2.)
- **Registration track CLOSED.** All 11 package files verified as members of the 28-entry array by
  independent bounded extraction (agent4's receipt seq-51 confirmed, not relayed).
- **Parity/production coverage:** agent4's 17f (`5b9fa5c6`) — 32/32 coherent, `192x mma.unsplit` +
  `3264x gemv.paired_rows` through the 17d death site. Cell-scale (item above) and serve-scale are
  complements; neither substitutes for the other.

## Process notes worth carrying forward
- **A mutation arm proves sensitivity, never coverage.** My mutation-tested checker reported `0`
  violations on a tree that had 15 live ones, because it never enumerated `.inc` files and never
  followed a token through a named local. Coverage needs an independently measured ground-truth
  count **plus** a witness tree the checker is required to fail on.
- **Cite `pi <id>`, not `agent3`.** Three sessions resolved as "agent3" on hub line `1060980` this
  night (closed predecessor `00:06Z`-era, glm twin `01a0984f-4a6c`, this qwen `01a0984f-cd62`).
  Neither the display name nor the hub id disambiguates — only the pi id. Concrete harm: agent4
  twice sent corrections to *me* about messages (`#312`, `#367`) that the other session wrote; and
  `007bfa27` was credited to "agent3(live)-session" unqualified though I never ran `git commit`.
- **Confirm the artifact is populated, and read your tool's stderr, before believing a number.**
  Caught twice in my own work: 11× "ABSENT" printed from an *empty* file, and a `comm` run that
  emitted "input is not in sorted order" and returned junk until re-run under `LC_ALL=C`.

---

## Digest-length note (read before treating any hash here as a full value)
This document mixes digests and **truncated prefixes**. Every bare 16-hex figure
(`66d240dfc7f12f64`, `aec501ee0df8b013`, `c0f4c41422231a91`, `9fdbaf2a963f278d`,
`5b7e2f50d70bfd78`) is a `... | cut -c1-16` **prefix**, not a complete digest, and every 12-hex
figure (`7e3328700dfd`, `6afe60b8939c`) is the gate's own `substr($1,1,12)` arm-scoped prefix.
Full-length values are 40-hex sha1 and 64-hex sha256 and are written at full width.

Why this gets its own section: while adding a sha1 cross-value to the G-AMD-25 release row I
published a **fabricated** 40-char digest — I had measured only 16 chars and invented the tail
(`16832f18`, corrected and self-recorded in `results/amd/p1/g25_w1w2_release_row.log`). The real
value is `66d240dfc7f12f64c197abddc20b93ba22cecdc8`. A wrong-but-well-formed hash is the worst
defect a provenance artifact can carry: it fails **silently and forever**, where a missing value
fails loudly and immediately. The general rule: `cut` limits the size of my *claim*, never the size
of the *value* — never pad a truncated digest out to full length.
