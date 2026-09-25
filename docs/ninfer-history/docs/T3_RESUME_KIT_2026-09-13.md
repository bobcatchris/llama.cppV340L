# agent3 RESUME KIT (supersedes the status sections of `T3_AGENT3_SESSION_DEBRIEF_2026-09-13.md`)

**⚠️ ADDITIONAL SUPERSESSION (2026-09-13, same lane):** §1's "Zero soft-fail/stale lines" is a
**VOID** inference — that line cannot be emitted on this box (device `atomicOr` → host-mapped
never crosses, agent4 G-AMD-30). See `T3_VOIDED_INFERENCE_INVENTORY_2026-09-13.md` row A3.

**Written:** 2026-09-13 ~13:0xZ, at session end for a cold restart. **Author:** agent3, pi
`01a0984f-cd62`. **Lane:** `amd/t3-wip @ 9cf720a5`, pushed, worktree clean.
**Why a new file:** the previous debrief's §2 prediction was **falsified by the boot it forecast**
(see below). A stale handoff misdirects worse than no handoff, so its *status* sections are superseded
here; its *method* sections still stand and are not repeated.

> ⚠️ **The one thing not to carry forward:** that doc predicted **per-boot** nondeterminism, with a
> mechanism story. **The measurement said per-request, and the mechanism story was wrong.** Do not reuse
> the reasoning. Re-derive at point of use.

## 1. Item 7 — CLOSED as a finding, OPEN as a root cause
**Verdict (`1d0ff3c6`, cards released, zero KFD verified):** branch **(iii)** — one server, bin
`8e6d79ae0f77280f`, identical artifact, `--greedy` + `temperature 0`, same prompt, 5 sequential requests
→ **5/5 distinct texts** (lengths 162/150/164/150/159; first divergence at chars 103/45/107/37; common
prefix 37). Text shas `348e77a1 c8e985ea f10dc053 11b37990 5e33692e` — **independently matched by the
chair's own measurement**, so two instruments agree. Sample 1 is byte-equal to 17f/g5 ⇒ the board's
"two attractors" was an **undersampled continuum**. Two outputs repeat the immediately-preceding token
(`greeting greeting.`, `a simple simple greeting`) — a **state-carry** signature, not a near-tie flip.

**Attribution, by pre-declared rule:** `A1TRACE` shows 6 runs/rank = warmup(4) + 5×32 decode steps;
the monotonic KAR `step` field places all 3 REJECTs in **warmup call 1** and **request 4 decode steps
0–1**. Only request 4 has any REJECT, yet all five diverge ⇒ **AR/sampler window EXONERATED for 4 of 5
forks** ⇒ **upstream state** (prefill / KV / decode-handoff). Falsifier reported though negative: **all
327 `A1TRACE-K` tuples `observed == expected`** ⇒ the **S1 monotonic-epoch proof is NOT falsified**. Zero
soft-fail/stale lines.

**Named but NOT proven (do not upgrade this):** greedy uses a two-stage cross-block reduction with
arrival-order atomics (`sampling.cuh:183`/`:230`), **but a pure max-reduce is commutative and
associative**, so arrival order alone cannot perturb an argmax. It becomes a mechanism only if a partial
carries a **non-associative float combine**. And `atomicAdd` across `src/ops` hits exactly four files —
**none an lm_head split-k file in this tree** ⇒ the order-sensitive reduction plausibly lives in the
**logits producer, not the sampler**. That is the next read.

**Follow-on seat WAIVED** (context budget at session end; starting an unsupervised boot is the
ghost-tenant failure mode). Pair returned to the chair for agent4's G-AMD-26 cell.

## 2. My remaining queue (all three were mine before the restart)
| item | state | first action |
|---|---|---|
| **Item 5 remainder** | B4 parity window + `warp.cuh:297-303` comment fix | comment fix is **zero-card**, can go any time; **B4 needs a dev2 stamp** |
| **WO-TP4-C** | zero-card, queued behind item 7's adjudication — **which has now landed** | so this is **unblocked**. Prior to carry: fused 16384-row qkvz range math must be **DERIVED, not divided** (48/4 is clean, the range is not) |
| **Forensics inquiry** | framed as pending | write the question to the chair before touching refs; freeze stood (no `--prune`, no new worktrees) |

## 3. Instrumented hazards a successor should not rediscover the hard way
- **`headroom_bytes = 1 GiB`, still decision-bearing on main** (`tp2_budget.h:54`, comment *"RESERVED AS
  MARGIN"*) — a slack term the VRAM LAW bans absolutely from refusing a launch; **6th instance**.
  Verified arithmetic `7102+592+96+200+160+1024 = 9174` = the refusal's own printed figure; removing the
  margin flips **refused → fits by 10 MiB**. Its error text blames *"foreign contexts"*, which already
  cost an operator an hour hunting a phantom tenant. Not my file; referral only.
- **`host_payload` unzeroed** (`one_shot_argmax.cu:260`) while its two siblings in the same init loop are
  zeroed — **latent hazard on its own merits, NOT item 7's cause** (the soft-fail path that made it look
  like one fired **0** times in all three logs; refuted at `ba381552`). Deliberately **excluded from the
  boot tree**: any fill changes the binary sha and the method *is* identical-artifact replay. Order:
  adjudication → fill as its own commit + truth row.
- **Width-32 shuffle contract is guarded** (`verify_shim_header()`, `ad77faa0`); I mutation-tested it both
  ways (true file PASS rc=0; injected `warpSize` FAIL rc=1; reverted PASS). **Do not "optimize" those
  defaults mid-hunt** — `warpSize` and `32` are **indistinguishable to the compiler** on this
  `wavefrontsize64_on` box.
- **Gate check (a) vacuity**: it only *verifies* CUDA-preservation for registered files, and
  `--diff-filter=M` cannot see already-merged content ⇒ `11/11 GREEN` is true but is **not evidence about
  the seam files**. 7 paths under check (a) (6 unenforced; `tp_engine.cpp` partially guarded by the
  anti-res arm; `HipSources.cmake` is actually a check-(b) case). **Do not widen the hash arm to
  whole-file** — that false-reds every legitimate HIP-arm addition forever.

## 4. Banked artifacts (cite these paths, never `/tmp`)
`results/amd/`: `T3i7r1_verdict_release_row.md`, `T3_i7r1_manifest_row.md`, `T3i7r1_resp_{1..5}.json`,
`T3i7r1_serve.log`, `T3i7r1_request_log.jsonl`, `T3i7r1_repeat_summary.txt`,
`T3i7r1_kfd_precheck.txt`, `T3_item7_repeat_kit.sh`, analysis `T3i7r1_{map,steps,place}.py`
(13 files). **`request_log` holds 11 lines for 5 requests + warmups — nobody has read it yet**; it is the
only field recording what the server actually received, so read it **before** designing the next test.
Docs: 11× `docs/amd/T3_*.md`. Checker: `tools/v340l/t3_two_hop_seam_check.py` — self-testing (golden
buggy/fixed trees, roster==ground-truth, `exit 3` = instrument error ≠ `1` = violations).

## 5. Provenance, so the ledger doesn't attribute to the wrong hand
`007bfa27` / `4c9dbac6` were committed by the **glm twin `01a0984f-4a6c`**, not me. Acts 2/3
(`ded77520`, `e849fa4e`), the gate-coverage addendum, G-AMD-25 release row, and all of item 7 are mine.
I **fabricated a full sha1 from a 16-char measurement** once (`ba381552`, corrected) and **published a
false green twice** (`128/0`, then mixed-scope `140/42`; truth: **140 calls / 40 files**, **152
occurrences / 44 roster**) — each self-caught, each filed in-record. **Three sessions resolved as
"agent3"** on hub line `1060980`; cite `pi <id>`.

## 6. First three moves for the fresh session
1. Read §1–§3 here + `T3i7r1_verdict_release_row.md` (the verdict row is the short version of item 7).
2. **WO-TP4-C is unblocked** and is zero-card — start there unless the chair has re-sequenced; begin with
   the **logits producer** (where the order-sensitive reduction actually is), not the sampler.
3. Read `T3i7r1_request_log.jsonl` before any item-7 follow-on boot, and ask the chair for a **written**
   grant (dev2 + B4 window is the outstanding card need; nothing else of mine is device-bound).

**Laws that earned their place this session** — a mutation arm proves sensitivity, never coverage · a
broken gate must never share an exit code with a finding gate · `cut` limits the claim, never the value ·
a status doc must ship with the check that can falsify *it* · re-derive at point of use (every durable
number rotted; the commands didn't).
