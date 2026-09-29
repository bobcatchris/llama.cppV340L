# CLOSEOUT — A1 lane: k5v4 §5 validation, k4v4 prologue, and the review chain

**Written 2026-09-05 ~09:00Z. Read this first, then `docs/k4v4_extraction_plan.md`.**
**Updated 2026-09-05 ~11:40Z (same session, continued): task #1 CLOSED (§2a), merge landed (§2b),
Step C DONE AND PROVEN (§4a). Resume point is Step D. §3 is superseded — read §2b + §4a instead.**
Non-numbered filename deliberately — doc numbers come from the coordinator (AGENTS.md).

---

## 1. The task

Two assignments, one finished, one mid-flight.

**A. k5v4 §5 real-model validation** (docs/117 §5) — **COMPLETE.** 4 of 5 executable checks PASS on
the real 18 GB artifact. KLD blocked for a structural reason, documented. Coordinator ratified
k5v4 as validated-for-serve; the serve refusal stays because 4/5 ≠ 5/5.

**B. k4v4 (KVarN K4/V4) decode prologue** — user directive "squeeze in kvarn4/kvarn4". **STEPS
A+B+B2+C+D ALL DONE AND PROVEN.** Task #1 (the `budget-seam-fix` marker repair) also closed (§2a).
k4v4 `<4,4>` is wired, oracle-proven 10/10, route-proven 15/15, and refused at serve until §5.

Chain: `28100ec7` merge → `e8326873` Step C → `346dfe49` handoff → `a5dc8ba1`+`b55bead3` cleanups →
`467fb837` **Step B2** → `61559b5e` slice7 pick → `7836a08a` **Step D** → `1afe078e` serve refusal.

**Resume point is NOT more k4v4 kernel work.** What is left for a shipped k4v4 tier:
1. **§5 end-to-end validation on the real 18 GB artifact** — needs a server, so a fresh GPU grant.
   k4v4 is deliberately refused at serve until this runs (same posture as k5v4).
2. **A2 must implement slice7\'s green path** — see §4c, it is currently a `return 0;` stub.
3. The (c) pairing session with A2 (§4 "Then"), still not started.

---

## 2. Current state — verify, don't trust

| branch | head | what |
|---|---|---|
| `wo/k4v4-prologue` | `1afe078e` | **continue here.** Steps A+B+B2+C+D, both baselines + k4v4 seal |
| `wo/117-int4` | `6ac04411`+ | §5 work, reviews, docs source |
| `wo/budget-seam-fix` | `c48e302f` | ✅ markers RESOLVED (was `5280e6df`), suite proven green — see §2a. MERGED to main |
| `wo/dflash2-scope` | `b81bc616` | A2's DFlash2 stack; the ORIGINAL base of k4v4-prologue (superseded — see §2b) |
| `wo/k4v4-cpu` | **DELETED** | was a trap — see §5 |
| `backup/k4v4-prologue-premerge` | `d666f00b` | pre-merge safety ref; keep until Step D lands |
| `main` | `36be5ad0` | has the seam (`b9fbf522` merged `c48e302f`) + docs/151 marker fix (`36be5ad0`) |

```bash
cd /home/intel/ninfer/worktrees/wo-117-int4
git rev-parse --abbrev-ref HEAD   # want: wo/k4v4-prologue
git status --porcelain            # want: empty
```

**Proven (hash, not argument):**
- `<5,4>` reproduces byte-for-byte after Step B **and again after Step C** — `oracle.txt`,
  `dispatch.txt`, `known_answer.txt` all MATCH `docs/baselines/k4v4/MANIFEST.txt`. The post-Step-C
  re-check is the one that matters: slice4 and slice6 now share the prologue header, so a Step C
  mistake in the shared file WOULD have moved `<5,4>`.
- `<4,2>` is data-equivalent across Step C — proven by direct pre/post build-and-compare, because
  **there is no sealed `<4,2>` byte baseline** (see §4a; the gate as written was unfalsifiable).

**Not proven:** k4v4 correctness (Step D not done).

---

## 2a. ✅ RESOLVED 2026-09-05 — `budget-seam-fix` markers fixed at **`c48e302f`** (was `5280e6df`)

**Status: DONE and proven.** Fix-forward commit `c48e302f` on `wo/budget-seam-fix`, reported to the
coordinator for the merge to main. Recording what actually happened, because two of the working
diagnoses in the plan below were wrong and the next reader will be told otherwise:

- **The pending cherry-pick was a duplicate, not a remaining step.** `5280e6df` *was* the replay of
  `b81bc616` — identical `tp_engine.cpp` delta (5+/26−), and all 24 lines of its `tp2_budget.h` delta
  were already present at the parent. A stale `CHERRY_PICK_HEAD` made git believe that same pick was
  still in progress. `--continue` would have produced a second commit with the same subject. Cleared
  with `--skip`. **No SHA was rewritten** — `5280e6df` is quoted in this file and in coordinator
  messages, so amending would have stranded those refs. Fix-forward, deliberately.
- **`tp2_budget.h` was never damaged.** Byte-identical to both `wo/dflash2-scope` and
  `wo/k4v4-prologue`; `DrafterBudgetBackend` enum + factory intact. A2's dropped-enum worry was a
  false alarm, and the "markers are only in 2 files" scope claim was checked, not assumed.
- **A resolved-but-unstaged file is not a fix.** A2 had correctly resolved `tp_engine.cpp` in the
  worktree, but `git status` read ` M` (unstaged) and I first misread that column as staged. HEAD and
  the index were both still the marker version. Distinguish them with `git show :path` vs the worktree
  file — never by the status column alone.

**A fresh build dir here needs TWO configure flags, not one:**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DBUILD_TESTING=ON
```

Default `/usr/local/cuda` is **12.9** and fails the version gate. `BUILD_TESTING` defaults **OFF**, so
`ninfer_tp2_budget_test` simply does not exist — the failure is `No rule to make target`, which reads
like a wrong target name but is a config gate. Cost a cycle; recorded so it does not cost another.

**Proof obtained (rule 11 + rule 14)** — after the coordinator correctly rejected the first run as
predating the final binary, I discarded it and did a forced clean rebuild (`rm` object + binary):

```
source  tests/test_tp2_budget.cpp              07:13:01.355457
object  .../test_tp2_budget.cpp.o              07:13:20.865896
binary  build/tests/ninfer_tp2_budget_test     07:13:20.913897
log     /tmp/suite_final_071337.txt            07:13:37.143263   <- newer than binary
```

binary > object > source, log newer than binary, so the run used *this* binary. Suite rc=0, 45 checks.
`ar t libninfer_engine.a | grep tp_engine` = 1 — the object is in the archive, not merely compiled.
Two mutation runs, both fired, both reverted (tree verified back to `c48e302f`):

- flipped a test expectation (768→769 MiB) → rc=1, names the MTP-768 check ⇒ checks execute and gate rc
- **neutered the source guard** `validate_cache_type_widths` → rc=1 with 3 named FAILs (k4v2+k5,
  k5v4+v2, cache-type-k 99) ⇒ the tier-consistency and range guards are genuinely covered, not present

### ⚠️ Rebase heads-up — neither branch is a superset of the other

`wo/k4v4-prologue` and `wo/budget-seam-fix` both carry the seam, marker-free and literal-free, but
they diverge in exactly **two content spots**, and "merged cleanly" will not tell you which way you
landed:

1. **The always-on `[preflight]` telemetry printf** (`0f7760ee`, main side). `budget-seam-fix` has the
   richer version — required / fixed / context / slack breakdown. `k4v4-prologue` still has the older
   one-line printf. A naive rebase **loses the telemetry fix**.
2. **Declaration hoisting and order** (`is_kvarn_storage_v`, `is_i4_kv`, `is_i8_kv` hoisted above the
   auto-capacity block, plus the "SEAM, single call, no copy" comment). `k4v4-prologue` has it;
   `budget-seam-fix` declares the same things earlier but differently. Functionally equivalent,
   textually conflicting.

**So after rebasing onto post-merge main, re-run the greps — do not trust git's success message.**
Same failure mode as the `wo/k4v4-cpu` trap in §5, one layer up.

```bash
grep -rn "^<<<<<<<\|^=======\|^>>>>>>>" src/ tests/                  # must be EMPTY
grep -n "18496ULL\|34ULL \* 1024ULL\|8000ULL\|768ULL" src/runtime/tp2/tp_engine.cpp  # EMPTY
grep -c "kv_bytes_for_tier" src/runtime/tp2/tp_engine.cpp            # >= 1 (CALLED, not just defined)
grep -c "slack" src/runtime/tp2/tp_engine.cpp                        # >= 1 -> telemetry SURVIVED
# then re-seal the <5,4>/<4,2> baseline per §3 step 3
```

**Minor, do not re-derive:** this branch's `test_tp2_budget.cpp` lacks three lines that
`k4v4-prologue`'s copy has — `#include "ninfer/types.h"`, a `return;`, and the `ok:` print in
`check()`. That is why the suite is silent on pass. It is not inert (the mutations fire), but
per-check visibility returns on rebase.

The meta-lesson, kept from the original plan because it is the pattern of the whole session: the
coordinator's pre-merge check grepped the *invariants* (they passed) but not for *conflict markers* —
shape verified, substrate not. Grep for the thing that means "this was never finished", not only for
the things that mean "this is wrong".

---

## 2b. Base of record is now `main`, not `wo/dflash2-scope`

`wo/k4v4-prologue` merged `main@b9fbf522` at `28100ec7`. The directed **rebase was refused on
measurement**: `git cherry main wo/k4v4-prologue` reported 48 of 50 commits replaying, only 2
recognised as upstream, and ~15 of the 48 re-apply seam changes main already has (patch-ids differ
because contexts differ, so git cannot see them as upstream). Replaying 15 duplicate seam patches
onto a tree that already carries the seam is exactly how `5280e6df` got committed with markers in
it. A merge resolved the same spots once. Coordinator approved this on the measured reasoning.

Two conflicts were predicted (§2a) and behaved; a third was not:

1. `tp_engine.cpp`, 5 blocks — declaration order only. Collapsed to exactly one declaration of each
   of `is_kvarn` / `kvarn_widths` / `is_i8_kv` / `is_i4_kv` / `tier_kv_bytes_per_token`, verified by
   use-site rather than by eye.
2. The always-on `[preflight]` telemetry printf was **not** a conflict — this branch never touched
   that region, so main's richer fixed/context/slack version merged cleanly. Now asserted
   (`grep -c slack` = 2), not hoped.
3. **`docs/151_dflash2_scope.md` carried 10 committed conflict markers on `main`**, introduced by
   `300c1616` and pushed in `b9fbf522`. Cause: task #1's gate was
   `grep -rn "^<<<<<<<" src/ tests/` — **code directories only**, so `docs/` was never in scope.
   Second trap: that naive pattern is wrong for markdown anyway, because `=======` matches
   legitimate section underlines. The markdown-safe pattern, no dir scope:
   ```bash
   git grep -nE "^(<{7}( |$)|={7}$|>{7}( |$))"
   ```
   This is now the standing marker gate. Coordinator landed the docs fix on main as `36be5ad0`.

---

## 3. Immediate start — the exact sequence ⚠️ SUPERSEDED, kept as the record of what was tried

> **This sequence is historical.** Task #1 is closed (§2a) and the rebase in step 2 was replaced by
> a merge (§2b). Step C is done (§4a). **The live resume point is Step D in §4 — read §2b and §4a
> instead of following the commands below.** Kept because the GPU-lease mechanics, the
> rebuild-harnesses-from-source rule, and the baseline re-verify loop in step 3 are all still
> correct and still needed for Step D. The `NVCC`/`C`/`L` variables below are the working ones —
> note they assume `build/` was configured with CUDA 13.1 (see §2a for the two required flags).


```bash
# 1. GPU grant FIRST. Written grant from `coordinator` via intercom. "Clear to claim" is not a grant.
#    Guard on foreign CUDA contexts, never on ports.
cd /home/intel/ninfer/worktrees/wo-2a-batched-serving
bash tools/smoke/diag/gpu_lease.sh acquire 0 agent1-k4v4 8100 5400
bash tools/smoke/diag/gpu_lease.sh acquire 1 agent1-k4v4 8101 5400
bash tools/smoke/diag/gpu_lease.sh show

# 2. Rebase onto budget-seam-fix ONLY AFTER §2a is resolved and proven to compile.
#    As of this writing it contains committed conflict markers; rebasing onto it unresolved would
#    inherit them. Check first:
git -C /home/intel/ninfer/repo show wo/budget-seam-fix:src/runtime/tp2/tp_engine.cpp \
  | grep -c "^<<<<<<<\|^>>>>>>>" | xargs echo "conflict markers (must be 0 before rebasing):"

# 3. RE-VERIFY THE BASELINE BEFORE EDITING ANYTHING. Non-negotiable.
cd /home/intel/ninfer/worktrees/wo-117-int4
NVCC=/usr/local/cuda-13.1/bin/nvcc
C="-std=c++20 -O2 -I include -I src -I tests -I third_party --expt-relaxed-constexpr \
   -diag-suppress 20050 -diag-suppress 177 -gencode arch=compute_120a,code=sm_120a"
L="build/src/libninfer_ops.a build/src/libninfer_core.a build/src/libninfer_nvfp4_tma.a \
   -lcudart -lcublasLt -ldl"
for t in k5v4_oracle k5v4_dispatch; do
  $NVCC $C tools/i4_oracle/$t.cu $L -lcuda -o /tmp/v_$t
done
$NVCC $C tools/i4_oracle/k5v4_known_answer.cu $L -o /tmp/v_ka
/tmp/v_k5v4_oracle    > /tmp/v_oracle.txt    2>&1
/tmp/v_k5v4_dispatch  > /tmp/v_dispatch.txt  2>&1
/tmp/v_ka             > /tmp/v_ka.txt        2>&1
for pair in "oracle:/tmp/v_oracle.txt" "dispatch:/tmp/v_dispatch.txt" "known_answer:/tmp/v_ka.txt"; do
  f=${pair%%:*}; p=${pair#*:}
  [ "$(sha256sum <$p|cut -d' ' -f1)" = "$(grep "  $f.txt" docs/baselines/k4v4/MANIFEST.txt|awk '{print $1}')" ] \
    && echo "MATCH  $f" || echo "DRIFT  $f  <-- STOP, recapture, do not proceed"
done

# 4. Release when done. Works from a different shell (fixed tooling):
bash tools/smoke/diag/gpu_lease.sh release 0 agent1-k4v4
bash tools/smoke/diag/gpu_lease.sh release 1 agent1-k4v4
```

**Rebuild the harnesses from source every time.** They are built by a hand-written `nvcc` line, not a
CMake target — nothing in the build system notices when they go stale, and a stale binary produces a
meaningless green (REPO.md rule 11 footnote).

---

## 4. Next steps

### Step C — redirect slice4, prove `<4,2>` bit-identical

**The trap that makes this harder than Step B looked:** slice4's symbol names *differ* from the
shared header's (`kKvarnKTileBytes` vs `kKvarnK4TileBytes`, `kvarn_k_tile_off` vs
`kvarn_k4_tile_off`). **Nothing collides**, so the edit compiles and passes while slice4 still
carries its own copies and the duplication survives silently. Step B was protected by redefinition
errors; Step C has no such guard. The only check is the grep in §6.

Measured reference counts (so the blast radius is known before editing):

| symbol | refs | action |
|---|---|---|
| `kKvarnStageSmemBytes` | **12** | used by `gqa_attention_kvarn.cu:575` + `tests/slice4_kvarn_bench.cu` — **must stay defined here**, but derived: `2 * (kKvarnK4TileBytes + kKvarnV2TileBytes)` |
| `kKvarnKTileBytes` | 5 | alias to `kKvarnK4TileBytes` |
| `kKvarnVTileBytes` | 3 | alias to `kKvarnV2TileBytes` |
| `kvarn_k_tile_off` | 3 | replace call sites with `kvarn_k4_tile_off`, then delete |
| `kKvarnKStageRow` | 2 | legacy, unused by tile banks — leave alone |

Then replace slice4's fused prefetch loop (`op < 512 + 128`) with `kvarn_prefetch_k4(...)` +
`kvarn_prefetch_v2(...)`, and its K/V dequant blocks with `kvarn_dequant_k4_tile(...)` /
`kvarn_dequant_v2_tile(...)`. **Keep the `!page_ok` zero-fill in the caller** — it writes both tiles;
duplicating it per side can leave one stale on a bad physical page.

**Gate:** `<4,2>` byte-identical to the sealed baseline. Also rebuild and keep green:
`ninfer_slice4_kvarn_test`, `ninfer_kvarn_batched_ops_test`, `ninfer_kvarn_gqa_test`.

### ✅ Step C — DONE AND PROVEN (`e8326873`). Read this before re-doing it.

slice4 redirected to the shared header; 584 → 549 lines. Fused prefetch loop →
`kvarn_prefetch_k4(...)` + `kvarn_prefetch_v2(...)`; K/V dequant blocks →
`kvarn_dequant_k4_tile(...)` / `kvarn_dequant_v2_tile(...)`; stage pointers → header constants.
`kKvarnStageSmemBytes` stays defined in slice4 (12 external refs) but as an **alias** to the
header's `kKvarnK4V2StageBytes`.

**The silent failure mode the plan predicted was real, and my first grep got it wrong in the
opposite direction.** slice4's names differ from the header's, so nothing collides — the edit
compiles and passes while slice4 keeps its own copies. But when I ran the rule-15 grep it returned
two hits that were **my own explanatory comments** mentioning `kvarn_k_tile_off` and
`op < 512 + 128`. The gate that means something excludes comments:

```bash
grep -vE "^\s*//|/\*" src/ops/kernel/gqa_decode_slice4_kvarn.cuh \
  | grep -cE "kKvarnKTileBytes|kvarn_k_tile_off|= *(4224|2048|12544)\b"   # -> 0
```

Old symbols DELETED not shadowed. I also deleted a trailing `// 2*(4224+2048) = 12544` comment —
a comment restating a derived value goes stale silently — then **rebuilt and re-ran to confirm that
edit was inert** instead of assuming a comment cannot matter. It was: hash unchanged.

**Barriers preserved exactly:** slice4 has *never* had a `__syncthreads()` between the K and V
stages, only one at the end of `prologue_kvarn_kv_tile`. Neither helper contains a sync, so the two
calls cannot introduce one. The `!page_ok` zero-fill stays in the caller, shared (header invariant
2). The `DbgSkip` guards stay in the caller so phase-skip attribution in `slice4_kvarn_bench.cu`
keys off the same bits.

**Incidental fix:** slice4's `#include` block was duplicated verbatim (six includes, twice) — the
same defect Step B fixed in slice6, never applied here.

### ⚠️ The Step C gate was unfalsifiable as written — and that is a baseline-file bug

The plan said "prove `<4,2>` byte-identical to the sealed baseline". But `MANIFEST.txt` seals
`<5,4>` as three sha256'd outputs and records **only `PASS rc=0`** for the k4v2 side. There was
nothing to be byte-identical *to*. A refactor that moved `max_err` from 0.0019 to 0.0021 while
staying inside tolerance would still print `PASS` and exit 0. Rule 14's failure mode was living in
a **baseline file**, not in a test — which is why it survived the review that produced the plan.

Proven instead by direct pre/post comparison:

1. build + run the three k4v2 tests at post-Step-C → `/tmp/c4post/`
2. `git checkout 28100ec7 -- src/ops/kernel/gqa_decode_slice4_kvarn.cuh`, rebuild, run → `/tmp/c4pre/`
3. compare sha256 of full stdout+stderr
4. restore the Step-C header **verified by its own sha256**, rebuild, re-run → confirms POST reproduces

Result: all three outputs byte-identical PRE vs POST, and all **10** `max_err=`/`tol=` values
identical. `<5,4>` re-checked independently: still MATCHES all three MANIFEST hashes.

The `<4,2>` outputs are now sealed as `docs/baselines/k4v4/k4v2_*.txt` + `k4v2_README.md`, so Step
D inherits a real gate. **Apply the same question to any other sealed baseline you rely on: does it
record bytes, or only rc=0?**
### Step D — the k4v4 tier itself. **HARD-STOP before this without a fresh session's budget.**

`k4v4 = K4 + V4`, zero new bit-manipulation code. Stage smem `kKvarnK4V4StageBytes` = **8320 B**,
smaller than k5v4's 9232, so slice6's single-buffered pipeline ordering carries over with no
`__launch_bounds__` change. Budget figure **11152 B/token / `kv_unit` 11849** comes free from
`kvarn_tier_widths` — **do not restate it anywhere.**

Then: dispatch route in `gqa_attention_cached`; open `is_registered_kvarn_widths` for `(4,4)` **only
when the prologue exists** (that gate is what makes an unwired tier fail loudly instead of reading
garbage — preserve it); clone k5v4's oracle + dispatch harnesses for k4v4.

**Step D's test is already largely paid for — do not re-derive it.** A2 landed
`tests/slice7_kvarn_k4v4_test.cu` at `fc8159a8` on `wo/shape-parity`: 11 FP64 oracle cases for
`<4,4>` (incl. T=1 Wc=4), 4-bit K **and** V known-answer anchors, public-op dispatch through
`gqa_attention_cached` with `route_diff(k4v4 vs k5v4)` and `route_diff(k4v4 vs k4v2)`, an
unregistered-width guard that throws, and a `--mutate-k` hook. It currently reports EXPECTED RED
and **exits 1**, so it cannot pass vacuously — my `<4,4>` dispatch is what flips it green. Two
caveats before relying on it: (a) the coordinator said they would verify `fc8159a8` independently
before recording it, so confirm it exists as described; (b) it lives on `wo/shape-parity`, so it has
to be brought across — and per §7 that means diffing section-by-section, not trusting a size that
looks plausible.

**Both baselines now exist, so Step D has two gates, not one:** `<5,4>` via
`docs/baselines/k4v4/MANIFEST.txt` and `<4,2>` via `docs/baselines/k4v4/k4v2_README.md`. Adding the
`<4,4>` instantiation must move **neither**. Re-run both after the dispatch route lands, before
writing any k4v4-specific test.

### Then: the (c) pairing session (A1 + A2)

Agreed split — **ownership is deliberately inverted, recorded in docs/151 §18.3**:
- **A1:** `types.h` enum + `speculative_options.h` parse/name/validate with per-target ceiling
- **A2:** `layouts_impl.h:577` arm + trailing throw, `apps/cli/options.cpp`, §19.1(a) enumeration test
- **Paired decision:** `program_impl` io slots → **explicit not-wired-yet throw**, no slot allocation

**Bar for the io-slot test (rule 14):** it must fail if the wiring is removed. Both directions —
DFlash2 requested → throws naming the slice that wires it; slots allocated but unpopulated →
`has_value()` inconsistency caught.

**Acceptance gate for the drafter (docs/151 §19.2):** divergence-**rate** vs the MTP/bf16 baseline,
**never byte-identity**, and **state n**. See §5.

---

## 5. Findings that must not be re-derived

**A sealed baseline that records only `rc=0` is not a gate.** `MANIFEST.txt` pinned `<5,4>` with
three sha256s but pinned `<4,2>` with `PASS rc=0`. Every instruction that said "prove `<4,2>`
byte-identical to the sealed baseline" was therefore unfalsifiable — and it read as rigorous, because
the file *was* a sealed baseline. Ask of any gate: does it record bytes, or only an exit code? An
`rc=0` gate passes under any change that stays inside tolerance, which is precisely the class of
change a refactor is most likely to introduce. Fixed for `<4,2>` in `docs/baselines/k4v4/k4v2_README.md`.

**Grep for conflict markers with a markdown-safe pattern, and with no directory scope.** Two separate
bugs in the check the whole session relied on. `grep -rn "^<<<<<<<\|^=======\|^>>>>>>>" src/ tests/`
(a) never looked at `docs/`, which is how 10 committed markers reached `main` inside
`docs/151_dflash2_scope.md` and got pushed; and (b) `^=======` matches legitimate markdown section
underlines, so the naive pattern is noisy in `docs/` and gets ignored rather than fixed. Use:
```bash
git grep -nE "^(<{7}( |$)|={7}$|>{7}( |$))"      # exactly 7 chars, all tracked files
```

**The §5 identity gate was unsound.** docs/117 asked for `mtp1==mtp0` byte-identity. Measured at
4 prompts × 160 tokens: **bf16 diverges 2/4** — a tier with no quantization to blame. Byte-identity
is not a property the MTP path guarantees on any tier (verify vs plain decode differ numerically the
way unified-vs-packed is documented to). Redefined as divergence-rate vs bf16; k5v4 (1/4) beats the
reference. **Struck through in docs/117 §5, not silently rewritten.** At n=1 this read as "k4v2 is
broken" — one extra experiment showed the *gate* was broken.

**KLD is blocked structurally, not just absent.** No `NVKLDMP1` producer exists, and the existing
hooks are unusable: KVarN decode is split-K with `kKvarnDecodeSplits = 54` while the dump is guarded
`kv_head == 0 && split == 0`, so it emits a **partial distribution over ~1/54 of the key window**
that sums to 1 and looks valid. The four kernels also disagree on indexing (flat vs 2-D row-major).
Recommendation: dump **post-combine** probabilities from the merge kernel + a permanent sum-to-1.0
selftest. Do not wire it as a hot-path change.

**Shipped k4v2 acceptance is 2.6 pt below bf16**, outside the ±0.5 pt bar. Surfaced, not fixed —
whether that's expected for the more aggressive tier or needs a tier-specific bar is a user call.

**Auto KV capacity had a real sizing bug** (probe over-proposed 409,344 tokens against ~2459 MiB
free). Root-caused to two causes, fixed by A2 via a shared factory. Workaround in the §5 driver:
explicit `--kv-capacity`.

**k3v3 is NOT free** — needs new code for both sides (12 B K rows, 96 B V rows, 3 ∤ 8).
**k3v3 is VRAM-identical to k4v2** (8976 B/token) because the budget formula depends on
`k_bits + v_bits` — a K/V balance change is free in capacity terms, costing only accuracy.

**The `wo/k4v4-cpu` trap.** It looked like the k4v4 home but lacked the seam refactor. Rebasing onto
it produced `Automatic merge went well` with **no conflicts** while silently reverting
`kv_bytes_for_tier`, `validate_cache_type_widths` and `verify_no_divergence` to dead code and
restoring the superseded inline `768ULL`. Caught only by grepping *after* the merge instead of
trusting git's success message. Branch is now deleted; `wo/budget-seam-fix` replaces it.

---

## 6. The checks to run, not skip

```bash
# After ANY merge, cherry-pick, or doc move — markdown-safe, NO directory scope (§5):
git grep -nE "^(<{7}( |$)|={7}$|>{7}( |$))"      # must be EMPTY; do not scope this to src/ tests/

# After ANY merge or refactor of the budget code (rule 15 — single source means the copy is GONE):
grep -n "18496ULL\|34ULL \* 1024ULL\|8000ULL\|768ULL\|kvarn_kv_bytes_per_token" \
     src/runtime/tp2/tp_engine.cpp          # must be EMPTY
grep -c "kv_bytes_for_tier" src/runtime/tp2/tp_engine.cpp   # want >=1 (CALLED, not just defined)

# After deleting a block (rule 16 — removals take too much, not just too little):
git show <commit> | grep "^-.*throw"        # every refusal this commit deleted

# After redirecting a kernel to a shared header (Step C's silent failure mode):
# comments MUST be excluded or your own explanatory comments are false positives.
grep -vE "^\s*//|/\*" <kernel>.cuh | grep -cE "<old-symbol>|<old-fn>|= *(<restated-literal>)\b"  # want 0

# After any claimed-green compile (rule 11):
touch src/runtime/tp2/tp_engine.cpp && cmake --build build --target ninfer_engine -j6
find build -name "tp_engine.cpp.o" -newer src/runtime/tp2/tp_engine.cpp   # must be non-empty

# Never trust a pipeline's exit code — capture the real one:
nvcc ... > /tmp/out.txt 2>&1; RC=$?; echo "rc=$RC errors=$(grep -c 'error:' /tmp/out.txt)"
```

**Standing rules (AGENTS.md):** never system-wide `pkill` for `ninfer-serve`; kill only own PIDs;
written GPU grant before launching a server; `df -h /` before any build >5 GB; never `cmake --build
build` without `--target` (88+ exes ≈ 17 GB); doc numbers come from the coordinator only.

---

## 7. Honest caveats for whoever picks this up

- **Step B's hash is the claim; the data-equivalence argument is commentary.** If Step C's hash moves,
  the extraction is wrong — not the baseline. Stop and report rather than continuing on a broken gate.
- **The baseline was verified on `wo/k4v4-prologue` after the base switch**, rebuilt from that tree.
  Re-verify again after any rebase; it is cheap and the base has already changed once.
- **This closeout nearly shipped broken.** Bringing docs from `wo/117-int4` into `wo/k4v4-prologue`
  silently truncated `k4v4_extraction_plan.md` to 5,402 of 16,124 bytes — the rule-16 over-deletion
  failure, reproduced by the person reporting it, minutes after naming it. `cat >>` on a branch that
  lacks the file **creates it with only the appended content**. If you merge docs across branches,
  diff section headings both ways; do not trust a size that "looks plausible".
- **Nine findings and six rules came out of this session; three of the findings were mine and one
  cost A2 a correct number they'd retracted on my authority.** Every rule is a generalisation of a
  specific mistake. Rules 11–16 live in REPO.md §4.

---

## 4c. Step D record + the two findings that changed its shape (2026-09-05, later same session)

**Step D is done** (`7836a08a` + `1afe078e`). k4v4 is **not** a third kernel: slice6 gained a
*trailing* `int KBits = 5`, so every existing `<5,4>` instantiation is textually unchanged and
resolves to the identical specialization. The launcher's `bool K5V4` became `int KSide`
(0=slice4/k4v2, 5=slice6 K5, 4=slice6 K4) — one value carries both "which kernel" and "which K
side", so a second flag cannot drift out of step with the first. `is_registered_kvarn_widths`
opened for `(4,4)` **in the same commit as the dispatch route, never before it**.

Evidence is sealed in `docs/baselines/k4v4/k4v4_README.md`. Headline: `<4,4>` oracle 10/10
(worst 0.0254412 vs tol 0.0610617), routing 15/15 with `rel(k4v4 vs k4v2)=1.001803` over identical
bytes, native-fixture `err(k4v4 vs bf16)=0.116836`. Both shipped tiers unmoved; the dispatch
harness diff is 7 lines added and **0 removed**.

### Finding 1 — Step B never moved the dequant (fixed as Step B2, `467fb837`)

Step B's commit message says "slice6 redirected to the shared per-side prologues". Only the
**prefetch** was. The K5 and V4 dequant blocks were still inline in slice6, and the header's
`kvarn_dequant_k5_tile` / `kvarn_dequant_v4_tile` had **zero callers repo-wide** — dead code
duplicating live code.

This was load-bearing for Step D, not cosmetic: k4v4's V side *is* `kvarn_dequant_v4_tile`.
Composing it before closing this would have put a never-executed copy on the acceptance path while
the proven copy sat in slice6. They looked semantically identical — that is an argument.

**Generalisable check for any "extracted / deduplicated / redirected" claim:** does the original
still contain a copy, and does the new one have a caller? A green build proves neither.

### Finding 2 — slice7's green path is a stub (do NOT define the macro)

`tests/slice7_kvarn_k4v4_test.cu` (A2's, cherry-picked at `61559b5e`) gates on
`HAS_SLICE7_K4V4` ← `NINFER_HAS_KVARN_K4V4`. Its green path is:

```cpp
#if HAS_SLICE7_K4V4
template <int TokenTile, int WarpsPerCta> int run_case(const Cfg& cfg) { return 0; }
int run(const Cfg& cfg) { return 0; }
#endif
```

Defining the macro would flip **11 oracle cases green without executing anything**. Its
public-op path IS implemented, which is why this is subtle. The coordinator verified "exit 1,
12× EXPECTED RED" — that proves non-vacuity in the RED direction only, which is the half that is
easy to get right. **Slice7 is left RED deliberately.** Whoever picks this up: do not "fix" it by
defining the macro; implement `run_case` against the now-wired dispatch, or delete the oracle
section and rely on the sealed harnesses.

### Two of my own failures, recorded because both are the pattern of this lane

- **The cherry-pick over-import.** `tests/CMakeLists.txt` had an EMPTY HEAD side, so take-INCOMING
  looked identical to task #1 and was resolved the same way. It was not: git's conflict region
  spanned shape-parity's surrounding context, so it imported slice6's test registration too — 6
  lines where the commit's own diff was 3 — and cmake's generate step failed on the missing file.
  An empty HEAD side does not make take-INCOMING safe; what is safe is the commit's **diff**.
  Check: `git diff <pre-pick> HEAD -- f` against `git show <pick> -- f`.
- **Mutating with uncommitted work.** I ran the serve-guard mutation (which correctly fired, rc=1)
  while the Step D serve edit was uncommitted, and `git checkout <file>` to revert the mutation
  destroyed the real work. The closeout's opening paragraph warns about exactly this. Commit
  before mutating. I also caught my own rule-16 over-deletion there — I had deleted
  `if (failures == 0) { std::cout << "ok\n"; }` — via `git diff | grep '^-'` before building.

### §5 result for k4v4 — and an open user decision (do not bury)

k4v4 §5 ran on the extraction branch: vram PASS (0.015 MiB), t/s PASS (+1.9%), **acceptance
FAILS the bar in the favourable direction** — 0.580 vs bf16 0.540 = +4.0 pt against a ±0.5 pt
bar. All three KVarN tiers breach it on the same side, ordered by K precision (k4v2 4K/2V +7.0 >
k4v4 4K/4V +4.0 > k5v4 5K/4V +5.0 is NOT monotone in total bits but the K-precision axis is:
ordered deviations are less likely noise). A quantized-KV target with HIGHER draft acceptance
than bf16 is physically odd, and n=4 cannot separate a real effect from a scope-dependent
acceptance counter or a prompt-set artefact.

An earlier draft of the §5 doc called this "comfortably inside the ±0.5 pt bar" in the same
sentence as "+4.0 pt". That was a buried violation, corrected at 0750b47b/812a075f. **The open
user decision is the same one the k5v4 run left for k4v2** (tier-specific bar / one-sided bar /
bigger n before the bar means anything). Full numbers: docs/k4v4_s5_validation_results.md.

### Left unfixed on purpose (needs a deliberate re-seal, not a drive-by)

`tools/i4_oracle/k5v4_oracle.cu` prints its verdict as `ninfer_slice4_kvarn_test: PASS` — a stale
clone label colliding with the real slice4 test's output. And `MANIFEST.txt` says "oracle: 11 PASS"
where there are 10 cases + 1 verdict line. Both are inside the sealed `<5,4>` baseline, so fixing
either moves the anchor this whole lane depends on. Re-seal deliberately or not at all.

---

## ADDENDUM — KVarN closeout completion (2026-09-05, A1, coordinator-assigned)

Closes the three remaining addendum items. Recorded here per coordinator routing; the Phase S
lane log lives in docs/154_multibatch_alldtypes.md §8.

### 1. Identity ladder n=24 — CITED, no new run

Measured + recorded at `src/serve/serve_options.cpp:399-406` (re-verified there this session,
matches the ruling): divergence ladder at n=24 = **bf16 42% / int8 38% / k5v4 38% / k4v4 46% /
k4v2 58%**. The lossless anchor (k4v2) diverges most, so byte-identity under speculation is
unsatisfiable for ANY tier — an architectural property of verify-width vs decode-width
reduction orders (docs/130 §5.2 #2, docs/132 §4.5), not a tier defect. A fresh sealed run for
the artifact chain remains available under the active GPU grant if ever wanted; not required.

### 2. §5 (docs/117) final state — ALL THREE TIERS PASS

- **Acceptance** (one-sided gate: no degradation below bf16): k4v2 **0.610** / k5v4 **0.590** /
  k4v4 **0.580** vs bf16 **0.540** — i.e. +7.0 / +5.0 / +4.0 pt, ALL PASS.
- **t/s**: within 4% of bf16 (k4v4 +1.9%) — PASS.
- **Identity n=24**: flat (k4v4 46% ≈ bf16 42%); the mtp1-vs-mtp0 difference is architectural
  (item 1), not a tier defect.
- **KLD**: WAIVED by user ruling 2026-09-05 (never measured for any tier; no producer exists —
  see the serve_options.cpp comment for the 1/54-split-K plausible-garbage reasoning).
- **Serve refusals**: LIFTED at e002548d — k4v2/k5v4/k4v4 serve-ENABLED; q4_0 STILL REFUSED on
  its real prefill defect (prefill fills an I4 cache with bf16 values, `code_dtype =
  I8?I8:BF16`) — that defect is G4 of docs/154, sequenced there as Phase D.

### 3. Name tables — CLOSED RATIFIED-DISTINCT (coordinator ruling 2026-09-05)

`request_log.cpp kv_cache_name` (log-schema strings: "int8-group64"/"int4-group64") vs
canonical `kv_cache_storage_name` (types.h:100 help/warning text: "int8"/"q4_0") serve
different contracts; kvarn tier names match across both. No collapse, no schema bump.
Cross-reference comment landed at the switch (a9fc54d3). The KvarnK4V4 name-table lag A2
caught was already fixed on main at 0ab1d504.
