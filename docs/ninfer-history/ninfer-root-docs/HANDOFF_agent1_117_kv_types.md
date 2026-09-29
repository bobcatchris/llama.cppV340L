# HANDOFF — agent1, docs/117 §9 KV-cache-types lane (int4 + k5v4)

Written 2026-09-05 ~01:10Z by session `01a06ec2` (agent1). Resumption state, not a work order.
Read `AGENTS.md` standing rules first, then this.

## Where things stand

**Branch:** `wo/117-int4`, worktree `/home/intel/ninfer/worktrees/wo-117-int4`. This is a
SEPARATE tree from A2's `wo-117-kv-types` — the coordinator ruled this after A2 and I were
found sharing one working tree (see "lane hygiene" below).

Commits, oldest first:

| commit | what |
|---|---|
| `b37ba289` | int4 slice5_i4 kernel + i4 device helpers |
| `37fefaf5` | `DType::I4` marker + int4 KV pool planning |
| `b9c78dec` | int4 decode dispatch + oracle validation 9/9 |
| `e0fb88ce` | docs/150 int4 decode contract (tests lane) |
| `3bb53563` | renumber contract 117a → 150 |
| `13df0b65` | k5v4 slice6 prologue (clone of slice4) |
| `768884a4` | k5v4 oracle + known-answer anchor; k4v2 vacuity finding |

**int4 decode: GREEN.** 9/9 FP64-oracle cases, worst 0.0081 vs tol 0.0747. Mutation-tested.
**k5v4 decode: kernel + oracle GREEN** (10/10, worst 0.031 vs tol 0.062), independent
5-bit known-answer anchor passes. **k5v4 dispatch: NOT WIRED.**

> **SUPERSEDED — read Appendix C first.** The line above was true when written. The k5v4
> decode dispatch has since landed (`9759fa55`, harness `cbfbee31`) and is green through the
> public op. Appendix B's blocker was real but was only one of FOUR hardcoded-k4v2 sites; the
> fourth was downstream of everything listed there. Batching and §5 serve validation are still
> not done — see Appendix C for what is and is not claimed.

`ninfer_ops` and `ninfer-serve` both build clean at sm_120a as of `b9c78dec`. The k5v4
commits have not been rebuilt through the full library (kernel + test TU only) — do a
`cmake --build build --target ninfer_ops -j 16` before trusting the tree state.

## The one thing to escalate before anything else

The shipped `tests/slice4_kvarn_test.cu` **cannot see K**. Its scale generator uses one flat
range for all six fields (`0.02 + 0.01*r`), so dequantized K/V are ~1e-3, QK scores are ~0.01,
softmax degenerates to a uniform average, and the output is independent of K.

Proved by mutation: stubbing `gqa_decode_slice4_kvarn.cuh`'s K dequant to `code = 0` leaves
`ninfer_slice4_kvarn_test` **PASSING all 10 cases**. (I restored the header and re-ran green;
nothing shipped was changed by the probe.)

Not a claim that k4v2 serving is broken — a claim that every gate certified by that test
attests less than the record implies, and k4v2 is the shipped default KVarN tier. The fix is
per-field scale ranges (see `tools/i4_oracle/k5v4_oracle.cu` lines ~160-185 for a working
version). Porting it into `tests/` may flip a shipped tier's gate to red — that placement
call is the coordinator's, and I deliberately did not make it.

## What is left for k5v4 (A2's flip checklist items 2/3/5 are mine)

1. `layouts_impl.h` — add an explicit `KvarnK5V4` arm. My current code **throws** on it
   (deliberate: no silent fallthrough). It must map to `DType::KVARN_K4V2` **plus**
   `kvarn_k_bits=5 / kvarn_v_bits=4` on the view — the marker alone is not enough, the
   widths travel separately on `PagedKVLayerView`.
2. `tp2_backend.cpp:357` `decoder_spec` — `is_kvarn_kv` tests **only** `KvarnK4V2`, so
   k5v4 currently hits the BF16 fallthrough. This is the same bug class as int4's; I found
   it by grep, it is still live. Plane push is already width-generic (256*5/8 = 160 B).
3. Width guard in `serve_options.cpp` — refuse `--cache-type-k/v` overrides that don't
   match 5/4 for the k5v4 tier.
4. Launcher: a `launch_tc_partial_unified_kvarn` variant (or a width-parameterized one)
   that reaches slice6. slice4's launcher is in `src/ops/launcher/gqa_attention_kvarn.cu`.
5. **Batching stays k4v2-only** (checklist item 4): `tp_engine.cpp:184` and
   `tp2_backend.cpp:2386` deliberately remain k4v2-gated, so k5v4 is single-seq decode.
   Decide explicitly at the flip; do not let it degrade silently.
6. Prefill cache-fill is still Phase-3 for BOTH int4 and k5v4 → §5 serve validation
   (identity / acceptance / t/s / KLD / VRAM) stays blocked. Do not report either as
   end-to-end done.

## Lane hygiene — the rules that actually bit

- **Never share a working tree.** A2 and I were both in `wo-117-kv-types` simultaneously;
  HEAD moved under me twice and their uncommitted codec edits sat next to mine. The
  coordinator's "revert the kvarn_codec edits" was destructive *because* of this. I staged
  only my own paths and committed; nothing was lost. If you resume, confirm your tree is
  yours alone before editing.
- **Verify an instruction's premise before executing it**, especially a destructive one.
  Three catches this session (revert-A2's-codec, delete-the-artifact, flip-the-guard) and
  all three were wrong instructions from a good-faith source.
- **The guard I did NOT delete:** `serve_options.cpp` refuses q4_0. I narrowed the reason
  rather than removing it, because prefill branches `dtype==I8 ? int8 : bf16` and an I4
  cache would be filled with garbage. It comes out when Phase-3 prefill lands.
- **Doc numbers come from the coordinator.** 150 was claimed and verified free.
- **GPU:** get a written grant; guard on foreign CUDA contexts not ports; `gpu_guard.sh`
  is STILL branch-only (`wo-2a-batched-serving/tools/smoke/diag/`), not on main. Never
  system-wide pkill. Check `df -h /` before big builds — disk was 34 G free here.

## Test-harness discipline that paid for itself

- My first int4 run was 9/9 RED and the kernel was innocent — I'd written offset-binary
  `n-8` where the contract is two's-complement `(n^8)-8`. Fix your own side; do not bend
  the kernel to make a test pass.
- Then **mutation-test before believing a green.** Two mutations per type. The k5v4 one is
  what exposed the vacuous-K problem — a passing test that cannot see its own subject.
- **An oracle that mirrors the kernel's dequant self-certifies.** That's why the
  known-answer vector matters: A2's `0x41,0x0C,0x52,0xCC,0x41` is derived on paper from the
  docs/69 LSB-first rule, independent of both implementations.
- **Fused append needs two checks.** The oracle consumes the cache read back from the
  device, so a wrong quantizer is self-consistent. Checking the written codes/scales
  against an independent host quantization is what catches an `absmax/127`-style bug.
- Harnesses live in `tools/i4_oracle/` (mine), not `tests/` (the tests lane's):
  `i4_oracle.cu`, `k5v4_oracle.cu`, `k5v4_known_answer.cu`. Build line:
  ```
  /usr/local/cuda-13.1/bin/nvcc -std=c++20 tools/i4_oracle/k5v4_oracle.cu -o /tmp/k5v4_oracle \
    -I include -I src -I tests -I third_party --expt-relaxed-constexpr -O2 \
    -diag-suppress 20050 -diag-suppress 177 -gencode arch=compute_120a,code=sm_120a \
    build/src/libninfer_ops.a build/src/libninfer_core.a build/src/libninfer_nvfp4_tma.a \
    -lcudart -lcublasLt -ldl
  ```

## Known perf debt (recorded, not solved)

slice6's K bank is linear at 20 B/row, so the dequant's `d = 8*lane + i` reads land 160 B
apart and `160 mod 32 == 0` → all 32 lanes hit one smem bank → 32-way LDS.U8 conflict. The
4-bit route avoids this with a skew. Not gated: sub-8-bit tiers are capacity fixes (decode
is launch/occupancy-bound, not GEMV-bandwidth-bound — docs/28 §16-18, docs/05 §79), and
neither type is servable until Phase-3 prefill.

## §5 budget numbers to use (A2's, NOT §1's table)

int4 = **9792** B/token/rank. k5v4 = **12240** B/token/rank.

---

# Appendix — k4v2 test-vacuity, written for the tests lane

**Task (coordinator ruling, 2026-09-05 ~01:13Z):** port the per-field scale fix into
`tests/slice4_kvarn_test.cu` and re-run the k4v2 gate. Do **not** expect me to touch that
file — `tests/` is the tests lane.

**The defect.** `tests/slice4_kvarn_test.cu:161`:

```c
for (auto& f : d.scales) f = 0.02f + 0.01f * rng.next();  // small, non-degenerate
```

One flat range for all six scale fields. Consequence chain: dequantized K/V land at ~1e-3 →
every QK score within ~0.01 of zero → softmax degenerates to a **uniform average** over the
keys → the attention output stops depending on K at all. The test therefore asserts on
something that is a function of V only.

**Proof it is blind (reproducible).** Stub the k4v2 kernel's K dequant in
`src/ops/kernel/gqa_decode_slice4_kvarn.cuh` — replace the two unpack lines with
`x0[i] = (0.0f * s + z) * sr0; x1[i] = (0.0f * s + z) * sr1;` — rebuild
`tests/slice4_kvarn_test.cu`, run it. It **PASSES all 10 cases.** Restore the header
afterwards (`git checkout src/ops/kernel/gqa_decode_slice4_kvarn.cuh`) and re-run to confirm
the tree is back to green.

**The fix.** Per-field ranges so K and V are O(1) and scores have real spread. Working
implementation in `tools/i4_oracle/k5v4_oracle.cu` (search for "Per-FIELD scale ranges"),
which mirrors the 1152-float field layout:

| field | range | why |
|---|---|---|
| K `s_col` (0..255) | `0.03 + 0.02*r` | code*s spans ~[0,1] for 4-bit |
| K `zp` (256..511) | `0.6*r` | centered offset |
| K `s_row` (512..575) | `1.0 + 0.5*r` | pushes K to O(1) |
| V `s_col` (576..831) | `1.0 + 0.5*r` | V magnitude |
| V `zp` (832..895) | `0.6*r` | centered offset |
| V `s_row` (1088..1151) | `0.05 + 0.03*r` | code*s spans ~[0,0.8] for 4-bit |

Note the shipped tier is k4v2 (K 4-bit, V **2-bit**), so V's `s_row` range wants to be about
double the 4-bit values above to reach comparable magnitude — retune, don't just copy.

**After porting, re-run the mutation.** If the test still passes with K stubbed, the fix
didn't take. Expected: it goes RED.

**Then one of two things is true, and both are useful:**
- It goes red on the *unmutated* kernel too → a genuine k4v2 K-path bug is surfacing for the
  first time. That's mine to investigate; say so and I pick it up.
- It goes red only under mutation → k4v2's K path was already correct and the gate is now
  actually armed. Best case: the shipped tier is fine and we simply couldn't tell.

**What this is not.** Not evidence k4v2 serving is broken. It is evidence that every gate
this test certified — Phase-2/3/4, the 2a batched work — carries a weaker attestation than
the record implies, because the test could not see half of what the prologue dequantizes.

---

# Appendix B — k5v4 dispatch: what I traced before stopping (not started)

The blocker is **not** the enum arm, it's that **the bit-widths do not reach the single-GPU
planning path.** Verified by reading, not assumed:

- The widths DO travel fine on the TP2 path: `tp2_backend.cpp` `decoder_spec` sets
  `.kvarn_k_bits/.kvarn_v_bits` from `o.kvarn_k_bits`, and the chain
  `DecoderStateSpec` → `plan_cache(..., kvarn_k_bits, kvarn_v_bits)` → `PagedKVCacheLayout`
  → `PagedKVCache::kvarn_k_bits_` → `layer_view`/`batch_layer_view` is complete.
- The single-GPU path does **not**: `SequencePlanningInputs` (`layouts.h:69-70`) carries
  `kv_dtype` and `kv_quant_group` but has **no** `kvarn_k_bits`/`kvarn_v_bits` field, and
  `layouts_impl.h:118-120` builds the `DecoderStateSpec` from `plan.kv_dtype` /
  `plan.kv_quant_group` only. So `plan_cache` falls back to its `= 4` / `= 2` defaults and a
  k5v4 cache would be **planned as k4v2** — wrong plane sizes, silently.

So checklist item 2 needs one of:
  (a) add `kvarn_k_bits`/`kvarn_v_bits` to `SequencePlanningInputs`, populate from the
      storage enum next to the existing `kv_dtype` ternary, and pass them at the
      `DecoderStateSpec` construction site; or
  (b) derive them inside `layouts_impl.h` from `options.kv_cache` at the spec site.
(a) is more consistent with how `kv_dtype` already works. Either way, add a
`static_assert`/throw if a `KVARN_K4V2` dtype is planned with bits that don't match a
registered tier — that mismatch is exactly the silent-wrong-tier class.

**Do not forget the fallthrough I already found and did NOT fix:** `tp2_backend.cpp:357`
`is_kvarn_kv = o.kv_cache == KvCacheStorage::KvarnK4V2` — k5v4 misses it and hits the BF16
arm, which allocates a full-width bf16 pool and would fake the §1/§5 VRAM check. Fixing
`layouts_impl.h` alone leaves this one live.

**After wiring, the first thing to run is the known-answer + oracle, not a serve run** —
serve is still refused for k5v4 and prefill cache-fill is Phase-3, so a green serve is not
achievable in this lane and should not be chased.

---

# Appendix C — k5v4 decode dispatch: DONE (9759fa55 + cbfbee31). What appendix B missed.

Written after the dispatch landed and was rebased onto `main` (de4c0598). Verified by running,
not by reading. If you are resuming this lane, this section replaces the "NOT WIRED" status.

## Appendix B was right, and incomplete

The missing bit-widths on the single-GPU planning path were real, at the exact lines cited, and
fixed via option (a). But fixing everything in appendix B **still would not have served k5v4**.
There were four hardcoded-k4v2 sites, and two of them were not in any checklist:

| # | site | in checklist? | symptom if unfixed |
|---|---|---|---|
| 1 | `layouts_impl.h` / `SequencePlanningInputs` widths | appendix B | k5v4 planned at k4v2 plane sizes, silently |
| 2 | `tp2_backend.cpp:358` `is_kvarn_kv` | item 3 | full-width bf16 pool, faked §1/§5 VRAM |
| 3 | `kvarn_small_t_unified_launch` builds `PagedKVBatchLayerView` without copying `kvarn_k_bits`/`kvarn_v_bits` | **NO** | launcher always saw 4/2 → wrong prologue, even with perfect planning |
| 4 | `ops/wrapper/gqa_attention.cpp` `validate_cache`/`validate_batch_cache` require `k_pages {D/2}` / `v_pages {D/4}` for ANY `KVARN_K4V2` | **NO** | public wrapper **rejects** a correctly-planned k5v4 cache outright |

Site 4 is the one to remember: it is *downstream* of every site the checklist named, so the
whole upstream chain could have been perfect and the first k5v4 attend would still have thrown
`invalid shape for cache k pages`. It was found by running a harness through the public op.

**Also: slice6 was not in `libninfer_ops` at all.** `gqa_decode_slice6_kvarn_k5v4.cuh` was
`#include`d only by `tools/i4_oracle/k5v4_oracle.cu`, which launches the kernel *directly*. The
oracle was green against a kernel no production path could reach. A kernel-level oracle cannot
detect missing dispatch — that is a structural gap in this lane's test strategy, not a one-off.

## Two things that will bite the next person cloning a kernel

- **Duplicate `inline constexpr` across two kernel headers.** slice6 re-declared slice4's
  `kKvarnKStageRow = 36`. Harmless while each header is included once per TU; a redefinition
  error the moment one TU includes both. Rename, don't duplicate, when cloning into
  `ninfer::ops`.
- **Designated initialisers silently default the fields you forget.** Sites 3 and (the
  `DecoderStateSpec` at) 1 are both this: a struct with a `= 4`/`= 2` default, an initialiser
  that omits the field, and no compiler warning. Adding a field to a view struct means auditing
  every construction site of it.

## What is claimed, and what is not

Claimed, re-verified after the rebase onto main: k5v4 oracle 10/10 (worst 0.0312 / tol 0.0616,
unchanged), known-answer PASS, dispatch harness 6/6, and the shipped k4v2 tier green including
`kvarn_batched_ops` bit-identity and gemini's now-**non-vacuous** `slice4_kvarn_test`.

NOT claimed:
- **Batching** — `tp_engine.cpp:184`, `tp2_backend.cpp:2386` still k4v2-only (item 4, by
  decision). A hard throw now fires if `MultiBatch` + k5v4 ever co-occur.
- **Serve** — the k5v4 refusal stays. Its reason was narrowed from "no dispatch" to "dispatch
  wired, not §5-validated", which is not the same as removing it.
- **§5 end-to-end** — not run for either type. ~~prefill cache-fill is still Phase-3 for int4
  AND k5v4~~ **CORRECTED by d90f8556: that was true of int4 only.** KVarN prefill never reaches
  the `dtype == I8 ? int8 : bf16` branch that blocks int4; it writes through the same
  width-aware commit path as decode and reads through the width-generic materialize `_w`
  kernel, and a k5v4 T=8 run confirmed it (rel-L2 0.106 vs BF16, against 13.28 for a k4v2
  misread). So k5v4's §5 is DECOUPLED from Phase-3 prefill; int4's is not. Neither type is
  servable yet, but for different reasons — see Appendix D.
- **Shape coverage** — slice6 launches only the oracle-proven `(TokenTile, Wc)` pairs: Wc=2 at
  T=1, Wc=4 at T=2..6. `slice4` uses Wc=4 at T=1; **that shape has never been run for slice6.**
  If perf parity matters, add the oracle case first, then change `WarpsPerCta` in
  `launch_tc_partial_unified_kvarn`.

## Harness discipline: one near-miss worth recording

The dispatch harness's first run reported 1.06 rel-L2 against the BF16 reference — looked like a
kernel bug. It was the harness's own index mapping (source tensor is `{D, heads, tokens}`, so
`d + D*(h + heads*t)`; I wrote `d + D*(t + G*h)`). Fixed my side, not the kernel. Correct
numbers: 0.105 for k5v4 vs 13.75 for a k4v2 misread of the same bytes.

The harness is deliberately a **differential** test (same bytes, two tiers → must differ),
because the host codec in this tree is still k4v2-only, so a dequant-cancelled k5v4 reference is
not available here. A2's width-generic codec would upgrade it — worth doing.

## Repo-wide: do not build all targets

`cmake --build build` (no `--target`) builds 88+ test executables, each statically linking the
207 MB `libninfer_ops.a` → ~17 GB. Free space went 34 G → 17 G in one build. Build `ninfer_ops`
plus the specific test targets you need. Cleaned up after the fact (134 executables removed,
build dir 18 G → 898 MB), but this is a shared-disk hazard for every lane, not just this one.

---

# Appendix D — §5 readiness after the prep pass (3adc56ca). What can and cannot run.

The decode lane is merged. This section is for whoever picks up §5, and it exists because
"§5 is blocked on Phase-3 prefill" — the line in Appendix C — was only half true, and the real
blockers turned out to be elsewhere.

## §5 was not executable, and not because of the GPU

Three separate causes, all found by reading the driver's inputs rather than by running it:

1. **`ninfer_bench` could not select any KVarN tier** (`35d383ce`). There were three
   `parse_kv_cache` implementations with three different accepted-name sets:

   | site | knew |
   |---|---|
   | `src/serve/serve_options.cpp` | bf16 int8 kvarn_k4v2 q4_0 kvarn_k5v4 |
   | `apps/cli/options.cpp` | bf16 int8 kvarn_k4v2 |
   | `bench/.../ninfer_bench_support.cpp` | bf16 int8 |

   `ninfer_bench` is what produces acceptance rate and t/s — two of the five §5 checks — so
   those were unmeasurable for **every** KVarN tier, shipped k4v2 included. Now one shared
   table (`ninfer::kv_cache_from_name`) with the aliases pinned by tests.
2. **A2's §5 harness crashed on every real invocation** (`6ac82f77`). `main()` did
   `if args.selftest:` and no subparser ever defined that attribute → `AttributeError` before
   checking anything. `--selftest` passed the whole time because it returns *above* the broken
   line. **A selftest that bypasses the CLI is not testing the CLI.**
3. **KLD genuinely cannot run.** No producer exists. slice4/slice6 both implement
   `dump_scores`/`dump_probs`/`dump_q`, but every caller in `src/` passes `nullptr` (the docs/136
   comment in `gqa_attention_kvarn.cu` says so explicitly) and nothing writes the `NVKLDMP1`
   format A2's harness reads. A2's docstring already flagged this as "A1 lane; flag when wired" —
   it is flagged, not wired. **This is the one remaining §5 work item**, and it means allocation
   + file I/O on a decode hot path: do it in its own window with a run to validate it, not as a
   drive-by.

## Current §5 status per check

| check | status | how |
|---|---|---|
| VRAM | executable | `tools/ops/run_117_s5_validation.py` → `validate_117_dtypes.py vram` (differential; fixed terms cancel, so it validates the budget branches without replicating constants). k5v4 = 12240 B/token/rank. |
| acceptance | executable | `ninfer_bench` per tier, `spec.acceptance_rate`, vs the Phase-0 matrix |
| t/s | executable | `ninfer_bench` per tier |
| identity | executable | greedy stream, tier-vs-reference and mtp1-vs-plain |
| KLD | **BLOCKED** | needs the dump producer above |

The driver tracks `ran` separately from `ok` and prints `BLOCKED` with a reason, so a step that
did not execute can never be read as a pass. `--require-kld` exits 1. That property is the
point of the script; do not "simplify" it into silent skips.

## The escape hatch, and its deliberate asymmetry

`NINFER_ALLOW_UNVALIDATED_KV_DTYPE=1` (`58ed81e5`) lifts **only** the "not yet §5-validated"
refusal, and **only** for k5v4. It cannot reach q4_0, and it does not disarm the width registry
or the planning/dispatch throws. The distinction, which the coordinator approved:

- k5v4's blocker is a **process gate** — "we have not measured this". Liftable for a measurement
  run, which is precisely what §5 is.
- q4_0's blocker is a **correctness defect** — prefill fills an I4 cache as bf16, so any number
  from it measures garbage. Bypassing it would produce a plausible wrong result.

It prints a loud stderr warning, and `"0"`/`""` mean off. Behaviour is identical when unset.

## Two anti-patterns worth not repeating

- **N parsers over one enum with divergent accepted sets.** Each site looked fine alone; the
  capability gap was only visible as a missing cross-product. Shared table + pinned aliases.
- **A selftest that bypasses the CLI under test.** Green selftest, broken real invocations.

## Also fixed in passing

The merge left **two consecutive** `if (kv_cache == KvarnK5V4)` blocks in `serve_options.cpp`;
the stale pre-dispatch one threw first, so the accurate narrowed message was dead code and every
user was told the dispatch "is NOT yet wired" two commits after it landed. Functionally still a
refusal, but a confidently-wrong triage message. Syntax-clean ≠ semantically-clean.

## Appendix E — §5 validation closed (2026-09-05), and what is still owed

**Verdict: 4/5 executable checks PASS on the real 18 GB artifact; KLD blocked.** Committed
`02ba5583` (results + tooling) and `6d25c44f` (the corrected gate). Full numbers in
`docs/k5v4_s5_validation_results.md`. k5v4 is validated-for-serve.

- vram PASS — differential model validated <0.2% against 6 measured preflight cells.
- acceptance PASS — k5v4 0.738 vs bf16 0.739 (0.1 pt; bar is ±0.5 pt).
- t/s PASS — 1.54× over no-speculation, identical to bf16's 1.54×.
- identity PASS against reference — **and the gate itself was wrong.** See §5: byte-identity
  (mtp1==mtp0) is unachievable on any tier, bf16 included (2/4 prompts diverge). Redefined as
  divergence-rate/position vs bf16. k5v4 diverges 1/4, better than the reference.

**Re-derive these rather than trusting them** — they are run-dependent, from one build on one
box, and the acceptance/t-s figures are n=5 rounds over 4 prompts. The single-prompt identity
result was actively misleading; it took four prompts to see the gate was the problem.

### Owed work, in priority order

1. **KLD dump producer (`NVKLDMP1`)** — no producer exists anywhere in `src/`; slice4/slice6
   take `dump_scores`/`dump_probs`/`dump_q` and every caller passes `nullptr`. This is the last
   §5 check. Do NOT wire it as a hot-path change: it means an allocation plus file I/O on the
   decode path. Its own validated pass.
2. **Auto KV-capacity sizing bug** (coordinator added to my queue, not urgent, worked around).
   At `--max-context 8192` auto proposed 409,344 tokens for k4v2 (4043 MiB of context) against
   ~2459 MiB free after fixed (12,827) + workspace (1024), so preflight **refused a config that
   fits**. Reproduced on bf16 and k4v2. Workaround in the driver: explicit `--kv-capacity`.
3. **k4v2 acceptance 2.6 pt below bf16**, outside the ±0.5 pt bar — a shipped default failing a
   written criterion. Surfaced to the user; awaiting whether that is expected for the more
   aggressive tier or needs a tier-specific bar. Do not "fix" this unilaterally.
4. **DFlash2 enum pairing with agent2** — see below.

### The -Wswitch guarantee is much weaker than it looks (matters for the enum work)

Measured on `321ddc1e`: adding `DFlash2` to `SpeculativeBackend` makes the compiler ask about
**3 case labels in 2 files** (`speculative_options.h` 6 labels/2 switches, `layouts_impl.h`
3 labels/1 switch — the latter has no `default`, which is why it warns). The other **67 sites in
7 files are `==`/`!=` comparisons that go silently to the non-DFlash2 branch**, including all 51
in `program_impl.h`. So a green build after filling the switch arms means "the 3 switch sites
handled", **not** "all consumers handled".

Consequence: the exhaustive-switch-with-no-default pattern is good and agent2 landed it, but it
is not a safety net for removal. The `layouts_impl` DFlash2 refusal should stay until a **test**
proves the `program_impl` io-slot behavior per backend — not until the build goes green. This is
docs/117 §4 rule 10 again: a check covering a minority of sites buys confidence nobody earned.

Landing order is deliberately unresolved: landing the enum alone breaks agent2's
`speculative_options.h` build between our two commits. Either they land against a local enum
patch, or I hand over a combined patch so the branch never goes red.

## Appendix F — k4v4 is scoped, baselined, and ready to implement

Coordinator approved the recombination plan (template/extract, **not** a third 641-line fork) and
granted GPU for the baseline. All three preparatory steps are done and committed; the implementation
is deliberately NOT started, because it cannot be verified without a fresh grant and a half-edited
decode prologue is the worst thing to hand across a session boundary.

Commits: `87ff17fb` (plan) · `23fd2078` (baseline) · `9c232000` (stale-comment fixes) · `cdcfa340`
(extraction map).

### Do this first, in order

1. **Get a GPU grant**, then re-acquire a lease via `tools/smoke/diag/gpu_lease.sh`.
2. **Re-verify the baseline still holds** before editing anything — the tree has moved since it was
   captured (`docs/baselines/k4v4/MANIFEST.txt` has sha256 of each output). If a hash differs before
   you touched a line, the baseline is stale and must be recaptured; do not proceed against it.
3. Create `src/ops/kernel/gqa_decode_kvarn_prologues.cuh` and **lift verbatim** per the map's table.
   Do not re-derive any bit math — the 4-bit and 5-bit K sides differ in op size, op count, skew,
   and byte-pairing semantics, and re-deriving is how a tier ends up running and wrong.
4. Redirect slice4 and slice6 to the helpers. Keep `!page_ok` zero-fill **shared** (it writes both
   tiles; duplicating it per side can leave one stale on a bad page).
5. **Gate: byte-identical output for BOTH `<4,2>` and `<5,4>`** vs the manifest. Both, not one —
   both source kernels are edited.
6. Only then add `KvarnK4V4`: enum + `kvarn_tier_widths` `{4,4}` + one entry in the shared name
   table (which makes it parse in serve, CLI and bench simultaneously — the single-table work already
   paid for) + dispatch route + a §5 refusal like k5v4's, added rather than assumed absent.

### Facts established this session (re-derive the numbers, trust the conclusions)

- **k4v4 = K4 (slice4) + V4 (slice6). Zero new bit-manipulation code.** 11152 B/token, `kv_unit`
  11849 — between k4v2 (8976/9537) and k5v4 (12240/13005).
- Stage smem 4224 + 4096 = **8320 B**, *smaller* than k5v4's 9232, so slice6's single-buffered
  pipeline ordering carries over with no `__launch_bounds__` regression. The 4-bit skewed bank also
  lacks k5v4's recorded 32-way `LDS.U8` conflict, so k4v4's K side should be faster.
- **The §5 prefill premise is verified, not assumed**: `validate_cache` (gqa_attention.cpp:108-126)
  handles KVarN in a branch that early-returns with width-generic checks, so the
  `dtype == I8 ? I8 : BF16` line at :141 is unreachable for any KVarN tier. k4v4 does **not** hit
  int4's Phase-3 blocker.
- **k3v3 is NOT free** — it needs new code for both sides (12 B K rows, 96 B V rows, 3 ∤ 8 so codes
  straddle on a 3-byte/8-code period). Budget it at roughly slice6's original cost.
- **k3v3 is VRAM-identical to k4v2** (8976 B/token) because the budget formula depends on
  `k_bits + v_bits`. A K/V balance change is free in capacity terms, costing only accuracy
  trade-off. Useful for tier selection; not part of this work order.

### Two things this session should not be forgotten

- **The repo enables zero `-W` flags** (`docs/compiler_diagnostic_audit.md`). Any "the compiler will
  catch it" claim is unfounded until someone names the flag and where it's enabled. A2 landed
  `-Wswitch` narrowly in `wo/dflash2` (`a5c8ab81`); it is NOT yet on main.
- **A fork carries documentation for behaviour it does not have.** slice6 shipped three stale layout
  comments inherited from slice4, including one asserting a 4-bit K packing in a 5-bit kernel and
  one describing a `dq` smem scratch that exists nowhere in the file. Nothing tests comments. That is
  the concrete cost of forking, discovered by accident while planning around it.

### Rule 11 now applies to this lane's gate (REPO.md §4, main @ 275cd9df)

> An rc=0 from a mechanism that did not execute the thing under test is a tautology.

Step 2 of the k4v4 sequence above is where this bites hardest: the regression gate is a GPU re-run
compared against `docs/baselines/k4v4/MANIFEST.txt`. Before trusting a green run, confirm the
binaries you just executed were actually rebuilt from the edited headers —

```
find build -name 'tp_engine.cpp.o' -newer src/runtime/tp2/tp_engine.cpp    # must be non-empty
```

and for the oracle harnesses, rebuild explicitly (they are built by a hand-written `nvcc` line, not
by a CMake target, so nothing in the build system will notice they are stale). The oracle and
dispatch binaries were rebuilt from source before the baseline was captured precisely because
`types.h` had just changed and a stale binary would have pinned the wrong tree.

**Merge hold is in effect** (REPO.md §4a, owned by another agent, cleared via docs/77 §8). These 12
commits stay on `wo/117-int4` by design — do not merge to main, and do not treat the branch being
behind main (currently 2, both docs-only) as something to fix unilaterally.

### Merge hazard: 321ddc1e is SUPERSEDED by A2's seam, not additive

`321ddc1e` (this branch) and A2's `(f)` + auto-capacity work (`9e1c3c2c` on `wo/dflash2-scope`) edit
the **same region** of `tp_engine.cpp` — both mutate the budget construction a few lines apart:

| branch | line | what |
|---|---|---|
| `wo/117-int4` @ 321ddc1e | 749 | `budget.decoder_fixed_bytes += 768ULL*1024*1024;` guarded by `backend==Mtp && mtp_k>0` |
| `wo/dflash2` @ 9e1c3c2c | 785 | `budget.decoder_fixed_bytes += drafter_fixed_bytes(...)`; **zero** `768ULL` left in the file |

Expect a conflict. **The intuitive "keep both hunks" resolution charges MTP 1536 MiB instead of 768**
— spurious preflight refusals at exactly the long-context configs people want to serve, and it reads
as a budget-model drift rather than a merge artifact.

Correct resolution: **take A2's version wholesale.** Semantics were checked, not assumed:
`backend==Mtp ? Mtp : None` then `drafter_fixed_bytes(Mtp, mtp_k) = mtp_k>0 ? 768 : 0` reduces to
`backend==Mtp && mtp_k>0 -> +768`, identical to 321ddc1e. The behaviour is preserved by the seam;
only the text disappears.

**One-command check after merging:** `grep -n 768 src/runtime/tp2/tp_engine.cpp` must return
**nothing** — the constant belongs to `tp2_budget.h` now. Same shape as REPO.md rule 11: verify the
artifact, not the intent.

### SUPERSEDED — the :793-794 gate is CLOSED (verified 2026-09-05)

I flagged above that `tp_engine.cpp:793-794` re-derived KVarN widths inline while the probe moved to
`kvarn_tier_widths()`, and wrote "**do not add the tier to the table until :793-794 reads from it
too**". **A2 fixed it in `71907c53` and I verified the fix independently** — :795-797 now read
`widths->k_bits/v_bits` from the same `kvarn_widths` optional the probe consumes at :670/:672, both
derived once at :666. Zero surviving inline width literals.

So the constraint on step 6 is **lifted**: adding `KvarnK4V4` to `kvarn_tier_widths()` is now picked
up by both paths automatically, which is the property the fix was supposed to establish. That note
is kept struck rather than deleted because the reasoning behind it still applies to any future
per-tier literal.

**Still open on A2's side, and it does NOT gate step 6:** the runtime divergence guard (`81445f77`)
is inert — `tp_engine.cpp:820` overwrites the stamped reference one line before `:824` compares them,
so the condition is `x != x`. Proven by execution against the real header with a genuine mis-tier
(8976 vs 11152): no throw. The test-level guard (`9c7a3a24`) mirrors two dispatch chains inside the
test, so with production now sharing one derivation it reflects itself rather than the code. Neither
is a reason to block k4v4 — the single shared derivation is the actual protection — but do not treat
"the guard is green" as evidence that a tier is budgeted correctly. Verify the derivation count.

The general lesson for this lane, and the fifth instance tonight: **a guard that cannot fail is
indistinguishable from a guard that passed.** If a check has no test exercising its failure path,
assume nothing until you make it fire on purpose.

### Update: step 6 is HALF DONE by A2 (`wo/k4v4-cpu` @ 58dd47d0) — and it carries a merge-order trap

The branch appeared after I first could not resolve it. Its 11-line `types.h` edit is exactly the
right shape: `KvarnK4V4` enumerator, `kvarn_tier_widths` case `{4,4}`, parse aliases
`kvarn_k4v4`/`k4v4`, name mapping — and `is_registered_kvarn_widths` **deliberately not extended**,
with a comment that a planned k4v4 cache "must fail loudly, not read garbage." Correct call: the tier
becomes parseable and budgetable while the registry that gates serving stays shut until a prologue
exists.

**So step 6 above is done except the dispatch route. Do not re-write the tier table.**

⚠️ `git merge-base --is-ancestor 71907c53 58dd47d0` → **NO**. Branch point is `a6757205`, which
predates A2's own `:793` fix and my `DFlash2` enum commit (their `types.h` has zero `DFlash2`, so
expect a visible — and harmless — conflict in the enum).

If `wo/k4v4-cpu` lands **without** `71907c53`:

| path | derivation | k4v4 result |
|---|---|---|
| probe | `kvarn_tier_widths` → {4,4} | 11152 B/token ✓ |
| preflight | old `== KvarnK5V4 ? 5 : 4` → {4,2} | 8976 B/token ✗ |

— the reverse mis-tier, short by 289/361/452 MiB at 131k/164k/205k. Invisible under auto-capacity;
on an explicit `--kv-capacity` preflight rubber-stamps a config that OOMs at load. The branch that
adds the tier is older than the fix that makes adding it safe.

**Required order:** `71907c53` (fix) + `481b65a4` (working guard) → `wo/k4v4-cpu` → decode prologue.
After any merge, re-run the derivation-count check rather than trusting the ordering:

```
grep -n '? 5 : 4\|? 4 : 2\|8000ULL' src/runtime/tp2/tp_engine.cpp    # must be empty
grep -n 'kvarn_widths\|widths->'    src/runtime/tp2/tp_engine.cpp    # probe AND preflight
```

Note the guard now genuinely works (verified in both directions against the real header: fires on
divergence, silent on agreement), so if the fix is ever missing but the guard present, startup throws
with both numbers. First mechanism tonight that would have caught something on its own.

### ⚠️ `wo/k4v4-cpu` is NOT safe to merge as-is (found after the cherry-picks)

A2 cherry-picked `71907c53` + the guard onto `wo/k4v4-cpu` to satisfy the ordering constraint, and
reported `Full suite rc=0`. The constraint is satisfied. **The branch is still wrong**, because the
cherry-picks did not include `a77094ac` (the probe factory):

```
wo/k4v4-cpu:668   ... kv_cache == KvCacheStorage::KvarnK4V2 ? 8000ULL
```

That literal is absent from `wo/dflash2-scope` and present only here. The probe is the pre-factory
hand-built `Budget probe;`, so **both original bugs are back** (no drafter charge; stale 8000), and a
third is worse — **the ternary chain has no `KvarnK4V4` case**, so the tier this branch adds falls
through to the **bf16** figure:

| | B/token | kv_unit |
|---|---|---|
| probe | 34,816 (bf16) | 36,992 |
| preflight | 11,152 (correct) | 11,849 |

At the measured 3638 MiB slack: auto-capacity proposes **103,122** tokens vs a correct **321,944** —
k4v4 silently serves **3.1× less context than it can**. It fails *safe*: no crash, no refusal, no
test failure. That makes it harder to find than the mis-tier, not easier.

`rc=0` missed it because the every-tier test asserts the test's own two mirrored chains agree;
neither mirror is the branch's real probe.

**Before touching k4v4 decode work, verify:**
```
git show wo/k4v4-cpu:src/runtime/tp2/tp_engine.cpp | grep -n "8000ULL"        # must be EMPTY
git show wo/k4v4-cpu:src/runtime/tp2/tp_engine.cpp | grep -n "Budget probe;"  # must be EMPTY
```
Fix is to cherry-pick `a77094ac` onto the branch.

**Also open:** `083132c5` and the call-site comment at `tp_engine.cpp:827` both claim the guard's
"failure path unit-tested". `grep -rn verify_no_divergence tests/` → **0 hits**, and the commit
touched no test file. The refactor is good (pure method, verified to fire on divergence and stay
silent on agreement); only the coverage claim is false. A comment asserting a test that doesn't exist
is the slice6 stale-comment pattern relocated into a call site — and it is the reason a future
reviewer will skip writing the test.

### Design consequence for the k4v4 prologue work (from the matrix finding)

A2's rule-13 matrix is vacuous for a reason that generalises: **enumerating the sites is not enough
— the sites must be callable.** The test does not link `ninfer_engine` and does not include
`tp_engine.cpp`, so no mutation of the probe or preflight can affect it; its "two sites" are three
copies of one if/else inside the test function. The precondition for rule 13 is a seam.

That applies directly to the k4v4 work in this appendix, so design for it now rather than retrofitting:

- **Tier → per-token figure.** Do not add a `KvarnK4V4` arm to any inline ternary chain. The
  fallthrough-to-bf16 bug that crippled k4v4 by 3.1× existed precisely because the dispatch was an
  inline chain with a default branch. Route everything through `kvarn_tier_widths()` and, if a
  figure-dispatch function is needed, make it a pure `Budget::` static so a test can call it.
- **Tier → decode prologue.** The extraction plan already makes this a composition of named per-side
  helpers, which is the callable form. Keep it that way — a `switch` inside `gqa_attention_cached`
  with a bf16 fallback would recreate the same silent-misdispatch class one layer down.
- **Registry gate.** `is_registered_kvarn_widths` must gain `(4,4)` only when a prologue exists
  (A2's `wo/k4v4-cpu` deliberately leaves it out — correct, and the fail-loud pattern to preserve).

**Verification commands, re-run rather than trusted:**
```
grep -rn "verify_no_divergence" tests/                     # 0 hits as of 2026-09-05
sed -n '104p' tests/CMakeLists.txt                         # no LIBRARIES => engine not linked
git show wo/k4v4-cpu:src/runtime/tp2/tp_engine.cpp | grep -c "8000ULL"   # 0 = probe fix landed
```
