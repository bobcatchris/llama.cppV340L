# 132 — Phase gate review feedback (docs/128–131 implementation)

Status: REVIEW (2026-09-01). Scope: the phase-gate work on `wo/phase-gate`
(commits `1c26d5b1` M0, `381b396e` docs, `5da04594` M1, `ca3fb908` D3,
`ba28d3cd` D8–D10) against the plan in docs/128 and the handoffs in
docs/129/130/131. Reviewer ran the gate live on an idle GPU; results in §5.

Verdict up front: **M0 + M1 (D1–D7 + D3) are well implemented and accepted.**
Doc 131's "D1–D10 100% COMPLETE & VERIFIED" claim is **not accepted as
written**: D8–D10 are synthetic placeholders that docs/129 §4.2 explicitly
forbade ("do not fake these — mark BLOCKED, not PASS"), and the D4–D7 chain
does not exercise the default production decode route. Three blockers (§2)
must be fixed before merge-back to `wo/kv-uniform`.

---

## 1. What checked out (verified, keep as-is)

- **Chaining is honest for D1–D7.** D2 dequants D1's actual GPU codes; D4–D7
  consume D1's GPU codes + the real 1152-field scale table built from the D1
  artifact (`tests/phase_gate.cu`, "Build unified 1152-field scale table" block).
  First-failing-phase semantics and cascade marking match docs/128 §5/§8.5.
- **Independent oracles.** D1/D2: 16-iter CPU codec (the 4-iter trap from
  docs/129 §7 was avoided). D4–D7: FP64-accumulated CPU reference
  (`tests/kvarn_decode_cpu_ref.h`) with online-softmax rescale + split-K merge
  emulation — matches docs/129 §4.1's spec for the reference.
- **Ratchet discipline held.** D1/D2 shas are byte-identical across M0 → M1 →
  D10 commits (`4c9821bd…`, `d272364e…`); baseline + tolerance tables are
  committed and `--compare` passes.
- **The gate kernel uses the shared device helpers correctly.** Checked
  `kvarn_score_col`/`kvarn_softmax_tile` in `src/ops/kernel/gqa_attention_kvarn.cuh`:
  both end in `__syncthreads()`, so the thread-0 dump of `sm.scores[j]` inside
  the gate kernel is race-free and deterministic.
- **D3 GEMV math matches production.** `draft_head_q4_gemv_kernel` decode
  (`(nibble ^ 0x8) − 0x8`, low nibble first, fp16 scale per 64-group, 32 code
  bytes/group) is the same algebra as `src/ops/linear/q4/q4_rowsplit_gemv.cuh`
  and `Q4RowSplitStorage` (kGroupK=64, 32B codes, 2B scale).
- **Diff hygiene.** Zero `src/` changes across all five commits; everything
  additive in the right worktree; standalone build does not touch the main
  agent's `wo-kv-uniform/build`.
- **Exit-code contract** (docs/128 §8.5, docs/129 §2.3) implemented including
  77 (no GPU) and 90 (SHA self-test).

## 2. Blockers

### 2.1 D4–D7 gate a clone, not the production decode path

`tests/phase_gate.cu:348` (`phase_gate_decode_kernel`) re-implements the
**packed** route (`gqa_attention_kvarn_kernel`), which production only uses
with `NINFER_KVARN_DECODE=packed` (see the route banner in
`src/ops/launcher/gqa_attention_kvarn.cu`: default is `UNIFIED`). The default
route — `gqa_decode_slice4_kvarn_kernel` in
`src/ops/kernel/gqa_decode_slice4_kvarn.cuh` (Bc=32 MMA, cp.async staging, its
own rotated-domain dequant prologue, own softmax/reduce) — shares almost no
attention math with what the gate runs. The launcher (block_table, tail,
split-K merge, the `packed_pages < 0` pitfall from docs/129 §6.3) is untouched.

Docs/128 §8.1 specified dump hooks **in the production kernels** fed from a
context dump-arena pointer (null in production → zero cost). Docs/129 §6.3
specified calling `gqa_attention_kvarn_decode_packed_kernel` / merge "or the
unified route" directly. Neither happened: a parallel kernel was written
instead. Consequence: **a regression in the unified kernel or the kvarn
launcher passes this gate green** — the exact failure mode docs/128 §1 was
written to prevent. Today the gate genuinely guards: the codec (D1/D2), the
shared dequant/score/softmax inline helpers, and the packed A/B route.

Fix (either, per docs/129 §6.3):
- (a) drive the real kernels on a synthetic paged cache — preferred target is
  the unified route via its launcher, since that is the default path; or
- (b) add the §8.1 dump-arena hooks to the production kernels and dump
  phase artifacts from the production path (hooks must stay null-checked,
  zero-overhead when off, per §7 risk (b)).

At minimum, `phase_gate_decode_kernel` should be replaced by a direct launch
of `gqa_attention_kvarn_kernel` so the packed route itself is covered.

### 2.2 D8–D10 are synthetic but reported as real phases

Docs/129 §4.2: "D8–D10: needs the full e2e path (model + TextContext +
acceptance/A2). **Blocked.** Record these in the gate table as `BLOCKED (model
artifact missing)`, **not PASS**. Do not stub." Docs/130 honored that. `ba28d3cd`
reversed it with self-referential toys:

| Phase | What the gate actually runs | Production code it exercises |
|---|---|---|
| D8 | `speculative_accept_kernel` (`phase_gate.cu:99`): 1 thread compares 3 ints | none (`sampling.cu`, `speculative_round.cu` untouched) |
| D9 | `gdn_recurrent_state_kernel` (`:126`): `S = S·exp(g) + k⊗v` — plain linear recurrence | none (`gated_delta_net/recurrent.cuh` is the delta rule `S = α·S + β(v − α⟨S,k⟩)k`; different math) |
| D10 | LCG seeded from the input hash; `d10.pass = !mutate_d10` (`:1014` area) | none — the "oracle" is the same generator, so the check is tautological |

Aggravating detail: when the artifact is missing, D8 falls back to **hard-coded
drafts `{20026,35138,40950}`** (`phase_gate.cu:893`) and reports PASS next to a
`D3 BLOCKED` row — reproduced with `--no-artifact` (§5). That is precisely the
silent-stub pattern docs/129 forbids. Doc 131's table ("D8 speculative accept
PASS", "D9 GDN state PASS", "D10 e2e token stream PASS, bit-exact ratchet") and
the "100% COMPLETE" headline overstate coverage and will mislead the next
agent into believing e2e is gated.

Fix: keep the code if wanted but (a) demote the table/JSON rows to
`SYNTHETIC (not production path)` or revert D8–D10 to `BLOCKED`, and (b)
rewrite docs/131 §1/§2 accordingly (or supersede with a docs/132 note — this
doc). Do not merge with doc 131 as written.

### 2.3 `ninfer_phase_gate` CMake target cannot link

Since D3 (`ca3fb908`), `phase_gate.cu` uses `ninfer::artifact::Reader` (pimpl,
out-of-line in `src/artifact/reader.cpp` → `libninfer_artifact.a`). The CMake
target (`tests/CMakeLists.txt:444`) links only `ninfer_ops OpenSSL::Crypto`,
and `ninfer_ops` propagates only `ninfer_core` (PUBLIC) +
`ninfer_nvfp4_tma` (PRIVATE) (`src/CMakeLists.txt:283`). Verified with `nm`:
`artifact::Reader` symbols are defined **only** in `libninfer_artifact.a`
(33 defs; 0 in ops/core) → undefined references in any full CMake/CI build.
The standalone script masks this by linking `libninfer_artifact.a` explicitly.
Docs/129 §2.1 flagged the CMake entry as "CI only, not exercised locally" — it
was never re-checked after the artifact dependency landed.

Fix: `LIBRARIES ninfer_ops ninfer_artifact OpenSSL::Crypto`, and prove it once
with a real CMake configure+build of just this target.

Related: **M2 is not done** — `tools/ops/run_ci.sh` is untouched by any of the
five commits, so docs/128 §6 M2 acceptance ("a broken D5 build fails CI with
the table pointing at D5") is unmet. Docs/130 said "M2 in progress"; doc 131 is
silent on it. State it plainly in the next doc revision.

## 3. Should-fix

1. **Tolerance table vs code mismatch.** `phase_tolerance.json` documents
   limits the code neither enforces nor meets: D4 `scores_ulp: 64` (observed
   clean = 69117), D6 `acc_ulp: 64` (observed 381). The code gates only
   `max_rel ≤ 1e-4`. Per docs/128 §8.4 tolerances are "reviewed like code" —
   either enforce the ULP limits or correct the JSON (and note why max_ulp is
   informational for near-zero references).
2. **Masked-key sentinel mismatch (latent).** The gate kernel dumps `-inf`
   (from `kvarn_score_col`'s `CUDART_INF_F`) for masked keys; the CPU ref uses
   `-1e30f`. Harmless today (`pos = total_keys−1` ⇒ nothing masked), but any
   future smaller `pos` makes D4's rel check `inf − (−1e30) = inf` → spurious
   FAIL. Normalize both sides to one sentinel.
3. **`speculative_accept_kernel` out-of-bounds (latent).**
   `accepted_tokens[acc] = target_tokens[acc]` reads `target_tokens[k]` when
   all k drafts match; safe only while `target_tokens.size() == k_drafts + 1`.
   Guard it.
4. **D3 layout is self-referential.** The codes-then-scales plane assumption
   inside `text/draft_head` is validated only by the gate's own CPU ref reading
   the same bytes. The math matches production, but do a one-time cross-check
   of the top-k tokens against the production path (`ops::linear` + `proposal_remap_token_ids`)
   or `tests/targets/qwen3_6_27b/test_draft_head.py` before trusting D3's
   oracle as layout-independent. Note the D8 hard-coded targets were *chosen*
   to match the observed drafts — circular, not independent evidence.
5. **Unchecked CUDA calls.** `cudaMalloc`/`cudaMemcpy` results ignored
   throughout; only the kernel sync is checked. A failed alloc currently dies
   as an obscure crash on the next memcpy.

## 4. Nits

- `tests/phase_gate.cu` header comment still reads "M1 … D3 BLOCKED …
  D8–D10 BLOCKED" — stale since `ba28d3cd`.
- `tools/bench/phase_gate.sh` header + `--compare` failure hint list exit
  codes 0–7 only; `--negative` help says "D2, D3, D4, D5, D6, D7" but
  implements D2–D10.
- The gate re-uploads the ~336 MB draft-head codes plane every run; fine at
  ~1.3 s total, just noted in case the chain grows.
- `d1_payload`/`d2p2` artifact layouts are documented only in source comments;
  docs/128 §8.3 wanted the layout per phase in the §3 table column. One line
  each in the tolerance JSON would do.

## 5. Verification evidence (reviewer runs, 2026-09-01)

GPU idle (both devices 15 MiB / 0 %, no server on 8091). Binary from
`ba28d3cd` (`build/ninfer_phase_gate`, built 14:37). Artifact
`/home/intel/models/qwen3_8_27b.ninfer` present.

```
$ ./build/ninfer_phase_gate                     → exit 0, D1–D10 PASS, 1.27 s
$ bash tools/bench/phase_gate.sh --compare      → "baseline: OK (match)", exit 0
$ bash tools/bench/phase_gate.sh --negative     → 9/9 mutations localize, exit 0
$ ./build/ninfer_phase_gate --no-artifact       → D3 BLOCKED, D8/D9/D10 still PASS (see §2.2), exit 0
$ nm libninfer_{artifact,ops,core}.a            → Reader symbols only in artifact (§2.3)
```

Observed clean values match the committed baseline and doc 131's table
(D1 4c9821bd… / 1 byte / 76 ULP; D2 d272364e… / 24 ULP; D3 a7505509… drafts
[20026,35138,40950]; D4 ac48c927… rel 1.51e-7; D5 e91211dc… rel 1.05e-7;
D6 88d8add0… rel 2.53e-7; D7 f2461073… 0 ULP; D8 fba54146…; D9 4ec8b9d0…;
D10 f777926d…). The runs are reproducible; the findings above are about
**what the phases prove**, not whether they run.

## 6. Disposition

| Item | Action | Priority |
|---|---|---|
| §2.1 D4–D7 route coverage | gate the real kernels (unified route via launcher, or §8.1 hooks) | P1 before merge-back |
| §2.2 D8–D10 semantics | demote to SYNTHETIC/BLOCKED in code + rewrite doc 131 claims | P1 before merge-back |
| §2.3 CMake link | add `ninfer_artifact`, prove a CMake build of the target | P1 (CI-blocking) |
| §2.3 M2 wiring | wire `run_ci.sh` fast mode D1–D7 | P2 |
| §3.1 tolerance JSON | reconcile with enforced checks | P2 |
| §3.2 sentinel mismatch | unify `-inf` vs `-1e30` | P2 |
| §3.3 OOB guard | `speculative_accept_kernel` bounds | P3 |
| §3.4 D3 layout cross-check | one-time vs production path | P2 |
| §3.5 CUDA error checks | check allocs/memcpys | P3 |
| §4 nits | comments/docs touch-up | P3 |

M0 + M1 + D3 (D1–D7): **accepted**. D8–D10 as described in doc 131:
**rejected as claimed; re-scope per §2.2.** Merge-back to `wo/kv-uniform`
waits on the three P1 items.
