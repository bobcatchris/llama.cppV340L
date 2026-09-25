# 136 — MTP-pool frontier: ext-gate hunt CLOSED, beLLaMA cross-compare, and a semantics-independent bug

Status: ANALYSIS (agent 2, ci-gate worktree). Consumes docs/135, answers the coordinator's
2026-09-02 directive (pin the ext-gate / derive the invariant / prepare one fix).
Companions: 130 §9.8, 135. Branch `wo/kv-uniform-ci-gate` @ d5de20e7.

## 0. TL;DR

1. **There is no ext-gate in the single-seq path.** `st.extents` is written once at
   `tp2_backend.cpp:750` (`ext = k`) and never updated; `next_extents` is computed by
   `mtp_prepare_next_round` and **dropped on the floor**. The "advances by ext" model in
   docs/135 §1 is not what the code does. This is why three probes couldn't find the gate —
   the thing being looked for doesn't exist.
2. **Proven by exhaustive no-GPU replay**: a faithful transcription of the KVarN bookkeeping
   state machine, replayed over the single-seq round pattern, **cannot** make the pool lag
   `cur_F`: 13,107,200 configs (plen 1..200 × 65,536 accept patterns × 16 rounds) and
   819,200 configs at 400 rounds (crossing many page boundaries) → **0 throws, 0 lags**.
   The pool sits exactly +2 ahead of `cur_F` at all times.
3. **beLLaMA's KVarN makes the hole structurally unrepresentable** — they never derive a
   valid-row count from a write cursor. That settles the semantics ruling by derivation
   rather than by guess (docs/135 §6 path E → the answer is path **D**).
4. **Found a real, semantics-independent defect**: `set_kvarn_batched_active(false)` is not
   exception-safe, and both runners share ONE `TextContext`. After a throwing batched run, a
   later single-seq run appends into **lane** workspaces while rewinding a **different**
   workspace. Append-target ≠ rewind-target ⇒ frontier desync ⇒ "append left a hole".

## 1. Finding 1 — the ext-gate does not exist (CLOSED)

Single-seq MTP loop, `src/runtime/tp2/tp2_backend.cpp::run_tp2_request`:

| line | site | fact |
|------|------|------|
| 745–750 | `int ext = k; memcpyAsync(state->extents.data, &ext, …)` | the ONLY write to `extents` |
| 1659 | `speculative_prepare_verify_inputs(…, st.extents, …)` | read |
| 1809 | `speculative_accept_greedy_drafts(…, st.extents, …)` | read |
| 1926 | `mtp_prepare_next_round(…, st.next_extents, …)` | writes `next_extents` |
| — | **no `next_extents → extents` copy anywhere in the single-seq path** | the clamp is discarded |

Contrast the batched runner, which *does* feed it back (`lane.extents = next_ext[j]`,
line 2849) and gates on it (`vc_h[j] = min(T, extents+1)` 2525; `if (maxw2 == 0) continue`
3036–3044). So the ext machinery is a **batched-path** concept. The single-seq path always
verifies full width `k+1`.

Consequence for docs/135 §4: "the ext-gate's exact code location is STILL UNIDENTIFIED" →
**unidentified because absent.** Stop hunting it. (A1's refinement #1 — "the verify MTP-layer
append DOES write all 4 true positions" — is consistent with this and was the right call.)

## 2. Finding 2 — the round pattern is provably self-consistent (no-GPU)

`tools/ops/pool_frontier_model.h` transcribes, with per-line citations, the host-side state
machine of `src/ops/kvarn/kvarn_workspace.cpp`: `rewind_tail` (298–307, incl. the no-extend
early-return at 305), `discard_partial`, `commit_completed` (484–486), `hydrate_page`
(554–555), `prepare_page_for_append` (559–577), `append_inner` (375–421, incl. the hole
throw at 398), `append` (580–625), `rewind_to_token_count` (316–353, incl. the unconditional
`committed_pages =` at 341 and the stale-tail return at 347). CUDA elided; arithmetic kept.

Drivers:
- `pool_frontier_replay.cpp` — one round pattern, prints the frontier trace.
- `pool_frontier_search.cpp` / `…2.cpp` — exhaustive sweep.
- `pool_frontier_seed.cpp` — the seed-deficit probe.

Results:

```
replay (k=3, plen=14, 60 rounds):  mtp.frontier == cur_F + 2 at every round; no lag, no throw
search (plen 1..200 × 65536 patterns × 16 rounds):  configs=13107200 throws=0 lags=0
search (400 rounds, crosses page boundaries):       configs=  819200 throws=0 lags=0
seed probe (MTP pool seeded D short of cur_F):      D>=1 -> THROW "append left a hole
                                        (pos=8 page=0 slot=8 tail=7 tile_page=0 committed=0)"
```

**This is the important negative result.** The steady-state round pattern is sound. A deficit
must be *injected* — by a wrong seed, a lost page, or a rewind that targets a different
object than the append. Note the seed probe reproduces docs/135's exact error *shape*
(`slot > tail`, same `tile_page`, `committed` unchanged) from a pure frontier offset — so
"the pool lags cur_F by 2" is a **symptom of a desync**, not of span arithmetic.

Also worth recording: the reported deficit is exactly 2 = `(k+1) − (k−1)`, i.e. one round's
worth of `cur_F` advance minus one round's worth of AR-only advance. That number falls out of
§4's split-target mechanism, not out of any accept/ext arithmetic.

## 3. Finding 3 — beLLaMA cross-compare: why they don't have this problem

Source: `github.com/Anbeeld/beellama.cpp` (llama.cpp fork, KVarN), cloned to `/tmp/beellama_cpp`.
Compared **operation-for-operation**, not format-for-format.

### 3.1 The role split (the whole answer)

Our `KvarnLayerWorkspace::tail_count` carries **four** responsibilities at once:

| role | our read site |
|------|---------------|
| (a) write cursor for the contiguity invariant | `kvarn_workspace.cpp:398` `slot > tail_count` → throw |
| (b) number of valid rows the attend may read | `text_context_impl.h:746,802` `tail.tail_count = layer.tail_count` |
| (c) commit trigger | `:382,432,561,618` `tail_count == kG` |
| (d) rollback target | `:305–306` `rewind_tail` |

One monotone integer cannot serve (a) and (b) when speculation makes the write cursor run
ahead of the committed frontier. That conflation **is** the bug class.

beLLaMA's `llama_kv_tail_store` (`src/llama-kv-cache-tail.{h,cpp}`) splits them:

| our concept | theirs | note |
|-------------|--------|------|
| `slot = pos % 64` | `acquire()` → slot from a **free-list + ring cursor** (`:668–683`) | slot is *allocated*, never derived from position |
| `tail_count` as valid set | `sequences[seq]` = list of `{identity, position, insertion_ordinal, slot}` + `slot_used[]` bitmap | validity is an **explicit set** |
| attend window `committed*64+tail` | `build_source_plan(seq, visible)` → `std::vector<int32_t>` slot list → `run_desc` | the kernel **gathers by named slots** (`fattn-tail.cuh:23–33` `desc[6+it]`) |
| `tail_count == 64` → pack | aging out of the exact window into `k_records` | commit keyed on *retirement*, not on a counter hitting a tile size |
| `rewind_tail(keep)` | `seq_rm(seq, p0, p1)` (`:964–981`) — erases entries, releases slots | deletes a **range of positions**, not a counter clamp |

A hole is *unrepresentable* there: the read set is enumerated, so "rows 123,124 are missing"
is simply "the plan doesn't contain them" — no invariant to violate, no abort, and (important)
no silent wrong-data either.

### 3.2 Rollback is declared, queryable, and LOUD

This is the discipline we are missing, and it is independent of storage format:

- **Declared capacity up front.** `cparams.kv_tail_rollback_tokens = params.n_rs_seq`
  (`llama-context.cpp:284`); the arena is sized `n_tokens + rollback_tokens`
  (`llama-kv-cache-tail.cpp:89`), and ≥1 is forced whenever a tail exists
  (`llama-context.cpp:443–447`). The system *knows* how far back speculation must roll.
- **Queryable precondition.** `get_seq_rm_capability()` publishes
  `{full_clear, arbitrary_ranges, suffix_rollback_tokens}` (`llama-kv-cache.cpp:1347–1355`);
  `can_seq_rm()` (`:1358`) is checked *before* the rollback is attempted.
- **Loud refusal.** `seq_rm()` **returns false and logs a warning** when the request exceeds
  the reserve (`:1393–1407`). Our `gqa_kvarn_rewind_tail` instead does
  `if (keep >= tail_count) return;` — a **silent no-op**. That silent clamp is precisely how
  docs/135 §1's deficit becomes *permanent*.

### 3.3 Speculation is transactional, and the speculative region is never quantized

- `begin_batch()` snapshots the entire bookkeeping (`sequences`, `slot_used`,
  `write_cursors`, `degradation`, `recovery_commits`); `finish_batch(success=false)` restores
  it wholesale and sets `LLAMA_KV_TAIL_DEGRADED_PAYLOAD_INVALID`
  (`llama-kv-cache-tail.cpp:582–635`). Rejected drafts are **atomically discarded**, not
  clawed back with a counter.
- The tail is **exact**: `tail_type` is asserted to be F16 or BF16 only
  (`llama-kv-cache.cpp:408–414`). Quantization applies only to the body/records. So the region
  that speculation churns is *never* the region that must survive packing.

That last point is the structural answer to docs/135 §2 regime 3 (the §5.2/§6.3 "silent wrong
data" class): they cannot have it, because speculative bytes are never in the quantized
artifact. We can't adopt the format cheaply, but we can adopt the *rule*: **never let a row
that has not been accepted be reachable by the commit/pack path.**

## 4. Finding 4 — a real, semantics-independent bug: append-target ≠ rewind-target

`state->text` is ONE `TextContext` per rank, created at backend init
(`tp2_backend.cpp:496–537`) with **both** workspace sets bound:
`kvarn_lane_ws_` (batched, `lanes>1`) *and* `kvarn_text_ws_ = kvarn_mtp_ws_ = &kvarn_ws`
(single-seq). Routing is a mutable flag on that shared object:

```cpp
// text_context_impl.h:991 (TP branch, mtp_forward_tail)
if (kvarn_lane_ws_ != nullptr && (active_sequence_batch_ > 1 || kvarn_batched_active_) &&
    batch_mtp_kv_ != nullptr && …)  kvarn_attend_mtp_batched(…);   // -> kvarn_lane_ws_[lane].mtp
else if (kvarn_mtp_ws_ != nullptr && …)  kvarn_attend_mtp(…);      // -> kvarn_mtp_ws_->mtp
```

and the rewind:

```cpp
// text_context.h:285-288
void kvarn_rewind_mtp(int32_t keep) { gqa_kvarn_rewind_to_token_count(*kvarn_mtp_ws_, keep, true); }
//                                                              ^^^^^^^^^^^^^^^ ALWAYS the single-seq ws
```

The batched runner sets the flag at `tp2_backend.cpp:2438` and clears it at **3234 — inside
the `try`, after the decode loop**. The `catch (...)` at **3235 rethrows without clearing**.
`run_tp2_request` never clears it either (only two setters exist in the whole tree). Same
leak for `set_kvarn_batch_lane_ids(nullptr)` at 3233.

⇒ **After any throwing batched run, every later single-seq run in that process appends MTP
KV into `kvarn_lane_ws_[stale_lane].mtp` and rewinds `kvarn_mtp_ws_->mtp`.** The appended
workspace is never rewound; the rewound workspace is never appended. The frontier of the
appended one is whatever the dead batched run left — a different sequence, different prompt
lengths — so `slot > tail_count` fires as soon as the true positions cross it.

This fits every awkward observation:
- explains "the pool lags cur_F by an accumulated deficit" **without** any ext-gate (§2 proves
  the span arithmetic is sound);
- explains why it is **prompt-length dependent** and deterministic (the stale lane state comes
  from the other sequence);
- explains why MTPAPPEND saw all 4 true positions written — it instruments the *append*, and
  the append is correct; the **rewind is aimed at a different object**;
- matches docs/135 §5's own harness gotcha: the SEQ_ONLY process runs the batched phase
  **first** (`tp2_batched_decode.cpp:152`), so the leak is armed before the "reference" run;
- matches "MTPDISPATCH shows the MTP-layer attend always routes to the batched branch in the
  failing run" — which docs/135 §4 recorded as "no skipped dispatch" and moved past. That
  observation was the clue, not a dead end.

**PROVEN vs HYPOTHESIS.** Proven by reading: the shared `TextContext`, the four roles on
`tail_count`, the non-exception-safe reset, the rewind/append target mismatch, §2's exhaustive
negative. NOT yet proven: that this is the trigger for the specific T128_uneven_plen abort
rather than a second, adjacent defect. That needs one GPU run (below).

## 5. The invariant (derived, not guessed) — this IS the ruling docs/135 §6 asked for

> **I1.** There is exactly one authoritative frontier: `cur_F`. Every pool's *valid extent*
> is a pure function of `cur_F`; no pool may derive its valid extent from a write cursor.
>
> **I2.** The write cursor and the valid extent are separate fields. Appends may legally run
> ahead of `cur_F` (that is speculation); reads may never exceed `cur_F`.
>
> **I3.** Rollback moves the valid extent **both ways** and is **loud** when it cannot. A
> rewind that silently no-ops is a defect, not a safety property.
>
> **I4.** A row that has not been accepted is never reachable by the commit/pack path.
>
> **I5.** The workspace an append writes to and the workspace a rewind truncates are the same
> object, structurally — not by a runtime flag that can leak.

I5 is the one that needs no ruling at all. I1–I3 are docs/135 path **D**; beLLaMA's design is
independent evidence that D is the only path that doesn't trade one failure mode for another
(theirs has no analogue of A/B/C, and no analogue of the hole).

## 6. Candidate fix (staged; stage 1 is ruling-independent)

**Stage 1 — make the routing exception-safe (no semantics needed, kills the desync):**
- RAII the batched-runner flag: set `kvarn_batched_active_` / `lane_ids` / `kv_view` through a
  scope guard whose destructor clears them, so the `catch` path cannot leak.
- Belt-and-braces: `run_tp2_request` explicitly clears both at entry (it is the single-seq
  path; it should assert its own preconditions, not inherit them).
- Add a debug assert in `kvarn_attend_mtp*` that the rewind target and append target agree.

**Stage 2 — split the roles (needs the ruling, or is provable-safe):**
- Add `int32_t frontier_tokens` to `KvarnLayerWorkspace`; `tail_count` stays the write cursor.
  The attend reads `min(frontier_tokens, cursor)`; `rewind` sets `frontier_tokens` (both
  directions) and **throws** if it must advance past the cursor.
- Delete the `keep >= tail_count` silent no-op in favour of an explicit, checked update.

**Stage 3 — declare the rollback capacity** (beLLaMA §3.2): size the MTP tile's speculative
reserve from `k` (`k+1` columns/round ⇒ ≥ `k+1` slack), and make an over-capacity rewind a
queryable, logged refusal instead of a clamp.

## 7. Decisive next check (needs GPU — 1 cell, ~2 min)

Cheap, and it separates §4 from "some other desync":

1. Reproduce T128_uneven_plen as today → expect the hole abort.
2. Same cell, but run the reference in a process where the batched phase is **skipped**
   (honour `NINFER_SEQ_ONLY` before the batched run at `tp2_batched_decode.cpp:152`).
   - **Abort disappears** ⇒ §4 is the mechanism; Stage 1 is the fix; docs/135 §4's
     "single-seq-path bug" claim needs correcting to "shared-runner state leak".
   - **Abort persists** ⇒ §4 is real but adjacent; then dump `kvarn_mtp_ws_->mtp` vs
     `kvarn_lane_ws_[b].mtp` `tail_count`/`tile_page`/`committed_pages` at the throw to find
     the other injector.
3. Either way, print both workspaces' triples at the throw — one line each, no new probe
   framework needed.

## 8. Files

- `tools/ops/pool_frontier_model.h` — transcription (cited per site).
- `tools/ops/pool_frontier_replay.cpp` / `pool_frontier_search.cpp` / `pool_frontier_search2.cpp`
  / `pool_frontier_seed.cpp` — drivers.
- Build: `g++ -O2 -std=c++17 -o /tmp/x tools/ops/pool_frontier_search.cpp && /tmp/x`
  (no CUDA, no GPU; ~40–60 s for the full sweep).

---

## 9. Coordinator Q: is the BATCHED KVarN MTP path exposed too? (answers to seq 23)

### 9.1 First: the C1 evidence does NOT exonerate the batched path

`results/20260902_164208_mtp_t2.log` says
`actual=rc=-6 [reference run failed T128_uneven_plen]`. That label is **not** an attribution.
The process the harness calls "the reference" is `NINFER_SEQ_ONLY`, and that same process runs
the **batched** phase first:

| `tests/multi_gpu/tp2_batched_decode.cpp` | line |
|---|---|
| `run_tp2_requests_batched(...)` | **152** |
| `if (const char* only = getenv("NINFER_SEQ_ONLY"))` → `run_single(...)` | 171 |

So `rc=-6` means "the batched phase **or** the seq phase aborted" — indistinguishable, because
`t2_golden_matrix.py` captured `err_ref` and **never printed it**, and the abort message is
exactly what carries the discriminator (`pool=mtp pos=… tail=…`, and whether
`[batched] 2 lanes finished` was reached). docs/135 §5 already flagged this mislabel; the log
shows it biting in the other direction too. **Fixed**: the harness now dumps the last 25
stderr lines on both the reference and batched failure paths, so the next CI run attributes
the abort by itself. No product code touched.

### 9.2 Q2 — shared vs distinct pipeline: **shared positions, DIFFERENT rewind targeting**

| aspect | single-seq (`run_tp2_request`) | batched (`run_tp2_requests_batched`) | shared? |
|---|---|---|---|
| verify positions | device `verify_pos = base+j`, all k+1 true (`speculative_round.cuh:47`) | host `host_pos[j*T+t] = F_h[j]+t` (cpp:2527), all k+1 true | **YES — identical convention** |
| alignment append span | `mtp_forward_decode_batch(…, verify_pos, …)` cpp:1931 | `mtp_forward_decode_batch(…, vpos, …)` cpp:2996-3001 | **YES** |
| `lane.extents` feedback | **never fed back** — `extents` stays `k` forever (§1) | fed back: `lane.extents = next_ext[j]` cpp:2849 | NO — batched-only |
| AR step skip | none — always `k-1` appends | `if (maxw2 == 0) continue` cpp:3030 | NO — batched-only |
| **append target** | flag-selected: `kvarn_lane_ws_[lane].mtp` **or** `kvarn_mtp_ws_->mtp` (`text_context_impl.h:991-1002`) | always `kvarn_lane_ws_[lane].mtp` | — |
| **rewind target** | **always** `kvarn_mtp_ws_->mtp` (`text_context.h:285-288`) | `kvarn_rewind_lane(b, …)` → `kvarn_lane_ws_[b]` (cpp:2908-2909) | — |
| append target == rewind target? | **NOT GUARANTEED** — depends on a leakable flag | **YES, structurally** | — |

That last row is the whole answer. The batched path's append and rewind are pinned to the same
lane workspace by construction. The single-seq path's rewind target is a fixed field while its
append target is chosen by a mutable flag — so only the single-seq path can have them disagree.

### 9.3 Q1 — can a batched config hit the hole from span accounting? **No (proven).**

`tools/ops/pool_frontier_batched.cpp` replays the batched pattern through the same faithful
transcription — two lanes, uneven prompt lengths (pl0,pl1 ∈ 1..60), 1024 accept patterns,
60 rounds, **with** the batched path's real `next_extents` budget feedback and the
`maxw2==0` global AR skip:

```
BATCHED pattern: configs=7372800 throws=0 lags=0
```

Combined with §2's single-seq result (13.1M + 819K configs, 0/0): **neither path's span
accounting can produce the deficit.** The "ext vs a+1" framing in docs/135 §1 is a dead end
for both paths. The batched path's remaining exposure is not span accounting but the
shared-object hazards (§4) and the stale-lane hazards docs/129/130 already documented.

So: **batched cannot hit this bug via the mechanism docs/135 describes; single-seq additionally
cannot hit it via that mechanism either** — which is why the mechanism is wrong, not why the
batched path is lucky.

### 9.4 Q3 — is the batched T128_uneven cell genuinely unvalidated? **Yes.**

`t2_golden_matrix.py` does `fails += 1; continue` on reference failure — so for
`T128_uneven_plen` the batched run is **never even launched**, and its tokens are compared
against nothing. The cell is not "batched passed", it is **"batched unmeasured"**. Of the 6
T2.1 cells, 5 have an oracle and 1 has none. The suite reports `1 errors`, which understates
it: one golden comparison is missing, not just one reference run.

Consequences for the verdict:
- The batched path on uneven prompt lengths is **unproven**, and uneven prompt length is
  precisely the geometry that stresses per-lane frontier divergence.
- Restoring the single-seq oracle is therefore not just a crash fix — it is what makes the
  batched claim testable at all. Stage 1 (§6) is the minimum to restore the oracle.
- If the abort turns out to be in the batched phase (§9.1 is unresolved until one GPU run),
  then the batched path **does** hit a hole today and docs/135 §4's "batched-only: disproven"
  needs reopening. The harness fix in §9.1 is what distinguishes these; it costs one CI run.

### 9.5 What I still need the GPU for (one cell, ~2 min)

§9.1's ambiguity is the only thing blocking a clean verdict, and it is not resolvable by
reading. The check in §7 (run the reference with the batched phase skipped, print both
workspace triples at the throw) settles Q1's residual and attributes the abort.

---

## 10. GPU VERDICT (2026-09-02, kv-mtpfix, GPU priority) — the abort is in the BATCHED phase

Repro (fixed binary, 5838fe45 = f6b7cb76 RAII + launcher build fix), exact T2 cell:

```
NINFER_SEQ_ONLY=/tmp/mtpfix_ref ./build/tests/ninfer_tp2_batched_decode_test \
  --artifact /home/intel/models/qwen3_8_27b.ninfer --tokens 128 --mtp 3 \
  --prompt-a "Explain general relativity in simple terms for a high school student." \
  --prompt-b "Hello!"
RC=134 (SIGABRT)
terminate called after throwing an instance of 'std::logic_error'
  what():  gqa_kv_append_kvarn: append left a hole in the tile
           (pool=mtp pos=125 page=1 slot=61 tail=59 tile_page=1 committed=1)
```

**Bit-identical to docs/135's signature — WITH the RAII fix present.** Three conclusions:

1. **The RAII fix does NOT clear this abort.** Finding 4 is a real latent defect (and is worth
   keeping on its own merits), but it is not the mechanism here. In a fresh process nothing had
   leaked yet, so the leak cannot be the trigger.
2. **The abort is inside `run_tp2_requests_batched`, not the single-seq reference.** Proof:
   `[batched] 2 lanes finished` is an unconditional `printf` at
   `tp2_batched_decode.cpp:163`, immediately after the batched call at :152 returns. It is
   ABSENT from stdout (as are `[SEQ-ONLY] dumped`, `MTP-LOSSLESS`, `RESULT: PASS`). The batched
   call therefore never returned.
3. **docs/135 §4's scope claim is overturned**, and with it the reading of the C1 log:
   "the SINGLE-SEQ REFERENCE aborted, NOT the batched decode" is backwards. This is a
   **batched-path** bug that the harness labels "reference run failed" — docs/135 §5's own
   mislabel gotcha, firing in the direction §4 did not consider. §4's "Batched-only: disproven"
   must be reopened.

Model correction needed for the next attempt: §9.3's batched replay modelled the AR step as a
PER-LANE skip. The code skips only GLOBALLY (`if (maxw2 == 0) continue`, cpp:3030) and otherwise
appends **every** lane's column — including lanes whose `ar_valid_columns[s]` is 0. That is the
assumption that made the batched sweep come up clean; it is not faithful. Re-run the sweep with
global-skip-only before proposing any second fix.

## 11. RECORD: a6da3c98 is a MIXED commit (traceability defect, remediation Option A)

`a6da3c98` is titled "docs/136 + tools/ops" but contains **25 files / +4349 lines**. Cause: in
the shared ci-gate worktree I ran `git add <6 explicit paths>`, verified with
`git diff --cached --stat` (correctly showing only my 6), then ran `git commit` with **no
pathspec**; another agent's `git add` interleaved and the commit took the whole index.

Belongs to the **phase-gate / ci-sync** work (NOT this investigation):
`src/ops/kernel/gqa_decode_slice4_kvarn.cuh` (+22/-3 — the dump params that broke the build),
`tools/ops/run_ci.sh` (+77/-3), `tests/phase_gate.cu`, `tests/phase_gate_{artifact_io,artifacts,
pinned_input}.h`, `tests/phase_gate_pinned_input_check.cpp`, `tests/phase_gate_slice4_{compare,
synth_dump}.cpp`, `tests/phase_gate_tolerance_draft.json`, `tests/kvarn_slice4_cpu_ref.{h,cpp}`,
`tests/kvarn_draft_head_cpu_ref.cpp`, `tests/slice4_phase_gate_dump.cu`,
`tools/bench/{phase_gate.sh,build_phase_gate.sh,build_slice4_dump.sh,phase_baseline.json,
phase_tolerance.json}`.

Belongs to **docs/136 + this investigation**: `docs/136_*.md`,
`tools/ops/pool_frontier_{model.h,replay,search,search2,seed}.cpp`.

Consequence recorded honestly: **the HEAD build break is attributable to this commit** (it
landed kernel dump params without the matching launcher callsite). Fixed separately in
`5838fe45`. Team-wide rule adopted: never `git commit` without an explicit pathspec in a shared
worktree.

## 12. Corrected batched sweep (global-skip-only) — STILL CLEAN, and what that rules out

`tools/ops/pool_frontier_batched2.cpp` fixes §9.3's unfaithful per-lane AR skip to the real
global skip (`if (maxw2==0) continue`, cpp:3030) + append EVERY lane's column otherwise:

```
BATCHED pattern (global-skip-only): configs=7372800 throws=0 lags=0
```

So the deficit is NOT reproducible from the batched round-pattern bookkeeping either. Both
transcriptions (single-seq §2, batched §9.3+§12) are sound ⇒ **the injector is a call site my
model does not contain**, not span arithmetic. Ranked candidates, most-likely first:

1. **Per-lane MTP prefill SEEDING** (`tp2_backend.cpp:2250-2292`, "Migrate THIS lane's
   post-prefill tail into its batched-workspace slice NOW" :2235). This is the only
   prompt-length-dependent state establishment in the batched path, and the failing cell is
   precisely the UNEVEN-prompt-length one (T64_uneven passes, T128_uneven fails — more page
   boundaries crossed). §2's `pool_frontier_seed.cpp` already proved a seed deficit of D>=1
   throws with the SAME signature shape. **Check first: is each lane's MTP pool seeded to
   exactly plen, or plen-1 / shifted by `plan_mtp_alignment_window`?**
2. `gqa_kvarn_rewind_to_token_count`'s `tile_page < page` branch (cpp:347) returns leaving
   `tail_count` belonging to an OLDER page while `committed_pages` was already overwritten
   (cpp:341) — the derived frontier then mixes two pages.
3. The device-side tail publish (`lane_tail_counts`, `maxlane = kvarn_batch_` sizing) diverging
   from the host `tail_count` the rewind updates.
4. INV-3 passes at the pre-verify binding, so whatever injects the deficit does so AFTER that
   assert and BEFORE the next verify — bracket those two points per lane per round.

Handoff: model + all drivers in `tools/ops/pool_frontier_*` (CPU-only, no CUDA; build
`g++ -O2 -std=c++17 tools/ops/<file>.cpp -o /tmp/x`). `pool_frontier_model.h` is the cited
transcription — extend it with the seeding path (candidate 1) rather than rewriting.

## 13. GPU INSTRUMENTATION — mechanism PINNED (NINFER_MB_POOLTRACE, commit pending)

One printf per lane per round at the batched rewind site (`tp2_backend.cpp:2908`) ends the hunt.
Lane 1 (the short prompt, still active) on the failing cell:

```
r=47 cur_F=100 | mtp fr= 99 (c=1 t=55->tp=1) | ext=3 a=2   lag 1
r=48 cur_F=104 | mtp fr=102 (c=1 t=38)       | ext=3 a=3   lag 2
r=49 cur_F=108 | mtp fr=106                  | ext=3 a=3   lag 2
r=50 cur_F=112 | mtp fr=110                  | ext=3 a=3   lag 2
r=51 cur_F=113 | mtp fr=113                  | ext=3 a=0   lag 0   <-- partial accept RE-SYNCs
r=52 cur_F=117 | mtp fr=116                  | ext=3 a=3   lag 1
r=53 cur_F=121 | mtp fr=119                  | ext=3 a=3   lag 2
r=54 cur_F=125 | mtp fr=123 (c=1 t=59 tp=1)  | ext=0 a=3   lag 2   <-- EXACT docs/135 signature
r=55 ... aborts: append left a hole (pos=125 page=1 slot=61 tail=59 tile_page=1 committed=1)
```

Lane 0 (finished, `ext=0`) holds `mtp fr == cur_F` EXACTLY at every round — it never lags.

### What this proves

1. **docs/135 §1's model is CORRECT after all — but it lives in the BATCHED path, not
   single-seq.** The MTP pool advances 3 per round while `cur_F` advances 4 on a full accept
   (`a == k`), so the deficit grows **+1 per full-accept round**, and a partial-accept round's
   rewind re-syncs it (r=51). It is the accumulation, not a single event, that kills.
2. **The rewind's no-extend clamp makes it permanent and then fatal.** At r=54 `cur_F=125`
   ⇒ `keep = 125%64 = 61`, `tail = 59`; `gqa_kvarn_rewind_tail` early-returns at
   `kvarn_workspace.cpp:305` because `keep >= tail_count`, so the pool stays at 123. The next
   append at the true position 125 has `slot 61 > tail 59` → hole. **This is exactly the silent
   clamp beLLaMA refuses to have** (§3.2: their `seq_rm` returns false and logs).
3. **Why my replays (§2, §9.3, §12) all came up clean:** they model the AR append as reaching
   `cur_F + k - 2`, which keeps the pool AHEAD. Reality advances it by only 3 on an `a=3` round,
   so the AR span in the real batched path is NARROWER than transcribed. **The unfaithful
   assumption is now identified**: find why the AR/alignment span yields +3 not +4 when `a=k`.
   Prime suspect: the global `if (maxw2 == 0) continue` (cpp:3030) interacting with
   `ar_valid_columns[s] = s+1 < next_ext` — at r=54 `ext` drops to 0 (`ext=0 a=3`), so
   `next_ext=0` ⇒ `ar_valid_columns[s]=0` for all s ⇒ `maxw2==0` ⇒ **the entire AR phase is
   skipped**, so the pool never gets its `cur_F+s` appends for that round.
4. **`ext=0` with `a=3` at r=54 is itself suspicious** — a full accept immediately followed by
   zero extents. That is the budget/context clamp in `mtp_prepare_next_round_kernel`
   (`context_extent = max_context - frontier - 1`). With `max_context=4096` and `cur_F=125` it
   should NOT be 0, so check the `remaining_budgets` input: lane 1's prompt is short so it is
   still generating, and `remaining - 1` vs `budget_extent` may be zeroing it.

### Fix candidate (NOT landed — needs confirmation)

Two independent, both-defensible changes; the first is the safety property, the second the cause:
- **A (invariant I3, ruling-independent):** make `gqa_kvarn_rewind_tail` refuse LOUDLY instead
  of silently no-op'ing when `keep > tail_count`, i.e. when a rewind must EXTEND the pool past
  what was written. That converts a late, misattributed SIGABRT into an immediate, localized
  diagnostic at the true fault point — and matches beLLaMA's declared-capacity discipline.
- **B (the actual deficit):** ensure the MTP pool always covers `[0, cur_F)` after a round,
  i.e. the AR/alignment span must be keyed on `max(a+1, next_ext)` rather than `next_ext` alone,
  so a full-accept round never leaves the pool short of the committed frontier.

Confirm B by re-running the replay with the AR span gated on `next_ext` (not `k-1` steps) — that
should now reproduce the +1/round lag, and then the fix can be validated against it BEFORE any
GPU run.

### 13.1 STRONGEST LEAD for the short append (from the trace's own `ext` column)

The trace prints `lanes[b].extents` AFTER the :2849 update, i.e. `next_ext`. At r=54 lane 1 shows
**`ext=0` with `a=3`**. `next_ext == 0` for every lane makes `any_draft` false, and the batched
loop then does `if (!any_draft) { sync; continue; }` (cpp:2940-2947) — **which skips the entire
draft phase, INCLUDING the alignment forward that advances the MTP pool by F..F+k.**

So on a round where the budget/context clamp zeroes `next_ext`, `cur_F` still advances by `a+1`
(the accept already committed) but the MTP pool gets NO alignment append at all. That is exactly
the "+1 per full-accept round" deficit the trace shows, and it explains why my replays never
reproduced it: they always ran the alignment append.

Corollary: the deficit is not in the append SPAN arithmetic at all — it is that the append is
CONDITIONALLY SKIPPED by a draft-phase gate that is keyed on *future drafting need*, while the
pool frontier is keyed on *committed history*. Those two are different quantities, and conflating
them is the bug. This is invariant I1/I2 (docs/136 §5) violated in a place none of the earlier
probes looked.

NEXT RUN (5 min, one more printf): log whether the draft phase was entered per round
(`any_draft`, `next_ext`, `cur_F`, `a`) alongside the existing `[POOL]` line. If every
lag-increasing round coincides with `any_draft == false`, the mechanism is CONFIRMED and fix B
becomes: run the alignment append whenever `a+1 > 0` (committed history advanced), regardless of
`any_draft` — i.e. decouple "the pool must cover committed history" from "we need more drafts".

### 13.2 CORRECTION — §13.1's `any_draft` lead is DISCONFIRMED by the trace's own data

§13.1 is wrong and I am retracting it. The `ext` column printed at each round is `next_ext`
(post-:2849). The lag INCREASES at r=48, r=49, r=50, r=52, r=53 — and at every one of those
rounds `ext == 3`, so `any_draft` was TRUE and the draft phase (alignment included) DID run.
Only r=54 has `ext=0`. So the skipped-draft-phase gate is not the mechanism.

What the numbers actually say: taking r=47→r=48 (F=100, a=3, cur_F_new=104), the pool reached
102 = **F+2**, i.e. the MTP pool advanced only TWO columns from F even though the alignment
forward appends `width = T = 4` columns at `vpos = F..F+3`. Same at r=53→r=54 (F=121 → 123 = F+2).
A consistent +2, not a skipped phase.

So the open question is now narrow and concrete: **why does the batched MTP append cover only
2 of the 4 verify columns?** Two candidates, both checkable in one more instrumented run:
 1. The append is receiving a `vpos` whose columns 2..3 are NOT F+2/F+3 (non-contiguous ⇒
    `append_inner` breaks the run at `p2 != pos + run`, cpp:401, and the later columns then
    hit the hole check instead of extending the tail).
 2. `width` reaching `gqa_kv_append_kvarn_batched` is 2, not 4, for the MTP layer specifically
    (`k.ne[2]`, cpp:191) — i.e. the MTP alignment tensor is narrower than the text one.

Candidate 1 is the more likely of the two and is the SAME failure family as the a13ae028 flake
(positions in the device tensor disagreeing with the host bookkeeping), which makes it worth
checking first. Print `vpos[0..T-1]` for the lagging lane plus `k.ne[2]` at the MTP append site.

Lesson for whoever picks this up: do not re-derive from span models — three exhaustive replays
(§2, §9.3, §12) already prove the round-pattern bookkeeping is sound, so the answer is in the
DATA reaching the append, not in the arithmetic over it.

### 13.3 APPEND DATA IS CORRECT — §13.2's two candidates both DISCONFIRMED (GPU, MTPAPPEND trace)

`NINFER_MB_POOLTRACE` now also prints `width` + the lane's positions at the batched MTP append.
Lane 1, contiguous run (each line duplicated by the two rank threads):

```
[MTPAPPEND] lane=1 width=4 pos=[90 91 92 93] tail=28 committed=1 tile_page=1   <- alignment F=90
[MTPAPPEND] lane=1 width=1 pos=[94]          tail=30                            <- AR s=0
[MTPAPPEND] lane=1 width=1 pos=[95]          tail=31                            <- AR s=1
[MTPAPPEND] lane=1 width=4 pos=[94 95 96 97] tail=31                            <- alignment F=94
[MTPAPPEND] lane=1 width=1 pos=[95]          tail=34                            <- AR s=0
[MTPAPPEND] lane=1 width=1 pos=[96]          tail=34                            <- AR s=1
```

Everything is right: `width=4` for the alignment (not 2), positions fully contiguous
(`F..F+3`), AR at `cur_F+s`, and the tail advances correctly (alignment at F=94 → tail 34 ⇒
frontier 98 = F+4). So:

- **Candidate 1 (non-contiguous `vpos`) — DISCONFIRMED.** Positions are contiguous.
- **Candidate 2 (`width` == 2) — DISCONFIRMED.** `width=4`.
- **§13.1 (`any_draft` skip) — already retracted in §13.2.**

This is the important narrowing: **the data reaching the append is correct, and the append
advances the host-side frontier correctly to F+4.** Yet the `[POOL]` line at the rewind reports
the frontier 2 BEHIND `cur_F` on the very next round. The deficit therefore appears BETWEEN the
append and the next round's rewind — i.e. in what the rewind reads, not what the append wrote.

That leaves §12's candidate 3 as the live hypothesis, and it is now the ONLY one consistent with
all the evidence: **the rewind and/or the attend read a DIFFERENT frontier than the one the
append updated.** Concretely, `kvarn_rewind_lane(b, …)` mutates `kvarn_lane_ws_[b].mtp`, while
the attend publishes `kvarn_mtp_tail_dev_` from the same struct — but `gqa_kvarn_rewind_to_token_count`
has the `tile_page < page` branch (`kvarn_workspace.cpp:347`) that returns leaving `tail_count`
belonging to an OLDER page after `committed_pages` was already overwritten unconditionally at
`:341`. That mixes two pages into one derived frontier — and the failing lane is exactly the one
whose `committed` stays at 1 while `cur_F` crosses toward 128.

NEXT STEP (no span model needed — read the two fields, don't re-derive): print
`committed_pages / tail_count / tile_page` for the lagging lane at BOTH sides of the rewind call
(`tp2_backend.cpp:2909`) for one round where lag increases. If `tile_page < page` at the rewind,
candidate 3 is CONFIRMED and the fix is in `gqa_kvarn_rewind_to_token_count` (do not overwrite
`committed_pages` before the tile is retired), not in the MTP pipeline at all.

Session summary of what was eliminated, so nobody re-walks it: span keying on ext vs a+1 (§1,
13.1M configs), batched span arithmetic (§9.3, §12, 7.37M configs), the routing-flag leak as the
cause (f6b7cb76 present, identical abort, fresh process), `any_draft` skip (§13.2), non-contiguous
positions and narrow width (§13.3). The append is correct; the deficit is in the rewind's read of
the frontier.

### 13.4 SELF-CORRECTION: candidate 3 is ALSO contradicted — and the constraint is now tight

§13.3's "next step" pointed at `rewind_to_token_count`'s `tile_page < page` branch. That cannot be
firing: the abort state is `committed=1 tile_page=1` with `cur_F=125` ⇒ `page = 125/64 = 1`, so
`tile_page == page`, not `<`. The branch is not taken.

And the arithmetic now closes the net hard. Given the observed order (verify → accept → **rewind**
→ alignment → AR) and the trace showing the alignment correctly advancing `tail` by 4 from `F`:

- the rewind clamps `tail` to `slot = cur_F % 64`, so after it `frontier == cur_F` exactly;
- the alignment then appends `F = cur_F .. F+3`, giving `frontier = cur_F + 4`;
- next round's rewind clamps to `cur_F' = F + a + 1 ≤ F + 4` — a clamp, never below.

So `frontier >= cur_F` is forced by the observed behaviour, yet the trace measures `cur_F - 2`.
**One of those two premises is false, and it must be resolved before any fix is written.** The
remaining suspects, in order:
 1. The rewind is NOT clamping to `cur_F` as assumed — check `kvarn_rewind_lane`'s `keep` argument
    against the `cur_F` the trace prints (they are read at different points in the loop).
 2. The alignment append that the MTPAPPEND trace shows advancing the tail is being UNDONE by a
    later `discard_partial` (page crossing) before the next rewind.
 3. The `[POOL]` line and the `[MTPAPPEND]` line are reading different lane indices — the trace
    prints `lane` from two different loops (`b` vs `live[j]`), and docs/129/130 already have a
    documented history of `b != j` after a shrink. **Check this first — it is the cheapest and
    would make the whole "lag" an artifact of comparing lane 0's rewind against lane 1's append.**

Print, for one lag-increasing round, the rewind's `(lane, keep, tail_before, tail_after)` next to
the append's `(lane, pos[0], tail_after)`. If the lane indices disagree, §13's deficit is a
measurement artifact and the real bug is elsewhere.

### 13.5 §13.3 IS VOID — my MTPAPPEND conclusion was drawn from the wrong rounds (proved by reading)

The two traces are indexed DIFFERENTLY:

| trace | loop | index means |
|---|---|---|
| `[POOL]` (`tp2_backend.cpp`, inside `for j`, `const std::size_t b = live[j]`) | `b` | **true LANE id** |
| `[MTPAPPEND]` (`text_context_impl.h:786`, `for (int b2 = 0; b2 < batch; ++b2)`) | `b2` | **batch COLUMN index** |

These coincide only while every lane is live (`live[j] == j`). The `[POOL]` trace shows lane 0
stops appearing after r=45 — it finished, so `B` shrank to 1 and `live[0] == 1`. From r≥46 there
IS no column 1.

Therefore the `[MTPAPPEND] lane=1` lines quoted in §13.3 (`pos=[90..97]`, `width=4`, contiguous)
come from the **pre-shrink B=2 rounds**, not from the lag-growing rounds r=47–54. §13.3's
disconfirmation of candidates 1 (non-contiguous `vpos`) and 2 (narrow `width`) is **unsupported
for the phase that matters and is hereby withdrawn.**

Corrected state: candidates 1 and 2 are BACK IN PLAY, and they are now the leading hypotheses —
because a shrink is precisely the event that makes column index ≠ lane index, and the deficit
starts accumulating right after r=45/46.

To test properly, print the column's LANE ID alongside it — `kvarn_batch_lane_ids_[b2]` (or
`b2` when null) — so the two traces can be joined on the same key, and read the lines for
r ≥ 46 only.

Meta-lesson for the next session, from this session's record: three of four leads were wrong and
one "elimination" was an artifact of my own instrumentation's indexing. Trust a negative result
only when the trace key provably matches the phenomenon's key.

## 14. CONFIRMED MECHANISM — append and rewind observe DIFFERENT frontier for the SAME lane

With the trace joined on the true lane id (`col=0 lane=1`), the two sites are interleaved for the
same lane 1 at the same moment:

```
[MTPAPPEND] col=0 lane=1 width=4 pos=[125 126 127 128] tail=16 committed=2 tile_page=2   <- APPEND sees fr=144
[POOL]      r=54 lane=1 cur_F=125  mtp fr=123 (c=1 t=59 tp=1)                            <- REWIND sees fr=123
```

`committed=2, tile_page=2, tail=16` ⇒ frontier **144**.
`committed=1, tile_page=1, tail=59` ⇒ frontier **123**.

**These cannot be the same `KvarnLayerWorkspace`.** The batched MTP append is writing/reading one
workspace while `kvarn_rewind_lane()` truncates another — for the same lane index. So:

- The deficit is NOT span arithmetic (settled by §2/§9.3/§12 replays).
- It is NOT the leaked `kvarn_batched_active_` flag (§10: identical abort, fresh process).
- It IS an **append-target ≠ rewind-target** mismatch — the same *class* as §4, but structural in
  the batched path rather than caused by the exception-path leak.

Where to look next (concrete, no more tracing needed): the append reaches
`kvarn_lane_ws_[kvarn_batch_lane_ids_[col]]` (`text_context_impl.h:778-795`), while the rewind
goes `kvarn_rewind_lane(b, …)` → `kvarn_lane_ws_[b]` (`text_context.h:274-277`) with `b = live[j]`.
If `kvarn_batch_lane_ids_` at that moment is NOT `live[j]` — e.g. it still points at the AR step's
`ph`-era array, or at `rows_h` from a previous iteration (both are per-step locals, §9) — the two
indices name different lanes. The `committed=2` vs `committed=1` split is exactly what a
lane-index mismatch after the shrink (B=2 → B=1, `live[0]=1`) produces.

**Fix direction (confirm before landing):** make the batched MTP append and the batched MTP rewind
resolve the lane by ONE shared, explicit accessor instead of two independently-derived indices, and
have `kvarn_rewind_lane`'s silent `return` on out-of-range lane (text_context.h:275) fail loudly —
it is the same silent-refusal anti-pattern beLLaMA avoids (§3.2), and it would have surfaced this
immediately.

Note the through-line: §4's leak, §13.5's index mismatch, and this all reduce to the same design
fault — **the lane/workspace identity is recomputed independently at each call site instead of
being resolved once and carried.** That is the thing to fix, and it is independent of the MTP-pool
semantics ruling docs/135 §6 asked for.

---

## 15. HANDOFF (coordinator, 2026-09-02) — repro, verify, DONE criteria for the fix owner

**Entry point: §14 (confirmed mechanism) + §6 (staged fix) + this §15.** Do NOT re-walk the
eliminations (§2 span-arithmetic, §9.3/§12 batched span, §10 routing-flag leak, §13.1/13.2 any_draft,
§13.3/13.5 vpos-index, §13.4 rewind tile_page<page).

**REPRO (GPU, the exact failing cell):**
```
bash tools/bench/run_t2.sh            # builds ninfer_tp2_batched_decode_test + runs t2_golden_matrix.py
```
`t2_golden_matrix.py` (tools/bench) T2.1 golden matrix includes `T128_uneven_plen` (config line 49).
Current behaviour: that cell ABORTS with `append left a hole (pool=mtp pos=125 page=1 slot=61 tail=59
tile_page=1 committed=1)`; the T2.1 run reports `rc=-6 [reference run failed T128_uneven_plen]`.

**THE FIX (2 parts, both ruling-independent — docs/135 §6 semantics is NOT a blocker):**
1. Resolve the lane/workspace identity ONCE and carry it to BOTH the batched MTP append
   (`text_context_impl.h:778-795`, currently `kvarn_lane_ws_[kvarn_batch_lane_ids_[col]]`) and the
   batched MTP rewind (`tp2_backend.cpp` `kvarn_rewind_lane(b, …)` with `b=live[j]`), so they name
   the same lane. Root fault: identity recomputed independently at each call site (see §14 note).
2. Make `kvarn_rewind_lane` (text_context.h:317) fail LOUDLY on out-of-range lane (`lane<0 ||
   lane>=kvarn_batch_`) instead of silently `return` — `fprintf(stderr,…); abort();` (it's `noexcept`,
   so abort, not throw). Defense-in-depth; would have surfaced the mismatch immediately.

**VERIFY (all must hold — one full T2 run):**
- `bash tools/bench/run_t2.sh` T2.1 golden matrix: ALL cells MATCH (exact tokens both lanes), NO
  abort. This is the batched==seq_mtp oracle (the batched decode matches the isolated single-seq
  reference `seq_mtp_lane{0,1}.txt`).
- The full Tier-0..Tier-3 suite stays green (no regression). batched==seq_mtp invariant 6/6.
- Decode (non-batched) side byte-identical — do NOT touch the packed T≤6 decode kernels.

**DONE = fix landed (branch e.g. wo-kvarn-lane-fix) + T2.1 golden matrix all MATCH + no abort +
CI suite green + decode untouched (byte-identical).** Report the commit + the before/after T2.1 result.

## 16. FIX LANDED — §15's fix #1 was aimed at a measurement artifact; the real injector is §13.1

**Status: FIXED + VERIFIED (agent 2, `wo-kvarn-lane-fix`).** §13.1 was right; §13.2's retraction
of it was wrong; §14's "confirmed mechanism" is void. Read this before re-opening any of them.

### 16.1 The mechanism (GPU-proved, one line of trace)

`if (!any_draft) { sync; continue; }` (was `tp2_backend.cpp:2979`) skipped the **entire** draft
phase on a round where every live lane had `next_ext == 0`. The alignment forward inside that
phase is the **only** thing that appends MTP-layer KV for the just-committed span `[F, F+T)`.
`cur_F` advances by `a+1` on **every** round regardless. So the round that needs no drafts is
exactly the round that leaves the pool short, and the **next drafting round** appends past the
tile tail → `append left a hole in the tile (pool=mtp …)`.

Lane 1, failing cell, `NINFER_MB_POOLTRACE` — the `fr` column is the proof:

```
r=53 cur_F=121 | mtp fr=119 (c=1 t=55) | ext=3 a=3
r=54 cur_F=125 | mtp fr=123 (c=1 t=59) | ext=0 a=3   <-- any_draft FALSE -> alignment SKIPPED
r=55 cur_F=126 | mtp fr=123 (c=1 t=59) | ext=1 a=0   <-- fr BIT-IDENTICAL: no append happened
   -> ABORT at r=55's alignment append: vpos=125..128, slot 61 > tail 59
```

`committed/tail/tile_page` unchanged between r=54 and r=55 is direct evidence that no MTP append
ran at r=54 — not an inference over span arithmetic.

### 16.2 Why §13.2's retraction was wrong (the trap for the next reader)

The `[POOL]` line prints **after** the rewind but **before** that same round's alignment. So a
pool reading of `cur_F − max(0, a−1)` is the *benign steady state*, not a deficit — the alignment
that runs later in the round is what restores coverage. Measured, and it fits `lag = max(0,a−1)`
exactly: `a=1→lag 0` (r=46), `a=2→lag 1` (r=47), `a=3→lag 2` (r=48,49,50,53,54), `a=0→lag 0`
(r=51). §13 read that **oscillation as accumulation** and concluded the deficit grew +1/round;
§13.2 then "disconfirmed" §13.1 because the lag-increasing rounds had `ext==3`. But the fatal
event is not the oscillation, it is the **single** `ext=0` round where the append never happens.
The deficit is bounded at 2 and harmless *until* the skip; the skip makes it permanent and then
fatal.

### 16.3 Why §14 is void — the trace lied, one level deeper than §13.5

§13.5 correctly withdrew the `lane`-vs-`col` confusion for the **positions**, but the same bug was
still in the **workspace read**: `[MTPAPPEND]` printed `lane=kvarn_batch_lane_ids_[b2]` while
dumping `kvarn_lane_ws_[b2]` — the column, not the lane (`text_context_impl.h:793`). Post-shrink
`col=0 → lane=1`, so §14's "APPEND sees fr=144 (`committed=2 tail=16`)" was **lane 0's frozen
tile** (lane 0 finished near 144) wearing a `lane=1` label, compared against lane 1's rewind
(`fr=123`). Two different lanes, presented as one. §14's "these cannot be the same
`KvarnLayerWorkspace`" is an artifact of that mislabel.

The append was never mis-targeted: `gqa_kv_append_kvarn_batched` keys `ws`/block-table/packed on
`lane_ids[b]` (`kvarn_workspace.cpp:197,237`), and the rewind keys on `b = live[j]` — the **same**
lane id. §15's fix #1 ("resolve lane identity once and carry it") therefore had no defect to fix;
landing it would have churned the hot path and still aborted. **The trace is now fixed to read
`ws[lane]`**, so this cannot be re-derived from it again.

### 16.4 What landed

1. **The gate is deleted** (`tp2_backend.cpp`, step 5b). The draft phase now always runs, so the
   alignment forward always discharges the pool-coverage obligation. This is the batched-only
   divergence from single-seq: `run_tp2_request` has no such gate and runs the alignment
   unconditionally (`:1926–1945`), so removing it also restores batched == single-seq parity —
   which is precisely what the T2 oracle asserts. Cost is one wasted MTP forward on the rare
   terminal `ext=0` round. The AR-step global skip (`maxw2 == 0`) **stays**: AR appends land at
   `cur_F + s`, i.e. strictly *ahead* of the frontier, so they are pure speculation and skipping
   them cannot starve coverage. The all-extents-zero round is still not "all done" — the
   loop-top all-done check remains the only break, so the §13.1-era off-by-one (dropping the last
   lane's final token) does not come back.
2. **§15 fix #2 landed as written**: `kvarn_rewind_lane` now `fprintf`+`abort()`s on an
   out-of-range lane instead of silently returning (`text_context.h`). Defense-in-depth — it did
   not fire in the verified run, which is itself confirmation that no identity mismatch exists.
3. **The `[MTPAPPEND]` trace reads `ws[lane]`** (§16.3).

### 16.5 DO NOT land §6 "fix A" / Stage 2's loud rewind — the silent clamp is load-bearing

`gqa_kvarn_rewind_tail`'s `if (keep >= tail_count) return;` (`kvarn_workspace.cpp:305`) is **not**
the anti-pattern here. It fires on *every* full-accept round by design, because at the rewind
point the pool is legitimately `max(0,a−1)` behind `cur_F` and is extended later in the same round
by the alignment (§16.2). Making it throw converts correct rounds into aborts and takes the whole
suite down. beLLaMA's loud-refusal discipline (§3.2) is the right *goal*, but it maps onto the
**lane-index** case (fix #2, landed), not onto this clamp. Invariant I3 should be read that way.

### 16.6 Verification

Exact T128_uneven_plen cell, same command as §10:
```
BEFORE: RC=134  gqa_kv_append_kvarn: append left a hole in the tile
              (pool=mtp pos=125 page=1 slot=61 tail=59 tile_page=1 committed=1)
AFTER : RC=0    [batched] 2 lanes finished in 4.7s, 256 tokens total / RESULT: PASS
        r=55 now reads mtp fr=125 (c=1 t=61) — was 123 — coverage restored at the skip round
```
`bash tools/bench/run_t2.sh` → **PASS: Tier 2 End-to-End Suite CLEAN**, T2.1 **6/6 MATCH**
(T64/T128 × both_full/b1_shrink/uneven_plen), so §9.4's "batched unmeasured" cell now has an
oracle and passes. T2.2 flake battery 4/4 PASS, T2.4 soak 4/4 PASS, T2.5 canaries 3/3 still trip
(the net is not weakened). No CUDA/kernel file touched → packed T≤6 decode byte-identical by
construction.

Meta-lesson, third time in this doc: **a trace's key must provably match the phenomenon's key.**
§13.3/§13.5 tripped on it for positions; §14 tripped on it for the workspace handle. Both times
the instrumentation was read as if it shared the caller's index resolution. Fix the probe before
trusting the elimination.

### 16.7 Post-fix CI + a pre-existing build break found on the way

`bash tools/ops/run_ci.sh` on the fix → **Verdict: PASS** (Tier 0 geometry battery, slice4
in-kernel phase gate D1–D7, T4 gate, unit battery all green).

The first CI attempt did NOT get that far, and the blocker was **not** this fix:
`ninfer_slice4_kvarn_test` and `ninfer_slice4_kvarn_bench` failed to compile with
`too few arguments in function call`. That is §11's traceability defect still biting — `a6da3c98`
landed the `dump_scores/dump_probs/dump_q` kernel params, and `5838fe45` ("pass the slice4 dump
args explicitly") fixed **only** `src/ops/launcher/gqa_attention_kvarn.cu`, missing five callsites
in the two test files. Confirmed unrelated to this work: neither TU includes `text_context.h` or
`tp2_backend.cpp`, and the branch did not build before this commit either. Fixed in `25e3a3bc`.

Worth recording for whoever hits this next: **default arguments attach to the function declarator,
not to the pointer type.** `auto* kern = kernel<...>` (or a C-style cast of it) therefore loses
them, and every parameter must be passed at the callsite. That is why a defaulted trailing param
is not a source-compatible ABI for kernel launches taken through a pointer.

Also raised the decode-guard ratchet from `ITERS=1` to `ITERS=2` (`725ce382`): the CI wrapper was
overriding `decode_guard.sh`'s own default, and a single sample per cell cannot separate a real
regression from noise — which is the entire purpose of a monotonic baseline ratchet.

Final state: `0257878c` (fix) + `25e3a3bc` (build) + `725ce382` (ratchet) on `wo-kvarn-lane-fix`,
T2.1 6/6 MATCH, CI PASS, no CUDA/kernel file touched by the fix itself.

### 16.8 `run_ci.sh --full`: every stage passed; the trailing "CI: FAIL" was a link-path bug

`--full` on `de754490` printed `CI: FAIL ✗ (verify=0, serve=0, correctness=0, unit=0, mtp_t0=0,
phase_gate=0, full_extra=0)` while the results JSON for the same run said `_verdict: PASS,
_fails: []`. Both were right, and the discrepancy is worth recording because it will re-appear:

* every counter printed `0` because they are **exit codes** (0 = pass) — but `EXTRA_OK` is the
  odd one out: it is a **flag** where `1` = ok, and the final test is `[ $EXTRA_OK -eq 1 ]`
  (`run_ci.sh:545`). So `full_extra=0` reads as "zero failures" to a skimming eye and actually
  means "the flag was cleared".
* the single stage that cleared it was the D1–D10 phase gate, which **failed to link**, not to
  pass: `ld: cannot find .../worktrees/wo-kv-uniform/build/src/libninfer_{ops,core,artifact,nvfp4_tma}.a`.
  `build_phase_gate.sh:13` / `phase_gate.sh:19` defaulted `KLIBS` to a **sibling worktree's** build
  dir (a historical disk-saving shortcut), so this tree's gate silently depended on another
  worktree's artifacts — and that dir had been cleaned for disk. Nothing to do with the MTP fix.

Fixed in `d1a0d80f` (resolve `KLIBS` to this tree, sibling only as fallback). Verified by running
the failing stage **directly** rather than re-paying for the whole battery:
`build_phase_gate.sh` links, `phase_gate.sh --compare` rc=0, `phase_gate.sh --negative` rc=0 with
D2–D7 mutations all localized. Everything else in `--full` had already passed on its own line:
int8 full (0 failed), KVarN battery (16 passed/0 failed), T14 (0 failed), unified-kernel harness
RESULT: PASS, decode-guard `GATE: PASS`, T1/T2/T3 `TIER_EXIT=0`.

Lesson, consistent with §9.1's mislabel gotcha: a harness verdict line that disagrees with the
per-stage evidence is itself a defect. Read the stage logs, not the banner.

Resolved on `wo-kvarn-lane-fix`: the convention bug was closed, not just the KLIBS break.
`EXTRA_OK` is normalized to the SAME exit-code convention as the other verdict counters
(0 = pass, non-zero = fail) in `tools/ops/run_ci.sh`, so a clean ``--full`` run now prints
`full_extra=0` (= pass) consistently with `verify=0`/`serve=0`/`correctness=0`/etc. A skimming
eye can no longer read a genuine FAIL as "zero failures", and the banner cannot disagree with the
per-stage exit codes. (The dispatch around ``$FULL`` is unchanged; every ``--full`` failure path sets
`EXTRA_OK=1` and the final test is ``[ $EXTRA_OK -eq 0 ]``.)
