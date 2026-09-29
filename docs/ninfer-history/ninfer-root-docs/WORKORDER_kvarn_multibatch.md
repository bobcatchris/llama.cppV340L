# WORK ORDER — KVarN multibatch: wire k5v4/k4v4 (+int8/bf16 matrix) to batched decode

**Status:** READY TO START · **Owner (mainline + GPU):** A1 · **Tests:** gemini ·
**Helper packages (CPU-only):** defined in §3.3, assignable to a fresh agent ·
**Doc number:** TBD — coordinator assigns (filename deliberately non-numbered per AGENTS.md).
**This document is also the lane's RUNNING LOG** (§9) — append entries, do not spawn new docs.

---

## 0. RESUME HERE (fresh session: read this section only, then §4)

**State at writing:** main `e002548d`, pushed. k4v4 + k5v4 serve by default (refusals lifted);
q4_0 still refused (real prefill defect, separate). GPU: full keys, no lease paperwork — note
usage in reports. Merge autonomy: granted. Sole-agent status may change; A2 may return on DFlash2
(do not touch `src/ops/dflash2/**` or the drafter-budget seam — see §6).

**Setup (fresh session):**
```bash
cd /home/intel/ninfer
git -C repo worktree add ../worktrees/wo-kvarn-multibatch -b wo/kvarn-multibatch main
cd worktrees/wo-kvarn-multibatch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --target ninfer_engine ninfer-serve ninfer_ops -j6
```
Never bare `cmake --build build` (rule 9). Default `/usr/local/cuda` is 12.9 — the flag above is
mandatory. **GPU: no lease paperwork needed (user grant); note usage in reports. Kill only PIDs
you started — NEVER `pkill -f <pattern>` (it matches your own shell; done twice this session).**

**Next step:** §5 Step 1 (widen the tp_engine batched gate). Everything up to here is done.

## 1. Context (60-second version)

The three KVarN tiers (k4v2, k5v4, k4v4) are servable single-sequence but **batched decode is
k4v2-only**: `tp_engine.cpp:184` hardcodes `kvarn == KvarnK4V2`, and the launcher throws on
`(k5v4||k4v4) && multi`. Today k5v4/k4v4 under `--max-concurrency>1` run through the per-lane
fallback — serialized, no batched throughput (Step 0 confirms exact behavior).

Why do it now (two independent reasons):
1. **Concurrency:** batched decode is the throughput path; the per-lane fallback forgoes it.
2. **Correctness (docs/132 §4.5):** KVarN single-seq MTP is **not bit-lossless** vs plain decode
   (verify-width T=k+1 kernel numerics vs decode-width T=1 → near-tie greedy flips; root-caused
   2026-09-04; user-accepted as documented attribute, "no kernel bit-identity fix work"). In that
   same investigation **batched MTP was BIT-EXACT vs plain — 48/48** on kvarn_k4v2, and
   batched-vs-plain was ruled "the real lossless contract". So multibatch is not just throughput:
   it is the path that makes KVarN speculation lossless.

## 2. Verified facts this plan rests on (all checked this session, not inherited)

- The slice6 kernel template already carries `MultiBatch` and the lane buffers
  (`lane_packed_pages`, `lane_tail_counts`, `tail_batch_elems`) — wiring is a route, not a rewrite.
  **BUT the `<MultiBatch=true, KBits=4/5>` instantiation has never been compiled** — the gates made
  it unreachable. Expect latent compile/codegen issues on first instantiation (§5 Step 2).
- `tp2_backend.cpp` per-lane batched workspace is already width-generic (binds
  `kvarn_k_bits/kvarn_v_bits` from the tier table; see "kvarn batched workspace bind (lanes=%d)").
- The single hardcoded gate is `tp_engine.cpp:184`: `const bool kvarn = bopts.kv_cache ==
  KvCacheStorage::KvarnK4V2;` → widen via `kvarn_tier_widths(...).has_value()`.
- The launcher gate is the Step D throw `(k5v4 || k4v4) && multi` in
  `src/ops/launcher/gqa_attention_kvarn.cu` (~line 125) plus the k4v2-only route table.
- Identity/divergence is NOT tier-specific (n=24 ladder, this session): bf16 10/24, int8 9/24,
  k5v4 9/24, k4v4 11/24, k4v2 14/24. The lossless anchor diverges at 42% — byte-identity under
  speculation is unsatisfiable for ANY tier and was user-ruled an architectural attribute.
- §5 for k4v4/k5v4: vram + acceptance (one-sided gate, user ruling) + t/s + identity all
  recorded; refusals lifted at `e002548d`. Acceptance gate is ONE-SIDED (fail only >0.5pt BELOW
  bf16) — do not re-litigate; the "monotone in K precision" ordering claim was retracted.
- `q4_0` is OUT OF SCOPE: prefill fills an I4 cache as bf16 (correctness defect, refused). Its
  multibatch wiring is meaningful only after the Phase-3 per-TKV prefill prologue lands.
- `k3v3` does not exist in code (no enum/parse/kernel) — separate future work order.
- User data point: k4v4 ≈ q5-equivalent; lossless-adjacent behavior expected and observed
  (divergence 46% vs bf16 42% — flat, not quantization-driven).

## 3. Roles & work split

### 3.1 A1 — mainline + GPU (this lane's owner)
Gate widening (Step 1), launcher route (Step 2), all GPU validation runs (Steps 4–6), merges,
this document's log.

### 3.2 gemini — ALL test authoring (per user instruction)
Test deliverables, each as a commit on `wo/kvarn-multibatch` (coordinate file paths with A1):
- **T1 Batched-vs-plain bit-exactness battery**, per tier (k4v2, k5v4, k4v4): drive
  `gqa_attention_cached` (public op, REPO.md rule 10 pattern — mirror gemini's landed slice7
  green path) with identical prompts through batched and per-lane configs; assert byte-equality.
  This is the docs/132 "real lossless contract". Pattern reference: gemini c6146840 (slice7).
- **T2 Lane-isolation test**: lane N's output must be independent of other lanes' cache content
  (cross-lane pollution check under MultiBatch=true, varied lane counts 2/4/8).
- **T3 Batched MTP verify test for KVarN**: the docs/132 48/48-style bit-exact contract at
  draft_tokens=3, batched, per tier.
- **T4 Registry/enumeration test**: batched dispatch covers every registered kvarn tier —
  exhaustive no-default switch pattern (mirror `test_speculative_backend_enum.cpp`).
- Each test needs a rule-14 mutation run recorded (must fail when the thing it covers is broken).
Note: `tests/slice6_kvarn_k5v4_test.cu` exists only on `wo/shape-parity` — if T1/T3 need it,
bring it across deliberately (diff section headings both ways; see CLOSEOUT §7 truncation trap).

### 3.3 Helper agent (CPU-only packages, self-contained)
- **H1** Re-seal (coordinator-approved): fix the stale `ninfer_slice4_kvarn_test: PASS` clone
  label in `tools/i4_oracle/k5v4_oracle.cu` + the MANIFEST "11 PASS"→"10 cases + verdict" count,
  re-capture the sealed hashes, update `docs/baselines/k4v4/*`. Must re-run oracle+dispatch to
  re-seal honestly (GPU for the re-run — coordinate with A1).
- **H2** Fold the n=24 identity ladder (numbers in §9) into `docs/k4v4_s5_validation_results.md`.
- **H3** Implement the ONE-SIDED acceptance gate in `tools/ops/validate_117_dtypes.py`
  (acceptance check is currently "not automated"; user ruling: refuse only >0.5pt BELOW bf16).
- **H4** Audit docs/151 §17(e)/§5(d)(e) per-lane staging invariants vs the kvarn batched path;
  report gaps as a comment list (no code).

## 4. The exact change sites (verified line refs at e002548d; re-grep after any rebase)

| site | file:line | current | change |
|---|---|---|---|
| batched gate | `src/runtime/tp2/tp_engine.cpp:184` | `kvarn = kv_cache == KvarnK4V2` | `kvarn_tier_widths(kv_cache).has_value()` |
| launcher throw | `src/ops/launcher/gqa_attention_kvarn.cu:~125` | throw on `(k5v4\|\|k4v4) && multi` | route MultiBatch=true; keep the throw ONLY for genuinely unregistered widths |
| launcher route | same file, `kernel_fn` lambda | KSide=4/5 pass `MultiBatch=false` only | pass the caller's MultiBatch; KSide 0→slice4, 4/5→slice6 with KBits |
| stage smem | slice6 `.cuh` `kStageSmemBytes` | single-buffer 9232/8320 B | unchanged (single-buffer carries over; do NOT double-buffer) |

Do not touch: `src/ops/dflash2/**`, the drafter-budget seam in `tp2_budget.h` (A2's DFlash2 lane),
`validate_speculative_cli_options` arms.

## 5. Execution order (commit + test each step; log to §9)

- **Step 0** — Branch setup (§0), then confirm current behavior: run k4v4 + `--max-concurrency 4`
  and record exactly what the per-lane fallback does today (serialized-success vs throw).
  Reproduce the n=4 identity gate for k4v4 (baseline before changes).
- **Step 1** — Widen `tp_engine.cpp:184`. CPU test: `ninfer_tp2_budget_test` + `ninfer_serve_options_test`
  rc=0. Commit.
- **Step 2** — Launcher route + first compile of `<MultiBatch=true, KBits=4/5>`. Expect possible
  compile/codegen issues (never instantiated). Fix forward; do not paper over with `if constexpr`
  de-scoping. Commit when compiling + single-seq regression green.
- **Step 3** — Single-seq regression gates (must be byte-identical to pre-change):
  `<5,4>` oracle/dispatch/known-answer vs `docs/baselines/k4v4/MANIFEST.txt` (dispatch additive-only
  is acceptable IF 0 lines removed — record the diff), `<4,2>` three tests, `<4,4>` oracle 10/10,
  tp2_budget, serve_options. Commit.
- **Step 4** — gemini T1+T2 land → run on GPU. Lane isolation + batched-vs-plain per tier.
- **Step 5** — gemini T3 lands → batched MTP verify bit-exact battery per tier (the 48/48 contract).
- **Step 6** — Perf sanity: batched vs per-lane t/s at lanes=2/4/8 (this is the payoff measurement).
- **Step 7** — CI re-point: run_ci cells for k5v4/k4v4 batched; docs update; final log entry; merge.

## 6. Constraints (non-negotiable)

- Never bare `cmake --build build`; targeted targets only (rule 9). Kill only own PIDs; never
  `pkill -f <pattern>` (matches your own shell — done twice this session).
- Single-seq byte-identity gates are GATES: `<5,4>`/`<4,2>`/`<4,4>` single-seq outputs must remain
  byte-identical to the sealed baselines. Divergence = finding, stop and report.
- Batched-vs-plain BIT-EXACT is the new contract per docs/132 — a near-tie-flip failure there is
  a real defect, NOT acceptable-as-attribute (that ruling covered single-seq only).
- No DFlash2 files; no drafter-budget seam changes; no docs/117 §5 gate re-litigation
  (acceptance is one-sided by user ruling).
- Do not lift/alter the q4_0 refusal (correctness defect).
- Commit messages name this work order; append to §9 per step.

## 7. Definition of done

- `--max-concurrency 4 --kv-dtype kvarn_k4v4` (and k5v4) serves with batched throughput, verified.
- T1–T4 green on GPU for k4v2/k5v4/k4v4; batched-vs-plain bit-exact per tier.
- Single-seq sealed baselines unchanged (oracle/known-answer MATCH; dispatch additive-only).
- Perf numbers recorded (batched vs per-lane, lanes=2/4/8).
- §9 log complete; this doc closed out with final SHAs; merged to main + pushed.

## 8. Stall / escalation

- STOP and report if: ≥3 real conflicts against main; the `<MultiBatch=true, KBits=*>`
  instantiation fails to compile in a way that needs design (not mechanical) fixes; or batched-vs-
  plain bit-exactness fails on k4v2 (it was 48/48 — a regression there means the lane touched
  something shared).
- Coordinator may be slow/unavailable: merge autonomy is granted for verified kvarn-lane work;
  docs-only commits may go straight to main.

## 9. RUNNING LOG (append-only — newest at top)

### 2026-09-05 ~15:00Z — work order created (A1)
- Pre-work evidence recorded: identity ladder n=24 (bf16 10, int8 9, k5v4 9, k4v4 11, k4v2 14
  diverged of 24; lossless anchor ~42% floor), docs/132 §4.5 batched-MTP bit-exact 48/48,
  single-seq kvarn MTP not bit-lossless (user-accepted attribute), refusals lifted e002548d.
- Mainline owner A1; tests → gemini (T1–T4); helper packages H1–H4 defined.
- NEXT: Step 0.
