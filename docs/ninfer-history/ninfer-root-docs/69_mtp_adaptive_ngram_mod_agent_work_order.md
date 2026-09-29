# 69 — MTP adaptive draft depth + ngram-mod port: Agent Work Order

**Status:** CURRENT — work order for a single implementer agent.
**Mission:** port two proven llama.cpp spec-decoding features into ninfer-serve:
(1) adaptive MTP draft depth (3→12, counting state machine, PR ggml-org/
llama.cpp#27210) and (2) ngram-mod (rolling-hash n-gram draft pool with
variable 48–64 token drafts, PR #19164), then their combination. Done = a
live MTP server decodes measurably faster than the fixed-k=3 baseline on
reasoning/code/prose probe classes (their published deltas: 84→134 t/s
anecdote; C3/C7 table 51–78 t/s vs 30 baseline on Qwen3.8-27B), lossless
verification proven (T10 determinism + battery), no VRAM regression, spec
disabled ⇒ bit-identical to current behavior.
Read this document fully before writing code.

---

## 1. Context (60-second version)

Our speculative decoding is fixed-depth MTP (`--spec mtp --draft-tokens 3`,
~82% acceptance baseline, docs/50). llama.cpp has since added (a) **adaptive
MTP depth** — a counting state machine that climbs draft depth after
consecutive full accepts and drops after misses (PR #27210, head
`a8f2138e`, open/unmerged, 92-line self-contained header + 218-line unit
test) and (b) **ngram-mod** — a ~16 MB rolling LCG-hash pool mapping 24-grams
to next tokens, shared across requests, producing variable-length drafts of
48–64 tokens (PR #19164, merged). Their results on Qwen3.8-27B show the
combination (adaptive MTP fronting ngram-mod) roughly doubling t/s on
repetitive coding/reasoning traffic. Both are opt-in upstream; both must be
opt-in here (default OFF, bit-identical when off).

**Already built and verified (do not redo):**
- MTP spec rounds, fixed k=3: draft chain (`src/targets/qwen3_6/impl/runtime/
  mtp_impl.h`), verify+accept kernels (`src/ops/kernel/speculative_round.cuh`),
  TP2 round loop + per-round accept tracking (`src/runtime/tp2/
  tp2_backend.cpp`).
- Simple per-request n-gram lookup (`--lookup`, 2–4 grams, fixed k):
  `find_context_lookup_drafts` (tp2_backend.cpp:46) — **NOT** ngram-mod;
  do not conflate or modify it.
- MTP acceptance stats (`acc_accepted`/`acc_rounds`, tp2_backend.cpp:~1365).

**What you are doing:** steps 0–5 of §6 — baseline, adaptive port, k=12
ceiling, ngram-mod pool, combination+tuning, closeout.

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; remote `github`).
- **Work protocol (MANDATORY):** ALL work happens in a worktree + branch —
  never edit the main tree directly (2026-08-24 incident).
  ```bash
  cd /home/intel/ninfer/repo
  git worktree add ~/ninfer/worktrees/wo-mtp-adaptive -b wo/mtp-adaptive
  cd ~/ninfer/worktrees/wo-mtp-adaptive
  cmake -S . -B build && cmake --build build -j 16
  ```
  Commit per step to `wo/mtp-adaptive` and push. **Merging to main is done by
  the main-side agent/user.**
- **DO NOT TOUCH other agents' worktrees.** Active as of 2026-08-26:
  `wo/kvarn-hold` (docs/78 C4 packed-only decode kernel — owns the live
  server), `wo/kvarn-d21` (docs/79, D-21 MTP chunked-prefill fix),
  `wo/magic-dict` (docs/98 phase 1 corpus collector). File sets are
  disjoint: this order owns the tp2 round loop, serve options, mtp_impl +
  the new ngram-mod/adaptive modules; the KVarN orders own the KVarN
  kernels and the MTP commit path. Merges are coordinated main-side.
- Build: `cmake --build build -j 16` (CUDA 13.1, `sm_120a`, 2× 5060 Ti).
- Unit tests: **`/usr/bin/ctest`** (PATH `ctest` is broken). From `build/`:
  `/usr/bin/ctest -R "spec|adaptive|mtp"`.
- **Test entry points (standard):** battery
  `tools/smoke/serve_correctness_ci.sh` (env-overridable; modes `ci`/`full`;
  `--only` wins; exit = #failed); must-pass lists: **docs/50 §7.1 is the
  source of truth**; T19 threshold never modified; full pipeline
  `bash tools/ops/run_ci.sh --full` (closeout only); live server per
  LAUNCH.md via `setsid`, one server at a time, swap protocol §7; prompt
  sizes from `usage.prompt_tokens` never char estimates; decode rate from
  serve.log (`decode=Xtok/s`); `nvidia-smi` clocks with every perf number;
  all probe outputs committed under `results/` with config + clocks.
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.
- Upstream sources to port from (MIT): PR #27210 head `a8f2138e` —
  `common/speculative-adaptive.h` (92 lines), `tests/test-speculative-
  adaptive.cpp` (218 lines), wiring in `common/speculative.cpp`; PR #19164 —
  the ngram-mod hasher + `--spec-ngram-mod-*` flags; and read its follow-ups
  before porting: #23458 (EMA acceptance tracking), #22168 (low-acceptance
  reset), #23929 (`-sm tensor`+MTP+ngram-mod crash), #25819 (stuck-loop
  escape, WIP). Record upstream commit SHAs in your report.

## 3. Architecture facts (verified — do not re-derive)

- Engine path is **TPEngine** (`src/runtime/tp2/tp_engine.h`), NOT
  ConcurrentExecutor (REPO.md §2a). Model: 24 q / 4 kv heads, head_dim 256,
  hidden 5120, vocab 248320 (`src/targets/qwen3_6_27b/impl/config.h`).
- Draft depth `k` is a **runtime value, not compile-time**: `--draft-tokens
  N` (serve_options.cpp:237) → `speculative.draft_tokens` → `b_opts.mtp_k`
  (tp_engine.cpp:295) → `req.mtp_k` (tp_engine.cpp:157; default 3,
  tp2_request.h:25) → `k = mtp ? req.mtp_k : 0` (tp2_backend.cpp:696).
  Buffers already size with k: `local_tok max(k+1,8)` (tp2_backend.cpp:500),
  `verify_hidden {5120,k+1,1}` (559), `alignment_hidden` (564),
  `licensed_counts` k+1, `accept_res_pinned`.
- MTP draft chain is a **sequential loop**: each draft token conditions on
  the previous draft token (`mtp_impl.h:35-57`) — same design as llama.cpp's
  MTP; k=12 means 12 sequential MTP module steps (draft phase ~4× longer
  than today; pays off when accepts run).
- **Per-round accept tracking exists**: accept kernel → `a` (accepted
  drafts) → `accepted_count = a + 1` (tp2_backend.cpp:1374) →
  `acc_accepted/acc_rounds` stats (the 82% MTP-accept data). This is the
  adaptive state machine's input — nothing to instrument.
- Per-round draft selection: `st.drafts1` (MTP module prefill output, ~931),
  optionally overwritten by `--lookup` n-grams (1174–1182). The new
  ngram-mod chain and adaptive depth both insert at this seam.
- Round state: `RoundStateSpec{.hidden=5120, .output_rows=248320,
  .batch_capacity=1}` (tp2_backend.cpp:230/284) — verify of k+1 tokens is a
  single forward pass; verify logits scale linearly with k (k=64 ⇒ ~32 MB
  staging, transient).
- Spec is lossless by construction: verify kernel rejects any wrong draft;
  the guards for this work are **T10 determinism** + battery + "spec OFF ⇒
  bit-identical", NOT a pp decode guard (this is decode-side work).

## 4. Key call sites (anchors — verify line numbers before editing)

- `src/runtime/tp2/tp2_backend.cpp:696` — `k` per request; the round loop
  (1070–1450) where drafts are generated, verified, accepted; insert
  adaptive update right after `accepted_count` (1374), insert next round's
  depth before draft generation.
- `src/runtime/tp2/tp2_backend.cpp:46` — `find_context_lookup_drafts` (the
  EXISTING `--lookup` n-gram; leave untouched).
- `src/targets/qwen3_6/impl/runtime/mtp_impl.h:35-57` — MTP chain loop.
- `src/serve/serve_options.cpp:71-76, 237-239` — usage string +
  `--draft-tokens` parsing (new flags follow this pattern).
- `src/runtime/tp2/tp2_request.h:25`, `tp2_backend.h:119` — `mtp_k`
  defaults (3).
- `src/ops/kernel/speculative_round.cuh` — accept kernel (parameterized on
  k; should need no changes for variable per-round k — verify).

## 5. Design decisions (FINAL — do not re-litigate)

1. **Port the adaptive state machine verbatim** (92-line header, MIT, pin
   `a8f2138e`, attribution comment) — its climb/drop constants were
   empirically tuned upstream; do not "improve" them in this work order.
   `--spec-draft-p-min` is OUT of scope (upstream C4 shows it hurts
   reasoning).
2. **Opt-in, default OFF, both features.** Spec behavior with no new flags
   must be bit-identical to current (T10 + byte-identical A/B on a fixed
   prompt).
3. **ngram-mod pool is HOST RAM** (~16 MB, process-lifetime, shared across
   requests — that sharing IS the feature). No new persistent DEVICE
   allocations; the k=12/64 buffer deltas are the existing staging arena,
   measured and reported.
4. **Per-round variable k:** ngram-mod chains (n-min..n-max) and adaptive
   depth coexist by selecting the draft source per round — ngram chain if a
   hit yields ≥ n-min tokens, else adaptive-MTP chain of current depth.
   Buffers size to `max(k_mtp_max, ngram_n_max) = 64`; per-round k is a
   runtime parameter to the verify/accept path.
5. **REJECTED: raising the default draft depth** — default stays 3 until
   the main side decides.
6. **Their absolute t/s numbers are AMD R9700 + Q8_0** — do not gate on
   them. Gates are RELATIVE: t/s ≥ fixed-k=3 baseline on every probe class,
   improvement expected on reasoning/code; hit/accept trajectories reported.
7. **Read the ngram-mod follow-ups before porting** (#23458, #22168,
   #23929, #25819) — upstream hit stuck loops, low-acceptance resets, and a
   tensor-split crash; encode their fixes in the port.

## 6. Execution order (commit + test each step before the next)

**Testing standard (applies to every step):** live server + real request
sequences for every server-facing step; kernel/unit isolation is necessary,
never sufficient.

### Step 0 — Baseline (no behavior changes)
Probe: 3 prompt classes (reasoning, prose, code — construct fixed prompts,
≥2k generated tokens each, greedy) × fixed k=3, 3 runs, median decode t/s
from serve.log; record MTP acceptance (acc stats) + `nvidia-smi` clocks.
Commit to `results/`.
**Tests:** probe reproduces known state (acceptance ~82%); tree untouched.

### Step 1 — Adaptive state machine (opt-in)
Port header → `src/runtime/tp2/mtp_adaptive.h` (MIT attribution, pinned
SHA); port the 218-line unit test → `tests/test_mtp_adaptive.cpp`; add
`--mtp-adaptive` + `--mtp-draft-max 12` + `--mtp-draft-min-adaptive 3`
(names may match house style; semantics exactly the PR's); wire: update
state after `accepted_count` (tp2_backend.cpp:1374), next round's depth =
clamp(state, min, max), next round's MTP chain length = depth (k=3 until
step 2).
**Tests (must pass before moving on):**
- `/usr/bin/ctest -R adaptive` — state machine unit tests green (climb,
  drop-pressure, floor, ceiling, cold start 3).
- Live: `--mtp-adaptive` OFF ⇒ byte-identical output vs current on a fixed
  greedy prompt; ON ⇒ depth trajectory logged (depth, accepts/round per
  round); t/s probe ≥ baseline on all three classes; must-pass battery
  subset green (docs/50 §7.1); T10 determinism green.

### Step 2 — k ceiling 12
Lift `mtp_k` acceptance to 12 (buffers already k+1-parameterized — verify
smem/VRAM at k=12 live: report MiB delta; 12 sequential MTP steps — report
draft-phase ms).
**Tests:** T10 green; battery subset green; VRAM delta reported (must be
small, staging arena only); t/s probe with adaptive+12 vs step-1 baseline
(reported; no regression on any class).

### Step 3 — ngram-mod pool (opt-in)
Host-side pool (~16 MB, LCG rolling hash, 24-gram, process-shared) per
upstream PR #19164 + its follow-up fixes (§5.7); per round: rolling hash of
generated history; on hit, draft chain length = consecutive pool hits,
clamped to `--ngram-mod-n-min 48`..`--ngram-mod-n-max 64` (flags named
upstream-style); select chain when ≥ n-min, else fall through to MTP
drafts; per-round variable k through verify/accept (buffers to 64).
**Tests:**
- Unit: hasher + rolling hash (known-token vectors, wraparound, eviction).
- Live: OFF ⇒ byte-identical; ON on a repetition-heavy prompt (model
  re-emitting its own reasoning; code boilerplate) ⇒ hit rate + chain
  lengths logged; t/s probe ≥ baseline on all classes (wrong drafts are
  lossless but cost compute — a hit-rate collapse = report, don't tune
  blindly); T10 green; battery subset green; VRAM: pool ~16 MB host (report
  RSS), staging delta for k=64 reported.

### Step 4 — Combination + tuning
Per-round policy (ngram chain if ≥ n-min else adaptive depth); tune
n-match/n-min/n-max on our three probe classes; final t/s table
(reasoning/prose/code, t/s + acceptance + hit rate) vs step-0 baseline.
**Tests:** T10 green; battery subset green; table committed to `results/`.

### Step 5 — Closeout
`bash tools/ops/run_ci.sh --full` (stock int8) + one 250k KVarN MTP launch
with `--mtp-adaptive` (sanity: no cross-path regression). All green or
documented exceptions; commit all probe data.
**Tests:** full CI green; report per §8.

## 7. Constraints (non-negotiable)

- **Worktree only:** no edits in `/home/intel/ninfer/repo` outside
  `~/ninfer/worktrees/wo-mtp-adaptive`, ever (§2). **Never touch
  `wo/kvarn-pp` or its branch** (active agent, docs/68).
- **Live end-to-end before "done":** every server-facing step runs the real
  server with the real request sequence (§6).
- **No damage:** commit per step with a message naming the step; tree
  buildable at every commit; no untested code.
- **GPU/server:** one live server at a time; port 8091 serves an active
  conversation — swap protocol: ask → `pkill -x ninfer-serve` (NEVER
  `pkill -f`) → GPUs <500 MiB → launch per LAUNCH.md → run → restore; check
  `pgrep -x ninfer-serve` + `nvidia-smi` before every launch.
- **No new persistent device allocations.** Pool is host RAM; staging
  arena deltas measured and reported. Default flags ⇒ current behavior,
  bit-identical (T10 + A/B byte compare).
- **Do NOT touch:** KVarN prefill path (`src/ops/launcher/
  gqa_attention_kvarn.cu`, KVarN kernels — docs/68's territory), I8/BF16 KV
  paths, prefix-tail machinery, the existing `--lookup` n-gram, T19
  threshold.
- **Losslessness guard:** any T10 determinism failure or non-bit-identical
  spec-OFF output ⇒ revert the step, not tune around it.
- **Measurement data is project data:** every probe run committed under
  `results/` with server config + clocks.

## 8. Definition of done

1. Steps 0–5 committed to `wo/mtp-adaptive` with passing tests at each step.
2. **Live proof:** fresh server launch per final config serving real
   reasoning/code/prose prompts; launch commands recorded in the report.
3. **t/s table** (step-0 vs final, per class) with acceptance + ngram hit
   rates; every final number ≥ its step-0 counterpart; the relative gain vs
   fixed-k=3 stated in t/s.
4. **Lossless:** T10 green at every step; spec-OFF byte-identical; upstream
   commit SHAs recorded for the ported code.
5. All measurement data in `results/`; report = one paragraph per step +
   the key numbers table (t/s, acceptance, hit rate, chain lengths, VRAM,
   clocks).
