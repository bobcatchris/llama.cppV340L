# Doc 101 — MTP sampling-acceptance collapse (D-22) + plain-decode seed divergence (D-23)

**Status:** OPEN — queued first when the server is free. Takes priority over
magic-dict phase-1 validation, ngram live step, and adaptive MTP.

**Owner:** agent (GPU). **Reviewer:** plan owner (gate every step; do not trust
self-reported root causes — verify against code + fresh measurements).

**Scope:** two separate sampling defects, both temp>0. Do NOT mix with the D-21
closeout server items (different work, but same server — run in one session,
separate sections in the results doc).

---

## 1. Evidence base (verified by reviewer 2026-08-28)

### D-22 — MTP acceptance under non-greedy sampling collapsed 2026-08-21, never recovered

Raw verify logs (`repo/results/reports/*samp_s42.log`), config: `--mtp 3 --temp 1
--top-k 20 --top-p 0.95 --seed 42`, prompt "The capital of France is" (low-entropy):

| window | acceptance | mean a/round | t/s |
|---|---|---|---|
| **08-21 16:35–19:29 (6 runs)** | **46.4–47.2%** | 1.39–1.42 | **~63** |
| 08-21 21:04 → 08-22 06:30 | 2.4% | 0.07 | 28 |
| 08-22 07:29+ (after partial fix) | 13.4% | 0.40 | 37 |
| 08-25 → 08-26 11:05 (stable) | 14.4% | 0.43 | 33 |
| **08-28 18:41 (current build)** | **9.2%** | 0.28 | 28 |

- The collapse 47.2% → 2.4% happened in one ~1.5 h window (08-21 19:29 → 21:04).
  **Mainline history contains NO functional commit in that window** (only
  `184e9d15`, docs). The regression was introduced in a **dirty worktree**
  (`/tmp/ninfer`, branch `mtp-perf`, per doc 41) — that tree no longer exists.
- Doc 41 (`repo/docs/41_tcp_issue2_bisect_handoff.md`) is the contemporaneous
  handoff for exactly this era: TC-P (prefix ON/OFF identity) bisect data, ranked
  suspects, eliminated list. Read it before touching code.
- `0a3b50e3` (08-22 07:36, "fix MTP round 0 base position index to prevent KV
  cache prompt corruption") was the partial fix → 13.4%. `84ba2c8b` (08-28,
  D-21 anchor) is in the current tree but acceptance is still 9.2% → the
  round-0 position family is not fully resolved, or a second cause remains.
- llama.cpp parity (user, 2026-08-28): **>70% acceptance on basically all
  requests at temperature 1**. So 9–14% is not the physics; 47% may also be
  below a healthy baseline (see §2.4).

### D-23 — plain-decode sampling: different seeds produce identical output (S3 FAIL)

- S3 (`tools/smoke/serve_battery.py`, "Sampling API Determinism & Divergence"):
  temp 0.8, no top-k/top-p, thinking=on, max_tokens 32, poem prompt.
  Same seed (42/42) identical ✓; **seed 42 vs 43 identical ✗** (FAIL) on the
  08-28 16:41 build; PASSED on the 08-26 11:05 build.
- The failing run was **plain decode (`speculative=off`), KVarN packed KV** —
  the 08-26 passing run predates packed-only decode (bf16 shadow path).
- Regression window = the packed-decode/A2/wall-removal commits
  (`f72df497` → `1304ada9` → `98db350b` → `de8b6439` → `b9ed8949` → `0ed8d990`
  → `ebc38f43`). The D-21 anchor fix (`84ba2c8b`) is out of scope for S3
  (speculative=off).
- Note: the verify suite's MTP sampling runs (temp 1, top-k 20, top-p 0.95)
  still show `sampling_div_ok: true` on the same 08-28 build — i.e. seeds DO
  diverge there. Discriminators to exploit: (a) MTP vs plain path, (b)
  truncation (top-k/top-p) vs raw temp 0.8, (c) serve (thinking=on) vs
  decode_test (no thinking), (d) packed KV vs bf16 shadow KV.
- Sampler implementation: stateless hash RNG —
  `sampling_uniform(seed, position, purpose, sub)`
  (`src/ops/kernel/sampling_device.cuh:62`, splitmix64-based, seed IS mixed in),
  consumed in `src/ops/kernel/sampling.cuh:78,264` and
  `src/ops/kernel/speculative_round.cuh:159-171`. Seed wiring:
  `src/serve/translate.cpp:47-52` → `tp2_backend.cpp:884`
  (`host_cfg.seed = req.sampling.seed`), copied to device pre-decode.
  No stateful RNG; nothing to desync between requests, but a stuck
  `logical_positions`/`purpose` or a field-ordering mismatch in
  `SamplingConfig` host→device would make the seed ineffective.

---

## 2. D-22 work order

### 2.1 Read, don't re-derive
1. `repo/docs/41_tcp_issue2_bisect_handoff.md` — especially §4.3 leak gate
   (leak requires >1 MTP-generated token), §4.4 eliminated list (do not
   re-investigate), §5 ranked suspects (A: cross-stream arena aliasing race on
   SwiGLU GEMV partials; B: unordered read in SwiGLU GEMV).
2. `git show 0a3b50e3` and `git show 84ba2c8b` — the two round-0 position fixes.
   Verify both are present and consistent in the current tree (they touch the
   same code region; check for interaction/oversight).

### 2.2 Reproduce + instrument (server session, section 1)
1. Run `tools/verify_mtp_sampling.sh` (40% floor) + the verify suite's
   `samp_s42/samp_s43` on the current build. Record acceptance, mean a/round,
   t/s, build commit + model identity in results.
2. Add temporary instrumentation (env-gated, default off): per-position
   acceptance (which of the k drafts reject first) + the target's top-1
   probability and p(draft_token) at each reject position. One run is enough
   to classify the failure mode:
   - **rejects concentrated at position 0 with p(draft) ≈ uniform-flat** →
     target distribution corrupted/flat (KV/state bug).
   - **p(draft) looks sane (~40-70%) but accepts still low** → acceptance
     logic bug (read `speculative_round.cuh:150-180` and verify the rule —
     see §2.3).
   - **p(draft) flat specifically for KVarN packed KV, sane for bf16** → packed
     decode numerics (cross with D-23 suspects).

### 2.3 Reference behavior (what "correct" is) — researched 2026-08-28
- **llama.cpp** (`common/sampling.cpp` `common_sampler_sample_and_accept_n`):
  *sample-and-match* — sample from the target's sampling chain at each draft
  position; accept the draft token iff the target sample == draft token; on
  rejection the target sample is used (exact distribution, no bias). With an
  argmax draft, per-position acceptance = p_target(draft) — **identical math to
  our degenerate-draft case**.
- **llama.cpp MTP draft** (`common/speculative.cpp`): draft head runs its own
  sampler (top-k) with a **`p_min` confidence gate** — drafts whose top-1
  confidence < p_min are NOT proposed. This prevents low-confidence drafts from
  diluting the acceptance average. We currently propose all k=3 drafts
  unconditionally (argmax).
- **vLLM** (`vllm/v1/sample/rejection_sampler.py`): standard rejection sampling
  via the exponential/Gumbel trick (fused kernel), with accepted / **recovered**
  (residual (p_t−p_d)+) / bonus tokens; per-request generators for
  determinism. Same acceptance math.
- Consequences: (a) our acceptance math is correct in kind; 9–14% cannot be the
  physics at temp 1 for a confident 27B model (llama.cpp: >70%). (b) Even our
  08-21 47% is ~25 pts below llama.cpp parity — either the user's llama.cpp
  model/prompt mix differs, or our target distribution is milder (flatter) than
  it should be at baseline. Quantify in 2.2 step 2: **measure mean p(argmax) at
  temp 1 on the current build; if it is ~0.1-0.3, the target distribution is
  flat (bug); if ~0.7, acceptance math/positioning is the bug.**

### 2.4 Fix (pick per diagnosis, one at a time, commit each)
- If KV/state corruption (doc 41 suspect A/B or round-0 position remainder):
  fix in `src/runtime/tp2/` (arena/stream ordering or position index),
  verify with the leak gate from doc 41 §4.3 (`--mtp 3 --tokens 64` ON/OFF
  prefix identity) AND acceptance recovery.
- If packed-numerics (cross with D-23): coordinate with D-23 bisection — one
  root cause may explain both.
- If acceptance logic: fix `speculative_round.cuh` to the sample-and-match rule
  (verify the bonus token is the target sample at the reject position, not a
  re-draft).
- Do NOT re-apply reverted commit `c82cf7b6` ("assert temperature scaling, not
  an absolute bar"). Its premise ("40% floor physically unreachable, ~9% is
  correct") is **wrong**: known-good is 47%, llama.cpp parity is 70%+. Keep the
  40% floor in `tools/verify_mtp_sampling.sh`.

### 2.5 Gate (D-22 done)
1. `tools/verify_mtp_sampling.sh` PASS with **acceptance ≥ 40%**, target ≥ 46%
   (08-21 known-good). Report the number; if stuck 35–45%, document the gap vs
   llama.cpp parity before accepting.
2. `samp_s42/samp_s43` in the verify suite: det + div still true.
3. Doc 41 leak gate (TC-P prefix ON/OFF) clean.
4. Greedy acceptance (80.4% era) not regressed >2 pts.
5. Results committed with build commit + model identity; docs/50 D-22 line
   updated.

---

## 3. D-23 work order

### 3.1 Discriminate (cheapest first)
1. **CPU sampler unit test** (new, in `tests/`): fixed synthetic logits (known
   top-2 gap), temp 0.8, seed 42 vs 43 over 32 positions → MUST diverge;
   seed 42 vs 42 → MUST match. This isolates the sampler from model/KV path.
2. **decode_test, no thinking, no MTP**: temp 0.8 seed 42 vs 43, same poem
   prompt, 32 tokens, current build (packed) — diverge or not? Then rerun on a
   pre-packed build (bf16 shadow, e.g. `b9ed8949^`) if the server allows a
   rebuild. This splits {sampler bug} vs {KV numerics} vs {thinking path}.
3. **Top-1 probe**: on the failing (thinking=on) request, dump per-token
   top-1 probability (temp 0.8). If every token's top-1 p ≈ 0.99+ (deterministic
   reasoning chain), S3 is testing a near-deterministic regime and the test
   itself is too sensitive → fix the TEST (higher-entropy prompt and/or temp
   ≥ 1.2, or assert divergence over more tokens), documented. If top-1 p is
   moderate (0.7–0.95) yet seeds don't diverge → sampler/seed bug, continue.
4. **Bisection** (only if 2/3 point at code): binary-search
   `f72df497 → ebc38f43` with the 3.1.2 repro. Suspect ranking (reviewer):
   A2 T=1 epilogue (`de8b6439`/`0ed8d990`) > packed decode routing
   (`1304ada9`/`f72df497`) > int8 template refactor (`ebc38f43`, claimed
   byte-identical at `<false>` — verify the claim in-tree).

### 3.2 Fix + gate (D-23 done)
1. Root cause fixed (or test redesigned per 3.1.3 with rationale + reviewer
   sign-off).
2. S3 PASS on the fixed build; S1–S5 battery all pass.
3. `sampling_det_ok`/`sampling_div_ok` unchanged (true/true).
4. Results + docs/50 D-23 line committed.

---

## 4. Session plan (one server session, in order)

1. Commit or stash the in-flight uncommitted work in this tree
   (`gqa_attention_kvarn_decode_packed.inc`, `gqa_attention_kvarn.cu`,
   `bench_kvarn_attention.cu`) — do not bisect on a dirty tree.
2. D-21 closeout server items (temp 0.8 robot/sunsets re-run, full-width H5
   matrix, full CI) — already queued in
   `results/d21_mtp_verify_anchor_fix.md` "Pending server items".
3. D-22 §2.2 → §2.4 → §2.5.
4. D-23 §3.1 → §3.2.
5. Full verify suite + S1–S5 battery on the final build; commit results.

## 5. Non-goals
- No speed work (docs/83 M3/M4, lever-1) until both defects are closed.
- No magic-dict / ngram / adaptive-MTP server steps until D-22+D-23 gates pass.
- No draft-strategy changes (p_min gate, sampled drafts, top-k>1) — those are
  speed-side enhancements for a follow-up doc; they must not mask the bug fix.
  (Reference for the follow-up: llama.cpp `p_min` confidence gate, §2.3.)
- No merges (docs/77 §8 hold still in effect).

## 6. References
- `tools/verify_mtp_sampling.sh` (40% floor; history: `5a69b16e`, reverted
  `c82cf7b6`/`d00d2779`) · `tools/smoke/serve_battery.py` S3
- `results/d21_mtp_verify_anchor_fix.md` (D-21 closeout + pending server items)
- `repo/docs/41_tcp_issue2_bisect_handoff.md` · commits `0a3b50e3`, `84ba2c8b`
- llama.cpp: `common/sampling.cpp` (`sample_and_accept_n`),
  `common/speculative.cpp` (MTP draft + p_min gate) · vLLM:
  `vllm/v1/sample/rejection_sampler.py`, `docs/features/speculative_decoding/mtp.md`
