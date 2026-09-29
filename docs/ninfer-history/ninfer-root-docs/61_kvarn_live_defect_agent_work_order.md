# 61 — KVarN live defect (T16): Agent Work Order

**Status:** Step 5 COMPLETE + committed (`dca3ca7f`); **Step 6 (D-16) COMPLETE**
(2026-08-24) — T17 PASS on KVarN via prefix-tail snapshot/restore.
Worktree protocol unchanged (docs/99): work here, commit + push per step,
main side merges.

**Step 5 acceptance (done):** unit bit-exact; T8 curve 678→720 tok/s; must-pass
green; staged shadow 1022 MiB/rank; D-15 closed in docs/50.

**Step 6 — D-16 prefix tail (T17 green) — DONE:**
- Snapshot open-page bf16 tails + MTP seed at prefix-cache save (end of prefill).
- On full hit: restore tails into workspace, stage open page, reuse snap MTP seed.
- Removed unaligned→re-prefill force. Verified: T17 PASS; T2/T16 green.
- Must pass: **T17 green on KVarN at 250k** (repeat ≥2× faster than cold);
  T1/T2/T3 (prefix equivalence/divergence) green; T16 green; must-pass subset
  green. Unit test for the tail snapshot/restore round-trip first.
- Close D-16 in docs/50 with commit refs + numbers.

**Mission:** make `--kv-dtype kvarn_k4v2` serve real requests end-to-end.
Done = a fresh 250k launch serves battery T16 (and the must-pass subset)
green, with root cause documented in docs/54 and the defect closed in
docs/50.

---

## 1. Context (60-second version)

KVarN P2c steps 1–3 are merged on main (pool layout `0b2a073e`, write path
`2c841fbb`, read kernel + dispatch — verified: isolation rel_l2 9.9e-5 vs
bf16-rounded fp32 CPU reference, deterministic). Step 4 was started from the
main side: budget constant set to the layout-derived value and the startup
guard lifted (uncommitted as of this writing — see §6 step 1). The server now
launches at 250k (`--kv-dtype kvarn_k4v2 --kv-capacity 250000 --max-context
250000`, 13,611 MiB/rank measured) **but every request fails** with:

```
gqa_kv_append_kvarn: append jumped pages with a partial tile still resident
```

Live repro state at handoff: server running on port 8091 with exactly that
config; battery T16 (new, `tools/smoke/test_serve_correctness.py`) fails with
HTTP 500. Log signature per request: `prefix partial (45/53): GDN restore
skipped` → `prefill: 65/67 tok (97%)` → the error on both ranks.

docs/54 §9 P2c has the full design history; docs/50 is the defect register.

**Already built and verified (do not redo):**
- Tile codec + attention kernel correctness (unit: `ninfer_kvarn_tile_cuda_test`,
  `ninfer_kvarn_gqa_test` — both green on main).
- Write-path unit tests incl. a new request-boundary reset test
  (`test_stale_workspace_reset` in `tests/test_kvarn_write_path.cpp`, green).
- Budget model + guard lift (main-side, uncommitted — commit it per §6 step 1).

**What you are doing:** root-cause and fix the live append failure (§6), then
finish step 4 (P3 test-suite run at 250k).

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; remote `github`).
- **Work protocol (MANDATORY):** all work in a worktree + branch — never edit
  the main tree directly:
  ```bash
  cd /home/intel/ninfer/repo
  git worktree add ~/ninfer/worktrees/wo-kvarn-live -b wo/kvarn-live
  cd ~/ninfer/worktrees/wo-kvarn-live
  cmake -S . -B build && cmake --build build -j 16   # own build dir, one-time full build
  ```
  Commit per step to `wo/kvarn-live` and push. **Merging to main is done by
  the main-side agent/user** — do not merge or push to main yourself.
- Build (inside your worktree): `cmake --build build -j 16` (sm_120a; 2× RTX
  5060 Ti 16 GB). Unit tests: **`/usr/bin/ctest`** (the PATH `ctest` is a
  broken Python wrapper), from your `build/`.
- Test suite (NOT "battery"): `python3 tools/smoke/test_serve_correctness.py
  --only T16` (repro) / must-pass subset `--only T1,T2,T3,T5,T7,T8,T10,T11`
  (docs/50 §7.1). MTP acceptance baseline 82.0%, fail only < 78%.
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.
- **Live server:** one at a time, port 8091 currently serves an active
  conversation — swap protocol: ask the user → `pkill -x ninfer-serve` (exact
  name; NEVER `pkill -f`) → wait GPUs < 500 MiB → launch per LAUNCH.md "250k
  context (KVarN KV cache)" → run → restore.

## 3. Architecture facts (verified — do not re-derive)

- TP2 only: TPEngine + TpBackend (`src/runtime/tp2/`), REPO.md §2a. Qwen3.8-27B
  hybrid GDN: 16 full-attention (GQA) layers + 48 GDN; kv_heads=2/rank,
  head_dim=256; MTP adds 1 GQA layer. page = G = 64 tokens.
- KVarN storage: per layer/head/page — K codes 8,192 B (int4), V codes 4,096 B
  (int2), scale side table fp32 `[layers, heads, physical_pages, 1152]`
  (index formula in `kvarn_scale_at`; write and read share it). Packed page =
  16,896 B vs int8 34,816 B.
- Write path: bf16 scatter into a per-sequence tile workspace
  (`KvarnSequenceWorkspace`, one instance for text + MTP pools — see
  `tp2_backend.cpp` bind ~L301); commit is the only writer of packed storage,
  launched when a tile completes (in `gqa_kv_append_kvarn_and_commit`, which
  splits multi-page calls into per-page runs and commits between them).
- Workspace state (`tile_page`, `tail_count`) **persists across requests** —
  it is only touched by appends, MTP trims (`kvarn_rewind_text/mtp`), and the
  main-side reset added in the re-prefill branch of the per-request setup
  (`tp2_backend.cpp` ~L783, uncommitted).
- Prefix restore: token-level match; **partial matches (match < full cached
  sequence) zero `prefix_len` to 0** — i.e., re-prefill from token 0 with GDN
  slot zeroed. Full matches copy the GDN slot and publish the KV mapping.

## 4. Key call sites (anchors — verify line numbers before editing)

- `src/runtime/tp2/tp2_backend.cpp` ~L745–800 — per-request setup: prefix
  match, "prefix partial" printf (~L765), restore vs re-prefill branch,
  main-side reset (~L783).
- `src/runtime/tp2/tp2_backend.cpp` ~L186–204, ~L301–304 — KVarN pool dtype +
  workspace bind (`kvarn_bind_sequence_workspace`, `set_kvarn_workspaces`).
- `src/ops/kvarn/kvarn_workspace.cpp` — `gqa_kv_append_kvarn` (all the
  invariant checks that throw), `gqa_kv_append_kvarn_and_commit` (per-page run
  splitting + commit), `gqa_kvarn_commit_completed`, `gqa_kvarn_rewind_to_token_count`.
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h` — `kvarn_attend_text`
  / `kvarn_attend_mtp` (append + attend), `mtp_prefill_chunk` append (~L697),
  `attn_mix` / `attn_mix_tp` dispatch.
- `src/targets/qwen3_6/impl/runtime/text_context.h` ~L213–221 — rewind API.
- `tests/test_kvarn_write_path.cpp` — write-path unit tests incl. the new
  reset test; `tools/smoke/test_serve_correctness.py` T16 — live repro.

## 5. Design decisions (FINAL — do not re-litigate)

1. Read path = fused dequant in smem inside the GQA kernel (no staging pass).
2. Commit is the only writer of packed storage; scales never in pool planes.
3. Re-prefill from token 0 must start from a clean workspace (main-side reset
   direction is correct — keep it, find why it is insufficient).
4. KVarN prefix reuse at non-64-aligned boundaries is NOT supported by the
   tile model — partial matches re-prefill from 0 (current behavior; do not
   "fix" it into mid-page restore without a design doc).

## 6. Execution order (commit + test each step before the next)

### Step 1 — Commit the main-side groundwork
Commit the uncommitted budget constant (`tp_engine.cpp`, `kv_bytes_per_token
= 18496 * 16896 / 34816` = 8,972, layout-derived) + guard lift + the re-prefill
reset in `tp2_backend.cpp`. **Test:** build green; unit suite green.

### Step 2 — Root-cause the live failure (T16 must go red→green here)
The reset exists but T16 still fails at the same spot. Open questions to close
with evidence (not guesses):
- Confirm the running binary contains the fix (object/binary mtimes, or a
  deliberate canary log line in the reset branch).
- Determine which append call throws: add a temporary diagnostic (position,
  page, `tile_page`, `tail_count`, text-vs-MTP pool) at the throw site; read it
  from the serve log on a T16 run.
- Candidates to check against that evidence: (a) reset runs after some append
  already happened in this request's path; (b) the MTP pool workspace or a
  second workspace instance is stale; (c) warmup leaves state in a different
  structure than the per-request reset clears; (d) chunked-prefill call order
  vs `publish_mapping`.
**Tests (must pass before moving on):**
- Unit: extend `tests/test_kvarn_write_path.cpp` with the exact failing
  sequence as observed live (warmup-like fill → request-boundary reset →
  re-prefill from 0, including the MTP pool if implicated). It must fail
  before your fix and pass after.
- Live: T16 against a fresh 250k launch — this is the gate for the step.

### Step 3 — Fix + regression
Apply the minimal fix consistent with §5. **Tests:** unit suite green; T16
green on a fresh launch; must-pass subset `T1,T2,T3,T5,T7,T8,T10,T11` green at
250k; MTP acceptance vs 82.0% baseline (fail only < 78%).

### Step 4 — Close out
Update docs/54 §9 P2c step-4 status with numbers (VRAM/rank, per-token cost
measured, MTP acceptance); register/close the defect in docs/50; record the
launch command + key log lines.

## 7. Constraints (non-negotiable)

- **Worktree only:** no edits in `/home/intel/ninfer/repo` outside your
  worktree, ever (§2).
- **Live end-to-end before "done":** a step touching a server-facing path is
  not complete until the real server runs it — spin up, execute the code path
  with its dependencies, verify the response. Kernel-level parity alone does
  not count.
- **No damage:** never leave uncommitted or untested code; commit per step;
  keep the tree buildable at every commit.
- **GPU/server:** one live server at a time; port 8091 serves an active
  conversation — swap only with explicit user permission (§2 protocol).
- Do not touch I8/BF16 KV paths (KVarN is opt-in via `--kv-dtype`); no
  staging/shadow pass; no int8 QK tensor cores for KVarN.

## 8. Definition of done

1. Steps 1–4 committed to `wo/kvarn-live` with passing tests at each step —
   including the live-path test (T16) for every server-facing step.
2. **Live proof:** a fresh 250k KVarN launch serves T16 and the must-pass
   subset green; launch command recorded in the report.
3. Functional end state: LAUNCH.md "250k context (KVarN KV cache)" works as
   written, no step-4 gate note needed.
4. Measurement data committed (VRAM/rank, MTP acceptance, per-token cost).
5. Report format when done: one paragraph per step + the key numbers table.
