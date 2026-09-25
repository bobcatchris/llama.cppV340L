# 107 — mtp-adaptive + ngram-mod-seed: land & verify (evening work order)

**Status:** CURRENT — for one implementer agent, tonight (after the ~19:00 GPU
slot opens). Short order on purpose: the branch is IMPLEMENTED and CPU-proven;
this is a rebase + live-proof + land job. Read fully before starting.
NOTE (10:45, schedule slip): rebase onto the CURRENT wo/kvarn-hold head
(9a31a070, includes baseline v1 + docs/110/111) instead of waiting for the
~14:00 merge — same conflict profile, saves 2.5h. Plan owner merges
wo/kvarn-hold → main before your GPU slot, so the branch ends on main anyway.

**Mission:** Take `wo/mtp-adaptive` from "CPU-proven, live parked" to landed on
main: rebase onto post-merge main, rebuild, prove live behavior (byte-identity
vs main + acceptance/t/s delta of the ngram seed), full CI, report. Plan owner
merges.

---

## 1. Context (60-second version)

The branch (12 commits from `c63b2443`) carries two features, all CPU-proven
(green ctest, evidence packs committed to `results/`):
- **docs/69** — adaptive MTP draft depth (llama.cpp #27210) + ngram-mod draft
  pool (llama.cpp #19164) wired into the decode round loop, controller kept
  honest under ngram wins.
- **docs/80** — `--ngram-mod-seed <file>`: replay a project token stream into
  the shared draft pool at startup (docs/98 §C). Loader + flag + wiring.

What is NOT done: **any live/server verification** (docs/80 status: "live
baseline/seeded delta parked"; the "do not start the server" restriction in
docs/80 §2 was because the GPU belonged to the kvarn-hold agent — that
restriction is LIFTED for this work order, under the GPU queue rule in §2).

**Why the live gate is strict (read before step 1b):** speculative decoding is
lossless only if verify+sample is unchanged. Drafts change *speed*, not output —
*except* when variable draft depth changes verify batch shape, which is exactly
the D-21 bug class (batched-vs-single attention ~1 ulp → argmax flips). D-21
was fixed for the GDN gate (1d3a0533) and for fixed MTP widths (H5/battery);
**adaptive depth = variable T per round is a new T pattern nobody has
byte-identity-verified.** So the gate below is byte-identity, not tolerance.
If it diverges, that is a finding, not a ship-it.

## 2. Environment & build/test

- Worktree: `~/ninfer/worktrees/wo-mtp-adaptive` (branch `wo/mtp-adaptive`) —
  already exists; **build dir was deleted, rebuild from scratch.**
- Work protocol: work ONLY in that worktree; never the main tree; commit per
  step to `wo/mtp-adaptive`; push. **No merges** — plan owner merges.
- Rebuild: `cmake -S . -B build && cmake --build build -j 16` (CUDA `sm_120a`).
- Unit tests: **`/usr/bin/ctest`** from `build/` (PATH `ctest` is broken).
  Green set = whatever the docs/80 evidence pack (commit `4e261d29`) ran.
- **GPU QUEUE (tonight):** your slot opens ~19:00, after the `wo/kv-uniform`
  agent's GPU validation commits its step-3/4 results (check
  `git log wo/kv-uniform` or ask the plan owner). CPU steps (0) run BEFORE
  the slot, starting ~17:00 right after the kvarn-hold merge lands.
  One live server at a time; `pkill -x ninfer-serve` (NEVER `pkill -f`);
  `pgrep -x ninfer-serve` + `nvidia-smi` before every launch; leave the server
  restored at the end.
- Server: `bash tools/ops/build_and_serve.sh` conventions; model
  `/home/intel/models/qwen3_8_27b.ninfer`.
- Server tests: `bash tools/ops/run_ci.sh` from YOUR worktree root (relative
  path); `--full` at closeout only.

## 3. Execution order (commit + test each step)

### Step 0 — Rebase + rebuild + ctest (CPU; start ~17:00, before your GPU slot)
Rebase `wo/mtp-adaptive` onto the NEW main (post-kvarn-merge; docs/107/108
will be on main by then). Expected conflicts: `src/runtime/tp2/tp2_backend.cpp`
(kvarn D-21/D-22/D-23 fixes vs your loop wiring) and possibly `tp2_backend.h`.
Resolve by keeping BOTH sides (the D-21 anchor fix must survive verbatim —
`cur_F_mtp = plen` in the MTP branch; the NINFER_D21_DBG block is debug, keep
it too). **If a conflict is semantic, not mechanical — STOP and report before
resolving.** Rebuild, `/usr/bin/ctest` green (evidence-pack set). Commit
`rebase(wo/mtp-adaptive): onto post-kvarn-merge main (docs/107)`.
**Tests:** build green; ctest green; `git log` shows main's merge head as
ancestor.

### Step 1 — Live verification (GPU slot, ~19:00)
On your rebuilt binary:
- **(a) Off-path byte-identity:** greedy decode (temp 0), 3 prompts × 10k ctx,
  192 tokens, NO `--ngram-mod-seed` → compare byte-identical to a fresh
  serve of **new main** (build main in a scratch build dir if needed). Also
  with the pool active but no seed. Save outputs under
  `results/107_live/`. **Any byte diff = STOP, report (D-21-class).**
- **(b) Seeded delta:** serve WITH `--ngram-mod-seed <file>` (the branch's
  own seed artifact — `results/ngram_seed_report.md` names it; if absent,
  build one per docs/98 §C from tracked sources). Same prompt family + the
  6-prompt battery (k=3, temp 0.8 and greedy): record MTP acceptance + t/s
  vs step-1(a). Also run each config twice → **determinism** (identical
  output across runs).
- **(c) D-21 depth canary (required, per docs/119 §11 Option B'):** the
  binary under test must carry the k_buf clamp (`rk = min(rk, 7)` at both
  k_buf sites in tp2_backend.cpp, verify width T = rk+1 <= 8). Across the
  whole (a)+(b) battery, LOG the max verify T actually reached and ASSERT
  it <= 8 (the adaptive `[adaptive] round N depth=D` lines + any ngram
  depth give T=D+1). A max-T >= 9 means a verify round hit MmaUnsplit
  (cols>=9), which is NOT byte-identical to the T=1 Gemv plain
  continuation -> the (a) byte-identity gate would be vacuous even if it
  "passed". **Max verify T >= 9 = STOP, report (D-21-class), do not land.**
- Success = (a) byte-identical; (b) deterministic AND acceptance improves
  (or holds) AND t/s holds/improves; (c) max verify T <= 8. Flat or negative
  delta = report with numbers, **do not land** (seed/prompt mismatch is a
  design question for the plan owner).
Commit results + `live(mtp-adaptive): byte-identity + seeded delta (docs/107)`.

### Step 2 — Full CI
`bash tools/ops/run_ci.sh --full` from your worktree (flag off = default CI
path). Commit report with build commit + model identity.

### Step 3 — Report + status
Update docs/80 status (landed, evidence) + a `results/107_landing_report.md`
(rebase summary, byte-identity verdict, delta table, CI). Commit
`docs(80,107): mtp-adaptive landed`.

## 4. Constraints (non-negotiable)

- Worktree only; no merges; commit per step; keep the tree buildable.
- GPU queue rule is absolute (a premature launch corrupts the other agent's
  run). No GPU before your slot; no killing any server that isn't yours.
- Do not touch: KVarN path, sampler, the D-21 fix itself (keep it verbatim),
  docs/105 scope (GDN gating).
- Byte-identity gate is a GATE: no tolerance, no "close enough", no Option-C
  invocation. Divergence = finding.
- Commit messages name docs/107.

## 5. Definition of done

1. Steps 0-3 committed to `wo/mtp-adaptive` with green tests at each.
2. **Live proof:** byte-identity (off-path, vs new main) + deterministic
   seeded runs + delta table, outputs committed.
3. Full CI 0 fails; report committed with model identity + build commit.
4. docs/80 status = landed-with-evidence.
