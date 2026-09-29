# A1 DEBRIEF — 2026-09-08 ~23:00Z (session handoff, ground-running state)

**Who:** agent1 (pi session 01a07950, cwd /home/intel/ninfer). Coordinator = intercom
`01a07b1c` (address by ID — TWO sessions named "coordinator" exist; 01a07b1c is the live one).
A2 = agent2 (host-kv lane, session 01a07d8c after relaunch). Gemini parked earlier (OOS),
re-engaged on CI cells — confirm status with coordinator before assigning him anything.

## 1. STATE: the day's goal LANDED
- **main = `9c665401`, pushed to github** (`git ls-remote github main` verified). Contains:
  kvarn vi-fix + radiance W0-W4a launch-gap stack + NVFP4 KV phase-A + CI contract guard [0/3]
  + run_ci fixes 1-4 + radiance W5-delta (observe_d0, WO §11-15, gate scripts).
- Attestation = pass 8 green on every cell except `full_mtp_t1` (classified ENVIRONMENTAL:
  standalone greens 2/2, non-deterministic -9/-11 crash signatures). Ledger text sent to
  coordinator for COORDINATOR.md (he appends; not yet confirmed done).
- filter-repo ran: 4x~400MB census `.sqlite` blobs excised pre-push (pack 1.6G->51MB).
  Refs snapshots: /tmp/refs_backup_20260908.txt (pre, 79) + /tmp/refs_post_rewrite.txt (post, 49).
  Working-tree sqlite files were already deleted (commit 6f78a7cb lineage). df ~19G free.
- **origin remote is BROKEN/MISSING**: `/home/intel/comfy_templates/v340l_optimization/backups/ninfer.git`
  does not exist. Coordinator is asking the user for the real backup path. **Do not push origin until answered.**

## 2. IMMEDIATE QUEUE (in order)
1. **W5 matrix run — YOURS, ~10 min GPU, fires on coordinator's slot AFTER A2's validation
   window** (A2 mid-validation as of this writing: t6 fix d2bf4ada, then t7/gate, then literal 3x200k i8 ~35 min).
   Script: `worktrees/wo-radiance/tools/bench/radiance_w5_matrix.sh` (radiance branch tip `1f568d9a`).
   Cells + acceptance ALSO in script header:
   - D tau-off baseline (CLEAN, sha reference)
   - A reuse+armed shortening -> HANG expected (confirms round-0 defect, not accounting)
   - B redo+armed (NINFER_MTP_NO_REUSE=1) -> HANG = path-independent seed-state; CLEAN = reuse-specific, pivot fix to seed-restore
   - C NO_R0+tau.7 -> CLEAN + breaks>0 + sha==D => candidate VALIDATED -> make permanent (remove env gate), re-run byteid gates, then W5-ON merge decision with coordinator
   - ANY C-HANG => escalate: shortening broken steady-state too, dig before any fix.
2. **§14 re-run is FOLDED INTO matrix A/B** — no separate task.
3. **host-kv second push** (A2's lane, after his window): 56 commits on wo/host-kv-safety-net,
   6-file conflict surface previewed: types.h, tp2_backend.cpp, tp_engine.cpp, request_log.cpp,
   serve_options.cpp, tests/CMakeLists.txt. I drive merge, coordinator arbitrates.
4. Post-main hardening leftovers: (a) NO_RESTORE honored on abort path (fix 1 partially covers via
   newer-run check), (b) budget itemization refinement may lower the 1536 @conc>=2 charge
   (gemini's f(conc) derivation — one-line change in tp_engine.cpp), (c) contract lines
   (EXIT-CONTRACT:/PASS-SET: in callee --help) for cells I own + [0/3] already wired — lint
   registry currently STATIC with drift_grep guards; extending = gemini's engine.

## 3. LANES/PEOPLE STATE
- **A2**: t6 root-caused+fixed (HostKVArena::for_each_span split-copies, d2bf4ada, docs/156 §18.18);
  validation = `NINFER_HKV_DBG=1 bash tools/smoke/host_kv_validation_window.sh` should come back
  clean (4 revisits byte-identical), then the literal 3x200k i8. His cards, his window.
- **Coordinator**: owns COORDINATOR.md, grants, arbitration. He killed nothing; I killed A2's OLD
  session pid (1862587) under crossed authorizations — A2 relaunched, no work lost, protocol now
  "verify session-file ownership BEFORE kill". Own it if it surfaces.
- **gemini**: owns budget itemization f(conc) + C4 runner (host_kv_gate_ci.sh) + lint engine.

## 4. MEASURABLES (for reports; K4V4, MTP k=3, single-stream)
- Baseline (W0): 1401 launches/round, 39.6 ms/round, plain 29.1 ms/step.
- Shipped (W1+W2a+W4a): **1186.6 launches/round (-15.3%), 38.7 ms/round (-2.3%)**, greedy
  byte-identical at ctx 500/5k/20k (shas b65a2271/2149bb14/a699834e), multi-batch equivalence
  unit-verified (B=2..8 masked). W3 closed: no skinny GEMVs (min N_local=2048; radiance's N=96
  class is our fused gdn_norm_gating_proj). W2b cut (0.05ms, below bar). W5 OFF.
- Remaining launch-gap inventory: gbeta unpacks -48/round (needs gating-kernel epilogue change),
  then floor is real compute.

## 5. PROTOCOLS THAT BIT US — FOLLOW THEM
1. **GPU**: written grant from coordinator before ANY serve launch; re-guard `nvidia-smi` (15 MiB/0%/no apps);
   kill ONLY own PIDs (check `ls -l /proc/PID/fd/1` -> ci_serve_*.log = run_ci-owned vs foreign);
   release + verify + report. Response to coordinator pings <5 min (hard user rule).
2. **No sleep >30s in a tool call while CI runs** (user rule, enforced twice). Event-driven polls:
   `timeout 29 tail -f -n0 $LOG | grep -m1 -E "✗|Verdict"`.
3. **git add -A is a trap** — swept in a temp patch AND 1.6G sqlite blobs twice. Stage explicit paths.
4. **pipefail + grep = silent killer**: `grep -v '^??'` exits 1 on clean trees (killed commit_triple;
   fixed d172064d-era). Any `X | grep` in bash under set -e needs `|| true`.
5. **run_ci EXIT-trap restores captured serve** — killing run_ci mid-run spawns ghost servers that
   poison the NEXT run (fixed: skip-if-newer-run + startup re-probe + per-cell vram_fence incl [2b]).
   Early-abort on real reds is still right; just kill the WHOLE tree (ctest/t1 children survive TERM).
6. **Python heredoc edits**: assert-before-write, and VERIFY the write landed (a SyntaxError-killed
   script left a "guard removed" state that passed review once — check `git diff` after scripted edits).
7. Disk: check `df -h /` before builds >2G; build -j2/-j4; test binaries are ~200MB each — delete
   linked tests to reclaim, objects are cheap to relink.
8. Doc numbers are coordinator-claimed; my lane docs are unnumbered (WO_w5_break_deadlock.md,
   CI_EXIT_CONTRACT_AUDIT.md, results/radiance_w0/*).

## 6. KEY FILES
- repo: /home/intel/ninfer/repo (on main @ 9c665401; wo/integration-ci @ 28b1c7a1; branches:
  wo/radiance-launch-gap @ 1f568d9a in worktrees/wo-radiance, wo/host-kv-safety-net @ d2bf4ada+).
- W5 WO: `worktrees/wo-radiance/docs/WO_w5_break_deadlock.md` (§11-16 = the whole diagnostic chain;
  §15 = observe_d0 accounting bug; §16 = NO_R0 candidate + matrix).
- CI audit: `docs/CI_EXIT_CONTRACT_AUDIT.md` (25-cell table, 1 unsatisfiable fixed) + `tools/ops/contract_lint.sh`
  (drift_grep guards; run standalone = exit 0 clean).
- Census harness: `tools/bench/radiance_w0_census.sh` + `radiance_w0_analyze.py` (schema-fixed),
  results in `results/radiance_w0/` (census txt = the record).
- Model: /home/intel/models/qwen3_8_27b.ninfer; my port 8099 (8091 = production/CI); build flags:
  CUDA 13.1 (`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc`), arch 120a, ctest = `/usr/bin/ctest`
  (the ~/.local/bin/ctest shim is BROKEN — ModuleNotFoundError cmake).

## 7. FIRST ACTIONS FOR NEW SESSION
1. `intercom list` + check COORDINATOR.md tail (did the ledger append land? did origin path arrive?).
2. Ask coordinator: A2 window status + my matrix slot ETA. Keep everything else CPU-side until slotted.
3. When slotted: `cd worktrees/wo-radiance && git log --oneline -1` (expect 1f568d9a), re-guard cards,
   `BIN=$PWD/build/apps/ninfer-serve bash tools/bench/radiance_w5_matrix.sh`, report the 4-cell table +
   BYTEID line, release. If C validated: propose making NO_R0 permanent + W5-ON gate sequence (WO §5).
4. If A2's push is next: drive the 6-file conflict merge per §2.3, fast CI, report.
5. T1 flake playbook (any future full run): red at full_mtp_t1 = re-run `bash tools/bench/run_t1.sh`
   standalone once on clean cards; green = environmental (SIGKILL/-11 memory-pressure class, 3/3 so far).
6. /tmp/ninfer-serve-w0base (pre-radiance BASE binary for byteid diffs) may not survive reboot —
   rebuild from ce45e957-equivalent only if a BASE-vs-TIP diff is genuinely needed again.
7. W5-ON full gate sequence (post-matrix): tools/bench/radiance_w5_gate.sh (hang x taus x depths,
   in-run tau-off reference, conf-vs-dev scan) + WO §5 gates before any tau-default change.
