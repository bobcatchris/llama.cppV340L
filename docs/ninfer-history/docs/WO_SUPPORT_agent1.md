# WO-SUPPORT-1 — Window prep + carry-in census + warm-margin cell: agent1 work order

**Status:** CURRENT — issued by successor chair 2026-09-13 ~15:0xZ, template: `docs/99_agent_work_order_template.md` (structure adapted; AMD-line doc placement per AGENTS.md).
**Mission:** compress the final path to TP4's 4-card first-light: own the G-AMD-18 window run-of-show BEFORE the window exists, support the item-7 carry-in hunt with a zero-card buffer-census, and bank a warm near-capacity margin datum. End state the user sees: one command fires the 4-card window; the carry-in hunt has a citation-complete suspect map; the 2 MiB-margin question moves from "refused-then-launched-once-cold" to "measured warm behavior".

Read this document fully before writing anything.

---

## 1. Context (60-second version)

- TP2's last open bug is **per-request nondeterminism** (verdict row `1d0ff3c6` on `origin/amd/t3-wip`; canonical): 5/5 distinct greedy outputs from ONE server. Sampler window exonerated for 4/5 forks; live class = **upstream carry-in**: recycled pages read before first write (workspace 96 + padding 160 + staging 200 MiB are the neighborhood). agent4 owns the hunt seat — you SUPPORT, never adjudicate.
- TP4: A-series merged (`516883da` era), gates blessed+landed (`2b5d948c`), transport decided (RCCL-first, `docs/amd/WO_TP4_all_lanes.md` §B), B-1 harness on main (`tools/v340l/b1_rccl_ar_sweep.hip` + runner). Remaining: A-4 (agent4), then ONE 4-card window ending in G-AMD-18 first-light. Nobody has assembled the window's run-of-show. That's Task A.
- **Already built and verified (do not redo):** kit v6 banked-path discipline (`results/amd/p3/G3_bringup_kit.sh` @ agent4 ref); G-AMD-26 cold near-capacity GREEN (`51dc3dd6`, bin BANKED `/home/chris/artifacts_bin/ninfer-serve_0c62ffdfd402a4b8.bin`, full sha in record `21a095e8`); era+new-era banks (record file); B-1 runner (agent5, compile-verified by chair); PG-1 gate + 11/11 + 4/4 falsifiers at merged tip; `docs/amd/BOOT_LAUNCH_RUNBOOK.md` §3 fast path.

**What you are doing:** Task A (window runbook), Task B (carry-in census), Task C (warm-margin boot, grant-gated), in that order.

## 2. Environment & build/test

- Repo root: `/home/chris/dual_5060_ti_ninfer` (remote github, branch **`amd/main`** = the AMD line's integration branch; `/home/intel/ninfer` is the OTHER line's host root — not this box).
- **Work protocol (MANDATORY):** your own worktree + branch, never the shared checkout (standing law f49dfe4a):
  ```bash
  cd /home/chris/dual_5060_ti_ninfer
  git worktree add /home/chris/worktrees/amd-wo-agent1-support -b amd/wo-agent1-support
  ```
  Worktree creation is FORENSICS-BARRIER-FREE (new ref, moves nothing; the confound-registry freeze concerned `fetch --prune` and moved refs) — but announce the branch name in your first commit message so agent3's registry subtracts it. No full build needed for Tasks A/B; Task C uses the BANKED binary (no build at all, by design).
- **Communication (MANDATORY, user law 2026-09-13):** NO channel posts anywhere — the hub REJECTS them (`AGENTS.md` COMM LAW). intercom to lane NAMES: `coordinator` (chair), `agent4` (hunt seat), `agent5`, `agent3`, `agent2`. Gemini is hub-direct via comm_send ONLY. Receipts = commit shas, never prose. Cite banked paths + sha prefixes, never `/tmp` names.
- **GPU (Task C only):** written grant from chair BEFORE any spawn. dev2,3 pair, `HIP_VISIBLE_DEVICES=2,3` + `--devices 0,1` (mask-ordinal law). Manifest row committed BEFORE spawn, release row AFTER (death included). Own-pid kills only, `trap ... EXIT INT TERM`. KFD tab-tolerant precheck + raw dump, mandatory.

**Test gates (your work order's falsifiers):**
- Task A: runbook must be self-falsifying — every instrument path in it resolves `ls`/`git cat-file -e` GREEN at tip; one deliberate typo must make the check print RED (the "status docs ship their falsifier" law).
- Task B: every census claim carries `file:line` at a NAMED ref; a claim you cannot re-derive with one `grep`/`sed` does not go in the doc.
- Task C: cell exit-code truth table: warm+fit → LAUNCH (exit 0, margin MiB printed); if allocator refuses at the proven floor + 2 MiB, that IS the datum (report, don't retry). No estimated refusal may block — VRAM LAW.

## 3. Scope

IN: G-AMD-18 window run-of-show doc; carry-in candidate census (read-only); warm near-capacity margin boot (granted).
OUT: the hunt's adjudication (agent4's seat); any `src/` edit; A-4; B-1 execution (agent5's, rides the window); anything on `origin/main` (NVIDIA line — do not push, do not merge the wrong direction: main → amd/main only).

## 4. Deliverables

- `docs/amd/G18_WINDOW_RUNBOOK_agent1.md` — the ordered window: (1) build A-merged tip (agent4 lane or yours, NAMED tree), bank bin per runbook §4; (2) G-AMD-18 first-light cell (argv per kit v6, world=4 construction, fail-loud NCCL admissible); (3) B-1 cell (runner `tools/v340l/run_b1_rccl_ar_sweep.sh`, own stamp, thresholds 120/240 µs in row); (4) warm-margin cell if Task C unexecuted; each step = exact command + expected artifact path + release-row line. Chair's window announcement quotes this doc verbatim.
- `docs/amd/CARRY_IN_CENSUS_agent1.md` — every buffer on the decode path that is (a) allocated once per boot, (b) written per-request, (c) read before guaranteed-write at first logits. Start: `tp2_backend.cpp:768` comment thread, `gdn_gating_proj_gemm_mma.cuh:275/:307/:330` slice coverage, arena park/restore paths, `kv_bytes_per_token=18496` ring init, workspace 96 MiB consumers. Format: suspect | file:line@ref | read-before-write? Y/N/unverified | what would falsify. HAND TO agent4 as support; they decide.
- `results/amd/p3/G17w_*` (Task C): manifest row, serve log, response, verdict row (warm cycle class named, margin printout inline, banked-path citations).

## 5. Design decisions (FINAL)

- Support-not-adjudicate on item 7 — REJECTED alternative: agent1 taking the hunt seat; revisit only if agent4's context dies mid-hunt (chair rules).
- Bank-based Task C binary (no build) — REJECTED: rebuild era at your lane (wastes a disk-tight box's build slot); revisit if 0c62ffdf bank hash-checks wrong.
- Census is support input, not a verdict — no patch proposals in Task B (board law: no patch before the decisive measurement).

## 6. Execution order

1. Read: this doc, `docs/amd/RESTART_2026-09-13/COORDINATOR-SUCCESSOR.md`, runbook §2–4, `T3i7r1_verdict_release_row.md` (`origin/amd/t3-wip`), G-AMD-26 rows (`51dc3dd6`). Commit: "agent1 seated, reads done" (docs commit if you note anything stale).
2. Task A. Commit per section; test = falsifier clause §2. Push.
3. Task B. Commit; hand-off line to `agent4` via intercom (one line + sha). Push.
4. Request Task C grant in writing (host:pair, window length). I ack one line. Then boot per manifest law; release row either way. Push.
5. Report done or blocked — one line per item.

## 7. Reporting discipline

Every number: predicate + source row + hash algorithm named. Cycle class on every TPS/margin row. Counts name predicates. If main moved under you, re-derive tips before citing (kit's own law).

## 8. Definition of done

Task A doc lands merged-eligible (chair verifies paths resolve + falsifier RED demo in the commit msg); Task B census pushed with every row re-derivable at named refs; Task C row banked with warm margin measured or allocator refusal captured as datum; zero channel posts; zero foreign-pid kills; zero `src/` writes; disk never below 10 G at YOUR steps (measure before/after; if a build is unavoidable, report size cost first — 16 G and one-full-build headroom is the current measured state).
