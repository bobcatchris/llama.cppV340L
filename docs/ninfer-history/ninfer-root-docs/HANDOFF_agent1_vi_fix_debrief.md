# HANDOFF — agent1 (vi) lane: ROOT CAUSE FIXED + VERIFIED, S2 bf16 sizing is the last step

**Read this first if you are continuing the (vi) lane.** Written 2026-09-06 ~12:5xZ by the
session that found, fixed, and verified the defect. Supersedes the *hunt* debrief
(`HANDOFF_agent1_vi_debrief.md` — keep it for the instrument suite + elimination chain; its
defect description is now HISTORICAL: the defect is FIXED).

---

## 0. STATE AT HANDOFF (30-second version)

- **Branch:** `wo/kvarn-multibatch`. **Fix commit: `a4066f5d`.** Evidence through `cf4a9465`.
- **Verified:** cache transition clean, role-keyed hashes clean, responses byte-identical
  across lane orders, **S2 diag int8 exit 0** (post-fix, committed artifacts).
- **The ONE remaining critical-path step:** S2 diag on **bf16**. Two attempts failed with
  VRAM-preflight OOM (server refuses to start — NOT an order-dependence failure; see §4 for
  the exact sizing math and the ready-to-run command).
  **→ DONE 2026-09-06 ~13:05Z, successor session: exit 0 at 40000/45000, commit `b1e99a2c`**
  (A1==A2==B1; artifacts `results/phase_c/s2diag_fix_bf16_130048/`). Critical path closed.
- **GPUs:** released, 15 MiB both. **Disk:** ~13 G free (95% full) — check `df -h /`.
- **Peers:** coordinator (intercom) drives grants/merges; A2 (`agent2`) = dflash2 lane, CPU-only
  right now; **gemini owns the formal tests** (agent_comm mesh, may be down — coordinator relays).

## 1. THE (vi) DEFECT — root cause, one paragraph

On the batched MultiBatch runner (i8/bf16 kv-dtype), the MTP-layer KV-cache prefill append did
not target the lane's own allocation row. **Two independent MTP-cache writers ran per lane:**

- **Writer #1** — `mtp_prefill_chunk`'s append (`text_context_impl.h:1348`,
  `ops::gqa_kv_append(kn, v, positions, mtp_kv_.layer_view(0), s)`), fired from inside the
  **text prefill's last chunk** (call sites text_context_impl.h:2727/:2747). It keyed the cache
  row off the currently-bound `mtp_kv_` **view**.
- **Writer #2** — `mtp_forward_tail`'s non-batched TP arm fused-append
  (`ops::gqa_attention(..., io_.backend_kv_table_row, ..., batch_mtp_kv_->batch_layer_view(0))`,
  text_context_impl.h:~1102), fired from the runner's per-lane `mtp_forward_batch` loop
  (tp2_backend.cpp:2750). It keyed the row off the **scalar** `io_.backend_kv_table_row`.

G3 (docs/154 Phase C) wrote `prefill_row = b` into the scalar before each lane's prefill,
covering **writer #2** — but writer #1 uses the **view**, and `set_mtp_view(lane b)` only ran
**after** lane b's text prefill (old location tp2_backend.cpp:2719, inside the MTP-layer
prefill section). So writer #1 executed while `mtp_kv_` still held the **previous lane's**
view: lane 1's prefill appended into **row 0**, clobbering lane 0's MTP history. Lane 1 is
always prefilled second, so row 0 always ended up holding lane 1's content. The round-1 align
forward then read lane 0's row expecting lane 0's history and got lane 1's — order-dependent
hidden → order-dependent drafts → the (vi) order-dependence. Deterministic; i8/bf16 only;
kvarn immune (its MTP arm uses per-lane tiles, never these pool rows).

## 2. THE FIX (a4066f5d, ~14 lines incl. comment)

In `tp2_backend.cpp` per-lane prefill loop, next to G3's `prefill_row` writes (~:2656-2690):
hoist `set_mtp_view(b==0 ? st.mtp_view : st.mtp_lane_views[b-1])` + `publish_mapping` to
**before the text prefill** (writer #1's window). The :2719 rebind before the MTP-layer prefill
loop stays (idempotent). No behavior change for kvarn (branch untouched) or single-lane.

## 3. VERIFICATION DONE (evidence `cf4a9465`)

1. **Cache transition table** (win30, `results/phase_c/hashpt_121420_win30/`): row0-page0
   keeps lane0's content through lane1's prefill in BOTH runs; row1-page0 = lane1's ✓.
   Pre-fix, row0-page0 was **58.65%** different after lane1's prefill. (The 0.02-0.03% residual
   vs my CPU reference quantizer is the emulation's rounding floor = "identical".)
2. **Role-keyed hash points** (win30): B2c_verifyhid, B2b_alignout, B4_accept, B5_state match
   at ALL rounds; round-2 B1_pack drafts MATCH role-keyed (pre-fix they differed).
3. **End-to-end**: greedy + sampled responses byte-identical across lane orders.
4. **S2 diag int8: exit 0** (`results/phase_c/s2_order_diag_fix_i8.log`; artifacts
   `results/phase_c/s2diag_fix_122158/`; A1==A2 (repro) and A1==B1 (order flip); acceptance
   stats mirror 0.60/0.67 across lanes).

## 4. YOUR FIRST HOUR (exact commands, in order)

```bash
cd /home/intel/ninfer/worktrees/wo-kvarn-multibatch
git log --oneline -3                    # expect cf4a9465 on top; fix = a4066f5d
git status --short                      # clean
nvidia-smi --query-gpu=index,memory.used --format=csv | tail -2   # expect 15 MiB both

# bf16 S2 diag — the sizing math (from the two failed preflights, VRAM per rank):
#   required = 13269 + 0.0618 MiB/token(max_context); device usable = 16310 MiB
#   80000 tok -> 18211 MiB FAIL | 60000 tok -> 16976 MiB FAIL | 40000 tok -> ~15741 MiB OK
# So 40000/45000 is the tested-safe pair; 48000/50000 is the untested edge. Use 40000.
tools/smoke/diag/gpu_guard.sh gpu_refuse_if_busy && echo GUARD-CLEAR

nohup env KV_DTYPE=bf16 MAX_CONTEXT=40000 KV_CAPACITY=45000 \
    bash tools/smoke/diag/s2_order_diag.sh > results/phase_c/s2_order_diag_fix_bf16.log 2>&1 &
# (launch the nohup ALONE — never append & to an && chain; see gotcha 1)

sleep 430; tail -12 results/phase_c/s2_order_diag_fix_bf16.log
# EXPECT: "A1 vs A2 (same order): SAME" / "A1 vs B1 (order flip): SAME" / exit=0
# IF you see "FATAL: server died" + a [preflight] line instead: it is an OOM, NOT an
# order-dependence failure — check the MiB math and lower MAX_CONTEXT further.

# if exit 0: commit + release
git add results/phase_c/s2_order_diag_fix_bf16.log && \
  git commit -m "evidence: (vi) S2 diag bf16 post-fix — exit 0 (40000/45000)"
nvidia-smi --query-gpu=index,memory.used --format=csv | tail -2   # verify 15 MiB = released
```

Then coordinate (one message each):
- **gemini** (via coordinator if the mesh is down): owns the **(vi) formal test + bf16/i8 S2
  re-gating test**. Fixture = the two-prompt pair both scripts use: greedy temp=0 + sampled
  temp=1.0, 300 ms stagger, 500 ms batch window, `--seed 20260905`, max-concurrency 2.
- **coordinator**: owns flipping the serve-CI S2 cell from kvarn-only to bf16/i8. Hand over:
  fix `a4066f5d`, both S2 logs, this debrief path.

## 5. OPEN ITEMS (none blocking; all flagged for the re-gate discussion)

1. **bf16 diag result unknown at handoff** — the two failed attempts were VRAM preflights
   (`required 18211/16976 MiB > usable 16310`), i.e. the server never started; the "exit=1"
   verdicts in those logs are empty-output comparisons, NOT order-dependence. Do not cite them
   as failures of the fix.
   **→ CLOSED: third run at 40000/45000 exit 0 (commit `b1e99a2c`, 2026-09-06 ~13:05Z);**
   **role-keyed acceptance stats swap with leader role, not lane position.**
2. **B3_draft r1 `arh/prop` hash diff in win30 is sub-visible but unexplained**: everything
   downstream matches role-paired (B3c r1, round-2 drafts, B4/B5 all rounds, final text).
   Suspect probe timing (B3_draft fprintf may read `mb_arh`/`mb_prop` staging before the
   select-kernel/allreduce_argmax finalizes). Look before gemini's formal test.
3. **`first_diff=29` artifacts in win30 parses are a PARSER bug, not a defect**: round 29 is
   where the sampled lane finishes (B shrinks 2→1) and the analyzer's flat-column
   normalization (`col - lane*4`) breaks on shrunken batches. Compare only rounds where BOTH
   lanes are alive (win30's sampled lane finished at round 29 — see B1_pack cmp counts 37
   vs 28).
4. **Probe hygiene debt**: TH/PARTHASH/MTPKV/MTPC/MTPP/MTPROW/APPENDLOG remain in the tree,
   all env-gated and inert. Keep or drop at fold time (drop-or-keep is free); don't refactor.

## 6. GOTCHAS (each cost real time this session — do not repeat)

1. **`cmd && nohup cmd2 &` backgrounds the WHOLE chain** — build runs, probe silently doesn't
   (or the commit silently doesn't). Separate statements; launch background jobs ALONE.
2. **Tensor `ne[0]` is the INNERMOST (d-fastest) dim.** KV-plane page stride = 32768 B
   (64 tokens × 2 local heads × 256 dims, I8, TP-split). `block_tables` is
   `[logical_pages][table_rows]` (page-major). `allocation.page_ids()` is the ONLY
   authoritative source for a lane's physical pages. My MTPP/MTPROW probes were wrong until I
   keyed dumps on `page_ids()` with contiguous per-(lane,page) slices.
3. **Both TP ranks share staging** — per-lane probes need `if (rank == 0)` AND the rank in the
   filename (`kn_r%d_...`), or the ranks' writes collide.
4. **Probe output files must live under $OUT** — the probe script rotates `*_A/_B` by globbing
   `$OUT`; an externally-forced `MTPC_DUMP_DIR` silently broke rotation (win24 lost runA's dump).
5. **`getenv("X")` is true for `X=0`** — treat literal "0" as off in every new gate.
6. **Every probe env var must be added to `hashpt_order_probe.sh`'s launch env AND its A/B
   rotation list** — win26 ran without the MTPROW table because the script never set the env
   the code gated on.
7. **Role pairing is law** (hunt debrief §4.2): sampled = A.lane1 vs B.lane0; greedy = A.lane0
   vs B.lane1 (leader = lane 0 = first fired). Normalize storage coords (`slot/slt/ringbase`,
   B2_verify `col`) before comparing. B1b's `ids` sub-hash is whole-tensor by design — never
   compare it role-paired; use `arpos`/`arvalid`.
8. **bf16 VRAM preflight**: per-rank budget ≈ 16310 MiB; required ≈ 13269 + 0.0618 MiB/token.
   int8/kvarn default 80000/100000 fits; bf16 needs 40000/45000 (tested math) or smaller.
9. **Disk 95% full** — `df -h /` before builds; evidence dirs ~10 MB each; coordinator
   previously freed 20 G by trimming `build/tests` + `build/bench` (rebuildable).
10. **Never pkill ninfer-serve globally** — only PIDs you started; `gpu_guard.sh` is the
    protocol; claim cards only after a written coordinator grant + fresh re-guard.

## 7. KEY FILES (map)

| file | role |
|---|---|
| `tp2_backend.cpp:2656-2690` | **THE FIX** (hoisted MTP view rebind + the why-comment) |
| `tp2_backend.cpp:2719-2755` | per-lane MTP-layer prefill loop (writer #2 site, unchanged) |
| `text_context_impl.h:1348` | writer #1 append (mtp_prefill_chunk; APPENDLOG probe above it) |
| `text_context_impl.h:2727/2747` | text-prefill last-chunk mtp_prefill_chunk call sites |
| `text_context_impl.h:1002-1121` | TH tail-bisection probes (`NINFER_MB_TAILHASH`) |
| `text_context_impl.h:1275-1315` | MTPKV pre-quant kn/v hashes + raw kn dump (`NINFER_MB_MTPKV`) |
| `gqa_attention.cpp` (SmallT route ~:562) | PARTHASH probe (`NINFER_MB_PARTHASH`) |
| `tp2_backend.cpp:3755-3800` | B1b_alignids probe (`NINFER_MB_HASHPT`) |
| `tp2_backend.cpp:~3805-3860` | MTPC/MTPP/MTPROW probes (`MTPC_DUMP_DIR`, `NINFER_MB_MTPROW`) |
| `tools/smoke/diag/hashpt_order_probe.sh` | two-order capture; ALL probe envs wired + A/B rotation |
| `tools/smoke/diag/s2_order_diag.sh` | S2 discriminator (`KV_DTYPE`, `MAX_CONTEXT`, `KV_CAPACITY`; exit 0 = pass) |
| `results/phase_c/vlhash_layer_analysis.py` | hashpt analyzer (`--steps` mode fixed; mind open item 3) |
| `results/phase_c/vlhash_block_compare.py` | block-aligned VLHASH comparator (parser-artifact demo) |
| `docs/vi_step0_reconciliation.md` | full evidence trail §1-§11 (every measurement, in order) |
| `docs/HANDOFF_agent1_vi_debrief.md` | the HUNT debrief (instrument suite; defect now historical) |

## 8. EVIDENCE INDEX (preserved, newest last)

| dir | proves |
|---|---|
| `hashpt_051417_vlhash/`, `hashpt_053551_steps/` | pre-session captures (anchors of the hunt debrief) |
| `hashpt_070449_window9/` | fixed probes: align inputs clean r1, alignout dirty r1 |
| `hashpt_072602_window10/` | TH bisection: divergence INSIDE the align forward |
| `hashpt_085859_win17/`, `hashpt_091028_win18/` | MTP cache planes differ; same request's page-0 K/V differs BY LANE |
| `hashpt_104429_win24/` | (kn-dump rotation broken here — gotcha 6.4) |
| `hashpt_105219_win25/` | byte matrix: rows hold SECOND-prefilled request's content; CPU-quantizer floor 0.02% |
| `hashpt_111944_win27/` | pool addressing decoded; mtprow v1 files INVALID (documented) |
| `hashpt_115052_win29/` | transition table: lane1's prefill writes BOTH rows; APPENDLOG: all appends on row0's table |
| `hashpt_121420_win30/` | **POST-FIX**: transition clean; hashes clean; text identical |
| `s2diag_fix_122158/` + `s2_order_diag_fix_i8.log` | S2 diag int8 exit 0 |
| `s2_order_diag_fix_bf16.log` | bf16 attempts 1-2 = VRAM OOM only; **attempt 3 = exit 0 (open item 1 closed, `b1e99a2c`)** |

## 9. WHO'S WHO

- **coordinator** (intercom): written GPU grants, merge plan, doc-number assignment (never
  self-assign), relays to gemini. Protocol: claim only after grant + re-guard; release
  immediately after; report one line per commit.
- **A2** (`agent2`): dflash2 lane, 3 commits on `wo/dflash2-scope` (d7f728c6, bd51d8bc,
  27c6c1ab). Merge notes: `kMaxTokens` 8→16 (`one_shot_argmax.h`), staging 32→48 MiB
  (`tp2_backend.cpp:96`), `drafting` keying at `tp2_backend.cpp:5020`. No overlap with the fix
  region (:2656-2690) or probes (:3755-3860). Merge-never-rebase; whoever lands second rebases.
- **gemini**: formal tests (vi + S2 re-gate) via agent_comm mesh.
