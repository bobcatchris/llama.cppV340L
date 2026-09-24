# HANDOFF — agent1 session state (wo-kvarn-multibatch lane, WO 154)

**Read this first if you are a fresh agent1 session on the MultiBatch lane.** Resumption note,
not a work order. Written 2026-09-05 ~15:5xZ by the Phase S/A/B session. The authoritative
running log is `docs/154_multibatch_alldtypes.md` §8 — this handoff points at it; where they
disagree, §8 wins. Work through AGENTS.md standing rules too; nothing here overrides them.

---

## 0. RESUME HERE (2026-09-06 ~06:4xZ state — re-written by the (iii)-(v) session; supersedes the queue below)

**State:** lane branch `wo/kvarn-multibatch` tip `a8d60f08` (pushed), tree clean, GPU idle.
Tonights session delivered (all device-verified, see §8 2026-09-06 entries):

- **S2 kvarn gate CLOSED + S1-i8 negative gate COMPLETE** — deterministic trigger trio (control
  PASS / mutated FAIL / re-applied PASS) via new `tools/smoke/diag/s1_det_trigger.sh`; plus
  `s2_order_diag.sh` re-run with the fix in: exit=0 (order-invariant) → tracked-open S2 k4v2 gate
  CLOSED. `13ed7f0f`.
- **(iii) bf16 OOM** — preflight TP2 runtime reserve, 1536 MiB, **BF16 tiers only** (the
  dtype-blind variant was tried on device and falsely refused i8@100k — retracted); bf16 CI cell
  at 40960 + media shave, 0 failed. `b13d85a5`.
- **(iv) §5 :470 collapse** — route-exact KVarN workspace bound (`detail::
  gqa_attention_kvarn_capacity_bytes`) + `NINFER_WS_PEAK` TC4 instrument (route-local, 5 routes);
  ZERO UNDER across 4 configs incl. batched. `cb9c12af`.
- **(v) materialize_tp mtp/ filter** — 241 MB/rank recovered on no-MTP serve, device-verified
  both directions; DFlash2 serving gets it automatically. `7ff8fd95`.

**OPEN: (vi) fused-append lane-slot order-sensitivity — SITE PINNED, FIX NOT YET WRITTEN.**
Defect: same requests, flipped lane order → sampled text diverges from ~token 3 on bf16/i8
(S2 order-diag exit=2 on both dtypes); kvarn bit-exact (order-diag exit=0; kvarn S2 gate stays
closed/clean). ELIMINATED: attention split boundaries (bf16 MultiBatch small-T kernel + reduce
read in full: per-slot pure), stale inactive-split partials (reduce recomputes active splits),
order-assigned RNG seeds (server --seed inherits deterministically), B==N compact-vs-original
remap (S1 fix covers), verify-forward numerics at rounds 0-1 (accepts byte-matched). PINNED
(ordered D2H reads; the unordered first attempt produced stale-read artifacts): the divergence is
visible in the ROUND-1 ALIGNMENT INPUT hiddens (v_vhid) — ALL columns, both requests — so
contamination enters at or before the round-1 verify hidden stream / its staging reuse. NEXT
INSTRUMENT (bounded, named): hash the round-1 verify hidden PER COLUMN AT WRITE TIME (right after
`target_verify_batch`) + hash the round-0 prefill hidden — splits write-side vs read-side.
Candidate mechanisms: staging-buffer reuse across rounds/lane assignment (first-execution
MultiBatch code, §6 class), or a batch-max-coupled geometry in the fused-append verify hidden
path. The alignment forward itself (`mtp_forward_decode_batch`, `Envelope{maxa,maxa}`) has NOT
been audited kernel-side. Formal test (gemini, TEST SPLIT) + bf16/i8 S2 re-gating wait on the fix.
Evidence: `results/phase_c/hashpt_order{,2,3}/`, `b3_analysis.txt`, probe logs. NOTE: the
`rotation_finding.txt` rotation claim is SUPERSEDED by the ordered-read run; the request-pairing
lesson stands (same lane id across orders = different requests — pair by request ROLE).

**Merge basis (coordinator):** partial-merge — (iv)+(v)+S2 kvarn-only gating are merge-safe; the
(vi) defect is tracked, same class as S1-i8-composition. Coordinator drives the merge autonomously.

**Standing facts (see §3 + §4 + §8):** rebuild after ANY source change (targeted
`cmake --build build --target ninfer-serve -j6`); guard before every launch; single runs,
KEEP_WORK, evidence committed with explicit pathspec; doc numbers from the coordinator; marker
gate v2 before push (expect 0).

---

### 0c. TONIGHT'S QUEUE (superseded by 0d above — kept for the ruling trail)

1. ~~Kernel half / gate collapse / Phase D as originally scoped~~ **SUPERSEDED — §8 2026-09-05 21:2x–23:5xZ entries win.** Reality: the BF16/I8 MultiBatch arms were already compiled (no instantiation campaign); Phase C = runner dtype-genericity, LANDED as e804971f (attn_mix_tp batched non-KVarN arm; :2426 fallback REMOVED — the rule-15 grep targets the REMOVED fallback, EMPTY; the gqa_attention_cached_batched KVarN-only throw STAYS per coordinator ruling (a), bf16/i8 ride the fused-append batched route) + the REAL root cause fixed (batched per-lane prefill never wrote io_.text/backend_kv_table_row → every lane's prompt K/V landed in ROW 0; per-lane row writes added). Harness 5/5 PASS (bf16/i8 ±MTP + kvarn regression, 48/48 each).
2. ~~S1-i8-composition negative gate~~ **DONE 2026-09-06** (13ed7f0f). ~~S1-i8-composition FIX STAGED (b73fd4aa)...~~
3. QUEUE after (2): (ii) s2_order_diag re-diag WITH fix (pass closes tracked-open S2 — report either way) → (iii) bf16 serve OOM [preflight under-counts the TP2 fixed bucket; fix = cell budget shave (60k + media/prefix cuts) AND preflight runtime-reserve ~768MiB world==2 — both pre-coded in §8 22:2x] → (iv) §5 :470 collapse [inventory COMPLETE incl. prompt-route-allocates-nothing; implement the draft bound; validation = TC4 arena-peak ≥ bound on real kvarn rounds] → (v) materialize_tp mtp/ filter (tp_load.cpp:152; NOTE A2's 430MB/rank figure was a BINDER-PLAN delta NOT VRAM recovery — still the real load-side lever; the trap = binder object-index consistency per bindings.cpp:181; OWN commit + device verification).
4. Mesh: A2 = DFlash2 (their fold waits for Phase C → main; their tp_engine deliver() refusal arm may textually touch can_batch — coordinator handles at fold; TokenTile-7/8 refusal ruling, conv-base bind-time-transpose ruling, and the 430MB retraction are recorded in docs/151 §22.13/§22.21). gemini = agent_comm (may be down; coordinator relays). GPU/commit/shell protocols (§4) unchanged.

## 0b. WHO YOU TALK TO (the mesh split will bite you)

- **You are `agent1`** on the **intercom** mesh (tool: `intercom`; sessions: list).
- **`coordinator`** — intercom (`01a07140` at last session end). Rulings, gate policy, GPU
  grants, doc numbers. Current ruling on record: merge-gate policy (b) applied to Phase S;
  S2 = tracked open gate; Phases B/C/D cleared with **gemini authoring all B/C/D formal
  tests** (TEST SPLIT rule — my harness/CI verification is valid but formal tests are
  gemini's; hand scope, don't author).
- **`gemini`** — **NOT on intercom.** gemini lives on the **agent_comm mesh**
  (`agent_comm` tool, action send/to "gemini"; id 256adaf6-e151-41b8-99b9-78dbe6275f1f).
  Mesh was DOWN at session end — the coordinator relays (they have the TC scope).
  gemini was on DFlash2 tests (DT1-DT5), then MultiBatch B/C/D tests.
- **`agent2`** — intercom (`01a0713f`), DFlash2 lane. Do not touch `src/ops/dflash2/**` or
  the drafter-budget seam.
- **The user** directs directly and overrides everything; flag wrong instructions before
  executing (that discipline caught a destructive directive before, and a coordinator cite
  error this session).

## 1. What is DONE (verified, evidence committed)

- **Phase S (batched sampling) — MERGED to main `cbf613fa`** (gate-policy ruling (b): greedy
  contracts green, S2 tracked-open). Per-lane SamplingConfig in the batched runner (uploaded
  ONCE per batch), sampled plain-decode arm (allgather + ops::sample; fused argmax kept on
  ALL columns — the one-shot handshake needs identical call counts on both ranks),
  sampling-mode MTP accept (the op was already sample-capable; the batched path just fed it
  all-greedy configs), eligible() temp gate dropped. Lookup-draft + uniform-mtp_k exclusions
  STAY (negative gates verified on-device by TS5).
- **Phase A (tier-family batched decode) — `037d47f2`.** can_batch + runner fallback widened
  to `is_kvarn_storage` (i8/bf16 keep the docs/138 M4 per-lane fallback until Phase C);
  launcher `(k5v4||k4v4)&&multi` throw replaced by tier dispatch; `<MultiBatch=true,
  Masked=true, KSide=5/4>` compiled clean FIRST BUILD. Harness 4/4 (k5v4/k4v4 × ±MTP, 48/48
  MATCH per lane); serve CI 0-failed at k5v4 AND k4v4 (P0/B0-B3/S1/S2). Harness gained
  `--kv-dtype kvarn_k5v4|kvarn_k4v4`.
- **Phase B (non-MTP batching un-gated) — `9bc82131`.** The docs/143 valid-columns crash was
  ALREADY fixed by the 2a runner rewrite; self_mtp gate removed from can_batch.
  Serve-verified: concurrent no-spec pair dispatches `mtp=off k=0 lanes=2`, byte-exact vs
  sequential (`tools/smoke/diag/phase_b_repro.sh` — kept as the no-spec concurrency diag).
- **KVarN closeout addendum — CLOSED** (3be2c226, coordinator-ratified): identity ladder
  cited (serve_options.cpp:399-406 — bf16 42 / int8 38 / k5v4 38 / k4v4 46 / k4v2 58 @ n=24),
  docs/117 §5 final state (all tiers PASS, KLD waived, refusals lifted e002548d, q4_0 still
  refused = Phase D), name tables RATIFIED-DISTINCT (request_log schema strings vs canonical
  help text; cross-ref comment at the kv_cache_name switch).
- **gemini's TS-suite** (cherry-picked as 1c51790f..db41d106) PASS on device 0 after their
  TS1 fix (7fedd1c6: per-rank dev_draft_vocab — my device-0 runs exposed a cross-device
  pointer their device-1 runs masked; rank-1 kernel deref'd a dev-0 pointer, no peer access,
  sticky illegal address). TS5 verified the gap-3 negative gates on-device + workspace
  invariant (max_lanes=8 → 0.62 MiB).

## 2. OPEN work

1. **Phase C** — per §0. The multi-hour item; fresh-budget material.
2. **Phase D (G4)** — per §0 item 5.
3. **S2 localization (tracked-open gate, k4v2 only, coordinator-ruled non-blocking)** —
   sampled lane is lane-ORDER dependent: same-order runs byte-identical, order flip changes
   the sampled lane only (greedy byte-stable; proven with pinned seed via
   `tools/smoke/diag/s2_order_diag.sh`, exit=2 discriminator). Instrument (designed, not
   built): round-1 verify-logit dump per lane order + diff — differs across orders ⇒
   prefill-side state leak (second-prefilled lane contaminated); identical ⇒ verify-numerics.
   **`NINFER_MB_SEQ_STEP` does NOT gate the MTP round** (only the plain step,
   tp2_backend.cpp:2869/3689) — that probe is VOID, don't repeat it.
4. **gemini coordination** — Phase B test scope (mutation = re-add a self_mtp-style gate to
   can_batch → non-MTP must FALL BACK single-sequence, no dispatch line, replies usable;
   positive control = phase_b_repro.sh byte parity) + Phase C TC1-TC4 scope (§8 15:3xZ
   entry). Both relayed via coordinator; repeat directly when the mesh returns.

## 3. Facts established this session (do not re-derive)

- `one_shot_argmax.cu`: ties break on min GLOBAL SLICE idx (rank*n_rows+max_idx), remap AFTER
  (`final_tok = draft_vocab_ids[best_idx]`); draft vocab is JSON-loaded
  (tp2_backend.cpp:620), NOT monotonic. "min(final_tok)" is NOT the tie-break invariant.
- `translate.cpp:47-52`: unpinned seed → RANDOM per request (observed). Any byte-
  reproducibility gate MUST pin `--seed` (S-cells use `--seed 20260905` via
  EXTRA_SERVER_ARGS in serve_batched_ci.sh).
- The accept op (`speculative_accept_greedy_drafts`, include/ninfer/ops/speculative_round.h)
  is fully sampling-capable: per-row configs, target-probability accept + residual resample
  + bonus, RNG positions derived from old length. NO stochastic drafting exists anywhere —
  single-seq drafts greedily even at temp>0; proposals are one-hot.
- Batched runner Phase S internals: configs COMPACTED per round in live-lane order (mb_scfg)
  — indexing configs by compacted position against the original-lane array was a real bug
  the old all-greedy upload hid; per-lane token_counts slices only when penalties present;
  the fused allreduce_argmax still runs on ALL columns (one-shot handshake parity), sampled
  lanes overwrite their tok_pinned entry.
- Streams are `cudaStreamNonBlocking` (device.cu:67): default-stream copies carry NO
  ordering vs stream-s work. Always `cudaMemcpyAsync(...,s)` + `cudaStreamSynchronize(s)`.
- `allgather_local_bf16` sendcount is in ELEMENTS of bf16, not bytes.
- `NINFER_MB_SEQ_STEP` gates only the plain step. Live instrumentation:
  `NINFER_MB_SERVE_MUTATE` (B3 negative), `NINFER_MB_PHASEGATE`, `NINFER_MB_LICDBG`,
  `NINFER_MB_HASHPT`, `NINFER_BATCH_DISABLE` (kill-switch), `NINFER_BATCH_WINDOW_MS`
  (order-control lever: 500ms + 300ms stagger = deterministic lane order, used by
  s2_order_diag.sh).
- A server can wedge past SIGTERM holding 13.5 GiB/card: stop_server escalates to SIGKILL
  (own-PID only) after 30s grace. HAPPENED once today (poisoned a TS run — that run VOID).

## 4. Protocols (each exists because something broke)

- **GPU**: written coordinator grant before launching. TP2 needs BOTH cards (serve CI
  `DEVICES=0,1` default) — sequence with gemini's device-1 work.
  `gpu_guard.sh gpu_refuse_if_busy` before every launch. NO system-wide pkill — ever.
  Kill only PIDs you started; escalate to -9 only on your own wedged server, sleep 3 first.
- **Rebuild after ANY source change** (not just merges): a stale ninfer-serve voided a CI
  run today (binary predated the Phase A widening → silent fallback behavior). Targeted
  builds only (`cmake --build build --target <t> -j6`); `df -h /` before big builds (full
  build ~17G, targeted ~600M; disk was 24-25G all session).
- **Commits**: explicit pathspec ALWAYS (`git commit -- <paths>`); a rename's deletion-half
  was dropped by a pathspec mistake once — complete renames carefully. Never `rm -rf` an
  output dir; timestamped dirs only; evidence logs get COMMITTED (results/phase_*).
- **Shell `&` discipline**: `cd X && long_chain &` backgrounds the WHOLE chain (bit me twice:
  nohup log never created / setup silently skipped). Run setup in one call, background
  launch in the next, with `nohup ... > explicit.log 2>&1 &` alone.
- **pgrep self-match**: `pgrep -f phase_b_repro` matches your OWN grep/poll command line —
  verify liveness with `pgrep -fa` and read the actual artifact dirs.
- **nvcc flag mandatory**: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc` (default
  12.9). Worktree `build/` is already configured; binaries were current at handoff, but
  REBUILD both test targets on wake and re-run the suite once (external commits arrived
  mid-session; 2 min removes all doubt).
- **Doc numbers from the coordinator only** (this WO = docs/154). Marker gate v2 before
  pushes: `git grep -nE "^(<{7}( |$)|={7}$|>{7}( |$))"` repo-wide, expect 0.
- **Model artifact**: `/home/intel/models/qwen3_8_27b.ninfer` (17G, TP2, both 5060 Ti 16G).
- **The branch is shared**: gemini's fix (7fedd1c6) appeared in my local branch mid-session
  without an explicit pull. `git fetch` + check `git log` before assuming HEAD.

## 5. Tooling inventory (all committed)

- `tools/smoke/serve_batched_ci.sh` — the serve cell: P0/B0-B3 + S1/S2 (S-cells pin
  `--seed` via EXTRA_SERVER_ARGS). Env overrides: ART MODEL MAX_CONTEXT KV_CAPACITY
  KV_DTYPE SPEC DRAFT_TOKENS MAX_CONCURRENCY DEVICES PORT REPO BIN LOGDIR KEEP_WORK
  EXTRA_SERVER_ARGS. ~7-12 min for the full run.
- `tools/smoke/diag/phase_b_repro.sh` — no-spec concurrency byte-parity (Phase B regression).
- `tools/smoke/diag/s2_order_diag.sh` — S2 order discriminator (v2: window 500ms + 300ms
  stagger = deterministic lane order; exit 0/1/2/3 pre-stated).
- `tools/smoke/diag/gpu_guard.sh` — refuse_if_busy / kill_own. USE IT.
- `tests/multi_gpu/tp2_batched_decode.cpp` → `ninfer_tp2_batched_decode_test`
  (`--artifact --kv-dtype --mtp N`; batched==sequential bit-exact gate).
- `tests/test_batched_sampling.cu` → `ninfer_batched_sampling_test` (gemini's TS1-TS5).
- Evidence: `results/phase_s/`, `results/phase_a/` (committed); server logs in
  `/home/intel/verify_logs/` (timestamped, auto-created by the CI script).

## 6. First three things on wake

1. `cd /home/intel/ninfer/worktrees/wo-kvarn-multibatch && git fetch && git log --oneline -5
   && git status` — external commits may have arrived (gemini/coordinator share the remote
   and object store). Read new inbox messages (intercom + agent_comm).
2. Skim docs/154 §8 entries from 2026-09-05 (Phase S/A/B completion + Phase C scoping) —
   then start Phase C per §0: kernel half first, gate collapse in the same change.
3. Rebuild `ninfer_batched_sampling_test` + `ninfer_tp2_batched_decode_test`, re-run the TS
   suite once (expect rc=0), THEN any new device work — after
   `gpu_guard.sh gpu_refuse_if_busy`.
