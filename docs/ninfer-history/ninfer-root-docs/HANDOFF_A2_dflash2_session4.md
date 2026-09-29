# HANDOFF — A2 dflash2 lane, session 4 (2b-iv(b) EXECUTED)

Resumption point for the next A2 session. Session 3's handoff
(`HANDOFF_A2_dflash2_session3.md`) is still accurate where not superseded here;
§22.55 in `docs/151_dflash2_scope.md` is the authoritative record of what this
session did. Read §22.54 (the plan) and §22.55 (what landed) before acting.

## State at handoff

- Branch `wo/dflash2-scope`, tip `5882d3ea`, tree clean, pushed? — verify with
  `git status` + `git ls-remote github wo/dflash2-scope` (session 3's rule).
- **DFlash2 DECODES on the batched dispatch.** Gate (the 2a throw) is GONE.
  d2_probe ALL PASS; real server run (ninfer-serve --spec dflash2, 2 lanes)
  produced coherent, factually correct text in both lanes.
- Commits this session: `d7f728c6` (A: six fork points + one lifted callable
  `dflash2_run_chain`, gate in), `bd51d8bc` (B: gate out + device run; staging
  48 MiB; OneShotArgmax kMaxTokens 16), `27c6c1ab` (C: serve stats mirror +
  backend label key on the drafting backend), `5882d3ea` (docs §22.55).

## The one open result, and its suspects

ACCEPTANCE ~0-3% on real prompts (recorded, not judged — §22.54 said this is a
result, not a gate). The drafter's ONLY prompt context is the anchor's fused
tap column (hist adds only its own mask-key history — that is the DFlash2
design). Suspects in order: mask-token column convention; drafter RoPE
base/rot; fused column selection (anchor col 0 vs the LAST accepted column's
tap — the §17.2 "VERIFY at wiring" item). Off-line discriminator:
`NINFER_MB_LICDBG=1 NINFER_DFLASH2_TRACE=1` on d2_probe prints target argmax
(lic) AND draft ids per round — compare lane drafts vs lic[1..] directly.

## Do not re-derive (measured this session)

- Staging overflow throws a BARE `std::bad_alloc` (arena.cu:208) — it looks
  like host OOM from the runner; it is the 48 MiB `TpRankState::staging`.
  Current live peak at the head allgather ≈ 27.7 MiB (mb_* ~10.2 + chain
  ~17.5). Headroom ~16 MiB.
- Verify argmax runs T·B columns; ceiling now 16 (`kMaxTokens`). At 2 lanes
  that admits k≤7; the ATTEND ceiling (TokenTile≤6) still caps k≤5 — blocker
  B unchanged.
- `kvarn_rewind_lane` is a no-op without batched workspaces (probe calls it
  unguarded — safe).
- The warmup single-request refusal ("multibatch-native: a batch of one") is
  CORRECT DFlash2 behavior in the server log.
- hist_lo = `max(plen, cur_F - 2048)`, deliberately NOT §17.2's
  `max(0, ...)` — slots below the lane's first anchor were never written;
  attending zeros dilutes the softmax. §22.55 records the deviation.

## Protocols that still bind (unchanged)

- Written GPU grant from the coordinator (intercom, session 01a07140) before
  ANY device work; re-guard at claim (`nvidia-smi --query-compute-apps`);
  release + verify when done. A1 shares the cards — coordinate via intercom
  01a07628. NEVER system-wide pkill; kill only your own PIDs (setsid children
  survive the captured PID — check `nvidia-smi --query-compute-apps`).
- Device iteration ~15 s (load ~9 s + gates 1.3 s). CPU suite 10/10 (~3 s):
  the ctest regex is in session 3's handoff §1 and still passes.
- d2_probe: `./tools/smoke/diag/build_d2_probe.sh && /tmp/d2_probe --mode
  kvarn` (also `negatives --quick`, `bf16`, `cap`). Part 2 asserts DECODE now.
- Doc numbers come from the coordinator; never self-assigned.

## Next steps, in order

1. Acceptance investigation (suspects + discriminator above). No new forks
   before the discriminator names the broken stage.
2. If fused-column selection is the suspect: the anchor tap is column 0 of the
   lane's fused stream by construction of the verify (column 0 = anchor token).
   Before "fixing", confirm with LICDBG which target column the drafts SHOULD
   match (lic[1] is the target's argmax at position F, conditioned on
   [anchor]; d1 is the drafter's same prediction — they should agree often IF
   the drafter sees what the target sees).
3. TokenTile 7/8 (blocker B) only if k>5 becomes product-relevant.
4. Keep the probe's verbose evidence unconditional; round-loop prints stay
   behind NINFER_DFLASH2_TRACE (log-spam discipline, §22.55).

## SESSION-4 ADDENDUM (acceptance investigation — READ docs/151 §22.56 FIRST, then this)

**One-paragraph state:** DFlash2 decodes (done, committed). Acceptance ~0-3%
is CAUSED by the drafter's w8 row-split linears computing non-canonical
outputs — proven by an FP64 oracle (validated torch port of
block_graph_ref.py) fed the engine's own dumped stage tensors: rmsnorm is
bit-faithful (cos 1.0000), the first row-split linear diverges (cos 0.1185,
magnitude 1.75×) at the GENERIC fallback launcher launch_w8_simt_r8_c8
(n=1280,k=5120,t=16 — no specialized table entry). Everything else (weights
bytes, dequant variants, layouts, permutations, aliasing, tensor mix-ups) is
RULED OUT with the evidence in §22.56.

**Hit-the-ground-running sequence (in order):**
1. `git log --oneline -3` — the BLK0DUMP tooling + committed oracle are in
   0f022a34; confirm your checkout has it (or is ahead of it).
2. Read docs/151 §22.56 top to bottom (2 pages). It contains the anomaly
   statement, the ruled-out list, and the decisive experiment spec.
3. Run the isolated-linear scratch TU (spec in §22.56 "THE NEXT EXPERIMENT"):
   pattern-copy tools/smoke/diag/dflash2_rope_onehot.cpp's build line; feed
   /tmp/d2blk0.r0.h into a direct ops::linear with a freshly-bound
   blk.0.attn_conv_proj; compare vs the engine's dumped proj AND vs canonical.
   Either outcome NAMES the fix location (kernel vs runtime state). If the
   kernel: add a non-table shape (n=1280, k=5120) to
   tests/ops/linear/test_w8_a16.cpp FIRST — linear_test_common.h already
   provides make_w8g32_f16s_weight(n, k) — expect RED, fix, expect GREEN.
   (The existing cases only cover the dispatch table's shapes, which is
   exactly how the generic-path bug stayed invisible.)
4. Fix, then re-verify in this order: (a) the scratch TU matches canonical;
   (b) d2_probe --mode kvarn ALL PASS with the oracle-vs-engine final-stream
   cosine ≈ 1 (regenerate dumps with BLK0DUMP/STAGEDRAW/TAPDRAW, all env-gated);
   (c) the REAL acceptance number: ninfer-serve --spec dflash2 2 lanes
   (command in §22.55) — expect the tok/round to move off 1.02.
5. Report one line per commit to the coordinator; the acceptance number gets
   appended to §22.56 (recorded, not judged).

**Files/tools that exist (do not rebuild):**
- tools/convert/qwen3_8_27b/dflash2/d2_engine_oracle.py — validated FP64 oracle (torch). Feeds on
  /tmp/d2stage.r0.{fused,mask} + sidecar; compares vs d2stage.r0.final.
- tools/convert/qwen3_8_27b/dflash2/d2_discriminate.py — PACK/LICDBG parser (d1 vs lic0).
- /tmp dumps: d2blk0.r{0,1}.* (block-0 stages), d2stage.r{0,1}.{fused,final,mask},
  d2raw.r{0,1} (taps). Regeneration command in §22.56 (~40 s device, guard
  first per protocol).
- Engine checkpoints (rank0 lane0): fused 859be55174842267, final
  90400224e2693bb8, drafts "220 11 430 274 198".

**Rules that still bind:** written GPU grant + guard before device work (A1
shares the cards — coordinate via intercom 01a07628); never system-wide pkill
(setsid children survive — check nvidia-smi --query-compute-apps); CPU suite
10/10 regex in session-3 handoff §1; doc numbers from the coordinator only.

**SESSION-5 CORRECTION (supersedes the addendum above):** the w8-row-split
"localization" was a DUMP ARTIFACT. The isolated-linear TU REFUTED the kernel
(fresh binding + direct ops::linear at (1280,5120,16) is canonical, cos
1.0000); the real cause was BLK0DUMP's legacy-null-stream cudaMemcpy racing
the engine's cudaStreamNonBlocking compute stream (device.cu:67). Read
docs/151 §22.57 FIRST. Dumps fixed (async on in.stream + sync); re-dump and
re-run the oracle before trusting ANY /tmp stage number.

**SESSION-5 FINAL STATE (read together with §22.57/§22.58):** the drafter is
CANONICAL — every dumpable stage matches the FP64 oracle to bf16 rounding
(x_in exact; h/proj/c_in/v/qn/kn cos ≥ 0.99999 at the round's true positions
[95..102]). §22.56's w8 localization and §22.55's fused-column suspect are
both DEAD; the anomalies were (1) the dump race (fixed), (2) the oracle's
conv-base container-order swap (fixed), (3) the oracle assuming cold-round
positions for LAST-round dumps (fixed: NINFER_DFLASH2_ROUND_POS). Acceptance
~0-3% is NOT drafter math: next device step is the §22.55 discriminator
(NINFER_MB_LICDBG=1 NINFER_DFLASH2_TRACE=1 /tmp/d2_probe --mode kvarn) — do
drafts agree with lic[1..]? If yes → the accept/rewind commit path is the bug;
if no → the verify-side conditioning (what the drafter is fed vs what the
target sees) is.

**SESSION-5 FINAL STATE (complete):** read §22.57-§22.61. The ENGINE IS FULLY
VINDICATED: assembly, fuse, fc linear, all 5 drafter blocks (every stage,
bisected via NINFER_DFLASH2_BLKIDX), attention (admits/scores/values), weight
provenance (sidecar == source gguf == arena), conversion — all verified
canonical. §22.56's w8 localization and §22.55's fused-column suspect are
dead; three measurement artifacts were fixed (dump race, conv-base container
order, stale positions); four oracle/tooling bugs found (fnv half-word, k128
head read, lane-crossing attention model, round-confused dumps).
ACCEPTANCE ~0-3% is NOT an engine computation bug. Rank analysis: the
target's next token ranks #552/#4711 of 248320 in the drafter's own
distribution (better than random, ~50× worse than useful) — conditioning
carries weak signal under the current semantics. THE remaining hypotheses:
(a) training-side tap semantics differ from the runtime capture+fusion (the
upstream z-lab/Qwen3.8-27B-DFlash2 reference is the source of truth),
(b) the drafter checkpoint in the gguf is weak/early, (c) accept/rewind
(unmeasurable until (a)/(b) resolve). Tooling stands: first-chain gated
dumps (BLK0DUMP/STAGEDRAW/TAPDRAW/FUSEDUMP/ADMIT/BLKIDX), d2_block_check.py,
d2_selector_port.py, d2_cold_forward.py, d2_iso_linear.cpp (generalized),
d2_fc_compare.py, d2_attn_map.py, d2_iso_compare.py, d2_stage_trace.py.
Branch pushed through 3dce954d.
