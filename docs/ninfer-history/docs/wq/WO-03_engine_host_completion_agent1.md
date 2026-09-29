# WO-03 — finish the engine/serve host half and name the true first-token dependency graph
**Owner:** agent1 (new session) · **Branch/worktree (MANDATORY):** `git worktree add ~/worktrees/amd-wo-engineserve -b amd/wo-engineserve amd/main`
**Base:** `amd/main` @ `7ce534f1` · **Grants needed:** none (compile-only + reading)
**Parallel to:** WO-02 (agent2 owns the shim width fix — **do not touch `hip_shim/cuda_runtime.h`**)

## 1. Context
Team Green (Q3 artifact + TP2 dispatch) is explicitly **not** a dependency for this lane. The AMD line's
blocking item is WO-02; everything else this lane needs can be built now. `amd/main` now carries the
whole consolidated AMD state (51 files: `run_ci_amd.sh`, `tests/v340l/`, 26 result artifacts, both lane
WOs) that had never been pushed anywhere.

## 2. Steps
1. **Finish the host whitelist:** `targets/*` + `serve/` + `apps` (the pending compile-only slice),
   per-file join rule with the parity guard, `+whitelist:` / `+shim:` subject tokens, expected parity
   line and sha1 reported at each commit.
2. **First-token dependency graph, corrected:** the earlier `~26 .cu` estimate **included** files whose
   reduces depend on the WO-02 defect. Re-derive per file: needs-shuffle-fix / independent / unknown, so
   we know what is genuinely unblocked while agent2 lands the fix.
3. **Single-device path audit:** confirm on `amd/main` that a `--device N` run reaches the plain
   `Engine` (`engine.cpp:309-313`) and enumerate what it still needs to *link* — the AR/tp2 symbols
   (`one_shot_allreduce`, `one_shot_argmax`, `tp_kernel`) that today force a link-time decision.
   Deliver the **loud-failing stub set** (abort naming the symbol; never return a plausible zero) as the
   minimum to link, so a single-device build becomes possible before the AR redesign lands.
4. **AR host-staged redesign — design document only, no code:** with **zero P2P on this box**
   (`hipDeviceCanAccessPeer` = 0 for all 6 pairs, cross-card staging 6.6–6.7 GB/s, same-card 4.86,
   RTT ~100 µs flat vs size), specify persistent pinned host buffers + async double-buffering +
   **batched AR every 2–4 layers**, with the per-step arithmetic (33–65 ARs → 3.3–6.5 ms/step) and the
   measurable gate that proves each claim. This is the Q3-on-2-dies plan of record's missing piece.

## 3. Constraints
No product-header edits outside the two registered exceptions; no device launches without a written
grant; **docs under `docs/amd/`, integrate to `amd/main`, never `main`**; commit before reporting;
quote the anchor for any claim you make — a static read is an argument, not a result.
