# WO_L2STALL_REFIRE — armed-route submission stall: 5th-cure attempt via NEW mechanisms

- **Why:** Tier 2 #7, RE-EVALUATION MANDATE item. The armed-route submission stall (libhsa
  semaphore class) survived 4 cures — a close-after-4-attempts, not exhaustion — and it
  BLOCKS Branch-B (+3-4.5 ms/round decode, banked) and AR-quant-on-one-shot. The mandate
  names the new mechanism classes; this desk executes them.
- **Desk:** agent #2 (freed madmix-int slot). Worktree: `git worktree add
  /home/chris/worktrees/amd-wo-l2stall amd/main -b amd/wo-l2stall`.
- **Evidence base (re-read FIRST — the 4 failed cures' rows):** search
  `results/amd/coherence/` + PERF_LOG_AMD.md for the Branch-B / armed-route stall rows
  (Branch-B bin 183da007801e0fb6 is banked-unpromoted). Name for each failed cure: what was
  tried, the exact failure signature, and why the mechanism class was thought sufficient.
- **THE NEW MECHANISMS (execute in this order; each is a genuinely distinct class):**
  1. **hsa_amd_memory_* API audit (GPU-free)** — read the armed-route submission path
     (src/core/multi_gpu/ — the libhsa signal/semaphore use sites) and audit against
     hsa_amd_memory_* semantics: are we mixing hsa_signal wait policies with HIP-stream
     scheduling? Is there a lazy-init or per-process agent lookup inside the hot path?
     Deliverable: annotated call-chain + the 3 most suspicious lines.
  2. **world=2 armed probe (small window, dies 2/3)** — reproduce the stall at world=2 with
     the armed route on the free die pair. world=2 removes 2 ranks from the semaphore
     mesh; if the stall VANISHES at w=2 it is contention/ordering, not a hard libhsa bug —
     that decides between per-rank staggered entry and warmup-skip classes.
  3. **warmup-skip-then-serve** — if the stall binds at first-armed-submission only,
     serve N warmup requests with the route DISABLED, enable at runtime, and measure.
  4. **per-rank staggered entry** — if (2) says contention: stagger ranks' armed entry by
     rank-index * k ms and map the stall boundary vs stagger.
- **Gates (pre-registered):** G-L2-0: any mechanism that eliminates the stall signature in
  the w=2 probe (submission latency p99 within 2x of the disarmed route) graduates to the
  TP4 probe. G-L2-1: TP4 armed route serves 600-round probe with NO stall event and
  Branch-B's +3-4.5 ms/round measured on 3 reproduced pairs (PLOG-060 design, clocks
  sidebands) => Branch-B re-fire PROMOTED to a serving A/B. Miss => bank the row, the
  5th attempt closes the family with a decisive mechanism map (no further re-evaluation
  without a platform change).
- **Window protocol:** phase 1 is GPU-free. Phases 2-4 need boots: BEFORE any boot, append
  "WINDOW CLAIM: <leg>" to this file on YOUR branch and check `pgrep -af ninfer-serve` —
  the k4v4gates desk (rank 0) owns the window until its close; the coordinator's G-MM-2 +
  instrument slot queue behind it. Never retire a serve you did not start; retire law form
  ONLY (`pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"`); restore canonical
  serve_fast + health at your close. No builds in the shared checkout EVER; df -h / before
  big writes; PROGRESS LOG newest-first after every step; infra = tag "BLOCKER:".

---

## PROGRESS LOG (agent #2 desk, newest first)

### STEP 6 [2026-09-19 ~19:5xZ] — E1 REPRODUCTION + WINDOW CLOSE: wedge reproduced on cured bin (silent face + device GPU Hang); w2 serve arms window-blocked; family-close mechanism map banked

- **Cell matrix (PROBE_matrix_row.txt, 4 arms, 20 s storms, ~825k-862k timed launches/arm): ALL
  CLEAN.** armed-w4 p50 2048 ns / p99 4096 ns / max 663919 ns; ring-w4 p99 8192 ns;
  armed-w2 p99 4096 ns; ring-w2 p99 8192 ns. No STALL event anywhere. Per the pre-registered
  interpretation: the cell CANNOT reproduce the stall even with the faithful armed deltas
  (interleaved per-rank MAPPED|PORTABLE ctor sequence + pre-NCCL watchdog thread + concurrent
  ncclCommInitRank + lockstep 655360-elem storm + per-collect sync) at either world => **the
  armed deltas are NOT sufficient triggers in isolation**; the stall requires the full serve
  process context. This kills the "cheap cell cure" path and demotes the cell's w2-clean signal
  (its w4-clean is demonstrably a fidelity gap, since the serve reproduces).
- **E1 (serve-level, banked bin c9075897, armed w4, deadline 60 s, R7 geometry): WEDGE
  REPRODUCED** — model load OK (17.6 s), "warming up..." then ZERO warmup phase prints for the
  whole life (ring prints ~1225 by 18.7 s) + **"HW Exception by GPU node-2: GPU Hang"** — the
  family's first banked DEVICE-SIDE face (prior faces were host-side parks). No B3 fire (no
  crossing was ever stamped — silent face, C3/C4's shape, now with a GPU Hang on top). Boot
  killed at +154 s by the canonical restore's retire line (serve_fast pkill matches any
  artifacts_bin serve, mine included) — no gdb freeze possible. Row: E1_w4armed_repro_row.txt.
- **Ring control, same box, same minutes:** canonical 2c8901d3 ring boots clean and serves
  healthy (/health ok) — box not globally degraded; armed posture remains the discriminator.
- **Window lost mid-desk:** the canonical restore (another actor, 19:39:35) took the box while
  E1 was wedged. E2 (w2-armed-deltas serve) + E3 (w2-ring control) are WINDOW-BLOCKED, not
  verdict-blocked; they remain the named continuation if a window returns (short boots,
  env knob NINFER_L2STALL_W2_DELTAS=1 + NINFER_TP_ONESHOT_AR=1 + --devices 0,1; VRAM caveat:
  the w2 load may die at the allocator per the TP4-minimum-world law — that refusal row is
  itself bankable and closes the serve-level w2 arm honestly).
- **GATES:** G-L2-0 NOT MET (no mechanism eliminated the stall — phase 2 discrimination is
  incomplete without the w2 serve arms). G-L2-1 NOT REACHED (no clean armed boot exists to
  probe). **Miss branch taken: the 5th attempt CLOSES THE FAMILY** with the map below.
- **MECHANISM MAP (family close, 5th attempt):**
  1. Not an OUR-side API misuse: hot path audited clean (API_AUDIT_row.md) — no per-call
     allocs/lookups/lazy inits, no direct hsa_* use anywhere in src/.
  2. Not the mapped-range COUNT (C1), not the deadline constant (C2 proven L1-only), not
     ctor-warmup order alone (C3 made it worse, not better), not JIT/cache state (C4).
  3. NOT sufficient in isolation (NEW, this attempt): the full armed delta set — interleaved
     4-device fine-grained mapped allocs + pre-NCCL watchdog + concurrent NCCL init + launch
     storm — runs CLEAN in a faithful cell at BOTH worlds.
  4. Requires the full serve process context; presents as host submission parks (R7:
     sem_wait + KFD WAIT_EVENTS, decoded this attempt) and, on the cured lineage, as a silent
     zero-progress warmup with a device-side GPU Hang (E1).
  5. Class verdict: **platform-class — gfx900 / ROCm 6.2 / KFD interaction under the serve
     process's full context with the armed multi-GPU fine-grained GTT mesh.** Code cures
     (5 attempted across two desks) have never produced one clean armed boot (0/12 counted
     across both desks' matrices). Per the WO's miss clause: no further re-evaluation without
     a platform change — candidates: fresh FLR/reboot + immediate armed validation (the 1/7
     race), ROCm upgrade, or a ROCm SR carrying R7 + E1 + the cell's clean matrix.
- Branch-B re-fire on the one-shot route: **NO** (unchanged from PLOG-070, now with the
  stronger map). The copy-add/pinned lane (PLOG-068 option 0) remains the elevated path and
  needs none of this machinery.
- Box at close: canonical 2c8901d3 serving, /health ok — restored by the other actor at
  19:39:35 and verified by this desk at 19:52; nothing of this desk's left running
  (cell processes exited rc=0; E1 killed by the restore).

### STEP 4 [2026-09-19 ~19:4xZ] — WINDOW CLAIM: L2 phase-2 cell matrix (l2stall_probe)

- WINDOW CLAIM: L2 phase-2 cell matrix — 4 base arms + conditional 5/6 arms, each ~25 s,
  90 s hard bound, NO ninfer-serve boots by this desk; canonical serve_fast + /health restore
  at close. Foreign-serve check before EACH arm; any foreign boot mid-matrix = yield.
- Escalation vehicle pre-staged GPU-free while the window was busy: NINFER_L2STALL_W2_DELTAS
  knob landed + serve bin BANKED `ninfer-serve_c9075897caa4f027.bin` (commit d2ae8668d) —
  runs only if the cell can NOT reproduce the stall (fidelity-gap escalation, pre-registered).

### STEP 2 [2026-09-19 ~18:3xZ] — phase 2 vehicle BUILT (GPU-free): l2stall_probe cell + discriminating design fix

- **Design fix to phase 2 (pre-registered in the audit row BEFORE any boot):** the naive world=2
  armed probe is VACUOUS as the code stands — at w2 the gate takes the bit-frozen branch
  (tp_group.cpp:173): no watchdog, no block staging, no progress/diag, so "armed w2" is
  byte-identical to default w2 and a "vanish" would prove nothing. The DISCRIMINATING probe
  carries the world>2 armed DELTAS to w2.
- **Vehicle (disk-bound choice):** a GPU cell, `results/amd/l2stall/l2stall_probe.cpp` — mimics
  the armed ctor sequence (interleaved per-rank hipSetDevice + hipHostMalloc MAPPED|PORTABLE
  words + 16 MiB/rank staging + hipMalloc + device-pointer queries, all pre-ncclCommInitRank on
  the main thread), the pre-NCCL 250 ms watchdog thread, concurrent ncclCommInitRank x world,
  then a timed launch storm with lockstep 655360-elem collectives. Matrix:
  {world=4, world=2} x {mode=armed, mode=ring}. Readout = per-launch submission latency
  p50/p99/max (G-L2-0 shape: armed p99 within 2x of ring at the same world = stall eliminated);
  any launch >= L2STALL_STUCK_MS = STALL EVENT, loud exit 42. Fidelity named: a cell REPRODUCING
  the stall is decisive; a cell NOT reproducing it is weak evidence (fidelity gap, not
  exoneration) and escalates to a serve-level probe.
- **BLOCKER (disk, standing):** / at 100% (840 MB free) — a serve-level build tree (~550 MB
  measured on kvarnport) does not fit alongside the other lanes. Cell vehicle chosen for this
  reason; worktree is a sparse checkout (results/ bank excluded) created with --no-checkout +
  sparse-checkout set — the worktree itself cost ~120 MB after trimming an over-broad first
  pattern (coherence re-include pulled 635 MB; reverted within minutes, net peak cost ~120 MB).
- Cell compiled clean with the WO's cell-style compile line (hipcc -O2 -std=c++20
  --offload-arch=gfx900, links librccl.so.1 + libamdhip64.so.6, 38 KB). rccl-dev quirk: nccl
  API lives at <rccl/rccl.h> (same shim rule as src/common/hip_shim/nccl.h).
- **Window: NOT claimed.** pgrep shows the k4v4gates/kvarn k4v4 serve (PID 797749,
  f3312f25c8201da0, :8100, devices 0-3) live — not mine, not touched; boots wait for a genuinely
  free window.

### STEP 1 [2026-09-19 ~17:5xZ] — phase 1 COMPLETE (GPU-free): hsa_amd_memory_*/hsa_signal audit

- Deliverable banked: `results/amd/l2stall/API_AUDIT_row.md` — full call-chain, the 4 failed
  cures named (C1 ranges 1044->20, C2 deadline 60 s, C3 ctor collect, C4 cache warmth; gate 0/5),
  the 3 most suspicious lines, and the audit answers.
- **Headline findings:**
  1. NEW BANKED FACT: the R7 dump's mystery ioctl request 3222817548 decodes to
     `0xC0184B0C` = **AMDKFD_IOC_WAIT_EVENTS** (kfd_ioctl.h:1501-1502) — the frozen wedge is the
     HSA event thread parked in KFD WAIT_EVENTS (an event the GPU never signals) while the main
     thread parks INSIDE libhsa in a plain one_shot_axpy_bf16 LAUNCH and the other rank runners
     queue on per-device submission semaphores in libamdhip64. Submission is not slow — it is
     PARKED behind a runtime-internal event handshake.
  2. Our-side submission path is API-CLEAN: zero direct hsa_* calls in src/; no per-call
     allocations/agent lookups/lazy inits in the hot path (2 init-once getenv statics, one
     mapped status read, one launch, one sync). The WO's named suspect classes have no our-side
     instance. The stall is runtime-side.
  3. The armed-only deltas never falsified: (a) the pre-NCCL interleaved 4-device
     mapped-alloc SEQUENCE (C1 varied range COUNT, never presence/bytes/order), (b) the watchdog
     thread — alive in 100% of wedge boots (R7 thread 4 is it), 0% of ring boots, never removed
     in any cure. The ctor's 8-elem warmup collect bypasses allreduce_local_bf16 (dispatch_all ->
     NCCL ring), which is why the one-shot kernel never launches pre-wedge (gen=0) — the banked
     matrices are consistent with the wedge being the RING crossing + launch mesh.
- Verdict: evidence points to the WO's **contention/ordering class at the RUNTIME layer**
  (event/signal-delivery race perturbed by the armed ctor's concurrency+ordering), NOT a hard
  deterministic libhsa bug and NOT an our-side API misuse. Phase 2 (w2-with-deltas) and phase 4
  (staggered entry) are the right discriminators; a watchdog-only arm is the cheapest
  single-delta falsifier nobody has run.
- Box untouched (no boot, no build, no GPU work).
