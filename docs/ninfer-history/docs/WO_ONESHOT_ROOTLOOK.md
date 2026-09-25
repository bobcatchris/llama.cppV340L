# WO_ONESHOT_ROOTLOOK — one-shot AR wedge root-look desk (amd/wo-w7-body)

Desk opened: 2026-09-19 (one-shot wedge root-look, Team Red / V340L 4x gfx900).
Worktree: /home/chris/worktrees/amd-wo-w7-body (branch amd/wo-w7-body). Serving window owned.

MISSION: GATES two landings — Branch-B decode cure (banked-unpromoted, untestable while
wedges persist) + AR-quant implementation. Root-look the oneshot-arm warmup wedge:
- Wedge 6/7 boots today across TWO bins (183da007 with defer edit + LKG c618d356 WITHOUT
  it) → defer edit EXONERATED.
- Signature: rank0 stuck at FIRST AR crossing, gen=0, slot=-1, inflight_ms=0, hb_seq=1 —
  host-side handshake deadlock, PRE-AR. Canonical ring boots 4/4 healthy.
- dmesg clean. B3 watchdog caught 2 with full matrix (exit 72); others hung to SIGTERM.

ROOT-LOOK QUESTIONS (order):
1. Cheap repro: how deterministic post-reboot+post-FLR? (3-5 boots max)
2. WHERE does rank0 block: name the exact wait — which rank's commit stamp never arrives?
   Warmup asymmetry vs stamp-published-but-not-visible (mapped-pinned ordering post-FLR)?
3. WHY NOW: was oneshot warmup ever exercised pre-reboot on this box? What changed?
4. CURE cheapest first: (a) init-order/barrier fix; (b) timeout+fallback-to-ring first N
   collects; (c) :661-665 gen/stamp discipline.
   PRE-REGISTERED CURE GATE: oneshot boots complete warmup 5/5 AND decode t/s within 2%
   of pre-wedge-era numbers.

LAWS honored: checkpoint this file after EVERY step; retire = EXACTLY
pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"; restore canonical + health at
every close; wedged = SIGTERM; clocks noted on timings; no estimated VRAM refusals;
F1b desk may probe :8100 lightly — coordinate here.

---

## LOG

### STEP 0 — desk open (2026-09-19, clock TBD on first command)

- Box surveyed: rocm-smi 4x "Vega 10" confirmed. Canonical
  /home/chris/artifacts_bin/ninfer-serve_2c8901d3d18adef1.bin UP on :8100, health ok
  (PID 173317, started 10:42 by prior desk; inherited, will preserve/restore).
- Artifact bank contains both wedge bins: ninfer-serve_183da007801e0fb6.bin (defer edit)
  and ninfer-serve_c618d356f0cdc401.bin (LKG pre-defer). Good — repro can A/B them.
- Desk file created (this file). Next: read one_shot_allreduce.cu/.h + tp_group.cpp gate,
  then prior art (w7_defer_landing logs, kernlog_pre_flr, PLOG-066).

### STEP 1 — evidence read (2026-09-19, ~1h into window)

READ (all in-tree):
- one_shot_allreduce.cu FULL (1182 ln) + .h + ar_liveness.h + tp_group.cpp gate/worker.
- w7_defer_landing/{ARMA2,ARMCNTLD,LKG_oneshot}_serve.log — the two B3 matrices.
- PERF_LOG_AMD PLOG-066/067/068 + W4AR_repro_row.txt + ONESHOT_W4 notes.

INTERPRETATION OF THE BANKED MATRICES (ARMA2 @10:41 defer-bin, ARMCNTLD control):
- ALL ranks gen=0 slot=-1 inflight_ms=0: NO rank ever stamped the one-shot progress
  clock (stamped pre-launch, never reset to 0 on success). The one-shot route was
  NEVER ENTERED by ANY rank before the wedge.
- rank0 hb_seq=1 (inside FIRST allreduce_local_bf16 crossing, +2078ms), rank3 hb_seq=3
  (inside 2nd), rank1 hb_seq=12 (6 crossings ENQUEUED-and-returned), rank2 hb_seq=0
  (never entered). NOT lockstepped — per-rank progress through layer-0 collectives.
- hb covers ONLY allreduce_local_bf16 (tp_group.cpp:383-403). Inside it, between hb
  entry and the one-shot progress stamp there is exactly ONE blocking site: the NCCL
  branch's ncclAllReduce HOST call (n_elems > kMaxElements=65536 → ring branch).
- => RANK0 IS BLOCKED INSIDE ncclAllReduce (host) ON ITS FIRST LAYER-0 COLLECTIVE of
  the first warmup request ("hi", max_tokens=4). Rank1 enqueued 6 such collectives;
  rank2 never scheduled/entered its first.
- This is NOT the :661-665 recycling hole (that is an in-kernel accept ambiguity, and
  the kernel was never entered), NOT a warmup-asymmetry deadlock of the one-shot
  handshake (no one-shot handshake happened), and NOT a memory-ordering miss on the
  mapped stamps (no stamp was ever written).
- Failure-mode history: 09-18 pre-FLR (W4AR_repro_row, bin aec8ac56): same arm died
  LOUD at ~20s warmup — "Memory access fault by GPU node-3, Page not present"
  (~130-collect window) + older "17-min silent spin". PLOG-067 INCIDENT 1: canonical
  bin 0/6 boots page-faulting at random dies pre-FLR — box/firmware degradation,
  cured by kern.log -> FLR x4 -> reboot. POST-reboot canonical is 4/4 healthy, but
  oneshot-arm wedges 6/7 (this blocker) — 1/7 got through and SERVED (PLOG-068 A/B:
  -34% t/s, acceptance 0.96->0.85 ulp-drift — route works but is slow+drifty).
- CONSEQUENCE for the root-look: the wedge lives in the INTERACTION of the armed gate
  with the ring/NCCL warmup path, NOT inside the one-shot kernel. The gate's ctor
  delta vs canonical: OneShotAllReduce(4) mapped-pinned staging (~64 MB pinned) +
  watchdog thread armed BEFORE ncclCommInitRank; plus per-layer decode ARs re-routed.

NEXT: STEP 2 live repro (LKG c618d356 + gate, then canonical 2c8901d3 + gate), gdb
thread stacks at the wedge to name rank0's exact PC. Script:
results/amd/coherence/w7_oneshot_rootlook/w7_oneshot_repro.sh

### STEP 2 — REPRO + ROOT CAUSE NAMED (2026-09-19, ~2h into window)

REPRO MATRIX (this desk, boots on :8100, all NINFER_TP_ONESHOT_AR=1, canonical env):
- R1  LKG c618d356 ungdb'd      -> WEDGED_B3 (warmup+2.1s, rank0 kind=2 hb crossing#1) 1/1
- R2  LKG + SIGSTOP harness     -> LOAD-TIME OOM (hipErrorOutOfMemory; boot fits with 1 MiB/die
                                   slack, dying predecessor strands VRAM) — harness artifact, not the wedge
- R3  LKG under gdb             -> SHAPE B: SILENT SPIN 13+ min (GPUs 0,1,2 100% one-CU-each,
                                   GPU3 0%, 3 host threads 91% user-spin, NO watchdog fire, NO
                                   [AR-RETRY]) = the original T2 RED face (2/2 on 09-17, bin 409450b8)
- R5/R6 LKG + LD_PRELOAD probe  -> ncclCommInitRank rc=3 internal error x2/2 — the probe PERTURBS
                                   (instrument retired; ring boots untouched by it, but armed init breaks)
- R7  LKG under gdb, break _exit-> SHAPE A CAPTURED: B3 _exit(72) breakpoint froze the wedge state;
                                   23-thread backtraces banked (R7_LKG_final_gdb_console.txt)

ROOT CAUSE (mechanism, from the R7 all-thread dump at the frozen wedge):
- EVERY compute thread is stuck in the HIP/HSA kernel-SUBMISSION path — NOT in any
  one-shot handshake, NOT in RCCL logic:
    thread 22: ncclAllReduce -> librccl -> libamdhip64 launch path -> sem_wait
               (this IS rank0's hb crossing #1 — the B3 kind=2 signal)
    thread 1 (main): one_shot_axpy_bf16 -> libamdhip64 -> libhsa deep wait
    threads 23/24: tp_gemv/launch_nvfp4_small_t, tp_unpack_gbeta_strided -> same stall
- TpGroup workers (6-9) idle in cv_task wait; RCCL proxy threads (10-21) idle in poll.
- => the process cannot SUBMIT work to ANY of the 4 devices anymore. The first ring
  crossing that was mid-launch when the stall bit sits in ncclAllReduce >2s, the B3
  crossing-heartbeat watchdog kills (Shape A, exit 72). If the stall bites when no
  crossing is in flight, nothing is armed to see it -> silent spin (Shape B).
- THE GATE'S DELTA THAT DOES IT: OneShotAllReduce(4) ctor makes ~1,044 individually
  hipHostAlloc(MAPPED|PORTABLE)-ed host ranges (128 slots x 4 ranks x (128KiB buf +
  flag word) + gen/epoch/status/progress words) BEFORE ncclCommInitRank. Thousands of
  SVM/GTT ranges on this ROCm 6.2 gfx900 box stall the launch/submission path.
- WHY NOW (Q3 answered): the wedge is NOT new — it is the ORIGINAL W4-LOCKSTEP-LIVENESS
  RED (ONESHOT_W4_IMPL_NOTES: 2/2 bin 409450b8, "3 GPUs pinned at 100%, 1 CU each,
  17+ min silent"). The 09-18 W4AR face was a device PAGE FAULT at ~130 collects
  ("Page not present"); today's faces are the submission stall (Shapes A/B). All three
  are mapped-pinned-page-table faces of one class; the FLR/reboot flipped which face
  presents. Ring boots allocate ZERO one-shot ranges -> 4/4 immune. The 1/7 armed boot
  that served (PLOG-068 A/B) is the race winning.
- EXONERATIONS: the :661-665 recycling hole (kernel never entered — no stamp ever
  written, gen=0 everywhere); the defer edit (LKG has none, still wedges); warmup
  asymmetry of the one-shot handshake (no one-shot handshake happens before the stall).

CURE (cheapest-first, class-level): collapse the staging to ONE mapped block per rank
(world>2 arm only; world==2 path stays byte-frozen per law): 1,044 ranges -> 4.
Buffs carved at 128KiB stride, flags at 64B stride (no cross-rank cache-line sharing).
Then boot gate-armed x5 vs the pre-registered gate.

### STEP 3 — ROOT CAUSE CORRECTED: the 2s B3 deadline fires on LEGAL slow warmup crossings

C1 cure boot (fbd36d21, 20 mapped ranges): STILL WEDGED_B3 at +2189ms, identical signature
=> the mapped-range COUNT is NOT the trigger. Range-cure demoted.

DECISIVE CONTROL (R8, ring posture, cured bin, NO gate): "warming up..." 12:01:34.319 ->
"listening" 12:01:53.055 = **18.7 s of LEGAL warmup** on this cold-cache box. The B3
crossing-heartbeat bound is kArCallDeadlineNs = 2 s. ANY single crossing inside those 18.7 s
that takes >2 s (HIP first-use module loads serialize host-side with the launch of the next
collective; rank progress is per-thread unsynchronized) kills the armed boot — while the SAME
crossing in ring posture is INVISIBLE (no watchdog exists there; warmup just finishes slow).
This is why:
- ring controls read "4/4 healthy" (blind, not clean),
- armed boots die at the first crossing (Shape A: rank0 hb_seq=1, gen=0 — the one-shot route
  was never even reached; the R7 dump's launch-path stalls are the runtime serialization
  behind the cold module loads),
- some armed boots instead pile up silently when no crossing is open at scan time (Shape B),
- the pre-cure T2 era (unbounded world>2 kernel polls) turned the same stall into the
  17-min GPU-pinned hang,
- 1/7 armed boots served: later-in-day boots had warmer HIP code-object caches (shorter
  module loads) — the crossing dips under 2 s,
- both bins wedge identically (the bound, not the code region, is the killer).
LITH DISCIPLINE VIOLATION NAMED: the B3 bound fires on LEGAL slow paths — the exact class
the VRAM law bans for launch gates ("never refuse/kill on an estimated constant"). A liveness
deadline must exceed the worst LEGAL crossing; 2 s does not (measured legal warmup = 18.7 s
on this box cold).

CURE v2: NINFER_AR_DEADLINE_MS (single source ar_deadline_ns() in ar_liveness.h; read once)
feeding B2 (allreduce_bf16 deadline) AND B3 (watchdog arm). Default stays 2000 for the
byte-identical law; the widening env is the cure knob for the armed route. Proof boot next:
armed + NINFER_AR_DEADLINE_MS=60000 — if warmup completes, the "wedge" is PROVEN to be the
bound firing on legal slow crossings, and the banked matrix gives the true crossing ages.

### REPRO DONE 2026-09-19T12:12 CDT — WINDOW HANDED TO COORDINATOR (Path A matrix)

Box state at handover: all 4 dies 0% busy / 0% VRAM-allocated, no ninfer-serve running.
Canonical (2c8901d3 + 10k grade) NOT yet restored — returns at my final close, after
cure-validation boots. Cured banks available for your matrix if useful:
- ninfer-serve_fbd36d2143a59218.bin (single-block staging cure, deadline still 2s)
- ninfer-serve_eea46f6766f5b9d9.bin (staging cure + NINFER_AR_DEADLINE_MS knob, default 2s)
Both gate-armed only via NINFER_TP_ONESHOT_AR=1 (absent = ring, byte-identical routes).

ROOT-LOOK RESULT (full detail above, STEP 1-3):
1. The wedge faces are (A) B3 exit-72 at the FIRST crossing and (B) silent launch-path
   pileup. The R7 frozen-wedge 23-thread dump + R8 control NAME the mechanism:
   - R8 RING control: legal warmup on this box = 18.7 s (12:01:34.3 -> 12:01:53.1).
   - B3's crossing-heartbeat deadline = 2 s and it covers ALL crossings (NCCL arm
     included). Any legal >2 s crossing (cold HIP first-use module loads serialize the
     launch path host-side; per-rank threads unsynchronized) kills armed boots at
     "wedged_rank=0 hb_seq=1 gen=0" — the one-shot route is never even reached.
   - Ring posture is BLIND not clean: no watchdog exists there, the same >2 s crossings
     just pass slowly. "4/4 healthy ring" controls do not falsify this.
2. Mapped-pinned staging (your prime suspect) is DEMOTED: C1 (staging collapsed from
   ~1,044 mapped ranges to 20, one block per rank) still wedged identically at +2189 ms.
   Range count is not the trigger. (Answer to your question: plain-malloc staging is NOT
   viable for the KERNEL route — the device publishes/polls through device-visible host
   pages; malloc pages are invisible to the device and hipHostRegister re-enters the same
   page-table machinery. For copy-add / Path A PA-2 the AR becomes D2H copy + host add +
   H2D: plain pinned (even pageable) staging works, NO mapped staging at all — copy-add
   indeed makes the mapped staging optional.)
3. C2 (armed + NINFER_AR_DEADLINE_MS=60000): NO exit-72 — the kill is gone — and warmup
   PROGRESSES (GQA width=27 + MTP-TAIL chunk prints past 130 s) but ~7-10x slower than
   ring's 18.7 s. Layer-2 open item: WHY armed warmup is so slow (my working theory: the
   one-shot world kernel's first launches + per-collect poll lags + host-side retry loop
   hitting the same cold-JIT serialization). Quantifying without boots during your window;
   validation boots (armed, long bound, then the 5/5 + t/s gate) when the window returns.

REPRO DONE 2026-09-19T12:12:00-05:00

### STEP 4 (during coordinator window — NO boots) — layer-2 quantification + gate prep

C2 progress census (CORRECTED 12:2x, was mis-stated 10-30x): 1216 [GQA] prints in ~130 s
of warmup vs ring R8's 1225 in its full 18.7 s => C2 had done ~99% of the warmup print
volume when the 150 s boot bound cut it — armed warmup is ~6-7x ring (130+ s vs 18.7 s),
SLOW BUT COMPLETING. ZERO
[AR-RETRY] lines in C2 => every one-shot collect passed FIRST try; the slowness is
DESIGN, not fault: (a) the world>2 arm syncs PER COLLECT (cudaStreamSynchronize at
one_shot_allreduce.cu:1058 + B2 bookkeeping) where the ring branch is enqueue-only —
128 syncs/decode-step serialize the pipeline; (b) the one-shot world kernel is a NEW
code object (4 cold per-device module loads at its first decode step); (c) per-collect
poll latency (~100-200 us x 128). This is the same structural pessimization PLOG-068
measured end-to-end (-34%); Branch-B's defer was the attempt to remove exactly (a).

CURE-GATE REFERENCE (pre-wedge served oneshot boot, ARMCNTLD2 probe0, mt=600 count
class, fresh-boot-cold ordinal): decode 32.7 t/s, wall 18.33 s, 222 rounds,
acceptance 0.85. GATE on window return: 5/5 armed boots complete warmup (bound 600 s
each) THEN one count600 mt=600 probe (run FIRST = fresh-boot-cold, matching ARMCNTLD2's
ordinal) vs 32.7 t/s within +-2%, acceptance recorded (expect ~0.85 — the ulp-drift
collapse is the ROUTE's known property, not the wedge cure's scope).

VALIDATION BOOT PLAN (when window returns):
V1-V5: armed boots, NINFER_AR_DEADLINE_MS=60000, bound 600 s each, verdict = warmup
completes + listening line. (C2 pattern says expect ~3-5 min warmup each on cold dies.)
V6: the t/s probe leg on a 6th fresh boot.
Retire law between boots; canonical restore + 10k-grade health at final close.

### STEP 5 — CLOSE (2026-09-19 ~13:05 CDT): cure gate verdict + Branch-B re-fire verdict

VALIDATION MATRIX (cure boots, all NINFER_AR_DEADLINE_MS=60000 except C1):
- C1 fbd36d21 (staging-block cure, 2s deadline): WEDGED_B3 +2189ms — staging count NOT the trigger
- C2 eea46f67 (staging + deadline 60s): NO exit-72 ever; warmup crawled to ~99% print volume by
  150s (1216/1225 of ring's phase prints) — the kill is GONE, boot SLOW
- probe eea46f67 (fresh boot): B3 fired at +60104ms on rank2 kind=2 (RING crossing — progress
  unstamped) = some crossings never complete even at 60x the nominal bound; wedged rank VARIES
- C3 b7b20017 (v3 ctor one-shot warmup collect): WEDGED_SILENT 600s, ZERO warmup prints — WORSE
- C4 b7b20017 (identical re-boot, warm cache): WEDGED_SILENT 600s identical — cache-warmth falsified

GATE VERDICT (pre-registered: 5/5 armed warmups + decode t/s within 2% of 32.7): **FAIL — 0/5
attempted; no cure boot reached listening.** The route is multi-factor blocked on this box:
L1 the 2s deadline kills legal slow crossings (FIXED by NINFER_AR_DEADLINE_MS, proven); L2 an
armed-only launch/submission stall (R7 dump: every rank's launch leg blocked in the HIP runtime
semaphore/HSA path) that survives staging-size cure (C1), deadline cure (C2/probe), ctor-warmup
reordering (C3), and cache warmth (C4). Remaining armed-only delta at stall time is
ctor-warmup-collect + mapped control words + watchdog thread + heartbeat writes — the next
discriminating step needs device-side instrumentation (rocprof/HTA) or a ROCm SR on gfx900/ROCm
6.2 submission stalls, or a fresh FLR/reboot window with immediate validation (the 1/7 pattern).

CURES LANDED (default-identical, committed fd9198ae8):
- single-block staging world>2 (1044 -> 20 mapped ranges; world==2 bit-frozen) — measured neutral
- NINFER_AR_DEADLINE_MS single source ar_deadline_ns() feeding B2+B3 — measured: removes the
  exit-72 kills; default 2s unchanged
- guard cell evolved (RED on pre-fix shape = 5 findings incl. class naming; GREEN post-fix;
  selftest 27 mutations detected)
- v3 ctor warmup collect: MEASURED NO-GO, REVERTED from tree (banked b7b20017 for the record)

BRANCH-B RE-FIRE VERDICT: **NO — not on the one-shot route today.** Three structural cures did
not produce a single clean armed boot; the A/B cannot run. The evidence-backed paths:
(1) PLOG-068's elevation of copy-add (option 0 — no one-shot machinery; needs no mapped staging,
    no world kernel, no gate) is now STRONGER — the AR-quant desk should implement on copy-add;
(2) if the one-shot route must live, a FLR/reboot + immediate armed validation window (the
    1-in-7 pattern, caches warm by prior boots) is a coin flip to be attempted ONCE, not a landing.

BOX STATE AT CLOSE: canonical 2c8901d3d18adef1 UP :8100, health ok, FAST PROBE PASS 18.3s stop.
Banks added: fbd36d21 (staging cure), eea46f67 (staging+deadline), b7b20017 (v3, no-go record),
8127002e (v2-state relink of the committed tree). Dies: serving posture, probe PASS.
