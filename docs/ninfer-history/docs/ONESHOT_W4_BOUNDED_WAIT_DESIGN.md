# ONESHOT-W4 BOUNDED-WAIT DESIGN — the world=4 AR wedge: root cause, system-level bounded wait, GREEN-leg plan

Branch `amd/tp4-cure` (NO-GPU design seat). Laws honored: no GPU runs, no builds, **no `src/`
edits** (the patch in §6 is a SKETCH inside this doc only). Untouched by order:
`nvfp4_small_t_hip.cu`, `nvfp4_launch.h`, `nvfp4_smallt_roofline_bench.cu` (in-flight,
another agent's). All paths cited from this worktree at HEAD (`2688e6a29` line of descent).

RED being cured: commit `b420b86fe` / bin `409450b83ffa4be8` — `NINFER_TP_ONESHOT_AR=1` at
TP4 (world=4) wedges the boot **deterministically (2/2)**: warmup never completes, one host
thread spins at 100% CPU, exactly 3/4 rank GPUs sit at 100% (CU occupancy 1 each), the
stranded (0%-GPU) rank **varies** (rank 3 boot B, rank 1 boot B2), and **NO AR-FAILOUT ever
fires** — the R2 bounded-wait law's ~7 s declared death never arrives (17+ min observed).
Evidence: `results/amd/coherence/ONESHOT_{A,B,B2}_serve.log`,
`ONESHOT_{B,B2}_wedge_evidence.txt`, `ONESHOT_AR_row.txt`, design history in
`results/amd/coherence/ONESHOT_W4_notes.md` and `ONESHOT_AR_notes.md`.

---

## 1. Root-cause statement

### 1a. What the code actually contains (re-derived, not trusted from notes)

The bounded wait **IS wired into the world kernel** — this seat re-read the shipped bytes
(git: zero diff between the banked build commit `b420b86fe` and HEAD for
`one_shot_allreduce.cu`, `tp_group.cpp`, `status_transport.h`, so the binary that wedged is
exactly this source):

- `one_shot_allreduce.cu:344-383` (`ar_wait_peer`): HIP arm, per-peer **flag gate then gen
  gate, both bounded** by `timeout_polls` (`kFlagTimeoutPolls = 1<<15`, :24 — ~65-98 ms per
  gate at the measured ~2-3 µs/pinned-host poll); timeout sets the mapped status bit via
  `ninfer_status_set_bits` (the G-AMD-30a **measured** volatile transport,
  `status_transport.h:34-37` — a single volatile RMW store, **not** a spin) and returns
  false; block soft-fails out (`s_fail`, :471-475) **without producing output**.
- Worst-case bounded path per launch: 2 gates × (world-1) peers = 6 ceilings ≈ 0.4-0.6 s;
  host retry loop `:772-854` consumes status after each `cudaStreamSynchronize`, retries
  same gen/slot up to `kArLagRetries = 16` (:763), then `[AR-FAILOUT]` + `_exit(70)`
  (:823-830). Declared death ≈ 7-10 s. This is the R2 law, present and correct **at kernel
  scope**.

So the puzzle is not "the bounded wait is missing". The puzzle is that a mechanism that
cannot stay wedged past ~10 s stayed wedged 17+ minutes **silently**. Resolution below.

### 1b. The two strongest evidence lines

1. `ONESHOT_AR_row.txt:17-20` (boot B): "100% CPU single-thread spin (21 thr); GPUs 0,1,2 at
   100% use, GPU3 0% … NO AR-FAILOUT, NO REJECT-candidate, no soft-fail declaration — the
   R2 bounded-wait law (2x(world-1) poll budgets x16 retries ~= 7 s to declared death)
   NEVER FIRES." A bounded kernel cannot be GPU-resident for 17 minutes; therefore the
   resident spinning kernels are **not** the one-shot world kernel. The only unbounded
   on-device spin-waits in this stack are RCCL collective kernels (NCCL has no kernel-level
   timeout) and host-side `cudaStreamSynchronize` (HIP spin-sync = the 100% CPU thread).
2. `ONESHOT_B_wedge_evidence.txt` vs `ONESHOT_B2_wedge_evidence.txt`: same wedge signature,
   **stranded rank varies** (rank 3, then rank 1). A fixed-order bug (canonical-order,
   NUMA-rank-order, poll-visit-order) would strand the same rank every boot. Variance says
   the trigger is a **race in the cross-rank lockstep**, and the wedge site is one level
   above the kernel.

### 1c. Root cause (top hypothesis)

**The one-shot protocol's liveness contract is lockstep equality of `rank_step` across
ranks, and nothing verifies it.** The gates that guard each peer
(`ar_wait_peer`, `one_shot_allreduce.cu:354/:360`) are **monotone `>=` comparisons** against
my call's stamp. They accept *any newer* peer stamp as satisfaction:

- A peer **ahead** of me (higher gen) passes my gates instantly, and my combine then reads
  whatever payload currently sits in the peer's slot buffer — **the wrong call's data**
  (the gen word is per-rank global, not per-slot, so a stamp says nothing about WHICH
  call's payload is in the slot; the (slot, call) pairing is an assumption, not a checked
  invariant). Lag is thereby silently converted into **wrong-generation combines** instead
  of a loud timeout.
- On retry-after-timeout the same `>=` shape makes the retry pass on the peer's **stale or
  wrapped** stamp without the retry loop ever being able to distinguish "peer caught up
  with my call" from "peer is somewhere else entirely".

One wrong-generation combine early in warmup (the first decode round runs ~130 one-shot
collects; round 1 also carries the largest launch skew, since every rank's stream is still
draining its prefill RCCL collectives) gives one rank a **divergent** hidden state. All four
rank threads then run worker control flow that is data-dependent on AR results
(acceptance, re-prefill/reset branches, round counts — the branch-split family is a known,
banked event: `G18c §4`, `rb n=2/n=1` root cause in `one_shot_allreduce.cu:671-686`). The
divergent rank stops issuing the collective sequence the other three are executing, parks at
the phase barrier (`std::barrier sync_bar(world)`, `tp2_backend.cpp:1751` — sleeping, GPU
0% = **the stranded rank**, which one varies with who loses the race), and the other three
park inside **RCCL collectives whose fourth participant never comes** — RCCL kernels spin
without bound (3 GPUs at 100%, 1 CU each) — while their hosts spin in
`cudaStreamSynchronize`/block in channel enqueue (the one 100% CPU thread).

**Why no AR-FAILOUT:** every death path of the R2 law requires the bounded kernel to *run*.
Once the divergence has happened, subsequent one-shot calls on the desynced ranks keep
"succeeding" via the same `>=` stale/ahead passes (status never set → retry counters never
advance → no death), and on the wedged ranks the one-shot kernel is queued **behind the
stuck RCCL collective on the same per-rank stream** (`tp_group.cpp:346-357` routes NCCL and
one-shot onto the *same* `ctx(r).stream`; the inline `cudaStreamSynchronize` at
`one_shot_allreduce.cu:811` therefore waits on work that never drains). **The bounded wait
is unreachable exactly when it is needed most: the kernel never launches.** The R2 law's
reach ends at the kernel boundary; the wedge lives one level up, in the unbounded waits the
kernel sits behind (RCCL spin, `sync_bar`, spin-sync).

Consistency check against all observations: deterministic 2/2 (≈260 collects per warmup ×
per-collect skew → P(at least one bad catch-up) ≈ 1 per boot), varying stranded rank (who
diverges first is a race), no failout (nothing bounded ever times out on the road to the
wedge; the kernels that could time out never launch), world=2 bit-green (each poller watches
ONE peer, cross-board skew is a 2-party problem, and the 2-rank path's stale-pass blast
radius never produced a divergence in any banked boot).

### 1d. Candidate causes weighed (per tasking)

- **Flag protocol assumptions (arrival-count vs rank-count):** CONFIRMED as the core hole —
  but sharper: there is *no arrival count at all*. Liveness = "each poller's monotone gates
  eventually pass", with `rank_step` equality assumed and never verified. A
  per-collective arrival counter (world-acknowledged) does not exist anywhere in the path.
- **Varied stranded rank (race, not deadlock order):** CONFIRMED — rules out canonical-order
  and NUMA-visit-order deadlock as the *trigger*; the canonical-order law (§1d of
  `ONESHOT_W4_notes.md`) governs the reduce tree's numerics, not liveness, and is not
  implicated by the evidence.
- **Spin-wait without timeout actually wired into the world>=3 path:** REFUTED at kernel
  scope (the HIP arm of `ar_wait_peer` is bounded and correct), **CONFIRMED at system
  scope** — no bound exists on: RCCL collective kernel spin, `sync_bar` waits, the
  stream-frontwork the one-shot sync sits behind, or cross-rank catch-up distance. The
  bounded-wait law is per-launch; nothing composes it into a per-boot liveness bound. (The
  CUDA arm of the world kernel is genuinely unbounded — `one_shot_allreduce.cu:367-382` —
  but never executes on this box; flagged as the same CLASS for the NVIDIA line per R7.)
- **NCCL-vs-one-shot handle mixing:** CONFIRMED as the wedge *amplifier*, not the trigger —
  per-call transport switching (`tp_group.cpp:348`: `n_elems <= 65536` → one-shot, else
  inline `ncclAllReduce` on the same rank stream, same caller thread) makes the one-shot's
  watchdog dependent on unrelated RCCL progress. A wedged RCCL collective renders the
  bounded-wait kernel unreachable (§1c). Mixing is legal only while lockstep holds — the
  same unverified assumption.
- **4-die NUMA/canonical-order effects:** RULED OUT as trigger (stranded rank varies);
  retained as *skew amplifier*: two boards × two dies means cross-board mapped-memory hops
  (measured ~98-101 µs RTT flat, COORDINATOR) widen inter-rank launch skew at world=4 vs
  world=2, which is why the race window opens here and not on the bit-green 2-rank path.

### 1e. BUG CLASS (closure-law naming — the cell must guard THIS, not the instance)

> **CLASS `W4-LOCKSTEP-LIVENESS` — system-unbounded collective wedge behind per-launch
> bounded waits.** At world>2, a lockstep collective whose per-launch polls are bounded but
> whose *system* liveness is not: (i) monotone (`>=`) arrival gates accept any newer peer
> stamp, silently converting peer lag/desync into wrong-generation data; (ii) the collective
> kernel may be queued behind other unbounded transports on the same stream, making its
> bounded wait unreachable; (iii) no watchdog bounds the end-to-end collective crossing, so
> divergence from any cause degenerates into a permanent, silent multi-transport wedge.

Guard the class = the GREEN cell must fire on ANY of: a stalled peer, an ahead peer, a
skipped call, a stuck stream-front — not merely on the T2 repro's exact trace.

---

## 2. Bounded-wait design: wired timeout + progress-clocking + fail-loud-with-diag

Design law: **no silent failover.** The arm does not fall back to the ring on trouble (that
masks the bug and violates the closure law); every bound terminates in a **loud,
diagnostic-rich process death** (`_exit`, distinct codes, one-glance log block). Three
concentric bounds, each independently loud, each naming its measured anchor (LITH discipline;
these are LIVENESS bounds — the R2 law's own shape — never VRAM-law capacity constants, and
they refuse no launch: they only end one that is already wedged):

### B1 — keep the per-launch bounded poll (exists; tighten the semantics)

`ar_wait_peer` stays as shipped. ADD the stale-pass instrument (§2d below) so its two
outcomes become three: PASS-fresh (peer exactly at my call), PASS-stale/ahead (census +
desync window check), TIMEOUT (status bit → retry → death). All three are host-visible.

### B2 — per-collective host deadline (new; closes the silent-catch-up hole)

In `OneShotAllReduce::allreduce_bf16` (world>2 arm), around the retry loop:
- Read a monotonic clock before the first launch; after each `sync + status consume`,
  compare elapsed against `kArCallDeadlineNs = 2 s` (anchor: ~1000× the theoretical worst
  bounded path of ~0.6 s/try × 16 tries ≈ 10 s total — deadline set BELOW the law's ~7-10 s
  declared death so it only fires when the loop itself is stuck, never in normal lag).
- On expiry: do NOT retry silently. Emit the **diag matrix** (below) and die loud
  `[AR-DEADLINE]` `_exit(71)`.
- Every retry of a world>2 call emits `[AR-RETRY] kind=ar` **unconditionally** (the
  `NINFER_MC31` gate is dropped for the N-rank arm; rate is bounded by kArLagRetries per
  call) — a wedge under construction is visible in the log from the first retry.

### B3 — system watchdog thread (new; closes the unreachable-kernel hole — the actual T2 wedge)

One host thread per process, started when the env arm constructs a world>2 one-shot
(`tp_group.cpp:140-149` gate — owner's flip site), stopped at shutdown:

- **Heartbeats:** each rank thread bumps a mapped/atomic `heartbeat[r]` seq counter on entry
  and exit of every collective crossing (`allreduce_local_bf16` both arms, barrier arrivals).
  The one-shot host loop additionally writes `inflight_since[r] = now` and the
  `(gen, slot)` pair into a mapped triple **before** its first launch, and clears
  `inflight_since[r] = 0` after success — this is the **progress clock**, visible to the
  watchdog even when the rank thread is spinning inside a stream sync.
- **Watchdog loop (250 ms period):** for each r: if `inflight_since[r] != 0` and
  `now - inflight_since[r] > kArCallDeadlineNs` (same constant, same measured anchor) →
  WEDGE DECLARED. (A rank stuck inside `cudaStreamSynchronize` behind a never-launched
  kernel or a dead RCCL collective cannot update the clock — exactly the T2 state.)
- **On declaration:** dump the diag matrix (below), tag `[AR-WEDGE-WATCHDOG]`, `_exit(72)`.
  The watchdog must also arm a **barrier census**: `sync_bar` arrivals are counted per
  phase; the dump names which ranks reached which phase — the one-glance "who diverged"
  answer.

### 2d. The diag matrix (fail-loud-WITH-DIAG — the triage artifact)

World×world mapped int pairs `last_obs[observer][peer] = (flag, gen)` written by the kernel
at each gate sample's timeout (one store each — same volatile transport law), plus the
per-rank `(gen, slot, inflight_ms, phase)` host triple. A wedge dump then reads, e.g.:

```
[AR-WEDGE-WATCHDOG] world=4 t=+2.31s since inflight
  rank0 gen=141 slot=13 phase=verify  inflight_ms=2310 peers_last_seen={1:(f141,g141) 2:(f139,g139) 3:(f0,g0)}
  rank1 gen=141 slot=13 phase=verify  inflight_ms=2308 ...
  rank2 gen=139 slot=11 phase=verify  inflight_ms=0      <- one call behind: THE DIVERGED RANK
  rank3 MISSING heartbeat (barrier phase=decode_round_3 arrivals={0,1,2})
```

One glance: who is behind, who is absent, at which (gen, slot), in which phase. No
interpretation needed — the same discipline as `grep -c 'REJECT-candidate'`.

### 2e. Desync window (make the assumption checkable)

The kernel gains a `stale_pass` signal: at the flag-pass instant, if `gen_at_flag >
step_gen` (peer is AHEAD — wrong-generation payload risk is REAL) the kernel sets status
bit 4; host logs `[AR-STALEPASS] rank gen peer gen_at_flag delta` (census line, not a
death — small positive deltas are benign reverse-lag). If `delta > kDesyncWindow = 64`
calls (anchor: 128-slot ring / 2 — a peer half a ring ahead can never be legitimate
lockstep), host dies loud `[AR-DESYNC] _exit(73)`. This converts §1c's silent corruption
channel into a named signal with its own RED/GREEN cell.

### 2f. Coverage of ALL ranks including the non-stranded

- The watchdog watches every rank's clock; the diag matrix is per-observer-per-peer.
- Death is process-global (`_exit`) — all ranks end together with the matrix banked; no
  rank is left spinning as a zombie symptom (the T2 state had exactly that: 3 GPUs pinned
  at 100% for 17 min as an unowned symptom).
- The non-stranded ranks' state at death is the CAUSE EVIDENCE (who they were waiting for),
  which is why the matrix is observer-keyed, not just stranded-rank-keyed.

---

## 3. Four-rank lockstep re-derivation on gfx900 (what guarantees the design needs)

The protocol needs six guarantees; the canonical-order law provides three, the measured
transport provides two, and the design in §2 must ADD the sixth (the one the wedge broke):

- **(V1) publish-before-flag visibility:** payload stores → `__threadfence_system` →
  `*gen = step_gen` → `__threadfence_system` → `*flag = expected_flag` (KAR-v2 chain,
  `one_shot_allreduce.cu:446-450`). On gfx900, VMEM stores are write-through to L2; the
  nontemporal `glc slc` hints pick the system path; **the fence is the visibility edge**,
  the hints only choose the path (`ONESHOT_AR_notes.md §3`). Unchanged by this design.
- **(V2) fresh poll reads:** flag/gen words ride plain volatile dwords (`flat_load_dword
  glc`) — the G-AMD-30a **measured** transport (volatile bits cross host-mapped memory on
  this box; atomic bits do not — hence `ninfer_status_set_bits`' volatile RMW). Payload
  reads are nontemporal 128-bit (`glc` = L1 bypass = the `.cv` semantic), data/control-
  dependent on the flag observation. Unchanged.
- **(V3) identical reduce tree:** CANONICAL ascending-rank left-association
  `((L0+L1)+L2)+L3`, own term from `local_v` (never a self-read) — cross-rank bit-exactness
  BY CONSTRUCTION (witness leg, `ONESHOT_W4_notes.md §3(f)`). bf16 add is commutative, not
  associative; the law is what makes divergence-from-arithmetic impossible so that any
  observed divergence is attributable to the transport/liveness class. Unchanged.
- **(V4) monotone stamps:** per-rank across-boot `rank_gen` (G-AMD-31 pairing law) makes
  recycled slots read "not yet" forever. Unchanged.
- **(V5) slot pairing:** all ranks map call k to slot `k % 128` **only if** their call
  sequences are identical. This is the assumption the wedge broke. It is NOT provided by
  the memory model — it is a protocol invariant, and per §2e it becomes a CHECKED invariant
  (stale-pass census + desync window) instead of an assumed one.
- **(V6) end-to-end liveness:** NEW (§2 B2/B3). The memory model bounds nothing about
  *when* peers arrive; only the three concentric bounds make the crossing bounded end to
  end. On 4 dies / 2 boards the skew amplifier is real (cross-board mapped hops, ~98-101 µs
  RTT measured), so every bound is set ≥ 1000× measured steady state (B1: 1<<15 polls ≈
  65-98 ms vs ~60 µs transfer; B2/B3: 2 s vs ~10 s worst legal bounded path — the deadline
  fires only where B1's own law would already be dead) — a NUMA asymmetry can slow a rank
  but can never fire a bound by itself.

---

## 4. GREEN-leg test plan (deterministic cells, both directions)

Sha law: every row names the artifact sha; RED = pre-fix bin `409450b83ffa4be8` (banked RED
2/2, `b420b86fe`), GREEN = the post-fix bin's boot-stamped sha banked per
`docs/amd/BOOT_LAUNCH_RUNBOOK.md §4`. All device legs ride the per-window boot battery,
results banked per-LABEL. Host-only legs go to the CI farm + PG-1.

### 4a. Cells

| LABEL | what it does | RED (pre-fix) | GREEN (post-fix) |
|---|---|---|---|
| `W4AR-WEDGE-REPRO` | env=1, TP4, boot + warmup, 300 s cap | silent wedge (2/2 banked) | EITHER warmup completes with warmup `[ids] == 760 1156 1018 328` byte-match AND cross-rank bit-exact post-AR witness, OR a loud `[AR-*]` death ≤ 15 s with diag matrix. **No third outcome.** |
| `W4AR-STALL-DRILL` | env `NINFER_AR_TEST_STALL_RANK=<r>`: rank r's world kernel skips its publish (test-only inject) | 17-min silent wedge | ALL ranks die `[AR-DEADLINE]`/`[AR-WEDGE-WATCHDOG]` ≤ ~12 s; matrix names r as never-arrived; exit code 71/72 |
| `W4AR-AHEAD-PEER` | env `NINFER_AR_TEST_AHEAD=<k>`: peer p advances k=2 extra calls before rank r's poll | silent wrong-generation combine, wedge downstream | `[AR-STALEPASS]` census fires; at k>kDesyncWindow: `[AR-DESYNC]` loud death |
| `W4AR-STREAMFRONT-DRILL` | env `NINFER_AR_TEST_STUCK_NCCL=1`: a pre-queued fake collective blocks rank r's stream front | kernel-unreachable silent spin | watchdog declares via `inflight_since` clock (kernel never ran) — proves B3 covers the unreachable-bounded-wait case |
| `W4AR-W2-BITFREEZE` | world=2, env on/off, count600 | (already green) | streams byte-identical on/off AND vs pre-fix binary (guard: the cure must not touch the frozen 2-rank path) |
| `W4AR-W4-IDENTITY` | env=1, TP4, count600 vs ring arm | never served | `[ids]` + completion byte-compare vs `ONESHOT_A_count600.json`; within-arm 4-rank post-AR bitwise equality (canonical-tree witness) |

Injectors are env-gated test-only arms in the kernel/host (both-direction falsifiers per
E-17); default binaries never carry active injectors.

### 4b. Env matrix

`NINFER_TP_ONESHOT_AR ∈ {absent, 1}` × `world ∈ {2, 4}` × `watchdog ∈ {on (implied by arm), off-drill}` ×
`injector ∈ {off, STALL_RANK=3, STALL_RANK=1, AHEAD=2, STUCK_NCCL}`. Absent-env arms must be
byte-identical to default boots at every world (gate discipline). Full matrix at boot
battery cadence; the five RED-direction cells above are the minimum per-sha gate.

### 4c. What a wedge looks like in logs (ops one-glance triage)

PRE-FIX (current, the T2 signature — memorize it):
```
ninfer-serve: warming up...                     <- last line, silence after
ps: 100% CPU, one thread; rocm-smi: 3 GPUs 100% (1 CU each), 1 GPU 0%
grep AR-FAILOUT serve.log  -> 0                 <- the violation of the R2 law
```
Ops rule: **any `warming up...` older than 2 minutes without an `[AR-FAILOUT]`,
`[AR-DEADLINE]`, `[AR-WEDGE-WATCHDOG]`, `[AR-DESYNC]` line = watchdog missing = protocol
violation — file against this doc's class, do not debug the boot.**

POST-FIX: the death is loud, ≤ ~15 s, exit code 70/71/72/73 naming which bound fired, with
the §2d matrix block banked to console + row. `[AR-RETRY] kind=ar` / `[AR-STALEPASS]`
census lines precede any death and are grep-countable like `REJECT-candidate`.

---

## 5. Ownership and sequencing (so the cure lands once)

- `one_shot_allreduce.cu` + `.h`: diag matrix, stale-pass bit, host deadline, mapped
  triples — design-seat sketch §6; landing seat = the AR owner (same seat as the W4
  extension).
- `tp_group.cpp:140-149` (gate site): watchdog thread start/stop + heartbeat hooks — owner's
  (the flip is already owner-owned per `ONESHOT_W4_notes.md §2`).
- `tp2_backend.cpp:1751` (`sync_bar`): per-phase arrival census hook — tp2 owner.
- The GPU-owner then runs §4a's cells; the RED rows on `409450b83ffa4be8` are already
  banked (T2) for `W4AR-WEDGE-REPRO`; the other four need one window with injectors.

---

## 6. SKETCH PATCH (unified diff — NOT applied; no src/ edits by law)

```diff
--- a/src/core/multi_gpu/one_shot_allreduce.cu
+++ b/src/core/multi_gpu/one_shot_allreduce.cu
@@ namespace {
+// W4-LOCKSTEP-LIVENESS cure (docs/amd/ONESHOT_W4_BOUNDED_WAIT_DESIGN.md §2):
+// liveness bounds + diag matrix. LITH anchors: B1 polls ~2-3us (G-AMD-30a box);
+// B2/B3 deadline 2s < the R2 law's own ~7-10s declared death, >1000x steady state.
+constexpr unsigned long long kArCallDeadlineNs = 2000000000ull; // B2/B3, measured-anchor law
+constexpr int kDesyncWindow = 64;                              // §2e: half the 128-slot ring
+constexpr std::uint32_t kArStatusStalePass = 4u;               // status bit 4 (bit 1 = timeout, unchanged)
+
+// Per-rank mapped progress triple + observer-keyed last-seen matrix (§2d).
+struct ArProgress {          // one per rank, mapped pinned, host+device visible
+    volatile std::uint64_t inflight_since_ns; // host writes before first launch; 0 = idle
+    volatile int inflight_gen;                // kernel arg gen, for the dump
+    volatile int inflight_slot;               // slot_idx, for the dump
+};
+struct ArDiagCell { volatile int flag; volatile int gen; }; // last-observed peer state
+struct ArDiagMatrix { ArDiagCell cell[kOneShotMaxWorld]; }; // [observer][peer], mapped
+
 // ar_wait_peer gains the stale-pass instrument: same bounds, three outcomes.
 __device__ __forceinline__ bool ar_wait_peer(volatile const int* peer_flag,
                                              volatile const int* peer_gen,
                                              int expected_flag, int step_gen,
                                              int& gen_at_flag, int& gen_final,
-                                             unsigned int* status,
+                                             unsigned int* status,
+                                             ArDiagCell* __restrict__ diag_cell,   // NEW
                                              unsigned long long timeout_polls) {
 #if defined(__HIP__)
     unsigned long long polls = 0;
     while (*peer_flag < expected_flag) {
         __builtin_amdgcn_s_sleep(1);
-        if (++polls > timeout_polls) { ninfer_status_set_bits(reinterpret_cast<volatile std::uint32_t*>(status), 1u); return false; }
+        if (++polls > timeout_polls) {
+            ninfer_status_set_bits(reinterpret_cast<volatile std::uint32_t*>(status), 1u);
+            diag_cell->flag = *peer_flag; diag_cell->gen = *peer_gen;   // §2d matrix
+            return false;
+        }
     }
     gen_at_flag = *peer_gen;
     polls = 0;
     while (gen_at_flag < step_gen) {
         __builtin_amdgcn_s_sleep(1);
         gen_at_flag = *peer_gen;
-        if (++polls > timeout_polls) { ninfer_status_set_bits(reinterpret_cast<volatile std::uint32_t*>(status), 1u); gen_final = gen_at_flag; return false; }
+        if (++polls > timeout_polls) {
+            ninfer_status_set_bits(reinterpret_cast<volatile std::uint32_t*>(status), 1u);
+            diag_cell->flag = *peer_flag; diag_cell->gen = gen_at_flag;
+            gen_final = gen_at_flag; return false;
+        }
     }
     gen_final = gen_at_flag;
+    if (gen_at_flag > step_gen) {   // §2e: peer AHEAD = wrong-generation payload risk
+        ninfer_status_set_bits(reinterpret_cast<volatile std::uint32_t*>(status), kArStatusStalePass);
+    }
     return true;
 #else
     ... /* CUDA arm unchanged — R7, NVIDIA seat's cell */ ...
 #endif
 }

 __global__ void one_shot_ar_pinned_vec_kernel_world(..., unsigned int* status,
+                                                    ArDiagCell* __restrict__ my_diag_row,
                                                     unsigned long long timeout_polls) {
     ...
         for (int p = 0; p < world; ++p) {
             if (p == rank) continue;
             ar_ok = ar_wait_peer(peers.flag[p], peers.gen[p], expected_flag, step_gen,
-                                 gen_at_flag, gen_final, status, timeout_polls) && ar_ok;
+                                 gen_at_flag, gen_final, status,
+                                 &my_diag_row[p], timeout_polls) && ar_ok;
             if (trace && (step_gen <= 16 || gen_at_flag < step_gen)) { /* unchanged KAR-W print */ }
             if (!ar_ok) break;
         }
     ...
 }

 struct OneShotAllReduce::Impl {
     ...
+    ArProgress* host_progress[OneShotAllReduce::kMaxWorld] = {};
+    ArDiagMatrix* host_diag[OneShotAllReduce::kMaxWorld] = {};
+    int* dev_diag_view[OneShotAllReduce::kMaxWorld] = {};
     // ctor: hipHostAlloc(Mapped|Portable) + hipHostGetDevicePointer per rank, zeroed —
     // same alloc shape as host_status (:614-622). ~world*(24+8*world) bytes total.

 void OneShotAllReduce::allreduce_bf16(int rank, ...) {
     ...
+    // B2 progress clock (host): stamp BEFORE launch, clear after success. The watchdog
+    // reads this even when this thread spins inside cudaStreamSynchronize below.
+    auto now_ns = []() {
+        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
+        return (std::uint64_t)ts.tv_sec * 1000000000ull + (std::uint64_t)ts.tv_nsec;
+    };
+    impl_->host_progress[rank]->inflight_since_ns = now_ns();
+    impl_->host_progress[rank]->inflight_gen = (int)gen;
+    impl_->host_progress[rank]->inflight_slot = (int)slot_idx;
+    const std::uint64_t deadline = now_ns() + kArCallDeadlineNs;
     const bool mc31h = ...;
     for (int try_no = 0;; ++try_no) {
         ...
         } else {
             one_shot_ar_pinned_vec_kernel_world<<<blocks, threads, 0, stream>>>(
                 ..., reinterpret_cast<unsigned int*>(impl_->dev_status_view[rank]),
+                reinterpret_cast<ArDiagCell*>(impl_->dev_diag_view[rank]),
                 kFlagTimeoutPolls);
         }
         CUDA_CHECK(cudaStreamSynchronize(stream));
         const int pending = *impl_->host_status[rank];
-        if (pending == 0) break;
+        if (pending & kArStatusStalePass) {
+            // §2e census — loud signal, not a death (small deltas = reverse lag).
+            std::fprintf(stderr, "[AR-STALEPASS] rank=%d gen=%llu slot=%zu\n",
+                         rank, (unsigned long long)gen, slot_idx);
+        }
+        if (now_ns() > deadline) {
+            // B2: never retry silently past the deadline — dump + die (§2/§2d).
+            this->dump_wedge_matrix_and_exit(71, "[AR-DEADLINE]", rank, gen, slot_idx);
+        }
+        if (pending == 0) { break; }   // clean: commit the ring position
         *impl_->host_status[rank] = 0; // clear for retry (bit-4 census is sticky-cleared here too)
         ...
     }
+    impl_->host_progress[rank]->inflight_since_ns = 0;   // B2: idle stamp
     impl_->rank_step[rank]++;
 }
+
+void OneShotAllReduce::dump_wedge_matrix_and_exit(int code, const char* tag, ...) {
+    // §2d: one-glance block — per-rank (gen, slot, inflight_ms) + observer-keyed
+    // last-seen peer (flag, gen) pairs + heartbeat/phase if the tp2 hooks are in.
+    std::fprintf(stderr, "%s world=%d rank=%d gen=%llu slot=%zu\n", tag, impl_->world, ...);
+    for (int r = 0; r < impl_->world; ++r) { /* print host_progress[r], host_diag[r] rows */ }
+    std::fflush(stderr); std::fflush(stdout);
+    ::_exit(code);
+}
 --- a/src/core/multi_gpu/one_shot_allreduce.h
+++ b/src/core/multi_gpu/one_shot_allreduce.h
@@
+    void dump_wedge_matrix_and_exit(int code, const char* tag, int rank,
+                                    std::uint64_t gen, std::size_t slot);   // §2d
+    ArProgress* progress(int rank) const;   // for the watchdog + tp2 heartbeat hooks
+    ArDiagMatrix* diag(int rank) const;
 --- a/src/core/multi_gpu/ar_watchdog.h  (NEW, host-only — CI-farm testable, no device)
+++ b/src/core/multi_gpu/ar_watchdog.h
@@
+// B3 system watchdog (§2, docs/amd/ONESHOT_W4_BOUNDED_WAIT_DESIGN.md). One thread per
+// process, armed by the tp_group gate when a world>2 one-shot is constructed. Period
+// 250 ms; declares a wedge iff inflight_since_ns is nonzero and older than
+// kArCallDeadlineNs (the same measured anchor as B2 — a rank stuck inside
+// cudaStreamSynchronize behind a never-launched kernel or a dead RCCL collective cannot
+// clear its own clock; this is exactly the banked T2 state). Fires LOUD:
+// dump + _exit(72). Never refuses a launch (liveness bound, not a VRAM-law constant).
+class ArWatchdog { public: void arm(const OneShotAllReduce* ar); void disarm(); };
 --- a/src/core/multi_gpu/tp_group.cpp   (OWNER'S — sketch only, per ONESHOT_W4_notes.md §2 convention)
+++ b/src/core/multi_gpu/tp_group.cpp
@@
     } else if (tp_oneshot_ar && I.n <= OneShotAllReduce::kMaxWorld) {
         I.one_shot = std::make_unique<OneShotAllReduce>(I.n);
+        I.ar_watchdog.arm(I.one_shot.get());   // B3: world>2 arm can never wedge silently
     }
```

Notes for the landing seat: (1) the diag/progress allocations must join the existing
per-rank mapped-alloc block so the dtor free-loop covers them (null-guarded, world=2
byte-frozen); (2) `dump_wedge_matrix_and_exit` must print the phase census only if the tp2
hook landed, else omit the line — never print a placeholder; (3) injectors for §4a are
test-only env arms in the kernel/host and must default-absent.

---

## 7. RED/GREEN ledger (open rows)

| row | RED sha | GREEN sha | cell | status |
|---|---|---|---|---|
| W4AR-WEDGE-REPRO | `409450b83ffa4be8` (banked 2/2, `b420b86fe`) | — post-fix bin pending | §4a | RED banked; GREEN owed |
| W4AR-STALL-DRILL | owed (one injector window) | owed | §4a | open |
| W4AR-AHEAD-PEER | owed | owed | §4a | open |
| W4AR-STREAMFRONT-DRILL | owed | owed | §4a | open |
| W4AR-W2-BITFREEZE | n/a (regression guard) | owed | §4a | open |
| W4AR-W4-IDENTITY | never served at TP4 (T2) | owed | §4a | open |

Ring anchor for the perf leg (unchanged by this design): 7.63 ms/round, 5.9% of round
(`ONESHOT_A_serve.log` B1, 200 rounds) — the one-shot arm still owes its measured ms/round
before any adoption talk (T2 verdict 1 stands).

## 8. Honest residuals

- The §1c trigger (first wrong-generation combine via stale/ahead pass) is the top
  hypothesis, not a device-proven fact — no run has yet captured a `[AR-STALEPASS]` or a
  diverging rank live (the post-fix census exists precisely to capture it). The CLASS and
  the cure are robust to the alternative trigger variants (pure barrier desync, stuck NCCL
  front): every variant lands in the same B1/B2/B3 nets and dies loud with the matrix.
- The L2-stale-line residual on host-rewritten staging (`ONESHOT_AR_notes.md §3`) remains
  open and orthogonal; if it is the real trigger, the `W4AR-AHEAD-PEER`/RETRY census and the
  KAR-W `REJECT-candidate` line will name it, and the fix conversation is the
  `buffer_wbinvl1_vol` family — not this design.
