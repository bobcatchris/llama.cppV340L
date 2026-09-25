# ONESHOT-W4: one-shot AR N-rank extension — `one_shot_allreduce.cu` (world=4 ready)

Branch `amd/tp4-cure` (no-GPU seat), riding `ONESHOT-AR` 4967b90cd (the gfx900-legal
128-bit nontemporal one-shot). Scope honored: only `src/core/multi_gpu/one_shot_allreduce.cu`
(+ its header, for the world ctor) touched; `tp_group.cpp`, `tp2_backend.cpp`,
`text_context_impl.h`, `one_shot_argmax.*`, `HipSources.cmake`, `optrace_analyze.py` NOT
edited (owners'). The 2-rank kernel `one_shot_ar_pinned_vec_kernel` is BYTE-UNTOUCHED and
remains the world==2 path (default ctor still constructs world=2 — the NVIDIA line and
world=2 serving ride it bit-identically; only the throw-message TEXT of the rank guard
changed, and that path never fires for a valid world=2 call).

## 0. Tree facts at task time

- `tp_group.cpp:136-139` constructs `OneShotAllReduce` + `OneShotArgmax` only at `I.n == 2`;
  `:338` routes `n_elems <= kMaxElements` (65536) to it whenever constructed. The ROUTE is
  already world-agnostic — the ONLY flip needed is the construction gate (§2). No
  `HipSources.cmake` change: the TU is already whitelisted (`NINFER_HIP_STEP1_SOURCES`).
- `kMaxWorld = 8` mirrors `kR1MaxWorld` (argmax_r1.h:36) — same fixed-wire law, same
  construction-time refusal pattern as `require_r1_shape`. tp_group's own `kR1MaxWorld`
  check (tp_group.cpp:130) already caps group world at 8, so the AR refusal is unreachable
  through TpGroup — it guards direct mis-construction only.
- Staging cost at world=4: 128 slots × 4 ranks × 128 KiB = 64 MiB pinned host (world=2 was
  32 MiB). Trivial, stated per disk/pinned discipline.

## 1. Design

### 1a. Slot layout at world=4 (identical mechanics to 2-rank, N-ranked)

```
slot s = rank_step[r] % kNumSlots (128)   — all ranks map call k to the SAME s (see 1c)
payload  <= 65536 bf16 = 128 KiB          — the ~10 KB hidden vector rides fine

                     rank 0          rank 1          rank 2          rank 3
                   +--------------+ +--------------+ +--------------+ +--------------+
slots[s].host_buf  | L0  (bf16^n) | | L1           | | L2           | | L3           |  mapped pinned; publish leg
slots[s].flag      | f0           | | f1           | | f2           | | f3           |  volatile dword / slot+rank (G-AMD-30a)
impl gen (NOT      | g0           | | g1           | | g2           | | g3           |  PER-RANK KAR stamps, across-boot
  per-slot)        +--------------+ +--------------+ +--------------+ +--------------+  monotonic (G-AMD-31 pairing law)
impl status        | st0          | | st1          | | st2          | | st3          |  soft-fail words (unchanged law)

call k, rank r (one block, 1024 threads):
  PUBLISH  st_writethrough_uint4(local -> slots[s].host_buf[r])  (128-bit glc slc stores)
           __threadfence_system; g_r = gen; __threadfence_system; f_r = gen   <- KAR-v2 chain
  POLL     tid==0: for each p != r (ascending): wait f_p >= gen (bounded, s_sleep),
           then capture g_p at flag-pass and wait g_p >= gen (the torn-read instrument)
  REDUCE   for each i: acc = ((L0 + L1) + L2) + L3   <- CANONICAL ascending-rank tree;
           peer terms via global/flat dwordx4 glc slc loads, OWN term from local_v
           (never a self-read of own slot); residual += acc (fused, unchanged semantics)
  LATENCY  one publish + one poll sweep + one read sweep — independent of world
           (NCCL ring at TP4: 2x(N-1) = 6 serialized PCIe hops per collect;
            128 collects/round x ~98-101 us ~= 13 ms/round vs one-shot sweep target 2-4 ms)
```

### 1b. Transport discipline carried over — ISA verified (this toolchain, §5)

Whole-TU device ISA dump (`--cuda-device-only -S`): `slc`-bearing ops went 3 → 8, exactly
the prior 3 helper sites (2-rank kernel, untouched) + the world kernel's 5 (publish store
×1, peer-read loads ×4 across the residual/no-residual arms):

| leg | 2-rank kernel (prior commit, untouched) | world kernel (this commit) |
|---|---|---|
| publish store | `global_store_dwordx4 … off glc slc` | `global_store_dwordx4 … off glc slc` — SAME form |
| peer read | `global_load_dwordx4 … off glc slc` | `flat_load_dwordx4 … glc slc` — see below |
| flag/gen/status words | `flat_load_dword … glc` (volatile, G-AMD-30a MEASURED) | UNCHANGED (same helper-free volatile derefs) |

Honest delta, stated not hidden [design]: because the peer pointers arrive through the
BY-VALUE `PeerArgs` kernel param, the world kernel's read leg lowers to the FLAT segment
instead of global. The memory-semantic content is identical — one 128-bit transaction,
`glc` = L1 bypass = the `.cv` semantic, `slc` = system-line hint = the `.wt`-class path —
and FLAT IS THE SEGMENT OF THE G-AMD-30a MEASURED transport: the flag/gen words have
crossed these same host-mapped addresses as `flat_load_dword glc` through every
KAR-validated boot. Restrict-qualifying the pointer array does NOT flip the segment
(tried, ISA-checked — the qualifier stays for the noalias contract). If the device window
ever wants the global-segment form anyway, the one-line owner experiment is hoisting the
per-world pointers to direct `__restrict__` kernel params (arg-space cost, zero semantic
delta) — a falsifier arm, not a fix.

### 1c. The four hard questions (per the mission)

- **Slot addressing at world=4**: unchanged arithmetic (`step%kNumSlots`, step advanced
  per-rank on success only). Every rank executes the same collective call sequence, so
  their `rank_step` values stay equal and all ranks map call k to the SAME `slot_idx` —
  this pairing is ALREADY the 2-rank path's load-bearing assumption (its kernel reads
  `slot.host_buf[peer]` at the reader's slot index). World=4 inherits it, blast radius x3.
- **Flag/gen handshake per slot vs per-rank**: per-slot flags (one per rank per slot —
  `slots[s].flag[p]`) and PER-RANK gens (`host_gen[p]`, NOT slot-scoped). The gen carries
  the across-boot monotonic commit stamp (G-AMD-31 pairing law); making gen per-slot would
  reset the domain every slot trip and resurrect the recycled-slot auto-accept hole. Each
  peer is gated flag-then-gen exactly as the 2-rank gate; the KAR witness captures
  `gen_at_flag` per peer.
- **Fence/visibility ordering**: unchanged chain, per rank: payload stores →
  `__threadfence_system` → gen → `__threadfence_system` → flag; peers poll via the
  G-AMD-30a volatile transport, then issue nontemporal payload reads data/control-dependent
  on the flag observation (single leader thread). The nontemporal helpers choose the path;
  the fence is the edge.
- **Buffer sizing**: 4 slots live at once (one per rank) = 512 KiB per slot trip; 128-slot
  ring = 64 MiB pinned total. Payload here ~10 KB hidden vector; kMaxElements 65536 covers
  T=12 x 5120. Trivial, nothing resized.

### 1d. CANONICAL ORDER LAW (new, load-bearing for TP4)

bf16 add is commutative but NOT associative. If each rank started the sum from its own
contribution, the four ranks would evaluate four different trees and diverge at rounding —
a coherence bug by construction. Every rank therefore evaluates the SAME left-associated
ascending-rank tree ((L0+L1)+L2)+L3; its own term joins from `local_v` (no self-read).
Within the one-shot arm, cross-rank bit-exactness is then BY CONSTRUCTION (witnessable,
§3(f)). One-shot's tree still differs from NCCL ring's reduction order — that is a
numerics-order fact for the §3(d) classification, not a transport bug.

## 2. What the GPU owner must flip (EXACT patch — this seat did NOT edit tp_group.cpp)

The route (`tp_group.cpp:338`) needs NO change. Only the construction gate changes,
env-gated, default OFF (E-17 discipline: absent env = byte-identical default behavior):

```cpp
    // tp_group.cpp:136 — BEFORE:
    if (I.n == 2) {
        I.one_shot = std::make_unique<OneShotAllReduce>();
        I.one_shot_argmax = std::make_unique<OneShotArgmax>();
    }
    // AFTER (ONESHOT-W4 owner flip; env NINFER_TP_ONESHOT_AR=1):
    static const bool tp_oneshot_ar = [](){ const char* v = getenv("NINFER_TP_ONESHOT_AR");
                                            return v != nullptr && v[0] == '1'; }();
    if (I.n == 2) {
        I.one_shot = std::make_unique<OneShotAllReduce>(I.n);   // world=2 == old default, bit-frozen path
        I.one_shot_argmax = std::make_unique<OneShotArgmax>();
    } else if (tp_oneshot_ar && I.n <= OneShotAllReduce::kMaxWorld) {
        I.one_shot = std::make_unique<OneShotAllReduce>(I.n);   // N-rank AR; OneShotArgmax stays world==2 (R6)
    }
```

Notes for the owner: `OneShotAllReduce(2)` is behaviorally the old default ctor (same
kernel, same staging sequence); the `else if` arm deliberately does NOT construct the
argmax at world>2. `advance_epoch`/`reset_step`/`last_call_timed_out` forwarders
(tp_group.cpp:373-383) are world-safe after this commit (rank range [0, world)). Whether
the tp2 side should CALL them at ranks 2-3 is the tp2 owner's runtime decision.

## 3. Device-test plan for the GPU owner (no-GPU seat wrote this; owner executes)

Gate: env-gated, default OFF; default binary behavior unchanged when absent. RED/GREEN
closure law: every leg names the pre-fix and post-fix artifact shas; results banked
per-LABEL with the boot battery.

(a) **2-rank regression falsifier (first, cheap)**: world=2 boot, env OFF vs ON — token
streams must be byte-identical to each other AND to the pre-commit binary's streams (env
only changes the ctor arg at world==2; this proves the bit-freeze). Any divergence here is
a regression in THIS commit — stop, no TP4 leg.
(b) **World=4 boot battery**: `NINFER_TP_ONESHOT_AR=1` at TP4, guard cells + ladder;
`[A1TRACE-KAR-W]` lines present for steps 1-16 (positive control: verdict=ACCEPT on all
peer pairs), then `grep -c 'REJECT-candidate' serve.log` == 0 steady state; `NINFER_MC31`
census: kind=ar retry lines == 0 steady state (G-AMD-34 field already carries the kind).
Any REJECT-candidate is the torn-read window caught live — loud by design, do not mask.
(c) **count600 A/B + clocks sidebands**: A = default transport (NCCL ring, 128 serialized
collects/round ~= 13 ms naive), B = one-shot arm. `rocm-smi` clock samples banked alongside
both. Report MEASURED ms/round both arms (COORDINATOR RTT law: one-shot should land in the
2-4 ms/round class if the sweep dominates; transport is host-staged PCIe, no P2P).
(d) **[ids] + token byte-check vs ring (re-scoped E-17 bar)**: temp 0, count600, SAME
geometry — capture token stream + draft [ids] from ring arm and one-shot arm. Within the
one-shot arm, all 4 ranks must agree bit-exactly (§1d, canonical tree — this is the
COHERENCE witness). One-shot vs ring may diverge at bf16 rounding on near-tie logits
(different reduction trees): classify FIRST per E-17 discipline (numerics-of-order vs
coherence) before anything may be called a transport bug. Cross-arm acceptance <=
target-argmax if an MTP leg rides the window.
(e) **KAR witnesses at world=4**: as (b) plus the slot-reuse leg — one run long enough to
wrap the 128-slot ring (>2 requests at ~64 AR calls/request); stale-slot reuse must show
zero REJECT-candidate (stamp ordering law) — the L2-stale-line falsifier from
ONESHOT_AR_notes §4(f) carries over unchanged.
(f) **Cross-rank bit-exactness witness (new, world=4-specific)**: at least one leg dumping
the post-AR buffer from all 4 ranks (debug tap or ladder cell) — bitwise EQUAL across
ranks is the closing artifact for §1d. If it ever differs, the canonical tree is broken —
a bug, with this leg as its RED capture.
(g) **Soft-fail drill**: kill/pause one rank mid-round at world=4 — the surviving ranks'
blocks must exit via the bounded wait (status bit), host retry census arms, and after
kArLagRetries the process dies loudly (AR-FAILOUT). No silent hang. Bounded-wait ceiling
at world=4 is 2x(world-1) = 6 poll budgets per try (~0.4-0.5 s) — verify the declared
death time matches, never estimated VRAM/ETA.
(h) **Parity/whitelist**: re-run HipSources parity check-(h) on these bytes (TU already
whitelisted; no cmake change shipped).

## 4. Risks

- **R1 — PCIe read amplification**: each rank now reads 3 peer slots per call (~30 KB
  payload-class traffic) vs 1 at world=2. Latency-class at ~10 KB/slot; the clocks
  sidebands in §3(c) are the check.
- **R2 — bounded-wait ceiling x(world-1)**: per-peer budgets (matching the 2-rank flag-
  then-gen shape) make worst-case block wait 2x(world-1) ceilings ~= 0.4-0.5 s/try at
  world=4; x16 retries ~= 7 s to declared death. Bounded, but slower to declare than the
  2-rank path. A shared-budget variant would tighten it at the cost of cross-peer
  starvation semantics — left as a measured follow-up if §3(g) shows the drill outliving
  patience.
- **R3 — collective-lockstep slot pairing**: all ranks must execute the same AR call
  sequence or `rank_step` desyncs and the poll sweep wedges (existing census shows it).
  Not new (2-rank rides the same assumption), but a TP4 branch-split now strands 3 peers.
- **R4 — one-shot tree ≠ ring tree**: §3(d) classification burden is real; do not skip it.
- **R5 — residual semantics**: fused `residual += full_sum` per rank, unchanged from
  2-rank; tp2 callers own whether world=4 wants that (owner's runtime decision, §2 note).
- **R6 — OneShotArgmax stays world==2**: the owner snippet deliberately does not construct
  it at world>2; TP4 argmax traffic keeps its existing path (argmax owners' scope).
- **R7 — CUDA arm of the world kernel compiles only under nvcc** (this seat exercised the
  HIP arm, §5): the NVIDIA line never reaches it (they construct world=2 → old kernel).
  The CUDA arm mirrors the 2-rank kernel's unbounded-poll semantics; if the NVIDIA line
  ever opts into N-rank one-shot, its bounded-wait/soft-fail parity is THEIR seat's cell.
- **R8 — int32 gen width**: ~2^31 calls/rank wrap law unchanged (carried verbatim).
- **R9 — pinned host memory 64 MiB** (world=4) vs 32 MiB: trivial, declared.

## 5. Compile status (this seat, zero-GPU)

```
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -O3 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 \
  -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I src/common/hip_shim -I include -I src -I third_party -DNDEBUG -std=gnu++20 \
  --offload-arch=gfx900 -x hip -c src/core/multi_gpu/one_shot_allreduce.cu -o oneshot_check.o
# RC=0, 0 errors, 24 warnings — ALL the pre-existing host nodiscard class
# (hipSetDevice/hipFree/hipFreeHost), same count and class as ONESHOT-AR 4967b90cd
# (12 host + 12 gfx900); object 93600 B (was 55464 B — the second kernel + widened staging).
# Device ISA: slc-bearing ops 3 -> 8 (§1b); flag/gen words remain plain volatile flat dword.
```

Not committed to amd/main; branch `amd/tp4-cure` only, commit prefix `ONESHOT-W4:`.
