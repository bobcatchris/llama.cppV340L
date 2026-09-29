# GFX906 DIFFERENTIAL (agent5) — control-tree debug vs /tmp/ninfer-gfx906

**Lane**: `amd/wo-gfx906-diff` @ `/home/chris/worktrees/amd-wo-gfx906-diff`, base `785bca6f`.
**Control**: `/tmp/ninfer-gfx906` @ `7a3c18d9` (branch `gfx906-port`, read-only, chris-owned).
**Method**: zero-card; file reads + `git log/show` in both trees. Predecessor rows read first
(TP4_AR_TRANSPORT_DECISION §5-4 amendments, TP4_GEOMETRY_SHADOW S3/S4) — nothing cited from them
is re-used as fact; all rows below re-derived from the trees at the shas named.
**Companion**: `GFX906_QUICKDIFF_agent5.md` @ `0adce46b` (chair 18:4xZ re-scope: Q1-Q4) — this doc
is its full three-section form; the quickdiff stays the "why it works" provenance if tonight greens.

**Lineage predicate (frames every row)**: `git log --all --oneline -- '*one_shot*'` in the control
tree = **zero commits**, `git rev-list --all | ls-tree grep` = zero objects on every ref. The donor
never had a `src/core/multi_gpu/one_shot*` family; ours was invented on our side of the fork
(`e5d798cd` 2026-08-20 "vectorized 128-bit OneShotAllReduce on mapped pinned host memory").
"Diff the files" therefore resolves to a DESIGN diff. The asymmetry the chair flagged — their
same-config records byte-identical, ours per-request forks — has a structural reading: their
eager transport keeps **no device-visible protocol state across calls**, and ours keeps six kinds
(step counters, per-slot epoch flags, per-rank gen, mapped status words, per-request resets, and a
ring index derived from all of them).

---

## §1 — TRANSPORT DIFF (predicates + anchors)

### 1.1 The two designs, side by side

| aspect | DONOR (gfx906 @7a3c18d9) | OURS (amd/main lineage) |
|---|---|---|
| AR data path, eager | host-API choreography: `record(inputs_ready)` → `waitInputs` → `cudaMemcpyAsync` D2D pull → `record(pull_done)` → local combine kernel — `src/ops/common/allreduce.cu:444-470` (allreduce_sum), `:558-589` (allgather_rows) | in-kernel handshake: publish payload to OWN pinned slot (`one_shot_allreduce.cu:96`), fence, publish gen+flag (:117-119), **poll PEER host flag across PCIe** (:133), read PEER payload (:191,:220) |
| AR data path, captured | one kernel per rank, PUSH slice into PEER's uncached device staging, store seq into PEER's signal, **poll OWN flag** (spin never crosses PCIe) — `allreduce.cu:185-217` (`flag_allreduce_bf16_kernel`; `flag_barrier` :167-183), gate `capturing(ec)` :424,:517 + commit `7c02856c` | same in-kernel handshake as eager — `tp_group.cpp:268` gates on payload size ONLY; no capture check, no env off-switch; our eager corpus (`--no-cuda-graph`, G18 runbook :103) runs it 128×/token |
| Token pick | cross-rank argmax DOES NOT EXIST: vocab is allgathered to full `[V,1]` LOCAL tensor per rank (`text_context_impl.h:1891,:1919-1929`), then `ops::argmax(whole[0],…)` `:2503` on local bytes; tie rule lowest-index `src/ops/kernel/argmax.cuh:22-24` | `one_shot_tp_argmax_kernel`: token = f(PEER payload read over PCIe inside the kernel) — `one_shot_argmax.cu:199-236`; the pick ITSELF is on the handshake critical path |
| Issuer model | ONE host thread enqueues both ranks, back-to-back, program-order serialized — `text_context_impl.h:1458` + deadlock-rule comment `allreduce.cu:426` | TWO rank host threads, independent `rank_step[rank]++` counters — OURS `tp2_backend.cpp:3270-3280,:4331,:7368` + `one_shot_allreduce.cu:376`, `one_shot_argmax.cu:386` |
| Cross-call identity | per-call: events re-recorded (fresh by construction), or monotonic per-block seq in DEVICE uncached memory, NEVER reset (`allreduce.cu:196,:232`) | per-rank host counters + slot ring (`step % kNumSlots`) + epoch = `step/128+1` (AR, `one_shot_allreduce.cu:377-378`) / step+1 (argmax, `one_shot_argmax.cu:393-396` S1 fix) + per-request `reset_step` |
| Recycle protection | staging reused every call but guarded by a SECOND (end) barrier before exit — `TP2-SLICES.md:229-230`, `end[b]` at `allreduce.cu:210,:246` | depth-only: publish-before-wait; race closes with slack only IF call counts stay equal (chain-unroll in §3.5 of the quickdiff); no destroy gate |
| Memory classes refused by measurement | fine-grained signal = DEADLOCK (L2-served local poll); plain cudaMalloc staging = STALE PARTIALS ("DIFFER at 5 rounds"); poll-peer-via-UVA + peer-READ = deadlocks/stale (remote VRAM cacheable, no L2 writeback on gfx906) — probe matrix `TP2-SLICES.md:231-236` | host-pinned mapped (`cudaHostAlloc`), componentwise volatile u32 publish (`one_shot_allreduce.cu:24-38` — the #784 tear question), device-memory uncached NOT used at all (grep `MallocUncached\|Finegrained` OURS = 0 hits in src/) |
| Timeout shape | 1<<26 polls, DEVICE flags (sub-us polls); on expiry `atomicOr(status)` into uncached DEVICE word, `allreduce.cu:169-177` | 1<<15 polls, PINNED-host flags (~2-3 us/poll; arithmetic comment `one_shot_allreduce.cu:13-20`); status = `atomicOr` into HOST-MAPPED word (:135,:159) — **and per eb4846df the mapped atomicOr raise channel is dead on gfx900**: their timeout status crosses (device uncached), ours may not |

### 1.2 The slice-9 history verdict: we carry neither their fix nor their pre-revert state

Their 99bd9f9d→7a3c18d9 sequence, from `git show` + TP2-SLICES S9/S9b/S9c:
- `99bd9f9d` flag-sync transport + per-rank split graphs (mxxm idiom; the memory-class findings above).
- `7c02856c` **capture-only gate**: eager/prefill keep the event transport ("byte-identical S7 path").
- `de210ae4` default flip: flag-sync ON at tp2.
- `7a3c18d9` **revert to opt-in** after request 12 of h2h-v2 (`reuse=full_reset`, sampling) wedged
  card 2 — GPU hang, MODE1 reset FAILED, box reboot (`TP2-SLICES.md:275-291`). Their own reading:
  "a spin-wait inside a per-device graph that never sees its flag is exactly the shape that wedges
  an MI50"; named candidates: stale flag epoch at full_reset re-arm, staging overrun on wider
  verify batch, MTP verify-width switch under sampling (:283-286). **Never root-caused — withdrawn.**

Our tree runs (a) the poll-remote/read-remote DIRECTION they measured dead on Vega-family silicon
(row above), (b) inside the EAGER path that `7c02856c` exists to protect, and (c) with a
per-request re-arm surface (`reset_step`, six call sites in `tp2_backend.cpp` :1921,:2147,:3740,:5536…)
that is the same CLASS as their wedge candidate. None of their three mitigations is in our lineage:
capture-only gating, uncached device staging, and end-barrier recycling. Conversely our bounded
poll + soft-fail + deferred throw is mitigation they never built — and whose raise rides a channel
our own board has proven dead (b28d25fd: "the ATOMIC is" dead, not the mapped channel). Net: the
trees do not share a transport at all; every determinism argument must be re-made per design, and
"they're deterministic with the same transport" is FALSE as a premise — it is not the same transport.

### 1.3 What their determinism records actually cover (so nobody over-cites them)

Their byte-identity claims (all @TP2-SLICES): S9b parity "byte-for-byte the S6 numbers" (:265) and
graph_tp2 "reproducible across two graph runs" — the TEST (`test_graph_tp2.cpp:13-14,:217-218,
:233-242`: same engine, two `generate_greedy` calls = two in-process requests, exact vector
equality, first-differing-token report) runs with **events (eager) or flag-sync (capture-only)**:
both variants have no per-request counter to skew. The serve-path stress that WEDGED was flag-sync
under `full_reset`. So their evidence: (i) licenses "same-config identity is achievable on this
engine family" — the chair's reading, correct; (ii) does NOT cover a counted-ring handshake
re-armed per request — that construct never ran there, so their green rows neither prove nor
disprove our suspect; they only sharpen it (the delta is exactly the thing their records cannot see).

---

## §2 — KNOWN-BAD CENSUS (their written refusals × our serving closure)

Refusal sources: `PASS2-DESIGN.md:78` (fp32-atomic 2-way K-split — "avoid, breaks
bit-determinism"), `TP2-SLICES.md:231-236` (memory-class refusals above), S9c (unbounded/concealed
spin under serve). Census scope: files compiled into ninfer-serve TP2 greedy decode — predicate:
grep over `src/` (708 tracked .cu/.cuh/.h/.cpp) + the tp2 closure's call graph from
`text_context_impl.h` / `tp2_backend.cpp`.

| # | refused construct | predicate run | OURS result | route if hit |
|---|---|---|---|---|
| C1 | fp32 atomic accumulation in any reduction | `grep -rn atomicAdd src/ include/ --include=*.cu --include=*.cuh --include=*.h --include=*.cpp` → 15 hits, each type-checked by hand | **CLEAN — zero float/double/bf16 atomicAdd.** Hits: i32 counters — `sampling.cuh:89,:275` token_counts `std::int32_t*` (declared `include/ninfer/ops/sampling.h:32`), `group_done` gates (:183,:230; `speculative_round.cuh:351,:393`), `admit_diag` int32 diagnostics (`dflash2_attention.cu:57` signature `std::int32_t*`), MoE rank counts (`sparse_moe_prefill_kernels.cu:163`). Identical set exists in the donor tree (same files) — inherited, not delta | — |
| C2 | split-K reductions with nondeterministic combine | grep `split.?k\|k.?split` | one SplitK family: `bf16_gdn_gating_proj_gemm_mma.cuh:58-341` — partials staged in a buffer, combined by a FIXED-ORDER serial loop `for (s = 0; s < SplitK; ++s)` (:315,:330,:341), no atomics — conformant with their rule (deterministic despite K-split) | — |
| C3 | 2-way K-split GEMV shapes à la PASS2 | their `q_gemv_gfx906` pass-2 route set (`q5_linear_add_gemv.cu:21-26` etc., gate `gfx906_pass2_gemv_enabled` — ours: `NINFER_GFX906_PASS2` absent; our T=1 routes are the w8/q4/q5 SIMT + mma lines, single-accumulator per row | N/A — we never ported pass-2 GEMV; the refusal never had a chance to bind here | — |
| C4 | fine-grained/uncached device staging for cross-rank flags+payload | `grep -rn 'Finegrained\|MallocUncached\|hipExtMallocWithFlags' src/` OURS = **0 hits**; all cross-rank state is `cudaHostAlloc(Mapped\|Portable)` (`one_shot_allreduce.cu:262-309`, `one_shot_argmax.cu:300-330` — anchor lines are ours) | NOT-USED (structural difference, not violation — but see C5) | — |
| C5 | poll-remote + read-remote as cross-rank data movement (their first design, "deadlocks or reads stale data", `TP2-SLICES.md:234-236`) | this is our shipping direction on the eager path: `one_shot_allreduce.cu:133` (`while (*peer_flag < expected_epoch)`), :191/:220 (`ld_volatile_uint4(&peer_v[i])`); `one_shot_argmax.cu:214,:219` (`while (peer_flag[t] < expected_epoch)`), :236 (`ld_volatile_payload(&peer_host_payload[t])`) | **HIT — the whole family.** Mitigating deltas from their dead shape: our remote target is host-pinned (UC on the CPU side; their stale-L2 mechanism was remote-VRAM cacheability) and our poll is bounded where theirs was unbounded. This row does NOT convict the design (their gfx9000 sibling of the probe matrix was never run here — zero-card limit); it names the CLASS as the one they measured, and the arm that reproduces their probe on our silicon is the closure | route: their probe matrix (`tools/tp2/replay_probe.cu` shapes) on gfx9000 = the missing cell; G-AMD-31 adjacent |
| C6 | componentwise volatile publish of a multi-word payload (tear class, #784) | `one_shot_allreduce.cu:24-38` (HIP lane: `st/ld_volatile_uint4` = 4×u32, comment admits "no PTX on AMDGCN"), `one_shot_argmax.cu:27-36` (HIP lane `st/ld_volatile_payload` = 4 independent field stores); CUDA lanes keep single-transaction v4 (:44-65) | **HIT — ours alone** (donor has no volatile-publish construct at all). Already under hunt (G-AMD-30 arming passed; ARTAG instrument exists); this row contributes: their tree sidestepped the class by moving payloads via copy engines/DMA, not via stores visible to a peer kernel | route: agent4 census (parity tag arm) |
| C7 | concealed/timeout spin as serve posture (S9c: spin that never sees its flag = wedge shape) | our polls ARE bounded + soft-fail by the 14:4xZ line law (`one_shot_allreduce.cu:122-170`) — better than theirs — BUT the raise path (`atomicOr` → host-mapped status :135 → `*impl_->host_status[rank]` throw :365-373) is the dead channel per b28d25fd/eb4846df | **HIT with a twist:** we hold their fix SHAPE (bounded) on a channel that cannot report — their design had no bound but reported through uncached DEVICE words where atomics work. If a poll soft-fails at boot, the token is poisoned AND the deferred throw never arms: silent, exactly the outcome the status word exists to prevent | route: agent4's ARTAG raise-path one-liner generalizes — the AR status throw (`one_shot_allreduce.cu:369-373`) rides the same dead word and needs the same volatile-RMW fix; not just `[ARTAG]` |
| C8 | atomic-CAS winner merge across blocks (their argmax uses one; `argmax.cuh:118-126`) | our `one_shot_argmax.cu` merges warp leaders via fixed-order shared-mem ladder (:139-160) + per-token single block — no cross-block CAS on token values | CLEAN (and NOTE: their CAS loop is itself order-safe only because the tie rule is total; ours avoids the question with a fixed ladder) | — |

**Census bottom line:** on their list, our tree is clean at every VALUE-ARITHMETIC construct (C1-C3)
and dirty at every TRANSPORT construct (C5-C7) — the refusals they wrote were about numerics and
memory classes; it is the memory-class half that our serving path violates by design, and our
G-CELL RED rows are forks at token selection, which is downstream of exactly C5/C6's critical path.

---

## §3 — VERDICT LINE (ranked: what OUR tree runs that THEIRS demonstrably never did)

Ranking = (delta from a deterministic baseline) × (presence on the token critical path) × (has a
donor counter-precedent that ran green). This order is what agent3's G-AMD-31 arms and agent4's
hunt should probe first→last.

1. **Counted ring re-armed per request, on the eager token path.** Six reset sites
   (`tp2_backend.cpp:1921,:2147,:3740,:5536` twins at `one_shot_argmax.cu`), `step%N` slot pairing,
   epoch words, two independent thread-local counters. **RESET-ROUTE CONFIRMED at tip (answers
   agent3's grep-confirm ask, 20:4xZ): both plain-route resets (:1921 short-prefill branch,
   :2147 long-prefill else-branch of `if (plen <= P)` :1783) sit inside `run_tp2_request`'s worker
   (:1653) — the PLAIN serve runner; the `if (mtp)` neighbours (:1909/:2135) guard only kv-mapping
   publishes. The AR counter IS re-armed every request on every plain boot; any arm built on
   'boot-monotonic rank_step' is built on the gen-not-reset defect artifact (F1), not on the
   counter.** The 128-slot ring wraps ~every TOKEN anyway (128 calls/token : kNumSlots=128), so
   the (step, epoch, slot) print arm stays load-bearing. Donor: events per-call-fresh, or
   monotonic-never-reset seq; its ONE serve-path per-request re-arm attempt wedged the box
   (`TP2-SLICES.md:275-291`) and was withdrawn un-fixed. **SERVED-PATH TRAFFIC ROW (my supersession
   to agent3, 19:1xZ — both rings ride EVERY plain-decode token, families distinct but not
   absent):** the 128-slot AR ring carries ~2 collectives/layer ≈ **128 calls/token** via the layer
   loop (`text_context_impl.h:2444` → `attn_mix_tp` :2047 / `gdn_mix_tp` :2219,:2278 / `mlp_tail`
   :2405 → `tp_group.cpp:268`, decode payload 5120 ≤ 65536 always one-shot); the 32-slot argmax
   ring carries the pick 1×/token (`tp2_backend.cpp:2498`, greedy arm — `nullptr` draft_vocab, no
   emit_conf). What plain decode does NOT ride: conf readers (`NINFER_AR_PARITY` arm
   `one_shot_argmax.cu:433`), draft remap, MTP/DFlash batched shapes (:2107,:2236,:2281).
   Consequences for the census: (i) AR KAR blindness is LIVE in the plain bin (128 auto-ACCEPTs per
   request past request 1) — AR-family staleness rows must come from counters banked per
   request-index, not from KAR verdict counts; (ii) branch-(2) isolation from the AR gen-bug stands
   via BRANCH LOGIC (identical logits ⇒ AR values matched ⇒ divergence is downstream of the AR
   ring), not via absence of AR traffic; (iii) any "0 AR mismatches" absence row must name its
   arming at the ~128/token rate per the durable rule. Probe shape for G-AMD-31: our fork is
   REQUEST-INDEXED — arm must compare request-1 tokens vs requests 2+ of the SAME boot (if the
   census shows divergence probability that jumps after the first `reset_step`, item 1 is indicted
   with it; the host_gen-not-reset bug (quickdiff F1) predicts request-2+ KAR blindness, so
   pre-declare that row or the arm cannot see it). **KAR CORPUS PARTITION (my seat, warmup-verified):
   warmup (`generation_service.cpp:513-532`, 'hi' max_tokens=4 — every collective ≤ 65536 ⇒ all
   one-shot, zero RCCL in its window) IS request 1 of the boot, so its gen gate was still counting
   from zero and DISCRIMINATING — the three warmup REJECT-candidates (rows 720/752/753) are among
   the few unblinded KAR rows on the record; served requests 2+ are auto-ACCEPT noise. Reading law:
   request-1 rows = signal, request-2+ rows = noise, until the across-boot gen stamp lands.
   **FIX-SHAPE LAW for this row (agent3 19:0xZ, verified at my seat — fold before anyone "fixes"
   it by zeroing):** zeroing `host_gen` in `reset_step` does NOT fix the KAR gate — a fresh-request
   gen=0 is indistinguishable from a never-published slot, re-creating the request-1 ambiguity for
   EVERY request. Kill shape = monotonic ACROSS-BOOT stamp: keep slot-ring/epoch arithmetic on the
   per-request `step` (`one_shot_allreduce.cu:377-378`) but pass a never-reset cumulative counter as
   `step_gen` (launch arg :404, kernel 3b gate :156-169) so the gate discriminates stale peers in
   BOTH directions; one-line-ish, no token math touched. Agent3's banked-data corroboration of the
   direction law: T3i7r1's single rank0 step=1 REJECT (gen_at_flag=0) could only fire while
   own-gen==0 — stale-high never false-rejects, it only blinds.
   PAIRING LAW (agent3, adopted 19:0xZ — the census's quotable form): within-request ordering =
   the epoch/slot pair (`expected_epoch` restarts its comparison domain at every reset,
   `one_shot_allreduce.cu:378`); cross-request staleness = the across-boot gen stamp. NEITHER
   alone closes both directions. MONOTONIC STAMP MUST BE 64-BIT (or a 63-bit-safe mask): int32
   `step_gen` wraps at 2^31 calls — unreachable tonight, live in the TP4 era where call counts are
   the point. And the direction rule for all fix-side arms: **shrink the reset surface, never grow
   it** — the monotonic-gen change DELETES reset ambiguity instead of adding zeroing work, which
   is why it is safe on this box where the donor's one full-re-arm attempt wedged a card.
2. **Token = f(peer payload read inside a kernel across PCIe).** Their token is f(local gathered
   bytes); the peer's contribution arrives by copy engine and dies in a stream, invisible to the
   argmax kernel. Ours makes the PCIe round-trip, a 4×u32 volatile publish (C6), and a peer-host
   flag poll (C5) LOAD-BEARING for every greedy token. This is the tear window (#784) that their
   architecture structurally cannot have. **Branch-(2) discriminator, refined by agent3 (adopted):**
   the MC31 print sits at tid==0 AFTER the merge, from live registers — the printed (my,peer) bits
   ARE the decision inputs, so 'identical bits, different winner' cannot be a payload-transport
   story; it is either (i) the decision rule misapplying its inputs (`one_shot_argmax.cu:258` merge
   + tie arm) or (ii) a coherently-stale whole slot = epoch pairing = the C6 generation-mix family
   (the tag includes `expected_epoch`, so a cross-generation mix re-computes a different tag; mix
   miss-prob 2^-23 per publish, masking note :186-196). ARTAG HIT alongside branch (2) → C6 window;
   ARTAG silent AND win != argmax(inputs) → decision rule, not wire. Symmetrically (agent3,
   census-design): branch-(1)/(3) value-divergence rows cannot exclude an AR-ring tear as the
   upstream carrier — the discriminator there is C5/C6 classing + per-request counters, which makes
   AR-counter arming part of the minimum boot. Arms: the AR-parity tag (already armed) + the
   single-transaction pack-comparison boot my §5-4 amendment asked for — the donor tree contributes
   the third pack: copy-engine transport as a control arm (their `pull_peer` shape works on
   gfx9000 too — our P1 facts say memcpyPeerAsync returns success driver-staged,
   `HANDOFF_2026-09-12-20Z.md:43`).
3. **Two-thread issue model over lockstep-counted state.** Donor serialized both ranks' enqueues in
   one thread (program order = publish order, deadlock-rule comment `allreduce.cu:426`); our
   rank-0/rank-1 threads make call-count parity a runtime invariant with only host-side
   `sync_bar`/watch-point comments guarding it (`tp2_backend.cpp:3729-3740` names the failure mode
   verbatim: "if the two ranks ever make a DIFFERENT number of calls… rank reads a stale peer
   payload as a token id"). Ranked third only because it is the enabling condition for #1, not an
   independent data path. Probe: per-request AR+argmax call-count equality per rank, logged (free
   from existing counters; `rank_step_now()` is already the read surface, `one_shot_argmax.h:75`).
4. **Soft-fail detection whose raise channel is dead on this silicon.** (C7). Not a fork mechanism
   itself — it's the reason a fork mechanism could run SILENTLY: every timeout/parity verdict in
   the one-shot family currently routes through the mapped atomicOr that eb4846df proved cannot
   land. This ranks above the fix-shape items because until it's raised, absence rows ("0 REJECT
   lines") for the ring are unarmed by the arming-proof law. Agent4's volatile-RMW one-liner
   generalizes to `one_shot_allreduce.cu:365-373` + `one_shot_argmax.cu:371-382` throw paths.
5. **Bounded-poll constants calibrated on an unmeasured poll cost.** Our 1<<15 bound cites
   "pinned-host flag poll ~= 2-3 us" (`one_shot_allreduce.cu:13-20`) — a prose anchor, not a
   measured cell (their 1<<26 was calibrated on device-memory probes with receipts). Under VRAM-law
   spirit the number should be measured on THIS silicon at THIS geometry; it changes no verdict
   tonight, it changes how loud a timeout is when one fires.

**What their precedent licenses (for the parity bars, per the chair's framing):** cross-config
near-ties accepted + envelope-budgeted (`TP2-SLICES.md:§4.3`, `PASS2-DESIGN.md:117-118`) — that
bar transfers to our TP1-vs-TP2 comparisons. Same-config in-process identity they enforced as hard
equality (`test_graph_tp2.cpp:233-242`) — transfers directly to G-CELL's shape. What does NOT
transfer: any inference from their green eager repeats to our ring — their eager path and ours
share the word "eager" and nothing else (per-call-fresh driver events vs persistent counted state).
**Symmetry note (agent3, adopted): AR-family findings are donor-comparable (their `allreduce.cu`
covers the same problem space); argmax-family findings are OURS ALONE — the donor has no cross-rank
argmax at all, so census branch (2) ("identical logits, different winner → comm/ring re-indict")
has NO donor precedent and, if it fires, stands on our own measurement alone.**
Their corpus is structurally blind to the per-request reset surface; G-AMD-31 is the first witness
this engine family has ever had for it. Their proof STRUCTURE — same engine object × N identical
requests × exact token-vector equality × first-differing-token report — is the template our G-CELL
green should match, and item 1's request-indexed arm is the one they could never run.

---

### Supersessions by name (predecessor rows vs my reads)

- None of the §5-4 mesh-conditioning or S3/S4 rows are contradicted by this pass. One STALENESS
  note on a board citation, not the predecessor's: the chair's 18:2xZ dispatch text said "both
  trees have these files — diff them" for `one_shot_allreduce.cu`/`one_shot_argmax.cu`; predicate
  above (lineage note) shows the donor has ZERO `one_shot*` objects on any ref. The diff is a design
  diff; §1 is written that way. The predecessor's own §5-4 amendment trail already models the
  correct reflex (re-derive at use), so this is a plan-text correction, not a lane correction.
