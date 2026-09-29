# GFX906 QUICKDIFF (agent5) — 40-min re-scope, chair orders 18:4xZ 2026-09-13

**Refs**: CONTROL = `/tmp/ninfer-gfx906` @ `7a3c18d9` (read-only, gfx906-port). OURS = worktree
`/home/chris/worktrees/amd-wo-gfx906-diff`, branch `amd/wo-gfx906-diff` @ base `785bca6f`.
Zero-card, file reads + git history only. The full three-section differential doc follows tomorrow;
THIS file answers Q1–Q4, every row predicate + file:line@ref.

**Lineage note (predicate: `git log --all -- '*one_shot*'` in control = 0 commits, all history):**
the donor tree has NO `src/core/multi_gpu/one_shot*` family at all — not deleted, never existed.
Their TP2 transport lives in `src/ops/common/allreduce.cu` (event transport + a capture-only
flag-sync variant) and it is architecturally a DIFFERENT design, so "diff the files" resolves to
a design diff, not a text diff. Our one-shot ring is our own lineage (invented at `e5d798cd`,
2026-08-20, on our side of the fork).

---

## Q1 — THEIR ARGMAX AT BOOT: there is no cross-rank argmax. The pick is LOCAL on gathered logits.

Donor T=1 token path: lm_head is COLUMN-parallel — each rank computes its half-vocab logits
(`part[r] = [V/2, 1]`, control `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1915-1917`),
then `logits_tp2` (`:1891`) runs `ops::allgather_rows` ONE COLUMN AT A TIME (`:1919-1929`) so that
**both ranks end up holding the FULL [V,1] logits tensor in LOCAL device memory**; the pick is then
`ops::argmax(whole[0], ...)` on rank 0's stream (`:2503`, and the same shape at `:1274`, `:824`) —
an ordinary single-device kernel whose tie rule is `value > best || (value == best && index < best)`
(control `src/ops/kernel/argmax.cuh:24`), deterministic per block + a CAS-merge loop (`:118-126`).
The MTP/draft proposal picks do the same (`:2485-2504`). The ONLY cross-rank collectives in their
decode are `allreduce_sum` (2/layer, 128/token) and `allgather_rows` (the logits/vocab gathers), and
on the EAGER path — which is what their byte-identical serve/md5 records ran post-revert — every
one of those is host-API choreography: `cudaEventRecord` → `cudaStreamWaitEvent` →
`cudaMemcpyAsync` D2D → local combine kernel (control `src/ops/common/allreduce.cu:444-470`,
`:558-589`; "There is no host synchronization anywhere inside the layer loop", `text_context_impl.h:1455`).

**Where their cross-rank pick is, and what it does at the moment ours forks:** ours forks inside
`one_shot_tp_argmax_kernel` / `one_shot_ar_pinned_vec_kernel` — an in-kernel PCIe handshake:
publish payload to OWN pinned slot → `*my_gen`/`*my_flag` stores → **poll the PEER's host flag
across PCIe** → read the PEER's payload (ours `src/core/multi_gpu/one_shot_allreduce.cu:96,:117-133,:191`;
`one_shot_argmax.cu:199-231`). At that moment the donor has NOTHING in flight device-side: the
cross-rank ordering is carried by driver events re-recorded fresh per call — zero counters, zero
ring slots, zero epoch words, nothing to be stale, nothing to reset, and the payload words are
moved by a DMA copy engine, not by componentwise 32-bit volatile stores that can in principle tear.
**Second asymmetry (this is the one to brief agent4 on): the donor's engine issues BOTH ranks from
ONE host thread** — `for_each_rank` sets the current device and enqueues rank 0 then rank 1
back-to-back (control `text_context_impl.h:1458` + the deadlock rule comment at
`src/ops/common/allreduce.cu:426`), so publish ORDERING is serialized by program order.
OUR engine runs the two ranks on **two host threads** (`src/runtime/tp2/tp2_backend.cpp:3270-3280`,
`:4331`, `:7368` — `worker(1,...)` on `std::thread`, `worker(0,...)` on the caller), and the
one-shot step counters (`rank_step[rank]++`, ours `one_shot_allreduce.cu:376`,
`one_shot_argmax.cu:386`) increment independently per core. Same code lineage; the donor removed
the whole inter-thread timing surface from the cross-rank protocol, and we never had that removal.

Also: donor gates their in-kernel (flag-sync) transport to **capture-only** — `capturing(ec)` check
at control `allreduce.cu:424,:517`, policy commit `7c02856c` ("prefill and eager decode KEEP the
event transport... eager decode keeps its byte-identical S7 path"). Our `tp_group.cpp:268` routes
EVERY small AR to the pinned-ring handshake with no capture gate and no env off-switch. **Answer to
"do we carry their fixes or their pre-revert state": NEITHER — we run their rejected shape on the
eager path they deliberately protected, with a per-request reset protocol they never built.**

---

## Q2 — NEAR-TIE HANDLING: deterministic tie-break in-kernel; cross-config near-ties ACCEPTED-and-documented; same-config repeats enforced bit-exact.

Three distinct rules, each with a donor anchor:

1. **In-kernel tie rule**: argmax equality breaks to LOWEST index — control
   `src/ops/kernel/argmax.cuh:22-24` (`argmax_better`), used by both the single-block and the
   tiled-CAS kernels. (Ours carries the same rule — `one_shot_argmax.cu:108,:258` — so ties are
   NOT a differential between the trees; what differs is only WHERE the max is taken, Q1.)
2. **Cross-config near-ties = accepted, documented, budgeted.** When their tp1-vs-tp2 greedy test
   diverged at token 12/32, the verdict recorded was attribution + acceptance: "a synthetic-prompt
   + Int8Group64-KV split divergence, the same verdict the S6 note anticipated" (control
   `docs/gfx906/TP2-SLICES.md:256-257`), and the pass-2 design says outright: one near-tie greedy
   divergence "expected and acceptable per the stage-6 rule" (`docs/gfx906/PASS2-DESIGN.md:117-118`).
   The parity bar is explicitly COMPARATIVE, not byte-exact: "tp2 inside tp1's own prefill-chunk
   perturbation envelope on argmax / KL / (1-cos)" (`TP2-SLICES.md:§4.3`, bar `KL 0.0157 vs 0.0714`).
3. **Same-config repeats = enforced bit-exact, as a TEST, not prose.** `tests/targets/qwen3_6_27b/test_graph_tp2.cpp`
   is their G-CELL: same engine object, same prompt, `generate_greedy` TWICE (`:217-218` — two
   REQUESTS in one process), compared with exact vector equality, plus eager-vs-graphs exact
   equality (`:238-242`), failing with "first differ at token N" (`:233`). Their md5s (p3
   `52960e3a…`) repeated across boots/transports are the serve-level instance of the same rule.

**Candidate closure for our G-CELL bar (chair asked):** the donor's structure is exactly what the
plan converged on — same-config identity is a HARD equality assertion run inside one process,
cross-config parity is an envelope with documented knife-edges. Their precedent LICENSES the
near-tie tolerance band only for cross-config comparisons; it never uses tolerance for a
same-config repeat. No prompt-corpus tie-avoidance anywhere — they hit ties head-on (token-12
divergence) and answered it with the two-tier bar, not with corpus design.

---

## Q3 — ARMS AUDIT (for agent4, read THIS first): the donor has NO per-request protocol-state reset — there is no arm Z in their lineage, and their ring-hazard answer is an END BARRIER, not depth.

Grep census on control (`memset|reset|zero|carry` over `TP2-SLICES.md`, `STAGE*-LOG.md`,
`SERVE-DEBUG-LOG.md` + the runtime sources): the only per-request resets in their serve path are
the WORKSPACE ARENA bump-pointer (`work_.reset()`, text_context_impl.h:747,:1979,:2152… — allocator
bookkeeping, no protocol state) and engine submission structs. **Their cross-rank transport has no
per-request state to reset, BY DESIGN:**

- Event transport (their eager = our corpus's mode): per-call ordering state is a driver event,
  re-`cudaEventRecord`ed every call (`allreduce.cu:444-470`) — freshness by construction, no epoch,
  no step, no flags, no ring.
- Flag-sync transport (capture-only, opt-in at HEAD): per-block sequence in DEVICE uncached memory,
  **monotonic, NEVER reset** (`self_sg->seq[b] + 1`, `allreduce.cu:196,:232`); no per-request
  boundary touches it.

**Finding 1 — arm Z (full state-span re-zero at request boundary) has no donor precedent, and the
donor's ONE experience with per-request re-arm of an in-kernel protocol is the card-2 WEDGE.**
Their S9c revert text names the failure candidate as exactly a per-request state event: "graph
re-instantiation on full_reset with a STALE FLAG EPOCH" (`TP2-SLICES.md:283-284`) — request 12,
`reuse=full_reset`, GPU hang, box reboot — and their disposition was NOT "fix the reset" but
"withdraw the transport from default until root-caused" (`7a3c18d9`). If agent4's Z arm makes the
reset bigger (full-span re-zero), the donor record says: every device-visible protocol word that
must be re-armed at a request boundary is itself the wedge surface; the safe end-state of this
comparison is "fewer persistent words, or none", not "more thorough zeroing". (Our reset shape,
for contrast: `one_shot_allreduce.cu:336-345` zeroes epoch + own flags but **NOT `host_gen`** —
gen keeps counting across requests while `rank_step` restarts, so the KAR gen-stamp gate
(`one_shot_allreduce.cu:143-170`) auto-ACCEPTs for every call whose new-request `step_gen` is below
the old gen. That is an instrument-integrity finding for agent3's G-AMD-31 reading: REJECT/ACCEPT
counts from request 2+ of a boot are NOT comparable to request 1's. It does not poison tokens —
the gate only observes — but do not read post-request-1 KAR rows as proof of anything.)
**F1-addendum (agent3, verified at my seat, 19:0xZ):** the naive fix — zeroing `host_gen` in
`reset_step` — is WRONG: a fresh-request gen=0 is indistinguishable from a never-published slot,
re-creating the request-1 ambiguity for EVERY request. Kill-fix shape = **monotonic across-boot
stamp**: keep the slot-ring/epoch arithmetic on per-request `step` (:377-378) but pass a never-reset
cumulative counter as `step_gen` (launch arg :404, kernel gate :156-169) so the 3b gate
discriminates stale peers in BOTH directions. One-line-ish, no token math touched. Agent3 also
corroborated the direction law from banked data: T3i7r1's single rank0 step=1 REJECT (gen_at_flag=0)
could only fire while own-gen==0 — stale-high can never false-reject, it only blinds. Pairing law
(adopted into the differential doc): within-request ordering = epoch/slot pair (:378, comparison
domain restarts per reset), cross-request staleness = the across-boot gen — neither alone closes
both directions; the monotonic stamp MUST be 64-bit (int32 `step_gen` wraps at 2^31 calls —
unreachable tonight, live at TP4 call counts).

**Finding 2 — arm R (ring depth/slot logic): the donor solved the slot-recycle hazard with an end
barrier, not depth.** Their flag-sync staging is ONE fixed 16 MB buffer reused every call, made
safe by a SECOND barrier protecting it from the next call (`TP2-SLICES.md:229-230`; `end[b]`
flag barriers at `allreduce.cu:205-212,:241-248`; end[b] writes at `:210,:246`): the kernel does not exit until the peer has
finished reading. Our rings (128 slots AR / 32 argmax, `step % kNumSlots`) have NO destroy gate:
a rank publishes (overwrites slot `step%D`) BEFORE waiting for the peer. I checked the recycle race
by unrolling our publish/poll chain: for our even depths the chain A@S+D ⇒ … ⇒ B@S-complete closes
with slack at D=32 and D=128 **given both ranks make the same number of calls per request** — depth
alone is load-bearing on call-count parity, which the :3729-3740 "RULE-8 WATCH-POINT" comment
already names as the known-fragile invariant (their own ab3b P10 datum: a counter skew turns the
ring into cross-generation reads, timeout soft-fails, and a rank-asymmetric silent exit). The donor
architecture cannot skew that way: there is no counter. If agent4's R arm can carry an arm that
OBSERVES the destroy window (post-combine "done" flag polled by the next same-slot publisher, or a
cheap slot-generation parity like the argmax one extended to the AR ring), the donor precedent says
that is the stronger fix shape than any depth increase — depth is a probabilistic guard on call-count
parity; the end barrier is a structural one.

**Finding 3 — their memory-class refusals map onto our exact suspects (from the probe matrix,
`TP2-SLICES.md:231-236`):** fine-grained signal blocks DEADLOCK (local poll served from L2);
plain-cudaMalloc staging returns STALE PARTIALS ("DIFFER at 5 rounds"); their FIRST design —
"poll the peer's flag via UVA, peer-READ data" — deadlocked or read stale and was ABANDONED because
remote VRAM is mapped cacheable and gfx906 has no L2 writeback instruction. Our shipping design is
structurally the poll-remote + read-remote shape (across PCIe to host-pinned instead of to remote
VRAM — the memory is different, the DIRECTION is the one they measured dead; gfx9000/gfx906 Vega
share the weak cross-link writeback story, and per our own P1 fact the pinned-host volatile stores
are componentwise u32, `one_shot_allreduce.cu:24-38` — the #784 tear question lives HERE and has no
donor precedent either way). Their fix direction was always: PUSH the data, keep the poll LOCAL,
never read remote. That's a third arm shape for the census bin: publish-to-peer-staging + poll-own-flag,
and it is the only handshake variant this engine family ever ran green for 20 000 replays
(`TP2-SLICES.md:236-238`, with the caveat that its serve-path debut was the wedge).

---

## Q4 — DOES THEIR DETERMINISM PROOF DEPEND ON GRAPH/CAPTURE-ADJACENT MACHINERY OUR EAGER PATH LACKS? No — it depends on machinery our eager path LACKS BY OMITTING IT: the proof era ran the transport with zero in-kernel cross-rank state.

- The "reproducible across two graph runs" quote is `test_graph_tp2.cpp` leg 2 (see Q2.3) — that leg
  IS capture-adjacent, BUT its in-process repeat safety rides the EVENT transport's structure
  (events replayed inside a graph are driver-managed semaphores; no counter appears anywhere in the
  captured stream). Under their HEAD default (post-`7a3c18d9`) serve is EAGER with the event
  transport: per-call ordering lives in the host API call sequence issued by ONE thread, which is
  deterministic because it is just program order. **Nothing capture-specific is load-bearing for
  their eager-repeat identity — the load-bearing property is "no device-visible protocol state
  persists across calls", which our eager path does NOT have** (ours carries rank_step, epoch,
  128+32 pinned flags, gens, and per-request resets across every call).
- Their slice-9 saga is the CONTROL of this experiment in reverse: they had capture-only in-kernel
  sync, they flipped it to DEFAULT (`de210ae4`), and the first serve-path per-request re-arm
  (`full_reset`, request 12) wedged a card — and their conclusion was "the CLI gates never exercised
  the serve path's sampling + full_reset re-arm sequence" (`TP2-SLICES.md:283-285`). We have been
  running the equivalent riskier design (in-kernel handshake ON eager + per-request resets) as our
  steady default all week.
- **Does their eager-repeat evidence isolate in-process per-request state the way our census will?**
  Only negatively: their in-process two-request test (`captured` vs `replayed`, two `generate` calls
  on one engine) and their cross-boot md5 repeats PROVE identity on a transport with no
  per-request-reset surface, so they cannot localize a reset-surface defect — they never had one to
  localize. Our G-AMD-31 census asks a question their corpus is structurally unable to answer;
  what transfers is the PROOF STRUCTURE (one engine ×N identical requests ×exact token equality +
  eager/graphs cross-check + "first differing token" reporting) and its bar (hard equality
  same-config), plus the negative datum that BOTH trees' in-kernel sync designs have bitten
  per-request-boundary bugs at least once — theirs fatally enough to revert, ours mysteriously
  enough to fork tokens.

**HEADLINE for the closeout doc (chair's requested sentence):** their eager era never had our
in-process fork surface — its per-request state was a driver event re-recorded per call and its
token pick was a LOCAL argmax over gathered bytes — so the delta is real, it is ours (the pinned
publish-poll ring + the two-rank-thread issue model + the per-request counter reset), and its
boundary is now drawn.

---

## Ranked verdict preview for agent3/agent4 (what OUR tree runs that THEIRS demonstrably never did)

1. **In-kernel cross-rank handshake on the EAGER serving path** (publish-poll-combine in one
   kernel, peer-poll across PCIe): donor eager = host-API event choreography only (`allreduce.cu:444-470`).
2. **Per-request reset of device-visible protocol counters** (epoch/flags/steps): donor = monotonic
   or per-call-fresh, never re-armed; their one re-arm attempt = card wedge (`TP2-SLICES.md:283-284`).
3. **Independent per-rank call streams (two host threads) driving a lock-step-counted ring**:
   donor = single issuing thread, program-order serialized (`text_context_impl.h:1458`).
4. **Cross-rank argmax inside a kernel** (token = f(peer payload read over PCIe)): donor token =
   local argmax over DMA-gathered local logits (`text_context_impl.h:1891,:2503`).
5. **Componentwise 32-bit volatile publish of multi-word payloads with depth-only recycle protection**
   (no end/destroy barrier): donor staging protected by explicit second barrier
   (`TP2-SLICES.md:229-230`); our `host_gen` not re-zeroed at reset (instrument-integrity note, Q3 F1).

Items 1–3 are the arms the census should discriminate; 4–5 name the two fix shapes the donor
lineage actually validated.
