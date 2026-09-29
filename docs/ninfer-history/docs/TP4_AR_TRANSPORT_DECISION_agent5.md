# WO-TP4-B — AR transport decision table (world=4), zero-card, agent5

**Seat**: agent5, per WO_TP4_all_lanes.md §B + chair dispatch (intercom, 2026-09-13 ~12:25Z).
**Baselines (re-derive at use)**: read at local `amd/main @ aee8efa0`; chair-cited newer tip
`7612272a` (WO-VRAM-1 polarity clause, docs-only delta) and remote `origin/amd/main @ 081b36df`.
Source commits cited below are verified PRESENT as local objects (`git cat-file -e`) — this pass
ran ZERO fetches beyond the chair-authorized one (12:14Z, logged for the forensics registry).
**Method law**: every row carries predicate + source anchor + hardware locus + cycle class where
TPS-derived. NVIDIA-line numbers SHAPE the argument, never magnitudes on this box.

---

## 0. THE DECISION (one line for the WO)

**RCCL-first for bring-up: CONFIRMED and narrowed to bring-up-only** — it is the only world-generic
transport that exists in the tree today (`TpGroup` already constructs at any world; HIP `nccl.h`
shims to installed RCCL 2.20.5). **Steady-state: OPEN, decided by one named measured cell (§4,
B-1)** — the 4-way one-shot mesh is the ONLY option with a measured per-call cost anchor on THIS
silicon; tree/mesh/RCCL-at-4 all have zero measured anchors here; and the one-shot family carries
item-7's OPEN correctness flag until adjudicated, so the chair's correctness-before-speed ordering
is not merely prudent, it is the only state in which a transport with an unadjudicated
nondeterminism suspect becomes a perf lever. No option is refuted; the table's honest output is a
bracketed cost model + one decisive cheap cell, not a winner.

---

## 1. Measured anchors (every term, with locus and tense)

| # | term | value | predicate (exact) | source anchor | locus |
|---|---|---|---|---|---|
| A1 | AR sites / generated token | **128** | one AR per RowK tensor per token; 128 RowK of 129 Q3G64_F16S tensors; arch-derived, WORLD-INVARIANT | `docs/amd/TP4_DELTA_MAP_v1.md` §2.4 census @ local tree | model, not HW |
| A2 | hidden | **5120** | `mlp/down [5120,17408]`, `attn/out [5120,6144]`, `gdn/out [5120,6144]` all have full-width dim 5120 → AR output is full hidden width on every rank | map §2.1 (artifact rank-agnostic, merged) | artifact |
| A3 | AR payload / site / token | **10 KiB** | RowK partial = [T,5120] bf16, T=1 decode: 5120×2 = 10 240 B | derived from A1×A2 | — |
| A4 | one-shot eligibility | **T ≤ 12** | `n_elems <= kMaxElements=65536` fast-path gate @ `tp_group.cpp:268-272`; 65536/5120=12.8 | `src/core/multi_gpu/tp_group.cpp` @ local tree | CUDA+HIP share the .cpp |
| A5 | one-shot TP2-only (live) | **yes** | `TpGroup` ctor: `if (I.n == 2) I.one_shot = make_unique<...>` @ :108-112 — at world=4 the SAME code already routes every AR to RCCL with zero changes | `src/core/multi_gpu/tp_group.cpp` @ local tree | code fact |
| A6 | staged D2D bandwidth | **3.13 GiB/s** (low bound) / **6.61–6.70 GB/s** (convention: one-way payload, per peer_probe) | 10×128 MiB copies; dev0–dev2 cross-card | low: P1 window `dd819c76` @ `results/amd/p1/`; high: G-AMD-5 `99ac7d97` @ `results/v340l/peer_probe_run.log` | THIS box, V340L |
| A7 | **staged 10 KiB one-way copy** | **14.10 / 14.32 µs mean over 1000** | THE AR payload size itself, measured exactly: small-transfer latency at 10 240 B, dev0↔dev1, eager (API+sync included). RAW-RE-READ CORRECTION: the facts-table prose line says "14 us small-copy" loosely — `p2p_probe.out` verbatim says **10 KiB** | P1 `dd819c76` @ `results/amd/p1/p2p_probe.out` | THIS box |
| A7b | A6↔A7 consistency | 10 240 B ÷ 3.13 GiB/s, both legs = **6.3 µs** wire + ~8 µs fixed ≈ 14.1 measured | the P1 pair (bandwidth + small-copy) reconciles at 10 KiB: fixed/API-dominated, size-FLAT (64 B ≈ 4 KiB ≈ 10 KiB across probes) → **latency regime confirmed** | derived | convention check |
| A8 | pairwise one-shot round | **~60 µs typical transfer** | bounded-poll sizing note: 1<<15 ≈ 65–80 ms ceiling vs ~60 µs typical — a TP2 (2-payload) host-pinned KERNEL round (device-side publish+poll, no per-hop API) | merge `e9f80dda` commit text (agent3's analysis) | THIS box; prose-grade anchor, never row-measured — B-1 pins it |
| A9 | decode token time | warm **9.8 / 7.5 / 4.8**, cold **0.87** tok/s | serve-log `decode=` on gen=32 greedy rows; cycle class per chair #774 | `results/amd/p3/G17f_serve.log` (cold, 21:53Z), `G17g5_serve.log` @ `bc122675` (warm 4.8), warm spread 6.9–9.8 banked at chair | THIS box, TP2 |
| A10 | AR share, post-M2 | **~15%** at 0.485–0.678 ms/call | nsys NCCL-kernel share of prefill-dominated census runs, n=7170 calls; **payload size not named in the source rows — do not derive per-byte rates from this row** | my census rows `3cde3b9b`/`3a3ad193` (`results/a5_prefill_baseline_w0_2026-09-13.md`) | **NVIDIA 5060 Ti — NOT this box; shape input only** |
| A11 | — | (folded into A7b) | — | — | — |

## 2. Cost model, per generated token (world=4)

Model per option: `tokens_AR_ms = 128 sites × (rounds × round-latency + payload/wire-bandwidth)`.
Decode round-latency BRACKET: lower = 2×A7 staged-copy round (**28 µs**, host-API path, measured at
the exact payload); upper = A8 kernel round (**60 µs**, device-pinned, prose anchor; widens toward
~90 µs under mesh's 3-reader poll fan). Bandwidth at 10 KiB = 3.2–6.3 µs both legs (A7b) — always
INSIDE the latency term: **latency-bound**, P1's design law re-derived at world=4. Round-latency at
world=4 itself is UNMEASURED for every option — that gap, not any disagreement, is B-1's subject.

| option | rounds/rank | wire per rank (payload units, S=10 KiB) | projected ms/token | confidence | basis |
|---|---|---|---|---|---|
| **TP2 one-shot (reference, today)** | 1 | 1 round: publish S, peer reads S — BOTH ranks get the sum (local+peer combine is computed on each side) | 128×28–60 µs = **3.6–7.7** | MEASURED-class at both endpoints (A7 exact-payload lower bound; A8 kernel-path upper) | A7/A8, A9 → §3 |
| **A. pairwise 2-phase tree** | 2 | 4S (minimum for FULL allreduce: the map's 3-exchange sketch (01,23,02) leaves ranks 1,3 without the result; correct schedule is round-1 {01,23} + round-2 {02,13} — both members of each round-2 pair receive the global sum — 4 exchanges / 2 rounds) | 2×(3.6–7.7) = **7.2–15.4** | model-only: NO host-pinned kernel round has ever run at world=4 on this box (the kernel itself is pair-signature; A5) | A1×A8 extrapolated through an unbuilt kernel |
| **B. 4-way mesh one-shot (extension)** | 1 | publish S + 3 readers of each slot: host arena carries 4S/slot (4×10 KiB = 40 KiB; 128 slots/token = 5.1 MB host-resident traffic) | **3.6–12** (28–90 µs/round: same publish path, 3-deep poll fan) | anchored at pairwise at BOTH bracket endpoints; the +poll term is the unmeasured middle | A6/A7/A8; `one_shot_allreduce.h:35-65` [2]-arrays must go [4] |
| **C. RCCL ring (bring-up default; literally what world=4 runs TODAY)** | 6 steps | 1.5S | **UNKNOWN** | zero measured anchors on this silicon without P2P; every hop = 2 PCIe traversals through the root complex (G-AMD-5: canAccessPeer=0 all pairs); step-latency-dominated at S=10 KiB | A5 + G-AMD-5; A10 is the ONLY per-call NCCL-class AR datum anywhere and it is (a) other hardware, (b) 128 KiB prefill ARs |

Prefill (chunk-128, kit geometry, T=128): per-site AR = 1.31 MB, over the one-shot cap (A4) → RCCL
at any world today. Per PREFILL CHUNK-128: wire/rank = 1.5 × 128 sites × 1.31 MB = **252 MB** →
**76–157 ms per chunk** at the A6 brackets — i.e. **0.6–1.2 ms per prefill token**, vs a measured
total prefill token of 60–106 ms (A9): **prefill AR staging ≈ 1–2% of prefill time today — NOT the
prefill bottleneck at TP2.** Where it bites: at world=4 the ring volume stays ~1.5S per site while
per-rank GEMM compute QUARTERS, so AR's SHARE of prefill grows ~4× (to roughly the 4–8% band) and
AR COUNT moves from "visible" (P1's word) to first-order — fusion/batching of sites stays the perf
axis at 4, same law, bigger lever. THE BOUND WORTH STATING: even the pessimistic full-serialized
traversal reading (252 MB per chunk ÷ 3.13 GiB/s) leaves prefill under a third of its own measured
token budget at TP4 — no option above is bandwidth-strangled at 128-chunk; B-1's 1.31 MB cell is
still the one that settles the A6 convention question.

## 3. Decode share, honestly bracketed (cycle-class column per new law)

| geometry | ms/token AR | decode token budget (cycle class) | share |
|---|---|---|---|
| TP2 one-shot today | 3.6–7.7 (A7/A8 bracket × A1) | 102 ms — warm, 9.8 tok/s (17g-class fast boot) | **3.5–7.5%** |
| TP2 one-shot today | 3.6–7.7 | 133 ms — warm, 7.5 tok/s (g5 throughput-window line) | **2.7–5.8%** |
| TP2 one-shot today | 3.6–7.7 | 208 ms — warm-but-slow, 4.8 tok/s (g5 request line) | **1.7–3.7%** |
| TP2 one-shot today | 3.6–7.7 | 1149 ms — COLD, 0.87 tok/s (17f boot) | **0.3–0.7%** |
| mesh@4 | 3.6–12 | same rows | 3.5–11.8% (warm-fast) |
| tree@4 | 7.2–15.4 | same rows | 7.1–15.1% (warm-fast) |
| RCCL@4 | ? | ? | **the §4 B-1 row decides this cell** |

Cross-check vs A10 (NVIDIA): the census rows do not name the payload of the n=7170 NCCL AR calls,
so NO per-byte rate is derivable from A10 and none is claimed here — only its share (~15%) and
class (NCCL-mediated, prefill-dominant run). Direction only: NCCL-class ARs on the NVIDIA pair
landed at ~0.5-0.7 ms/call while this box's pairwise pinned round is ~60 µs (A8); magnitude is
NOT transferable across silicon, and A10's regime (NCCL, large-token census) differs from A8's
(one-shot kernel, small payload) as well. Nobody may budget TP4 AR off A10.

## 4. Named cells for the chair to slot (the table's real deliverable)

- **B-1 (rides the first 4-card window, ~2 min, same binary discipline as kit):** RCCL
  `ncclAllReduce` per-call latency sweep at S = 10 KiB / 128 KiB / 1.31 MB, world ∈ {2, 4}, eager
  (graphs DEAD per P1), 200 timed reps + median/p5 reported as its own truth-stamp row. This is the
  missing A-column entry: no RCCL AR has ever been timed on this box, and it is the bring-up
  default the whole TP4 first-light run will sit on. Pre-declared reading (against the §2 bracket
  28–60 µs): if median@10KiB(w=4) ≤ **2× the bracket top (≤120 µs)**, one-shot extension is NOT
  worth a perf WO at bring-up shapes — RCCL carries TP4; if ≥ **4× the bracket top (≥240 µs)**,
  extend-the-mesh (option B) is the lever and gets its own WO; anything between = report and hold.
  Tree (option A) never wins on latency at this payload (4 exchanges vs 1 round) and is retired by
  this row's math unless B-1 surprises.
- **B-2 (zero-card, CPU, rides WO-TP4-A):** a compile-only 4-way mesh skeleton — `Slot::host_buf[4]`
  + epoch arrays generalized — proving the storage generalization is additive (no CUDA-arm diff in
  `#else`), so if B-1 names mesh as the lever, the WO is days not weeks. No device time.
- **B-3 (rides any serve row, free):** the `--request-log-jsonl` field agent3 named (#775-②) — per
  AR timing is otherwise unattributable (decode-path `[gating]` lines are untimestamped, agent2's
  finding), and item-7's adjudication needs the same field. One flag, serves B-1's interpretation.

## 5. Known conflicts kept visible (not smoothed over)

1. **A6 discrepancy (3.13 GiB/s vs 6.61–6.70 GB/s)** — per the P1 ruling: *nobody budgets AR off
   either until explained*. This table brackets with BOTH everywhere; no conclusion flips inside the
   bracket (decode is latency-bound under either — A7/A7b: the fixed term dominates at 10 KiB both
   ways; prefill AR is 1–2% at the slow convention and less at the fast one). The two numbers
   differ in DIRECTION-CONVENTION (one-way vs both-legs), MESSAGE SIZE (256 MiB vs 128 MiB), and
   PAIR (dev0↔dev1 same-card-ish leg vs dev0↔dev2 cross-card) — any of which could carry the 2.1×;
   A7b proves the P1 pair is at least SELF-consistent. Proposed closure (cheap, rides B-1's
   harness): one instrument, both conventions, same buffers, both pair classes.
2. **A7 (14 µs eager staged copy @ 10 KiB) vs G-AMD-5 (97–101 µs "RTT", 64 B/4 KiB)** — different
   paths (single eager copy each way vs API+sync round-trip PAIR per iteration); the G-AMD-5 figure
   is ~2 hops × ~50 µs with sync overhead — consistent with 14 µs data-path + ~35 µs sync/API per
   hop. Cited separately, never summed; B-1's harness prints both forms so the next reader gets one
   instrument.
3. **Map's 3-exchange tree sketch** is short by one exchange for full allreduce semantics (§2 row
   A) — flagged to the map owner as an amendment, not silently fixed (WO-TP4-C law).
4. **One-shot family is under item-7 suspicion** (agent3's #784: HIP 4×4-byte volatile publish vs
   CUDA single-transaction `st.global.wt.v4`). Until adjudicated, extending one-shot = extending an
   open nondeterminism surface; B-1's threshold reading above is therefore conditioned: **mesh
   extension may only be WO'd after item 7 closes or indicts-and-fixes the publish path.** The
   chair's RCCL-first prior stands unchallenged for bring-up on correctness grounds ALONE,
   independent of cost.
   **AMENDMENT 2026-09-13 (agent5 re-derivation on chair request, after verdict row `1d0ff3c6`):**
   item 7 ADJUDICATED per-request-live with cause = upstream state carry-in; the one-shot window is
   EXONERATED for 4/5 forks (only request 4 carries REJECT events) and the S1 monotonic-epoch proof
   stands unfalsified (327/327 observed==expected, zero soft-fails). Effect on THIS table: **none on
   B-1** — the thresholds, bracket, and geometry are cause-neutral, and RCCL-vs-one-shot economics
   never depended on the fork's mechanism. Effect on the mesh-WO CONDITION: relaxes from "after item
   7 closes" to "after the request-4 residual is disposed by agent4's hunt" (one fork with an
   AR-window REJECT at decode steps 0–1 stays OPEN; the exoneration is 4/5, not 5/5, and a publish
   extension that turns out to carry even a rare tear inherits item 7's whole cost). Trigger order
   unchanged: B-1's number first (window), residual's disposition second, WO drafting third.
   **AMENDMENT 2 (2026-09-13T15:3xZ, trail-named per chair re-derivation notice; ground moved by
   agent3's SELF-correction `232d1797`, merged `4f1f8bd7`, re-verified at my seat: REJECT rows
   720/752/753 precede `listening` 762 — line math closes):** the three KAR REJECT-candidates are
   WARMUP-phase, not request 4 (agent3's own words: 'I inferred a phase from a counter's NAME') ⇒
   **5/5 served forks now have clean served-window traces** — zero served-request evidence against
   the publish path. My amendment-1's "request-4 residual" placement is hereby SUPERSEDED (its
   decode-steps-0–1 detail was the same line-position error, inherited by trust from the row my
   amendment cited — re-derive-at-use cuts both ways, including into my own table). Residue now:
   (i) a measured WARMUP rank-step asymmetry (rows 752/753: rank1 transiently issued more
   allreduce_bf16 calls than rank0; unadjudicated — the warmup exception-swallow path, named by
   agent3 at generation_service.cpp:513-531, is a live alternative explanation not yet settled);
   (ii) a named INSTRUMENT GAP: A1TRACE-K is epoch-compare only and structurally CANNOT see a
   4-store payload tear — #784's mechanism survives on that BLINDNESS; it was never tested and
   beaten, only never seen. Mesh-conditioning under this: **RELAXES FURTHER, STILL NOT LIFTED** —
   no served-window evidence remains, but extending a publish path whose sole surviving suspicion
   is defined by the instrument that cannot see it, with an unexplained same-family warmup asymmetry
   on the record, is exactly the rare-suspected-tear case my rule covers. Disposal paths, both
   cheap, neither mine to schedule: adjudicate the warmup asymmetry zero-card (throw-path vs
   swallow-trace placement), and give the tear hypothesis a capable witness (single-transaction pack
   comparison boot — B-1-harness adjacent, one artifact, rides any future stamp). B-1's SLOTTING IS
   UNCHANGED — it fires in the 4-card window as board law regardless; thresholds remain
   cause-neutral economics.
   **WRAP CAVEAT (agent4's pass-1 self-correction, `CARRYIN_HUNT_agent4_pass1.md` §3; added per
   chair 15:37Z notice):** every step↔phase placement cited in this §5-4 (rows 720/752/753 vs
   listening 762; "rank1 AR-step 100/101") reconstructs KAR telemetry from calls far beyond one
   ring revolution: the ARMAX payload ring is 32 slots (`one_shot_argmax.h:14 kNumSlots=32`) and a
   6-run boot makes ~164 calls ⇒ ~5 wraps, slot-indexed retro reads can address RECYCLED slots.
   Bounded exactly as agent4 bounded it: READ-SIDE TELEMETRY hazard only — out_token is written
   in-kernel per call, so the token path cannot be stale-slot-poisoned and NO verdict flips on this
   line. But it degrades the evidential grade of every placement claim above from line-position
   FACT to wrap-caveated INFERENCE, and any future re-use of this corpus's step↔request mappings
   must carry the caveat. The llREDUCE ring is 128 slots (`one_shot_allreduce.h:14` — my B-2
   skeleton's kB2NumSlots) and my table's cost rows are SLOT-INDEPENDENT (they count calls, never
   read slots), so no geometry/economics row in this document is wrap-exposed; the exposure is
   confined to the §5-4 placement archaeology quoted from agent3's row.

## 6. What this table does NOT say

No option is measured-dead. No number here is budgetable as a promise (VRAM-LAW spirit: models
report, allocators and launches decide). A10 must never be re-cited as this box's AR cost. The
120k-class context goal (§B.2) is NOT transport-limited at any option above — decode AR cost is
≤15% of even the fastest measured token at world=4 modeling; the context ceiling lives in VRAM and
KV math (agent4's WO-VRAM-1/E-series seat), and this seat hands that half of the question there.
