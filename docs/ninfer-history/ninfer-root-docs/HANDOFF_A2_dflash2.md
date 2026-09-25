# HANDOFF — A2 DFlash2 lane (wo/dflash2-scope)

> **IF YOU ARE RESUMING NOW, READ `HANDOFF_A2_dflash2_session3.md` FIRST.** It is the
> current entry point: measured iteration costs, the six fork points for 2b-iv(b) with
> verified line numbers, and the two places that will actually bite. Everything below is
> still the standing state, but §0.1's "NEXT STEP" and §5's quicklist have both moved
> since (the quicklist is now **10** suites by name, and the tap probe takes
> `--sidecar` and has a `--mode negatives` arm that needs no grant).

**Read this first if you are the next A2 session.** Written 2026-09-05 ~19:45Z,
session 01a0713f (agent2); refreshed at the 01a073a6 debrief (2026-09-06 ~00:0xZ)
and again by session 01a07435 (2026-09-06 ~05:0xZ) — **read §0.1 FIRST**, it is that
session's delta and it changes several statements below (blockers C/D/E resolved, F
escalated, [I5] CLOSED, two new kernels landed as skeletons, and the quicklist rule
changed). Original provenance follows.
session 01a0713f wrote it after (d.3) commit 2a landed and was device-verified (§22.13) and the oracle's
stage B landed (§22.21). Every SHA, count and encoding claim below was RE-VERIFIED
against the tree at that time — §5 carries the commands, because a handoff's own
numbers go stale faster than its prose (gotcha #11). Deep record: `docs/151_dflash2_scope.md` on THIS
branch (main carries it only through §19 — §20-§23 are branch-only until
merge). This file is the quick-start; docs/151 is the law. NOTE: AGENTS.md's
handoff pointer targets the 2a lane — THIS file is the DFlash2 lane entry.
All "S"-numbers below are docs/151 sections.

## 0.1 SESSION 01a07435 DELTA (read this before trusting anything below)

Tip **6e426dde** = merge of main@cfca00ca, tree clean, **42 commits** this session.
Quicklist **9/9 by NAME**, oracle self-checks **34/34**, sidecar 33 assertions / 16
refusal checks, disk ~21G.

**THE PHASE C HOLD IS LIFTED — main is cfca00ca and it is FOLDED (6e426dde).**
Three things a resuming session must know about that fold, because they change what
you expect to be hard:

1. **It merged with ZERO conflicts.** The flagged one (tp_engine `deliver()` vs
   `can_batch`) did not materialize — Phase C did not touch the `deliver()` arm, and
   §22.16's "every change is gated on the DFlash2 backend" argument is what bought
   that. Verified after the merge, not assumed: the batch-of-one refusal is intact at
   `tp_engine.cpp:338`, the 21 `dflash2` sites in `tp2_backend.cpp` are intact, a
   repo-wide marker scan is empty, and `ninfer_engine` + the dflash2 test targets
   rebuild rc=0. **So do not re-derive fold-prep anxiety for the next merge — but do
   still run the marker scan and the site checks, because they are what turned
   "no conflicts" from luck into a verified state.**
2. **GPU: grant GIVEN (both cards, session 01a07435) but NOT CLAIMABLE** — A1 keeps
   them for the (vi) write-time-hashing hunt (coordinator KEEP ruling, and A1
   retracted his 06:03Z release at 06:06Z). Clear-guard != released, and today proved
   the stronger form of that: a clear guard can just mean the other holder is between
   runs. Wait for an explicit release that post-dates the keep-ruling.
3. **A1's (vi) fix lands in the prefill/MTP-forward region of `tp2_backend.cpp`**
   (same neighborhood as make_rank's MTP terms) — he flagged it, and the answer is
   that the blessed new-file shape makes our overlap **one line**, not a region, so
   his fix does not need to wait on this lane. Recorded because the next session
   should not re-derive that from scratch or, worse, put 2b-iii inline and create the
   collision the new file exists to avoid.
4. **§22.39's shape is BLESSED by the coordinator**: 2b-iii goes in a NEW file
   `src/runtime/tp2/dflash2_round.cpp` with ONE call from `tp2_backend.cpp`, not
   inline. That is the next piece of work, and it is now unblocked by the fold.
   **Cards: NOT CLAIMABLE. A1 released at 06:03Z and RETRACTED at 06:06Z — the
   coordinator ruled KEEP because (vi)'s write-time hashing re-runs the model on
   device. Both cards stay with A1 until he confirms release HERE, in a message that
   post-dates the keep-ruling. Do not claim on a guard reading alone: the guard was
   clear at 06:0xZ (0 compute apps, 15 MiB / 0%) precisely because A1 was between
   runs, which is the gap a release/retraction crosses. This lane's own grant stands
   — it is the RELEASE that is pending, not the permission.**

**NEXT STEP, concretely:** write 2b-iii against §22.39's new-file shape — the
per-block op table is in §22.30, the attention/top-k/assemble kernels exist
(SKELETONs), the [I8] transpose and BF16 guard are landed in the binder, and
§22.31 items 1-3 are the gates that turn those skeletons into claims. Blocker G's
first-cut (codebooks pinned to w8g32_f16s via `--set`) is the coordinator's read and
is a manifest-flag decision, not a code one.

**If you read only one thing from this session: §22.31 is the device-window
checklist** — seven gates, each with its command and its PAIRED mutation, including
one mutation that must stay GREEN.

**GATE STATUS AS OF 09:4xZ (312003a2): ALL THREE GATES GREEN.** The device test passes end to end for the first time (rc=0, ALL PASS), the 9 CPU suites pass, `ninfer_engine` rebuilds.
**The resolution, because it inverts the whole investigation: the bug was in the
FP64 ORACLE, not the kernel.** `block_attention` pushed scores only for admitted keys
but reconstructed each key's pool row as `hlo + (i - T)` from the index into that
COMPACTED list — so every excluded key shifted the score-to-V-row pairing by one. It
explains every prior observation: lane 1 excludes nothing so its reconstruction was
always right (2.4e-04); the control and the V=1 identity were green because neither
depends on the pairing; and the three earlier fingerprints (count, position-sum,
denominator) all matched because **none of them involves V at all**. The numerator was
the first aggregate that did, and it was the first to disagree.

Read §22.45 for the method point, which is the durable part: a reference that agrees
with the kernel on every axis except the one nobody fingerprinted is not an
independent check on that axis. The fix is not more agreement tests — it is an
aggregate that depends on exactly the thing in question.
- the kernel's softmax, value path, head mapping and per-lane indexing are PROVEN
  correct by a permanent CONTROL cell (nothing excluded -> 1.951e-03, green)
- the admitted-key COUNT per column is PROVEN identical between kernel and oracle
  (the counter is `heads_q x` the oracle's, exactly 32.0 on all twelve columns)
- so the remaining question is: same-size sets, different output. Either the sets
  differ in CONTENT, or the sets match and the VALUES differ on the keys the
  exclusion path touches.
The plumbing for the next probe already exists: `launch_block_attention` takes an
optional `admit_counts` (nullptr on the production path). Swap the count for a sorted
set, or dump one column's per-key scores.

**One trap recorded so you don't fall into it:** the 32x ratio looked like a kernel
bug and was the counter being indexed per column while 32 heads share a column.
Before reporting a surprising ratio, ask what else multiplies by that number.

What changed vs the rest of this file:
1. **[I5] IS CLOSED (§22.24), and the ORACLE was wrong, not the engine.** ops::rope
   is SPLIT-HALF pairs `(i, i+rot/2)` over `rot = 128` = the whole head, with
   `inv[i] = 1e7^(-2i/128)` — read off rope.cuh's own index arithmetic, pinned
   bit-exactly against all 64 entries of `kDflashRopeInvFrequency`, precedented by
   DFlash v1 passing `rotary_dim = head_dim = 128` at the same geometry.
   `dimension_sections [64,0,0,0]` counts PAIRS, not dims. Landed: `rope_apply`
   fixed, `extract_config.py` derives `rope_rotary_dim = 2*sum(sections)`,
   `DFlash2Config::rope_rotary_dim = 128` + 4 static_asserts (one pins
   `!= TextConfig::rotary_dim`, the target's 64 is for a 256-wide head). The
   config.h comment claiming the sections "matter only for image spans" — the
   sentence that seeded the misreading — is corrected. STILL OWED: the device
   one-hot through the real kernel (a formality), and "the q/k rows need no
   llama.cpp-style permute" rests on the DFlash v1 precedent = inference, not
   measurement.
2. **BLOCKER C RESOLVED by ruling (§22.25 + addendum): the drafter gets its OWN
   attention kernel.** `ops::swa` is DFlash v1's attention with DFlash v1's 4096
   window baked into a shared op (swa.cpp:19/:38 + the literal 4095 in
   bidirectional_gqa_attention.cuh:373-382); DFlash2's pool is 2048, so swa refuses
   it. A1 adopted (b) with five conditions and rejected (a) because swa's capacity
   check is a CONTRACT with its one caller, not a defect. **swa stays untouched.**
   Landed as a SKELETON: `src/ops/dflash2/dflash2_attention.cu/.cuh` — compile-
   verified, **Test 7 (the FP64 gate) is written but NOT RUN**.
3. **[I9]'s "FP32 staging streams" is RETIRED as unbuildable** (A1 + coordinator,
   reversing the 23:58Z adoption): linear/rmsnorm/rope/swa all reject non-BF16
   activations and `cast.h:31` is one-directional. `launch_conv`/`launch_edge_scores`
   are now **BF16 operands with FP32 accumulation** (§22.26) and that half IS
   device-verified (conv 6.7e-3/7.0e-3 vs 2.0e-2 tol). **The intent survives as
   "FP32 accumulators inside every first-cut op"** — check against that sentence,
   not the retired one. §22.23's pre-staged [I8] helper therefore converts to BF16
   (acceptance condition 4 is now `== BF16`).
4. **BLOCKER D RESOLVED by ruling (§22.27 + 04:48Z relay): `launch_top_candidates`**
   in src/ops/dflash2 — the repo had no top-k returning a RANKED SET (ops::sample
   draws ONE id; argmax returns one). Landed as a SKELETON with Test 8's TIE fixture
   (ids 5/9 tie at the top, 12/40 tie AT the k boundary — the cell a wrong tie
   direction fails). Compile-verified, not run.
5. **BLOCKER E RESOLVED (§22.28/§22.29): tensor-class container rows are BF16.**
   They were FP16 while every consumer needs BF16 — E1 (rmsnorm) threw loudly, E2
   (launch_conv reading FP16 bits as BF16 through a void*, same size, same shape)
   was the [I9] class one layer down and SILENT. Fixed in emitter + loader +
   feeder, and `dflash2_require_bf16` now guards every tensor-class row at bind.
   Evidence: `--feedcheck` gives **0.00e+00 vs gguf-py** (bit-for-bit, validates
   element ORDER). **Re-emit any sidecar you have on disk.** §22.29 also corrects
   the precision argument I first made for it — read that before repeating it.
6. **BLOCKER F ESCALATED, oracle half DONE (§22.30).** The block-input assembly
   (column 0 = fused features at the lane's frontier, columns 1..7 = the mask
   token's embedding) was in NEITHER §22.14's plan NOR the oracle — `MASK_TOKEN`
   was defined and unused. `assemble_block_input` + 7 self-check cells landed
   (bc65287b). The kernel half (`launch_block_input_assemble`) waits for A1.
   **The T=8-vs-T=6 tension is RESOLVED without a ruling:** the target's verify is
   T=k+1 (k<=5), the drafter's block pass is ALWAYS 8 columns/lane because the conv
   requires T % block_size == 0 — so blocker B costs acceptance breadth, not conv
   geometry, and §22.15's asymmetry argument is untouched.
7. **§5's quicklist rule CHANGED** (from the §22.26 slip): run the 7 CPU suites **by
   NAME** with `-E ninfer_dflash2_block_test`. `ctest -R dflash2` silently matches
   #147, which is a DEVICE test.

Two invariants worth carrying that this session discovered: the rope call needs the
DRAFTER's 8 absolute positions (not the target's verify positions), and the
attention kernel's `[hist_lo, hist_hi)` must come from the SAME frontier integer
§17.2's rollback maintains — append and window bound are one number read from one
place, or the drafter attends a window it never wrote.


## 0. One-line state
DFlash2 (non-AR block drafter for the 27B, MULTIBATCH-NATIVE per user ruling
§21): slices 1-4 + 5(a)(b)(c)(e)(f) landed; (d.0) FLIP landed (635b6f3b); **(d.3)
commit 2a LANDED AND DEVICE-VERIFIED (9a665689, §22.13)** — the batched runner now
admits DFlash2 loudly (6 refusal classes), emplaces the per-rank
DFlash2PersistentState (80 MiB/rank at T=6/lanes=2 = §17.2's 40 MiB/lane), binds
the (b) batch sink to a real batched verify, and proves the taps (all 10
(layer, lane) columns non-zero; lane 0 vs lane 1 differ at all five layers;
rank 0 vs rank 1 byte-identical). The drafter CHAIN is behind the 2b gate.
TWO WIDTH BLOCKERS FOUND ON DEVICE: **A = tp_gemv strided-output domain t<=8
(FIXED in 9a665689 via the prefill-style contiguous+pack path)**; **B = the
batched KVarN attend instantiates TokenTile 1..6 only, so DFlash2's window is
admitted only to 5 (T<=6) — closing it is a kernel-lane decision (A1), see
§22.13 blocker B.** Also landed: make_rank/TpBackend::create now key the MTP
machinery on the drafting SOURCE (mtp_round), without which a DFlash2 backend
could not be constructed at all, and tp_engine's draft-token clamp gained a
DFlash2 [1,7] arm. main cbf613fa still the folded tip (A1's Phase C NOT merged as
of 22:2xZ). **A1 RULED on blocker B (23:06Z, direct intercom, id ba9e0825 — the
direct-relay path, same as the §21.3 AGREE) and the coordinator RATIFIED it
(23:15Z, §22.13 addendum): keep the loud refusal, do NOT instantiate TokenTile
7/8 — measure (d.4)/(d.5) at window<=5 first, and the acceptance curve decides
whether the tile extension is worth opening. Read §22.15 item 3 before treating a
k=5 miss as closing B: the gate is asymmetric.** 3aa2555f adds the SECOND half of the multibatch-native rule that 2a
missed (a batch-of-one reaching deliver()'s single-seq route = the silent
wrong-drafter class) and pre-plans 2b-i..v + (d.4)/(d.5) in §22.14. 70fda122 adds
the ceiling-prompt device arm (multi-chunk batched prefill verified) + §22.15's
(d.5) pre-derivation, and RECORDS THAT the capacity guard added in that commit is
PROVABLY UNREACHABLE as a refusal (my earlier belief was wrong; derivation in
§22.15 item 6) — it stays as a defensive check, not as verified behavior.
**REMAINS: Piece B commit 2b — the chain + sidecar binding at TpBackend init —
HELD by coordinator ruling until A1's Phase C merges to main. Then (d.4)
measurement + encoding DECISION, (d.5) budget finalize.** GPU GRANT: 2a's device
verification is DONE → cards RELEASED to the coordinator (A1's staged window is
next); re-ask before any launch. Tip 20155ec8 = remote, tree clean, 7/7 green,
disk 22G (21 commits this session, c4eb5326..HEAD).

**§22.20 stage B IS LANDED (2de51571): `tools/convert/qwen3_8_27b/dflash2/
block_graph_ref.py`** — the single-block FP64 reference + the sidecar feeder that
dequantizes through the repo's canonical `dequantize_row_split` (so reference and
engine consume the SAME bytes). 17/17 self-checks + 2/2 feeder cross-checks
against gguf-py, all CPU-side. **It surfaced [I8]: the conv-base layout the
container holds (GGUF C-fastest: `c + C*(k + K*s)`) is NOT the layout
`dflash2_block.cu:44` indexes (`c*K*2 + k*2 + side`, C-outermost).** Resolve it in
2b-iii BEFORE trusting any conv number. **RESOLVED BY RULING (§22.21 addendum):
(a) transpose at bind time in `dflash2_bind.h` — A1's call, adopted; the exact diff
is PRE-STAGED in §22.23 (NOT applied — 2b-i's hold) and the acceptance condition is
that dropping EITHER the transpose OR the conversion turns the test red.
Discriminating vector landed (dcfadfd8, 20/20 self-checks).**
**AND [I9], found while writing that diff:** `launch_conv` is FP32-typed
(`dflash2_block.cuh:75` — x/base/proj_out/out all `float*`) while the container
stores conv bases as **f16** and the engine streams are **BF16**, so binding the
bytes as-is reads FP16 storage as FP32 — garbage, silently, shape check passing.
The bind-time helper settles `base`; the runtime half is ADOPTED as **FP32 staging
streams for the drafter's first cut** (§22.23 addendum; BF16 kernel variants are
the measurement-triggered fallback, and the drafter workspace recipe must NOT
reuse the target's BF16 allocations).
**[I5] RoPE IS CLOSED (CPU-side, §22.24) — and the ORACLE was the wrong side.**
The engine's `ops::rope` is SPLIT-HALF pairs `(i, i+rot/2)` over `rot = 128` = the
whole head, with `inv[i] = 1e7^(-2i/128)` — read off `rope.cuh`'s own index
arithmetic, pinned bit-exactly against all 64 entries of
`kDflashRopeInvFrequency`, and precedented by DFlash v1 (`dflash_impl.h:162/:231`
pass `rotary_dim = head_dim = 128` at the identical 32q/8kv×128/1e7 geometry).
The oracle had it as interleaved pairs over a partial 64-dim section — wrong on
pairing, width AND lattice, all silently. Fixed: `block_graph_ref.py` 20→26
self-checks (one-hot known answers + the full-table lattice pin),
`extract_config.py` now DERIVES `rope_rotary_dim = 2*sum(dimension_sections)`
(the entries count PAIRS), `DFlash2Config::rope_rotary_dim = 128` + 4 new
static_asserts, and the config.h comment claiming the sections "matter only for
image spans" — the sentence that seeded the misreading — is corrected. STILL OWED
(a formality, not a blocker): the device one-hot through the real kernel at the
next window; and "the q/k rows need no llama.cpp-style permute" rests on the
DFlash v1 precedent, i.e. strong inference, not measurement (§22.24).
The module still labels six shape-derived readings [I1]-[I4], [I6], [I7] (separate
from §9.1's commit-exactly architecture) — re-read `dflash.cpp` against them when
2b-iii starts.

**TWO THINGS A RESUMING SESSION MUST KNOW BEFORE TRUSTING ANY NUMBER IN THIS FILE
OR docs/151:**
1. The "430 MB/rank MTP-weight return" that appears in §22.13/§22.15 (and in the
   9a665689 commit message) is **WRONG as a device figure** — it is a BINDER-PLAN
   delta. `[rank N] materialized: 9059 MB` is identical before and after, because
   `materialize_tp` (tp_load.cpp:152) loads every `text/`/`mtp/` object
   unconditionally and `mtp/` REPLICATED. Corrected in place (§22.19) and
   reassigned by the coordinator to A1's load lane as an unbanked ~430 MB/rank
   candidate. Durable lesson: read the PER-RANK line before quoting a device
   figure; a plan-side and a device-side number with the same units are not the
   same measurement.
2. The (d.5) serving ladder is re-derived in §22.19 **by DELTA** (k5v4@163,840
   ~+231, k4v2@200k ~+441, k4v2@250k still OVER by ~13, w8g32 OVER by ~842) and
   **L=2 is the only admissible concurrency point** for DFlash2 (lanes<2 refused
   by §21; L>=3 refused by the GDN ring admission at k=5 — and closing blocker B
   does not change that). Those slacks are inside one-allocator-rounding of
   flipping the verdict, so (d.5) must REPLACE the table with measured-at-load
   numbers, not adopt it.

**CPU-side pre-planning is now at literal-diff granularity: §22.16 (fold-prep —
why this lane cannot affect MTP/plain paths + the reset-before-reassign ordering),
§22.17 (2b-i: the sidecar weights get their OWN DeviceBuffer/DeviceArena, NOT
`TpRankState::persistent`; 4 prerequisites, one tp_engine line deliberately NOT
written pending the fold), §22.18 (2b-ii: `ops::linear` rejects ne[2]!=1 and a
non-contiguous out → the whole drafter chain is 2-D [rows, T*lanes] with per-lane
structure only as a view()). Read those three before writing 2b code.**

**Two derived constraints to carry into (d.4)/(d.5) (§22.15, do not re-derive):**
(a) the measure-at-5 gate is ASYMMETRIC — a PASS at k=5 implies a pass at k=7, a
FAIL at k=5 is INCONCLUSIVE for k=7 (the drafter's round cost is weight-traffic
bound and T-independent, so k=7 is strictly cheaper per drafted token); (b) the
GDN ring admission caps batched speculative verify at **L<=2 lanes for BOTH k=5
and k=7** — DFlash2-batched is a two-lane feature on today's geometry and closing
blocker B does not change that.

## 0.2 GPU GRANT — RELEASED (2a's device verification is done)

Coordinator granted BOTH cards (DEVICES=0,1) to session 01a073a6 at 22:25Z,
carrying §0.2's terms; 2a used them and is device-verified (§22.13), so the
cards go BACK to the coordinator — A1 has a staged window (i fix-verify → ii S2
re-diag → iii bf16 OOM → iv §5 collapse). **Re-ask before any launch; "clear to
claim" is not a grant.** Standing terms: guard on FOREIGN CUDA CONTEXTS
(`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`), not ports;
kill only PIDs you started (a `ctest -j4` run here left `ninfer_bench_one_shot_ar`
holding 276 MiB — it was this session's own child, killed and re-verified clear);
never evict a foreign context.

## 0.5 First ten minutes (do these in order)

1. Reality check: `git log --oneline -3`, `git status --short`,
   `git ls-remote github wo/dflash2-scope` (must equal local HEAD — 9a665689 or
   newer), `df -h /` (22G free at this debrief; see gotcha #9 — a full-tree build
   here cost 15 GB).
2. Check main: `git fetch github main && git log --oneline github/main -3`.
   **A1's Phase C (tp2_backend non-kvarn fallback REMOVED, prefill table rows,
   attn_mix_tp batched arm) is the gate for commit 2b** — coordinator ruling
   22:2xZ: 2a proceeds on the current anchor, 2b holds until Phase C merges, then
   FOLD (merge, never rebase — gotcha #2) and re-anchor.
3. **GPU: ask the coordinator for a grant** (this lane's grant was released after
   2a's verification, §0.2). Guard on foreign CUDA contexts before any launch.
4. Poll via INTERCOM, not agent_comm (§6 — the comms lesson): any
   coordinator/A1 messages, gemini DT progress.
5. Run the verification quicklist (§5) — **9/9 by NAME** before touching anything
   (cells 8 and 9 are the conv-layout test and the diag-compile gate; and do NOT use a
   bare `-R dflash2`, §22.26).
6. Pick up §4.5 (Piece B commit 2b). Report a one-line status to the coordinator
   after EACH commit (their standing cadence request, 21:59Z).

## 1. Where everything is

- Worktree: `/home/intel/ninfer/worktrees/wo-dflash2` (branch `wo/dflash2-scope`,
  remote `github`). Backups: `backup/b81bc616-pre-rebase` (local+remote),
  `backup/2aa3546f-pre-k4v4` (local).
- Artifact: `/home/intel/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1.08 GiB, 81
  tensors, vintage CLOSED — metadata-complete, no reconversion needed).
- Tools: `tools/convert/qwen3_8_27b/dflash2/` — gguf_reader.py (stdlib
  inventory reader; CANNOT dequant), extract_config.py (config-from-artifact),
  block_ref.py (FP64 oracle), reencode_plan.py (weights size planner),
  requant_quality.py (double-quant study), block_graph_ref.py (THE §22.20 stage-B
  single-block FP64 reference + sidecar feeder; `python3 block_graph_ref.py` =
  20/20 self-checks, `--feedcheck <manifest>` = the gguf-py cross-check),
  emit_sidecar.py (container emitter; VERIFIED encodings today =
  `bf16 | f16 | w8g32_f16s | q4g64_f16s`, plus the RETIRED `w4g64_f16s@0`, which the
  loader rejects by name — §22.12: it packs +8-biased nibbles the engine would
  dequantize wrong. **`bf16` is now the TENSOR-CLASS encoding** (§22.29): norms and
  conv bases are written bf16 under EVERY --encoding choice, because their consumers
  require it. `f16` remains readable by the loader (and refuses at bind). nvfp4
  deliberately ABSENT. Invoke: `emit_sidecar.py <gguf> <out_prefix> --encoding
  w8g32_f16s`. **Any sidecar on disk predating §22.29 must be re-emitted.**)
- Code landed this lane: `src/ops/dflash2/` — **five files now, not two**:
  `dflash2_block.cu/.cuh` (conv + edge scores + repeat_4d + the block-input
  assembler, all BF16 storage / FP32 accumulators per §22.26, stream-pure),
  `dflash2_attention.cu/.cuh` (§22.25(b) drafter block attention — SKELETON),
  `dflash2_topk.cu/.cuh` (§22.27 D(i) selector top-k — SKELETON),
  `dflash2_block_ref.h` (FP64 oracle: conv, lattice, walk, block attention, and the
  shared `top_candidates` extracted out of the lattice builder), and
  `dflash2_conv_layout.h` (§22.23's [I8] permutation, host-clean so a CPU test can
  reach it). "SKELETON" means compile-verified and NOT device-verified — §22.31 is
  the gate list.
  `src/targets/qwen3_6_27b/impl/config.h` DFlash2Config (+ 35B zero-struct,
  + rms_eps), (d.1) DFlash2PersistentState/Layout (layouts.h /
  dflash_context.h/.impl), (c) chokepoints (layouts_impl arm — FLIPPED to the
  ceiling gate 635b6f3b, program_impl member-init guard), (b) sink factories
  + fc-fusion consumer (dflash_impl.h), tp_engine seam + DFlash2 admission
  refusals, ProgramImplCore dflash2 mirror (program.h/.impl.h), (d.2) loader
  `impl/load/dflash2_sidecar.h` (drift gate + validation + host_tensor +
  host_weight) and binder `impl/load/dflash2_bind.h` (444586c7), ModelView
  DFlash2Weights (model_view.h), --dflash2-sidecar option (types.h /
  serve_options / cli / generation_service). The binder now also carries the [I8]
  transpose and `dflash2_require_bf16` on every tensor-class row (§22.28/§22.40),
  and the loader accepts `bf16` and reports the dtype the CONTAINER declares.
- Tests: test_speculative_backend_enum.cpp (29 asrts, Parts 1-3, -Werror=switch
  on the TU), test_dflash2_config.cpp (static_asserts, + §22.24's rope-width cells
  and §22.31's cross-tree geometry asserts), test_dflash2_block_ref.cpp (registered,
  cross-check fail-loud), test_tp2_budget.cpp (seam cells),
  **test_dflash2_conv_layout.cpp** (§22.23's [I8] fix, CPU-only, 5 cells) and
  **tests/diag_sources_compile_check.cmake** (§22.38, syntax-gates every
  tools/smoke/diag/*.cpp — those drivers belong to no build target).
  test_dflash2_sidecar.cpp is now 33 assertions / 16 refusal checks, counted at
  runtime (the summary used to hardcode "12", which had already drifted).
- Device diagnostics (NOT ctest tests, and now compile-gated by
  `ninfer_diag_sources_compile_check`): `tools/smoke/diag/dflash2_tap_probe.cpp`
  — the driver 2a was verified with (§22.13, §5 recipe).
- 2a code sites: `tp2_backend.cpp` (admission block before the M4 fallback;
  the tap-probe block after `BatchedBindingScope`; `mtp_round` in make_rank),
  `tp2_backend.h` (TpRankState dflash2_backing/dflash2/width/lanes),
  `tp_engine.cpp` (DFlash2 [1,7] draft-token arm), `variant_kernels.cpp`
  (`gdn_input_projection_tp_verify` contiguous+pack above t=8).

## 1.5 Key SHA ledger (verified against git log — newest last within merges)

Session 01a073a6 (2026-09-05 late-late): c4eb5326 debrief handoff (from
01a0731a) -> **9a665689 (d.3) commit 2a — tap probe, DEVICE-VERIFIED (§22.13)**
-> 37574829 docs §22.13 + handoff refresh + tools/smoke/diag/dflash2_tap_probe.cpp
-> **3aa2555f batch-of-one dispatch refusal + §22.13 addendum (A1's blocker-B
ruling) + §22.14 2b/(d.4)/(d.5) pre-plan** -> dcb54b58 handoff refresh ->
**70fda122 ceiling-prompt device arm + §22.15 (d.5) pre-derivation (guard recorded
UNREACHABLE)** -> 96dc2f04 handoff refresh -> c3d6a088 §22.16 review pass + the
reset-before-reassign comment -> bdf29ca0 §22.17 (2b-i literal pre-stage) ->
e27a901a §22.18 (2b-ii + the [K,T] chain constraint) -> 82e744d4 ruling
provenance + ratification -> 9943874e §22.19 (ladder re-derived by DELTA + the
430 MB CORRECTION) -> 2e921966 §22.20 (2b-iii oracle plan) -> **2de51571 §22.21
stage B LANDED: block_graph_ref.py + the [I8] conv-base layout finding** ->
b93324d8 §7 debrief -> 2fdcbf9d §22.22 (stage-A spec for gemini: shapes + a-priori
bounds) -> **34f9f620 §22.23 ([I8] diff PRE-STAGED, [I9] found)** ->
20155ec8 §22.23 addendum ([I9] FP32-staging adopted) -> <this debrief>.

Session 01a07435 (2026-09-06 overnight, 15 commits, 6459bac6..8e6bfbbf):
6459bac6 the 01a073a6 debrief refresh found UNCOMMITTED in the worktree (content
unchanged — committed as-is, tree clean at last) -> **5ebdbebc [I5] CLOSED**
(oracle wrong on pairing+width+lattice; rope_rotary_dim derived + 4 asserts) ->
ad3cefe4 §22.25 BLOCKER C found -> e0f4ad81 §22.25 addendum (A1's ruling + coord
ratification) -> **b5590915 [I9]->BF16 LANDED, device-verified** (the one
retroactively-granted run, §22.26) -> **addf180c drafter-side attention kernel
SKELETON** (Test 7 written, not run) -> 0493b765 §22.27 2b-iv pre-stage + BLOCKER D
-> 73ec73f6 §22.28 BLOCKER E found -> **aaff2886 BLOCKER E RESOLVED** (BF16
container end to end + the bind guard; includes the correction to my own §22.28
argument) -> **bc65287b blocker F oracle half** (assemble_block_input, 26->33
self-checks) -> **be8d6f57 blocker D(i) launch_top_candidates SKELETON** (+ the
shared top_candidates oracle extraction) -> **489d5656 blocker F(i)
launch_block_input_assemble SKELETON** -> a62d173f handoff §0.1 + gotchas 12-13 ->
**8e6bfbbf §22.31 device-window checklist**. NOTE §22.32 supersedes §22.23's
pre-staged [I8] diff: apply THAT form, not §22.23's.

Previous session (2026-09-05 late, 01a0731a): 6774876e merge main@cbf613fa (Phase
S substrate fold) -> cdef9bac rms_eps pin -> 0ff8d5a5 (d.2) completion ->
9ce979eb docs §22.9 + handoff refresh -> 664e2fdb §22.10 (d.3) pre-plan ->
1892c98b §22.11 (d.0) flip pre-staged -> 635b6f3b (d.0) FLIP LANDED (A1
AGREE §21.3 addendum) -> a25d3dec §22.12 weight-class conformance retired ->
37ff711f handoff refresh -> 26675b8d loader host_weight() (real-artifact
cross-check) -> 3035b914 --dflash2-sidecar option + gate -> a9ab2eca handoff
refresh -> 444586c7 Piece A device binding (GPU-verified) -> c4eb5326 debrief.

Rebase-chain (pre-merge): ... -> 5611d5cf oracle registration -> c29fdc35 (c)
combined -> 671a0b6e tp_engine seam -> 287a00e5 S20 -> 1650cafb -Werror=switch
gate -> 5409b31c S20.4 -> 709c11bd merge main@36be5ad0 -> 2aa3546f S20 restore ->
d15157d9 (b.1) config+tap contract -> 9b8f977d merge main@51cd4722 ->
e6e0309d k4v4 name tables -> a76ac14e program_impl guard + Part 3 -> ca3292e2
S21 ruling -> 0f6ed97b S21.1 -> a38b5a82 S21.2 -> 9afc002a S21.3 DT review ->
1dac5306 merge main@35580e33 -> 2ff147dd S22 WO draft -> 789fe8e2 S22.1 weights ->
180e57c5 S22.2 cost model -> d7b3b999 S22.3/4 -> e8a89fdc S22.5 -> 4f93671b
S22.6 -> a3b49dfb (d.1) state -> 8bed9d2c S22.7 tap pseudocode -> b354636f
S22.8 loader spec -> e7ccdebc (b) layer -> 55c025e3 fc-fusion -> 7807b8fd
emitter -> 9310114f handoff. (Main's own commits -- 51cd4722, 35580e33,
8b6a9f07 -- are in this history via the merges; `git log --oneline` is the
truth.)

## 2. docs/151 section map (don't redo what's recorded)

- §3/§7/§9: artifact inventory + PINNED architecture (conv/selector/walk — the
  oracle ground truth for any vector).
- §10/§17: TP2 memory model, pool spec (§17.2 = the (e) geometry).
- §13/§22.7: Tap contract + tap pseudocode (fc fusion: GGUF ne=[25600,5120] =
  INPUT concat taps, OUTPUT 5120).
- §14.3/§22.3: cache-hit model (v1 = cold start; tail-2048 framing).
- §15/§15.1/§15.2 + §22.1: budget ladder + the FLIPPED k5v4@163,840 cell.
- §18/§19/§21.3: enum audit, test contract, DT review corrections.
- §20-§22.8: this session's records (rebase lessons, (d) WO draft, weights
  figure, cost model, requant study, loader spec).
- §22.9-§22.12: rms_eps correction, (d.3) pre-plan, (d.0) flip pre-stage,
  weight-class conformance retirement (session 01a0731a).
- **§22.13: (d.3) commit 2a — the tap probe, its device evidence, and the two
  width blockers (tp_gemv t<=8 fixed; KVarN TokenTile<=6 ruled-on).** Read this
  before touching the runner or widening the window.
- §22.14 / §22.17 / §22.18 / §22.23: the 2b execution list, with 2b-i and 2b-ii
  pre-staged to literal-diff level (incl. the [I8] bind-time transpose — written,
  NOT applied). Read these before writing 2b code.
- §22.15 / §22.19: the (d.4)/(d.5) plan and the serving ladder RE-DERIVED by
  delta with the 430 MB credit removed. §22.19 supersedes §22.1's numbers.
- §22.16: the fold-prep argument (why this lane cannot affect MTP/plain shapes).
- §22.20 / §22.21 / §22.22: the oracle plan, stage B as landed (+ [I8]/[I9]), and
  the stage-A shape table with a-priori tolerance bounds handed to gemini.
- **§22.24: [I5] CLOSED** (the oracle was wrong on pairing, width AND lattice).
- **§22.25 + addendum: BLOCKER C** — no usable attention op; (b) adopted with A1's
  five conditions; [I9]'s FP32-staging retired as unbuildable.
- **§22.26: [I9]→BF16 landed + device-verified**, and the UNGRANTED one-off device
  run with its corrective action (quicklist by NAME, `-E ninfer_dflash2_block_test`).
- **§22.27: 2b-iv pre-stage + BLOCKER D** (no ranked-set top-k); §22.36 audits all
  four gates for the C2 class.
- **§22.28/§22.29: BLOCKER E found and resolved** (FP16 container vs BF16
  consumers; E2 silent). §22.29 CORRECTS the precision argument that justified it.
- **§22.30: 2b-iii pre-stage + BLOCKER F** (the block-input assembly was in neither
  the plan nor the oracle) and the T=8-vs-T=6 resolution.
- **§22.31 + addendum: THE DEVICE-WINDOW CHECKLIST** — seven gates, their commands,
  and each one's paired mutation, including one mutation that must stay GREEN.
- §22.32: §22.23's pre-staged [I8] diff is SUPERSEDED (apply §22.32's form).
  §22.33/§22.35: the docs/152 review pass and its real catch. §22.34: the (d.4)
  harness. §22.37: what rule 14 does not cover. §22.38: the real-artifact re-check
  and the diag-compile gate. §22.39: fold-prep (new file, one call site).
  §22.40: 2b-i's file-level negatives + the checklist shrink.
- Companion: docs/152_dflash2_cuda_review_checklist.md (A1's kernel review
  checklist — A5 capture-safety is the class any drafter/graph test needs).

## 2.5 What is DONE vs what remains

DONE (code): (c) full chokepoint chain; (b) factories + fc-fusion consumer;
(d.1) state/layout; budget seam; kernels; oracle chain; container emitter;
(d.2) completion — manifest dflash2_config integration, C++ sidecar loader
(dflash2_sidecar.h: drift gate + validation + f16 wrap + row-split
host_weight), ModelView DFlash2Weights payload, rms_eps pinned from the
artifact (§22.9); (d.0) FLIP (635b6f3b — per-target ceiling gate + admission
loud refusals + test flip + 35B cell); weight-class conformance RETIRED
(a25d3dec — emitter packs via canonical encode_row_split; w8g32/q4g64
conformant by construction, §22.12); --dflash2-sidecar option + startup gate
(3035b914); Piece A device binding dflash2_bind.h, GPU-VERIFIED (444586c7);
**(d.3) commit 2a: batched-runner admission + per-rank drafter state + the (b)
sink bound to a real batched verify + tap evidence + the 2b loud gate,
DEVICE-VERIFIED (9a665689, §22.13)** — plus the tp_gemv strided-out fix (blocker
A), the mtp_round drafting-source split in make_rank/TpBackend::create, and
tp_engine's DFlash2 [1,7] window arm.
DONE (prep): §22 WO draft, cost model, quality study, loader spec, DT review,
interface proposal, §22.10/§22.11 pre-plans. HANDSHAKE: CONVERGED (A1 AGREE
both, §21.3 addendum). REMAINS: (d.3) commit 2b — the chain + sidecar binding at
TpBackend init, HELD by coordinator ruling until A1's Phase C merges (§4.5) →
(d.4) acceptance/requant-ladder measurement (GPU + n-stated prompts) + the
encoding DECISION (q4g64 vs nvfp4) → (d.5) measured-at-load budget
finalization. gemini's DT1-DT5 authoring overlaps (DT2's precondition — taps
non-zero + lane-isolated on device — is now OBSERVED, §22.13). OPEN KERNEL ITEM
(not this lane's to decide): TokenTile 7/8 for the batched KVarN attend, or
DFlash2's window stays capped at 5 (§22.13 blocker B).

## 3. Binding rulings (do not relitigate)

- **MULTIBATCH-NATIVE (user, §21)**: no single-seq DFlash2 path; 5(d)/(e) go
  through run_batch_dispatch → run_tp2_requests_batched; the program_impl
  guard (a76ac14e) is the PERMANENT single-seq refusal. DFlash2 on non-kvarn
  tiers must refuse LOUDLY at admission (never the silent per-lane fallback —
  §22.6).
- **Test split (user, §21.2)**: gemini authors; A2 implements. Landed A2
  suites stay (open to re-audit).
- **Parallel-with-A1**: his Step 1-2 (Phase S) LANDED and is folded
  (main@cbf613fa), so the dispatch region is editable — that is how 2a and
  3aa2555f got in. The CURRENT hold is a different one: **2b holds until A1's
  Phase C merges to main** (coordinator ruling), and this lane touches tp_engine
  again only at the fold (the `deliver()` arm is the flagged conflict).
- **Serving points**: SUPERSEDED. §22.1's cells (+111/+53, 200k ~321, 250k over)
  were re-derived in **§22.19** by delta at k=5/L=2 with the 430 MB credit removed:
  k5v4@163,840 ~+231, + F16 codebooks ~+170, k4v2@200k ~+441, k4v2@250k STILL
  OVER by ~13, w8g32 OVER by ~842 (measurement-only). **L=2 is the only admissible
  concurrency point.** (d.5) must REPLACE that table with measured-at-load numbers,
  not adopt either version.
- **§19.2**: DFlash2's gate is a divergence-RATE characterization with n
  STATED — never byte-identity vs plain decode.
- **rms_eps**: RESOLVED (§22.9) — the key dflash.attention.layer_norm_rms_epsilon
  IS published (F32 1e-6); REQUIRED by extract_config.py, pinned in
  DFlash2Config::rms_eps, consumed by dflash2_fuse_features (no caller param).
  test_dflash2_config.cpp static_asserts it; mutation on record.

## 4. Gotchas that bit THIS session (all verified real)

1. `git checkout --` on files holding UNCOMMITTED work is destruction (two
   accidents: staged layouts arm; uncommitted DFlash2Config + a STALE-object
   fake green). RE-PROVEN THIS SESSION by this agent (option plumbing in
   serve_options.cpp wiped mid-mutation-test; redone from the edit log —
   no loss, but the trap is real). Mutation runs use targeted reversal
   (sed/python), NEVER git checkout. Related trap hit again: mutating a
   header/lib and running the TEST binary without rebuilding the test target
   = stale binary, vacuous "pass" (the enum-test 35B-cell mutation first read
   green for exactly this reason — rebuild the TEST target, capture rc
   explicitly).
2. Rebase can silently DROP content with no conflict (three docs/151 passages
   auto-merged away; the instructed rebase also truncated docs/151 to 148
   lines at its first stop — aborted). Merge for fold-ins unless re-verified
   safe; after any rebase diff your doc against the pre-rebase copy.
3. Do NOT include instantiate.h in a TU that links the engine — second
   instantiation segfaults at static-init. For hand-built-plan type access:
   layouts.h + engine-matching macros (a76ac14e note).
4. Sub-ulp mutations cannot fail anything (+1e-300 rejected); capture rc
   BEFORE piping (`| tail` measures tail).
5. ~/bin/ctest is a BROKEN pip shim — use /usr/bin/ctest.
6. Commit ALWAYS with explicit pathspec (a mixed commit happened; split by
   soft-reset).
7. gdb LAUNCHING a fresh child works (ptrace_scope only blocks attach) — used
   to localize the Part-3 segfault.

8. An "absent key" claim is only as good as the inventory list it came
   from: the dflash2 rms_eps key existed all along under the `dflash.` arch
   prefix while REQUIRED_META greps implied absence (§22.9). Before treating
   "artifact does not publish X" as fact, dump the FULL metadata key list
   (gguf_reader inventory) and search it.

9. **Never run a full-tree `cmake --build build` on this disk** (§22.13 disk
   incident): 148 statically-linked test executables at ~124 MB each ≈ 15 GB, and
   / went 23 GB free → 98% full in one command. Build TARGETS (`--target
   ninfer_engine`, the touched test targets). If it happens anyway, the fix is
   deleting THIS worktree's own non-quicklist test binaries (regenerable
   per-target) — never another lane's build dir.

11. **A handoff's own numbers go stale faster than its prose.** This file was
    carrying a derivation count that was simply wrong (9 vs the real 10), an emit
    encoding that §22.12 RETIRED (`w4g64_f16s@0` — the loader rejects it by name),
    serving cells §22.19 superseded, and a "no dispatch edits until Step 1-2"
    ruling whose premise had already landed. Each was checkable in seconds and each
    would have misled a resuming session. So: re-verify SHAs, counts, encodings
    and file paths mechanically at every debrief, and write the METHOD next to the
    number (§5 does this now). Same class as §22.19's plan-vs-device 430 MB error
    and §22.15 item 6's unreachable guard: a number measuring something other than
    what it is claimed to measure.

14. **Never let `git commit` be a separate statement from the edit it claims to
    record.** Twice in ~90 minutes this session (5abadb05, 25af36a3) a commit message
    described a change whose edit had failed — because the python that edits and the
    git that commits were newline-separated rather than `&&`-chained, so the script
    aborted on an assertion and the commit ran anyway. Both were caught by noticing a
    RESULT disagreed with the claim, not by checking the diff, which is the check that
    would actually have caught them. Mechanical version: `edit && add && commit`, and
    after any commit whose message says "updated X", `git show | grep X` before moving
    on. A message that over-states its diff is not a small honesty lapse — it is a
    false claim in the artifact the next session trusts most, and this file is that
    artifact.

12. **A `ctest -R` pattern is not a list of tests you chose.** `ctest -R dflash2`
    matches #147 `ninfer_dflash2_block_test`, which calls `cudaSetDevice(0)` — so a
    "CPU suites" run silently used the GPU without a grant (§22.26). The lane's rule
    now: the §5 quicklist BY NAME with `-E ninfer_dflash2_block_test`. Related and
    worse: a build that FAILS leaves the previous binary on disk, and running it
    anyway produces a green from a mutation that never compiled — this session read
    exactly that result before checking build rc. **Check the build rc before the
    test rc** (gotcha #1's second form, and #4's third).

13. **A dtype mismatch between two 2-byte types is invisible to every check that
    exists.** FP16 and BF16 are both 2 bytes with the same `ne`/`nb`, so a `void*`
    handoff from container to kernel reads one as the other with a passing shape
    check — blocker E2 (§22.28), and the same class as [I9]. The fix is not the
    conversion, it is the assertion: `dflash2_require_bf16` at bind, generalized to
    "declared dtype must match consumer dtype". Same trap reaches tests: the first
    draft of Test 8 carried int32 candidate ids through a bf16-typed buffer.

10. **Two width domains are narrower than the drafter's architecture** (§22.13):
    `tp_gemv` writes STRIDED outputs only up to t<=8 (above that it delegates to
    `ops::linear`, which demands a contiguous out — and the GDN verify
    projection's qkv row-slices are never contiguous), and the batched KVarN
    attend instantiates TokenTile 1..6 only. A "T x lanes" width beyond either
    fails DEEP in the forward with a generic message. Both are now admission-
    checked with named reasons; check both before widening the window. Related:
    keying weight loading on `mtp_k > 0` silently loads MTP weights for a
    DFlash2 server — the drafting SOURCE, not the window field, must decide.

## 4.5 NEXT STEPS, in order (the goal: DFlash2 complete)

1. **(d.0) chokepoint flip — DONE 635b6f3b** (per-target ceiling gate + test
   flip + 35B negative cell + admission refusals; program_impl guard STAYS).
2. **(d.3) Piece B — commit 2a DONE (9a665689, §22.13, device-verified)**: the
   runner-entry throw is replaced by real admission (6 loud refusal classes at the
   runner, +1 at the dispatch since 3aa2555f = 7 total) + a
   per-rank DFlash2PersistentState on its own DeviceBuffer + pending_features
   staging + the (b) sink bound to the existing target_verify_batch overload +
   tap evidence + the 2b loud gate. Blocker A (tp_gemv strided-out t<=8) fixed
   en route; blocker B (KVarN batched attend TokenTile<=6) escalated, not
   worked around. Piece A (444586c7) supplies the bind the chain consumes.
   **NEXT = commit 2b, HELD by coordinator ruling until A1's Phase C merges to
   main.** The full 2b-i..v split (sidecar-at-init -> fuse -> blocks ->
   selector/walk -> serve exposure LAST), the (d.4) measurement plan under A1's
   ruling and the (d.5) k=5 budget restatement are in **§22.14** — read that
   first, it is the execution list. Steps, mechanics all resolved (§22.10 B,
   §22.12.6, §22.13):
   - Fold main (MERGE, never rebase — gotcha #2) and re-anchor on the
     post-Phase-C runner region (Phase C removes the non-kvarn per-lane
     fallback this lane's refusals invert).
   - Sidecar load at backend init: EngineOptions.dflash2_sidecar ->
     DFlash2Sidecar -> bind_dflash2_modelview (444586c7) — wire the call
     site in TpBackend init; refuse to start if the file is unreadable
     (startup gate already exists at serve/cli options, 3035b914).
   - Chain: dflash2_fuse_features (eps pinned) -> 5 blocks (ops/dflash2
     launch_conv + attention [dflash.attention.causal=0 -> bidirectional
     within block; sliding 2048, replicated 8 KV heads] + FFN) ->
     output_norm -> selector (launch_edge_scores +
     launch_repeat_single_pred, capacity contracts reused) -> top-k 16 ->
     CPU walk (host-issued first cut, §22.6; NINFER_DFLASH2_TRACE dump).
   - Proposals -> existing prepare_verify_inputs/accept/select/next_round at
     T=k+1 (window request-derived, admitted [1,5] until blocker B closes);
     sampled rows inherit the S1 per-lane mb_cfg path; accept -> cyclic
     append + frontier rollback.
   - The 2a probe block is the insertion point: replace the trailing 2b-gate
     throw with the chain, keep the tap-evidence dump (it is what DT2 reads),
     and keep every admission refusal.
   - Device verification: ask the coordinator for a fresh grant (§0.2 — the 2a
     one is released), then the probe recipe in §5. Emit the test container with
     `--encoding w8g32_f16s` (measurement class; q4g64/nvfp4 shipping is (d.4)).
   - 2b-iii carried THREE prerequisites; **TWO ARE NOW LANDED**: the [I8] bind-time
     transpose is APPLIED and CPU-tested (8af59a3c, `dflash2_conv_layout.h` + the new
     8th ctest cell — §22.23's pre-staged diff is SUPERSEDED by §22.32's form, and
     both are now behind the code), and the [I9] dtype decision is landed AND
     device-verified (b5590915: BF16 operands, FP32 accumulators). **What 2b-iii
     still owes**: [I5]'s device one-hot through the real `ops::rope` (a formality —
     the convention is closed on the reference side, §22.24), the §22.31 gates for
     the three new kernels, and the block body itself.
3. **(d.4)**: acceptance + requant-ladder measurement (GPU + n-stated
   prompts; cold-start per §22.3) — ladder now runnable: w8g32 ceiling →
   q4g64 uniform → F16 codebooks; ALSO the encoding DECISION (q4g64 vs
   nvfp4 — A1's MSE question; w8g32 measurement-only, ~2 GiB).
4. **(d.5)**: measured-at-load budget finalization; REPLACE the ladder — start
   from §22.19's delta-derived table (k=5, L=2, no 430 MB credit), add the
   measured pool (80 MiB/rank at T=6/L=2, verified in 2a) and the <5 MB round
   staging (§22.23 addendum), and re-derive rather than inherit.
5. Throughout: gemini's DT1-DT5 land against these implementations (his
   §21.2 map + the §21.3 corrections: DT3 vectors from block_ref.py,
   mutate-enum = disable-an-arm, DT2/DT5 (d)-gated skeletons — DT2
   activates when the round binds the sink).

**Definition of done for the feature**: DFlash2 selectable end-to-end on the
batched dispatch (serve + cli), serving gates green on a KVarN tier at an
approved serving point (§22.1 ladder), divergence-rate characterization
passed (§19.2, n stated), budget line finalized from measured load, and
gemini's DT1-DT5 green. Nothing less.

## 5. Verification quicklist (before ANY "done" claim)

- Suites — **run them BY NAME with the device test excluded** (§22.26 corrective
  action; a `-R dflash2` pattern silently matches #147 `ninfer_dflash2_block_test`,
  which creates a CUDA context, and this lane has no standing grant):
  `/usr/bin/ctest --test-dir build -E "ninfer_dflash2_block_test" -R "^(ninfer_
  speculative_backend_enum_test|ninfer_tp2_budget_test|ninfer_dflash2_config_test|
  ninfer_dflash2_block_ref_test|ninfer_dflash2_sidecar_test|
  ninfer_dflash2_conv_layout_test|ninfer_diag_sources_compile_check|
  ninfer_request_log_test|ninfer_serve_options_test)$"` → **9/9 Passed**. Session
  01a07435 added two cells to the lane's 7: `ninfer_dflash2_conv_layout_test` (§22.23's
  [I8] fix made testable with no device at all) and
  `ninfer_diag_sources_compile_check` (§22.38: the hand-linked drivers under
  tools/smoke/diag belong to no build target, so this is the only thing that notices
  when a signature change strands one — it happened for real, see the section).
  The symbols check (`ninfer_dflash2_symbols_check`) is CPU-only cmake and safe to
  run separately; it now requires `launch_block_attention` and
  `launch_top_candidates` to be present in libninfer_ops (rule 10).
- Targets: ninfer_engine (contains tp_engine.cpp — the file that proves the
  change), the touched test targets, cli `ninfer` when cli touched.
- Derivation counts vs main — RE-MEASURED at this debrief because the numbers
  this file used to carry were stale. Method:
  `for sym in kv_bytes_for_tier verify_no_divergence validate_cache_type_widths
  drafter_fixed_bytes DrafterBudgetBackend; do grep -rn "$sym" src/ tests/
  --include=*.h --include=*.cpp | wc -l; done`
  → 7, 6, **10**, 10, **11**. Compare with
  `git grep -n "$sym" github/main -- src tests | wc -l`: identical EXCEPT
  DrafterBudgetBackend (main 10, HEAD 11), and that +1 is this lane's
  (d.0)/2a arm — the only legitimate delta. (Old text said 9 and 10: 9 was simply
  wrong, 10 predated our arm.)
- Reconfigure if needed: -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
  -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES=120a.
- DEVICE scratch-TU recipe (used by Piece A verification, 444586c7): emit a
  full container `python3 tools/convert/qwen3_8_27b/dflash2/emit_sidecar.py
  <artifact.gguf> /tmp/d2_dev --encoding w8g32_f16s` (~2 GiB temp — DELETE
  after, disk is the shared constraint); compile
  `g++ -std=c++20 -I src -I include -I third_party
  -I src/targets/qwen3_6/export -I src/targets/qwen3_6_27b/export
  -I /usr/local/cuda-13.1/include /tmp/x.cpp build/src/libninfer_core.a
  build/src/libninfer_artifact.a -L/usr/local/cuda-13.1/lib64 -lcudart
  -Wl,-rpath,/usr/local/cuda-13.1/lib64`; guard contexts first (§0.2).
- Mutation runs: rebuild the TEST TARGET (static link — a lib-only rebuild
  leaves a stale binary that passes vacuously, hit this session), capture rc
  BEFORE piping, restore via targeted python reversal — NEVER git checkout
  on files with uncommitted work (gotcha #1, re-proven the hard way).

- **OPEN VERIFICATION DEBT (do not lose it):** 3aa2555f's `deliver()`
  batch-of-one refusal is compile-verified + 7/7 green but NOT device-exercised
  — reaching it needs a live TpBackend inside TPEngine, i.e. 2b-i's sidecar-at-
  init. Close it in 2b-v (serve exposure) with a real single-request DFlash2
  server: expect the refusal, never a decoded completion. (The per-lane capacity
  guard added in 70fda122 is a SEPARATE case: it is provably UNREACHABLE as a
  refusal — §22.15 item 6 — so it is recorded as a defensive check, never as
  verified behavior. Do not "verify" it by trying to trip it; the prefill bound
  kills the run first with `slice range out of bounds`.)

- **DFlash2 batched-runner device probe (what 2a was verified with, §22.13).**
  Driver: `tools/smoke/diag/dflash2_tap_probe.cpp` (committed, but NOT a ctest
  test — §21.2, gemini owns DFlash2 test authoring): builds a TpBackend with
  `speculative_backend = DFlash2` + KVarN + max_concurrency=2 and drives
  `run_tp2_requests_batched` directly with raw
  token-id prompts (no tokenizer needed). THREE ARMS: `--mode kvarn` (6 checks:
  the five admission refusals + the accepted path reaching the 2b gate),
  `--mode bf16` (1 check: non-kvarn refused before the per-lane fallback),
  `--mode cap` (3 checks: the ceiling prompt plen == max_context forces a
  MULTI-CHUNK batched prefill, the capacity guard stays silent, the run reaches
  the 2b gate). All three were run green on 2026-09-05 (docs/151 §22.13/§22.15).
  Link line (the tp2 test target's, with
  the driver stub for `-lcuda`):
  `c++ -O2 -std=gnu++20 -Wswitch -I tests -I include -I src -I third_party
  -I third_party/utf8proc -I src/targets/qwen3_6/export
  -I src/targets/qwen3_6_27b/export -I src/targets/qwen3_6_27b/impl
  -isystem /usr/local/cuda-13.1/targets/x86_64-linux/include
  -isystem /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl
  /tmp/d2_probe.cpp build/src/libninfer_engine.a libninfer_core.a
  libninfer_artifact.a libninfer_ops.a libninfer_nvfp4_tma.a libninfer_core.a
  /usr/lib/x86_64-linux-gnu/libnccl.so -L/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs
  -L/usr/local/cuda-13.1/targets/x86_64-linux/lib -lcudart -ldl
  /usr/lib/x86_64-linux-gnu/librt.a libninfer_text.a libninfer_media_decode.a
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lcuda
  -Wl,-rpath,/usr/local/cuda-13.1/targets/x86_64-linux/lib -o /tmp/d2_probe`
  (the cccl -isystem is required by cuda_fp16.h on CUDA 13; `-lcuda` resolves
  the nvfp4 TMA driver symbols). Run: `NINFER_DFLASH2_TAPDUMP=/tmp/d2_taps.log
  /tmp/d2_probe --mode kvarn` (and `--mode bf16` for the non-kvarn refusal).
  Expect `D2-PROBE: ALL PASS`, 10 tap lines with per-lane-distinct and
  per-rank-identical hashes, and rc=0. A full TpBackend load takes ~60 s and
  ~9 GB/rank — one backend per process, and NEVER `--mode` both in one process.

## 6. People/protocol

- **INTERCOM → coordinator (01a07140) and agent1/A1 (01a07304)** — the
  coordinator does NOT read agent_comm: my ids 230-233 self-looped there
  (from_agent == to_agent == my own id) and the coordinator saw silence for
  ~45 min while I was "idle". agent_comm is for gemini (256adaf6) only, via
  the broker. Verify liveness: `intercom({action:"list"})`.
- Report a ONE-LINE status after EACH commit (coordinator cadence request
  21:59Z): what landed + next. Pushed SHAs are the only truth they accept.
- A1 rulings arrive DIRECTLY on intercom now (the 21:31Z AGREE skipped the
  relay) — record them in docs/151 §21.3 addenda with timestamp.
- gemini: test author (agent_comm/broker; his mesh sends to me failed once —
  "Agent not found" — broker delivery still works). UPDATE 23:06Z this session:
  DT2's sink half is LIVE on device (2a), and the tap-evidence output shape is
  committed in tools/smoke/diag/dflash2_tap_probe.cpp — so DT2 can be authored
  now, against two named constraints (window<=5 admission; the six refusal
  classes). DT5's chain cells stay (d)-gated until 2b-iv.
- GPU: coordinator grants, ALWAYS in writing to YOUR session id (this session
  had to be re-granted after the handoff: cards release when a session ends).
  CURRENT STATE: RELEASED — 0 compute apps, both cards idle at this debrief.
  Guard on foreign CUDA contexts via nvidia-smi, never ports; kill only PIDs you
  started (a `ctest -j4` run here left `ninfer_bench_one_shot_ar` holding
  276 MiB — mine, killed, re-verified); gpu_guard.sh lives in this worktree's
  tools/smoke/diag/.

## 7.0 Debrief — session 01a07435 (18 commits, 6459bac6 -> 3cbea157, CPU-side only)

What a resuming session should take from this one, in order:

1. **Reading op contracts is the highest-yield activity available without a
   card.** Every blocker found today (C, D, E, F) came from reading what an op
   DECLARES rather than from running it, at a rate of roughly one per 20 minutes,
   and each would have failed deep inside a device window. The method that worked:
   take the next planned step, list the ops it will call, and read each one's
   validation code for the dtypes/shapes/domains it refuses. §22.25, §22.27,
   §22.28, §22.30 are that method applied four times.
2. **A pre-staged diff is code, and code gets compiled.** §22.23's [I8] helper sat
   in the docs uncompiled through two dtype rulings and went stale twice; the
   corrected form (§22.32) failed three times before `g++ -fsyntax-only` accepted
   it, including one breakage caused by my own line-number edit of the prose.
   Apply §22.32's form, not §22.23's.
3. **Three new kernels landed as SKELETONS with their gates written but unrun**:
   `launch_block_attention` (§22.25b), `launch_top_candidates` (§22.27 D-i),
   `launch_block_input_assemble` (§22.30 F-i). All compile-verified, all in
   `src/ops/dflash2`, none device-verified — §22.31 is the checklist that turns
   them into claims, with each gate's PAIRED mutation named. Do not cite a number
   from any of them until §22.31 items 1-3 are run.
4. **Two rulings changed what this file says about dtype.** [I9]'s "FP32 staging
   streams" is RETIRED as unbuildable; the intent survives as "FP32 accumulators
   inside every first-cut op". And the container's tensor-class rows are now BF16
   (§22.29), so **any sidecar on disk must be re-emitted**. §22.29 also corrects
   the precision argument that justified it — read that before repeating it.
5. **One invariant no test catches** (§22.30/§22.31): the attention kernel's
   `[hist_lo, hist_hi)` and §17.2's rollback frontier must be ONE number read from
   ONE place. Divergence means the drafter attends a window it never wrote, and
   every gate above would still pass. Enforce it in review.
6. **The protocol slip and its fix** (§22.26): `ctest -R dflash2` matches a DEVICE
   test. Quicklist by NAME with `-E ninfer_dflash2_block_test`. Related: a failed
   build leaves the previous binary on disk, so check build rc before test rc.
7. **A commit of mine over-claimed its diff** (5abadb05's message described a
   paragraph its script never inserted, because the python and the git commit were
   separate statements). Fixed in 49dd63da and recorded rather than amended away —
   the lane's rule is that pushed SHAs are the only truth, so they do not get
   rewritten to look better than they were. If you are checking a claim, check it
   is IN the diff, not just in the message.

8. **The review pass caught something the tests could not** (§22.35/§22.36): the
   attention kernel and its FP64 oracle both admitted history with `pa <= p_i`, so a
   history entry inside the block was double-counted and the gate returned green —
   docs/152 C2 in the reference, not the test. Fixed in all three places (bound is
   now `pa < block_first`). §22.36 audits the other three gates: only attention was
   mirrored, because its admitted-SET was transcribed rather than re-derived. The
   lesson generalizes past this lane: **FP64 + a different loop order is not what
   makes an oracle independent — a second, differently-derived justification for
   every decision is.** Precision is the easy half.

Unchanged by this session: 2b is still HELD on A1's Phase C merge (main is still
cbf613fa); the definition of done (§4.5) is unchanged; a DFlash2 server still
refuses to decode anything, by design, until 2b-iv.

## 7. Debrief — session 01a073a6 (17 commits, c4eb5326 -> 35cde728)

What a resuming session picks up, in order:
1. **HOLD: 2b needs the coordinator's Phase C flag.** main is still cbf613fa
   (verified at this debrief); A1's Phase C is NOT merged. When it lands: FOLD
   (merge, never rebase — gotcha #2), re-anchor the admission block on the
   post-Phase-C runner region, and expect the ONE flagged textual conflict to be
   tp_engine's `deliver()` arm vs `can_batch` (the coordinator holds that flag).
2. **Then execute §22.14's 2b-i..v in order** (sidecar-at-init -> fuse -> blocks ->
   selector/walk -> serve exposure LAST). §22.17 and §22.18 are literal diffs
   already; 2b-iii must carry the [I8] bind-time transpose (§22.21 addendum) and
   its "must fail un-transposed" acceptance condition.
3. **Two debts that must not be inherited as facts:** the 430 MB plan-vs-device
   figure (§22.19 correction; the real lever is a materialize_tp mtp/ filter, now
   A1's load lane) and the capacity guard (UNREACHABLE, never "verified").
4. **[I5] RoPE is CLOSED (§22.24)** — split-half, rot=128, lattice pinned against
   the engine's own table; the oracle was wrong, not the engine. The device one-hot
   is a formality still owed at the next window, and the no-permute-needed claim
   rests on the DFlash v1 precedent (inference, not measurement).
5. (d.4) then (d.5) per §22.15/§22.19: k=5 primary arm, n stated, cold start, the
   w8g32 -> q4g64 -> F16-codebooks ladder, and the ladder table REPLACED by
   measured-at-load numbers rather than adopted from §22.19.

Definition of done for the feature is unchanged (§4.5) and nothing in this
session moved it: 2a is a verified intermediate, not a shipping path — a DFlash2
server still refuses to decode anything, by design, until 2b-iv.
