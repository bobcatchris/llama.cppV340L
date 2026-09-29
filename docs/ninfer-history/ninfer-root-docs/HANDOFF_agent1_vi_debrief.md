# DEBRIEF — agent1 (vi) order-dependence hunt → ROOT CAUSE FOUND (2026-09-06)

**Read this first if you are continuing the (vi) work.** This is the fast runway: the complete
state of the (vi) order-dependence investigation, the instrument suite, the elimination chain,
the root cause, and the exact next steps. Written 2026-09-06 ~10:3xZ by the session that ran the
(iii)/(iv)/(v) deliveries and the (vi) hunt.

**Lane:** `wo/kvarn-multibatch`, tip **`2d440662`** (pushed), tree clean.
**Main:** `cfca00ca` (Phase C partial-merge done by the coordinator: (iii)+(iv)+(v)+S2 kvarn-only
gating are IN MAIN).
**Cards:** RELEASED to A2 (2b device work). Re-request a bounded window for the (vi) verification
run — the coordinator's handoff flow works (claim directly after re-guard when he grants).

---

## 1. THE DEFECT (vi) — one paragraph

On the batched MultiBatch runner, the same two requests (one greedy temp=0, one sampled temp=1.0)
produce **different generated text depending on which lane (0/1) each request occupies** — on
**bf16 and i8** the sampled lane's text diverges from ~token 3 under a lane order flip; on
**kvarn_k4v2 the same test is byte-identical** (order-invariant). The divergence persists for the
whole generation (not a single token flip). This blocks bf16/i8 S2 gating (S2 = sampled-lane
reproducibility across servers) and is the last open item before bf16/i8 S2 re-enable.
Related DFlash2-lane rulings made tonight (context, not actionable here): BLOCKER C = drafter-side
attention kernel (b) + [I9]→BF16 (ratified); BLOCKER D = dflash2-local launch_top_candidates (i);
BLOCKER F = launch_block_input_assemble (i); BLOCKER G = codebooks at w8g32 first cut (i) + the
embedding Q4G64 case as follow-up (ii) — G's (ii) is THE SAME ops::embedding Q4 gap as the
parked single-GPU kernel doc (docs/single_gpu_capacity_kernel_parked.md §4).

**Test harness:** `tools/smoke/diag/hashpt_order_probe.sh` — two single-server sessions (runA:
greedy fired first = lane0; runB: sampled fired first = lane0), staggered fire
(NINFER_BATCH_WINDOW_MS=500 + 300ms), NINFER_MB_HASHPT=<file> capturing all hash points.
`tools/smoke/diag/s2_order_diag.sh` (KV_DTYPE/KV_CAPACITY/MAX_CONTEXT parameterized) is the
3-run discriminator (exit 0 = order-invariant; exit 2 = order-dependent).

**Verified mapping (plen instrument in B1_pack): greedy prompt = 64 tokens (lane0 = leader =
first-fired); sampled prompt = 61 tokens.** The leader is lane 0 in both orders.

---

## 2. ROOT CAUSE (found, device-proven, committed 2d440662)

**The round-1 ACCEPT result (per-lane accepted token count) is ORDER-DEPENDENT.**

Measured (B3b/B3c per-step capture, correct request pairing):

| request | runA r1→r2 position advance | runB r1→r2 position advance |
|---|---|---|
| greedy | 69 → 70 (**+1**) | 69 → 72 (**+3**) |
| sampled | 66 → 69 (**+3**) | 66 → 66 (**+0**) |

With round-1 inputs byte-identical across orders (see §3): matched draft ids
(greedy 4087,1156; sampled 6587,264), matched windows (68/65), matched positions, matched
round-1 step-states (B3b: pos/rope/valid/win all equal), and round-1 verify hiddens that differ
only from the first draft step onward (B3c: hid differs at sl=0 while drafts still match).

Also: round-1 draft-step OUTPUT hiddens differ at sl=0 (greedy d13aa079 vs 5b5e8664; sampled
cc487b48 vs baa6d033) while the draft IDs still match — the hidden divergence is sub-argmax at
round 1, and the accepted-count divergence at the r1→2 boundary is the first VISIBLE divergence.

**This is an ACCEPT-LEVEL defect** (in or around `speculative_accept_greedy_drafts` /
its runner-level inputs at the r1→2 boundary) — NOT kernel numerics (see §3: every batched
verify op is per-slot pure by code read).

---

## 3. ELIMINATION CHAIN (what is CLEARED — do not re-derive)

Every item below was CLEARED by device measurement (ordered reads, correct request pairing) or
full code read:

1. **The 48 GDN layers' delta-net inputs** — QKVDBG per-slot hashes (NINFER_MB_QKVDBG_STEP=1):
   q4/k4/v4/g4/b4 slot hashes MATCH across orders at round 1, all 48 GDN layers. The residual
   stream matched at every GDN boundary.
2. **The conv snapshot kernel** (`causal_conv1d_batched_snapshot_smallt_kernel`,
   causal_conv1d.cuh:467) — batch = blockIdx.y, history from its own initial slot, snapshots to
   its own base. Per-slot pure.
3. **The delta-net recurrent body** (`recurrent_bf16_body`, recurrent.cuh:647 + `apply_gdn_transition`
   :212) — per-lane state tiles, sequential token loop, warp_sum over the slot's own 32 lanes.
   Per-slot pure.
4. **The bf16 small-T verify kernel + reduce** (`gqa_attention_small_t_tc_partial_bf16_kernel`,
   gqa_attention_decode_bf16.cuh:20 + the reduce :146) — per-slot column_base/table_row/valid/
   partials; causal mask slot-local; the reduce recomputes active splits from positions (stale
   inactive-split partials are never read). Per-slot pure.
5. **The I8 tiled verify kernel** (`gqa_attention_decode_i8_tiled_kernel`,
   gqa_attention_decode_i8.cuh:58, MultiBatch/Masked/DynamicArena arm) — same per-slot structure;
   per-token quant scales from the slot's own projections (warp_max within the slot's 32 lanes).
   NO cross-slot coupling. Per-slot pure.
6. **Order-assigned RNG seeds** — translate.cpp: a request without a seed inherits the server
   --seed deterministically (both lanes share 20260905). Order-invariant.
7. **B==N compact-vs-original remap** (the S1-i8-composition defect, b73fd4aa) — invisible at
   B==N; the (vi) divergence appears at B==N round 1-2.
8. **The prefill** — B0_seedhid (round-0 prefill hidden per lane): MATCHES across orders per
   request (sampled 5b04 both runs; greedy 59d3 both runs). Order-invariant.
9. **The full-attention layers 0-62 and all GDN layers** — implied by (1): the residual matched
   at every GDN boundary, so everything before the last GDN boundary is order-invariant in bf16.

**THE ONLY UNPROBED SEGMENT: the post-accept draft chain of round 1** —
`speculative_select_accepted_hidden` → `launch_draft_head` → `mtp_forward_decode_batch` ×3 —
whose round-2 draft OUTPUTS differ while all visible inputs (accepted hidden v_arh per B3*,
accepted ids per B4, anchor/F per B5) matched. NOTE the subtlety discovered late: the draft-chain
hidden ALREADY differs at round-1 draft step 0's output (B3c) with matched step inputs — so the
order-dependence may be INSIDE `mtp_forward_decode_batch` (the batched [1,B] draft forward —
first-execution MultiBatch code, docs/154 §6 class) rather than at the r1→2 accept arithmetic.
**The per-step probe (B3b/B3c, landed c576d788, not yet RUN) splits this**: it hashes hid_cur at
each draft step entry and the step output — the first diverging step names the op.

---

## 4. INSTRUMENT SUITE (all landed, all NINFER_MB_HASHPT-gated)

Written to the hash FILE (hashpt_A/B.log via NINFER_MB_HASHPT=<path>):

| point | where | what |
|---|---|---|
| B0_seedhid | seeding block (tp2_backend ~:2824) | round-0 prefill hidden per lane |
| B1_pack | round loop, pre-verify | raw fields: plen, win, slt, ringbase, drafts, anchor, F, slot + pack hash |
| B2_verify | post-verify allgather | per-column full-vocab verify-logit hashes |
| B2b_alignin/out | round loop, around the alignment forward | per-(lane,t) input/output hidden hashes |
| B2c_verifyhid | right after target_verify_batch | per-(lane,t) verify-hidden-at-write-time hashes |
| B3_draft | round draft chain entry | per-lane v_arh + v_prop hashes |
| B3b_step | each draft step | per-lane host step-state (pos/rope/valid/win) — NO D2H |
| B3c_stepout | each draft step output | per-lane step-output hidden hash + produced draft id |

To stderr (server.log): `[VLHASH]` per-layer dumps — **CURRENTLY REVERTED** (20479e9f): the
run_layers hook fired in every run_layers(Verify) context including empty-stage-range calls
(1040 -1-only blocks), making per-layer attribution unreliable. **The correct redesign is a
HashTap via the EXISTING Tap::capture_layer hook** — `struct HashTap` is ALREADY LANDED in
text_context.h (+ its impl + the NINFER_MB_HASHPT dispatch in the non-sink target_verify_batch
overload) — a future session only needs to USE it (it is wired: when NINFER_MB_HASHPT is set,
target_verify_batch routes through HashTap).

QKVDBG (`NINFER_MB_QKVDBG_STEP=<step>`, text_context_impl.h gdn_mix_tp): per-GDN-layer
q/k/v/g/beta whole-tensor + **per-slot sub-hashes** (batch is the slowest dim → per-b slices are
contiguous). Whole-tensor hashes are order-sensitive under a lane swap (trivial) — use the
per-slot `b=` lines only.

**CRITICAL INSTRUMENT RULES (each earned the hard way tonight):**
1. **Diagnostic D2H reads MUST be `cudaMemcpyAsync(..., s)` + `cudaStreamSynchronize(s)`** —
   plain cudaMemcpy on the default stream is UNORDERED vs the NonBlocking stream-s work and
   returns STALE data (produced the false "rotation by 2" clue and a false "logits match").
2. **Pair by REQUEST ROLE, never by lane id across orders**: under the order flip, the same lane
   id holds DIFFERENT requests. sampled = A.lane1 vs B.lane0; greedy = A.lane0 vs B.lane1
   (leader = lane 0 = first-fired — verified via the B1_pack plen field). The same-lane-id
   comparison produced three false findings tonight (the "arh differs at round 1", the
   "round-1 logits match", the B3-era pin).
3. **Parse VLHASH blocks on (layer==-1 AND col==0)** — the layer=-1 dump prints one line per
   column; splitting on every layer=-1 line fragments each round into 8 garbage blocks.
4. `getenv("X")` is true for `X=0` — treat the literal "0" as off.

---

## 5. ANALYSIS SCRIPTS + THE EXACT DEVICE COMMANDS

- `results/phase_c/vlhash_layer_analysis.py <server_log_A> <server_log_B>` — per-round per-layer
  divergence with request pairing (the committed version; the `--steps` mode parses B3b/B3c).
- `results/phase_c/vlhash_layer_analysis.py --steps <hashpt_A.log> <hashpt_B.log>` — per-step
  draft-chain analysis (the next capture's analyzer).
- Evidence dirs: `results/phase_c/hashpt_order{,2,3,4}/`, `gdn_audit_notes.txt` (the full
  per-op audit + elimination chain), probe logs `hashpt_order_probe*.log`.

**The device capture commands (the ~6-min window, both orders):**
```bash
cd /home/intel/ninfer/worktrees/wo-kvarn-multibatch
tools/smoke/diag/gpu_guard.sh gpu_refuse_if_busy   # re-guard at claim
nohup tools/smoke/diag/hashpt_order_probe.sh \
    > results/phase_c/hashpt_order_probe_next.log 2>&1 &
# the probe runs BOTH sessions (greedy-leader + sampled-leader) and stops itself;
# ~6 min; then release (nvidia-smi to confirm 15 MiB) and analyze:
python3 results/phase_c/vlhash_layer_analysis.py --steps \
    /tmp/hashpt_<latest>/hashpt_A.log /tmp/hashpt_<latest>/hashpt_B.log
```
The current binary carries: B0/B1_pack(plen)/B2_verify/B2b/B2c/B3/B3b/B3c hash points + the
per-slot QKVDBG extension + the HashTap dispatch (NINFER_MB_HASHPT-gated).

---

## 6. NEXT SESSION'S FIRST MOVES (in order)

0. **FIRST: reconcile the round-2 contradiction** — the committed evidence contains an unresolved
   pair: (a) B1_pack round-2 drafts DIFFER across orders (A.l1: 11316,883,264 vs B.l0:
   13901,883,9561 — verified mapping), which means the round-2 verify embedding MUST differ;
   (b) the refined-VLHASH full-block-1 (= round 2) shows NO divergence. These cannot both be
   true. Likely causes: the VLHASH full-block ↔ round mapping is offset (the empty blocks are
   parser fragments — each vl_dump(-1) prints 8 layer=-1 lines and my parser started a new block
   on every one; the correct block = [(-1,c7) + layers 0..63 + 9999]), or the B1_pack drafts
   print timing differs from the verify input upload. The committed probe data
   (results/phase_c/hashpt_order4/vlhash_server.log + hashpt_order3/ has the B1_pack raw lines)
   is sufficient to resolve this OFFLINE — no device needed.
1. **Then the per-layer capture via HashTap** — the struct/impl/dispatch are LANDED (the
   reverted run_layers hook must NOT be re-added); run with NINFER_MB_HASHPT set and verify the
   [VLHASH] blocks cover layers -1..63+9999 (if the blocks are again fragmented/-1-only, debug
   the HashTap path).
2. **The two-order capture (~6 min)** + the analysis script: pin the first diverging (round,
   layer) with request pairing. The known-true anchors: round-1 draft ids match (B3c), round-1
   draft-chain hiddens differ sub-argmax (B3c), round-2 drafts differ (B1_pack), round-2 accepts
   differ (the position advance +1/+3 vs +3/+0).
3. **Then the per-step B3b/B3c capture** (same window; the probe emits both): the first
   diverging draft STEP names the op (step-input assembly vs mtp_forward_decode_batch forward).
4. **Fix design** at the named op → device-verify (S2 diag exit 0 on bf16/i8 + the accept-count
   probe) → gemini formal test → bf16/i8 S2 re-gating.

**The end state to reach:** `s2_order_diag.sh` exit 0 on int8 AND bf16 (order-invariance
restored), the accept-count probe matching across orders, then S2 re-enabled for bf16/i8.

**Known-true anchors for the analysis (verified, do not re-derive):** round-1 draft ids match
across orders (B3c: greedy 4087,1156; sampled 6587,264); round-1 draft-step hiddens differ
sub-argmax from step 0 (B3c: greedy d13aa079 vs 5b5e8664; sampled cc487b48 vs baa6d033); the
r1→2 accepted counts differ (greedy +1 vs +3; sampled +3 vs +0 — B3b positions); round-2 drafts
differ (B1_pack). Lane mapping verified: leader (first-fired) = lane 0; greedy plen=64, sampled
plen=61.

---

## 7. GOTCHAS HIT TONIGHT (each cost time; do not repeat)

1. `nohup ... &` appended to a `&&` chain backgrounds the WHOLE chain — the build ran but the
   launch never happened (twice), with output lost to a closed pipe. Setup in one foreground
   call; background launch alone in the next.
2. `wait` with no args in a driver waits on the SERVER child (never exits) — wait the curl PIDs
   explicitly.
3. Diagnostic D2Hs: cudaMemcpyAsync on s + sync (rule above).
4. Request-role pairing (rule §4.2).
5. VLHASH block parsing (rule §4.3).
6. fprintf argument order: when adding fields to an existing fprintf, the %specifiers and the
   args MUST stay in sync (the plen/hash swap produced garbage for one probe run).
7. The conv snapshot kernel, delta-net body, and both attention kernels are per-slot pure —
   do not re-audit them for this defect.
8. The serve CI's S2 cell is kvarn-only GATING right now (per coordinator ruling) — bf16/i8 S2
   diffs print the tracked-defect warning. Re-gate after the fix + gemini's formal test.
9. A2's 2b-iii is a NEW file (src/runtime/tp2/dflash2_round.cpp) — fold conflict surface is the
   tp_engine can_batch/deliver region + make_rank's MTP terms + my tp2_backend diagnostic
   insertions (drop-or-keep is free; they are env-gated). A2 notified directly.
10. Diagnostic fprintf arg-order and format-string changes: verify with one test print before
    the device run (the plen field arrived garbage once).

---

## 8. MESH STATE (as of this debrief)

- **Coordinator** (01a07140, intercom): driving the Phase C merge (DONE — main cfca00ca), the
  (vi) hunt, and the A2 card handoffs. Responsive to pushes.
- **A2** (01a07435, intercom): 2b device work — claimed the cards after my release; running the
  §22.31 gates. Fold plan: merge-never-rebase onto cfca00ca; new file dflash2_round.cpp.
- **gemini**: agent_comm mesh (may be down; the coordinator relays). Owns the formal tests
  (TEST SPLIT): the (vi) formal test + the bf16/i8 S2 re-gating test are gemini's.
- The coordinator's overnight order: continuous work, report per commit, evidence committed.
