# MTP acceptance-rate bug — deep dive (handoff doc)

**Goal of this doc:** a self-contained briefing so another agent can diagnose why
MTP acceptance collapses to `a=0` after round 0 in our custom TP2+MTP driver.

Status: everything runs clean (no crashes, coherent output, deterministic). The only
remaining problem is **acceptance**: round 0 gets `a=2`, rounds 1+ get `a=0`. That
makes MTP nearly useless (16.8 t/s vs 32.5 t/s plain TP2 decode — MTP is actually
*slower* because each 4-token verify yields only 1 committed token).

---

## 1. System / what we built

- Engine: **NInfer** (CUDA 13.1, sm_120a, hard-locked to RTX 5060 Ti / Blackwell).
  Repo cloned at `/tmp/ninfer`. Build dir `/tmp/ninfer/build` (CMake+Makefile, not ninja).
- Model: **Qwen3.6-27B groupwise-int**, artifact `/home/intel/models/qwen3_6_27b.ninfer`
  (1124 objects, ~15.6 GB device). Config: hidden=5120, layers=64, intermediate=17408,
  output_rows(vocab)=248320, query_heads=24, kv_heads=4, head_dim=256. GDN (gated delta
  net) config: gdn_k_heads=16, gdn_v_heads=48, k_dim=128, v_dim=128, G=3.
- Topology: **TP2** (tensor-parallel, world=2) across 2× RTX 5060 Ti (PHB, no P2P,
  NCCL 2.30.4 host-staged). Local dims: k_loc=8, v_loc=24, k_rows=1024, v_rows=3072,
  qkv_rows=5120. All-reduce after MLP tail + attention out.
- Driver (the test harness we've been modifying): `/tmp/ninfer/tests/multi_gpu/tp2_decode.cpp`.
  It is a **standalone** program (defines its own `main`, its own variant instantiation,
  does NOT link `ninfer_engine`). It materializes sharded weights per rank and runs a
  hand-rolled MTP round loop.

Build + run:
```
cd /tmp/ninfer/build && make -j24 ninfer_tp2_decode_test
timeout 200 stdbuf -o0 ./tests/ninfer_tp2_decode_test \
  --artifact /home/intel/models/qwen3_6_27b.ninfer --tokens 60 --ctx 4096 \
  --mtp 3 --prompt "The capital of France is"
```
`--mtp 3` = MTP k=3 (draft 3 + 1 bonus). `--mtp 0` = plain decode (reference, works,
coherent). There are also `--noverify` (skip verify, force a=k) and `--fakeaccept`
(run verify, skip accept, force a=k) debug flags.

Relevant recent log: `/tmp/mtp_probe.log`.

---

## 2. What MTP is supposed to do here

Stock NInfer MTP uses a **1-layer MTP head** (artifact present: input_projection
[5120,10240], one fused decoder layer = attention+MLP, norms). The head predicts the
next token given (current token id, current target hidden state). Protocol per round
(k draft tokens):

1. **Bridge**: MTP head consumes (anchor token, seed hidden) → draft d0.
2. **AR steps**: MTP head autoregressively emits d1, d2, d3 (k-1 more).
3. **Verify**: run the full 64-layer target model as a **batched T=k+1 forward**
   over `[anchor, d0, d1, d2]` at positions `[F, F+1, F+2, F+3]`. One weight read,
   batched (this is the whole point — GEMM at T>1, not 4 separate T=1 steps).
4. **Accept** (greedy): `a` = number of leading drafts that equal the target's own
   argmax at that column. `t_star = target[a]` (bonus token). Commit `d0..d_{a-1}`
   plus `t_star`. New anchor = t_star, new frontier F ← F + a + 1.

Correct acceptance is the entire value of MTP. Target: high `a` (the head is trained to
mimic the target). We see a=2 once then a=0 forever.

---

## 3. The exact stock next-round protocol (reference)

File: `src/targets/qwen3_6/impl/runtime/mtp_impl.h` (~lines 100-170). The stock,
**after each accept and before the next round's proposal**, does:

```
mtp_prepare_next_round(...)                     // builds alignment batch
mtp_forward_decode_batch(                       // <-- KEY: runs MTP head over the
    alignment_ids, target_hidden,               //    WHOLE alignment batch (T=k+1)
    cache_positions, rope_positions, valid,     //    writes MTP KV cache at
    kv_rows, envelope, alignment_hidden)        //    cache_positions, emits per-column hidden
speculative_select_accepted_hidden(             // alignment_hidden[a] -> ar_hidden
    alignment_hidden, accepted, ar_hidden)
mtp_propose_batch(ar_hidden, logits, draft0)    // lm_head+argmax on ar_hidden -> d0
<AR steps for d1..d_{k-1}>
```

Critical facts I confirmed by reading the code:
- `mtp_forward_decode_batch` (text_context_impl.h:887) runs the MTP head **stem+tail**
  (including its attention) over a **T=k+1 batch**, writing the MTP head's KV cache at
  `cache_positions` and producing a per-column `alignment_hidden`.
- `mtp_propose_batch` (text_context_impl.h:917) is ONLY `proposal_argmax` =
  lm_head + argmax on the already-computed MTP hidden. **It does NOT run MTP attention.**
- `mtp_prepare_next_round` (kernel in `src/ops/kernel/mtp_round.cuh`) builds
  `alignment_ids[j] = (j<a) ? verify_ids[j+1] : next_anchors`, i.e.
  `[d0, d1, ..., t_star, t_star, ...]` (T=k+1 entries), and sets ar/ar_rope positions
  and next extents.

So the MTP head's attention, in the stock, is executed **once per round over the full
alignment batch** (the committed drafts + t_star), and its KV cache is updated with
those committed tokens. The seed for the next proposal is `alignment_hidden[a]`
= MTP-head-full(t_star, target_hidden[a], at the alignment position).

---

## 4. What our driver does instead (the divergence)

Our driver (`tp2_decode.cpp`, MTP round loop) does a **simplified** protocol:

1. **Bridge**: `mtp_forward_batch(anchor, ar_hidden, base_pos=F, Envelope{F+1,F+1}, ...)`
   — a **single-token** (T=1) MTP head forward for the anchor only. Produces d0
   (lm_head+argmax of column 0). Writes anchor k/v into the MTP KV cache at position F.
2. **AR steps**: `mtp_forward_ar_step(...)` × (k-1) for d1..d_{k-1}.
3. **Verify**: batched T=k+1 target forward (T>1) — this part we built and it is T-general
   and correct (attn T-general, GDN verify via snapshot ops, TP GEMM at T>1 is free).
4. **Accept**: `speculative_accept_greedy_drafts` (greedy path) — correct, matches stock.
5. **Rebase GDN state**: `copy_slot(a, 0)` on the linear-attention state pool (committed
   slot always 0). I verified this is correct: after a T=k+1 verify with
   `initial_state_slots == snapshot_base_slots == [0]`, slot `j` holds the GDN state after
   column j; slot `a` = state at position F+a = exactly the input state the next round's
   anchor needs. So the GDN rebase is NOT the bug.

**The suspected bug:** we replace the stock's
`mtp_prepare_next_round` + `mtp_forward_decode_batch` + `mtp_propose_batch` sequence with a
single `mtp_forward_batch(anchor, ar_hidden)` bridge. Consequences:

- The MTP head's **KV cache is not updated with the current round's committed tokens**
  (`d0..d_{a-1}`, and crucially t_star's MTP-head representation). The stock runs the MTP
  head over the whole alignment batch (writing their k/v); we only ever run it over the
  single anchor. So by round 1 the MTP head's attention context is stale/incomplete.
- Our seed `ar_hidden = verify_hidden[a]` (the raw *target* hidden at position F+a, i.e.
  the hidden that *predicted* t_star) is passed straight to the next bridge's MTP head
  stem. The stock instead uses `alignment_hidden[a]` = the MTP head's **own** output for
  t_star (stem+tail, including its attention), which is a deeper/refined representation.

Both differences mean the round-1+ MTP head input/context differs from the stock, plausibly
enough to break acceptance.

---

## 5. Measured evidence

Prompt "The capital of France is", k=3, ctx 4096. Round-by-round (rank 0):

```
[r0] d0=13  target=[11751 13 271 198]      -> [round 0] a=2  tokens=11751 13 271 0
[r1] d0=314 target=[248068 369 6511 9338]  -> [round 1] a=0  tokens=248068 0 0 0
[r2] d0=271 target=[271 369 6511 248068]   -> [round 2] a=0  tokens=271 0 0 0
[r3] d0=760 target=[248069 369 6511 6511]  -> [round 3] a=0  ...
```

Acceptance kernel (greedy), `speculative_round.cuh:72`:
```
int a = 0;
while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }   // d_a == target[a]
const int t_star = row_targets[a];
// committed = drafts[0..a-1], bonus = t_star; anchor <- t_star; produced = a+1
```
Verify layout, `speculative_round.cuh:18`:
```
verify_ids[0]=anchor; verify_ids[j]=drafts[j-1] (j=1..k)
positions[0]=F; positions[j]=F+j
```
So verify batch = `[anchor, d0, d1, d2]` at `[F, F+1, F+2, F+3]`; `target[a]` = argmax of
the verify logits at column a. Acceptance compares `d_a == target[a]`.

**Probe is now fixed** — it D2H's the raw `st.drafts` data directly (I32 [3,1], ne[0]-fast).
The corrected probe reveals the **smoking gun**:

```
[r0] host_anchor=369   F=4  drafts=[11751 13    248046]  target=[11751 13 271 198]
     arh=[1.484 -2.703 2.719 0.148]  mh=[1.836 -3.734 -0.965 -0.785]
[r1] host_anchor=271   F=7  drafts=[11751 760   314]     target=[248068 369 6511 9338]
     arh=[1.680 -3.844 4.188 -0.004]  mh=[-1.078 -5.875 0.193 -0.539]
[r2] host_anchor=248068 F=8 drafts=[11751 760   271]     target=[271 369 6511 248068]
     arh=[-0.402 -4.625 0.941 -0.801] mh=[-0.340 -0.256 -2.422 0.141]
[r3] host_anchor=...  F=... drafts=[11751 760   760]     target=[248069 369 6511 6511]
```
where `arh` = `ar_hidden[0:4]` (the seed hidden the bridge is given) and `mh` = `st.mh0[0:4]`
(the MTP head's output hidden, before lm_head).

**THE DECISIVE FINDING:**
1. `host_anchor` and `F` change correctly every round (271→248068→... , 4→7→8). The bridge
   is receiving the right anchor.
2. `arh` (seed) **changes** every round.
3. `mh` (the MTP head output) **changes** every round.
4. BUT `drafts[0]` (the MTP head's d0 = argmax(lm_head(mh))) is **frozen at 11751** ("Paris")
   across rounds 1, 2, 3 — the identical integer, even though mh differs.

So the MTP head's final hidden `mh` is a *changing* vector, yet `argmax(lm_head(mh))` always
lands on 11751. This means the MTP head's representation is being **dominated by a stale
attention context** that keeps pushing "Paris" — i.e. the MTP head is NOT actually seeing the
newly-committed tokens (d0, d1, t_star from the previous round) in its KV cache. It is
re-running essentially the *same prompt-context attention* every round and therefore
predicting the *same* next token ("Paris") every round.

Root-cause candidate (to test first): **the MTP head's per-round KV-cache update is not
actually landing.** In the stock, `mtp_forward_decode_batch(alignment_ids, target_hidden,
target_positions, target_rope, licensed_counts, mtp_rows, ...)` re-runs the MTP head over the
committed alignment tokens and writes their k/v into the MTP KV at `target_positions` (the
verify positions). Our driver skips this entirely and relies on the AR steps' k/v writes,
which (a) write the *draft* tokens at the *verify* positions and (b) after a partial accept
(a<k) are the **wrong tokens** at those positions. So the MTP KV beyond the prompt is
stale/garbage, and the MTP head re-attends the prompt context → frozen "Paris".

The robust, trustworthy signals: (a) `a` is 2 in round 0 and 0 thereafter; (b) **d0 is
frozen at 11751** while the target's `target[0]` changes every round.

---

## 6. What has been ruled out (already investigated)

- **Crashes / illegal addresses:** all fixed (GDN snapshot slot OOB, contiguity of
  T>1 slices, NCCL JIT + compute-sanitizer incompat). Runs EXIT=0, deterministic,
  coherent text. Not the current issue.
- **GDN state rebase:** verified `copy_slot(a, 0)` gives the correct committed GDN state
  (slot a = state at position F+a = next anchor's input state). Correct.
- **Verify (T=k+1) target forward:** attn path is T-general; GDN verify uses snapshot ops
  (`gated_delta_net_snapshot`, `causal_conv1d_silu_snapshot`) with per-column slots; TP GEMM
  at T>1 is exact. Round 0 accept proves the verify produces correct target predictions
  (d0==target[0], d1==target[1] in round 0).
- **MTP KV prefill:** I added a prefill that populates the MTP head KV cache with the
  prompt context (positions 0..plen-1) via `mtp_forward_batch(prompt_ids, prompt_hidden,
  positions, Envelope{plen,plen}, ...)`. This made round 0 work (a=2) but did **not** fix
  round 1+ (still a=0). So the prompt-context KV is present; the missing piece is the
  **per-round committed-token MTP KV update** (the stock's `mtp_forward_decode_batch`).
- **Plain decode reference:** `--mtp 0` works and produces coherent output (target model
  itself is fine on this driver).

---

## 7. Most likely root cause (hypothesis to test)

The MTP head is **not being run over the committed tokens of each round**, so its KV cache
and its seed representation drift from the stock's. Two candidate fixes, in order:

1. **Replicate the stock next-round MTP sequence exactly.** After accept, before the next
   bridge, call (mirroring `mtp_impl.h`):
   - `mtp_prepare_next_round` (build `alignment_ids=[d0..d_{a-1}, t_star, t_star, ...]`,
     ar positions, next extents) — kernel exists in `src/ops/kernel/mtp_round.cuh`.
   - `mtp_forward_decode_batch(alignment_ids, target_hidden, cache_positions,
     rope_positions, valid, kv_rows, envelope, alignment_hidden)` — this is the piece we
     currently skip. It updates the MTP KV cache with the committed tokens and yields
     per-column MTP hidden.
   - `speculative_select_accepted_hidden(alignment_hidden, a, ar_hidden)`.
   - Then `mtp_propose_batch(ar_hidden, ...)` for d0 + AR steps (instead of our single
     `mtp_forward_batch` bridge).

   This requires us to pass the **target hidden for the alignment batch**
   (`verify_hidden` [5120, k+1]) and the correct `cache_positions`/`rope_positions`
   (need to confirm: are these the verify positions `[F..F+k]` or the next-round
   positions? See §8 open question Q1).

2. If (1) is too invasive, minimal variant: keep the single bridge but (a) also run the MTP
   head over the committed drafts each round to update the MTP KV, and (b) use the MTP head's
   own output (not raw target hidden) as the next seed.

---

## 8. Open questions for the next agent

- **Q1 (cache_positions for `mtp_forward_decode_batch`): ANSWERED.** The stock call site
  (`mtp_impl.h` ~L148) is:
  ```
  card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions, target_rope,
                                licensed_counts, mtp_rows, envelopes.batch, alignment_hidden);
  ```
  So `cache_positions = target_positions` (the **verify positions** `[F, F+1, ..., F+k]`) and
  `rope_positions = target_rope`. The alignment tokens land at the **verify slots**. So the
  committed token `d_j` is written to the MTP KV at verify position `F+j` — i.e. the MTP head
  re-derives d_j's k/v at the exact position the target verified it. **Our AR steps write the
  drafts at the *AR* positions (also `F+1..F+k`), which coincide positionally, BUT they use the
  DRAFT tokens (and the wrong tokens after a partial accept), and their `hidden`/rope come from
  the MTP head's own chain, not the target hidden.** That positional/token mismatch is the
  prime suspect for the stale MTP KV.
  Also note the stock's `envelopes.batch` (not a single `{F+1,F+1}`) — the MTP head attends
  causally across the whole alignment batch in one call.
- **Q2 (target_hidden source):** confirm the `hidden` arg to `mtp_forward_decode_batch` is the
  verify target hidden `[5120, k+1]` (i.e. our `st.verify_hidden`), column a = hidden at
  position F+a. Is `alignment_hidden[a]` the correct seed (stock uses it), and does that
  correspond to the hidden *after* t_star or the hidden that *predicted* t_star?
- **Q3 (probe fix):** fix the driver's debug probe to D2H raw `st.drafts.data`
  (column-major I32 [3,1]) and raw `st.target_tokens.data` so per-token draft-vs-target
  diffs are trustworthy. Then confirm whether round-1 d0 is "close" (wrong token, plausible
  context) or "garbage" (broken context) — this distinguishes "MTP head just inaccurate"
  from "MTP head got corrupted input."
- **Q4 (MTP KV write in AR steps):** our `mtp_forward_ar_step` — does it write the AR token's
  k/v into the MTP KV cache at the right position each step? If the AR steps don't write MTP
  KV, the MTP head never sees d1..d_{k-1} either. Check `mtp_forward_ar_step` in
  `text_context_impl.h`.

---

## 9. Key file map

| Concern | File | Note |
|---|---|---|
| Driver MTP loop (ours) | `/tmp/ninfer/tests/multi_gpu/tp2_decode.cpp` | the thing to modify; bridge/AR/verify/accept |
| Stock MTP next-round | `src/targets/qwen3_6/impl/runtime/mtp_impl.h` | reference protocol (~L100-170) |
| Accept / prepare kernels | `src/ops/kernel/speculative_round.cuh` | L18 prepare, L72 accept (greedy), L429 select_hidden |
| `mtp_forward_decode_batch` | `src/targets/qwen3_6/impl/runtime/text_context_impl.h:887` | the piece we skip |
| `mtp_propose_batch` | `text_context_impl.h:917` | lm_head+argmax only |
| `mtp_forward_batch` / `_stem` / `_tail` / `_ar_step` | `text_context_impl.h` | what we currently use for the bridge |
| MTP round prep kernel | `src/ops/kernel/mtp_round.cuh` | `mtp_prepare_next_round` |
| TP variant kernels (27b) | `src/targets/qwen3_6_27b/impl/variant_kernels.cpp` | TP attention/MLP/GDN paths |
| TP sharding | `src/core/multi_gpu/tp_kernel.*`, `weight_shard.*`, `tp_group.*` | working, don't touch |

## 10. Performance targets (for context, from doc 05)

| config | expected | measured |
|---|---|---|
| TP2 full split (MTP off) | 36-40 t/s | **32.5 t/s** (working) |
| TP2 + MTP k=3 (fixed) | 70-90 t/s | **16.8 t/s** (a=0 → MTP hurts) |

The whole payoff of MTP is on the line: fix acceptance → expect ~2-3× over the 32.5 t/s
baseline. This is the last big piece before the V340L (gfx900) port.
