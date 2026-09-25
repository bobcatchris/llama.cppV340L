# (iii) STATIC VERDICT — content paths to the rank-1-only exit are EXHAUSTED; one residual class survives: per-device divergence of the accept kernel itself

Method: enumerate EVERY input to the round-tail decision chain
(lic_h -> toks -> hit_stop/gen_before -> done -> while-condition) and classify
each as REPLICATED (same bytes both ranks by construction), COMPLEMENTARY
(different by shard design, merged consistently by a cleared handshake), or
OPEN. Cleared-handshake = argmax (272+112 K-lines zero passes incl. at-hang)
and AR (87+58 KAR-lines zero REJECTs on both routes) — the one-shot family is
exonerated as a CLASS (CORD seq-85).

## Enumeration (all cites at tree 7cf54433-era, src f2da5e8a+d949 lineage)
| input to tail | class | why |
|---|---|---|
| d2_vids/h_vids (verify ids) | REPLICATED | anchor+chain_drafts; drafts PRINTED byte-identical through r12 |
| d2_vpos/h_vpos, h_win | REPLICATED | cur_F/plen host-replicated |
| d2_slt2 [0,ring0] | REPLICATED | constants (AB2 sub-check 2) |
| text KV pool window (kvarn) | COMPLEMENTARY-consistent | head-sharded; attend out via AR (buried both routes) |
| GDN state slots | COMPLEMENTARY-consistent | shard-local; fed by allreduce'd inputs (AB2 shape-clean) |
| pending_features (sink) | COMPLEMENTARY-consistent | from d2_hid (AR'd per layer); fuse kernel deterministic local |
| d2_log (local shard logits) | COMPLEMENTARY BY DESIGN | vocab halves; merged by argmax (CLOSED) |
| mb_vfull (verify logits) | REPLICATED post-allgather | NCCL allgather, constant counts (door (ii) demote) |
| mb_tgt_rm (global tokens) | REPLICATED | argmax output (CLOSED class) |
| mb_drafts/ext/len/anch/scfg | REPLICATED | host vectors from replicated state; scfg identical |
| gen_before | REPLICATED (per-rank copy, same increments) | += toks.size(); toks from lic_h |
| **a = acc_h[0], lic_h[0..a]** | **OPEN** | computed by the ACCEPT KERNEL on two DIFFERENT GPUs from the (identical) inputs above |

## The residual: identical inputs, two devices, one output assumed
Everything feeding ops::speculative_accept_greedy_drafts is REPLICATED as of the
handshake closures — yet its OUTPUT (a, licensed tokens) is computed INDEPENDENTLY
on each GPU and consumed as if shared. Two sub-shapes survive:
(A) UNINIT-READ inside the accept path: st.work scratch / envelope regions beyond
    written bytes (the ext_now>0 envelope is read on a>0 rounds — which occur
    FROM ROUND ~1 in every capture (vpos deltas 2,3,3,4,5,3,6,3,4,2 — checked,
    correcting my own earlier 'first a>0 at 12' draft claim before it shipped);
    so the per-round divergence coin is flipped ~12 times before the hang, and
    step-13 is 'first hang after enough flips', NOT a special-content round.)
    Garbage differs per device -> a differs -> toks diverge -> hit_stop
    rank-1-only -> the silent clean exit, with NO collective involved (fits:
    rank-1 exits BEFORE round 13's first collective; rank-0's round-13 stall is
    the CONSEQUENCE, matching the D2H-block frame behind stream work whose
    participant is gone).
(B) DEVICE NONDETERMINISM on identical bytes (fp reassociation/order-dependent
    reduce in the greedy branch) — same divergence shape, different fix.
Why m6/battery PASSes don't refute: divergence needs the garbage to differ in
the COMPARE WINDOW (a's boundary) — a coin per (device-state, alloc-history);
8P/3H fits. The alloc-history angle also explains the script correlation
(p1ab3 vs p1bt differ in what touched the arena/work buffers pre-request —
TIMING was the wrong axis; ALLOC HISTORY is the right one).

## Decisive instrument (boot-8-shaped, NOT built, awaiting CORD stamp + A1 compare)
Per-rank tail dump, one printf at the existing sync point (:3859 post-streamSync):
[rank N] D2SS-TAIL step=K a=A lic=[t0..tA] gen_before=G — 4 lines/round, both
ranks. Verdict metric is mechanical: DIFFERENT a at the same step = accept-kernel
divergence NAMED (then (A) vs (B) splits by an mb_lic D2H re-read or memset-
poison delta); IDENTICAL a everywhere + hang persists = the exit theory itself
is WRONG and the hunt moves to thread-lifecycle (rank-1 vanishing without
return/throw = pthread-level event — much deeper, much rarer).
Cheapest (A)-probe first: ONE-TIME cudaMemset(mb_lic/mb_acc/st.work envelope,
0x5A) at request start under an env — if the hang RATE moves, uninit-read is
confirmed without any per-round prints.

## Compare-before-concluding note (A1)
This table is the enumeration; your pool-window/tap-feed surfaces are SUBSUMED
by rows 4-6 (they were content-divergence theories; the handshake closures +
replicated-inputs table make content divergence impossible EXCEPT post-kernel).
If your read finds a row I misclassified as REPLICATED, that's the correction
that matters — the table is the claim.
