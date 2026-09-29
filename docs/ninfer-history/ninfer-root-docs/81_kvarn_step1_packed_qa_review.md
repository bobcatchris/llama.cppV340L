# 81 — KVarN docs/78 step 1: QA review of f72df497 (packed decode kernel) — **BLOCKED**

Date: 2026-08-26. Reviewer: QA (static review; no GPU runs — server occupied by the
kernel agent). Subject: wo/kvarn-hold @ `f72df497` "doc78: packed decode —
warp-specialized MMA + cp.async K staging (80KB smem)" plus **uncommitted
worktree changes** (see F2).

## Verdict

**BLOCKED.** The kernel is unverified (no results committed, no A/B numbers),
contains three correctness bugs found by static review (F3–F5), and the commit
violates the G1 protocol in QA_FEEDBACK_FOR_AGENT.txt (F8). Do not proceed to
step 2. Fix + evidence list in §4.

What is **correct** (credit where due): the numeric conventions are consistent
with the split baseline — F = normalized FWHT is an involution (F²=I), so
Q_s = F(Q), page K_s = s_row·F(K) give score = F(Q)·(s_row·F(K)) = s_row·(Q·K)
after ×scale(1/√D), exactly matching the split kernel's
`s_row·(q_rot·kT)·scale` with q_rot = F(Q); page V_s = F(F(V)) = V matches the
split kernel's `kvarn_dequant_v`+`kvarn_fwht_channel` = V; the
`gqa_prefill_swz` stage/read pairing is the same verified prefill pairing
(gqa_attention_prefill_common.cuh:78); m/l/acc partial ordering, merge kernel
reuse, and (sp,kv,tq,g) slot indexing all match the split kernel. The bench
A/B harness (compare_split_vs_packed, byte compare tol 0, max-abs/rel
tracking) is sound infrastructure.

## Findings

### F1 — No build/run evidence committed
Step 0 pattern: `results/doc78_step0_baseline.md` committed with the numbers.
Step 1 commit `f72df497` contains **no results/ file** and no A/B output. The
commit message's "Measured: 34.15 GB/s (up from 25-30)" is unverifiable from
the tree. A/B result (mismatch count) is nowhere.

### F2 — Dirty worktree: MMA-only stub, dequant removed
Uncommitted diff vs f72df497 (1 file, +7/−64): the real `dequant_page` and
`prefetch_page` (committed lines 168, 224) are replaced by

- `dequant_page` → `k_s[i] = v_s[i] = 0` memset ("MEASURE: MMA-only — replace
  dequant with memset of k_s/v_s"),
- `prefetch_page` → no-op ("cp.async issue lambda: no-op for MMA-only
  measurement").

In the current tree state the kernel **reads zero code bytes and computes
nothing** (scores ≡ 0, output ≡ 0). Consequences:
- Any bench timing in this state is an MMA-pipeline microbenchmark; the
  "effective code read = 21.12 MB / time" metric is meaningless (the 21.12 MB
  is never read). If 34.15 GB/s comes from this state, it is an artifact.
- Any byte A/B in this state fails 100% (packed out = all zeros).
- Either way the state must not be committed as "the kernel".

### F3 (critical) — In-loop K prefetch misses 25% of the page
Committed `prefetch_page` (packed.inc:224): `for (int off = tid; off < 512;
off += 128)` executed by whichever threads call it. Prologue (line 238) runs
with all 128 threads → full coverage of the 8 KB page ✓. Loop (line 246) runs
only under `if (is_deq)` → warps 1–3, tid ∈ [32,127] → off ∈
[32,127]∪[160,255]∪[288,383]∪[416,511]; **off ∈ [0,31]∪[128,159]∪[256,287]∪
[384,415] are never issued** (128 of 512 slots = 2048 B = 25% of each K page).
For every page lp0+1 onward those K code slots keep the previous page's bytes
→ wrong k_s → wrong scores. The comment ("96 threads, each does ceil(512/96) =
6 ops") does not match the code (stride-128 loop; each participating thread
does 4 ops).

### F4 (critical) — cp.async: dequant reads stage_k before its wait
Loop order (packed.inc:246, 376, 380): (1) `prefetch_page(lp+1)` issues
cp.async **into stage_k**; (2) warp 0 MMA; (3) warps 1–3 `dequant_page(lp)`
**reads stage_k** — the very buffer being overwritten by the in-flight copy
issued in (1); (4) `cp_async_wait(0)`. The read at (3) is not ordered with
respect to the async copy, so dequant consumes partially-written (or stale)
bytes. The intended one-stage-ahead pipeline (dequant(lp) consumes page lp+1
for the next MMA) is unsound as written: the wait for lp+1 happens after the
read. (Even under the "consume lp" interpretation, the buffer is already being
overwritten by (1).)

### F5 (critical for production) — Tail K staged in the wrong domain
Page K tile: k_s = s_row·F(K) (transformed domain; dequant at committed line
168 has no second FWHT, norm 1.0). Tail: `kvarn_mma_stage_tail_k` (packed.inc:388)
stages **raw original-domain K** (plain bf16 copy — same helper prefill uses
where k_s is also original-domain, but here the page tile is F-domain). Tail
score = F(Q)·K_tail ≠ Q·K_tail. The bench always passes `tail_count=0`
(compare_split_vs_packed + time_decode), so this is untested; **every
beyond-wall decode step in production has a non-empty tail** → wrong attention
→ wrong logits if enabled. The split kernel's tail (decode_split.inc:178–236)
does the correct thing: original-domain direct dot of raw q × raw tail_k.
V side is consistent (page v_s = V, tail v_s = V — no transform needed).

### F6 (moderate) — Launcher smem mismatch
Launcher (gqa_attention_kvarn.cu:161) sets `kPackedSmem = 3·64·256·2 = 96 KB`
(stale comment implies q_s = 64 rows); kernel layout and bench use 80 KB
(q_s 16 rows = 8 KB + k_s 32 + v_s 32 + stage_k 8, packed.inc:73–84). Launches
today (99 KB opt-in) but is wrong; fix to 80 KB.

### F7 (moderate) — Bench scale layout mismatch
Bench buffer: scales[head][page][1152]. Kernel indexing (both kernels, via
`kvarn_scale_at`: table[layer + n_layers·(head + n_heads·(page + n_pages·
field))]): with n_layers=1, n_heads=2, n_pages_scale=625 the element base is
h + 2·phys + 1250·field — a different permutation. Harmless for the A/B (both
kernels see identical inputs) and for perf, but the bench's "dequantized"
tiles are not real (code, scale) pairs. Add a real-data A/B (codes + scales
from the quantizer path, e.g. via kvarn_quantize_head_k/v on synthetic tiles)
before the gate.

### F8 — G1 protocol violation
QA_FEEDBACK_FOR_AGENT.txt (current): G1 = effective packed-code read ≥150 GB/s
at N=40k AND N=250k, 3-run median, step-0 harness; "If G1 misses -> STOP,
follow docs/78 Rollback, **do not ship a partial kernel** and do not continue
to step 2." The commit reports 34.15 GB/s (4.4× short, agent's own words) yet
shipped a partial kernel (buggy, unverified — F3–F5, F1) and continued
iterating (F2). The protocol requires STOP + rollback + report.

## Requirements before re-review (in order)

1. **Revert the worktree to f72df497** (discard the stub). If an MMA-only
   microbenchmark is wanted, it is a separate labeled tool/commit, never "the
   kernel".
2. **Fix F3** — full 512-slot coverage per page in the loop (e.g. run the
   prefetch from all 128 threads via a separate predicate, or a coverage loop
   that actually covers 512 with 96 threads).
3. **Fix F4** — sound pipeline: 2-stage lookahead (issue lp+2 in iteration
   lp, wait for lp+1 before dequant reads it), or equivalent with explicit
   per-group waits. No read of stage_k without a covering wait.
4. **Fix F5** — stage tail K in the F domain (normalized FWHT of each tail key
   row before staging) or handle the tail with an original-domain direct dot
   like the split kernel.
5. **Fix F6** — launcher smem → 80 KB (match kernel + bench).
6. **Extend bench**: (a) tail A/B case (tail_count = 16 with real staged
   tail_k/v); (b) real-data A/B (F7). Keep byte tol 0.
7. **Run + commit evidence**: step-0 harness (3-run median, N=40k AND N=250k,
   T=1) for pass-1/pass-2/effective GB/s **and** A/B results (mismatch count,
   max abs/rel, incl. tail + real-data cases) → `results/doc78_step1_packed.md`
   in the branch, same pattern as step 0.
8. QA reviews the numbers. G1 still ≥150 required; if missed → STOP +
   docs/78 Rollback (reconsider design — e.g. L2-resident direct MMA without
   smem staging — per docs/78 step 1 design space). No step 2 regardless.
