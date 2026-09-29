# m3_boot1 — branch (iv): early single-rank spin hang (NEW fourth shape, CORD-amended pre-registration)

Era bundle (rule 6): tree `67b18499`, BIN md5 **2f0d14a14b39578474b300301e79ac22** (prefix
2f0d14a14b39 = C441 GO line match, verified pre-boot; `ninja -n` no-work at tip = no pre/post
ambiguity), body **m2b_10k_a** (curl `-d @.../m2b_10k_a.json` — card.txt line-3 mislabel was
stale template text, FIXED in d58b20ba; datum was always the clean body), slt=[0 2] throughout
(post-static era), trace gate + GDNHASH step<=4 armed, fresh empty guard at claim (both cards
15 MiB, no foreign contexts, port free).

## Timeline (log = local EDT; boot fired 06:44Z)
- warmup bleed at lines 868–872 (`Paged KV allocation is not bound` + `warmup failed
  (continuing)`) — era-consistent, p1r1 lines 865–868 identical pattern. NOT branch (iii)
  (A1's read, confirmed: this is warmup-path bleed, zero_slot family lineage).
- prefill + decode advanced normally: D2SS steps 1..29, drafts per rank printed,
  **both ranks byte-identical through step 29** (`169550 36020 120883 23258 233469`),
  vids/pos/win sane (win=11661 at step 29, 10k fill + 29 rounds).
- ~02:47 local: log frozen at 7798 lines mid-SLICEDBG stream (7563 lines; p1r1 full request
  = 7692 — i.e. near-normal slice volume, stall NOT prefill-side).
- 02:50–03:05: stall held. p1r1's FULL 58-round request took **12.7 s wall**; this request
  was still open at **13+ min**, so the shape is not slow progress — it is a hang.

## Signature (measured via /proc; gdb blocked by yama ptrace_scope=1, no sudo — noted)
- ONE thread spinning: tid 789421, state R, wchan=0, syscall=running, +1000 ticks / 10 s
  = exactly one core at 100% userspace (no syscalls — pure spin, consistent with CUDA
  spin-wait scheduling or host-side poll loop).
- ALL other 20 tasks state S, wchan=futex_wait_queue, cpu_ticks≈0 — including (by topology)
  rank-1's worker: silent, NOT spinning.
- GPU0 100% util / GPU1 0%, both cards held 14.4 GB.
- Zero new error lines after warmup: no deep `not bound`, no bad_alloc, no RANK1 print —
  because at 67b18499 rank-1's death print sat only AFTER `th1.join()` (unreachable while
  rank-0's worker never returned). Fixed-at-root in d58b20ba (immediate fflush'd
  `[D2-SS-RANK1-ERR][immediate]` at catch time).

## Classification against the pre-registered three (+ CORD amendment seq-2)
- (i) drain-cured: REFUTED for acceptance (no curve exists — rounds never reached the 50-print)
  and the ~58-round `not bound` never fired, but the run did not complete. Drain question
  UNRESOLVED — a hang at step ~30 is not evidence the 1 GiB exhaustion is cured.
- (ii) acc near-zero stable: NOT REACHABLE (no death-of-loop datum, hang instead).
- (iii) binding-throw at depth: NOT SEEN (only the era-consistent warmup bleed).
- (iv) **early-barrier-spin**: CONFIRMED as a new shape. First observation at step ~30.
  Two mechanism candidates handed to A1 (see intercom 06:5xZ msg): (a) rank-1 threw
  silently and rank-0 spin-waits a collective with a dead peer (capture hunk names it on
  next boot); (b) genuine step-30 collective race in the twin path. 29×17.5 MiB ≈ 507 MiB
  coincidence to the pre-arena twin-death depth noted and DEMOTED in the same message
  (post-arena scopes rewind per round — cap math doesn't accumulate that way).

## Run-2 discriminator (pre-committed before any re-boot, endorsed CORD seq-2)
Deterministic host-side collective deadlock with the capture hunk landed => run-2 should
hang at the SAME step ± 1 and now PRINT the rank-1 death reason. Async race => different
step or no hang. Either outcome is datum; halt-on-first-divergence stands.

Files: `serve.log` (frozen original), `stall_serve.log` (cp at halt, before teardown),
`STALL_NOTE.txt`, `seeddraw.txt`, `card.txt` (pre-fix template text — kept AS THE ARTIFACT
that documents the mislabel class; corrected form in d58b20ba's template).
