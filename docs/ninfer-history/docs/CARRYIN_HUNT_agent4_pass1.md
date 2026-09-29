# Carry-in hunt, agent4 reading pass #1 — what statics EXONERATE, and the one door left open

Datum under test (agent3's G-AMD-27, canonical 1d0ff3c6): ONE server, identical binary/prompt/flags,
greedy, 5 sequential requests → 5/5 DISTINCT outputs, first-fork char 37–107, two forks carry the
just-emitted-token repetition shape. Their attribution: request-4-only REJECTs ⇒ upstream of the AR
window; S1 monotonic-epoch NOT falsified (all 327 KAR tuples observed==expected).

## CLOSED by this desk at static (files @ lane tip f96b729a; predicate: re-grep before quoting)
1. **Sampler arrival order (agent3 §3 candidate) — EXONERATED for the token.**
   `sampling.cuh:168-201` greedy route = pure `unsigned long long` max-reduce over
   `partial_keys` — max is commutative AND associative, so `group_done` arrival order cannot
   perturb `out[col]`. The counter hygiene is also closed on inspection: `sampling_partial_topk_kernel`
   zeroes `group_done[col]` at (partial==0, tid==0) — partials launch PRECEDES the group kernel
   on the same stream, so every column's counter is initialized in-stream before any
   `atomicAdd` (:99 + :183/:230), and the finalize path also self-resets (:197/:276).
   A stale-carried counter is structurally impossible on this route. **The whole (i) family
   lives in the split-k COMBINE, and the only split-k reduces in this tree are fixed-order
   shared-memory trees:** `w8_rowsplit_gemm_medium_t_splitk.cuh:168-212` combines splits by
   `k_split` parity in static loop order (`for split=2; split<KSplits; split+=2`), no global
   atomics anywhere in `src/ops/linear/` (grep: the four atomicAdd files are dflash2 diag,
   sampler counters, spec-round counters, moe diag — all INT counters or diag, never a logit
   value). **No order-varying float combine exists on the logits producer path at world=2.**
2. **The cross-rank token pick itself — deterministic by construction.**
   `one_shot_argmax.cu:150-230`: local winner via fixed butterfly (`ov > max_v || (ov == max_v &&
   oi < max_idx)` — tie picks LOW idx, deterministic); payload `st_volatile → fence → flag=epoch →
   poll(peer>=epoch) → ld_volatile → max+min-idx combine`; `emit_conf` combines in the named
   rank-ascending fixed order. The poll only DELAYS, never reorders: by the time a rank reads
   `peer_host_payload[t]`, the peer's fence proves the payload precedes its flag. Soft-fail timeout
   path returns BEFORE writing out_token (stale slot) and throws at next entry — zero such lines
   in the G-AMD-27 corpus (agent3 §2 falsifier), so the stale-token branch never fired.
3. **Warmup/request slot crosstalk: none visible.** `reset_step` + monotonic epoch per call,
   slot = step % 32; T<=16, 5×32-token requests + 4 warmups = 164 steps < 32-slot wrap per TOKEN
   column t (t resets per call), no wrap reached inside the boot (step is global — wrap needs
   32 CALLS, the boot made 164... **CORRECTION while writing, this one is OPEN**: 164 calls DO
   wrap the 32-slot ring many times; payload slot `step%32` is rewritten every 32 calls — if a
   HOST reader (conf_from_payloads_at) lags a full ring, it reads a recycled slot. That is a
   READ-SIDE telemetry hazard, NOT the token path (out_token is written in-kernel per call), so
   it cannot explain the forks — but agent3's step↔request mapping reconstruction inherits the
   wrap caveat for any retro read of KAR payloads beyond 32 calls back. Named, bounded, not a
   suspect for (iii); a census row for their instrument re-use.

## The one door left open — per-request STATE, not per-argv compute
Every serve line in the corpus reads `reuse=full_reset` (and `cache=0`). The compute above is
deterministic GIVEN its inputs; the fork must therefore live in inputs that full_reset does NOT
guarantee equal across requests. Candidates ordered by this desk's reading:
  a. **GDN linear-state slot recycling.** Pool zeroed ONCE at rank construction
     (`tp2_backend.cpp:812` memsetAsync at make_rank — stable-within-boot, varies-between-boots:
     EXACTLY the two-level signature: cross-boot attractor variation AND per-request drift, if a
     later request reads a slot that an earlier request's decode wrote and full_reset restores
     from the *committed/park* view rather than zero. `set_linear_state_slots(0,1)` (:1936/:2162/
     :3547) selects slot INDICES per step — the selection is static; the VALUE zeroing per new
     session is the unread line. **Next read target: the park/restore path
     (`host_kv_parked.*`, `text_context` session-init) — does a brand-new session's committed GDN
     slot get memset, restored-from-host, or trusted-as-already-zero?**
  b. **KV ring rewind residue** (`paged_kv_cache.cpp:183-186` zero-paths on page free/alloc —
     check which of the two full_reset takes; a rewind-without-zero that later ops fully
     overwrite would be innocent; the census should table every read of a KV page with
     generation < current request's write set).
  c. conv-state window at first chunk (conv_width 3, `valid_columns`/`extents`/`state_slots`
     I32s set per :1200 — only 2 ints because `state_slots` IS a pair of slot indices (verified
     consumer `gdn_projected_conv.cu:31` reads `initial_state_slots[batch]` as an INDEX — this
     desk's own earlier `sizeof*2 vs slot_count` suspicion REFUTED at the consumer; recorded as
     self-caught).

## What would PROBE it (design note only — NO boot without the written stamp)
Discriminating arm (their §5 thread): `NINFER_GDN_SLOTZERO=1` env — at session-init, force-memset
the new session's committed GDN slots + conv window before first step (5 lines at the a- site).
If forks vanish across a repeated 5× cell on the SAME boot: carry-in proven, mechanism located.
If they persist: (a) dead, next door is (b). Pair with the zero-card census from agent1 (WO-SUPPORT-1)
— their files:line table decides which arm is cheapest. Instrument law: the probe must run on the
BANKED era bin `8e6d79ae…` ONLY via a build-with-flag OFF default (additive env-gate, unset =
byte-identical behavior IS the precedent at :733 ws-override; a rebuilt bin breaks the identical-
binary method — so the probe boot carries a NEW era pin, comparable to 17g only at OUTPUT-CLASS
level, and the row must say so). Until then: reading continues; nothing boots.
