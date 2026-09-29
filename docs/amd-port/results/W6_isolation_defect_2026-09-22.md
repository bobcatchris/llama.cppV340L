# W6-isolation defect E-038: TP2@10k "Context size has been exceeded. off = 69"

Date: 2026-09-22. Desk: wt-draft-isolation. Route-back from the TP2 desk's
served verification (E-037): the E-035 full-isolation build failed the
TP2@10k decode cell - a 7857-token request exhausted 28 KV-space retries
(down to n_batch=1) and returned HTTP 500, while the v1 flag build on "the
identical request/geometry" had zero retries. Coordinator hypothesis:
dual-cache cell-accounting aliasing (the duplicated token_embd/output
tensors interacting with KV cell accounting).

## Verdict

REFUTED - there is no cell-accounting defect in the isolation build, and no
aliasing exists to fix. The failing run served TWO CONCURRENT 7857-token
requests (plus a 3122 and a 32-token request) on one 10240-cell unified KV
cache: demand 15714 cells vs 10240 supplied. The failure point, the retry
count and the "off = 69" signature all close to the exact cell against pure
capacity arithmetic. The v1 comparison was clean-serial vs contested-port -
not the same geometry.

## Evidence (tp2feas_t2flag10k4_20260921_213414_server.log, TP2 worktree)

1. The battery posts exactly TWO completion requests (guard_battery.py main:
   prefill_guard 3122 tokens, then decode_guard 7857 tokens, serially, 300 s
   timeout; battery sha 4f14d959 identical in the v1 run). The failing server
   log shows FIVE "new prompt" events:

       task  0  3122 tok  slot 3  (battery prefill guard, PASS)
       task 14  7857 tok  slot 2  (battery decode guard, posted 0.53s)
       task 20  3122 tok  slot 1  (FOREIGN - posted 1.19s, mid-task-14)
       task 35  7857 tok  slot 0  (FOREIGN - posted 3.02s, mid-task-14)
       task 46    32 tok           (FOREIGN smoke)

   Tasks 20/35/46 match the TP3 A/B lane's guard sizes; that lane's server
   was being free_port-killed by the port contention documented in E-037
   (hub #1343-#1345), so its clients landed on whichever server owned 8080.

2. Timeline: task 14 progressed to 6075 cells; task 35 (slot 0 - first in the
   server's slot fill order) then took the whole 512-token batch every round
   and reached 3584; task 14 starved (no progress lines after 3.02s). At the
   next chunk the attention cache refused placement: "init_batch: failed to
   prepare attention ubatches" - the cache was full.

3. Exact capacity closure (host-tested in test_w6_isolation_host.cpp):

       10240-cell unified cache (n_parallel=auto -> 4, kv_unified=true)
       task 14 placed           6075 + 512 in-flight chunk =  6587
       task 35 placed     3584 + 64 + 4 + 1 trickle       =  3653
       total                                            = 10240 (zero free)

   The retry storms sit at batch offsets 0, 64, 68 and finally 69 - the
   trickle that squeezed 69 tokens into the last 69 free cells (batch
   offsets, not KV cell offsets). At nb=1 with zero cells free the server
   throws "Context size has been exceeded. off = 69" - reproduced number for
   number. The "28 retries" = the counted "retrying with smaller batch size"
   lines across the four storms.

4. Same build, serial load, same request: run 10k3 (21:25, killed externally
   at t+120 s by the TP3 lane's free_port) served the identical 7857-token
   decode-guard request with ZERO retries through cached 5120 cells when the
   log ends. The isolation build has no cell-accounting failure mode on
   uncontended load.

5. No aliasing exists to fix (code-audited): create_memory for
   mtp_on_hybrid_qwen35 builds a plain llama_kv_cache with mem_other =
   nullptr (src/llama-model.cpp, the only mem_other consumer in the whole
   switch is the GEMMA4_ASSISTANT iswa path). The draft cache is a separate
   cells object; nothing in the duplicated-token_embd/output path touches
   llama_kv_cells. The duplicated tensors are weight buffers - they never
   enter cell arithmetic.

## Residual exposures this incident DID surface (build-independent)

1. The server admits slots by per-slot n_ctx without checking GLOBAL
   unified-cache occupancy: with kv_unified=true and n_parallel=4, two
   slots each believing they own 10240 cells overcommit the shared array and
   fail mid-prompt with HTTP 500 instead of queueing. Stock server behavior,
   identical in v1; out of this desk's scope - flagged for the coordinator.
2. Slot starvation: slot fill order (slots[0] first) lets a newer request
   monopolize the per-round batch while an older mid-prompt request freezes.
   Also stock behavior, build-independent.
3. The draft-cache independence invariant was implicit. Now logged.

## Hardening shipped (this desk)

1. tools/server/server-context.cpp: on --spec-mtp-device boots the server
   now logs "MTP draft cache: independent cells (not shared with the target)"
   (or SHARED, loudly) right after the isolation audit line - a shared-cache
   misconfiguration would have refuted the aliasing question in one glance
   at the boot log.
2. docs/amd-port/tests/test_w6_isolation_host.cpp extended: the cache
   independence rule table (the ONLY cells-sharing path is the
   GEMMA4_ASSISTANT iswa mem_other wiring; qwen35 MTP is NOT it) and the
   incident capacity arithmetic (10240 = 6587 + 3653, trickle 64+4+1, two
   x 7857 > 10240) as permanent regression documentation. ALL PASS; the
   predecessor W6 suites still ALL PASS; llama-server gfx900 rebuild clean
   (zero warnings).

## What the isolation build did NOT cost

Re-confirmed from the same logs: prefill_guard PASSED at 88.74/88.91 t/s on
the iso build (v1 87.61/87.68 - noise); duplication exact (+854.5 MiB);
audit gate 0.00 MiB residual. The decode cell was never cleanly measured on
the iso build (10k3 killed externally, 10k4 contaminated) - v1's 14.43 t/s /
0.66667 stands as the arm reference until a clean window.

## Re-verification request (via coordinator)

One TP2@10k decode-cell window for the iso build with CLIENT exclusivity,
not just boot exclusivity: the boot lock covers server boots, but foreign
clients on port 8080 caused this incident. Options (coordinator's pick):
(a) run the iso verification on a distinct port (e.g. 8081) so stray TP3-lane
clients cannot land on it, or (b) a hub GO handshake that the TP3 lane's
harness is quiescent for the window. Serial battery (the standard
guard_battery flow is serial already - no concurrent curls). Expected:
zero retries, accept ~0.66667, decode in the v1 14.43 t/s class (the draft
now runs a single die2 split - the honest estimate remains the E-034 range
around v1 +/- a few percent; the A/B decides promotion).
