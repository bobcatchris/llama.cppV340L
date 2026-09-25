# HOST-KV SAFETY NET — SESSION DEBRIEF (2026-09-07, sessions 2-3 night push)

Branch: `wo/host-kv-safety-net`, all work pushed to `github` (the `origin`
remote is dead — always push to `github`). Debrief-time HEAD: `6d2a6b71` +
this doc's commit. Verify with `git log --oneline -3` on pickup; the tree
was clean at handoff (only untracked results/ CI logs from another lane). Base: `wo/kvarn-multibatch` → target: `wo/integration-ci` (merge preview §18.16: clean).

---

## 1. MISSION (coordinator decision B, user-directed)

Port the gzenz fork's complete host-KV page-to-host story onto our tree.
Reference: `/tmp/gzenz_ninfer` @ `4b882d0` (read-only; PRESERVE — volatile /tmp).
Scope: docs/156 §18 (derived from the fork's code, not summaries). No v1/v2
framing. Completion bar: the fork's full path (mid-flight-capable continuation
spill, scatter-gather extents, LRU under pressure, 3 concurrent sessions, no
refuse, byte-identical restores, fallbacks=0) — validated by the approved
literal (3×200k @ i8, 12 GiB host arena) + the CI cells (P8).

## 2. WHAT WORKS TODAY (validated)

* Park-on-rotation + restore-with-prefix-skip, single-sequence BF16 arm.
* t2 golden gate: A→B→A' byte-identical (multiple runs, incl. post-refactor).
* t5: graceful park-skip @ 64 MiB arena (all requests survive).
* S2b/S3b/S4b revisits byte-identical under pressure eviction (@ 460 MiB).
* Unit suite 134/134 (canonical-scope run).
* All three §18.10-§18.14 probe/fix infrastructure commits.

Landed phases (§18.3): P1 (page-run arena) ✅, P2 (pool page D2H/H2D
primitives + CUDA unit) ✅, P3 (safety-net upgrade: frontiers, net-level pin,
take/re-add lifecycle) ✅, P4-single-seat (scatter-gather image addressing,
F4 pre-check, trigger re-resolve) ✅, P7 (lifecycle counters → jsonl) ✅,
P8-gate (CI cell) ✅.

## 3. THE OPEN DEFECT (t6, precisely bounded — CONFIRMED POST-PARK WILD WRITE)

**Repro:** 4-session rotation @ `--host-kv-mib 460` (pressure eviction).
S1's revisit DIFFERS (fluent-but-wrong, diverges ~2 tokens in); S2/S3/S4
revisits byte-identical. 10 evictions, 4 restores, 0 failures.

**Reliable full-scope fingerprints (canonical 4096 B probes, post-sync):**
S1b's post-restore state vs S1a's post-prefill state:
  * pool page 0 (l0+l15): ✔ MATCH
  * pool page 1 (l0+l15): ✗ DIFFER — partially expected (the image's page-1
    tail holds the parked session's own decode KV by design; masked by the
    envelope; S2b/S3b/S4b prove masked stale tails are harmless)
  * GDN rec l0 slot 0 / rec l47: ✔ MATCH (92c0467a / e8318fed)
  * GDN conv l47: ✗ DIFFERS from A — and conv l0: ✗ 77b793c7 ≠ A's 5e951f89
  * ph ✔, t0 ✔

**THE CONFIRMED MECHANISM (magic-fill discriminator, t2c run):** the park
magic-fills every LinearConv region with 0xC5 (park hash 32ca757a = FNV(4096
× 0xC5), verified at req2). At req5's restore, the src hash for conv l0 =
77b793c7 ≠ the magic ⇒ **the arena's entry-2 conv region was overwritten
AFTER the park fill** — a post-park wild writer, deterministic content. The
restore faithfully delivered the corrupted bytes (dest == src everywhere);
the decode consumed them; S1 diverged.

**Rank asymmetry clue:** rank 1's parks also magic-filled correctly, but
rank 0's restore delivered 77b793c7 too — both ranks' entry-2 conv regions
read the same wrong content at restore ⇒ the writer is not rank-racy; it
writes the same deterministic bytes into the same logical region on both
ranks.

## 4. THE CORRUPTED CONTENT — WHAT 77b793c7 IS

Unknown, and that is the finding: it matches no tracked component, no
session's conv state, no device dump. It appears only at entry-2's conv
offset at restore time, deterministically. Two structural suspects:
  W1 a later park's capture D2H writing components at entry-2's arena
     offsets (an extent-overlap the allocator accounting missed — the
     carve/split paths are hand-verified but not exhaustively tested);
  W2 the S1b restore's own H2D for a DIFFERENT component overshooting into
     the conv region (byte_view walk or an offset collision) — would explain
     "restore src = arena bytes at its own offset" being innocent while the
     delivered conv is wrong... but the restore src probe reads the arena at
     restore time and returned 77b793c7, i.e. the ARENA itself already held
     the foreign bytes before the H2D ⇒ W1 is the leading theory: a later
     park's capture strayed into entry 2's extent.
The page-fence window decides: entry 2's conv page is fenced (PROT_READ) at
req2's park; if the wild writer strikes before req5's restore, the fence
SIGSEGVs it and the backtrace names the writing instruction.

## 5. WHAT WAS RULED OUT (don't re-derive)

  * Wrong entry restored (tag logging ✔).
  * Allocator extent overlap (offset math on e1/e2/e3/e5/e7 ✔ all disjoint).
  * Wrong entry restored (tag logging ✔).
  * Allocator extent overlap (offset math on e1/e2/e3/e5/e7 ✔ all disjoint).
  * Restore H2D infidelity for the probed components (src==dest ✔ per probe).
  * The conv capture itself being wrong: the capture at req2 read slot 1's
    conv = S1's post-prefill state, and the magic survived in it at park —
    the anomaly appeared LATER (see §3/§4).
  * Probe races (compute stream is NON-BLOCKING; probes now sync first —
    21cb2a50) and cross-scope hash comparisons (canonical 4096 B — 39a39811).
    Both defects produced false conclusions before being fixed; do not trust
    pre-21cb2a50/39a39811 fingerprint analysis.
  * Device-side uninitialized reads as the conv cause (initcheck found 8
    UNRELATED NCCL-allreduce uninit reads — separate pre-existing lane,
    backtraces committed: docs/156_initcheck_backtraces.log; ran PARTIAL).
  * Page-1 stale tail as the divergence cause (S2b/S3b/S4b same shape, pass).
  * Masking the conv as a FIX — coordinator ruling: it hides a memory-safety
    bug (the arena region changes with no tracked writer). Patch staged
    (docs/156_t2_capture_masking.patch), NOT to be applied until the writer
    is named and eliminated.

## 6. THE HUNT PLAN (staged, next session = one window + CPU analysis)

  1. `NINFER_HKV_DBG=1 bash tools/smoke/host_kv_validation_window.sh` —
     the canonical run. EXPECTED (per the magic-fill evidence): entry 2's
     conv src at req5 = 77b793c7 ≠ the magic — the wild writer confirmed, its
     content stable. The fence (step 2) then names the instruction.
  2. tools/ops/host_kv_gdb_window.sh — the page-fence pass: mprotect the
     parked conv page PROT_READ after park; the wild writer SIGSEGVs; gdb
     -batch names the faulting instruction + backtrace. (Caveat: if the write
     is a DMA/copy-engine op, the fault may surface differently — fall back
     to the magic-fill bracketing, which is already proven.)
  3. Fix per findings. If the fix is "the capture grabs the wrong conv
     slot/state": re-target the capture (the correct state = post-prefill
     slot 0's conv at the copy_slot moment — verify what copy_slot(0→1)
     actually copies and when; read linear_attention_state.cpp:201).
  4. Clean t6 (all 4 revisits byte-identical) + t7 + gate → literal
     (tools/smoke/host_kv_literal.sh, ~30-40 min GPU) → docs/155 flip at
     integration (coordinator).

## 7. REMAINING SCOPE AFTER THE DEFECT (§18.3 queue)

  * P4-batched (W1): pressure-triggered owner eviction on the batched arm —
    blocked on the batched arm carrying the net.
  * P6 session keys: soundness-constrained (§18.9 — restore selection must
    gate the divergence; the fork's checkpoint-state-at-frontier design).
    Requires session identity our frontend lacks (§18.8 recon) — a design
    decision, not a mechanical port.
  * W4 scatter-gather extent store (fork HostKVExtentStore semantics) if
    fragmentation demands it beyond the current 2-run fallback.

## 8. INFRASTRUCTURE + TRAPS (each cost real time — don't repeat)

  * BUILD: targeted targets only (`ninfer-serve ninfer_host_kv_arena_test
    ninfer_paged_kv_host_mirror_test`). Never bare `cmake --build .`
    (relinks 8.7G of test binaries). Disk ~7 G free — check df first.
  * The build/ dir was deleted mid-session once (cleanup pass) — reconfigure:
    `cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON`.
  * CARDS: written grant → guard (0 apps) → work → release → verify. A1 and
    gemini share the queue; the coordinator verifies releases. A stale
    server on 8093 from a previous run answers /v1/models — CHECK pgrep
    before trusting readiness (cost a window once).
  * pkill -x ninfer-serve ONLY (never -f: matches your own shell).
  * PROBES: all fingerprints are canonical 4096 B scopes now. NEVER mix
    scopes in a comparison (cost a false bisection conclusion once). The
    compute stream is NON-BLOCKING: sync before reading device state in any
    probe (cost a false "bit-identical" conclusion once).
  * LOG PARSING: the [HKV-STATE] dumps are 7 lines per request but 5 for the
    warmup (1 page) — group alignment errors produced two false analyses.
    Include the req markers in the extraction.
  * A header-only rename once left the tree uncompilable while "landed" —
    verify compile before trusting any landed state.
  * The results/ dir has untracked CI logs from another lane's run — leave
    them; docs/*.log is gitignored (initcheck backtraces needed -f).
  * The completion expectation is CONTINUOUS progress ("agents always working
    toward the goal, land everything today" — the user's standing directive;
    the coordinator re-asserted it). Don't idle waiting for windows: the
    CPU-side queue (P6 recon, P4-batched recon, probe refinement) is in
    docs/156 §18.3/§18.9-§18.10. The session handoff with the full resume
    state: docs/HANDOFF_host_kv_session2.md (+ the 3b/3c addenda at top).
  * PASS/FAIL asymmetry for the hunt: the A/B/A' gate (ONE intervening
    session) PASSES bit-identical; the 4-session rotation (THREE intervening)
    fails at the FIRST revisit. The wild writer's activity scales with
    intervening parks/evictions — or the arena-pressure path (460 MiB forces
    scatter-gather + evictions; 6144 does not) is the trigger. Discriminate
    in the same gdb window: run the rotation @ 6144 first (expect PASS), then
    @ 460 (expect FAIL) — the delta brackets the mechanism.

## 9. KEY FILES

  * src/runtime/tp2/host_kv_parked.{h,cpp} — the image, capture, restore,
    probes, magic-fill (T2c diagnostic, in the binary), extent allocation.
  * src/runtime/tp2/tp2_backend.cpp — HostKvNet (~30-230), the admission
    trigger (~1395-1490), the park/restore probes, the decode-start
    fingerprint dump (~2064+), the page-fence.
  * src/core/host_kv_arena.{h,cpp} — the arena + page-run allocation.
  * src/core/paged_kv_cache.{h,cpp} — the host-mirror primitives (P2).
  * tools/smoke/host_kv_{gate,h3,literal,validation_window}.sh +
    tools/ops/host_kv_{gate_ci,gdb_window}.sh — the drivers.
  * docs/156 §18.1-§18.17 — the full trail. §18.10 has the canonical
    per-request fingerprint table. §18.13 the scope-error correction.
EOF

---

## ADDENDUM (agent2, same day, CPU-only session — ROOT CAUSE NAMED, §18.18)

The §3/§4 wild-write defect is SOLVED without a window: it is our own capture
D2H overshooting a scatter run boundary (layer-6 K-page copy, 47104 bytes into
entry 2's conv head). W1's "later park's capture strayed into entry 2" was
right in kind; the stray is a RUN-BOUNDARY STRADDLE, not extent overlap —
extents were disjoint, the COPY wasn't. Fix landed (for_each_span split-copies,
capture+restore+memset; AllocGuard leak fixed; straddle unit test green).
See docs/156 §18.18 for the full evidence chain. Debrief corrections: hash
quotes are HIGH-32; the "rank 0 park-time zeros" line is moot (t2c log was
overwritten; the surviving 14:48 log shows magic intact at every park print);
and §3's "both ranks' entry-2 conv regions read the same wrong content ⇒ the
writer is not rank-racy" is WRONG — the restore-time contents DIFFER per rank
(rank0 77b793c753a7f890 vs rank1 669ce06f78e3d188 — §18.15's "third distinct
content" resolved) because the wild bytes are rank-sharded K-page payload;
only the WRITER and the STOMPED RANGE are rank-independent.
The staged window now VALIDATES the fix (expect all-four-revisits clean);
the gdb hunt is demoted to contingency. Masking patch stays dead.
