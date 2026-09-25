# 156 — Host-KV safety net (`--host-kv-mib`, `--host-state-slots`): Agent Work Order

**Status:** CURRENT — work order for agent A2 (session of 2026-09-06/07).
**Mission:** long-context and multi-session serving must survive device-KV
pressure by parking continuation state (KVarN committed pages + GDN checkpoint
+ token ledger) into a pinned host arena at request completion, and restore it
H2D with prefill-skip when a prompt prefix-matches — no OOM, no owners_evicted,
prefill skipped for the matched prefix. End state: `--host-kv-mib` on → a
stress test (3 concurrent 200k-token sessions) completes with
`host_kv_occupied > 0`, `maximal_fallbacks = 0`, and correct outputs.

Read this document fully before writing code.

---

## 1. Context (60-second version)

gzenz's fork (`/tmp/gzenz_ninfer`, tip 4b882d0, read-only reference) ships a
host-KV safety net: when device KV is full, evicted continuations are spilled
to a pinned host arena (D2H) and restored (H2D) on prefix match, with
smallest-first LRU eviction and pinned-entry protection. Verified upstream
across 140+ requests with 3 concurrent 330k–470k sessions (host_kv_occupied
2.81 GB, zero fallbacks/evictions). Our triage (docs/155 §5, item 3): absent
here, "needs an eviction story in our pool; big".

**The architectural difference that scopes this work order:** the fork's
substrate is a shared paged KV (page tables, logical extents, COW) — their
safety net parks EVICTED PAGE EXTENTS. Ours is per-LANE KVarN workspaces
(`pages_per_lane`, committed-page counters `kvarn_lpp_dev`/`kvarn_mtp_lpp_dev`,
tp2_backend.h:76-77): a lane's committed KV prefix is a CONTIGUOUS extent of
its private ring. We therefore do NOT need their scatter-gather extent store
for v1 — a parked continuation is one contiguous byte range per side (K/V
planes) plus the GDN checkpoint state (docs/70's prefix-reuse checkpoint
buffer, already built) plus the token ledger. The fork's arena/handle
machinery is worth porting only if v1 outgrows contiguous extents.

**What already exists and must be reused (do not rebuild):**
- Per-request prefix matching: tp2_backend.cpp ~1195 (`prefix_len` loop over
  `st.cached_tokens`) — the same match drives the restore-skip.
- GDN checkpoint buffer (docs/70, 73.4 MiB/rank, `gdn_ckpt` path ~1374) — the
  recurrent-state image the fork calls `state_host`.
- KVarN lane workspaces + committed-page counters (tp2_backend.h:76-77).
- Admission budget: tp2_budget.h (Budget/preflight) — parking must interact
  with the SAME admission gate, not a parallel one.

**What you are doing:** §6 steps H1..H4.

## 2. Environment & build/test

- Worktree: `/home/intel/ninfer/worktrees/wo-host-kv`, branch
  `wo/host-kv-safety-net`, base `wo/kvarn-multibatch` (NOT main).
- Build: `cmake --build build -j 16` (own build dir; first build is full).
- Tests: `/usr/bin/ctest` (NOT the ctest on PATH). Server tests through
  `bash tools/ops/run_ci.sh` from the worktree root (relative path).
- GPU work needs a written coordinator grant + `nvidia-smi` guard at claim
  (standing rule); H1 is CPU-only.
- Reference fork (read-only): `/tmp/gzenz_ninfer`
  (`host_kv_arena.{h,cpp}`, `host_kv_safety_net.h`,
  docs/maintainer/resource-scheduling-and-context-cache.md).

## 3. Architecture facts (verified 2026-09-06 — do not re-derive)

- A lane's KVarN workspace is a private ring: committed prefix =
  `[0, committed_pages)` contiguous per side; `kvarn_lpp_dev`/`kvarn_mtp_lpp_dev`
  hold the committed-page counts the attend kernels read.
- The per-request "prefix cache" (`st.cached_ph`, `st.cached_tokens`,
  `st.cache_valid`) is SAME-REQUEST ONLY (re-prefill skip for one request's
  own state); there is NO cross-request KV retention in our tree today. A
  finished request's KV is simply released.
- Admission: `Budget` (tp2_budget.h) preflight + per-request gates; when
  capacity is exceeded the request is refused or downgraded — there is no
  park/restore path.
- The fork's HostKVArena is substrate-agnostic above the page layout
  (`plan_host_kv_page_layout(KVPageGeometry)`) but its allocation handles are
  scatter-gather because THEIR pages are shared/tabled. Ours are lane-private
  and contiguous → v1 uses a pinned EXTENT arena (bump/SLAB over
  `--host-kv-mib`), not their handle machinery.
- The GDN checkpoint (docs/70) is the exact recurrent-state image a prefill
  skip needs; without it a prefix restore silently corrupts attention. It
  rides along in every park (the fork's `state_host` analog).

## 4. Non-goals (this work order)

- Mid-flight eviction of RUNNING requests (the fork's hardest piece). v1
  parks at request COMPLETION and on admission pressure of NEW requests only.
- The fork's checkpoint/turn-closure fallback and thinking-mode session-key
  matching (our tree has no thinking-mode parity; re-scope later).
- NVFP4 host tiers (A1's lane) — the arena stores the lane's NATIVE tier bytes
  verbatim; a compressed host tier is a follow-up.

## 5. Design decisions (initial — revisit only with a written reason)

1. Park-on-complete first (H2) before pressure-driven parking (H3): it
   exercises the whole D2H→ledger→match→H2D→skip path with the simplest
   trigger, and the fork's stress numbers show the pressure path is the same
   machinery with a different trigger.
2. Contiguous extent arena, not scatter-gather: one `cudaMemcpyAsync` D2H per
   side (K, V) + checkpoint + ledger. If a future shared-page substrate lands,
   revisit.
3. The token ledger is authoritative for matching (byte-for-byte prompt
   prefix, like tp2_backend's existing loop); no hashing shortcuts in v1.
4. Restore-skip correctness gate: after an H2D restore, the first NEW token's
   logits must equal a full-prefill run's logits for the same prompt (golden
   test, CPU-comparable via the reference path or a debug dump) — the fork's
   "materialization root" bug class. No restore ships without this test.
5. Eviction: smallest-parked-first LRU with a pinned flag (fork semantics) —
   pinned during the restore reserve→start pipeline.

## 6. Steps

### H1 — Host arena + layout + unit tests (CPU only)
- `src/core/host_kv_arena.{h,cpp}`: pinned-host extent arena —
  `--host-kv-mib` sizing, alloc/free extents, LRU metadata, occupancy stats.
  Port the fork's arena ONLY if the extent model fits after reading it;
  otherwise write the minimal SLAB (prefer minimal).
- Unit test: alloc/free/fragmentation/LRU-eviction/pin behavior
  (`host_kv_arena_test`), CPU-only, in the ctest suite.
- Commit: one for arena+test.

### H2 — Park-on-complete + restore-with-skip (GPU window for E2E)
- At request completion: if `--host-kv-mib` > 0 and committed pages ≥
  `host_kv_min_pages` (new flag, default 1): D2H the KVarN committed prefix
  (both sides) + GDN checkpoint + token ledger into the arena; record prefix
  identity. Lane released normally.
- At admission: prefix-match new prompts against parked ledgers; on match,
  H2D restore into the lane's workspace BEFORE prefill, set the committed-page
  counters, skip the matched prefix (reuse the existing `prefix_len` skip).
- Correctness gate (§5.4) + a two-turn E2E test (prompt A → complete; prompt
  A+B → restored, no re-prefill of A, correct output).
- Commits: park, restore, E2E test.

### H3 — Pressure parking + flags + stats
- `--host-state-slots` (cap on parked entries), admission-pressure trigger
  (park-eligible oldest-first when a new request would otherwise refuse),
  pin during restore, `host_kv_occupied` in stats_json.
- Stress test: 3 × 200k concurrent sessions (the fork's table) — no refuse,
  no OOM, correct outputs.

## 7. Constraints

- Never edit `/home/intel/ninfer/repo` directly (worktree only).
- No system-wide pkill; GPU via coordinator grants only.
- The arena is PINNED host memory: sizing interacts with host RAM budgets —
  cap and report occupancy; never allocate without a slot budget check.
- Restore must be stream-ordered with the lane's compute stream (§22.57's
  lesson: no bare null-stream copies against non-blocking streams).

## 8. Definition of done

- H1-H3 merged to `wo/host-kv-safety-net`, pushed, with the unit + E2E tests
  green via run_ci.sh, the stress table filled in this doc, and docs/155
  item 3 updated to "landed (contiguous-extent v1)".

---

## §9 H2 EXECUTION PLAN (anchors verified 2026-09-07; next session starts here)

**Data captured per parked entry (struct `HostKVParkedEntry`, new header
`src/runtime/tp2/host_kv_parked.h`):**
- `std::vector<int> ledger` — the request's prompt tokens (authoritative
  match key; prefix semantics = tp2_backend.cpp:1195's `prefix_len` loop),
- arena offsets+bytes for: K plane extent, V plane extent, GDN checkpoint
  image, MTP pool committed prefix (if `mtp_k > 0`),
- `committed_pages` (lpp), `mtp_committed_pages`, `kv_tag` (KVarN tier id),
  `pages_per_lane` at park time (restore requires equal-or-larger lane),
- pin flag + entry id + LRU stamp (the H1 arena tracks extents; the entry
  struct tracks the semantic metadata).

**Park trigger — request completion.** Anchor: the lane-release path at the
END of the per-request flow in tp2_backend.cpp (find via the
`st.cached_tokens` / `st.cache_valid` writes ~1413; the completion point is
where the request's decode loop exits). Conditions: `host_kv_mib > 0`,
`committed_pages >= host_kv_min_pages` (new option, default 1), arena has
room (else evict_for with the pin rules; if still short, skip parking and log
once). D2H: cudaMemcpyAsync on the lane's compute stream, THEN
cudaStreamSynchronize (§22.57 rule — never bare null-stream copies), into
arena extents. Record ledger + geometry. Free nothing at park time: the lane
release already frees the workspace — ordering is D2H → release.

**Restore trigger — admission.** Anchor: after the Budget admission accepts a
request and before prefill, where `st.cached_tokens` is first consulted
(~1188-1200). Match: longest prefix among parked ledgers (the v1 rule:
ledger is a prefix of the prompt, ≥ `host_kv_min_pages*64` tokens). On match:
(1) pin the entry, (2) H2D the K/V planes into the lane's KVarN workspace
prefix + GDN checkpoint image into the docs/70 checkpoint buffer + set
`kvarn_lpp_dev`/`kvarn_mtp_lpp_dev` counters, (3) set
`st.cached_tokens = ledger` + `st.cache_valid = true` so the EXISTING
prefix-skip loop takes over, (4) unpin after the first attend ran.
Restore-into-smaller-lane: truncate the restored prefix to the lane's page
count (partial restore, ledger stays for future larger lanes).

**Correctness golden gate (blocks the step):** run prompt A once (full
prefill), capture the first decode-step logits; run A again through the
park→restore path with `host_kv_min_pages=1`; the first decode-step logits
must be bit-identical (the GDN checkpoint + KVarN pages are the ENTIRE
recurrent state — any difference means a missed state component). If they
differ, find the missing state before shipping (candidate list: the MTP pool
prefix, the ring write cursor, the committed-pages counters).

**Tests:** (t1) unit — park/restore roundtrip of the metadata (CPU);
(t2) E2E two-turn restore-skip with logits golden gate (GPU window);
(t3) three-turn chain; (t4) restore-into-larger-lane; (t5) park refusal when
the arena is full (graceful). H3 after t1-t5: pressure parking + flags +
stats (`host_kv_occupied`, `host_kv_entries`), then docs/155 item 3 flip.

## §10 BLOCKING TODO for the E2E gate (found 2026-09-07, pre-gate)

The image currently captures: linear-attention slots, KVarN open-page TILES +
tails + committed/tile counters (host struct fields), tail snapshot, GDN
checkpoint, cached_ph prefix. It does NOT yet capture the COMMITTED PAGED KV
PAGES (PagedKVCacheView planes [0, committed_pages) per layer) — the bulk of
the cache. A restored entry would keep the open-page tail + GDN state but lose
every committed page: the golden gate WILL fail for prompts > 1 page until
PagedCommittedK/V capture is added (per layer: D2H the lane's
[0, committed_pages) range from each PagedKV plane; the lane view +
kvarn_lpp_dev counters give the range). Naming per coordinator: this is the
REAL committed KV data (k4v2 codes+scales) — the docs/83 "staged bf16 shadow"
was removed (has_staged=false) and is unrelated.

## §11 E2E RE-RUN FINDINGS (2026-09-07, window 10 — 6144 MiB arena)

With --host-kv-mib 6144: the image FITS (147 MiB: linear slots + GDN ckpt +
tiles + tails + snapshot; the "staged shadow" is gone — this IS the real
cache, per the coordinator's naming note). Park FIRED on both ranks.
Two precise defects remain (both are t-ladder iterations, not design flaws):
  1. RESTORE GEOMETRY MISMATCH: "restore component byte mismatch" — the
     restore's validate() correctly refused a stale-shape component. Prime
     suspect: kvarn_prefix_snap tensors captured pre-B vs the live snap
     shapes post-B (the snap is re-captured every prefill; sizes may track
     the tail geometry of the most recent prompt). Fix: restore the snapshot
     into the PARKED shapes by re-allocating the snap tensors, or skip
     snapshot restore when shapes drift (full-prefill fallback).
  2. EMPTY-LEDGER PARKS on rank 0: "parked ... ledger 0 tokens" — the park
     guard requires !st.cached_tokens.empty(), so a 0-ledger image means the
     capture ran when cached_tokens was already cleared (guard/capture raced
     across the two rank threads, or the clear happens between guard and
     capture). Fix: re-check inside capture and bail before alloc.
Also: the >1-page committed-pages gap (§10) stands — the gate prompt was 81
templated tokens (>64), so one committed page existed and was not captured;
A' output was empty vs A empty — NOT a pass (both empty = the real tell).
NEXT SESSION: fix 1+2 (both localized), add PagedCommittedK/V (§10), THEN the
golden gate window. The graceful fallback path (restore fails → full prefill,
request survives) is PROVEN — the 500-class failure mode is gone.

## §12 PagedCommittedK/V CAPTURE DESIGN (API mapped 2026-09-07; implementation ready to start)

The PagedKV API (src/core/paged_kv_cache.h) provides everything the capture
needs:
  * `lane_allocs[b]->page_ids()` — the lane's logical page ids
    ([0, committed_pages) are the committed ones; `committed_pages` comes
    from kvarn_ws.text[l].committed_pages — PER LAYER, so capture per layer
    uses THAT layer's counter, not the text-wide one),
  * `kv_view.plane(i)` — the storage planes (PagedKVPlaneSpec: dtype,
    leading_extent, head_extent; plane order PageMajor or HeadMajor),
  * `bound_row()` / `block_table_row(row)` — the lane's table row.
Capture (per plane, per committed page p): D2H the page's extent — for
PageMajor order the page's region is contiguous per (leading_extent, head)
tile; use the plane's own stride math from PagedKVPlaneLayout. Restore: H2D
back through the NEW lane's page_ids (the page ids may differ — the ledger
tokens are what must match, not the physical pages; the block table is
re-published by the normal bind path). Add `page_ids` to the image metadata.
Extent sizing: sum over planes of committed_pages × plane.page_bytes — for
the k4v2 tier ≈ 2.2 MiB per layer-page-tile set (see kvarn_workspace.h's
tile math); size --host-kv-mib accordingly (6144 MiB covers ~64-token ×
multi-turn parks comfortably).
Implementation home: extend capture_slot_image/restore_slot_image with a
`const PagedKVCacheView& kv_view, const PagedKVAllocation& alloc` parameter
pair + the per-layer committed-pages vector (already in the image as
text_committed). The trigger sites already hold st (which owns
lane_allocs/lane_views).

## §13 SESSION 2 (2026-09-07): §10 + §11 landed (96b6c14b); golden gate is next

**Found on resume:** the committed tree did NOT compile — c80f2246 renamed the
component kinds in the header only; the cpp still used KvarnStagedK/etc.
Window 10's E2E ran the pre-rename .o (17:27 object vs 17:44 commit). Lesson:
a rename commit that leaves a TU uncompilable invalidates every "verified"
claim built on the stale binary.

**Landed (96b6c14b, engine builds, ninfer_host_kv_arena_test green):**
  1. §10/§12 committed-page capture, as designed, with two deviations that
     matter: (a) the scale side table is NOT page-contiguous — per (layer,
     page, head) tile it sits at `1152*(layer + n_layers*(h + heads*phys))`
     floats (the commit path's convention, kvarn_workspace.cpp), so scales
     copy per tile, not per page; (b) the image carries three components per
     layer (K codes, V codes, scales) sized `committed_pages × per-page
     bytes` (k4v2, 2 heads/rank: 16K + 8K + 9K ≈ 33 KiB per layer-page,
     ~532 KiB per page across 16 layers). Capture walks [0,
     ws.committed_pages) — decode-committed pages beyond the snap point are
     over-captured deliberately; the restore flow's rewind clamps the
     counters down and those pages are never read.
  2. §11.2: the empty-ledger guard is re-checked inside capture, before the
     arena alloc; park distinguishes the empty-state skip from the
     arena-full skip in the log.
  3. §11.1: restore rebuilds its walk from the IMAGE's component list (snap
     presence = SnapTextK kind present; NEVER the live snap.valid — that
     asymmetry was the only way the old walk could see a different component
     set than capture) and validates count+kind+bytes of EVERY component
     before the first H2D. Drift errors name the component and both sizes.
     A refused restore leaves live state untouched → the caller's full-
     prefill fallback is genuinely clean. Capture frees its extent on a
     failed copy (the old code leaked it on the park-catch path).
  4. Snap STRUCT scalars now ride in the image (valid, token_count,
     per-layer tails/pages, mtp_tail/page, d0): the prefill flow gates the
     tail restore and the MTP-seed reuse (ar_hidden/d0) on
     `snap.valid && snap.token_count == prefix_len`, so without these the
     restored state would pass validation and still mis-serve.

**Disk event:** the full-tests link (83 test binaries, 8.7G) hit 100%. My
lane's stale test binaries were deleted (relinkable from kept objects;
host_kv_arena_test + apps kept) → 9.6G free. run_ci.sh full suite is
blocked until the shared disk has headroom — coordinator informed.

**Gate protocol for the E2E window (unchanged from §11):** prompts must
exceed 64 templated tokens so a committed page exists; PASS = A''s third
response identical to A's (temp 0) + parked/restored lines both ranks +
prefix-hit/full-snap lines in the rank logs. If the gate still fails, the
first suspect is NO LONGER missing state — diff the first decode-step
logits (full prefill vs restore) per §5.4 and bisect from there.

## §14 GOLDEN GATE: PASS (e03d5d9c, 2026-09-07)

A (101 templated tok) → B (98 tok, unrelated) → A' (=A), temp 0, third
response **byte-identical** to the first (1646 chars, reasoning+content).
`parked`/`restored` lines on both ranks, `prefix hit: 101 tokens (slot 1)`,
`reuse=restore_response_checkpoint`, ttft 261 ms → 58 ms on the restored
turn (park/restore overhead ≈ net win already at 101 tokens). §5.4 state
identity proven: with NINFER_HKV_DBG, A2's post-restore fingerprints equal
A's post-prefill fingerprints exactly (pool l0/l15 pages 0+1, GDN slot-0
conv/rec l0+l47, cached_ph, t0).

**The bug that was blocking the gate (the session's big find):** capture and
restore addressed the arena as `arena.at(component.offset)` — WITHOUT
`image.arena_offset`. Every park wrote its components at the same absolute
region (stomping the previous entry), and every restore read the LATEST
park. With one parked entry everything looked fine; with three (warmup, A,
B) the A2 restore silently reinstated B's pool pages + GDN slots while
prefix-hit/ttft/log lines all looked healthy. `cached_ph` was doubly
misleading: A and B share the chat-template prefix, so the first 4096 bytes
of ph are identical — a "successful" ph restore that wasn't. Diagnostics
that cracked it (kept, env-gated under NINFER_HKV_DBG): park-side arena
hashes (`[HKV-ARENA] rank tag extoff kind coff h`), restore-side
source-vs-dest hashes (`[HKV-RESTORE] arena tag ... src dest`), and
decode-step-1 state fingerprints (`[HKV-STATE]`). The H1 allocator itself is
correct (extents disjoint: 0 / 156583936 / 315756544).

Also this session: the §13 capture work (committed paged KV pages for BF16
pools — the DEFAULT serving tier, kvarn_ws unbound — plus kvarn k4v2),
§11.1/§11.2 fixes, and the discovery that c80f2246 left the tree
uncompilable (header-only rename).

**Next (H3, per §6):** pressure parking (admission-pressure trigger),
`--host-state-slots` enforcement stats, `host_kv_occupied` in stats_json,
pin during restore; stress table (3 × 200k). The t-ladder: t1 ✔ (unit),
t2 ✔ (this gate), t3 (three-turn chain), t4 (restore-into-larger-lane),
t5 (arena-full refusal — graceful path already exercised by §11's logs).
Watch: multi-entry LRU/eviction is now LIVE for the first time (3 entries
were parked in the gate) — evict_for/pin paths have still never run.

## §15 H3 IMPLEMENTATION (2026-09-07, 0f3ce8c0 + 5140242b; validation pending GPU window)

Landed:
  * stats_json surface: RuntimeStats.host_kv_{occupied,capacity}_bytes +
    host_kv_{entries,max_entries}; HostKvNet keeps lock-free atomics
    (sync_counters at push/drop/evict); tp2::host_kv_snapshot() sums rank
    nets; TPEngine::runtime_stats() fills them; the throughput reporter
    prints `host-kv: X/Y MiB (P%) | entries=N/M` when the net is on.
  * arena-pressure eviction (§6 "park-eligible oldest-first" mapped to our
    substrate): extent doesn't fit → evict_for(bytes) (§5.5 smallest-first,
    oldest tiebreak, pin-protected) → erase the dropped entries → capture
    retries ONCE → still-short = graceful skip (t5 behavior).
  * pin/touch during restore (§5.5): extent pinned for the H2D window,
    PinGuard unpins on all exits; LRU touch renews recency.
  * TRIGGER FIX (found by inspection): park's pressure-eviction can drop the
    entry the pre-park find_best matched → restoring the stale index would
    reinstate the WRONG session. The trigger now parks first, then re-resolves
    the match and restores only if it still exists with match > live.

**STRESS RE-SCOPE (§8 DoD — written reason per §5):** the "3 × 200k-token
concurrent sessions, no refuse" table is not achievable in v1: v1 parks
CONTINUATION state at session rotation and does NOT page device KV to host
(§4 non-goal: no mid-flight eviction). A 200k session does not fit the 50k
device pool regardless of the net. v1 stress = 4 sessions rotating through
the net at kv_capacity scale with eviction forced (arena-sized to pressure:
--host-kv-mib 460 for arena-pressure eviction; default 6144 exercises the
slot cap), PASS = every revisit byte-identical to its first output, evicted/
restored lines present, no refuse/OOM/failed restore. A literal 200k×3
requires a v2 with device-side paging — separate work order if wanted.

Validation plan (single GPU window): (1) stress @ --host-kv-mib 460,
(2) stress @ 6144 (slot-cap eviction), (3) t5 @ --host-kv-mib 64 (parks
gracefully skipped, outputs still deterministic). Driver:
tools/smoke/host_kv_h3.sh (MODE=stress / MODE=t5).

## §16 H3 VALIDATED (2026-09-07, 68ad7336; single GPU window, three configs)

All runs rank-paired, 4 sessions (S1..S4, 96-101 templated tokens each),
every revisit byte-identical to its first output, zero refuse/OOM/failed
restore:
  1. **--host-kv-mib 460 (arena-pressure path):** `evicted entry` on both
     ranks; one park then gracefully skipped — after evicting the 149.3 MiB
     warmup extent the largest contiguous free region was still < the 151 MiB
     image (the §5.2 contiguous-extent limit; the fork's scatter-gather
     exists for exactly this). 2 restores/rank; non-restored revisits
     full-prefilled to identical output.
  2. **--host-kv-mib 6144 (slot-cap path):** the 5th park evicted the warmup
     entry; ALL FOUR revisits restored (`3 entries remain`) and identical.
  3. **t5, --host-kv-mib 64:** every park gracefully skipped (image never
     fits), all requests complete, revisits deterministic.
  4. jsonl throughput events now carry `host_kv {occupied_bytes,
     capacity_bytes, entries, max_entries}`. NOTE: the periodic throughput
     reporter has never fired on the tp2 arm — `report_has_activity` counts
     request-slot fields TPEngine::runtime_stats leaves at zero
     (pre-existing, separate lane).

**t-ladder final:** t1 ✔ t2 ✔ t3 ✔ (S1'..S4' revisits after 3 intervening
parks) t5 ✔. **t4 (restore-into-larger-lane): N/A on this arm** — single
static lane, same pool every request; nothing varies pages_per_lane.

**Remaining for DoD:**
  * docs/155 item-3 flip — BLOCKED ON INTEGRATION: docs/155 lives on main
    (repo/docs; the worktree rule forbids editing repo/ directly).
  * The coordinator's scope call on the 200k×3 literal (§15 re-scope or v2).
  * Untested scopes, explicitly: MTP-on configs, batched arm
    (max_concurrency>1), concurrent multi-request rotation (the driver is
    sequential single-seat, matching the current arm).

## §17 V2 SCOPE — derived FROM THE FORK (2026-09-07; operator-directed)

Directive: "properly scope from the fork and start immediately." Scope below is
read from gzenz's code (/tmp/gzenz_ninfer @ 4b882d0), not from §1's summary.

### 17.1 What the fork actually does (verified in code)

Layered subsystem:
  1. `LogicalKVPageStore` / handles — logical page layer over shared pools.
  2. `HostKVExtentStore` — owns typed HOST extents + the ordered logical pages
     each covers; scatter-gather over `HostKVArena`; fragmentation-aware
     (suballocation, partition runs).
  3. `HostKVSafetyNet` — evicted-continuation store: ledger + execution
     frontier + CHECKPOINT frontier (turn boundary), session-key + compact-
     prefix matching (thinking mode), pin-by-stable-ID, take_pinned/re-add.
  4. Materialization planner + ResourceManager (~3k lines) — pressure actions
     on PRIVATE and SHARED (COW) page owners: `pressure_spill_pages`,
     `pressure_private_owners_evicted`, `pressure_shared_owners_evicted`;
     feasibility/value/cost model, bounded anytime search, deterministic
     selection (maintainer doc §7-9, 979 lines, zh).

**THE critical mechanism finding:** safety-net restore fires ONLY on
`ReusePath::Root` (cold-start materialization): "check if the prompt prefix
matches a previously evicted continuation… restore KV via H2D instead of a
full prefill." The fork has **NO host-tier attention** — a RUNNING owner must
be device-resident; the planner evicts OTHER (idle/queued) owners to admit it.
Their 330k–470k sessions therefore required device residency of the active
session, which their tier footprint allowed.

### 17.2 Hardware boundary for OUR 27b (the 200k arithmetic)

bf16 pool ≈ 33 KiB/token → 200k tokens ≈ 6.5 GiB device KV; post-weights free
VRAM ≈ 2.7 GiB. **One 200k bf16 session cannot be device-resident on this
pair.** At kvarn k4v2 (~÷6 vs bf16 incl. scales) 200k ≈ 1.1-1.4 GiB → FITS.
⇒ **The 3×200k mission is achievable on this hardware ONLY as
tier-compressed pools (A1's kvarn/NVFP4 lane) + fork-style owner eviction.**
Host-tier attention is NOT required (the fork proves the model works without
it) and is explicitly OUT OF SCOPE.

### 17.3 V2 work breakdown (fork-mapped, ordered)

  * W1 — pressure-planner hook (the core): when admission/resize cannot
    materialize an owner within pool capacity, evict the LRU idle owner
    (spill whole continuation → v1 net) and retry. Generalizes v1's
    admission-time park to planner-triggered spill of ANY idle owner.
  * W2 — spill/restore of ANY owner, incl. MID-TURN: checkpoint-frontier
    restore (fork's two-level find). Our snap (turn-end) + gdn ckpt already
    cover the checkpoint state; add execution-frontier spill (prompt+generated)
    with the checkpoint fallback ride-along.
  * W3 — session-key + compact-prefix matching for thinking-mode follow-ups
    (fork's session-key fallback); our prefix-only match misses
    preserve_thinking=off flows.
  * W4 — scatter-gather host extents (port HostKVExtentStore semantics over
    our HostKVArena) so ≥200k-token parks survive fragmentation.
  * W5 — capacity: kv_capacity raise + tier (config; pairs with A1's tiers).
  * W6 — stress harness: N sessions × 200k-token ledgers rotating under
    pressure; fill the §8 table; maximal_fallbacks == 0.

NON-GOALS for v2.1: host-tier attention; COW page sharing across requests
(our pool has no cross-request sharing — fork's shared-owner machinery is
deferred until multi-request pools exist); thinking-mode session keys beyond
W3's match layer.

### 17.4 Status

Scope: this section (operator-approved direction). W1-W6: NOT STARTED — this
session spent its budget on the fork read + this scope. Next session starts
W1 (hook point: the Budget preflight refusal path + our existing HostKvNet).

### 17.5 W1 reconnaissance (same session; corrects the W1 hook point)

`tp2_budget.h`'s Budget/preflight is a STATIC VRAM budget (auto-capacity probe
at engine construction) — it is NOT a per-request runtime admission gate. The
runtime refusal surfaces are: (a) prompt > kv_capacity (inadmissible at the
request layer), (b) lane saturation (max_concurrency). Consequences:
  * On the SINGLE-SEAT arm (lanes==1, the validated arm) there is at most ONE
    live owner, and "evict an idle owner to materialize a new one" is
    semantically what v1 already does at admission. W1's fork-faithful
    pressure planner only gains meaning with MULTIPLE live owners.
  * ⇒ W1's real target is the BATCHED arm (max_concurrency>1): N live lane
    owners + pressure = evict-and-spill the LRU lane when admission/resize
    cannot fit a new/returned owner. Prerequisite: batched arm + host-kv
    (currently unsupported — kvarn_lane_ws/batched attend paths bypass the
    single-seq net).
  * W5 (tier + kv_capacity raise) is what makes >50k SESSIONS admissible at
    all (200k bf16 cannot be device-resident; k4v2 200k ≈ 1.1-1.4 GiB fits
    the post-weights budget). W1/W5 are complementary, not alternatives.
Revised order: W5 (config+capacity, small) → W3 (session-key match, small)
→ W2 (mid-turn checkpoint spill, medium) → W1 (batched-arm planner, large,
blocked on the batched-arm net) → W4 (scatter-gather, medium, on demand).

## §18 PORT SCOPE — fork-complete host-KV (coordinator decision B, 2026-09-07)

**Fork reference:** gzenz/ninfer fork, `/tmp/gzenz_ninfer` (read-only), tip
`4b882d0` ("Merge pull request #2 from gzenz/fix/materialization-root-fallback").
PRESERVE THIS TREE — it is the port source (volatile /tmp location flagged).

### 18.1 What the fork implements (verified in code this session)

  F1 `HostKVArena` (typed page-layout arena: layouts per pool geometry,
     page_stride, allocate/allocate_multi scatter-gather, writable_view).
  F2 `HostKVExtentStore` — descriptor/membership/release-mark bookkeeping for
     multi-extent allocations; suballocation + partition-run scratch.
  F3 `HostKVSafetyNet` — evicted-continuation store: ledger + prefix_identity,
     EXECUTION frontier + CHECKPOINT frontier (two-level find), session_key +
     compact_prefix (thinking-mode fallback), pin-by-stable-ID / take_pinned,
     zero-suffix-reuse rejection.
  F4 SPILL (`spill_victim_to_host_kv_safety_net`, program_impl.h:5799): whole
     continuation D2H — text pages + backend (MTP) pages + state image +
     checkpoint image — via `physical_pool().copy_to_host(device_pages,
     host_view, stream)`; pre-check "reclaimable even after evicting ALL" before
     touching anything; two-phase eviction (all unpinned → then pinned, with a
     stream-sync landmine guard), smallest-total-pages first, oldest tiebreak;
     single-contiguous alloc first, scatter-gather fallback.
  F5 RESTORE — on Root-reuse materialization ONLY: find → pin (entry stays in
     net) → take_pinned in start_sequence → reserve device pages →
     `copy_from_host` H2D per page + state H2D → RE-ADD entry with refreshed
     timestamp. Checkpoint-level selection when the follow-up rewinds.
  F6 Materialization planner + ResourceManager (pressure actions over PRIVATE
     and SHARED/COW owners; feasibility/cost/anytime-search; pressure stats:
     spill_pages, private/shared owners evicted) + `prefix_identity` guard +
     serve-side stats_json fields.

### 18.2 What we have vs what ports

  Feature                     | Fork | Ours (H1-H3)          | Port work
  ----------------------------|------|-----------------------|-----------
  extent arena                | F1   | contiguous-only v1    | P1 layouts+multi
  scatter-gather store        | F2   | —                     | P1 port
  continuation store          | F3   | ledger-only, 1-level  | P3 frontiers+keys+pin-ID
  whole-continuation D2H      | F4   | per-component capture | P2 pool primitives + P4
  eviction policy             | F4   | slot-cap + 1-retry    | P4 two-phase policy
  restore                     | F5   | drop-on-restore       | P4 take_pinned/re-add
  planner pressure trigger    | F6   | admission-time only   | P4/P7 (arm-dependent)
  checkpoint frontier         | F3/F5| snap+ckpt exist, no frontier bookkeeping | P5
  session-key/compact match   | F3   | —                     | P6 (adapt)
  stats_json                  | F6   | jsonl host_kv block   | P7 counters (spill/evict/restore/fallbacks)

### 18.3 Port plan (P1-P8, files, adaptations)

  P1 arena layouts + multi-extent + extent store — `src/core/
     host_kv_arena.{h,cpp}` (extend), new `src/core/host_kv_extent_store.{h,cpp}`.
     CPU unit: fragmentation/suballocation/eviction-order.
  P2 pool page-bulk primitives — `src/core/paged_kv_cache.{h,cpp}`:
     `copy_to_host(page_handles, host_view, stream)` / `copy_from_host` +
     per-page device handles. CUDA unit: roundtrip checksum per page.
  P3 safety-net upgrade — `src/runtime/tp2/host_kv_parked.{h,cpp}`:
     execution_frontier ledger (prompt+generated), checkpoint-frontier
     ride-along image, session_key/compact_prefix fields, pin-by-ID +
     take_pinned, zero-suffix-reuse rejection. CPU unit: matching/frontiers.
  P4 spill/restore integration — `tp2_backend.cpp`: generalize the trigger
     from admission-time to pressure-time (planner hook per arm); take_pinned
     → H2D → re-add (replaces drop-on-restore); two-phase eviction policy.
     Gates: t2/t3/t5 + new t6 (fragmented scatter-gather), t7 (pinned-phase
     eviction), revisit-after-drop-reasoning (checkpoint level).
  P5 checkpoint-frontier bookkeeping — turn-boundary capture (our snap/gdn
     ckpt already carry the state; add frontier fields + selection logic).
  P6 session-key adaptation — OUR tree's prepared-prompt/session identity
     differs from the fork's; map PreparedSessionKey/compact_prefix onto our
     tokenizer frontend (recon needed; risk: our frontend may lack session
     keys → follow-up for the serve layer).
  P7 stats counters — spill_pages/owners_evicted/restores/fallbacks into the
     RuntimeStats host_kv block + jsonl.
  P8 CI cells — host_kv_gate.sh (A→B→A' byte-identity) as a full-mode cell +
     batched-rotation cell (Line M, conc=2) in tools/ops/run_ci.sh; both
     green = done (coordinator condition 2).

### 18.4 Validation literal (hardware-bound → user sign-off via coordinator)

  3 concurrent 200k-token sessions, **i8 KV tier (20 KiB/token)**:
  host arena 12 GiB (fits ~20 GB available RAM); device pool = active session
  ≈ 2 GB/rank (fits ~2.7 GiB post-weights). bf16 (24 GB host, 4 GB/rank
  device) does NOT fit — the literal is i8-bound. Sessions rotate under
  pressure; PASS = every revisit byte-identical, spill/evict/restore lines
  present, fallbacks == 0. (Fork reference point: 140+ reqs, 3 × 330k-470k,
  host_kv_occupied 2.81 GB, zero fallbacks.)

### 18.5 Adaptation risks

  R1 our PagedKVPool has no per-page device-handle layer (P2 adds it);
  R2 single-seat trigger ≠ planner pressure — full parity needs the batched
     arm (multiple live owners); single-seat ships first with generalized
     triggers;
  R3 session keys may not exist in our frontend (P6 recon before commit);
  R4 i8-tier pool geometry for the literal (pool bytes/page differs from
     bf16 — budget math via kv_bytes_for_tier, docs/151 §18.1bis).

### 18.6 Execution order

  P1+P2 (foundation, CPU+CUDA units) → P3+P4 (feature, gates t2/t3/t5/t6/t7)
  → P5+P6 (matching completeness) → P7+P8 (observability + CI cells) →
  literal run → coordinator/user sign-off.

### 18.7 PROGRESS LEDGER (night push, 2026-09-07)

  P1 ✅ e1ad2759 — page-run arena allocation (+CPU units: fragmentation/
     rollback/coalesce).
  P2 ✅ e284c0ba + 991caa45 — pool copy_pages_to_host/from_host + CUDA unit
     (layout, roundtrip, scatter-restore, collateral, HeadMajor refusal).
  P3 ✅ 35ce7897 — net-level pin, take/re-add lifecycle (entries survive
     restore), frontier fields, find_best skips pinned.
  P4 ✅(single-seat) 5464ad79 — scatter-gather image addressing (byte_view),
     allocation-wide pin/touch, F4 pre-check (skip without evicting when the
     spill is doomed), two-phase policy; drop-on-restore retired. The
     batched-arm half of P4 remains (W1, blocked on the batched net).
  P7 ✅ c84610fd — lifecycle counters (parks/restores/evictions/
     restore_failures) through net→snapshot→RuntimeStats→jsonl.
  P8 ✅(gate cell) ca1424d8 — full-mode gate cell PASS-capable now;
     batched-rotation cell registered, PENDING verdict until W1 (STRICT=1
     flips to hard-fail). Both cells live in run_ci.sh --full.
  P5 ⏳ merged into P6 (checkpoint matching without session/compact-prefix
     matching cannot fire on our arm — prefix match fails first).
  P6 ⏳ next: recon our frontend for session identity, then adapt.

VALIDATION still owed: GPU window for t6 (fragmented scatter-gather E2E),
t7 (pinned-phase), t2/t3/t5 re-run on the new addressing, and the i8 literal
(sign-off pending). CURRENT STATE IS GREEN ON CPU UNITS + previous gates;
the re-addressed image path needs its window before "done" is claimed.

### 18.8 P6 RECON RESULT (night push close-out)

Our `PreparedPromptData` (prepared_prompt.h) has NO session identity — the
fork's `PreparedSessionKey` has no counterpart; we carry only
`PromptIdentity` (the prefix-content guard). Recommended P6 design when
implemented: optional `session_id` on the OpenAI request (or X-Session-Id
header) → thread through prepare into PreparedPromptData → net matches by
(key, checkpoint_frontier) per the fork's fallback. Until then,
preserve_thinking=off follow-ups fall back to full prefill (correct, slower)
— no correctness risk, a recall gap only. P6 stays ⏳ with this design note.

### 18.9 P6 SOUNDNESS CONSTRAINT (why session-key restore is not a mechanical port)

A session-key hit restores the checkpoint image; the follow-up's prompt
DIVERGES from the ledger at the reasoning-drop point (that is why prefix
matching failed). Restoring state AT the ledger point and then prefilling
from the divergence point double-consumes tokens — silent corruption. The
fork stays sound because its checkpoint STATE IMAGE is captured AT the
checkpoint frontier (state matches checkpoint tokens exactly), and its
compact_prefix guarantees the shared prefix extends to that frontier.

Therefore P6 requires, in order:
  1. Our park image must be a valid CHECKPOINT image — on the BF16 arm it
     already is (ledger = prompt; cache slot + pool pages [0, ceil(plen/64))
     are exactly the prompt-end state; the flow's rewind bounds attention to
     the restored prefix). Verified property, not an assumption.
  2. session_id plumbing (serve → request → PreparedPromptData → net).
  3. Restore selection: session-key hit sets prefix_len = checkpoint_frontier
     and MUST gate the divergence — prefill runs [checkpoint_frontier, plen),
     never [divergence, plen) with ledger-point state.
  4. compact_prefix (reasoning-stripped ledger) so the shared prefix actually
     reaches the frontier for thinking-mode follow-ups.
Without 3+4 the feature corrupts silently; do not ship a partial P6. Until
then the failure mode is recall-only (full prefill), which is safe.

### 18.10 t6 REGRESSION — OPEN, BISECTION STATE (2026-09-07, HEAD 8f5f9301+)

**Repro:** 4-session rotation @ --host-kv-mib 460 (pressure eviction): S1's
revisit DIFFERS (1599 vs 1646 chars); S2/S3/S4 revisits byte-identical; all
4 restored the correct entries (tag logging); 0 failures; 10 evictions.
Serve12's 6144 A/B/A' gate on the same binary PASSED — the regression is
specific to the pressure-eviction rotation.

**Established (reliable probes):**
  * restore target correct: "restoring entry 2 (ledger 101, runs 1)" at S1b.
  * restore delivery correct for probed components: [HKV-RESTORE] src==dest
    for rec0, pagedK l0 pages 0+1, ph (the src/dest probes are consistent).
  * pre-decode state fingerprints (NINFER_HKV_FULL) vs A: page0 ✔, GDN rec ✔,
    ph ✔, t0 ✔ — but **page1 (l0+l15) full-page hashes DIFFER from A's**
    (db77fbea/3542ec7f vs A's 29be8433/cb721973) and **conv l0 = 377a3029 vs
    A's 5e951f89** (conv l0 is CONSTANT 5e951f89 across warmup + all 4
    different-prompt sessions in normal operation — so S1b's value is
    wrong-but-fluent).

**⚠ INSTRUMENTATION WARNING (read before probing):** the three probes hash
DIFFERENT scopes — [HKV-STATE] = full page/slot (under NINFER_HKV_FULL),
[HKV-RESTORE] src = 4096 B of arena, [HKV-RESTORE] dest = 1 MiB from the page
base (overshoots into adjacent pages — racy AND oversized). Earlier
"matches/mismatches" across scopes are NOT valid comparisons. Fix the probes
to ONE canonical scope before drawing conclusions (recommended: hash exactly
page_stride bytes for pages; exactly slot bytes for GDN; exactly
min(component_bytes, 64 KiB) for ph).

**Working theories (in order):**
  T1 the image's page-1 stale tail (slots 37-63 hold the parked session's own
     decode KV by design) is read by the attend under some envelope condition
     specific to the rotation — BUT S3b restored an identical-shaped stale
     tail and PASSED, so the trigger is S1-specific, not shape-specific;
  T2 conv l0 slot corruption — the only component whose normal value is
     CONSTANT (5e951f89 = likely zeros) and whose restored value is nonzero
     (377a3029) — nonzero conv state entering decode changes logits subtly
     (early-small divergence ✔). ASK: why would entry 2's conv capture differ
     from the constant? The capture reads conv_slot(0, cache_slot=1) at req2's
     admission; slot 1 = copy_slot(0→1) at S1a's prefill end. If layer 0's
     conv slot is normally zeros at decode start but the cache copy captured
     NONZERO prefill-end state — then the RESTORE of that state into slot 0
     changes what decode sees (the normal path has zeros there!). I.e., the
     v1 net captures the CACHE slot faithfully — but the CACHE slot's conv
     state is not what the decode consumes at that point in the flow.
     VERIFY: hash conv_slot(0, slot=1) vs conv_slot(0, slot=0) at req1's
     decode start in a NORMAL (no host-kv) run — if they differ, T2 is
     confirmed and the fix is capture masking (zero the conv components in
     the image, or capture conv from slot 0's pre-copy value).
  T3 ph/page-0 slot 4-63 corruption — not excluded (fingerprints only cover
     slots 0-3 in the 4096 runs; the FULL run's page1 comparison is confounded
     by T1's expected tail difference).

**Next session:** fix probe scopes to canonical, run T2's verification probe
(conv slot 0 vs slot 1 in a normal run), then T2 fix if confirmed. THEN
t6/t7 clean + literal. Do NOT chase T1 further before T2 — the conv evidence
is the strongest lead (constant-vs-nonzero is not a masking artifact).

### 18.11 t6 NIGHT FORENSICS (HEAD 21cb2a50..ea0b60ff + probes)

**Found + fixed:** the [HKV-STATE] dump raced the non-blocking compute stream
(legacy-default-stream cudaMemcpy does not order against
cudaStreamNonBlocking) — all pre-sync fingerprint comparisons were
unreliable. Fixed: sync before hashing (21cb2a50); NINFER_HKV_FULL lifts the
4096 B probe cap; probes extended (restore src/dest per conv/rec component +
paged pages 0/1; park-side arena hashes cover conv; arena hashes full-scope
under NINFER_HKV_FULL).

**Reliable full-scope data (rotation @ 460, rank 0):** S1b's post-restore
state vs S1a's post-prefill state: page0 ✔, GDN rec l0+l47 ✔, ph ✔, t0 ✔ —
but **page1 (l0+l15) ✗ and conv l0 ✗**. And the arena-side: entry 2's conv l0
park-time hash = 5e951f89 (correct, = A's post-prefill conv); **the same
arena region at req5's restore-time src = 77b793c7 (wrong)**. The restore
delivered arena bytes faithfully (src==dest per component).

⇒ **Entry 2's ARENA REGION CHANGED between req2 (park) and req5 (restore) —
with NO allocator overlap** (all extents verified disjoint by offset math:
e1 [0,149.3M] evicted; e2 [149.3M,307.7M]; e3 [307.7M,473.2M]; e5 scatter
[0,149.3M]∪[473.2M,475.0M]; e7 scatter [0,149.3M]∪[473.2M,475.0M]). A wild
write into entry 2's extent is suspected — candidates: a component-addressing
bug in a LATER capture's D2H (entry 5/7's scatter copies), or an
arena-region aliasing in byte_view for multi-run allocations. The extended
per-component probes (committed) will name the writer on the next window.

**Interpretation for the fix:** once the wild write is pinned, the fix is
NOT masking — masking would only hide an arena-integrity bug that will bite
other consumers. The arena/extent addressing must be corrected.

**Also verified this session:** conv l0 slot is CONSTANT (5e951f89) across
warmup + 4 different-prompt sessions at decode start in normal operation
(likely all-zeros) — capture masking would have "fixed" the symptom while
hiding this real bug. Good thing it wasn't applied.

### 18.12 t6 CANONICAL-SCOPE WINDOW RESULT (39a39811; regression STILL OPEN, narrowed)

Reliable full-scope per-request state table (rank 0, decode start):
  * page0 (l0+l15): S1a == S1b == S2a == ... ✔ (template-identical slots 0-3
    dominate the 4096 view; full-page also matches — page 0 is never
    rewritten by decode, so this is the CORRECT restored content).
  * page1 (l0+l15): S1b (db77fbea) ≠ S1a (29be8433) — **EXPECTED by the stale
    tail by design** (slots 37-63 hold the parked session's own decode KV);
    S2b's page1 == S2a's page1 ✔ (S2b restored correctly and PASSED).
  * gdn rec l0: S1b == S1a ✔. gdn l47: S1b == S1a ✔. ph ✔ t0 ✔.
  * **conv l0: CONSTANT 5e951f89 across warmup + ALL sessions in normal
    operation, but S1b = 377a3029 (nonzero, never-seen) — the ONLY component
    whose normal value is constant and whose restored value differs.**

**Interpretation:** the pool restore is CORRECT (page-1's full-hash difference
is the designed stale tail — masked by the envelope; S2b proves masked tails
are harmless). **The diverging input is the CONV state: the normal path's
decode consumes a CONSTANT (zero-ish) conv state at that flow point; the
restore delivered NONZERO conv state (377a3029) — decode then diverges.**
T1 (page-tail) demoted; T2 (conv) promoted to PRIMARY.

**Why the restored conv is nonzero — leading candidates:**
  C1 the capture read conv_slot(0, cache_slot=1) — the CACHE slot's conv —
     but the conv state the decode consumes lives in the WORKING slot 0 AND
     IS ZERO there in normal operation (zeroed at prefill end? or layer-0
     conv is unused) — i.e., the capture grabs state the decode never reads;
  C2 the restore's conv H2D writes the captured (nonzero) bytes into the
     WORKING slot 0 (or the copy_slot path re-propagates them), where the
     normal path has zeros.
**Fix once confirmed: mask conv components to zeros in the image (the
commit-ready patch docs/156_t2_capture_masking.patch) — BUT ONLY after the
window proves the conv slot is constant-zero in normal decode start; if the
normal conv state is real-and-nonzero, re-target the capture instead.**

**Next session (one window):** NINFER_HKV_DBG=1 (canonical 4096 scopes —
committed) + the §18.8-style probe extended to dump conv_slot(0, slot=0) AND
conv_slot(0, slot=1) separately (already in the binary) — the T2 verdict
falls out of the existing t2 window output: normal (slot0=5e951f89,
slot1=5e951f89) vs restored (slot0=377a3029, slot1=377a3029) — the restore
wrote 377a3029-content into BOTH slots ⇒ the ARENA's conv bytes are nonzero
⇒ capture-side. Fix = mask (patch ready). Then t6/t7 clean + literal.

### 18.13 CONV-L0 CORRECTION + FINAL NIGHT STATE (HEAD 39a39811)

**Scope-error correction (mine):** the earlier "conv l0 restore src = 77b793c7
≠ park 5e951f89" comparison was CROSS-SCOPE (restore probe = first 4096 B;
park print under NINFER_HKV_FULL = FULL component). Not comparable — the
capture may well be correct. What REMAINS valid (same-scope comparisons):
  * the conv l0 SLOT at decode start is CONSTANT (5e951f89) in every normal
    request (warmup + 4 different prompts) — likely all-zeros;
  * after the S1b restore it is 377a3029 (nonzero) in BOTH slots 0 and 1 —
    the decode then consumes it and diverges;
  * rec l0/l47, page0, ph restore correctly; page1's full-hash difference is
    the designed stale tail (masked by the envelope; S2b/S3b/S4b prove
    masked tails harmless).
**The nonzero restored conv is the divergence input; the writer is
unidentified.** Candidates: the conv H2D address (slot mix-up), a park-time
capture reading a modified slot, or an interaction with copy_slot.

**Next session, first act (one window):** with NINFER_HKV_DBG=1 (canonical
4096 scopes everywhere), dump conv l0 slot0 AND slot1 at (i) park time,
(ii) post-restore, (iii) decode start — plus the copy_slot probe. The
377a3029-content's first appearance pinpoints the writer. THEN the fix
(mask vs re-target per §18.9), t6/t7 clean, literal.

**Everything else in this port is validated and green:** P1/P2/P3/P4/P7/P8
landed; t2 golden ✔ (canonical probes, this window: S2b/S3b/S4b
byte-identical ✔); unit suite 134/134; the t6 S1-revisit regression is the
single open defect, precisely instrumented.

### 18.14 T2B WINDOW RESULT — CORRUPTION CONFIRMED, WRITER UNIDENTIFIED (HEAD 39a39811)

**Reproducible across runs (same 77b793c7 value = deterministic, not a race):**
entry 2's arena conv l0 region reads 5e951f89 (correct S1 conv) at park time,
but 77b793c7 at req5's restore time. All tracked arena writers verified
disjoint from entry 2's extent [149.3M..307.7M] (e3/e5/e7 extents offset-math
verified). rec l0 restored CORRECTLY from the same extent — so the extent is
not uniformly stale: the conv region changed, the rec region didn't.

**Conv-specific pattern:** in normal operation conv l0 slot 0 is CONSTANT
(5e951f89) across warmup + 4 different prompts at decode start — i.e., the
normal path's decode consumes a constant conv state (likely zeros) — and
S1b's restored conv = 377a3029 (nonzero, fluent-but-wrong output, early-small
divergence). The restore faithfully delivered the corrupted arena bytes.

**Next session — memory-guard pass (the wild-write hunter):**
  1. park(): after the capture sync, FILL the conv components in the arena
     with a magic pattern (0xC5) instead of trusting the D2H — if the restore
     still sees 77b793c7, the writer is POST-park (between req2 and req5);
     if it sees the magic, the capture's D2H itself strays.
  2. If post-park: the magic survives until req5's restore (verified), and
     S1b STILL diverges with conv = magic → the conv state is irrelevant to
     decode → the divergence cause is NOT the conv slot at all — re-examine
     the unfingerprinted tail (page1 slots 4-63) with per-slot hashes.
  3. compute-sanitizer initcheck/racecheck on a minimal repro if (2) is clean.
  DO NOT ship the masking patch: the corruption is real, deterministic, and
  outside the tracked writes — masking hides a memory-safety bug.

**Everything else stands:** P1-P8 landed; t2 golden + t5 + S2b/S3b/S4b
byte-identical ✔; 134/134 unit; the t6 S1-revisit divergence is the single
open defect; the literal (GO'd) runs after it.

### 18.15 T2c MAGIC-FILL RESULT + FINAL NIGHT STATE (HEAD = this commit)

**The magic-fill discriminator half-worked — with a rank asymmetry:**
  * rank 1: park-time conv hashes = FNV(0xC5×4096) = 32ca757a ✔ MAGIC
    PRESENT at park.
  * rank 0: tag-2 conv l0 park hash = **5e951f89 = the all-zeros constant**
    (NOT the magic) — yet the magic-fill writes 0xC5 unconditionally for
    LinearConv components, rank-agnostically. Rank 1 shows the magic; rank 0
    shows zeros. Either (i) rank 0's fill was zeroed afterwards, or (ii) the
    4096-scope reads differ in timing (fill enqueued on s, print reads host
    post-sync — should be fine), or (iii) a zeroing writer runs on rank 0
    only. UNRESOLVED — needs fresh-session investigation with the print
    extended to print rank + a post-sync re-read.
  * S1b's restored conv = 377a3029 in BOTH slots — matching NEITHER the
    zeros constant NOR the magic — a THIRD distinct content. The conv slot
    content is being written by something the image/capture/restore
    accounting does not track.

**VERDICT: the conv-region corruption is real, deterministic-ish, and NOT yet
writer-attributed. The masking patch (docs/156_t2_capture_masking.patch)
would ship a memory-safety bug. DO NOT APPLY. The t6 gate remains red until
the writer is named and eliminated.**

**Everything else is DONE and green:** P1-P8 landed; t2 golden ✔; t5 ✔;
S2b/S3b/S4b byte-identical ✔ (only S1 — the FIRST revisit after 2+
intervening sessions — diverges); unit suite 134/134; the literal
(3×200k @ i8, GO'd) runs after the fix; docs/155 flip at integration.

**Next session's first act:** extend [HKV-ARENA] to print rank + re-read the
conv region post-sync at park AND at restore (already sync'd); compare
rank-0-vs-rank-1 magic survival; then compute-sanitizer initcheck on the
minimal repro (2 sessions + 1 revisit @ 460 MiB arena). The bug is a wild
write into the pinned arena between parks — likely a stale pointer in a
later capture's D2H or an arena-region aliasing in the multi-run byte_view
walk for the SECOND-EVER multi allocation (e5/e7 pattern).

### 18.17 POST-FIX ONE-SHOT COMMANDS (staged — zero-thinking window per the coordinator)

In order, each a single launch (grants per the coordinator's queue):
  1. t6 clean re-run (scatter-gather + pressure rotation):
     `NINFER_HKV_DBG=1 bash tools/smoke/host_kv_validation_window.sh`   (~6 min)
  2. t7 — pinned-phase eviction: covered by (1) (the 460 MiB arena forces
     evict_for through pinned+unpinned phases; the [HKV-FENCE]/[HKV-GUARD]
     lines in the log are the t7 evidence). If a dedicated pinned-only run is
     wanted: MODE=stress with --host-state-slots 2 (forces slot-cap eviction
     of pinned... currently unpinned-only — the victim loop skips pinned; a
     pinned-phase test needs the fork's escalate-to-pinned behavior, §18.14).
  3. THE LITERAL (GO'd): `bash tools/smoke/host_kv_literal.sh`
     (3 × 200k @ i8, 12 GiB host arena; ~30-40 min GPU incl. 3 × 200k-tok
     prefills; PASS = all 3 revisits byte-identical + zero failures +
     host_kv_restore_failures=0). Driver: tools/smoke/host_kv_literal.sh —
     KV_CAPACITY tunable (200k default; 170k/150k fallbacks if the i8
     preflight refuses — post-weights budget is tight, see the driver header).
  4. C4 — gemini's host_kv_gate_ci.sh review (CPU-side, when it lands).

**Defect note:** the literal assumes the t6 S1-revisit divergence is FIXED.
It is not yet — the wild-write hunt (§18.15 gdb window) names the writer
first; then the fix; then this sequence.

### 18.18 t6 ROOT CAUSE NAMED AND FIXED — SCATTER-RUN-BOUNDARY STRADDLE (agent2, CPU-only session; HEAD = this commit)

**The wild writer is identified without a GPU window.** It is OUR OWN capture
D2H — no foreign writer, no allocator accounting error, no masking question.
The masking patch (docs/156_t2_capture_masking.patch) stays dead, as ruled.

**Mechanism (every number below is log-derived from the surviving 14:48
window, /tmp/host_kv_val/serve_rotation.log — rank-0 and rank-1 ARENA/RESTORE
prints plus the [host-kv] event stream):**
  1. Entry 2 (S1a's image, single-run) occupies arena [156583936, +151 MiB).
  2. Entries 5 and 7 (151 MiB images each) park after entry 1 is evicted;
     largest-span-first scatter carves entry 1's freed hole as their run0 =
     [0, 156583936) — run0's end is EXACTLY entry 2's start, and the 1.67 MiB
     remainder goes to run1 in the arena tail. The extents are DISJOINT —
     the earlier offset-math verification was correct and yet irrelevant.
  3. byte_view(allocation, off) resolves ONE pointer; the capture issues one
     cudaMemcpyAsync per PAGE. The run boundary (image byte 156583936) falls
     MID-COMPONENT: layer 6's PagedCommittedK component spans image
     [156499968, 156631040); its page-1 copy = image [156565504, 156631040),
     dst = run0_base + 156565504. A CONTIGUOUS 65536-byte copy starting
     inside run0 overshoots its end by 156631040 − 156583936 = **47104
     bytes — written straight into entry 2's head**, i.e. the conv l0/l1
     region (each slot 30720 B).
  4. Probe arithmetic predicts EXACTLY the observed fingerprint pattern:
     conv l0 probe [0,4096) ⊂ [0,47104) corrupted ✓; conv l1 probe
     [30720,34816) ⊂ corrupted ✓; conv l2 probe [61440,65536) clean, magic
     intact ✓ (matches the restore print: idx0 = 77b793c7…, idx1 = 12b28960…,
     idx2..47 = magic 32ca757a…). The wild content = S4a's (entry 7 = last
     writer) layer-6 K-page bytes — rank-sharded, hence rank0 77b793c753a7f890
     vs rank1 669ce06f78e3d188; deterministic prompts, hence the identical
     values across t2b/t2c/14:48 runs. Restore delivered them faithfully
     (src==dest); decode consumed a wrong conv state; S1 diverged.
  5. Everything previously observed falls out: t2/t5/6144 pass (no eviction →
     no scatter → no straddle); only the pressure rotation fails; "conv l0
     only" was the 4096-probe granularity — l1 was corrupted too, l2 not.

**Collateral findings:**
  * The mirrored READ bug: restore's H2D read side had the same straddle —
    a scatter image's straddling component restores garbage EVEN IF UNCORRUPTED
    (reads past run0's end into the next extent). Entry 2 itself was single-run
    ("alloc runs 1" in the log), so t6's failure is capture-side only — but a
    clean t6 REQUIRED fixing both directions.
  * Entries 5/7's own images carry a 47104-byte hole (the straddling page's
    tail never reached run1). Had they been restored, they would diverge too.
  * The debrief's hash quotes are HIGH-32 of the 64-bit FNV (32ca757a =
    32ca757afbe2d383), not low-32. The "rank 0 park-time zeros" anomaly in
    §18.15 does not reproduce in the surviving log: BOTH ranks' tag-2 park
    prints show the magic intact — the stomp happened LATER (entries 5/7
    captures), between park and restore, as the mechanism requires. The t2c
    run's log was overwritten by the 14:48 run, so that one line is moot.
  * AllocGuard leak (pre-existing): the error-path guard held a POINTER and
    the catch reset out.allocation first, so valid() was false and a failed
    capture LEAKED its extents. The guard now owns a copy and actually frees.

**THE FIX (this commit, CPU-validated):**
  * HostKVArena::for_each_span(allocation, byte_index, bytes, f) — splits a
    byte range at run boundaries and invokes f(host_ptr, span_bytes) per
    contiguous piece. Hosted in the header, CUDA-free, unit-testable.
  * capture d2h, the conv magic-fill memset, and restore h2d all route
    through it (device-side pointer advanced per span; error from any span
    surfaces via check_cuda).
  * Unit: tests/core/test_host_kv_arena.cpp test_for_each_span_straddle —
    the t6 geometry miniaturized (hole-largest-first run0 ending at a live
    sentinel extent; straddling write splits 2048|2048; neighbor sentinel
    byte-identical; read side reconstructs the payload). Full suite PASS
    (pageable arena, no GPU context needed).
  * Probes ([HKV-RESTORE]/[HKV-ARENA]/[HKV-GUARD] 4096-byte reads) left as-is:
    a straddling probe can misread but cannot corrupt.

**Next window (supersedes §18.17's zero-thinking sequence):**
  1. `NINFER_HKV_DBG=1 bash tools/smoke/host_kv_validation_window.sh` —
     EXPECT NOW: t6 ALL FOUR revisits byte-identical; every restore's conv
     src = magic (no 77b793c7/669ce06f); park/restore counts unchanged.
     If ANY conv src ≠ magic: the fence pass (NINFER_HKV_GUARD=1, §18.15)
     is STILL the next step — but the straddle explained every observation,
     so failure is not expected.
  2. t7 + gate re-runs per §18.17 (covered by the same driver).
  3. THE LITERAL (3×200k @ i8) per §18.17 step 3 — the defect note above is
     now satisfied by this fix, pending the window's confirmation.

### 18.19 VALIDATION WINDOW (22:28) — STRADDLE FIX VERIFIED; T2c DIAGNOSTIC RETIRED (9b96fc74)

**Cell 1 of the granted sequence (NINFER_HKV_DBG=1 validation window) ran;
rotation exit=1 → STOP exercised per the grant (no t7/gate/literal cells).**

**The §18.18 straddle fix is VERIFIED by the window:**
  * 396 [HKV-RESTORE] lines; ALL 192 conv-component restores read
    src == dest == 32ca757afbe2d383 (the magic) — ZERO non-magic conv srcs.
    The wild content (77b793c7/669ce06f) never appears. Arena integrity is
    restored under the exact pressure geometry that used to corrupt it
    (10 evictions, scatter runs adjacent to live extents).
  * rec restores are CORRECT: S1b's decode-start rec l0 = 4e48243b… ==
    S1a's; S2b's = 779561f9… == S2a's — real captured state, faithfully
    delivered through the same extents that used to be stomped.

**The remaining rotation failure is the T2c DIAGNOSTIC ITSELF** (still in the
binary at window time): it replaces the conv CAPTURE with a 0xC5 fill, so
every RESTORE faithfully delivers magic as the conv state. Fingerprint table
(rank 0, decode start):
  * normal + all first visits:            conv l0 = d90dc2b4… (constant)
  * S1b (restored tag 2):                 conv l0 = 32ca757a… = MAGIC, rec correct
  * S2b (restored tag 3):                 conv l0 = 32ca757a… = MAGIC, rec correct
  * S3b/S4b (images evicted → full prefill): conv l0 = d90dc2b4… → byte-identical ✓
Event-stream mapping (log): S1b restored#2, S2b restored#3; tag7/tag9 evicted
→ S3b/S4b prefix-missed → clean full prefills → passed. S1b/S2b outputs are
degenerate from token 0 ("Weacco isted …", same wrong prefix) = magic conv
state consumed by decode. Consistent with pre-T2c history: S2b restored with
REAL captured conv and passed byte-identical (§18.10 original t6) — real conv
is proven good; magic conv is proven fatal; the old fluent-but-wrong S1b was
the wild stomp, now fixed.

**ACTION TAKEN (9b96fc74, pushed, binary rebuilt): T2c magic-fill RETIRED —
the capture D2H for LinearConv is restored (through for_each_span). The
diagnostic is retained in the trail (§18.14/§18.15) and can be re-lit by
reverting that one hunk if a wild-write hunt is ever needed again.

**NEXT WINDOW (pending coordinator ack — grant's STOP clause honored):**
re-run cell 1 only. EXPECT: all four revisits byte-identical; conv src at
restore = real captured content (S1b decode-start conv back to d90dc2b4…,
NOT magic); then, on green, t7/gate → literal per the original sequence.

### 18.20 RE-RUN WINDOW (22:50) — t6 CLOSED: 4/4 BYTE-IDENTICAL + GATE PASS (cell 1 + t7/gate green; literal pending ack)

On the coordinator's re-issued grant, cell 1 re-ran @ 9b96fc74's binary
(T2c retired):
  * ROTATION (t6 + t7 in one, per §18.17): **ALL FOUR revisits
    byte-identical** (1646/1741/1596/1462), 10 evictions, 4 restores,
    0 failures, exit=0 — the exact geometry that failed every window since
    §18.10. t7's pinned-phase coverage rides the same run (4 restore-window
    pins interleaved with the 10 evictions).
  * GOLDEN GATE t2: PASS (A→B→A' byte-identical, gate exit=0).
  * FINGERPRINT EXPECTS, all met: conv l0 restore src = d90dc2b4… (real
    captured content — not magic, not wild); conv l1 = ce29f731… real;
    0/392 src≠dest across ALL restores; all 9 decode-start dumps show the
    normal constant. Magic content: zero occurrences.

Defect t6 (docs/155 item 3's blocker) is CLOSED at the rotation level; the
literal (3×200k @ i8) is the completion bar and awaits the coordinator's ack
on this report (grant: "STOP and report before literal"). Cards held for the
same window; guard re-check before the literal launch.

### 18.21 LITERAL RUN (23:00) — BYTE-IDENTITY PASS; ENGAGEMENT BAR NOT MET (generator gap; coordinator to rule on re-run)

tools/smoke/host_kv_literal.sh completed: **LITERAL: PASS (all revisits
byte-identical)** — S1a/S2a/S3a (91237/91201/91216 prompt tokens) vs
S1b/S2b/S3b: IDENTICAL/IDENTICAL/IDENTICAL, temp 0, i8 tier, 12 GiB-class
arena, 0 failures/refusals/OOM.

**HOWEVER — the §18.4 engagement bar is NOT met by this run, and the PASS
must not be oversold:** the host-KV tier never carried a 91k session.
Evidence: exactly 2 parks, both 53-token ledgers (148 MiB, dtype 5, warmup-
adjacent); 0 arena evictions; 0 restores. All three revisits passed through
the DEVICE-level prefix path.

**Root cause of the gap: the prompt generator's chars-per-token heuristic**
(n_words = cap × 4.6 / max-word-len, loop to n_words × 7 chars) undershot —
495k chars ≈ **91k real tokens** (~5.43 chars/token), ~2.2× under the 200k
target. 3 × 91k = 273k tokens of demand vs the 200k-token device pool: the
revisit phase had only 182k co-resident (S1a+S2a), so the device prefix
absorbed everything; the admission trigger fired only for warmup-adjacent
tiny states. The driver's PASS criterion is byte-identity only — §18.4's
"spill/evict/restore evidence + fallbacks == 0" needs the host tier to
actually engage.

**Proposed re-parameterizations (coordinator's call, no re-run attempted —
the no-retry-without-me clause is honored in spirit):**
  (a) cheapest: KV_CAPACITY=100000 (same prompts) → 273k ≫ 100k forces
      park/restore on every revisit; one env var, zero code;
  (b) faithful: fix the generator to tokenize-count (tokenizer-aware
      target), keeping the true 3×200k shape (~40 min);
  (c) both: (b) merged with (a)'s pressure as a stress variant.
Byte-identity at 91k×3 under temp 0 with real restores is ALSO already
proven by the §18.20 rotation, so (a) isolates exactly the missing
engagement evidence.

### 18.22 OPTION (a) RUN (23:15) — BYTE-IDENTITY PASS ×2; ENGAGEMENT STILL ABSENT — THE LEVER IS CONCURRENCY, NOT CAPACITY

KV_CAPACITY=100000 (granted option a): sessions came out 45167/45332/45118
tokens (135k total demand vs 100k pool — real pressure by the numbers).
Result: **LITERAL: PASS byte-identical ×3 again — but the engagement bar is
STILL not met**: exactly 2 parks, both 53-token warmup-adjacent ledgers
(148 MiB, dtype 5); 0 arena evictions; 0 restores; stats 296/14336 MiB,
entries=2/16. Capacity was NOT the lever.

**Why the trigger never fires in this shape (first-pass analysis, to be
confirmed against tp2_backend ~1395-1490):** the driver sends the three
first-visits SEQUENTIALLY. At each admission the lane's previous session has
already COMPLETED and its cached state cleared — the admission-time park
trigger (H3: park the lane's occupied state when a new session needs the
lane) finds nothing to park. The 460-MiB rotation parked/restored precisely
because its request shape kept sessions interleaved on the lane (plus arena
pressure). §18.4's own wording is "3 CONCURRENT 200k sessions" — the driver's
sequential for-loop never creates that condition. Falsification predicted by
this model: concurrent first-visits → parks with ~45k ledgers appear.

**Proposed next (coordinator's call):** (i) CPU: read the admission trigger,
pin the exact firing conditions against this analysis; (ii) driver: send the
3 first-visits CONCURRENTLY (background + wait), then the 3 revisits —
matches §18.4's wording; then the same KV_CAPACITY=100000 run. The
byte-identity dimension is now PASS under three separate parameterizations;
only the engagement evidence is missing.

Window state: cards RELEASED + verified (15 MiB / 0 apps) after this run.

### 18.22-bis TRIGGER FIRING CONDITIONS PINNED — CONCURRENCY THEORY FALSIFIED; REAL GATE = prefix_cache_capacity (§18.21 addendum per coordinator seq 71)

Trigger (tp2_backend, admission, lane 0): park fires iff
  st.cache_valid && !st.cached_tokens.empty() && !old_retained &&
  (better_match || st.gdn_ckpt_token_count > 0),
where live = match_len(new prompt, lane ledger); old_retained = live >= ledger;
gdn_ckpt_token_count is set (>0) ONLY inside the chunked-prefill loop when
req.prefix_cache (checkpoint one chunk before the end).

**The whole state that feeds this trigger lives inside
`if (plen <= P)`, P = min(prefix_cache_capacity, max_context) — server
default 16384.** The literal's 45k/91k sessions exceed P: no GDN checkpoint,
cached_tokens never set, cache_valid false → the trigger has nothing to park
and the revisits full-prefill. Log proof (both literal runs): exactly ONE
"GDN checkpoint captured @51 tok" (warmup), ZERO "prefix hit" lines, zero
big-ledger parks. The 460-MiB rotation engaged because its 101-token
sessions sat under the cap.

**Concurrency theory (§18.22) FALSIFIED by my own (i):** the rotation parked
with SEQUENTIAL requests — the trigger fires per-admission given a
checkpointed lane state; concurrency is unnecessary for engagement on the
single-seat arm (client-side concurrency serializes at the lane anyway).
The real lever: raise --prefix-cache-capacity ≥ the session token count.

DRIVER FIX (this commit): PREFIX_CAP env on host_kv_literal.sh →
--prefix-cache-capacity. ph buffer cost = P × 10 KiB/rank (50000 → ~500 MiB,
+340 over default) — if the i8 preflight refuses at KV_CAPACITY=100000,
drop KV (sessions are 45k; 80k keeps 3×45k ≫ pool). PREDICTION: parks with
~45k ledgers + evictions + restores on every revisit; if parks STILL don't
fire with P ≥ plen, the admission trigger itself is the defect → stop+report
(coordinator's clause).

### 18.23 PREDICTION-TEST RUN (23:34, PREFIX_CAP=50000) — ENGAGEMENT ACHIEVED; RESTORE FIDELITY AT 45k DIVERGES → STOP

KV_CAPACITY=100000 + PREFIX_CAP=50000 (driver fix 5969d76b). Preflight
accepted (context 2850 MiB, slack 1401 MiB, GDN ckpt buffers 73.4 MiB/rank).
Sessions 45230/45168/45213 tok.

**PREDICTION CONFIRMED — the tier engages:** GDN checkpoints captured for all
3 sessions (@~45k, not just warmup @51); **12 park lines = 6 parks with
45230/45230/45168/45168-token ledgers** (2 ranks); **6 restore lines = 3
restores, 0 restore-failures, 0 refusals/OOM**. The §18.21/§18.22-bis
analysis was right: the prefix-cap gate was starving the trigger.

**BUT: S1b/S2b/S3b ALL DIFFER (LITERAL: FAIL) — mismatch → STOP per grant.**
The outputs are FLUENT-BUT-WRONG (plausible answers to a slightly different
reading — not garbage: conv is real content now, not magic). So the restore
delivers subtly-wrong state at 45k scale. This run had NO fingerprint probes
(NINFER_HKV_DBG not set — my launch omitted it), so the diverging component
is NOT identified. Candidate fidelity gaps for the diagnostic window:
  (a) paged capture/restore at 707 pages × 16 layers × K/V per image
      (~22.6k copies/park) — per-page phys-table fidelity;
  (b) ph slice at 460 MiB/component (45230 × 5120 × 2 B) — slice/take
      semantics at scale;
  (c) decode-advanced tail: parks happen post-decode (400 tok generated);
      page-707 slots [46..64) hold decode KV (stale-tail — masked at 101-tok
      scale, unverified at 45k); conv/rec slots at park = post-decode state
      vs ledger = prefill-end (should be repaired by the gdn-checkpoint
      re-prefill [45184..45230), IF the GdnCkpt components are carried and
      restored — enumerate_fixed pushes them (vector populated), and the
      restore H2D covers all fixed components, but this is UNVERIFIED at
      the fingerprint level);
  (d) i8-tier specifics (dtype 5): kvarn_ws tails/tile_page scalars vs the
      bf16-arm paths the rotation validated.
NOTE: my §18.22-bis cross-session-checkpoint-contamination hypothesis was
WRONG in one respect — enumerate_fixed DOES carry the GdnCkpt tensors (the
129-component count from the rotation was a PRINT-FILTER artifact: the
[HKV-ARENA] print omits GdnCkpt kinds). Corrected here for the record.

**NEXT WINDOW (proposed, needs grant): re-run this exact shape with
NINFER_HKV_DBG=1 NINFER_HKV_FULL=1 — the restore probes + decode-start
fingerprints name the diverging component (same discipline that closed t6).
Cheaper variant: 3 × ~20k sessions @ PREFIX_CAP=25000 (same code path,
~4× less GPU).**

Cards RELEASED + verified (15 MiB / 0 apps) — server down, driver finishing.

### 18.24 FULL-FIDELITY DIAGNOSTIC WINDOW (23:51) — RESTORED FIXED-STATE EXONERATED; DIVERGENCE AT TOKEN ~10; SEED CONFOUND SURFACED (coordinator's not-named branch)

Same failing shape + NINFER_HKV_DBG=1 + NINFER_HKV_FULL=1. Result: S1b/S2b
DIFFERS, **S3b IDENTICAL** (mixed — a discriminator the all-red runs lacked).

**Probe coverage (what the window CLEARED):**
  * [HKV-RESTORE] fidelity: **0/588 src≠dest** across 3 restores — every
    probed component (all 96 conv/rec slots, ph, pagedK l0 pages 0-1)
    delivered exactly what the arena held.
  * Decode-start [HKV-STATE] (FULL scopes): S1b == S1a, S2b == S2a,
    S3b == S3a **bit-identical** on l0/l47 conv+rec, ph, t0 (7 groups
    extracted). The restored lane state is, as far as probes reach,
    PERFECT — including the GDN-checkpoint re-prefill path (a wrong
    checkpoint would perturb the re-prefilled slot state; it doesn't).

**The divergence signature:** S1a/S1b first differ at **byte 40 ≈ token ~10**
— tokens 1-9 IDENTICAL, then a flip ("a huge" vs "a long"). Not
first-token, not garbage: either (i) a specific page/region among the
UNPROBED restore surface (deep pages, layers ≠ 0/15, pagedV, i8 scale/tail
components, GdnCkpt bytes — none fingerprinted) perturbing attention
mid-stream, or (ii) sampling/numeric jitter crossing an argmax boundary.

**SEED CONFOUND (prime suspect for (ii)):** the literal driver never pins a
seed (server `--seed` unpinned → random per request, translate.cpp:47-52),
and the multibatch lane's S-cell protocol established that byte-reproducibility
gates REQUIRE a pinned seed even at temp 0. S3b passing unpinned is
consistent with per-request seed luck. The §18.20 rotation's 4/4 unpinned is
the counterpoint (4/4 by luck is possible but less comfortable).

**Two discriminators, coordinator's call:**
  (A) cheapest: re-run with `--seed` PINNED (server arg via the driver, one
      line). Green ⇒ the literal driver needed seed pinning like every other
      byte gate (S-cell protocol); the host-KV tier is exonerated at 45k and
      the literal bar is closable. Still red ⇒ (B): deepen the probes
      (random page/layer sampling incl. pagedV/scales/GdnCkpt) — the
      unprobed-surface hunt proper.
Cards RELEASED + verified (15 MiB / 0 apps). Driver: still no retry without
you.

### 18.25 (A)+(B) OUTCOME — (A) RED (seed falsified); (B) 9992/9992 FAITHFUL → DEFECT IN CAPTURE SEMANTICS / TAIL-FLOW LAYER

(A) seed-pinned literal (34f90348, --seed 20260905): **RED, pattern unchanged**
S1b✗ S2b✗ S3b✓, divergence byte 40/byte 15. H1 falsified — identical restored
state + identical seed + different output = real infidelity, not RNG.

(B) deep-probe window (2a995b9b): SAME shape, 3/3 DIFFERS (bytes 40/15/40).
Probe extension: layer-0 EVERY page K+V, first+last page of every plan,
full-byte GdnCkpt components. Result: **9992/9992 src==dest — zero restore
fidelity defects at any observable scope.** Decode-start [HKV-STATE] again
bit-identical ×3 pairs (l0rec/ph/t0). Committed-page accounting exact
(ceil(ledger/64), no post-decode page leak).

**Localization (by elimination):** restore delivery ✓, fixed-state ✓,
page-accounting ✓ → the defect is in CAPTURE SEMANTICS or the POST-RESTORE
DECODE FLOW. Prime suspect: the i8-tier (dtype 5) kvarn tail/open-page
state — parks happen post-~400-decode, so the image's tail tiles + tail
scalars are the parked session's POST-DECODE tail; the post-restore decode
commits into that geometry differently than after a fresh prefill
(tokens 1-9 identical then flip = tail-geometry divergence signature).
NOT yet fingerprinted: the tail tiles themselves + tail scalars at decode
start (S1a vs S1b) and pre-park vs post-restore.

**PROPOSED (awaiting coordinator): tail-state probe window — dump kvarn
tail scalars + open-tile hash at decode start S1a-vs-S1b and pre-park vs
post-restore; names the tail-layer defect directly. Cards free, branch
@ 2a995b9b, all pushed. Protocol note: the (A)-verdict report was owed at
the (A)-red moment and went out late — night-shift no-silent-gaps rule
re-armed (every cycle: report or holding line).**

### 18.26 TAIL-PROBE RESULT — TAIL HYPOTHESIS FALSIFIED; BISECTOR PROPOSED (gdn-skip vs pool-content)

The approved tail-state probe returned a DECISIVE negative:
**kvarn_ws.text is EMPTY on the i8 tier** — [HKV-TAIL] printed the empty
branch for all 7 decode-start groups; [HKV-TAIL-CAP] zero lines (guard).
There IS no kvarn tail state to poison — candidate (d) dead. (Also resolves
the pool-shape confusion: the i8 pool's ARENA print shows 16 pool layers
(16 pagedK + 16 pagedV components for a 45k session); the "paged 706 plans"
park-line field does NOT count restore-relevant components — the image's
paged surface is 16 layers, fully probed faithful in §18.25.)

Elimination state after this window:
  restore delivery ✓ (9992+588+392 probes) | fixed-state at decode start ✓
  (bit-identical ×3 pairs) | page accounting ✓ | seed ✓ pinned | tails N/A.
Yet outputs diverge from ~token 10 with bit-identical decode-start state.

**REMAINING SPLIT — flow-level, two halves, cleanly bisected by one
env-gated line:** force `st.gdn_ckpt_valid = false` on the restore path
(NINFER_HKV_NO_GDN_SKIP=1): the revisit then re-runs the GDN layers over the
FULL restored prefix from the restored pool pages instead of trusting the
checkpoint + [ckpt..ledger] re-prefill.
  * GREEN under the bisector ⇒ the defect is the GDN-SKIP FLOW (the
    [ckpt..ledger] re-prefill after restore — envelope/position/commit
    semantics in that window; tokens 1-9 fine then flip fits a boundary
    effect inside the re-prefilled tail region).
  * STILL RED ⇒ the defect is POOL-PAGE CONTENT at positions the probes
    never covered (middle pages of layers 1-15 — only l0-every and
    first/last-per-layer are fingerprinted) ⇒ extend probes to all-layer
    full-page sampling.
Awaiting coordinator call on the bisector (one line + one window).

### 18.27 (B) FULL-COVERAGE OUTCOME — PAGED SURFACE + GdnCkpt EXONERATED; CONV/REC PROBE CONTRADICTION OPEN; DEFECT NOT NAMED

Full-coverage window (cf345da6): every page of every layer, K+V, 1024-byte
scopes + GdnCkpt full-byte. Result: **136,384 paged probes + 576 GdnCkpt
probes: 0 mismatches** — the restored pool pages and the restored GDN
checkpoint are byte-perfect at complete coverage. Outputs: S1b✗ S2b✗ S3b✓
(LITERAL: FAIL — divergence positions bytes 15/40/none).

**OPEN CONTRADICTION (instrument-level, logged honestly):** the conv/rec
fixed-component probes mismatched 576/576 at 1024-byte scope — with dest
values CONSTANT across tags (0ca2e0a7… for conv l0 idx=0 on every restore)
— while the [HKV-STATE] decode-start dumps (4096 scope, later in the same
requests) show slot0 == slot1 == the CORRECT per-session restored state,
bit-identical to the first visits, in the SAME run. Both cannot describe the
same memory: either the conv/rec probe reads a stale/wrong location (probe
artifact — the scope change interacts with something), or the slots were
written correctly only AFTER the probe (by the flow's gdn_from_buffer
re-prefill, which recomputes conv/rec for [ckpt..ledger] from the restored
checkpoint — making the conv/rec H2D a dead write on this path and the probe
a pre-rewrite snapshot; but then dest should vary per-tag with the pre-restore
lane state, and it doesn't). Unresolved — needs code-read of the
gdn_from_buffer flow, not more probes.

**WHERE THIS LEAVES THE DEFECT:** every restored byte probed is faithful;
decode-start state is bit-identical; outputs diverge deterministically from
~token 10 with the seed pinned. The divergence input is in the DECODE FLOW's
use of state (post-restore commit path / envelope / cursor arithmetic), not
in any restored byte. Next instruments (coordinator's call):
  (i) per-step divergence localization: NINFER_HKV_STEP_HASH-style probe —
      hash logits at every decode step for S1a and S1b, find the exact step,
      then dump that step's envelope/cursor/page-table registers;
  (ii) CPU code-read of the gdn_from_buffer re-prefill path (free, no window);
  (iii) the NINFER_HKV_NO_GDN_SKIP bisector — NOTE: as written it is a NO-OP
      for full-ledger matches (the skip block requires prefix_len <
      cached_tokens.size(); an exact-match revisit skips it entirely) — it
      would need re-shaping (e.g., force prefix_len = 0) to engage at all.
Cards RELEASED + verified (15 MiB / 0 apps). Branch @ cf345da6.

### 18.28 (ii) CODE-READ COMPLETE — ROOT CAUSE NAMED: THE OPEN TILE IS NOT CAPTURED (i8 kvarn-packed tier)

Scope-pair correction first: the §18.27 conv/rec "576 mismatches" were MY
probe bug (dest scope changed to 1024, src left at 4096 — scope-pair
artifact). The restore is byte-faithful for conv/rec too. No dead-write
mystery.

**ROOT CAUSE (mechanism, fully consistent with all runs):** the i8 tier is a
kvarn PACKED pool — codes commit to pool pages only when a 64-token tile
closes (gqa_kvarn_commit_completed); the open page's live tail (bf16, up to
63 tokens) lives in the TextContext SEQUENCE workspace tile
(kvarn_text_ws_ — NOT st.kvarn_ws.text, which is the kvarn-attention arm's
list and is empty on this arm — why [HKV-TAIL] printed EMPTY: right fields,
wrong struct).
  * park() captures ceil(ledger/64) pool pages. The LAST page is the OPEN
    tile's page: the capture reads the physical page's STALE
    previous-occupancy codes, while the live tail codes sit uncommitted in
    the sequence-workspace tile — **the image never carries them.**
  * The image's tail scalars (text_tails/text_pages) come from the same
    empty struct → zeros/-1; restore's scalar phase is a vacuous dead write
    (0 == 0 passes).
  * Post-restore kvarn_rewind_text(ledger) sets committed=705, tile=705,
    tail=52 ON A WORKSPACE TILE HOLDING THE PREVIOUS LANE OCCUPANT'S
    GARBAGE → the revisit's attention dequantizes the last ~52 positions
    from garbage → subtle logit shift → argmax flips at ~token 10 →
    fluent-but-wrong. Tokens 1-9 survive (small contaminated attention
    mass). S3b's occasional passes = run-dependent full-prefill/parking
    variance.
  * The BF16 rotation never hit this: the bf16 pool has NO tile boundary —
    decode appends bf16 directly into the pool page, so the capture reads
    the live tail for free. The i8 literal is the first shape that exposed
    the tile boundary under the host-KV park.

**FIX DESIGN (awaiting ack): carry the open tile in the image.**
  Preferred: small TextContext API — kvarn_capture_open_tile() /
  kvarn_restore_open_tile() wrapping kvarn_text_ws_ (per-layer tile_page/
  tail_count + bf16 tile bytes ≈ 25 MiB/image at 48 layers) — wired as new
  image components + a REAL scalar restore path (not the vacuous
  st.kvarn_ws.text one).
  Alternative: force-commit the open tile at park — INSUFFICIENT ALONE:
  the post-restore rewind RE-OPENS page 705 as the working tile (committed
  = token_count/64, tile = page, tail = slot), and the attention serves the
  tail from the workspace tile — so the tile content must be carried
  regardless; commit-at-park can optionally complement it.
Validation: same literal shape (PREFIX_CAP=50000, KV_CAPACITY=100000, seed
pinned, DBG+FULL) — expect 3/3 byte-identical WITH big-ledger parks/restores
present.

**Also dissolved:** the §18.27 conv/rec probe contradiction (scope-pair
artifact). The NINFER_HKV_NO_GDN_SKIP bisector remains a no-op for
full-ledger matches (documented §18.26) — moot now.

Cards RELEASED + verified (15 MiB / 0 apps). Branch @ a9e31adc + this
commit. The literal's engagement bar is ACHIEVED (parks/restores fire); the
byte-identity bar is blocked on this fix.

### 18.30 FIX VALIDATED — LITERAL PASS WITH ENGAGEMENT: 3/3 BYTE-IDENTICAL, RESTORES EXERCISED, SCALES FAITHFUL (f6b68590)

Validation window (grant pre-issued): PREFIX_CAP=50000, KV_CAPACITY=100000,
seed pinned, DBG+FULL, sessions 45215/45218/45215 tok.

**LITERAL: PASS — ALL THREE REVISITS BYTE-IDENTICAL, WITH ENGAGEMENT:**
  * parks 12 (6 images × 2 ranks, 45k-ledger class), restores 6 (3 × 2
    ranks), restore-failures 0, fallbacks 0, refusals/OOM 0.
  * arena occupancy 13452/14336 MiB (94%) — heavy packing, no eviction
    needed (12 entries ≤ 16 slots).
  * **Scale-plane fidelity: 0/384 pagedKS/pagedVS probes mismatched** (the
    newly-carried surface), within 137,094 total restore probes, 0
    mismatches overall.
  * outputs byte-identical per session, temp 0, pinned seed.

**THE LITERAL BAR (§18.4) IS CLOSED**: byte-identity ✓ + spill/restore
evidence ✓ + fallbacks == 0 ✓ at the 45k×3 i8 shape with the packed-pool
tier actually exercised end-to-end through park → arena → restore →
dequant-correct decode.

Fix recap (f6b68590): the i8 group-scale planes (k_scale_pages/v_scale_pages,
pool planes base+2/base+3, FP16 [head_dim/qg, 64, heads, pages]) are now
captured/restored as a PagedCommittedScales component per layer (K-scale
pages || V-scale pages concat), with kind/bytes geometry validation on the
restore path and 1024-byte fidelity probes (pagedKS/pagedVS) under DBG.
The workspace-tile-bytes half of the approved design is covered by the
EXISTING hydrate path (gqa_kvarn_prepare_page_for_append →
gqa_kvarn_hydrate_page dequantizes the restored committed page into the
workspace tile at the first decode append, slot≠0) — empirically confirmed
by this green run; a belt-and-braces tile-bytes capture remains available
as follow-up if a future tier lacks the hydrate path.

Cards RELEASED + verified (15 MiB / 0 apps). The literal is closed; the
merge sequence (A1 port-commit review → merge → build+ctest → C4 → 2nd
push) is unblocked.
