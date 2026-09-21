# HANDOFF — host-KV safety net lane (sessions 2-3, 2026-09-07)

## SESSION 4 POINTER (agent2, CPU-only — READ FIRST; supersedes the defect state below)

t6's wild writer is NAMED and FIXED without a window: our own capture D2H
straddles a scatter run boundary and overshoots 47104 bytes into the next live
extent (entry 2's conv head). Fix = HostKVArena::for_each_span split-copies on
capture d2h / magic-fill / restore h2d (+ AllocGuard leak fix + straddle unit
test, suite PASS). Branch @ 3582de93, pushed. Full chain: docs/156 §18.18.
NEXT WINDOW: the §18.17 sequence now VALIDATES the fix (expect t6 all-four
byte-identical, conv src = magic) → t7/gate → literal. gdb hunt = contingency.
Resume point. Verified against the pushed state (`wo/host-kv-safety-net` @ f72ec77d,
github remote — note: the `origin` remote path
`/home/intel/comfy_templates/v340l_optimization/backups/ninfer.git` NO LONGER EXISTS;
push via `git push github wo/host-kv-safety-net`).

## SESSION 3c — DECISION B EXECUTED: PORT SCOPE POSTED, P1+P2 LANDED

Coordinator decision: B — fork-complete page-to-host, no v1/v2 framing.
docs/156 §18 = the port scope (fork F1-F6 inventory verified in code,
have-vs-port table, P1-P8 plan with files/tests, i8-bound literal: 3×200k @
i8, 12 GiB host arena — awaiting USER sign-off). Key fork finding: restore
fires only on cold-start materialization — the fork has NO host-tier
attention; 200k needs tier compression, not host-attend.

LANDED: P1 (HostKVArena page-run allocation: allocate_pages/allocate_multi_
pages largest-span-first w/ rollback + free_pages + page_view; CPU unit
green incl. fragmentation/rollback) + P2 (PagedKVPool host_page_layout +
copy_pages_to_host/from_host, page-major host mirror, per-page phys-id
scatter; PageMajor only; CUDA roundtrip unit PENDING a window).

NEXT SESSION: P3 (safety-net upgrade — frontiers/session-keys/pin-ID in
host_kv_parked) → P4 (spill/restore integration in tp2_backend, take_pinned/
re-add replaces drop-on-restore) → P5-P8 per §18.6. P2's CUDA roundtrip unit
joins the next GPU window. CI cells (P8): run_ci.sh --full battery at
tools/ops/run_ci.sh:530+; add host_kv_gate.sh cell + batched-rotation
(L"M", conc=2). Fork reference: /tmp/gzenz_ninfer @ 4b882d0 — PRESERVE IT.

## SESSION 3c FINAL (night push complete, HEAD ea0b60ff)

§18.7 progress ledger: P1/P2/P3/P4-single-seat/P7/P8-gate ✅; P5 merged into
P6; P6 recon + SOUNDNESS CONSTRAINT posted (§18.8/§18.9 — session-key restore
without checkpoint-frontier gating corrupts silently; requirements (a)-(d)
listed). Validation driver READY: tools/smoke/host_kv_validation_window.sh
(t6/t7 + t2/t3/t5 re-runs @ --host-kv-mib 460; one command). i8 literal
awaiting user sign-off. CI cells live in run_ci.sh --full.

## SESSION 3b — V2 SCOPE SET (operator directive: "scope from the fork, start immediately")

docs/156 §17 + §17.5 now carry the v2 scope, derived from gzenz's CODE
(/tmp/gzenz_ninfer @ 4b882d0, still on disk — preserve it). Key findings:
the fork's restore fires ONLY on cold-start (Root) materialization — it has
NO host-tier attention; the 200k mission on our hardware = tier-compressed
pools (k4v2/nvfp4: 200k ≈ 1.1-1.4 GiB fits post-weights) + fork-style owner
eviction. Revised build order: **W5 (tier+capacity config) → W3
(session-key match) → W2 (mid-turn checkpoint spill) → W1 (batched-arm
planner — Budget is static VRAM, real pressure = multi-owner, blocked on
batched net) → W4 (scatter-gather extents on demand).**
NEXT SESSION STARTS AT W5. Preserve /tmp/gzenz_ninfer.

## SESSION 3 ADDENDUM (read this first — supersedes the "H3 next" below)

H3 IS LANDED AND VALIDATED (commits 0f3ce8c0, 5140242b, f57bae46 docs §15,
68ad7336 + §16): stats surface (RuntimeStats.host_kv_* + jsonl host_kv block
+ reporter line), arena-pressure eviction with one-retry capture, pin/touch
during restore, trigger re-resolve after park (stale-index fix), t3/t5 +
4-session rotation stress all green (see §16 for the three configs).
t-ladder: t1-t3,t5 ✔; t4 N/A (static single lane).

REMAINING (the honest list): docs/155 item-3 flip (lives on main / repo/docs
— worktree rule forbids repo/ edits; coordinator's move), the 200k×3 scope
call (v1 re-scope documented in §15, literal needs a v2 device-paging work
order), and the untested scopes: MTP-on, batched arm, concurrent rotation.

NOTE: build/ was deleted mid-session by a cleanup pass — rebuild is
`cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON` then TARGETED builds only
(`ninfer-serve ninfer_host_kv_arena_test`); never bare `cmake --build .`
(83 test binaries ≈ 8.7G).

## Where things stand (60 seconds)

- **GOLDEN GATE: PASS** (docs/156 §14, e03d5d9c). A (101 tok) → B (98 tok) → A'
  (=A), temp 0: third response byte-identical, park/restore on both ranks,
  `prefix hit: 101 tokens`, ttft 261 ms → 58 ms, `reuse=restore_response_checkpoint`.
  §5.4 state identity proven with NINFER_HKV_DBG fingerprints (pool pages, GDN
  slots, ph, t0 — all bit-equal to A's post-prefill state).
- H1 ✔ H2 ✔ (arena; capture/restore incl. committed paged KV pages for BOTH the
  BF16 pool tier — the DEFAULT serving tier — and kvarn k4v2).
- t-ladder: t1 ✔, t2 ✔. t3 (three-turn), t4 (larger-lane), t5 (arena-full
  refusal) still open. **H3 next** (pressure parking, slots stats,
  host_kv_occupied, pin-during-restore, 3×200k stress table).
- LRU/eviction (`evict_for`, pin) has NEVER run live — the gate parked 3 entries
  (< max_entries=4), no eviction exercised. H3 must exercise it.

## The session's big find (do not un-learn)

The gate was blocked by ONE line-class of bug: `arena.at(component.offset)` —
missing `image.arena_offset`. Every park stomped the previous entry's extent;
every restore read the latest park. All surface signals lied: prefix-hit fired,
ttft improved, logs healthy — only the output diverged (at token ~2), and the
`cached_ph` fingerprint "matched" purely because A/B share the chat-template
prefix (first 4096 bytes of hidden identical). It took rank-attributed
park-vs-restore arena hashes to pin it. **Rule: any arena access is
`extent_base + offset`. The env-gated probes are committed — reuse them.**

Also found on resume: c80f2246 had renamed enum kinds in the HEADER ONLY — the
committed tree did not compile; window 10's E2E ran the pre-rename .o. Lesson:
a rename that leaves a TU broken invalidates every "verified" claim built on
the stale binary.

## Environment facts (verified this session)

- This serving config is the **BF16 pool tier** (kv_cache default BFloat16;
  2101 MiB context for 50496 tokens): `st.kvarn_ws.text` is EMPTY (bind is
  kvarn-only), `has_mtp` false, snap never bound/captured. Pool planes are
  page-major `[head_dim, 64, heads, phys_pages]` → per-page bytes =
  `plane.bytes() / ne[3]`, no scale tiles. Decode positions/cursors (cur_F,
  cpos, lengths) are DERIVED per request from plen — nothing else to restore.
- `st.cache_slot` = 1 (mtp off); prefill saves GDN state via
  `copy_slot(0 → cache_slot)` at line ~1574; a full prefix hit replays
  `copy_slot(cache_slot → 0)`. The cache slot is the thing to capture ✔.
- Model artifact: /home/intel/models/qwen3_8_27b.ninfer (27b, reasoning model —
  output lands in `reasoning_content` first; the gate driver
  `tools/smoke/host_kv_gate.sh` compares reasoning+content).
- Gates: NINFER_HKV_DBG (state/arena/restore fingerprints),
  NINFER_MB_DBG (now prints decode steps 1..8: `[SEQ-STEP1] step=N next=T`).
- Prompt must exceed 64 templated tokens to exercise a committed page.

## GPU + disk protocol (unchanged, binding)

Written coordinator grant (intercom) → `nvidia-smi --query-compute-apps` guard
at claim → work → release → verify 15 MiB / 0 apps. **Also: pkill -x
ninfer-serve, then CHECK the port is actually free before starting a new
server — a stale server from a previous run answers /v1/models and you will
test the WRONG BINARY** (cost one round this session).

Disk: was at 100% mid-session; freed by deleting MY OWN stale test binaries
(relinkable; kept ninfer_host_kv_arena_test + apps) → coordinator freed other
lanes' builds → ~21G free at session end. Full run_ci.sh relinks ~8.7G of test
binaries — feasible again but coordinate first.

## Next session's likely first moves

1. H3 pressure parking + stats (§6): the trigger sites and HostKvNet are all in
   tp2_backend.cpp ~30-135 (HostKvNet) and ~1316-1345 (admission trigger).
2. t3/t4/t5 ladder (t5's graceful path is already proven by §11's logs).
3. Exercise evict_for/pin (first live eviction).
4. docs/155 item 3 flip + stress table after H3.
