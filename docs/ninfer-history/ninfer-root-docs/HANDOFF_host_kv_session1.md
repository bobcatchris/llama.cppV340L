# HANDOFF — host-KV safety net lane (session 1, 2026-09-06/07)

Resume point. Everything below is verified against the pushed state.

## Where things stand (60 seconds)

- Branch `wo/host-kv-safety-net` @ `5a63cd4b`, base `wo/kvarn-multibatch`
  (NOT main). Work order: `docs/156` (also on main @ 9d36d2a4).
- **H1 LANDED** (`6e558163`): `src/core/host_kv_arena.{h,cpp}` — pinned/pageable
  extent arena, smallest-first LRU + pin protection. Unit tests green
  (`ninfer_host_kv_arena_test`).
- **H2 LANDED** (`58e998bf` + `6e94eccb` + `dcdd8410`):
  `src/runtime/tp2/host_kv_parked.{h,cpp}` — capture/restore of the cached
  slot state; park/restore triggers in the single-seq prefill flow;
  `--host-kv-mib/--host-state-slots/--host-kv-min-pages` wired
  (serve_options.h → ServeOptions → generation_service → EngineOptions
  (types.h) → tp_engine.cpp b_opts → TpBackendOptions → my trigger).
- **E2E attempted (window 10)**: park/restore FIRES on both ranks; image 147
  MiB fits a 6144 MiB arena; graceful fallback PROVEN (a failed restore never
  kills a request). Findings + two open defects: docs/156 §11.

## The three things between you and a passing golden gate

1. **§10 — COMMITTED PAGED KV PAGES ARE NOT CAPTURED (blocking).** The image
   has linear slots, open-page tiles, tails, snapshot, GDN ckpt, cached_ph —
   but NOT the committed pages that live in the PagedKVCacheView planes. A
   restored prompt longer than one page loses its committed prefix. Design is
   DONE: docs/156 §12 (page_ids() + plane(i) + per-layer committed counters;
   gather at park via the old lane's page ids, scatter at restore via the new
   lane's — physical pages may differ, the LEDGER is the semantic match).
2. **§11.1 — restore geometry-mismatch refusal.** kvarn_prefix_snap is
   re-captured every prefill; its tensor shapes can drift between park and
   restore. Fix: re-allocate the snap tensors to the captured shapes at
   restore, or skip snapshot restore on drift (full-prefill fallback).
3. **§11.2 — empty-ledger park race.** rank 0 parked a 0-token ledger even
   though the guard requires a non-empty one. Fix: re-check
   `!st.cached_tokens.empty()` INSIDE capture_slot_image and bail before the
   arena alloc (guard/capture race across the two rank threads).

Then: the golden-gate E2E (two-turn, prompt A → B parks A → A' restores;
A' output must equal A's), then H3 (pressure parking + stats), per §6.

## The golden gate (how you know you're done)

Server (worktree build): `./build/apps/ninfer-serve
/home/intel/models/qwen3_8_27b.ninfer --host 127.0.0.1 --port 8093 --devices
0,1 --host-kv-mib 6144 --host-state-slots 4 --host-kv-min-pages 1`.
Drive `/v1/chat/completions` (temperature 0): A (≥64 tokens) → B (different,
≥64) → A again. PASS = third response IDENTICAL to the first (greedy) AND
`[host-kv] parked`/`restored` lines in the server log. Watch: A's templated
prompt was 81 tokens → one committed page exists → gate fails until §12 lands.
Prompts ≤64 templated tokens only exercise the tile path.

## Pitfalls that cost this session (do not repeat)

1. **Dumps from different rounds/chains are different data.** All diagnostic
   dumps are overwritten per chain — the first-chain gate
   (NINFER_DFLASH2_BLK0DUMP + dflash2_blk0_seq) fixes it for the DFlash2
   side; host-kv dumps use the same gate.
2. **Never diff a dump against a log line from a different run/round** — that
   manufactured the fake "fuse mismatch" and a fake tap-layout "mismatch".
3. **fnv hashing of bf16: use the HIGH 16 bits of the f32 encoding** (the
   oracle's fnv_hash_bf16 hashed the low half — every hash it printed was
   garbage until fixed).
4. **tensor.view(N, ...) row-major traps**: a [4096,16] row-major tensor
   viewed as (32,128,16) IS (head, dim, col) — but reshaping a flat dump
   needs reshape(cols, -1).t(); getting this wrong silently yields cos ≈ 0
   (§22.56's warning is load-bearing).
5. **The moved-from print**: printing `img.ledger` after
   `entries.push_back(std::move(img))` prints the moved-from empty vector —
   read `entries.back()` instead.
6. **pkill -f "ninfer-serve ..."** matches YOUR OWN shell's cmdline and kills
   it mid-command. Use `pkill -x ninfer-serve` or kill the exact PID.
7. **Disk**: the worktree full-tests build hit 100% disk. df before builds;
   delete only your own /tmp artifacts (fc dumps, head dequants).

## GPU protocol (unchanged, binding)

Written coordinator grant → `nvidia-smi --query-compute-apps` guard at claim
→ work → release → verify 15 MiB / 0 apps. Nine grants used this session,
zero conflicts. Server start for E2E: the command above (kill by exact PID or
`pkill -x ninfer-serve`, never -f).

## Tooling inventory (all committed on the branch)

- `d2_block_check.py <blk> <prefix>` — per-block stage census (bisect tool).
- `d2_selector_port.py` — the §22.27 selector port (head = text/output_head,
  NOT draft_head; the walk skips col 0).
- `d2_cold_forward.py` — full 5-block cold forward (per-lane attention).
- `d2_iso_linear.cpp` — generalized iso-linear TU (argv row/geometry/x).
- `d2_fc_compare.py`, `d2_attn_map.py`, `d2_iso_compare.py`,
  `d2_stage_trace.py`, `d2_cold_sweep.py` (pre-§22.58 fixes; its conv-base
  view is the CORRECTED one).
- Env gates: BLK0DUMP (+BLKIDX), STAGEDRAW, TAPDRAW, FUSEDUMP, ADMIT — all
  first-chain gated.

## Reference chain

docs/156 §9 (H2 plan) → §10 (committed-pages gap) → §11 (E2E findings) →
§12 (paged capture design). DFlash2 context: docs/151 §22.57-§22.61
(engine vindication; the acceptance bug is NOT engine math — training-side
tap semantics or checkpoint quality, rank evidence #552/#4711 of 248320).
