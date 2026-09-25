# KVARn BOOT BATTERY — standing cells for every kvarn serving window

Scope: per the RED→GREEN closure law, device-touching cells run at the END of every
kvarn boot window, results banked per window LABEL. This file is the canonical cell
list for the kvarn KV tiers (kvarn_k4v4 default, kvarn_k4v2 rollback — see
BOOT_LAUNCH_RUNBOOK.md §5). Cells are zero-build: they boot a BANKED bin and replay
banked/generated bodies, grading exactly.

## Environment law for kvarn windows

Canonical env (serve_10k.sh exports) **plus `NINFER_WORKSPACE_MIB=512`** for any leg
beyond ~10k tokens (the era-pinned 96 MiB chunk arena exhausts on the materialize
route's O(position) temps; 512 is the live-verified lever, PLOG-079). Boot argv:
canonical serve_fast shape + `--kv-dtype <tier> --max-context N --kv-capacity N`.

## Ceiling facts every window must respect (measured, bin 8ac1ba93 class)

- **Arena position ceiling:** the materialize prefill route allocates k_temp+v_temp
  BF16 = 2 × 256×64×8 heads × 2 B per 64-tok page = **8 KiB per token of position**
  (gqa_attention_kvarn.cu:355-358; kKvarnAttnD/G = 256/64). Usable position therefore
  ≈ ws_MiB × 128 tokens: ws96 → ~12.3k; **ws512 → ~65,536** (binding BEFORE any pool
  ceiling on long legs).
- **Pool ceilings (preflight-charged):** k4v4 = 11,152 B/t/rank, k4v2 = 8,976 B/t/rank.
  With ws512 (fixed ~7369 MiB): k4v4 pool ≤ ~74k tok, k4v2 ≤ ~74.6k tok. With ws96
  (fixed ~6947): k4v4 ~113,472, k4v2 ~141,056 — but the ws96 arena caps position at
  ~12.3k, so those pools are unreachable for prefill on this route.
- **113,664-gen legs (≈89.5k real) are UNREACHABLE on this bin** — refusal receipts
  banked at results/amd/k4v4fin/ (mc 113664 boots, both tiers, ws512). They unlock
  only via the docs/120 B2 direct prefill route (deletes the O(P) temps; parked lane
  decision).

## Cells

### KV-B0 — small-ctx serving sanity (port desk, PLOG-074/076 provenance)
- What: 5-probe behavioral battery (BLUE/ORANGE/391/ZEBRA-48213/ZEBRA-73942 class;
  tools/v340l/w7_k4v4_gates.sh G-Q1 section) + 2k BLUE parity.
- PASS: all finish=stop, exact keywords, no mojibake.

### KV-B1 — needle20k replay+grade (ovf desk, PLOG-079; RED→GREEN closure cell)
- What: replay the banked 19,709-tok needle body at ws512; grade exact "ORCHID-TUNNEL".
- PASS: finish=stop, exact substring, zero worker errors.
- Guards: the arena-exhaustion bug class (std::bad_alloc wall at ~10.1k) — the cell
  was RED on f3312f25, GREEN on 8ac1ba93.

### KV-B2 — needle60k mid-depth replay+grade (k4v4fin desk, 2026-09-20; this window)
- What: body = `tools/v340l/w7_needle_gen.py --ctx 76800 --depths 0.5` + model field
  + max_tokens 64 (regenerable deterministically; banked copy with this window's
  transcripts). Boot kvarn_k4v4 ws512 cap 65536; fire; grade.
- PASS: exact "MARBLE-COMPASS" substring in content+reasoning_content, finish=stop,
  prompt ≈ 59.9k real tok (±2%).
- Guards: the CLASS "long-context retrieval at the arena-ceiling paired point" — any
  regression in kvarn long prefill correctness (position arithmetic, page mapping,
  scale tables at depth) fails the exact match.
- Runtime: ~15-20 min (prefill-dominated). Run at least the d0.5 body per window;
  full d0.25/0.5/0.75 sweep owed on tier/format changes.

### KV-B3 (REGISTERED, PARKED) — needle113k
- Registered per the WO's gate text; **skips loudly on this bin** (assert cap ≥
  113664 bootable first). Unlocks with docs/120 B2 direct route; keep registered so
  the class survives the route change.

## Receipts this window (2026-09-20, k4v4fin desk)

- Mission-literal boots (mc 113664 cap 113664, ws512): REFUSED live, both tiers —
  results/amd/k4v4fin/boot_refused_*_receipt.log (k4v4 ~8582/8160 class, k4v2
  ~8346/8160 class; allocator = the gate, live lines in the receipts).
- Working posture mc/cap 65536: see results/amd/k4v4fin/grades_*.txt for the live
  preflight lines (the VRAM record; this bin's /health carries no VRAM field).
