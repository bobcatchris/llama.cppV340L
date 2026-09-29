# VRAM LEDGER — measured vs claimed (2026-09-08, agent1, user-directed)

**Commit:** main@25fab77d + ledger instrumentation `92d6dcf0` (branch `wo/vram-ledger`)
**Method:** every `cudaMalloc` site traced (`NINFER_VRAM_TRACE=1`, `src/core/vram_trace.h`) + phase
`cudaMemGetInfo` checkpoints (both cards) + 3s `nvidia-smi` sampler + the `[preflight]` line as the
"claimed" source. Raw artifacts: `results/vram_ledger/` (serve logs, sampler CSVs, responses).
**Configs:** LINE M — 40960 ctx, BF16 KV, MTP k=3, qwen3.8-27b, TP2 (per-rank = per-card figures).

## 1. HEADLINE: claimed vs measured floor

| Config | Preflight CLAIMED | MEASURED steady/card | Overestimate | Over % |
|---|---|---|---|---|
| conc=1 (single-seq) | 14,507 MiB (fixed 12,827 + ctx 1,680) | **12,998 MiB** (2,850 free) | **1,509 MiB** | 10.4% |
| conc=2 (batched 2-lane, VERIFIED dispatched) | 16,043 MiB (fixed 14,363 + ctx 1,680) | **14,508–14,510 MiB** (1,337 free) | **~1,535 MiB** | 9.6% |
| int8 @ 200k, conc=1 (row 1, user-directed proof) | 16,679 MiB (REFUSED) | **15,069 MiB** (780 free, ran to completion) | **1,610 MiB** | 9.6% |

**The overestimate is a near-CONSTANT ~1.5 GB across configs** — it is NOT per-lane, NOT
context-proportional, NOT workspace-driven. It is the reserved-never-touched stack:
prefix-cache pool at full cap (16,384 tok) + flat workspace reserve (1,024 MiB) + slack padding.
The refusal gate charges all of it; real execution never touches most of it.

## 2. Measured allocation floor, per component (conc=1, per rank — site traces)

| Site | MiB | Notes |
|---|---|---|
| DeviceArena.reserve (main/weights) | 9,059.9 | model materialization |
| DeviceArena.reserve (pool A) | 1,024.0 | KV/prefix pool slab |
| DeviceArena.reserve (pool B) | 256.0 | logits/hidden slabs |
| DeviceArena.reserve (pool C) | 32.0 | small slabs |
| DeviceArena.reserve (KV pool @40k) | 2,505.8 | context pool (conc=1) |
| tp2.gdn_ckpt.conv+rec × 64 sites | ~73.4 total | 1.53 MiB/rec + 0.03/conv per layer, ×2 ranks ×32 layers |
| tp2.sample_cfg + ar.epoch | ~0.04 | |
| **SUM** | **≈12,998** | **= measured 12,998 exactly — no hidden allocations** |

conc=2 delta vs conc=1 (measured): KV pool slab 2,505.8 → 4,014.7 (**+1,509** for lane-2
KV+slots); GDN checkpoints unchanged; decode.steady.batched adds only **+2 MiB** over startup
(rings/workspaces are startup-preallocated). The claimed per-lane charge is **+1,536** —
measured **+1,510** — this line item is essentially HONEST (Δ 26 MiB).

## 3. The 1.5 GB overestimate, localized

Preflight fixed+ctx+reservations vs measured: the claim includes, beyond the measured floor:
1. **Prefix-cache pool at full cap** (16,384 tok BF16 ≈ 1 GiB/rank) — sized as if fully
   resident from token 0; actual usage is a rolling window that stays within the KV pool.
2. **Flat workspace reserve 1,024 MiB** — decode steady shows +2 MiB over startup; the bulk of
   the workspace figure is never touched at these shapes.
3. **Slack padding** on top of the two above.
Together ≈ 1.5 GB — precisely the constant gap. The context-per-token formula itself is
ACCURATE (claimed ctx 1,680 MiB @ 40k BF16 ≈ measured pool 2,506 minus fixed KV share; the
per-token slope checks out against the int8@200k row: 3,852 MiB claimed ctx @ 200k i8).

## 4. CORRECTED CONSTANTS — PROPOSAL ONLY (NOT landed; user sign-off required, LITH rule)

- **Refusal gate should charge the MEASURED floor + safety margin**, not floor + untouched
  reservations: gate = fixed_measured + ctx_formula + 256 MiB margin.
  - conc=1 @40k BF16: gate would say ~13,254 vs current 14,507 (−1,253).
  - int8 @200k: gate would say ~15,325 vs current 16,679 (−1,354) → the user's 200k int8
    becomes launchable WITHOUT the override, with 780 MiB real headroom still uncovered.
- Per-lane charge (+1,536 claimed vs +1,510 measured) stays as-is — honest within 26 MiB.
- Keep the 1,024 MiB workspace + prefix pool as RESERVES for diagnostics, but exclude them
  from the refusal comparison (they are address-space reservations, not live pressure).
- Risk note: the margin is what stood between pass-7's ghost-server era and clean VRAM fences;
  256 MiB margin + pre-launch fence (both cards ≤100 MiB) preserves that guarantee.
- **ONE-LANDING PACKAGE (A2 conditions folded, coordinator 2026-09-08):** the constants change
  lands TOGETHER WITH the fence-baseline tightening — `wait_gpus_free` 500→50 MiB, serve-fence
  100→20 MiB — so pre-serve residue cannot eat the margin (at the old 100 MiB baseline, ~85 MiB
  of the 256 was exposed; tightened, effective ≈251 MiB). Host-kv literal shape added as §6 row
  (i8 @ 45k×3, PREFIX_CAP=50000, arena 94% packed, ran green unmeasured). Preflight, once
  corrected, prints BOTH the legacy claimed figure and the measured floor.

## 5. Reproducibility

- `tools/bench/vram_ledger.sh` (conc=1 + conc=2) — batched path additionally verified via
  `NINFER_BATCH_WINDOW_MS=50` two-client race (`[tp2] batched decode: dispatched 2-lane batch`).
- Raw logs: `results/vram_ledger/serve_*.log` (`[VRAM]`/`[preflight]` lines), `sampler_*.csv`.
- int8@200k override run: `/home/intel/ninfer/PROOF_int8_200k_vram_override.txt`.
- Trace overhead: zero when `NINFER_VRAM_TRACE` unset (static-flag gated).

## 6. REVIEW (A2, 2026-09-08): SOUND — margin conditions + untraced-alloc sweep

- 256 MiB margin ADEQUATE: samplers FLAT post-startup in all three configs (zero lazy
  growth steps at LINE-M shapes); int8@200k ceiling variance ±2 MiB ⇒ margin covers ~100×
  jitter. CONDITIONS: (i) pair with fence-baseline tightening 100→20/50 MiB (else pre-serve
  residue eats ~85 MiB of the margin); (ii) host-kv literal shape noted below.
- Host-kv literal row (ran GREEN unmeasured, 2026-09-08 §18.30): i8, 45k×3 sessions,
  PREFIX_CAP=50000, arena 94% packed, 6 parks/3 restores, byte-identical restores.
- Untraced allocations: NONE on host-kv/restore paths (memcpy-only into existing pools;
  gdn_ckpt 73.4 MiB already in table). host-kv arena = 14 GiB cudaHostAlloc = HOST RAM —
  out of VRAM scope; flag for a future HOST-memory ledger (pinned-host OOM class).
- I8 ctx formula × scale planes (A2 verify-item): CONFIRMED INCLUDED empirically — the
  200k run held 199,860 tokens inside the claimed ctx envelope (3,852 MiB) with 780 MiB
  spare; missing scale planes (base+2/3) would have overrun the margin. Per-tier constant
  18,496 B/token carries them (Int8Group64 K+V+2 scale planes, 17 layers × 2 sides).
- Implementation request for the corrected gate: preflight must print BOTH the legacy
  claimed figure and the measured floor (tonight's debugging leaned on that line).
