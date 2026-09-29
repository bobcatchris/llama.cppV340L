# 52 — 200k Context Design (D-07)

Status: **IMPLEMENTING** (2026-08-23, separate branch).
Owner: REMOTE (design + code) / GPU (verification). Supersedes the D-07 entry in docs/50 §2.

**Workflow (binding):** this is a defect-driven change — REPO.md rules apply.
Every new defect found while implementing gets a docs/50 register entry **and a
battery test before its fix is accepted**. No merge to mainline until the full
gate below is green; one command: `bash ~/ninfer/scripts/run_ci.sh`.

## 1. Goal

Serve `--max-context 200000` on 2× 5060 Ti (TP2, MTP k=3, q8 KV) at **≤ ~15 GB/rank**
(reference: llama.cpp runs 200k at 14.4 GB/GPU on the same GPUs, q8 K+V).
Today 200k is infeasible because two persistent buffers are sized to
`max_context` instead of to the prefix-cache capability.

## 2. Current cost (measured, post-712c0fd5, 80k config, `/tmp/serve_test2.log`)

Per rank (16 GB GPU):

| Component | Size @80k | Scales with | Notes |
|---|---|---|---|
| Model + MTP (Q4) | 9059 MB | — | fixed |
| Decoder state (KV pages 1251×64 + GDN 8 slots + tables) | 1957 MB | ~16.5 KiB/token | incl. MTP KV pool |
| `cached_ph {5120, max_context}` | 819 MB | **10.24 KiB/token** | persistent, `tp2_backend.cpp:249` |
| `prefill_dummy {5120, max_context}` | 819 MB | **10.24 KiB/token** | persistent, `:285` |
| Arena padding + work/staging | ~0.5 GB | — | 160 MiB pad, `:236` |
| CUDA context + fragmentation | ~1.5 GB | — | estimate |
| **Total** | **~14.8 GB** | **~37 KiB/token** | 80k barely fits |

@200k projection: 14.8 + (200k−80k)×37 KiB ≈ **19.2 GB/rank → OOM.**
The only context-scaling terms that are *design* costs (not physics) are the
two 10.24 KiB/token buffers. KV ring growth (≈+2.5 GB) is unavoidable and
matches llama.cpp.

## 3. What the two buffers actually hold (data flow, `tp2_backend.cpp` 700–861)

1. Main text prefill (chunked, T=512, `:794`) writes the **per-token hidden
   states of all prompt tokens** into `prefill_dummy` (`:806-811` copies them
   into `cached_ph` at offset `prefix_len`).
2. `cached_ph` = **prefix cache**: a later request sharing the prefix restores
   it (`:706` alias, `:735` comment) instead of re-prefilling.
3. `prefill_dummy` is re-aliased as `mtp_mh` for the **MTP module prefill**
   (`:843-860`, chunked by `mpc=512`). After prefill **only row `plen-1` is
   used** (draft head + `ar_hidden`, `:856-863`). All other rows are garbage.
4. `mtp_ph` (aliased `cached_ph`) holds the full-prompt hidden states that the
   chunked MTP module prefill reads as input (`:848`).

Consequences:
- `cached_ph` only ever needs **prefix-cache-cap** rows, never `max_context`.
- `prefill_dummy` as `mtp_mh` only needs **one chunk** at a time — *if* the MTP
  module prefill consumes each main-prefill chunk's hidden states before the
  next chunk runs (interleaving).

## 4. Design

### A. Prefix-cache cap (new option `--prefix-cache-capacity P`, default 16384)

- `cached_ph` allocated `{5120, P}` (P≤max_context, P≤16k ⇒ ≤168 MB/rank).
- Prefix match/restore unchanged **when plen ≤ P**. When `plen > P`: prefix
  caching disabled for that request (`zero_slot(0)` path, `:737`), and after
  prefill nothing is saved (`cache_valid=false`).
- `prefill_dummy` allocated `{5120, P}` for the plen≤P path (main prefill
  hidden-state output + mtp_mh alias, as today).

### B. Interleaved chunked prefill for plen > P (scratch only)

- Work-arena scratch: `ph_scratch {5120, mpc}` (~5.2 MB) + `last_row {5120,1}`.
- For each main-prefill chunk `c` (offset `off`, len `len`):
  1. run main prefill chunk into `ph_scratch`;
  2. immediately run MTP module chunk: `mtp_forward_batch(ids_c, ph_scratch,
     pos_c, ..., mh_c=ph_scratch, ...)` — same sequential-chunk exactness as
     today's loop (`:826-860` comment: chunks attend over accumulated history);
  3. keep `last_row` from the final chunk for draft head/`ar_hidden`.
- No per-request `{5120, plen}` allocation (this is what D-10 hit); no
  persistent growth beyond P.

### C. KV ring & sizing (no design change, config change)

- `cap = max_context + mtp_k + 4` already sizes pages to max_context (`:180`).
- 200k launch: `--kv-capacity 200000 --max-context 200000` (admission control
  from D-03 now rejects oversize; D-02 auto-sizing ports next).
- Cost ≈ +2.5 GB/rank vs 80k — unavoidable, matches reference.

### D. Minor

- Reduce 160 MiB arena padding to measured need (saves ~0.1 GB).
- GDN slot count (8) stays — verify-snapshot protocol depends on it.

## 5. Projected budget @200k, per rank (P=16k)

| Component | GB |
|---|---|
| Model + MTP | 9.06 |
| Decoder fixed (GDN slots, tables, round) | ~0.9 |
| KV ring 200k (both pools) | ~3.3 |
| `cached_ph` @P | 0.17 |
| `prefill_dummy` @P (≤P path only) | 0.17 |
| scratch / staging / padding | ~0.3 |
| CUDA context + fragmentation | ~1.5 |
| **Total** | **~15.4** |

~1.0–1.5 GB of headroom on 16 GB. If P1 measurements show less than 0.8 GB
headroom, fallbacks in order: (1) P→8k (−0.09 GB), (2) arena padding → 32 MiB
(−0.13), (3) GDN slots 8→6 (−~0.15), (4) accept 192k.

## 6. Phases & acceptance

| Phase | Who | Acceptance |
|---|---|---|
| P1 measure | GPU | exact per-rank breakdown from startup log + `nvidia-smi` at idle / 80k / 160k; confirm §5 numbers; log committed to docs/52 §5 |
| P2 implement A+B | REMOTE | compiles; **unit: `plen > P` path produces byte-identical first-token + d0 vs current `plen ≤ P` path on 12k-token prompt** (GPU runs) |
| P3 200k relaunch | GPU | server up at 200k; full battery §7.1 green; T2 prefix-hit still ≥2× faster at P; `det2.py` n2==n3 |
| P4 soak | GPU | `serve_battery` S1–S5 @200k; VRAM flat over 2h |

### Testing criteria (must all hold before merge)

1. **Battery gate (docs/50 §7.1)** — run at the 200k config, not just 80k:
   - Must PASS: T1 T2 T3 T5 T7 T8 T10 T11.
   - Expected PASS: T4 T6 T9 (nightly) T12.
   - T9 (long-prefill under MTP) is the direct regression test for the
     interleaved chunking in §4B — it must pass at 200k before merge.
2. **New unit test (P2, committed with the code):** `plen > P` interleaved path
   produces byte-identical first token + d0 vs the `plen ≤ P` path on a 12k-token
   prompt (P set below/above the prompt to exercise both sides). Lives in
   `tests/`, runs under CI — not just a one-off GPU probe.
3. **New battery test (T13, read-only client):** prefix-cache boundary at 200k —
   a `plen ≤ P` request gets a prefix hit on repeat (T2-style speed check), and
   a `plen > P` request re-prefills fully yet returns identical content to the
   first run. Register entry in docs/50 before it is accepted.
4. **Startup memory proof (P1):** per-rank breakdown from the startup log +
   `nvidia-smi` at idle, committed next to the §5 budget table — the §5 budget
   must be confirmed, not assumed.
5. **CI:** `run_ci.sh` green on the branch before push-to-mainline; perf
   snapshot (`latest.json`) updated so the 200k baseline is on record.

## 7. Risk register

- **R1** interleave changes main-prefill chunk ownership — the current
  `prefill_chunk(..., finalize_at_end=true)` single call hides the per-chunk
  hidden-state write; splitting it may touch `TextContext` internals. Mitigate
  by keeping a `plen≤P` fast path unchanged.
- **R2** MTP module prefill exactness depends on sequential chunks attending
  over accumulated MTP KV — already true (`:826-829`), interleave preserves it.
- **R3** prefix-hit speed at the P boundary: requests with `plen > P` pay full
  prefill (expected, documented).
- **R4** fragmentation: two large 2 GB allocations disappear at 200k — good;
  P-sized allocations are small enough to sit in the existing arena.
- **R5** memory estimate error bars (CUDA ctx, fragmentation). P1 resolves.

## 8. Non-goals

400k+ context, multi-sequence batching, KV offload, changing q8 KV format.

## 9. Fixed-cost reduction: how 200k actually fits (2026-08-23)

**Problem:** at the time of this design, `--kv-capacity 200000` was rejected by
the VRAM preflight (`16696 MiB required vs 16310 available` per rank). The KV
pool itself is NOT more expensive than llama.cpp q8_0 (ours ≈17–20 KB/token/rank
vs ≈16.5 KB/GPU for q8_0 — same order). The gap was **fixed overhead**:
≈13 GB fixed before any KV, vs ≈9–10 GB for llama.cpp.

**What the change was (the whole fix):** the transient scratch arena
(`b_opts.workspace_bytes`, `src/runtime/tp2/tp_engine.cpp`) was a hardcoded
**2 GiB per rank**, never measured. It was cut to **1 GiB per rank** (same
constant in `TpBackendOptions` default, `src/runtime/tp2/tp2_backend.h`).

- Why it is safe: `DeviceArena::alloc_bytes()` **throws on exhaustion** — a
  too-small arena fails loudly in tests, never silently corrupts. The fast
  battery (T1/T2/T3/T5/T8) passes at 200k with no arena errors.
- Why it was enough: cutting 1 GiB/rank drops the fixed budget from ≈13,003 to
  ≈11,980 MiB, so 200k totals ≈16.1 GB ≤ 16.31 GB available. That single 1 GiB
  is the entire delta between "max 150k" and "200k works."
- **Verified live (2026-08-23):** `--kv-capacity 200000 --max-context 200000`
  starts clean, decoder state 4012 MB / 3126 kv pages (cap 200007) per rank,
  **14,895 MiB/rank** total. Fast battery T1/T2/T3/T5/T8 all pass.

**Remaining fixed-cost candidates** (not needed for 200k, listed for >200k):
the `persistent` arena (256 MB) and `staging` (32 MB) in `TpRankState`, the
CUDA context (~0.7 GB), and the P-capped hidden-state buffers. KVarN P2 is
still the lever for 400k+.
