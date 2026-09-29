# WHY 113k+ CONTEXT IS BLOCKED ON THE AMD/V340 LINE — the two-ceiling explainer

**Audience:** anyone reviewing the kvarn work (internal or external). **Date:** 2026-09-20.
**Claims below are measured on this box (4× AMD V340 / gfx900 dies, ROCm 6.2.0, TP4, world=4)
with receipts cited; nothing is estimated. Receipt paths are in the repo under
`results/amd/` and `docs/amd/` (PLOG chain: docs/amd/PERF_LOG_AMD.md, PLOG-079/084).**

---

## 1. The situation in one paragraph

The AMD line promoted a quantized KV cache (`kvarn_k4v4`: K and V each at 4 bits, 11,152
bytes/token/rank vs 34,816 for bf16 — 3.1× denser) with a serving context of **65,536
tokens, up from 36,352 under bf16**. The quality battery passed 8/8 (needle retrieval exact
at 25/50/75 % depth of a ~60k-token prompt, both tiers). But the *pool arithmetic* says the
8 GB-per-die cards could hold ~113k tokens (k4v4) / ~141k (k4v2), and we ship 65,536. This
document explains the gap: it is a **work-arena scratch buffer that scales with sequence
position**, and — the honest part — that scaling is a real design debt in the current
kvarn prefill route, not a hardware law.

## 2. The two ceilings

A kvarn boot must fit, in each rank's ~8,160 MiB usable VRAM:

```
fixed side          +  KV pool                    +  work arena
(weights 5,131         11,152 B/token/rank           scratch for the prefill
 + decoder 1,360       (k4v2: 8,976)                 dequantization temp
 + staging 200                                       (see §3)
 + padding 160
 + workspace W)
```

| posture | KV pool allows | arena allows (position) | result |
|---|---|---|---|
| W = 96 MiB  | ~113,472 tok (k4v4) / ~141,056 (k4v2) | **~12,300 tok** | boots, then `std::bad_alloc` mid-prefill (measured 4/4 repro, PLOG-era; receipts `results/amd/k4v4gates/`) |
| W = 512 MiB | ~74,000 tok | **65,536 tok** | **the promoted posture** — 65,536 boots with 94 MiB slack (preflight log, `results/amd/k4v4fin/`) |
| W = 512, mc=113,664 | — | — | **refused live at boot**: required 8,577 / 8,160 MiB (k4v4), 8,341 (k4v2) — receipts `results/amd/k4v4fin/boot_refused_*.log` |

The two ceilings move in opposition: freeing the pool (small W) starves the arena; freeing
the arena (large W) eats the pool. 65,536 is the 64-multiple sweet spot where they touch.

## 3. The design debt — stated plainly

The current kvarn **materialize** prefill route dequantizes the packed K/V cache into
scratch buffers (`k_temp`, `v_temp`) and then runs attention over the dequantized copies.
Those temps scale with **sequence position** (8 KiB per token of position —
`gqa_attention_kvarn.cu:355`, `kKvarnAttnD/G = 256/64`):

- **The criticism that sticks:** scratch should scale with the *prefill chunk window*
  (O(chunk), bounded) — not with the whole sequence (O(position), unbounded). An
  O(position) temp is the straightforward "dequantize-then-attend" delivery: it reuses the
  existing attention kernels and gets a quantized-KV format serving fast, but it converts
  context length into a memory requirement, which is exactly wrong for memory-limited dies.
- **The aggravation that was ours and is fixed:** the HIP port initially carried a *second*
  O(position) temp the CUDA original did not have. That was a genuine port defect —
  root-caused by the overflow desk (int32 hypothesis falsified; it was arena exhaustion),
  fixed in-place with a RED→GREEN cell (`results/amd/ovf/RED_OVF_row.md`), and the fix is
  in the promoted bin.
- **What was NOT defective:** the limitation was never hidden. The position ceiling was
  *derived from source and confirmed by measurement before promotion* (8 KiB/token →
  12.3k @ 96 MiB, 65,536 @ 512 MiB), the un-bootable postures were refused live by the
  allocator and the refusal receipts banked, and the default was set to the verified
  posture — not to an arithmetic hope.

## 4. Why the NVIDIA line does not see this

Same stack, different memory budget — the wall sits where nobody has driven:

1. **Per-rank memory is 2×:** the V340 is a 16 GB board split across two dies — each TP4
   rank sees ~8 GB. The NVIDIA line's RTX 5060 Ti ranks are full 16 GB parts (TP2).
2. **The arena binds late there:** the NVIDIA line runs the 1,024 MiB default work arena.
   At 8 KiB/token that position ceiling is ~131k — past every real use on that line, so
   the constraint is invisible rather than solved.
3. **KV economics are looser:** production on the CUDA line is kvarn k4v2 (8,976 B/t)
   against a 16 GB pool, vs our starting point of bf16 (34,816 B/t) on 8 GB.
4. If either line ever pushes toward the model's native 262,144-token context, it meets
   this same wall — the arena is stack-wide, and the pool then binds next.

## 5. The cure, and why it is parked (not forgotten)

**docs/120 "B2 direct route":** attend directly against the packed quantized cache with
in-kernel dequantization — the O(position) temps disappear, and the pool alone sets the
ceiling (~113k k4v4 / ~141k k4v2 at today's posture; posture levers stretch toward ~151k;
151,552 was already refused live even at ws96 by −89 MiB for k4v2 — the allocator is the
gate, receipts banked).

Parked for two declared reasons: (a) it is **not byte-identical** — in-kernel dequant
changes the arithmetic path, so it needs its own RED→GREEN quality battery, not a diff
review; (b) it is a real kernel-engineering desk, and it only pays once >65k-context
kvarn serving is actually demanded. The standing **KV-B3** cell (needle retrieval at
113,664) is registered and skips loudly today — it auto-unlocks the day the route lands,
so the class cannot be quietly dropped.

## 6. What we would do differently (the takeaway)

For a quantized-KV format on memory-limited dies: dequantize-then-attend is a fine
*bootstrap* (it got kvarn serving in days instead of weeks), but the scratch budget must
be O(chunk) from day one, or the context ceiling is silently coupled to the workspace
budget. The direct/fused route is the destination; the materialize route is the road that
got a verified, quality-gated 65,536-token serving default onto this box in the meantime.

*— Team Red (AMD/V340L), coordinator chair, 2026-09-20. Receipts: PLOG-078/079/084,
`docs/amd/WO_Q4KV_K4V4.md`, `docs/amd/KVARN_BOOT_BATTERY.md`, `results/amd/k4v4fin/`,
`results/amd/ovf/`, `results/amd/k4v4gates/`.*
