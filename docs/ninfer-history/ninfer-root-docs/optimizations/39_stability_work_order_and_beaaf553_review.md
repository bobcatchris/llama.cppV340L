# Document 39: Stability Work Order + Code Review of `beaaf553` (REV 2 — battery results in)

Date: 2026-08-21. Assistant ran the battery (`20260821_175703`) and triaged all 3 FAILs.
**The agent's job is code fixes + re-runs, NOT battery discovery.**

## 1. Battery results (20260821_175703)

| Case | Result | Numbers |
|---|---|---|
| mtp (BF16) | PASS | **93.98 t/s**, acceptance **85.6%** (370/432) — matches baseline |
| kv_i8 | FAIL (gate) | **88.00 t/s** (−6.4%), acceptance **77.1%** (−8.5pp) |
| prefix_on / prefix_off | FAIL (identity) | hit **works**: 2nd prefill 229.1 ms vs 6,710.6 ms (29×, only 7 suffix tokens re-prefilled); but request-2 output **diverges at the first generated token** |
| all legacy cases (pp, plain, mtp×2, A2, draft vocab, sampling det/div) | PASS | no regressions |

## 2. Findings (severity-ordered)

### P1 — CRITICAL BUG: prefix restore corrupts target-model state (identity FAIL at token 1)
Verified by review, suspects ranked:
- **Ruled OUT: KV page clobber.** `kv_alloc`/`mtp_kv_alloc` are reserved ONCE at
  make_rank (`tp2_backend.cpp:245-255`, row 0 bound) — physical pages persist across
  requests, so prefix KV is intact on hit.
- **Ruled OUT: snapshot timing.** The cache save (`copy_slot(0→cache_slot)` +
  `cached_ph ← mtp_ph`) is taken after the prompt prefill + t0 argmax, **before** the
  MTP generation rounds — exactly the state-after-prompt. Correct.
- **Ruled OUT: hit not happening.** 229 ms 2nd-prefill proves `prefix_len` ≈ 225.
- **PRIMARY SUSPECT: hit path skips the `decoder_state_span` reset.** Miss path does
  `cudaMemsetAsync(st.decoder_state_span, 0, bytes)` + `zero_slot(0)` + KV-row reset
  (line 707-714). Hit path only does `copy_slot(cache→0)` + mtp_ph restore (701-704).
  Any decoder state inside `decoder_state_span` that is NOT part of GDN slot 0 (per-layer
  aux state, position/ring counters, etc.) stays at request-1 end-of-request values on a
  hit → target logits diverge at the first generated token. Matches the symptom exactly.
- **SECONDARY SUSPECT: `copy_slot` completeness** — verify it copies all 48 GDN layers'
  full per-layer state (check the impl, not the call sites).

### P2 — BUG (code smell confirmed): mtp_ph restore before allocation.
Hit branch writes `st.mtp_ph.data` at line 703, but `st.mtp_ph` is allocated at line 747
from a request-scoped arena (rewound at line 689). Line 703 writes to a **stale pointer**
from the previous request's scope. It only survives because the arena returns the same
address — move the restore to AFTER line 747's allocation. Affects MTP-head input on hit
(draft quality), not t0.

### P3 — MEASURED: I8 KV costs −8.5pp acceptance, −6.4% t/s.
88.00 t/s / 77.1% vs 93.98 / 85.6% (matched 256-token battery case). Real quantization
cost, not a bug. **Decision: I8 stays OPT-IN** (`--kv-dtype int8`), default BF16. I8's
value is long-context capacity (KV bytes/token 34 KB → 18.5 KB), not speed. Battery gate
now a hard floor (acceptance ≥ 70%, t/s ≥ 80% of BF16) — edited in `verify_battery.sh`.

### P4 — REPORT BUG: prefill line lies on hits.
`prefill: 232 tokens in 229.1 ms (1012.8 t/s pp)` — reports full `plen`, not the
re-prefilled count (7). Log `prefix_len` on hit and report tokens actually processed.

## 3. Agent work order (REV 2 — in this order)

**WO-1 — Fix P1 (the blocker).**
- D1: In the hit path (`tp2_backend.cpp` ~701): either (a) memset the parts of
  `decoder_state_span` not covered by the GDN slot copy, or (b) restructure: full reset
  THEN restore slot + mtp_ph + (re-materialize prefix KV if needed). Prefer (b) for
  auditability.
- D2: First reproduce minimally: battery `prefix_on`/`prefix_off` pair, or driver
  `--prompt2`. Add a one-line log: `prefix hit: N tokens (slot K)`.
- D3: If still divergent after D1: run the pair with `--mtp 0` (isolates MTP), then dump
  `st.hidden` at prefill-end (position plen−1) hit vs clean — first differing value →
  bisection layer id → GDN vs attention.
- D4: Verify `copy_slot` impl copies every GDN layer's full state.
- Gate: `prefix_identity_ok` PASS (token-identical), `prefix_skip_ok` PASS, full battery
  green. Then `--update-baseline`.

**WO-2 — Fix P2** (move mtp_ph restore after allocation, line 747). 5-minute fix, do
with WO-1.

**WO-3 — Fix P4** (log prefix_len, report re-prefilled token count in the prefill line).

**WO-4 — Preflight VRAM fix** (from REV 1): include `max_context × 5120 × 2` per rank in
the TPEngine estimate; add `--prefix-cache-max-tokens` cap (default max_context).

**WO-5 — TC-M multi-request case**: driver `--repeat N` / `--prompt-file-list`; battery
case = N large prompts, outputs token-identical to single-run baselines, VRAM flat.

**WO-6 — `serve_battery.sh` (S1–S6, doc 38 §7)** after driver cases green.

**WO-7 — M6 batched/chunked prefill** (unchanged; still 0%; T=1 prefill ≈ 32 t/s is the
wall for long contexts).

## 4. Test infra landed (assistant, this batch)
- Driver: `--prompt2 "text"` (2nd request in-process; own prefill/acceptance2/output2
  blocks), `--no-prefix-cache`. Compile-verified.
- Battery: `kv_i8`, `prefix_on`, `prefix_off` cases + gates `kv_i8_ok` (hard floor),
  `prefix_identity_ok` (TC-P crown jewel), `prefix_skip_ok` (wall-time ≤ 90%).
- Both gates corrected post-run (I8 floor; skip=ms-only since the prefill line reports
  full plen for both runs — see P4).

## 5. v1 scorecard (REV 2)
| Must | % | note |
|---|---|---|
| M1 multi-request stability | 75 | |
| M2 sampling | 70 | |
| M3 option plumbing | 80 | |
| M4 I8 KV | 85 | **verified**: 88.0 t/s / 77.1%, opt-in |
| M5 prefix cache | **45** | **P1 bug: restore corrupts state** |
| M6 batched prefill | 0 | |
| M7 gates | 60 | TC-KV/TC-P wired + run; TC-M/serve pending |

**Critical path: WO-1 (+2,3, one commit) → re-run battery → green → WO-4/5 → serve
battery → M6 → DoD.** If D1 resolves P1 first try: v1 in ~2–3 agent-days. The prefix bug
is the only real blocker now.
