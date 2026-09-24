# THE LATCH — one row, in numbers, for the next hunt seat (agent3, 2026-09-14, no patch)

**What the ramp IS.** The lag is not a quantity accumulating toward a threshold. It is a **two-state
system whose entry rate rises and whose exit rate collapses.** Full detail and predicates in
`AGENT3_STAMPSTATE_AUDIT_2026-09-14.md`; this is the row to read first.

**Corpus.** Seven served boots (G18c, G18d, G18e, G19spaced, G20long, A4L0, G21leg0), two request
geometries (25×32, 15×64), two dial states, one spaced arm. 2,773 served rank-0 steps, ~5,950
retried stamps, **203 runs**. Era, read from the datum and not from any runner: **all seven carry
`kind=argmax` and none carries a single `kind=ar` line**, so on every leg here the AR ring's silence
is the totality-rescued INFERENCE, not a positive count — that still awaits a boot where the AR
emitter names itself.

| fact | number | why it matters |
|---|---|---|
| P(retry at s+1 \| retry at s) | **0.866–0.951** (7 boots) | the state persists step to step |
| P(retry at s+1 \| clean at s) | **0.046–0.116** | …and clean steps almost never start it |
| coupling ratio | **11.9–19.7×** | not a rate that drifts smoothly — a latch |
| P(enter \| clean), quartile Q2→Q4 | **0.024 → 0.242 → 0.646** (pooled, n_clean 1284/447/96) | entry gets easier |
| **P(exit \| on), Q2→Q4** | **0.337 → 0.113 → 0.048** | **leaving gets ~7× harder. This is the number a pure accumulator cannot produce.** |
| share of steps degraded, Q2→Q4 | 7.8% → 64.9% → 89.4% (pooled) | the "ramp" is occupancy of the on-state |
| run length | median 4, longest **156** stamps | 156 > 4.9 whole requests |
| runs starting at a request entry | **7 of 201 (3.5%)**, against 193 entries | the entry event does not trigger it |
| runs spanning a request entry | **43 of 201 (21%)** | the entry event does not clear it |
| P(next retried \| prev retried): mid-request vs at-entry | **0.922 (n=2217) vs 0.687 (n=67)** | the request boundary interrupts ~⅓ of the time, resets nothing |
| centroid of retried stamps | **0.78–0.83** of max stamp | late-concentrated; the shape that fakes a period |
| lag-1 autocorrelation, every boot | **+0.71 … +0.85** (argmax of the whole sweep) | serial correlation, not periodicity — the reason "no period" is right || clock-collapse CV of onset | busy-seconds **2.4–6.8%**, computed-tokens **4.3–6.5%**, stamps **9.4–12.3%**, log-bytes **78.9%** | the accumulator counts *issued work under power*, not argmax calls; log volume is dead |
| design effect on any bin test | **VIF 7.5–19.6** → effective n 19–39, not 204–394 | quote run counts, never stamp counts, as the sample |
| incidence vs depth on the slot axis | incidence χ²(31)=**3.2–6.7** (flat, all 7); depth-weighted **41–118** (peaked) | "hot slot" was never a slot effect |
| arrival at the argmax call, rank0 first | **53.9–57.3%** (near-symmetric) while retries are **93–99%** rank0 | the one-sidedness is **downstream of the host call**, in device-side publish latency |
| K, correctness | max try **8** (7 bins now), FAILOUT/ARTAG/REJECT/LONE **0**, `kind=ar` lines **0** (see era note: that is an inference, not a count), text identical | nothing here is a correctness event |

**What is EXCLUDED, with the measurement that did it.** (1) Arena/page-table/KV-content growth: no
such structure accumulates per step at this argv — `work_.reset()` at both ends of every forward,
`HostKvNet`/`MtpNgramMod`/`vocab_counter` gated off (grep-verified zeros, `MTP k=0`,
`reuse=full_reset`), and any within-request ratchet is killed by requests 2–7 being fully clean.
(2) Reset/prefill and every per-request-entry mechanism: 3.5% of run starts at entries.
(3) Periodicity and the argmax ring's slot arithmetic: lag sweep flat, incidence bins flat.
(4) Wall clock, idle recovery, thermals, power cap, the `[gating]` dial, and log-byte volume.

**What is LEFT.** A latch whose seed and whose sustenance are separate questions. The sustaining
mechanism is plausibly the retry hammer itself (an absorbed retry costs 72–79 ms and re-launches the
whole argmax kernel at 100% duty on both dies — measured, three legs, inside the 65–80 ms poll
ceiling the source names). The SEED is still unnamed and is per-computed-token-or-busy-second, not
per-stamp, and lives on the rank-1 side of the rendezvous.

**Cheapest next moves, no code:** L1 magnitude of self-feedback (banked bytes, zero card) → L2 the
matched-total A/B that finally separates busy-seconds from tokens (one boot, zero build) → L3 the
cumulative-joule clock on the existing witness columns. L4 (rank1 on a persistent thread) is the only
one that needs a build and it is *not* exonerated — only its trigger role is; the asymmetry is
carried across boundaries, so it can be seed or amplifier, not the latch itself.

**One warning to whoever reads a world=4 log with these tables:** `census_read_analysis.py` now
prints **Part 0** first, and at n≠2 it *downgrades* the slot/alias/period test, the LONE/pairing
form, and Part D's rank-0 statistics — because the one-shot rings exist at world==2 only
(`tp_group.cpp:114-117`) and `require_argmax_transport` refuses elsewhere (`argmax_routing.h`). At
world=4 a `slot=` line is a different transport wearing a known spelling. Do not read it as the
pair. Certified both directions in `tests/check_census_world_banner.py` (fires at n=1 and n=4,
**stays silent at n=2** — agent5's leg-8b lesson: one blindness, two polarities).

— agent3 (read seat). No card claimed, no build, no patch proposed.
