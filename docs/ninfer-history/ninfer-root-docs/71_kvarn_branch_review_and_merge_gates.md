# 71 — KVarN branch (wo/kvarn-pp) review: status vs docs/68, merge gates, doc 70 review

**Status:** REVIEW DELIVERABLE — 2026-08-25, main-side agent. Branch
`wo/kvarn-pp` @ `f76f00bd` (13 commits past merge-base `ee8f56ff`). No
uncommitted work on the branch.

---

## 1. Verdict

Priority #1 of docs/68 — **the beyond-wall decode collapse — is fixed and
live-verified.** The pp half of the mission and the closeout/DoD gates are
**not met as written**, and the branch is **not merge-ready**: one
merge-blocking verify FAIL (pre-existing, docs/68 §5.13) plus missing
closeout evidence (§3 below). Doc 66 (D-18) is closed with no pending
feedback.

## 2. Done (verified against branch evidence)

| Gate (docs/68) | Required | Actual | Evidence |
|---|---|---|---|
| 40k MTP-on decode | ≥32.8 tok/s (step-0: 17.5) | **60.6** (31.5k: 62.8) | `acd6f791` live + user's live log 2026-08-25: decode stable 49–71 tok/s to ~59k total context |
| MTP acceptance | within 5pp of 82.0 | 85.7% | `acd6f791` |
| In-capacity decode byte-identical | A/B byte compare | 0/3072 mismatch, kvarn tests 7/7 | `c970fa13` |
| Small-T path determinism | fixed summation order | enforced, documented 5e-3 gate deviation (beyond-wall only) | `060860fe` |
| Live end-to-end per §6 testing standard | real server, real path per step | every step commit carries a live result | `933a4e2a`, `8e579417`, `acd6f791`, `f76f00bd` |
| Final CI (ci mode) | green | T1,T2,T3,T5,T8 + S + battery PASS | `results/20260825_142740_*` |
| Live pp beyond wall (sanity) | no regression | 625–720 tok/s @ 32.9k prompt | user's live log, post-fix build |

**Root-cause correction (recorded, legitimate pivot):** the collapse was NOT
the `tokens == 1` decode branch (docs/66 §5.2 / docs/68 step-1 assumption).
Live profiling (`f6230dff`) showed it is the **MTP small-T verify rounds
(T=4) beyond the wall hitting the chunk-prefill 2-pass path** — pass-2 grid
collapses to 12 CTAs at T=4; cliff 88.8→20.8 tok/s exactly at the 30.7k wall.
Fix: route over-shadow T≤6 attention to the proven TC split-K small-T kernel
(`acd6f791`); materialize+flash stays the route (tiled direct-read lost:
19.5 vs 39.4, `8e579417`).

## 3. Gaps (must close before merge, in suggested order)

1. **G1 (merge-blocking, docs/68 §5.13):** `mtp_long_ok: false` in the final
   run (`results/20260825_142740_report.log`) — the pre-existing "MTP
   long-prefill >512 tok (MTP==plain)" FAIL. Root-cause and report required;
   "do not let it be merged past." Check first whether the D-19 small-T
   reroute (same code region) already changed it.
2. **G2:** Step 2 (Br sweep) was never executed — 60k/88k pp remains at
   step-0 baselines (641/607). Docs/68 mission requires ≥1.25× step-0
   (~801/759) **or a documented ceiling** with the smem/occupancy table
   showing why (Br 64→32→16, KV-tile sizing, 2 CTAs/SM target). Also
   close the derived 0.80 ms pass-1 number via direct event timing
   (`tests/bench_kvarn_2pass.cu`).
3. **G3:** Step 5 closeout not run — the final CI was `--mode ci`
   (T1,T2,T3,T5,T8 only). Required: `run_ci.sh --full` (int8 T9–T12, T14
   zone, KVarN battery T16–T19 @250k incl. **T19: 88k prefill ≥450**).
4. **G4:** DoD #2 live proof not committed: fresh 250k KVarN server on the
   final code, real **88k and 225k** prompts prefilled past the wall, launch
   command + `nvidia-smi` clocks recorded in the report.
5. **G5:** No committed post-final-change 10k MTP-on decode guard measured
   per §5.10 protocol (serve.log, 3-run median, 48 tokens). The step-1
   wall-clock probe (`results/d19_step1_decode_*`, ~45–52 tok/s from
   wall_s incl. tail prefill) is NOT a §5.10 measurement — re-measure on
   the final build to confirm ±2% of 65.6 before claiming the guard.
6. **G6:** No closeout report (DoD #5): one paragraph per step + key numbers
   table (pp per size, decode per ctx, acceptance, VRAM, smem/occupancy per
   Br if run, clocks) in `results/`.
7. **G7 (merge mechanics):** the branch's CI-script fix (`5ce33ded`)
   overlaps main's `c30c99e7`. Resolve by taking **main's version** — it is a
   strict superset (caller-tree FATAL guard, pre-launch port check,
   `/proc/PID/exe == BIN` verification).

Suggested sequence (one server session, when the live server is free):
G1 → G3+G4+G5 together → G2 (or documented ceiling) → G6 → merge (G7).

## 4. Doc 70 review (KVarN prefix reuse — emergency order)

Diagnosis matches live evidence (`kvarn reset inflight (long-prefill
re-prefill)` at 32.9k; ~48 s re-prefill every turn; P=16,384 gate in
`tp2_backend.cpp`). VRAM sanity checks out: 65k window ≈ 0.53 GiB/rank
fits current ~1.67 GiB/rank headroom (16,310 − 14,635 MiB); 128k ≈ 1.07
GiB/rank is tight, consistent with §5's fallback. Steps D (MTP draft
parity) and §5 (D-19 interaction) are correctly called out. Four fixes
requested:

- **F1 (self-contradictory guard):** §4.3 "startup VRAM log unchanged for
  default flags" contradicts step B's outcome — a larger reuse window adds
  VRAM by design. Reword: "startup VRAM ≤ 16,310 MiB − 256 MiB margin at
  the 250k KVarN config, and the effective reuse window + its VRAM cost are
  printed at startup."
- **F2 (eviction test may not evict):** §4.1.4's interleaved case at
  65k + 65k = 130k only exercises eviction if the fitted window is <130k.
  Make it explicit: the interleaved prefixes' sum must **exceed the
  effective window**, verified from log lines.
- **F3 (bit-exactness achievability):** a restored prefix can have a
  different paged/tile layout than a fresh prefill, which changes bf16
  summation order → ~1e-3 rel logits diff → greedy usually identical but
  near-tie flips are possible. Keep "any token difference is FAIL" as the
  gate, but add a fallback: log logits rel_l2; < 1e-2 ⇒ rounding (record,
  proceed), else corruption (STOP).
- **F4 (token accounting):** the N-token probe prompts must be verified via
  `usage.prompt_tokens`, never char estimates (docs/68 §3 calibration).
- Also: inherit docs/68 §7 server-swap protocol (port 8091 serves a live
  conversation — ask the user before swapping).

## 5. Doc 66 status

Closed: D-18 merged (`ad95d5af`/`879bc70d`). Its step-2 decode scope was
re-scoped into docs/68 and executed as D-19 with the root-cause correction
in §2 above (already recorded in branch commits — no doc 66 edits needed).
No pending feedback.
