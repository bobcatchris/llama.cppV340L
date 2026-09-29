# Speedup Ranking — NInfer TP2 on 2× RTX 5060 Ti (as of 2026-08-20)

Target model: **Qwen3.8-27B** (`qwen3_8_27b.ninfer`, groupwise-int). All numbers battery-verified
(`tools/verify_battery.sh`, baseline `tools/verify_baseline.json`) unless noted.
Stock reference: llama.cpp 2×5060 Ti TP + MTP k=4 = **19.15 t/s at ~4k ctx** (published stock
number — our tests run at ~200–550 ctx, an easier condition, so the 4.1× below is slightly
favorable to us).

> ## ⚠️ CORRECTION (doc 27, 2026-08-21 — per-commit re-bisection)
> The **90.81 t/s / 92.6%** headline (rows 10–11, "4.7×") **is not a reproducible state.**
> Per-commit battery bisection (doc 27) found the **CUDA-graph verify path is
> nondeterministic** (introduced `9a6aba8f`): graph mode turns draft acceptance into a
> lottery (55–96% per run), and the 88.73/90.81/92.6% numbers were single lucky draws.
> **True deterministic high-water mark: 82.0 t/s, 85.6% acceptance** (reached at
> `e5d798cd` one-shot AR; bit-identical across runs; still valid at HEAD via
> `--no-graph`). OneShotArgmax contributes ~0 in deterministic mode. The graph is
> **net-negative at HEAD** (graph draws 64.6–71.8% accept < 85.6% no-graph) and the
> default has been flipped to `--no-graph` until the race is fixed. Real deterministic
> gain to date: **19.15 → 82.0 t/s = 4.3×**. See doc 27 for the full matrix.

## 1. Milestone timeline

| Step | What | Measured | Δ vs prev | Notes |
|---|---|---|---|---|
| 0 | llama.cpp stock (TP, MTP k=4, 4k ctx) | 19.15 t/s | — | published, not our run |
| 1 | PP2 (layer split, decode only) | 23.8 t/s | +4.7 | 41.6 ms/step; proved 2-rank pipeline |
| 2 | TP2 Phase 2a (GEMV+MLP split) | ~29 t/s | +5.2 | 34 ms/step |
| 3 | TP2 Phase 2b (attn + GDN full split) | 32.5–33.3 t/s | +3.6 | 30.1–30.8 ms/tok |
| 4 | MTP k=3 first clean run (128 tok) | 44.67 t/s | +11.4 | 64.4% acceptance, a=1.9 |
| 5 | MTP steady state (512 tok) + A1/B fixes | 53.77 t/s | +9.1 | run-length convergence + protocol fixes; verify 38.26 ms |
| 6 | P1: ColumnN lm_head + MMA small-T + stream-ordered AR | 60.88 t/s | +7.1 | round 50.76→44.48 ms; plain 18.94→21.28 |
| 7 | **Obj 1: draft vocab (40,960 rows)** | **79.20 t/s** | **+18.3** | acceptance 64.9→85.6%, 3.58 tok/round, round 40.20 ms |
| 8 | cdf0f758: attn geometry + T≤7 dispatch | k3 unchanged; k4 round 68.19→63.88 ms | k4 only | k4 = 47.11 t/s, still < k3 |
| 9 | One-shot AR on mapped pinned host (doc 18) | 80.77 → 81.92 | +1.2 | only ~1.5 ms of the 7.5 ms NCCL time was exposed | 
| 10 | **CUDA-graph verify round + epoch-tracked AR (doc 20)** | 81.99 → 88.73 ~~+6.7~~ | — | **CORRECTED (doc 27): ILLUSORY** — graph path is nondeterministic; 88.73 was a lucky draw; no-graph stays 82.1. Net-negative at HEAD; default flipped to `--no-graph` |
| 11 | **OneShotArgmax + GQA TP2 routing (doc 21)** | ~~88.73 → 90.81~~ | — | **CORRECTED (doc 27): ILLUSORY** — +0.0 in deterministic mode (82.18→82.11); the "90.81/92.6%" was a graph-luck draw |
| 12 | k=4 investigation (doc 22) | closed | — | root cause: T=5 exceeds 4-token NVFP4 MMA tile → 2× HBM weight passes (c=4) or register spills (c=8) |

**Overall (corrected, doc 27): 19.15 → 82.0 t/s = 4.3×** (deterministic, `--no-graph`; the 90.81 figure was a nondeterministic graph draw and is not reproducible). Plain decode ~21.3 t/s. PP ~33 t/s
(driver's T=1 prefill loop — not a real prefill kernel; see doc 13 "out of scope").

**Recurring lesson (docs 18/20/21): kernel/host *duration* ≠ *exposed* cost.** NCCL AR
showed 7.5 ms/round of kernel time but only ~1.5 ms was exposed; OneShotArgmax removed host
barriers that were fully hidden. Always measure the phase delta, not the component microbench.

## 2. Ranked by contribution

| # | Optimization | Δ t/s | × | Useful? | V340L (2× V340L, 4× gfx900 dies, 32 GB) |
|---|---|---|---|---|---|
| 1 | **TP2 + MTP foundation** (weight sharding, MTP round protocol, GDN slot protocol) | +25.5 (19.15→44.67) | 2.33× | **YES — the whole engine** | Portable at technique level. Requires HIP/gfx900 kernel port. v100-skinny's SIMT skinny-kernel idea (r8c4/r8c8, ~94% BW on 5060 Ti) originates from Vega-style design — **directly the right shape for gfx900**, which has the same 32-lane SIMT structure. Highest-value port target. |
| 2 | **Draft vocab head (Obj 1)**: 40,960-row slice of output_head for drafter | +18.3 (60.88→79.20) | 1.30× | **YES — biggest single change** | Fully portable (model property, hardware-invariant). 40,960 IDs artifact reusable. Also cuts ~1.1 GB VRAM/rank — matters on 16 GB V340L cards (27B is tight there). |
| 3 | **MTP stabilization** (prefill, GDN snapshot OOB, lockstep, acceptance fixes; docs 08–11) | +9.1 (44.67→53.77) | 1.20× | **YES — correctness enabler** (some Δ is run-length convergence) | Protocol is GPU-agnostic; ports with the engine. gfx900 will have its own driver quirks — keep the acceptance-probe + A2 identity harness. |
| 4 | **P1: ColumnN lm_head sharding** (124,160 rows/rank + host argmax exchange) | +7.1 (53.77→60.88) | 1.12× | **YES** | Portable; on V340L the cross-die argmax exchange rides the PCIe bridge — keep it to 8 bytes (max_val + tok_id), as implemented. |
| 5 | **One-shot AR, mapped-pinned host + wt/cv PTX (doc 18, DONE)** | +1.2 (80.77→81.92) | 1.01× | **YES** — but only ~1.5 ms was exposed, not the 6.4 ms the microbench predicted | **MOST valuable AR pattern for V340L**: no GPU P2P exists cross-die there, so host-mapped staging is mandatory, not optional. Ports directly (same wt/cv coherency logic works over PCIe). |
| 6 | **T≤7 small-T GEMV dispatch** (cdf0f758) | k4 only: round −4.3 ms | 1.07× (k4 path) | **YES but k4 still loses** (see §3) | Portable with the SIMT kernel port; keep the t≤7 threshold logic. |
| 7 | **FP16 GDN recurrent state** (dc20c949) | not isolated (inside 79.20) | ~1.01× | **Marginal on 5060 Ti** | Probably marginally **more** useful on gfx900: HBM2 bandwidth (~165 GB/s/die) is lower, so halving state read/write traffic helps the GDN verify steps proportionally more. Cheap to port (dtype change). |
| 8 | **CUDA-graph verify round (doc 20, DONE → CORRECTED doc 27)** | ~~+6.7~~ → 0 (nondeterministic) | — | **NO (as of doc 27)** — path is nondeterministic (acceptance lottery 55–96%); net-negative at HEAD; default flipped to `--no-graph`. Revisit only after the race is fixed (doc 27 §root-cause) | Portable in principle; **ROCm hipGraph on gfx900 is weaker/less battle-tested than CUDA** — expect capture quirks. Keep the epoch-indirection pattern (device-side round counter) since host-mapped AR needs it too. Deprioritize vs one-shot AR/argmax. |
| 9 | **OneShotArgmax device-level proposal exchange (doc 21, DONE → CORRECTED doc 27)** | ~~+2.1~~ → ~0 | ~1.00× | **MARGINAL (corrected)** — +0.0 in deterministic mode (82.18→82.11); keep (harmless, may matter post-graph-fix) | Portable (host-mapped staging again). Note: acceptance is now **per-execution-mode** (85.6→89.7→92.6 across refactors — close-margin argmax flips change trajectories). V340L A2 gates must be scoped per mode. |
| 10 | **INT4 verify GEMV (Obj 4b, GPTQ-calibrated)** | proj. +50–70 (~140–160 t/s) | 1.5–1.8× | **OPEN — the biggest remaining lever** (draft-head INT4, doc 19, was the wrong target: −0.22 ms, gate failed, W8 retained) | Portable; V340L's lower HBM2 bandwidth makes the 2× weight-bandwidth reduction **proportionally bigger** — higher-priority there than here. Needs GPTQ calibration (RTN fails: doc 19) + A2 gate redefined vs W8G32 reference. |
| 11 | **r8c5 single-pass T=5 schedule (doc 24, DONE `6849071a`)** | k4: 58.89→37.27 ms verify; k4 t/s 47.75→**76.8–85.0** | 1.0 (k3 unchanged) | **Kernel win real, config still loses** — k4 best (85.0) < k3 (90.81). My 105–120 projection was wrong: see §4 break-even. | **The kernel pattern is the portable part**: 40-reg no-spill single-pass c=5 proves high-c tiles work in SIMT. On gfx900 (254 VGPR headroom, no NVFP4 tiles) c=8/16 single-pass T=5/6 tiles should be even easier — do a T-schedule analysis in the SIMT port. **But the k decision is draft-quality-bound, not kernel-bound** (hardware-invariant): marginal acceptance of the 4th draft is 37% here; k=4 only wins if that reaches ~50%+. |

## 3. Marked NOT useful (null / negative results)

| Change | Result | V340L note |
|---|---|---|
| **Split-KV attention for verify (Obj 2)** | **NULL**: attention = 0.2 ms = 0.5% of the 36.55 ms verify (doc 16). 3090-repo patch targeted T=1 single-GPU; our T-batched TP verify already saturates SMs via head-block parallelism. | **Also not useful there.** Same T-batched TP structure; on gfx900 the verify breakdown shifts toward GEMV/AR (lower HBM BW, PCIe AR), attention stays a sliver. Don't port the 3090 split-KV patch. |
| **DecodeSplitScale 1→2** (attn grid scale-up) | **NULL on k3**: verify 36.74→36.55 (noise). Zero measurable delta, exactly what a 0.5% item should produce. | **Not useful.** Same reason — attention is not the wall on either platform. |
| **C2: MTP head TP split** (pre-draft-vocab) | ~1.3 ms/round — right idea, wrong priority; superseded by draft vocab (made the draft head 6× smaller instead). | Moot — port draft vocab, not C2. |
| **k=4 MTP** | **CLOSED-DEAD on sm_120a** (doc 22): T=5 exceeds the 4-token NVFP4 MMA tile → c=4 chunks into 4+1 = **2× HBM weight passes** (54.73 ms verify); c=8 single-pass = 64 regs/thread = spills (58.89 ms). Best k=4 = 47.75 t/s vs 90.81 (k=3). | **The cliff is tile/register-driven, not bandwidth.** On gfx900 there are no NVFP4 MMA tiles (SIMT port) and 254 VGPR/thread headroom — a single-pass T=5/6 tile is likely feasible, so **k=4 legitimately reopens for V340L** after the SIMT T-schedule analysis. |
| **CUDA graph for PP2 decode** (early, pre-MTP) | Abandoned — PP2 was host-serialized per step, not launch-bound in a graphable way. | Status **reversed for the MTP round** (fixed shape, ~4 ms recoverable) — but see row 8: deprioritized for gfx900 due to ROCm graph maturity. |
| **GPTQ int4 lm_head (3090-repo technique)** | Superseded by draft vocab (−4.8 ms) which also fixed acceptance; int4 head now only applies to the *verify* path (Obj 4). | As Obj 4 above — useful, quality-gated, higher value on V340L. |

## 4. Current state & what's in flight

- **90.81 t/s** MTP k=3 (agent-verified docs 18–23; my battery re-verify pending), 92.6% acceptance,
  3.79 tok/round, round 36.38 ms phases (~41.7 ms wall incl. ~5 ms unmeasured host), verify 32.85 ms,
  plain ~21.3 t/s, PP ~33 t/s. Default path: graph on + one-shot AR + one-shot argmax + W8 draft head.
  Token identity is **per-execution-mode** (see row 9 note).
- Closed/dead: Obj 2 (split-KV, null), Obj 5 (greedy — already was), Obj 7 (k=4 on sm_120a).
- **Remaining levers ranked (2026-08-20, post doc 24):**
  1. **~5 ms unmeasured host wall** (top for the k=3 production config) — pipeline accept D2H behind bridge/propose; graph the MTP bridge (~0.3–0.5 ms); layer stream concurrency to hide launch latency (~1–2 ms). Cumulative +8–12 t/s.
  2. **INT4/GPTQ verify GEMV** (Obj 4b) — 23.5→~12 ms, ~140+ t/s, quality-gated (calibration mandatory per doc 19).
  3. **k=4 break-even via draft quality** (optional, research) — r8c5 fixed the kernel (doc 24); k=4 now loses only because the 4th draft's *marginal* acceptance is 37% vs the ~46–57% needed (break-even in §4). Better drafts (GPTQ-calibrated draft head, larger draft head, or 4th-draft-specific calibration) could flip k=4 to a win: +1 token/round at +4.4 ms.
  4. **One-off divergence probe** (eager vs graph vs OneShotArgmax, first 32 tokens) — validates the per-mode identity story.
  5. **Housekeeping** — battery re-verify + `--update-baseline` (k4 baseline 47.11 is stale — now 76.8–85.0), doc-13 numbering sync, push.
- V340L port priority (when cards land): **SIMT skinny kernels + T-schedule analysis (k=4 may be free there)** →
  **one-shot AR + OneShotArgmax (host-mapped patterns are mandatory there — no GPU P2P cross-die)** →
  draft vocab + ColumnN head → MTP protocol → INT4 verify (bigger relative value, HBM2 ~165 GB/s/die) →
  CUDA graph (last, hipGraph immaturity). Gates: A2/determinism scoped per execution mode.

### 4b. k=4 break-even (why the perfect T=5 kernel still loses)

Per-token cost (ms/token) = round wall ÷ tok/round:

| Config | round phases | +~5 ms host wall | tok/round | ms/token (wall) |
|---|---|---|---|---|
| k=3 (production) | 36.38 | 41.38 | 3.79 | **10.92** |
| k=4 r8c5 (doc 24) | 41.86 | 46.86 | 4.16 | **11.27** |

k=4 buys +0.37 tok/round for +5.48 ms wall → the marginal 4th-draft token costs **14.8 ms** vs
10.92 for the base token. k=4 only wins when the 4th draft's **marginal acceptance** m satisfies
(46.86)/(4.16+m) ≤ 10.92 → **m ≥ 0.10 absolute** beyond the current 0.37, i.e. **4th-draft marginal
acceptance ≥ ~46%** (phases-only basis: ~57%). Current: 37% (a=3.16/4 vs k3 a=2.79/3 → Δ=0.37).

**Conclusion:** the T=5 kernel is fixed and optimal (37.27 ms ≈ predicted 37–38); the k=4 config
is now limited purely by draft quality — a hardware-invariant property. This is the same wall for
V340L: better drafts (GPTQ draft head, larger head, 4th-position calibration) are what would unlock
k=4/5 anywhere, not more kernel work.
