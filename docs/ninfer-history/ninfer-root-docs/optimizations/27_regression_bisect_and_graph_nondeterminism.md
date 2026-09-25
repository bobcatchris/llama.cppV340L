# Doc 27 — Regression Bisection: the CUDA Graph Is Nondeterministic (2026-08-21)

## TL;DR
The "90.81 t/s / 92.6% acceptance" headline numbers **never existed as a reproducible
state**. They were single-run draws from a **nondeterministic CUDA-graph path**
introduced at `9a6aba8f`. The true, deterministic high-water mark — stable to the bit
across runs and commits — is **82.0 t/s, 85.6% acceptance** (reached at `e5d798cd`,
still valid at HEAD `63c12da6`).

**There is one bug, not three.** The graph verify path (`use_graph = true`, default
since `9a6aba8f`) makes draft acceptance a lottery: 55–96% depending on the draw,
vs a rock-stable 85.6% with `--no-graph`. At HEAD the graph mode is net-negative
(measured draws: 64.6–71.8% accept, 70.6–77.4 t/s, all below the no-graph 82.0/85.6%).

## How this was found
User reported "we hit 90 t/s before and now we are down 5 t/s; the agent appears to be
hallucinating" and requested a per-commit battery re-run. Battery =
`/home/intel/verify_battery.sh` (pp, plain, mtp×2 determinism pair, MTP-vs-plain A2
identity, vs saved baseline). All runs: `qwen3_8_27b.ninfer` (Q5 production artifact),
prompt "The capital of France is", k=3, 512 tokens.

## Full commit matrix (measured, this bisection)
| commit | change | MTP t/s | accept | det | A2 | verdict |
|---|---|---|---|---|---|---|
| cdf0f758 | (anchor, 21:17 battery) | 79.20 | 85.6 | PASS | PASS | GOOD |
| e5d798cd | OneShotAllReduce | 82.18 | 85.6 | PASS | PASS | **GOOD — true high-water** |
| 4a87951f | Q4 draft head opt-in | 82.01 | 85.6 | PASS | PASS | GOOD (bit-identical default) |
| 9a6aba8f | CUDA graph verify (default on) | 73.30 | 65.5 | **FAIL** | **FAIL** | **BAD — nondeterminism introduced** |
| f3614a83 | OneShotArgmax | 92.86 | 96.0 | FAIL | FAIL | lucky draw (see below) |
| 7908592a | "robust argmax reset" fix | 90.82 | 92.4 | FAIL | FAIL | lucky draw |
| 6849071a | r8c5 T=5 schedules | 89.31 / 92.36 | 90.1 / 95.0 | FAIL | FAIL | 2-run spread = variance |
| cdce43a8 | round0 drafts1 + barrier | 91.74 | 94.0 | FAIL | FAIL | lucky draw |
| f45200b1 | "adaptive weight loader" | 66.39 (graph) | 55.1 (graph) | FAIL | PASS | **loader is innocent** (see proof) |
| 63c12da6 (HEAD) | zero-copy hidden | 70.6–77.4 (graph) | 61.5–71.8 (graph) | FAIL | FAIL | graph draws all bad |

`--no-graph` at f45200b1 and HEAD (3 runs each): **82.07–82.14 t/s, 85.6% (370/432),
3.58 tok/round — bit-identical every run.** This is the proof:
1. The loader change in `f45200b1` is behaviorally neutral (same 82.1/85.6 as
   `e5d798cd`/`4a87951f`). The 66.39/55.1% "regression" was a bad graph draw.
2. All nondeterminism is in the graph path.
3. OneShotArgmax (`f3614a83`+`7908592a`) contributes **~0 in deterministic mode**
   (82.18 → 82.11). Its "90.81/92.6%" claim was a graph-luck artifact.

## Revised truth about the doc 18–24 "speedups"
| claimed milestone | deterministic reality |
|---|---|
| one-shot AR → 81.92 | **REAL**: +3.0 t/s (79.20 → 82.18), deterministic |
| CUDA graph → 88.73 | **ILLUSORY**: graph mode is a lottery; no-graph stays 82.1 |
| OneShotArgmax → 90.81/92.6% | **ILLUSORY**: +0.0 in deterministic mode |
| r8c5 T=5 (k=4) | kernel fix real (verify T=5 58.9→37.3), k=4 still < k=3 |
| docs 25/26 all-Q4 "85.02 vs 79.22 baseline" | **STALE BASELINE**: vs true production 82.0, all-Q4 is a **regression** (−85.6→78.0% accept); also quantizes the target model (quality risk, unmeasured) |

Doc 26's "89.2% HBM saturation" also uses 288 GB/s; our Window #1 microbench measured
~427 GB/s peak (GEMV runs at ~99% of *that*, per doc 16). True utilization ≈ 60%.

## Root-cause direction (for the fix)
Graph mode changes the verify computation non-deterministically per round/run, which
shifts target argmaxes → draft agreement becomes a lottery. Prime suspects:
1. **One-shot AR/argmax pinned-memory polling inside graph replay** — the kernel spins
   on `*peer_flag < expected_epoch` reading host-mapped pinned memory; under replay the
   epoch/flag lifecycle may desync (host `advance_epoch` is stream-ordered but the
   replayed kernel may poll before/after the peer's write lands).
2. **Per-round state not properly indirected**: KV cache position/cursor, GDN state
   slot, MTP-head KV position, or drafts/hidden pointers baked into the graph instead
   of read from stable device memory each replay. (Doc 20 already raised the
   "per-round KV position indirection" question — it was the right instinct.)
Diagnostic that works: run k=3 twice with `--no-graph` (deterministic) and twice with
graph, diff the round-0 token sequences; instrument verify logits before/after replay.

## Required actions
1. **Ship `--no-graph` as the default** until the race is fixed (one line:
   `bool use_graph = false;`). The graph is net-negative at HEAD.
2. **Fix the graph race** (or delete the graph path) — only then re-measure graph value.
3. **Battery is now deterministic by default** (MTP runs use `--no-graph`); graph mode
   is reported as INFO-only (t/s + accept, no verdict). Baseline updated to the
   deterministic numbers (82.0/85.6). Determinism check now requires non-empty text.
4. After determinism is restored in graph mode (or it is removed), re-rank the
   remaining levers: (a) ~5ms host wall, (b) INT4/GPTQ verify GEMV (Obj 4b — GPTQ
   artifact exists at `/home/intel/models/qwen3_8_27b_q4_gptq.ninfer`, but note docs
   25/26's all-Q4 results were measured against a stale baseline AND quantize the
   target → re-measure in deterministic mode with quality check before believing them),
   (c) k=4 draft quality.
5. Docs 25/26 need correction headers (stale baseline, HBM figure, all-Q4 = regression).

## State
- Repo: `mtp-perf` @ `63c12da6`, clean.
- Battery: `/home/intel/verify_battery.sh` (deterministic default + graph INFO).
- Baseline: `/home/intel/verify_baseline.json` = deterministic HEAD numbers.
- Working notes: `/home/intel/bisect_notes.md`; raw logs in `/home/intel/verify_logs/`
  (timestamped per commit).
