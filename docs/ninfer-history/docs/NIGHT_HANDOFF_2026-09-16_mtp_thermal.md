# NIGHT HANDOFF — 2026-09-16/17 MTP + thermal window (night seat)

> ## **WARNING — TP4 BOOTS FAULTING ON NODE-3 since 06:10Z 2026-09-17 — ROOT RESET REQUIRED**
> Every TP4 serve warmup aborts with `Memory access fault by GPU node-3 ... Reason: Page not
> present or supervisor privilege.` — 6/6 across BOTH binaries (pre-tuning 6c1fe9366f0ef83d,
> byte-reproducible from 9751436f6, 4/4 incl `--no-cuda-graph`; tuned e341a2072c6f5078, 2/2;
> the same tuned binary booted GREEN at 05:55Z). Machine/driver state, NOT a kernel
> regression (KFD=0, VRAM clean, single-die benches at full ceiling throughout). **Do not
> burn windows re-trying TP4 boots** — root driver reset / reboot required first, then
> re-open with BOOT_BATTERY.sh. Single-die benches OK. Full evidence:
> `results/amd/coherence/TP4FAULT_row.txt` (PLOG-030/031).

State-on-a-page for the next session. Every claim cites a banked row. Branch: `amd/tp4-cure`
(merged through `amd/main`, pushed). PERF_LOG chain head: **PLOG-028** (see §7).

## 1. MTP serving posture (the recommendation)

**Serve k=2, graphs-ON, CWD=repo root.** k=2 wins on real prose (10.79 vs 9.75 tok/s, acc
0.65 vs 0.51, same sky-blue prompt, first-probe-on-boot both arms —
`results/amd/coherence/REALK23_row.txt`, PLOG-025/026). The counting-shape k-table
(k3 "best" at 14.86 tok/s, TIMINGK3_row) does NOT transfer to prose.

Exact line (from repo root, graphs ON by default, k=2 default — spell it anyway):
```
setsid nohup env --allow-nvfp4-weights (N7, 2026-09-17; legacy env NINFER_ALLOW_NVFP4_TP2=1 still accepted, value-parsed) NINFER_WORKSPACE_MIB=96 \
  /home/chris/artifacts_bin/<BANKED>.bin <artifact>.ninfer \
  --port 8100 --devices 0,1,2,3 --prefill-chunk 128 --no-prefix-reuse \
  --prefix-cache-capacity 256 --greedy --default-max-tokens 16 --spec mtp --draft-tokens 2
```
KV capacity cost of MTP: 36352 tokens (auto-KV line at boot). max_tokens>=48 on
BLUE-body probes (E-4 class; >=192 for 10k legs).

## 2. Banked binaries inventory (boot from the bank, never a lane build path)

- `ninfer-serve_07ad7eccc0b97cc0.bin` — the window's perf reference (TIMING instrumentation,
  env-gated, commit d32e6c53). All k-sweep/thermal/E1 rows ran on it.
- `ninfer-serve_fcef463d655e8150.bin` — B3: draft-vocab path hardening (env → exe-relative →
  CWD + loud warning). count600 [ids] byte-identical to 07ad7ecc (B3HARDEN_row.txt).
- `ninfer-serve_c6361bde8bc0dcc1.bin` — C1b: accept-audit tap (`NINFER_ACCEPT_AUDIT=1`,
  default off). Token stream == E1A (passive proof, E1K1_row.txt).

## 3. Thermal law (the night's headline; all perf numbers are state-dependent)

- **Trigger is a minutes-scale integrator of sustained load duration — NOT edge temp.** At
  84C edge, same boot: no-gap after 5.5 min continuous load = 606.44 ms/round (2.82x slow);
  after 3 min idle = 222.62 ms/round (full recovery, sclk 991-1500 sysfs-verified) —
  `THARM1_row.txt`, PLOG-018/019. Mechanism visible: sclk collapses 1500→560-775 MHz at knee
  onset, spreading GPU0→others (`TIMINGPF2_clocks.log`, PLOG-017).
- **Duty-cycle steady state**: 5x [probe + 60s idle] walks knee 768→640→512→512→512 tok and
  LOCKS (~14.8 tok/s mean vs 24 cold) — `INTEG_row.txt`, PLOG-027. Sustained serving settles
  at a predictable ~1.75x prefill / ~2.6x decode degradation.
- **Operating rules**: keep continuous load <60 s OR take 2-3 min idle gaps; every perf row
  MUST carry a clocks+temps sideband (BOOT_BATTERY step 0c does this automatically);
  probes-on-a-thermal-streak are not comparable to cold probes.
- **Forced-clock A/B stays BLOCKED on root**: all rocm-smi set verbs + fan/power-cap are
  root-gated, no NOPASSWD (v340l/01:38-41). It fires the moment the box grants rocm-smi.

## 4. Verify owns the round (8-phase verdict, NINFER_TP2_TIMING=1)

count600, MTP k=2 graphs-ON, 211.95 ms/round mean (`TIMING1_row.txt`, PLOG-011):
Target Verify (T=3) **195.34 ms = 92.2%** · AR Draft Chain 7.61 (3.6%, ~76-78 collects —
AR-batching ceiling <4%) · Alignment Fwd 8.56 (4.0%) · bookkeeping 2+3+4+6+7 = 0.45 ms.
Verify T-sweep: intercept ~126.5 ms @T=1, +34.4 ms/position (T2→T3), +42.1 (T3→T4) —
superlinear. The lever is the TARGET step cost (launch-bound, ~10x off memory-bound ideal);
honest levers named in SPEED_PLAN §6-1. Use `NINFER_TP2_TIMING=1` for the B1 breakdown.

## 5. E1 / E-17: MTP-vs-non-MTP byte-divergence CLASSIFIED (not a bug)

- Non-MTP arm is deterministic (same-boot AND cross-boot byte-identical, `E1K1_row.txt`).
- Divergence point MOVES with k: k2 flips gen-pos ~2 (`486` vs `1472 220`), k1 flips ~15
  (`874 4799 1414` vs `1534 1132 1103`); k1+k2 concur at the second site (PLOG-024).
- **Accept path EXONERATED**: `NINFER_ACCEPT_AUDIT=1` tap (c6361bde) — 206 rounds, **0
  retained-mismatches**, tap passive (audited ids == E1A).
- **Margins (E1MARGIN_row.txt, PLOG-028)**: flip columns are bf16 near-ties/degenerate —
  p_am 0.17-0.55 at committed tokens; exact bf16 ties observed (`local_am=16 global_am=486
  p_am=1.0000`); gathered-bf16 argmax disagrees with committed allreduce argmax at several
  columns. Mechanism: T=k+1 batched verify reorders near-degenerate numeral logits.
- **Re-scoped E1 bar (BUGTRACK E-17)**: (i) same-geometry identity across boots/binaries —
  holds byte-level; (ii) acceptance never exceeds target argmax — holds by tap; (iii)
  cross-geometry delta characterized, flips confined to numeral near-ties, no derailment.
  Both divergence streams count 1..200 coherently. E-17 follow-up CLOSED (D1).

## 6. Guards, battery, discipline

- `tools/guards/check_bf16_low_guard.py` — G1-G5 host guard; G5 cites the hardened
  draft-vocab site (env→exe-relative→CWD, tp_engine.cpp ~:1080). `--selftest` = 21
  mutations PASS. Run GREEN at close (see §7).
- `tools/guards/BOOT_BATTERY.sh [port]` — per-window battery: guard → 0c clocks+temps
  sideband (5 s, sysfs pp_dpm_sclk) → 0b draft-vocab pre-flight → device cells (SKIP if not
  compiled) → coherence ladder vs an ALREADY-RUNNING server. Row + clocks log banked per run.
- Session laws that bit tonight: spawn ONLY at KFD=0 (two VOID spawns logged: C1b/C2 —
  previous server still held VRAM; boots clean-fail loudly, curls get answered by the
  survivor); sampler loops outlive `kill $!` (nohup parent) — kill the loop bash and
  verify with pgrep; `cd X && A &` backgrounds the cd — use absolute log paths.

## 7. Records + ranked open queue

- PERF_LOG_AMD.md: PLOG-001..028, hash-chained (head **cbb233e8b9217334** → PLOG-028).
  BUGTRACK: E-16 (coherence closed) → **E-17** (E1 classification, follow-up CLOSED).
- 1. **Forced-clock A/B** (needs root grant for rocm-smi set verbs) — names the throttle
  ceiling; design ready in SPEED_PLAN §6-3.
- 2. **Verify-forward launch-bound pricing** — extend B1 per-phase tap per-layer; levers:
  launch-count reduction, AR-layer batching inside the forward, weight-streaming layout.
- 3. **E-17 margin tap** — CLOSED tonight (D1); re-open only if a non-bf16 logit path ships.
- 4. **k-table non-transfer** — done (k=2 serving); re-check only if draft-head retraining
  or prose-heavy acceptance tuning lands.
- 5. Deep-ctx decode split (thermal vs depth) — needs the forced-clock cell; current 10k
  number 3.65 tok/s is soak-confounded (COOL10K_row.txt, PLOG-021).
