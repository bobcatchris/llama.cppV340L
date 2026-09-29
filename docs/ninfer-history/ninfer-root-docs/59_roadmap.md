# 59 — Roadmap (live)

**LAST RECONCILED: 09:10 EDT 2026-09-08** (post host-kv merge + LITH VRAM correction
+ W5 pause; reconciled against the coordinator ledger and the week's landed work).
**Status:** CURRENT — single source of truth for what happens next, in order.
Updated when decisions are made; each item links its work-order/design doc.
**Status column discipline:** the second column carries the live state — scan it to
see what's done; every status is one of: DONE · IN PROGRESS · BLOCKED · QUEUED ·
ON HOLD (user) · DISCUSSION · HORIZON. A stale roadmap is a coordinator error —
flag it (per the coordinator doc for your hardware line — see AGENTS.md "Which coordinator
doc applies"; §11.2 there makes this doc the canonical global roadmap).

---

## LANDED THIS WEEK (2026-09-07/08 — the sprint block)

| # | Status | Item | Notes |
|---|--------|------|-------|
| L1 | **DONE** (`0afc100f`/`25fab77d`) | **Host-KV Safety Net merged to main** — GPU↔RAM park/restore for over-long conversations | t6 wild-write straddle fixed; i8 group-scale planes captured/restored; literal 3/3 byte-identical with engagement (6 parks/3 restores/0 failures); CI cell C4 green. Docs/156 §18.21–§18.30 |
| L2 | **DONE** (`55a41b81`) | **LITH VRAM correction — refusal gate = measured floor + ctx + 256 MiB margin; fences tightened** (wait_gpus_free 500→50, serve-fence 100→20) | int8@200k launches WITHOUT override (measured 15,069 MiB, 780 spare, 199.8k tok @ 59.0 tok/s). Ledger: docs/VRAM_LEDGER.md. User sign-off 07:08 EDT |
| L3 | **DONE** (`48bff27f`) | **pkill era terminated** — all 9 system-wide name-kill sites (5 files) → PID-scoped/port-scoped shutdowns | Live-smoked (target killed, neighbor survived). [HKV-RESTORE]-class evidence greps fixed in gate_ci (bde10f08) |
| L4 | **DONE** (`b0e9a8f2`) | **CI Exit Contract Guard** — static linter (28/28) as the run_ci prerun gate | Kills mis-declared-results CI runs before they burn 90 min |
| L5 | **DONE** (`e278bd4f`, local) | **W5 confidence-break rank-race + conf_break fixes folded to main-local** — two-barrier consume, atomic break | Matrix 4/4 clean, byte-identical; DORMANT (W5 off — see BLOCKED row B1). NOT related to Adaptive MTP Depth — separate feature (docs/69) |
| L6 | **DONE** (merge `2e8f25b5`→`55a41b81`) | **NVFP4 KV tier landed on main** (single-seq verified 7/7; batched = reverse gap, see Q3) | Docs/117 §9 lineage |
| L7 | **DONE** (docs/100/108/118 lineage) | **Magic Dictionary machinery** — domain draft-vocab builder (byte-exact vs the official tokenizer across 256,916 lines), output counter + serving hook, observation-only | Landed; the acceptance A/B (does the specialized vocabulary clear the +3-point bar?) is DESIGNED but not yet run — the value question is open (docs/118) |
| L8 | **DONE** (docs/69; off by default) | **Adaptive MTP Depth** (its own feature — NOT W5) — acceptance-feedback-driven chain length in single-request mode | Off by default; batched mode = Inverse-Gap Lane (A-queue) |

## ACTIVE QUEUE (in order)

| # | Status | Item | Notes |
|---|--------|------|-------|
| A1 | **IN PROGRESS** (branch `wo/single-parity`, resumed) | **Single-Seq Parity Port (WO #1)** — draft-skip + §5 I1/I2 coverage separation into `run_tp2_request` + use_lookup verify | Carries single-request-only features toward shared code; user priority 06:15 EDT. Suspended for the ledger 06:40–07:08, resumed |
| A2 | **IN PROGRESS** (A1, WO §18, ~15–20 min instrument-don't-fix) | **W5 (confidence break) shortening defect dig — the BLOCKER on W5 activation** (W5 is unrelated to Adaptive MTP Depth) | tau-ON hangs at prompts ≳5k (dtype/env-independent; 2k matrix too narrow). Mechanism candidate: unsynchronized shared break-decision reads (00:10 block); discriminator trace `[w5d-brk]`. Paused-state design doc: docs/W5_PAUSED_STATE.md @ `03241d26` (route-map premise SUPERSEDED per docs/83 — shadow removed; 2048-boundary candidates N1 committed-vs-open-tile / N2 GDN chunk-frontier under re-derivation). W5 stays OFF; flip = user decision after this closes |
| A3 | **DISCUSSION** (user wants direction first) | **KV Defrag & Dynamic Compaction** | Runtime cleanup of fragmented conversation memory without re-prefill/restart. WO recalled from A1 (23:36). KVarN tail migration + host-KV page tables are the materials |
| A4 | **QUEUED** | **Host-RAM Ledger** | Exact measured system-RAM numbers for host-KV parking (the host arena pins up to 14 GiB) — long sessions must not exhaust computer RAM either. Depends on L1 (landed) |
| A6 | **QUEUED — AWAITING GEMINI ACCEPT/DECLINE** | **V340L AMD phase-gate tests PG-0…PG-F + `run_ci_amd.sh` wiring** (test lane) | Specs authored by agent1 in `docs/amd/v340l/00_scope_and_work_order.md` §6; implementation is gemini's exclusive lane (§7.x — NOT reassignable to A1/A2). Brief sent 2026-09-12 (hub msg #13). Zero-GPU work open now; GPU stages need a written grant |
| A5 | **PENDING USER** | **Matrix + roadmap review** | `wo/feature-matrix-roadmap` @ `9055849b` (plain-English dictionary, 5-way matrix, this roadmap) — redumped to the user's Desktop; PR to main on their word |

## PARKED / ON HOLD (user decisions)

| # | Status | Item | Notes |
|---|--------|------|-------|
| H1 | **ON HOLD (user decision)** | **DFlash (in-tree 6-layer drafter)** — BUILT, mechanism complete (context, feature feed, CUDA-graph profiles); 27B config switched off (`supported=false`, 0 draft words) | Would draft up to 15 tokens vs MTP's 7. Un-hold = config flip + validation battery |
| H2 | **PENDING USER FLIP (blocked by A2)** | **W5 (Confidence Break) Activation** | Flip-decision table ready (23:36 block): pre-fix non-deterministic hangs at rounds 1/14/28; post-fix 4/4 clean, 40 breaks fired, byte-identical — BUT the second defect (A2) blocks activation at ≥5k prompts. Separate feature from Adaptive MTP Depth (docs/69) |

## ESTABLISHED ROADMAP (older, still live — unchanged scope)

| # | Status | Item | Notes / work order |
|---|--------|------|--------------------|
| 1 | **DONE (2026-08-24)** | KVarN P2c steps 1–6 (pool layout, write path, read kernel, budget+guard, prefill staging D-15, prefix tails D-16); merged `6b535c76` | Verified on the MAIN binary at 250k; docs/54 §9 P2c + docs/61 (complete) |
| 1a | **CLOSED (2026-08-26)** | KVarN D-18/D-19 wall fix: **FIXED 2026-08-26 per docs/50** (`acd6f791`): T19 (80k prefill ≥450 tok/s) achieved, full CI 20260826_095551 green; yesterday's pass-8 decode-guard re-proved 160k–250k prefill green. The staged-shadow premise the row was written under is also gone (shadow deleted per docs/83; code-verified `tp2_backend.cpp:765`/`:927` — stage_pages=0) | docs/50 D-18/D-19 entries; docs/66 spec remains the feature scope. User ruling 09:18 EDT: the 're-measure pending' task was wasted motion (killed before cards were spent) — reconciled rows citing other docs must check those docs' status |
| 2 | **DONE (2026-08-24)** | Accepted PR audits #24891, #27173 (A/B/C) | docs/63–64 |
| 3 | **DONE** (`3ed0569a`) | Pi client `/v1/compact` wiring | docs/55; T15 live |
| 4 | **NEXT BIG FEATURE** | DFlash Path B + DFlash2 block drafter — official GGUF artifact + llama.cpp PR #27342 reference are PUBLIC; DFlash2 measured 4.80 tok/round vs MTP 3.58 (122–132 t/s on a 3090, same 27B weights) | docs/56 (Path B) + docs/151 (corrected mechanism: 1.92B 5-layer NON-autoregressive block drafter) + docs/152 (CUDA review checklist). Needs the recalibrated VRAM headroom; the in-tree DFlash drafter (H1) is the fallback drafter |
| 5 | **QUEUED after #4** | Single-RTX-5000 support: single-GPU load path + 27B on the single-device engine; primary target 5090 32 GB | docs/62: build is sm_120a-only — 5090/PRO-5000 are sm_100 (port + re-baseline, NOT unchanged); long context needs KVarN first (~29 GB @ 200k KVarN vs ~35 GB int8) |
| 6 | **QUEUED after #4–#5** | Perplexity / KLD tooling: `top_logprobs` endpoint + PPL/KLD script | docs/65; the permanent quality gate for #7, KVarN P4, DFlash diagnostics |
| 7 | **QUEUED after #6** | All-Q4 artifact validation (18.2 GB → ~13.5 GB) with #6's tooling + battery + MTP acceptance vs 82% | docs/33 (why selective q5 was the default); full-Q4 quality delta unmeasured |
| 8 | **HORIZON** | Multi-GPU (TP4+): N-rank state arrays, N-way collectives, full device list | Weight-sharding math already card-count-generic; motivation 5090×4; docs/60 |
| 9 | **IN PROGRESS** (lane scoped 2026-09-12; was HORIZON) | **V340L / AMD (HIP) port — ACTIVE, and the target machine is THIS host** | Global pointer + measured board facts: **docs/amd/README.md**; lane docs `docs/amd/v340l/`; branch `wo/v340l-hip`; CI `tools/ops/run_ci_amd.sh`. 4× gfx900 devices @ 7.98 GiB, 2 cards × 2 dies, ROCm 6.2.0-66. Older lead-up material still in `~/comfy_templates/v340l_optimization/` |

**Small open items (fold into whichever step touches the file):** D-08 remainder
(skip prefix-buffer allocation when `--no-prefix-reuse`). D-11 closed 2026-08-24
(mitigated via `setsid`; no external process management per user decision).
