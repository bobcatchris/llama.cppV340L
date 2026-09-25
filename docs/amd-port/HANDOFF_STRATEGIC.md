# HANDOFF: V340L Campaign - Strategic Planning Brief

Date: 2026-09-25 ~15:00 CDT. Written by the standing coordinator on owner
order (campaign stood down, 30-min timer removed). Audience: a strategic
planning agent scoping the next big lift. Read CAMPAIGN_DOSSIER.md for the
full technical backstory; this file is the distilled decision brief.

## 1. STATE OF RECORD (what exists today, all verified)

- Serving config of record: Qwen3.8-27B-ASCII-P1M.gguf (imatrix mixed
  ladder, NOT flat q4_0 - only KV is q4_0), TP4 across 4x AMD V340L
  (gfx900 Vega 10, 8 GiB/die, PCIe Gen3), MTP speculative decode,
  200k context, q4_0 KV, geometry 1024/1024.
- **decode 25.33 t/s, prefill 244.18 t/s** (ratcheted E-166; was
  24.69/217.71 two days ago). Acceptance 0.66-0.69.
- Promoted serving assets: fa40 q4_0-direct decode attention (E-137),
  prefill q4_0-direct tile arm (E-159/E-161, owner-approved, +12.2 pct
  prefill, text-identical at depth - verified invariance cell),
  ub1024 geometry (E-148), all share/env arms in BASEENV.
- Everything is on branch amd/v340-port-v2 and snapshot-pushed to the
  bobcatchris fork (backup/v340-port-v2). Last push: rev19 era.

## 2. THE MACHINE (operating constraint that dominates planning)

4 hard deaths on 09-25 (06:59, 08:47, 09:52, 11:15) under GPU churn;
41 crash-terminated sessions in last -x history. Forensics (E-162):
instant whole-PC resets, zero OS trace (no MCE/panic/thermal log),
no ECC on this board (memory errors invisible), no hardware watchdog.
NOT correlated with the svm hog counter (hog 19 survived; hog 0 died).
Prime suspects: power delivery under 4-GPU burst transients, or
non-ECC memory. Mitigation regime that held: watchdog v2 (kills serving
PIDs at sustained svm-hog >= 4 - 2 live saves on 09-25), one GPU item
per boot, >= 10 min boot spacing, <= ~6 guard cycles/hour. Under that
regime the box ran 4+ hours of churn with zero deaths.
=> ANY big-lift plan must budget for: state pushes after every landing,
one-item-per-boot pacing, and possible loss of an in-flight cell.

## 3. CLOSED DOORS (do not reopen without new information)

Measured dead this campaign, all with paired/oracle evidence:
- Clock forcing via power_dpm_sclk (E-149: loses pairs, HBM hazard).
- Allreduce transport replacement: peer-access and IPC both refused 0/12
  pairs (W38). Channel counts bracketed: NCCL 1-ch worse, 4-ch worse,
  default 2 optimal (E-155). NCCL_SHM_USE_CUDA_MEMCPY hangs init.
- W37 kernel ladder: Opp-1 RPB1 (oracle-fail + slower), Opp-2 q8_1 dedup
  (<0.3 pct served) + TP4 copies (load-bearing), Opp-3 attention
  fill-split (slower at all splits, DUST-class numerics) - E-156/157/165.
  The FA tile kernel is residency-saturated (126/128 CTAs already
  in flight, W43).
- L7 iq4_xs requant family (E-164): both variants breach quality bars
  (needle recall, acceptance -0.03..-0.06) with NO served speedup.
  Artifacts parked (tierA sha16 1e0f8250, tierAcons 18da578b).
- Older: tensor-split, tile-fp16, hipBLASLt, wide6, PP/tp2xpp2, chain-4.
- Detail: llama-quant --tensor-type-file patterns are FIRST-MATCH REGEX -
  anchor and escape every entry, census-verify every artifact.

## 4. OWNER DECISIONS STILL OPEN

1. U2 --long: 50k/100k prefill depth curve for the promoted arm (~7 h
   run). Script ready: /home/chris/run_u2_prefill_window.sh --long.
2. U1 v2 deep rerun (acceptance-vs-depth at 150k/199k, ignore_eos):
   sizes any future acceptance-side lever. Script ready:
   /home/chris/run_u1_window_v2.sh.
3. Hardware: memtest86+ overnight, PSU inspection (wattage/age/rails),
   optional ROCm 6.3+/kernel upgrade targeting the KFD svm death class.
4. Persistent-server protocol adopted (E-158/E-163): persist-2 = max 2
   cells per server boot.

## 5. STRATEGIC OPTIONS FOR THE NEXT BIG LIFT (ranked by our data)

A. STABILITY FIRST (prerequisite for everything): hardware inspection +
   ROCm/kernel upgrade. The death class caps all experiment throughput;
   every plan below multiplies by machine uptime. Cheapest de-risk:
   PSU check + memtest; then ROCm upgrade on a fresh boot.
B. DEPTH CURVE + ACCEPTANCE SCIENCE: U2 --long + U1 v2 (both scripts
   ready). The paper result (arxiv 2609.26333: decode-side quant hurts
   2-4x more than prefill-side) plus our untested 150k+ acceptance data
   would tell us where the real headroom is at 200k serving.
C. PAST-FAMILY KERNEL REDESIGN: the weight stream runs at 54.7 GB/s vs
   a 70-120 GB/s schedule ceiling (W37); past it lies a real redesign
   (LDS-y staging killed twice on gfx900 - needs a different attack,
   possibly the LUT-kernel idea from the paper: software LUT decode is
   gfx900-feasible). High effort, bounded by W37's family math.
D. TRUE LOW-BIT L7: needs the f16 source (~55 GB download) + imatrix
   regen + the paper's insight that decode-side quant is the dangerous
   side. Only worth it with C or a new kernel.
E. SERVING PLATFORM: persistent-server tiering is partially adopted;
   a full persistent daemon (server always warm, batteries attach)
   would cut churn (the death driver) AND wall time.

## 6. OPERATIONAL RULES THAT MUST SURVIVE

- AGENTS.md rules apply (ASCII, no pushes/PRs without owner, Assisted-by).
- Pushes to the bobcatchris fork ONLY (never duford/upstream), via
  /home/chris/push_backups.sh (branch pushes are pre-receive-blocked, E-115).
- Paired-design + oracle-before-timing + provenance fail-closed (the gate
  stack caught 3 bad arms this campaign - it pays for itself).
- sudo password 0911 via stdin only, never in any committed file.
- The svm-watchdog v2 timer must stay enabled on any GPU work.
- Dies are power-capped at 110 W each; PCI addresses 05:00/08:00/0d:00/10:00.

## 7. WHERE EVERYTHING LIVES

- Ledger (append-only): docs/amd-port/OPTIMIZATION_PLAN_TP3_200K.md (E-166).
- Plan: docs/amd-port/PLAN_CURRENT.md (rev19). Dossier: CAMPAIGN_DOSSIER.md
  (E-155 era). Receipts W17-W44: docs/amd-port/results/.
- Serving: /home/chris/launch_tp3_200k.sh. Battery: run_post_u1_battery.sh
  (windows incl. nccl_a1/a2/q81) + run_ci_q40merge.sh (CI wrapper) +
  run_persist_battery.sh (persistent protocol) + run_hb_finish.sh.
- Worktrees: wt-smallk (q81 arm, merged), wt-attn3 (fill falsifier),
  wt-mmvq2 (RPB1, closed), wt-q40prefill (merged), wt-oneshot-ar, wt-skew,
  wt-roofline, wt-wides2r (stale). Artifacts: gguf/*tierA*.gguf (parked).
