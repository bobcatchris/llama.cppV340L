# WORK ORDER: M=256 prefill-chunk RE-LEG on the promoted mad-mix build (REV2 W5, scheduled "AFTER GEMM" — GEMM has now landed)

Owner: desk agent (slot A). PERFORMANCE desk. Phase 1 GPU-free prep; Phase 2 is the
serving window (QUEUED — the k4v4fin long-needle battery owns the GPU, then the
coordinator's G-BF-1 pairs; poll, do not take).

## WHY THIS IS THE TOP SCHEDULED LEVER (banked receipts; cite, do not re-derive)

- REV2 §4: "W5 thermal + M=256 re-leg AFTER GEMM: steady wall rises 979→1032 within one
  request (droop); the M=256 ladder measured +15 % with GEMM FLAT (receipt:
  results/amd/coherence/W7_row_ladder_chunk256.txt) — the faster the GEMM, the more the
  ladder's fixed-cost amortization matters."
- The GEMM step has NOW LANDED: mad-mix promoted (+8.83% serving wall, PLOG-080,
  bin cc122cc0ab616e4e). The re-leg condition "AFTER GEMM" is SATISFIED.
- The old ladder closure ("clean no-gain at 512", CLOSED SET) tested M=512 on the OLD
  GEMM; M=256 is the REV2-scheduled point. Do not redo 512.
- Geometry law: this leg runs at the STANDARD probe geometry (plen-2075, canonical env,
  ws96) where the G-MM-2 numbers live — NOT the kvarn long-ctx posture.

## PHASE 1 — PREP (GPU-free)

1. Read the +15% receipt (results/amd/coherence/W7_row_ladder_chunk256.txt, amd/main) and
   record what the +15% was measured ON (which bin, which clocks band, which probe).
2. Stage `tools/v340l/wo_chunk256_pairs.sh` in YOUR worktree by adapting the PLOG-060
   skeleton (tools/v340l/wo_madmix_gmm2_pairs.sh on amd/main): arms = C128 (canonical
   serve_10k BIN line, default chunk) vs C256 (SAME bin, `--prefill-chunk 256` added);
   3x plen-2075 mt64 probes per arm, alternating arm order across 3 boot pairs
   (basetip/tipbase/basetip pattern), +-2% within-pair law, clock sideband per boot,
   retire law EXACT form for own boots, canonical restore + health at exit. bash -n it.
3. VRAM note for the WO log: chunk workspace scales with M; the canonical ws96 boot at
   plen-2075 has gigabytes of measured slack (cite the boot preflight from any canonical
   log) — the allocator is the gate; NO estimated refusals.

## PHASE 2 — THE WINDOW (queued; one boot-pair cycle ~50 min)

- Window claim protocol: "WINDOW CLAIM: chunk256-leg" in this WO + pgrep
  `^/home/chris/artifacts_bin/ninfer-serve` EMPTY + no open claim in WO_BODY_FUSION.md
  (G-BF-1) or WO_Q4KV_K4V4.md (battery). Poll ~10 min.
- PRE-REGISTERED GATES (do not loosen): promote C256 as a serve_10k.sh default change
  ONLY IF (a) 9/9 probes BLUEPASS both arms, (b) within-pair spreads inside +-2%,
  (c) C256 median wall beats C128 by >=5.0%. Any quality FAIL (mojibake, finish!=stop)
  = kill. Rollback is trivial: serve_10k.sh unchanged.
- Bank: results/amd/chunk256/ (row + all walls + sidebands + verdict). PLOG row drafted
  in your report; coordinator appends to the chain.

## LAWS

- Own worktree + branch amd/wo-chunk256; progress log newest-first in THIS file after
  every step; resumable from this file alone. NO channel posts. NO cmake builds (both
  arms boot the SAME banked bin — zero builds). df -h / before >1G writes.
- Thinking-model contract: bodies carry "model":"qwen3.8-27b"; max_tokens>=128 for
  needle-class probes (the mt64 BLUE probe is proven fine from G-MM-2).

## PROGRESS LOG (newest-first)

### 2026-09-20 — RUNNER HARDENED pre-window (2 defects found by dry-run simulation; gates UNTOUCHED); poll continues

- **Defect 1 (verdict parser, RED row reproduced in /tmp dry-run):** the staged parser's
  boot-block regex `-- boot …\n((?:   .*\n)+)` stops at the FIRST non-3-space line after a
  boot — the `== sideband <tag>-start` block — so the probe lines (which sit between the
  start/end sidebands) were NEVER captured: a full 18-wall run would have been parsed as
  "WALLS INCOMPLETE … NO VERDICT". Simulated against a synthetic row in the runner's exact
  append order (boot → SERVING → sideband-start → 3 probes → sideband-end): staged regex
  captured **0/6** walls; fixed (split-on-boot-marker, scan whole segment) captures **6/6**;
  full 18-wall row → "C128 median 25.10s (spread 1.00%) | C256 median 22.90s (spread 0.87%)
  | C256 win +8.76% → PROMOTE C256"; FAIL-parity row → NO-PROMOTE; truncated row →
  INCOMPLETE (no verdict). All three verdict paths validated pre-window. NO gate loosened:
  (a) 9/9 BLUEPASS, (b) +-2% ordinal-median spreads, (c) >=5.0% C256 median win — verbatim.
- **Defect 2 (restore-at-exit on failure paths):** BOOT FAILED exited WITHOUT restoring
  canonical (law says restore + health at exit, any path). Fixed: boot-failure now retires
  the own (possibly half-booted) serve by the law form, runs `bash /home/chris/serve_fast.sh`,
  health-checks, THEN exits 1. RETIRE FAILED still exits for coordinator attention unchanged
  (a live zombie must be resolved before any restore anyway).
- Re-checked post-patch: `bash -n` OK; both embedded python blocks compile; boot line still
  byte-identical to serve_10k.sh's except the parameterized `--prefill-chunk`.
- **Window state:** k4v4fin battery STILL LIVE (serve pid 246819, bin 8ac1ba93eb7fbfad);
  WO_Q4KV_K4V4.md (amd/main) has NO battery-closed/canonical-restored entry yet;
  WO_BODY_FUSION.md has NO open "WINDOW CLAIM: G-BF-1" (grep count 0). Poll loop continues.

### 2026-09-20 — PHASE 1 COMPLETE (GPU-free prep); window polling begins

- **Receipt read (step 1).** The +15% was measured ON: bin `a943225ad6da661e` ("same as
  serving" at the time), 2026-09-18, window 7, lane amd/wo-w7-body
  (`results/amd/coherence/W7_row_ladder_chunk256.txt`). Probe: plen-2075 mt64 temp0.
  Env there: P2 known-good **MINUS** the ws96 override (arena free-sized for chunk 256) —
  this leg deliberately DIFFERS: canonical env incl. ws96, per the geometry law above.
  Numbers: 4-rank means per 256-token chunk — wall 2730-2840 ms (~2770) => 92.4 tok/s
  prefill class vs 80.1 at chunk 128 (+15%); GEMM FLAT 1748-1822 ms (~2.0-2.1 TF/s/die —
  M only tiles the grid, no free GEMM win); dn 541-554 (2.15 ms/tok vs 3.76 — launch-bound,
  amortizes better at M=256); gap 391-501 (~16% of chunk, token-proportional or worse);
  n=9 chunk-forwards per 2075 tokens (vs 17 at 128). Clocks sideband: PARTIAL coverage
  (leg tail only) — this runner takes full sidebands per boot.
- **Runner staged (step 2).** `tools/v340l/wo_chunk256_pairs.sh` adapted from the PLOG-060
  skeleton (`tools/v340l/wo_madmix_gmm2_pairs.sh`, amd/main): arms **C128** (canonical
  serve_10k BIN line, `--prefill-chunk 128` explicit) vs **C256** (SAME bin,
  `--prefill-chunk 256`) — BOTH arms boot the SAME promoted bank bin
  `/home/chris/artifacts_bin/ninfer-serve_cc122cc0ab616e4e.bin` (PLOG-080); the ONLY delta
  between arms is the chunk flag. 3 ordinals x fresh boots, order basetip/tipbase/basetip
  (C128C256 / C256C128 / C128C256); 3x plen-2075 mt64 temp0 probes per leg (9/arm);
  use/temp/mclk/sclk sidebands per boot; WINDOW GUARD aborts if a serve is already up
  (retire touches own boots only, EXACT law form); canonical serve_fast + health restore
  at exit. Checks: `bash -n` OK; both embedded python blocks compile; env block diffed
  BYTE-IDENTICAL vs serve_10k.sh; BIN default grep resolves to cc122cc0ab616e4e.
  Output row: `results/amd/chunk256/WO_CHUNK256_pairs_row.txt`.
- **VRAM note (step 3).** The chunk workspace scales with M; the canonical ws96 boot
  preflight (serve_fast.log, boot 2026-09-20 03:25:56) measures: required 8159 MiB /
  usable 8160 MiB, live-free 8160 MiB on all four dies, auto-KV sized capacity=36352
  tokens from MEASURED free; boot banner: "[tp2] NINFER_WORKSPACE_MIB=96 overrides the
  1024 MiB default work arena (…arena throws loudly on exhaustion)". Auto-KV sizing
  absorbs fixed-side growth by shrinking KV capacity (plen-2075 needs ~2k of 36k) — that
  is the measured slack, live-gated at boot. The runner has ZERO refusal constants: the
  allocator is the gate; an arena throw is a measured event recorded as leg FAIL.
- **Window state at staging.** k4v4fin battery LIVE (serve pid 246819, bin
  8ac1ba93eb7fbfad, kvarn_k4v4 flags, 4/4 dies at 100%); WO_Q4KV_K4V4.md on amd/main has
  NO battery-closed entry yet; WO_BODY_FUSION.md has NO open "WINDOW CLAIM: G-BF-1".
  Entering the ~10-min poll loop per dispatch. df -h / at staging: no large writes
  planned by this desk (rows/logs are KB-scale; runner records df at start).

### 2026-09-20 10:1x — RUN 1 INVALID (RED banked, no verdict): probe ==2075 class (third occurrence — all arms FAILed at 2134) + C256@ws96 arena throw (rank2 std::bad_alloc — live-refused posture; the +15% receipt era predates the ws96 canonical pin). FIX committed: range check + ws512 BOTH arms (fair pair). Re-run on pre-warmed box.

### 2026-09-20 10:2x — RUN 2 READ RULE (declared before any run-2 wall was read; consistent with the G-BF-1 re-run standard): ordinal 1 is DISCARDED (fast-band ramp — run-1 receipt + PLOG-064); verdict = ordinal 2-3 medians, +-2% within-pair, >=5.0% C256 promote bar unchanged. Both arms ws512 (run-2's fix); alternation as registered.

### 2026-09-20 10:4x — RUN 2 VERDICT: NO-PROMOTE (honest close). Declared read (ord 2-3, hot band): C128 med 45.11 vs C256 med 45.04 => C256 win +0.17% vs the 5.0% bar. The +15% chunk-level receipt (old-GEMM era, 1024 default arena) does NOT convert to request wall at plen-2075 on the current stack — the per-chunk fixed costs it amortized (finalize stall, pageable H2D) were already cured (REV2 §3, PLOG-058/059), so chunk count no longer sits on the request's critical path. Third instance of the conversion lesson (PLOG-082 V-arm, G-BF-1 now): CELL WINS THAT DELETE/HIDE LATENCY DO NOT AUTOMATICALLY CONVERT; only request-critical-path reductions do. C256 stays unpromoted; script + receipts banked on this branch.
