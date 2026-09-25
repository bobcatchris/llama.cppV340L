# WO_K4V4_EVIDENCE — cold cold-pair prefill re-run + k5v4 family-completion leg

## PROGRESS LOG (newest first)

- **2026-09-20T02:25Z — WINDOW RELEASED (all legs complete, canonical restored).**
  Close law satisfied: my k5v4 boot retired by the law form; canonical
  `bash /home/chris/serve_fast.sh` (bin 2c8901d3, canonical env) booted, /health ok,
  **FAST PROBE PASS wall=24.5s finish=stop** — the GQA width=53 warmup-fault class did
  NOT hit this restore (intermittent class; gates+ovf closes hit it, mine did not — no
  incident, no kern.log capture needed). Canonical serve LEFT RUNNING on :8100.
  Window free for the coordinator's G-MM-2. Desk summary: LEG A = k4v4-vs-k4v2 cold
  cold-pair complete (delta -0.03..-0.09% ttft-paired at plen-2075, sign not flipped,
  magnitude claim dead at 2k, 10k magnitude unquotable); LEG B = k5v4 family leg GREEN
  (boots, coherent, 12,240 B/t measured vs 12,234 predicted, battery+needle exact).
  All evidence on amd/wo-k5v4 under results/amd/k5v4/ + this log; row file
  W7_k4v4_coldpair_row.txt.

- **2026-09-20T02:15Z — LEG B COMPLETE: k5v4 BOOTS, COHERENT, 12,240 B/t — FAMILY
  ANSWER YES.** Boot f3312f25 `--kv-dtype kvarn_k5v4 --max-context 8192
  --kv-capacity 8192`: health ok, 4 ranks materialized, UNIFIED decode route, zero
  stub-die, zero death, no fault. **Preflight names the real k5v4 KV unit:
  kv(12240 B/t x 8192 tok + mtp 2) 95 MiB — measured 12,240 B/t/rank vs the 12,234
  prediction = +0.049%, prediction VERIFIED.** Family ladder measured on HIP:
  k4v2 8976 < k4v4 11152 < k5v4 12240 << bf16 34816 B/t/rank (k5v4 = +9.7% over k4v4,
  35.2% of bf16). Preflight receipt: required 7048/8160 MiB (slack 1111) — allocator
  gate, no estimated refusals anywhere. G-Q1 5/5 finish=stop coherent no-mojibake
  (essay 670/story 559/count 239/code 306/stop 211); G-Q2 3k needle 3/3 EXACT
  (ORCHID-TUNNEL/MARBLE-COMPASS/CINDER-HARBOR). 2k prefill wall: ttft 25097 ms,
  82.7 tok/s vs k4v4 cold 111-113 tok/s => k5v4 ~-26% at 2k — THERMAL CAVEAT LOUD:
  no cold gate on LEG B per WO; probe ran post-battery at ~84-85 C (k5v4's own needle
  prefills sagged 105.8->86.1 tok/s across the battery); matched-temperature
  k5v4-vs-k4v4 perf is UNMEASURED — boot/cost/coherence facts are
  temperature-independent. All banked: gates_transcripts_legB_k5v4/, resp + sidebands,
  row updated. NEXT: close — law-form retire, canonical serve_fast attempt, FAST
  PROBE; fault-class incident #3 goes to coordinator.

- **2026-09-20T02:07Z — WINDOW CLAIM: LEG B (k5v4 family boot).** Continuation of the
  LEG A hold (same coordinator grant; no other desk booted since 01:35Z). Retiring MY
  arm-2 k4v2 boot by the law form inside the boot script, then f3312f25
  `--kv-dtype kvarn_k5v4 --max-context 8192 --kv-capacity 8192`. No cold gate required
  for LEG B (boot/behavior/cost leg, not a thermal comparison). Any GQA width=53 warmup
  fault = incident #3 protocol: no retry, kern.log capture, retire, BLOCKER tag, stop.

- **2026-09-20T02:05Z — LEG A VERDICT (both arms done, cold, clean). DELTA DOES NOT FLIP
  SIGN BUT MAGNITUDE COLLAPSES TO NOISE.** Arm 2 k4v2: cold-gate PASS 44.0 C max edge
  after 481 s logged soak (70->44 curve banked); boot clean (preflight receipt: required
  7263/8160 MiB, context 316 MiB); probes 20.47/20.71/20.92 s all stop/prompt=2075,
  completion=54, zero mojibake. **The pair:** server ttft (clean prefill measure):
  k4v4 18295/18518/18693 ms (113.4/112.1/111.0 tok/s) vs k4v2 18302/18534/18699 ms
  (113.4/112.0/111.0 tok/s) => k4v4 faster in ALL 3 ordinals but by 0.03-0.09% = NOISE.
  Client-wall "-3.7%" is a decode-length artifact (completions 35 vs 54 tok at ~24 tok/s
  = the whole 0.78 s gap) — never quote it as prefill. **mclk bands:** BOTH arms mclk
  lvl3-945 MHz every probe pre+post sample, all 4 GPUs (vs the confounded bank's
  k4v2 lvl0-167 / k4v4 lvl1-500); temps within 1-2 C at every matched sample
  (p1_pre 53/40/53/45 vs 54/38/52/44; p3_post 83/63/82/69 vs 84/63/82/70). Within-pair
  ttft ramp +2.18%/+2.17% monotonic both arms (marginally over the +-2% gate; identical
  shape, paired delta stable -0.03..-0.09%). **VERDICT: the PLOG-076 banked win
  (-13..-36%) does not survive de-confounding at plen-2075 — quote k4v4-vs-k4v2 prefill
  as "no advantage at 2k cold"; the 10k-geometry magnitude stays UNQUOTABLE (cold 10k
  cell unmeasured; kvarn KV traffic scales with position so a real 10k divergence is not
  excluded — it is just not evidenced). Direction "k4v4 not slower" HOLDS cold.** Decode
  sideband: k4v2 rate +1-3%, MTP acceptance k4v2 81.0% vs k4v4 78.6% (+2.4 pp, same
  direction as gates G-Q3). Row banked: results/amd/k5v4/W7_k4v4_coldpair_row.txt.
  NEXT: WINDOW CLAIM LEG B, boot kvarn_k5v4 @8192.

- **2026-09-20T01:50Z — LEG A ARM 1 (kvarn_k4v4 @36352) DONE — cold, clean.** Cold gate
  PASS: edge 41/30/42/35 C, max 42.0 <45 (0 s wait — box idle since ovf retire 01:35Z).
  Boot f3312f25 `--kv-dtype kvarn_k4v4 --max-context 36352 --kv-capacity 36352`: health OK;
  preflight receipt: required 7339 / usable 8160 MiB, fixed 6947, context 391 MiB (=
  11152 B/t x 36352 tok / 1024^2 rounded — matches the gates desk's measured 11152 B/t
  k4v4 constant), slack 820. POSTBOOT max edge 53 C (boot warmup heat, expected; arm 2
  gets the identical warmup before its probes). 3x plen-2075 mt64 probes (identical banked
  body, temp 0): wall 19.71 / 19.94 / 20.14 s, ALL finish=stop, prompt=2075,
  completion=35 identical across all 3 (greedy determinism), zero mojibake. Spread
  0.43 s = +2.2% monotonic drift (post-boot thermal ramp) — ordinal pairing vs arm 2 is
  the verdict basis, per PLOG-060. Sidebands banked: sb_arm1_k4v4_{armstart,p1..p3_{pre,post}}.txt.
  Next: retire (law form), cold-soak, ARM 2 kvarn_k4v2.

- **2026-09-20T01:39Z — WINDOW CLAIM: LEG A (cold cold-pair k4v4-vs-k4v2).** Authority:
  coordinator grant (G-MM-2 stood down; LEG A rank-0) + ovf desk "WINDOW RELEASED
  2026-09-20T01:25Z (coordinator owns restore)" in its WO + pgrep EMPTY since 01:35:10Z.
  Box history this watch: ovf ran 5 RED boots (00:44-01:35Z, last = red6 gdb session
  01:23-01:35Z); dies 27-32 C edge at 01:15Z mid-gap, now idle-cooling from the last boot.
  Holding the window through LEG A (both arms, law-form retire between) + LEG B, then
  law-form retire + canonical-restore attempt (coordinator owns recovery if the GQA
  width=53 fault class hits — incident #3 per coordinator playbook v3).

- **2026-09-20T00:55Z — desk opened, GPU-free prep done, window QUEUED.** Worktree
  `/home/chris/worktrees/amd-wo-k5v4` created off `amd/main` (c23be7c44), branch
  `amd/wo-k5v4`. NO builds (law). Banked bin verified present:
  `/home/chris/artifacts_bin/ninfer-serve_f3312f25c8201da0.bin`; model path verified
  `/home/chris/models/qwen3_8_27b_nvfp4.ninfer` (18 GB NVMe copy). **df -h / = 2.2 G free
  (99%) — binding; this desk writes text-only MBs and flags if it drops near <1 G.**
  Prep artifacts committed on this branch: `tools/v340l/k5v4_leg_boot.sh` (banked-bin boot,
  canonical env from serve_10k.sh, law-form retire inside), `tools/v340l/k5v4_coldpair_probe.sh`
  (timed probe + rocm-smi clocks/temps sidebands before AND after — the WO law: no sidebands,
  invalid probe), `tools/v340l/k5v4_coldpair_arm.sh` (cold-gate: all-4-die edge <45 C before
  boot, 30 s poll, 5-min WO budget with logged 2-min extensions hard-capped 30 min; 3x
  plen-2075 mt64 probes), `results/amd/k5v4/probe_body_plen2075_mt64.json` (byte-identical
  FAST-PROBE body, banked once so both arms fire the SAME bytes; note: the body file is
  12,234 BYTES — pure coincidence with the 12,234 B/t k5v4 prediction, no meaning).
  Replay bodies read from `/home/chris/worktrees/amd-wo-k4v4gates/results/amd/k4v4gates/`
  (W7_k4v4_phase2_row.txt = confounded baseline: ordinals -35.8/-33.4/-13.3%, k4v2 arm
  mclk lvl0-167/sclk-560 hot vs k4v4 lvl1-500/sclk-991). Window status: ovf desk holds it
  (RED repro boot PID live on :8100, no "WINDOW RELEASED" in its WO yet); dies 54-69 C edge.
  Polling per protocol; boot only after pgrep empty + WINDOW CLAIM appended here.

- **Why:** (1) the gates desk banked the k4v4 prefill-pair win (k4v4 faster every ordinal,
  -13 to -36%) with a THERMAL CONFOUND (k4v2 arm hotter, mclk lvl0-167 vs lvl1-500) — the
  magnitude needs a COLD cold-pair before it is quotable as kernel fact (PLOG-076);
  (2) the principal asked whether the whole kvarn family came in with the port — k5v4
  (5-bit K, quality-up dial) ships in the ported slice6 header but has never booted on HIP.
  One boot + probe battery completes the family answer.
- **Desk:** agent #2 (slot freed by l2stall close). Worktree: `git worktree add
  /home/chris/worktrees/amd-wo-k5v4 amd/main -b amd/wo-k5v4`. NO builds needed — boot the
  BANKED bins (kvarn family bin f3312f25c8201da0 serves k4v4, k4v2 AND k5v4 — same TU).
- **WINDOW QUEUE (absolute):** the ovf desk (rank-0 blocker fix) claims the window first
  for its RED boot and will append "WINDOW RELEASED" to docs/amd/WO_KVARN_10K_OVERFLOW.md
  on its branch when it releases; the coordinator runs G-MM-2 next. Watch both
  (`/home/chris/worktrees/amd-wo-ovf/docs/amd/WO_KVARN_10K_OVERFLOW.md` + a pgrep of
  `^/home/chris/artifacts_bin/ninfer-serve` every few minutes); append YOUR
  "WINDOW CLAIM: <leg>" to THIS file on YOUR branch only when no serve is running, then
  boot. Retire law form ONLY (`pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"`).
  Never retire another desk's boot. Restore canonical serve_fast + health at every close.
- **LEG A — cold cold-pair (PLOG-060 design, thermal-controlled):**
  1. Verify dies COLD before each arm: rocm-smi edge temps <45 C on all 4 dies (cold-soak
     idle up to 5 min if needed — the thermal state function is the confound being removed).
  2. Arm 1: boot f3312f25 + `--kv-dtype kvarn_k4v4 --max-context 36352 --kv-capacity 36352`,
     3x plen-2075 mt64 temp-0 probes (the gates desk's bodies are replayable from
     /home/chris/worktrees/amd-wo-k4v4gates/results/amd/k4v4gates/), client wall each,
     clocks+temps sideband before/after every probe.
  3. Retire, COLD-SOAK to <45 C, Arm 2: same boot + `--kv-dtype kvarn_k4v2` (same context
     flags), same 3 probes, same sidebands.
  4. Verdict: +-2% within-pair; report the k4v4-vs-k4v2 wall delta with BOTH arms' mclk
     bands. If the delta flips sign vs the confounded run, SAY SO LOUD — that would mean
     the win was thermal, not kernel.
- **LEG B — k5v4 family leg:** boot f3312f25 + `--kv-dtype kvarn_k5v4 --max-context 8192
  --kv-capacity 8192` (small ctx first; the budget constant is 12,234 B/t — allocator is
  the gate). Battery: the 5-probe behavioral battery + one 3k needle (tools on amd/main:
  w7_k4v4_gates.sh + w7_needle_gen.py). Record: boot OK?, outputs coherent?, KV bytes/tok
  from the preflight line (names the real k5v4 unit vs the 12,234 prediction), quick prefill
  wall at 2k vs the k4v4 number. Any stub-die/death = retire + bank the row LOUD (that
  would mean a slice6-k5v4-specific defect).
- **Bank everything:** results/amd/k5v4/ on YOUR branch (row file W7_k4v4_coldpair_row.txt +
  transcripts + boot logs). PROGRESS LOG newest-first after every step in THIS file on your
  branch. df -h / before big writes. No builds in the shared checkout EVER.
- **Laws:** no estimated VRAM refusals (the preflight is the gate — bank its lines); retire
  law form only; clocks sidebands with every number; infra = tag "BLOCKER:".
