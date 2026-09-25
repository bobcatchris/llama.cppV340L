# WORK ORDER: KV quality upgrade k4v2 -> k4v4 + 200k context ceiling (the q4-KV milestone)

Owner: desk agent (dispatch when a slot frees; <=2 agents total per principal order).
Coordinator owns infrastructure; you execute. This plan is self-sufficient — work top to
bottom, append "## PROGRESS LOG" (newest-first) after EVERY step.

## CONTEXT YOU MUST KNOW (verified in-tree, do not re-derive)

- The KV cache is ALREADY quantized in production: KVarN k4v2 = K at 4 bits + V at 2 bits
  (~3 bits effective, ~108 B/token/head vs int8's 256 B). The V@2-bit plane is the accuracy
  bottleneck — that is what this work order upgrades.
- **k4v4 is ALREADY SUPPORTED by the attention kernels**: src/ops/launcher/gqa_attention_kvarn.cu
  routes k4v4 (and k5v4) through the slice6 decode prologue (`cache.kvarn_k_bits` /
  `cache.kvarn_v_bits` are read from the cache config — find where they are set: model
  config or env; that is your dial).
- Format spec (docs/54_kvarn_implementation.md §Storage): K = 4-bit codes [D,G/2] uint8 +
  per-channel scale/zp fp16 + per-token sinkhorn scale fp16; V mirrors at the preset's bits.
  Dequant `K̂ = (q − zp)·s_col·s_row` folds into the fused kernel (no staging pass).
- Cost table (per token per KV head, D=G=128): k4v2 ≈ 108 B | **k4v4 ≈ 140 B** (V codes
  32->64 B; ~4.4 bits/value) | int8 256 B | bf16 512 B. Total pool at 200k ctx:
  k4v2 ≈ 3.6 GB -> k4v4 ≈ 4.7 GB (all ranks; per-rank ~1.2 GB) — fits VRAM easily.
- vs "regular q4" (q4g64-class, ~4.25 bits/value incl. scales): **k4v4 is a wash on bytes
  (~+3-5%)** — the "cheaper than q8/bf16" claim is true, and vs plain q4 the KVarN scale
  structure (per-channel × per-token) is built for attention's dot-product; plus the kernel
  path already exists. That is why k4v4 is the right quality upgrade — not a from-scratch
  q4 port.
- Live ceiling today: serve config caps at max_ctx=36352 (seen in finish-trace). The 200k
  milestone is: k4v4 + ceiling raise + validated quality/perf.

## PHASE 1 — enable k4v4 + raise the ceiling (config + sizing, no kernel edits expected)

1. Find the dial: where do `cache.kvarn_k_bits` / `cache.kvarn_v_bits` get set (grep the
   config/env read sites)? If an env/config already exposes them, the upgrade may be a boot
   flag. If the pool allocator hardcodes 2-bit V sizing, find that sizing path
   (PagedKVPool alloc, kvarn_workspace) and make it follow the bits.
2. Raise the context ceiling: find the max_ctx=36352 source (arg default? config?). Set a
   200k configuration; size the pool (4.7 GB total KV) inside the VRAM budget (weights ~15 GB
   + KV 4.7 + workspace — check /health VRAM% after boot; the allocator is the gate, no
   estimated refusals anywhere).
3. Env/config arm: NINFER_KV_K4V4=1 (or the native config field if one exists — prefer the
   native one; if you add an env, unset = current k4v2 byte-identical). Log the active
   format at boot ("[KV] kvarn k=4 v=4 pages=...") so legs are self-evidencing.

## PHASE 2 — correctness gates (all must pass before perf claims)

1. Short-context parity: 2k probe vs k4v2 boot — outputs will NOT be token-identical
   (different quant), so the gate is behavioral: BLUE/stop/no-mojibake + no completion-length
   collapse, across 5 diverse probes (essay/story/counting/code).
2. Long-context retrieval: needle-in-haystack at 50k and 150k (needle = a unique code phrase
   embedded at 25%/50%/75% depth; ask for it verbatim). PASS = exact retrieval at all depths.
   Bank the transcripts. Compare k4v2 vs k4v4 on the same needles — k4v4 should be >= k4v2;
   if k4v4 < k4v2 anywhere, STOP and report (that would falsify the quality premise).
3. MTP acceptance gate (docs/54 §6): acceptance at mt=600 must not drop >4 pp vs the k4v2
   boot's same-context measurement (k4v2 baseline: 0.94 cold at mt=600, PLOG-066 class).
4. 206-round-style soak at mt=600: no drift, no mojibake, acceptance stable.

## PHASE 3 — perf characterization (ordinal-paired, PLOG-060 design, clocks noted)

1. Prefill: fresh-boot pairs (k4v2 vs k4v4), 3x 10k probes per arm. Expected: k4v4 slightly
   slower (KV reads +30% during attention) — gate: within 3% = acceptable (the quality is
   the product). If >3% slower, report — do not hide it.
2. Decode: mt=600 legs per arm (counting prompt), read tok/round + acceptance + decode t/s
   from [tp2] lines. Expected: near-wash at short ctx; the long-ctx decode number is
   Phase 3b (100k-context decode t/s) — measure if the ceiling raise works.
3. VRAM: record /health VRAM% both arms at 10k and at the largest context booted.
4. Bank: results/amd/coherence/W7_k4v4_row.txt + logs. PRE-REGISTERED PROMOTION GATE:
   quality gates all PASS + prefill within 3% + decode within noise => flip the default
   (k4v4 becomes the serving format; k4v2 stays as rollback dial) + BOOT_BATTERY + PLOG row
   (chain head 3b049e35e145a35d) + runbook update. Any gate fails => bank the row, leave
   k4v2 default, report.

## PHASE 4 (OPTIONAL, separate gate — only if Phase 2 shows k4v4 quality still short)

NVFP4-format KV (e2m1 codes + e4m3 scales per 16, same as the weight format): new encode +
dequant-in-kernel work (~a real desk). Do NOT start without coordinator go after Phase 3.

## LAWS

- Infrastructure (boot faults, windows, restores) = coordinator's: tag "BLOCKER: <line>"
  in this file and keep working; use `bash /home/chris/serve_fast.sh` for leg boots
  (40 s, NVMe model). If a boot faults at GQA width=53 warmup: that incident is CLOSED
  post-reboot — if you see it again, tag BLOCKER immediately (it would mean the fault
  survives reboots = new information).
- No estimated VRAM refusals — the allocator is the gate, cleanly, at runtime.
- Fresh boot per arm, position-matched probes, ±2% within-pair, clocks note with every
  number (PLOG-060/064 laws).
- Max 2 agents total in flight (principal order) — you are one of them; do not spawn.

## PROGRESS LOG

### 2026-09-20 04:0x local — WINDOW CLAIMED (coordinator grant 03:58); mission-literal receipts BANKED: mc113664+ws512 REFUSED live BOTH tiers (k4v4 8577/8160 slack -417; k4v2 8341/8160 slack -181 — allocator the gate, clean exit in ~5 s, receipts results/amd/k4v4fin/boot_refused_*_receipt.log); ARM A k4v4 @65536 BOOTED GREEN (preflight 8065/8160 slack 94, BLUE PASS) — anchor prefill RUNNING at 82-91 tok/s, measured prompt 49,276 tok = 0.01% off the banked 63,150→49,269 calibration
- Coordinator notes applied: window grant + retire of promoted canonical (cc122cc0) via law form; needle max_tokens 64→512 (>=128 law; gates-desk-proven budget); bin stays 8ac1ba93 per coordinator instruction.
- Ceiling fact now MEASURED, not just derived: ws512 preflight fixed = 7363 MiB (5131 weights + 1360 decoder_fixed + 512 ws + 200 staging + 160 arena_padding), so mc113664 is arithmetically unreachable at ws512 on BOTH tiers; fallback paired posture = mc/cap 65536 (arena O(position) ceiling 65,536 tok at 8 KiB/tok = 512 KiB/page measured-confirmed in the 96-MiB-era death at 160 pages). 113k legs park behind docs/120 B2 direct route; battery cell KV-B3 registered-parked.

### 2026-09-20 03:1x local — WINDOW CLAIM (k4v4fin desk, agent #1): G-Q2 long-needle completion at the corrected paired point + the pre-registered promotion decision. Bin 8ac1ba93eb7fbfad (banked ovf-fix bin), canonical env + NINFER_WORKSPACE_MIB=512. pgrep at claim: coordinator G-MM-2 pair (bin cc122cc0) OWNS the window — desk HOLDS all boots until pgrep empty. Plan: mission-literal boots first (kvarn_k4v4/kvarn_k4v2 @ --max-context 113664 --kv-capacity 113664, ws512) for live receipts; source-derived arena fact (k_temp+v_temp = 256x64x8x2B x2 = 512 KiB/page = 8 KiB/token of position, gqa_attention_kvarn.cu:355-358 + kKvarnAttnD/G = 256/64) caps prefill POSITION at 65,536 tok under ws512 — binding BEFORE the 113,472 pool ceiling — so the fallback fair paired posture is cap 65536 both tiers with gen 76800 needles (~59.9k real by the 1.2816 gen/real calibration) + gen 63150 anchor (~49.3k real). Legs per arm: anchor + d0.25/0.5/0.75, exact-substring grading on content+reasoning_content, max_tokens 64, "model":"qwen3.8-27b". Results -> results/amd/k4v4fin/ on THIS branch; then the WO's pre-registered promotion rule executes (all gates PASS + prefill within 3% + decode within noise => flip default; any fail => bank + report).

### 2026-09-19 19:4xZ local — COORDINATOR CEILING CORRECTION ACKNOWLEDGED (19ebc646d): adapted G-Q2 long legs = 113664-tok needles on BOTH tiers (fair paired point) + 50k mid anchor; ALL still queued behind BLOCKER #1 (the ~10.1k bad_alloc wall — no kvarn boot can prefill past it on f3312f25, so 113664/50k are unreachable until kvarnport kills the wall). Re-run recipe ready: this branch's tools replay verbatim; bodies regenerate at --ctx scaled x~7.95 gen-tok per real tok (calibration 63150 -> 49269 real; 9473-real via ctx 12000); k4v2 long boot may need --max-context 141312 explicit.
- Refusal lines from BOOT3/4 recorded as the ceiling receipt per coordinator instruction (row W7_k4v4_phase2_row.txt VRAM RECORDS section).

### 2026-09-19 19:3xZ local — BLOCKER #2 (window close): canonical restore boot of 2c8901d3 FAULTED at GQA width=53 warmup — PLOG-066 INCIDENT 1 class RECURRING; coordinator-run playbook required (no passwordless sudo on this box); canonical serve NOT restored
- FACTS: bash /home/chris/serve_fast.sh (bin 2c8901d3d18adef1) died at warmup: "Memory access fault by GPU node-2 ... Reason: Page not present or supervisor privilege" after the [GQA] splitk width=53 lines; no serve left running. kern.log: amdgpu 0000:08:00.0 [gfxhub0] no-retry page fault (ring:24 vmid:8 pasid:1473) at 19:30:47 local — same class as the morning 08:13/08:21 entries of PLOG-066 INCIDENT 1 (which was "closed post-reboot" — recurrence = NEW INFORMATION per this WO's law).
- Box state at fault: temps NORMAL (42-56 C edge — NOT the stuck-hot wedge class), 4 dies idle, no serve running. Standing rule honored: kern.log captured FIRST (results/amd/k4v4gates/kernlog_pre_flr_191019.log), NO retry through the fault.
- Desk has NO passwordless sudo -> sysfs FLR (/sys/class/drm/cardN/device/reset) + rocm-smi --gpureset + full reboot are COORDINATOR-RUN per playbook v3 (docs/amd/PREFILL_1K_PLAN_2026-09-18.md:51). Retire law pkill used only, own boots only.
- NET WINDOW STATE: all Phase 2/3 measurements BANKED and committed before the fault (commits b90b668a1..0a7d3ce4e on amd/wo-k4v4gates); the restore debt is the only open item. Next serve boot on this box (ANY bin) should follow playbook v3 after coordinator FLR/reboot.

### 2026-09-19 00:1xZ (2026-09-20Z) — PHASE 2/3 EXECUTED TO VERDICT: G-Q1/G-Q3/G-Q4 PASS, G-Q2 PASS<=9.5k + BLOCKED>10.1k; perf: k4v4 no-regression (prefill faster every ordinal, decode -1.2%); PROMOTION NOT MET (G-Q2 long legs unrun — BLOCKER, not a quality fail); default unchanged; row banked
- BOOT2 kvarn_k4v2 baseline arm (fresh boot, same bodies, same order): auto cap 141056 tok (kv 8976 B/t/rank; k4v4/k4v2 = 1.2425). G-Q1 5/5 stop/no-mojibake. 3k needles 3/3 exact. mt=600: acceptance 0.88 tok/round 2.75. Soak 10/10 clean, acceptance 0.88 FLAT, zero mojibake, edge 83-84 C.
- GATE TABLE: G-Q1 PASS (5/5 both arms) | G-Q2 PASS<=9.5k (k4v4 3/3@3k + exact@9473; k4v2 3/3@3k) + BLOCKED>~10.1k (std::bad_alloc wall, BLOCKER entry above) | G-Q3 PASS (0.86 vs 0.88 = -2pp < 4pp gate) | G-Q4 PASS (10/10 stable both arms). STOP-condition "k4v4 < k4v2 anywhere" did NOT fire.
- PHASE 3: prefill pairs (identical bodies+order, fresh boots): k4v4 81.84/111.04/157.64 s vs k4v2 127.58/166.82/181.92 s -> k4v4 FASTER -35.8/-33.4/-13.3% wall (96.9/71.3/50.2 vs 62.2/47.5/43.6 tok/s); the ">3% slower" trigger never fired. CONFOUND banked: k4v2 arm hotter (84 C) with mclk mostly lvl0-167 vs k4v4's lvl1-500 (sideband files) — direction safe, magnitude needs a cold cold-pair re-run before quoting v4>v2 as a kernel fact. Decode: 13.83 vs 14.00 tok/s = -1.2% (within +-2% noise); tok/round 2.73 vs 2.75. VRAM: k4v4@151552 REFUSED live (8564/8160, slack -404); k4v2@151552 REFUSED live (8249/8160, slack -89); ws=96 maxima ~113k (k4v4) / ~141k (k4v2); 151552 needs ~456 MiB of posture levers (staging 200 + pad 160 + ws 96). Phase 3b blocked by the wall.
- PROMOTION LAW: NOT MET — a gate that cannot run is not a pass. Nothing flipped (production default on THIS line remains bf16-KV; k4v2 the rollback dial). All short-ctx k4v4 evidence GREEN. Re-run path after kvarnport kills the wall: this branch's tools + bodies replay verbatim.
- ROW: results/amd/k4v4gates/W7_k4v4_phase2_row.txt (full tables + evidence index). Window close: canonical restore + health + fast probe next entry.

### 2026-09-19 19:05Z — BLOCKER TAGGED (new class): kvarn long-prefill host std::bad_alloc at ~10.0-10.2k real tokens — 50k/150k needle legs IMPOSSIBLE on bin f3312f25; wall measured + bracketed; short-ctx gates unaffected (continuing per keep-working law)
- REPRO (deterministic, 4/4): kvarn_k4v4 boot + any prompt >~10.1k real tok -> ALL-4-rank "[tp2 worker error rank N] std::bad_alloc" mid-prefill, at 9984-10112 progress marks (49269-tok prompt x3: died ~3.9 min in at ~10.1k; 19709-tok prompt: died at 9984/19709 = 51%). Server returns {"error":"std::bad_alloc"} (internal_error).
- NOT memory pressure (falsified): host RSS FLAT 2978->2995 MB across the whole dying prefill; 55-56 GB available (RSS-traced); VRAM pool fine (49269 tok x 11152 B/t = 527 MiB of a 1206 MiB pool). Pattern = position-locked allocation failure (overflow/limit class), same position across prompt sizes 19709 and 49269.
- BRACKET: PASS 9473 tok (needle d0.5 exact: ORCHID-TUNNEL, finish=stop, prefill 66.9 tok/s) and 7930 tok (10k probes); DIE 9984..10240. Wall = (9473, ~10.2k] real tokens. 3k/6k-class needles unaffected (port battery + this desk's 3k 3/3 PASS).
- HISTORICAL ANCHOR: no kvarn-tier boot has ever prefilled >~8k real tok (port battery max = needle@6k); bf16 routinely did 36352 — wall is kvarn-route-specific. Owner: kvarnport desk (device-port lane). Repro line: boot f3312f25 + --kv-dtype kvarn_k4v4, POST ~11k+-tok prompt.
- IMPACT: WO Phase 2 gate 2 (50k/150k needles) + Phase 3b (100k decode) UNREACHABLE on this bin; the 113472 auto ceiling is not exercisable past ~10.1k. Everything short-ctx proceeds: G-Q1, 3k needles, G-Q3, G-Q4, prefill pairs (7.9k), BOOT2 k4v2 baseline arm NOW.
- Evidence: results/amd/k4v4gates/{resp_needle20k.json, resp_needle9500.json, rss_trace_20k.txt, serve_boot1_k4v4_short.log (req 32-36)}.

### 2026-09-19 23:5xZ — BOOT1 k4v4-short EXECUTED (bin f3312f25c8201da0, --kv-dtype kvarn_k4v4, canonical env): G-Q1 5/5 GREEN; AMBER RESIDUE GONE (needle ~3k real 3/3 PASS exact, finish=stop); G-Q4 soak k4v4-side 10/10 clean stable; auto-KV ceiling for k4v4 = 113472 tok (KV 11152 B/t/rank, required 8159/8160 MiB)
- BOOT FACTS: auto KV sized capacity=max-context=113472 (vs bf16 36352) — the funder buys a 3.1x auto ceiling; preflight fixed 6947 + kv 1206 + 5 = 8159 MiB, slack 0. /health carries NO VRAM field in this bin -> preflight lines are the VRAM record (noted).
- TOOL FIXES (3, committed on amd/wo-k4v4gates, RED->GREEN each): (T1) request bodies need "model" field (RED: server 'missing required field: model' on every staged-tool probe); (T2) reasoning model — needle max_tokens 24 -> 512 + probe budgets (RED: 24 tok all spent in reasoning_content, content='' finish=length — the port's AMBER signature reproduced as a TOOL artifact, not a serving defect; green bin at 512 tok answers finish=stop); (T3) NEEDLE_CTXS='' skip + -d @file posting (RED: 'Argument list too long' at 400KB bodies vs 128K MAX_ARG_STRLEN).
- G-Q1 (k4v4 side): 5/5 finish=stop, coherent, no mojibake (essay 670 / story 545 / count 239 / code 157 / stop-BLUE 211 chars). k4v2 side-B on BOOT2.
- PREFILL ARM k4v4 (3x 10k-gen probes, 7915-7930 real tok, fresh boot, ordinal 1-3): 81.84 s / 111.04 s / 157.64 s = 96.9 / 71.3 / 50.2 tok/s — monotonic thermal/mclk drift; WITHIN-ORDINAL pairing vs BOOT2 absorbs (PLOG-065 law). Clocks sideband: sclk lvl3 991 MHz, mclk lvl1 500 MHz during load (mclk NOT at 945 top — flagged per clock law), edge 56 C start.
- G-Q3 k4v4 side: mt=600 counting leg acceptance=0.86 tok/round=2.73 rounds=220 gen=600/600 (prompt-dependent absolutes; GATE = k4v2 BOOT2 same-prompt not drop >4pp).
- G-Q4 k4v4 side: 10-leg soak ALL 600/600, acceptance 0.86 FLAT, tok/round 2.73 flat, ZERO mojibake, no drift; decode 31.2 -> 43.4 s (thermal ramp then 43.0+/-0.5 plateau), edge 80->84 C. STABLE.
- AMBER (port desk question) CLOSED: needle at ~3.0k real tok (prompt=3041/3056) 3/3 exact retrieval (ORCHID-TUNNEL/MARBLE-COMPASS/CINDER-HARBOR), finish=stop gen=271-272 — the finish=length+empty class does NOT reproduce on the green bin with 512-token budget.
- 50k needles (real ~50k, --ctx 63150) RUNNING on this boot (max_ctx 113472 covers). Transcripts: results/amd/k4v4gates/ (GQ1_k4v4_boot1.txt, needle3k_k4v4_boot1.txt, soak_k4v4_leg*.json, resp_prefill10k_*.json, serve_boot1_k4v4_short.log).

### 2026-09-19 23:15Z — WINDOW CLAIM (k4v4gates desk, agent #2): BOOT1 k4v4-short — Phase 2 gates (G-Q1 behavioral 5-probe, 3k needle AMBER sanity, G-Q3 MTP mt=600, G-Q4 soak) + Phase 3 k4v4 prefill arm (3x 10k). Bin f3312f25c8201da0, canonical env, --kv-dtype kvarn_k4v4. pgrep at claim: canonical 2c8901d3 on :8100 (retireable per retire law). Boots planned: BOOT2 k4v2-short baseline arm; BOOT3 k4v4-long (50k+150k needles, allocator is the gate); BOOT4 k4v2-long. Results -> results/amd/k4v4gates/ on this branch.

### 2026-09-19 ~20:05Z — Phase 1 EXECUTED TO THE BLOCKER; enablement is NOT config-level on HIP (BLOCKER tagged; coordinator decision required)

**BLOCKER: k4v4 (any KVarN tier) cannot boot on this HIP lane — two linked capability facts, not capacity: (B1) G-AMD-31 parse gate (serve_options.cpp `#if defined(NINFER_HIP_ROSTER)`, commit f23197980) refuses `--kv-dtype kvarn_*` at parse, naming the roster line; (B2) the fused kvarn attention TU `gqa_attention_kvarn.cu` is a deliberate HIP-roster remainder (HipSources.cmake:215-218; stub arms perf_stub_die at hip_link_stub_arms_perf.cpp:179/:182/:185). Deleting B1 without the B2 port = the agent2 k5v4 death-row class (mid-request stub death, request 1). Owner per the roster law: T3/A3 device-port lane; port surface is NARROWER than WO-06 era — gqa_attention_prefill.cu + gqa_attention_decode.cu already joined the roster (A3-wave, HipSources.cmake:304-305); the remainder is the kvarn launcher TU + its kernel headers (slice4/slice6 .cuh, kvarn_mma.cuh, packed .inc).**

Executed (evidence banked, `results/amd/coherence/W7_k4v4_row.txt` + 2 transcripts):

1. **Dial found, fully wired** — `--kv-dtype kvarn_k4v4` (accepted-name table types.h:89-103); widths registered incl. (4,4) (types.h:124-134, docs/117 Step D); tier table types.h:63-70; budget width-generic tp2_budget.h:138 (`kvarn_kv_bytes_per_token(k,v)`) — no hardcoded 2-bit V sizing anywhere (pool sizing kvarn_workspace.cpp:459/532 follows bits); engine dispatch routes k4v4 via slice6 KSide=4 single-seq AND MultiBatch (gqa_attention_kvarn.cu:130-166); batched gates tier-generic (tp_engine.cpp:190-196). Phase 1's "find the dial" step: DONE — the dial exists; the ROUTE does not.
2. **TEST A (RED row)**: shipped canonical bin 2c8901d3d18adef1 + `--kv-dtype kvarn_k4v4` → exit 1 at parse (<1 s, pre-model-load) with the exact G-AMD-31 text ("not servable on the HIP lane … named roster remainder … CAPABILITY, not capacity"). Transcript: `W7_k4v4_testA_parse_refusal.log`.
3. **WO premise corrected by measurement**: production KV on THIS box is **BF16**, not k4v2 — the serving line passes no `--kv-dtype` (default BFloat16, serve_options.h:48); boot preflight prints kv(34816 B/t x 36352 tok) = BF16 class (serve_fast.log 2026-09-19 13:08). The `[rank 0] kvarn reset inflight` lines are unconditional bookkeeping (tp2_backend.cpp:2094/2343), NOT tier evidence. k4v2-as-production holds on the CUDA/main line only.
4. **Ceiling decoded**: max_ctx=36352 is the auto-KV sizing result (8160 MiB usable/rank, ws 96 MiB), not a hardcoded cap; native context 262144 (tp_engine.cpp:1079). **TEST B (measured refusal, legal live-gate)**: bf16 + `--max-context 200704 --kv-capacity 200704` → required 14544 MiB vs usable 8160 MiB per rank (context 6669, fixed 7875, slack −6384). Transcript: `W7_k4v4_testB_200k_ceiling_bf16.log`. The 200k ceiling is INSEPARABLE from the KV tier. Arithmetic on measured anchors (34816 B/t ÷ 512 = 68 K+V head-slots/rank ⇒ 34 KV heads; 34 × ~140 B = 4760 B/t): k4v4@200704 ≈ 912 MiB/rank → fits ONLY with workspace auto (6947+912+5 ≈ 7864 < 8160), not the explicit-capacity ws=1024 posture. PREDICTION ONLY — allocator is the gate at boot.
5. Phases 2/3 (parity, needle 50k/150k, MTP gate, perf pairs): impossible until a kvarn tier boots on HIP — no gates run, no quality claims. Promotion: nothing flipped; default line untouched.
6. No `cmake --build` performed (no code edits needed for the findings) → no BUILD CLAIM filed; fp8-on-ring desk's build-tree coordination untouched.
7. Close hygiene: canonical restored via `bash /home/chris/serve_fast.sh` — health `{"status":"ok"}`, FAST PROBE PASS (wall 18.4 s, finish=stop), bin 2c8901d3d18adef1 confirmed on :8100. Retires used EXACT law form only. Clocks at tests: mclk lvl3 945 MHz, sclk idle 300 (boot-behavior evidence only; no perf numbers).

**Decision needed from coordinator (pick one):** (a) dispatch the T3/A3 device port of `gqa_attention_kvarn.cu` (+ slice .cuh family) — this desk then resumes Phase 1→3 exactly as written (dial + explicit ceiling flags + ws-auto note above); (b) re-scope this WO to the CUDA/main line where the route exists (not this host); (c) park. Desk holds at this checkpoint per ping-pong law — QUEUE NOT EMPTY: ready to execute (a)'s verification gates the moment a kvarn-capable bin exists, or to take the next TIER 1 item on coordinator word.

### (prior sessions — none; this file had no log entries before 2026-09-19. Work order received ~19:40Z; recon 19:40-19:58Z: dial/ceiling/roster traced, parallel-desk docs read, no build claim needed.)

### COORDINATOR DECISION (2026-09-19 ~16:3x): (a) — port dispatched and EXECUTING
- amd/wo-kvarnport desk landed the port (roster join + 3 stub arms retired + G-AMD-31 gate deleted, bin banked nm-verified). Leg battery RUNNING: A1 kvarn_k4v4 and EXP1 kvarn_k4v2 both RED (boots, finish=stop, EMPTY outputs — shared-seam class, desk actively debugging in gqa_attention_kvarn_mma.cuh). A read-only falsifier desk (WO_KVARN_RED_AUDIT.md, agent #2) is racing the root cause in parallel — its ranked hypotheses will land in that WO's log.
- THIS desk's resume path is UNCHANGED: the moment a kvarn bin produces GREEN probes, Phase 2/3 gates run with the banked tools (needle generator + parity/needle battery: tools/v340l/w7_needle_gen.py + w7_k4v4_gates.sh on amd/wo-w7-body, commit-series 21ffda04f..ac9ec1f49). Executor: coordinator inline or next freed agent, per window law.

### COORDINATOR CEILING CORRECTION (2026-09-19 ~19:2x) — the 150k legs cannot fit; honest ceilings are ~114k (k4v4) / ~141k (k4v2)
- Your BOOT3/BOOT4 refusals are CORRECT physics, not a bug: the WO's original ceiling premise ("k4v4@200704 ≈ 912 MiB/rank → fits") used a 4760 B/t/rank estimate that is 2.34x optimistic. The MEASURED kvarn unit is 11,152 B/t/rank (budget constant kvarn_kv_bytes_per_token(4,4) = 18496x20992/34816 = 11,152; port measurement 386 MiB/rank at 36352 = 11,134 — 0.16% agreement). Live: k4v4@151552 required 8564 / usable 8160 (slack -404); k4v2@151552 required 8249 / 8160 (-89).
- HONEST CEILINGS at today's fixed footprint (6947 MiB incl. ws 96, live-usable 8160): k4v4 ~114,050 tokens; k4v2 ~141,700 tokens. Native 262144 is not reachable on this box at any kvarn tier without shrinking the fixed side (decoder_fixed 1360 / staging 200 / padding 160) or hosted-KV Phase 2 (parked by principal order).
- GATE ADAPTATION (pre-registered-gate compliant — the gate is long-context RETRIEVAL, not a token count): run the needle gates at the largest context that boots per tier — k4v4 at 114,048 (113664 to be safe: 64-multiple, ~1244 MiB, ~+80 MiB slack) and k4v2 at 141,312 — with 50k as the mid anchor. k4v4-vs-k4v2 quality comparison stays valid AT EQUAL context: run the 50k needles on BOTH tiers, and the ~114k needles on BOTH (k4v2@114k has 27k tokens of headroom, so 113664 works for both — that is the fair paired point). Record the refusal lines from BOOT3/4 as the ceiling receipt.
- The budget constant itself is VERIFIED against measurement (0.16%) — do not "fix" it.

### 2026-09-20 05:4x local — COORDINATOR INHERITANCE (desk died platform-class mid-arm; battery kept RUNNING — driver2/run_arm2 are kill-proof by design); arm A receipts banked (95a73a3f4); promotion-decision checklist pre-staged
- INHERITANCE FACTS: driver2.sh -> run_arm2.sh survived the agent death (setsid); arm A executed on unattended: anchor PASS (49,276 prompt, ORCHID-TUNNEL exact, wall 1435 s), d0.25 PASS (59,935 prompt, exact, wall 2095 s). d0.5/d0.75 in flight at write time; arm B k4v2 auto-follows; RESTORE=1 auto-restores canonical at arm B close.
- PRE-STAGED DECISION (executes ONLY when arm B grades land): pre-registered rule = G-Q1/G-Q3/G-Q4 already PASS (gates desk, banked); G-Q2 completes iff all three 76800-gen needles PASS on BOTH tiers at the paired 65536 posture AND the STOP condition "k4v4 < k4v2 anywhere" does not fire. Then: flip default per WO Phase 3.4 => serve_10k.sh gains `NINFER_WORKSPACE_MIB=512 --kv-dtype kvarn_k4v4 --max-context 65536 --kv-capacity 65536` (ws512 is MANDATORY in the flip — the ws96 arena bad_allocs prefill past ~12.3k real tok; measured preflight at this posture 8065/8160 slack 94); serve_fast inherits; rollback = `--kv-dtype kvarn_k4v2` (or unset for bf16); runbook §0 KV-tier note update; BOOT_BATTERY cells KV-B0/B1/B2 join the promoted default's standing battery; PLOG row. Any FAIL => bank + report, default untouched.
- ANCHOR-LOSS FALSE ALARM retracted: resp_kvarn_k4v4_anchor63150.json is valid (519 B, mtime 04:56) — grade line PASS was already banked; earlier 0-byte listing was a stale read.
- Ordinal note for arm B pairing: arm A anchor prefilled at 23-35 tok/s (boot-cold mclk) vs d0.25 at ~40 tok/s (warmed) — SAME-boot drift, absorbed by the ordinal-paired design (arm B runs the identical leg order on a fresh boot).

### 2026-09-20 09:1x local — PROMOTION EXECUTED (coordinator): battery 4/4 + 4/4 exact, all gates GREEN, default FLIPPED to kvarn_k4v4@65536/ws512 on NEW UNION BIN 058a7b85859c5cd0 — with one honest lineage incident banked
- VERDICT: arm A k4v4 4/4 PASS (anchor 49,276 + 59,935 d0.25 + 59,830 d0.5 + 59,984 d0.75 — ORCHID-TUNNEL/MARBLE-COMPASS/CINDER-HARBOR all EXACT, finish=stop); arm B k4v2 4/4 PASS identical grading — the STOP condition "k4v4 < k4v2 anywhere" never fired; G-Q1/G-Q3/G-Q4 PASS (banked gates-desk rows); prefill "within 3%" satisfied a fortiori (gates-desk pairs measured k4v4 FASTER every ordinal). PROMOTION MET per the pre-registered rule.
- INCIDENT (caught by the flip's boot-verify, exactly as the runbook demands): the then-canonical bin cc122cc0 (mad-mix lineage, built PRE-ovf-merge) REFUSES kvarn at parse ("cached-attention route is a named roster remainder") — the KV promotion and the GEMM promotion lived on different branch lineages and NO union bin existed. First flip attempt booted bf16-at-ws512 (serve_fast builds its own arg line; only BIN+envs are inherited — now patched in sync). FIX: union build from amd/main tip ece88d382 (tree verified byte-identical to 8ac1ba93e in all roster/link files + carries mad-mix) -> banked ninfer-serve_058a7b85859c5cd0.bin -> both serving scripts flipped (BIN + ws512 export + kvarn flags, rollback comments inline). LIVE-VERIFIED: preflight kv(11152 B/t x 65536) slack 0, FAST PROBE PASS, healthy past the 90 s mark.
- OBSERVATION (cause unnamed, not reproduced): the battery's own 08:58 restore boot exited gracefully ~90 s post-probe (vocab flush(shutdown); SIGTERM-class, NOT the law-form SIGKILL); boots launched from the coordinator session persist. New close rule: canonical restores verify liveness at health-OK AND at +3 min, not health-OK alone.
- CAPACITY LEDGER (the deliverable): serving default moves bf16@36,352 -> kvarn_k4v4@65,536 tokens (+80% context at ~1/3.1 the KV bytes/token; the 113k-class pool ceilings remain parked behind docs/120 B2 with receipts). Rollback dial: --kv-dtype kvarn_k4v2 (V@2-bit) or full bf16 revert per the script comments.
