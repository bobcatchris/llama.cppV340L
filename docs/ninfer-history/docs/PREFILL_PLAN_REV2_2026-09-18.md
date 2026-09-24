# PREFILL PLAN REV2 — the 700-class program (goal change, window 7)

Written 2026-09-18, window 7, after the user's plan-review order. Supersedes the WORKSTREAMS
ordering of PREFILL_1K_PLAN_2026-09-18.md (that doc's LEDGER and RULES remain canonical — this
doc re-orders the attack by measured payoff and records the goal change).

## 0. The user orders that revised this plan (2026-09-18, this session)

1. **Goal change:** "get me as close to 1k as possible, if it's 700 is okay but you really need
   to get these numbers moving" — the acceptance bar moves from the 1k-derived 54 % pk16 peak
   to the 700-class-derived ~43 % (see §2).
2. **Attack order:** "focus on the largest windows — a short ordered list of attempted
   optimizations where the biggest payoffs are at the top."
3. **Plan review:** "review the plan, really think through where we can multiply our results" —
   the review concluded: everything landed so far is ADDITIVE; the only ×3+ multiplier on this
   hardware is the GEMM pk16 family, reopened by the goal change (§2).

## 1. Where the wall is (measured, window 7, dn-promoted bin 1e03e5af5fb438b9)

Steady chunk (128 tok, n=15 steady chunks of the plen-2075 OPTRACE=3 leg, band-uncontrolled):

| window | ms | share | state |
|---|---|---|---|
| **GEMM** (192 nvfp4-dequant launches) | **~770** | **75 %** | walled at fp32 (5 ledger kills) — §2 |
| body (unbr 53, gqa 34, dn 30, gate 12, rest 36) | ~165 | 16 % | dn/gqa/ggate landed this window |
| AR | ~60 | 6 % | floor ~35 spec'd (W2_AR_FLOOR_desk) |
| gap (steady) | ~20 | 2 % | ALREADY at the plan's 15–30 target — the old "gap 122.7/180" was a request-mean artifact of chunk-1 + finalize (W2 desk per-chunk decomposition) |
| finalize chunk (per REQUEST) | 1393 unbracketed | ~82/chunk amortized | anomaly §3 |

**Scoreboard:** 76.4 (W6 close) → **~116–121 tok/s class** (W7 request-mean INCLUDING the
finalize anomaly, band-uncontrolled); steady-chunk arithmetic excluding finalize ≈ 128.
Every W7 number is band-uncontrolled (single legs, droop creep 979→1032 ms across one request
is visible in the OPTRACE rows) — clean-band re-legs owed before any cross-window % claims
(PLOG-043 law).

**The additive ceiling (the review's blunt conclusion):** AR floor + capture + finalize fix
sum to ~-165 ms/chunk ⇒ ~700 ms/chunk ⇒ **~180 tok/s ceiling without touching GEMM.**
No amount of gap/body work multiplies past that. GEMM is 75 % of the wall.

## 2. THE ONE MULTIPLIER — GEMM pk16 (W3 sketch, REOPENED under the new goal)

Ledger honesty (R4): the W3 row was killed against the **1k bar** (≥54 % of pk16 peak
sustained). The kill's WHY is untouched — no tile reaches 54 %. But the census MEASURED
(ISA-level, zero GPU): 102 VGPR, 0 B scratch, native `v_pk_fma_f16` ×128/body — PK's
resourcing death cured — with a measured issue model of **39–49 % of pk16 peak** vs V0's
2.0–2.1 TF/s/die today. 700 tok/s needs 37.8 TF/s aggregate = ~9.45 TF/s/die = **~43 % of
pk16 peak — INSIDE the sketch's band.** The 1k kill stands; this is a NEW acceptance under a
NEW target. Recorded as a reopen row (goal change), not an overwrite.

- **STALL-RISK HONESTY (read before believing 3.5×):** the 39–49 % is the census's INSTRUCTION
  ISSUE MODEL at zero stall — a ceiling, not a floor. The same census measured 44 waits/body vs
  the ≤20 acceptance; if those waits do not hide behind pk16 math, the effective rate erodes —
  possibly all the way back to V0-class. THE BENCH DECIDES; nothing downstream may plan on the
  band until the GF/s table exists.
- **Prize IF the low band holds (3.5–4×):** GEMM 770 → ~190–220 ms ⇒ chunk ~430–500 ⇒
  **~260–300 tok/s**; stacked with §3/§4 and the M=256 ladder: **350–450 honest band**. Top of
  band (4.9×) ⇒ ~500+. If the bench lands <1.5×, the family dies permanently and the program's
  honest end state is the additive ceiling ~180 tok/s.
- **Pre-registered acceptance (bench, not model — R5):** ≥2.0× V0 GF/s mean at M=128 across the
  four W4 geometries ⇒ PASS to integration; 1.5–2.0× ⇒ AMBER (record, decide by integration
  cost); <1.5× ⇒ KILL the family permanently.
- **Kill-switches, in order:** (1) census the numerics-complete kernel FIRST — VGPR >128 or
  scratch >0 = PK's death returning = STOP, no blind tuning; (2) correctness rel-L2 within 2×
  V0's own vs the fp32-dequant reference on identical data; (3) the bench verdict above.
- **Method:** build from tools/v340l/w3_pk16_sketch.cu (its staging/bridge/LUT pattern is
  PK-proven); numerics-complete = real e2m1 LUT + nvfp4 group-scale fp32 fold + the bf16→fp16
  x bridge; bench A/B vs the shipped V0 in ONE standalone binary with the sweep's methodology;
  clocks sideband before/after; M=128 first (serving posture), then M=256.
- **Status:** desk chartered (agent dispatch hit box concurrency limits twice — owner executes
  inline; bench binaries are single-die, short-lived, server untouched).

**VERDICT (window 7, same day): MEASURED KILL.** Mean pk16/V0 = 1.34× (1.28–1.43; V0 validated
±2% vs SWEEP_row). The stall-risk caveat fired: waits do not hide, ~4.0 TF/s/die is the structure's
ceiling. Full receipt: results/amd/coherence/W7_pk16_bench_row.txt. The additive path below is the
program's real road; 300+ tok/s is hardware/compiler-gated.

**COMPILER-ROAD CENSUS (window 7, same day): DEAD.** The ledger's named reopener — "a compiler
that batches memory waits at ≤112 VGPR" (W3 reopen condition 1) — was TESTED, not assumed: the
SAME census (w3_census.py, LEANMAC §1 law flags) run on the SAME TUs under two NEWER toolchains
acquired userspace (dpkg-deb extraction to EMTEC256, host 6.2.0 untouched): ROCm 6.4.4
(LLVM 19) and 7.2.4 (LLVM 22), vs the 6.2.0 anchor (verified reproducing the banked receipts
exactly). Result: the sketch A16 point emits the IDENTICAL 44-wait body (35 lgkm + 9 vm) at
102/102/104 VGPR on ALL THREE — the wait structure is compiler-invariant across five releases,
so the ≤30/body pre-registered bar fails everywhere — and shipped V0 REGRESSES under both new
compilers (256 VGPR + 388–472 B spill, 2× static body). gfx900 codegen remains present in LLVM
19/22 (MI25 left official support at ROCm 5.0 — moot here). Bench phase NOT reached: kill-switch
fired at census, zero GPU seconds. Receipt:
results/amd/coherence/W7_rocm_census_row.txt. The GEMM front's compiler branch is CLOSED BY
MEASUREMENT; the family's only remaining reopener is hardware class (gfx906+ dot10).


**§2.5 — VRAM-FUNDED PRE-DEQUANT: PARKED BY USER ORDER (2026-09-18). DO NOT DISCUSS, PLAN, OR CITE.**
The idea (materialize NVFP4 weights as bf16 in VRAM to delete the GEMM decode-instruction stream)
is EXPLICITLY parked until BOTH: (1) prefill is maxed out on the current software path, AND (2) the
q4 KV cache migration has landed. The operator's reason is precedent, not preference: on the NVIDIA
line this exact lever ("donating VRAM") consumed multiple weeks, and its mere presence in the plan
caused every agent session to treat it as NECESSARY for good performance — blinding the whole team
to other optimization barriers. It was removed, and the NVIDIA side ended with MORE prefill AND
more decode than it ever had with the donated VRAM. That agent-reasoning trap is the thing this
parking notice exists to prevent: do not re-derive this lever as "the path", do not price it into
projections, do not stage censuses for it. The in-scope prefill program is: additive stack
(finalize/AR/capture) + pinned clocks + iommu=pt + M=256 re-leg. Any session that finds itself
arguing for pre-dequant is re-reading this notice.

## 3.## 3. #2 payoff — the finalize-chunk anomaly (VERDICT 2026-09-18, named + cured; full ledger results/amd/coherence/W7_finbracket_row_v2.txt)

**VERDICT: HOST-side stall in the MTP tail path, cured by `NINFER_MTP_TAIL_ASYNC=1` (device-side
tail composition). NOT GPU compute; NOT the recurrent delta-net (dn 7-8 ms at tok=27); NOT the
MTP finalize/AR path (extent==0 — never runs in serve).**

- Bracket chain (each step a banked leg, parity 'BLUE'/stop/52 throughout): v3 event brackets
  put the WHOLE gap in `prep` (mark0→mark1: 922-1602 ms vs gap 925-1605); removing the
  sync+D2H (v1) did NOT move it; FINHOST host stamps put it in the **pageable H2D of mtp_ids**
  (659.6 ms on all four ranks, ±0.4 ms — a driver staging size class new to the process at
  108 B) plus the seeding host section (~330 ms quanta, process-serialized).
- Cure: compose mtp_ids ENTIRELY on device — prompt columns are the chunk's own ids shifted by
  one (D2D from the already-resident ids_device), final column is io_.token (D2D, same-stream
  after the is_last argmax; allreduce_argmax leaves the same champion on every rank). Guard:
  `shifted_embedding_begin == text_prefill->begin + 1 && multimodal == nullptr`. Unset =
  byte-identical legacy path.
- Before/after (same band, same probe, FINAL platform pinned+pt): chunk-17 gap
  925-1605 → **16.6-17.3 ms on 3/4 ranks** (steady-chunk level; straggler 320-336 ms is a
  ONE-TIME-per-process first-T=27 `mtp_forward_stem` host quantum — width-class-specific, not
  warm-up: T=2/51/128 stems are 0.015-1.2 ms from their first call; proposal: warm a ~27-token
  prefill at boot). Chunk-17 wall max 2266 → ~1360 ms. Request mean unbracketed gap/chunk
  126-167 → **73-92 ms** (§5 projected −82 amortized; measured −53..−75).
- Standing cell: [PREFILL-FIN]/[PREFILL-FINHOST]/[PREFILL-FINMTP] env-gated prints — prep
  regression re-appears per request; zero cost unset. Promoted bin 0572d4108317ee85
  (BOOT_BATTERY GREEN 14:57, serve_10k.sh BIN+env flipped, serving).

## 4. #3/#4 payoffs (additive, in order)

- **AR floor:** 60 → ~35 (W2_AR_FLOOR_desk one-window plan: B-1 RCCL sweep + F-ENV arms).
- **Capture medicine** (unbr 53 + steady gap 20): chunk-graph capture design spec'd in
  W2_GAP_CAPTURE_desk (gate `NINFER_PREFILL_GRAPH=1`, default OFF byte-identical; capture is
  de-facto proven feasible — decode graphs already hold ncclAllReduce at world=4). Acceptance:
  steady-chunk wall −40…−90 ms, byte-identical greedy text, kill if <20 ms.
- **W5 thermal + M=256 re-leg AFTER GEMM:** steady wall rises 979→1032 within one request
  (droop); the M=256 ladder measured +15 % with GEMM FLAT (receipt:
  results/amd/coherence/W7_row_ladder_chunk256.txt) — the faster the GEMM, the more the
  ladder's fixed-cost amortization matters. The finalize fix also shrinks per-request cost,
  which the ladder amortizes further (9 chunks vs 17).

## 5. Scoreboard projection (honest, band-uncontrolled classes)

| step | steady chunk ms | tok/s class |
|---|---|---|
| W7 landed (dn×16 + gga + gqa) | ~1000 + finalize | 116–121 |
| + finalize fix (§3) | −82 amortized | ~127 |
| + AR floor + capture (§4) | −85 | ~140 |
| + GEMM pk16 low band ×3.5–4 (§2) | gemm 770→190–220 | **~260–300** |
| + M=256 + thermal control | — | **~350–450** |
| 1k (54 % pk16 peak) | — | **KILLED (W3 row)** — needs gfx906 dot10 hardware or compiler wait-emission reopeners |

## 6. Rules unchanged

R1–R7 (PREFILL_1K_PLAN), RED/GREEN closure, VRAM law, kill-switch-first, incremental banking,
PLOG chain via plog_append.py. The W3 reopen is a NEW row citing the goal change; the original
kill row is preserved verbatim above it.

## §6 EXTERNAL-FINDINGS PROGRAM (search desk 2026-09-18, user order: parallel workers, 200 target)

Web-search desk returned live GCN5-family optimization work. All levers below are UNPARKED (VRAM-neutral unless noted), ranked, with dispatch state:

| # | lever | evidence | mechanism | state |
|---|---|---|---|---|
| M1 | **v_mad_mix_f32** (VOP3P mixed-precision FMA: f16×f16+f32 in 1 VALU op, gfx900-native, inline-asm, ROCm 6.2 OK) | llama.cpp patch on GCN5: decode +24%@4K KV, +157%@32K KV; FA spills 10,598→6; MORE accurate | deletes the operand-conversion share of V0's measured 52% issue tax; fp32 accumulate = no pk16 flush tax | **DEAD — MEASURED KILL 0.48× V0** (window 7): mad-mix works (probe 65023 = fp32 product; numerics bit-exact vs REF) but the tile resourcing dies: 2048-op opaque asm body → 256 VGPR + 268-376 B spill + occupancy 1. **MECHANISM INSIGHT: V0's 52% 'conversion' stream is the scheduler's latency-hiding material — deleting it exposes LUT/load latency. The GEMM wall is a LATENCY wall (VGPR-capped occupancy), not an instruction-count wall.** Citable for FA-style attention desks (the patch's wins were FA tiles) | W7_madmix_row.txt + W7_madmix_bench_run1.log | GEMM scope closed on gfx900/ROCm-6.2; reopen = FA/attention application or hardware |
| M2 | **GPU-side weight repack** (mx-llama.cpp: separate quant/scale planes, pre-folded scales, de-aliased rows; on-device at load, VRAM-neutral) | MI50 measured: MXFP4 prefill +24–34%, IQ4_NL +133%, Q8_0 +12–41%, decode +19% (MXFP4); "prefill stops paying per-block scale gathers" | layout: kills per-block scale gathers + row aliasing in the GEMM loop — attacks the same 52% census share by layout | **DESK QUEUED** (after M1 census, or parallel on die 2/3) |
| M3 | **v_perm nibble decode** (iacopPBK/llama.cpp-gfx906: 2× v_perm_b32 per 8 nibbles, xor+add sign ≈ 0.25 op/value, no LDS LUT) | code proven on gfx906 (143-star fork); benchmarks as SVGs | instruction-level: replaces shift/mask/LUT decode (~half the 52% tax) | stacks inside M1/M2 kernels; port = small |
| D1 | **peer-write AllReduce** (mx-llama.cpp: beats RCCL ring over PCIe for decode; +9% decode at TP4-PCIe; needs HSA_FORCE_FINE_GRAIN_PCIE=1) | measured 43.9→48.0 t/s decode, 4× MI50 | decode AR replacement at token sizes (prefill keeps RCCL) | queued (decode lever) |
| D2 | **MTP graph-churn + verify-width packing** (constant-node rollback graphs, deferred KV staging, fused gate+up matvec per expert: +41% @ 4-token verify) | mx-llama.cpp: 15.8k prompt prefill 421→1047 t/s (drafting was ~2/3 of prefill!) | kills draft-cost churn — OUR STEM QUANTUM (330–676 ms first-T=27) is the same disease | **STEM WARMUP = the local cure, dispatching now** |
| A1 | **GDN strsm audit** (upstream llama.cpp: rocBLAS strsm FAILS on pre-gfx90a GCN — issue #19972) | fixed upstream via custom solve_tri | verify our chunked dn never touches rocBLAS strsm | quick grep — owner |
| — | LUT-GEMM/FLUTE/T-MAC (partial-sum LUT inference) | 2.1–6.9× on NVIDIA/CPU; **no AMD/GCN port exists anywhere** | documented gap = we'd be first; v_perm is the hardware primitive | parked idea, note only |
| — | aiter/CK/MIOpen on gfx900 | AMD never targeted gfx900 | dead end, confirmed | closed |

THERMAL POLICY UPDATE (same window): pinned-high clocks REVOKE for sustained prefill — auto wins 1.8× (109.7 vs 59.9 burst 2k; power-cap throttling, W7_thermal_pin_ab_row.txt). Serving = AUTO.

## §7 TEAM-GREEN TRANSFER PROGRAM (wo/q3-cg guides, user-directed 2026-09-18 eve)

Sibling line's measured levers, mapped to our gfx900 prefill wall (latency-serial K-chain at 2-waves/SIMD, per the mad-mix insight):

| # | lever | their measured | our state |
|---|---|---|---|
| G1 | **Cross-CTA K-split** (KSC4 grammar: split the serial 80-tile K-chain across S CTAs, fp32 partials + reduce; "the win is CTA COUNT, not chain halving"; intra-CTA variant DEAD) | +32.4% kernel, +6.8% serve | **DESK DISPATCHED** (S∈{2,4,8}, 4-ablation limiter kit first: pure-consume / k-scaling / T-scaling / residency; pre-registered ≥1.20× bar). Attacks our measured disease exactly: 1 CTA per 64-row block walking 80 tiles at 2-waves/SIMD. Never tried on the prefill GEMM. |
| G2 | **Cache-policy/alignment audit** (adjacent-group-pair 16B-aligned chunking; "the floor was the policy, not the silicon") | +19–35% kernel, +9.4% serve | fold into G1 desk (same kernel; our u64 row-group loads → adjacent-pair 16B audit) |
| G3 | **PRMT decode atom** ("a table without memory"; their smem-LUT NO-GO mirrors ours) | +8.2% kernel, +3.2% serve | already in flight = repack desk's v_perm arm (AMD analog: v_perm_b32) |
| G4 | 4-ablation limiter kit (pure-consume / k-scaling / T-scaling / CTA-tiling; no ncu needed) | found an unknown access-pattern ceiling (46% of HBM) | inside G1 step-0.5 |
| G5 | acceptance exchange-rate law (ride-the-neighbor +16.9% kernel but acceptance −1.9 pts → DOMINATED, default-OFF) | caution law | applies to our pending mad-mix/repack arms — both measured bit-exact, no acceptance risk ✓ |
| G6 | GDN strsm audit (upstream: rocBLAS strsm fails pre-gfx90a) | correctness | verified clean — our dn is all-custom kernels, zero rocBLAS strsm |

Meta-laws adopted verbatim (already our practice, now explicit): reproduce-first (±2% same-binary baseline before any arm counts) · measure-don't-estimate · fail-loud · keep the old arm as fixture · banner the kernel that served the rate · tune S per SM count (56, not their 36).

## §8 RE-ENTRY MAP — conditional closures (user order: crossed-off items re-open after upstream fixes; track the conditions)

Every kill below is valid ONLY in the stated configuration. When an upstream condition changes, re-run the named cheap check — do not re-derive from memory, and do not assume the kill still holds.

| closed item | killed in configuration | re-open when | cheap re-check |
|---|---|---|---|
| pk16 (1.34×, numerics-broken) | M=128, 1-CTA-per-64rows serial K-chain, 80 tiles, auto clocks | **if K-split lands** (occupancy/chain structure changes — the 2× MAC density may land differently with latency hidden by CTA count) **or** a numerics-correct variant appears | re-census + same-band bench of the corrected arm (~1 hr) |
| mad-mix (0.48×) | opaque 2048-op asm body at 2-waves/SIMD, 256 VGPR + spill | **if K-split lands** (more CTAs resident = more latency-hiding capacity, may absorb the asm body's scheduling blindness) **or** FA-style attention application (the patch's wins were FA tiles) | re-census VGPR/scratch at the new structure (~1 hr) |
| graph capture (no steady win) | steady chunks not launch-bound (async queue hides launches); GEMM wall 770 ms | **if the GEMM wall shrinks ≥1.4×** (host enqueue becomes a visible fraction) | single A/B leg (~30 min) |
| thermal pin (hurt sustained 1.8×) | 110 W firmware cap + auto-DVFS comparison | **if the power cap is raised** (upp runtime table experiment — needs operator go) | burst+sustained A/B (~30 min) |
| padded swizzle (wash/loss) | current XOR-swizzle baseline + uint4 staging | unlikely; only if the inner-loop access pattern is restructured | census diff (~30 min) |
| compiler road (invariant 17/19/22) | ROCm 6.2 production; 44-wait structure | **each new major ROCm/LLVM release** | re-census (~30 min, CPU-only) |
| pre-dequant (PARKED §2.5, 52% tax measured) | 16 GB/die VRAM; bf16 KV working path | **prefill maxed on current path AND q4 KV migration landed** (both per user order) | VRAM budget pass + prototype bench |
| peer-write AR (NOT APPLICABLE) | P2P canAccess=0 on ALL pairs (fresh, post-iommu=pt) | never on this hardware class | P2P probe per pair (~10 min) if platform changes |
| AR floor tuning (F-ENV booked zero) | RCCL ring at 849.5 µs/1.31 MB | if F-BISECT (gated) shows ≥2% post-finalize-cure | gated bench |
| M=256 ladder (GEMM/token worse pre-pin) | pre-pin droop band | **CLOSED-SKIP (coordinator decision 2026-09-18 ~21:00Z, this row):** the re-entry condition was "only if G1 K-split lands" — K-split is dead (V0 co-residency kill, PLOG-056 receipt set), so the condition is permanently unmet; the +15% ladder class was measured pre-pin (droop-band confound) and GEMM/token got WORSE, i.e. band-equivalent-at-best on the current platform | none — reopening requires a NEW GEMM-economics reopener (gfx906 class or compiler wait-emission change), not a condition on this map |

| **V-arm integration (SCHEDULED — top remaining prefill software lever; coordinator decision 2026-09-19 ~00:45Z):** v_perm LUT kernel = 1.1x on the tiled-GEMM arm (74% of chunk wall => ~+5-8% prefill best case; AMBER "serve-leg decides" from W7_REPACK_RESUME) | bench-level, uniform data, pre-thermal-model (W7_repack runs) | NO condition gates it — but the execution spec is MANDATED by two laws learned since: (1) CARRIAGE: integration MUST carry the 2 latent V-family bug fixes (PLOG-056; fixes live in w7_repack_bench.cu + w7_vperm3_bench.cu, NOT yet in src); (2) MEASUREMENT: the serve-leg A/B MUST use the ordinal-paired design (PLOG-060: same bin fresh boots, position-matched probes, ±2% within-pair) — the bench-level 1.1x must survive the thermal model before promotion | execution: fresh-context desk (agent capacity resets ~12:42 2026-09-19, or a fresh inline session); est. full desk = integration + cells + paired serve-leg + battery |

STANDING AGENT CHECKPOINT LAW (user order, after a desk death on model-request failure): every dispatched desk banks incrementally (commit+push per milestone) AND maintains/updates a resume file at every milestone — a desk death (model-request failure, reboot, box wedge) must never lose more than the current un-banked step. Dispatch prompts carry this explicitly.

## §9 DAY-END SYNTHESIS (window 7, 2026-09-18) — the wall is now fully characterized

**THE GEMM WALL, NAMED (K-split desk limiter kit + all prior receipts):** the prefill GEMM is
**access-pattern bound at fixed co-residency**. The pure-consume ablation (all decode+MMA deleted,
same loads, xor-consume) runs at **0.85–0.95× of the FULL GEMM time** at 36–51 GB/s effective — the
math was never the constraint; the access pattern's achievable rate at 2-blocks/CU co-residency is.
The 52% conversion stream is the scheduler's latency-hiding material for that pattern (mad-mix proof);
the K-chain is linear in k (not chain-bound); M-scaling shows headroom exists only above the served
M=128; CTA-count changes cannot add co-residency (128 VGPR already holds 2 blocks/CU — K-split
monotone loss 0.88–0.98×, measured today).

**The three exits, final status:**
1. **Reduce the latency source:** v_perm register-LUT decode = **AMBER 1.136× mean, 1.24× on the
   starved geometry, waits 212→32, bit-exact, no requant** (W7_repack_row.txt). EXTENSION: the V arm
   sits at 113 VGPR — **if register pressure drops ≤96, co-residency rises to 3 blocks/CU**, directly
   raising the access-pattern ceiling. This is the one live GEMM lever.
2. **Fill stalls with independent work:** falsified five ways (V1/V2 shapes, mad-mix, padded swizzle,
   K-split, graph capture). Closed.
3. **Raise the silicon cap:** gfx906-class (dot10/v_dot4, 2× fp16 rate, P2P) or a power-cap raise
   (upp, operator-gated). The 110 W cap + no-P2P topology are platform facts (measured today).

**DAY LEDGER (window 7):** prefill 76 → **109.7 burst / 62.5 @10k-first-soak** client class
(+44%/+77%); decode 41 (permanent). Landed: finalize cure (660ms H2D → 0.008), stem warmup,
dn ×16, gqa, gate, thermal policy reversal (auto 1.8× over pin), P2P closure, AR floor closure,
graph-capture kill, mad-mix kill, padded-swizzle kill, K-split kill + limiter naming, V0 census
(52% conversion share measured), compiler-road closure (LLVM 17/19/22 invariance). All receipted,
all merged, PLOG chain current. The pre-dequant lever (§2.5) remains parked per user order; today's
access-pattern naming is the informed context for when its conditions unlock.

## §9b NIGHT-RUN SYNTHESIS (2026-09-18 21:00 → 2026-09-19 ~03:00, inline after platform usage limit killed all desk agents ~21:45) — the stall is cured, the instrument is honest, and the platform ceiling moved

**THE LANDINGS (all paired/battery-gated, all merged to amd/main):**
1. **Pinned-staging cure (NINFER_H2D_PINNED_STAGE=1, PLOG-058/059):** the per-chunk ~995 ms
   pageable-H2D enqueue stall (each ~512-B chunk-ids upload could not return until the
   in-order stream executed it at chunk end) is cured by a per-thread ring of 64 pinned
   slots with event-guarded reuse (torn-payload impossible by construction). Cell-priced
   ~300x host-block collapse; **paired same-window A/B: -18.6/-27.1/-32.6% per ordinal,
   pinned FLAT vs unset RISING** (PLOG-060). Runbook BIN 2c8901d3d18adef1.
2. **Trace instrument fixed (PLOG-062):** the ar bracket was last-record-wins (mlp AR
   overwrote the mixer AR's begin) — ar read 60.3 while nccl device busy was 119.5. Fixed
   with a per-layer kAr=4 subslot ring; **R6-verified: ar column now reads 123.0/123.3 =
   rocprof ground truth**, body rebalanced 152 -> 43-74, wall flat.
3. **The platform ceiling, measured (PLOG-064):** the 10k soak is a thermal STATE FUNCTION,
   not a band — **hot start 63.1 / warm 70.7 / COLD START 104.0 tok/s prefill** on the same
   cure config (1.66x the banked 62.5 era). The thermal integrator is quantified
   (46->80 C in 85 s of bursts; ~90 s drain constant; DVFS parks instantly; ~3 min idle =
   reset to the fast band). Single-soak claims banned; start-temp mandatory on future rows.

**THE FALSIFICATIONS (the map, shortened with receipts):** co-residency (2 independent desks,
PLOG-056), AR/compute overlap (0.0% on all four dies even on the cure timeline, PLOG-063),
RCCL env tuning (16ch/Tree worse, LL128 wash — the SHM ring at 2 channels IS the envelope,
PLOG-065), draft-vocab widening (coverage 97.6-99.4%, refused by rule), and the decode
"50x GEMV gap" (already cured at PLOG-044; head at 1.15x of roofline, PLOG-057). Two latent
V-family bugs found and RED->GREEN closed along the way (--v3parity guard, PLOG-056).

**THE HONEST BOOKS (per 128-chunk, cure config, honest columns):** wall ~978-992 ms =
gemm 756-806 (76%) + ar 101-141 (12%, host-SHM-bound, structural — zero overlap possible,
configuration-immovable) + body 43-74 + gap 15-69. Decode: acceptance 0.71-0.94 by register
(class-dependent), 2.45-2.87 tok/round, KV-decay 40.7->27.0 t/s over 100->600 generated.

**WHAT REMAINS (next-window desks, parked with receipts):** (a) V-arm integration — top
remaining prefill software lever, ~+5-8% best case, REV2 §8 SCHEDULED row mandates the 2
bug-fix carriage + ordinal-paired serve-leg; (b) AR algorithmic reduction — fuse per-layer
mixer+mlp collectives (halve count) or quantized AR messages (halve bytes); overlap and env
tuning are both dead, so this is the only AR lever left; (c) q4 KV migration (user's stated
next milestone) — unlocks the parked §2.5 pre-dequant per its double-keyed conditions.
**NIGHT LEDGER:** 10k prefill 62.5-era -> 104.0 cold-start / 70.7 warm / 63.1 hot (state
function); 2k burst ~109.7-era -> 18.4-18.8 s walls held FLAT under sustained load (the cure
kills the load-time decay the old config suffered). PLOG chain 056-065, 62 links
byte-reproducible at last verify, all merged. Agent capacity returns ~12:42 2026-09-19;
V-arm and AR-fusion desks are ready-to-dispatch with full specs banked.
