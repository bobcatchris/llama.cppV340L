# VERIFY DECOMP — the 33-vs-270 GB/s gap, decomposed (no-GPU desk, 2026-09-17, amd/tp4-cure)

**Seat:** NO-GPU analysis agent. **Method:** banked-artifact mining (`results/amd/coherence/`)
+ code-read of the verify round. Zero GPU work, zero builds, zero src/ edits, no process
touched. **Question:** why does the in-serving verify body imply ~33 GB/s NVFP4 weight
stream (given) while the isolated tuned GEMV/small_t benches hit 265-280 GB/s (76% of the
378 GB/s read ceiling) — and what single measurement kills the biggest remaining owner.

**Binaries named by row:** pre-tune `07ad7eccc0b97cc0` (TIMING1/K1/K3/THARM/PF), HOLE1
`5735aaf3e9ddd410`, GEMV-tuned `e341a2072c6f5078`, SMALLT pre/post `be4c9dd763cd649f` →
`1e5745f9f8febcc3`, current certified `409450b83ffa4be8` (tiled race-fix; ONESHOT_A anchor).

---

## 1. Round budget table (banked, count600 shape, MTP k=2 GRAPHS-ON, TP4 world=4)

| phase (B1) | TIMING1 pre-tune (bin 07ad7ecc, cold) | HOLE1 remap-on (bin 5735aaf3) | CURRENT tuned (SMALLT2 1e5745f9 / ONESHOT_A + TILED 409450b8) |
|---|---|---|---|
| **1. Target Verify (T=3)** | **195.34 ms (92.2%)** | **215.03 (92.1%)** | **111.4-112.8 (87.2%)** |
| 2. Accept & D2H | 0.06 | ~0.1 | ~0.1 |
| 3. GDN State Rebase | 0.01 | ~0.05 | ~0.05 |
| 4. Prepare Next Round | 0.08 | ~1.8 (prep incl) | ~1.8 |
| 5. MTP Alignment (T=1 fwd) | 8.56 | 8.92 | 8.4-8.5 |
| 6. Select Accepted Hidden | 0.01 | ~0.01 | ~0.01 |
| 7. MTP Propose (d0) | 0.29 | 0.34 | ~0.3 |
| 8. AR Draft Chain (d1..dk) | 7.61 (3.6%) | 8.83 | 7.5-7.6 |
| **round total** | **211.95** | **233.40** | **128.4-132.6** |
| decode tok/s | 13.87 | 8.8 (cooler-window boot) | **23.0-23.2** |

Sources: `TIMING1_row.txt`, `TIMINGK1_row.txt`, `TIMINGK3_row.txt`, `HOLE1_row.txt`,
`SMALLT_row.txt` (BOOT 2), `TILED_DATUM_row.txt`, `ONESHOT_AR_row.txt` (boot A).
Structure (verify ~87-92%) reproduces across every bin; absolute wall carries a ±25%
cross-boot thermal band (THARM1/B3HARDEN). k-sweep (pre-tune): T=2 160.97, T=3 195.34,
T=4 237.43 → slope 34.4→42.1 ms/position (superlinear beyond T=3), intercept ~126.5 @T=1.

**Verify interior, pre-tune (HOLE1 LTRACE, same boot as its 203 ms p50 — `HOLE1_ltrace_table.txt`):**

| segment | ms/round | derivation |
|---|---|---|
| layer loop (64 layers, body events) | **138.9** | ~2.15 ms/layer, UNIFORM; FULL-attention layers only +0.15 vs GDN |
| — NVFP4 linears (5 W4 problems, small_t arm) | 62.7 | 48×0.987 + 16×0.956 (bench BASE 53-57 GB/s cross-check) |
| — in-verify TP ARs (2/layer, axpy incl) | 14.5 | 128 × ~114 µs ≈ eager ring RTT law (98-101 µs) + arrival |
| — non-linear body + intra-layer gaps | **61.7** | loop remainder — no per-op instrument pointed at it |
| outside the loop | **~64** | verify 203 − loop 138.9 |
| — named tail ops (embedding, finalnorm, BF16 lm_head 635 MB/rank, argmax, 3 logits allgathers ~0.35, allreduce_argmax) | ~4 | embedding+head are BF16 (`inventory_nvfp4.py:51` excludes both); lm_head ≈ 3-4 ms |
| — **unnamed tail residual** | **~60** | subtraction; no owner named yet |
| **verify phase** | **203-215** | |

**Post-tune split (loop vs outside): UNMEASURED — this is the open hole (see M1).**
The SMALLT landing cut verify 194.0→111.95 (−82 ms) against a −43.3 ms linear-only
projection (62.7→19.4 at S5 rates 147-192 GB/s); either the unattributed segments shrank
too (skew/gap coupling) or cross-boot thermal contributed. Current implied whole-phase
rate: ~4.1 GB/rank over 112.8 ms ≈ 36 GB/s; the linears slice alone streams at S5 bench
rate (19.4 ms for the layer weights ≈ 186 GB/s) — **the tuned kernels DO run at bench
class in serving; the 8x gap is not one owner, it is the sum of everything that is not a
linears launch.**

**Target math:** 50 tok/s at t/r 2.94 needs round ≤ 58.8 ms → verify ≤ ~41 ms with today's
draft side (16.9 ms). If every NVFP4 byte streamed at the 265 GB/s bench rate plus current
AR floor + body + tail, the round lands ~55-60 ms — i.e. **the 50 tok/s target IS the
gap-closure target; no new physics required.**

## 2. Launch census per verify round (code-read, T=3, batch=1, this worktree's line numbers)

Path: `tp2_backend.cpp` worker → :3015 `target_verify_batch` → `text_context_impl.h:1795`
`target_verify_batch_impl` → :1843 embedding → :1845 `run_layers(Verify)` → :1853 finalnorm
→ :1855 lm_head → :1857 argmax → then 3× `allgather_local_bf16` (:3036-3048) +
`allreduce_argmax` (:3143) + accept kernel + pinned D2H (:3161-3164).

| site | ops per instance | cite |
|---|---|---|
| GDN mixer (verify arm) | rmsnorm+gating proj (REPLICATED weights) · input-proj fused tp_gemv (small_t tuned) · `causal_conv1d_silu_snapshot_split3` · gbeta unpack · `gated_delta_net_snapshot` · gated rmsnorm · out_proj tp_gemv ≈ **9-10 kernels** | `text_context_impl.h:2234-2420` |
| FULL mixer (every 4th layer) | input rmsnorm · proj tp_gemv · qkv unpack · q-rms · k-rms · rope · attend · sigmoid_mul · o_proj tp_gemv ≈ **11-12 kernels** | `:2021-2116` |
| mlp_tail (every layer) | rmsnorm · gate_up tp_gemv · silu_mul · down tp_gemv = **4 kernels** | `:2760-2784` |
| collectives/layer | mixer AR + mlp AR, each **ncclAllReduce + separate axpy** = 4 device ops | `tp_group.cpp:336-347` |
| head/tail | embedding · finalnorm · lm_head (BF16) · argmax · accept · **3 ncclAllGather** · allreduce_argmax | cites above |

**Totals:** 48 GDN×~15 + 16 FULL×~17 + head/tail ≈ **~1000 device ops/round**
(~740 kernel launches + ~260 collective-class ops incl 128 axpys). The round is **EAGER**
(no `cudaStreamBeginCapture` in `src/runtime/tp2/`; audit §0-H4), so all ~1000 are
host-enqueued fresh each round: at the banked micro-bench 1.56-1.83 µs/launch
(`VERIFY_LAUNCH_AUDIT_2026-09-17.md` §5) that is **~1.6-1.8 ms host enqueue per round,
fully hidden** behind 112 ms of device work (device 100% busy, `HOLE1REMAP_clocks_t12.txt`).

**Serialization skeleton (the load-bearing structure):**
1. **Exactly ONE hard host sync per round**: `cudaStreamSynchronize` at `tp2_backend.cpp:3188`
   (accept D2H drain). Everything of the verify forward is enqueue-ahead; host stalls there
   until the whole forward drains. This is why round wall ≈ device verify + host-paced
   draft side: 112 + 8.5 + 7.6 + 0.3 + 0.5 ≈ 129 ✓.
2. **The draft chain is host-staged**: propose/AR-chain (~16.4 ms) is per-step D2H → host →
   re-enqueue; 7.6 ms ≈ 76-78 serialized ~98-101 µs collectives (TIMING1 note).
3. **In-verify ARs are device-side but latency-owned**: 30 KB payloads (T=3×5120×bf16)
   pay the full ring RTT; arrival skew is measured nil across ranks (OPTRACE sum-skew
   0.017-0.019 ms, H1 acquitted).
4. Phase arithmetic closes the wall with **no unaccounted phase** — the mystery lives
   strictly INSIDE the verify event window (§1 residual rows).

## 3. Ranked hypotheses for the 33-vs-270 GB/s gap

**H1 — Unattributed intra-forward stream time (body kernels + inter-op gaps), ~50-60 ms of
the pre-tune round, current post-tune share UNKNOWN. RANK 1 by mass.**
Supporting: pre-tune same-boot identity: loop remainder 61.7 ms after naming linears+ARs;
tail residual ~60 ms after naming ~4 ms of tail ops; post-tune verify (112) exceeds
linears (19.4) + ARs (~7-14) + named tail (~4) + body-bytes estimate (~3-5) by ~50-60 ms.
The body ops are latency-class tiny kernels at T=3 (conv/delta-net/norms/unpacks: µs-scale
each, dependencies serialize them) — they cannot own 60 ms at their byte sizes, so the
mass is most likely **inter-op stream gaps exposed by the eager dependency chain** —
which no banked instrument currently resolves below phase level (`tpv_probe` at
`text_context_impl.h:945` is a `cudaGetLastError` check only, zero timing).
**Kill: M1, escalate M2.**

**H2 — Thermal integrator: a MULTIPLIER (up to 2.82x), not the 8x owner.**
Supporting: THARM1 same-boot A/B: 213 ms/round cold vs 606 ms soaked (no-gap), full
recovery after ~3 min idle at the same 84-85 C edge; TIMINGPF2 clock proof 1500→560-775
MHz sclk collapse at knee; B3HARDEN 294.9 warm vs 244.3 cooled; COOL10K decode 3.65 is
soak-confounded. **Exonerating:** HOLE1's 203 ms verify was sampled at SCLK 1269-1350 /
MCLK 945 (full clocks, power AT the 110 W cap) — the gap exists at good clocks.
**Domesticate: M0 (row-law), already banked discipline; every A/B leg <60 s or ≥2-3 min gaps.**

**H3 — AR latency class: ~14.5 ms pre-tune inside verify + 7.6 draft chain + 0.4 tail
collectives ≈ 22 ms/round ceiling. RANK 3 — real, capped, currently unfixable.**
Supporting: 128 ring ARs × ~114 µs (RTT law) + 128 axpy launches (one-shot AR is
world==2-gated, `tp_group.cpp:136-139`); the W4 one-shot env arm (`NINFER_TP_ONESHOT_AR=1`)
**WEDGES 2/2 at TP4** (T2 RED captures: stranded rank varies, R2 bounded-wait never fires,
R3 rank_step pairing suspect) — NOT adoptable in `409450b83ffa4be8`; repair is
design-level. T2 also notes the SMALLT landing halved the AR-phase share like every other
phase (arrival-skew coupling). **Kill: M3 after the AR-owner repair; ceiling ~12-15 ms —
second order vs H1.**

**H4 — Small-M verify width (T=3 issue wall): prices the SLOPE, not the intercept.**
Supporting: small_t S5 at T=3 = 39-50% of read ceiling vs GEMV T=1 73-77% — every weight
byte carries 3x the ALU/LDS work; the kernel leaves the DRAM-wall regime (issue wall).
Pre-tune T-slope 34.4→42.1 ms/position superlinear. Remaining kernel-side levers
(split-K across warps, extra accumulator chains) are banned by the
`nvfp4_amd_codec.h` contract pending documented re-scope; tiled GEMM (M≥64) does NOT
apply at M=3. **Kill/price: M4 (post-tune k-sweep); if intercept dominates, H1 owns.**

**H5 — mclk not ramped in serving bursts: EXONERATED for steady decode.**
mclk idles at 167 MHz but the HOLE1 mid-decode sample shows 945 MHz (= max) under load;
benches' 5 s mclk hammer matches. Only a round-entry transient remains (sub-ms class).
No cell owed.

**H6 — LDS/occupancy differences bench-vs-serving: EXONERATED for the linear slice.**
The serving arm runs the SAME kernel at the SAME (n,k,T) as the bench
(`[SMALLT] arm=nvfp4_small_t_tuned_kernel` proof lines, all five problems), and the
serving delta direction matches the bench projection. Reopens only if M1 shows in-serving
linears BELOW S5 bench rate.

## 4. Next GPU window — decisive measurement list (cheap falsification first, SD-1)

| # | measurement | env/arm | runnable how | kills/prices |
|---|---|---|---|---|
| **M0** | clocks+temp sideband sampler on EVERY leg (2 s, sysfs pp_dpm_sclk + edge); legs <60 s continuous or ≥2-3 min idle gaps | none (discipline) | TIMINGPF2 sampler pattern, banked per row-law | guards all rows vs H2 (2.82x multiplier) |
| **M1** ★ | **Post-tune LTRACE+OPTRACE re-run**: one boot of banked `409450b83ffa4be8` (FIRST verify the arms are in the binary: `strings` for `NINFER_VERIFY_LAYER_TRACE`/`[OPTRACE]`, else rebuild+bank per BOOT_LAUNCH_RUNBOOK §4), count600, cool window | `NINFER_TP2_TIMING=1 NINFER_TP2_OPTRACE=1 NINFER_VERIFY_LAYER_TRACE=1 NINFER_DRAFT_VOCAB=<abs>` | one boot, zero new code if arms present; feeds `tools/optrace_analyze.py` + the ltrace table | **H1**: splits verify 112 into loop-vs-outside and body-vs-AR per layer; re-ranks everything below |
| **M2** | per-op timing probe at the existing `tpv_probe` sites (one GDN + one FULL layer + tail ops, hipEvent pairs summed per tag per rank) | new env-gated arm (owner's file — `text_context_impl.h:945` is error-check-only today) | one boot; B1/OPTRACE pattern is the template (`tp2_backend.cpp:597-672`) | H1 residual: names gap vs kernel inside the layer, µs resolution |
| **M3** | ONESHOT-W4 A/B rerun — only after the AR owner's design repair (R2 bounded-wait + R3 pairing) | `NINFER_TP_ONESHOT_AR=1` vs absent | T2 pattern; GREEN leg must byte-match `ONESHOT_A_count600.json` + warmup ids 760 1156 1018 328 | H3 (ceiling ~12-15 ms) |
| **M4** | post-tune k-sweep k=1 and k=3 (TIMINGK1/K3 pattern, same boot per leg discipline) | `--draft-tokens 1` / `--draft-tokens 3` boots | two short legs, cool-window | H4: post-tune T-slope + intercept; also refreshes the marginal-k table for the 50 tok/s plan |
| **M5** | rides any boot above: `[SMALLT]` volume lines + per-request decode/prefill + round B1 — keeps the cross-bin anchor chain unbroken | `NINFER_SMALLT_TRACE=1` | free | continuity/thermal drift witness |

**The single most decisive measurement: M1.** One boot, no new code (if the HOLE1 arms are
present in the banked binary), and it converts the biggest unowned number in the round —
~50-60 ms of verify — into either (a) loop-resident body/gap time (→ M2 per-op arm is the
follow-on, fusion/graph-capture become the levers) or (b) tail-resident time (→ logits
path, lm_head, and the 4 tail collectives become the levers). Every other row is priced
and capped below it.

## 5. Honesty row

- Pre-tune loop/tail residuals (61.7 + ~60 ms) are subtraction numbers, not per-op
  measurements; H1's mass estimate inherits that coarseness — M1 exists to replace it.
- SMALLT BOOT1-vs-BOOT2 (−82 ms verify) is a same-day but not same-hour comparison; the
  −43.3 ms linear projection leaves ~39 ms attributable to either unattributed-segment
  coupling or thermal drift. Not resolvable retroactively; M1 re-measures forward.
- This worktree carries ANOTHER agent's uncommitted edits in
  `src/ops/linear/nvfp4/*` and `results/amd/coherence/nvfp4_smallt_roofline_bench.cu`;
  this doc cites none of them by line number, and this seat did not touch them.
- Line numbers are from THIS worktree's working tree (amd/tp4-cure) and may drift.

---
Bases: TIMING1/K1/K3/PF/PF2 · THARM1 · COOL10K · BASE2 · HOLE1_row + HOLE1_ltrace_table +
HOLE1REMAP_clocks_t12 · SMALLT_row · TILED_DATUM_row · ONESHOT_AR_row (T2) ·
ONESHOT_W4_notes · ROOFLINE/GEMV_TUNING rows · VERIFY_LAUNCH_AUDIT_2026-09-17 (§0 measured
supersession noted) · code: `src/runtime/tp2/tp2_backend.cpp` (:2981-3188 verify→accept),
`src/runtime/tp2/tp_group.cpp` (:136-139, :336-356), `src/core/multi_gpu/one_shot_allreduce.*`,
`src/targets/qwen3_6/impl/runtime/text_context_impl.h` (:945, :1795-1861, :2021-2116,
:2234-2420, :2760-2973), `tools/convert/qwen3_6_27b/inventory_nvfp4.py:51`.
