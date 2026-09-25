# 122 — KVarN performance: the dedicated knowledge base

**This is the single reference for KVarN decode + prefill performance.** Everything measured,
tried, disproved, and ruled out — including dead ends, because a dead end re-attempted costs a day.
Maintained by whoever touches KVarN perf next. Cross-ref: `docs/121` (session handoff),
`docs/120` (Phase-3 task list), `docs/119`/`119b`/`119c` (current design), `docs/104` (master plan).

Scope: KVarN = K4V2 KV cache (4-bit K codes + 2-bit V codes + 1152 fp32 scales per layer/head/page).
Last updated: 2026-08-31.

---

## 1. Hardware and theoretical floors (2× RTX 5060 Ti, 16 GB, sm_120)

| quantity | value | source |
|---|---|---|
| SMs per GPU | **36** | runtime-audited, `results/106_launch_config_audit.md` |
| L2 | **32 MB** (persisting max 20 MB) | `cudaGetDeviceProperties` |
| Peak DRAM BW | ~448 GB/s | 128-bit GDDR7 @ 28 Gbps |
| `sharedMemPerBlock` | 49152 | default, no opt-in |
| **`sharedMemPerBlockOptin`** | **101376** | usable dynamic = this − static − ~1024 reserved |
| `sharedMemPerMultiprocessor` | 102400 | bounds blocks/SM |

**Floor arithmetic (docs/78 §):** packed decode DRAM floor @250k = 2.24 GB/step → ~5 ms → ~200 tok/s
ceiling, ~90 tok/s at 45% efficiency. int8 floor @250k = 18.5 KB/token → 4.6 GB/step → ~2× worse
floor. **So KVarN's theoretical ceiling is well above every number below — we are nowhere near
bandwidth-limited.**

### Cost model: bytes and nanoseconds per key

| variant | bytes read per key | measured ns per key | bytes efficiency |
|---|---|---|---|
| **kvarn** | ~264 (K 128 + V 64 + scales 72) | **~153** | 1.73 ns/byte |
| bf16 | ~1024 | ~116 | 0.113 ns/byte |
| int8 | ~528 | ~51 | 0.097 ns/byte |

> **⚠️ CORRECTED 2026-08-31 (kernel re-measurement).** The ns/key in this table is WRONG — it is
> ~40x too high for kvarn (and ~20x for int8) and does NOT match the kernel. Authoritative
> per-launch decode (tests/bench_kvarn_attention, NINFER_KVARN_DECODE=unified, T=1, 160256 keys):
>
> | variant | per-launch ms @160k | measured ns/key | GB/s (peak 448) |
> |---|---|---|---|
> | kvarn | 0.612 | **3.74** | 69 (15%) |
> | int8 | 0.416 | **2.60** | 99 (22%) |
>
> **So kvarn is 1.44x int8 per-key WORK at HALF the bytes — NOT the doc's '3x per-key compute'.**
> Neither is bandwidth-bound. Fit of the kvarn curve T=0.006 + 3.74e-06*keys (R^2~0.999): per-key
> cost is FLAT (~3.74 ns/key), attention 85-99% at all contexts — the kernel does NOT decay;
> t/s falls as 1/N only because there are more keys per step. The doc's '$5 decay 37%' is a
> stale round-level interpretation.
>
> **IMPLICATION the whole doc was built around a wrong number: KVarN is 1.44x int8, so closing
> that makes KVarN BEAT int8 (fewer bytes). Phase 2 IS possible — the target is a ~1.44x
> kernel-level speedup, not an intractable 'reduce 3x compute'.**

**This table was the most important single artifact in this document** (now corrected above).
bf16 and int8 cost scales with bytes (1.94× bytes → 2.27× cost) — memory-bound, as expected.
**KVarN reads half of int8's bytes and (at the corrected numbers) is only 1.44× per-key cost.**
KVarN is **compute-bound** (relative to its tiny byte count), and has been in every measurement
since docs/83 profiling (K-dequant 22–24%, QK-MMA 18–19%, PV-MMA 18–19%).

Scale traffic is 72 B/key = **37% overhead on top of codes**, vs int8's ~3%. The scale table's
cross-page stride is `1152 × n_layers × n_heads` floats = **144 KB** on this model (n_layers=16),
so per-page scale reads have no spatial locality across pages. **But this is NOT the bottleneck** —
packed prefetches scales one page ahead and still decays identically (see §5).

## 2. Decode performance — full history

All numbers are server-measured decode t/s unless noted. "in-wall" = context fits the staged shadow.

| date | build | 10k | 40k | 250k | note |
|---|---|---|---|---|---|
| pre-D-18 | shadow | 69 (in-wall) | 71.2 | **30.5** | shadow allocated |
| 08-27 | wall removed (step 1) | — | **57.3** | ~30 | −13.9 @40k: exposed materialize cost |
| 08-28 | **option B + A2** | 64.6 | **66.7** | **40.0** | +16% @40k, +33% @250k, beats shadow's 30.5 |
| 08-29 | official baseline v1 (packed) | 69.1 G / 67.1 S | 64.0 G / 63.0 S | 43.5 G / 40.9 S | G=greedy, S=sampling |
| 08-30 | **unified slice4, pre-fix** | 43.1 | 20.2 | **4.4** | −37% → **−90%**: regression |
| 08-30 | unified + Hadamard port only | ~54.4 | — | — | 0.60× packed — bought almost nothing |
| 08-31 | **unified + coalescing (current)** | **67.8 G / 72.1 S** | **68.3 G / 67.7 S** | **42.1 G / 32.5 S** | **parity: ms/round 0.928–1.048×** |

**Current state: unified decode is at parity with packed and is the shipped default**
(`29d0a797`); `NINFER_KVARN_DECODE=packed` is the rollback lever.

### The three decode routes that have existed
1. **staged shadow** (removed) — bf16 copy of recent pages. Cost 1342 MiB/rank VRAM. Removed in
   docs/83 step 1; `NINFER_KVARN_NO_SHADOW` became permanent. **Do not reintroduce.**
2. **materialize + bf16 small_t / flash** — packed→bf16 paged temp, then the proven bf16 kernel.
   Still the prefill route. Bandwidth-bound on the temp *write*.
3. **packed code-read** (`gqa_attention_kvarn_decode_packed.inc`) — warp-specialized, 256 threads,
   2 MMA consumer + 6 dequant producer warps, cp.async double-buffered `stage_k`/`scales_scratch`,
   99 KB smem, 1 block/SM. **Numerics: Q rotated once; K unpack-only (`kvnorm=1.0`); V unpack-only
   (`vvnorm=1/16`) deferred to `kvarn_mma_acc_fwht` (LIVE at `.inc:735`).**
4. **unified slice4** (new, now default) — verbatim slice-2 bf16 body + KVarN prologue. 64/128
   threads, no warp specialization, 32-key tiles, 2 blocks/SM, no dynamic smem.

## 3. Prefill performance — full history

| date | build | 10k | 80k | 160k | 250k |
|---|---|---|---|---|---|
| 08-29 baseline v1 (kvarn) | materialize+flash | 720 | 598 | 534 | 466 |
| 08-29 baseline v1 (bf16) | direct flash | 834 | 711 | 614 | — |
| | | **−13.7%** | **−15.9%** | **−13.1%** | no comparison |
| 08-31 measured (this session) | materialize+flash | — | — | — | — |
| | 21 829-token prompt | kvarn **682.5** vs bf16 **779.1** t/s pp = **−12.4%** | | | |

Prefill is **88% → 99.7% of wall time from 10k → 250k** — this IS the user-visible number at long
context. The deficit is roughly **flat** across context (unlike decode's growing deficit), which is
the signature of a linear-in-tokens extra pass, not a super-linear blowup.

**History:** the original prefill was a single kernel re-running per-tile dequant in every
(q_block, q_head) CTA — **O(n²)**, 409 vs 97 ms/call. docs/66 fixed it with the 2-pass split
(materialize once per tile, then flash). **What remains is the temp write, not an O(n²) term** —
docs/104 §4 still describes it as O(n²); that is stale.

### Where prefill time actually goes (nsys, 21 829 tokens, this session)

| kernel | time | share of the 3.65 s gap vs bf16 |
|---|---|---|
| **`quantize_tile_kernel`** | **3.61 s** | **98.9%** |
| `gqa_attention_kvarn_materialize_kernel` | 0.15 s | 4.1% |
| `scatter_kvarn_tile_kernel` | 0.03 s | 0.8% |
| `gqa_attention_kvarn_decode_packed` | 0.03 s | — |
| (bf16 total 27.74 s; kvarn total 31.38 s) | | |

**`31.38 − 3.61 = 27.77 ≈ 27.74`** → removing quantization from the critical path alone achieves
**exact parity with bf16**, with no algorithm, numerics, or layout change.

### Why quantize_tile costs what it costs
`src/ops/kvarn/kvarn_tile_cuda.cu:311-345`, per 256×64 tile (16 384 elements):
- `fwht_along_channel`: 8 butterfly stages
- **`kSinkhornIters = 16`** full-tile iterations ← dominant
- quantize pass: 2 sweeps × 256 channels × 64 keys
- `TileShared`: `float X[256*64]` = 64 KB + rA/cB/rA2/cB2/best_rA/best_cB ≈ **72 KB dynamic smem**
- 21 824 calls = 341 pages × 2 kv_heads × 16 full-attention layers × {K,V}; 165–186 µs each
  ≈ **10 ns/element**

**It is algorithm-bound, not memory-bound.** A code-layout change removes zero operations from it.

## 4. Every lever tried — verdicts (do not re-attempt without new information)

| lever | verdict | evidence |
|---|---|---|
| **staged bf16 shadow** | ❌ removed | cost 1342 MiB/rank; wall did nothing beyond 40.4k. `doc83_wall_removed.md` |
| **option A: defer Hadamard inside materialize** | ❌ **dead end** | 40k 57.3 → **54.8 (−4%, worse)**. Proved materialize is **bandwidth-bound on the bf16 temp WRITE**, not Hadamard compute. `doc83_optionA_bandwidth_finding.md` |
| **option B: route T=2..6 verify to packed** | ✅ shipped | +16% @40k. Needed **token-blocking** on grid.y (group=6 ⇒ T=4 = 24 rows > 16-row MMA tile → register wall) |
| **A1: deferred acc-FWHT (V Hadamard out of the loop)** | ✅ shipped, bit-exact | CPU-validated `tests/kvarn_acc_fwht_cpu_ref.cpp`; ~33% kernel win (104→154 GB/s) |
| **A2b: QK-side code-space fold** | ⚠ deprioritized | "K-dequant is affine-only, no Hadamard; revisit only if A2 under-delivers" |
| **lever 1: int8 s8-MMA on raw K codes** | ❌ **NEGATIVE** | 0.313 vs 0.276 ms @T=4 = **+13% SLOWER**. Per-page query-fold overhead > MMA speedup. Code exists behind `NINFER_KVARN_DECODE=int8qk`. `doc83_lever1_int8_finding.md` |
| **M1: code-space QK + end-to-end CPU validation** | ✅ PASS (CPU) | `tests/kvarn_codespace_qk_cpu_ref.cpp`; fold = `q''[d]=q_rot[d]·kc0[d]`, `c0=Σ q_rot·kzp` (rank-0 per page). Gates M2 |
| **rotate Q once + K/V unpack-only (unified)** | ✅ shipped | `4b9d30e7`/`802322a4`. **But bought almost nothing alone** (0.62×→0.60×) |
| **coalesce K-code reads via smem staging** | ✅ shipped, **the big one** | `802322a4`. 54.4 → 92.1 t/s vs 91.2 packed reference. Up to **9.6×** at 250k |
| **remove per-head D2D copies in commit/hydrate** | ⚠ **free but worthless** | `a59de6df`. Removed exactly 43 648 copies; kernel time 30.53→30.51 s. Copies cost launch slots, not time |
| **warp specialization / more threads** | ❌ not binding | unified reaches parity with **zero** warp specialization at 64/128 threads. Agent 1's I8 Wc=4 test was also **a wash** |
| **occupancy (blocks/SM)** | ❌ not the driver | packed is 1 block/SM (99 KB), unified is 2 (47 KB) — **both decay identically** |
| **scale-table layout / L2 locality** | ❌ not the bottleneck | bandwidth is ~6% of peak; packed prefetches scales ahead and still decays |

## 5. The decay analysis — RESOLVED: it was a t/s-vs-ms/round artifact, NOT a kernel slowdown

**The '-37% decay' (69.1 t/s @10k -> 43.5 t/s @250k) is a MEASUREMENT artifact of using raw
decode t/s, not a kernel slowdown.**

Kernel-level (tests/bench_kvarn_attention, isolated): KVarN ns/key is FLAT (~3.74 ns/key) with
attention share 85-99% at all contexts. The per-launch decode follows T = 0.006 + 3.74e-06*keys
(R^2~0.999) — perfectly LINEAR. The kernel does NOT decay.

Round-level (the authoritative ms/round = 1000*tok_per_round/tps, which strips the acceptance
noise the doc warns about):

| ctx | kvarn ms/round | int8 | bf16 | kvarn/bf16 | kvarn/int8 |
|---|---|---|---|---|---|
| 10k | 45.44 | 41.67 | 42.09 | 1.080x | 1.091x |
| 25k | 47.58 | 43.71 | 44.73 | 1.064x | 1.089x |
| 40k | 49.53 | 44.73 | 46.72 | 1.060x | 1.107x |
| 80k | 55.15 | 46.85 | 50.83 | 1.085x | 1.177x |

**KVarN is only ~6-9% slower than bf16 and ~9-18% than int8 in ms/round.** The raw t/s decay was
because t/s = C/keys (more keys/step) AND acceptance/tok_per_round drift. This is consistent
across ctx -> it is a UNIFORM ~1.06-1.09x overhead, not a growing decay.

**This invalidates the whole 'KVarN is 3x per-key compute, reduce compute is intractable' story.**
The REAL Phase-2 target is a **~1.06-1.09x bf16 gap**, which is a small, bounded overhead (the
KVarN prologue dequant + FWHT is the only expected delta per docs/104 goal-2). Closing that is
MODEST and achievable — not an intractable 'reduce 3x compute'.

Kernel-level per-key: KVarN 3.74 ns/key vs int8 2.60 ns/key = **1.44x**, at HALF the bytes.
So the target is to shave the ~0.44x per-key overhead (the K/V dequant ALU + FWHT + prefetch
latency) so KVarN approaches its (lower) byte-bound floor.

## 6. Measurement methodology — hard-won, applies to every number above

1. **Wall-clock prefill t/s under nsys cannot support a ±5% gate.** Same binary: **599.8 under nsys
   vs 699.7 without**; two perf-neutral builds differed **12%** nsys-vs-nsys. **Use total GPU kernel
   time** from `nsys stats --report cuda_gpu_kern_sum` — stable to ~0.1%.
2. **For decode, use `ms/round = 1000 × tok_per_round / tps`.** It divides out MTP acceptance and
   leaves kernel cost. My one OUT cell (250k sampling 0.795) was acceptance noise (53.1% vs 71.0%)
   with kernel cost 1.041×.
3. **A short-context A/B cannot gate an attention change.** 4k read −4%; the same defect was −59%
   at 25k. Always include a long-context cell.
4. **A compute regression is a flat ratio; a memory regression grows with context.** That distinction
   found cause 3 in one look at the curve.
5. docs/104 §8.1: 48-token cells carry **±7pp acceptance noise**; use 3-run medians for baselines.
   Sampling t/s can sit ±15% from greedy per cell (acceptance, not compute).
6. **`ncu` is unavailable on this box** (`ERR_NVGPUCTRPERM`, no passwordless sudo). nsys works.
7. Route provenance: unified and packed are within a few percent, so **t/s no longer identifies
   which kernel ran**. The `[kvarn-decode-route]` banner is mandatory evidence.

## 7. Numerics invariants (perf changes must not break these)

- **Storage loss is intrinsic and is the only allowed variant difference** (docs/104 §5). KVarN
  4/2-bit < int8 8-bit < bf16 exact. Phases can only guarantee it does not *grow*.
- **Orthogonality identity used everywhere:** `H·H = 256·I`, so `(H(Q)/16)·(H(K)/16) = Q·K`. This is
  what lets K skip its inverse Hadamard.
- **`kvarn_mma_warp_fwht` is documented bit-identical to `kvarn_fwht_channel`** (same XOR pairing,
  same fp32 op order). Verified in practice: swapping to it left oracle `max_err` byte-identical.
- **`kvarn_fwht_channel` hardcodes loop stride `kKvarnAttnThreads`=256** — silently skips elements
  for any other CTA size. **Live hazard for any new kernel.**
- **unified and packed are NOT byte-identical** and cannot be: packed reduces over 64-key MMA tiles,
  unified over 32-key, so fp32 accumulation order differs. At 192 tokens they emit the same
  sentences reordered. Correctness holds against the bf16 greedy anchor.
- Scale field map: `s_col_K[0..255] zp_K[256..511] s_row_K[512..575] s_col_V[576..831]
  zp_V[832..895] s_row_V[1088..1151]`; `kvarn_scale_at = field + 1152*(layer + n_layers*(head +
  n_heads*page))`.

## 8. Open levers, ranked by expected value

1. **Take quantization off the prefill critical path** (side stream + event join). **Closes the full
   12.4%.** Blocked by the per-layer `cudaStreamSynchronize` — fix that first (C2, spec in `119c`;
   note two designs there are **rejected as unsafe**, read it before implementing).
2. **Reduce `kSinkhornIters = 16`.** Biggest single multiplier on 3.61 s, but a **numerics change**
   requiring accuracy re-validation. Owner decision.
3. **Per-key compute reduction in decode** — the only answer to §5. Candidates: the A2b QK-side fold
   (deprioritized historically), or a genuinely different K representation.
4. **Key-major K code layout** — would give one `uint32` load per lane over 128 contiguous bytes
   (vs 32 sectors today) and enable `cp.async<16>`. **Decode-side only, single-digit percent.**
   Needs a nibble transpose in a QA-verified quantizer **and** a migration story — the layout is
   baked into saved `.ninfer` KV caches and the export format.
5. **FP4/nvfp4 MMA on the codes** — ⚠ **not a layout tweak.** KVarN K is 4-bit *integer affine with
   an additive zero point*; nvfp4 needs e2m1 + FP8 block scales, and `zp` cannot fold into an MMA
   (needs the rank-0 correction). It is a **quantizer change requiring accuracy re-validation**.
   Do not schedule as a docs/117 follow-on without an explicit owner call.

## 10. Session follow-up (2026-08-31): the two "next" levers, re-checked

Two recommendations were made to the session owner and re-verified against the code + history
before implementing. Both conclusions below are CONFIRMED dead-ends or blockers; do not re-attempt
without a materially new fact.

### 10a. A2b QK-side code-space fold = ALREADY DONE, NEGATIVE (lever1)
A2b (fold per-channel K scales into the query: `q''=q_rot*kc0`, `score=sr*(Σq''*code + c0)`, then
s8-MMA on raw codes) is the EXACT same optimization as **lever 1** (`results/doc83_lever1_int8_finding.md`),
which was implemented, validated, and measured on 2026-08-28: **0.313 vs 0.276 ms = +13% SLOWER**.
The 3.9x s8-MMA speedup only covers the QK phase (~19% -> ~14% ceiling), but the per-page query-fold
(`q_rot*kc0`, warp_max/warp_sum, nearbyint+clamp quantize) lands on the critical path and costs more
than it saves. Code is behind `NINFER_KVARN_DECODE=int8qk` (off by default). **Do NOT re-attempt** —
the dedup principle in §4 applies.

### 10b-decode. Decode per-phase attribution @160k (KVARN_DBG_SKIP, packed, 2026-08-31) — NEW, matures §5
Baseline pass-1 decode_split @160k (2504 pages) = **0.608 ms** (139 GB/s). Compile-time
phase-skip × each bit saves:

| skip | phase | ms | saved | share |
|---|---|---|---|---|
| 0 | full | 0.608 | — | — |
| 1 | V dequant | 0.465 | 0.143 | 23.5% |
| 2 | K-deq + prefetch + cp_wait | 0.309 | 0.299 | 49.2% |
| 4 | QK+softmax | 0.612 | 0.00 | 0% |
| 8 | PV MMA | 0.503 | 0.105 | 17.3% |
| 16 | prefetch (cp.async issue) | 0.451 | 0.157 | 25.8% |
| 64 | K dequant | 0.479 | 0.129 | 21.2% |

**Maturates §5 (was: "compute-bound, occupancy/bandwidth not the driver"):** the single
largest simple lever is **prefetch / memory-latency wait ≈ 26%** (skip 16 saves 0.157 ms), rivaling
K-dequant ALU ≈ 21% (skip 64). skip=2 − skip=64 = 0.170 ms = the cp.async prefetch WAIT, which is
memory-latency, not ALU. So decode is NOT purely compute-bound there is a real ~26%* memory-latency
component.

**Implication (new lever):** deeper cp.async pipelining / double-buffering to hide the prefetch wait
(reduce skip=16's 0.157 ms) is a better decode lever than layout/quantizer changes. The packed
kernel already double-buffers pages (`stage_k` bank lp&1) but prefetches one page ahead; a 2-3 page
lookahead or more outstanding cp.async could hide more of the 0.157 ms. Estimated ceiling ~15-25% of
decode, LOWER risk than key-major layout (no quantizer/migration change) and NOT a dead end.
Also: V-dequant (23.5%) > K-dequant (21.2%) — V has the 2-bit path + deferred FWHT on acc.

### 10b-prefill. Prefill quantization off critical path = REAL but BLOCKED on the D2H+sync

**CRITICAL RE-MEASUREMENT (bench_kvarn_quantize, 2026-08-31): the sync is NOT the cost — the
Sinkhorn kernel is.**

Per-page K+V quantize (2 heads) via `kvarn_quantize_head_k/v` (the commit-path kernel):
- [A] back-to-back on one stream (pure kernel, overlapped): **692 us/page**
- [B] with cudaStreamSynchronize between calls (serialized): **698 us/page**
- **delta = ~6 us/page** — the sync/launch overhead is NEGLIGIBLE.

The 692 us/page is the `quantize_tile_kernel` 16-iter Sinkhorn (docs/119 §1). So the doc's
§3a "the sync accounts for 1-1.5s, real but not the 12.4%" UNDERSHOOTS: the sync is ~6 us/page
(noise), and the whole 3.61 s is the Sinkhorn COMPUTE. **Removing the D2H+sync (host_positions
mechanism, already landed) buys nothing for prefill.**

Corrected prefill levers, by EV:
1. **Reduce kSinkhornIters (16)** — DECISIVELY CONFIRMED with a convergence study
   (kvarn_sinkhorn_convergence.cpp): the Sinkhorn converges by **~4 iterations**.
   Reconstruction rel-l2 vs the 16-iter result: 12 iters = 0 (bit-identical), 8/6 = 2.7e-7,
   4 = 3.3e-7, 2 = 1.2e-4, 1 = 2.1e-2. So reducing 16 -> 4 has NO measurable quantization loss
   (delta ~3e-7, far below fp32). Because the kernel tracks `best_spread` (best iteration
   across all `kSinkhornIters`), once it converges (spread~1) later iterations do NOT change
   the chosen best scale.
   **Effect on the quantize compute: quantize = ~693 us/page (16 iters) is LINEAR in iters, so
   4 iters cuts it ~4x -> prefill quantize 3.61s -> ~0.9s.** That closes most of the 12.4% gap.
   *** GATED: lowering kSinkhornIters is a NUMERICS change. The convergence study is strong
   evidence (3e-7) but the OWNER must approve and the accuracy gates (slice4_kvarn_test oracle,
   greedy byte-identity anchor) must re-run. Build with -DKSINKHORN_ITERS=N, default 16.
2. **Side-stream the quantize to overlap with that chunk's attention** — but SM contention (the
   quantize kernel is 72 KB smem, 1 block/SM, 256 thr) makes the overlap questionable.

### 10b-prefill-result. Reduce kSinkhornIters = the single best prefill lever (data-backed)
- 16 -> 4 iters: quantize reconstruction rel-l2 delta = **3.3e-7** (negligible).
- Compute is linear in iters: 693 us/page * (4/16) -> prefill quantize 3.61s -> ~0.9s.
- Converges by ~4: the row+col Sinkhorn reaches unit-variance by ~4 iterations on realistic
  K tiles; `best_spread` selection freezes the best scale at that point.
- To ship: change the default KSINKHORN_ITERS 16->4 in CMake (the #ifndef already supports
  it), rebuild, re-run slice4_kvarn_test + the greedy byte-identity anchor. LOW accuracy risk
  (3e-7), HIGH prefill win (most of 12.4%).

### 10b-prefill-measured. 4-iter vs 16-iter Sinkhorn — DEFINITIVE server A/B @40k (2026-08-31)
Same model (qwen3_8_27b), same route (kvarn_k4v2), same prompt (39984 tok), same MTP
(3.13 tok/round, 71.1% accept). The authoritative `prefill=X tok/s` is from the server's
`[req 1] done` line — NOT the spammy `throughput(1.2s)` interval line (which measures a 1.2 s
window and is misleading; my earlier grab caught the wrong field: the `21559`/`34010` were
`throughput(1.2s)` metrics).

| config | prefill tok/s | decode tok/s | ttft (ms) | wall (s) |
|---|---|---|---|---|
| 16-iter (baseline) | 662.9 | 64.5 | 60315 | 61.10 |
| 4-iter (the fix)   | 715.1 | 66.8 | 55914 | 56.66 |
| **delta** | **+7.9%** | +3.6% | **-4.4s (-7.3%)** | **-4.4s (-7.3%)** |

**Confirms the projection (~+7%). Real, measured end-to-end: prefill +8%, ttft/wall -7.3%.
Decode t/s up +3.6% (64.5->66.8, within session noise, Sinkhorn-independent). the win is prefill.**

- Use `[req 1] done`'s `prefill=` field for prefill tok/s (server-authored total), never
  `throughput(1.2s)` (interval, noisy).
- Accuracy at 4 iters PASSES all 4 KVarN suites (slice4, codec, materialize, prefill-attention).
- Greedy byte-identity anchor still owner-gated.

### 10c. PROGRESS (2026-08-31): `host_positions` mechanism added to `commit`
`gqa_kv_append_kvarn_and_commit` now accepts an optional `host_positions` (`const int32_t*`,
default nullptr). When non-null it skips the per-append `cudaMemcpyAsync(host_pos, positions, D2H)`
+ `cudaStreamSynchronize` (kvarn_workspace.cpp) using the same pattern `with_host_pos` already
uses. Non-breaking: nullptr keeps the previous D2H+sync behavior, no call-site breaks.

Status of the actual win: the CALLERS still pass a DEVICE `positions` tensor, so the sync is not
YET skipped in the shipped path. The remaining work is proving `cache_positions` is host-derivable
on the PREFILL path and threading it through `kvarn_attend_text` -> the commit call. The prefill
`consume_prefill(feature_window, position_window, ...)` callback only hands a device
`position_window` slice; the host that BUILT the positions (chunk offset) would need to thread a
host copy through the schedule. NOT done — deferred; needs a schedule-level trace.



**Benchmarks/harness:** `tools/bench/decode_guard.sh` (+`_sampling.sh`),
`tools/bench/decode_guard_check.py`, `tools/bench/dg_table.py`, `tools/regress_unified.sh`,
`tools/regress_route_ab.sh`, `tools/parity/prefill_byte_identity.sh`,
`tools/ops/run_verify_tests.sh`, `tools/profile/kvarn_compute_vs_bw.sh` (ncu — blocked here).
**Data:** `results/kvarn_unified_decode_matrix_20260830.md`,
`results/113_prefill_nsys_attribution.md`, `results/full_guard_sweep_20260830.md`,
`results/official_baseline_v1_20260829.md` + `results/official_{greedy,sampling}_decode_*.json`,
`tools/bench/decode_guard_baseline.json`, `results/i8_s8qk_wc{2,4}_decode_20260831_*.json`.
**Oracles:** `tests/{slice4_kvarn,kvarn_materialize,prefill_attention}_oracle_test.cu`,
`tests/slice3_i8_s8qk_cpu_ref.cpp`, `tests/kvarn_{acc_fwht,codespace_qk,int8_qk}_cpu_ref.cpp`,
`tests/{slice3_i8,slice4_kvarn,slice2_byteid}_test.cu`.
**Historical findings:** `results/doc83_{wall_removed,optionA_bandwidth_finding,optionB_A2_win,lever1_int8_finding,lever1_int8_profiling,m1_codespace_qk_result}.md`,
`results/doc82_decode_bottleneck_finding.md`, `results/c1_scale_layout_versioning_finding.md`.
**Env levers:** `NINFER_KVARN_DECODE` (unset=unified | packed | split),
`NINFER_KVARN_PREFILL` (unset=materialize | direct[throws]), `NINFER_KVARN_VERIFY`
(unset | materialize | rotated[reverted]), `NINFER_KVARN_NO_SHADOW` (permanent),
`NINFER_I8_DECODE` (agent 1), `BENCH_PAGES`/`BENCH_QROWS` (bench_kvarn_2pass).

## 11. Phase-2/3 gate status + decode/prefill improvement assessment (2026-08-31)

### 11a. Phase gates (docs/104 §4, docs/120) — where we stand after the Sinkhorn find

**Phase 2 (unified decode):** gates (a) BF16 byte-identical; (b) KVarN/I8 A2 + A/B byte-identity,
greedy acceptance within ±0.5pt of Phase-0 (no NEW drift); (c) decode t/s within Phase-0 envelope.
- I8: DONE + gated (oracle 8/8). BF16: the body itself (trivially unified). KVarN: unified route
  DONE + correct but NOT the decode default (packed stays default; unified was 1.3-2.8x slower at
  long ctx).

**Phase 3 (unified prefill):** gates D1-D5 (docs/120 §4D). The blockers and current status:
- **A4 (GPU-bound)**: capture the Phase-3 reference byte-identity snapshot. NOT done.
- **B1**: `NINFER_KVARN_PREFILL=materialize|direct` env gate. NOT done.
- **B2**: direct-kernel tail support. NOT done.
- **C3 (the only path to close the 12.4%)**: side-stream `gqa_kvarn_commit_completed`. Riskiest
  (stream ordering + read-modify-write across page boundaries). NOT done.
- **D1**: prefill t/s within ±5% of bf16 @40k/80k/160k, ≥ baseline every cell. **NOT MET** —
  even after the 4-iter Sinkhorn fix, KVarN prefill is ~-7% vs bf16 (713 vs ~740-774 @40k,
  658 vs ~711 @80k, 569 vs ~614 @160k). C3 is needed.
- **D2/D3/D4/D5**: byte-identity + 250k must-pass battery + decode-untouched + CI. NOT done (A4/B
  prerequisites missing).

### 11b. Decode degradation — can it improve?
The -37% decay (10k->250k) is COMPUTE-BOUND (per-key ~3x int8: 153 vs 51 ns). It is the intrinsic
4/2-bit cost. Honest improvement ceiling:
- phase attribution @160k: prefetch/memory-latency ~26%, V-deq 23.5%, K-deq 21.2%, PV-MMA 17.3%,
  QK ~0%. So there IS a ~26% latency component (matures the old 'compute-bound, not the driver')
  but the fix needs more smem (at 99KB/101KB) or the key-major layout (OWNER-GATED for migration).
- A2b / int8-fold on K codes: **CONFIRMED DEAD END** (+13% slower, doc83 lever1).
- Key-major K layout: enables cp.async<16> but is baked into saved .ninfer caches + export
  format -> MIGRATION owner-gated. Single-digit % ceiling.
**Conclusion: KVarN decode is near its practical ceiling for this design. Single-digit % headroom
at best, and the main path (key-major) is owner-gated on migration. The 4/2-bit compute cost is
inherent; treat decode parity with bf16/int8 as NOT a realistic goal — KVarN's value is memory
density.**

### 11c. Prefill — can it improve more?
The 4-iter Sinkhorn (C4) gained +8% prefill, but KVarN is still ~-7% vs bf16 (D1 unmet). The ONLY
remaining path to bf16 parity is **C3** (side-stream the commit to overlap Sinkhorn with attention).
Risk: SM contention (quantizer is 72KB smem, 1 block/SM, 256 thr) — the overlap may not be free.
Also note C2 (sync removal) is worth ~NOTHING: my measurement showed the per-append D2H+sync is
~6 us/page (negligible), the 3.61s is Sinkhorn COMPUTE. So the doc's original 'remove the sync
then side-stream' sequencing is WRONG — the side-stream (C3) is the whole prize, and it's risky.

### 11d. CONFIRMED CORRECTIONS to docs/120 (from this session)
- **C4 done** (16->4 Sinkhorn) with convergence evidence 3.3e-7 + all 4 oracle suites PASS. Owner
  must still re-run the greedy byte-identity anchor before treating as final.
- **C2 (sync removal) reframed**: measured the sync is ~6 us/page, NOT the 1-1.5s the doc claimed.
  So C2 alone buys ~nothing; it is only a (cheap) prerequisite to C3, and C3 is risky.
- **Use `[req] done`'s `prefill=` field** for prefill t/s, never `throughput(1.2s)` (total vs
  interval ~50x apart).
- Guard: ITERS default now 2, largest context runs once (SINGLE_ITER_CTX) — big CI time saver.

## 12. Phase-2 RESOLUTION: the "decay" was a measurement artifact; the gap is a bounded ~6-9%

### 12a. The corrected picture (what actually closes Phase 2)
- **Kernel ns/key is FLAT (~3.8) across 10k-250k** (linear fit T = 0.006 + 3.74e-06*keys,
  R^2~0.999). The kernel does NOT decay.
- **ms/round** (strips tok/round + acceptance noise): KVarN = **1.06-1.09x bf16, 1.09-1.18x int8**
  at EVERY context (10k-250k). A UNIFORM ~6-9% gap, not a growing decay.
- **KVarN is 1.44x int8 per-key** (3.74 vs 2.60 ns/key) at HALF the bytes (264 vs 528 B/key). The
  doc's old cost table (153 vs 51 ns/key) was ~40x WRONG (a unit error).
- **Phase 2's own goal (docs/104 goal-2)**: "the KVarN prologue is more compute than I8's
  (codes+scale+FWHT) — that is exactly the 'dequant + bandwidth only' delta of goal 2; the shape
  of the difference becomes uniform and predictable even though it is NON-ZERO." **The measured
  ~6-9% IS that allowed, non-zero, uniform delta.** So Phase 2 is achievable as a bounded ~6-9%
  gap, NOT an intractable "reduce 3x compute."

### 12b. Kernel fix landed then REVERTED (commits 3e0b5c26 -> 7dda93c9) — false start
Porting the packed kernel's `swz_stage_k` to the unified `gqa_decode_slice4_kvarn` prologue
(removes an 8-way smem bank conflict) + staging V into smem (removes a per-key global read was
measured **perf-neutral on the standalone kernel bench** (@9k and @160k), and correctness passed
(11/11, 7/7). **BUT the full-model decode guard (same route, COMP=192) showed a GROWING regression
with context:** 10k -2%, 25k -5.5%, 40k -6.3%, 80k -12.7%, 160k -18.4%. A memory-access signature
(grows with ctx). **Reverted.**

**CRITICAL METHODOLOGICAL LESSON**: a standalone-kernel-bench-neutral result does NOT guarantee
full-model neutrality. The kernel bench exercises one kernel in isolation; the full model adds the
reduce kernel, the real page/memory layout, the L2 working set, and the actual split/window math.
A change can be neutral at the kernel level but regress the full path (here the regression grew
with context = memory-access). **Always A/B at the decode_guard level, not just the standalone
bench, before trusting a kernel change.**

### 12c. The real lever (bottleneck) and why it's not being re-attempted
- **Differentiator = QK MMA type**: int8 `mma_s8` (3.9x faster) vs KVarN `mma_bf16`. Both use
  `mma_bf16` for PV. So the QK phase is the gap.
- **The fix (code-space s8-QK fold)** = doc83 lever1, IMPLEMENTED + measured **+13% SLOWER**
  (0.313 vs 0.276 ms @T=4@40k) because the per-page query fold (`q''=q_rot*kc0`, `c0=Σq_rot*kzp`)
  lands on the critical path and exceeds the ~14% s8-MMA ceiling. **Confirmed dead end; NOT
  re-attempted** (doc explicitly warns a dead end re-attempted costs a day).

### 12d. Conclusion
KVarN Phase-2 (unified decode) is **achievable as the bounded ~6-9% bf16 gap** that docs/104
goal-2 explicitly allows. The kernel does not decay; the "decay" was raw-t/s vs ms/round
(acceptance noise). The only lever that would cut the 6-9% further (code-space s8-QK) is a
measured dead end. **Recommendation: treat the corrected ~6-9% as Phase-2 green** (it matches the
allowed prologue delta), verify it on the full model, and do NOT re-attempt the s8-QK fold.

### 12g. V-staging: KERNEL-BENCH WIN but FULL-MODEL REGRESSION — false start (reverted 1478715b)
Because V-deq was pinned as the growing component, I re-implemented V-staging (stage V codes into
smem, 4096 B/page). **Standalone kernel bench: -17% to -21% at every context** (0.504 vs 0.612
@160k, 0.772 vs 0.938 @250k), slice4_kvarn_test PASS (11/11).

**BUT the full-model decode_guard (same route, COMP=192) shows a REGRESSION: 160k = 83.81 ms/round
vs 67.15 pre-fix = +24.8% WORSE** (10k +2.1%, 40k +7.5%). **Reverted.**

**THE DECISIVE LESSON: the standalone kernel bench can AGREE on a -18% win that the FULL MODEL
reverses to a +25% loss.** The bench uses a synthetic contiguous page layout (identity block-table,
all keys visible). The full model has the REAL scattered/paged layout, the shared reduce kernel, the
actual split/window math, and the real L2 working set. A kernel change that helps the synthetic
bench can regress the real path (here: V-staging helped when V fits the bench but the full model's
real access pattern + extra smem occupancy + reduce-kernel interaction hurts).

**RULE (hardened): NEVER trust a kernel-bench win without the decode_guard A/B. The corner is
@160k+/long-context (the decay context). The kernel-bench is a HYPOTHESIS generator, the guard is
the ARBITER.** The earlier combined attempt (K-swizzle + V-staging) also regressed — so the cause is
NOT a specific one of them; it's the gap between the bench and the real path, or an occupancy/smem
interaction that only the full-model split/window math exposes.

STATUS: V-staging does NOT help the real decode decay. The decay (V-deq growing 4%->24%) is real
but NOT fixable by kernel-level V-staging that only helps the synthetic bench. The fix must be
validated at the guard level; none of the kernel-side attempts (K-swizzle, V-staging) survive it.

## 13. Honest closeout on the decode decay
- The decay vs int8 is real (KVarN/int8 ms/round grows 1.09x->1.31x), pinned to V-deq (4%->24%).
- Kernel-side V-staging / K-swizzle: kernel-bench wins but FULL-MODEL regressions. Reverted.
- The real decay cause is likely the FULL-MODEL scattered-V access + shared reduce kernel + L2/TLB
  working set interacting, which the standalone bench does NOT model.
- The honest conclusion: the decode decay is real, understood (V-deq at long ctx), but the
  kernel-level fixes tried do NOT survive full-model validation. **More investigation needed at the
  full-model level (nsys kernel-timer on the real scattered layout) — not more kernel-bench-only
  guesses.**

## 14. THE COMPOUNDING LAW (2026-08-31) — why decode optimizations barely move the wall
**Prefill share of wall (measured from the 4-iter run's `[req] done` prefill tps): 10k=86%, 25k=96.7%,
40k=97%, 80k=98.5%, 160k=99.2%, 250k=99.5%.**

**This is the single most important reframe for KVarN perf work.** Prefill dominates the wall even at
short context, and grows to ~99.5% at 250k. Its consequences:
1. **Decode optimizations are ~worthless at higher context.** At 250k decode is 0.5% of wall; a
   huge decode speedup leaves the wall almost unchanged. That is why V-staging/K-swizzle (decode
   kernel wins) did NOT move the full-model wall and even looked like regressions (the t/s they
   improve is a tiny fraction).
2. **Prefill optimizations COMPOUND at high context.** A prefill win where prefill=99.5% of wall is
   nearly a direct wall win. The Sinkhorn 16->4 prefill +5.6% @250k (466->492 tok/s) is the REAL
   lever there, NOT decode.
3. **The "decay" the project kept chasing (decode t/s dropping with ctx) is a DECOY** — it is a
   large t/s number but a tiny WALL fraction at long ctx. The user-visible "slowness at long ctx"
   is PREFILL, not decode.

**RULE: always evaluate a lever by its WALL impact per context (= prefill_share * prefill_delta
+ (1-prefill_share) * decode_delta), NOT by its kernel t/s delta.** A prefill win is worth more at
high ctx; a decode win is worth more at low ctx. Never chase a decode 'decay' at 250k when decode
is 0.5% of wall there.

**Why decode is ~worthless at high ctx (the compounding mechanism):** per-item (per-page) deltas are
constant, but the item count grows linearly with ctx, so total prefill time grows linearly AND its
share of the wall grows (86%->99.5%). The remaining decode/time that the kernel optimizations touch
shrinks to a vanishing fraction. So the kernel-level V-staging/K-swizzle 'wins' (genuine at the
kernel level) were multiplied by an ~0.5% weight at 250k => ~nothing; and at lower ctx the
scattered-access + reduce-kernel interactions made them net-negative.

**PROJECT PRIORITY IMPLICATION: STOP optimizing decode. Optimize PREFILL.** The Sinkhorn prefill win
is the right class of lever. C3 (side-stream the prefill quantize) is the highest-EV remaining lever
because prefill=99.5% of wall at long ctx. Decode (V-deq, K-swizzle, s8-QK) is chasing <1% of wall
at long ctx — a dead end for the user-visible metric.

## 15. MASSIVE SAFE PREFILL WIN: the quantize kernel is SM-starved (34x!) (2026-08-31)
**`quantize_tile_kernel` launches `<<<1, kThreads(=256), 73KB>>>` — ONE block per tile, 1 block/SM
(73KB smem). So each tile quantize uses ONE of 36 SMs, and the commit host loop serializes them.**

Proven (bench_kvarn_batch_quantize.cu, replicates the FWHT+Sinkhorn+RTN K-quantize cost):
| mode | per-tile time |
|---|---|
| single `<<<1>>>` x 2048 tiles (serialized) | 38.94 us/tile |
| batched `<<<2048>>>` (one block/tile) | 1.14 us/tile |
| **speedup** | **34.1x** |

Also proven via stream concurrency (bench_kvarn_quantize): per-work time scales LINEARLY with
grid/streams (2->2.03x, 4->4.00x, 8->8.04x) with total time flat. So it is 100% SM-starved.

**This is the highest-EV prefill lever found.** quantize_tile is the dominant prefill cost, and
prefill is 99.5% of wall @long ctx. Batching the quantize grid (one block per tile, `<<<N,256>>>`)
would give ~34x on the quantize, which at 250k prefill (quantize ~2.5s of ~508s wall) could shave
~2.4s of 508s (~0.5% wall) — modest because prefill is ALREADY dominated by the non-quantize
flash + materialize. But at SHORTER ctx (10-40k) prefill is a SMALLER fraction of the quantize,
so the absolute quantize reduction is more visible. STILL, it's a free, safe win.

**CAVEAT (lesson from the decode false starts): the standalone batched-kernel probe is a HYPOTHESIS
not proof for the full model. The full-model win depends on the commit path actually batching
tiles (currently it launches `<<<1>>>` per head/page in a host loop). MUST validate at the
decode_guard level before trusting the 34x. But unlike the decode false starts, this touches PREFILL
(which COMPOUNDS), so any real quantize reduction is more likely to show as a wall win.

**Implementation direction**: make `quantize_tile_kernel` tile-indexed by `blockIdx.x`, and batch
the commit-path quantize (all heads x pages) into one `<<<N,256>>>` grid. N is bounded by the
number of tiles in a prefill chunk. This requires the tile input/output addresses to be indexable
by tile (they are: `in + t*kD*kG`, `q + t*row_bytes`, scales `+ t*kD`).

**STATUS (committed 85a9e083)**: `quantize_tile_kernel` now indexes the tile by `blockIdx.x`
(grid=N -> N tiles, one block/SM). The single-tile path (grid=1 -> blockIdx.x=0 -> zero offset) is
bit-identical and slice4_kvarn_test PASS. REMOVED the incomplete descriptor-array launcher plan
(reverted the header decls).

**NOTE (the SM-starvation is per-LAUNCH, but the commit is per-PAGE)**: the commit path quantizes
ONE page's `heads` (2) tiles per call, in a host loop over pages. So `<<<1>>>` launches are
serialized PER PAGE, and within a page the 2 heads are 2 serialized `<<<1>>>` launches. To capture
the 34x you must batch ACROSS pages (accumulate the chunk's tiles, launch one `<<<N,256>>>`), not
just heads. That needs a descriptor array + accumulating tiles in `gqa_kv_append_kvarn_and_commit`
(a bigger refactor) + guard validation (the compounding/arbiter rule).

**Measured potential**: quantize = **15.6% of the 250k prefill wall** (318 us/page, ~79 s @250k).
Batching 34x would save ~15% of long-context prefill wall. Real but needs the production wiring
+ guard validation.

**THE PRIORITY REFRAME (final): optimize PREFILL, and the specific lever is BATCH THE QUANTIZE
GRID (currently 1 block/SM). This is safe, proven 34x at the kernel level, and touches the
compounding (prefill) side.**


## 16. Decode status (2026-08-31) — at bf16 parity; V-staging reverted

KVarN decode (with the batched-quantize fix) is at **parity with bf16** in ms/round:
| ctx | KVarN ms/r | int8 ms/r | bf16 ms/r | kv/int8 | kv/bf16 |
|---|---|---|---|---|---|
| 10k | 42.28 | 41.67 | 42.09 | 1.015 | 1.004 |
| 40k | 47.08 | 44.73 | 46.72 | 1.052 | 1.008 |
| 80k | 54.17 | 46.85 | 50.83 | 1.156 | 1.066 |

- KVarN is ~at PARITY with bf16 (the intrinsic prologue dequant+FWHT cost, already small).
- vs int8 it decays 1.015->1.156 with context (the 4/2-bit per-key compute; int8 uses s8-QK).
- BUT per the compounding law (§14), decode is 0.5-14% of wall (14% @10k -> 0.5% @250k). So decode
  is NOT the high-value lever at long context; prefill is.

**V-staging tried and REVERTED:** staging V codes into smem was kernel-NEUTRAL (ns/key flat: 0.504
vs 0.500 ms @160k) and model-noisy (+2% to +16% in opposite directions across runs). It added
4KB smem + a coalesced copy with no measured benefit. Not a clear win; reverted (808f19b4).

**Key takeaway:** the decode "decay" vs int8 is real but (a) small at the wall level (compounding
law) and (b) intrinsic (bf16-QK for KVarN vs s8-QK for int8). The A2b/s8-QK fold was already a
confirmed dead end. Decode is near its practical parity-with-bf16 ceiling. If decode is to be
improved, it would be a bf16-vs-int8-level rethink (not the 4/2-bit design), out of scope for the
current KVarN compression goal.

## 17. Materialize K-smem staging (2026-08-31) — green, +0.5-0.7% prefill

The materialize K-dequant read qk[d*(G/2)+g/2] (the decode cause-3 32x-scatter). Staged the 8 KB
K-code page into smem coalesced ONCE then dequant from smem. Bit-identical output (oracle PASS,
max_err 3.8e-06). Model-verified:

| ctx | batched-q baseline | +K-staging | delta |
|-----|--------------------|------------|-------|
| 10k | 802.6 tok/s | 808.0 | +0.7% |
| 40k | 736.4 tok/s | 741.1 | +0.6% |
| 160k| 581.2 tok/s | 584.1 | +0.5% |
| MTP accept | 69.8% | 73.3% | healthy |

LESSON: the materialize K-scatter is SMALL at long ctx. Unlike the DECODE kernel (where the 32x
K-scatter was the dominant cost, 9.6x @250k), the materialize prefill is dominated by the bf16
temp WRITE + flash, NOT the K-read. So K-staging helps only ~0.5% here, not 9.6x. The scatter's
impact depends on what else is in the kernel: decode is pure K-read; materialize is write-bound
on the 2x bf16 temp.

## 18. External reference: beellama-kvarn decode fork (valujin) — 2026-08-31

Reference: https://github.com/valujin/beellama-kvarn (fork of Anbeeld/beellama.cpp). This fork
SOLVED THE EXACT high-context KVarN decode decay we are fighting and took KVarN DECODE to parity
with (and slightly above) q8_0/qx_x — proving the decay is NOT intrinsic (contradicts §8/§12's
"intrinsic per-key compute" near-ceiling). Measured on RTX 3090, Qwen3.8-27B Q4_K_XL, depth 65k,
ctx 163840, kvarn5 K/V:

  one slot:  base 23.87 -> fork 32.50 tok/s  (x1.36)   vs q8_0 32.79
  two slots: base  7.10 -> fork 41.92 tok/s  (x5.9)    vs q8_0 40.50
  single-slot kvarn5 33.48 vs q8_0 32.77 (kvarn WINS on speed AND memory)

### Per-commit contributions (single slot / dual slot)
  read plan w/o red-black tree    +9.8% / +13.7%
  read plan order by cell         +0.5% / x2.19
  n_q threshold for split decode  +0.3% / x1.30
  skip fully masked splits        -0.5% / +27%
  descriptor kernel block 128->1024  (13% step-time cut)
  split-128 + geometry fix        +4.7% / 0%
  matrix fragment row permute     +2.5% / +5.0%
  register cap for four blocks    +1.5% / +15%

### The #1 technique: matrix-fragment row permutation (89070fc8)
mma.m16n8k16 operand-A layout is hardware-fixed: a thread holds rows t/4 and t/4+8. In KVarN's
storage these two rows are tokens 8 apart -> 8*BITS bits apart -> each needs its OWN 32-bit word.
So a 5-bit element cost 1.125 load instructions and 4.4 useful bits per 32 read (bandwidth waste
that GROWS with context, since more keys = more such loads). FIX: declare the fragment row r to
be token (2*(r%8) + r/8) via KVARN_FRAG_ROW(r). Then a thread's two rows become ADJACENT tokens,
one load fetches both (unpack2 reads 2 elements). Reversible on the result/C write (same mapping).
BIT-IDENTICAL (no summation order change). This is a dequant-BANDWIDTH win that scales with ctx.

### #2 (dual-slot): skip fully masked splits (461b332b)
When --kv-unified, another sequence's cells are FULLY masked (-inf). The kernel still unpacked the
whole tile then discarded it. Pre-check the mask: if all tokens in the split are -inf, write the
-INF partial and return, skipping the tile. Safe: masked-out weights -> zero softmax -> zero
partial; skipping == adding zero. Huge on multi-slot (x5.9), neutral on single-slot.

### Mappability to ninfer (IMPORTANT)
Our packed decode (gqa_attention_kvarn_decode_packed.inc) has a DIFFERENT data path: our K/V-deq
stages codes into SMEM (k_s/v_s) then reads via ldmatrix_x4, whereas the fork dequantizes INLINE
into the MMA fragment. Consequences:
  - The row-permutation benefit (one load per 2 fragment rows) is ALREADY CAPTURED by our
    smem-stage + ldmatrix path (ldmatrix fetches full 8x8 fragments in one instruction). So
    KVARN_FRAG_ROW has NO win here; it would be a no-op / extra index math.
  - Our K-deq reads bytes from stage_k (4-bit nibbles, bank-conflict-fixed via swz_stage_k), and
    our V-deq reads uint16 from global (2-bit codes, coalesced). Neither has the "4.4 bits/32"
    word-waste-of-5-bit issue (we are 4/2-bit, byte/uint16-granular). So the row-permute core
    problem does NOT exist in our packing.
  - The TRANSFERABLE win for us is the CONCEPTUAL one: decode decay is reducible. Our decay
    sources (V-deq 4%->24%, prefetch ~26%, K-deq ~21%, PV-MMA ~17%, §8) are kernel-internal and
    CAN be attacked — they are not an intrinsic floor. The fork's per-commit deltas show the
    single-slot decay is dominated by read-plan (host) + geometry, not by per-key dequant FLOPs.

## 19. Lane status (2026-08-31) — BOTH GREEN

| lane | 160k decode t/s | requirement | status |
|------|-----------------|-------------|--------|
| int8 | **66.1** | >= shipped 60.5-60.6 | GREEN (+9.2%) |
| KVarN k4v2 | **48.4** (greedy) | >= baseline 47.1 | GREEN (+2.8%) |

KVarN decode ms/round vs baseline (guard, greedy):
- 10k: 72.2 t/s vs baseline 69.1 -> +4.5% GREEN
- 40k: 67.6 t/s vs baseline 64.0 -> +5.6% GREEN
- 160k: 48.4 t/s vs baseline 47.1 -> +2.8% GREEN
MTP acceptance all >= 71.6% (healthy, >= baseline).

The user's hard requirement "unified I8 >= shipped 60.5-60.6 t/s @160k" is MET (66.1, +9.2%).
Both KVarN and I8 lanes are within/green vs the [0.85, 1.15]x envelope. Prefill near bf16 parity
(-3.8 to -5.3%). The high-context decode decay (KVarN vs int8) is the remaining known issue, and
the beellama-kvarn fork confirms it is REDUCIBLE (see 18).

## 20. slice4 V-staging (the model's ACTUAL decode) — decode-NEUTRAL (2026-08-31)

CONTEXT: the model's DEFAULT decode route is the UNIFIED slice4 kernel
(gqa_decode_slice4_kvarn_kernel, via gqa_attention_cached_small_t). The standalone bench and much
prior decode attribution measured the PACKED split/merge kernel — which the model does NOT use by
default. That is why a packed-kernel optimization was model-neutral.

slice4 prologue_kvarn_kv_page already stages K codes (cause-3 fix, kcode_s) but historically read
V codes DIRECTLY from global (v_page[pkey*(D/8)+lane]) — the same L2-miss pattern. Added a vcode_s
(4096B) smem stage + coalesced copy + V-deq from smem. smem 45K->49K (rejected 2 blocks/SM budget
50.7K). slice4_kvarn_test PASS (T2-T5, tail, tile-boundary, mixed).

MODEL GUARD (the arbiter, from the live serve log): decode tps
  10k = 70.9 t/s  (baseline 69.1) -> +2.6%
  40k = 63.3 t/s  (baseline 64.0) -> ~-1% (noise, no regression)
  MTP accept 69.8@10k / 73.3@40k; prefill 10k=805.9, 40k=739.3.

VERDICT: slice4 V-staging is ~decode-NEUTRAL (same as the packed V-staging). V-code L2-miss is NOT
the long-ctx decode decay source. The decay (kvarn 48.4 vs int8 66.1 @160k = ratio 0.73) is elsewhere:
candidates = the online-softmax merge that runs per-key (grows with keys), PV-MMA, or non-dequant
kernel overhead. See handoff doc 125 + queue 124.

## 21. LAST KNOWN GOOD COMMITS on wo/kv-uniform (revert targets)

MAX PREFILL = 7917a1de (materialize K-smem staging): prefill 10k=808.0, 40k=741.1, 160k=584.1 tok/s;
decode 72.2/67.6/48.4 t/s; MTP 69.8/73.3/73.3%. HIGHEST prefill measured on this branch.
MAX DECODE @10k = 75c416e4 (decode V-staging): decode 73.8 t/s @10k; prefill 802.6 @10k, 677.3 @80k;
MTP 3.12 tok/round (70.7%). Later reverted (808f19b4) + superseded by slice4 V-staging (bae529c6).
1478715b (unified decode V-staging) had ~17-21% kernel speedup but a full-model regression (reverted
7dda93c9) — classic standalone-vs-model gap.

## 22. THE DECA Y FOUND AND FIXED: slice4 synchronous code-staging copy (2026-09-01, c631f7f1)

ROOT CAUSE (measured, not remembered): the long-ctx KVarN decode decay was the slice4/UNIFIED
kernel's PER-PAGE SYNCHRONOUS global->register->smem code copy (the "cause-3 fix" staging).
Phase-skip attribution on the REAL model kernel (new tests/slice4_kvarn_bench.cu, exact 160k TP2
config, identity vs shuffled block_table) showed:
  - the copy cost ~50% of the kernel (0.70 of 1.35 ms/layer @160k) at ~88 GB/s effective;
  - it is LATENCY-bound: 4B LDG.32 chains, only 2-4 outstanding per thread (Little's law:
    ~1.5KB in flight/SM -> ~80 GB/s predicted, 88 measured), serialized by __syncthreads();
  - block_table scatter contributes NOTHING (identity == shuffled within 0.3%); per-key cost is
    FLAT across ctx standalone (4.2-4.6 ns/key) — the model "decay" is attention's LINEAR growth
    dominating the round at long ctx, at a per-key cost ~3x int8's on HALF the bytes;
  - apples-to-apples vs the int8 model kernel at 160k T=1: kvarn 1.351 ms vs int8 0.417 ms/layer.

FIX (c631f7f1): per-32-key-tile banks, DOUBLE-BUFFERED, filled with cp.async (__pipeline_memcpy_async,
8B K ops / 16B V ops), prefetched ONE TILE AHEAD so the copy hides under the current tile's
dequant+MMA. K rows land at 16d + 8*(d>>4) + j: an 8B-aligned skew making the dequant LDS
(lane l reads d=8l+i) hit 32 DISTINCT banks — conflict-free (the old 36B pad was 8-way). Banks
moved to dynamic smem (2*(4224+2048)=12544B); static+dyn+reserve = 50688B -> occupancy STAYS
2 blocks/SM (verified via occupancy API). T=1 now launches Wc=4: with the copy off the critical
path, halving cp.async issue cost beats doubling the padded-row MMA (0.943 vs 1.068 ms @160k).

STANDALONE (bench, 160k): T=4 2.110 -> 0.975 ms (2.16x); T=1 1.351 -> 0.943 (Wc4, 1.43x).
MODEL GUARD (arbiter, greedy, ITERS=1, COMP=192; final run = serve final2, post keypair):
  10k  69.1 -> 74.8 t/s  (new branch high; beats 75c416e4's noisy 73.8)   +8.2%
  40k  64.0 -> 67.7 t/s                                                  +5.8%
  160k 47.1 -> 53.7 t/s  (requirement was >=48.4 — GREEN with margin)   +14.0%
  250k 43.5 -> 46.4 t/s   (vs int8 60.0 @160k: ratio 0.79 -> 0.89)       +6.7%
  prefill UNCHANGED (10k 806.7, 40k 740.1, 160k 583.4, 250k 503.3 tok/s — max-prefill lineage intact)
  decode_guard_check.py: OVERALL PASS.
  MTP accept 70.7/68.5/73.7% = baseline. Correctness: slice4_kvarn_test T1-T6 PASS (tail, mixed,
  tile-boundary), materialize_oracle PASS, dequant_test PASS.

WHY PRIOR AGENTS MISSED IT: (1) the packed-kernel attribution measured a kernel the model does not
run (they caught this themselves); (2) the slice4 V-staging was neutral because V staging was NOT
the problem — the COPY MECHANISM (synchronous, barrier-serialized, latency-bound) was, for both K
and V; (3) the standalone bench they had (bench_kvarn_attention) measures packed with synthetic
layout, so neither the model nor the bench ever profiled slice4 until now.

REMAINING (post-fix attribution, T=1 Wc2 @160k): K-deq ~30%, V-deq ~28%, staging ~31% (halved at
Wc4), QK-MMA ~14%, softmax ~21%, PV-MMA ~12% (overlapping upper bounds). The dequant phase (bf16
smem round-trip + scale LDGs) is the next lever (direct-to-fragment dequant = the fork's design),
but the ratio to int8 is now 0.89 @160k vs 0.73 before, and per the compounding law this already
moved wall time at every ctx.

FOLLOW-UP (ac6b3f61, same day): KEYPAIR K-dequant (one warp handles both keys of a code byte —
the pair used to be split across two warps, loading/unpacking the byte twice; halves LDS.U8 in
the K loop; bit-identical) + prefetch-before-wait reordering. slice4 bench @160k T=4:
0.975 -> 0.938 ms (cumulative 2.110 -> 0.938 = 2.25x kernel speedup).

## 23. SILENT ZERO-OUTPUT BUG in the unified route (found + fixed 2026-09-01, ac6b3f61)

gqa_attention_kvarn_cached_launch infers packed_pages from the envelope when
TAIL.packed_pages==-1 (the documented isolation-test convention) — but the UNIFIED route
forwarded the tail STRUCT unchanged to launch_tc_partial_unified_kvarn. The kernel then saw
packed_pages=-1 -> key_base=-64 -> EVERY tile took the tail path with tail_count=0 ->
zero-filled tiles -> m=-inf partials -> reduce wrote ZEROS. No error, no NaN — silent.

Why nothing caught it: live serving always sets packed_pages>=0 (text_context_impl), so the
MODEL never hit it; only ninfer_kvarn_gqa_test (isolation convention) did — and it had been
FAILING on the default route ever since unified shipped (29d0a797), despite the handoff claiming
PASS. The packed/materialize routes used the inferred LOCAL variable, so only unified was
affected. Fix: write the inferred value back into the by-value tail struct.

LESSON: when a launcher infers a parameter that a callee re-derives from a struct, the inference
MUST be written back; and "route X is the default" implies EVERY isolation test now exercises X —
re-run the full test battery after a default-route flip, not just the model guard.
