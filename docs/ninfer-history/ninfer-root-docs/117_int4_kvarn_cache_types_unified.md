# 117 — int4 + KVarN cache types (against the unified kernel)

**Status:** DESIGN — 2026-08-30. This is the design for the post-merge-kernel
work: implement + validate the full table of KV cache types (the non-KVarN int4
`q4_0/q4_0` + all the KVarN bit-width variants), **against the new unified
kernel** (docs/104) as per-TKV dequant PROLOGUES. GATED behind the merge kernel
(docs/104 Phase 2-3) being done (Agent 1's lane).

## 0. The constraint (user, 2026-08-30)
The int4 + KVarN cache types MUST be implemented **against the new unified
kernel** (docs/104), as per-TKV dequant PROLOGUES (like the existing Slice-3 I8
+ Slice-4 KVarN prologues), NOT as separate kernels. The canonical bf16 body
NEVER changes; only the prologue dequants K/V to bf16 smem. So the new types
(q4_0/q4_0, kvarn8/kvarn8, kvarn6/kvarn6, etc.) are new prologue variants (new
bit-widths + code byte sizes) in the unified kernel.

## 1. The full KV cache types table (user-provided, 2026-08-30)
```
K / V     Tail     Size     Size vs bf16     Median KLD     99.9% KLD     What it's for
bf16 / bf16     0     4096 MiB     100.0%     0     0.000050     Reference
q8_0 / q8_0     1024     2272 MiB     55.5%     0.000897     0.087699     Standard fidelity with a precision tail
kvarn8 / kvarn8     1024     2256 MiB     55.1%     0.000871     0.087639     Best measured quality below BF16
q8_0 / q8_0     0     2176 MiB     53.1%     0.000909     0.093029     Standard fidelity
q8_0 / q6_0     1024     2016 MiB     49.2%     0.000894     0.091098     q8_0 quality within noise, 256 MiB less
kvarn6 / kvarn6     1024     1744 MiB     42.6%     0.000879     0.084629     High-end value pick
kvarn6 / kvarn5     1024     1616 MiB     39.5%     0.000886     0.092778     Much cheaper, almost as good
kvarn5 / kvarn5     1024     1488 MiB     36.3%     0.000897     0.087666     Highest value in the mid-range
q5_0 / q4_1     1024     1440 MiB     35.2%     0.000966     0.089128     Standard option when VRAM-constrained
kvarn5 / kvarn4     1024     1360 MiB     33.2%     0.000936     0.089469     Balanced default
q4_0 / q4_0     1024     1248 MiB     30.5%     0.001057     0.104486     Compact standard cache
kvarn4 / kvarn4     1024     1232 MiB     30.1%     0.000994     0.090391     Cleaner than q4_0 for less memory
kvarn4 / kvarn3     1024     1104 MiB     27.0%     0.001112     0.113968     Smallest recommended tier
kvarn3 / kvarn3     1024     976 MiB     23.8%     0.001316     0.139558     When the context must fit
kvarn3 / kvarn2     1024     848 MiB     20.7%     0.002424     0.238780     Emergency compression
kvarn2 / kvarn2     1024     720 MiB     17.6%     0.003811     0.450496     Last resort
```

## 2. Current state (which types exist)
- The enum has ONLY 3 variants (include/ninfer/types.h:26-29):
  - BFloat16 (bf16 / bf16) — DONE.
  - Int8Group64 (q8_0 / q8_0, with + without the 1024 tail) — DONE.
  - KvarnK4V2 (kvarn4 / kvarn2 — K int4, V int2) — DONE (the Slice-4 KVarN
    prologue in the unified kernel).
- MISSING (to implement/validate, in priority order):
  - **q4_0 / q4_0** (non-KVarN int4) — the "regular int4 cache" (the user's
    #1 priority).
  - **kvarn4 / kvarn4** (the KVarN balanced default — the most-used KVarN type).
  - **kvarn5 / kvarn4** (the KVarN balanced default).
  - The rest (kvarn8/kvarn8, kvarn6/kvarn6, kvarn6/kvarn5, kvarn5/kvarn5,
    kvarn4/kvarn3, kvarn3/kvarn3, kvarn3/kvarn2, kvarn2/kvarn2, q8_0/q6_0,
    q5_0/q4_1).

## 3. The existing quantization code (what to build on)

### 3.1 The non-KVarN int8 (I8) — the model for the int4 prologue
- `gqa_attention_kv_quant.cuh` (the shared device helpers — no standalone
  quant/dequant kernel; the quant + dequant are FUSED into the GQA kernels):
  - `kGqaKvQuantHeadDim = 256` (the quantization head dim — 2× the 128 head dim).
  - `kGqaKvQuantGroup = 64` (the group size).
  - `kGqaKvQuantGroups = 256/64 = 4` (the number of groups).
  - `gqa_kv_quant_code(x, inv_scale)` — quantize to int8 (code = round(x *
    inv_scale), clamp to [-127, 127]).
  - `gqa_kv_dequant_i8x8_from(codes8, s)` — dequantize 8 consecutive int8 codes
    (code * scale), packed as an int4 (8 bf16 = 16 bytes).
  - `gqa_kv_quant_code_index` / `gqa_kv_quant_scale_index` — the index math.
- The prologue (gqa_decode_slice3_i8.cuh:23-50): load the int8 codes + the
  per-group __half scales, dequant to bf16 (code * scale), write to smem.

### 3.2 The KVarN K4V2 — the model for the KVarN bit-width variants
- `kvarn_codec.h` + `kvarn_codec.cpp` (the CPU codec) + `kvarn_tile_cuda.h`
  (the GPU codec):
  - K tile: [D=128 channels, G=128 tokens], int4 asymmetric RTN, packed as 2
    nibbles per byte. `q_packed: [D, G/2] bytes`. Folded scales: `s_col_K: [D]`,
    `zp_K: [D]`, `s_row_K: [G]`.
  - V tile: [G=128 tokens, D=128 channels], int2 asymmetric RTN, packed as 4
    2-bit codes per byte. `q_packed: [G, D/4] bytes`. Folded scales: `s_row_V:
    [G]`, `zp_V: [G]`, `s_col_V: [D]`.
  - FWHT along the head dimension D (power of 2) + Sinkhorn variance
    normalization (calibration-free, 16 iterations).
- The prologue (gqa_decode_slice4_kvarn.cuh): load the codes + the scales + the
  FWHT, dequant to bf16 (code * scale + FWHT), write to smem. The FWHT moves to
  the PROLOGUE (the canonical body does NOT do the FWHT).

## 4. The design (against the unified kernel)

### 4.1 The non-KVarN int4 prologue (q4_0 / q4_0) — the #1 priority
- The K and V are both int4 (4-bit, symmetric, per-group scale — like I8 but
  4-bit instead of 8-bit).
- The quantization: code = round(x * inv_scale), clamp to [-7, 7] (4-bit
  symmetric). The group size = 64 (like I8). The scale = per-group (like I8's
  per-64-group __half scale, but 4-bit = absmax/7).
- The code byte size: K int4 = D/2 bytes per group (2 codes per byte); V int4 =
  D/2 bytes per group (2 codes per byte).
- The prologue: load the int4 codes + the per-group scales, dequant to bf16
  (code * scale), write to smem. Like the I8 prologue but 4-bit.
- The new device helpers (gqa_attention_kv_quant.cuh):
  - `gqa_kv_quant_code_i4(x, inv_scale)` — quantize to int4 (clamp to [-7, 7]).
  - `gqa_kv_dequant_i4x8_from(codes8, s)` — dequantize 8 consecutive int4 codes
    (2 per byte, code * scale), packed as an int4 (8 bf16 = 16 bytes).
- The new prologue (gqa_decode_slice5_i4.cuh): like the I8 prologue but 4-bit.
- The new kernel (gqa_decode_slice5_i4_kernel): like the I8 kernel but 4-bit.
- The new test (slice5_i4_test.cu + slice5_i4_dequant_test.cpp): like the I8
  tests but 4-bit.
- The effort estimate: ~2-3 days (the quantization + the prologue + the kernel +
  the test + the gate).

#### 4.1.1 Scope split (2026-08-30, coordinator) — what is the DECODE slice
vs what rides on Phase-3
- I8's prefill (gqa_attention_prefill_i8.cuh, 605 lines) is a standalone
  INT8-NATIVE kernel (QK stays s8 through the m16n8k32 MMA; cache is filled
  + attention read in the same pass). It predates the unified kernel. There is
  NO int4 s4-MMA path, so an "int4-native prefill" would dequant codes to
  bf16/fp16 in a prologue and run the canonical body — i.e. it IS the
  Phase-3 unified-prefill shape (docs/113) with an int4 dequant prologue,
  exactly like KVarN/I8 prefill will be. **Decision: the int4 prefill/cache-
  fill is NOT a standalone kernel — it is one more per-TKV prologue of the
  Phase-3 unified prefill (Agent 1's lane).** The int4 work that can start
  immediately (no Phase-3 dependency) is the DECODE slice, which is a direct
  clone of the I8 decode template:
  1. `gqa_attention_kv_quant.cuh`: + `gqa_kv_quant_code_i4` (clamp [-7,7]),
     + `gqa_kv_dequant_i4x8_from` (4-bit: 2 codes/byte, code*scale → 8×bf16
     int4 vector). Both trivial; the I8 versions are the template (77-line
     file today).
  2. `gqa_decode_slice5_i4.cuh`: clone of gqa_decode_slice3_i8.cuh (393 lines,
     now clean + oracle-tested). Deltas: code byte size D/2 per head-row (vs D),
     scale = absmax/7 (vs /127), dequant call. The body is UNTOUCHED (join-not-
     rewrite: the same bf16 canonical body the I8/KVarN prologues feed).
  3. Dispatch: gqa_attention_decode.cu already has the dtype-branching shape
     (I8 added `cache.dtype == DType::I8 → launch_tc_partial_unified_i8`);
     int4 adds `DType::I4 → launch_tc_partial_unified_i4` with the same
     split/capacity/partial layout (gqa_attention_split_capacity for I4).
  4. Enum + parse (types.h:26-29, serve_options.cpp:48-54): `Int4Group64` +
     `--kv-dtype q4_0`.
  5. Oracle test: clone of tests/slice3_i8_test.cu (8 cases incl. the
     T=5 split-window + T=6 Br-capacity + fused-append + MB2; tolerance =
     0.01 + 0.008·max|V|; the shipped-kernel precision invariant becomes
     "unified ≤ bf16-dequant-of-shipped-int4" — NOTE: there is no shipped
     standalone int4 decode kernel today (enum has only 3 variants), so the
     invariant is against the FP64 oracle only, tol scaled for 4-bit
     (dequant error ~2× the I8 per-code error; keep the V-scaled form).
  6. Gate: ~~A2 identity (mtp1==mtp0)~~ + acceptance/t/s vs the Phase-0 matrix —
     (identity gate corrected — see §5: divergence-rate comparison vs bf16, not byte-identity)
     — BLOCKED until the prefill cache-fill exists (Phase-3); unit oracle +
     dequant CPU test unblock first.
- Effort (re-estimate with the template in hand): quant helpers 0.1d, slice5
  clone + dispatch 0.5d, oracle test 0.5d, enum/parse 0.1d → **~1.2 days for
  the decode slice**; the prefill half is inside Phase-3's per-TKV prologue
  work (marginal +0.5d for int4 once the Phase-3 I8/KVarN prologue machinery
  exists).

### 4.2 The KVarN bit-width variants (kvarn8/kvarn8, kvarn6/kvarn6, etc.)
- The KVarN K4V2 uses kKvarnKCodeBytes = 256*32 (K int4) + kKvarnVCodeBytes =
  64*64 (V int2). The other bit-widths (K8V8, K6V6, etc.) need new code byte
  sizes + new quantization kernels (different bit-widths).
- The code byte size: K intN = D * N/8 bytes per group; V intM = D * M/8 bytes
  per group (like K4V2: K int4 = 256*32 = 8192 bytes, V int2 = 64*64 = 4096
  bytes).
- The quantization: like K4V2 but different bit-widths (K intN + V intM
  asymmetric RTN + FWHT + Sinkhorn).
- The prologue: like the KVarN K4V2 prologue but different bit-widths (load the
  codes + the scales + the FWHT, dequant to bf16).
- The new prologue (gqa_decode_sliceN_kvarn.cuh): like the KVarN K4V2 prologue
  but different bit-widths.
- The new kernel (gqa_decode_sliceN_kvarn_kernel): like the KVarN K4V2 kernel
  but different bit-widths.
- The new test (sliceN_kvarn_test.cu + sliceN_kvarn_dequant_test.cpp): like the
  KVarN K4V2 tests but different bit-widths.
- The effort estimate: ~3-5 days per type (the quantization + the prologue + the
  kernel + the test + the gate). But the types are similar (the same structure,
  different bit-widths), so the marginal effort decreases (the first type is
  ~3-5 days, the rest are ~1-2 days each).

### 4.3 The enum + the parse
- Extend the enum (include/ninfer/types.h:26-29) with the new types (e.g.
  Int4Group64, KvarnK8V8, KvarnK6V6, KvarnK6V5, KvarnK5V5, KvarnK5V4,
  KvarnK4V4, KvarnK4V3, KvarnK3V3, KvarnK3V2, KvarnK2V2, Q8Q6, Q5Q4).
- Extend the parse (src/serve/serve_options.cpp:48-54) with the new types (e.g.
  `--kv-dtype q4_0` / `kvarn4` / `kvarn5` / etc.).
- The effort estimate: ~0.5 day (the enum + the parse).

## 5. The validation (per type)
- ~~The A2 token identity (MTP==plain)~~ + the acceptance gate: a tier is
    refused only if acceptance falls more than 0.5pt BELOW the bf16
  reference + the t/s within the Phase-0 envelope (the same gate spec as the
  I8 + KVarN K4V2 gates).
  - **IDENTITY GATE CORRECTED (2026-09-05, measured; coordinator-ratified, surfaced to
    the user for ratification).** Byte-identity between `mtp1` and `mtp0` was a
    **mis-specified gate, not a quality bar**: it tests a property the MTP path does not
    guarantee by design. The verify path and the plain decode path differ numerically the
    same way unified-vs-packed decode is documented to (reduction order, tile shape), so a
    near-tie argmax flips. Evidence below is on the real 18 GB artifact, 4 prompts × 160
    tokens, greedy, TP2 (`tools/smoke/diag/s5_identity_gate.sh`, `s5_identity_multi.sh` —
    first implementation of this gate; `serve_correctness_ci.sh` T1/T2/T3/T5/T8 never
    covered it):

    | tier | prompts diverging (n=4) | divergence position |
    |---|---|---|
    | bf16 (**no quantization to blame**) | **2/4** | 62%, 17% |
    | kvarn_k5v4 | **1/4** | 16% |
    | kvarn_k4v2 | 4/4 | 88%, 50%, 28%, 16% |

    Every divergence is late and both sides are coherent paraphrases ("accuracy"/"quality",
    "leads to failure"/"will result in failure") — never garbage, never token 0. The haiku
    prompt diverges on **every** tier, which is the tell: it is the maximally unconstrained
    prompt, so near-ties are guaranteed.
  - **Corrected gate:** the tier's `mtp1`-vs-`mtp0` divergence **rate, position and magnitude
    must be comparable to the bf16 reference's**, not zero. Under the corrected gate k5v4
    (1/4 ≤ bf16 2/4) **PASSES**. This is fixing a broken gate, not weakening a working one —
    as literally written it would fail any quantized tier for a reason that also fails bf16.
  - Process note, because it nearly cost a shipped tier: at n=1 prompt this read as "k4v2 is
    broken". One extra experiment (4 prompts, 2 server loads) showed the **gate** was broken,
    not the tier. Do not sign off an identity verdict from a single prompt.
- The acceptance gate (refused only if acceptance falls more than 0.5pt BELOW the
    bf16 reference; ONE-SIDED per user ruling 2026-09-05) + the t/s within the Phase-0 envelope —
  measured 2026-09-05 on k5v4: acceptance 0.738 vs bf16 0.739 (**0.1 pt, PASS**), t/s 1.54×
  over no-speculation vs bf16 1.54× (**PASS**). Side finding: the **shipped k4v2 tier is 2.6 pt
  below bf16**, outside the ±0.5 pt bar — surfaced to the user (may be expected for the
  more-aggressive tier, or warrant a tier-specific bar; not this lane's call to bury).
- The VRAM — measured 2026-09-05: the differential model validates **<0.2%** against 6 real
  preflight cells (per-token slopes 9542 vs `kv_unit` 9537, and 13017 vs 13005; k5v4−k4v2
  differential 678 MiB measured vs 677.4 predicted). Full table in
  `docs/k5v4_s5_validation_results.md`.
- The KLD (the user's table has the Median KLD + the 99.9% KLD — validate that
  the measured KLD matches the table).
- The VRAM (the user's table has the Size + the Size vs bf16 — validate that the
  measured VRAM matches the table).
- The effort estimate: ~0.5 day per type (the A2 + the acceptance + the t/s +
  the KLD + the VRAM).

## 6. The total effort estimate
- The enum + the parse: ~0.5 day.
- The non-KVarN int4 (q4_0/q4_0): ~2-3 days.
- The KVarN bit-width variants (10 types): ~3-5 days for the first type (the
  balanced default, kvarn4/kvarn4) + ~1-2 days for each of the rest (the same
  structure, different bit-widths) = ~13-25 days.
- The validation (11 types): ~0.5 day per type = ~5.5 days.
- **Total: ~21-34 days** (the int4 + the 10 KVarN bit-width variants + the
  validation).

## 7. The priority order (the user's #1 priority first)
1. **q4_0 / q4_0** (non-KVarN int4) — the #1 priority.
2. **kvarn4 / kvarn4** (the KVarN balanced default — the most-used KVarN type).
3. **kvarn5 / kvarn4** (the KVarN balanced default).
4. The rest (kvarn8/kvarn8, kvarn6/kvarn6, kvarn6/kvarn5, kvarn5/kvarn5,
   kvarn4/kvarn3, kvarn3/kvarn3, kvarn3/kvarn2, kvarn2/kvarn2, q8_0/q6_0,
   q5_0/q4_1).

## 8. The next phase (DFlash)
- After the int4 + KVarN cache types are done, the next phase is DFlash
  (docs/56, Path B: DFlash2 block drafter). See docs/56 for the details.

## 9. Execution — scoped to 2 variants (int4, then kvarn5/4) (2026-09-04, coordinator)

> **SCOPE NARROWED (user, 2026-09-04):** implement **q4_0/q4_0 (int4) FIRST**, then **kvarn5/kvarn4**.
> **kvarn3/kvarn3 and all other §1 types are DEFERRED to backlog** — not current scope. Where item 3
> or the team lines below mention k3v3, treat it as backlog (skip for now).

**Gate resolved.** The docs/104 unified kernel this design was gated behind is now MERGED
(github/main `899e8d4b`). These types build directly against the shipped unified kernel as per-TKV
dequant prologues. No Phase-2/3 blocker remains for the DECODE slice; only the prefill/cache-fill
half rides on the Phase-3 unified-prefill prologue.

**Parameterization verdict — why 3, not all 11 (investigated 2026-09-04).** A new KVarN bit-width is
NOT a launch-var flip into a table. The width is baked into: (1) the CPU codec structs —
`KvarnKTile` hardcodes int4 `[D,G/2]`, `KvarnVTile` hardcodes int2 `[G,D/4]`; (2) the decode
prologue `gqa_decode_slice4_kvarn.cuh` — nibble shifts `b&0x0F`/`b>>4`, `kKvarnKTileBytes=4224`,
2-bit V unpack; (3) **39** `KVARN_K4V2` references/guards that `throw` otherwise. The MMA kernel has
*partial* parameterization (`kvarn_unpack_code(row,d,v_bits)`, `if (v_bits==2)`) but that alone does
not carry a new type. And the requested k5v4/k3v3 use **non-power-of-2 widths (5, 3) that need
bitstream packing** (codes span byte boundaries) — absent everywhere today. Per the user's rule
(hard → don't do all): implement ONLY the 3 below; the other 8 stay a §1 backlog.

**Per-variant difficulty (so the team sizes honestly):**
- **q4_0/q4_0** — power-of-2 (nibble), direct I8-template clone. Moderate, ~1.2 d decode.
- **kvarn5/kvarn4** — V=4 is a nibble (like the existing K4 packing) but **K=5 needs bitstream
  packing** (5-bit codes across byte boundaries). The K=5 unpack/repack is the hard part.
- **kvarn3/kvarn3** — **both 3-bit, both bitstream.** Hardest; no byte-aligned shortcut.

> **CORRECTION (A2, 2026-09-05):** the "bitstream packing absent everywhere" claim above is STALE. The GPU codec (`kvarn_tile_cuda.cu`, docs/69) is ALREADY width-generic — `kvarn_pack_row/unpack_row<BITS>` for BITS 2..8 incl 5 and 3, LSB-first, byte-aligned-safe. So **k5v4 is NOT the hard case** — A1's prologue reuses `kvarn_unpack_row<5>`. The *CPU* codec (`kvarn_codec`) was the hardcoded part; A2 generalized it (width-generic, round-trip tests PASS). Budget for §5 VRAM: int4 = 9792 B/token/rank, k5v4 = 12240 (use these, not §1's foreign-tail table).

**Scope (user directive 2026-09-04, "asap") — three variants, not the full §1 table:**
1. **q4_0 / q4_0** — non-KVarN int4 (user's #1). Clone the I8 decode template (§4.1.1). Decode
   slice ~1.2 d. Enum `Int4Group64`, `--kv-dtype q4_0`.
2. **kvarn5 / kvarn4** — KVarN balanced default (§1: 1360 MiB, 33.2% of bf16, median KLD 0.000936).
   Enum `KvarnK5V4`, `--kv-dtype kvarn_k5v4`.
3. **kvarn3 / kvarn3** — "when the context must fit" (§1: 976 MiB, 23.8%, median KLD 0.001316).
   Enum `KvarnK3V3`, `--kv-dtype kvarn_k3v3`.

**Team assignment (no two owners on one file):**
- **A1 — implementation (CUDA).** The decode prologues + kernels: `gqa_decode_slice5_i4.cuh` (clone
  of `gqa_decode_slice3_i8.cuh`, code byte size D/2, scale absmax/7), and the KVarN k5v4 / k3v3
  prologues (clone of `gqa_decode_slice4_kvarn.cuh` with new bit-widths + code byte sizes: K intN =
  D·N/8 bytes/group, V intM = D·M/8). New device helpers in `gqa_attention_kv_quant.cuh`
  (`gqa_kv_quant_code_i4`, `gqa_kv_dequant_i4x8_from`). Owns `src/ops/` + the kernel dispatch.
- **gemini — tests (contracts-first, no src/).** Oracle tests: clone `tests/slice3_i8_test.cu` for
  i4 (8 cases incl. T=5 split-window, T=6 Br-capacity, fused-append, MB2; tol scaled for 4-bit —
  dequant error ~2× I8 per-code, keep the V-scaled form, FP64-oracle-only since no shipped int4
  decode kernel exists). KVarN k5v4/k3v3 oracle tests clone the K4V2 tests. Plus the §5 validation
  specs (~~A2 identity mtp1==mtp0~~ → identity = divergence rate/position vs bf16, see §5;
  acceptance >0.5pt BELOW bf16 refuses (one-sided), t/s envelope, KLD vs §1, VRAM vs §1).
- **A2 — CPU-only support, lands FIRST to unblock the others (no GPU):**
  (a) enum + parse: `Int4Group64`/`KvarnK5V4`/`KvarnK3V3` in `types.h:26-29` + `serve_options.cpp:48-54`
  + the `request_log` string + the `--kv-dtype` help; (b) CPU codec: `kvarn_codec.cpp` encode/decode
  for k5v4 + k3v3 (the reference gemini's tests check against); (c) dequant CPU round-trip tests;
  (d) budget/sizing: per-token bytes for the 3 new types (feeds the preflight + the §1 VRAM check);
  (e) CI gate wiring: `run_ci.sh` cells for the new dtypes (wiring only — the runs are GPU-gated).

**Sequence:** A2(a) enum/parse → unblocks A1 dispatch + gemini test compile → A1 prologues/kernels →
gemini oracle tests green → A2(b–e) in parallel → §5 validation on a GPU window.

**Pre-landing standing check (A1, 2026-09-05 — a PATTERN, not a pair):** before each new dtype lands,
grep for `DType::I8` / `DType::BF16` fallthroughs and any `?:`/switch default that silently coerces an
unhandled tier into a real one. Wiring int4 surfaced TWO such bugs, both live on main, neither in
anyone's lane: `layouts_impl.h` mapped every non-BF16 enum → I8 (q4_0 would've served int8 under the
int4 name); `tp2_backend.cpp` decoder_spec fell through to BF16 (TP2 would allocate full-width bf16 +
report §1 VRAM as if int4). Fix: explicit arm per dtype + **throw on unhandled tier** (never fall back).
The same shape will be waiting for k5v4 — check before landing, don't rediscover per-tier.

**Toy-slice / single-GPU lane:** PAUSED (not dead) per user — A1 pivots here; resume when the user
redirects. The world=1 driver work already landed (barrier fix, preflight truth) is preserved on
wo/2a-batched-serving.
