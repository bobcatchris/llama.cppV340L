# 150 — int4 (q4_0) decode contract — for the tests lane

**Owner:** A1 (`wo/117-int4`, commits `b37ba289` → `37fefaf5` → `b9c78dec`).
**Audience:** whoever fills in `tests/slice5_i4_test.cu::run_case()`.
Number 150 claimed from the coordinator (was a `117a` placeholder).

## 1. Status

The int4 **decode** slice is implemented and validated: 9/9 FP64-oracle cases pass at
~9x tolerance margin, and the harness is mutation-tested. What is **not** done is the
prefill cache-fill, so `--kv-dtype q4_0` still refuses to serve. See §5.

## 2. ⚠ The existing test file currently cannot fail

`tests/slice5_i4_test.cu` (as committed in `53ddeec5`) is:

```c
#if __has_include("ops/kernel/gqa_decode_slice5_i4.cuh")
#include "ops/kernel/gqa_decode_slice5_i4.cuh"
#define HAS_SLICE5_I4 1
...
int run_case(const Cfg& cfg) {
#if !HAS_SLICE5_I4
    return 1;                       // was correct: header absent -> EXPECTED RED
#else
    // Placeholder for kernel invocation once A1 lands gqa_decode_slice5_i4.cuh
    return 0;                       // <-- now arms VACUOUSLY
#endif
}
```

`HAS_SLICE5_I4` flipped to `1` when `b37ba289` landed the header. So the file prints
`ninfer_slice5_i4_test: PASS (oracle match)` for all 9 cases while launching **nothing**.
The `#if !defined` guard was the right design when written; the hazard is that its
"green" branch is a stub. Until `run_case()` really launches the kernel and really
compares, that test must not be counted as a gate.

A green that is armed by the mere existence of a header is the same shape as the
"fake-armed readiness echo" in the agent1 handoff §7: the success path is reachable
without the thing under test having run.

## 3. Kernel signature (what to call)

`src/ops/kernel/gqa_decode_slice5_i4.cuh`, namespace `ninfer::ops`:

```c
template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__global__ void gqa_decode_slice5_i4_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos,
    std::uint8_t* cache_k, std::uint8_t* cache_v,      // NOTE: U8 packed codes, not int8
    __half* cache_k_scale, __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
    float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l);
```

Launch: `grid(Geometry::KVHeads, splits, batch)`, `block(WarpsPerCta * 32)`,
dynamic smem `kI4UnifiedSmemBytes<WarpsPerCta*16, 32, WarpsPerCta, 64>` — and you must
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem)` first
(it is ~45 KB, above the 48 KB default ceiling once you count the rest).

`WarpsPerCta` is a **capacity** requirement, not a perf knob:
`WarpsPerCta * 16 >= TokenTile * Geometry::GroupSize`, and `WarpsPerCta <= 4` because the
Q tile stages across `qkv_s` (2*Bc rows). The launcher uses 2 for T=1, 4 for T>=2.

### The one thing that differs from the I8 clone

Reduce kernel: launch with **`Int8 = false`**, and the split policy inside the kernel is
`gqa_small_t_active_splits<Geometry, false>`. The I8 route uses `true`. These two must
agree or the reduce skips or double-counts a split. int4 rides the canonical bf16 body,
so it takes the default formula; the I8 specializations are s8-MMA perf tuning.

If you clone `slice3_i8_test.cu` verbatim you will inherit `Int8=true` in both the
partial and reduce launches and get wrong answers at any window where the two formulas
disagree (T=5 window 129..512, T=6 window 129..160 and 5001..8198).

## 4. Layout + numeric conventions

Per `(physical_page, kv_head, page_offset)` code row: **D/2 = 128 bytes** holding 256
signed 4-bit codes. Scale row: unchanged from int8 — 4 `__half`, one per 64-dim group.

```
code byte = 128*64*(kv_head + KVHeads*page) + 128*off + d/2
scale     =   4*64*(kv_head + KVHeads*page) +   4*off + d/64
```

These match `code_byte_off` / `scale_off` already in the test file. Good.

**Nibble order:** low nibble = even dimension, high nibble = odd dimension. Matches
`pack_i4_pair` / `dequant_i4` already in the test file. Good.

**Sign extension — the one place the test file and I disagree, and the test file is right:**
codes are **two's-complement 4-bit**: `n >= 8 ? n - 16 : n`, i.e. range `[-8, 7]`.
The kernel uses `(n ^ 8) - 8`, which is identical. My first harness run was 9/9 red
because I wrote offset-binary `n - 8` there; the kernel was correct. `dequant_i4` in the
test file already does the right thing — do not "fix" it to match anything else.

**Quantization (fused append):** `scale = __float2half_rn(absmax / 7.0f)`,
`code = __float2int_rn(x * (1.0f/scale))` clamped to `[-7, 7]`. Note it is
multiply-by-reciprocal, exactly as in the int8 route, not division.

## 5. Fused append needs TWO checks, not one

The oracle consumes the cache **as read back from the device**. That makes the output
comparison *self-consistent under a wrong quantizer*: if the kernel writes codes using a
bad scale, the oracle dequantizes the same bad scale and they agree.

So check both:
1. attention output vs FP64 oracle over the read-back cache, and
2. the written codes + scales vs an independent host quantization of `input.k`/`input.v`.

Proof this matters: mutating the kernel's append scale from `absmax/7` to `absmax/127`
(the int8 value) is caught **only** by check 2 — 528 mismatches — while check 1 passes
all 9 cases. A test with only check 1 would wave through an int8-scale bug.

## 6. Tolerance

`tol = 0.02 + 0.034 * maxV`, per §4.1.1 item 5. Measured worst across the 9 cases is
0.0081 against tol 0.0747, so there is ~9x headroom; do not loosen it further without a
reason, and note `maxV` must be computed over the **post-append** cache.

## 7. What is NOT wired (do not gate on it)

- **Prefill / cache-fill.** `gqa_attention_prefill.cu` branches
  `cache.dtype == DType::I8 ? int8-kernel : bf16-kernel`. An `DType::I4` cache would hit
  the bf16 arm and be filled with garbage. This is Phase-3 per §4.1.1.
- Consequently `serve_options.cpp` still **refuses** `--kv-dtype q4_0`, with the reason
  narrowed to "decode prologue wired, prefill cache-fill not". §5 serve validation
  (A2 identity, acceptance ±0.5pt, t/s envelope, KLD, VRAM) stays blocked. Do not run it
  and do not report int4 as end-to-end done.
- Budget numbers for when it does land are A2's, not §1's table: **int4 = 9792
  B/token/rank**.

## 8. Reference implementation

`tools/i4_oracle/i4_oracle.cu` on `wo/117-int4` is a complete working version of all of
the above — 9 cases, both append checks, FP64 oracle. It lives under `tools/` rather
than `tests/` because `tests/` is the tests lane's; lift whatever you want from it.
