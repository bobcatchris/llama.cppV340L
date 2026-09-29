# V0-PK — the HFMA2 tiled GEMM arm: what landed, gate matrix, runbook (CODE desk, 2026-09-17, amd/tp4-cure)

Spec: `docs/amd/PREFILL_GEMM_SCALE_2026-09-17.md` §2/§5 (commit **9a38c0c50**) — read it first;
this file only records what landed and how to drive it. ZERO GPU work happened at this desk:
compile-only, the bench binary was built and never executed (GPU-window law).

## 1. What landed (one commit, exact paths)

| file | change |
|---|---|
| `src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu` | `nvfp4_tiled_gemm_hip_kernel_pk<STILES, TN, WINDOW_W>` — V0's EXACT tile (64-row x TN, 4 rows x TN/16 tokens, K-tile 64, XOR-swizzled s_x, u64 code loads, scale-quad trick) with the SMALLT-PK arithmetic: e2m1 pair table as half2 bits (1 KB, filled FROM `amd::e2m1_bits` via `__float2half_rn` — exact by construction, dyadic values), x staged as packed fp16 via the D3 bridge (`bf16x2_bits_to_f16x2_bits`, math.cuh, exact-class C441) before the swizzled LDS store, coeff = fp32 contract fold per (row, group) then one `__float2half2_rn` cast (hoisted out of the v loop, in-window arms), `pw2 = __hmul2(pair, coeff2)` once per (row, v) invariant over tokens, ONE `__hfma2` per (row, token, value-pair), fp16 window `s2` flushed to the fp32 contract accumulator `acc`. Plus `launch_tiled_pk_exact` / `dispatch_tiled_pk` (15-problem switch, mirrors V0's machinery) and the public A/B symbol `launch_nvfp4_tiled_gemm_pk(x, w, out, stream, window_w)`. V0's kernel/launcher byte-untouched. |
| `src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.h` | `launch_nvfp4_tiled_gemm_pk` decl + the numerics block: k-ascending consumption ORDER preserved; the ASSOCIATION change is DECLARED tolerance-gated per `nvfp4_amd_codec.h` :131-138 (the codec text's own declaration grammar); gate = rel-L2(PK, V0) < 1e-2; NO FNV cell may gate this arm (design §2 gate (c)). |
| `src/ops/linear/nvfp4/nvfp4_dispatch.cpp` | `tiled_pk_gate()` — `NINFER_TILED_PK`, VALUE-parsed per the N7 house law (see matrix); `launch_a16` routes the M>=64 tiled-eligible shapes to `launch_nvfp4_tiled_gemm_pk(..., window_w)` when the gate is nonzero, BEFORE the `NINFER_TILED_VERIFY` FNV cell (a PK leg must not enter a bit-equality diff that is false BY DESIGN); `[TILED]` trace label follows the gate: `arm=pk window=W` vs `arm=nvfp4_tiled_gemm`. |
| `results/amd/coherence/nvfp4_prefill_bench.cu` | `--pk` mode (`run_pk`): V0 vs V0-PK at M=128 on the FIVE W4 SHARD geometries (below), W in {16,32,64}, rel-L2(PK, V0) column (gate 1e-2, exit rc 1 on any FAIL), xV0 ratio vs the pre-registered floors, `ENVELOPE` + `WINDOWCENSUS` lines measured from the ACTUAL staged buffers (max|x| over bf16 bits, max|coeff| over e4m3 bytes; bound = W*6*max|x| for W=16 / W*6*max|x|*max|coeff| for W=32/64), 5 s mclk hammer + best-effort `pp_dpm_mclk` sysfs print + hipEvent timing (3 warmup + 20 iters), %fp32-nominal column (10.75 TF/s anchor). Plus `run_tiled_pk` and `print_mclk_sysfs` helpers. Default and `--sweep` modes untouched. |

**W4 shard geometries** (`nvfp4_config.h`, never measured before — prior sweeps ran the four
FULL geometries): AttnInputW4 3584x5120 (STILES 80), GdnInputW4 4096x5120 (80),
MlpGateUpW4 8704x5120 (80), Residual6144W4 5120x1536 (24), Residual17408W4 5120x4352 (68).

**Window semantics** (all three divide the 64-wide k-tile, windows never cross it):
W=16 = SMALLT-PK shape verbatim (window holds raw code*x products; the fp32 coeff multiplies
at flush: `acc = fmaf(lo+hi, coeff, acc)`); W=32 (recommended) / W=64 (one k-tile) fold
`coeff2` into the weight pair, flush is `acc += lo+hi`.

## 2. Gate matrix (N7 value-parse law — never presence-parsed)

| `NINFER_TILED_PK` | behavior |
|---|---|
| unset | OFF — byte-identical production (V0 path, zero new code on the launch path) |
| `=1` or `=w32` | PK arm, W=32 (the design's recommended window) |
| `=w16` | PK arm, W=16 |
| `=w64` | PK arm, W=64 |
| `=0` / `=off` / `=""` / anything else | **REFUSES LOUD** at first route (std::runtime_error naming the value + accepted set) — must not silently open or close a route gate |

Interactions: `NINFER_A16_FORCE_SMALLT=1` forces the legacy small_t chunk loop (route
predicate false — PK never engages, forced arm wins). `NINFER_TILED_VERIFY=1` with PK set:
PK wins, the TILEDV FNV diff is bypassed (bit-equality is false BY DESIGN under PK — design
§2 gate (c)); TILEDV keeps its exact semantics on non-PK boots. `NINFER_TILED_TRACE=1`
prints `[TILED] arm=pk window=32 n=.. k=.. M=.. (launch #..)` — the serving-side route proof.
Only the registered whole-M tiles {128, 256} (the M>=64 threshold) reach the PK route, same
predicate as V0; M=32 and remainders keep the small_t chunk loop.

## 3. Compile status (zero-GPU desk, no execution of anything)

Law line (ONESHOT_AR_notes §5 form, from the worktree root `W=/home/chris/worktrees/amd-tp4-cure`):

```bash
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -O3 -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 \
  -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I src/common/hip_shim -I include -I src -I third_party -DNDEBUG -std=gnu++20 \
  --offload-arch=gfx900 -x hip \
  results/amd/coherence/nvfp4_prefill_bench.cu src/core/device.cu \
  -L/opt/rocm-6.2.0/lib -lamdhip64 -o /tmp/nvfp4_prefill_bench_pk
```

- bench TU (+ device.cu, one-shot form): **RC=0**, 201 warnings = 186 `-Wunused-result`
  nodiscard class (cudaMalloc/Memset/Memcpy/Free/EventCreate ignores — the documented bench
  class; the sweep build of this same file counted 143 on its own cell set) + 15
  `-Wpass-failed` in nvfp4_small_t_hip.cu (**count unchanged** from HEAD).
- tiled TU standalone (production HIP line): **RC=0, 0 warnings**.
- dispatch TU (production host line, `-Wswitch`): **RC=0**, 7 warnings = the pre-existing
  nodiscard set inside `tiled_verify_arm` (untouched code; HEAD baseline compiles to the
  same 7).
- Kernel census in the bench object: 126 unique `nvfp4_tiled_gemm_hip_kernel_pk` symbols =
  7 distinct STILES x {32,128,256} x {16,32,64} (same-STILES problems dedup — STILES is the
  only kernel-varying template param); the W4 shard STILES {80, 24, 68} all present x all
  TN x all W. V0 path untouched.
- Binary `/tmp/nvfp4_prefill_bench_pk` exists and was **NOT executed** (no GPU grant at this
  desk; the GPU window runs it).

## 4. GPU-owner runbook (ONE window, in order)

1. **Rebuild** the boot artifact from this branch (`amd/tp4-cure`) per
   `docs/amd/BOOT_LAUNCH_RUNBOOK.md`; **BANK-BEFORE-RELINK** (§4) — bank the boot-stamped
   binary before any relink; boot only from the bank.
2. **BOOT_BATTERY** on the new artifact, gate OFF (default): must be OVERALL GREEN and
   byte-identical-behavior to the standing battery (default path is unchanged by law — this
   leg PROVES that, it is not a formality).
3. **The decisive bench**: run the law command above (rebuild fresh if /tmp was cleaned),
   then `NINFER_A16_FORCE_SMALLT` UNSET, no env needed:
   `/tmp/nvfp4_prefill_bench_pk --pk` — 5 s mclk hammer + ceilings inside; watch the sclk
   sideband per house discipline. Bank the raw log as the `--pk` row per-LABEL.
4. **Decide on the printed verdicts**: decisive row = `PK-W32 xV0` at M=128.
   - `>= 1.8x` (with relL2 PASS on every row) → serving leg:
5. **Serving leg**: boot the banked artifact + `NINFER_TILED_PK=1` (winner W; `=w16`/`=w64`
   if W16/W64 won the bench), `NINFER_TILED_TRACE=1` (route proof must print
   `arm=pk window=..`), `NINFER_PREFILL_OPTRACE=1`, plen-1996 probe (the P1 OPTRACE class)
   + standing ladder battery. Decisive number: **`[PREFILL-SUM] gemm < 798.8 ms/chunk`**
   (P1 instrumentation already in the bin — zero new code for the readout). Pre-registered:
   gemm <= 500 = arm confirmed in serving; gemm > 700 = the serving/bench class factor ate
   the arm — read the sclk sideband before concluding anything. TILEDV stays OFF for PK legs.
   Then the promote decision goes to the coordinator with both rows linked.
   - `1.3x - 1.8x` at W=32 → try W=64 row before any tuning; if still < 1.8x the floor is
     missed: dump the ISA (flush/cvt/int bill dominates the model), NO further blind knobs.
   - ALL PK arms `< 1.3x` → **the arm is DEAD** (pre-registered): write the ISA post-mortem,
     reopen nothing (design §5).
   - any `relL2 FAIL` row → do NOT promote; the fp16 window/flush design is falsified at
     that geometry — bank the census first.

## 5. The honest prediction band (carried from the design, unchanged)

- **Bench** (cache-served class = model x the 0.52-0.69 model-to-silicon band, @1.5 GHz x
  56 CU nominal): W=16 **5.0-6.6 TF/s** (2.1-2.8x V0), W=32 **6.1-8.2 TF/s** (2.4-3.3x,
  central ~6-8), W=64 **7.0-9.3 TF/s** (2.7-3.7x). Pre-registered floor W=32 >= 1.8x;
  all-arms < 1.3x = dead.
- **Serving**: the measured serving/bench class factor 0.66-0.67 (T1 A/B + P1; mechanism
  unexplained) carried multiplicatively gives **250-470 ms/chunk** (central ~350) vs the
  798.8 baseline — i.e. 1.7-3.2x; best case (factor proven clock-only) ~190-240. Wall
  impact with other slices frozen: 1642 → ~1090-1190 ms ⇒ ~107-117 tok/s. This is the
  single biggest measured-available lever and NOT the 500 goal by itself (design §4: no
  GEMM-only lever reaches even 152).

## 6. Honesty row

- The predicted PK TF/s inherit the static-census uncertainty (design §6); the bench leg
  measures them end-to-end — nothing here is claimed measured that was not.
- The max|window| census is the envelope bound computed from the ACTUAL staged buffers
  (memset-pattern bench data), not a per-thread runtime max; it measures guard (b) exactly
  as the design's formula defines it, on the data the kernels actually consume.
- gfx900 fp16 denormal behavior inside the D3 envelope is part of what rel-L2 gates; the
  ENVELOPE line prints the measured min/max |x| against the fp16-normal law.
- This desk touched ONLY the five files in §1. Live GPU-window artifacts
  (results/amd/coherence rows/logs, serve logs) were read, never written.
