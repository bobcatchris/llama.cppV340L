# CAMPAIGN DOSSIER - AMD V340L (gfx900) llama.cpp Long-Context Serving Optimization

**Document date:** 2026-09-24 | **Repo state:** branch `amd/v340-port-v2` @ `a710244d4`
**Audience:** a brand-new reviewer who has never seen this machine, this model, or this
campaign, and needs to understand the problem, the stack, the work, and the walls
immediately - without reading the 214-entry ledger first (though every claim here
cites it: ledger entries are `- E-NNN` in `docs/amd-port/OPTIMIZATION_PLAN_TP3_200K.md`,
desk receipts are `docs/amd-port/results/W*.md`).

**The one-paragraph version:** we serve a 27B hybrid-attention MTP model on four
AMD Vega-10 dies (gfx900) over PCIe, tensor-parallel, with quantized KV and a
quantized speculative draft, and the entire optimization campaign revolves around
one structural fact: at short context, **~70-75% of every decode round is spent in
the 136 per-layer inter-die allreduce synchronizations** that tensor parallelism
requires on a PCIe fabric. Compute itself is nearly free by comparison. Around that
fact we have: promoted a miscompile-workaround attention kernel (+5% served,
bit-exact), characterized and fixed a hipcc codegen bug, survived five hard machine
deaths from a ROCm KFD SVM defect, named and instrumented a thermal throttling
mechanism that costs 26% of decode, found and fixed a correctness crash in the
graph shape cache, built a provenance-gated measurement battery so no number is
trusted without identity, and queued a stack of small compounding wins that sum to
a realistic 27-28 t/s short-prompt decode (ceiling ~30) with the depth-table
measurements landing as this document is written.

---

## TABLE OF CONTENTS

1. Executive summary and current numbers
2. Hardware (exact)
3. Software stack (exact)
4. The model (exact)
5. The serving of record (exact launch line, flag by flag)
6. Anatomy of one decode round (code-level chain of events)
7. Anatomy of prefill at depth
8. The walls (eight of them - evidence, attempts, verdicts)
9. Everything we tried (master results table)
10. Instrumentation, integrity laws, and tooling
11. Incident log (crashes, corruption, silent failures)
12. Open frontiers and pending owner decisions
13. File map and receipt index
14. Reproduce it yourself
15. Timeline (compressed)

---

## 1. EXECUTIVE SUMMARY AND CURRENT NUMBERS

### 1.1 The problem

Serve Qwen3.8-27B-ASCII-P1M (a 27B hybrid-attention model with a 1-layer MTP
draft block, built for 200k+ context) on a 4-die AMD Vega-10 server at maximum
decode and prefill throughput, at 200k-token context, with quantized KV cache
(q4_0) and speculative decoding (draft-mtp).

The hardware makes this interesting: each die is a Vega-10 (gfx900) with 8 GiB,
wave64, no matrix cores, DP4A-emulated int8 dot products, and all four dies talk
over PCIe Gen3 through two PLX PM8533 switches - not NVLink, not XGMI, not
infinity fabric. Every tensor-parallel synchronization crosses that PCIe fabric.

### 1.2 Current numbers (serving of record, 2026-09-24)

| metric | value | condition |
|---|---|---|
| decode | **24.55 t/s** | 8k-10k primed context, temp 0, draft-mtp on, TP4+RCCL+fa40 arm (E-137 ratchet) |
| decode (old stamp) | 23.23 @10k / 23.34 @200k | E-105 lineage, superseded 09-24 |
| decode, no speculative, no graph tricks | 12.24 t/s | reference row in the baseline (shows what MTP+optimizations buy) |
| prefill | 217.71 t/s @2k | 82.17 t/s @8k (PLOG-101) |
| prefill at depth | 170.35 @10k / 29.41 @51k | U1 depth cells, measured 09-24 |
| MTP acceptance | 0.66-0.68 | temp 0, mean accepted length ~3.04 of a 3-token chain |
| untraced reference | 12.24 t/s | no-spec baseline row |

### 1.3 Where the decode round actually goes (the structural fact)

Short-prompt decode round = ~40.7 ms at 24.55 t/s:

| component | ms/round | share | evidence |
|---|---|---|---|
| **136 TP4 allreduce boundaries** (RCCL ring, per-layer sync) | **~29-31** | **~70-75%** | W8 receipt: per-die medians 128.8/151.9/191.6/210.8 us; the wall pays the slowest die (136 x 210.8 us = 28.7 ms on die 1) |
| MMVQ weight-streaming (all decode matvecs, 3.08 GB/die) | ~8-10 | ~20-25% | W23: implied per-type BW 39.2-75.0 GB/s vs ~327 available |
| attention (fa40 var-11 direct arm) + sampler + host | ~2-3 | ~5% | W17: -37.5%/launch vs f16-pool arm, bit-exact |
| KV pool conversion (f16 pool -> tiles) | 0 | 0% | eliminated by the fa40 arm (`need_f16=false`); was 3.98 us/1k tokens/launch (0.02-0.6% of wall) |

So: **the dominant cost is inter-die synchronization, not compute.** The compute
side has been squeezed hard; the sync side is structural (tensor-parallelism
requires 2 allreduces per layer x 65 layers + catchup + draft = 136/round), with
one live attack vector: per-die clock/power asymmetry (W22: the slowest die's
per-boundary cost is 59-141 us of arrival skew above the ring transport floor of
~69.5 us - clock floors are queued to compress it).

### 1.4 The prospect stack toward 30 t/s (owner's target)

| prospect | mechanism | expected | status |
|---|---|---|---|
| W22 C1 per-die clock floors | compress boundary arrival skew; also cures the thermal soak decay | +2.4-4.8% decode | spec'd, queued (root step required) |
| W23 env flips (IQ4XS_SHARE + IQ3 S2R gates) | MMVQ pool -3.4 ms | +2-3% | spec'd, queued (env-only) |
| ub1024 re-promotion | decode penalty vanished on the fa40 config (E-139a) | +1-4% | dedicated window queued |
| W19 shape-cache | draft graph replay, ~3 ms/round | +2-3% | fix merged, window queued |
| w28nmax chain=4 | mean_len 3.04 -> ~3.6+ if acceptance holds | up to +20% (acceptance-risk) | wired as battery window 5 |

Compounded best case ~28.5-30; realistic if half deliver: ~27-28. The 30+ ceiling
needs either the PP re-evaluation (W29, in flight) or the owner-guarded weight-bit
reduction (L7). On the deep axis (200k context), the fa40 lever predicts ~6 -> 8.5+
t/s and is being measured right now by the U1 window.

---

## 2. HARDWARE (exact)

### 2.1 Compute: 4x AMD Vega-10 dies

- **Card:** `Vega 10 [Radeon Pro V340/Instinct MI25x2]`, device `0x6864`, SKU `D0531800`
  (rocm-smi --showproductname). Four cards in one chassis; the campaign calls them
  "dies". gfx900 (Vega 10 ISA).
- **ISA class:** wave64 (no wave32), **no matrix cores (no MFMA/WMMA)** - all int8
  dot products are DP4A-emulated on the VALU pipes; GCN-style in-order scalar+vector.
  This is why every kernel decision here differs from mainstream RDNA/CDNA advice.
- **VRAM:** 8 GiB per die (`mem_info_vram_total` = 8,576,157,376 B). NOTE: early
  campaign notes assumed 16 GiB - corrected 09-24 (W28). Live envelope under the
  of-record server: ~6,978 / 8,176 MiB used per die (~1.2 GiB/die free).
- **Die PCI addresses (stable across boots):** `0000:05:00.0`, `0000:08:00.0`,
  `0000:0d:00.0`, `0000:10:00.0`. **Card NUMBERS (card0..card4) reshuffle every
  boot** - an NVIDIA display GPU shares the box and has landed on card1; all
  campaign tooling resolves dies via PCI, never via card index (this bit us once:
  the thermal sampler hardcoded card0/1/2 and spent weeks sampling the NVIDIA GPU).
- **Fabric:** each die sits x16 Gen3 behind a PLX PM8533 switch; two switch
  complexes (`00:01.0 -> 01:00.0` feeding 05:00/08:00, `00:01.1 -> 09:00.0` feeding
  0d:00/10:00), uplinks x8. `iommu=pt`, no ACS. NUMA: single node (all dies -1).
- **Thermal character (W18):** far dies (0d:00, 10:00) latch into sclk throttle
  bands (991/775/560 MHz) under sustained load while junction stays flat at
  83-85 C on ALL dies - the trigger sensor is unexposed (likely VRM/board). Mem
  temps reach 84-95 C. Recovery by ~13 min passive cooling. This costs 26% of
  decode over ~20 min of sustained serving and is a first-class campaign variable.

### 2.2 Host

- **CPU:** Intel i7-12700K (12th gen, 8 P-cores + 4 E-cores = 20 threads), single
  NUMA node. Server runs with `-t 8` (8 threads), unpinned (W22 measured pinning
  as a 0-1% class lever, queued as C2).
- **Board/RAM:** MSI MS-7D28 (Z690), 62 GB system RAM.
- **Kernel:** `7.0.0-31-generic` (Ubuntu 24.04 base).
- **Kernel cmdline:** `iommu=pt pci=realloc amdgpu.ppfeaturemask=0xffffffff`
  (+ `vt.handoff` standard). `ppfeaturemask=0xffffffff` unlocks the pp_od/pp_dpm
  clock-control surfaces the W22 clock-floor experiment needs.
- **amdgpu module option:** `noretry=1` (runtime + `/etc/modprobe.d/amdgpu-noretry.conf`)
  - added after the SVM death forensics (E-120); see section 11.

### 2.3 Measured fabric facts (W8/W9/W10 - the boundary physics)

- RCCL ring allreduce at decode sizes has a **transport-free floor of ~69.5 us per
  boundary** (80 KB payload, 4-rank single cooperative kernel) on the FAST die.
- The per-boundary cost seen served = ring floor + **T(die)** where T = 59 us
  (die 3, fast) to 141 us (die 1, slow). W10 proved the per-boundary compute term
  scales multiplicatively with the die's own compute rate (t_d = 0.893/0.846/0.964/1.000
  across dies, to 0.1%) - the medians are wait REDISTRIBUTION, not transport.
- Host enqueue is a no-op class (0.2 us). NUMA/pinning is not the mechanism
  (single NUMA; W22 refuted the host-affinity hypothesis).
- **RCCL refuses single-die multi-rank** (`ncclInvalidUsage`, "Duplicate GPU
  detected") - one rank per GPU enforced per communicator (W8 section 1.1).

---

## 3. SOFTWARE STACK (exact)

| layer | version / path | notes |
|---|---|---|
| OS | Ubuntu 24.04, kernel `7.0.0-31-generic` | |
| ROCm | **6.2.0-66** (`/opt/rocm-6.2.0`) | 6.2-era headers lack `__hip_fp8_e4m3` for gfx900 (fp8 gate needed, commit 24e9e5dc4) |
| hipcc | **AMD clang version 18.0.0git** (roc-6.2.0) | **THE MISCOMPILE** - see wall 2 |
| RCCL | **2.20.5.60200-66~24.04** | one rank per GPU enforced; ring; `GGML_CUDA_ALLREDUCE=nccl` opt-in |
| llama.cpp | branch `amd/v340-port-v2` @ `a710244d4` (private port line; NOT upstream master) | carries the campaign's kernel/context work |
| build | cmake at `/home/chris/opt/cmake/bin/cmake`; flags: `GGML_HIP=ON CMAKE_BUILD_TYPE=Release GGML_NATIVE=ON CMAKE_HIP_ARCHITECTURES=gfx900 GGML_HIP_RCCL=ON LLAMA_CURL=OFF`; build dir `build-hip` | BUILD-EXIT:$? law: never pipe builds through tail (a silent `cmake: command not found` once read as success) |
| sampler | rocm-smi 6.2 + direct sysfs (hwmon + pp_dpm_sclk) | the 18-col thermal sideband sampler keys on PCI (E-136 fix) |

Campaign-relevant llama.cpp deltas vs upstream master (this is a PRIVATE PORT -
upstream has none of this): the fa40 q4_0-direct attention gate (fattn-tile.cuh),
the MMVQ compile-time dispatch template + share gates (mmvq.cu), the LLAMA_DRAFT_*
fast-path envs, per-shard on-device argmax, the meta-backend replay/heal machinery
additions, the RCCL allreduce opt-in, and the hybrid-MTP model support the model
requires.

---

## 4. THE MODEL (exact)

**File:** `/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf`

- 27B params, 65 layers, hidden dim 5120, **hybrid attention**: only **16 of the
  65 layers carry KV cache** (qwen35 MTP-style hybrid filter, interval 4; the rest
  are linear/attention-free layers - this is why 200k context fits in 8 GiB/die).
- **MTP block:** 1 `nextn` layer (`nextn_predict_layers=1` in the GGUF) - the draft
  is a single transformer layer + head that proposes 3 tokens per step (chain),
  verified greedily by the full model. The draft context caches ONLY that 1 layer's
  KV: 195.3 MiB at f16 / 200k (W28).
- **Served weight-quant mix (per die at TP4, W23 census - matches E-115a to 0.6%):**

| type | tensors | GB/die | served ms/round | implied GB/s | share gate |
|---|---|---|---|---|---|
| IQ3_S | 123 | 0.890 | 17.6 | 50.6 | ON |
| IQ4_XS | 94 | 0.818 | 10.9 | **75.0** | OFF (E-104; re-open queued W23-C1) |
| IQ3_XXS | 84 | 0.615 | 14.8 | 41.6 | ON |
| Q4_K | 88 | 0.415 | 8.5 | 48.8 | ON |
| Q6_K | 13 | 0.145 | 3.7 | 39.2 | ON |
| Q3_K/Q5_K/rest | 478 | 0.18 | ~1.5 | ~45-50 | ON |

- **KV:** q4_0 for both K and V on the 16 KV layers (`-ctk q4_0 -ctv q4_0`), read
  IN-KERNEL by the fa40 decode attention (no f16 staging pool on the hot path).
  The DRAFT context's KV is f16 (defaults; `-ctkd/-ctvd` untouched) - W28 census.
- **Context:** served `-c 200000` (ceiling 262144). At 200k the KV pool + weights
  fill ~6.8 of 8 GiB per die.

---

## 5. THE SERVING OF RECORD (exact launch line)

From `/home/chris/launch_tp3_200k.sh` (Chris-signed config + the optimization set):

```bash
GGML_CUDA_ALLREDUCE=nccl \            # RCCL ring allreduce (+25-27% decode vs butterfly, E-090; numerics class changed by signed sign-off E-090/E-094)
GGML_CUDA_MMVQ_IQ3S_SHARE=1 \         # MMVQ share gates: share one y-read across
GGML_CUDA_MMVQ_IQ3XXS_SHARE=1 \       #   token-columns; bit-exact-gated per type
GGML_CUDA_MMVQ_Q3K_SHARE=1 \
GGML_CUDA_MMVQ_Q4K_SHARE=1 \
GGML_CUDA_MMVQ_Q5K_SHARE=1 \
GGML_CUDA_MMVQ_Q6K_SHARE=1 \
LLAMA_DRAFT_FAST_TOPK=1 \             # draft top-k fast path
LLAMA_DRAFT_PACKED_GET=1 \            # packed draft-tensor fetch
LLAMA_DRAFT_LIGHT_SYNC=1 \            # lighter draft-ctx sync
LLAMA_VERIFY_ROW_SAMPLING=1 \         # row-sampling verify path
LLAMA_ASYNC_INPUT=1 \                 # async input copy
GGML_CUDA_FATTN_TILE_Q40_DIRECT=1 \   # THE FA40 ARM: attention reads q4_0 KV in-kernel (E-137, +5.0% served paired)
GGML_PINNED_DEV_COPY=1 \
HIP_VISIBLE_DEVICES=0,1,2,3 /media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin/llama-server \
  -m /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf \
  -ngl 999 -sm tensor -c 200000 --batch-size 512 --ubatch-size 512 \
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
  --device ROCm0,ROCm1,ROCm2,ROCm3 \
  --port 8080 -t 8
```

Flag-by-flag rationale and validation receipts: RCCL allreduce (E-090/E-094,
+25-27%), share gates (E-095/E-104, bit-exact gate per type), TP4 4-rank
(E-105, 5/5 GREEN, Chris-approved), fa40 arm (E-137, paired +5.0%, oracle
bit-exact W17). Note `-sm tensor` is the sharding mode (4-way tensor parallel);
`--device ROCm0..3` names the four dies; batch/ubatch 512 is the decode-optimized
geometry (1024 is a queued re-promotion candidate - E-139a found its old decode
penalty vanished on the fa40 config).

The validation battery (`docs/amd-port/tests/guard_battery.py`, baseline
`docs/amd-port/tests/baseline_tp3_200k.json`) gates every change: prefill_guard,
decode_guard (vs the ratcheted 24.55), mtp_canary (accept >= 0.63),
determinism_guard (byte-identical double-run), needle_recall_guard (3/3 depths),
all stamped with a provenance block (clean tree + commit + binary sha256 +
CMakeCache sha256 + launch config) that FAILS CLOSED.

---

## 6. ANATOMY OF ONE DECODE ROUND (the full chain of events, code-level)

A "round" = one generation step for the batch (M<=4 tokens at decode), including
the draft chain and verification. At 24.55 t/s the round is ~40.7 ms. Here is
exactly what happens, with the code that does it.

### 6.1 Server -> decode entry

```
tools/server/server-context.cpp  server_context_impl::update_slots()
  -> server_context_impl::decode(int&, int, llama_batch&)          [crash bt frame #11]
  -> common_speculative_process() / common_speculative_impl_draft_mtp::process()
```

The draft-mtp flow per round: the **draft context** (a separate llama_context
owning only the 1 nextn layer) runs first, proposing up to 3 tokens (the chain);
the **target context** then runs the full model over [draft tokens + the real
token] in one ubatch, verifying each proposal greedily (accepted iff it matches
the target's argmax at that position). The crash backtrace we healed (section 11)
shows the exact frame chain: `update_slots -> decode -> llama_decode ->
process_ubatch -> graph_compute -> sched_graph_compute_async ->
ggml_backend_meta_graph_compute`.

### 6.2 Graph build / re-entry - the shape cache (src/llama-context.cpp:226-242, 1613-1671)

```cpp
// src/llama-context.cpp:226-242 (constructor, draft ctx included)
{
    // 1 = on (2 slots for the catchup/step shape pair), 2-4 = slot count
    const char * LLAMA_DRAFT_SHAPE_CACHE = getenv("LLAMA_DRAFT_SHAPE_CACHE");
    const int n_slots = LLAMA_DRAFT_SHAPE_CACHE ? atoi(LLAMA_DRAFT_SHAPE_CACHE) : 0;
    if (n_slots >= 1) {
        gf_res_shape.resize(std::min(n_slots == 1 ? 2 : n_slots, 4));
        // WARN, not INFO: the server routes library logs through
        // common_log_default_callback, which drops ggml INFO at the served
        // verbosity (3) - engagement evidence must survive it (E-117 law)
        LLAMA_LOG_WARN("%s: draft shape cache enabled (%d slots)\n", __func__, ...);
    }
}
```

The cache stores up to 2 fully-built graphs (the draft's catchup shape and step
shape) and re-enters them instead of rebuilding. The re-entry branch
(llama-context.cpp:1613-1671, `can_reuse && !sched_is_res`) re-allocs the stored
graph via `ggml_backend_sched_alloc_graph`. **This was the site of a real served
crash** (section 11, E-138): a re-entered graph's tensors keep `data != NULL`, so
`ggml_gallocr_init_tensor` never re-initializes them, while the meta backend's
double-buffered container cycling clears the other container's registrations -
the rebuild mapping then reads NULL and asserts. Fixed by lazy re-registration
(fail-open), see 6.4.

### 6.3 The meta backend compute + heal (ggml/src/ggml-backend-meta.cpp)

```cpp
// ggml-backend-meta.cpp:1898-1913 (per compute with changed uid: cycle the
// double-buffered simple-tensor containers, clearing the one the NEXT graph
// will need - the crash mechanism)
for (ggml_backend_buffer_t buf : used_buffers) {
    ggml_backend_meta_buffer_context * buf_ctx = ...;
    buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
    for (ggml_context_ptr & ctx : stc.ctxs) ggml_reset(ctx.get());
    stc.simple_tensors.clear();
    buf_ctx->heal_warned = false;
}
...
// ggml-backend-meta.cpp:1928-1929 (PRE-FIX): the rebuild mapping hit NULL here
// GGML_ASSERT(bcj.nodes[i]);   <-- served crash 09-24, draft-mtp path
// POST-FIX (a7a1f0fda, merged): ggml_backend_meta_buffer_simple_tensor()
// lazily RE-REGISTERS the missing tensor (fail-open) instead of returning
// NULL; the uid memo still hits, so device-graph replay (the cache's win)
// is preserved; heals converge to zero steady-state.
```

### 6.4 Per-layer compute - the weight matvecs (MMVQ)

Every decode matvec against a quantized weight runs through one templated kernel
(ggml/src/ggml-cuda/mmvq.cu:475-479):

```cpp
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false,
          bool ALN = false, bool S2R = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*..., 1)
static __global__ void mul_mat_vec_q(const void * vx_ptr, ..., float * dst_ptr, ...);
```

Compile-time dispatch (`switch_type` / `launch_variant`) selects the instantiation
from the env share gates (per-type, E-095/E-104) and the s2r streaming flag
(E-117 fixed a wiring defect where the flag was dropped in dispatch - the gates
were no-ops served). Engagement is provable: the gate lines print at WARN because
the served log filter drops ggml INFO at verbosity 3 (the E-117 law that now
applies to EVERY new gate in this port). Per-type served rates span 39-75 GB/s
(section 4 table) against ~327 GB/s available - the gap is latency starvation
(1-2 kbx loop iterations/lane at K=5120; register-capped occupancy 8/40 waves),
NOT bandwidth saturation (W23; the wide-s2r hoist candidate measured NULL, W25 -
the compiler already hoists).

### 6.5 Per-layer attention - the fa40 direct arm

16 of 65 layers run flash attention over q4_0 KV. Since E-137 the decode path
dequantizes q4_0 KV IN-KERNEL into the same shared half2 tiles the f16 path
produced (no f16 staging pool, `need_f16=false`):

```cpp
// ggml/src/ggml-cuda/fattn-tile.cuh:1609-1625 (served decode arm)
if constexpr (DKQ == 256 && DV == 256) {
    static const bool q40_direct = getenv("GGML_CUDA_FATTN_TILE_Q40_DIRECT") != nullptr;
    if (q40_direct && use_gqa_opt && Q->ne[1] <= 4 && K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0) {
        ...
        GGML_LOG_WARN("ggml-cuda: GGML_CUDA_FATTN_TILE_Q40_DIRECT=1, fattn tile ncols1=4 ncols2=2 q4_0 KV in-kernel ...");
        // W17 fa40-codegen: variant 11 (scalar half2 shared-tile stores).
        // The int2/int4 copy-store granules of ggml_cuda_memcpy_1 corrupt
        // the tile in this instantiation on gfx900 (hipcc clang 18) at
        // every -O level; scalar stores are bit-exact (W17 receipt).
        launch_fattn<DV, 4, 2>(ctx, dst, flash_attn_tile<DKQ, DV, 4, 2, use_logit_softcap, 11>, ...);
        return;
    }
}
```

The `11` template parameter is the codegen-workaround variant: **scalar half2
shared-tile stores**. This exists because hipcc (AMD clang 18, ROCm 6.2.0) MISCOMPILES
every 8-byte shared-tile store granule (`ggml_cuda_memcpy_1<8>`) in this
instantiation on gfx900 - the K tile is corrupt immediately after the load, at
every optimization level, across 8 restructure attempts (inline, volatile,
launch_bounds, unroll, half-K, optnone - all produce garbage). Variant 11 (scalar
stores) is bit-exact at every depth including 200k AND runs -37.5% per launch
(W17 receipt; the direct path also deletes the f16 pool conversion entirely).
Promotion: paired served +5.0%/+4.3% (E-137), baseline ratcheted 23.23 -> 24.55.

### 6.6 Per-layer synchronization - the wall

Each layer ends with an allreduce across the 4 ranks. The butterfly allreduce
(llama.cpp's default) was replaced by RCCL ring via:

```cpp
// ggml/src/ggml-cuda/ggml-cuda.cu:1473 (opt-in)
const char * env = getenv("GGML_CUDA_ALLREDUCE");   // "nccl" -> RCCL ring
```

That swap alone was +25-27% decode (E-090) - the numerics class changed (fp sum
order differs, bounded dust) and was Chris-signed. But it exposed the structural
cost: **136 boundaries/round x per-boundary cost, where the WALL pays the slowest
die** (die 1: 210.8 us/boundary = 28.7-31.2 ms/round; die 3: 137.5 us = 18.7 ms).
W8/W10/W22 decomposition: per-boundary cost = ring transport floor (~69.5 us,
PCIe Gen3 x8 through PLX switches) + arrival skew driven by each die's compute
clock state (t_d = 0.893/0.846/0.964/1.000, multiplicative - NOT host, NOT NUMA,
NOT transport, NOT channel count: chan4 failed -12.6% served, E-119). The queued
W22-C1 clock floors compress the skew term. Count reduction and same-tensor
clustering are PROVEN NEGATIVE (W8 section 3: the 2-cuts-per-layer structure is
forced by the sharding table; no legal clustering candidate exists).

### 6.7 Sampler + draft chain

Sampling runs on CPU (`backend sampling not supported with SPLIT_MODE_TENSOR`,
server log - a known TP-mode property). The draft chain: 1 nextn layer proposes
3 tokens/step; proposal selection is greedy top-1 (optionally per-shard on-device
argmax via LLAMA_DRAFT_ONDEVICE_ARGMAX - an exact-equivalence contract,
src/llama-ext.h:113-120, bandwidth lever only). Acceptance is head-calibration
bound: ~0.66-0.68 (W28 - the draft KV is ALREADY f16, so no precision lever
exists; the verifier's q4_0 noise is shared conditioning and cancels).

---

## 7. ANATOMY OF PREFILL AT DEPTH

Prefill (prompt processing) collapses with depth: 218 t/s @2k, 170.35 @10k
(U1, measured), 29.41 @51k (U1, measured), 13.9 @65k (E-124, served).

**W24 closed the decomposition with zero free parameters:**

```
cost per token c(D) = 3.73 + 0.42 * D/1000  [ms/token]
```

- The linear floor 3.73 ms/token matches the W13 census classes (GEMM 41% +
  nccl 17% + other 12%; host overhead <3.4%).
- The quadratic term is the full-attention row-scan over the KV at depth - and
  it is LEGITIMATE: the scan rate per row-KV-token is identical between decode
  and prefill instances (25.2 ns/row-KV-token both, W24 A2). Attention FLOPs
  scale O(N^2); that part cannot be removed.
- The **excess** is the f16-pool read amplification inside that scan: 2048 vs
  288 bytes per KV-token (7.1x) - exactly the cut the fa40 arm made on decode
  (-37.5%, bit-exact). Plus a pool-specific x1.4 L2 superlinearity beyond 36k.
- The served 50k point sits 2.29x over the fresh model - that multiplier is the
  W18 thermal soak (the cell was VOID-gated at junction 96 C), i.e. W22's
  problem, not the arm's.

**The fix is designed and implemented, awaiting bench (W27, worktree
`wt-q40prefill`, branch `amd/q40-prefill`)**: a second env gate
`GGML_CUDA_FATTN_TILE_Q40_PREFILL` beside the decode gate (fattn-tile.cuh:1627-1644),
lifting the `Q->ne[1] <= 4` restriction for prefill M, launching var-11 at the
prefill instance `<256,256,8,2>` with `need_f16=false`; the decode gate stays
byte-identical as a canary. Predicted: prefill 170->199 @10k, 67->94 @50k,
38->57 @100k, 22->34 @199k - "the only lever that bends 200k prefill below
hour-class". Oracle + timing are staged as one command; run them when the
machine frees, then the U2 served window (kill: <+5% @10k, any GARBAGE, or a
decode_guard breach).

---

## 8. THE WALLS (evidence, attempts, verdicts)

### WALL 1 - The inter-GPU synchronization wall (the structural one)

- **What:** 136 per-layer allreduces/round over PCIe Gen3 (TP4 tensor sharding).
  Wall = slowest die: 28.7-31.2 ms of the ~41 ms round (70-75%).
- **Measured:** W8 probe (single-die RCCL floors: transport-free ring floor
  69.5 us at 80 KB; per-die medians 128.8/151.9/191.6/210.8 us; host enqueue
  0.2 us no-op). W10: per-boundary compute term is multiplicative device-side
  (t_d to 0.1%); the medians are wait redistribution. W9: isolated allreduce
  flat x1.07 - transport itself is NOT the variable.
- **Tried and SEALED (do not re-open):**
  - A2 same-tensor allreduce clustering: NEGATIVE - boundaries are data-dependent,
    one tensor each, separated by nonlinear consumers; enqueue already
    stream-pipelined (W8 section 3).
  - A3 count reduction: NEGATIVE - the 2-cuts-per-layer structure is forced by
    the sharding table + the PARTIAL cut rule; alternatives multiply weight
    streaming (the #1 compute pool) or keep the count (W8).
  - chan4 (RCCL channel count): FAILED served -12.64% @200k / -6.16% @10k (E-119).
  - tensor-split rebalance: FAILED paired -1.92/-2.33 t/s (E-125).
- **LIVE:** W22 C1 per-die clock floors (compress arrival skew; +2.4-4.8% +
  anti-decay; root step via /home/chris/set_clock_floor.sh). W29 (in flight):
  full PP-vs-TP re-evaluation under current kernels, including the tp2xpp2
  hybrid - pipeline eliminates per-layer allreduce entirely at the cost of
  serial staging; it lost in the early era but under the current kernel stack
  the trade was never re-measured.
- **Physics floor:** even at zero skew, ring transport floor x 136 = ~9.5 ms/round
  of irreducible sync + ~10 ms compute -> ~20-23 ms round -> ~43-50 t/s absolute
  ceiling for THIS fabric at TP4 short decode.

### WALL 2 - The hipcc miscompile (solved, with scars)

- **What:** AMD clang 18 (ROCm 6.2.0) miscompiles every 8-byte shared-tile store
  granule (`ggml_cuda_memcpy_1<8>`) in the q4_0-direct fattn-tile instantiation
  on gfx900: the K tile is corrupt immediately after load, at EVERY -O level.
  Symptom was -38.9% end-to-end time with GARBAGE output (6144/6144 elements
  wrong) - fail-closed at first discovery.
- **The sweep (W17/W25-era variant pack):** 9 variants - control (8B stores),
  noinline dequant, volatile LDS, noinline mad chain, half-K, launch bounds,
  unroll-1, optnone, 16B shared stores - ALL produce identical garbage at the
  K-tile stage. **Variant 11 (scalar half2 stores) is bit-exact at all depths
  (7168/10240/65536/199936) and -35..-40% per launch.** Miscompile localized to
  the store granule; upstream-report characterization table banked.
- **Code:** fattn-tile.cuh:1609-1625 (section 6.5). The arm is env-gated
  (GGML_CUDA_FATTN_TILE_Q40_DIRECT) with a WARN engagement line.
- **Lesson institutionalized:** every new kernel arm ships with an oracle mode
  and NO performance verdict is admissible without same-session bit-exactness.

### WALL 3 - Prefill-at-depth collapse (fix implemented, bench pending)

See section 7. The collapse is closed by c(D) = 3.73 + 0.42*D/1000 ms/token;
the excess over first-principles is the 7.1x f16-pool read amplification; the
q40-prefill arm (W27) eliminates it for prefill M. Staged; bench + U2 window
pending. Kill criteria: <+5% @10k, any GARBAGE, decode_guard breach.

### WALL 4 - ROCm KFD SVM machine deaths (mitigated, not solved)

- **Five hard deaths** (no panic, instant reset): 09-23 11:03/17:18/18:55,
  09-24 05:15, 09-24 07:04 - plus a 06:46 boot that lasted 19 minutes. All
  preceded by `workqueue: svm_range_deferred_list_work [amdgpu] hogged CPU`
  escalations (per-boot counters reached 4-5 before death) and clustered at
  llama-server boot/teardown cycles (allocation churn).
- **Tried:** `amdgpu noretry=1` (runtime + modprobe.d) - did NOT prevent
  deaths #4/#5. Churn spacing (75 -> 180 s inter-cell). 
- **WORKS:** the **svm-watchdog** (svm-watchdog.timer, every minute): reads the
  per-boot hog counter from the kernel log; at >= 3 it claims the campaign GPU
  lock (/tmp/campaign_gpu_boot.lock) so every pipeline pauses at its next cell
  boundary; releases after 10 quiet minutes. Live-fire proven: held all work
  through counter=5 at 08:09 with zero deaths after.
- **Structural fix named:** persistent-server battery protocol (one boot, N
  cells - W21, zero-code via --slot-save-path + POST /slots/0?action=erase);
  owner decision pending.
- **Collateral:** hard deaths corrupt git objects (two purges: fa40 WIP tip and
  earlier era) - recovered via reflog + working tree both times; snapshot pushes
  after every landing are the insurance.

### WALL 5 - The thermal soak (mechanism named, fix queued)

- **Phenomenon:** decode decays ~26% within ~20 min of sustained serving
  (23.3 -> 17.3 t/s) while prefill barely moves (-1.4%) - decode is paced by
  the throttled slowest die.
- **Mechanism (W18, zero-GPU desk, proven from banked sidebands):** thermal sclk
  power-management throttling of the far dies (0d:00/10:00): decode 23.2-23.6
  iff die-3 sclk act-mean >= ~1190 MHz (<=991 MHz duty 0-20%); 17.3-17.7 iff
  act-mean 1068-1135 (duty 44-53%). Hot-but-unthrottled cells still decode
  21.8-22.5 - hot alone is not slow, THROTTLED is slow. Junction is FLAT
  83-85 C on all dies while clocks split (the trigger sensor is unexposed -
  likely VRM/board). Recovery: ~13 min passive idle. A boot following a crash
  INHERITS the previous boot's die heat (boot-3 measured 17.6 from position 1).
- **Adopted practices:** decision cells at position-1 or after >= 5-10 min idle;
  the void_gate.py adjudicator (VOID iff die-3 sclk act-mean < 1150 MHz or
  <=991 MHz duty > 20% - a backstop, not a license); die_hot<60 pre-boot gates;
  780 s soak settles in the U1 window. The W22 clock-floor arm is the candidate
  CURE (floors prevent the throttle bands).

### WALL 6 - MTP acceptance ceiling (characterized, no served lever)

- Acceptance ~0.66-0.68 at temp 0, mean_len ~3.04 of a 3-token chain.
- **W28 census:** the draft KV is ALREADY f16 (defaults; -ctk/-ctv touch only
  the target) - the "another quant" hypothesis is moot (an A/B of two identical
  arms). Acceptance is head-calibration-bound: the 1-layer MTP head was trained
  against the original backbone; the verifier's q4_0 noise is shared conditioning
  and cancels. LLAMA_DRAFT_ONDEVICE_ARGMAX is an exact-equivalence contract
  (acceptance-neutral by design).
- **Live lever:** chain shape (--spec-draft-n-max 4, wired as battery window 5,
  content-invariance sha-checked). Cross-project evidence (ninfer, 103 rows):
  context is the biggest acceptance lever (+25 pts 10k->72k on their stack) -
  registered as a falsifiable prediction on our U1 deep cells (acceptance
  should climb above 0.68 at 100k+ depth).

### WALL 7 - MMVQ latency starvation (partially cashed)

- The 3.08 GB/die weight pool runs at 39-75 GB/s implied (per type) vs ~327
  available. W23 decomposition: stall-bound - 1-2 kbx iterations/lane at
  K=5120, register-capped occupancy (8/40 waves on q4_K/q5_K/iq4_xs), and the
  E-106 zero-compute ablation still ceilings at 70-120 GB/s - so ~2.7-4.7x of
  the 6x gap is the ACCESS PATTERN itself, the rest is decode tax.
- **Cashed/queued:** share gates (served), S2R wiring fix (E-117), the env
  window (IQ4XS_SHARE re-open + IQ3 S2R gates, queued, ~-3.4 ms), C4 (lift the
  T=1-only GLU fusion ban to T=4: -2 to -4.5 ms, code change, OPEN).
- **Refuted:** wide-s2r y-preload hoist (W25: oracle-perfect, timing NULL -
  the compiler already hoists; +0-3 regs, occupancy unchanged).

### WALL 8 - Measurement integrity (the meta-wall)

Early in the campaign a real -20% regression shipped as "-0.73% PASS" because
the baseline anchor was stale (E-111/E-112). That incident created today's
integrity stack (section 10): provenance-fail-closed batteries, arm-identity
freshness gates, paired-design law, the VOID gate, and the rule that nothing
promotes without paired served cells. Every silent-failure class we have hit
(stale binary, blind identity stamp, grep-swallowed stderr, root-run git
blindness, EOS-killed decodes) is now a fail-loud gate somewhere.

---

## 9. EVERYTHING WE TRIED (master results table)

| # | change / experiment | result | evidence |
|---|---|---|---|
| 1 | RCCL ring allreduce (vs butterfly) | **+25-27% decode - PROMOTED** | E-090/E-094 |
| 2 | TP4 4-rank (from smaller TP) | **5/5 GREEN - PROMOTED** (23.34/23.23) | E-105 |
| 3 | MMVQ share gates (6 types) | **PROMOTED** (bit-exact-gated) | E-095/E-104 |
| 4 | fa40 var-11 q4_0-direct decode attention | **+5.0%/+4.3% paired - PROMOTED, ratchet 24.55** | E-136/E-137/W17 |
| 5 | compile-time ALN/S2R dispatch (mmvq) | restored -20% regression; of-record restored | E-111/E-113 |
| 6 | S2R wiring fix (dispatch dropped the flag) | fixed + host test | E-117 |
| 7 | draft shape-cache crash fix (lazy re-registration) | merged, CI 2c suite | E-138 |
| 8 | 50k-pair tensor-split rebalance | REJECTED (-1.92/-2.33 paired) | E-125 |
| 9 | wide6 (6-token verify width) | SEALED (221-225 us/instance) | E-123 |
| 10 | tile-fp16 attention | DOWNGRADED (parity at served M) | ledger |
| 11 | hipBLASLt for dense prefill | DEAD on gfx900 | ledger |
| 12 | chan4 RCCL channels | REJECTED (-12.6% @200k) | E-119 |
| 13 | ub1024 (round 1) | prefill +4-5%, decode -14.9% -> REJECTED then | E-127/E-133 |
| 14 | ub1024 (round 2, on fa40 config) | decode penalty VANISHED (+0.28/+0.86, prefill +4.8/+5.3%) -> re-promotion window queued | E-139a |
| 15 | wide-s2r y-preload hoist | NULL (oracle 5/5, timing flat - compiler already hoists) | W25 |
| 16 | draft-shape-cache served (pre-fix) | CRASHED (assert 1929) - root-caused + fixed | E-135/E-138 |
| 17 | prefill ub1024 flag | kept as documented flag for prefill-heavy profiles | E-127 |
| 18 | per-shard on-device argmax | merged (exact-equivalence contract; bandwidth lever) | d2eb60966 |
| 19 | MMVQ S2R gates (never set served) | queued (window 1, ~-1.4 ms) | W23-C2 |
| 20 | IQ4XS_SHARE re-open | queued (window 1, ~-2.0 ms; E-104 disabled it on stale numbers) | W23-C1 |
| 21 | per-die clock floors | queued (window 4; +2.4-4.8% + anti-decay) | W22-C1 |
| 22 | taskset P-core pinning | queued (0-1% class) | W22-C2 |
| 23 | chain shape n_max=4 | wired (window 5; content-invariance checked) | W28 |
| 24 | q4_0-direct PREFILL arm | implemented, oracle staged | W24/W27 |
| 25 | PP-vs-TP re-evaluation | IN FLIGHT (W29) | - |
| 26 | depth table measurement | IN FLIGHT (U1; 10k + 50k-prefill landed) | W26 |
| 27 | svm-watchdog pause machinery | LIVE (proven at counter=5) | E-134 |
| 28 | provenance/identity/void gates | LIVE (fail-closed battery) | E-112/E-133 |
| 29 | persistent-server battery protocol | designed, staged, OWNER DECISION pending | W21 |
| 30 | weight-bit reduction | OWNER-GUARDED (the only short-prompt true-2x door) | L7 |

---

## 10. INSTRUMENTATION, INTEGRITY LAWS, AND TOOLING

### 10.1 The guard battery (docs/amd-port/tests/guard_battery.py)

Every measurement runs a 5-cell battery against the ratcheted baseline
(baseline_tp3_200k.json): prefill_guard, decode_guard (warn -5% / fail -10%),
mtp_canary (accept >= 0.63), determinism_guard (byte-identical double-run),
needle_recall_guard (3/3 depths). Over it sits the **provenance gate that fails
closed**: no result is trusted without (a) a clean tracked tree, (b) the commit
hash, (c) the binary sha256, (d) the CMakeCache sha256, (e) the full launch
config - all stamped into every result row and receipt. A 5 s thermal sideband
(18-column CSV, c0..c3 = die position in PCI order 05:00/08:00/0d:00/10:00 -
the E-136 fix that ended NVIDIA-GPU sampling) is recorded per cell.

### 10.2 The adjudication stack

- **Paired-design law:** chained windows cannot resolve sub-5% candidates -
  every verdict is interleaved R/A/A/R cells at matched positions; both pairs
  must agree in direction; all cells persist (never delete a negative).
- **void_gate.py:** adjudicates any cell's thermal sideband against the W18 rule
  (VOID iff die-3 sclk act-mean < 1150 MHz or <=991 MHz duty > 20%). A VOID
  decision cell cannot bank a promotion. Known gray band: mid-soak cells can
  squeak past - the gate is a backstop, position-1 scheduling is the primary law.
- **Arm-identity freshness gate** (in run_combined_window.sh +
  run_postreboot_hardened.sh): fails loud (exit 2) if the git stamp is empty or
  the binary predates the last commit touching CODE (docs-only commits exempt).
  Exists because a real window once served a binary built BEFORE the feature it
  was testing (E-126/E-133) - the arm was inert and the verdict void.
- **Content-invariance law:** greedy speculative decoding must produce identical
  text regardless of draft chain shape - any chain-shape window sha-checks the
  arms and hard-VOIDs on mismatch.

### 10.3 The scripts (all in /home/chris unless noted)

| script | role |
|---|---|
| launch_tp3_200k.sh | canonical serving boot (section 5) |
| run_combined_window.sh | paired served window runner: lock arbitration, PCI die gates, die_hot<60 gate, arm-identity gate, per-cell battery + logs |
| run_postreboot_hardened.sh | boot-time queue (systemd campaign-postreboot.service): verifies noretry=1 + 4 PCI dies + identity gate, runs hb cells, then fires the U1 window if /home/chris/U1_ARMED exists |
| run_u1_window.sh (+v2) | depth window: one persistent server, depth cells 10k/50k/100k/150k/199k, per-cell void-gate; v2 adds ignore_eos (the EOS kill fix) |
| run_post_u1_battery.sh | the 5-window ordered battery: w23env, ub1024, w19, w22c1 (clock floors), w28nmax - self-gating, --only/--dry-run |
| set_clock_floor.sh | root: set/clear/status per-die power_dpm_force_performance_level, PCI-resolved, readback-verified, auto-rollback |
| svm_watchdog.sh (+svm-watchdog.timer) | claims the campaign lock at svm-hog counter >= 3 (deaths hit 4-5); releases after 10 quiet min |
| run_premerge_ci.sh | CI law: tree hygiene + 10 host suites + wiring tests; CI-VERDICT must PASS before every merge |
| push_backups.sh | orphan-tree snapshot pushes to the origin fork (never duford/upstream; history push impossible - >100 MB blobs) |
| run_fa40_window.sh / run_dc_window.sh | completed paired windows (fa40 promoted; dc verdict superseded by the crash fix) |
| docs/amd-port/tests/ | guard_battery.py, void_gate.py (repo), bench_attn_real.cu, bench_mmvq_real.cu, test_meta_reentry_host.cpp, test_engagement_routing_host.cpp, host suites |

### 10.4 The integrity laws (one line each)

1. Nothing promotes without paired served cells (position-matched, void-gated).
2. No performance verdict without same-session bit-exactness (kernel arms).
3. Provenance or it did not happen (fail-closed battery).
4. Every gate fails LOUD on missing inputs; every engagement line prints at WARN
   (served log drops ggml INFO).
5. Dies are PCI addresses, never card numbers.
6. Campaign GPU chains run system-scope (sudo systemd-run, User=chris, HOME set).
7. Snapshots after every landing (hard deaths corrupt git objects - twice proven).
8. append-only ledger; never rewrite an entry, corrections are new entries.
9. The sudo password never enters any committed file, receipt, or log.
10. Single sleeps <= 150 s in monitoring loops; PID-targeted kills only.

---

## 11. INCIDENT LOG (the failures a new reviewer should know we hit)

| when | incident | root cause | resolution / law born |
|---|---|---|---|
| 09-23 | -20% decode regression shipped as "-0.73% PASS" | stale baseline anchor (E-111 misread; also an ALN hot-path tax) | E-112 correction; the provenance gate; baseline re-stamp law; CI law (E-118) |
| 09-23 | draft-cache window verdict VOID | served a binary built BEFORE the feature merged; "(built from:)" stamp empty under root | E-126/E-133: freshness gates (empty stamp / binary-vs-CODE-commit fail loud) |
| 09-23/24 | engagement lines invisible served | common_log drops ggml INFO at verbosity 3 | INFO->WARN law for all gates (E-117/E-135); host routing test |
| 09-23 x3, 09-24 x2 | hard machine deaths (instant reset, no panic) | ROCm KFD svm_range_deferred_list_work instability under allocation churn; noretry=1 insufficient | svm-watchdog (pause at counter>=3), churn spacing 180 s, persistent-server protocol named (owner decision) |
| 09-24 06:4x | git objects emptied (2x, at hard deaths) | writeback loss on hard reset | reflog + working-tree recovery; snapshot-push law reinforced |
| 09-24 06:46 boot | postreboot queue cells failed silently | queue's grep pipe swallowed battery stderr | tee to <cell>.battery_full before filtering (E-134) |
| 09-24 07:5x | dc window cells Permission denied | dead-boot ROOT run left root-owned log files in /home/chris | chowned 138 files; law: never mix root-run and user-run outputs; chown after any root window |
| 09-24 08:0x | freshness gate false-tripped by a docs-only commit | gate compared binary mtime to ANY HEAD commit | gate keys on last commit touching code paths (E-138) |
| 09-24 11:0x | ratchet silently no-opped | json ratchet walked top-level keys only; the field lives at cells/decode_8k_10k | fixed (5ad25e902); the guard was gating vs the stale 23.23 |
| 09-24 11:2x | U1 50k decode measured nothing | temp-0 EOS right after depth filler -> 1-token generation | ignore_eos in run_u1_window_v2.sh; live-analyst marks EOS-void cells; deep rerun if >=2 die |
| 09-24 | fa40 first attempt (-38.9%) had GARBAGE output | hipcc 8B-store miscompile | variant sweep -> var-11 scalar stores (bit-exact); W17 characterization banked |
| 09-24 | served crash in shape cache (assert 1929) | re-entered persistent graph lost registrations in the meta backend's container cycling | lazy re-registration, fail-open; host suite 2c (E-138) |
| 09-24 | thermal sampler sampled the NVIDIA display GPU for weeks | card0/1/2 hardcode; card1 = NVIDIA after reshuffle | PCI-keyed sysfs sampler, 18-col (E-136) |
| 09-24 | fa40 oracle first "bit-exact" run was vs a stale replica | replica harness != real kernel | law: real-kernel benches link THEIR tree's libggml-hip; replica harnesses inadmissible |

---

## 12. OPEN FRONTIERS AND PENDING DECISIONS

1. **U1 depth table** (running): the first measured decode-at-depth table.
   Watch: (a) do deep cells land UNDER the E-124 law (the fa40 deep signature),
   (b) does acceptance rise with depth (cross-project L4 prediction).
2. **W27 q40-prefill** (staged): bench on lock-free, then the U2 served window.
   Pre-fill prize: 22->34 t/s @199k.
3. **Post-U1 battery** (staged): 5 windows, ~8 h. Promotion candidates: W23 env
   flips, ub1024, shape cache, clock floors, chain=4.
4. **W29 PP re-evaluation** (in flight): the structural question - pipeline
   eliminates the 136-sync wall; it lost in the early era, the trade was never
   re-measured under current kernels. Includes the tp2xpp2 hybrid.
5. **OWNER DECISIONS:** persistent-server tiering (W21 section 8: screening vs
   arbitration tiers, Phase-1 shadow cross-calibration); weight-bit reduction
   (L7, the only short-prompt true-2x door, changes exactness class); deep-cell
   rerun authorization (v2 script) if >=2 U1 decodes die to EOS.
6. **Rejected-and-stay-rejected without new evidence:** tensor-split, wide6,
   tile-fp16, hipBLASLt, chan4, boundary count/clustering, MMVQ wide-s2r.

---

## 13. FILE MAP AND RECEIPT INDEX

```
docs/amd-port/
  OPTIMIZATION_PLAN_TP3_200K.md      <- the append-only ledger, E-001..E-139+ (214+ entries; AUTHORITATIVE)
  CAMPAIGN_DOSSIER.md                <- this document
  tests/                             <- guard_battery.py, void_gate.py, all host suites + benches
  scripts/                           <- soak_timeline.py, void helpers, persistent-battery prototype
  results/                           <- 498 files: W1..W28 receipts, thermal sidebands, probe logs,
                                        census CSVs, window jsonl receipts (every number cites one)
/home/chris/
  launch_tp3_200k.sh                 <- canonical serving boot
  run_combined_window.sh             <- paired window runner (identity + freshness gates)
  run_postreboot_hardened.sh         <- boot queue (systemd: campaign-postreboot.service)
  run_u1_window.sh / _v2.sh          <- depth window (v2: ignore_eos)
  run_post_u1_battery.sh             <- the 5-window post-U1 battery
  set_clock_floor.sh                 <- root clock-floor helper (W22 C1)
  svm_watchdog.sh                    <- hog counter -> lock claim (timer: svm-watchdog.timer)
  push_backups.sh                    <- snapshot pushes
  run_premerge_ci.sh                 <- CI law script
  hb_p*, combowin_*, u1_*            <- per-cell receipts (logs + thermal sidebands)
~/.config/autostart/zcode.desktop    <- ZCode auto-launch (app-drawer context) after any boot
/etc/systemd/system/campaign-postreboot.service  <- self-firing boot queue
/etc/systemd/system/svm-watchdog.{service,timer} <- the pause machinery
/etc/modprobe.d/amdgpu-noretry.conf  <- noretry=1
```

Receipt index (the W-series): W5 MMVQ-bw v1, W6 rungs, W7 ldsy + occupancy, W8
boundary probe, W9/W10 isolated/END-wait studies, W11 spread law, W12 W6 probes,
W13 census, W14 traces, W15 deep-census, W16 fa40 era, W17 fa40 oracle+timing,
W18 soak mechanism, W19 dc engagement, W20 void gate, W21 persistent-server,
W22 die asymmetry, W23 MMVQ bw v2, W24 prefill depth, W25 wide-s2r (null),
W26 U1 depth table (live), W27 q40-prefill (staged), W28 MTP acceptance.

---

## 14. REPRODUCE IT YOURSELF

```bash
# build (from the repo root; NEVER pipe through tail)
cd /media/chris/ssd128/llamacpp/llama.cpp
/home/chris/opt/cmake/bin/cmake -B build-hip \
  -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx900 -DGGML_HIP_RCCL=ON -DLLAMA_CURL=OFF
/home/chris/opt/cmake/bin/cmake --build build-hip -j
echo "BUILD-EXIT:$?"

# pre-merge CI (tree hygiene + 10 host suites + wiring; must print CI-VERDICT: PASS)
bash /home/chris/run_premerge_ci.sh

# serve (the of-record launch; port 8080)
bash /home/chris/launch_tp3_200k.sh

# measure one cell (paired window; the runner self-gates on lock/dies/identity)
COMBO_CELLS=10k /home/chris/run_combined_window.sh mytest "SOME_ENV=1"

# adjudicate a cell's thermal sideband (VOID gate)
python3 docs/amd-port/scripts/void_gate.py mytest=/home/chris/combowin_mytest_10k_thermal.log

# the depth table (one boot, 7-14 h; v2 carries the ignore_eos fix)
bash /home/chris/run_u1_window_v2.sh

# the post-U1 battery (5 windows, ~8 h)
bash /home/chris/run_post_u1_battery.sh            # full
bash /home/chris/run_post_u1_battery.sh --only w23env   # single window smoke

# snapshots (backup law)
/home/chris/push_backups.sh
```

Reading order for a new reviewer: this document -> the E-124/E-133/E-135/E-136/
E-137/E-138/E-139 ledger entries -> W8 (boundary physics) -> W17 (miscompile) ->
W18 (soak) -> W22/W23 (open levers) -> then the live logs under /home/chris.

---

## 15. TIMELINE (compressed)

| when | era | headline |
|---|---|---|
| 09-20/21 | TP3 bring-up | first serves, T-band work, E-027..E-058; temp program |
| 09-22 | the optimization day | parity table (E-069/E-070); RCCL allreduce +25-27% (E-090); share gates shipped (E-095); cooling upgrades; serving default promoted (E-079/E-094) |
| 09-23 morning | TP4 era | 5/5 GREEN TP4+RCCL+set, 23.34/23.23 (E-105); MMVQ BW desk (E-109); the -20% regression caught -> provenance gate (E-111/E-112) |
| 09-23 evening | boundary + depth | boundary desk seals count/clustering (E-116); s2r wiring defect fixed (E-117); CI law (E-118); deep census: round NOT flat, +1.95 ms/1k (E-124); first SVM death |
| 09-24 05-07 | hardened boot + fleet | hardened boot queue; 3 reboots; 5 deaths total -> svm-watchdog; hardened paired queue: ub1024 verdict; W18 far-die soak named |
| 09-24 06-09 | the desks era | W20 void gate, W21 persistent-server, W22 die asymmetry, W23 MMVQ v2; dc stale-binary scandal -> fixed; fa40 var-11 oracle-proven + MERGED + PROMOTED (24.55); U1 chain armed; planned reboot |
| 09-24 10-13 | boot #4 | queue 4/4 PASS on new config; ub1024 contradiction found; W24/W25/W27/W28 landed; U1 depth window marching (10k measured, 50k prefill-valid); EOS bug fixed in v2; acceptance synthesis adopted |

*End of dossier. Every claim traces to a ledger entry or receipt; when this
document and the ledger disagree, the ledger wins (append-only, authoritative).*
