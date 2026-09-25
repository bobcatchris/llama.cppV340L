# THE V340L LATENCY DOSSIER - v2 (self-contained review package)

**Repo:** llama.cpp, branch `amd/v340-port-v2`, HEAD `68654e245` era, synced to ledger
E-148 (2026-09-24 ~18:30). NOT upstream
master - this is a private port carrying campaign-specific kernel/context work, all described
fully below. **Audience:** you are (likely) an LLM receiving ONLY this document. Everything
needed is inline: full code of every hot path, all measured numbers verbatim, all failed
attempts, all incident logs. You have no filesystem access to our receipts - nothing important
is left as an external reference. **Your job:** find what we missed. The standing owner
question: short-prompt decode is 24.69 t/s (ratcheted 24.55 -> 24.69 by the ub1024
promotion, E-148); the owner believes ~30 t/s should be reachable. The cycle budget
(E-141 correction): ~24-25% of each speculative cycle is
inter-GPU allreduce latency (136 boundaries/cycle over PCIe - structurally hard to
remove, proven); ~65% is per-die weight-stream compute (MMVQ at 39-75 GB/s vs ~327
available - latency-starved, and the LARGER pool). ASCII throughout (project rule). Claims cite: ledger entry
(E-NNN in docs/amd-port/OPTIMIZATION_PLAN_TP3_200K.md, 223+ append-only entries), desk
receipt (WNN in docs/amd-port/results/), or file:line in the tree. Ledger wins on conflict.

---

## PART 1 - THE LATENCY PROBLEM, STATED PRECISELY

### 1.1 What we serve

Qwen3.8-27B-ASCII-P1M.gguf: 27B params, 65 layers, hidden 5120, HYBRID attention (only 16
of 65 layers carry KV cache - the rest are attention-free layers, qwen35 MTP-style), with a
1-layer MTP draft block (nextn_predict_layers=1) for speculative decoding. Served at
200,000-token context with q4_0 quantized KV on 4x AMD Vega-10 dies (gfx900) in
tensor-parallel mode over PCIe Gen3.

### 1.2 The numbers (2026-09-24, provenance-stamped)

| metric | value | condition |
|---|---|---|
| decode | **24.69 t/s** | 8k-10k context, temp 0, full stack (E-148 ratchet, ub1024; 24.55 = the pre-ub1024 E-137 stamp) |
| decode reference without spec | 12.24 t/s | same stack, speculative off (baseline row) |
| decode (superseded stamp) | 23.23 @10k / 23.34 @200k | E-105 lineage |
| prefill | ~224 t/s of-record (ub1024 co-metric, E-148); 217.71 @2k / 82.17 @8k | baseline was pre-ub1024 |
| prefill at depth | 170.35 @10k; 29.41 @51k | U1 cells, measured 09-24 (pre-ub1024 binary) |
| MTP acceptance | 0.66-0.68 | temp 0; mean accepted len ~3.04 of 3-token chain |
| VRAM per die under load | 6,978 / 8,176 MiB | dies are 8 GiB (8,576,157,376 B), NOT 16 |

### 1.3 The decode round budget (~40.5 ms/token at the 24.69 of-record; 40.7 at the 24.55 stamp)

| component | ms/round | share | how measured |
|---|---|---|---|
| **136 allreduce boundaries** (TP4 per-layer sync) | **~28.7-31.2 per CYCLE** | **24-25% of cycle** | W8 probe: per-die per-boundary medians 128.8/151.9/191.6/210.8 us; wall = slowest die: 136 x 210.8 us = 28.7 ms. SCALE NOTE (E-141): the wall is per speculative CYCLE (~124 ms = 3.04 tokens / 24.69 t/s), not per token |
| MMVQ weight matvecs (3.08 GB/die/CYCLE) | ~80-85 of the concurrent cycle | **~65% of cycle - THE LARGEST TERM** | W23 per-type implied GB/s (section 5.4); runs concurrently at TP width 4 |
| attention (16 KV layers, fa40 direct arm) | ~2-3 | ~5% | W17: -37.5%/launch vs the f16-pool arm |
| KV pool conversion (f16 pool -> tiles) | 0 | 0% | eliminated by fa40 (need_f16=false); was 3.98 us/1k tokens/launch |
| host/sampler | <1.4 | <3.4% | W13 census |

The structural statement (CORRECTED E-141 - the earlier "sync is 70-75%, compute is
nearly free" framing divided a per-cycle wall by a per-token time): **~65% of the cycle
is per-die weight-stream compute (MMVQ, latency-starved at 39-75 of ~327 GB/s); ~24-25%
is the allreduce wall (136 boundaries/cycle, structural); the rest is small.** Tensor
parallelism requires 2 allreduces per layer x 65 layers + 2 catch-up + 6 draft-chain
boundaries = 136/cycle; the count is STRUCTURAL (W8 section 3). CONSEQUENCE: the MMVQ
access-pattern work (section 4.4) targets the LARGER term; the sync wall's live lever is
arrival-skew compression (clock floors). PP (serial staging) removes the sync but
serializes the concurrent compute: PP4 predicts 8-9 t/s (-65%) and tp2xpp2 is
unimplementable in this tree - NO-GO x2, CLOSED by derivation (E-141/W29); the early-era
PP "loss" was an anecdote (the layer arm crashed at decode entry - PP decode was never
measured).

### 1.4 The boundary's own anatomy (why each allreduce costs 128-211 us)

Per-boundary cost = **ring transport floor** + **arrival skew**:

| term | value | evidence |
|---|---|---|
| transport-free ring floor (4 ranks, 80 KB payload, single cooperative kernel) | **~69.5 us** | W8 probe, die 3, 2000 iterations |
| host enqueue | 0.2 us | W8 (no-op class - the host is NOT the problem) |
| arrival skew T(die) | **59 us (die 3) to 141 us (die 1)** | W10: multiplicative device-side scaling, t_d = 0.893/0.846/0.964/1.000 per die (to 0.1%); the medians are WAIT REDISTRIBUTION |
| transport itself | flat x1.07 isolated | W9 - NOT the variable |
| RCCL channel count | changing it: -12.6% served - sealed | E-119 |

The skew term is CLOCK/POWER STATE (W22): the far dies throttle under load (W18), the
slowest compute die arrives last on 58% of boundaries and pays pure ring while the others
wait. Full compression would take the round to ~30 ms (~33 t/s); the realistic queued
compression (per-die clock floors) is +2.4-4.8%.

---

## PART 2 - THE EXACT STACK

| item | value |
|---|---|
| GPUs | 4x `Vega 10 [Radeon Pro V340/Instinct MI25x2]`, device 0x6864, SKU D0531800 |
| ISA | gfx900 (Vega 10): wave64 only, NO matrix cores (no MFMA/WMMA), DP4A-emulated int8 dots, GCN-style |
| VRAM | 8 GiB/die (8,576,157,376 B) - early notes saying 16 GiB were WRONG (corrected W28) |
| die PCI (stable) | 0000:05:00.0, 0000:08:00.0, 0000:0d:00.0, 0000:10:00.0 - card NUMBERS reshuffle per boot (an NVIDIA display GPU shares the box; it has landed on card1) |
| fabric | PCIe Gen3 x16 per die through 2x PLX PM8533 switches (roots 00:01.0/00:01.1, uplinks x8), iommu=pt, no ACS, single NUMA node |
| host | i7-12700K (8P+4E, 20 threads), 62 GB RAM, MSI MS-7D28 (Z690) |
| kernel | 7.0.0-31-generic; cmdline `iommu=pt pci=realloc amdgpu.ppfeaturemask=0xffffffff`; amdgpu `noretry=1` (modprobe.d) |
| ROCm | 6.2.0-66 |
| hipcc | AMD clang 18.0.0git (roc-6.2.0) - THE MISCOMPILING COMPILER |
| RCCL | 2.20.5.60200-66~24.04; refuses single-die multi-rank ("Duplicate GPU detected", ncclInvalidUsage) |
| llama.cpp | branch amd/v340-port-v2 (private port; upstream has NONE of: fa40 arm, MMVQ dispatch/share/s2r, meta replay machinery, draft fast paths, RCCL opt-in) |

Build: `/home/chris/opt/cmake/bin/cmake -B build-hip -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
-DGGML_NATIVE=ON -DCMAKE_HIP_ARCHITECTURES=gfx900 -DGGML_HIP_RCCL=ON -DLLAMA_CURL=OFF`.

---

## PART 3 - THE FULL SERVING CONFIGURATION (verbatim)

From /home/chris/launch_tp3_200k.sh:

```bash
GGML_CUDA_ALLREDUCE=nccl \
GGML_CUDA_MMVQ_IQ3S_SHARE=1 \
GGML_CUDA_MMVQ_IQ3XXS_SHARE=1 \
GGML_CUDA_MMVQ_Q3K_SHARE=1 \
GGML_CUDA_MMVQ_Q4K_SHARE=1 \
GGML_CUDA_MMVQ_Q5K_SHARE=1 \
GGML_CUDA_MMVQ_Q6K_SHARE=1 \
LLAMA_DRAFT_FAST_TOPK=1 \
LLAMA_DRAFT_PACKED_GET=1 \
LLAMA_DRAFT_LIGHT_SYNC=1 \
LLAMA_VERIFY_ROW_SAMPLING=1 \
LLAMA_ASYNC_INPUT=1 \
GGML_CUDA_FATTN_TILE_Q40_DIRECT=1 \
GGML_PINNED_DEV_COPY=1 \
HIP_VISIBLE_DEVICES=0,1,2,3 /media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin/llama-server \
  -m /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf \
  -ngl 999 -sm tensor -c 200000 --batch-size 1024 --ubatch-size 1024 \
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
  -device ROCm0,ROCm1,ROCm2,ROCm3 \
  --port 8080 -t 8
```

Every env is individually validated (receipts E-090/E-094/E-095/E-104/E-137). The 1024/1024
geometry is the E-148 promotion (battery decode +0.36/+0.33 paired, prefill co-metric
+5.02/+4.14% recovered from the server logs after the --decode-only harness gap skipped
the prefill guard). The baseline gates (docs/amd-port/tests/baseline_tp3_200k.json,
ratcheted E-137 then E-148): decode 24.69 (warn -5% / fail -10%), prefill 217.71, MTP canary accept >= 0.63, determinism double-run
byte-identical, needle recall 3/3 depths - all behind a provenance block that FAILS CLOSED
without a clean tracked tree + commit hash + binary sha256 + CMakeCache sha256 + full
launch config.

---

## PART 4 - ONE DECODE ROUND, END TO END, IN CODE

A "round" = one generation step (M<=4 tokens): the draft context proposes 3 tokens (the
MTP chain), the target context verifies them plus the real token in one ubatch. ~40.5 ms
at the 24.69 of-record. Follow the code.

### 4.1 Entry chain (the exact frame order, from a real crash backtrace)

```
server_context_impl::update_slots()                        [tools/server]
  server_context_impl::decode(int&, int, llama_batch&)
    common_speculative_process(common_speculative*, llama_batch const&)
      common_speculative_impl_draft_mtp::process(llama_batch const&)      <- draft proposals
        llama_decode()
          llama_context::decode(llama_batch const&)
            llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, ...)
              llama_context::graph_compute(ggml_cgraph*, bool)
                ggml_backend_sched_graph_compute_async()
                  ggml_backend_meta_graph_compute(ggml_backend*, ggml_cgraph*)  <- the wall executes under here
```

The draft context proposes the chain first (its own small forward), then the target
forward verifies draft tokens + real token together; accepted proposals extend the
sequence, rejected ones are discarded. Greedy (temp 0) => a proposal is accepted iff it
equals the target's argmax at that position, and the FINAL TEXT is invariant to the draft
mechanism (content-invariance law used in chain-shape experiments).

### 4.2 Graph build / re-entry / shape cache (src/llama-context.cpp)

The graph is cached by shape (the draft ctx alternates two shapes: catchup and step).
Environment gate and engagement line (constructor):

```cpp
// src/llama-context.cpp:226-242
{
    // 1 = on (2 slots for the catchup/step shape pair), 2-4 = slot count
    const char * LLAMA_DRAFT_SHAPE_CACHE = getenv("LLAMA_DRAFT_SHAPE_CACHE");
    const int n_slots = LLAMA_DRAFT_SHAPE_CACHE ? atoi(LLAMA_DRAFT_SHAPE_CACHE) : 0;

    if (n_slots >= 1) {
        // 1 = on: the default 2 slots cover the draft catchup/step pair; a
        // single slot would re-derive the default single-slot thrash
        gf_res_shape.resize(std::min(n_slots == 1 ? 2 : n_slots, 4));
        // WARN, not INFO: the server routes library logs through
        // common_log_default_callback, which drops ggml INFO at the served
        // verbosity (3) - engagement evidence must survive it (E-117 law)
        LLAMA_LOG_WARN("%s: draft shape cache enabled (%d slots)\n", __func__, (int) gf_res_shape.size());
    }
}
```

Re-entry decision (src/llama-context.cpp:1600-1680, load-bearing path verbatim):

```cpp
    llm_graph_result * res = gf_res_active();
    // in order to correctly reuse a graph, it's full topology has to be uniquely
    // determined by these parameters
    auto gparams = graph_params(res, ubatch, mctx, gtype);
    bool can_reuse = !graph_reuse_disable && res->can_reuse(gparams);

    if (!gf_res_shape.empty()) {
        // shape cache: scan the other slots with the same reuse predicate, so
        // each recurring shape (draft catchup/steps) keeps its own built graph
        for (size_t i = 0; i < gf_res_shape.size() && !can_reuse; i++) {
            auto * cand = gf_res_shape[i].get();
            if (cand == res) continue;
            if (!graph_reuse_disable && cand->can_reuse(gparams)) {
                gf_res_shape_i = i; res = cand; gparams.res = res; can_reuse = true;
            }
        }
        if (!can_reuse) {   // full miss: rebuild into the least recently used slot
            gf_res_shape_i = (gf_res_shape_i + 1) % gf_res_shape.size();
            res = gf_res_shape[gf_res_shape_i].get(); gparams.res = res;
        }
    }
    auto * gf  = res->get_gf();
    // with the shape cache the schedule may still hold another shape's graph,
    // in which case even a reused graph needs a re-split
    const bool sched_is_res = gf_res_shape.empty() || res == gf_res_shape_sched;

    if (can_reuse && sched_is_res) {
        graph_reused = true;
        if (cparams.pipeline_parallel) ggml_backend_sched_synchronize(sched.get());
        n_reused++;
    } else if (can_reuse) {
        // shape re-entry: the built graph is KEPT, only the schedule is
        // re-split for it; the graph keeps its uid so backend device graphs replay
        graph_reused = true; n_reused++;
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) { /* LLAMA_LOG_ERROR; ALLOC_FAILED */ }
        gf_res_shape_sched = res;
    } else {
        /* full rebuild: res->reset(); sched_reset + eval callback; rebuild graph */
    }
```

**The served crash this caused (FIXED):** the re-entry branch re-allocs a PERSISTENT
built graph whose tensors already have `data != NULL` - and `ggml_gallocr_init_tensor`
(ggml/src/ggml-alloc.c:969-994) only initializes tensors with `data == NULL`, so
init_tensor never re-runs for the reused graph. Meanwhile each meta compute with a
changed uid CYCLES the meta backend's double-buffered simple-tensor containers and
clears the other one - destroying the registrations of exactly the graph about to be
re-entered. The draft ctx is the first victim because its catchup/step shapes ALTERNATE
decode-over-decode (constant re-entry); the target (one stable shape) never re-enters.
The rebuild mapping then read NULL: `GGML_ASSERT(bcj.nodes[i])` at
ggml-backend-meta.cpp:1929 - a hard abort on the first draft decode of the first honest
run of this feature (it had never engaged before the log-visibility fix; Part 8.6).

**The heal (fail-open, merged a7a1f0fda):** the lookup
(ggml_backend_meta_buffer_simple_tensor) lazily re-registers the missing tensor via
ggml_backend_meta_buffer_init_tensor_impl instead of returning NULL - the 1929 assert is
unreachable by construction; the uid memo still hits after healing so device-graph replay
(the cache's actual win) is preserved; heals converge (steady-state alternation needs no
further heals); one rate-limited WARN per clear epoch names the healed tensor. Default-off
behavior byte-identical. Host-proven by docs/amd-port/tests/test_meta_reentry_host.cpp
(CI section 2c): pre-fix reproduces the abort class, post-fix ALL PASS with convergence.

**Second crash path - the feature is PARKED (E-147).** The battery's w19 window re-opened
the feature with the heal live: engagement banner 2/2 (target + draft contexts), then FOUR
successful lazy re-registrations (norm-64 simple + cache_k_l64 view, both warmup rounds),
then a hard `GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)` at
ggml-backend-meta.cpp:832 - a THIRD draft-graph shape variant arrives at the re-split with
an UNKNOWN split source axis. Deterministic 2/2 arm cells, identical timestamps; controls
PASS. The E-138 heal fixed the 1929 class; the re-split source-axis assignment is a second
unguarded hole (needs an UNKNOWN-axis fallback or axis capture at clear time). Two crash
paths for a ~3 ms lever: PARKED permanently, default-off, docket documented.

### 4.3 The meta backend (ggml/src/ggml-backend-meta.cpp) - where the wall executes

The meta backend executes the whole graph across the 4 die backends and owns the
butterfly-allreduce fallback plus the container bookkeeping. The uid-cycle that cleared
registrations (the crash mechanism, verbatim):

```cpp
// ggml-backend-meta.cpp:1898-1913
for (ggml_backend_buffer_t buf : used_buffers) {
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
    buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;     // double-buffer flip
    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
    for (ggml_context_ptr & ctx : stc.ctxs) ggml_reset(ctx.get());
    stc.simple_tensors.clear();      // <-- drops the other graph's registrations
    buf_ctx->heal_warned = false;
}
```

All 136 allreduce boundaries execute under ggml_backend_meta_graph_compute via the sched;
the comm path chosen by the env in section 4.6 runs inside this loop.

### 4.4 The weight matvecs - MMVQ (ggml/src/ggml-cuda/mmvq.cu) - ~10 ms/round

One templated kernel serves ALL decode matvecs for all weight types (3.08 GB/die/round:
IQ3_S 0.890, IQ4_XS 0.818, IQ3_XXS 0.615, Q4_K 0.415, Q6_K 0.145, Q3_K+Q5_K+rest ~0.18):

```cpp
// ggml/src/ggml-cuda/mmvq.cu:475-479
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false,
          bool ALN = false, bool S2R = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const void * vy_aln_ptr,
        const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint3 sample_ratio, const uint32_t ids_stride, const bool share) {
```

The main loop - the LATENCY STARVATION site (W23: at K=5120, blocks_per_row_x = 20 vs
blocks_per_iter = 16, so each lane gets ONE OR TWO iterations; occupancy is
register-capped at 8/40 waves - `__launch_bounds__(..., 1)`, 101-104 regs on
q4_K/q5_K/iq4_xs - so there is not enough memory parallelism in flight: 39-75 GB/s
achieved vs ~327 available):

```cpp
// mmvq.cu:573-575 (then the share branches at 577+)
const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);        // y block index aligned with kbx
    const int kqs = vdr * (tid % (qi/vdr));  // x block quant index when casting to int
```

The `share` machinery (the promoted gates) - the y-INDEPENDENT decode of the quant block
runs ONCE per (row, kbx, lane) instead of once per token (the GCN/wave64 win). IQ3_S
branch verbatim (mmvq.cu:577+):

```cpp
if (share) {
    // decode-once share: the y-independent decode runs once per (row, kbx, lane)
    if constexpr (type == GGML_TYPE_IQ3_S) {
        int dec[rows_per_cuda_block][8];
        int scale_share[rows_per_cuda_block];
        float d_share[rows_per_cuda_block];
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            vec_dot_iq3_s_q8_1_decode(vx, kbx_offset + i*stride_row_x + kbx, kqs,
                dec[i], scale_share[i], d_share[i]);
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            if constexpr (S2R) {
                const mmvq_yw8 w = vec_dot_iq3_s_q8_1_preload(&y[j*stride_col_y + kby], kqs);
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_iq3_s_q8_1_apply_u(w, dec[i], scale_share[i], d_share[i]);
                }
            } else { /* direct vec_dot path */ }
        }
    }
}
```

Per-type served rates (W23; cross-checked against the W7 REAL-kernel bench - no
bench-to-serving multiplier exists):

| type | GB/die | served ms/round | implied GB/s | gate today |
|---|---|---|---|---|
| IQ3_S | 0.890 | 17.6 | 50.6 | SHARE=1 |
| IQ4_XS | 0.818 | 10.9 | **75.0** (base 78.8 in W7) | share OFF (E-104; re-open measured NO-PROMOTE, E-147) |
| IQ3_XXS | 0.615 | 14.8 | 41.6 | SHARE=1 |
| Q4_K | 0.415 | 8.5 | 48.8 | SHARE=1 |
| Q6_K | 0.145 | 3.7 | 39.2 | SHARE=1 |

The zero-compute ablation (E-106 cfull): the exact same schedule with compute removed
still ceilings at 70-120 GB/s - ~2.7-4.7x of the 6x aggregate gap is the ACCESS PATTERN
itself. The attempted fix (wide-s2r: hoist all 4 tokens' y preloads into registers)
measured NULL (W25): the compiler already hoists (+0-3 regs, CTAs/CU unchanged). The two
remaining named cash items were both measured 09-24 evening and both came back negative:
the env window (IQ4XS_SHARE re-open + the IQ3 S2R gates, ~-3.4 ms predicted) promoted
NOTHING - +0.32% mean inside the window's own 0.28 t/s repeat spread; gates stay unset
(E-145/E-147/W33). C4 (lift the T=1-only GLU fusion ban to T=4 at ggml-cuda.cu:2648 /
mmvq.cu:1205) is DEAD: oracle-fail on iq4_xs (19449/69632 els, max_rel 0.114) and +21.8%
SLOWER on iq3_s - FMA-contraction divergence between template instantiations,
instantiation-local, not source-expressible (W30/E-144, double-derived E-146). The
70-120 GB/s access-pattern ceiling stands as the wall.

### 4.5 Attention - fattn-tile + the fa40 arm (ggml/src/ggml-cuda/fattn-tile.cuh) - ~2-3 ms/round

16 of 65 layers run flash attention over q4_0 KV. The gate (fattn-tile.cuh:1609-1625):

```cpp
    // Q4_0-direct decode arm (GGML_CUDA_FATTN_TILE_Q40_DIRECT): the tile
    // kernel dequantizes the q4_0 KV blocks to the same shared half2 tiles
    // in-kernel instead of reading the per-launch f16 pool, deleting the
    // full-depth pool conversion. Served instance shape only (ncols1=4,
    // ncols2=2), so the launch geometry and the parallel-block scan are
    // identical to the f16-pool arm.
    if constexpr (DKQ == 256 && DV == 256) {
        static const bool q40_direct = getenv("GGML_CUDA_FATTN_TILE_Q40_DIRECT") != nullptr;
        if (q40_direct && use_gqa_opt && Q->ne[1] <= 4 && K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            const int warp_size = 32;
            constexpr size_t nbytes_shared = 0;
            const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, 8, cc) / warp_size;
            const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, 8, cc);
            GGML_LOG_WARN("ggml-cuda: GGML_CUDA_FATTN_TILE_Q40_DIRECT=1, fattn tile ncols1=4 ncols2=2 q4_0 KV in-kernel (gqa_ratio %d, T %d)\n", gqa_ratio, (int) Q->ne[1]);
            // W17 fa40-codegen: variant 11 (scalar half2 shared-tile stores).
            // The int2/int4 copy-store granules of ggml_cuda_memcpy_1 corrupt
            // the tile in this instantiation on gfx900 (hipcc clang 18) at
            // every -O level; scalar stores are bit-exact (W17 receipt).
            launch_fattn<DV, 4, 2>(ctx, dst, flash_attn_tile<DKQ, DV, 4, 2, use_logit_softcap, 11>,
                nwarps, nbytes_shared, nbatch_fa, false, false, false, warp_size);
            return;
        }
    }
```

The dequant at the center of the miscompile (fattn-tile.cuh:493-514, the noinline
diagnostic variant):

```cpp
__device__ __noinline__ static void flash_attn_tile_q40_deq_block_noinline(const block_q4_0 * x, half2 * vals) {
    const half2 d2 = __half2half2(x->d);
    int v[QK4_0/8];
#pragma unroll
    for (int w = 0; w < QK4_0/8; ++w) {
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v[w], x->qs + 4*w);   // 4-byte load of 8 nibbles
    }
#pragma unroll
    for (int w = 0; w < QK4_0/8; ++w) {
        const int vlo = __vsubss4( v[w]       & 0x0F0F0F0F, 0x08080808);
        const int vhi = __vsubss4((v[w] >> 4) & 0x0F0F0F0F, 0x08080808);
        const int8_t * qlo = (const int8_t *) &vlo;
        const int8_t * qhi = (const int8_t *) &vhi;
#pragma unroll
        for (int l = 0; l < 4; l += 2) {
            vals[2*w + l/2]     = d2 * make_half2(qlo[l], qlo[l+1]);
            vals[8 + 2*w + l/2] = d2 * make_half2(qhi[l], qhi[l+1]);
        }
    }
}
```

The copy primitive whose 8-byte granule miscompiles (ggml/src/ggml-cuda/common.cuh:782+):

```cpp
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;
#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) ((char *) dst)[i] = ((const char *) src)[i];
        else if constexpr (nb_per_cpy == 2) ((short *) dst)[i] = ((const short *) src)[i];
        else if constexpr (nb_per_cpy == 4) ((int *) dst)[i] = ((const int *) src)[i];
        /* wider paths: 8/16 bytes via wider vector types */
    }
}
```

**Oracle results (W17, die 3, dst memcmp vs the f16-KV base arm, 6144 f32 elements,
depths 7168/10240/65536/199936):** variant 11 (scalar half2 stores) BIT-EXACT at all
depths - staged KT/VT tiles AND the KQ accumulator identical to the f16 reference. Every
8B-store variant and every restructure (noinline dequant, volatile LDS, noinline mad
chain, half-K, launch bounds, unroll-1, optnone, 16B shared stores) produces DUST
(6144/6144 elements garbage) diverging at the FIRST staged stage - the shared K tile is
corrupt immediately after the load. Timing: **-37.5% per launch at serve depth 7168;
-39.1% @10k; -38.5% @65k; -39.4% @199k** (rerun -34.7%, min-to-min -45.5%). Latency-bound,
not DRAM-bound (modeled 15.2x byte reduction lands as ~35-40% wall, roughly flat in
depth).

### 4.6 The allreduce - the wall itself (ggml/src/ggml-cuda/ggml-cuda.cu:1442-1530)

```cpp
// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ...
    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal.
        // HIP keeps the meta-backend butterfly: the internal pipeline is not
        // compiled there, and RCCL reorders the fp32 sum, so switching to it
        // must stay opt-in via GGML_CUDA_ALLREDUCE=nccl.
#if defined(GGML_USE_HIP)
        ggml_backend_cuda_comm_init_none(ret);       // HIP DEFAULT = butterfly
#elif defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif
    } else {
        std::string env_str(env);
        if (env_str == "nccl")          ggml_backend_cuda_comm_init_nccl(ret);
        else if (env_str == "internal") ggml_backend_cuda_comm_init_internal(ret);
        else if (env_str == "none")     ggml_backend_cuda_comm_init_none(ret);
        else { GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env); ... }
    }
    ...
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    ...
    // Size-class gate: prefill-sized boundaries follow GGML_RCCL_PREFILL; the
    // f32/bf16 classes fall back to the butterfly when the comms are not up.
    ...
}
```

**Why this is the most important code in the document:** on HIP, WITHOUT the env, every
boundary runs the meta backend's internal butterfly. Setting GGML_CUDA_ALLREDUCE=nccl
switched the boundaries to RCCL ring and gained **+25-27% decode** (E-090) - the single
largest win of the campaign, from one env var. The numerics class changed (RCCL reorders
the fp32 sum - bounded dust, ranks consistent, boot-to-boot deterministic) with owner
sign-off. A separate prefill size-class gate (GGML_RCCL_PREFILL=f32|bf16|butterfly,
E-101) selects the prefill boundary class (unset = bf16-compress, decided by data).

Per-boundary cost served (W8, 29-round exact window):

| die | per-boundary median | 136-boundary round |
|---|---|---|
| die 0 (05:00) | 128.8 us | 17.5 ms |
| die 1 (08:00) | 151.9 us | 20.7 ms |
| die 2 (0d:00) | 191.6 us | 26.1 ms |
| die 3 (10:00) | 210.8 us | **28.7 ms = the wall** |

W10's correction: the medians are WAIT REDISTRIBUTION - the per-boundary compute term
scales multiplicatively with each die's own compute rate (t_d = 0.893/0.846/0.964/1.000);
the slowest-compute die arrives last on 58% of boundaries and pays pure ring while the
others wait. W22: the arrival skew is CLOCK/POWER STATE (refuting host/NUMA/transport/
channels - each refutation has a receipt). The transport floor itself: ~69.5 us/boundary
= 9.5 ms/round at 136 boundaries - irreducible at TP4 on this fabric.

What does NOT move it (all proven): host enqueue (0.2 us no-op), NUMA/pinning (single
NUMA), transport tuning (W9 flat x1.07), channel count (chan4 -12.6% served), count
reduction (forced structure), same-tensor clustering (no legal candidate). What DOES:
compressing the skew (clock floors - the lever is still UNMEASURED: the battery window
was a structural CTRL-FAIL, the hook refused unprivileged execution; root retry queued,
E-147/E-148). Eliminating boundaries structurally is CLOSED: PP NO-GO x2 by derivation
(E-141/W29), and the reviewer's flat P2P bypass of the ring is DEAD EMPIRICALLY - all 12
ordered die-pairs canAccess=0 (driver-level refusal, explaining RCCL's own x12 enable
failures) and the host-staged-flat fallback WEDGES (W31/E-144).

### 4.7 The draft chain + sampler

The draft ctx (1 nextn layer, f16 KV - 195.3 MiB at 200k) proposes 3 tokens/step;
selection is greedy top-1, optionally per-shard on-device (LLAMA_DRAFT_ONDEVICE_ARGMAX -
exact-equivalence contract, src/llama-ext.h:113-120: per-shard argmax merged by max,
first-on-equal = same token; bandwidth lever only). Sampler on CPU ("backend sampling
not supported with SPLIT_MODE_TENSOR" - server log). Acceptance 0.66-0.68 is
head-calibration-bound (W28): the draft KV is ALREADY f16 (defaults common/common.h:340-341;
-ctk/-ctv touch only the target, common/common.cpp:1602-1603), the verifier's q4_0 noise
is shared conditioning and cancels. Chain shape (--spec-draft-n-max 4) was MEASURED and
REJECTED (battery w28nmax, E-147): paired -43.4%; the 4-token chain drafted 167 but
accepted 84 - the IDENTICAL 84 that chain-3 accepted, i.e. position-4 acceptance ~0% and
the extra slot is pure verify overhead. Head-calibration-bound is now empirical, not
inferred. The acceptance AXIS itself became measurable on 09-24: the W34 fixed-text
instrument (built, E-146) uses the server's own per-position acceptance line ("acc per
pos = (...)" at -lv 5, server-context.cpp:619-633) with frozen prompt-side token-id
cascades (identical text across arms by construction) and an A/A null cell; runner
staged, unrun. Known gap: target top-k logprobs are NOT populated on the spec path (TODO
server-context.cpp:4091). The L4 acceptance-rises-with-depth prediction stays untested
until the U1 v2 deep rerun.

---

## PART 5 - THE MEASUREMENTS (verbatim artifacts)

### 5.1 The paired window that promoted the fa40 arm (verbatim adjudication, 09-24 09:23)

```
== FA40 PAIRED VERDICT (fa40win 2026-09-24 09:23:45) ==
r1 engagement=0 verdict=PASS decode=23.38 t/s
fa1 engagement=320 verdict=PASS decode=24.55 t/s
r2 engagement=0 verdict=PASS decode=23.28 t/s
fa2 engagement=320 verdict=PASS decode=24.29 t/s
PAIR1 fa1-r1 = +1.17 t/s (+5.00%)
PAIR2 fa2-r2 = +1.01 t/s (+4.34%)
PAIRED MEAN = +1.09 t/s (+4.67%)
FA40-VERDICT: WIN
```

(engagement=320 = the WARN gate line fires per graph build in the direct arm; 0 in
controls. All 4 cells provenance-stamped: commit lineage 000a198da, binary sha
1b4bd1513070a137, tree clean. Cells r2/fa2 were VOID-gated for thermal soak - both still
agree in direction with the clean pair.)

### 5.2 The first measured depth cell (verbatim, U1, 09-24)

```
PROVENANCE: commit=2e27d757b633 tree_clean=True binary=049075824e23da47 config=canonical-200k-inline
prompt_eval: depth=10235 tokens prompt_tps=170.35 prompt_ms=60083
decode_guard: cell=u1_10k tps=23.78 ms_per_round=127.8 n_predict=128 cache_prompt=false status=MEASURED
law_check: depth=10235 law_ms_per_round=130.6 measured_ms_per_round=127.8 delta_pct=-2.2 -> LAW-OK
draft_accept: ratio=0.6800 mean_len=3.04 draft_n=125
VOID-GATE: PASS (die3 act-mean=1204MHz <=991duty=0% win=[55.1, 70.5])
OVERALL VERDICT: MEASURED (informational depth cell - no baseline exists at depth)
```

The E-124 depth law: ms/round = 129 + 1.95 x (depth-9400)/1000 - anchored on the OLD
config. The deep cells (100k/150k/199k) measure whether the fa40 arm lands UNDER that
law - that delta is the deep win, measured in the wild.

### 5.3 Thermal sideband rows (verbatim CSV - the soak signature)

```csv
timestamp,elapsed_s,c0_edge,c0_junc,c0_mem,c0_sclk,c1_edge,c1_junc,c1_mem,c1_sclk,...
2026-09-24T05:49:53.965893,263.7,85.0,87.0,86.0,1350,82.0,90.0,90.0,1500,85.0,87.0,90.0,775,...
2026-09-24T05:49:59.039492,268.7,85.0,89.0,90.0,1138,82.0,87.0,95.0,1350,85.0,88.0,92.0,991,...
```

(c0..c3 = die position in PCI order 05:00/08:00/0d:00/10:00 after the E-136 sampler fix;
the OLD 14-col format keyed on card0/1/2 which included the NVIDIA display GPU - the
far-die blind spot that hid this data for weeks.) The W18 rule these feed: **cell VOID
iff die-3 sclk act-mean < 1150 MHz or <=991 MHz duty > 20%** (decode 23.2-23.6 iff
act-mean >= ~1190; 17.3-17.7 iff duty 44-53%; hot-but-unthrottled cells still decode
21.8-22.5 - hot alone is not slow, THROTTLED is slow).

### 5.4 The ub1024 promotion (battery window, 09-24 16:50-17:20; adjudicated E-148)

The battery's machine verdict for the ub1024 window was SPLIT (decode positive, prefill
"not measured") - a harness artifact: --decode-only cells never run the prefill guard.
The desk adjudicated the prefill co-metric straight from the server logs' print_timing
lines (provenance: all 4 cells tree_clean=True, binary 0d5eb814 unchanged throughout):

```
decode (cells/decode_8k_10k):  r1=24.48  a1=24.84  a2=24.75  r2=24.42
  PAIRS: a1-r1 = +0.36   a2-r2 = +0.33   mean +0.34 t/s (+1.41%)  both-positive -> WIN
prefill (server logs, print_timing): 187.08 / 196.47 / 195.02 / 187.26 t/s
  arm pairs +5.02% / +4.14% vs predicted +4.8/+5.3%  -> INSIDE the E-139a band
acceptance: 0.6667 ctrl vs 0.6614 arm (-0.5 pts; draft chain 127 vs 126 - benign)
VOID-GATE 4/4 PASS (indep 1250/1352/1288/1230)
```

PROMOTION EXECUTED (E-148): baseline decode ratcheted 24.55 -> 24.69 on the guard-read
field (cells/decode_8k_10k/decode_per_second - the correct field this time); 1024/1024
applied to launch_tp3_200k.sh + run_combined_window.sh + run_postreboot_hardened.sh +
run_u1_window_v2.sh; args-only, no rebuild. Of-record now: decode 24.69, prefill ~224
(ub1024 co-metric), accept 0.66-0.68, TP4+RCCL+fa40+ub1024.

---

## PART 6 - THE WALLS (each: the barrier, the attempts, the verdict)

### WALL 1 - The inter-GPU synchronization wall (structural; partially attackable)

The 136 per-layer TP4 allreduces cost 28.7-31.2 ms per speculative CYCLE (~124 ms) -
24-25% of the cycle (E-141 correction; the earlier per-token "70-75%" framing was a
scale error). Proven sealed: count reduction (2 cuts/layer forced by the sharding table + PARTIAL cut rule;
alternatives multiply weight streaming) and same-tensor clustering (boundaries are
data-dependent, one tensor each, separated by nonlinear consumers; enqueue already
stream-pipelined). Proven dead: chan4 channels (-12.6% served), tensor-split rebalance
(paired -1.92/-2.33), host pinning (0-1% class), transport tuning (W9 flat). LIVE:
W22-C1 per-die clock floors (compress arrival skew: +2.4% halved, +4.8% full t_d
equalization, plus it cures the W18 soak) - the lever itself is UNMEASURED: the battery
window was a structural CTRL-FAIL (the clock hook refuses unprivileged execution), root
retry queued (E-147/E-148). CLOSED: PP NO-GO x2 by derivation (E-141/W29) - PP4
serializes the 65% concurrent compute: cycle = 4C + ~25 ms = ~350-365 ms = 8-9 t/s
(-65%), and winning would need a 3.3x per-die compute gain the banked physics forbids;
tp2xpp2 is not implementable (no composable TPxPP mode in the tree) and TP2-class
capacity cannot boot at 200k (E-032). Kernel progress SHRINKS C, which makes PP
monotonically worse. The absolute TP4
floor: ring transport 69.5 us x 136 = 9.5 ms sync + ~10 ms compute -> ~20-23 ms round ->
~43-50 t/s IF skew were fully eliminated - that is the physical ceiling of this fabric.

### WALL 2 - The hipcc miscompile (SOLVED)

AMD clang 18 (ROCm 6.2.0) miscompiles every 8-byte shared-tile store granule
(ggml_cuda_memcpy_1<8>) in the q4_0-direct fattn-tile instantiation on gfx900: the K tile
is corrupt immediately after the load, at every -O level. First discovered as -38.9%
end-to-end time with GARBAGE output - fail-closed. The variant sweep (W17): 9 candidates
- control 8B stores, noinline dequant, volatile LDS, noinline mad chain, half-K, launch
bounds, unroll-1, optnone, 16B shared stores - ALL produce identical garbage (6144/6144
elements wrong) at the FIRST staged stage. **Variant 11 (scalar half2 stores) is
bit-exact at all depths and -35..-40% per launch.** If all candidates had failed, the
characterization table was the deliverable (names the miscompiled pass for upstream).
Institutionalized: every kernel arm ships an oracle; no perf verdict without same-session
bit-exactness.

### WALL 3 - Prefill-at-depth collapse (BENCHED: 9/9 bit-exact, -40..-44% at depth; merge staged)

Law: cost/token = 3.73 + 0.42 x D/1000 ms (zero free parameters; lands on both clean
measured points). The linear floor matches W13 census (GEMM 41% + nccl 17% + other 12%;
host <3.4%). The quadratic term is legitimate (the row-scan rate per KV token is
IDENTICAL between decode and prefill instances - 25.2 ns). The excess: f16-pool read
amplification 7.1x (2048 vs 288 B/KV-token) inside the scan + x1.4 L2 superlinearity
beyond 36k (pool-specific). Fix: the q40-PREFILL arm (worktree wt-q40prefill, branch
amd/q40-prefill, commits 0d973076a/c472fa1d2/9af2edc3a) - new env
GGML_CUDA_FATTN_TILE_Q40_PREFILL as a disjoint T>4 gate beside the byte-identical decode
canary gate; var-11 at the prefill instance <256,256,8,2> with need_f16=false. Bench
staged as one command (9 oracle cells incl tail M=268 and non-multiple-of-256 n_kv, then
timing at 4 depths vs basep). Predicted: prefill 170->199 @10k, 67->94 @50k, 38->57
@100k, 22->34 @199k. Kill: <+5% @10k, any GARBAGE, decode_guard breach.

BENCHED (W27/E-143, 09-24 15:35): oracle 9/9 cells BIT-EXACT (0 els differ) at depths
7168-199936 - M=512, tail M=268, padded used-boundary, both M geometries, incl depth
199936 (the staged KQ-X mismatches were a DBG capture-region artifact; stop rule refined
to dst-is-truth). Timing (basep -> v11p, medians): d51200 550.9 -> 313.4 ms (-43.0%);
d102400 1409.2 -> 830.7 (-40.8%); d199936 3061.7 -> 1687.7 (-44.0%) - BETTER than the
W24 model (-37.5%): the prefill tile is even more read-amp-bound than decode. U2 served
window JUSTIFIED and queued. Merge-order law: merge amd/q40-prefill + rebuild + CI after
the lever battery, BEFORE the U2 served window (branch NOT yet merged as of E-148).

### WALL 4 - ROCm KFD SVM machine deaths (mitigated)

FIVE hard deaths (instant reset, no panic): 09-23 11:03/17:18/18:55, 09-24 05:15 (90 min
uptime), 09-24 07:04 (19 min uptime). Every death preceded by kernel lines:

```
workqueue: svm_range_deferred_list_work [amdgpu] hogged CPU for >10000us 4 times, ...
workqueue: svm_range_deferred_list_work [amdgpu] hogged CPU for >10000us 5 times, ...
```

with deaths at per-boot counters 4-5, clustered at llama-server boot/teardown
(allocation churn: 27B weights to 4 dies + 200k KV pool per server cycle).
`amdgpu noretry=1` did NOT prevent deaths #4/#5. The enabled boot-queue was
crash-looping the machine (fires within 600 s of every boot). FIXES: the svm-watchdog
(timer, minutely, root) reads the per-boot counter; at >= 3 it claims
/tmp/campaign_gpu_boot.lock so every queue/window/desk pauses at its next cell boundary;
releases after 10 quiet minutes. Proven live-fire: held everything through counter=5-7 on
boot #3 with zero deaths. Plus 180 s inter-cell churn spacing. Structural fix named:
persistent-server battery (one boot, N cells; slot-erase between cells needs zero server
code via --slot-save-path; W21) - owner decision pending. Collateral: hard deaths corrupt
git objects (two purges hit desk WIP tips; recovered via reflog + working tree both
times) - hence the snapshot-push law. A SIXTH, userspace death class: systemd-oomd
SIGKILLed the U1 cgroup at 14:26:55 (mid-150k-prefill) and the user terminal; it is now
DISABLED system-wide (kernel OOM-killer remains as backstop) - E-143. Corollary: oomd/kill
events leave zero-byte .o objects that cmake treats as fresh (35 repaired in a desk
build; watch for this after every crash event, E-144).

### WALL 5 - The thermal soak (mechanism proven, cure queued)

Decode decays ~26% within ~20 min of sustained serving (23.3 -> 17.3 t/s) while prefill
moves -1.4% (decode is paced by the throttled slowest die; compute-bound prefill is not).
W18 mechanism (zero-GPU desk, from banked sidebands): thermal sclk power-management
throttling of far dies (0d:00/10:00): decode 23.2-23.6 iff die-3 sclk act-mean >= ~1190
MHz (<=991 MHz duty 0-20%); 17.3-17.7 iff act-mean 1068-1135 (duty 44-53%);
hot-but-unthrottled cells still decode 21.8-22.5 (hot alone is not slow - THROTTLED is
slow). Junction FLAT 83-85 C on all dies while clocks split - the trigger sensor is
unexposed (likely VRM/board). Recovery: ~13 min passive idle (a boot after a crash
INHERITS the previous boot's die heat - boot-3 measured 17.6 from position 1). Adopted:
decision cells at position-1 or post-idle; void_gate.py adjudicator (VOID iff act-mean
< 1150 or <=991 duty > 20% - a backstop, not a license); die_hot<60 pre-boot gates; 780 s
soak settles in the U1 window. Candidate cure: the clock floors (same arm as W22-C1).

### WALL 6 - MTP acceptance ceiling (characterized; no served precision lever)

Acceptance ~0.66-0.68 (temp 0), mean_len ~3.04 of a 3-token chain. W28 census: the draft
KV is ALREADY f16 (defaults common/common.h:340-341; -ctk/-ctv touch only the target,
common/common.cpp:1602-1603; no inheritance either way) - the "another quant" hypothesis
is moot (an A/B of two identical arms). The draft ctx caches ONE nextn layer only:
195.3 MiB f16 at 200k (q4_0 would save 140 MiB for <0.1% decode - not worth the L8
lottery). Acceptance is head-calibration-bound: the 1-layer MTP head was trained against
the original backbone; the verifier's q4_0 noise is shared conditioning and cancels.
LLAMA_DRAFT_ONDEVICE_ARGMAX carries an exact-equivalence contract
(src/llama-ext.h:113-120) - acceptance-neutral by design. Cross-project evidence (ninfer
103-row study, adopted): context is the biggest acceptance lever (+25 pts 10k->72k on
their stack) - registered as a falsifiable prediction on our U1 deep cells (still
untested - the deep decodes were EOS-voided; U1 v2 rerun REQUIRED). The chain-shape
lever (--spec-draft-n-max 4) was MEASURED 09-24 evening and REJECTED decisively: paired
-43.4%, 167 drafted / 84 accepted = the identical 84 of chain-3, position-4 acceptance
~0 - the head's calibration ends at 3 tokens (E-147/W32). "Head-calibration-bound" is
therefore empirical, not inferred. The W34 fixed-text acceptance instrument (BUILT,
E-146) makes any future head-quality idea measurable: server-exposed per-position
acceptance at -lv 5 (server-context.cpp:619-633), frozen prompt-side token-id cascades
(divergence cannot enter), r1-vs-r2 A/A null; runner staged, unrun.

### WALL 7 - MMVQ latency starvation (partially cashed)

39-75 GB/s served per type vs ~327 available (section 4.4). W23 decomposition:
stall-bound - 1-2 kbx iterations/lane at K=5120, register-capped occupancy (8/40 waves),
and the E-106 zero-compute ablation still ceilings at 70-120 GB/s (the access pattern
itself). Cashed: share gates (served, promoted), the S2R wiring fix (E-117 - dispatch
dropped the s2r flag; the gates were no-ops served). Measured, NO-PROMOTE: the env
window (IQ4XS_SHARE re-open + IQ3 S2R gates, ~-3.4 ms predicted) - +0.32% mean inside
the window's own 0.28 t/s repeat spread; gates stay unset; the window's EXACT-VOID
verdict was a harness false-positive (see WALL 8) and the kernels are exonerated
(E-145/E-147/W33). Refuted: wide-s2r (W25 NULL). CLOSED DEAD: the C4 fusion-ban lift to
T=4 (oracle-fail iq4_xs, +21.8% slower iq3_s - FMA contraction, instantiation-local,
W30/E-144/E-146) and the occupancy rungs (lb2 codegen no-op; max-vgpr 80/96 arms
unbuildable - ROCm 6.2 LLVM 18 has no -mllvm -amdgpu-max-vgpr, E-144). The 70-120 GB/s
access-pattern ceiling stands as the wall's current face.

### WALL 8 - Measurement integrity (the meta-wall, solved by law)

The origin incident: a real -20% decode regression shipped as "-0.73% PASS" because the
baseline anchor was stale (E-111/E-112). Everything in section 9 exists because of that
and the silent-failure classes listed in section 8. NEWEST CLASS - the harness
false-positive (E-145/W33, 09-24): the w23env window's cells ran --decode-only, which
structurally never emits the determinism_guard line (guard_battery.py:895 skips it under
that flag); the adjudicator demanded the row anyway and compared det=None != PASS ->
"EXACT-VOID" with a misleading "text_sha256 diverged" ratchet. The kernels were
EXONERATED: all three gates route through the s2r-capable entry at the served shape (no
second E-117-class gap), the composition is host-proven bit-exact at served geometry
(W7 A4), and all four cells' decode counters are identical (draft_n=126, accepted=84,
ratio 0.66667). Fix queued: the adjudicator must distinguish det-missing from det-fail
under --decode-only. Twin lesson banked the same evening (E-148): the guard and the
ratchet must read the SAME field path - three ratchet-anchor incidents now share it
(E-137 walked the wrong path; E-148 fixed it; a battery-side assertion that
baseline-decode == the guard-printed value is named). Also banked: --decode-only skips
the prefill guard entirely - the ub1024 prefill co-metric had to be recovered from the
server logs by hand (E-148). Standing summary: nothing promotes
without paired served cells; no perf verdict without same-session bit-exactness;
provenance or it did not happen; every gate fails loud; every engagement line prints at
WARN (served log drops ggml INFO); dies are PCI addresses; chains run system-scope;
snapshots after every landing.

---

## PART 7 - EVERYTHING WE TRIED (master results table)

| # | change / experiment | result | evidence |
|---|---|---|---|
| 1 | RCCL ring allreduce (vs butterfly) | **+25-27% decode - PROMOTED** | E-090/E-094 |
| 2 | TP4 4-rank tensor parallel | **5/5 GREEN - PROMOTED** (23.34/23.23) | E-105 |
| 3 | MMVQ share gates (6 types) | **PROMOTED** (bit-exact-gated) | E-095/E-104 |
| 4 | fa40 var-11 q4_0-direct decode attention | **+5.0%/+4.3% paired - PROMOTED, ratchet 24.55** | E-136/E-137/W17 |
| 5 | compile-time ALN/S2R dispatch (mmvq) | restored a -20% regression; of-record restored | E-111/E-113 |
| 6 | S2R wiring fix (dispatch dropped the flag) | fixed + host test | E-117 |
| 7 | shape-cache crash fix (lazy re-registration) | merged; CI suite 2c | E-138 |
| 8 | 50k-pair tensor-split rebalance | REJECTED (paired -1.92/-2.33) | E-125 |
| 9 | wide6 (6-token verify width) | SEALED (221-225 us/instance) | E-123 |
| 10 | tile-fp16 attention | DOWNGRADED (parity at served M) | ledger |
| 11 | hipBLASLt dense prefill | DEAD on gfx900 | ledger |
| 12 | chan4 RCCL channels | REJECTED (-12.6% @200k served) | E-119 |
| 13 | ub1024 round 1 | prefill +4-5%, decode -14.9% -> REJECTED then | E-127/E-133 |
| 14 | ub1024 round 2 (on fa40 config) | penalty VANISHED (+0.28/+0.86 decode, +4.8/+5.3% prefill) -> re-promotion queued | E-139a |
| 15 | wide-s2r y-preload hoist | NULL (oracle 5/5, timing flat) | W25 |
| 16 | draft shape-cache served (pre-fix) | CRASHED -> root-caused + fixed | E-135/E-138 |
| 17 | prefill ub1024 flag | documented for prefill-heavy profiles | E-127 |
| 18 | per-shard on-device argmax | merged (exact-equivalence; bandwidth lever) | d2eb60966 |
| 19 | MMVQ S2R gates in serving | queued (never set; ~-1.4 ms) | W23-C2 |
| 20 | IQ4XS_SHARE re-open | queued (disabled on stale numbers; ~-2.0 ms) | W23-C1 |
| 21 | per-die clock floors | battery window CTRL-FAIL structural (hook needs root); UNMEASURED - root retry queued | W22-C1, E-147/E-148 |
| 22 | taskset P-core pinning | queued (0-1% class) | W22-C2 |
| 23 | chain shape n_max=4 (w28nmax) | REJECTED -43.4% paired: 167 drafted, 84 accepted (= chain-3's identical 84; position-4 acceptance ~0) | E-147/W32 |
| 24 | q4_0-direct PREFILL arm | **BENCHED: oracle 9/9 BIT-EXACT, -40..-44% per launch at depth; merge staged, U2 queued** | E-143/W27 |
| 25 | PP-vs-TP re-evaluation (W29) | **NO-GO x2 by derivation: PP4 = 8-9 t/s (-65%); tp2xpp2 unimplementable + TP2-class cannot boot 200k; early-era PP loss was an anecdote (crash at decode entry, decode never measured)** | E-141/W29 |
| 26 | depth table measurement (U1) | PARTIAL: 10k full 23.78; 50k/100k prefill-valid; deep decodes EOS-void; oomd-killed at 14:26 -> v2 rerun REQUIRED | E-140/E-143/W26 |
| 27 | svm-watchdog | LIVE (proven at counter 5-7) | E-134 |
| 28 | provenance/identity/void gates | LIVE (fail-closed) | E-112/E-133 |
| 29 | persistent-server battery protocol | designed; OWNER DECISION pending | W21 |
| 30 | weight-bit reduction | OWNER-GUARDED (the only short-prompt true-2x door) | L7 |
| 31 | W29 PP4 / tp2xpp2 pipeline modes | NO-GO x2, closed by derivation (zero GPU; reopen conditions registered) | E-141/W29 |
| 32 | W30 lb2 launch-bounds occupancy rung | codegen no-op (regs 84/77/112 + occupancy unchanged) | E-144 |
| 33 | W30 C4 GLU-fusion lift to T=4 | DEAD: oracle-fail iq4_xs (19449/69632 els, max_rel 0.114), +21.8% slower iq3_s; FMA-contraction, instantiation-local | E-144/E-146 |
| 34 | W30 max-vgpr occupancy rungs (80/96) | UNBUILDABLE: ROCm 6.2.0 LLVM 18 lacks -mllvm -amdgpu-max-vgpr | E-144 |
| 35 | W31 flat P2P allreduce bypass | DEAD EMPIRICALLY: 12/12 die-pairs canAccess=0 (driver refusal); host-staged fallback WEDGES | E-144/W31 |
| 36 | w23env env window (IQ4XS share + IQ3 S2R gates) | NO-PROMOTE on merits (+0.32% in 0.28 t/s spread); its EXACT-VOID was a harness false-positive - kernels exonerated | E-145/E-147/W33 |
| 37 | ub1024 round 3 (battery R/A/A/R) | **PROMOTED: decode +0.36/+0.33 (+1.41%), prefill co-metric +5.02/+4.14% (server logs); ratchet 24.55 -> 24.69; 1024/1024 canonical** | E-147/E-148 |
| 38 | draft shape-cache served post-heal (w19) | ARM-CRASH x2 deterministic at NEW site ggml-backend-meta.cpp:832 (UNKNOWN split axis on re-split) -> PARKED permanently | E-147 |
| 39 | W34 fixed-text acceptance instrument | BUILT (zero GPU): server -lv 5 per-position acceptance (server-context.cpp:619-633), frozen token-id cascades, A/A null; runner staged | E-146/W34 |
| 40 | systemd-oomd | DISABLED system-wide (SIGKILLed U1 cgroup + user terminal 14:26:55); kernel OOM-killer remains | E-143 |

---

## PART 8 - INCIDENT LOG (verbatim where it matters)

1. **The -20% regression that read as PASS.** A hot-path tax landed after the baseline
   was stamped; chained windows showed "-0.73% PASS" against the stale anchor. With the
   true anchor it was -20%. Born: the provenance gate, the CI law, baseline re-stamps.
   (E-111 misread corrected in E-112 - append-only, never rewritten.)
2. **The stale served binary.** A window tested the draft-shape-cache by env alone; the
   binary predated the feature merge. strings over the binary: zero occurrences. The arm
   was inert by construction; the verdict VOID. Born: freshness gates (binary vs last
   CODE commit; engagement strings pre-check; empty-stamp fail-loud).
3. **The blind identity stamp.** The same window ran as root; "(built from: $(git ...))"
   produced an EMPTY stamp in all 4 cells (git dubious-ownership under root). The
   arm-identity law was blind exactly when it mattered. Born: safe.directory flags,
   CODE_TS-based freshness, User=chris for every chain.
4. **The grep that ate the evidence.** The hardened queue's battery output went through
   `| grep -E "..."` - a control cell failed with ZERO diagnostic output (613-byte cell
   files). Born: tee <cell>.battery_full before filtering.
5. **Five hard machine deaths.** Every one preceded by svm_range_deferred_list_work hog
   escalations (counters 4-5) at server boot/teardown churn; noretry=1 insufficient. Born:
   the watchdog (pause at >=3), churn spacing 180 s, the persistent-server protocol (owner
   decision), and the rule that a boot after a crash inherits die heat (cooldown gates).
6. **Git object purges (2x).** Hard deaths at desk-WIP-commit moments emptied object
   files ("error: object file ... is empty; fatal: bad object HEAD"). Recovered both
   times via reflog + working tree. Born: snapshot pushes after every landing.
7. **The root/user output collision.** A root-run window left root-owned combowin_* files
   in /home/chris; the next user-run chain got Permission denied writing its own cell
   logs - two cells lost with zero diagnostics. Born: chown after any root window; the
   law against mixing root-run and user-run outputs.
8. **The docs-commit false trip.** The freshness gate compared the binary mtime to ANY
   HEAD commit; a docs-only commit tripped it and bounced four cells. Born: CODE_TS
   (last commit touching code paths only).
9. **The ratchet that no-opped.** The E-137 decode ratchet wrote the description but
   walked the wrong JSON path for the value (top-level loop vs cells/decode_8k_10k); the
   guard kept gating against 23.23 while serving produced 24.55. Caught because a fresh
   cell printed "vs baseline 23.23". Born: the walk-all-fields fix (5ad25e902).
   Postscript (E-148): the class recurred from the other side - the E-137 ratchet
   anchored the wrong field while the promotion procedure then had to pick the SAME
   field the guard reads (cells/decode_8k_10k/decode_per_second). Three ratchet-anchor
   incidents share one lesson: guard and ratchet must read the same path; a
   battery-side assertion (baseline == guard-printed value) is named.
10. **The EOS-killed depth decode.** U1's 50k decode: temp-0 EOS as the first token
    after 51k filler tokens -> 1-token generation, sentinel tps=1e6, draft_n=0. The
    server was healthy. Born: ignore_eos in run_u1_window_v2.sh + EOS-void cell marking.
11. **The thermal sampler that sampled the wrong GPU.** card0/1/2 hardcode; card1 was
    the NVIDIA display after a reshuffle; die 4 never sampled. Born: PCI-keyed sysfs
    sampler, 18-column format (E-136).
12. **The fa40 first attempt.** -38.9% end-to-end time with GARBAGE values - an earlier
    desk failed closed on the oracle and the lever sat dormant until the miscompile was
    characterized. Born: the oracle-before-timing law.
13. **The served shape-cache crash.** First honest engagement aborted at
    GGML_ASSERT(bcj.nodes[i]) (ggml-backend-meta.cpp:1929) in the draft path. Born: the
    heal + the meta re-entry host suite (section 4.2/4.3). SECOND PATH (E-147): the
    battery w19 window crashed 2/2 at ggml-backend-meta.cpp:832 (UNKNOWN split axis on
    a third shape variant during re-split) with the heal live - the feature is PARKED,
    not fixed.
14. **systemd-oomd killed U1 (14:26:55).** Socket-activated oomd SIGKILLed the U1
    cgroup mid-150k-prefill AND the user terminal; a second U1 instance had restarted
    from cell 10k on the OLD plan and was found alive-but-degraded at 15:03, then
    stopped by PID. U1 ends PARTIAL (10k full 23.78; 50k/100k prefill-valid; deep
    decodes EOS-void 2/2). Born: oomd DISABLED system-wide (kernel OOM-killer remains),
    run_u1_window_v2.sh (ignore_eos) rerun is REQUIRED and deferred, and the
    zero-byte-.o corollary (oomd/kill events leave objects cmake treats as fresh - 35
    repaired, E-144).
15. **The EXACT-VOID that wasn't.** The w23env battery window's cells ran --decode-only,
    which structurally never emits the determinism_guard row; the adjudicator compared
    None != PASS and voided the window ("text_sha256 diverged"). Kernels exonerated
    (counters identical across all 4 cells; composition host-proven W7; no second
    E-117-class gap). Born: det-missing vs det-fail adjudicator fix queued; the
    same-field guard/ratchet law (E-145/E-148, WALL 8).

---

## PART 9 - THE LAWS AND TOOLING (what a reviewer must trust and how to verify)

1. NOTHING PROMOTES without paired served cells (interleaved R/A/A/R, position-matched,
   void-gated; both pairs must agree in direction). Chained windows cannot resolve
   sub-5% candidates - measured, not assumed.
2. NO PERFORMANCE VERDICT without same-session bit-exactness (kernel arms). Replica
   harnesses are inadmissible - benches link THEIR tree's libggml-hip.
3. PROVENANCE OR IT DID NOT HAPPEN: every cell carries commit + tree-clean + binary
   sha256 + CMakeCache sha256 + full launch config; the battery fails closed without them.
4. EVERY GATE FAILS LOUD on missing inputs; every engagement line prints at WARN (the
   served log filter drops ggml INFO at verbosity 3 - common/log.cpp:444 via
   common_log_default_callback).
5. DIES ARE PCI ADDRESSES (0000:05:00.0 / 08:00.0 / 0d:00.0 / 10:00.0); card numbers
   reshuffle per boot.
6. GPU CHAINS RUN SYSTEM-SCOPE: `echo <pw> | sudo -S systemd-run --collect --unit=<name>
   -p User=chris -p Environment=HOME=/home/chris <script>` (user-session crashes killed
   three chains on 09-23; the password never enters any committed file).
7. SNAPSHOTS AFTER EVERY LANDING: /home/chris/push_backups.sh (orphan-tree commits to the
   origin fork; never upstream/duford; history push impossible - >100 MB blobs).
8. APPEND-ONLY LEDGER; corrections are new entries (E-111/E-112 precedent).
9. Watchdog: svm-watchdog.timer reads the per-boot hog counter; >= 3 claims the lock;
   releases after 10 quiet minutes. All runners wait on that lock (single-file pause
   machinery).
10. Decisions cell placement: position-1 or after >= 5-10 min idle; void_gate.py on every
    decision cell's thermal sideband; a boot after a crash inherits die heat.

---

## PART 10 - OPEN QUESTIONS (status after the 09-24 evening desks; originally posed to the reviewing LLM)

1. The sync wall / pipeline structures: **CLOSED (E-141/W29).** PP4 = 8-9 t/s (-65%): it
   removes the 31 ms boundary wall but serializes the ~65% concurrent per-die compute
   (cycle = 4C + ~25 = ~350-365 ms); winning needs a 3.3x per-die compute gain the
   banked physics forbids. tp2xpp2 is unimplementable (no composable TPxPP mode in the
   tree) and TP2-class capacity cannot boot at 200k. The early-era PP "loss" was an
   anecdote (layer arm crashed at decode entry - decode never measured), so this closure
   is by derivation, with falsification cells pre-registered (boundary wall > 250
   ms/cycle, or C < 31 ms/cycle, or bubble-free multi-ubatch PP reopens it). Still open
   only in the weak sense: a sharding/mode not in this tree that removes per-layer sync
   without serial staging.
2. The arrival skew: 59-141 us/die multiplicative with compute rate. Clock floors are
   the named lever - the battery window was a structural skip (unprivileged hook), root
   retry queued (E-147/E-148); +2.4-4.8% predicted. Still open: a way to make the WAIT
   cheap instead of preventing it (overlap the wait with useful per-die work).
3. MMVQ: the named follow-ups are all measured and dead - C4 DEAD (oracle-fail iq4_xs,
   +21.8% iq3_s; FMA-contraction, instantiation-local), occupancy rungs DEAD (lb2
   no-op; max-vgpr unbuildable on ROCm 6.2 LLVM 18), env window NO-PROMOTE (+0.32% in
   0.28 spread) (W30/E-144/E-146/E-147). Still open: an access-pattern restructure that
   beats the 70-120 GB/s cfull ceiling on gfx900 (DP4A, wave64).
4. Acceptance: head-calibration-bound is now EMPIRICAL - chain-4 drafted 167, accepted
   the identical 84 of chain-3, position-4 acceptance ~0 (E-147). The W34 fixed-text
   instrument (built, E-146) makes acceptance measurable for any future head-quality
   idea. Still open: any served-side way to raise proposal quality without a head
   re-fit; and the L4 acceptance-rises-with-depth prediction is untested until the U1
   v2 deep rerun.
5. The ring floor / cheaper exchange: **CLOSED for this stack (W31/E-144).** Flat P2P
   bypass DEAD EMPIRICALLY: all 12 ordered die-pairs canAccess=0 enable=0 (driver-level
   refusal; explains RCCL's own x12 enable failures on this PLX-switched fabric) and the
   host-staged-flat fallback WEDGES (zero completed iterations; probe watchdog fired).
   Multicast/shared-memory staging would face the same fabric refusal; the butterfly
   remains 25% slower end-to-end.

---

## PART 11 - TIMELINE (compressed; ledger is authoritative)

| when | era | headline |
|---|---|---|
| 09-20/21 | TP3 bring-up | first serves; T-band work; served gates (E-027..E-058) |
| 09-22 | optimization day | parity table (E-069/E-070); RCCL +25-27% (E-090); share gates shipped (E-095); serving default promoted (E-079/E-094) |
| 09-23 morning | TP4 era | 5/5 GREEN, 23.34/23.23 (E-105); MMVQ BW desk (E-109); the -20% regression caught -> provenance gate (E-111/E-112) |
| 09-23 evening | boundary + depth | boundary desk seals count/clustering (E-116); s2r wiring defect fixed (E-117); CI law (E-118); deep census: +1.95 ms/1k (E-124); first SVM death |
| 09-24 05-07 | hardened boot + fleet | boot queue; 3 reboots; deaths #4-#5 -> svm-watchdog; hardened paired queue: ub1024 verdict; W18 soak named |
| 09-24 06-09 | desks era | W20 void gate; W21 persistent-server; W22 asymmetry; W23 MMVQ v2; dc stale-binary scandal fixed; fa40 var-11 oracle-proven + MERGED + PROMOTED (24.55); planned reboot |
| 09-24 10-13 | boot #4 | queue 4/4 PASS on new config; ub1024 contradiction; W24/W25/W27/W28 landed; U1 marching (10k measured); EOS bug fixed in v2; acceptance synthesis adopted |
| 09-24 13-14 | dossier + review | CAMPAIGN_DOSSIER v2 written 13:30; W29 PP NO-GO x2 + the scale correction (E-141); external review adopted -> W30 (MMVQ-C4+occupancy) and W31 (P2P-allreduce) dispatched, battery confirmed as the vehicle (E-142) |
| 09-24 14-16 | oomd incident + negatives | systemd-oomd SIGKILLs U1 cgroup + user terminal 14:26:55 -> DISABLED system-wide; dual U1 instance discovered 15:03 (restarted from 10k on the old plan), stopped by PID; U1 PARTIAL, v2 rerun REQUIRED (E-140/E-143); W27 q40-prefill 9/9 BIT-EXACT -40..-44% at depth, U2 queued (E-143); W30 arms fail + W31 P2P dead; battery launched 16:06 (E-144) |
| 09-24 16-19 | battery + promotions | W33: w23env EXACT-VOID overturned - harness false-positive, kernels exonerated, no-promote on merits (E-145); W34 fixed-text acceptance instrument built (E-146); battery DONE 18:02 - w19 ARM-CRASH x2 at meta.cpp:832 -> PARKED, w22c1 CTRL-FAIL (root hook), w28nmax REJECTED -43.4% (E-147); ub1024 PROMOTED from server-log adjudication -> ratchet 24.69, 1024/1024 canonical (E-148) |

*End of dossier. When this document and the ledger disagree, the ledger wins
(append-only, authoritative). Latest ledger head at writing: E-148, commit 68654e245.*
Queued behind this sync: amd/q40-prefill merge + rebuild + CI (merge-order law), the
w22c1 clock-floor re-run under root, U2 served window, U1 v2 deep rerun.
