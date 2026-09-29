# 05 · NInfer engine on 2× RTX 5060 Ti — development plan

**Status:** ARCHIVE

**Purpose:** build and validate a multi-GPU NInfer engine on the 2× 5060 Ti *now*, while the 2× V340L is in transit. The 5060 Ti is the ideal development platform because the NInfer kernel tree ports to it **for free** (same `sm_120a`), and the one piece of net-new work — the **multi-GPU (TP) layer** — is *exactly* the piece the V340L will also need. Develop it on hardware you have, then carry it over.

Companion docs: `03_ninfer_analysis.md` (how NInfer works), `04_consolidated_v340l_estimates.md` (V340L targets).

---

## 0. Why the 5060 Ti is the right dev platform

| Property | 5060 Ti (dev) | V340L (final) | Consequence |
|---|---|---|---|
| ISA | **sm_120a** | gfx900 (GCN5) | NInfer kernels **compile & run unmodified** on the 5060 Ti; need a full CUDA→HIP rewrite for the V340L |
| FP4 tensor cores | **native** (`kind::mxf4nvf4`) | none | Can validate the W4A4 prefill path here; V340L must fall back to A16 W4 |
| Multi-GPU layer | **net-new** | **net-new (same)** | Build it once on the 5060 Ti, reuse on the V340L |
| Target model | **Qwen3.8-27B — already registered** in NInfer | Qwen3.6-27B (registered) | No new model registration |
| Availability | **in the box now** | in transit | Unblocks all engineering immediately |

**Strategy in one line:** use the 5060 Ti to (a) prove the NInfer build + kernels on real hardware, (b) build + validate the TP2 layer with byte-diffs, and (c) produce a working 27B serving engine — then port the *engine + TP layer* to the V340L and rewrite only the kernel tree.

---

## 1. Measured hardware (this machine, 2026-08-19)

| Spec | Value | Source |
|---|---|---|
| GPUs | 2× RTX 5060 Ti (GB206, Blackwell) | `nvidia-smi` |
| SMs / CUDA cores | **36 SMs / 4608 cores** each | CUDA device props |
| Compute capability | **12.0 (sm_120)** = NInfer's target | `pynvml` |
| Memory | **16 GB GDDR7** each (16311 MiB) | `nvidia-smi` |
| Bus / L2 | 128-bit / **32 MB L2** | device props |
| Memory bandwidth | **~448 GB/s** (28 Gbps × 128-bit) — *spec, to confirm* | datasheet; confirm w/ NInfer HBM probe |
| SM clock | ~2600 MHz base, **3090 MHz max** | `nvidia-smi` |
| PCIe | **5.0 × 8** (max 16x, current 8x) both | user + `nvidia-smi` |
| Power | 180 W each | `nvidia-smi` |
| Driver / CUDA | 580.173.02 / **12.9 installed** (NInfer needs **≥13.1**) | `nvidia-smi`, `nvcc` |
| **Currently occupied** | llama.cpp server, **~15.5 GB used/GPU**, ~790 MB free | `nvidia-smi` |

**Current baseline being replaced** (the server running this agent):
`llama-server --model Qwen3.8-27B-Q5_K_M.gguf -sm tensor -ts 1,1 --spec-type draft-mtp --spec-draft-n-max 3 --ctx-size 200111 --cache-type-k/v q8_0 --flash-attn on -b 4096 -ub 512` (custom build at `/home/intel/complete-fix-llama-cpp`).

---

## 2. 5060 Ti performance projections (targets, 27B NVFP4, TP2)

Weight basis: 27B **NVFP4 = 15.2 GB** (NInfer native) → **7.6 GB/GPU** at TP2. (The current server uses Q5_K_M ≈ 18.8 GB; NVFP4 is lower precision but NInfer-native and what the FP4 TC path needs.)

| Workload | Projection | Basis |
|---|---|---|
| tg plain (MTP0) | **~32–36 t/s** | 7.6 GB @ 70–75% of 448 GB/s = ~23.5 ms + AR ~3 ms + KV/misc ~2.5 ms |
| tg MTP k=3 math | **~85–95 t/s** | round ~37 ms, 1+3×0.78 = 3.34 tok/round |
| tg MTP k=3 structured | **~95–105 t/s** | 1+3×0.90 = 3.7 tok/round |
| tg MTP k=4 math | **~100–110 t/s** | round ~40 ms, 4.12 tok/round |
| tg MTP k=4 structured | **~110–120 t/s** | 4.6 tok/round |
| pp512 (prefill) | **~8000–9500 t/s** | W4A4 FP4 TC; ≈ 0.86× one 5090's 11,191 t/s |

Context: the 5060 Ti is a **decode underdog** vs the V340L (896 vs 1935 GB/s aggregate BW) but a **prefill monster** (native FP4 TC vs the V340L's fp16-only 86 TFLOPS). Expect the 5060 Ti to beat the current llama.cpp server by a wide margin on both, since llama.cpp is far from these roofs.

### Measured so far (2026-08-19, real 2× RTX 5060 Ti, 27B groupwise-int Q4)

| Stage | Status | Measured |
|---|---|---|
| HBM roofline (single GPU) | ✅ | 427.3 GB/s read |
| TP GEMV (all quants, tuned SIMT) | ✅ | 414 GB/s = 97% of roofline |
| NCCL allreduce (PHB, ≤242 KB) | ✅ | 6–9 µs/call |
| PP2 (pipeline, 32 layers/rank, MTP0) | ✅ | **23.8 t/s** warm (41.6 ms/step, 92% of HBM roof) |
| **TP2 Phase 2a (MLP-only split, MTP0)** | ✅ **correct** | **~29 t/s** warm (34 ms/token, 1.2× over PP2) |
| **TP2 Phase 2b inc1 (MLP+attention split, MTP0)** | ✅ **correct, deterministic** | **~30 t/s** warm (33.3 ms/token), 10454 MB/rank |
| **TP2 Phase 2b inc2 (full split incl GDN, MTP0)** | ✅ **correct, deterministic** | **~32.5 t/s** warm (30.7 ms/token), 8780 MB/rank |

**TP2 Phase 2a correctness**: token-identical to the PP2 reference (split `gate_up`→`tp_gemv`+`silu_mul`→`down`→`allreduce`→residual reproduces the single-GPU fused `post_mixer` math exactly).

**Full-split (Phase 2b) correctness**: every layer is now tensor-parallel — attention (`query_key`/`gate_value` multi-range column split + GQA with local KV heads, sharded KV cache), GDN (multi-range `[q;k]`/`[v;z]` split + channel-sharded causal conv + head-split delta-net + RowK `out` + AR), MLP (multi-range `[gate;up]` + RowK `down` + AR). Output is token-identical to PP2 and deterministic across runs.

**Bug found & fixed in GDN bring-up**: the `GdnConv` sharding read the `q_local` block from channel offset 0 on *both* ranks (missing the rank offset `r*1024`), silently corrupting rank 1's conv → degenerate output. Also eliminated the per-step D2D copies in `gdn_input_projection_tp` (96 `cudaMemcpyAsync`/step) by allocating a combined `[q;k;v;z]` buffer so both GEMVs write directly (host-enqueue 5.9→4.1 ms/step).

**Why full split caps ~32.5 (not the 45–50 target)**: the per-rank GEMV byte count is now ~half (~8.8 GB vs 15.6 GB), but steady-state is only ~8% faster than Phase 2a because the step is **not GEMV-bandwidth-bound** — it is launch/occupancy-bound by the many small non-GEMV kernels (causal conv, recurrent delta-net, rope, RMSNorm, column-extract, argmax) that drain the memory pipeline between GEMVs. The real remaining lever is **MTP k=3** (amortizes the per-step fixed cost across 3 tokens → target ~90–105 t/s).

> Note: the table above projects the full-split NVFP4 target. The current bring-up uses the **groupwise-int Q4** artifact (15.6 GB → ~10.9 GB/rank Phase 2a) to validate the TP pipeline first; NVFP4 comes after Phase 2b.

---

## 3. Key architectural decisions

1. **TP2 (tensor split) across the 2 GPUs** — matches the current `-ts 1,1` topology; parallel weight read (2× BW), not serial layer split.
2. **NVFP4 weights** (NInfer native) — enables the W4A4 FP4 prefill path. Accept the Q5_K_M→NVFP4 precision step-down; validate with a quality spot-check (see Phase 3). Optional later: register a groupwise-int profile if quality needs it.
3. **Custom one-shot allreduce + AR batching (every 2–4 layers)** — the V340L-validated design; over PCIe 5.0 ×8 the 10 KB hidden-vector AR is latency-bound, so batching is the main win.
4. **MTP k=3 default** (matches current server), k=4 as the ceiling test.
5. **Graph-captured decode round** across both GPUs (NInfer's `DecodeGraph` extended to 2 streams).

---

## 4. Phased plan

### Phase 0 — Environment & baseline (day 0, no free GPU needed)
- [ ] **Install CUDA 13.1+** (NInfer hard-fails on <13.1; box has 12.9). Keep 12.9 for the running server.
- [ ] Measure the **current llama.cpp baseline** (tg MTP0/k=3, pp512/2048) → the "before" numbers.
- [ ] Confirm PCIe 5.0 ×8 (`nvidia-smi -q | grep -A3 "Link"`), confirm 448 GB/s later via HBM probe.
- [ ] Locate/download **Qwen3.8-27B HF safetensors** (converter input).
- **Deliverable:** baseline CSV + CUDA 13.1 toolchain + HF weights.

### Phase 1 — Build + single-GPU kernel validation (days 1–3, needs a GPU test window)
- [ ] Build NInfer: `cmake -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_BUILD_BENCHMARKS=ON` → confirm it **compiles clean on the 5060 Ti** (the whole point of this platform).
- [ ] Run **per-op benchmarks** (`bench/ops/*`) on one 5060 Ti → GEMV/GEMM/GDN efficiency + confirm kernels are correct.
- [ ] Run **HBM bandwidth probe** (`tools/hbm_bandwidth_probe.cu`) → real 5060 Ti BW (confirm 448 GB/s).
- [ ] **Convert** Qwen3.8-27B → NVFP4 `.ninfer` artifact (`tools/convert/qwen3_8_27b/convert_nvfp4.py`).
- [ ] Single-GPU smoke test on a shape that fits (or the 35B-A3B if it fits) to validate the engine end-to-end.
- **Deliverable:** green build, per-op bench CSV, HBM BW number, `.ninfer` artifact.

### Phase 2 — Multi-GPU (TP2) layer (days 3–10) — **the net-new work**
- [ ] **Weight sharding:** at load, split each linear tensor across 2 GPUs (column/row split per NInfer's problem classes: `AttnInput`, `GdnInput`, `MlpGateUp`, `Residual*`).
- [ ] **Allreduce:** custom one-shot AR kernel + **AR batching** (accumulate 2–4 layers, reduce once). 2-GPU over PCIe 5.0 ×8.
- [ ] **KV + GDN state sharding:** attention KV heads split across GPUs; GDN linear-attention state split.
- [ ] **Graph capture across 2 streams** (extend `DecodeGraphDefinition/Executable`).
- [ ] **Byte-diff validation** against the single-GPU reference (for shapes that fit) and/or the llama.cpp output.
- **Deliverable:** TP2 layer, byte-validated. *This is the artifact that carries over to the V340L.*

### Phase 3 — 27B on 2× 5060 Ti (days 10–14, needs the server stopped)
- [ ] Load 27B NVFP4 at TP2, 200K ctx.
- [ ] **Correctness:** byte-diff / quality spot-check vs the current Q5_K_M server (same prompt, same seed) — confirm NVFP4 is acceptable.
- [ ] **Measure:** tg (MTP0, k=3, k=4) + pp512/2048 → compare to §2 projections and the Phase 0 baseline.
- [ ] Concurrency sweep (C=1,2,4,8) if useful.
- **Deliverable:** measured 5060 Ti results table + go/no-go on NVFP4 quality.

### Phase 4 — Handoff to V340L (after arrival)
- [ ] Port the **engine + TP2 layer** to HIP/ROCm (the multi-GPU logic is device-agnostic C++; the CUDA calls → HIP).
- [ ] **Rewrite the kernel tree** CUDA/PTX → HIP/GCN5 (the big work; use `01_v100_skinny_findings_for_v340l.md` for the GEMV/GEMM design).
- [ ] Re-derive the prepack layout for gfx900; re-tune schedules.
- [ ] Run 27B Q4_1 on 2× V340L → compare to `04` targets.

---

## 5. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **GPUs occupied** by the running llama.cpp server (~790 MB free) | Can't run any GPU test | Do all Phase 0–2 *code* work first; schedule a single **GPU test window** where the server is stopped (Phase 1 benches + Phase 3). The agent itself runs on that server, so the user drives the stop/start. |
| CUDA 13.1 not installed | Build fails immediately | Phase 0 installs it (keep 12.9 for the server). |
| 5060 Ti BW < 448 GB/s (power/clock) | Decode projections drop | HBM probe in Phase 1 measures the real number; re-derive projections. |
| NVFP4 quality < Q5_K_M | User unhappy with output | Phase 3 quality spot-check; fallback = register a groupwise-int profile (NInfer has one for the 27B). |
| AR batching drifts numerics | Wrong tokens | Byte-diff vs unbatched; fall back to per-layer AR. |
| TP2 allreduce over PCIe slower than modeled | Decode below target | Measure in Phase 2; tune AR batch factor; consider fusing more layers. |

---

## 6. What "done" looks like

1. NInfer **builds clean** on the 5060 Ti (sm_120a) with benchmarks.
2. A **byte-validated TP2 layer** (the reusable artifact for the V340L).
3. A **working 27B NVFP4 server** on 2× 5060 Ti with measured tg/pp beating the current llama.cpp baseline.
4. A **measured 5060 Ti results table** + confirmed HBM BW.
5. A **concrete V340L port checklist** (Phase 4) ready to execute the day the cards arrive.

## 7. Immediate next actions (starting now)
1. Write this plan (done) + consolidated V340L doc (done).
2. **Start Phase 0/1 code work that needs no free GPU:** set up the CUDA 13.1 install, attempt the NInfer build for `sm_120a`, and scaffold the TP2 layer structure.
3. Flag the **GPU test window** dependency to the user before any kernel/bench run.
