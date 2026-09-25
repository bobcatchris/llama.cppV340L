# 60 — v340l Code Reuse Assessment (AMD datacenter GPU)

**Status:** CURRENT — assessment of what the dual-5060 Ti ninfer project
contributes to the v340l build. **v340l is an AMD datacenter GPU machine
(ROCm/HIP), not NVIDIA.** Written to be shareable; no internal context required
beyond "ninfer = our custom CUDA inference engine for Qwen3.8-27B."

---

## Premise

The 5060 Ti project exists as a lead-up: prove the architecture, burn through
the correctness defects, and build the verification machinery on cheap
hardware — so that v340l focuses on the **cross-vendor port + performance
tuning** rather than on from-scratch design. This document is the honest
inventory of what transfers across the NVIDIA→AMD boundary.

## 1. Reuses as-is (hardware-agnostic)

| Area | What it is | Why it transfers |
|---|---|---|
| **Serve layer** | HTTP server, OpenAI-compatible API, tool-call parse/emit, `/v1/compact` recursive chunked compaction engine, per-request progress logging (prefill + decode phases), auto-KV sizing from measured free VRAM, request admission control, context-overshoot tolerance zone | Pure C++, no GPU assumptions. Every defect fixed during the 5060 Ti stabilization effort (D-01…D-13) lives in this layer and transfers untouched. |
| **Paged KV core** | Pool/plane/block-table abstraction, 64-token pages, allocation entitlements, resize/trim semantics | Device-agnostic C++; the KVarN extension (packed code planes + per-tile scale side table) is already built on top of it. |
| **KVarN sub-8-bit KV** | CPU codec reference (verified), CUDA tile codec, pool layout, write-on-commit semantics, MTP commit-on-accept proof | The algorithm and the verified CPU reference transfer directly; ~50% KV memory reduction vs int8 carries over. The CUDA tile kernel is plain CUDA C++ (no inline assembly) — a mostly mechanical HIP port. |
| **TP sharding math** | Weight role classification (row/col/replicated/GDN-specific), world-size-generic local-shape computation, sharded materialization from the artifact container | Already parameterized by `world` — the same code shards to N ranks on any vendor. |
| **MTP speculative decoding design** | Verify-column state snapshots, commit-on-accept semantics, narrowed draft-vocab argmax, fused cross-rank argmax exchange (zero logit traffic for greedy) | The *design* is proven and hardware-agnostic; only the collective calls underneath get swapped. |
| **Verification machinery** | 15-test correctness battery (determinism, prefix reuse, tool calls, long-prefill, context boundaries), CI wiring with server lifecycle management, damage-proof gate policy, defect-ledger methodology | The most valuable asset: a v340l build is verifiable from day one with the same pass/fail criteria. Matters even more on AMD, where the tooling ecosystem is less battle-tested — the battery becomes the safety net. |
| **Conversion tooling** | Artifact container format, selective-GPTQ q4/q5 converter, full-Q4 converter, nvfp4 experiments | CPU/PyTorch — runs anywhere, produces artifacts any GPU can load (the artifact loader itself is device-agnostic C++). |

## 2. Ported to ROCm/HIP (design transfers, kernels are reworked)

- **KVarN tile codec** — no inline assembly; `hipify` covers most of the API
  surface. Expect a straightforward port with retuning of block sizes.
- **GQA attention kernels** — these lean heavily on NVIDIA-specific primitives:
  inline PTX tensor-core MMA (`m16n8k32`), `cp.async`, `ldmatrix`, XOR-swizzled
  smem tiles, split-KV decode, int8-QK path. The *algorithmic design* (swizzle
  layout, split-KV partitioning, online softmax structure, the int8-QK idea)
  transfers as a spec, but the kernels are **reimplemented** against AMD's
  matrix instructions — this is the largest kernel effort on v340l.
- **GDN (linear attention) chunked kernels** — WY-inverse / state-passing /
  output pipeline. Same story: structure and math transfer; the CUDA
  implementations are reworked for the target ISA, then tuned.
- **Toolchain** — nvcc→hipcc, `sm_120a`→the target gfx arch, build system and
  CI adjusted accordingly.

## 3. Replaced (deliberately)

- **Collectives.** The one-shot allreduce is a 2-peer PCIe design for consumer
  NVIDIA cards. v340l uses RCCL over Infinity Fabric; the sharding and
  synchronization *structure* around collectives stays, the implementation goes.
- **Memory constants.** Workspace/arena sizes were measured for 16 GB cards;
  they get re-measured. The auto-sizing logic that computes them from free VRAM
  transfers as-is.
- **Perf targets.** Acceptance-rate baselines, throughput gates, and the
  80k/200k context numbers are hardware-specific and get re-baselined on v340l
  (the battery + ledger machinery for doing that is section 1).

## 4. Strategic implication

The hardware-agnostic half of the codebase — serve, runtime orchestration,
paged KV, KVarN design, sharding math, converters, tests — transfers directly.
The kernel layer is a **cross-vendor port/reimplementation** (the dominant v340l
engineering effort), and only the collectives plus tuned constants are fully
replaced. Crucially, the correctness risk — the part that consumed most of the
5060 Ti effort: speculative semantics, prefix-cache behavior, context
exhaustion handling, streaming/tool-call API conformance, OOM/leak classes —
is already paid for on hardware where iteration is fast. v340l work should
concentrate on: (1) HIP port of the tile codec + collectives swap (RCCL),
(2) reimplementation of the GQA/GDN kernels against AMD matrix instructions,
(3) re-baselining perf gates with the inherited battery.

## 5. Caveats

- The specific AMD part (CDNA generation, memory size/bandwidth) was not
  available at writing time; kernel targeting and tuning strategy depend on it.
  The v340l reference docs folder (`~/comfy_templates/v340l_optimization/`) was
  empty when this was written — section 2's effort estimates assume a CDNA-class
  Instinct part and should be revisited against the actual spec.
- AMD tooling (ROCm version, hipify coverage of our kernel idioms) may surface
  gaps early in the port; the inherited test battery is the mitigation.
