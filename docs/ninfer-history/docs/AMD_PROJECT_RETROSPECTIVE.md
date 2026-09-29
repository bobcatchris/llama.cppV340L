# THE AMD/V340L PROJECT — FULL RETROSPECTIVE (what worked, what didn't, what ports)

**Date:** 2026-09-20. **Author:** Team Red coordinator chair.
**Purpose:** the principal declared the port to llama.cpp (branch: DEV-DUFORD/llama.cpp).
This document is the complete account of the ninfer-on-AMD campaign — what was achieved,
what was measured and failed, and what carries across the port.
**Everything cited is banked:** docs/amd/PERF_LOG_AMD.md (PLOG-001..096, hash-chained),
results/amd/**, the work orders in docs/amd/WO_*.md.

---

## 1. WHAT THE PROJECT WAS

Serve a 27B-class LLM (Qwen3.8-27B, NVFP4 weights) on 2× AMD V340 boards (4 gfx900 dies,
8 GB visible per die, ROCm 6.2.0) at TP4, on the in-house `ninfer` stack — with the goal
of maximum tokens/second. The goal moved twice: 1,000 tok/s prefill (killed by ISA
physics), then "700-class" (REV2), and on 2026-09-20 the principal moved the model and the
stack: **llama.cpp, smaller/better-quantized weights, NVFP4 dropped**.

## 2. WHAT WORKED (kept numbers, all measured)

**The serving stack was built and promoted:**
- KVarN quantized KV cache ported to HIP (the CUDA line's kvarn route re-implemented):
  k4v2/k4v4/k5v4 all serving; **kvarn_k4v4 PROMOTED 2026-09-20** — 65,536-token context
  (3.1× denser than bf16), quality battery 8/8 needle-exact at 25/50/75 % depth of ~60k
  prompts. PLOG-084.
- The long-context unlock chain: the ~10.1k arena crash root-caused (int32 hypothesis
  falsified; 96 MiB chunk-arena exhaustion + a port-introduced O(position) temp), fixed
  in-place (bf16→fp16 V-temp, byte-identical below the wall), NINFER_WORKSPACE_MIB=512
  lever measured (position ceiling 12.3k → 65,536). RED→GREEN with cells.
- **mad-mix**: the NVFP4 GEMM's fp32-accumulation fix — +8.83 % serving wall (PLOG-080),
  after the first delivery attempt (inline-asm) had killed the same idea at 0.48×. The
  lesson: the ISA wasn't the problem, the delivery was.
- **Decode-path cures**: pinned-slot H2D staging (killed a ~995 ms/chunk pageable-enqueue
  stall; paired −18 to −33 % per ordinal), device-side MTP finalize tail (1,394 → ~14 ms),
  SIMT promotions for dn/gqa/ggate body ops (~120 tok/s class era, PLOG-067).
- **Speculative decode (MTP)**: acceptance 0.88 / 2.59 tok-per-round at draft-2 on the
  promoted line — decode 13.8 tok/s at 2k.
- **The measurement infrastructure**: hash-chained perf ledger (96 byte-reproducible
  links), ordinal-paired A/B protocol (PLOG-060), kill-proof detached runners, per-window
  boot battery, the fair-2k comparison matrix, the fixed clock sampler.
- **The thermal map** (PLOG-083): prefill is a state function of accumulated heat —
  104 cold / 70.7 warm / 63.1 hot tok/s (1.65×) on identical config; minutes-scale load
  integrator, ~90 s drain, ~3 min idle = full recovery. Operator levers quantified
  (+52-65 % interactive from a 3-minute-idle policy; power-cap A/B named).

## 3. WHAT DID NOT WORK (measured, with the why)

- **The prefill ceiling.** The 1,000 tok/s goal died on ISA physics (gfx900 fp32 peak
  10.75-12.5 TF/s/die; 54 % of pk16 unreachable — the W3 ledger row). The 700-class REV2
  goal was never reached: best sustained prefill on the final promoted line is
  **~183-194 tok/s** at 2k geometry (PLOG-087). Five GEMM families tried; mad-mix's
  +8.83 % was the one that converted.
- **Absolute performance vs the project's own reference.** The NVIDIA line's 2026-09-09
  logs (results/157_p0/) show **2,835 tok/s prefill and 85-101 tok/s decode at 40-80k
  context** (TP2, 16 GB ranks, int8/requant KV, MTP acceptance 0.879 at draft-3). Team
  Red's best-ever: 190/13.8 at 2k. The AMD line ran 15× under its sibling on prefill.
- **kvarn's decode tax**: k4v4 decode 10.4 tok/s vs bf16 16.9 (median, 2k ctx; ~1.6×) —
  attention dequant overhead. kvarn buys capacity (65,536 vs 23,808), never speed.
- **Long context beyond 65,536 is a die-capacity wall**: 200k/250k/300k kvarn_k4v4 all
  refused live (slack −1,335/−1,867/−2,399 MiB; PLOG-089). The O(position) dequant-scratch
  design of the materialize prefill route caps position at 8 KiB/token × workspace. The
  structural cure (docs/120 B2 direct route) was parked as not-byte-identical.
- **Every chunk-level lever failed to convert to serving**: chunk ladder (no-gain at 512),
  M=256 re-leg (+0.17 % vs the 5 % bar), graph capture (twice: no win, then degradation),
  V-arm (+5.7/6.6 % slower), fp8-on-ring AR (numerics-bound), first-chunk "gap" (refuted —
  a boot transient + cured stall), body-op fusion (the seam was 3 %; the real lever was
  launch shape → ksplit 2.03× cell but 0.0 % serving).
- **The platform walls**: L2 stall wedges = platform-class (5 cures, AMDKFD_IOC_WAIT_EVENTS
  named, reopen = ROCm upgrade); compiler road dead (LLVM 17/19/22 invariant); P2P
  canAccess=0; random GQA warmup page-faults (2 incidents, FLR-recovered).
- **AR is structural**: 101-141 ms/chunk (12 %), host-SHM-bound, zero compute overlap
  (measured 0.0 % on all four dies), environment-immovable (16 RCCL configurations).

## 4. THE HONEST END-STATE TABLE (2k context, this box, this session)

| config | prefill | decode |
|---|---|---|
| ninfer bf16 + draft-2 | 75.8-101.0 | **16.9 med (peak 23.5)** |
| ninfer bf16 + draft-3 | 64.9-69.5 | 14.2-16.6 |
| ninfer k4v4 + draft-2 (promoted) | 64.3-67.5 | 10.4-10.6 |
| **llama.cpp TP2 layer, q4_0** | **103.2** | 8.95 |
| **llama.cpp TP4 layer, q4_0 + MTP** | 87.6 | **11.93** |

And the sibling reference the campaign never chased: **2,835 / 85-101 at 40-80k (NVIDIA
TP2, 09-09)**.

## 5. WHAT PORTS TO LLAMA.CPP (the reason this document exists)

- **The gfx900 kernel work already exists in the destination**: DEV-DUFORD/llama.cpp
  master carries the gfx900-tuned MMQ prompt-processing path (commits 717920bf9,
  76fd31245; branches gfx900-mmq/-tune) — the exact lever our custom GEMM stack tried to
  be, upstream-shaped. Tensor split on this GPU is known-working there.
- **One patch, found and applied**: upstream's HIP header gates FP8 types on
  `HIP_VERSION >= 6.2` but ROCm 6.2 only defines them for gfx942+ — on gfx900 the build
  fails until `FP8_AVAILABLE` is additionally gated on `__gfx942__/__gfx950__`. No FP8 is
  needed for IQ4_XS-class models.
- **MTP exists on both sides**: llama.cpp `--spec-type draft-mtp` works with the
  Qwen3.8-27B GGUF (NextN tensors), acceptance 0.65 / mean 2.95 tok/step; the depth knob
  is `--spec-draft-n-max` (d2 11.02 / d3 11.5 / unbounded 11.93 tok/s decode on TP4 bf16
  KV). Our GGUF (byteshape IQ4_XS, 3.84 bpw, 12.17 GiB) carries the tensors.
- **The measurement kit ports as-is**: ordinal-paired protocol, the fair-matrix runners
  (results/amd/llamacpp/), the fixed clock sampler, the census leg (with
  EXTRACT-BEFORE-RESTORE now enforced by tooling), the thermal-map method (start-temp
  mandatory on every row).
- **The laws that transfer**: kill-switch-first (env-gated, unset = byte-identical),
  allocator-is-the-gate (no estimated refusals), RED→GREEN with both-direction
  falsifiers, bars belong to goals, era-myopia and fair-geometry rules.

## 6. THE LESSONS (each one paid for)

1. **Delivery beats idea**: mad-mix died at 0.48× as inline-asm, won at 1.10× as plain
   C++ — same math, different emission.
2. **Cell wins don't convert unless they shrink the request-critical path** (3-for-3 this
   window: V-arm, ksplit, chunk-256). Name the path term and its share BEFORE the window.
3. **Era-myopia**: every capacity/ceiling claim must cite its manifest era (bf16 TP4 once
   booted 98,944 tokens; "36,352 is the ceiling" was true for one week's config).
4. **Fair geometry**: never compare across context lengths or thermal bands. Both rules
   were learned by being publicly wrong.
5. **Reservation ≠ residency**: the ~867 MiB of "fixed" workspace is lazy pages; boot-log
   arithmetic needs rocm-smi truth to match.
6. **Peak numbers are not sustained numbers**: the "40 t/s" promise was a
   first-hundred-tokens figure; sustained at 2k was half that. Report the curve.
7. **The reference matters more than the ratio**: 190 tok/s looked like a win until the
   sibling line's 2,835 surfaced. Always benchmark against the best known state
   anywhere in the project, not against your own last week.

## 7. WHERE THE BODIES ARE BURIED (the parked-with-keys list)

| item | park receipt | reopen key |
|---|---|---|
| kvarn long-context B2 direct route | docs/120 + refusal receipts | a >65,536-context requirement |
| hosted-KV Phase 2 | WO_KVHOSTED_P2.md resume checklist | principal word (200k+ demand) |
| llama.cpp comparison (now EXECUTED) | PLOG-092..096 + brief | — (done) |
| MTP draft-3 on ninfer | acceptance collapse 61 % | acceptance-rate fix |
| gate-kernel 116 µs residual | PLOG-085 | >1%-of-wall pricing |
| power-cap A/B / cooling | PLOG-083 leaflet | operator hardware (in transit) |

*The AMD box is not a failure — it is a fully instrumented, honestly measured machine
that ran a 27B at 65,536 tokens with quantized KV and a promoted, quality-batteried
serving line, and produced the measurement discipline the port now inherits. The numbers
were never good enough, and now they never have to be lied about.*
