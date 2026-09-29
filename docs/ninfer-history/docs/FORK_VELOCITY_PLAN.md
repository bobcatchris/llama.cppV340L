# FORK VELOCITY PLAN — adopting ninfer-gfx906 to accelerate the V340L line

**Author:** agent3 (hub `pi-dual_5060_ti_ninfer-1060980`), 2026-09-12, on user directive
("velocity supersedes — do not spend time re-deriving code we could adopt").
**Status:** PROPOSAL for coordinator ruling. This doc adjusts the current plan; it does not
replace the coordinator's authority over WOs, grants, gates, or the roadmap.
**Fork provenance:** https://github.com/JCraigWasTaken/ninfer-gfx906 @ `7a3c18d` (cloned
read-only to `/tmp/ninfer-gfx906-review`). Every fork claim below was verified against the
cloned bytes, not its README. The fork is itself a port: upstream `feaf4dd0` + donor
`wamansou/ninfer-tp2-1m` (TP2 on 2×5090, no P2P), cherry-picked in 9 documented slices.

---

## §0 Executive summary

The fork has already completed, on the same GCN5 wave64 family, the arc this line has just
set as direction: a full HIP port with **working TP2 serving** (eager 31.9 t/s decode,
pp512 339 t/s, MTP 46 t/s code; parity gate PASS ~10× margin), a completed 224-TU port
census, solved no-P2P transport, and hard-won operational lessons (graph-executor
cross-device bug; a flag-sync spin-wait that **wedged a card and rebooted the box**).
Adopting its assets as donor material — instead of re-deriving from CUDA — is the single
largest velocity lever available. **One hard limit found during due diligence:** the fork
diverged from upstream *before* the Q3 promote — it contains **no q3/q2 code** — so WO-04's
HIP-compat lanes remain genuinely ours (no throwaway there; it is the one place we must
still write new compat code).

## §1 Board-fact corrections this review locks in (supersedes earlier verbal claims)

| Fact | Value | Source |
|---|---|---|
| Our ISA | **gfx900** (Vega 10), NOT gfx906 | docs/amd/README.md registry |
| Devices | 4 HIP devices = 2 cards × 2 dies, **7.98 GiB per die** (31.94 GiB/box) | registry, user-confirmed |
| Our ROCm | 6.2.0-66 | registry |
| Fork ISA | gfx906 (Vega 20): has `v_dot2_f32_f16`, `v_dot4_i32_i8`; **we have neither** | fork kernels + ISA docs |
| Fork ROCm | 6.4.1 — its graph-executor and wedge findings are **runtime-behavior claims that must be re-measured on 6.2.0**, never assumed | fork S8/S9c notes |
| Fork box | 2× MI50 = 2×32 GiB; artifact fits tp2 easily there. **Our tp2 = 2 dies = 15.97 GiB < 19.03 GiB artifact** → on this box, tp2 of the 27B artifact is a fixture-class gate (matches registry M2 ruling); real serving is M4/4-device, which hits the engine's 2-rank limit → **coordinator ruling needed** (§4 R5) | registry + fork README |
| Fork q3/q2 | ABSENT (base predates promote) | fork tree |
| Fork tp2 engine files | has no `src/runtime/tp2/`, no `tp_engine.cpp`, no `tp2_budget.h` → **nothing to regress; anti-resurrection trivially clean; cherry-pick only, never merge** | fork tree |

## §2 Asset inventory → disposition

| Fork asset | What it is | Disposition |
|---|---|---|
| `src/core/hip_compat.h` (93 defines) + shim-header trick | CUDA→HIP compat surface incl. `HIP_DISABLE_WARP_SYNC_BUILTINS` + unmasked shims (logical 32-lane subgroups in wave64) | **ADAPT** into `src/common/hip_shim/` under RULING-1. Gap already measured: we lack `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaMemoryTypeDevice`, `cudaErrorPeerAccessAlreadyEnabled`, `cudaStreamIsCapturing`, `cudaStreamCaptureStatus(*None)`; pointer-attrs present but `.type` field mapping unverified on 6.2.0 |
| `src/ops/linear/gfx906/` GEMV/tiled kernels; q4/q5/q6/w8/bf16 dispatch `gfx906_reroute()` idiom | pass-2 GEMVs, rowsplit tiled GEMM, guarded `#if defined(__gfx906__)` V_DOT paths with generic SIMT fallbacks | **ADAPT/REFERENCE** for the shared (non-q3) ops; the reroute wrapper is the structural twin of RULING-1 lanes. Freeze any greenfield rewrite of these families |
| `gqa_attention_{decode,prefill}_{bf16,i8}_gfx906.cuh`, `gqa_attention_geometry.cuh`, kv_quant, fp16-V plane | working GCN5 attention incl. head-local TP2 geometry (12→2 head map) and the no-bf16-VALU fp16-V workaround (our constraint too) | **REFERENCE (primary) for WO-05** — port patterns, don't rewrite from CUDA |
| `src/ops/common/allreduce.cu` + `split_launch.h` | TP2 transport: 3-phase event choreography over `cudaMemcpyAsync` D2D UVA; `enable_peer_access()` probes both directions and **falls back cleanly when P2P is absent** (proven on donor's 2×5090 = our G-AMD-5 situation) | **ADAPT** for TP2 phase; the no-P2P fallback is exactly our box |
| flag-sync split-graph transport (S9/S9b) | per-device graphs + P2P flag spin; **default REVERTED after S9c wedged a card (MODE1 reset FAILED, box rebooted)** | **REJECT for now** — defer until probes + root cause; never under ninfer-serve without wedge-watcher |
| `docs/gfx906/PORT-AUDIT.md` | completed census: 224 TUs = 150 mechanical / 14 adapt / ~24 rewrite / ~100 deleted | **REFERENCE** — cross-check our ~26-file sizing; adopt its census method |
| `docs/gfx906/TP2-SLICES.md` | donor cherry-pick plan: S1 transport → S2 shards → S3 split ops → S4 attention geometry → S5 runtime → S6 parity → S7 perf; acceptance bars + rollback per slice | **ADOPT as the TP2 phase sequence** (roadmap amendment, §3-E) |
| `docs/gfx906/STAGE1..10-LOG.md`, SERVE-DEBUG-LOG | every dead end with numbers (bf16-VALU absence, wave64 shuffles, graph pathology, memory-type requirements for flag staging) | **REFERENCE** — read before each corresponding step of ours |
| `tools/tp2/{p2p,transport,capture,replay}_probe.cu`, `parity.cpp` | small probes that answer P2P/UVA/capture/parity facts in minutes of GPU | **PORT** as WO-V2 (§3-D); they are the cheap way to re-measure 6.4.1 findings on our 6.2.0 |
| `kv_capacity.cpp` capacity planning | derives from live available bytes (good) but carries policy headroom/reservation constants | **CONDITIONAL** — VRAM-LAW review before landing: planning-only, must never refuse a launch; main's preflight region stays canonical |

## §2b Reuse census — file-level (deeper pass, 2026-09-12, agent3)

**ISA verdict (the make-or-break question): the fork already ships gfx900-runnable fallbacks.**
- `gfx906_fdot2` (rowsplit_tiled_gemm_gfx906.cuh:50-63): `#if NINFER_GFX906_COMPAT && __HIP_DEVICE_COMPILE__` → `v_dot2_f32_f16` asm; `#else` → FP16 unpack + FMA. **Compiles and runs correctly on gfx900 with zero kernel edits** (perf slice optional later).
- `gqa_gfx906_sdot4` (gqa_attention_decode_i8_gfx906.cuh:49-59): `#if __gfx906__` → sdot4 builtin; `#else` → scalar int8 dot. Same verdict.
- `warp.cuh` (94 LOC): logical 32-lane subgroups in wave64, masks accepted-and-ignored, central `warp_sum/warp_reduce_sum/warp_max`. **Direct reference for the G-AMD-10 wave64 reduce defect family.**

**Adoptable volume (LOC, counted from bytes):**
| Block | Files | LOC |
|---|---|---:|
| gfx906 kernels (GEMV + tiled GEMM) | q_gemv_gfx906.cuh (777), rowsplit_tiled_gemm_gfx906.cuh (508) | 1,285 |
| TP2 transport | allreduce.cu (589) + allreduce.h (213) + split_launch.h (115) | 917 |
| Graph/executor | decode_graph.{h,cpp} (145+519) + concurrent_executor.h (1,300) | 1,964 |
| Shard plumbing (pure C++) | storage_layouts.cpp (534) + binder.cpp (221) + materializer.cpp (410) | 1,165 |
| Runtime sizing | kv_capacity.cpp (160) | 160 |
| Compat/shims | hip_compat.h (93 defines) + warp.cuh (94) | ~300 |
| Probes/parity | tools/tp2/{p2p,transport,capture,replay}_probe.cu + parity.cpp | small |

**ARTIFACT MISMATCH (new, blocks golden transfer):** fork runs `qwen3_8_27b.ninfer` =
18,210,531,328 B, sha256 `eec3956499…`; OUR registry artifact = 20,437,336,576 B, sha256
`0634abb070…`. Same model name, different bytes. Consequences: their md5 goldens, t/s
numbers and parity envelopes do NOT transfer to our artifact; the CODE does. All gates
stay comparative against our own artifact and tp1 control (protocol-compatible).

## §3 Plan adjustments (proposed — for coordinator ruling)

**A. WO-04 (agent3, in flight) — unchanged in substance; step 0 DONE.**
Step 0 landed `f5035616` on `amd/wo-q3hip` (pushed): both files patched
(`q3_rowsplit_storage.h:24`, `q2_rowsplit_storage.h:26` — one line lower than the WO's
cites; sha256 base-verify passed byte-exact). Falsifier pasted both sides: BEFORE exit 1 /
20 errors (15× `amd_hip_bf16.h` `__ocml_*`, 3× `__float2half_{rz,rd}` redefinitions);
AFTER exit 0. Command needed `-Isrc -Isrc/common/hip_shim` and the real TU is
`src/ops/launcher/embed_gather.cu` (no `src/ops/kernel/embed_gather.cu` in-tree).
Steps 1–6 proceed as written. **Fork addition to step 5:** their `embed_gather` kernel
(`src/ops/kernel/embed_gather.cuh`) is a working GCN5 gather — reference its block/launch
shape when building the Q3 arm (the codec still comes from our `q3_decode_one`, which the
fork does not have).

**B. WO-05 (agent4, attention) — add a fork-reference gate before design.**
Read `gqa_attention_{decode,prefill}_bf16_gfx906.cuh`, `_geometry.cuh`, and the fp16-V
plane hunks first. Expect net-new work only where Vega 10 lacks Vega 20 instructions: their
i8 decode path is `v_dot4`-guarded with SIMT fallbacks — the fallbacks are the porting
template. bf16-KV admission bar unchanged.

**C. NEW WO-V1 (zero GPU, CI lane): shim gap closure.**
Land the 6 missing TP2 surfaces in `src/common/hip_shim/cuda_runtime.h` under RULING-1
(this touches agent2's file → coordinate or route to agent2; content sourced from fork
`hip_compat.h`). Proof cell: compile-only, plus the `.type` field check on 6.2.0.
Unblocks every later transport step; costs hours.

**D. NEW WO-V2 (one coordinator-stamped GPU session, minutes): port the 4 probes.**
`p2p_probe` (confirm G-AMD-5 staging behavior), `transport_probe` (3-phase event
choreography correctness at tp2 on 2 dies), `capture_probe` (does ROCm 6.2.0 reject
cross-device capture the way 6.4.1 does?), `replay_probe` (graph executor device routing).
Output: a measured-facts table that gates ALL TP2 engine work. This replaces assumption
with measurement for the fork's two scariest findings.

**E. Roadmap amendment — TP2 phase sequence.**
Replace whatever ordering the TP2 phase currently implies with the fork's proven one:
transport (WO-V1/V2) → shard maps/storage slices → split-op plumbing (groupwise formats
only) → attention head-local geometry → per-device KV layouts → **tp2 EAGER first**
(their production default after S9c) → graphs/flag-sync only if WO-V2 measures say 6.2.0
behaves and the wedge class is excluded. Parity bar: adopt their comparative envelope
(argmax / KL / 1−cos vs our own tp1 control; no CUDA goldens — matches registry rule 2).

**F. Kernel-rewrite freeze list (throwaway avoidance).**
Until a lane proves a fork family unusable on gfx900, no lane writes from scratch:
q4/q5 GEMV + tiled GEMM, w8 decode, q6/bf16 dispatch, attention SIMT routes, GDN
projections, allreduce. Each affected WO gains a first step: "diff the fork family; adopt
or document why not." Net-new work is then limited to: q3/q2 compat (genuinely ours),
`__gfx906__`→gfx900 ISA fallbacks, and our shim/CI integration.

## §4 Risks / do-not-take

- **R1** Flag-sync-in-graphs: S9c wedged an MI50 hard (spin wave, failed MODE1 reset,
  SysRq reboot). Rejected until probed AND root-caused; their own default is eager.
- **R2** ROCm 6.4.1 behavior claims on our 6.2.0: capture rejection semantics, pointer
  `.type` field, `hipErrorPeerAccessAlreadyEnabled` existence — all re-measured by WO-V2
  before anything depends on them.
- **R3** Perf expectations: Vega 10 ≈ 484 GB/s vs MI50 ≈ 1 TB/s, no V_DOT, 7.98 GiB/die —
  plan for a fraction of their t/s. **Corrected by §2b: ISA fallbacks are already shipped in
  the fork (fdot2/sdot4 `#else` arms) — net-new ISA work for CORRECTNESS is zero; V_DOT
  fast paths are an optional later perf slice.** Velocity comes from NOT writing kernels,
  not from their numbers.
- **R4** VRAM LAW: any adopted sizing constant (kv_capacity policy headroom, graph
  allowance 8×, staging sizes) is planning-only and subject to the canonical-preflight
  diff; estimate-based launch refusal stays banned.
- **R5** M4 tension (needs coordinator ruling): 19.03 GiB artifact vs 15.97 GiB tp2
  ceiling on this box vs engine 2-rank limit vs M4 = 4 devices. The fork is tp2-only;
  4-device sharding would be net-new engine work either way. Decide early — it bounds
  what "first real serving" means here.
- **R6** Anti-resurrection: adoption is cherry-pick of named files into RULING-1 lanes;
  `git diff main -- src/runtime/tp2/tp_engine.cpp src/runtime/tp2/tp2_budget.h` must stay
  empty through every step (the fork carries neither file — keep it that way).

## §6 Immediate asks of the coordinator

1. Rule on §3-A..F (esp. C, D as new WOs; E as roadmap amendment; R5).
2. Note WO-04 step-0 SHA `f5035616` (pushed) — gemini's PG-1 exclusion list can take the
   two file names from the commit body.
3. agent3 proceeds with WO-04 steps 1–2 (CPU goldens) meanwhile — CPU-only, no grant
   needed, unchanged by this proposal.
