# docs/157 — sglang TP2 prefill cross-examination: why 3,000 t/s vs our 850, and the plan to close it

**Status:** ACTIVE work order (docs/157 claimed from coordinator 2026-09-08; NOT registered in docs/59 per user directive).
**Owner:** A1 (user-directed lane). P0 measurement window queued after A2's W5 battery — written grant + guard discipline applies.
**Companion tool:** `tools/p2p_probe.cu` (this commit) — ~5 s capability + bandwidth probe for the P0 window.
**Host event:** D-1 (driver 610.57.04 + patched modules + GRUB `iommu=pt` + reboot) is user-green-light-only; no session touches GRUB/driver.
**Mission:** explain the reported prefill gap on the 2×5060 Ti TP2 deployment (ours ≤850 tok/s vs sglang ~3,000 tok/s, same model, same GPU pair), attribute it in OUR code with measurements, and land the changes that close it. Target: **≥2,500 tok/s prefill on TP2 at short context, ≥1,500 at 130k**, attribution-backed — not a blind kernel rewrite.
**Trigger intel (user report, UNVERIFIED — treat as hypothesis seed only):** a user runs
`RadixArk/Qwen3.8-27B-NVFP4` on dual 5060 Ti, PCIe 4.0 x8 **with a "p2p patch"**, sglang (reported as "5.7.0"),
+300 core/+3000 mem OC, 180k context — **3,000 tok/s prefill, 40 tok/s decode (no MTP)**.

---

## 1. The parity arithmetic (why 3,000 is not magic — it's what we should already get)

| datum | value | source |
|---|---|---|
| our prefill, **single RTX 5090**, qwen3_8_27b NVFP4, @7.7k prompt | **8,340 tok/s** (measured, 5 seeds) | `worktrees/*/docs/performance.md` §qwen3_8_27b/nvfp4 MTP0 table |
| SM count | 5090 = 170; 5060 Ti = 36 | docs/122 §1 (36 audited), NVIDIA specs |
| naive SM-scaled expectation, 2×5060 Ti | 8,340 × (72/170) ≈ **3,530 tok/s** | arithmetic |
| sglang user report, 2×5060 Ti TP2 | **~3,000 tok/s** | user report (UNVERIFIED) |
| our TP2, 2×5060 Ti | **≤850 tok/s** (user statement; no TP2 prefill table found in PROGRESS.md/results) | user report (UNVERIFIED — P0 must re-measure) |

**Reading:** sglang on the same GPU pair is running prefill at roughly our own per-SM efficiency from the 5090
campaign. Our TP2 path delivers ~24% of that. **The gap is ours, not the hardware's.** And the sglang user did
it on PCIe 4.0 x8; we have **PCIe 5.0 x8 = 2× the transport budget** (31.5 vs 15.8 GB/s per direction, `nvidia-smi`
gen-max 5, width-max 16, current width 8; topology PHB through root complex — verified 2026-09-08).

---

## 2. What sglang actually does (scoped from source, ref/sglang @ `a6b54281`, 2026-09-08 main)

> The reporter said "5.7.0"; we scoped main. Mechanisms below are structural (vLLM-lineage custom AR +
> chunked prefill + flashinfer fp4 + fla GDN) and stable across versions; exact-version diff is a P0 item.

### 2.1 Collectives — the "p2p patch" story
- `CustomAllreduce` (`python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`):
  - `_MAX_CAR_SIZE = 8192*1024` = **8 MB** (line 42): payloads ≤8 MB go through custom AR, not NCCL.
  - `should_custom_ar` (line ~268): **for `world_size == 2` custom AR is used regardless of NVLink** —
    `if self.world_size == 2 or self.full_nvlink: return inp_size <= self.max_size`.
  - Buffers are **IPC-registered device buffers** (`create_shared_buffer`, `ops.register_buffer`) — peer reads
    the other GPU's memory **directly**; no host staging.
  - `full_nvlink` feeds the C++ kernel's algorithm choice (one-shot vs two-shot).
- `can_actually_p2p` (`custom_all_reduce_utils.py:148+`): sglang **does not trust** `canDeviceAccessPeer` — it
  spawns producer/consumer processes and performs a **real 1 KB cross-GPU write test** through IPC-mapped
  pointers, caches the verdict on disk (`~/.cache`, "generating/reading GPU P2P access cache"), with
  `SGLANG_SKIP_P2P_CHECK=1` to trust the driver.
- **The "p2p patch"** (external to sglang — a driver-level change making GeForce pairs do PCIe P2P DMA; public
  artifacts of this class: tinygrad's `p2p` 4090 patch and descendants, BIOS ACS-off, `nvidia-smi`/registry
  enablement). With it, the real-P2P test passes ⇒ custom AR goes card-to-card AND NCCL (payloads >8 MB)
  selects its **P2P transport** instead of SHM-through-host. Without it, the same machinery silently degrades
  to host-mediated copies.
- Consequence on their box: all 128 per-chunk layer allreduces move **directly over PCIe between the cards**,
  no host round-trip, no extra copy, no NCCL SHM staging.

### 2.2 Prefill scheduling
- Chunked prefill sized in **thousands of tokens**: `max_prefill_tokens = 16384`
  (`srt/arg_groups/fields/schedule.py:70`), `chunked_prefill_size` default resolved in hooks (family
  4096–8192; `-1` = whole prompt in one forward), plus `enable_dynamic_chunking`. ⇒ 4–8× fewer chunk
  traversals than ours, bigger GEMMs, better TC utilization, amortized per-layer overheads.
- Overlap scheduler hides CPU scheduling under GPU work (our `std::barrier` design has no such cover).

### 2.3 NVFP4 weights
- ModelOpt NVFP4 checkpoints dispatch to **flashinfer `mm_fp4` (w4a4)** — activations quantized to FP4 too,
  full-rate Blackwell FP4 tensor-core GEMM (`srt/layers/quantization/modelopt_quant.py:100-170`, incl. a
  `try_qwen3x_nvfp4_gemm` special-case fusion and `mm_bf16_fp4` w4a16 fallback).
- KV cache stays **bf16 by default** (fp4-KV exists as an option: `fp4_kv_cache_quant_method.py`) — i.e. their
  prefill attention writes bf16 pages, **no quantize/commit pass on the critical path**.

### 2.4 GDN (hybrid linear attention) layers
- `chunk_gated_delta_rule` from the **fla (flash-linear-attention) triton kernel family**,
  `CHUNK_SIZE = 64` (`sglang/kernels/ops/attention/fla/chunk.py:33`), behind
  `linear/kernels/gdn_triton.py` (+ cuTeDSL and flashinfer variants), with a **fused GDN projection kernel**
  (`triton_gdn_fused_proj`, imported by `models/qwen3_next.py:11`).

### 2.5 Full-attention layers
- Standard paged bf16 flash-attention prefill backends (flashinfer/FA/trtllm families,
  `layers/attention/attention_registry.py`). No materialize/quantize detour.

---

## 3. What we do today (scoped from repo @ `5fd8fe8a`)

| # | mechanism | our fact | sglang's fact | delta class |
|---|---|---|---|---|
| 1 | **Collective transport** | `OneShotAllReduce` caps at **65,536 elems = 128 KiB** ("covers up to T=12 for 5120 hidden" — comment's own admission, `one_shot_allreduce.h:15`); GPU writes **host-mapped** payload, polls peer flag over PCIe. **Everything bigger → NCCL** (`tp_group.cpp:268-280`). **No `cudaDeviceEnablePeerAccess`/`canAccessPeer` call anywhere in `src/`** (grep verified) ⇒ on a GeForce PHB pair NCCL silently picks **SHM-through-host** for the ~5 MB per-layer payloads: 2 extra copies + root-complex traversal + per-call latency, **×128 per chunk** (64 layers × o_proj + gdn out_proj) | custom AR IPC ≤8 MB + NCCL P2P above — **all direct card-to-card** | **P2P** |
| 2 | **Prefill chunk** | default **1024** (`src/serve/serve_options.h:36`; older builds 512, docs/102 §2), alignment-constrained (`layouts_impl.h:547`) | thousands (4096–16384 budget) | scheduling |
| 3 | **Host sync model** | `std::barrier sync_bar(2)` between steps; one request at a time, batch-1 prefill (docs/102 §1) | continuous batching + overlap scheduler | scheduling |
| 4 | **NVFP4 GEMM** | w4a4 exists (`ops/linear_add/nvfp4/nvfp4_w4a4_plan.h`, `nvfp4_linear_add_w4a4_launch`) — **coverage on the TP2 prefill route unverified** | flashinfer `mm_fp4` w4a4 everywhere + qwen3x fusion | parity plausible — verify |
| 5 | **KV write path in prefill** | variant-dependent: KVarN pays `quantize_tile_kernel` = **98.9% of the kvarn-vs-bf16 prefill gap** (docs/113 nsys: 3.61 s of 3.65 s @21.8k tokens); NVFP4-G16 KV landed recently (docs/155-era); which variant the TP2 deployment runs is the P0 question | bf16 pages, zero commit cost | **KV dtype** |
| 6 | **GDN chunked kernels** | custom CUDA, launch config now SKU-aware (`chunked/output.cu:7-10`, runtime SM tracking) | fla triton CHUNK=64 + fused proj | unmeasured — bench |
| 7 | **Attention prefill** | flash/materialize paths per variant | bf16 FA paged | covered by #5 |

**Transport arithmetic (why #1 is first):** per 1024-token chunk, per rank, per allreduce payload =
1024×5120×2 B = 10.5 MB ⇒ 128 payloads = **1.34 GB moved per chunk**. At their effective PCIe-4 P2P
(~13 GB/s) ≈ 103 ms pure transport floor; at our SHM-through-host (extra staging copies, realistic
~5–8 GB/s effective) ≈ 170–270 ms **plus** per-call latency tax ×128. At 850 tok/s a chunk costs 1.2 s —
transport alone plausibly explains 15–25%+ of it, and its removal is what their patch bought them.
PCIe-5 x8 direct P2P floor for us: **~52 ms/chunk**. H1's profiling step will put real numbers on every sink.

---

## 4. Hypothesis ledger (each = claim · arbitration · expected magnitude)

| id | hypothesis | arbitration | if true, expected |
|---|---|---|---|
| **H1** | NCCL SHM (no-P2P) allreduce dominates our TP2 prefill | nsys TP2 prefill + `NCCL_DEBUG=INFO` transport log + per-AR timer | 15–30% of chunk time + latency tax |
| **H2** | chunk 1024 is too small (launch/efficiency tax ×64 layers) | sweep `--prefill-chunk` 1024/2048/4096/8192 at fixed prompt | 10–25% |
| **H3** | KV write path (quantize/commit on critical path) at TP2 deployment variant | nsys kernel table vs variant matrix (kvarn/int8/nvfp4-g16/bf16) | 0–15% (variant-dependent; docs/113 already proved the kvarn case) |
| **H4** | our w4a4 GEMM coverage/efficiency < flashinfer mm_fp4 | GEMM microbench vs kernel-level trace on prefill shapes | 0–2× on GEMM fraction |
| **H5** | GDN chunked kernels slower than fla-class on 36 SMs | head-to-head microbench, correctness oracle first | 0–20% |
| **H6** | host barrier + non-overlapped scheduling adds fixed per-chunk cost | timer around sync_bar regions | few % at prefill sizes |

Ordering follows expected payoff ÷ risk: **H1 → H2 → H3 → H4 → H5 → H6.** H1 and H2 are also the two the
sglang report directly implicates (p2p + big-chunk scheduling).

---

## 5. Implementation plan

### P0 — Measurement & probes (no behavior change; CPU + one granted GPU window)
1. **Re-measure our TP2 prefill** (the ≤850 figure is user-reported): qwen3_8_27b NVFP4 artifact, TP2, 7.7k/64k/130k prompts, same method as performance.md MTP0 table. This becomes the baseline row.
2. **P2P probe** (2 s, needs coordinator GPU grant): `cudaDeviceCanAccessPeer` + real 1 KB mapped write, exactly sglang's `can_actually_p2p` protocol. Records whether our driver/ACS already permits P2P (some 50-series pairs do) — this decides whether P1a needs a driver-level patch (user action, host root) or only code.
3. **nsys attribution** on one 8k-prompt TP2 prefill: kernel time table split {GEMM, attention, GDN, NCCL/AR, quantize/commit, other}; `NCCL_DEBUG=INFO` run logged to confirm SHM vs P2P transport selection. Fill the H1–H6 attribution table.
4. **Version check**: reporter's "sglang 5.7.0" — diff the four scoped mechanisms against that tag if obtainable; note deltas here.

### P1 — Direct-P2P collectives (attacks H1)
- **P1a-host prerequisite (USER decision, host root — outside repo scope):** the identified patch is
  **`github.com/aikitoria/open-gpu-kernel-modules`** (tinygrad-p2p lineage; HEAD `461d638` 2026-09-05,
  base driver **610.57.04**; clone kept at `ref/ogkm-p2p`, 168 MB). BAR1-mapping P2P for RTX 30/40/50
  series **explicitly including sub-xx90 models**; same-generation pairs need **only the patched kernel
  modules** (libcuda patch is for mixed-generation only). Host readiness audit (verified 2026-09-08,
  read-only):

  | requirement | our state |
  |---|---|
  | Resizable BAR / Above-4G, BAR1 ≥ VRAM | ✅ **BAR1 = 16 GB on both GPUs** (`lspci` region 2, devices 01:00.0/02:00.0, `2d04` GB206) |
  | SecureBoot off (unsigned modules load) | ✅ disabled (`mokutil`) |
  | IOMMU passthrough | ⚠️ IOMMU **active**, no `iommu=pt` in cmdline → needs `intel_iommu=on iommu=pt` + `update-grub` + reboot |
  | Driver base match | ⚠️ installed = **580.173.02 open module**; fork base = **610.57.04** → decision D-1 below |

  **D-1 (recommended): upgrade driver to 610.57.04 + `./install.sh`** — matches the fork exactly; our
  CUDA 13.1 toolkit is forward-compatible (min driver ~580). Alternative: backport the P2P hunks
  (`nv-p2p.c`, `nv-dma.c`, `nv-pci.c`, `nv.c` `nv_dma_remap_peer_mmio`/`rmp2pdefines.h`) onto 580.173.02 —
  proven portable (tinygrad's original targets 550.54.15) but it is maintenance we don't need.
  Coordination: driver swap + reboot = shared-box event — schedule through coordinator when A2's W5
  window is idle. Note the fork README's warning: `iommu=pt` removes DMA isolation (acceptable on this
  single-user dev box, user's call).
- **P1a transport (repo work, after P1a-host):** enable `cudaDeviceEnablePeerAccess` at TP init behind the
  gate; add a **device-side two-shot P2P allreduce** (bf16, payloads 128 KiB…32 MB) beside
  `OneShotAllReduce`, peer/IPC-registered buffers, in-stream, no host poll on the payload path. If the
  P0 probe still fails on the patched driver ⇒ document the exact blocker and proceed with P1b.
- **P1b NCCL hygiene (probe-independent):** verify/force NCCL transport choice via env (`NCCL_P2P_LEVEL`, channel settings); measure SHM vs best-available on this board; keep NCCL as fallback for >32 MB.
- **Feature gate:** `NINFER_TP_P2P=probe|on|off` (default `probe` = detect, log verdict, choose). Rollback = `off` = today's path byte-for-byte.
- **Gates:** greedy byte-identical outputs vs `off` at ctx 500/5k/20k (sha discipline per docs/126 runbook); prefill t/s improvement ≥ the P0-attributed H1 share minus margin; no VRAM regression beyond peer-mapped buffers (~MBs).

### P2 — Chunk sizing (attacks H2)
- Sweep `--prefill-chunk` {1024, 2048, 4096, 8192} on TP2 NVFP4; watch the VRAM ledger (activation temps + KVarN materialize temp scale with chunk; 16 GB/rank is the constraint, docs/52 class).
- Land the best default **per deployment VRAM class**; keep alignment constraint (`layouts_impl.h:547`) intact.
- Gates: byte-identical shas across the sweep at fixed chunk; no OOM at 200k ctx; TTFT and t/s both reported.

### P3 — KV write path off the critical path (attacks H3, only if P0 implicates the deployed variant)
- If TP2 deployment runs kvarn/int8 KV: execute the docs/113 revised steps (quantize straight into destination pages, commit on side stream behind event dep — producing chunk reads its own keys via raw/tail path) or route long-prompt TP2 serving to NVFP4-G16/bf16 KV per docs/104/117.
- Gates: FP64 oracle first (docs/113 step 3′), then variant A/B; capacity win tracked separately from t/s.

### P4 — GEMM coverage audit (attacks H4)
- Verify every prefill linear on the TP2 NVFP4 route actually dispatches `w4a4` (not `small_t`/decode variants): add a one-shot dispatch-coverage log at init; microbench our w4a4 vs flashinfer `mm_fp4` on our exact shapes (5120→17408 swiglu, o_proj 5120, qkv). If a shape class loses >10%, port or retune; else close.

### P5 — GDN head-to-head (attacks H5)
- Correctness oracle first, then bench our chunked kernels vs fla-triton semantics on 36-SM shapes (proj fusion included). Land only what beats ours on the real layer shapes.

### Exit criteria
- P0 attribution table committed (results/ doc) explaining ≥80% of the baseline gap.
- P1+P2 landed: **TP2 prefill ≥2,500 tok/s @7.7k / ≥1,500 @130k** on the deployment VRAM class, all shas green, VRAM ledger clean. Stretch (P3–P5): close to the ~3,530 hardware-parity line.

---

## 6. Protocols that bind this lane (from AGENTS.md + handoff, all still in force)
- GPU: **zero card contact** until coordinator grants a window (A2 holds them for W5 S1 — coordinator confirmed 2026-09-08). P0's probe + nsys + baselines batch into ONE granted window.
- Doc number: claimed from coordinator (this draft = unnumbered by design).
- Disk: `df -h /` before any build >2 GB (8.1 G free at drafting — the sglang clone lives at `ref/sglang`, 168 MB, shallow, untracked).
- Byte-identical sha gates for any collective/scheduler change; rollback env gates on everything.
- Unverified-user-numbers stay flagged as such until P0 replaces them with measured rows.

## 7. Prior art to consult when executing P1
- sglang `custom_all_reduce*.py` + their CUDA one/two-shot AR (vLLM lineage) — algorithm reference for buffer registration + flag protocol; license-compatible clean-room implementation required.
- sglang `can_actually_p2p` probe — the exact runtime test our `probe` gate should mirror.
- docs/17/18 (our one-shot AR design + results) — why the 128 KiB host-mapped design existed (decode-sized payloads) and why prefill outgrew it.
- **P2P driver patch, identified:** `aikitoria/open-gpu-kernel-modules` (§5 P1a-host) — tinygrad
  `550.54.15-p2p` lineage, BAR1 path, RTX 30/40/50 series. This is almost certainly what the sglang
  reporter runs (their "p2p patch"). Host-level prerequisite; user decision + coordinator-scheduled reboot.
