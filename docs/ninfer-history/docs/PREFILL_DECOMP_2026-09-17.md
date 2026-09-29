# PREFILL DECOMP — the 3.5x-vs-1.8x dilution, decomposed (no-GPU desk, 2026-09-17, amd/tp4-cure)

**Seat:** NO-GPU analysis agent. **Method:** banked-artifact mining (`results/amd/coherence/`) +
code-read of the prefill chunk. Zero GPU work, zero builds, zero src/ edits, no process touched.
**Question:** prefill is the co-equal goal; tiled GEMM landed and serving prefill went
42.1 → 75.7 tok/s at plen 1996, but the isolated-kernel win was 3.5x — **where does the
dilution live, and what single measurement names its owner?** Method mirror:
`docs/amd/VERIFY_DECOMP_2026-09-17.md` (what that doc did for decode's verify round, this does
for the prefill chunk).

**Binaries named by row:** current certified `409450b83ffa4be8` (tiled race-fix; TILED_DATUM /
TILED6 anchor, M>=64 dispatch default ON in this bin).

---

## 1. The chunk math, derived from the anchor (TILED_DATUM_row.txt, TILED1_serve.log)

plen = 1996, `--prefill-chunk 128` → **16 chunks = 15×128 + 1×76**. ttft **26 354.5 ms** →
75.7 tok/s ⇒ **mean 1647 ms per chunk** (per rank; all four ranks lockstep through 2 ARs/layer).
The serve-log progress marks give the per-segment walls (TILED1_serve.log, req 2):

| segment (tokens) | wall | per full chunk |
|---|---|---|
| 128 → 640 | 6.322 s / 4 chunks | **1581 ms** |
| 640 → 1152 | 6.438 s / 4 | **1610 ms** |
| 1152 → 1664 | 6.557 s / 4 | **1639 ms** |
| 1664 → 1996 | 5.460 s / 2 full + 1×76 | ~1650 + **~500 ms last-chunk extra** |

Growth term: **+~15 ms/chunk per +1024 visible tokens** — real but second-order (its byte
equivalent, 16 FULL layers × 1024 tok × 1 kv-head/rank × 256 × 2 × 2 B = 16.8 MB, is 0.05 ms at
the 354 GB/s copy anchor; whatever grows is ~300x its byte cost — a second latency-class pocket,
see §4-G). Last-chunk extra ≈ +500 ms: lm_head + allreduce_argmax + MTP finalize + AR steps +
the 12-token sequential delta-net tail (76 = 64 + 12, `gated_delta_net.cpp:297-312`).

**A/B cross-check (the GEMM slice, no new measurement needed):** smallt-force arm
(`NINFER_A16_FORCE_SMALLT=1`): ttft 47.41/48.03 s (42.1/41.6 tok/s, thermally flat) vs tiled
26.35 s. Δ = **1316-1355 ms/chunk is exactly the GEMM-slice savings** — non-GEMM is identical
between arms. With the isolated-kernel ratio 3.5x (TILED-GEMM-1 bench, T1 VERDICT row):
G_smallt = 3.5·G_tiled ⇒ **G_tiled ≈ 526-542 ms/chunk, G_smallt ≈ 1842-1897 ms/chunk**.
Cross-check vs FLOPs: the ROUTE PROOF's five per-rank problems at M=128
(`[TILED]` lines: n=4096 k=5120 ×48 GDN + n=5120 k=1536 ×64 out/o_proj + n=8704 k=5120 ×64
gate_up + n=5120 k=4352 ×64 down + n=3584 k=5120 ×16 attn_in) are
**1556 GF/chunk/rank** ⇒ 1556/0.532 s = **2.93 TF/s = the bench V0 class** (SWEEP_row:
2366-3103 GF/s, weighted ~2.9 TF/s). **The tiled GEMM in serving runs AT bench class — the M1
lesson reproduces on the prefill side: the kernel is not the dilution.** (SWEEP caveat: the bench
was cache-served at 5-6.8 TB/s effective; serving streams 3.43 GB/rank/chunk of weights from
DRAM, but the DRAM floor at 368.7 GB/s is only 9.3 ms — 50x slack under the issue-bound 530 ms.
Weight streaming is NOT the GEMM's binding term at M=128.)

## 2. Per-chunk op census (code-read, this worktree; TP4, T=128)

Per GDN layer (48, `gdn_mix_tp` prefill arm :2448-2665): input rmsnorm + control proj
(replicated, :2284) · tiled GEMM n=4096 k=5120 (:2451) · `tp_unpack_gdn_qkvz` (:2487) ·
`causal_conv1d_silu_split3` w/ in-place conv state (:2502) · 2× l2norm + 3-kernel chunked WY
delta net (`gated_delta_net.cpp:253-313` → `chunked/launch.cu:17-75`: prepare_wy_wu →
state_passing → output; in-place recurrent state, NO per-column snapshots at prefill — the
snapshot ops are the verify arm :2415) · `tp_unpack_gbeta_strided` (:2531) · gated rmsnorm
(:2551) · tiled GEMM n=5120 k=1536 (:2556) · **AR#1** ncclAllReduce+axpy (:2600).
Per FULL layer (16, `attn_mix_tp` :2038-2249): rmsnorm · tiled GEMM n=3584 (:2050) ·
`tp_unpack_qkv_strided` (:2058) · q/k rmsnorm · rope · gqa prefill (token-block grid,
`gqa_attention_prefill.cu`) + KV append · sigmoid_mul · tiled GEMM n=5120 k=1536 (:2242) ·
**AR#2** (:2246). Every layer: `mlp_tail` (:2777-2801): rmsnorm · tiled GEMM n=8704 · silu_mul ·
tiled GEMM n=5120 k=4352 · **AR#3,4**. Per chunk: embedding, ids H2D, positions fill, MTP
prefill chunk (`mtp_prefill_chunk` :1337 — runs EVERY chunk when `--spec mtp`: stem fc + KV
projection + attention over T=128 ≈ one extra layer-class forward), final norm; **last chunk
only**: lm_head over the LAST TOKEN ONLY — `xf.slice(1, len-1, 1)` :3165, M=1 GEMV over the
62080-row vocab shard (635 MB/rank BF16) :3167 · `allreduce_argmax` :3174 · one 4-byte
next-token D2H inside the MTP finalize :3224-3227 · MTP AR steps. Chunk ends with a hard
`ctx_.synchronize()` :3305 (one per chunk; the driver loop `tp2_backend.cpp:2166-2205` re-enqueues
the next chunk only after the drain).

**Counts per chunk:** ~256 tiled GEMM launches (+~5 MTP-class) · **128 TP ARs** (2/layer × 64)
each **1.31 MB** bf16 (5120×128×2) = 43.7x verify's T=3 30 KB payload · ~1300 device ops total
(eager; `cudaStreamBeginCapture` exists ONLY in `src/core/decode_graph.cpp` — decode is
GRAPHS-ON, the prefill path is not). Host enqueue ~1300 × 1.56-1.83 µs ≈ 2-3 ms, plus a ~2 ms
device-idle bubble at each chunk entry (post-sync re-enqueue) — both noise.

## 3. Attribution table — per full 128-token chunk, per rank (sums to the measured 1647 ms)

| slice | ms (central) | basis |
|---|---|---|
| Tiled GEMM (5 problems, 256 launches) | **~530** (450-620) | A/B ΔGEMM ÷ (3.5−1) + 1556 GF ⇒ 2.93 TF/s = bench class (§1) |
| TP ARs: 128 × 1.31 MB | **~68-200** | banked class: host-staged PCIe ~3.13 GiB/s + RTT 98-101 µs (`ONESHOT_AR_notes.md:68`, W2 ring) ⇒ 0.53-1.56 ms/AR; the TP4/1.31 MB bandwidth term is UNMEASURED (tail risk ~400 ms if the 2-root PCIe topology halves it again) |
| GDN delta-net chunked (48 × [2 l2norm + 3 WY kernels]) | **~10-30** | ~11.5 GF/chunk/rank at 0.4-2 TF/s (12 v-heads × 2 chunks × ~9 MF); 3 dependent launches/layer |
| elementwise body (norms, conv, unpacks ×3, rope, silu/sigmoid) | **~12-25** | ~65 MB r+w per GDN layer at the 354.4 GB/s copy anchor |
| GQA attention (16 layers) + growth term | **~5-15** | bytes trivial; the measured +15 ms/1024-visible is the latency-class pocket (§4-G) |
| MTP prefill chunk (every chunk) | **~15-30** | code-read: one extra layer-class forward over T=128 |
| chunk head/tail (embed, finalnorm, per-chunk sync bubble, host enqueue) | **~3-6** | bytes + VERIFY_LAUNCH_AUDIT §5 launch law |
| **NAMED total** | **~640-900** | |
| **UNNAMED RESIDUAL** | **~750-1000 (central ~950, 58%)** | subtraction: 1647 − named |

**Where the 3.5x → 1.8x dilution lives (the mission question):** NOT in the GEMM — the A/B
proves the GEMM slice shrank by the full ~1.33 s/chunk and the residual rate is bench class.
The dilution is the **~1.1 s/chunk of non-GEMM wall that neither kernel touches**, and inside it
the two live suspects are (ranked): **(α) the eager per-op gap/launch-tail mass of the ~1300-op
dependency chain** — prefill is NOT graph-captured while decode is GRAPHS-ON, and this is the
same class VERIFY_DECOMP's H1 named for decode (unnamed 50-60 ms/verify-round at T=3; prefill
has 43x the AR bytes, 16 chunks, identical eager structure); **(β) the TP AR lockstep chain**
(128 × 1.31 MB collects whose TP4 bandwidth term has never been measured). **Exonerated by
arithmetic:** state copies (prefill conv/recurrent states update IN-PLACE, same tensor in/out
:2502/:2540; snapshot machinery is verify-only; 302 MB/chunk r+w ≈ 0.9 ms), conv (trivial
bytes/FLOPs, one kernel), lm_head (last chunk only, ~2-3 ms), logits D2H (one 4-byte token),
KV appends (MBs).

## 4. Ranked hypotheses for the ~950 ms unnamed residual

- **G — Eager inter-op stream gaps / launch-tail latency on the serialized body. RANK 1.**
  Per layer the wall carries ~16 ms of non-GEMM, non-AR time over ~15 µs-class-bytes ops —
  ~40-60 µs of exposed gap per op, the exact signature of an eager dependency chain paced by
  launch completion + memory latency. Decode's H1 (same structure, T=3) left 50-60 ms/round
  unnamed by the same instruments that named linears+ARs; M2/[TAIL] exists for verify, NOTHING
  exists for prefill (`tpv_probe` is a cudaGetLastError check; `[TILED]` prints route names,
  no times). **Kill: P1 below.** Fix if confirmed: Fix A (chunk graph capture).
- **H — TP AR bandwidth term at 1.31 MB × 128/chunk. RANK 2 (priced 68-200 ms, tail 400).**
  The only banked AR anchor is W2-class (98-101 µs RTT, ~3.13 GiB/s host-staged, 30 KB scale).
  At 43.7x payload the bandwidth term dominates and is unmeasured at TP4 over the 2-board PCIe
  topology. **Kill: P1's ar_us column directly (the vt.ar_begin/ar_end event pairs already
  bracket every AR at :2244-2247/:2437-2443/:2790-2795).** Fix if confirmed: Fix B.
- **I — GEMM below bench class in serving (cold-weight DRAM/TLB stalls in the streamed-weights
  path). RANK 3 — the A/B consistency (§1) argues against, but it rests on the isolated 3.5x
  carrying to serving.** **Kill: P1's gemm_us column.** If gemm_us >> ~700 ms/chunk, the lever
  flips to the tiled kernel's cold-weight streaming path (weights are "streamed, never staged",
  TILED_GEMM_notes §2 — a first-touch L2/TLB stall class the cache-served bench cannot see).
- **J — Delta-net/conv kernels slower than priced. RANK 4.** No banked per-op number exists;
  P1 prices both. Bounded by the body envelope (~100 ms class) — cannot own 950 ms alone.
- **K — Cross-rank arrival skew at the ARs (thermal or host-thread jitter).** OPTRACE measured
  nil skew at T=3 decode; P1 measures it at T=128 (ar_us minus the W2-class floor). Folded into H.

## 5. TOP prefill lever + fix designs + the decisive check

**Top lever: the unnamed ~950 ms/chunk (58% of the wall) — G/H above. Per the M1 law (no patch
before its decisive measurement), the lever's first move is one tracer boot, and the fix design
is staged on its readout.**

**DECISIVE CHECK P1 — prefill per-op event tracer, one boot, zero new kernel code.** The
`VerifyLayerTrace` scaffold (`text_context_impl.h:962-1060`) ALREADY has event pairs bracketing
every layer (`layer_begin/layer_end` :2928/:2986) and every AR (`ar_begin/ar_end` at all three
AR sites) — the prefill path simply never sets `vt.active` (gate :2886 is
`ph == Phase::Verify && vl_cols > 1`). Arm it for prefill (per-chunk drain is race-free by the
same argument as verify: the chunk-end sync :3305 completes every event), add ONE event pair
around each Variant GEMM call site (6 sites, or piggyback the `[TILED]` dispatch seam
`nvfp4_dispatch.cpp:132/:194`) and one chunk-wall pair in `prefill_impl` (:3093/:3288), print
`[PFTRACE] chunk=T wall= layers= ar= gemm=` per chunk + `[PFTRACE-AVG]` at request end.
Env: `NINFER_PREFILL_OPTRACE=1` (default off = byte-identical: every added line event-gated,
TILED6/T2 grammar). Run: banked `409450b83ffa4be8` posture minus nothing, plen-1996 probe,
cool window (TILED probe-b pattern, <60 s bursts).

Predicted P1 readout (S1, central): wall 1650 = gemm 530 + ar 130 + body/named 90 + gap ~900.
**Falsifiers that re-rank the levers on the same boot:** `gemm_us` > ~1100 ⇒ hypothesis I (the
kernel lever reopens); `ar_us` > ~250 ms/chunk ⇒ hypothesis H owns; `layers − ar − gemm` small
(<300 ms) and `chunk wall − layers` large ⇒ the tail/sync/MTP class owns; else G (gaps) owns.

**Fix A (G confirmed) — prefill chunk graph capture/replay.** Decode already runs GRAPHS-ON via
`DecodeGraphDefinition/Executable` (`src/core/decode_graph.cpp:58-79/:135-138`); prefill never
captures. Design: capture the chunk body (run_layers + final norm) once per (T, visible) shape
— T=128 recurs 15/16 chunks — replay per chunk; positions/base stay eager OUTSIDE the graph
(they are ~1 ms) so the GQA visible-envelope kernel args baked at capture stay valid per shape;
ARs stay inside (RCCL allreduce is stream-capturable; if gfx900 RCCL refuses capture, fall back
to per-layer-body graphs between ARs — the gap kill survives). Op order unchanged ⇒
bit-identical claim is checkable by the standing battery (no new numeric cell owed; RED/GREEN =
tok/s before/after + battery PASS, shas named per the closure law). Expected: removes most of
the gap share → **prefill 100-130 tok/s** at plen 1996.
**Fix B (H confirmed) — AR thinning, env first:** zero-code A/B of RCCL channel/topology env
(`NCCL_MIN_NCHANNELS`, `NCCL_P2P_LEVEL`, DEBUG=INFO to name the transport) on the same boot;
then code: fold the 128 axpy epilogues into the following rmsnorm's read (saves 128 launches +
1.31 MB r+w each) and, design-level, revisit W4 one-shot only after ONESHOT-W4 repair (T2 wedge
class — NOT adoptable today). Expected if H owns: 1650 → ~1300-1500 ms → **85-98 tok/s**.
**Fix C (I confirmed) — cold-weight streaming path in the tiled kernel** (LDS-stage the u64
code loads per k-tile, TLB-friendly weight layout): out of scope until P1 prices it.

**Predicted post-fix band: 100-130 tok/s at plen ~2000** (Fix A central), with the
today's-slices floor ~700-760 ms/chunk ⇒ **hard ceiling ~164-178 tok/s** without reopening the
GEMM kernel family (SWEEP closed tile shapes at 28.9%-of-nominal; the ceiling's GEMM term is
that measured class, not 10.75 TF/s).

## 6. Honesty row

- The 530 ms GEMM slice is a DERIVED number (A/B Δ ÷ ratio), not a per-op measurement; it is
  consistent with the FLOP/bench cross-check to ~5%, but P1's gemm_us is what makes it a row.
- The AR range 68-200 ms extrapolates a W2-class anchor to TP4 payload; the bandwidth term at
  1.31 MB over the 2-root topology has never been measured — P1 measures it.
- The delta-net/body/MTP rows are byte-and-FLOP estimates, no banked per-op number exists for
  any of them at T=128; P1 prices all three in the same boot.
- The unnamed residual (§3, ~750-1000 ms) is a subtraction number, exactly like VERIFY_DECOMP's
  pre-M1 residuals; its mass is the reason P1 is staged ahead of any fix.
- Growth term (+15 ms/chunk/1024-visible) is from serve-log segment marks (±50 ms print
  granularity); treated as second-order, owned by P1's per-chunk wall column.
- Line numbers are from THIS worktree's working tree (amd/tp4-cure) and may drift.
- This seat touched only this doc; the working tree carries the GPU window's live artifacts
  (results/amd/coherence/*, M1_clocks_sideband.log) — none cited as evidence beyond their rows.

---

Bases: TILED_DATUM_row.txt (75.7/42.1/41.6, ttft 26.35/47.41/48.03, ROUTE PROOF, T1 VERDICT) ·
TILED1_serve.log (segment marks) · TILED6_row.txt (certification) · SWEEP_row.txt +
TILED_SWEEP_notes.md (V0 2366-3103 GF/s, ceilings 354.4/368.7 GB/s, 10.75 TF/s anchor,
cache-served caveat, tile family CLOSED) · TILED_GEMM_notes.md (kernel design, weights streamed,
M>=64 seam) · ONESHOT_AR_notes.md (:68 host-staged ~3.13 GiB/s, RTT 98-101 µs; one-shot
world==2-gated tp_group.cpp:348) · VERIFY_DECOMP_2026-09-17.md (method, H1/H3 class, launch law)
· VERIFY_LAUNCH_AUDIT_2026-09-17 §5 (1.56-1.83 µs/launch) · code:
`src/targets/qwen3_6/impl/runtime/text_context_impl.h` (:962-1060 vtrace, :1337 mtp_prefill_chunk,
:2038-2249 attn_mix_tp, :2251-2666 gdn_mix_tp, :2777-2801 mlp_tail, :2998-3309 prefill_impl),
`src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp` (:253-313) +
`chunked/launch.cu` + `common.h:8` (kChunkSize=64), `src/core/multi_gpu/tp_group.cpp`
(:346-357), `src/runtime/tp2/tp2_backend.cpp` (:2162-2205), `src/ops/linear/nvfp4/
nvfp4_dispatch.cpp` (:47-49, :55-85, :132, :194), `src/core/decode_graph.cpp`,
`src/targets/qwen3_6_27b/impl/config.h` (:11-42, :66-67), `src/ops/launcher/
gqa_attention_prefill.cu`.
