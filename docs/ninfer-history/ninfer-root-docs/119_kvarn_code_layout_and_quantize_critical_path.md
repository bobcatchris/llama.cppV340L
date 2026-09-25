# 119 — KVarN code layout and the quantization critical path

Status: design proposal, 2026-08-31. Author: Agent 1 (session). For plan-owner decision.
Inputs: nsys attribution `results/113_prefill_nsys_attribution.md` (9167829f), Stage-2 decode
fixes (802322a4), and a read of `src/ops/kvarn/kvarn_tile_cuda.cu` + `kvarn_workspace.cpp`.

## 1. Correction to my own earlier recommendation

I reported that `quantize_tile_kernel` is 3.61 s = 99% of KVarN's prefill gap, and implied the
fix might be a layout change. **Reading the quantizer shows the layout is not where its time is.**

Per launch (`kvarn_tile_cuda.cu:311-345`), on a `kD=256 × kG=64` tile = 16 384 elements:

| phase | work |
|---|---|
| `fwht_along_channel` | 8 butterfly stages over 16 384 elements |
| **`kSinkhornIters = 16`** | **16 full-tile iterations** (`:27`), each touching all 16 384 elements plus per-row/per-col reductions |
| quantize pass | 2 sweeps × 256 channels × 64 keys (min/max, then RTN) |
| `TileShared` | `float X[256*64]` = 64 KB **plus** rA/cB/rA2/cB2/best_rA/best_cB ≈ **72 KB dynamic smem** |

So the dominant cost is the **16-iteration Sinkhorn algorithm**, not memory access. A key-major
layout would not reduce it by a single operation. Measured 165 µs per 16 384 elements ≈ 10 ns/
element is simply what 16 iterative full-tile passes over a 72 KB-smem tile costs.

## 2. The arithmetic that decides priority

From the clean nsys runs (21 829-token prompt, prefill-dominated):

| | GPU kernel time |
|---|---|
| bf16 prefill | 27.74 s |
| kvarn prefill | 31.38 s |
| of which `quantize_tile` | 3.61 s |

**31.38 − 3.61 = 27.77 s ≈ bf16's 27.74 s.** So removing quantization from the critical path
entirely — with no change to the algorithm, no numerics change, no layout change — lands KVarN
prefill on **exact parity with bf16**. That is the whole 12.4% gap, and it is the only option that
closes it.

## 2b. MEASURED RESULT — (b) below is implemented and it is NOT a win (a59de6df)

Steps (b) and (c) were implemented and measured with the same nsys recipe. The change does
exactly what was intended — memcpy op count fell by **43 648**, precisely
`341 pages x 2 heads x 16 layers x 4 copies`, confirming both D2D pairs were removed from both
the commit and hydrate paths — and correctness is clean (regress_unified.sh PASS for all three
variants, slice3_i8 and slice4_kvarn oracles PASS).

But it is **perf-neutral**:

| metric | before | after |
|---|---|---|
| total GPU kernel time | 30.53 s | 30.51 s |
| `quantize_tile_kernel` | 3.61 s | 3.60 s |
| total memcpy time | 3.341 s | 3.368 s |

The copies were ~10 KB each, so they cost launch slots, not time: removing 43 648 of them moved
nothing. **My "free wins" framing below was overstated** — (b) is free but worthless, and only
(a), actually moving the 3.61 s of Sinkhorn compute off the critical path, can close the gap.
Keep (b) (strictly less work, and it deletes a pointless 12 KB/page/head of D2D traffic), but do
not count it as progress toward the gate.

**Methodological warning for anyone reading wall numbers on this box:** the same binary measured
599.8 t/s pp under nsys and 699.7 t/s pp without it, and the nsys-vs-nsys comparison of two
perf-neutral builds differed by 12%. Wall-clock prefill t/s under nsys is far too noisy for the
±5% gate. **Use total GPU kernel time from `cuda_gpu_kern_sum`** — it is stable to ~0.1% here and
is what the attribution above is built on. This also means the owner's one-run protocol is not
adequate for prefill perf claims; prefill needs either kernel-time totals or 3-run medians.

## 3. Can quantization leave the critical path? (the dependency question)

The commit path (`kvarn_workspace.cpp:286-318`) is, per completed 64-key page, per head:

```
kvarn_quantize_head_k(k_tile, qk_scratch, ...)      // -> scratch
kvarn_quantize_head_v(v_tile, qv_scratch, ...)      // -> scratch
cudaMemcpyAsync(k_dst, qk, 8192, D2D)               // scratch -> page
cudaMemcpyAsync(v_dst, qv, 4096, D2D)
kvarn_store_scales_launch(...)
```

Two structural facts make overlap attractive:

1. **The consumer of a quantized page is a *later* attention over that page as history.** Within
   the prefill chunk that produced the keys, attention reads the new keys through the raw/tail
   path, not from the paged codes. So page P's quantization does not gate the current chunk's
   attention — only a subsequent one.
2. It is already a **separate write-path stage** operating on its own scratch buffers, so it is
   not entangled with the attention kernels' state.

Design options, cheapest first:

- **(a) Side-stream overlap.** Issue commit onto a second stream, with an event dependency only on
  the producer that filled `k_tile`/`v_tile`, and a join before any subsequent attention that could
  read that page as history. Sinkhorn is compute-bound and attention is TC+memory-bound, so they
  should co-schedule reasonably. Risk: SM contention — a 72 KB-smem, 256-thread kernel is 1 block/SM
  and will steal SMs from attention. Needs measurement; may need a concurrency cap.
- **(b) Remove the D2D copies.** Quantize directly into the destination page instead of scratch +
  `cudaMemcpyAsync`. Saves 12 KB of copy per head per page and 2 launches. Small but free, and it
  also removes the scratch buffers.
- **(c) De-batch the head loop.** `for h in heads` serializes ~5 launches per head; with 2 heads
  that is ~10 sequential launches per page commit, each ~165 µs of work but with launch latency
  between. A single launch over (head × page) would amortize better.
- **(d) Reduce `kSinkhornIters`.** This is a **numerics change** and therefore out of scope for a
  perf task — but it is the single biggest knob (16 iterations is a lot), and if accuracy permits
  fewer, it is a direct multiplier on the 3.61 s. Flag for the owner rather than doing it here.

Recommendation was **(b) and (c) first, then (a)**. (b) has since been done and measured — see
§2b — and is perf-neutral, so **(a) is the only remaining candidate** for the prefill gap. Treat **(d)** as a separate accuracy-gated proposal.

## 3a. BLOCKER for option (a): the commit path hard-synchronizes the stream

Reading the driver (`kvarn_workspace.cpp:401-435`) — `gqa_kv_append_kvarn_and_commit` does, on
**every call**:

```
cudaMemcpyAsync(host_pos, positions.data, tokens*4, DeviceToHost, stream);
cudaStreamSynchronize(stream);                 // <- full pipeline drain
int t = 0;
while (t < tokens) {                           // <- HOST-side page-run loop
    ... compute page = pos/kG and the contiguous run ...
    gqa_kvarn_prepare_page_for_append(...);    // may hydrate a page
    gqa_kv_append_kvarn_with_host_pos(...);
    if (workspace.tail_count == kG) gqa_kvarn_commit_completed(...);   // <- quantize issued here
    t += run;
}
```

So quantization is issued from a host loop that has already **drained the GPU pipeline**. A side
stream would not help until this sync is gone — the drain serializes everything around it, which
is exactly the property (a) is trying to break.

Two things to fix, in this order, before attempting (a):

1. **Remove the D2H + sync.** The caller already knows the token positions host-side (they are
   derived from the sequence length during prefill chunking). If that is true on every path, the
   positions tensor does not need to come back from the device at all. If it genuinely is
   device-only, compute the page-run boundaries on device instead of in a host loop.
2. **Then** move `gqa_kvarn_commit_completed` onto a second stream with an event dependency on the
   producer that filled `k_tile`/`v_tile`, and a join only before a subsequent attention could read
   that page as history.

Estimated prize: the sync itself accounts for the ~1-1.5 s of wall-vs-kernel-time slack measured
here (31.2 s wall vs 30.5 s kernel on a clean run), i.e. **3-5%** — real but not the 12.4%. The
12.4% still requires (a) hiding the 3.61 s of Sinkhorn compute, which cannot be done while (1)
forces a drain per append call.

Caveat: `gqa_kvarn_prepare_page_for_append` also calls `gqa_kvarn_hydrate_page`, which READS pages
back into the bf16 tile. So the host loop is doing read-modify-write on page boundaries, and
removing the sync has to preserve that ordering. Not a trivial change — worth a design review
before implementation.

## 4. Code layout — still worth doing, but for the DECODE side only

Keep this as a decode-throughput item, not a prefill item.

**Current state and why.** K is `[channel][key/2]` (8192 B/page/head) and V is `[key][channel/4]`
(4096 B/page/head). The asymmetry is not accidental: the quantizer's K loop is **per channel**
(`for i in 0..kD`, deriving `scale`/`zp` from min/max over the 64 keys, then
`kvarn_pack_row(q, kG, q_packed + i*32)`), so channel-major is the natural write. V's loop is per
key, hence key-major. This is why my Stage-2 V reads were coalesced and my K reads were not.

**What key-major K buys the consumer.** With K as `[key][channel/2]`, key k's 128 code bytes are
contiguous, and lane l needs channels `8l..8l+7` → bytes `4l..4l+3` → **one `uint32` load per
lane**, 32 lanes covering 128 contiguous bytes. Fully coalesced, and it is exactly the 8-channel
granularity `kvarn_mma_pack_bf16x8` wants. Today the same read costs 32 sectors per warp
instruction.

It also lets staging use `cp.async<16>` instead of the 4 B loads forced by the 36 B padded row
agent 1 flagged.

**What it costs.** A nibble-matrix transpose in the quantizer: 256×64 codes, currently written
row-major per channel. Options: stage codes as bytes in smem (`[channel][key]`, 16 KB — fits in the
existing 72 KB budget) and do a second pass writing key-major words; or keep the per-channel loop
and scatter. Either way it is a **write-path change in a QA-verified kernel**, so it needs the
codec tests (`test_kvarn_codec`, `test_kvarn_layout`, `test_kvarn_tile_cuda`) and a
**migration/versioning story for existing artifacts** — the layout is baked into saved `.ninfer`
KV caches and into the export format. That last point may be the real blocker; check with the
owner before anyone writes code.

**Break-even.** Decode-side, agent 1 measured ~20.9 µs per page with ~15 µs of stall above ~5-6 µs
of identified work. 16 B staging cuts the staging op count 4x, but the stall is not proven to be
staging-bound, so the honest expectation is **single-digit percent of decode**, not a multiple.
Do it after §3's (a)/(b)/(c), and only if the phase-skip attribution says staging is where the
stall lives.

## 5. Interaction with docs/117 (int4 / FP4)

If docs/117 contemplates nvfp4 MMA on the codes: KVarN K is **4-bit integer affine with an
additive zero point** (`(code*s_col + zp)*s_row`), not e2m1. FP4 MMA cannot absorb the additive
`zp`, so it needs the same per-page rank-0 correction (`Σ q·zp`) that packed already uses for the
affine. So "FP4 = no dequant" is not true here, and the alphabet mismatch makes it a **quantizer
change requiring accuracy re-validation**, not a layout tweak. Do not schedule it as a docs/117
follow-on without the owner's explicit call.

## 6. Proposed ordering

1. §3(b)(c): drop the D2D copies, de-batch the head loop. No numerics, no scheduling risk.
2. §3(a): side-stream overlap of commit. Gate on prefill t/s reaching bf16 parity at 20k/40k/80k.
3. Phase-skip attribution result → only then decide on §4's layout change for decode.
4. §3(d) and §5 as separate, accuracy-gated proposals to the owner.

Step 2 is the one that can close a 12.4% gap by itself.
