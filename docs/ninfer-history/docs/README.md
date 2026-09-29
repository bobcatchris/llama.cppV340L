# V340L / AMD (HIP) port lane — registry

**Owner:** coordinator (Team Red / AMD), 2026-09-12. This is the AMD line's registry and entry
point. The line lives on branch **`amd/main`** and all of its documents live under **`docs/amd/`**;
it no longer claims global `docs/NN` numbers, because that namespace is shared with the NVIDIA line
(167/168/171/173 were already taken there) and cross-line pushes to `main` were colliding.

**Historical note:** this file was `docs/174_v340l_amd_lane_pointer.md` on `main` before the line
split; older references to "docs/174" in other documents and commit messages mean this file.

| What | Where |
|------|-------|
| Lane home | `docs/amd/v340l/` (`README.md`, `00_scope_and_work_order.md`, `PROGRESS.md` = append-only gate log) |
| Work order | `docs/amd/v340l/00_scope_and_work_order.md` — steps 0–8, phase gates PG-0…PG-F, per `docs/99` template |
| Branch | `wo/v340l-hip` (MANDATORY worktree; agent1 was found scoping in the shared main checkout — see STATE 2026-09-12) |
| AMD CI lane | `tools/ops/run_ci_amd.sh` (zero-GPU stages grant-free; GPU stages require a written coordinator grant) |
| Coordinator doc | `docs/amd/COORDINATOR.md` (this host answers the AGENTS.md rocm-smi probe) |

## Measured board facts (2026-09-12, this host — do not re-derive)

- **This box IS the v340l target.** `rocm-smi --showproductname` → 4× `Vega 10 [Radeon Pro
  V340/Instinct MI25x2]`; `nvidia-smi` → a single GTX 1050 Ti (display only). The repo
  directory name `dual_5060_ti_ninfer` is inherited and is **not** the arbiter.
- 4 HIP devices (kfd nodes 1–4), **gfx900** (`gfx_target_version 90000`) — not gfx906.
- **7.98 GiB VRAM per device** (8,573,157,376 B). 2 physical V340L cards × 2 dies each
  (user-confirmed) = 15.97 GiB/card, 31.94 GiB/box.
- ROCm **6.2.0-66** (`/opt/rocm/.info/version`).
- Artifact `neroued/Qwen3.8-27B-NInfer`: **20,437,336,576 B** (= 19.03 GiB; "20.44 GB" is
  decimal), sha256 `0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3`,
  `weights_id=groupwise-int`, `container_version=2`.

## Ruled 2026-09-12 (coordinator)

1. **"2 GPUs then 4" = HIP devices**, not cards — the cards reading demands M4 = 8 devices /
   4 cards, which this box does not have. M2 = 2 devices (1 card, fixture-based: 19.03 GiB >
   15.97 GiB), M4 = 4 devices (2 cards, first real serving).
2. **No CUDA host exists in this checkout**, so the "CUDA path builds byte-identically"
   guarantee is **source-diff/whitelist-only here**; genuine CUDA regression proof must come
   from the NVIDIA-line host or ride the merge package as an explicit OPEN item. A CI cell
   that prints CUDA-side green on this box is a false gate.
3. **Phase-gate implementation + CI wiring belongs to gemini's exclusive test lane** (§7.x);
   agent1 keeps the PG specs as design input. Neither A1 nor A2 may absorb it.
4. VRAM LAW and the ANTI-RESURRECTION rule bind the HIP tree identically: `hipMemGetInfo` +
   actual bytes, zero estimate constants, and the step-0 parity cell must diff main's tp2
   preflight region / `tp2_budget.h` for every branch landing on main.

## Additional measured facts (coordinator, 2026-09-12)

- **HIP header asymmetry, port-relevant:** CUDA's `cuda_bf16.h` transitively pulls `cuda_fp16.h`;
  ROCm's `hip_bf16.h` does **not**. Measured via `tp2_backend.cpp` reaching `__half` through
  `one_shot_allreduce.h`'s include chain (agent1, `903d6828`). Any HIP file needing `__half` must
  include `cuda_fp16.h` (shimmed) explicitly — it will not arrive via the bf16 shim.
- **bf16 naming on ROCm 6.2 AMD path:** `__nv_bfloat16` / `__nv_bfloat162` are **absent** from
  `hip/amd_detail/amd_hip_bf16.h` (they exist only under `hip/nvidia_detail/`, the CUDA-emulation
  path). The lane now defines them deliberately in `src/common/hip_shim/cuda_bf16.h`. Scalar
  `__float2bfloat16` / `__bfloat162float` DO exist; bf16 **arithmetic** (`__hadd` on bf16x2) does
  not compile on gfx900 — decision D3 (kernel numerics in fp16/fp32) is therefore **compiler-enforced**,
  and gemini's PG-1 carries a compile probe asserting that failure.
- **HIP device → card mapping is NOT identity:** dev0→card1, dev1→card3, dev2→card0, dev3→card4
  (card2 = GTX 1050 Ti, `driver=nvidia`). Per-device sysfs sampling must look up the card.
- **Silent-drop class:** a HIP-only CMake project **drops unclaimed `.cu` sources and still exits 0**
  (measured: 16/21 objects on two "green" builds). Fixed by explicit `LANGUAGE HIP` +
  `src/CheckHipArchive.cmake` POST_BUILD parity guard, which compares **name sets**, not counts
  (proven: equal counts with one swapped name → PARITY FAIL) and prints a sha1 of the source list so
  a stale guard line cannot masquerade as current. **Caveat open:** the guard still accepts a
  count-only `SRC_COUNT` fallback — a weaker check that must not be logged as a name-set result.
- **Toolchain:** `/home/chris/opt/cmake/bin/cmake` and `/home/chris/opt/cmake/bin/ctest`, both 3.30.5,
  upstream tarball, sha256 `f747d9b23e1a252a8beafb4ed2bc2ddf78cff7f04a8e4de19f4ff88e9b51dc9d`
  (recomputed, not read from the `.sha256` file). **pip cmake forbidden on this box.**

## Cross-lane additions convention (registered 2026-09-12, agent1 @ `db1dc50e`)

Product additions to the HIP whitelist are **expected drift**, not regressions. Both lanes obey:

1. Any commit adding a shim header or whitelist entry carries **`+shim:`** or **`+whitelist:`** in the
   SUBJECT, with exact paths in the body.
2. The test lane must **derive, not restate**: `src/common/hip_shim/**` is the product lane's namespace
   and is permitted by construction; the authoritative file list is machine-readable from
   `src/HipSources.cmake`. A hardcoded allow-list array **will** drift (observed: `4ddefff6` went red
   on four legitimate new shim headers — a correct red — and was patched at `fd7eaa1b` by adding names,
   which is the fix that breaks again on the next port).
3. **A red addition-gate following a `+shim:`/`+whitelist:` commit = expected drift.** Ping the product
   lane before anyone widens a pattern until it passes. That line is what keeps a red gate trustworthy
   instead of something people learn to ignore.

## cp.async on gfx900: sync-staging shim (ruled 2026-09-12, bounded)

`src/common/hip_shim/cuda_pipeline.h` implements CUDA's `__pipeline_memcpy_async` as a
**synchronous** global→shared copy, with `__pipeline_commit()` / `__pipeline_wait_prior(n)` as
**no-ops** (gfx900 has no `cp.async`). Correct for any stage-then-use call site — data is resident
before any wait observes it — and the 16 B path is split into two 8 B writes rather than assume 16 B
alignment on a shared pointer.

**Scope limit, recorded because it is not glue-only:** pipeline-pattern hits in the decode bodies —
`gqa_decode_slice4_kvarn.cuh` 12 (KVarN, out of first-token scope), `gqa_decode_slice3_i8.cuh` 4
(**int8-KV = T3b = production parity**), `gqa_decode_unified_slice2.cuh` 1, `gqa_decode_body.cuh` 0.
What the shim gives up is **cross-iteration latency hiding**, which is exactly what those bodies
pipeline for. So: correctness preserved, throughput unproven.

**Ruling:** accepted for T1. At T3, any body keeping the sync path must ship a number with a **named
denominator** against the ~183 GB/s PG-0b anchor; if the inner loop shows a stall from lost overlap,
the fix is a real gfx900 staging path (manual double-buffer: global→register→LDS with the next
iteration's load issued before consume) — **not a wider shim**. Where a body genuinely overlaps
staging with compute, the no-op `wait_prior` must never be treated as equivalent.

## Device geometry (verified 2026-09-12 — and one coordinator number corrected)

- **`Compute Unit: 56` per gfx900 device** on this host — `rocminfo` (each of the 4 devices) agrees
  with HIP `props.multiProcessorCount` as printed in the committed PG-A log. **224 CUs box-wide.**
  The **64** figure is full Vega 10 **XT** (MI25-class SKUs); V340 exposes 56. Any occupancy, wave, or
  split-K sizing must use 56 per device. *(Coordinator asserted 64 in a dispatch on 2026-09-12 from
  memory rather than measurement and was wrong; gemini's 56 was right and verified — the same
  verify-before-asserting standard applies to me.)*
- **`SM count: 90` is NOT a CU count.** `ctx.sm()` returns `props.major*10 + props.minor` = 90 for
  gfx900 — the CUDA-style compute-capability code. On a HIP build this field is meaningless as
  hardware geometry; the log now prints `SM / CC code: 90 (…; physical CUs = 56)`. Anyone porting a
  CUDA occupancy heuristic keyed on `sm` must rewrite it against CUs/waves, not read 90 as anything.
- **Idle sysfs samples prove nothing about which device ran work.** A post-run read showing
  `busy 0%, mclk 167 MHz, sclk 300 MHz` is the idle state AFTER exit; it can corroborate that a card
  is amdgpu and healthy, but not that it carried a given HIP device's load. Device↔card pairing must be
  established by a **concurrent** sampler (`pp_dpm_mclk`/`pp_dpm_sclk`/`gpu_busy_percent` during the
  run, ≤200 ms) — the pattern already built for PG-0b in `results/v340l/pg0b/`.

## Interconnect reality (G-AMD-5, measured 2026-09-12) — **no P2P on this box**

- **`hipDeviceCanAccessPeer` = 0 for all 6 device pairs, both directions, including the two dies of the
  same physical card.** There is no device-to-device path; **every cross-die byte is host-staged.**
- Host-staged copy bandwidth: **cross-card 6.61–6.70 GB/s**, **same-card siblings 4.86 GB/s** (one-way
  payload convention). **Place TP2 cross-card** (e.g. dev0+dev2) — 37% faster, contrary to the usual
  on-package instinct, because both sibling streams contend the card's upstream.
- Small-message RTT **flat with size** (64 B ≈ 4 KiB ≈ 98–101 µs per 2 hops, ≈50 µs/hop): the cost is
  **per round trip, not per byte** → collective **count** is the performance lever at TP2 decode
  (~130 AR/step ≈ 13 ms/step unbatched = not viable; batched 2–4 layers = 3.3–6.5 ms/step).
- **Card grouping confirmed three independent ways:** CPU root ports (`00:01.0` → {card1, card3} =
  {dev0, dev1}; `00:01.1` → {card0, card4} = {dev2, dev3}); the bandwidth split landing exactly on the
  sibling legs; PG-0b's device→card mapping. **Device index ≠ card index; same-card ≠ fastest.**
- **Consequence for the AR trio:** `one_shot_allreduce.cu` / `one_shot_argmax.cu` (kernel writes peer
  memory + device doorbell) **cannot be translated — it is a host-staged redesign** (persistent pinned
  host buffers, async double-buffering, no per-collective alloc/sync). The earlier "~1k loc to port"
  budget is **retracted**; coordinator's own error, recorded.
- Any cross-device figure above the ~183 GB/s intra-device anchor is a broken instrument. Probe values
  landed well under it, which is part of why the result is credible.

## Sanctioned shared-header exceptions (ruled 2026-09-12, option B)

`src/ops/common/memory.cuh` and `src/ops/common/math.cuh` contain hand-written PTX
(`cp.async`, `"l"` constraints, `__cvta_generic_to_shared`; `ex2.approx.f32` with `"=f"`) that cannot
compile on GCN and blocks ~half of T1 plus T2/T3 includes. **Ruled: sanctioned edits behind
`#if defined(__HIP__)`, emulation in the HIP branch, `#else` preserves original text untouched** —
*not* an include-path overlay, because the NVIDIA line actively edits these files (465 commits merged into
main on 09-12) and a shadowing overlay would drift silently. **Exactly these two files; a third requires a
new ruling.** Both are registered exceptions for gemini's PG-1 derive-with-exception-list, never a widened
pattern. Fidelity notes: zfill = `min(src_bytes, Bytes)` + zero tail per CUDA spec;
`ex2.approx → exp2f` is **approximation-class, not bit-identical**, so rmsnorm/silu gates need tolerance
bands. CUDA-path equivalence remains **unprovable here** (no nvcc) and rides the merge package as OPEN.

## Kernel/launch conventions & addressing (measured 2026-09-12)

- **Launcher shape convention is `{d, rows}`** — `ne[0]` is the **feature dimension**, not the row count.
  A harness that assumed `{M,D}` ran `layer_norm` at `d=4` and produced NaN-degenerate "passes." Related:
  `scatter` is a **2-D column-form** API — read the kernel, not the header name.
- **argmax logits are token-major `[t*V + v]`.** Transposing the reference hides a real mismatch.
- **`gfx900:xnack-` on this host ⇒ unified addressing is OFF.** A device kernel dereferencing a **host**
  pointer is a genuine semantic fork from CUDA (works there, faults/misbehaves here), not a flake. The
  production contract is **device-side configs** (`tp2_backend.cpp:4672`); any launcher still passing
  host pointers is a **product port item**, and it will recur across every launcher ported after this one.
- **bf16 host conversion is NOT broken.** `#include <cuda_bf16.h>` (our shim) → `__float2bfloat16(-1.0f)`
  yields raw **`0xbf80`** and round-trips exactly, in a `.cu` TU built with
  `hipcc -O2 --offload-arch=gfx900` (verified twice: hip_bf16.h directly, and through the shim). An
  earlier attribution of `-nan` to ROCm's host converters **did not reproduce**; the shape bug above is
  the better explanation. Pure-C reference helpers remain good practice — for *independence from the
  library under test*, not because the toolchain is broken.
- **Ruling for claims of this class:** a toolchain-defect assertion ships only with a **minimal
  standalone repro** attached; otherwise it is recorded as "observed anomaly, cause unresolved." Two
  nearly-propagated non-reproducible blockers tonight (mine: "no Q3 dtype exists", retracted on the q3
  branch; agent1's: host bf16 converters) — both caught by running the thing.

## Confirmed HIP toolchain divergence: `__bfloat16_as_ushort` is NOT a bit reinterpretation

`amd_hip_bf16.h:537` implements it as `unsigned short ret = h;` — a **numeric conversion** — while its own
docstring says *"Reinterprets bits in a __hip_bfloat16 as an unsigned signed short integer."* **The
function contradicts its own documentation**, and on CUDA the same-named primitive really does
reinterpret. Verified on this host through our own shim (`hipcc -O2 --offload-arch=gfx900`, host call):

| value | `__bfloat16_as_ushort`-style numeric | true bits | |
|---|---|---|---|
| 1.0 | `0x0001` | `0x3f80` | DIVERGE |
| −0.5 | `0x0000` | `0xbf00` | DIVERGE |
| π | `0x0003` | `0x4049` | DIVERGE |

**Rules:**
- **Never** use `__bfloat16_as_ushort` (or any `*_as_ushort`) to obtain bit patterns on HIP. Use
  `__hip_bfloat16_raw` / `.data`, or a pure-C `memcpy` into `unsigned short`.
- **`__float2bfloat16` / `__bfloat162float` ARE correct** — measured exact with proper RNE encodings
  (−1.0 → `0xbf80`, π → `0x4049`), both via `hip_bf16.h` and through `common/hip_shim/cuda_bf16.h`. An
  earlier "host converters unreliable" report was a **misattribution** of this same defect: the NaN came
  from reading uninitialized host memory under a wrong shape convention, not from the converters.
- Applies to **reference/gate code as much as product code**: a cosine bar or golden vector built with
  `*_as_ushort` is silently comparing garbage. gemini (PG-B) and the q3 team both notified with the
  precise wording — the wrong version of this warning would have had them rewriting a working path.
- **Ruling reminder:** a toolchain-defect claim ships only with a minimal standalone repro. This one
  earned its place that way, on the second pass, after the first version was correctly rejected.

## Freshness proof standard for device runs (adopted from agent1, G-AMD-7 self-void)

`mtime` does not prove an artifact matches committed source — incremental builds and relink skips make
"compile-only fix" runs silently execute stale binaries (measured: a launch reproduced pre-fix numbers
exactly, and the tell was visible only afterward). **Required for any stamped device run: print into the
run log a marker string unique to the harness version **plus** the binary's hash, so
"this ran the committed source" is evidence inside the artifact.** A run whose log lacks that line is
void by definition, regardless of what it reports.

## Launcher conventions (registering agent1's two, both learned the expensive way)

- **Tensor shape convention is `{d, rows}` — `ne[0]` is the FEATURE dimension**, not the row count.
  Measured across `l2norm` / `layer_norm` / `scatter`. A harness using `{M,D}` ran `layer_norm` at `d=4`
  and produced NaN-degenerate "passes" that were not results.
- **`argmax`'s `valid_rows` is the VOCAB SCAN LIMIT, not the token count.** Passing `T=2` scanned 2 of
  1024 logits. The name invites exactly this error; anyone writing a reference against it should read
  `ops/kernel/argmax.cuh` first.
- **Artifact provenance is proven by content, not mtime** (agent1's wording). Incremental builds and
  skipped relinks make a stale binary run "clean" — this happened twice in one hour, once reproducing
  pre-fix numbers exactly. Required in every stamped device run's log: a **version marker string unique
  to the source version + the binary/archive hashes**. **A run whose log lacks that line is void by
  definition**, whatever it reports.

## Wavefront-64 reduce path — **OPEN PRODUCT DEFECT (measured; mechanism unnamed)**

Superseding two wrong static reads and one premature closure, all preserved so nobody re-derives them:

1. **Coordinator read:** `kWarpSize = 32` + `kFullWarpMask = 0xffffffffu` on 64-lane wavefronts ⇒ both
   halves of a wavefront recompute the same row ⇒ ~half throughput. **Falsified by anchor** —
   `l2norm.cuh:22-23`: `warp = threadIdx.x / kWarpSize`, `row = blockIdx.x*kWarpsPerBlock + warp`, so
   threads 32–63 are `warp=1` on a **different row**.
2. **agent1 rebuttal:** group-scoped width-32 reduces are correctly scoped per row ⇒ predicts **neither**
   duplication **nor** halving. **Also contradicted by measurement.**
3. **Coordinator "closed: no tax" entry — WITHDRAWN.** Right to withdraw the falsified claim; wrong to
   declare the question closed while the probe was still unrun.

**Measured (G-AMD-10, `results/v340l/batch2c_run.log`, one launch, freshness by construction):**
l2norm probe gave **`inv = 0.2080`** vs `1/sqrt(half) = 0.2033` vs `1/sqrt(full) = 0.1438`.
**Inverting the three values matters more than the labels:** implied kernel Σ = **23.114**, half-row Σ =
**24.195**, full-row Σ = **48.360** ⇒ kernel is **47.8% of full** but **95.5% of half — i.e. 4.5% BELOW
half**. Dropping 32 of 64 lanes removes a known set of pairs and lands on **exactly 50%**; it cannot shave
4.5% off. **So the signature is wrong or partially-wrong INPUT data** (a subset that is neither the whole
row nor its front half — strided/pair-offset reads, crossed rows, or elements outside the initialized
region), **not** a reduce losing lanes.

**gemini's shim-diagnosis rebutted by enumeration, not argument:** `__shfl_down_sync` returns own-value
whenever `(lane % width) + delta >= width`; an exhaustive scan over lanes 0–63 × butterfly deltas
{1,2,4,8,16} finds **zero cross-group leaks**. The specific claimed case ("thread 16 reads thread 32") is
exactly the guarded one: (16%32)+16 = 32 ≥ 32 → own value, no shuffle. `__shfl_sync` broadcast is likewise
correctly group-scoped. **Do not "fix" the shim on that diagnosis** — the emulation is shared by every
ported kernel, so a change made for a refuted reason trades a phantom bug for a real regression.

**Candidate-Σ table, computed on CPU by the coordinator (reproducing the harness's own RNE `to_bf` and
generator `v = 0.01*((i*37)%211) - 1.0`, row 0 = elements 0–127, matching the kernel's contiguous
`row_base = row*pairs`):**

| candidate | Σx² |
|---|---|
| full row (128 elems) | **48.3670** |
| k=0 only (elems 0–63) | 24.6267 |
| k=1 only (elems 64–127) | 23.7403 |
| lanes 0–15 contribute, both k | 11.2117 |
| lanes 16–31 contribute, both k | 11.9509 |
| **probe-implied Σ (from `inv = 0.208008`)** | **23.1121** |

**23.1121 matches none of them** — 2.6% off k=1-only, 6.2% off k=0-only, and near half of *neither*
half-lane value. **The "missing lanes / missing k / one butterfly step short / exclude pair 31" family is
therefore dead**, including the subset match gemini already retracted. Recorded so nobody resurrects it.

**Why the probe's quantity is the wrong one:** it infers Σ from **`out[0]/in[0]`** — a bf16-quantized
output ratio, i.e. the reduce's result read **through the store path** after `__floats2bfloat162_rn`, with
`in[0]` re-read from the buffer. bf16 quantization alone is ~0.5%, so it cannot explain a 5% gap, but we
are reasoning about a ~5% discrepancy using a quantity whose total error budget was never bounded.

**Reduce EXONERATED by CPU emulation (coordinator, after fixing an arithmetic slip in my own first
attempt — I overwrote instead of accumulating):** emulating the exact loop and both readings of the shim,
with `d=128`, `pairs=64`, `KMAX=4`:

```
per-lane partials: lane0 = 1.7034   lane15 = 0.3612   lane31 = 2.3384   Σ lanes = 48.3670
group-local width-32 butterfly (16,8,4,2,1): lane0 = 48.3670                    <- correct
physical 64-lane wavefront, 32-based guard:  lane0 = 48.3670, lane32 = 47.0767  <- row 1's own sum, correct
```

**Both interpretations of the shim produce the full row sum**, and under the physical-64 reading each
logical warp still reduces its own 32 lanes onto its own row. So the **shuffle-leak story and the
lane-loss story are both dead by arithmetic** — third static claim tonight to die to a computation rather
than a debate. Residue shape: probe-implied Σ / correct = **1/2.093**, and `0.208008/0.143789 = 1.4466 =
sqrt(2.093)` — internally consistent, i.e. a **single scale factor on the accumulated Σ**, pointing at
what the *compiled* kernel accumulated or how the harness extracted `inv`, not the reduction.

**Probe spec that follows (sent as #70):** answer exactly two questions — (1) the kernel's runtime `d`,
`pairs`, `kMaxPairsPerLane` and lane 0/lane 31 pre-reduce partials; (2) the raw `out[0]` bits, the
kernel's own `inv`, cross-checked against the harness's `out[0]/in[0]` extraction. If the kernel's own
`inv ≈ 0.1438` while the harness derives 0.2080, **the defect is in the probe** and l2norm closes as
harness. **Zero product mutation**: instrument in the harness/test-local TU launching the real
`l2norm_launch` — no env-gated print inside `l2norm.cuh`, since the exception pair stays two files.

**Correct next diagnostic (one launch, no inference chain):** with an env- or compile-gated debug print
inside `l2norm_warp_bf16x2_kernel`, have lane 0 of a few rows print from the device: `d`, `pairs`, `row`,
`kMaxPairsPerLane`, its **pre-reduce partial** and its **post-reduce** Σ. That separates accumulate-short
from reduce-short, exposes any kernel-vs-harness disagreement about `d`/layout instantly, and measures the
same quantity twice — once inside the kernel, once outside it.

**CPU-only discriminator (run it before spending a launch):** compute Σx² over candidate element subsets
of the harness's own input and find which equals **23.114** — front half, back half, even/odd-indexed
pairs, lanes 0–15 of each logical half, row0-front + row1-front, row-1 spillover, each also under the
`{d,rows}` vs `{M,D}` transpose. A match names the mechanism and whether it sits in the harness indexing or
in the kernel's `pair = lane + k*kWarpSize`. "No subset matches" is also a result — then go device-side
with a per-lane partial dump.

**Prime suspect — suspect only, not conclusion:** **our own shim's explicit-`Width` paths**
(`src/common/hip_shim/cuda_runtime.h`; the 3-arg overload defaults `width = 32` at :151). **Both static
reads assumed that layer was correct and neither analyzed it** — they argued about product code and
skipped the compatibility layer. A width-limited butterfly landing on a full-wavefront shuffle, or a
5-step (offsets 16…1) butterfly covering only 32 of 64 lanes, produces exactly this magnitude error on a
**wavefront-64** device.

**Binding while open — not after diagnosis:**
- **gemini: no PG-B tolerance bands for any kernel with a cross-lane reduction.** A band fitted around a
  half-summing reduce certifies the bug.
- **T3 attention is squarely in the blast radius** (densest shuffle/reduce use) — this closes **before**
  retile work starts.
- **Concrete review, CPU-only:** enumerate `__shfl_*_sync` call sites and check each one's `Width` against
  64. This is the same "correctness, not throughput" hazard raised while losing the geometry argument —
  and it turned out to be the live one.
- **Still no touching `warp.cuh`'s `kWarpSize`** — CUDA-shaped by design, shared with the NVIDIA line; the
  product-header exception list stays exactly the two named files.

**Rule with three instances behind it:** *a static read is an argument, not a result — quote the anchor,
and don't declare a question closed until the probe reports.* Confident static analysis failed twice in
one lane tonight, in the same direction; measurement failed zero times.
