# 03 — External gfx906 fork: ISA-level audit

**Auditor:** agent2 · **Requested by:** C441 · **Date:** 2026-09-12
**Subject:** `/tmp/ninfer-gfx906` @ `7a3c18d9` (working tree clean, 68 MB)
**Scope as assigned:** ISA-level only, three items. CPU-only; **zero device time taken, no grant
used, cards untouched.** Not WO-04 (agent3 owns that); I have not touched either q3/q2 header.
**Comparison baseline:** our `amd/main` @ `b5b90bcc` (WO-02 fix merged).

Every claim below carries a reproducible command. Where I first got a result wrong, the wrong
result is kept in the receipt so nobody re-trusts my read of it.

---

## (a) Their width-32-on-wave64 subgroup mechanism — **CLEAN. Does not carry our hazard.**

**Verdict: no `ds_bpermute` publish hazard, on either arch. Their design is structurally immune,
and it is simpler than the fix we just landed.**

Their chain (`src/ops/common/warp.cuh` + `src/core/hip_compat.h:143-159`):

```cpp
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1                       // :16
__device__ __forceinline__ T __shfl_down_sync(unsigned, T var, unsigned delta, int width = 32) {
    return __shfl_down(var, delta, width);                     // :157-159 — pure pass-through
}
```

Three properties, all decisive and all verified:

1. **No early return.** The body is a single unconditional `return`. Our defect required a lane to
   *skip* the wavefull instruction; in their code **no lane can ever skip it**. There is no guard
   predicate to be right or wrong about.
2. **They forward the caller's `width`** — the half of our dispatch that *was* correct.
3. **They set `HIP_DISABLE_WARP_SYNC_BUILTINS`** (`:16`), so ROCm's 64-bit-mask builtins are off and
   their shim is unambiguously what runs. This is the preprocessor question my first audit pass left
   open, and they resolved it explicitly.

**Receipt — tripwire on an identical 3-primitive probe (`warp_reduce_sum`, `warp_max`, broadcast),
both arches, plus a negative control proving the tripwire can actually fail:**

| tree | arch | ds_bpermute | under narrowed EXEC | verdict |
|---|---|---|---|---|
| THEM | gfx900 | 11 | **0** | CLEAN |
| THEM | gfx906 | 11 | **0** | CLEAN |
| OURS (post-fix) | gfx900 | 11 | **0** | CLEAN |
| OURS (post-fix) | gfx906 | 11 | **0** | CLEAN |
| **control: pre-fix shape reconstructed** | gfx900 | 11 | **10** | **HAZARD DETECTED** |

The control matters: without it, four greens in a row is indistinguishable from a broken detector.
I built the control by reinstating our exact pre-fix guard (`if (out-of-group) return val;`) in a
header-only replica — it fires 10/11, so the tripwire discriminates the hazard and is not
vacuously passing.

Reproduce:
```sh
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -O2 -x hip --offload-arch=gfx906 -DNINFER_GFX906_COMPAT \
  -I /tmp/ninfer-gfx906/src -I /tmp/ninfer-gfx906/src/compat/gfx906/include \
  -S probe.cpp -o out.s        # then count bpermutes inside s_and_saveexec..s_or_b64 exec regions
```

### (a.1) LOAD-BEARING: the masks-ignored verdict (gates agent3's TA2 adoption)

Both lines drop the `mask` argument — ours at `cuda_runtime.h` (`(void)mask`), theirs at
`hip_compat.h:142` ("Masks are dropped (all call sites pass full masks)"). For TA2 that claim has to
be a verdict, not a parenthesis, so I checked it exhaustively rather than restating either comment.

**VERDICT: dropping `mask` is safe on our tree today, and the safety is structural, not incidental —
so TA2 may adopt the mechanism. But it is safe only because of an invariant no gate enforces.**

Method: transitive `#include` closure over the TUs named in `HipSources.cmake` to get the real
in-build set, then every shuffle site and every caller of `warp_sum`/`warp_reduce_sum`/`warp_max`/
`block_reduce_sum` in it, then resolved each named mask constant from source.

| finding | count |
|---|---|
| in-build user shuffle sites | 10 |
| whose mask is a full-warp constant (`kMask`/`mask`/`kFullWarpMask` = `0xffffffffu`, all resolved from source) | **10 / 10** |
| whose mask is narrower than its `width` | **0** |
| in-build callers of the reduce helpers | 12 |
| passing a sub-group `Width` (< 32), which is where a mask-vs-group mismatch could bite | **0** |

Every direct mask resolves to `0xffffffffu` at source level: `sampling_device.cuh:36,46`,
`argmax.cuh:30,31` (`kMask`), `layer_norm.cuh:36,37,38` (`mask`), and the 12 reduce-helper callers all
use the `kFullWarpMask` default. So for all 10 sites `mask ⊇ participating lanes` and `mask` is
exactly the full group — ignoring it cannot change a result, and CUDA's mask is only advisory for
convergence anyway (it is UB to call a `*_sync` shuffle with a lane in `mask` that is not converged).

**The real hazard is convergence, not the mask value — and that is precisely what the fix removed.**
A CUDA `*_sync` builtin requires all masked lanes to converge at the call; a GCN `ds_bpermute`
additionally requires them to *execute* it. That is why our pre-fix early return was both UB-adjacent
and wrong on hardware, and why their unconditional pass-through is the cleaner construction: it cannot
diverge because there is no branch to diverge around. **TA2 should adopt the pass-through shape, not
merely the "masks can be ignored" conclusion** — the latter is safe here, the former is what keeps it
safe when someone adds a guarded shuffle.

**The invariant to know about:** `warp_max<4>` / `warp_sum<4>` sub-group reduces exist in our tree
(28 sites across `gqa_decode_*`/`gqa_attention_*kvarn*`) and pass `FullMask` with `Width=4`, i.e.
`mask ⊋ group`. That is legal CUDA and is handled by both mechanisms, but **none of those files is in
the HIP build** — so zero in-build exposure. The moment a `warp_*<N>` with `N<32` enters the
whitelist, mask-dropping is still fine but the group-scoping math is what starts mattering, and it
should be re-audited then.

Two corrections to claims in circulation, mine included:
- **Their `hip_compat.h:142` comment is stronger than ours and is the one that should be cited:** it
  states the *reason* masks are droppable (all call sites pass full masks). Our header says nothing,
  so a reader of our shim cannot tell whether `(void)mask` was decided or forgotten. Worth importing
  that sentence into `cuda_runtime.h` when WO-TB1 next touches the file.
- My own §summary line said masks were "unverified". That was true when written and is now closed.

**Negative controls on my method, because a 10/10 clean sweep deserves the same suspicion I applied
to the tripwire:** my first pass at this reported "0 sites" (grep pattern assumed clang-format's
`file(line:col)` output, which is not `grep -rn` format) and then reported 2 false-positive "narrow
mask" hits by regex-matching `block_reduce_sum(x, warp_sums)` against the masked-helper signature —
`warp.cuh:41` takes `float* sums` as its second parameter, not a mask, and `block_reduce_sum` accepts
no mask at all. Both were caught by reading the signatures, not by trusting the script. The 10/0/12/0
counts above are from the corrected pass.

### (a.2) Their extra mechanism (`gfx906_reduce_sum32`, `warp.cuh:47-70`)** is a DPP/ds_swizzle ladder
(`update_dpp` 0xB1/0x4E/0x124/0x128 + `ds_swizzle 0x401F`) that avoids `ds_bpermute` entirely.
This is genuinely better than a shuffle ladder on this hardware: DPP and swizzle are *register*
operations, not wavefull LDS exchanges, so the publish hazard is architecturally unreachable —
not merely avoided by correct control flow. Their comment at `:44-45` ("the swizzle never crosses a
32-lane group, which is the point") shows they understood the group-scoping requirement from the
opposite direction to us. My probe did not instantiate that helper (it is gated behind
`NINFER_GFX906_COMPAT` *and* used only by their pass-2 GEMV), so I am reporting it as sound by
construction and read, **not** as ISA-verified — 0 dpp/swizzle instructions appeared because no
kernel in my probe called it. Flagging that gap rather than papering it.

**Their `warp.cuh` is otherwise line-for-line ours**, including `kWarpSize = 32` and the
`block_reduce_sum` shape. So this is a fork of our tree, and the divergence is confined to the
compat layer.

---

## (b) bf16-via-fp16 staging vs our D3 rejection — **the two lines AGREE on hardware and DISAGREE on enforcement, and our stated premise is false.**

**Verdict: no strategy conflict. But the D3 "compiler-enforced" claim recorded in our own shim
header is wrong on our own tree, and this is the finding that matters more than the fork.**

Convergence first — they reach our conclusion independently:
- them: `docs/gfx906/PORT-AUDIT.md:36` — "gfx906 has no bf16 VALU, so rewritten kernels should stage
  in fp16 (bf16x2 intrinsics compile via ...)".
- us: `docs/amd/v340l/00_scope_and_work_order.md:33` — "no bf16 ALU", decision D3, fp16/fp32 numerics.

Same hardware fact, same conclusion, different vendors' GPUs (MI50/MI60 vs Vega 10/Instinct MI25x2).
That is real corroboration for D3, from an independent implementation that had to discover it alone.

**Now the correction.** Our `src/common/hip_shim/cuda_bf16.h` header comment asserts:

> "bf16 ARITHMETIC (e.g. `__hadd` on `__hip_bfloat162`) **FAILS TO COMPILE** on gfx900 — no bf16 ALU.
> This is compiler-ENFORCED ... If a whitelisted file tries bf16 arithmetic, the build fails loudly
> (that failure is the D3 contract working)"

**It does not fail. It compiles, and silently lowers to software emulation.** Measured:

```
$ clang++ -O2 -x hip --offload-arch=gfx900 ... -fsyntax-only bf16arith.cpp   # __hadd2 on __nv_bfloat162
exit 0     # compiles clean on OURS, on THEM/gfx906, and on THEM/gfx900
```

Emitted for `__hadd2` on **gfx900**, from our own tree:

```
v_add_f32_e32   v3, v1, v1       # unpack to fp32, add
v_add3_u32      v1, v3, v1, s4   # round-to-nearest-even bias trick
v_cndmask_b32_e32 v1, v4, v3, vcc# tie-handling select
v_add_f32_e32   v2, v2, v2       # ... second lane
v_add3_u32      v3, v2, v3, s4
v_cndmask_b32_e32 v3, v4, v2, vcc
```

39 instructions in a two-lambda kernel. The hardware has no bf16 VALU, so the compiler *emulates*
it correctly-but-expensively rather than refusing. **All three configurations behave the same** —
ours/gfx900, theirs/gfx906, theirs/gfx900 — so this is a property of the ROCm toolchain, not a defect
in either shim. That is what makes it a shared finding rather than a self-own: neither line's bf16
contract has compiler teeth, and both lines' docs describe it as if it did.

**Why this is a live risk and not a documentation nit:** the contract's entire safety story is "a
violation cannot compile." A kernel author who writes bf16 arithmetic on this line gets a
**passing build and a silently wrong cost model**, not a loud failure. Two concrete consequences:

1. The D3 gate cannot catch a D3 violation. The only thing standing between us and a tree full of
   emulated bf16 math is reviewer attention, which is precisely the resource tonight's registry
   shows is unreliable at this task (four wrong confident static reads).
2. **This likely contributed to the very defect I just fixed.** `batch2_verify.cu`'s header records
   that `__bfloat16_as_ushort` is a *numeric cast* on HIP (`amd_hip_bf16.h:537`) where CUDA's is a
   *bit reinterpretation* — `-1.0f` gives `ffff` vs `bf80`. That is the same class of surprise:
   bf16 intrinsics silently doing something non-CUDA on this target, in a header we `typedef` into.
   Our shim's option-(b) storage-only typedefs inherit that surface.

**Recommendation (mine to make as an audit finding, not to implement — the shim is my file but
D3 is a decision):** if D3's teeth matter, they need a *real* gate. Cheapest that works: a CPU CI
cell grepping device ISA for the emulation fingerprint (`v_add3_u32` + `v_cndmask` RNE pattern, or
`__ocml_*` calls from bf16 ops) in whitelisted kernels, or `-Wmacro-redefined`-style lint on use of
bf16 arithmetic intrinsics in `ops/kernel/**`. Compile-pass is not evidence of D3 compliance; that
is the sentence worth putting in the header in place of the current claim.

---

## (c) flag-sync TP2 transport — **they root-caused a card wedge. We hold the identical shape with two of its three safeguards missing. This is actionable on our tree.**

**The history is the finding, and the coordinator's framing is right.** Slice sequence
`de210ae4` (9b, made flag-sync the tp2 **default**) → `40fce80c` (9b, graphs-on gates) →
`7a3c18d9` (9c, **reverted to opt-in**). The revert is one line in
`src/ops/common/allreduce.cu:249-251`:

```cpp
-    return text == nullptr || !(text[0] == '0' && text[1] == '\0');   // 9b: default ON
+    return text != nullptr && text[0] == '1' && text[1] == '\0';      // 9c: default OFF
```

**What actually happened, in their own words** (`docs/gfx906/TP2-SLICES.md`, S9c): 11 requests
succeeded at ~30 t/s, then request 12 (`code`, `reuse=full_reset`) hung — `HSA "HW Exception by GPU
node-2 ... GPU Hang"`, kernel `ring page0 timeout`, **`MODE1 reset FAILED (-22)`**, SMU messages
failing, sysfs unreadable, forced SysRq reboot. Their reading, which I independently assess as
**correct and the sharpest sentence in the fork**:

> "a spin-wait inside a per-device graph that never sees its flag is exactly the shape that wedges
> an MI50 (the ring cannot preempt a spinning wave; the reset fails)."

And the governance sentence, which our line should adopt verbatim:

> "The S9b CLI gates (parity, 128-step graph replay, MTP md5s) **never exercised the serve path's**
> sampling + `full_reset` re-arm sequence."

They also left three named suspects for the root cause (graph re-instantiation on `full_reset` with
a stale flag epoch; 16 MB staging overrun on a wider verify batch; MTP verify-width switch under
sampling) and a standing rule: *do not run tp2 graphs-on under serve without a wedge-watcher-armed
box and production stopped.*

**Their spin has three safeguards** (`allreduce.cu:171-179`):
1. `__builtin_amdgcn_s_sleep(1)` — yields the wavefront
2. `kFlagTimeoutPolls = 1ull << 26` bound (`:158`) — **cannot hang forever**
3. On timeout: `atomicOr(status, 1u)` then `return false` — **fails soft, the kernel exits**

**Our tree holds the same shape with #2 and #3 absent.** `src/core/multi_gpu/one_shot_allreduce.cu`
(our AR transport, a Step-5 port target), lines 80 and 95:

```cpp
while (*peer_flag < expected_epoch) {
    #if __CUDA_ARCH__ >= 700
    __nanosleep(10);
    #endif
}                                    // no bound, no failure path
...
while (gen_obs < step_gen) {
    #if __CUDA_ARCH__ >= 700
    __nanosleep(10);
    #endif
    gen_obs = *peer_gen;
}                                    // no bound, no failure path
```

**Two verified facts make this worse than it looks, not better:**

- **`__CUDA_ARCH__` is not defined under hipcc** (measured: `#if defined(__CUDA_ARCH__)` takes the
  `#else` branch on `--offload-arch=gfx900`). So the moment this file joins the HIP whitelist, the
  `__nanosleep` **compiles out entirely** — a *tighter* spin than the one that wedged their MI50,
  with no yield at all. The arch gate silently disables our only mitigation during porting. That is
  exactly the "silently does something different on HIP" trap this audit found twice already.
- **No timeout means no wedge recovery.** Their box survived because the loop gave up and returned
  `false`. Our loop has no exit condition other than the peer flag arriving; if it never does, the
  wave spins until the ring times out — and per their field evidence, **on this GPU family the mode
  reset can fail outright and the box needs a hard reboot.**

**Update during this audit (post-first-draft): the q3/q2 blocker in (c)'s sibling class is FIXED.**
Agent3's step-0 (`f5035616`, merged at `039ab28b`) patches both headers to
`#if !defined(__CUDACC__) && !defined(__HIPCC__)` at `q3_rowsplit_storage.h:24` and
`q2_rowsplit_storage.h:26` — one line lower than my original cites. Re-verified here: `embed_gather.cu`
now compiles with **0 errors and 0 `macro redefined` warnings**, so my §(c)-adjacent blocker claim is
historical. I also swept the rest of the tree: `#define __host__/__device__/__forceinline__` exists in
exactly those two files and nowhere else, so the class is closed. Agent3's scoping was correct to
patch only the first guard in each file — the *second* `#if !defined(__CUDACC__)` (`q3:64`, `q2:58`)
wraps host-only helper functions (fp16<->float emulation, a CPU reference GEMV) and redefines no
attributes, so it is benign. My §(a)/(b) receipts still quote the pre-fix `:27-29` lines because that
is what the measurement was taken against; the fix moves them, it does not invalidate them.

**Current exposure of the spin-wait finding is genuinely zero, and I want that on the record so this is not over-read:**
`one_shot_allreduce.cu` is **deliberately not in the HIP whitelist** (`src/HipSources.cmake:40-46`,
deferred to Step-5 with the PTX `ld.global.cv/st.global.wt` "l"-constraint blocker measured and
documented). So this is not a live bug; it is a **porting landmine with a fuse already lit**. When
Step-5 lands that file — which WO-03/AR-design territory, agent1's, is actively designing toward —
it inherits an unbounded, yieldless device spin on hardware that cannot recover from it.

**Concrete recommendation for whoever ports AR (gemini's gate, agent1's design, coordinator's
ruling):**
1. Do not whiten-list `one_shot_allreduce.cu` until both spin loops have a bounded poll count and a
   soft-failure path — i.e. copy their `kFlagTimeoutPolls` + `status |= 1; return false` shape.
2. Make the sleep unconditional on the HIP path (`__builtin_amdgcn_s_sleep`), never
   `__CUDA_ARCH__`-gated, and add a CPU CI cell asserting no unbounded `while (*peer_...)` in
   whitelisted device sources.
3. Treat "CLI/one-shot gate passed" as insufficient for graphs-on TP2 under serve. Their 9b passed
   every gate they had and wedged a production box on request 12. Ours should carry a
   serve-path-with-sampling cell before any flag-sync/graph-replay default is flipped ON, and
   flipping any spin-wait transport to default-on should require the wedge-watcher precondition
   they wrote down.

---

## The question I was asked to answer: is there a shorter path to functional q3 TP2?

**Answer: their fork does NOT collapse WO-05's pole, but it is closer than us on one axis we
havent measured, and that gap is the actual news.**

- **Attention kernels: PARTIALLY landed, and more than I first credited.** Four family files exist:
  `gqa_attention_{prefill,decode}_{bf16,i8}_gfx906.cuh` — i.e. both directions in both bf16 **and
  int8** (the i8 files carry 14 `i8` references). So attention is further along than the single-kernel
  goal in their planning doc implies. Still not "done": `docs/gfx906/PORT-AUDIT.md:94-95` scopes their
  own first end-to-end target to "**One** wave64 GQA decode kernel (bf16 KV, T=1) + **a simple**
  prefill attention", `:108` lists "the wave64 / no-bf16-VALU long tail" as an open risk, and `:34-36`
  has their int-quant *GEMM* rewrite still ahead ("dequant -> packed-FP16 FMA"). Files present is not
  the same as attention verified end-to-end, and I found no receipt in their tree showing served-token
  correctness for the i8 variants.
- **Q3 specifically: they have none — verified three ways, not just by absence.** (1) No `Q3`/`IQ3`
  entries in their dtype headers; (2) their only Q3 mention is `docs/maintainer/tensor-formats.md:733`,
  which lists "other integer widths, including Q2 and Q3" among formats **not** implemented; (3)
  `PORT-AUDIT.md:20` enumerates their supported set as Q4G64/Q5G64/W8G32/Q6/BF16/FP32/I32 with "zero
  FP8/NVFP4". Q3 is a format our Team Green promote carries and theirs deliberately excludes. **Their
  fork therefore cannot answer the q3-TP2 question at all** — the formats do not overlap on q3.
- **The gap that IS real: they run TP2 on AMD silicon today; we do not.** They have a working
  two-rank flag-sync transport with peer staging, epoch flags, and an allgather/allreduce family
  (`allreduce.cu:249+`), and their tp2 has produced *served* tokens (11 requests, ~30 t/s) before
  the wedge. **Our tp2 has never run on a card at all** — we are at T1 single-kernel verification,
  6 of 27 in-build shuffle sites with device evidence, and no AR file in the whitelist. So on the
  "functional TP2 on AMD" axis they are ahead of us by a full stage, having solved the peer-transport
  problem we are still designing.

**So the shorter path is not their code, it is their failure record.** WO-05 should not re-derive
flag-sync TP2 from scratch on gfx900, and it should not re-run their discovery sequence. Three things
they have already paid for, in dollars of rebooted hardware:
the `full_reset`-re-arm-under-sampling hazard class, the wedge-watcher precondition, and the
bounded-spin-with-soft-fail transport shape. Cheapest real acceleration available to us right now is
to *design WO-05's AR transport against their documented failure mode* rather than rediscover it.
I have not verified their transport is correct under their own opt-in — only that it is bounded and
fails soft, which is the property our tree lacks.

---

## Summary of verdicts

| item | verdict | confidence |
|---|---|---|
| (a) their shuffle width mechanism | **CLEAN on gfx900 + gfx906**; structurally immune (no early return, forwards `width`, builtins disabled); their DPP ladder avoids `ds_bpermute` entirely | **high** — tripwire with validated negative control firing 10/11 |
| (b) bf16 staging vs D3 | **No strategy conflict** — independent convergence on "no bf16 VALU". But **our D3 "compiler-enforced" premise is false**: bf16 arithmetic compiles and silently emulates (39-instr lowering measured) | **high** — ISA emitted, both trees, all three configs |
| (c) flag-sync TP2 | Their root cause is **correct and directly applicable**; we hold the same spin shape **missing 2 of 3 safeguards**, currently unreachable (Step-5, not whitelisted) = landmine, not live bug | **high** — file:line for both trees; `__CUDA_ARCH__` non-definition measured |
| q3 TP2 shorter path | Their fork does **not** collapse WO-05 (no Q3 at all, attention partial and differently scoped), but they are **one stage ahead on AMD TP2 itself** | **medium** — inference from their docs + our known T1-only state, no device comparison run |

**Corrections to my own first-pass reads, kept visible:** (i) my first "OURS rejects bf16 arithmetic"
test was invalid — I omitted `cuda_runtime.h` so `threadIdx` was undeclared and the compile failed
for an unrelated reason; had I reported that, I would have written the *opposite* finding. (ii) My
first THEM probe failed on include-path (`src/compat/gfx906/include`, not `hip_shim`) — a
build-failure line in the table, not a result. Both re-run before anything was concluded.

---

## Reconciliation with agent3's §2b (required before finalizing reuse verdicts)

Read `docs/amd/FORK_VELOCITY_PLAN.md` §2b at `039ab28b` as instructed. **No contradiction — the two
audits corroborate each other, on independent passes, with one scope difference worth stating.**

| question | agent3 §2b | this audit | status |
|---|---|---|---|
| is their `warp.cuh` adoptable for the wave64 reduce family? | "logical 32-lane subgroups in wave64, masks accepted-and-ignored... **Direct reference for the G-AMD-10 wave64 reduce defect family**" | CLEAN on both arches, ISA-verified with a validated negative control; masks-dropping safe because 10/10 in-build masks are full-warp | **AGREE**, and this audit adds the ISA evidence §2b's file-level pass does not claim |
| do their gfx906 kernels run on our GPU? | `gfx906_fdot2` and `gqa_gfx906_sdot4` both have `#else` fallbacks → "compiles and runs correctly on gfx900 with zero kernel edits" | not tested by me — outside my assigned items (a)(b)(c) | **NOT IN CONFLICT; UNVERIFIED BY ME.** Note "runs correctly" is a device claim and §2b's stated method is file-level; if that phrasing is load-bearing for a reuse decision it deserves a compile-at-minimum receipt on gfx900 before adoption |
| Q3/attention overlap | not addressed | they have **no Q3** (three ways), attention partial (4 files, below their own "done") | additive |
| TP2 transport volume | `allreduce.cu` 589 LOC + h + split_launch = 917 LOC adoptable | same file, and the **wedge history is the finding**; our `one_shot_allreduce.cu:80,95` lacks bound + soft-fail | **AGREE on volume, and this audit supplies the caution §2b's census omits**: 917 LOC of adoptable transport is only adoptable *with* their bounded-spin/soft-fail properties. Adopting the spin loops without the `kFlagTimeoutPolls` + `return false` shape imports a box-wedge onto hardware where MODE1 reset fails |
| artifact mismatch blocking golden transfer | fork `qwen3_8_27b.ninfer` 18,210,531,328 B / `eec3956499…` vs ours 20,437,336,576 B / `0634abb070…` | not checked; not in my scope | **AGREE it matters** and it is consistent with my own §(c) conclusion — their t/s and parity numbers are *their hardware + their artifact*, so neither transfers. Reinforces: take the code and the failure record, not the numbers |
| WO-V1 shim gap closure (the 6 TP2 surfaces = my queue item 2) | sourced from fork `hip_compat.h`, "proof cell: compile-only, plus the `.type` field check on 6.2.0" | I have now answered the `.type` question | see below |

### The `.type` question — answered, and it is safe on our version

`TP2_AMD_SUBSET_PLAN` / §2b flag `hipPointerAttribute_t.type` as "fork proved 6.4.1 only, unverified
across versions". **Verified on the box we actually run, ROCm 6.2.0** (`/opt/rocm` symlinks to
`/opt/rocm-6.2.0`; no other version present):

```c
// /opt/rocm-6.2.0/include/hip/hip_runtime_api.h:262-270
typedef struct hipPointerAttribute_t {
    enum hipMemoryType type;      // <-- :263, PRESENT on 6.2.0
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    unsigned allocationFlags;
} hipPointerAttribute_t;
```

`hipMemoryType` is also declared at `:245` with `hipMemoryTypeUnregistered = 0`, `Host = 1`,
`Device = 2`, `Managed = 3`, `Array = 10`. So the field exists and the enum is present on 6.2.0 —
`.type` is usable. **One caveat I am flagging rather than asserting:** the fork's code was written
against 6.4.1, and I have not enumerated 6.4.1's `hipMemoryType` values, so if their transport
compares `.type` against a *literal integer* rather than the enum name, the numeric values could
differ between versions. Recommend the WO-TB1 shim/transport land enum names only, never a bare
integer for `.type`. That is a one-line review rule, not a blocker.

### Net effect on my own reuse verdicts

Nothing in §2b overturns my (a)/(b)/(c) verdicts. Two of my conclusions get **sharpened** by it:
their census treats `warp.cuh` (94 LOC) and the 917 LOC of transport as reusable volume, and my ISA
pass confirms the first is genuinely sound *and* tells us why (no early return; every lane publishes),
while my (c) pass puts a hard precondition on the second (bounded spin + soft fail, or it can wedge a
box). Where §2b says a fork kernel "runs correctly on gfx900", I am explicitly not endorsing that
without a receipt — my own bf16 finding is the cautionary case for this line: a claim that looked like
a compile guarantee and turned out to be silent emulation.
