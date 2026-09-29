# 152 — CUDA review checklist (pre-flight for any new kernel)

Prepared by agent1 (KV lane) as the assigned DFlash2 CUDA reviewer. Written as an unnumbered
lane artifact; the coordinator numbered it here and widened it to a **repo-wide pre-flight for
any new kernel**, not just DFlash2 — so read the DFlash2-specific notes as worked examples of
classes that apply to every new kernel, and apply the rest unconditionally.

**Scope note.** Items are ordered by how this repo actually got hurt, not by CUDA convention.
A is launch-environment and geometry classes (A1–A4 came from the review assignment; **A5 was
not** — it came out of the first review this checklist was applied to). B is structural traps
that produced real bugs (B1–B4 from the KVarN lane; **B5 from DFlash2**). C is test quality,
deliberately about the checks that FOOLED us rather than the ones that caught things.

**Purpose.** Every item is a bug that already happened in this repo, with the commit or comment
that records it. The point is not general CUDA advice — it is that this codebase has a specific
injury history, and a fresh kernel directory walks into the same traps because they are invisible
individually.

**Known limit, stated on purpose.** The two most serious findings in the first review cycle that
used this checklist (A5, B5) were **not** in it. Both were only findable by running something —
a sanitizer and a capture — rather than by reading. A checklist catches the previous war; the
instruction that goes with it is that every review ends by adding what it missed.

**For whoever is writing a new kernel:** self-running this before requesting review is worth more
than waiting for the review. Anything you disagree with, say so in the ping — several items here
were contested findings, and two of my own claims from earlier in this lane turned out wrong and
were corrected in the record.

---

## A. Launch-environment and geometry classes

### A1. Width / dtype hardcoding behind a shared marker
**Precedent:** docs/117 §9. Four separate sites baked k4v2 geometry in while the tier signal
travelled in a different field. The worst was `ops/wrapper/gqa_attention.cpp` `validate_cache`
requiring `k_pages {D/2}` / `v_pages {D/4}` for *any* `KVARN_K4V2` — it sat **downstream** of
every site the checklist named, so a perfect upstream chain still failed. Fixed in `35d383ce` /
`9759fa55`.

**DFlash2 is exposed to this more than KVarN was.** W4A16 means the weight width is fixed and
the activation width varies — the inverse of the KVarN layout. Check:
- Does any shape check, plane push, or smem size derive from a literal (`/2`, `/4`, `* 4`)
  instead of the width field?
- If a second quant format ever lands (W8A16, the Q4_K_M/Q8_0 variants named in docs/56), does
  every site that reads "the" weight width see it?
- Is there a single table mapping format → widths, or is the pair repeated per call site?

### A2. Latent 2-rank assumptions
**Precedent:** `tp2_backend.cpp` carries `kv_heads = 2  // TP: local kv-head block (4/2)`,
`conv_channels = 5120`, `output_rows = 248320` as literals. Those are *derived* quantities
written as constants, so world=2 is baked into the shape of the tensors rather than expressed
as a divisor.

**Specific to DFlash2:** it reads target hidden states at layers 5/19/33/47/61 and selects over
16 candidates/slot (docs/118 §1). Under TP2, are the layer indices, the candidate dimension, or
the hidden width sharded? If the drafter is **replicated** per rank rather than sharded, say so
in a comment at the top of the kernel — a reader who assumes sharding will "fix" it wrongly.
Check that any `head_dim / 2`, `n_candidates / world`, or vocab-shard arithmetic is written as
a division by the rank count, not as the evaluated constant.

### A3. smem sizing and the occupancy cliff
**Precedent:** slice6 had to go **single-buffered** because double-buffering k5v4 would need
18464 B dynamic on top of ~37 KB static, past half the 100 KB/SM budget, dropping the CTA to
1/SM. Recorded in the header, not discovered later.

**Also precedent, and nastier:** `cudaFuncSetAttribute` applies to the function **on the
caller's current device**. The launcher comment is explicit: a once-per-process `static bool`
guard let the first rank consume the flag, and the second rank then launched with the attribute
**UNSET** on its device → `cudaErrorInvalidValue`, observed as a warmup crash under
decode_guard, timing-dependent. The fix is `static thread_local bool`.
- If DFlash2 opts into dynamic smem, is the guard `thread_local`?
- Is the static+dynamic arithmetic stated in a comment with the resulting CTAs/SM?

### A4. Shared-memory bank conflicts
**Precedent:** slice6's K bank is linear at 20 B/row, so the dequant's `d = 8*lane + i` reads
land 160 B apart and `160 mod 32 == 0` → all 32 lanes hit one bank → 32-way `LDS.U8` conflict.
The 4-bit route avoids it with a skew. **Recorded as known debt, not solved** — which is the
right outcome for a capacity tier, but only because it was written down.

**DFlash2 exposure:** a W4A16 dequant reads packed 4-bit weights, so the same arithmetic
applies with different constants. Compute `row_stride mod 32` for the actual per-lane read
stride and state it in the header. If it's 0, either skew it or record it as debt with the
number.

### A5. Capture-safety — legal standalone, illegal under graph capture
**Precedent:** DFlash2 review, `a4913686`. The conv/edge-score launchers called
`cudaStreamSynchronize` for test determinism, and the header labelled it *"known debt, not
hidden"* to be dropped at wiring. It is not debt. It cannot ever run inside a captured graph:

    launcher threw=1 msg="dflash2 conv sync: operation not permitted when stream is capturing"
    endCapture=operation failed due to a previous error during capture

`use_cuda_graph` defaults **true** (`types.h:193`) and DFlash has its own graph family
(`program.h:278` `DecodeGraphFamily dflash_graphs`), captured with
`cudaStreamCaptureModeThreadLocal` (`decode_graph.cpp:61`). So the sync does not merely cost
latency — it throws *and* poisons the entire capture.

**Why this class is easy to miss:** the API is perfectly legal on its own, the test passes, and
the label says "debt". Only running it under capture reveals it as fatal. Same shape as A3's
per-device `cudaFuncSetAttribute` hazard — an interaction with the launch environment that no
amount of reading the kernel body surfaces.

**Check, for any launcher that could land in a graph family:**
- No synchronising call (`cudaStreamSynchronize`, `cudaDeviceSynchronize`, blocking memcpy /
  D2H + `cudaStreamSynchronize`) inside the launcher. Get determinism by synchronising in the
  **test**, not in the production path — that satisfies both goals instead of trading them.
- No host read of device state, no allocation, no `cudaFuncSetAttribute` on the capture path.
- If a blocking call is genuinely required, say so in the header and state that the op is
  therefore not graph-capturable — do not file it as "debt to drop later".

---

## B. Structural traps that produced real bugs this session

### B1. Designated initialisers silently default the fields you omit
**Precedent:** `kvarn_small_t_unified_launch` built a `PagedKVBatchLayerView` with a designated
initialiser that did not list `kvarn_k_bits`/`kvarn_v_bits`, so every view there carried the
struct's `4/2` defaults. **No compiler warning.** Even with perfect planning, the launcher could
never have seen k5v4. Found only by running.
- Every struct with a defaulted field: audit all construction sites when adding a field.

### B2. Duplicate `inline constexpr` across two headers in one namespace
**Precedent:** slice6 re-declared slice4's `kKvarnKStageRow = 36`. Harmless while each header
is included once per TU; a **redefinition error** the moment one TU includes both. Broke the
launcher TU on first inclusion.
- Cloning a kernel into `ninfer::ops` (or a new `dflash2` namespace): rename constants, don't
  duplicate.

### B3. Silent fallback arms
**Precedent:** non-BF16 storage mapped to `DType::I8`, so `--kv-dtype q4_0` would have served
int8 under the int4 name and **faked the §5 KLD/VRAM validation**. Same class for the BF16 pool
fallthrough faking the §1 VRAM check.
- Every `switch`/ternary over an enum in the new code: does the default arm throw, or guess?
- Note this file's own precedent for env overrides: an unknown `NINFER_KVARN_PREFILL` value is
  **fatal rather than silently falling back**, because "a typo in a perf A/B that quietly runs
  the other route produces a plausible wrong number."

### B4. N parsers over one enum with divergent accepted sets
**Precedent:** three `parse_kv_cache` implementations; `ninfer_bench` knew only `bf16|int8`, so
acceptance and t/s were unmeasurable for **every** KVarN tier including the shipped one. Fixed
in `35d383ce` with `ninfer::kv_cache_from_name`.
- DFlash2 adds a speculative backend / artifact format. Is it in the **shared** table, or a new
  local parser? If new, which tools can no longer select it?

### B5. A buffer capacity that lives in prose instead of the signature
**Precedent:** DFlash2 review, `a4913686` — the highest-severity finding of that review, and it
was invisible to reading, mutation testing and a passing suite. `launch_edge_scores` documented
its output as `scores [top_k * n_pred]`; with `n_pred == 1` that is `top_k` floats.
`launch_repeat_single_pred` then writes indices `top_k .. top_k²-1`. A caller doing **exactly
what the header says** gets an out-of-bounds device write:

    $ compute-sanitizer --tool memcheck ./cap     # TK floats allocated, per the documented contract
    ========= Invalid __global__ write of size 4 bytes        (241 errors)

The shipped test never caught it because it happened to allocate `TK*TK` — the correct size for
reasons unrelated to the documented contract. That is the general trap: **when the required
capacity is stated in a comment and enforced by luck, a passing test is not evidence.**

Device OOB writes usually corrupt other allocations rather than fault, so this class surfaces
much later as something unrecognisable.

**Check:**
- Any helper whose required buffer size is larger than what a sibling function's documented
  contract implies: put the capacity **in the signature** and throw when it is short.
- Prefer a parameter over a prose invariant. `int capacity_floats` beats "caller must allocate
  top_k*top_k" in three places in a comment block.
- Run `compute-sanitizer --tool memcheck` on at least one test that allocates exactly what the
  documentation promises — not what the author knew to allocate. That is the only way this class
  shows up.

**Related, same root cause:** unchecked preconditions that imply OOB *reads*. In `a4913686`,
`C == n_groups * group_size` is never validated, so `g = c / group_size` can index past the
`proj_out` rows. Validate the relationships the indexing assumes (B3's rule: throw, don't guess).

---

## C. Test-quality checks (the ones that fooled us, not the ones that caught things)

### C1. A kernel-level oracle cannot detect missing dispatch — REPO.md §4 rule 10
**Precedent:** slice6 was `#include`d by nothing but its own oracle, which launches the kernel
**directly**. The oracle was green against a kernel no production path could reach;
`libninfer_ops` contained zero slice6 symbols. At least one test must drive the **public op**.
- For DFlash2: is there a test that goes through the runtime/launcher, not just the kernel?

### C2. An oracle that mirrors the kernel's dequant self-certifies
**Precedent:** the k5v4 oracle reproduced the kernel's own unpack, so a shared misunderstanding
would pass. That is why the **known-answer vector** mattered: `0x41,0x0C,0x52,0xCC,0x41` derived
on paper from the docs/69 LSB-first rule, independent of both implementations.
- For W4A16: is there at least one hand-derived vector computed from the packing spec, not from
  the code?

### C3. Vacuous test data — a passing test that cannot see its subject
**Precedent:** `tests/slice4_kvarn_test.cu` used one flat scale range for all six fields, so
dequantised K/V landed at ~1e-3, softmax degenerated to a uniform average, and the output was
**independent of K**. Proved by mutation: stubbing the K dequant to zero left the test passing
10/10 — on a shipped tier's gate.
- Mutation-test DFlash2 the same way: stub the hidden-state input, the path selector, the
  candidate gather. If the test still passes, it is not testing what it claims.

### C4. Fused append needs two checks
**Precedent:** the oracle consumed the cache read back from the device, so a wrong quantiser was
self-consistent. Catching an `absmax/127`-style bug required checking the **written** codes and
scales against an independent host quantisation.
- If DFlash2 writes a cache or a draft buffer, verify the written bytes independently, not just
  the round-trip.

### C5. A selftest that bypasses the CLI is not testing the CLI
**Precedent:** `validate_117_dtypes.py` raised `AttributeError` on **every** real `kld`/`vram`
invocation, while `--selftest` reported ALL PASS — because the selftest returns above the broken
line. Fixed in `6ac82f77`.
- If DFlash2 ships a tool or script, does its selftest exercise the same entry point users do?

---

## D. Process items

- **Disk:** never `cmake --build build` without `--target` — 88+ executables at ~200 MB each
  (each statically links the 207 MB `libninfer_ops.a`) ≈ 17 GB. REPO.md §4 rule 9.
- **GPU:** written grant before a server launch; guard on foreign CUDA contexts, not ports;
  kill only PIDs you started; `df -h /` before any build > ~5 GB.
- **Syntax-clean ≠ semantically-clean:** a merge left two consecutive identical `if` blocks in
  `serve_options.cpp`, the stale one shadowing the accurate one, so users were told the dispatch
  was unwired two commits after it landed. Build passing did not detect it. Re-read your own
  merged hunks.
- **Verify an instruction's premise before executing it**, especially a destructive one. Three
  wrong instructions arrived from a good-faith source in one session.

---

## E. Suggested review order when the kernel lands

1. Read the header comments first: does it state smem arithmetic, bank behaviour, TP replication
   or sharding, and known debt? Absence of these is itself a finding. Then **distrust the
   labels**: "known debt" on something that is actually fatal is a real failure mode (A5).
2. `grep` for literals where a derived quantity belongs: `/ 2`, `/ 4`, `* 4`, `8192`, `4096`,
   `160`, `128`, and any bare head/width/vocab constant.
3. Enumerate every struct construction with a designated initialiser; diff the field list
   against the struct definition.
4. Check every `switch`/ternary default arm throws.
5. **Diff every buffer's documented capacity against every writer's actual index range** (B5).
   Then check the launcher for anything illegal under capture (A5) — the cheapest version is to
   wrap one call in `cudaStreamBeginCapture`/`EndCapture` and read the error.
6. Only then read the kernel body.
7. Run the tests, then **mutate** them. Report which mutations were caught. A mutation battery
   proves the tests that exist are armed; it says nothing about paths no test reaches — exercise
   every arity/branch the signature allows (in `a4913686`, `n_pred > 1` was the lattice's real
   use and was never called, though it happened to be correct).
8. Confirm at least one test drives the public op (rule 10).
9. Confirm the **recorded build command runs as written**. A validation gate whose repro doesn't
   work is a gate nobody can re-check. `a4913686`'s failed twice: a missing `-I` for the oracle
   header, and no arch flag ("PTX compiled with an unsupported toolchain" on `sm_120a`).

Steps 1–5 are cheap and find the highest-severity issues in this codebase; they are the ones
that get skipped. Steps 5 and 9 were added after the DFlash2 review, which is the first cycle
where the two most serious findings were **not** in this checklist — they were found only by
running a sanitizer and a capture. That is the honest limit of a checklist: it catches the
previous war, and each review should end by adding what it missed.
