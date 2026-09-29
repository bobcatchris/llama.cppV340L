# PIPE — STEP-0' ISA acceptance: **KILL** (zero-GPU CODE desk, 2026-09-18, amd/tp4-cure)

**Verdict in one line:** STEP-0 **confirmed** the design's diagnosis (shipped V0 = **129
`s_waitcnt lgkmcnt(0)`/body + 24 scratch ops + 80 B private segment + `next_free_vgpr` 128**
at TN=128, STILES ∈ {80,24,68} — the wall is real, exactly as `GEMM_PIPELINE_2026-09-18.md`
§1 priced it), but the pre-registered FIX failed its own step-0' acceptance in **10 measured
structural cells**: the one-wait-per-(g,v)-block batch is either **re-sunk by the scheduler**
(every plain-C++ batch, 8/8 cells) or **realizes at 0.9-2.4 KB/thread spill** (every pinned
cell), and the three signatures (`lgkmcnt` ≤ 20/body, scratch 0, VGPR ≤ 128) are **jointly
unsatisfiable at V0's tile on this toolchain**. Per the design's own pre-registration (§4:
"any signature failing at step-0 → no window; … resolve by structure, never by relaxing the
bar"): **no window owed, no `--pipe` bench arm, no `NINFER_TILED_PIPE` gate — the
load-placement family closes on this kernel.** The x1.45-1.6 / 2.9-3.9 TF/s predictions stay
UNCLAIMED; serving gemm slice stays 798.8 ms/chunk at V0; the 420-590 band is retracted with
the fix.

## 1. STEP-0 census — diagnosis CONFIRMED (method receipts, law flags verbatim)

```
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx900 \
  -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 -DUSE_PROF_API=1 \
  -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I src/common/hip_shim -I include -I src -isystem /opt/rocm/include -isystem /opt/rocm-6.2.0/include \
  -x hip -S src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu -o /tmp/pipe_desk/v0_step0.s   # RC=0
```

Committed source (tree clean at 9fe491f8f). Census of
`nvfp4_tiled_gemm_hip_kernel<STILES,128>` (whole-kernel static counts = one rolled body;
all seven STILES instantiations byte-identical in op mix; shown: the five W4 shards):

| STILES | lgkmcnt(0) | buf_load | buf_store | ds_read_b32 | ds_read_b64 | v_fma_f32 | v_mul_f32 | next_free_vgpr | private B | LDS B |
|---|---|---|---|---|---|---|---|---|---|---|
| 80  | 129 | 8 | 16 | 144 | 64 | 1024 | 144 | 128 | 80 | 19456 |
| 24  | 129 | 8 | 16 | 144 | 64 | 1024 | 144 | 128 | 80 | 19456 |
| 68  | 129 | 8 | 16 | 144 | 64 | 1024 | 144 | 128 | 80 | 19456 |

Matches the design's §1 decomposition to the digit (129 = one wait per xw token load; 24
scratch ops = coeff[4][4] spilled; 1024 = 16 blocks × 8 tokens × 4 rows × 2). **Not a kill —
the fix set went to build.**

## 2. What was built (then reverted at the kill — provenance)

The PIPE family per design §6: `nvfp4_tiled_gemm_hip_pipe_kernel` (V0-verbatim
tables/staging/epilogue, per-(g,v) xw batch `xw[kS]`, pw-overwrites-pr fold, per-group coeff,
`__launch_bounds__(kThreads,2)`), `launch_tiled_pipe_exact` (m=128-only instantiation — at
TN=256 the acc+xw ledger measured 700 B spill at 1 wave, the V4 failure shape; TN=32 spilled
160 B; registered set = what passes the bar, N7 loud-refuse for the rest),
`launch_nvfp4_tiled_gemm_pipe` + header decl, and the `NINFER_TILED_PIPE` value-parsed gate
(strict "1", loud-refuse, PK mutual-exclusion throw, `[TILED] arm=pipe` trace). TU compiled
**RC=0** at the law line. **All reverted** — a failed-acceptance arm ships as nothing
(LEANMAC kill precedent: the notes are the artifact). Tree clean at 9fe491f8f for `src/`.

## 3. STEP-0' acceptance — the probe matrix (all cells at TN=128, STILES=80; RC=0 each)

| cell | structure delta vs V0 | static body | lgkmcnt(0) | scratch B (bl+bs) | VGPR | signatures |
|---|---|---|---|---|---|---|
| V0 | — (the anchor) | 1024 fma | 129 | 80 (8+16) | 128 | the baseline |
| B | + `launch_bounds(256,2)` ONLY | 1024 | 129 | 80 | 128 | **V0 sits AT the cliff — the bound is free only because there is zero slack** |
| v1 | design §6 sketch: xw batch + sq hold + per-group coeff + bounds | 2048 | 263 | 1396 | 128 | FAIL 1,2 |
| v2 | v1 + `#pragma unroll 1` on kt loop | 2048 | 263 | 1396 | 128 | pragma inert |
| v3 | bases cut (1 code base + 1 scale base, per-row delta 16r) + per-group coeff from global byte + batch | 2048 | 302 | 1352 | 128 | FAIL 1,2 |
| A | v3, bounds REMOVED | 2048 | 259 | 1640 | **64** | allocator RETREATS to 64 VGPR + 1.6 KB scratch |
| C | v3 MINUS the xw batch | 2048 | 292 | 1352 | 128 | restructure alone spills — **the batch is not the spill's cause** |
| F | V0 + per-group coeff ONLY (V0 arrays, no batch) | 2048 | 261 | 884 | 256 | the coeff restructure alone flips LLVM to a 2×-static-body shape |
| H | F + `#pragma unroll 1` | 2048 | 261 | 884 | 256 | 2× body is NOT kt-unrolling; the pragma cannot reach it |
| I | v3 + ALL xw loads `volatile`-pinned | 2048 | **134** | 2364 | 128 | **batch SURVIVES** (4.2 waits/block; pinned reads merged to ds_read_b64) — spill 2.4 KB |
| J | V0 coeff block (verbatim) + pin-LAST-load + batch + bases + bounds | 2048 | 270 | 2396 | 128 | pin works, funding impossible |
| K | J with ALL loads volatile | 2048 | 270 | 2396 | 128 | ≡ J |

Signature scorecard against the pre-registration (§4): **sig 1** (≤ 20/body) — best cell 134
whole-kernel, over a 32-block rolled body = 67 per V0's 16-block census unit; never ≤ 20.
**sig 2** (scratch 0) — never; minimum non-V0 cell = 884 B (F), 11× V0's own 80 B. **sig 3**
(≤ 128) — reachable only under the bound, paid in 0.9-2.4 KB spill that breaks sig 2.

## 4. Mechanism findings (what the ISA actually does — the part worth keeping)

1. **Source-order batching is a suggestion, not load placement.** At this register pressure
   LLVM's post-RA scheduler sinks unpinned LDS reads to their uses at a 100% rate (8/8
   plain-batch cells). The design §6 sketch's "explicit batching caps the scheduler's
   freedom" is refuted: nothing in plain C++ caps it.
2. **`volatile` LDS reads DO pin the batch** (the scheduler will not move memory ops across
   them; LLVM even merges pinned reads into `ds_read_b64` — 12 reads/block became 4 b64 +
   issued as one flight). The batch EMITS. Waits fell 129 → 67/body-unit (cell I). This part
   of the mechanism works.
3. **The +8 registers of a real batch have no funding at V0's tile.** acc[4][8] = 32 pins the
   tile exactly at the 128 cliff (probe B: zero slack). Every restructure priced to free
   registers measured NEGATIVE: per-group coeff recompute (+4-12 regs on paper) flips the kt
   loop into a 2×-static-body pipelined shape (2048 v_fma) doubling pipeline state; the bases
   cut (−10) is real but insufficient; dropping the bound lets the allocator retreat to 64
   VGPR + 1.6 KB spill instead of going tall; `#pragma unroll 1` is inert against the 2×
   shape. The design §2 ledger's "78-86 source regs → predicted 88-106" was refuted at its
   base: it counted coeff as register pressure V0 had ALREADY spilled away (V0's coeff lives
   in the 80 B scratch, not in VGPR) — so (b) frees nothing that was a register, and the
   ledger starts AT the cliff, not under it.
4. **Scratch-0 and the batch are enemies at this tile:** the only cells that realize the
   batch hold 2.3-2.4 KB scratch (60× V0); the only scratch-lean cell is V0 itself.

## 5. What stands, what dies, what would unfreeze (NOT this desk's mandate)

- **Stands:** the latency-serialization diagnosis. The 129-wait wall is confirmed
  independently (STEP-0 census, this desk). V0's 1989-2444 GF/s W4 band and 798.8 ms/chunk
  serving slice remain the truth; V4's lesson (never spend VGPR past 128) is re-confirmed
  from the compiler side.
- **Dies:** the no-arithmetic-change load-placement fix AT V0'S TILE. The register file at
  that tile has no pipeline budget — that is a measured statement, not a judgment.
- **Would unfreeze (a NEW design + its own step-0 gate, not a revival):** a tile re-sweep
  WITH the pipeline as a CONSTRAINT — shapes with ≤ 16 acc/thread (e.g. 32-row blocks: acc
  16, batch 4-8, byte-granular code loads instead of the c8 array) can fund a pinned batch
  under 128. V1's shape lost the UNPIPELINED sweep; the sweep never priced a pipeline. An
  asm/sched-intrinsic-class pin was NOT tried (the file's plain-C++ eligibility audit forbids
  it; adopting it is an eligibility-law decision for a future order) and would hit the same
  funding wall — pinning was never the binding constraint; registers were.

## 6. Method receipts for the next ISA desk (two landmines, both hit this desk)

- Counting unit: instructions live between the FIRST occurrence of a function label and the
  next; the `.s` RE-EMITS each kernel symbol at EOF (section reopen, no instructions) — a
  naive per-label counter RESETS on the second occurrence and reads 0 (this desk shipped a
  census that way for an hour). Anchor labels to `^(_\S+):` (mangled only) or `; %bb.*:`
  parses as a label under looser regexes and steals the body.
- `.amdhsa_kernel <mangled>` blocks carry `next_free_vgpr` / `private_segment_fixed_size` /
  `group_segment_fixed_size` in text `-S` output; demangle via `c++filt` and match template
  args (`<STILES, TN>`).
- `/tmp/pipe_desk/` holds all artifacts (`v0_step0.s`, `pipe_v*.s`, `probe_[A CK IJ]*.s`,
  `census.py`); re-derivable from the command in §1.

## 7. Desk law accounting

No `src/` change survives (implemented + reverted; tree clean at 9fe491f8f), no gate, no
bench arm, no GPU time, no window requested. GPU owner: **nothing to run** — do not rebuild,
do not bank, do not boot for PIPE; the standing V0/PK state is untouched. The
`results/amd/coherence/` battery/boot rows in this worktree are the GPU owner's in-flight
files — not this desk's.
