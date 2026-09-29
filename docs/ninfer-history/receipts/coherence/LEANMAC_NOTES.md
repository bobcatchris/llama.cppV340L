# LEANMAC — STEP-0 ISA census: **KILL** (zero-GPU CODE desk, 2026-09-18, amd/tp4-cure)

**Verdict in one line:** the shipped V0 kernel's inner loop **ALREADY HOISTS** the
dequant·coeff product — the emitted ISA issues **8 `v_mul_f32` per (g,v) body, not 64** — so
LEANMAC's entire premise (deleting 56 re-issued muls) deletes muls that do not exist. Per the
design doc's own pre-registration (`docs/amd/GEMM_LEANMAC_2026-09-18.md` §5: "~8 v_mul (already
hoisted) ⇒ NO BUILD — the census model was wrong about the wall … ⇒ post-mortem desk, the
fp32-restructure family closes without a window"), this is the **STEP-0 KILL branch**: no
kernel variant, no `NINFER_TILED_LEANMAC` gate, no `--leanmac` bench, no serving leg, no GPU
window owed. R1 was the kill and R1 was pre-registered as decisive.

## 1. Method receipts (HFMA2_notes method, law flags verbatim)

Compile-only `-S` pass of the **committed** `src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu`
(working tree file, not in any desk's modified set; the shipped `.o` in
`build-hip-amd/` was built from this exact source 4 min after its mtime), staged in
`/tmp/leanmac_build/v0_current.s`, **RC=0**:

```
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx900 \
  -DNINFER_HIP_ROSTER=1 -DNINFER_NVFP4_SIMT_LANE=1 -DUSE_PROF_API=1 \
  -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I src/common/hip_shim -I include -I src -isystem /opt/rocm/include -isystem /opt/rocm-6.2.0/include \
  -x hip -S src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu -o /tmp/leanmac_build/v0_current.s
```

(Flags read off `build-hip-amd/src/CMakeFiles/ninfer_hip_host.dir/flags.make` — the exact law
line; `-x hip` is required, bare `.cu` trips CUDA-mode arch rejection.) Production bodies
counted: `nvfp4_tiled_gemm_hip_kernel<STILES,128>` at STILES ∈ {80 (K=5120, three of the five
W4 shards), 96, 24, 272} — **all four byte-for-byte identical in op mix** (same 144/1024/64/144
counts below), so the census covers every W4 shard geometry.

## 2. The census, model vs emitted (per rolled kt-loop body; the g/h/v/t/r nest is fully
unrolled inside it)

Emitted body (STILES=80, TN=128, VGPR 128, private-segment 80 B/thread):

| op | emitted count | decomposition | census model said |
|----|---------------|---------------|-------------------|
| `v_mul_f32` | **144** | 16 coeff (per (row,group) fold) + **16 blocks × 8 pw muls** | 16 + 32×**64** = 2064 |
| `v_fma_f32` | 1024 | 16 blocks × 8 tokens × 4 rows × 2 elements | 2048 (2 bodies' worth) |
| `ds_read_b64` | 64 | 16 × 4 rows (pair table) | 128 |
| `ds_read_b32` | 144 | 16 × 8 xw + 16 scale-table (coeff) | 256 + 16 |
| `buffer_load_dword` (scratch) | 8 | coeff[r][g] reloaded per (g,v) block | not in the model |
| `buffer_store_dword` (scratch) | 16 | coeff[4][4] stored once per k-tile | not in the model |
| `v_bfe_u32` | 33 | the code-byte field extract **already landed** (census §1 row 3's open question: answered YES) | — |
| `s_waitcnt lgkmcnt(0)` | **129** | **one per xw token load — every LDS read's full latency exposed serially** | not in the model |

Read the emitted (g,v) block structure directly (this is LEANMAC's §3 pseudocode, verbatim,
compiler-generated): 4× `ds_read_b64` (pair table `pr[r]`) → 8× `buffer_load_dword` (coeff from
scratch) → **8× `v_mul_f32`** forming `pw0[4]/pw1[4]` (e.g. `v_mul_f32_e32 v119, v69, v99`) →
8 token iterations of pure `v_fma_f32` chains reusing the identical pw registers (`fma(v119,
xl, acc)` ×4 then `fma(v84, xh, acc)` ×4 per token). LICM performed the CSE **because V0's
invariant is clean** — `bits_to_f32(pr[r].x/y) * coeff[r][g]` is textually loop-invariant over
the token nest. (Why did S5 keep 34 alive per HFMA2_notes while V0 hoists fully? S5's coeff was
not cleanly invariant over its loop shape; V0's is. The "evidence AGAINST full hoisting" datum
was the exception, not the rule.) The body covers 16 (g,v) blocks per iteration and the kt loop
walks each 64-wide k-tile in two such iterations — loop-structure detail, irrelevant to the
op-mix verdict.

**Ops/MAC ledger, emitted reality:** (144+1024)/2048 MACs ≈ **1.57 ops/MAC** ALU-class —
LEANMAC's target 1.50 is already what ships (and v_bfe already landed, so census (c) is taken
too). LEANMAC's delta would be exactly zero instructions.

## 3. Where V0's 0.28-0.34-of-model gap actually lives (the post-mortem this kill buys)

Spec §4-R1 wrote: "in which case V0's 2.0-2.4 TF/s is 0.28-0.34 of a 7.2 model" — measured
band 1.989-2.444 TF/s against the ~96-op emitted body's ~7.2 TF/s issue model = **0.28-0.34,
branch realized exactly**. The wall is not ALU issue count; the dump shows the non-ALU legs
the source-static census under-counted:

1. **Serialized xw loads — the dominant term.** 129 `s_waitcnt lgkmcnt(0)` per body: each of
   the 128 token-loop xw loads is load → **wait-full-LDS-latency** → unpack → FMA, with zero
   software pipelining of the next xw under the current FMA chain. ~128 exposed LDS
   round-trips per body vs ~192 ALU-issue slots: the load-use stall, not the op count, is the
   ceiling. This is why V0 measures 0.44-0.54 of even the WRONG (2064-op) census model's 4.54.
2. **coeff spilled to scratch.** `.amdhsa_private_segment_fixed_size 80` (bytes/thread):
   LICM computed the 16 coeff values per k-tile, then chose scratch (`buffer_store_dword` ×16
   per body, `buffer_load_dword` ×8 per block) over +16 VGPR — VGPR pressure is real
   (`next_free_vgpr` 128, acc file 32/thread at TN=128). Scratch ops bill the VMEM pipe and
   add vmcnt waits inside the loop. (The spec's R2 "any spill in the loop body = RED" would
   have flagged the SHIPPED kernel; it shipped and measured fine — the RED rule was for
   protecting a *new* arm's delta, and its absence here underlines that VGPR/spill is not
   where V0's remaining headroom is either.)
3. Barriers/LDS-port share — as censused (R3), the minor term.

**Family closure:** the fp32-restructure family (LEANMAC and any mul-CSE sibling) is closed
without a window, per the pre-registered branch. The bench floor (≥1.4× everywhere), the FNV
gate, and the 402-494/610-750 ms fork are **moot** — they gate a build that step-0 forbids.
Honest note for any future desk: the visible next lever in this kernel is (1), pipelining the
xw loads (prefetch next token's `ds_read_b32` before the current FMA chain — V4-double-buffer
class restructuring at fixed shape), NOT arithmetic reform; and (2) suggests asking why 16
coeff live in scratch at VGPR 128 rather than hoisting the coeff loads per group. Both are
post-mortem-desk material, priced nowhere, claimed nowhere — recorded only so the closure is
not read as "nothing left."

## 4. Free post-mortem owed: the PK tiled kernel's convert/stage bill, DUMPED AT LAST

`V0PK_row`'s "convert/stage bill dominates" was an inference; here is the dump
(`nvfp4_tiled_gemm_hip_kernel_pk<80,128,W>`, same TU, same flags). **The packed arithmetic is
innocent**: 1024 native `v_pk_fma_f16` per body (2048 MACs), zero scalarized-f16 fallback —
the small_t-PK ISA result (native v_pk_fma_f16, no emulation) transfers to the tiled
instantiation, and W ∈ {16,32,64} all show it. The real kill, now proven, is the **spill/occupancy
storm**: `next_free_vgpr` **256** — the entire per-lane file (vs V0's 128; vs small_t-PK's
clean 56) → 1 wave64/SIMD, 4× the occupancy loss — and `.amdhsa_private_segment_fixed_size`
**804/820/820 B per thread** (W=16/32/64) with **204-212 `buffer_store/load_dword` per body**:
the `s2[r][t]` half2 window, `coeff2`, and the staged-x working set do not fit the file and
cycle through scratch on the VMEM pipe, serialized under **478-570 `s_waitcnt`** per body. The
cvt bill — the suspected dominant term — is real but minor: 256/128/64 `v_cvt_f32_f16` per
body (W=16/32/64; fewer at wider W = fewer flushes), 6-12% of the MAC count. Verdict: PK died
of register-file exhaustion (half2 state doubles the live set → scratch → VMEM-serialized
inner loop → 0.21-0.36×), exactly the "suspected spills, never proven" half of §4-R4 — now
proven, and the "scalarized half2" half is refuted. Lesson re-encoded for any packed-arithmetic
return: the design constraint is not the cvt pipe, it is keeping the packed window IN REGISTERS
(i.e. a fp16 window must shrink the fp32 accumulator live set, not add to it).

## 5. What was NOT done (deliberately, per pre-registration)

- No edit to `nvfp4_tiled_gemm_hip.cu` (V0 byte-untouched; FIX-A's `gated_delta_net/` files
  never read-written, per desk law).
- No `NINFER_TILED_LEANMAC` gate, no `--leanmac` bench arm, no serving leg, no GPU window.
- Zero GPU work; one compile-only `-S` pass, RC=0, zero warnings relevant to this desk.

## 6. Provenance

Bases: `docs/amd/GEMM_LEANMAC_2026-09-18.md` commit cc455b648 (the spec; §5 step-0 branch text
quoted above; §1 census; §4-R1/R2/R4) · `src/ops/linear/nvfp4/nvfp4_tiled_gemm_hip.cu`
(:53-177 V0 kernel, :584-746 PK kernel; committed source, no drift) ·
`build-hip-amd/src/CMakeFiles/ninfer_hip_host.dir/flags.make` (law flags) ·
`results/amd/coherence/HFMA2_notes.md` (method; S5 34-v_mul datum, now explained) ·
V0PK_row / PK_row / PLOG-045 (the measured 0.21-0.36× this dump now explains) ·
TILED_SWEEP_notes (shape family still closed; unaffected). All counts in this file are from
`/tmp/leanmac_build/v0_current.s` and `/tmp/leanmac_build/v0_tn128_body.s` (re-derivable with
the §1 command).
