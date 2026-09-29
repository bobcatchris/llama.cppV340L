# 57 — KVarN P2 Attention-Path Integration: Agent Handoff (self-contained)

**Status:** CURRENT — work order for a single implementer agent.
**Mission:** complete KVarN P2 (attention-path integration) and P3 (live
battery) so that `--kv-dtype kvarn_k4v2` is a working, tested first-class KV
storage mode on this TP2 stack. Read this document fully before writing code.
Background reading (optional but recommended): `docs/54_kvarn_implementation.md`
(§9 has the history + P1/P2b results), `REPO.md` (repo standards).

---

## 1. What KVarN is (60-second version)

KVarN (huawei-csl/KVarN, Apache 2.0, arXiv:2606.03458) is calibration-free KV
quantization: per 64-token tile, per head — Hadamard transform along the
256-channel axis, Sinkhorn variance normalization (16 fixed iterations), then
asymmetric RTN: **K at 4-bit, V at 2-bit**. Storage cost ≈ 48.5% of int8 KV
(16,896 B vs 34,816 B per page/head). Goal on this machine: free ~1.9 GB/rank
at 200k context so the model fits with headroom (docs/52, docs/56 DFlash).

**Already built and verified (do not redo):**
- P1 CPU codec reference: `src/ops/kvarn/kvarn_codec.{h,cpp}` +
  `tests/test_kvarn_codec.cpp` (determinism, round-trip SNR/cosine).
- P2b GPU tile codec: `src/ops/kvarn/kvarn_tile_cuda.{h,cu}` +
  `tests/test_kvarn_tile_cuda.cpp` — bit-deterministic across runs, matches
  the CPU reference (codes <0.1% last-ulp flips, dequant cosine >0.999, SNR
  within 0.5 dB).
- P2c step 1 pool layout: `DType::KVARN_K4V2`, packed code planes in the paged
  pool, per-tile scale side table, view plumbing (commit `0b2a073e`).

**What you are doing:** steps 2–4 of §6 below (write path, read kernels,
budget + guard lift + P3 battery).

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; branch `main`; remote `github`).
  **This is the only checkout — work in it directly, commit per step.**
- Build: `cd /home/intel/ninfer/repo/build && cmake --build . -j 16`
  (CUDA arch forced to `sm_120a`; 2× RTX 5060 Ti 16 GB).
- Unit tests: **use `/usr/bin/ctest`** (the `ctest` on PATH is a broken Python
  wrapper). From `build/`: `/usr/bin/ctest -R "kvarn|kv_cache|runtime_mechanisms"`.
- GPU tile codec test: `./tests/ninfer_kvarn_tile_cuda_test` (needs GPU 0 free;
  prints "all checks passed").
- Live server battery: `tools/smoke/test_serve_correctness.py` (T1–T15) and
  `tools/ops/run_ci.sh` (build + full gate, manages server lifecycle).
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.
- Reference standard for behavior: the user's llama.cpp command runs this same
  model at 200k on these GPUs; we must not regress below int8-KV quality.

## 3. Architecture facts you need (verified, do not re-derive)

- **TP2 only.** This project is tensor-parallel-2 across both GPUs. Single-GPU
  paths exist in the codebase but are irrelevant here.
- **The serve engine is `TPEngine`** (`src/runtime/tp2/tp_engine.cpp`) using
  `TpBackend` (`src/runtime/tp2/tp2_backend.{h,cpp}`) and rank state in
  `src/targets/qwen3_6/impl/runtime/`. `ConcurrentExecutor` is single-device
  only — changes there do nothing on this server (see REPO.md §2a).
- **Model: Qwen3.8-27B, hybrid GDN.** Per rank: 16 full-attention (GQA) layers
  + 48 linear-attention (GDN) layers; `kv_heads = 2` per rank (local block of
  4/2), `head_dim = 256`; MTP adds 1 GQA layer (`mtp_kv_heads = 2`). Only the
  GQA layers use paged KV — that's what KVarN touches. Spec constants are in
  the `DecoderStateSpec` construction in `tp2_backend.cpp` (~line 190).
- **Paged pool** (`src/core/paged_kv_cache.{h,cpp}`): page = 64 tokens
  (`kPagedKVPageSize`). Planes are per-token: storage shape
  `{leading_extent, 64, heads, physical_pages}` (PageMajor). `block_table`
  maps logical→physical pages. KVarN code planes (already in place):
  - K codes: `{U8, leading = D/2 = 128}` — byte = `(q_tok_hi << 4) | q_tok_lo`
    over token pairs; `q ∈ 0..15`.
  - V codes: `{U8, leading = D/4 = 64}` — byte = `(c3<<6)|(c2<<4)|(c1<<2)|c0`
    over channel quads; `q ∈ 0..3`.
- **KVarN tile = one page** (G=64 tokens, D=256 channels). Fixed.
- **Scale side table** (already in place): fp32
  `[layers, heads, physical_pages, 1152]` per pool (text and MTP each have
  their own), carried on views as `kvarn_scale_pages`. Field order per tile:

  | offset | count | field            |
  |--------|-------|------------------|
  | 0      | 256   | s_col_K          |
  | 256    | 256   | zp_K             |
  | 512    | 64    | s_row_K          |
  | 576    | 256   | s_col_V          |
  | 832    | 256   | zp_V             |
  | 1088   | 64    | s_row_V          |

- **Dequant math** (what the read kernel computes in smem):
  - K: `B[i][j] = (q[i][j] − zp_K[i]) · s_col_K[i] · s_row_K[j]`
    (i=channel, j=token), then inverse FWHT along the channel axis → K̂.
  - V: `B[j][i] = (q[j][i] − zp_V[j]) · s_row_V[j] · s_col_V[i]`
    (j=token, i=channel), then inverse FWHT along the channel axis → V̂.
  - Inverse FWHT: in-place normalized butterfly over 256 channels, 8 stages
    (`len = 1,2,…,128`), pair (u,v) → (u+v, u−v), then × `1/sqrt(256)`;
    self-inverse. The exact unpack + math already exists in the dequant
    kernels of `kvarn_tile_cuda.cu` — mirror it.
- **Tile codec API** (`src/ops/kvarn/kvarn_tile_cuda.h`, verified):
  - `quantize_k_tile_gpu(in bf16 [D][G] channel-major, q_packed, s_col[D], zp[D], s_row[G], stream)`
  - `dequantize_k_tile_gpu(q_packed, s_col, zp, s_row, out bf16 [D][G], stream)`
  - `quantize_v_tile_gpu(in bf16 [G][D] token-major, q_packed, s_row[G], zp[G], s_col[D], stream)`
  - `dequantize_v_tile_gpu(q_packed, s_row, zp, s_col, out bf16 [G][D], stream)`
  CPU reference for host-side tests: `kvarn_codec.h` (float in/out, same names
  without `_gpu`).

## 4. Key call sites (anchors — verify line numbers before editing)

- KV spec construction (dtype wiring already done): `tp2_backend.cpp` ~L190
  (`kv_dtype = … KVARN_K4V2`, `kv_quant_group = 64`).
- Text-pool attention + **fused append**: `src/targets/qwen3_6/impl/runtime/text_context_impl.h`
  L1092–1160 — `ops::gqa_attention(q, k, v, positions, …)` with non-empty
  k/v appends the new tokens into the pool inside the kernel, then attends.
  **This is where KVarN's write path differs most from today.**
- MTP-pool explicit append: same file L619 (`ops::gqa_kv_append(…, mtp_kv_.layer_view(0), …)`);
  MTP attention L663 (`gqa_attention_cached`).
- Trim/rollback (MTP rejection): `src/targets/qwen3_6/impl/runtime/program_impl.h`
  L1015–1022 (`trim_tokens`, `cancel_unmapped_entitlement`).
- GQA public API: `include/ninfer/ops/gqa_attention.h` — `gqa_attention`
  (batched, fused append), `gqa_attention_cached` (single-seq, no new tokens),
  `gqa_kv_append`.
- Wrapper + cache validation: `src/ops/wrapper/gqa_attention.cpp`
  (`validate_cache` currently asserts I8/BF16 plane shapes — add a KVarN
  branch: U8 code planes leading D/2 & D/4, `kvarn_scale_pages` present).
- Kernel files to mirror: `src/ops/kernel/gqa_attention_decode_i8.cuh`
  (decode T=1..6; shared scaffolding in `gqa_attention_decode.cuh`, geometry
  in `gqa_attention_geometry.cuh`) and `gqa_attention_prefill_i8.cuh` /
  `gqa_attention_prefill_bf16.cuh` (+ `_common.cuh`).
- Budget + startup guard: `src/runtime/tp2/tp_engine.cpp` — L~305 hardcoded
  `kv_bytes_per_token = 8000` for KVarN; L~357–369 the guard that rejects
  `--kv-dtype kvarn_k4v2` (remove in step 4).
- Pool planning: `src/targets/qwen3_6/impl/state/decoder_state.cpp`
  (`plan_cache` — KVarN branch already done).

## 5. Design decisions (FINAL — do not re-litigate)

1. **No staging/shadow pass.** Dequantizing the whole history to bf16 every
   decode step is ~4.7 GB extra DRAM traffic/step at 200k. Dequant happens in
   smem inside the attention kernel.
2. **bf16 QK + PV MMA for KVarN** (no int8-QK tensor cores — K must be
   dequanted first). Bandwidth win is preserved: DRAM bytes/token are halved
   for K, quartered for V vs int8.
3. **Write path split:**
   - `gqa_kv_append_kvarn(k, v, positions, view, stream)`: scatters bf16 tokens
     into a per-sequence **bf16 tile workspace** (≤ 64 tokens × D × 2 heads per
     layer; ~4 MB/sequence for all layers; allocated in the work arena by
     tp2_backend) instead of pool pages. For the text pool this replaces the
     fused append inside the attention kernel (the KVarN attention kernel does
     NOT append).
   - **Host-side commit:** wherever `positions` are known host-side
     (text_context/tp2), after an append that crosses a page boundary
     (`new_token_count % 64 == 0`), launch one quantize per layer over the
     completed workspace tile → packed codes into pool planes + scales into
     the side table (reuse `quantize_*_tile_gpu`), then clear the tile.
     **Commit is the only writer of packed storage.** No page zeroing needed
     (attention reads only committed pages + tail).
   - **Tail:** attention reads committed pages from the pool and the in-flight
     partial tile (≤ 63 tokens) directly from the workspace. Batched decode →
     per-row device arrays `tail_k_ptr[row]`, `tail_v_ptr[row]`,
     `tail_count[row]` (I32) as extra kernel params; processed after packed
     pages, same math minus dequant. Updated host-side on commit/workspace
     changes.
4. **MTP rollback is clean by construction:** rejected verify tokens live only
   in the workspace and are discarded on trim; no packed page is ever written
   for uncommitted tokens. The MTP pool gets identical treatment — do not
   forget it.
5. **Prefix cache** stores committed packed page ids → works unchanged; ensure
   the prefix slot never owns a workspace tile.
6. KVarN stays **opt-in** via `--kv-dtype kvarn_k4v2`; I8/BF16 paths untouched.

## 6. Execution order (commit + test each step before the next)

### Step 2 — Write path
Implement `gqa_kv_append_kvarn` + host-side commit launch (+ MTP pool),
workspace allocation in tp2_backend, and rewire the text-pool fused-append
call sites to append-to-workspace for KVarN.
**Tests (must pass before moving on):**
- GPU round-trip unit: fill a page's 64 tokens → commit → read packed codes +
  scales back → dequant with the CPU codec → SNR/cosine vs original matches
  `test_kvarn_tile_cuda`'s numbers.
- Boundary crossing across two appends (32 + 32) commits exactly one tile at
  the right page.
- Trim that discards a partial tile leaves packed storage untouched and the
  next commit correct.
- Existing full build + `/usr/bin/ctest` suite stays green.

### Step 3 — Read kernels + dispatch
New `gqa_attention_decode_kvarn` (+ prefill fill+attention variant) mirroring
the i8/bf16 kernel structure: cp.async packed tile → smem, dequant in smem
(§3 math), bf16 QK + PV MMA, per-row tail arrays. Dispatch + `validate_cache`
branch in `gqa_attention.cpp`.
**Tests (isolation trick — isolates kernel bugs from codec quality):**
- Synthetic random bf16 KV → quantize into the pool with step-2 code →
  compare (a) kvarn GQA kernel output vs (b) existing BF16 GQA kernel fed the
  CPU-dequantized KV. Must match to ~1e-3 relative (codec error). Any larger
  delta = kernel bug (swizzle/alignment/smem), not quantization.
- Determinism: two identical kvarn requests → byte-identical outputs.
- Live single-request check vs the int8 and bf16 servers (same prompt,
  compare first tokens + a short generation).

### Step 4 — Budget, guard lift, P3 battery
- Replace hardcoded `kv_bytes_per_token = 8000` in tp_engine.cpp with the real
  `(payload_bytes / capacity)` from the planned layout.
- Remove the KVarN startup guard (tp_engine.cpp L~357–369).
- **Server swap protocol (ask the user first — the live server serves an
  active conversation):** stop with `pkill -x ninfer-serve` (exact name;
  NEVER `pkill -f`), wait for GPUs < 500 MiB, launch per LAUNCH.md with
  `--kv-dtype kvarn_k4v2` (keep `--spec mtp --draft-tokens 3`), run P3, then
  restore the original server exactly as it was running.
- **P3 gate (all must pass at 80k with kvarn):** battery must-pass set
  T1 T2 T3 T5 T7 T8 T10 T11; expected T4 T6 T9 T13 T14; MTP acceptance within
  the regression gate (baseline 82.0%, fail only if < 78%); determinism
  (T10) clean. Log results to `results/` (measurement data is project data —
  commit it).

## 7. Constraints (non-negotiable)

- **No damage:** never leave uncommitted or untested CUDA kernels; commit per
  step with a message naming the step; keep the repo buildable at every commit.
- **One live server at a time**; port 8091 is serving an active conversation —
  do not kill/restart it without explicit user permission (see §6 step 4
  protocol). `pkill -x ninfer-serve` only, exact process name.
- Don't touch I8/BF16 code paths; don't add staging passes; don't use int8 QK
  tensor cores for KVarN; don't write packed storage outside commit; don't
  store scales in pool planes.
- TP2 is the only supported config — no single-GPU workarounds.
- Docs live in `docs/` (numbered); update `docs/54` §9 status as you go
  (P2 → COMPLETE with commit hashes when done).

## 8. Definition of done

1. Steps 2–4 committed to `main` with passing tests at each step.
2. `--kv-dtype kvarn_k4v2` launches cleanly, P3 gate green at 80k, MTP
   acceptance within gate, determinism clean.
3. `results/` snapshot of the P3 run committed; docs/54 status updated to
   "P2 COMPLETE / P3 PASSED" with numbers (bytes/token, MiB/rank freed vs
   int8, acceptance %, battery verdict).
4. Report format when done: one paragraph per step + the P3 numbers table.
