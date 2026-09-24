# 121 — Agent 1 session handoff: KVarN slice-4 decode (done) + Phase 3 (diagnosed, not started)

Date: 2026-08-31. Branch `wo/kv-uniform`, worktree `~/ninfer/worktrees/wo-kv-uniform`.
HEAD at time of writing: `3d2297fa`.
Purpose: complete takeover context for another agent. Assumes nothing from this session's chat.

---

## 0. Read these first (in order)

| doc | why |
|---|---|
| `docs/104_kvcache_uniform_performance_plan.md` | the master plan. §4 phases, §7 risks, §8 baselines/gates. **§4's Phase-2 KVarN residual block and Phase-3 line are both stale — see §6 below; a §9 addendum I added partially corrects it.** |
| `docs/120_phase3_status_and_task_list.md` | **Phase 3 task list, written by me.** Start here for what to do next. |
| `docs/119_kvarn_code_layout_and_quantize_critical_path.md` | why prefill is slow + the critical-path design. §2b = measured null result; §3a = the sync blocker. |
| `results/119b_commit_sync_removal_design.md` | C2 design (sync removal). |
| `results/119c_c2_sync_removal_implementation.md` | C2 ready-to-apply spec, incl. two designs I rejected as unsafe. |
| `results/113_prefill_nsys_attribution.md` | the nsys data that invalidated docs/113's premise. |
| `results/kvarn_unified_decode_matrix_20260830.md` | Phase-2 KVarN decode parity evidence. |
| `results/104_kvarn_unified_prologue_hadamard_design.md` | the decode Hadamard port design + cause-3 analysis. |
| `docs/113_phase3_prefill_fix_work_order_draft.md` | the Phase-3 work order — **has my RESCOPED banner at the top; do not execute its Steps 0–2 as written.** |
| `results/debrief_session.md`, `results/debrief_s8qk_20260831.md` | agent 1's session debriefs (I8 lane). |

## 1. Environment / operational facts you will need

- **2× RTX 5060 Ti, 16 GB each.** cc **12.0**, **36 SMs**, L2 **32 MB** (persisting max 20 MB),
  `sharedMemPerBlock` 49152, **`sharedMemPerBlockOptin` 101376**, `sharedMemPerMultiprocessor` 102400.
  Usable dynamic smem = `101376 − static − ~1024 reserved`. **This bit me:** a 64 KB dynamic request
  with 35 KB static failed `cudaFuncSetAttribute` with a bare `invalid argument` (max was 65280 B).
- Model: `/home/intel/models/qwen3_8_27b.ninfer`, meta sha `45912dd83c71a1b3`.
- **Use `/usr/bin/ctest`.** The `ctest` on PATH is a broken Python shim (`ModuleNotFoundError: cmake`).
- **`ncu` does NOT work on this box**: `ERR_NVGPUCTRPERM`, no passwordless sudo. Needs
  `NVreg_RestrictProfilingToAdminUsers=0` from an admin. **`nsys` works fine.**
- **GPU is a serial resource shared between agents.** Check `nvidia-smi --query-compute-apps` before
  launching; the TP2 decode test needs ~13 GB free per card. Small op tests (~50 MB) can overlap a
  running server, but that perturbs its timings — ask first.
- Disk has hit 100% during this session. nsys `.sqlite` exports are ~400 MB each and are regenerable
  from the `.nsys-rep` — delete those first.
- Typical run:
  `./build/tests/ninfer_tp2_decode_test --artifact ~/models/qwen3_8_27b.ninfer --kv-dtype kvarn --mtp 3 --tokens 128 --ctx 4096 --prompt "The capital of France is"`

## 2. What I accomplished (Phase 2 — KVarN unified decode): COMPLETE

Wired `gqa_decode_slice4_kvarn_kernel` (verbatim slice-2 bf16 body + KVarN dequant prologue) into
the KVarN launcher, replacing the packed-kernel+merge route for T=1..6. **30 commits.**

**Six defects fixed** (all silent-wrong, none crashed):
1. 64 KB FWHT scratch + 35 KB static exceeded the device opt-in → `cudaFuncSetAttribute` invalid-value.
2. Packed-code unpack math: K is `[channel][key/2]` (low nibble = even key), per-key scale field
   `512+key` (not `512+2*key`), plus a missing `(kv_head + phys*kv_heads)*CodeBytes` page base.
3. **`kvarn_fwht_channel` hardcodes its loop stride as `kKvarnAttnThreads`=256**, so it silently
   SKIPS ELEMENTS for any non-256-thread CTA. Use `kvarn_mma_warp_fwht` (register+shuffle,
   documented bit-identical) or a `blockDim.x` stride.
4. Scale table is the FULL `[layers=16, kv_heads, pages, 1152]` tensor ⇒ the (head,page) stride is
   **`1152*n_layers` = 18432**, not 1152. Launcher passes `scale_page_stride`.
5. **The reduce kernel derives `token = blockIdx.z`** — my helper launched `grid.z = 1`, so only
   token 0 was reduced. Correct at T=1, silently stale for every T>1. Canonical:
   `dim3(QHeads, div_up(D,kDChunk), invocation.width * invocation.batch_size)`.
6. MultiBatch partial offsets used compile-time `TokenTile` where the contract needs runtime `tokens`.

**Then a 2–3× perf regression I shipped and had to fix** (default was reverted to packed in
`94218bd6`, then re-flipped to unified in `29d0a797` once fixed):
- **Cause 1 (Hadamards)**: ported packed's conventions — rotate Q once (`qnorm=1/16`), K
  unpack-only (`kvnorm=1.0`, codes already decode to `H(K)/16`, orthogonality gives Q·K), V
  unpack-only (`vvnorm=1/16`) deferred to `kvarn_mma_acc_fwht`. **Bought almost nothing**
  (0.62×→0.60× of packed).
- **Cause 3 (the real one)**: K-code global read `k_page[(8*lane+i)*32 + pkey/2]` puts consecutive
  lanes **256 bytes apart** → ~32× DRAM sector amplification. Fixed by bulk-copying the 8 KB page
  into a padded smem bank (row stride 36 keeps uint32 stores aligned; odd stride 33 would give
  2-way vs 8-way bank conflicts). **This did all the work.**
- **The tell**: the deficit *grew* with context (−37% @10k → −59% @25k → −90% @250k). Compute
  regressions are flat; memory regressions scale. **Remember this diagnostic.**
- Bonus: removing the float scratch restored `__launch_bounds__(Wc*32, 2)` → 2 blocks/SM
  (verified `SHARED=47360B` via `cuobjdump -res-usage`).

**Result:** acceptance-free kernel cost (`ms/round = 1000*tok_per_round/tps`) is **0.928–1.048×
packed across all 12 cells** (6 contexts × greedy/sampling). 250k greedy 4.4 → **42.1 t/s**.

## 3. Phase 3 (prefill): DIAGNOSED, ZERO IMPROVEMENT

**Prefill is still −12.4% vs bf16** (682.5 vs 779.1 t/s pp, identical 21 829-token prompt).
docs/113 Steps 0–5: **0 of 5.**

What I established:
- **docs/113's premise is wrong.** Of a 3.65 s gap vs bf16: `quantize_tile_kernel` = **3.61 s
  (99%)**; `gqa_attention_kvarn_materialize_kernel` = **0.15 s (4%)**. Deleting materialize
  perfectly recovers ~0.5% against a 12.4% deficit. It is also not O(n²) — docs/66 removed that.
- **quantize_tile is algorithm-bound**: `kSinkhornIters = 16` full-tile passes + 8-stage FWHT in
  ~72 KB smem. Not memory. A layout change removes zero operations.
- **`31.38 − 3.61 = 27.77 ≈ bf16's 27.74`** → taking quantization OFF THE CRITICAL PATH closes the
  whole gap with no algorithm/numerics/layout change. That is the real Phase 3.
- **Blocker for that**: `gqa_kv_append_kvarn_and_commit` (`kvarn_workspace.cpp:401-435`) does a D2H
  of positions + **`cudaStreamSynchronize` on every call**, called per full-attention layer ⇒
  **16 pipeline drains per forward pass**. Quantization is issued *after* the drain, so a side
  stream overlaps nothing. Fix = C2 (spec in 119c), then C3.
- **The direct prefill kernel is unwired dead code with NO tail support** (no
  `tail_k/tail_v/tail_count/packed_pages` params) → it silently drops every uncommitted key.
  It is **WRONG, not slow**, on any non-empty tail. docs/113 Step 0 expects it to be "slower,
  that's the data point, not a bug" — that is false; the A/B would measure a broken route.
- **Why nobody noticed**: `tests/bench_kvarn_2pass.cu` is the direct kernel's only exerciser and
  hardcodes `tail_count = 0` — the one config both kernels handle. Now flagged in-file.
- **The repo had ZERO prefill tests** and **no stored token-ID sequences anywhere**, so
  "byte-identity vs Phase-0 prefill output" (docs/104 §4) was unevaluable. Fixed the tooling;
  the reference still needs generating (task A4).

## 4. Test/CI infrastructure I added (all mutation-validated)

| artifact | what it gates |
|---|---|
| `tests/slice4_kvarn_test.cu` (rewritten; old one called a stale signature with host pointers) | decode prologue vs FP64 oracle. 10/10. Non-identity page permutation, `n_layers=3`, real `kvarn_dequant_k/v`, raw tail, key-63/64 boundary, T=1..6 |
| `tests/kvarn_materialize_oracle_test.cu` | what materialize writes (all 3 regions). 6/6. Mutations: scale-stride ✗, code-layout ✗, logical/physical ✗ → all FAIL correctly |
| `tests/prefill_attention_oracle_test.cu` | prefill attention vs FP64. 6/6. Same three mutations → FAIL |
| `tests/slice3_i8_s8qk_cpu_ref.cpp` | host-only CPU oracle for agent 1's s8-QK port. Shows s8 is 3.5–4.6× the bf16 error ⇒ **bf16-vs-s8 byte-identity is not a valid acceptance criterion** |
| `tools/regress_route_ab.sh` | A/B two routes: token identity at short AND long ctx (hard fail), long-ctx t/s ratio ≥0.95. Short-ctx deliberately NOT gated |
| `tools/parity/prefill_byte_identity.sh` + `--dump-tokens` on the decode test | makes byte-identity measurable. Compare defaults to **IDS-ONLY** (route A/B headers differ by construction) |
| `tools/ops/run_verify_tests.sh` KVarN row | CI had **zero** KVarN coverage while it's a shipped default. Includes a **route assertion** off the banner |
| `[kvarn-decode-route]` banner (`9da5f86d`) | one-time stderr line naming the route. **Essential**: unified and packed are within a few percent, so t/s alone no longer proves which ran |

## 5. Measurement methodology — the trap that cost me the most time

**Wall-clock prefill t/s under nsys cannot support a ±5% gate on this box.** Measured: same binary
**599.8 t/s under nsys vs 699.7 without**; two perf-neutral builds differed **12%** nsys-vs-nsys.
I briefly believed I'd shipped a 12% regression that was nothing of the kind.

**Use total GPU kernel time from `nsys stats --report cuda_gpu_kern_sum`** — stable to ~0.1% here.
For decode, use **`ms/round = 1000*tok_per_round/tps`**, which divides out MTP acceptance. The one
OUT cell in my matrix (250k sampling, 0.795) was pure acceptance noise (53.1% vs 71.0%) with
kernel cost 1.041×.

Also: **a short-context A/B cannot gate an attention-kernel change.** My 4k A/B read −4%; the same
defect was −59% at 25k. And docs/104 §8.1 already warns 48-token cells carry ±7pp acceptance noise.

## 6. Known-stale documentation (fix or be aware)

- **docs/104 §4 Phase 2 KVarN residual block** (agent 1's): says "KVarN DECODE default STAYS packed"
  and describes slice4 as using smem `kvarn_fwht_channel` per key through a 32 KB scratch at
  1 block/SM. **All superseded** by `802322a4`/`29d0a797`. My §9 addendum corrects it — but agent 1
  has uncommitted edits to that same file, so check before trusting either.
- **docs/104 §4 Phase 3 line**: "delete `gqa_kvarn_materialize_kernel` (O(n²) pass)" — wrong twice.
- **docs/104 §4 Phase 3 gate**: "vs its own Phase-0 prefill output" — that data does not exist.
  Use docs/113 Step 1's route-vs-route form.
- **Stale comments I fixed in agent 1's files** (`802322a4`): `kvarn_mma.cuh` claimed
  `kvarn_mma_acc_fwht` was "currently UNUSED" (it is LIVE at `.inc:735`); packed `.inc:211-219` and
  the "4b (REMOVED)" block both claimed V is FWHT'd inline and deferral was removed (the live path
  IS deferred). **These three stale comments caused two wrong diagnoses in one session.**

## 7. Immediate next actions

**GPU-bound (needs a slot, ~15 min):**
- **A4** — `tools/parity/prefill_byte_identity.sh capture <dir> --kv kvarn` → creates the prefill
  byte-identity reference that has never existed. Label it "Phase-3 reference (route-vs-route)".
- Baseline the new CI row: first `run_verify_tests.sh --update-baseline` so `kv_kvarn_tps` stops
  reading NEW.
- Verify `a59de6df` (commit-path copies) hasn't regressed prefill — I measured it neutral on kernel
  time but only once.

**CPU-only, unblocked:**
- **B2** — tail support in the direct prefill kernel (blocks all Phase-3 route work).
- Apply **119c** (C2) and hand it to whoever has the card.
- Reconcile docs/104 §4 with docs/120 §3.

**Owner decisions outstanding:**
1. Approve C2→C3 sequencing (the only path to the 12.4%).
2. `kSinkhornIters = 16` is the biggest single multiplier but is a **numerics change** — needs an
   accuracy-gated proposal.
3. docs/104's "A/B byte-identical per variant" needs a **KVarN carve-out**: unified and packed are
   NOT byte-identical (64-key vs 32-key MMA reduction widths → different fp32 accumulation order).
   At 192 tokens they emit the same sentences reordered. Correctness vs the bf16 anchor holds.
4. KVarN's −37% 10k→250k **decode** decay is intrinsic per-key compute (~153 ns/key vs int8's ~51
   on roughly half the bytes), present in BOTH routes. Not fixable by any decode change.

## 8. Things I got wrong (so you don't repeat them)

- Claimed packed had "no tensor cores" — grepped `gqa_attention_kvarn.cuh` instead of
  `gqa_attention_kvarn_decode_packed.inc`. It has 18 `mma_bf16`/`ldmatrix` sites.
- Then over-corrected and claimed V's FWHT was inline — it's deferred; I trusted the stale comment.
- Claimed removing the Hadamards would fix decode; the coalescing fix did it.
- Claimed V-rounding caused the packed-vs-unified divergence; C2 disproved it (reduction width does).
- Called the D2D-copy removal a "free win"; it was free and worthless.
- **Two self-inflicted build breaks caught before committing**: an edit whose replacement region
  spanned into the banner and deleted it; another that swallowed a `return 1`, which would have made
  every bench run exit 1. **Both were caught by re-running the thing that observes the behaviour**
  (banner check, reading emitted control flow) — not by the compiler. Do that.
- My first CI gate (per-element 1 bf16 ULP) was over-clever and wrong: the inverse-FWHT cancels, so
  near-zero elements legitimately sit many ULPs apart at 1.9e-06 absolute error.

## 9. Open items from agent 1's lane (not mine, but blocking shared gates)

- I8 unified route: `cudaErrorIllegalAddress` in `scatter.cpp:86` observed once at model level while
  the oracle passed 8/8; **non-recurring**. Agent 1's theory is environmental (stray
  compute-sanitizer holding 12.5 GB). My counter: illegal-address ≠ OOM (I hit real OOM twice that
  day), so if pressure was the trigger, a failed arena/pool allocation is not propagating.
  Discriminator in `results/119c`. Agent 1 has `tests/repro_tp_crash.cu` untracked.
- I8 A/B in flight: shipped 60.6 t/s vs unified 52.5 (15% gap) but unified measured 56.6 elsewhere
  ⇒ ±7% run-to-run variance on unified while shipped is stable. Wc=4 was a wash → keeping Wc=2.
- Until I8 is green, `regress_unified.sh` cannot pass tree-wide, which gates Phase-3 closeout D4/D5.

## 10. Key constants (reference)

`kKvarnAttnG=64` (keys/page, == `kPagedKVPageSize`, `kPagedKVPageShift=6`), `kKvarnAttnD=256`,
`kKvarnKCodeBytes=8192` (=D·G/2, layout `[channel][key/2]`), `kKvarnVCodeBytes=4096` (=G·D/4,
layout `[key][channel/4]`), **1152 scale fields per (layer,head,page)**: `s_col_K[0..255]`,
`zp_K[256..511]`, `s_row_K[512..575]`, `s_col_V[576..831]`, `zp_V[832..895]`,
`s_row_V[1088..1151]`. `kvarn_scale_at = field + 1152*(layer + n_layers*(head + n_heads*page))`.
Tail: `tail_k[key + 64*(d + 256*head)]`, `tail_v[d + 256*(key + 64*head)]`.
`kGqaKvQuantGroup=64`, `kGqaKvQuantGroups=4`; I8 Q quant is per **(row, GROUP)**, symmetric
`max|q|/127`, no zero point. `kGqaPrefillBr=kGqaPrefillBc=64`, `kGqaPrefillThreads=128`.
`gqa_small_t_tc_swz(row,col) = ((col>>3) ^ (row&7))<<3 | (col&7)` — keeps 8-channel groups
contiguous and 16 B-aligned.
