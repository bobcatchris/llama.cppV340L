# 66 — KVarN D-18 (beyond-capacity prefill collapse): Agent Work Order

**Status:** CURRENT — work order for a single implementer agent.
**Mission:** make KVarN attention fast BEYOND the staged-shadow capacity so a
250k-context server serves long prompts at sustained rate. Done = an 80k-token
prompt prefills at ≥450 tok/s average on a live 250k KVarN MTP server (battery
T19 green, currently FAILING), decode beyond capacity within 2× of in-capacity
rate, must-pass battery green, startup VRAM unchanged.
Read this document fully before writing code.

---

## 1. Context (60-second version)

KVarN KV (`--kv-dtype kvarn_k4v2`) stores packed int4/int2 pages + a bf16
"staged shadow" that holds the most recent ~30.7k tokens (481 pages, 1 GiB
budget). While history fits the shadow, attention runs the existing fast BF16
flash kernels (~700 tok/s prefill). **Past the shadow boundary, dispatch falls
back to the step-3 fused kernel** (`gqa_attention_kvarn.cuh`), which has one CTA
per (q_head, query token) and re-dequants EVERY packed page per query → O(T×P)
with an expensive constant. Measured on a live 33k-token conversation
(2026-08-24): prefill flat ~724→652 tok/s up to 29.7k tokens, then **12.6 →
4.0 tok/s** at 31–32k. Decode beyond capacity is affected too (fused read every
step, ×16 per-q-head redundancy). This blocks KVarN's primary use case — long
context on the 250k server.

**What must NOT change:** in-capacity behavior (staged path) stays bit-identical
(T10/T18 determinism); I8/BF16 KV paths untouched; startup VRAM unchanged
(14,635 MiB/rank at 250k — see §7 VRAM lesson).

Deeper background: docs/54 §9 P2c (steps 1–6 history + step 7 design brief),
docs/50 D-18 register entry.

**Already built and verified (do not redo):**
- KVarN steps 1–6 on main (`6b535c76`): pool layout, write path, read kernel,
  budget+guard, staged shadow (D-15), prefix tails (D-16). Battery T16/T17/T18
  green at 250k; unit `test_prefix_snapshot_roundtrip` bit-exact.
- **T19 is already in the battery** (`tools/smoke/test_serve_correctness.py`,
  `t19_long_prefill_no_collapse`): 80k-token prompt must sustain ≥450 tok/s
  average. It currently FAILS (takes hours). It is your primary gate — do not
  modify its threshold.
- A shadow-budget bump to 2.4 GiB was tried and **reverted** (startup OOM at
  the 250k config). The budget constant is back at 1 GiB.

**What you are doing:** §6 steps 1–3 below: dequant-to-half2 + tensor-core
flash fused prefill kernel (see §5.2), GQA-shared fused decode kernel,
closeout.

**Progress already on `wo/kvarn-d18` (same-day iterations, do not redo):**
must-pass T1–T18 green; original fused ~4 tok/s → Br=64 per-q-head ~12–44 tok/s
→ GQA-shared Br=8 compromise ~119 tok/s at 31.2k → scalar Q-rotation kernel
~47 tok/s (still far from the gate). **The wall is NOT dequant amortization —
it's that every candidate so far does attention with scalar FMA + reductions
instead of MMA.** beellama.cpp (MIT) ships a working native KVarN CUDA path:
dequant-to-half2 in smem then `mma_f16` flash, near-flash speed at 64k. §5.2 is
now that design; read its reference file before coding.

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; remote `github`). Start from main
  HEAD (post-revert commit) — check `git log --oneline -3` first.
- **Work protocol (MANDATORY):** ALL work in a worktree + branch — never edit
  the main tree directly.
  ```bash
  cd /home/intel/ninfer/repo
  git worktree add ~/ninfer/worktrees/wo-kvarn-d18 -b wo/kvarn-d18
  cd ~/ninfer/worktrees/wo-kvarn-d18
  cmake -S . -B build && cmake --build build -j 16   # own build dir, one-time full build
  ```
  Commit per step to `wo/kvarn-d18` and push. **Merging to main is done by the
  main-side agent/user** — do not merge or push to main yourself.
- Build (inside your worktree): `cmake --build build -j 16` (CUDA arch forced
  to `sm_120a`; 2× RTX 5060 Ti 16 GB).
- Unit tests: **use `/usr/bin/ctest`** (the `ctest` on PATH is a broken Python
  wrapper). From your `build/`: `/usr/bin/ctest -R "kvarn"`.
- **Env gotchas (hit by prior iteration, 2026-08-24):** CUDA must be **13.1**
  (`CUDACXX=.../cuda/bin/nvcc`; default cmake picks 12.9 and fails); build with
  `-DBUILD_TESTING=ON` for the GPU unit tests; a stale `ninfer-serve` holding
  ~14 GB/rank causes OOM on relaunch — check `pgrep -x ninfer-serve` and
  `nvidia-smi` BEFORE every launch.
- **Fast iteration loop (use before every full T19 run):** one live request
  with ~40k-token context crosses the wall once and shows the beyond-capacity
  rate in minutes (T19 at 80k burns tens of minutes per failed kernel). Gate a
  commit on: fast-loop rate within 2× of in-capacity, then run full T19.
- Battery: `python3 tools/smoke/test_serve_correctness.py --only T1,T2,T3,T5,T8,T10,T11,T16,T17,T18,T19`
  against a live server. T19 is the slow one (~80k tokens).
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.
- **Server launch (250k KVarN MTP — the only config that matters here):**
  ```bash
  setsid ./build/apps/ninfer-serve /home/intel/models/qwen3_8_27b.ninfer \
    --port 8091 --devices 0,1 --spec mtp --draft-tokens 3 \
    --kv-dtype kvarn_k4v2 --kv-capacity 250000 --max-context 250000 \
    >> ~/ninfer/logs/serve.log 2>&1 &
  ```
  Startup takes ~90 s; "listening on http://127.0.0.1:8091" = ready.

## 3. Architecture facts (verified — do not re-derive)

- Engine path is **TPEngine** (NOT ConcurrentExecutor — REPO.md §2a).
- Model geometry per rank: D=256, G=64 (= page size), kv_heads=2, q_heads=32
  → GQA group = 16 q-heads per kv-head. 16 text layers + 1 MTP layer.
- Packed storage per (layer, head, page): K codes `kKvarnKCodeBytes` = 256×32 B
  ([D][G/2] int4), V codes `kKvarnVCodeBytes` = 64×64 B ([G][D/4] int2); scale
  side table fp32 `[layers, heads, pages, 1152]`, field order in
  `kvarn_scale_at` (gqa_attention_kvarn.cuh). Dequant math = unpack + per-column
  scale/zp + one 256-channel FWHT butterfly (`kvarn_fwht_channel`) — reuse the
  existing device functions verbatim.
- Staged shadow: `KvarnStagedLayer` bf16 `[D, G, heads, staged_pages]` K+V per
  layer; budget constant `kKvarnStagedBudgetBytes` (1 GiB → 481 pages ≈ 30.7k
  tokens); per-page cost 2.125 MiB across 17 layers.
- **Dispatch** (text_context_impl.h ~L321 text, ~L361 MTP): `need_pages <=
  staged_pages` → BF16 flash over the staged shadow (`gqa_attention_cached`);
  else → fused KVarN kernel (`gqa_kvarn_staged_layer_view` NOT used; call is
  `detail::gqa_attention_kvarn_cached_launch` in src/ops/wrapper/gqa_attention.cpp
  ~L504). You change the ELSE branch's kernel, not the dispatch.
- **Reference architecture to mirror:** int8 prefill =
  `gqa_attention_prefill_fill_i8_page_kernel` (src/ops/kernel/gqa_attention_prefill_i8.cuh:148,
  one CTA per page tile, materializes the quantized page ONCE) + flash q_block
  kernel (Br tokens/CTA, online softmax). Your fused KVarN version replaces
  "read quantized page" with "dequant page in smem".
- Workspace tail: ≤63 uncommitted tokens live bf16 in `KvarnLayerWorkspace`
  (k_tile/v_tile + tail_count/tile_page); the current fused kernel already
  handles the tail — keep that behavior.
- **VRAM ceiling lesson (2026-08-24):** the 250k config sits at 14,635 MiB/rank
  with ~zero real headroom (a +1.5 GiB shadow bump → startup OOM). The fix must
  be **smem-only** — no new persistent device allocations, no budget changes.

## 4. Key call sites (anchors — verify line numbers before editing)

- `src/ops/kernel/gqa_attention_kvarn.cuh` (228 lines) — the slow fused kernel:
  grid (q_heads, tokens), smem `X[256×64]` fp32 + scores; dequant functions
  (`kvarn_dequant_k/v`, `kvarn_fwht_channel`, `kvarn_scale_at`) are reusable as-is.
- `src/ops/launcher/gqa_attention_kvarn.cu` (49 lines) — launch wrapper; add the
  new kernel's launch here (or a sibling file).
- `src/ops/wrapper/gqa_attention.cpp` ~L504 — `gqa_attention_kvarn_cached_launch`
  dispatch into the fused kernel.
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h` ~L321/~L361 — staged-vs-fused
  dispatch (text + MTP). Do not change the condition; only what the fused branch runs.
- `src/ops/kernel/gqa_attention_prefill_i8.cuh` — fill kernel (L148) + flash q_block
  kernel (~L256, `q_block = blockIdx.x`, `Br` tokens/block): the structural template.
- `tools/smoke/test_serve_correctness.py` — T19 (`t19_long_prefill_no_collapse`),
  T10 (determinism), T16/T17/T18 (KVarN gates).
- `include/ninfer/ops/kvarn_workspace.h` — geometry constants + staged shadow types.

## 5. Design decisions (FINAL — do not re-litigate)

1. **Prefill: q-block flash kernel, GQA-shared, Br=64.** Grid = (kv_heads,
   q_blocks); one CTA handles all 16 q-heads of its group × Br=64 query tokens
   (tune 32/128 only if smem allows). Both the ×T and ×16 redundancies must be
   gone — either alone leaves an order of magnitude on the table (measured:
   per-q-head CTAs ~4 tok/s; +Br=64 ~40; GQA-share needed for the rest).
2. **Dequant-to-half2 + tensor-core flash (PRIMARY — proven by beellama).**
   Per KV page tile: unpack raw codes → apply scales/zp (fp32) → warp-level
   FWHT butterfly → store `half2` K and V tiles in smem ONCE; then run standard
   flash attention with **MMA** (`mma_f16` for QKᵀ and PV — the same structure
   as our existing int8 prefill kernel `gqa_attention_prefill_i8_kernel`, which
   already does mma_s8 QK + mma_f16 PV). Dequant cost is paid once per tile
   and amortized over Br×(GQA group) queries. This is exactly what beellama.cpp
   (MIT, Anbeeld's llama.cpp fork — same KVarN concept, `kvarn2..8` + precision
   tail) ships as its native CUDA path: `ggml/src/ggml-cuda/fattn-mma-kvarn-impl.cuh`
   (`load_rotated_slice_record_warp` = unpack+scale, `inverse_wht_128_warp` =
   butterfly, then half2 flash tile + MMA). They run 64k prefill at near-flash
   speed — the design is validated. **READ THAT FILE BEFORE CODING.**
   - Smem: if a full [D=256][G=64] K+V half2 pair (64 KiB) + Q block doesn't
     fit sm_120a's per-block limit, slice D into two 128-column passes exactly
     like beellama (`GGML_CUDA_FATTN_KVARN_DIM=128`, adaptive tile, 64-column
     fallback).
   - GQA: reuse the dequanted tile across all 16 group q-heads (beellama does
     this via head-slice sign combination; a simpler sequential head loop over
     the same smem tile is acceptable if it fits occupancy).
3. **Q-rotation scoring — REJECTED as the main path** (was §5.2, 2026-08-24):
   score against raw int4 codes with `v = Rᵀq` precomputed per CTA. The algebra
   is correct (verified against `kvarn_dequant_k`: `k'[d,g] =
   R[d,:]·((u⊙s_col)+zp)·s_row[g]`; so `score(i,g) = s_row[g]·(Σ_c w_i,c·u[c,g]
   + c_i)` with `w = (Rᵀq)⊙s_col`, `c_i = v_i·zp`), but the implementation is
   scalar FMA + shuffle/CTA reductions — it cannot beat MMA, and beellama's
   public kernel proves the materialization path reaches near-flash speed. Keep
   the derivation for a future int8-TC variant; do not build on it now.
3. **V side (either option, your call — document which):**
   (a) simple: dequant V tile to bf16 in smem once per page, standard O += P·V
   (~10–15% overhead vs pure-flash); or (b) rotate P: `o = (P⊙s_rowv)·R·y` with
   y the raw V codes — +Br×D FWHT on P per page, no V smem tile. Start with (a);
   switch to (b) only if (a) misses T19.
4. **Precision:** dequant-to-half2 matches the staged BF16 path's value domain
   (bf16 tile + f16 MMA), so expect rel_l2 < 1e-3 vs a bf16-rounded fp32
   reference (the corrected isolation methodology from docs/54 step 3).
   Bit-exact cross-path parity is NOT required; T10/T18 only need the fused
   path itself to be deterministic.
5. **Smem budget check FIRST** (sm_120a): raw K codes 8 KiB + V tile ≤ 32 KiB
   (option a) or 8+8 KiB (option b) + scores/softmax state for Br×(heads if
   not folded). No [D][G] float tile anywhere. If it doesn't fit, split D in
   halves — do not grow to two full tiles resident.
6. **Tail handling unchanged:** ≤63 workspace tokens as a partial block after
   the packed pages, same as today.
7. **Dispatch condition unchanged** (`need_pages > staged_pages` → fused); only
   the kernel in that branch changes. Keep the old kernel compiled for unit
   tests / fallback.
8. **REJECTED: raising `kKvarnStagedBudgetBytes`** — OOM at 250k (measured,
   2026-08-24); revisit only with startup free-VRAM measurement + a config that
   leaves ≥512 MiB headroom. **REJECTED: int8 QK tensor cores for KVarN**
   (design decision from docs/54). **REJECTED: staging/shadow growth of any
   kind** — smem only.
9. **REJECTED: 2-pass "dequant chunk to global scratchpad, then standard
   flash" producer-consumer design** (proposed externally, 2026-08-24). A
   small scratchpad holds a handful of pages; flash cannot read the remaining
   packed pages as bf16 because they don't exist in bf16. Full materialization
   is 34 KiB/token/rank (80k prompt ≈ 2.7 GB/rank) — that is why KVarN exists;
   it does not fit at 250k. Chunked-KV with online-softmax state round-tripped
   through global memory adds ~O((T/Br)·(P/chunk)·Br·D) traffic (hundreds of GB
   at T19 scale). The only viable version is a grid-sync pipelined cooperative
   kernel — 3–4× the effort for the same end state as §5.2 dequant+MMA, which
   needs no new persistent memory (smem tiles only). Do not pursue.

## 6. Execution order (commit + test each step before the next)

**Testing standard (applies to every step):** tests must simulate REAL behavior,
not kernel-level interactions in isolation. A step touching any server-facing
path is NOT done on unit tests alone — spin up the server with the new code and
execute the real call sequence (chunked prefill past the capacity boundary, MTP
rounds, request lifecycle). Kernel/unit isolation is necessary, never sufficient.

### Step 1 — Prefill dequant+MMA fused kernel (the T19 gate)
Implement §5.2: per-page dequant-to-half2 in smem (unpack → scales/zp → warp
FWHT) + standard flash MMA (QKᵀ f16, PV f16), GQA-shared, Br=64 (or 128-column
slices if smem requires — beellama's `fattn-mma-kvarn-impl.cuh` is the
reference; our int8 prefill kernel shows the in-repo MMA idiom). Switch the
fused branch to it for prefill (decode may still use the old kernel this
step). Your current scalar Q-rotation kernel stays compiled as fallback.
**Do NOT optimize the scalar path further — pivot to MMA.** If you find a math
error in §5.2/§5.3, report it with the derivation; do not silently fall back
to dequant-in-smem scalar.
**Tests (must pass before moving on):**
- Unit: on identical data with history ≤ staged capacity, new fused prefill
  output vs staged-path output — bit-exact (or §5.3 bf16-equivalence) + a
  determinism check (two runs, same input → same output).
- **Live T19 GREEN:** 250k KVarN server, 80k-token prompt, ≥450 tok/s average.
  This is the gate; if it fails, you are not done with step 1.
- Must-pass subset green at 250k: T1,T2,T3,T5,T8,T10,T11,T16,T17,T18.

### Step 2 — Decode GQA-shared fused kernel
Implement §5.2; switch the fused branch's decode to it.
**Tests (must pass before moving on):**
- Live: a request with ~40k-token context (beyond capacity) decoding — measure
  decode tok/s from serve.log (`decode=Xtok/s`); must be within 2× of an
  in-capacity (~10k context) decode rate. Record both numbers.
- T10 determinism still green (greedy, long generation).
- Must-pass subset green at 250k (same list as step 1).

### Step 3 — Closeout
Full battery at 250k (fast subset + T19); MTP acceptance same-workload check
vs int8 (~50% expected, per docs/50 clarification — must be within ~5pp of an
int8 run of the same prompt); update docs/54 step 7 with measured numbers; close
D-18 in docs/50 with commit refs.

## 7. Constraints (non-negotiable)

- **Worktree only:** no edits in `/home/intel/ninfer/repo` outside your worktree, ever (§2).
- **Live end-to-end before "done":** a step touching a server-facing path is not
  complete until the real server runs it (§6 testing standard).
- **No damage:** never leave uncommitted or untested code; commit per step with a
  message naming the step; keep the tree buildable at every commit.
- **GPU/server:** one live server at a time; port 8091 serves an active
  conversation — do not kill/restart it without explicit user permission. Swap
  protocol: ask → `pkill -x ninfer-serve` (exact name; NEVER `pkill -f`) → wait
  for GPUs < 500 MiB → launch per §2 → run → restore.
- **No new persistent device allocations** — smem only. Startup VRAM at 250k
  must remain 14,635 MiB/rank (± a few MiB). If your design needs device memory,
  stop and report instead of shipping it.
- Do NOT touch the I8/BF16 KV paths, the staged-shadow path, or the prefix-tail
  (D-16) machinery. KVarN is opt-in via `--kv-dtype`; other modes must be
  bit-identical before/after your change.
- Check smem fit for sm_120 BEFORE writing the kernel (one [D][G] bf16 tile at a
  time + Q block + accumulators); if it doesn't fit, split D or G — do not grow
  to two full tiles resident.

## 8. Definition of done

1. Steps 1–3 committed to `wo/kvarn-d18` with passing tests at each step —
   including the live-path test for every server-facing step.
2. **Live proof:** a fresh 250k KVarN MTP server launch (command recorded) serves
   an 80k-token prompt at ≥450 tok/s average (T19 green) and a ~40k-context
   decode within 2× of in-capacity rate.
3. Must-pass subset + T16/T17/T18/T19 green at 250k; startup VRAM unchanged.
4. MTP acceptance same-workload vs int8 within ~5pp (measurement committed to
   results/ — measurement data is project data).
5. docs/54 step 7 marked DONE with numbers; D-18 CLOSED in docs/50 with commit
   refs. Report format: one paragraph per step + the key numbers table.
