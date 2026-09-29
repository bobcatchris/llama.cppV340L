# GQA_SPLITK_NOTES — FIX-C: prompt SIMT flash split-K over key tiles (PLOG-048 #3)

**Desk:** CODE desk, `amd/tp4-cure`, 2026-09-17. Compile-only session (no GPU runs; :8100
undisturbed). Spec: `docs/amd/PREFILL_BODY_2026-09-17.md` §4 S3 + §6 (the doc's "FIX-B"
design = this desk's FIX-C by the staffing-tree numbering: FIX-A dn, FIX-B gate, FIX-C gqa).
**Baseline (measured, P2 row `PREFILL_BODY_row.txt`):** `gqa` = 54.7-57.4 ms/chunk at T=128,
TP4, plen-2016 probe — prefill body owner #3.

## 1. The three named defects and what was done

Kernel: `src/ops/kernel/gqa_attention_prefill_bf16_gfx906.cuh`
(`gqa_attention_prefill_simt_bf16_kernel`), reached from the prompt routes of `ops::gqa_attention`
(fused-append — the TP4 prefill call, `text_context_impl.h` attn_mix_tp) and
`gqa_attention_cached`. Base grid at the served shape: `(width/32, q_heads) = (4, 6)` = 24 CTAs,
128 threads, serial 32-key-tile walk over the window (~34 tiles at plen-1996), and — kv_heads
local = 1 — all 6 q-head CTAs of the group restage identical K/V tiles.

| defect | cure |
|---|---|
| (a) 24 CTAs, 56-CU box 43% empty | split-K: grid `(width/Br, q_heads, splits)` — 48-96 CTAs at the served shape |
| (b) serial 32-key tile loop (~34 tiles) | each split owns a DISJOINT key-tile range `[s*tps, min((s+1)*tps, n_tiles))`, `tps = ceil(n_tiles/splits)` — ~4x shorter walk at auto S=4 |
| (c) 6x duplicate K/V staging per kv-group | grid-level L2 reuse (documented call — see §3; true LDS cooperation is register/LDS-infeasible on gfx900, math below) |

## 2. The split scheme (two-pass, existing flash bookkeeping)

The kernel's flash bookkeeping was read first: the SmallT route already owns the exact
acc/m/l workspace this desk needed — `allocate_small_t_workspace`
(`src/ops/wrapper/gqa_attention.cpp`) slabs `acc bf16 {D, H, width, splits}`,
`m/l fp32 {H, width, splits}`, layouts `gqa_partial_acc_index` / `gqa_partial_stat_index`
(`src/ops/kernel/gqa_attention_decode.cuh`). So: **workspace-flags-free two-pass reusing the
existing slabs** — no new arena primitives, no flags, no atomics; pass 1 and pass 2 are two
launches on the same stream (ordered, race-free).

- **Pass 1 — `gqa_attention_prefill_simt_bf16_splitk_partial_kernel`** (new, same LDS tile
  pattern as the base kernel, tile loop bounds = the split's disjoint range): per (q_head,
  row-block, split) CTA, flash-attends its key-tile range with the base kernel's phase
  A/B/C arithmetic verbatim (exp2_approx basis, Log2E scaling), then publishes unnormalized
  `m_s/l_s` and bf16 `acc_s` per row. `tps` is computed ON DEVICE from the ACTUAL window
  (`positions[0] + valid_tokens`) — a host-side cap-window partition would go blind on short
  early-chunk windows (cap ≫ actual ⇒ all keys land in split 0, zero speedup). The last
  split absorbs any window overflow; rows whose split range is causally empty publish
  `m=-inf, l=0` (the slabs are uninitialized arena memory — silence would leak garbage into
  the combine). Split 0 owns the base kernel's beyond-valid-tokens zero duty (both the
  early-exit class and the epilogue class, one range).
- **Pass 2 — `gqa_attention_prefill_simt_bf16_splitk_reduce_kernel`**: one CTA per
  (q_head, token), one thread per dim (D=256), combine convention matches the shipped
  small-T reducer exactly: `w_s = expf(m_s - M)`, `head_l = Σ l_s·w_s`,
  `out = Σ acc_s·w_s / head_l`, `l==0` slots skipped. With the FIX-C partials storing acc_s
  pre-normalized (`acc_s/l_s` — arithmetic-neutral, keeps bf16 partials in [~-1,1] for
  better resolution on long windows), the reduce multiplies by `l_s·w_s`. Active splits =
  `min(S, ceil(n_tiles_actual/tps))` — the same predicate the partial kernel used for
  `kb0 < kb1`, so partial and reduce can never disagree.
- Partials allocated per prompt call in the caller's arena scope (freed on return); shapes
  at the probe: `256·6·128·4·2 B ≈ 1.57 MB` acc + 2·`6·128·4·4 B ≈ 25 KB` m/l — sized from
  shape math at allocation time, never a refusal constant (VRAM law).

**Numerics (documented + gated):** flash softmax is order-changed vs the single-pass kernel
(per-split online max/sum, cross-split combine with `exp(m_s-M)` weights). This is the
gated delta — rel-L2 < 1e-2 old-vs-new in the cell (§6). Same combine the ChunkedSmallT
route has shipped since docs/117.

## 3. Occupancy math and the defect-(c) call

**Occupancy.** Each CTA holds ~48.4 KiB static LDS (3 tiles × 32×258 × bf16/half) + 128
threads. gfx906: 64 KiB LDS/CU ⇒ **1 CTA per CU** — grid-level occupancy IS the CTA count.
Base grid 24 CTAs on 56 CUs = 43% occupied. Auto split factor targets ~96 CTAs
(`wanted = ceil(96 / q_blocks·q_heads)`, clamp [2, min(8, q_blocks)]):

- S=2 → 48 CTAs = one full wave (mission floor 48+), 2x serial cut;
- S=4 (auto at width=128, q=6) → 96 CTAs ≈ 1.71 waves, **~4x serial cut** —
  ~8-16 tiles per CTA at the probe windows (128-1996).

Auto caps S at `q_blocks` (window ≥ width always ⇒ every split owns ≥1 tile at ANY chunk);
explicit factors pass through (over-split is correct: tail splits publish empty partials,
reduce skips them). Workspace scales with S and is accounted (§5).

**Defect (c) — the documented call: GRID-LEVEL REUSE (L2), not LDS cooperation.**
True cross-CTA LDS sharing is infeasible on gfx900 for this shape: one CTA holding the whole
group's row-state = 192 rows (6 heads × 32) × 256 dims fp32 accumulators = **196 KiB against
64 KiB LDS/CU and 256 regs/lane** (48 live row-states per warp = 384 acc registers); the
alternative tiles-outer/head-inner schedule needs those 48 row-states live per warp
simultaneously. Same verdict as llama.cpp's gfx906 flash: K/V reuse across GQA heads lives
in the cache hierarchy, not LDS. Mechanism here: `splits` is the OUTERMOST grid axis, so all
24 CTAs of one split (the whole kv-group at kv_heads=1) are rasterization-adjacent and sweep
the SAME key tiles concurrently; the first HBM fetch of a tile warms L2 (probe window's K+V
working set ≈ 63 tiles × 32 KB ≈ 2 MB vs 4 MB L2) and the group's other CTAs hit L2.

**Traffic saved (probe geometry, per chunk, per rank):** K/V tile staging before = 24 CTAs ×
34 tiles × 32 KB ≈ 26 MB of tile fills, historically 6 of every 6 same-group fills racing to
HBM; after: same fill count but the HBM leg drops ~6x → ~1x per (tile, kv_group) (~4.3 MB
HBM, rest L2 hits), AND the wall cut is the dominant term anyway — the kernel is
compute/latency-bound at 0.2% of peak, not bandwidth-bound (doc §3).

## 4. Gate matrix

`NINFER_GQA_SPLITK` — value-parsed, prefill PROMPT path only; implemented as pure host
policy in `src/ops/launcher/gqa_prompt_splitk.h`, shared by launcher and wrapper so launch
and arena-capacity accounting cannot disagree.

| env | behavior |
|---|---|
| unset / empty / `"0"` / unparsable / out of [0,16] | **OFF — byte-identical**: armed call sites take the pre-FIX-C dispatch, Prompt-route capacity stays exactly 0, capacity scan bound unchanged, no new kernels launched |
| `"1"` | AUTO: `S = clamp(ceil(96/(qb·H)), 2, min(8, qb))` — served shape ⇒ S=4, 96 CTAs |
| `"2"`..`"16"` | explicit split factor (over-split windows are correct; empty splits are the operator's knob) |

First-seen trace (once per process, stderr):
`[GQA] splitk gate NINFER_GQA_SPLITK armed (prefill prompt path only)` then per first launch
`[GQA] splitk=on splits=4 ctas=4x6x4=96 (base 4x6=24) width=128 geom=q6/kv1 br=32`.

Coverage: both prompt arms gated — fused-append (`gqa_attention` →
`gqa_attention_prompt_splitk_launch`, the arm the TP4 prefill runs) and cached
(`gqa_attention_cached` → `gqa_attention_prompt_attention_splitk_launch`). SmallT /
ChunkedSmallT routes untouched. KVarN/dtype special routes untouched. CUDA (!__HIP__) arm of
the new entry points throws LOUD if ever armed there (house pattern). Decode path untouched.

## 5. Workspace / VRAM-law receipt

`gqa_attention_workspace_capacity_bytes` Prompt branch: OFF → returns exactly 0 (unchanged
byte-for-byte); ON → the same slabs the SmallT capacity pass sizes, at prompt width, batch
plane 0 (the prompt kernel serves the first sequence's plane only — split-K preserves that
semantic exactly), and the width scan extends to max_width only when the env is armed.
Allocation-time shape math only; nothing in this fix can refuse a launch.

## 6. Cell receipts

**Cell:** `tools/v340l/gqa_splitk_cell.cu` (house standalone pattern per
`tools/i4_oracle/i4_oracle.cu`). Shape per mission: cell-local `GqaGeometry<2,1,1>` (1
kv-group, 2 q-heads), 64-key mission window + 3 more shapes (odd window 49 = ragged tile,
single-tile window over-split 4x, two-page window 96), splits {1,2,4}.

- GREEN arms: rel-L2(splitk, base kernel) < 1e-2 (the flash-order gate); rel-L2(base, fp64
  oracle) < 1e-2; rel-L2(splitk, fp64 oracle) < 1e-2.
- RED arm (both directions): one V row perturbed +0.75 in the split-K cache — comparator
  MUST exceed 1e-2 (a gate that cannot go red is not a gate).
- Host-only policy asserts: auto(6,128)=4, auto(24,128)=2, auto(6,17)=1, env parse "1".
- Every HIP call checked; cell carries zero warnings of its own.

| receipt | result |
|---|---|
| compile (gfx900, `-fsyntax-only`, c++20) | **RC=0, zero warnings** @ 209436171 |
| run (needs gfx900 device — NO-GPU desk) | **PENDING GPU window** — runbook §7 step 6 |
| RED/GREEN closure law | the cell is authored to yield both rows in one run; rows get banked with kernel-sha named when the GPU owner fires it |

## 7. GPU-OWNER RUNBOOK (per docs/amd/BOOT_LAUNCH_RUNBOOK.md; grant + manifest row first)

1. **Rebuild:** `make -C build-hip-amd -j20 ninfer_hip_host ninfer-serve` — expect RC=0,
   warning classes = pre-existing only (this desk's receipt: 26 unused-result + 1
   cuda-compat, zero on added lines, @ 5e2d28c85).
2. **BANK-BEFORE-RELINK (runbook §4):** `cp -p` the serving binary to
   `/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin` (filename-is-stamp) + commit RECORD
   row. Pinned boots load from the bank, never a lane build path.
3. **BOOT_BATTERY** on the boot pair (residence probe first; grant-ack in arguments; cool
   window + clocks sideband per the P2 flow — `results/amd/coherence/BOOT_BATTERY_*.row`
   grammar).
4. **Cell (device window, before the probe):**
   `/opt/rocm-6.2.0/lib/llvm/bin/clang++ -x hip --offload-arch=gfx900 -O2 -std=c++20 -I src/common/hip_shim -I src -I include tools/v340l/gqa_splitk_cell.cu -o /tmp/gqa_splitk_cell && /tmp/gqa_splitk_cell`
   — expect `GQA-SPLITK CELL: ALL PASS` RC=0; bank the output as the GREEN row, the RED arm
   prints its deliberate FAIL line inline. Name kernel shas in the row (RED/GREEN law).
5. **Probe:** manifest-row boot of the banked binary with
   `NINFER_GQA_SPLITK=1 NINFER_PREFILL_OPTRACE=2`, known-good nvfp4 line, ONE plen-1996
   conc=1 first-request (same grammar as the P2 row).
6. **Readout / parity gate:**
   - `[GQA] splitk=on splits=4 ctas=4x6x4=96 (base 4x6=24) width=128 geom=q6/kv1 br=32`
     in the log (first chunk);
   - **`gqa` 56 → < 20 ms** expected (P2 `PREFILL_BODY_row.txt` mean_ms/chunk grammar);
     S=2 A/B (~30 ms expected) is one env re-boot away if S=4 disappoints;
   - text content sane (`finish=stop`, coherent continuation — greedy-text parity gate);
   - output-token fingerprint family 202/0.985/2.97 unchanged class;
   - body sum within ~5% of columns; default-OFF leg (one probe WITHOUT the env) must
     reproduce the pre-FIX-C `gqa` numbers byte-class — byte-identical claim verified.
7. If `gqa` does not drop or parity trips: bank the row, DO NOT iterate on the card —
   desk re-engages with the numbers.

## 8. Provenance

- Implementation: 5e2d28c85 (kernels + policy header + launcher + wrapper, 5 files).
- Cell: 209436171 (`tools/v340l/gqa_splitk_cell.cu`).
- Baseline: `results/amd/coherence/PREFILL_BODY_row.txt` (P2, bin 74219298c4f3d4a4,
  coordinator-run 2026-09-18 ~02:4x): `gqa` 54.7-57.4 ms/chunk.
- Compile receipts this desk: `make -C build-hip-amd -j20 ninfer_hip_host` RC=0 and
  `ninfer-serve` RC=0; warning census 26x -Wunused-result + 1x -Wcuda-compat
  (`prefill_bf16.cuh:25` fill-kernel inline) — all pre-existing classes, none on added
  lines. Re-verified in isolation after the tree turned red: all three FIX-C TUs
  (launcher gqa_attention_prefill.cu.o, wrapper gqa_attention.cpp.o, launcher
  gqa_attention_decode.cu.o) RC=0 ZERO warnings — a same-session full-target failure was
  100% the FIX-A desk's IN-FLIGHT uncommitted `gated_delta_net/chunked/prepare_wy_wu.cu`
  edit (forbidden path for this desk; attribution guard, not a FIX-C defect).
  No GPU touched; :8100 undisturbed; no forbidden paths opened
  (gated_delta_net/, bf16_gdn_gating_proj*, coherence rows untouched — this doc is an ADD).
