# W1 PHASE-G DELTA desk — residual W/U convention+indexing deltas in prepare_wy_wu_simt.cuh

**Desk:** static-analysis (read-only, no GPU, no builds; one local float64
re-simulation of both kernels' algebra), Team Red W1, 2026-09-18, window 7.
**Inputs:** the post-fix lane tree (`prepare_wy_wu_simt.cuh` with the W1
hc_base/v_row fixes applied), the mma originals, the W7 cell
(`tools/v340l/dn_simt_chunk_cell.cu`), and the cell's observed numbers
(W 1.21/1.19, U 0.136/0.128, v_new 0.136/0.141, attn 0.136/0.156,
h_chunk 0.0/3.2e-6, state 3.3e-8/8.9e-7, handoff leg C 0.0).

## R2 rule — closest prior ledger and what this desk does differently

Closest prior ledger: **`docs/amd/W1_DN_WEDGE_ROOTCAUSE_desk.md`** (2026-09-18,
same lane) — it root-caused the W6 trio wedge from source alone, shipped the
three device-addressing fixes (chunk-index `hc_base` in
`state_passing_simt.cuh:127-128` + `output_simt.cuh:79-80`; v_row double-offset
removal in `prepare_wy_wu_simt.cuh`), and designed the W7 cell. Its GREEN-with-
caveats outcome (state/h_chunk/handoff pass; W/U/v_new/attn fail) is this
desk's input. That desk explicitly did NOT audit Phase G's *algebra* against
the mma WY phase beyond addressing. This desk does the different thing:
**element-by-element convention/indexing diff of the SIMT Phase-G pipeline
(g scan → beta/bg staging → T_inv construction → U/W products → stores)
against the mma WY phase, plus a float64 re-simulation of both kernels' exact
algorithms on the cell's exact LCG inputs** to decide which candidate deltas
the observed signature admits — and it ships the precise fix diffs.

## 0. Verdict (two independent deltas, both confirmed numerically)

1. **DELTA-1 (W-only, indexing): the W product reads K column `d2` instead of
   `d_off + d2`** — `prepare_wy_wu_simt.cuh:272-273` vs mma
   `prepare_wy_wu.cuh:716-717` (+ loader `:245-246`, base `:685-686`). Strips
   1-3 of every (chunk, head) write `T_inv @ (bg·K[:, 0:32])` into W columns
   32-127. Predicted relL2 = **1.201** (fp64, exact cell inputs: **1.210 /
   1.189**) vs cell **1.21 / 1.19**. Smoking gun: in the SIMT arm, W's four
   32-column blocks are EXACTLY equal to each other (sim: 0.0).
2. **DELTA-2 (U+W-capable, algebra): SIMT Phase E (block-Schur) computes
   `X[i][j] = M[i][j] + Σ_{k=j+1}^{i-1} M[i][k]·X[k][j]`, missing both
   unit-diagonal block factors** that the mma Schur applies
   (`X = (I+Y_ii)·(M[i][j]·(I+Y_jj) + Σ_{k=j+1}^{i-1} M[i][k]·X[k][j])`,
   `compute_off_diag`, `prepare_wy_wu.cuh:169-199`, waves `:648-671`). The SIMT
   T_inv is therefore NOT `(I−M)^{-1}` in any 16×16 off-block-diagonal cell.
   Predicted U relL2 = **0.136 / 0.128** (exact cell inputs: 0.1359 / 0.1275)
   vs cell 0.136 / 0.128. Its W-side contribution under the cell's strong-decay
   g is **4e-11** — invisible — which is why W's error is fully owned by
   DELTA-1 and U's fully by DELTA-2.

Both fixes together predict the whole cell GREEN (residuals ~1e-6, the declared
fp32-order-noise floor of the rel-L2 < 1e-2 gate).

## 1. Why the state/h_chunk agreement is NOT in contradiction (the paradox resolved)

The state recurrence both arms run (`state_passing_simt.cuh:145-190` vs
`state_passing.cuh` Phase A-Z) is
`h ← γ_C·h + Σ_t k_t·(v_new_t)·dec_t`, `dec_t = e^{g_C − g_t}`. On the cell's
g ∈ [−3, −0.5] per-token draw: `sum(dec²) = 1.01 ≈ dec_63²` (dec_62 = 0.097,
dec_61 = 0.013, dec_60 = 0.002) — **the state update reads only the last ~2
tokens of each chunk**. At those rows:

- `v_new = U − W·h` with `W[C−1] ~ bg_{C−1}·K` and `bg_s = β_s·e^{g_cumsum[s]}`
  is already dead: bg = (3.4e-2, 3.3e-3, 1.9e-4, 2.0e-5, 2e-6, ~0, …). So the
  W·h term is ~0 in BOTH arms at the rows the state can see — DELTA-1's W is
  invisible to the state regardless of its 120% size.
- `U[C−1] ≈ β·V + (dead band)`: T_inv is exponentially banded under this decay,
  and rows 62-63 have no live cross-16-block entries, so DELTA-2's U error
  (which lives in rows 16-19/32-35/48-51 and decays with t−s) does not reach
  them either.

`h_chunk[c=0]` is the `state_in` snapshot (zeros in both arms → the cell's
0.0 at T=64 is **vacuous**, no discriminating power), and `h_chunk[c=1]` =
h after chunk 0 inherits the same blindness (3.2e-6 = bf16-snapshot rounding
floor). Leg C (chained-vs-wide) compares SIMT against SIMT — both arms share
both deltas — so its 0.0 is also delta-blind. **The only cells that see the
Phase-G deltas are the W and U comparisons themselves.**

## 2. DELTA-1 — W product K column: `d2` where `d_off + d2` is required

SIMT (`prepare_wy_wu_simt.cuh`), Phase G:

```cuda
241    // === Phase G: U and W for this strip's 32 value columns ===
248    const __nv_bfloat16* v_stp = v_in + v_base + d_off;        // V: +d_off ✓
266        const __nv_bfloat162 vp = *reinterpret_cast<const __nv_bfloat162*>(
267            v_stp + static_cast<int64_t>(s) * v_st + d2);      // V col = d_off+d2 ✓
272        const float2 k0 = bf16x2_to_float2(
273            *reinterpret_cast<const __nv_bfloat162*>(&k_smem[s][d2]));   // ← K col = d2 ✗
279        const int64_t dst = out_base + static_cast<int64_t>(t) * out_row_st + d_off + d2;
280        store_vec(U + dst, ...);                                // store col = d_off+d2 ✓
281        store_vec(W + dst, ...);
```

`d_off = blockIdx.z * STRIP_COLS` (`:105`) is the strip's global column base.
`k_smem` holds the FULL 128 K columns (`:94, 115-122`), so the W product for
strip z sums `T_inv[t][s]·bg_s·K[s][d2]` (columns 0-31) but stores it at W
columns `d_off+d2` (32-63 / 64-95 / 96-127 for strips 1-3). **All four strips
write the same values into disjoint column ranges**: W[:, 32:64] ==
W[:, 64:96] == W[:, 96:128] == W[:, 0:32] (the only correct block, strip 0,
where d_off = 0 collapses the bug).

MMA ground truth (`prepare_wy_wu.cuh`): the W panel is loaded from GLOBAL K at
the true column,

```cuda
716        load_scaled_wu_panel<WU_PANEL_COLS, BLOCK_THREADS>(WU_view, k_in + k_wu_base, k_stride_t,
717                                                           bg_smem, panel_col, tid);
685    const int64_t k_wu_base =
686        cs * k_stride_t + static_cast<int64_t>(qk_map.qk_head(h_v)) * kStateDim;
245        const Bf16x4Pack packed = load_vec<Bf16x4Pack>(
246            input_row0 + (std::int64_t)row * input_row_stride + panel_col + col4);
```

i.e. `K[cs+row][qk_head(h_v), panel_col + col4]` — `panel_col` sweeps 0..127,
global column = panel_col+col4, then `W = T_inv @ (bg·K)` stored at
`W + out_base + row*output_row_stride + panel_col + warp_panel_col + col8`
(`:347-349`). Same bg convention both sides (`bg = β·e^{+g_cumsum}`: SIMT
`:237` `exp2_approx(g_smem[tid]*kLog2E)`; mma `:681` `expf(g_smem[tid])`), same
bg per-s (index = chunk-local token row in both) — **the bg exponent is NOT a
delta** (hypothesis ruled out, §5).

**Signature fit:** with independent K columns, E[||W_simt − W_mma||²/||W_mma||²]
= (3 blocks × 2 − 3)/4 → relL2 ≈ √1.5 = 1.2247 (random-draw mean measured
1.2009; exact cell inputs 1.210 / 1.189 vs cell 1.21 / 1.19). U untouched
(U relL2 with DELTA-2 alone fixed = 0.0). v_new tracks U because W·h ≈ 0 at the
rows that matter (§1) — the tiny T=128 excess (0.141 vs 0.128) is DELTA-1's W·h
leaking at chunk-1 rows 64-68, where bg is alive again (per-chunk cumsum reset)
but dec kills their state contribution. attn tracks v_new (attn = scale·(γ·q·h
+ A·v_new), both arms same A, h_chunk equal to 3e-6). State: see §1.

**Fix (lane owner applies):**

```diff
--- src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu_simt.cuh
@@ Phase G, W product (lines 272-273)
-            const float2 k0 = bf16x2_to_float2(
-                *reinterpret_cast<const __nv_bfloat162*>(&k_smem[s][d2]));
+            const float2 k0 = bf16x2_to_float2(
+                *reinterpret_cast<const __nv_bfloat162*>(&k_smem[s][d_off + d2]));
```

(`d_off + d2 ≤ 96+30 = 126 < K_ROW_STRIDE(130)` — in bounds; the `(s·130+c)`
mod-32 bank spread is unchanged by a constant column shift. No other line
changes: the U product's V read already carries `d_off` via `v_stp`.)

## 3. DELTA-2 — Phase E block-Schur missing the unit-diagonal block factors

Let M be the strict-lower M of the header math (`M[t][s] = −β_t·e^{g_t−g_s}·(k_t·k_s)`,
identical both sides: SIMT Phase C `:173-180`, mma `:580-595`), partitioned into
4×4 16×16 blocks. After the (identical, verified) Phase D / `solve_diag_block`
in-place diagonal inversions (SIMT `:186-200`, mma `:216-232`, `:624-637`), the
diagonal blocks hold `Y_jj = (I−M_jj)^{-1} − I`. The true T_inv = (I−M)^{-1}
off-diagonal block is

```
X[i][j] = (I + Y_ii) · ( M[i][j]·(I + Y_jj) + Σ_{k=j+1}^{i-1} M[i][k]·X[k][j] )
```

The mma Schur implements exactly this (`compute_off_diag`, `:169-199`: the k=j
iteration adds `A_reg[j] @ M_view[j][j] = M[i][j]·Y_jj`, then the `k == MY_J`
correction adds raw `M[i][j]`; then `prod = M_view[i][i] @ sum = Y_ii·sum` and
`out = prod + sum`). Float64 check: mma transcription vs `np.linalg.inv(I−M)` =
**7.7e-18**. The SIMT Phase E instead computes

```cuda
213    {
214        const int r = tid >> 4;
215        const int c = tid & 15;
216        for (int i = 1; i < 4; ++i) {
217            const int i0 = i * 16;
218            for (int j = 0; j < i; ++j) {
219                const int j0 = j * 16;
220                float acc    = m_smem[i0 + r][j0 + c];          // ← raw M[i][j] only
221                for (int k = j + 1; k < i; ++k) {
222                    const int k0 = k * 16;
223 #pragma unroll
224                    for (int m = 0; m < 16; ++m) {
225                        acc += m_smem[i0 + r][k0 + m] * m_smem[k0 + m][j0 + c];
226                    }
227                }
228                m_smem[i0 + r][j0 + c] = acc;                   // ← no (I+Y_ii)·(…)·(I+Y_jj)
229                __syncthreads();
230            }
231        }
232    }
```

i.e. `X[i][j] = M[i][j] + Σ_{k>j} M[i][k]·X[k][j]` — the header's own comment
(`:204`) names the wrong formula. Minimal counterexample (i=1, j=0): true
`X_10 = (I+Y_11)·M_10·(I+Y_00)`, SIMT emits raw `M_10`. With the cell's inputs
the Y blocks are O(0.3-2), so every 16×16 off-block cell of T_inv is wrong at
O(1) relative. Effect on U: rows near each block boundary (t = 16-19, 32-35,
48-51, decaying over ~4-6 rows) carry per-row errors of tens of percent →
aggregate **U relL2 = 0.136 / 0.128, exactly the observed numbers**. Effect on
W: the same wrong T_inv cells multiply `bg_s`, which is ≤ 2e-5 beyond s ≈ 4 —
so the W-side error is **4e-11 (invisible)**, consistent with W's error being
entirely DELTA-1's. Effect on state/h_chunk/attn: none beyond 1e-6 (§1).
Leg C: delta-blind (SIMT vs SIMT).

**Fix (lane owner applies)** — makes Phase E compute the mma formula in the
per-thread-cell idiom. Needs one 16×16 fp32 scratch (+1 KiB static smem:
34.0 → 35.0 KiB, still 1 CTA/CU — update the header's occupancy comment):

```diff
--- src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu_simt.cuh
@@ smem block (after line 98, bg_smem)
     __shared__ float bg_smem[BT];
+    __shared__ float schur_smem[16][16];   // Phase E pass-1 staging (S block)
@@ Phase E (lines 213-232), replace the pair body
     {
         const int r = tid >> 4;
         const int c = tid & 15;
         for (int i = 1; i < 4; ++i) {
             const int i0 = i * 16;
             for (int j = 0; j < i; ++j) {
                 const int j0 = j * 16;
-                float acc    = m_smem[i0 + r][j0 + c];
+                // pass 1: S = M[i][j]·Y_jj + Σ_{k=j+1}^{i-1} M[i][k]·X[k][j] + M[i][j]
+                // (all reads raw: block-row i is overwritten only after pass 2)
+                float acc = 0.0f;
+#pragma unroll
+                for (int m = 0; m < 16; ++m) {
+                    acc += m_smem[i0 + r][j0 + m] * m_smem[j0 + m][j0 + c];
+                }
                 for (int k = j + 1; k < i; ++k) {
                     const int k0 = k * 16;
 #pragma unroll
                     for (int m = 0; m < 16; ++m) {
                         acc += m_smem[i0 + r][k0 + m] * m_smem[k0 + m][j0 + c];
                     }
                 }
-                m_smem[i0 + r][j0 + c] = acc;
+                acc += m_smem[i0 + r][j0 + c];
+                schur_smem[r][c] = acc;
                 __syncthreads();
+                // pass 2: X[i][j] = Y_ii·S + S   (mma compute_off_diag semantics)
+                float acc2 = schur_smem[r][c];
+#pragma unroll
+                for (int m = 0; m < 16; ++m) {
+                    acc2 += m_smem[i0 + r][i0 + m] * schur_smem[m][c];
+                }
+                m_smem[i0 + r][j0 + c] = acc2;
+                __syncthreads();
             }
         }
     }
```

Ordering safety (same invariants the current code already relies on, now
stronger because the write moved after pass 2): pass 1 reads raw `M[i][·]`
(block-row i written only by its own pairs, at pair end), stable `Y_jj`
(diagonal blocks, Phase D), and `X[k][j]` from earlier i-waves; pass 2 reads
`Y_ii` + scratch. j-ascending per row preserved. Float64 check: this fixed
variant equals the mma transcription equals `(I−M)^{-1}` to 1e-17.

## 4. Store-layout check (done FIRST per briefing; ruled out as a delta)

Element-by-element, SIMT `:279` vs mma `:347-349`:
base `out_base = cs·H_v·128 + h_v·128` identical (`:245` / `:683-684`); token
stride `H_v·kStateDim` identical (`:246` / `:684`); column `d_off + d2` vs
`panel_col + warp_panel_col + col8` — both sweep the same global 0..127; bf16
rounding boundary identical (single `__floats2bfloat162_rn`; the mma's smem
stage is a lossless bf16 copy). **No permutation.** This is also proven by the
re-simulation: with the two real deltas in, the exact cell numbers reproduce
with NO store difference — a store permutation cannot produce U=0.136/W=1.21.
Likewise the U/W token-stride (`cs·H_v·128` = token offset) is consistent with
every consumer: `state_passing_simt.cuh:109,158,200` and mma
`state_passing.cuh`/`output.cuh` all read `(cs+t)·H_v·kStateDim + h_v·kStateDim + d`
(the `workspace_layout` `{kStateDim, value_heads, tokens}` naming in
`launch.h:38-41` is dim-outermost nomenclature, but both producers and all
consumers use the token-major formula — pre-existing, symmetric, not a
simt-vs-mma delta; flag for a later naming pass only).

## 5. Other candidates examined and RULED OUT (cell numbers + lines)

1. **bg exponent convention** (`exp(g)` vs `exp(−g)` vs `exp(g_last−g)`):
   identical `β·e^{+g_cumsum[s]}` both sides (SIMT `:237`, mma `:681`); a sign
   flip would corrupt W's dominant rows 0-4 (within block 0, delta-blind to
   both deltas) and the state with them — the observed state 1e-7 rules it out.
2. **ti_row construction / upper-triangle masking / `if (x == 0.0f) continue`**
   (briefing item 4): both sides hold EXACT 0 above the diagonal (SIMT writes
   0 at `:163-165` and never touches upper; mma zero-fills `:601-606` and
   select-writes 0 `:580-595`); skipping `x==0` adds exactly 0.0f — no numeric
   delta; the s=t diagonal enters U with `T_inv[t][t]=1` both sides (+1 at
   `:236` / `:675`). The re-simulation reproduces U=0.136 with the skip
   unmodeled — ruled out.
3. **g scan / beta staging** (briefing item 5): identical Hillis-Steele
   inclusive scans, same publish layout (`(cs+t)·H_v + h_v`), strip-0-only
   publish is deterministic (SIMT `:143-146`, mma `:441-442`); the cell's
   g_cumsum-consumers already PASS. Ruled out.
4. **exp2_approx vs expf** (C441 class, declared in the SIMT header `:45-47`):
   1-2 ulp; cannot produce 0.13/1.2. Ruled out by magnitude.
5. **K/V panel staging incl. `qk_head` mapping**: identical
   `cs·k_st + qk_head(h_v)·128 + row·k_st + col` (SIMT `:118-120`, mma `:453,
   496`); V read post-fix identical (`:267` vs `:533-534, 699-702`). Ruled out.
6. **`beta_smem` per-s indexing**: identical (`:123` / `:412-415`). Ruled out.
7. **W/U store permutation**: §4 — ruled out.

**Ranking by explanatory power:** DELTA-1 owns W (1.201 vs 1.21/1.19 — exact),
DELTA-2 owns U (0.136/0.128 — exact); they are independent (different code
paths, each tensor's error fully explained by exactly one delta, per the
fix-only decomposition: d_off-fix-only U = 0.0; PhaseE-only W = 4e-11). No
third delta is needed: v_new/attn/state/h_chunk/leg-C numbers all fall out as
downstream consequences (§1).

## 6. Predicted post-fix cell (for the GREEN row)

With both diffs applied, float64 + fp32-order-noise model predicts:
W/U/v_new ≈ 1e-6, attn ≈ 1e-6, state ≈ 1e-7 (unchanged), h_chunk ≈ 3e-6
(bf16 snapshot floor, unchanged), leg C 0.0, poison clean → ALL PASS under the
1e-2 gate. RED capture for closure law: the current cell run (W 1.21/1.19,
U 0.136/0.128) at the pre-fix shas is the red row; re-run the same cell at the
post-fix shas for the green row. Add to the permanent per-window battery as-is.

## 7. Confidence and falsifier

**Confidence: HIGH (near-certain) on both deltas** — each reproduces its cell
number to 3-4 digits on the cell's exact LCG inputs in float64 (W 1.210/1.189
vs 1.21/1.19; U 0.1359/0.1275 vs 0.136/0.128; state 8.3e-8/3.7e-7 vs
3.3e-8/8.9e-7), and each is a line-visible indexing/algebra defect against the
mma ground truth. MEDIUM-HIGH on the exact post-fix residuals (fp32 tree-order
noise class, far under the gate).

**Single cell observation that would falsify the top hypothesis (DELTA-1):**
dump the SIMT arm's W tensor and compare its 32-column blocks — the hypothesis
predicts `W[:, 32:64] == W[:, 64:96] == W[:, 96:128] == W[:, 0:32]` EXACTLY
(bf16-equal, sim: 0.0). Any difference between blocks 0 and 1 at unchanged
inputs kills DELTA-1 and re-opens the W hunt. (Twin falsifier for DELTA-2:
apply only the §2 diff and re-run — U must stay at ≈0.136 while W drops to
≈1.2; if U drops too, DELTA-2 was not the U delta.)
