> **Landed on main 2026-08-26 as docs/75** (QA review + numbering map: docs/77).
> Original: `wo/kvarn-layout-verify` @ fab93b97, file `docs/73_kvarn_staged_layout_zero_copy_findings.md` (content as of landing).
> Status as of landing: **VERIFIED** — CPU-only test passes (ninfer_kvarn_layout_test; ran clean 2026-08-26 at QA).
> In-text references to docs/69–74 use the BRANCH numbering: 69=full-matrix(docs/70), 70=prefix-reuse(docs/72), 71=direct-read(docs/73), 72=attn-work-order(docs/74), 73=staged-layout(docs/75), 74=attn-scope(docs/76).

---

# 75 — KVarN Staged-Shadow / Beyond-Wall-Temp Layout: Verified

**Status:** VERIFIED — arithmetic proof + CPU-only regression test.
**Branch:** `wo/kvarn-layout-verify`. **Owner:** supports docs/66 §6 (the D-18
2-pass prefill / T19 gate). **No kernel change** — this is a validation + the
resolution of the open layout question below.

---

## 1. The question

The D-18 2-pass prefill materializes every tile once into a paged BF16 temp,
then reuses the proven BF16 FA2 flash (`gqa_attention_kvarn_flash.cuh`). The
materialize kernel's below-wall branch was written/described as a **transpose**
(staged → `[G][D]`). The open question: is that branch actually a transpose, or
a straight copy — and if a copy, can it be eliminated entirely (zero-copy)?

## 2. Layout facts (each verified against source)

| Tensor | Shape | Layout | Source |
|---|---|---|---|
| Staged shadow `k_pages`/`v_pages` | `{kD, kG, kv_heads, stage_pages}` = `[D,G,H,P]` | **d-fastest** | `src/ops/kvarn/kvarn_workspace.cpp:111-113` |
| Beyond-wall temp `k_temp`/`v_temp` | `{kKvarnAttnD, kKvarnAttnG, kv_heads, n_tiles}` = `[D,G,H,P]` | **d-fastest** | `src/ops/launcher/gqa_attention_kvarn.cu` |
| Flash read addressing | `paged_kv_element_offset<D, KVHeads>` | `D·G·(h + H·p) + D·g + d` | `src/ops/kernel/paged_kv_address.cuh`, `gqa_attention_prefill_bf16.cuh:72` |
| Materialize below-wall write | `kd[g·D+d0] = sk[d0 + D·g]` | `[D,G,H,P]` → `[D,G,H,P]` | `gqa_attention_kvarn_flash.cuh:63-72` |

All three share the **same** `[D,G,H,P]` d-fastest mapping, with the invariant
`kKvarnAttnG == kPagedKVPageSize` (both 64) required for the flash's page
addressing to line up with the staged page extent.

## 3. Conclusion

1. **The materialize below-wall "transpose" is a straight element copy, not a
   transpose.** Both the staged shadow and the temp are `[D,G,H,P]` d-fastest,
   so `kd[g·D+d0] = sk[d0 + D·g]` moves each element to *the same*
   `(d0, g, h, kb)` slot. The `src/ops/kernel/gqa_attention_kvarn_flash.cuh`
   header comment that says "transpose" is a misnomer (left over from an earlier
   `[G][D]` design); the code is correct.

2. **The flash reader addresses the staged shadow directly.** For a page-relative
   key `g` and head `h` on physical page `p`, `paged_kv_element_offset<D,H>` and
   the staged `[D,G,H,P]` offset produce the *same linear element*. So removing
   the copy for below-wall pages (reading the shadow in place) changes nothing
   about what the flash sees.

3. **Zero-copy aliasing is feasible** with one required condition: the shadow and
   the overflow region must live in a **single block-table-addressable
   allocation** (one contiguous `[D,G,H,P_total]`, where `P_total =
   staged_pages + overflow_pages`). Then a single identity `block_table` maps
   logical page `kb` to physical page `kb`, and pages `kb < staged_pages` hit the
   shadow directly while `kb >= staged_pages` hit the pre-materialized overflow
   region — all through the one offset formula. No separate per-call temp tensor,
   no copy.

## 4. How this was verified

`tests/test_kvarn_layout.cpp` (**host-only**, no CUDA headers, no ninfer libs)
re-declares the source-of-truth constants and re-expresses the two offset
formulas exactly as the production headers write them, then asserts:

- **T1 – identity copy:** simulate the materialize below-wall loop and compare
  `temp` vs `staged` for all `(d,g,h,p)` by raw bit pattern; assert identity and
  assert it is *not* a transpose (`temp[d,g] != staged[g,d]`).
- **T2 – flash equivalence:** `paged_kv_element_offset<D,H>` vs staged offset
  agree for every sampled `(page, head, key, d)` under an identity `block_table`.
- **T3 – zero-copy aliasing:** below-wall and overflow reads land in one
  contiguous allocation, and the overflow begins immediately after the shadow
  page.
- **T4 – 16B-vector copy coverage:** the materialize below-wall copy's thread
  decomposition (`g = c>>5, d0 = 8*(c&31)`, `Tiles = G*(D/8)`) touches every
  `(d,g)` exactly once (no overlap / no stale byte).
- **T5 – tail tile branch:** the in-flight tail is K = `[G][D][H]` (g fastest) and
  V = `[D][G][H]` (d fastest); the kernel transposes K, straight-copies V, and
  zero-fills keys `>= tail_count` — all verified.

Run standalone (or via `ctest -R kvarn_layout`):
```bash
g++ -std=c++17 -O2 -o test_kvarn_layout tests/test_kvarn_layout.cpp && ./test_kvarn_layout
```
Registered in `tests/CMakeLists.txt` as `ninfer_kvarn_layout_test`.

## 5. Consequences / follow-ups

- **Airtight:** the layout reasoning behind docs/66 §6 (the D-18 2-pass prefill)
  and §5.1's "staged shadow is the correct architecture" now has a committed regression
  test. Any future change to the staged layout, `kKvarnAttnG`, or the paged
  offset formula that silently diverges will fail `ninfer_kvarn_layout_test`.
- **Not implemented here:** actually removing the below-wall copy and routing the
  flash straight at the shadow is a separate, larger change (the shadow is a
  per-sequence persistent allocation, the temp is per-call; unifying them touches
  the workspace + dispatch). This doc + test de-risk it; it is intentionally out
  of scope for this branch (validation only).
- **Invariant to guard going forward:** `kKvarnAttnG == kPagedKVPageSize`.
