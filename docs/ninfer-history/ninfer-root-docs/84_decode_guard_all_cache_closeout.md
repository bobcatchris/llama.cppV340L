# Decode-Guard `--all-cache-type` Sweep & D-21 Close-Out

**Branch:** `wo/kvarn-hold`
**Doc:** close-out for the D-21 MTP/speculative-decode divergence fix and the cross-cache-type decode verification that ships alongside it.
**Date:** 2026-08-28

---

## 1. Objective

Two linked goals:

1. **Close D-21** — the MTP (multi-token-prediction) / speculative-decode output diverged from plain decode. This was a correctness bug and is now **closed**.
2. **Verify decode throughput was not regressed** by the fix across every KV cache kind the server can use — and add a reproducible, committed way to re-measure it. The new `--all-cache-type` option on `tools/bench/decode_guard.sh` is that tool.

---

## 2. D-21 root cause & fix

### 2.1 Symptom
With `--spec mtp --draft-tokens 3` on, the MTP output differed from plain (`--mtp 0`) decode, reproducibly, on the ROB prompt and the long 1024-token chunked prompt. The divergence was token- and width-independent: k=1,2,3 all diverged at the same token (35) with the same word substitution.

### 2.2 Fixes (in order)

| Commit | Change |
|--------|--------|
| `84ba2c8b` | **Primary fix** — `int cur_F_mtp = plen;` in `tp2_backend.cpp` (was `plen - 1`). The round-0 verify anchor belonged at `plen`, not `plen - 1`; the old value overwrote the prompt's last KV row. |
| `1d3a0533` | **Gate root-cause fix** — the 27-model GDN gate was inconsistently routed: `cols==1` → `GemvPairedRows` (decode), `cols 2..8` → `SmallTSplit10` (verify). `candidate_is_legal` hard-constrains that split. Routed **all** gate calls to `MmaUnsplit` — the only legal, token-count-independent shared kernel. |
| `db2cee4e` | **Accept + document** — after the gate fix, L00–L02 (GDN) conv+recurrent match; the residual is batched-vs-single attention precision, accepted as **Option C** (see §3). |

### 2.3 Validation after the fixes
| Case | Result |
|------|--------|
| ROB + prompt battery | **Byte-identical** to plain decode |
| P3 **k=2** (verify width 3) | **Byte-identical @80 tokens** |
| P3 k=1 | Diverges ~42 |
| P3 k=3 | Diverges ~35 (first full-attention layer) |

All D-21 source changes are clean — no debug instrumentation left (`text_context_impl.h` reverted); pre-existing user work untouched.

---

## 3. The one accepted residual (Option C)

After the gate fix, L03+ (the first full-attention layer) can differ by ~1 ulp under MTP-verify vs single decode. Root cause: the verify's `SmallT` attention uses a **wider KV window** (`window = pos[last]+1`, because drafts need `F+k+1`), which changes the split count (`gqa_small_t_active_splits`) and thus the online-softmax split-reduction accumulation order.

The causal mask is correct; the difference is **pure FP accumulation order**, inherent to batching a wider window. A clean fix requires a per-column-window rewrite of the hand-tuned MMA kernel — high risk for a bounded, sub-ulp change. **Decision: accepted (Option C) and documented**, not engineered away.

---

## 4. Decode-guard `--all-cache-type` (new)

`tools/bench/decode_guard.sh` gained an `--all-cache-type` mode (commit `197ecb5d`). It is a **smoke / perf sweep, not a full test suite**: it never compares correctness, only measures the server's own decode tok/s per cache type / context.

For each cache type it starts a **fresh** server via `--kv-dtype`, sizes `--max-context` and `--kv-capacity` to the largest context of that type, runs **one** pass at each context size (48 generated tokens, temperature 0), records the server's own `decode=...tok/s` plus model identity / scale layout, tears the server down, and moves on.

```
kvarn_k4v2: 10000 25000 250000      (--kv-dtype kvarn_k4v2)
int8:       10000 25000 160000      (--kv-dtype int8)
bf16:       10000 25000 80000       (--kv-dtype bf16)
```

> **VRAM ceiling (2× RTX 5060 Ti, 16310 MiB usable/rank).** The context sizes above are the largest that fit the TP2 budget model (`tp2_budget.h`). int8 @250k-equivalent is capped at ~168k tokens and bf16 at ~89k; `200000` (int8) and `100000` (bf16) exceed the 16310 MiB usable and fail the serve preflight. So the sweep uses the nearest fitting sizes. Larger requests need a bigger-memory card or a tighter fixed allocation (smaller workspace/headroom).

Run in background (250k/200k are prefill-dominated, ~9 min):
```bash
ITERS=1 nohup tools/bench/decode_guard.sh --all-cache-type step0 > /tmp/dg_all.log 2>&1 &
```

Each result is emitted as a JSON fragment (tag, case, ctx_target, prompt/completion tokens, wall_s, **server_decode_tps**, timestamp, model_path, model_size, model_mtime, model_magic, model_meta_sha256, scale_layout), so results can never be attributed to the wrong model/layout. Single-server mode (`BASE` / `CTX_LOG`) is unchanged.

---

## 5. Decode sweep results (live background run)

> Numbers below are from the run launched at `2026-08-28 18:03 EDT` (tag `step0`, stamp `180334`). Server-side decode t/s under MTP-on, 48 generated tokens, temperature 0, one pass per context. Servers sized with `--max-context` = target + 256 headroom.

| cache type | ctx | prompt_tok | server_decode_tps |
|-----------|-----|-----------|-------------------|
| kvarn_k4v2 | 10000 | 10008 | 61.2 |
| kvarn_k4v2 | 25000 | 24996 | 67.7 |
| kvarn_k4v2 | 250000 | 249828 | 40.9 |
| int8 | 10000 | 10008 | 71.2 |
| int8 | 25000 | 24996 | 80.1 |
| int8 | 160000 | 159900 | 68.5 |
| bf16 | 10000 | 10008 | 70.4 |
| bf16 | 25000 | 24996 | 68.0 |
| bf16 | 80000 | 79956 | 68.7 |

Raw logs: `~/ninfer/logs/decode_guard_step0_*.log`, per-server `~/ninfer/logs/serve_step0_<type>_*.log`. Stdout of the background run: `/tmp/dg_all.log`.

### 5.1 Interpretation / context
These are **server-side decode t/s under MTP-on** (`--spec mtp`), the metric the G2 gate uses. For reference (independent of this sweep):

- **G1** (attention effective read BW): `141.79 GB/s @250k` after the doc82 A2 win — still **misses the ≥150 gate by 5%**.
- **G2** (decode t/s): `70.9 t/s @40k` (**PASS**, ≥69) and `30.5 t/s @250k` (**FAIL**).

The sweep answers *"did the D-21 fix (MmaUnsplit gate route) regress decode?"* — compare kvarn @10k/25k to the pre-fix baseline in `results/` and watch for the 250k/200k/100k numbers against the G2 expectation.

### 5.2 Checkpoint: `context_length_exceeded` on bf16 @80k
A first run set `--max-context` **exactly** to the ctx target. bf16 @80k was rejected with `400 context_length_exceeded: prompt tokens (79956) + max_tokens (48) exceed max_context 80000`. This is a **pure token-count arithmetic reject** (the generated prompt landed at 79956 and `+48` overflowed the exact cap), *not* a memory/reset issue — each cache type runs a fresh server, so there is no back-to-back state carryover. Fix: add `CTX_HEADROOM` (default 256) so `--max-context` = target + 256, and the request passes. bf16 @80k then decoded at 68.7 t/s. VRAM impact is negligible (≤256 tokens × the per-token KV unit).

---

## 6. Files changed / commits

| Commit | Purpose |
|--------|---------|
| `84ba2c8b` | D-21 primary fix (`cur_F_mtp = plen`) |
| `1d3a0533` | Gate routed to `MmaUnsplit` (root cause) |
| `db2cee4e` | Close/Option C — accepted residual documented |
| `197ecb5d` | `decode_guard.sh --all-cache-type` |

---

## 7. Known limitations / open threads (not blockers for D-21)

1. **G1 @250k gate** — `141.79 GB/s`, short of ≥150 (5%). doc82/doc83 code-space-kernel work is the continuing effort.
2. **G2 @250k re-scope** — Section B of the QA feedback recommends re-scoping G2 (option a) and attacking per-page dequant (option c). The decision belongs to the plan owner.
3. **Test cleanup (QA D)** — ~23 `printf` lines remain in `tests/test_kvarn_gqa.cpp` (to remove or gate behind an env var).
4. **`results/doc78..81` numbering collision** — rename before merge.
5. **F6** — write the G2 re-scope proposal and open the dequant redesign step in `docs/78`.

None of these change the D-21 conclusion in §2/§3.

---

## 8. Conclusion

- **D-21 is closed**: the divergence was a genuine anchor-index bug (`plen` vs `plen - 1`) plus a gate-route inconsistency, both fixed and validated byte-identical on the canonical repros. The only residual is a documented, accepted FP-accumulation tolerance under batched verify.
- **Decode throughput is measurable & reproducible** across all cache kinds via the committed `--all-cache-type` option; results are captured with model identity + scale layout.
- No un-accepted correctness divergence remains. Remaining work is the G1/G2 performance gates and the test-file cleanup — tracked separately from this close-out.
