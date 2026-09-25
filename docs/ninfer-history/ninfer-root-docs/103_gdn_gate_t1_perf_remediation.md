# 103 — GDN-gate T=1 performance remediation (−7% decode / −6% verify since 08-26)

**Status:** OPEN — root cause identified by code review; fix not yet implemented. GPU work parked (server busy, 2026-08-29 06:35).
**Owner:** TP2 agent. **Reviewer:** user. **Companions:** docs/102 (pipeline map), docs/104 (KV-uniform perf plan), docs/50, results/d21_mtp_verify_anchor_fix.md.

---

## 1. Finding

CI verify-suite throughput dropped between **2026-08-26 11:05** and **2026-08-28 16:41** and has not recovered:

| CI run | plain t/s (40 tok, bf16) | MTP k=3 t/s (512 tok, bf16) | verify_ms (T=4) | MTP accept % (greedy) |
|---|---|---|---|---|
| 20260823_133856 | 35.47 | 80.68 | 35.27 | 82.0 |
| 20260825_125625 | 35.46 | 80.78 | 35.22 | 82.0 |
| 20260826_095551 | 35.45 | 80.77 | 35.22 | 82.0 |
| 20260826_110519 | 35.40 | 80.71 | 35.24 | 82.0 |
| **20260828_184152** | **32.80** | **75.76** | **37.44** | 80.4 |
| 20260828_223347 | 32.82 | 75.41 | 37.65 | 80.4 |
| 20260828_224030 | 32.88 | 75.38 | 37.67 | 80.4 |

(20260825_122804 — plain 16.76 / verify 73.5 ms — is the known dirty `/tmp/ninfer` build window, not a data point.)

Deltas vs the 08-26 regime: **plain −7.5%, MTP −6.6%, verify +6.2%**. The CI verdict has been `WARN` (perf checks vs the 08-23-era baseline) ever since; these are not new in the D-22/D-23 commits (see §5).

## 2. Attribution

The only commit between 08-26 11:05 and 08-28 16:41 that touches the **shared bf16 decode path** is:

`1d3a0533` (08-28 16:42) *fix(qwen3_6): route 27-model GDN gate to token-count-independent MmaUnsplit*

It changed the 27B GDN control-gate routing table (`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp`):

```cpp
// before (k27Routes):
{1, 1}    -> GemvPairedRows      // T=1 decode: memory-bound GEMV
{2, 8}    -> SmallTSplit10       // T=2..8 verify
{9, any}  -> MmaUnsplit          // T>=9

// after:
{1, any}  -> MmaUnsplit          // ALL token counts, MMA-based
```

Why this costs time:
- Every decode step runs **48 GDN layers**, each with a control-gate projection. At T=1 that projection is a GEMV (bandwidth-optimal); `MmaUnsplit` is an MMA schedule (compute-bound, tensor-core tiles with mostly-empty work at T=1). The GEMV→MMA swap at T=1 across 48 layers per step is the right magnitude for a 2.2 ms/step hit on a 31.2 ms step (≈+7%).
- Verify (T=4) went from `SmallTSplit10` to `MmaUnsplit` — verify_ms +6.2% (35.2→37.6). Verify is 83% of an MTP round, so MTP t/s inherits the hit.
- The MTP t/s delta decomposes: verify +6.2% (GDN routing) × tokens/round −1.4% (3.46→3.41, a consequence of the *separate* D-21 anchor fix `84ba2c8b`, which corrected a position bug — the 82.0→80.4 acceptance delta is the acceptance of the *corrected* path, not a defect).
- All other commits in the window are KVarN-only (packed kernel, lever-1, docs/82/83) or docs; the CI main runs use **bf16 KV** (default; `--kv-dtype int8` only on the `kv_i8` row), so they cannot be the cause.

The change was **necessary for correctness**: `candidate_is_legal` hard-constrains the old split (GemvPairedRows only at cols==1, SmallTSplit10 only at 2..8), so decode and verify could never share a kernel unless both went to MmaUnsplit. The old split accumulated the gate ~1 ulp differently between T=1 and T>1, which drifted the recurrent state and flipped greedy argmax (D-21 residual; `results/d21_mtp_verify_anchor_fix.md`). The fix bought byte-identity at a 7% speed cost — that cost is what this doc removes.

## 3. Remediation (recommended)

**Add a T=1 schedule that is bit-identical to MmaUnsplit's accumulation but loads like a GEMV.**

- New `Bf16GdnGatingScheduleId::MmaUnsplitT1` (name TBD): same accumulation order, same dtype ops, same rounding as `MmaUnsplit` for any `cols` (so the T=1 column is bit-identical to the T=4 column by construction — the D-21 invariant is preserved *by design*, not by luck), but with the T=1 memory access pattern (single-token tile, coalesced weight read, no empty MMA lanes on the K dimension where possible).
- Route: `k27Routes = {{1,1} -> MmaUnsplitT1, {2,any} -> MmaUnsplit}`.
- Acceptance gate (byte identity): the D-21 battery (`P3 k=1`, `k=2` byte-identical to plain; H5 1024-token; 6-prompt battery) must stay byte-identical after the change — this is a pure routing/schedule swap, so any output change is a defect, full stop.
- Perf gate (decode_test, bf16, same harness as CI):
  - `plain_tps` (40 tok) ≥ **35.0** (restore 08-26 level; 08-26 was 35.4–35.5, allow 0.5 thermal margin),
  - `verify_ms` (T=4) ≤ **35.6** (restore 08-26 level),
  - `mtp_tps` ≥ **79.0** (80.7 era minus the legitimate anchor-fix tokens/round change).
- If `MmaUnsplitT1` cannot hit the perf gate (e.g. the MMA schedule is intrinsically ~7% slower here), fall back to **Option B**: make the batched kernel's accumulation bit-identical to the T=1 GEMV accumulation per column (fix the numerics on the T>1 side instead), keep `GemvPairedRows` at T=1. This is more invasive (verify/prefill accumulation order changes) and must re-run the full D-21 battery in both directions.

**Do NOT** re-introduce the old split (GemvPairedRows + SmallTSplit10): that is the D-21 bug.

## 4. Verification plan (GPU, in order)

1. Build from `1d3a0533~1` (pre-fix routing) + current tree otherwise → measure plain/verify/mtp. Confirms the attribution (expect ≈35.4 / 35.2 / 80.7). *(This step is diagnostic only — the pre-fix tree has the D-21 bug; do not leave it in place.)*
2. Implement `MmaUnsplitT1` + routing → full D-21 byte-identity battery (must be byte-identical) + perf gate above.
3. Full CI gate → `PASS` with the WARNs cleared (or baseline re-set with a written note).
4. Record: `results/103_gdn_gate_t1_remediation.md` with the A/B tables, provenance (commit + sha256 + mtime per `tools/build_and_serve.sh`), model identity.

## 5. Scope notes

- **The D-22/D-23 commits under review (257832b7, 6ad60d42, 9a33b896, 502b42ca) contribute nothing to this drop.** Pre-fix vs post-fix on the same day: plain 32.80→32.88, mtp 75.76→75.38, verify 37.44→37.67 — zero delta (the D-22 fix adds 4×248 KiB NCCL allgathers per MTP round ≈ 40 µs/round, within noise; it is also currently unconditional in greedy rounds — a possible micro-optimization later, ~0.1%).
- The D-21 *acceptance* change (82.0→80.4) is the corrected-path number, not a regression; do not "fix" it back.
- This doc is the GDN-gate instance of the general problem — **token-count-dependent kernel routing splits both numerics and perf** — addressed systematically in docs/104 (KV-cache uniform performance plan).

## 6. Non-goals

- No changes to D-21's byte-identity property (that is the invariant we protect).
- No KVarN/packed-kernel work (docs/83 lever-1 stays parked per the bug-free-first directive).
- No serve-path work; decode_test only.
- No baseline re-baseline without a written justification in the results file.

## 7. Risks

- `MmaUnsplitT1` may not reach GEMV-class bandwidth at T=1 (MMA tile shape is fixed); if so, Option B carries a numerics-change risk to the batched path — mitigate by making the change under an env gate and A/B-ing byte identity before landing.
- Thermal/clock variance on the 2×5060 Ti box can move t/s by ~1–2%; use A/B on the same day, not absolute thresholds across days, for the attribution step.
