# 120 — Phase 3 status, the missing byte-identity data, and the complete task list

Status: 2026-08-31. Author: Agent 1 (session). Supersedes parts of docs/104 §4 Phase 2/3 and
docs/113. Companion: docs/119 (layout + quantize critical path), results/119b (sync removal design).

---

## 1. Headline: Phase 3 has no measured progress, and one of its gates is unevaluable

**Prefill is still −12.4% vs bf16** (682.5 vs 779.1 t/s pp on an identical 21 829-token prompt).
docs/113 Steps 0–5: **0 of 5 complete**. The `NINFER_KVARN_PREFILL` gate does not exist; the
direct kernel is unwired dead code with no tail support.

Everything delivered so far on Phase 3 is diagnosis and safety infrastructure, not improvement.
That is stated plainly so nobody reads the commit count as progress on this phase.

## 2. BLOCKER: the byte-identity gate has no reference data

docs/104 §4 Phase 3 gate: *"byte-identity per variant vs its own **Phase-0 prefill output** at
4k/40k"*.

**That data does not exist.** Verified by inspecting every candidate artifact:

| artifact | contents | output text? |
|---|---|---|
| `results/official_greedy_decode_20260829_093825.json` (15 records) | `server_decode_tps`, `mtp_accept_pct`, `mtp_tok_per_round`, `wall_s`, `completion_tokens`, `prompt_tokens`, `model_meta_sha256`, `sample_cfg` | **no** |
| `results/official_sampling_decode_20260829_090807.json` (15 records) | same fields | **no** |
| `tools/verify_baseline.json` | `pp_tps`, `mtp_tps`, `round_phase_ms`, `determinism_ok`, … | **no** |
| `tools/bench/decode_guard_baseline.json` | per-cell `decode_tps`, `mtp_accept_pct`, `mtp_tok_per_round`, `wall_s` | **no** |
| `tools/freq_corpus/fixtures/` | ranking manifests | **no** |

`completion_tokens` is a **count**, not a token sequence. `model_meta_sha256` identifies the model,
not the output. The only stored expected-output fixtures in the repo are GDN/causal-conv unit-test
vectors, unrelated to model output.

**Consequence:** the Phase-0-referenced byte-identity gate cannot be evaluated, now or after the
work. It needs either (a) regenerating the reference from the Phase-0 binary `99111d52` — which is
pre-GDN-fix and was recorded as having a stale-build crash, so this is unreliable — or (b)
redefining the gate.

**Recommendation: (b), and note that docs/113 already specifies the better gate.** docs/113 Step 1
says *"byte-identity vs the **materialize route** at 4k/40k (per-variant A/B, same request, diff
tokens)"* — route-vs-route on the **same binary**. That is evaluable today, strictly stronger as a
regression test (it isolates the route change from everything else that changed since August 29),
and it is what the new oracle supports. docs/104 §4 and docs/113 §5 currently state two different
requirements; they should be reconciled to the docs/113 Step 1 form.

## 3. Corrections needed to docs/104 (agent 1 owns its pending edits — please fold these in)

1. **§4 Phase 3 line** — *"delete `gqa_kvarn_materialize_kernel` (O(n²) pass)"* is wrong twice:
   the O(n²) regime was removed by the docs/66 2-pass split, and materialize is **4.1%** of the
   current prefill gap (0.15 s of 3.65 s), not its cause. Measured in
   `results/113_prefill_nsys_attribution.md`.
2. **§4 Phase 3 gate** — replace "vs its own Phase-0 prefill output" with the docs/113 Step 1
   route-vs-route form, per §2 above.
3. **§4 Phase 2 KVarN residual block** — now stale as of `802322a4` / `29d0a797`:
   - *"KVarN DECODE default STAYS packed"* → **superseded**; default is unified (29d0a797), packed
     retained as `NINFER_KVARN_DECODE=packed` rollback lever.
   - *"slice4 does the opposite: smem-based `kvarn_fwht_channel` per key for BOTH K and V through a
     32KB scratch, 64/128 threads, 32-key tiling, both routes 1 block/SM"* → **no longer true**.
     slice4 now uses register+shuffle `kvarn_mma_warp_fwht`, no dynamic smem, and
     `__launch_bounds__(Wc*32, 2)` restored (2 blocks/SM, verified SHARED=47360B via cuobjdump).
   - The measured residual numbers (43.1/27.4/20.2/10.9) are the **pre-fix** curve; post-fix is
     67.8/72.0/68.3/56.9/46.3/42.1 with kernel cost 0.928–1.048× packed.
   - What the residual block got right and should keep: the packed-kernel numerics description
     (Q rotated once, K unpack-only, V deferred to the LIVE `kvarn_mma_acc_fwht` at `.inc:735`) —
     that is exactly what was ported, and it is now also true of slice4.
4. **§5 "What stays different"** — add the newly-established KVarN-specific fact: prefill deficit is
   `quantize_tile` (99% of the gap), driven by `kSinkhornIters = 16`, i.e. quantizer algorithm cost
   that bf16 simply does not pay. This is a permanent structural difference unless the iteration
   count or the critical-path placement changes.

## 4. Complete Phase 3 task list

Ordered by dependency. Effort assumes the GPU is contended with agent 1's I8 lane.

### A. Prerequisites (must exist before any route work is testable)

- **A1 ✅ DONE** — prefill attribution by nsys kernel totals (`9167829f`). Established that
  materialize is 4% and quantize_tile is 99%.
- **A2 ✅ DONE** — FP-accurate materialize oracle (`d0eab82b`), 6/6 PASS, mutation-validated
  (scale-stride / code-layout / logical-vs-physical mutations all caught). This is the independent
  reference that makes "byte-identity vs materialize" meaningful.
- **A3 ✅ DONE** (`80b8a661`) — **Prefill byte-identity harness.** `--dump-tokens PATH` on the TP2
  decode test writes the full generated token-ID sequence (no backend change was needed —
  `TpRunStats::output_tokens` already carries it), plus
  `tools/parity/prefill_byte_identity.sh` with capture/compare modes. Compare defaults to
  IDS-ONLY, because in a route-vs-route A/B the provenance header differs by construction and
  comparing it would always "fail"; `--full` exists for same-config determinism. On divergence it
  reports the first differing token INDEX, not just a diff. Validated offline (identical-with-different-
  headers → PASS; real divergence → pinpointed at index 1; GPU busy → 77).
  *This is the answer to "do we have the test data": we did not, and now the tool to create it exists.
  The reference itself is still missing — that is A4, which needs a free GPU.*
- **A4 ✅ DONE 2026-09-03 (docs/141)** — captured at `results/141_gate_verify/a4_phase3_ref/` (6 cells: short/long @4k, long @40k x greedy/sampling), labelled "Phase-3 reference (route-vs-route)". NOTE: required fixing a bug in A3's `capture` — it generated a fixed ~52k-token prompt regardless of `--ctx`, so all 4 long cells aborted in `gqa_kvarn_commit: logical page out of block table`. Original: record the Record the **Phase-3 reference snapshot** with A3's `capture` mode on
  the pre-change build, so later steps have a same-tree baseline. Explicitly label it
  "Phase-3 reference (route-vs-route)", not "Phase 0", to avoid repeating the §2 confusion.
  Blocked only on a free GPU; expected ~10-15 min for the 4k+40k x greedy/sampling matrix.

### B. The prefill route itself (docs/113's actual subject)

- **B1 ✅ DONE (`b247fc4d`, in-tree)** — `NINFER_KVARN_PREFILL=materialize|direct` gate, default
  `materialize`, FATAL on unknown value (docs/113 §3) — unknown-value abort verified by docs/141.
- **B2 ⏸ PASSED 2026-09-03 (user decision: "we can pass on b2 and later revisit if we have a ton
  of free time")** — Tail support in the direct kernel (`tail_k`/`tail_v`/`tail_count`/
  `packed_pages` + `key_base = packed_pages*64` boundary logic, ~1 day). **Not worth building on
  current data:** `results/113_prefill_nsys_attribution.md` measured the materialize kernel at
  **0.15 s of the 3.65 s gap (4.1%; ~0.5% of prefill time)** — quantize is 98.9% and runs at
  commit on BOTH routes — and the gap itself is now **~0 vs int8 at 40k/80k** (results/106 §7).
  Best case ≤0.5%, plausibly negative (in-kernel dequant in the 99%-of-time flash loop). Revisit
  only if batched long-ctx VRAM pressure materializes (2a) — and even there chunked materialize is
  the cheaper fix. Until built, the route stays hard-thrown (correct guard: direct without tail
  support is WRONG, not slow — silently drops uncommitted keys).
- **B3 ⏸ PASSED with B2** — remove the two device `printf`s in the direct kernel (lines 249, 252);
  moot while the direct route is not wired.
- **B4 ⏸ PASSED with B2** — extend the A2 oracle to the direct route + run the B1 A/B. Its purpose
  (A/B of direct vs materialize) is moot: the A/B cannot win ~0.5% back.

### C. The thing that actually closes the 12.4% (not in docs/113)

- **C1 ✅ DONE** — Remove four per-head D2D copies in commit + hydrate (`a59de6df`). Verified it
  removed exactly 43 648 copies; **measured perf-neutral** (30.53→30.51 s kernel time). Kept as
  strictly-less-work, not counted as progress.
- **C2 ⬜** — **Remove the per-layer `cudaStreamSynchronize`** in `gqa_kv_append_kvarn_and_commit`.
  Design done (`results/119b`): positions come from `fill_i32_positions` (`start + i`, host-known),
  so the page-run split is closed-form; but DFlash positions are device-sourced, so it needs a
  provenance marker + fallback + a one-time assertion that the closed form equals the current loop.
  Expected **3–5%**, and it is the **prerequisite** for C3. ~half day + GPU verify.
- **C3 ⬜** — Move `gqa_kvarn_commit_completed` to a side stream with an event dependency on the
  tile producer and a join before any attention that could read the page as history. This is the
  only option that closes the full 12.4% (31.38 − 3.61 = 27.77 ≈ bf16's 27.74). Riskiest item:
  changes stream ordering in the KV write path, and `prepare_page_for_append` does
  read-modify-write across page boundaries.
- **C4 ⬜ (owner decision, not mine)** — Reduce `kSinkhornIters = 16`. Biggest single multiplier on
  the 3.61 s, but it is a **numerics change** requiring accuracy re-validation. Flag, don't do.

### D. Gates to pass at closeout

- **D1 ✅ (redefined, 2026-09-03)** — prefill t/s within ±5% of **int8** at 40k/80k/160k, ≥
  baseline v1 every cell (bf16 is invalid on this 36-SM box — it crashes at 40k; docs/124).
  **Result: PASS @40k (+0.2%) / 80k (−0.3%); 160k = −6.8%** (572.9 vs 614.4, 4b run 09-03 14:59 —
  a known residual slightly outside ±5%; the O(n²) materialize cost widens at the longest ctx).
  Measured by prefill t/s (excludes decode). See results/106 §7, docs/142.
- **D2 ⏸ DEFERRED 2026-09-03 (user: pass on B2, revisit only with free time)** — byte-identity
  direct vs materialize at 4k/40k: BLOCKED on B2+B4 (both halves — the `direct` route has NO
  launch site in src/, throw at `gqa_attention_kvarn.cu:300` UNCONDITIONAL, page alignment does
  not help; runtime-verified exit 134 at T=1969; trap: throw unreachable below T>=7, short probes
  FALSE-PASS). B2 now PASSED as not-worth-building (results/113: materialize = 4.1% of a gap now
  ~0), so D2 is **deferred, not owed**. The **2x determinism half PASSES 6/6**; A4 snapshot DONE
  (see §A4). Docs/141 evidence: `results/141_gate_verify/d2_route_probe/`.
- **D3 ⬜** — 250k KVarN must-pass battery T1/T2/T3/T5/T8/T10/T11/T15/T16/T17/T18 unchanged.
- **D4 ✅ PASS by static proof (docs/141)** — `b247fc4d`'s entire diff is 36 lines in one file and its only executable statement sits inside `tokens > kKvarnSmallTMax && !packed_verify` => tokens>=7; decode is T=1, verify T=2..6, so the prefill route is unreachable from decode by construction. Literal battery vs pin `aa8ef21b` = 30/32 identical; the 2 diffs (packed/h5 at index 5) are **attributed drift** to `9a58008a`, in the direction of packed *converging onto* unified. **Correction debt: the window `29d0a797..b247fc4d` does not compile** (2-arg template call against a 1-param kernel, fixed only at `aa110a22`), so the A2/A3/A4 prerequisite chain landed on a broken tree and no commit is simultaneously pre-Step-1, buildable, and post-unification-default. Original: decode side provably untouched: byte-diff battery vs the pre-Step-1 build must be
  IDENTICAL. Note this is now **harder** because the decode default is unified (29d0a797) — pin the
  comparison build explicitly.
- **D5 ⬜** — fast gate 0 fails per step; `run_ci.sh --full` 0 fails at closeout.

### E. Known-blocked / external

- **E1 ⬜** — agent 1's I8 unified route currently crashes at model level
  (`cudaErrorIllegalAddress` in `scatter.cpp:86`) while its oracle passes 8/8. Until fixed,
  `regress_unified.sh` cannot go green tree-wide, which is the shared pre-flight for D4/D5.

## 5. Measurement protocol correction

Wall-clock prefill t/s **cannot support a ±5% gate on this box.** Measured: the same binary read
599.8 t/s pp under nsys and 699.7 without it; two perf-neutral builds differed by 12% nsys-vs-nsys.
Total GPU kernel time from `cuda_gpu_kern_sum` is stable to ~0.1% and is what all the attribution
above is built on.

Also: docs/104 §8.1 already warns "48 tokens/cell → ±~7pp acceptance noise; use 3-run medians".
The owner's one-run protocol is adequate for the decode green bar (acceptance is controlled out by
ms/round) but **not** for prefill perf claims. Prefill needs kernel-time totals or 3-run medians.

## 6. Progress since this doc was written (2026-08-31)

Done, all CPU-only:
- **A3 ✅** byte-identity harness (`80b8a661`) — see above.
- **CI gap closed** (`639a479b`) — `run_verify_tests.sh` had ZERO KVarN coverage while KVarN is a
  shipped default route. Added a `kv_kvarn` row with baseline-graded t/s + acceptance, a hard
  floor, and a **route assertion** off the launcher banner so an accidental route flip fails CI
  instead of appearing as an unexplained perf delta. Also fixed the banner, which printed
  "UNIFIED (default)" even when the route was set explicitly.
- **Bench coverage hole made loud** (`0ab41da2`) — `bench_kvarn_2pass.cu` is the direct kernel's
  only exerciser and it hardcodes `tail_count = 0`, which is exactly why the direct kernel's
  missing tail support went unnoticed. Now warns in-file and in output that a passing `rel_l2`
  is not tail coverage, and points at the oracle. Full tail wiring deliberately deferred to B2
  (it needs buffer/tile-count changes in a GPU program that cannot be run blind).

Still open, in order: ~~**A4**~~ ✅ DONE (docs/141) → ~~**B1**~~ ✅ DONE (`b247fc4d`) →
**B2** ⏸ PASSED (user 2026-09-03, see §B) → ~~**C2/C3**~~ ✅ DONE + measured 0.0% (`64c2d705`).

**Prefill performance: parity with int8 at 40k/80k (±0.3%), −6.8% @160k** (results/106 §7,
docs/142) — the old "−12.4% vs bf16" figure is STALE (earlier build; bf16 reference invalid at
≥40k on this box). The residual 160k gap is quantize/commit-side (98.9% per results/113), which
B2 cannot touch.
