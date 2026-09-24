# HANDOFF — NVFP4@TP4 incoherence: **FIXED & CLOSED 2026-09-16 ~17:50 CDT** (see bugtrack E-14)

> **E-15 ADDENDUM (~18:40 CDT):** root cause REFINED — not a miscompile: ROCm 6.2's
> `amd_hip_bf16.h:610` `__low2bfloat16` binds the INTEGER-converting ctor (unbraced
> `__hip_bfloat16(hr.x)`, raw bits read as an integer: 0x3f40 → 16192.0 = 0x467d); same defect
> in `__low2bfloat162` (:617). CLASS FIX landed: shim redirects both names to raw-braced
> correct bodies (with the pre-fix kernel now passing the parity cell through it). NEW suite
> cell `ninfer_bf16_halves_contract_test`: RED on stock (exit 1, 1200 mismatches) / GREEN via
> shim (exit 0). Receipts in `results/amd/coherence/E15_*`.

> **STATUS UPDATE (E-14, supersedes §4 below):** root cause was `__low2bfloat16` miscompiling
> on gfx900/ROCm 6.2, reached ONLY by the prefill-pairs kernel's split3 store branch (its lone
> use in the tree). Fix: re-round halves from the float accumulators (`causal_conv1d.cuh`
> split3 branch). Closure: permanent suite cell `tests/ops/test_causal_conv_route_parity.cu`
> (RED pre-fix / GREEN post-fix, banked outs in `results/amd/coherence/E13_route_parity_*.out`)
> and ladder GREEN — banked binary `ninfer-serve_b864220912aec151.bin` scores **16/16 legs PASS
> with `--long`** (prompt 56..2000). Coherence crisis CLOSED; next axis is throughput (§4.4).

# HANDOFF — NVFP4@TP4 incoherence: MECHANISM CONVICTED (causal-conv T-gated kernel path)

**Date:** 2026-09-16 ~17:40 CDT · **Branch:** `amd/tp4-cure` (pushed to origin) ·
**Seat:** audit-seat continuation ("continue-here-docsamdbugtracknvfp4tp4incoherence")
**Parent doc:** `docs/amd/BUGTRACK_NVFP4_TP4_INCOHERENCE.md` — append-only entries **E-11, E-12, E-13** are this session's record; this file is the human-facing checkpoint.

---

## 0. THE ONE-PARAGRAPH STATE

The NVFP4@TP4 served-output garble (coherent ≤66 prompt tokens, mojibake ≥67, deterministic)
is **convicted to the causal-conv organ**: layer-0's `ops::causal_conv1d_silu_split3` writes
wrong values into the **even channels of QC/KC/VC from row 0** whenever the main prefill pass
has **T ≥ 65**, because the wrapper dispatch (`causal_conv1d_silu.cpp:162-169`) routes
**T ≤ 64 to `causal_conv1d_sequence_kernel`** and **T ≥ 65 to `causal_conv1d_prefill_pairs_kernel`**
(`kCausalConvSequenceMaxTokens = 64`). Identical inputs, different kernel ⇒ different outputs —
E-9's impossibility resolved. Everything downstream (delta-net, norms, out_proj, allreduce,
mlp) decays from that row-0 wound; the served ids were always correct. **No fix is landed yet**;
the pin-the-line → fix → RED/GREEN-cell plan is in E-13 and §4 below.

## 1. WHERE TO RESUME

```bash
# the branch
git fetch origin amd/tp4-cure && git worktree add ../worktrees/amd-tp4-cure amd/tp4-cure
# the parent tracker (read E-11/E-12/E-13 first; §1 do-not-redo list is binding)
$EDITOR docs/amd/BUGTRACK_NVFP4_TP4_INCOHERENCE.md
```

- **Lane worktree:** `/home/chris/worktrees/amd-tp4-cure` (branch `amd/tp4-cure`)
- **Head at handoff:** see `git log --oneline -8` — E-11 taps, E12/E12b runner+analyzer, E-12/E-13 doc entries all committed
- **Bank binary under test:** `/home/chris/artifacts_bin/ninfer-serve_1f068ea7aa8a6aa2.bin`
  (sha256 `1f068ea7aa8a6aa2c749b5f801166c15aeee509a987f60de22d23588276eeb6e`) — boots from the bank, never a lane build path
- **Model artifact:** `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer` (18,324,067,840 B — size-check it)
- **GPU grant:** issued DIRECT BY THE USER 2026-09-16 ("you are granted usage of anything, just fix the problem") — recorded in E-12. Standing laws still apply: manifest before spawn, KFD check, kill only recorded PIDs, never a pattern.

## 2. WHAT THIS SESSION DID (chronological, all committed)

1. **E-11 taps built** (commit "BUGTRACK E-11/E-12 taps armed…"):
   - `NINFER_COH_ALLOC` — full arena allocation table in `src/core/arena.{h,cu}`: every alloc/PUSH/POP/RESET with seq/offset/bytes/align, MARK-segmented by `REQ/PASS/LAYER/DEC` (+ `OWNER` rank handshake), per-arena files, default OFF.
   - `NINFER_COH_OPS` — 16-stage per-op bf16 dumps at layer-0 gidx0 prefill (`X0 HN G0 B0 QK QV ZB QC KC VC GB BB O0 ON PA XA` + `XL`) in `text_context_impl.h`, via the proven `c_dump_bf16` writer, default OFF.
2. **Runner + analyzer:** `results/amd/p3/E12_coh_tp4_agent6.sh` (STAMP_ACK gate, KFD-clean precondition, df gate, env-verify per the E-4 stale-server lesson, legs k=4/5/36/4-glass, mt=64 per the E-4 warning, manifest+release rows) and `results/amd/coherence/e12_analysis.py` (stdlib-only; self-tested on planted faults). Analyzer ROWS table uses **measured** world-4 dims (k_rows=512, v_rows=1536, qkvz=4096, hidden=5120; fp32 stages = 2× bf16-words).
3. **Census generalized (zero boot):** banked E-10 `e11_ptr_serve.log` (sha16 `5ab611c9a2784fd9`) re-read in full — 1152 lines, all legs/ranks/gidx: no overlap, no extent drift among the six GDN tensors (banked already; no action).
4. **Orphan released:** exited E-10 seat's server (PID 2059260, port 8098, binary 38b30843…) killed by exact PID after user grant; KFD verified 0.
5. **Two windows fired** (E12, then E12b after rank-tagging marks — the 4 ranks are THREADS of one process and interleaved marks; `OWNER R=<rank> work=<base>` handshake fixes segmentation):
   - RED row both times, deterministic: plen66 PASS "BLUE", plen67 FAIL mojibake, plen98 FAIL, plen66 glass PASS.
   - **Part A:** work-arena alloc tables T64 vs T65 — **1235 vs 1235 allocs, zero anomalies, all ranks** ⇒ allocator exonerated.
   - **Part B:** the bisect table — inputs bit-identical through QV/ZB (post-unpack), **QC/KC/VC dirty from row 0**, downstream propagation. Analyzer output: `results/amd/coherence/E12b_analysis.out`.
6. **Fine structure + the gate (E-13):** even-channels-only dirt (256/512, every row 0..58; odd channels bit-clean), wrong values ≈ +6.4..7.2 (catastrophic), T65≡T96 rows 0..58 bit-identical, and the **dispatch gate at T=64/65** (`kCausalConvSequenceMaxTokens=64`, `launcher/causal_conv1d.h:14`; kernel refs `kernel/causal_conv1d.cuh:71` pairs vs `:156` sequence; prefill picks pairs by alignment `launcher/causal_conv1d.cu:78-88`).

## 3. WHY EVERY PRIOR OBSERVATION FITS (no re-derivation needed)

| Record law | Explained by |
|---|---|
| E-3: cliff exactly at T=65 | dispatch switches kernel at T=65 |
| E-9: impossibility (identical inputs ⇒ different outputs) | different KERNEL, not different math |
| E-8: T65..T96 share ONE bit-identical wrong computation | same kernel + identical inputs ⇒ identical wrong outputs |
| E-4: force-recurrent didn't cure | the conv runs on both delta-net paths |
| E-2: garbage from generated token 1 | row-0 wound, downstream decay |
| E-5: LE clean / LA@0 wholesale dirty (5120/5120 bits) | even-channel conv dirt → recurrence+norms spread it |
| Z1 zero-state byte-identical | state content isn't the trigger; the kernel CLASS is |

## 4. NEXT STEPS (in order — this is the whole remaining work)

1. **Pin the line (one device cell, seconds):** run `causal_conv1d_prefill_pairs_kernel` vs `causal_conv1d_sequence_kernel` on identical synthetic [C,T]+state. Prime suspect: the pairs kernel's weight-tap addressing in bf162 units (`weight2[p + k*C2]`, kernel/causal_conv1d.cuh ~:103-106) vs the `[C,4]` tap-major layout (channel 2p's tap k = element 8p+k) — the sequence kernel's per-channel indexing is the semantically-green reference. Same audit for the scalar `causal_conv1d_prefill_kernel` (:41) if time permits.
2. **Fix minimal:** correct the pairs addressing (landing), or route T>64 through the sequence kernel (diagnostic stopgap only — perf regression, not the landing).
3. **RED→GREEN closure (law):** (a) a permanent suite cell — sequence-vs-pairs bit-equality on identical input across C/T classes (guards the CLASS: any kernel-pair divergence under one dispatch); (b) definition-of-done: NEW banked binary sha on which `results/amd/coherence/coherence_ladder.py` exits 0 with `--long`, citing §1C row 26 / E12b rows as RED.
4. Only after coherence GREEN: throughput axis (prefill ~10 tok/s at 10k — separate crisis, phase-axis rule).
5. Housekeeping: `NINFER_COH_ALLOC`/`NINFER_COH_OPS` are default-off and can ship as-is; consider retiring the E12b runner's port 8100 assumption if the board's port table moved.

## 5. REPRODUCTION (deterministic, ~90 s of card time)

```bash
cd /home/chris/worktrees/amd-tp4-cure
STAMP_ACK=G-AMD-E12b bash results/amd/p3/E12_coh_tp4_agent6.sh     # boots TP4, 4 legs, banks rows
python3 results/amd/coherence/e12_analysis.py /tmp/coh_e12b | tee results/amd/coherence/E12b_analysis.out
```
Dumps are volatile (/tmp/coh_e12b) but byte-reproducible on re-run (deterministic defect). Banked rows/logs/analysis live in `results/amd/coherence/E12b_*`.

## 6. WARNINGS (each paid for)

- **max_tokens ≥ 48** on the BLUE body — smaller mt makes FALSE FAILs (E-4).
- **/tmp names lie** — cite banked path+sha only (chair provenance ruling).
- The 4 ranks are THREADS of one process — per-rank evidence needs rank-TAGGED marks (the OWNER handshake), not per-process assumptions.
- Verify `/proc/<pid>/environ` of the listener before trusting any boot's rows (stale-server class).
- `rocm-smi --showproductname | grep -c "Vega 10 \[Radeon Pro V340"` picks the coordinator doc — 4 ⇒ this lane's laws apply.
- Disk gate in the runner is 12G (amended from 20G with citation; no-build window).

— End of handoff. The bug is cornered; one kernel-addressing fix and one green ladder remain.
