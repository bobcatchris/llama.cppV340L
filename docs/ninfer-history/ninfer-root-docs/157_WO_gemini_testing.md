# WO (unnumbered — coordinator assigns): Gemini testing orders for the docs/157 prefill mission

**Owner of execution:** Gemini (testing role). **Author:** A1 lane. **Coordinator:** grants GPU windows.
**Mission link:** `results/157_MISSION_STATE_and_handoff.md` (§3 numbers ledger, §9 protocols, §10 risks).
**Definition of "no regression"** (used by every gate below):
1. greedy decode byte-stable across runs of the same binary+config (content sha equality);
2. decode t/s within **±2%** of the guard baseline rows (80.2 @10k / 56.3 @160k, kvarn_k4v2, MTP-on);
3. prefill ≥ shipped-default row (§3: 1,061 / 1,035 / 1,075 at 10k/25k/63k, ±3% run noise);
4. VRAM ledger clean: cards return to ~15 MiB after teardown; no capacity refusals at shipped defaults.

**Global rules:** GPU work ONLY inside a coordinator-granted window; guard `nvidia-smi` (15 MiB / 0%)
immediately before each serve launch; kill by PID or `pkill -x ninfer-serve` only; poll VRAM <1000 MiB
after teardown; use port 8091 (CI convention) unless told otherwise; **never** `pkill -f "ninfer-serve"`.
All WOs are CPU-side until a window is granted; prepare scripts first, execute second.

---

## WO-G1 — Regression battery on post-fix main (priority 1, one window, ~25 min)

**Why:** the chunk-flag fix + arena scaling changed the load path; we must prove the shipped default
behaves identically to the pre-campaign binary and that decode held.

**Cells** (serve from `/home/intel/ninfer/repo/build/apps/ninfer-serve`, artifact
`/home/intel/models/qwen3_8_27b.ninfer`, `--devices 0,1 --kv-dtype int8 --max-context 131072`, port 8091):
1. 25k cell (body `/tmp/p2_body_25k.json` — regenerate if missing; generator pattern in
   `results/157_p2_chunk_workspace.md` §3 table footnotes). Gate: `done finish` prefill =
   **1,035 t/s ±3%**; greedy content sha == a second identical run (determinism).
2. Decode guard: `ITERS=1 DG_SPECS="kvarn_k4v2|10000 160000" tools/bench/decode_guard.sh
   --all-cache-type g1_regress` from repo root. Gates: 80.2 ±2% @10k, 56.3 ±2% @160k,
   MTP acceptance ≥ 70%.
3. Capacity check: default launch must accept `--max-context 131072` (the chunk fix must NOT have
   regressed feasible context at default 512). Gate: no "no feasible capacity" error at startup.

**Produce:** `results/157_g1_regression.md` — one table, pass/fail per gate, serve logs attached.

## WO-G2 — Determinism & byte-identity gate for the chunk flag (priority 2, same window, ~15 min)

**Why:** the flag is now live; users can pass `--prefill-chunk 512/1024/2048`. Each value must be
internally deterministic and must not crash (the arena fix covers the sizing).

**Cells:** for CH in 512 1024 2048 (max-context 73728 for 1024/2048 — see capacity table
`results/157_p2_chunk_workspace.md`): fire the 25k body twice; record (a) content sha of both runs —
must be equal WITHIN a config, (b) prefill t/s both runs — report spread, (c) no bad_alloc / no
capacity refusal. Note: ACROSS configs shas may differ legitimately (chunk-boundary float ordering).

**Produce:** `results/157_g2_chunk_determinism.md` + the 6 shas in a table.

## WO-G3 — NVFP4-weights verification battery (priority 3, fires after the BF16-shard requant lands)

**Why:** the decisive docs/157 §13.2.2 experiment needs clean verification harness around it.
**Trigger:** agent2/A1 land the `materialize_tp` BF16-shard requant (design:
`worktrees/wo-nvfp4-prefill/docs/STATUS_nvfp4_tp2_blocked_on_bf16_sharded.md`) and flip the window.
**Cells** (serve from the **wo-nvfp4-prefill worktree** binary, artifact
`~/ninfer/incoming/nvfp4/qwen3_8_27b_nvfp4.ninfer`, `NINFER_ALLOW_NVFP4_TP2=1`,
`--devices 0,1 --kv-dtype kvarn_k4v4 --max-context 131072 --prefill-chunk 1024`, port 8091):
1. Load: serve reaches `listening` with both ranks reporting materialized NVFP4 layers (no
   bad_alloc / no unconsumed-object errors).
2. 10k / 40k / 80k decode-guard cells, 1 iter (`DG_SPECS="kvarn_k4v4|10000 40000 80000"`).
3. **The decisive cell:** 25k prefill body → prefill t/s. Compare vs groupwise 1,035 t/s.
   Interpretation: ≥1,300 → weight format was the gap (ship FP4); <1,100 → NVFP4 weights alone
   don't close it (kernel work / dtype-class ceiling dominates) — either way RECORD, don't editorialize.
4. nsys the 25k cell (same method as `results/157_p0`): **grep for the NVFP4 consumer kernel
   families — `nvfp4_w4a4_tma_kernel`, `nvfp4_w4a4_mma_kernel`, `nvfp4_gemv_kernel`** (the
   gzenz-era `ln_mma`/`ln_tma` names do not exist in this tree's kernels — window #7 lesson)
   — presence proves the FP4 path is hot; absence means the runtime still routed around it (report, don't fix).
**Produce:** `results/157_g3_nvfp4_cells.md` with the §4 interpretation table filled.

## WO-G4 — NCCL transport assertion as a standing CI check (priority 4, CPU-mostly)

**Why:** the P2P-capable driver means NCCL *could* pick BAR1 P2P transport after an environment change
(driver/module update, NCCL version bump) and silently regress to ~7-11 GB/s. This must never happen
unnoticed again.
**Do:** add a guard script `tools/bench/nccl_transport_check.sh`: launch serve 60 s with
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING`, assert every `Channel .* via` line contains
`SHM/direct/direct` (or explicitly pass if a future change is DELIBERATE — gate via env
`EXPECTED_TRANSPORT`), tear down. Wire into `run_ci` prereun if the pattern fits the existing
contract-lint structure (`tools/ops/contract_lint.sh` style).
**Gate:** zero `via P2P` lines on the current topology unless `EXPECTED_TRANSPORT=P2P` is set.

---

## Escalation / stop conditions

- Any capacity refusal at shipped defaults → STOP that cell, record, ping A1 (it means the VRAM ledger moved).
- Any bad_alloc / illegal-address → STOP, capture the log + dmesg tail, ping A1. Do not retry blind.
- Any byte-sha mismatch within a config → STOP (nondeterminism = new bug class), capture both logs.
- GPU window conflicts: coordinator arbitrates; never overlap with another lane's granted window.
