# 129 — Phased kernel gate: M0 done, M1 handoff

Status: M0 SHIPPED (committed + pushed), M1 in progress (blocked on the model
artifact). This is a **handoff** so the next agent can pick up M1 without re-deriving
anything. **Read the plan first: `docs/128_phased_kernel_gate_pipeline.md`** — this
doc is the execution-state layer on top of it. Where this doc and §8 conflict, the
kernel source is ground truth.

Worktree: `/home/intel/ninfer/worktrees/wo-phase-gate` (branch `wo/phase-gate`,
based on `wo/kv-uniform`). Pushed to `github` (the `origin` remote is a dead local
backup path — push to `github`).

---

## 1. TL;DR current state

- **M0 (docs/128 §6): DONE, committed `1c26d5b1`, pushed to `github`.**
  - Chain `D1 (GPU quantize) → D2 (GPU dequant of the ACTUAL GPU D1 artifact)`.
  - Clean run → exit 0; **negative test** (`--mutate-d2`) → D1 PASS / D2 FAIL / exit 2.
  - Byte-ratchet baseline committed (`tools/bench/phase_baseline.json`); CPU codec is
    the correctness oracle that validates the baseline.
  - Runs <1 s, GPU output is **byte-stable** (deterministic), so per-phase sha256
    divergence vs baseline = regression at the first failing phase.
- **M1 (full decode chain D1–D7): in progress, mostly BLOCKED.**
  - **No Qwen model artifact on this machine** (only unrelated ComfyUI `.safetensors`)
    → the real TextContext path (D3 draft head, D8–D10 e2e) is **not runnable** here.
  - **GPU is shared with the main working agent** (used off and on). The gate's GPU
    use cases are quick (5–10 s per phase, <10 s chain), so work around them — check
    `nvidia-smi` and only run when the GPU is idle; if busy, do CPU-only work and retry.
  - The achievable, unblocked M1 work is the **D4–D7 CPU reference** (the plan's stated
    "long pole") — pure CPU, no model. That is the next agent's main task.

---

## 2. M0 — what exists, how to build/run, and the invariants

### 2.1 Files (all committed on `wo/phase-gate`)

| File | Role |
|------|------|
| `tests/phase_gate.cu` | the gate binary: builds pinned input, runs D1/D2 on GPU, compares vs CPU codec, byte-ratchets, prints the chain table, exits with first-failing index. |
| `tests/phase_gate_artifacts.h` | self-contained SHA-256 (OpenSSL `libcrypto`, `-lcrypto`) + the `Artifact` struct. Self-test vector `sha256("abc")=ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`. |
| `tools/bench/phase_gate.sh` | runner: `clean`, `--negative`, `--baseline`, `--compare`, `--build`, and pass-through (e.g. `--json`, `--mutate-d2`). |
| `tools/bench/build_phase_gate.sh` | **standalone build** — compiles only the test, links the prebuilt `libninfer_ops.a`/`libninfer_core.a`/`libninfer_nvfp4_tma.a` from the sibling `wo-kv-uniform` build. Portable: `WT`/`KLIBS`/`CUDA` env overridable. |
| `tools/bench/phase_baseline.json` | the **green ratchet** (the clean `--json` output: `prompt_sha256`, D1/D2 `gpu_sha256` + deltas). |
| `tools/bench/phase_tolerance.json` | documents ratchet tolerance (hash equality) vs correctness-oracle `max_delta` (D1 codes 64 / scales 128 ULP, D2 tiles 64 ULP) with observed-clean values. |
| `tests/CMakeLists.txt` | CI entry: `ninfer_phase_gate` op test (`ninfer_ops` + `OpenSSL::Crypto`; `find_package(OpenSSL REQUIRED)`). Not exercised locally (the local build is standalone) — CI only. |

### 2.2 Build + run (verified)

```
cd /home/intel/ninfer/worktrees/wo-phase-gate
tools/bench/build_phase_gate.sh                 # ~40 s (links prebuilt libs)
tools/bench/phase_gate.sh                       # clean  -> exit 0
tools/bench/phase_gate.sh --negative            # negative test -> "caught at D2, exit 2"
tools/bench/phase_gate.sh --compare             # vs baseline -> "MATCH baseline"
tools/bench/phase_gate.sh --json                # pure JSON on stdout (human on stderr)
tools/bench/phase_gate.sh --mutate-d2 --compare # diverges at D2 -> exit 2
```

### 2.3 Exit codes (docs/128 §8.5)

`0`=clean · `1`=D1 · `2`=D2 · `77`=no GPU · `90`=SHA self-test fail.
Exit = **first failing phase index**. A byte phase FAILs if its per-phase sha
diverges from the baseline (byte-parity, tolerance=0) **or** its correctness-oracle
delta exceeds tolerance.

### 2.4 What each phase does

- **D1 (commit/quantize):** GPU `quantize_k_tile_gpu`/`quantize_v_tile_gpu` on a
  synthetic pinned input vs the **CPU codec** `quantize_k_tile`/`quantize_v_tile`
  (`src/ops/kvarn/kvarn_codec.cpp`). Artifact = packed codes + scales + layout.
  Oracle: codes byte-diff (tol 64), scales ULP (tol 128).
- **D2 (dequant/materialize):** GPU `dequantize_k_tile_gpu`/`dequantize_v_tile_gpu`
  of the **actual GPU D1 artifact** (chained — D1 fail ⇒ D2=CASCADE) vs CPU codec
  dequant. Artifact = bf16 K/V tiles. Oracle: tiles ULP (tol 64).
- **Input:** synthetic, pinned, **seed 20260901**, 2 tiles K + 2 tiles V, D=256 G=64.
  (NOT a live prompt — the real pinned-prompt TextContext path is M1+/blocked.)

### 2.5 Measured (inherent GPU-vs-CPU offset — NOT a regression)

- D1: codes byte-diff = **1**, scales max|Δ| = **76 ULP**.
- D2: tiles max|Δ| = **24 ULP** (bf16 rounding + fp re-association).
- The 4-iter vs 16-iter Sinkhorn difference is **input-dependent**; the compiled GPU
  kernel behaves like ~16-iter on the gate input, so the **16-iter CPU codec** is the
  reference (a 4-iter copy is garbage: 63274 code bytes / 11.8M ULP off).

### 2.6 The lockstep rule (docs/128 §2.5) — applies to M1 too

Any kernel change that touches a phase's implementation MUST re-run the gate and, if
the artifact diverges with accuracy preserved, update the reference/tolerance in the
SAME commit with a byte-diff A/B record. The gate is a ratchet, not a snapshot.

---

## 3. CONSTRAINTS (read these before touching anything)

1. **GPU is shared with the main working agent — used off and on (the real constraint).**
   The gate's GPU use cases are quick (5–10 s per phase, <10 s for the D1–D7 chain), so
   work around the main agent rather than competing with it: check `nvidia-smi` (two
   16 G GPUs) and `ss -tln` (port 8091) first, and **only run the gate when the GPU is
   idle** — i.e. when the main agent is not using it. If the GPU is busy, do CPU-only
   work (e.g. the D4–D7 CPU reference) and retry the GPU run later. Do not disrupt the
   main agent's serve or builds.
2. **No Qwen model artifact.** `find` for `.safetensors`/`*draft*head*` returns only
   ComfyUI models. D3 (W8G32 draft head) and D8–D10 (e2e tokens, acceptance, A2)
   **cannot run** without the model. Mark them blocked, don't stub them silently.
3. **Build pattern.** Prefer `tools/bench/build_phase_gate.sh` (links the prebuilt
   `libninfer_ops.a`, ~40 s) — it's fast and doesn't touch the main agent's build. A
   full CMake build is also feasible if needed, but unnecessary for the gate.
4. **Do not build into `wo-kv-uniform/build`** (main agent's active build) or
   `wo-kv-uniform-ci-gate` (CI-only). All M0/M1 work lives on `wo/phase-gate`.
5. **Push to `github`, not `origin`.** `origin` = `/home/intel/.../backups/ninfer.git`
   (does not exist). `github` = `git@github.com:chrisconcepcion/dual_5060_ti_ninfer.git`.

---

## 4. M1 scope (docs/128 §6) + what is actually unblocked

Docs/128 M1 = **full decode chain D1–D7**, accept: (a) known-good D1–D7 PASS;
(b) the **08-31 decay bug** localizes at D1 or D2; (c) chain <10 s.

| Phase | What it is | Status on this machine |
|-------|-----------|------------------------|
| D1, D2 | done (M0) | ✅ runnable |
| D3 | MTP draft head (W8G32 → 3 drafts); `test_draft_head.py` pattern | 🔴 blocked (needs model weights) |
| D4 | QK (slice kernel Q@Kᵀ, fp32 scores) | ⬜ CPU ref unblocked; GPU wiring needs cache setup |
| D5 | softmax + online rescale (fp32 probs) | ⬜ CPU ref unblocked |
| D6 | PV + per-tile scales + fp32 accumulation | ⬜ CPU ref unblocked |
| D7 | reduce (split-K / merge) | ⬜ CPU ref unblocked |
| D8–D10 | acceptance / GDN / A2 e2e | 🔴 blocked (needs model) |

**The 08-31 decay bug is already localizable by M0** (it was a D1/D2 keypair
bug → fails at D1 or D2 today). M1's incremental value is localizing *future*
attention-path (D4–D7) and draft (D3) regressions.

### 4.1 The unblocked M1 task: the D4–D7 CPU reference (the "long pole")

Write a **pure-CPU reference** that mirrors the packed decode kernel exactly, so the
GPU kernel's D4–D7 output can be gated against it (byte-identity where the fp32
accumulation order matches, else FP64 + pinned tolerance). Reuse the dormant
`tests/kvarn_codespace_qk_cpu_ref.cpp` (docs/83) as the algebraic-identity starting
point — it already proves code-space ≡ dequant-space QK (Test A) and end-to-end
attention (Test B). Extend it to:
1. take the **real KVarN scale layout** (field-innermost, 1152 fields per
   (layer,head,page); see §5.2) instead of per-page `kc0/kzp/sr`;
2. cover the **online-softmax rescale** across 64-key tiles (D5) and the
   **split-K merge** order (D7);
3. emit per-phase artifacts (scores = D4, probs = D5, accumulated = D6, final = D7)
   so the gate can localize to a single phase, not just the final output.

Then wire the GPU **packed decode kernel** (`gqa_attention_kvarn_decode_packed_kernel`,
see §5.1) on a **synthetic paged KV cache** (codes + scales from D1/D2, a synthetic
Q, a block_table) and gate its output against this reference. This is the part that
needs care — see §6.

### 4.2 Blocked M1 work (do not fake these)

- **D3 draft head:** needs the W8G32 draft head weights. Blocked on model artifact.
- **D8–D10:** needs the full e2e path (model + TextContext + acceptance/A2). Blocked.
- Record these in the gate table as `BLOCKED (model artifact missing)`, not `PASS`.

---

## 5. Kernel conventions (captured for the reference — ground truth)

Source: `src/ops/kernel/gqa_attention_kvarn.cuh` (packed decode
`gqa_attention_kvarn_kernel` + `kvarn_dequant_k`/`kvarn_dequant_v`/
`kvarn_softmax_tile`) and the launcher `src/ops/launcher/gqa_attention_kvarn.cu`.

### 5.1 Geometry + the packed decode kernel (one CTA per (q_head, token))

- `D = kKvarnAttnD = 256` (channels/head dim), `G = kKvarnAttnG = 64` (keys per page).
- K codes = 4-bit (`kKvarnBitsK`, 8192 B/page), V codes = 2-bit (`kKvarnBitsV`, 4096 B/page).
- **Q layout** (bf16): `q[d + D*(q_head + q_heads*tq)]` — i.e. `[D][q_heads*tokens]`,
  channel-major. The Q is used **as-is** (no in-kernel Q rotation in this kernel; the
  K is dequantized to original domain, so the dot is the plain `q · K_original`).
- **Per page** `lp` (grid over `packed_pages`, `block_table[lp]` = physical page,
  `key_base = lp*64`, `key = key_base + j`):
  1. **Dequant K** (`kvarn_dequant_k`): `X[d][j] = (code[d][j]*s_col[d] + zp[d]) *
     s_row[j]` for d∈[0,256), j∈[0,64), then **FWHT over d** (`kvarn_fwht_channel`,
     norm `rsqrt(256)=1/16`). K codes packed `[d][j/2]` (2 4-bit codes/byte, low nibble
     first). `X` is `[D][G]` in smem (the **original-domain** K).
  2. **QK (D4):** `score[j] = scale * Σ_d q[d]*X[d][j]`, **causal mask** `key<=pos`
     else `-inf` (`pos = positions[tq]`). Dot reduced by warp-shuffle
     (`kvarn_dot_reduce`).
  3. **Softmax (D5):** online, per 64-key tile: `tile_max`, `exp(score-tile_max)`,
     `tile_sum`; rescale `acc *= alpha=exp(m_i-mx)`, `l_i = l_i*alpha + tile_sum*beta`,
     `m_i = mx` (`kvarn_softmax_tile`).
  4. **Dequant V** (`kvarn_dequant_v`): `X[i][j] = (code[j][i]*s_row[j] + zp[j]) *
     s_col[i]` (i=D, j=G), then **FWHT over i**, norm 1/16. V codes packed `[j][i/4]`
     (4 2-bit codes/byte). `X` is `[D][G]` original-domain V.
  5. **PV (D6):** `pv = Σ_j score[j]*X[d][j]` (dot over the 64 keys), `acc += pv*beta`.
- **Tail** (uncommitted keys, `tail_count`, bf16 `tail_k`/`tail_v` already post-FWHT):
  same QK/softmax/PV, applied after the packed pages.
- **Reduce (D7):** `out[d] = acc * (1/l_i)` → bf16, layout `[D][q_heads*tokens]`.
- **Split-K variant** (`gqa_attention_kvarn_decode_split_kernel` +
  `gqa_attention_kvarn_decode_merge_kernel`): pages split across `splits`, partials
  `pm/pl/pa` (fp32, `[split][kv][tq][group](+dim)`) merged by the merge kernel. The
  **default decode route is UNIFIED** (`gqa_decode_slice4_kvarn_kernel` + shared
  reduce, `NINFER_KVARN_DECODE` unset); `=packed` is the A/B escape hatch. The split
  route is a documented known-bad escape hatch.

### 5.2 Scale layout (field-innermost, critical for the reference)

`kvarn_scale_at(table, layer, head, page, field, n_layers, n_heads, n_pages) =
table[field + 1152*(layer + n_layers*(head + n_heads*page))]`. 1152 fields per
(layer,head,page), contiguous. Field offsets (per head, per page):
- K: `s_col` = `d` (0–255), `zp` = `256+d`, `s_row` = `512+j` (j 0–63).
- V: `s_col` = `576+i` (i 0–255), `zp` = `832+j`, `s_row` = `1088+j`.

The launcher pre-offsets the scale pointer by `layer*1152` and passes the
`1152*n_layers` page/head stride to the unified kernel.

### 5.3 `PagedKVLayerView` (what the launcher needs)

`src/core/paged_kv_cache.h` — `k_pages`, `v_pages`, `k_scale_pages`, `v_scale_pages`,
`kvarn_scale_pages` (the 1152-field table, `[layers, kv_heads, pages, 1152]`),
`block_table`, `head_dim`, `num_kv_heads`, `dtype`, `quant_group`, `kvarn_layer`.
`GqaKvarnTail` (`include/ninfer/ops/gqa_attention.h`): `k_tile`, `v_tile`,
`staged_k/v`, `packed_pages`, `tail_count`, lane pointers. **Setting this up
correctly for a synthetic cache is the main wiring effort** (see §6).

---

## 6. Next steps (ordered, actionable for the next agent)

1. **GPU check before any GPU run:** `nvidia-smi` + `ss -tln`; run the gate only when
   the GPU is idle (the main working agent uses it off and on — its build in
   `wo-kv-uniform/build` is live and must not be touched).
2. **Build the D4–D7 CPU reference** (pure C++, no GPU): start from
   `tests/kvarn_codespace_qk_cpu_ref.cpp`, rewire it to the **real scale layout**
   (§5.2), add the **online-rescale** (D5) and **split-K merge** (D7) order, and
   emit per-phase artifacts (scores/probs/acc/final). Validate the algebraic
   identity (code-space ≡ dequant-space) standalone first.
3. **Wire the GPU packed decode kernel** on a synthetic paged KV cache:
   - build `k_pages`/`v_pages` (codes) + `kvarn_scale_pages` (1152-field table) from
     the D1 artifact; a `block_table`; a synthetic `q` `[D][q_heads*tokens]`;
   - call `gqa_attention_kvarn_decode_packed_kernel` + `gqa_attention_kvarn_decode_merge_kernel`
     (or the unified route) directly;
   - gate the final bf16 output (+ optionally the dumped scores/probs) vs the CPU ref.
   - **The `packed_pages`/tail pitfall** (docs/128 §8.1, launcher comment): if
     `tail.packed_pages < 0` the unified route zeroes the output (m=-inf); always set
     it `>= 0`.
4. **Add D4–D7 to the gate table** in `tests/phase_gate.cu` with their own shas +
   correctness oracles; keep D1/D2 as-is. Update `phase_baseline.json` +
   `phase_tolerance.json` (lockstep, §2.6).
5. **Mark D3/D8–D10 `BLOCKED (model artifact missing)`** in the table; do not stub.
6. **Negative test for M1:** a mutation that corrupts the QK (e.g. wrong scale field
   offset, or a score ×(1+1e-3)) must FAIL at D4 (not D2, not D7). Extend
   `--mutate-*` flags accordingly.
7. **CI:** extend `tests/CMakeLists.txt` (already has the M0 entry) if the reference
   grows into its own target; the gate stays one binary.
8. **When the model artifact becomes available:** add the pinned-prompt
   TextContext path (docs/128 §8.1/§8.2), D3 via `test_draft_head.py`, and D8–D10;
   then the gate runs the *real* chain instead of the synthetic input.

---

## 7. Gotchas / lessons learned (so they are not re-learned)

- **Hand-rolled SHA-256 had a subtle bug** → use OpenSSL `libcrypto` (`-lcrypto`),
  not a from-scratch implementation. Keep the `phasegate::Sha256` API
  (`hash/hash2/hex/self_test`).
- **The self-test vector was mistyped** (`...f20e17b5`); correct is
  `...f20015ad` for `sha256("abc")`. Verify with `sha256sum` before trusting.
- **4-iter CPU ref ≠ the GPU kernel.** The compiled kernel behaves ~16-iter on the
  gate input; a 4-iter "faithful copy" is byte-identical on simple inputs but 63k
  codes off on the pinned input. Use the 16-iter CPU codec as the reference.
- **Byte-parity (sha ratchet) is the gate for byte phases** (docs/128 §8.4,
  tolerance=0); the CPU codec is the *correctness oracle* that proves the baseline
  is right (a broken-but-stable baseline would otherwise pass sha forever). The 76
  ULP GPU-vs-CPU difference is irrelevant to the ratchet.
- **`--json` mode**: human-readable output must go to **stderr**, pure JSON to
  **stdout** (the baseline file is the raw `--json` output). `--compare` parses that
  nested format (top-level `prompt_sha256/ntiles_k/ntiles_v/seed`, then `D1{...}`,
  `D2{...}`).
- **`build/` is gitignored** → the build script lives in `tools/bench/`, not `build/`.
- **Don't `set -e` in `phase_gate.sh`** around the binary: the gate *exits non-zero on
  a real failure by design*.
- **GPU is byte-stable** across runs — confirmed by two clean `--json` runs giving
  identical D1/D2 shas. That is what makes the ratchet valid.

---

## 8. Quick reference (measured + verified)

- Build: `tools/bench/build_phase_gate.sh` (~40 s, prebuilt libs).
- Clean: D1 PASS `4c9821bd85385d12` (1 byte / 76 ULP), D2 PASS `d272364e22065333`
  (24 ULP) → **exit 0**.
- Negative (`--mutate-d2`): D1 PASS, D2 FAIL → **exit 2**.
- `--compare`: MATCH baseline → exit 0.
- prompt_sha256 (synthetic seed 20260901): `7cf4b3a65e26c1e877c9ee35c96c71f916b7b50302b0893aaa0eb10bc838fbc2`.
- Git: branch `wo/phase-gate`, HEAD `1c26d5b1`, pushed to `github`.
