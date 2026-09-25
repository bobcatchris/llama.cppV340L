# HANDOFF — A1 lane: gzenz NVFP4 backport (docs/155), steps 1-5 DONE

**Branch:** `wo/nvfp4-kv` (base `wo/kvarn-multibatch` @ 4ada1561). Work order:
`repo/docs/155_gzenz_backport_agent_work_order.md` (commit 6c7d8463 on main).
**Source fork:** `/tmp/gzenz_ninfer` (read-only, tip 4b882d0).

## DONE (all device-verified)

| Step | Commit | What |
|---|---|---|
| 1 | `c5fa5306` | Tier plumbing: enum/marker/name-table, 4-plane U8 layout (codes D/2 + E4M3 scales D/16, quant_group 16), decoder_spec arm, budget 9792 B/token/rank (= q4_0, −47% vs int8), BF16-reserve exclusion, refusal guard |
| 2 | `c62e6b4b` | CPU reference codec (`src/ops/kernel/nvfp4_kv_codec_ref.h`) + golden vectors — ALL PASS. This header is THE oracle; device kernels are cross-checked against it |
| 3 | `d4ae1c9d` | Fused quantize-append fill kernel + H256 Hadamard port (`nvfp4_hadamard_d256.cuh`); V planes byte-exact vs CPU oracle; crafted exact-lattice K rows byte-exact post-rotation |
| 4-5 | `5838e408` | Prompt flash kernel (dequant-staged K/V + in-smem Q rotation) + slice7 unified decode kernel (fused append with rotated K, DEFAULT split policy) + launcher/wrapper wiring; **refusal FLIPPED — nvfp4 is servable**. Decode T=1 + prefill T=64 vs CPU reference: ALL PASS |

Test binaries: `build/tests/ninfer_nvfp4_{kv_codec_ref,kv_append_test,attention_test}`,
`ninfer_tp2_budget_test`. **GPU protocol honored throughout** (3 grant windows used,
released after each — verified 15 MiB).

## STEPS 6-8 STATUS (2026-09-06 ~19:0xZ)

- **Step 6 DONE** (`04308a6e`): MTP order gate on NVFP4 — S2 EXIT=0 at 80000/90000,
  role-keyed acceptance 0.60/0.53 (int8: 0.60/0.67). FAKE-PASS lesson: the FIRST
  exit=0 was a stale-binary error-JSON empty-compare (bf16-handoff §5 trap) — voided;
  always grep the run log for KeyError/error before trusting a diag verdict.
- **Step 7 DONE** (`2babed40`): needle 3/3 probes (combined 5/5 @ 127,158-token prompt,
  spot25, spot90). 127k prompts SERVABLE at 13,087 MiB/rank — impossible on int8/bf16.
  Harness: tools/smoke/diag/nvfp4_needle.sh (counter print is cosmetic-buggy — subshell;
  PASS/FAIL lines are the record). Fixture calibration: this filler class is 2.15
  chars/token (NOT the 5.33 launch-doc figure — measured via usage.prompt_tokens).
- **Step 8 PARTIAL** (`2babed40`): VRAM evidence committed (decoder state 13.6-13.8
  KB/token incl. metadata, ~30% under int8's 19.6). **run_ci --full DEFERRED on disk**
  (relink needs ~7G; 2.9G free). When disk allows: `bash tools/ops/run_ci.sh --full`
  from the worktree root, then the closeout report (§8 item 6).

## ENVIRONMENT NOTES (cost real time — do not repeat)

- **Disk is the binding constraint (2.7G free, 99%).** Full builds with BUILD_TESTING=ON
  relink 60+ test binaries at ~210 MB each (they statically embed libninfer_ops.a's
  cubins) → 8+ GB. Build with `--target <specific>`; strip big test binaries if disk
  drops (`find build/tests -maxdepth 1 -type f -size +20M | xargs strip` helps little —
  the weight is cubins, delete unneeded binaries instead). `-j 4` per coordinator.
- **`__shfl_xor_sync(FullMask, ...)` must run BEFORE any lane `continue`s** — a
  divergent sync silently corrupts the exchange (every odd code slot saturated to 6.0).
- **Test-side indexing bug pattern:** KV row layout is `d + kD*(h + heads*t)` — token
  stride is `kD*kv_heads`, NOT `kD`. A kD-stride "crafted row" lands across heads and
  fakes a kernel bug.
- **`getenv()` does not work in device code** — probe kernels via host env + extra
  kernel args, or raw launches with pulled buffers.
- **Python `str.replace` on source files silently no-ops on whitespace mismatch** —
  assert `old in src` for every patch (this bit twice: macro arm, test q1 upload).
- Raw-kernel probe (direct launch + partials inspection) is the fastest
  wrapper-vs-kernel isolation tool; see `tests/test_nvfp4_attention.cu` raw-probe block.
- doc 155's §3/§4 anchors were verified against this tree before implementation.

## Who's who

- **coordinator** (intercom): GPU grants (3 used, all honored), disk management.
- **A2** (`agent2`): dflash2 lane — no overlap with the NVFP4 surface.
- **gemini** (agent_comm mesh): owns the CI cell flip + formal tests (S2 re-gate).
