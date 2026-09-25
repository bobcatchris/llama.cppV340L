# 62 — Single-GPU (RTX 5090 / PRO 5000) support: scope

Status: SCOPE — static analysis 2026-08-24, no code changes.
Roadmap item #5 (docs/59). Primary target: one 32 GB Blackwell card running
Qwen3.8-27B end-to-end (serve + MTP), so a single machine can run the model
and free the second GPU for other work.

## Verified facts (code, 2026-08-24)

1. **Engine selection already branches** — `make_engine`
   (`src/runtime/engine/engine.cpp:309-314`): `devices.size() == 2` →
   TPEngine, otherwise the plain single-device Engine (ConcurrentExecutor).
   The single-device runtime path exists; it is not exercised by our battery.
2. **TPEngine hard-fails on one device** — ctor defaults to
   `dev0 = 0, dev1 = 1` when `< 2` devices are given
   (`src/runtime/tp2/tp_engine.cpp:271-275`). No single-device path inside it.
3. **Build is sm_120a-only and enforced** — `CMakeLists.txt:6-13` rejects any
   other `CMAKE_CUDA_ARCHITECTURES`. RTX 5060 Ti = GB206 = **sm_120**;
   RTX 5090 / RTX PRO 5000 = GB202 = **sm_100**. The roadmap note "kernels
   unchanged" is WRONG for a 5090 target: the guard must be relaxed and every
   kernel family (GQA, GDN recurrent, MTP speculative round, allreduce-free
   single-device ops) re-verified on sm_100a. Warp-level MMA / cp.async /
   ldmatrix exist on both, so this is expected to be a port + re-baseline, not
   a rewrite — but it is real work, and perf may differ (GB202 prefers
   tcgen05 tensor-core paths).
4. **27B loading is TP-shaped** — the known load path is
   `materialize_tp` (`src/targets/qwen3_6_27b/impl/load/tp_load.h:40`);
   `bindings.cpp` still carries 32 rank/world references, so a non-TP
   materialization plan (keep-all-on-device-0) must be written or verified.
5. **Kernels are parameterized by head count** (num_kv_heads etc.) — full
   heads on one device should work, but the MTP path
   (`src/ops/kernel/speculative_round.cu`) and GDN recurrent state layout have
   not been exercised single-device in this repo. Sibling evidence: the 3090
   machine ran speculative decoding single-GPU (docs/56), so the kernel
   families are capable of it.

## Memory sizing (estimates — verify at implementation)

Per-rank calibrated constants (tp_engine.cpp budget): int8 KV ≈ 18,496
B/token/rank, KVarN ≈ 8,972 B/token/rank. Single GPU holds **both** ranks'
weights and **full** KV + GDN state:

| Component | TP2 per rank | single GPU (32 GB card) |
|---|---|---|
| Weights (Q4 artifact) | ~9.1 GB | 18.2 GB |
| GDN/decoder state | ~2.7 GB (log: "decoder state: 2727 MB") | ~5.4 GB (full heads) |
| KV @ 200k, int8 | ~5.8 GB | ~11.6 GB → **~35 GB total: does not fit** |
| KV @ 200k, KVarN | ~2.9 GB | ~5.8 GB → **~29 GB total: fits, ~3 GB headroom** |
| KV @ 100k, int8 | — | ~5.8 GB → ~29 GB: fits |

Conclusion: **long-context single-GPU requires KVarN (roadmap #1) or the
all-Q4 weights (#7); int8 alone caps around ~100k context on 32 GB.** This is
why #5 is ordered after #1/#4 and why KVarN-first was the right call.

## Work items (when picked up — own work order per docs/99)

1. **CMake arch guard** → allow `100a` (keep 120a default); build matrix note
   in REPO.md. First gate: all unit tests green on the target arch.
2. **Non-TP materialization plan** for qwen3_6_27b (keep-all plan or a
   `world=1` path through `materialize_tp`); load test: weights + GDN state
   land on device 0, sizes match §table.
3. **Single-device run of the full stack**: serve + MTP k=3 + battery
   must-pass subset (T1 T2 T3 T5 T7 T8 T10 T11) with `--devices 0` on a 2-GPU
   machine first (proves the path without new hardware), then on the actual
   card.
4. **Budget model**: single-device per-token constants (2× the per-rank KV
   value; GDN state not sharded); preflight must name the real limit.
5. **Ops**: LAUNCH.md single-GPU section; run_ci.sh/status.sh device-count
   variants (low priority).

## Explicit non-goals

- Not a replacement for TP2 on this machine (TP2 stays the reference config).
- No sharding experiments (that is roadmap #8, multi-GPU TP4+).
- No perf re-baseline until correctness battery is green.
