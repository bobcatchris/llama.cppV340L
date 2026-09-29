# docs/157 P4 — NVFP4 qtype tag-trace (lane-prep for the tp_gemv consumer sweep)

**Author:** agent2 · **Date:** 2026-09-08 20:53Z (coordinator-ordered finalization of intercom tag-note)
**Purpose:** exactly what qtype/geometry each consumer of an NVFP4-weighted tensor sees,
end-to-end, on TP2. Whoever extends `tp_gemv` (A1, user-greenlit sweep) must make his NVFP4
ARM agree with the tags below — they are emitted correctly TODAY; nothing in placement
(tp_load.cpp / my lane) needs to change for any fix shape.

## The tag chain (manifest -> switch)
1. **Manifest**: `NumericFormat::NVFP4` per tensor (artifact census: 247 NVFP4 objects).
2. **Bind** (`bindings.cpp bind_nvfp4_weight`): `WeightPlan{.format=NVFP4,
   weight_scale_divisor_bits, input_scale_divisor_bits}` — divisors READ FROM PAYLOAD at bind
   time (weight_divisor word inside the parent tensor + FP32 input_scale_divisor sidecar).
3. **Placement** (`tp_load.cpp materialize_tp pass-2 -> nvfp4_shard_image`): emits RAW BYTES
   ONLY — codes [N,K/2] + SWIZZLED scales + 4-byte divisor at LOCAL dims. Placement carries NO
   tag by contract (MaterializedArtifact is metadata-free; see A1 seq-7 "RUNTIME CONTRACT").
   => placement bytes can never disagree with the qtype switch: the tag is stamped downstream.
4. **Descriptor** (`bindings.cpp materialized_weight` NVFP4 arm ~:131-145):
   `Weight{qtype=QType::NVFP4, group_size=16, payload_bytes=block_scale_geometry(local).encoded,
   qdata=base, scales=base+scale_plane_offset, n=local_N, k=local_K}` — computed from the SAME
   local (rows, columns) the loader wrote. **So the switch ALREADY sees QType::NVFP4 correctly
   at decode; F1 is a MISSING ARM, not a missing tag.**
5. **Consumers today**:
   - `tp_kernel.cu tp_gemv`: `t>8` -> `ops::linear(x, weight, out, stream)` at :40, BEFORE the
     switch (never reaches NVFP4 case even if added below); `t<=8` -> switch handles ONLY
     Q4/Q5/Q6/W8 (int-family SIMT gemv) -> `default: throw "tp_gemv: unsupported qtype"` = **F1**.
   - 4-arg `ops::linear` has NO workspace arg; NVFP4 dispatch needs workspace + policy
     (`linear.cpp QType::NVFP4 -> nvfp4_dispatch(..., policy, workspace, stream)`). The t>8
     escape therefore cannot serve NVFP4 as written — candidate mechanisms for F2
     (`cudaErrorIllegalAddress`, kvarn_workspace.cpp:594): policy/workspace default routing
     around, or a rows=full vs rows=local geometry write. F2 causality stays TBD per report;
     re-run after the F1+dispatch fix before treating F2 as separate.

## What the fix must match (frozen facts from my battery)
- Slot planes at LOCAL dims are byte-exact vs the real exporter (11/11 placements + 4/4
  contract rejections + gqkv4-pair disjoint-assembly audit green;
  `tests/multi_gpu/nvfp4_shard_host.cpp`). Any consumer arm can assume:
  `qdata == codes[local_N][local_K/2]`, `scales == swizzled per layouts.py with local
  k_tiles`, `divisor == last 4 bytes`, `local_N % 128 == 0`, `local_K % 64 == 0`.
- Registered local problems (A1 02089ed4): 7168x5120 / 8192x5120 / 17408x5120 / 5120x3072 /
  5120x8704 — the GEMV geometry templates exist; the tp_gemv arm just needs to route
  `QType::NVFP4` to them (t<=8), and t>8 needs an AR-aware prefill call with workspace
  threaded (likely widen tp_gemv's signature — call sites: attention_projection_tp,
  gdn_input_projection_tp(+verify), mtp_attention_projection_tp, post_mixer_tp).
- Regression harness for the sweep: `ninfer_nvfp4_shard_host_test` +
  `ninfer_bf16_requant_host_test` must stay green under any change on this branch
  (targeted relink only; disk rule: prune build/tests after).

## F1-verify probe (10-min grant runbook for A1 — one paste)
Pre-flight (CPU): `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
expect `15 MiB, 0 %` x2; branch tip built; then IN GRANTED WINDOW:
```
BINARY=/home/intel/ninfer/worktrees/wo-nvfp4-prefill/build/apps/ninfer-serve
ARTIFACT=/home/intel/ninfer/incoming/nvfp4/qwen3_8_27b_nvfp4.ninfer
export NINFER_ALLOW_NVFP4_TP2=1
"$BINARY" "$ARTIFACT" --port 8099 --devices 0,1 --kv-dtype kvarn_k4v4 \
  --max-context 10240 --prefill-chunk 1024 --spec mtp --draft-tokens 3 \
  > /tmp/f1_serve.log 2>&1 &
SP=$!
for i in $(seq 90); do grep -q listening /tmp/f1_serve.log && break; sleep 2; done
curl -s -m 120 localhost:8099/v1/chat/completions -d @/tmp/p2_body_10k.json | head -c 400
grep -c "tp_gemv: unsupported qtype" /tmp/f1_serve.log || true
kill $SP; sleep 5
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l   # expect 0
```
(ARTIFACT is POSITIONAL per run_wo_g3.sh:135 — flags otherwise match the harness.)
PASS = 0 unsupported-qtype hits + completion text returned + 0 compute apps after teardown.
(Exact flag names per run_wo_g3.sh Cell 2 pre-flight; adjust --max-context to 10240-class for
the minimal surface.)
