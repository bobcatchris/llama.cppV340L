# Verification runbook — post-decay-fix test gaps (agent handoff)

Created 2026-09-01. The decode-decay fix (cp.async staging etc.) is landed and guard-verified,
but four verification gaps remain. This runbook closes them. Work happens in
`/home/intel/ninfer/worktrees/wo-kv-uniform` (branch `wo/kv-uniform`, HEAD `3c58dfad`).

## 0. HARD COORDINATION RULES (read first — another agent is working in this tree)

1. **DO NOT rebuild anything in `build/`** and do NOT run `cmake` there. The main agent is
   actively editing `src/` for MultiBatch work. The existing binaries in `build/` are CURRENT:
   they were built at `61dcecd8`, and the only later commit touching code (`a1ef0283`) is a
   comment-only edit. Run tests; never build.
2. **A KVarN server is RUNNING on port 8091** (TP2, both GPUs, fixed binary, capacity 250256):
   log `/home/intel/ninfer/logs/serve_postfix_20260901_040659.log`. Do NOT kill it. Do NOT run
   `decode_guard.sh --all-cache-type` (it `fuser -k`s port 8091 and spawns its own server —
   this is the documented collision failure mode from docs/125).
3. **GPU exclusivity**: Tiers 2 and 3 need the GPUs to themselves (they load their own model).
   Ask the main agent to stop the 8091 server and pause GPU work before starting them; resume
   after. Tier 1 is small-allocation and safe to run alongside the server.
4. Ignore stray binaries `build/tests/ninfer_gqa_kvarn_window64_repro*` — their sources were
   deleted; they are stale artifacts.
5. Record everything you run + raw outputs under `results/` or `/tmp/verify_*/` and report
   against the pass criteria below. Any FAIL: stop, save logs, report — do not "fix" source.

## Tier 1 — unit battery against the existing build (safe concurrent; ~5 min)

```bash
cd /home/intel/ninfer/worktrees/wo-kv-uniform/build/tests
for t in ninfer_slice4_kvarn_test ninfer_kvarn_gqa_test \
         ninfer_kvarn_materialize_oracle_test ninfer_slice4_kvarn_dequant_test \
         ninfer_kvarn_write_path_test ninfer_kvarn_codec_test ninfer_kvarn_codec_edge_test \
         ninfer_kvarn_layout_test ninfer_kvarn_budget_test ninfer_kvarn_tile_cuda_test \
         ninfer_slice2_byteid_test ninfer_slice3_i8_test ninfer_slice3_i8_dequant_test \
         ninfer_gqa_attention_test; do
  printf "%-40s " "$t"; ./$t >/tmp/verify_$t.out 2>&1 && tail -1 /tmp/verify_$t.out || { echo "FAIL"; tail -5 /tmp/verify_$t.out; }
done
```
PASS criteria: every line ends in PASS / "all checks passed" / equivalent success token.
`ninfer_kvarn_gqa_test` MUST pass on the DEFAULT route (no `NINFER_KVARN_DECODE` env set) —
that was one of the bugs fixed (`ac6b3f61`). Also run the kernel bench sanity:
```bash
cd /home/intel/ninfer/worktrees/wo-kv-uniform/build/tests
SELFTEST64=1 ./ninfer_slice4_kvarn_bench            # nonzero acc, no FAIL
./ninfer_slice4_kvarn_bench 160256 2>&1 | grep "T=4"  # expect ~0.94 ms (was 2.11 pre-fix)
```

## Tier 2a — regress_unified.sh (GPU-EXCLUSIVE; coordinate first; ~10 min)

The unified-kernel harness (docs/104): bf16/int8/kvarn x mtp{0,1}, checks A2 identity
(mtp1 == plain) and token-for-token match vs the bf16 greedy anchor
(`results/bf16_plain_ref.txt`). This is the "EXIT 0 is not a pass" harness.

```bash
# after the 8091 server is stopped:
cd /home/intel/ninfer/worktrees/wo-kv-uniform
bash tools/regress_unified.sh 2>&1 | tee /tmp/verify_regress_unified.log
echo "rc=$?   # 0=PASS, 1=FAIL, 2=env"
```
PASS: `RESULT: PASS` with no SKIP lines (a SKIP means the bf16 anchor file went missing).
Note: kvarn is NOT byte-identical to packed (32- vs 64-key reduction widths) — the harness
compares against the BF16 reference at token level, which is the accepted contract.

## Tier 2b — decode_guard 25k/80k cells (uses the RUNNING server; run when main agent is not GPU-busy)

The 10k/40k/160k/250k cells were re-measured post-fix (74.8/67.7/53.7/46.4); 25k and 80k were
not. Single-server mode against the live fixed-binary server:

```bash
cd /home/intel/ninfer/worktrees/wo-kv-uniform
CASES="25000 80000" ITERS=1 COMP=192 TEMP=0 BASE=http://127.0.0.1:8091 \
  CTX_LOG=/home/intel/ninfer/logs/serve_postfix_20260901_040659.log \
  bash tools/bench/decode_guard.sh ab_cells 2>&1 | tee /tmp/verify_ab_cells.log
```
PASS: JSON lines with `server_decode_tps` ≈ **25k: 70-73** (baseline 68.3) and
**80k: 61-64** (baseline 58.2), `mtp_accept_pct` within ±5pp of baseline (75.0 / 73.7).
If the numbers look noisy, re-run once — decode t/s is per-step and contention-sensitive.

## Tier 3 — byte-diff A/B: keypair dequant vs pre-keypair build (GPU-EXCLUSIVE; ~40 min)

Claim to verify: `ac6b3f61`'s keypair K-dequant + prefetch-reorder are bit-identical to
`c631f7f1` (same FMA order; only the LDS/unpack schedule changed). Procedure:

```bash
# 1. Build the OLD commit in a SEPARATE worktree (never touch the main build dir):
cd /home/intel/ninfer/worktrees/wo-kv-uniform
git worktree add /tmp/ab-c631f7f1 c631f7f1
cd /tmp/ab-c631f7f1 && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a && cmake --build build -j8

# 2. Capture from the NEW server (running on 8091) BEFORE stopping anything:
mkdir -p /tmp/verify_bytediff
for P in "Explain what a lighthouse keeper does." "Write a haiku about GPUs." "List three prime numbers."; do
  curl -s http://127.0.0.1:8091/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":64,\"temperature\":0}" \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'])" \
    >> /tmp/verify_bytediff/new.out
done
# ALSO capture a long greedy cell: reuse the guard's 10k prompt generator:
UNIT="The lighthouse keeper logged every ship that passed the harbor. "
python3 -c "import sys; n=int(10000*2625/31524); print(('${UNIT}'*n)+'\nQuestion: How many ships did he log on Monday?')" > /tmp/verify_bytediff/p10k.txt
curl -s http://127.0.0.1:8091/v1/chat/completions -H 'Content-Type: application/json' \
  -d "$(python3 -c "import json;print(json.dumps({'model':'qwen3.8-27b','messages':[{'role':'user','content':open('/tmp/verify_bytediff/p10k.txt').read()}],'max_tokens':64,'temperature':0}))")" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'])" >> /tmp/verify_bytediff/new.out

# 3. Coordinate: stop the 8091 server. Start the OLD-commit server on the SAME args:
cd /tmp/ab-c631f7f1
setsid nohup build/apps/ninfer-serve /home/intel/models/qwen3_8_27b.ninfer --port 8091 \
  --devices 0,1 --spec mtp --draft-tokens 3 --kv-dtype kvarn_k4v2 \
  --kv-capacity 250256 --max-context 250256 > /tmp/verify_bytediff/serve_old.log 2>&1 < /dev/null &
# wait for "listening" in the log, then repeat step 2's captures into old.out, then:
diff /tmp/verify_bytediff/new.out /tmp/verify_bytediff/old.out && echo "BYTE-IDENTICAL" \
  || python3 /home/intel/ninfer/worktrees/wo-kv-uniform/tools/bench/byte_diff.py \
       /tmp/verify_bytediff/new.out /tmp/verify_bytediff/old.out

# 4. Cleanup: kill the old server, tell the main agent to restart the fixed-binary server:
#    cd ~/ninfer/worktrees/wo-kv-uniform && setsid nohup build/apps/ninfer-serve \
#      /home/intel/models/qwen3_8_27b.ninfer --port 8091 --devices 0,1 --spec mtp \
#      --draft-tokens 3 --kv-dtype kvarn_k4v2 --kv-capacity 250256 --max-context 250256 \
#      > ~/ninfer/logs/serve_postfix_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
git worktree remove /tmp/ab-c631f7f1 --force
```
PASS: `BYTE-IDENTICAL` (empty diff). A near-tie divergence at one token is NOT auto-fail —
report the byte_diff position/context and let the main agent judge against the accepted-
tolerance contract (docs/105 §3).

## Reporting

Summarize per tier: command, rc, key numbers, log paths. Update the standing baseline ONLY if
told to (`decode_guard_check.py --ratchet` is an owner decision). Do not push anything.

NOTE: `tools/ops/run_ci.sh --full` (build + int8/KVarN smoke batteries @250k, ~35+ min) is the
FINAL gate and is the MAIN agent's step after the MultiBatch work lands — it runs `cmake --build`
itself and must NOT be run by the verification agent while `src/` is being edited.
