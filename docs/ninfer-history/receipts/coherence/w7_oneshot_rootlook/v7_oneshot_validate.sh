#!/usr/bin/env bash
# v7_oneshot_validate.sh — WO_ONESHOT_ROOTLOOK cure-gate validation.
# Phase 1: 5 armed boots (NINFER_TP_ONESHOT_AR=1 + NINFER_AR_DEADLINE_MS=60000),
#          bound 600 s each; verdict per boot = warmup completes (listening line).
# Phase 2: one fresh boot, same env, count600 mt=600 probe (the WO_DECODE_LANDING
#          ARMCNTLD2 probe0 ordinal) -> decode t/s vs the 32.7 t/s reference +-2%.
# Usage: bash v7_oneshot_validate.sh <bin_path> <label_prefix> [boots=5] [skip_probe=0]
set -u
BIN=${1:?bin}
PFX=${2:?label prefix}
NBOOTS=${3:-5}
SKIPPROBE=${4:-0}
OUT=/home/chris/worktrees/amd-wo-w7-body/results/amd/coherence/w7_oneshot_rootlook
REPRO=$OUT/w7_oneshot_repro.sh
mkdir -p "$OUT"

PASS=0; FAIL=0
for i in $(seq 1 "$NBOOTS"); do
  echo "=== validation boot $i/$NBOOTS ==="
  R=$(NINFER_AR_DEADLINE_MS=60000 bash "$REPRO" "$BIN" "${PFX}_V$i" 600 | tail -4)
  echo "$R" | tail -2
  if echo "$R" | grep -q "VERDICT.*=CLEAN"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
done
echo "GATE PHASE1: warmup $PASS/$NBOOTS (need 5/5)"

if [ "$SKIPPROBE" != "1" ]; then
  echo "=== phase 2: decode t/s probe (fresh boot) ==="
  export NINFER_ALLOW_NVFP4_TP2=1 NINFER_WORKSPACE_MIB=96
  export NINFER_DRAFT_VOCAB=/home/chris/dual_5060_ti_ninfer/tests/multi_gpu/data/qwen38_draft_vocab_ids.json
  export NINFER_MTP_TAIL_ASYNC=1 NINFER_H2D_PINNED_STAGE=1
  export NINFER_VOCAB_COUNT_DIR=/home/chris/vocab_counts
  export NINFER_TP_ONESHOT_AR=1 NINFER_AR_DEADLINE_MS=60000
  pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve" 2>/dev/null; sleep 2
  for i in $(seq 1 30); do
    BUSY=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "VRAM%\): [0-9]+" | grep -oE "[0-9]+$" | sort -rn | head -1)
    [ "${BUSY:-0}" -le 2 ] && break; sleep 2
  done
  LOG=$OUT/${PFX}_probe_serve.log
  setsid nohup "$BIN" /home/chris/models/qwen3_8_27b_nvfp4.ninfer --port 8100 --devices 0,1,2,3 \
    --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 --greedy \
    --default-max-tokens 16 --spec mtp --draft-tokens 2 --allow-nvfp4-weights > "$LOG" 2>&1 < /dev/null &
  echo "probe boot pid=$! $(date -Ins)"
  for i in $(seq 1 200); do
    sleep 3
    grep -q "listening on http" "$LOG" 2>/dev/null && break
    grep -q "AR-WEDGE-WATCHDOG\|AR-FAILOUT" "$LOG" 2>/dev/null && break
  done
  grep -q "listening on http" "$LOG" || { echo "PROBE BOOT FAILED — no listening"; grep -E "AR-WEDGE|AR-FAIL" "$LOG" | tail -3; exit 1; }
  S=$(date +%s.%N)
  curl -s -m 1200 http://127.0.0.1:8100/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Count from 1 to 500, one number per line."}],"max_tokens":600,"temperature":0}' \
    -o $OUT/${PFX}_probe.json
  E=$(date +%s.%N)
  python3 - "$E" "$S" "$PFX" "$OUT" <<'EOF'
import json, sys
E, S, PFX, OUT = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
p = json.load(open(f"{OUT}/{PFX}_probe.json"))
ch = p["choices"][0]; u = p.get("usage", {})
wall = E - S; gen = u.get("completion_tokens", 0)
tps = gen / wall if wall else 0
print(f"PROBE: wall={wall:.2f}s gen={gen} t/s={tps:.1f} finish={ch.get('finish_reason')} (ref 32.7 +-2% = 32.05..33.35)")
EOF
  pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve" 2>/dev/null
fi
echo "VALIDATION COMPLETE"
