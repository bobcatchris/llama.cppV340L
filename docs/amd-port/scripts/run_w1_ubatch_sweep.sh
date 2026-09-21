#!/usr/bin/env bash
# run_w1_ubatch_sweep.sh - Automated interleaved ubatch sweep for W1 milestone.
#
# Protocol per ZCode dispatch:
#   - Control: canonical /home/chris/launch_tp3_200k.sh (ubatch 128)
#   - Variant arms: /home/chris/launch_tp3_200k_ub256.sh, /home/chris/launch_tp3_200k_ub512.sh
#   - Order: interleaved ub256, ub512, ub256, ub512
#   - Per boot: 2 reps (Rep 1: fresh cold-stamped with 180s idle-wait; Rep 2: warm no-boot idle-wait 0)
#   - Bank receipts and markdown summary receipt in docs/amd-port/results/
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PORT=8080
RESULTS_DIR="$REPO_ROOT/docs/amd-port/results"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="$RESULTS_DIR/W1_ubatch_sweep_${STAMP}.md"
mkdir -p "$RESULTS_DIR"

SERVER_PID=""

free_port() {
  local p="$1"
  fuser -k -s "127.0.0.1:$p/tcp" 2>/dev/null || true
  local pids
  pids="$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u || true)"
  for pid in $pids; do
    kill -9 "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 15); do
    ss -ltn 2>/dev/null | grep -q ":$p " || return 0
    sleep 1
  done
}

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  [teardown] Stopping server PID $SERVER_PID..."
    pkill -P "$SERVER_PID" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    pkill -9 -P "$SERVER_PID" 2>/dev/null || true
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  free_port "$PORT"
}
trap cleanup EXIT INT TERM

wait_server_ready() {
  local p="$1"
  local pid="$2"
  local s_log="$3"
  echo -n "  [boot] Waiting for llama-server on port $p..."
  for i in $(seq 1 72); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo ""
      echo "  [ERROR] Server process exited during boot! Check $s_log"
      return 1
    fi
    if curl -s -m 5 "http://127.0.0.1:$p/health" 2>/dev/null | grep -q '"status":"ok"'; then
      echo " Ready in ~$((i * 5))s."
      return 0
    fi
    echo -n "."
    sleep 5
  done
  echo ""
  echo "  [ERROR] Timeout waiting for server readiness."
  return 1
}

echo "=== W1 ubatch sweep: ub256 vs ub512 (Interleaved A/B) ==="
echo "Date: $(date)"
echo "Target: Qwen3.8-27B-ASCII-P1M on 3x AMD V340/MI25 dies"
echo "Results -> $SUMMARY_FILE"
echo ""

BOOTS=(
  "ub256:boot1:/home/chris/launch_tp3_200k_ub256.sh"
  "ub512:boot1:/home/chris/launch_tp3_200k_ub512.sh"
  "ub256:boot2:/home/chris/launch_tp3_200k_ub256.sh"
  "ub512:boot2:/home/chris/launch_tp3_200k_ub512.sh"
)

# Output summary header
cat > "$SUMMARY_FILE" << EOF
# W1 ubatch Sweep Receipt: ub256 vs ub512 (Interleaved A/B)

Date: $(date -Iseconds)
Branch: $(git -C "$REPO_ROOT" branch --show-current)
Commit: $(git -C "$REPO_ROOT" rev-parse HEAD)
Battery Fingerprint: ab3dfba685c5cfb7

## Control Baseline of Record (ub128)
- Prefill: 93.87 t/s (cold-stamped baseline 96.50 t/s)
- Decode: 15.34 t/s (baseline 15.57 t/s)
- MTP Accept: 0.6667 (gate: >=0.63)
- Batch Size: 512, ubatch: 128

## Sweep Matrix
EOF

echo "| Boot | Arm | Rep | Mode | Prefill t/s | vs Ctrl | Decode t/s | vs Ctrl | MTP Accept | Determ | Needle | Max Edge °C | Thermal Drift | Receipt |" >> "$SUMMARY_FILE"
echo "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |" >> "$SUMMARY_FILE"

BOOT_IDX=0
for entry in "${BOOTS[@]}"; do
  BOOT_IDX=$((BOOT_IDX + 1))
  IFS=":" read -r ARM BOOT_TAG SCRIPT <<< "$entry"
  echo "================================================================================"
  echo ">>> Starting Boot $BOOT_IDX/4: $ARM ($BOOT_TAG) via $SCRIPT"
  echo "================================================================================"

  free_port "$PORT"
  sleep 2

  SERVER_LOG="$RESULTS_DIR/server_${ARM}_${BOOT_TAG}_${STAMP}.log"
  echo "  [boot] Launching server -> $SERVER_LOG"
  bash "$SCRIPT" > "$SERVER_LOG" 2>&1 < /dev/null &
  SERVER_PID=$!

  wait_server_ready "$PORT" "$SERVER_PID" "$SERVER_LOG"

  # Rep 1: Cold-stamped with 180s idle cooldown
  echo ""
  echo "--> Running Rep 1 (Cold-Stamped, --idle-wait 180)..."
  RECEIPT_REP1="$RESULTS_DIR/receipt_${ARM}_${BOOT_TAG}_rep1_cold_${STAMP}.jsonl"
  python3 "$REPO_ROOT/docs/amd-port/tests/guard_battery.py" \
    --port "$PORT" \
    --output-jsonl "$RECEIPT_REP1" \
    --server-log "$SERVER_LOG" \
    --idle-wait 180

  # Rep 2: Warm with --idle-wait 0
  echo ""
  echo "--> Running Rep 2 (Warm, --idle-wait 0)..."
  RECEIPT_REP2="$RESULTS_DIR/receipt_${ARM}_${BOOT_TAG}_rep2_warm_${STAMP}.jsonl"
  python3 "$REPO_ROOT/docs/amd-port/tests/guard_battery.py" \
    --port "$PORT" \
    --output-jsonl "$RECEIPT_REP2" \
    --server-log "$SERVER_LOG" \
    --idle-wait 0

  # Teardown
  echo "  [teardown] Stopping server..."
  pkill -P "$SERVER_PID" 2>/dev/null || true
  kill "$SERVER_PID" 2>/dev/null || true
  sleep 2
  pkill -9 -P "$SERVER_PID" 2>/dev/null || true
  kill -9 "$SERVER_PID" 2>/dev/null || true
  free_port "$PORT"
  SERVER_PID=""
  sleep 3

  # Parse receipts into summary table
  python3 - << PYEOF >> "$SUMMARY_FILE"
import json

ctrl_pp = 93.87
ctrl_dec = 15.34

for rep_tag, r_path in [("Rep1", "$RECEIPT_REP1"), ("Rep2", "$RECEIPT_REP2")]:
    mode = "Cold (180s)" if rep_tag == "Rep1" else "Warm"
    try:
        with open(r_path) as f:
            data = json.loads(f.readline())
        res = {r["guard"]: r for r in data.get("results", [])}
        therm = data.get("thermal", {})
        
        pp = res.get("prefill_guard", {}).get("measured_tps", 0.0)
        pp_d = ((pp - ctrl_pp) / ctrl_pp) * 100.0
        
        dec = res.get("decode_guard", {}).get("measured_tps", 0.0)
        dec_d = ((dec - ctrl_dec) / ctrl_dec) * 100.0
        
        accept = res.get("mtp_canary", {}).get("accept_ratio", 0.0)
        det = "PASS" if res.get("determinism_guard", {}).get("status") == "PASS" else "FAIL"
        needle = f"{res.get('needle_recall_guard', {}).get('recalled_count', 0)}/3"
        
        max_edge = therm.get("max_edge_c", "-")
        drift = f"+{therm.get('thermal_drift_c', 0.0)}°C"
        r_name = "$ARM" + "_" + "$BOOT_TAG" + "_" + rep_tag
        
        print(f"| $BOOT_TAG | $ARM | {rep_tag} | {mode} | {pp:.2f} | {pp_d:+.1f}% | {dec:.2f} | {dec_d:+.1f}% | {accept:.4f} | {det} | {needle} | {max_edge} | {drift} | {r_name} |")
    except Exception as e:
        print(f"| $BOOT_TAG | $ARM | {rep_tag} | {mode} | ERR | - | ERR | - | - | - | - | - | - | error: {e} |")
PYEOF

done

echo ""
echo "=== Sweep Completed Successfully ==="
echo "Summary markdown: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
