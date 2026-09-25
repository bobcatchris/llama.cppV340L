#!/usr/bin/env bash
# run_tp3_guards.sh - Batch runner for llama.cpp TP3 200k served regression guards.
#
# House Testing Law:
#   llama-bench is NOT our method - we boot llama-server with the config of record
#   (/home/chris/launch_tp3_200k.sh) and run served guards across prefill and decode.
#
# Usage:
#   ./run_tp3_guards.sh              # Full PCIe boot (3-4 min) + guard battery
#   ./run_tp3_guards.sh --no-boot    # Run guards against an already running server
#   ./run_tp3_guards.sh --ratchet    # Raise baseline if measured results exceed it
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PORT="${PORT:-8080}"
BOOT_SCRIPT="${BOOT_SCRIPT:-/home/chris/launch_tp3_200k.sh}"
NO_BOOT=false
RATCHET=false
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --no-boot)
      NO_BOOT=true
      shift
      ;;
    --ratchet)
      RATCHET=true
      shift
      ;;
    --boot-script)
      BOOT_SCRIPT="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/docs/amd-port/results"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server_tp3_200k_${STAMP}.log"
RECEIPT_FILE="$LOG_DIR/tp3_guards_${STAMP}.jsonl"

SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  [teardown] Stopping llama-server (PID $SERVER_PID and children)..."
    pkill -P "$SERVER_PID" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    pkill -9 -P "$SERVER_PID" 2>/dev/null || true
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  free_port "$PORT"
  echo "  [teardown] llama-server stopped and port $PORT released."
}
trap cleanup EXIT INT TERM

free_port() {
  local p="$1"
  local pids
  fuser -k -s "127.0.0.1:$p/tcp" 2>/dev/null || true
  pids="$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u || true)"
  for pid in $pids; do
    kill -9 "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ":$p " || return 0
    sleep 1
  done
}

wait_server_ready() {
  local p="$1"
  local pid="$2"
  echo -n "  [boot] Waiting for llama-server on port $p (timeout: 360s)"
  for i in $(seq 1 72); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo ""
      echo "  [ERROR] Server process exited unexpectedly during boot. Check $SERVER_LOG"
      return 1
    fi
    if curl -s -m 5 "http://127.0.0.1:$p/health" 2>/dev/null | grep -q '"status":"ok"'; then
      echo " -> Ready in ~$((i * 5))s."
      return 0
    fi
    echo -n "."
    sleep 5
  done
  echo ""
  echo "  [ERROR] Timeout waiting for server readiness. Check $SERVER_LOG"
  return 1
}

echo "=== llama.cpp TP3 200k Served Regression Guard ==="
echo "Timestamp: $(date)"
echo "Config of Record: $BOOT_SCRIPT"

if [ "$NO_BOOT" = false ]; then
  if [ ! -f "$BOOT_SCRIPT" ]; then
    echo "ERROR: Boot script $BOOT_SCRIPT not found!"
    exit 1
  fi

  # Port conflict guard
  free_port "$PORT"

  echo "  [boot] Launching server via $BOOT_SCRIPT (log -> $SERVER_LOG)..."
  bash "$BOOT_SCRIPT" > "$SERVER_LOG" 2>&1 < /dev/null &
  SERVER_PID=$!

  wait_server_ready "$PORT" "$SERVER_PID"
else
  echo "  [no-boot] Assuming llama-server is already running on port $PORT."
  if ! curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    echo "ERROR: Server at http://127.0.0.1:$PORT is not reachable or not healthy!"
    exit 1
  fi
fi

# Run the Python guard battery
BATTERY_CMD=(
  python3 "$HERE/guard_battery.py"
  --port "$PORT"
  --output-jsonl "$RECEIPT_FILE"
)

if [ -f "$SERVER_LOG" ]; then
  BATTERY_CMD+=(--server-log "$SERVER_LOG")
fi

if [ "$RATCHET" = true ]; then
  BATTERY_CMD+=(--ratchet)
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  BATTERY_CMD+=("${EXTRA_ARGS[@]}")
fi

echo ""
echo "--> Executing Guard Battery..."
set +e
"${BATTERY_CMD[@]}"
EXIT_CODE=$?
set -e

echo ""
echo "Guard Battery completed with exit code: $EXIT_CODE"
if [ -f "$RECEIPT_FILE" ]; then
  echo "JSONL Receipt banked at: $RECEIPT_FILE"
fi

exit $EXIT_CODE
