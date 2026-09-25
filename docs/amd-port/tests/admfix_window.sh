#!/usr/bin/env bash
# admfix_window.sh - ADMISSION STARVATION FIX desk (E-090): served validation.
#
# One boot of the FIXED build at 10k-class (-c 10240, campaign in-split line,
# lane 8083), then probe_kv_admission.py in fixed mode (must PASS: defer line
# + both requests 200 + zero exceeded). Optionally a second boot with
# --no-kv-admission for the --control arm (legacy failure signature).
#
# Usage:
#   admfix_window.sh [--skip-control]
#
# Outputs under docs/amd-port/results/:
#   admfix_fixed_<stamp>.log    boot + probe log (fixed build)
#   admfix_control_<stamp>.log  boot + probe log (--no-kv-admission, optional)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULTS="$REPO_ROOT/docs/amd-port/results"
BIN="$REPO_ROOT/build-hip/bin/llama-server"
MODEL="/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf"
PORT=8083
LOCK_FILE="/tmp/campaign_gpu_boot.lock"
DESK_NAME="ADMISSION STARVATION FIX"
SKIP_CONTROL=0
[ "${1:-}" = "--skip-control" ] && SKIP_CONTROL=1

if [ ! -x "$BIN" ]; then
  echo "ERROR: server binary not found at $BIN"; exit 1
fi

# campaign GPU boot lock (cross-desk convention): check-and-hold, 150 s cadence
while [ -e "$LOCK_FILE" ]; do
  echo "[lock] held by: $(cat "$LOCK_FILE" 2>/dev/null | head -1) - waiting 150s"
  sleep 150
done
echo "desk=$DESK_NAME ts=$(date +%s) start=$(date '+%F %T')" > "$LOCK_FILE"

# die-capacity guard: only launch when every needed die is actually idle
for _ in $(seq 1 20); do
  NONIDLE=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
need=[int(d.get('card'+c,{}).get('VRAM Total Used Memory (B)',0))/1048576 for c in '0 1 2'.split()]
print(sum(1 for u in need if u > 200))
" 2>/dev/null)
  [ "$NONIDLE" = "0" ] && break
  echo "[dies] busy (dies 0 1 2 needed; a boot on another port may hold them) - waiting 150s"
  sleep 150
done

SERVER_PID=""
SERVER_LOG=""

free_port() {
  for _ in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0
    sleep 1
  done
}

teardown() {
  rm -f "$LOCK_FILE"
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    pkill -P "$SERVER_PID" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 3
    kill -9 "$SERVER_PID" 2>/dev/null || true
    pkill -9 -P "$SERVER_PID" 2>/dev/null || true
  fi
  free_port
  echo "[teardown] done."
}
trap teardown EXIT INT TERM

boot_server() { # $1 = log path, $2 = extra args
  SERVER_LOG="$1"
  free_port
  echo "[boot] launching llama-server ($2) -> $SERVER_LOG"
  HIP_VISIBLE_DEVICES=0,1,2 "$BIN" -m "$MODEL" \
    --device ROCm0,ROCm1,ROCm2 \
    -ngl 999 -sm tensor -c 10240 -b 512 -ub 512 \
    -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
    $2 \
    --port $PORT -t 8 > "$SERVER_LOG" 2>&1 < /dev/null &
  SERVER_PID=$!

  echo -n "[boot] waiting for health (timeout 420s)"
  local READY=0
  for i in $(seq 1 84); do
    if curl -sf "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
      READY=1; break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo " SERVER DIED"; return 1
    fi
    echo -n "."
    sleep 5
  done
  if [ "$READY" != "1" ]; then echo " HEALTH TIMEOUT"; return 1; fi
  echo " ok"
}

run_probe() { # $1 = arm label, $2 = extra probe args
  echo "[probe] arm=$1"
  python3 "$HERE/probe_kv_admission.py" --port "$PORT" --log "$SERVER_LOG" $2
}

STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- arm 1: fixed build (kv-admission on, default) -------------------------->
FIXED_LOG="$RESULTS/admfix_fixed_${STAMP}.log"
if ! boot_server "$FIXED_LOG" ""; then
  echo "[arm fixed] BOOT FAILED"; exit 1
fi

grep -m1 "kv_unified" "$SERVER_LOG" || echo "[WARN] kv_unified line not found"
grep -m1 "n_parallel" "$SERVER_LOG" || true

run_probe fixed "" | tee -a "$FIXED_LOG"
FIXED_VERDICT="${PIPESTATUS[0]}"
echo "[arm fixed] probe exit=$FIXED_VERDICT"

# ---- arm 2 (optional): control build (--no-kv-admission) ------------------->
CONTROL_VERDICT="skipped"
if [ "$SKIP_CONTROL" != "1" ]; then
  kill "$SERVER_PID" 2>/dev/null || true
  sleep 5
  pkill -9 -P "$SERVER_PID" 2>/dev/null || true
  kill -9 "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  free_port

  CONTROL_LOG="$RESULTS/admfix_control_${STAMP}.log"
  if boot_server "$CONTROL_LOG" "--no-kv-admission"; then
    run_probe control "--control" | tee -a "$CONTROL_LOG"
    CONTROL_VERDICT="${PIPESTATUS[0]}"
    echo "[arm control] probe exit=$CONTROL_VERDICT"
  else
    echo "[arm control] BOOT FAILED"
    CONTROL_VERDICT="boot_failed"
  fi
fi

echo "[window] fixed=$FIXED_VERDICT control=$CONTROL_VERDICT"
[ "$FIXED_VERDICT" = "0" ] || exit 1
exit 0
