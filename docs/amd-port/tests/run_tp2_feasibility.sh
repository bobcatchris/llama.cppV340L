#!/usr/bin/env bash
# run_tp2_feasibility.sh - TP2+MTP@200k feasibility arms with per-die VRAM capture.
#
# Arms:
#   off    - TP2 (dies 0,1), 200k, MTP disabled                  -> baseline per-die VRAM
#   noflag - TP2 (dies 0,1), 200k, draft-mtp ON, draft on serving dies (historical config,
#            expects the PLOG-098/099 fit refusal at first request)
#   flag   - TP2 (dies 0,1) + draft die 2 via --spec-mtp-device  -> the headline arm
#
# Usage:
#   run_tp2_feasibility.sh --arm off|noflag|flag --rep N [--idle-wait S] [extra battery args...]
#
# Outputs under docs/amd-port/results/:
#   tp2feas_<arm><rep>_<stamp>_server.log    boot + request log
#   tp2feas_<arm><rep>_<stamp>_vram.log      2s per-die VRAM CSV with #PHASE markers
#   tp2feas_<arm><rep>_<stamp>_battery.jsonl guard battery receipt
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULTS="$REPO_ROOT/docs/amd-port/results"
ARM_BASELINE="$HERE/baseline_tp2_200k.json"
BIN="$REPO_ROOT/build-hip/bin/llama-server"
MODEL="/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf"
PORT=8080
ARM=""
REP=""
IDLE_WAIT=60
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --rep) REP="$2"; shift 2 ;;
    --idle-wait) IDLE_WAIT="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [ -z "$ARM" ] || [ -z "$REP" ]; then
  echo "ERROR: --arm and --rep are required"; exit 1
fi
if [ ! -x "$BIN" ]; then
  echo "ERROR: server binary not found at $BIN"; exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
PREFIX="$RESULTS/tp2feas_${ARM}${REP}_${STAMP}"
SERVER_LOG="$PREFIX""_server.log"
VRAM_LOG="$PREFIX""_vram.log"
BATTERY="$PREFIX""_battery.jsonl"

SERVER_PID=""
SAMPLER_PID=""


run_battery() {
  local args=()
  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then args=("${EXTRA_ARGS[@]}"); fi
  python3 "$HERE/guard_battery.py" --port $PORT --baseline "$ARM_BASELINE" \
    --server-log "$SERVER_LOG" --output-jsonl "$BATTERY" "$@"
}

free_port() {
  local pids
  fuser -k -s "127.0.0.1:$PORT/tcp" 2>/dev/null || true
  pids="$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u || true)"
  for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
  for _ in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0
    sleep 1
  done
}

teardown() {
  if [ -n "$SAMPLER_PID" ] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill "$SAMPLER_PID" 2>/dev/null || true
  fi
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

mark() {
  echo "#PHASE $1 $(date +%s) $(date +%s.%N)" >> "$VRAM_LOG"
}

free_port

echo "[sampler] starting per-die VRAM sampler -> $VRAM_LOG"
python3 "$HERE/vram_sampler.py" --out "$VRAM_LOG" --interval 2 &
SAMPLER_PID=$!

echo "[boot] arm=$ARM rep=$REP launching llama-server -> $SERVER_LOG"
case "$ARM" in
  off)
    HIP_VISIBLE_DEVICES=0,1 "$BIN" -m "$MODEL" \
      --device ROCm0,ROCm1 \
      -ngl 999 -sm tensor -c 200000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  noflag)
    HIP_VISIBLE_DEVICES=0,1 "$BIN" -m "$MODEL" \
      --device ROCm0,ROCm1 \
      -ngl 999 -sm tensor -c 200000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  t2off10k)
    HIP_VISIBLE_DEVICES=0,1 "$BIN" -m "$MODEL" \
      --device ROCm0,ROCm1 \
      -ngl 999 -sm tensor -c 10000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  t2noflag10k)
    HIP_VISIBLE_DEVICES=0,1 "$BIN" -m "$MODEL" \
      --device ROCm0,ROCm1 \
      -ngl 999 -sm tensor -c 10000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  t2flag10k)
    HIP_VISIBLE_DEVICES=0,1,2 "$BIN" -m "$MODEL" \
      --device ROCm0,ROCm1 \
      -ngl 999 -sm tensor -c 10000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --spec-mtp-device ROCm2 \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  t2flag8k)
    HIP_VISIBLE_DEVICES=0,1,2 "$BIN" -m "$MODEL" \
      --device ROCm0,ROCm1 \
      -ngl 999 -sm tensor -c 200000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --spec-mtp-device ROCm2 \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  *) echo "ERROR: unknown arm '$ARM'"; exit 1 ;;
esac
SERVER_PID=$!

echo -n "[boot] waiting for health (timeout 420s)"
READY=0
for i in $(seq 1 84); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "[ERROR] server exited during boot; log tail:"
    tail -30 "$SERVER_LOG"
    exit 1
  fi
  if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    echo " -> ready in ~$((i * 5))s"
    READY=1
    break
  fi
  echo -n "."
  sleep 5
done
if [ "$READY" != "1" ]; then
  echo ""
  echo "[ERROR] boot timeout; log tail:"
  tail -30 "$SERVER_LOG"
  exit 1
fi

sleep 8
mark boot_ready
snapshot() {
  echo "[vram] $1 snapshot (total / used / free MiB):"
  rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
for c in ('card0','card1','card2','card3'):
    cd=d.get(c,{})
    t=cd.get('VRAM Total Memory (B)'); u=cd.get('VRAM Total Used Memory (B)')
    if t and u:
        t,u=int(t),int(u)
        print(f'  {c}: total {t/1048576:.0f} / used {u/1048576:.1f} / free {(t-u)/1048576:.1f} MiB')
    else:
        print(f'  {c}: n/a')
"
}
snapshot boot-ready

case "$ARM" in
  off)
    mark probe_prefill
    run_battery --idle-wait "$IDLE_WAIT" --prefill-only
    mark probe_decode
    run_battery --idle-wait 0 --decode-only
    ;;
  noflag)
    mark probe_first_request
    echo "[probe] sending first request (expecting the historical fit refusal)..."
    run_battery --idle-wait 0 --prefill-only || true
    sleep 5
    mark post_first_request
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[surprise] server SURVIVED the first request - checking health and continuing to decode probe"
      if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        mark probe_decode
        run_battery --idle-wait 0 --decode-only || true
      fi
    else
      echo "[refusal] server died at first request, as historically expected. Log tail:"
      tail -40 "$SERVER_LOG"
    fi
    ;;
  flag)
    mark probe_prefill
    run_battery --idle-wait "$IDLE_WAIT"
    ;;
  t3on)
    ARM_BASELINE="$HERE/baseline_tp3_200k.json"
    HIP_VISIBLE_DEVICES=0,1,2 "$BIN" -m "$MODEL" \
      --device ROCm0,ROCm1,ROCm2 \
      -ngl 999 -sm tensor -c 200000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  t3flag)
    ARM_BASELINE="$HERE/baseline_tp3_200k.json"
    HIP_VISIBLE_DEVICES=0,1,2,3 "$BIN" -m "$MODEL" \
      --device ROCm0,ROCm1,ROCm2 \
      -ngl 999 -sm tensor -c 200000 -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --spec-mtp-device ROCm3 \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  t2off10k)
    mark probe_prefill
    run_battery --idle-wait "$IDLE_WAIT" --prefill-only || true
    mark probe_decode
    run_battery --idle-wait 0 --decode-only || true
    ;;
  t3on | t3flag)
    mark probe_prefill
    run_battery --idle-wait "$IDLE_WAIT" || true
    ;;
  t2noflag10k | t2flag10k | t2flag8k)
    mark probe_first_request
    run_battery --idle-wait 0 --prefill-only || true
    sleep 3
    mark probe_decode
    if kill -0 "$SERVER_PID" 2>/dev/null && curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
      run_battery --idle-wait 0 --decode-only || true
    else
      echo "[refusal] server died at first request. Log tail:"
      tail -40 "$SERVER_LOG"
    fi
    ;;
esac

sleep 5
mark session_end
if kill -0 "$SERVER_PID" 2>/dev/null; then
  snapshot end-of-session
fi

echo "[done] arm=$ARM rep=$REP; artifacts:"
echo "  $SERVER_LOG"
echo "  $VRAM_LOG"
[ -f "$BATTERY" ] && echo "  $BATTERY"
exit 0
