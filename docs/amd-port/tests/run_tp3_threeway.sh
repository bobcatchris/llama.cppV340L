#!/usr/bin/env bash
# run_tp3_threeway.sh - TP3 three-way MTP VRAM + served perf arm (one rep per call).
#
# Arms (all -c 10000, canonical launch config otherwise: ub512, q4_0 KV, FA, -ngl 999):
#   A (off)   - MTP OFF, dies 0,1,2
#   B (noflag) - MTP in-split draft on the 3 serving dies
#   C (iso)   - MTP via dedicated die: HIP_VISIBLE_DEVICES=0,1,2,3 + --device
#               ROCm0,ROCm1,ROCm2 --spec-mtp-device ROCm3 (ROCm names on this HIP build)
#
# Usage:
#   run_tp3_threeway.sh --arm off|noflag|iso --rep N [--idle-wait S]
#
# Lane: PORT=8083 exclusively. Campaign GPU boot lock held per boot through teardown.
#
# Outputs under docs/amd-port/results/:
#   tp3way_<arm><rep>_<stamp>_server.log     boot + request log
#   tp3way_<arm><rep>_<stamp>_vram.log       2s per-die VRAM CSV with #PHASE markers
#   tp3way_<arm><rep>_<stamp>_battery.jsonl  guard battery receipt (baseline_tp3_200k.json)
#   vram_tp3way_<arm><rep>_<stamp>_<phase>.json  raw rocm-smi snapshots
#   tp3way_<arm><rep>_<stamp>_phases.txt     human phase timestamps
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULTS="$REPO_ROOT/docs/amd-port/results"
ARM_BASELINE="$HERE/baseline_tp3_200k.json"
BIN="$REPO_ROOT/build-hip/bin/llama-server"
MODEL="/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf"
PORT=8083
LOCK=/tmp/campaign_gpu_boot.lock
DESK="TP3-threeway"
ARM=""
REP=""
IDLE_WAIT=180
CTX=10000

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --rep) REP="$2"; shift 2 ;;
    --idle-wait) IDLE_WAIT="$2"; shift 2 ;;
    *) echo "ERROR: unknown arg $1"; exit 1 ;;
  esac
done

if [ -z "$ARM" ] || [ -z "$REP" ]; then
  echo "ERROR: --arm and --rep are required"; exit 1
fi
if [ ! -x "$BIN" ]; then
  echo "ERROR: server binary not found at $BIN"; exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
PREFIX="$RESULTS/tp3way_${ARM}${REP}_${STAMP}"
SERVER_LOG="$PREFIX""_server.log"
VRAM_LOG="$PREFIX""_vram.log"
BATTERY="$PREFIX""_battery.jsonl"
PHASES="$PREFIX""_phases.txt"
SNAP_BASE="$PREFIX"

SERVER_PID=""
SAMPLER_PID=""

note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$PHASES"; }
mark() { echo "#PHASE $1 $(date +%s) $(date +%s.%N)" >> "$VRAM_LOG"; note "PHASE $1"; }

# --- campaign GPU boot lock: check-and-wait, then hold -----------------------
lock_holder_alive() {
  # empty file with old mtime = stale (no holder); content with live pid = alive
  local info pid
  info="$(cat "$LOCK" 2>/dev/null)" || return 1
  [ -z "$info" ] && return 1
  pid="$(printf '%s' "$info" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("pid",""))' 2>/dev/null || true)"
  [ -n "$pid" ] && [ -d "/proc/$pid" ]
}

wait_lock() {
  local waited=0
  while [ -f "$LOCK" ]; do
    if lock_holder_alive; then
      echo "[lock] held by: $(cat "$LOCK") - waiting ($waited s so far)"
    else
      echo "[lock] STALE (empty or dead holder): $(cat "$LOCK" 2>/dev/null | head -c 200) - taking over"
      break
    fi
    sleep 20; waited=$((waited + 20))
    if [ "$waited" -ge 1800 ]; then
      echo "[lock] waited ${waited}s with a live holder - aborting this rep"; exit 3
    fi
  done
  printf '{"desk":"%s","arm":"%s%s","ts":"%s","duration_min":15,"pid":%d}\n' \
    "$DESK" "$ARM" "$REP" "$(date -Iseconds)" "$$" > "$LOCK"
  note "LOCK held by $DESK arm=$ARM$REP pid=$$"
}

release_lock() { rm -f "$LOCK"; note "LOCK released"; }

run_battery() {
  python3 "$HERE/guard_battery.py" --port $PORT --baseline "$ARM_BASELINE" \
    --server-log "$SERVER_LOG" --output-jsonl "$BATTERY" "$@"
  echo "$? battery_exit" >> "$PHASES"
}

snapshot() {
  local phase="$1" file="$SNAP_BASE"_"$1".json
  rocm-smi --showmeminfo vram --json > "$file" 2>/dev/null
  echo "[vram] $phase ($file):"
  python3 - "$file" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for c in ('card0','card1','card2','card3'):
    cd = d.get(c, {})
    t = cd.get('VRAM Total Memory (B)'); u = cd.get('VRAM Total Used Memory (B)')
    if t and u:
        t, u = int(t), int(u)
        print(f'  {c}: used {u/1048576:.1f} / free {(t-u)/1048576:.1f} MiB')
    else:
        print(f'  {c}: n/a')
EOF
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
  local pids
  pids="$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u || true)"
  for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
  release_lock
  echo "[teardown] done."
}
trap teardown EXIT INT TERM

# --- thermal settle gate: edge <= 35C on cards 0-2 ---------------------------
thermal_gate() {
  local waited=0 t0 t1 t2
  while true; do
    read -r t0 t1 t2 < <(rocm-smi --showtemp --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
vals=[]
for c in ('card0','card1','card2'):
    try: vals.append(float(d[c]['Temperature (Sensor edge) (C)']))
    except Exception: vals.append(0.0)
print(' '.join(str(v) for v in vals))
" || echo "0 0 0")
    note "thermal edge card0/1/2 = ${t0}/${t1}/${t2} C"
    if python3 -c "import sys; sys.exit(0 if max($t0,$t1,$t2) <= 35.0 else 1)"; then
      note "thermal gate PASS (<=35C)"
      return 0
    fi
    if [ "$waited" -ge 600 ]; then
      note "thermal gate TIMEOUT after ${waited}s (proceeding, edge ${t0}/${t1}/${t2})"
      return 0
    fi
    sleep 30; waited=$((waited + 30))
  done
}

note "=== tp3way arm=$ARM rep=$REP stamp=$STAMP ctx=$CTX port=$PORT idle_wait=$IDLE_WAIT ==="
git -C "$REPO_ROOT" log -1 --format='tree HEAD %h %ci' >> "$PHASES" 2>/dev/null

wait_lock
thermal_gate

python3 "$HERE/vram_sampler.py" --out "$VRAM_LOG" --interval 2 &
SAMPLER_PID=$!

mark preboot
snapshot preboot

note "[boot] arm=$ARM launching llama-server -> $SERVER_LOG"
case "$ARM" in
  off)
    HIP_VISIBLE_DEVICES=0,1,2 "$BIN" -m "$MODEL" \
      -ngl 999 -sm tensor -c $CTX -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  noflag)
    HIP_VISIBLE_DEVICES=0,1,2 "$BIN" -m "$MODEL" \
      -ngl 999 -sm tensor -c $CTX -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --port $PORT -t 8 --verbose > "$SERVER_LOG" 2>&1 < /dev/null &
    ;;
  iso)
    HIP_VISIBLE_DEVICES=0,1,2,3 "$BIN" -m "$MODEL" \
      --device ROCm0,ROCm1,ROCm2 \
      -ngl 999 -sm tensor -c $CTX -b 512 -ub 512 \
      -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
      --spec-mtp-device ROCm3 \
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
note "BOOT_READY t=$(date +%s)"

sleep 8
mark boot_ready
snapshot boot_ready

mark probe_battery
run_battery --idle-wait "$IDLE_WAIT" || true

sleep 5
mark postprobe
snapshot postprobe

mark teardown_start
if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
  pkill -P "$SERVER_PID" 2>/dev/null || true
  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  pkill -9 -P "$SERVER_PID" 2>/dev/null || true
fi
SERVER_PID=""

# wait for VRAM drain back near idle before the post-teardown snapshot
for _ in $(seq 1 30); do
  used=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
u=[int(d.get(c,{}).get('VRAM Total Used Memory (B)',0))/1048576 for c in ('card0','card1','card2','card3')]
print(int(max(u)))
")
  [ "$used" -le 100 ] && break
  sleep 2
done
mark postteardown
snapshot postteardown

if [ -n "$SAMPLER_PID" ] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
  kill "$SAMPLER_PID" 2>/dev/null || true
  SAMPLER_PID=""
fi
mark session_end

note "[done] arm=$ARM rep=$REP; artifacts:"
note "  $SERVER_LOG"
note "  $VRAM_LOG"
note "  $BATTERY"
note "  $PHASES"
exit 0
