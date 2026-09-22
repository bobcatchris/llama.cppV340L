#!/usr/bin/env bash
# voltage_step.sh - one full step: apply profile -> boot -> battery -> 10-min soak -> teardown.
# Usage: echo '<pw>' | voltage_step.sh <stepname> <mode> [mV]
# Appends: docs/amd-port/results/voltage_sweep_summary.txt
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$HERE/../../results"
TESTS="$HERE"
read -r PW
STEPNAME="${1:-step}"
MODE="${2:-verify}"
MV="${3:-}"
SUM="$RESULTS/voltage_sweep_summary.txt"
BIN="$HERE/../../../build-hip/bin/llama-server"
MODEL="/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf"
PORT=8080

apply() {
  local card="$1" sclk="$2" mvv="$3"
  printf '%s\n' "$PW" | sudo -S sh -c "echo 's 7 $sclk $mvv' > /sys/class/drm/$card/device/pp_od_clk_voltage" 2>/dev/null
}

# 1) apply profile
printf '%s\n' "$PW" | "$HERE/sweep_voltage.sh" "$MODE" "$MV" > /tmp/vapply_$STEPNAME.log 2>&1
grep -E "applied|reverted" /tmp/vapply_$STEPNAME.log | head -1
S7=$(for c in card1 card3 card0 card4; do printf '%s\n' "$PW" | sudo -S sed -n '9p' /sys/class/drm/$c/device/pp_od_clk_voltage 2>/dev/null; done | sort -u | tr '\n' ' ')
echo "readback sclk7: $S7"

# die-idle wait: dies are shared; never boot over a foreign server
for _ in $(seq 1 60); do
  USED=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
u=[int(d.get(f'card{c}',{}).get('VRAM Total Used Memory (B)',0))/1048576 for c in (0,1)]
print(sum(1 for x in u if x > 200))")
  [ "$USED" = "0" ] && break
  echo "[dies busy] waiting 30s"
  sleep 30
done
fuser -k -s 127.0.0.1:$PORT/tcp 2>/dev/null; sleep 2

# 2) boot B-arm (TP2 in-split @10k)
HIP_VISIBLE_DEVICES=0,1 "$BIN" -m "$MODEL" --device ROCm0,ROCm1 \
  -ngl 999 -sm tensor -c 10000 -b 512 -ub 512 \
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
  --port $PORT -t 8 > /tmp/vserver_$STEPNAME.log 2>&1 < /dev/null &
SPID=$!
READY=0
for i in $(seq 1 60); do
  curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { READY=1; break; }
  kill -0 $SPID 2>/dev/null || break
  sleep 3
done
if [ "$READY" != "1" ]; then
  echo "$(date '+%F %T') step=$STEPNAME mode=$MODE mv=$MV BOOT_FAILED" >> "$SUM"
  echo "[boot failed]"; kill -9 $SPID 2>/dev/null; exit 1
fi

# 3) battery: prefill + decode (accept/canary data included in decode receipt)
python3 "$TESTS/guard_battery.py" --port $PORT --baseline "$TESTS/baseline_tp2_200k.json" \
  --server-log /tmp/vserver_$STEPNAME.log --output-jsonl /tmp/vbat_$STEPNAME.jsonl \
  --idle-wait 0 --prefill-only > /tmp/vbat1_$STEPNAME.log 2>&1
python3 "$TESTS/guard_battery.py" --port $PORT --baseline "$TESTS/baseline_tp2_200k.json" \
  --server-log /tmp/vserver_$STEPNAME.log --output-jsonl /tmp/vbat_$STEPNAME.jsonl \
  --idle-wait 0 --decode-only > /tmp/vbat2_$STEPNAME.log 2>&1
PP=$(grep -oE "prompt_tps +[0-9.]+" /tmp/vbat1_$STEPNAME.log | head -1 | grep -oE "[0-9.]+")
DECODE=$(grep -oE "decode_tps +[0-9.]+" /tmp/vbat2_$STEPNAME.log | head -1 | grep -oE "[0-9.]+")
ACCEPT=$(python3 -c "
import json
try:
    for line in open('/tmp/vbat_$STEPNAME.jsonl'):
        r=json.loads(line)
        for c in r['results']:
            if c.get('cell')=='decode_8k_10k': print(c.get('accept_ratio'))
except Exception: pass" | head -1)

# 4) 10-min decode soak with temp sampling
(
  END=$(( $(date +%s) + 600 ))
  REQ='{"prompt":"Summarize the key architectural findings of the V340L campaign in detail:","n_predict":256,"temperature":0.0,"cache_prompt":false}'
  while [ "$(date +%s)" -lt "$END" ]; do
    curl -s -m 120 "http://127.0.0.1:$PORT/completion" -d "$REQ" > /dev/null 2>&1
  done
) &
SOAK_PID=$!
(
  END=$(( $(date +%s) + 610 ))
  while [ "$(date +%s)" -lt "$END" ]; do
    date +%s
    rocm-smi --showtemp 2>/dev/null | grep -E "junction|edge" | grep -oE "[0-9]+\.[0-9]" | tr '\n' ' '
    echo ""
    sleep 5
  done
) > /tmp/vsoak_$STEPNAME.log &
TPID=$!
wait $SOAK_PID
kill $TPID 2>/dev/null
JUNC_MAX=$(grep -oE "[0-9]+\.[0-9]" /tmp/vsoak_$STEPNAME.log | sort -n | tail -1)
EDGE_MAX=$(python3 -c "
vals=[float(x) for x in open('/tmp/vsoak_$STEPNAME.log').read().split()]
print(max(vals))" 2>/dev/null)
SOAK_TPS=$(grep -c "eval time" /tmp/vserver_$STEPNAME.log)

echo "$(date '+%F %T') step=$STEPNAME mode=$MODE mv=$MV sclk7=[$S7] pp=$PP decode=$DECODE accept=$ACCEPT soak=600s junc_max=$JUNC_MAX profile_max_mV_in_table=$(printf '%s\n' "$PW" | sudo -S sed -n '9p' /sys/class/drm/card1/device/pp_od_clk_voltage 2>/dev/null | grep -oE '[0-9]+mV')" >> "$SUM"
echo "[step done $STEPNAME] pp=$PP decode=$DECODE accept=$ACCEPT junc_max=$JUNC_MAX"

kill $SPID 2>/dev/null; sleep 3; kill -9 $SPID 2>/dev/null
exit 0
