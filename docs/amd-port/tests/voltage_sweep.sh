#!/usr/bin/env bash
# voltage_sweep.sh - V340L temp program: short 10k experiments to get
# junction temps under control (Chris directive: bunch of short experiments,
# 10k runs; MTP-overhead work proceeds in parallel on a separate desk).
#
# Pin sclk 1200 MHz, keep mclk at 945 MHz (NEVER reduce memory clock),
# undervolt in -25 mV steps, instant revert on instability.
#
# Structure (all at -c 10240, one server boot for the whole sweep):
#   phase 0: stock reference decode cell (no pin, perf auto)  -> thermal + t/s ref
#   phase 1: pin 1200 MHz @ 1125 mV, reference decode cell
#   phase 2: steps 1100..850; per-step gates: server alive, mclk still 945,
#            decode t/s >= 97% of pinned baseline, accept >= 0.63 (when the
#            canary reports), junction < 95 C. Any failure -> stock, stop.
#   phase 3: winner on all four dies, full guard battery (10k), restore.
#
# Mechanism: rewrite OD_SCLK level 5 to 1200 MHz @ <mv> via pp_od_clk_voltage
# (stock level 5 = 1269 MHz @ 1150 mV; stock curve puts 1200 MHz at ~1125 mV),
# commit, force with power_dpm_force_performance_level=manual + level write to
# pp_dpm_sclk. OD_MCLK is never written.
#
# Cards: serving dies are card0/card1/card3 (card2 is the NVIDIA boot display,
# card4 is the draft die - stock until the winner is applied in phase 3).
# Run as root (sudo -S). The invocation supplies credentials; nothing here
# stores them. Server lane: 8081 (validation). /tmp/campaign_gpu_boot.lock is
# checked-and-held by the watcher that launches this script.
set -uo pipefail

PORT=8081
CARDS="card0 card1 card3"
DRAFT_CARD="card4"
BIN=/media/chris/ssd128/llamacpp/wt-tp2-mtp/build-hip/bin/llama-server
MODEL=/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/../results"
TS=$(date +%Y%m%d_%H%M%S)
RLOG="$LOGDIR/voltage_sweep_${TS}.log"
TLOG="$LOGDIR/voltage_sweep_${TS}_temps.log"
BASE_MV=1125
STEPS="1100 1075 1050 1025 1000 975 950 925 900 875 850"
STOCK_S5="s 5 1269 1150"

SRV_PID=""
SAMP_PID=""
DC_TPS=""; DC_ACC=""
declare -A HWMON

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RLOG"; }

map_hwmon() {
  for hw in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$hw/name" 2>/dev/null)" = "amdgpu" ] || continue
    local pci; pci=$(basename "$(readlink -f "$hw/device")")
    for c in $CARDS $DRAFT_CARD; do
      if [ "$(basename "$(readlink -f "/sys/class/drm/$c/device")")" = "$pci" ]; then
        HWMON[$c]="$hw"
      fi
    done
  done
}

junc_c() {  # card -> junction temp (temp2, millidegC)
  local h="${HWMON[$1]:-}"; [ -n "$h" ] || { echo 999; return; }
  echo $(( $(cat "$h/temp2_input" 2>/dev/null || echo 999000) / 1000 ))
}

max_junc() { local m=0 t; for c in $CARDS; do t=$(junc_c "$c"); [ "$t" -gt "$m" ] && m=$t; done; echo "$m"; }

temps_start() { : > "$TLOG"; ( while :; do echo "$(date +%s) $(max_junc)" >> "$TLOG"; sleep 2; done ) & SAMP_PID=$!; }
temps_stop() {  # echoes phase max junction
  kill "$SAMP_PID" 2>/dev/null; wait "$SAMP_PID" 2>/dev/null; SAMP_PID=""
  awk '{if ($2+0 > m) m = $2+0} END {print m+0}' "$TLOG"
}

od_write() {  # card, s5 spec like "s 5 1200 1100"
  local d="/sys/class/drm/$1/device"
  echo manual > "$d/power_dpm_force_performance_level" 2>/dev/null
  echo "$2" > "$d/pp_od_clk_voltage" && echo c > "$d/pp_od_clk_voltage"
}

force_level() { echo 5 > "/sys/class/drm/$1/device/pp_dpm_sclk" 2>/dev/null; }

cur_sclk()  { grep '\*' "/sys/class/drm/$1/device/pp_dpm_sclk" | awk '{print $2}'; }
cur_mclk()  { grep '\*' "/sys/class/drm/$1/device/pp_dpm_mclk" | awk '{print $2}'; }

restore_all() {
  for c in $CARDS $DRAFT_CARD; do
    od_write "$c" "$STOCK_S5"
    echo auto > "/sys/class/drm/$c/device/power_dpm_force_performance_level" 2>/dev/null
  done
  log "RESTORE: stock OD tables + perf level auto on all cards"
}

cleanup() {
  [ -n "$SAMP_PID" ] && kill "$SAMP_PID" 2>/dev/null
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
  restore_all
}
trap cleanup EXIT INT TERM

boot_server() {
  HIP_VISIBLE_DEVICES=0,1,2 "$BIN" -m "$MODEL" --device ROCm0,ROCm1,ROCm2 \
    -ngl 999 -sm tensor -c 10240 -b 512 -ub 512 -ctk q4_0 -ctv q4_0 -fa on \
    --spec-type draft-mtp --port "$PORT" -t 8 \
    >> "$LOGDIR/voltage_sweep_${TS}_server.log" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 120); do
    curl -s "http://127.0.0.1:$PORT/health" | grep -q '"status": *"ok"' && return 0
    sleep 3
  done
  return 1
}

decode_cell() {  # sets DC_TPS / DC_ACC from the guard decode cell
  local out
  out=$(python3 "$HERE/guard_battery.py" --port "$PORT" --decode-only \
    --baseline "$HERE/baseline_tp3_200k.json" --idle-wait 5 \
    --server-log "$LOGDIR/voltage_sweep_${TS}_server.log" \
    --output-jsonl "$LOGDIR/voltage_sweep_${TS}.jsonl" 2>&1 | tee -a "$RLOG")
  DC_TPS=$(echo "$out" | awk '$3=="decode_tps" {print $4}' | head -1)
  DC_ACC=$(echo "$out" | awk '$3=="draft_accept" {print $4}' | head -1)
  DC_TPS=${DC_TPS:-0}; DC_ACC=${DC_ACC:-0}
}

step_ok() {  # $1 = mv, $2 = baseline tps -> gates one step
  local mv="$1" base="$2" c
  for c in $CARDS; do
    log "  $c: sclk=$(cur_sclk "$c") mclk=$(cur_mclk "$c") junc=$(junc_c "$c")C"
  done
  [ "$(max_junc)" -lt 95 ] || { log "  GATE FAIL: junction >= 95 C - instant revert"; return 1; }
  for c in $CARDS; do
    [ "$(cur_mclk "$c")" = "945Mhz" ] || { log "  GATE FAIL: mclk left 945 on $c - revert"; return 1; }
  done
  decode_cell
  log "  decode tps=$DC_TPS accept=$DC_ACC (baseline $base)"
  [ "$DC_TPS" != "0" ] || { log "  GATE FAIL: decode cell produced no t/s - revert"; return 1; }
  awk -v a="$DC_TPS" -v b="$base" 'BEGIN {exit !(a >= 0.97*b)}' || { log "  GATE FAIL: t/s < 97% of baseline - revert"; return 1; }
  if [ "$DC_ACC" != "0" ]; then
    awk -v a="$DC_ACC" 'BEGIN {exit !(a >= 0.63)}' || { log "  GATE FAIL: accept < 0.63 - revert"; return 1; }
  fi
  for c in $CARDS; do
    [ "$(cur_sclk "$c")" = "1200Mhz" ] || log "  WARN: $c sclk=$(cur_sclk "$c") != 1200 under load"
  done
  return 0
}

mkdir -p "$LOGDIR"
: > "$RLOG"
log "V340L temp sweep (short 10k experiments); serving cards: $CARDS; draft: $DRAFT_CARD (stock)"
map_hwmon
for c in $CARDS $DRAFT_CARD; do
  log "$c hwmon=${HWMON[$c]:-MISSING} od_s5=$(awk '/^5:/{print $2"@"$3; exit}' "/sys/class/drm/$c/device/pp_od_clk_voltage" 2>/dev/null)"
done

boot_server || { log "FATAL: server did not come up"; exit 1; }
log "server up on 8081 (10k, in-split MTP)"

# phase 0: stock reference
log "== PHASE 0: stock reference cell =="
temps_start; decode_cell; J0=$(temps_stop)
[ "$DC_TPS" != "0" ] || { log "FATAL: stock reference cell failed"; exit 1; }
STOCK_TPS="$DC_TPS"; STOCK_ACC="$DC_ACC"
log "STOCK: tps=$STOCK_TPS accept=$STOCK_ACC max_junc=${J0} C"

# phase 1: pin 1200 MHz at stock-curve voltage
log "== PHASE 1: pin 1200 MHz @ ${BASE_MV} mV =="
for c in $CARDS; do
  od_write "$c" "s 5 1200 $BASE_MV" || { log "FATAL: OD write failed on $c"; exit 1; }
  force_level "$c"
done
temps_start; decode_cell; J1=$(temps_stop)
[ "$DC_TPS" != "0" ] || { log "FATAL: pinned baseline cell failed - reverting"; exit 1; }
BASE_TPS="$DC_TPS"; BASE_ACC="$DC_ACC"
log "PINNED ${BASE_MV} mV: tps=$BASE_TPS accept=$BASE_ACC max_junc=${J1} C (stock was $STOCK_TPS @ ${J0} C)"

# phase 2: undervolt steps
WINNER="$BASE_MV"; WINNER_TPS="$BASE_TPS"
for mv in $STEPS; do
  log "== STEP ${mv} mV =="
  apply_failed=0
  for c in $CARDS; do
    od_write "$c" "s 5 1200 $mv" || { log "OD write failed on $c"; apply_failed=1; break; }
  done
  if [ "$apply_failed" = "1" ]; then
    log "reverting to winner ${WINNER} mV"
    for c in $CARDS; do od_write "$c" "s 5 1200 $WINNER"; done
    break
  fi
  temps_start
  if step_ok "$mv" "$BASE_TPS"; then
    JSTEP=$(temps_stop)
    WINNER="$mv"; WINNER_TPS="$DC_TPS"
    log "step ${mv} mV STABLE: tps=$WINNER_TPS max_junc=${JSTEP} C"
  else
    JSTEP=$(temps_stop)
    log "step ${mv} mV UNSTABLE (max_junc=${JSTEP} C) - reverted; winner stands at ${WINNER} mV"
    for c in $CARDS; do od_write "$c" "s 5 1200 $WINNER"; done
    break
  fi
done
log "SWEEP RESULT: winner ${WINNER} mV @ 1200 MHz, tps=$WINNER_TPS | stock $STOCK_TPS @ ${J0} C | pinned ref $BASE_TPS @ ${J1} C"

# phase 3: winner on all four dies, full guard battery (10k), restore via trap
log "== PHASE 3: winner on all dies, full guard battery =="
od_write "$DRAFT_CARD" "s 5 1200 $WINNER"
for c in $CARDS; do force_level "$c"; done; force_level "$DRAFT_CARD"
temps_start
python3 "$HERE/guard_battery.py" --port "$PORT" \
  --baseline "$HERE/baseline_tp3_200k.json" --idle-wait 5 \
  --server-log "$LOGDIR/voltage_sweep_${TS}_server.log" \
  --output-jsonl "$LOGDIR/voltage_sweep_${TS}_winner.jsonl" 2>&1 | tee -a "$RLOG" | tail -12
JW=$(temps_stop)
log "winner battery max_junc=${JW} C"
log "DONE - all cards restored by EXIT trap"
