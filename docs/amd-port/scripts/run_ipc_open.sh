#!/usr/bin/env bash
# W38 ONE-SHOT-AR DESK: staged IPC-open feasibility probe (~30 s GPU).
# Closes the one mechanism W31 never tested (hipIpcGet/OpenMemHandle +
# device read through the opened pointer, hipMemcpyPeerAsync round trip)
# before the one-shot AR desk is finally buried or revived.
# Laws: boot lock check-and-hold (JSON line desk/arm/ts/pid), single sleeps
# up to 150 s, cool-die gate < 60 C, PID-targeted teardown, prompt release,
# line-buffered capture (W31 session 1 lost buffered output to the watchdog).
set -u
WT=/media/chris/ssd128/llamacpp/wt-oneshot-ar
BIN="$WT/docs/amd-port/probes/ipc_open_probe"
RES="$WT/docs/amd-port/results"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RES/W38_ipc_open_$STAMP.log"
LOCK=/tmp/campaign_gpu_boot.lock
TIMEOUT=300

die_hot() {  # prints max die junction temp (C)
  local c h t m=0
  for c in card0 card1 card3 card4; do
    h=$(for hw in /sys/class/hwmon/hwmon*; do [ "$(cat $hw/name 2>/dev/null)" = amdgpu ] && [ "$(basename $(readlink -f $hw/device))" = "$(basename $(readlink -f /sys/class/drm/$c/device))" ] && echo $hw && break; done)
    t=$(($(cat $h/temp2_input 2>/dev/null || echo 0)/1000)); [ "$t" -gt "$m" ] && m=$t
  done
  echo $m
}
die_idle() {  # all four dies < 200 MiB vram used
  local c v
  for c in card0 card1 card3 card4; do
    v=$(cat /sys/class/drm/$c/device/mem_info_vram_used 2>/dev/null || echo 999999999)
    [ "$v" -ge 200000000 ] && return 1
  done
  return 0
}

echo "== W38 IPC-open probe $STAMP =="
if [ ! -x "$BIN" ]; then
  echo "BUILD: $BIN missing, building"
  hipcc -O2 -x hip "$WT/docs/amd-port/probes/ipc_open_probe.cu" -o "$BIN" \
    && echo "BUILD-EXIT:0" || { echo "BUILD-EXIT:$?"; exit 1; }
fi

# --- lock check-and-hold ---
WAITED=0
while [ -f "$LOCK" ]; do
  HOLDER=$(cat "$LOCK" 2>/dev/null)
  HPID=$(echo "$HOLDER" | sed -n 's/.*"pid":\([0-9]*\).*/\1/p')
  if [ -n "$HPID" ] && ! kill -0 "$HPID" 2>/dev/null; then
    echo "LOCK: stale holder pid $HPID dead, removing with evidence: $HOLDER"
    rm -f "$LOCK"
    break
  fi
  echo "LOCK: held, waiting ($WAITED s): $HOLDER"
  sleep 150
  WAITED=$((WAITED+150))
done
while ! die_idle; do echo "GATE: dies busy, waiting"; sleep 60; done
while [ "$(die_hot)" -ge 60 ]; do echo "GATE: dies hot ($(die_hot) C), waiting"; sleep 30; done
printf '{"desk":"oneshot-ar","arm":"W38-ipc-open","ts":"%s","pid":%d}\n' \
  "$(date +%s)" "$$" > "$LOCK"
echo "LOCK: held by pid $$"

# --- run (line-buffered, watchdog inside the probe, kill by PID) ---
stdbuf -oL -eL "$BIN" --ranks 4 --timeout "$TIMEOUT" > "$LOG" 2>&1 &
PROBE_PID=$!
wait "$PROBE_PID"; RC=$?

cat "$LOG"
echo "PROBE-EXIT:$RC log=$LOG"

# --- release ---
rm -f "$LOCK"
echo "LOCK: released"
exit $RC
