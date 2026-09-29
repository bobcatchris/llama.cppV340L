#!/usr/bin/env bash
# dp_uv_guard.sh - DP-1 thermal/revert guard (E-149 hazard law). Run as root in
# a second terminal BEFORE dp_uv_apply.sh --go. Watches every die's HBM temp
# (hwmon temp3 on Vega10 Instinct; verify against the campaign sideband
# sampler's 'mem' column once) and force-reverts ALL dies to stock curves if
# any die crosses the trip point. Also auto-reverts after a wall-clock timeout.

set -u
TRIP_MEM_C=88          # crit 95; campaign hazard witness 90-93 under forced clocks
TIMEOUT_S=1800         # hard session cap: revert after 30 min no matter what
POLL_S=1
LOG=/tmp/dp_uv_guard.log
STOCK_REVERT='for d in /sys/class/drm/card*/device/pp_od_clk_voltage; do echo r > "$d" 2>/dev/null; echo c > "$d" 2>/dev/null; done'

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

revert_all() {
    log "REVERT: $1"
    eval "$STOCK_REVERT"
    log "REVERT done (stock curves restored on all dies)"
    exit 2
}
trap 'revert_all "guard shutdown"' INT TERM

log "guard UP: trip_mem=${TRIP_MEM_C}C timeout=${TIMEOUT_S}s poll=${POLL_S}s"
start=$(date +%s)
while :; do
    for d in /sys/class/drm/card*/device; do
        slot=$(grep PCI_SLOT_NAME "$d/uevent" 2>/dev/null | cut -d= -f2)
        [ -z "$slot" ] && continue
        # skip the NVIDIA display GPU (no pp_table)
        [ -f "$d/pp_table" ] || continue
        for h in "$d"/hwmon/hwmon*; do
            mem=""
            [ -f "$h/temp3_input" ] && mem=$(( $(cat "$h/temp3_input") / 1000 ))
            [ -n "$mem" ] && [ "$mem" -ge "$TRIP_MEM_C" ] && revert_all "$slot mem=${mem}C >= ${TRIP_MEM_C}C"
        done
    done
    [ $(( $(date +%s) - start )) -ge "$TIMEOUT_S" ] && revert_all "timeout"
    sleep "$POLL_S"
done
