#!/usr/bin/env bash
# sweep_voltage.sh - Chris-directed sclk undervolt sweep, one step per invocation.
#
# The sudo credential arrives on STDIN (first line) and is never stored here.
#
# Usage: echo '<pw>' | sweep_voltage.sh <mode> [voltage_mV]
#   mode = stock        : revert all dies to stock ('r'), verify readback
#   mode = pin1200      : sclk level 7 = 1200 MHz at stock-class voltage (1150 mV)
#   mode = step <mV>    : sclk level 7 = 1200 MHz at <mV> (undervolt step)
#   mode = verify       : read back all four die tables + current clocks/temps
#
# Die -> card map (PCI verified): die0=card1 die1=card3 die2=card0 die3=card4.
# mclk is NEVER written. Writes are sclk level 7 only (voltage per step).
set -u

read -r PW

declare -A CARDS=( [die0]=card1 [die1]=card3 [die2]=card0 [die3]=card4 )

apply() {
  local card="$1" sclk="$2" mv="$3"
  printf '%s\n' "$PW" | sudo -S sh -c "echo 's 7 $sclk $mv' > /sys/class/drm/$card/device/pp_od_clk_voltage" 2>/dev/null
}

reset_all() {
  for d in die0 die1 die2 die3; do
    printf '%s\n' "$PW" | sudo -S sh -c "echo 'r' > /sys/class/drm/${CARDS[$d]}/device/pp_od_clk_voltage" 2>/dev/null
  done
}

mode="${1:-verify}"
volt="${2:-}"

case "$mode" in
  stock)   reset_all; echo "[stock] reverted all dies";;
  pin1200) for d in die0 die1 die2 die3; do apply "${CARDS[$d]}" 1200 1150; done; echo "[pin1200] sclk7=1200@1150 applied";;
  step)    for d in die0 die1 die2 die3; do apply "${CARDS[$d]}" 1200 "$volt"; done; echo "[step] sclk7=1200@$volt applied";;
  verify)  ;;
  *) echo "unknown mode"; exit 1;;
esac

echo "--- readback $(date '+%F %T') ---"
for d in die0 die1 die2 die3; do
  echo "[$d $(printf '%s\n' "$PW" | sudo -S cat /sys/class/drm/${CARDS[$d]}/device/pp_od_clk_voltage 2>/dev/null | sed -n '9p')]"
done
echo "--- sclk now ---"
rocm-smi --showclocks 2>/dev/null | grep -E "GPU|sclk" | head -8
echo "--- temps ---"
rocm-smi --showtemp 2>/dev/null | grep -E "GPU|edge|junction|Timestamp" | head -12
